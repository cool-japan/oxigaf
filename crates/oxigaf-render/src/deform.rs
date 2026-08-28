//! GPU-accelerated Gaussian deformation pipeline.
//!
//! Binds 3D Gaussians to a FLAME mesh and deforms them to world space using
//! a compute shader. Each Gaussian is associated with a face on the mesh via
//! barycentric coordinates and a learnable local offset in the face's TBN
//! (Tangent-Bitangent-Normal) frame.
//!
//! # Pipeline overview
//!
//! 1. [`DeformPipeline::new`] — compile the `deform_gaussians.wgsl` shader and
//!    build the bind group layouts.
//! 2. [`DeformPipeline::upload_mesh`] — upload FLAME mesh geometry to GPU
//!    buffers, returning a [`MeshBuffers`] handle.
//! 3. [`DeformPipeline::deform`] — for a given [`GaussianModel`] and mesh,
//!    allocate per-Gaussian input/output buffers, dispatch the shader, and
//!    read back the deformed positions and rotations.
//!
//! [`deform_cpu`] is a pure-CPU implementation of the same algorithm (no
//! GPU device required), usable as a fallback and as a ground-truth oracle
//! for testing [`DeformPipeline::deform`].
//!
//! # Memory layout
//!
//! WGSL requires `vec3<f32>` array elements to be 16-byte aligned
//! (same stride as `vec4<f32>`).  All vec3 data is therefore padded to
//! `[f32; 4]` before uploading and the shader uses `array<vec4<f32>>`.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::{gaussian::GaussianModel, RenderError};

// ---------------------------------------------------------------------------
// Public data types
// ---------------------------------------------------------------------------

/// Number of storage buffers `deform_bgl_0` declares in the compute stage.
///
/// Six per-Gaussian inputs + three mesh inputs + two outputs. This exceeds
/// wgpu's *default* `max_storage_buffers_per_shader_stage` of 8, so a device
/// requested with `wgpu::Limits::default()` cannot host this pipeline — see
/// [`DeformPipeline::new`].
const DEFORM_STORAGE_BINDINGS: u32 = 11;

/// Monotonic source of [`MeshBuffers::id`] values.
///
/// A plain counter rather than the allocation-address trick
/// `rasterizer.rs`'s `UploadedModel` uses: `MeshBuffers` owns GPU buffers,
/// not a host `Vec`, so there is no stable host pointer to key on, and a
/// freed `wgpu::Buffer`'s slot could be reused by a later allocation.
static NEXT_MESH_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Allocate the next never-reused mesh identity.
fn next_mesh_id() -> u64 {
    NEXT_MESH_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

/// GPU buffers holding the FLAME mesh geometry needed for deformation.
pub struct MeshBuffers {
    /// Identity of this upload, unique for the process lifetime.
    ///
    /// [`DeformPipeline::deform`] binds the caller-supplied `MeshBuffers`
    /// directly into bind group 0, so any future capacity-keyed cache of
    /// per-Gaussian buffers and bind groups must *also* key on mesh
    /// identity — a cached bind group still references the previous mesh's
    /// buffers, and a same-size mesh is not the same mesh. This field is
    /// that key. It is assigned by [`DeformPipeline::upload_mesh`] and never
    /// reused, so two distinct uploads never compare equal even if one is
    /// dropped before the other is created.
    pub id: u64,
    /// Vertex positions (xyz + pad) — stride 16 bytes per vertex, \[V\] entries.
    pub vertices: wgpu::Buffer,
    /// Vertex normals  (xyz + pad) — stride 16 bytes per vertex, \[V\] entries.
    pub normals: wgpu::Buffer,
    /// Face vertex indices (v0, v1, v2, pad) — 16 bytes per face, \[F\] entries.
    pub faces: wgpu::Buffer,
    /// Number of vertices.
    pub num_vertices: u32,
    /// Number of faces.
    pub num_faces: u32,
}

// ---------------------------------------------------------------------------
// Internal GPU-layout types  (Pod + Zeroable so bytemuck::cast_slice works)
// ---------------------------------------------------------------------------

/// A vec3 padded to 16 bytes for WGSL storage-buffer alignment.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct Vec4f32 {
    x: f32,
    y: f32,
    z: f32,
    w: f32,
}

impl Vec4f32 {
    fn from_xyz(xyz: [f32; 3]) -> Self {
        Self {
            x: xyz[0],
            y: xyz[1],
            z: xyz[2],
            w: 0.0,
        }
    }

    fn from_xyzw(xyzw: [f32; 4]) -> Self {
        Self {
            x: xyzw[0],
            y: xyzw[1],
            z: xyzw[2],
            w: xyzw[3],
        }
    }
}

/// A vec3 of u32, padded to 16 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct Vec4u32 {
    x: u32,
    y: u32,
    z: u32,
    w: u32,
}

/// Uniform block for bind group 1.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct DeformUniforms {
    num_gaussians: u32,
    num_faces: u32,
    _pad0: u32,
    _pad1: u32,
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

/// GPU compute pipeline for deforming Gaussians bound to a FLAME mesh.
pub struct DeformPipeline {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipeline: wgpu::ComputePipeline,
    bgl_0: wgpu::BindGroupLayout,
    bgl_1: wgpu::BindGroupLayout,
}

/// Output type for [`DeformPipeline::deform`]: `(deformed_positions, deformed_rotations)`.
///
/// Positions are `[x, y, z, 1.0]`; rotations are `[x, y, z, w]` quaternions.
type DeformResult = (Vec<[f32; 4]>, Vec<[f32; 4]>);

/// Validate that `model`'s per-Gaussian FLAME-binding arrays
/// (`face_indices`, `barycentric`, `local_offsets`, `is_rigid`) all have
/// exactly `model.len()` entries, matching `model.gaussians`.
///
/// These fields are all `pub` and independently mutable (or a
/// `GaussianModel` may be built by hand), so nothing else guarantees they
/// stay in sync; a short array would otherwise hand `create_buffer_init` a
/// zero-sized (rejected by wgpu) or shorter-than-expected buffer, silently
/// producing wrong deformations for the tail Gaussians once the shader
/// clamps out-of-range reads. Factored out of [`DeformPipeline::deform`]
/// so it is unit-testable without a GPU device.
fn validate_deform_model_lengths(model: &GaussianModel) -> Result<(), RenderError> {
    let n = model.len();
    for (field, len) in [
        ("face_indices", model.face_indices.len()),
        ("barycentric", model.barycentric.len()),
        ("local_offsets", model.local_offsets.len()),
        ("is_rigid", model.is_rigid.len()),
    ] {
        if len != n {
            return Err(RenderError::Rasterize(format!(
                "deform: GaussianModel.{field} has {len} entries, expected {n} (== model.len())"
            )));
        }
    }
    Ok(())
}

impl DeformPipeline {
    /// Compile the deform shader and create the pipeline.
    ///
    /// # Device requirements
    ///
    /// `device` must have been requested with
    /// `max_storage_buffers_per_shader_stage >= DEFORM_STORAGE_BINDINGS`
    /// (11); wgpu's default is 8. A device that does not meet this is
    /// rejected here with [`RenderError::GpuInit`] — without the check,
    /// `create_bind_group_layout` raises an *uncaptured* wgpu validation
    /// error, which aborts the process through the global error handler
    /// instead of returning through this `Result`.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::GpuInit`] if `device`'s storage-buffer limit
    /// is too low for this pipeline's bind group layout.
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Result<Self, RenderError> {
        // ---- Device capability pre-check ----
        // Must run before create_bind_group_layout: a wgpu validation error
        // there cannot be caught, so this is the only place the caller can
        // be handed a real Result.
        let available = device.limits().max_storage_buffers_per_shader_stage;
        if available < DEFORM_STORAGE_BINDINGS {
            return Err(RenderError::GpuInit(format!(
                "DeformPipeline needs max_storage_buffers_per_shader_stage >= \
                 {DEFORM_STORAGE_BINDINGS}, but this device was created with {available}. \
                 Request the device with `wgpu::Limits {{ \
                 max_storage_buffers_per_shader_stage: 16, ..Default::default() }}`."
            )));
        }

        // ---- Shader module ----
        let shader_src = include_str!("../shaders/deform_gaussians.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("deform_gaussians"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        // ---- Bind group 0: per-Gaussian and mesh buffers ----
        let bgl_0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("deform_bgl_0"),
            entries: &[
                // Gaussian inputs
                storage_ro_entry(0), // gaussian_positions
                storage_ro_entry(1), // gaussian_rotations
                storage_ro_entry(2), // gaussian_face_indices
                storage_ro_entry(3), // gaussian_barycentric
                storage_ro_entry(4), // gaussian_local_offsets
                storage_ro_entry(5), // gaussian_is_rigid
                // Mesh data
                storage_ro_entry(6), // mesh_vertices
                storage_ro_entry(7), // mesh_normals
                storage_ro_entry(8), // mesh_faces
                // Outputs
                storage_rw_entry(9),  // out_positions
                storage_rw_entry(10), // out_rotations
            ],
        });

        // ---- Bind group 1: uniforms ----
        // The shader has a single `DeformUniforms` struct at binding 0 that
        // contains both `num_gaussians` and `num_faces`.  One layout entry and
        // one buffer is sufficient.
        let bgl_1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("deform_bgl_1"),
            entries: &[
                uniform_entry(0), // DeformUniforms { num_gaussians, num_faces, ... }
            ],
        });

        // ---- Pipeline layout (two bind groups) ----
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("deform_pipeline_layout"),
            bind_group_layouts: &[Some(&bgl_0), Some(&bgl_1)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("deform_gaussians_pipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("deform_gaussians"),
            compilation_options: Default::default(),
            cache: None,
        });

        Ok(Self {
            device,
            queue,
            pipeline,
            bgl_0,
            bgl_1,
        })
    }

    /// Upload FLAME mesh geometry to GPU buffers.
    ///
    /// All `vec3` data is padded to `vec4` (16 bytes) for WGSL alignment.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::Rasterize`] if `vertices` or `faces` is
    /// empty (`create_buffer_init` is handed a zero-sized buffer contents
    /// slice otherwise, which wgpu rejects) or if `normals.len() !=
    /// vertices.len()` (the shader indexes both arrays by the same vertex
    /// index, so a mismatch would silently read normals out of sync with
    /// their vertices).
    pub fn upload_mesh(
        &self,
        vertices: &[[f32; 3]],
        normals: &[[f32; 3]],
        faces: &[[u32; 3]],
    ) -> Result<MeshBuffers, RenderError> {
        if vertices.is_empty() {
            return Err(RenderError::Rasterize(
                "upload_mesh: vertices must not be empty".to_string(),
            ));
        }
        if faces.is_empty() {
            return Err(RenderError::Rasterize(
                "upload_mesh: faces must not be empty".to_string(),
            ));
        }
        if normals.len() != vertices.len() {
            return Err(RenderError::Rasterize(format!(
                "upload_mesh: normals has {} entries, expected {} (== vertices.len())",
                normals.len(),
                vertices.len()
            )));
        }

        let vert_data: Vec<Vec4f32> = vertices.iter().map(|v| Vec4f32::from_xyz(*v)).collect();
        let norm_data: Vec<Vec4f32> = normals.iter().map(|n| Vec4f32::from_xyz(*n)).collect();
        let face_data: Vec<Vec4u32> = faces
            .iter()
            .map(|f| Vec4u32 {
                x: f[0],
                y: f[1],
                z: f[2],
                w: 0,
            })
            .collect();

        let vert_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("mesh_vertices"),
                contents: bytemuck::cast_slice(&vert_data),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let norm_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("mesh_normals"),
                contents: bytemuck::cast_slice(&norm_data),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let face_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("mesh_faces"),
                contents: bytemuck::cast_slice(&face_data),
                usage: wgpu::BufferUsages::STORAGE,
            });

        Ok(MeshBuffers {
            id: next_mesh_id(),
            vertices: vert_buf,
            normals: norm_buf,
            faces: face_buf,
            num_vertices: vertices.len() as u32,
            num_faces: faces.len() as u32,
        })
    }

    /// Deform Gaussians bound to the given mesh.
    ///
    /// Returns `(deformed_positions, deformed_rotations)` where positions are
    /// `[x, y, z, 1.0]` and rotations are `[x, y, z, w]` quaternions.
    ///
    /// A Gaussian flagged rigid in `model.is_rigid` is *not* bound to the
    /// mesh: it is returned with its own authored position/rotation, with no
    /// barycentric interpolation, no TBN local offset and no TBN rotation
    /// composition applied. The compute kernel enforces this itself (see the
    /// `gaussian_is_rigid` early-out in `shaders/deform_gaussians.wgsl`), so
    /// the readback is returned unmodified. [`deform_cpu`] applies the same
    /// rule in the same order and is an exact oracle for this method.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::Rasterize`] if `model.face_indices`,
    /// `model.barycentric`, `model.local_offsets`, or `model.is_rigid` does
    /// not have exactly `model.len()` entries. These fields are all `pub`
    /// and independently mutable (or a `GaussianModel` may be built by
    /// hand), so nothing else guarantees they stay in sync with
    /// `model.gaussians`; a short array would otherwise hand
    /// `create_buffer_init` a zero-sized (rejected by wgpu) or
    /// shorter-than-expected buffer, silently producing wrong deformations
    /// for the tail Gaussians once the shader clamps out-of-range reads.
    pub fn deform(
        &self,
        model: &GaussianModel,
        mesh: &MeshBuffers,
    ) -> Result<DeformResult, RenderError> {
        let n = model.len() as u32;
        if n == 0 {
            return Ok((Vec::new(), Vec::new()));
        }
        validate_deform_model_lengths(model)?;

        // ---- Build per-Gaussian input buffers ----
        let pos_data: Vec<Vec4f32> = model
            .gaussians
            .iter()
            .map(|g| Vec4f32::from_xyz(g.position))
            .collect();
        let rot_data: Vec<Vec4f32> = model
            .gaussians
            .iter()
            .map(|g| Vec4f32::from_xyzw(g.rotation))
            .collect();

        let fi_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("gaussian_face_indices"),
                contents: bytemuck::cast_slice(&model.face_indices),
                usage: wgpu::BufferUsages::STORAGE,
            });

        // Pad barycentric and local_offsets to vec4 for WGSL alignment.
        let bary_data: Vec<Vec4f32> = model
            .barycentric
            .iter()
            .map(|b| Vec4f32::from_xyz(*b))
            .collect();
        let offset_data: Vec<Vec4f32> = model
            .local_offsets
            .iter()
            .map(|o| Vec4f32::from_xyz(*o))
            .collect();

        let rigid_data: Vec<u32> = model
            .is_rigid
            .iter()
            .map(|&r| if r { 1u32 } else { 0u32 })
            .collect();

        let pos_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("g_positions"),
                contents: bytemuck::cast_slice(&pos_data),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let rot_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("g_rotations"),
                contents: bytemuck::cast_slice(&rot_data),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let bary_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("g_barycentric"),
                contents: bytemuck::cast_slice(&bary_data),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let offset_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("g_local_offsets"),
                contents: bytemuck::cast_slice(&offset_data),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let rigid_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("g_is_rigid"),
                contents: bytemuck::cast_slice(&rigid_data),
                usage: wgpu::BufferUsages::STORAGE,
            });

        // ---- Output buffers ----
        let out_stride = std::mem::size_of::<Vec4f32>() as u64;
        let out_byte_size = n as u64 * out_stride;

        let out_pos_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("out_positions"),
            size: out_byte_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_rot_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("out_rotations"),
            size: out_byte_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // ---- Uniform buffer for bind group 1 ----
        // A single DeformUniforms struct packs both num_gaussians and num_faces
        // into one 16-byte uniform buffer, matching the WGSL declaration:
        //   @group(1) @binding(0) var<uniform> uniforms: DeformUniforms
        let uniforms_data = DeformUniforms {
            num_gaussians: n,
            num_faces: mesh.num_faces,
            _pad0: 0,
            _pad1: 0,
        };
        let uniform_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("deform_uniforms"),
                contents: bytemuck::bytes_of(&uniforms_data),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // ---- Bind groups ----
        let bg_0 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("deform_bg_0"),
            layout: &self.bgl_0,
            entries: &[
                buf_entry(0, &pos_buf),
                buf_entry(1, &rot_buf),
                buf_entry(2, &fi_buf),
                buf_entry(3, &bary_buf),
                buf_entry(4, &offset_buf),
                buf_entry(5, &rigid_buf),
                buf_entry(6, &mesh.vertices),
                buf_entry(7, &mesh.normals),
                buf_entry(8, &mesh.faces),
                buf_entry(9, &out_pos_buf),
                buf_entry(10, &out_rot_buf),
            ],
        });
        let bg_1 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("deform_bg_1"),
            layout: &self.bgl_1,
            entries: &[buf_entry(0, &uniform_buf)],
        });

        // ---- Dispatch ----
        let workgroup_size = 256u32;
        let num_workgroups = n.div_ceil(workgroup_size);

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("deform_encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("deform_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg_0, &[]);
            pass.set_bind_group(1, &bg_1, &[]);
            pass.dispatch_workgroups(num_workgroups, 1, 1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));

        // ---- Read back results ----
        // Both outputs are pulled through a single staging buffer with one
        // `submit`/`poll` (see `readback_vec4_pair`), instead of two
        // independent GPU round-trip stalls.
        //
        // The readback is returned as-is: rigid Gaussians are handled by the
        // kernel itself (`gaussian_is_rigid`, bind group 0 slot 5, drives an
        // early-out that writes the authored position/rotation straight
        // through), so there is no post-readback fix-up pass over the CPU
        // copy of the results.
        self.readback_vec4_pair(&out_pos_buf, &out_rot_buf, n as usize)
    }

    // ---- Internal helpers ----

    /// Read back `count` `vec4<f32>` entries from each of two GPU storage
    /// buffers, sharing a single staging buffer and one `submit`/`poll`
    /// round-trip between them (rather than reading each buffer back
    /// independently, which costs two full GPU stalls).
    ///
    /// The returned pair is `(buffer_a's contents, buffer_b's contents)` —
    /// the same shape as [`DeformResult`], which is what
    /// [`DeformPipeline::deform`] forwards it as.
    fn readback_vec4_pair(
        &self,
        buffer_a: &wgpu::Buffer,
        buffer_b: &wgpu::Buffer,
        count: usize,
    ) -> Result<DeformResult, RenderError> {
        let half_byte_size = (count * std::mem::size_of::<[f32; 4]>()) as u64;
        let total_byte_size = half_byte_size * 2;

        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("deform_staging_pair"),
            size: total_byte_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("deform_readback_pair"),
            });
        encoder.copy_buffer_to_buffer(buffer_a, 0, &staging, 0, half_byte_size);
        encoder.copy_buffer_to_buffer(buffer_b, 0, &staging, half_byte_size, half_byte_size);
        self.queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).ok();
        });
        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        rx.recv()
            .map_err(|e| RenderError::Rasterize(format!("Deform readback channel error: {e}")))?
            .map_err(|e| RenderError::Rasterize(format!("Deform buffer map failed: {e}")))?;

        let mapped = slice
            .get_mapped_range()
            .map_err(|e| RenderError::Rasterize(format!("Deform mapped range failed: {e}")))?;
        let floats: &[f32] = bytemuck::cast_slice(&mapped);
        let floats_per_half = count * 4;
        let a: Vec<[f32; 4]> = floats[..floats_per_half]
            .chunks_exact(4)
            .map(|c| [c[0], c[1], c[2], c[3]])
            .collect();
        let b: Vec<[f32; 4]> = floats[floats_per_half..floats_per_half * 2]
            .chunks_exact(4)
            .map(|c| [c[0], c[1], c[2], c[3]])
            .collect();
        drop(mapped);
        staging.unmap();

        Ok((a, b))
    }
}

// ---------------------------------------------------------------------------
// Bind group layout entry helpers (local to this module)
// ---------------------------------------------------------------------------

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn storage_ro_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn storage_rw_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn buf_entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

// ---------------------------------------------------------------------------
// CPU-side deformation (mirrors shaders/deform_gaussians.wgsl exactly)
// ---------------------------------------------------------------------------

/// Pure-CPU reference implementation of the FLAME→Gaussian deformation
/// pipeline: a fallback when no GPU is available, and a ground-truth oracle
/// for testing [`DeformPipeline::deform`]. Mirrors
/// `shaders/deform_gaussians.wgsl`'s `deform_gaussians` kernel exactly,
/// including its out-of-range-face-index pass-through and (via
/// [`build_tbn`]) its degenerate-triangle/edge fallbacks.
///
/// `faces[f] = [v0, v1, v2]` indexes into `vertices`/`normals`. A rigid
/// Gaussian (`model.is_rigid[i]`) is unaffected by the mesh binding and
/// keeps its own authored position/rotation — the same rule the kernel
/// applies via its own `gaussian_is_rigid` early-out, checked in the same
/// order (rigid before face-index validity), so the two paths agree even
/// for a rigid Gaussian carrying an out-of-range face index.
///
/// # Errors
///
/// Returns [`RenderError::Rasterize`] if `model`'s per-Gaussian
/// FLAME-binding arrays don't match `model.len()` (see this module's
/// private `validate_deform_model_lengths` helper), if
/// `normals.len() != vertices.len()`, or if a face referenced by
/// `model.face_indices` has a vertex index `>= vertices.len()`.
pub fn deform_cpu(
    model: &GaussianModel,
    vertices: &[[f32; 3]],
    normals: &[[f32; 3]],
    faces: &[[u32; 3]],
) -> Result<DeformResult, RenderError> {
    let n = model.len();
    if n == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    validate_deform_model_lengths(model)?;
    if normals.len() != vertices.len() {
        return Err(RenderError::Rasterize(format!(
            "deform_cpu: normals has {} entries, expected {} (== vertices.len())",
            normals.len(),
            vertices.len()
        )));
    }

    let num_faces = faces.len() as u32;
    let mut out_positions = Vec::with_capacity(n);
    let mut out_rotations = Vec::with_capacity(n);

    for i in 0..n {
        let g = &model.gaussians[i];

        if model.is_rigid[i] {
            out_positions.push([g.position[0], g.position[1], g.position[2], 1.0]);
            out_rotations.push(g.rotation);
            continue;
        }

        let fi = model.face_indices[i];
        if fi >= num_faces {
            // Invalid face index: pass through unchanged, matching the
            // shader's `if fi >= uniforms.num_faces { pass through }`.
            out_positions.push([g.position[0], g.position[1], g.position[2], 1.0]);
            out_rotations.push(g.rotation);
            continue;
        }

        let face = faces[fi as usize];
        let (vi0, vi1, vi2) = (face[0] as usize, face[1] as usize, face[2] as usize);
        if vi0 >= vertices.len() || vi1 >= vertices.len() || vi2 >= vertices.len() {
            return Err(RenderError::Rasterize(format!(
                "deform_cpu: face {fi} references an out-of-range vertex index (have {} vertices)",
                vertices.len()
            )));
        }

        let bary = model.barycentric[i];
        let local = model.local_offsets[i];

        let p0 = vertices[vi0];
        let p1 = vertices[vi1];
        let p2 = vertices[vi2];
        let interp_pos = [
            bary[0] * p0[0] + bary[1] * p1[0] + bary[2] * p2[0],
            bary[0] * p0[1] + bary[1] * p1[1] + bary[2] * p2[1],
            bary[0] * p0[2] + bary[1] * p1[2] + bary[2] * p2[2],
        ];

        let n0 = normals[vi0];
        let n1 = normals[vi1];
        let n2 = normals[vi2];
        let interp_normal = [
            bary[0] * n0[0] + bary[1] * n1[0] + bary[2] * n2[0],
            bary[0] * n0[1] + bary[1] * n1[1] + bary[2] * n2[1],
            bary[0] * n0[2] + bary[1] * n1[2] + bary[2] * n2[2],
        ];

        let (t, bt, tbn_n) = build_tbn(p0, p1, p2, interp_normal);
        let world_pos = apply_local_offset(interp_pos, t, bt, tbn_n, local);
        let q_tbn = mat3_cols_to_quat(t, bt, tbn_n);
        let world_rot = quat_mul(q_tbn, g.rotation);

        out_positions.push([world_pos[0], world_pos[1], world_pos[2], 1.0]);
        out_rotations.push(world_rot);
    }

    Ok((out_positions, out_rotations))
}

/// Multiply two quaternions (x, y, z, w) — Hamilton product.
pub fn quat_mul(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    let [ax, ay, az, aw] = a;
    let [bx, by, bz, bw] = b;
    [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]
}

/// Convert an orthonormal 3×3 matrix (columns t, bt, n) to a quaternion
/// `(x, y, z, w)` using Shepperd's method.
pub fn mat3_cols_to_quat(t: [f32; 3], bt: [f32; 3], n: [f32; 3]) -> [f32; 4] {
    let m00 = t[0];
    let m01 = bt[0];
    let m02 = n[0];
    let m10 = t[1];
    let m11 = bt[1];
    let m12 = n[1];
    let m20 = t[2];
    let m21 = bt[2];
    let m22 = n[2];

    let trace = m00 + m11 + m22;

    let (qx, qy, qz, qw);
    if trace > 0.0 {
        let s = (trace + 1.0_f32).sqrt() * 2.0; // s = 4w
        qw = 0.25 * s;
        qx = (m21 - m12) / s;
        qy = (m02 - m20) / s;
        qz = (m10 - m01) / s;
    } else if (m00 > m11) && (m00 > m22) {
        let s = (1.0 + m00 - m11 - m22).sqrt() * 2.0; // s = 4x
        qw = (m21 - m12) / s;
        qx = 0.25 * s;
        qy = (m01 + m10) / s;
        qz = (m02 + m20) / s;
    } else if m11 > m22 {
        let s = (1.0 + m11 - m00 - m22).sqrt() * 2.0; // s = 4y
        qw = (m02 - m20) / s;
        qx = (m01 + m10) / s;
        qy = 0.25 * s;
        qz = (m12 + m21) / s;
    } else {
        let s = (1.0 + m22 - m00 - m11).sqrt() * 2.0; // s = 4z
        qw = (m10 - m01) / s;
        qx = (m02 + m20) / s;
        qy = (m12 + m21) / s;
        qz = 0.25 * s;
    }

    // Normalise.
    let norm = (qx * qx + qy * qy + qz * qz + qw * qw).sqrt();
    if norm < 1e-10 {
        [0.0, 0.0, 0.0, 1.0]
    } else {
        [qx / norm, qy / norm, qz / norm, qw / norm]
    }
}

/// Build the TBN frame for a triangle with vertices `v0, v1, v2` and
/// interpolated vertex normal `interp_n`.
///
/// Returns `(tangent, bitangent, normal)` as unit vectors.
pub fn build_tbn(
    v0: [f32; 3],
    v1: [f32; 3],
    v2: [f32; 3],
    interp_n: [f32; 3],
) -> ([f32; 3], [f32; 3], [f32; 3]) {
    let e1 = vec3_sub(v1, v0);
    let e2 = vec3_sub(v2, v0);

    let geom_normal = vec3_cross(e1, e2);
    let geom_len = vec3_len(geom_normal);

    if geom_len < 1e-7 {
        // Degenerate triangle.
        return ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]);
    }

    // Choose the surface normal.
    let interp_len = vec3_len(interp_n);
    let n = if interp_len > 1e-7 {
        vec3_scale(interp_n, 1.0 / interp_len)
    } else {
        vec3_scale(geom_normal, 1.0 / geom_len)
    };

    // Tangent from e1, Gram-Schmidt.
    let e1_len = vec3_len(e1);
    let raw_t = if e1_len > 1e-7 {
        vec3_scale(e1, 1.0 / e1_len)
    } else {
        pick_perpendicular(n)
    };

    let t_proj = vec3_sub(raw_t, vec3_scale(n, vec3_dot(raw_t, n)));
    let t_proj_len = vec3_len(t_proj);
    let t = if t_proj_len > 1e-7 {
        vec3_scale(t_proj, 1.0 / t_proj_len)
    } else {
        let fallback = pick_perpendicular(n);
        let fp = vec3_sub(fallback, vec3_scale(n, vec3_dot(fallback, n)));
        let fp_len = vec3_len(fp);
        vec3_scale(fp, 1.0 / fp_len.max(1e-10))
    };

    // Bitangent = n × t.
    let bt_raw = vec3_cross(n, t);
    let bt = vec3_normalize(bt_raw);

    // Re-orthogonalise t = bt × n.
    let t_final = vec3_cross(bt, n);

    (t_final, bt, n)
}

/// Apply a local TBN offset to a barycentric-interpolated position.
pub fn apply_local_offset(
    bary_pos: [f32; 3],
    t: [f32; 3],
    bt: [f32; 3],
    n: [f32; 3],
    local: [f32; 3],
) -> [f32; 3] {
    [
        bary_pos[0] + t[0] * local[0] + bt[0] * local[1] + n[0] * local[2],
        bary_pos[1] + t[1] * local[0] + bt[1] * local[1] + n[1] * local[2],
        bary_pos[2] + t[2] * local[0] + bt[2] * local[1] + n[2] * local[2],
    ]
}

// ---------------------------------------------------------------------------
// Minimal vec3 helpers (no external crate dependency in this module)
// ---------------------------------------------------------------------------

fn vec3_sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn vec3_scale(a: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

fn vec3_dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn vec3_cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn vec3_len(a: [f32; 3]) -> f32 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}

fn vec3_normalize(a: [f32; 3]) -> [f32; 3] {
    let len = vec3_len(a);
    if len < 1e-10 {
        a
    } else {
        vec3_scale(a, 1.0 / len)
    }
}

/// Pick a unit vector perpendicular to `n` (used for degenerate-edge fallback).
fn pick_perpendicular(n: [f32; 3]) -> [f32; 3] {
    let abs_n = [n[0].abs(), n[1].abs(), n[2].abs()];
    if abs_n[0] <= abs_n[1] && abs_n[0] <= abs_n[2] {
        [1.0, 0.0, 0.0]
    } else if abs_n[1] <= abs_n[2] {
        [0.0, 1.0, 0.0]
    } else {
        [0.0, 0.0, 1.0]
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    fn assert_approx_eq4(a: [f32; 4], b: [f32; 4], eps: f32, msg: &str) {
        for i in 0..4 {
            assert!(
                (a[i] - b[i]).abs() < eps,
                "{msg}: component {i}: {a:?} vs {b:?}"
            );
        }
    }

    fn assert_approx_eq3(a: [f32; 3], b: [f32; 3], eps: f32, msg: &str) {
        for i in 0..3 {
            assert!(
                (a[i] - b[i]).abs() < eps,
                "{msg}: component {i}: {a:?} vs {b:?}"
            );
        }
    }

    // Quaternion sign normalisation: choose the canonical representative
    // with qw >= 0 (or, if qw == 0, leading positive non-zero component).
    fn quat_canonical(q: [f32; 4]) -> [f32; 4] {
        let [x, y, z, w] = q;
        if w < 0.0 || (w == 0.0 && x < 0.0) || (w == 0.0 && x == 0.0 && y < 0.0) {
            [-x, -y, -z, -w]
        } else {
            [x, y, z, w]
        }
    }

    // -----------------------------------------------------------------------
    // Test 1: identity matrix → identity quaternion
    // -----------------------------------------------------------------------
    #[test]
    fn test_mat3_to_quat_identity() {
        let t = [1.0_f32, 0.0, 0.0];
        let bt = [0.0_f32, 1.0, 0.0];
        let n = [0.0_f32, 0.0, 1.0];

        let q = quat_canonical(mat3_cols_to_quat(t, bt, n));
        let expected = [0.0_f32, 0.0, 0.0, 1.0]; // identity quaternion
        assert_approx_eq4(q, expected, EPS, "identity mat → quat");
    }

    // -----------------------------------------------------------------------
    // Test 2: 90° rotation around Z (col-major: t=Y, bt=-X, n=Z)
    // -----------------------------------------------------------------------
    #[test]
    fn test_mat3_to_quat_rot90_z() {
        // Rotation by +90° around Z:  R_z(90) = [[0,−1,0],[1,0,0],[0,0,1]]
        // Columns: t=(0,1,0), bt=(-1,0,0), n=(0,0,1)
        let t = [0.0_f32, 1.0, 0.0];
        let bt = [-1.0_f32, 0.0, 0.0];
        let n = [0.0_f32, 0.0, 1.0];

        let q = quat_canonical(mat3_cols_to_quat(t, bt, n));
        // q_z(90°) = (0, 0, sin45°, cos45°) = (0,0,√½,√½)
        let s = std::f32::consts::FRAC_1_SQRT_2;
        let expected = quat_canonical([0.0, 0.0, s, s]);
        assert_approx_eq4(q, expected, EPS, "90° Z rotation");
    }

    // -----------------------------------------------------------------------
    // Test 3: 90° rotation around X
    // -----------------------------------------------------------------------
    #[test]
    fn test_mat3_to_quat_rot90_x() {
        // R_x(90) columns: t=(1,0,0), bt=(0,0,1), n=(0,−1,0)
        let t = [1.0_f32, 0.0, 0.0];
        let bt = [0.0_f32, 0.0, 1.0];
        let n = [0.0_f32, -1.0, 0.0];

        let q = quat_canonical(mat3_cols_to_quat(t, bt, n));
        // q_x(90°) = (sin45°, 0, 0, cos45°)
        let s = std::f32::consts::FRAC_1_SQRT_2;
        let expected = quat_canonical([s, 0.0, 0.0, s]);
        assert_approx_eq4(q, expected, EPS, "90° X rotation");
    }

    // -----------------------------------------------------------------------
    // Test 4: quaternion multiply — identity composition
    // -----------------------------------------------------------------------
    #[test]
    fn test_quat_mul_identity() {
        let id = [0.0_f32, 0.0, 0.0, 1.0];
        let q = [0.1_f32, 0.2, 0.3, 0.9272727_f32]; // arbitrary unit quat
        let norm = (0.1f32 * 0.1 + 0.2 * 0.2 + 0.3 * 0.3 + 0.9272727 * 0.9272727).sqrt();
        let q_unit = [q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm];

        // q * id = q
        let result = quat_mul(q_unit, id);
        assert_approx_eq4(result, q_unit, EPS, "q * id = q");

        // id * q = q
        let result2 = quat_mul(id, q_unit);
        assert_approx_eq4(result2, q_unit, EPS, "id * q = q");
    }

    // -----------------------------------------------------------------------
    // Test 5: quaternion multiply — composition of 90° Z rotations
    // -----------------------------------------------------------------------
    #[test]
    fn test_quat_mul_compose_z() {
        // Two 90° Z rotations should give 180° Z rotation.
        let s = std::f32::consts::FRAC_1_SQRT_2;
        let q90z = [0.0_f32, 0.0, s, s]; // 90° around Z
        let q180z = quat_canonical(quat_mul(q90z, q90z));
        // 180° around Z: (0, 0, 1, 0)
        let expected = quat_canonical([0.0, 0.0, 1.0, 0.0]);
        assert_approx_eq4(q180z, expected, EPS, "90° + 90° Z = 180° Z");
    }

    // -----------------------------------------------------------------------
    // Test 6: TBN construction — flat triangle in XY plane
    // -----------------------------------------------------------------------
    #[test]
    fn test_build_tbn_xy_plane() {
        let v0 = [0.0_f32, 0.0, 0.0];
        let v1 = [1.0_f32, 0.0, 0.0];
        let v2 = [0.0_f32, 1.0, 0.0];
        // Normal pointing +Z.
        let interp_n = [0.0_f32, 0.0, 1.0];

        let (t, bt, n) = build_tbn(v0, v1, v2, interp_n);

        // Normal should be +Z.
        assert_approx_eq3(n, [0.0, 0.0, 1.0], EPS, "XY-plane normal");

        // TBN should be orthonormal.
        let dot_t_n = vec3_dot(t, n);
        let dot_bt_n = vec3_dot(bt, n);
        let dot_t_bt = vec3_dot(t, bt);
        assert!(dot_t_n.abs() < EPS, "t ⊥ n: {dot_t_n}");
        assert!(dot_bt_n.abs() < EPS, "bt ⊥ n: {dot_bt_n}");
        assert!(dot_t_bt.abs() < EPS, "t ⊥ bt: {dot_t_bt}");
        assert!((vec3_len(t) - 1.0).abs() < EPS, "|t| = 1");
        assert!((vec3_len(bt) - 1.0).abs() < EPS, "|bt| = 1");
        assert!((vec3_len(n) - 1.0).abs() < EPS, "|n| = 1");
    }

    // -----------------------------------------------------------------------
    // Test 7: apply_local_offset — no offset is identity
    // -----------------------------------------------------------------------
    #[test]
    fn test_apply_local_offset_zero() {
        let base = [1.0_f32, 2.0, 3.0];
        let t = [1.0_f32, 0.0, 0.0];
        let bt = [0.0_f32, 1.0, 0.0];
        let n = [0.0_f32, 0.0, 1.0];
        let local = [0.0_f32, 0.0, 0.0];

        let result = apply_local_offset(base, t, bt, n, local);
        assert_approx_eq3(result, base, EPS, "zero offset = identity");
    }

    // -----------------------------------------------------------------------
    // Test 8: apply_local_offset — normal offset moves along n
    // -----------------------------------------------------------------------
    #[test]
    fn test_apply_local_offset_normal_direction() {
        let base = [0.0_f32, 0.0, 0.0];
        let t = [1.0_f32, 0.0, 0.0];
        let bt = [0.0_f32, 1.0, 0.0];
        let n = [0.0_f32, 0.0, 1.0];
        // Offset 0.5 along n (which is +Z).
        let local = [0.0_f32, 0.0, 0.5];

        let result = apply_local_offset(base, t, bt, n, local);
        assert_approx_eq3(result, [0.0, 0.0, 0.5], EPS, "normal offset → +Z");
    }

    // -----------------------------------------------------------------------
    // Test 9: degenerate triangle falls back to identity axes
    // -----------------------------------------------------------------------
    #[test]
    fn test_build_tbn_degenerate_triangle() {
        // All three vertices at the same point → degenerate.
        let v = [1.0_f32, 2.0, 3.0];
        let (t, bt, n) = build_tbn(v, v, v, [0.0, 0.0, 0.0]);
        // Should return the canonical identity frame.
        assert_approx_eq3(t, [1.0, 0.0, 0.0], EPS, "degenerate t");
        assert_approx_eq3(bt, [0.0, 1.0, 0.0], EPS, "degenerate bt");
        assert_approx_eq3(n, [0.0, 0.0, 1.0], EPS, "degenerate n");
    }

    // -----------------------------------------------------------------------
    // Test 10: TBN + offset round-trip (barycentric interpolation)
    // -----------------------------------------------------------------------
    #[test]
    fn test_barycentric_interpolation_centroid() {
        let v0 = [0.0_f32, 0.0, 0.0];
        let v1 = [3.0_f32, 0.0, 0.0];
        let v2 = [0.0_f32, 3.0, 0.0];
        let bary = [1.0_f32 / 3.0, 1.0 / 3.0, 1.0 / 3.0];

        let interp = [
            bary[0] * v0[0] + bary[1] * v1[0] + bary[2] * v2[0],
            bary[0] * v0[1] + bary[1] * v1[1] + bary[2] * v2[1],
            bary[0] * v0[2] + bary[1] * v1[2] + bary[2] * v2[2],
        ];
        // Centroid of equilateral-ish triangle with v0=(0,0), v1=(3,0), v2=(0,3)
        assert_approx_eq3(interp, [1.0, 1.0, 0.0], EPS, "centroid interpolation");

        // Apply TBN offset at centroid.
        let interp_n = [0.0_f32, 0.0, 1.0];
        let (t, bt, n) = build_tbn(v0, v1, v2, interp_n);
        // Offset 0.5 in tangent direction.
        let result = apply_local_offset(interp, t, bt, n, [0.5, 0.0, 0.0]);
        // Position should have moved 0.5 along t (which is +X for this triangle).
        let expected_x = interp[0] + 0.5 * t[0];
        let expected_y = interp[1] + 0.5 * t[1];
        let expected_z = interp[2] + 0.5 * t[2];
        assert_approx_eq3(
            result,
            [expected_x, expected_y, expected_z],
            EPS,
            "offset round-trip",
        );
    }

    // -----------------------------------------------------------------------
    // Test 11: quat_mul is not commutative (sanity check)
    // -----------------------------------------------------------------------
    #[test]
    fn test_quat_mul_non_commutative() {
        let s = std::f32::consts::FRAC_1_SQRT_2;
        let qx = [s, 0.0, 0.0, s]; // 90° around X
        let qz = [0.0, 0.0, s, s]; // 90° around Z

        let qxz = quat_mul(qx, qz);
        let qzx = quat_mul(qz, qx);

        // The results should differ (quaternion mul is non-commutative).
        let are_equal = qxz.iter().zip(qzx.iter()).all(|(a, b)| (a - b).abs() < EPS);
        assert!(
            !are_equal,
            "quat_mul should be non-commutative for different axes"
        );
    }

    // -----------------------------------------------------------------------
    // Test 12: 90° Y rotation matrix → quaternion
    // -----------------------------------------------------------------------
    #[test]
    fn test_mat3_to_quat_rot90_y() {
        // R_y(90°) columns: t=(0,0,-1), bt=(0,1,0), n=(1,0,0)
        let t = [0.0_f32, 0.0, -1.0];
        let bt = [0.0_f32, 1.0, 0.0];
        let n = [1.0_f32, 0.0, 0.0];

        let q = quat_canonical(mat3_cols_to_quat(t, bt, n));
        // q_y(90°) = (0, sin45°, 0, cos45°)
        let s = std::f32::consts::FRAC_1_SQRT_2;
        let expected = quat_canonical([0.0, s, 0.0, s]);
        assert_approx_eq4(q, expected, EPS, "90° Y rotation");
    }

    // -----------------------------------------------------------------------
    // Tests 13-18: validate_deform_model_lengths (no GPU required)
    // -----------------------------------------------------------------------

    fn make_test_model(n: usize) -> GaussianModel {
        GaussianModel {
            gaussians: vec![
                crate::gaussian::GaussianAttributes {
                    position: [0.0, 0.0, 0.0],
                    _pad0: 0.0,
                    rotation: [0.0, 0.0, 0.0, 1.0],
                    scale: [0.0, 0.0, 0.0],
                    opacity: 0.0,
                };
                n
            ],
            sh_coeffs: Vec::new(),
            sh_degree: 0,
            face_indices: vec![0u32; n],
            barycentric: vec![[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]; n],
            local_offsets: vec![[0.0, 0.0, 0.0]; n],
            is_rigid: vec![false; n],
        }
    }

    #[test]
    fn test_validate_deform_model_lengths_ok() {
        let model = make_test_model(5);
        assert!(validate_deform_model_lengths(&model).is_ok());
    }

    #[test]
    fn test_validate_deform_model_lengths_empty_model_ok() {
        // n == 0: all four per-Gaussian arrays are legitimately empty too.
        let model = make_test_model(0);
        assert!(validate_deform_model_lengths(&model).is_ok());
    }

    #[test]
    fn test_validate_deform_model_lengths_rejects_short_face_indices() {
        let mut model = make_test_model(5);
        model.face_indices.pop();
        assert!(validate_deform_model_lengths(&model).is_err());
    }

    #[test]
    fn test_validate_deform_model_lengths_rejects_short_barycentric() {
        let mut model = make_test_model(5);
        model.barycentric.pop();
        assert!(validate_deform_model_lengths(&model).is_err());
    }

    #[test]
    fn test_validate_deform_model_lengths_rejects_short_local_offsets() {
        let mut model = make_test_model(5);
        model.local_offsets.pop();
        assert!(validate_deform_model_lengths(&model).is_err());
    }

    #[test]
    fn test_validate_deform_model_lengths_rejects_short_is_rigid() {
        let mut model = make_test_model(5);
        model.is_rigid.pop();
        assert!(validate_deform_model_lengths(&model).is_err());
    }

    // -----------------------------------------------------------------------
    // Tests 19-24: deform_cpu (no GPU required)
    // -----------------------------------------------------------------------

    #[test]
    fn test_deform_cpu_empty_model() {
        let model = make_test_model(0);
        let vertices = vec![[0.0_f32, 0.0, 0.0]];
        let normals = vec![[0.0_f32, 0.0, 1.0]];
        let faces = vec![[0u32, 0, 0]];
        let (positions, rotations) = deform_cpu(&model, &vertices, &normals, &faces).unwrap();
        assert!(positions.is_empty());
        assert!(rotations.is_empty());
    }

    #[test]
    fn test_deform_cpu_rejects_mismatched_arrays() {
        let mut model = make_test_model(3);
        model.local_offsets.pop();
        let vertices = vec![[0.0_f32, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let normals = vec![[0.0_f32, 0.0, 1.0]; 3];
        let faces = vec![[0u32, 1, 2]];
        let result = deform_cpu(&model, &vertices, &normals, &faces);
        assert!(result.is_err());
    }

    #[test]
    fn test_deform_cpu_rejects_normals_length_mismatch() {
        let model = make_test_model(1);
        let vertices = vec![[0.0_f32, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let normals = vec![[0.0_f32, 0.0, 1.0]; 2]; // wrong length
        let faces = vec![[0u32, 1, 2]];
        let result = deform_cpu(&model, &vertices, &normals, &faces);
        assert!(result.is_err());
    }

    #[test]
    fn test_deform_cpu_matches_barycentric_interpolation() {
        // Same scenario as the GPU-only `test_deform_gpu_single_gaussian`
        // below: zero local offset, so the deformed position is exactly
        // the barycentric interpolation of the triangle, and (since this
        // triangle's TBN frame happens to be the identity) the rotation is
        // unchanged.
        let vertices = vec![[0.0_f32, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let normals = vec![[0.0_f32, 0.0, 1.0]; 3];
        let faces = vec![[0u32, 1, 2]];

        let model = GaussianModel {
            gaussians: vec![crate::gaussian::GaussianAttributes {
                position: [1.0 / 3.0, 1.0 / 3.0, 0.0],
                _pad0: 0.0,
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: [1.0, 1.0, 1.0],
                opacity: 0.0,
            }],
            sh_coeffs: vec![0.0; 3],
            sh_degree: 0,
            face_indices: vec![0],
            barycentric: vec![[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]],
            local_offsets: vec![[0.0, 0.0, 0.0]],
            is_rigid: vec![false],
        };

        let (positions, rotations) = deform_cpu(&model, &vertices, &normals, &faces).unwrap();
        assert_eq!(positions.len(), 1);
        assert_eq!(rotations.len(), 1);
        assert_approx_eq4(
            positions[0],
            [1.0 / 3.0, 1.0 / 3.0, 0.0, 1.0],
            EPS,
            "barycentric-interpolated position",
        );
        assert_approx_eq4(
            rotations[0],
            [0.0, 0.0, 0.0, 1.0],
            EPS,
            "identity TBN frame leaves rotation unchanged",
        );
    }

    #[test]
    fn test_deform_cpu_invalid_face_index_passes_through() {
        let vertices = vec![[0.0_f32, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let normals = vec![[0.0_f32, 0.0, 1.0]; 3];
        let faces = vec![[0u32, 1, 2]];

        let orig_pos = [5.0_f32, 6.0, 7.0];
        let orig_rot = [0.1_f32, 0.2, 0.3, 0.9];
        let model = GaussianModel {
            gaussians: vec![crate::gaussian::GaussianAttributes {
                position: orig_pos,
                _pad0: 0.0,
                rotation: orig_rot,
                scale: [1.0, 1.0, 1.0],
                opacity: 0.0,
            }],
            sh_coeffs: vec![0.0; 3],
            sh_degree: 0,
            face_indices: vec![99], // out of range: only 1 face exists
            barycentric: vec![[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]],
            local_offsets: vec![[0.0, 0.0, 0.0]],
            is_rigid: vec![false],
        };

        let (positions, rotations) = deform_cpu(&model, &vertices, &normals, &faces).unwrap();
        assert_approx_eq4(
            positions[0],
            [orig_pos[0], orig_pos[1], orig_pos[2], 1.0],
            EPS,
            "out-of-range face index should pass position through unchanged",
        );
        assert_approx_eq4(
            rotations[0],
            orig_rot,
            EPS,
            "out-of-range face index should pass rotation through unchanged",
        );
    }

    #[test]
    fn test_deform_cpu_rigid_gaussian_ignores_mesh_binding() {
        // Same valid, non-degenerate triangle as
        // `test_deform_cpu_matches_barycentric_interpolation`, but with a
        // large local offset and `is_rigid = true`: the mesh binding must
        // have *no* effect on a rigid Gaussian.
        let vertices = vec![[0.0_f32, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let normals = vec![[0.0_f32, 0.0, 1.0]; 3];
        let faces = vec![[0u32, 1, 2]];

        let orig_pos = [10.0_f32, -3.0, 2.5];
        let orig_rot = [0.0_f32, 0.0, 0.0, 1.0];
        let model = GaussianModel {
            gaussians: vec![crate::gaussian::GaussianAttributes {
                position: orig_pos,
                _pad0: 0.0,
                rotation: orig_rot,
                scale: [1.0, 1.0, 1.0],
                opacity: 0.0,
            }],
            sh_coeffs: vec![0.0; 3],
            sh_degree: 0,
            face_indices: vec![0],
            barycentric: vec![[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]],
            // A large local offset that *would* move the Gaussian far away
            // if the mesh binding were applied.
            local_offsets: vec![[100.0, 100.0, 100.0]],
            is_rigid: vec![true],
        };

        let (positions, rotations) = deform_cpu(&model, &vertices, &normals, &faces).unwrap();
        assert_approx_eq4(
            positions[0],
            [orig_pos[0], orig_pos[1], orig_pos[2], 1.0],
            EPS,
            "rigid Gaussian position must be unaffected by the mesh binding",
        );
        assert_approx_eq4(
            rotations[0],
            orig_rot,
            EPS,
            "rigid Gaussian rotation must be unaffected by the mesh binding",
        );
    }

    #[test]
    fn test_deform_cpu_rejects_out_of_range_face_vertex() {
        let vertices = vec![[0.0_f32, 0.0, 0.0], [1.0, 0.0, 0.0]]; // only 2 vertices
        let normals = vec![[0.0_f32, 0.0, 1.0]; 2];
        let faces = vec![[0u32, 1, 2]]; // references vertex 2, which doesn't exist

        let model = make_test_model(1);
        let result = deform_cpu(&model, &vertices, &normals, &faces);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // Regression: the deform kernel must branch on `gaussian_is_rigid`.
    //
    // `DeformPipeline::deform` returns the GPU readback verbatim, so the
    // rigid pass-through is now the *kernel's* responsibility alone. If the
    // early-out is ever dropped from the shader, rigid Gaussians silently
    // get mesh-bound again and nothing on the host re-applies their authored
    // transform. Guards that pairing without needing a GPU adapter; the
    // behavioural proof is `test_deform_gpu_matches_deform_cpu_oracle`.
    // -----------------------------------------------------------------------
    #[test]
    fn test_shader_honours_is_rigid_binding() {
        let shader_src = include_str!("../shaders/deform_gaussians.wgsl");

        // The kernel must actually *read* the binding, not merely declare it.
        assert!(
            shader_src.contains("if gaussian_is_rigid[gaussian_id] != 0u {"),
            "deform_gaussians.wgsl must early-out on gaussian_is_rigid: \
             DeformPipeline::deform no longer post-processes the readback"
        );

        // The rigid early-out must precede the face-index check, so a rigid
        // Gaussian with an out-of-range face index still passes through
        // (matching deform_cpu's ordering).
        let rigid_at = shader_src
            .find("if gaussian_is_rigid[gaussian_id] != 0u {")
            .expect("rigid branch present");
        let face_at = shader_src
            .find("if fi >= uniforms.num_faces {")
            .expect("face-index branch present");
        assert!(
            rigid_at < face_at,
            "the rigid early-out must be checked before the face-index bound"
        );
    }

    // -----------------------------------------------------------------------
    // MeshBuffers identity (F248 prerequisite for buffer/bind-group caching)
    // -----------------------------------------------------------------------
    #[test]
    fn test_next_mesh_id_is_unique_and_monotonic() {
        let a = next_mesh_id();
        let b = next_mesh_id();
        let c = next_mesh_id();
        assert!(a < b && b < c, "mesh ids must be strictly increasing");
    }

    // -----------------------------------------------------------------------
    // GPU integration test (requires a real GPU adapter)
    // -----------------------------------------------------------------------

    /// Create a device suitable for `DeformPipeline`, or `None` when no
    /// adapter is available.
    fn try_make_gpu_device() -> Option<(Arc<wgpu::Device>, Arc<wgpu::Queue>)> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }))
        .ok()?;

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("test_device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits {
                // See DEFORM_STORAGE_BINDINGS: 11 storage buffers in one
                // stage, above the wgpu default of 8.
                max_storage_buffers_per_shader_stage: 16,
                ..wgpu::Limits::default()
            },
            memory_hints: wgpu::MemoryHints::Performance,
            experimental_features: wgpu::ExperimentalFeatures::default(),
            trace: wgpu::Trace::Off,
        }))
        .ok()?;

        Some((Arc::new(device), Arc::new(queue)))
    }

    /// The GPU kernel and [`deform_cpu`] must agree exactly — that oracle
    /// relationship is what `deform_cpu`'s doc claims, and it is the real
    /// proof that the shader honours `gaussian_is_rigid` itself now that
    /// `DeformPipeline::deform` returns the readback unmodified.
    ///
    /// Covers all three kernel paths in one dispatch: the rigid early-out
    /// (including a rigid Gaussian whose face index is out of range), the
    /// invalid-face pass-through, and the normal barycentric + TBN path.
    #[test]
    #[ignore = "requires GPU adapter"]
    fn test_deform_gpu_matches_deform_cpu_oracle() {
        // A single well-conditioned unit triangle: the shader's degenerate
        // thresholds (1e-4) and build_tbn's (1e-7) differ, so near-degenerate
        // geometry could take different branches for reasons unrelated to
        // the rigid flag.
        let vertices = vec![
            [0.0_f32, 0.0, 0.0],
            [1.0_f32, 0.0, 0.0],
            [0.0_f32, 1.0, 0.0],
        ];
        let normals = vec![[0.0_f32, 0.0, 1.0]; 3];
        let faces = vec![[0u32, 1, 2]];

        let attrs = |position: [f32; 3], rotation: [f32; 4]| crate::gaussian::GaussianAttributes {
            position,
            _pad0: 0.0,
            rotation,
            scale: [1.0, 1.0, 1.0],
            opacity: 0.0,
        };
        let s = std::f32::consts::FRAC_1_SQRT_2;

        let model = GaussianModel {
            gaussians: vec![
                // 0: flexible, mesh-bound, non-zero offset and rotation.
                attrs([0.0, 0.0, 0.0], [0.0, 0.0, s, s]),
                // 1: rigid with a large offset that would move it far away
                //    if the mesh binding were (wrongly) applied.
                attrs([10.0, -3.0, 2.5], [0.0, 0.0, 0.0, 1.0]),
                // 2: rigid *and* out-of-range face index — exercises the
                //    ordering of the two early-outs.
                attrs([-4.0, 8.0, 1.5], [s, 0.0, 0.0, s]),
                // 3: flexible with an out-of-range face index.
                attrs([5.0, 6.0, 7.0], [0.0, s, 0.0, s]),
            ],
            sh_coeffs: vec![0.0; 4 * 3],
            sh_degree: 0,
            face_indices: vec![0, 0, 99, 99],
            barycentric: vec![[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]; 4],
            local_offsets: vec![
                [0.25, -0.125, 0.5],
                [100.0, 100.0, 100.0],
                [100.0, 100.0, 100.0],
                [1.0, 2.0, 3.0],
            ],
            is_rigid: vec![false, true, true, false],
        };

        let Some((device, queue)) = try_make_gpu_device() else {
            eprintln!("No GPU adapter available, skipping GPU test");
            return;
        };

        let pipeline = DeformPipeline::new(Arc::clone(&device), Arc::clone(&queue))
            .expect("Failed to create DeformPipeline");
        let mesh = pipeline
            .upload_mesh(&vertices, &normals, &faces)
            .expect("upload_mesh failed");
        let (gpu_pos, gpu_rot) = pipeline.deform(&model, &mesh).expect("Deform failed");
        let (cpu_pos, cpu_rot) =
            deform_cpu(&model, &vertices, &normals, &faces).expect("deform_cpu failed");

        assert_eq!(gpu_pos.len(), model.len());
        assert_eq!(cpu_pos.len(), model.len());

        const GPU_EPS: f32 = 1e-4;
        for i in 0..model.len() {
            assert_approx_eq4(
                gpu_pos[i],
                cpu_pos[i],
                GPU_EPS,
                &format!("Gaussian {i}: GPU position must match the deform_cpu oracle"),
            );
            assert_approx_eq4(
                quat_canonical(gpu_rot[i]),
                quat_canonical(cpu_rot[i]),
                GPU_EPS,
                &format!("Gaussian {i}: GPU rotation must match the deform_cpu oracle"),
            );
        }

        // Pin the rigid semantics explicitly rather than trusting only the
        // oracle: both rigid Gaussians must be byte-for-byte pass-throughs.
        for i in [1_usize, 2] {
            let g = &model.gaussians[i];
            assert_approx_eq4(
                gpu_pos[i],
                [g.position[0], g.position[1], g.position[2], 1.0],
                GPU_EPS,
                &format!("rigid Gaussian {i} must keep its authored position"),
            );
            assert_approx_eq4(
                gpu_rot[i],
                g.rotation,
                GPU_EPS,
                &format!("rigid Gaussian {i} must keep its authored rotation"),
            );
        }

        // The flexible, mesh-bound Gaussian must actually have moved —
        // otherwise an all-pass-through kernel would satisfy the oracle.
        let g0 = &model.gaussians[0];
        let moved = (gpu_pos[0][0] - g0.position[0]).abs()
            + (gpu_pos[0][1] - g0.position[1]).abs()
            + (gpu_pos[0][2] - g0.position[2]).abs();
        assert!(
            moved > 1e-3,
            "flexible Gaussian 0 should be repositioned by the mesh binding, got {:?}",
            gpu_pos[0]
        );

        // Every position is a homogeneous point: w == 1.0 on all three
        // kernel paths (normal, rigid early-out, invalid-face pass-through).
        for (i, p) in gpu_pos.iter().enumerate() {
            assert!(
                (p[3] - 1.0).abs() < GPU_EPS,
                "Gaussian {i}: position w must be 1.0, got {}",
                p[3]
            );
        }
    }

    /// A device created with wgpu's default limits cannot host this
    /// pipeline (8 storage buffers available, DEFORM_STORAGE_BINDINGS
    /// needed). `DeformPipeline::new` must reject it with a clean error
    /// rather than letting `create_bind_group_layout` raise an uncaptured
    /// wgpu validation error, which aborts the process.
    #[test]
    #[ignore = "requires GPU adapter"]
    fn test_deform_pipeline_new_rejects_insufficient_storage_buffer_limit() {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let Ok(adapter) =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
                apply_limit_buckets: false,
            }))
        else {
            eprintln!("No GPU adapter available, skipping GPU test");
            return;
        };

        // Deliberately default limits: only 8 storage buffers per stage.
        let Ok((device, queue)) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("test_device_default_limits"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                experimental_features: wgpu::ExperimentalFeatures::default(),
                trace: wgpu::Trace::Off,
            }))
        else {
            eprintln!("Could not create a default-limits device, skipping");
            return;
        };

        assert!(
            device.limits().max_storage_buffers_per_shader_stage < DEFORM_STORAGE_BINDINGS,
            "precondition: the default-limits device must be under the requirement"
        );

        let result = DeformPipeline::new(Arc::new(device), Arc::new(queue));
        match result {
            Ok(_) => {
                panic!("DeformPipeline::new must reject a device with too few storage buffers")
            }
            Err(e) => {
                let msg = e.to_string();
                assert!(
                    msg.contains("max_storage_buffers_per_shader_stage"),
                    "error should name the offending limit, got: {msg}"
                );
            }
        }
    }

    /// Distinct [`MeshBuffers`] uploads must carry distinct identities — the
    /// prerequisite for keying any future bind-group cache on the mesh.
    #[test]
    #[ignore = "requires GPU adapter"]
    fn test_upload_mesh_assigns_unique_ids() {
        let vertices = vec![
            [0.0_f32, 0.0, 0.0],
            [1.0_f32, 0.0, 0.0],
            [0.0_f32, 1.0, 0.0],
        ];
        let normals = vec![[0.0_f32, 0.0, 1.0]; 3];
        let faces = vec![[0u32, 1, 2]];

        let Some((device, queue)) = try_make_gpu_device() else {
            eprintln!("No GPU adapter available, skipping GPU test");
            return;
        };
        let pipeline = DeformPipeline::new(device, queue).expect("Failed to create DeformPipeline");

        let mesh_a = pipeline
            .upload_mesh(&vertices, &normals, &faces)
            .expect("upload_mesh failed");
        let mesh_b = pipeline
            .upload_mesh(&vertices, &normals, &faces)
            .expect("upload_mesh failed");

        // Same geometry, same sizes — but a different upload, so a cache
        // keyed only on capacity would wrongly treat these as identical.
        assert_ne!(
            mesh_a.id, mesh_b.id,
            "two uploads must never share a mesh id"
        );
        assert_eq!(mesh_a.num_vertices, mesh_b.num_vertices);
        assert_eq!(mesh_a.num_faces, mesh_b.num_faces);
    }

    #[test]
    #[ignore = "requires GPU adapter"]
    fn test_deform_gpu_single_gaussian() {
        // Build a trivial triangle mesh (XY plane, unit triangle).
        let vertices = vec![
            [0.0_f32, 0.0, 0.0],
            [1.0_f32, 0.0, 0.0],
            [0.0_f32, 1.0, 0.0],
        ];
        let normals = vec![
            [0.0_f32, 0.0, 1.0],
            [0.0_f32, 0.0, 1.0],
            [0.0_f32, 0.0, 1.0],
        ];
        let faces = vec![[0u32, 1, 2]];

        // A single Gaussian at the centroid with no offset.
        let model = GaussianModel {
            gaussians: vec![crate::gaussian::GaussianAttributes {
                position: [1.0 / 3.0, 1.0 / 3.0, 0.0],
                _pad0: 0.0,
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: [1.0, 1.0, 1.0],
                opacity: 0.0,
            }],
            sh_coeffs: vec![0.0; 3],
            sh_degree: 0,
            face_indices: vec![0],
            barycentric: vec![[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]],
            local_offsets: vec![[0.0, 0.0, 0.0]],
            is_rigid: vec![false],
        };

        // Set up GPU.
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }));
        let adapter = match adapter {
            Ok(a) => a,
            Err(_) => {
                eprintln!("No GPU adapter available, skipping GPU test");
                return;
            }
        };

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("test_device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits {
                // `deform_bgl_0` declares DEFORM_STORAGE_BINDINGS (11) storage
                // buffers in one stage; the wgpu default is 8, which makes
                // `create_bind_group_layout` fail. Same override the
                // rasterizer's own device request uses.
                max_storage_buffers_per_shader_stage: 16,
                ..wgpu::Limits::default()
            },
            memory_hints: wgpu::MemoryHints::Performance,
            experimental_features: wgpu::ExperimentalFeatures::default(),
            trace: wgpu::Trace::Off,
        }))
        .expect("Failed to create device");

        let device = Arc::new(device);
        let queue = Arc::new(queue);

        let pipeline = DeformPipeline::new(Arc::clone(&device), Arc::clone(&queue))
            .expect("Failed to create DeformPipeline");

        let mesh = pipeline
            .upload_mesh(&vertices, &normals, &faces)
            .expect("upload_mesh failed");
        let (positions, rotations) = pipeline.deform(&model, &mesh).expect("Deform failed");

        assert_eq!(positions.len(), 1, "Should have 1 output position");
        assert_eq!(rotations.len(), 1, "Should have 1 output rotation");

        // With zero offset, the world position should be the barycentric interpolation.
        let expected_pos = [1.0 / 3.0, 1.0 / 3.0, 0.0];
        let [px, py, pz, _] = positions[0];
        assert!((px - expected_pos[0]).abs() < 1e-4, "x: {px}");
        assert!((py - expected_pos[1]).abs() < 1e-4, "y: {py}");
        assert!((pz - expected_pos[2]).abs() < 1e-4, "z: {pz}");
    }
}
