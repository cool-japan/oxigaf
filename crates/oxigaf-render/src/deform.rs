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

/// GPU buffers holding the FLAME mesh geometry needed for deformation.
pub struct MeshBuffers {
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

    #[allow(dead_code)]
    fn to_array4(self) -> [f32; 4] {
        [self.x, self.y, self.z, self.w]
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

impl DeformPipeline {
    /// Compile the deform shader and create the pipeline.
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Result<Self, RenderError> {
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
    pub fn upload_mesh(
        &self,
        vertices: &[[f32; 3]],
        normals: &[[f32; 3]],
        faces: &[[u32; 3]],
    ) -> MeshBuffers {
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

        MeshBuffers {
            vertices: vert_buf,
            normals: norm_buf,
            faces: face_buf,
            num_vertices: vertices.len() as u32,
            num_faces: faces.len() as u32,
        }
    }

    /// Deform Gaussians bound to the given mesh.
    ///
    /// Returns `(deformed_positions, deformed_rotations)` where positions are
    /// `[x, y, z, 1.0]` and rotations are `[x, y, z, w]` quaternions.
    pub fn deform(
        &self,
        model: &GaussianModel,
        mesh: &MeshBuffers,
    ) -> Result<DeformResult, RenderError> {
        let n = model.len() as u32;
        if n == 0 {
            return Ok((Vec::new(), Vec::new()));
        }

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
        let deformed_pos = self.readback_vec4(&out_pos_buf, n as usize)?;
        let deformed_rot = self.readback_vec4(&out_rot_buf, n as usize)?;

        Ok((deformed_pos, deformed_rot))
    }

    // ---- Internal helpers ----

    /// Read back `count` `vec4<f32>` entries from a GPU storage buffer.
    fn readback_vec4(
        &self,
        buffer: &wgpu::Buffer,
        count: usize,
    ) -> Result<Vec<[f32; 4]>, RenderError> {
        let byte_size = (count * std::mem::size_of::<[f32; 4]>()) as u64;

        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("deform_staging"),
            size: byte_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("deform_readback"),
            });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, byte_size);
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
        let result: Vec<[f32; 4]> = floats
            .chunks_exact(4)
            .take(count)
            .map(|c| [c[0], c[1], c[2], c[3]])
            .collect();
        drop(mapped);
        staging.unmap();

        Ok(result)
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
// CPU-side math helpers (mirroring the WGSL shader logic)
// Used for testing and for any CPU fallback path.
// ---------------------------------------------------------------------------

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
    // GPU integration test (requires a real GPU adapter)
    // -----------------------------------------------------------------------
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
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::Performance,
            experimental_features: wgpu::ExperimentalFeatures::default(),
            trace: wgpu::Trace::Off,
        }))
        .expect("Failed to create device");

        let device = Arc::new(device);
        let queue = Arc::new(queue);

        let pipeline = DeformPipeline::new(Arc::clone(&device), Arc::clone(&queue))
            .expect("Failed to create DeformPipeline");

        let mesh = pipeline.upload_mesh(&vertices, &normals, &faces);
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
