//! FLAME mesh binding for Gaussians.
//!
//! Each Gaussian is bound to a face on the FLAME mesh via barycentric
//! coordinates and a learnable local offset. This module computes the
//! world-space Gaussian positions from the current FLAME mesh pose and
//! the binding parameters.
//!
//! # Indexing contract
//!
//! Two different frame arrays appear in this module and they are **not**
//! interchangeable:
//!
//! * [`BindingResult::face_frames`] — one frame **per Gaussian**, the frame of
//!   the face that Gaussian is bound to. Used to orient the Gaussian
//!   ([`frame_to_quaternion`]).
//! * [`build_vertex_frames`] — one frame **per mesh vertex**. This is what the
//!   `flame_binding_bwd` shader indexes (`tbn_frames[vertex_id]`), so it is what
//!   [`FlameBindingBuffers::update_tbn_frames`] requires.

use nalgebra as na;
use oxigaf_flame::Mesh;

use crate::gaussian::GaussianModel;
use crate::RenderError;

/// Workgroup size of the `flame_binding_backward` kernel
/// (`shaders/flame_binding_bwd.wgsl`).
const BINDING_BWD_WORKGROUP_SIZE: u32 = 256;

/// Degeneracy threshold for vector lengths.
const FRAME_EPSILON: f32 = 1e-10;

/// Result of applying FLAME mesh binding to a Gaussian model.
#[derive(Debug)]
pub struct BindingResult {
    /// Updated world-space positions for each Gaussian.
    pub positions: Vec<[f32; 3]>,
    /// Coordinate frame of the bound face, **one entry per Gaussian**
    /// (not per mesh vertex — see the module-level indexing contract).
    pub face_frames: Vec<FaceFrame>,
}

/// Local coordinate frame on a mesh face.
#[derive(Debug, Clone, Copy)]
pub struct FaceFrame {
    /// Tangent direction (edge0 normalized).
    pub tangent: na::Vector3<f32>,
    /// Bitangent (normal × tangent).
    pub bitangent: na::Vector3<f32>,
    /// Face normal.
    pub normal: na::Vector3<f32>,
}

impl FaceFrame {
    /// Axis-aligned fallback frame, used when a normal or tangent is degenerate.
    #[must_use]
    pub fn axis_aligned() -> Self {
        Self {
            tangent: na::Vector3::x(),
            bitangent: na::Vector3::y(),
            normal: na::Vector3::z(),
        }
    }

    /// Build an orthonormal frame from a (possibly unnormalized) `normal` and a
    /// `tangent_hint`.
    ///
    /// The hint is Gram-Schmidt orthogonalized against the normal. When the
    /// normal is degenerate the frame falls back to [`FaceFrame::axis_aligned`];
    /// when only the hint is unusable (zero, non-finite, or parallel to the
    /// normal) an arbitrary perpendicular axis is chosen instead.
    #[must_use]
    pub fn from_normal_and_tangent(
        normal: na::Vector3<f32>,
        tangent_hint: na::Vector3<f32>,
    ) -> Self {
        let normal_len = normal.norm();
        if !normal_len.is_finite() || normal_len <= FRAME_EPSILON {
            return Self::axis_aligned();
        }
        let unit_normal = normal / normal_len;

        // Remove the normal component from the hint.
        let mut tangent = tangent_hint - unit_normal * unit_normal.dot(&tangent_hint);
        let mut tangent_len = tangent.norm();
        if !tangent_len.is_finite() || tangent_len <= FRAME_EPSILON {
            // Hint unusable: pick the world axis least aligned with the normal.
            let nx = unit_normal.x.abs();
            let ny = unit_normal.y.abs();
            let nz = unit_normal.z.abs();
            let axis = if nx <= ny && nx <= nz {
                na::Vector3::x()
            } else if ny <= nz {
                na::Vector3::y()
            } else {
                na::Vector3::z()
            };
            tangent = axis - unit_normal * unit_normal.dot(&axis);
            tangent_len = tangent.norm();
        }
        if !tangent_len.is_finite() || tangent_len <= FRAME_EPSILON {
            return Self::axis_aligned();
        }

        let tangent = tangent / tangent_len;
        let bitangent = unit_normal.cross(&tangent);
        Self {
            tangent,
            bitangent,
            normal: unit_normal,
        }
    }
}

/// Look up a mesh vertex, reporting the owning Gaussian on failure.
fn lookup_vertex(
    mesh: &Mesh,
    vertex_id: u32,
    gaussian_index: usize,
) -> Result<&na::Point3<f32>, RenderError> {
    mesh.vertices
        .get(vertex_id as usize)
        .ok_or_else(|| RenderError::InvalidGaussian {
            index: gaussian_index,
            reason: format!(
                "bound face references vertex {vertex_id}, but the mesh has {} vertices",
                mesh.vertices.len()
            ),
        })
}

/// Compute the binding positions for all Gaussians given the current FLAME mesh.
///
/// For each Gaussian:
/// 1. Look up its bound face on the mesh.
/// 2. Interpolate the position using barycentric coordinates.
/// 3. Compute the face's local frame (tangent, bitangent, normal).
/// 4. Apply the local offset in the face's local frame.
///
/// Rigid Gaussians have `local_offsets = [0, 0, 0]`.
///
/// # Errors
///
/// * [`RenderError::MismatchedBufferSizes`] when `face_indices`, `barycentric`
///   or `local_offsets` do not have exactly one entry per Gaussian (they are
///   empty for any model loaded from PLY/SafeTensors without binding data).
/// * [`RenderError::ValidationError`] when the mesh has no faces or vertices.
/// * [`RenderError::InvalidGaussian`] when a Gaussian references a face index
///   outside the mesh, or a face references a vertex outside the mesh.
pub fn apply_binding(model: &GaussianModel, mesh: &Mesh) -> Result<BindingResult, RenderError> {
    let n = model.len();

    // The three binding arrays are independent Vecs; all must be fully populated.
    if model.face_indices.len() != n {
        return Err(RenderError::MismatchedBufferSizes {
            expected: n,
            actual: model.face_indices.len(),
        });
    }
    if model.barycentric.len() != n {
        return Err(RenderError::MismatchedBufferSizes {
            expected: n,
            actual: model.barycentric.len(),
        });
    }
    if model.local_offsets.len() != n {
        return Err(RenderError::MismatchedBufferSizes {
            expected: n,
            actual: model.local_offsets.len(),
        });
    }

    if n == 0 {
        return Ok(BindingResult {
            positions: Vec::new(),
            face_frames: Vec::new(),
        });
    }

    if mesh.faces.is_empty() || mesh.vertices.is_empty() {
        return Err(RenderError::ValidationError(format!(
            "FLAME binding requires a non-empty mesh, got {} faces and {} vertices",
            mesh.faces.len(),
            mesh.vertices.len()
        )));
    }

    let mut positions = Vec::with_capacity(n);
    let mut face_frames = Vec::with_capacity(n);

    for i in 0..n {
        let face_idx = model.face_indices[i] as usize;
        let bary = model.barycentric[i];
        let offset = model.local_offsets[i];

        // Get face vertices (no silent clamping: an out-of-range index is a bug
        // in the binding data, not something to paper over with the last face).
        let face = mesh
            .faces
            .get(face_idx)
            .ok_or_else(|| RenderError::InvalidGaussian {
                index: i,
                reason: format!(
                    "bound face index {face_idx} out of range (mesh has {} faces)",
                    mesh.faces.len()
                ),
            })?;
        let v0 = lookup_vertex(mesh, face[0], i)?;
        let v1 = lookup_vertex(mesh, face[1], i)?;
        let v2 = lookup_vertex(mesh, face[2], i)?;

        // Interpolate position
        let base_pos = v0.coords * bary[0] + v1.coords * bary[1] + v2.coords * bary[2];

        // Compute face frame (edge0 is already perpendicular to the face normal,
        // so the Gram-Schmidt step inside `from_normal_and_tangent` is a no-op).
        let edge0 = v1 - v0;
        let edge1 = v2 - v0;
        let frame = FaceFrame::from_normal_and_tangent(edge0.cross(&edge1), edge0);

        // Apply local offset in face frame
        let world_offset =
            frame.tangent * offset[0] + frame.bitangent * offset[1] + frame.normal * offset[2];

        let pos = base_pos + world_offset;
        positions.push([pos.x, pos.y, pos.z]);
        face_frames.push(frame);
    }

    Ok(BindingResult {
        positions,
        face_frames,
    })
}

/// Build one TBN frame **per mesh vertex**.
///
/// This is the array the `flame_binding_bwd` shader expects: it indexes
/// `tbn_frames[vertex_id]`, so entry `k` must be the frame of mesh vertex `k`.
/// The returned vector always has exactly `mesh.vertices.len()` entries.
///
/// The normal is taken from `mesh.normals` when that array is present, sized
/// consistently and non-degenerate; otherwise it is recomputed as the
/// area-weighted average of the incident face normals. The tangent is the
/// average incident edge direction, orthogonalized against the normal.
/// Isolated or degenerate vertices get [`FaceFrame::axis_aligned`].
#[must_use]
pub fn build_vertex_frames(mesh: &Mesh) -> Vec<FaceFrame> {
    let num_vertices = mesh.vertices.len();
    let mut accum_normal = vec![na::Vector3::<f32>::zeros(); num_vertices];
    let mut accum_tangent = vec![na::Vector3::<f32>::zeros(); num_vertices];

    for face in &mesh.faces {
        let i0 = face[0] as usize;
        let i1 = face[1] as usize;
        let i2 = face[2] as usize;
        let (Some(v0), Some(v1), Some(v2)) = (
            mesh.vertices.get(i0),
            mesh.vertices.get(i1),
            mesh.vertices.get(i2),
        ) else {
            // Skip faces referencing vertices outside the mesh.
            continue;
        };

        let edge0 = v1 - v0;
        let edge1 = v2 - v0;
        // Un-normalized cross product => area weighting.
        let face_normal = edge0.cross(&edge1);

        for &vid in &[i0, i1, i2] {
            if let Some(slot) = accum_normal.get_mut(vid) {
                *slot += face_normal;
            }
            if let Some(slot) = accum_tangent.get_mut(vid) {
                *slot += edge0;
            }
        }
    }

    // Only trust the mesh's own normals when the array is sized consistently.
    let use_mesh_normals = mesh.normals.len() == num_vertices;

    (0..num_vertices)
        .map(|i| {
            let mut normal = accum_normal[i];
            if use_mesh_normals {
                if let Some(mesh_normal) = mesh.normals.get(i) {
                    let len = mesh_normal.norm();
                    if len.is_finite() && len > FRAME_EPSILON {
                        normal = *mesh_normal;
                    }
                }
            }
            FaceFrame::from_normal_and_tangent(normal, accum_tangent[i])
        })
        .collect()
}

/// Build one TBN frame **per mesh face**, in face order.
///
/// This is the frame [`apply_binding`] actually applies the local offset in, so
/// a backward pass that projects gradients onto `build_face_frames(mesh)`
/// indexed by `face_indices[i]` is the exact adjoint of the forward binding.
/// (Per-vertex frames from [`build_vertex_frames`] are an approximation of it.)
///
/// Faces referencing vertices outside the mesh get [`FaceFrame::axis_aligned`],
/// so the returned vector always has exactly `mesh.faces.len()` entries.
#[must_use]
pub fn build_face_frames(mesh: &Mesh) -> Vec<FaceFrame> {
    mesh.faces
        .iter()
        .map(|face| {
            let (Some(v0), Some(v1), Some(v2)) = (
                mesh.vertices.get(face[0] as usize),
                mesh.vertices.get(face[1] as usize),
                mesh.vertices.get(face[2] as usize),
            ) else {
                return FaceFrame::axis_aligned();
            };
            let edge0 = v1 - v0;
            let edge1 = v2 - v0;
            FaceFrame::from_normal_and_tangent(edge0.cross(&edge1), edge0)
        })
        .collect()
}

/// Compute position regularization: sum of squared distances between
/// each Gaussian's current position and its binding position.
pub fn position_regularization(
    current_positions: &[[f32; 3]],
    binding_positions: &[[f32; 3]],
) -> f32 {
    current_positions
        .iter()
        .zip(binding_positions.iter())
        .map(|(cur, bind)| {
            let dx = cur[0] - bind[0];
            let dy = cur[1] - bind[1];
            let dz = cur[2] - bind[2];
            dx * dx + dy * dy + dz * dz
        })
        .sum()
}

/// Compute scale regularization: penalize Gaussians whose scale exceeds `max_scale`.
pub fn scale_regularization(scales: &[[f32; 3]], max_scale: f32) -> f32 {
    scales
        .iter()
        .map(|s| {
            let excess_x = (s[0].exp() - max_scale).max(0.0);
            let excess_y = (s[1].exp() - max_scale).max(0.0);
            let excess_z = (s[2].exp() - max_scale).max(0.0);
            excess_x * excess_x + excess_y * excess_y + excess_z * excess_z
        })
        .sum()
}

/// Convert a face frame to a quaternion (for initializing Gaussian rotations).
pub fn frame_to_quaternion(frame: &FaceFrame) -> [f32; 4] {
    // Build rotation matrix from frame columns (tangent, bitangent, normal)
    let rot = na::Matrix3::from_columns(&[frame.tangent, frame.bitangent, frame.normal]);
    let uq = na::UnitQuaternion::from_rotation_matrix(&na::Rotation3::from_matrix_unchecked(rot));
    let q = uq.quaternion();
    // wgpu convention: [x, y, z, w]
    [q.i, q.j, q.k, q.w]
}

/// CPU reference for the `flame_binding_backward` kernel.
///
/// Projects `∂L/∂position` onto each Gaussian's bound TBN frame:
/// `∂L/∂offset = [dot(g, T), dot(g, B), dot(g, N)]`. Gaussians whose frame is
/// degenerate (zero normal) receive a zero gradient, matching the shader.
///
/// `vertex_frames` must be indexed **by mesh vertex** (see
/// [`build_vertex_frames`]), `position_grads` and `vertex_ids` by Gaussian.
///
/// # Errors
///
/// * [`RenderError::MismatchedBufferSizes`] when `vertex_ids` and
///   `position_grads` differ in length.
/// * [`RenderError::InvalidGaussian`] when a vertex id is out of range.
pub fn offset_gradients_cpu(
    position_grads: &[[f32; 3]],
    vertex_ids: &[u32],
    vertex_frames: &[FaceFrame],
) -> Result<Vec<[f32; 3]>, RenderError> {
    if vertex_ids.len() != position_grads.len() {
        return Err(RenderError::MismatchedBufferSizes {
            expected: position_grads.len(),
            actual: vertex_ids.len(),
        });
    }

    let mut out = Vec::with_capacity(position_grads.len());
    for (i, (&vid, grad)) in vertex_ids.iter().zip(position_grads.iter()).enumerate() {
        let frame =
            vertex_frames
                .get(vid as usize)
                .ok_or_else(|| RenderError::InvalidGaussian {
                    index: i,
                    reason: format!(
                        "bound vertex id {vid} out of range ({} vertex frames)",
                        vertex_frames.len()
                    ),
                })?;
        // Match the shader's degenerate-frame guard (`length(N) > 0.0001`).
        if frame.normal.norm() <= 1e-4 {
            out.push([0.0, 0.0, 0.0]);
            continue;
        }
        let g = na::Vector3::new(grad[0], grad[1], grad[2]);
        out.push([
            g.dot(&frame.tangent),
            g.dot(&frame.bitangent),
            g.dot(&frame.normal),
        ]);
    }
    Ok(out)
}

/// GPU buffers for the FLAME binding backward pass.
///
/// These buffers feed the `flame_binding_bwd` compute pipeline, which computes
/// gradients w.r.t. the learnable local offsets given gradients w.r.t. the
/// world-space Gaussian positions.
///
/// # Frame table
///
/// The shader reads `tbn_frames[binding_info[gaussian_id].vertex_id]`, i.e.
/// `tbn_frames` is a **frame table** of `num_vertices` entries and
/// `binding_info` holds one index into it per Gaussian. Two configurations are
/// valid, and `num_vertices` names the table length in both:
///
/// * frame table = [`build_vertex_frames`], indices = mesh vertex ids —
///   an approximation, one frame per vertex;
/// * frame table = [`build_face_frames`], indices = `GaussianModel::face_indices` —
///   exactly the frames [`apply_binding`] used in the forward pass.
///
/// Passing the per-Gaussian [`BindingResult::face_frames`] here is always wrong:
/// entry `k` would be the frame of Gaussian `k`, not of index `k`.
///
/// Typical use, once per training step:
///
/// 1. [`update_binding_info`](Self::update_binding_info) — vertex id per Gaussian (static).
/// 2. [`update_tbn_frames`](Self::update_tbn_frames) — per-**vertex** frames for the current pose.
/// 3. [`update_position_gradients`](Self::update_position_gradients) or
///    [`copy_position_gradients_from`](Self::copy_position_gradients_from) — `∂L/∂position`.
/// 4. [`backward`](Self::backward) (or [`record_backward`](Self::record_backward) +
///    [`read_offset_gradients`](Self::read_offset_gradients)) — dispatch and read `∂L/∂offset`.
pub struct FlameBindingBuffers {
    /// Binding info buffer (vertex_id per Gaussian).
    pub binding_info: wgpu::Buffer,
    /// TBN frames buffer (tangent, bitangent, normal **per mesh vertex**).
    pub tbn_frames: wgpu::Buffer,
    /// Position gradients input buffer.
    pub position_grads: wgpu::Buffer,
    /// Local offset gradients output buffer.
    pub offset_grads: wgpu::Buffer,
    /// Uniform buffer for parameters.
    pub uniforms: wgpu::Buffer,
    /// Bind group for the backward pass.
    pub bind_group: wgpu::BindGroup,
    /// Number of Gaussians.
    num_gaussians: usize,
    /// Number of vertices.
    num_vertices: usize,
}

impl FlameBindingBuffers {
    /// Create new FLAME binding buffers.
    ///
    /// # Arguments
    ///
    /// * `device` - WGPU device
    /// * `bind_group_layout` - Bind group layout for the flame_binding_bwd shader
    /// * `num_gaussians` - Number of Gaussians
    /// * `num_vertices` - Number of mesh vertices
    pub fn new(
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
        num_gaussians: usize,
        num_vertices: usize,
    ) -> Self {
        use wgpu::util::DeviceExt;

        // Zero-sized storage buffers are a validation error, so every buffer is
        // allocated with room for at least one element.
        let gaussian_slots = num_gaussians.max(1) as u64;
        let vertex_slots = num_vertices.max(1) as u64;

        // Binding info buffer: [vertex_id, pad, pad, pad] per Gaussian
        let binding_info = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flame_binding_info"),
            size: gaussian_slots * 16, // 4 u32s = 16 bytes
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // TBN frames buffer: [T, pad, B, pad, N, pad] per vertex
        let tbn_frames = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flame_tbn_frames"),
            size: vertex_slots * 48, // 3 vec4s = 48 bytes
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Position gradients buffer: [grad, pad] per Gaussian
        let position_grads = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flame_position_grads"),
            size: gaussian_slots * 16, // vec4 = 16 bytes
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Offset gradients buffer: [grad, pad] per Gaussian
        let offset_grads = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flame_offset_grads"),
            size: gaussian_slots * 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Uniforms buffer
        let num_gaussians_u32 = u32::try_from(num_gaussians).unwrap_or(u32::MAX);
        let uniforms_data = [num_gaussians_u32, 0, 0, 0]; // [num_gaussians, pad, pad, pad]
        let uniforms = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("flame_binding_uniforms"),
            contents: bytemuck::cast_slice(&uniforms_data),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("flame_binding_bwd_bind_group"),
            layout: bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniforms.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: binding_info.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: tbn_frames.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: position_grads.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: offset_grads.as_entire_binding(),
                },
            ],
        });

        Self {
            binding_info,
            tbn_frames,
            position_grads,
            offset_grads,
            uniforms,
            bind_group,
            num_gaussians,
            num_vertices,
        }
    }

    /// Number of Gaussians these buffers were allocated for.
    #[inline]
    #[must_use]
    pub fn num_gaussians(&self) -> usize {
        self.num_gaussians
    }

    /// Number of mesh vertices these buffers were allocated for.
    #[inline]
    #[must_use]
    pub fn num_vertices(&self) -> usize {
        self.num_vertices
    }

    /// Update binding info (vertex IDs per Gaussian).
    ///
    /// # Arguments
    ///
    /// * `queue` - WGPU queue
    /// * `vertex_ids` - Vertex ID for each Gaussian; must contain exactly
    ///   `num_gaussians` entries so no stale entry survives in the buffer.
    ///
    /// # Errors
    ///
    /// * [`RenderError::MismatchedBufferSizes`] when the slice length differs
    ///   from `num_gaussians`.
    /// * [`RenderError::InvalidGaussian`] when a vertex id would index past the
    ///   TBN frame buffer (the shader has no bounds check).
    pub fn update_binding_info(
        &self,
        queue: &wgpu::Queue,
        vertex_ids: &[u32],
    ) -> Result<(), RenderError> {
        if vertex_ids.len() != self.num_gaussians {
            return Err(RenderError::MismatchedBufferSizes {
                expected: self.num_gaussians,
                actual: vertex_ids.len(),
            });
        }

        // Pack as [vertex_id, pad, pad, pad]
        let mut data = Vec::with_capacity(self.num_gaussians * 4);
        for (i, &vid) in vertex_ids.iter().enumerate() {
            if vid as usize >= self.num_vertices {
                return Err(RenderError::InvalidGaussian {
                    index: i,
                    reason: format!(
                        "bound vertex id {vid} out of range (mesh has {} vertices)",
                        self.num_vertices
                    ),
                });
            }
            data.extend_from_slice(&[vid, 0, 0, 0]);
        }
        if !data.is_empty() {
            queue.write_buffer(&self.binding_info, 0, bytemuck::cast_slice(&data));
        }
        Ok(())
    }

    /// Update TBN frames.
    ///
    /// # Arguments
    ///
    /// * `queue` - WGPU queue
    /// * `vertex_frames` - **Per-vertex** frames from [`build_vertex_frames`];
    ///   must contain exactly `num_vertices` entries. Passing the per-Gaussian
    ///   `BindingResult::face_frames` here is a bug: the shader indexes this
    ///   buffer by vertex id.
    ///
    /// # Errors
    ///
    /// [`RenderError::MismatchedBufferSizes`] when the slice length differs from
    /// `num_vertices`.
    pub fn update_tbn_frames(
        &self,
        queue: &wgpu::Queue,
        vertex_frames: &[FaceFrame],
    ) -> Result<(), RenderError> {
        if vertex_frames.len() != self.num_vertices {
            return Err(RenderError::MismatchedBufferSizes {
                expected: self.num_vertices,
                actual: vertex_frames.len(),
            });
        }

        // Pack as [T, pad, B, pad, N, pad] (3 vec4s = 48 bytes per frame)
        let mut data = Vec::with_capacity(vertex_frames.len() * 12);
        for frame in vertex_frames {
            // Tangent
            data.extend_from_slice(&[frame.tangent.x, frame.tangent.y, frame.tangent.z, 0.0]);
            // Bitangent
            data.extend_from_slice(&[frame.bitangent.x, frame.bitangent.y, frame.bitangent.z, 0.0]);
            // Normal
            data.extend_from_slice(&[frame.normal.x, frame.normal.y, frame.normal.z, 0.0]);
        }
        if !data.is_empty() {
            queue.write_buffer(&self.tbn_frames, 0, bytemuck::cast_slice(&data));
        }
        Ok(())
    }

    /// Update position gradients from CPU data.
    ///
    /// # Arguments
    ///
    /// * `queue` - WGPU queue
    /// * `gradients` - Position gradients `[N × 3]`; must contain exactly
    ///   `num_gaussians` entries.
    ///
    /// # Errors
    ///
    /// [`RenderError::MismatchedBufferSizes`] when the slice length differs from
    /// `num_gaussians`.
    pub fn update_position_gradients(
        &self,
        queue: &wgpu::Queue,
        gradients: &[[f32; 3]],
    ) -> Result<(), RenderError> {
        if gradients.len() != self.num_gaussians {
            return Err(RenderError::MismatchedBufferSizes {
                expected: self.num_gaussians,
                actual: gradients.len(),
            });
        }

        // Pack as vec4 [grad, pad]
        let mut data = Vec::with_capacity(self.num_gaussians * 4);
        for grad in gradients {
            data.extend_from_slice(&[grad[0], grad[1], grad[2], 0.0]);
        }
        if !data.is_empty() {
            queue.write_buffer(&self.position_grads, 0, bytemuck::cast_slice(&data));
        }
        Ok(())
    }

    /// Copy position gradients straight from a GPU gradient buffer.
    ///
    /// `src` must use the padded `[f32; 4]`-per-Gaussian layout of
    /// `GradientBuffers::grad_positions` and carry `COPY_SRC` usage. This avoids
    /// a GPU→CPU→GPU round trip when chaining `preprocess_bwd` into
    /// `flame_binding_bwd`.
    ///
    /// # Errors
    ///
    /// [`RenderError::BufferOverflow`] when `src` is smaller than
    /// `num_gaussians × 16` bytes.
    pub fn copy_position_gradients_from(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        src: &wgpu::Buffer,
    ) -> Result<(), RenderError> {
        let required = (self.num_gaussians as u64) * 16;
        if required == 0 {
            return Ok(());
        }
        if src.size() < required {
            return Err(RenderError::BufferOverflow {
                buffer_name: "flame_position_grads".to_string(),
                max_size: src.size(),
                requested: required,
            });
        }
        encoder.copy_buffer_to_buffer(src, 0, &self.position_grads, 0, required);
        Ok(())
    }

    /// Record the `flame_binding_backward` dispatch into `encoder`.
    ///
    /// `pipeline` is the `flame_binding_bwd` compute pipeline (see
    /// `Pipelines::flame_binding_bwd`). The kernel writes every element of
    /// `offset_grads`, so no clear is needed beforehand.
    pub fn record_backward(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::ComputePipeline,
    ) {
        if self.num_gaussians == 0 {
            return;
        }
        let n = u32::try_from(self.num_gaussians).unwrap_or(u32::MAX);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("flame_binding_bwd"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(n.div_ceil(BINDING_BWD_WORKGROUP_SIZE), 1, 1);
    }

    /// Read `∂L/∂local_offset` back to the CPU.
    ///
    /// # Errors
    ///
    /// [`RenderError::BufferMapFailed`] when the staging buffer cannot be mapped.
    pub fn read_offset_gradients(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<Vec<[f32; 3]>, RenderError> {
        if self.num_gaussians == 0 {
            return Ok(Vec::new());
        }
        let byte_size = (self.num_gaussians as u64) * 16;

        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flame_offset_grads_staging"),
            size: byte_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("flame_offset_grads_readback"),
        });
        encoder.copy_buffer_to_buffer(&self.offset_grads, 0, &staging, 0, byte_size);
        queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).ok();
        });
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        rx.recv()
            .map_err(|e| RenderError::BufferMapFailed {
                buffer_name: "flame_offset_grads".to_string(),
                error: format!("Channel recv failed: {e}"),
            })?
            .map_err(|e| RenderError::BufferMapFailed {
                buffer_name: "flame_offset_grads".to_string(),
                error: e.to_string(),
            })?;

        let data = slice
            .get_mapped_range()
            .map_err(|e| RenderError::BufferMapFailed {
                buffer_name: "flame_offset_grads".to_string(),
                error: format!("Mapped range failed: {e}"),
            })?;
        let floats: &[f32] = bytemuck::cast_slice(&data);
        let grads: Vec<[f32; 3]> = floats
            .chunks_exact(4)
            .take(self.num_gaussians)
            .map(|c| [c[0], c[1], c[2]])
            .collect();
        drop(data);
        staging.unmap();

        Ok(grads)
    }

    /// Dispatch the backward pass and read the resulting offset gradients.
    ///
    /// Convenience wrapper around [`record_backward`](Self::record_backward) and
    /// [`read_offset_gradients`](Self::read_offset_gradients); it submits its own
    /// command buffer and blocks until the GPU is done.
    ///
    /// # Errors
    ///
    /// [`RenderError::BufferMapFailed`] when the readback fails.
    pub fn backward(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pipeline: &wgpu::ComputePipeline,
    ) -> Result<Vec<[f32; 3]>, RenderError> {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("flame_binding_backward"),
        });
        self.record_backward(&mut encoder, pipeline);
        queue.submit(std::iter::once(encoder.finish()));
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        self.read_offset_gradients(device, queue)
    }

    /// Get the bind group for rendering.
    pub fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One unit triangle in the z = 0 plane: normal = +z, edge0 = +x.
    fn unit_triangle_mesh() -> Mesh {
        Mesh::new(
            vec![
                na::Point3::new(0.0, 0.0, 0.0),
                na::Point3::new(1.0, 0.0, 0.0),
                na::Point3::new(0.0, 1.0, 0.0),
            ],
            vec![[0, 1, 2]],
        )
    }

    fn model_with_gaussians(n: usize) -> GaussianModel {
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
            sh_coeffs: vec![0.0; n * 3],
            sh_degree: 0,
            face_indices: Vec::new(),
            barycentric: Vec::new(),
            local_offsets: Vec::new(),
            is_rigid: Vec::new(),
        }
    }

    #[test]
    fn test_position_reg_zero_for_matching() {
        let positions = vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let binding = vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        assert!((position_regularization(&positions, &binding)).abs() < 1e-10);
    }

    #[test]
    fn test_scale_reg_zero_under_max() {
        // log-scales of 0.0 → exp(0) = 1.0, max_scale = 2.0 → no penalty
        let scales = vec![[0.0, 0.0, 0.0]];
        assert!((scale_regularization(&scales, 2.0)).abs() < 1e-10);
    }

    // --- Regression: apply_binding must not panic on missing binding data ---

    #[test]
    fn test_apply_binding_missing_arrays_is_error_not_panic() {
        let model = model_with_gaussians(3); // no face_indices/barycentric/offsets
        let mesh = unit_triangle_mesh();
        let err = apply_binding(&model, &mesh).expect_err("must reject empty binding arrays");
        assert!(matches!(
            err,
            RenderError::MismatchedBufferSizes {
                expected: 3,
                actual: 0
            }
        ));
    }

    #[test]
    fn test_apply_binding_empty_mesh_is_error_not_panic() {
        let mut model = model_with_gaussians(1);
        model.face_indices = vec![0];
        model.barycentric = vec![[1.0, 0.0, 0.0]];
        model.local_offsets = vec![[0.0, 0.0, 0.0]];
        let empty_mesh = Mesh::new(Vec::new(), Vec::new());
        let err = apply_binding(&model, &empty_mesh).expect_err("must reject an empty mesh");
        assert!(matches!(err, RenderError::ValidationError(_)));
    }

    #[test]
    fn test_apply_binding_out_of_range_face_is_error_not_clamped() {
        let mut model = model_with_gaussians(1);
        model.face_indices = vec![7]; // mesh has 1 face
        model.barycentric = vec![[1.0, 0.0, 0.0]];
        model.local_offsets = vec![[0.0, 0.0, 0.0]];
        let mesh = unit_triangle_mesh();
        let err = apply_binding(&model, &mesh).expect_err("must reject out-of-range face index");
        assert!(matches!(err, RenderError::InvalidGaussian { index: 0, .. }));
    }

    #[test]
    fn test_apply_binding_out_of_range_vertex_is_error() {
        let mut model = model_with_gaussians(1);
        model.face_indices = vec![0];
        model.barycentric = vec![[1.0, 0.0, 0.0]];
        model.local_offsets = vec![[0.0, 0.0, 0.0]];
        // Face references vertex 9 which does not exist. Built by struct literal
        // because `Mesh::new` itself indexes vertices while computing normals.
        let mesh = Mesh {
            vertices: vec![
                na::Point3::new(0.0, 0.0, 0.0),
                na::Point3::new(1.0, 0.0, 0.0),
            ],
            normals: vec![na::Vector3::zeros(); 2],
            faces: vec![[0, 1, 9]],
            uv_coords: Vec::new(),
        };
        let err = apply_binding(&model, &mesh).expect_err("must reject out-of-range vertex index");
        assert!(matches!(err, RenderError::InvalidGaussian { index: 0, .. }));
    }

    #[test]
    fn test_apply_binding_empty_model_is_ok() {
        let model = model_with_gaussians(0);
        let mesh = unit_triangle_mesh();
        let result = apply_binding(&model, &mesh).expect("empty model binds trivially");
        assert!(result.positions.is_empty());
        assert!(result.face_frames.is_empty());
    }

    #[test]
    fn test_apply_binding_offset_applied_in_face_frame() {
        let mut model = model_with_gaussians(1);
        model.face_indices = vec![0];
        model.barycentric = vec![[1.0, 0.0, 0.0]]; // v0
        model.local_offsets = vec![[0.0, 0.0, 2.0]]; // 2 along the normal (+z)
        let mesh = unit_triangle_mesh();
        let result = apply_binding(&model, &mesh).expect("binding");
        let p = result.positions[0];
        assert!((p[0]).abs() < 1e-6, "x = {}", p[0]);
        assert!((p[1]).abs() < 1e-6, "y = {}", p[1]);
        assert!((p[2] - 2.0).abs() < 1e-6, "z = {}", p[2]);
        // face_frames is per Gaussian, not per vertex.
        assert_eq!(result.face_frames.len(), 1);
    }

    // --- Regression: TBN frames must be per vertex ---

    #[test]
    fn test_build_vertex_frames_length_matches_vertex_count() {
        let mesh = unit_triangle_mesh();
        let frames = build_vertex_frames(&mesh);
        assert_eq!(frames.len(), mesh.vertices.len());
        for frame in &frames {
            // z-plane triangle → +z normal at every vertex.
            assert!((frame.normal.z - 1.0).abs() < 1e-5, "{:?}", frame.normal);
            // Orthonormality.
            assert!(frame.tangent.dot(&frame.normal).abs() < 1e-5);
            assert!((frame.tangent.norm() - 1.0).abs() < 1e-5);
            assert!((frame.bitangent.norm() - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_build_face_frames_matches_apply_binding_frames() {
        let mesh = unit_triangle_mesh();
        let face_frames = build_face_frames(&mesh);
        assert_eq!(face_frames.len(), mesh.faces.len());

        let mut model = model_with_gaussians(1);
        model.face_indices = vec![0];
        model.barycentric = vec![[0.5, 0.25, 0.25]];
        model.local_offsets = vec![[0.1, 0.2, 0.3]];
        let result = apply_binding(&model, &mesh).expect("binding");

        // The frame the forward pass used must be the face-indexed table entry.
        let forward = result.face_frames[0];
        let table = face_frames[0];
        assert!((forward.tangent - table.tangent).norm() < 1e-6);
        assert!((forward.bitangent - table.bitangent).norm() < 1e-6);
        assert!((forward.normal - table.normal).norm() < 1e-6);
    }

    #[test]
    fn test_build_face_frames_bad_face_gets_fallback() {
        let mesh = Mesh {
            vertices: vec![
                na::Point3::new(0.0, 0.0, 0.0),
                na::Point3::new(1.0, 0.0, 0.0),
            ],
            normals: vec![na::Vector3::zeros(); 2],
            faces: vec![[0, 1, 5]],
            uv_coords: Vec::new(),
        };
        let frames = build_face_frames(&mesh);
        assert_eq!(frames.len(), 1);
        assert!((frames[0].normal.z - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_build_vertex_frames_independent_of_gaussian_count() {
        // 4 vertices, 2 faces: the frame array must stay vertex-sized even
        // though a bound model could have any number of Gaussians.
        let mesh = Mesh::new(
            vec![
                na::Point3::new(0.0, 0.0, 0.0),
                na::Point3::new(1.0, 0.0, 0.0),
                na::Point3::new(0.0, 1.0, 0.0),
                na::Point3::new(1.0, 1.0, 0.0),
            ],
            vec![[0, 1, 2], [1, 3, 2]],
        );
        assert_eq!(build_vertex_frames(&mesh).len(), 4);
    }

    #[test]
    fn test_build_vertex_frames_isolated_vertex_gets_fallback() {
        // Vertex 3 belongs to no face → degenerate accumulators.
        let mesh = Mesh::new(
            vec![
                na::Point3::new(0.0, 0.0, 0.0),
                na::Point3::new(1.0, 0.0, 0.0),
                na::Point3::new(0.0, 1.0, 0.0),
                na::Point3::new(5.0, 5.0, 5.0),
            ],
            vec![[0, 1, 2]],
        );
        let frames = build_vertex_frames(&mesh);
        assert_eq!(frames.len(), 4);
        let isolated = frames[3];
        assert!((isolated.tangent.norm() - 1.0).abs() < 1e-5);
        assert!((isolated.normal.norm() - 1.0).abs() < 1e-5);
        assert!(isolated.tangent.dot(&isolated.normal).abs() < 1e-5);
    }

    #[test]
    fn test_build_vertex_frames_skips_faces_with_bad_indices() {
        let mesh = Mesh {
            vertices: vec![
                na::Point3::new(0.0, 0.0, 0.0),
                na::Point3::new(1.0, 0.0, 0.0),
                na::Point3::new(0.0, 1.0, 0.0),
            ],
            normals: vec![na::Vector3::zeros(); 3],
            faces: vec![[0, 1, 2], [0, 1, 42]],
            uv_coords: Vec::new(),
        };
        let frames = build_vertex_frames(&mesh);
        assert_eq!(frames.len(), 3);
        assert!((frames[0].normal.z - 1.0).abs() < 1e-5);
    }

    // --- Backward-pass math ---

    #[test]
    fn test_offset_gradients_cpu_projects_onto_vertex_frame() {
        let mesh = unit_triangle_mesh();
        let frames = build_vertex_frames(&mesh);
        // Two Gaussians bound to vertex 2 and vertex 0 respectively.
        let vertex_ids = vec![2u32, 0u32];
        let grads = vec![[0.0, 0.0, 1.0], [0.0, 0.0, -3.0]];
        let out = offset_gradients_cpu(&grads, &vertex_ids, &frames).expect("projection");
        assert_eq!(out.len(), 2);
        // Gradient purely along +z projects entirely onto the normal component.
        assert!((out[0][2] - 1.0).abs() < 1e-5, "{:?}", out[0]);
        assert!(out[0][0].abs() < 1e-5 && out[0][1].abs() < 1e-5);
        assert!((out[1][2] + 3.0).abs() < 1e-5, "{:?}", out[1]);
    }

    #[test]
    fn test_offset_gradients_cpu_rejects_length_mismatch() {
        let frames = build_vertex_frames(&unit_triangle_mesh());
        let err = offset_gradients_cpu(&[[1.0, 0.0, 0.0]], &[], &frames)
            .expect_err("length mismatch must error");
        assert!(matches!(err, RenderError::MismatchedBufferSizes { .. }));
    }

    #[test]
    fn test_offset_gradients_cpu_rejects_out_of_range_vertex() {
        let frames = build_vertex_frames(&unit_triangle_mesh());
        let err = offset_gradients_cpu(&[[1.0, 0.0, 0.0]], &[99], &frames)
            .expect_err("out-of-range vertex id must error");
        assert!(matches!(err, RenderError::InvalidGaussian { index: 0, .. }));
    }

    #[test]
    fn test_frame_from_degenerate_normal_falls_back() {
        let frame = FaceFrame::from_normal_and_tangent(na::Vector3::zeros(), na::Vector3::x());
        assert!((frame.normal.z - 1.0).abs() < 1e-6);
        assert!((frame.tangent.x - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_frame_from_parallel_tangent_hint_stays_orthonormal() {
        // Hint parallel to the normal is unusable; an arbitrary axis is chosen.
        let frame = FaceFrame::from_normal_and_tangent(
            na::Vector3::new(0.0, 0.0, 2.0),
            na::Vector3::new(0.0, 0.0, 5.0),
        );
        assert!((frame.normal.z - 1.0).abs() < 1e-6);
        assert!((frame.tangent.norm() - 1.0).abs() < 1e-6);
        assert!(frame.tangent.dot(&frame.normal).abs() < 1e-6);
        assert!((frame.bitangent.norm() - 1.0).abs() < 1e-6);
    }
}
