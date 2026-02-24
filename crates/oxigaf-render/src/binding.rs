//! FLAME mesh binding for Gaussians.
//!
//! Each Gaussian is bound to a face on the FLAME mesh via barycentric
//! coordinates and a learnable local offset. This module computes the
//! world-space Gaussian positions from the current FLAME mesh pose and
//! the binding parameters.

use nalgebra as na;
use oxigaf_flame::Mesh;

use crate::gaussian::GaussianModel;

/// Result of applying FLAME mesh binding to a Gaussian model.
#[derive(Debug)]
pub struct BindingResult {
    /// Updated world-space positions for each Gaussian.
    pub positions: Vec<[f32; 3]>,
    /// Face-local coordinate frame per Gaussian (for orientation).
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

/// Compute the binding positions for all Gaussians given the current FLAME mesh.
///
/// For each Gaussian:
/// 1. Look up its bound face on the mesh.
/// 2. Interpolate the position using barycentric coordinates.
/// 3. Compute the face's local frame (tangent, bitangent, normal).
/// 4. Apply the local offset in the face's local frame.
///
/// Rigid Gaussians have `local_offsets = [0, 0, 0]`.
pub fn apply_binding(model: &GaussianModel, mesh: &Mesh) -> BindingResult {
    let n = model.len();
    let mut positions = Vec::with_capacity(n);
    let mut face_frames = Vec::with_capacity(n);

    for i in 0..n {
        let face_idx = model.face_indices[i] as usize;
        let bary = model.barycentric[i];
        let offset = model.local_offsets[i];

        // Get face vertices
        let face = &mesh.faces[face_idx.min(mesh.faces.len().saturating_sub(1))];
        let v0 = &mesh.vertices[face[0] as usize];
        let v1 = &mesh.vertices[face[1] as usize];
        let v2 = &mesh.vertices[face[2] as usize];

        // Interpolate position
        let base_pos = v0.coords * bary[0] + v1.coords * bary[1] + v2.coords * bary[2];

        // Compute face frame
        let edge0 = v1 - v0;
        let edge1 = v2 - v0;
        let face_normal = edge0.cross(&edge1);
        let normal_len = face_normal.norm();

        let frame = if normal_len > 1e-10 {
            let normal = face_normal / normal_len;
            let tangent = edge0.normalize();
            let bitangent = normal.cross(&tangent);
            FaceFrame {
                tangent,
                bitangent,
                normal,
            }
        } else {
            FaceFrame {
                tangent: na::Vector3::x(),
                bitangent: na::Vector3::y(),
                normal: na::Vector3::z(),
            }
        };

        // Apply local offset in face frame
        let world_offset =
            frame.tangent * offset[0] + frame.bitangent * offset[1] + frame.normal * offset[2];

        let pos = base_pos + world_offset;
        positions.push([pos.x, pos.y, pos.z]);
        face_frames.push(frame);
    }

    BindingResult {
        positions,
        face_frames,
    }
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

/// GPU buffers for FLAME binding backward pass.
///
/// These buffers are used to compute gradients w.r.t. local offsets
/// given gradients w.r.t. Gaussian positions.
pub struct FlameBindingBuffers {
    /// Binding info buffer (vertex_id per Gaussian).
    pub binding_info: wgpu::Buffer,
    /// TBN frames buffer (tangent, bitangent, normal per vertex).
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

        // Binding info buffer: [vertex_id, pad, pad, pad] per Gaussian
        let binding_info_size = (num_gaussians * 16) as u64; // 4 u32s = 16 bytes
        let binding_info = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flame_binding_info"),
            size: binding_info_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // TBN frames buffer: [T, pad, B, pad, N, pad] per vertex
        let tbn_frame_size = (num_vertices * 48) as u64; // 3 vec4s = 48 bytes
        let tbn_frames = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flame_tbn_frames"),
            size: tbn_frame_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Position gradients buffer: [grad, pad] per Gaussian
        let position_grads_size = (num_gaussians * 16) as u64; // vec4 = 16 bytes
        let position_grads = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flame_position_grads"),
            size: position_grads_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Offset gradients buffer: [grad, pad] per Gaussian
        let offset_grads_size = (num_gaussians * 16) as u64;
        let offset_grads = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flame_offset_grads"),
            size: offset_grads_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Uniforms buffer
        let uniforms_data = [num_gaussians as u32, 0, 0, 0]; // [num_gaussians, pad, pad, pad]
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

    /// Update binding info (vertex IDs per Gaussian).
    ///
    /// # Arguments
    ///
    /// * `queue` - WGPU queue
    /// * `vertex_ids` - Vertex ID for each Gaussian
    pub fn update_binding_info(&self, queue: &wgpu::Queue, vertex_ids: &[u32]) {
        // Pack as [vertex_id, pad, pad, pad]
        let mut data = Vec::with_capacity(self.num_gaussians * 4);
        for &vid in vertex_ids.iter().take(self.num_gaussians) {
            data.extend_from_slice(&[vid, 0, 0, 0]);
        }
        queue.write_buffer(&self.binding_info, 0, bytemuck::cast_slice(&data));
    }

    /// Update TBN frames from binding result.
    ///
    /// # Arguments
    ///
    /// * `queue` - WGPU queue
    /// * `face_frames` - Face frames from binding computation
    pub fn update_tbn_frames(&self, queue: &wgpu::Queue, face_frames: &[FaceFrame]) {
        // Pack as [T, pad, B, pad, N, pad] (3 vec4s = 48 bytes per frame)
        let mut data = Vec::with_capacity(face_frames.len() * 12);
        for frame in face_frames.iter().take(self.num_vertices) {
            // Tangent
            data.extend_from_slice(&[frame.tangent.x, frame.tangent.y, frame.tangent.z, 0.0]);
            // Bitangent
            data.extend_from_slice(&[frame.bitangent.x, frame.bitangent.y, frame.bitangent.z, 0.0]);
            // Normal
            data.extend_from_slice(&[frame.normal.x, frame.normal.y, frame.normal.z, 0.0]);
        }
        queue.write_buffer(&self.tbn_frames, 0, bytemuck::cast_slice(&data));
    }

    /// Update position gradients.
    ///
    /// # Arguments
    ///
    /// * `queue` - WGPU queue
    /// * `gradients` - Position gradients [N × 3]
    pub fn update_position_gradients(&self, queue: &wgpu::Queue, gradients: &[[f32; 3]]) {
        // Pack as vec4 [grad, pad]
        let mut data = Vec::with_capacity(self.num_gaussians * 4);
        for grad in gradients.iter().take(self.num_gaussians) {
            data.extend_from_slice(&[grad[0], grad[1], grad[2], 0.0]);
        }
        queue.write_buffer(&self.position_grads, 0, bytemuck::cast_slice(&data));
    }

    /// Get the bind group for rendering.
    pub fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
