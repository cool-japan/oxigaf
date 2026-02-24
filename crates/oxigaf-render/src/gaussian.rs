//! 3D Gaussian model data structures.

use bytemuck::{Pod, Zeroable};

/// Attributes of a single 3D Gaussian primitive.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GaussianAttributes {
    /// Position (x, y, z).
    pub position: [f32; 3],
    pub _pad0: f32,
    /// Rotation quaternion (x, y, z, w).
    pub rotation: [f32; 4],
    /// Log-scale (sx, sy, sz) — exponentiated before use.
    pub scale: [f32; 3],
    /// Sigmoid-inverse opacity.
    pub opacity: f32,
}

/// A collection of 3D Gaussians that form an avatar.
#[derive(Debug, Clone)]
pub struct GaussianModel {
    /// Per-Gaussian attributes.
    pub gaussians: Vec<GaussianAttributes>,
    /// Spherical harmonics coefficients per Gaussian `[N, C]`
    /// where C = (sh_degree+1)² × 3.
    pub sh_coeffs: Vec<f32>,
    /// SH degree (0–3).
    pub sh_degree: u32,

    // --- FLAME binding ---
    /// Face index on the FLAME mesh for each Gaussian.
    pub face_indices: Vec<u32>,
    /// Barycentric coordinates on the bound face.
    pub barycentric: Vec<[f32; 3]>,
    /// Learnable local offset from the mesh surface.
    pub local_offsets: Vec<[f32; 3]>,
    /// Whether each Gaussian is rigid (true) or flexible (false).
    pub is_rigid: Vec<bool>,
}

impl GaussianModel {
    /// Number of Gaussians.
    pub fn len(&self) -> usize {
        self.gaussians.len()
    }

    /// Whether the model is empty.
    pub fn is_empty(&self) -> bool {
        self.gaussians.is_empty()
    }
}
