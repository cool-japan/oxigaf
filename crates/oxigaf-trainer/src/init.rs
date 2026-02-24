//! Gaussian initialization on the FLAME mesh surface.
//!
//! Samples points uniformly (area-weighted) on the FLAME mesh and creates an
//! initial [`GaussianModel`] with rigid / flexible split.

use rand::Rng;

use oxigaf_flame::{sample_mesh_surface, Mesh};
use oxigaf_render::gaussian::{GaussianAttributes, GaussianModel};

use crate::config::InitConfig;

// ---------------------------------------------------------------------------
// GaussianInitializer
// ---------------------------------------------------------------------------

/// Creates an initial [`GaussianModel`] by sampling surface points on a FLAME
/// mesh.
pub struct GaussianInitializer;

impl GaussianInitializer {
    /// Initialise Gaussians on `mesh`.
    ///
    /// * The first `config.num_rigid` samples are flagged **rigid** (move with
    ///   the head bone only).
    /// * The remaining `config.num_flexible` samples are **flexible**
    ///   (deformable via expression / jaw blendshapes).
    pub fn initialize(mesh: &Mesh, config: &InitConfig, rng: &mut impl Rng) -> GaussianModel {
        let total = config.num_rigid + config.num_flexible;
        let samples = sample_mesh_surface(mesh, total, rng);

        // SH coefficients per Gaussian: (degree+1)² bands × 3 RGB channels.
        let sh_channels = ((config.sh_degree + 1) * (config.sh_degree + 1) * 3) as usize;

        let mut gaussians = Vec::with_capacity(total);
        let mut sh_coeffs = Vec::with_capacity(total * sh_channels);
        let mut face_indices = Vec::with_capacity(total);
        let mut barycentric = Vec::with_capacity(total);
        let mut local_offsets = Vec::with_capacity(total);
        let mut is_rigid = Vec::with_capacity(total);

        // SH DC normalisation constant: 0.5 / Y_0^0 where Y_0^0 = 0.5*sqrt(1/π).
        let sh_c0: f32 = 0.282_094_8;
        let dc_value = 0.5 / sh_c0;

        for (i, sp) in samples.iter().enumerate() {
            // --- Gaussian attributes ---
            let attr = GaussianAttributes {
                position: [sp.position.x, sp.position.y, sp.position.z],
                _pad0: 0.0,
                rotation: [0.0, 0.0, 0.0, 1.0], // identity quaternion (xyzw)
                scale: [config.initial_scale; 3],
                opacity: config.initial_opacity,
            };
            gaussians.push(attr);

            // --- SH coefficients (DC = grey, higher bands = 0) ---
            for c in 0..sh_channels {
                if c < 3 {
                    sh_coeffs.push(dc_value);
                } else {
                    sh_coeffs.push(0.0);
                }
            }

            // --- FLAME binding info ---
            face_indices.push(sp.face_index);
            barycentric.push(sp.barycentric);
            local_offsets.push([0.0; 3]);
            is_rigid.push(i < config.num_rigid);
        }

        tracing::info!(
            "Initialised {} Gaussians ({} rigid, {} flexible), SH degree {}",
            total,
            config.num_rigid,
            config.num_flexible,
            config.sh_degree,
        );

        GaussianModel {
            gaussians,
            sh_coeffs,
            sh_degree: config.sh_degree,
            face_indices,
            barycentric,
            local_offsets,
            is_rigid,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra as na;
    use rand::SeedableRng;

    fn tiny_mesh() -> Mesh {
        Mesh::new(
            vec![
                na::Point3::new(0.0, 0.0, 0.0),
                na::Point3::new(1.0, 0.0, 0.0),
                na::Point3::new(0.0, 1.0, 0.0),
            ],
            vec![[0, 1, 2]],
        )
    }

    #[test]
    fn init_produces_correct_counts() {
        let mesh = tiny_mesh();
        let cfg = InitConfig {
            num_rigid: 10,
            num_flexible: 5,
            initial_scale: -5.0,
            initial_opacity: -2.0,
            sh_degree: 2,
        };
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let model = GaussianInitializer::initialize(&mesh, &cfg, &mut rng);

        assert_eq!(model.len(), 15);
        assert_eq!(model.is_rigid.iter().filter(|&&r| r).count(), 10);
        assert_eq!(model.is_rigid.iter().filter(|&&r| !r).count(), 5);

        let sh_per = ((cfg.sh_degree + 1) * (cfg.sh_degree + 1) * 3) as usize;
        assert_eq!(model.sh_coeffs.len(), 15 * sh_per);
    }
}
