//! Gaussian initialization on the FLAME mesh surface.
//!
//! Samples points uniformly (area-weighted) on the FLAME mesh and creates an
//! initial [`GaussianModel`] with rigid / flexible split.

use rand::Rng;

use oxigaf_flame::{sample_mesh_surface, DefaultSampler, Mesh, MeshSurfaceSampler};
use oxigaf_render::gaussian::{GaussianAttributes, GaussianModel};

use crate::config::InitConfig;
use crate::TrainerError;

/// One sampled surface binding: `(position, face_index, barycentric)`.
type SurfaceBinding = ([f32; 3], u32, [f32; 3]);

// ---------------------------------------------------------------------------
// GaussianInitializer
// ---------------------------------------------------------------------------

/// Creates an initial [`GaussianModel`] by sampling surface points on a FLAME
/// mesh.
pub struct GaussianInitializer;

impl GaussianInitializer {
    /// Initialise Gaussians on `mesh` using the built-in area-weighted
    /// sampler, driven by the caller's `rng`.
    ///
    /// * The first `config.num_rigid` samples are flagged **rigid** (move with
    ///   the head bone only).
    /// * The remaining `config.num_flexible` samples are **flexible**
    ///   (deformable via expression / jaw blendshapes).
    ///
    /// Requesting zero Gaussians (`num_rigid + num_flexible == 0`) is honoured
    /// literally and yields an empty model; it is the caller's explicit
    /// request, not a failure.
    ///
    /// # Errors
    ///
    /// * [`TrainerError::EmptyMesh`] if `mesh` has no faces.
    /// * [`TrainerError::Flame`] if `mesh` is malformed (its `normals` length
    ///   does not match `vertices`, or a face references an out-of-range
    ///   vertex). Previously such a mesh silently produced a zero-Gaussian
    ///   model while logging a full Gaussian count.
    /// * [`TrainerError::Init`] if the mesh is well-formed but fully
    ///   collapsed (near-zero total area), so no point can be placed on it.
    pub fn initialize(
        mesh: &Mesh,
        config: &InitConfig,
        rng: &mut impl Rng,
    ) -> Result<GaussianModel, TrainerError> {
        let total = Self::validate(mesh, config)?;
        let samples = sample_mesh_surface(mesh, total, rng)?;
        Self::finish(
            || {
                format!(
                    "the built-in area-weighted sampler; the mesh's {} faces cover a \
                     total area too small to sample",
                    mesh.faces.len()
                )
            },
            config,
            total,
            samples.len(),
            samples.into_iter().map(|sp| {
                (
                    [sp.position.x, sp.position.y, sp.position.z],
                    sp.face_index,
                    sp.barycentric,
                )
            }),
        )
    }

    /// Initialise Gaussians on `mesh` through any [`MeshSurfaceSampler`]
    /// implementation, making the sampling strategy pluggable.
    ///
    /// [`initialize`](Self::initialize) is the `rng`-driven convenience form
    /// that always uses [`DefaultSampler`]; this form lets a caller swap in a
    /// different strategy (e.g. curvature- or texture-weighted sampling) while
    /// keeping identical binding/SH setup. `seed` makes the sampler's output
    /// reproducible.
    ///
    /// # Errors
    ///
    /// Same conditions as [`initialize`](Self::initialize); any error reported
    /// by `sampler` is propagated as [`TrainerError::Flame`].
    pub fn initialize_with_sampler(
        mesh: &Mesh,
        config: &InitConfig,
        sampler: &dyn MeshSurfaceSampler,
        seed: u64,
    ) -> Result<GaussianModel, TrainerError> {
        let total = Self::validate(mesh, config)?;
        let sample = sampler.sample_surface(mesh, total, seed)?;

        // The trait contract is parallel arrays; a third-party implementation
        // could get that wrong, so check rather than zip-truncate silently.
        let n = sample.positions.len();
        if sample.face_indices.len() != n || sample.barycentric.len() != n {
            return Err(TrainerError::Init(format!(
                "MeshSurfaceSampler returned ragged output: {n} positions but {} face \
                 indices and {} barycentric coordinates",
                sample.face_indices.len(),
                sample.barycentric.len()
            )));
        }

        Self::finish(
            || "the supplied MeshSurfaceSampler".to_string(),
            config,
            total,
            n,
            sample
                .positions
                .into_iter()
                .zip(sample.face_indices)
                .zip(sample.barycentric)
                .map(|((position, face_index), bary)| (position, face_index, bary)),
        )
    }

    /// Initialise Gaussians with the default area-weighted sampler from a
    /// reproducible `seed` instead of a caller-held RNG.
    ///
    /// # Errors
    ///
    /// Same conditions as [`initialize`](Self::initialize).
    pub fn initialize_seeded(
        mesh: &Mesh,
        config: &InitConfig,
        seed: u64,
    ) -> Result<GaussianModel, TrainerError> {
        Self::initialize_with_sampler(mesh, config, &DefaultSampler, seed)
    }

    /// Reject inputs that cannot yield a usable model, returning the total
    /// number of Gaussians to place.
    fn validate(mesh: &Mesh, config: &InitConfig) -> Result<usize, TrainerError> {
        if mesh.faces.is_empty() {
            return Err(TrainerError::EmptyMesh);
        }
        Ok(config.num_rigid + config.num_flexible)
    }

    /// Turn sampled surface bindings into a [`GaussianModel`], after checking
    /// that sampling actually produced the requested number of points.
    ///
    /// `source` names (and, where the cause is actually known, explains) the
    /// sampler that fell short — the built-in sampler can only under-deliver
    /// on a zero-area mesh, but an arbitrary [`MeshSurfaceSampler`] may do so
    /// for reasons this function cannot diagnose, so it must not assert one.
    fn finish(
        source: impl FnOnce() -> String,
        config: &InitConfig,
        total: usize,
        produced: usize,
        bindings: impl Iterator<Item = SurfaceBinding>,
    ) -> Result<GaussianModel, TrainerError> {
        if produced != total {
            return Err(TrainerError::Init(format!(
                "surface sampling produced {produced} points for {total} requested \
                 Gaussians, via {}",
                source()
            )));
        }

        let model = Self::build_model(bindings, config, total);

        let rigid = model.is_rigid.iter().filter(|&&r| r).count();
        tracing::info!(
            "Initialised {} Gaussians ({} rigid, {} flexible), SH degree {}",
            model.len(),
            rigid,
            model.len() - rigid,
            config.sh_degree,
        );

        Ok(model)
    }

    /// Build the [`GaussianModel`] from `total` surface bindings.
    fn build_model(
        bindings: impl Iterator<Item = SurfaceBinding>,
        config: &InitConfig,
        total: usize,
    ) -> GaussianModel {
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

        for (i, (position, face_index, bary)) in bindings.enumerate() {
            // --- Gaussian attributes ---
            gaussians.push(GaussianAttributes {
                position,
                _pad0: 0.0,
                rotation: [0.0, 0.0, 0.0, 1.0], // identity quaternion (xyzw)
                scale: [config.initial_scale; 3],
                opacity: config.initial_opacity,
            });

            // --- SH coefficients (DC = grey, higher bands = 0) ---
            for c in 0..sh_channels {
                if c < 3 {
                    sh_coeffs.push(dc_value);
                } else {
                    sh_coeffs.push(0.0);
                }
            }

            // --- FLAME binding info ---
            face_indices.push(face_index);
            barycentric.push(bary);
            local_offsets.push([0.0; 3]);
            is_rigid.push(i < config.num_rigid);
        }

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

    fn tiny_config() -> InitConfig {
        InitConfig {
            num_rigid: 10,
            num_flexible: 5,
            initial_scale: -5.0,
            initial_opacity: -2.0,
            sh_degree: 2,
        }
    }

    #[test]
    fn init_produces_correct_counts() {
        let mesh = tiny_mesh();
        let cfg = tiny_config();
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let model = GaussianInitializer::initialize(&mesh, &cfg, &mut rng)
            .expect("a mesh built by Mesh::new is well-formed");

        assert_eq!(model.len(), 15);
        assert_eq!(model.is_rigid.iter().filter(|&&r| r).count(), 10);
        assert_eq!(model.is_rigid.iter().filter(|&&r| !r).count(), 5);

        let sh_per = ((cfg.sh_degree + 1) * (cfg.sh_degree + 1) * 3) as usize;
        assert_eq!(model.sh_coeffs.len(), 15 * sh_per);
    }

    // -----------------------------------------------------------------------
    // Regression: `sample_mesh_surface` used to swallow a malformed mesh and
    // return an empty `Vec`. `initialize` then built a zero-Gaussian model
    // *and logged the full requested count* ("Initialised 15 Gaussians"),
    // hiding the failure until training later failed on an empty model.
    // -----------------------------------------------------------------------

    #[test]
    fn init_rejects_malformed_mesh_instead_of_returning_empty_model() {
        // `Mesh`'s fields are public, so this bypasses `Mesh::new`: three
        // vertices, no normals, so no per-sample normal can be interpolated.
        let mesh = Mesh {
            vertices: vec![
                na::Point3::new(0.0, 0.0, 0.0),
                na::Point3::new(1.0, 0.0, 0.0),
                na::Point3::new(0.0, 1.0, 0.0),
            ],
            normals: Vec::new(),
            faces: vec![[0, 1, 2]],
            uv_coords: Vec::new(),
        };
        let cfg = tiny_config();
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);

        match GaussianInitializer::initialize(&mesh, &cfg, &mut rng) {
            Err(TrainerError::Flame(e)) => {
                assert!(
                    e.to_string().contains("normals"),
                    "error should name the malformed field, got: {e}"
                );
            }
            Err(other) => panic!("expected TrainerError::Flame, got {other:?}"),
            Ok(model) => panic!(
                "a malformed mesh must be an error, but initialization \
                 succeeded with {} Gaussians",
                model.len()
            ),
        }
    }

    #[test]
    fn init_rejects_out_of_range_face_index() {
        let mesh = Mesh {
            vertices: vec![
                na::Point3::new(0.0, 0.0, 0.0),
                na::Point3::new(1.0, 0.0, 0.0),
            ],
            normals: vec![na::Vector3::new(0.0, 0.0, 1.0); 2],
            faces: vec![[0, 1, 7]],
            uv_coords: Vec::new(),
        };
        let cfg = tiny_config();
        let mut rng = rand::rngs::StdRng::seed_from_u64(2);
        assert!(matches!(
            GaussianInitializer::initialize(&mesh, &cfg, &mut rng),
            Err(TrainerError::Flame(_))
        ));
    }

    #[test]
    fn init_rejects_face_less_mesh() {
        let mesh = Mesh::new(vec![], vec![]);
        let cfg = tiny_config();
        let mut rng = rand::rngs::StdRng::seed_from_u64(3);
        assert!(matches!(
            GaussianInitializer::initialize(&mesh, &cfg, &mut rng),
            Err(TrainerError::EmptyMesh)
        ));
    }

    #[test]
    fn init_rejects_fully_collapsed_mesh() {
        // Well-formed topology, but every vertex is coincident: total area is
        // zero, so no Gaussian can be placed. That must be an error, not a
        // silently empty model.
        let mesh = Mesh::new(vec![na::Point3::origin(); 3], vec![[0, 1, 2]]);
        let cfg = tiny_config();
        let mut rng = rand::rngs::StdRng::seed_from_u64(4);
        match GaussianInitializer::initialize(&mesh, &cfg, &mut rng) {
            Err(TrainerError::Init(msg)) => assert!(
                msg.contains("produced 0 points"),
                "error should report the shortfall, got: {msg}"
            ),
            other => panic!("expected TrainerError::Init, got {other:?}"),
        }
    }

    #[test]
    fn init_zero_requested_gaussians_is_an_empty_model_not_an_error() {
        let mesh = tiny_mesh();
        let cfg = InitConfig {
            num_rigid: 0,
            num_flexible: 0,
            ..tiny_config()
        };
        let mut rng = rand::rngs::StdRng::seed_from_u64(5);
        let model = GaussianInitializer::initialize(&mesh, &cfg, &mut rng)
            .expect("requesting zero Gaussians is honoured literally");
        assert_eq!(model.len(), 0);
    }

    // -----------------------------------------------------------------------
    // `MeshSurfaceSampler` is now a real seam (F240): `initialize_with_sampler`
    // routes through the trait, so the trait's doc claim is literally true.
    // -----------------------------------------------------------------------

    #[test]
    fn initialize_with_default_sampler_matches_seeded_helper() {
        let mesh = tiny_mesh();
        let cfg = tiny_config();

        let via_trait =
            GaussianInitializer::initialize_with_sampler(&mesh, &cfg, &DefaultSampler, 77)
                .expect("well-formed mesh");
        let via_helper =
            GaussianInitializer::initialize_seeded(&mesh, &cfg, 77).expect("well-formed mesh");

        assert_eq!(via_trait.len(), 15);
        assert_eq!(via_trait.face_indices, via_helper.face_indices);
        assert_eq!(via_trait.barycentric, via_helper.barycentric);
        assert_eq!(via_trait.is_rigid, via_helper.is_rigid);
        assert_eq!(via_trait.sh_coeffs, via_helper.sh_coeffs);
    }

    #[test]
    fn initialize_with_sampler_accepts_a_custom_strategy() {
        use oxigaf_flame::{FlameError, SurfaceSample};

        /// Places every Gaussian at the first vertex of face 0.
        struct FirstVertexSampler;

        impl MeshSurfaceSampler for FirstVertexSampler {
            fn sample_surface(
                &self,
                mesh: &Mesh,
                n: usize,
                _seed: u64,
            ) -> Result<SurfaceSample, FlameError> {
                let v = mesh
                    .vertices
                    .first()
                    .ok_or_else(|| FlameError::InvalidParams("no vertices".to_string()))?;
                Ok(SurfaceSample {
                    positions: vec![[v.x, v.y, v.z]; n],
                    normals: vec![[0.0, 0.0, 1.0]; n],
                    face_indices: vec![0; n],
                    barycentric: vec![[1.0, 0.0, 0.0]; n],
                })
            }
        }

        let mesh = tiny_mesh();
        let cfg = tiny_config();
        let model =
            GaussianInitializer::initialize_with_sampler(&mesh, &cfg, &FirstVertexSampler, 0)
                .expect("custom sampler should be honoured");

        assert_eq!(model.len(), 15);
        assert!(
            model
                .gaussians
                .iter()
                .all(|g| g.position == [0.0, 0.0, 0.0]),
            "the custom sampler's positions must actually be used"
        );
        assert!(model.barycentric.iter().all(|b| *b == [1.0, 0.0, 0.0]));
    }

    #[test]
    fn initialize_with_sampler_shortfall_blames_the_sampler_not_the_mesh() {
        use oxigaf_flame::{FlameError, SurfaceSample};

        /// Returns fewer points than requested, for reasons the initializer
        /// cannot know. The error must not invent a mesh-area explanation.
        struct ShortSampler;

        impl MeshSurfaceSampler for ShortSampler {
            fn sample_surface(
                &self,
                _mesh: &Mesh,
                n: usize,
                _seed: u64,
            ) -> Result<SurfaceSample, FlameError> {
                let k = n / 2;
                Ok(SurfaceSample {
                    positions: vec![[0.0; 3]; k],
                    normals: vec![[0.0, 0.0, 1.0]; k],
                    face_indices: vec![0; k],
                    barycentric: vec![[1.0, 0.0, 0.0]; k],
                })
            }
        }

        // A perfectly healthy, non-degenerate mesh: any "too small an area"
        // claim would be a fabricated diagnosis.
        let mesh = tiny_mesh();
        let cfg = tiny_config();
        match GaussianInitializer::initialize_with_sampler(&mesh, &cfg, &ShortSampler, 0) {
            Err(TrainerError::Init(msg)) => {
                assert!(
                    msg.contains("produced 7 points for 15"),
                    "error should report the shortfall, got: {msg}"
                );
                assert!(
                    msg.contains("MeshSurfaceSampler"),
                    "error should name the sampler as the source, got: {msg}"
                );
                assert!(
                    !msg.contains("area"),
                    "the initializer cannot know why a third-party sampler fell short, \
                     so it must not blame the mesh area: {msg}"
                );
            }
            other => panic!("expected TrainerError::Init, got {other:?}"),
        }
    }

    #[test]
    fn initialize_with_sampler_rejects_ragged_sampler_output() {
        use oxigaf_flame::{FlameError, SurfaceSample};

        /// Violates the trait's parallel-array contract.
        struct RaggedSampler;

        impl MeshSurfaceSampler for RaggedSampler {
            fn sample_surface(
                &self,
                _mesh: &Mesh,
                n: usize,
                _seed: u64,
            ) -> Result<SurfaceSample, FlameError> {
                Ok(SurfaceSample {
                    positions: vec![[0.0; 3]; n],
                    normals: vec![[0.0, 0.0, 1.0]; n],
                    face_indices: vec![0; n.saturating_sub(1)],
                    barycentric: vec![[1.0, 0.0, 0.0]; n],
                })
            }
        }

        let mesh = tiny_mesh();
        let cfg = tiny_config();
        match GaussianInitializer::initialize_with_sampler(&mesh, &cfg, &RaggedSampler, 0) {
            Err(TrainerError::Init(msg)) => assert!(
                msg.contains("ragged"),
                "error should call out the contract violation, got: {msg}"
            ),
            other => panic!("expected TrainerError::Init, got {other:?}"),
        }
    }
}
