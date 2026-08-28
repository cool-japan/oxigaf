//! Integration traits for FLAME components.
//!
//! These traits provide abstraction boundaries that allow testing and
//! interchangeability of normal map generation and surface sampling.

use crate::model::FlameModel;
use crate::normal_map::{Camera, NormalMapRenderer};
use crate::{FlameError, FlameParams, Mesh};
use nalgebra as na;
use rand::SeedableRng;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// NormalMapProvider
// ---------------------------------------------------------------------------

/// Trait for any type that can provide normal maps from FLAME parameters.
///
/// Implemented by concrete renderers but also mockable for testing.
/// All implementations must be `Send + Sync` for use across threads.
pub trait NormalMapProvider: Send + Sync {
    /// Generate a normal map (RGB u8 image) from FLAME parameters.
    ///
    /// Returns flat RGB bytes in row-major order: `width × height × 3` bytes.
    /// Each pixel encodes a surface normal as `RGB = (normal + 1) / 2 * 255`.
    ///
    /// # Errors
    ///
    /// Returns [`FlameError`] if normal map generation fails.
    fn generate_normal_map(
        &self,
        params: &FlameParams,
        width: u32,
        height: u32,
    ) -> Result<Vec<u8>, FlameError>;

    /// Generate multiple normal maps from the same parameters but different camera poses.
    ///
    /// `camera_poses` are 4×4 view matrices (one per view) in row-major order.
    ///
    /// The default implementation cannot honour distinct camera poses (it has
    /// no camera model to apply them with), so it only handles the
    /// unambiguous cases: zero poses yields zero images, and exactly one pose
    /// delegates to [`generate_normal_map`] once (there is nothing for a
    /// single view to disagree with). For two or more poses it returns
    /// [`FlameError::InvalidParams`] rather than silently returning N
    /// identical images — implementors that support camera transformations
    /// (e.g. [`FlameNormalMapProvider`]) must override this method.
    ///
    /// # Errors
    ///
    /// Returns [`FlameError::InvalidParams`] if `camera_poses.len() > 1`.
    /// Otherwise returns the first error encountered during generation.
    ///
    /// [`generate_normal_map`]: NormalMapProvider::generate_normal_map
    fn generate_normal_maps_multi_view(
        &self,
        params: &FlameParams,
        camera_poses: &[[[f32; 4]; 4]],
        width: u32,
        height: u32,
    ) -> Result<Vec<Vec<u8>>, FlameError> {
        match camera_poses.len() {
            0 => Ok(Vec::new()),
            1 => Ok(vec![self.generate_normal_map(params, width, height)?]),
            n => Err(FlameError::InvalidParams(format!(
                "generate_normal_maps_multi_view: the default implementation ignores \
                 camera poses and cannot honour {n} distinct views; the provider must \
                 override this method to support multi-view generation"
            ))),
        }
    }

    /// Image dimensions this provider outputs by default.
    ///
    /// Returns `(width, height)` in pixels.
    fn default_resolution(&self) -> (u32, u32) {
        (512, 512)
    }
}

// ---------------------------------------------------------------------------
// SurfaceSample
// ---------------------------------------------------------------------------

/// Output of surface sampling: positions, normals, face indices, and barycentric coords.
#[derive(Debug, Clone)]
pub struct SurfaceSample {
    /// World-space positions of sampled points (each as `[x, y, z]`).
    pub positions: Vec<[f32; 3]>,
    /// Interpolated unit normals at sampled points (each as `[nx, ny, nz]`).
    pub normals: Vec<[f32; 3]>,
    /// Face index for each sampled point.
    pub face_indices: Vec<u32>,
    /// Barycentric coordinates `[u, v, w]` with `u + v + w ≈ 1.0`.
    pub barycentric: Vec<[f32; 3]>,
}

// ---------------------------------------------------------------------------
// MeshSurfaceSampler
// ---------------------------------------------------------------------------

/// Trait for sampling surface points from a FLAME mesh.
///
/// This is the pluggable seam for Gaussian initialization: the trainer's
/// `GaussianInitializer::initialize_with_sampler` accepts any
/// `&dyn MeshSurfaceSampler`, so an alternative sampling strategy can be
/// substituted without touching the initializer. [`DefaultSampler`] is the
/// area-weighted implementation used by default.
pub trait MeshSurfaceSampler: Send + Sync {
    /// Sample `n` surface points from the mesh.
    ///
    /// Uses area-weighted random sampling so that denser regions are sampled
    /// proportionally. The `seed` parameter makes results reproducible.
    ///
    /// # Errors
    ///
    /// Returns [`FlameError`] if the mesh has no faces or sampling fails.
    fn sample_surface(&self, mesh: &Mesh, n: usize, seed: u64)
        -> Result<SurfaceSample, FlameError>;
}

// ---------------------------------------------------------------------------
// DefaultSampler
// ---------------------------------------------------------------------------

/// Default implementation of [`MeshSurfaceSampler`] using area-weighted sampling.
///
/// Delegates to [`crate::sampler::sample_mesh_surface`], propagating its
/// malformed-mesh errors (mismatched `normals` length, out-of-range face
/// indices) instead of reporting them as an empty sample set.
pub struct DefaultSampler;

impl MeshSurfaceSampler for DefaultSampler {
    fn sample_surface(
        &self,
        mesh: &Mesh,
        n: usize,
        seed: u64,
    ) -> Result<SurfaceSample, FlameError> {
        // Validate before honouring `n == 0`: a caller probing a mesh with a
        // zero-sample request must still learn that the mesh is unusable,
        // and this keeps the trait's contract identical to
        // `sample_mesh_surface`'s regardless of `n`. (An `n == 0` fast path
        // here would silently accept a malformed mesh that every non-zero
        // `n` rejects.)
        if mesh.faces.is_empty() {
            return Err(FlameError::InvalidParams(
                "cannot sample from a mesh with no faces".to_string(),
            ));
        }

        // `sample_mesh_surface` performs the remaining topology validation
        // and itself returns no points when `n == 0`.
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let points = crate::sampler::sample_mesh_surface(mesh, n, &mut rng)?;

        let num = points.len();
        let mut positions = Vec::with_capacity(num);
        let mut normals = Vec::with_capacity(num);
        let mut face_indices = Vec::with_capacity(num);
        let mut barycentric = Vec::with_capacity(num);

        for pt in points {
            positions.push([pt.position.x, pt.position.y, pt.position.z]);
            normals.push([pt.normal.x, pt.normal.y, pt.normal.z]);
            face_indices.push(pt.face_index);
            barycentric.push(pt.barycentric);
        }

        Ok(SurfaceSample {
            positions,
            normals,
            face_indices,
            barycentric,
        })
    }
}

// ---------------------------------------------------------------------------
// FlameNormalMapProvider
// ---------------------------------------------------------------------------

/// Production [`NormalMapProvider`] backed by a loaded [`FlameModel`] and the
/// CPU [`NormalMapRenderer`] rasterizer.
///
/// [`Self::generate_normal_map`] runs the FLAME forward pass on `params` and
/// rasterizes the resulting mesh from a default front-facing camera.
/// [`Self::generate_normal_maps_multi_view`] evaluates the mesh once and
/// re-renders it from each supplied camera pose, so — unlike the trait's
/// default implementation — the output genuinely varies per view.
#[derive(Clone)]
pub struct FlameNormalMapProvider {
    model: Arc<FlameModel>,
}

impl FlameNormalMapProvider {
    /// Wrap a loaded FLAME model for normal-map generation.
    #[must_use]
    pub fn new(model: Arc<FlameModel>) -> Self {
        Self { model }
    }

    /// Build a [`Camera`] for `width`×`height` from a row-major 4×4
    /// world-to-camera view matrix: the upper-left 3×3 block is the
    /// rotation and column 3 (rows 0..3) is the translation, matching the
    /// convention documented on
    /// [`NormalMapProvider::generate_normal_maps_multi_view`]. Intrinsics
    /// (focal length, principal point, near/far planes) come from
    /// [`Camera::default_front`].
    fn camera_from_pose(pose: &[[f32; 4]; 4], width: u32, height: u32) -> Camera {
        let rotation = na::Matrix3::new(
            pose[0][0], pose[0][1], pose[0][2], pose[1][0], pose[1][1], pose[1][2], pose[2][0],
            pose[2][1], pose[2][2],
        );
        let translation = na::Vector3::new(pose[0][3], pose[1][3], pose[2][3]);
        let mut camera = Camera::default_front(width, height);
        camera.rotation = rotation;
        camera.translation = translation;
        camera
    }
}

impl NormalMapProvider for FlameNormalMapProvider {
    fn generate_normal_map(
        &self,
        params: &FlameParams,
        width: u32,
        height: u32,
    ) -> Result<Vec<u8>, FlameError> {
        if width == 0 || height == 0 {
            return Err(FlameError::InvalidParams(format!(
                "generate_normal_map: width and height must be > 0; got {width}x{height}"
            )));
        }
        let mesh = self.model.forward(params);
        let camera = Camera::default_front(width, height);
        Ok(NormalMapRenderer::render(&mesh, &camera).into_raw())
    }

    fn generate_normal_maps_multi_view(
        &self,
        params: &FlameParams,
        camera_poses: &[[[f32; 4]; 4]],
        width: u32,
        height: u32,
    ) -> Result<Vec<Vec<u8>>, FlameError> {
        if width == 0 || height == 0 {
            return Err(FlameError::InvalidParams(format!(
                "generate_normal_maps_multi_view: width and height must be > 0; got \
                 {width}x{height}"
            )));
        }
        // Evaluate the FLAME forward pass once and re-render it from every
        // requested pose, so the output genuinely differs per view (unlike
        // the trait's pose-ignoring default).
        let mesh = self.model.forward(params);
        Ok(camera_poses
            .iter()
            .map(|pose| {
                let camera = Self::camera_from_pose(pose, width, height);
                NormalMapRenderer::render(&mesh, &camera).into_raw()
            })
            .collect())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra as na;

    // -----------------------------------------------------------------------
    // Test helpers
    // -----------------------------------------------------------------------

    /// A simple flat triangle mesh suitable for sampling tests.
    ///
    /// Three vertices forming a unit right triangle in the XY plane.
    fn unit_triangle_mesh() -> Mesh {
        Mesh::new(
            vec![
                na::Point3::new(0.0f32, 0.0, 0.0),
                na::Point3::new(1.0f32, 0.0, 0.0),
                na::Point3::new(0.0f32, 1.0, 0.0),
            ],
            vec![[0, 1, 2]],
        )
    }

    /// A simple quad mesh (two triangles) for more comprehensive tests.
    fn quad_mesh() -> Mesh {
        Mesh::new(
            vec![
                na::Point3::new(0.0f32, 0.0, 0.0),
                na::Point3::new(1.0f32, 0.0, 0.0),
                na::Point3::new(0.0f32, 1.0, 0.0),
                na::Point3::new(1.0f32, 1.0, 0.0),
            ],
            vec![[0, 1, 2], [1, 3, 2]],
        )
    }

    // -----------------------------------------------------------------------
    // Mock NormalMapProvider for trait compilation test
    // -----------------------------------------------------------------------

    struct MockNormalMapProvider {
        width: u32,
        height: u32,
    }

    impl NormalMapProvider for MockNormalMapProvider {
        fn generate_normal_map(
            &self,
            _params: &FlameParams,
            width: u32,
            height: u32,
        ) -> Result<Vec<u8>, FlameError> {
            // Return a flat blue normal map (pointing towards the viewer in Z)
            // Blue channel = 255, R = G = 128 (neutral direction)
            let n = (width * height * 3) as usize;
            let mut buf = vec![128u8; n];
            // Set Z channel (every third byte starting at index 2) to 255
            let mut i = 2;
            while i < n {
                buf[i] = 255;
                i += 3;
            }
            Ok(buf)
        }

        fn default_resolution(&self) -> (u32, u32) {
            (self.width, self.height)
        }
    }

    // -----------------------------------------------------------------------
    // DefaultSampler tests
    // -----------------------------------------------------------------------

    #[test]
    fn default_sampler_produces_correct_count() {
        let mesh = unit_triangle_mesh();
        let sampler = DefaultSampler;
        let result = sampler
            .sample_surface(&mesh, 50, 42)
            .expect("sampling failed");
        assert_eq!(
            result.positions.len(),
            50,
            "should produce exactly 50 samples"
        );
        assert_eq!(result.normals.len(), 50);
        assert_eq!(result.face_indices.len(), 50);
        assert_eq!(result.barycentric.len(), 50);
    }

    #[test]
    fn default_sampler_positions_within_bounding_box() {
        let mesh = quad_mesh();
        let sampler = DefaultSampler;
        let result = sampler
            .sample_surface(&mesh, 200, 7)
            .expect("sampling failed");

        for (i, pos) in result.positions.iter().enumerate() {
            // All vertices are in [0,1] × [0,1] × {0} so sampled points must be too
            assert!(
                pos[0] >= -1e-5 && pos[0] <= 1.0 + 1e-5,
                "sample {i}: x={} out of [0,1] range",
                pos[0]
            );
            assert!(
                pos[1] >= -1e-5 && pos[1] <= 1.0 + 1e-5,
                "sample {i}: y={} out of [0,1] range",
                pos[1]
            );
            assert!(
                pos[2].abs() < 1e-5,
                "sample {i}: z={} should be ~0 (planar mesh)",
                pos[2]
            );
        }
    }

    #[test]
    fn default_sampler_normals_are_unit_vectors() {
        let mesh = quad_mesh();
        let sampler = DefaultSampler;
        let result = sampler
            .sample_surface(&mesh, 100, 99)
            .expect("sampling failed");

        for (i, normal) in result.normals.iter().enumerate() {
            let len_sq = normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2];
            assert!(
                (len_sq - 1.0).abs() < 1e-4,
                "sample {i}: normal magnitude squared = {len_sq} (expected ~1)"
            );
        }
    }

    #[test]
    fn default_sampler_barycentric_coords_sum_to_one() {
        let mesh = unit_triangle_mesh();
        let sampler = DefaultSampler;
        let result = sampler
            .sample_surface(&mesh, 150, 123)
            .expect("sampling failed");

        for (i, bary) in result.barycentric.iter().enumerate() {
            let sum = bary[0] + bary[1] + bary[2];
            assert!(
                (sum - 1.0).abs() < 1e-4,
                "sample {i}: barycentric sum = {sum} (expected ~1)"
            );
        }
    }

    #[test]
    fn default_sampler_face_indices_within_bounds() {
        let mesh = quad_mesh();
        let num_faces = mesh.faces.len() as u32;
        let sampler = DefaultSampler;
        let result = sampler
            .sample_surface(&mesh, 100, 55)
            .expect("sampling failed");

        for (i, &fi) in result.face_indices.iter().enumerate() {
            assert!(
                fi < num_faces,
                "sample {i}: face_index {fi} >= num_faces {num_faces}"
            );
        }
    }

    #[test]
    fn default_sampler_zero_samples_returns_empty() {
        let mesh = quad_mesh();
        let sampler = DefaultSampler;
        let result = sampler
            .sample_surface(&mesh, 0, 0)
            .expect("sampling with n=0 failed");
        assert!(result.positions.is_empty());
        assert!(result.normals.is_empty());
        assert!(result.face_indices.is_empty());
        assert!(result.barycentric.is_empty());
    }

    #[test]
    fn default_sampler_validates_even_when_zero_samples_requested() {
        // Regression: an `n == 0` fast path used to return `Ok(empty)` before
        // any validation, so a mesh that every non-zero `n` rejects was
        // silently accepted at `n == 0`. Validation must not depend on `n`.
        let sampler = DefaultSampler;

        let no_faces = Mesh::new(vec![], vec![]);
        assert!(
            sampler.sample_surface(&no_faces, 0, 0).is_err(),
            "a face-less mesh must be rejected regardless of the sample count"
        );

        let malformed = Mesh {
            vertices: vec![na::Point3::new(0.0f32, 0.0, 0.0)],
            normals: Vec::new(),
            faces: vec![[0, 0, 0]],
            uv_coords: Vec::new(),
        };
        assert!(
            sampler.sample_surface(&malformed, 0, 0).is_err(),
            "a malformed mesh must be rejected regardless of the sample count"
        );

        // A well-formed mesh with n == 0 is still an ordinary empty success.
        let ok = sampler
            .sample_surface(&quad_mesh(), 0, 0)
            .expect("n == 0 on a well-formed mesh is not an error");
        assert!(ok.positions.is_empty());
    }

    #[test]
    fn default_sampler_empty_mesh_returns_error() {
        let mesh = Mesh::new(vec![], vec![]);
        let sampler = DefaultSampler;
        let result = sampler.sample_surface(&mesh, 10, 0);
        assert!(
            result.is_err(),
            "sampling from empty mesh should return error"
        );
    }

    // -----------------------------------------------------------------------
    // Regression: `sample_mesh_surface` reports a malformed mesh as an error
    // rather than an empty `Vec`. `DefaultSampler` must propagate that error
    // instead of handing back a zero-length `SurfaceSample` that the caller
    // cannot tell apart from a legitimately empty result.
    // -----------------------------------------------------------------------

    #[test]
    fn default_sampler_propagates_malformed_mesh_error() {
        // `Mesh`'s fields are public, so this bypasses `Mesh::new`'s
        // invariants: three vertices but no normals, and a face pointing at a
        // vertex index that does not exist.
        let mismatched_normals = Mesh {
            vertices: vec![
                na::Point3::new(0.0f32, 0.0, 0.0),
                na::Point3::new(1.0f32, 0.0, 0.0),
                na::Point3::new(0.0f32, 1.0, 0.0),
            ],
            normals: Vec::new(),
            faces: vec![[0, 1, 2]],
            uv_coords: Vec::new(),
        };
        let out_of_range_face = Mesh {
            vertices: vec![
                na::Point3::new(0.0f32, 0.0, 0.0),
                na::Point3::new(1.0f32, 0.0, 0.0),
            ],
            normals: vec![na::Vector3::new(0.0f32, 0.0, 1.0); 2],
            faces: vec![[0, 1, 9]],
            uv_coords: Vec::new(),
        };

        let sampler = DefaultSampler;
        for (label, mesh) in [
            ("mismatched normals length", mismatched_normals),
            ("out-of-range face index", out_of_range_face),
        ] {
            match sampler.sample_surface(&mesh, 16, 11) {
                Err(FlameError::InvalidParams(msg)) => {
                    assert!(
                        msg.contains("sample_mesh_surface"),
                        "{label}: error should come from the sampler, got: {msg}"
                    );
                }
                Err(other) => panic!("{label}: expected InvalidParams, got {other:?}"),
                Ok(sample) => panic!(
                    "{label}: a malformed mesh must be an error, but sampling \
                     succeeded with {} points",
                    sample.positions.len()
                ),
            }
        }
    }

    // -----------------------------------------------------------------------
    // NormalMapProvider tests
    // -----------------------------------------------------------------------

    #[test]
    fn normal_map_provider_produces_correct_dimensions() {
        let provider = MockNormalMapProvider {
            width: 256,
            height: 128,
        };
        let params = FlameParams::neutral();
        let result = provider
            .generate_normal_map(&params, 256, 128)
            .expect("generate_normal_map failed");

        let expected_size = 256 * 128 * 3;
        assert_eq!(
            result.len(),
            expected_size,
            "expected {} bytes, got {}",
            expected_size,
            result.len()
        );
    }

    #[test]
    fn generate_normal_maps_multi_view_default_impl_rejects_multiple_poses() {
        // The default implementation has no camera model, so it cannot
        // honour distinct poses. Even when the caller happens to pass
        // identical matrices (as here), it must fail loudly rather than
        // silently return N duplicate images — a real (differing) pose set
        // would be silently wrong under the old "just ignore the pose"
        // default.
        let provider = MockNormalMapProvider {
            width: 64,
            height: 64,
        };
        let params = FlameParams::neutral();

        let identity: [[f32; 4]; 4] = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let poses = vec![identity; 3];

        let result = provider.generate_normal_maps_multi_view(&params, &poses, 64, 64);
        assert!(
            result.is_err(),
            "default multi-view impl must reject more than one pose, not silently \
             duplicate images"
        );
    }

    #[test]
    fn generate_normal_maps_multi_view_default_impl_single_pose_delegates() {
        // Exactly one pose has no ambiguity to silently get wrong, so the
        // default implementation may (and does) delegate to
        // `generate_normal_map` directly.
        let provider = MockNormalMapProvider {
            width: 64,
            height: 64,
        };
        let params = FlameParams::neutral();
        let identity: [[f32; 4]; 4] = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];

        let results = provider
            .generate_normal_maps_multi_view(&params, &[identity], 64, 64)
            .expect("single pose has no ambiguity, default impl should succeed");

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].len(), 64 * 64 * 3);
    }

    #[test]
    fn mock_normal_map_provider_implements_trait() {
        // This test verifies that a custom type correctly implements NormalMapProvider.
        // The function below requires a boxed trait object — if it compiles, the impl is correct.
        fn accepts_provider(_p: &dyn NormalMapProvider) {}

        let provider = MockNormalMapProvider {
            width: 512,
            height: 512,
        };
        accepts_provider(&provider);

        // Also verify default_resolution is correct
        let (w, h) = provider.default_resolution();
        assert_eq!(w, 512);
        assert_eq!(h, 512);
    }

    #[test]
    fn generate_normal_maps_multi_view_zero_poses_returns_empty() {
        let provider = MockNormalMapProvider {
            width: 64,
            height: 64,
        };
        let params = FlameParams::neutral();
        let results = provider
            .generate_normal_maps_multi_view(&params, &[], 64, 64)
            .expect("empty pose list should succeed");
        assert!(results.is_empty(), "zero poses should yield zero images");
    }

    // -----------------------------------------------------------------------
    // FlameNormalMapProvider (production impl) tests
    // -----------------------------------------------------------------------

    /// A minimal 3-vertex, 2-joint synthetic FLAME model with a single,
    /// non-degenerate, visible-from-`Camera::default_front` triangle. Shape,
    /// expression, and pose-corrective blend shapes are all zero, so
    /// `model.forward(&FlameParams::neutral())` reproduces `v_template`
    /// unchanged — only camera pose affects the render.
    fn build_visible_triangle_model() -> FlameModel {
        use ndarray::{Array2, Array3};

        let n_verts = 3;
        let n_joints = 2;
        let n_shape = 1;
        let n_expr = 1;
        let n_pose_dirs = (n_joints - 1) * 9;

        let v_template = Array2::from_shape_vec(
            (n_verts, 3),
            vec![0.0, 0.05, 0.0, -0.05, -0.05, 0.0, 0.05, -0.05, 0.0],
        )
        .expect("test: fixed shape matches data length");

        let faces = vec![[0u32, 1, 2]];
        let shapedirs = Array3::zeros((n_verts, 3, n_shape));
        let expressiondirs = Array3::zeros((n_verts, 3, n_expr));
        let posedirs = Array3::zeros((n_verts, 3, n_pose_dirs));
        let j_regressor =
            Array2::from_shape_vec((n_joints, n_verts), vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
                .expect("test: fixed shape matches data length");
        let parents = vec![-1i32, 0];
        let lbs_weights =
            Array2::from_shape_vec((n_verts, n_joints), vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
                .expect("test: fixed shape matches data length");

        FlameModel::from_arrays(
            v_template,
            faces,
            shapedirs,
            expressiondirs,
            posedirs,
            j_regressor,
            parents,
            lbs_weights,
            n_joints,
        )
    }

    #[test]
    fn flame_normal_map_provider_single_view_matches_dimensions() {
        let model = Arc::new(build_visible_triangle_model());
        let provider = FlameNormalMapProvider::new(model);
        let params = FlameParams::neutral();
        let img = provider
            .generate_normal_map(&params, 32, 48)
            .expect("generation should succeed");
        assert_eq!(img.len(), 32 * 48 * 3);
    }

    #[test]
    fn flame_normal_map_provider_multi_view_varies_by_pose() {
        // Two poses that differ only by a small camera-space X translation:
        // the rendered triangle must shift accordingly, proving (unlike the
        // trait's pose-ignoring default) that the pose is actually honoured.
        let model = Arc::new(build_visible_triangle_model());
        let provider = FlameNormalMapProvider::new(model);
        let params = FlameParams::neutral();

        let pose_a: [[f32; 4]; 4] = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.6],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let pose_b: [[f32; 4]; 4] = [
            [1.0, 0.0, 0.0, 0.05],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.6],
            [0.0, 0.0, 0.0, 1.0],
        ];

        let images = provider
            .generate_normal_maps_multi_view(&params, &[pose_a, pose_b], 64, 64)
            .expect("multi-view generation should succeed");

        assert_eq!(images.len(), 2);
        assert_eq!(images[0].len(), 64 * 64 * 3);
        assert_eq!(images[1].len(), 64 * 64 * 3);
        assert_ne!(
            images[0], images[1],
            "different camera poses must produce different renders, unlike the \
             trait's pose-ignoring default implementation"
        );
    }

    #[test]
    fn flame_normal_map_provider_rejects_zero_dimensions() {
        let model = Arc::new(build_visible_triangle_model());
        let provider = FlameNormalMapProvider::new(model);
        let params = FlameParams::neutral();

        assert!(
            provider.generate_normal_map(&params, 0, 64).is_err(),
            "zero width must be rejected, not panic inside the rasterizer"
        );
        assert!(
            provider
                .generate_normal_maps_multi_view(&params, &[[[0.0; 4]; 4]], 64, 0)
                .is_err(),
            "zero height must be rejected, not panic inside the rasterizer"
        );
    }
}
