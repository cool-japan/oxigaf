//! Integration traits for FLAME components.
//!
//! These traits provide abstraction boundaries that allow testing and
//! interchangeability of normal map generation and surface sampling.

use crate::{FlameError, FlameParams, Mesh};
use rand::SeedableRng;

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
    /// The default implementation calls [`generate_normal_map`] once per pose,
    /// ignoring the pose matrix. Implementors that support camera transformations
    /// should override this method.
    ///
    /// # Errors
    ///
    /// Returns the first error encountered during generation.
    ///
    /// [`generate_normal_map`]: NormalMapProvider::generate_normal_map
    fn generate_normal_maps_multi_view(
        &self,
        params: &FlameParams,
        camera_poses: &[[[f32; 4]; 4]],
        width: u32,
        height: u32,
    ) -> Result<Vec<Vec<u8>>, FlameError> {
        camera_poses
            .iter()
            .map(|_pose| self.generate_normal_map(params, width, height))
            .collect()
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
/// Used by Gaussian initialization in the trainer crate.
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
/// Delegates to [`crate::sampler::sample_mesh_surface`].
pub struct DefaultSampler;

impl MeshSurfaceSampler for DefaultSampler {
    fn sample_surface(
        &self,
        mesh: &Mesh,
        n: usize,
        seed: u64,
    ) -> Result<SurfaceSample, FlameError> {
        if n == 0 {
            return Ok(SurfaceSample {
                positions: Vec::new(),
                normals: Vec::new(),
                face_indices: Vec::new(),
                barycentric: Vec::new(),
            });
        }

        if mesh.faces.is_empty() {
            return Err(FlameError::InvalidParams(
                "cannot sample from a mesh with no faces".to_string(),
            ));
        }

        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let points = crate::sampler::sample_mesh_surface(mesh, n, &mut rng);

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
    fn generate_normal_maps_multi_view_default_impl_works() {
        let provider = MockNormalMapProvider {
            width: 64,
            height: 64,
        };
        let params = FlameParams::neutral();

        // Create 3 dummy camera pose matrices (identity)
        let identity: [[f32; 4]; 4] = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let poses = vec![identity; 3];

        let results = provider
            .generate_normal_maps_multi_view(&params, &poses, 64, 64)
            .expect("multi-view generation failed");

        assert_eq!(results.len(), 3, "should produce one image per pose");
        for (i, img) in results.iter().enumerate() {
            let expected = 64 * 64 * 3;
            assert_eq!(
                img.len(),
                expected,
                "view {i}: expected {expected} bytes, got {}",
                img.len()
            );
        }
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
}
