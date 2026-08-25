//! Area-weighted random point sampling on a triangle mesh surface.

use nalgebra as na;
use rand::{Rng, RngExt};

use crate::error::FlameError;
use crate::mesh::Mesh;

/// A point sampled on a mesh surface, with its face binding information.
#[derive(Debug, Clone)]
pub struct SurfacePoint {
    /// World-space position.
    pub position: na::Point3<f32>,
    /// Interpolated unit normal.
    pub normal: na::Vector3<f32>,
    /// Index of the face this point lies on.
    pub face_index: u32,
    /// Barycentric coordinates `[u, v, w]` with `u + v + w ≈ 1`.
    pub barycentric: [f32; 3],
}

/// Interpolate a unit surface normal from three vertex normals at
/// barycentric weights `(u, v, w)`, falling back to the triangle's own
/// geometric (cross-product) normal when the weighted combination is
/// degenerate (near-zero norm, e.g. from near-opposite vertex normals at
/// these particular weights) instead of normalizing a near-zero vector,
/// which would silently produce a NaN-filled normal.
fn interpolate_normal_with_fallback(
    vertex_normals: [&na::Vector3<f32>; 3],
    barycentric: [f32; 3],
    vertices: [&na::Point3<f32>; 3],
) -> na::Vector3<f32> {
    let [n0, n1, n2] = vertex_normals;
    let [u, v, w] = barycentric;
    let [v0, v1, v2] = vertices;
    let n_interp = n0 * u + n1 * v + n2 * w;
    if n_interp.norm() > 1e-10 {
        return n_interp.normalize();
    }
    // Fall back to the triangle's actual geometric normal rather than an
    // arbitrary constant — it is the honest "real" normal at this point
    // when the per-vertex normals cancel out.
    let face_normal = (v1 - v0).cross(&(v2 - v0));
    if face_normal.norm() > 1e-10 {
        face_normal.normalize()
    } else {
        // Degenerate (zero-area or colinear) triangle too: nothing
        // meaningful to derive a normal from.
        na::Vector3::new(0.0, 0.0, 1.0)
    }
}

/// Sample `count` points uniformly (area-weighted) on the surface of `mesh`.
///
/// Uses the standard method:
/// 1. Build a CDF over triangle areas.
/// 2. For each sample, pick a random triangle and generate random barycentric
///    coordinates.
///
/// [`Mesh`]'s fields are all public, so a caller can construct one by
/// struct literal without going through [`Mesh::new`]'s invariants (a
/// `normals` entry per vertex, in-range face indices). This function
/// validates both up front and reports a *malformed* mesh as an error
/// rather than indexing blindly and panicking — or, worse, silently
/// handing the caller an empty `Vec` that is indistinguishable from a
/// legitimately empty result.
///
/// An *empty but well-formed* input is not an error: a mesh with no faces,
/// a `count` of zero, and a mesh whose triangles all have (near-)zero area
/// each yield `Ok(Vec::new())`. A collapsed, zero-area mesh is degenerate
/// geometry rather than invalid topology — there is simply nothing to
/// area-weight — so it stays on the success path.
///
/// # Errors
///
/// Returns [`FlameError::InvalidParams`] when `mesh` is malformed:
///
/// * `mesh.normals.len() != mesh.vertices.len()`, so no per-sample normal
///   can be interpolated; or
/// * some face references a vertex index `>= mesh.vertices.len()`, so the
///   topology is invalid.
pub fn sample_mesh_surface(
    mesh: &Mesh,
    count: usize,
    rng: &mut impl Rng,
) -> Result<Vec<SurfacePoint>, FlameError> {
    // Validate topology first, so a malformed mesh is reported even when the
    // caller asked for zero points or the mesh has no faces at all.
    if mesh.normals.len() != mesh.vertices.len() {
        return Err(FlameError::InvalidParams(format!(
            "sample_mesh_surface: mesh.normals.len() ({}) != mesh.vertices.len() ({}); \
             cannot interpolate a normal per sample",
            mesh.normals.len(),
            mesh.vertices.len()
        )));
    }
    if let Some(bad) = mesh
        .faces
        .iter()
        .flat_map(|f| f.iter())
        .copied()
        .find(|&vi| vi as usize >= mesh.vertices.len())
    {
        return Err(FlameError::InvalidParams(format!(
            "sample_mesh_surface: mesh has a face referencing vertex index {bad}, but the \
             mesh only has {} vertices; mesh topology is invalid",
            mesh.vertices.len()
        )));
    }

    if mesh.faces.is_empty() || count == 0 {
        return Ok(Vec::new());
    }

    // 1. Compute face areas and build CDF
    let areas: Vec<f32> = mesh.faces.iter().map(|f| mesh.face_area(f)).collect();
    let total_area: f32 = areas.iter().sum();

    if total_area < 1e-12 {
        // Degenerate (fully collapsed) geometry: well-formed topology with
        // nothing to area-weight. Zero points is the honest answer.
        tracing::debug!(
            "sample_mesh_surface: total mesh area is {total_area:e}; no points to sample"
        );
        return Ok(Vec::new());
    }

    let mut cdf = Vec::with_capacity(areas.len());
    let mut cumulative = 0.0f32;
    for &a in &areas {
        cumulative += a / total_area;
        cdf.push(cumulative);
    }
    // Ensure last entry is exactly 1.0
    if let Some(last) = cdf.last_mut() {
        *last = 1.0;
    }

    // 2. Sample points
    let mut points = Vec::with_capacity(count);

    for _ in 0..count {
        // Pick a face via the CDF
        let r: f32 = rng.random();
        let face_idx = cdf.partition_point(|&x| x < r).min(mesh.faces.len() - 1);
        let face = &mesh.faces[face_idx];

        let v0 = &mesh.vertices[face[0] as usize];
        let v1 = &mesh.vertices[face[1] as usize];
        let v2 = &mesh.vertices[face[2] as usize];

        let n0 = &mesh.normals[face[0] as usize];
        let n1 = &mesh.normals[face[1] as usize];
        let n2 = &mesh.normals[face[2] as usize];

        // Random barycentric coordinates (uniform on triangle)
        let r1: f32 = rng.random::<f32>().sqrt();
        let r2: f32 = rng.random();
        let u = 1.0 - r1;
        let v = r2 * r1;
        let w = 1.0 - u - v;

        let position = na::Point3::from(v0.coords * u + v1.coords * v + v2.coords * w);
        let normal = interpolate_normal_with_fallback([n0, n1, n2], [u, v, w], [v0, v1, v2]);

        points.push(SurfacePoint {
            position,
            normal,
            face_index: face_idx as u32,
            barycentric: [u, v, w],
        });
    }

    Ok(points)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    /// Build a minimal unit triangle mesh for testing.
    fn unit_triangle() -> Mesh {
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
    fn sampled_points_are_on_surface() {
        let mesh = unit_triangle();
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let points = sample_mesh_surface(&mesh, 100, &mut rng).expect("well-formed mesh");

        assert_eq!(points.len(), 100);
        for p in &points {
            // Point should be in the triangle: x >= 0, y >= 0, x + y <= 1
            assert!(p.position.x >= -1e-6, "x = {}", p.position.x);
            assert!(p.position.y >= -1e-6, "y = {}", p.position.y);
            assert!(
                p.position.x + p.position.y <= 1.0 + 1e-6,
                "x+y = {}",
                p.position.x + p.position.y
            );
            // z should be 0 (planar triangle on XY plane)
            assert!(p.position.z.abs() < 1e-6);
            // Barycentric coords should sum to ~1
            let sum: f32 = p.barycentric.iter().sum();
            assert!((sum - 1.0).abs() < 1e-5, "bary sum = {sum}");
        }
    }

    // -----------------------------------------------------------------------
    // Regression: `Mesh`'s fields are all public, so a caller can build one
    // that violates `Mesh::new`'s invariants (mismatched normals/vertices
    // lengths, out-of-range face indices). `sample_mesh_surface` must
    // report such input as an error — not panic, and not silently return an
    // empty `Vec` that a caller cannot distinguish from "nothing to sample".
    // -----------------------------------------------------------------------

    #[test]
    fn sample_mesh_surface_rejects_mismatched_normals_len() {
        let mesh = Mesh {
            vertices: vec![
                na::Point3::new(0.0, 0.0, 0.0),
                na::Point3::new(1.0, 0.0, 0.0),
                na::Point3::new(0.0, 1.0, 0.0),
            ],
            normals: Vec::new(), // deliberately empty / mismatched
            faces: vec![[0, 1, 2]],
            uv_coords: Vec::new(),
        };
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        let err = sample_mesh_surface(&mesh, 10, &mut rng)
            .expect_err("mismatched normals/vertices lengths must be an error, not empty output");
        assert!(
            matches!(err, FlameError::InvalidParams(_)),
            "expected InvalidParams, got {err:?}"
        );
        assert!(
            err.to_string().contains("normals"),
            "the error must name the offending field, got: {err}"
        );
    }

    #[test]
    fn sample_mesh_surface_rejects_out_of_range_face_index() {
        let mesh = Mesh {
            vertices: vec![
                na::Point3::new(0.0, 0.0, 0.0),
                na::Point3::new(1.0, 0.0, 0.0),
            ],
            normals: vec![na::Vector3::new(0.0, 0.0, 1.0); 2],
            // Face references vertex index 5, which doesn't exist.
            faces: vec![[0, 1, 5]],
            uv_coords: Vec::new(),
        };
        let mut rng = rand::rngs::StdRng::seed_from_u64(2);
        let err = sample_mesh_surface(&mesh, 10, &mut rng)
            .expect_err("an out-of-range face index must be an error, not empty output");
        assert!(
            matches!(err, FlameError::InvalidParams(_)),
            "expected InvalidParams, got {err:?}"
        );
        assert!(
            err.to_string().contains('5'),
            "the error must name the offending vertex index, got: {err}"
        );
    }

    #[test]
    fn sample_mesh_surface_rejects_malformed_mesh_even_when_count_is_zero() {
        // Validation happens before the `count == 0` fast path, so a caller
        // probing with zero samples still learns the mesh is broken.
        let mesh = Mesh {
            vertices: vec![na::Point3::new(0.0, 0.0, 0.0)],
            normals: Vec::new(),
            faces: vec![[0, 0, 0]],
            uv_coords: Vec::new(),
        };
        let mut rng = rand::rngs::StdRng::seed_from_u64(4);
        assert!(sample_mesh_surface(&mesh, 0, &mut rng).is_err());
    }

    #[test]
    fn sample_mesh_surface_valid_mesh_still_samples_normally() {
        // The validation must not reject a perfectly ordinary mesh.
        let mesh = unit_triangle();
        let mut rng = rand::rngs::StdRng::seed_from_u64(3);
        let points = sample_mesh_surface(&mesh, 5, &mut rng).expect("well-formed mesh");
        assert_eq!(points.len(), 5);
    }

    #[test]
    fn sample_mesh_surface_wellformed_but_empty_inputs_are_not_errors() {
        // A face-less mesh, a zero `count`, and a fully collapsed (zero-area)
        // mesh are all well-formed: they must succeed with zero points rather
        // than be conflated with malformed input.
        let mut rng = rand::rngs::StdRng::seed_from_u64(5);

        let empty = Mesh::new(vec![], vec![]);
        assert!(sample_mesh_surface(&empty, 100, &mut rng)
            .expect("a face-less mesh is well-formed")
            .is_empty());

        let triangle = unit_triangle();
        assert!(sample_mesh_surface(&triangle, 0, &mut rng)
            .expect("count == 0 is well-formed")
            .is_empty());

        let collapsed = Mesh::new(vec![na::Point3::origin(); 3], vec![[0, 1, 2]]);
        assert!(sample_mesh_surface(&collapsed, 10, &mut rng)
            .expect("a zero-area mesh is degenerate geometry, not malformed topology")
            .is_empty());
    }

    // -----------------------------------------------------------------------
    // Regression: interpolate_normal_with_fallback must never return a NaN
    // normal, even when the per-vertex normals cancel out exactly at the
    // sampled barycentric weights.
    // -----------------------------------------------------------------------

    #[test]
    fn interpolate_normal_with_fallback_handles_cancelling_normals() {
        // n0 and n1 are exact opposites; at u = v = 0.5, w = 0 the weighted
        // sum is exactly zero. A plain `.normalize()` here would produce a
        // NaN-filled vector; the fallback must instead return the
        // triangle's finite, unit-length geometric normal.
        let n0 = na::Vector3::new(1.0f32, 0.0, 0.0);
        let n1 = na::Vector3::new(-1.0f32, 0.0, 0.0);
        let n2 = na::Vector3::new(0.0f32, 0.0, 1.0);
        let v0 = na::Point3::new(0.0f32, 0.0, 0.0);
        let v1 = na::Point3::new(1.0f32, 0.0, 0.0);
        let v2 = na::Point3::new(0.0f32, 1.0, 0.0);

        let normal =
            interpolate_normal_with_fallback([&n0, &n1, &n2], [0.5, 0.5, 0.0], [&v0, &v1, &v2]);

        assert!(
            normal.x.is_finite() && normal.y.is_finite() && normal.z.is_finite(),
            "fallback normal must be finite, got {normal:?}"
        );
        assert!(
            (normal.norm() - 1.0).abs() < 1e-5,
            "fallback normal must be unit length, got norm = {}",
            normal.norm()
        );
    }

    #[test]
    fn interpolate_normal_with_fallback_non_degenerate_matches_plain_interpolation() {
        // Away from the degenerate case, the fallback path must not
        // engage: the result must match plain normalized interpolation.
        let n0 = na::Vector3::new(0.0f32, 0.0, 1.0);
        let n1 = na::Vector3::new(0.0f32, 0.0, 1.0);
        let n2 = na::Vector3::new(0.0f32, 0.0, 1.0);
        let v0 = na::Point3::new(0.0f32, 0.0, 0.0);
        let v1 = na::Point3::new(1.0f32, 0.0, 0.0);
        let v2 = na::Point3::new(0.0f32, 1.0, 0.0);

        let normal =
            interpolate_normal_with_fallback([&n0, &n1, &n2], [0.3, 0.3, 0.4], [&v0, &v1, &v2]);
        assert!((normal - na::Vector3::new(0.0, 0.0, 1.0)).norm() < 1e-5);
    }
}
