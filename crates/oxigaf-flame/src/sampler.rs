//! Area-weighted random point sampling on a triangle mesh surface.

use nalgebra as na;
use rand::{Rng, RngExt};

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

/// Sample `count` points uniformly (area-weighted) on the surface of `mesh`.
///
/// Uses the standard method:
/// 1. Build a CDF over triangle areas.
/// 2. For each sample, pick a random triangle and generate random barycentric
///    coordinates.
#[must_use]
pub fn sample_mesh_surface(mesh: &Mesh, count: usize, rng: &mut impl Rng) -> Vec<SurfacePoint> {
    if mesh.faces.is_empty() || count == 0 {
        return Vec::new();
    }

    // 1. Compute face areas and build CDF
    let areas: Vec<f32> = mesh.faces.iter().map(|f| mesh.face_area(f)).collect();
    let total_area: f32 = areas.iter().sum();

    if total_area < 1e-12 {
        return Vec::new();
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
        let normal = (n0 * u + n1 * v + n2 * w).normalize();

        points.push(SurfacePoint {
            position,
            normal,
            face_index: face_idx as u32,
            barycentric: [u, v, w],
        });
    }

    points
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
        let points = sample_mesh_surface(&mesh, 100, &mut rng);

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
}
