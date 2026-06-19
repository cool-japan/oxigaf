//! Mesh quality analysis for FLAME meshes.
//!
//! Provides structs and free functions for computing quality metrics on
//! triangle meshes: face areas, aspect ratios, boundary detection, manifold
//! testing, and a complete [`MeshQualityReport`].

use std::collections::HashMap;
use std::fmt::Write as FmtWrite;

// ---------------------------------------------------------------------------
// Public data structures
// ---------------------------------------------------------------------------

/// Statistics about face (triangle) quality.
#[derive(Debug, Clone)]
pub struct FaceQualityStats {
    /// Total number of triangles in the mesh.
    pub total_faces: usize,
    /// Number of degenerate triangles (area < 1e-10 or duplicate vertex indices).
    pub degenerate_count: usize,
    /// Area of the smallest triangle.
    pub min_area: f32,
    /// Area of the largest triangle.
    pub max_area: f32,
    /// Mean (average) triangle area.
    pub mean_area: f32,
    /// Population standard deviation of triangle areas.
    pub std_area: f32,
    /// Minimum aspect ratio across all triangles (`min_edge` / `max_edge`; 1.0 = equilateral).
    pub min_aspect_ratio: f32,
    /// Mean aspect ratio across all triangles.
    pub mean_aspect_ratio: f32,
    /// Number of triangles with at least one interior angle greater than 90°.
    pub obtuse_count: usize,
}

/// Statistics about vertex distribution.
#[derive(Debug, Clone)]
pub struct VertexQualityStats {
    /// Total number of vertices in the mesh.
    pub total_vertices: usize,
    /// Number of vertices not referenced by any face.
    pub isolated_count: usize,
    /// Minimum number of faces sharing a vertex (among non-isolated vertices; 0 if all isolated).
    pub min_valence: u32,
    /// Maximum number of faces sharing a vertex (0 if all vertices are isolated).
    pub max_valence: u32,
    /// Mean face count per vertex (over non-isolated vertices only; 0.0 if all isolated).
    pub mean_valence: f32,
    /// Number of vertices on the mesh boundary (incident to at least one boundary edge).
    pub boundary_count: usize,
}

/// Complete mesh quality report.
#[derive(Debug, Clone)]
pub struct MeshQualityReport {
    /// Face-level quality statistics.
    pub face_stats: FaceQualityStats,
    /// Vertex-level quality statistics.
    pub vertex_stats: VertexQualityStats,
    /// `true` if every edge is shared by at most 2 faces.
    pub is_manifold: bool,
    /// `true` if any boundary edges exist.
    pub has_boundary: bool,
    /// Euler characteristic V - E + F (2 for a closed sphere topology).
    pub euler_characteristic: i32,
    /// Human-readable warnings (e.g. degenerate faces, non-manifold edges).
    pub warnings: Vec<String>,
}

impl MeshQualityReport {
    /// Format the report as a multi-line human-readable table.
    #[must_use]
    pub fn format_report(&self) -> String {
        let mut out = String::new();

        let _ = writeln!(out, "=== Mesh Quality Report ===");
        let _ = writeln!(out, "--- Face Statistics ---");
        let _ = writeln!(out, "  Total faces:        {}", self.face_stats.total_faces);
        let _ = writeln!(
            out,
            "  Degenerate faces:   {}",
            self.face_stats.degenerate_count
        );
        let _ = writeln!(
            out,
            "  Area (min/mean/max):{:.6} / {:.6} / {:.6}",
            self.face_stats.min_area, self.face_stats.mean_area, self.face_stats.max_area
        );
        let _ = writeln!(out, "  Area std dev:       {:.6}", self.face_stats.std_area);
        let _ = writeln!(
            out,
            "  Aspect ratio (min/mean): {:.4} / {:.4}",
            self.face_stats.min_aspect_ratio, self.face_stats.mean_aspect_ratio
        );
        let _ = writeln!(
            out,
            "  Obtuse triangles:   {}",
            self.face_stats.obtuse_count
        );

        let _ = writeln!(out, "--- Vertex Statistics ---");
        let _ = writeln!(
            out,
            "  Total vertices:     {}",
            self.vertex_stats.total_vertices
        );
        let _ = writeln!(
            out,
            "  Isolated vertices:  {}",
            self.vertex_stats.isolated_count
        );
        let _ = writeln!(
            out,
            "  Boundary vertices:  {}",
            self.vertex_stats.boundary_count
        );
        let _ = writeln!(
            out,
            "  Valence (min/mean/max): {} / {:.2} / {}",
            self.vertex_stats.min_valence,
            self.vertex_stats.mean_valence,
            self.vertex_stats.max_valence
        );

        let _ = writeln!(out, "--- Topology ---");
        let _ = writeln!(out, "  Manifold:           {}", self.is_manifold);
        let _ = writeln!(out, "  Has boundary:       {}", self.has_boundary);
        let _ = writeln!(out, "  Euler characteristic: {}", self.euler_characteristic);

        if self.warnings.is_empty() {
            let _ = writeln!(out, "--- Warnings: none ---");
        } else {
            let _ = writeln!(out, "--- Warnings ({}) ---", self.warnings.len());
            for w in &self.warnings {
                let _ = writeln!(out, "  ! {w}");
            }
        }

        out
    }

    /// Returns `true` if the mesh has critical structural problems:
    /// degenerate faces, isolated vertices, or non-manifold topology.
    #[must_use]
    pub fn has_critical_issues(&self) -> bool {
        self.face_stats.degenerate_count > 0
            || self.vertex_stats.isolated_count > 0
            || !self.is_manifold
    }
}

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

/// Return the canonical (sorted) form of an undirected edge.
#[inline]
fn canonical_edge(a: u32, b: u32) -> (u32, u32) {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

/// Return `true` if any two vertex indices in the face are equal.
#[inline]
fn has_duplicate_indices(face: &[u32; 3]) -> bool {
    face[0] == face[1] || face[1] == face[2] || face[0] == face[2]
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// Compute the area of every triangle in the mesh.
///
/// Area of triangle `(v0, v1, v2)` = 0.5 × ‖(v1−v0) × (v2−v0)‖.
/// Returns a `Vec<f32>` of length equal to `faces.len()`.
#[must_use]
pub fn compute_face_areas(vertices: &[[f32; 3]], faces: &[[u32; 3]]) -> Vec<f32> {
    faces
        .iter()
        .map(|face| {
            let v0 = vertices[face[0] as usize];
            let v1 = vertices[face[1] as usize];
            let v2 = vertices[face[2] as usize];
            let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
            let e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];
            let cross = [
                e1[1] * e2[2] - e1[2] * e2[1],
                e1[2] * e2[0] - e1[0] * e2[2],
                e1[0] * e2[1] - e1[1] * e2[0],
            ];
            let norm = (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
            norm * 0.5
        })
        .collect()
}

/// Compute the aspect ratio of every triangle in the mesh.
///
/// Aspect ratio = `min_edge_length` / `max_edge_length`.
/// Returns 1.0 for an equilateral triangle, 0.0 for a degenerate triangle.
#[must_use]
pub fn compute_face_aspect_ratios(vertices: &[[f32; 3]], faces: &[[u32; 3]]) -> Vec<f32> {
    faces
        .iter()
        .map(|face| {
            let v0 = vertices[face[0] as usize];
            let v1 = vertices[face[1] as usize];
            let v2 = vertices[face[2] as usize];

            let edge_len_sq = |a: [f32; 3], b: [f32; 3]| -> f32 {
                let dx = b[0] - a[0];
                let dy = b[1] - a[1];
                let dz = b[2] - a[2];
                dx * dx + dy * dy + dz * dz
            };

            let l01 = edge_len_sq(v0, v1).sqrt();
            let l12 = edge_len_sq(v1, v2).sqrt();
            let l20 = edge_len_sq(v2, v0).sqrt();

            let max_len = l01.max(l12).max(l20);
            if max_len < f32::EPSILON {
                return 0.0;
            }
            let min_len = l01.min(l12).min(l20);
            min_len / max_len
        })
        .collect()
}

/// Find all boundary edges: edges shared by exactly one triangle.
///
/// Returns a `Vec` of `(u32, u32)` canonical (sorted) edge pairs.
#[must_use]
pub fn find_boundary_edges(faces: &[[u32; 3]]) -> Vec<(u32, u32)> {
    let mut edge_count: HashMap<(u32, u32), u32> = HashMap::new();
    for face in faces {
        let edges = [
            canonical_edge(face[0], face[1]),
            canonical_edge(face[1], face[2]),
            canonical_edge(face[2], face[0]),
        ];
        for edge in edges {
            *edge_count.entry(edge).or_insert(0) += 1;
        }
    }
    edge_count
        .into_iter()
        .filter(|(_, count)| *count == 1)
        .map(|(edge, _)| edge)
        .collect()
}

/// Return `true` if the mesh is manifold: every edge appears at most twice
/// (in either orientation, counted together via canonical form).
#[must_use]
pub fn is_manifold_mesh(faces: &[[u32; 3]]) -> bool {
    let mut edge_count: HashMap<(u32, u32), u32> = HashMap::new();
    for face in faces {
        let edges = [
            canonical_edge(face[0], face[1]),
            canonical_edge(face[1], face[2]),
            canonical_edge(face[2], face[0]),
        ];
        for edge in edges {
            let count = edge_count.entry(edge).or_insert(0);
            *count += 1;
            if *count > 2 {
                return false;
            }
        }
    }
    true
}

// ── mesh quality helpers ─────────────────────────────────────────────────────

/// Return type of [`build_edge_and_valence`]: `(edge_count, valence, boundary_edges)`.
type EdgeValenceResult = (HashMap<(u32, u32), u32>, Vec<u32>, Vec<(u32, u32)>);

/// Build edge occurrence map and per-vertex valence from a face list.
/// Returns `(edge_count, valence, boundary_edges)`.
fn build_edge_and_valence(faces: &[[u32; 3]], num_verts: usize) -> EdgeValenceResult {
    let mut edge_count: HashMap<(u32, u32), u32> = HashMap::with_capacity(faces.len() * 3);
    let mut valence = vec![0u32; num_verts];
    for face in faces {
        for &[edge_a, edge_b] in &[[face[0], face[1]], [face[1], face[2]], [face[2], face[0]]] {
            *edge_count
                .entry(canonical_edge(edge_a, edge_b))
                .or_insert(0) += 1;
        }
        for &idx in face {
            valence[idx as usize] += 1;
        }
    }
    let boundary_edges: Vec<(u32, u32)> = edge_count
        .iter()
        .filter(|(_, &cnt)| cnt == 1)
        .map(|(&edge, _)| edge)
        .collect();
    (edge_count, valence, boundary_edges)
}

/// Compute (min, max, mean, std-dev) over a non-empty slice of areas.
fn area_stats(areas: &[f32]) -> (f32, f32, f32, f32) {
    if areas.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let min_a = areas.iter().copied().fold(f32::INFINITY, f32::min);
    let max_a = areas.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mean_a = areas.iter().sum::<f32>() / areas.len() as f32;
    let variance = areas
        .iter()
        .map(|&val| {
            let diff = val - mean_a;
            diff * diff
        })
        .sum::<f32>()
        / areas.len() as f32;
    (min_a, max_a, mean_a, variance.sqrt())
}

/// Count faces where at least one interior angle is obtuse (dot product < 0).
fn count_obtuse_faces(vertices: &[[f32; 3]], faces: &[[u32; 3]]) -> usize {
    let sub = |p: [f32; 3], q: [f32; 3]| -> [f32; 3] { [p[0] - q[0], p[1] - q[1], p[2] - q[2]] };
    let dot = |p: [f32; 3], q: [f32; 3]| -> f32 { p[0] * q[0] + p[1] * q[1] + p[2] * q[2] };
    faces
        .iter()
        .filter(|face| {
            let v0 = vertices[face[0] as usize];
            let v1 = vertices[face[1] as usize];
            let v2 = vertices[face[2] as usize];
            dot(sub(v1, v0), sub(v2, v0)) < 0.0
                || dot(sub(v0, v1), sub(v2, v1)) < 0.0
                || dot(sub(v0, v2), sub(v1, v2)) < 0.0
        })
        .count()
}

/// Compute a complete quality report for the given mesh.
///
/// # Arguments
///
/// * `vertices` — vertex positions as `[f32; 3]` slices.
/// * `faces`    — triangle face indices as `[u32; 3]` slices.
#[must_use]
pub fn compute_mesh_quality(vertices: &[[f32; 3]], faces: &[[u32; 3]]) -> MeshQualityReport {
    let num_verts = vertices.len();
    let num_faces = faces.len();

    // 1. Edge map, valence, boundary edges
    let (edge_count, valence, boundary_edges) = build_edge_and_valence(faces, num_verts);
    let num_unique_edges = edge_count.len();
    let is_manifold = edge_count.values().all(|&cnt| cnt <= 2);
    let has_boundary = !boundary_edges.is_empty();

    // 2. Vertex stats
    let isolated_count = valence.iter().filter(|&&val| val == 0).count();
    let referenced_valences: Vec<u32> = valence.iter().copied().filter(|&val| val > 0).collect();
    let (min_valence, max_valence, mean_valence) = if referenced_valences.is_empty() {
        (0u32, 0u32, 0.0f32)
    } else {
        let min_v = referenced_valences.iter().copied().fold(u32::MAX, u32::min);
        let max_v = referenced_valences.iter().copied().fold(0u32, u32::max);
        let mean_v =
            referenced_valences.iter().sum::<u32>() as f32 / referenced_valences.len() as f32;
        (min_v, max_v, mean_v)
    };
    let mut boundary_vertex_set: std::collections::HashSet<u32> = std::collections::HashSet::new();
    for (bv_a, bv_b) in &boundary_edges {
        boundary_vertex_set.insert(*bv_a);
        boundary_vertex_set.insert(*bv_b);
    }
    let boundary_count = boundary_vertex_set.len();

    // 3. Face areas, aspect ratios, degeneracy
    let areas = compute_face_areas(vertices, faces);
    let aspect_ratios = compute_face_aspect_ratios(vertices, faces);
    let degenerate_count = faces
        .iter()
        .enumerate()
        .filter(|(idx, face)| has_duplicate_indices(face) || areas[*idx] < 1e-10)
        .count();
    let (min_area, max_area, mean_area, std_area) = area_stats(&areas);
    let (min_aspect_ratio, mean_aspect_ratio) = if aspect_ratios.is_empty() {
        (0.0f32, 0.0f32)
    } else {
        let min_ar = aspect_ratios.iter().copied().fold(f32::INFINITY, f32::min);
        let mean_ar = aspect_ratios.iter().sum::<f32>() / aspect_ratios.len() as f32;
        (min_ar, mean_ar)
    };

    // 4. Obtuse triangle count
    let obtuse_count = count_obtuse_faces(vertices, faces);

    // 5. Euler characteristic: V - E + F
    let euler_characteristic = i32::try_from(num_verts).unwrap_or(i32::MAX)
        - i32::try_from(num_unique_edges).unwrap_or(i32::MAX)
        + i32::try_from(num_faces).unwrap_or(i32::MAX);

    // 6. Warnings
    let mut warnings = Vec::new();
    if degenerate_count > 0 {
        warnings.push(format!("Degenerate faces: {degenerate_count}"));
    }
    if max_valence > 20 {
        warnings.push(format!("High valence vertex: {max_valence}"));
    }
    let non_manifold_count = edge_count.values().filter(|&&cnt| cnt > 2).count();
    if non_manifold_count > 0 {
        warnings.push(format!("Non-manifold edges: {non_manifold_count}"));
    }

    MeshQualityReport {
        face_stats: FaceQualityStats {
            total_faces: num_faces,
            degenerate_count,
            min_area,
            max_area,
            mean_area,
            std_area,
            min_aspect_ratio,
            mean_aspect_ratio,
            obtuse_count,
        },
        vertex_stats: VertexQualityStats {
            total_vertices: num_verts,
            isolated_count,
            min_valence,
            max_valence,
            mean_valence,
            boundary_count,
        },
        is_manifold,
        has_boundary,
        euler_characteristic,
        warnings,
    }
}

// ---------------------------------------------------------------------------
// Mesh trait extension
// ---------------------------------------------------------------------------

impl super::mesh::Mesh {
    /// Compute a complete [`MeshQualityReport`] for this mesh.
    #[must_use]
    pub fn quality_report(&self) -> MeshQualityReport {
        let verts: Vec<[f32; 3]> = self.vertices.iter().map(|v| [v.x, v.y, v.z]).collect();
        compute_mesh_quality(&verts, &self.faces)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Equilateral triangle with side length 1 in the XY plane.
    /// Area = sqrt(3)/4 ≈ 0.4330127
    fn equilateral_triangle_verts() -> Vec<[f32; 3]> {
        vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 3_f32.sqrt() / 2.0, 0.0],
        ]
    }

    fn equilateral_triangle_faces() -> Vec<[u32; 3]> {
        vec![[0, 1, 2]]
    }

    /// Tetrahedron: 4 vertices, 4 faces, 6 edges — closed manifold, no boundary.
    fn tetrahedron() -> (Vec<[f32; 3]>, Vec<[u32; 3]>) {
        let verts = vec![
            [1.0_f32, 1.0, 1.0],
            [1.0_f32, -1.0, -1.0],
            [-1.0_f32, 1.0, -1.0],
            [-1.0_f32, -1.0, 1.0],
        ];
        let faces = vec![[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]];
        (verts, faces)
    }

    /// Open strip: two triangles sharing an edge, with 4 boundary edges.
    fn open_strip() -> (Vec<[f32; 3]>, Vec<[u32; 3]>) {
        let verts = vec![
            [0.0_f32, 0.0, 0.0],
            [1.0_f32, 0.0, 0.0],
            [0.0_f32, 1.0, 0.0],
            [1.0_f32, 1.0, 0.0],
        ];
        // faces: (0,1,2) and (1,3,2)
        // Shared edge: canonical(1,2) = (1,2)
        // Boundary edges: (0,1),(0,2),(1,3),(2,3)
        let faces = vec![[0, 1, 2], [1, 3, 2]];
        (verts, faces)
    }

    // -----------------------------------------------------------------------
    // face area tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_face_areas_equilateral() {
        let verts = equilateral_triangle_verts();
        let faces = equilateral_triangle_faces();
        let areas = compute_face_areas(&verts, &faces);
        assert_eq!(areas.len(), 1);
        let expected = 3_f32.sqrt() / 4.0;
        assert!(
            (areas[0] - expected).abs() < 1e-6,
            "equilateral area: expected {expected}, got {}",
            areas[0]
        );
    }

    #[test]
    fn test_face_areas_degenerate() {
        // All three vertices at the same point — area must be ≈ 0
        let verts = vec![[0.0_f32; 3], [0.0_f32; 3], [0.0_f32; 3]];
        let faces = vec![[0u32, 1, 2]];
        let areas = compute_face_areas(&verts, &faces);
        assert!(
            areas[0] < 1e-10,
            "degenerate area should be ~0, got {}",
            areas[0]
        );
    }

    // -----------------------------------------------------------------------
    // aspect ratio tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_aspect_ratio_equilateral() {
        let verts = equilateral_triangle_verts();
        let faces = equilateral_triangle_faces();
        let ratios = compute_face_aspect_ratios(&verts, &faces);
        assert_eq!(ratios.len(), 1);
        // All edges have length 1, so ratio = 1/1 = 1.0
        assert!(
            (ratios[0] - 1.0).abs() < 1e-5,
            "equilateral aspect ratio should be 1.0, got {}",
            ratios[0]
        );
    }

    #[test]
    fn test_aspect_ratio_degenerate() {
        // Collinear vertices — max edge approaches sum of the other two
        let verts: Vec<[f32; 3]> = vec![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]];
        let faces = vec![[0u32, 1, 2]];
        let ratios = compute_face_aspect_ratios(&verts, &faces);
        assert_eq!(
            ratios[0], 0.0,
            "degenerate triangle aspect ratio should be 0.0"
        );
    }

    // -----------------------------------------------------------------------
    // boundary edge tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_boundary_edges_open_mesh() {
        let (_, faces) = open_strip();
        let boundary = find_boundary_edges(&faces);
        // 5 total edges: 1 shared (count=2), 4 boundary (count=1)
        assert_eq!(
            boundary.len(),
            4,
            "open strip should have 4 boundary edges, got {}",
            boundary.len()
        );
    }

    #[test]
    fn test_boundary_edges_closed_mesh() {
        let (_, faces) = tetrahedron();
        let boundary = find_boundary_edges(&faces);
        assert!(
            boundary.is_empty(),
            "tetrahedron should have no boundary edges, got {}",
            boundary.len()
        );
    }

    // -----------------------------------------------------------------------
    // manifold tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_is_manifold_valid_mesh() {
        let (_, faces) = tetrahedron();
        assert!(is_manifold_mesh(&faces), "tetrahedron should be manifold");
    }

    #[test]
    fn test_is_manifold_non_manifold() {
        // Three triangles sharing one edge (0,1) → edge appears 3 times
        let faces: Vec<[u32; 3]> = vec![[0, 1, 2], [0, 1, 3], [0, 1, 4]];
        assert!(
            !is_manifold_mesh(&faces),
            "mesh with triple-shared edge should not be manifold"
        );
    }

    // -----------------------------------------------------------------------
    // compute_mesh_quality
    // -----------------------------------------------------------------------

    #[test]
    fn test_compute_mesh_quality() {
        let (verts, faces) = open_strip();
        let report = compute_mesh_quality(&verts, &faces);
        assert_eq!(report.face_stats.total_faces, 2);
        assert_eq!(report.vertex_stats.total_vertices, 4);
        assert_eq!(report.face_stats.degenerate_count, 0);
        assert!(report.has_boundary, "open strip must have boundary");
        assert!(
            report.is_manifold,
            "open strip (each edge ≤ 2) should be manifold"
        );
    }

    // -----------------------------------------------------------------------
    // Euler characteristic
    // -----------------------------------------------------------------------

    #[test]
    fn test_quality_report_euler_characteristic() {
        let (verts, faces) = tetrahedron();
        let report = compute_mesh_quality(&verts, &faces);
        // V=4, E=6, F=4 → 4 - 6 + 4 = 2
        assert_eq!(
            report.euler_characteristic, 2,
            "tetrahedron Euler characteristic should be 2, got {}",
            report.euler_characteristic
        );
    }

    // -----------------------------------------------------------------------
    // Warnings
    // -----------------------------------------------------------------------

    #[test]
    fn test_quality_report_warnings_degenerate() {
        // Single degenerate face (duplicate indices)
        let verts = vec![[0.0_f32; 3], [1.0_f32, 0.0, 0.0], [2.0_f32, 0.0, 0.0]];
        let faces = vec![[0u32, 0, 1]]; // vertex 0 duplicated
        let report = compute_mesh_quality(&verts, &faces);
        assert_eq!(
            report.face_stats.degenerate_count, 1,
            "duplicate-index face should count as degenerate"
        );
        let has_degen_warn = report.warnings.iter().any(|w| w.contains("Degenerate"));
        assert!(has_degen_warn, "should have a Degenerate faces warning");
    }

    #[test]
    fn test_quality_report_warnings_non_manifold() {
        let faces: Vec<[u32; 3]> = vec![[0, 1, 2], [0, 1, 3], [0, 1, 4]];
        let verts: Vec<[f32; 3]> = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [0.5, -1.0, 0.0],
            [0.5, 0.0, 1.0],
        ];
        let report = compute_mesh_quality(&verts, &faces);
        assert!(!report.is_manifold);
        let has_manifold_warn = report.warnings.iter().any(|w| w.contains("Non-manifold"));
        assert!(
            has_manifold_warn,
            "should have a Non-manifold edges warning"
        );
    }

    // -----------------------------------------------------------------------
    // format_report
    // -----------------------------------------------------------------------

    #[test]
    fn test_quality_report_format() {
        let (verts, faces) = tetrahedron();
        let report = compute_mesh_quality(&verts, &faces);
        let formatted = report.format_report();
        assert!(
            formatted.contains("Mesh Quality Report"),
            "report should have header"
        );
        assert!(
            formatted.contains("Euler characteristic"),
            "report should include Euler characteristic"
        );
        assert!(
            formatted.contains("Total faces"),
            "report should include face count"
        );
        assert!(
            formatted.contains("Total vertices"),
            "report should include vertex count"
        );
    }

    // -----------------------------------------------------------------------
    // Mesh::quality_report integration
    // -----------------------------------------------------------------------

    #[test]
    fn test_mesh_quality_report_method() {
        use crate::mesh::Mesh;
        use nalgebra as na;

        let vertices = vec![
            na::Point3::new(0.0_f32, 0.0, 0.0),
            na::Point3::new(1.0_f32, 0.0, 0.0),
            na::Point3::new(0.0_f32, 1.0, 0.0),
            na::Point3::new(1.0_f32, 1.0, 0.0),
        ];
        let faces = vec![[0u32, 1, 2], [1, 3, 2]];
        let mesh = Mesh::new(vertices, faces);
        let report = mesh.quality_report();

        assert_eq!(report.face_stats.total_faces, 2);
        assert_eq!(report.vertex_stats.total_vertices, 4);
        assert_eq!(report.face_stats.degenerate_count, 0);
        assert!(report.has_boundary);
        assert!(report.is_manifold);
    }
}
