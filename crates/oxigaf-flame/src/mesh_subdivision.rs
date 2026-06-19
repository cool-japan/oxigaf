//! Loop subdivision of triangle meshes.
//!
//! Implements Loop subdivision to refine a FLAME mesh to higher resolution,
//! producing smoother, more detailed geometry for high-quality rendering.
//!
//! # Algorithm
//!
//! Each subdivision level transforms every triangle into 4 child triangles:
//! 1. Add a new vertex at each edge midpoint (Loop or boundary rule)
//! 2. Update original vertex positions using neighbor averaging (Loop rule)
//! 3. Reconnect into 4 smaller triangles per original triangle
//!
//! Reference: Charles Loop, *Smooth Subdivision Surfaces Based on Triangles*,
//! M.S. Mathematics Thesis, University of Utah, 1987.

use nalgebra as na;
use std::collections::{HashMap, HashSet};
use thiserror::Error;

use crate::mesh::Mesh;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during mesh subdivision.
#[derive(Debug, Error)]
pub enum SubdivisionError {
    /// The input mesh has no vertices.
    #[error("Empty mesh: no vertices")]
    EmptyMesh,
    /// The input mesh has no faces.
    #[error("Empty mesh: no faces")]
    NoFaces,
    /// A face has more or fewer than 3 vertices.
    #[error("Non-triangular face detected at index {idx}: has {n_verts} vertices")]
    NonTriangularFace { idx: usize, n_verts: usize },
    /// A face references a vertex index beyond the vertex array.
    #[error("Vertex index {idx} out of range (n_vertices = {n})")]
    VertexIndexOutOfRange { idx: usize, n: usize },
    /// Subdivision would exceed the configured vertex cap.
    #[error(
        "Subdivision level {level} would produce too many vertices (estimated {estimated} > {max})"
    )]
    TooManyVertices {
        level: usize,
        estimated: usize,
        max: usize,
    },
    /// All three vertices of a face share the same index.
    #[error("Degenerate face {idx}: all three vertices are identical")]
    DegenerateFace { idx: usize },
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for Loop subdivision.
#[derive(Debug, Clone)]
pub struct SubdivisionConfig {
    /// Number of subdivision levels to apply (default: 1, max: 4).
    pub levels: usize,
    /// Apply the Loop vertex-update rule to smooth original vertices (default: true).
    pub update_vertices: bool,
    /// Recompute per-vertex normals after subdivision (default: true).
    pub recompute_normals: bool,
    /// Safety limit: refuse to subdivide if the estimated vertex count exceeds this
    /// value (default: 500 000).
    pub max_vertices: usize,
}

impl Default for SubdivisionConfig {
    fn default() -> Self {
        Self {
            levels: 1,
            update_vertices: true,
            recompute_normals: true,
            max_vertices: 500_000,
        }
    }
}

// ---------------------------------------------------------------------------
// Result / stats types
// ---------------------------------------------------------------------------

/// Result of applying Loop subdivision.
#[derive(Debug, Clone)]
pub struct SubdivisionResult {
    /// The refined mesh.
    pub mesh: Mesh,
    /// Vertex count before subdivision.
    pub n_original_vertices: usize,
    /// Number of new edge-midpoint vertices added.
    pub n_new_vertices: usize,
    /// Face count before subdivision.
    pub n_original_faces: usize,
    /// Face count after subdivision (should be 4× per original face per level).
    pub n_new_faces: usize,
    /// How many levels of subdivision were actually applied.
    pub levels_applied: usize,
}

/// Geometric statistics about a subdivided mesh.
#[derive(Debug, Clone)]
pub struct SubdivisionStats {
    /// Vertex count before all subdivision levels.
    pub original_vertices: usize,
    /// Vertex count after all subdivision levels.
    pub final_vertices: usize,
    /// Face count before all subdivision levels.
    pub original_faces: usize,
    /// Face count after all subdivision levels.
    pub final_faces: usize,
    /// Mean edge length in the final mesh.
    pub mean_edge_length: f32,
    /// Maximum edge length in the final mesh.
    pub max_edge_length: f32,
    /// Minimum edge length in the final mesh.
    pub min_edge_length: f32,
}

// ---------------------------------------------------------------------------
// Private math helpers
// ---------------------------------------------------------------------------

/// Compute the midpoint between two `Point3<f32>` values.
#[inline]
fn midpoint_vertex(a: &na::Point3<f32>, b: &na::Point3<f32>) -> na::Point3<f32> {
    na::Point3::from((a.coords + b.coords) / 2.0)
}

/// Add two `Point3<f32>` values (treating them as position vectors).
#[inline]
fn add_points(a: &na::Point3<f32>, b: &na::Point3<f32>) -> na::Point3<f32> {
    na::Point3::from(a.coords + b.coords)
}

/// Scale a `Point3<f32>` by a scalar (treating it as a position vector).
#[inline]
fn scale_point(a: &na::Point3<f32>, s: f32) -> na::Point3<f32> {
    na::Point3::from(a.coords * s)
}

/// Return the canonical (ordered) edge key with `v0 < v1`.
#[inline]
fn edge_key(a: u32, b: u32) -> (u32, u32) {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

// ---------------------------------------------------------------------------
// Private subdivision helpers
// ---------------------------------------------------------------------------

/// Build a per-vertex neighbor list from mesh faces.
///
/// Returns `Vec<Vec<u32>>` of length `n_vertices`; each element lists the
/// unique vertex indices reachable from that vertex via a mesh edge.
fn build_vertex_neighbors(faces: &[[u32; 3]], n_vertices: usize) -> Vec<Vec<u32>> {
    let mut neighbors: Vec<HashSet<u32>> = vec![HashSet::new(); n_vertices];
    for face in faces {
        let [v0, v1, v2] = *face;
        neighbors[v0 as usize].insert(v1);
        neighbors[v0 as usize].insert(v2);
        neighbors[v1 as usize].insert(v0);
        neighbors[v1 as usize].insert(v2);
        neighbors[v2 as usize].insert(v0);
        neighbors[v2 as usize].insert(v1);
    }
    neighbors
        .into_iter()
        .map(|s| s.into_iter().collect())
        .collect()
}

/// Identify boundary edges: edges that appear in exactly one face.
///
/// Returns a `HashSet<(u32, u32)>` of canonical (v0 < v1) edge pairs.
fn identify_boundary_edges(faces: &[[u32; 3]]) -> HashSet<(u32, u32)> {
    let mut count: HashMap<(u32, u32), u32> = HashMap::new();
    for face in faces {
        let [v0, v1, v2] = *face;
        *count.entry(edge_key(v0, v1)).or_insert(0) += 1;
        *count.entry(edge_key(v1, v2)).or_insert(0) += 1;
        *count.entry(edge_key(v0, v2)).or_insert(0) += 1;
    }
    count
        .into_iter()
        .filter_map(|(k, v)| if v == 1 { Some(k) } else { None })
        .collect()
}

/// Return `true` if vertex `v_idx` is a boundary vertex
/// (i.e., it belongs to at least one boundary edge).
fn is_boundary_vertex(v_idx: u32, boundary_edges: &HashSet<(u32, u32)>) -> bool {
    boundary_edges
        .iter()
        .any(|&(a, b)| a == v_idx || b == v_idx)
}

/// Compute the Loop subdivision β weight for an interior vertex with `n` neighbors.
///
/// Uses Warren's approximation:
/// - n = 3: β = 3/16
/// - n > 3: β = 3 / (8 n)
#[inline]
fn compute_loop_vertex_weight(n_neighbors: usize) -> f32 {
    if n_neighbors == 3 {
        3.0 / 16.0
    } else {
        3.0 / (8.0 * n_neighbors as f32)
    }
}

/// Compute the updated position of an original vertex using the Loop rule.
///
/// - Interior vertex: `(1 − n·β) · v + β · Σ neighbors`
/// - Boundary vertex (simplified): `(3/4) · v + (1/4) · avg(neighbors)`
fn update_vertex_loop(
    v: &na::Point3<f32>,
    neighbors: &[u32],
    vertices: &[na::Point3<f32>],
    is_boundary: bool,
) -> na::Point3<f32> {
    let n = neighbors.len();
    if n == 0 {
        return *v;
    }

    // Accumulate neighbor positions
    let mut neighbor_sum = na::Point3::origin();
    for &nb in neighbors {
        neighbor_sum = add_points(&neighbor_sum, &vertices[nb as usize]);
    }

    if is_boundary {
        // Simplified boundary rule: 3/4 * v + 1/4 * avg(neighbors)
        let avg = scale_point(&neighbor_sum, 1.0 / n as f32);
        let part_v = scale_point(v, 0.75);
        let part_nb = scale_point(&avg, 0.25);
        add_points(&part_v, &part_nb)
    } else {
        // Interior Loop rule
        let beta = compute_loop_vertex_weight(n);
        let n_beta = beta * n as f32;
        // (1 - n*beta) * v
        let part_v = scale_point(v, 1.0 - n_beta);
        // beta * sum(neighbors)
        let part_nb = scale_point(&neighbor_sum, beta);
        add_points(&part_v, &part_nb)
    }
}

/// Compute the Loop edge-midpoint vertex position.
///
/// - Interior edge (2 adjacent faces): `3/8 · (v0 + v1) + 1/8 · (adj0 + adj1)`
/// - Boundary edge (1 adjacent face):  `(v0 + v1) / 2`
fn compute_edge_midpoint_loop(
    v0: &na::Point3<f32>,
    v1: &na::Point3<f32>,
    opposite_vertices: &[na::Point3<f32>],
) -> na::Point3<f32> {
    match opposite_vertices {
        [adj0, adj1] => {
            // Interior edge: 3/8*(v0+v1) + 1/8*(adj0+adj1)
            let edge_sum = add_points(v0, v1);
            let adj_sum = add_points(adj0, adj1);
            let part_edge = scale_point(&edge_sum, 3.0 / 8.0);
            let part_adj = scale_point(&adj_sum, 1.0 / 8.0);
            add_points(&part_edge, &part_adj)
        }
        _ => {
            // Boundary edge or degenerate: simple midpoint
            midpoint_vertex(v0, v1)
        }
    }
}

/// Recompute per-vertex normals from vertex positions and faces.
///
/// Accumulates area-weighted face normals for each vertex and normalizes.
/// This function uses `na::Point3<f32>` / `na::Vector3<f32>` directly,
/// distinguishing it from `compute_vertex_normals` in `multiresolution`
/// which operates on `&[[f32; 3]]`.
#[must_use]
pub fn recompute_mesh_normals(
    vertices: &[na::Point3<f32>],
    faces: &[[u32; 3]],
) -> Vec<na::Vector3<f32>> {
    let mut normals = vec![na::Vector3::zeros(); vertices.len()];

    for face in faces {
        let [i0, i1, i2] = [face[0] as usize, face[1] as usize, face[2] as usize];
        if i0 >= vertices.len() || i1 >= vertices.len() || i2 >= vertices.len() {
            continue;
        }
        let v0 = &vertices[i0];
        let v1 = &vertices[i1];
        let v2 = &vertices[i2];
        let edge1 = v1 - v0;
        let edge2 = v2 - v0;
        let face_normal = edge1.cross(&edge2); // magnitude ∝ area
        normals[i0] += face_normal;
        normals[i1] += face_normal;
        normals[i2] += face_normal;
    }

    for n in &mut normals {
        let len = n.norm();
        if len > 1e-10 {
            *n /= len;
        }
    }

    normals
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

/// Validate that `mesh` is suitable for Loop subdivision.
///
/// Checks:
/// - At least one vertex
/// - At least one face
/// - All face indices are in range
/// - All faces reference 3 distinct indices (non-degenerate)
///
/// # Errors
///
/// Returns a [`SubdivisionError`] if any check fails.
pub fn validate_mesh_for_subdivision(mesh: &Mesh) -> Result<(), SubdivisionError> {
    if mesh.vertices.is_empty() {
        return Err(SubdivisionError::EmptyMesh);
    }
    if mesh.faces.is_empty() {
        return Err(SubdivisionError::NoFaces);
    }

    let n = mesh.vertices.len();
    for (idx, face) in mesh.faces.iter().enumerate() {
        // FLAME mesh always has [u32; 3] faces, so this check is for the API contract
        // (all faces have exactly 3 vertices by type), but we still validate indices.
        for &vi in face {
            if vi as usize >= n {
                return Err(SubdivisionError::VertexIndexOutOfRange {
                    idx: vi as usize,
                    n,
                });
            }
        }

        // Check for degenerate faces (all three vertices identical)
        if face[0] == face[1] && face[1] == face[2] {
            return Err(SubdivisionError::DegenerateFace { idx });
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Vertex-count estimator
// ---------------------------------------------------------------------------

/// Estimate the vertex count after `levels` rounds of Loop subdivision.
///
/// Uses the Euler-formula approximation:
/// - `V' = V + E`  where `E ≈ 3 · F / 2` for a closed mesh
/// - `F' = 4 · F`
///
/// Applied iteratively for each level.
#[must_use]
pub fn estimate_subdivided_vertex_count(n_verts: usize, n_faces: usize, levels: usize) -> usize {
    let mut v = n_verts;
    let mut f = n_faces;
    for _ in 0..levels {
        // Number of unique edges ≈ 3*F/2 for closed mesh (Euler: V - E + F = 2)
        // New vertices = original + edge midpoints
        let e_approx = (3 * f).div_ceil(2); // ceiling division to be conservative
        v += e_approx;
        f *= 4;
    }
    v
}

// ---------------------------------------------------------------------------
// Single subdivision step
// ---------------------------------------------------------------------------

/// Apply one level of Loop subdivision to `mesh`.
///
/// # Errors
///
/// Returns [`SubdivisionError`] if the mesh is invalid.
pub fn subdivide_once(mesh: &Mesh, config: &SubdivisionConfig) -> Result<Mesh, SubdivisionError> {
    validate_mesh_for_subdivision(mesh)?;

    let n_orig_verts = mesh.vertices.len();
    let faces = &mesh.faces;

    // ------------------------------------------------------------------
    // Step 1: Build edge → opposite-vertex map
    // edge_key → Vec of opposite vertex indices (one per adjacent face)
    // ------------------------------------------------------------------
    let mut edge_opposite: HashMap<(u32, u32), Vec<u32>> = HashMap::new();

    for face in faces {
        let [v0, v1, v2] = *face;
        // Edge (v0,v1) opposite v2
        edge_opposite.entry(edge_key(v0, v1)).or_default().push(v2);
        // Edge (v1,v2) opposite v0
        edge_opposite.entry(edge_key(v1, v2)).or_default().push(v0);
        // Edge (v0,v2) opposite v1
        edge_opposite.entry(edge_key(v0, v2)).or_default().push(v1);
    }

    // ------------------------------------------------------------------
    // Step 2: Compute edge midpoints and assign indices
    // ------------------------------------------------------------------
    // New vertex index for edge midpoints starts at n_orig_verts
    let mut edge_midpoint_idx: HashMap<(u32, u32), u32> = HashMap::new();
    let mut new_vertices: Vec<na::Point3<f32>> = mesh.vertices.clone();

    for (&ek, opposites) in &edge_opposite {
        let (a, b) = ek;
        let v_a = &mesh.vertices[a as usize];
        let v_b = &mesh.vertices[b as usize];

        // Collect opposite vertex positions (at most 2 for manifold mesh)
        let opp_positions: Vec<na::Point3<f32>> = opposites
            .iter()
            .take(2)
            .map(|&oi| mesh.vertices[oi as usize])
            .collect();

        let midpoint = compute_edge_midpoint_loop(v_a, v_b, &opp_positions);

        let new_idx = new_vertices.len() as u32;
        new_vertices.push(midpoint);
        edge_midpoint_idx.insert(ek, new_idx);
    }

    // ------------------------------------------------------------------
    // Step 3: Optionally update original vertex positions (Loop rule)
    // ------------------------------------------------------------------
    if config.update_vertices {
        let boundary_edges = identify_boundary_edges(faces);
        let vertex_neighbors = build_vertex_neighbors(faces, n_orig_verts);

        for vi in 0..n_orig_verts {
            let v = &mesh.vertices[vi];
            let neighbors = &vertex_neighbors[vi];
            let is_boundary = is_boundary_vertex(vi as u32, &boundary_edges);
            let updated = update_vertex_loop(v, neighbors, &mesh.vertices, is_boundary);
            new_vertices[vi] = updated;
        }
    }

    // ------------------------------------------------------------------
    // Step 4: Build new face list
    // For each original face [v0, v1, v2], produce 4 child faces:
    //   [v0,  e01, e20]
    //   [v1,  e12, e01]
    //   [v2,  e20, e12]
    //   [e01, e12, e20]
    // ------------------------------------------------------------------
    let mut new_faces: Vec<[u32; 3]> = Vec::with_capacity(faces.len() * 4);

    for face in faces {
        let [v0, v1, v2] = *face;

        let e01 = *edge_midpoint_idx.get(&edge_key(v0, v1)).ok_or(
            SubdivisionError::VertexIndexOutOfRange {
                idx: v0 as usize,
                n: n_orig_verts,
            },
        )?;
        let e12 = *edge_midpoint_idx.get(&edge_key(v1, v2)).ok_or(
            SubdivisionError::VertexIndexOutOfRange {
                idx: v1 as usize,
                n: n_orig_verts,
            },
        )?;
        let e20 = *edge_midpoint_idx.get(&edge_key(v2, v0)).ok_or(
            SubdivisionError::VertexIndexOutOfRange {
                idx: v2 as usize,
                n: n_orig_verts,
            },
        )?;

        new_faces.push([v0, e01, e20]);
        new_faces.push([v1, e12, e01]);
        new_faces.push([v2, e20, e12]);
        new_faces.push([e01, e12, e20]);
    }

    // ------------------------------------------------------------------
    // Step 5: Assemble output Mesh
    // ------------------------------------------------------------------
    let mut out_normals = vec![na::Vector3::zeros(); new_vertices.len()];
    if config.recompute_normals {
        out_normals = recompute_mesh_normals(&new_vertices, &new_faces);
    }

    Ok(Mesh {
        normals: out_normals,
        vertices: new_vertices,
        faces: new_faces,
        uv_coords: Vec::new(),
    })
}

// ---------------------------------------------------------------------------
// Public multi-level subdivision
// ---------------------------------------------------------------------------

/// Apply Loop subdivision to `mesh` for `config.levels` rounds.
///
/// Checks the estimated vertex count against `config.max_vertices` before
/// each level and aborts with [`SubdivisionError::TooManyVertices`] if the
/// limit would be exceeded.
///
/// # Errors
///
/// Returns [`SubdivisionError`] if the mesh is invalid or the vertex limit
/// would be exceeded.
pub fn subdivide_mesh(
    mesh: &Mesh,
    config: &SubdivisionConfig,
) -> Result<SubdivisionResult, SubdivisionError> {
    validate_mesh_for_subdivision(mesh)?;

    let n_original_vertices = mesh.vertices.len();
    let n_original_faces = mesh.faces.len();

    let mut current = mesh.clone();
    let mut levels_applied = 0usize;

    for level in 1..=config.levels {
        // Safety check before each level
        let estimated =
            estimate_subdivided_vertex_count(current.vertices.len(), current.faces.len(), 1);
        if estimated > config.max_vertices {
            return Err(SubdivisionError::TooManyVertices {
                level,
                estimated,
                max: config.max_vertices,
            });
        }

        current = subdivide_once(&current, config)?;
        levels_applied += 1;
    }

    let n_new_vertices = current.vertices.len() - n_original_vertices;
    let n_new_faces = current.faces.len();

    Ok(SubdivisionResult {
        n_original_vertices,
        n_new_vertices,
        n_original_faces,
        n_new_faces,
        levels_applied,
        mesh: current,
    })
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Compute all edge lengths for a mesh (one per directed edge of each face).
fn all_edge_lengths(mesh: &Mesh) -> Vec<f32> {
    let mut lengths = Vec::with_capacity(mesh.faces.len() * 3);
    for face in &mesh.faces {
        let [i0, i1, i2] = [face[0] as usize, face[1] as usize, face[2] as usize];
        if i0 < mesh.vertices.len() && i1 < mesh.vertices.len() && i2 < mesh.vertices.len() {
            lengths.push((mesh.vertices[i1] - mesh.vertices[i0]).norm());
            lengths.push((mesh.vertices[i2] - mesh.vertices[i1]).norm());
            lengths.push((mesh.vertices[i0] - mesh.vertices[i2]).norm());
        }
    }
    lengths
}

/// Compute the mean edge length of a mesh.
///
/// Each triangle contributes 3 edges (edges shared between triangles are
/// counted once per adjacent face — this is acceptable for computing the mean).
#[must_use]
pub fn compute_mean_edge_length(mesh: &Mesh) -> f32 {
    let lengths = all_edge_lengths(mesh);
    if lengths.is_empty() {
        return 0.0;
    }
    lengths.iter().sum::<f32>() / lengths.len() as f32
}

/// Compute statistics about the subdivided mesh relative to the original.
#[must_use]
pub fn compute_subdivision_stats(original: &Mesh, result: &SubdivisionResult) -> SubdivisionStats {
    let lengths = all_edge_lengths(&result.mesh);

    let mean_edge_length = if lengths.is_empty() {
        0.0
    } else {
        lengths.iter().sum::<f32>() / lengths.len() as f32
    };

    let max_edge_length = lengths.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let min_edge_length = lengths.iter().copied().fold(f32::INFINITY, f32::min);

    SubdivisionStats {
        original_vertices: original.vertices.len(),
        final_vertices: result.mesh.vertices.len(),
        original_faces: original.faces.len(),
        final_faces: result.mesh.faces.len(),
        mean_edge_length,
        max_edge_length: if max_edge_length.is_infinite() {
            0.0
        } else {
            max_edge_length
        },
        min_edge_length: if min_edge_length.is_infinite() {
            0.0
        } else {
            min_edge_length
        },
    }
}

/// Format a subdivision result as a concise human-readable string.
#[must_use]
pub fn format_subdivision_result(result: &SubdivisionResult) -> String {
    let orig_v = result.n_original_vertices;
    let final_v = result.mesh.vertices.len();
    let orig_f = result.n_original_faces;
    let final_f = result.mesh.faces.len();
    let lvl = result.levels_applied;
    format!(
        "Loop subdivision [{lvl} level{}]: V: {orig_v}→{final_v}, F: {orig_f}→{final_f}",
        if lvl == 1 { "" } else { "s" }
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra as na;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Build a single equilateral-ish triangle in the XY plane.
    fn single_triangle() -> Mesh {
        let vertices = vec![
            na::Point3::new(0.0f32, 0.0, 0.0),
            na::Point3::new(1.0f32, 0.0, 0.0),
            na::Point3::new(0.5f32, 0.866_025, 0.0),
        ];
        let faces = vec![[0, 1, 2]];
        Mesh::new(vertices, faces)
    }

    /// Build a two-triangle mesh (a unit square split along the diagonal).
    fn two_triangle_mesh() -> Mesh {
        let vertices = vec![
            na::Point3::new(0.0f32, 0.0, 0.0),
            na::Point3::new(1.0f32, 0.0, 0.0),
            na::Point3::new(1.0f32, 1.0, 0.0),
            na::Point3::new(0.0f32, 1.0, 0.0),
        ];
        let faces = vec![[0, 1, 2], [0, 2, 3]];
        Mesh::new(vertices, faces)
    }

    /// Build a regular tetrahedron mesh (4 triangular faces, closed).
    fn tetrahedron_mesh() -> Mesh {
        let vertices = vec![
            na::Point3::new(0.0f32, 0.0, 0.0),
            na::Point3::new(1.0f32, 0.0, 0.0),
            na::Point3::new(0.5f32, 0.866, 0.0),
            na::Point3::new(0.5f32, 0.289, 0.816),
        ];
        let faces = vec![[0, 1, 2], [0, 1, 3], [1, 2, 3], [0, 2, 3]];
        Mesh::new(vertices, faces)
    }

    // -----------------------------------------------------------------------
    // 1. estimate_subdivided_vertex_count
    // -----------------------------------------------------------------------

    #[test]
    fn test_estimate_vertex_count_zero_levels() {
        // 0 levels → no change
        let v = estimate_subdivided_vertex_count(10, 16, 0);
        assert_eq!(v, 10);
    }

    #[test]
    fn test_estimate_vertex_count_one_level_faces_quadrupled() {
        // After 1 level: F' = 4 * F
        // V' > V (new edge midpoints added)
        let n_v = 5;
        let n_f = 6;
        let result = estimate_subdivided_vertex_count(n_v, n_f, 1);
        // F = 6, E ≈ 9, V' = 5 + 9 = 14
        assert!(result > n_v, "new vertex count should exceed original");
    }

    #[test]
    fn test_estimate_vertex_count_single_triangle_one_level() {
        // Triangle: V=3, F=1, E≈3/2 ≈ 2 (rounded up)
        let v = estimate_subdivided_vertex_count(3, 1, 1);
        // E_approx = ceil(3/2) = 2, so V' = 3 + 2 = 5
        assert!(v > 3, "should add at least one vertex");
    }

    #[test]
    fn test_estimate_vertex_count_grows_monotonically() {
        let n_v = 100;
        let n_f = 180;
        let v1 = estimate_subdivided_vertex_count(n_v, n_f, 1);
        let v2 = estimate_subdivided_vertex_count(n_v, n_f, 2);
        let v3 = estimate_subdivided_vertex_count(n_v, n_f, 3);
        assert!(v1 < v2 && v2 < v3, "estimate grows with more levels");
    }

    // -----------------------------------------------------------------------
    // 2. validate_mesh_for_subdivision
    // -----------------------------------------------------------------------

    #[test]
    fn test_validate_empty_vertices() {
        let mesh = Mesh::new(vec![], vec![]);
        assert!(matches!(
            validate_mesh_for_subdivision(&mesh),
            Err(SubdivisionError::EmptyMesh)
        ));
    }

    #[test]
    fn test_validate_no_faces() {
        let vertices = vec![na::Point3::new(0.0f32, 0.0, 0.0)];
        let mesh = Mesh {
            normals: vec![na::Vector3::zeros()],
            vertices,
            faces: vec![],
            uv_coords: vec![],
        };
        assert!(matches!(
            validate_mesh_for_subdivision(&mesh),
            Err(SubdivisionError::NoFaces)
        ));
    }

    #[test]
    fn test_validate_out_of_range_index() {
        let vertices = vec![
            na::Point3::new(0.0f32, 0.0, 0.0),
            na::Point3::new(1.0f32, 0.0, 0.0),
        ];
        // Face references vertex 5 which doesn't exist
        let mesh = Mesh {
            normals: vec![na::Vector3::zeros(); 2],
            vertices,
            faces: vec![[0, 1, 5]],
            uv_coords: vec![],
        };
        assert!(matches!(
            validate_mesh_for_subdivision(&mesh),
            Err(SubdivisionError::VertexIndexOutOfRange { .. })
        ));
    }

    #[test]
    fn test_validate_degenerate_face() {
        let vertices = vec![
            na::Point3::new(0.0f32, 0.0, 0.0),
            na::Point3::new(1.0f32, 0.0, 0.0),
        ];
        // All three vertex indices are the same → degenerate
        let mesh = Mesh {
            normals: vec![na::Vector3::zeros(); 2],
            vertices,
            faces: vec![[0, 0, 0]],
            uv_coords: vec![],
        };
        assert!(matches!(
            validate_mesh_for_subdivision(&mesh),
            Err(SubdivisionError::DegenerateFace { idx: 0 })
        ));
    }

    #[test]
    fn test_validate_valid_mesh_ok() {
        let mesh = single_triangle();
        assert!(validate_mesh_for_subdivision(&mesh).is_ok());
    }

    // -----------------------------------------------------------------------
    // 3. midpoint_vertex helper
    // -----------------------------------------------------------------------

    #[test]
    fn test_midpoint_vertex_known_points() {
        let a = na::Point3::new(0.0f32, 0.0, 0.0);
        let b = na::Point3::new(2.0f32, 4.0, 6.0);
        let mid = midpoint_vertex(&a, &b);
        assert!((mid.x - 1.0).abs() < 1e-6);
        assert!((mid.y - 2.0).abs() < 1e-6);
        assert!((mid.z - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_midpoint_vertex_same_point() {
        let a = na::Point3::new(3.0f32, 5.0, 7.0);
        let mid = midpoint_vertex(&a, &a);
        assert!((mid.x - a.x).abs() < 1e-6);
        assert!((mid.y - a.y).abs() < 1e-6);
        assert!((mid.z - a.z).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // 4. compute_loop_vertex_weight
    // -----------------------------------------------------------------------

    #[test]
    fn test_loop_weight_n3() {
        let beta = compute_loop_vertex_weight(3);
        assert!((beta - 3.0 / 16.0).abs() < 1e-7);
    }

    #[test]
    fn test_loop_weight_n6() {
        let beta = compute_loop_vertex_weight(6);
        // 3 / (8 * 6) = 3/48 = 1/16
        assert!((beta - 3.0 / 48.0).abs() < 1e-7);
    }

    #[test]
    fn test_loop_weight_n4() {
        let beta = compute_loop_vertex_weight(4);
        // 3 / (8 * 4) = 3/32
        assert!((beta - 3.0 / 32.0).abs() < 1e-7);
    }

    #[test]
    fn test_loop_weight_n3_is_larger_than_n6() {
        // Higher connectivity → smaller beta
        assert!(compute_loop_vertex_weight(3) > compute_loop_vertex_weight(6));
    }

    // -----------------------------------------------------------------------
    // 5. build_vertex_neighbors
    // -----------------------------------------------------------------------

    #[test]
    fn test_build_vertex_neighbors_single_triangle() {
        // Triangle [0,1,2]: each vertex connects to the other 2
        let faces = vec![[0u32, 1, 2]];
        let nb = build_vertex_neighbors(&faces, 3);
        assert_eq!(nb.len(), 3);
        // Each vertex should have exactly 2 neighbors
        for (i, nbrs) in nb.iter().enumerate().take(3) {
            assert_eq!(nbrs.len(), 2, "vertex {i} should have 2 neighbors");
        }
    }

    #[test]
    fn test_build_vertex_neighbors_two_triangles() {
        // Square: [0,1,2], [0,2,3]
        let faces = vec![[0u32, 1, 2], [0, 2, 3]];
        let nb = build_vertex_neighbors(&faces, 4);
        // Vertex 2 is shared by both faces and should connect to 0,1,3
        let mut nb2 = nb[2].clone();
        nb2.sort_unstable();
        assert_eq!(nb2, vec![0, 1, 3]);
    }

    #[test]
    fn test_build_vertex_neighbors_isolated_vertex() {
        // Vertex 3 is isolated (not in any face)
        let faces = vec![[0u32, 1, 2]];
        let nb = build_vertex_neighbors(&faces, 4);
        assert!(nb[3].is_empty(), "isolated vertex should have no neighbors");
    }

    // -----------------------------------------------------------------------
    // 6. identify_boundary_edges
    // -----------------------------------------------------------------------

    #[test]
    fn test_boundary_edges_single_triangle() {
        // All 3 edges of a lone triangle are boundary edges
        let faces = vec![[0u32, 1, 2]];
        let be = identify_boundary_edges(&faces);
        assert_eq!(be.len(), 3);
        assert!(be.contains(&edge_key(0, 1)));
        assert!(be.contains(&edge_key(1, 2)));
        assert!(be.contains(&edge_key(0, 2)));
    }

    #[test]
    fn test_boundary_edges_two_triangles() {
        // [0,1,2] and [0,2,3]: edge (0,2) is shared → interior
        let faces = vec![[0u32, 1, 2], [0, 2, 3]];
        let be = identify_boundary_edges(&faces);
        // (0,2) should NOT be in boundary set
        assert!(!be.contains(&edge_key(0, 2)));
        // The other 4 edges should be boundary
        assert_eq!(be.len(), 4);
    }

    #[test]
    fn test_boundary_edges_closed_tetrahedron() {
        // Closed solid: every edge appears in exactly 2 faces → no boundary edges
        let mesh = tetrahedron_mesh();
        let be = identify_boundary_edges(&mesh.faces);
        assert_eq!(be.len(), 0, "closed tetrahedron has no boundary edges");
    }

    // -----------------------------------------------------------------------
    // 7. is_boundary_vertex
    // -----------------------------------------------------------------------

    #[test]
    fn test_is_boundary_vertex_lone_triangle() {
        // All vertices of a lone triangle are boundary
        let faces = vec![[0u32, 1, 2]];
        let be = identify_boundary_edges(&faces);
        assert!(is_boundary_vertex(0, &be));
        assert!(is_boundary_vertex(1, &be));
        assert!(is_boundary_vertex(2, &be));
    }

    #[test]
    fn test_is_boundary_vertex_interior_vertex() {
        // [0,1,2], [0,2,3], [0,3,1]: vertex 0 is surrounded, so interior
        let faces = vec![[0u32, 1, 2], [0, 2, 3], [0, 3, 1]];
        let be = identify_boundary_edges(&faces);
        // vertex 0 is not on any boundary edge (all its edges appear twice)
        assert!(!is_boundary_vertex(0, &be));
    }

    // -----------------------------------------------------------------------
    // 8. compute_edge_midpoint_loop
    // -----------------------------------------------------------------------

    #[test]
    fn test_edge_midpoint_loop_boundary_edge() {
        // Boundary edge → simple midpoint
        let v0 = na::Point3::new(0.0f32, 0.0, 0.0);
        let v1 = na::Point3::new(2.0f32, 0.0, 0.0);
        let mid = compute_edge_midpoint_loop(&v0, &v1, &[]);
        assert!((mid.x - 1.0).abs() < 1e-6);
        assert!((mid.y).abs() < 1e-6);
    }

    #[test]
    fn test_edge_midpoint_loop_interior_edge() {
        // Interior edge: 3/8*(v0+v1) + 1/8*(adj0+adj1)
        let v0 = na::Point3::new(0.0f32, 0.0, 0.0);
        let v1 = na::Point3::new(2.0f32, 0.0, 0.0);
        let adj0 = na::Point3::new(1.0f32, 1.0, 0.0);
        let adj1 = na::Point3::new(1.0f32, -1.0, 0.0);
        let mid = compute_edge_midpoint_loop(&v0, &v1, &[adj0, adj1]);
        // Expected x: 3/8*(0+2) + 1/8*(1+1) = 3/4 + 1/4 = 1.0
        // Expected y: 3/8*(0+0) + 1/8*(1+(-1)) = 0
        assert!((mid.x - 1.0).abs() < 1e-6, "x should be 1.0, got {}", mid.x);
        assert!(mid.y.abs() < 1e-6, "y should be 0.0, got {}", mid.y);
    }

    // -----------------------------------------------------------------------
    // 9. update_vertex_loop
    // -----------------------------------------------------------------------

    #[test]
    fn test_update_vertex_loop_interior() {
        let vertices = vec![
            na::Point3::new(0.0f32, 0.0, 0.0), // v0 (updating this)
            na::Point3::new(1.0f32, 0.0, 0.0),
            na::Point3::new(0.0f32, 1.0, 0.0),
            na::Point3::new(-1.0f32, 0.0, 0.0),
        ];
        let neighbors = vec![1u32, 2, 3];
        let v = &vertices[0];
        let updated = update_vertex_loop(v, &neighbors, &vertices, false);
        // Beta for n=3: 3/16; n*beta = 9/16
        // new_v = (1 - 9/16)*[0,0,0] + 3/16*(1+0-1, 0+1+0, 0) = 3/16*(0, 1, 0)
        // = [0, 3/16, 0] = [0, 0.1875, 0]
        assert!(updated.x.abs() < 1e-6, "x = {}", updated.x);
        assert!((updated.y - 0.1875).abs() < 1e-5, "y = {}", updated.y);
    }

    #[test]
    fn test_update_vertex_loop_boundary_moves_toward_neighbors() {
        let vertices = vec![
            na::Point3::new(0.0f32, 0.0, 0.0), // v0 (updating)
            na::Point3::new(4.0f32, 0.0, 0.0), // neighbor
        ];
        let neighbors = vec![1u32];
        let v = &vertices[0];
        let updated = update_vertex_loop(v, &neighbors, &vertices, true);
        // 3/4*[0,0,0] + 1/4*[4,0,0] = [1, 0, 0]
        assert!((updated.x - 1.0).abs() < 1e-5, "x = {}", updated.x);
    }

    #[test]
    fn test_update_vertex_loop_no_neighbors_returns_original() {
        let vertices = vec![na::Point3::new(3.0f32, 7.0, 2.0)];
        let v = &vertices[0];
        let updated = update_vertex_loop(v, &[], &vertices, false);
        assert!((updated.x - v.x).abs() < 1e-7);
        assert!((updated.y - v.y).abs() < 1e-7);
    }

    // -----------------------------------------------------------------------
    // 10. recompute_mesh_normals
    // -----------------------------------------------------------------------

    #[test]
    fn test_recompute_mesh_normals_flat_mesh_all_z() {
        // All faces in XY plane → all normals should point in +Z
        let mesh = two_triangle_mesh();
        let normals = recompute_mesh_normals(&mesh.vertices, &mesh.faces);
        for (i, n) in normals.iter().enumerate() {
            assert!(n.z > 0.9, "vertex {i} normal z = {}", n.z);
        }
    }

    #[test]
    fn test_recompute_mesh_normals_normalized() {
        let mesh = tetrahedron_mesh();
        let normals = recompute_mesh_normals(&mesh.vertices, &mesh.faces);
        for (i, n) in normals.iter().enumerate() {
            let len = n.norm();
            assert!(
                (len - 1.0).abs() < 1e-5 || len < 1e-10,
                "normal {i} not unit: len = {len}"
            );
        }
    }

    #[test]
    fn test_recompute_mesh_normals_count_matches_vertices() {
        let mesh = single_triangle();
        let normals = recompute_mesh_normals(&mesh.vertices, &mesh.faces);
        assert_eq!(normals.len(), mesh.vertices.len());
    }

    // -----------------------------------------------------------------------
    // 11. subdivide_once
    // -----------------------------------------------------------------------

    #[test]
    fn test_subdivide_once_single_triangle_face_count() {
        let mesh = single_triangle();
        let config = SubdivisionConfig::default();
        let subdivided = subdivide_once(&mesh, &config).expect("subdivision should succeed");
        assert_eq!(subdivided.faces.len(), 4, "1 triangle → 4 triangles");
    }

    #[test]
    fn test_subdivide_once_single_triangle_vertex_count() {
        let mesh = single_triangle();
        let config = SubdivisionConfig::default();
        let subdivided = subdivide_once(&mesh, &config).expect("subdivision should succeed");
        // Original 3 vertices + 3 edge midpoints = 6
        assert_eq!(subdivided.vertices.len(), 6, "should have 6 vertices");
    }

    #[test]
    fn test_subdivide_once_two_triangles_face_count() {
        let mesh = two_triangle_mesh();
        let config = SubdivisionConfig::default();
        let subdivided = subdivide_once(&mesh, &config).expect("subdivision should succeed");
        assert_eq!(subdivided.faces.len(), 8, "2 triangles → 8 triangles");
    }

    #[test]
    fn test_subdivide_once_no_vertex_update() {
        let mesh = single_triangle();
        let config = SubdivisionConfig {
            update_vertices: false,
            ..Default::default()
        };
        let subdivided = subdivide_once(&mesh, &config).expect("subdivision should succeed");
        // Original vertex positions should be preserved exactly
        for (i, orig) in mesh.vertices.iter().enumerate() {
            let new_v = &subdivided.vertices[i];
            assert!((new_v.x - orig.x).abs() < 1e-7, "vertex {i} x changed");
            assert!((new_v.y - orig.y).abs() < 1e-7, "vertex {i} y changed");
            assert!((new_v.z - orig.z).abs() < 1e-7, "vertex {i} z changed");
        }
    }

    #[test]
    fn test_subdivide_once_all_face_indices_valid() {
        let mesh = two_triangle_mesh();
        let config = SubdivisionConfig::default();
        let subdivided = subdivide_once(&mesh, &config).expect("subdivision should succeed");
        let n_verts = subdivided.vertices.len();
        for face in &subdivided.faces {
            for &idx in face {
                assert!((idx as usize) < n_verts, "face index {idx} out of bounds");
            }
        }
    }

    #[test]
    fn test_subdivide_once_empty_mesh_fails() {
        let mesh = Mesh::new(vec![], vec![]);
        let config = SubdivisionConfig::default();
        assert!(subdivide_once(&mesh, &config).is_err());
    }

    // -----------------------------------------------------------------------
    // 12. subdivide_mesh (multi-level)
    // -----------------------------------------------------------------------

    #[test]
    fn test_subdivide_mesh_1_level() {
        let mesh = two_triangle_mesh();
        let config = SubdivisionConfig {
            levels: 1,
            ..Default::default()
        };
        let result = subdivide_mesh(&mesh, &config).expect("should succeed");
        assert_eq!(result.levels_applied, 1);
        assert_eq!(result.n_original_faces, 2);
        assert_eq!(result.n_new_faces, 8);
    }

    #[test]
    fn test_subdivide_mesh_2_levels_face_count() {
        let mesh = two_triangle_mesh();
        let config = SubdivisionConfig {
            levels: 2,
            ..Default::default()
        };
        let result = subdivide_mesh(&mesh, &config).expect("should succeed");
        // 2 faces × 4^2 = 32 faces
        assert_eq!(result.mesh.faces.len(), 32);
    }

    #[test]
    fn test_subdivide_mesh_face_count_4_power_n() {
        // Face count after N levels = 4^N * original
        let mesh = single_triangle();
        for n in 1usize..=3 {
            let config = SubdivisionConfig {
                levels: n,
                ..Default::default()
            };
            let result = subdivide_mesh(&mesh, &config).expect("subdivide should succeed");
            let expected = 4usize.pow(n as u32);
            assert_eq!(
                result.mesh.faces.len(),
                expected,
                "N={n} levels: expected {expected} faces, got {}",
                result.mesh.faces.len()
            );
        }
    }

    #[test]
    fn test_subdivide_mesh_too_many_vertices_error() {
        let mesh = two_triangle_mesh();
        let config = SubdivisionConfig {
            levels: 1,
            max_vertices: 3, // impossibly small
            ..Default::default()
        };
        assert!(matches!(
            subdivide_mesh(&mesh, &config),
            Err(SubdivisionError::TooManyVertices { .. })
        ));
    }

    #[test]
    fn test_subdivide_mesh_levels_applied() {
        let mesh = single_triangle();
        let config = SubdivisionConfig {
            levels: 2,
            ..Default::default()
        };
        let result = subdivide_mesh(&mesh, &config).expect("should succeed");
        assert_eq!(result.levels_applied, 2);
    }

    #[test]
    fn test_subdivide_mesh_original_counts_preserved() {
        let mesh = two_triangle_mesh();
        let n_v = mesh.vertices.len();
        let n_f = mesh.faces.len();
        let config = SubdivisionConfig::default();
        let result = subdivide_mesh(&mesh, &config).expect("should succeed");
        assert_eq!(result.n_original_vertices, n_v);
        assert_eq!(result.n_original_faces, n_f);
    }

    // -----------------------------------------------------------------------
    // 13. compute_subdivision_stats
    // -----------------------------------------------------------------------

    #[test]
    fn test_compute_subdivision_stats_vertex_face_counts() {
        let mesh = two_triangle_mesh();
        let config = SubdivisionConfig::default();
        let result = subdivide_mesh(&mesh, &config).expect("should succeed");
        let stats = compute_subdivision_stats(&mesh, &result);
        assert_eq!(stats.original_vertices, mesh.vertices.len());
        assert_eq!(stats.final_vertices, result.mesh.vertices.len());
        assert_eq!(stats.original_faces, mesh.faces.len());
        assert_eq!(stats.final_faces, result.mesh.faces.len());
    }

    #[test]
    fn test_compute_subdivision_stats_edge_lengths_non_negative() {
        let mesh = single_triangle();
        let config = SubdivisionConfig::default();
        let result = subdivide_mesh(&mesh, &config).expect("should succeed");
        let stats = compute_subdivision_stats(&mesh, &result);
        assert!(stats.mean_edge_length >= 0.0);
        assert!(stats.min_edge_length >= 0.0);
        assert!(stats.max_edge_length >= stats.min_edge_length);
    }

    #[test]
    fn test_compute_subdivision_stats_final_greater_than_original() {
        let mesh = two_triangle_mesh();
        let config = SubdivisionConfig::default();
        let result = subdivide_mesh(&mesh, &config).expect("should succeed");
        let stats = compute_subdivision_stats(&mesh, &result);
        assert!(stats.final_vertices > stats.original_vertices);
        assert!(stats.final_faces > stats.original_faces);
    }

    // -----------------------------------------------------------------------
    // 14. compute_mean_edge_length
    // -----------------------------------------------------------------------

    #[test]
    fn test_compute_mean_edge_length_equilateral_triangle() {
        // Unit equilateral triangle: all edges = 1.0
        let mesh = Mesh::new(
            vec![
                na::Point3::new(0.0f32, 0.0, 0.0),
                na::Point3::new(1.0f32, 0.0, 0.0),
                na::Point3::new(0.5f32, 0.866_025, 0.0),
            ],
            vec![[0, 1, 2]],
        );
        let mean = compute_mean_edge_length(&mesh);
        assert!(
            (mean - 1.0).abs() < 1e-4,
            "equilateral mean edge ≈ 1, got {mean}"
        );
    }

    #[test]
    fn test_compute_mean_edge_length_empty_faces() {
        let mesh = Mesh {
            vertices: vec![na::Point3::new(0.0f32, 0.0, 0.0)],
            normals: vec![na::Vector3::zeros()],
            faces: vec![],
            uv_coords: vec![],
        };
        let mean = compute_mean_edge_length(&mesh);
        assert_eq!(mean, 0.0);
    }

    #[test]
    fn test_compute_mean_edge_length_subdivided_smaller() {
        // After subdivision, edge lengths should be roughly half the original
        let mesh = single_triangle();
        let orig_mean = compute_mean_edge_length(&mesh);
        let config = SubdivisionConfig {
            update_vertices: false,
            ..Default::default()
        };
        let result = subdivide_mesh(&mesh, &config).expect("should succeed");
        let new_mean = compute_mean_edge_length(&result.mesh);
        // Midpoint subdivision halves edges, so new_mean ≈ orig_mean / 2
        assert!(
            new_mean < orig_mean * 0.9,
            "edge length should decrease after subdivision"
        );
    }

    // -----------------------------------------------------------------------
    // 15. format_subdivision_result
    // -----------------------------------------------------------------------

    #[test]
    fn test_format_subdivision_result_non_empty() {
        let mesh = two_triangle_mesh();
        let config = SubdivisionConfig::default();
        let result = subdivide_mesh(&mesh, &config).expect("should succeed");
        let s = format_subdivision_result(&result);
        assert!(!s.is_empty(), "formatted string should be non-empty");
        assert!(
            s.contains("Loop subdivision"),
            "should mention loop subdivision"
        );
    }

    #[test]
    fn test_format_subdivision_result_contains_vertex_counts() {
        let mesh = two_triangle_mesh();
        let config = SubdivisionConfig::default();
        let result = subdivide_mesh(&mesh, &config).expect("should succeed");
        let s = format_subdivision_result(&result);
        // Should contain the arrow symbol
        assert!(s.contains('→'), "should contain arrow symbol");
    }

    #[test]
    fn test_format_subdivision_result_plural_levels() {
        let mesh = single_triangle();
        let config = SubdivisionConfig {
            levels: 2,
            ..Default::default()
        };
        let result = subdivide_mesh(&mesh, &config).expect("should succeed");
        let s = format_subdivision_result(&result);
        assert!(s.contains("levels"), "2 levels should use plural");
    }

    #[test]
    fn test_format_subdivision_result_singular_level() {
        let mesh = single_triangle();
        let config = SubdivisionConfig {
            levels: 1,
            ..Default::default()
        };
        let result = subdivide_mesh(&mesh, &config).expect("should succeed");
        let s = format_subdivision_result(&result);
        // "1 level" not "1 levels"
        assert!(s.contains("1 level"), "1 level should use singular: {s}");
        assert!(!s.contains("1 levels"), "should not say '1 levels': {s}");
    }

    // -----------------------------------------------------------------------
    // 16. SubdivisionError variants
    // -----------------------------------------------------------------------

    #[test]
    fn test_error_empty_mesh_display() {
        let e = SubdivisionError::EmptyMesh;
        assert!(!e.to_string().is_empty());
    }

    #[test]
    fn test_error_no_faces_display() {
        let e = SubdivisionError::NoFaces;
        assert!(e.to_string().contains("no faces"));
    }

    #[test]
    fn test_error_too_many_vertices_display() {
        let e = SubdivisionError::TooManyVertices {
            level: 3,
            estimated: 1_000_000,
            max: 500_000,
        };
        let s = e.to_string();
        assert!(
            s.contains("1000000") || s.contains("1_000_000") || s.contains("too many"),
            "got: {s}"
        );
    }

    #[test]
    fn test_error_degenerate_face_display() {
        let e = SubdivisionError::DegenerateFace { idx: 42 };
        assert!(e.to_string().contains("42"));
    }

    #[test]
    fn test_error_vertex_index_out_of_range_display() {
        let e = SubdivisionError::VertexIndexOutOfRange { idx: 99, n: 10 };
        let s = e.to_string();
        assert!(s.contains("99") && s.contains("10"), "got: {s}");
    }

    // -----------------------------------------------------------------------
    // 17. SubdivisionConfig defaults
    // -----------------------------------------------------------------------

    #[test]
    fn test_subdivision_config_defaults() {
        let config = SubdivisionConfig::default();
        assert_eq!(config.levels, 1);
        assert!(config.update_vertices);
        assert!(config.recompute_normals);
        assert_eq!(config.max_vertices, 500_000);
    }

    // -----------------------------------------------------------------------
    // 18. SubdivisionResult fields
    // -----------------------------------------------------------------------

    #[test]
    fn test_subdivision_result_fields() {
        let mesh = two_triangle_mesh();
        let config = SubdivisionConfig::default();
        let result = subdivide_mesh(&mesh, &config).expect("should succeed");
        assert_eq!(result.n_original_vertices, 4);
        assert_eq!(result.n_original_faces, 2);
        assert!(result.n_new_vertices > 0);
        assert_eq!(result.levels_applied, 1);
    }

    // -----------------------------------------------------------------------
    // 19. SubdivisionStats fields
    // -----------------------------------------------------------------------

    #[test]
    fn test_subdivision_stats_fields_valid() {
        let mesh = single_triangle();
        let config = SubdivisionConfig::default();
        let result = subdivide_mesh(&mesh, &config).expect("should succeed");
        let stats = compute_subdivision_stats(&mesh, &result);
        assert!(stats.mean_edge_length.is_finite());
        assert!(stats.min_edge_length.is_finite());
        assert!(stats.max_edge_length.is_finite());
        assert!(stats.max_edge_length >= stats.min_edge_length);
    }

    // -----------------------------------------------------------------------
    // 20. Edge key canonicalization
    // -----------------------------------------------------------------------

    #[test]
    fn test_edge_key_canonical_form() {
        let k1 = edge_key(3, 7);
        let k2 = edge_key(7, 3);
        assert_eq!(k1, k2, "edge key should be canonical regardless of order");
    }

    #[test]
    fn test_edge_key_same_vertex() {
        let k = edge_key(5, 5);
        assert_eq!(k.0, 5);
        assert_eq!(k.1, 5);
    }

    // -----------------------------------------------------------------------
    // 21. Normals after subdivision
    // -----------------------------------------------------------------------

    #[test]
    fn test_subdivided_normals_are_unit() {
        let mesh = two_triangle_mesh();
        let config = SubdivisionConfig::default();
        let result = subdivide_mesh(&mesh, &config).expect("should succeed");
        for (i, n) in result.mesh.normals.iter().enumerate() {
            let len = n.norm();
            assert!(
                len < 1e-10 || (len - 1.0).abs() < 1e-4,
                "normal {i} is not unit: len = {len}"
            );
        }
    }

    #[test]
    fn test_subdivided_flat_mesh_normals_consistent() {
        // Flat mesh in XY plane: all normals should point in +Z
        let mesh = two_triangle_mesh();
        let config = SubdivisionConfig::default();
        let result = subdivide_mesh(&mesh, &config).expect("should succeed");
        for (i, n) in result.mesh.normals.iter().enumerate() {
            // Normals of flat mesh should point in +Z
            assert!(n.z > 0.8, "vertex {i} normal z = {}", n.z);
        }
    }

    // -----------------------------------------------------------------------
    // 22. Face count power law across multiple levels
    // -----------------------------------------------------------------------

    #[test]
    fn test_face_count_four_power_n_tetrahedron() {
        let mesh = tetrahedron_mesh();
        let orig_f = mesh.faces.len();
        for n in 1usize..=3 {
            let config = SubdivisionConfig {
                levels: n,
                ..Default::default()
            };
            let result = subdivide_mesh(&mesh, &config).expect("should succeed");
            let expected = orig_f * 4usize.pow(n as u32);
            assert_eq!(
                result.mesh.faces.len(),
                expected,
                "N={n}: expected {expected}, got {}",
                result.mesh.faces.len()
            );
        }
    }

    // -----------------------------------------------------------------------
    // 23. Mesh grows after subdivision (vertex count strictly increases)
    // -----------------------------------------------------------------------

    #[test]
    fn test_vertex_count_grows_per_level() {
        let mesh = tetrahedron_mesh();
        let config_l1 = SubdivisionConfig {
            levels: 1,
            ..Default::default()
        };
        let config_l2 = SubdivisionConfig {
            levels: 2,
            ..Default::default()
        };
        let r1 = subdivide_mesh(&mesh, &config_l1).expect("l1");
        let r2 = subdivide_mesh(&mesh, &config_l2).expect("l2");
        assert!(
            r2.mesh.vertices.len() > r1.mesh.vertices.len(),
            "more levels → more vertices"
        );
    }
}
