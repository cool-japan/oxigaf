//! Mesh smoothing and subdivision operations.
//!
//! Provides Laplacian smoothing, Taubin smoothing (shrinkage-free), cotangent-weighted
//! Laplacian smoothing, Loop subdivision, and midpoint subdivision for triangle meshes.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during mesh subdivision.
#[derive(Debug, thiserror::Error)]
pub enum MeshOpsError {
    /// A face references a vertex index outside the supplied vertex array.
    #[error(
        "face {face} references vertex index {index}, but mesh has only {vertex_count} vertices"
    )]
    InvalidFaceIndex {
        /// Index (0-based) of the offending face.
        face: usize,
        /// The out-of-range vertex index found in the face.
        index: u32,
        /// Total number of vertices (valid indices are `0..vertex_count`).
        vertex_count: usize,
    },
}

// ---------------------------------------------------------------------------
// Configuration types
// ---------------------------------------------------------------------------

/// Weighting scheme for Laplacian computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightMode {
    /// Equal weights for all neighbors.
    Uniform,
    /// Cotangent weights (angle-based, better quality).
    Cotangent,
}

/// Configuration for mesh smoothing operations.
#[derive(Debug, Clone)]
pub struct MeshSmoothingConfig {
    /// Number of smoothing iterations (default: 1).
    pub iterations: u32,
    /// Taubin λ expansion factor (default: 0.5).
    pub lambda: f32,
    /// Taubin μ contraction factor, must be negative (default: -0.52).
    pub mu: f32,
    /// Don't move boundary vertices when true (default: true).
    pub preserve_boundary: bool,
    /// Weight mode for Laplacian computation (default: Uniform).
    pub weight_mode: WeightMode,
}

impl Default for MeshSmoothingConfig {
    fn default() -> Self {
        Self {
            iterations: 1,
            lambda: 0.5,
            mu: -0.52,
            preserve_boundary: true,
            weight_mode: WeightMode::Uniform,
        }
    }
}

impl MeshSmoothingConfig {
    /// Taubin smoothing with cotangent weights: prevents shrinkage while improving quality.
    #[must_use]
    pub fn taubin() -> Self {
        Self {
            iterations: 1,
            lambda: 0.5,
            mu: -0.52,
            preserve_boundary: true,
            weight_mode: WeightMode::Cotangent,
        }
    }

    /// Laplacian smoothing with uniform weights for the given number of iterations.
    #[must_use]
    pub fn laplacian(iterations: u32) -> Self {
        Self {
            iterations,
            lambda: 0.5,
            mu: -0.52,
            preserve_boundary: true,
            weight_mode: WeightMode::Uniform,
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
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

/// 3D vector subtraction.
#[inline]
fn vec3_sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// 3D vector dot product.
#[inline]
fn vec3_dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// 3D vector cross product.
#[inline]
fn vec3_cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// 3D vector magnitude.
#[inline]
fn vec3_len(a: [f32; 3]) -> f32 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}

/// 3D face area = 0.5 * |cross(e1, e2)|.
#[inline]
fn face_area(v0: [f32; 3], v1: [f32; 3], v2: [f32; 3]) -> f32 {
    let e1 = vec3_sub(v1, v0);
    let e2 = vec3_sub(v2, v0);
    vec3_len(vec3_cross(e1, e2)) * 0.5
}

// ---------------------------------------------------------------------------
// Boundary detection
// ---------------------------------------------------------------------------

/// Find boundary vertices: vertices on edges shared by exactly 1 face.
///
/// Returns a bool array of length `vertices_len` where `true` means the vertex
/// lies on the boundary.
#[must_use]
pub fn find_boundary_vertices(vertices_len: usize, faces: &[[u32; 3]]) -> Vec<bool> {
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

    let mut boundary = vec![false; vertices_len];
    for ((a, b), count) in &edge_count {
        if *count == 1 {
            let a_idx = *a as usize;
            let b_idx = *b as usize;
            if a_idx < vertices_len {
                boundary[a_idx] = true;
            }
            if b_idx < vertices_len {
                boundary[b_idx] = true;
            }
        }
    }
    boundary
}

// ---------------------------------------------------------------------------
// Adjacency
// ---------------------------------------------------------------------------

/// Compute adjacency list from face list.
///
/// `adjacency[i]` is a sorted list of vertex indices adjacent to vertex `i`.
#[must_use]
pub fn build_adjacency(vertices: &[[f32; 3]], faces: &[[u32; 3]]) -> Vec<Vec<usize>> {
    let n = vertices.len();
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];

    for face in faces {
        let i0 = face[0] as usize;
        let i1 = face[1] as usize;
        let i2 = face[2] as usize;

        // Add each undirected edge once per direction
        let pairs = [(i0, i1), (i0, i2), (i1, i0), (i1, i2), (i2, i0), (i2, i1)];
        for (a, b) in pairs {
            if a < n && b < n && !adj[a].contains(&b) {
                adj[a].push(b);
            }
        }
    }

    for list in &mut adj {
        list.sort_unstable();
    }

    adj
}

// ---------------------------------------------------------------------------
// Uniform Laplacian
// ---------------------------------------------------------------------------

/// Compute Laplacian of vertices using uniform weights.
///
/// For each vertex: `delta = mean(neighbor_positions) - position`.
/// Returns per-vertex Laplacian displacement vectors.
#[must_use]
pub fn compute_laplacian_uniform(vertices: &[[f32; 3]], adjacency: &[Vec<usize>]) -> Vec<[f32; 3]> {
    let n = vertices.len();
    let mut result = vec![[0.0f32; 3]; n];

    for i in 0..n {
        let neighbors = &adjacency[i];
        if neighbors.is_empty() {
            continue;
        }
        let count = neighbors.len() as f32;
        let mut mean = [0.0f32; 3];
        for &j in neighbors {
            let vj = vertices[j];
            mean[0] += vj[0];
            mean[1] += vj[1];
            mean[2] += vj[2];
        }
        mean[0] /= count;
        mean[1] /= count;
        mean[2] /= count;

        let vi = vertices[i];
        result[i] = [mean[0] - vi[0], mean[1] - vi[1], mean[2] - vi[2]];
    }

    result
}

/// One step of Laplacian smoothing: `v_new = v + lambda * L(v)`.
///
/// If `preserve_boundary` is true, boundary vertices (as indicated by
/// `boundary_mask`) are not moved.
#[must_use]
pub fn laplacian_smooth_step(
    vertices: &[[f32; 3]],
    adjacency: &[Vec<usize>],
    lambda: f32,
    preserve_boundary: bool,
    boundary_mask: &[bool],
) -> Vec<[f32; 3]> {
    let laplacian = compute_laplacian_uniform(vertices, adjacency);
    let n = vertices.len();
    let mut result = vertices.to_vec();

    for i in 0..n {
        let is_boundary = boundary_mask.get(i).copied().unwrap_or(false);
        if preserve_boundary && is_boundary {
            continue;
        }
        let vi = vertices[i];
        let li = laplacian[i];
        result[i] = [
            vi[0] + lambda * li[0],
            vi[1] + lambda * li[1],
            vi[2] + lambda * li[2],
        ];
    }

    result
}

/// Apply Laplacian smoothing for N iterations.
///
/// Builds adjacency from faces, detects boundary vertices, then repeatedly
/// applies Laplacian steps with `config.lambda`.
#[must_use]
pub fn laplacian_smooth(
    vertices: &[[f32; 3]],
    faces: &[[u32; 3]],
    config: &MeshSmoothingConfig,
) -> Vec<[f32; 3]> {
    let adjacency = build_adjacency(vertices, faces);
    let boundary_mask = find_boundary_vertices(vertices.len(), faces);
    let mut verts = vertices.to_vec();

    for _ in 0..config.iterations {
        verts = laplacian_smooth_step(
            &verts,
            &adjacency,
            config.lambda,
            config.preserve_boundary,
            &boundary_mask,
        );
    }

    verts
}

// ---------------------------------------------------------------------------
// Taubin smoothing
// ---------------------------------------------------------------------------

/// Apply Taubin smoothing (lambda/mu scheme, prevents shrinkage).
///
/// Each iteration alternates:
/// 1. `v += lambda * L(v)` (expansion step)
/// 2. `v += mu * L(v)` (contraction step, mu < 0)
///
/// The combination of expansion and contraction preserves volume better than
/// pure Laplacian smoothing.
#[must_use]
pub fn taubin_smooth(
    vertices: &[[f32; 3]],
    faces: &[[u32; 3]],
    config: &MeshSmoothingConfig,
) -> Vec<[f32; 3]> {
    let adjacency = build_adjacency(vertices, faces);
    let boundary_mask = find_boundary_vertices(vertices.len(), faces);
    let mut verts = vertices.to_vec();

    for _ in 0..config.iterations {
        // Lambda step (expansion)
        verts = laplacian_smooth_step(
            &verts,
            &adjacency,
            config.lambda,
            config.preserve_boundary,
            &boundary_mask,
        );
        // Mu step (contraction, mu < 0)
        verts = laplacian_smooth_step(
            &verts,
            &adjacency,
            config.mu,
            config.preserve_boundary,
            &boundary_mask,
        );
    }

    verts
}

// ---------------------------------------------------------------------------
// Cotangent Laplacian
// ---------------------------------------------------------------------------

/// Compute cotangent weight for an edge in a triangle.
///
/// `c` is the vertex opposite to edge `(a, b)`.
/// Returns `cot(angle at c)` = `dot(ca, cb) / |ca × cb|`, clamped to `[0, 10]`.
///
/// A degenerate (zero-area / collinear) triangle returns `0.0` rather than
/// saturating to the top of the clamp range: it carries no reliable surface
/// information, so it must not be able to out-weigh well-formed neighbors in
/// `compute_laplacian_cotangent`. Note the clamp itself is intentionally
/// one-sided (`[0, 10]`, discarding the negative cotangents of obtuse
/// triangles) -- this is a common practical stabilization for the
/// cotangent Laplacian, not the unclamped discrete Laplace-Beltrami
/// operator.
#[must_use]
fn cotangent_weight(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> f32 {
    let ca = vec3_sub(a, c);
    let cb = vec3_sub(b, c);
    let dot = vec3_dot(ca, cb);
    let cross_mag = vec3_len(vec3_cross(ca, cb));
    if cross_mag < 1e-10 {
        // Degenerate triangle: contributes no weight.
        return 0.0;
    }
    (dot / cross_mag).clamp(0.0, 10.0)
}

/// Compute cotangent-weighted Laplacian.
///
/// For each vertex `v_i`:
///   `L(v_i) = (1/A_mixed) * sum_neighbors w_ij * (v_j - v_i)`
///
/// where `w_ij = 0.5 * (cot(alpha_ij) + cot(beta_ij))` and alpha, beta are the
/// angles opposite to edge `(i, j)` in the two triangles sharing it.
/// `A_mixed` is approximated as `(1/3) * sum_of_face_areas_in_1-ring`.
#[must_use]
pub fn compute_laplacian_cotangent(vertices: &[[f32; 3]], faces: &[[u32; 3]]) -> Vec<[f32; 3]> {
    let n = vertices.len();

    // Accumulate: weighted displacement sum and mixed area per vertex
    let mut weighted_sum: Vec<[f32; 3]> = vec![[0.0f32; 3]; n];
    let mut mixed_area: Vec<f32> = vec![0.0f32; n];

    for face in faces {
        let i0 = face[0] as usize;
        let i1 = face[1] as usize;
        let i2 = face[2] as usize;

        if i0 >= n || i1 >= n || i2 >= n {
            continue;
        }

        let v0 = vertices[i0];
        let v1 = vertices[i1];
        let v2 = vertices[i2];

        // Cotangent at each vertex of this face (opposite to the other two vertices)
        // cot at v0 (opposite to edge v1-v2) = cot of angle at v0
        let cot0 = cotangent_weight(v1, v2, v0);
        // cot at v1 (opposite to edge v0-v2)
        let cot1 = cotangent_weight(v0, v2, v1);
        // cot at v2 (opposite to edge v0-v1)
        let cot2 = cotangent_weight(v0, v1, v2);

        // Edge (i1, i2): weight = 0.5 * cot at vertex i0 (opposite vertex for this edge)
        let w12 = 0.5 * cot0;
        // Edge (i0, i2): weight = 0.5 * cot at vertex i1
        let w02 = 0.5 * cot1;
        // Edge (i0, i1): weight = 0.5 * cot at vertex i2
        let w01 = 0.5 * cot2;

        // Accumulate for vertex i0: affected by edges (i0,i1) and (i0,i2)
        let d01 = vec3_sub(v1, v0);
        let d02 = vec3_sub(v2, v0);
        weighted_sum[i0][0] += w01 * d01[0] + w02 * d02[0];
        weighted_sum[i0][1] += w01 * d01[1] + w02 * d02[1];
        weighted_sum[i0][2] += w01 * d01[2] + w02 * d02[2];

        // Accumulate for vertex i1: affected by edges (i0,i1) and (i1,i2)
        let d10 = vec3_sub(v0, v1);
        let d12 = vec3_sub(v2, v1);
        weighted_sum[i1][0] += w01 * d10[0] + w12 * d12[0];
        weighted_sum[i1][1] += w01 * d10[1] + w12 * d12[1];
        weighted_sum[i1][2] += w01 * d10[2] + w12 * d12[2];

        // Accumulate for vertex i2: affected by edges (i0,i2) and (i1,i2)
        let d20 = vec3_sub(v0, v2);
        let d21 = vec3_sub(v1, v2);
        weighted_sum[i2][0] += w02 * d20[0] + w12 * d21[0];
        weighted_sum[i2][1] += w02 * d20[1] + w12 * d21[1];
        weighted_sum[i2][2] += w02 * d20[2] + w12 * d21[2];

        // Mixed area contribution: 1/3 of face area for each vertex
        let area = face_area(v0, v1, v2);
        let area_contrib = area / 3.0;
        mixed_area[i0] += area_contrib;
        mixed_area[i1] += area_contrib;
        mixed_area[i2] += area_contrib;
    }

    // Normalize by mixed area
    let mut result = vec![[0.0f32; 3]; n];
    for i in 0..n {
        let a = mixed_area[i];
        if a > 1e-10 {
            result[i] = [
                weighted_sum[i][0] / a,
                weighted_sum[i][1] / a,
                weighted_sum[i][2] / a,
            ];
        }
    }

    result
}

/// Apply cotangent-weighted Laplacian smoothing.
///
/// For each iteration applies: `v_new = v + lambda * L_cot(v)`.
/// Boundary vertices are preserved if `config.preserve_boundary` is true.
#[must_use]
pub fn cotangent_smooth(
    vertices: &[[f32; 3]],
    faces: &[[u32; 3]],
    config: &MeshSmoothingConfig,
) -> Vec<[f32; 3]> {
    let boundary_mask = find_boundary_vertices(vertices.len(), faces);
    let mut verts = vertices.to_vec();
    let n = verts.len();

    for _ in 0..config.iterations {
        // Recompute cotangent Laplacian each iteration (positions change)
        let laplacian = compute_laplacian_cotangent(&verts, faces);

        let mut new_verts = verts.clone();
        for i in 0..n {
            let is_boundary = boundary_mask.get(i).copied().unwrap_or(false);
            if config.preserve_boundary && is_boundary {
                continue;
            }
            let vi = verts[i];
            let li = laplacian[i];
            new_verts[i] = [
                vi[0] + config.lambda * li[0],
                vi[1] + config.lambda * li[1],
                vi[2] + config.lambda * li[2],
            ];
        }
        verts = new_verts;
    }

    verts
}

// ---------------------------------------------------------------------------
// Subdivision
// ---------------------------------------------------------------------------

/// Get or create a midpoint vertex between v0 and v1.
///
/// Uses canonical edge ordering `(min, max)` to avoid duplicates.
/// The new vertex position is the arithmetic midpoint `(v0 + v1) / 2`.
fn get_or_create_midpoint(
    v0: u32,
    v1: u32,
    vertices: &mut Vec<[f32; 3]>,
    edge_midpoints: &mut HashMap<(u32, u32), u32>,
) -> u32 {
    let edge = canonical_edge(v0, v1);
    if let Some(&mid_idx) = edge_midpoints.get(&edge) {
        return mid_idx;
    }
    let p0 = vertices[v0 as usize];
    let p1 = vertices[v1 as usize];
    let mid = [
        (p0[0] + p1[0]) * 0.5,
        (p0[1] + p1[1]) * 0.5,
        (p0[2] + p1[2]) * 0.5,
    ];
    let mid_idx = vertices.len() as u32;
    vertices.push(mid);
    edge_midpoints.insert(edge, mid_idx);
    mid_idx
}

/// Validate that every face index is within `0..vertices_len`, matching the
/// check `mesh_subdivision::validate_mesh_for_subdivision` performs for its
/// own subdivision entry points.
///
/// Without this, `get_or_create_midpoint` / `loop_compute_edge_midpoints`
/// index `vertices[idx as usize]` unchecked and panic on a malformed face;
/// running it once up front lets both `midpoint_subdivide` and
/// `loop_subdivide` assume in-range indices everywhere downstream.
fn validate_face_indices(vertices_len: usize, faces: &[[u32; 3]]) -> Result<(), MeshOpsError> {
    for (fi, face) in faces.iter().enumerate() {
        for &idx in face {
            if idx as usize >= vertices_len {
                return Err(MeshOpsError::InvalidFaceIndex {
                    face: fi,
                    index: idx,
                    vertex_count: vertices_len,
                });
            }
        }
    }
    Ok(())
}

/// Simple midpoint subdivision (no smoothing).
///
/// Splits each triangle into 4 sub-triangles by inserting midpoints at each
/// edge center without applying Loop's weighting.
///
/// The new vertex count is `V + E` and the new face count is `4 * F`.
///
/// # Errors
///
/// Returns [`MeshOpsError::InvalidFaceIndex`] if any face references a
/// vertex index outside `0..vertices.len()`.
pub fn midpoint_subdivide(
    vertices: &[[f32; 3]],
    faces: &[[u32; 3]],
) -> Result<(Vec<[f32; 3]>, Vec<[u32; 3]>), MeshOpsError> {
    validate_face_indices(vertices.len(), faces)?;

    let mut new_vertices = vertices.to_vec();
    let mut edge_midpoints: HashMap<(u32, u32), u32> = HashMap::new();
    let mut new_faces: Vec<[u32; 3]> = Vec::with_capacity(faces.len() * 4);

    for face in faces {
        let a = face[0];
        let b = face[1];
        let c = face[2];

        // Find or create edge midpoints
        let m_ab = get_or_create_midpoint(a, b, &mut new_vertices, &mut edge_midpoints);
        let m_bc = get_or_create_midpoint(b, c, &mut new_vertices, &mut edge_midpoints);
        let m_ca = get_or_create_midpoint(c, a, &mut new_vertices, &mut edge_midpoints);

        // Each original triangle becomes 4 triangles
        new_faces.push([a, m_ab, m_ca]);
        new_faces.push([b, m_bc, m_ab]);
        new_faces.push([c, m_ca, m_bc]);
        new_faces.push([m_ab, m_bc, m_ca]);
    }

    Ok((new_vertices, new_faces))
}

/// Perform one iteration of Loop subdivision.
///
/// Splits each triangle into 4 sub-triangles by adding edge midpoints and
/// applying Loop's weighting for smooth interpolation.
///
/// # Loop weighting rules
///
/// **New edge-midpoint vertices**:
/// - Interior edge (shared by 2 faces): `3/8*(v_i + v_j) + 1/8*(opp1 + opp2)`
/// - Boundary edge: `(v_i + v_j) / 2`
///
/// **Updated original vertices** (applied after edge-point computation):
/// - Interior vertex with `n` neighbors:
///   - `beta = 3/16` if `n == 3`; otherwise `beta = 3/(8*n)`
///   - `updated = (1 - n*beta)*v + beta*sum(neighbors)`
/// - Boundary vertex: `(6/8)*v + (1/8)*(prev + next)`, where `prev`/`next`
///   are the two neighbors reachable via a genuine boundary EDGE (an edge
///   belonging to exactly one face) -- not merely any neighbor that happens
///   to also touch the boundary somewhere else (see
///   [`boundary_edge_neighbors`]).
///
/// # Errors
///
/// Returns [`MeshOpsError::InvalidFaceIndex`] if any face references a
/// vertex index outside `0..vertices.len()`.
pub fn loop_subdivide(
    vertices: &[[f32; 3]],
    faces: &[[u32; 3]],
) -> Result<(Vec<[f32; 3]>, Vec<[u32; 3]>), MeshOpsError> {
    let n_verts = vertices.len();
    validate_face_indices(n_verts, faces)?;

    let mut edge_faces: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
    for (fi, face) in faces.iter().enumerate() {
        for edge in [
            canonical_edge(face[0], face[1]),
            canonical_edge(face[1], face[2]),
            canonical_edge(face[2], face[0]),
        ] {
            edge_faces.entry(edge).or_default().push(fi);
        }
    }
    let boundary_vertex_mask = find_boundary_vertices(n_verts, faces);
    let edge_midpoint_pos =
        loop_compute_edge_midpoints(faces, vertices, &edge_faces, &boundary_vertex_mask);
    let boundary_nbrs = boundary_edge_neighbors(n_verts, &edge_faces);
    let adjacency = build_adjacency(vertices, faces);
    let updated_vertices = loop_update_vertices(vertices, &adjacency, &boundary_nbrs);
    loop_build_output(faces, updated_vertices, &edge_midpoint_pos)
}

/// Build, for each vertex, the list of neighbors reachable via a genuine
/// boundary edge (an edge belonging to exactly one face).
///
/// Unlike `boundary_vertex_mask` (which only records whether a vertex
/// touches *some* boundary edge somewhere in the mesh), this identifies the
/// specific edges, so Loop subdivision's boundary rule averages against the
/// (at most two) neighbors actually connected to `v` along the boundary --
/// not any boundary-marked vertex that happens to also be in the 1-ring.
/// Without this distinction, a vertex whose 1-ring contains three or more
/// boundary-marked vertices (common wherever the 1-ring includes vertices
/// from a different part of the same boundary loop, e.g. near a
/// high-curvature boundary or a pinch point) picks two arbitrary
/// lowest-index neighbors instead of its true boundary predecessor/successor.
fn boundary_edge_neighbors(
    n_verts: usize,
    edge_faces: &HashMap<(u32, u32), Vec<usize>>,
) -> Vec<Vec<u32>> {
    let mut neighbors: Vec<Vec<u32>> = vec![Vec::new(); n_verts];
    for (&(a, b), owning_faces) in edge_faces {
        if owning_faces.len() != 1 {
            continue;
        }
        if (a as usize) < n_verts {
            neighbors[a as usize].push(b);
        }
        if (b as usize) < n_verts {
            neighbors[b as usize].push(a);
        }
    }
    neighbors
}

/// Compute Loop midpoint position for each edge (step 2 of loop subdivision).
fn loop_compute_edge_midpoints(
    faces: &[[u32; 3]],
    vertices: &[[f32; 3]],
    edge_faces: &HashMap<(u32, u32), Vec<usize>>,
    boundary_vertex_mask: &[bool],
) -> HashMap<(u32, u32), [f32; 3]> {
    let _ = boundary_vertex_mask; // used implicitly via fi_list.len() check
    let mut edge_midpoint_pos: HashMap<(u32, u32), [f32; 3]> = HashMap::new();
    for face in faces {
        let verts_idx = [face[0], face[1], face[2]];
        let edge_pairs: [((u32, u32), u32); 3] = [
            (canonical_edge(face[0], face[1]), face[2]),
            (canonical_edge(face[1], face[2]), face[0]),
            (canonical_edge(face[2], face[0]), face[1]),
        ];
        for ((ei, ej), opp_in_this_face) in &edge_pairs {
            let edge = (*ei, *ej);
            if edge_midpoint_pos.contains_key(&edge) {
                continue;
            }
            let vi = vertices[*ei as usize];
            let vj = vertices[*ej as usize];
            let fi_list = edge_faces.get(&edge).map_or(&[][..], Vec::as_slice);
            let mid_pos = if fi_list.len() == 2 {
                let other_fi = fi_list.iter().copied().find(|&fi| {
                    let face_ref = &faces[fi];
                    !verts_idx.iter().all(|&v| face_ref.contains(&v))
                });
                let opp2_pos = if let Some(other_fi) = other_fi {
                    let other_face = &faces[other_fi];
                    let opp2_idx = other_face.iter().copied().find(|&v| v != *ei && v != *ej);
                    opp2_idx.map_or(vertices[*opp_in_this_face as usize], |idx| {
                        vertices[idx as usize]
                    })
                } else {
                    vertices[*opp_in_this_face as usize]
                };
                let opp1 = vertices[*opp_in_this_face as usize];
                [
                    (3.0 / 8.0) * (vi[0] + vj[0]) + (1.0 / 8.0) * (opp1[0] + opp2_pos[0]),
                    (3.0 / 8.0) * (vi[1] + vj[1]) + (1.0 / 8.0) * (opp1[1] + opp2_pos[1]),
                    (3.0 / 8.0) * (vi[2] + vj[2]) + (1.0 / 8.0) * (opp1[2] + opp2_pos[2]),
                ]
            } else {
                [
                    (vi[0] + vj[0]) * 0.5,
                    (vi[1] + vj[1]) * 0.5,
                    (vi[2] + vj[2]) * 0.5,
                ]
            };
            edge_midpoint_pos.insert(edge, mid_pos);
        }
    }
    edge_midpoint_pos
}

/// Update original vertices with Loop subdivision weights (step 3).
///
/// `boundary_edge_nbrs[v]` holds the (at most two, for a well-formed
/// 2-manifold-with-boundary mesh) neighbors reachable from `v` via a
/// genuine boundary edge -- see [`boundary_edge_neighbors`]. A vertex with
/// zero such neighbors is interior; exactly two means the standard boundary
/// rule applies; any other count (a non-manifold or dangling boundary
/// vertex) is left unmoved rather than averaged against an arbitrary or
/// undercounted set.
fn loop_update_vertices(
    vertices: &[[f32; 3]],
    adjacency: &[Vec<usize>],
    boundary_edge_nbrs: &[Vec<u32>],
) -> Vec<[f32; 3]> {
    vertices
        .iter()
        .enumerate()
        .map(|(vert_i, &vi)| {
            let bnd_nbrs = &boundary_edge_nbrs[vert_i];
            if !bnd_nbrs.is_empty() {
                if bnd_nbrs.len() == 2 {
                    let prev = vertices[bnd_nbrs[0] as usize];
                    let next = vertices[bnd_nbrs[1] as usize];
                    [
                        (6.0 / 8.0) * vi[0] + (1.0 / 8.0) * (prev[0] + next[0]),
                        (6.0 / 8.0) * vi[1] + (1.0 / 8.0) * (prev[1] + next[1]),
                        (6.0 / 8.0) * vi[2] + (1.0 / 8.0) * (prev[2] + next[2]),
                    ]
                } else {
                    vi
                }
            } else {
                let neighbors = &adjacency[vert_i];
                let neighbor_count = neighbors.len();
                if neighbor_count == 0 {
                    vi
                } else {
                    let beta = if neighbor_count == 3 {
                        3.0_f32 / 16.0
                    } else {
                        3.0 / (8.0 * neighbor_count as f32)
                    };
                    let alpha = 1.0 - neighbor_count as f32 * beta;
                    let sum = neighbors.iter().fold([0.0f32; 3], |mut acc, &j| {
                        let vj = vertices[j];
                        acc[0] += vj[0];
                        acc[1] += vj[1];
                        acc[2] += vj[2];
                        acc
                    });
                    [
                        alpha * vi[0] + beta * sum[0],
                        alpha * vi[1] + beta * sum[1],
                        alpha * vi[2] + beta * sum[2],
                    ]
                }
            }
        })
        .collect()
}

/// Assign edge-midpoint indices and build the 4× subdivided face list (steps 4–5).
fn loop_build_output(
    faces: &[[u32; 3]],
    updated_vertices: Vec<[f32; 3]>,
    edge_midpoint_pos: &HashMap<(u32, u32), [f32; 3]>,
) -> (Vec<[f32; 3]>, Vec<[u32; 3]>) {
    let mut edge_midpoint_idx: HashMap<(u32, u32), u32> = HashMap::new();
    let mut all_vertices = updated_vertices;
    for face in faces {
        for edge in [
            canonical_edge(face[0], face[1]),
            canonical_edge(face[1], face[2]),
            canonical_edge(face[2], face[0]),
        ] {
            edge_midpoint_idx.entry(edge).or_insert_with(|| {
                let pos = edge_midpoint_pos.get(&edge).copied().unwrap_or([0.0; 3]);
                let idx = all_vertices.len() as u32;
                all_vertices.push(pos);
                idx
            });
        }
    }
    let mut new_faces: Vec<[u32; 3]> = Vec::with_capacity(faces.len() * 4);
    for face in faces {
        let (fa, fb, fc) = (face[0], face[1], face[2]);
        let m_ab = edge_midpoint_idx
            .get(&canonical_edge(fa, fb))
            .copied()
            .unwrap_or(fa);
        let m_bc = edge_midpoint_idx
            .get(&canonical_edge(fb, fc))
            .copied()
            .unwrap_or(fb);
        let m_ca = edge_midpoint_idx
            .get(&canonical_edge(fc, fa))
            .copied()
            .unwrap_or(fc);
        new_faces.push([fa, m_ab, m_ca]);
        new_faces.push([fb, m_bc, m_ab]);
        new_faces.push([fc, m_ca, m_bc]);
        new_faces.push([m_ab, m_bc, m_ca]);
    }
    (all_vertices, new_faces)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Test helpers
    // -----------------------------------------------------------------------

    /// Single equilateral triangle with side length 1 in the XY plane.
    fn triangle_verts() -> Vec<[f32; 3]> {
        vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, (3.0f32).sqrt() / 2.0, 0.0],
        ]
    }

    fn triangle_faces() -> Vec<[u32; 3]> {
        vec![[0, 1, 2]]
    }

    /// Tetrahedron: 4 vertices, 4 faces — closed manifold, no boundary.
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

    /// Open quad strip: two triangles sharing edge (1,2), with boundary.
    fn quad_strip() -> (Vec<[f32; 3]>, Vec<[u32; 3]>) {
        let verts = vec![
            [0.0_f32, 0.0, 0.0],
            [1.0_f32, 0.0, 0.0],
            [0.0_f32, 1.0, 0.0],
            [1.0_f32, 1.0, 0.0],
        ];
        let faces = vec![[0, 1, 2], [1, 3, 2]];
        (verts, faces)
    }

    /// Larger grid mesh for convergence tests.
    fn grid_mesh(size: u32) -> (Vec<[f32; 3]>, Vec<[u32; 3]>) {
        let n = size + 1;
        let mut verts = Vec::new();
        for j in 0..n {
            for i in 0..n {
                verts.push([i as f32 / size as f32, j as f32 / size as f32, 0.0]);
            }
        }
        let mut faces = Vec::new();
        for j in 0..size {
            for i in 0..size {
                let tl = j * n + i;
                let tr = tl + 1;
                let bl = (j + 1) * n + i;
                let br = bl + 1;
                faces.push([tl, tr, bl]);
                faces.push([tr, br, bl]);
            }
        }
        (verts, faces)
    }

    // -----------------------------------------------------------------------
    // test_build_adjacency_triangle
    // -----------------------------------------------------------------------

    #[test]
    fn test_build_adjacency_triangle() {
        let verts = triangle_verts();
        let faces = triangle_faces();
        let adj = build_adjacency(&verts, &faces);
        assert_eq!(adj.len(), 3);
        // Each vertex in a triangle is adjacent to the other two
        for (i, neighbors) in adj.iter().enumerate().take(3) {
            assert_eq!(neighbors.len(), 2, "vertex {i} should have 2 neighbors");
        }
        // Check that adjacency is symmetric
        for (i, neighbors) in adj.iter().enumerate().take(3) {
            for &j in neighbors {
                assert!(
                    adj[j].contains(&i),
                    "adjacency must be symmetric: {j} should contain {i}"
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // test_build_adjacency_quad_strip
    // -----------------------------------------------------------------------

    #[test]
    fn test_build_adjacency_quad_strip() {
        let (verts, faces) = quad_strip();
        let adj = build_adjacency(&verts, &faces);
        // vertex 0: adjacent to 1, 2
        assert_eq!(adj[0].len(), 2);
        // vertex 3: adjacent to 1, 2
        assert_eq!(adj[3].len(), 2);
        // vertex 1 and 2 are shared, should have 3 neighbors each
        assert_eq!(adj[1].len(), 3, "vertex 1 should have 3 neighbors");
        assert_eq!(adj[2].len(), 3, "vertex 2 should have 3 neighbors");
    }

    // -----------------------------------------------------------------------
    // test_find_boundary_vertices_closed_mesh
    // -----------------------------------------------------------------------

    #[test]
    fn test_find_boundary_vertices_closed_mesh() {
        let (verts, faces) = tetrahedron();
        let boundary = find_boundary_vertices(verts.len(), &faces);
        assert_eq!(boundary.len(), 4);
        // All vertices in a closed tetrahedron are interior (not on boundary)
        assert!(
            boundary.iter().all(|&b| !b),
            "tetrahedron should have no boundary vertices"
        );
    }

    // -----------------------------------------------------------------------
    // test_find_boundary_vertices_open_mesh
    // -----------------------------------------------------------------------

    #[test]
    fn test_find_boundary_vertices_open_mesh() {
        let (verts, faces) = quad_strip();
        let boundary = find_boundary_vertices(verts.len(), &faces);
        assert_eq!(boundary.len(), 4);
        // All 4 vertices of the quad strip are on boundary
        let boundary_count = boundary.iter().filter(|&&b| b).count();
        assert_eq!(
            boundary_count, 4,
            "all 4 vertices of open strip should be on boundary"
        );
    }

    // -----------------------------------------------------------------------
    // test_laplacian_smooth_single_step
    // -----------------------------------------------------------------------

    #[test]
    fn test_laplacian_smooth_single_step() {
        let (verts, faces) = grid_mesh(4);
        let config = MeshSmoothingConfig::laplacian(1);
        let smoothed = laplacian_smooth(&verts, &faces, &config);
        assert_eq!(smoothed.len(), verts.len());
        // Vertices should be finite
        for v in &smoothed {
            assert!(v[0].is_finite() && v[1].is_finite() && v[2].is_finite());
        }
    }

    // -----------------------------------------------------------------------
    // test_laplacian_smooth_converges
    // -----------------------------------------------------------------------

    #[test]
    fn test_laplacian_smooth_converges() {
        // After many iterations, interior vertices should approach their Laplacian mean.
        // We test that the mesh changes smoothly and doesn't diverge.
        let (verts, faces) = grid_mesh(4);
        let config = MeshSmoothingConfig::laplacian(20);
        let smoothed = laplacian_smooth(&verts, &faces, &config);
        // All values should remain finite and bounded within the original range
        for v in &smoothed {
            assert!(v[0].is_finite() && v[1].is_finite() && v[2].is_finite());
            assert!(v[0] >= -0.1 && v[0] <= 1.1);
            assert!(v[1] >= -0.1 && v[1] <= 1.1);
        }
    }

    // -----------------------------------------------------------------------
    // test_laplacian_smooth_preserves_boundary
    // -----------------------------------------------------------------------

    #[test]
    fn test_laplacian_smooth_preserves_boundary() {
        let (verts, faces) = grid_mesh(4);
        let config = MeshSmoothingConfig {
            iterations: 10,
            preserve_boundary: true,
            ..MeshSmoothingConfig::default()
        };
        let boundary_mask = find_boundary_vertices(verts.len(), &faces);
        let smoothed = laplacian_smooth(&verts, &faces, &config);

        for (i, (&original, &smoothed_v)) in verts.iter().zip(smoothed.iter()).enumerate() {
            if boundary_mask[i] {
                // Boundary vertices must not move
                assert!(
                    (original[0] - smoothed_v[0]).abs() < 1e-6
                        && (original[1] - smoothed_v[1]).abs() < 1e-6
                        && (original[2] - smoothed_v[2]).abs() < 1e-6,
                    "boundary vertex {i} moved during smoothing"
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // test_taubin_smooth_preserves_volume_better
    // -----------------------------------------------------------------------

    #[test]
    fn test_taubin_smooth_preserves_volume_better() {
        // Compare centroid displacement: Taubin should keep it closer to original
        // than pure Laplacian (because it alternates expansion/contraction).
        let (verts, faces) = grid_mesh(6);

        let centroid_before: [f32; 3] = {
            let n = verts.len() as f32;
            let mut s = [0.0f32; 3];
            for v in &verts {
                s[0] += v[0];
                s[1] += v[1];
                s[2] += v[2];
            }
            [s[0] / n, s[1] / n, s[2] / n]
        };

        let config_taubin = MeshSmoothingConfig {
            iterations: 10,
            preserve_boundary: false,
            ..MeshSmoothingConfig::default()
        };
        let config_laplacian = MeshSmoothingConfig {
            iterations: 10,
            lambda: 0.5,
            preserve_boundary: false,
            ..MeshSmoothingConfig::default()
        };

        let taubin_verts = taubin_smooth(&verts, &faces, &config_taubin);
        let laplacian_verts = laplacian_smooth(&verts, &faces, &config_laplacian);

        let centroid_after_taubin: [f32; 3] = {
            let n = taubin_verts.len() as f32;
            let mut s = [0.0f32; 3];
            for v in &taubin_verts {
                s[0] += v[0];
                s[1] += v[1];
                s[2] += v[2];
            }
            [s[0] / n, s[1] / n, s[2] / n]
        };

        let centroid_after_laplacian: [f32; 3] = {
            let n = laplacian_verts.len() as f32;
            let mut s = [0.0f32; 3];
            for v in &laplacian_verts {
                s[0] += v[0];
                s[1] += v[1];
                s[2] += v[2];
            }
            [s[0] / n, s[1] / n, s[2] / n]
        };

        let dist_sq = |a: [f32; 3], b: [f32; 3]| -> f32 {
            (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)
        };

        let taubin_centroid_err = dist_sq(centroid_before, centroid_after_taubin);
        let laplacian_centroid_err = dist_sq(centroid_before, centroid_after_laplacian);

        // Both should be finite
        assert!(taubin_centroid_err.is_finite());
        assert!(laplacian_centroid_err.is_finite());
    }

    // -----------------------------------------------------------------------
    // test_taubin_iterations
    // -----------------------------------------------------------------------

    #[test]
    fn test_taubin_iterations() {
        let (verts, faces) = grid_mesh(4);
        let config = MeshSmoothingConfig {
            iterations: 5,
            ..MeshSmoothingConfig::default()
        };
        let smoothed = taubin_smooth(&verts, &faces, &config);
        assert_eq!(smoothed.len(), verts.len());
        for v in &smoothed {
            assert!(v[0].is_finite() && v[1].is_finite() && v[2].is_finite());
        }
    }

    // -----------------------------------------------------------------------
    // test_cotangent_weight_equilateral
    // -----------------------------------------------------------------------

    #[test]
    fn test_cotangent_weight_equilateral() {
        // Equilateral triangle: all angles are 60°, cot(60°) = 1/sqrt(3) ≈ 0.5774
        let a = [1.0f32, 0.0, 0.0];
        let b = [0.5f32, (3.0f32).sqrt() / 2.0, 0.0];
        let c = [0.0f32, 0.0, 0.0]; // opposite vertex
        let w = cotangent_weight(a, b, c);
        let expected = 1.0_f32 / 3.0_f32.sqrt();
        assert!(
            (w - expected).abs() < 1e-5,
            "cotangent weight of equilateral should be ~{expected}, got {w}"
        );
    }

    // -----------------------------------------------------------------------
    // test_cotangent_weight_clamp
    // -----------------------------------------------------------------------

    #[test]
    fn test_cotangent_weight_degenerate_triangle_returns_zero() {
        // Degenerate triangle: c is collinear with a and b → cross product ≈ 0.
        // A zero-area triangle carries no surface information and must not
        // be able to out-weigh well-formed neighbors, so it contributes
        // 0.0 rather than saturating to the top of the clamp range.
        let a = [0.0f32, 0.0, 0.0];
        let b = [2.0f32, 0.0, 0.0];
        let c = [1.0f32, 0.0, 0.0]; // collinear
        let w = cotangent_weight(a, b, c);
        assert_eq!(w, 0.0, "degenerate cotangent weight should be 0.0, got {w}");
    }

    #[test]
    fn test_compute_laplacian_cotangent_ignores_degenerate_sliver_face() {
        // A degenerate (duplicate-vertex) "sliver" face touching an
        // existing triangle's vertices must contribute nothing to the
        // cotangent Laplacian. Previously it injected a large spurious
        // displacement because `cotangent_weight` saturated to the clamp's
        // maximum (10.0) for a degenerate triangle instead of returning
        // 0.0 -- this test would fail under that behavior (the two
        // Laplacians would differ), and passes now that the degenerate
        // face is correctly weightless.
        let verts = triangle_verts();
        let faces_clean = triangle_faces();
        let mut faces_with_sliver = faces_clean.clone();
        faces_with_sliver.push([0, 0, 1]); // degenerate: vertex 0 repeated

        let laplacian_clean = compute_laplacian_cotangent(&verts, &faces_clean);
        let laplacian_with_sliver = compute_laplacian_cotangent(&verts, &faces_with_sliver);

        for (clean, with_sliver) in laplacian_clean.iter().zip(laplacian_with_sliver.iter()) {
            for k in 0..3 {
                assert!(
                    (clean[k] - with_sliver[k]).abs() < 1e-6,
                    "a degenerate sliver face must not change the Laplacian: \
                     clean={clean:?} with_sliver={with_sliver:?}"
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // test_cotangent_smooth_runs
    // -----------------------------------------------------------------------

    #[test]
    fn test_cotangent_smooth_runs() {
        let (verts, faces) = grid_mesh(4);
        let config = MeshSmoothingConfig::taubin();
        let smoothed = cotangent_smooth(&verts, &faces, &config);
        assert_eq!(smoothed.len(), verts.len());
        for v in &smoothed {
            assert!(v[0].is_finite() && v[1].is_finite() && v[2].is_finite());
        }
    }

    // -----------------------------------------------------------------------
    // test_midpoint_subdivide_triangle_count
    // -----------------------------------------------------------------------

    #[test]
    fn test_midpoint_subdivide_triangle_count() {
        let verts = triangle_verts();
        let faces = triangle_faces();
        let (_, new_faces) = midpoint_subdivide(&verts, &faces);
        assert_eq!(
            new_faces.len(),
            4,
            "midpoint subdivision of 1 triangle should produce 4 triangles, got {}",
            new_faces.len()
        );
    }

    // -----------------------------------------------------------------------
    // test_midpoint_subdivide_vertex_count
    // -----------------------------------------------------------------------

    #[test]
    fn test_midpoint_subdivide_vertex_count() {
        // For a single triangle: V=3, E=3 → V'=3+3=6
        let verts = triangle_verts();
        let faces = triangle_faces();
        let (new_verts, _) = midpoint_subdivide(&verts, &faces);
        assert_eq!(
            new_verts.len(),
            6,
            "midpoint subdivision of single triangle should produce 6 vertices, got {}",
            new_verts.len()
        );
    }

    // -----------------------------------------------------------------------
    // test_loop_subdivide_triangle_count
    // -----------------------------------------------------------------------

    #[test]
    fn test_loop_subdivide_triangle_count() {
        let (verts, faces) = tetrahedron();
        let (_, new_faces) = loop_subdivide(&verts, &faces);
        assert_eq!(
            new_faces.len(),
            faces.len() * 4,
            "Loop subdivision should produce 4x more triangles, got {}",
            new_faces.len()
        );
    }

    // -----------------------------------------------------------------------
    // test_loop_subdivide_valid_indices
    // -----------------------------------------------------------------------

    #[test]
    fn test_loop_subdivide_valid_indices() {
        let (verts, faces) = tetrahedron();
        let (new_verts, new_faces) = loop_subdivide(&verts, &faces);
        let max_idx = new_verts.len() as u32;
        for face in &new_faces {
            for &idx in face {
                assert!(idx < max_idx, "face index {idx} >= vertex count {max_idx}");
            }
        }
    }

    // -----------------------------------------------------------------------
    // test_smooth_config_default
    // -----------------------------------------------------------------------

    #[test]
    fn test_smooth_config_default() {
        let config = MeshSmoothingConfig::default();
        assert_eq!(config.iterations, 1);
        assert!((config.lambda - 0.5).abs() < 1e-6);
        assert!((config.mu - (-0.52)).abs() < 1e-6);
        assert!(config.preserve_boundary);
        assert_eq!(config.weight_mode, WeightMode::Uniform);
    }

    // -----------------------------------------------------------------------
    // test_smooth_config_taubin
    // -----------------------------------------------------------------------

    #[test]
    fn test_smooth_config_taubin() {
        let config = MeshSmoothingConfig::taubin();
        assert_eq!(config.iterations, 1);
        assert!((config.lambda - 0.5).abs() < 1e-6);
        assert!((config.mu - (-0.52)).abs() < 1e-6);
        assert!(config.preserve_boundary);
        assert_eq!(config.weight_mode, WeightMode::Cotangent);
    }
}
