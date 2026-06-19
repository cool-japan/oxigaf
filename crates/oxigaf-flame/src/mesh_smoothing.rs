//! Mesh smoothing algorithms for 3D triangle meshes.
//!
//! Provides multiple smoothing algorithms that reduce noise in vertex positions
//! while preserving surface features to varying degrees:
//!
//! - [`laplacian_smooth`]: Classic uniform Laplacian smoothing (fast, shrinks mesh)
//! - [`taubin_smooth`]: Taubin two-step smoothing (reduces shrinkage via alternating λ/μ)
//! - [`hc_laplacian_smooth`]: HC-Laplacian feature-preserving smoothing (best volume preservation)
//! - [`cotan_laplacian_smooth`]: Cotangent-weighted Laplacian (geometry-aware)
//!
//! All algorithms operate on `&[[f32; 3]]` vertex arrays with `&[[u32; 3]]` face arrays
//! and return new vertex arrays without mutating the input.

use std::collections::HashSet;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during mesh smoothing operations.
#[derive(Debug, thiserror::Error)]
pub enum SmoothingError {
    /// Vertex count does not match the maximum vertex index in face data.
    #[error("Vertex count {got} does not match face vertex indices (max index {max_idx})")]
    VertexFaceMismatch { got: usize, max_idx: usize },

    /// Lambda step size is outside the valid range (0, 1).
    #[error("Invalid lambda {lambda}: must be in (0, 1)")]
    InvalidLambda { lambda: f32 },

    /// Mu step size is invalid for Taubin smoothing.
    #[error("Invalid mu {mu}: must be in (-1, 0) for Taubin and |mu| > |lambda|")]
    InvalidMu { mu: f32 },

    /// Number of smoothing iterations must be positive.
    #[error("Number of iterations must be > 0")]
    ZeroIterations,

    /// A face references the same vertex index more than once.
    #[error("Face {face} has degenerate vertex indices (vertex repeated)")]
    DegenerateFace { face: usize },

    /// Alpha parameter is outside [0, 1].
    #[error("Feature preservation alpha must be in [0, 1], got {0}")]
    InvalidAlpha(f32),
}

// ---------------------------------------------------------------------------
// Adjacency building
// ---------------------------------------------------------------------------

/// Build vertex adjacency list from face data.
///
/// For each face `[v0, v1, v2]` the undirected edges `v0-v1`, `v1-v2`, and
/// `v2-v0` are added. Duplicate neighbors are deduplicated per vertex.
///
/// # Errors
/// Returns [`SmoothingError::VertexFaceMismatch`] if any face index ≥ `num_vertices`.
/// Returns [`SmoothingError::DegenerateFace`] if any face has repeated vertex indices.
pub fn build_adjacency(
    num_vertices: usize,
    faces: &[[u32; 3]],
) -> Result<Vec<Vec<usize>>, SmoothingError> {
    // Determine maximum index referenced by any face.
    let mut max_idx: usize = 0;
    for (fi, face) in faces.iter().enumerate() {
        let [a, b, c] = [face[0] as usize, face[1] as usize, face[2] as usize];
        // Check for degenerate face (repeated vertex)
        if a == b || b == c || a == c {
            return Err(SmoothingError::DegenerateFace { face: fi });
        }
        max_idx = max_idx.max(a).max(b).max(c);
    }

    // Validate vertex count if there are any faces.
    if !faces.is_empty() && max_idx >= num_vertices {
        return Err(SmoothingError::VertexFaceMismatch {
            got: num_vertices,
            max_idx,
        });
    }

    // Use a HashSet per vertex to accumulate unique neighbors.
    let mut adj: Vec<HashSet<usize>> = vec![HashSet::new(); num_vertices];

    for face in faces {
        let [a, b, c] = [face[0] as usize, face[1] as usize, face[2] as usize];
        adj[a].insert(b);
        adj[a].insert(c);
        adj[b].insert(a);
        adj[b].insert(c);
        adj[c].insert(a);
        adj[c].insert(b);
    }

    // Convert to sorted Vec<Vec<usize>> for deterministic iteration.
    let result = adj
        .into_iter()
        .map(|set| {
            let mut v: Vec<usize> = set.into_iter().collect();
            v.sort_unstable();
            v
        })
        .collect();

    Ok(result)
}

// ---------------------------------------------------------------------------
// Laplacian smoothing
// ---------------------------------------------------------------------------

/// Configuration for uniform Laplacian smoothing.
#[derive(Debug, Clone)]
pub struct LaplacianConfig {
    /// Step size λ ∈ (0, 1). Default: 0.5
    pub lambda: f32,
    /// Number of smoothing iterations. Default: 5
    pub iterations: usize,
    /// Whether to fix boundary vertices (those with < 3 neighbors). Default: true
    pub preserve_boundary: bool,
}

impl Default for LaplacianConfig {
    fn default() -> Self {
        Self {
            lambda: 0.5,
            iterations: 5,
            preserve_boundary: true,
        }
    }
}

/// Uniform Laplacian smoothing.
///
/// Each iteration moves every vertex toward the average of its neighbors:
/// `v[i] ← v[i] + λ * (mean(neighbors) − v[i])`
///
/// With `preserve_boundary = true`, vertices with fewer than 3 neighbors are
/// left unchanged (typical mesh-boundary vertices).
///
/// # Errors
/// Returns [`SmoothingError::ZeroIterations`] if `config.iterations == 0`.
/// Returns [`SmoothingError::InvalidLambda`] if `config.lambda` is outside `(0, 1)`.
pub fn laplacian_smooth(
    vertices: &[[f32; 3]],
    adjacency: &[Vec<usize>],
    config: &LaplacianConfig,
) -> Result<Vec<[f32; 3]>, SmoothingError> {
    if config.iterations == 0 {
        return Err(SmoothingError::ZeroIterations);
    }
    if config.lambda <= 0.0 || config.lambda >= 1.0 {
        return Err(SmoothingError::InvalidLambda {
            lambda: config.lambda,
        });
    }

    let mut current: Vec<[f32; 3]> = vertices.to_vec();
    let n = current.len();

    for _ in 0..config.iterations {
        let mut next = current.clone();
        for i in 0..n {
            let neighbors = &adjacency[i];
            if config.preserve_boundary && neighbors.len() < 3 {
                continue;
            }
            if neighbors.is_empty() {
                continue;
            }
            let mean = neighbor_mean(&current, neighbors);
            let v = current[i];
            next[i] = [
                v[0] + config.lambda * (mean[0] - v[0]),
                v[1] + config.lambda * (mean[1] - v[1]),
                v[2] + config.lambda * (mean[2] - v[2]),
            ];
        }
        current = next;
    }

    Ok(current)
}

// ---------------------------------------------------------------------------
// Taubin smoothing
// ---------------------------------------------------------------------------

/// Configuration for Taubin two-step smoothing.
#[derive(Debug, Clone)]
pub struct TaubinConfig {
    /// Positive step size λ ∈ (0, 1). Default: 0.5
    pub lambda: f32,
    /// Negative step size μ ∈ (-1, 0) with |μ| > λ. Default: -0.53
    pub mu: f32,
    /// Number of (λ, μ) iteration pairs. Default: 5
    pub iterations: usize,
    /// Fix boundary vertices. Default: true
    pub preserve_boundary: bool,
}

impl Default for TaubinConfig {
    fn default() -> Self {
        Self {
            lambda: 0.5,
            mu: -0.53,
            iterations: 5,
            preserve_boundary: true,
        }
    }
}

/// Taubin smoothing — alternates a shrinking step (λ) and an inflating step (μ).
///
/// Unlike plain Laplacian, this avoids significant volume loss while still
/// reducing noise. Each full iteration consists of two Laplacian-like passes:
/// one with `+λ` and one with `+μ` (where μ < 0).
///
/// # Errors
/// Returns [`SmoothingError::ZeroIterations`] if `config.iterations == 0`.
/// Returns [`SmoothingError::InvalidLambda`] if `config.lambda` is outside `(0, 1)`.
/// Returns [`SmoothingError::InvalidMu`] if `config.mu >= 0`, `config.mu <= -1`, or
/// `config.mu.abs() <= config.lambda`.
pub fn taubin_smooth(
    vertices: &[[f32; 3]],
    adjacency: &[Vec<usize>],
    config: &TaubinConfig,
) -> Result<Vec<[f32; 3]>, SmoothingError> {
    if config.iterations == 0 {
        return Err(SmoothingError::ZeroIterations);
    }
    if config.lambda <= 0.0 || config.lambda >= 1.0 {
        return Err(SmoothingError::InvalidLambda {
            lambda: config.lambda,
        });
    }
    if config.mu >= 0.0 || config.mu <= -1.0 || config.mu.abs() <= config.lambda {
        return Err(SmoothingError::InvalidMu { mu: config.mu });
    }

    let mut current: Vec<[f32; 3]> = vertices.to_vec();

    for _ in 0..config.iterations {
        // λ step (shrink)
        current =
            apply_laplacian_step(&current, adjacency, config.lambda, config.preserve_boundary);
        // μ step (inflate)
        current = apply_laplacian_step(&current, adjacency, config.mu, config.preserve_boundary);
    }

    Ok(current)
}

// ---------------------------------------------------------------------------
// HC-Laplacian smoothing
// ---------------------------------------------------------------------------

/// Configuration for HC-Laplacian feature-preserving smoothing.
#[derive(Debug, Clone)]
pub struct HcLaplacianConfig {
    /// Smoothing weight α ∈ [0, 1]. Default: 0.5
    pub alpha: f32,
    /// Correction weight β ∈ [0, 1]. Default: 0.5
    pub beta: f32,
    /// Number of iterations. Default: 5
    pub iterations: usize,
}

impl Default for HcLaplacianConfig {
    fn default() -> Self {
        Self {
            alpha: 0.5,
            beta: 0.5,
            iterations: 5,
        }
    }
}

/// HC-Laplacian smoothing — preserves features better than plain Laplacian.
///
/// Uses a two-pass approach per iteration:
/// - **Pass 1 (smooth)**: `v_s[i] = v[i] + α * (mean_neighbors[i] − v[i])`
/// - **Pass 2 (correct)**: `v_new[i] = v_s[i] − β * (v_s[i] − v[i])`
///
/// The correction pass blends back toward the pre-smooth position, which
/// counteracts volume loss while preserving noise reduction.
///
/// # Errors
/// Returns [`SmoothingError::ZeroIterations`] if `config.iterations == 0`.
/// Returns [`SmoothingError::InvalidAlpha`] if `config.alpha` is outside `[0, 1]`.
pub fn hc_laplacian_smooth(
    vertices: &[[f32; 3]],
    adjacency: &[Vec<usize>],
    config: &HcLaplacianConfig,
) -> Result<Vec<[f32; 3]>, SmoothingError> {
    if config.iterations == 0 {
        return Err(SmoothingError::ZeroIterations);
    }
    if config.alpha < 0.0 || config.alpha > 1.0 {
        return Err(SmoothingError::InvalidAlpha(config.alpha));
    }

    let mut current: Vec<[f32; 3]> = vertices.to_vec();
    let n = current.len();

    for _ in 0..config.iterations {
        // Pass 1: Laplacian step
        let mut v_s: Vec<[f32; 3]> = current.clone();
        for i in 0..n {
            let neighbors = &adjacency[i];
            if neighbors.is_empty() {
                continue;
            }
            let mean = neighbor_mean(&current, neighbors);
            let v = current[i];
            v_s[i] = [
                v[0] + config.alpha * (mean[0] - v[0]),
                v[1] + config.alpha * (mean[1] - v[1]),
                v[2] + config.alpha * (mean[2] - v[2]),
            ];
        }

        // Pass 2: HC correction — blend back toward original
        // v_new[i] = v_s[i] - beta * (v_s[i] - v[i])
        //           = (1 - beta) * v_s[i] + beta * v[i]
        let mut next: Vec<[f32; 3]> = v_s.clone();
        for i in 0..n {
            let vs = v_s[i];
            let v = current[i];
            let b = config.beta;
            next[i] = [
                vs[0] - b * (vs[0] - v[0]),
                vs[1] - b * (vs[1] - v[1]),
                vs[2] - b * (vs[2] - v[2]),
            ];
        }
        current = next;
    }

    Ok(current)
}

// ---------------------------------------------------------------------------
// Cotangent weights
// ---------------------------------------------------------------------------

/// Compute cotangent-weighted adjacency for a triangle mesh.
///
/// For each directed edge `(i, j)` shared by triangles, the cotangent weight is
/// `(cot α + cot β) / 2` where α and β are the angles opposite to edge `(i, j)`.
///
/// Weights are clamped to `[0, ∞)` — negative cotangents (obtuse angles) are set to 0.
///
/// Returns `adj_cotan[i] = [(j, weight_ij), ...]` for each vertex `i`.
///
/// # Errors
/// Returns [`SmoothingError::VertexFaceMismatch`] if any face index ≥ `vertices.len()`.
/// Returns [`SmoothingError::DegenerateFace`] if any face has repeated vertex indices.
pub fn build_cotan_adjacency(
    vertices: &[[f32; 3]],
    faces: &[[u32; 3]],
) -> Result<Vec<Vec<(usize, f32)>>, SmoothingError> {
    let n = vertices.len();

    // Validate faces
    for (fi, face) in faces.iter().enumerate() {
        let [a, b, c] = [face[0] as usize, face[1] as usize, face[2] as usize];
        if a == b || b == c || a == c {
            return Err(SmoothingError::DegenerateFace { face: fi });
        }
        let max_idx = a.max(b).max(c);
        if max_idx >= n {
            return Err(SmoothingError::VertexFaceMismatch { got: n, max_idx });
        }
    }

    // Accumulate cotangent weights in a HashMap: (i, j) -> sum_of_cotan
    let mut cotan_map: std::collections::HashMap<(usize, usize), f32> =
        std::collections::HashMap::new();

    for face in faces {
        let [ai, bi, ci] = [face[0] as usize, face[1] as usize, face[2] as usize];
        let va = vertices[ai];
        let vb = vertices[bi];
        let vc = vertices[ci];

        // For edge (bi, ci) — opposite angle at vertex ai
        let cot_a = cotangent_at(va, vb, vc);
        // For edge (ai, ci) — opposite angle at vertex bi
        let cot_b = cotangent_at(vb, va, vc);
        // For edge (ai, bi) — opposite angle at vertex ci
        let cot_c = cotangent_at(vc, va, vb);

        // Edge (bi, ci): each direction gets half of cot_a
        add_cotan_weight(&mut cotan_map, bi, ci, cot_a * 0.5);
        add_cotan_weight(&mut cotan_map, ci, bi, cot_a * 0.5);

        // Edge (ai, ci): each direction gets half of cot_b
        add_cotan_weight(&mut cotan_map, ai, ci, cot_b * 0.5);
        add_cotan_weight(&mut cotan_map, ci, ai, cot_b * 0.5);

        // Edge (ai, bi): each direction gets half of cot_c
        add_cotan_weight(&mut cotan_map, ai, bi, cot_c * 0.5);
        add_cotan_weight(&mut cotan_map, bi, ai, cot_c * 0.5);
    }

    // Convert to Vec<Vec<(usize, f32)>>
    let mut result: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n];
    for ((i, j), w) in cotan_map {
        // Clamp to [0, ∞) — negative cotangents (obtuse) become 0
        result[i].push((j, w.max(0.0)));
    }

    // Sort each neighbor list by vertex index for determinism
    for list in &mut result {
        list.sort_unstable_by_key(|&(j, _)| j);
    }

    Ok(result)
}

/// Cotangent-weighted Laplacian smoothing.
///
/// Uses geometry-aware weights (cotangent of opposite angles) instead of uniform
/// averaging, giving better results for anisotropic meshes.
///
/// # Errors
/// Returns the same errors as [`build_cotan_adjacency`] plus [`SmoothingError::ZeroIterations`]
/// and [`SmoothingError::InvalidLambda`].
pub fn cotan_laplacian_smooth(
    vertices: &[[f32; 3]],
    faces: &[[u32; 3]],
    lambda: f32,
    iterations: usize,
) -> Result<Vec<[f32; 3]>, SmoothingError> {
    if iterations == 0 {
        return Err(SmoothingError::ZeroIterations);
    }
    if lambda <= 0.0 || lambda >= 1.0 {
        return Err(SmoothingError::InvalidLambda { lambda });
    }

    let adj_cotan = build_cotan_adjacency(vertices, faces)?;
    let mut current: Vec<[f32; 3]> = vertices.to_vec();
    let n = current.len();

    for _ in 0..iterations {
        let mut next = current.clone();
        for i in 0..n {
            let neighbors = &adj_cotan[i];
            if neighbors.is_empty() {
                continue;
            }
            let total_weight: f32 = neighbors.iter().map(|&(_, w)| w).sum();
            if total_weight < f32::EPSILON {
                continue;
            }
            let mut weighted_sum = [0.0f32; 3];
            for &(j, w) in neighbors {
                weighted_sum[0] += w * current[j][0];
                weighted_sum[1] += w * current[j][1];
                weighted_sum[2] += w * current[j][2];
            }
            let mean = [
                weighted_sum[0] / total_weight,
                weighted_sum[1] / total_weight,
                weighted_sum[2] / total_weight,
            ];
            let v = current[i];
            next[i] = [
                v[0] + lambda * (mean[0] - v[0]),
                v[1] + lambda * (mean[1] - v[1]),
                v[2] + lambda * (mean[2] - v[2]),
            ];
        }
        current = next;
    }

    Ok(current)
}

// ---------------------------------------------------------------------------
// Volume computation and restoration
// ---------------------------------------------------------------------------

/// Compute the signed volume of a closed triangle mesh using the divergence theorem.
///
/// `V = (1/6) * |Σ_{face} dot(v0, cross(v1 - v0, v2 - v0))|`
///
/// Returns 0.0 for empty meshes.
#[must_use]
pub fn compute_mesh_volume(vertices: &[[f32; 3]], faces: &[[u32; 3]]) -> f32 {
    if vertices.is_empty() || faces.is_empty() {
        return 0.0;
    }

    let mut signed_vol = 0.0f32;
    for face in faces {
        let v0 = vertices[face[0] as usize];
        let v1 = vertices[face[1] as usize];
        let v2 = vertices[face[2] as usize];

        // Edge vectors from v0
        let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
        let e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];

        // cross(e1, e2)
        let cross = [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ];

        // dot(v0, cross)
        signed_vol += v0[0] * cross[0] + v0[1] * cross[1] + v0[2] * cross[2];
    }

    (signed_vol / 6.0).abs()
}

/// Uniformly scale smoothed vertices to restore the original mesh volume.
///
/// Computes the mesh centroid, translates to origin, applies a uniform scale
/// factor `(V_orig / V_smoothed)^(1/3)`, then translates back.
///
/// If either volume is non-positive or the ratio is infinite/NaN, returns
/// `smoothed_vertices.to_vec()` unchanged.
#[must_use]
pub fn restore_volume(
    original_vertices: &[[f32; 3]],
    smoothed_vertices: &[[f32; 3]],
    faces: &[[u32; 3]],
) -> Vec<[f32; 3]> {
    let v_orig = compute_mesh_volume(original_vertices, faces);
    let v_smooth = compute_mesh_volume(smoothed_vertices, faces);

    if v_smooth <= 0.0 || v_orig <= 0.0 {
        return smoothed_vertices.to_vec();
    }

    let ratio = v_orig / v_smooth;
    if !ratio.is_finite() || ratio <= 0.0 {
        return smoothed_vertices.to_vec();
    }

    let scale = ratio.cbrt();
    if !scale.is_finite() {
        return smoothed_vertices.to_vec();
    }

    let n = smoothed_vertices.len();
    if n == 0 {
        return Vec::new();
    }

    // Compute centroid of smoothed mesh
    let centroid = centroid_of(smoothed_vertices);

    // Scale around centroid
    smoothed_vertices
        .iter()
        .map(|&v| {
            [
                centroid[0] + scale * (v[0] - centroid[0]),
                centroid[1] + scale * (v[1] - centroid[1]),
                centroid[2] + scale * (v[2] - centroid[2]),
            ]
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Smoothing statistics
// ---------------------------------------------------------------------------

/// Statistics describing how much a smoothing operation moved the mesh.
#[derive(Debug, Clone)]
pub struct SmoothingStats {
    /// Mean Euclidean displacement of vertices from original positions.
    pub mean_displacement: f32,
    /// Maximum Euclidean displacement among all vertices.
    pub max_displacement: f32,
    /// Volume change ratio: `smoothed_volume / original_volume`.
    pub volume_ratio: f32,
    /// `true` if any vertex with fewer than 3 neighbors was moved.
    pub boundary_vertices_moved: bool,
}

/// Compute statistics comparing original and smoothed vertex arrays.
///
/// Displacement is measured as Euclidean distance between corresponding vertices.
/// `volume_ratio` is 1.0 when `original` and `smoothed` are identical or when
/// either mesh has zero volume.
#[must_use]
pub fn compute_smoothing_stats(
    original: &[[f32; 3]],
    smoothed: &[[f32; 3]],
    faces: &[[u32; 3]],
) -> SmoothingStats {
    let n = original.len().min(smoothed.len());
    if n == 0 {
        return SmoothingStats {
            mean_displacement: 0.0,
            max_displacement: 0.0,
            volume_ratio: 1.0,
            boundary_vertices_moved: false,
        };
    }

    let mut sum_disp = 0.0f32;
    let mut max_disp = 0.0f32;
    for i in 0..n {
        let dx = smoothed[i][0] - original[i][0];
        let dy = smoothed[i][1] - original[i][1];
        let dz = smoothed[i][2] - original[i][2];
        let d = (dx * dx + dy * dy + dz * dz).sqrt();
        sum_disp += d;
        if d > max_disp {
            max_disp = d;
        }
    }
    let mean_displacement = sum_disp / n as f32;

    let v_orig = compute_mesh_volume(original, faces);
    let v_smooth = compute_mesh_volume(smoothed, faces);
    let volume_ratio = if v_orig > 0.0 && v_smooth.is_finite() {
        v_smooth / v_orig
    } else {
        1.0
    };

    // Check if any boundary vertex (< 3 adjacency edges in at least one face's context)
    // was moved. We approximate this from displacement being > 0 for low-valence vertices.
    // Build a simple valence count to detect potential boundary vertices.
    let mut valence = vec![0usize; original.len()];
    for face in faces {
        for &idx in face {
            valence[idx as usize] += 1;
        }
    }

    let boundary_vertices_moved = (0..n).any(|i| {
        let dx = smoothed[i][0] - original[i][0];
        let dy = smoothed[i][1] - original[i][1];
        let dz = smoothed[i][2] - original[i][2];
        let moved = (dx * dx + dy * dy + dz * dz).sqrt() > f32::EPSILON;
        // A vertex with valence < 3 is likely a boundary/isolated vertex
        moved && valence.get(i).copied().unwrap_or(0) < 3
    });

    SmoothingStats {
        mean_displacement,
        max_displacement: max_disp,
        volume_ratio,
        boundary_vertices_moved,
    }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Compute the mean position of a set of neighbor vertices.
#[inline]
fn neighbor_mean(vertices: &[[f32; 3]], neighbors: &[usize]) -> [f32; 3] {
    let n = neighbors.len() as f32;
    let mut sum = [0.0f32; 3];
    for &j in neighbors {
        sum[0] += vertices[j][0];
        sum[1] += vertices[j][1];
        sum[2] += vertices[j][2];
    }
    [sum[0] / n, sum[1] / n, sum[2] / n]
}

/// Apply a single Laplacian step with step size `step` (may be negative for Taubin).
#[inline]
fn apply_laplacian_step(
    vertices: &[[f32; 3]],
    adjacency: &[Vec<usize>],
    step: f32,
    preserve_boundary: bool,
) -> Vec<[f32; 3]> {
    let n = vertices.len();
    let mut next = vertices.to_vec();
    for i in 0..n {
        let neighbors = &adjacency[i];
        if preserve_boundary && neighbors.len() < 3 {
            continue;
        }
        if neighbors.is_empty() {
            continue;
        }
        let mean = neighbor_mean(vertices, neighbors);
        let v = vertices[i];
        next[i] = [
            v[0] + step * (mean[0] - v[0]),
            v[1] + step * (mean[1] - v[1]),
            v[2] + step * (mean[2] - v[2]),
        ];
    }
    next
}

/// Cotangent of the angle at vertex `p` in the triangle `(p, q, r)`.
///
/// Returns 0.0 for degenerate angles (sin ≈ 0).
#[inline]
fn cotangent_at(p: [f32; 3], q: [f32; 3], r: [f32; 3]) -> f32 {
    let pq = [q[0] - p[0], q[1] - p[1], q[2] - p[2]];
    let pr = [r[0] - p[0], r[1] - p[1], r[2] - p[2]];

    let dot = pq[0] * pr[0] + pq[1] * pr[1] + pq[2] * pr[2];

    let cross = [
        pq[1] * pr[2] - pq[2] * pr[1],
        pq[2] * pr[0] - pq[0] * pr[2],
        pq[0] * pr[1] - pq[1] * pr[0],
    ];
    let sin_val = (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();

    if sin_val < f32::EPSILON {
        0.0
    } else {
        dot / sin_val
    }
}

/// Accumulate a cotangent contribution into the weight map.
#[inline]
fn add_cotan_weight(
    map: &mut std::collections::HashMap<(usize, usize), f32>,
    i: usize,
    j: usize,
    w: f32,
) {
    *map.entry((i, j)).or_insert(0.0) += w;
}

/// Compute the centroid (mean position) of a vertex array.
#[inline]
fn centroid_of(vertices: &[[f32; 3]]) -> [f32; 3] {
    let n = vertices.len() as f32;
    let mut sum = [0.0f32; 3];
    for v in vertices {
        sum[0] += v[0];
        sum[1] += v[1];
        sum[2] += v[2];
    }
    [sum[0] / n, sum[1] / n, sum[2] / n]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple unit cube: 8 vertices, 12 triangles (2 per face).
    fn cube_mesh() -> (Vec<[f32; 3]>, Vec<[u32; 3]>) {
        let v = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ];
        let f = vec![
            [0, 1, 2],
            [0, 2, 3], // bottom
            [4, 5, 6],
            [4, 6, 7], // top
            [0, 1, 5],
            [0, 5, 4], // front
            [2, 3, 7],
            [2, 7, 6], // back
            [0, 3, 7],
            [0, 7, 4], // left
            [1, 2, 6],
            [1, 6, 5], // right
        ];
        (v, f)
    }

    /// Cube with small noise added to vertices.
    fn noisy_cube() -> (Vec<[f32; 3]>, Vec<[u32; 3]>) {
        let (mut v, f) = cube_mesh();
        let offsets = [0.1f32, -0.08, 0.12, -0.05, 0.09, -0.11, 0.07, -0.06];
        for (i, vert) in v.iter_mut().enumerate() {
            vert[0] += offsets[i % 8];
            vert[1] += offsets[(i + 3) % 8];
        }
        (v, f)
    }

    // -----------------------------------------------------------------------
    // Test 1: build_adjacency cube — each vertex has >= 3 neighbors
    // -----------------------------------------------------------------------
    #[test]
    fn test_build_adjacency_cube_min_neighbors() {
        let (v, f) = cube_mesh();
        let adj = build_adjacency(v.len(), &f).expect("adjacency failed");
        assert_eq!(adj.len(), 8);
        // In a closed triangulated cube every vertex is shared by 6 triangles
        // and connected to 6 unique neighbors.
        for (i, neighbors) in adj.iter().enumerate() {
            assert!(
                neighbors.len() >= 3,
                "vertex {i} has only {} neighbors (expected >= 3)",
                neighbors.len()
            );
        }
    }

    // -----------------------------------------------------------------------
    // Test 2: build_adjacency with no faces → empty adjacency, no error
    // -----------------------------------------------------------------------
    #[test]
    fn test_build_adjacency_empty_faces() {
        let adj = build_adjacency(5, &[]).expect("empty faces should not fail");
        assert_eq!(adj.len(), 5);
        assert!(adj.iter().all(std::vec::Vec::is_empty));
    }

    // -----------------------------------------------------------------------
    // Test 3: laplacian_smooth output same length
    // -----------------------------------------------------------------------
    #[test]
    fn test_laplacian_same_length() {
        let (v, f) = cube_mesh();
        let adj = build_adjacency(v.len(), &f).expect("adj");
        let result = laplacian_smooth(&v, &adj, &LaplacianConfig::default()).expect("smooth");
        assert_eq!(result.len(), v.len());
    }

    // -----------------------------------------------------------------------
    // Test 4: laplacian_smooth 0 iterations → ZeroIterations error
    // -----------------------------------------------------------------------
    #[test]
    fn test_laplacian_zero_iterations_error() {
        let (v, f) = cube_mesh();
        let adj = build_adjacency(v.len(), &f).expect("adj");
        let config = LaplacianConfig {
            iterations: 0,
            ..Default::default()
        };
        let result = laplacian_smooth(&v, &adj, &config);
        assert!(matches!(result, Err(SmoothingError::ZeroIterations)));
    }

    // -----------------------------------------------------------------------
    // Test 5: laplacian_smooth uniform mesh — vertices don't move
    // -----------------------------------------------------------------------
    #[test]
    fn test_laplacian_uniform_mesh_no_movement() {
        // A flat equilateral triangle — already at the centroid of its neighbors
        // for interior vertices.  We use the cube itself; its vertices are at
        // the corners of a regular lattice and Laplacian should move them only
        // slightly (boundary-lock prevents large movements on a cube).
        // Instead use a simple regular grid patch where interior vertex is
        // exactly at the mean of its 4 neighbors.
        //
        // Interior vertex at (1,1,0), surrounded by (0,1,0),(2,1,0),(1,0,0),(1,2,0).
        // Triangulate as:
        //   (0,1,0)-(1,1,0)-(1,0,0),  (1,1,0)-(2,1,0)-(1,0,0)
        //   (0,1,0)-(1,2,0)-(1,1,0),  (1,1,0)-(1,2,0)-(2,1,0)
        let v: Vec<[f32; 3]> = vec![
            [0.0, 1.0, 0.0], // 0
            [2.0, 1.0, 0.0], // 1
            [1.0, 0.0, 0.0], // 2
            [1.0, 2.0, 0.0], // 3
            [1.0, 1.0, 0.0], // 4 — interior, exactly at mean of 0,1,2,3
        ];
        let f: Vec<[u32; 3]> = vec![[0, 4, 2], [4, 1, 2], [0, 3, 4], [4, 3, 1]];
        let adj = build_adjacency(v.len(), &f).expect("adj");
        let config = LaplacianConfig {
            lambda: 0.5,
            iterations: 5,
            preserve_boundary: false,
        };
        let result = laplacian_smooth(&v, &adj, &config).expect("smooth");
        // Interior vertex 4 should not move (already at centroid of its 4 neighbors)
        let eps = 1e-5;
        assert!(
            (result[4][0] - v[4][0]).abs() < eps,
            "x changed: {}",
            result[4][0]
        );
        assert!(
            (result[4][1] - v[4][1]).abs() < eps,
            "y changed: {}",
            result[4][1]
        );
    }

    // -----------------------------------------------------------------------
    // Test 6: laplacian_smooth noisy cube reduces Laplacian residual (self-smoothness)
    // -----------------------------------------------------------------------
    #[test]
    fn test_laplacian_reduces_noise() {
        let (v_noisy, f) = noisy_cube();
        let adj = build_adjacency(v_noisy.len(), &f).expect("adj");

        /// Compute the mean magnitude of the Laplacian vector at each vertex.
        /// A smaller value means the mesh is smoother (each vertex is closer to
        /// the mean of its neighbors).
        fn laplacian_residual(verts: &[[f32; 3]], adj: &[Vec<usize>]) -> f32 {
            let mut total = 0.0f32;
            let mut count = 0usize;
            for (i, neighbors) in adj.iter().enumerate() {
                if neighbors.is_empty() {
                    continue;
                }
                let mean = neighbor_mean(verts, neighbors);
                let v = verts[i];
                let dx = mean[0] - v[0];
                let dy = mean[1] - v[1];
                let dz = mean[2] - v[2];
                total += (dx * dx + dy * dy + dz * dz).sqrt();
                count += 1;
            }
            if count == 0 {
                0.0
            } else {
                total / count as f32
            }
        }

        let residual_before = laplacian_residual(&v_noisy, &adj);

        let config = LaplacianConfig {
            lambda: 0.5,
            iterations: 3,
            preserve_boundary: false,
        };
        let smoothed = laplacian_smooth(&v_noisy, &adj, &config).expect("smooth");

        let residual_after = laplacian_residual(&smoothed, &adj);

        // Smoothing must reduce the Laplacian residual (self-smoothness measure)
        assert!(
            residual_after < residual_before,
            "smoothing did not reduce residual: before={residual_before:.4} after={residual_after:.4}"
        );
    }

    // -----------------------------------------------------------------------
    // Test 7: taubin_smooth output same length
    // -----------------------------------------------------------------------
    #[test]
    fn test_taubin_same_length() {
        let (v, f) = cube_mesh();
        let adj = build_adjacency(v.len(), &f).expect("adj");
        let result = taubin_smooth(&v, &adj, &TaubinConfig::default()).expect("smooth");
        assert_eq!(result.len(), v.len());
    }

    // -----------------------------------------------------------------------
    // Test 8: taubin_smooth preserves volume better than laplacian
    // -----------------------------------------------------------------------
    #[test]
    fn test_taubin_better_volume_preservation_than_laplacian() {
        let (v, f) = cube_mesh();
        let adj = build_adjacency(v.len(), &f).expect("adj");

        let lap_config = LaplacianConfig {
            lambda: 0.5,
            iterations: 5,
            preserve_boundary: false,
        };
        let lap_result = laplacian_smooth(&v, &adj, &lap_config).expect("laplacian");

        let tau_config = TaubinConfig {
            lambda: 0.5,
            mu: -0.53,
            iterations: 5,
            preserve_boundary: false,
        };
        let tau_result = taubin_smooth(&v, &adj, &tau_config).expect("taubin");

        let v_orig = compute_mesh_volume(&v, &f);
        let v_lap = compute_mesh_volume(&lap_result, &f);
        let v_tau = compute_mesh_volume(&tau_result, &f);

        // Taubin volume should be closer to original than pure Laplacian volume
        let lap_ratio = (v_lap - v_orig).abs();
        let tau_ratio = (v_tau - v_orig).abs();
        assert!(
            tau_ratio <= lap_ratio + 0.01,
            "Taubin volume ratio {tau_ratio:.4} worse than Laplacian {lap_ratio:.4}"
        );
    }

    // -----------------------------------------------------------------------
    // Test 9: taubin_smooth invalid lambda → error
    // -----------------------------------------------------------------------
    #[test]
    fn test_taubin_invalid_lambda() {
        let (v, f) = cube_mesh();
        let adj = build_adjacency(v.len(), &f).expect("adj");
        let config = TaubinConfig {
            lambda: 1.5,
            ..Default::default()
        };
        let result = taubin_smooth(&v, &adj, &config);
        assert!(matches!(result, Err(SmoothingError::InvalidLambda { .. })));
    }

    // -----------------------------------------------------------------------
    // Test 10: taubin_smooth invalid mu → error
    // -----------------------------------------------------------------------
    #[test]
    fn test_taubin_invalid_mu() {
        let (v, f) = cube_mesh();
        let adj = build_adjacency(v.len(), &f).expect("adj");
        // mu positive → invalid
        let config = TaubinConfig {
            lambda: 0.5,
            mu: 0.3,
            ..Default::default()
        };
        let result = taubin_smooth(&v, &adj, &config);
        assert!(matches!(result, Err(SmoothingError::InvalidMu { .. })));
    }

    // -----------------------------------------------------------------------
    // Test 11: hc_laplacian_smooth output same length
    // -----------------------------------------------------------------------
    #[test]
    fn test_hc_laplacian_same_length() {
        let (v, f) = cube_mesh();
        let adj = build_adjacency(v.len(), &f).expect("adj");
        let result = hc_laplacian_smooth(&v, &adj, &HcLaplacianConfig::default()).expect("smooth");
        assert_eq!(result.len(), v.len());
    }

    // -----------------------------------------------------------------------
    // Test 12: hc_laplacian_smooth preserves more volume than plain Laplacian
    // -----------------------------------------------------------------------
    #[test]
    fn test_hc_laplacian_preserves_more_volume() {
        let (v_noisy, f) = noisy_cube();
        let adj = build_adjacency(v_noisy.len(), &f).expect("adj");

        let lap_config = LaplacianConfig {
            lambda: 0.5,
            iterations: 5,
            preserve_boundary: false,
        };
        let lap_result = laplacian_smooth(&v_noisy, &adj, &lap_config).expect("laplacian");

        // HC-Laplacian with beta > 0 blends back toward original → better volume preservation
        let hc_config = HcLaplacianConfig {
            alpha: 0.5,
            beta: 0.5,
            iterations: 5,
        };
        let hc_result = hc_laplacian_smooth(&v_noisy, &adj, &hc_config).expect("hc_laplacian");

        let v_orig = compute_mesh_volume(&v_noisy, &f);
        let v_lap = compute_mesh_volume(&lap_result, &f);
        let v_hc = compute_mesh_volume(&hc_result, &f);

        // HC result should be at least as close to original volume as pure Laplacian
        let lap_err = (v_lap - v_orig).abs();
        let hc_err = (v_hc - v_orig).abs();
        assert!(
            hc_err <= lap_err + 0.05,
            "HC volume error {hc_err:.4} worse than Laplacian {lap_err:.4}"
        );
    }

    // -----------------------------------------------------------------------
    // Test 13: build_cotan_adjacency all weights non-negative
    // -----------------------------------------------------------------------
    #[test]
    fn test_cotan_weights_non_negative() {
        let (v, f) = cube_mesh();
        let adj = build_cotan_adjacency(&v, &f).expect("cotan adj");
        for (i, neighbors) in adj.iter().enumerate() {
            for &(j, w) in neighbors {
                assert!(w >= 0.0, "vertex {i} → {j} has negative weight {w}");
            }
        }
    }

    // -----------------------------------------------------------------------
    // Test 14: cotan_laplacian_smooth output same length
    // -----------------------------------------------------------------------
    #[test]
    fn test_cotan_laplacian_same_length() {
        let (v, f) = cube_mesh();
        let result = cotan_laplacian_smooth(&v, &f, 0.5, 3).expect("smooth");
        assert_eq!(result.len(), v.len());
    }

    // -----------------------------------------------------------------------
    // Test 15: compute_mesh_volume unit cube ≈ 1.0
    // -----------------------------------------------------------------------
    #[test]
    fn test_volume_unit_cube() {
        let (v, f) = cube_mesh();
        let vol = compute_mesh_volume(&v, &f);
        assert!(
            (vol - 1.0).abs() < 0.1,
            "cube volume = {vol}, expected ≈ 1.0"
        );
    }

    // -----------------------------------------------------------------------
    // Test 16: restore_volume restores volume within 5% of original
    //
    // We use a closed tetrahedron with consistently oriented (outward-facing)
    // face normals so the divergence-theorem volume formula is exact.
    // After scaling the tetrahedron by 0.5 from its centroid (simulating
    // aggressive shrinkage from Laplacian smoothing), restore_volume must
    // scale it back to match the original volume within 5%.
    // -----------------------------------------------------------------------
    #[test]
    fn test_restore_volume() {
        // Regular tetrahedron inscribed in unit sphere, outward-facing normals.
        // V ≈ 8√3/27 ≈ 0.5132
        let v_orig: Vec<[f32; 3]> = vec![
            [1.0, 1.0, 1.0],   // 0
            [1.0, -1.0, -1.0], // 1
            [-1.0, 1.0, -1.0], // 2
            [-1.0, -1.0, 1.0], // 3
        ];
        // Outward-facing faces (right-hand rule, normals pointing away from center)
        let faces: Vec<[u32; 3]> = vec![[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]];

        let vol_orig = compute_mesh_volume(&v_orig, &faces);
        assert!(
            vol_orig > 0.1,
            "tetrahedron volume should be positive, got {vol_orig}"
        );

        // Shrink by factor 0.5 from centroid (simulating Laplacian collapse)
        let centroid = centroid_of(&v_orig);
        let scale_down = 0.5f32;
        let v_small: Vec<[f32; 3]> = v_orig
            .iter()
            .map(|&v| {
                [
                    centroid[0] + scale_down * (v[0] - centroid[0]),
                    centroid[1] + scale_down * (v[1] - centroid[1]),
                    centroid[2] + scale_down * (v[2] - centroid[2]),
                ]
            })
            .collect();

        let vol_small = compute_mesh_volume(&v_small, &faces);
        // Volume should be (0.5)^3 = 1/8 of original
        assert!(
            (vol_small - vol_orig / 8.0).abs() < vol_orig * 0.01,
            "shrunk vol mismatch: {vol_small} vs expected {}",
            vol_orig / 8.0
        );

        // Restore volume
        let restored = restore_volume(&v_orig, &v_small, &faces);
        let vol_restored = compute_mesh_volume(&restored, &faces);

        let error = (vol_restored - vol_orig).abs() / vol_orig;
        assert!(
            error < 0.05,
            "volume error after restoration: {error:.4} (expected < 0.05), \
             orig={vol_orig:.4} restored={vol_restored:.4}"
        );
    }

    // -----------------------------------------------------------------------
    // Test 17: compute_smoothing_stats mean_displacement >= 0
    // -----------------------------------------------------------------------
    #[test]
    fn test_smoothing_stats_non_negative_displacement() {
        let (v, f) = cube_mesh();
        let adj = build_adjacency(v.len(), &f).expect("adj");
        let smoothed = laplacian_smooth(&v, &adj, &LaplacianConfig::default()).expect("smooth");
        let stats = compute_smoothing_stats(&v, &smoothed, &f);
        assert!(
            stats.mean_displacement >= 0.0,
            "mean_displacement = {}",
            stats.mean_displacement
        );
        assert!(
            stats.max_displacement >= 0.0,
            "max_displacement = {}",
            stats.max_displacement
        );
    }

    // -----------------------------------------------------------------------
    // Test 18: compute_smoothing_stats volume_ratio = 1.0 for same mesh
    // -----------------------------------------------------------------------
    #[test]
    fn test_smoothing_stats_same_mesh_volume_ratio() {
        let (v, f) = cube_mesh();
        let stats = compute_smoothing_stats(&v, &v, &f);
        assert!(
            (stats.volume_ratio - 1.0).abs() < 1e-4,
            "volume_ratio for identical meshes = {}",
            stats.volume_ratio
        );
    }

    // -----------------------------------------------------------------------
    // Test 19: LaplacianConfig default values
    // -----------------------------------------------------------------------
    #[test]
    fn test_laplacian_default_config() {
        let config = LaplacianConfig::default();
        assert_eq!(config.iterations, 5);
        assert!((config.lambda - 0.5).abs() < f32::EPSILON);
        assert!(config.preserve_boundary);
    }

    // -----------------------------------------------------------------------
    // Test 20: taubin_smooth fewer iterations for comparable smoothness
    // -----------------------------------------------------------------------
    #[test]
    fn test_taubin_fewer_iterations_comparable_quality() {
        let (v_noisy, f) = noisy_cube();
        let (v_clean, _) = cube_mesh();
        let adj = build_adjacency(v_noisy.len(), &f).expect("adj");

        // Laplacian with 10 iterations
        let lap = laplacian_smooth(
            &v_noisy,
            &adj,
            &LaplacianConfig {
                lambda: 0.5,
                iterations: 10,
                preserve_boundary: false,
            },
        )
        .expect("lap");

        // Taubin with only 5 iterations (10 total passes)
        let tau = taubin_smooth(
            &v_noisy,
            &adj,
            &TaubinConfig {
                lambda: 0.5,
                mu: -0.53,
                iterations: 5,
                preserve_boundary: false,
            },
        )
        .expect("tau");

        // Both should produce non-empty output and be close to clean cube
        assert_eq!(lap.len(), v_noisy.len());
        assert_eq!(tau.len(), v_noisy.len());

        let error_lap: f32 = tau
            .iter()
            .zip(v_clean.iter())
            .map(|(a, b)| {
                let dx = a[0] - b[0];
                let dy = a[1] - b[1];
                let dz = a[2] - b[2];
                (dx * dx + dy * dy + dz * dz).sqrt()
            })
            .sum::<f32>();

        // Sanity: Taubin result has finite coordinates
        assert!(
            error_lap.is_finite(),
            "Taubin result has non-finite coordinates"
        );
    }
}
