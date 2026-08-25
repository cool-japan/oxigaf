//! Graph Laplacian spectral analysis of the FLAME mesh.
//!
//! Provides spectral decomposition, filtering, clustering, and Laplacian smoothing
//! using the mesh adjacency graph.
#![allow(clippy::too_many_lines)]
#![allow(clippy::cast_precision_loss)]

use nalgebra as na;
use std::collections::BTreeMap;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors for spectral analysis operations.
#[derive(Debug, thiserror::Error)]
pub enum SpectralError {
    /// No vertices provided.
    #[error("Empty mesh: no vertices")]
    EmptyMesh,
    /// A vertex with no neighbors was encountered.
    #[error("Isolated vertex {idx}: no neighbors in mesh")]
    IsolatedVertex { idx: usize },
    /// Signal length does not match vertex count.
    #[error("Dimension mismatch: signal has {sig}, mesh has {mesh} vertices")]
    DimensionMismatch { sig: usize, mesh: usize },
    /// Fewer eigenvectors computed than requested.
    #[error("Not enough eigenvectors: requested {k}, computed {available}")]
    InsufficientBasis { k: usize, available: usize },
    /// Power iteration failed to converge.
    #[error("Diverged after {iters} power iterations")]
    PowerIterationDiverged { iters: usize },
    /// A configuration parameter is invalid.
    #[error("Invalid config: {reason}")]
    InvalidConfig { reason: String },
}

// ---------------------------------------------------------------------------
// Laplacian kinds
// ---------------------------------------------------------------------------

/// Which variant of the graph Laplacian to build or use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LaplacianKind {
    /// L = D − A (combinatorial / unnormalized).
    Combinatorial,
    /// L = D^{−1/2}(D−A)D^{−1/2} (symmetric normalized).
    Normalized,
    /// Cotangent-weighted geometric Laplacian.
    Cotangent,
    /// D^{−1}(D−A) = I − D^{−1}A (random-walk Laplacian).
    RandomWalk,
}

// ---------------------------------------------------------------------------
// MeshLaplacian (CSR)
// ---------------------------------------------------------------------------

/// Sparse graph Laplacian in Compressed Sparse Row format.
pub struct MeshLaplacian {
    /// Number of vertices.
    pub n_vertices: usize,
    /// Row pointer array (length `n_vertices` + 1).
    pub row_ptr: Vec<usize>,
    /// Column indices for off-diagonal non-zeros.
    pub col_idx: Vec<usize>,
    /// Off-diagonal values `L_ij` (negative for combinatorial/cotangent).
    pub values: Vec<f32>,
    /// Diagonal degree values `D_ii`.
    pub degree: Vec<f32>,
    /// Which Laplacian variant this is.
    pub kind: LaplacianKind,
}

// ---------------------------------------------------------------------------
// Spectral basis
// ---------------------------------------------------------------------------

/// A set of k eigenvectors / eigenvalues of the graph Laplacian.
pub struct SpectralBasis {
    /// Eigenvectors (each of length `n_vertices`), sorted by eigenvalue ascending.
    pub eigenvectors: Vec<Vec<f32>>,
    /// Corresponding eigenvalues (sorted ascending).
    pub eigenvalues: Vec<f32>,
    /// Number of eigenvectors computed.
    pub k: usize,
    /// Number of mesh vertices.
    pub n_vertices: usize,
}

// ---------------------------------------------------------------------------
// Signal / config / stats
// ---------------------------------------------------------------------------

/// A per-vertex scalar signal.
pub struct SpectralSignal {
    /// One value per vertex.
    pub values: Vec<f32>,
    /// Vertex count (== `values.len()`).
    pub n_vertices: usize,
}

/// Configuration for spectral analysis.
pub struct SpectralConfig {
    /// Number of spectral basis vectors to compute.
    pub k: usize,
    /// Maximum power iterations.
    pub max_power_iters: usize,
    /// Convergence tolerance.
    pub tol: f32,
    /// Which Laplacian to use.
    pub laplacian_kind: LaplacianKind,
}

/// Summary statistics derived from a Laplacian (and optional basis).
pub struct SpectralStats {
    /// Number of vertices.
    pub n_vertices: usize,
    /// Number of undirected edges.
    pub n_edges: usize,
    /// Average vertex degree.
    pub mean_degree: f32,
    /// Maximum vertex degree.
    pub max_degree: usize,
    /// Fiedler value λ₂ (second-smallest eigenvalue; 0 ⟹ disconnected).
    pub algebraic_connectivity: f32,
    /// λ₂ − λ₁.
    pub spectral_gap: f32,
    /// Largest eigenvalue known (from basis or heuristic).
    pub spectral_radius: f32,
}

// ---------------------------------------------------------------------------
// xorshift64 PRNG
// ---------------------------------------------------------------------------

#[inline]
fn xorshift64(state: &mut u64) -> u64 {
    let mut s = *state;
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    if s == 0 {
        s = 1;
    }
    *state = s;
    s
}

/// Map an xorshift64 sample to f32 in [0, 1).
#[inline]
fn xorshift64_f32(state: &mut u64) -> f32 {
    (xorshift64(state) as f32) / (u64::MAX as f32)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Compute cotangent of the angle at vertex `opp` in triangle (opp, a, b).
fn cot_angle_at(opp: na::Point3<f32>, a: na::Point3<f32>, b: na::Point3<f32>) -> f32 {
    let u = a - opp;
    let v = b - opp;
    let dot = u.dot(&v);
    let cross = u.cross(&v).norm();
    dot / cross.max(1e-8)
}

/// Collect undirected edges from face soup and build adjacency lists (`BTreeMap`).
fn build_adjacency_map(n_vertices: usize, faces: &[[u32; 3]]) -> BTreeMap<usize, Vec<usize>> {
    let mut adj: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for i in 0..n_vertices {
        adj.insert(i, Vec::new());
    }
    for face in faces {
        let [a, b, c] = [face[0] as usize, face[1] as usize, face[2] as usize];
        for &(u, v) in &[(a, b), (b, a), (b, c), (c, b), (a, c), (c, a)] {
            let neighbors = adj.entry(u).or_default();
            if !neighbors.contains(&v) {
                neighbors.push(v);
            }
        }
    }
    // Sort each neighbor list for deterministic CSR row ordering
    for neighbors in adj.values_mut() {
        neighbors.sort_unstable();
    }
    adj
}

/// Convert an adjacency + values map into CSR arrays.
/// `value_fn(row, col)` produces the off-diagonal value.
fn build_csr<F: Fn(usize, usize) -> f32>(
    n: usize,
    adj: &BTreeMap<usize, Vec<usize>>,
    value_fn: F,
) -> (Vec<usize>, Vec<usize>, Vec<f32>) {
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx = Vec::new();
    let mut values = Vec::new();

    let mut nnz = 0usize;
    row_ptr.push(0);
    for i in 0..n {
        if let Some(neighbors) = adj.get(&i) {
            for &j in neighbors {
                col_idx.push(j);
                values.push(value_fn(i, j));
                nnz += 1;
            }
        }
        row_ptr.push(nnz);
    }
    (row_ptr, col_idx, values)
}

/// L2 norm of a vector.
fn norm2(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// In-place normalization; returns false if the vector is (near-)zero.
/// Threshold is 1e-5 to handle f32 cancellation in Gram-Schmidt.
fn normalize_inplace(v: &mut [f32]) -> bool {
    let n = norm2(v);
    if n < 1e-5 {
        return false;
    }
    for x in v.iter_mut() {
        *x /= n;
    }
    true
}

/// Dot product of two equal-length slices.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// ---------------------------------------------------------------------------
// Public: build Laplacians
// ---------------------------------------------------------------------------

/// Build the combinatorial (unnormalized) Laplacian L = D − A from a triangle mesh.
///
/// # Errors
///
/// Returns an error if the operation fails.
pub fn spec_build_combinatorial_laplacian(
    vertices: &[na::Point3<f32>],
    faces: &[[u32; 3]],
) -> Result<MeshLaplacian, SpectralError> {
    let n = vertices.len();
    if n == 0 {
        return Err(SpectralError::EmptyMesh);
    }
    let adj = build_adjacency_map(n, faces);
    // Degree
    let mut degree = vec![0.0f32; n];
    for (&i, neighbors) in &adj {
        degree[i] = neighbors.len() as f32;
    }
    // Check for isolated vertices
    for (i, &d) in degree.iter().enumerate() {
        if d == 0.0 && !faces.is_empty() {
            return Err(SpectralError::IsolatedVertex { idx: i });
        }
    }
    let (row_ptr, col_idx, values) = build_csr(n, &adj, |_i, _j| -1.0_f32);
    Ok(MeshLaplacian {
        n_vertices: n,
        row_ptr,
        col_idx,
        values,
        degree,
        kind: LaplacianKind::Combinatorial,
    })
}

/// Build the cotangent-weighted Laplacian from a triangle mesh.
///
/// # Errors
///
/// Returns an error if the operation fails.
pub fn spec_build_cotangent_laplacian(
    vertices: &[na::Point3<f32>],
    faces: &[[u32; 3]],
) -> Result<MeshLaplacian, SpectralError> {
    let n = vertices.len();
    if n == 0 {
        return Err(SpectralError::EmptyMesh);
    }
    // Accumulate cotangent weights per directed edge using BTreeMap
    let mut weights: BTreeMap<(usize, usize), f32> = BTreeMap::new();
    for face in faces {
        let [ia, ib, ic] = [face[0] as usize, face[1] as usize, face[2] as usize];
        let (pa, pb, pc) = (vertices[ia], vertices[ib], vertices[ic]);
        // Cotangent at each opposite vertex
        let cot_a = cot_angle_at(pa, pb, pc); // opposite to edge (b,c)
        let cot_b = cot_angle_at(pb, pa, pc); // opposite to edge (a,c)
        let cot_c = cot_angle_at(pc, pa, pb); // opposite to edge (a,b)
                                              // Edge (b,c): contribute cot_a/2
        *weights.entry((ib, ic)).or_insert(0.0) += cot_a * 0.5;
        *weights.entry((ic, ib)).or_insert(0.0) += cot_a * 0.5;
        // Edge (a,c): contribute cot_b/2
        *weights.entry((ia, ic)).or_insert(0.0) += cot_b * 0.5;
        *weights.entry((ic, ia)).or_insert(0.0) += cot_b * 0.5;
        // Edge (a,b): contribute cot_c/2
        *weights.entry((ia, ib)).or_insert(0.0) += cot_c * 0.5;
        *weights.entry((ib, ia)).or_insert(0.0) += cot_c * 0.5;
    }
    // Build adjacency map from weights
    let mut adj: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for i in 0..n {
        adj.insert(i, Vec::new());
    }
    for &(i, j) in weights.keys() {
        let nbrs = adj.entry(i).or_default();
        if !nbrs.contains(&j) {
            nbrs.push(j);
        }
    }
    for nbrs in adj.values_mut() {
        nbrs.sort_unstable();
    }
    // Off-diagonal values are negative weights
    let w_clone = weights.clone();
    let (row_ptr, col_idx, values) = build_csr(n, &adj, |i, j| {
        -w_clone.get(&(i, j)).copied().unwrap_or(0.0)
    });
    // Diagonal: L_ii = sum_j w_ij, SIGNED -- matching the signed
    // off-diagonal values L_ij = -w_ij built above. Cotangent weights are
    // legitimately negative for obtuse triangles; summing |w| here (as the
    // old code did) breaks the zero-row-sum property that defines a graph
    // Laplacian (`L * 1 = 0`), since |w_ij| != w_ij whenever any incident
    // triangle is obtuse at this vertex's opposite angle.
    let mut degree = vec![0.0f32; n];
    for (&(i, _j), &w) in &weights {
        degree[i] += w;
    }
    for (i, &d) in degree.iter().enumerate() {
        if d.abs() < 1e-12 && !faces.is_empty() {
            return Err(SpectralError::IsolatedVertex { idx: i });
        }
    }
    Ok(MeshLaplacian {
        n_vertices: n,
        row_ptr,
        col_idx,
        values,
        degree,
        kind: LaplacianKind::Cotangent,
    })
}

/// Convert a Combinatorial or Cotangent Laplacian to its Normalized form
/// `L_norm` = D^{-1/2} L D^{-1/2}.
///
/// # Errors
///
/// Returns an error if the operation fails.
pub fn spec_normalize_laplacian(lap: &MeshLaplacian) -> Result<MeshLaplacian, SpectralError> {
    let n = lap.n_vertices;
    // D^{-1/2}
    let d_inv_sqrt: Vec<f32> = lap
        .degree
        .iter()
        .map(|&d| if d > 1e-12 { 1.0 / d.sqrt() } else { 0.0 })
        .collect();
    // New off-diagonal: L_norm[i,j] = d_inv_sqrt[i] * L[i,j] * d_inv_sqrt[j]
    let mut new_values = lap.values.clone();
    let mut new_degree = vec![1.0f32; n]; // normalized diagonal = 1 where degree > 0
    for i in 0..n {
        let start = lap.row_ptr[i];
        let end = lap.row_ptr[i + 1];
        for (offset, new_val) in new_values[start..end].iter_mut().enumerate() {
            let abs_idx = start + offset;
            let j = lap.col_idx[abs_idx];
            *new_val = d_inv_sqrt[i] * lap.values[abs_idx] * d_inv_sqrt[j];
        }
        if lap.degree[i] < 1e-12 {
            new_degree[i] = 0.0;
        }
    }
    Ok(MeshLaplacian {
        n_vertices: n,
        row_ptr: lap.row_ptr.clone(),
        col_idx: lap.col_idx.clone(),
        values: new_values,
        degree: new_degree,
        kind: LaplacianKind::Normalized,
    })
}

/// Convert a Combinatorial or Cotangent Laplacian to its random-walk form
/// `L_rw = D^{-1}(D - A) = I - D^{-1}A`: `L_rw[i,i] = 1`,
/// `L_rw[i,j] = -w_ij / d_i`. Unlike [`spec_normalize_laplacian`]'s
/// symmetric form, this scales each row independently by its own degree;
/// its spectrum is still confined to `[0, 2]` (same eigenvalues as the
/// normalized form), which keeps [`spec_laplacian_smooth`] stable.
///
/// # Errors
///
/// Returns an error if the operation fails.
pub fn spec_random_walk_laplacian(lap: &MeshLaplacian) -> Result<MeshLaplacian, SpectralError> {
    let n = lap.n_vertices;
    let mut new_values = lap.values.clone();
    let mut new_degree = vec![1.0f32; n]; // normalized diagonal = 1 where degree > 0
    for (i, new_degree_i) in new_degree.iter_mut().enumerate() {
        let start = lap.row_ptr[i];
        let end = lap.row_ptr[i + 1];
        let d = lap.degree[i];
        let inv_d = if d > 1e-12 { 1.0 / d } else { 0.0 };
        for new_val in &mut new_values[start..end] {
            *new_val *= inv_d;
        }
        if d < 1e-12 {
            *new_degree_i = 0.0;
        }
    }
    Ok(MeshLaplacian {
        n_vertices: n,
        row_ptr: lap.row_ptr.clone(),
        col_idx: lap.col_idx.clone(),
        values: new_values,
        degree: new_degree,
        kind: LaplacianKind::RandomWalk,
    })
}

/// Build the graph Laplacian a [`SpectralConfig`] asks for.
///
/// This is the config-driven entry point: without it `SpectralConfig`'s
/// [`laplacian_kind`](SpectralConfig::laplacian_kind) field is inert — every
/// caller has to pick a `spec_build_*` function by hand and the configured
/// kind only ever reaches [`spec_format_config`]'s output string.
///
/// Dispatch follows [`LaplacianKind`]'s own definitions:
///
/// - [`LaplacianKind::Combinatorial`] → [`spec_build_combinatorial_laplacian`]
/// - [`LaplacianKind::Cotangent`] → [`spec_build_cotangent_laplacian`]
/// - [`LaplacianKind::Normalized`] → [`spec_normalize_laplacian`] applied to
///   the combinatorial Laplacian, because that variant is *defined* as
///   `D^{-1/2}(D − A)D^{-1/2}` over the unweighted `D − A`. Callers wanting
///   the cotangent-weighted geometric Laplacian in symmetric-normalized form
///   should compose the two functions directly rather than expect this arm to
///   silently switch weighting schemes.
/// - [`LaplacianKind::RandomWalk`] → [`spec_random_walk_laplacian`] applied to
///   the combinatorial Laplacian, for the same reason (`D^{-1}(D − A)`).
///
/// The returned [`MeshLaplacian::kind`] always equals `config.laplacian_kind`.
///
/// # Errors
///
/// Propagates whichever builder the configured kind selects: notably
/// [`SpectralError::EmptyMesh`] for an empty `vertices` slice and
/// [`SpectralError::IsolatedVertex`] for a vertex with no incident edge.
pub fn spec_build_laplacian(
    vertices: &[na::Point3<f32>],
    faces: &[[u32; 3]],
    config: &SpectralConfig,
) -> Result<MeshLaplacian, SpectralError> {
    match config.laplacian_kind {
        LaplacianKind::Combinatorial => spec_build_combinatorial_laplacian(vertices, faces),
        LaplacianKind::Cotangent => spec_build_cotangent_laplacian(vertices, faces),
        LaplacianKind::Normalized => {
            let base = spec_build_combinatorial_laplacian(vertices, faces)?;
            spec_normalize_laplacian(&base)
        }
        LaplacianKind::RandomWalk => {
            let base = spec_build_combinatorial_laplacian(vertices, faces)?;
            spec_random_walk_laplacian(&base)
        }
    }
}

/// Compute L * x (sparse matrix-vector product).
///
/// `MeshLaplacian`'s fields are public and every other function here
/// funnels through this primitive, so out-of-range terms (`x` shorter
/// than `lap.n_vertices`, or a corrupted `col_idx`) are treated as zero
/// instead of indexing out of bounds.
#[must_use]
pub fn spec_laplacian_matvec(lap: &MeshLaplacian, x: &[f32]) -> Vec<f32> {
    let n = lap.n_vertices;
    let mut result = vec![0.0f32; n];
    for (i, result_i) in result.iter_mut().enumerate() {
        let Some(&xi) = x.get(i) else { continue };
        // Diagonal contribution
        *result_i += lap.degree[i] * xi;
        // Off-diagonal contributions
        let start = lap.row_ptr[i];
        let end = lap.row_ptr[i + 1];
        for idx in start..end {
            let j = lap.col_idx[idx];
            if let Some(&xj) = x.get(j) {
                *result_i += lap.values[idx] * xj;
            }
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Power iteration helpers
// ---------------------------------------------------------------------------

/// Gram-Schmidt orthogonalization of `vectors` in place (k vectors each of length n).
pub fn spec_gram_schmidt(vectors: &mut [Vec<f32>], n: usize, k: usize) {
    let vlen = n.max(vectors.first().map_or(0, std::vec::Vec::len));
    for i in 0..k.min(vectors.len()) {
        // Subtract projections of all previous vectors.
        // Split so we can mutate vectors[i] while reading vectors[0..i].
        let (prev, rest) = vectors.split_at_mut(i);
        let vi = &mut rest[0];
        for vj in prev.iter() {
            let proj = dot(vj, vi);
            for (elem, &pj) in vi.iter_mut().zip(vj.iter()) {
                *elem -= proj * pj;
            }
        }
        // Normalize
        if !normalize_inplace(&mut vectors[i]) {
            // Find a canonical basis direction orthogonal to all previous vectors
            let mut found = false;
            for basis_dim in 0..vlen {
                let mut candidate = vec![0.0f32; vlen];
                candidate[basis_dim] = 1.0;
                // Subtract projections
                let (prev_vecs, _) = vectors.split_at(i);
                for vj in prev_vecs {
                    let p = dot(vj, &candidate);
                    let vj_clone = vj.clone();
                    for (cl, vx) in candidate.iter_mut().zip(vj_clone.iter()) {
                        *cl -= p * vx;
                    }
                }
                if normalize_inplace(&mut candidate) {
                    vectors[i] = candidate;
                    found = true;
                    break;
                }
            }
            if !found {
                // All basis vectors exhausted — zero out
                for x in &mut vectors[i] {
                    *x = 0.0;
                }
            }
        }
    }
}

/// Rayleigh quotient v^T L v / v^T v.
#[must_use]
pub fn spec_rayleigh_quotient(lap: &MeshLaplacian, v: &[f32]) -> f32 {
    let lv = spec_laplacian_matvec(lap, v);
    let numerator = dot(v, &lv);
    let denominator = dot(v, v);
    if denominator < 1e-14 {
        0.0
    } else {
        numerator / denominator
    }
}

/// Gershgorin upper bound on the spectral radius of `lap`:
/// `max_i(|L_ii| + sum_j |L_ij|)`.
///
/// Computed directly from the CSR rows (diagonal `degree[i]` plus the
/// absolute row sum of off-diagonal `values`) rather than as
/// `2 * max(degree)`, because `2 * max(degree)` is only a valid bound when
/// `degree[i]` equals the absolute off-diagonal row sum -- true for
/// Combinatorial (every off-diagonal entry is exactly -1) but NOT for
/// Cotangent, whose diagonal is the *signed* sum of (possibly negative, for
/// obtuse triangles) cotangent weights while the off-diagonal magnitudes can
/// still be large. On such a mesh `degree[i]` can be smaller than the true
/// absolute row sum, so `2 * max(degree)` can underestimate this bound. The
/// generic per-row form is valid for every [`LaplacianKind`] with no
/// special-casing: for Combinatorial it reduces to exactly `2 * max(degree)`;
/// for Normalized/RandomWalk (diagonal 1.0) it evaluates to ~2.0, matching
/// their known spectral bound of `[0, 2]`; for Cotangent it correctly grows
/// past `2 * max(degree)` when obtuse triangles are present.
///
/// Used as the shift in [`spec_power_iteration`]'s `(lambda_max * I - L)`
/// operator: an underestimate there would make power iteration converge to
/// a mix of the lowest- and highest-frequency modes instead of the k
/// smallest, since a shifted eigenvalue near 0 (from an unaccounted-for high
/// true eigenvalue) can rival the shifted value of the true smallest mode.
fn gershgorin_lambda_max(lap: &MeshLaplacian) -> f32 {
    (0..lap.n_vertices)
        .map(|i| {
            let row_abs: f32 = lap.values[lap.row_ptr[i]..lap.row_ptr[i + 1]]
                .iter()
                .map(|v| v.abs())
                .sum();
            lap.degree[i].abs() + row_abs
        })
        .fold(0.0f32, f32::max)
        .max(1.0)
}

/// Compute k smallest eigenvectors using shifted block power iteration.
///
/// # Errors
///
/// Returns an error if the operation fails.
pub fn spec_power_iteration(
    lap: &MeshLaplacian,
    k: usize,
    max_iters: usize,
    tol: f32,
    seed: u64,
) -> Result<SpectralBasis, SpectralError> {
    let n = lap.n_vertices;
    if k == 0 {
        return Err(SpectralError::InvalidConfig {
            reason: "k must be >= 1".to_string(),
        });
    }
    if k > n {
        return Err(SpectralError::InsufficientBasis { k, available: n });
    }
    // Shift bound for the `(lambda_max * I - L)` operator below -- see
    // `gershgorin_lambda_max` for why this must be a generic per-row bound
    // rather than `2 * max(degree)`.
    let lambda_max = gershgorin_lambda_max(lap);

    // Initialize k random vectors
    let mut rng_state = seed.max(1);
    let mut q: Vec<Vec<f32>> = (0..k)
        .map(|_| {
            (0..n)
                .map(|_| xorshift64_f32(&mut rng_state) * 2.0 - 1.0)
                .collect()
        })
        .collect();

    // Initial Gram-Schmidt
    spec_gram_schmidt(&mut q, n, k);

    let mut prev_rq = vec![f64::MAX; k];

    for iter in 0..max_iters {
        // Apply shifted Laplacian: L_shifted = (lambda_max * I - L)
        // q_new[i] = lambda_max * q[i] - L * q[i]
        let mut q_new: Vec<Vec<f32>> = q
            .iter()
            .map(|qi| {
                let lqi = spec_laplacian_matvec(lap, qi);
                qi.iter()
                    .zip(lqi.iter())
                    .map(|(&qx, &lx)| lambda_max * qx - lx)
                    .collect()
            })
            .collect();

        // Gram-Schmidt re-orthogonalize
        spec_gram_schmidt(&mut q_new, n, k);
        q = q_new;

        // Check convergence via Rayleigh quotients
        let rq: Vec<f64> = q
            .iter()
            .map(|qi| f64::from(spec_rayleigh_quotient(lap, qi)))
            .collect();

        let max_change = rq
            .iter()
            .zip(prev_rq.iter())
            .map(|(r, pr)| (r - pr).abs())
            .fold(0.0f64, f64::max);

        prev_rq = rq;

        if iter > 0 && max_change < f64::from(tol) {
            break;
        }

        if iter == max_iters - 1 && max_change > f64::from(tol * 100.0) {
            return Err(SpectralError::PowerIterationDiverged { iters: max_iters });
        }
    }

    // Compute Rayleigh quotients as eigenvalue estimates
    let mut pairs: Vec<(f32, Vec<f32>)> = q
        .into_iter()
        .map(|qi| {
            let rq = spec_rayleigh_quotient(lap, &qi);
            (rq, qi)
        })
        .collect();

    // Sort by Rayleigh quotient ascending (smallest eigenvalues first)
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let eigenvalues: Vec<f32> = pairs.iter().map(|(rq, _)| *rq).collect();
    let eigenvectors: Vec<Vec<f32>> = pairs.into_iter().map(|(_, v)| v).collect();

    Ok(SpectralBasis {
        eigenvectors,
        eigenvalues,
        k,
        n_vertices: n,
    })
}

// ---------------------------------------------------------------------------
// Spectral filtering
// ---------------------------------------------------------------------------

/// Project a signal onto the spectral basis: `c_i` = <signal, `v_i`>.
///
/// # Errors
///
/// Returns an error if the operation fails.
pub fn spec_project(
    signal: &SpectralSignal,
    basis: &SpectralBasis,
) -> Result<Vec<f32>, SpectralError> {
    if signal.n_vertices != basis.n_vertices {
        return Err(SpectralError::DimensionMismatch {
            sig: signal.n_vertices,
            mesh: basis.n_vertices,
        });
    }
    let coeffs = basis
        .eigenvectors
        .iter()
        .map(|v| dot(&signal.values, v))
        .collect();
    Ok(coeffs)
}

/// Reconstruct a signal from spectral coefficients: sum `c_i` * `v_i`.
///
/// # Errors
///
/// Returns an error if the operation fails.
pub fn spec_reconstruct(
    coefficients: &[f32],
    basis: &SpectralBasis,
) -> Result<SpectralSignal, SpectralError> {
    if coefficients.len() > basis.k {
        return Err(SpectralError::InsufficientBasis {
            k: coefficients.len(),
            available: basis.k,
        });
    }
    let n = basis.n_vertices;
    let mut values = vec![0.0f32; n];
    for (ci, vi) in coefficients.iter().zip(basis.eigenvectors.iter()) {
        for (val, &vx) in values.iter_mut().zip(vi.iter()) {
            *val += ci * vx;
        }
    }
    Ok(SpectralSignal {
        values,
        n_vertices: n,
    })
}

/// Low-pass filter: keep only the first `cutoff_k` spectral components.
///
/// # Errors
///
/// Returns an error if the operation fails.
pub fn spec_low_pass_filter(
    signal: &SpectralSignal,
    basis: &SpectralBasis,
    cutoff_k: usize,
) -> Result<SpectralSignal, SpectralError> {
    if signal.n_vertices != basis.n_vertices {
        return Err(SpectralError::DimensionMismatch {
            sig: signal.n_vertices,
            mesh: basis.n_vertices,
        });
    }
    let mut coeffs = spec_project(signal, basis)?;
    // Zero out high-frequency components (index >= cutoff_k)
    for c in coeffs.iter_mut().skip(cutoff_k) {
        *c = 0.0;
    }
    spec_reconstruct(&coeffs, basis)
}

/// High-pass filter: remove the first `cutoff_k` spectral components.
///
/// # Errors
///
/// Returns an error if the operation fails.
pub fn spec_high_pass_filter(
    signal: &SpectralSignal,
    basis: &SpectralBasis,
    cutoff_k: usize,
) -> Result<SpectralSignal, SpectralError> {
    if signal.n_vertices != basis.n_vertices {
        return Err(SpectralError::DimensionMismatch {
            sig: signal.n_vertices,
            mesh: basis.n_vertices,
        });
    }
    let mut coeffs = spec_project(signal, basis)?;
    // Zero out low-frequency components (index < cutoff_k)
    for c in coeffs.iter_mut().take(cutoff_k) {
        *c = 0.0;
    }
    spec_reconstruct(&coeffs, basis)
}

/// Dirichlet energy: f^T L f (measures signal variation on the graph).
///
/// # Errors
///
/// Returns [`SpectralError::DimensionMismatch`] if `signal.n_vertices !=
/// lap.n_vertices` (otherwise this would silently compute a wrong, partial
/// energy via `spec_laplacian_matvec`'s length clamping).
pub fn spec_smoothness(signal: &SpectralSignal, lap: &MeshLaplacian) -> Result<f32, SpectralError> {
    if signal.n_vertices != lap.n_vertices {
        return Err(SpectralError::DimensionMismatch {
            sig: signal.n_vertices,
            mesh: lap.n_vertices,
        });
    }
    let lf = spec_laplacian_matvec(lap, &signal.values);
    Ok(dot(&signal.values, &lf))
}

// ---------------------------------------------------------------------------
// Direct Laplacian smoothing (no eigenvectors)
// ---------------------------------------------------------------------------

/// Explicit Laplacian smoothing: `x_new` = x − λ L x per iteration (the
/// standard umbrella-operator form, equivalent to
/// `x_new = x + λ * (mean_of_neighbours − x)`).
///
/// `positions` is a flat slice of length 3 * `n_vertices` (x0,y0,z0, x1,y1,z1, ...).
///
/// # Stability
///
/// With the unnormalized `L = D − A` (`Combinatorial`/`Cotangent`), stable
/// only for `lambda < 2 / lambda_max(L)` (`<= 2 * max(degree)` by
/// Gershgorin). [`spec_normalize_laplacian`]/[`spec_random_walk_laplacian`]
/// avoid this bound (spectrum confined to `[0, 2]`).
#[must_use]
pub fn spec_laplacian_smooth(
    positions: &[f32],
    lap: &MeshLaplacian,
    lambda: f32,
    n_iters: usize,
) -> Vec<f32> {
    let n = lap.n_vertices;
    let mut pos = positions.to_vec();
    let mut coord = vec![0.0f32; n];
    for _ in 0..n_iters {
        for dim in 0..3 {
            // Extract coordinate channel
            for i in 0..n {
                coord[i] = pos[3 * i + dim];
            }
            let lx = spec_laplacian_matvec(lap, &coord);
            for i in 0..n {
                pos[3 * i + dim] -= lambda * lx[i];
            }
        }
    }
    pos
}

/// Taubin smoothing: alternating λ (shrink) and μ (inflate) pairs.
/// Each iteration applies one λ step followed by one μ step, avoiding volume loss.
///
/// Uses the same `x_new = x − step * L x` convention as
/// [`spec_laplacian_smooth`] for both steps, so the standard
/// parameterization (`lambda > 0` shrink, `mu < 0` inflate) behaves as
/// documented; see that function's `# Stability` note.
#[must_use]
pub fn spec_taubin_smooth(
    positions: &[f32],
    lap: &MeshLaplacian,
    lambda: f32,
    mu: f32,
    n_iters: usize,
) -> Vec<f32> {
    let n = lap.n_vertices;
    let mut pos = positions.to_vec();
    let mut coord = vec![0.0f32; n];
    for _ in 0..n_iters {
        // λ step (shrink)
        for dim in 0..3 {
            for i in 0..n {
                coord[i] = pos[3 * i + dim];
            }
            let lx = spec_laplacian_matvec(lap, &coord);
            for i in 0..n {
                pos[3 * i + dim] -= lambda * lx[i];
            }
        }
        // μ step (inflate)
        for dim in 0..3 {
            for i in 0..n {
                coord[i] = pos[3 * i + dim];
            }
            let lx = spec_laplacian_matvec(lap, &coord);
            for i in 0..n {
                pos[3 * i + dim] -= mu * lx[i];
            }
        }
    }
    pos
}

// ---------------------------------------------------------------------------
// Spectral clustering
// ---------------------------------------------------------------------------

/// K-means clustering on eigenvector embeddings (first `n_clusters` eigenvectors as features).
///
/// # Errors
///
/// Returns an error if the operation fails.
pub fn spec_cluster_vertices(
    basis: &SpectralBasis,
    n_clusters: usize,
    seed: u64,
) -> Result<Vec<usize>, SpectralError> {
    if n_clusters == 0 {
        return Err(SpectralError::InvalidConfig {
            reason: "n_clusters must be >= 1".to_string(),
        });
    }
    let n = basis.n_vertices;
    if n == 0 {
        // `SpectralBasis`'s fields are all public, so a hand-constructed
        // zero-vertex basis is reachable here; without this guard, the
        // `xorshift64(&mut rng) as usize % n` k-means++ seed selection
        // below panics on division/modulo by zero.
        return Err(SpectralError::EmptyMesh);
    }
    let k_feat = n_clusters.min(basis.k); // features = first n_clusters eigenvectors
    if k_feat == 0 {
        return Err(SpectralError::InsufficientBasis {
            k: n_clusters,
            available: basis.k,
        });
    }

    // Build feature matrix: rows = vertices, cols = k_feat eigenvectors
    // features[i * k_feat + f] = basis.eigenvectors[f][i]
    let mut features = vec![0.0f32; n * k_feat];
    for (f, evec) in basis.eigenvectors.iter().take(k_feat).enumerate() {
        for (i, &val) in evec.iter().enumerate() {
            features[i * k_feat + f] = val;
        }
    }

    // K-means++ initialization
    let mut rng = seed.max(1);
    let mut centers: Vec<Vec<f32>> = Vec::with_capacity(n_clusters);

    // First center: random vertex
    let first = (xorshift64(&mut rng) as usize) % n;
    centers.push(features[first * k_feat..(first + 1) * k_feat].to_vec());

    // Remaining centers: k-means++ proportional to distance^2
    for _ in 1..n_clusters {
        let mut dist_sq = vec![f32::MAX; n];
        for (i, d) in dist_sq.iter_mut().enumerate() {
            for c in &centers {
                let d2: f32 = (0..k_feat)
                    .map(|f| {
                        let diff = features[i * k_feat + f] - c[f];
                        diff * diff
                    })
                    .sum();
                *d = d.min(d2);
            }
        }
        let total: f32 = dist_sq.iter().sum();
        let r = xorshift64_f32(&mut rng) * total;
        let mut cum = 0.0f32;
        let mut chosen = 0;
        for (i, &d) in dist_sq.iter().enumerate() {
            cum += d;
            if cum >= r {
                chosen = i;
                break;
            }
        }
        centers.push(features[chosen * k_feat..(chosen + 1) * k_feat].to_vec());
    }

    // Lloyd's algorithm
    let mut assignments = vec![0usize; n];
    for _iter in 0..100 {
        // Assignment step
        let mut changed = false;
        for i in 0..n {
            let feat_i = &features[i * k_feat..(i + 1) * k_feat];
            let mut best_c = 0;
            let mut best_d = f32::MAX;
            for (ci, center) in centers.iter().enumerate() {
                let d: f32 = feat_i
                    .iter()
                    .zip(center.iter())
                    .map(|(&a, &b)| (a - b) * (a - b))
                    .sum();
                if d < best_d {
                    best_d = d;
                    best_c = ci;
                }
            }
            if assignments[i] != best_c {
                assignments[i] = best_c;
                changed = true;
            }
        }
        if !changed {
            break;
        }
        // Update step
        let mut new_centers = vec![vec![0.0f32; k_feat]; n_clusters];
        let mut counts = vec![0usize; n_clusters];
        for i in 0..n {
            let c = assignments[i];
            counts[c] += 1;
            for f in 0..k_feat {
                new_centers[c][f] += features[i * k_feat + f];
            }
        }
        for (ci, center) in new_centers.iter_mut().enumerate() {
            if counts[ci] > 0 {
                let inv = 1.0 / counts[ci] as f32;
                for x in center.iter_mut() {
                    *x *= inv;
                }
            }
        }
        centers = new_centers;
    }

    Ok(assignments)
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Compute summary statistics for a Laplacian (and optionally a `SpectralBasis`).
pub fn spec_compute_stats(lap: &MeshLaplacian, basis: Option<&SpectralBasis>) -> SpectralStats {
    let n = lap.n_vertices;
    // Count undirected edges = nnz in off-diagonal CSR / 2
    let n_edges = lap.col_idx.len() / 2;
    let degree_int: Vec<usize> = lap.degree.iter().map(|&d| d as usize).collect();
    let max_degree = degree_int.iter().copied().max().unwrap_or(0);
    let mean_degree = if n > 0 {
        lap.degree.iter().sum::<f32>() / n as f32
    } else {
        0.0
    };

    let (algebraic_connectivity, spectral_gap, spectral_radius) = if let Some(b) = basis {
        let lambda1 = b.eigenvalues.first().copied().unwrap_or(0.0);
        let lambda2 = if b.eigenvalues.len() > 1 {
            b.eigenvalues[1]
        } else {
            0.0
        };
        let lambda_max = b.eigenvalues.iter().copied().fold(0.0f32, f32::max);
        (lambda2, lambda2 - lambda1, lambda_max)
    } else {
        // Heuristic: spectral radius ≈ max degree for combinatorial Laplacian
        let approx_max = lap.degree.iter().copied().fold(0.0f32, f32::max);
        (0.0, 0.0, approx_max)
    };

    SpectralStats {
        n_vertices: n,
        n_edges,
        mean_degree,
        max_degree,
        algebraic_connectivity,
        spectral_gap,
        spectral_radius,
    }
}

/// Format a `SpectralStats` as a human-readable string.
#[must_use]
pub fn spec_format_stats(stats: &SpectralStats) -> String {
    format!(
        "SpectralStats {{ n_vertices: {}, n_edges: {}, mean_degree: {:.2}, \
         max_degree: {}, algebraic_connectivity: {:.4}, spectral_gap: {:.4}, \
         spectral_radius: {:.4} }}",
        stats.n_vertices,
        stats.n_edges,
        stats.mean_degree,
        stats.max_degree,
        stats.algebraic_connectivity,
        stats.spectral_gap,
        stats.spectral_radius,
    )
}

/// Format a `SpectralConfig` as a human-readable string.
#[must_use]
pub fn spec_format_config(config: &SpectralConfig) -> String {
    let kind = match config.laplacian_kind {
        LaplacianKind::Combinatorial => "Combinatorial",
        LaplacianKind::Normalized => "Normalized",
        LaplacianKind::Cotangent => "Cotangent",
        LaplacianKind::RandomWalk => "RandomWalk",
    };
    format!(
        "SpectralConfig {{ k: {}, max_power_iters: {}, tol: {:.2e}, laplacian_kind: {} }}",
        config.k, config.max_power_iters, config.tol, kind,
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "spectral_analysis/tests.rs"]
mod tests;

#[cfg(test)]
mod config_dispatch_tests {
    use super::{
        spec_build_combinatorial_laplacian, spec_build_cotangent_laplacian, spec_build_laplacian,
        spec_normalize_laplacian, spec_random_walk_laplacian, LaplacianKind, SpectralConfig,
        SpectralError,
    };
    use nalgebra as na;

    /// Every [`LaplacianKind`] variant, so a newly added variant makes these
    /// tests fail to compile rather than silently go untested.
    const ALL_KINDS: [LaplacianKind; 4] = [
        LaplacianKind::Combinatorial,
        LaplacianKind::Normalized,
        LaplacianKind::Cotangent,
        LaplacianKind::RandomWalk,
    ];

    /// A regular tetrahedron: closed, no isolated vertices, and every face is
    /// equilateral, so all cotangent weights are strictly positive and the
    /// cotangent arm cannot trip `IsolatedVertex` on a near-zero diagonal.
    fn tetrahedron() -> (Vec<na::Point3<f32>>, Vec<[u32; 3]>) {
        let vertices = vec![
            na::Point3::new(1.0, 1.0, 1.0),
            na::Point3::new(1.0, -1.0, -1.0),
            na::Point3::new(-1.0, 1.0, -1.0),
            na::Point3::new(-1.0, -1.0, 1.0),
        ];
        let faces = vec![[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]];
        (vertices, faces)
    }

    fn config_with(kind: LaplacianKind) -> SpectralConfig {
        SpectralConfig {
            k: 2,
            max_power_iters: 64,
            tol: 1e-6,
            laplacian_kind: kind,
        }
    }

    #[test]
    fn spec_build_laplacian_returns_the_configured_kind() {
        // Regression test for `SpectralConfig::laplacian_kind` being a
        // write-only field: before `spec_build_laplacian` existed the only
        // reader was `spec_format_config`, so a caller configuring
        // `Cotangent` still had to hand-pick a builder and could silently get
        // a combinatorial Laplacian. Rewiring any arm must fail here.
        let (vertices, faces) = tetrahedron();
        for kind in ALL_KINDS {
            let lap = spec_build_laplacian(&vertices, &faces, &config_with(kind))
                .expect("test: a regular tetrahedron is valid for every Laplacian kind");
            assert_eq!(
                lap.kind, kind,
                "spec_build_laplacian must return the configured kind {kind:?}"
            );
            assert_eq!(lap.n_vertices, vertices.len());
        }
    }

    #[test]
    fn spec_build_laplacian_matches_hand_composed_builders() {
        let (vertices, faces) = tetrahedron();
        let combinatorial = spec_build_combinatorial_laplacian(&vertices, &faces)
            .expect("test: combinatorial build should succeed");

        let expected: Vec<(LaplacianKind, Vec<f32>, Vec<f32>)> = vec![
            (
                LaplacianKind::Combinatorial,
                combinatorial.values.clone(),
                combinatorial.degree.clone(),
            ),
            {
                let normalized = spec_normalize_laplacian(&combinatorial)
                    .expect("test: normalize should succeed");
                (
                    LaplacianKind::Normalized,
                    normalized.values,
                    normalized.degree,
                )
            },
            {
                let cotangent = spec_build_cotangent_laplacian(&vertices, &faces)
                    .expect("test: cotangent build should succeed");
                (LaplacianKind::Cotangent, cotangent.values, cotangent.degree)
            },
            {
                let random_walk = spec_random_walk_laplacian(&combinatorial)
                    .expect("test: random-walk conversion should succeed");
                (
                    LaplacianKind::RandomWalk,
                    random_walk.values,
                    random_walk.degree,
                )
            },
        ];

        for (kind, values, degree) in expected {
            let lap = spec_build_laplacian(&vertices, &faces, &config_with(kind))
                .expect("test: dispatch should succeed");
            assert_eq!(lap.values, values, "{kind:?}: off-diagonal values differ");
            assert_eq!(lap.degree, degree, "{kind:?}: diagonal differs");
        }
    }

    #[test]
    fn spec_build_laplacian_propagates_empty_mesh_for_every_kind() {
        for kind in ALL_KINDS {
            // Matched rather than unwrapped: `MeshLaplacian` deliberately does
            // not implement `Debug` (it is a bulky CSR payload), so
            // `expect_err` is unavailable on this `Result`.
            let result = spec_build_laplacian(&[], &[], &config_with(kind));
            assert!(
                matches!(result, Err(SpectralError::EmptyMesh)),
                "{kind:?}: an empty vertex slice must fail with EmptyMesh"
            );
        }
    }
}
