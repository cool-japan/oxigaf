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
    // Diagonal: sum of positive weights for each row
    let mut degree = vec![0.0f32; n];
    for (&(i, _j), &w) in &weights {
        if w > 0.0 {
            degree[i] += w;
        } else {
            degree[i] -= w; // absolute value
        }
    }
    // degree[i] = sum of positive w_{ij}
    // Re-compute: degree[i] = sum_j w_{ij}
    // For cotangent L: L_ii = sum_j w_ij (positive), L_ij = -w_ij
    let mut deg2 = vec![0.0f32; n];
    for (&(i, _j), &w) in &weights {
        deg2[i] += w.abs();
    }
    // Each undirected edge was counted twice (i→j and j→i), so divide by 2? No: we already
    // stored each directed half-edge, so deg2[i] = sum_j w_{ij} which is correct for L_ii.
    for (i, &d) in deg2.iter().enumerate() {
        if d == 0.0 && !faces.is_empty() {
            return Err(SpectralError::IsolatedVertex { idx: i });
        }
    }
    Ok(MeshLaplacian {
        n_vertices: n,
        row_ptr,
        col_idx,
        values,
        degree: deg2,
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

/// Compute L * x (sparse matrix-vector product).
#[must_use]
pub fn spec_laplacian_matvec(lap: &MeshLaplacian, x: &[f32]) -> Vec<f32> {
    let n = lap.n_vertices;
    let mut result = vec![0.0f32; n];
    for i in 0..n {
        // Diagonal contribution
        result[i] += lap.degree[i] * x[i];
        // Off-diagonal contributions
        let start = lap.row_ptr[i];
        let end = lap.row_ptr[i + 1];
        for idx in start..end {
            let j = lap.col_idx[idx];
            result[i] += lap.values[idx] * x[j];
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
    // Approximate lambda_max = max(degree) for shift
    let lambda_max = lap.degree.iter().copied().fold(0.0f32, f32::max).max(1.0);

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
#[must_use]
pub fn spec_smoothness(signal: &SpectralSignal, lap: &MeshLaplacian) -> f32 {
    let lf = spec_laplacian_matvec(lap, &signal.values);
    dot(&signal.values, &lf)
}

// ---------------------------------------------------------------------------
// Direct Laplacian smoothing (no eigenvectors)
// ---------------------------------------------------------------------------

/// Explicit Laplacian smoothing: `x_new` = x + λ L x per iteration.
/// `positions` is a flat slice of length 3 * `n_vertices` (x0,y0,z0, x1,y1,z1, ...).
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
                pos[3 * i + dim] += lambda * lx[i];
            }
        }
    }
    pos
}

/// Taubin smoothing: alternating λ (shrink) and μ (inflate) pairs.
/// Each iteration applies one λ step followed by one μ step, avoiding volume loss.
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
                pos[3 * i + dim] += lambda * lx[i];
            }
        }
        // μ step (inflate)
        for dim in 0..3 {
            for i in 0..n {
                coord[i] = pos[3 * i + dim];
            }
            let lx = spec_laplacian_matvec(lap, &coord);
            for i in 0..n {
                pos[3 * i + dim] += mu * lx[i];
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
mod tests {
    use super::*;

    // Helper: single equilateral triangle vertices
    fn triangle_verts() -> Vec<na::Point3<f32>> {
        vec![
            na::Point3::new(0.0, 0.0, 0.0),
            na::Point3::new(1.0, 0.0, 0.0),
            na::Point3::new(0.5, 3.0_f32.sqrt() / 2.0, 0.0),
        ]
    }
    fn triangle_faces() -> Vec<[u32; 3]> {
        vec![[0, 1, 2]]
    }

    // Helper: path graph 0-1-2 (2 edges)
    fn path_verts() -> Vec<na::Point3<f32>> {
        vec![
            na::Point3::new(0.0, 0.0, 0.0),
            na::Point3::new(1.0, 0.0, 0.0),
            na::Point3::new(2.0, 0.0, 0.0),
        ]
    }
    fn path_faces() -> Vec<[u32; 3]> {
        // Degenerate triangle to get edges 0-1 and 1-2
        // Use two faces sharing edge 0-1-2
        vec![[0, 1, 2]]
    }

    // Helper: tetrahedron (4 vertices, each connected to the other 3)
    fn tet_verts() -> Vec<na::Point3<f32>> {
        vec![
            na::Point3::new(0.0, 0.0, 0.0),
            na::Point3::new(1.0, 0.0, 0.0),
            na::Point3::new(0.5, 1.0, 0.0),
            na::Point3::new(0.5, 0.5, 1.0),
        ]
    }
    fn tet_faces() -> Vec<[u32; 3]> {
        vec![[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
    }

    // Test 1: single triangle — degree = 2 for all vertices
    #[test]
    fn test_combinatorial_triangle_degree() {
        let verts = triangle_verts();
        let faces = triangle_faces();
        let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
        assert_eq!(lap.n_vertices, 3);
        for &d in &lap.degree {
            assert!((d - 2.0).abs() < 1e-6, "expected degree 2, got {d}");
        }
    }

    // Test 2: row_ptr length = n + 1
    #[test]
    fn test_combinatorial_row_ptr_len() {
        let verts = triangle_verts();
        let faces = triangle_faces();
        let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
        assert_eq!(lap.row_ptr.len(), 4); // n+1 = 4
    }

    // Test 3: row sums of combinatorial Laplacian = 0
    #[test]
    fn test_combinatorial_row_sums_zero() {
        let verts = triangle_verts();
        let faces = triangle_faces();
        let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
        for i in 0..lap.n_vertices {
            let mut row_sum = lap.degree[i]; // diagonal
            let start = lap.row_ptr[i];
            let end = lap.row_ptr[i + 1];
            for idx in start..end {
                row_sum += lap.values[idx];
            }
            assert!(row_sum.abs() < 1e-5, "row {i} sum = {row_sum}");
        }
    }

    // Test 4: L * 1 = 0 (constant vector in null space)
    #[test]
    fn test_matvec_constant_zero() {
        let verts = triangle_verts();
        let faces = triangle_faces();
        let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
        let ones = vec![1.0f32; lap.n_vertices];
        let result = spec_laplacian_matvec(&lap, &ones);
        for &r in &result {
            assert!(r.abs() < 1e-5, "L*1 != 0: {r}");
        }
    }

    // Test 5: matvec dimension correctness
    #[test]
    fn test_matvec_dimension() {
        let verts = tet_verts();
        let faces = tet_faces();
        let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let result = spec_laplacian_matvec(&lap, &x);
        assert_eq!(result.len(), 4);
    }

    // Test 6: cotangent Laplacian on equilateral triangle — symmetric weights
    #[test]
    fn test_cotangent_symmetric() {
        let verts = triangle_verts();
        let faces = triangle_faces();
        let lap = spec_build_cotangent_laplacian(&verts, &faces).expect("cotangent lap");
        // For equilateral triangle all angles are 60°, cot(60°) = 1/sqrt(3)
        // All off-diagonal weights should be equal (symmetric)
        let n = lap.n_vertices;
        let mut w_vals = Vec::new();
        for i in 0..n {
            let start = lap.row_ptr[i];
            let end = lap.row_ptr[i + 1];
            for idx in start..end {
                w_vals.push(lap.values[idx].abs());
            }
        }
        let first = w_vals[0];
        for &w in &w_vals {
            assert!(
                (w - first).abs() < 1e-4,
                "weights not equal: {w} vs {first}"
            );
        }
    }

    // Test 7: cotangent row sums ≈ 0
    #[test]
    fn test_cotangent_row_sums_zero() {
        let verts = triangle_verts();
        let faces = triangle_faces();
        let lap = spec_build_cotangent_laplacian(&verts, &faces).expect("cotangent lap");
        for i in 0..lap.n_vertices {
            let mut row_sum = lap.degree[i];
            let start = lap.row_ptr[i];
            let end = lap.row_ptr[i + 1];
            for idx in start..end {
                row_sum += lap.values[idx];
            }
            assert!(row_sum.abs() < 1e-3, "cotangent row {i} sum = {row_sum}");
        }
    }

    // Test 8: normalized Laplacian — diagonal values are 1 (or 0 for isolated)
    #[test]
    fn test_normalize_laplacian_diagonal() {
        let verts = triangle_verts();
        let faces = triangle_faces();
        let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
        let norm_lap = spec_normalize_laplacian(&lap).expect("normalize");
        for &d in &norm_lap.degree {
            // Normalized diagonal should be 1 for connected vertices
            assert!(
                (d - 1.0).abs() < 1e-6,
                "normalized degree should be 1, got {d}"
            );
        }
    }

    // Test 9: normalized Laplacian — off-diagonal |values| <= 1
    #[test]
    fn test_normalize_laplacian_values_bounded() {
        let verts = tet_verts();
        let faces = tet_faces();
        let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
        let norm_lap = spec_normalize_laplacian(&lap).expect("normalize");
        for &v in &norm_lap.values {
            assert!(v.abs() <= 1.0 + 1e-5, "normalized value out of range: {v}");
        }
    }

    // Test 10: Gram-Schmidt — output vectors are orthonormal
    #[test]
    fn test_gram_schmidt_orthonormal() {
        let n = 4;
        let mut vecs = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![5.0, 6.0, 7.0, 8.0],
            vec![9.0, 10.0, 11.0, 12.0],
        ];
        let k = vecs.len();
        spec_gram_schmidt(&mut vecs, n, k);
        for (i, vi) in vecs.iter().enumerate() {
            // Self dot = 1
            let self_d = dot(vi, vi);
            assert!((self_d - 1.0).abs() < 1e-5, "v{i} not normalized: {self_d}");
            for (j, vj) in vecs.iter().enumerate() {
                if i != j {
                    let cross_d = dot(vi, vj).abs();
                    assert!(cross_d < 1e-5, "v{i} · v{j} = {cross_d}, not orthogonal");
                }
            }
        }
    }

    // Test 11: Gram-Schmidt — doesn't change span (reconstructed same vector up to sign)
    #[test]
    fn test_gram_schmidt_preserves_span() {
        let n = 3;
        let original = vec![1.0f32, 2.0, 3.0];
        let mut vecs = vec![original.clone(), vec![0.0, 1.0, 0.0]];
        spec_gram_schmidt(&mut vecs, n, 2);
        // First GS vector should be proportional to original
        let d = dot(&vecs[0], &original);
        assert!(d.abs() > 0.99, "GS first vector not along original: {d}");
    }

    // Test 12: Rayleigh quotient of constant vector = 0 (L*1=0)
    #[test]
    fn test_rayleigh_quotient_constant() {
        let verts = tet_verts();
        let faces = tet_faces();
        let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
        let ones = vec![1.0f32; lap.n_vertices];
        let rq = spec_rayleigh_quotient(&lap, &ones);
        assert!(
            rq.abs() < 1e-4,
            "RQ of constant vector should be 0, got {rq}"
        );
    }

    // Test 13: Rayleigh quotient of non-constant vector > 0
    #[test]
    fn test_rayleigh_quotient_positive() {
        let verts = tet_verts();
        let faces = tet_faces();
        let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
        let v = vec![1.0f32, -1.0, 1.0, -1.0];
        let rq = spec_rayleigh_quotient(&lap, &v);
        assert!(
            rq > 0.0,
            "RQ of non-constant vector should be positive, got {rq}"
        );
    }

    // Test 14: power iteration on path graph k=2, eigenvalues sorted ascending
    #[test]
    fn test_power_iteration_sorted() {
        let verts = path_verts();
        let faces = path_faces();
        let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
        let basis = spec_power_iteration(&lap, 2, 2000, 1e-7, 42).expect("power iter");
        assert_eq!(basis.k, 2);
        assert!(
            basis.eigenvalues[0] <= basis.eigenvalues[1] + 1e-5,
            "eigenvalues not sorted: {:?}",
            basis.eigenvalues
        );
    }

    // Test 15: power iteration eigenvectors are orthogonal
    #[test]
    fn test_power_iteration_orthogonal() {
        let verts = tet_verts();
        let faces = tet_faces();
        let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
        let basis = spec_power_iteration(&lap, 3, 2000, 1e-7, 7).expect("power iter");
        let k = basis.k;
        for i in 0..k {
            for j in (i + 1)..k {
                let d = dot(&basis.eigenvectors[i], &basis.eigenvectors[j]).abs();
                assert!(d < 0.05, "eigenvectors {i} and {j} not orthogonal: dot={d}");
            }
        }
    }

    // Test 16: smallest eigenvalue ≈ 0 for connected graph
    #[test]
    fn test_power_iteration_smallest_near_zero() {
        let verts = triangle_verts();
        let faces = triangle_faces();
        let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
        let basis = spec_power_iteration(&lap, 2, 2000, 1e-7, 1).expect("power iter");
        assert!(
            basis.eigenvalues[0].abs() < 0.3,
            "smallest eigenvalue should be ~0 for connected graph, got {}",
            basis.eigenvalues[0]
        );
    }

    // Test 17: eigenvalues ordered ascending
    #[test]
    fn test_eigenvalues_ascending() {
        let verts = tet_verts();
        let faces = tet_faces();
        let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
        let basis = spec_power_iteration(&lap, 3, 2000, 1e-7, 99).expect("power iter");
        for i in 0..basis.eigenvalues.len() - 1 {
            assert!(
                basis.eigenvalues[i] <= basis.eigenvalues[i + 1] + 1e-4,
                "eigenvalues not ascending at {i}: {:?}",
                basis.eigenvalues
            );
        }
    }

    // Test 18: project → reconstruct round-trip
    #[test]
    fn test_project_reconstruct_roundtrip() {
        let verts = tet_verts();
        let faces = tet_faces();
        let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
        let basis = spec_power_iteration(&lap, 4, 3000, 1e-8, 5).expect("power iter");
        let signal = SpectralSignal {
            values: vec![1.0, 2.0, 3.0, 4.0],
            n_vertices: 4,
        };
        let coeffs = spec_project(&signal, &basis).expect("project");
        let reconstructed = spec_reconstruct(&coeffs, &basis).expect("reconstruct");
        let mse: f32 = signal
            .values
            .iter()
            .zip(reconstructed.values.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            / 4.0;
        assert!(mse < 0.1, "round-trip MSE = {mse} (should be near 0)");
    }

    // Test 19: low-pass filter produces smoother output
    #[test]
    fn test_low_pass_filter_smoother() {
        let verts = tet_verts();
        let faces = tet_faces();
        let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
        let basis = spec_power_iteration(&lap, 3, 2000, 1e-7, 13).expect("power iter");
        let signal = SpectralSignal {
            values: vec![1.0, -2.0, 3.0, -4.0],
            n_vertices: 4,
        };
        let smoothed = spec_low_pass_filter(&signal, &basis, 1).expect("low pass");
        let s_before = spec_smoothness(&signal, &lap);
        let s_after = spec_smoothness(&smoothed, &lap);
        assert!(
            s_after <= s_before + 1e-3,
            "low pass did not reduce smoothness: before={s_before}, after={s_after}"
        );
    }

    // Test 20: high-pass filter removes low-freq content
    #[test]
    fn test_high_pass_filter() {
        let verts = tet_verts();
        let faces = tet_faces();
        let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
        let basis = spec_power_iteration(&lap, 4, 2000, 1e-7, 21).expect("power iter");
        let signal = SpectralSignal {
            values: vec![1.0, 1.0, 1.0, 2.0],
            n_vertices: 4,
        };
        let coeffs_full = spec_project(&signal, &basis).expect("project full");
        let hp = spec_high_pass_filter(&signal, &basis, 1).expect("high pass");
        let coeffs_hp = spec_project(&hp, &basis).expect("project hp");
        // First coefficient should be zeroed or near zero
        assert!(
            coeffs_hp[0].abs() < coeffs_full[0].abs() + 0.5,
            "high pass did not attenuate low freq: full={}, hp={}",
            coeffs_full[0],
            coeffs_hp[0]
        );
    }

    // Test 21: smoothness of constant signal = 0
    #[test]
    fn test_smoothness_constant_zero() {
        let verts = triangle_verts();
        let faces = triangle_faces();
        let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
        let signal = SpectralSignal {
            values: vec![3.0, 3.0, 3.0],
            n_vertices: 3,
        };
        let s = spec_smoothness(&signal, &lap);
        assert!(s.abs() < 1e-4, "smoothness of constant = {s}");
    }

    // Test 22: smoothness of non-constant > 0
    #[test]
    fn test_smoothness_nonconstant_positive() {
        let verts = triangle_verts();
        let faces = triangle_faces();
        let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
        let signal = SpectralSignal {
            values: vec![1.0, -1.0, 0.0],
            n_vertices: 3,
        };
        let s = spec_smoothness(&signal, &lap);
        assert!(
            s > 0.0,
            "smoothness of non-constant should be positive, got {s}"
        );
    }

    // Test 23: Laplacian smooth changes positions
    #[test]
    fn test_laplacian_smooth_changes() {
        let verts = triangle_verts();
        let faces = triangle_faces();
        let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
        let positions: Vec<f32> = verts.iter().flat_map(|p| [p.x, p.y, p.z]).collect();
        let smoothed = spec_laplacian_smooth(&positions, &lap, 0.1, 5);
        let diff: f32 = positions
            .iter()
            .zip(smoothed.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 1e-6, "Laplacian smooth did not change positions");
    }

    // Test 24: Laplacian smooth with constant positions → no change
    #[test]
    fn test_laplacian_smooth_constant_no_change() {
        let verts = triangle_verts();
        let faces = triangle_faces();
        let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
        // Constant position (all the same): gradient is zero
        let positions = vec![1.0f32; 9]; // 3 vertices × 3 coords, all 1.0
        let smoothed = spec_laplacian_smooth(&positions, &lap, 0.1, 5);
        let diff: f32 = positions
            .iter()
            .zip(smoothed.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff < 1e-5,
            "Constant positions changed under smooth: diff={diff}"
        );
    }

    // Test 25: Taubin smooth changes positions
    #[test]
    fn test_taubin_smooth_changes() {
        let verts = triangle_verts();
        let faces = triangle_faces();
        let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
        let positions: Vec<f32> = verts.iter().flat_map(|p| [p.x, p.y, p.z]).collect();
        let smoothed = spec_taubin_smooth(&positions, &lap, 0.5, -0.53, 10);
        let diff: f32 = positions
            .iter()
            .zip(smoothed.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 0.0, "Taubin smooth did not change positions");
    }

    // Test 26: Taubin closer to original than pure Laplacian (less shrinkage)
    // Use stable parameters: for tet with max eigenvalue ~6, lambda < 1/6 ≈ 0.1
    // Laplacian shrinks mesh each iteration; Taubin compensates with mu step
    #[test]
    fn test_taubin_less_shrinkage() {
        let verts = tet_verts();
        let faces = tet_faces();
        let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
        let positions: Vec<f32> = verts.iter().flat_map(|p| [p.x, p.y, p.z]).collect();
        // Pure Laplacian: 20 iters with lambda=0.05 → steady shrinkage
        let lap_smooth = spec_laplacian_smooth(&positions, &lap, 0.05, 20);
        // Taubin: 5 pairs with stable lambda=0.05, mu=-0.06 → less shrinkage
        let taubin = spec_taubin_smooth(&positions, &lap, 0.05, -0.06, 5);
        let err_lap: f32 = positions
            .iter()
            .zip(lap_smooth.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum();
        let err_taubin: f32 = positions
            .iter()
            .zip(taubin.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum();
        // Taubin with fewer iters must deviate less than pure Laplacian with 20 iters
        assert!(
            err_taubin < err_lap + 1e-3,
            "Taubin should have less deviation from original: lap={err_lap}, taubin={err_taubin}"
        );
    }

    // Test 27: cluster_vertices k=3 returns labels all in [0,3)
    #[test]
    fn test_cluster_labels_in_range() {
        let verts = tet_verts();
        let faces = tet_faces();
        let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
        let basis = spec_power_iteration(&lap, 3, 2000, 1e-7, 77).expect("power iter");
        let labels = spec_cluster_vertices(&basis, 3, 42).expect("cluster");
        assert_eq!(labels.len(), 4);
        for &l in &labels {
            assert!(l < 3, "label {l} out of range [0,3)");
        }
    }

    // Test 28: cluster_vertices returns n_vertices assignments
    #[test]
    fn test_cluster_returns_n_assignments() {
        let verts = tet_verts();
        let faces = tet_faces();
        let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
        let basis = spec_power_iteration(&lap, 2, 2000, 1e-7, 3).expect("power iter");
        let labels = spec_cluster_vertices(&basis, 2, 11).expect("cluster");
        assert_eq!(labels.len(), verts.len());
    }

    // Test 29: stats n_edges correct for single triangle (= 3 undirected edges)
    #[test]
    fn test_stats_n_edges_triangle() {
        let verts = triangle_verts();
        let faces = triangle_faces();
        let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
        let stats = spec_compute_stats(&lap, None);
        assert_eq!(
            stats.n_edges, 3,
            "triangle has 3 edges, got {}",
            stats.n_edges
        );
    }

    // Test 30: stats mean_degree correct for triangle (= 2.0)
    #[test]
    fn test_stats_mean_degree_triangle() {
        let verts = triangle_verts();
        let faces = triangle_faces();
        let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
        let stats = spec_compute_stats(&lap, None);
        assert!(
            (stats.mean_degree - 2.0).abs() < 1e-5,
            "mean degree should be 2.0, got {}",
            stats.mean_degree
        );
    }

    // Test 31: format_stats returns non-empty string
    #[test]
    fn test_format_stats_nonempty() {
        let verts = triangle_verts();
        let faces = triangle_faces();
        let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
        let stats = spec_compute_stats(&lap, None);
        let s = spec_format_stats(&stats);
        assert!(!s.is_empty(), "format_stats returned empty string");
    }

    // Test 32: format_config returns non-empty string
    #[test]
    fn test_format_config_nonempty() {
        let config = SpectralConfig {
            k: 10,
            max_power_iters: 500,
            tol: 1e-6,
            laplacian_kind: LaplacianKind::Combinatorial,
        };
        let s = spec_format_config(&config);
        assert!(!s.is_empty(), "format_config returned empty string");
    }

    // Test 33: EmptyMesh error for 0 vertices
    #[test]
    fn test_empty_mesh_error() {
        let result = spec_build_combinatorial_laplacian(&[], &[]);
        assert!(
            matches!(result, Err(SpectralError::EmptyMesh)),
            "expected EmptyMesh error"
        );
    }

    // Test 34: EmptyMesh error for cotangent with 0 vertices
    #[test]
    fn test_empty_mesh_cotangent() {
        let result = spec_build_cotangent_laplacian(&[], &[]);
        assert!(
            matches!(result, Err(SpectralError::EmptyMesh)),
            "expected EmptyMesh error"
        );
    }

    // Test 35: DimensionMismatch error for project
    #[test]
    fn test_dimension_mismatch_project() {
        let verts = triangle_verts();
        let faces = triangle_faces();
        let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
        let basis = spec_power_iteration(&lap, 2, 1000, 1e-6, 1).expect("power iter");
        let signal = SpectralSignal {
            values: vec![1.0, 2.0, 3.0, 4.0], // 4 != 3
            n_vertices: 4,
        };
        let result = spec_project(&signal, &basis);
        assert!(
            matches!(result, Err(SpectralError::DimensionMismatch { .. })),
            "expected DimensionMismatch error"
        );
    }

    // Test 36: InsufficientBasis error for reconstruct
    #[test]
    fn test_insufficient_basis_reconstruct() {
        let verts = triangle_verts();
        let faces = triangle_faces();
        let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
        let basis = spec_power_iteration(&lap, 2, 1000, 1e-6, 1).expect("power iter");
        // More coefficients than k
        let coeffs = vec![1.0f32; 5];
        let result = spec_reconstruct(&coeffs, &basis);
        assert!(
            matches!(result, Err(SpectralError::InsufficientBasis { .. })),
            "expected InsufficientBasis error"
        );
    }

    // Test 37: InsufficientBasis error for power_iteration with k > n
    #[test]
    fn test_insufficient_basis_k_gt_n() {
        let verts = triangle_verts();
        let faces = triangle_faces();
        let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
        let result = spec_power_iteration(&lap, 10, 100, 1e-6, 1);
        assert!(
            matches!(result, Err(SpectralError::InsufficientBasis { .. })),
            "expected InsufficientBasis error for k > n"
        );
    }

    // Test 38: tetrahedron Laplacian — all degrees = 3
    #[test]
    fn test_tet_all_degrees_three() {
        let verts = tet_verts();
        let faces = tet_faces();
        let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
        for (i, &d) in lap.degree.iter().enumerate() {
            assert!(
                (d - 3.0).abs() < 1e-6,
                "tet vertex {i} degree = {d}, expected 3"
            );
        }
    }

    // Test 39: algebraic connectivity (λ₂) > 0 for connected mesh
    #[test]
    fn test_algebraic_connectivity_positive() {
        let verts = triangle_verts();
        let faces = triangle_faces();
        let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
        let basis = spec_power_iteration(&lap, 2, 2000, 1e-7, 42).expect("power iter");
        let stats = spec_compute_stats(&lap, Some(&basis));
        assert!(
            stats.algebraic_connectivity > -0.1,
            "algebraic connectivity should be >= 0 for connected mesh, got {}",
            stats.algebraic_connectivity
        );
    }

    // Test 40: signal projection coefficients length = k
    #[test]
    fn test_projection_coefficients_length() {
        let verts = tet_verts();
        let faces = tet_faces();
        let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
        let basis = spec_power_iteration(&lap, 3, 2000, 1e-7, 55).expect("power iter");
        let signal = SpectralSignal {
            values: vec![1.0, 2.0, 3.0, 4.0],
            n_vertices: 4,
        };
        let coeffs = spec_project(&signal, &basis).expect("project");
        assert_eq!(coeffs.len(), 3, "coefficients length should be k=3");
    }

    // Test 41: low-pass filter with cutoff_k=0 → all zeros (no basis kept)
    #[test]
    fn test_low_pass_cutoff_zero() {
        let verts = triangle_verts();
        let faces = triangle_faces();
        let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
        let basis = spec_power_iteration(&lap, 2, 2000, 1e-7, 99).expect("power iter");
        let signal = SpectralSignal {
            values: vec![1.0, 2.0, 3.0],
            n_vertices: 3,
        };
        let filtered = spec_low_pass_filter(&signal, &basis, 0).expect("low pass k=0");
        let total: f32 = filtered.values.iter().map(|x| x.abs()).sum();
        assert!(
            total < 1e-4,
            "low pass with cutoff 0 should give ~0 output, got {total}"
        );
    }

    // Test 42: CSR row_ptr is non-decreasing
    #[test]
    fn test_csr_row_ptr_nondecreasing() {
        let verts = tet_verts();
        let faces = tet_faces();
        let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
        for i in 0..lap.row_ptr.len() - 1 {
            assert!(
                lap.row_ptr[i] <= lap.row_ptr[i + 1],
                "row_ptr not non-decreasing at {i}"
            );
        }
    }

    // Test 43: CSR col_idx all in valid range
    #[test]
    fn test_csr_col_idx_valid() {
        let verts = tet_verts();
        let faces = tet_faces();
        let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
        let n = lap.n_vertices;
        for &c in &lap.col_idx {
            assert!(c < n, "col_idx {c} out of range [0, {n})");
        }
    }

    // Test 44: off-diagonal values of combinatorial Laplacian are -1
    #[test]
    fn test_combinatorial_off_diag_neg_one() {
        let verts = triangle_verts();
        let faces = triangle_faces();
        let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
        for &v in &lap.values {
            assert!(
                (v + 1.0).abs() < 1e-6,
                "off-diag value should be -1, got {v}"
            );
        }
    }

    // Test 45: laplacian_matvec with zero vector gives zero
    #[test]
    fn test_matvec_zero_vector() {
        let verts = tet_verts();
        let faces = tet_faces();
        let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
        let zeros = vec![0.0f32; lap.n_vertices];
        let result = spec_laplacian_matvec(&lap, &zeros);
        for &r in &result {
            assert!(r.abs() < 1e-10, "L*0 should be 0, got {r}");
        }
    }

    // Test 46: cluster_vertices with n_clusters=1 → all zeros
    #[test]
    fn test_cluster_one_cluster() {
        let verts = tet_verts();
        let faces = tet_faces();
        let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
        let basis = spec_power_iteration(&lap, 1, 1000, 1e-6, 1).expect("power iter");
        let labels = spec_cluster_vertices(&basis, 1, 7).expect("cluster");
        for &l in &labels {
            assert_eq!(l, 0, "all labels should be 0 for n_clusters=1");
        }
    }

    // Test 47: spectral_gap = algebraic_connectivity - lambda1
    #[test]
    fn test_spectral_gap_definition() {
        let verts = tet_verts();
        let faces = tet_faces();
        let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
        let basis = spec_power_iteration(&lap, 2, 2000, 1e-7, 123).expect("power iter");
        let stats = spec_compute_stats(&lap, Some(&basis));
        let expected_gap = basis.eigenvalues[1] - basis.eigenvalues[0];
        assert!(
            (stats.spectral_gap - expected_gap).abs() < 0.01,
            "spectral gap mismatch: {} vs {}",
            stats.spectral_gap,
            expected_gap
        );
    }

    // Test 48: normalize Laplacian kind is Normalized
    #[test]
    fn test_normalized_laplacian_kind() {
        let verts = triangle_verts();
        let faces = triangle_faces();
        let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
        let norm = spec_normalize_laplacian(&lap).expect("normalize");
        assert!(
            matches!(norm.kind, LaplacianKind::Normalized),
            "kind should be Normalized"
        );
    }

    // Test 49: cotangent Laplacian kind is Cotangent
    #[test]
    fn test_cotangent_laplacian_kind() {
        let verts = triangle_verts();
        let faces = triangle_faces();
        let lap = spec_build_cotangent_laplacian(&verts, &faces).expect("cotangent lap");
        assert!(
            matches!(lap.kind, LaplacianKind::Cotangent),
            "kind should be Cotangent"
        );
    }

    // Test 50: SpectralSignal n_vertices matches values.len()
    #[test]
    fn test_spectral_signal_consistency() {
        let signal = SpectralSignal {
            values: vec![1.0, 2.0, 3.0, 4.0],
            n_vertices: 4,
        };
        assert_eq!(signal.values.len(), signal.n_vertices);
    }

    // Test 51: spec_compute_stats with basis has spectral_radius from basis
    #[test]
    fn test_stats_spectral_radius_from_basis() {
        let verts = tet_verts();
        let faces = tet_faces();
        let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
        let basis = spec_power_iteration(&lap, 3, 2000, 1e-7, 456).expect("power iter");
        let stats = spec_compute_stats(&lap, Some(&basis));
        // spectral_radius should equal max eigenvalue in basis
        let max_ev = basis.eigenvalues.iter().copied().fold(0.0f32, f32::max);
        assert!(
            (stats.spectral_radius - max_ev).abs() < 1e-4,
            "spectral_radius mismatch: {} vs {}",
            stats.spectral_radius,
            max_ev
        );
    }

    // Test 52: tetrahedron has 6 undirected edges
    #[test]
    fn test_tet_n_edges() {
        let verts = tet_verts();
        let faces = tet_faces();
        let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
        let stats = spec_compute_stats(&lap, None);
        assert_eq!(
            stats.n_edges, 6,
            "tetrahedron has 6 edges, got {}",
            stats.n_edges
        );
    }

    // Test 53: xorshift64 never returns 0
    #[test]
    fn test_xorshift64_nonzero() {
        let mut state = 1u64;
        for _ in 0..10_000 {
            let v = xorshift64(&mut state);
            assert_ne!(v, 0, "xorshift64 returned 0");
        }
    }

    // Test 54: format_stats contains expected fields
    #[test]
    fn test_format_stats_contains_fields() {
        let verts = triangle_verts();
        let faces = triangle_faces();
        let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
        let stats = spec_compute_stats(&lap, None);
        let s = spec_format_stats(&stats);
        assert!(s.contains("n_vertices"), "missing n_vertices");
        assert!(s.contains("n_edges"), "missing n_edges");
        assert!(s.contains("mean_degree"), "missing mean_degree");
    }

    // Test 55: format_config contains expected fields
    #[test]
    fn test_format_config_contains_fields() {
        let config = SpectralConfig {
            k: 5,
            max_power_iters: 100,
            tol: 1e-5,
            laplacian_kind: LaplacianKind::Cotangent,
        };
        let s = spec_format_config(&config);
        assert!(s.contains("k:"), "missing k");
        assert!(s.contains("Cotangent"), "missing laplacian kind");
    }

    // Test 56: spec_laplacian_smooth output length = input length
    #[test]
    fn test_laplacian_smooth_output_length() {
        let verts = tet_verts();
        let faces = tet_faces();
        let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
        let positions: Vec<f32> = verts.iter().flat_map(|p| [p.x, p.y, p.z]).collect();
        let smoothed = spec_laplacian_smooth(&positions, &lap, 0.1, 3);
        assert_eq!(smoothed.len(), positions.len());
    }

    // Test 57: spec_taubin_smooth output length = input length
    #[test]
    fn test_taubin_smooth_output_length() {
        let verts = tet_verts();
        let faces = tet_faces();
        let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
        let positions: Vec<f32> = verts.iter().flat_map(|p| [p.x, p.y, p.z]).collect();
        let smoothed = spec_taubin_smooth(&positions, &lap, 0.5, -0.53, 4);
        assert_eq!(smoothed.len(), positions.len());
    }

    // Test 58: cluster_vertices with 0 clusters returns error
    #[test]
    fn test_cluster_zero_clusters_error() {
        let verts = tet_verts();
        let faces = tet_faces();
        let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
        let basis = spec_power_iteration(&lap, 2, 1000, 1e-6, 1).expect("power iter");
        let result = spec_cluster_vertices(&basis, 0, 1);
        assert!(
            matches!(result, Err(SpectralError::InvalidConfig { .. })),
            "expected InvalidConfig error for 0 clusters"
        );
    }

    // Test 59: power_iteration with k=0 returns error
    #[test]
    fn test_power_iteration_k_zero_error() {
        let verts = triangle_verts();
        let faces = triangle_faces();
        let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
        let result = spec_power_iteration(&lap, 0, 100, 1e-6, 1);
        assert!(
            matches!(result, Err(SpectralError::InvalidConfig { .. })),
            "expected InvalidConfig error for k=0"
        );
    }

    // Test 60: n_vertices in basis matches mesh
    #[test]
    fn test_basis_n_vertices_matches() {
        let verts = tet_verts();
        let faces = tet_faces();
        let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
        let basis = spec_power_iteration(&lap, 2, 1000, 1e-6, 1).expect("power iter");
        assert_eq!(basis.n_vertices, verts.len());
    }
}
