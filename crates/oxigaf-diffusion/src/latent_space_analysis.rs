//! Latent space analysis for diffusion models.
//!
//! Provides tools for analyzing the structure of diffusion latent spaces:
//! computing statistics, finding principal components (PCA via power iteration),
//! clustering latents (k-means), and measuring interpolation quality.

use thiserror::Error;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during latent space analysis.
#[derive(Debug, Error)]
pub enum LatentAnalysisError {
    #[error("Empty latent dataset: no vectors provided")]
    EmptyDataset,
    #[error(
        "Latent vectors have inconsistent dimensions: expected {expected}, got {got} at index {idx}"
    )]
    InconsistentDimensions {
        expected: usize,
        got: usize,
        idx: usize,
    },
    #[error("Requested {k} principal components but latent dimension is only {dim}")]
    InsufficientDimension { k: usize, dim: usize },
    #[error("K-means requires k >= 1, got {k}")]
    InvalidK { k: usize },
    #[error("Power iteration diverged: norm {norm:.4e} is too large")]
    PowerIterationDiverged { norm: f32 },
    #[error("Empty latent vector")]
    EmptyLatent,
}

// ---------------------------------------------------------------------------
// Core data structures
// ---------------------------------------------------------------------------

/// Per-dimension statistics for a set of latent vectors.
#[derive(Debug, Clone)]
pub struct LatentDimStats {
    pub dim_idx: usize,
    pub mean: f32,
    pub variance: f32,
    /// Raw 4th moment / variance^2 (3.0 for a normal distribution).
    pub kurtosis: f32,
    /// 3rd moment / std^3.
    pub skewness: f32,
    pub min: f32,
    pub max: f32,
}

/// Overall statistics for a latent dataset.
#[derive(Debug, Clone)]
pub struct LatentDatasetStats {
    pub n_latents: usize,
    pub latent_dim: usize,
    /// Per-dimension stats.
    pub dim_stats: Vec<LatentDimStats>,
    /// Mean L2 norm across all latents.
    pub mean_l2_norm: f32,
    /// Std of L2 norms.
    pub std_l2_norm: f32,
    /// Mean pairwise L2 distance (computed on a subset).
    pub mean_pairwise_distance: f32,
}

/// Result of PCA on a latent dataset.
#[derive(Debug, Clone)]
pub struct PcaResult {
    /// Top-k principal components; each vector has length `latent_dim`.
    pub components: Vec<Vec<f32>>,
    /// Variance explained by each component (eigenvalues of the covariance matrix).
    pub explained_variance: Vec<f32>,
    /// Fraction of total variance explained by each component.
    pub explained_variance_ratio: Vec<f32>,
    /// Cumulative explained variance ratio.
    pub cumulative_variance_ratio: Vec<f32>,
    pub total_variance: f32,
}

/// K-means clustering result.
#[derive(Debug, Clone)]
pub struct LatentClusterResult {
    /// Cluster centroids (k × dim).
    pub centroids: Vec<Vec<f32>>,
    /// Cluster index for each latent.
    pub assignments: Vec<usize>,
    /// Sum of squared distances to assigned centroid.
    pub inertia: f32,
    pub n_iterations: usize,
    /// Number of latents in each cluster.
    pub cluster_sizes: Vec<usize>,
}

/// Configuration for latent analysis.
#[derive(Debug, Clone)]
pub struct LatentAnalysisConfig {
    /// Number of PCA components (default: 10).
    pub pca_n_components: usize,
    /// Power iteration steps (default: 50).
    pub pca_n_iter: usize,
    /// K-means clusters (default: 8).
    pub kmeans_k: usize,
    /// K-means max iterations (default: 100).
    pub kmeans_max_iter: usize,
    /// Max latents to use for pairwise distance computation (default: 50).
    pub pairwise_sample_size: usize,
    /// Seed for reproducibility (default: 42).
    pub random_seed: u64,
}

impl Default for LatentAnalysisConfig {
    fn default() -> Self {
        Self {
            pca_n_components: 10,
            pca_n_iter: 50,
            kmeans_k: 8,
            kmeans_max_iter: 100,
            pairwise_sample_size: 50,
            random_seed: 42,
        }
    }
}

/// Interpolation quality measures for a linear path in latent space.
#[derive(Debug, Clone)]
pub struct InterpolationQuality {
    /// Mean absolute difference between consecutive interpolation steps (averaged across dimensions).
    pub smoothness: f32,
    /// How close to linear: 1.0 = perfectly linear path.
    pub linearity: f32,
    /// Total path length in latent space.
    pub arc_length: f32,
    /// Direct distance from start to end.
    pub chord_length: f32,
    /// arc_length / chord_length (1.0 = straight line).
    pub detour_ratio: f32,
}

// ---------------------------------------------------------------------------
// Private vector math helpers (lsa_ prefix to avoid symbol conflicts)
// ---------------------------------------------------------------------------

/// L2 norm of a vector.
fn lsa_l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Dot product of two equal-length vectors.
fn lsa_dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// L2 distance between two equal-length vectors.
fn lsa_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

/// Arithmetic mean of a slice.
fn lsa_mean(v: &[f32]) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    v.iter().sum::<f32>() / v.len() as f32
}

/// Population standard deviation given a precomputed mean.
fn lsa_std(v: &[f32], mean: f32) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    let var = v.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / v.len() as f32;
    var.sqrt()
}

/// Normalize a vector to unit L2 norm. Returns `None` if the norm is near zero.
fn lsa_normalize(v: &[f32]) -> Option<Vec<f32>> {
    let norm = lsa_l2_norm(v);
    if norm < 1e-10 {
        return None;
    }
    Some(v.iter().map(|x| x / norm).collect())
}

// ---------------------------------------------------------------------------
// xorshift64 RNG (no rand crate)
// ---------------------------------------------------------------------------

/// Advance the xorshift64 state and return the next pseudo-random u64.
fn xorshift64(state: &mut u64) -> u64 {
    if *state == 0 {
        *state = 1;
    }
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

/// Generate a pseudo-random f32 in `[-1, 1]` from the RNG state.
fn rng_f32(state: &mut u64) -> f32 {
    let raw = xorshift64(state);
    // Map to [0, 1) then shift to [-1, 1).
    (raw as f64 / u64::MAX as f64) as f32 * 2.0 - 1.0
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Compute per-dimension statistics for a single dimension across all latents.
pub fn compute_dim_stats(
    latents: &[Vec<f32>],
    dim_idx: usize,
) -> Result<LatentDimStats, LatentAnalysisError> {
    if latents.is_empty() {
        return Err(LatentAnalysisError::EmptyDataset);
    }

    let values: Vec<f32> = latents
        .iter()
        .map(|v| {
            v.get(dim_idx)
                .copied()
                .ok_or(LatentAnalysisError::EmptyLatent)
        })
        .collect::<Result<Vec<_>, _>>()?;

    let n = values.len() as f32;
    let mean = lsa_mean(&values);
    let variance = values.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n;
    let min = values.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    let std = variance.sqrt();
    let (skewness, kurtosis) = if std < 1e-8 {
        (0.0, 0.0)
    } else {
        let skewness = values
            .iter()
            .map(|x| ((x - mean) / std).powi(3))
            .sum::<f32>()
            / n;
        let kurtosis = values
            .iter()
            .map(|x| ((x - mean) / std).powi(4))
            .sum::<f32>()
            / n;
        (skewness, kurtosis)
    };

    Ok(LatentDimStats {
        dim_idx,
        mean,
        variance,
        kurtosis,
        skewness,
        min,
        max,
    })
}

/// Compute overall statistics for a latent dataset.
pub fn compute_latent_dataset_stats(
    latents: &[Vec<f32>],
    config: &LatentAnalysisConfig,
) -> Result<LatentDatasetStats, LatentAnalysisError> {
    if latents.is_empty() {
        return Err(LatentAnalysisError::EmptyDataset);
    }

    let dim = latents[0].len();

    // Validate consistency.
    for (idx, v) in latents.iter().enumerate() {
        if v.len() != dim {
            return Err(LatentAnalysisError::InconsistentDimensions {
                expected: dim,
                got: v.len(),
                idx,
            });
        }
    }

    // Per-dimension stats.
    let dim_stats: Vec<LatentDimStats> = (0..dim)
        .map(|d| compute_dim_stats(latents, d))
        .collect::<Result<Vec<_>, _>>()?;

    // L2 norms.
    let norms: Vec<f32> = latents.iter().map(|v| lsa_l2_norm(v)).collect();
    let norm_mean = lsa_mean(&norms);
    let norm_std = lsa_std(&norms, norm_mean);

    // Pairwise distances on first `pairwise_sample_size` latents.
    let sample_n = latents.len().min(config.pairwise_sample_size);
    let sample = &latents[..sample_n];
    let mut total_dist = 0.0f32;
    let mut n_pairs = 0usize;
    for i in 0..sample_n {
        for j in (i + 1)..sample_n {
            total_dist += lsa_distance(&sample[i], &sample[j]);
            n_pairs += 1;
        }
    }
    let mean_pairwise_distance = if n_pairs > 0 {
        total_dist / n_pairs as f32
    } else {
        0.0
    };

    Ok(LatentDatasetStats {
        n_latents: latents.len(),
        latent_dim: dim,
        dim_stats,
        mean_l2_norm: norm_mean,
        std_l2_norm: norm_std,
        mean_pairwise_distance,
    })
}

// ---------------------------------------------------------------------------
// PCA via deflation / power iteration
// ---------------------------------------------------------------------------

/// One step of power iteration using the implicit covariance matrix.
///
/// Computes `C * vector` where `C = (1/n) X^T X` (X is centered data, n×d).
///
/// # Parameters
/// - `matrix`: n rows, each of length d (centered data)
/// - `vector`: length d
///
/// Returns the unnormalized d-dimensional result.
pub fn power_iteration_step(matrix: &[Vec<f32>], vector: &[f32]) -> Vec<f32> {
    let n = matrix.len();
    if n == 0 || vector.is_empty() {
        return vec![0.0; vector.len()];
    }
    let d = vector.len();

    // u[i] = dot(matrix[i], vector)
    let u: Vec<f32> = matrix.iter().map(|row| lsa_dot(row, vector)).collect();

    // w[j] = sum_i(u[i] * matrix[i][j]) / n
    let mut w = vec![0.0f32; d];
    for (i, row) in matrix.iter().enumerate() {
        let ui = u[i];
        for j in 0..d {
            w[j] += ui * row[j];
        }
    }
    let inv_n = 1.0 / n as f32;
    for wj in &mut w {
        *wj *= inv_n;
    }
    w
}

/// Compute PCA via deflation power iteration.
///
/// Returns the top `n_components` principal components with explained variances.
pub fn compute_pca(
    latents: &[Vec<f32>],
    n_components: usize,
    n_iter: usize,
    seed: u64,
) -> Result<PcaResult, LatentAnalysisError> {
    if latents.is_empty() {
        return Err(LatentAnalysisError::EmptyDataset);
    }

    let dim = latents[0].len();
    if dim == 0 {
        return Err(LatentAnalysisError::EmptyLatent);
    }
    if n_components > dim {
        return Err(LatentAnalysisError::InsufficientDimension {
            k: n_components,
            dim,
        });
    }

    // Center the data: subtract per-dimension mean.
    let mut col_means = vec![0.0f32; dim];
    for lat in latents.iter() {
        for (j, &x) in lat.iter().enumerate() {
            col_means[j] += x;
        }
    }
    let inv_n = 1.0 / latents.len() as f32;
    for m in &mut col_means {
        *m *= inv_n;
    }

    // Centered data (mutable for deflation).
    let mut centered: Vec<Vec<f32>> = latents
        .iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .map(|(j, &x)| x - col_means[j])
                .collect()
        })
        .collect();

    let mut components = Vec::with_capacity(n_components);
    let mut explained_variance = Vec::with_capacity(n_components);
    let mut rng_state = seed;
    if rng_state == 0 {
        rng_state = 1;
    }

    for _k in 0..n_components {
        // Initialise random unit vector.
        let mut v: Vec<f32> = (0..dim).map(|_| rng_f32(&mut rng_state)).collect();
        v = match lsa_normalize(&v) {
            Some(u) => u,
            None => {
                // Degenerate — fall back to a basis vector.
                let mut e = vec![0.0f32; dim];
                e[0] = 1.0;
                e
            }
        };

        // Power iteration.
        let mut eigenvalue = 0.0f32;
        for _ in 0..n_iter {
            let w = power_iteration_step(&centered, &v);
            let norm = lsa_l2_norm(&w);
            if norm < 1e-12 {
                // Zero eigenvector; stop.
                break;
            }
            eigenvalue = norm;
            if eigenvalue > 1e10 {
                return Err(LatentAnalysisError::PowerIterationDiverged { norm: eigenvalue });
            }
            v = w.iter().map(|x| x / norm).collect();
        }

        // Deflate: X[i] -= (X[i] . v) * v
        for row in centered.iter_mut() {
            let proj = lsa_dot(row, &v);
            for j in 0..dim {
                row[j] -= proj * v[j];
            }
        }

        components.push(v);
        explained_variance.push(eigenvalue);
    }

    let total_variance: f32 = explained_variance.iter().sum();
    let explained_variance_ratio: Vec<f32> = if total_variance > 1e-12 {
        explained_variance
            .iter()
            .map(|&e| e / total_variance)
            .collect()
    } else {
        vec![0.0f32; n_components]
    };

    let mut cumulative = 0.0f32;
    let cumulative_variance_ratio: Vec<f32> = explained_variance_ratio
        .iter()
        .map(|&r| {
            cumulative += r;
            cumulative
        })
        .collect();

    Ok(PcaResult {
        components,
        explained_variance,
        explained_variance_ratio,
        cumulative_variance_ratio,
        total_variance,
    })
}

// ---------------------------------------------------------------------------
// K-means clustering
// ---------------------------------------------------------------------------

/// K-means++ initialization: pick `k` initial centroids.
pub fn kmeans_init(latents: &[Vec<f32>], k: usize, rng: &mut u64) -> Vec<Vec<f32>> {
    let n = latents.len();
    if n == 0 || k == 0 {
        return Vec::new();
    }

    let first_idx = (xorshift64(rng) as usize) % n;
    let mut centroids = vec![latents[first_idx].clone()];

    while centroids.len() < k {
        // Compute squared distances from each point to the nearest centroid.
        let sq_dists: Vec<f32> = latents
            .iter()
            .map(|lat| {
                centroids
                    .iter()
                    .map(|c| {
                        let d = lsa_distance(lat, c);
                        d * d
                    })
                    .fold(f32::INFINITY, f32::min)
            })
            .collect();

        let total: f64 = sq_dists.iter().map(|&d| d as f64).sum();
        if total < 1e-30 {
            // All points coincide with existing centroids; fill remainder by repetition.
            centroids.push(latents[0].clone());
            continue;
        }

        // Sample proportional to squared distance.
        let threshold = (xorshift64(rng) as f64 / u64::MAX as f64) * total;
        let mut cumsum = 0.0f64;
        let mut chosen = n - 1;
        for (i, &d) in sq_dists.iter().enumerate() {
            cumsum += d as f64;
            if cumsum >= threshold {
                chosen = i;
                break;
            }
        }
        centroids.push(latents[chosen].clone());
    }

    centroids
}

/// Assign each latent to the nearest centroid (by L2 distance).
pub fn assign_to_clusters(latents: &[Vec<f32>], centroids: &[Vec<f32>]) -> Vec<usize> {
    latents
        .iter()
        .map(|lat| {
            centroids
                .iter()
                .enumerate()
                .map(|(k, c)| (k, lsa_distance(lat, c)))
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(k, _)| k)
                .unwrap_or(0)
        })
        .collect()
}

/// Recompute centroids as the mean of assigned points. Empty clusters keep the old centroid.
pub fn update_centroids(
    latents: &[Vec<f32>],
    assignments: &[usize],
    k: usize,
    dim: usize,
    old_centroids: &[Vec<f32>],
) -> Vec<Vec<f32>> {
    let mut sums = vec![vec![0.0f32; dim]; k];
    let mut counts = vec![0usize; k];

    for (lat, &cluster) in latents.iter().zip(assignments.iter()) {
        if cluster < k {
            for j in 0..dim {
                sums[cluster][j] += lat[j];
            }
            counts[cluster] += 1;
        }
    }

    sums.into_iter()
        .enumerate()
        .map(|(ki, sum)| {
            if counts[ki] == 0 {
                old_centroids[ki].clone()
            } else {
                sum.iter().map(|&s| s / counts[ki] as f32).collect()
            }
        })
        .collect()
}

/// Sum of squared distances from each point to its assigned centroid.
pub fn compute_inertia(latents: &[Vec<f32>], assignments: &[usize], centroids: &[Vec<f32>]) -> f32 {
    latents
        .iter()
        .zip(assignments.iter())
        .map(|(lat, &cluster)| {
            if cluster < centroids.len() {
                let d = lsa_distance(lat, &centroids[cluster]);
                d * d
            } else {
                0.0
            }
        })
        .sum()
}

/// Run k-means clustering on a set of latent vectors.
pub fn run_kmeans(
    latents: &[Vec<f32>],
    k: usize,
    max_iter: usize,
    seed: u64,
) -> Result<LatentClusterResult, LatentAnalysisError> {
    if latents.is_empty() {
        return Err(LatentAnalysisError::EmptyDataset);
    }
    if k == 0 {
        return Err(LatentAnalysisError::InvalidK { k });
    }

    let dim = latents[0].len();
    let mut rng = if seed == 0 { 1 } else { seed };

    let mut centroids = kmeans_init(latents, k, &mut rng);
    let mut assignments = assign_to_clusters(latents, &centroids);
    let mut n_iterations = 0usize;

    for _ in 0..max_iter {
        let new_centroids = update_centroids(latents, &assignments, k, dim, &centroids);
        let new_assignments = assign_to_clusters(latents, &new_centroids);
        n_iterations += 1;

        let changed = new_assignments
            .iter()
            .zip(assignments.iter())
            .any(|(a, b)| a != b);
        centroids = new_centroids;
        assignments = new_assignments;

        if !changed {
            break;
        }
    }

    let inertia = compute_inertia(latents, &assignments, &centroids);

    let mut cluster_sizes = vec![0usize; k];
    for &a in &assignments {
        if a < k {
            cluster_sizes[a] += 1;
        }
    }

    Ok(LatentClusterResult {
        centroids,
        assignments,
        inertia,
        n_iterations,
        cluster_sizes,
    })
}

// ---------------------------------------------------------------------------
// Interpolation analysis
// ---------------------------------------------------------------------------

/// Measure the quality of linear interpolation between two latent vectors.
///
/// `n_steps` must be >= 2.
pub fn compute_interpolation_quality(
    start: &[f32],
    end: &[f32],
    n_steps: usize,
) -> Result<InterpolationQuality, LatentAnalysisError> {
    if start.is_empty() || end.is_empty() {
        return Err(LatentAnalysisError::EmptyLatent);
    }
    if n_steps < 2 {
        // Treat n_steps=1 or 0 as degenerate: just return chord with no path.
        let chord_length = lsa_distance(start, end);
        return Ok(InterpolationQuality {
            smoothness: 0.0,
            linearity: 1.0,
            arc_length: chord_length,
            chord_length,
            detour_ratio: 1.0,
        });
    }

    let dim = start.len();
    // Generate interpolated sequence.
    let interp: Vec<Vec<f32>> = (0..n_steps)
        .map(|i| {
            let t = i as f32 / (n_steps - 1) as f32;
            start
                .iter()
                .zip(end.iter())
                .map(|(&s, &e)| (1.0 - t) * s + t * e)
                .collect()
        })
        .collect();

    // Arc length: sum of L2 distances between consecutive steps.
    let mut arc_length = 0.0f32;
    for i in 0..(n_steps - 1) {
        arc_length += lsa_distance(&interp[i], &interp[i + 1]);
    }

    let chord_length = lsa_distance(start, end);

    // Smoothness: mean absolute change per step, averaged across dimensions.
    let mut total_abs_change = 0.0f32;
    let mut n_pairs = 0usize;
    for i in 0..(n_steps - 1) {
        for (&a, &b) in interp[i + 1].iter().zip(interp[i].iter()) {
            total_abs_change += (a - b).abs();
        }
        n_pairs += 1;
    }
    let smoothness = if n_pairs > 0 && dim > 0 {
        total_abs_change / (n_pairs as f32 * dim as f32)
    } else {
        0.0
    };

    let linearity = if arc_length > 1e-8 {
        chord_length / arc_length
    } else {
        1.0
    };
    let detour_ratio = arc_length / chord_length.max(1e-8);

    Ok(InterpolationQuality {
        smoothness,
        linearity,
        arc_length,
        chord_length,
        detour_ratio,
    })
}

// ---------------------------------------------------------------------------
// Intrinsic dimensionality estimate
// ---------------------------------------------------------------------------

/// Estimate the intrinsic dimensionality of a latent dataset using PCA.
///
/// Returns the number of components needed to explain 90% of variance (as f32).
pub fn estimate_intrinsic_dimensionality(
    latents: &[Vec<f32>],
    config: &LatentAnalysisConfig,
) -> Result<f32, LatentAnalysisError> {
    if latents.is_empty() {
        return Err(LatentAnalysisError::EmptyDataset);
    }
    let dim = latents[0].len();
    if dim == 0 {
        return Err(LatentAnalysisError::EmptyLatent);
    }

    let n_components = config.pca_n_components.min(dim);
    let pca = compute_pca(latents, n_components, config.pca_n_iter, config.random_seed)?;

    // Find first index where cumulative variance >= 0.9.
    for (idx, &cum) in pca.cumulative_variance_ratio.iter().enumerate() {
        if cum >= 0.9 {
            return Ok((idx + 1) as f32);
        }
    }

    // All components don't reach 90%; return the number we computed.
    Ok(n_components as f32)
}

// ---------------------------------------------------------------------------
// Dataset entropy
// ---------------------------------------------------------------------------

/// Compute the Shannon entropy (in bits) of the first dimension of all latent vectors.
///
/// A histogram with `n_bins` equal-width bins over `[min, max]` is used.
pub fn compute_dataset_entropy(
    latents: &[Vec<f32>],
    n_bins: usize,
) -> Result<f32, LatentAnalysisError> {
    if latents.is_empty() {
        return Err(LatentAnalysisError::EmptyDataset);
    }
    if n_bins == 0 {
        return Ok(0.0);
    }

    let values: Vec<f32> = latents
        .iter()
        .map(|v| v.first().copied().ok_or(LatentAnalysisError::EmptyLatent))
        .collect::<Result<Vec<_>, _>>()?;

    let min_val = values.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    if (max_val - min_val).abs() < 1e-12 {
        // All values the same → entropy 0.
        return Ok(0.0);
    }

    let range = max_val - min_val;
    let mut counts = vec![0usize; n_bins];
    let n = values.len();

    for &v in &values {
        let bin_f = (v - min_val) / range * n_bins as f32;
        let bin = (bin_f as usize).min(n_bins - 1);
        counts[bin] += 1;
    }

    let entropy: f32 = counts
        .iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f32 / n as f32;
            -p * p.log2()
        })
        .sum();

    Ok(entropy)
}

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------

/// Human-readable summary of a `PcaResult`.
///
/// Example: `"PCA: top-3 explain [23.4%, 15.1%, 8.9%] = 47.4% cumulative"`
pub fn format_pca_summary(pca: &PcaResult) -> String {
    let top = pca.explained_variance_ratio.len().min(3);
    let parts: Vec<String> = pca.explained_variance_ratio[..top]
        .iter()
        .map(|&r| format!("{:.1}%", r * 100.0))
        .collect();
    let cumulative = pca
        .cumulative_variance_ratio
        .get(top.saturating_sub(1))
        .copied()
        .unwrap_or(0.0);
    format!(
        "PCA: top-{} explain [{}] = {:.1}% cumulative",
        top,
        parts.join(", "),
        cumulative * 100.0
    )
}

/// Human-readable summary of a `LatentClusterResult`.
///
/// Example: `"KMeans[k=4]: inertia=3.142, sizes=[2, 3, 1, 4]"`
pub fn format_cluster_summary(result: &LatentClusterResult) -> String {
    let k = result.centroids.len();
    let sizes: Vec<String> = result.cluster_sizes.iter().map(|s| s.to_string()).collect();
    format!(
        "KMeans[k={}]: inertia={:.3}, sizes=[{}]",
        k,
        result.inertia,
        sizes.join(", ")
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- lsa_l2_norm ---

    #[test]
    fn test_lsa_l2_norm_zero_vector() {
        let v = vec![0.0f32, 0.0, 0.0];
        assert_eq!(lsa_l2_norm(&v), 0.0);
    }

    #[test]
    fn test_lsa_l2_norm_known() {
        let v = vec![3.0f32, 4.0];
        let n = lsa_l2_norm(&v);
        assert!((n - 5.0).abs() < 1e-6, "expected 5.0, got {n}");
    }

    #[test]
    fn test_lsa_l2_norm_unit_vector() {
        let v = vec![1.0f32, 0.0, 0.0];
        assert!((lsa_l2_norm(&v) - 1.0).abs() < 1e-7);
    }

    // --- lsa_distance ---

    #[test]
    fn test_lsa_distance_same_vector() {
        let v = vec![1.0f32, 2.0, 3.0];
        assert!(lsa_distance(&v, &v) < 1e-7);
    }

    #[test]
    fn test_lsa_distance_known() {
        let a = vec![0.0f32, 0.0];
        let b = vec![3.0f32, 4.0];
        assert!((lsa_distance(&a, &b) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_lsa_distance_single_element() {
        let a = vec![1.0f32];
        let b = vec![4.0f32];
        assert!((lsa_distance(&a, &b) - 3.0).abs() < 1e-6);
    }

    // --- lsa_normalize ---

    #[test]
    fn test_lsa_normalize_unit_norm() {
        let v = vec![1.0f32, 2.0, 3.0];
        let u = lsa_normalize(&v).expect("should normalize non-zero");
        let norm = lsa_l2_norm(&u);
        assert!((norm - 1.0).abs() < 1e-6, "norm should be 1.0, got {norm}");
    }

    #[test]
    fn test_lsa_normalize_zero_returns_none() {
        let v = vec![0.0f32, 0.0];
        assert!(lsa_normalize(&v).is_none());
    }

    #[test]
    fn test_lsa_normalize_direction_preserved() {
        let v = vec![0.0f32, 5.0];
        let u = lsa_normalize(&v).expect("should normalize");
        assert!((u[0]).abs() < 1e-7);
        assert!((u[1] - 1.0).abs() < 1e-7);
    }

    // --- compute_dim_stats ---

    #[test]
    fn test_compute_dim_stats_constant() -> Result<(), LatentAnalysisError> {
        let latents = vec![vec![3.0f32], vec![3.0f32], vec![3.0f32]];
        let s = compute_dim_stats(&latents, 0)?;
        assert!((s.mean - 3.0).abs() < 1e-6, "mean should be 3");
        assert!(s.variance < 1e-12, "variance should be ~0");
        assert_eq!(s.skewness, 0.0, "skewness should be 0 when std~0");
        assert_eq!(s.kurtosis, 0.0, "kurtosis should be 0 when std~0");
        Ok(())
    }

    #[test]
    fn test_compute_dim_stats_known_mean() -> Result<(), LatentAnalysisError> {
        let latents = vec![vec![1.0f32, 0.0], vec![3.0f32, 0.0]];
        let s = compute_dim_stats(&latents, 0)?;
        assert!((s.mean - 2.0).abs() < 1e-6, "mean should be 2.0");
        assert!((s.min - 1.0).abs() < 1e-6);
        assert!((s.max - 3.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_compute_dim_stats_empty_dataset() {
        let result = compute_dim_stats(&[], 0);
        assert!(matches!(result, Err(LatentAnalysisError::EmptyDataset)));
    }

    #[test]
    fn test_compute_dim_stats_variance() -> Result<(), LatentAnalysisError> {
        // Population variance of [1, 3] = ((1-2)^2 + (3-2)^2) / 2 = 1.0
        let latents = vec![vec![1.0f32], vec![3.0f32]];
        let s = compute_dim_stats(&latents, 0)?;
        assert!((s.variance - 1.0).abs() < 1e-6, "variance should be 1.0");
        Ok(())
    }

    // --- compute_latent_dataset_stats ---

    #[test]
    fn test_compute_dataset_stats_single() -> Result<(), LatentAnalysisError> {
        let latents = vec![vec![1.0f32, 2.0, 3.0]];
        let config = LatentAnalysisConfig::default();
        let stats = compute_latent_dataset_stats(&latents, &config)?;
        assert_eq!(stats.n_latents, 1);
        assert_eq!(stats.latent_dim, 3);
        assert_eq!(stats.mean_pairwise_distance, 0.0);
        Ok(())
    }

    #[test]
    fn test_compute_dataset_stats_multiple() -> Result<(), LatentAnalysisError> {
        let latents = vec![vec![0.0f32, 0.0], vec![3.0f32, 4.0]];
        let config = LatentAnalysisConfig::default();
        let stats = compute_latent_dataset_stats(&latents, &config)?;
        assert_eq!(stats.n_latents, 2);
        assert_eq!(stats.latent_dim, 2);
        // Mean pairwise distance should be 5.0.
        assert!((stats.mean_pairwise_distance - 5.0).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn test_compute_dataset_stats_inconsistent_dims() {
        let latents = vec![vec![1.0f32, 2.0], vec![1.0f32]];
        let config = LatentAnalysisConfig::default();
        let result = compute_latent_dataset_stats(&latents, &config);
        assert!(matches!(
            result,
            Err(LatentAnalysisError::InconsistentDimensions { .. })
        ));
    }

    #[test]
    fn test_compute_dataset_stats_empty() {
        let config = LatentAnalysisConfig::default();
        let result = compute_latent_dataset_stats(&[], &config);
        assert!(matches!(result, Err(LatentAnalysisError::EmptyDataset)));
    }

    #[test]
    fn test_compute_dataset_stats_dim_count() -> Result<(), LatentAnalysisError> {
        let latents: Vec<Vec<f32>> = (0..5).map(|_| vec![1.0f32; 4]).collect();
        let config = LatentAnalysisConfig::default();
        let stats = compute_latent_dataset_stats(&latents, &config)?;
        assert_eq!(stats.dim_stats.len(), 4);
        Ok(())
    }

    // --- power_iteration_step ---

    #[test]
    fn test_power_iteration_step_smoke() {
        // 2 data points, dim=2; should return a 2-element result.
        let matrix = vec![vec![1.0f32, 0.0], vec![0.0f32, 1.0]];
        let vector = vec![1.0f32, 0.0];
        let result = power_iteration_step(&matrix, &vector);
        assert_eq!(result.len(), 2);
        // C = I/2; C*e1 = 0.5 * e1
        assert!(
            (result[0] - 0.5).abs() < 1e-6,
            "expected 0.5, got {}",
            result[0]
        );
        assert!(result[1].abs() < 1e-6);
    }

    #[test]
    fn test_power_iteration_step_empty_matrix() {
        let matrix: Vec<Vec<f32>> = vec![];
        let vector = vec![1.0f32, 0.0];
        let result = power_iteration_step(&matrix, &vector);
        assert_eq!(result.len(), 2);
        assert!(result.iter().all(|&x| x == 0.0));
    }

    // --- compute_pca ---

    #[test]
    fn test_compute_pca_single_component() -> Result<(), LatentAnalysisError> {
        // 1D-like data: all variation in first coordinate.
        let latents = vec![vec![1.0f32, 0.0], vec![2.0f32, 0.0], vec![3.0f32, 0.0]];
        let pca = compute_pca(&latents, 1, 100, 42)?;
        assert_eq!(pca.components.len(), 1);
        assert_eq!(pca.components[0].len(), 2);
        // First component should point mostly along dim 0.
        assert!(pca.components[0][0].abs() > 0.9);
        Ok(())
    }

    #[test]
    fn test_compute_pca_explained_variance_nonneg() -> Result<(), LatentAnalysisError> {
        let latents: Vec<Vec<f32>> = (0..10).map(|i| vec![i as f32, (i as f32) * 0.5]).collect();
        let pca = compute_pca(&latents, 2, 50, 42)?;
        for &ev in &pca.explained_variance {
            assert!(ev >= 0.0, "explained variance should be non-negative");
        }
        Ok(())
    }

    #[test]
    fn test_compute_pca_ratios_sum_to_one() -> Result<(), LatentAnalysisError> {
        let latents: Vec<Vec<f32>> = (0..8)
            .map(|i| vec![i as f32, i as f32 * 2.0, i as f32 * 0.1])
            .collect();
        let pca = compute_pca(&latents, 3, 50, 42)?;
        let ratio_sum: f32 = pca.explained_variance_ratio.iter().sum();
        assert!(
            (ratio_sum - 1.0).abs() < 1e-5,
            "ratios should sum to 1, got {ratio_sum}"
        );
        Ok(())
    }

    #[test]
    fn test_compute_pca_insufficient_dim() {
        let latents = vec![vec![1.0f32, 2.0], vec![3.0f32, 4.0]];
        let result = compute_pca(&latents, 5, 10, 42);
        assert!(matches!(
            result,
            Err(LatentAnalysisError::InsufficientDimension { .. })
        ));
    }

    #[test]
    fn test_compute_pca_cumulative_monotone() -> Result<(), LatentAnalysisError> {
        let latents: Vec<Vec<f32>> = (0..5).map(|i| vec![i as f32, i as f32]).collect();
        let pca = compute_pca(&latents, 2, 30, 7)?;
        let cum = &pca.cumulative_variance_ratio;
        for i in 1..cum.len() {
            assert!(
                cum[i] >= cum[i - 1] - 1e-7,
                "cumulative ratio should be monotone"
            );
        }
        Ok(())
    }

    // --- assign_to_clusters ---

    #[test]
    fn test_assign_to_clusters_known() {
        let centroids = vec![vec![0.0f32, 0.0], vec![10.0f32, 0.0]];
        let latents = vec![vec![1.0f32, 0.0], vec![9.0f32, 0.0]];
        let assignments = assign_to_clusters(&latents, &centroids);
        assert_eq!(assignments[0], 0);
        assert_eq!(assignments[1], 1);
    }

    #[test]
    fn test_assign_to_clusters_equidistant() {
        let centroids = vec![vec![0.0f32], vec![2.0f32]];
        let latents = vec![vec![1.0f32]];
        // 1.0 is equidistant; expect either cluster (just not panic).
        let assignments = assign_to_clusters(&latents, &centroids);
        assert!(assignments[0] == 0 || assignments[0] == 1);
    }

    // --- update_centroids ---

    #[test]
    fn test_update_centroids_known() {
        let latents = vec![vec![0.0f32, 0.0], vec![2.0f32, 0.0], vec![10.0f32, 0.0]];
        let assignments = vec![0, 0, 1];
        let old_centroids = vec![vec![0.0f32, 0.0], vec![0.0f32, 0.0]];
        let new_centroids = update_centroids(&latents, &assignments, 2, 2, &old_centroids);
        // Cluster 0: mean of [0,0] and [2,0] = [1,0].
        assert!((new_centroids[0][0] - 1.0).abs() < 1e-6);
        // Cluster 1: mean of [10,0].
        assert!((new_centroids[1][0] - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_update_centroids_empty_cluster_keeps_old() {
        let latents = vec![vec![1.0f32]];
        let assignments = vec![0]; // cluster 1 is empty
        let old_centroids = vec![vec![0.0f32], vec![99.0f32]];
        let new_centroids = update_centroids(&latents, &assignments, 2, 1, &old_centroids);
        // Cluster 1 was empty; keep old centroid 99.
        assert!((new_centroids[1][0] - 99.0).abs() < 1e-6);
    }

    // --- compute_inertia ---

    #[test]
    fn test_compute_inertia_zero_when_points_equal_centroids() {
        let centroids = vec![vec![1.0f32, 2.0], vec![3.0f32, 4.0]];
        let latents = vec![vec![1.0f32, 2.0], vec![3.0f32, 4.0]];
        let assignments = vec![0, 1];
        let inertia = compute_inertia(&latents, &assignments, &centroids);
        assert!(inertia < 1e-10, "inertia should be 0, got {inertia}");
    }

    #[test]
    fn test_compute_inertia_known() {
        let centroids = vec![vec![0.0f32]];
        let latents = vec![vec![3.0f32], vec![4.0f32]];
        let assignments = vec![0, 0];
        let inertia = compute_inertia(&latents, &assignments, &centroids);
        // (3^2 + 4^2) = 25.
        assert!((inertia - 25.0).abs() < 1e-5, "expected 25, got {inertia}");
    }

    // --- run_kmeans ---

    #[test]
    fn test_run_kmeans_k1() -> Result<(), LatentAnalysisError> {
        let latents: Vec<Vec<f32>> = (0..5).map(|i| vec![i as f32]).collect();
        let result = run_kmeans(&latents, 1, 100, 42)?;
        assert_eq!(result.centroids.len(), 1);
        assert!(result.assignments.iter().all(|&a| a == 0));
        assert_eq!(result.cluster_sizes[0], 5);
        Ok(())
    }

    #[test]
    fn test_run_kmeans_invalid_k() {
        let latents = vec![vec![1.0f32]];
        let result = run_kmeans(&latents, 0, 10, 42);
        assert!(matches!(result, Err(LatentAnalysisError::InvalidK { .. })));
    }

    #[test]
    fn test_run_kmeans_k_equals_n() -> Result<(), LatentAnalysisError> {
        let latents = vec![vec![0.0f32], vec![100.0f32], vec![200.0f32]];
        let result = run_kmeans(&latents, 3, 100, 42)?;
        assert_eq!(result.centroids.len(), 3);
        assert_eq!(result.cluster_sizes.iter().sum::<usize>(), 3);
        Ok(())
    }

    #[test]
    fn test_run_kmeans_inertia_nonneg() -> Result<(), LatentAnalysisError> {
        let latents: Vec<Vec<f32>> = (0..20).map(|i| vec![i as f32, (i as f32).sin()]).collect();
        let result = run_kmeans(&latents, 4, 50, 7)?;
        assert!(result.inertia >= 0.0);
        Ok(())
    }

    #[test]
    fn test_run_kmeans_cluster_sizes_sum() -> Result<(), LatentAnalysisError> {
        let latents: Vec<Vec<f32>> = (0..12).map(|i| vec![i as f32]).collect();
        let result = run_kmeans(&latents, 3, 50, 42)?;
        let total: usize = result.cluster_sizes.iter().sum();
        assert_eq!(total, 12);
        Ok(())
    }

    // --- compute_interpolation_quality ---

    #[test]
    fn test_interpolation_quality_n_steps_2() -> Result<(), LatentAnalysisError> {
        let start = vec![0.0f32, 0.0];
        let end = vec![3.0f32, 4.0];
        let q = compute_interpolation_quality(&start, &end, 2)?;
        // For linear interp, arc_length == chord_length.
        assert!((q.arc_length - 5.0).abs() < 1e-5);
        assert!((q.chord_length - 5.0).abs() < 1e-5);
        assert!((q.linearity - 1.0).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn test_interpolation_quality_same_start_end() -> Result<(), LatentAnalysisError> {
        let v = vec![1.0f32, 2.0];
        let q = compute_interpolation_quality(&v, &v, 10)?;
        // All interp points are the same.
        assert!(q.arc_length < 1e-7);
        assert!(q.chord_length < 1e-7);
        assert!((q.linearity - 1.0).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn test_interpolation_quality_linearity_is_one() -> Result<(), LatentAnalysisError> {
        // Linear interpolation should have perfect linearity.
        let start = vec![0.0f32, 0.0, 0.0];
        let end = vec![1.0f32, 1.0, 1.0];
        let q = compute_interpolation_quality(&start, &end, 11)?;
        assert!(
            (q.linearity - 1.0).abs() < 1e-5,
            "linearity={}",
            q.linearity
        );
        Ok(())
    }

    #[test]
    fn test_interpolation_quality_arc_equals_chord_for_linear() -> Result<(), LatentAnalysisError> {
        let start = vec![-1.0f32, 0.0];
        let end = vec![1.0f32, 0.0];
        let q = compute_interpolation_quality(&start, &end, 20)?;
        assert!((q.arc_length - q.chord_length).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn test_interpolation_quality_empty_returns_err() {
        let result = compute_interpolation_quality(&[], &[1.0], 5);
        assert!(matches!(result, Err(LatentAnalysisError::EmptyLatent)));
    }

    // --- estimate_intrinsic_dimensionality ---

    #[test]
    fn test_estimate_intrinsic_dim_1d_data() -> Result<(), LatentAnalysisError> {
        // Variation only along dim 0 → intrinsic dim should be 1.
        let latents: Vec<Vec<f32>> = (0..10).map(|i| vec![i as f32, 0.0, 0.0, 0.0]).collect();
        let config = LatentAnalysisConfig {
            pca_n_components: 4,
            ..Default::default()
        };
        let d = estimate_intrinsic_dimensionality(&latents, &config)?;
        assert!(d <= 2.0, "expected low intrinsic dim, got {d}");
        Ok(())
    }

    #[test]
    fn test_estimate_intrinsic_dim_empty() {
        let config = LatentAnalysisConfig::default();
        let result = estimate_intrinsic_dimensionality(&[], &config);
        assert!(matches!(result, Err(LatentAnalysisError::EmptyDataset)));
    }

    #[test]
    fn test_estimate_intrinsic_dim_is_positive() -> Result<(), LatentAnalysisError> {
        let latents: Vec<Vec<f32>> = (0..15)
            .map(|i| vec![i as f32, (i as f32) * 0.5, (i as f32) * 0.1])
            .collect();
        let config = LatentAnalysisConfig {
            pca_n_components: 3,
            ..Default::default()
        };
        let d = estimate_intrinsic_dimensionality(&latents, &config)?;
        assert!(d >= 1.0, "intrinsic dim should be >= 1, got {d}");
        Ok(())
    }

    // --- compute_dataset_entropy ---

    #[test]
    fn test_entropy_all_same_value() -> Result<(), LatentAnalysisError> {
        let latents: Vec<Vec<f32>> = (0..10).map(|_| vec![3.0f32]).collect();
        let entropy = compute_dataset_entropy(&latents, 10)?;
        assert!(
            entropy < 1e-6,
            "all-same → entropy should be 0, got {entropy}"
        );
        Ok(())
    }

    #[test]
    fn test_entropy_uniform_is_higher() -> Result<(), LatentAnalysisError> {
        // Uniform distribution over many bins should be higher entropy than concentrated.
        let uniform: Vec<Vec<f32>> = (0..100).map(|i| vec![i as f32]).collect();
        let concentrated: Vec<Vec<f32>> = (0..100).map(|_| vec![1.0f32]).collect();
        let h_uniform = compute_dataset_entropy(&uniform, 20)?;
        let h_concentrated = compute_dataset_entropy(&concentrated, 20)?;
        assert!(
            h_uniform > h_concentrated,
            "uniform entropy {h_uniform} should exceed concentrated {h_concentrated}"
        );
        Ok(())
    }

    #[test]
    fn test_entropy_empty_dataset() {
        let result = compute_dataset_entropy(&[], 10);
        assert!(matches!(result, Err(LatentAnalysisError::EmptyDataset)));
    }

    #[test]
    fn test_entropy_nonneg() -> Result<(), LatentAnalysisError> {
        let latents: Vec<Vec<f32>> = (0..20).map(|i| vec![(i % 5) as f32]).collect();
        let h = compute_dataset_entropy(&latents, 5)?;
        assert!(h >= 0.0, "entropy should be non-negative");
        Ok(())
    }

    // --- format_pca_summary ---

    #[test]
    fn test_format_pca_summary_nonempty() {
        let pca = PcaResult {
            components: vec![vec![1.0f32, 0.0]],
            explained_variance: vec![2.0],
            explained_variance_ratio: vec![1.0],
            cumulative_variance_ratio: vec![1.0],
            total_variance: 2.0,
        };
        let s = format_pca_summary(&pca);
        assert!(!s.is_empty());
        assert!(s.contains('%'), "should contain percentage");
        assert!(s.contains("PCA"), "should start with PCA");
    }

    #[test]
    fn test_format_pca_summary_three_components() {
        let pca = PcaResult {
            components: vec![vec![1.0f32]; 3],
            explained_variance: vec![3.0, 2.0, 1.0],
            explained_variance_ratio: vec![0.5, 0.333, 0.167],
            cumulative_variance_ratio: vec![0.5, 0.833, 1.0],
            total_variance: 6.0,
        };
        let s = format_pca_summary(&pca);
        assert!(s.contains("top-3"), "should mention top-3");
    }

    // --- format_cluster_summary ---

    #[test]
    fn test_format_cluster_summary_nonempty() {
        let result = LatentClusterResult {
            centroids: vec![vec![0.0f32], vec![1.0f32]],
            assignments: vec![0, 0, 1],
            inertia: 3.1,
            n_iterations: 5,
            cluster_sizes: vec![2, 1],
        };
        let s = format_cluster_summary(&result);
        assert!(!s.is_empty());
        assert!(s.contains("KMeans"), "should contain KMeans");
        assert!(s.contains("inertia"), "should contain inertia");
    }

    #[test]
    fn test_format_cluster_summary_k4() {
        let result = LatentClusterResult {
            centroids: vec![vec![0.0f32]; 4],
            assignments: vec![0, 1, 2, 3],
            inertia: 0.0,
            n_iterations: 1,
            cluster_sizes: vec![1, 1, 1, 1],
        };
        let s = format_cluster_summary(&result);
        assert!(s.contains("k=4"), "should show k=4");
    }

    // --- LatentAnalysisError variants ---

    #[test]
    fn test_error_empty_dataset_display() {
        let e = LatentAnalysisError::EmptyDataset;
        let s = e.to_string();
        assert!(s.contains("Empty"), "error message should mention Empty");
    }

    #[test]
    fn test_error_inconsistent_dimensions() {
        let e = LatentAnalysisError::InconsistentDimensions {
            expected: 4,
            got: 3,
            idx: 2,
        };
        let s = e.to_string();
        assert!(s.contains("4") && s.contains("3") && s.contains("2"));
    }

    #[test]
    fn test_error_insufficient_dimension() {
        let e = LatentAnalysisError::InsufficientDimension { k: 10, dim: 3 };
        let s = e.to_string();
        assert!(s.contains("10") && s.contains("3"));
    }

    #[test]
    fn test_error_invalid_k() {
        let e = LatentAnalysisError::InvalidK { k: 0 };
        let s = e.to_string();
        assert!(s.contains('0'));
    }

    #[test]
    fn test_error_power_iteration_diverged() {
        let e = LatentAnalysisError::PowerIterationDiverged { norm: 1e12 };
        let s = e.to_string();
        assert!(s.contains("diverged") || s.contains("norm"));
    }

    #[test]
    fn test_error_empty_latent() {
        let e = LatentAnalysisError::EmptyLatent;
        let s = e.to_string();
        assert!(s.contains("Empty"));
    }

    // --- PcaResult fields ---

    #[test]
    fn test_pca_result_explained_variance_leq_total() -> Result<(), LatentAnalysisError> {
        let latents: Vec<Vec<f32>> = (0..6).map(|i| vec![i as f32, i as f32]).collect();
        let pca = compute_pca(&latents, 2, 30, 42)?;
        let sum_ev: f32 = pca.explained_variance.iter().sum();
        assert!(
            sum_ev <= pca.total_variance + 1e-4,
            "sum of explained variance ({sum_ev}) should not exceed total ({})",
            pca.total_variance
        );
        Ok(())
    }

    // --- LatentClusterResult fields ---

    #[test]
    fn test_cluster_result_n_iterations_positive() -> Result<(), LatentAnalysisError> {
        let latents: Vec<Vec<f32>> = (0..6).map(|i| vec![i as f32]).collect();
        let result = run_kmeans(&latents, 2, 50, 42)?;
        assert!(result.n_iterations >= 1);
        Ok(())
    }

    #[test]
    fn test_cluster_result_assignments_in_range() -> Result<(), LatentAnalysisError> {
        let latents: Vec<Vec<f32>> = (0..8).map(|i| vec![i as f32]).collect();
        let k = 3;
        let result = run_kmeans(&latents, k, 50, 42)?;
        for &a in &result.assignments {
            assert!(a < k, "assignment {a} out of range for k={k}");
        }
        Ok(())
    }

    // --- InterpolationQuality fields ---

    #[test]
    fn test_interp_quality_detour_ratio_one_for_linear() -> Result<(), LatentAnalysisError> {
        let start = vec![0.0f32; 5];
        let end = vec![1.0f32; 5];
        let q = compute_interpolation_quality(&start, &end, 15)?;
        assert!(
            (q.detour_ratio - 1.0).abs() < 1e-5,
            "detour_ratio={}",
            q.detour_ratio
        );
        Ok(())
    }

    #[test]
    fn test_interp_quality_smoothness_nonneg() -> Result<(), LatentAnalysisError> {
        let start = vec![0.0f32, 1.0];
        let end = vec![2.0f32, 3.0];
        let q = compute_interpolation_quality(&start, &end, 7)?;
        assert!(q.smoothness >= 0.0);
        Ok(())
    }
}
