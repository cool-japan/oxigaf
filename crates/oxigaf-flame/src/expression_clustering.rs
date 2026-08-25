//! K-means clustering and analysis of facial expressions in FLAME expression
//! parameter space.
//!
//! Useful for building expression libraries, finding representative
//! expressions, and categorising expression data collected from FLAME
//! parameter trajectories.

use thiserror::Error;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during expression clustering operations.
#[derive(Debug, Error)]
pub enum ExpressionClusterError {
    /// Input expression set is empty.
    #[error("Empty expression set")]
    EmptyExpressions,

    /// Requested k is out of valid range.
    #[error("Invalid k: {k} (must be 1 ≤ k ≤ n_samples)")]
    InvalidK { k: usize },

    /// Vector dimensions do not match.
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    /// Configuration value is invalid.
    #[error("Invalid config: {0}")]
    InvalidConfig(String),

    /// K-means did not converge within the allowed iteration budget.
    #[error("Failed to converge after {max_iter} iterations")]
    ConvergenceError { max_iter: usize },

    /// A cluster ended up with no members.
    #[error("Empty cluster {id}")]
    EmptyCluster { id: usize },
}

/// Result type for one K-means iteration step: (assignments, centroids, inertia).
pub type KmeansStepResult = Result<(Vec<usize>, Vec<Vec<f32>>, f32), ExpressionClusterError>;

// ---------------------------------------------------------------------------
// Inline xorshift64 PRNG (no rand crate)
// ---------------------------------------------------------------------------

#[inline]
fn xorshift64(state: &mut u64) -> u64 {
    *state = (*state).max(1);
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// Generate a pseudo-random `f32` in `[0.0, 1.0)`.
#[inline]
fn rand_f32(state: &mut u64) -> f32 {
    let bits = xorshift64(state);
    let mantissa = (bits >> 41) as u32;
    let float_bits: u32 = 0x3f80_0000_u32 | mantissa;
    f32::from_bits(float_bits) - 1.0_f32
}

// ---------------------------------------------------------------------------
// ExpressionDataset
// ---------------------------------------------------------------------------

/// A collection of expression parameter vectors (FLAME ψ params).
#[derive(Debug, Clone)]
pub struct ExpressionDataset {
    /// Each inner `Vec<f32>` is one expression (length == `dim`).
    pub expressions: Vec<Vec<f32>>,
    /// Optional human-readable names, one per expression.
    pub labels: Option<Vec<String>>,
    /// Dimensionality: `expressions[i].len()` for every `i`.
    pub dim: usize,
}

impl ExpressionDataset {
    /// Create a dataset from a collection of expression vectors.
    ///
    /// All vectors must have the same length.  Returns
    /// [`ExpressionClusterError::EmptyExpressions`] when `expressions` is
    /// empty, and [`ExpressionClusterError::DimensionMismatch`] when lengths
    /// differ.
    ///
    /// # Errors
    ///
    /// Returns an error if the operation fails.
    pub fn new(expressions: Vec<Vec<f32>>) -> Result<Self, ExpressionClusterError> {
        if expressions.is_empty() {
            return Err(ExpressionClusterError::EmptyExpressions);
        }
        let dim = expressions[0].len();
        for (_idx, expr) in expressions.iter().enumerate().skip(1) {
            if expr.len() != dim {
                return Err(ExpressionClusterError::DimensionMismatch {
                    expected: dim,
                    actual: expr.len(),
                });
            }
        }
        Ok(Self {
            expressions,
            labels: None,
            dim,
        })
    }

    /// Attach per-expression labels.
    ///
    /// The `labels` slice must have the same length as `self.expressions`.
    ///
    /// # Errors
    ///
    /// Returns an error if the operation fails.
    pub fn with_labels(mut self, labels: Vec<String>) -> Result<Self, ExpressionClusterError> {
        if labels.len() != self.expressions.len() {
            return Err(ExpressionClusterError::DimensionMismatch {
                expected: self.expressions.len(),
                actual: labels.len(),
            });
        }
        self.labels = Some(labels);
        Ok(self)
    }

    /// Number of expression samples.
    #[must_use]
    pub fn n_samples(&self) -> usize {
        self.expressions.len()
    }

    /// Retrieve the `i`-th expression, or `None` if out of range.
    #[must_use]
    pub fn get(&self, i: usize) -> Option<&Vec<f32>> {
        self.expressions.get(i)
    }

    /// Retrieve the label for expression `i`, or `None`.
    #[must_use]
    pub fn label(&self, i: usize) -> Option<&str> {
        self.labels.as_ref()?.get(i).map(String::as_str)
    }

    /// Compute the element-wise mean across all expressions.
    #[must_use]
    pub fn mean(&self) -> Vec<f32> {
        let n = self.expressions.len() as f32;
        let mut mean = vec![0.0_f32; self.dim];
        for expr in &self.expressions {
            for (m, &v) in mean.iter_mut().zip(expr.iter()) {
                *m += v;
            }
        }
        for m in &mut mean {
            *m /= n;
        }
        mean
    }

    /// Compute the element-wise standard deviation across all expressions.
    #[must_use]
    pub fn std(&self) -> Vec<f32> {
        let mean = self.mean();
        let n = self.expressions.len() as f32;
        let mut var = vec![0.0_f32; self.dim];
        for expr in &self.expressions {
            for (v_var, (&val, &m)) in var.iter_mut().zip(expr.iter().zip(mean.iter())) {
                let diff = val - m;
                *v_var += diff * diff;
            }
        }
        var.iter().map(|&v| (v / n).sqrt()).collect()
    }

    /// Return a new dataset where each dimension is mean-centred and scaled by
    /// its standard deviation.
    ///
    /// Dimensions with near-zero std (< 1e-12) are left as zero rather than
    /// producing NaN/inf.
    ///
    /// # Errors
    ///
    /// Returns an error if the operation fails.
    pub fn normalized(&self) -> Result<ExpressionDataset, ExpressionClusterError> {
        let mean = self.mean();
        let std_devs = self.std();
        let normed: Vec<Vec<f32>> = self
            .expressions
            .iter()
            .map(|expr| {
                expr.iter()
                    .zip(mean.iter().zip(std_devs.iter()))
                    .map(
                        |(&v, (&m, &s))| {
                            if s < 1e-12_f32 {
                                0.0_f32
                            } else {
                                (v - m) / s
                            }
                        },
                    )
                    .collect()
            })
            .collect();
        let mut ds = ExpressionDataset::new(normed)?;
        if let Some(ref lbls) = self.labels {
            ds = ds.with_labels(lbls.clone())?;
        }
        Ok(ds)
    }
}

// ---------------------------------------------------------------------------
// KMeansConfig
// ---------------------------------------------------------------------------

/// Configuration for K-means clustering.
#[derive(Debug, Clone)]
pub struct KMeansConfig {
    /// Number of clusters.
    pub k: usize,
    /// Maximum number of iterations per run.
    pub max_iter: usize,
    /// Number of random restarts; the run with lowest inertia is kept.
    pub n_restarts: usize,
    /// Centroid shift tolerance for convergence detection.
    pub tol: f32,
    /// PRNG seed.
    pub seed: u64,
}

impl Default for KMeansConfig {
    fn default() -> Self {
        Self {
            k: 2,
            max_iter: 100,
            n_restarts: 3,
            tol: 1e-4,
            seed: 42,
        }
    }
}

impl KMeansConfig {
    /// Create a config with the given `k` and sensible defaults.
    #[must_use]
    pub fn new(k: usize) -> Self {
        Self {
            k,
            ..Self::default()
        }
    }

    /// Validate all fields.  Returns an error if any value is out of range.
    ///
    /// # Errors
    ///
    /// Returns an error if the operation fails.
    pub fn validate(&self) -> Result<(), ExpressionClusterError> {
        if self.k == 0 {
            return Err(ExpressionClusterError::InvalidConfig(
                "k must be ≥ 1".to_string(),
            ));
        }
        if self.max_iter == 0 {
            return Err(ExpressionClusterError::InvalidConfig(
                "max_iter must be ≥ 1".to_string(),
            ));
        }
        if self.n_restarts == 0 {
            return Err(ExpressionClusterError::InvalidConfig(
                "n_restarts must be ≥ 1".to_string(),
            ));
        }
        if self.tol < 0.0 {
            return Err(ExpressionClusterError::InvalidConfig(
                "tol must be ≥ 0".to_string(),
            ));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// ClusteringResult
// ---------------------------------------------------------------------------

/// The result of a K-means clustering run.
#[derive(Debug, Clone)]
pub struct ClusteringResult {
    /// Cluster id (0-indexed) assigned to each expression.
    pub assignments: Vec<usize>,
    /// One centroid per cluster.
    pub centroids: Vec<Vec<f32>>,
    /// Sum of squared distances from each point to its assigned centroid.
    pub inertia: f32,
    /// Number of iterations executed in the winning run.
    pub n_iter: usize,
    /// Whether the algorithm converged before `max_iter`.
    pub converged: bool,
}

impl ClusteringResult {
    /// Indices (into the original dataset) of all members of `cluster_id`.
    #[must_use]
    pub fn cluster_members(&self, cluster_id: usize) -> Vec<usize> {
        self.assignments
            .iter()
            .enumerate()
            .filter_map(|(i, &a)| if a == cluster_id { Some(i) } else { None })
            .collect()
    }

    /// Number of samples assigned to `cluster_id`.
    #[must_use]
    pub fn cluster_size(&self, cluster_id: usize) -> usize {
        self.assignments
            .iter()
            .filter(|&&a| a == cluster_id)
            .count()
    }

    /// Build a table (indexed by cluster id) of member sample indices, in a
    /// single O(n) pass over `assignments`.
    ///
    /// This is the shared building block behind [`Self::silhouette_score`]
    /// and the free functions [`davies_bouldin_index`] and
    /// [`describe_clusters`]/[`compute_cluster_stats`]: call it once and
    /// reuse the result across all clusters/samples instead of re-scanning
    /// `assignments` once per cluster (or, in `silhouette_score`'s case,
    /// once per cluster *per sample*). Any assignment `>= self.centroids.len()`
    /// (only possible via manual construction, since `assignments` and
    /// `centroids` are independently public fields) is silently dropped
    /// rather than panicking.
    #[must_use]
    pub fn membership_table(&self) -> Vec<Vec<usize>> {
        let k = self.centroids.len();
        let mut table = vec![Vec::new(); k];
        for (i, &c) in self.assignments.iter().enumerate() {
            if let Some(members) = table.get_mut(c) {
                members.push(i);
            }
        }
        table
    }

    /// Mean silhouette score over all samples.
    ///
    /// For k == 1 (no meaningful inter-cluster distance), returns 0.0.
    ///
    /// Silhouette for sample `i`:  `(b_i - a_i) / max(a_i, b_i)` where
    /// - `a_i` = mean intra-cluster distance to other members, and
    /// - `b_i` = mean distance to members of the nearest other cluster.
    ///
    /// # Errors
    ///
    /// Returns an error if the operation fails.
    pub fn silhouette_score(
        &self,
        dataset: &ExpressionDataset,
    ) -> Result<f32, ExpressionClusterError> {
        let num_clusters = self.centroids.len();
        if num_clusters <= 1 {
            return Ok(0.0);
        }
        let num_samples = dataset.n_samples();
        if num_samples == 0 {
            return Err(ExpressionClusterError::EmptyExpressions);
        }

        // Precompute cluster membership once (a single O(n) pass) instead
        // of re-scanning `assignments` for every (sample, cluster) pair
        // below — previously O(n * k) scans on top of the unavoidable
        // O(n^2) pairwise-distance work.
        let membership = self.membership_table();
        let empty: Vec<usize> = Vec::new();

        let mut total = 0.0_f32;
        for i in 0..num_samples {
            let expr_i = dataset
                .get(i)
                .ok_or(ExpressionClusterError::EmptyExpressions)?;
            // `.get` (rather than direct indexing) so a `self.assignments`
            // shorter than `dataset` degrades gracefully instead of
            // panicking; `usize::MAX` never matches a real cluster id.
            let cluster_i = self.assignments.get(i).copied().unwrap_or(usize::MAX);
            let own_members = membership.get(cluster_i).unwrap_or(&empty);

            // Intra-cluster mean distance (excluding `i` itself).
            let intra_mean_dist = if own_members.len() <= 1 {
                0.0_f32
            } else {
                let sum: f32 = own_members
                    .iter()
                    .filter(|&&j| j != i)
                    .map(|&j| euclidean_sq(expr_i, &dataset.expressions[j]).sqrt())
                    .sum();
                sum / (own_members.len() - 1) as f32
            };

            // Nearest-cluster mean distance
            let mut nearest_cluster_dist = f32::INFINITY;
            for (cluster_idx, other_members) in membership.iter().enumerate() {
                if cluster_idx == cluster_i || other_members.is_empty() {
                    continue;
                }
                let mean_dist: f32 = other_members
                    .iter()
                    .map(|&j| euclidean_sq(expr_i, &dataset.expressions[j]).sqrt())
                    .sum::<f32>()
                    / other_members.len() as f32;
                if mean_dist < nearest_cluster_dist {
                    nearest_cluster_dist = mean_dist;
                }
            }

            let denom = intra_mean_dist.max(nearest_cluster_dist);
            let sil = if denom < 1e-12 {
                0.0
            } else {
                (nearest_cluster_dist - intra_mean_dist) / denom
            };
            total += sil;
        }
        Ok(total / num_samples as f32)
    }
}

// ---------------------------------------------------------------------------
// ExpressionCluster (descriptor)
// ---------------------------------------------------------------------------

/// Detailed descriptor for one cluster produced by [`describe_clusters`].
#[derive(Debug, Clone)]
pub struct ExpressionCluster {
    /// Zero-based cluster identifier.
    pub id: usize,
    /// Centroid vector.
    pub centroid: Vec<f32>,
    /// Indices into the original dataset.
    pub members: Vec<usize>,
    /// Mean distance from members to the centroid.
    pub mean_distance: f32,
    /// Maximum distance from any member to the centroid (radius).
    pub max_distance: f32,
    /// Optional human-readable name for this cluster.
    pub label: Option<String>,
}

// ---------------------------------------------------------------------------
// Cluster statistics summary
// ---------------------------------------------------------------------------

/// Summary statistics for a complete clustering solution.
#[derive(Debug, Clone)]
pub struct ClusterStats {
    /// Number of clusters.
    pub k: usize,
    /// Total within-cluster sum of squared distances.
    pub total_inertia: f32,
    /// Mean silhouette score.
    pub silhouette: f32,
    /// Davies–Bouldin index (lower = better separation).
    pub davies_bouldin: f32,
    /// Number of samples in each cluster.
    pub cluster_sizes: Vec<usize>,
    /// Mean of the per-cluster radii.
    pub mean_cluster_radius: f32,
}

// ---------------------------------------------------------------------------
// Helper: squared Euclidean distance
// ---------------------------------------------------------------------------

#[inline]
fn euclidean_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y) * (x - y))
        .sum()
}

// ---------------------------------------------------------------------------
// Public functions
// ---------------------------------------------------------------------------

/// L2 (Euclidean) distance between two expression vectors.
///
/// # Errors
///
/// Returns an error if the operation fails.
pub fn expression_distance(a: &[f32], b: &[f32]) -> Result<f32, ExpressionClusterError> {
    if a.len() != b.len() {
        return Err(ExpressionClusterError::DimensionMismatch {
            expected: a.len(),
            actual: b.len(),
        });
    }
    Ok(euclidean_sq(a, b).sqrt())
}

/// Cosine similarity between two expression vectors.
///
/// Returns values in `[-1.0, 1.0]`.  Returns 0.0 for zero vectors.
///
/// # Errors
///
/// Returns an error if the operation fails.
pub fn expression_cosine_similarity(a: &[f32], b: &[f32]) -> Result<f32, ExpressionClusterError> {
    if a.len() != b.len() {
        return Err(ExpressionClusterError::DimensionMismatch {
            expected: a.len(),
            actual: b.len(),
        });
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    if norm_a < 1e-12 || norm_b < 1e-12 {
        return Ok(0.0);
    }
    Ok((dot / (norm_a * norm_b)).clamp(-1.0, 1.0))
}

/// K-means++ initialisation: choose `k` centroids with probability
/// proportional to squared distance from the already-chosen centroids.
///
/// # Errors
///
/// Returns an error if the operation fails.
pub fn kmeans_plus_plus_init(
    expressions: &[Vec<f32>],
    k: usize,
    seed: u64,
) -> Result<Vec<Vec<f32>>, ExpressionClusterError> {
    let n = expressions.len();
    if n == 0 {
        return Err(ExpressionClusterError::EmptyExpressions);
    }
    if k == 0 || k > n {
        return Err(ExpressionClusterError::InvalidK { k });
    }

    let mut state = seed.max(1);
    let mut centroids: Vec<Vec<f32>> = Vec::with_capacity(k);

    // Choose the first centroid uniformly at random.
    let first = (xorshift64(&mut state) % n as u64) as usize;
    centroids.push(expressions[first].clone());

    for _ in 1..k {
        // Compute squared distances from every point to the nearest centroid.
        let mut dists: Vec<f32> = expressions
            .iter()
            .map(|expr| {
                centroids
                    .iter()
                    .map(|c| euclidean_sq(expr, c))
                    .fold(f32::INFINITY, f32::min)
            })
            .collect();

        let total: f32 = dists.iter().sum();
        if total < 1e-30 {
            // All points coincide with existing centroids (to within
            // floating-point tolerance), so no point is distinct enough to
            // pick by variance. Prefer a genuinely distinct point if one
            // exists; otherwise fall back to a deterministic (reproducible)
            // duplicate so this loop iteration always pushes exactly one
            // centroid — guaranteeing the function returns exactly `k`
            // centroids as documented, rather than silently returning
            // fewer when the data is this degenerate.
            let pick = expressions
                .iter()
                .find(|expr| {
                    !centroids.iter().any(|c| {
                        c.iter()
                            .zip(expr.iter())
                            .all(|(&a, &b)| (a - b).abs() < 1e-12)
                    })
                })
                .cloned()
                .unwrap_or_else(|| expressions[centroids.len() % n].clone());
            centroids.push(pick);
            continue;
        }

        // Normalise to a probability distribution and sample.
        for d in &mut dists {
            *d /= total;
        }
        let r = rand_f32(&mut state);
        let mut cumulative = 0.0_f32;
        let mut chosen = n - 1; // fallback
        for (i, &p) in dists.iter().enumerate() {
            cumulative += p;
            if r <= cumulative {
                chosen = i;
                break;
            }
        }
        centroids.push(expressions[chosen].clone());
    }

    Ok(centroids)
}

/// Run one assignment + update step of K-means.
///
/// Returns `(new_assignments, new_centroids, inertia)`.
///
/// # Errors
///
/// Returns an error if the operation fails.
pub fn kmeans_iteration(expressions: &[Vec<f32>], centroids: &[Vec<f32>]) -> KmeansStepResult {
    let n = expressions.len();
    let k = centroids.len();
    if n == 0 {
        return Err(ExpressionClusterError::EmptyExpressions);
    }
    if k == 0 {
        return Err(ExpressionClusterError::InvalidK { k: 0 });
    }
    let dim = expressions[0].len();

    // --- Assignment step ---
    let mut assignments = vec![0usize; n];
    let mut inertia = 0.0_f32;
    for (i, expr) in expressions.iter().enumerate() {
        let (best_c, best_d) = centroids
            .iter()
            .enumerate()
            .map(|(c, centroid)| (c, euclidean_sq(expr, centroid)))
            .fold(
                (0, f32::INFINITY),
                |(bc, bd), (c, d)| {
                    if d < bd {
                        (c, d)
                    } else {
                        (bc, bd)
                    }
                },
            );
        assignments[i] = best_c;
        inertia += best_d;
    }

    // --- Update step ---
    let mut sums = vec![vec![0.0_f32; dim]; k];
    let mut counts = vec![0usize; k];
    for (i, expr) in expressions.iter().enumerate() {
        let c = assignments[i];
        counts[c] += 1;
        for (s, &v) in sums[c].iter_mut().zip(expr.iter()) {
            *s += v;
        }
    }

    let mut new_centroids = Vec::with_capacity(k);
    for (c, (sum, &cnt)) in sums.iter().zip(counts.iter()).enumerate() {
        if cnt == 0 {
            // Keep the old centroid when the cluster is empty.
            new_centroids.push(centroids[c].clone());
        } else {
            new_centroids.push(sum.iter().map(|&s| s / cnt as f32).collect());
        }
    }

    Ok((assignments, new_centroids, inertia))
}

/// Full K-means clustering with multiple random restarts.
///
/// Uses K-means++ initialisation.  The run with the lowest final inertia is
/// returned.
///
/// # Errors
///
/// Returns an error if the operation fails.
pub fn kmeans(
    dataset: &ExpressionDataset,
    config: &KMeansConfig,
) -> Result<ClusteringResult, ExpressionClusterError> {
    config.validate()?;
    let n = dataset.n_samples();
    if n == 0 {
        return Err(ExpressionClusterError::EmptyExpressions);
    }
    if config.k > n {
        return Err(ExpressionClusterError::InvalidK { k: config.k });
    }

    let exprs = &dataset.expressions;
    let mut best: Option<ClusteringResult> = None;

    for restart in 0..config.n_restarts {
        // Vary the seed per restart so we get different initialisations.
        let restart_seed = config
            .seed
            .wrapping_add((restart as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15));
        let mut centroids = kmeans_plus_plus_init(exprs, config.k, restart_seed)?;

        let mut assignments = vec![0usize; n];
        let mut inertia = 0.0_f32;
        let mut n_iter = 0usize;
        let mut converged = false;

        for iter in 0..config.max_iter {
            let (new_assignments, new_centroids, new_inertia) =
                kmeans_iteration(exprs, &centroids)?;

            // Check centroid shift for convergence.
            let max_shift = centroids
                .iter()
                .zip(new_centroids.iter())
                .map(|(old, new)| euclidean_sq(old, new).sqrt())
                .fold(0.0_f32, f32::max);

            assignments = new_assignments;
            centroids = new_centroids;
            inertia = new_inertia;
            n_iter = iter + 1;

            if max_shift <= config.tol {
                converged = true;
                break;
            }
        }

        let result = ClusteringResult {
            assignments,
            centroids,
            inertia,
            n_iter,
            converged,
        };

        let is_better = best.as_ref().is_none_or(|b| result.inertia < b.inertia);
        if is_better {
            best = Some(result);
        }
    }

    best.ok_or(ExpressionClusterError::InvalidK { k: config.k })
}

/// Build detailed cluster descriptors from a clustering result.
///
/// # Errors
///
/// Returns an error if the operation fails.
pub fn describe_clusters(
    result: &ClusteringResult,
    dataset: &ExpressionDataset,
) -> Result<Vec<ExpressionCluster>, ExpressionClusterError> {
    let k = result.centroids.len();
    let mut clusters = Vec::with_capacity(k);
    // Precompute cluster membership once (a single O(n) pass) instead of
    // calling `cluster_members` (an O(n) scan) once per cluster.
    let membership = result.membership_table();

    for (id, (centroid, members)) in result.centroids.iter().zip(membership).enumerate() {
        let mut sum_dist = 0.0_f32;
        let mut max_dist = 0.0_f32;
        for &m in &members {
            let expr = dataset
                .get(m)
                .ok_or(ExpressionClusterError::EmptyExpressions)?;
            let d = euclidean_sq(expr, centroid).sqrt();
            sum_dist += d;
            if d > max_dist {
                max_dist = d;
            }
        }
        let mean_distance = if members.is_empty() {
            0.0
        } else {
            sum_dist / members.len() as f32
        };

        clusters.push(ExpressionCluster {
            id,
            centroid: centroid.clone(),
            members,
            mean_distance,
            max_distance: max_dist,
            label: None,
        });
    }

    Ok(clusters)
}

/// Find the `k_per_cluster` expressions closest to their cluster centroid.
///
/// Returns a `Vec` (indexed by cluster id) of `Vec<usize>` (indices into the
/// dataset).  If a cluster has fewer than `k_per_cluster` members, all
/// members are returned.
///
/// # Errors
///
/// Returns an error if the operation fails.
pub fn cluster_prototypes(
    result: &ClusteringResult,
    dataset: &ExpressionDataset,
    k_per_cluster: usize,
) -> Result<Vec<Vec<usize>>, ExpressionClusterError> {
    let k = result.centroids.len();
    let mut out = Vec::with_capacity(k);

    for id in 0..k {
        let centroid = &result.centroids[id];
        let mut member_dists: Vec<(usize, f32)> = result
            .cluster_members(id)
            .into_iter()
            .map(|m| {
                let d = dataset
                    .get(m)
                    .map_or(f32::INFINITY, |expr| euclidean_sq(expr, centroid).sqrt());
                (m, d)
            })
            .collect();

        // Sort ascending by distance — closest first.
        member_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let take = k_per_cluster.min(member_dists.len());
        out.push(member_dists[..take].iter().map(|&(idx, _)| idx).collect());
    }

    Ok(out)
}

/// Compute the full N×N pairwise Euclidean distance matrix.
///
/// Returns a flat, row-major `Vec<f32>` of length N².  The matrix is
/// symmetric with zeros on the diagonal.
///
/// # Errors
///
/// Returns an error if the operation fails.
pub fn pairwise_distance_matrix_expr(
    dataset: &ExpressionDataset,
) -> Result<Vec<f32>, ExpressionClusterError> {
    let n = dataset.n_samples();
    if n == 0 {
        return Err(ExpressionClusterError::EmptyExpressions);
    }
    let mut mat = vec![0.0_f32; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d = euclidean_sq(&dataset.expressions[i], &dataset.expressions[j]).sqrt();
            mat[i * n + j] = d;
            mat[j * n + i] = d;
        }
    }
    Ok(mat)
}

/// Run K-means for `k = 1..=max_k` and return `(k, inertia)` pairs.
///
/// Useful for the "elbow method" to choose an appropriate k.
///
/// # Errors
///
/// Returns an error if the operation fails.
pub fn elbow_analysis(
    dataset: &ExpressionDataset,
    max_k: usize,
    config_template: &KMeansConfig,
) -> Result<Vec<(usize, f32)>, ExpressionClusterError> {
    let n = dataset.n_samples();
    if n == 0 {
        return Err(ExpressionClusterError::EmptyExpressions);
    }
    let effective_max = max_k.min(n);
    let mut results = Vec::with_capacity(effective_max);
    for k in 1..=effective_max {
        let cfg = KMeansConfig {
            k,
            seed: config_template.seed.wrapping_add(k as u64),
            max_iter: config_template.max_iter,
            n_restarts: config_template.n_restarts,
            tol: config_template.tol,
        };
        let r = kmeans(dataset, &cfg)?;
        results.push((k, r.inertia));
    }
    Ok(results)
}

/// Davies–Bouldin index: mean of per-cluster "worst-case similarity" ratios.
///
/// A lower value indicates better-separated clusters.  For k == 1 the index
/// is 0.0 by convention (no inter-cluster comparisons exist).
///
/// # Errors
///
/// Returns an error if the operation fails.
pub fn davies_bouldin_index(
    result: &ClusteringResult,
    dataset: &ExpressionDataset,
) -> Result<f32, ExpressionClusterError> {
    let k = result.centroids.len();
    if k <= 1 {
        return Ok(0.0);
    }

    // Precompute cluster membership once (a single O(n) pass) instead of
    // calling `cluster_members` (an O(n) scan) once per cluster.
    let membership = result.membership_table();

    // Average intra-cluster scatter per cluster (mean distance to centroid).
    let mut scatter = vec![0.0_f32; k];
    for (cluster_id, scatter_val) in scatter.iter_mut().enumerate().take(k) {
        let members = &membership[cluster_id];
        if members.is_empty() {
            // Degenerate; treat as 0 scatter.
            continue;
        }
        let centroid = &result.centroids[cluster_id];
        let sum: f32 = members
            .iter()
            .map(|&m| {
                let expr = &dataset.expressions[m];
                euclidean_sq(expr, centroid).sqrt()
            })
            .sum();
        *scatter_val = sum / members.len() as f32;
    }

    // DB index = (1/k) * sum_i max_{j≠i} (scatter_i + scatter_j) / d(c_i, c_j)
    let mut db_sum = 0.0_f32;
    for i in 0..k {
        let mut max_ratio = 0.0_f32;
        for j in 0..k {
            if i == j {
                continue;
            }
            let centroid_dist = euclidean_sq(&result.centroids[i], &result.centroids[j]).sqrt();
            if centroid_dist < 1e-12 {
                // Coincident centroids — contribute infinite ratio (very bad).
                max_ratio = f32::MAX;
                break;
            }
            let ratio = (scatter[i] + scatter[j]) / centroid_dist;
            if ratio > max_ratio {
                max_ratio = ratio;
            }
        }
        db_sum += max_ratio;
    }
    Ok(db_sum / k as f32)
}

/// Assign `expression` to the nearest centroid by L2 distance.
///
/// Returns the zero-based cluster id.
///
/// # Errors
///
/// Returns an error if the operation fails.
pub fn assign_to_cluster(
    expression: &[f32],
    centroids: &[Vec<f32>],
) -> Result<usize, ExpressionClusterError> {
    if centroids.is_empty() {
        return Err(ExpressionClusterError::EmptyExpressions);
    }
    let mut best_id = 0;
    let mut best_d = f32::INFINITY;
    for (id, c) in centroids.iter().enumerate() {
        if c.len() != expression.len() {
            return Err(ExpressionClusterError::DimensionMismatch {
                expected: expression.len(),
                actual: c.len(),
            });
        }
        let d = euclidean_sq(expression, c);
        if d < best_d {
            best_d = d;
            best_id = id;
        }
    }
    Ok(best_id)
}

/// Produce a linearly interpolated path of `n_steps` expressions between
/// `start` and `end` (inclusive on both ends).
///
/// When `n_steps == 1`, only the start is returned.
///
/// # Errors
///
/// Returns an error if the operation fails.
pub fn cluster_path(
    start: &[f32],
    end: &[f32],
    n_steps: usize,
) -> Result<Vec<Vec<f32>>, ExpressionClusterError> {
    if start.len() != end.len() {
        return Err(ExpressionClusterError::DimensionMismatch {
            expected: start.len(),
            actual: end.len(),
        });
    }
    if n_steps == 0 {
        return Ok(Vec::new());
    }
    if n_steps == 1 {
        return Ok(vec![start.to_vec()]);
    }
    let steps = n_steps - 1;
    (0..n_steps)
        .map(|step| {
            let t = step as f32 / steps as f32;
            Ok(start
                .iter()
                .zip(end.iter())
                .map(|(&s, &e)| s + t * (e - s))
                .collect())
        })
        .collect()
}

/// Compute a comprehensive summary of clustering quality.
///
/// # Errors
///
/// Returns an error if the operation fails.
pub fn compute_cluster_stats(
    result: &ClusteringResult,
    dataset: &ExpressionDataset,
) -> Result<ClusterStats, ExpressionClusterError> {
    let k = result.centroids.len();
    // A single O(n) pass instead of `k` separate O(n) `cluster_size` scans.
    let cluster_sizes: Vec<usize> = result.membership_table().iter().map(Vec::len).collect();

    let silhouette = result.silhouette_score(dataset)?;
    let davies_bouldin = davies_bouldin_index(result, dataset)?;

    // Mean cluster radius = mean of per-cluster max distances.
    let clusters = describe_clusters(result, dataset)?;
    let mean_cluster_radius = if clusters.is_empty() {
        0.0
    } else {
        clusters.iter().map(|c| c.max_distance).sum::<f32>() / clusters.len() as f32
    };

    Ok(ClusterStats {
        k,
        total_inertia: result.inertia,
        silhouette,
        davies_bouldin,
        cluster_sizes,
        mean_cluster_radius,
    })
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: build a simple dataset where the first half is near [1, 0] and
    // the second half is near [0, 1] (clearly two clusters in 2-D space).
    fn two_cluster_dataset() -> ExpressionDataset {
        let mut exprs = Vec::new();
        // Cluster A: around (10.0, 0.0)
        for i in 0..5_i32 {
            exprs.push(vec![10.0 + i as f32 * 0.01, 0.0]);
        }
        // Cluster B: around (0.0, 10.0)
        for i in 0..5_i32 {
            exprs.push(vec![0.0, 10.0 + i as f32 * 0.01]);
        }
        ExpressionDataset::new(exprs).expect("two_cluster_dataset")
    }

    fn single_cluster_dataset() -> ExpressionDataset {
        let exprs = vec![vec![1.0, 2.0], vec![1.1, 1.9], vec![0.9, 2.1]];
        ExpressionDataset::new(exprs).expect("single_cluster_dataset")
    }

    // -----------------------------------------------------------------------
    // ExpressionDataset
    // -----------------------------------------------------------------------

    #[test]
    fn test_dataset_new_valid() {
        let ds = ExpressionDataset::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        assert!(ds.is_ok());
        let ds = ds.expect("valid dataset");
        assert_eq!(ds.dim, 2);
        assert_eq!(ds.n_samples(), 2);
    }

    #[test]
    fn test_dataset_new_empty_error() {
        let result: Result<ExpressionDataset, ExpressionClusterError> =
            ExpressionDataset::new(vec![]);
        assert!(matches!(
            result,
            Err(ExpressionClusterError::EmptyExpressions)
        ));
    }

    #[test]
    fn test_dataset_new_dimension_mismatch() {
        let result = ExpressionDataset::new(vec![vec![1.0, 2.0], vec![3.0]]);
        assert!(matches!(
            result,
            Err(ExpressionClusterError::DimensionMismatch {
                expected: 2,
                actual: 1
            })
        ));
    }

    #[test]
    fn test_dataset_with_labels_ok() {
        let ds = ExpressionDataset::new(vec![vec![1.0], vec![2.0]])
            .expect("dataset")
            .with_labels(vec!["a".to_string(), "b".to_string()]);
        assert!(ds.is_ok());
        let ds = ds.expect("with_labels");
        assert_eq!(ds.label(0), Some("a"));
        assert_eq!(ds.label(1), Some("b"));
    }

    #[test]
    fn test_dataset_with_labels_wrong_count() {
        let result = ExpressionDataset::new(vec![vec![1.0], vec![2.0]])
            .expect("dataset")
            .with_labels(vec!["only_one".to_string()]);
        assert!(matches!(
            result,
            Err(ExpressionClusterError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_dataset_get_and_label() {
        let ds = ExpressionDataset::new(vec![vec![1.0, 2.0]])
            .expect("dataset")
            .with_labels(vec!["expr0".to_string()])
            .expect("labels");
        assert_eq!(ds.get(0), Some(&vec![1.0_f32, 2.0_f32]));
        assert_eq!(ds.get(99), None);
        assert_eq!(ds.label(0), Some("expr0"));
        assert_eq!(ds.label(1), None);
    }

    #[test]
    fn test_dataset_mean() {
        let ds = ExpressionDataset::new(vec![vec![1.0, 3.0], vec![3.0, 1.0]]).expect("ds");
        let mean = ds.mean();
        assert!((mean[0] - 2.0).abs() < 1e-6, "mean[0]={}", mean[0]);
        assert!((mean[1] - 2.0).abs() < 1e-6, "mean[1]={}", mean[1]);
    }

    #[test]
    fn test_dataset_std() {
        // Two identical vectors → std = 0.
        let ds = ExpressionDataset::new(vec![vec![5.0, 5.0], vec![5.0, 5.0]]).expect("ds");
        let std = ds.std();
        assert!(std.iter().all(|&s| s.abs() < 1e-6));
    }

    #[test]
    fn test_dataset_normalized() {
        let ds = ExpressionDataset::new(vec![vec![0.0, 2.0], vec![2.0, 4.0]]).expect("ds");
        let normed = ds.normalized().expect("normalized");
        // After normalisation mean should be ~0 and std ~1.
        let m = normed.mean();
        for &v in &m {
            assert!(v.abs() < 1e-5, "mean not zero after normalisation: {v}");
        }
    }

    #[test]
    fn test_dataset_normalized_zero_std() {
        // All same value in one dimension → std=0, result should be 0 not NaN.
        let ds = ExpressionDataset::new(vec![vec![3.0, 1.0], vec![3.0, 2.0]]).expect("ds");
        let normed = ds.normalized().expect("normalized");
        for expr in &normed.expressions {
            for &v in expr {
                assert!(
                    !v.is_nan() && !v.is_infinite(),
                    "NaN/inf in normalised output"
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // KMeansConfig
    // -----------------------------------------------------------------------

    #[test]
    fn test_config_validate_k_zero() {
        let cfg = KMeansConfig::new(0);
        assert!(matches!(
            cfg.validate(),
            Err(ExpressionClusterError::InvalidConfig(_))
        ));
    }

    #[test]
    fn test_config_validate_max_iter_zero() {
        let mut cfg = KMeansConfig::new(2);
        cfg.max_iter = 0;
        assert!(matches!(
            cfg.validate(),
            Err(ExpressionClusterError::InvalidConfig(_))
        ));
    }

    #[test]
    fn test_config_validate_ok() {
        let cfg = KMeansConfig::new(3);
        assert!(cfg.validate().is_ok());
    }

    // -----------------------------------------------------------------------
    // expression_distance
    // -----------------------------------------------------------------------

    #[test]
    fn test_expression_distance_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let d = expression_distance(&a, &a).expect("distance");
        assert!(d.abs() < 1e-6, "identical → 0, got {d}");
    }

    #[test]
    fn test_expression_distance_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let d = expression_distance(&a, &b).expect("distance");
        assert!((d - std::f32::consts::SQRT_2).abs() < 1e-5, "got {d}");
    }

    #[test]
    fn test_expression_distance_length_mismatch() {
        let result = expression_distance(&[1.0, 2.0], &[1.0]);
        assert!(matches!(
            result,
            Err(ExpressionClusterError::DimensionMismatch { .. })
        ));
    }

    // -----------------------------------------------------------------------
    // expression_cosine_similarity
    // -----------------------------------------------------------------------

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let sim = expression_cosine_similarity(&a, &a).expect("cosine");
        assert!((sim - 1.0).abs() < 1e-6, "got {sim}");
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = expression_cosine_similarity(&a, &b).expect("cosine");
        assert!(sim.abs() < 1e-6, "got {sim}");
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        let sim = expression_cosine_similarity(&a, &b).expect("cosine");
        assert!((sim + 1.0).abs() < 1e-6, "got {sim}");
    }

    // -----------------------------------------------------------------------
    // kmeans_plus_plus_init
    // -----------------------------------------------------------------------

    #[test]
    fn test_kmeans_pp_returns_k_centroids() {
        let ds = two_cluster_dataset();
        let centroids = kmeans_plus_plus_init(&ds.expressions, 3, 42).expect("kpp init");
        assert_eq!(centroids.len(), 3);
    }

    #[test]
    fn test_kmeans_pp_k_equals_1() {
        let ds = single_cluster_dataset();
        let centroids = kmeans_plus_plus_init(&ds.expressions, 1, 7).expect("kpp k=1");
        assert_eq!(centroids.len(), 1);
    }

    #[test]
    fn test_kmeans_pp_k_equals_n() {
        let exprs = vec![vec![1.0], vec![2.0], vec![3.0]];
        let centroids = kmeans_plus_plus_init(&exprs, 3, 99).expect("kpp k=n");
        assert_eq!(centroids.len(), 3);
    }

    #[test]
    fn test_kmeans_pp_identical_points_still_returns_k_centroids() {
        // Regression test: when every point coincides (a degenerate
        // dataset), the fallback branch must still push a centroid on
        // every iteration so the result always has exactly `k` entries,
        // rather than silently returning fewer.
        let exprs = vec![vec![1.0, 1.0]; 5];
        let centroids = kmeans_plus_plus_init(&exprs, 3, 123).expect("kpp degenerate");
        assert_eq!(
            centroids.len(),
            3,
            "must return exactly k=3 centroids even for identical points"
        );
        for c in &centroids {
            assert_eq!(c, &vec![1.0, 1.0]);
        }
    }

    // -----------------------------------------------------------------------
    // kmeans_iteration
    // -----------------------------------------------------------------------

    #[test]
    fn test_kmeans_iteration_single_cluster() {
        let exprs = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let centroids = vec![vec![2.0, 3.0]];
        let (assignments, new_centroids, _inertia) =
            kmeans_iteration(&exprs, &centroids).expect("iter");
        assert_eq!(assignments, vec![0, 0]);
        assert_eq!(new_centroids.len(), 1);
    }

    #[test]
    fn test_kmeans_iteration_all_same_position() {
        // When all points coincide the centroid should equal them.
        let exprs = vec![vec![5.0, 5.0]; 4];
        let centroids = vec![vec![5.0, 5.0]];
        let (_, new_centroids, inertia) = kmeans_iteration(&exprs, &centroids).expect("iter");
        assert!(inertia.abs() < 1e-6, "inertia={inertia}");
        assert!((new_centroids[0][0] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_kmeans_iteration_empty_centroids_errors() {
        // Regression test: an empty centroid slice previously panicked
        // (index out of bounds) instead of returning an error.
        let exprs = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let centroids: Vec<Vec<f32>> = Vec::new();
        let result = kmeans_iteration(&exprs, &centroids);
        assert!(matches!(
            result,
            Err(ExpressionClusterError::InvalidK { k: 0 })
        ));
    }

    // -----------------------------------------------------------------------
    // kmeans
    // -----------------------------------------------------------------------

    #[test]
    fn test_kmeans_k1_inertia_from_centroid() {
        let ds = single_cluster_dataset();
        let cfg = KMeansConfig::new(1);
        let result = kmeans(&ds, &cfg).expect("kmeans k=1");
        // All assigned to cluster 0.
        assert!(result.assignments.iter().all(|&a| a == 0));
        // Inertia should be the sum of squared distances from the (single) centroid.
        assert!(result.inertia >= 0.0);
    }

    #[test]
    fn test_kmeans_k2_clear_separation() {
        let ds = two_cluster_dataset();
        let cfg = KMeansConfig {
            k: 2,
            seed: 1,
            n_restarts: 5,
            ..KMeansConfig::default()
        };
        let result = kmeans(&ds, &cfg).expect("kmeans k=2");
        // The first 5 samples should all be in one cluster, the last 5 in the other.
        let a = result.assignments[0];
        let b = result.assignments[5];
        assert_ne!(a, b, "Two distinct clusters expected");
        for i in 0..5 {
            assert_eq!(
                result.assignments[i], a,
                "Sample {i} should be in cluster A"
            );
        }
        for i in 5..10 {
            assert_eq!(
                result.assignments[i], b,
                "Sample {i} should be in cluster B"
            );
        }
    }

    // -----------------------------------------------------------------------
    // ClusteringResult helpers
    // -----------------------------------------------------------------------

    #[test]
    fn test_cluster_members_and_size() {
        let assignments = vec![0, 1, 0, 0, 1];
        let result = ClusteringResult {
            assignments,
            centroids: vec![vec![0.0], vec![1.0]],
            inertia: 0.0,
            n_iter: 1,
            converged: true,
        };
        assert_eq!(result.cluster_members(0), vec![0, 2, 3]);
        assert_eq!(result.cluster_members(1), vec![1, 4]);
        assert_eq!(result.cluster_size(0), 3);
        assert_eq!(result.cluster_size(1), 2);
    }

    #[test]
    fn test_membership_table_matches_cluster_members() {
        // `membership_table` is the shared O(n) building block behind
        // `silhouette_score`, `davies_bouldin_index`, `describe_clusters`
        // and `compute_cluster_stats`; it must agree exactly with the
        // (slower, per-cluster) `cluster_members`/`cluster_size`.
        let result = ClusteringResult {
            assignments: vec![0, 1, 0, 0, 1, 2],
            centroids: vec![vec![0.0], vec![1.0], vec![2.0]],
            inertia: 0.0,
            n_iter: 1,
            converged: true,
        };
        let table = result.membership_table();
        assert_eq!(table.len(), 3);
        for (id, members) in table.iter().enumerate() {
            assert_eq!(members, &result.cluster_members(id), "cluster {id}");
            assert_eq!(members.len(), result.cluster_size(id), "cluster {id}");
        }
    }

    #[test]
    fn test_membership_table_drops_out_of_range_assignments() {
        // `assignments` and `centroids` are independently public fields, so
        // an assignment >= centroids.len() is a shape the type permits;
        // `membership_table` must drop it rather than panicking.
        let result = ClusteringResult {
            assignments: vec![0, 99, 1],
            centroids: vec![vec![0.0], vec![1.0]],
            inertia: 0.0,
            n_iter: 1,
            converged: true,
        };
        let table = result.membership_table();
        assert_eq!(table.len(), 2);
        assert_eq!(table[0], vec![0]);
        assert_eq!(table[1], vec![2]);
    }

    #[test]
    fn test_silhouette_score_k1() {
        let ds = single_cluster_dataset();
        let result = ClusteringResult {
            assignments: vec![0, 0, 0],
            centroids: vec![vec![1.0, 2.0]],
            inertia: 0.1,
            n_iter: 1,
            converged: true,
        };
        let score = result.silhouette_score(&ds).expect("silhouette k=1");
        assert!(
            (score - 0.0).abs() < 1e-6,
            "k=1 silhouette should be 0.0, got {score}"
        );
    }

    #[test]
    fn test_silhouette_score_well_separated() {
        let ds = two_cluster_dataset();
        let cfg = KMeansConfig {
            k: 2,
            seed: 1,
            n_restarts: 5,
            ..KMeansConfig::default()
        };
        let result = kmeans(&ds, &cfg).expect("kmeans");
        let score = result.silhouette_score(&ds).expect("silhouette");
        // Well-separated clusters → high silhouette.
        assert!(score > 0.5, "expected silhouette > 0.5, got {score}");
    }

    // -----------------------------------------------------------------------
    // describe_clusters
    // -----------------------------------------------------------------------

    #[test]
    fn test_describe_clusters_member_count() {
        let ds = two_cluster_dataset();
        let cfg = KMeansConfig {
            k: 2,
            seed: 1,
            n_restarts: 5,
            ..KMeansConfig::default()
        };
        let result = kmeans(&ds, &cfg).expect("kmeans");
        let clusters = describe_clusters(&result, &ds).expect("describe");
        let total: usize = clusters.iter().map(|c| c.members.len()).sum();
        assert_eq!(total, ds.n_samples());
    }

    #[test]
    fn test_describe_clusters_mean_distance_non_negative() {
        let ds = two_cluster_dataset();
        let cfg = KMeansConfig {
            k: 2,
            seed: 1,
            n_restarts: 5,
            ..KMeansConfig::default()
        };
        let result = kmeans(&ds, &cfg).expect("kmeans");
        let clusters = describe_clusters(&result, &ds).expect("describe");
        for c in &clusters {
            assert!(c.mean_distance >= 0.0);
            assert!(c.max_distance >= c.mean_distance - 1e-5);
        }
    }

    // -----------------------------------------------------------------------
    // cluster_prototypes
    // -----------------------------------------------------------------------

    #[test]
    fn test_cluster_prototypes_count() {
        let ds = two_cluster_dataset();
        let cfg = KMeansConfig {
            k: 2,
            seed: 1,
            n_restarts: 5,
            ..KMeansConfig::default()
        };
        let result = kmeans(&ds, &cfg).expect("kmeans");
        let protos = cluster_prototypes(&result, &ds, 2).expect("prototypes");
        assert_eq!(protos.len(), 2);
        for proto_list in &protos {
            assert!(proto_list.len() <= 2);
        }
    }

    #[test]
    fn test_cluster_prototypes_k_per_cluster_gt_size() {
        let ds = single_cluster_dataset();
        let cfg = KMeansConfig::new(1);
        let result = kmeans(&ds, &cfg).expect("kmeans");
        // Requesting more prototypes than samples should just return all.
        let protos = cluster_prototypes(&result, &ds, 100).expect("prototypes");
        assert_eq!(protos[0].len(), ds.n_samples());
    }

    // -----------------------------------------------------------------------
    // pairwise_distance_matrix_expr
    // -----------------------------------------------------------------------

    #[test]
    fn test_pairwise_matrix_diagonal_zero() {
        let ds = two_cluster_dataset();
        let mat = pairwise_distance_matrix_expr(&ds).expect("matrix");
        let n = ds.n_samples();
        for i in 0..n {
            assert!(
                mat[i * n + i].abs() < 1e-6,
                "diagonal not zero at ({}, {}): {}",
                i,
                i,
                mat[i * n + i]
            );
        }
    }

    #[test]
    fn test_pairwise_matrix_symmetric() {
        let ds = two_cluster_dataset();
        let mat = pairwise_distance_matrix_expr(&ds).expect("matrix");
        let n = ds.n_samples();
        for i in 0..n {
            for j in 0..n {
                let diff = (mat[i * n + j] - mat[j * n + i]).abs();
                assert!(diff < 1e-6, "not symmetric at ({i}, {j}): diff={diff}");
            }
        }
    }

    // -----------------------------------------------------------------------
    // elbow_analysis
    // -----------------------------------------------------------------------

    #[test]
    fn test_elbow_analysis_inertia_decreases() {
        let ds = two_cluster_dataset();
        let template = KMeansConfig {
            n_restarts: 2,
            ..KMeansConfig::default()
        };
        let elbow = elbow_analysis(&ds, 4, &template).expect("elbow");
        assert_eq!(elbow.len(), 4);
        // Inertia at k=1 should be >= inertia at k=2.
        let inertia_k1 = elbow[0].1;
        let inertia_k2 = elbow[1].1;
        assert!(
            inertia_k1 >= inertia_k2 - 1e-4,
            "inertia should not increase: k1={inertia_k1}, k2={inertia_k2}"
        );
    }

    // -----------------------------------------------------------------------
    // davies_bouldin_index
    // -----------------------------------------------------------------------

    #[test]
    fn test_davies_bouldin_k1_is_zero() {
        let ds = single_cluster_dataset();
        let cfg = KMeansConfig::new(1);
        let result = kmeans(&ds, &cfg).expect("kmeans");
        let db = davies_bouldin_index(&result, &ds).expect("db");
        assert!((db - 0.0).abs() < 1e-6, "k=1 → 0.0, got {db}");
    }

    #[test]
    fn test_davies_bouldin_two_clusters() {
        let ds = two_cluster_dataset();
        let cfg = KMeansConfig {
            k: 2,
            seed: 1,
            n_restarts: 5,
            ..KMeansConfig::default()
        };
        let result = kmeans(&ds, &cfg).expect("kmeans");
        let db = davies_bouldin_index(&result, &ds).expect("db");
        // Well separated → low DB index.
        assert!(db >= 0.0, "DB index must be non-negative, got {db}");
        assert!(
            db < 1.0,
            "Well separated clusters should have DB < 1.0, got {db}"
        );
    }

    // -----------------------------------------------------------------------
    // assign_to_cluster
    // -----------------------------------------------------------------------

    #[test]
    fn test_assign_to_cluster_closest() {
        let centroids = vec![vec![0.0_f32, 0.0], vec![10.0, 0.0]];
        let expr = vec![9.5_f32, 0.0];
        let id = assign_to_cluster(&expr, &centroids).expect("assign");
        assert_eq!(id, 1, "should assign to centroid at [10, 0], got {id}");
    }

    #[test]
    fn test_assign_to_cluster_exact_match() {
        let centroids = vec![vec![1.0_f32], vec![2.0]];
        let id = assign_to_cluster(&[2.0_f32], &centroids).expect("assign");
        assert_eq!(id, 1);
    }

    // -----------------------------------------------------------------------
    // cluster_path
    // -----------------------------------------------------------------------

    #[test]
    fn test_cluster_path_n_steps() {
        let start = vec![0.0_f32, 0.0];
        let end = vec![1.0_f32, 1.0];
        let path = cluster_path(&start, &end, 5).expect("path");
        assert_eq!(path.len(), 5);
    }

    #[test]
    fn test_cluster_path_endpoints() {
        let start = vec![0.0_f32, 0.0];
        let end = vec![4.0_f32, 0.0];
        let path = cluster_path(&start, &end, 5).expect("path");
        assert_eq!(path.len(), 5);
        // First element == start.
        assert!((path[0][0] - 0.0).abs() < 1e-6);
        // Last element == end.
        assert!((path[4][0] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_cluster_path_single_step() {
        let start = vec![1.0_f32, 2.0];
        let end = vec![5.0_f32, 6.0];
        let path = cluster_path(&start, &end, 1).expect("path");
        assert_eq!(path.len(), 1);
        assert_eq!(path[0], start);
    }

    #[test]
    fn test_cluster_path_zero_steps() {
        let path = cluster_path(&[1.0_f32], &[2.0_f32], 0).expect("path");
        assert!(path.is_empty());
    }

    #[test]
    fn test_cluster_path_dimension_mismatch() {
        let result = cluster_path(&[1.0_f32, 2.0], &[3.0_f32], 3);
        assert!(matches!(
            result,
            Err(ExpressionClusterError::DimensionMismatch { .. })
        ));
    }

    // -----------------------------------------------------------------------
    // compute_cluster_stats
    // -----------------------------------------------------------------------

    #[test]
    fn test_compute_cluster_stats_valid() {
        let ds = two_cluster_dataset();
        let cfg = KMeansConfig {
            k: 2,
            seed: 1,
            n_restarts: 5,
            ..KMeansConfig::default()
        };
        let result = kmeans(&ds, &cfg).expect("kmeans");
        let stats = compute_cluster_stats(&result, &ds).expect("stats");
        assert_eq!(stats.k, 2);
        assert_eq!(stats.cluster_sizes.len(), 2);
        assert_eq!(stats.cluster_sizes.iter().sum::<usize>(), ds.n_samples());
        assert!(stats.total_inertia >= 0.0);
        assert!(stats.mean_cluster_radius >= 0.0);
    }

    #[test]
    fn test_compute_cluster_stats_k1() {
        let ds = single_cluster_dataset();
        let cfg = KMeansConfig::new(1);
        let result = kmeans(&ds, &cfg).expect("kmeans");
        let stats = compute_cluster_stats(&result, &ds).expect("stats");
        assert_eq!(stats.k, 1);
        assert!((stats.silhouette - 0.0).abs() < 1e-6);
        assert!((stats.davies_bouldin - 0.0).abs() < 1e-6);
    }
}
