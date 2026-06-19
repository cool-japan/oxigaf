//! Gradient Surgery for Multi-Task Learning (PCGrad).
//!
//! Implements the PCGrad algorithm from Yu et al., "Gradient Surgery for
//! Multi-Task Learning" (NeurIPS 2020). When gradients from two loss
//! components conflict (negative dot product), one is projected to be
//! orthogonal to the other, reducing destructive interference.
//!
//! # Quick start
//! ```rust
//! use oxigaf_trainer::gradient_surgery::{TaskGradient, pcgrad};
//!
//! let g1 = TaskGradient::new("photometric", vec![1.0, 0.0], 1.0).unwrap();
//! let g2 = TaskGradient::new("regularization", vec![0.0, 1.0], 1.0).unwrap();
//! let combined = pcgrad(&[g1, g2], 0).unwrap();
//! assert_eq!(combined.len(), 2);
//! ```

use std::collections::HashMap;
use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// PRNG (xorshift64, no rand crate)
// ─────────────────────────────────────────────────────────────────────────────

fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    if x == 0 {
        x = 0x123456789ABCDEF0;
    }
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

// ─────────────────────────────────────────────────────────────────────────────
// GradientSurgeryError
// ─────────────────────────────────────────────────────────────────────────────

/// Errors produced by the gradient surgery subsystem.
#[derive(Debug, Error)]
pub enum GradientSurgeryError {
    #[error("Gradient lengths differ: task {task_a} has {len_a}, task {task_b} has {len_b}")]
    LengthMismatch {
        task_a: usize,
        task_b: usize,
        len_a: usize,
        len_b: usize,
    },

    #[error("No gradients provided")]
    EmptyGradients,

    #[error("Invalid weight for task {task}: {weight} (must be positive)")]
    InvalidWeight { task: usize, weight: f32 },

    #[error("Gradient is zero vector (cannot project onto it)")]
    ZeroGradient,
}

// ─────────────────────────────────────────────────────────────────────────────
// TaskGradient
// ─────────────────────────────────────────────────────────────────────────────

/// A named gradient from one loss component.
#[derive(Debug, Clone)]
pub struct TaskGradient {
    /// Human-readable name of the loss component.
    pub name: String,
    /// Flat parameter gradient vector.
    pub gradient: Vec<f32>,
    /// Scalar weight for this loss component.
    pub weight: f32,
}

impl TaskGradient {
    /// Create a new `TaskGradient`.
    ///
    /// Returns [`GradientSurgeryError::InvalidWeight`] if `weight <= 0`.
    pub fn new(
        name: impl Into<String>,
        gradient: Vec<f32>,
        weight: f32,
    ) -> Result<Self, GradientSurgeryError> {
        if weight <= 0.0 {
            return Err(GradientSurgeryError::InvalidWeight { task: 0, weight });
        }
        Ok(Self {
            name: name.into(),
            gradient,
            weight,
        })
    }

    /// L2 norm of the gradient.
    pub fn l2_norm(&self) -> f32 {
        self.gradient.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    /// Dot product with another `TaskGradient`.
    ///
    /// Returns [`GradientSurgeryError::LengthMismatch`] if lengths differ.
    pub fn dot(&self, other: &TaskGradient) -> Result<f32, GradientSurgeryError> {
        if self.gradient.len() != other.gradient.len() {
            return Err(GradientSurgeryError::LengthMismatch {
                task_a: 0,
                task_b: 1,
                len_a: self.gradient.len(),
                len_b: other.gradient.len(),
            });
        }
        let d = self
            .gradient
            .iter()
            .zip(other.gradient.iter())
            .map(|(a, b)| a * b)
            .sum();
        Ok(d)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Core PCGrad functions
// ─────────────────────────────────────────────────────────────────────────────

/// Detect whether two gradients conflict (dot product < 0).
///
/// Returns [`GradientSurgeryError::LengthMismatch`] if lengths differ.
pub fn gradients_conflict(g_i: &[f32], g_j: &[f32]) -> Result<bool, GradientSurgeryError> {
    if g_i.len() != g_j.len() {
        return Err(GradientSurgeryError::LengthMismatch {
            task_a: 0,
            task_b: 1,
            len_a: g_i.len(),
            len_b: g_j.len(),
        });
    }
    let dot: f32 = g_i.iter().zip(g_j.iter()).map(|(a, b)| a * b).sum();
    Ok(dot < 0.0)
}

/// Project `g_i` to remove components conflicting with `g_j`.
///
/// If `dot(g_i, g_j) < 0`:
/// ```text
///   g_i_proj = g_i - (dot(g_i, g_j) / |g_j|²) * g_j
/// ```
/// Otherwise `g_i` is returned unchanged.
///
/// If `|g_j|² < 1e-12` (zero vector), `g_i` is returned unchanged.
///
/// Returns [`GradientSurgeryError::LengthMismatch`] if lengths differ.
pub fn project_gradient(g_i: &[f32], g_j: &[f32]) -> Result<Vec<f32>, GradientSurgeryError> {
    if g_i.len() != g_j.len() {
        return Err(GradientSurgeryError::LengthMismatch {
            task_a: 0,
            task_b: 1,
            len_a: g_i.len(),
            len_b: g_j.len(),
        });
    }

    let g_j_sq: f32 = g_j.iter().map(|x| x * x).sum();
    if g_j_sq < 1e-12 {
        // g_j is effectively zero; no projection possible.
        return Ok(g_i.to_vec());
    }

    let dot_ij: f32 = g_i.iter().zip(g_j.iter()).map(|(a, b)| a * b).sum();

    if dot_ij >= 0.0 {
        // Gradients are not conflicting; return g_i unchanged.
        return Ok(g_i.to_vec());
    }

    // Subtract the conflicting component.
    let scale = dot_ij / g_j_sq;
    let projected: Vec<f32> = g_i
        .iter()
        .zip(g_j.iter())
        .map(|(a, b)| a - scale * b)
        .collect();

    Ok(projected)
}

/// Apply PCGrad to a set of task gradients.
///
/// For each task `i`:
/// - Build a list of all other task indices, shuffled using xorshift64
///   seeded by `step XOR (i as u64)` for per-task determinism.
/// - Project `g_i` against each `g_j` in shuffled order, removing
///   conflicting components.
///
/// Returns `Σ_i (weight_i * projected_g_i)`.
///
/// Returns [`GradientSurgeryError::EmptyGradients`] if the slice is empty.
/// Returns [`GradientSurgeryError::LengthMismatch`] if gradient lengths differ.
pub fn pcgrad(
    task_gradients: &[TaskGradient],
    step: u64,
) -> Result<Vec<f32>, GradientSurgeryError> {
    if task_gradients.is_empty() {
        return Err(GradientSurgeryError::EmptyGradients);
    }

    let n = task_gradients.len();
    let dim = task_gradients[0].gradient.len();

    // Validate all lengths match.
    for (j, tg) in task_gradients.iter().enumerate() {
        if tg.gradient.len() != dim {
            return Err(GradientSurgeryError::LengthMismatch {
                task_a: 0,
                task_b: j,
                len_a: dim,
                len_b: tg.gradient.len(),
            });
        }
    }

    let mut result = vec![0.0f32; dim];

    for i in 0..n {
        // Build indices of all other tasks.
        let mut others: Vec<usize> = (0..n).filter(|&j| j != i).collect();

        // Shuffle with xorshift64, incorporating task index for distinct orders.
        let mut rng_state = step ^ (i as u64).wrapping_mul(0x9E3779B97F4A7C15);
        fisher_yates_shuffle(&mut others, &mut rng_state);

        // Iteratively project g_i against each other gradient.
        let mut g_i = task_gradients[i].gradient.clone();
        for j in others {
            let g_j = &task_gradients[j].gradient;
            g_i = project_gradient(&g_i, g_j)?;
        }

        // Accumulate weight_i * projected_g_i.
        let w = task_gradients[i].weight;
        for (r, v) in result.iter_mut().zip(g_i.iter()) {
            *r += w * v;
        }
    }

    Ok(result)
}

/// Fisher-Yates (Knuth) shuffle using xorshift64.
fn fisher_yates_shuffle(indices: &mut [usize], state: &mut u64) {
    let n = indices.len();
    for i in (1..n).rev() {
        let j = (xorshift64(state) as usize) % (i + 1);
        indices.swap(i, j);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Conflict Analysis
// ─────────────────────────────────────────────────────────────────────────────

/// Analysis of gradient conflicts across all pairwise task combinations.
#[derive(Debug, Clone)]
pub struct ConflictReport {
    /// Each entry: (task_i_name, task_j_name, dot_product).
    pub conflicts: Vec<(String, String, f32)>,
    /// Number of conflicting pairs (dot product < 0).
    pub num_conflicts: usize,
    /// Total number of unique pairs.
    pub total_pairs: usize,
    /// Fraction of pairs that conflict: `num_conflicts / total_pairs`.
    pub conflict_fraction: f32,
    /// Mean cosine similarity across all pairs (negative = conflicting on average).
    pub mean_cosine_similarity: f32,
}

/// Analyze all pairwise conflicts between task gradients.
///
/// Returns [`GradientSurgeryError::EmptyGradients`] if the slice is empty.
/// Returns [`GradientSurgeryError::LengthMismatch`] if gradient lengths differ.
pub fn analyze_conflicts(
    task_gradients: &[TaskGradient],
) -> Result<ConflictReport, GradientSurgeryError> {
    if task_gradients.is_empty() {
        return Err(GradientSurgeryError::EmptyGradients);
    }

    let n = task_gradients.len();
    let dim = task_gradients[0].gradient.len();

    for (j, tg) in task_gradients.iter().enumerate() {
        if tg.gradient.len() != dim {
            return Err(GradientSurgeryError::LengthMismatch {
                task_a: 0,
                task_b: j,
                len_a: dim,
                len_b: tg.gradient.len(),
            });
        }
    }

    let mut conflicts = Vec::new();
    let mut num_conflicts = 0usize;
    let mut cosine_sum = 0.0f32;

    for i in 0..n {
        for j in (i + 1)..n {
            let g_i = &task_gradients[i].gradient;
            let g_j = &task_gradients[j].gradient;

            let dot: f32 = g_i.iter().zip(g_j.iter()).map(|(a, b)| a * b).sum();

            let norm_i: f32 = g_i.iter().map(|x| x * x).sum::<f32>().sqrt();
            let norm_j: f32 = g_j.iter().map(|x| x * x).sum::<f32>().sqrt();

            let cosine = if norm_i > 1e-12 && norm_j > 1e-12 {
                dot / (norm_i * norm_j)
            } else {
                0.0
            };

            cosine_sum += cosine;
            conflicts.push((
                task_gradients[i].name.clone(),
                task_gradients[j].name.clone(),
                dot,
            ));

            if dot < 0.0 {
                num_conflicts += 1;
            }
        }
    }

    let total_pairs = conflicts.len();
    let conflict_fraction = if total_pairs == 0 {
        0.0
    } else {
        num_conflicts as f32 / total_pairs as f32
    };
    let mean_cosine_similarity = if total_pairs == 0 {
        0.0
    } else {
        cosine_sum / total_pairs as f32
    };

    Ok(ConflictReport {
        conflicts,
        num_conflicts,
        total_pairs,
        conflict_fraction,
        mean_cosine_similarity,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Aggregation Strategies
// ─────────────────────────────────────────────────────────────────────────────

/// Strategy for combining multiple task gradients into one.
#[derive(Debug, Clone, PartialEq)]
pub enum AggregationStrategy {
    /// Simple weighted sum: `Σ w_i * g_i`.
    WeightedSum,
    /// PCGrad: project conflicting gradients, then sum.
    PCGrad,
    /// GradNorm: normalize each gradient to unit length before weighted sum.
    GradNormalized,
    /// AlignedOnly: keep only aligned components (equivalent to PCGrad).
    AlignedOnly,
}

/// Aggregate gradients using the specified strategy.
///
/// - `WeightedSum`: `Σ (weight_i * g_i)`.
/// - `PCGrad`: delegates to [`pcgrad`].
/// - `GradNormalized`: normalize each `g_i` to unit norm (skip if zero),
///   then compute weighted sum.
/// - `AlignedOnly`: equivalent to PCGrad; delegates to [`pcgrad`].
///
/// Returns [`GradientSurgeryError::EmptyGradients`] if the slice is empty.
pub fn aggregate_gradients(
    task_gradients: &[TaskGradient],
    strategy: AggregationStrategy,
    step: u64,
) -> Result<Vec<f32>, GradientSurgeryError> {
    if task_gradients.is_empty() {
        return Err(GradientSurgeryError::EmptyGradients);
    }

    match strategy {
        AggregationStrategy::WeightedSum => {
            let dim = task_gradients[0].gradient.len();
            let mut result = vec![0.0f32; dim];
            for tg in task_gradients {
                for (r, v) in result.iter_mut().zip(tg.gradient.iter()) {
                    *r += tg.weight * v;
                }
            }
            Ok(result)
        }

        AggregationStrategy::PCGrad | AggregationStrategy::AlignedOnly => {
            // AlignedOnly is equivalent to PCGrad per spec.
            pcgrad(task_gradients, step)
        }

        AggregationStrategy::GradNormalized => {
            let dim = task_gradients[0].gradient.len();
            let mut result = vec![0.0f32; dim];
            for tg in task_gradients {
                let norm: f32 = tg.gradient.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm < 1e-12 {
                    // Zero gradient; skip to avoid division by zero.
                    continue;
                }
                let inv_norm = 1.0 / norm;
                for (r, v) in result.iter_mut().zip(tg.gradient.iter()) {
                    *r += tg.weight * v * inv_norm;
                }
            }
            Ok(result)
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Conflict History Tracking
// ─────────────────────────────────────────────────────────────────────────────

/// Track conflict statistics over training using EMA smoothing.
pub struct ConflictTracker {
    /// EMA conflict fraction per (alphabetically sorted) task pair.
    pair_conflict_ema: HashMap<(String, String), f32>,
    /// EMA smoothing factor.
    ema_alpha: f32,
    /// Current training step.
    pub step: u64,
}

impl ConflictTracker {
    /// Create a new `ConflictTracker` with the given EMA smoothing factor.
    pub fn new(ema_alpha: f32) -> Self {
        Self {
            pair_conflict_ema: HashMap::new(),
            ema_alpha,
            step: 0,
        }
    }

    /// Update EMA conflict statistics from a `ConflictReport`.
    ///
    /// For each pair in the report, the EMA is updated as:
    /// `ema = alpha * conflict + (1 - alpha) * ema`
    /// where `conflict` is 1.0 if dot < 0, else 0.0.
    pub fn update(&mut self, report: &ConflictReport) {
        let alpha = self.ema_alpha;

        for (name_i, name_j, dot) in &report.conflicts {
            // Normalize key order alphabetically.
            let key = sorted_pair_key(name_i, name_j);
            let conflict_val = if *dot < 0.0 { 1.0f32 } else { 0.0f32 };

            let ema = self.pair_conflict_ema.entry(key).or_insert(0.0);
            *ema = alpha * conflict_val + (1.0 - alpha) * *ema;
        }

        self.step += 1;
    }

    /// Return the pair with the highest EMA conflict rate.
    ///
    /// Returns `None` if no pairs have been tracked yet.
    pub fn most_conflicting_pair(&self) -> Option<(&str, &str, f32)> {
        self.pair_conflict_ema
            .iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|((k0, k1), &rate)| (k0.as_str(), k1.as_str(), rate))
    }

    /// Mean EMA conflict rate across all tracked pairs.
    pub fn mean_conflict_rate(&self) -> f32 {
        if self.pair_conflict_ema.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.pair_conflict_ema.values().sum();
        sum / self.pair_conflict_ema.len() as f32
    }

    /// Current training step count.
    pub fn step_count(&self) -> u64 {
        self.step
    }
}

/// Build an alphabetically sorted (name_a, name_b) key for the HashMap.
fn sorted_pair_key(a: &str, b: &str) -> (String, String) {
    if a <= b {
        (a.to_string(), b.to_string())
    } else {
        (b.to_string(), a.to_string())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn grad(values: &[f32]) -> Vec<f32> {
        values.to_vec()
    }

    fn task(name: &str, g: Vec<f32>) -> TaskGradient {
        TaskGradient::new(name, g, 1.0).unwrap()
    }

    // 1. TaskGradient::l2_norm correct for [3,4] → 5.0
    #[test]
    fn test_l2_norm_3_4() {
        let tg = task("a", grad(&[3.0, 4.0]));
        let norm = tg.l2_norm();
        assert!((norm - 5.0).abs() < 1e-5, "expected 5.0, got {norm}");
    }

    // 2. TaskGradient::dot correct for [1,0] · [0,1] → 0.0
    #[test]
    fn test_dot_perpendicular() {
        let a = task("a", grad(&[1.0, 0.0]));
        let b = task("b", grad(&[0.0, 1.0]));
        let d = a.dot(&b).unwrap();
        assert!((d - 0.0).abs() < 1e-6, "expected 0.0, got {d}");
    }

    // 3. TaskGradient::dot length mismatch → LengthMismatch error
    #[test]
    fn test_dot_length_mismatch() {
        let a = task("a", grad(&[1.0, 2.0]));
        let b = task("b", grad(&[1.0]));
        let result = a.dot(&b);
        assert!(
            matches!(result, Err(GradientSurgeryError::LengthMismatch { .. })),
            "expected LengthMismatch"
        );
    }

    // 4. gradients_conflict perpendicular gradients → false (dot = 0)
    #[test]
    fn test_conflict_perpendicular_false() {
        let g_i = grad(&[1.0, 0.0]);
        let g_j = grad(&[0.0, 1.0]);
        let conflict = gradients_conflict(&g_i, &g_j).unwrap();
        assert!(!conflict, "perpendicular gradients should not conflict");
    }

    // 5. gradients_conflict opposing gradients → true (dot < 0)
    #[test]
    fn test_conflict_opposing_true() {
        let g_i = grad(&[1.0, 0.0]);
        let g_j = grad(&[-1.0, 0.0]);
        let conflict = gradients_conflict(&g_i, &g_j).unwrap();
        assert!(conflict, "opposing gradients should conflict");
    }

    // 6. gradients_conflict aligned gradients → false (dot > 0)
    #[test]
    fn test_conflict_aligned_false() {
        let g_i = grad(&[1.0, 1.0]);
        let g_j = grad(&[2.0, 0.5]);
        let conflict = gradients_conflict(&g_i, &g_j).unwrap();
        assert!(!conflict, "aligned gradients should not conflict");
    }

    // 7. project_gradient non-conflicting: same as input
    #[test]
    fn test_project_non_conflicting_unchanged() {
        let g_i = grad(&[1.0, 0.0]);
        let g_j = grad(&[1.0, 0.0]);
        let proj = project_gradient(&g_i, &g_j).unwrap();
        assert!((proj[0] - 1.0).abs() < 1e-5);
        assert!((proj[1] - 0.0).abs() < 1e-5);
    }

    // 8. project_gradient perfectly opposing: result is orthogonal to g_j
    #[test]
    fn test_project_opposing_orthogonal() {
        // g_i at 45 degrees, g_j pointing left — conflicts with x component
        let g_i = vec![1.0f32, 1.0];
        let g_j = vec![-1.0f32, 0.0];
        let proj = project_gradient(&g_i, &g_j).unwrap();
        // dot(g_i, g_j) = -1, |g_j|² = 1
        // proj = [1,1] - (-1/1)*[-1,0] = [1,1] - [1,0] = [0,1]
        assert!(
            (proj[0]).abs() < 1e-4,
            "x component should be ~0, got {}",
            proj[0]
        );
        assert!(
            (proj[1] - 1.0).abs() < 1e-4,
            "y component should be ~1.0, got {}",
            proj[1]
        );
    }

    // 9. project_gradient length mismatch → error
    #[test]
    fn test_project_length_mismatch() {
        let g_i = grad(&[1.0, 2.0]);
        let g_j = grad(&[1.0]);
        let result = project_gradient(&g_i, &g_j);
        assert!(
            matches!(result, Err(GradientSurgeryError::LengthMismatch { .. })),
            "expected LengthMismatch"
        );
    }

    // 10. pcgrad single task → returns that gradient * weight
    #[test]
    fn test_pcgrad_single_task() {
        let tg = TaskGradient::new("a", grad(&[2.0, 3.0]), 0.5).unwrap();
        let result = pcgrad(&[tg], 0).unwrap();
        assert!(
            (result[0] - 1.0).abs() < 1e-5,
            "expected 1.0, got {}",
            result[0]
        );
        assert!(
            (result[1] - 1.5).abs() < 1e-5,
            "expected 1.5, got {}",
            result[1]
        );
    }

    // 11. pcgrad two aligned tasks → result similar to weighted sum
    #[test]
    fn test_pcgrad_aligned_tasks_similar_to_weighted_sum() {
        let g1 = TaskGradient::new("a", grad(&[1.0, 0.0]), 1.0).unwrap();
        let g2 = TaskGradient::new("b", grad(&[0.5, 0.0]), 1.0).unwrap();
        // Both point in +x direction (aligned)
        let pcgrad_result = pcgrad(&[g1.clone(), g2.clone()], 0).unwrap();
        let weighted_sum: Vec<f32> = g1
            .gradient
            .iter()
            .zip(g2.gradient.iter())
            .map(|(a, b)| a * g1.weight + b * g2.weight)
            .collect();
        // For aligned gradients, projection does nothing, so result == weighted sum
        assert!(
            (pcgrad_result[0] - weighted_sum[0]).abs() < 1e-4,
            "expected {}, got {}",
            weighted_sum[0],
            pcgrad_result[0]
        );
    }

    // 12. pcgrad two conflicting tasks → reduces conflict
    #[test]
    fn test_pcgrad_conflicting_reduces_conflict() {
        // Strongly opposing gradients
        let g1 = TaskGradient::new("a", grad(&[1.0, 0.0]), 1.0).unwrap();
        let g2 = TaskGradient::new("b", grad(&[-1.0, 0.5]), 1.0).unwrap();

        let pcgrad_result = pcgrad(&[g1.clone(), g2.clone()], 0).unwrap();
        let weighted_result: Vec<f32> = g1
            .gradient
            .iter()
            .zip(g2.gradient.iter())
            .map(|(a, b)| a + b)
            .collect();

        // Dot product of pcgrad result vs each gradient should be less negative
        // than the naive weighted sum's interaction
        let naive_dot: f32 = weighted_result
            .iter()
            .zip(g1.gradient.iter())
            .map(|(a, b)| a * b)
            .sum();
        let pcgrad_dot: f32 = pcgrad_result
            .iter()
            .zip(g1.gradient.iter())
            .map(|(a, b)| a * b)
            .sum();
        // pcgrad should result in a more positive (less conflicting) aggregation
        assert!(
            pcgrad_dot >= naive_dot - 1e-4,
            "pcgrad_dot={pcgrad_dot}, naive_dot={naive_dot}: PCGrad should reduce conflict"
        );
    }

    // 13. pcgrad empty → EmptyGradients error
    #[test]
    fn test_pcgrad_empty_returns_error() {
        let result = pcgrad(&[], 0);
        assert!(
            matches!(result, Err(GradientSurgeryError::EmptyGradients)),
            "expected EmptyGradients"
        );
    }

    // 14. analyze_conflicts all-aligned → conflict_fraction = 0.0
    #[test]
    fn test_analyze_conflicts_all_aligned() {
        let tasks = vec![
            task("a", grad(&[1.0, 0.0])),
            task("b", grad(&[2.0, 0.0])),
            task("c", grad(&[0.5, 0.0])),
        ];
        let report = analyze_conflicts(&tasks).unwrap();
        assert_eq!(report.num_conflicts, 0);
        assert!(
            (report.conflict_fraction - 0.0).abs() < 1e-6,
            "expected 0.0, got {}",
            report.conflict_fraction
        );
    }

    // 15. analyze_conflicts all-conflicting → conflict_fraction = 1.0
    #[test]
    fn test_analyze_conflicts_all_conflicting() {
        // Two tasks with opposing gradients
        let tasks = vec![task("a", grad(&[1.0, 0.0])), task("b", grad(&[-1.0, 0.0]))];
        let report = analyze_conflicts(&tasks).unwrap();
        assert_eq!(report.num_conflicts, 1);
        assert!(
            (report.conflict_fraction - 1.0).abs() < 1e-6,
            "expected 1.0, got {}",
            report.conflict_fraction
        );
    }

    // 16. analyze_conflicts reports num_conflicts correctly
    #[test]
    fn test_analyze_conflicts_num_conflicts() {
        // 3 tasks: (a,b) conflict, (a,c) aligned, (b,c) conflict
        let tasks = vec![
            task("a", grad(&[1.0, 0.0])),
            task("b", grad(&[-1.0, 0.5])), // conflicts with a (dot=-1)
            task("c", grad(&[0.5, 1.0])), // aligned with a (dot=0.5), (b,c): -0.5+0.5=0 → not conflict
        ];
        let report = analyze_conflicts(&tasks).unwrap();
        assert_eq!(report.total_pairs, 3);
        // (a,b): 1*-1 + 0*0.5 = -1 → conflict
        // (a,c): 1*0.5 + 0*1 = 0.5 → aligned
        // (b,c): -1*0.5 + 0.5*1 = -0.5+0.5 = 0 → not conflict (dot=0, not < 0)
        assert_eq!(
            report.num_conflicts, 1,
            "expected 1 conflict pair, got {}",
            report.num_conflicts
        );
    }

    // 17. aggregate_gradients WeightedSum = Σ w_i * g_i
    #[test]
    fn test_aggregate_weighted_sum() {
        let g1 = TaskGradient::new("a", grad(&[2.0, 0.0]), 0.5).unwrap();
        let g2 = TaskGradient::new("b", grad(&[0.0, 4.0]), 0.25).unwrap();
        let result = aggregate_gradients(&[g1, g2], AggregationStrategy::WeightedSum, 0).unwrap();
        // 0.5*2 + 0.25*0 = 1.0
        // 0.5*0 + 0.25*4 = 1.0
        assert!(
            (result[0] - 1.0).abs() < 1e-5,
            "expected 1.0, got {}",
            result[0]
        );
        assert!(
            (result[1] - 1.0).abs() < 1e-5,
            "expected 1.0, got {}",
            result[1]
        );
    }

    // 18. aggregate_gradients GradNormalized: each gradient normalized before sum
    #[test]
    fn test_aggregate_grad_normalized() {
        // g1=[3,4] norm=5, normalized=[0.6,0.8]
        // g2=[0,1] norm=1, normalized=[0,1]
        let g1 = TaskGradient::new("a", grad(&[3.0, 4.0]), 1.0).unwrap();
        let g2 = TaskGradient::new("b", grad(&[0.0, 1.0]), 1.0).unwrap();
        let result =
            aggregate_gradients(&[g1, g2], AggregationStrategy::GradNormalized, 0).unwrap();
        // result[0] = 0.6*1 + 0*1 = 0.6
        // result[1] = 0.8*1 + 1*1 = 1.8
        assert!(
            (result[0] - 0.6).abs() < 1e-4,
            "expected 0.6, got {}",
            result[0]
        );
        assert!(
            (result[1] - 1.8).abs() < 1e-4,
            "expected 1.8, got {}",
            result[1]
        );
    }

    // 19. aggregate_gradients PCGrad reduces conflict vs WeightedSum (dot product test)
    #[test]
    fn test_aggregate_pcgrad_reduces_conflict() {
        // Two strongly opposing gradients
        let g1 = TaskGradient::new("a", grad(&[1.0, 0.0, 0.0]), 1.0).unwrap();
        let g2 = TaskGradient::new("b", grad(&[-1.0, 0.0, 1.0]), 1.0).unwrap();

        let ws = aggregate_gradients(
            &[g1.clone(), g2.clone()],
            AggregationStrategy::WeightedSum,
            0,
        )
        .unwrap();
        let pg = aggregate_gradients(&[g1, g2], AggregationStrategy::PCGrad, 0).unwrap();

        // WeightedSum = [0, 0, 1]; PCGrad should resolve the -x conflict
        // PCGrad result should have a higher (or equal) x-component than WS
        // The x-components of ws=0; pcgrad should be >= ws in x direction
        // because it removes the conflicting component from g2's x=-1
        // More specifically, pcgrad_x should be >= ws_x
        assert!(
            pg[0] >= ws[0] - 1e-4,
            "PCGrad x={} should be >= WeightedSum x={}",
            pg[0],
            ws[0]
        );
    }

    // 20. ConflictTracker::update increases step
    #[test]
    fn test_conflict_tracker_update_increments_step() {
        let mut tracker = ConflictTracker::new(0.3);
        assert_eq!(tracker.step_count(), 0);

        let tasks = vec![task("a", grad(&[1.0, 0.0])), task("b", grad(&[0.0, 1.0]))];
        let report = analyze_conflicts(&tasks).unwrap();
        tracker.update(&report);
        assert_eq!(tracker.step_count(), 1);
        tracker.update(&report);
        assert_eq!(tracker.step_count(), 2);
    }

    // 21. ConflictTracker::mean_conflict_rate after zero conflicts → 0.0
    #[test]
    fn test_conflict_tracker_zero_conflict_rate() {
        let mut tracker = ConflictTracker::new(0.3);
        // All aligned gradients — no conflicts
        let tasks = vec![task("a", grad(&[1.0, 0.0])), task("b", grad(&[2.0, 0.0]))];
        let report = analyze_conflicts(&tasks).unwrap();
        tracker.update(&report);
        let rate = tracker.mean_conflict_rate();
        assert!(rate.abs() < 1e-5, "expected ~0.0, got {rate}");
    }

    // 22. ConflictTracker::most_conflicting_pair returns None when empty
    #[test]
    fn test_conflict_tracker_most_conflicting_none_when_empty() {
        let tracker = ConflictTracker::new(0.3);
        assert!(tracker.most_conflicting_pair().is_none());
    }
}
