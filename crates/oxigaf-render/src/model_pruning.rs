//! Gaussian model structured pruning utilities.
//!
//! Provides criteria-based pruning that removes low-importance Gaussians from
//! a 3D Gaussian Splatting model to reduce memory and computation costs.
//!
//! # Example
//! ```rust
//! use oxigaf_render::model_pruning::{
//!     PruningCriteria, PruningConfig, compute_pruning_mask,
//! };
//!
//! let opacities = vec![-1.0_f32, 0.0, 2.0, 4.0];
//! let scales = vec![[0.1_f32, 0.2, 0.1]; 4];
//! let config = PruningConfig {
//!     criteria: PruningCriteria::TopK { k: 2 },
//!     min_survivors: 1,
//! };
//! let result = compute_pruning_mask(&opacities, &scales, &config).unwrap();
//! assert_eq!(result.kept_count, 2);
//! ```

use std::fmt;

// ---------------------------------------------------------------------------
// Math helpers
// ---------------------------------------------------------------------------

/// Numerically stable sigmoid: σ(x) = 1 / (1 + exp(−x)).
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Estimated memory footprint of one Gaussian (bytes).
///
/// Breakdown:
/// - position  : f32 × 3 =  12 bytes
/// - rotation  : f32 × 4 =  16 bytes
/// - scale     : f32 × 3 =  12 bytes
/// - opacity   : f32 × 1 =   4 bytes
/// - sh_coeffs : f32 × 48 = 192 bytes  (degree 3, 16 coefficients × 3 channels)
pub const BYTES_PER_GAUSSIAN: usize = 3 * 4   // position  f32×3
  + 4 * 4   // rotation  f32×4
  + 3 * 4   // scale     f32×3
  + 4   // opacity   f32
  + 48 * 4; // sh_coeffs (degree 3 → 16 coeffs × 3 channels)
            // Total: 12 + 16 + 12 + 4 + 192 = 236 bytes
            // (Note: the constant resolves to 236; the doc comment says 248 but the
            //  arithmetic above gives the precise breakdown.)

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during pruning operations.
#[derive(Debug)]
pub enum PruningError {
    /// The model contains no Gaussians.
    EmptyModel,
    /// Two arrays that should have the same length do not.
    LengthMismatch {
        /// Which field/array has the wrong length.
        field: &'static str,
        /// Expected length (derived from `opacities`).
        expected: usize,
        /// Actual length of the offending array.
        got: usize,
    },
    /// The `mask` passed to `apply_pruning_mask` or `prune_gaussian_arrays`
    /// has a different length from the data arrays.
    MaskLengthMismatch {
        /// Expected mask length.
        expected: usize,
        /// Actual mask length.
        got: usize,
    },
    /// `min_survivors` exceeds the total number of Gaussians in the model.
    MinSurvivorsTooLarge {
        /// Requested minimum survivors.
        requested: usize,
        /// Total Gaussians available.
        available: usize,
    },
}

impl fmt::Display for PruningError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PruningError::EmptyModel => write!(f, "model contains no Gaussians"),
            PruningError::LengthMismatch {
                field,
                expected,
                got,
            } => write!(
                f,
                "length mismatch in field '{field}': expected {expected}, got {got}"
            ),
            PruningError::MaskLengthMismatch { expected, got } => {
                write!(f, "mask length mismatch: expected {expected}, got {got}")
            }
            PruningError::MinSurvivorsTooLarge {
                requested,
                available,
            } => write!(
                f,
                "min_survivors ({requested}) exceeds total Gaussian count ({available})"
            ),
        }
    }
}

impl std::error::Error for PruningError {}

// ---------------------------------------------------------------------------
// Pruning criteria
// ---------------------------------------------------------------------------

/// Strategy used to decide which Gaussians to remove.
#[derive(Debug, Clone, PartialEq)]
pub enum PruningCriteria {
    /// Prune any Gaussian whose `sigmoid(opacity_logit) < min_opacity`.
    Opacity {
        /// Opacity threshold in [0, 1]. Gaussians below this are pruned.
        min_opacity: f32,
    },
    /// Prune any Gaussian whose `exp(max_scale_component) > max_scale`.
    Scale {
        /// Maximum allowed world-space scale. Gaussians larger than this are pruned.
        max_scale: f32,
    },
    /// Keep the `target_count` highest-scoring Gaussians, where the score is
    /// a weighted combination of opacity (higher = better) and inverse scale
    /// (smaller scale = better).
    CompositeScore {
        /// Weight given to opacity in the composite score.
        opacity_weight: f32,
        /// Weight given to inverse scale (smaller scale → higher contribution).
        scale_weight: f32,
        /// Number of Gaussians to retain.
        target_count: usize,
    },
    /// Keep the `k` Gaussians with the highest `sigmoid(opacity_logit)`.
    TopK {
        /// Number of Gaussians to retain.
        k: usize,
    },
}

// ---------------------------------------------------------------------------
// Pruning config
// ---------------------------------------------------------------------------

/// Configuration for a single pruning pass.
#[derive(Debug, Clone)]
pub struct PruningConfig {
    /// Which criterion to apply.
    pub criteria: PruningCriteria,
    /// Minimum number of Gaussians that must survive pruning.
    ///
    /// If the chosen criterion would discard more than `n - min_survivors`
    /// Gaussians, the top `min_survivors` by importance score are kept instead.
    pub min_survivors: usize,
}

impl Default for PruningConfig {
    fn default() -> Self {
        PruningConfig {
            criteria: PruningCriteria::Opacity { min_opacity: 0.005 },
            min_survivors: 1,
        }
    }
}

// ---------------------------------------------------------------------------
// Importance scorer
// ---------------------------------------------------------------------------

/// Computes per-Gaussian importance scores used for ranking and pruning.
pub struct ImportanceScorer;

impl ImportanceScorer {
    /// Score each Gaussian according to `criteria`.
    ///
    /// A **higher** score means the Gaussian is **more important** (keep it).
    ///
    /// | Criteria        | Score formula                                                             |
    /// |-----------------|---------------------------------------------------------------------------|
    /// | `Opacity`       | `sigmoid(opacity_logit)`                                                  |
    /// | `Scale`         | `-exp(max_scale_component)` — negative so smaller scale → higher score   |
    /// | `CompositeScore`| `ow*sigmoid(op) + sw*(1 - exp(max_s)/max_possible_scale)`               |
    /// | `TopK`          | `sigmoid(opacity_logit)` — same as Opacity                               |
    ///
    /// `opacities` are logit-space (before sigmoid).
    /// `scales` are log-space `[sx, sy, sz]` (before `exp`).
    pub fn score(opacities: &[f32], scales: &[[f32; 3]], criteria: &PruningCriteria) -> Vec<f32> {
        let n = opacities.len();
        let mut scores = Vec::with_capacity(n);

        match criteria {
            PruningCriteria::Opacity { .. } | PruningCriteria::TopK { .. } => {
                for &op in opacities {
                    scores.push(sigmoid(op));
                }
            }

            PruningCriteria::Scale { .. } => {
                for s in scales {
                    let max_s = s[0].max(s[1]).max(s[2]);
                    scores.push(-max_s.exp());
                }
            }

            PruningCriteria::CompositeScore {
                opacity_weight,
                scale_weight,
                ..
            } => {
                // Compute all world-space max-scales to find the global maximum
                // (used for normalization of the scale term).
                let world_max_scales: Vec<f32> = scales
                    .iter()
                    .map(|s| s[0].max(s[1]).max(s[2]).exp())
                    .collect();

                let global_max = world_max_scales
                    .iter()
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max);

                // Guard: if all scales are identical (or model is empty),
                // the scale term equals 1 for everyone — no differentiation.
                let effective_max = if global_max > f32::EPSILON {
                    global_max
                } else {
                    1.0
                };

                for (i, &op) in opacities.iter().enumerate() {
                    let opacity_score = sigmoid(op);
                    let scale_score = 1.0 - world_max_scales[i] / effective_max;
                    scores.push(opacity_weight * opacity_score + scale_weight * scale_score);
                }
            }
        }

        scores
    }
}

// ---------------------------------------------------------------------------
// Pruning result
// ---------------------------------------------------------------------------

/// Outcome of a pruning pass.
#[derive(Debug, Clone)]
pub struct PruningResult {
    /// `true` for each Gaussian that should be kept after pruning.
    pub mask: Vec<bool>,
    /// Number of Gaussians retained.
    pub kept_count: usize,
    /// Number of Gaussians removed.
    pub pruned_count: usize,
    /// Total Gaussians before pruning.
    pub original_count: usize,
    /// Estimated bytes freed: `pruned_count × BYTES_PER_GAUSSIAN`.
    pub memory_saved_bytes: usize,
}

impl PruningResult {
    /// Human-readable one-liner summarising the pruning outcome.
    pub fn format_summary(&self) -> String {
        let pct_kept = self.kept_fraction() * 100.0;
        let pct_pruned = 100.0 - pct_kept;
        let mb_saved = self.memory_saved_bytes as f64 / (1024.0 * 1024.0);
        format!(
            "Pruning: {kept}/{orig} kept ({pct_kept:.1}%), \
             {pruned} removed ({pct_pruned:.1}%), \
             ~{mb:.2} MiB freed",
            kept = self.kept_count,
            orig = self.original_count,
            pct_kept = pct_kept,
            pruned = self.pruned_count,
            pct_pruned = pct_pruned,
            mb = mb_saved,
        )
    }

    /// Fraction of Gaussians that survived pruning, in `[0, 1]`.
    pub fn kept_fraction(&self) -> f32 {
        if self.original_count == 0 {
            return 1.0;
        }
        self.kept_count as f32 / self.original_count as f32
    }
}

// ---------------------------------------------------------------------------
// Core pruning logic
// ---------------------------------------------------------------------------

/// Compute a boolean keep/prune mask for a set of Gaussians.
///
/// # Arguments
/// * `opacities` — logit-space (pre-sigmoid) opacity for each Gaussian
/// * `scales`    — log-space `[sx, sy, sz]` for each Gaussian
/// * `config`    — pruning strategy and minimum-survivors floor
///
/// # Errors
/// Returns [`PruningError::EmptyModel`] when `opacities` is empty, or
/// [`PruningError::LengthMismatch`] when `scales.len() != opacities.len()`.
pub fn compute_pruning_mask(
    opacities: &[f32],
    scales: &[[f32; 3]],
    config: &PruningConfig,
) -> Result<PruningResult, PruningError> {
    let n = opacities.len();

    // --- Validate inputs ---
    if n == 0 {
        return Err(PruningError::EmptyModel);
    }
    if scales.len() != n {
        return Err(PruningError::LengthMismatch {
            field: "scales",
            expected: n,
            got: scales.len(),
        });
    }
    if config.min_survivors > n {
        return Err(PruningError::MinSurvivorsTooLarge {
            requested: config.min_survivors,
            available: n,
        });
    }

    // --- Build mask ---
    let mask = match &config.criteria {
        PruningCriteria::Opacity { min_opacity } => threshold_mask_with_floor(
            opacities,
            scales,
            &config.criteria,
            config.min_survivors,
            |scores, idx| scores[idx] >= *min_opacity,
        ),

        PruningCriteria::Scale { max_scale } => {
            let ms = *max_scale;
            threshold_mask_with_floor(
                opacities,
                scales,
                &config.criteria,
                config.min_survivors,
                |_scores, idx| {
                    let s = &scales[idx];
                    let world = s[0].max(s[1]).max(s[2]).exp();
                    world <= ms
                },
            )
        }

        PruningCriteria::TopK { k } => {
            let effective_k = (*k).max(config.min_survivors).min(n);
            topk_mask(opacities, scales, &config.criteria, effective_k, n)
        }

        PruningCriteria::CompositeScore { target_count, .. } => {
            let effective_k = (*target_count).max(config.min_survivors).min(n);
            topk_mask(opacities, scales, &config.criteria, effective_k, n)
        }
    };

    let kept_count = mask.iter().filter(|&&b| b).count();
    let pruned_count = n - kept_count;

    Ok(PruningResult {
        mask,
        kept_count,
        pruned_count,
        original_count: n,
        memory_saved_bytes: pruned_count * BYTES_PER_GAUSSIAN,
    })
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Build a mask using a per-element threshold predicate.
///
/// If fewer than `min_survivors` elements pass the predicate, falls back to
/// keeping the top `min_survivors` by score (descending).
fn threshold_mask_with_floor<F>(
    opacities: &[f32],
    scales: &[[f32; 3]],
    criteria: &PruningCriteria,
    min_survivors: usize,
    predicate: F,
) -> Vec<bool>
where
    F: Fn(&[f32], usize) -> bool,
{
    let n = opacities.len();
    let scores = ImportanceScorer::score(opacities, scales, criteria);

    // Apply the threshold predicate.
    let mut mask: Vec<bool> = (0..n).map(|i| predicate(&scores, i)).collect();

    let kept = mask.iter().filter(|&&b| b).count();
    if kept < min_survivors {
        // Fall back: keep the top min_survivors by score.
        // Build an index list sorted by score descending.
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_unstable_by(|&a, &b| scores[b].total_cmp(&scores[a]));

        // Reset mask and mark top min_survivors.
        mask.fill(false);
        for &idx in indices.iter().take(min_survivors) {
            mask[idx] = true;
        }
    }

    mask
}

/// Build a mask keeping the top `k` Gaussians by score (descending).
fn topk_mask(
    opacities: &[f32],
    scales: &[[f32; 3]],
    criteria: &PruningCriteria,
    k: usize,
    n: usize,
) -> Vec<bool> {
    let scores = ImportanceScorer::score(opacities, scales, criteria);

    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_unstable_by(|&a, &b| scores[b].total_cmp(&scores[a]));

    let mut mask = vec![false; n];
    for &idx in indices.iter().take(k) {
        mask[idx] = true;
    }
    mask
}

// ---------------------------------------------------------------------------
// Apply pruning mask to generic slices
// ---------------------------------------------------------------------------

/// Filter `data` according to `mask`, returning only the kept elements.
///
/// # Errors
/// Returns [`PruningError::MaskLengthMismatch`] when lengths differ.
pub fn apply_pruning_mask<T: Clone>(data: &[T], mask: &[bool]) -> Result<Vec<T>, PruningError> {
    if data.len() != mask.len() {
        return Err(PruningError::MaskLengthMismatch {
            expected: data.len(),
            got: mask.len(),
        });
    }
    Ok(data
        .iter()
        .zip(mask.iter())
        .filter_map(|(item, &keep)| if keep { Some(item.clone()) } else { None })
        .collect())
}

// ---------------------------------------------------------------------------
// Pruned arrays result
// ---------------------------------------------------------------------------

/// All per-Gaussian arrays after applying a pruning mask.
#[derive(Debug, Clone)]
pub struct PrunedArrays {
    /// World-space positions `[x, y, z]` for each surviving Gaussian.
    pub positions: Vec<[f32; 3]>,
    /// Unit quaternions `[x, y, z, w]` for each surviving Gaussian.
    pub rotations: Vec<[f32; 4]>,
    /// Log-space scales `[sx, sy, sz]` for each surviving Gaussian.
    pub scales: Vec<[f32; 3]>,
    /// Logit-space opacities for each surviving Gaussian.
    pub opacities: Vec<f32>,
    /// Spherical harmonics coefficient vectors for each surviving Gaussian.
    pub sh_coeffs: Vec<Vec<f32>>,
}

/// Apply a pruning mask to all per-Gaussian arrays simultaneously.
///
/// All input slices must have the same length (number of Gaussians), and
/// `mask` must also match that length.
///
/// # Arguments
/// * `positions`  — world-space positions `[x, y, z]`
/// * `rotations`  — unit quaternions `[x, y, z, w]`
/// * `scales`     — log-space scales `[sx, sy, sz]`
/// * `opacities`  — logit-space opacities
/// * `sh_coeffs`  — spherical harmonics coefficient slices (one `Vec<f32>` per Gaussian)
/// * `mask`       — boolean keep/prune flags (must match `opacities.len()`)
///
/// # Errors
/// Returns an appropriate [`PruningError`] on any length mismatch.
pub fn prune_gaussian_arrays(
    positions: &[[f32; 3]],
    rotations: &[[f32; 4]],
    scales: &[[f32; 3]],
    opacities: &[f32],
    sh_coeffs: &[Vec<f32>],
    mask: &[bool],
) -> Result<PrunedArrays, PruningError> {
    let n = opacities.len();

    // Validate that every field has the expected length.
    if positions.len() != n {
        return Err(PruningError::LengthMismatch {
            field: "positions",
            expected: n,
            got: positions.len(),
        });
    }
    if rotations.len() != n {
        return Err(PruningError::LengthMismatch {
            field: "rotations",
            expected: n,
            got: rotations.len(),
        });
    }
    if scales.len() != n {
        return Err(PruningError::LengthMismatch {
            field: "scales",
            expected: n,
            got: scales.len(),
        });
    }
    if sh_coeffs.len() != n {
        return Err(PruningError::LengthMismatch {
            field: "sh_coeffs",
            expected: n,
            got: sh_coeffs.len(),
        });
    }
    if mask.len() != n {
        return Err(PruningError::MaskLengthMismatch {
            expected: n,
            got: mask.len(),
        });
    }

    // Apply the mask to each field.
    let positions_out = apply_pruning_mask(positions, mask)?;
    let rotations_out = apply_pruning_mask(rotations, mask)?;
    let scales_out = apply_pruning_mask(scales, mask)?;
    let opacities_out = apply_pruning_mask(opacities, mask)?;
    let sh_coeffs_out = apply_pruning_mask(sh_coeffs, mask)?;

    Ok(PrunedArrays {
        positions: positions_out,
        rotations: rotations_out,
        scales: scales_out,
        opacities: opacities_out,
        sh_coeffs: sh_coeffs_out,
    })
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

    fn make_scales(n: usize, val: f32) -> Vec<[f32; 3]> {
        vec![[val, val, val]; n]
    }

    fn default_rotations(n: usize) -> Vec<[f32; 4]> {
        vec![[0.0, 0.0, 0.0, 1.0]; n]
    }

    fn default_positions(n: usize) -> Vec<[f32; 3]> {
        vec![[0.0, 0.0, 0.0]; n]
    }

    fn default_sh(n: usize) -> Vec<Vec<f32>> {
        vec![vec![0.0_f32; 48]; n]
    }

    // -----------------------------------------------------------------------
    // sigmoid
    // -----------------------------------------------------------------------

    #[test]
    fn test_sigmoid_values() {
        // At 0 → 0.5
        let s0 = sigmoid(0.0);
        assert!(
            (s0 - 0.5).abs() < 1e-6,
            "sigmoid(0) should be 0.5, got {s0}"
        );

        // Large positive → approaching 1
        let s_pos = sigmoid(100.0);
        assert!(
            (s_pos - 1.0).abs() < 1e-6,
            "sigmoid(100) should be ~1.0, got {s_pos}"
        );

        // Large negative → approaching 0
        let s_neg = sigmoid(-100.0);
        assert!(s_neg < 1e-6, "sigmoid(-100) should be ~0.0, got {s_neg}");

        // Known value: sigmoid(1) ≈ 0.731059
        let s1 = sigmoid(1.0);
        assert!(
            (s1 - 0.731_059).abs() < 1e-4,
            "sigmoid(1) ≈ 0.731059, got {s1}"
        );

        // Symmetry: sigmoid(-x) = 1 - sigmoid(x)
        for x in [0.5_f32, 1.0, 2.0, 5.0] {
            let diff = (sigmoid(-x) - (1.0 - sigmoid(x))).abs();
            assert!(diff < 1e-6, "symmetry failed at x={x}: diff={diff}");
        }
    }

    // -----------------------------------------------------------------------
    // ImportanceScorer
    // -----------------------------------------------------------------------

    #[test]
    fn test_score_opacity_criteria() {
        let opacities = vec![0.0_f32, 1.0, -1.0, 10.0];
        let scales = make_scales(4, 0.0);
        let criteria = PruningCriteria::Opacity { min_opacity: 0.5 };
        let scores = ImportanceScorer::score(&opacities, &scales, &criteria);

        assert_eq!(scores.len(), 4);
        // scores should equal sigmoid of each opacity
        for (i, &op) in opacities.iter().enumerate() {
            let expected = sigmoid(op);
            assert!(
                (scores[i] - expected).abs() < 1e-6,
                "score[{i}] = {}, expected {}",
                scores[i],
                expected
            );
        }
        // Monotonically increasing with opacity logit
        assert!(scores[3] > scores[1], "higher logit → higher score");
        assert!(scores[2] < scores[0], "lower logit → lower score");
    }

    #[test]
    fn test_score_scale_criteria() {
        // scales are in log-space: exp(scale[i]) is world-space size
        // score = -exp(max_scale_component), so larger scale → more negative score
        let opacities = vec![0.0_f32; 3];
        let scales = vec![
            [0.0_f32, 0.0, 0.0],    // exp(0)=1  → score=-1
            [1.0_f32, 0.0, 0.0],    // exp(1)≈2.72 → score≈-2.72
            [-1.0_f32, -1.0, -1.0], // exp(-1)≈0.37 → score≈-0.37
        ];
        let criteria = PruningCriteria::Scale { max_scale: 1.5 };
        let scores = ImportanceScorer::score(&opacities, &scales, &criteria);

        assert_eq!(scores.len(), 3);
        // smallest scale (index 2) → highest (least negative) score
        assert!(scores[2] > scores[0], "small scale should score higher");
        assert!(
            scores[0] > scores[1],
            "medium scale should score higher than large"
        );

        // Exact checks
        assert!(
            (scores[0] - (-1.0_f32)).abs() < 1e-5,
            "score[0]={}",
            scores[0]
        );
        assert!(
            (scores[1] - (-1.0_f32.exp())).abs() < 1e-5,
            "score[1]={}",
            scores[1]
        );
        assert!(
            (scores[2] - (-(-1.0_f32).exp())).abs() < 1e-5,
            "score[2]={}",
            scores[2]
        );
    }

    #[test]
    fn test_score_composite_criteria() {
        // 3 Gaussians with different opacity/scale combos
        let opacities = vec![0.0_f32, 2.0, -2.0]; // sigmoid: 0.5, ~0.88, ~0.12
        let scales = vec![
            [0.0_f32, 0.0, 0.0],    // exp max = 1.0
            [2.0_f32, 0.0, 0.0],    // exp max ≈ 7.39
            [-2.0_f32, -2.0, -2.0], // exp max ≈ 0.135
        ];
        let criteria = PruningCriteria::CompositeScore {
            opacity_weight: 0.5,
            scale_weight: 0.5,
            target_count: 2,
        };
        let scores = ImportanceScorer::score(&opacities, &scales, &criteria);

        assert_eq!(scores.len(), 3);
        // Index 2 has lowest opacity but smallest scale — depends on weights.
        // All scores should be in a valid range (0 ≤ score ≤ 1 for equal weights).
        for (i, &s) in scores.iter().enumerate() {
            assert!(s.is_finite(), "score[{i}] should be finite, got {s}");
        }
        // The Gaussian with the largest scale (index 1) gets penalty from scale term.
        // The Gaussian with highest opacity (index 1) gets boost from opacity term.
        // Just confirm scores are distinct (not all equal).
        assert!(
            (scores[0] - scores[1]).abs() > 1e-4 || (scores[0] - scores[2]).abs() > 1e-4,
            "composite scores should differentiate Gaussians"
        );
    }

    #[test]
    fn test_score_topk_criteria() {
        // TopK uses sigmoid(opacity) — same as Opacity criteria
        let opacities = vec![-5.0_f32, 0.0, 5.0];
        let scales = make_scales(3, 0.0);
        let criteria_topk = PruningCriteria::TopK { k: 2 };
        let criteria_op = PruningCriteria::Opacity { min_opacity: 0.5 };

        let scores_topk = ImportanceScorer::score(&opacities, &scales, &criteria_topk);
        let scores_op = ImportanceScorer::score(&opacities, &scales, &criteria_op);

        assert_eq!(scores_topk.len(), scores_op.len());
        for (i, (a, b)) in scores_topk.iter().zip(scores_op.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "TopK and Opacity scores should be equal at index {i}: {a} vs {b}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // compute_pruning_mask — Opacity
    // -----------------------------------------------------------------------

    #[test]
    fn test_compute_pruning_mask_opacity() {
        // opacities (logit): -2, 0, 2, 4
        // sigmoid:           ~0.12, 0.5, ~0.88, ~0.98
        let opacities = vec![-2.0_f32, 0.0, 2.0, 4.0];
        let scales = make_scales(4, 0.0);
        let config = PruningConfig {
            criteria: PruningCriteria::Opacity { min_opacity: 0.5 },
            min_survivors: 1,
        };

        let result =
            compute_pruning_mask(&opacities, &scales, &config).expect("pruning should succeed");

        assert_eq!(result.original_count, 4);
        // logit -2 (σ≈0.12) and 0 (σ=0.5) should be pruned
        // Note: sigmoid(0) == 0.5 which equals min_opacity=0.5 → keep
        assert!(!result.mask[0], "opacity -2 should be pruned");
        assert!(result.mask[1], "opacity 0 (σ=0.5) should be kept");
        assert!(result.mask[2], "opacity 2 should be kept");
        assert!(result.mask[3], "opacity 4 should be kept");

        assert_eq!(result.kept_count, 3);
        assert_eq!(result.pruned_count, 1);
        assert_eq!(
            result.memory_saved_bytes, BYTES_PER_GAUSSIAN,
            "one Gaussian worth of memory freed"
        );
    }

    // -----------------------------------------------------------------------
    // compute_pruning_mask — Scale
    // -----------------------------------------------------------------------

    #[test]
    fn test_compute_pruning_mask_scale() {
        // scales (log-space): 0=exp(0)=1, 1=exp(1)≈2.72, -1=exp(-1)≈0.37
        let opacities = vec![0.0_f32; 3];
        let scales = vec![
            [0.0_f32, 0.0, 0.0],    // world max = 1.0
            [1.0_f32, 0.0, 0.0],    // world max ≈ 2.72
            [-1.0_f32, -1.0, -1.0], // world max ≈ 0.37
        ];
        let config = PruningConfig {
            criteria: PruningCriteria::Scale { max_scale: 1.5 },
            min_survivors: 1,
        };

        let result =
            compute_pruning_mask(&opacities, &scales, &config).expect("pruning should succeed");

        assert!(result.mask[0], "scale exp(0)=1 ≤ 1.5 → keep");
        assert!(!result.mask[1], "scale exp(1)≈2.72 > 1.5 → prune");
        assert!(result.mask[2], "scale exp(-1)≈0.37 ≤ 1.5 → keep");
        assert_eq!(result.kept_count, 2);
        assert_eq!(result.pruned_count, 1);
    }

    // -----------------------------------------------------------------------
    // compute_pruning_mask — TopK
    // -----------------------------------------------------------------------

    #[test]
    fn test_compute_pruning_mask_topk() {
        // 6 Gaussians; keep top 3 by opacity
        let opacities = vec![-3.0_f32, -1.0, 0.0, 1.0, 3.0, 5.0];
        let scales = make_scales(6, 0.0);
        let config = PruningConfig {
            criteria: PruningCriteria::TopK { k: 3 },
            min_survivors: 1,
        };

        let result =
            compute_pruning_mask(&opacities, &scales, &config).expect("pruning should succeed");

        assert_eq!(result.kept_count, 3);
        assert_eq!(result.pruned_count, 3);

        // The top 3 by sigmoid(opacity) are indices 3,4,5 (logits 1,3,5)
        assert!(!result.mask[0], "logit -3 should be pruned");
        assert!(!result.mask[1], "logit -1 should be pruned");
        assert!(!result.mask[2], "logit 0 should be pruned");
        assert!(result.mask[3], "logit 1 should be kept");
        assert!(result.mask[4], "logit 3 should be kept");
        assert!(result.mask[5], "logit 5 should be kept");
    }

    // -----------------------------------------------------------------------
    // compute_pruning_mask — CompositeScore
    // -----------------------------------------------------------------------

    #[test]
    fn test_compute_pruning_mask_composite() {
        // 4 Gaussians — we keep top 2 by composite score
        let opacities = vec![-2.0_f32, -1.0, 1.0, 2.0];
        // All same scale → scale term equal; result driven by opacity
        let scales = make_scales(4, 0.0);
        let config = PruningConfig {
            criteria: PruningCriteria::CompositeScore {
                opacity_weight: 1.0,
                scale_weight: 0.0,
                target_count: 2,
            },
            min_survivors: 1,
        };

        let result =
            compute_pruning_mask(&opacities, &scales, &config).expect("pruning should succeed");

        assert_eq!(result.kept_count, 2);
        // With scale_weight=0, composite = opacity. Top 2 = indices 2,3 (logits 1,2).
        assert!(!result.mask[0], "logit -2 should be pruned");
        assert!(!result.mask[1], "logit -1 should be pruned");
        assert!(result.mask[2], "logit 1 should be kept");
        assert!(result.mask[3], "logit 2 should be kept");
    }

    // -----------------------------------------------------------------------
    // min_survivors floor
    // -----------------------------------------------------------------------

    #[test]
    fn test_min_survivors_respected() {
        // All opacities are well below threshold → threshold would prune all.
        // min_survivors=2 should force the 2 best to survive.
        let opacities = vec![-10.0_f32, -9.0, -8.0, -7.0];
        let scales = make_scales(4, 0.0);
        let config = PruningConfig {
            criteria: PruningCriteria::Opacity { min_opacity: 0.99 },
            min_survivors: 2,
        };

        let result =
            compute_pruning_mask(&opacities, &scales, &config).expect("pruning should succeed");

        // Natural survivors: none (all σ << 0.99).
        // Floor kicks in → top 2 by sigmoid(opacity) are indices 2,3 (logits -8,-7).
        assert_eq!(result.kept_count, 2, "min_survivors=2 must be respected");
        assert!(!result.mask[0], "logit -10 should be pruned");
        assert!(!result.mask[1], "logit -9 should be pruned");
        assert!(result.mask[2], "logit -8 should be forced-kept (floor)");
        assert!(result.mask[3], "logit -7 should be forced-kept (floor)");
    }

    // -----------------------------------------------------------------------
    // apply_pruning_mask
    // -----------------------------------------------------------------------

    #[test]
    fn test_apply_pruning_mask_basic() {
        let data = vec![10_u32, 20, 30, 40, 50];
        let mask = vec![true, false, true, false, true];

        let filtered = apply_pruning_mask(&data, &mask).expect("apply_pruning_mask should succeed");

        assert_eq!(filtered, vec![10, 30, 50]);
    }

    #[test]
    fn test_apply_pruning_mask_empty_result() {
        let data = vec![1_i32, 2, 3];
        let mask = vec![false, false, false];

        let filtered =
            apply_pruning_mask(&data, &mask).expect("should succeed even with all-false mask");

        assert!(filtered.is_empty(), "all pruned → empty result");
    }

    #[test]
    fn test_apply_pruning_mask_length_mismatch_returns_error() {
        let data = vec![1_i32, 2, 3];
        let mask = vec![true, false]; // wrong length

        match apply_pruning_mask(&data, &mask) {
            Err(PruningError::MaskLengthMismatch { expected, got }) => {
                assert_eq!(expected, 3);
                assert_eq!(got, 2);
            }
            other => panic!("expected MaskLengthMismatch, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // prune_gaussian_arrays
    // -----------------------------------------------------------------------

    #[test]
    fn test_prune_gaussian_arrays() {
        let n = 4;
        let positions = vec![
            [1.0_f32, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ];
        let rotations = default_rotations(n);
        let scales = make_scales(n, 0.0);
        let opacities = vec![-2.0_f32, 0.0, 2.0, 4.0];
        let sh_coeffs: Vec<Vec<f32>> = (0..n).map(|i| vec![i as f32; 48]).collect();
        let mask = vec![false, true, false, true]; // keep indices 1 and 3

        let pruned = prune_gaussian_arrays(
            &positions, &rotations, &scales, &opacities, &sh_coeffs, &mask,
        )
        .expect("prune_gaussian_arrays should succeed");

        assert_eq!(pruned.positions.len(), 2);
        assert_eq!(pruned.rotations.len(), 2);
        assert_eq!(pruned.scales.len(), 2);
        assert_eq!(pruned.opacities.len(), 2);
        assert_eq!(pruned.sh_coeffs.len(), 2);

        // positions: kept indices 1 and 3
        assert_eq!(pruned.positions[0], [2.0, 0.0, 0.0]);
        assert_eq!(pruned.positions[1], [4.0, 0.0, 0.0]);

        // opacities: kept 0.0 and 4.0
        assert!((pruned.opacities[0] - 0.0).abs() < 1e-6);
        assert!((pruned.opacities[1] - 4.0).abs() < 1e-6);

        // sh_coeffs: index 1 → all 1.0, index 3 → all 3.0
        assert!(pruned.sh_coeffs[0].iter().all(|&v| (v - 1.0).abs() < 1e-6));
        assert!(pruned.sh_coeffs[1].iter().all(|&v| (v - 3.0).abs() < 1e-6));
    }

    // -----------------------------------------------------------------------
    // PruningResult::format_summary
    // -----------------------------------------------------------------------

    #[test]
    fn test_pruning_result_format_summary() {
        let result = PruningResult {
            mask: vec![true, true, false, false],
            kept_count: 2,
            pruned_count: 2,
            original_count: 4,
            memory_saved_bytes: 2 * BYTES_PER_GAUSSIAN,
        };

        let summary = result.format_summary();
        assert!(
            summary.contains("2/4"),
            "summary should contain '2/4', got: {summary}"
        );
        assert!(
            summary.contains("50.0%"),
            "summary should contain '50.0%' kept, got: {summary}"
        );
    }

    #[test]
    fn test_kept_fraction() {
        let result = PruningResult {
            mask: vec![true, false, false, false],
            kept_count: 1,
            pruned_count: 3,
            original_count: 4,
            memory_saved_bytes: 3 * BYTES_PER_GAUSSIAN,
        };
        let frac = result.kept_fraction();
        assert!(
            (frac - 0.25).abs() < 1e-6,
            "kept_fraction should be 0.25, got {frac}"
        );
    }

    // -----------------------------------------------------------------------
    // Error cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_length_mismatch_error() {
        let opacities = vec![0.0_f32; 5];
        let scales = make_scales(3, 0.0); // wrong length
        let config = PruningConfig::default();

        match compute_pruning_mask(&opacities, &scales, &config) {
            Err(PruningError::LengthMismatch {
                field,
                expected,
                got,
            }) => {
                assert_eq!(field, "scales");
                assert_eq!(expected, 5);
                assert_eq!(got, 3);
            }
            other => panic!("expected LengthMismatch, got {other:?}"),
        }
    }

    #[test]
    fn test_empty_model_error() {
        let opacities: Vec<f32> = vec![];
        let scales: Vec<[f32; 3]> = vec![];
        let config = PruningConfig::default();

        match compute_pruning_mask(&opacities, &scales, &config) {
            Err(PruningError::EmptyModel) => {}
            other => panic!("expected EmptyModel, got {other:?}"),
        }
    }

    #[test]
    fn test_min_survivors_too_large_error() {
        let opacities = vec![0.0_f32; 3];
        let scales = make_scales(3, 0.0);
        let config = PruningConfig {
            criteria: PruningCriteria::TopK { k: 2 },
            min_survivors: 10, // larger than n=3
        };

        match compute_pruning_mask(&opacities, &scales, &config) {
            Err(PruningError::MinSurvivorsTooLarge {
                requested,
                available,
            }) => {
                assert_eq!(requested, 10);
                assert_eq!(available, 3);
            }
            other => panic!("expected MinSurvivorsTooLarge, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Integration: full pipeline
    // -----------------------------------------------------------------------

    #[test]
    fn test_full_pruning_pipeline() {
        // Build 8 Gaussians, prune to top 4 by opacity, then apply mask.
        let n = 8;
        let opacities: Vec<f32> = (0..n as i32).map(|i| (i - 4) as f32).collect();
        // logits: -4, -3, -2, -1, 0, 1, 2, 3
        let scales = make_scales(n, 0.0);
        let positions = (0..n).map(|i| [i as f32, 0.0, 0.0]).collect::<Vec<_>>();
        let rotations = default_rotations(n);
        let sh = default_sh(n);

        let config = PruningConfig {
            criteria: PruningCriteria::TopK { k: 4 },
            min_survivors: 2,
        };

        let result =
            compute_pruning_mask(&opacities, &scales, &config).expect("compute should succeed");
        assert_eq!(result.kept_count, 4);

        let pruned = prune_gaussian_arrays(
            &positions,
            &rotations,
            &scales,
            &opacities,
            &sh,
            &result.mask,
        )
        .expect("apply should succeed");

        assert_eq!(pruned.positions.len(), 4);
        // Top 4 opacities are logits 0,1,2,3 → indices 4,5,6,7
        assert!(
            pruned.opacities.iter().all(|&o| o >= 0.0),
            "all kept opacities should be non-negative logits"
        );
    }

    // -----------------------------------------------------------------------
    // BYTES_PER_GAUSSIAN sanity
    // -----------------------------------------------------------------------

    #[test]
    fn test_bytes_per_gaussian_constant() {
        // 3*4 + 4*4 + 3*4 + 1*4 + 48*4 = 12+16+12+4+192 = 236
        assert_eq!(BYTES_PER_GAUSSIAN, 236, "constant should be 236 bytes");
    }

    #[test]
    fn test_prune_gaussian_arrays_length_mismatch() {
        let n = 3;
        let positions = default_positions(n);
        let rotations = default_rotations(n);
        let scales = make_scales(n, 0.0);
        let opacities = vec![0.0_f32; n];
        let sh = default_sh(n);
        let mask_bad = vec![true, false]; // length 2, not 3

        match prune_gaussian_arrays(&positions, &rotations, &scales, &opacities, &sh, &mask_bad) {
            Err(PruningError::MaskLengthMismatch { expected, got }) => {
                assert_eq!(expected, 3);
                assert_eq!(got, 2);
            }
            other => panic!("expected MaskLengthMismatch, got {other:?}"),
        }
    }

    #[test]
    fn test_pruning_error_display() {
        let e1 = PruningError::EmptyModel;
        assert!(e1.to_string().contains("no Gaussian"));

        let e2 = PruningError::LengthMismatch {
            field: "positions",
            expected: 5,
            got: 3,
        };
        assert!(e2.to_string().contains("positions"));
        assert!(e2.to_string().contains("5"));

        let e3 = PruningError::MaskLengthMismatch {
            expected: 4,
            got: 2,
        };
        assert!(e3.to_string().contains("mask length"));

        let e4 = PruningError::MinSurvivorsTooLarge {
            requested: 10,
            available: 5,
        };
        assert!(e4.to_string().contains("10"));
        assert!(e4.to_string().contains("5"));
    }
}
