//! Self-training utilities: prediction entropy, confidence masking and the
//! pseudo-label loss.

use super::common::DomainAdaptationError;

// ---------------------------------------------------------------------------
// Self-training: entropy utilities
// ---------------------------------------------------------------------------

/// Compute information entropy: `H(p) = -sum_k p_k * log(p_k + eps)`.
///
/// Uses a hardcoded eps of 1e-12 to handle p=0 conventionally (0·log0 = 0).
pub fn da_entropy(probs: &[f32]) -> f32 {
    const EPS: f32 = 1e-12;
    -probs.iter().map(|&p| p * (p + EPS).ln()).sum::<f32>()
}

/// Compute the entropy loss for domain adaptation.
///
/// Encourages confident (low-entropy) predictions on the target domain.
/// Returns the mean target entropy (source entropy is not minimised here).
pub fn da_entropy_loss(
    _source_probs: &[f32],
    n_src: usize,
    target_probs: &[f32],
    n_tgt: usize,
) -> Result<f32, DomainAdaptationError> {
    if n_src == 0 || n_tgt == 0 {
        return Err(DomainAdaptationError::EmptyFeatures);
    }
    if target_probs.is_empty() {
        return Err(DomainAdaptationError::EmptyFeatures);
    }
    let classes = target_probs.len() / n_tgt;
    if classes == 0 {
        return Err(DomainAdaptationError::InvalidConfig {
            reason: "target_probs length must be divisible by n_tgt".to_owned(),
        });
    }
    let total_entropy: f32 = (0..n_tgt)
        .map(|i| {
            let slice = &target_probs[i * classes..(i + 1) * classes];
            da_entropy(slice)
        })
        .sum();
    Ok(total_entropy / n_tgt as f32)
}

// ---------------------------------------------------------------------------
// Self-training: confidence threshold mask
// ---------------------------------------------------------------------------

/// Build a pseudo-label confidence mask.
///
/// `probs` has layout \[n × classes\] row-major. Each sample's mask entry is
/// `true` if `max(probs_for_sample) > threshold`. Returns a mask of length
/// `n`.
///
/// # Errors
///
/// Returns [`DomainAdaptationError::InvalidConfig`] if `n == 0` while
/// `probs` is non-empty, or if `probs.len()` is not evenly divisible by `n`
/// (so `classes` cannot be inferred).
pub fn da_confidence_threshold_mask(
    probs: &[f32],
    n: usize,
    threshold: f32,
) -> Result<Vec<bool>, DomainAdaptationError> {
    if probs.is_empty() {
        return Ok(Vec::new());
    }
    if n == 0 || !probs.len().is_multiple_of(n) {
        return Err(DomainAdaptationError::InvalidConfig {
            reason: format!(
                "probs length {} is not evenly divisible by n={n}",
                probs.len()
            ),
        });
    }
    let classes = probs.len() / n;
    Ok((0..n)
        .map(|i| {
            let row = &probs[i * classes..(i + 1) * classes];
            let max_p = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            max_p > threshold
        })
        .collect())
}

// ---------------------------------------------------------------------------
// Self-training: pseudo-label loss
// ---------------------------------------------------------------------------

/// Cross-entropy loss on confident target samples using pseudo-labels.
///
/// - `target_logits`: \[n × classes\] raw logits (row-major)
/// - `target_probs`:  \[n × classes\] softmax probabilities (row-major)
/// - Pseudo-label for sample i = argmax of `target_probs[i, :]`
/// - Only samples where `max(target_probs[i,:]) > confidence_threshold` contribute.
///
/// Returns 0.0 if no samples exceed the threshold.
pub fn da_pseudo_label_loss(
    target_logits: &[f32],
    target_probs: &[f32],
    n: usize,
    confidence_threshold: f32,
    eps: f32,
) -> Result<f32, DomainAdaptationError> {
    if n == 0 {
        return Err(DomainAdaptationError::EmptyFeatures);
    }
    if target_logits.len() != target_probs.len() {
        return Err(DomainAdaptationError::DimensionMismatch {
            src: target_logits.len(),
            tgt: target_probs.len(),
        });
    }
    let total_len = target_probs.len();
    if !total_len.is_multiple_of(n) {
        return Err(DomainAdaptationError::InvalidConfig {
            reason: format!("target_probs length {} not divisible by n={}", total_len, n),
        });
    }
    let classes = total_len / n;
    if classes == 0 {
        return Err(DomainAdaptationError::InvalidConfig {
            reason: "zero classes".to_owned(),
        });
    }

    let mut loss_sum = 0.0f32;
    let mut count = 0usize;

    for i in 0..n {
        let prob_slice = &target_probs[i * classes..(i + 1) * classes];
        let logit_slice = &target_logits[i * classes..(i + 1) * classes];

        // Find max prob and its index (pseudo-label)
        let (pseudo_label, max_prob) = prob_slice.iter().enumerate().fold(
            (0usize, f32::NEG_INFINITY),
            |(best_idx, best_val), (j, &p)| {
                if p > best_val {
                    (j, p)
                } else {
                    (best_idx, best_val)
                }
            },
        );

        if max_prob > confidence_threshold {
            // Compute log-softmax at pseudo_label
            let max_logit = logit_slice
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            let log_sum_exp: f32 = logit_slice
                .iter()
                .map(|&l| (l - max_logit).exp())
                .sum::<f32>()
                .ln()
                + max_logit;
            let log_prob = logit_slice[pseudo_label] - log_sum_exp;
            loss_sum += -(log_prob + eps);
            count += 1;
        }
    }

    if count == 0 {
        Ok(0.0)
    } else {
        Ok(loss_sum / count as f32)
    }
}
