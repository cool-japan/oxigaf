//! Latent-space style transfer for diffusion models.
//!
//! Provides purely arithmetic style-transfer operations that manipulate diffusion
//! model latents to transfer appearance (style) from one image/latent to another
//! while preserving structural content. All operations are pure Rust with no
//! neural-network forward passes.
//!
//! ## Key Operations
//!
//! - **AdaIN** (Adaptive Instance Normalization): channels statistics of a content
//!   latent are matched to those of a style latent — classic Huang & Belongie 2017.
//! - **Style interpolation**: smoothly blends content/style statistics at a
//!   user-controlled `content_weight`.
//! - **Whitening / Coloring**: separate the normalization (whiten) from the
//!   statistics injection (color) for more compositional pipelines.
//! - **Histogram matching**: per-channel CDF-based value remapping.
//! - **StylePalette**: manages a library of named styles and supports weighted
//!   style blending.
//!
//! ## Latent Format
//!
//! All functions expect *channel-first* flat layout:
//! channel `c` occupies indices `[c * spatial_size .. (c+1) * spatial_size]`
//! where `spatial_size = latent.len() / n_channels`.

use thiserror::Error;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors produced by the style-transfer module.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum StyleTransferError {
    /// Latent length is not divisible by `n_channels`, or two latents have
    /// incompatible sizes for a binary operation.
    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    /// A configuration parameter is invalid (e.g., weight out of range).
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// The latent vector contains no elements.
    #[error("Latent is empty")]
    EmptyLatent,

    /// A numerical operation failed (e.g., NaN/Inf produced).
    #[error("Numerical error: {0}")]
    NumericalError(String),
}

// ---------------------------------------------------------------------------
// Global latent statistics
// ---------------------------------------------------------------------------

/// Compute the global mean of a latent vector.
///
/// Returns `0.0` for an empty slice.
pub fn latent_mean(latent: &[f32]) -> f32 {
    if latent.is_empty() {
        return 0.0;
    }
    latent.iter().sum::<f32>() / latent.len() as f32
}

/// Compute the global variance of a latent vector (`Var = mean((x − mean)²`).
///
/// Returns `0.0` for an empty or single-element slice.
pub fn latent_variance(latent: &[f32]) -> f32 {
    if latent.len() <= 1 {
        return 0.0;
    }
    let mean = latent_mean(latent);
    latent.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / latent.len() as f32
}

// ---------------------------------------------------------------------------
// Channel-wise statistics
// ---------------------------------------------------------------------------

/// Compute per-channel mean and standard deviation for a channel-first latent.
///
/// `latent`: flat buffer with layout `[c * spatial_size .. (c+1) * spatial_size]`
/// for each channel `c`.
///
/// Returns a `Vec<(mean, std)>` of length `n_channels`.
///
/// # Errors
///
/// - [`StyleTransferError::EmptyLatent`] if `latent` is empty.
/// - [`StyleTransferError::DimensionMismatch`] if `latent.len() % n_channels != 0`.
pub fn channel_statistics(
    latent: &[f32],
    n_channels: usize,
) -> Result<Vec<(f32, f32)>, StyleTransferError> {
    if latent.is_empty() {
        return Err(StyleTransferError::EmptyLatent);
    }
    if n_channels == 0 || !latent.len().is_multiple_of(n_channels) {
        return Err(StyleTransferError::DimensionMismatch {
            expected: n_channels,
            got: latent.len() % n_channels.max(1),
        });
    }
    let spatial = latent.len() / n_channels;
    let mut stats = Vec::with_capacity(n_channels);
    for c in 0..n_channels {
        let slice = &latent[c * spatial..(c + 1) * spatial];
        let mean = slice.iter().sum::<f32>() / spatial as f32;
        let var = slice.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / spatial as f32;
        let std = var.sqrt();
        stats.push((mean, std));
    }
    Ok(stats)
}

/// Per-channel mean of a channel-first latent.
pub fn per_channel_mean(latent: &[f32], n_channels: usize) -> Result<Vec<f32>, StyleTransferError> {
    channel_statistics(latent, n_channels).map(|s| s.into_iter().map(|(m, _)| m).collect())
}

/// Per-channel standard deviation of a channel-first latent.
pub fn per_channel_std(latent: &[f32], n_channels: usize) -> Result<Vec<f32>, StyleTransferError> {
    channel_statistics(latent, n_channels).map(|s| s.into_iter().map(|(_, sd)| sd).collect())
}

// ---------------------------------------------------------------------------
// AdaIN (Adaptive Instance Normalization)
// ---------------------------------------------------------------------------

/// Apply Adaptive Instance Normalization.
///
/// The content latent is normalized per-channel to zero mean / unit variance and
/// then rescaled to match the style's per-channel statistics.
///
/// Formula per element in channel `c`:
/// ```text
/// output = std_s * (x − mean_c) / (std_c + 1e-5) + mean_s
/// ```
///
/// # Errors
///
/// - [`StyleTransferError::DimensionMismatch`] if `content.len() != style.len()`.
/// - Propagates errors from [`channel_statistics`].
pub fn adain(
    content: &[f32],
    style: &[f32],
    n_channels: usize,
) -> Result<Vec<f32>, StyleTransferError> {
    if content.is_empty() {
        return Err(StyleTransferError::EmptyLatent);
    }
    if content.len() != style.len() {
        return Err(StyleTransferError::DimensionMismatch {
            expected: content.len(),
            got: style.len(),
        });
    }
    let content_stats = channel_statistics(content, n_channels)?;
    let style_stats = channel_statistics(style, n_channels)?;
    let spatial = content.len() / n_channels;

    let mut output = vec![0.0f32; content.len()];
    for c in 0..n_channels {
        let (mean_c, std_c) = content_stats[c];
        let (mean_s, std_s) = style_stats[c];
        let base = c * spatial;
        for i in 0..spatial {
            let x = content[base + i];
            output[base + i] = std_s * (x - mean_c) / (std_c + 1e-5) + mean_s;
        }
    }
    Ok(output)
}

// ---------------------------------------------------------------------------
// Style interpolation
// ---------------------------------------------------------------------------

/// Configuration for style interpolation.
#[derive(Debug, Clone)]
pub struct StyleConfig {
    /// Weight for content (`0.0` = pure style, `1.0` = pure content).
    pub content_weight: f32,
    /// Epsilon for numerical stability in normalization.
    pub eps: f32,
    /// Number of channels in the latent.
    pub n_channels: usize,
}

impl Default for StyleConfig {
    fn default() -> Self {
        Self {
            content_weight: 0.5,
            eps: 1e-5,
            n_channels: 4,
        }
    }
}

impl StyleConfig {
    /// Validate all configuration parameters.
    ///
    /// # Errors
    ///
    /// Returns [`StyleTransferError::InvalidConfig`] if any parameter is invalid.
    pub fn validate(&self) -> Result<(), StyleTransferError> {
        if !(0.0..=1.0).contains(&self.content_weight) {
            return Err(StyleTransferError::InvalidConfig(format!(
                "content_weight must be in [0, 1], got {}",
                self.content_weight
            )));
        }
        if self.eps <= 0.0 {
            return Err(StyleTransferError::InvalidConfig(format!(
                "eps must be > 0, got {}",
                self.eps
            )));
        }
        if self.n_channels == 0 {
            return Err(StyleTransferError::InvalidConfig(
                "n_channels must be >= 1".to_string(),
            ));
        }
        Ok(())
    }
}

/// Interpolate between content and style in the statistics space.
///
/// For each channel, the target statistics are a weighted blend:
/// ```text
/// alpha        = 1 − content_weight
/// target_mean  = (1−alpha) * mean_c + alpha * mean_s
/// target_std   = (1−alpha) * std_c  + alpha * std_s
/// output[c*S+i] = target_std * (content[c*S+i] − mean_c) / (std_c + eps) + target_mean
/// ```
///
/// # Errors
///
/// - Propagates [`StyleConfig::validate`] errors.
/// - [`StyleTransferError::DimensionMismatch`] if `content.len() != style.len()`.
/// - Propagates errors from [`channel_statistics`].
pub fn interpolate_style(
    content: &[f32],
    style: &[f32],
    config: &StyleConfig,
) -> Result<Vec<f32>, StyleTransferError> {
    config.validate()?;
    if content.is_empty() {
        return Err(StyleTransferError::EmptyLatent);
    }
    if content.len() != style.len() {
        return Err(StyleTransferError::DimensionMismatch {
            expected: content.len(),
            got: style.len(),
        });
    }
    let content_stats = channel_statistics(content, config.n_channels)?;
    let style_stats = channel_statistics(style, config.n_channels)?;
    let spatial = content.len() / config.n_channels;
    let alpha = 1.0 - config.content_weight;

    let mut output = vec![0.0f32; content.len()];
    for c in 0..config.n_channels {
        let (mean_c, std_c) = content_stats[c];
        let (mean_s, std_s) = style_stats[c];
        let target_mean = (1.0 - alpha) * mean_c + alpha * mean_s;
        let target_std = (1.0 - alpha) * std_c + alpha * std_s;
        let base = c * spatial;
        for i in 0..spatial {
            let x = content[base + i];
            output[base + i] = target_std * (x - mean_c) / (std_c + config.eps) + target_mean;
        }
    }
    Ok(output)
}

// ---------------------------------------------------------------------------
// Whitening and coloring
// ---------------------------------------------------------------------------

/// Normalize a latent to zero mean and unit variance per channel.
///
/// # Errors
///
/// - [`StyleTransferError::EmptyLatent`] if `latent` is empty.
/// - [`StyleTransferError::DimensionMismatch`] if size is not divisible by `n_channels`.
pub fn whiten_latent(
    latent: &[f32],
    n_channels: usize,
    eps: f32,
) -> Result<Vec<f32>, StyleTransferError> {
    let stats = channel_statistics(latent, n_channels)?;
    let spatial = latent.len() / n_channels;
    let mut output = vec![0.0f32; latent.len()];
    for (c, &(mean, std)) in stats.iter().enumerate() {
        let base = c * spatial;
        for i in 0..spatial {
            output[base + i] = (latent[base + i] - mean) / (std + eps);
        }
    }
    Ok(output)
}

/// Apply target statistics to a whitened latent (inverse of whitening).
///
/// For each channel `c`:
/// ```text
/// output[c*S+i] = target_std[c] * whitened[c*S+i] + target_mean[c]
/// ```
///
/// # Errors
///
/// - [`StyleTransferError::EmptyLatent`] if `whitened` is empty.
/// - [`StyleTransferError::DimensionMismatch`] if `target_mean.len() != n_channels`
///   or `target_std.len() != n_channels`.
/// - [`StyleTransferError::DimensionMismatch`] if `whitened.len() % n_channels != 0`.
pub fn color_latent(
    whitened: &[f32],
    target_mean: &[f32],
    target_std: &[f32],
    n_channels: usize,
) -> Result<Vec<f32>, StyleTransferError> {
    if whitened.is_empty() {
        return Err(StyleTransferError::EmptyLatent);
    }
    if target_mean.len() != n_channels {
        return Err(StyleTransferError::DimensionMismatch {
            expected: n_channels,
            got: target_mean.len(),
        });
    }
    if target_std.len() != n_channels {
        return Err(StyleTransferError::DimensionMismatch {
            expected: n_channels,
            got: target_std.len(),
        });
    }
    if n_channels == 0 || !whitened.len().is_multiple_of(n_channels) {
        return Err(StyleTransferError::DimensionMismatch {
            expected: n_channels,
            got: whitened.len() % n_channels.max(1),
        });
    }
    let spatial = whitened.len() / n_channels;
    let mut output = vec![0.0f32; whitened.len()];
    for c in 0..n_channels {
        let base = c * spatial;
        for i in 0..spatial {
            output[base + i] = target_std[c] * whitened[base + i] + target_mean[c];
        }
    }
    Ok(output)
}

// ---------------------------------------------------------------------------
// Histogram matching
// ---------------------------------------------------------------------------

/// Approximate CDF-based histogram matching per channel.
///
/// For each channel the source values are sorted; each value is mapped to the
/// target percentile at the same normalized rank. When source and target have
/// different spatial sizes, linear interpolation is used to sample from the
/// sorted target array.
///
/// # Parameters
///
/// - `n_bins`: number of quantile bins (currently used as documentation;
///   the implementation always uses the full sort for maximum accuracy).
///
/// # Errors
///
/// - [`StyleTransferError::EmptyLatent`] if either slice is empty.
/// - [`StyleTransferError::DimensionMismatch`] if either slice is not divisible
///   by `n_channels`.
pub fn histogram_match(
    source: &[f32],
    target: &[f32],
    n_channels: usize,
    _n_bins: usize,
) -> Result<Vec<f32>, StyleTransferError> {
    if source.is_empty() || target.is_empty() {
        return Err(StyleTransferError::EmptyLatent);
    }
    if n_channels == 0 {
        return Err(StyleTransferError::DimensionMismatch {
            expected: 1,
            got: 0,
        });
    }
    if !source.len().is_multiple_of(n_channels) {
        return Err(StyleTransferError::DimensionMismatch {
            expected: n_channels,
            got: source.len() % n_channels,
        });
    }
    if !target.len().is_multiple_of(n_channels) {
        return Err(StyleTransferError::DimensionMismatch {
            expected: n_channels,
            got: target.len() % n_channels,
        });
    }

    let src_spatial = source.len() / n_channels;
    let tgt_spatial = target.len() / n_channels;
    let mut output = vec![0.0f32; source.len()];

    for c in 0..n_channels {
        let src_base = c * src_spatial;
        let tgt_base = c * tgt_spatial;

        // Collect (value, original_index) pairs for source
        let mut src_indexed: Vec<(f32, usize)> = source[src_base..src_base + src_spatial]
            .iter()
            .copied()
            .enumerate()
            .map(|(i, v)| (v, i))
            .collect();
        src_indexed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // Sort target channel
        let mut sorted_target: Vec<f32> = target[tgt_base..tgt_base + tgt_spatial].to_vec();
        sorted_target.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Map each source rank to the corresponding target percentile
        for (rank, &(_, orig_idx)) in src_indexed.iter().enumerate() {
            let mapped = if tgt_spatial == 1 {
                sorted_target[0]
            } else if src_spatial == 1 {
                // single source element → map to median of target
                let mid = (tgt_spatial - 1) as f32 / 2.0;
                let lo = mid.floor() as usize;
                let hi = lo + 1;
                let frac = mid - lo as f32;
                if hi < tgt_spatial {
                    sorted_target[lo] * (1.0 - frac) + sorted_target[hi] * frac
                } else {
                    sorted_target[lo]
                }
            } else {
                // rank_frac in [0, 1]; interpolate into sorted_target
                let rank_frac = rank as f32 / (src_spatial - 1) as f32;
                let tgt_pos = rank_frac * (tgt_spatial - 1) as f32;
                let lo = tgt_pos.floor() as usize;
                let hi = (lo + 1).min(tgt_spatial - 1);
                let frac = tgt_pos - lo as f32;
                sorted_target[lo] * (1.0 - frac) + sorted_target[hi] * frac
            };
            output[src_base + orig_idx] = mapped;
        }
    }
    Ok(output)
}

// ---------------------------------------------------------------------------
// StylePalette
// ---------------------------------------------------------------------------

/// A library of named style descriptors (per-channel mean/std pairs).
///
/// Styles are stored as compact `(means, stds)` vectors so that applying a
/// style to any size of content latent is O(N) regardless of how large the
/// original style images were.
#[derive(Debug, Clone)]
pub struct StylePalette {
    /// Number of latent channels expected by all stored styles.
    pub n_channels: usize,
    /// Per-style: `(means_per_channel, stds_per_channel)`.
    ///
    /// Every entry must have exactly `n_channels` elements in both vectors —
    /// [`Self::apply_style`] and [`Self::blend_styles`] index them
    /// unconditionally and will panic on a shorter entry. Prefer
    /// [`Self::push_style`] or [`Self::add_from_latent`], which validate the
    /// length before inserting, over pushing onto this field directly.
    pub styles: Vec<(Vec<f32>, Vec<f32>)>,
    /// Human-readable name for each style.
    pub names: Vec<String>,
}

impl StylePalette {
    /// Create an empty palette for latents with `n_channels` channels.
    pub fn new(n_channels: usize) -> Self {
        Self {
            n_channels,
            styles: Vec::new(),
            names: Vec::new(),
        }
    }

    /// Add a style extracted from a latent vector.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`channel_statistics`].
    pub fn add_from_latent(
        &mut self,
        latent: &[f32],
        name: impl Into<String>,
    ) -> Result<(), StyleTransferError> {
        let stats = channel_statistics(latent, self.n_channels)?;
        let means: Vec<f32> = stats.iter().map(|&(m, _)| m).collect();
        let stds: Vec<f32> = stats.iter().map(|&(_, s)| s).collect();
        self.push_style(means, stds, name)
    }

    /// Add a style directly from precomputed per-channel `(means, stds)`.
    ///
    /// Unlike pushing onto the public [`Self::styles`] field by hand, this
    /// validates both vectors have exactly `n_channels` entries before
    /// accepting them — a shorter vector would otherwise panic later inside
    /// [`Self::apply_style`] or [`Self::blend_styles`].
    ///
    /// # Errors
    ///
    /// [`StyleTransferError::DimensionMismatch`] if `means.len()` or
    /// `stds.len()` is not exactly `self.n_channels`.
    pub fn push_style(
        &mut self,
        means: Vec<f32>,
        stds: Vec<f32>,
        name: impl Into<String>,
    ) -> Result<(), StyleTransferError> {
        if means.len() != self.n_channels {
            return Err(StyleTransferError::DimensionMismatch {
                expected: self.n_channels,
                got: means.len(),
            });
        }
        if stds.len() != self.n_channels {
            return Err(StyleTransferError::DimensionMismatch {
                expected: self.n_channels,
                got: stds.len(),
            });
        }
        self.styles.push((means, stds));
        self.names.push(name.into());
        Ok(())
    }

    /// Apply style `style_idx` to `content` via AdaIN.
    ///
    /// # Errors
    ///
    /// - [`StyleTransferError::DimensionMismatch`] if `style_idx >= self.len()`.
    /// - Propagates errors from [`channel_statistics`].
    pub fn apply_style(
        &self,
        content: &[f32],
        style_idx: usize,
        eps: f32,
    ) -> Result<Vec<f32>, StyleTransferError> {
        if style_idx >= self.styles.len() {
            return Err(StyleTransferError::DimensionMismatch {
                expected: self.styles.len(),
                got: style_idx,
            });
        }
        if content.is_empty() {
            return Err(StyleTransferError::EmptyLatent);
        }
        let content_stats = channel_statistics(content, self.n_channels)?;
        let (target_means, target_stds) = &self.styles[style_idx];
        let spatial = content.len() / self.n_channels;
        let mut output = vec![0.0f32; content.len()];
        for c in 0..self.n_channels {
            let (mean_c, std_c) = content_stats[c];
            let base = c * spatial;
            for i in 0..spatial {
                let x = content[base + i];
                output[base + i] = target_stds[c] * (x - mean_c) / (std_c + eps) + target_means[c];
            }
        }
        Ok(output)
    }

    /// Blend multiple styles via a weighted mean of their per-channel statistics.
    ///
    /// `weights` must have the same length as `self.styles`. Weights need
    /// not already sum to 1 — they are normalised internally, so passing
    /// e.g. `[1.0, 1.0]` blends the two styles evenly rather than doubling
    /// the result.
    ///
    /// # Returns
    ///
    /// `(blended_means, blended_stds)` — each of length `n_channels`.
    ///
    /// # Errors
    ///
    /// - [`StyleTransferError::InvalidConfig`] if the palette is empty, or
    ///   if `weights` sums to a non-positive value (the normalisation would
    ///   divide by zero or flip signs).
    /// - [`StyleTransferError::DimensionMismatch`] if `weights.len() != self.styles.len()`.
    pub fn blend_styles(
        &self,
        weights: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>), StyleTransferError> {
        if self.styles.is_empty() {
            return Err(StyleTransferError::InvalidConfig(
                "style palette is empty".to_string(),
            ));
        }
        if weights.len() != self.styles.len() {
            return Err(StyleTransferError::DimensionMismatch {
                expected: self.styles.len(),
                got: weights.len(),
            });
        }
        let weight_sum: f32 = weights.iter().sum();
        // `weight_sum.is_nan() || weight_sum <= 0.0` (rather than the
        // clippy-flagged `!(weight_sum > 0.0)`) is deliberate, not just a
        // style swap: `NaN <= 0.0` is `false`, so a bare `<= 0.0` rewrite
        // would let a NaN sum (e.g. from a NaN weight) fall through to the
        // divide below and silently poison `blended_means`/`blended_stds`
        // with NaN. The explicit `is_nan()` check keeps NaN rejected, same
        // as the original negated-comparison form (`NaN > 0.0` is `false`,
        // so `!(NaN > 0.0)` was already `true` and errored).
        if weight_sum.is_nan() || weight_sum <= 0.0 {
            return Err(StyleTransferError::InvalidConfig(format!(
                "blend_styles weights must sum to a positive value, got {weight_sum}"
            )));
        }
        let mut blended_means = vec![0.0f32; self.n_channels];
        let mut blended_stds = vec![0.0f32; self.n_channels];
        for (w, (means, stds)) in weights.iter().zip(self.styles.iter()) {
            for c in 0..self.n_channels {
                blended_means[c] += w * means[c];
                blended_stds[c] += w * stds[c];
            }
        }
        for c in 0..self.n_channels {
            blended_means[c] /= weight_sum;
            blended_stds[c] /= weight_sum;
        }
        Ok((blended_means, blended_stds))
    }

    /// Number of styles in the palette.
    pub fn len(&self) -> usize {
        self.styles.len()
    }

    /// Returns `true` if the palette contains no styles.
    pub fn is_empty(&self) -> bool {
        self.styles.is_empty()
    }
}

// ---------------------------------------------------------------------------
// StyleTransferStats
// ---------------------------------------------------------------------------

/// Quality metrics for a style-transfer operation.
#[derive(Debug, Clone)]
pub struct StyleTransferStats {
    /// Per-element L2 distance between the content and output latents.
    pub content_deviation: f32,
    /// Mean absolute per-channel mean difference between output and style.
    pub style_mean_error: f32,
    /// Mean absolute per-channel std difference between output and style.
    pub style_std_error: f32,
    /// Style fidelity: `1 / (1 + style_mean_error + style_std_error)`.
    pub style_fidelity: f32,
}

/// Compute style-transfer quality statistics.
///
/// # Errors
///
/// Propagates errors from [`channel_statistics`] and checks for empty slices.
pub fn compute_style_stats(
    content: &[f32],
    style: &[f32],
    output: &[f32],
    n_channels: usize,
) -> Result<StyleTransferStats, StyleTransferError> {
    if content.is_empty() || style.is_empty() || output.is_empty() {
        return Err(StyleTransferError::EmptyLatent);
    }
    if content.len() != output.len() {
        return Err(StyleTransferError::DimensionMismatch {
            expected: content.len(),
            got: output.len(),
        });
    }

    // content_deviation = RMSE(content, output), matching the "per-element
    // L2 distance" doc on the field: sqrt(mean((c - o)^2)), not the bare
    // mean-squared-error (which has squared units and under-reports for any
    // deviation smaller than 1.0).
    let mse = content
        .iter()
        .zip(output.iter())
        .map(|(&c, &o)| (c - o).powi(2))
        .sum::<f32>()
        / content.len() as f32;
    let content_deviation = mse.sqrt();

    // Per-channel stats for output and style
    let out_stats = channel_statistics(output, n_channels)?;
    let sty_stats = channel_statistics(style, n_channels)?;

    let style_mean_error = out_stats
        .iter()
        .zip(sty_stats.iter())
        .map(|(&(om, _), &(sm, _))| (om - sm).abs())
        .sum::<f32>()
        / n_channels as f32;

    let style_std_error = out_stats
        .iter()
        .zip(sty_stats.iter())
        .map(|(&(_, os), &(_, ss))| (os - ss).abs())
        .sum::<f32>()
        / n_channels as f32;

    let style_fidelity = 1.0 / (1.0 + style_mean_error + style_std_error);

    Ok(StyleTransferStats {
        content_deviation,
        style_mean_error,
        style_std_error,
        style_fidelity,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- helpers -----------------------------------------------------------

    /// Generate a linearly spaced latent: [start, start+step, …] of length `len`.
    fn linspace(start: f32, end: f32, len: usize) -> Vec<f32> {
        if len == 0 {
            return vec![];
        }
        if len == 1 {
            return vec![start];
        }
        (0..len)
            .map(|i| start + (end - start) * i as f32 / (len - 1) as f32)
            .collect()
    }

    /// Constant-value latent.
    fn const_latent(val: f32, len: usize) -> Vec<f32> {
        vec![val; len]
    }

    // ---- latent_mean / latent_variance ------------------------------------

    #[test]
    fn test_latent_mean_basic() {
        let v = vec![1.0, 2.0, 3.0, 4.0];
        let m = latent_mean(&v);
        assert!((m - 2.5).abs() < 1e-6, "mean={m}");
    }

    #[test]
    fn test_latent_mean_empty() {
        assert_eq!(latent_mean(&[]), 0.0);
    }

    #[test]
    fn test_latent_variance_basic() {
        // values: [0, 1, 2, 3], mean=1.5, var = mean of [2.25,0.25,0.25,2.25] = 1.25
        let v = vec![0.0, 1.0, 2.0, 3.0];
        let var = latent_variance(&v);
        assert!((var - 1.25).abs() < 1e-5, "variance={var}");
    }

    #[test]
    fn test_latent_variance_single() {
        assert_eq!(latent_variance(&[42.0]), 0.0);
    }

    #[test]
    fn test_latent_variance_empty() {
        assert_eq!(latent_variance(&[]), 0.0);
    }

    // ---- channel_statistics ------------------------------------------------

    #[test]
    fn test_channel_statistics_wrong_n_channels_error() {
        // 7 elements is not divisible by 3
        let v = vec![1.0f32; 7];
        let result = channel_statistics(&v, 3);
        assert!(
            matches!(result, Err(StyleTransferError::DimensionMismatch { .. })),
            "expected DimensionMismatch, got {result:?}"
        );
    }

    #[test]
    fn test_channel_statistics_single_channel_constant() {
        // All same → std should be 0
        let v = const_latent(3.0, 8);
        let stats = channel_statistics(&v, 1).unwrap();
        let (mean, std) = stats[0];
        assert!((mean - 3.0).abs() < 1e-6);
        assert!(
            std.abs() < 1e-6,
            "std for constant channel should be 0, got {std}"
        );
    }

    #[test]
    fn test_channel_statistics_two_channels() {
        // 2 channels, spatial = 4
        // ch0 = [1,1,1,1], ch1 = [2,3,4,5]
        let mut v = vec![1.0f32; 4]; // ch0
        v.extend_from_slice(&[2.0, 3.0, 4.0, 5.0]); // ch1
        let stats = channel_statistics(&v, 2).unwrap();
        let (m0, s0) = stats[0];
        let (m1, s1) = stats[1];
        assert!((m0 - 1.0).abs() < 1e-5);
        assert!(s0.abs() < 1e-5);
        // ch1: mean=3.5, var = (2.25+0.25+0.25+2.25)/4 = 1.25, std≈1.118
        assert!((m1 - 3.5).abs() < 1e-5, "m1={m1}");
        assert!((s1 - 1.25f32.sqrt()).abs() < 1e-4, "s1={s1}");
    }

    // ---- per_channel_mean / per_channel_std --------------------------------

    #[test]
    fn test_per_channel_mean_basic() {
        let mut v = vec![2.0f32; 4]; // ch0 all 2.0
        v.extend(vec![4.0f32; 4]); // ch1 all 4.0
        let means = per_channel_mean(&v, 2).unwrap();
        assert!((means[0] - 2.0).abs() < 1e-6);
        assert!((means[1] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_per_channel_std_basic() {
        let v = linspace(0.0, 7.0, 8); // 8 elements, 2 channels
        let stds = per_channel_std(&v, 2).unwrap();
        assert_eq!(stds.len(), 2);
        // Both stds should be positive for linearly spaced data
        assert!(stds[0] > 0.0);
        assert!(stds[1] > 0.0);
    }

    // ---- adain -------------------------------------------------------------

    #[test]
    fn test_adain_same_content_and_style_identity() {
        // AdaIN(x, x) should reproduce x (up to fp precision)
        let v: Vec<f32> = linspace(-2.0, 2.0, 16);
        let out = adain(&v, &v, 4).unwrap();
        for (a, b) in v.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1e-4, "a={a} b={b}");
        }
    }

    #[test]
    fn test_adain_output_stats_match_style() {
        // After AdaIN, output channel stats should match style stats
        let content: Vec<f32> = linspace(-1.0, 1.0, 16);
        let style: Vec<f32> = linspace(5.0, 9.0, 16);
        let out = adain(&content, &style, 4).unwrap();

        let out_stats = channel_statistics(&out, 4).unwrap();
        let sty_stats = channel_statistics(&style, 4).unwrap();
        for c in 0..4 {
            let (om, os) = out_stats[c];
            let (sm, ss) = sty_stats[c];
            assert!((om - sm).abs() < 1e-4, "ch{c} mean: out={om} style={sm}");
            assert!((os - ss).abs() < 1e-4, "ch{c} std:  out={os} style={ss}");
        }
    }

    #[test]
    fn test_adain_dimension_mismatch() {
        let a = vec![1.0f32; 16];
        let b = vec![1.0f32; 12];
        let result = adain(&a, &b, 4);
        assert!(matches!(
            result,
            Err(StyleTransferError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_adain_different_content_and_style() {
        // Content≠style: output channel stats should match style, not content
        let content: Vec<f32> = linspace(0.0, 1.0, 16);
        let style: Vec<f32> = linspace(10.0, 20.0, 16);
        let out = adain(&content, &style, 4).unwrap();

        let out_stats = channel_statistics(&out, 4).unwrap();
        let sty_stats = channel_statistics(&style, 4).unwrap();
        for c in 0..4 {
            let (om, _) = out_stats[c];
            let (sm, _) = sty_stats[c];
            assert!((om - sm).abs() < 1e-3, "ch{c} mean mismatch: {om} vs {sm}");
        }
    }

    // ---- StyleConfig::validate ---------------------------------------------

    #[test]
    fn test_style_config_validate_valid() {
        let cfg = StyleConfig::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_style_config_validate_bad_content_weight() {
        let cfg = StyleConfig {
            content_weight: 1.5,
            ..Default::default()
        };
        assert!(matches!(
            cfg.validate(),
            Err(StyleTransferError::InvalidConfig(_))
        ));
    }

    #[test]
    fn test_style_config_validate_bad_eps() {
        let cfg = StyleConfig {
            eps: -1e-5,
            ..Default::default()
        };
        assert!(matches!(
            cfg.validate(),
            Err(StyleTransferError::InvalidConfig(_))
        ));
    }

    #[test]
    fn test_style_config_validate_bad_n_channels() {
        let cfg = StyleConfig {
            n_channels: 0,
            ..Default::default()
        };
        assert!(matches!(
            cfg.validate(),
            Err(StyleTransferError::InvalidConfig(_))
        ));
    }

    // ---- interpolate_style -------------------------------------------------

    #[test]
    fn test_interpolate_style_content_weight_one_preserves_content_stats() {
        let content: Vec<f32> = linspace(0.0, 4.0, 16);
        let style: Vec<f32> = linspace(10.0, 20.0, 16);
        let cfg = StyleConfig {
            content_weight: 1.0,
            n_channels: 4,
            ..Default::default()
        };
        let out = interpolate_style(&content, &style, &cfg).unwrap();
        let out_stats = channel_statistics(&out, 4).unwrap();
        let content_stats = channel_statistics(&content, 4).unwrap();
        for c in 0..4 {
            let (om, os) = out_stats[c];
            let (cm, cs) = content_stats[c];
            assert!((om - cm).abs() < 1e-4, "ch{c} mean: {om} vs content {cm}");
            assert!((os - cs).abs() < 1e-4, "ch{c} std:  {os} vs content {cs}");
        }
    }

    #[test]
    fn test_interpolate_style_content_weight_zero_preserves_style_stats() {
        let content: Vec<f32> = linspace(0.0, 4.0, 16);
        let style: Vec<f32> = linspace(10.0, 20.0, 16);
        let cfg = StyleConfig {
            content_weight: 0.0,
            n_channels: 4,
            ..Default::default()
        };
        let out = interpolate_style(&content, &style, &cfg).unwrap();
        let out_stats = channel_statistics(&out, 4).unwrap();
        let sty_stats = channel_statistics(&style, 4).unwrap();
        for c in 0..4 {
            let (om, os) = out_stats[c];
            let (sm, ss) = sty_stats[c];
            assert!((om - sm).abs() < 1e-4, "ch{c} mean: {om} vs style {sm}");
            assert!((os - ss).abs() < 1e-4, "ch{c} std:  {os} vs style {ss}");
        }
    }

    // ---- whiten_latent -----------------------------------------------------

    #[test]
    fn test_whiten_latent_output_zero_mean_unit_std() {
        let v: Vec<f32> = linspace(-3.0, 3.0, 16);
        let whitened = whiten_latent(&v, 4, 1e-5).unwrap();
        let stats = channel_statistics(&whitened, 4).unwrap();
        for (c, &(mean, std)) in stats.iter().enumerate() {
            assert!(mean.abs() < 1e-5, "ch{c} mean after whitening: {mean}");
            // std should be ~1.0 (as long as the channel isn't constant)
            assert!((std - 1.0).abs() < 1e-4, "ch{c} std after whitening: {std}");
        }
    }

    // ---- color_latent -------------------------------------------------------

    #[test]
    fn test_color_latent_known_stats() {
        // Whitened = all zeros → coloring to (mean=5, std=2) → all 5s
        let whitened = vec![0.0f32; 8]; // 2 channels × 4 spatial
        let means = vec![5.0f32; 2];
        let stds = vec![2.0f32; 2];
        let out = color_latent(&whitened, &means, &stds, 2).unwrap();
        for &v in &out {
            assert!((v - 5.0).abs() < 1e-6, "expected 5.0, got {v}");
        }
    }

    #[test]
    fn test_color_latent_dimension_mismatch_target_mean() {
        let whitened = vec![0.0f32; 8];
        let means = vec![1.0f32; 3]; // wrong: should be 2
        let stds = vec![1.0f32; 2];
        let result = color_latent(&whitened, &means, &stds, 2);
        assert!(matches!(
            result,
            Err(StyleTransferError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_color_latent_dimension_mismatch_target_std() {
        let whitened = vec![0.0f32; 8];
        let means = vec![1.0f32; 2];
        let stds = vec![1.0f32; 5]; // wrong
        let result = color_latent(&whitened, &means, &stds, 2);
        assert!(matches!(
            result,
            Err(StyleTransferError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_color_latent_roundtrip() {
        // whiten then color → recover original (approximately)
        let v: Vec<f32> = linspace(1.0, 5.0, 8);
        let orig_stats = channel_statistics(&v, 2).unwrap();
        let target_means: Vec<f32> = orig_stats.iter().map(|&(m, _)| m).collect();
        let target_stds: Vec<f32> = orig_stats.iter().map(|&(_, s)| s).collect();
        let whitened = whiten_latent(&v, 2, 1e-5).unwrap();
        let recovered = color_latent(&whitened, &target_means, &target_stds, 2).unwrap();
        for (a, b) in v.iter().zip(recovered.iter()) {
            assert!((a - b).abs() < 1e-4, "a={a} b={b}");
        }
    }

    // ---- histogram_match ---------------------------------------------------

    #[test]
    fn test_histogram_match_same_latent_identity() {
        let v: Vec<f32> = linspace(0.0, 7.0, 8);
        let out = histogram_match(&v, &v, 2, 256).unwrap();
        // Sorted output values should match sorted source (which is the same as target)
        let mut sorted_out = out.clone();
        sorted_out.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mut sorted_v = v.clone();
        sorted_v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        for (a, b) in sorted_out.iter().zip(sorted_v.iter()) {
            assert!((a - b).abs() < 1e-5, "a={a} b={b}");
        }
    }

    #[test]
    fn test_histogram_match_output_sorted_matches_target() {
        // After matching, sorted output values should ≈ sorted target values
        let source: Vec<f32> = linspace(0.0, 1.0, 8);
        let target: Vec<f32> = linspace(10.0, 20.0, 8);
        let out = histogram_match(&source, &target, 2, 256).unwrap();
        let mut sorted_out = out.clone();
        sorted_out.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mut sorted_tgt = target.clone();
        sorted_tgt.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        for (a, b) in sorted_out.iter().zip(sorted_tgt.iter()) {
            assert!((a - b).abs() < 1e-4, "a={a} b={b}");
        }
    }

    #[test]
    fn test_histogram_match_dimension_mismatch_channels() {
        // 7 elements is not divisible by 2
        let source = vec![1.0f32; 7];
        let target = vec![1.0f32; 8];
        let result = histogram_match(&source, &target, 2, 256);
        assert!(matches!(
            result,
            Err(StyleTransferError::DimensionMismatch { .. })
        ));
    }

    // ---- StylePalette -------------------------------------------------------

    #[test]
    fn test_style_palette_new_empty() {
        let palette = StylePalette::new(4);
        assert_eq!(palette.len(), 0);
        assert!(palette.is_empty());
    }

    #[test]
    fn test_style_palette_add_from_latent() {
        let mut palette = StylePalette::new(4);
        let latent: Vec<f32> = linspace(0.0, 15.0, 16);
        palette.add_from_latent(&latent, "style_a").unwrap();
        assert_eq!(palette.len(), 1);
        assert!(!palette.is_empty());
        assert_eq!(palette.names[0], "style_a");
    }

    #[test]
    fn test_style_palette_apply_style_known() {
        let mut palette = StylePalette::new(4);
        let style_latent: Vec<f32> = const_latent(5.0, 16);
        palette.add_from_latent(&style_latent, "flat").unwrap();

        let content: Vec<f32> = linspace(0.0, 3.0, 16);
        let out = palette.apply_style(&content, 0, 1e-5).unwrap();

        // Style mean per channel should be 5.0; output means should be ≈ 5.0
        let out_stats = channel_statistics(&out, 4).unwrap();
        for (c, &(om, _)) in out_stats.iter().enumerate() {
            assert!((om - 5.0).abs() < 1e-4, "ch{c} mean={om}");
        }
    }

    #[test]
    fn test_style_palette_blend_styles_single_weight_one() {
        let mut palette = StylePalette::new(4);
        let latent: Vec<f32> = linspace(2.0, 10.0, 16);
        palette.add_from_latent(&latent, "s1").unwrap();

        let (bm, bs) = palette.blend_styles(&[1.0]).unwrap();
        let original_stats = channel_statistics(&latent, 4).unwrap();
        for c in 0..4 {
            let (om, os) = original_stats[c];
            assert!(
                (bm[c] - om).abs() < 1e-5,
                "ch{c} mean: {bm_c} vs {om}",
                bm_c = bm[c]
            );
            assert!(
                (bs[c] - os).abs() < 1e-5,
                "ch{c} std:  {bs_c} vs {os}",
                bs_c = bs[c]
            );
        }
    }

    /// Regression test: `blend_styles([1.0, 1.0])` must average the two
    /// styles, not sum them. Previously the weighted accumulation was never
    /// divided by the weight sum, so equal unit weights doubled both the
    /// blended means and stds instead of averaging them.
    #[test]
    fn test_style_palette_blend_styles_equal_weights_averages_not_sums() {
        let mut palette = StylePalette::new(1);
        palette.push_style(vec![2.0], vec![4.0], "a").unwrap();
        palette.push_style(vec![6.0], vec![8.0], "b").unwrap();

        let (bm, bs) = palette.blend_styles(&[1.0, 1.0]).unwrap();
        assert!(
            (bm[0] - 4.0).abs() < 1e-5,
            "mean should be the average (2+6)/2=4, got {}",
            bm[0]
        );
        assert!(
            (bs[0] - 6.0).abs() < 1e-5,
            "std should be the average (4+8)/2=6, got {}",
            bs[0]
        );
    }

    #[test]
    fn test_style_palette_blend_styles_nonpositive_weight_sum_err() {
        let mut palette = StylePalette::new(1);
        palette.push_style(vec![2.0], vec![4.0], "a").unwrap();
        palette.push_style(vec![6.0], vec![8.0], "b").unwrap();

        let result = palette.blend_styles(&[1.0, -1.0]);
        assert!(matches!(result, Err(StyleTransferError::InvalidConfig(_))));
    }

    #[test]
    fn test_style_palette_blend_styles_nan_weight_sum_err() {
        // Regression for the clippy `neg_cmp_op_on_partial_ord` fix: the
        // check used to read `!(weight_sum > 0.0)`. The naive rewrite
        // `weight_sum <= 0.0` is NOT behavior-preserving for NaN, because
        // `NaN <= 0.0` is `false` in IEEE 754 — it would let a NaN weight
        // sum fall through to the divide and silently poison
        // `blended_means`/`blended_stds` with NaN instead of erroring. A NaN
        // weight sum must still be rejected, exactly as it was before.
        let mut palette = StylePalette::new(1);
        palette.push_style(vec![2.0], vec![4.0], "a").unwrap();
        palette.push_style(vec![6.0], vec![8.0], "b").unwrap();

        let result = palette.blend_styles(&[f32::NAN, 1.0]);
        assert!(
            matches!(result, Err(StyleTransferError::InvalidConfig(_))),
            "a NaN weight sum must be rejected, not silently propagated into \
             the blended output: {result:?}"
        );
    }

    #[test]
    fn test_style_palette_push_style_wrong_length_err() {
        let mut palette = StylePalette::new(4);
        let result = palette.push_style(vec![1.0, 2.0], vec![1.0, 2.0, 3.0, 4.0], "bad");
        assert!(matches!(
            result,
            Err(StyleTransferError::DimensionMismatch {
                expected: 4,
                got: 2
            })
        ));
    }

    #[test]
    fn test_style_palette_blend_styles_wrong_weights_length() {
        let mut palette = StylePalette::new(4);
        let latent: Vec<f32> = linspace(0.0, 15.0, 16);
        palette.add_from_latent(&latent, "s1").unwrap();
        // 2 weights for 1 style
        let result = palette.blend_styles(&[0.5, 0.5]);
        assert!(matches!(
            result,
            Err(StyleTransferError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_style_palette_blend_styles_empty_palette() {
        let palette = StylePalette::new(4);
        let result = palette.blend_styles(&[1.0]);
        assert!(matches!(result, Err(StyleTransferError::InvalidConfig(_))));
    }

    // ---- compute_style_stats -----------------------------------------------

    #[test]
    fn test_compute_style_stats_content_equals_output_zero_deviation() {
        let v: Vec<f32> = linspace(0.0, 7.0, 8);
        let style: Vec<f32> = linspace(5.0, 12.0, 8);
        let stats = compute_style_stats(&v, &style, &v, 2).unwrap();
        assert!(
            stats.content_deviation.abs() < 1e-6,
            "deviation={}",
            stats.content_deviation
        );
    }

    /// Regression test: `content_deviation` is documented as a per-element
    /// L2 distance (RMSE), not a mean squared error. A constant offset of 3
    /// between every element of `content` and `output` must report 3.0, not
    /// 9.0 (which is what the un-rooted mean-squared-error would give).
    #[test]
    fn test_compute_style_stats_content_deviation_is_rmse_not_mse() {
        let content = vec![0.0f32; 8];
        let output = vec![3.0f32; 8];
        let style: Vec<f32> = linspace(5.0, 12.0, 8);
        let stats = compute_style_stats(&content, &style, &output, 2).unwrap();
        assert!(
            (stats.content_deviation - 3.0).abs() < 1e-5,
            "expected RMSE=3.0 (constant offset), got {} (9.0 would indicate \
             an un-rooted MSE)",
            stats.content_deviation
        );
    }

    #[test]
    fn test_compute_style_stats_perfect_style_match_fidelity() {
        // When output == style, style_mean_error and style_std_error are both 0
        // so fidelity = 1/(1+0+0) = 1.0
        let content: Vec<f32> = linspace(0.0, 7.0, 8);
        let style: Vec<f32> = linspace(5.0, 12.0, 8);
        let stats = compute_style_stats(&content, &style, &style, 2).unwrap();
        assert!(
            (stats.style_fidelity - 1.0).abs() < 1e-5,
            "fidelity={}",
            stats.style_fidelity
        );
    }

    #[test]
    fn test_compute_style_stats_fields_non_negative() {
        let content: Vec<f32> = linspace(0.0, 3.0, 16);
        let style: Vec<f32> = linspace(5.0, 9.0, 16);
        let out = adain(&content, &style, 4).unwrap();
        let stats = compute_style_stats(&content, &style, &out, 4).unwrap();
        assert!(stats.content_deviation >= 0.0);
        assert!(stats.style_mean_error >= 0.0);
        assert!(stats.style_std_error >= 0.0);
        assert!(stats.style_fidelity > 0.0 && stats.style_fidelity <= 1.0);
    }

    #[test]
    fn test_compute_style_stats_adain_output_high_fidelity() {
        // AdaIN output should have near-perfect style fidelity
        let content: Vec<f32> = linspace(0.0, 3.0, 16);
        let style: Vec<f32> = linspace(5.0, 9.0, 16);
        let out = adain(&content, &style, 4).unwrap();
        let stats = compute_style_stats(&content, &style, &out, 4).unwrap();
        // fidelity should be close to 1.0 since adain exactly matches style stats
        assert!(
            stats.style_fidelity > 0.95,
            "expected high fidelity, got {}",
            stats.style_fidelity
        );
    }
}
