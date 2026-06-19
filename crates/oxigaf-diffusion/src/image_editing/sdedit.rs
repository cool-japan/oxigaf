//! # sdedit
//!
//! SDEdit-style latent image editing utilities using `&[f32]` slice-based API.
//!
//! Provides:
//! - Noise schedule computation (cosine and linear alpha-bar schedules)
//! - SDEdit noise perturbation (forward diffusion to a target timestep)
//! - Mask-based region editing (blend, expand channels)
//! - Latent manipulation (lerp, slerp, distance, similarity, projection)
//! - Collection statistics (mean, variance map)
//! - Edit statistics summary
//!
//! ## PRNG
//! All stochastic functions use xorshift64 + Box-Muller. No `rand` crate.
//!
//! ## Layout convention
//! Latents are flat `Vec<f32>` in row-major order. Mask data is `[H × W]`.

use super::EditMask;
use thiserror::Error;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors produced by SDEdit-style image editing operations.
#[derive(Debug, Error)]
pub enum ImageEditError {
    /// Dimension mismatch between operands.
    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    /// Strength value is outside the valid `(0, 1]` range.
    #[error("Strength {0} must be in (0, 1]")]
    InvalidStrength(f32),

    /// Timestep is outside the valid `[0, max]` range.
    #[error("Timestep {t} out of range [0, {max}]")]
    TimestepOutOfRange { t: usize, max: usize },

    /// Mask is structurally invalid.
    #[error("Invalid mask: {0}")]
    InvalidMask(String),

    /// Empty input was provided where at least one element is required.
    #[error("Empty input")]
    EmptyInput,

    /// Invalid parameter value.
    #[error("Invalid parameter: {0}")]
    InvalidParam(String),
}

// ---------------------------------------------------------------------------
// Edit configuration
// ---------------------------------------------------------------------------

/// Configuration for an SDEdit-style editing operation.
#[derive(Debug, Clone)]
pub struct EditConfig {
    /// Strength controls how much noise is added.
    /// `0.0` = no edit, `1.0` = full denoise from noise.
    pub strength: f32,
    /// Total number of diffusion timesteps.
    pub n_timesteps: usize,
    /// Guidance scale for classifier-free guidance.
    pub guidance_scale: f32,
    /// Whether to use mask-based editing.
    pub use_mask: bool,
}

impl Default for EditConfig {
    fn default() -> Self {
        Self {
            strength: 0.7,
            n_timesteps: 1000,
            guidance_scale: 7.5,
            use_mask: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Inline xorshift64 PRNG + Box-Muller
// ---------------------------------------------------------------------------

/// Advance one xorshift64 step, returning the next state.
/// Enforces non-zero invariant after the shifts.
#[inline]
fn edit_xorshift64(state: &mut u64) -> u64 {
    (*state) ^= (*state) << 13;
    (*state) ^= (*state) >> 7;
    (*state) ^= (*state) << 17;
    if *state == 0 {
        *state = 1;
    }
    *state
}

/// Convert a xorshift64 sample to `f32` in `[0, 1)`.
#[inline]
fn edit_xorshift_f32(state: &mut u64) -> f32 {
    (edit_xorshift64(state) >> 11) as f32 / (1u64 << 53) as f32
}

/// Box-Muller transform from two uniform samples in `(0, 1)`.
/// Returns a pair of standard-normal samples.
#[inline]
fn edit_box_muller(u1: f32, u2: f32) -> (f32, f32) {
    let r = (-2.0_f32 * u1.max(1e-10).ln()).sqrt();
    let theta = 2.0 * std::f32::consts::PI * u2;
    (r * theta.cos(), r * theta.sin())
}

// ---------------------------------------------------------------------------
// Timestep and schedule functions
// ---------------------------------------------------------------------------

/// Compute the starting timestep for SDEdit from strength.
///
/// `t_start = floor(n_timesteps * strength)`
///
/// Returns `InvalidStrength` if `strength` is not in `(0, 1]`.
pub fn edit_start_timestep(strength: f32, n_timesteps: usize) -> Result<usize, ImageEditError> {
    if strength <= 0.0 || strength > 1.0 {
        return Err(ImageEditError::InvalidStrength(strength));
    }
    Ok((n_timesteps as f32 * strength).floor() as usize)
}

/// Compute the cosine noise schedule alpha-bar values.
///
/// Formula (Nichol & Dhariwal, 2021):
/// ```text
/// s = 0.008
/// f(t) = cos((t/T + s) / (1 + s) * PI/2)^2
/// alpha_bar[t] = f(t) / f(0), clamped to [0.001, 0.999]
/// ```
///
/// Returns a `Vec<f32>` of length `n_timesteps`.
pub fn edit_cosine_alpha_bars(n_timesteps: usize) -> Vec<f32> {
    let s = 0.008_f32;
    let denom = 1.0 + s;
    // f(0) = cos(s / (1+s) * PI/2)^2
    let f0 = {
        let angle = (s / denom) * std::f32::consts::FRAC_PI_2;
        angle.cos() * angle.cos()
    };

    (0..n_timesteps)
        .map(|t| {
            let angle = ((t as f32 / n_timesteps as f32) + s) / denom * std::f32::consts::FRAC_PI_2;
            let ft = angle.cos() * angle.cos();
            (ft / f0).clamp(0.001, 0.999)
        })
        .collect()
}

/// Compute the linear noise schedule alpha-bar values.
///
/// `beta[t] = beta_start + (beta_end - beta_start) * t / (n_timesteps - 1)`
/// `alpha_bar[t] = product(1 - beta[i], i = 0..=t)`
///
/// Returns a `Vec<f32>` of length `n_timesteps`.
pub fn edit_linear_alpha_bars(n_timesteps: usize, beta_start: f32, beta_end: f32) -> Vec<f32> {
    if n_timesteps == 0 {
        return Vec::new();
    }

    let mut alpha_bars = Vec::with_capacity(n_timesteps);
    let mut running_alpha_bar = 1.0f32;

    for t in 0..n_timesteps {
        let beta = if n_timesteps > 1 {
            beta_start + (beta_end - beta_start) * t as f32 / (n_timesteps - 1) as f32
        } else {
            beta_start
        };
        let alpha = 1.0 - beta;
        running_alpha_bar *= alpha;
        alpha_bars.push(running_alpha_bar);
    }

    alpha_bars
}

// ---------------------------------------------------------------------------
// Noise addition
// ---------------------------------------------------------------------------

/// Add noise at a specific diffusion timestep (forward diffusion).
///
/// `x_noisy = sqrt(alpha_bar[t]) * x + sqrt(1 - alpha_bar[t]) * noise`
///
/// Returns `DimensionMismatch` if `latent` and `noise` have different lengths.
/// Returns `TimestepOutOfRange` if `timestep >= alpha_bars.len()`.
pub fn edit_add_noise(
    latent: &[f32],
    noise: &[f32],
    timestep: usize,
    alpha_bars: &[f32],
) -> Result<Vec<f32>, ImageEditError> {
    if latent.is_empty() {
        return Err(ImageEditError::EmptyInput);
    }
    if noise.len() != latent.len() {
        return Err(ImageEditError::DimensionMismatch {
            expected: latent.len(),
            got: noise.len(),
        });
    }
    if alpha_bars.is_empty() || timestep >= alpha_bars.len() {
        return Err(ImageEditError::TimestepOutOfRange {
            t: timestep,
            max: alpha_bars.len().saturating_sub(1),
        });
    }

    let ab = alpha_bars[timestep].clamp(0.0, 1.0);
    let sqrt_ab = ab.sqrt();
    let sqrt_one_minus_ab = (1.0 - ab).sqrt();

    let result: Vec<f32> = latent
        .iter()
        .zip(noise.iter())
        .map(|(&x, &z)| sqrt_ab * x + sqrt_one_minus_ab * z)
        .collect();

    Ok(result)
}

/// Generate `n` independent standard-normal noise samples using xorshift64 + Box-Muller.
///
/// `state` is the PRNG seed / state (modified in place).
pub fn edit_sample_noise(n: usize, state: &mut u64) -> Vec<f32> {
    if *state == 0 {
        *state = 1;
    }
    let mut out = Vec::with_capacity(n);
    let pairs = n / 2;
    for _ in 0..pairs {
        let u1 = edit_xorshift_f32(state).max(1e-10);
        let u2 = edit_xorshift_f32(state);
        let (z0, z1) = edit_box_muller(u1, u2);
        out.push(z0);
        out.push(z1);
    }
    if n % 2 == 1 {
        let u1 = edit_xorshift_f32(state).max(1e-10);
        let u2 = edit_xorshift_f32(state);
        let (z0, _) = edit_box_muller(u1, u2);
        out.push(z0);
    }
    out
}

// ---------------------------------------------------------------------------
// SDEdit perturbation
// ---------------------------------------------------------------------------

/// Apply SDEdit: add noise at the strength-controlled starting timestep.
///
/// 1. Compute `t_start = floor(n_timesteps * strength)`.
/// 2. Sample Gaussian noise.
/// 3. Apply `edit_add_noise` at `t_start`.
///
/// Returns `(noisy_latent, timestep_used)`.
pub fn sdedit_perturb(
    latent: &[f32],
    config: &EditConfig,
    alpha_bars: &[f32],
    state: &mut u64,
) -> Result<(Vec<f32>, usize), ImageEditError> {
    if latent.is_empty() {
        return Err(ImageEditError::EmptyInput);
    }
    let t_start = edit_start_timestep(config.strength, config.n_timesteps)?;
    let noise = edit_sample_noise(latent.len(), state);
    let noisy = edit_add_noise(latent, &noise, t_start, alpha_bars)?;
    Ok((noisy, t_start))
}

// ---------------------------------------------------------------------------
// Mask-based editing
// ---------------------------------------------------------------------------

/// Blend edited and original latents using a spatial mask (slice-based API).
///
/// `result[i] = mask_expanded[i] * edited[i] + (1 - mask_expanded[i]) * original[i]`
///
/// The mask is spatial `(H x W)` and is applied per-channel to match the
/// latent size. Both `original` and `edited` must have length equal to
/// `mask.height * mask.width * channels` for some integer `channels`.
pub fn edit_blend_with_mask(
    original: &[f32],
    edited: &[f32],
    mask: &EditMask,
) -> Result<Vec<f32>, ImageEditError> {
    if original.is_empty() {
        return Err(ImageEditError::EmptyInput);
    }
    if original.len() != edited.len() {
        return Err(ImageEditError::DimensionMismatch {
            expected: original.len(),
            got: edited.len(),
        });
    }

    let hw = mask.height * mask.width;
    if hw == 0 {
        return Err(ImageEditError::InvalidMask(
            "mask has zero spatial extent".to_string(),
        ));
    }
    if !original.len().is_multiple_of(hw) {
        return Err(ImageEditError::DimensionMismatch {
            expected: original.len(),
            got: hw,
        });
    }

    let channels = original.len() / hw;
    let mut result = Vec::with_capacity(original.len());

    for c in 0..channels {
        for hw_idx in 0..hw {
            let m = mask.data[hw_idx].clamp(0.0, 1.0);
            let o = original[c * hw + hw_idx];
            let e = edited[c * hw + hw_idx];
            result.push(m * e + (1.0 - m) * o);
        }
    }

    Ok(result)
}

/// Expand a spatial mask `(H x W)` to a latent `(H x W x C)` by repeating per channel.
///
/// Output layout is channels-first: for each channel `c`, the `H x W` mask values
/// are written contiguously.
pub fn edit_expand_mask_to_channels(mask: &EditMask, channels: usize) -> Vec<f32> {
    let hw = mask.height * mask.width;
    let mut out = Vec::with_capacity(hw * channels);
    for _ in 0..channels {
        out.extend_from_slice(&mask.data);
    }
    out
}

// ---------------------------------------------------------------------------
// Latent manipulation utilities
// ---------------------------------------------------------------------------

/// Linear interpolation between two latents.
///
/// `result = alpha * a + (1 - alpha) * b`
///
/// `alpha = 1.0` returns `a`; `alpha = 0.0` returns `b`.
pub fn edit_lerp_latents(a: &[f32], b: &[f32], alpha: f32) -> Result<Vec<f32>, ImageEditError> {
    if a.is_empty() {
        return Err(ImageEditError::EmptyInput);
    }
    if a.len() != b.len() {
        return Err(ImageEditError::DimensionMismatch {
            expected: a.len(),
            got: b.len(),
        });
    }
    let result: Vec<f32> = a
        .iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| alpha * ai + (1.0 - alpha) * bi)
        .collect();
    Ok(result)
}

/// Spherical linear interpolation (slerp) between two latents.
///
/// Treats each latent as a vector on the unit hypersphere.
/// Falls back to linear interpolation when the vectors are nearly parallel
/// (`dot > 0.9999` after normalization).
pub fn edit_slerp_latents(a: &[f32], b: &[f32], alpha: f32) -> Result<Vec<f32>, ImageEditError> {
    if a.is_empty() {
        return Err(ImageEditError::EmptyInput);
    }
    if a.len() != b.len() {
        return Err(ImageEditError::DimensionMismatch {
            expected: a.len(),
            got: b.len(),
        });
    }

    let norm_a = a.iter().fold(0.0f32, |acc, &x| acc + x * x).sqrt();
    let norm_b = b.iter().fold(0.0f32, |acc, &x| acc + x * x).sqrt();

    // If either vector is zero-norm, fall back to lerp.
    if norm_a < 1e-12 || norm_b < 1e-12 {
        return edit_lerp_latents(a, b, alpha);
    }

    let dot: f32 = a
        .iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| (ai / norm_a) * (bi / norm_b))
        .sum();

    if dot > 0.9999 {
        // Nearly parallel - linear interpolation is stable.
        return edit_lerp_latents(a, b, alpha);
    }

    let theta = dot.clamp(-1.0, 1.0).acos();
    let sin_theta = theta.sin();

    let scale_a = ((1.0 - alpha) * theta).sin() / sin_theta;
    let scale_b = (alpha * theta).sin() / sin_theta;

    let result: Vec<f32> = a
        .iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| scale_a * ai + scale_b * bi)
        .collect();

    Ok(result)
}

/// Compute the L2 (Euclidean) distance between two latents.
pub fn edit_latent_distance(a: &[f32], b: &[f32]) -> Result<f32, ImageEditError> {
    if a.is_empty() {
        return Err(ImageEditError::EmptyInput);
    }
    if a.len() != b.len() {
        return Err(ImageEditError::DimensionMismatch {
            expected: a.len(),
            got: b.len(),
        });
    }
    let sq_sum: f32 = a
        .iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| {
            let d = ai - bi;
            d * d
        })
        .sum();
    Ok(sq_sum.sqrt())
}

/// Compute the cosine similarity between two latents.
///
/// Returns `0.0` when either input is zero-norm.
pub fn edit_cosine_similarity(a: &[f32], b: &[f32]) -> Result<f32, ImageEditError> {
    if a.is_empty() {
        return Err(ImageEditError::EmptyInput);
    }
    if a.len() != b.len() {
        return Err(ImageEditError::DimensionMismatch {
            expected: a.len(),
            got: b.len(),
        });
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(&ai, &bi)| ai * bi).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();

    if norm_a < 1e-12 || norm_b < 1e-12 {
        return Ok(0.0);
    }

    Ok((dot / (norm_a * norm_b)).clamp(-1.0, 1.0))
}

/// Normalize a latent to unit L2 norm.
///
/// Returns `InvalidParam` when the latent is zero-norm.
pub fn edit_normalize_latent(latent: &[f32]) -> Result<Vec<f32>, ImageEditError> {
    if latent.is_empty() {
        return Err(ImageEditError::EmptyInput);
    }
    let norm: f32 = latent.iter().map(|&x| x * x).sum::<f32>().sqrt();
    if norm < 1e-12 {
        return Err(ImageEditError::InvalidParam(
            "cannot normalize a zero-norm latent".to_string(),
        ));
    }
    let result: Vec<f32> = latent.iter().map(|&x| x / norm).collect();
    Ok(result)
}

/// Project latent `x` onto the subspace perpendicular to `direction`.
///
/// `result = x - dot(x, direction) * direction`
///
/// `direction` must be a unit vector; if not, results are mathematically
/// valid but not strictly an orthogonal projection.
pub fn edit_project_out(x: &[f32], direction: &[f32]) -> Result<Vec<f32>, ImageEditError> {
    if x.is_empty() {
        return Err(ImageEditError::EmptyInput);
    }
    if x.len() != direction.len() {
        return Err(ImageEditError::DimensionMismatch {
            expected: x.len(),
            got: direction.len(),
        });
    }
    let proj: f32 = x
        .iter()
        .zip(direction.iter())
        .map(|(&xi, &di)| xi * di)
        .sum();
    let result: Vec<f32> = x
        .iter()
        .zip(direction.iter())
        .map(|(&xi, &di)| xi - proj * di)
        .collect();
    Ok(result)
}

/// Compute the per-element mean across a collection of latents.
///
/// All latents must have the same length. Returns `EmptyInput` for an
/// empty collection or zero-length latents.
pub fn edit_mean_latent(latents: &[Vec<f32>]) -> Result<Vec<f32>, ImageEditError> {
    if latents.is_empty() {
        return Err(ImageEditError::EmptyInput);
    }
    let n = latents[0].len();
    if n == 0 {
        return Err(ImageEditError::EmptyInput);
    }
    for lat in latents {
        if lat.len() != n {
            return Err(ImageEditError::DimensionMismatch {
                expected: n,
                got: lat.len(),
            });
        }
    }

    let count = latents.len() as f32;
    let mut mean = vec![0.0f32; n];
    for lat in latents {
        for (m, &v) in mean.iter_mut().zip(lat.iter()) {
            *m += v;
        }
    }
    for m in &mut mean {
        *m /= count;
    }
    Ok(mean)
}

/// Compute the per-element variance across a collection of latents.
///
/// Uses the two-pass algorithm: first compute the mean, then accumulate
/// squared deviations. All latents must have the same length.
pub fn edit_variance_map(latents: &[Vec<f32>]) -> Result<Vec<f32>, ImageEditError> {
    if latents.is_empty() {
        return Err(ImageEditError::EmptyInput);
    }
    let n = latents[0].len();
    if n == 0 {
        return Err(ImageEditError::EmptyInput);
    }
    for lat in latents {
        if lat.len() != n {
            return Err(ImageEditError::DimensionMismatch {
                expected: n,
                got: lat.len(),
            });
        }
    }

    let mean = edit_mean_latent(latents)?;
    let count = latents.len() as f32;
    let mut variance = vec![0.0f32; n];

    for lat in latents {
        for (v, (&l, &m)) in variance.iter_mut().zip(lat.iter().zip(mean.iter())) {
            let diff = l - m;
            *v += diff * diff;
        }
    }
    for v in &mut variance {
        *v /= count;
    }
    Ok(variance)
}

// ---------------------------------------------------------------------------
// Editing statistics
// ---------------------------------------------------------------------------

/// Summary statistics for a single SDEdit editing operation.
#[derive(Debug, Clone)]
pub struct SdeditStats {
    /// Estimated noise level: RMS difference between noisy and original latent.
    pub noise_level: f32,
    /// Strength value actually used.
    pub edit_strength: f32,
    /// Fraction of pixels being edited (from mask, or 1.0 if no mask).
    pub mask_coverage: f32,
    /// L2 norm of the original latent.
    pub latent_norm: f32,
    /// L2 norm of the noisy latent.
    pub noisy_norm: f32,
}

/// Compute statistics comparing `original` to `noisy`.
///
/// - `noise_level`: sqrt of mean squared difference between `noisy` and `original`.
/// - `mask_coverage`: `mask.coverage()` if a mask is provided, else `1.0`.
/// - `latent_norm` / `noisy_norm`: L2 norms.
pub fn compute_sdedit_stats(
    original: &[f32],
    noisy: &[f32],
    mask: Option<&EditMask>,
    edit_strength: f32,
) -> Result<SdeditStats, ImageEditError> {
    if original.is_empty() {
        return Err(ImageEditError::EmptyInput);
    }
    if original.len() != noisy.len() {
        return Err(ImageEditError::DimensionMismatch {
            expected: original.len(),
            got: noisy.len(),
        });
    }

    let n = original.len() as f32;

    let noise_level = {
        let mse: f32 = original
            .iter()
            .zip(noisy.iter())
            .map(|(&o, &nv)| {
                let d = nv - o;
                d * d
            })
            .sum::<f32>()
            / n;
        mse.sqrt()
    };

    let latent_norm = original.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let noisy_norm = noisy.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let mask_coverage = mask.map_or(1.0, |m| m.coverage());

    Ok(SdeditStats {
        noise_level,
        edit_strength,
        mask_coverage,
        latent_norm,
        noisy_norm,
    })
}

/// Format edit statistics into a human-readable summary string.
pub fn format_sdedit_stats(stats: &SdeditStats) -> String {
    format!(
        "SdeditStats {{ noise_level: {:.4}, edit_strength: {:.4}, mask_coverage: {:.4}, latent_norm: {:.4}, noisy_norm: {:.4} }}",
        stats.noise_level,
        stats.edit_strength,
        stats.mask_coverage,
        stats.latent_norm,
        stats.noisy_norm,
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_latent_val(n: usize, val: f32) -> Vec<f32> {
        vec![val; n]
    }

    fn make_latent_range(n: usize) -> Vec<f32> {
        (0..n).map(|i| i as f32).collect()
    }

    fn make_mask(width: usize, height: usize, fill: f32) -> EditMask {
        EditMask::new(height, width, fill)
    }

    // -----------------------------------------------------------------------
    // edit_start_timestep
    // -----------------------------------------------------------------------

    #[test]
    fn start_timestep_strength_one() {
        let t = edit_start_timestep(1.0, 1000).unwrap();
        assert_eq!(t, 1000);
    }

    #[test]
    fn start_timestep_strength_half() {
        let t = edit_start_timestep(0.5, 1000).unwrap();
        assert_eq!(t, 500);
    }

    #[test]
    fn start_timestep_small_strength() {
        let t = edit_start_timestep(0.1, 100).unwrap();
        assert_eq!(t, 10);
    }

    #[test]
    fn start_timestep_invalid_zero() {
        let err = edit_start_timestep(0.0, 1000).unwrap_err();
        assert!(matches!(err, ImageEditError::InvalidStrength(_)));
    }

    #[test]
    fn start_timestep_invalid_above_one() {
        let err = edit_start_timestep(1.5, 1000).unwrap_err();
        assert!(matches!(err, ImageEditError::InvalidStrength(_)));
    }

    #[test]
    fn start_timestep_invalid_negative() {
        let err = edit_start_timestep(-0.1, 1000).unwrap_err();
        assert!(matches!(err, ImageEditError::InvalidStrength(_)));
    }

    // -----------------------------------------------------------------------
    // edit_cosine_alpha_bars
    // -----------------------------------------------------------------------

    #[test]
    fn cosine_alpha_bars_length() {
        let ab = edit_cosine_alpha_bars(1000);
        assert_eq!(ab.len(), 1000);
    }

    #[test]
    fn cosine_alpha_bars_first_near_one() {
        let ab = edit_cosine_alpha_bars(1000);
        assert!(
            ab[0] >= 0.99,
            "alpha_bar[0] should be near 1.0, got {}",
            ab[0]
        );
    }

    #[test]
    fn cosine_alpha_bars_last_near_zero() {
        let ab = edit_cosine_alpha_bars(1000);
        assert!(
            ab[999] <= 0.01,
            "alpha_bar[999] should be near 0.0, got {}",
            ab[999]
        );
    }

    #[test]
    fn cosine_alpha_bars_monotonically_decreasing() {
        let ab = edit_cosine_alpha_bars(1000);
        for i in 1..ab.len() {
            assert!(
                ab[i] <= ab[i - 1],
                "alpha_bar not monotonically decreasing at index {}: {} > {}",
                i,
                ab[i],
                ab[i - 1]
            );
        }
    }

    #[test]
    fn cosine_alpha_bars_clamped() {
        let ab = edit_cosine_alpha_bars(1000);
        for &v in &ab {
            assert!(
                (0.001..=0.999).contains(&v),
                "alpha_bar out of clamp range: {}",
                v
            );
        }
    }

    // -----------------------------------------------------------------------
    // edit_linear_alpha_bars
    // -----------------------------------------------------------------------

    #[test]
    fn linear_alpha_bars_length() {
        let ab = edit_linear_alpha_bars(1000, 0.0001, 0.02);
        assert_eq!(ab.len(), 1000);
    }

    #[test]
    fn linear_alpha_bars_first_near_one() {
        let ab = edit_linear_alpha_bars(1000, 0.0001, 0.02);
        assert!(ab[0] > 0.999);
    }

    #[test]
    fn linear_alpha_bars_monotonically_decreasing() {
        let ab = edit_linear_alpha_bars(1000, 0.0001, 0.02);
        for i in 1..ab.len() {
            assert!(
                ab[i] <= ab[i - 1],
                "linear alpha_bar not monotone at {}: {} > {}",
                i,
                ab[i],
                ab[i - 1]
            );
        }
    }

    #[test]
    fn linear_alpha_bars_single_timestep() {
        let ab = edit_linear_alpha_bars(1, 0.5, 0.5);
        assert_eq!(ab.len(), 1);
        assert!((ab[0] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn linear_alpha_bars_empty() {
        let ab = edit_linear_alpha_bars(0, 0.0001, 0.02);
        assert!(ab.is_empty());
    }

    // -----------------------------------------------------------------------
    // edit_add_noise
    // -----------------------------------------------------------------------

    #[test]
    fn add_noise_t0_near_pure_latent() {
        let ab = edit_cosine_alpha_bars(1000);
        let latent = make_latent_val(16, 1.0);
        let noise = make_latent_val(16, 0.0);
        // At t=0, ab[0] is near 0.999, so result ~ sqrt(0.999) * 1.0 ~ 0.9995
        let result = edit_add_noise(&latent, &noise, 0, &ab).unwrap();
        for &v in &result {
            assert!(
                v > 0.99,
                "at t=0 result should be near latent value, got {}",
                v
            );
        }
    }

    #[test]
    fn add_noise_t_max_dominated_by_noise() {
        let ab = edit_cosine_alpha_bars(1000);
        let latent = make_latent_val(16, 0.0);
        let noise = make_latent_val(16, 1.0);
        // At t=999, ab[999] is near 0.001, so result ~ sqrt(1-0.001)*1.0 ~ 0.9995
        let result = edit_add_noise(&latent, &noise, 999, &ab).unwrap();
        for &v in &result {
            assert!(v > 0.99, "at t=999 noise should dominate, got {}", v);
        }
    }

    #[test]
    fn add_noise_dimension_mismatch() {
        let ab = edit_cosine_alpha_bars(100);
        let latent = make_latent_val(16, 0.0);
        let noise = make_latent_val(8, 0.0);
        let err = edit_add_noise(&latent, &noise, 0, &ab).unwrap_err();
        assert!(matches!(err, ImageEditError::DimensionMismatch { .. }));
    }

    #[test]
    fn add_noise_timestep_out_of_range() {
        let ab = edit_cosine_alpha_bars(10);
        let latent = make_latent_val(8, 1.0);
        let noise = make_latent_val(8, 0.0);
        let err = edit_add_noise(&latent, &noise, 20, &ab).unwrap_err();
        assert!(matches!(err, ImageEditError::TimestepOutOfRange { .. }));
    }

    #[test]
    fn add_noise_empty_input() {
        let ab = edit_cosine_alpha_bars(100);
        let err = edit_add_noise(&[], &[], 0, &ab).unwrap_err();
        assert!(matches!(err, ImageEditError::EmptyInput));
    }

    // -----------------------------------------------------------------------
    // edit_sample_noise
    // -----------------------------------------------------------------------

    #[test]
    fn sample_noise_correct_length() {
        let mut state = 42u64;
        let noise = edit_sample_noise(100, &mut state);
        assert_eq!(noise.len(), 100);
    }

    #[test]
    fn sample_noise_odd_length() {
        let mut state = 17u64;
        let noise = edit_sample_noise(101, &mut state);
        assert_eq!(noise.len(), 101);
    }

    #[test]
    fn sample_noise_roughly_zero_mean() {
        let mut state = 12345u64;
        let noise = edit_sample_noise(10000, &mut state);
        let mean: f32 = noise.iter().sum::<f32>() / noise.len() as f32;
        assert!(mean.abs() < 0.1, "mean far from 0: {}", mean);
    }

    #[test]
    fn sample_noise_different_seeds() {
        let mut s1 = 1u64;
        let mut s2 = 2u64;
        let n1 = edit_sample_noise(20, &mut s1);
        let n2 = edit_sample_noise(20, &mut s2);
        let any_diff = n1.iter().zip(n2.iter()).any(|(a, b)| (a - b).abs() > 1e-10);
        assert!(any_diff, "different seeds should produce different noise");
    }

    // -----------------------------------------------------------------------
    // sdedit_perturb
    // -----------------------------------------------------------------------

    #[test]
    fn sdedit_perturb_returns_correct_timestep() {
        let ab = edit_cosine_alpha_bars(1000);
        let latent = make_latent_val(16, 0.5);
        let config = EditConfig {
            strength: 0.7,
            n_timesteps: 1000,
            ..Default::default()
        };
        let mut state = 42u64;
        let (_, t) = sdedit_perturb(&latent, &config, &ab, &mut state).unwrap();
        assert_eq!(t, 700);
    }

    #[test]
    fn sdedit_perturb_noisy_differs_from_original() {
        let ab = edit_cosine_alpha_bars(1000);
        let latent = make_latent_val(64, 1.0);
        let config = EditConfig {
            strength: 0.5,
            n_timesteps: 1000,
            ..Default::default()
        };
        let mut state = 99u64;
        let (noisy, _) = sdedit_perturb(&latent, &config, &ab, &mut state).unwrap();
        let any_diff = latent
            .iter()
            .zip(noisy.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(any_diff, "noisy latent should differ from original");
    }

    #[test]
    fn sdedit_perturb_same_length() {
        let ab = edit_cosine_alpha_bars(1000);
        let latent = make_latent_val(32, 0.0);
        let config = EditConfig::default();
        let mut state = 7u64;
        let (noisy, _) = sdedit_perturb(&latent, &config, &ab, &mut state).unwrap();
        assert_eq!(noisy.len(), latent.len());
    }

    // -----------------------------------------------------------------------
    // EditMask methods (new ones added to mod.rs)
    // -----------------------------------------------------------------------

    #[test]
    fn mask_new_all_zeros() {
        let m = EditMask::new(4, 4, 0.0);
        assert!(m.data.iter().all(|&v| v == 0.0));
        assert_eq!(m.data.len(), 16);
    }

    #[test]
    fn mask_all_ones_constructor() {
        let m = EditMask::all_ones(4, 4);
        assert!(m.data.iter().all(|&v| (v - 1.0).abs() < 1e-6));
        assert_eq!(m.data.len(), 16);
    }

    #[test]
    fn mask_from_data_invalid_size() {
        let result = EditMask::from_data(3, 3, vec![0.0; 5]);
        assert!(result.is_err());
    }

    #[test]
    fn mask_invert_ones_become_zeros() {
        let m = EditMask::all_ones(4, 4);
        let inv = m.invert();
        assert!(inv.data.iter().all(|&v| v.abs() < 1e-6));
    }

    #[test]
    fn mask_invert_zeros_become_ones() {
        let m = EditMask::new(4, 4, 0.0);
        let inv = m.invert();
        assert!(inv.data.iter().all(|&v| (v - 1.0).abs() < 1e-6));
    }

    #[test]
    fn mask_dilate_single_pixel_spreads() {
        let mut data = vec![0.0f32; 25];
        data[12] = 1.0; // centre of 5x5
        let m = EditMask::from_data(5, 5, data).unwrap();
        let d = m.dilate(1);
        // 3x3 neighbourhood of centre should be 1.0
        for r in 1..=3 {
            for c in 1..=3 {
                assert_eq!(d.data[r * 5 + c], 1.0);
            }
        }
        assert_eq!(d.data[0], 0.0); // corner unchanged
    }

    #[test]
    fn mask_erode_border_shrinks() {
        // Set 3x3 interior of a 5x5 grid to 1.0; erode with radius=1 should
        // shrink border pixels that are adjacent to 0.0 regions.
        let mut data = vec![0.0f32; 25];
        for r in 1..=3 {
            for c in 1..=3 {
                data[r * 5 + c] = 1.0;
            }
        }
        let m = EditMask::from_data(5, 5, data).unwrap();
        let e = m.erode(1);
        // The centre pixel (2,2) should still be 1.0 (surrounded by 1.0s).
        assert_eq!(e.data[2 * 5 + 2], 1.0);
        // The top-left corner (0,0) was 0.0 and stays 0.0.
        assert_eq!(e.data[0], 0.0);
        // Pixel (1,1): neighbourhood includes (0,0)=0.0 so min=0.0.
        assert_eq!(e.data[5 + 1], 0.0);
    }

    #[test]
    fn mask_coverage_all_ones() {
        let m = EditMask::all_ones(4, 4);
        assert!((m.coverage() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn mask_coverage_all_zeros() {
        let m = EditMask::new(4, 4, 0.0);
        assert!(m.coverage().abs() < 1e-6);
    }

    #[test]
    fn mask_coverage_half() {
        let data: Vec<f32> = (0..16).map(|i| if i < 8 { 1.0 } else { 0.0 }).collect();
        let m = EditMask::from_data(4, 4, data).unwrap();
        assert!((m.coverage() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn mask_smooth_all_ones_stays_one() {
        let m = EditMask::all_ones(8, 8);
        let s = m.smooth(1.0);
        assert!(s.data.iter().all(|&v| (v - 1.0).abs() < 1e-5));
    }

    #[test]
    fn mask_smooth_all_zeros_stays_zero() {
        let m = EditMask::new(8, 8, 0.0);
        let s = m.smooth(1.0);
        assert!(s.data.iter().all(|&v| v.abs() < 1e-5));
    }

    // -----------------------------------------------------------------------
    // edit_blend_with_mask
    // -----------------------------------------------------------------------

    #[test]
    fn blend_mask_all_ones_gives_edited() {
        let orig = make_latent_val(16, 0.0);
        let edited = make_latent_val(16, 5.0);
        let mask = make_mask(4, 4, 1.0);
        let result = edit_blend_with_mask(&orig, &edited, &mask).unwrap();
        assert!(result.iter().all(|&v| (v - 5.0).abs() < 1e-5));
    }

    #[test]
    fn blend_mask_all_zeros_gives_original() {
        let orig = make_latent_val(16, 3.0);
        let edited = make_latent_val(16, 9.0);
        let mask = make_mask(4, 4, 0.0);
        let result = edit_blend_with_mask(&orig, &edited, &mask).unwrap();
        assert!(result.iter().all(|&v| (v - 3.0).abs() < 1e-5));
    }

    #[test]
    fn blend_mask_half_gives_midpoint() {
        let orig = make_latent_val(1, 0.0);
        let edited = make_latent_val(1, 10.0);
        let mask = make_mask(1, 1, 0.5);
        let result = edit_blend_with_mask(&orig, &edited, &mask).unwrap();
        assert!((result[0] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn blend_mask_dim_mismatch() {
        let orig = make_latent_val(16, 0.0);
        let edited = make_latent_val(8, 0.0);
        let mask = make_mask(4, 4, 1.0);
        assert!(edit_blend_with_mask(&orig, &edited, &mask).is_err());
    }

    // -----------------------------------------------------------------------
    // edit_expand_mask_to_channels
    // -----------------------------------------------------------------------

    #[test]
    fn expand_mask_correct_size() {
        let mask = make_mask(4, 4, 1.0);
        let expanded = edit_expand_mask_to_channels(&mask, 3);
        assert_eq!(expanded.len(), 4 * 4 * 3);
    }

    #[test]
    fn expand_mask_values_preserved() {
        let data: Vec<f32> = (0..4).map(|i| i as f32 * 0.1).collect();
        let mask = EditMask::from_data(2, 2, data.clone()).unwrap();
        let expanded = edit_expand_mask_to_channels(&mask, 2);
        assert_eq!(expanded.len(), 8);
        for i in 0..4 {
            assert!((expanded[i] - data[i]).abs() < 1e-6);
        }
        for i in 0..4 {
            assert!((expanded[4 + i] - data[i]).abs() < 1e-6);
        }
    }

    // -----------------------------------------------------------------------
    // edit_lerp_latents
    // -----------------------------------------------------------------------

    #[test]
    fn lerp_alpha_zero_gives_b() {
        let a = make_latent_val(8, 1.0);
        let b = make_latent_val(8, 3.0);
        let result = edit_lerp_latents(&a, &b, 0.0).unwrap();
        assert!(result.iter().all(|&v| (v - 3.0).abs() < 1e-5));
    }

    #[test]
    fn lerp_alpha_one_gives_a() {
        let a = make_latent_val(8, 1.0);
        let b = make_latent_val(8, 3.0);
        let result = edit_lerp_latents(&a, &b, 1.0).unwrap();
        assert!(result.iter().all(|&v| (v - 1.0).abs() < 1e-5));
    }

    #[test]
    fn lerp_alpha_half_gives_midpoint() {
        let a = make_latent_val(1, 0.0);
        let b = make_latent_val(1, 10.0);
        let result = edit_lerp_latents(&a, &b, 0.5).unwrap();
        assert!((result[0] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn lerp_dimension_mismatch() {
        let a = make_latent_val(8, 1.0);
        let b = make_latent_val(4, 1.0);
        assert!(edit_lerp_latents(&a, &b, 0.5).is_err());
    }

    // -----------------------------------------------------------------------
    // edit_slerp_latents
    // -----------------------------------------------------------------------

    #[test]
    fn slerp_nearly_parallel_like_lerp() {
        let a: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0];
        let b: Vec<f32> = vec![1.0, 1e-8, 0.0, 0.0];
        let lerp_result = edit_lerp_latents(&a, &b, 0.5).unwrap();
        let slerp_result = edit_slerp_latents(&a, &b, 0.5).unwrap();
        for (l, s) in lerp_result.iter().zip(slerp_result.iter()) {
            assert!((l - s).abs() < 1e-4, "slerp vs lerp: {} vs {}", l, s);
        }
    }

    #[test]
    fn slerp_orthogonal_vectors() {
        let a = vec![1.0f32, 0.0, 0.0, 0.0];
        let b = vec![0.0f32, 1.0, 0.0, 0.0];
        let result = edit_slerp_latents(&a, &b, 0.5).unwrap();
        let expected = (0.5f32).sqrt();
        assert!((result[0] - expected).abs() < 1e-4);
        assert!((result[1] - expected).abs() < 1e-4);
    }

    #[test]
    fn slerp_dimension_mismatch() {
        let a = make_latent_val(8, 1.0);
        let b = make_latent_val(4, 1.0);
        assert!(edit_slerp_latents(&a, &b, 0.5).is_err());
    }

    // -----------------------------------------------------------------------
    // edit_latent_distance
    // -----------------------------------------------------------------------

    #[test]
    fn latent_distance_identical_zero() {
        let a = make_latent_range(16);
        let dist = edit_latent_distance(&a, &a).unwrap();
        assert!(dist.abs() < 1e-5);
    }

    #[test]
    fn latent_distance_known_value() {
        let a = vec![0.0f32; 4];
        let b = vec![3.0f32; 4]; // 4 elements each diff by 3 -> sum=36 -> sqrt=6
        let dist = edit_latent_distance(&a, &b).unwrap();
        assert!((dist - 6.0).abs() < 1e-5);
    }

    #[test]
    fn latent_distance_mismatch() {
        let a = make_latent_val(8, 1.0);
        let b = make_latent_val(4, 1.0);
        assert!(edit_latent_distance(&a, &b).is_err());
    }

    // -----------------------------------------------------------------------
    // edit_cosine_similarity
    // -----------------------------------------------------------------------

    #[test]
    fn cosine_sim_identical_is_one() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let sim = edit_cosine_similarity(&a, &a).unwrap();
        assert!((sim - 1.0).abs() < 1e-5);
    }

    #[test]
    fn cosine_sim_orthogonal_is_zero() {
        let a = vec![1.0f32, 0.0];
        let b = vec![0.0f32, 1.0];
        let sim = edit_cosine_similarity(&a, &b).unwrap();
        assert!(sim.abs() < 1e-5);
    }

    #[test]
    fn cosine_sim_zero_norm_returns_zero() {
        let a = vec![0.0f32; 4];
        let b = vec![1.0f32, 0.0, 0.0, 0.0];
        let sim = edit_cosine_similarity(&a, &b).unwrap();
        assert!(sim.abs() < 1e-5);
    }

    #[test]
    fn cosine_sim_mismatch() {
        let a = make_latent_val(8, 1.0);
        let b = make_latent_val(4, 1.0);
        assert!(edit_cosine_similarity(&a, &b).is_err());
    }

    // -----------------------------------------------------------------------
    // edit_normalize_latent
    // -----------------------------------------------------------------------

    #[test]
    fn normalize_unit_norm() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let n = edit_normalize_latent(&a).unwrap();
        let norm: f32 = n.iter().map(|&x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn normalize_zero_norm_error() {
        let a = vec![0.0f32; 4];
        assert!(edit_normalize_latent(&a).is_err());
    }

    #[test]
    fn normalize_single_element() {
        let a = vec![5.0f32];
        let n = edit_normalize_latent(&a).unwrap();
        assert!((n[0] - 1.0).abs() < 1e-5);
    }

    // -----------------------------------------------------------------------
    // edit_project_out
    // -----------------------------------------------------------------------

    #[test]
    fn project_out_orthogonal_to_direction() {
        let x = vec![1.0f32, 1.0, 0.0, 0.0];
        let d = vec![1.0f32, 0.0, 0.0, 0.0]; // unit vector along first axis
        let result = edit_project_out(&x, &d).unwrap();
        let dot: f32 = result.iter().zip(d.iter()).map(|(&r, &di)| r * di).sum();
        assert!(
            dot.abs() < 1e-5,
            "result not orthogonal to direction: dot={}",
            dot
        );
        assert!((result[0]).abs() < 1e-5);
        assert!((result[1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn project_out_mismatch() {
        let x = make_latent_val(8, 1.0);
        let d = make_latent_val(4, 1.0);
        assert!(edit_project_out(&x, &d).is_err());
    }

    // -----------------------------------------------------------------------
    // edit_mean_latent
    // -----------------------------------------------------------------------

    #[test]
    fn mean_latent_three_latents() {
        let a = vec![0.0f32, 0.0, 0.0];
        let b = vec![3.0f32, 6.0, 9.0];
        let c = vec![6.0f32, 12.0, 18.0];
        let mean = edit_mean_latent(&[a, b, c]).unwrap();
        assert!((mean[0] - 3.0).abs() < 1e-5);
        assert!((mean[1] - 6.0).abs() < 1e-5);
        assert!((mean[2] - 9.0).abs() < 1e-5);
    }

    #[test]
    fn mean_latent_single() {
        let a = vec![7.0f32, 8.0, 9.0];
        let mean = edit_mean_latent(std::slice::from_ref(&a)).unwrap();
        for (m, v) in mean.iter().zip(a.iter()) {
            assert!((m - v).abs() < 1e-5);
        }
    }

    #[test]
    fn mean_latent_empty_collection() {
        assert!(edit_mean_latent(&[]).is_err());
    }

    #[test]
    fn mean_latent_length_mismatch() {
        let a = vec![1.0f32, 2.0];
        let b = vec![1.0f32, 2.0, 3.0];
        assert!(edit_mean_latent(&[a, b]).is_err());
    }

    // -----------------------------------------------------------------------
    // edit_variance_map
    // -----------------------------------------------------------------------

    #[test]
    fn variance_constant_latents_zero() {
        let a = vec![5.0f32, 5.0, 5.0];
        let b = vec![5.0f32, 5.0, 5.0];
        let var = edit_variance_map(&[a, b]).unwrap();
        assert!(var.iter().all(|&v| v.abs() < 1e-5));
    }

    #[test]
    fn variance_two_latents_known() {
        let a = vec![0.0f32, 0.0];
        let b = vec![2.0f32, 4.0];
        // mean=[1,2], variance=[(0-1)^2+(2-1)^2]/2,[...] = [1,4]
        let var = edit_variance_map(&[a, b]).unwrap();
        assert!((var[0] - 1.0).abs() < 1e-5, "var[0]={}", var[0]);
        assert!((var[1] - 4.0).abs() < 1e-5, "var[1]={}", var[1]);
    }

    #[test]
    fn variance_empty_collection() {
        assert!(edit_variance_map(&[]).is_err());
    }

    // -----------------------------------------------------------------------
    // compute_sdedit_stats / format_sdedit_stats
    // -----------------------------------------------------------------------

    #[test]
    fn compute_stats_no_mask() {
        let orig = make_latent_val(16, 1.0);
        let noisy = make_latent_val(16, 2.0);
        let stats = compute_sdedit_stats(&orig, &noisy, None, 0.5).unwrap();
        // RMS difference = 1.0
        assert!((stats.noise_level - 1.0).abs() < 1e-4);
        assert!((stats.edit_strength - 0.5).abs() < 1e-6);
        assert!((stats.mask_coverage - 1.0).abs() < 1e-6);
        assert!(stats.latent_norm > 0.0);
        assert!(stats.noisy_norm > 0.0);
    }

    #[test]
    fn compute_stats_with_mask() {
        let orig = make_latent_val(16, 0.0);
        let noisy = make_latent_val(16, 1.0);
        let mask = make_mask(4, 4, 1.0);
        let stats = compute_sdedit_stats(&orig, &noisy, Some(&mask), 0.7).unwrap();
        assert!((stats.mask_coverage - 1.0).abs() < 1e-5);
        assert!((stats.edit_strength - 0.7).abs() < 1e-6);
    }

    #[test]
    fn compute_stats_identical_latents_zero_noise() {
        let x = make_latent_val(16, 2.5);
        let stats = compute_sdedit_stats(&x, &x, None, 0.3).unwrap();
        assert!(stats.noise_level.abs() < 1e-5);
    }

    #[test]
    fn compute_stats_empty_error() {
        let err = compute_sdedit_stats(&[], &[], None, 0.5).unwrap_err();
        assert!(matches!(err, ImageEditError::EmptyInput));
    }

    #[test]
    fn compute_stats_mismatch_error() {
        let orig = make_latent_val(8, 1.0);
        let noisy = make_latent_val(4, 1.0);
        assert!(compute_sdedit_stats(&orig, &noisy, None, 0.5).is_err());
    }

    #[test]
    fn format_stats_contains_fields() {
        let stats = SdeditStats {
            noise_level: 0.25,
            edit_strength: 0.7,
            mask_coverage: 0.5,
            latent_norm: 3.1,
            noisy_norm: 4.0,
        };
        let s = format_sdedit_stats(&stats);
        assert!(s.contains("noise_level"));
        assert!(s.contains("edit_strength"));
        assert!(s.contains("mask_coverage"));
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn lerp_empty_error() {
        assert!(edit_lerp_latents(&[], &[], 0.5).is_err());
    }

    #[test]
    fn slerp_empty_error() {
        assert!(edit_slerp_latents(&[], &[], 0.5).is_err());
    }

    #[test]
    fn latent_distance_empty_error() {
        assert!(edit_latent_distance(&[], &[]).is_err());
    }

    #[test]
    fn cosine_similarity_empty_error() {
        assert!(edit_cosine_similarity(&[], &[]).is_err());
    }

    #[test]
    fn normalize_empty_error() {
        assert!(edit_normalize_latent(&[]).is_err());
    }

    #[test]
    fn project_out_empty_error() {
        assert!(edit_project_out(&[], &[]).is_err());
    }

    #[test]
    fn mean_latent_empty_vec_error() {
        assert!(edit_mean_latent(&[vec![]]).is_err());
    }

    #[test]
    fn variance_empty_vec_error() {
        assert!(edit_variance_map(&[vec![]]).is_err());
    }

    #[test]
    fn add_noise_single_element() {
        let ab = edit_cosine_alpha_bars(10);
        let lat = vec![1.0f32];
        let noise = vec![0.0f32];
        let result = edit_add_noise(&lat, &noise, 0, &ab).unwrap();
        assert_eq!(result.len(), 1);
    }
}
