//! # image_editing
//!
//! SDEdit-based image editing for the diffusion pipeline.
//!
//! Implements the SDEdit algorithm (Song et al., 2021): add noise to an input
//! image latent up to a given noise level, then denoise back — enabling
//! structure-preserving edits guided by a target prompt or mask.
//!
//! ## Supported edit modes
//!
//! - **SdEdit**: Classic SDEdit — noise to `t`, then denoise.
//! - **Inpaint**: Blend edited and original latents using a binary mask.
//! - **Interpolate**: Lerp between source and target latents before denoising.
//! - **NoiseOnly**: Add noise only (caller handles denoising step).
//!
//! ## Layout convention
//!
//! [`EditLatent`] stores data in channels-first, row-major order:
//! `index = c * (H * W) + h * W + w`
//!
//! ## PRNG
//!
//! All stochastic functions use an inline xorshift64 + Box-Muller transform.
//! No external `rand` crate is required. Seed guard: `state = state.max(1)`.
//!
//! ## Example
//! ```rust
//! use oxigaf_diffusion::image_editing::{
//!     EditLatent, ImageEditingConfig, EditMode, prepare_edit_input,
//! };
//!
//! let source = EditLatent::new(4, 8, 8);
//! let config = ImageEditingConfig::quick_edit();
//! let result = prepare_edit_input(&source, None, None, &config).unwrap();
//! assert_eq!(result.numel(), source.numel());
//! ```

pub mod sdedit;
pub use sdedit::{
    compute_sdedit_stats, edit_add_noise, edit_blend_with_mask, edit_cosine_alpha_bars,
    edit_cosine_similarity, edit_expand_mask_to_channels, edit_latent_distance, edit_lerp_latents,
    edit_linear_alpha_bars, edit_mean_latent, edit_normalize_latent, edit_project_out,
    edit_sample_noise, edit_slerp_latents, edit_start_timestep, edit_variance_map,
    format_sdedit_stats, sdedit_perturb, sdedit_perturb_with_mask, EditConfig, SdeditStats,
};

use thiserror::Error;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors produced by image-editing operations.
#[derive(Debug, Error)]
pub enum ImageEditingError {
    /// Invalid configuration parameter.
    #[error("Invalid config: {0}")]
    InvalidConfig(String),

    /// Invalid image data or shape.
    #[error("Invalid image: {0}")]
    InvalidImage(String),

    /// Noise level is outside the valid `[0, max_t]` range.
    #[error("Noise level out of range: {t} not in [0, {max_t}]")]
    NoiseLevelOutOfRange { t: f32, max_t: f32 },

    /// Dimension mismatch between two operands.
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    /// A mask was required but was empty (all-zero or zero-size).
    #[error("Empty mask")]
    EmptyMask,

    /// A numerical computation failed (e.g. NaN/Inf detected).
    #[error("Numerical error: {0}")]
    NumericalError(String),

    /// An empty input was provided where at least one element is required.
    #[error("Empty input")]
    EmptyInput,
}

// ---------------------------------------------------------------------------
// Inline xorshift64 PRNG + Box-Muller
// ---------------------------------------------------------------------------

/// Advance one xorshift64 step, enforcing the non-zero invariant.
#[inline]
fn xorshift64(state: &mut u64) -> u64 {
    // Ensure state is never zero (required by xorshift64 correctness).
    if *state == 0 {
        *state = 1;
    }
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// Convert a `u64` to a uniform `f64` in `(0, 1]`.
///
/// Strictly positive so that `ln` is always finite.
#[inline]
fn u64_to_f64_01(bits: u64) -> f64 {
    let numer = (bits as u128 + 1) as f64;
    let denom = (u64::MAX as u128 + 1) as f64; // 2^64
    numer / denom
}

/// Generate a pair of independent standard-normal samples via Box-Muller.
///
/// Neither output will be NaN or ±∞ because `u1 ∈ (0, 1]` and `u2 ∈ (0, 1]`.
#[inline]
fn box_muller(state: &mut u64) -> (f32, f32) {
    let u1 = u64_to_f64_01(xorshift64(state));
    let u2 = u64_to_f64_01(xorshift64(state));
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = std::f64::consts::TAU * u2;
    ((r * theta.cos()) as f32, (r * theta.sin()) as f32)
}

/// Generate a Gaussian-noise buffer of length `n` with standard deviation `scale`.
fn gaussian_noise_vec(n: usize, scale: f32, seed: u64) -> Vec<f32> {
    let mut state: u64 = seed.max(1);
    let mut data = Vec::with_capacity(n);
    let pairs = n / 2;
    for _ in 0..pairs {
        let (z0, z1) = box_muller(&mut state);
        data.push(z0 * scale);
        data.push(z1 * scale);
    }
    if n % 2 == 1 {
        let (z0, _) = box_muller(&mut state);
        data.push(z0 * scale);
    }
    data
}

// ---------------------------------------------------------------------------
// EditLatent
// ---------------------------------------------------------------------------

/// A latent representation for image editing, stored in channels-first
/// row-major layout `[C, H, W]`.
///
/// Distinct from `LatentVector` (1-D) and `VarLatentVector` (3-D with
/// extra methods) defined elsewhere in this crate.
#[derive(Debug, Clone)]
pub struct EditLatent {
    /// Number of channels (C).
    pub channels: usize,
    /// Height (H).
    pub height: usize,
    /// Width (W).
    pub width: usize,
    /// Raw data in channels-first, row-major `[C, H, W]` layout.
    pub data: Vec<f32>,
}

impl EditLatent {
    /// Create a zero-initialised latent with the given spatial dimensions.
    pub fn new(channels: usize, height: usize, width: usize) -> Self {
        let n = channels * height * width;
        Self {
            channels,
            height,
            width,
            data: vec![0.0f32; n],
        }
    }

    /// Construct from an existing data buffer, verifying that its length
    /// matches `channels * height * width`.
    pub fn from_data(
        channels: usize,
        height: usize,
        width: usize,
        data: Vec<f32>,
    ) -> Result<Self, ImageEditingError> {
        let expected = channels * height * width;
        if data.len() != expected {
            return Err(ImageEditingError::DimensionMismatch {
                expected,
                actual: data.len(),
            });
        }
        Ok(Self {
            channels,
            height,
            width,
            data,
        })
    }

    /// Total number of elements: `channels * height * width`.
    #[inline]
    pub fn numel(&self) -> usize {
        self.channels * self.height * self.width
    }

    /// L2 norm (Euclidean length) of the data vector.
    pub fn l2_norm(&self) -> f32 {
        self.data.iter().fold(0.0f32, |acc, &x| acc + x * x).sqrt()
    }

    /// Return a new latent with every element multiplied by `scale`.
    pub fn clone_with_scale(&self, scale: f32) -> Self {
        let data = self.data.iter().map(|&x| x * scale).collect();
        Self {
            channels: self.channels,
            height: self.height,
            width: self.width,
            data,
        }
    }
}

// ---------------------------------------------------------------------------
// EditMask
// ---------------------------------------------------------------------------

/// Binary mask for inpainting operations.
///
/// Values are in `[0, 1]`:  `1.0` = edit this pixel,  `0.0` = preserve it.
/// Stored in row-major order: `index = h * W + w`.
#[derive(Debug, Clone)]
pub struct EditMask {
    /// Height (H).
    pub height: usize,
    /// Width (W).
    pub width: usize,
    /// Mask data in row-major order.  Values nominally in `[0, 1]`.
    pub data: Vec<f32>,
}

impl EditMask {
    /// Create a mask of the given size filled uniformly with `fill`.
    pub fn new(height: usize, width: usize, fill: f32) -> Self {
        let n = height * width;
        Self {
            height,
            width,
            data: vec![fill; n],
        }
    }

    /// Construct from an existing data buffer, verifying that its length
    /// matches `height * width`.
    pub fn from_data(
        height: usize,
        width: usize,
        data: Vec<f32>,
    ) -> Result<Self, ImageEditingError> {
        let expected = height * width;
        if data.len() != expected {
            return Err(ImageEditingError::DimensionMismatch {
                expected,
                actual: data.len(),
            });
        }
        Ok(Self {
            height,
            width,
            data,
        })
    }

    /// Return a mask where every value is flipped: `out[i] = 1.0 - self[i]`.
    pub fn invert(&self) -> Self {
        let data = self.data.iter().map(|&v| 1.0 - v).collect();
        Self {
            height: self.height,
            width: self.width,
            data,
        }
    }

    /// Mean mask value — the fraction of pixels in the "edit" region.
    pub fn mask_fraction(&self) -> f32 {
        let n = self.data.len();
        if n == 0 {
            return 0.0;
        }
        self.data.iter().sum::<f32>() / n as f32
    }

    /// Max-pooling dilation with a square neighbourhood of half-size `radius`.
    ///
    /// Each output pixel takes the maximum of all mask values within the
    /// `(2*radius + 1) × (2*radius + 1)` window centred on that pixel.
    /// Boundary pixels simply use whatever neighbours exist (no padding).
    pub fn dilate(&self, radius: usize) -> Self {
        let h = self.height;
        let w = self.width;
        let mut out = vec![0.0f32; h * w];

        for row in 0..h {
            for col in 0..w {
                let r_start = row.saturating_sub(radius);
                let r_end = (row + radius + 1).min(h);
                let c_start = col.saturating_sub(radius);
                let c_end = (col + radius + 1).min(w);

                let mut max_val = 0.0f32;
                for nr in r_start..r_end {
                    for nc in c_start..c_end {
                        let v = self.data[nr * w + nc];
                        if v > max_val {
                            max_val = v;
                        }
                    }
                }
                out[row * w + col] = max_val;
            }
        }

        Self {
            height: h,
            width: w,
            data: out,
        }
    }

    /// Min-pooling erosion with a square neighbourhood of half-size `radius`.
    ///
    /// Each output pixel takes the minimum of all mask values within the
    /// `(2*radius + 1) × (2*radius + 1)` window centred on that pixel.
    pub fn erode(&self, radius: usize) -> Self {
        let h = self.height;
        let w = self.width;
        let mut out = vec![0.0f32; h * w];

        for row in 0..h {
            for col in 0..w {
                let r_start = row.saturating_sub(radius);
                let r_end = (row + radius + 1).min(h);
                let c_start = col.saturating_sub(radius);
                let c_end = (col + radius + 1).min(w);

                let mut min_val = f32::INFINITY;
                for nr in r_start..r_end {
                    for nc in c_start..c_end {
                        let v = self.data[nr * w + nc];
                        if v < min_val {
                            min_val = v;
                        }
                    }
                }
                out[row * w + col] = if min_val.is_finite() { min_val } else { 0.0 };
            }
        }

        Self {
            height: h,
            width: w,
            data: out,
        }
    }

    /// Gaussian smooth the mask and then threshold at 0.5.
    ///
    /// Builds a 1-D Gaussian kernel of size `2*ceil(3*sigma)+1`, applies it
    /// separably (horizontal then vertical pass), and thresholds the result
    /// to produce a binary mask. When `sigma` is non-finite (e.g. NaN) or
    /// `<= 0.0`, the Gaussian pass is skipped but the 0.5 threshold is still
    /// applied to the original data, so the result is always a proper
    /// binary mask as documented.
    pub fn smooth(&self, sigma: f32) -> Self {
        let h = self.height;
        let w = self.width;

        if h == 0 || w == 0 || !sigma.is_finite() || sigma <= 0.0 {
            let data = self
                .data
                .iter()
                .map(|&v| if v >= 0.5 { 1.0 } else { 0.0 })
                .collect();
            return Self {
                height: h,
                width: w,
                data,
            };
        }

        // Build 1-D kernel.
        let half = (3.0 * sigma).ceil() as usize;
        let ksize = 2 * half + 1;
        let mut kernel = vec![0.0f32; ksize];
        let mut ksum = 0.0f32;
        for (i, kv) in kernel.iter_mut().enumerate() {
            let x = i as f32 - half as f32;
            let v = (-0.5 * (x / sigma) * (x / sigma)).exp();
            *kv = v;
            ksum += v;
        }
        for k in &mut kernel {
            *k /= ksum;
        }

        // Horizontal pass.
        let mut tmp = vec![0.0f32; h * w];
        for row in 0..h {
            for col in 0..w {
                let mut acc = 0.0f32;
                let mut wsum = 0.0f32;
                for (ki, &kv) in kernel.iter().enumerate() {
                    let kc = ki as isize - half as isize + col as isize;
                    if kc >= 0 && kc < w as isize {
                        acc += kv * self.data[row * w + kc as usize];
                        wsum += kv;
                    }
                }
                tmp[row * w + col] = if wsum > 0.0 { acc / wsum } else { 0.0 };
            }
        }

        // Vertical pass.
        let mut out = vec![0.0f32; h * w];
        for row in 0..h {
            for col in 0..w {
                let mut acc = 0.0f32;
                let mut wsum = 0.0f32;
                for (ki, &kv) in kernel.iter().enumerate() {
                    let kr = ki as isize - half as isize + row as isize;
                    if kr >= 0 && kr < h as isize {
                        acc += kv * tmp[kr as usize * w + col];
                        wsum += kv;
                    }
                }
                let v = if wsum > 0.0 { acc / wsum } else { 0.0 };
                out[row * w + col] = if v >= 0.5 { 1.0 } else { 0.0 };
            }
        }

        Self {
            height: h,
            width: w,
            data: out,
        }
    }

    /// Fraction of pixels set to 1.0 (i.e. values > 0.5).
    pub fn coverage(&self) -> f32 {
        let n = self.data.len();
        if n == 0 {
            return 0.0;
        }
        let count = self.data.iter().filter(|&&v| v > 0.5).count();
        count as f32 / n as f32
    }

    /// Create a mask of the given size filled entirely with 1.0.
    ///
    /// Takes `(height, width)`, matching [`Self::new`] and [`Self::from_data`].
    pub fn all_ones(height: usize, width: usize) -> Self {
        let n = height * width;
        Self {
            height,
            width,
            data: vec![1.0f32; n],
        }
    }
}

// ---------------------------------------------------------------------------
// EditMode
// ---------------------------------------------------------------------------

/// Strategy used to prepare the editing input latent.
#[derive(Debug, Clone, PartialEq)]
pub enum EditMode {
    /// SDEdit: add noise to `noise_level`, then let the caller denoise.
    SdEdit,
    /// Inpainting: blend edited and original latents using a mask.
    Inpaint,
    /// Interpolation: lerp between source and target before denoising.
    Interpolate,
    /// Noise injection only — the caller handles all denoising.
    NoiseOnly,
}

// ---------------------------------------------------------------------------
// ImageEditingConfig
// ---------------------------------------------------------------------------

/// Configuration for an image-editing operation.
#[derive(Debug, Clone)]
pub struct ImageEditingConfig {
    /// Edit strategy.
    pub mode: EditMode,
    /// Noise level `t ∈ [0, 1]`: how much noise to add (higher = more creative).
    pub noise_level: f32,
    /// Edit strength for [`EditMode::Interpolate`] (`0` = source, `1` = target).
    pub strength: f32,
    /// Number of denoising steps the caller should run after noising.
    ///
    /// This module (see the module docs) only prepares the noised/edited
    /// input latent; it never runs denoising steps itself. This value is
    /// therefore **advisory only** — it is range-checked by [`Self::validate`]
    /// but the caller's own sampling loop is responsible for honouring it.
    pub num_denoise_steps: usize,
    /// Base seed for the PRNG.
    pub seed: u64,
    /// When `true`, clamp `noise_level` to a lower range for structure-preserving edits.
    pub preserve_structure: bool,
}

impl Default for ImageEditingConfig {
    fn default() -> Self {
        Self {
            mode: EditMode::SdEdit,
            noise_level: 0.5,
            strength: 0.5,
            num_denoise_steps: 20,
            seed: 42,
            preserve_structure: false,
        }
    }
}

impl ImageEditingConfig {
    /// Validate the configuration, returning a descriptive error on failure.
    ///
    /// Checks:
    /// - `noise_level` in `[0, 1]`
    /// - `strength` in `[0, 1]`
    /// - `num_denoise_steps >= 1`
    pub fn validate(&self) -> Result<(), ImageEditingError> {
        if !(0.0..=1.0).contains(&self.noise_level) {
            return Err(ImageEditingError::NoiseLevelOutOfRange {
                t: self.noise_level,
                max_t: 1.0,
            });
        }
        if !(0.0..=1.0).contains(&self.strength) {
            return Err(ImageEditingError::InvalidConfig(format!(
                "strength must be in [0, 1], got {}",
                self.strength
            )));
        }
        if self.num_denoise_steps < 1 {
            return Err(ImageEditingError::InvalidConfig(
                "num_denoise_steps must be >= 1".to_string(),
            ));
        }
        Ok(())
    }

    /// Preset for light, structure-preserving edits.
    ///
    /// `noise_level = 0.3`, `SdEdit` mode.
    pub fn quick_edit() -> Self {
        Self {
            mode: EditMode::SdEdit,
            noise_level: 0.3,
            strength: 0.5,
            num_denoise_steps: 20,
            seed: 42,
            preserve_structure: true,
        }
    }

    /// Preset for more creative, free-form edits.
    ///
    /// `noise_level = 0.7`, `SdEdit` mode.
    pub fn creative_edit() -> Self {
        Self {
            mode: EditMode::SdEdit,
            noise_level: 0.7,
            strength: 0.5,
            num_denoise_steps: 20,
            seed: 42,
            preserve_structure: false,
        }
    }

    /// Preset for inpainting operations.
    ///
    /// `Inpaint` mode, `noise_level = 0.99` (near-fully noised inpaint region).
    pub fn inpaint() -> Self {
        Self {
            mode: EditMode::Inpaint,
            noise_level: 0.99,
            strength: 1.0,
            num_denoise_steps: 50,
            seed: 42,
            preserve_structure: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Core free functions
// ---------------------------------------------------------------------------

/// Add noise to a latent following the DDPM/SDEdit forward marginal.
///
/// `noise_level ∈ [0, 1]` is mapped onto a 1000-step cosine noise schedule
/// ([`sdedit::edit_cosine_alpha_bars`], Nichol & Dhariwal 2021) to obtain
/// `alpha_bar`, then:
///
/// ```text
/// x_t = sqrt(alpha_bar) * x + sqrt(1 - alpha_bar) * noise
/// ```
///
/// This keeps `Var(x_t)` bounded and attenuates the original signal toward
/// zero as `noise_level -> 1`, matching what a denoiser trained on this
/// marginal expects. (A naive `x + sqrt(noise_level) * noise` — this
/// function's previous implementation — leaves the signal at full strength
/// regardless of `noise_level`, so the "fully noised" output at
/// `noise_level = 1` would still be dominated by the original latent.)
///
/// Uses xorshift64 + Box-Muller PRNG (no `rand` crate).
pub fn add_edit_noise(
    latent: &EditLatent,
    noise_level: f32,
    seed: u64,
) -> Result<EditLatent, ImageEditingError> {
    if latent.numel() == 0 {
        return Err(ImageEditingError::EmptyInput);
    }
    if !(0.0..=1.0).contains(&noise_level) {
        return Err(ImageEditingError::NoiseLevelOutOfRange {
            t: noise_level,
            max_t: 1.0,
        });
    }

    const SCHEDULE_LEN: usize = 1000;
    let alpha_bars = sdedit::edit_cosine_alpha_bars(SCHEDULE_LEN);
    let timestep = (noise_level * (SCHEDULE_LEN - 1) as f32)
        .round()
        .clamp(0.0, (SCHEDULE_LEN - 1) as f32) as usize;

    let n = latent.numel();
    // Raw N(0,1) noise: edit_add_noise applies the alpha-bar scaling itself.
    let noise = gaussian_noise_vec(n, 1.0, seed);

    // `sdedit::edit_add_noise` shares this module's `ImageEditingError`, so its
    // specific variant (e.g. `NoiseLevelOutOfRange`) now propagates directly
    // instead of being collapsed into a generic `NumericalError` string.
    let data = sdedit::edit_add_noise(&latent.data, &noise, timestep, &alpha_bars)?;

    Ok(EditLatent {
        channels: latent.channels,
        height: latent.height,
        width: latent.width,
        data,
    })
}

/// Blend two latents element-wise using a spatial mask.
///
/// For each spatial position `(h, w)` and every channel `c`:
/// ```text
/// out[c, h, w] = mask[h, w] * edited[c, h, w] + (1 - mask[h, w]) * original[c, h, w]
/// ```
///
/// `mask = 1.0` → use edited;  `mask = 0.0` → preserve original.
pub fn blend_with_mask(
    original: &EditLatent,
    edited: &EditLatent,
    mask: &EditMask,
) -> Result<EditLatent, ImageEditingError> {
    if original.numel() == 0 {
        return Err(ImageEditingError::EmptyInput);
    }
    if original.channels != edited.channels
        || original.height != edited.height
        || original.width != edited.width
    {
        return Err(ImageEditingError::DimensionMismatch {
            expected: original.numel(),
            actual: edited.numel(),
        });
    }
    if mask.height != original.height || mask.width != original.width {
        return Err(ImageEditingError::DimensionMismatch {
            expected: original.height * original.width,
            actual: mask.height * mask.width,
        });
    }

    let hw = original.height * original.width;
    let mut data = Vec::with_capacity(original.numel());

    for c in 0..original.channels {
        for hw_idx in 0..hw {
            let m = mask.data[hw_idx];
            let orig = original.data[c * hw + hw_idx];
            let edit = edited.data[c * hw + hw_idx];
            data.push(m * edit + (1.0 - m) * orig);
        }
    }

    Ok(EditLatent {
        channels: original.channels,
        height: original.height,
        width: original.width,
        data,
    })
}

/// SDEdit noise step: add noise to the latent at `config.noise_level` and
/// return `(noised_latent, actual_noise_level_used)`.
///
/// When `config.preserve_structure` is `true` the effective noise level is
/// clamped to `noise_level * 0.5` to keep more structure intact.
pub fn sdedit_noise_step(
    latent: &EditLatent,
    config: &ImageEditingConfig,
) -> Result<(EditLatent, f32), ImageEditingError> {
    config.validate()?;

    let effective_noise = if config.preserve_structure {
        config.noise_level * 0.5
    } else {
        config.noise_level
    };

    let noised = add_edit_noise(latent, effective_noise, config.seed)?;
    Ok((noised, effective_noise))
}

/// Linearly interpolate two latents: `out = (1 - strength) * source + strength * target`.
///
/// `strength = 0` → source, `strength = 1` → target.
/// Returns an error when the shapes differ.
pub fn interpolate_edit_latents(
    source: &EditLatent,
    target: &EditLatent,
    strength: f32,
) -> Result<EditLatent, ImageEditingError> {
    if source.numel() == 0 {
        return Err(ImageEditingError::EmptyInput);
    }
    if source.channels != target.channels
        || source.height != target.height
        || source.width != target.width
    {
        return Err(ImageEditingError::DimensionMismatch {
            expected: source.numel(),
            actual: target.numel(),
        });
    }

    let t = strength.clamp(0.0, 1.0);
    let data: Vec<f32> = source
        .data
        .iter()
        .zip(target.data.iter())
        .map(|(&s, &tgt)| (1.0 - t) * s + t * tgt)
        .collect();

    Ok(EditLatent {
        channels: source.channels,
        height: source.height,
        width: source.width,
        data,
    })
}

/// Apply an inpaint mask to combine a noised-edited latent with the original.
///
/// `combined = noised_edited * mask + original * (1 - mask)`
///
/// Semantics: the mask marks the *edit* region (`1.0`); the preserve region
/// (`0.0`) is filled from the original latent so that the caller's denoiser
/// only reconstructs the masked area.
pub fn apply_inpaint_mask(
    noised_edited: &EditLatent,
    original: &EditLatent,
    mask: &EditMask,
) -> Result<EditLatent, ImageEditingError> {
    blend_with_mask(original, noised_edited, mask)
}

/// Prepare the editing input latent based on the configured [`EditMode`].
///
/// | Mode            | Operation                                              |
/// |-----------------|--------------------------------------------------------|
/// | `SdEdit`        | `add_edit_noise(source, noise_level)`                  |
/// | `Inpaint`       | noise source, then `apply_inpaint_mask`                |
/// | `Interpolate`   | `interpolate_edit_latents`, then noise                 |
/// | `NoiseOnly`     | `add_edit_noise(source, noise_level)`                  |
///
/// `target` is required for `Interpolate` mode; `mask` is required for
/// `Inpaint` mode.
pub fn prepare_edit_input(
    source: &EditLatent,
    target: Option<&EditLatent>,
    mask: Option<&EditMask>,
    config: &ImageEditingConfig,
) -> Result<EditLatent, ImageEditingError> {
    config.validate()?;

    match &config.mode {
        EditMode::SdEdit | EditMode::NoiseOnly => {
            add_edit_noise(source, config.noise_level, config.seed)
        }

        EditMode::Inpaint => {
            let m = mask.ok_or(ImageEditingError::EmptyMask)?;
            if m.mask_fraction() == 0.0 && m.data.iter().all(|&v| v == 0.0) {
                return Err(ImageEditingError::EmptyMask);
            }
            let noised = add_edit_noise(source, config.noise_level, config.seed)?;
            apply_inpaint_mask(&noised, source, m)
        }

        EditMode::Interpolate => {
            let tgt = target.ok_or_else(|| {
                ImageEditingError::InvalidConfig(
                    "Interpolate mode requires a target latent".to_string(),
                )
            })?;
            let interpolated = interpolate_edit_latents(source, tgt, config.strength)?;
            add_edit_noise(&interpolated, config.noise_level, config.seed)
        }
    }
}

/// Compute the L2 distance between two latents.
///
/// Returns `0.0` for identical latents, and an error if the shapes differ.
pub fn edit_distance(original: &EditLatent, edited: &EditLatent) -> Result<f32, ImageEditingError> {
    if original.channels != edited.channels
        || original.height != edited.height
        || original.width != edited.width
    {
        return Err(ImageEditingError::DimensionMismatch {
            expected: original.numel(),
            actual: edited.numel(),
        });
    }
    if original.numel() == 0 {
        return Err(ImageEditingError::EmptyInput);
    }

    let sq_sum = original
        .data
        .iter()
        .zip(edited.data.iter())
        .fold(0.0f32, |acc, (&a, &b)| {
            let d = a - b;
            acc + d * d
        });
    Ok(sq_sum.sqrt())
}

/// Generate multiple noised variants of `source` at each noise level in `noise_levels`.
///
/// Each output latent has the same shape as `source`.  A different seed is
/// derived for each level by mixing the base seed with the level index.
pub fn sample_edit_at_levels(
    source: &EditLatent,
    noise_levels: &[f32],
    seed: u64,
) -> Result<Vec<EditLatent>, ImageEditingError> {
    if noise_levels.is_empty() {
        return Err(ImageEditingError::EmptyInput);
    }
    if source.numel() == 0 {
        return Err(ImageEditingError::EmptyInput);
    }

    let mut results = Vec::with_capacity(noise_levels.len());
    for (i, &level) in noise_levels.iter().enumerate() {
        // Derive a per-level seed: mix base seed with index via wrapping arithmetic.
        let level_seed = seed
            .wrapping_add(i as u64)
            .wrapping_mul(6364136223846793005)
            .max(1);
        let noised = add_edit_noise(source, level, level_seed)?;
        results.push(noised);
    }
    Ok(results)
}

// ---------------------------------------------------------------------------
// EditStats
// ---------------------------------------------------------------------------

/// Summary statistics for a single editing operation.
#[derive(Debug, Clone)]
pub struct EditStats {
    /// The noise level actually applied.
    pub noise_level_used: f32,
    /// L2 distance between original and edited latent.
    pub edit_distance: f32,
    /// Mean mask value; `0.0` when no mask was provided.
    pub masked_fraction: f32,
    /// Mean absolute element-wise change between original and edited.
    pub mean_edit_magnitude: f32,
}

/// Compute editing statistics comparing `original` to `edited`.
///
/// - `noise_level`: the noise level used during editing.
/// - `mask`: optional spatial mask; when `None`, `masked_fraction` is `0.0`.
pub fn compute_edit_stats(
    original: &EditLatent,
    edited: &EditLatent,
    mask: Option<&EditMask>,
    noise_level: f32,
) -> Result<EditStats, ImageEditingError> {
    if original.numel() == 0 || edited.numel() == 0 {
        return Err(ImageEditingError::EmptyInput);
    }
    if original.channels != edited.channels
        || original.height != edited.height
        || original.width != edited.width
    {
        return Err(ImageEditingError::DimensionMismatch {
            expected: original.numel(),
            actual: edited.numel(),
        });
    }

    let dist = edit_distance(original, edited)?;

    let n = original.numel() as f32;
    let mean_magnitude = original
        .data
        .iter()
        .zip(edited.data.iter())
        .fold(0.0f32, |acc, (&a, &b)| acc + (a - b).abs())
        / n;

    let masked_fraction = mask.map_or(0.0, |m| m.mask_fraction());

    Ok(EditStats {
        noise_level_used: noise_level,
        edit_distance: dist,
        masked_fraction,
        mean_edit_magnitude: mean_magnitude,
    })
}

// ---------------------------------------------------------------------------
// EditHistory
// ---------------------------------------------------------------------------

/// A bounded history of `(EditLatent, EditStats)` pairs, enabling undo/redo
/// workflows.
///
/// When [`push`](EditHistory::push) is called and `edits.len() == max_history`,
/// the oldest entry is evicted (FIFO).
pub struct EditHistory {
    /// Stored `(latent, stats)` pairs in chronological order.
    pub edits: Vec<(EditLatent, EditStats)>,
    /// Maximum number of entries retained.
    pub max_history: usize,
}

impl EditHistory {
    /// Create an empty history with the given capacity.
    pub fn new(max_history: usize) -> Self {
        Self {
            edits: Vec::new(),
            max_history,
        }
    }

    /// Append a new entry.  If the history is full, the oldest entry is dropped.
    pub fn push(&mut self, latent: EditLatent, stats: EditStats) {
        if self.max_history == 0 {
            return;
        }
        if self.edits.len() >= self.max_history {
            self.edits.remove(0);
        }
        self.edits.push((latent, stats));
    }

    /// Return a reference to the most recently pushed latent, or `None` if empty.
    pub fn latest(&self) -> Option<&EditLatent> {
        self.edits.last().map(|(lat, _)| lat)
    }

    /// Remove and return the most recently pushed latent, or `None` if empty.
    pub fn revert(&mut self) -> Option<EditLatent> {
        self.edits.pop().map(|(lat, _)| lat)
    }

    /// Number of entries currently stored.
    pub fn len(&self) -> usize {
        self.edits.len()
    }

    /// Returns `true` when no entries are stored.
    pub fn is_empty(&self) -> bool {
        self.edits.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // Helper: fill a latent with a known pattern
    // ------------------------------------------------------------------
    fn make_latent(c: usize, h: usize, w: usize, val: f32) -> EditLatent {
        let n = c * h * w;
        EditLatent {
            channels: c,
            height: h,
            width: w,
            data: vec![val; n],
        }
    }

    fn make_latent_range(c: usize, h: usize, w: usize) -> EditLatent {
        let n = c * h * w;
        let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        EditLatent {
            channels: c,
            height: h,
            width: w,
            data,
        }
    }

    fn zero_stats(noise_level: f32) -> EditStats {
        EditStats {
            noise_level_used: noise_level,
            edit_distance: 0.0,
            masked_fraction: 0.0,
            mean_edit_magnitude: 0.0,
        }
    }

    // ------------------------------------------------------------------
    // EditLatent
    // ------------------------------------------------------------------

    #[test]
    fn edit_latent_new_zeros() {
        let lat = EditLatent::new(4, 8, 8);
        assert_eq!(lat.numel(), 4 * 8 * 8);
        assert!(lat.data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn edit_latent_from_data_ok() {
        let data = vec![1.0f32; 4 * 4 * 4];
        let lat = EditLatent::from_data(4, 4, 4, data).unwrap();
        assert_eq!(lat.numel(), 64);
    }

    #[test]
    fn edit_latent_from_data_wrong_size() {
        let data = vec![0.0f32; 10]; // too short
        let result = EditLatent::from_data(4, 4, 4, data);
        assert!(result.is_err());
        match result.unwrap_err() {
            ImageEditingError::DimensionMismatch { expected, actual } => {
                assert_eq!(expected, 64);
                assert_eq!(actual, 10);
            }
            e => panic!("unexpected error: {e}"),
        }
    }

    #[test]
    fn edit_latent_numel() {
        let lat = EditLatent::new(3, 5, 7);
        assert_eq!(lat.numel(), 3 * 5 * 7);
    }

    #[test]
    fn edit_latent_l2_norm() {
        // 4 elements all equal to 2.0 → norm = 2*2 = sqrt(4*4) = 4
        let data = vec![2.0f32; 4];
        let lat = EditLatent::from_data(1, 2, 2, data).unwrap();
        let expected = (4.0f32 * 4.0).sqrt(); // sqrt(16) = 4
        assert!((lat.l2_norm() - expected).abs() < 1e-5);
    }

    #[test]
    fn edit_latent_clone_with_scale() {
        let lat = make_latent(2, 3, 3, 2.0);
        let scaled = lat.clone_with_scale(3.0);
        assert!(scaled.data.iter().all(|&x| (x - 6.0).abs() < 1e-6));
        assert_eq!(scaled.channels, lat.channels);
    }

    // ------------------------------------------------------------------
    // EditMask
    // ------------------------------------------------------------------

    #[test]
    fn edit_mask_new() {
        let m = EditMask::new(4, 4, 0.5);
        assert_eq!(m.data.len(), 16);
        assert!(m.data.iter().all(|&v| (v - 0.5).abs() < 1e-6));
    }

    #[test]
    fn edit_mask_from_data_ok() {
        let data = vec![1.0f32; 9];
        let m = EditMask::from_data(3, 3, data).unwrap();
        assert_eq!(m.height, 3);
        assert_eq!(m.width, 3);
    }

    #[test]
    fn edit_mask_from_data_wrong_size() {
        let data = vec![0.0f32; 5];
        let result = EditMask::from_data(3, 3, data);
        assert!(result.is_err());
    }

    #[test]
    fn edit_mask_invert() {
        let m = EditMask::new(2, 2, 0.3);
        let inv = m.invert();
        assert!(inv.data.iter().all(|&v| (v - 0.7).abs() < 1e-6));
    }

    #[test]
    fn edit_mask_mask_fraction() {
        // Half ones, half zeros → fraction = 0.5
        let data = vec![1.0, 1.0, 0.0, 0.0];
        let m = EditMask::from_data(2, 2, data).unwrap();
        assert!((m.mask_fraction() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn edit_mask_dilate_radius_0_identity() {
        let data = vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let m = EditMask::from_data(3, 3, data.clone()).unwrap();
        let d = m.dilate(0);
        assert_eq!(d.data, data);
    }

    #[test]
    fn edit_mask_dilate_expands_region() {
        // Single 1.0 at centre of 5x5 grid; radius=1 should spread to 3x3 block.
        let mut data = vec![0.0f32; 25];
        data[12] = 1.0; // row=2, col=2
        let m = EditMask::from_data(5, 5, data).unwrap();
        let d = m.dilate(1);
        // The 3x3 neighbourhood of (2,2) should all be 1.0.
        for r in 1..=3 {
            for c in 1..=3 {
                assert_eq!(d.data[r * 5 + c], 1.0, "pixel ({r},{c}) should be 1.0");
            }
        }
        // Corners (0,0) etc. should remain 0.0
        assert_eq!(d.data[0], 0.0);
        assert_eq!(d.data[4], 0.0);
    }

    // ------------------------------------------------------------------
    // ImageEditingConfig
    // ------------------------------------------------------------------

    #[test]
    fn config_validate_ok() {
        let cfg = ImageEditingConfig::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn config_validate_noise_out_of_range() {
        let cfg = ImageEditingConfig {
            noise_level: 1.5,
            ..Default::default()
        };
        assert!(matches!(
            cfg.validate().unwrap_err(),
            ImageEditingError::NoiseLevelOutOfRange { .. }
        ));
    }

    #[test]
    fn config_validate_negative_noise() {
        let cfg = ImageEditingConfig {
            noise_level: -0.1,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_validate_zero_steps() {
        let cfg = ImageEditingConfig {
            num_denoise_steps: 0,
            ..Default::default()
        };
        assert!(matches!(
            cfg.validate().unwrap_err(),
            ImageEditingError::InvalidConfig(_)
        ));
    }

    #[test]
    fn config_validate_strength_out_of_range() {
        let cfg = ImageEditingConfig {
            strength: 1.5,
            ..Default::default()
        };
        assert!(matches!(
            cfg.validate().unwrap_err(),
            ImageEditingError::InvalidConfig(_)
        ));
    }

    #[test]
    fn config_presets_valid() {
        assert!(ImageEditingConfig::quick_edit().validate().is_ok());
        assert!(ImageEditingConfig::creative_edit().validate().is_ok());
        assert!(ImageEditingConfig::inpaint().validate().is_ok());
    }

    #[test]
    fn config_quick_edit_noise_level() {
        assert!((ImageEditingConfig::quick_edit().noise_level - 0.3).abs() < 1e-6);
    }

    #[test]
    fn config_creative_edit_noise_level() {
        assert!((ImageEditingConfig::creative_edit().noise_level - 0.7).abs() < 1e-6);
    }

    #[test]
    fn config_inpaint_mode() {
        assert_eq!(ImageEditingConfig::inpaint().mode, EditMode::Inpaint);
    }

    // ------------------------------------------------------------------
    // add_edit_noise
    // ------------------------------------------------------------------

    #[test]
    fn add_noise_same_shape() {
        let lat = make_latent(4, 8, 8, 0.0);
        let out = add_edit_noise(&lat, 0.5, 42).unwrap();
        assert_eq!(out.channels, 4);
        assert_eq!(out.height, 8);
        assert_eq!(out.width, 8);
        assert_eq!(out.numel(), lat.numel());
    }

    #[test]
    fn add_noise_different_seeds_different_output() {
        let lat = make_latent(2, 4, 4, 0.0);
        let out_a = add_edit_noise(&lat, 0.5, 1).unwrap();
        let out_b = add_edit_noise(&lat, 0.5, 2).unwrap();
        // At least some values should differ
        let any_diff = out_a
            .data
            .iter()
            .zip(out_b.data.iter())
            .any(|(a, b)| a != b);
        assert!(any_diff, "different seeds should produce different noise");
    }

    #[test]
    fn add_noise_level_zero_stays_close_to_input() {
        // With the corrected DDPM/SDEdit forward marginal, noise_level=0
        // maps to alpha_bar close to (but not exactly) 1.0 — the cosine
        // schedule's s=0.008 offset keeps a small noise floor even at the
        // very first timestep — so the output correlates strongly with the
        // input without being bit-identical to it.
        let lat = make_latent(2, 4, 4, 1.0);
        let out = add_edit_noise(&lat, 0.0, 42).unwrap();
        for (&a, &b) in lat.data.iter().zip(out.data.iter()) {
            assert!(
                (a - b).abs() < 0.5,
                "noise_level=0 should stay close to input: {} vs {}",
                a,
                b
            );
        }
    }

    #[test]
    fn add_noise_high_level_attenuates_large_signal() {
        // Regression test for the SDEdit forward-marginal bug: the old
        // `x + sqrt(noise_level) * z` formula left the signal at full
        // strength regardless of noise_level, so a large-magnitude latent
        // stayed large even at noise_level=1.0 ("pure noise"). The correct
        // marginal sqrt(alpha_bar)*x + sqrt(1-alpha_bar)*z must shrink it.
        let lat = make_latent(2, 8, 8, 1000.0);
        let out = add_edit_noise(&lat, 1.0, 7).unwrap();
        let max_abs = out.data.iter().cloned().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(
            max_abs < 200.0,
            "noise_level=1.0 should mostly destroy a large-magnitude signal, got max_abs={}",
            max_abs
        );
    }

    #[test]
    fn add_noise_invalid_level() {
        let lat = make_latent(1, 2, 2, 0.0);
        assert!(add_edit_noise(&lat, 1.5, 1).is_err());
        assert!(add_edit_noise(&lat, -0.1, 1).is_err());
    }

    // ------------------------------------------------------------------
    // blend_with_mask
    // ------------------------------------------------------------------

    #[test]
    fn blend_all_zero_mask_gives_original() {
        let orig = make_latent(2, 4, 4, 1.0);
        let edit = make_latent(2, 4, 4, 5.0);
        let mask = EditMask::new(4, 4, 0.0);
        let out = blend_with_mask(&orig, &edit, &mask).unwrap();
        assert!(out.data.iter().all(|&x| (x - 1.0).abs() < 1e-6));
    }

    #[test]
    fn blend_all_one_mask_gives_edited() {
        let orig = make_latent(2, 4, 4, 1.0);
        let edit = make_latent(2, 4, 4, 5.0);
        let mask = EditMask::new(4, 4, 1.0);
        let out = blend_with_mask(&orig, &edit, &mask).unwrap();
        assert!(out.data.iter().all(|&x| (x - 5.0).abs() < 1e-6));
    }

    #[test]
    fn blend_half_mask() {
        let orig = make_latent(1, 1, 1, 0.0);
        let edit = make_latent(1, 1, 1, 10.0);
        let mask = EditMask::new(1, 1, 0.5);
        let out = blend_with_mask(&orig, &edit, &mask).unwrap();
        assert!((out.data[0] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn blend_dimension_mismatch() {
        let orig = make_latent(2, 4, 4, 0.0);
        let edit = make_latent(2, 8, 8, 0.0); // different spatial size
        let mask = EditMask::new(4, 4, 0.5);
        assert!(blend_with_mask(&orig, &edit, &mask).is_err());
    }

    // ------------------------------------------------------------------
    // sdedit_noise_step
    // ------------------------------------------------------------------

    #[test]
    fn sdedit_noise_step_valid() {
        let lat = make_latent(4, 8, 8, 1.0);
        let cfg = ImageEditingConfig::quick_edit();
        let (out, level) = sdedit_noise_step(&lat, &cfg).unwrap();
        assert_eq!(out.numel(), lat.numel());
        // preserve_structure halves the noise level
        assert!((level - cfg.noise_level * 0.5).abs() < 1e-6);
    }

    #[test]
    fn sdedit_noise_step_no_structure_preserving() {
        let lat = make_latent(4, 8, 8, 0.0);
        let cfg = ImageEditingConfig {
            noise_level: 0.6,
            preserve_structure: false,
            ..ImageEditingConfig::default()
        };
        let (_, level) = sdedit_noise_step(&lat, &cfg).unwrap();
        assert!((level - 0.6).abs() < 1e-6);
    }

    // ------------------------------------------------------------------
    // interpolate_edit_latents
    // ------------------------------------------------------------------

    #[test]
    fn interpolate_strength_zero_gives_source() {
        let src = make_latent(2, 4, 4, 0.0);
        let tgt = make_latent(2, 4, 4, 1.0);
        let out = interpolate_edit_latents(&src, &tgt, 0.0).unwrap();
        assert!(out.data.iter().all(|&x| x.abs() < 1e-6));
    }

    #[test]
    fn interpolate_strength_one_gives_target() {
        let src = make_latent(2, 4, 4, 0.0);
        let tgt = make_latent(2, 4, 4, 1.0);
        let out = interpolate_edit_latents(&src, &tgt, 1.0).unwrap();
        assert!(out.data.iter().all(|&x| (x - 1.0).abs() < 1e-6));
    }

    #[test]
    fn interpolate_midpoint() {
        let src = make_latent(1, 1, 1, 0.0);
        let tgt = make_latent(1, 1, 1, 10.0);
        let out = interpolate_edit_latents(&src, &tgt, 0.5).unwrap();
        assert!((out.data[0] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn interpolate_dimension_mismatch() {
        let src = make_latent(2, 4, 4, 0.0);
        let tgt = make_latent(2, 8, 8, 1.0);
        assert!(matches!(
            interpolate_edit_latents(&src, &tgt, 0.5).unwrap_err(),
            ImageEditingError::DimensionMismatch { .. }
        ));
    }

    // ------------------------------------------------------------------
    // apply_inpaint_mask
    // ------------------------------------------------------------------

    #[test]
    fn apply_inpaint_mask_all_one_gives_edited() {
        let noised = make_latent(2, 4, 4, 9.0);
        let orig = make_latent(2, 4, 4, 1.0);
        let mask = EditMask::new(4, 4, 1.0);
        let out = apply_inpaint_mask(&noised, &orig, &mask).unwrap();
        assert!(out.data.iter().all(|&x| (x - 9.0).abs() < 1e-5));
    }

    #[test]
    fn apply_inpaint_mask_dimension_check() {
        let noised = make_latent(2, 4, 4, 0.0);
        let orig = make_latent(2, 8, 8, 0.0);
        let mask = EditMask::new(4, 4, 0.5);
        assert!(apply_inpaint_mask(&noised, &orig, &mask).is_err());
    }

    // ------------------------------------------------------------------
    // prepare_edit_input
    // ------------------------------------------------------------------

    #[test]
    fn prepare_edit_sdedit_mode() {
        let src = make_latent(4, 8, 8, 0.5);
        let cfg = ImageEditingConfig {
            mode: EditMode::SdEdit,
            noise_level: 0.4,
            ..Default::default()
        };
        let out = prepare_edit_input(&src, None, None, &cfg).unwrap();
        assert_eq!(out.numel(), src.numel());
    }

    #[test]
    fn prepare_edit_noise_only_mode() {
        let src = make_latent(4, 8, 8, 0.5);
        let cfg = ImageEditingConfig {
            mode: EditMode::NoiseOnly,
            noise_level: 0.4,
            ..Default::default()
        };
        let out = prepare_edit_input(&src, None, None, &cfg).unwrap();
        assert_eq!(out.numel(), src.numel());
    }

    #[test]
    fn prepare_edit_inpaint_mode_needs_mask() {
        let src = make_latent(4, 8, 8, 0.5);
        let cfg = ImageEditingConfig {
            mode: EditMode::Inpaint,
            noise_level: 0.99,
            ..Default::default()
        };
        // No mask provided → error
        assert!(prepare_edit_input(&src, None, None, &cfg).is_err());
    }

    #[test]
    fn prepare_edit_inpaint_mode_with_mask() {
        let src = make_latent(4, 8, 8, 0.5);
        let mask = EditMask::new(8, 8, 0.5);
        let cfg = ImageEditingConfig {
            mode: EditMode::Inpaint,
            noise_level: 0.99,
            num_denoise_steps: 1,
            ..Default::default()
        };
        let out = prepare_edit_input(&src, None, Some(&mask), &cfg).unwrap();
        assert_eq!(out.numel(), src.numel());
    }

    #[test]
    fn prepare_edit_interpolate_mode() {
        let src = make_latent(4, 8, 8, 0.0);
        let tgt = make_latent(4, 8, 8, 1.0);
        let cfg = ImageEditingConfig {
            mode: EditMode::Interpolate,
            noise_level: 0.3,
            strength: 0.5,
            num_denoise_steps: 5,
            ..Default::default()
        };
        let out = prepare_edit_input(&src, Some(&tgt), None, &cfg).unwrap();
        assert_eq!(out.numel(), src.numel());
    }

    #[test]
    fn prepare_edit_interpolate_missing_target() {
        let src = make_latent(4, 8, 8, 0.0);
        let cfg = ImageEditingConfig {
            mode: EditMode::Interpolate,
            ..Default::default()
        };
        assert!(prepare_edit_input(&src, None, None, &cfg).is_err());
    }

    // ------------------------------------------------------------------
    // edit_distance
    // ------------------------------------------------------------------

    #[test]
    fn edit_distance_identical_is_zero() {
        let lat = make_latent_range(2, 4, 4);
        let dist = edit_distance(&lat, &lat).unwrap();
        assert!(dist.abs() < 1e-5);
    }

    #[test]
    fn edit_distance_known_value() {
        // Two 1x1x1 latents: [0.0] and [3.0] → distance = 3.0
        let a = EditLatent::from_data(1, 1, 1, vec![0.0]).unwrap();
        let b = EditLatent::from_data(1, 1, 1, vec![3.0]).unwrap();
        assert!((edit_distance(&a, &b).unwrap() - 3.0).abs() < 1e-5);
    }

    #[test]
    fn edit_distance_shape_mismatch() {
        let a = make_latent(2, 4, 4, 0.0);
        let b = make_latent(2, 8, 8, 0.0);
        assert!(edit_distance(&a, &b).is_err());
    }

    // ------------------------------------------------------------------
    // sample_edit_at_levels
    // ------------------------------------------------------------------

    #[test]
    fn sample_at_levels_correct_count() {
        let src = make_latent(4, 8, 8, 0.5);
        let levels = vec![0.1, 0.3, 0.5, 0.7, 0.9];
        let results = sample_edit_at_levels(&src, &levels, 42).unwrap();
        assert_eq!(results.len(), levels.len());
    }

    #[test]
    fn sample_at_levels_all_same_shape() {
        let src = make_latent(2, 6, 6, 1.0);
        let levels = vec![0.2, 0.4, 0.8];
        let results = sample_edit_at_levels(&src, &levels, 7).unwrap();
        for r in &results {
            assert_eq!(r.channels, src.channels);
            assert_eq!(r.height, src.height);
            assert_eq!(r.width, src.width);
        }
    }

    #[test]
    fn sample_at_levels_empty_levels_error() {
        let src = make_latent(2, 4, 4, 0.0);
        assert!(sample_edit_at_levels(&src, &[], 1).is_err());
    }

    // ------------------------------------------------------------------
    // compute_edit_stats
    // ------------------------------------------------------------------

    #[test]
    fn compute_stats_valid_no_mask() {
        let orig = make_latent(2, 4, 4, 0.0);
        let edited = make_latent(2, 4, 4, 1.0);
        let stats = compute_edit_stats(&orig, &edited, None, 0.5).unwrap();
        assert!((stats.noise_level_used - 0.5).abs() < 1e-6);
        assert!(stats.edit_distance > 0.0);
        assert!((stats.masked_fraction).abs() < 1e-6);
        assert!(stats.mean_edit_magnitude > 0.0);
    }

    #[test]
    fn compute_stats_with_mask() {
        let orig = make_latent(2, 4, 4, 0.0);
        let edited = make_latent(2, 4, 4, 1.0);
        let mask = EditMask::new(4, 4, 0.75);
        let stats = compute_edit_stats(&orig, &edited, Some(&mask), 0.3).unwrap();
        assert!((stats.masked_fraction - 0.75).abs() < 1e-5);
    }

    #[test]
    fn compute_stats_identical_latents() {
        let lat = make_latent(2, 4, 4, 1.0);
        let stats = compute_edit_stats(&lat, &lat, None, 0.2).unwrap();
        assert!(stats.edit_distance.abs() < 1e-5);
        assert!(stats.mean_edit_magnitude.abs() < 1e-5);
    }

    // ------------------------------------------------------------------
    // EditHistory
    // ------------------------------------------------------------------

    #[test]
    fn history_push_and_latest() {
        let mut history = EditHistory::new(5);
        assert!(history.is_empty());
        let lat = make_latent(2, 4, 4, 1.0);
        let stats = zero_stats(0.5);
        history.push(lat.clone(), stats);
        assert_eq!(history.len(), 1);
        assert!(history.latest().is_some());
        assert!((history.latest().unwrap().data[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn history_revert() {
        let mut history = EditHistory::new(5);
        let lat = make_latent(1, 2, 2, 3.0);
        history.push(lat, zero_stats(0.1));
        let reverted = history.revert();
        assert!(reverted.is_some());
        assert!(history.is_empty());
    }

    #[test]
    fn history_max_history_eviction() {
        let mut history = EditHistory::new(3);
        for i in 0..5 {
            let lat = make_latent(1, 1, 1, i as f32);
            history.push(lat, zero_stats(0.1));
        }
        assert_eq!(history.len(), 3);
        // The latest should be 4.0 (last pushed)
        assert!((history.latest().unwrap().data[0] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn history_revert_empty_is_none() {
        let mut history = EditHistory::new(5);
        assert!(history.revert().is_none());
    }

    #[test]
    fn history_zero_max_history() {
        let mut history = EditHistory::new(0);
        let lat = make_latent(1, 1, 1, 1.0);
        history.push(lat, zero_stats(0.0));
        assert!(history.is_empty());
    }

    // -----------------------------------------------------------------------
    // ImageEditError / ImageEditingError collapse
    //
    // `sdedit` used to declare its own `ImageEditError`, bridged into this
    // module's `ImageEditingError` via a `From` impl so a caller mixing both
    // APIs could use `?` throughout. The two error enums are now one type
    // (`ImageEditingError`, declared here and used directly by `sdedit`), so
    // these tests call `sdedit`'s functions directly and check that each
    // failure mode still produces the same shape the old `From` mapping
    // documented, and that mixing both APIs needs no conversion at all.
    // -----------------------------------------------------------------------

    #[test]
    fn sdedit_dimension_mismatch_is_directly_this_modules_error() {
        let err = sdedit::edit_lerp_latents(&[1.0, 2.0, 3.0], &[1.0, 2.0], 0.5)
            .expect_err("mismatched lengths must fail");
        match err {
            ImageEditingError::DimensionMismatch { expected, actual } => {
                assert_eq!(expected, 3);
                assert_eq!(actual, 2);
            }
            other => panic!("Expected DimensionMismatch, got {other:?}"),
        }
    }

    #[test]
    fn sdedit_empty_input_is_directly_this_modules_error() {
        let err = sdedit::edit_latent_distance(&[], &[]).expect_err("empty input must fail");
        assert!(matches!(err, ImageEditingError::EmptyInput));
    }

    #[test]
    fn sdedit_invalid_strength_and_param_widen_to_invalid_config() {
        let strength =
            sdedit::edit_start_timestep(1.5, 1000).expect_err("strength outside (0, 1] must fail");
        match strength {
            ImageEditingError::InvalidConfig(msg) => {
                assert!(msg.contains("1.5"), "value must survive: {msg}");
            }
            other => panic!("Expected InvalidConfig, got {other:?}"),
        }

        let param = EditConfig {
            guidance_scale: f32::NAN,
            ..Default::default()
        }
        .validate()
        .expect_err("non-finite guidance_scale must fail");
        assert!(matches!(param, ImageEditingError::InvalidConfig(_)));
    }

    #[test]
    fn sdedit_timestep_out_of_range_widens_to_noise_level_out_of_range() {
        let ab = sdedit::edit_cosine_alpha_bars(10);
        let err = sdedit::edit_add_noise(&[0.0; 4], &[0.0; 4], 20, &ab)
            .expect_err("timestep past the schedule length must fail");
        match err {
            ImageEditingError::NoiseLevelOutOfRange { t, max_t } => {
                assert!((t - 20.0).abs() < f32::EPSILON);
                assert!((max_t - 9.0).abs() < f32::EPSILON);
            }
            other => panic!("Expected NoiseLevelOutOfRange, got {other:?}"),
        }
    }

    #[test]
    fn sdedit_invalid_mask_becomes_invalid_image_not_empty_mask() {
        // `EmptyMask` means specifically all-zero/zero-size; a structurally
        // invalid (missing) mask is a different failure and must not be
        // conflated.
        let config = EditConfig {
            use_mask: true,
            ..Default::default()
        };
        let ab = sdedit::edit_cosine_alpha_bars(config.n_timesteps);
        let mut state = 7u64;
        let err = sdedit::sdedit_perturb_with_mask(&[0.5, 0.5], &config, &ab, &mut state, None)
            .expect_err("use_mask without a mask must fail");
        match err {
            ImageEditingError::InvalidImage(msg) => assert!(msg.contains("mask"), "{msg}"),
            other => panic!("Expected InvalidImage, got {other:?}"),
        }
    }

    #[test]
    fn mixed_api_caller_needs_no_error_conversion() {
        // Proves the merge is complete: a function calling both the
        // slice-based `sdedit` API and this module's `EditLatent`-based API
        // needs only `?` at every fallible call site, with no
        // `.into()`/`map_err` bridging the two error types.
        fn mixed(a: &EditLatent, b: &EditLatent, mask: &EditMask) -> Result<(), ImageEditingError> {
            let ab = sdedit::edit_cosine_alpha_bars(10);
            let noise = vec![0.0f32; a.numel()];
            let _ = sdedit::edit_add_noise(&a.data, &noise, 0, &ab)?;
            let _ = blend_with_mask(a, b, mask)?;
            Ok(())
        }

        let a = EditLatent::new(1, 2, 2);
        let b = EditLatent::new(1, 2, 2);
        let mask = EditMask::new(2, 2, 1.0);
        assert!(mixed(&a, &b, &mask).is_ok());
    }

    #[test]
    fn add_edit_noise_propagates_sdedits_dimension_mismatch_directly() {
        // `EditLatent`'s fields are all `pub`, so a caller can build one
        // whose `data` length disagrees with `numel()` — bypassing
        // `EditLatent::new` / `from_data`'s own validation.
        // `add_edit_noise` only checks `numel() == 0` and the noise level
        // itself before delegating to `sdedit::edit_add_noise`, so this is
        // the case that used to reach the (now-removed)
        // `.map_err(|e| NumericalError(format!("add_edit_noise: {e}")))`
        // wrapper. Now that both APIs share `ImageEditingError`, it must
        // surface `sdedit`'s own `DimensionMismatch` unchanged instead of a
        // collapsed `NumericalError` string.
        let malformed = EditLatent {
            channels: 1,
            height: 2,
            width: 2,
            data: vec![0.0f32; 3], // one short of numel() == 4
        };
        let err = add_edit_noise(&malformed, 0.5, 42).expect_err("data/numel mismatch must fail");
        match err {
            ImageEditingError::DimensionMismatch { expected, actual } => {
                assert_eq!(expected, 3, "expected == latent.data.len()");
                assert_eq!(actual, 4, "actual == the numel()-sized noise buffer");
            }
            other => {
                panic!("Expected DimensionMismatch, not a collapsed NumericalError: {other:?}")
            }
        }
    }
}
