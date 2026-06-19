//! Post-processing effects for rendered 3DGS images.
//!
//! Applies visual effects to flat `Vec<f32>` images (H×W×3, RGB in \[0,1\]).
//!
//! # Effects
//!
//! - **Bloom**: light scattering from bright regions via luminance thresholding
//!   and Gaussian blur.
//! - **Vignette**: edge darkening using a smoothstep falloff.
//! - **Chromatic Aberration**: lens color fringing by per-channel sampling offsets.
//! - **Sharpening**: unsharp mask using a Gaussian blur and residual.
//! - **Film Grain**: deterministic xorshift64 noise added to pixels.
//!
//! Effects are composable via [`PostProcessingPipeline`].

use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// PRNG — xorshift64 (same pattern as ssao.rs)
// ─────────────────────────────────────────────────────────────────────────────

/// Advance a 64-bit xorshift state and return the new value.
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

/// Draw a uniformly distributed f32 in [0, 1) from the xorshift64 state.
fn xorshift_f32(state: &mut u64) -> f32 {
    (xorshift64(state) >> 11) as f32 / (1u64 << 53) as f32
}

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors produced by post-processing operations.
#[derive(Debug, Error)]
pub enum PostProcessError {
    /// Data length does not match declared dimensions.
    #[error("Image data length {len} does not match {w}×{h}×3")]
    DataLengthMismatch { len: usize, w: usize, h: usize },

    /// Width must be positive.
    #[error("Width {0} must be positive")]
    InvalidWidth(usize),

    /// Height must be positive.
    #[error("Height {0} must be positive")]
    InvalidHeight(usize),

    /// Invalid configuration parameter.
    #[error("Invalid parameter '{param}': {reason}")]
    InvalidParam { param: String, reason: String },

    /// Kernel size must be odd and positive.
    #[error("Kernel size must be odd and > 0, got {0}")]
    InvalidKernelSize(usize),
}

// ─────────────────────────────────────────────────────────────────────────────
// PostImage
// ─────────────────────────────────────────────────────────────────────────────

/// A post-processing image: H×W×3 flat buffer, RGB in [0, 1].
#[derive(Debug, Clone)]
pub struct PostImage {
    /// Flat pixel data in HWC order: index = (y * width + x) * 3.
    pub data: Vec<f32>,
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
}

impl PostImage {
    /// Construct from existing data, validating dimensions.
    ///
    /// # Errors
    ///
    /// - [`PostProcessError::InvalidWidth`] if `width == 0`.
    /// - [`PostProcessError::InvalidHeight`] if `height == 0`.
    /// - [`PostProcessError::DataLengthMismatch`] if `data.len() != width * height * 3`.
    pub fn new(data: Vec<f32>, width: usize, height: usize) -> Result<Self, PostProcessError> {
        if width == 0 {
            return Err(PostProcessError::InvalidWidth(width));
        }
        if height == 0 {
            return Err(PostProcessError::InvalidHeight(height));
        }
        let expected = width * height * 3;
        if data.len() != expected {
            return Err(PostProcessError::DataLengthMismatch {
                len: data.len(),
                w: width,
                h: height,
            });
        }
        Ok(Self {
            data,
            width,
            height,
        })
    }

    /// Create an all-zero (black) image of the given dimensions.
    ///
    /// # Panics
    ///
    /// Does not panic — zero dimensions simply produce empty data; callers
    /// building zero-size images should use this only for temporary buffers
    /// that are later replaced.
    pub fn zeros(width: usize, height: usize) -> Self {
        let data = vec![0.0_f32; width * height * 3];
        Self {
            data,
            width,
            height,
        }
    }

    /// Return the RGB triple for column `x`, row `y` (clipped to bounds).
    ///
    /// If coordinates are out of range the edge pixel is returned.
    pub fn pixel(&self, x: usize, y: usize) -> [f32; 3] {
        let cx = x.min(self.width.saturating_sub(1));
        let cy = y.min(self.height.saturating_sub(1));
        let base = (cy * self.width + cx) * 3;
        if base + 2 < self.data.len() {
            [self.data[base], self.data[base + 1], self.data[base + 2]]
        } else {
            [0.0, 0.0, 0.0]
        }
    }

    /// Write an RGB triple at column `x`, row `y` (no-op if out of bounds).
    pub fn set_pixel(&mut self, x: usize, y: usize, rgb: [f32; 3]) {
        if x >= self.width || y >= self.height {
            return;
        }
        let base = (y * self.width + x) * 3;
        if base + 2 < self.data.len() {
            self.data[base] = rgb[0];
            self.data[base + 1] = rgb[1];
            self.data[base + 2] = rgb[2];
        }
    }

    /// Bilinear sample with clamp-to-edge.
    ///
    /// Float coordinates `(x, y)` are treated as pixel-center positions.
    /// Out-of-bounds coordinates clamp to the nearest valid pixel.
    pub fn sample_clamp(&self, x: f32, y: f32) -> [f32; 3] {
        if self.width == 0 || self.height == 0 {
            return [0.0, 0.0, 0.0];
        }

        let max_x = (self.width - 1) as f32;
        let max_y = (self.height - 1) as f32;

        let cx = x.clamp(0.0, max_x);
        let cy = y.clamp(0.0, max_y);

        let x0 = cx.floor() as usize;
        let y0 = cy.floor() as usize;
        let x1 = (x0 + 1).min(self.width - 1);
        let y1 = (y0 + 1).min(self.height - 1);

        let tx = cx - x0 as f32;
        let ty = cy - y0 as f32;

        let p00 = self.pixel(x0, y0);
        let p10 = self.pixel(x1, y0);
        let p01 = self.pixel(x0, y1);
        let p11 = self.pixel(x1, y1);

        let mut out = [0.0_f32; 3];
        for i in 0..3 {
            let top = p00[i] * (1.0 - tx) + p10[i] * tx;
            let bot = p01[i] * (1.0 - tx) + p11[i] * tx;
            out[i] = top * (1.0 - ty) + bot * ty;
        }
        out
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Gaussian blur helper (internal)
// ─────────────────────────────────────────────────────────────────────────────

/// Build a 1-D Gaussian kernel.
///
/// Kernel size = `2 * ceil(3 * sigma) + 1` (always odd).
/// Weights normalised to sum 1.
fn make_gaussian_kernel_1d(sigma: f32) -> Vec<f32> {
    let half = (3.0 * sigma).ceil() as isize;
    let size = (2 * half + 1) as usize;
    let mut kernel = Vec::with_capacity(size);

    let denom = 2.0 * sigma * sigma;
    let mut sum = 0.0_f32;

    for k in -half..=half {
        let w = (-(k * k) as f32 / denom).exp();
        kernel.push(w);
        sum += w;
    }

    if sum > 0.0 {
        for w in &mut kernel {
            *w /= sum;
        }
    }

    kernel
}

/// Separable Gaussian blur.  `sigma > 0`, returns a new `PostImage`.
fn gaussian_blur_image(img: &PostImage, sigma: f32) -> PostImage {
    let w = img.width;
    let h = img.height;

    if w == 0 || h == 0 || sigma <= 0.0 {
        return img.clone();
    }

    let kernel = make_gaussian_kernel_1d(sigma);
    let half = (kernel.len() / 2) as isize;

    // --- Horizontal pass: img → temp ---
    let mut temp = PostImage::zeros(w, h);
    for py in 0..h {
        for px in 0..w {
            let mut acc = [0.0_f32; 3];
            for (ki, &kw) in kernel.iter().enumerate() {
                let sx = ((px as isize + ki as isize - half).clamp(0, w as isize - 1)) as usize;
                let p = img.pixel(sx, py);
                acc[0] += p[0] * kw;
                acc[1] += p[1] * kw;
                acc[2] += p[2] * kw;
            }
            temp.set_pixel(px, py, acc);
        }
    }

    // --- Vertical pass: temp → out ---
    let mut out = PostImage::zeros(w, h);
    for py in 0..h {
        for px in 0..w {
            let mut acc = [0.0_f32; 3];
            for (ki, &kw) in kernel.iter().enumerate() {
                let sy = ((py as isize + ki as isize - half).clamp(0, h as isize - 1)) as usize;
                let p = temp.pixel(px, sy);
                acc[0] += p[0] * kw;
                acc[1] += p[1] * kw;
                acc[2] += p[2] * kw;
            }
            out.set_pixel(px, py, acc);
        }
    }

    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Bloom
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the bloom post-processing effect.
#[derive(Debug, Clone)]
pub struct BloomConfig {
    /// Luminance threshold above which pixels are considered "bright". Default: 0.8.
    pub threshold: f32,
    /// Sigma for Gaussian blur of extracted bright regions. Default: 3.0.
    pub blur_sigma: f32,
    /// Bloom intensity: how much to add back. Default: 0.3.
    pub intensity: f32,
    /// Number of blur passes (more = softer bloom). Default: 2.
    pub num_passes: usize,
}

impl Default for BloomConfig {
    fn default() -> Self {
        Self {
            threshold: 0.8,
            blur_sigma: 3.0,
            intensity: 0.3,
            num_passes: 2,
        }
    }
}

/// Apply bloom effect to an image.
///
/// # Algorithm
///
/// 1. Extract bright pixels: `bright[x,y] = max(0, lum - threshold) * rgb/lum`
///    where `lum = 0.299r + 0.587g + 0.114b`.
/// 2. Gaussian blur the bright image (`num_passes` times).
/// 3. `output = clamp(original + intensity * blurred_bright, 0, 1)`.
///
/// # Errors
///
/// - [`PostProcessError::InvalidParam`] if `blur_sigma <= 0`.
pub fn apply_bloom(img: &PostImage, config: &BloomConfig) -> Result<PostImage, PostProcessError> {
    if config.blur_sigma <= 0.0 {
        return Err(PostProcessError::InvalidParam {
            param: "blur_sigma".to_string(),
            reason: "must be positive".to_string(),
        });
    }

    let w = img.width;
    let h = img.height;

    // 1. Extract bright regions.
    let mut bright = PostImage::zeros(w, h);
    for py in 0..h {
        for px in 0..w {
            let rgb = img.pixel(px, py);
            let lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2];
            let excess = (lum - config.threshold).max(0.0);
            if excess > 0.0 && lum > 1e-7 {
                let scale = excess / lum;
                bright.set_pixel(px, py, [rgb[0] * scale, rgb[1] * scale, rgb[2] * scale]);
            }
        }
    }

    // 2. Gaussian blur (num_passes times).
    let mut blurred = bright;
    for _ in 0..config.num_passes {
        blurred = gaussian_blur_image(&blurred, config.blur_sigma);
    }

    // 3. Composite.
    let mut out = PostImage::zeros(w, h);
    for py in 0..h {
        for px in 0..w {
            let orig = img.pixel(px, py);
            let bloom = blurred.pixel(px, py);
            let result = [
                (orig[0] + config.intensity * bloom[0]).clamp(0.0, 1.0),
                (orig[1] + config.intensity * bloom[1]).clamp(0.0, 1.0),
                (orig[2] + config.intensity * bloom[2]).clamp(0.0, 1.0),
            ];
            out.set_pixel(px, py, result);
        }
    }

    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// Vignette
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the vignette post-processing effect.
#[derive(Debug, Clone)]
pub struct VignetteConfig {
    /// Radius of clear center (0.0-1.0, fraction of min(W,H)/2). Default: 0.5.
    pub inner_radius: f32,
    /// Outer radius where vignette is maximum. Default: 1.0.
    pub outer_radius: f32,
    /// Vignette strength (0.0=none, 1.0=full black at outer radius). Default: 0.8.
    pub strength: f32,
    /// Vignette color. Default: `[0.0, 0.0, 0.0]` (black).
    pub color: [f32; 3],
}

impl Default for VignetteConfig {
    fn default() -> Self {
        Self {
            inner_radius: 0.5,
            outer_radius: 1.0,
            strength: 0.8,
            color: [0.0, 0.0, 0.0],
        }
    }
}

/// Apply vignette darkening to image edges.
///
/// # Algorithm
///
/// For each pixel `(x, y)`:
/// ```text
/// cx = (x - W/2) / (min(W,H)/2)
/// cy = (y - H/2) / (min(W,H)/2)
/// dist = sqrt(cx² + cy²)
/// t = smoothstep(inner_radius, outer_radius, dist) * strength
/// output[x,y] = lerp(original, color, t)
/// ```
///
/// # Errors
///
/// - [`PostProcessError::InvalidParam`] if `strength < 0` or `strength > 1`.
pub fn apply_vignette(
    img: &PostImage,
    config: &VignetteConfig,
) -> Result<PostImage, PostProcessError> {
    if config.strength < 0.0 || config.strength > 1.0 {
        return Err(PostProcessError::InvalidParam {
            param: "strength".to_string(),
            reason: "must be in [0.0, 1.0]".to_string(),
        });
    }

    let w = img.width;
    let h = img.height;
    let half_min = (w.min(h) as f32) / 2.0;
    let cx_center = w as f32 / 2.0;
    let cy_center = h as f32 / 2.0;

    let mut out = PostImage::zeros(w, h);

    for py in 0..h {
        for px in 0..w {
            let orig = img.pixel(px, py);

            // Normalized distance from center.
            let nx = if half_min > 0.0 {
                (px as f32 - cx_center) / half_min
            } else {
                0.0
            };
            let ny = if half_min > 0.0 {
                (py as f32 - cy_center) / half_min
            } else {
                0.0
            };
            let dist = (nx * nx + ny * ny).sqrt();

            // Smoothstep from inner to outer radius.
            let t = smoothstep(config.inner_radius, config.outer_radius, dist) * config.strength;

            let result = [
                lerp(orig[0], config.color[0], t),
                lerp(orig[1], config.color[1], t),
                lerp(orig[2], config.color[2], t),
            ];
            out.set_pixel(px, py, result);
        }
    }

    Ok(out)
}

/// Smoothstep `clamp((x-a)/(b-a), 0, 1)² * (3 - 2 * clamp(...))`.
#[inline]
fn smoothstep(a: f32, b: f32, x: f32) -> f32 {
    let t = if (b - a).abs() < f32::EPSILON {
        if x >= b {
            1.0
        } else {
            0.0
        }
    } else {
        ((x - a) / (b - a)).clamp(0.0, 1.0)
    };
    t * t * (3.0 - 2.0 * t)
}

/// Linear interpolation: `a * (1-t) + b * t`.
#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a * (1.0 - t) + b * t
}

// ─────────────────────────────────────────────────────────────────────────────
// Chromatic Aberration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for chromatic aberration.
#[derive(Debug, Clone)]
pub struct ChromaticAberrationConfig {
    /// Red channel displacement in pixels (dx, dy). Default: `[2.0, 0.0]`.
    pub red_offset: [f32; 2],
    /// Blue channel displacement in pixels (dx, dy). Default: `[-2.0, 0.0]`.
    pub blue_offset: [f32; 2],
    /// Green channel displacement in pixels (dx, dy). Default: `[0.0, 0.0]`.
    pub green_offset: [f32; 2],
}

impl Default for ChromaticAberrationConfig {
    fn default() -> Self {
        Self {
            red_offset: [2.0, 0.0],
            blue_offset: [-2.0, 0.0],
            green_offset: [0.0, 0.0],
        }
    }
}

/// Apply chromatic aberration by sampling each channel at different positions.
///
/// For each output pixel `(x, y)`:
/// ```text
/// r = sample_clamp(x + red_offset.x,   y + red_offset.y).r
/// g = sample_clamp(x + green_offset.x, y + green_offset.y).g
/// b = sample_clamp(x + blue_offset.x,  y + blue_offset.y).b
/// ```
pub fn apply_chromatic_aberration(
    img: &PostImage,
    config: &ChromaticAberrationConfig,
) -> Result<PostImage, PostProcessError> {
    let w = img.width;
    let h = img.height;
    let mut out = PostImage::zeros(w, h);

    for py in 0..h {
        for px in 0..w {
            let fx = px as f32;
            let fy = py as f32;

            let r_sample = img.sample_clamp(fx + config.red_offset[0], fy + config.red_offset[1]);
            let g_sample =
                img.sample_clamp(fx + config.green_offset[0], fy + config.green_offset[1]);
            let b_sample = img.sample_clamp(fx + config.blue_offset[0], fy + config.blue_offset[1]);

            out.set_pixel(px, py, [r_sample[0], g_sample[1], b_sample[2]]);
        }
    }

    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// Sharpening
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for unsharp mask sharpening.
#[derive(Debug, Clone)]
pub struct SharpenConfig {
    /// Sharpening strength. Default: 0.5.
    pub strength: f32,
    /// Radius: `1 = 3×3`, `2 = 5×5`, etc. Default: 1.
    pub radius: usize,
}

impl Default for SharpenConfig {
    fn default() -> Self {
        Self {
            strength: 0.5,
            radius: 1,
        }
    }
}

/// Apply unsharp mask sharpening.
///
/// # Algorithm
///
/// 1. Gaussian blur with `sigma = radius` (kernel size `2*radius+1`).
/// 2. `detail = original - blurred`.
/// 3. `output = clamp(original + strength * detail, 0, 1)`.
///
/// # Errors
///
/// - [`PostProcessError::InvalidParam`] if `strength < 0`.
pub fn apply_sharpening(
    img: &PostImage,
    config: &SharpenConfig,
) -> Result<PostImage, PostProcessError> {
    if config.strength < 0.0 {
        return Err(PostProcessError::InvalidParam {
            param: "strength".to_string(),
            reason: "must be non-negative".to_string(),
        });
    }

    let sigma = config.radius.max(1) as f32;
    let blurred = gaussian_blur_image(img, sigma);

    let w = img.width;
    let h = img.height;
    let mut out = PostImage::zeros(w, h);

    for py in 0..h {
        for px in 0..w {
            let orig = img.pixel(px, py);
            let blur = blurred.pixel(px, py);
            let result = [
                (orig[0] + config.strength * (orig[0] - blur[0])).clamp(0.0, 1.0),
                (orig[1] + config.strength * (orig[1] - blur[1])).clamp(0.0, 1.0),
                (orig[2] + config.strength * (orig[2] - blur[2])).clamp(0.0, 1.0),
            ];
            out.set_pixel(px, py, result);
        }
    }

    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// Film Grain
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for film grain noise.
#[derive(Debug, Clone)]
pub struct FilmGrainConfig {
    /// Grain intensity. Default: 0.05.
    pub intensity: f32,
    /// Monochromatic (same noise per channel) or colored. Default: `true` (mono).
    pub monochromatic: bool,
    /// Seed for deterministic noise. Default: 42.
    pub seed: u64,
}

impl Default for FilmGrainConfig {
    fn default() -> Self {
        Self {
            intensity: 0.05,
            monochromatic: true,
            seed: 42,
        }
    }
}

/// Add film grain using xorshift64.
///
/// For each pixel, samples noise in `[-1, 1]`, multiplies by `intensity`,
/// adds to pixel value, then clamps to `[0, 1]`.
///
/// This function is infallible.
pub fn apply_film_grain(img: &PostImage, config: &FilmGrainConfig) -> PostImage {
    let w = img.width;
    let h = img.height;
    let mut out = PostImage::zeros(w, h);
    let mut state = if config.seed == 0 {
        0xDEAD_BEEF_CAFE_1234u64
    } else {
        config.seed
    };

    for py in 0..h {
        for px in 0..w {
            let orig = img.pixel(px, py);

            let result = if config.monochromatic {
                let noise = 2.0 * xorshift_f32(&mut state) - 1.0;
                let n = noise * config.intensity;
                [
                    (orig[0] + n).clamp(0.0, 1.0),
                    (orig[1] + n).clamp(0.0, 1.0),
                    (orig[2] + n).clamp(0.0, 1.0),
                ]
            } else {
                let nr = (2.0 * xorshift_f32(&mut state) - 1.0) * config.intensity;
                let ng = (2.0 * xorshift_f32(&mut state) - 1.0) * config.intensity;
                let nb = (2.0 * xorshift_f32(&mut state) - 1.0) * config.intensity;
                [
                    (orig[0] + nr).clamp(0.0, 1.0),
                    (orig[1] + ng).clamp(0.0, 1.0),
                    (orig[2] + nb).clamp(0.0, 1.0),
                ]
            };

            out.set_pixel(px, py, result);
        }
    }

    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Composable Pipeline
// ─────────────────────────────────────────────────────────────────────────────

/// A single post-processing effect, used in a [`PostProcessingPipeline`].
#[derive(Debug, Clone)]
pub enum PostEffect {
    /// Bloom light scattering.
    Bloom(BloomConfig),
    /// Edge vignette darkening.
    Vignette(VignetteConfig),
    /// Chromatic aberration (lens fringing).
    ChromaticAberration(ChromaticAberrationConfig),
    /// Unsharp mask sharpening.
    Sharpen(SharpenConfig),
    /// Film grain noise.
    FilmGrain(FilmGrainConfig),
}

/// Ordered sequence of post-processing effects applied to a [`PostImage`].
pub struct PostProcessingPipeline {
    effects: Vec<PostEffect>,
}

impl PostProcessingPipeline {
    /// Create an empty pipeline.
    pub fn new() -> Self {
        Self {
            effects: Vec::new(),
        }
    }

    /// Append an effect and return `self` (builder pattern).
    pub fn push_effect(mut self, effect: PostEffect) -> Self {
        self.effects.push(effect);
        self
    }

    /// Number of effects in the pipeline.
    pub fn num_effects(&self) -> usize {
        self.effects.len()
    }

    /// Apply all effects in order to `img`, returning the processed image.
    ///
    /// # Errors
    ///
    /// Propagates any error from the underlying effect functions.
    pub fn apply(&self, img: &PostImage) -> Result<PostImage, PostProcessError> {
        let mut current = img.clone();
        for effect in &self.effects {
            current = match effect {
                PostEffect::Bloom(cfg) => apply_bloom(&current, cfg)?,
                PostEffect::Vignette(cfg) => apply_vignette(&current, cfg)?,
                PostEffect::ChromaticAberration(cfg) => apply_chromatic_aberration(&current, cfg)?,
                PostEffect::Sharpen(cfg) => apply_sharpening(&current, cfg)?,
                PostEffect::FilmGrain(cfg) => apply_film_grain(&current, cfg),
            };
        }
        Ok(current)
    }

    /// Standard rendering pipeline for 3DGS avatar outputs.
    ///
    /// Effects: `Sharpen(default)` → `Vignette(default)`.
    pub fn standard_avatar() -> Self {
        Self::new()
            .push_effect(PostEffect::Sharpen(SharpenConfig::default()))
            .push_effect(PostEffect::Vignette(VignetteConfig::default()))
    }

    /// Cinematic rendering pipeline.
    ///
    /// Effects: `Bloom(default)` → `ChromaticAberration(default)` →
    /// `Vignette(default)` → `FilmGrain(default)`.
    pub fn cinematic() -> Self {
        Self::new()
            .push_effect(PostEffect::Bloom(BloomConfig::default()))
            .push_effect(PostEffect::ChromaticAberration(
                ChromaticAberrationConfig::default(),
            ))
            .push_effect(PostEffect::Vignette(VignetteConfig::default()))
            .push_effect(PostEffect::FilmGrain(FilmGrainConfig::default()))
    }
}

impl Default for PostProcessingPipeline {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helper ────────────────────────────────────────────────────────────────

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    fn solid_image(w: usize, h: usize, rgb: [f32; 3]) -> PostImage {
        let data: Vec<f32> = (0..w * h).flat_map(|_| [rgb[0], rgb[1], rgb[2]]).collect();
        PostImage::zeros(w, h).tap(|img| img.data = data)
    }

    fn gradient_image(w: usize, h: usize) -> PostImage {
        let data: Vec<f32> = (0..w * h * 3).map(|i| (i % 97) as f32 / 96.0).collect();
        PostImage {
            data,
            width: w,
            height: h,
        }
    }

    fn all_in_range(img: &PostImage) -> bool {
        img.data.iter().all(|&v| (0.0..=1.0).contains(&v))
    }

    // Tap helper for test ergonomics.
    trait Tap: Sized {
        fn tap(mut self, f: impl FnOnce(&mut Self)) -> Self {
            f(&mut self);
            self
        }
    }
    impl<T> Tap for T {}

    // ── 1. PostImage::new correct dimensions ──────────────────────────────────

    #[test]
    fn test_post_image_new_correct() {
        let data = vec![0.5_f32; 4 * 3 * 3];
        let img = PostImage::new(data, 4, 3).expect("PostImage::new failed");
        assert_eq!(img.width, 4);
        assert_eq!(img.height, 3);
        assert_eq!(img.data.len(), 4 * 3 * 3);
    }

    // ── 2. PostImage::new wrong length → DataLengthMismatch ───────────────────

    #[test]
    fn test_post_image_new_wrong_length() {
        let data = vec![0.0_f32; 10]; // wrong
        let result = PostImage::new(data, 4, 3);
        assert!(
            matches!(result, Err(PostProcessError::DataLengthMismatch { .. })),
            "Expected DataLengthMismatch"
        );
    }

    // ── 3. pixel / set_pixel roundtrip ───────────────────────────────────────

    #[test]
    fn test_pixel_set_pixel_roundtrip() {
        let mut img = PostImage::zeros(5, 5);
        let color = [0.1, 0.5, 0.9];
        img.set_pixel(2, 3, color);
        let got = img.pixel(2, 3);
        for i in 0..3 {
            assert!(
                approx_eq(got[i], color[i], 1e-6),
                "channel {i}: expected {}, got {}",
                color[i],
                got[i]
            );
        }
    }

    // ── 4. apply_bloom output same size as input ──────────────────────────────

    #[test]
    fn test_bloom_output_same_size() {
        let img = gradient_image(8, 6);
        let out = apply_bloom(&img, &BloomConfig::default()).expect("apply_bloom failed");
        assert_eq!(out.width, img.width);
        assert_eq!(out.height, img.height);
        assert_eq!(out.data.len(), img.data.len());
    }

    // ── 5. apply_bloom output values in [0, 1] ────────────────────────────────

    #[test]
    fn test_bloom_output_in_range() {
        let img = gradient_image(8, 8);
        let out = apply_bloom(&img, &BloomConfig::default()).expect("apply_bloom failed");
        assert!(all_in_range(&out), "Bloom output out of [0,1]");
    }

    // ── 6. apply_bloom dark image: near-identical output ─────────────────────

    #[test]
    fn test_bloom_dark_image_near_identical() {
        // All-0.5 image is below threshold=0.8, so bloom contributes nothing.
        let img = solid_image(8, 8, [0.5, 0.5, 0.5]);
        let out = apply_bloom(&img, &BloomConfig::default()).expect("apply_bloom failed");
        for (a, b) in img.data.iter().zip(out.data.iter()) {
            assert!(
                approx_eq(*a, *b, 1e-5),
                "Below-threshold image should be unchanged: {a} vs {b}"
            );
        }
    }

    // ── 7. apply_vignette center pixel unchanged ──────────────────────────────

    #[test]
    fn test_vignette_center_unchanged() {
        let w = 9_usize;
        let h = 9_usize;
        let img = solid_image(w, h, [0.6, 0.4, 0.2]);
        let out = apply_vignette(&img, &VignetteConfig::default()).expect("apply_vignette failed");

        // Center pixel: dist = 0.0 < inner_radius=0.5 → t=0 → unchanged.
        let cx = w / 2;
        let cy = h / 2;
        let orig = img.pixel(cx, cy);
        let got = out.pixel(cx, cy);
        for i in 0..3 {
            assert!(
                approx_eq(orig[i], got[i], 1e-5),
                "Center pixel channel {i} changed: {} → {}",
                orig[i],
                got[i]
            );
        }
    }

    // ── 8. apply_vignette corner pixels darker ────────────────────────────────

    #[test]
    fn test_vignette_corners_darker() {
        let w = 20_usize;
        let h = 20_usize;
        let img = solid_image(w, h, [1.0, 1.0, 1.0]);
        let out = apply_vignette(&img, &VignetteConfig::default()).expect("apply_vignette failed");

        // Corner (0, 0) must be darker than center.
        let center = out.pixel(w / 2, h / 2);
        let corner = out.pixel(0, 0);
        assert!(
            corner[0] < center[0],
            "Corner should be darker than center: corner={} center={}",
            corner[0],
            center[0]
        );
    }

    // ── 9. apply_vignette output values in [0, 1] ─────────────────────────────

    #[test]
    fn test_vignette_output_in_range() {
        let img = gradient_image(10, 10);
        let out = apply_vignette(&img, &VignetteConfig::default()).expect("apply_vignette failed");
        assert!(all_in_range(&out), "Vignette output out of [0,1]");
    }

    // ── 10. apply_chromatic_aberration output same size ───────────────────────

    #[test]
    fn test_chrab_output_same_size() {
        let img = gradient_image(12, 8);
        let out = apply_chromatic_aberration(&img, &ChromaticAberrationConfig::default())
            .expect("apply_chromatic_aberration failed");
        assert_eq!(out.width, img.width);
        assert_eq!(out.height, img.height);
    }

    // ── 11. apply_chromatic_aberration zero offsets: nearly identical ─────────

    #[test]
    fn test_chrab_zero_offsets_identical() {
        let img = gradient_image(8, 8);
        let config = ChromaticAberrationConfig {
            red_offset: [0.0, 0.0],
            green_offset: [0.0, 0.0],
            blue_offset: [0.0, 0.0],
        };
        let out =
            apply_chromatic_aberration(&img, &config).expect("apply_chromatic_aberration failed");
        for (a, b) in img.data.iter().zip(out.data.iter()) {
            assert!(
                approx_eq(*a, *b, 1e-5),
                "Zero-offset chromatic aberration must be identity: {a} vs {b}"
            );
        }
    }

    // ── 12. apply_sharpening output same size ─────────────────────────────────

    #[test]
    fn test_sharpen_output_same_size() {
        let img = gradient_image(10, 8);
        let out =
            apply_sharpening(&img, &SharpenConfig::default()).expect("apply_sharpening failed");
        assert_eq!(out.width, img.width);
        assert_eq!(out.height, img.height);
    }

    // ── 13. apply_sharpening output values in [0, 1] ──────────────────────────

    #[test]
    fn test_sharpen_output_in_range() {
        let img = gradient_image(10, 10);
        let out =
            apply_sharpening(&img, &SharpenConfig::default()).expect("apply_sharpening failed");
        assert!(all_in_range(&out), "Sharpening output out of [0,1]");
    }

    // ── 14. apply_sharpening uniform image unchanged ──────────────────────────

    #[test]
    fn test_sharpen_uniform_image_unchanged() {
        let img = solid_image(8, 8, [0.5, 0.5, 0.5]);
        let out =
            apply_sharpening(&img, &SharpenConfig::default()).expect("apply_sharpening failed");
        for (a, b) in img.data.iter().zip(out.data.iter()) {
            assert!(
                approx_eq(*a, *b, 1e-4),
                "Uniform image detail=0, sharpened should be unchanged: {a} vs {b}"
            );
        }
    }

    // ── 15. apply_film_grain output same size ─────────────────────────────────

    #[test]
    fn test_film_grain_output_same_size() {
        let img = gradient_image(8, 6);
        let out = apply_film_grain(&img, &FilmGrainConfig::default());
        assert_eq!(out.width, img.width);
        assert_eq!(out.height, img.height);
        assert_eq!(out.data.len(), img.data.len());
    }

    // ── 16. apply_film_grain output values in [0, 1] ──────────────────────────

    #[test]
    fn test_film_grain_output_in_range() {
        let img = gradient_image(8, 8);
        let out = apply_film_grain(&img, &FilmGrainConfig::default());
        assert!(all_in_range(&out), "Film grain output out of [0,1]");
    }

    // ── 17. apply_film_grain same seed → same output ──────────────────────────

    #[test]
    fn test_film_grain_deterministic() {
        let img = gradient_image(8, 8);
        let config = FilmGrainConfig {
            seed: 12345,
            ..FilmGrainConfig::default()
        };
        let out1 = apply_film_grain(&img, &config);
        let out2 = apply_film_grain(&img, &config);
        for (a, b) in out1.data.iter().zip(out2.data.iter()) {
            assert!(
                approx_eq(*a, *b, 1e-7),
                "Same seed must produce same output: {a} vs {b}"
            );
        }
    }

    // ── 18. PostProcessingPipeline::new starts empty ──────────────────────────

    #[test]
    fn test_pipeline_new_empty() {
        let pipeline = PostProcessingPipeline::new();
        assert_eq!(pipeline.num_effects(), 0);
    }

    // ── 19. PostProcessingPipeline::standard_avatar has correct effects ───────

    #[test]
    fn test_pipeline_standard_avatar_effects() {
        let pipeline = PostProcessingPipeline::standard_avatar();
        assert_eq!(
            pipeline.num_effects(),
            2,
            "standard_avatar should have 2 effects"
        );
        assert!(
            matches!(pipeline.effects[0], PostEffect::Sharpen(_)),
            "First effect should be Sharpen"
        );
        assert!(
            matches!(pipeline.effects[1], PostEffect::Vignette(_)),
            "Second effect should be Vignette"
        );
    }

    // ── 20. PostProcessingPipeline::apply output same size as input ───────────

    #[test]
    fn test_pipeline_apply_same_size() {
        let img = gradient_image(10, 8);
        let pipeline = PostProcessingPipeline::standard_avatar();
        let out = pipeline.apply(&img).expect("pipeline.apply failed");
        assert_eq!(out.width, img.width);
        assert_eq!(out.height, img.height);
        assert_eq!(out.data.len(), img.data.len());
    }

    // ── 21. PostProcessingPipeline::apply output values in [0, 1] ────────────

    #[test]
    fn test_pipeline_apply_in_range() {
        let img = gradient_image(12, 10);
        let pipeline = PostProcessingPipeline::cinematic();
        let out = pipeline
            .apply(&img)
            .expect("cinematic pipeline.apply failed");
        assert!(all_in_range(&out), "Cinematic pipeline output out of [0,1]");
    }
}
