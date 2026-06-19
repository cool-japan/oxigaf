//! Training augmentation transforms for GAF images.
//!
//! Images are represented as flat `Vec<f32>` (H×W×3, row-major RGB, values in \[0,1\]).
//! All transforms are deterministic given a seed and use an inline xorshift64 PRNG
//! (no `rand` crate dependency).

use thiserror::Error;

// ---- Error Type ---------------------------------------------------------------

/// Errors produced by augmentation operations.
#[derive(Debug, Error)]
pub enum AugmentationError {
    #[error("Image dimensions {width}×{height} incompatible with channel count {channels}")]
    DimensionMismatch {
        width: usize,
        height: usize,
        channels: usize,
    },

    #[error("Image data length {len} does not match {width}×{height}×{channels}")]
    DataLengthMismatch {
        len: usize,
        width: usize,
        height: usize,
        channels: usize,
    },

    #[error("Invalid parameter: {0}")]
    InvalidParam(String),

    #[error("Crop region {x}+{w} > {img_w} or {y}+{h} > {img_h}")]
    CropOutOfBounds {
        x: usize,
        y: usize,
        w: usize,
        h: usize,
        img_w: usize,
        img_h: usize,
    },
}

// ---- PRNG helpers (xorshift64, no rand crate) ---------------------------------

/// xorshift64 PRNG step — modifies state in place, returns next pseudo-random u64.
#[inline]
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// Uniform float in [0, 1) with 53 bits of precision.
#[inline]
fn xorshift_f32(state: &mut u64) -> f32 {
    (xorshift64(state) >> 11) as f32 / (1u64 << 53) as f32
}

/// Uniform float in [lo, hi).
#[inline]
fn xorshift_range(state: &mut u64, lo: f32, hi: f32) -> f32 {
    lo + xorshift_f32(state) * (hi - lo)
}

// ---- AugImage ----------------------------------------------------------------

/// Flat row-major RGB image, values in [0, 1].
#[derive(Debug, Clone)]
pub struct AugImage {
    /// Pixel data: H × W × 3 interleaved RGB.
    pub data: Vec<f32>,
    pub width: usize,
    pub height: usize,
}

impl AugImage {
    /// Construct from existing data, validating dimensions.
    pub fn new(data: Vec<f32>, width: usize, height: usize) -> Result<Self, AugmentationError> {
        let expected = width * height * 3;
        if data.len() != expected {
            return Err(AugmentationError::DataLengthMismatch {
                len: data.len(),
                width,
                height,
                channels: 3,
            });
        }
        Ok(Self {
            data,
            width,
            height,
        })
    }

    /// Create a black (all-zero) image of the given size.
    pub fn zeros(width: usize, height: usize) -> Self {
        Self {
            data: vec![0.0_f32; width * height * 3],
            width,
            height,
        }
    }

    /// Get the RGB triple at column `x`, row `y`.
    #[inline]
    pub fn pixel(&self, x: usize, y: usize) -> [f32; 3] {
        let base = (y * self.width + x) * 3;
        [self.data[base], self.data[base + 1], self.data[base + 2]]
    }

    /// Set the RGB triple at column `x`, row `y`.
    #[inline]
    pub fn set_pixel(&mut self, x: usize, y: usize, rgb: [f32; 3]) {
        let base = (y * self.width + x) * 3;
        self.data[base] = rgb[0];
        self.data[base + 1] = rgb[1];
        self.data[base + 2] = rgb[2];
    }

    /// Total number of pixels.
    #[inline]
    pub fn num_pixels(&self) -> usize {
        self.width * self.height
    }

    /// Sample with bilinear interpolation.
    ///
    /// Coordinates are clamped to `[0, W-1]` and `[0, H-1]`, so this never
    /// returns out-of-bounds black (clamp-to-edge semantics).
    pub fn sample_bilinear(&self, fx: f32, fy: f32) -> [f32; 3] {
        let w = self.width as f32;
        let h = self.height as f32;

        // Clamp to valid range
        let cx = fx.clamp(0.0, w - 1.0);
        let cy = fy.clamp(0.0, h - 1.0);

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

        let w00 = (1.0 - tx) * (1.0 - ty);
        let w10 = tx * (1.0 - ty);
        let w01 = (1.0 - tx) * ty;
        let w11 = tx * ty;

        [
            w00 * p00[0] + w10 * p10[0] + w01 * p01[0] + w11 * p11[0],
            w00 * p00[1] + w10 * p10[1] + w01 * p01[1] + w11 * p11[1],
            w00 * p00[2] + w10 * p10[2] + w01 * p01[2] + w11 * p11[2],
        ]
    }
}

// ---- RGB ↔ HSV conversion ----------------------------------------------------

/// Convert RGB (all in [0, 1]) to HSV (all in [0, 1]).
pub fn rgb_to_hsv(r: f32, g: f32, b: f32) -> [f32; 3] {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;

    let v = max;
    let s = if max > 1e-7 { delta / max } else { 0.0 };

    let h = if delta < 1e-7 {
        // Achromatic
        0.0
    } else if (max - r).abs() < 1e-7 {
        // Red is max
        let sector = (g - b) / delta;
        let raw = sector % 6.0;
        // rem_euclid to handle negatives
        raw.rem_euclid(6.0) / 6.0
    } else if (max - g).abs() < 1e-7 {
        // Green is max
        ((b - r) / delta + 2.0) / 6.0
    } else {
        // Blue is max
        ((r - g) / delta + 4.0) / 6.0
    };

    [h, s, v]
}

/// Convert HSV (all in [0, 1]) to RGB (all in [0, 1]).
pub fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [f32; 3] {
    if s < 1e-7 {
        return [v, v, v];
    }

    let hh = (h * 6.0).rem_euclid(6.0);
    let i = hh.floor() as u32;
    let f = hh - i as f32;

    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));

    match i % 6 {
        0 => [v, t, p],
        1 => [q, v, p],
        2 => [p, v, t],
        3 => [p, q, v],
        4 => [t, p, v],
        5 => [v, p, q],
        _ => [v, v, v], // unreachable
    }
}

// ---- ColorJitter --------------------------------------------------------------

/// Randomly adjust brightness, contrast, saturation, and hue.
#[derive(Debug, Clone)]
pub struct ColorJitter {
    /// Multiplicative brightness factor range, e.g. `[0.8, 1.2]`.
    pub brightness_range: [f32; 2],
    /// Contrast factor range, e.g. `[0.8, 1.2]`.
    pub contrast_range: [f32; 2],
    /// Saturation factor range, e.g. `[0.8, 1.2]`.
    pub saturation_range: [f32; 2],
    /// Hue shift in `[-0.5, 0.5]` range.
    pub hue_shift_range: [f32; 2],
}

impl ColorJitter {
    /// Construct with explicit parameter ranges, validating all inputs.
    pub fn new(
        brightness_range: [f32; 2],
        contrast_range: [f32; 2],
        saturation_range: [f32; 2],
        hue_shift_range: [f32; 2],
    ) -> Result<Self, AugmentationError> {
        if brightness_range[0] > brightness_range[1] {
            return Err(AugmentationError::InvalidParam(
                "brightness_range[0] > brightness_range[1]".into(),
            ));
        }
        if contrast_range[0] > contrast_range[1] {
            return Err(AugmentationError::InvalidParam(
                "contrast_range[0] > contrast_range[1]".into(),
            ));
        }
        if saturation_range[0] > saturation_range[1] {
            return Err(AugmentationError::InvalidParam(
                "saturation_range[0] > saturation_range[1]".into(),
            ));
        }
        if hue_shift_range[0] > hue_shift_range[1] {
            return Err(AugmentationError::InvalidParam(
                "hue_shift_range[0] > hue_shift_range[1]".into(),
            ));
        }
        if hue_shift_range[0] < -0.5 || hue_shift_range[1] > 0.5 {
            return Err(AugmentationError::InvalidParam(
                "hue_shift_range must be within [-0.5, 0.5]".into(),
            ));
        }
        Ok(Self {
            brightness_range,
            contrast_range,
            saturation_range,
            hue_shift_range,
        })
    }

    /// Small-jitter defaults suitable for most training scenarios.
    ///
    /// Brightness [0.9, 1.1], contrast [0.9, 1.1], saturation [0.9, 1.1], hue [-0.05, 0.05].
    pub fn small() -> Self {
        Self {
            brightness_range: [0.9, 1.1],
            contrast_range: [0.9, 1.1],
            saturation_range: [0.9, 1.1],
            hue_shift_range: [-0.05, 0.05],
        }
    }

    /// Apply color jitter to `img`, drawing random factors from the configured ranges.
    ///
    /// Steps (in order):
    /// 1. **Brightness**: `pixel *= brightness_factor`
    /// 2. **Contrast**: `pixel = 0.5 + (pixel - 0.5) * contrast_factor`
    /// 3. **Saturation**: lerp each pixel toward its grayscale equivalent
    /// 4. **Hue shift**: convert RGB→HSV, shift H, convert back
    pub fn apply(&self, img: &AugImage, rng: &mut u64) -> Result<AugImage, AugmentationError> {
        let brightness = xorshift_range(rng, self.brightness_range[0], self.brightness_range[1]);
        let contrast = xorshift_range(rng, self.contrast_range[0], self.contrast_range[1]);
        let saturation = xorshift_range(rng, self.saturation_range[0], self.saturation_range[1]);
        let hue_shift = xorshift_range(rng, self.hue_shift_range[0], self.hue_shift_range[1]);

        let mut out = AugImage::zeros(img.width, img.height);

        for y in 0..img.height {
            for x in 0..img.width {
                let [r, g, b] = img.pixel(x, y);

                // 1. Brightness
                let r = (r * brightness).clamp(0.0, 1.0);
                let g = (g * brightness).clamp(0.0, 1.0);
                let b = (b * brightness).clamp(0.0, 1.0);

                // 2. Contrast
                let r = (0.5 + (r - 0.5) * contrast).clamp(0.0, 1.0);
                let g = (0.5 + (g - 0.5) * contrast).clamp(0.0, 1.0);
                let b = (0.5 + (b - 0.5) * contrast).clamp(0.0, 1.0);

                // 3. Saturation: lerp toward grayscale luminance
                let gray = 0.299 * r + 0.587 * g + 0.114 * b;
                let r = (gray + (r - gray) * saturation).clamp(0.0, 1.0);
                let g = (gray + (g - gray) * saturation).clamp(0.0, 1.0);
                let b = (gray + (b - gray) * saturation).clamp(0.0, 1.0);

                // 4. Hue shift via HSV
                let [h, s, v] = rgb_to_hsv(r, g, b);
                let h_new = (h + hue_shift + 1.0).rem_euclid(1.0);
                let [r, g, b] = hsv_to_rgb(h_new, s, v);

                out.set_pixel(x, y, [r, g, b]);
            }
        }

        Ok(out)
    }
}

impl Default for ColorJitter {
    fn default() -> Self {
        Self::small()
    }
}

// ---- RandomFlip --------------------------------------------------------------

/// Random horizontal and/or vertical flip.
#[derive(Debug, Clone)]
pub struct RandomFlip {
    /// Probability of horizontal flip in `[0, 1]`.
    pub p_horizontal: f32,
    /// Probability of vertical flip in `[0, 1]`.
    pub p_vertical: f32,
}

impl RandomFlip {
    /// Construct with validated flip probabilities.
    pub fn new(p_h: f32, p_v: f32) -> Result<Self, AugmentationError> {
        if !(0.0..=1.0).contains(&p_h) {
            return Err(AugmentationError::InvalidParam(format!(
                "p_horizontal {p_h} not in [0, 1]"
            )));
        }
        if !(0.0..=1.0).contains(&p_v) {
            return Err(AugmentationError::InvalidParam(format!(
                "p_vertical {p_v} not in [0, 1]"
            )));
        }
        Ok(Self {
            p_horizontal: p_h,
            p_vertical: p_v,
        })
    }

    /// Horizontal flip only with p=0.5.
    pub fn horizontal_only() -> Self {
        Self {
            p_horizontal: 0.5,
            p_vertical: 0.0,
        }
    }

    /// Apply random flip(s).
    pub fn apply(&self, img: &AugImage, rng: &mut u64) -> AugImage {
        let do_h = xorshift_f32(rng) < self.p_horizontal;
        let do_v = xorshift_f32(rng) < self.p_vertical;

        let mut out = AugImage::zeros(img.width, img.height);

        for y in 0..img.height {
            for x in 0..img.width {
                let src_x = if do_h { img.width - 1 - x } else { x };
                let src_y = if do_v { img.height - 1 - y } else { y };
                out.set_pixel(x, y, img.pixel(src_x, src_y));
            }
        }

        out
    }
}

impl Default for RandomFlip {
    fn default() -> Self {
        Self::horizontal_only()
    }
}

// ---- RandomCropResize --------------------------------------------------------

/// Random crop followed by bilinear resize back to original dimensions.
#[derive(Debug, Clone)]
pub struct RandomCropResize {
    /// Minimum crop size as fraction of original area, e.g. `0.7`.
    pub min_crop_fraction: f32,
    /// Aspect ratio range `[min, max]`, e.g. `[0.75, 1.33]`.
    pub aspect_ratio_range: [f32; 2],
}

impl RandomCropResize {
    /// Construct with validated parameters.
    pub fn new(min_crop: f32, aspect_range: [f32; 2]) -> Result<Self, AugmentationError> {
        if !(0.0..=1.0).contains(&min_crop) {
            return Err(AugmentationError::InvalidParam(format!(
                "min_crop_fraction {min_crop} not in (0, 1]"
            )));
        }
        if aspect_range[0] > aspect_range[1] || aspect_range[0] <= 0.0 {
            return Err(AugmentationError::InvalidParam(
                "aspect_ratio_range must satisfy 0 < min <= max".into(),
            ));
        }
        Ok(Self {
            min_crop_fraction: min_crop,
            aspect_ratio_range: aspect_range,
        })
    }

    /// Conservative defaults: min_crop=0.8, aspect=[0.9, 1.1].
    pub fn conservative() -> Self {
        Self {
            min_crop_fraction: 0.8,
            aspect_ratio_range: [0.9, 1.1],
        }
    }

    /// Apply random crop-and-resize.
    ///
    /// 1. Sample a crop area fraction in `[min_crop_fraction, 1.0]`.
    /// 2. Sample an aspect ratio from the configured range.
    /// 3. Derive crop width and height from area and aspect.
    /// 4. Clamp to image bounds, then sample a random top-left.
    /// 5. Bilinear-resize the crop back to the original dimensions.
    pub fn apply(&self, img: &AugImage, rng: &mut u64) -> Result<AugImage, AugmentationError> {
        let w = img.width as f32;
        let h = img.height as f32;

        // Sample area fraction and aspect ratio
        let area_frac = xorshift_range(rng, self.min_crop_fraction, 1.0);
        let aspect = xorshift_range(rng, self.aspect_ratio_range[0], self.aspect_ratio_range[1]);

        // Crop dimensions derived from: area = area_frac * W * H
        //   crop_w = sqrt(area * aspect)
        //   crop_h = sqrt(area / aspect)
        let area = area_frac * w * h;
        let crop_w_f = (area * aspect).sqrt();
        let crop_h_f = (area / aspect).sqrt();

        // Round and clamp to valid pixel dimensions
        let crop_w = (crop_w_f.round() as usize).clamp(1, img.width);
        let crop_h = (crop_h_f.round() as usize).clamp(1, img.height);

        // Sample top-left corner
        let max_x = img.width - crop_w;
        let max_y = img.height - crop_h;

        let x0 = if max_x == 0 {
            0
        } else {
            (xorshift_f32(rng) * max_x as f32) as usize % (max_x + 1)
        };
        let y0 = if max_y == 0 {
            0
        } else {
            (xorshift_f32(rng) * max_y as f32) as usize % (max_y + 1)
        };

        // Bilinear resize crop → original dimensions
        let mut out = AugImage::zeros(img.width, img.height);
        let scale_x = crop_w as f32 / img.width as f32;
        let scale_y = crop_h as f32 / img.height as f32;

        for oy in 0..img.height {
            for ox in 0..img.width {
                let src_x = x0 as f32 + ox as f32 * scale_x;
                let src_y = y0 as f32 + oy as f32 * scale_y;
                let rgb = img.sample_bilinear(src_x, src_y);
                out.set_pixel(ox, oy, rgb);
            }
        }

        Ok(out)
    }
}

impl Default for RandomCropResize {
    fn default() -> Self {
        Self::conservative()
    }
}

// ---- GaussianNoise -----------------------------------------------------------

/// Additive Gaussian noise using Box-Muller transform.
#[derive(Debug, Clone)]
pub struct GaussianNoise {
    /// Range of noise standard deviation to sample from each call, e.g. `[0.0, 0.05]`.
    pub std_range: [f32; 2],
}

impl GaussianNoise {
    /// Construct with validated std range.
    pub fn new(std_range: [f32; 2]) -> Result<Self, AugmentationError> {
        if std_range[0] < 0.0 {
            return Err(AugmentationError::InvalidParam(
                "std_range[0] must be >= 0".into(),
            ));
        }
        if std_range[0] > std_range[1] {
            return Err(AugmentationError::InvalidParam(
                "std_range[0] > std_range[1]".into(),
            ));
        }
        Ok(Self { std_range })
    }

    /// Conservative defaults: `std_range = [0.0, 0.02]`.
    pub fn conservative() -> Self {
        Self {
            std_range: [0.0, 0.02],
        }
    }

    /// Apply Gaussian noise to each pixel channel independently.
    ///
    /// Uses Box-Muller: `n = sqrt(-2 * ln(u1)) * cos(2π * u2)` where `u1, u2 ~ U(0,1)`.
    pub fn apply(&self, img: &AugImage, rng: &mut u64) -> AugImage {
        let std = xorshift_range(rng, self.std_range[0], self.std_range[1]);
        let mut out = img.clone();

        if std < 1e-7 {
            return out;
        }

        let n_channels = img.data.len();
        let mut i = 0;
        while i < n_channels {
            // Box-Muller: generate two independent Gaussian samples
            let u1 = xorshift_f32(rng).max(1e-10_f32); // avoid log(0)
            let u2 = xorshift_f32(rng);

            let mag = (-2.0 * u1.ln()).sqrt() * std;
            let angle = 2.0 * core::f32::consts::PI * u2;

            let n1 = mag * angle.cos();
            out.data[i] = (out.data[i] + n1).clamp(0.0, 1.0);
            i += 1;

            if i < n_channels {
                let n2 = mag * angle.sin();
                out.data[i] = (out.data[i] + n2).clamp(0.0, 1.0);
                i += 1;
            }
        }

        out
    }
}

impl Default for GaussianNoise {
    fn default() -> Self {
        Self::conservative()
    }
}

// ---- GammaDistortion ---------------------------------------------------------

/// Photometric distortion via random gamma correction.
#[derive(Debug, Clone)]
pub struct GammaDistortion {
    /// Range of gamma values to sample from, e.g. `[0.8, 1.2]`.
    pub gamma_range: [f32; 2],
}

impl GammaDistortion {
    /// Construct with validated gamma range.
    pub fn new(gamma_range: [f32; 2]) -> Result<Self, AugmentationError> {
        if gamma_range[0] <= 0.0 {
            return Err(AugmentationError::InvalidParam(
                "gamma_range[0] must be > 0".into(),
            ));
        }
        if gamma_range[0] > gamma_range[1] {
            return Err(AugmentationError::InvalidParam(
                "gamma_range[0] > gamma_range[1]".into(),
            ));
        }
        Ok(Self { gamma_range })
    }

    /// Conservative defaults: `gamma_range = [0.85, 1.15]`.
    pub fn conservative() -> Self {
        Self {
            gamma_range: [0.85, 1.15],
        }
    }

    /// Apply `pixel = clamp(pixel^gamma, 0, 1)` using a randomly sampled gamma.
    pub fn apply(&self, img: &AugImage, rng: &mut u64) -> AugImage {
        let gamma = xorshift_range(rng, self.gamma_range[0], self.gamma_range[1]);
        let mut out = img.clone();
        for v in out.data.iter_mut() {
            *v = v.powf(gamma).clamp(0.0, 1.0);
        }
        out
    }
}

impl Default for GammaDistortion {
    fn default() -> Self {
        Self::conservative()
    }
}

// ---- AugStep / AugmentationPipeline -----------------------------------------

/// Type-erased augmentation step for use in a pipeline.
pub enum AugStep {
    ColorJitter(ColorJitter),
    Flip(RandomFlip),
    CropResize(RandomCropResize),
    Noise(GaussianNoise),
    Gamma(GammaDistortion),
}

impl AugStep {
    /// Apply this step to `img` using `rng`.
    pub fn apply(&self, img: &AugImage, rng: &mut u64) -> Result<AugImage, AugmentationError> {
        match self {
            AugStep::ColorJitter(t) => t.apply(img, rng),
            AugStep::Flip(t) => Ok(t.apply(img, rng)),
            AugStep::CropResize(t) => t.apply(img, rng),
            AugStep::Noise(t) => Ok(t.apply(img, rng)),
            AugStep::Gamma(t) => Ok(t.apply(img, rng)),
        }
    }
}

/// Composable augmentation pipeline.
pub struct AugmentationPipeline {
    steps: Vec<AugStep>,
    /// Per-step probability of being applied.
    step_probs: Vec<f32>,
    base_seed: u64,
    call_count: u64,
}

impl AugmentationPipeline {
    /// Create an empty pipeline with a base seed.
    pub fn new(base_seed: u64) -> Self {
        Self {
            steps: Vec::new(),
            step_probs: Vec::new(),
            base_seed,
            call_count: 0,
        }
    }

    /// Add a step with the given probability of being applied on each call.
    pub fn add_step(&mut self, step: AugStep, prob: f32) -> Result<(), AugmentationError> {
        if !(0.0..=1.0).contains(&prob) {
            return Err(AugmentationError::InvalidParam(format!(
                "step probability {prob} not in [0, 1]"
            )));
        }
        self.steps.push(step);
        self.step_probs.push(prob);
        Ok(())
    }

    /// Apply the pipeline to `img`, advancing the internal call counter.
    ///
    /// Each step is applied only when its probability check passes.
    /// The RNG seed for step `i` is derived deterministically as:
    /// `base_seed.wrapping_add(call_count * 997 + step_idx * 31)`
    pub fn apply(&mut self, img: &AugImage) -> Result<AugImage, AugmentationError> {
        let mut current = img.clone();
        let count = self.call_count;

        for (step_idx, (step, &prob)) in self.steps.iter().zip(self.step_probs.iter()).enumerate() {
            let mut rng = self
                .base_seed
                .wrapping_add(count.wrapping_mul(997).wrapping_add(step_idx as u64 * 31));

            // Probability gate
            if xorshift_f32(&mut rng) < prob {
                // Re-derive rng after the gate check so factor sampling is independent
                let mut rng2 = self.base_seed.wrapping_add(
                    count
                        .wrapping_mul(997)
                        .wrapping_add(step_idx as u64 * 31 + 1),
                );
                current = step.apply(&current, &mut rng2)?;
            }
        }

        self.call_count = count.wrapping_add(1);
        Ok(current)
    }

    /// Standard GAF augmentation pipeline with sensible defaults.
    ///
    /// - `ColorJitter::default()` with prob 0.8
    /// - `RandomFlip::horizontal_only()` with prob 0.5
    /// - `GammaDistortion::default()` with prob 0.5
    /// - `GaussianNoise::default()` with prob 0.3
    pub fn standard_gaf(seed: u64) -> Self {
        let mut pipeline = Self::new(seed);

        // These can't fail with valid defaults
        let _ = pipeline.add_step(AugStep::ColorJitter(ColorJitter::default()), 0.8);
        let _ = pipeline.add_step(AugStep::Flip(RandomFlip::horizontal_only()), 0.5);
        let _ = pipeline.add_step(AugStep::Gamma(GammaDistortion::default()), 0.5);
        let _ = pipeline.add_step(AugStep::Noise(GaussianNoise::default()), 0.3);

        pipeline
    }

    /// Total number of `apply()` calls made on this pipeline.
    pub fn call_count(&self) -> u64 {
        self.call_count
    }
}

// ---- Tests -------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::unwrap_used)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    /// Create a gradient test image: pixel (x,y) = [x/W, y/H, 0.5]
    fn test_image(width: usize, height: usize) -> AugImage {
        let mut data = Vec::with_capacity(width * height * 3);
        for y in 0..height {
            for x in 0..width {
                data.push(x as f32 / width as f32);
                data.push(y as f32 / height as f32);
                data.push(0.5);
            }
        }
        AugImage::new(data, width, height).unwrap()
    }

    // 1. AugImage::new correct dimensions
    #[test]
    fn test_aug_image_new_correct() {
        let img = test_image(8, 6);
        assert_eq!(img.width, 8);
        assert_eq!(img.height, 6);
        assert_eq!(img.data.len(), 8 * 6 * 3);
    }

    // 2. AugImage::new length mismatch → DataLengthMismatch
    #[test]
    fn test_aug_image_new_length_mismatch() {
        let data = vec![0.0_f32; 10]; // wrong length
        let result = AugImage::new(data, 4, 4);
        assert!(matches!(
            result,
            Err(AugmentationError::DataLengthMismatch { .. })
        ));
    }

    // 3. AugImage::pixel returns correct RGB
    #[test]
    fn test_aug_image_pixel() {
        let img = test_image(8, 6);
        let [r, g, b] = img.pixel(4, 3);
        assert!((r - 4.0 / 8.0).abs() < 1e-5);
        assert!((g - 3.0 / 6.0).abs() < 1e-5);
        assert!((b - 0.5).abs() < 1e-5);
    }

    // 4. AugImage::set_pixel updates correctly
    #[test]
    fn test_aug_image_set_pixel() {
        let mut img = AugImage::zeros(4, 4);
        img.set_pixel(2, 1, [0.1, 0.5, 0.9]);
        let [r, g, b] = img.pixel(2, 1);
        assert!((r - 0.1).abs() < 1e-6);
        assert!((g - 0.5).abs() < 1e-6);
        assert!((b - 0.9).abs() < 1e-6);
    }

    // 5. sample_bilinear at integer coords == pixel()
    #[test]
    fn test_sample_bilinear_integer_coords() {
        let img = test_image(8, 6);
        for y in 0..6usize {
            for x in 0..8usize {
                let expected = img.pixel(x, y);
                let sampled = img.sample_bilinear(x as f32, y as f32);
                for c in 0..3 {
                    assert!(
                        (expected[c] - sampled[c]).abs() < 1e-5,
                        "Mismatch at ({x},{y}) channel {c}: expected {}, got {}",
                        expected[c],
                        sampled[c]
                    );
                }
            }
        }
    }

    // 6. ColorJitter::apply output has same dimensions
    #[test]
    fn test_color_jitter_dimensions() {
        let img = test_image(16, 12);
        let jitter = ColorJitter::default();
        let mut rng = 42u64;
        let out = jitter.apply(&img, &mut rng).unwrap();
        assert_eq!(out.width, img.width);
        assert_eq!(out.height, img.height);
        assert_eq!(out.data.len(), img.data.len());
    }

    // 7. ColorJitter::apply pixel values in [0, 1]
    #[test]
    fn test_color_jitter_range() {
        let img = test_image(16, 12);
        let jitter = ColorJitter::default();
        let mut rng = 12345u64;
        let out = jitter.apply(&img, &mut rng).unwrap();
        for &v in &out.data {
            assert!((0.0..=1.0).contains(&v), "Pixel value {v} out of [0,1]");
        }
    }

    // 8. ColorJitter::apply with zero range → nearly identical image
    #[test]
    fn test_color_jitter_zero_range() {
        let img = test_image(8, 8);
        let jitter = ColorJitter::new([1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [0.0, 0.0]).unwrap();
        let mut rng = 99u64;
        let out = jitter.apply(&img, &mut rng).unwrap();
        for (a, b) in img.data.iter().zip(out.data.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "Expected near-identity but got delta {}",
                (a - b).abs()
            );
        }
    }

    // 9. RandomFlip horizontal flip: pixel(x,y) == flipped.pixel(W-1-x, y)
    #[test]
    fn test_random_flip_horizontal_correctness() {
        let img = test_image(8, 6);
        // Force horizontal flip by using p_h=1.0
        let flip = RandomFlip::new(1.0, 0.0).unwrap();
        let mut rng = 1u64;
        let flipped = flip.apply(&img, &mut rng);
        for y in 0..6usize {
            for x in 0..8usize {
                let orig = img.pixel(x, y);
                let from_flipped = flipped.pixel(img.width - 1 - x, y);
                for c in 0..3 {
                    assert!((orig[c] - from_flipped[c]).abs() < 1e-6);
                }
            }
        }
    }

    // 10. RandomFlip p_h=1.0 always flips horizontal
    #[test]
    fn test_random_flip_always_horizontal() {
        let img = test_image(8, 6);
        let flip = RandomFlip::new(1.0, 0.0).unwrap();
        for seed in [1u64, 2, 100, 9999, 0xdeadbeef] {
            let mut rng = seed;
            let out = flip.apply(&img, &mut rng);
            // Check first and last pixel in row 0 are swapped
            let orig_first = img.pixel(0, 0);
            let out_last = out.pixel(img.width - 1, 0);
            for c in 0..3 {
                assert!((orig_first[c] - out_last[c]).abs() < 1e-6);
            }
        }
    }

    // 11. RandomFlip p_h=0.0 never flips
    #[test]
    fn test_random_flip_never_horizontal() {
        let img = test_image(8, 6);
        let flip = RandomFlip::new(0.0, 0.0).unwrap();
        for seed in [1u64, 2, 100, 9999] {
            let mut rng = seed;
            let out = flip.apply(&img, &mut rng);
            for (a, b) in img.data.iter().zip(out.data.iter()) {
                assert!((a - b).abs() < 1e-6);
            }
        }
    }

    // 12. RandomCropResize::apply output has same dimensions as input
    #[test]
    fn test_crop_resize_dimensions() {
        let img = test_image(32, 24);
        let crop = RandomCropResize::default();
        let mut rng = 42u64;
        let out = crop.apply(&img, &mut rng).unwrap();
        assert_eq!(out.width, img.width);
        assert_eq!(out.height, img.height);
    }

    // 13. RandomCropResize::apply output values in [0, 1]
    #[test]
    fn test_crop_resize_range() {
        let img = test_image(32, 24);
        let crop = RandomCropResize::default();
        let mut rng = 7777u64;
        let out = crop.apply(&img, &mut rng).unwrap();
        for &v in &out.data {
            assert!((0.0..=1.0).contains(&v), "Value {v} out of [0,1]");
        }
    }

    // 14. GaussianNoise::apply output has same dimensions
    #[test]
    fn test_gaussian_noise_dimensions() {
        let img = test_image(16, 16);
        let noise = GaussianNoise::default();
        let mut rng = 42u64;
        let out = noise.apply(&img, &mut rng);
        assert_eq!(out.width, img.width);
        assert_eq!(out.height, img.height);
        assert_eq!(out.data.len(), img.data.len());
    }

    // 15. GaussianNoise::apply pixel values in [0, 1]
    #[test]
    fn test_gaussian_noise_range() {
        let img = test_image(16, 16);
        let noise = GaussianNoise::default();
        let mut rng = 555u64;
        let out = noise.apply(&img, &mut rng);
        for &v in &out.data {
            assert!((0.0..=1.0).contains(&v), "Value {v} out of [0,1]");
        }
    }

    // 16. GammaDistortion::apply output same dimensions
    #[test]
    fn test_gamma_distortion_dimensions() {
        let img = test_image(12, 10);
        let gamma = GammaDistortion::default();
        let mut rng = 42u64;
        let out = gamma.apply(&img, &mut rng);
        assert_eq!(out.width, img.width);
        assert_eq!(out.height, img.height);
    }

    // 17. GammaDistortion::apply with gamma=1.0 → nearly identical image
    #[test]
    fn test_gamma_distortion_identity() {
        let img = test_image(8, 8);
        let gamma = GammaDistortion::new([1.0, 1.0]).unwrap();
        let mut rng = 1u64;
        let out = gamma.apply(&img, &mut rng);
        for (a, b) in img.data.iter().zip(out.data.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "Expected near-identity but delta = {}",
                (a - b).abs()
            );
        }
    }

    // 18. rgb_to_hsv then hsv_to_rgb roundtrip within 1e-4
    #[test]
    fn test_rgb_hsv_roundtrip() {
        let test_colors = [
            [0.0_f32, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.3, 0.8],
            [0.9, 0.1, 0.4],
            [0.2, 0.7, 0.5],
        ];
        for [r, g, b] in test_colors {
            let [h, s, v] = rgb_to_hsv(r, g, b);
            let [r2, g2, b2] = hsv_to_rgb(h, s, v);
            assert!((r - r2).abs() < 1e-4, "R roundtrip failed: {r} → {r2}");
            assert!((g - g2).abs() < 1e-4, "G roundtrip failed: {g} → {g2}");
            assert!((b - b2).abs() < 1e-4, "B roundtrip failed: {b} → {b2}");
        }
    }

    // 19. rgb_to_hsv pure red [1,0,0] → h≈0, s≈1, v≈1
    #[test]
    fn test_rgb_to_hsv_pure_red() {
        let [h, s, v] = rgb_to_hsv(1.0, 0.0, 0.0);
        assert!(h.abs() < 1e-5 || (h - 1.0).abs() < 1e-5, "h={h}");
        assert!((s - 1.0).abs() < 1e-5, "s={s}");
        assert!((v - 1.0).abs() < 1e-5, "v={v}");
    }

    // 20. AugmentationPipeline::standard_gaf creates pipeline
    #[test]
    fn test_standard_gaf_pipeline_creation() {
        let pipeline = AugmentationPipeline::standard_gaf(42);
        assert_eq!(pipeline.steps.len(), 4);
        assert_eq!(pipeline.call_count(), 0);
    }

    // 21. AugmentationPipeline::apply output same dimensions
    #[test]
    fn test_pipeline_apply_dimensions() {
        let img = test_image(32, 24);
        let mut pipeline = AugmentationPipeline::standard_gaf(1234);
        let out = pipeline.apply(&img).unwrap();
        assert_eq!(out.width, img.width);
        assert_eq!(out.height, img.height);
        assert_eq!(out.data.len(), img.data.len());
    }

    // 22. AugmentationPipeline::call_count increments
    #[test]
    fn test_pipeline_call_count() {
        let img = test_image(8, 8);
        let mut pipeline = AugmentationPipeline::standard_gaf(777);
        assert_eq!(pipeline.call_count(), 0);
        pipeline.apply(&img).unwrap();
        assert_eq!(pipeline.call_count(), 1);
        pipeline.apply(&img).unwrap();
        assert_eq!(pipeline.call_count(), 2);
        pipeline.apply(&img).unwrap();
        assert_eq!(pipeline.call_count(), 3);
    }
}
