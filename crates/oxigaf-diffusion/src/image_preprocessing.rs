//! Image preprocessing pipeline for diffusion model inputs.
//!
//! Provides a pure-Rust image preprocessing pipeline that operates on flat
//! `Vec<f32>` data in **NHWC** format (height × width × channels for single
//! images), with pixel values in `[0, 1]` unless otherwise noted.
//!
//! # Usage
//!
//! ```rust
//! use oxigaf_diffusion::image_preprocessing::{
//!     ImageDims, PreprocessingPipeline, ResizeFilter, NormalizationMode,
//!     PreprocessingStep,
//! };
//!
//! // Create a synthetic 4×4 RGB image (all 0.5 grey)
//! let dims = ImageDims::new(4, 4, 3);
//! let img = vec![0.5f32; dims.num_elements()];
//!
//! // Build a standard SD preprocessing pipeline
//! let pipeline = PreprocessingPipeline::standard_sd();
//! let (out, out_dims) = pipeline.apply(&img, dims).unwrap();
//! assert_eq!(out_dims.height, 512);
//! assert_eq!(out_dims.width, 512);
//! ```

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during image preprocessing operations.
#[derive(Debug)]
pub enum PreprocessError {
    /// Image dimensions are invalid (e.g. zero, overflow, or size mismatch).
    InvalidDimensions { reason: String },

    /// Requested crop region extends beyond the source image boundaries.
    CropOutOfBounds {
        offset_y: usize,
        offset_x: usize,
        crop_h: usize,
        crop_w: usize,
        src_h: usize,
        src_w: usize,
    },

    /// Normalization mode requires a specific channel count that was not met.
    UnsupportedChannelCount { expected: usize, got: usize },

    /// Input image contains no data.
    EmptyImage,

    /// Gaussian blur sigma must be positive and finite (σ ≥ 1e-8).
    InvalidSigma(f32),
}

impl std::fmt::Display for PreprocessError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PreprocessError::InvalidDimensions { reason } => {
                write!(f, "Invalid image dimensions: {}", reason)
            }
            PreprocessError::CropOutOfBounds {
                offset_y,
                offset_x,
                crop_h,
                crop_w,
                src_h,
                src_w,
            } => write!(
                f,
                "Crop out of bounds: offset=({},{}) crop={}×{} src={}×{}",
                offset_y, offset_x, crop_h, crop_w, src_h, src_w
            ),
            PreprocessError::UnsupportedChannelCount { expected, got } => {
                write!(
                    f,
                    "Unsupported channel count: expected {}, got {}",
                    expected, got
                )
            }
            PreprocessError::EmptyImage => write!(f, "Empty image"),
            PreprocessError::InvalidSigma(s) => {
                write!(f, "Invalid sigma value for Gaussian blur: {}", s)
            }
        }
    }
}

impl std::error::Error for PreprocessError {}

// ---------------------------------------------------------------------------
// ImageDims
// ---------------------------------------------------------------------------

/// Dimensions of a single image stored in HWC order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageDims {
    /// Number of rows (pixels high).
    pub height: usize,
    /// Number of columns (pixels wide).
    pub width: usize,
    /// Number of colour channels (e.g. 1 for greyscale, 3 for RGB, 4 for RGBA).
    pub channels: usize,
}

impl ImageDims {
    /// Create new `ImageDims`.
    #[inline]
    pub fn new(height: usize, width: usize, channels: usize) -> Self {
        Self {
            height,
            width,
            channels,
        }
    }

    /// Total number of `f32` elements: `height * width * channels`.
    #[inline]
    pub fn num_elements(&self) -> usize {
        self.height * self.width * self.channels
    }

    /// Number of pixels: `height * width`.
    #[inline]
    pub fn pixel_count(&self) -> usize {
        self.height * self.width
    }

    /// Aspect ratio `width / height`.
    #[inline]
    pub fn aspect_ratio(&self) -> f32 {
        self.width as f32 / self.height as f32
    }
}

// ---------------------------------------------------------------------------
// ResizeFilter
// ---------------------------------------------------------------------------

/// Interpolation filter used during image resize.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ResizeFilter {
    /// Nearest-neighbour — fast and pixel-perfect, but blocky.
    Nearest,
    /// Bilinear interpolation — smooth, good general-purpose filter.
    Bilinear,
}

// ---------------------------------------------------------------------------
// resize_image
// ---------------------------------------------------------------------------

/// Resize `src` from `src_dims` to `(target_h, target_w)` using `filter`.
///
/// The number of channels is preserved. Returns the resized image and its
/// new [`ImageDims`].
///
/// # Errors
/// - [`PreprocessError::EmptyImage`] when `src` is empty.
/// - [`PreprocessError::InvalidDimensions`] when target dimensions are zero or
///   the source buffer length does not match `src_dims`.
pub fn resize_image(
    src: &[f32],
    src_dims: ImageDims,
    target_h: usize,
    target_w: usize,
    filter: ResizeFilter,
) -> Result<(Vec<f32>, ImageDims), PreprocessError> {
    if src.is_empty() {
        return Err(PreprocessError::EmptyImage);
    }
    if src.len() != src_dims.num_elements() {
        return Err(PreprocessError::InvalidDimensions {
            reason: format!(
                "src length {} does not match dims {}×{}×{}={}",
                src.len(),
                src_dims.height,
                src_dims.width,
                src_dims.channels,
                src_dims.num_elements()
            ),
        });
    }
    if target_h == 0 || target_w == 0 {
        return Err(PreprocessError::InvalidDimensions {
            reason: format!(
                "target dimensions {}×{} must both be nonzero",
                target_h, target_w
            ),
        });
    }

    let src_h = src_dims.height;
    let src_w = src_dims.width;
    let c = src_dims.channels;
    let out_dims = ImageDims::new(target_h, target_w, c);
    let mut out = vec![0.0f32; out_dims.num_elements()];

    match filter {
        ResizeFilter::Nearest => {
            for ty in 0..target_h {
                // Map target pixel to nearest source pixel
                let sy = if target_h == 1 {
                    0
                } else {
                    ((ty as f64 * (src_h - 1) as f64 / (target_h - 1) as f64) + 0.5) as usize
                };
                let sy = sy.min(src_h - 1);
                for tx in 0..target_w {
                    let sx = if target_w == 1 {
                        0
                    } else {
                        ((tx as f64 * (src_w - 1) as f64 / (target_w - 1) as f64) + 0.5) as usize
                    };
                    let sx = sx.min(src_w - 1);
                    let src_base = (sy * src_w + sx) * c;
                    let dst_base = (ty * target_w + tx) * c;
                    out[dst_base..dst_base + c].copy_from_slice(&src[src_base..src_base + c]);
                }
            }
        }
        ResizeFilter::Bilinear => {
            for ty in 0..target_h {
                let src_y = if target_h == 1 {
                    0.0f64
                } else {
                    ty as f64 * (src_h - 1) as f64 / (target_h - 1) as f64
                };
                let y0 = src_y.floor() as usize;
                let y1 = (y0 + 1).min(src_h - 1);
                let fy = (src_y - src_y.floor()) as f32;

                for tx in 0..target_w {
                    let src_x = if target_w == 1 {
                        0.0f64
                    } else {
                        tx as f64 * (src_w - 1) as f64 / (target_w - 1) as f64
                    };
                    let x0 = src_x.floor() as usize;
                    let x1 = (x0 + 1).min(src_w - 1);
                    let fx = (src_x - src_x.floor()) as f32;

                    let dst_base = (ty * target_w + tx) * c;
                    let base00 = (y0 * src_w + x0) * c;
                    let base01 = (y0 * src_w + x1) * c;
                    let base10 = (y1 * src_w + x0) * c;
                    let base11 = (y1 * src_w + x1) * c;

                    for ch in 0..c {
                        let v00 = src[base00 + ch];
                        let v01 = src[base01 + ch];
                        let v10 = src[base10 + ch];
                        let v11 = src[base11 + ch];
                        // Bilinear interpolation
                        let top = v00 + fx * (v01 - v00);
                        let bot = v10 + fx * (v11 - v10);
                        out[dst_base + ch] = top + fy * (bot - top);
                    }
                }
            }
        }
    }

    Ok((out, out_dims))
}

// ---------------------------------------------------------------------------
// CropMode
// ---------------------------------------------------------------------------

/// How to position the crop window within the source image.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CropMode {
    /// Centre the crop window in the source image.
    Center,
    /// Align the crop window to the top-left corner `(0, 0)`.
    TopLeft,
    /// Specify an explicit top-left corner for the crop window.
    Custom { offset_y: usize, offset_x: usize },
}

// ---------------------------------------------------------------------------
// crop_image
// ---------------------------------------------------------------------------

/// Crop `src` to a `(target_h, target_w)` region according to `mode`.
///
/// # Errors
/// - [`PreprocessError::EmptyImage`] when `src` is empty.
/// - [`PreprocessError::InvalidDimensions`] when dimensions mismatch or target
///   is zero.
/// - [`PreprocessError::CropOutOfBounds`] when the crop region exceeds source
///   image bounds.
pub fn crop_image(
    src: &[f32],
    src_dims: ImageDims,
    target_h: usize,
    target_w: usize,
    mode: CropMode,
) -> Result<(Vec<f32>, ImageDims), PreprocessError> {
    if src.is_empty() {
        return Err(PreprocessError::EmptyImage);
    }
    if src.len() != src_dims.num_elements() {
        return Err(PreprocessError::InvalidDimensions {
            reason: format!(
                "src length {} does not match dims {}",
                src.len(),
                src_dims.num_elements()
            ),
        });
    }
    if target_h == 0 || target_w == 0 {
        return Err(PreprocessError::InvalidDimensions {
            reason: format!(
                "target dimensions {}×{} must both be nonzero",
                target_h, target_w
            ),
        });
    }

    let src_h = src_dims.height;
    let src_w = src_dims.width;
    let c = src_dims.channels;

    let (offset_y, offset_x) = match mode {
        CropMode::Center => {
            let oy = src_h.saturating_sub(target_h) / 2;
            let ox = src_w.saturating_sub(target_w) / 2;
            (oy, ox)
        }
        CropMode::TopLeft => (0, 0),
        CropMode::Custom { offset_y, offset_x } => (offset_y, offset_x),
    };

    // Validate that the crop region fits within the source image
    if offset_y + target_h > src_h || offset_x + target_w > src_w {
        return Err(PreprocessError::CropOutOfBounds {
            offset_y,
            offset_x,
            crop_h: target_h,
            crop_w: target_w,
            src_h,
            src_w,
        });
    }

    let out_dims = ImageDims::new(target_h, target_w, c);
    let mut out = vec![0.0f32; out_dims.num_elements()];

    for y in 0..target_h {
        let src_row = (offset_y + y) * src_w;
        let dst_row = y * target_w;
        for x in 0..target_w {
            let src_base = (src_row + offset_x + x) * c;
            let dst_base = (dst_row + x) * c;
            out[dst_base..dst_base + c].copy_from_slice(&src[src_base..src_base + c]);
        }
    }

    Ok((out, out_dims))
}

// ---------------------------------------------------------------------------
// pad_to_square
// ---------------------------------------------------------------------------

/// Pad `src` to a square image of size `max(h, w) × max(h, w)`.
///
/// Padding is distributed equally on both sides of the shorter dimension
/// (half-pixel rounding goes to the bottom/right). Padded regions are filled
/// with `pad_value`.
pub fn pad_to_square(src: &[f32], src_dims: ImageDims, pad_value: f32) -> (Vec<f32>, ImageDims) {
    let side = src_dims.height.max(src_dims.width);
    let c = src_dims.channels;
    let out_dims = ImageDims::new(side, side, c);
    let mut out = vec![pad_value; out_dims.num_elements()];

    let y_off = (side - src_dims.height) / 2;
    let x_off = (side - src_dims.width) / 2;

    for y in 0..src_dims.height {
        let src_row = y * src_dims.width;
        let dst_row = (y + y_off) * side;
        for x in 0..src_dims.width {
            let src_base = (src_row + x) * c;
            let dst_base = (dst_row + x + x_off) * c;
            out[dst_base..dst_base + c].copy_from_slice(&src[src_base..src_base + c]);
        }
    }

    (out, out_dims)
}

// ---------------------------------------------------------------------------
// NormalizationMode
// ---------------------------------------------------------------------------

/// Strategy to normalise pixel values.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NormalizationMode {
    /// Map `[0, 1]` → `[-1, 1]`: `x * 2 − 1`.
    ClipRange,

    /// Subtract per-channel mean, divide by per-channel std.
    ///
    /// Applied to 3-channel images only.
    ZeroMean {
        /// Per-channel means (R, G, B).
        mean: [f32; 3],
        /// Per-channel standard deviations (R, G, B).
        std: [f32; 3],
    },

    /// Data-driven standardisation: `(x − μ) / (σ + 1e-8)` where μ and σ are
    /// computed from the image itself across all pixels and channels.
    Standardize,

    /// ImageNet normalisation: mean = [0.485, 0.456, 0.406],
    /// std = [0.229, 0.224, 0.225]. Requires 3-channel input.
    ImageNet,

    /// Identity — values are returned unchanged.
    Identity,
}

// ---------------------------------------------------------------------------
// normalize_image
// ---------------------------------------------------------------------------

/// ImageNet per-channel statistics (RGB).
const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Normalise `src` according to `mode`.
///
/// # Errors
/// - [`PreprocessError::EmptyImage`] when `src` is empty.
/// - [`PreprocessError::UnsupportedChannelCount`] when using [`NormalizationMode::ImageNet`]
///   or [`NormalizationMode::ZeroMean`] with a non-3-channel image.
pub fn normalize_image(
    src: &[f32],
    dims: ImageDims,
    mode: NormalizationMode,
) -> Result<Vec<f32>, PreprocessError> {
    if src.is_empty() {
        return Err(PreprocessError::EmptyImage);
    }

    let mut out = src.to_vec();
    let c = dims.channels;

    match mode {
        NormalizationMode::Identity => {}

        NormalizationMode::ClipRange => {
            for v in out.iter_mut() {
                *v = *v * 2.0 - 1.0;
            }
        }

        NormalizationMode::Standardize => {
            let n = out.len() as f64;
            let sum: f64 = out.iter().map(|&x| x as f64).sum();
            let mean = (sum / n) as f32;
            let var: f64 = out
                .iter()
                .map(|&x| {
                    let d = x as f64 - mean as f64;
                    d * d
                })
                .sum::<f64>()
                / n;
            let std = (var.sqrt() as f32) + 1e-8;
            for v in out.iter_mut() {
                *v = (*v - mean) / std;
            }
        }

        NormalizationMode::ZeroMean { mean, std } => {
            if c != 3 {
                return Err(PreprocessError::UnsupportedChannelCount {
                    expected: 3,
                    got: c,
                });
            }
            for pixel_start in (0..out.len()).step_by(c) {
                for ch in 0..c {
                    out[pixel_start + ch] = (out[pixel_start + ch] - mean[ch]) / (std[ch] + 1e-8);
                }
            }
        }

        NormalizationMode::ImageNet => {
            if c != 3 {
                return Err(PreprocessError::UnsupportedChannelCount {
                    expected: 3,
                    got: c,
                });
            }
            for pixel_start in (0..out.len()).step_by(c) {
                for ch in 0..c {
                    out[pixel_start + ch] =
                        (out[pixel_start + ch] - IMAGENET_MEAN[ch]) / IMAGENET_STD[ch];
                }
            }
        }
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// Flip operations
// ---------------------------------------------------------------------------

/// Flip `src` horizontally (left ↔ right).
pub fn flip_horizontal(src: &[f32], dims: ImageDims) -> Vec<f32> {
    let h = dims.height;
    let w = dims.width;
    let c = dims.channels;
    let mut out = vec![0.0f32; src.len()];
    for y in 0..h {
        for x in 0..w {
            let src_base = (y * w + x) * c;
            let dst_base = (y * w + (w - 1 - x)) * c;
            out[dst_base..dst_base + c].copy_from_slice(&src[src_base..src_base + c]);
        }
    }
    out
}

/// Flip `src` vertically (top ↔ bottom).
pub fn flip_vertical(src: &[f32], dims: ImageDims) -> Vec<f32> {
    let h = dims.height;
    let w = dims.width;
    let c = dims.channels;
    let mut out = vec![0.0f32; src.len()];
    for y in 0..h {
        let src_row = y * w * c;
        let dst_row = (h - 1 - y) * w * c;
        out[dst_row..dst_row + w * c].copy_from_slice(&src[src_row..src_row + w * c]);
    }
    out
}

// ---------------------------------------------------------------------------
// Gaussian blur
// ---------------------------------------------------------------------------

/// Build a normalised 1-D Gaussian kernel of the given `kernel_size` and `sigma`.
fn build_gaussian_kernel(kernel_size: usize, sigma: f32) -> Vec<f32> {
    let center = (kernel_size / 2) as f32;
    let mut kernel: Vec<f32> = (0..kernel_size)
        .map(|i| {
            let d = i as f32 - center;
            (-0.5 * (d / sigma) * (d / sigma)).exp()
        })
        .collect();
    // Normalise
    let sum: f32 = kernel.iter().sum();
    if sum > 1e-12 {
        for v in kernel.iter_mut() {
            *v /= sum;
        }
    }
    kernel
}

/// Apply separable Gaussian blur with the given `sigma`.
///
/// The kernel size is computed as `max(3, 2 * ceil(3 * sigma) as usize + 1)`,
/// rounded up to the nearest odd integer.
///
/// # Errors
/// - [`PreprocessError::EmptyImage`] when `src` is empty.
/// - [`PreprocessError::InvalidSigma`] when `sigma < 1e-8` or non-finite.
pub fn gaussian_blur(
    src: &[f32],
    dims: ImageDims,
    sigma: f32,
) -> Result<Vec<f32>, PreprocessError> {
    if src.is_empty() {
        return Err(PreprocessError::EmptyImage);
    }
    if !sigma.is_finite() || sigma < 1e-8 {
        return Err(PreprocessError::InvalidSigma(sigma));
    }

    let h = dims.height;
    let w = dims.width;
    let c = dims.channels;

    // Derive kernel size: max(3, 2 * ceil(3 * sigma) + 1), clamped to odd
    let half_size = (3.0 * sigma).ceil() as usize;
    let kernel_size = (2 * half_size + 1).max(3);
    // Ensure odd
    let kernel_size = if kernel_size.is_multiple_of(2) {
        kernel_size + 1
    } else {
        kernel_size
    };

    let kernel = build_gaussian_kernel(kernel_size, sigma);
    let half = kernel_size / 2;

    // Horizontal pass: src → tmp
    let mut tmp = vec![0.0f32; h * w * c];
    for y in 0..h {
        for x in 0..w {
            let dst_base = (y * w + x) * c;
            for ch in 0..c {
                let mut acc = 0.0f32;
                for (ki, &kv) in kernel.iter().enumerate() {
                    let sx = x as isize + ki as isize - half as isize;
                    // Reflect at boundaries
                    let sx = sx.max(0).min(w as isize - 1) as usize;
                    acc += src[(y * w + sx) * c + ch] * kv;
                }
                tmp[dst_base + ch] = acc;
            }
        }
    }

    // Vertical pass: tmp → out
    let mut out = vec![0.0f32; h * w * c];
    for y in 0..h {
        for x in 0..w {
            let dst_base = (y * w + x) * c;
            for ch in 0..c {
                let mut acc = 0.0f32;
                for (ki, &kv) in kernel.iter().enumerate() {
                    let sy = y as isize + ki as isize - half as isize;
                    let sy = sy.max(0).min(h as isize - 1) as usize;
                    acc += tmp[(sy * w + x) * c + ch] * kv;
                }
                out[dst_base + ch] = acc;
            }
        }
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// ImageStats
// ---------------------------------------------------------------------------

/// Summary statistics for an image buffer.
#[derive(Debug, Clone)]
pub struct ImageStats {
    /// Mean pixel value across all channels and pixels.
    pub mean: f32,
    /// Standard deviation across all channels and pixels.
    pub std: f32,
    /// Minimum pixel value.
    pub min: f32,
    /// Maximum pixel value.
    pub max: f32,
    /// Number of pixels (`height * width`).
    pub num_pixels: usize,
    /// Number of channels.
    pub channels: usize,
}

impl ImageStats {
    /// Compute statistics from an image buffer.
    pub fn compute(src: &[f32], dims: ImageDims) -> Self {
        let n = src.len();
        if n == 0 {
            return Self {
                mean: 0.0,
                std: 0.0,
                min: 0.0,
                max: 0.0,
                num_pixels: 0,
                channels: dims.channels,
            };
        }

        let mut min = src[0];
        let mut max = src[0];
        let mut sum = 0.0f64;

        for &v in src {
            if v < min {
                min = v;
            }
            if v > max {
                max = v;
            }
            sum += v as f64;
        }

        let mean = (sum / n as f64) as f32;
        let var: f64 = src
            .iter()
            .map(|&v| {
                let d = v as f64 - mean as f64;
                d * d
            })
            .sum::<f64>()
            / n as f64;
        let std = var.sqrt() as f32;

        Self {
            mean,
            std,
            min,
            max,
            num_pixels: dims.pixel_count(),
            channels: dims.channels,
        }
    }

    /// Format a human-readable one-line summary.
    pub fn format_summary(&self) -> String {
        format!(
            "pixels={} channels={} mean={:.4} std={:.4} min={:.4} max={:.4}",
            self.num_pixels, self.channels, self.mean, self.std, self.min, self.max
        )
    }
}

// ---------------------------------------------------------------------------
// PreprocessingStep
// ---------------------------------------------------------------------------

/// A single step in an image preprocessing pipeline.
#[derive(Debug, Clone)]
pub enum PreprocessingStep {
    /// Resize to the given target dimensions.
    Resize {
        height: usize,
        width: usize,
        filter: ResizeFilter,
    },
    /// Centre-crop to the given target dimensions.
    CropCenter { height: usize, width: usize },
    /// Pad the image to a square using the given fill value.
    PadToSquare { pad_value: f32 },
    /// Apply the given normalisation mode.
    Normalize(NormalizationMode),
    /// Flip the image left ↔ right.
    FlipHorizontal,
    /// Flip the image top ↔ bottom.
    FlipVertical,
    /// Apply a separable Gaussian blur with the given sigma.
    GaussianBlur { sigma: f32 },
}

// ---------------------------------------------------------------------------
// PreprocessingPipeline
// ---------------------------------------------------------------------------

/// A composable chain of [`PreprocessingStep`]s applied in order.
#[derive(Debug, Clone)]
pub struct PreprocessingPipeline {
    steps: Vec<PreprocessingStep>,
}

impl PreprocessingPipeline {
    /// Create an empty pipeline.
    pub fn new() -> Self {
        Self { steps: Vec::new() }
    }

    /// Append a step and return `self` for builder-style chaining.
    pub fn add_step(mut self, step: PreprocessingStep) -> Self {
        self.steps.push(step);
        self
    }

    /// Number of steps in this pipeline.
    pub fn num_steps(&self) -> usize {
        self.steps.len()
    }

    /// Apply all steps in order. Returns `(output, final_dims)`.
    ///
    /// # Errors
    /// Returns the first [`PreprocessError`] encountered during any step.
    pub fn apply(
        &self,
        src: &[f32],
        src_dims: ImageDims,
    ) -> Result<(Vec<f32>, ImageDims), PreprocessError> {
        let mut data: Vec<f32> = src.to_vec();
        let mut dims = src_dims;

        for step in &self.steps {
            match step {
                PreprocessingStep::Resize {
                    height,
                    width,
                    filter,
                } => {
                    let (next, next_dims) = resize_image(&data, dims, *height, *width, *filter)?;
                    data = next;
                    dims = next_dims;
                }
                PreprocessingStep::CropCenter { height, width } => {
                    let (next, next_dims) =
                        crop_image(&data, dims, *height, *width, CropMode::Center)?;
                    data = next;
                    dims = next_dims;
                }
                PreprocessingStep::PadToSquare { pad_value } => {
                    let (next, next_dims) = pad_to_square(&data, dims, *pad_value);
                    data = next;
                    dims = next_dims;
                }
                PreprocessingStep::Normalize(mode) => {
                    data = normalize_image(&data, dims, *mode)?;
                }
                PreprocessingStep::FlipHorizontal => {
                    data = flip_horizontal(&data, dims);
                }
                PreprocessingStep::FlipVertical => {
                    data = flip_vertical(&data, dims);
                }
                PreprocessingStep::GaussianBlur { sigma } => {
                    data = gaussian_blur(&data, dims, *sigma)?;
                }
            }
        }

        Ok((data, dims))
    }

    /// Standard Stable Diffusion pipeline:
    /// 1. Resize to 512×512 with bilinear interpolation.
    /// 2. Normalise with [`NormalizationMode::ClipRange`] (\[0,1\] → \[-1,1\]).
    pub fn standard_sd() -> Self {
        Self::new()
            .add_step(PreprocessingStep::Resize {
                height: 512,
                width: 512,
                filter: ResizeFilter::Bilinear,
            })
            .add_step(PreprocessingStep::Normalize(NormalizationMode::ClipRange))
    }

    /// Face-crop pipeline:
    /// 1. Pad the image to a square (filling black).
    /// 2. Resize to `target_size × target_size` with bilinear interpolation.
    /// 3. Normalise with [`NormalizationMode::ClipRange`].
    pub fn face_crop(target_size: usize) -> Self {
        Self::new()
            .add_step(PreprocessingStep::PadToSquare { pad_value: 0.0 })
            .add_step(PreprocessingStep::Resize {
                height: target_size,
                width: target_size,
                filter: ResizeFilter::Bilinear,
            })
            .add_step(PreprocessingStep::Normalize(NormalizationMode::ClipRange))
    }
}

impl Default for PreprocessingPipeline {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: create a solid-colour image
    fn solid_image(h: usize, w: usize, c: usize, value: f32) -> Vec<f32> {
        vec![value; h * w * c]
    }

    // Helper: create a checkerboard (alternating 0.0 / 1.0 per pixel)
    fn checkerboard(h: usize, w: usize, c: usize) -> Vec<f32> {
        let mut img = Vec::with_capacity(h * w * c);
        for y in 0..h {
            for x in 0..w {
                let v = if (x + y) % 2 == 0 { 0.0f32 } else { 1.0f32 };
                for _ in 0..c {
                    img.push(v);
                }
            }
        }
        img
    }

    // ---------- ImageDims ----------

    #[test]
    fn test_image_dims_num_elements() {
        let d = ImageDims::new(4, 8, 3);
        assert_eq!(d.num_elements(), 96);
        assert_eq!(d.pixel_count(), 32);
        assert!((d.aspect_ratio() - 2.0).abs() < 1e-6);
    }

    // ---------- Resize: nearest ----------

    #[test]
    fn test_resize_nearest_double_size() {
        // 2×2 greyscale: [[0, 1], [2, 3]] (values 0.0..=1.0 scaled)
        let src = vec![0.0f32, 1.0, 0.5, 0.25];
        let dims = ImageDims::new(2, 2, 1);
        let (out, out_dims) = resize_image(&src, dims, 4, 4, ResizeFilter::Nearest).unwrap();
        assert_eq!(out_dims, ImageDims::new(4, 4, 1));
        assert_eq!(out.len(), 16);
        // Top-left quadrant should map to src[0] = 0.0
        assert!((out[0] - 0.0).abs() < 1e-6);
    }

    // ---------- Resize: bilinear ----------

    #[test]
    fn test_resize_bilinear_half_size() {
        // 4×4 uniform grey → 2×2 should remain grey
        let src = solid_image(4, 4, 3, 0.5);
        let dims = ImageDims::new(4, 4, 3);
        let (out, out_dims) = resize_image(&src, dims, 2, 2, ResizeFilter::Bilinear).unwrap();
        assert_eq!(out_dims, ImageDims::new(2, 2, 3));
        for &v in &out {
            assert!((v - 0.5).abs() < 1e-5, "got {}", v);
        }
    }

    #[test]
    fn test_resize_to_same_size_identity() {
        let src = checkerboard(4, 6, 3);
        let dims = ImageDims::new(4, 6, 3);
        let (out, out_dims) = resize_image(&src, dims, 4, 6, ResizeFilter::Bilinear).unwrap();
        assert_eq!(out_dims, dims);
        for (a, b) in src.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1e-5, "mismatch: {} vs {}", a, b);
        }
    }

    // ---------- Crop ----------

    #[test]
    fn test_crop_center() {
        // 4×4 image where value = y (row)
        let mut src = vec![0.0f32; 4 * 4];
        for y in 0..4_usize {
            for x in 0..4_usize {
                src[y * 4 + x] = y as f32;
            }
        }
        let dims = ImageDims::new(4, 4, 1);
        // Center crop to 2×2 → starts at (1,1)
        let (out, out_dims) = crop_image(&src, dims, 2, 2, CropMode::Center).unwrap();
        assert_eq!(out_dims, ImageDims::new(2, 2, 1));
        // Rows 1 and 2, cols 1 and 2
        assert!((out[0] - 1.0).abs() < 1e-6); // y=1, x=1
        assert!((out[1] - 1.0).abs() < 1e-6); // y=1, x=2
        assert!((out[2] - 2.0).abs() < 1e-6); // y=2, x=1
    }

    #[test]
    fn test_crop_topleft() {
        let src = checkerboard(6, 6, 1);
        let dims = ImageDims::new(6, 6, 1);
        let (out, out_dims) = crop_image(&src, dims, 3, 3, CropMode::TopLeft).unwrap();
        assert_eq!(out_dims, ImageDims::new(3, 3, 1));
        // Top-left 3×3 of the checkerboard
        for y in 0..3_usize {
            for x in 0..3_usize {
                let expected = if (x + y) % 2 == 0 { 0.0f32 } else { 1.0 };
                assert!((out[y * 3 + x] - expected).abs() < 1e-6);
            }
        }
    }

    // ---------- Pad to square ----------

    #[test]
    fn test_pad_to_square_wider() {
        // 2×4 image (h=2, w=4): pad to 4×4
        let src = solid_image(2, 4, 1, 0.7);
        let dims = ImageDims::new(2, 4, 1);
        let (out, out_dims) = pad_to_square(&src, dims, 0.0);
        assert_eq!(out_dims, ImageDims::new(4, 4, 1));
        // Padding rows (top and bottom) should be 0
        for x in 0..4_usize {
            assert!((out[x] - 0.0).abs() < 1e-6); // row 0
            assert!((out[3 * 4 + x] - 0.0).abs() < 1e-6); // row 3
        }
        // Inner rows 1,2 should be 0.7
        for x in 0..4_usize {
            assert!((out[4 + x] - 0.7).abs() < 1e-5);
            assert!((out[2 * 4 + x] - 0.7).abs() < 1e-5); // keep 2*4 as it needs the 2
        }
    }

    #[test]
    fn test_pad_to_square_taller() {
        // 4×2 image (h=4, w=2): pad to 4×4
        let src = solid_image(4, 2, 1, 0.3);
        let dims = ImageDims::new(4, 2, 1);
        let (out, out_dims) = pad_to_square(&src, dims, 0.0);
        assert_eq!(out_dims, ImageDims::new(4, 4, 1));
        // Left and right padding columns should be 0; inner columns 1,2 → 0.3
        for y in 0..4_usize {
            assert!((out[y * 4] - 0.0).abs() < 1e-6); // col 0 padded
            assert!((out[y * 4 + 1] - 0.3).abs() < 1e-5); // col 1 data
            assert!((out[y * 4 + 2] - 0.3).abs() < 1e-5); // col 2 data
            assert!((out[y * 4 + 3] - 0.0).abs() < 1e-6); // col 3 padded
        }
    }

    // ---------- Normalize ----------

    #[test]
    fn test_normalize_clip_range() {
        let src = vec![0.0f32, 0.5, 1.0];
        let dims = ImageDims::new(1, 1, 3);
        let out = normalize_image(&src, dims, NormalizationMode::ClipRange).unwrap();
        assert!((out[0] - (-1.0)).abs() < 1e-6);
        assert!((out[1] - 0.0).abs() < 1e-6);
        assert!((out[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_standardize_zero_mean() {
        // All-same image → std ≈ 0, output ≈ 0 after standardise
        let src = vec![0.5f32; 3 * 3 * 3];
        let dims = ImageDims::new(3, 3, 3);
        let out = normalize_image(&src, dims, NormalizationMode::Standardize).unwrap();
        for &v in &out {
            // (0.5 - 0.5) / (0 + 1e-8) = 0
            assert!(v.abs() < 1e-4, "expected ~0.0, got {}", v);
        }
    }

    #[test]
    fn test_normalize_imagenet_error_wrong_channels() {
        let src = vec![0.5f32; 4]; // 1 pixel × 4 channels → error
        let dims = ImageDims::new(1, 1, 4);
        let result = normalize_image(&src, dims, NormalizationMode::ImageNet);
        assert!(
            matches!(
                result,
                Err(PreprocessError::UnsupportedChannelCount {
                    expected: 3,
                    got: 4
                })
            ),
            "expected UnsupportedChannelCount error"
        );
    }

    // ---------- Flip ----------

    #[test]
    fn test_flip_horizontal() {
        // 1×3 single-channel image: [0.1, 0.2, 0.3]
        let src = vec![0.1f32, 0.2, 0.3];
        let dims = ImageDims::new(1, 3, 1);
        let out = flip_horizontal(&src, dims);
        assert!((out[0] - 0.3).abs() < 1e-6);
        assert!((out[1] - 0.2).abs() < 1e-6);
        assert!((out[2] - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_flip_vertical() {
        // 3×1 single-channel image: [0.1, 0.2, 0.3] (3 rows, 1 column)
        let src = vec![0.1f32, 0.2, 0.3];
        let dims = ImageDims::new(3, 1, 1);
        let out = flip_vertical(&src, dims);
        assert!((out[0] - 0.3).abs() < 1e-6);
        assert!((out[1] - 0.2).abs() < 1e-6);
        assert!((out[2] - 0.1).abs() < 1e-6);
    }

    // ---------- Gaussian blur ----------

    #[test]
    fn test_gaussian_blur_preserves_dims() {
        let src = checkerboard(8, 8, 3);
        let dims = ImageDims::new(8, 8, 3);
        let out = gaussian_blur(&src, dims, 1.0).unwrap();
        assert_eq!(out.len(), dims.num_elements());
    }

    #[test]
    fn test_gaussian_blur_uniform_image() {
        // Blurring a uniform image should leave values unchanged
        let src = solid_image(8, 8, 3, 0.6);
        let dims = ImageDims::new(8, 8, 3);
        let out = gaussian_blur(&src, dims, 1.5).unwrap();
        for &v in &out {
            assert!((v - 0.6).abs() < 1e-4, "expected ~0.6, got {}", v);
        }
    }

    #[test]
    fn test_gaussian_blur_invalid_sigma() {
        let src = solid_image(4, 4, 1, 0.5);
        let dims = ImageDims::new(4, 4, 1);
        assert!(matches!(
            gaussian_blur(&src, dims, 0.0),
            Err(PreprocessError::InvalidSigma(_))
        ));
        assert!(matches!(
            gaussian_blur(&src, dims, -1.0),
            Err(PreprocessError::InvalidSigma(_))
        ));
    }

    // ---------- ImageStats ----------

    #[test]
    fn test_image_stats_compute() {
        // Simple 1×2 greyscale: [0.0, 1.0] → mean=0.5, std=0.5, min=0, max=1
        let src = vec![0.0f32, 1.0];
        let dims = ImageDims::new(1, 2, 1);
        let stats = ImageStats::compute(&src, dims);
        assert!((stats.mean - 0.5).abs() < 1e-5, "mean={}", stats.mean);
        assert!((stats.std - 0.5).abs() < 1e-5, "std={}", stats.std);
        assert!((stats.min - 0.0).abs() < 1e-6);
        assert!((stats.max - 1.0).abs() < 1e-6);
        assert_eq!(stats.num_pixels, 2);
        assert_eq!(stats.channels, 1);
        // format_summary should include the key fields
        let summary = stats.format_summary();
        assert!(summary.contains("mean="));
        assert!(summary.contains("std="));
    }

    // ---------- Pipeline ----------

    #[test]
    fn test_pipeline_standard_sd() {
        let src = solid_image(64, 64, 3, 0.5);
        let dims = ImageDims::new(64, 64, 3);
        let pipeline = PreprocessingPipeline::standard_sd();
        assert_eq!(pipeline.num_steps(), 2);
        let (out, out_dims) = pipeline.apply(&src, dims).unwrap();
        assert_eq!(out_dims, ImageDims::new(512, 512, 3));
        // ClipRange: 0.5 → 0.0
        for &v in &out {
            assert!((v - 0.0).abs() < 1e-4, "expected ~0.0, got {}", v);
        }
    }

    #[test]
    fn test_pipeline_multi_step() {
        // face_crop: pad→resize→normalize
        // Use a square image so padding doesn't alter pixels, then all 1.0 → ClipRange → 1.0
        let src = solid_image(8, 8, 3, 1.0);
        let dims = ImageDims::new(8, 8, 3);
        let pipeline = PreprocessingPipeline::face_crop(64);
        assert_eq!(pipeline.num_steps(), 3);
        let (out, out_dims) = pipeline.apply(&src, dims).unwrap();
        assert_eq!(out_dims, ImageDims::new(64, 64, 3));
        // Square input → pad_to_square is a no-op → all pixels remain 1.0
        // ClipRange: 1.0 * 2 - 1 = 1.0
        for &v in &out {
            assert!((v - 1.0).abs() < 1e-4, "expected ~1.0, got {}", v);
        }
    }

    #[test]
    fn test_pipeline_empty() {
        let src = vec![0.3f32, 0.6, 0.9];
        let dims = ImageDims::new(1, 1, 3);
        let pipeline = PreprocessingPipeline::new();
        assert_eq!(pipeline.num_steps(), 0);
        let (out, out_dims) = pipeline.apply(&src, dims).unwrap();
        assert_eq!(out_dims, dims);
        for (a, b) in src.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }
}
