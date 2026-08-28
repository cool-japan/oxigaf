//! Scene compositing: combine multiple rendered RGBA layers into a final output.
//!
//! Implements Porter-Duff compositing, alpha matting, and mask operations for
//! combining 3DGS rendered layers.

use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// Error
// ─────────────────────────────────────────────────────────────────────────────

/// Errors produced by scene compositing operations.
#[derive(Debug, Error)]
pub enum CompositorError {
    /// Data length does not match declared dimensions and channel count.
    #[error("Image data length {len} does not match {w}×{h}×{channels}")]
    DataLengthMismatch {
        /// Actual data length.
        len: usize,
        /// Declared width.
        w: usize,
        /// Declared height.
        h: usize,
        /// Expected channel count.
        channels: usize,
    },

    /// Two images have incompatible dimensions.
    #[error("Image dimensions mismatch: ({w1}×{h1}) vs ({w2}×{h2})")]
    DimensionMismatch {
        /// Width of the first image.
        w1: usize,
        /// Height of the first image.
        h1: usize,
        /// Width of the second image.
        w2: usize,
        /// Height of the second image.
        h2: usize,
    },

    /// No layers were provided to composite.
    #[error("No layers to composite")]
    NoLayers,

    /// Opacity value outside [0, 1].
    #[error("Invalid opacity {0}: must be in [0, 1]")]
    InvalidOpacity(f32),

    /// Mask dimensions do not match the layer image dimensions.
    #[error("Mask dimensions must match image dimensions")]
    MaskDimensionMismatch,
}

// ─────────────────────────────────────────────────────────────────────────────
// RgbaImage
// ─────────────────────────────────────────────────────────────────────────────

/// Flat RGBA image: `Vec<f32>` of length `W*H*4`, values nominally in [0, 1].
///
/// Pixels are stored row-major: index of pixel `(x, y)` is `(y*width + x)*4`.
#[derive(Debug, Clone)]
pub struct RgbaImage {
    /// RGBA interleaved pixel data (length == width * height * 4).
    pub data: Vec<f32>,
    /// Image width in pixels.
    pub width: usize,
    /// Image height in pixels.
    pub height: usize,
}

impl RgbaImage {
    /// Create a new `RgbaImage`, validating that `data.len() == width * height * 4`.
    pub fn new(data: Vec<f32>, width: usize, height: usize) -> Result<Self, CompositorError> {
        let expected = width * height * 4;
        if data.len() != expected {
            return Err(CompositorError::DataLengthMismatch {
                len: data.len(),
                w: width,
                h: height,
                channels: 4,
            });
        }
        Ok(Self {
            data,
            width,
            height,
        })
    }

    /// Create a fully transparent black image of the given dimensions.
    pub fn zeros(width: usize, height: usize) -> Self {
        Self {
            data: vec![0.0_f32; width * height * 4],
            width,
            height,
        }
    }

    /// Create an `RgbaImage` from flat RGB data with a uniform alpha value.
    ///
    /// `rgb` must have length `width * height * 3`.
    pub fn from_rgb(
        rgb: &[f32],
        width: usize,
        height: usize,
        alpha: f32,
    ) -> Result<Self, CompositorError> {
        let expected = width * height * 3;
        if rgb.len() != expected {
            return Err(CompositorError::DataLengthMismatch {
                len: rgb.len(),
                w: width,
                h: height,
                channels: 3,
            });
        }
        let mut data = Vec::with_capacity(width * height * 4);
        for chunk in rgb.chunks_exact(3) {
            data.push(chunk[0]);
            data.push(chunk[1]);
            data.push(chunk[2]);
            data.push(alpha);
        }
        Ok(Self {
            data,
            width,
            height,
        })
    }

    /// Return pixel at column `x`, row `y` as `[r, g, b, a]`.
    #[inline]
    pub fn pixel(&self, x: usize, y: usize) -> [f32; 4] {
        let base = (y * self.width + x) * 4;
        [
            self.data[base],
            self.data[base + 1],
            self.data[base + 2],
            self.data[base + 3],
        ]
    }

    /// Set pixel at column `x`, row `y`.
    #[inline]
    pub fn set_pixel(&mut self, x: usize, y: usize, rgba: [f32; 4]) {
        let base = (y * self.width + x) * 4;
        self.data[base] = rgba[0];
        self.data[base + 1] = rgba[1];
        self.data[base + 2] = rgba[2];
        self.data[base + 3] = rgba[3];
    }

    /// Extract RGB channels as a flat `Vec<f32>`, discarding alpha.
    pub fn to_rgb(&self) -> Vec<f32> {
        let n = self.width * self.height;
        let mut out = Vec::with_capacity(n * 3);
        for i in 0..n {
            let base = i * 4;
            out.push(self.data[base]);
            out.push(self.data[base + 1]);
            out.push(self.data[base + 2]);
        }
        out
    }

    /// Extract the alpha channel as a flat `Vec<f32>`.
    pub fn alpha_channel(&self) -> Vec<f32> {
        let n = self.width * self.height;
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            out.push(self.data[i * 4 + 3]);
        }
        out
    }

    /// Return a copy with `out.rgb = in.rgb * in.alpha` (premultiplied alpha).
    pub fn premultiply_alpha(&self) -> Self {
        let mut data = self.data.clone();
        let n = self.width * self.height;
        for i in 0..n {
            let base = i * 4;
            let a = data[base + 3];
            data[base] *= a;
            data[base + 1] *= a;
            data[base + 2] *= a;
        }
        Self {
            data,
            width: self.width,
            height: self.height,
        }
    }

    /// Return a copy with `out.rgb = in.rgb / max(in.alpha, 1e-6)` (un-premultiplied).
    pub fn unpremultiply_alpha(&self) -> Self {
        let mut data = self.data.clone();
        let n = self.width * self.height;
        for i in 0..n {
            let base = i * 4;
            let a = data[base + 3].max(1e-6);
            data[base] /= a;
            data[base + 1] /= a;
            data[base + 2] /= a;
        }
        Self {
            data,
            width: self.width,
            height: self.height,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Blend modes
// ─────────────────────────────────────────────────────────────────────────────

/// Porter-Duff compositing blend modes.
#[derive(Debug, Clone, PartialEq)]
pub enum BlendMode {
    /// Over: `out = src + dst*(1-src.a)` (standard alpha compositing).
    Over,
    /// Under: dst is on top — equivalent to Over with src/dst swapped.
    Under,
    /// Multiply: `out.rgb = src.rgb * dst.rgb`, alpha uses screen formula.
    Multiply,
    /// Screen: `out = 1 - (1-src)*(1-dst)`.
    Screen,
    /// Overlay: per-channel multiply-or-screen depending on dst luminance.
    Overlay,
    /// Add: `out.rgb = clamp(src.rgb + dst.rgb, 0, 1)`.
    Add,
    /// Difference: `out.rgb = |src.rgb - dst.rgb|`.
    Difference,
    /// Normal (identical to Over).
    Normal,
}

/// Blend two RGBA pixels using the specified mode. Returns `[r, g, b, a]`.
///
/// RGB values for Over and similar modes are returned in straight (not
/// premultiplied) form. The `alpha` of the output is the Porter-Duff combined
/// alpha unless the blend mode specifies otherwise.
pub fn blend_pixels(src: [f32; 4], dst: [f32; 4], mode: &BlendMode) -> [f32; 4] {
    match mode {
        BlendMode::Normal | BlendMode::Over => blend_over(src, dst),
        BlendMode::Under => blend_over(dst, src),
        BlendMode::Multiply => blend_multiply(src, dst),
        BlendMode::Screen => blend_screen(src, dst),
        BlendMode::Overlay => blend_overlay(src, dst),
        BlendMode::Add => blend_add(src, dst),
        BlendMode::Difference => blend_difference(src, dst),
    }
}

// Over: standard Porter-Duff "A over B"
// out.a  = src.a + dst.a*(1-src.a)
// out.rgb = (src.rgb*src.a + dst.rgb*dst.a*(1-src.a)) / max(out.a, 1e-6)
fn blend_over(src: [f32; 4], dst: [f32; 4]) -> [f32; 4] {
    let sa = src[3];
    let da = dst[3];
    let out_a = sa + da * (1.0 - sa);
    let denom = out_a.max(1e-6);
    let factor = da * (1.0 - sa);
    [
        (src[0] * sa + dst[0] * factor) / denom,
        (src[1] * sa + dst[1] * factor) / denom,
        (src[2] * sa + dst[2] * factor) / denom,
        out_a,
    ]
}

// Multiply: blend colour is src.rgb * dst.rgb, then composited over dst
// weighted by the source alpha (Porter-Duff source-over using the blended
// colour as the "source" RGB) -- a fully transparent source must not
// darken dst at all, and out.a must still be src.a + dst.a - src.a*dst.a
// (which reusing `blend_over` here reproduces exactly: sa + da*(1-sa)).
fn blend_multiply(src: [f32; 4], dst: [f32; 4]) -> [f32; 4] {
    let blended = [src[0] * dst[0], src[1] * dst[1], src[2] * dst[2], src[3]];
    blend_over(blended, dst)
}

// Screen: out.rgb = 1 - (1-src.rgb)*(1-dst.rgb), composited over dst
// weighted by the source alpha; see `blend_multiply`.
fn blend_screen(src: [f32; 4], dst: [f32; 4]) -> [f32; 4] {
    let blended = [
        1.0 - (1.0 - src[0]) * (1.0 - dst[0]),
        1.0 - (1.0 - src[1]) * (1.0 - dst[1]),
        1.0 - (1.0 - src[2]) * (1.0 - dst[2]),
        src[3],
    ];
    blend_over(blended, dst)
}

// Overlay: if dst < 0.5 → 2*src*dst, else 1 - 2*(1-src)*(1-dst); composited
// over dst weighted by the source alpha; see `blend_multiply`.
fn blend_overlay(src: [f32; 4], dst: [f32; 4]) -> [f32; 4] {
    let overlay_ch = |s: f32, d: f32| -> f32 {
        if d < 0.5 {
            2.0 * s * d
        } else {
            1.0 - 2.0 * (1.0 - s) * (1.0 - d)
        }
    };
    let blended = [
        overlay_ch(src[0], dst[0]),
        overlay_ch(src[1], dst[1]),
        overlay_ch(src[2], dst[2]),
        src[3],
    ];
    blend_over(blended, dst)
}

// Add: clamp(src.rgb + dst.rgb, 0, 1), composited over dst weighted by the
// source alpha; see `blend_multiply`.
fn blend_add(src: [f32; 4], dst: [f32; 4]) -> [f32; 4] {
    let blended = [
        (src[0] + dst[0]).clamp(0.0, 1.0),
        (src[1] + dst[1]).clamp(0.0, 1.0),
        (src[2] + dst[2]).clamp(0.0, 1.0),
        src[3],
    ];
    blend_over(blended, dst)
}

// Difference: |src.rgb - dst.rgb|, composited over dst weighted by the
// source alpha; see `blend_multiply`.
fn blend_difference(src: [f32; 4], dst: [f32; 4]) -> [f32; 4] {
    let blended = [
        (src[0] - dst[0]).abs(),
        (src[1] - dst[1]).abs(),
        (src[2] - dst[2]).abs(),
        src[3],
    ];
    blend_over(blended, dst)
}

// ─────────────────────────────────────────────────────────────────────────────
// Layer
// ─────────────────────────────────────────────────────────────────────────────

/// A compositing layer: an RGBA image with blending parameters.
#[derive(Debug, Clone)]
pub struct Layer {
    /// The RGBA image for this layer.
    pub image: RgbaImage,
    /// How this layer blends with layers below it.
    pub blend_mode: BlendMode,
    /// Global opacity in [0, 1].
    pub opacity: f32,
    /// Optional grayscale mask in [0, 1], length == width * height.
    pub mask: Option<Vec<f32>>,
    /// Human-readable layer name.
    pub name: String,
}

impl Layer {
    /// Create a new layer with Over blend mode, full opacity, and no mask.
    pub fn new(image: RgbaImage, name: impl Into<String>) -> Self {
        Self {
            image,
            blend_mode: BlendMode::Over,
            opacity: 1.0,
            mask: None,
            name: name.into(),
        }
    }

    /// Set the blend mode (builder pattern).
    pub fn with_blend_mode(mut self, mode: BlendMode) -> Self {
        self.blend_mode = mode;
        self
    }

    /// Set global opacity; returns an error if `opacity` is outside [0, 1].
    pub fn with_opacity(mut self, opacity: f32) -> Result<Self, CompositorError> {
        if !(0.0..=1.0).contains(&opacity) {
            return Err(CompositorError::InvalidOpacity(opacity));
        }
        self.opacity = opacity;
        Ok(self)
    }

    /// Attach a grayscale mask; returns an error if its length doesn't match
    /// `width * height`.
    pub fn with_mask(mut self, mask: Vec<f32>) -> Result<Self, CompositorError> {
        if mask.len() != self.image.width * self.image.height {
            return Err(CompositorError::MaskDimensionMismatch);
        }
        self.mask = Some(mask);
        Ok(self)
    }

    /// Effective alpha at pixel `(x, y)`:
    /// `pixel.alpha * opacity * mask[y*width + x]`.
    pub fn effective_alpha(&self, x: usize, y: usize) -> f32 {
        let pixel_alpha = self.image.pixel(x, y)[3];
        let mask_val = match &self.mask {
            Some(m) => m[y * self.image.width + x],
            None => 1.0,
        };
        pixel_alpha * self.opacity * mask_val
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// composite_layers
// ─────────────────────────────────────────────────────────────────────────────

/// Composite all layers from bottom (`layers[0]`) to top (`layers[N-1]`).
///
/// Starts from transparent black `[0,0,0,0]` and applies each layer's
/// blend mode in order. For each pixel the layer's effective alpha
/// (`pixel.alpha * opacity * mask`) replaces the pixel's stored alpha before
/// blending.
pub fn composite_layers(layers: &[Layer]) -> Result<RgbaImage, CompositorError> {
    if layers.is_empty() {
        return Err(CompositorError::NoLayers);
    }

    let w = layers[0].image.width;
    let h = layers[0].image.height;

    // Verify all layers share the same dimensions.
    for layer in layers.iter().skip(1) {
        let lw = layer.image.width;
        let lh = layer.image.height;
        if lw != w || lh != h {
            return Err(CompositorError::DimensionMismatch {
                w1: w,
                h1: h,
                w2: lw,
                h2: lh,
            });
        }
    }

    let mut result = RgbaImage::zeros(w, h);

    for layer in layers {
        for y in 0..h {
            for x in 0..w {
                let dst = result.pixel(x, y);
                let raw_px = layer.image.pixel(x, y);
                let eff_alpha = layer.effective_alpha(x, y);
                // Replace stored alpha with effective alpha before blending.
                let src = [raw_px[0], raw_px[1], raw_px[2], eff_alpha];
                let blended = blend_pixels(src, dst, &layer.blend_mode);
                result.set_pixel(x, y, blended);
            }
        }
    }

    Ok(result)
}

// ─────────────────────────────────────────────────────────────────────────────
// Alpha matting
// ─────────────────────────────────────────────────────────────────────────────

/// Extract foreground and background using a trimap.
///
/// The trimap encodes per-pixel region:
/// - `1.0` = definite foreground → alpha = 1.0
/// - `0.0` = definite background → alpha = 0.0
/// - anything else = unknown → alpha estimated from the fraction of definite
///   foreground pixels in a `(2*radius+1)² box filter around that pixel.
///
/// Returns a copy of `image` with the alpha channel replaced by the estimated
/// mattes.
pub fn apply_trimap_matting(
    image: &RgbaImage,
    trimap: &[f32],
    radius: usize,
) -> Result<RgbaImage, CompositorError> {
    let w = image.width;
    let h = image.height;
    if trimap.len() != w * h {
        return Err(CompositorError::MaskDimensionMismatch);
    }

    let mut out = image.clone();

    for y in 0..h {
        for x in 0..w {
            let t = trimap[y * w + x];
            let alpha = if t >= 1.0 - f32::EPSILON {
                1.0_f32
            } else if t <= f32::EPSILON {
                0.0_f32
            } else {
                // Count foreground pixels in the box.
                let y_lo = y.saturating_sub(radius);
                let y_hi = (y + radius + 1).min(h);
                let x_lo = x.saturating_sub(radius);
                let x_hi = (x + radius + 1).min(w);

                let mut fg_count = 0_usize;
                let mut total = 0_usize;
                for ky in y_lo..y_hi {
                    for kx in x_lo..x_hi {
                        total += 1;
                        if trimap[ky * w + kx] >= 1.0 - f32::EPSILON {
                            fg_count += 1;
                        }
                    }
                }
                if total == 0 {
                    0.0
                } else {
                    fg_count as f32 / total as f32
                }
            };

            let base = (y * w + x) * 4;
            out.data[base + 3] = alpha;
        }
    }

    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// Mask operations
// ─────────────────────────────────────────────────────────────────────────────

/// Erode a binary mask (shrink foreground regions).
///
/// Each output pixel is the minimum value in a `(2*radius+1)²` neighbourhood.
/// Boundary pixels use clamp-to-edge.
pub fn erode_mask(mask: &[f32], width: usize, height: usize, radius: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; width * height];
    for y in 0..height {
        for x in 0..width {
            let y_lo = y.saturating_sub(radius);
            let y_hi = (y + radius + 1).min(height);
            let x_lo = x.saturating_sub(radius);
            let x_hi = (x + radius + 1).min(width);

            let mut min_val = 1.0_f32;
            for ky in y_lo..y_hi {
                for kx in x_lo..x_hi {
                    let v = mask[ky * width + kx];
                    if v < min_val {
                        min_val = v;
                    }
                }
            }
            out[y * width + x] = min_val;
        }
    }
    out
}

/// Dilate a binary mask (expand foreground regions).
///
/// Each output pixel is the maximum value in a `(2*radius+1)²` neighbourhood.
/// Boundary pixels use clamp-to-edge.
pub fn dilate_mask(mask: &[f32], width: usize, height: usize, radius: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; width * height];
    for y in 0..height {
        for x in 0..width {
            let y_lo = y.saturating_sub(radius);
            let y_hi = (y + radius + 1).min(height);
            let x_lo = x.saturating_sub(radius);
            let x_hi = (x + radius + 1).min(width);

            let mut max_val = 0.0_f32;
            for ky in y_lo..y_hi {
                for kx in x_lo..x_hi {
                    let v = mask[ky * width + kx];
                    if v > max_val {
                        max_val = v;
                    }
                }
            }
            out[y * width + x] = max_val;
        }
    }
    out
}

/// Feather (Gaussian-blur) a mask to smooth its edges.
///
/// Uses a separable 1-D Gaussian kernel with half-size `ceil(3*sigma)`.
/// If `sigma <= 0`, the mask is returned unchanged.
pub fn feather_mask(mask: &[f32], width: usize, height: usize, sigma: f32) -> Vec<f32> {
    if sigma <= 0.0 || width == 0 || height == 0 {
        return mask.to_vec();
    }

    let half = (3.0 * sigma).ceil() as usize;
    let kernel_len = 2 * half + 1;

    // Build 1-D Gaussian kernel.
    let mut kernel = vec![0.0_f32; kernel_len];
    let mut sum = 0.0_f32;
    for (k, kval) in kernel.iter_mut().enumerate() {
        let offset = k as f32 - half as f32;
        let w = (-0.5 * (offset / sigma) * (offset / sigma)).exp();
        *kval = w;
        sum += w;
    }
    for kval in kernel.iter_mut() {
        *kval /= sum;
    }

    // Horizontal pass.
    let mut tmp = vec![0.0_f32; width * height];
    for y in 0..height {
        for x in 0..width {
            let mut acc = 0.0_f32;
            for (k, &kval) in kernel.iter().enumerate() {
                let kx = x as isize + k as isize - half as isize;
                let kx_clamped = kx.clamp(0, width as isize - 1) as usize;
                acc += kval * mask[y * width + kx_clamped];
            }
            tmp[y * width + x] = acc;
        }
    }

    // Vertical pass.
    let mut out = vec![0.0_f32; width * height];
    for y in 0..height {
        for x in 0..width {
            let mut acc = 0.0_f32;
            for (k, &kval) in kernel.iter().enumerate() {
                let ky = y as isize + k as isize - half as isize;
                let ky_clamped = ky.clamp(0, height as isize - 1) as usize;
                acc += kval * tmp[ky_clamped * width + x];
            }
            out[y * width + x] = acc;
        }
    }

    out
}

/// Combine two masks with logical AND: `out[i] = a[i].min(b[i])`.
pub fn mask_and(a: &[f32], b: &[f32]) -> Result<Vec<f32>, CompositorError> {
    if a.len() != b.len() {
        return Err(CompositorError::MaskDimensionMismatch);
    }
    Ok(a.iter().zip(b.iter()).map(|(x, y)| x.min(*y)).collect())
}

/// Combine two masks with logical OR: `out[i] = a[i].max(b[i])`.
pub fn mask_or(a: &[f32], b: &[f32]) -> Result<Vec<f32>, CompositorError> {
    if a.len() != b.len() {
        return Err(CompositorError::MaskDimensionMismatch);
    }
    Ok(a.iter().zip(b.iter()).map(|(x, y)| x.max(*y)).collect())
}

/// Invert a mask: `out[i] = 1.0 - mask[i]`.
pub fn mask_not(mask: &[f32]) -> Vec<f32> {
    mask.iter().map(|v| 1.0 - v).collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Compositing statistics
// ─────────────────────────────────────────────────────────────────────────────

/// Summary statistics for a composited RGBA image.
#[derive(Debug, Clone)]
pub struct CompositeStats {
    /// Number of layers that were composited.
    pub num_layers: usize,
    /// Mean alpha value of the final composite.
    pub mean_coverage: f32,
    /// Number of pixels with alpha == 0 (fully transparent).
    pub fully_transparent_pixels: usize,
    /// Number of pixels with alpha == 1 (fully opaque).
    pub fully_opaque_pixels: usize,
    /// Fraction of pixels with alpha > 0.5.
    pub coverage_fraction: f32,
}

/// Compute coverage statistics for an RGBA image.
///
/// `num_layers` is not derived from the image; it is always 0 when called
/// standalone. Use [`composite_layers`] and then call this function to fill it.
pub fn compute_composite_stats(result: &RgbaImage) -> CompositeStats {
    let n = result.width * result.height;
    if n == 0 {
        return CompositeStats {
            num_layers: 0,
            mean_coverage: 0.0,
            fully_transparent_pixels: 0,
            fully_opaque_pixels: 0,
            coverage_fraction: 0.0,
        };
    }

    let mut sum_alpha = 0.0_f32;
    let mut fully_transparent = 0_usize;
    let mut fully_opaque = 0_usize;
    let mut coverage_count = 0_usize;

    for i in 0..n {
        let a = result.data[i * 4 + 3];
        sum_alpha += a;
        if a < f32::EPSILON {
            fully_transparent += 1;
        }
        if (1.0 - a).abs() < f32::EPSILON {
            fully_opaque += 1;
        }
        if a > 0.5 {
            coverage_count += 1;
        }
    }

    CompositeStats {
        num_layers: 0,
        mean_coverage: sum_alpha / n as f32,
        fully_transparent_pixels: fully_transparent,
        fully_opaque_pixels: fully_opaque,
        coverage_fraction: coverage_count as f32 / n as f32,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    // ── Helpers ──────────────────────────────────────────────────────────────

    fn opaque_red(w: usize, h: usize) -> RgbaImage {
        let data = (0..w * h).flat_map(|_| [1.0f32, 0.0, 0.0, 1.0]).collect();
        RgbaImage::new(data, w, h).unwrap()
    }

    fn opaque_blue(w: usize, h: usize) -> RgbaImage {
        let data = (0..w * h).flat_map(|_| [0.0f32, 0.0, 1.0, 1.0]).collect();
        RgbaImage::new(data, w, h).unwrap()
    }

    fn transparent(w: usize, h: usize) -> RgbaImage {
        RgbaImage::zeros(w, h)
    }

    const EPS: f32 = 1e-5;

    fn assert_approx(a: f32, b: f32, label: &str) {
        assert!(
            (a - b).abs() < EPS,
            "{label}: expected {b}, got {a} (diff {})",
            (a - b).abs()
        );
    }

    // ── 1. RgbaImage::new correct dimensions ─────────────────────────────────

    #[test]
    fn test_rgba_new_correct() {
        let img = RgbaImage::new(vec![0.0; 2 * 3 * 4], 2, 3).unwrap();
        assert_eq!(img.width, 2);
        assert_eq!(img.height, 3);
        assert_eq!(img.data.len(), 24);
    }

    // ── 2. RgbaImage::new wrong length → DataLengthMismatch ──────────────────

    #[test]
    fn test_rgba_new_wrong_length() {
        let err = RgbaImage::new(vec![0.0; 5], 2, 3).unwrap_err();
        assert!(matches!(err, CompositorError::DataLengthMismatch { .. }));
    }

    // ── 3. RgbaImage::from_rgb sets alpha correctly ───────────────────────────

    #[test]
    fn test_from_rgb_alpha() {
        let rgb = vec![1.0, 0.5, 0.0, 0.0, 0.0, 0.5];
        let img = RgbaImage::from_rgb(&rgb, 2, 1, 0.75).unwrap();
        assert_eq!(img.data.len(), 8);
        assert_approx(img.data[3], 0.75, "pixel 0 alpha");
        assert_approx(img.data[7], 0.75, "pixel 1 alpha");
        assert_approx(img.data[0], 1.0, "pixel 0 r");
        assert_approx(img.data[4], 0.0, "pixel 1 r");
    }

    // ── 4. RgbaImage::pixel returns correct RGBA ──────────────────────────────

    #[test]
    fn test_pixel_accessor() {
        let mut data = vec![0.0_f32; 2 * 2 * 4];
        // pixel (1, 0) → index 1
        data[4] = 0.2;
        data[5] = 0.4;
        data[6] = 0.6;
        data[7] = 0.8;
        let img = RgbaImage::new(data, 2, 2).unwrap();
        let px = img.pixel(1, 0);
        assert_approx(px[0], 0.2, "r");
        assert_approx(px[1], 0.4, "g");
        assert_approx(px[2], 0.6, "b");
        assert_approx(px[3], 0.8, "a");
    }

    // ── 5. blend_pixels Over: opaque src hides dst ────────────────────────────

    #[test]
    fn test_blend_over_opaque_src() {
        let src = [1.0, 0.0, 0.0, 1.0]; // opaque red
        let dst = [0.0, 0.0, 1.0, 1.0]; // opaque blue
        let out = blend_pixels(src, dst, &BlendMode::Over);
        assert_approx(out[0], 1.0, "r");
        assert_approx(out[1], 0.0, "g");
        assert_approx(out[2], 0.0, "b");
        assert_approx(out[3], 1.0, "a");
    }

    // ── 6. blend_pixels Over: transparent src shows dst ──────────────────────

    #[test]
    fn test_blend_over_transparent_src() {
        let src = [1.0, 0.0, 0.0, 0.0]; // transparent red
        let dst = [0.0, 0.0, 1.0, 1.0]; // opaque blue
        let out = blend_pixels(src, dst, &BlendMode::Over);
        assert_approx(out[0], 0.0, "r");
        assert_approx(out[1], 0.0, "g");
        assert_approx(out[2], 1.0, "b");
        assert_approx(out[3], 1.0, "a");
    }

    // ── 7. blend_pixels Multiply ──────────────────────────────────────────────

    #[test]
    fn test_blend_multiply() {
        let src = [0.5, 0.5, 0.5, 1.0];
        let dst = [0.5, 0.5, 0.5, 1.0];
        let out = blend_pixels(src, dst, &BlendMode::Multiply);
        assert_approx(out[0], 0.25, "r");
        assert_approx(out[1], 0.25, "g");
        assert_approx(out[2], 0.25, "b");
        assert_approx(out[3], 1.0, "a"); // 1 + 1 - 1*1
    }

    #[test]
    fn test_blend_modes_respect_source_alpha() {
        // Regression test: a fully transparent source (alpha == 0) must
        // leave the destination completely unchanged in every non-`Over`
        // blend mode, not fully overwrite it with the blended colour (the
        // bug: these modes computed the blend from src.rgb/dst.rgb alone
        // and used src.a only for the output alpha, never to weight rgb).
        let dst = [0.4_f32, 0.8, 0.5, 0.6];
        let transparent_src = [0.9_f32, 0.1, 0.2, 0.0];

        for mode in [
            BlendMode::Multiply,
            BlendMode::Screen,
            BlendMode::Overlay,
            BlendMode::Add,
            BlendMode::Difference,
        ] {
            let out = blend_pixels(transparent_src, dst, &mode);
            assert_approx(out[0], dst[0], &format!("{mode:?} r"));
            assert_approx(out[1], dst[1], &format!("{mode:?} g"));
            assert_approx(out[2], dst[2], &format!("{mode:?} b"));
            assert_approx(out[3], dst[3], &format!("{mode:?} a"));
        }
    }

    #[test]
    fn test_blend_multiply_partial_alpha_interpolates() {
        // With a half-opaque source, the multiply result must sit strictly
        // between the unblended destination and the fully-opaque blended
        // colour (a proper lerp by source alpha) -- not jump straight to
        // the fully-opaque blended result regardless of alpha.
        let dst = [0.8_f32, 0.8, 0.8, 1.0];
        let half_src = [0.2_f32, 0.2, 0.2, 0.5];
        let out = blend_pixels(half_src, dst, &BlendMode::Multiply);
        let full_blend = 0.2_f32 * 0.8; // src.rgb * dst.rgb, the fully-opaque case
        let expected = dst[0] * 0.5 + full_blend * 0.5; // lerp(dst, blend, sa=0.5)
        assert_approx(out[0], expected, "r");
        assert!(
            out[0] > full_blend && out[0] < dst[0],
            "half-alpha multiply result {} should lie strictly between the full blend {} and dst {}",
            out[0],
            full_blend,
            dst[0]
        );
    }

    // ── 8. blend_pixels Add: clamped to [0,1] ────────────────────────────────

    #[test]
    fn test_blend_add_clamped() {
        let src = [0.8, 0.8, 0.8, 1.0];
        let dst = [0.8, 0.8, 0.8, 1.0];
        let out = blend_pixels(src, dst, &BlendMode::Add);
        assert_approx(out[0], 1.0, "r clamped");
        assert_approx(out[1], 1.0, "g clamped");
        assert_approx(out[2], 1.0, "b clamped");
    }

    // ── 9. blend_pixels Difference ───────────────────────────────────────────

    #[test]
    fn test_blend_difference() {
        let src = [0.9, 0.3, 0.5, 1.0];
        let dst = [0.4, 0.8, 0.5, 1.0];
        let out = blend_pixels(src, dst, &BlendMode::Difference);
        assert_approx(out[0], 0.5, "r");
        assert_approx(out[1], 0.5, "g");
        assert_approx(out[2], 0.0, "b");
    }

    // ── 10. composite_layers single opaque layer → that image ────────────────

    #[test]
    fn test_composite_single_layer() {
        let red = opaque_red(4, 4);
        let layer = Layer::new(red.clone(), "red");
        let result = composite_layers(&[layer]).unwrap();
        let px = result.pixel(2, 2);
        assert_approx(px[0], 1.0, "r");
        assert_approx(px[1], 0.0, "g");
        assert_approx(px[2], 0.0, "b");
        assert_approx(px[3], 1.0, "a");
    }

    // ── 11. composite_layers two opaque layers (Over): top hides bottom ──────

    #[test]
    fn test_composite_two_opaque_layers() {
        let red = opaque_red(4, 4);
        let blue = opaque_blue(4, 4);
        let bottom = Layer::new(red, "red");
        let top = Layer::new(blue, "blue");
        let result = composite_layers(&[bottom, top]).unwrap();
        let px = result.pixel(0, 0);
        // Blue is on top → should see blue
        assert_approx(px[2], 1.0, "b");
        assert_approx(px[0], 0.0, "r");
    }

    // ── 12. transparent top over opaque bottom → bottom shows through ─────────

    #[test]
    fn test_composite_transparent_top() {
        let red = opaque_red(4, 4);
        let transp = transparent(4, 4);
        let bottom = Layer::new(red, "red");
        let top = Layer::new(transp, "transparent");
        let result = composite_layers(&[bottom, top]).unwrap();
        let px = result.pixel(1, 1);
        assert_approx(px[0], 1.0, "r");
        assert_approx(px[3], 1.0, "a");
    }

    // ── 13. composite_layers no layers → NoLayers error ──────────────────────

    #[test]
    fn test_composite_no_layers() {
        let err = composite_layers(&[]).unwrap_err();
        assert!(matches!(err, CompositorError::NoLayers));
    }

    // ── 14. Layer::with_opacity 0.5 makes layer semi-transparent ─────────────

    #[test]
    fn test_layer_opacity() {
        let red = opaque_red(2, 2);
        let layer = Layer::new(red, "red").with_opacity(0.5).unwrap();
        // effective alpha at any pixel = 1.0 * 0.5 * 1.0 (no mask)
        let eff = layer.effective_alpha(0, 0);
        assert_approx(eff, 0.5, "effective alpha");
    }

    // ── 15. Layer::effective_alpha with opacity + mask ────────────────────────

    #[test]
    fn test_layer_effective_alpha_with_mask() {
        let red = opaque_red(2, 2);
        let mask = vec![0.8_f32, 0.4, 0.6, 1.0];
        let layer = Layer::new(red, "red")
            .with_opacity(0.5)
            .unwrap()
            .with_mask(mask)
            .unwrap();
        // pixel (0, 0): alpha=1.0, opacity=0.5, mask=0.8 → 0.4
        assert_approx(layer.effective_alpha(0, 0), 0.4, "px(0,0)");
        // pixel (1, 0): alpha=1.0, opacity=0.5, mask=0.4 → 0.2
        assert_approx(layer.effective_alpha(1, 0), 0.2, "px(1,0)");
        // pixel (0, 1): alpha=1.0, opacity=0.5, mask=0.6 → 0.3
        assert_approx(layer.effective_alpha(0, 1), 0.3, "px(0,1)");
        // pixel (1, 1): alpha=1.0, opacity=0.5, mask=1.0 → 0.5
        assert_approx(layer.effective_alpha(1, 1), 0.5, "px(1,1)");
    }

    // ── 16. premultiply/unpremultiply roundtrip ───────────────────────────────

    #[test]
    fn test_premultiply_roundtrip() {
        let data: Vec<f32> = vec![0.8, 0.6, 0.4, 0.5, 0.2, 0.9, 0.1, 0.9];
        let img = RgbaImage::new(data.clone(), 2, 1).unwrap();
        let premul = img.premultiply_alpha();
        let unpremul = premul.unpremultiply_alpha();
        for (i, (&got, &orig)) in unpremul.data.iter().zip(data.iter()).enumerate().take(8) {
            let ch = i % 4;
            if ch != 3 {
                // RGB channels should roundtrip for alpha > 0.1
                assert!(
                    (got - orig).abs() < 1e-4,
                    "channel mismatch at index {i}: {} vs {}",
                    got,
                    orig
                );
            }
        }
    }

    // ── 17. erode_mask shrinks foreground region ──────────────────────────────

    #[test]
    fn test_erode_mask() {
        // 5×1 mask: [1,1,1,1,1], erode radius 1 → still [1,1,1,1,1]
        // because all neighbours are 1 within bounds.
        let mask = vec![1.0_f32; 5];
        let eroded = erode_mask(&mask, 5, 1, 1);
        assert_approx(eroded[2], 1.0, "center of all-ones eroded");

        // 5×1 mask: [0,1,1,1,0], erode radius 1 → [0,0,1,0,0]
        let mask2 = vec![0.0_f32, 1.0, 1.0, 1.0, 0.0];
        let eroded2 = erode_mask(&mask2, 5, 1, 1);
        // center pixel neighbourhood includes 1,1,1 → min=1
        // but index 1 has neighbours [0,1,1] → min=0
        assert_approx(eroded2[0], 0.0, "edge stays 0");
        assert_approx(eroded2[1], 0.0, "adjacent to 0 eroded");
        assert_approx(eroded2[2], 1.0, "center stays 1");
        assert_approx(eroded2[3], 0.0, "adjacent to 0 eroded");
        assert_approx(eroded2[4], 0.0, "edge stays 0");
    }

    // ── 18. dilate_mask expands foreground region ─────────────────────────────

    #[test]
    fn test_dilate_mask() {
        // 5×1: [0,0,1,0,0], dilate radius 1 → [0,1,1,1,0]
        let mask = vec![0.0_f32, 0.0, 1.0, 0.0, 0.0];
        let dilated = dilate_mask(&mask, 5, 1, 1);
        assert_approx(dilated[0], 0.0, "far left");
        assert_approx(dilated[1], 1.0, "left of center dilated");
        assert_approx(dilated[2], 1.0, "center");
        assert_approx(dilated[3], 1.0, "right of center dilated");
        assert_approx(dilated[4], 0.0, "far right");
    }

    // ── 19. mask_not inverts 0→1 and 1→0 ────────────────────────────────────

    #[test]
    fn test_mask_not() {
        let mask = vec![0.0_f32, 1.0, 0.5];
        let inv = mask_not(&mask);
        assert_approx(inv[0], 1.0, "0 → 1");
        assert_approx(inv[1], 0.0, "1 → 0");
        assert_approx(inv[2], 0.5, "0.5 → 0.5");
    }

    // ── 20. mask_and correct boolean AND ─────────────────────────────────────

    #[test]
    fn test_mask_and() {
        let a = vec![1.0_f32, 1.0, 0.0, 0.5];
        let b = vec![1.0_f32, 0.0, 1.0, 0.8];
        let out = mask_and(&a, &b).unwrap();
        assert_approx(out[0], 1.0, "[0]");
        assert_approx(out[1], 0.0, "[1]");
        assert_approx(out[2], 0.0, "[2]");
        assert_approx(out[3], 0.5, "[3]");
    }

    // ── 21. compute_composite_stats correct coverage_fraction ────────────────

    #[test]
    fn test_composite_stats_coverage() {
        // 4 pixels: alphas [0.0, 0.4, 0.6, 1.0]
        // > 0.5: indices 2 (0.6) and 3 (1.0) → fraction = 2/4 = 0.5
        let data = vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 1.0,
        ];
        let img = RgbaImage::new(data, 4, 1).unwrap();
        let stats = compute_composite_stats(&img);
        assert_approx(stats.coverage_fraction, 0.5, "coverage_fraction");
        assert_eq!(stats.fully_transparent_pixels, 1);
        assert_eq!(stats.fully_opaque_pixels, 1);
        assert_approx(
            stats.mean_coverage,
            (0.0 + 0.4 + 0.6 + 1.0) / 4.0,
            "mean_coverage",
        );
    }

    // ── 22. feather_mask output values in [0,1] ──────────────────────────────

    #[test]
    fn test_feather_mask_range() {
        let mask: Vec<f32> = (0..25)
            .map(|i| if i % 3 == 0 { 1.0 } else { 0.0 })
            .collect();
        let feathered = feather_mask(&mask, 5, 5, 1.0);
        assert_eq!(feathered.len(), 25);
        for (i, &v) in feathered.iter().enumerate() {
            assert!(
                (0.0..=1.0 + 1e-5).contains(&v),
                "feathered[{i}] = {v} out of [0,1]"
            );
        }
    }
}
