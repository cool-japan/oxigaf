//! Image compositing: blend mode operations, layer management, and alpha handling.
//!
//! Operates on RGBA u8 pixel data (row-major, H×W×4). Provides Porter-Duff
//! alpha compositing, multiple blend modes, and utility functions for layer
//! stack management.

use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors produced by image compositing operations.
#[derive(Debug, Error)]
pub enum ImageCompositorError {
    /// Image dimensions do not match the expected dimensions.
    #[error("Dimension mismatch: expected {expected_w}x{expected_h}, got {w}x{h}")]
    DimensionMismatch {
        /// Expected image width.
        expected_w: usize,
        /// Expected image height.
        expected_h: usize,
        /// Actual image width.
        w: usize,
        /// Actual image height.
        h: usize,
    },

    /// Image data is empty (zero pixels or zero-size buffer).
    #[error("Empty image")]
    EmptyImage,

    /// No layers were provided to composite.
    #[error("No layers")]
    NoLayers,

    /// Layer index is out of bounds.
    #[error("Invalid layer index: {index}, total layers: {total}")]
    InvalidLayerIndex {
        /// The requested index.
        index: usize,
        /// Total number of layers available.
        total: usize,
    },

    /// Opacity value is outside the valid [0, 1] range.
    #[error("Invalid opacity: {0}, must be in [0, 1]")]
    InvalidOpacity(f32),

    /// General configuration error.
    #[error("Invalid config: {0}")]
    InvalidConfig(String),
}

// ─────────────────────────────────────────────────────────────────────────────
// Blend modes
// ─────────────────────────────────────────────────────────────────────────────

/// Blend modes for compositing layers.
///
/// RGB blending is applied per-channel using [`BlendMode::apply_channel`].
/// Alpha compositing always uses Porter-Duff "over" regardless of blend mode.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BlendMode {
    /// Standard alpha compositing ("over").
    Normal,
    /// Multiply: `src * dst`.
    Multiply,
    /// Screen: `1 - (1-src)*(1-dst)`.
    Screen,
    /// Overlay: multiply if dst < 0.5, screen otherwise.
    Overlay,
    /// Soft light (Pegtop formula): `(1-2*src)*dst*dst + 2*src*dst`.
    SoftLight,
    /// Hard light: Overlay with src and dst roles swapped.
    HardLight,
    /// Color dodge: `dst / (1-src)` clamped to [0, 1].
    ColorDodge,
    /// Color burn: `1 - (1-dst)/src` clamped to [0, 1].
    ColorBurn,
    /// Darken: `min(src, dst)`.
    Darken,
    /// Lighten: `max(src, dst)`.
    Lighten,
    /// Difference: `|src - dst|`.
    Difference,
    /// Exclusion: `src + dst - 2*src*dst`.
    Exclusion,
    /// Add: `src + dst` clamped to [0, 1].
    Add,
    /// Subtract: `dst - src` clamped (no negatives).
    Subtract,
}

impl BlendMode {
    /// Apply the blend mode to a single channel.
    ///
    /// `src` and `dst` are in [0, 1]. Returns a value also in [0, 1].
    pub fn apply_channel(&self, src: f32, dst: f32) -> f32 {
        let v = match self {
            BlendMode::Normal => src,
            BlendMode::Multiply => src * dst,
            BlendMode::Screen => 1.0 - (1.0 - src) * (1.0 - dst),
            BlendMode::Overlay => {
                if dst < 0.5 {
                    2.0 * src * dst
                } else {
                    1.0 - 2.0 * (1.0 - src) * (1.0 - dst)
                }
            }
            BlendMode::SoftLight => (1.0 - 2.0 * src) * dst * dst + 2.0 * src * dst,
            BlendMode::HardLight => {
                // HardLight = Overlay with src/dst swapped
                if src < 0.5 {
                    2.0 * src * dst
                } else {
                    1.0 - 2.0 * (1.0 - src) * (1.0 - dst)
                }
            }
            BlendMode::ColorDodge => {
                let denom = 1.0 - src;
                if denom < 1e-6 {
                    1.0
                } else {
                    (dst / denom).min(1.0)
                }
            }
            BlendMode::ColorBurn => {
                if src < 1e-6 {
                    0.0
                } else {
                    (1.0 - (1.0 - dst) / src).max(0.0)
                }
            }
            BlendMode::Darken => src.min(dst),
            BlendMode::Lighten => src.max(dst),
            BlendMode::Difference => (src - dst).abs(),
            BlendMode::Exclusion => src + dst - 2.0 * src * dst,
            BlendMode::Add => (src + dst).min(1.0),
            BlendMode::Subtract => (dst - src).max(0.0),
        };
        v.clamp(0.0, 1.0)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CompositingLayer
// ─────────────────────────────────────────────────────────────────────────────

/// A single compositing layer holding RGBA pixel data.
///
/// Pixels are stored as `u8` values, row-major, 4 bytes per pixel (RGBA).
#[derive(Debug, Clone)]
pub struct CompositingLayer {
    /// RGBA pixel data (length == width * height * 4).
    pub pixels: Vec<u8>,
    /// Layer width in pixels.
    pub width: usize,
    /// Layer height in pixels.
    pub height: usize,
    /// Global layer opacity in [0, 1].
    pub opacity: f32,
    /// Blend mode used when compositing this layer over the one below.
    pub blend_mode: BlendMode,
    /// Human-readable layer name.
    pub name: String,
    /// Whether this layer is included in compositing.
    pub visible: bool,
}

impl CompositingLayer {
    /// Create a new layer from RGBA u8 pixels.
    ///
    /// Validates that `pixels.len() == width * height * 4` and that the image
    /// is non-empty.
    pub fn new(pixels: Vec<u8>, width: usize, height: usize) -> Result<Self, ImageCompositorError> {
        if width == 0 || height == 0 {
            return Err(ImageCompositorError::EmptyImage);
        }
        let expected = width * height * 4;
        if pixels.len() != expected {
            return Err(ImageCompositorError::DimensionMismatch {
                expected_w: width,
                expected_h: height,
                w: pixels.len() / height.max(1) / 4,
                h: height,
            });
        }
        Ok(Self {
            pixels,
            width,
            height,
            opacity: 1.0,
            blend_mode: BlendMode::Normal,
            name: String::new(),
            visible: true,
        })
    }

    /// Set the layer opacity, returning an error if it is outside [0, 1].
    pub fn with_opacity(mut self, opacity: f32) -> Result<Self, ImageCompositorError> {
        if !(0.0..=1.0).contains(&opacity) {
            return Err(ImageCompositorError::InvalidOpacity(opacity));
        }
        self.opacity = opacity;
        Ok(self)
    }

    /// Set the blend mode for this layer.
    pub fn with_blend_mode(mut self, mode: BlendMode) -> Self {
        self.blend_mode = mode;
        self
    }

    /// Set the layer name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Return the RGBA values of the pixel at `(x, y)`, or `None` if out of bounds.
    pub fn pixel_rgba(&self, x: usize, y: usize) -> Option<(u8, u8, u8, u8)> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let base = (y * self.width + x) * 4;
        self.pixels
            .get(base..base + 4)
            .map(|s| (s[0], s[1], s[2], s[3]))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CompositorConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the image compositor.
#[derive(Debug, Clone)]
pub struct CompositorConfig {
    /// When `true`, each layer's pixel data is treated as already
    /// premultiplied by alpha: [`composite_image_layers`] un-premultiplies
    /// it before compositing, since the Porter-Duff math in
    /// [`apply_blend_mode`] / [`composite_over`] assumes straight alpha.
    /// The composited output is always in straight-alpha form regardless
    /// of this flag. Default: `false`. Only consumed by
    /// [`composite_image_layers`] — the lower-level [`composite_over`] and
    /// [`apply_blend_mode`] functions always assume straight alpha input.
    pub premultiplied_alpha: bool,
    /// Background RGBA colour filled before any layers are applied.
    /// Default: opaque black `[0, 0, 0, 255]`.
    pub background_color: [u8; 4],
}

impl Default for CompositorConfig {
    fn default() -> Self {
        Self {
            premultiplied_alpha: false,
            background_color: [0, 0, 0, 255],
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Statistics
// ─────────────────────────────────────────────────────────────────────────────

/// Per-image statistics computed over all pixels.
#[derive(Debug, Clone)]
pub struct ImageCompositeStats {
    /// Mean alpha value over all pixels (in [0, 1]).
    pub mean_alpha: f32,
    /// Fraction of pixels with alpha >= 127 (in [0, 1]).
    pub opaque_fraction: f32,
    /// Fraction of pixels with alpha == 0 (in [0, 1]).
    pub transparent_fraction: f32,
    /// Mean RGB value over all pixels (each channel in [0, 1]).
    pub mean_rgb: [f32; 3],
}

// ─────────────────────────────────────────────────────────────────────────────
// Validation helpers
// ─────────────────────────────────────────────────────────────────────────────

fn validate_rgba_buffer(
    pixels: &[u8],
    width: usize,
    height: usize,
) -> Result<(), ImageCompositorError> {
    if width == 0 || height == 0 {
        return Err(ImageCompositorError::EmptyImage);
    }
    let expected = width * height * 4;
    if pixels.len() != expected {
        return Err(ImageCompositorError::DimensionMismatch {
            expected_w: width,
            expected_h: height,
            w: width,
            h: pixels.len() / (width * 4).max(1),
        });
    }
    Ok(())
}

fn validate_opacity(opacity: f32) -> Result<(), ImageCompositorError> {
    if !(0.0..=1.0).contains(&opacity) {
        return Err(ImageCompositorError::InvalidOpacity(opacity));
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Core compositing functions
// ─────────────────────────────────────────────────────────────────────────────

/// Composite two RGBA images using Porter-Duff "over" (top over bottom).
///
/// `top_opacity` scales the top layer's alpha before compositing.
/// Both images must have identical dimensions.
pub fn composite_over(
    bottom: &[u8],
    top: &[u8],
    width: usize,
    height: usize,
    top_opacity: f32,
) -> Result<Vec<u8>, ImageCompositorError> {
    validate_rgba_buffer(bottom, width, height)?;
    validate_rgba_buffer(top, width, height)?;
    validate_opacity(top_opacity)?;

    let n_pixels = width * height;
    let mut out = Vec::with_capacity(n_pixels * 4);

    for i in 0..n_pixels {
        let base = i * 4;

        let dst_r = bottom[base] as f32 / 255.0;
        let dst_g = bottom[base + 1] as f32 / 255.0;
        let dst_b = bottom[base + 2] as f32 / 255.0;
        let dst_a = bottom[base + 3] as f32 / 255.0;

        let src_r = top[base] as f32 / 255.0;
        let src_g = top[base + 1] as f32 / 255.0;
        let src_b = top[base + 2] as f32 / 255.0;
        let src_a = (top[base + 3] as f32 / 255.0) * top_opacity;

        let out_a = src_a + dst_a * (1.0 - src_a);
        let denom = out_a.max(1e-7);
        let k = dst_a * (1.0 - src_a);

        let out_r = (src_r * src_a + dst_r * k) / denom;
        let out_g = (src_g * src_a + dst_g * k) / denom;
        let out_b = (src_b * src_a + dst_b * k) / denom;

        out.push((out_r.clamp(0.0, 1.0) * 255.0).round() as u8);
        out.push((out_g.clamp(0.0, 1.0) * 255.0).round() as u8);
        out.push((out_b.clamp(0.0, 1.0) * 255.0).round() as u8);
        out.push((out_a.clamp(0.0, 1.0) * 255.0).round() as u8);
    }

    Ok(out)
}

/// Blend two RGB colour triples using the given blend mode.
///
/// Inputs and output are each in [0, 1].
pub fn blend_pixel(src: [f32; 3], dst: [f32; 3], mode: BlendMode) -> [f32; 3] {
    [
        mode.apply_channel(src[0], dst[0]),
        mode.apply_channel(src[1], dst[1]),
        mode.apply_channel(src[2], dst[2]),
    ]
}

/// Apply a blend mode to a layer on top of a base image.
///
/// The blend mode affects RGB channels only; alpha compositing uses Porter-Duff
/// "over" with `layer_opacity` scaling the layer's alpha.
pub fn apply_blend_mode(
    base: &[u8],
    layer: &[u8],
    width: usize,
    height: usize,
    mode: BlendMode,
    layer_opacity: f32,
) -> Result<Vec<u8>, ImageCompositorError> {
    validate_rgba_buffer(base, width, height)?;
    validate_rgba_buffer(layer, width, height)?;
    validate_opacity(layer_opacity)?;

    let n_pixels = width * height;
    let mut out = Vec::with_capacity(n_pixels * 4);

    for i in 0..n_pixels {
        let base_off = i * 4;

        let dst_r = base[base_off] as f32 / 255.0;
        let dst_g = base[base_off + 1] as f32 / 255.0;
        let dst_b = base[base_off + 2] as f32 / 255.0;
        let dst_a = base[base_off + 3] as f32 / 255.0;

        let src_r = layer[base_off] as f32 / 255.0;
        let src_g = layer[base_off + 1] as f32 / 255.0;
        let src_b = layer[base_off + 2] as f32 / 255.0;
        let src_a = (layer[base_off + 3] as f32 / 255.0) * layer_opacity;

        // RGB channels: blend using the requested mode
        let blended_r = mode.apply_channel(src_r, dst_r);
        let blended_g = mode.apply_channel(src_g, dst_g);
        let blended_b = mode.apply_channel(src_b, dst_b);

        // Porter-Duff "over" for final alpha and to combine blended/dst RGB
        let out_a = src_a + dst_a * (1.0 - src_a);
        let denom = out_a.max(1e-7);
        let k = dst_a * (1.0 - src_a);

        // Linear interpolation between blended result (for src) and dst
        let out_r = (blended_r * src_a + dst_r * k) / denom;
        let out_g = (blended_g * src_a + dst_g * k) / denom;
        let out_b = (blended_b * src_a + dst_b * k) / denom;

        out.push((out_r.clamp(0.0, 1.0) * 255.0).round() as u8);
        out.push((out_g.clamp(0.0, 1.0) * 255.0).round() as u8);
        out.push((out_b.clamp(0.0, 1.0) * 255.0).round() as u8);
        out.push((out_a.clamp(0.0, 1.0) * 255.0).round() as u8);
    }

    Ok(out)
}

/// Composite a stack of layers bottom-to-top.
///
/// Each layer's `blend_mode` and `opacity` are applied in order. Invisible
/// layers are skipped. The result is initialised with the background colour
/// from `config`.
pub fn composite_image_layers(
    layers: &[CompositingLayer],
    width: usize,
    height: usize,
    config: &CompositorConfig,
) -> Result<Vec<u8>, ImageCompositorError> {
    if layers.is_empty() {
        return Err(ImageCompositorError::NoLayers);
    }
    if width == 0 || height == 0 {
        return Err(ImageCompositorError::EmptyImage);
    }

    // Validate all layer dimensions up front
    for (idx, layer) in layers.iter().enumerate() {
        if layer.width != width || layer.height != height {
            return Err(ImageCompositorError::DimensionMismatch {
                expected_w: width,
                expected_h: height,
                w: layer.width,
                h: layer.height,
            });
        }
        let expected = width * height * 4;
        if layer.pixels.len() != expected {
            return Err(ImageCompositorError::InvalidLayerIndex {
                index: idx,
                total: layers.len(),
            });
        }
    }

    // Initialise accumulator with background colour
    let bg = config.background_color;
    let mut acc = solid_color(width, height, bg[0], bg[1], bg[2], bg[3]);

    for layer in layers {
        if !layer.visible {
            continue;
        }
        // `apply_blend_mode`'s Porter-Duff math (and every `BlendMode`
        // formula) assumes *straight* (non-premultiplied) alpha. If the
        // caller has declared the layer data as already premultiplied,
        // convert it to straight alpha first — otherwise the src*alpha
        // multiply happens a second time, producing doubly-multiplied,
        // over-dark output with no warning (the bug this branch fixes).
        let straight_pixels;
        let layer_pixels: &[u8] = if config.premultiplied_alpha {
            let mut p = layer.pixels.clone();
            unpremultiply_alpha(&mut p)?;
            straight_pixels = p;
            &straight_pixels
        } else {
            &layer.pixels
        };
        acc = apply_blend_mode(
            &acc,
            layer_pixels,
            width,
            height,
            layer.blend_mode,
            layer.opacity,
        )?;
    }

    Ok(acc)
}

// ─────────────────────────────────────────────────────────────────────────────
// Alpha helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Premultiply alpha in-place: `rgb = rgb * a / 255`.
pub fn premultiply_alpha(pixels: &mut [u8]) -> Result<(), ImageCompositorError> {
    if pixels.is_empty() {
        return Err(ImageCompositorError::EmptyImage);
    }
    if !pixels.len().is_multiple_of(4) {
        return Err(ImageCompositorError::InvalidConfig(
            "pixel buffer length must be a multiple of 4".to_string(),
        ));
    }
    for chunk in pixels.chunks_exact_mut(4) {
        let a = chunk[3] as u32;
        chunk[0] = ((chunk[0] as u32 * a + 127) / 255) as u8;
        chunk[1] = ((chunk[1] as u32 * a + 127) / 255) as u8;
        chunk[2] = ((chunk[2] as u32 * a + 127) / 255) as u8;
    }
    Ok(())
}

/// Undo premultiplied alpha in-place: `rgb = rgb * 255 / a`.
///
/// Pixels with alpha == 0 are left unchanged (black transparent).
pub fn unpremultiply_alpha(pixels: &mut [u8]) -> Result<(), ImageCompositorError> {
    if pixels.is_empty() {
        return Err(ImageCompositorError::EmptyImage);
    }
    if !pixels.len().is_multiple_of(4) {
        return Err(ImageCompositorError::InvalidConfig(
            "pixel buffer length must be a multiple of 4".to_string(),
        ));
    }
    for chunk in pixels.chunks_exact_mut(4) {
        let a = chunk[3];
        if a == 0 {
            chunk[0] = 0;
            chunk[1] = 0;
            chunk[2] = 0;
        } else {
            let a32 = a as u32;
            chunk[0] = ((chunk[0] as u32 * 255 + a32 / 2) / a32).min(255) as u8;
            chunk[1] = ((chunk[1] as u32 * 255 + a32 / 2) / a32).min(255) as u8;
            chunk[2] = ((chunk[2] as u32 * 255 + a32 / 2) / a32).min(255) as u8;
        }
    }
    Ok(())
}

/// Scale all alpha values by `factor` (layer masking).
///
/// `factor` must be in [0, 1].
pub fn scale_alpha(pixels: &mut [u8], factor: f32) -> Result<(), ImageCompositorError> {
    if pixels.is_empty() {
        return Err(ImageCompositorError::EmptyImage);
    }
    validate_opacity(factor)?;
    if !pixels.len().is_multiple_of(4) {
        return Err(ImageCompositorError::InvalidConfig(
            "pixel buffer length must be a multiple of 4".to_string(),
        ));
    }
    for chunk in pixels.chunks_exact_mut(4) {
        chunk[3] = ((chunk[3] as f32 * factor).round().clamp(0.0, 255.0)) as u8;
    }
    Ok(())
}

/// Extract the alpha channel as a grayscale `Vec<u8>` (one byte per pixel).
pub fn extract_alpha(pixels: &[u8]) -> Result<Vec<u8>, ImageCompositorError> {
    if pixels.is_empty() {
        return Err(ImageCompositorError::EmptyImage);
    }
    if !pixels.len().is_multiple_of(4) {
        return Err(ImageCompositorError::InvalidConfig(
            "pixel buffer length must be a multiple of 4".to_string(),
        ));
    }
    let n = pixels.len() / 4;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(pixels[i * 4 + 3]);
    }
    Ok(out)
}

/// Replace the alpha channel of an RGBA image with new grayscale alpha values.
pub fn replace_alpha(
    pixels: &[u8],
    new_alpha: &[u8],
    width: usize,
    height: usize,
) -> Result<Vec<u8>, ImageCompositorError> {
    validate_rgba_buffer(pixels, width, height)?;
    let n = width * height;
    if new_alpha.len() != n {
        return Err(ImageCompositorError::DimensionMismatch {
            expected_w: width,
            expected_h: height,
            w: new_alpha.len(),
            h: 1,
        });
    }
    let mut out = pixels.to_vec();
    for i in 0..n {
        out[i * 4 + 3] = new_alpha[i];
    }
    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// Generation helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Create a solid-colour RGBA image with the given dimensions.
pub fn solid_color(width: usize, height: usize, r: u8, g: u8, b: u8, a: u8) -> Vec<u8> {
    let n = width * height;
    let mut out = Vec::with_capacity(n * 4);
    for _ in 0..n {
        out.push(r);
        out.push(g);
        out.push(b);
        out.push(a);
    }
    out
}

/// Create a checkerboard pattern for previewing transparent regions.
///
/// Alternating `cell_size`×`cell_size` tiles of light grey (192) and dark
/// grey (128), both fully opaque.
pub fn transparency_checkerboard(width: usize, height: usize, cell_size: usize) -> Vec<u8> {
    let cell_size = cell_size.max(1);
    let n = width * height;
    let mut out = Vec::with_capacity(n * 4);
    for y in 0..height {
        for x in 0..width {
            let tile_x = x / cell_size;
            let tile_y = y / cell_size;
            let light = (tile_x + tile_y).is_multiple_of(2);
            let v: u8 = if light { 192 } else { 128 };
            out.push(v);
            out.push(v);
            out.push(v);
            out.push(255);
        }
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Conversion helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Flatten an RGBA image to RGB by compositing over a solid background colour.
pub fn flatten_to_rgb(
    pixels: &[u8],
    background: [u8; 3],
    width: usize,
    height: usize,
) -> Result<Vec<u8>, ImageCompositorError> {
    validate_rgba_buffer(pixels, width, height)?;
    let n = width * height;
    let mut out = Vec::with_capacity(n * 3);
    for i in 0..n {
        let base = i * 4;
        let r = pixels[base] as f32 / 255.0;
        let g = pixels[base + 1] as f32 / 255.0;
        let b = pixels[base + 2] as f32 / 255.0;
        let a = pixels[base + 3] as f32 / 255.0;

        let bg_r = background[0] as f32 / 255.0;
        let bg_g = background[1] as f32 / 255.0;
        let bg_b = background[2] as f32 / 255.0;

        // Composite: out = src * alpha + bg * (1 - alpha)
        let out_r = r * a + bg_r * (1.0 - a);
        let out_g = g * a + bg_g * (1.0 - a);
        let out_b = b * a + bg_b * (1.0 - a);

        out.push((out_r.clamp(0.0, 1.0) * 255.0).round() as u8);
        out.push((out_g.clamp(0.0, 1.0) * 255.0).round() as u8);
        out.push((out_b.clamp(0.0, 1.0) * 255.0).round() as u8);
    }
    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// Quality metrics
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the mean PSNR between two RGBA images, weighted by alpha.
///
/// Returns `f32::INFINITY` if the images are identical.
pub fn composite_psnr(
    img_a: &[u8],
    img_b: &[u8],
    width: usize,
    height: usize,
) -> Result<f32, ImageCompositorError> {
    validate_rgba_buffer(img_a, width, height)?;
    if img_b.len() != img_a.len() {
        return Err(ImageCompositorError::DimensionMismatch {
            expected_w: width,
            expected_h: height,
            w: width,
            h: img_b.len() / (width * 4).max(1),
        });
    }

    let n_pixels = width * height;
    let mut weighted_mse_sum = 0.0f64;
    let mut weight_sum = 0.0f64;

    for i in 0..n_pixels {
        let base = i * 4;
        let alpha_a = img_a[base + 3] as f64 / 255.0;
        let alpha_b = img_b[base + 3] as f64 / 255.0;
        let weight = (alpha_a + alpha_b) * 0.5;

        for ch in 0..4 {
            let diff = img_a[base + ch] as f64 - img_b[base + ch] as f64;
            weighted_mse_sum += weight * diff * diff;
        }
        weight_sum += weight;
    }

    let mse = if weight_sum < 1e-10 {
        // Fallback: unweighted MSE when both images are transparent
        let sum: f64 = img_a
            .iter()
            .zip(img_b.iter())
            .map(|(&a, &b)| {
                let d = a as f64 - b as f64;
                d * d
            })
            .sum();
        sum / (img_a.len() as f64).max(1.0)
    } else {
        weighted_mse_sum / (weight_sum * 4.0)
    };

    if mse < 1e-12 {
        return Ok(f32::INFINITY);
    }

    let psnr = 10.0 * (255.0f64 * 255.0 / mse).log10();
    Ok(psnr as f32)
}

// ─────────────────────────────────────────────────────────────────────────────
// Statistics
// ─────────────────────────────────────────────────────────────────────────────

/// Compute per-image statistics for a composited RGBA image.
pub fn compute_image_composite_stats(
    pixels: &[u8],
    width: usize,
    height: usize,
) -> Result<ImageCompositeStats, ImageCompositorError> {
    validate_rgba_buffer(pixels, width, height)?;

    let n = width * height;
    let mut sum_r = 0.0f64;
    let mut sum_g = 0.0f64;
    let mut sum_b = 0.0f64;
    let mut sum_a = 0.0f64;
    let mut opaque_count = 0u64;
    let mut transparent_count = 0u64;

    for i in 0..n {
        let base = i * 4;
        let r = pixels[base] as f64;
        let g = pixels[base + 1] as f64;
        let b = pixels[base + 2] as f64;
        let a = pixels[base + 3];

        sum_r += r;
        sum_g += g;
        sum_b += b;
        sum_a += a as f64;

        if a >= 127 {
            opaque_count += 1;
        }
        if a == 0 {
            transparent_count += 1;
        }
    }

    let n_f = n as f64;
    Ok(ImageCompositeStats {
        mean_alpha: (sum_a / (n_f * 255.0)) as f32,
        opaque_fraction: opaque_count as f32 / n as f32,
        transparent_fraction: transparent_count as f32 / n as f32,
        mean_rgb: [
            (sum_r / (n_f * 255.0)) as f32,
            (sum_g / (n_f * 255.0)) as f32,
            (sum_b / (n_f * 255.0)) as f32,
        ],
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── BlendMode::apply_channel ───────────────────────────────────────────

    #[test]
    fn test_blend_normal_returns_src() {
        let v = BlendMode::Normal.apply_channel(0.7, 0.3);
        assert!((v - 0.7).abs() < 1e-6, "Normal should return src");
    }

    #[test]
    fn test_blend_multiply() {
        let v = BlendMode::Multiply.apply_channel(0.5, 0.4);
        assert!((v - 0.2).abs() < 1e-6, "Multiply: 0.5*0.4=0.2");
    }

    #[test]
    fn test_blend_screen() {
        let v = BlendMode::Screen.apply_channel(0.5, 0.5);
        let expected = 1.0 - 0.5 * 0.5;
        assert!((v - expected).abs() < 1e-6, "Screen");
    }

    #[test]
    fn test_blend_add_clamped() {
        let v = BlendMode::Add.apply_channel(0.8, 0.6);
        assert!((v - 1.0).abs() < 1e-6, "Add should clamp at 1.0");
    }

    #[test]
    fn test_blend_add_no_clamp() {
        let v = BlendMode::Add.apply_channel(0.3, 0.2);
        assert!((v - 0.5).abs() < 1e-6, "Add: 0.3+0.2=0.5");
    }

    #[test]
    fn test_blend_difference() {
        let v = BlendMode::Difference.apply_channel(0.8, 0.3);
        assert!((v - 0.5).abs() < 1e-6, "|0.8-0.3|=0.5");
        let v2 = BlendMode::Difference.apply_channel(0.3, 0.8);
        assert!((v2 - 0.5).abs() < 1e-6, "|0.3-0.8|=0.5");
    }

    #[test]
    fn test_blend_darken() {
        let v = BlendMode::Darken.apply_channel(0.3, 0.7);
        assert!((v - 0.3).abs() < 1e-6, "Darken: min(0.3,0.7)=0.3");
    }

    #[test]
    fn test_blend_lighten() {
        let v = BlendMode::Lighten.apply_channel(0.3, 0.7);
        assert!((v - 0.7).abs() < 1e-6, "Lighten: max(0.3,0.7)=0.7");
    }

    #[test]
    fn test_blend_exclusion() {
        let v = BlendMode::Exclusion.apply_channel(0.5, 0.5);
        let expected = 0.5 + 0.5 - 2.0 * 0.5 * 0.5;
        assert!((v - expected).abs() < 1e-6, "Exclusion");
    }

    #[test]
    fn test_blend_subtract_clamped() {
        let v = BlendMode::Subtract.apply_channel(0.8, 0.2);
        // dst - src = 0.2 - 0.8 < 0 → 0
        assert!((v - 0.0).abs() < 1e-6, "Subtract no negatives");
    }

    #[test]
    fn test_blend_color_dodge_edge() {
        // src = 1.0 → denom = 0 → result clamped to 1.0
        let v = BlendMode::ColorDodge.apply_channel(1.0, 0.5);
        assert!((v - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_blend_color_burn_edge() {
        // src = 0 → result = 0
        let v = BlendMode::ColorBurn.apply_channel(0.0, 0.5);
        assert!((v - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_blend_soft_light_formula() {
        let src = 0.5;
        let dst = 0.5;
        let expected = (1.0 - 2.0 * src) * dst * dst + 2.0 * src * dst;
        let v = BlendMode::SoftLight.apply_channel(src, dst);
        assert!((v - expected).abs() < 1e-6, "SoftLight Pegtop formula");
    }

    // ── CompositingLayer ──────────────────────────────────────────────────

    #[test]
    fn test_layer_new_valid() {
        let pixels = vec![0u8; 4 * 4 * 4]; // 4x4 RGBA
        let layer = CompositingLayer::new(pixels, 4, 4);
        assert!(layer.is_ok());
    }

    #[test]
    fn test_layer_new_empty_width() {
        let result = CompositingLayer::new(vec![], 0, 4);
        assert!(matches!(result, Err(ImageCompositorError::EmptyImage)));
    }

    #[test]
    fn test_layer_with_opacity_invalid() {
        let pixels = vec![0u8; 4];
        let layer = CompositingLayer::new(pixels, 1, 1).expect("valid");
        let result = layer.with_opacity(1.5);
        assert!(matches!(
            result,
            Err(ImageCompositorError::InvalidOpacity(_))
        ));
    }

    #[test]
    fn test_layer_with_opacity_valid() {
        let pixels = vec![255u8; 4];
        let layer = CompositingLayer::new(pixels, 1, 1)
            .expect("valid")
            .with_opacity(0.5)
            .expect("valid opacity");
        assert!((layer.opacity - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_layer_pixel_rgba_in_bounds() {
        let mut pixels = vec![0u8; 8 * 8 * 4];
        // Set pixel (2, 3) = (10, 20, 30, 40)
        let idx = (3 * 8 + 2) * 4;
        pixels[idx] = 10;
        pixels[idx + 1] = 20;
        pixels[idx + 2] = 30;
        pixels[idx + 3] = 40;
        let layer = CompositingLayer::new(pixels, 8, 8).expect("valid");
        let px = layer.pixel_rgba(2, 3);
        assert_eq!(px, Some((10, 20, 30, 40)));
    }

    #[test]
    fn test_layer_pixel_rgba_out_of_bounds() {
        let pixels = vec![0u8; 4];
        let layer = CompositingLayer::new(pixels, 1, 1).expect("valid");
        assert_eq!(layer.pixel_rgba(1, 0), None);
        assert_eq!(layer.pixel_rgba(0, 1), None);
    }

    // ── composite_over ────────────────────────────────────────────────────

    #[test]
    fn test_composite_over_zero_alpha_top_unchanged() {
        let bottom = vec![200u8, 100u8, 50u8, 255u8];
        let top = vec![0u8, 0u8, 0u8, 0u8]; // fully transparent top
        let result = composite_over(&bottom, &top, 1, 1, 1.0).expect("ok");
        // Bottom should be unchanged
        assert_eq!(result[0], 200);
        assert_eq!(result[1], 100);
        assert_eq!(result[2], 50);
        assert_eq!(result[3], 255);
    }

    #[test]
    fn test_composite_over_full_alpha_replaces() {
        let bottom = vec![200u8, 100u8, 50u8, 255u8];
        let top = vec![10u8, 20u8, 30u8, 255u8]; // fully opaque top
        let result = composite_over(&bottom, &top, 1, 1, 1.0).expect("ok");
        assert_eq!(result[0], 10);
        assert_eq!(result[1], 20);
        assert_eq!(result[2], 30);
        assert_eq!(result[3], 255);
    }

    #[test]
    fn test_composite_over_opaque_top_is_identity() {
        // Regression: truncating `(v * 255.0) as u8` instead of rounding
        // could quantize an exact channel value like 200 down to 199 (the
        // float round-trip 200/255*255 can land a hair under 200.0). A
        // fully opaque top layer composited "over" anything must
        // reproduce the top layer exactly, for every channel value.
        let bottom = vec![10u8, 20u8, 30u8, 255u8];
        for v in 0u8..=255 {
            let top = vec![v, v, v, 255u8];
            let result = composite_over(&bottom, &top, 1, 1, 1.0).expect("ok");
            assert_eq!(
                result, top,
                "opaque composite_over should reproduce the top layer exactly for v={v}"
            );
        }
    }

    #[test]
    fn test_composite_over_half_alpha_blended() {
        // Both fully opaque bottom (128), top with alpha=128 (~50%)
        let bottom = vec![0u8, 0u8, 0u8, 255u8];
        let top = vec![255u8, 255u8, 255u8, 128u8];
        let result = composite_over(&bottom, &top, 1, 1, 1.0).expect("ok");
        // Result should be roughly half-and-half
        assert!(
            result[0] > 100 && result[0] < 160,
            "blended R={}",
            result[0]
        );
        assert_eq!(result[3], 255); // fully opaque result
    }

    #[test]
    fn test_composite_over_invalid_opacity() {
        let p = vec![0u8; 4];
        let res = composite_over(&p, &p, 1, 1, 1.5);
        assert!(matches!(res, Err(ImageCompositorError::InvalidOpacity(_))));
    }

    // ── blend_pixel ───────────────────────────────────────────────────────

    #[test]
    fn test_blend_pixel_normal() {
        let src = [1.0, 0.0, 0.0];
        let dst = [0.0, 1.0, 0.0];
        let out = blend_pixel(src, dst, BlendMode::Normal);
        // Normal returns src
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[1] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_blend_pixel_multiply_black() {
        let src = [0.5, 0.5, 0.5];
        let dst = [0.0, 0.0, 0.0]; // multiply with black → black
        let out = blend_pixel(src, dst, BlendMode::Multiply);
        assert!((out[0]).abs() < 1e-6);
    }

    #[test]
    fn test_blend_pixel_add_clamp() {
        let src = [1.0, 1.0, 1.0];
        let dst = [1.0, 1.0, 1.0];
        let out = blend_pixel(src, dst, BlendMode::Add);
        assert!((out[0] - 1.0).abs() < 1e-6);
    }

    // ── apply_blend_mode ──────────────────────────────────────────────────

    #[test]
    fn test_apply_blend_mode_normal_same_as_composite_over() {
        let base = vec![50u8, 100u8, 150u8, 255u8];
        let layer = vec![200u8, 150u8, 100u8, 128u8];
        let result_bm = apply_blend_mode(&base, &layer, 1, 1, BlendMode::Normal, 1.0).expect("ok");
        let result_co = composite_over(&base, &layer, 1, 1, 1.0).expect("ok");
        // Should be very close (same algorithm)
        for i in 0..4 {
            let diff = (result_bm[i] as i32 - result_co[i] as i32).abs();
            assert!(
                diff <= 1,
                "channel {i}: bm={} co={}",
                result_bm[i],
                result_co[i]
            );
        }
    }

    #[test]
    fn test_apply_blend_mode_add_bright_image() {
        let base = vec![200u8, 200u8, 200u8, 255u8];
        let layer = vec![200u8, 200u8, 200u8, 255u8];
        let result = apply_blend_mode(&base, &layer, 1, 1, BlendMode::Add, 1.0).expect("ok");
        // Add of two bright images → clamped to 255
        assert_eq!(result[0], 255);
        assert_eq!(result[1], 255);
        assert_eq!(result[2], 255);
    }

    // ── composite_image_layers ────────────────────────────────────────────

    #[test]
    fn test_composite_image_layers_no_layers_error() {
        let config = CompositorConfig::default();
        let result = composite_image_layers(&[], 4, 4, &config);
        assert!(matches!(result, Err(ImageCompositorError::NoLayers)));
    }

    #[test]
    fn test_composite_image_layers_single_layer() {
        let pixels = vec![255u8, 0u8, 0u8, 255u8]; // red opaque
        let layer = CompositingLayer::new(pixels, 1, 1).expect("ok");
        let config = CompositorConfig::default();
        let result = composite_image_layers(&[layer], 1, 1, &config).expect("ok");
        assert_eq!(result[0], 255);
        assert_eq!(result[1], 0);
        assert_eq!(result[2], 0);
        assert_eq!(result[3], 255);
    }

    #[test]
    fn test_composite_image_layers_two_layers() {
        // Bottom: opaque black, Top: opaque white
        let bottom_px = vec![0u8, 0u8, 0u8, 255u8];
        let top_px = vec![255u8, 255u8, 255u8, 255u8];
        let bottom = CompositingLayer::new(bottom_px, 1, 1).expect("ok");
        let top = CompositingLayer::new(top_px, 1, 1).expect("ok");
        let config = CompositorConfig::default();
        let result = composite_image_layers(&[bottom, top], 1, 1, &config).expect("ok");
        // Opaque white on top → white
        assert_eq!(result[0], 255);
    }

    #[test]
    fn test_composite_image_layers_invisible_layer_skipped() {
        let bottom_px = vec![0u8, 0u8, 0u8, 255u8];
        let mut top_px_layer =
            CompositingLayer::new(vec![255u8, 255u8, 255u8, 255u8], 1, 1).expect("ok");
        top_px_layer.visible = false;
        let bottom = CompositingLayer::new(bottom_px, 1, 1).expect("ok");
        let config = CompositorConfig::default();
        let result = composite_image_layers(&[bottom, top_px_layer], 1, 1, &config).expect("ok");
        // Invisible top layer skipped → black remains
        assert_eq!(result[0], 0);
    }

    #[test]
    fn test_composite_image_layers_premultiplied_alpha_respected() {
        // A 50%-alpha white layer over an opaque black background. In
        // *straight* alpha the layer's RGB is (255,255,255) regardless of
        // its alpha; the same visual pixel stored *premultiplied* is
        // (~128,~128,~128) at alpha=128. With `premultiplied_alpha: true`
        // both must composite to (approximately) the same result.
        let bottom_px = vec![0u8, 0u8, 0u8, 255u8];

        let straight_layer =
            CompositingLayer::new(vec![255u8, 255u8, 255u8, 128u8], 1, 1).expect("ok");
        let straight_config = CompositorConfig {
            premultiplied_alpha: false,
            ..Default::default()
        };
        let straight_result = composite_image_layers(
            &[
                CompositingLayer::new(bottom_px.clone(), 1, 1).expect("ok"),
                straight_layer,
            ],
            1,
            1,
            &straight_config,
        )
        .expect("ok");

        let premul_layer =
            CompositingLayer::new(vec![128u8, 128u8, 128u8, 128u8], 1, 1).expect("ok");
        let premul_config = CompositorConfig {
            premultiplied_alpha: true,
            ..Default::default()
        };
        let premul_result = composite_image_layers(
            &[
                CompositingLayer::new(bottom_px, 1, 1).expect("ok"),
                premul_layer,
            ],
            1,
            1,
            &premul_config,
        )
        .expect("ok");

        for i in 0..3 {
            let diff = (straight_result[i] as i32 - premul_result[i] as i32).abs();
            assert!(
                diff <= 3,
                "channel {i}: straight={} premul={}",
                straight_result[i],
                premul_result[i]
            );
        }
    }

    #[test]
    fn test_composite_image_layers_premultiplied_without_flag_is_darker() {
        // Feeding premultiplied data through *without* setting the flag
        // must double-apply the alpha multiply and produce a visibly
        // darker (wrong) result than declaring the flag correctly.
        let bottom_px = vec![0u8, 0u8, 0u8, 255u8];
        let premul_pixels = vec![128u8, 128u8, 128u8, 128u8];

        let wrong_config = CompositorConfig {
            premultiplied_alpha: false, // WRONG: data is actually premultiplied
            ..Default::default()
        };
        let wrong_result = composite_image_layers(
            &[
                CompositingLayer::new(bottom_px.clone(), 1, 1).expect("ok"),
                CompositingLayer::new(premul_pixels.clone(), 1, 1).expect("ok"),
            ],
            1,
            1,
            &wrong_config,
        )
        .expect("ok");

        let right_config = CompositorConfig {
            premultiplied_alpha: true,
            ..Default::default()
        };
        let right_result = composite_image_layers(
            &[
                CompositingLayer::new(bottom_px, 1, 1).expect("ok"),
                CompositingLayer::new(premul_pixels, 1, 1).expect("ok"),
            ],
            1,
            1,
            &right_config,
        )
        .expect("ok");

        assert!(
            wrong_result[0] < right_result[0],
            "mis-handled premultiplied data should be darker: wrong={} right={}",
            wrong_result[0],
            right_result[0]
        );
    }

    // ── premultiply_alpha / unpremultiply_alpha ───────────────────────────

    #[test]
    fn test_premultiply_alpha_roundtrip() {
        let original = vec![200u8, 100u8, 50u8, 128u8];
        let mut pixels = original.clone();
        premultiply_alpha(&mut pixels).expect("ok");
        unpremultiply_alpha(&mut pixels).expect("ok");
        // Round-trip should be close (±2 due to integer rounding)
        for (i, (&pix, &orig)) in pixels.iter().zip(original.iter()).enumerate().take(3) {
            let diff = (pix as i32 - orig as i32).abs();
            assert!(diff <= 3, "channel {i}: got {pix} expected {orig}",);
        }
        assert_eq!(pixels[3], original[3]); // alpha unchanged
    }

    #[test]
    fn test_premultiply_alpha_zero_alpha() {
        let mut pixels = vec![200u8, 100u8, 50u8, 0u8];
        premultiply_alpha(&mut pixels).expect("ok");
        assert_eq!(pixels[0], 0);
        assert_eq!(pixels[1], 0);
        assert_eq!(pixels[2], 0);
        assert_eq!(pixels[3], 0);
    }

    #[test]
    fn test_unpremultiply_alpha_zero_stays_black() {
        let mut pixels = vec![50u8, 50u8, 50u8, 0u8];
        unpremultiply_alpha(&mut pixels).expect("ok");
        assert_eq!(pixels[0], 0);
        assert_eq!(pixels[1], 0);
        assert_eq!(pixels[2], 0);
    }

    // ── solid_color ───────────────────────────────────────────────────────

    #[test]
    fn test_solid_color_size() {
        let pixels = solid_color(3, 5, 10, 20, 30, 255);
        assert_eq!(pixels.len(), 3 * 5 * 4);
    }

    #[test]
    fn test_solid_color_values() {
        let pixels = solid_color(2, 2, 10, 20, 30, 200);
        for i in 0..4 {
            assert_eq!(pixels[i * 4], 10);
            assert_eq!(pixels[i * 4 + 1], 20);
            assert_eq!(pixels[i * 4 + 2], 30);
            assert_eq!(pixels[i * 4 + 3], 200);
        }
    }

    // ── transparency_checkerboard ─────────────────────────────────────────

    #[test]
    fn test_checkerboard_correct_size() {
        let pixels = transparency_checkerboard(8, 8, 4);
        assert_eq!(pixels.len(), 8 * 8 * 4);
    }

    #[test]
    fn test_checkerboard_two_distinct_colors() {
        let pixels = transparency_checkerboard(8, 8, 4);
        // First pixel (0,0) tile_x=0 tile_y=0 → light (192)
        assert_eq!(pixels[0], 192);
        // Pixel (4,0) tile_x=1 tile_y=0 → dark (128)
        let off = 4 * 4;
        assert_eq!(pixels[off], 128);
    }

    // ── extract_alpha / replace_alpha ─────────────────────────────────────

    #[test]
    fn test_extract_alpha_correct_values() {
        let pixels = vec![255u8, 0u8, 0u8, 200u8, 0u8, 255u8, 0u8, 100u8];
        let alpha = extract_alpha(&pixels).expect("ok");
        assert_eq!(alpha, vec![200u8, 100u8]);
    }

    #[test]
    fn test_replace_alpha_applied() {
        let pixels = vec![100u8, 100u8, 100u8, 255u8];
        let new_alpha = vec![42u8];
        let result = replace_alpha(&pixels, &new_alpha, 1, 1).expect("ok");
        assert_eq!(result[3], 42);
        assert_eq!(result[0], 100); // RGB unchanged
    }

    // ── scale_alpha ───────────────────────────────────────────────────────

    #[test]
    fn test_scale_alpha_factor_one_unchanged() {
        let mut pixels = vec![100u8, 100u8, 100u8, 200u8];
        scale_alpha(&mut pixels, 1.0).expect("ok");
        assert_eq!(pixels[3], 200);
    }

    #[test]
    fn test_scale_alpha_factor_half() {
        let mut pixels = vec![100u8, 100u8, 100u8, 200u8];
        scale_alpha(&mut pixels, 0.5).expect("ok");
        // 200 * 0.5 = 100
        assert_eq!(pixels[3], 100);
    }

    #[test]
    fn test_scale_alpha_factor_zero() {
        let mut pixels = vec![100u8, 100u8, 100u8, 200u8];
        scale_alpha(&mut pixels, 0.0).expect("ok");
        assert_eq!(pixels[3], 0);
    }

    // ── flatten_to_rgb ────────────────────────────────────────────────────

    #[test]
    fn test_flatten_transparent_becomes_background() {
        let pixels = vec![255u8, 0u8, 0u8, 0u8]; // fully transparent red
        let bg = [128u8, 64u8, 32u8];
        let result = flatten_to_rgb(&pixels, bg, 1, 1).expect("ok");
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], 128);
        assert_eq!(result[1], 64);
        assert_eq!(result[2], 32);
    }

    #[test]
    fn test_flatten_opaque_becomes_pixel_color() {
        let pixels = vec![10u8, 20u8, 30u8, 255u8]; // fully opaque
        let bg = [255u8, 255u8, 255u8];
        let result = flatten_to_rgb(&pixels, bg, 1, 1).expect("ok");
        assert_eq!(result[0], 10);
        assert_eq!(result[1], 20);
        assert_eq!(result[2], 30);
    }

    // ── composite_psnr ────────────────────────────────────────────────────

    #[test]
    fn test_composite_psnr_identical_is_infinity() {
        let img = solid_color(4, 4, 128, 128, 128, 255);
        let psnr = composite_psnr(&img, &img, 4, 4).expect("ok");
        assert!(psnr.is_infinite() && psnr > 0.0, "identical images → ∞");
    }

    #[test]
    fn test_composite_psnr_different_is_finite() {
        let img_a = solid_color(4, 4, 0, 0, 0, 255);
        let img_b = solid_color(4, 4, 255, 255, 255, 255);
        let psnr = composite_psnr(&img_a, &img_b, 4, 4).expect("ok");
        assert!(psnr.is_finite(), "different images → finite psnr");
        assert!(psnr > 0.0);
    }

    // ── compute_image_composite_stats ─────────────────────────────────────

    #[test]
    fn test_composite_stats_solid_opaque() {
        let pixels = solid_color(4, 4, 128, 64, 32, 255);
        let stats = compute_image_composite_stats(&pixels, 4, 4).expect("ok");
        assert!((stats.mean_alpha - 1.0).abs() < 0.01, "fully opaque");
        assert!((stats.opaque_fraction - 1.0).abs() < 0.01);
        assert!((stats.transparent_fraction - 0.0).abs() < 0.01);
        let expected_r = 128.0 / 255.0;
        assert!((stats.mean_rgb[0] - expected_r).abs() < 0.01, "mean R");
    }

    #[test]
    fn test_composite_stats_all_transparent() {
        let pixels = solid_color(4, 4, 0, 0, 0, 0);
        let stats = compute_image_composite_stats(&pixels, 4, 4).expect("ok");
        assert!((stats.mean_alpha - 0.0).abs() < 0.01);
        assert!((stats.transparent_fraction - 1.0).abs() < 0.01);
        assert!((stats.opaque_fraction - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_composite_stats_error_on_empty() {
        let result = compute_image_composite_stats(&[], 0, 0);
        assert!(result.is_err());
    }
}
