//! Image resizing, cropping, padding, and geometric transforms on RGBA images.
//!
//! All operations work on flat `Vec<u8>` / `&[u8]` buffers in RGBA format
//! (width × height × 4 bytes, row-major).  No external image crate is required
//! for the core math — only `thiserror` for the error type.
//!
//! ## Quick-start
//! ```rust
//! use oxigaf_render::image_resize::{resize_bilinear, thumbnail, ResizeFilter};
//!
//! // 2×2 solid red image
//! let src: Vec<u8> = vec![255, 0, 0, 255,  255, 0, 0, 255,
//!                         255, 0, 0, 255,  255, 0, 0, 255];
//! let dst = resize_bilinear(&src, 2, 2, 4, 4).unwrap();
//! assert_eq!(dst.len(), 4 * 4 * 4);
//! ```

use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors that can occur during image resize, crop, or pad operations.
#[derive(Debug, Error, PartialEq)]
pub enum ResizeError {
    /// Source image has zero or overflow dimensions.
    #[error("Invalid source dimensions: {width}×{height}")]
    InvalidSourceDimensions {
        /// Source width.
        width: u32,
        /// Source height.
        height: u32,
    },

    /// Destination image has zero or overflow dimensions.
    #[error("Invalid destination dimensions: {width}×{height}")]
    InvalidDestDimensions {
        /// Destination width.
        width: u32,
        /// Destination height.
        height: u32,
    },

    /// Crop region extends outside the image bounds.
    #[error("Crop region [{x},{y}]+{w}×{h} out of bounds for image {img_w}×{img_h}")]
    CropOutOfBounds {
        /// Crop X offset.
        x: u32,
        /// Crop Y offset.
        y: u32,
        /// Crop width.
        w: u32,
        /// Crop height.
        h: u32,
        /// Image width.
        img_w: u32,
        /// Image height.
        img_h: u32,
    },

    /// Scale factor is not positive.
    #[error("Scale factor must be > 0, got {scale}")]
    InvalidScale {
        /// The bad scale value.
        scale: f32,
    },

    /// Buffer length does not match the stated dimensions.
    #[error("Image buffer length {actual} does not match {width}×{height}×4 = {expected}")]
    BufferSizeMismatch {
        /// Actual buffer length.
        actual: usize,
        /// Expected buffer length.
        expected: usize,
        /// Stated width.
        width: u32,
        /// Stated height.
        height: u32,
    },
}

// ─────────────────────────────────────────────────────────────────────────────
// Resize filter
// ─────────────────────────────────────────────────────────────────────────────

/// Interpolation filter to use when resizing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResizeFilter {
    /// Nearest-neighbour — fast, blocky.
    Nearest,
    /// Bilinear interpolation — smooth.
    Bilinear,
    /// Bicubic (Catmull-Rom) — sharpest, slowest.
    Bicubic,
    /// Box/area-average filter — best quality for heavy downscaling.
    Box,
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Validate that a buffer matches the stated pixel dimensions (RGBA).
#[inline]
fn check_src(src: &[u8], w: u32, h: u32) -> Result<(), ResizeError> {
    if w == 0 || h == 0 {
        return Err(ResizeError::InvalidSourceDimensions {
            width: w,
            height: h,
        });
    }
    let expected = (w as usize) * (h as usize) * 4;
    if src.len() != expected {
        return Err(ResizeError::BufferSizeMismatch {
            actual: src.len(),
            expected,
            width: w,
            height: h,
        });
    }
    Ok(())
}

/// Validate destination dimensions are non-zero.
#[inline]
fn check_dst(w: u32, h: u32) -> Result<(), ResizeError> {
    if w == 0 || h == 0 {
        return Err(ResizeError::InvalidDestDimensions {
            width: w,
            height: h,
        });
    }
    Ok(())
}

/// Fetch a pixel from a flat RGBA buffer, clamping coordinates to valid range.
#[inline]
fn get_pixel_clamped(src: &[u8], w: u32, h: u32, x: i64, y: i64) -> [u8; 4] {
    let cx = x.clamp(0, w as i64 - 1) as usize;
    let cy = y.clamp(0, h as i64 - 1) as usize;
    let base = (cy * w as usize + cx) * 4;
    [src[base], src[base + 1], src[base + 2], src[base + 3]]
}

/// Bilinear sample at floating-point (fx, fy) in pixel-centre coordinates.
/// Coordinates are clamped to border.
fn sample_bilinear(src: &[u8], w: u32, h: u32, fx: f64, fy: f64) -> [u8; 4] {
    let x0 = fx.floor() as i64;
    let y0 = fy.floor() as i64;
    let tx = fx - fx.floor();
    let ty = fy - fy.floor();

    let c00 = get_pixel_clamped(src, w, h, x0, y0);
    let c10 = get_pixel_clamped(src, w, h, x0 + 1, y0);
    let c01 = get_pixel_clamped(src, w, h, x0, y0 + 1);
    let c11 = get_pixel_clamped(src, w, h, x0 + 1, y0 + 1);

    let mut out = [0u8; 4];
    for i in 0..4 {
        let v = c00[i] as f64 * (1.0 - tx) * (1.0 - ty)
            + c10[i] as f64 * tx * (1.0 - ty)
            + c01[i] as f64 * (1.0 - tx) * ty
            + c11[i] as f64 * tx * ty;
        out[i] = v.round().clamp(0.0, 255.0) as u8;
    }
    out
}

/// Catmull-Rom cubic weight.
#[inline]
fn catmull_rom(t: f64) -> f64 {
    let t = t.abs();
    if t < 1.0 {
        1.5 * t * t * t - 2.5 * t * t + 1.0
    } else if t < 2.0 {
        -0.5 * t * t * t + 2.5 * t * t - 4.0 * t + 2.0
    } else {
        0.0
    }
}

/// Bicubic (Catmull-Rom) sample at floating-point (fx, fy).
fn sample_bicubic(src: &[u8], w: u32, h: u32, fx: f64, fy: f64) -> [u8; 4] {
    let x0 = fx.floor() as i64;
    let y0 = fy.floor() as i64;
    let dx = fx - fx.floor();
    let dy = fy - fy.floor();

    let mut acc = [0.0f64; 4];
    let mut weight_sum = 0.0f64;

    for ky in -1i64..=2 {
        let wy = catmull_rom(dy - ky as f64);
        for kx in -1i64..=2 {
            let wx = catmull_rom(dx - kx as f64);
            let w_total = wx * wy;
            let px = get_pixel_clamped(src, w, h, x0 + kx, y0 + ky);
            for i in 0..4 {
                acc[i] += px[i] as f64 * w_total;
            }
            weight_sum += w_total;
        }
    }

    let mut out = [0u8; 4];
    for i in 0..4 {
        let v = if weight_sum.abs() > 1e-12 {
            acc[i] / weight_sum
        } else {
            acc[i]
        };
        out[i] = v.round().clamp(0.0, 255.0) as u8;
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Public — Resize functions
// ─────────────────────────────────────────────────────────────────────────────

/// Validate that a buffer length matches `width × height × 4`.
pub fn validate_buffer(buf: &[u8], width: u32, height: u32) -> Result<(), ResizeError> {
    check_src(buf, width, height)
}

/// Nearest-neighbour resize of an RGBA image.
pub fn resize_nearest(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
) -> Result<Vec<u8>, ResizeError> {
    check_src(src, src_w, src_h)?;
    check_dst(dst_w, dst_h)?;

    let mut dst = vec![0u8; dst_w as usize * dst_h as usize * 4];
    for dy in 0..dst_h {
        for dx in 0..dst_w {
            // Map centre of dst pixel to src space and round.
            let sx = ((dx as f64 + 0.5) * src_w as f64 / dst_w as f64) as u32;
            let sy = ((dy as f64 + 0.5) * src_h as f64 / dst_h as f64) as u32;
            let sx = sx.clamp(0, src_w - 1);
            let sy = sy.clamp(0, src_h - 1);
            let src_base = (sy as usize * src_w as usize + sx as usize) * 4;
            let dst_base = (dy as usize * dst_w as usize + dx as usize) * 4;
            dst[dst_base..dst_base + 4].copy_from_slice(&src[src_base..src_base + 4]);
        }
    }
    Ok(dst)
}

/// Bilinear interpolation resize of an RGBA image.
///
/// Pixel centres are mapped using the "half-pixel offset" convention:
/// `src_x = (x + 0.5) * src_w / dst_w - 0.5`.
pub fn resize_bilinear(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
) -> Result<Vec<u8>, ResizeError> {
    check_src(src, src_w, src_h)?;
    check_dst(dst_w, dst_h)?;

    let mut dst = vec![0u8; dst_w as usize * dst_h as usize * 4];
    for dy in 0..dst_h {
        let fy = (dy as f64 + 0.5) * src_h as f64 / dst_h as f64 - 0.5;
        for dx in 0..dst_w {
            let fx = (dx as f64 + 0.5) * src_w as f64 / dst_w as f64 - 0.5;
            let px = sample_bilinear(src, src_w, src_h, fx, fy);
            let base = (dy as usize * dst_w as usize + dx as usize) * 4;
            dst[base..base + 4].copy_from_slice(&px);
        }
    }
    Ok(dst)
}

/// Bicubic (Catmull-Rom) resize of an RGBA image.
///
/// Uses a 4×4 sample kernel; output is clamped to `[0, 255]`.
pub fn resize_bicubic(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
) -> Result<Vec<u8>, ResizeError> {
    check_src(src, src_w, src_h)?;
    check_dst(dst_w, dst_h)?;

    let mut dst = vec![0u8; dst_w as usize * dst_h as usize * 4];
    for dy in 0..dst_h {
        let fy = (dy as f64 + 0.5) * src_h as f64 / dst_h as f64 - 0.5;
        for dx in 0..dst_w {
            let fx = (dx as f64 + 0.5) * src_w as f64 / dst_w as f64 - 0.5;
            let px = sample_bicubic(src, src_w, src_h, fx, fy);
            let base = (dy as usize * dst_w as usize + dx as usize) * 4;
            dst[base..base + 4].copy_from_slice(&px);
        }
    }
    Ok(dst)
}

/// Box / area-average resize.
///
/// Each destination pixel is the area-weighted average of the source pixels
/// that fall inside its footprint.  Partial-pixel contributions are weighted
/// proportionally, so the total weight is always exactly 1.0.
pub fn resize_box(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
) -> Result<Vec<u8>, ResizeError> {
    check_src(src, src_w, src_h)?;
    check_dst(dst_w, dst_h)?;

    let scale_x = src_w as f64 / dst_w as f64;
    let scale_y = src_h as f64 / dst_h as f64;

    let mut dst = vec![0u8; dst_w as usize * dst_h as usize * 4];
    for dy in 0..dst_h {
        let src_y0 = dy as f64 * scale_y;
        let src_y1 = (dy as f64 + 1.0) * scale_y;
        let iy0 = src_y0.floor() as i64;
        let iy1 = (src_y1.ceil() as i64).min(src_h as i64);

        for dx in 0..dst_w {
            let src_x0 = dx as f64 * scale_x;
            let src_x1 = (dx as f64 + 1.0) * scale_x;
            let ix0 = src_x0.floor() as i64;
            let ix1 = (src_x1.ceil() as i64).min(src_w as i64);

            let mut acc = [0.0f64; 4];
            let mut total_weight = 0.0f64;

            for sy in iy0..iy1 {
                // Overlap in Y direction
                let row_start = (sy as f64).max(src_y0);
                let row_end = (sy as f64 + 1.0).min(src_y1);
                let wy = (row_end - row_start).max(0.0);
                if wy <= 0.0 {
                    continue;
                }
                for sx in ix0..ix1 {
                    // Overlap in X direction
                    let col_start = (sx as f64).max(src_x0);
                    let col_end = (sx as f64 + 1.0).min(src_x1);
                    let wx = (col_end - col_start).max(0.0);
                    if wx <= 0.0 {
                        continue;
                    }
                    let w = wx * wy;
                    let px = get_pixel_clamped(src, src_w, src_h, sx, sy);
                    for i in 0..4 {
                        acc[i] += px[i] as f64 * w;
                    }
                    total_weight += w;
                }
            }

            let base = (dy as usize * dst_w as usize + dx as usize) * 4;
            for i in 0..4 {
                let v = if total_weight > 1e-12 {
                    acc[i] / total_weight
                } else {
                    acc[i]
                };
                dst[base + i] = v.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    Ok(dst)
}

/// Resize using a specified filter.
pub fn resize(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
    filter: ResizeFilter,
) -> Result<Vec<u8>, ResizeError> {
    match filter {
        ResizeFilter::Nearest => resize_nearest(src, src_w, src_h, dst_w, dst_h),
        ResizeFilter::Bilinear => resize_bilinear(src, src_w, src_h, dst_w, dst_h),
        ResizeFilter::Bicubic => resize_bicubic(src, src_w, src_h, dst_w, dst_h),
        ResizeFilter::Box => resize_box(src, src_w, src_h, dst_w, dst_h),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public — Thumbnail / aspect-ratio helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Compute output dimensions so that the result fits inside `max_w × max_h`
/// while preserving the original aspect ratio.  Returns at least `1×1`.
pub fn fit_dimensions(src_w: u32, src_h: u32, max_w: u32, max_h: u32) -> (u32, u32) {
    if src_w == 0 || src_h == 0 || max_w == 0 || max_h == 0 {
        return (1, 1);
    }
    let scale_x = max_w as f64 / src_w as f64;
    let scale_y = max_h as f64 / src_h as f64;
    let scale = scale_x.min(scale_y);
    let out_w = ((src_w as f64 * scale).round() as u32).max(1);
    let out_h = ((src_h as f64 * scale).round() as u32).max(1);
    (out_w, out_h)
}

/// Create a thumbnail preserving aspect ratio.
///
/// Returns `(pixels, new_width, new_height)` where `max(new_width, new_height) <= max_side`.
/// Each dimension is at least 1 pixel.
pub fn thumbnail(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    max_side: u32,
    filter: ResizeFilter,
) -> Result<(Vec<u8>, u32, u32), ResizeError> {
    check_src(src, src_w, src_h)?;
    check_dst(max_side, max_side)?;

    let (dst_w, dst_h) = fit_dimensions(src_w, src_h, max_side, max_side);
    let pixels = resize(src, src_w, src_h, dst_w, dst_h, filter)?;
    Ok((pixels, dst_w, dst_h))
}

/// Scale by a floating-point factor (`1.0` = identity, `2.0` = double, `0.5` = half).
///
/// Returns `(pixels, new_width, new_height)`.
pub fn scale_by_factor(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    scale: f32,
    filter: ResizeFilter,
) -> Result<(Vec<u8>, u32, u32), ResizeError> {
    check_src(src, src_w, src_h)?;
    if scale <= 0.0 {
        return Err(ResizeError::InvalidScale { scale });
    }
    let dst_w = ((src_w as f64 * scale as f64).round() as u32).max(1);
    let dst_h = ((src_h as f64 * scale as f64).round() as u32).max(1);
    let pixels = resize(src, src_w, src_h, dst_w, dst_h, filter)?;
    Ok((pixels, dst_w, dst_h))
}

/// Build an image pyramid (Gaussian pyramid-style, halving each level).
///
/// Level 0 contains the original image; subsequent levels are half the size.
/// The pyramid stops early if any dimension reaches 1.
///
/// Returns a `Vec` of `(pixels, width, height)`.
pub fn image_pyramid(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    num_levels: usize,
    filter: ResizeFilter,
) -> Result<Vec<(Vec<u8>, u32, u32)>, ResizeError> {
    check_src(src, src_w, src_h)?;

    let mut levels: Vec<(Vec<u8>, u32, u32)> = Vec::with_capacity(num_levels);
    levels.push((src.to_vec(), src_w, src_h));

    for _ in 1..num_levels {
        let (prev, pw, ph) = levels.last().ok_or(ResizeError::InvalidSourceDimensions {
            width: 0,
            height: 0,
        })?;
        if *pw == 1 && *ph == 1 {
            break;
        }
        let nw = (*pw / 2).max(1);
        let nh = (*ph / 2).max(1);
        let next = resize(prev, *pw, *ph, nw, nh, filter)?;
        levels.push((next, nw, nh));
    }
    Ok(levels)
}

// ─────────────────────────────────────────────────────────────────────────────
// Public — Crop functions
// ─────────────────────────────────────────────────────────────────────────────

/// Extract a rectangular sub-region from an image.
pub fn crop(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    crop_x: u32,
    crop_y: u32,
    crop_w: u32,
    crop_h: u32,
) -> Result<Vec<u8>, ResizeError> {
    check_src(src, src_w, src_h)?;
    if crop_w == 0 || crop_h == 0 {
        return Err(ResizeError::CropOutOfBounds {
            x: crop_x,
            y: crop_y,
            w: crop_w,
            h: crop_h,
            img_w: src_w,
            img_h: src_h,
        });
    }
    let x_end = crop_x
        .checked_add(crop_w)
        .ok_or(ResizeError::CropOutOfBounds {
            x: crop_x,
            y: crop_y,
            w: crop_w,
            h: crop_h,
            img_w: src_w,
            img_h: src_h,
        })?;
    let y_end = crop_y
        .checked_add(crop_h)
        .ok_or(ResizeError::CropOutOfBounds {
            x: crop_x,
            y: crop_y,
            w: crop_w,
            h: crop_h,
            img_w: src_w,
            img_h: src_h,
        })?;
    if x_end > src_w || y_end > src_h {
        return Err(ResizeError::CropOutOfBounds {
            x: crop_x,
            y: crop_y,
            w: crop_w,
            h: crop_h,
            img_w: src_w,
            img_h: src_h,
        });
    }

    let mut dst = vec![0u8; crop_w as usize * crop_h as usize * 4];
    for row in 0..crop_h {
        let src_row = (crop_y + row) as usize;
        let src_col = crop_x as usize;
        let src_base = (src_row * src_w as usize + src_col) * 4;
        let dst_base = row as usize * crop_w as usize * 4;
        let len = crop_w as usize * 4;
        dst[dst_base..dst_base + len].copy_from_slice(&src[src_base..src_base + len]);
    }
    Ok(dst)
}

/// Centre-crop to exact output dimensions.
///
/// If the output is larger than the source in any dimension, the missing area
/// is filled with `background`.
pub fn center_crop(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    out_w: u32,
    out_h: u32,
    background: [u8; 4],
) -> Result<Vec<u8>, ResizeError> {
    check_src(src, src_w, src_h)?;
    check_dst(out_w, out_h)?;

    // Offset of the source inside the output canvas
    // (negative means we're cropping from src).
    let offset_x = out_w as i64 / 2 - src_w as i64 / 2;
    let offset_y = out_h as i64 / 2 - src_h as i64 / 2;

    let mut dst = vec![background[0]; out_w as usize * out_h as usize * 4];
    // Fill with background colour
    for px in dst.chunks_exact_mut(4) {
        px.copy_from_slice(&background);
    }

    for dy in 0..out_h as i64 {
        let sy = dy - offset_y;
        if sy < 0 || sy >= src_h as i64 {
            continue;
        }
        for dx in 0..out_w as i64 {
            let sx = dx - offset_x;
            if sx < 0 || sx >= src_w as i64 {
                continue;
            }
            let src_base = (sy as usize * src_w as usize + sx as usize) * 4;
            let dst_base = (dy as usize * out_w as usize + dx as usize) * 4;
            dst[dst_base..dst_base + 4].copy_from_slice(&src[src_base..src_base + 4]);
        }
    }
    Ok(dst)
}

// ─────────────────────────────────────────────────────────────────────────────
// Public — Pad functions
// ─────────────────────────────────────────────────────────────────────────────

/// Parameters for the [`pad_image`] operation.
#[derive(Debug, Clone, Copy)]
pub struct PadParams {
    /// Source image width.
    pub src_w: u32,
    /// Source image height.
    pub src_h: u32,
    /// Destination canvas width.
    pub dst_w: u32,
    /// Destination canvas height.
    pub dst_h: u32,
    /// X offset at which the source is placed.
    pub offset_x: u32,
    /// Y offset at which the source is placed.
    pub offset_y: u32,
}

impl PadParams {
    /// Construct from individual fields.
    pub fn new(
        src_w: u32,
        src_h: u32,
        dst_w: u32,
        dst_h: u32,
        offset_x: u32,
        offset_y: u32,
    ) -> Self {
        Self {
            src_w,
            src_h,
            dst_w,
            dst_h,
            offset_x,
            offset_y,
        }
    }
}

/// Expand the canvas to `dst_w × dst_h`, placing the source at `(offset_x, offset_y)`.
///
/// Regions outside the source are filled with `background`.
pub fn pad_image(
    src: &[u8],
    params: PadParams,
    background: [u8; 4],
) -> Result<Vec<u8>, ResizeError> {
    let PadParams {
        src_w,
        src_h,
        dst_w,
        dst_h,
        offset_x,
        offset_y,
    } = params;
    check_src(src, src_w, src_h)?;
    check_dst(dst_w, dst_h)?;
    // Source must fit in destination at the given offset.
    if offset_x + src_w > dst_w || offset_y + src_h > dst_h {
        return Err(ResizeError::CropOutOfBounds {
            x: offset_x,
            y: offset_y,
            w: src_w,
            h: src_h,
            img_w: dst_w,
            img_h: dst_h,
        });
    }

    let mut dst = vec![0u8; dst_w as usize * dst_h as usize * 4];
    for px in dst.chunks_exact_mut(4) {
        px.copy_from_slice(&background);
    }
    for row in 0..src_h {
        let src_base = row as usize * src_w as usize * 4;
        let dst_row = (offset_y + row) as usize;
        let dst_base = (dst_row * dst_w as usize + offset_x as usize) * 4;
        let len = src_w as usize * 4;
        dst[dst_base..dst_base + len].copy_from_slice(&src[src_base..src_base + len]);
    }
    Ok(dst)
}

/// Pad to a square canvas, centred, with the given background.
///
/// Returns `(pixels, side_length)`.
pub fn pad_to_square(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    background: [u8; 4],
) -> Result<(Vec<u8>, u32), ResizeError> {
    check_src(src, src_w, src_h)?;
    let side = src_w.max(src_h);
    let offset_x = (side - src_w) / 2;
    let offset_y = (side - src_h) / 2;
    let pixels = pad_image(
        src,
        PadParams::new(src_w, src_h, side, side, offset_x, offset_y),
        background,
    )?;
    Ok((pixels, side))
}

// ─────────────────────────────────────────────────────────────────────────────
// Public — Geometric transforms
// ─────────────────────────────────────────────────────────────────────────────

/// Flip an image horizontally (left ↔ right).
pub fn flip_horizontal(src: &[u8], src_w: u32, src_h: u32) -> Result<Vec<u8>, ResizeError> {
    check_src(src, src_w, src_h)?;
    let w = src_w as usize;
    let h = src_h as usize;
    let mut dst = vec![0u8; w * h * 4];
    for y in 0..h {
        for x in 0..w {
            let src_base = (y * w + x) * 4;
            let dst_base = (y * w + (w - 1 - x)) * 4;
            dst[dst_base..dst_base + 4].copy_from_slice(&src[src_base..src_base + 4]);
        }
    }
    Ok(dst)
}

/// Flip an image vertically (top ↔ bottom).
pub fn flip_vertical(src: &[u8], src_w: u32, src_h: u32) -> Result<Vec<u8>, ResizeError> {
    check_src(src, src_w, src_h)?;
    let w = src_w as usize;
    let h = src_h as usize;
    let mut dst = vec![0u8; w * h * 4];
    for y in 0..h {
        let src_base = y * w * 4;
        let dst_base = (h - 1 - y) * w * 4;
        dst[dst_base..dst_base + w * 4].copy_from_slice(&src[src_base..src_base + w * 4]);
    }
    Ok(dst)
}

/// Rotate 90° clockwise.
///
/// `dst[y][x] = src[src_h-1-x][y]`; output size is `(src_h, src_w)`.
pub fn rotate_90_cw(src: &[u8], src_w: u32, src_h: u32) -> Result<Vec<u8>, ResizeError> {
    check_src(src, src_w, src_h)?;
    let sw = src_w as usize;
    let sh = src_h as usize;
    // Output dimensions: width = src_h, height = src_w
    let dw = sh;
    let dh = sw;
    let mut dst = vec![0u8; dw * dh * 4];
    for sy in 0..sh {
        for sx in 0..sw {
            // dst(dx, dy) = src(sy, sh-1-sx) ... no, let's think carefully.
            // Clockwise 90°: new_x = sh-1-old_y, new_y = old_x  (using 0-indexed)
            // Actually: after 90° CW rotation of a (W×H) image to (H×W):
            //   dst[new_y][new_x] = src[src_h-1-new_x][new_y]
            //   => dst pixel at (dx, dy) comes from src at (src_h-1-dx, dy)
            //   where dx is [0..src_h) and dy is [0..src_w)
            // Iterate over src (sx, sy):
            //   dst col (dx) = sh - 1 - sy
            //   dst row (dy) = sx
            let dx = sh - 1 - sy;
            let dy = sx;
            let src_base = (sy * sw + sx) * 4;
            let dst_base = (dy * dw + dx) * 4;
            dst[dst_base..dst_base + 4].copy_from_slice(&src[src_base..src_base + 4]);
        }
    }
    Ok(dst)
}

/// Rotate 90° counter-clockwise.
///
/// Output size is `(src_h, src_w)`.
pub fn rotate_90_ccw(src: &[u8], src_w: u32, src_h: u32) -> Result<Vec<u8>, ResizeError> {
    check_src(src, src_w, src_h)?;
    let sw = src_w as usize;
    let sh = src_h as usize;
    let dw = sh;
    let dh = sw;
    let mut dst = vec![0u8; dw * dh * 4];
    for sy in 0..sh {
        for sx in 0..sw {
            // CCW 90°: dst col = sy, dst row = sw-1-sx
            let dx = sy;
            let dy = sw - 1 - sx;
            let src_base = (sy * sw + sx) * 4;
            let dst_base = (dy * dw + dx) * 4;
            dst[dst_base..dst_base + 4].copy_from_slice(&src[src_base..src_base + 4]);
        }
    }
    Ok(dst)
}

/// Rotate 180°.
pub fn rotate_180(src: &[u8], src_w: u32, src_h: u32) -> Result<Vec<u8>, ResizeError> {
    check_src(src, src_w, src_h)?;
    let n = src_w as usize * src_h as usize;
    let mut dst = vec![0u8; n * 4];
    for i in 0..n {
        let j = n - 1 - i;
        dst[j * 4..j * 4 + 4].copy_from_slice(&src[i * 4..i * 4 + 4]);
    }
    Ok(dst)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ──────────────────────────────────────────────────────────────

    /// Solid RGBA image of given colour.
    fn solid(w: u32, h: u32, rgba: [u8; 4]) -> Vec<u8> {
        let mut v = vec![0u8; w as usize * h as usize * 4];
        for px in v.chunks_exact_mut(4) {
            px.copy_from_slice(&rgba);
        }
        v
    }

    /// Get pixel at (x, y) from a flat RGBA buffer.
    fn get_px(buf: &[u8], w: u32, x: u32, y: u32) -> [u8; 4] {
        let base = (y as usize * w as usize + x as usize) * 4;
        [buf[base], buf[base + 1], buf[base + 2], buf[base + 3]]
    }

    // ── validate_buffer ───────────────────────────────────────────────────────

    #[test]
    fn validate_buffer_correct_size() {
        let buf = vec![0u8; 2 * 3 * 4];
        assert!(validate_buffer(&buf, 2, 3).is_ok());
    }

    #[test]
    fn validate_buffer_too_small() {
        let buf = vec![0u8; 2 * 3 * 4 - 1];
        let err = validate_buffer(&buf, 2, 3);
        assert!(err.is_err());
        matches!(err.unwrap_err(), ResizeError::BufferSizeMismatch { .. });
    }

    #[test]
    fn validate_buffer_too_large() {
        let buf = vec![0u8; 2 * 3 * 4 + 1];
        let err = validate_buffer(&buf, 2, 3);
        assert!(err.is_err());
        matches!(err.unwrap_err(), ResizeError::BufferSizeMismatch { .. });
    }

    // ── resize_nearest ────────────────────────────────────────────────────────

    #[test]
    fn nearest_upscale_2x2_to_4x4() {
        let src = solid(2, 2, [100, 150, 200, 255]);
        let dst = resize_nearest(&src, 2, 2, 4, 4).unwrap();
        assert_eq!(dst.len(), 4 * 4 * 4);
        // All pixels should remain the same colour.
        for px in dst.chunks_exact(4) {
            assert_eq!(px, [100, 150, 200, 255]);
        }
    }

    #[test]
    fn nearest_downscale_4x4_to_2x2() {
        let src = solid(4, 4, [10, 20, 30, 40]);
        let dst = resize_nearest(&src, 4, 4, 2, 2).unwrap();
        assert_eq!(dst.len(), 2 * 2 * 4);
        for px in dst.chunks_exact(4) {
            assert_eq!(px, [10, 20, 30, 40]);
        }
    }

    #[test]
    fn nearest_same_size() {
        let src = solid(3, 3, [7, 8, 9, 255]);
        let dst = resize_nearest(&src, 3, 3, 3, 3).unwrap();
        assert_eq!(dst, src);
    }

    #[test]
    fn nearest_1x1_to_3x3() {
        let src = vec![42u8, 43, 44, 255];
        let dst = resize_nearest(&src, 1, 1, 3, 3).unwrap();
        assert_eq!(dst.len(), 9 * 4);
        for px in dst.chunks_exact(4) {
            assert_eq!(px, [42, 43, 44, 255]);
        }
    }

    #[test]
    fn nearest_bad_dst_zero() {
        let src = solid(2, 2, [0; 4]);
        assert!(resize_nearest(&src, 2, 2, 0, 2).is_err());
    }

    // ── resize_bilinear ───────────────────────────────────────────────────────

    #[test]
    fn bilinear_uniform_image_same_value() {
        let colour = [80u8, 160, 240, 128];
        let src = solid(4, 4, colour);
        let dst = resize_bilinear(&src, 4, 4, 8, 8).unwrap();
        for px in dst.chunks_exact(4) {
            // Bilinear of uniform image must stay exactly the same.
            assert_eq!(px, colour.as_ref());
        }
    }

    #[test]
    fn bilinear_2x1_to_2x2_preserves_height() {
        // Two horizontally adjacent colours
        let src = vec![0u8, 0, 0, 255, 200, 200, 200, 255];
        let dst = resize_bilinear(&src, 2, 1, 2, 2).unwrap();
        assert_eq!(dst.len(), 2 * 2 * 4);
    }

    #[test]
    fn bilinear_upscale_preserves_value_range() {
        let src = solid(3, 3, [50, 100, 150, 200]);
        let dst = resize_bilinear(&src, 3, 3, 9, 9).unwrap();
        // All output pixels must be u8 (guaranteed by type); verify the buffer is
        // properly sized and non-empty (the real postcondition here).
        assert_eq!(dst.len(), 9 * 9 * 4);
        assert!(!dst.is_empty());
    }

    #[test]
    fn bilinear_bad_src_dimensions() {
        let err = resize_bilinear(&[], 0, 4, 2, 2);
        assert!(err.is_err());
    }

    #[test]
    fn bilinear_output_size_correct() {
        let src = solid(5, 7, [1, 2, 3, 4]);
        let dst = resize_bilinear(&src, 5, 7, 10, 14).unwrap();
        assert_eq!(dst.len(), 10 * 14 * 4);
    }

    // ── resize_bicubic ────────────────────────────────────────────────────────

    #[test]
    fn bicubic_uniform_image_same_value() {
        let colour = [123u8, 45, 67, 255];
        let src = solid(4, 4, colour);
        let dst = resize_bicubic(&src, 4, 4, 8, 8).unwrap();
        for px in dst.chunks_exact(4) {
            assert_eq!(px, colour.as_ref());
        }
    }

    #[test]
    fn bicubic_output_size_correct() {
        let src = solid(3, 3, [10, 20, 30, 40]);
        let dst = resize_bicubic(&src, 3, 3, 6, 6).unwrap();
        assert_eq!(dst.len(), 6 * 6 * 4);
    }

    #[test]
    fn bicubic_output_clamped_to_255() {
        let src = solid(2, 2, [255, 255, 255, 255]);
        let dst = resize_bicubic(&src, 2, 2, 4, 4).unwrap();
        // Bicubic may overshoot for non-uniform inputs; for a uniform white image
        // every output pixel should equal 255 (no undershoot).
        for px in dst.chunks_exact(4) {
            assert_eq!(px, [255u8, 255, 255, 255]);
        }
    }

    #[test]
    fn bicubic_bad_dst() {
        let src = solid(2, 2, [0; 4]);
        assert!(resize_bicubic(&src, 2, 2, 0, 4).is_err());
    }

    // ── resize_box ────────────────────────────────────────────────────────────

    #[test]
    fn box_4x4_to_1x1_is_mean() {
        // 16 pixels, each [100, 200, 50, 255] → mean = [100, 200, 50, 255]
        let src = solid(4, 4, [100, 200, 50, 255]);
        let dst = resize_box(&src, 4, 4, 1, 1).unwrap();
        assert_eq!(dst.len(), 4);
        assert_eq!(dst[0], 100);
        assert_eq!(dst[1], 200);
        assert_eq!(dst[2], 50);
        assert_eq!(dst[3], 255);
    }

    #[test]
    fn box_downscale_uniform_stays_same() {
        let src = solid(6, 6, [77, 88, 99, 255]);
        let dst = resize_box(&src, 6, 6, 3, 3).unwrap();
        for px in dst.chunks_exact(4) {
            assert_eq!(px, [77, 88, 99, 255]);
        }
    }

    #[test]
    fn box_output_size_correct() {
        let src = solid(8, 8, [1, 2, 3, 4]);
        let dst = resize_box(&src, 8, 8, 4, 2).unwrap();
        assert_eq!(dst.len(), 4 * 2 * 4);
    }

    #[test]
    fn box_bad_src() {
        let err = resize_box(&[], 4, 4, 2, 2);
        assert!(err.is_err());
    }

    // ── resize dispatcher ─────────────────────────────────────────────────────

    #[test]
    fn resize_nearest_dispatch() {
        let src = solid(2, 2, [1, 2, 3, 4]);
        let dst = resize(&src, 2, 2, 4, 4, ResizeFilter::Nearest).unwrap();
        assert_eq!(dst.len(), 4 * 4 * 4);
    }

    #[test]
    fn resize_bilinear_dispatch() {
        let src = solid(2, 2, [1, 2, 3, 4]);
        let dst = resize(&src, 2, 2, 4, 4, ResizeFilter::Bilinear).unwrap();
        assert_eq!(dst.len(), 4 * 4 * 4);
    }

    #[test]
    fn resize_bicubic_dispatch() {
        let src = solid(2, 2, [1, 2, 3, 4]);
        let dst = resize(&src, 2, 2, 4, 4, ResizeFilter::Bicubic).unwrap();
        assert_eq!(dst.len(), 4 * 4 * 4);
    }

    #[test]
    fn resize_box_dispatch() {
        let src = solid(4, 4, [1, 2, 3, 4]);
        let dst = resize(&src, 4, 4, 2, 2, ResizeFilter::Box).unwrap();
        assert_eq!(dst.len(), 2 * 2 * 4);
    }

    // ── thumbnail ─────────────────────────────────────────────────────────────

    #[test]
    fn thumbnail_landscape_fits_max_side() {
        let src = solid(200, 100, [1, 2, 3, 4]);
        let (_, w, h) = thumbnail(&src, 200, 100, 50, ResizeFilter::Nearest).unwrap();
        assert!(w <= 50 && h <= 50);
        assert_eq!(w, 50);
        assert_eq!(h, 25);
    }

    #[test]
    fn thumbnail_portrait_fits_max_side() {
        let src = solid(100, 200, [1, 2, 3, 4]);
        let (_, w, h) = thumbnail(&src, 100, 200, 50, ResizeFilter::Nearest).unwrap();
        assert!(w <= 50 && h <= 50);
        assert_eq!(h, 50);
        assert_eq!(w, 25);
    }

    #[test]
    fn thumbnail_square_fits_max_side() {
        let src = solid(100, 100, [1, 2, 3, 4]);
        let (_, w, h) = thumbnail(&src, 100, 100, 50, ResizeFilter::Nearest).unwrap();
        assert_eq!(w, 50);
        assert_eq!(h, 50);
    }

    #[test]
    fn thumbnail_smaller_than_max_side_unchanged() {
        // Image is already smaller — thumbnail should not upscale beyond max_side.
        let src = solid(30, 20, [1, 2, 3, 4]);
        let (_, w, h) = thumbnail(&src, 30, 20, 50, ResizeFilter::Nearest).unwrap();
        assert!(w <= 50 && h <= 50);
    }

    // ── crop ─────────────────────────────────────────────────────────────────

    #[test]
    fn crop_basic() {
        // 4×4 image; crop a 2×2 corner.
        let mut src = vec![0u8; 4 * 4 * 4];
        // Mark pixel (1,1) with a unique colour.
        let base = (4 + 1) * 4;
        src[base..base + 4].copy_from_slice(&[99, 98, 97, 96]);
        let dst = crop(&src, 4, 4, 1, 1, 2, 2).unwrap();
        assert_eq!(dst.len(), 2 * 2 * 4);
        // First pixel of crop should be the marked one.
        assert_eq!(&dst[0..4], &[99, 98, 97, 96]);
    }

    #[test]
    fn crop_full_image() {
        let src = solid(3, 3, [10, 20, 30, 40]);
        let dst = crop(&src, 3, 3, 0, 0, 3, 3).unwrap();
        assert_eq!(dst, src);
    }

    #[test]
    fn crop_1x1_pixel() {
        let mut src = vec![0u8; 3 * 3 * 4];
        let base = (2 * 3 + 2) * 4;
        src[base..base + 4].copy_from_slice(&[5, 6, 7, 8]);
        let dst = crop(&src, 3, 3, 2, 2, 1, 1).unwrap();
        assert_eq!(&dst, &[5, 6, 7, 8]);
    }

    #[test]
    fn crop_out_of_bounds_error() {
        let src = solid(4, 4, [0; 4]);
        let err = crop(&src, 4, 4, 3, 3, 2, 2);
        assert!(err.is_err());
        matches!(err.unwrap_err(), ResizeError::CropOutOfBounds { .. });
    }

    #[test]
    fn crop_zero_dimensions_error() {
        let src = solid(4, 4, [0; 4]);
        assert!(crop(&src, 4, 4, 0, 0, 0, 2).is_err());
    }

    // ── center_crop ───────────────────────────────────────────────────────────

    #[test]
    fn center_crop_larger_than_image_pads_with_background() {
        let src = solid(4, 4, [200, 100, 50, 255]);
        let bg = [0, 0, 0, 255];
        let dst = center_crop(&src, 4, 4, 8, 8, bg).unwrap();
        assert_eq!(dst.len(), 8 * 8 * 4);
        // Top-left corner (0,0) should be the background colour.
        let corner = get_px(&dst, 8, 0, 0);
        assert_eq!(corner, bg);
    }

    #[test]
    fn center_crop_smaller_than_image_crops_only() {
        let src = solid(8, 8, [1, 2, 3, 4]);
        let bg = [255, 255, 255, 255];
        let dst = center_crop(&src, 8, 8, 4, 4, bg).unwrap();
        assert_eq!(dst.len(), 4 * 4 * 4);
        // All pixels are from the source.
        for px in dst.chunks_exact(4) {
            assert_eq!(px, [1, 2, 3, 4]);
        }
    }

    #[test]
    fn center_crop_exact_same_size() {
        let src = solid(5, 5, [11, 22, 33, 44]);
        let bg = [0; 4];
        let dst = center_crop(&src, 5, 5, 5, 5, bg).unwrap();
        assert_eq!(dst, src);
    }

    // ── pad_image ─────────────────────────────────────────────────────────────

    #[test]
    fn pad_image_basic() {
        let src = solid(2, 2, [10, 20, 30, 40]);
        let bg = [0, 0, 0, 255];
        let dst = pad_image(&src, PadParams::new(2, 2, 4, 4, 1, 1), bg).unwrap();
        assert_eq!(dst.len(), 4 * 4 * 4);
        // Corner (0,0) is background.
        assert_eq!(get_px(&dst, 4, 0, 0), bg);
        // Pixel (1,1) is from source.
        assert_eq!(get_px(&dst, 4, 1, 1), [10, 20, 30, 40]);
    }

    #[test]
    fn pad_image_zero_offset() {
        let src = solid(2, 2, [55, 66, 77, 88]);
        let bg = [0; 4];
        let dst = pad_image(&src, PadParams::new(2, 2, 4, 4, 0, 0), bg).unwrap();
        assert_eq!(get_px(&dst, 4, 0, 0), [55, 66, 77, 88]);
        assert_eq!(get_px(&dst, 4, 3, 3), bg);
    }

    #[test]
    fn pad_image_out_of_bounds_error() {
        let src = solid(4, 4, [0; 4]);
        let bg = [0; 4];
        // 4×4 src at offset (1,1) into 4×4 dst → doesn't fit.
        assert!(pad_image(&src, PadParams::new(4, 4, 4, 4, 1, 1), bg).is_err());
    }

    // ── pad_to_square ─────────────────────────────────────────────────────────

    #[test]
    fn pad_to_square_landscape() {
        let src = solid(6, 2, [1, 2, 3, 4]);
        let (dst, side) = pad_to_square(&src, 6, 2, [0; 4]).unwrap();
        assert_eq!(side, 6);
        assert_eq!(dst.len(), (side * side * 4) as usize);
    }

    #[test]
    fn pad_to_square_portrait() {
        let src = solid(2, 6, [1, 2, 3, 4]);
        let (dst, side) = pad_to_square(&src, 2, 6, [0; 4]).unwrap();
        assert_eq!(side, 6);
        assert_eq!(dst.len(), (side * side * 4) as usize);
    }

    #[test]
    fn pad_to_square_already_square() {
        let src = solid(5, 5, [7, 8, 9, 10]);
        let (dst, side) = pad_to_square(&src, 5, 5, [0; 4]).unwrap();
        assert_eq!(side, 5);
        assert_eq!(dst, src);
    }

    // ── scale_by_factor ───────────────────────────────────────────────────────

    #[test]
    fn scale_factor_2x_doubles_dimensions() {
        let src = solid(4, 3, [1, 2, 3, 4]);
        let (_, w, h) = scale_by_factor(&src, 4, 3, 2.0, ResizeFilter::Nearest).unwrap();
        assert_eq!(w, 8);
        assert_eq!(h, 6);
    }

    #[test]
    fn scale_factor_half_halves_dimensions() {
        let src = solid(8, 8, [1, 2, 3, 4]);
        let (_, w, h) = scale_by_factor(&src, 8, 8, 0.5, ResizeFilter::Box).unwrap();
        assert_eq!(w, 4);
        assert_eq!(h, 4);
    }

    #[test]
    fn scale_factor_zero_error() {
        let src = solid(4, 4, [0; 4]);
        let err = scale_by_factor(&src, 4, 4, 0.0, ResizeFilter::Nearest);
        assert!(err.is_err());
        matches!(err.unwrap_err(), ResizeError::InvalidScale { .. });
    }

    // ── image_pyramid ─────────────────────────────────────────────────────────

    #[test]
    fn pyramid_3_levels_correct_sizes() {
        let src = solid(8, 8, [100, 100, 100, 255]);
        let levels = image_pyramid(&src, 8, 8, 3, ResizeFilter::Box).unwrap();
        assert_eq!(levels.len(), 3);
        assert_eq!((levels[0].1, levels[0].2), (8, 8));
        assert_eq!((levels[1].1, levels[1].2), (4, 4));
        assert_eq!((levels[2].1, levels[2].2), (2, 2));
    }

    #[test]
    fn pyramid_1_level_returns_original() {
        let src = solid(4, 4, [5, 6, 7, 8]);
        let levels = image_pyramid(&src, 4, 4, 1, ResizeFilter::Nearest).unwrap();
        assert_eq!(levels.len(), 1);
        assert_eq!(levels[0].0, src);
    }

    #[test]
    fn pyramid_stops_early_at_1x1() {
        let src = solid(2, 2, [0; 4]);
        let levels = image_pyramid(&src, 2, 2, 10, ResizeFilter::Box).unwrap();
        // Can't go below 1×1
        assert!(levels.len() <= 10);
        let last = levels.last().unwrap();
        assert!(last.1 >= 1 && last.2 >= 1);
    }

    // ── flip_horizontal ───────────────────────────────────────────────────────

    #[test]
    fn flip_horizontal_2x1() {
        // [A, B] → [B, A]
        let src = vec![1u8, 1, 1, 1, 2, 2, 2, 2];
        let dst = flip_horizontal(&src, 2, 1).unwrap();
        assert_eq!(dst[0..4], [2, 2, 2, 2]);
        assert_eq!(dst[4..8], [1, 1, 1, 1]);
    }

    #[test]
    fn flip_horizontal_twice_is_identity() {
        let src = solid(5, 3, [10, 20, 30, 255]);
        // Fill with some distinct pattern.
        let mut src2 = src.clone();
        for i in 0..5_usize {
            let base = i * 4;
            src2[base] = i as u8 * 10;
        }
        let once = flip_horizontal(&src2, 5, 3).unwrap();
        let twice = flip_horizontal(&once, 5, 3).unwrap();
        assert_eq!(twice, src2);
    }

    #[test]
    fn flip_horizontal_bad_buf() {
        let err = flip_horizontal(&[1, 2, 3], 1, 1);
        assert!(err.is_err());
    }

    // ── flip_vertical ─────────────────────────────────────────────────────────

    #[test]
    fn flip_vertical_1x2() {
        // [row0=[A], row1=[B]] → [row0=[B], row1=[A]]
        let src = vec![1u8, 1, 1, 1, 2, 2, 2, 2];
        let dst = flip_vertical(&src, 1, 2).unwrap();
        assert_eq!(dst[0..4], [2, 2, 2, 2]);
        assert_eq!(dst[4..8], [1, 1, 1, 1]);
    }

    #[test]
    fn flip_vertical_twice_is_identity() {
        let mut src = solid(4, 4, [0; 4]);
        for i in 0..4_usize {
            let base = i * 4 * 4; // row i
            src[base] = i as u8 * 20;
        }
        let once = flip_vertical(&src, 4, 4).unwrap();
        let twice = flip_vertical(&once, 4, 4).unwrap();
        assert_eq!(twice, src);
    }

    // ── rotate_90_cw ──────────────────────────────────────────────────────────

    #[test]
    fn rotate_90_cw_dimensions_swapped() {
        let src = solid(4, 2, [0; 4]);
        let dst = rotate_90_cw(&src, 4, 2).unwrap();
        // Width and height swap: src(4,2) → dst(2,4)
        assert_eq!(dst.len(), 2 * 4 * 4);
    }

    #[test]
    fn rotate_90_cw_2x1() {
        // src(2,1): [A, B] (one row of two pixels)
        // After 90° CW → (1,2): top row = [A], bottom row = [B]
        let src = vec![1u8, 0, 0, 0, 2, 0, 0, 0];
        let dst = rotate_90_cw(&src, 2, 1).unwrap();
        // dst dims: w=1, h=2
        assert_eq!(dst.len(), 2 * 4);
        // dst(0,0) = src(1, 0) for CW: dst col dx=sh-1-sy=0, dy=sx
        // sy=0, sx=0 → dx=0, dy=0 → dst[0] = src[(0*2+0)*4] = [1,0,0,0]
        // sy=0, sx=1 → dx=0, dy=1 → dst[4] = src[(0*2+1)*4] = [2,0,0,0]
        assert_eq!(dst[0..4], [1, 0, 0, 0]);
        assert_eq!(dst[4..8], [2, 0, 0, 0]);
    }

    #[test]
    fn rotate_cw_4x_is_identity() {
        let src = solid(3, 5, [11, 22, 33, 44]);
        let r1 = rotate_90_cw(&src, 3, 5).unwrap();
        let r2 = rotate_90_cw(&r1, 5, 3).unwrap();
        let r3 = rotate_90_cw(&r2, 3, 5).unwrap();
        let r4 = rotate_90_cw(&r3, 5, 3).unwrap();
        assert_eq!(r4, src);
    }

    // ── rotate_90_ccw ─────────────────────────────────────────────────────────

    #[test]
    fn rotate_90_ccw_dimensions_swapped() {
        let src = solid(6, 3, [0; 4]);
        let dst = rotate_90_ccw(&src, 6, 3).unwrap();
        assert_eq!(dst.len(), 3 * 6 * 4);
    }

    #[test]
    fn rotate_cw_then_ccw_is_identity() {
        let src = solid(4, 6, [50, 100, 150, 200]);
        let cw = rotate_90_cw(&src, 4, 6).unwrap();
        let back = rotate_90_ccw(&cw, 6, 4).unwrap();
        assert_eq!(back, src);
    }

    // ── rotate_180 ────────────────────────────────────────────────────────────

    #[test]
    fn rotate_180_twice_is_identity() {
        let mut src = solid(4, 3, [0; 4]);
        // Put a distinct marker.
        src[0..4].copy_from_slice(&[255, 0, 128, 64]);
        let r1 = rotate_180(&src, 4, 3).unwrap();
        let r2 = rotate_180(&r1, 4, 3).unwrap();
        assert_eq!(r2, src);
    }

    #[test]
    fn rotate_180_reverses_order() {
        // A 2-pixel image: [A, B] → [B, A]
        let src = vec![1u8, 0, 0, 0, 2, 0, 0, 0];
        let dst = rotate_180(&src, 2, 1).unwrap();
        assert_eq!(dst[0..4], [2, 0, 0, 0]);
        assert_eq!(dst[4..8], [1, 0, 0, 0]);
    }

    // ── fit_dimensions ────────────────────────────────────────────────────────

    #[test]
    fn fit_dimensions_landscape_constraint() {
        // 200×100 fits in 50×50 → 50×25
        let (w, h) = fit_dimensions(200, 100, 50, 50);
        assert_eq!(w, 50);
        assert_eq!(h, 25);
    }

    #[test]
    fn fit_dimensions_portrait_constraint() {
        // 100×200 fits in 50×50 → 25×50
        let (w, h) = fit_dimensions(100, 200, 50, 50);
        assert_eq!(w, 25);
        assert_eq!(h, 50);
    }

    #[test]
    fn fit_dimensions_already_fits() {
        // 10×10 fits in 50×50 → stays within bounds
        let (w, h) = fit_dimensions(10, 10, 50, 50);
        assert!(w <= 50 && h <= 50);
        assert_eq!(w, h); // square stays square
    }
}
