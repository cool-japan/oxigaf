//! Variable-resolution support for the multi-view diffusion pipeline.
//!
//! The diffusion model can operate at 128×128, 256×256, or 512×512 input
//! resolution. The VAE uses a stride of 8, so latent sizes are 16, 32, and 64
//! respectively.
//!
//! # Coordinate conventions
//! - Image pixel layout: row-major HWC (height × width × channels) in [0, 1].
//! - Resize/crop always produce f32 data in the same layout.

use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// Enum: SupportedResolution
// ─────────────────────────────────────────────────────────────────────────────

/// The three image resolutions supported by the multi-view diffusion pipeline.
///
/// The VAE down-samples by 8×, so each resolution maps to a distinct latent
/// spatial size.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SupportedResolution {
    /// 128 × 128 pixels, 16 × 16 latent grid.
    R128,
    /// 256 × 256 pixels, 32 × 32 latent grid.
    R256,
    /// 512 × 512 pixels, 64 × 64 latent grid.
    R512,
}

impl SupportedResolution {
    /// Returns the square image side length in pixels (128, 256, or 512).
    #[inline]
    pub fn image_size(self) -> usize {
        match self {
            Self::R128 => 128,
            Self::R256 => 256,
            Self::R512 => 512,
        }
    }

    /// Returns the VAE latent spatial size (image_size / 8): 16, 32, or 64.
    #[inline]
    pub fn latent_size(self) -> usize {
        self.image_size() / 8
    }

    /// Returns the total number of pixels in one channel (`image_size²`).
    #[inline]
    pub fn pixel_count(self) -> usize {
        let s = self.image_size();
        s * s
    }

    /// Maps a pixel-side length to the corresponding variant, or `None` if the
    /// value is not 128, 256, or 512.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxigaf_diffusion::resolution::SupportedResolution;
    ///
    /// assert_eq!(SupportedResolution::from_image_size(256), Some(SupportedResolution::R256));
    /// assert_eq!(SupportedResolution::from_image_size(200), None);
    /// ```
    #[inline]
    pub fn from_image_size(n: usize) -> Option<Self> {
        match n {
            128 => Some(Self::R128),
            256 => Some(Self::R256),
            512 => Some(Self::R512),
            _ => None,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors arising from resolution-sensitive image operations.
#[derive(Debug, Error)]
pub enum ResolutionError {
    /// Flat buffer length does not match the expected `w × h × c` footprint.
    #[error("Image buffer length {got} doesn't match {expected} for {w}×{h}×{c}")]
    BufferSizeMismatch {
        got: usize,
        expected: usize,
        w: usize,
        h: usize,
        c: usize,
    },

    /// At least one image dimension is zero, which is invalid.
    #[error("Zero dimension: width={w} height={h}")]
    ZeroDimension { w: usize, h: usize },

    /// The requested crop exceeds the source image boundaries.
    #[error("Crop {cw}×{ch} exceeds source {w}×{h}")]
    CropTooLarge {
        w: usize,
        h: usize,
        cw: usize,
        ch: usize,
    },

    /// Channel count is not 1, 3, or 4.
    #[error("Channels must be 1, 3, or 4; got {0}")]
    UnsupportedChannels(usize),
}

// ─────────────────────────────────────────────────────────────────────────────
// Validation helpers (private)
// ─────────────────────────────────────────────────────────────────────────────

/// Validate that `channels` is one of 1, 3, or 4.
#[inline]
fn validate_channels(channels: usize) -> Result<(), ResolutionError> {
    match channels {
        1 | 3 | 4 => Ok(()),
        c => Err(ResolutionError::UnsupportedChannels(c)),
    }
}

/// Validate that neither `w` nor `h` is zero.
#[inline]
fn validate_dims(w: usize, h: usize) -> Result<(), ResolutionError> {
    if w == 0 || h == 0 {
        return Err(ResolutionError::ZeroDimension { w, h });
    }
    Ok(())
}

/// Validate that a flat buffer has exactly `w * h * c` elements.
#[inline]
fn validate_buffer(img: &[f32], w: usize, h: usize, c: usize) -> Result<(), ResolutionError> {
    let expected = w * h * c;
    if img.len() != expected {
        return Err(ResolutionError::BufferSizeMismatch {
            got: img.len(),
            expected,
            w,
            h,
            c,
        });
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Bilinear resize
// ─────────────────────────────────────────────────────────────────────────────

/// Resize a row-major **HWC** f32 image using bilinear interpolation.
///
/// # Arguments
///
/// * `img` — flat f32 buffer, row-major HWC format, values nominally in [0, 1]
/// * `in_w`, `in_h` — source width and height
/// * `out_w`, `out_h` — target width and height
/// * `channels` — number of channels (1, 3, or 4)
///
/// # Coordinate convention
///
/// Uses `scale_x = in_w / out_w`, `scale_y = in_h / out_h`.
/// For each output pixel `(ox, oy)`:
/// ```text
/// src_x = (ox + 0.5) * scale_x - 0.5   (align-corners=false, matches PIL/CV2)
/// src_y = (oy + 0.5) * scale_y - 0.5
/// ```
/// Both coordinates are clamped to `[0, in_{dim}-1]` before interpolation.
///
/// The four-point bilinear blend for floor coordinates `(x0, y0)` and ceil
/// coordinates `(x1, y1)` with fractional offsets `(tx, ty)`:
/// ```text
/// (1-tx)*(1-ty)*p00 + tx*(1-ty)*p10 + (1-tx)*ty*p01 + tx*ty*p11
/// ```
///
/// # Errors
///
/// Returns [`ResolutionError::ZeroDimension`] when any output dimension is 0,
/// [`ResolutionError::UnsupportedChannels`] when `channels ∉ {1,3,4}`, and
/// [`ResolutionError::BufferSizeMismatch`] when `img.len() ≠ in_w * in_h * channels`.
pub fn resize_image_bilinear(
    img: &[f32],
    in_w: usize,
    in_h: usize,
    out_w: usize,
    out_h: usize,
    channels: usize,
) -> Result<Vec<f32>, ResolutionError> {
    // ---- validate ----
    validate_channels(channels)?;
    validate_dims(in_w, in_h)?;
    validate_dims(out_w, out_h)?;
    validate_buffer(img, in_w, in_h, channels)?;

    // Precompute scale factors (align-corners = false)
    let scale_x = in_w as f32 / out_w as f32;
    let scale_y = in_h as f32 / out_h as f32;

    let in_w_f = in_w as f32;
    let in_h_f = in_h as f32;

    let mut out = vec![0.0_f32; out_w * out_h * channels];

    for oy in 0..out_h {
        // Continuous source coordinate along y (align-corners=false)
        let src_y_raw = (oy as f32 + 0.5) * scale_y - 0.5;
        let src_y = src_y_raw.max(0.0).min(in_h_f - 1.0);

        let y0 = src_y.floor() as usize;
        let y1 = (y0 + 1).min(in_h - 1);
        let ty = src_y - src_y.floor();
        let ty_inv = 1.0 - ty;

        for ox in 0..out_w {
            // Continuous source coordinate along x
            let src_x_raw = (ox as f32 + 0.5) * scale_x - 0.5;
            let src_x = src_x_raw.max(0.0).min(in_w_f - 1.0);

            let x0 = src_x.floor() as usize;
            let x1 = (x0 + 1).min(in_w - 1);
            let tx = src_x - src_x.floor();
            let tx_inv = 1.0 - tx;

            // Base byte offsets for the four surrounding pixels (HWC layout)
            let base_00 = (y0 * in_w + x0) * channels;
            let base_10 = (y0 * in_w + x1) * channels;
            let base_01 = (y1 * in_w + x0) * channels;
            let base_11 = (y1 * in_w + x1) * channels;

            // Blend weights
            let w00 = tx_inv * ty_inv;
            let w10 = tx * ty_inv;
            let w01 = tx_inv * ty;
            let w11 = tx * ty;

            let out_base = (oy * out_w + ox) * channels;

            for c in 0..channels {
                let val = w00 * img[base_00 + c]
                    + w10 * img[base_10 + c]
                    + w01 * img[base_01 + c]
                    + w11 * img[base_11 + c];
                out[out_base + c] = val;
            }
        }
    }

    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// Center crop
// ─────────────────────────────────────────────────────────────────────────────

/// Crop `cw×ch` pixels from the center of a `w×h` HWC f32 image.
///
/// The output buffer is a contiguous HWC slice of the same channel count.
///
/// # Errors
///
/// Returns [`ResolutionError::ZeroDimension`] when any dimension is 0,
/// [`ResolutionError::UnsupportedChannels`] when `channels ∉ {1,3,4}`,
/// [`ResolutionError::CropTooLarge`] when the crop exceeds the source, and
/// [`ResolutionError::BufferSizeMismatch`] when `img.len() ≠ w * h * channels`.
pub fn crop_center(
    img: &[f32],
    w: usize,
    h: usize,
    cw: usize,
    ch: usize,
    channels: usize,
) -> Result<Vec<f32>, ResolutionError> {
    validate_channels(channels)?;
    validate_dims(w, h)?;
    validate_dims(cw, ch)?;
    validate_buffer(img, w, h, channels)?;

    if cw > w || ch > h {
        return Err(ResolutionError::CropTooLarge { w, h, cw, ch });
    }

    // Integer center offset: (w - cw) / 2 rounds towards zero, which is the
    // standard "align-to-floor" convention used by PIL and torchvision.
    let x_off = (w - cw) / 2;
    let y_off = (h - ch) / 2;

    let mut out = vec![0.0_f32; cw * ch * channels];

    for row in 0..ch {
        let src_row = y_off + row;
        let src_row_base = (src_row * w + x_off) * channels;
        let dst_row_base = row * cw * channels;
        let row_len = cw * channels;
        out[dst_row_base..dst_row_base + row_len]
            .copy_from_slice(&img[src_row_base..src_row_base + row_len]);
    }

    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// Letterbox pad
// ─────────────────────────────────────────────────────────────────────────────

/// Scale an image to fit inside `target_size × target_size`, then pad to a
/// square with a constant fill value.
///
/// The aspect ratio of the source image is preserved.  Padding is applied
/// symmetrically (or as symmetrically as integer arithmetic allows for odd
/// residuals) on the axis that is shorter after scaling.
///
/// # Arguments
///
/// * `img` — flat f32 buffer, row-major HWC format
/// * `in_w`, `in_h` — source width and height
/// * `target_size` — side length of the square output
/// * `channels` — number of channels (1, 3, or 4)
/// * `fill` — constant value used for padding pixels
///
/// # Output
///
/// Always returns a buffer of length `target_size × target_size × channels`.
///
/// # Errors
///
/// Propagates all errors from [`resize_image_bilinear`].
pub fn letterbox_pad(
    img: &[f32],
    in_w: usize,
    in_h: usize,
    target_size: usize,
    channels: usize,
    fill: f32,
) -> Result<Vec<f32>, ResolutionError> {
    validate_channels(channels)?;
    validate_dims(in_w, in_h)?;
    validate_dims(target_size, target_size)?;
    validate_buffer(img, in_w, in_h, channels)?;

    // Determine the largest scale factor that keeps the image within the square.
    // Use integer-safe comparison to avoid f32 precision drift on exact squares.
    let (scaled_w, scaled_h) = if in_w * target_size <= in_h * target_size {
        // Height is the limiting dimension  (or square input)
        let sw = (in_w as f64 * target_size as f64 / in_h as f64).round() as usize;
        let sw = sw.min(target_size).max(1);
        (sw, target_size)
    } else {
        // Width is the limiting dimension
        let sh = (in_h as f64 * target_size as f64 / in_w as f64).round() as usize;
        let sh = sh.min(target_size).max(1);
        (target_size, sh)
    };

    // Bilinear-resize the image to the scaled dimensions.
    let resized = resize_image_bilinear(img, in_w, in_h, scaled_w, scaled_h, channels)?;

    // Allocate the output padded with `fill`.
    let out_len = target_size * target_size * channels;
    let mut out = vec![fill; out_len];

    // Compute top-left corner of where the resized image is placed.
    let pad_x = (target_size - scaled_w) / 2;
    let pad_y = (target_size - scaled_h) / 2;

    for row in 0..scaled_h {
        let src_base = row * scaled_w * channels;
        let dst_row = pad_y + row;
        let dst_base = (dst_row * target_size + pad_x) * channels;
        let row_len = scaled_w * channels;
        out[dst_base..dst_base + row_len].copy_from_slice(&resized[src_base..src_base + row_len]);
    }

    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// DiffusionConfig factory helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Build a [`crate::config::DiffusionConfig`] tuned for the given resolution.
///
/// Only `image_size`, `latent_size`, and `unet_in_channels` are overridden
/// relative to the [`Default`] implementation; all other fields retain their
/// default values so caller code remains forward-compatible.
///
/// # Channel accounting
///
/// The U-Net receives a concatenation of the noisy latent (4 ch) and the
/// normal-map latent (4 ch), giving `unet_in_channels = 8` for all resolutions.
/// This function keeps that value unchanged; it is listed explicitly so that
/// the relationship to `latent_channels` remains visible and auditable.
pub fn diffusion_config_for_resolution(res: SupportedResolution) -> crate::config::DiffusionConfig {
    let mut cfg = crate::config::DiffusionConfig::default();
    cfg.image_size = res.image_size();
    cfg.latent_size = res.latent_size();
    // 4-channel noisy latent + 4-channel normal-map latent = 8 input channels.
    cfg.unet_in_channels = 2 * cfg.latent_channels;
    cfg
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── SupportedResolution ───────────────────────────────────────────────────

    #[test]
    fn test_supported_resolution_image_sizes() {
        assert_eq!(SupportedResolution::R128.image_size(), 128);
        assert_eq!(SupportedResolution::R256.image_size(), 256);
        assert_eq!(SupportedResolution::R512.image_size(), 512);
    }

    #[test]
    fn test_supported_resolution_latent_sizes() {
        assert_eq!(SupportedResolution::R128.latent_size(), 16);
        assert_eq!(SupportedResolution::R256.latent_size(), 32);
        assert_eq!(SupportedResolution::R512.latent_size(), 64);
    }

    #[test]
    fn test_supported_resolution_from_image_size() {
        assert_eq!(
            SupportedResolution::from_image_size(128),
            Some(SupportedResolution::R128)
        );
        assert_eq!(
            SupportedResolution::from_image_size(256),
            Some(SupportedResolution::R256)
        );
        assert_eq!(
            SupportedResolution::from_image_size(512),
            Some(SupportedResolution::R512)
        );
        assert_eq!(SupportedResolution::from_image_size(200), None);
        assert_eq!(SupportedResolution::from_image_size(0), None);
        assert_eq!(SupportedResolution::from_image_size(1024), None);
    }

    #[test]
    fn test_pixel_count() {
        assert_eq!(SupportedResolution::R128.pixel_count(), 128 * 128);
        assert_eq!(SupportedResolution::R256.pixel_count(), 256 * 256);
        assert_eq!(SupportedResolution::R512.pixel_count(), 512 * 512);
    }

    // ── Bilinear resize ───────────────────────────────────────────────────────

    #[test]
    fn test_resize_identity() {
        // A 3×2 RGB image resized to itself must return an identical buffer.
        let img: Vec<f32> = (0..18).map(|v| v as f32 / 17.0).collect();
        let out = resize_image_bilinear(&img, 3, 2, 3, 2, 3).unwrap();
        assert_eq!(out.len(), img.len());
        for (i, (&a, &b)) in img.iter().zip(out.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "identity mismatch at index {i}: {a} vs {b}"
            );
        }
    }

    #[test]
    fn test_resize_2x_upscale() {
        // 2×2 single-channel image upscaled to 4×4.
        // Known pixel values at corners:
        //   (0,0)=0.0  (1,0)=1.0
        //   (0,1)=0.5  (1,1)=0.75
        //
        // With align-corners=false, the mapping for ox=0:
        //   src_x = (0 + 0.5) * 0.5 - 0.5 = -0.25 → clamped to 0.0
        // For ox=1:
        //   src_x = (1 + 0.5) * 0.5 - 0.5 = 0.25 → between 0 and 1, tx=0.25
        // For ox=2:
        //   src_x = (2 + 0.5) * 0.5 - 0.5 = 0.75 → tx=0.75
        // For ox=3:
        //   src_x = (3 + 0.5) * 0.5 - 0.5 = 1.25 → clamped to 1.0
        #[rustfmt::skip]
        let img: Vec<f32> = vec![
            0.0, 1.0,
            0.5, 0.75,
        ];
        let out = resize_image_bilinear(&img, 2, 2, 4, 4, 1).unwrap();
        assert_eq!(out.len(), 16);

        // Corner pixels of the output should equal the corresponding input corners.
        // Top-left output pixel maps to input (0,0) exactly.
        assert!((out[0] - 0.0).abs() < 1e-5, "TL corner: got {}", out[0]);
        // Top-right output pixel maps to input (1,0) exactly.
        assert!((out[3] - 1.0).abs() < 1e-5, "TR corner: got {}", out[3]);
        // Bottom-left maps to input (0,1).
        assert!((out[12] - 0.5).abs() < 1e-5, "BL corner: got {}", out[12]);
        // Bottom-right maps to input (1,1).
        assert!((out[15] - 0.75).abs() < 1e-5, "BR corner: got {}", out[15]);

        // Pixel at (1, 0): src_x=0.25, src_y=-0.25→0.  Bilinear between p[0]=0.0 and p[1]=1.0.
        let expected_01 = 0.0 * (1.0 - 0.25) + 1.0 * 0.25; // 0.25
        assert!(
            (out[1] - expected_01).abs() < 1e-5,
            "out[1]: expected {expected_01} got {}",
            out[1]
        );
    }

    #[test]
    fn test_resize_2x_downscale() {
        // 4×4 single-channel image of all-ones downscaled to 2×2 → all ones.
        let img = vec![1.0_f32; 16];
        let out = resize_image_bilinear(&img, 4, 4, 2, 2, 1).unwrap();
        assert_eq!(out.len(), 4);
        for &v in &out {
            assert!((v - 1.0).abs() < 1e-6, "expected 1.0, got {v}");
        }
    }

    #[test]
    fn test_resize_non_square() {
        // 4×2 (wide) single-channel image resized to 2×4 (tall).
        // Fill with a gradient so we can verify interpolation is happening.
        let img: Vec<f32> = (0..8).map(|v| v as f32).collect();
        let out = resize_image_bilinear(&img, 4, 2, 2, 4, 1).unwrap();
        // Output must have exactly 2×4×1 = 8 elements.
        assert_eq!(out.len(), 8);
        // All values must be within the input range [0, 7].
        for &v in &out {
            assert!((0.0..=7.0).contains(&v), "value out of range: {v}");
        }
    }

    #[test]
    fn test_resize_single_pixel() {
        // A 1×1 source pixel expanded to 3×3: all output pixels must equal the
        // source value because there are no neighbours to interpolate with.
        let img = vec![0.42_f32];
        let out = resize_image_bilinear(&img, 1, 1, 3, 3, 1).unwrap();
        assert_eq!(out.len(), 9);
        for &v in &out {
            assert!(
                (v - 0.42).abs() < 1e-6,
                "single-pixel expand: expected 0.42, got {v}"
            );
        }
    }

    #[test]
    fn test_resize_zero_dim_error() {
        let img = vec![0.0_f32; 4];
        let err = resize_image_bilinear(&img, 2, 2, 0, 2, 1).unwrap_err();
        assert!(
            matches!(err, ResolutionError::ZeroDimension { .. }),
            "expected ZeroDimension, got {err:?}"
        );
        let err2 = resize_image_bilinear(&img, 2, 2, 2, 0, 1).unwrap_err();
        assert!(
            matches!(err2, ResolutionError::ZeroDimension { .. }),
            "expected ZeroDimension (h=0), got {err2:?}"
        );
    }

    #[test]
    fn test_resize_buffer_mismatch_error() {
        // Buffer is too short: 2×2×1 = 4 expected, only 3 provided.
        let img = vec![0.0_f32; 3];
        let err = resize_image_bilinear(&img, 2, 2, 2, 2, 1).unwrap_err();
        assert!(
            matches!(
                err,
                ResolutionError::BufferSizeMismatch {
                    got: 3,
                    expected: 4,
                    ..
                }
            ),
            "expected BufferSizeMismatch, got {err:?}"
        );
    }

    // ── Center crop ───────────────────────────────────────────────────────────

    #[test]
    fn test_crop_center_exact() {
        // 4×4 single-channel image; crop the inner 2×2.
        // Row-major layout (row 0 at top):
        //   0  1  2  3
        //   4  5  6  7
        //   8  9 10 11
        //  12 13 14 15
        //
        // Center 2×2 (x_off=1, y_off=1): pixels 5, 6, 9, 10
        let img: Vec<f32> = (0..16).map(|v| v as f32).collect();
        let out = crop_center(&img, 4, 4, 2, 2, 1).unwrap();
        assert_eq!(out, vec![5.0, 6.0, 9.0, 10.0]);
    }

    #[test]
    fn test_crop_center_no_op() {
        // crop == source dimensions → identical output.
        let img: Vec<f32> = (0..12).map(|v| v as f32).collect();
        let out = crop_center(&img, 4, 3, 4, 3, 1).unwrap();
        assert_eq!(out, img);
    }

    #[test]
    fn test_crop_center_too_large() {
        let img = vec![0.0_f32; 4];
        let err = crop_center(&img, 2, 2, 3, 2, 1).unwrap_err();
        assert!(
            matches!(err, ResolutionError::CropTooLarge { cw: 3, ch: 2, .. }),
            "expected CropTooLarge, got {err:?}"
        );
    }

    // ── Letterbox pad ─────────────────────────────────────────────────────────

    #[test]
    fn test_letterbox_square_input() {
        // 4×4 single-channel input, target 4 → no padding needed, output == resize.
        let img: Vec<f32> = vec![0.5; 16];
        let out = letterbox_pad(&img, 4, 4, 4, 1, 0.0).unwrap();
        assert_eq!(out.len(), 16);
        // No padding because the image is already square and the same size.
        for &v in &out {
            assert!((v - 0.5).abs() < 1e-5, "expected 0.5, got {v}");
        }
    }

    #[test]
    fn test_letterbox_wide_input() {
        // 8×2 (wide) image into a 4×4 target.
        // Scale to fit width: scaled_w=4, scaled_h=1.
        // Padding along height: pad_y = (4-1)/2 = 1.
        // Rows 0 and 2..3 should be fill=0.0; row 1 should have image content.
        // Length is width × height × channels, at channels = 1.
        let img: Vec<f32> = vec![1.0; 8 * 2];
        let out = letterbox_pad(&img, 8, 2, 4, 1, 0.0).unwrap();
        assert_eq!(out.len(), 16, "output length should be 4*4*1=16");

        // Row 0 (y=0): pure padding
        for &v in &out[0..4] {
            assert!(
                (v - 0.0).abs() < 1e-6,
                "row0 pad pixel should be 0.0, got {v}"
            );
        }
        // Row 1 (y=1): image content (all-one source → bilinear stays 1.0)
        for &v in &out[4..8] {
            assert!(
                (v - 1.0).abs() < 1e-5,
                "row1 image pixel should be ~1.0, got {v}"
            );
        }
    }

    #[test]
    fn test_letterbox_tall_input() {
        // 2×8 (tall) image into a 4×4 target.
        // Scale to fit height: scaled_h=4, scaled_w=1.
        // Padding along width: pad_x = (4-1)/2 = 1.
        // Length is width × height × channels, at channels = 1.
        let img: Vec<f32> = vec![1.0; 2 * 8];
        let out = letterbox_pad(&img, 2, 8, 4, 1, 0.0).unwrap();
        assert_eq!(out.len(), 16);

        // For each row, pixels at x=0 and x=2,3 should be padding (0.0).
        for row in 0..4usize {
            let base = row * 4;
            // x=0: padding
            assert!(
                (out[base] - 0.0).abs() < 1e-6,
                "row {row} x=0 should be padding 0.0, got {}",
                out[base]
            );
            // x=1: image content (~1.0)
            assert!(
                (out[base + 1] - 1.0).abs() < 1e-5,
                "row {row} x=1 should be ~1.0, got {}",
                out[base + 1]
            );
            // x=2,3: padding
            assert!(
                (out[base + 2] - 0.0).abs() < 1e-6,
                "row {row} x=2 should be padding 0.0, got {}",
                out[base + 2]
            );
            assert!(
                (out[base + 3] - 0.0).abs() < 1e-6,
                "row {row} x=3 should be padding 0.0, got {}",
                out[base + 3]
            );
        }
    }

    #[test]
    fn test_letterbox_output_size() {
        // Regardless of input shape, output is always target_size×target_size×channels.
        let cases: &[(usize, usize, usize, usize)] = &[
            (1, 1, 8, 1),
            (3, 7, 16, 3),
            (100, 50, 64, 4),
            (50, 100, 32, 1),
            (64, 64, 64, 3),
        ];
        for &(w, h, target, c) in cases {
            let img = vec![0.5_f32; w * h * c];
            let out = letterbox_pad(&img, w, h, target, c, 0.0).unwrap();
            let expected_len = target * target * c;
            assert_eq!(
                out.len(),
                expected_len,
                "case ({w}×{h} → {target}, c={c}): expected {expected_len} got {}",
                out.len()
            );
        }
    }

    #[test]
    fn test_letterbox_fill_value() {
        // 1×1 source padded into a 4×4 target → all non-image pixels must equal fill.
        let img = vec![0.99_f32];
        let fill = 0.123;
        let out = letterbox_pad(&img, 1, 1, 4, 1, fill).unwrap();
        assert_eq!(out.len(), 16);

        // The scaled image sits in the center 1×1 region: pad_x=1, pad_y=1 (or
        // whichever of width/height dominates).  All other pixels must equal fill.
        // We identify the single image pixel conservatively: it is the one that
        // differs from fill by more than 0.1 (image=0.99, fill=0.123).
        let mut image_pixel_count = 0;
        for &v in &out {
            if (v - fill).abs() > 0.1 {
                image_pixel_count += 1;
                // The image pixel should be close to 0.99.
                assert!(
                    (v - 0.99).abs() < 0.01,
                    "image pixel expected ~0.99 got {v}"
                );
            } else {
                assert!((v - fill).abs() < 1e-6, "pad pixel expected {fill} got {v}");
            }
        }
        assert!(
            image_pixel_count >= 1,
            "no image pixel found in letterboxed output"
        );
    }

    // ── DiffusionConfig factory ───────────────────────────────────────────────

    #[test]
    fn test_diffusion_config_r128() {
        let cfg = diffusion_config_for_resolution(SupportedResolution::R128);
        assert_eq!(cfg.image_size, 128);
        assert_eq!(cfg.latent_size, 16);
        // unet_in_channels = 2 * latent_channels = 2 * 4 = 8.
        assert_eq!(cfg.unet_in_channels, 8);
    }

    #[test]
    fn test_diffusion_config_r256() {
        let cfg = diffusion_config_for_resolution(SupportedResolution::R256);
        assert_eq!(cfg.image_size, 256);
        assert_eq!(cfg.latent_size, 32);
        assert_eq!(cfg.unet_in_channels, 8);
    }

    #[test]
    fn test_diffusion_config_r512() {
        let cfg = diffusion_config_for_resolution(SupportedResolution::R512);
        assert_eq!(cfg.image_size, 512);
        assert_eq!(cfg.latent_size, 64);
        assert_eq!(cfg.unet_in_channels, 8);
    }
}
