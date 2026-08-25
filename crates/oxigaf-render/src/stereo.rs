//! Stereo rendering: paired left/right eye views for 3D displays, VR, and anaglyph images.
//!
//! This module provides:
//! - [`StereoConfig`]: camera pair configuration with interpupillary distance (IPD),
//!   convergence distance, and eye offset mode (Parallel, ToeIn, OffAxis).
//! - [`compose_anaglyph`]: combine left/right images into a single anaglyph frame.
//! - [`compose_side_by_side`] / [`compose_top_bottom`]: layout helpers for stereoscopic displays.
//! - [`split_side_by_side`]: inverse of side-by-side composition.
//! - [`compute_disparity_sad`]: block-matching SAD disparity estimation.
//! - [`disparity_to_depth`]: convert disparity values to depth.
//! - [`disparity_to_image`]: visualise a disparity map as a greyscale image.
//! - [`compute_stereo_stats`]: summary statistics of a disparity map.

use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors produced by stereo rendering operations.
#[derive(Debug, Error)]
pub enum StereoError {
    /// Interpupillary distance must be positive.
    #[error("Interpupillary distance must be positive, got {0}")]
    InvalidIpd(f32),

    /// The two images have different dimensions.
    #[error("Image dimensions mismatch: left {left_w}×{left_h}, right {right_w}×{right_h}")]
    DimensionMismatch {
        /// Width of the left image.
        left_w: usize,
        /// Height of the left image.
        left_h: usize,
        /// Width of the right image.
        right_w: usize,
        /// Height of the right image.
        right_h: usize,
    },

    /// The supplied data slice does not match the declared dimensions.
    #[error("Image data length {len} does not match {w}×{h}×3")]
    DataLengthMismatch {
        /// Actual data length.
        len: usize,
        /// Declared width.
        w: usize,
        /// Declared height.
        h: usize,
    },

    /// Convergence distance must be positive.
    #[error("Invalid convergence distance: {0} (must be > 0)")]
    InvalidConvergence(f32),

    /// max_disparity must be ≥ min_disparity.
    #[error("Invalid disparity range: max {max} < min {min}")]
    InvalidDisparityRange {
        /// Minimum disparity.
        min: f32,
        /// Maximum disparity.
        max: f32,
    },
}

// ─────────────────────────────────────────────────────────────────────────────
// EyeOffsetMode
// ─────────────────────────────────────────────────────────────────────────────

/// Method for positioning the two virtual cameras relative to the centre camera.
#[derive(Debug, Clone, PartialEq)]
pub enum EyeOffsetMode {
    /// Parallel cameras: both cameras point in the same direction; no toe-in.
    /// Suitable for subjects at infinity; causes divergence at close range.
    Parallel,

    /// Toe-in cameras: each camera is rotated toward the convergence point.
    /// Simple to implement but causes keystoning artefacts on flat screens.
    ToeIn,

    /// Off-axis frustum shift: cameras translate only, the frustum asymmetry
    /// handles convergence. Preferred method for stereo 3D production.
    OffAxis,
}

// ─────────────────────────────────────────────────────────────────────────────
// StereoConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Stereo camera pair configuration.
#[derive(Debug, Clone)]
pub struct StereoConfig {
    /// Interpupillary distance (IPD) in world units. Typical human value: 0.065 (65 mm).
    pub ipd: f32,

    /// Convergence distance: the distance at which the left and right views converge.
    pub convergence_distance: f32,

    /// Eye offset mode controlling how the two cameras diverge from the centre.
    pub offset_mode: EyeOffsetMode,

    /// Camera focal length in pixels; used to convert IPD to a pixel-space shift.
    pub focal_length: f32,
}

impl Default for StereoConfig {
    fn default() -> Self {
        Self {
            ipd: 0.065,
            convergence_distance: 1.0,
            offset_mode: EyeOffsetMode::OffAxis,
            focal_length: 500.0,
        }
    }
}

impl StereoConfig {
    /// Create a new `StereoConfig` with validated IPD and convergence distance.
    ///
    /// # Errors
    ///
    /// - [`StereoError::InvalidIpd`] if `ipd ≤ 0`.
    /// - [`StereoError::InvalidConvergence`] if `convergence ≤ 0`.
    pub fn new(ipd: f32, convergence: f32) -> Result<Self, StereoError> {
        if ipd <= 0.0 {
            return Err(StereoError::InvalidIpd(ipd));
        }
        if convergence <= 0.0 {
            return Err(StereoError::InvalidConvergence(convergence));
        }
        Ok(Self {
            ipd,
            convergence_distance: convergence,
            ..Self::default()
        })
    }

    /// Left eye translation offset `[tx, ty, tz]` in camera-local coordinates.
    ///
    /// The left eye is displaced by `-ipd/2` along the camera's local X-axis.
    pub fn left_eye_offset(&self) -> [f32; 3] {
        [-self.ipd / 2.0, 0.0, 0.0]
    }

    /// Right eye translation offset `[tx, ty, tz]` in camera-local coordinates.
    ///
    /// The right eye is displaced by `+ipd/2` along the camera's local X-axis.
    pub fn right_eye_offset(&self) -> [f32; 3] {
        [self.ipd / 2.0, 0.0, 0.0]
    }

    /// Horizontal pixel shift for off-axis stereo.
    ///
    /// `shift = (ipd / 2) * focal_length / convergence_distance`
    pub fn pixel_shift(&self) -> f32 {
        (self.ipd / 2.0) * self.focal_length / self.convergence_distance
    }

    /// Half-angle of toe-in rotation in radians.
    ///
    /// `angle = atan((ipd / 2) / convergence_distance)`
    pub fn toe_in_angle_rad(&self) -> f32 {
        (self.ipd / 2.0 / self.convergence_distance).atan()
    }

    /// Compute left and right 4×4 view matrices from a centre view matrix.
    ///
    /// `center_view` is stored row-major (`index = row*4+col`) and follows
    /// the row-vector convention used throughout this module: a world-space
    /// point `p` (as a row vector) is transformed into camera space via
    /// `p_cam = p * center_view`, with the rotation block in rows 0-2 and
    /// the translation in row 3 (indices 12, 13, 14) -- see
    /// this module's private `mat4_mul_row_major` and `ry_rotation` helpers.
    ///
    /// Returns `(left_view, right_view)`.
    ///
    /// # Eye offset strategies
    ///
    /// - **Parallel** / **OffAxis**: shift the camera by `ipd/2` along its
    ///   own local right (+X) axis, per [`Self::left_eye_offset`] /
    ///   [`Self::right_eye_offset`]. For a row-vector view matrix, applying
    ///   a pure local-+X translation *after* `center_view` (i.e.
    ///   post-composing) only ever changes the translation row's X
    ///   component (index 12) -- the rotation block is structurally
    ///   untouched by such a composition for *any* rotation, so no
    ///   extraction of a world-space right vector is needed (this can be
    ///   verified by expanding `center_view * translate(delta, 0, 0)`).
    ///   The left eye moves toward local -X, which is the mirror
    ///   transform: every point's *camera-space* X shifts by `+ipd/2` (an
    ///   eye moving left makes the world appear to shift right); the right
    ///   eye shifts camera-space X by `-ipd/2`.
    /// - **ToeIn**: same translation, then post-multiplied -- i.e. applied
    ///   in the eye's own local space, after the translation, not in world
    ///   space -- by a Y-axis rotation of ±`toe_in_angle_rad()`. Left eye
    ///   rotates by `+angle`, right by `-angle`.
    pub fn stereo_view_matrices(&self, center_view: &[f32; 16]) -> ([f32; 16], [f32; 16]) {
        let half_ipd = self.ipd / 2.0;

        let mut left = *center_view;
        let mut right = *center_view;

        // Local +X translation row (index 12); see the doc comment above.
        left[12] += half_ipd;
        right[12] -= half_ipd;

        if self.offset_mode == EyeOffsetMode::ToeIn {
            let angle = self.toe_in_angle_rad();

            // Post-multiply: apply the translation first, then rotate in
            // the resulting eye-local space, not in world space.
            left = mat4_mul_row_major(&left, &ry_rotation(angle));
            right = mat4_mul_row_major(&right, &ry_rotation(-angle));
        }

        (left, right)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal 4×4 matrix helpers (row-major)
// ─────────────────────────────────────────────────────────────────────────────

/// Build a row-major Y-axis rotation matrix for angle θ (radians).
///
/// ```text
/// [ cos θ   0   sin θ   0 ]
/// [  0      1    0      0 ]
/// [-sin θ   0   cos θ   0 ]
/// [  0      0    0      1 ]
/// ```
fn ry_rotation(angle: f32) -> [f32; 16] {
    let (s, c) = angle.sin_cos();
    [
        c, 0.0, s, 0.0, 0.0, 1.0, 0.0, 0.0, -s, 0.0, c, 0.0, 0.0, 0.0, 0.0, 1.0,
    ]
}

/// Multiply two row-major 4×4 matrices: `a * b`.
fn mat4_mul_row_major(a: &[f32; 16], b: &[f32; 16]) -> [f32; 16] {
    let mut out = [0.0_f32; 16];
    for row in 0..4 {
        for col in 0..4 {
            let mut sum = 0.0_f32;
            for k in 0..4 {
                sum += a[row * 4 + k] * b[k * 4 + col];
            }
            out[row * 4 + col] = sum;
        }
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// StereoImage
// ─────────────────────────────────────────────────────────────────────────────

/// A flat RGB image with `f32` pixel values in `[0, 1]`.
///
/// Data layout: `data[( y * width + x ) * 3 + channel]`.
#[derive(Debug, Clone)]
pub struct StereoImage {
    /// Flat pixel data: length = `width * height * 3`.
    pub data: Vec<f32>,
    /// Image width in pixels.
    pub width: usize,
    /// Image height in pixels.
    pub height: usize,
}

impl StereoImage {
    /// Create a new `StereoImage`, validating that `data.len() == width * height * 3`.
    ///
    /// # Errors
    ///
    /// Returns [`StereoError::DataLengthMismatch`] if the data length is wrong.
    pub fn new(data: Vec<f32>, width: usize, height: usize) -> Result<Self, StereoError> {
        let expected = width * height * 3;
        if data.len() != expected {
            return Err(StereoError::DataLengthMismatch {
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

    /// Create a black (all-zero) image of the given dimensions.
    pub fn zeros(width: usize, height: usize) -> Self {
        Self {
            data: vec![0.0_f32; width * height * 3],
            width,
            height,
        }
    }

    /// Read the RGB triple at column `x`, row `y`.
    ///
    /// Panics (debug) if `x >= width` or `y >= height`.
    #[inline]
    pub fn pixel(&self, x: usize, y: usize) -> [f32; 3] {
        let base = (y * self.width + x) * 3;
        [self.data[base], self.data[base + 1], self.data[base + 2]]
    }

    /// Write the RGB triple at column `x`, row `y`.
    #[inline]
    pub fn set_pixel(&mut self, x: usize, y: usize, rgb: [f32; 3]) {
        let base = (y * self.width + x) * 3;
        self.data[base] = rgb[0];
        self.data[base + 1] = rgb[1];
        self.data[base + 2] = rgb[2];
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AnaglyphMode
// ─────────────────────────────────────────────────────────────────────────────

/// Method for combining a stereo pair into a single anaglyph image.
#[derive(Debug, Clone, PartialEq)]
pub enum AnaglyphMode {
    /// True anaglyph: red from left, cyan (G+B) from right.
    RedCyan,

    /// Green-magenta anaglyph.
    GreenMagenta,

    /// Amber-blue anaglyph with better colour preservation.
    AmberBlue,

    /// Optimised (Dubois-style) anaglyph that approximates a least-squares colour fit.
    Optimized,
}

// ─────────────────────────────────────────────────────────────────────────────
// compose_anaglyph
// ─────────────────────────────────────────────────────────────────────────────

/// Combine left and right stereo images into a single anaglyph.
///
/// Both images must have identical dimensions; otherwise
/// [`StereoError::DimensionMismatch`] is returned.
///
/// # Channel assignments by mode
///
/// | Mode            | out.r                                  | out.g                                  | out.b                              |
/// |-----------------|----------------------------------------|----------------------------------------|------------------------------------|
/// | `RedCyan`       | left.r                                 | right.g                                | right.b                            |
/// | `GreenMagenta`  | right.r                                | left.g                                 | right.b                            |
/// | `AmberBlue`     | amber = left.r×0.45 + left.g×0.55      | amber (same value as out.r)            | right.b                            |
/// | `Optimized`     | Dubois R formula (clamped to \[0,1\])  | Dubois G formula (clamped to \[0,1\])  | Dubois B formula (clamped to \[0,1\])|
pub fn compose_anaglyph(
    left: &StereoImage,
    right: &StereoImage,
    mode: AnaglyphMode,
) -> Result<StereoImage, StereoError> {
    if left.width != right.width || left.height != right.height {
        return Err(StereoError::DimensionMismatch {
            left_w: left.width,
            left_h: left.height,
            right_w: right.width,
            right_h: right.height,
        });
    }

    let w = left.width;
    let h = left.height;
    let mut out = StereoImage::zeros(w, h);

    for y in 0..h {
        for x in 0..w {
            let l = left.pixel(x, y);
            let r = right.pixel(x, y);

            let rgb = match mode {
                AnaglyphMode::RedCyan => [l[0], r[1], r[2]],

                AnaglyphMode::GreenMagenta => [r[0], l[1], r[2]],

                AnaglyphMode::AmberBlue => {
                    // Amber is a red+green mixture: write the left eye's
                    // amber luma to BOTH R and G. Zeroing G (as a previous
                    // version of this code did) turns the result into a
                    // plain red-blue anaglyph and defeats the colour
                    // preservation this mode exists for.
                    let amber = l[0] * 0.45 + l[1] * 0.55;
                    [amber, amber, r[2]]
                }

                AnaglyphMode::Optimized => {
                    // Dubois red-cyan coefficients.
                    let lr = l[0];
                    let lg = l[1];
                    let lb = l[2];
                    let rr = r[0];
                    let rg = r[1];
                    let rb = r[2];

                    let out_r = (0.437 * lr + 0.449 * lg + 0.164 * lb
                        - 0.011 * rr
                        - 0.032 * rg
                        - 0.007 * rb)
                        .clamp(0.0, 1.0);
                    let out_g = (-0.062 * lr - 0.062 * lg - 0.024 * lb
                        + 0.377 * rr
                        + 0.761 * rg
                        + 0.009 * rb)
                        .clamp(0.0, 1.0);
                    let out_b = (-0.048 * lr - 0.050 * lg - 0.017 * lb - 0.026 * rr - 0.093 * rg
                        + 1.234 * rb)
                        .clamp(0.0, 1.0);
                    [out_r, out_g, out_b]
                }
            };

            out.set_pixel(x, y, rgb);
        }
    }

    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// Side-by-side and top-bottom layout
// ─────────────────────────────────────────────────────────────────────────────

/// Compose left and right images into a single wide side-by-side image.
///
/// Output width = `left.width + right.width`.
/// Output height = `max(left.height, right.height)`; the shorter image is padded
/// with black on the bottom.
///
/// # Errors
///
/// This function currently always succeeds; the `Result` return type is kept for
/// API consistency with the other composition functions.
pub fn compose_side_by_side(
    left: &StereoImage,
    right: &StereoImage,
) -> Result<StereoImage, StereoError> {
    let out_w = left.width + right.width;
    let out_h = left.height.max(right.height);
    let mut out = StereoImage::zeros(out_w, out_h);

    for y in 0..left.height {
        for x in 0..left.width {
            out.set_pixel(x, y, left.pixel(x, y));
        }
    }
    for y in 0..right.height {
        for x in 0..right.width {
            out.set_pixel(left.width + x, y, right.pixel(x, y));
        }
    }

    Ok(out)
}

/// Compose top and bottom images into a single tall stacked image.
///
/// Output width = `max(top.width, bottom.width)`; the narrower image is padded
/// with black on the right.
/// Output height = `top.height + bottom.height`.
///
/// # Errors
///
/// This function currently always succeeds; the `Result` return type is kept for
/// API consistency.
pub fn compose_top_bottom(
    top: &StereoImage,
    bottom: &StereoImage,
) -> Result<StereoImage, StereoError> {
    let out_w = top.width.max(bottom.width);
    let out_h = top.height + bottom.height;
    let mut out = StereoImage::zeros(out_w, out_h);

    for y in 0..top.height {
        for x in 0..top.width {
            out.set_pixel(x, y, top.pixel(x, y));
        }
    }
    for y in 0..bottom.height {
        for x in 0..bottom.width {
            out.set_pixel(x, top.height + y, bottom.pixel(x, y));
        }
    }

    Ok(out)
}

/// Split a side-by-side image into left and right halves.
///
/// The split point is at `combined.width / 2`.  For a perfect roundtrip with
/// [`compose_side_by_side`] the combined image must have even width (i.e. both
/// halves had the same width).
///
/// # Errors
///
/// This function currently always succeeds; the `Result` return type is kept for
/// API consistency.
pub fn split_side_by_side(
    combined: &StereoImage,
) -> Result<(StereoImage, StereoImage), StereoError> {
    let half_w = combined.width / 2;
    let h = combined.height;

    let mut left = StereoImage::zeros(half_w, h);
    let mut right = StereoImage::zeros(combined.width - half_w, h);

    for y in 0..h {
        for x in 0..half_w {
            left.set_pixel(x, y, combined.pixel(x, y));
        }
        for x in half_w..combined.width {
            right.set_pixel(x - half_w, y, combined.pixel(x, y));
        }
    }

    Ok((left, right))
}

// ─────────────────────────────────────────────────────────────────────────────
// Disparity computation
// ─────────────────────────────────────────────────────────────────────────────

/// Compute a disparity map from left and right stereo images using block-matching SAD.
///
/// For each pixel `(x, y)` in the left image the function searches for the best
/// matching block in the right image along the epipolar line (horizontal).  The
/// search range is `d ∈ [0, max_disparity]` (right-image x is `x - d`).
///
/// Matching quality is measured by the sum of absolute differences (SAD) over an
/// `(2*block_radius+1) × (2*block_radius+1)` window across all three RGB channels.
///
/// The returned vector has length `width × height` in row-major order.
/// Disparity values are in `[0, max_disparity]`.
///
/// # Errors
///
/// Returns [`StereoError::DimensionMismatch`] if the two images have different
/// dimensions.
pub fn compute_disparity_sad(
    left: &StereoImage,
    right: &StereoImage,
    max_disparity: i32,
    block_radius: usize,
) -> Result<Vec<f32>, StereoError> {
    if left.width != right.width || left.height != right.height {
        return Err(StereoError::DimensionMismatch {
            left_w: left.width,
            left_h: left.height,
            right_w: right.width,
            right_h: right.height,
        });
    }

    let w = left.width;
    let h = left.height;
    let mut disparity_map = vec![0.0_f32; w * h];

    let br = block_radius as isize;

    for y in 0..h {
        for x in 0..w {
            let xi = x as isize;
            let yi = y as isize;

            let mut best_sad = f32::MAX;
            let mut best_d = 0_i32;

            for d in 0..=max_disparity {
                let rx = xi - d as isize;
                if rx < 0 {
                    break; // further d values will be even more negative
                }

                let mut sad = 0.0_f32;

                for by in -br..=br {
                    let ly = yi + by;
                    let ry = yi + by;

                    if ly < 0 || ly >= h as isize || ry < 0 || ry >= h as isize {
                        continue;
                    }

                    for bx in -br..=br {
                        let lx = xi + bx;
                        let rxb = rx + bx;

                        if lx < 0 || lx >= w as isize || rxb < 0 || rxb >= w as isize {
                            continue;
                        }

                        let lp = left.pixel(lx as usize, ly as usize);
                        let rp = right.pixel(rxb as usize, ry as usize);

                        sad +=
                            (lp[0] - rp[0]).abs() + (lp[1] - rp[1]).abs() + (lp[2] - rp[2]).abs();
                    }
                }

                if sad < best_sad {
                    best_sad = sad;
                    best_d = d;
                }
            }

            disparity_map[y * w + x] = best_d as f32;
        }
    }

    Ok(disparity_map)
}

// ─────────────────────────────────────────────────────────────────────────────
// Disparity → depth conversion
// ─────────────────────────────────────────────────────────────────────────────

/// Convert a disparity map to a depth map.
///
/// `depth = focal_length * baseline / disparity`
///
/// Pixels with `disparity == 0` (or very close to zero) are assigned `far_plane`.
pub fn disparity_to_depth(
    disparity: &[f32],
    focal_length: f32,
    baseline: f32,
    far_plane: f32,
) -> Vec<f32> {
    let numerator = focal_length * baseline;
    disparity
        .iter()
        .map(|&d| {
            if d < 1e-6 {
                far_plane
            } else {
                (numerator / d).min(far_plane)
            }
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Disparity visualisation
// ─────────────────────────────────────────────────────────────────────────────

/// Render a disparity map as a greyscale [`StereoImage`] with values in `[0, 1]`.
///
/// The disparity values are normalised by the maximum value found in the slice.
/// An all-zero disparity map produces a fully black image.
pub fn disparity_to_image(disparity: &[f32], width: usize, height: usize) -> StereoImage {
    let max_d = disparity.iter().cloned().fold(0.0_f32, f32::max);

    let inv = if max_d > 0.0 { 1.0 / max_d } else { 0.0 };

    let mut img = StereoImage::zeros(width, height);

    for (i, &d) in disparity.iter().enumerate() {
        let v = (d * inv).clamp(0.0, 1.0);
        let x = i % width;
        let y = i / width;
        img.set_pixel(x, y, [v, v, v]);
    }

    img
}

// ─────────────────────────────────────────────────────────────────────────────
// StereoStats
// ─────────────────────────────────────────────────────────────────────────────

/// Summary statistics of a disparity map.
#[derive(Debug, Clone)]
pub struct StereoStats {
    /// Mean disparity value.
    pub mean_disparity: f32,
    /// Maximum disparity value.
    pub max_disparity: f32,
    /// Minimum disparity value.
    pub min_disparity: f32,
    /// Standard deviation of disparity values.
    pub std_disparity: f32,
    /// Total number of pixels.
    pub num_pixels: usize,
}

/// Compute summary statistics from a flat disparity map.
///
/// Returns an all-zero [`StereoStats`] for an empty slice.
pub fn compute_stereo_stats(disparity: &[f32]) -> StereoStats {
    let n = disparity.len();

    if n == 0 {
        return StereoStats {
            mean_disparity: 0.0,
            max_disparity: 0.0,
            min_disparity: 0.0,
            std_disparity: 0.0,
            num_pixels: 0,
        };
    }

    let mut sum = 0.0_f32;
    let mut sum_sq = 0.0_f32;
    let mut min_d = f32::MAX;
    let mut max_d = f32::MIN;

    for &d in disparity {
        sum += d;
        sum_sq += d * d;
        if d < min_d {
            min_d = d;
        }
        if d > max_d {
            max_d = d;
        }
    }

    let nf = n as f32;
    let mean = sum / nf;

    // Var = E[d²] - mean² (numerically stable guard against tiny negatives)
    let variance = (sum_sq / nf - mean * mean).max(0.0);
    let std = variance.sqrt();

    StereoStats {
        mean_disparity: mean,
        max_disparity: max_d,
        min_disparity: min_d,
        std_disparity: std,
        num_pixels: n,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    /// Create a uniform grey image (all pixels set to `value` on all channels).
    fn gray_image(w: usize, h: usize, value: f32) -> StereoImage {
        let mut img = StereoImage::zeros(w, h);
        for y in 0..h {
            for x in 0..w {
                img.set_pixel(x, y, [value, value, value]);
            }
        }
        img
    }

    /// Create a checkerboard pattern: bright cells when (x+y) is even, dark otherwise.
    fn checkerboard(w: usize, h: usize) -> StereoImage {
        let mut img = StereoImage::zeros(w, h);
        for y in 0..h {
            for x in 0..w {
                let v = if (x + y) % 2 == 0 { 1.0 } else { 0.0 };
                img.set_pixel(x, y, [v, v, v]);
            }
        }
        img
    }

    // ── StereoConfig tests ────────────────────────────────────────────────────

    #[test]
    fn test_default_config_valid() {
        let cfg = StereoConfig::default();
        assert!(cfg.ipd > 0.0, "ipd must be positive");
        assert!(
            cfg.convergence_distance > 0.0,
            "convergence must be positive"
        );
        assert!(cfg.focal_length > 0.0, "focal_length must be positive");
    }

    #[test]
    fn test_new_negative_ipd_error() {
        let result = StereoConfig::new(-0.065, 1.0);
        assert!(
            matches!(result, Err(StereoError::InvalidIpd(_))),
            "Expected InvalidIpd, got {:?}",
            result
        );
    }

    #[test]
    fn test_new_zero_ipd_error() {
        let result = StereoConfig::new(0.0, 1.0);
        assert!(
            matches!(result, Err(StereoError::InvalidIpd(_))),
            "Expected InvalidIpd for ipd=0, got {:?}",
            result
        );
    }

    #[test]
    fn test_new_negative_convergence_error() {
        let result = StereoConfig::new(0.065, -1.0);
        assert!(
            matches!(result, Err(StereoError::InvalidConvergence(_))),
            "Expected InvalidConvergence, got {:?}",
            result
        );
    }

    #[test]
    fn test_eye_offsets_symmetric() {
        let cfg = StereoConfig::default();
        let lo = cfg.left_eye_offset();
        let ro = cfg.right_eye_offset();

        // x components should be equal and opposite
        assert!(
            approx_eq(lo[0], -ro[0], 1e-6),
            "Left/right x offsets not symmetric: {} vs {}",
            lo[0],
            ro[0]
        );
        // y and z should both be 0
        assert!(approx_eq(lo[1], 0.0, 1e-6));
        assert!(approx_eq(lo[2], 0.0, 1e-6));
        assert!(approx_eq(ro[1], 0.0, 1e-6));
        assert!(approx_eq(ro[2], 0.0, 1e-6));

        // magnitude = ipd/2
        assert!(
            approx_eq(ro[0], cfg.ipd / 2.0, 1e-6),
            "right x offset should be +ipd/2"
        );
    }

    #[test]
    fn test_pixel_shift_positive() {
        let cfg = StereoConfig::default();
        let shift = cfg.pixel_shift();
        assert!(shift > 0.0, "pixel_shift must be positive, got {shift}");
    }

    #[test]
    fn test_toe_in_angle_positive() {
        let cfg = StereoConfig::default();
        let angle = cfg.toe_in_angle_rad();
        assert!(
            angle > 0.0,
            "toe_in_angle_rad must be positive, got {angle}"
        );
    }

    #[test]
    fn test_stereo_view_matrices_differ_from_center() {
        let cfg = StereoConfig::default();

        // Identity view matrix.
        let center = [
            1.0_f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];

        let (left, right) = cfg.stereo_view_matrices(&center);

        // Neither left nor right should equal the center.
        assert_ne!(left, center, "Left view must differ from center");
        assert_ne!(right, center, "Right view must differ from center");

        // Left and right should differ from each other.
        assert_ne!(left, right, "Left and right views must differ");
    }

    #[test]
    fn test_stereo_view_matrices_toe_in_differ() {
        let cfg = StereoConfig {
            offset_mode: EyeOffsetMode::ToeIn,
            ..Default::default()
        };

        let center = [
            1.0_f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];

        let (left, right) = cfg.stereo_view_matrices(&center);
        assert_ne!(left, center, "ToeIn left must differ from center");
        assert_ne!(right, center, "ToeIn right must differ from center");
    }

    #[test]
    fn test_stereo_view_matrices_left_eye_sees_point_further_right() {
        // Regression test: a world point directly ahead of an identity
        // camera must map to a larger camera-space X in the left-eye view
        // than in the right-eye view, for every offset mode (an eye that
        // moves left sees the world shift right). The previous
        // implementation extracted the wrong matrix axis and applied the
        // shift with the sign flipped, which produced the opposite
        // (physically wrong) ordering.
        let center = [
            1.0_f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let world_point = [5.0_f32, 0.0, 0.0, 1.0]; // world +X, homogeneous w=1

        // Row-vector transform: p_cam[j] = sum_i p_world[i] * m[i*4 + j].
        // We only need camera-space X (j = 0).
        let transform_x =
            |m: &[f32; 16]| -> f32 { (0..4).map(|i| world_point[i] * m[i * 4]).sum() };

        for mode in [
            EyeOffsetMode::Parallel,
            EyeOffsetMode::OffAxis,
            EyeOffsetMode::ToeIn,
        ] {
            let cfg = StereoConfig {
                offset_mode: mode.clone(),
                ..Default::default()
            };
            let (left, right) = cfg.stereo_view_matrices(&center);
            let left_x = transform_x(&left);
            let right_x = transform_x(&right);
            assert!(
                left_x > right_x,
                "{mode:?}: left-eye camera-space X ({left_x}) must exceed right-eye's ({right_x})"
            );
        }
    }

    // ── StereoImage tests ─────────────────────────────────────────────────────

    #[test]
    fn test_stereo_image_new_correct_length() {
        let data = vec![0.5_f32; 4 * 4 * 3];
        let img = StereoImage::new(data, 4, 4);
        assert!(img.is_ok(), "Expected Ok, got {:?}", img);
    }

    #[test]
    fn test_stereo_image_new_wrong_length() {
        let data = vec![0.5_f32; 4 * 4 * 3 + 1]; // one extra
        let result = StereoImage::new(data, 4, 4);
        assert!(
            matches!(result, Err(StereoError::DataLengthMismatch { .. })),
            "Expected DataLengthMismatch, got {:?}",
            result
        );
    }

    #[test]
    fn test_stereo_image_pixel_roundtrip() {
        let mut img = StereoImage::zeros(8, 8);
        let rgb = [0.1_f32, 0.5, 0.9];
        img.set_pixel(3, 5, rgb);
        let got = img.pixel(3, 5);
        for i in 0..3 {
            assert!(
                approx_eq(rgb[i], got[i], 1e-6),
                "channel {i}: {} vs {}",
                rgb[i],
                got[i]
            );
        }
    }

    // ── compose_anaglyph tests ────────────────────────────────────────────────

    #[test]
    fn test_anaglyph_red_cyan_r_equals_left_r() {
        let left = gray_image(4, 4, 0.8);
        let mut right = StereoImage::zeros(4, 4);
        // Set right image to a different value so channels are distinguishable.
        for y in 0..4 {
            for x in 0..4 {
                right.set_pixel(x, y, [0.2, 0.3, 0.4]);
            }
        }

        let out = compose_anaglyph(&left, &right, AnaglyphMode::RedCyan)
            .expect("compose_anaglyph failed");

        for y in 0..4 {
            for x in 0..4 {
                let op = out.pixel(x, y);
                let lp = left.pixel(x, y);
                assert!(
                    approx_eq(op[0], lp[0], 1e-6),
                    "RedCyan out.r != left.r at ({x},{y})"
                );
            }
        }
    }

    #[test]
    fn test_anaglyph_red_cyan_g_equals_right_g() {
        let left = gray_image(4, 4, 0.8);
        let mut right = StereoImage::zeros(4, 4);
        for y in 0..4 {
            for x in 0..4 {
                right.set_pixel(x, y, [0.2, 0.35, 0.4]);
            }
        }

        let out = compose_anaglyph(&left, &right, AnaglyphMode::RedCyan)
            .expect("compose_anaglyph failed");

        for y in 0..4 {
            for x in 0..4 {
                let op = out.pixel(x, y);
                let rp = right.pixel(x, y);
                assert!(
                    approx_eq(op[1], rp[1], 1e-6),
                    "RedCyan out.g != right.g at ({x},{y})"
                );
            }
        }
    }

    #[test]
    fn test_anaglyph_amber_blue_writes_amber_to_red_and_green() {
        // Regression test: AmberBlue must write the left eye's amber luma
        // to BOTH the red and green output channels (amber = red + green).
        // A previous version zeroed the green channel, which silently
        // turned this mode into a plain red-blue anaglyph.
        let mut left = StereoImage::zeros(4, 4);
        let mut right = StereoImage::zeros(4, 4);
        for y in 0..4 {
            for x in 0..4 {
                // Distinct R/G values so a zeroed-G bug is observable.
                left.set_pixel(x, y, [0.8, 0.6, 0.1]);
                right.set_pixel(x, y, [0.2, 0.35, 0.4]);
            }
        }

        let out = compose_anaglyph(&left, &right, AnaglyphMode::AmberBlue)
            .expect("compose_anaglyph failed");

        let expected_amber = 0.8_f32 * 0.45 + 0.6_f32 * 0.55;
        assert!(expected_amber > 0.0, "sanity: amber luma should be nonzero");

        for y in 0..4 {
            for x in 0..4 {
                let op = out.pixel(x, y);
                assert!(
                    approx_eq(op[0], expected_amber, 1e-6),
                    "AmberBlue out.r should be the amber luma at ({x},{y}), got {}",
                    op[0]
                );
                assert!(
                    approx_eq(op[1], expected_amber, 1e-6),
                    "AmberBlue out.g must equal out.r (amber), got {} vs {}",
                    op[1],
                    op[0]
                );
                assert!(
                    approx_eq(op[2], 0.4, 1e-6),
                    "AmberBlue out.b should be right.b at ({x},{y})"
                );
            }
        }
    }

    #[test]
    fn test_anaglyph_dimension_mismatch() {
        let left = gray_image(4, 4, 0.5);
        let right = gray_image(4, 5, 0.5); // different height

        let result = compose_anaglyph(&left, &right, AnaglyphMode::RedCyan);
        assert!(
            matches!(result, Err(StereoError::DimensionMismatch { .. })),
            "Expected DimensionMismatch, got {:?}",
            result
        );
    }

    #[test]
    fn test_anaglyph_optimized_values_in_range() {
        let left = checkerboard(8, 8);
        let right = checkerboard(8, 8);

        let out = compose_anaglyph(&left, &right, AnaglyphMode::Optimized)
            .expect("compose_anaglyph Optimized failed");

        for y in 0..8 {
            for x in 0..8 {
                let p = out.pixel(x, y);
                for (c, &val) in p.iter().enumerate() {
                    assert!(
                        (0.0..=1.0).contains(&val),
                        "Optimized: pixel ({x},{y}) channel {c} out of [0,1]: {}",
                        val
                    );
                }
            }
        }
    }

    // ── compose_side_by_side tests ────────────────────────────────────────────

    #[test]
    fn test_side_by_side_output_width() {
        let left = gray_image(6, 4, 0.3);
        let right = gray_image(6, 4, 0.7);

        let out = compose_side_by_side(&left, &right).expect("compose_side_by_side failed");
        assert_eq!(
            out.width,
            left.width + right.width,
            "side-by-side width should be sum of halves"
        );
    }

    #[test]
    fn test_side_by_side_left_half_matches() {
        let left = checkerboard(6, 4);
        let right = gray_image(6, 4, 0.5);

        let out = compose_side_by_side(&left, &right).expect("compose_side_by_side failed");

        for y in 0..left.height {
            for x in 0..left.width {
                let op = out.pixel(x, y);
                let lp = left.pixel(x, y);
                for c in 0..3 {
                    assert!(
                        approx_eq(op[c], lp[c], 1e-6),
                        "Left half mismatch at ({x},{y}) channel {c}: {} vs {}",
                        op[c],
                        lp[c]
                    );
                }
            }
        }
    }

    #[test]
    fn test_split_side_by_side_roundtrip() {
        let left_orig = checkerboard(6, 4);
        let right_orig = gray_image(6, 4, 0.6);

        let combined =
            compose_side_by_side(&left_orig, &right_orig).expect("compose_side_by_side failed");

        // Combined width = 12, which is even → clean split.
        let (left_split, right_split) =
            split_side_by_side(&combined).expect("split_side_by_side failed");

        assert_eq!(left_split.width, left_orig.width);
        assert_eq!(left_split.height, left_orig.height);
        assert_eq!(right_split.width, right_orig.width);
        assert_eq!(right_split.height, right_orig.height);

        for y in 0..left_orig.height {
            for x in 0..left_orig.width {
                let a = left_orig.pixel(x, y);
                let b = left_split.pixel(x, y);
                for c in 0..3 {
                    assert!(
                        approx_eq(a[c], b[c], 1e-6),
                        "Left roundtrip mismatch at ({x},{y}) c{c}"
                    );
                }
            }
        }

        for y in 0..right_orig.height {
            for x in 0..right_orig.width {
                let a = right_orig.pixel(x, y);
                let b = right_split.pixel(x, y);
                for c in 0..3 {
                    assert!(
                        approx_eq(a[c], b[c], 1e-6),
                        "Right roundtrip mismatch at ({x},{y}) c{c}"
                    );
                }
            }
        }
    }

    // ── compose_top_bottom tests ──────────────────────────────────────────────

    #[test]
    fn test_top_bottom_output_height() {
        let top = gray_image(4, 3, 0.2);
        let bottom = gray_image(4, 5, 0.8);

        let out = compose_top_bottom(&top, &bottom).expect("compose_top_bottom failed");
        assert_eq!(
            out.height,
            top.height + bottom.height,
            "top-bottom height should be sum"
        );
    }

    // ── compute_disparity_sad tests ───────────────────────────────────────────

    #[test]
    fn test_disparity_sad_output_length() {
        let left = checkerboard(8, 8);
        let right = checkerboard(8, 8);

        let disp =
            compute_disparity_sad(&left, &right, 4, 1).expect("compute_disparity_sad failed");

        assert_eq!(disp.len(), 8 * 8, "disparity map must have W×H elements");
    }

    #[test]
    fn test_disparity_sad_identical_images_zero_disparity() {
        // Identical images → best match is always d=0 → all disparity values = 0.
        let left = checkerboard(8, 8);
        let right = checkerboard(8, 8);

        let disp =
            compute_disparity_sad(&left, &right, 4, 1).expect("compute_disparity_sad failed");

        for (i, &d) in disp.iter().enumerate() {
            assert!(
                approx_eq(d, 0.0, 1e-6),
                "pixel {i}: expected disparity 0, got {d}"
            );
        }
    }

    #[test]
    fn test_disparity_sad_dimension_mismatch() {
        let left = gray_image(4, 4, 0.5);
        let right = gray_image(8, 4, 0.5);

        let result = compute_disparity_sad(&left, &right, 4, 1);
        assert!(
            matches!(result, Err(StereoError::DimensionMismatch { .. })),
            "Expected DimensionMismatch, got {:?}",
            result
        );
    }

    // ── disparity_to_depth tests ──────────────────────────────────────────────

    #[test]
    fn test_disparity_to_depth_zero_gets_far_plane() {
        let disp = vec![0.0_f32, 0.0, 0.0];
        let depth = disparity_to_depth(&disp, 500.0, 0.065, 1000.0);

        for &d in &depth {
            assert!(
                approx_eq(d, 1000.0, 1e-4),
                "zero disparity should give far_plane, got {d}"
            );
        }
    }

    #[test]
    fn test_disparity_to_depth_positive_gives_finite() {
        let disp = vec![4.0_f32, 8.0, 16.0];
        let depth = disparity_to_depth(&disp, 500.0, 0.065, 1000.0);

        for &d in &depth {
            assert!(d.is_finite(), "depth must be finite, got {d}");
            assert!(d > 0.0, "depth must be positive, got {d}");
        }
    }

    #[test]
    fn test_disparity_to_depth_formula() {
        // depth = focal * baseline / disparity = 500 * 0.065 / 10.0 = 3.25
        let disp = vec![10.0_f32];
        let depth = disparity_to_depth(&disp, 500.0, 0.065, 1000.0);
        assert!(
            approx_eq(depth[0], 3.25, 1e-4),
            "Expected depth=3.25, got {}",
            depth[0]
        );
    }

    // ── disparity_to_image tests ──────────────────────────────────────────────

    #[test]
    fn test_disparity_to_image_valid_stereo_image() {
        let disp = vec![0.0_f32, 1.0, 2.0, 4.0];
        let img = disparity_to_image(&disp, 2, 2);
        assert_eq!(img.width, 2);
        assert_eq!(img.height, 2);
        assert_eq!(img.data.len(), 2 * 2 * 3);

        // All pixel values should be in [0, 1].
        for y in 0..2 {
            for x in 0..2 {
                let p = img.pixel(x, y);
                for &channel_val in &p {
                    assert!(
                        (0.0..=1.0).contains(&channel_val),
                        "out of [0,1]: {}",
                        channel_val
                    );
                }
            }
        }
    }

    #[test]
    fn test_disparity_to_image_all_zero_black() {
        let disp = vec![0.0_f32; 4];
        let img = disparity_to_image(&disp, 2, 2);
        for &v in &img.data {
            assert!(
                approx_eq(v, 0.0, 1e-6),
                "all-zero disparity should give black image"
            );
        }
    }

    // ── StereoStats tests ─────────────────────────────────────────────────────

    #[test]
    fn test_stereo_stats_uniform_zero_std() {
        let disp = vec![5.0_f32; 16];
        let stats = compute_stereo_stats(&disp);

        assert!(
            approx_eq(stats.mean_disparity, 5.0, 1e-5),
            "mean = {}",
            stats.mean_disparity
        );
        assert!(
            approx_eq(stats.std_disparity, 0.0, 1e-4),
            "std of uniform should be ~0, got {}",
            stats.std_disparity
        );
        assert!(
            approx_eq(stats.min_disparity, 5.0, 1e-6),
            "min = {}",
            stats.min_disparity
        );
        assert!(
            approx_eq(stats.max_disparity, 5.0, 1e-6),
            "max = {}",
            stats.max_disparity
        );
        assert_eq!(stats.num_pixels, 16);
    }

    #[test]
    fn test_stereo_stats_mean_correct() {
        let disp = vec![0.0_f32, 2.0, 4.0, 6.0];
        let stats = compute_stereo_stats(&disp);

        // mean = (0+2+4+6)/4 = 3.0
        assert!(
            approx_eq(stats.mean_disparity, 3.0, 1e-5),
            "mean = {}",
            stats.mean_disparity
        );
        assert_eq!(stats.num_pixels, 4);
    }

    #[test]
    fn test_stereo_stats_empty() {
        let stats = compute_stereo_stats(&[]);
        assert_eq!(stats.num_pixels, 0);
        assert!(approx_eq(stats.mean_disparity, 0.0, 1e-6));
        assert!(approx_eq(stats.std_disparity, 0.0, 1e-6));
    }
}
