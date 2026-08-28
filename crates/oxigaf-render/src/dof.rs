//! Depth-of-field post-processing for rendered Gaussian frames.
//!
//! Implements a CPU-side gather-based depth-of-field effect using per-pixel
//! circle-of-confusion (CoC) radii computed from a depth map.
//!
//! # Algorithm overview
//!
//! 1. Compute per-pixel CoC radius from depth values and [`DofConfig`].
//! 2. For each output pixel, gather nearby pixels within a disc of radius equal
//!    to the pixel's CoC, accumulating a weighted colour sum.
//! 3. Pixels with CoC < 0.5 px are copied unchanged (sharp region).
//!
//! # Performance note
//!
//! The gather loop is O(W × H × r²). The effective radius is capped at 20 px
//! inside [`apply_dof`] to keep run-times reasonable during tests.

use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors produced by depth-of-field operations.
#[derive(Debug, Error)]
pub enum DofError {
    /// The image or depth map slice is empty.
    #[error("Image or depth map is empty")]
    EmptyImage,

    /// Image and depth map lengths are inconsistent.
    #[error("Size mismatch: image_len={image_len}, depth_len={depth_len}")]
    SizeMismatch {
        /// Flat length of the image buffer (H × W × C).
        image_len: usize,
        /// Flat length of the depth buffer (H × W).
        depth_len: usize,
    },

    /// Channel count is zero.
    #[error("Channel count must be at least 1")]
    ZeroChannels,

    /// Image width or height is zero.
    #[error("Image width and height must both be at least 1")]
    ZeroDimension,
}

// ─────────────────────────────────────────────────────────────────────────────
// DofConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for depth-of-field post-processing.
///
/// This is a **simplified linear model**: the blur radius in pixels is a
/// piecewise-linear function of how far a pixel's depth is from
/// `focal_distance`, controlled solely by `focal_distance`, `focal_range`
/// and `max_blur_radius` (see [`compute_coc`]). It does not model a real
/// lens (no focal length, aperture, or sensor geometry) - for a
/// physically-based thin-lens CoC model with aperture/f-stop and sensor-size
/// support, see [`crate::depth_of_field`].
#[derive(Debug, Clone)]
pub struct DofConfig {
    /// Distance in world units where the image is perfectly sharp.
    pub focal_distance: f32,
    /// Depth range around `focal_distance` that remains sharp (DoF range).
    pub focal_range: f32,
    /// Maximum blur radius in pixels.
    pub max_blur_radius: f32,
}

impl Default for DofConfig {
    fn default() -> Self {
        Self {
            focal_distance: 2.0,
            focal_range: 0.3,
            max_blur_radius: 15.0,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DofKernelShape
// ─────────────────────────────────────────────────────────────────────────────

/// Shape of the depth-of-field blur kernel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DofKernelShape {
    /// Circular disk (approximated as filled circle).
    Circular,
    /// Hexagonal shape (approximating camera aperture blades).
    Hexagonal,
    /// Square kernel (simpler, faster).
    Square,
}

// ─────────────────────────────────────────────────────────────────────────────
// DofStats
// ─────────────────────────────────────────────────────────────────────────────

/// Statistics about the depth-of-field CoC distribution.
#[derive(Debug, Clone)]
pub struct DofStats {
    /// Mean CoC radius across all pixels.
    pub mean_coc: f32,
    /// Maximum CoC radius.
    pub max_coc: f32,
    /// Fraction of pixels with CoC below `threshold` (sharp pixels).
    pub sharp_pixel_fraction: f32,
    /// Fraction of pixels with CoC at or above `threshold` (blurred pixels).
    pub blurred_pixel_fraction: f32,
}

impl DofStats {
    /// Compute statistics from a flat CoC map.
    ///
    /// `threshold` controls the boundary between "sharp" and "blurred" pixels;
    /// the canonical value is `0.5`.
    pub fn compute(coc_map: &[f32], threshold: f32) -> Self {
        if coc_map.is_empty() {
            return Self {
                mean_coc: 0.0,
                max_coc: 0.0,
                sharp_pixel_fraction: 1.0,
                blurred_pixel_fraction: 0.0,
            };
        }

        let n = coc_map.len() as f32;
        let mut sum = 0.0_f32;
        let mut max_coc = 0.0_f32;
        let mut sharp_count = 0usize;

        for &c in coc_map {
            sum += c;
            if c > max_coc {
                max_coc = c;
            }
            if c < threshold {
                sharp_count += 1;
            }
        }

        let mean_coc = sum / n;
        let sharp_pixel_fraction = sharp_count as f32 / n;
        let blurred_pixel_fraction = 1.0 - sharp_pixel_fraction;

        Self {
            mean_coc,
            max_coc,
            sharp_pixel_fraction,
            blurred_pixel_fraction,
        }
    }

    /// Return a human-readable one-line summary.
    pub fn format_summary(&self) -> String {
        format!(
            "DoF: mean_coc={:.2}px  max_coc={:.2}px  sharp={:.1}%  blurred={:.1}%",
            self.mean_coc,
            self.max_coc,
            self.sharp_pixel_fraction * 100.0,
            self.blurred_pixel_fraction * 100.0,
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// compute_coc
// ─────────────────────────────────────────────────────────────────────────────

/// Compute per-pixel circle-of-confusion radius in pixels.
///
/// The formula used is a simplified linear model:
///
/// ```text
/// half_range  = focal_range / 2
/// coc_norm    = |depth - focal_distance| / half_range   (clamped to [0, 1])
/// coc_px      = coc_norm * max_blur_radius
/// ```
///
/// Special cases:
/// - `depth == 0.0` or `depth.is_infinite()` → `coc = max_blur_radius`
/// - `depth.is_nan()` → `coc = max_blur_radius`
///
/// # Errors
///
/// Returns [`DofError::ZeroDimension`] if `image_width` or `image_height` is zero.
/// Returns [`DofError::SizeMismatch`] if `depth_map.len() != image_width * image_height`.
/// Returns [`DofError::EmptyImage`] if `depth_map` is empty.
pub fn compute_coc(
    depth_map: &[f32],
    config: &DofConfig,
    image_width: usize,
    image_height: usize,
) -> Result<Vec<f32>, DofError> {
    if image_width == 0 || image_height == 0 {
        return Err(DofError::ZeroDimension);
    }

    let expected = image_width * image_height;

    if depth_map.is_empty() {
        return Err(DofError::EmptyImage);
    }

    if depth_map.len() != expected {
        // `compute_coc` has no separate image buffer, so `image_len` here
        // holds the *expected* pixel count (`image_width * image_height`)
        // rather than an actual image buffer length. `depth_len` must still
        // hold the depth buffer's actual length, per its field doc - the
        // previous assignment had these two backwards (the actual depth
        // length under `image_len`, the expected count under `depth_len`),
        // which is doubly misleading in the rendered error message.
        return Err(DofError::SizeMismatch {
            image_len: expected,
            depth_len: depth_map.len(),
        });
    }

    let half_range = config.focal_range / 2.0;
    // Guard against degenerate focal_range of zero or negative.
    let half_range = if half_range <= 0.0 {
        f32::EPSILON
    } else {
        half_range
    };

    let mut coc_map = Vec::with_capacity(expected);

    for &depth in depth_map {
        let coc = if !depth.is_finite() || depth == 0.0 {
            config.max_blur_radius
        } else {
            let diff = (depth - config.focal_distance).abs();
            let norm = (diff / half_range).min(1.0);
            norm * config.max_blur_radius
        };
        // Ensure the result is clamped to [0, max_blur_radius] (handles NaN too).
        coc_map.push(coc.clamp(0.0, config.max_blur_radius));
    }

    Ok(coc_map)
}

// ─────────────────────────────────────────────────────────────────────────────
// generate_dof_kernel
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a normalised 2-D kernel for a given radius and shape.
///
/// Returns `(kernel_data, side)` where `side = 2 * ceil(radius) + 1`.
/// All weights are non-negative and sum to `1.0`.
///
/// At radius `0.0` this returns a 1×1 kernel with value `[1.0]`.
pub fn generate_dof_kernel(radius: f32, shape: DofKernelShape) -> (Vec<f32>, usize) {
    let radius = radius.max(0.0);
    let r_ceil = radius.ceil() as isize;
    let side = (2 * r_ceil + 1) as usize;
    let mut kernel = vec![0.0_f32; side * side];

    let sqrt3_over2: f32 = (3.0_f32).sqrt() / 2.0;

    for iy in 0..side {
        for ix in 0..side {
            let dy = iy as f32 - r_ceil as f32;
            let dx = ix as f32 - r_ceil as f32;

            let inside = match shape {
                DofKernelShape::Circular => dx * dx + dy * dy <= radius * radius,
                DofKernelShape::Hexagonal => {
                    let term_a = dx.abs();
                    let term_b = (dx / 2.0 + dy * sqrt3_over2).abs();
                    let term_c = (dx / 2.0 - dy * sqrt3_over2).abs();
                    term_a.max(term_b).max(term_c) <= radius
                }
                DofKernelShape::Square => dx.abs() <= radius && dy.abs() <= radius,
            };

            if inside {
                kernel[iy * side + ix] = 1.0;
            }
        }
    }

    // Normalise.
    let total: f32 = kernel.iter().sum();
    if total > 0.0 {
        for v in &mut kernel {
            *v /= total;
        }
    } else {
        // Degenerate: set centre pixel to 1.0 to avoid all-zero kernel.
        let centre = r_ceil as usize * side + r_ceil as usize;
        if centre < kernel.len() {
            kernel[centre] = 1.0;
        }
    }

    (kernel, side)
}

// ─────────────────────────────────────────────────────────────────────────────
// apply_dof
// ─────────────────────────────────────────────────────────────────────────────

/// Apply depth-of-field blur to an image using per-pixel CoC.
///
/// Uses a gather approach: for each output pixel the colour is accumulated from
/// nearby source pixels weighted by a circular disc of radius equal to the
/// pixel's CoC.  Sharp pixels (CoC < 0.5 px) are copied unchanged.
///
/// The gather radius is internally capped at **20 pixels** to keep performance
/// acceptable; the CoC map itself is not modified.
///
/// # Parameters
///
/// - `image`: flat `H × W × C` f32 buffer (HWC layout).
/// - `depth_map`: flat `H × W` f32 depth values.
/// - `config`: DoF configuration.
/// - `kernel_shape`: shape used to select samples within the CoC disc.
/// - `image_width`, `image_height`: spatial dimensions.
/// - `channels`: number of colour channels (typically 3).
///
/// # Errors
///
/// - [`DofError::ZeroDimension`] if either spatial dimension is zero.
/// - [`DofError::ZeroChannels`] if `channels` is zero.
/// - [`DofError::EmptyImage`] if the image slice is empty.
/// - [`DofError::SizeMismatch`] if sizes are inconsistent.
pub fn apply_dof(
    image: &[f32],
    depth_map: &[f32],
    config: &DofConfig,
    kernel_shape: DofKernelShape,
    image_width: usize,
    image_height: usize,
    channels: usize,
) -> Result<Vec<f32>, DofError> {
    if image_width == 0 || image_height == 0 {
        return Err(DofError::ZeroDimension);
    }
    if channels == 0 {
        return Err(DofError::ZeroChannels);
    }
    if image.is_empty() {
        return Err(DofError::EmptyImage);
    }

    let num_pixels = image_width * image_height;
    let expected_image = num_pixels * channels;
    let expected_depth = num_pixels;

    if image.len() != expected_image || depth_map.len() != expected_depth {
        return Err(DofError::SizeMismatch {
            image_len: image.len(),
            depth_len: depth_map.len(),
        });
    }

    // Compute CoC map.
    let coc_map = compute_coc(depth_map, config, image_width, image_height)?;

    // Maximum gather radius we'll ever use (performance guard).
    const MAX_GATHER_RADIUS: f32 = 20.0;

    let sqrt3_over2: f32 = (3.0_f32).sqrt() / 2.0;

    let mut output = vec![0.0_f32; expected_image];

    // Hoisted out of the per-pixel loop: `channels` is fixed for the whole
    // call, so a fresh `Vec` (heap allocation + deallocation) for every
    // blurred pixel is pure overhead - a 1920×1080 render performs roughly
    // two million such alloc/dealloc pairs on top of the O(W×H×r²) gather.
    // Re-zeroed at the top of each blurred pixel instead.
    let mut acc = vec![0.0_f32; channels];

    for py in 0..image_height {
        for px in 0..image_width {
            let pixel_idx = py * image_width + px;
            let coc_r = coc_map[pixel_idx];

            if coc_r < 0.5 {
                // Sharp pixel — copy unchanged.
                let base = pixel_idx * channels;
                output[base..base + channels].copy_from_slice(&image[base..base + channels]);
                continue;
            }

            // Capped gather radius.
            let r = coc_r.min(MAX_GATHER_RADIUS);
            let r_ceil = r.ceil() as isize;
            let r2 = r * r;

            acc.fill(0.0);
            let mut weight_sum = 0.0_f32;

            let y_lo = (py as isize - r_ceil).max(0) as usize;
            let y_hi = (py as isize + r_ceil).min(image_height as isize - 1) as usize;
            let x_lo = (px as isize - r_ceil).max(0) as usize;
            let x_hi = (px as isize + r_ceil).min(image_width as isize - 1) as usize;

            for sy in y_lo..=y_hi {
                for sx in x_lo..=x_hi {
                    let dy = sy as f32 - py as f32;
                    let dx = sx as f32 - px as f32;

                    let inside = match kernel_shape {
                        DofKernelShape::Circular => dx * dx + dy * dy <= r2,
                        DofKernelShape::Hexagonal => {
                            let term_a = dx.abs();
                            let term_b = (dx / 2.0 + dy * sqrt3_over2).abs();
                            let term_c = (dx / 2.0 - dy * sqrt3_over2).abs();
                            term_a.max(term_b).max(term_c) <= r
                        }
                        DofKernelShape::Square => dx.abs() <= r && dy.abs() <= r,
                    };

                    if inside {
                        let src_base = (sy * image_width + sx) * channels;
                        for c in 0..channels {
                            acc[c] += image[src_base + c];
                        }
                        weight_sum += 1.0;
                    }
                }
            }

            let dst_base = pixel_idx * channels;
            if weight_sum > 0.0 {
                let inv = 1.0 / weight_sum;
                for c in 0..channels {
                    output[dst_base + c] = acc[c] * inv;
                }
            } else {
                // Fallback: copy original pixel.
                output[dst_base..dst_base + channels]
                    .copy_from_slice(&image[dst_base..dst_base + channels]);
            }
        }
    }

    Ok(output)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helper ────────────────────────────────────────────────────────────────

    fn default_cfg() -> DofConfig {
        DofConfig::default()
    }

    /// Compare two f32 values within a loose epsilon.
    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    // ── compute_coc tests ─────────────────────────────────────────────────────

    #[test]
    fn test_coc_at_focal_distance() {
        let cfg = default_cfg(); // focal_distance = 2.0
        let depth_map = vec![2.0_f32; 4];
        let coc = compute_coc(&depth_map, &cfg, 2, 2).expect("compute_coc failed");
        for &c in &coc {
            assert!(
                approx_eq(c, 0.0, 1e-4),
                "Expected 0 coc at focal distance, got {c}"
            );
        }
    }

    #[test]
    fn test_coc_far_from_focus() {
        // depth very far → coc should saturate at max_blur_radius
        let cfg = DofConfig {
            focal_distance: 2.0,
            focal_range: 0.3,
            max_blur_radius: 15.0,
        };
        let depth_map = vec![100.0_f32; 4];
        let coc = compute_coc(&depth_map, &cfg, 2, 2).expect("compute_coc failed");
        for &c in &coc {
            assert!(
                approx_eq(c, 15.0, 1e-3),
                "Expected max_blur_radius=15.0, got {c}"
            );
        }
    }

    #[test]
    fn test_coc_linear_falloff() {
        // At distance = focal_distance + half_range the CoC should be max_blur_radius
        let cfg = DofConfig {
            focal_distance: 2.0,
            focal_range: 0.4, // half_range = 0.2
            max_blur_radius: 10.0,
        };
        // depth at exactly half_range away → normalised diff = 1.0 → coc = max
        let depth_map = vec![2.2_f32; 1];
        let coc = compute_coc(&depth_map, &cfg, 1, 1).expect("compute_coc failed");
        assert!(
            approx_eq(coc[0], 10.0, 1e-3),
            "Expected 10.0, got {}",
            coc[0]
        );

        // depth at half the half_range → normalised diff = 0.5 → coc = 5.0
        let depth_map2 = vec![2.1_f32; 1];
        let coc2 = compute_coc(&depth_map2, &cfg, 1, 1).expect("compute_coc failed");
        assert!(
            approx_eq(coc2[0], 5.0, 1e-3),
            "Expected 5.0, got {}",
            coc2[0]
        );
    }

    #[test]
    fn test_coc_clamp_at_max() {
        let cfg = DofConfig {
            focal_distance: 1.0,
            focal_range: 0.1,
            max_blur_radius: 8.0,
        };
        // Very far depth → clamped at max_blur_radius
        let depth_map = vec![999.0_f32; 1];
        let coc = compute_coc(&depth_map, &cfg, 1, 1).expect("compute_coc failed");
        assert!(coc[0] <= 8.0 + 1e-6, "CoC must not exceed max_blur_radius");
        assert!(approx_eq(coc[0], 8.0, 1e-3), "Expected 8.0, got {}", coc[0]);
    }

    #[test]
    fn test_coc_zero_depth_gets_max() {
        let cfg = default_cfg();
        let depth_map = vec![0.0_f32; 1];
        let coc = compute_coc(&depth_map, &cfg, 1, 1).expect("compute_coc failed");
        assert!(
            approx_eq(coc[0], cfg.max_blur_radius, 1e-4),
            "depth=0 should give max coc, got {}",
            coc[0]
        );
    }

    #[test]
    fn test_coc_infinite_depth_gets_max() {
        let cfg = default_cfg();
        let depth_map = vec![f32::INFINITY; 1];
        let coc = compute_coc(&depth_map, &cfg, 1, 1).expect("compute_coc failed");
        assert!(
            approx_eq(coc[0], cfg.max_blur_radius, 1e-4),
            "depth=inf should give max coc, got {}",
            coc[0]
        );
    }

    // ── generate_dof_kernel tests ─────────────────────────────────────────────

    #[test]
    fn test_generate_kernel_circular_sum_one() {
        let (kernel, side) = generate_dof_kernel(3.0, DofKernelShape::Circular);
        assert_eq!(kernel.len(), side * side);
        let sum: f32 = kernel.iter().sum();
        assert!(approx_eq(sum, 1.0, 1e-5), "Circular kernel sum = {sum}");
    }

    #[test]
    fn test_generate_kernel_hexagonal_sum_one() {
        let (kernel, side) = generate_dof_kernel(4.0, DofKernelShape::Hexagonal);
        assert_eq!(kernel.len(), side * side);
        let sum: f32 = kernel.iter().sum();
        assert!(approx_eq(sum, 1.0, 1e-5), "Hexagonal kernel sum = {sum}");
    }

    #[test]
    fn test_generate_kernel_square_sum_one() {
        let (kernel, side) = generate_dof_kernel(2.0, DofKernelShape::Square);
        assert_eq!(kernel.len(), side * side);
        let sum: f32 = kernel.iter().sum();
        assert!(approx_eq(sum, 1.0, 1e-5), "Square kernel sum = {sum}");
    }

    #[test]
    fn test_generate_kernel_radius_zero() {
        let (kernel, side) = generate_dof_kernel(0.0, DofKernelShape::Circular);
        assert_eq!(side, 1, "side for radius=0 should be 1");
        assert_eq!(kernel.len(), 1);
        assert!(approx_eq(kernel[0], 1.0, 1e-6), "1x1 kernel must be [1.0]");
    }

    #[test]
    fn test_generate_kernel_side_matches_radius() {
        // side = 2*ceil(r)+1
        let (_, side) = generate_dof_kernel(5.0, DofKernelShape::Square);
        assert_eq!(side, 11, "side for r=5 should be 11");

        let (_, side2) = generate_dof_kernel(2.7, DofKernelShape::Circular);
        assert_eq!(side2, 7, "side for r=2.7 (ceil=3) should be 7");
    }

    // ── apply_dof tests ───────────────────────────────────────────────────────

    #[test]
    fn test_apply_dof_sharp_region_unchanged() {
        // All depths at focal distance → all CoC=0 < 0.5 → output == input
        let cfg = DofConfig {
            focal_distance: 2.0,
            focal_range: 0.3,
            max_blur_radius: 15.0,
        };
        let w = 4_usize;
        let h = 4_usize;
        let c = 3_usize;
        let image: Vec<f32> = (0..w * h * c).map(|i| i as f32 / 100.0).collect();
        let depth_map = vec![2.0_f32; w * h]; // at focal distance

        let output = apply_dof(&image, &depth_map, &cfg, DofKernelShape::Circular, w, h, c)
            .expect("apply_dof failed");

        assert_eq!(output.len(), image.len());
        for (i, (&a, &b)) in image.iter().zip(output.iter()).enumerate() {
            assert!(approx_eq(a, b, 1e-5), "pixel {i}: expected {a}, got {b}");
        }
    }

    #[test]
    fn test_apply_dof_uniform_image_unchanged() {
        // Uniform-colour image → blur keeps same values everywhere
        let cfg = default_cfg();
        let w = 8_usize;
        let h = 8_usize;
        let c = 3_usize;
        let colour = [0.4_f32, 0.6_f32, 0.8_f32];
        let image: Vec<f32> = (0..w * h).flat_map(|_| colour).collect();
        let depth_map = vec![0.0_f32; w * h]; // all max blur

        let output = apply_dof(&image, &depth_map, &cfg, DofKernelShape::Circular, w, h, c)
            .expect("apply_dof failed");

        for chunk in output.chunks_exact(c) {
            for (ci, &expected) in colour.iter().enumerate() {
                assert!(
                    approx_eq(chunk[ci], expected, 1e-4),
                    "Uniform image: channel {ci} expected {expected}, got {}",
                    chunk[ci]
                );
            }
        }
    }

    #[test]
    fn test_apply_dof_size_mismatch_error() {
        let cfg = default_cfg();
        let image = vec![0.5_f32; 3 * 4 * 3]; // 3x4 image, 3 channels
        let depth_map = vec![2.0_f32; 3 * 5]; // wrong size (3x5)

        let result = apply_dof(&image, &depth_map, &cfg, DofKernelShape::Circular, 4, 3, 3);
        assert!(
            matches!(result, Err(DofError::SizeMismatch { .. })),
            "Expected SizeMismatch error"
        );
    }

    #[test]
    fn test_apply_dof_output_size_matches_input() {
        let cfg = default_cfg();
        let w = 5_usize;
        let h = 6_usize;
        let c = 4_usize;
        let image = vec![0.5_f32; w * h * c];
        let depth_map = vec![2.0_f32; w * h];

        let output = apply_dof(&image, &depth_map, &cfg, DofKernelShape::Square, w, h, c)
            .expect("apply_dof failed");
        assert_eq!(
            output.len(),
            image.len(),
            "Output size must match input size"
        );
    }

    // ── DofStats tests ────────────────────────────────────────────────────────

    #[test]
    fn test_dof_stats_compute() {
        // CoC values: [0.0, 0.0, 5.0, 10.0]
        let coc = vec![0.0_f32, 0.0, 5.0, 10.0];
        let stats = DofStats::compute(&coc, 0.5);

        assert!(
            approx_eq(stats.mean_coc, 3.75, 1e-4),
            "mean = {}",
            stats.mean_coc
        );
        assert!(
            approx_eq(stats.max_coc, 10.0, 1e-4),
            "max = {}",
            stats.max_coc
        );
        // sharp pixels: 0.0, 0.0 (< 0.5) → 2/4 = 0.5
        assert!(
            approx_eq(stats.sharp_pixel_fraction, 0.5, 1e-4),
            "sharp_frac = {}",
            stats.sharp_pixel_fraction
        );
        assert!(
            approx_eq(stats.blurred_pixel_fraction, 0.5, 1e-4),
            "blurred_frac = {}",
            stats.blurred_pixel_fraction
        );
    }

    #[test]
    fn test_dof_stats_sharp_fraction() {
        // All pixels sharp (coc = 0.0 < threshold=0.5)
        let coc = vec![0.0_f32; 10];
        let stats = DofStats::compute(&coc, 0.5);
        assert!(approx_eq(stats.sharp_pixel_fraction, 1.0, 1e-5));
        assert!(approx_eq(stats.blurred_pixel_fraction, 0.0, 1e-5));
    }

    #[test]
    fn test_dof_stats_empty() {
        let stats = DofStats::compute(&[], 0.5);
        assert!(approx_eq(stats.mean_coc, 0.0, 1e-6));
        assert!(approx_eq(stats.max_coc, 0.0, 1e-6));
        assert!(approx_eq(stats.sharp_pixel_fraction, 1.0, 1e-6));
        assert!(approx_eq(stats.blurred_pixel_fraction, 0.0, 1e-6));
    }

    #[test]
    fn test_dof_stats_format_summary_non_empty() {
        let coc = vec![2.0_f32, 4.0_f32];
        let stats = DofStats::compute(&coc, 0.5);
        let summary = stats.format_summary();
        assert!(summary.contains("DoF:"), "summary: {summary}");
        assert!(summary.contains("mean_coc"), "summary: {summary}");
        assert!(summary.contains("max_coc"), "summary: {summary}");
    }

    #[test]
    fn test_dof_config_default() {
        let cfg = DofConfig::default();
        assert!(approx_eq(cfg.focal_distance, 2.0, 1e-6));
        assert!(approx_eq(cfg.focal_range, 0.3, 1e-6));
        assert!(approx_eq(cfg.max_blur_radius, 15.0, 1e-6));
    }

    #[test]
    fn test_apply_dof_hexagonal_kernel_runs() {
        // Smoke test: hexagonal kernel should not panic and produce correct size output.
        let cfg = DofConfig {
            focal_distance: 5.0,
            focal_range: 0.1,
            max_blur_radius: 5.0,
        };
        let w = 6_usize;
        let h = 6_usize;
        let c = 3_usize;
        let image: Vec<f32> = (0..w * h * c).map(|i| (i % 256) as f32 / 255.0).collect();
        let depth_map = vec![1.0_f32; w * h]; // far from focal distance 5.0 → blurred

        let output = apply_dof(&image, &depth_map, &cfg, DofKernelShape::Hexagonal, w, h, c)
            .expect("Hexagonal apply_dof failed");
        assert_eq!(output.len(), image.len());
    }

    #[test]
    fn test_coc_error_zero_dimension() {
        let cfg = default_cfg();
        let result = compute_coc(&[1.0], &cfg, 0, 1);
        assert!(matches!(result, Err(DofError::ZeroDimension)));
    }

    #[test]
    fn test_apply_dof_zero_channels_error() {
        let cfg = default_cfg();
        let image = vec![0.5_f32; 4];
        let depth_map = vec![2.0_f32; 4];
        let result = apply_dof(&image, &depth_map, &cfg, DofKernelShape::Circular, 2, 2, 0);
        assert!(matches!(result, Err(DofError::ZeroChannels)));
    }

    #[test]
    fn test_compute_coc_size_mismatch_reports_correct_fields() {
        // Regression test: `image_len`/`depth_len` were previously swapped -
        // `depth_len` (documented as "flat length of the depth buffer")
        // must report the depth buffer's actual length, and `image_len`
        // must report the expected pixel count (there is no separate image
        // buffer in `compute_coc`).
        let cfg = default_cfg();
        let depth_map = vec![1.0_f32; 5]; // 3*2 = 6 expected, only 5 given
        let result = compute_coc(&depth_map, &cfg, 3, 2);
        match result {
            Err(DofError::SizeMismatch {
                image_len,
                depth_len,
            }) => {
                assert_eq!(
                    depth_len, 5,
                    "depth_len must be the depth buffer's actual length"
                );
                assert_eq!(image_len, 6, "image_len must be the expected pixel count");
            }
            other => panic!("expected SizeMismatch, got {other:?}"),
        }
    }
}
