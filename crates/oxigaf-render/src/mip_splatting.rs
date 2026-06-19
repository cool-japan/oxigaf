//! # Mip-Splatting: Alias-Free 3D Gaussian Splatting
//!
//! This module implements the mip-splatting technique from the paper
//! "Mip-Splatting: Alias-Free 3D Gaussian Splatting" (Barron et al.).
//!
//! The core idea is to adapt 3D Gaussian scales based on the imaging frequency
//! (pixel footprint) at each depth, preventing aliasing when viewing Gaussians
//! at different distances.
//!
//! ## Algorithm Overview
//!
//! 1. For each Gaussian at view-space depth z, compute the pixel footprint radius
//!    (world-space size of one pixel at that depth).
//! 2. Compute a filter variance to add to the projected 2D Gaussian covariance
//!    (EWA pre-filter), ensuring the Nyquist criterion is satisfied.
//! 3. Clamp minimum 3D scales to the pixel footprint to prevent under-sampling.
//! 4. Assign mip levels for progressive LOD.
//!
//! ## No-Unwrap Policy
//!
//! All fallible operations return `Result<T, MipSplattingError>`.

use thiserror::Error;

/// Errors that can occur during mip-splatting computations.
#[derive(Debug, Error)]
pub enum MipSplattingError {
    /// Pixel size parameter was not positive.
    #[error("Invalid pixel size: {0} (must be positive)")]
    InvalidPixelSize(f32),

    /// Scale parameter was not positive.
    #[error("Invalid scale: {0} (must be positive)")]
    InvalidScale(f32),

    /// Scales and positions have different lengths.
    #[error("Scale count {scales} does not match position count {positions}")]
    CountMismatch { scales: usize, positions: usize },

    /// Focal length parameter was not positive.
    #[error("Invalid focal length: {0} (must be positive)")]
    InvalidFocalLength(f32),
}

// ─── Filter Mode ────────────────────────────────────────────────────────────

/// Pre-filter mode used for anti-aliasing convolution.
///
/// Determines how the pixel footprint is convolved with the projected Gaussian.
#[derive(Debug, Clone, PartialEq)]
pub enum FilterMode {
    /// Gaussian pre-filter (smooth, standard mip-splatting).
    ///
    /// The filter variance is `(filter_scale * footprint_radius)²`.
    Gaussian,

    /// Box pre-filter (simpler, slightly sharper).
    ///
    /// The filter variance uses the box standard deviation:
    /// `(footprint_radius / sqrt(12))²`.
    Box,
}

// ─── MipCamera ───────────────────────────────────────────────────────────────

/// Camera parameters for mip-splatting computation.
///
/// All focal lengths are in pixels. The near/far planes gate depth clamping.
#[derive(Debug, Clone)]
pub struct MipCamera {
    /// Horizontal focal length in pixels.
    pub focal_length_x: f32,
    /// Vertical focal length in pixels.
    pub focal_length_y: f32,
    /// Image width in pixels.
    pub image_width: u32,
    /// Image height in pixels.
    pub image_height: u32,
    /// Near clip plane distance (default 0.01).
    pub near_plane: f32,
    /// Far clip plane distance (default 100.0).
    pub far_plane: f32,
}

impl MipCamera {
    /// Create a new camera from explicit focal lengths and image dimensions.
    ///
    /// # Errors
    ///
    /// Returns [`MipSplattingError::InvalidFocalLength`] if either focal length
    /// is not strictly positive.
    pub fn new(fx: f32, fy: f32, width: u32, height: u32) -> Result<Self, MipSplattingError> {
        if fx <= 0.0 {
            return Err(MipSplattingError::InvalidFocalLength(fx));
        }
        if fy <= 0.0 {
            return Err(MipSplattingError::InvalidFocalLength(fy));
        }
        Ok(Self {
            focal_length_x: fx,
            focal_length_y: fy,
            image_width: width,
            image_height: height,
            near_plane: 0.01,
            far_plane: 100.0,
        })
    }

    /// Derive focal lengths from a vertical field-of-view angle and image size.
    ///
    /// `fy = (height / 2) / tan(fov_y_rad / 2)`, `fx = fy * (width / height)`.
    ///
    /// # Errors
    ///
    /// Returns [`MipSplattingError::InvalidFocalLength`] if the resulting focal
    /// length is not positive (e.g., FoV ≥ π).
    pub fn from_fov(fov_y_rad: f32, width: u32, height: u32) -> Result<Self, MipSplattingError> {
        let half_tan = (fov_y_rad * 0.5).tan();
        if half_tan <= 0.0 {
            return Err(MipSplattingError::InvalidFocalLength(0.0));
        }
        let fy = (height as f32 * 0.5) / half_tan;
        let fx = fy * (width as f32 / height as f32);
        Self::new(fx, fy, width, height)
    }

    /// Pixel angular size at depth `depth`.
    ///
    /// The pixel size (radians ≈ world-units per pixel at unit depth) is:
    /// `max(image_width, image_height) / (focal_length_x * depth)`.
    ///
    /// The depth argument is here for API symmetry — the raw ratio
    /// `1/focal_length_x` is the per-unit-depth pixel size; callers that need
    /// the actual world footprint should use [`Self::pixel_footprint_radius`].
    pub fn pixel_size_at_depth(&self, depth: f32) -> f32 {
        let max_dim = self.image_width.max(self.image_height) as f32;
        let effective_depth = depth.max(f32::EPSILON);
        max_dim / (self.focal_length_x * effective_depth)
    }

    /// World-space footprint radius of one pixel at `depth`.
    ///
    /// `footprint_radius = depth / focal_length_x`
    ///
    /// This is the simplified, square-pixel approximation used throughout
    /// the mip-splatting algorithm.
    pub fn pixel_footprint_radius(&self, depth: f32) -> f32 {
        depth / self.focal_length_x
    }
}

// ─── MipConfig ───────────────────────────────────────────────────────────────

/// Configuration for the mip-splatting scale adjustment pipeline.
#[derive(Debug, Clone)]
pub struct MipConfig {
    /// Pre-filter mode for the pixel footprint convolution.
    pub filter_mode: FilterMode,

    /// Scale factor applied to the pixel footprint when computing the
    /// minimum Gaussian σ.  1.0 = exact Nyquist, 0.3 = 30 % of footprint.
    pub filter_scale: f32,

    /// Whether to clamp each Gaussian's minimum scale to the pixel footprint.
    pub clamp_to_footprint: bool,

    /// Maximum allowed ratio between the adjusted and original scale.
    /// E.g. 4.0 means up to 4× scale increase is permitted.
    pub max_scale_ratio: f32,
}

impl Default for MipConfig {
    fn default() -> Self {
        Self {
            filter_mode: FilterMode::Gaussian,
            filter_scale: 0.3,
            clamp_to_footprint: true,
            max_scale_ratio: 4.0,
        }
    }
}

// ─── MipSplattingStats ───────────────────────────────────────────────────────

/// Diagnostics produced by a mip-splatting pass.
#[derive(Debug, Clone)]
pub struct MipSplattingStats {
    /// Total number of Gaussians processed.
    pub num_gaussians: usize,
    /// Number of Gaussians whose scale was increased by clamping.
    pub num_clamped: usize,
    /// Mean per-Gaussian ratio (adjusted / original scale), geometric mean of
    /// the largest adjusted axis vs its original.
    pub mean_scale_ratio: f32,
    /// Maximum adjustment ratio observed across all Gaussians and axes.
    pub max_scale_ratio: f32,
    /// Count of Gaussians at each mip level.
    pub mip_level_histogram: Vec<u32>,
}

// ─── Core computations ───────────────────────────────────────────────────────

/// Compute the 2D isotropic filter variance added to a projected Gaussian.
///
/// The filter variance σ²_filter is:
/// - **Gaussian mode**: `(filter_scale * footprint_radius)²`
/// - **Box mode**: `(footprint_radius / sqrt(12))²`
///
/// # Errors
///
/// Returns [`MipSplattingError::InvalidPixelSize`] if `depth ≤ 0`.
pub fn compute_filter_variance(
    camera: &MipCamera,
    depth: f32,
    config: &MipConfig,
) -> Result<f32, MipSplattingError> {
    if depth <= 0.0 {
        return Err(MipSplattingError::InvalidPixelSize(depth));
    }
    let footprint = camera.pixel_footprint_radius(depth);
    let variance = match config.filter_mode {
        FilterMode::Gaussian => {
            let sigma = config.filter_scale * footprint;
            sigma * sigma
        }
        FilterMode::Box => {
            // Box distribution σ = width / sqrt(12)
            let sigma = footprint / 12.0_f32.sqrt();
            sigma * sigma
        }
    };
    Ok(variance)
}

/// Adjust log-scale 3D Gaussian parameters for mip-splatting anti-aliasing.
///
/// For each Gaussian `i` at view-space position `positions_view[i]`:
/// 1. Extract depth as `positions_view[i][2]`, clamped to `near_plane`.
/// 2. Compute the minimum σ from the pixel footprint: `σ_min = footprint_radius * filter_scale`.
/// 3. For each scale axis, clamp: `scale_log = max(scale_log, ln(σ_min))`.
/// 4. Apply `max_scale_ratio` ceiling: `scale_log = min(scale_log, original + ln(max_scale_ratio))`.
///
/// # Errors
///
/// Returns [`MipSplattingError::CountMismatch`] when `scales_log` and
/// `positions_view` have different lengths.
pub fn adjust_scales_for_mip(
    scales_log: &[[f32; 3]],
    positions_view: &[[f32; 3]],
    camera: &MipCamera,
    config: &MipConfig,
) -> Result<Vec<[f32; 3]>, MipSplattingError> {
    if scales_log.len() != positions_view.len() {
        return Err(MipSplattingError::CountMismatch {
            scales: scales_log.len(),
            positions: positions_view.len(),
        });
    }

    let mut adjusted = Vec::with_capacity(scales_log.len());

    for (scale_log, pos) in scales_log.iter().zip(positions_view.iter()) {
        let depth = pos[2].max(camera.near_plane);
        let footprint = camera.pixel_footprint_radius(depth);
        let sigma_min = footprint * config.filter_scale;

        // ln(σ_min) — floor for each log-scale component
        let log_sigma_min = if sigma_min > 0.0 {
            sigma_min.ln()
        } else {
            f32::NEG_INFINITY
        };

        let max_ratio_log = config.max_scale_ratio.ln();

        let mut out = [0.0f32; 3];
        for axis in 0..3 {
            let original = scale_log[axis];
            let mut s = if config.clamp_to_footprint {
                original.max(log_sigma_min)
            } else {
                original
            };
            // Apply max_scale_ratio ceiling relative to the original
            s = s.min(original + max_ratio_log);
            out[axis] = s;
        }
        adjusted.push(out);
    }

    Ok(adjusted)
}

/// Compute the EWA (Elliptical Weighted Average) 2×2 pre-filter matrix.
///
/// Given the pinhole camera Jacobian J at a 3D point `position_view = [x, y, z]`:
///
/// ```text
/// J = [[fx/z,  0,    -fx*x/z²],
///      [0,     fy/z, -fy*y/z²]]
/// ```
///
/// The 2×2 EWA filter matrix is `σ²_filter * J * J^T`, returned row-major as
/// `[m00, m01, m10, m11]`.
///
/// On-axis (x=0, y=0) the matrix is diagonal; off-axis it has non-zero
/// off-diagonal terms proportional to `fx·fy·x·y / z⁴`.
///
/// # Errors
///
/// Returns [`MipSplattingError::InvalidPixelSize`] if `position_view[2] ≤ 0`.
pub fn compute_ewa_filter(
    camera: &MipCamera,
    position_view: [f32; 3],
    config: &MipConfig,
) -> Result<[f32; 4], MipSplattingError> {
    let (x, y, z) = (position_view[0], position_view[1], position_view[2]);
    if z <= 0.0 {
        return Err(MipSplattingError::InvalidPixelSize(z));
    }

    let depth = z;
    let filter_var = compute_filter_variance(camera, depth, config)?;

    let fx = camera.focal_length_x;
    let fy = camera.focal_length_y;
    let z2 = z * z;

    // Jacobian rows
    let j00 = fx / z;
    let j02 = -fx * x / z2;
    let j11 = fy / z;
    let j12 = -fy * y / z2;

    // J * J^T (2×2):
    //   m00 = j00² + j02²    (row 0 · row 0)
    //   m01 = j00*0 + j02*j12 = j02*j12  (row 0 · row 1, where j01=0, j10=0)
    //   m10 = m01             (symmetric)
    //   m11 = j11² + j12²    (row 1 · row 1)
    let jjt00 = j00 * j00 + j02 * j02;
    let jjt01 = j02 * j12; // j00*j10 = 0, j01*j11 = 0
    let jjt11 = j11 * j11 + j12 * j12;

    Ok([
        filter_var * jjt00,
        filter_var * jjt01,
        filter_var * jjt01, // symmetric
        filter_var * jjt11,
    ])
}

/// Compute the mip level for a single Gaussian at a given depth.
///
/// `mip_level = clamp(floor(log2(depth / (focal_length_x * gaussian_scale))), 0, max_mip_levels)`
///
/// Intuitively: mip 0 when the Gaussian projects to ≥ 1 pixel; higher mip
/// when the Gaussian is far away or intrinsically small.
///
/// # Errors
///
/// Returns [`MipSplattingError::InvalidScale`] if `gaussian_scale ≤ 0`.
pub fn compute_mip_level(
    depth: f32,
    gaussian_scale: f32,
    camera: &MipCamera,
    max_mip_levels: u32,
) -> Result<u32, MipSplattingError> {
    if gaussian_scale <= 0.0 {
        return Err(MipSplattingError::InvalidScale(gaussian_scale));
    }

    let effective_depth = depth.max(camera.near_plane);
    let base = camera.focal_length_x * gaussian_scale;
    let ratio = effective_depth / base;

    let mip_f = if ratio > 0.0 { ratio.log2() } else { 0.0 };
    let mip = mip_f.floor().max(0.0) as u32;
    Ok(mip.min(max_mip_levels))
}

/// Batch-compute mip levels for all Gaussians.
///
/// The representative scale for each Gaussian is the geometric mean of the
/// exponentiated log-scales: `exp(mean(scale_log))`.
///
/// # Errors
///
/// Returns [`MipSplattingError::CountMismatch`] when the slice lengths differ.
pub fn compute_mip_levels_batch(
    scales_log: &[[f32; 3]],
    positions_view: &[[f32; 3]],
    camera: &MipCamera,
    max_mip_levels: u32,
) -> Result<Vec<u32>, MipSplattingError> {
    if scales_log.len() != positions_view.len() {
        return Err(MipSplattingError::CountMismatch {
            scales: scales_log.len(),
            positions: positions_view.len(),
        });
    }

    let mut levels = Vec::with_capacity(scales_log.len());

    for (scale_log, pos) in scales_log.iter().zip(positions_view.iter()) {
        let mean_log = (scale_log[0] + scale_log[1] + scale_log[2]) / 3.0;
        let gaussian_scale = mean_log.exp();
        let depth = pos[2].max(camera.near_plane);
        let level = compute_mip_level(depth, gaussian_scale, camera, max_mip_levels)?;
        levels.push(level);
    }

    Ok(levels)
}

/// Compute aggregate statistics for a mip-splatting pass.
///
/// Compares `original_scales` against `adjusted_scales` to count how many
/// Gaussians were clamped upward and by how much.  Mip level histograms are
/// computed from `mip_levels`.
///
/// The mean scale ratio is the arithmetic mean of the max-axis ratio per Gaussian.
pub fn compute_mip_stats(
    original_scales: &[[f32; 3]],
    adjusted_scales: &[[f32; 3]],
    mip_levels: &[u32],
) -> MipSplattingStats {
    let num_gaussians = original_scales.len();
    let mut num_clamped = 0usize;
    let mut sum_ratio = 0.0f32;
    let mut max_ratio = 1.0f32;

    for (orig, adj) in original_scales.iter().zip(adjusted_scales.iter()) {
        let mut clamped_this = false;
        let mut best_ratio = 1.0f32;

        for axis in 0..3 {
            let orig_scale = orig[axis].exp();
            let adj_scale = adj[axis].exp();
            let ratio = if orig_scale > 0.0 {
                adj_scale / orig_scale
            } else {
                1.0
            };
            // Track largest axis ratio for this Gaussian
            if ratio > best_ratio {
                best_ratio = ratio;
            }
            if adj_scale > orig_scale * (1.0 + f32::EPSILON) {
                clamped_this = true;
            }
        }

        sum_ratio += best_ratio;
        if best_ratio > max_ratio {
            max_ratio = best_ratio;
        }
        if clamped_this {
            num_clamped += 1;
        }
    }

    let mean_scale_ratio = if num_gaussians > 0 {
        sum_ratio / num_gaussians as f32
    } else {
        1.0
    };

    // Mip level histogram
    let max_level = mip_levels.iter().copied().max().unwrap_or(0) as usize;
    let mut histogram = vec![0u32; max_level + 1];
    for &level in mip_levels {
        let idx = level as usize;
        if idx < histogram.len() {
            histogram[idx] += 1;
        }
    }

    MipSplattingStats {
        num_gaussians,
        num_clamped,
        mean_scale_ratio,
        max_scale_ratio: max_ratio,
        mip_level_histogram: histogram,
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn test_camera() -> MipCamera {
        MipCamera::new(500.0, 500.0, 512, 512).unwrap()
    }

    fn test_config() -> MipConfig {
        MipConfig::default()
    }

    // 1. MipCamera::new with valid params succeeds
    #[test]
    fn test_camera_new_valid() {
        let cam = MipCamera::new(500.0, 500.0, 512, 512);
        assert!(cam.is_ok());
        let cam = cam.unwrap();
        assert_eq!(cam.image_width, 512);
        assert_eq!(cam.image_height, 512);
        assert!((cam.focal_length_x - 500.0).abs() < 1e-6);
    }

    // 2. MipCamera::new with negative fx → InvalidFocalLength
    #[test]
    fn test_camera_new_negative_fx() {
        let result = MipCamera::new(-1.0, 500.0, 512, 512);
        assert!(matches!(result, Err(MipSplattingError::InvalidFocalLength(v)) if v == -1.0));
    }

    // 3. from_fov with 60° FoV produces reasonable focal length (~886 for 1024px)
    #[test]
    fn test_from_fov_60_degrees() {
        let fov_y = std::f32::consts::PI / 3.0; // 60°
        let cam = MipCamera::from_fov(fov_y, 1024, 1024).unwrap();
        // fy = 512 / tan(30°) ≈ 512 / 0.57735 ≈ 886.7
        assert!(
            (cam.focal_length_y - 886.7).abs() < 1.0,
            "fy={}",
            cam.focal_length_y
        );
    }

    // 4. pixel_footprint_radius at depth 1.0: proportional to 1/focal_length
    #[test]
    fn test_footprint_radius_depth_1() {
        let cam = test_camera();
        let r = cam.pixel_footprint_radius(1.0);
        // footprint = depth / fx = 1.0 / 500.0 = 0.002
        assert!((r - 0.002).abs() < 1e-6, "r={}", r);
    }

    // 5. pixel_footprint_radius at depth 2.0 is 2× depth 1.0
    #[test]
    fn test_footprint_radius_depth_scaling() {
        let cam = test_camera();
        let r1 = cam.pixel_footprint_radius(1.0);
        let r2 = cam.pixel_footprint_radius(2.0);
        assert!((r2 / r1 - 2.0).abs() < 1e-6, "ratio={}", r2 / r1);
    }

    // 6. compute_filter_variance Gaussian mode returns positive value
    #[test]
    fn test_filter_variance_gaussian_positive() {
        let cam = test_camera();
        let config = MipConfig {
            filter_mode: FilterMode::Gaussian,
            ..test_config()
        };
        let v = compute_filter_variance(&cam, 1.0, &config).unwrap();
        assert!(v > 0.0, "variance={}", v);
    }

    // 7. compute_filter_variance Box mode returns positive value (different from Gaussian)
    #[test]
    fn test_filter_variance_box_positive_and_different() {
        let cam = test_camera();
        let config_g = MipConfig {
            filter_mode: FilterMode::Gaussian,
            ..test_config()
        };
        let config_b = MipConfig {
            filter_mode: FilterMode::Box,
            ..test_config()
        };
        let v_gauss = compute_filter_variance(&cam, 1.0, &config_g).unwrap();
        let v_box = compute_filter_variance(&cam, 1.0, &config_b).unwrap();
        assert!(v_box > 0.0, "box variance={}", v_box);
        // Box uses 1/sqrt(12) factor, Gaussian uses filter_scale=0.3; they differ
        assert!(
            (v_gauss - v_box).abs() > 1e-8,
            "should differ: gauss={} box={}",
            v_gauss,
            v_box
        );
    }

    // 8. compute_filter_variance at depth 0.1 < at depth 10.0
    #[test]
    fn test_filter_variance_increases_with_depth() {
        let cam = test_camera();
        let config = test_config();
        let v_near = compute_filter_variance(&cam, 0.1, &config).unwrap();
        let v_far = compute_filter_variance(&cam, 10.0, &config).unwrap();
        assert!(v_near < v_far, "near={} far={}", v_near, v_far);
    }

    // 9. adjust_scales_for_mip with already-large scales: no clamping, returns same
    #[test]
    fn test_adjust_scales_large_no_clamping() {
        let cam = test_camera();
        let config = test_config();
        // Large scales: exp(2.0) ≈ 7.39 — much bigger than footprint at depth 1
        let scales_log = vec![[2.0f32, 2.0, 2.0]];
        let positions_view = vec![[0.0f32, 0.0, 1.0]];

        let adjusted = adjust_scales_for_mip(&scales_log, &positions_view, &cam, &config).unwrap();
        assert_eq!(adjusted.len(), 1);
        for axis in 0..3 {
            assert!(
                (adjusted[0][axis] - scales_log[0][axis]).abs() < 1e-5,
                "axis {}: orig={} adj={}",
                axis,
                scales_log[0][axis],
                adjusted[0][axis]
            );
        }
    }

    // 10. adjust_scales_for_mip with very small scales at shallow depth: scale increased
    #[test]
    fn test_adjust_scales_small_gets_clamped() {
        let cam = test_camera();
        let config = test_config();
        // Very small scale: exp(-10) ≈ 4.5e-5 — far below footprint at depth 1
        let scales_log = vec![[-10.0f32, -10.0, -10.0]];
        let positions_view = vec![[0.0f32, 0.0, 1.0]];

        let adjusted = adjust_scales_for_mip(&scales_log, &positions_view, &cam, &config).unwrap();
        // At least one axis should be increased
        let any_increased = (0..3).any(|ax| adjusted[0][ax] > scales_log[0][ax] + 1e-6);
        assert!(
            any_increased,
            "expected scale clamping, got {:?}",
            adjusted[0]
        );
    }

    // 11. adjust_scales_for_mip count mismatch → CountMismatch error
    #[test]
    fn test_adjust_scales_count_mismatch() {
        let cam = test_camera();
        let config = test_config();
        let scales_log = vec![[0.0f32; 3]; 3];
        let positions_view = vec![[0.0f32; 3]; 5];
        let result = adjust_scales_for_mip(&scales_log, &positions_view, &cam, &config);
        assert!(matches!(
            result,
            Err(MipSplattingError::CountMismatch {
                scales: 3,
                positions: 5
            })
        ));
    }

    // 12. compute_ewa_filter returns non-trivial matrix for off-center position
    #[test]
    fn test_ewa_filter_off_center() {
        let cam = test_camera();
        let config = test_config();
        // Off-axis position: x ≠ 0 → off-diagonal terms should be non-zero
        let mat = compute_ewa_filter(&cam, [1.0, 1.0, 2.0], &config).unwrap();
        // mat is [m00, m01, m10, m11]; m01 = m10 ≠ 0 for off-axis
        assert!(mat[0] > 0.0, "m00 should be positive");
        assert!(mat[3] > 0.0, "m11 should be positive");
        // off-diagonal: -fx*x/z² * (-fy*y/z²) * filter_var ≠ 0
        assert!(
            mat[1].abs() > 0.0,
            "off-diagonal m01 should be non-zero for off-center"
        );
    }

    // 13. compute_ewa_filter at on-axis position [0,0,z]: diagonal matrix
    #[test]
    fn test_ewa_filter_on_axis_diagonal() {
        let cam = test_camera();
        let config = test_config();
        // On-axis: x=0, y=0 → j02=0, j12=0 → off-diagonal is 0
        let mat = compute_ewa_filter(&cam, [0.0, 0.0, 1.0], &config).unwrap();
        // m01 = m10 = filter_var * j02 * j12 = 0
        assert!(
            mat[1].abs() < 1e-10,
            "m01 should be 0 on-axis, got {}",
            mat[1]
        );
        assert!(
            mat[2].abs() < 1e-10,
            "m10 should be 0 on-axis, got {}",
            mat[2]
        );
        assert!(mat[0] > 0.0, "m00 should be positive");
        assert!(mat[3] > 0.0, "m11 should be positive");
    }

    // 14. compute_mip_level depth 1.0 with large scale → mip 0
    #[test]
    fn test_mip_level_large_scale_is_zero() {
        let cam = test_camera();
        // large gaussian_scale → depth/(fx*scale) < 1 → log2 < 0 → clamped to 0
        let level = compute_mip_level(1.0, 10.0, &cam, 8).unwrap();
        assert_eq!(level, 0, "expected mip 0 for large scale");
    }

    // 15. compute_mip_level depth 100.0 with tiny scale → high mip level (clamped)
    #[test]
    fn test_mip_level_tiny_scale_clamped() {
        let cam = test_camera();
        // tiny scale: depth/(fx*scale) = 100/(500*0.0001) = 2000 → log2 ≈ 11 → clamped to 8
        let level = compute_mip_level(100.0, 0.0001, &cam, 8).unwrap();
        assert_eq!(level, 8, "expected mip clamped to max=8, got {}", level);
    }

    // 16. compute_mip_levels_batch length matches input
    #[test]
    fn test_mip_levels_batch_length() {
        let cam = test_camera();
        let n = 7;
        let scales_log: Vec<[f32; 3]> = (0..n).map(|_| [0.0f32; 3]).collect();
        let positions_view: Vec<[f32; 3]> = (0..n).map(|i| [0.0, 0.0, (i + 1) as f32]).collect();
        let levels = compute_mip_levels_batch(&scales_log, &positions_view, &cam, 8).unwrap();
        assert_eq!(levels.len(), n);
    }

    // 17. compute_mip_stats num_clamped correct when some scales increased
    #[test]
    fn test_mip_stats_num_clamped() {
        // 3 Gaussians: first 2 have adjusted > original, last is unchanged
        let orig = vec![[-5.0f32, -5.0, -5.0], [-5.0, -5.0, -5.0], [2.0, 2.0, 2.0]];
        let adj = vec![[-4.0f32, -4.0, -4.0], [-3.0, -3.0, -3.0], [2.0, 2.0, 2.0]];
        let mip_levels = vec![0u32, 1, 0];
        let stats = compute_mip_stats(&orig, &adj, &mip_levels);
        assert_eq!(
            stats.num_clamped, 2,
            "expected 2 clamped, got {}",
            stats.num_clamped
        );
    }

    // 18. compute_mip_stats mean_scale_ratio ≥ 1.0
    #[test]
    fn test_mip_stats_mean_ratio_at_least_one() {
        // adjust_scales_for_mip only increases, so ratio ≥ 1
        let cam = test_camera();
        let config = test_config();
        let scales_log: Vec<[f32; 3]> = vec![[-8.0; 3], [-6.0; 3], [2.0; 3]];
        let positions_view: Vec<[f32; 3]> = vec![[0.0, 0.0, 1.0]; 3];
        let adjusted = adjust_scales_for_mip(&scales_log, &positions_view, &cam, &config).unwrap();
        let mip_levels = vec![0u32; 3];
        let stats = compute_mip_stats(&scales_log, &adjusted, &mip_levels);
        assert!(
            stats.mean_scale_ratio >= 1.0 - 1e-5,
            "mean ratio should be ≥ 1, got {}",
            stats.mean_scale_ratio
        );
    }

    // 19. adjust_scales_for_mip max_scale_ratio clamping works
    #[test]
    fn test_adjust_scales_max_ratio_clamped() {
        let cam = test_camera();
        let config = MipConfig {
            max_scale_ratio: 2.0, // at most 2× increase
            ..test_config()
        };
        // Start from a very negative log-scale so footprint would push it far up
        let scales_log = vec![[-20.0f32, -20.0, -20.0]];
        let positions_view = vec![[0.0f32, 0.0, 1.0]];
        let adjusted = adjust_scales_for_mip(&scales_log, &positions_view, &cam, &config).unwrap();
        let max_allowed = -20.0 + 2.0_f32.ln();
        for (axis, &val) in adjusted[0].iter().enumerate().take(3) {
            assert!(
                val <= max_allowed + 1e-5,
                "axis {}: expected ≤ {} got {}",
                axis,
                max_allowed,
                val
            );
        }
    }

    // 20. compute_mip_level with zero gaussian_scale → InvalidScale error
    #[test]
    fn test_mip_level_zero_scale_error() {
        let cam = test_camera();
        let result = compute_mip_level(1.0, 0.0, &cam, 8);
        assert!(matches!(result, Err(MipSplattingError::InvalidScale(v)) if v == 0.0));
    }
}
