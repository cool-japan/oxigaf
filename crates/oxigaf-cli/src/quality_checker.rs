//! Automated image quality assessment for rendered OxiGAF outputs.
//!
//! This module provides:
//! - [`ImageQualityMetrics`] — PSNR, SSIM, MSE, MAE, max-error per image pair
//! - [`ArtifactReport`] — clipping, color drift, noise, banding detection
//! - [`QualityReport`] — combined metrics + artifacts with pass/fail verdict
//! - [`BatchQualityReport`] — aggregate stats across many rendered frames
//!
//! All images are passed as flat `Vec<u8>` / `&[u8]` in RGBA format
//! (width × height × 4 bytes), or as `&[f32]` in [0, 1] range.
//!
//! # Design
//!
//! The module never uses `unwrap`, `expect`, `rand`, or `ndarray`.
//! All math operates on manual `Vec<f32>` / slice arithmetic.
//! Errors are typed via [`QualityError`] (using `thiserror`).

use std::fmt;
use thiserror::Error;

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

/// Errors that can occur during quality checking operations.
#[derive(Debug, Error)]
pub enum QualityError {
    /// Image dimensions are inconsistent between reference and rendered image.
    #[error(
        "Image dimensions mismatch: expected {expected_w}×{expected_h}, got {actual_w}×{actual_h}"
    )]
    DimensionMismatch {
        expected_w: u32,
        expected_h: u32,
        actual_w: u32,
        actual_h: u32,
    },
    /// The image contains no pixels.
    #[error("Empty image: 0 pixels")]
    EmptyImage,
    /// A threshold value is not valid.
    #[error("Invalid PSNR threshold {threshold}: must be > 0")]
    InvalidThreshold { threshold: f32 },
    /// A batch of images contains no entries.
    #[error("Batch is empty: no images to check")]
    EmptyBatch,
    /// An I/O error occurred.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

// ---------------------------------------------------------------------------
// ImageQualityMetrics
// ---------------------------------------------------------------------------

/// Per-image quality metrics computed between a reference and a rendered image.
#[derive(Debug, Clone)]
pub struct ImageQualityMetrics {
    /// Peak signal-to-noise ratio in dB; [`f32::INFINITY`] when images are identical.
    pub psnr: f32,
    /// Mean squared error in [0, 1] range.
    pub mse: f32,
    /// Mean absolute error in [0, 1] range.
    pub mae: f32,
    /// Simplified structural similarity index in [0, 1].
    pub ssim: f32,
    /// Maximum per-pixel absolute error in [0, 1].
    pub max_error: f32,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
}

impl ImageQualityMetrics {
    /// Returns `true` when both PSNR and SSIM satisfy the given thresholds.
    pub fn passes_threshold(&self, min_psnr: f32, min_ssim: f32) -> bool {
        self.psnr >= min_psnr && self.ssim >= min_ssim
    }

    /// Returns a single-line human-readable summary of the key metrics.
    pub fn format_summary(&self) -> String {
        format!(
            "{}×{} | PSNR: {:.2} dB | SSIM: {:.4} | MSE: {:.6} | MAE: {:.6} | MaxErr: {:.4}",
            self.width,
            self.height,
            if self.psnr.is_infinite() {
                999.99_f32
            } else {
                self.psnr
            },
            self.ssim,
            self.mse,
            self.mae,
            self.max_error,
        )
    }
}

impl fmt::Display for ImageQualityMetrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.format_summary())
    }
}

// ---------------------------------------------------------------------------
// ArtifactReport
// ---------------------------------------------------------------------------

/// Detection results for common rendering artifacts.
#[derive(Debug, Clone)]
pub struct ArtifactReport {
    /// Whether any pixel is fully saturated (component == 0 or component == 255).
    pub has_clipping: bool,
    /// Fraction of pixels that contain at least one clipped component.
    pub clipping_fraction: f32,
    /// Whether mean channel values deviate significantly from neutral gray.
    pub has_color_drift: bool,
    /// Maximum deviation of a channel mean from 128, normalised to [0, 1].
    pub color_drift_magnitude: f32,
    /// Whether estimated local noise exceeds the threshold.
    pub has_excessive_noise: bool,
    /// Estimated noise level: mean std-dev of 3×3 luminance neighbourhoods.
    pub noise_level: f32,
    /// Whether horizontal or vertical banding artefacts are present.
    pub has_banding: bool,
    /// Ratio of row/column variance to overall pixel variance (high → banding).
    pub banding_score: f32,
    /// Overall artefact score: 0 = clean, 1 = heavily artefacted.
    pub overall_score: f32,
}

// ---------------------------------------------------------------------------
// QualityThresholds
// ---------------------------------------------------------------------------

/// Thresholds used to decide whether a rendered image passes quality control.
#[derive(Debug, Clone)]
pub struct QualityThresholds {
    /// Minimum acceptable PSNR in dB (default: 25.0).
    pub min_psnr: f32,
    /// Minimum acceptable SSIM (default: 0.85).
    pub min_ssim: f32,
    /// Maximum acceptable fraction of clipped pixels (default: 0.05 = 5 %).
    pub max_clipping_pct: f32,
    /// Maximum acceptable estimated noise level (default: 0.05).
    pub max_noise_level: f32,
}

impl Default for QualityThresholds {
    fn default() -> Self {
        Self {
            min_psnr: 25.0,
            min_ssim: 0.85,
            max_clipping_pct: 0.05,
            max_noise_level: 0.05,
        }
    }
}

// ---------------------------------------------------------------------------
// QualityReport
// ---------------------------------------------------------------------------

/// Full quality report for a single (reference, rendered) image pair.
#[derive(Debug, Clone)]
pub struct QualityReport {
    /// Quantitative image quality metrics.
    pub metrics: ImageQualityMetrics,
    /// Rendering artifact detection results.
    pub artifacts: ArtifactReport,
    /// Whether the image passes all quality thresholds.
    pub passed: bool,
    /// Human-readable list of detected issues (empty if none).
    pub issues: Vec<String>,
}

impl QualityReport {
    /// Renders a multi-line, human-readable quality report.
    pub fn format_report(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!(
            "Quality Report [{}]",
            if self.passed { "PASS" } else { "FAIL" }
        ));
        lines.push(format!("  Metrics : {}", self.metrics.format_summary()));
        lines.push(format!(
            "  Clipping: {:.2}% ({})",
            self.artifacts.clipping_fraction * 100.0,
            if self.artifacts.has_clipping {
                "detected"
            } else {
                "clean"
            }
        ));
        lines.push(format!(
            "  Noise   : {:.5} ({})",
            self.artifacts.noise_level,
            if self.artifacts.has_excessive_noise {
                "excessive"
            } else {
                "ok"
            }
        ));
        lines.push(format!(
            "  Banding : {:.4} ({})",
            self.artifacts.banding_score,
            if self.artifacts.has_banding {
                "detected"
            } else {
                "ok"
            }
        ));
        lines.push(format!(
            "  Drift   : {:.4} ({})",
            self.artifacts.color_drift_magnitude,
            if self.artifacts.has_color_drift {
                "detected"
            } else {
                "ok"
            }
        ));
        if !self.issues.is_empty() {
            lines.push("  Issues:".to_string());
            for issue in &self.issues {
                lines.push(format!("    - {}", issue));
            }
        }
        lines.join("\n")
    }
}

// ---------------------------------------------------------------------------
// BatchQualityReport
// ---------------------------------------------------------------------------

/// Aggregate quality report across a batch of rendered images.
#[derive(Debug, Clone)]
pub struct BatchQualityReport {
    /// Total number of images in the batch.
    pub total_images: usize,
    /// Number of images that passed all thresholds.
    pub passed_count: usize,
    /// Number of images that failed at least one threshold.
    pub failed_count: usize,
    /// Arithmetic mean of PSNR across all images.
    pub mean_psnr: f32,
    /// Minimum PSNR across all images.
    pub min_psnr: f32,
    /// Maximum PSNR across all images.
    pub max_psnr: f32,
    /// Arithmetic mean of SSIM across all images.
    pub mean_ssim: f32,
    /// Per-image reports, one per input pair.
    pub reports: Vec<QualityReport>,
}

impl BatchQualityReport {
    /// Fraction of images that passed (in [0, 1]; `0.0` for empty batches).
    pub fn pass_rate(&self) -> f32 {
        if self.total_images == 0 {
            0.0
        } else {
            self.passed_count as f32 / self.total_images as f32
        }
    }

    /// Returns a single-line summary suitable for logging.
    pub fn format_summary(&self) -> String {
        format!(
            "Batch: {}/{} passed ({:.1}%) | PSNR mean={:.2} min={:.2} max={:.2} | SSIM mean={:.4}",
            self.passed_count,
            self.total_images,
            self.pass_rate() * 100.0,
            self.mean_psnr,
            self.min_psnr,
            self.max_psnr,
            self.mean_ssim,
        )
    }
}

// ---------------------------------------------------------------------------
// Free functions — colour conversion
// ---------------------------------------------------------------------------

/// Converts an RGBA `u8` slice to an interleaved RGB `f32` slice in [0, 1].
///
/// Input length must be a multiple of 4. Alpha is discarded.
/// Output length = `pixels.len() / 4 * 3`.
pub fn rgba_u8_to_rgb_f32(pixels: &[u8]) -> Vec<f32> {
    let n_pixels = pixels.len() / 4;
    let mut out = Vec::with_capacity(n_pixels * 3);
    for chunk in pixels.chunks_exact(4) {
        out.push(chunk[0] as f32 / 255.0);
        out.push(chunk[1] as f32 / 255.0);
        out.push(chunk[2] as f32 / 255.0);
    }
    out
}

/// Converts interleaved RGB f32 pixels to luminance (one value per pixel).
///
/// Uses the Rec. 601 luma coefficients: Y = 0.299·R + 0.587·G + 0.114·B.
fn rgb_f32_to_luminance(rgb: &[f32]) -> Vec<f32> {
    let n = rgb.len() / 3;
    let mut lum = Vec::with_capacity(n);
    for i in 0..n {
        let r = rgb[i * 3];
        let g = rgb[i * 3 + 1];
        let b = rgb[i * 3 + 2];
        lum.push(0.299 * r + 0.587 * g + 0.114 * b);
    }
    lum
}

// ---------------------------------------------------------------------------
// Free functions — basic metrics
// ---------------------------------------------------------------------------

/// Computes the mean squared error (MSE) between two f32 slices.
///
/// Both slices must have the same length. Values are expected in [0, 1].
pub fn compute_mse(a: &[f32], b: &[f32]) -> Result<f32, QualityError> {
    if a.is_empty() {
        return Err(QualityError::EmptyImage);
    }
    if a.len() != b.len() {
        return Err(QualityError::DimensionMismatch {
            expected_w: a.len() as u32,
            expected_h: 1,
            actual_w: b.len() as u32,
            actual_h: 1,
        });
    }
    let sum: f32 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum();
    Ok(sum / a.len() as f32)
}

/// Computes the mean absolute error (MAE) between two f32 slices.
///
/// Both slices must have the same length. Values are expected in [0, 1].
pub fn compute_mae(a: &[f32], b: &[f32]) -> Result<f32, QualityError> {
    if a.is_empty() {
        return Err(QualityError::EmptyImage);
    }
    if a.len() != b.len() {
        return Err(QualityError::DimensionMismatch {
            expected_w: a.len() as u32,
            expected_h: 1,
            actual_w: b.len() as u32,
            actual_h: 1,
        });
    }
    let sum: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum();
    Ok(sum / a.len() as f32)
}

/// Computes PSNR from a pre-computed MSE value.
///
/// Uses MAX_PIXEL = 1.0 (f32 range).
/// Returns [`f32::INFINITY`] when `mse == 0`.
pub fn psnr_from_mse(mse: f32) -> f32 {
    if mse == 0.0 {
        f32::INFINITY
    } else {
        10.0 * (1.0_f32 / mse).log10()
    }
}

/// Computes PSNR between two f32 images.
///
/// Returns [`f32::INFINITY`] when the images are identical.
pub fn compute_psnr(a: &[f32], b: &[f32]) -> Result<f32, QualityError> {
    let mse = compute_mse(a, b)?;
    Ok(psnr_from_mse(mse))
}

// ---------------------------------------------------------------------------
// Free functions — SSIM
// ---------------------------------------------------------------------------

/// Computes a simplified SSIM between two f32 images.
///
/// Images are converted to luminance first.  Patches of 8×8 pixels are
/// evaluated with a stride of 4 in both x and y.
///
/// Returns `Ok(1.0)` when the image is too small to produce ≥ 4 patches.
/// Returns an error when the inputs disagree in length or are empty.
pub fn compute_ssim(
    a: &[f32],
    b: &[f32],
    width: u32,
    height: u32,
    channels: u32,
) -> Result<f32, QualityError> {
    let n_pixels = (width * height) as usize;
    let expected_len = n_pixels * channels as usize;
    if expected_len == 0 {
        return Err(QualityError::EmptyImage);
    }
    if a.len() != expected_len {
        return Err(QualityError::DimensionMismatch {
            expected_w: width,
            expected_h: height,
            actual_w: (a.len() / channels.max(1) as usize) as u32,
            actual_h: 1,
        });
    }
    if b.len() != expected_len {
        return Err(QualityError::DimensionMismatch {
            expected_w: width,
            expected_h: height,
            actual_w: (b.len() / channels.max(1) as usize) as u32,
            actual_h: 1,
        });
    }

    // Convert to luminance.
    let lum_a: Vec<f32> = if channels == 1 {
        a.to_vec()
    } else if channels == 3 {
        rgb_f32_to_luminance(a)
    } else {
        // RGBA or other: treat first 3 as RGB
        let rgb_a: Vec<f32> = a
            .chunks(channels as usize)
            .flat_map(|c| [c[0], c[1], c[2]])
            .collect();
        rgb_f32_to_luminance(&rgb_a)
    };
    let lum_b: Vec<f32> = if channels == 1 {
        b.to_vec()
    } else if channels == 3 {
        rgb_f32_to_luminance(b)
    } else {
        let rgb_b: Vec<f32> = b
            .chunks(channels as usize)
            .flat_map(|c| [c[0], c[1], c[2]])
            .collect();
        rgb_f32_to_luminance(&rgb_b)
    };

    const PATCH: u32 = 8;
    const STRIDE: u32 = 4;
    const C1: f32 = 0.01 * 0.01; // (K1 * L)^2, K1=0.01, L=1
    const C2: f32 = 0.03 * 0.03; // (K2 * L)^2, K2=0.03, L=1

    let mut ssim_sum = 0.0_f32;
    let mut patch_count = 0_u32;

    let w = width;
    let h = height;

    if w < PATCH || h < PATCH {
        // Too small to generate even one complete patch.
        return Ok(1.0);
    }

    let mut py = 0_u32;
    while py + PATCH <= h {
        let mut px = 0_u32;
        while px + PATCH <= w {
            // Collect patch pixels.
            let mut sum_a = 0.0_f32;
            let mut sum_b = 0.0_f32;
            let patch_size = (PATCH * PATCH) as f32;

            for dy in 0..PATCH {
                for dx in 0..PATCH {
                    let idx = ((py + dy) * w + (px + dx)) as usize;
                    sum_a += lum_a[idx];
                    sum_b += lum_b[idx];
                }
            }
            let mu_a = sum_a / patch_size;
            let mu_b = sum_b / patch_size;

            let mut var_a = 0.0_f32;
            let mut var_b = 0.0_f32;
            let mut cov = 0.0_f32;
            for dy in 0..PATCH {
                for dx in 0..PATCH {
                    let idx = ((py + dy) * w + (px + dx)) as usize;
                    let da = lum_a[idx] - mu_a;
                    let db = lum_b[idx] - mu_b;
                    var_a += da * da;
                    var_b += db * db;
                    cov += da * db;
                }
            }
            var_a /= patch_size;
            var_b /= patch_size;
            cov /= patch_size;

            let numerator = (2.0 * mu_a * mu_b + C1) * (2.0 * cov + C2);
            let denominator = (mu_a * mu_a + mu_b * mu_b + C1) * (var_a + var_b + C2);
            ssim_sum += numerator / denominator;
            patch_count += 1;

            px += STRIDE;
        }
        py += STRIDE;
    }

    if patch_count < 4 {
        return Ok(1.0);
    }

    Ok(ssim_sum / patch_count as f32)
}

// ---------------------------------------------------------------------------
// Free functions — full metrics
// ---------------------------------------------------------------------------

/// Computes all quality metrics between a reference and a rendered RGBA image.
pub fn compute_quality_metrics(
    reference: &[u8],
    rendered: &[u8],
    width: u32,
    height: u32,
) -> Result<ImageQualityMetrics, QualityError> {
    let n_pixels = (width * height) as usize;
    if n_pixels == 0 {
        return Err(QualityError::EmptyImage);
    }
    let expected_bytes = n_pixels * 4;
    if reference.len() != expected_bytes {
        return Err(QualityError::DimensionMismatch {
            expected_w: width,
            expected_h: height,
            actual_w: (reference.len() / 4) as u32,
            actual_h: 1,
        });
    }
    if rendered.len() != expected_bytes {
        return Err(QualityError::DimensionMismatch {
            expected_w: width,
            expected_h: height,
            actual_w: (rendered.len() / 4) as u32,
            actual_h: 1,
        });
    }

    let ref_f32 = rgba_u8_to_rgb_f32(reference);
    let ren_f32 = rgba_u8_to_rgb_f32(rendered);

    let mse = compute_mse(&ref_f32, &ren_f32)?;
    let mae = compute_mae(&ref_f32, &ren_f32)?;
    let psnr = psnr_from_mse(mse);
    let ssim = compute_ssim(&ref_f32, &ren_f32, width, height, 3)?;

    let max_error = ref_f32
        .iter()
        .zip(ren_f32.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);

    Ok(ImageQualityMetrics {
        psnr,
        mse,
        mae,
        ssim,
        max_error,
        width,
        height,
    })
}

// ---------------------------------------------------------------------------
// Free functions — artifact detection
// ---------------------------------------------------------------------------

/// Detects common rendering artefacts in a single RGBA image.
pub fn detect_artifacts(
    pixels: &[u8],
    width: u32,
    height: u32,
    thresholds: &QualityThresholds,
) -> ArtifactReport {
    let n_pixels = (width * height) as usize;
    if n_pixels == 0 || pixels.len() < n_pixels * 4 {
        return ArtifactReport {
            has_clipping: false,
            clipping_fraction: 0.0,
            has_color_drift: false,
            color_drift_magnitude: 0.0,
            has_excessive_noise: false,
            noise_level: 0.0,
            has_banding: false,
            banding_score: 0.0,
            overall_score: 0.0,
        };
    }

    // --- Clipping detection ---
    let mut clipped_count = 0_usize;
    let mut sum_r = 0.0_f64;
    let mut sum_g = 0.0_f64;
    let mut sum_b = 0.0_f64;

    for chunk in pixels.chunks_exact(4) {
        let r = chunk[0];
        let g = chunk[1];
        let b = chunk[2];
        if r == 0 || r == 255 || g == 0 || g == 255 || b == 0 || b == 255 {
            clipped_count += 1;
        }
        sum_r += r as f64;
        sum_g += g as f64;
        sum_b += b as f64;
    }
    let clipping_fraction = clipped_count as f32 / n_pixels as f32;
    let has_clipping = clipping_fraction > 0.0;

    // --- Color drift ---
    let mean_r = (sum_r / n_pixels as f64) as f32;
    let mean_g = (sum_g / n_pixels as f64) as f32;
    let mean_b = (sum_b / n_pixels as f64) as f32;
    let drift_r = (mean_r - 128.0).abs() / 128.0;
    let drift_g = (mean_g - 128.0).abs() / 128.0;
    let drift_b = (mean_b - 128.0).abs() / 128.0;
    let color_drift_magnitude = drift_r.max(drift_g).max(drift_b);
    // Threshold: drift > 0.2 (20% deviation from neutral)
    let has_color_drift = color_drift_magnitude > 0.2;

    // --- Luminance image (f32) for noise/banding ---
    let lum: Vec<f32> = pixels
        .chunks_exact(4)
        .map(|c| {
            0.299 * c[0] as f32 / 255.0 + 0.587 * c[1] as f32 / 255.0 + 0.114 * c[2] as f32 / 255.0
        })
        .collect();

    // --- Noise estimation: mean std-dev of 3×3 neighbourhoods ---
    let noise_level = compute_local_noise(&lum, width, height);
    let has_excessive_noise = noise_level > thresholds.max_noise_level;

    // --- Banding detection ---
    let (banding_score, has_banding) = compute_banding_score(&lum, width, height);

    // --- Overall artefact score ---
    // Weighted sum of individual scores, clamped to [0, 1].
    let overall_score = (clipping_fraction * 0.3
        + color_drift_magnitude * 0.2
        + (noise_level / (thresholds.max_noise_level + 1e-8)).min(1.0) * 0.25
        + banding_score.min(1.0) * 0.25)
        .min(1.0);

    ArtifactReport {
        has_clipping,
        clipping_fraction,
        has_color_drift,
        color_drift_magnitude,
        has_excessive_noise,
        noise_level,
        has_banding,
        banding_score,
        overall_score,
    }
}

/// Computes the mean std-dev of 3×3 luminance neighbourhoods for all interior pixels.
fn compute_local_noise(lum: &[f32], width: u32, height: u32) -> f32 {
    if width < 3 || height < 3 {
        return 0.0;
    }
    let w = width as usize;
    let h = height as usize;
    let mut total_std = 0.0_f32;
    let mut count = 0_usize;

    for y in 1..(h - 1) {
        for x in 1..(w - 1) {
            let mut patch_sum = 0.0_f32;
            let mut patch_sq = 0.0_f32;
            for dy in 0..3_usize {
                for dx in 0..3_usize {
                    let v = lum[(y + dy - 1) * w + (x + dx - 1)];
                    patch_sum += v;
                    patch_sq += v * v;
                }
            }
            let mean = patch_sum / 9.0;
            let variance = (patch_sq / 9.0 - mean * mean).max(0.0);
            total_std += variance.sqrt();
            count += 1;
        }
    }

    if count == 0 {
        0.0
    } else {
        total_std / count as f32
    }
}

/// Computes a banding score from per-row and per-column variance.
///
/// Returns `(score, has_banding)`. `has_banding` is set when score > 0.3.
fn compute_banding_score(lum: &[f32], width: u32, height: u32) -> (f32, bool) {
    if width == 0 || height == 0 {
        return (0.0, false);
    }
    let w = width as usize;
    let h = height as usize;

    // Row means.
    let mut row_means: Vec<f32> = Vec::with_capacity(h);
    for y in 0..h {
        let row_sum: f32 = (0..w).map(|x| lum[y * w + x]).sum();
        row_means.push(row_sum / w as f32);
    }
    // Column means.
    let mut col_means: Vec<f32> = Vec::with_capacity(w);
    for x in 0..w {
        let col_sum: f32 = (0..h).map(|y| lum[y * w + x]).sum();
        col_means.push(col_sum / h as f32);
    }

    let row_var = slice_variance(&row_means);
    let col_var = slice_variance(&col_means);

    // Overall pixel variance.
    let overall_var = slice_variance(lum);

    let banding_score = row_var.max(col_var) / (overall_var + 1e-8);
    let has_banding = banding_score > 0.3;
    (banding_score, has_banding)
}

/// Population variance of a slice.
fn slice_variance(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mean: f32 = values.iter().sum::<f32>() / values.len() as f32;
    values.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / values.len() as f32
}

// ---------------------------------------------------------------------------
// Free functions — combined check
// ---------------------------------------------------------------------------

/// Runs a full quality check: metrics + artefact detection.
pub fn check_quality(
    reference: &[u8],
    rendered: &[u8],
    width: u32,
    height: u32,
    thresholds: &QualityThresholds,
) -> Result<QualityReport, QualityError> {
    let metrics = compute_quality_metrics(reference, rendered, width, height)?;
    let artifacts = detect_artifacts(rendered, width, height, thresholds);

    let mut issues = Vec::new();
    let mut passed = true;

    if metrics.psnr < thresholds.min_psnr && !metrics.psnr.is_infinite() {
        issues.push(format!(
            "PSNR {:.2} dB below threshold {:.2} dB",
            metrics.psnr, thresholds.min_psnr
        ));
        passed = false;
    }
    if metrics.ssim < thresholds.min_ssim {
        issues.push(format!(
            "SSIM {:.4} below threshold {:.4}",
            metrics.ssim, thresholds.min_ssim
        ));
        passed = false;
    }
    if artifacts.clipping_fraction > thresholds.max_clipping_pct {
        issues.push(format!(
            "Clipping {:.2}% exceeds threshold {:.2}%",
            artifacts.clipping_fraction * 100.0,
            thresholds.max_clipping_pct * 100.0,
        ));
        passed = false;
    }
    if artifacts.has_excessive_noise {
        issues.push(format!(
            "Noise level {:.5} exceeds threshold {:.5}",
            artifacts.noise_level, thresholds.max_noise_level
        ));
        passed = false;
    }

    Ok(QualityReport {
        metrics,
        artifacts,
        passed,
        issues,
    })
}

/// Runs quality checks over a batch of (reference, rendered) pairs.
pub fn check_quality_batch(
    pairs: &[(&[u8], &[u8])],
    width: u32,
    height: u32,
    thresholds: &QualityThresholds,
) -> Result<BatchQualityReport, QualityError> {
    if pairs.is_empty() {
        return Err(QualityError::EmptyBatch);
    }

    let mut reports = Vec::with_capacity(pairs.len());
    for (reference, rendered) in pairs {
        let report = check_quality(reference, rendered, width, height, thresholds)?;
        reports.push(report);
    }

    let total_images = reports.len();
    let passed_count = reports.iter().filter(|r| r.passed).count();
    let failed_count = total_images - passed_count;

    let psnr_values: Vec<f32> = reports
        .iter()
        .map(|r| {
            if r.metrics.psnr.is_infinite() {
                // Cap infinity for aggregation purposes.
                999.0
            } else {
                r.metrics.psnr
            }
        })
        .collect();

    let mean_psnr = psnr_values.iter().sum::<f32>() / total_images as f32;
    let min_psnr = psnr_values.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_psnr = psnr_values
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);

    let mean_ssim = reports.iter().map(|r| r.metrics.ssim).sum::<f32>() / total_images as f32;

    // Restore infinity for min_psnr when all images are identical.
    let min_psnr_out = if min_psnr >= 999.0 {
        f32::INFINITY
    } else {
        min_psnr
    };
    let max_psnr_out = if max_psnr >= 999.0 {
        f32::INFINITY
    } else {
        max_psnr
    };

    Ok(BatchQualityReport {
        total_images,
        passed_count,
        failed_count,
        mean_psnr,
        min_psnr: min_psnr_out,
        max_psnr: max_psnr_out,
        mean_ssim,
        reports,
    })
}

// ---------------------------------------------------------------------------
// Free functions — error map & heatmap
// ---------------------------------------------------------------------------

/// Computes a per-pixel absolute error map (RGB f32, [0, 1]) between two RGBA images.
///
/// Alpha is discarded. Output length = `a.len() / 4 * 3`.
pub fn error_map(a: &[u8], b: &[u8]) -> Result<Vec<f32>, QualityError> {
    if a.is_empty() {
        return Err(QualityError::EmptyImage);
    }
    if a.len() != b.len() {
        return Err(QualityError::DimensionMismatch {
            expected_w: a.len() as u32,
            expected_h: 1,
            actual_w: b.len() as u32,
            actual_h: 1,
        });
    }
    let n_pixels = a.len() / 4;
    let mut out = Vec::with_capacity(n_pixels * 3);
    for (ca, cb) in a.chunks_exact(4).zip(b.chunks_exact(4)) {
        let dr = (ca[0] as f32 - cb[0] as f32).abs() / 255.0;
        let dg = (ca[1] as f32 - cb[1] as f32).abs() / 255.0;
        let db = (ca[2] as f32 - cb[2] as f32).abs() / 255.0;
        out.push(dr);
        out.push(dg);
        out.push(db);
    }
    Ok(out)
}

/// Converts an RGB f32 error map into an RGBA u8 heatmap.
///
/// - low error  → blue  [0, 0, 255]
/// - mid error  → green [0, 255, 0]
/// - high error → red   [255, 0, 0]
///
/// Per-pixel error is the mean of R, G, B channels, normalised by the
/// maximum error found in the map. Output length = `error.len() / 3 * 4`.
pub fn error_map_to_heatmap(error: &[f32]) -> Vec<u8> {
    if error.is_empty() {
        return Vec::new();
    }
    // Compute per-pixel scalar error (mean of 3 channels).
    let n_pixels = error.len() / 3;
    let mut scalars: Vec<f32> = Vec::with_capacity(n_pixels);
    for chunk in error.chunks_exact(3) {
        scalars.push((chunk[0] + chunk[1] + chunk[2]) / 3.0);
    }
    let max_err = scalars.iter().cloned().fold(0.0_f32, f32::max);

    let mut out = Vec::with_capacity(n_pixels * 4);
    for &s in &scalars {
        let e = if max_err > 0.0 { s / max_err } else { 0.0 };
        let r_f = (2.0 * e - 1.0).max(0.0);
        let b_f = (1.0 - 2.0 * e).max(0.0);
        let g_f = 1.0 - 2.0 * (e - 0.5).abs();
        let g_f = g_f.max(0.0);
        out.push((r_f * 255.0) as u8);
        out.push((g_f * 255.0) as u8);
        out.push((b_f * 255.0) as u8);
        out.push(255_u8); // alpha
    }
    out
}

// ---------------------------------------------------------------------------
// Free functions — histogram
// ---------------------------------------------------------------------------

/// Computes a 256-bin histogram per RGB channel from an RGBA image.
///
/// The returned array is interleaved: `[R0, G0, B0, R1, G1, B1, ...]`,
/// i.e., `hist[bin * 3 + channel]`.  Alpha is ignored.
pub fn compute_histogram(pixels: &[u8]) -> [u32; 768] {
    let mut hist = [0_u32; 768];
    for chunk in pixels.chunks_exact(4) {
        let r = chunk[0] as usize;
        let g = chunk[1] as usize;
        let b = chunk[2] as usize;
        hist[r * 3] += 1;
        hist[g * 3 + 1] += 1;
        hist[b * 3 + 2] += 1;
    }
    hist
}

/// Compares two histograms using the L1 (Manhattan) distance, normalised by pixel count.
///
/// Returns the sum of `|h1[i] - h2[i]|` over all bins, divided by `n_pixels`.
/// Returns `0.0` when `n_pixels == 0`.
pub fn histogram_distance(h1: &[u32; 768], h2: &[u32; 768], n_pixels: u32) -> f32 {
    if n_pixels == 0 {
        return 0.0;
    }
    let sum: u64 = h1
        .iter()
        .zip(h2.iter())
        .map(|(a, b)| (*a as i64 - *b as i64).unsigned_abs())
        .sum();
    sum as f32 / n_pixels as f32
}

// ---------------------------------------------------------------------------
// Free functions — blank detection
// ---------------------------------------------------------------------------

/// Returns `true` when all pixels fall within `tolerance` of the first pixel's values.
///
/// Only RGB channels are compared; alpha is ignored.
pub fn is_blank_image(pixels: &[u8], tolerance: u8) -> bool {
    let mut iter = pixels.chunks_exact(4);
    let first = match iter.next() {
        Some(c) => [c[0], c[1], c[2]],
        None => return true, // empty is considered blank
    };
    for chunk in iter {
        for (i, &ref_val) in first.iter().enumerate() {
            let diff = (chunk[i] as i16 - ref_val as i16).unsigned_abs() as u8;
            if diff > tolerance {
                return false;
            }
        }
    }
    true
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Test helpers
    // -----------------------------------------------------------------------

    /// Generates a solid-colour RGBA image.
    fn solid_rgba(width: u32, height: u32, r: u8, g: u8, b: u8, a: u8) -> Vec<u8> {
        let n = (width * height) as usize;
        let mut v = Vec::with_capacity(n * 4);
        for _ in 0..n {
            v.push(r);
            v.push(g);
            v.push(b);
            v.push(a);
        }
        v
    }

    /// Generates a checkerboard RGBA image (alternating black and white 8×8 blocks).
    fn checkerboard(width: u32, height: u32) -> Vec<u8> {
        let mut v = Vec::with_capacity((width * height * 4) as usize);
        for y in 0..height {
            for x in 0..width {
                let block = (x / 8 + y / 8) % 2;
                let lum: u8 = if block == 0 { 0 } else { 255 };
                v.push(lum);
                v.push(lum);
                v.push(lum);
                v.push(255);
            }
        }
        v
    }

    /// Generates a gradient RGBA image (horizontal, black → white).
    fn gradient_rgba(width: u32, height: u32) -> Vec<u8> {
        let mut v = Vec::with_capacity((width * height * 4) as usize);
        for _y in 0..height {
            for x in 0..width {
                let lum = ((x as f32 / (width - 1).max(1) as f32) * 255.0) as u8;
                v.push(lum);
                v.push(lum);
                v.push(lum);
                v.push(255);
            }
        }
        v
    }

    /// Adds deterministic "noise" using xorshift64 to a copy of the image.
    fn add_xorshift_noise(pixels: &[u8], amplitude: u8) -> Vec<u8> {
        let mut state: u64 = 0xdeadbeef_cafebabe;
        pixels
            .iter()
            .enumerate()
            .map(|(i, &p)| {
                if i % 4 == 3 {
                    return p; // keep alpha
                }
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let noise = (state % (amplitude as u64 * 2 + 1)) as i16 - amplitude as i16;
                (p as i16 + noise).clamp(0, 255) as u8
            })
            .collect()
    }

    // -----------------------------------------------------------------------
    // rgba_u8_to_rgb_f32 tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_rgba_to_rgb_f32_white() {
        let white = vec![255u8, 255, 255, 255];
        let out = rgba_u8_to_rgb_f32(&white);
        assert_eq!(out.len(), 3);
        assert!((out[0] - 1.0).abs() < 1e-6, "R should be 1.0");
        assert!((out[1] - 1.0).abs() < 1e-6, "G should be 1.0");
        assert!((out[2] - 1.0).abs() < 1e-6, "B should be 1.0");
    }

    #[test]
    fn test_rgba_to_rgb_f32_black() {
        let black = vec![0u8, 0, 0, 255];
        let out = rgba_u8_to_rgb_f32(&black);
        assert_eq!(out.len(), 3);
        assert!(out[0].abs() < 1e-6);
        assert!(out[1].abs() < 1e-6);
        assert!(out[2].abs() < 1e-6);
    }

    #[test]
    fn test_rgba_to_rgb_f32_midgray() {
        let gray = vec![128u8, 128, 128, 255];
        let out = rgba_u8_to_rgb_f32(&gray);
        assert_eq!(out.len(), 3);
        let expected = 128.0 / 255.0;
        assert!((out[0] - expected).abs() < 1e-5);
        assert!((out[1] - expected).abs() < 1e-5);
        assert!((out[2] - expected).abs() < 1e-5);
    }

    // -----------------------------------------------------------------------
    // compute_mse tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_mse_identical() {
        let a = vec![0.5_f32, 0.3, 0.7];
        let mse = compute_mse(&a, &a).expect("mse should succeed");
        assert!(mse.abs() < 1e-9, "identical images → MSE 0");
    }

    #[test]
    fn test_mse_known_value() {
        let a = vec![0.0_f32, 0.0, 0.0];
        let b = vec![1.0_f32, 1.0, 1.0];
        let mse = compute_mse(&a, &b).expect("mse should succeed");
        assert!(
            (mse - 1.0).abs() < 1e-6,
            "MSE between 0 and 1 should be 1.0"
        );
    }

    #[test]
    fn test_mse_partial() {
        let a = vec![0.0_f32, 0.0];
        let b = vec![1.0_f32, 0.0];
        let mse = compute_mse(&a, &b).expect("mse");
        assert!(
            (mse - 0.5).abs() < 1e-6,
            "half the pixels differ by 1 → MSE 0.5"
        );
    }

    #[test]
    fn test_mse_length_mismatch() {
        let a = vec![0.5_f32];
        let b = vec![0.5_f32, 0.5];
        assert!(matches!(
            compute_mse(&a, &b),
            Err(QualityError::DimensionMismatch { .. })
        ));
    }

    // -----------------------------------------------------------------------
    // compute_mae tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_mae_identical() {
        let a = vec![0.2_f32, 0.8, 0.5];
        let mae = compute_mae(&a, &a).expect("mae");
        assert!(mae.abs() < 1e-9);
    }

    #[test]
    fn test_mae_known() {
        let a = vec![0.0_f32, 0.0];
        let b = vec![0.4_f32, 0.6];
        let mae = compute_mae(&a, &b).expect("mae");
        assert!((mae - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_mae_empty() {
        let a: Vec<f32> = Vec::new();
        assert!(matches!(compute_mae(&a, &a), Err(QualityError::EmptyImage)));
    }

    // -----------------------------------------------------------------------
    // psnr_from_mse tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_psnr_zero_mse() {
        assert_eq!(psnr_from_mse(0.0), f32::INFINITY);
    }

    #[test]
    fn test_psnr_mse_one() {
        let p = psnr_from_mse(1.0);
        assert!(p.abs() < 1e-5, "PSNR at MSE=1 should be 0 dB, got {}", p);
    }

    #[test]
    fn test_psnr_known_value() {
        // MSE = 0.01 → PSNR = 10*log10(100) = 20 dB
        let p = psnr_from_mse(0.01);
        assert!((p - 20.0).abs() < 1e-4, "expected 20 dB, got {}", p);
    }

    // -----------------------------------------------------------------------
    // compute_psnr tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_psnr_identical() {
        let a = vec![0.3_f32; 100];
        let p = compute_psnr(&a, &a).expect("psnr");
        assert!(p.is_infinite());
    }

    #[test]
    fn test_psnr_different() {
        let a = vec![0.0_f32; 4];
        let b = vec![1.0_f32; 4];
        let p = compute_psnr(&a, &b).expect("psnr");
        assert!(p.abs() < 1e-4, "PSNR should be 0 dB, got {}", p);
    }

    #[test]
    fn test_psnr_mismatch_error() {
        let a = vec![0.1_f32; 3];
        let b = vec![0.1_f32; 4];
        assert!(matches!(
            compute_psnr(&a, &b),
            Err(QualityError::DimensionMismatch { .. })
        ));
    }

    // -----------------------------------------------------------------------
    // compute_ssim tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_ssim_identical() {
        let img = gradient_rgba(32, 32);
        let f32_img = rgba_u8_to_rgb_f32(&img);
        let ssim = compute_ssim(&f32_img, &f32_img, 32, 32, 3).expect("ssim");
        assert!(
            (ssim - 1.0).abs() < 1e-4,
            "identical images → SSIM≈1, got {}",
            ssim
        );
    }

    #[test]
    fn test_ssim_very_different() {
        let a = solid_rgba(32, 32, 0, 0, 0, 255);
        let b = solid_rgba(32, 32, 255, 255, 255, 255);
        let fa = rgba_u8_to_rgb_f32(&a);
        let fb = rgba_u8_to_rgb_f32(&b);
        let ssim = compute_ssim(&fa, &fb, 32, 32, 3).expect("ssim");
        // Solid black vs solid white should give near-0 or very low SSIM.
        // (C1 and C2 stabilise the formula for uniform patches; result may not be exactly 0)
        assert!(
            ssim < 0.9,
            "black vs white SSIM should be low, got {}",
            ssim
        );
    }

    #[test]
    fn test_ssim_small_image_returns_one() {
        // Image smaller than one 8×8 patch.
        let a = vec![0.5_f32; 9]; // 3×3×1
        let b = vec![0.8_f32; 9];
        let ssim = compute_ssim(&a, &b, 3, 3, 1).expect("ssim");
        assert!(
            (ssim - 1.0).abs() < 1e-6,
            "tiny image → fallback 1.0, got {}",
            ssim
        );
    }

    #[test]
    fn test_ssim_gradient_vs_inverted() {
        let a = gradient_rgba(32, 32);
        // Invert luminance.
        let b: Vec<u8> = a
            .iter()
            .enumerate()
            .map(|(i, &v)| if i % 4 == 3 { v } else { 255 - v })
            .collect();
        let fa = rgba_u8_to_rgb_f32(&a);
        let fb = rgba_u8_to_rgb_f32(&b);
        let ssim = compute_ssim(&fa, &fb, 32, 32, 3).expect("ssim");
        assert!(ssim < 0.5, "inverted gradient → low SSIM, got {}", ssim);
    }

    // -----------------------------------------------------------------------
    // compute_quality_metrics tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_quality_metrics_same_image() {
        let img = gradient_rgba(16, 16);
        let m = compute_quality_metrics(&img, &img, 16, 16).expect("metrics");
        assert!(m.psnr.is_infinite(), "identical → infinite PSNR");
        assert!(m.mse.abs() < 1e-9);
        assert!(m.mae.abs() < 1e-9);
        assert!(m.max_error.abs() < 1e-6);
    }

    #[test]
    fn test_quality_metrics_blank_vs_noise() {
        let reference = solid_rgba(16, 16, 128, 128, 128, 255);
        let noisy = add_xorshift_noise(&reference, 30);
        let m = compute_quality_metrics(&reference, &noisy, 16, 16).expect("metrics");
        assert!(
            m.psnr < 40.0,
            "noisy image should have low PSNR, got {}",
            m.psnr
        );
        assert!(m.mse > 0.0);
    }

    #[test]
    fn test_quality_metrics_dimension_mismatch() {
        let a = solid_rgba(4, 4, 0, 0, 0, 255);
        let b = solid_rgba(8, 8, 0, 0, 0, 255);
        // reference wrong size
        assert!(matches!(
            compute_quality_metrics(&a, &b, 8, 8),
            Err(QualityError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_quality_metrics_ssim_range() {
        let a = checkerboard(32, 32);
        let b = checkerboard(32, 32);
        let m = compute_quality_metrics(&a, &b, 32, 32).expect("metrics");
        assert!(
            m.ssim >= 0.0 && m.ssim <= 1.0001,
            "SSIM out of range: {}",
            m.ssim
        );
    }

    // -----------------------------------------------------------------------
    // detect_artifacts tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_artifacts_clean_gray() {
        let img = solid_rgba(16, 16, 128, 128, 128, 255);
        let thresholds = QualityThresholds::default();
        let report = detect_artifacts(&img, 16, 16, &thresholds);
        // A solid mid-gray should have no excessive noise, banding, or drift.
        assert!(!report.has_excessive_noise, "no noise in solid gray");
        assert!(!report.has_banding, "no banding in solid gray");
        assert!(!report.has_color_drift, "no drift in neutral gray");
    }

    #[test]
    fn test_artifacts_clipping_all_white() {
        let img = solid_rgba(16, 16, 255, 255, 255, 255);
        let thresholds = QualityThresholds::default();
        let report = detect_artifacts(&img, 16, 16, &thresholds);
        assert!(report.has_clipping, "all-white → clipping");
        assert!((report.clipping_fraction - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_artifacts_clipping_all_black() {
        let img = solid_rgba(16, 16, 0, 0, 0, 255);
        let thresholds = QualityThresholds::default();
        let report = detect_artifacts(&img, 16, 16, &thresholds);
        assert!(report.has_clipping, "all-black → clipping");
    }

    #[test]
    fn test_artifacts_noisy_image() {
        let base = solid_rgba(32, 32, 128, 128, 128, 255);
        let noisy = add_xorshift_noise(&base, 50);
        let thresholds = QualityThresholds {
            max_noise_level: 0.001, // very tight threshold
            ..Default::default()
        };
        let report = detect_artifacts(&noisy, 32, 32, &thresholds);
        assert!(
            report.has_excessive_noise,
            "high-amplitude noise should be detected"
        );
        assert!(report.noise_level > 0.0);
    }

    #[test]
    fn test_artifacts_overall_score_range() {
        let img = gradient_rgba(32, 32);
        let thresholds = QualityThresholds::default();
        let report = detect_artifacts(&img, 32, 32, &thresholds);
        assert!(
            report.overall_score >= 0.0 && report.overall_score <= 1.0,
            "overall_score out of [0,1]: {}",
            report.overall_score
        );
    }

    // -----------------------------------------------------------------------
    // check_quality tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_check_quality_passes() {
        // Use a solid mid-gray image (no clipped pixels, no noise, no banding).
        let img = solid_rgba(32, 32, 128, 128, 128, 255);
        let thresholds = QualityThresholds::default();
        let report = check_quality(&img, &img, 32, 32, &thresholds).expect("check");
        assert!(report.passed, "identical solid-gray images should pass");
        assert!(report.issues.is_empty());
    }

    #[test]
    fn test_check_quality_fails_psnr() {
        let reference = solid_rgba(16, 16, 128, 128, 128, 255);
        let rendered = solid_rgba(16, 16, 0, 0, 0, 255);
        let thresholds = QualityThresholds {
            min_psnr: 30.0,
            ..Default::default()
        };
        let report = check_quality(&reference, &rendered, 16, 16, &thresholds).expect("check");
        assert!(!report.passed, "very different images should fail PSNR");
        assert!(!report.issues.is_empty());
    }

    #[test]
    fn test_check_quality_fails_noise() {
        let reference = solid_rgba(32, 32, 128, 128, 128, 255);
        let noisy = add_xorshift_noise(&reference, 60);
        let thresholds = QualityThresholds {
            max_noise_level: 0.001,
            min_psnr: 0.0, // ignore PSNR
            min_ssim: 0.0, // ignore SSIM
            ..Default::default()
        };
        let report = check_quality(&reference, &noisy, 32, 32, &thresholds).expect("check");
        assert!(!report.passed, "noisy image should fail");
    }

    #[test]
    fn test_check_quality_empty_error() {
        let a: Vec<u8> = Vec::new();
        let thresholds = QualityThresholds::default();
        assert!(matches!(
            check_quality(&a, &a, 0, 0, &thresholds),
            Err(QualityError::EmptyImage)
        ));
    }

    // -----------------------------------------------------------------------
    // check_quality_batch tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_batch_two_pairs() {
        // Use solid mid-gray images so no clipping threshold is triggered.
        let img_a = solid_rgba(16, 16, 100, 120, 140, 255);
        let img_b = solid_rgba(16, 16, 128, 128, 128, 255);
        let thresholds = QualityThresholds::default();
        let pairs: Vec<(&[u8], &[u8])> = vec![(&img_a, &img_a), (&img_b, &img_b)];
        let batch = check_quality_batch(&pairs, 16, 16, &thresholds).expect("batch");
        assert_eq!(batch.total_images, 2);
        assert_eq!(batch.passed_count, 2);
        assert_eq!(batch.failed_count, 0);
    }

    #[test]
    fn test_batch_mixed_pass_fail() {
        // good: solid mid-gray vs itself → passes PSNR + no clipping
        let good = solid_rgba(16, 16, 128, 128, 128, 255);
        // bad: mid-gray vs black → fails PSNR
        let reference = solid_rgba(16, 16, 128, 128, 128, 255);
        let bad = solid_rgba(16, 16, 0, 0, 0, 255);
        let thresholds = QualityThresholds {
            min_psnr: 30.0,
            max_clipping_pct: 1.1, // ignore clipping for this test
            ..Default::default()
        };
        let pairs: Vec<(&[u8], &[u8])> = vec![(&good, &good), (&reference, &bad)];
        let batch = check_quality_batch(&pairs, 16, 16, &thresholds).expect("batch");
        assert_eq!(batch.total_images, 2);
        assert_eq!(batch.passed_count, 1);
        assert_eq!(batch.failed_count, 1);
    }

    #[test]
    fn test_batch_empty_error() {
        let thresholds = QualityThresholds::default();
        let pairs: Vec<(&[u8], &[u8])> = Vec::new();
        assert!(matches!(
            check_quality_batch(&pairs, 16, 16, &thresholds),
            Err(QualityError::EmptyBatch)
        ));
    }

    // -----------------------------------------------------------------------
    // error_map tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_error_map_identical_all_zeros() {
        let img = solid_rgba(8, 8, 100, 150, 200, 255);
        let map = error_map(&img, &img).expect("error_map");
        assert_eq!(map.len(), 8 * 8 * 3);
        assert!(
            map.iter().all(|&v| v.abs() < 1e-9),
            "identical images → all-zero error map"
        );
    }

    #[test]
    fn test_error_map_different() {
        let a = solid_rgba(4, 4, 0, 0, 0, 255);
        let b = solid_rgba(4, 4, 255, 255, 255, 255);
        let map = error_map(&a, &b).expect("error_map");
        assert_eq!(map.len(), 4 * 4 * 3);
        assert!(
            map.iter().all(|&v| (v - 1.0).abs() < 1e-5),
            "max-contrast → all-one error map"
        );
    }

    #[test]
    fn test_error_map_length_mismatch() {
        let a = solid_rgba(4, 4, 0, 0, 0, 255);
        let b = solid_rgba(8, 8, 0, 0, 0, 255);
        assert!(matches!(
            error_map(&a, &b),
            Err(QualityError::DimensionMismatch { .. })
        ));
    }

    // -----------------------------------------------------------------------
    // error_map_to_heatmap tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_heatmap_zero_error_is_blue() {
        let error = vec![0.0_f32; 3]; // single pixel, all zeros
        let heat = error_map_to_heatmap(&error);
        assert_eq!(heat.len(), 4);
        // zero error: R=0, B=255
        assert_eq!(heat[0], 0, "R should be 0 for zero error");
        assert_eq!(heat[2], 255, "B should be 255 for zero error");
        assert_eq!(heat[3], 255, "alpha should be 255");
    }

    #[test]
    fn test_heatmap_max_error_is_red() {
        // Two pixels: one with error 0, one with error 1 (after normalisation).
        let error = vec![0.0_f32, 0.0, 0.0, 1.0, 1.0, 1.0];
        let heat = error_map_to_heatmap(&error);
        assert_eq!(heat.len(), 8);
        // second pixel (max error): R should be ~255, B should be 0
        assert_eq!(heat[4], 255, "R should be 255 for max error");
        assert_eq!(heat[6], 0, "B should be 0 for max error");
    }

    #[test]
    fn test_heatmap_empty_input() {
        let heat = error_map_to_heatmap(&[]);
        assert!(heat.is_empty());
    }

    // -----------------------------------------------------------------------
    // compute_histogram tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_histogram_all_white() {
        let img = solid_rgba(4, 4, 255, 255, 255, 255);
        let h = compute_histogram(&img);
        // Bin 255, channel R = index 255*3+0 = 765 -- wait, array is [768] but index max = 255*3+2=767
        // All 16 pixels → bin 255, all three channels.
        assert_eq!(h[255 * 3], 16, "R bin 255 should have 16");
        assert_eq!(h[255 * 3 + 1], 16, "G bin 255 should have 16");
        assert_eq!(h[255 * 3 + 2], 16, "B bin 255 should have 16");
        // bin 0 should be zero
        assert_eq!(h[0], 0, "R bin 0 should be 0");
    }

    #[test]
    fn test_histogram_all_black() {
        let img = solid_rgba(4, 4, 0, 0, 0, 255);
        let h = compute_histogram(&img);
        assert_eq!(h[0], 16, "R bin 0 should have 16");
        assert_eq!(h[1], 16, "G bin 0 should have 16");
        assert_eq!(h[2], 16, "B bin 0 should have 16");
        assert_eq!(h[255 * 3], 0, "R bin 255 should be 0");
    }

    #[test]
    fn test_histogram_mixed() {
        // 8 black pixels, 8 white pixels.
        let mut img = solid_rgba(4, 2, 0, 0, 0, 255);
        img.extend(solid_rgba(4, 2, 255, 255, 255, 255));
        let h = compute_histogram(&img);
        assert_eq!(h[0], 8, "R bin 0: 8 black pixels");
        assert_eq!(h[255 * 3], 8, "R bin 255: 8 white pixels");
    }

    // -----------------------------------------------------------------------
    // histogram_distance tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_histogram_distance_same() {
        let img = gradient_rgba(16, 16);
        let h = compute_histogram(&img);
        let d = histogram_distance(&h, &h, 16 * 16);
        assert!(d.abs() < 1e-9, "same histogram → distance 0");
    }

    #[test]
    fn test_histogram_distance_different() {
        let a = solid_rgba(4, 4, 0, 0, 0, 255);
        let b = solid_rgba(4, 4, 255, 255, 255, 255);
        let ha = compute_histogram(&a);
        let hb = compute_histogram(&b);
        let d = histogram_distance(&ha, &hb, 16);
        // All 16 pixels differ in all 3 channels: total L1 = 3*16*2=96, /16=6
        assert!(d > 0.0, "different histograms → nonzero distance");
    }

    #[test]
    fn test_histogram_distance_zero_pixels() {
        let h1 = [0_u32; 768];
        let h2 = [1_u32; 768];
        let d = histogram_distance(&h1, &h2, 0);
        assert!(d.abs() < 1e-9, "zero pixel count → distance 0");
    }

    // -----------------------------------------------------------------------
    // is_blank_image tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_is_blank_uniform() {
        let img = solid_rgba(16, 16, 100, 100, 100, 255);
        assert!(is_blank_image(&img, 0), "uniform image should be blank");
    }

    #[test]
    fn test_is_blank_with_tolerance() {
        // Vary by up to 3, tolerance 5 → still blank.
        let mut img = solid_rgba(4, 4, 128, 128, 128, 255);
        img[0] = 131; // R of first pixel +3
        assert!(is_blank_image(&img, 5), "within tolerance → blank");
    }

    #[test]
    fn test_is_blank_varying() {
        let img = gradient_rgba(16, 16);
        assert!(!is_blank_image(&img, 2), "gradient image is not blank");
    }

    // -----------------------------------------------------------------------
    // BatchQualityReport pass_rate and format_summary tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_batch_pass_rate_all_pass() {
        // Use solid mid-gray so no artifacts trigger.
        let img = solid_rgba(16, 16, 128, 128, 128, 255);
        let thresholds = QualityThresholds::default();
        let pairs: Vec<(&[u8], &[u8])> = vec![(&img, &img), (&img, &img)];
        let batch = check_quality_batch(&pairs, 16, 16, &thresholds).expect("batch");
        assert!((batch.pass_rate() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_batch_pass_rate_none_pass() {
        let reference = solid_rgba(16, 16, 128, 128, 128, 255);
        let rendered = solid_rgba(16, 16, 0, 0, 0, 255);
        let thresholds = QualityThresholds {
            min_psnr: 60.0,
            ..Default::default()
        };
        let pairs: Vec<(&[u8], &[u8])> = vec![(&reference, &rendered)];
        let batch = check_quality_batch(&pairs, 16, 16, &thresholds).expect("batch");
        assert!(batch.pass_rate().abs() < 1e-6);
    }

    #[test]
    fn test_batch_format_summary_contains_info() {
        let img = gradient_rgba(16, 16);
        let thresholds = QualityThresholds::default();
        let pairs: Vec<(&[u8], &[u8])> = vec![(&img, &img)];
        let batch = check_quality_batch(&pairs, 16, 16, &thresholds).expect("batch");
        let summary = batch.format_summary();
        assert!(
            summary.contains("Batch:"),
            "summary should start with 'Batch:'"
        );
        assert!(summary.contains("PSNR"), "summary should mention PSNR");
        assert!(summary.contains("SSIM"), "summary should mention SSIM");
    }

    // -----------------------------------------------------------------------
    // passes_threshold test
    // -----------------------------------------------------------------------

    #[test]
    fn test_passes_threshold() {
        let m = ImageQualityMetrics {
            psnr: 30.0,
            mse: 0.001,
            mae: 0.01,
            ssim: 0.95,
            max_error: 0.1,
            width: 16,
            height: 16,
        };
        assert!(m.passes_threshold(25.0, 0.85));
        assert!(
            !m.passes_threshold(35.0, 0.85),
            "should fail PSNR threshold"
        );
        assert!(
            !m.passes_threshold(25.0, 0.99),
            "should fail SSIM threshold"
        );
    }
}
