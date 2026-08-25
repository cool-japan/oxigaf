//! Validation of Gaussian model data and render pipeline configuration.
//!
//! This module validates inputs before GPU dispatch, catching issues early and
//! producing human-readable diagnostics with suggested fixes.

use crate::config::RasterConfig;
use crate::gaussian::GaussianModel;
use crate::rasterizer::RenderCamera;

// ---------------------------------------------------------------------------
// Severity
// ---------------------------------------------------------------------------

/// Severity level of a validation issue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum IssueSeverity {
    /// Informational notice — no action required.
    Info,
    /// Warning — rendering may still succeed but quality could be degraded.
    Warning,
    /// Error — rendering is expected to fail or produce garbage.
    Error,
}

impl std::fmt::Display for IssueSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Info => write!(f, "INFO"),
            Self::Warning => write!(f, "WARN"),
            Self::Error => write!(f, "ERROR"),
        }
    }
}

// ---------------------------------------------------------------------------
// ValidationIssue
// ---------------------------------------------------------------------------

/// A single validation issue found during pre-render checks.
#[derive(Debug, Clone)]
pub struct ValidationIssue {
    /// How serious the issue is.
    pub severity: IssueSeverity,
    /// Name of the field or subsystem where the issue was found.
    pub field: String,
    /// Human-readable description of the problem.
    pub message: String,
    /// Optional suggestion for resolving the issue.
    pub suggestion: Option<String>,
}

impl ValidationIssue {
    /// Construct an error-severity issue.
    pub fn error(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            severity: IssueSeverity::Error,
            field: field.into(),
            message: message.into(),
            suggestion: None,
        }
    }

    /// Construct a warning-severity issue.
    pub fn warning(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            severity: IssueSeverity::Warning,
            field: field.into(),
            message: message.into(),
            suggestion: None,
        }
    }

    /// Construct an info-severity issue.
    pub fn info(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            severity: IssueSeverity::Info,
            field: field.into(),
            message: message.into(),
            suggestion: None,
        }
    }

    /// Attach a suggestion string (builder pattern).
    #[must_use]
    pub fn with_suggestion(mut self, s: impl Into<String>) -> Self {
        self.suggestion = Some(s.into());
        self
    }
}

// ---------------------------------------------------------------------------
// GaussianModelValidation
// ---------------------------------------------------------------------------

/// Result of validating a [`GaussianModel`].
pub struct GaussianModelValidation {
    /// All issues found; may be empty.
    pub issues: Vec<ValidationIssue>,
    /// Number of Gaussians in the model.
    pub num_gaussians: usize,
    /// `true` when no `Error`-severity issues are present.
    pub is_valid: bool,
}

/// Maximum number of individual occurrences reported for barycentric warnings
/// before collapsing to an aggregate note.
const MAX_BARY_REPORTS: usize = 5;

/// Validate a [`GaussianModel`] for common data issues.
///
/// Checks NaN/Inf values, quaternion normalization, extreme log-scale values,
/// empty model, and FLAME binding consistency.
pub fn validate_gaussian_model(model: &GaussianModel) -> GaussianModelValidation {
    let mut issues: Vec<ValidationIssue> = Vec::new();
    let num_gaussians = model.gaussians.len();

    // ----- Empty model -------------------------------------------------
    if num_gaussians == 0 {
        issues.push(
            ValidationIssue::error("model", "Empty model: no Gaussians present").with_suggestion(
                "Load a trained .ply or .safetensors checkpoint before rendering.",
            ),
        );
        return GaussianModelValidation {
            is_valid: false,
            num_gaussians,
            issues,
        };
    }

    // ----- Positions: NaN / Inf ----------------------------------------
    let has_bad_pos = model
        .gaussians
        .iter()
        .any(|g| g.position.iter().any(|v| v.is_nan() || v.is_infinite()));
    if has_bad_pos {
        issues.push(
            ValidationIssue::error(
                "positions",
                "NaN/Inf in positions: one or more Gaussians have non-finite position values",
            )
            .with_suggestion(
                "Check the training pipeline for diverging optimizers or invalid input data.",
            ),
        );
    }

    // ----- Rotations: non-unit quaternions -----------------------------
    let non_unit_count = model
        .gaussians
        .iter()
        .filter(|g| {
            let [x, y, z, w] = g.rotation;
            let norm = (x * x + y * y + z * z + w * w).sqrt();
            (norm - 1.0_f32).abs() > 0.01
        })
        .count();
    if non_unit_count > 0 {
        issues.push(
            ValidationIssue::warning(
                "rotations",
                format!(
                    "Non-unit quaternions: {non_unit_count} of {num_gaussians} \
                     Gaussians have |q| deviating from 1.0 by more than 0.01"
                ),
            )
            .with_suggestion(
                "Normalize rotation quaternions before rendering (e.g., divide by their norm).",
            ),
        );
    }

    // ----- Scales: extreme log-space values ----------------------------
    let extreme_scale_count = model
        .gaussians
        .iter()
        .filter(|g| g.scale.iter().any(|&s| !(-15.0_f32..=5.0_f32).contains(&s)))
        .count();
    if extreme_scale_count > 0 {
        issues.push(
            ValidationIssue::warning(
                "scales",
                format!(
                    "Extreme scale values: {extreme_scale_count} of {num_gaussians} \
                     Gaussians have log-scale outside [-15.0, 5.0]"
                ),
            )
            .with_suggestion(
                "Extremely large scales cause Gaussians to cover the whole scene; \
                 extremely small scales (< exp(-15)) become invisible. \
                 Consider pruning or clamping.",
            ),
        );
    }

    // ----- Opacities: NaN / Inf ----------------------------------------
    let has_bad_opacity = model
        .gaussians
        .iter()
        .any(|g| g.opacity.is_nan() || g.opacity.is_infinite());
    if has_bad_opacity {
        issues.push(
            ValidationIssue::error(
                "opacities",
                "NaN/Inf in opacities: one or more Gaussians have non-finite opacity values",
            )
            .with_suggestion("Inspect the opacity head of your model for numerical instability."),
        );
    }

    // ----- SH coefficients: NaN / Inf ----------------------------------
    let has_bad_sh = model
        .sh_coeffs
        .iter()
        .any(|v| v.is_nan() || v.is_infinite());
    if has_bad_sh {
        issues.push(
            ValidationIssue::warning(
                "sh_coeffs",
                "NaN/Inf in SH coefficients: one or more coefficients are non-finite",
            )
            .with_suggestion(
                "NaN SH coefficients produce black or undefined color output. \
                 Re-initialize or prune affected Gaussians.",
            ),
        );
    }

    // ----- FLAME binding consistency -----------------------------------
    // face_indices length mismatch (only check if the binding is populated)
    if !model.face_indices.is_empty() && model.face_indices.len() != num_gaussians {
        issues.push(
            ValidationIssue::error(
                "face_indices",
                format!(
                    "face_indices length {} does not match num_gaussians {}",
                    model.face_indices.len(),
                    num_gaussians
                ),
            )
            .with_suggestion(
                "Ensure FLAME binding arrays are constructed with one entry per Gaussian.",
            ),
        );
    }

    // barycentric: length mismatch
    if !model.barycentric.is_empty() && model.barycentric.len() != num_gaussians {
        issues.push(
            ValidationIssue::error(
                "barycentric",
                format!(
                    "barycentric length {} does not match num_gaussians {}",
                    model.barycentric.len(),
                    num_gaussians
                ),
            )
            .with_suggestion(
                "Ensure FLAME binding arrays are constructed with one entry per Gaussian.",
            ),
        );
    }

    // barycentric sum out of [0.9, 1.1]
    if !model.barycentric.is_empty() && model.barycentric.len() == num_gaussians {
        let mut bad_indices: Vec<usize> = Vec::new();
        for (i, bary) in model.barycentric.iter().enumerate() {
            let sum = bary[0] + bary[1] + bary[2];
            if !(0.9..=1.1).contains(&sum) {
                bad_indices.push(i);
            }
        }
        if !bad_indices.is_empty() {
            let shown: Vec<_> = bad_indices.iter().take(MAX_BARY_REPORTS).collect();
            let extra = bad_indices.len().saturating_sub(MAX_BARY_REPORTS);
            let detail = if extra > 0 {
                format!(
                    "Invalid barycentric coordinates: {} occurrences (first {:?}, … +{} more)",
                    bad_indices.len(),
                    shown,
                    extra
                )
            } else {
                format!(
                    "Invalid barycentric coordinates at {} occurrence(s): {:?}",
                    bad_indices.len(),
                    shown
                )
            };
            issues.push(
                ValidationIssue::warning("barycentric", detail).with_suggestion(
                    "Barycentric coordinates (u, v, w) should sum to 1.0 ± 0.1. \
                     Re-project Gaussians onto the FLAME mesh surface.",
                ),
            );
        }
    }

    let is_valid = issues.iter().all(|i| i.severity != IssueSeverity::Error);
    GaussianModelValidation {
        issues,
        num_gaussians,
        is_valid,
    }
}

// ---------------------------------------------------------------------------
// Camera validation
// ---------------------------------------------------------------------------

/// Validate a [`RenderCamera`] for common configuration issues.
///
/// Checks view/projection matrices for NaN/Inf and degenerate projections.
pub fn validate_camera(camera: &RenderCamera) -> Vec<ValidationIssue> {
    let mut issues: Vec<ValidationIssue> = Vec::new();

    // view_matrix NaN / Inf
    let has_bad_view = camera
        .view_matrix
        .iter()
        .any(|v| v.is_nan() || v.is_infinite());
    if has_bad_view {
        issues.push(
            ValidationIssue::error(
                "view_matrix",
                "NaN/Inf in view matrix: one or more elements are non-finite",
            )
            .with_suggestion("Verify camera extrinsics (position, look-at, up vector) are finite."),
        );
    }

    // proj_matrix NaN / Inf
    let has_bad_proj = camera
        .proj_matrix
        .iter()
        .any(|v| v.is_nan() || v.is_infinite());
    if has_bad_proj {
        issues.push(
            ValidationIssue::error(
                "proj_matrix",
                "NaN/Inf in projection matrix: one or more elements are non-finite",
            )
            .with_suggestion(
                "Check that focal lengths, aspect ratio, and near/far planes are valid.",
            ),
        );
    }

    // Check projection matrix diagonal: column-major 4×4 diagonal at
    // [0],[5],[10]. Index 15 (the w-w entry) is deliberately excluded: it is
    // exactly 0.0 for every standard perspective projection matrix (only an
    // orthographic matrix has m[15] == 1), so including it here flagged
    // every correct perspective camera as "degenerate".
    let diag_indices = [0_usize, 5, 10];
    let has_zero_diag = diag_indices
        .iter()
        .any(|&i| camera.proj_matrix[i].abs() < f32::EPSILON);
    if has_zero_diag {
        issues.push(
            ValidationIssue::warning(
                "proj_matrix",
                "Projection matrix diagonal contains near-zero entries, \
                 which may indicate a degenerate projection",
            )
            .with_suggestion(
                "Ensure the focal lengths and near/far planes produce a non-singular \
                 projection matrix.",
            ),
        );
    }

    issues
}

// ---------------------------------------------------------------------------
// RasterConfig validation
// ---------------------------------------------------------------------------

/// Validate a [`RasterConfig`] for pipeline compatibility.
///
/// Checks tile size, background color bounds, and near/far clipping planes.
pub fn validate_raster_config(config: &RasterConfig) -> Vec<ValidationIssue> {
    let mut issues: Vec<ValidationIssue> = Vec::new();

    // tile_size: power of 2 in [4, 64]
    let ts = config.tile_size;
    let is_pow2 = ts > 0 && (ts & (ts - 1)) == 0;
    if !is_pow2 || !(4..=64).contains(&ts) {
        issues.push(
            ValidationIssue::warning(
                "tile_size",
                format!(
                    "tile_size={ts} is not a power of 2 in [4, 64]; \
                     GPU rasterization may be suboptimal or produce incorrect tiling"
                ),
            )
            .with_suggestion("Use a power-of-two tile size such as 8, 16, or 32."),
        );
    }

    // background: each component in [0, 1]
    for (i, &c) in config.background.iter().enumerate() {
        if !(0.0..=1.0).contains(&c) {
            let channel = ["R", "G", "B"][i];
            issues.push(
                ValidationIssue::warning(
                    "background",
                    format!(
                        "background[{channel}] = {c:.4} is outside [0, 1]; \
                         alpha-blending assumes components are in [0, 1]"
                    ),
                )
                .with_suggestion("Clamp each background color component to the [0.0, 1.0] range."),
            );
        }
    }

    // near_plane > 0
    if config.near_plane <= 0.0 {
        issues.push(
            ValidationIssue::error(
                "near_plane",
                format!(
                    "near_plane={} must be strictly positive; \
                     a zero or negative near plane will invert depth or produce NaN",
                    config.near_plane
                ),
            )
            .with_suggestion("Set near_plane to a small positive value such as 0.01."),
        );
    }

    // far_plane > near_plane
    if config.far_plane <= config.near_plane {
        issues.push(
            ValidationIssue::error(
                "far_plane",
                format!(
                    "far_plane={} must be greater than near_plane={}; \
                     depth buffer range would be zero or inverted",
                    config.far_plane, config.near_plane
                ),
            )
            .with_suggestion("Ensure far_plane > near_plane (e.g., near=0.01, far=100.0)."),
        );
    }

    issues
}

// ---------------------------------------------------------------------------
// RenderPipelineValidation
// ---------------------------------------------------------------------------

/// Aggregated validation report for a complete render pipeline.
pub struct RenderPipelineValidation {
    /// Detailed Gaussian model validation result.
    pub model_validation: GaussianModelValidation,
    /// Issues found in the camera configuration.
    pub camera_issues: Vec<ValidationIssue>,
    /// Issues found in the raster configuration.
    pub config_issues: Vec<ValidationIssue>,
    /// Total number of error-severity issues across all subsystems.
    pub total_errors: usize,
    /// Total number of warning-severity issues across all subsystems.
    pub total_warnings: usize,
    /// Total number of info-severity issues across all subsystems.
    pub total_infos: usize,
}

impl RenderPipelineValidation {
    /// Run all validations and return an aggregated report.
    pub fn run(model: &GaussianModel, camera: &RenderCamera, config: &RasterConfig) -> Self {
        let model_validation = validate_gaussian_model(model);
        let camera_issues = validate_camera(camera);
        let config_issues = validate_raster_config(config);

        // Count by severity across all issue lists
        let all_issues = model_validation
            .issues
            .iter()
            .chain(camera_issues.iter())
            .chain(config_issues.iter());

        let mut total_errors = 0usize;
        let mut total_warnings = 0usize;
        let mut total_infos = 0usize;
        for issue in all_issues {
            match issue.severity {
                IssueSeverity::Error => total_errors += 1,
                IssueSeverity::Warning => total_warnings += 1,
                IssueSeverity::Info => total_infos += 1,
            }
        }

        Self {
            model_validation,
            camera_issues,
            config_issues,
            total_errors,
            total_warnings,
            total_infos,
        }
    }

    /// Return `true` when there are no error-severity issues — safe to dispatch to GPU.
    #[must_use]
    pub fn is_safe_to_render(&self) -> bool {
        self.total_errors == 0
    }

    /// Format a human-readable multi-section report, grouped by severity.
    #[must_use]
    pub fn format_report(&self) -> String {
        use std::fmt::Write as _;

        let mut out = String::new();

        let _ = writeln!(
            &mut out,
            "=== Render Pipeline Validation Report ===\n\
             Model: {} Gaussian(s) | Errors: {} | Warnings: {} | Infos: {}",
            self.model_validation.num_gaussians,
            self.total_errors,
            self.total_warnings,
            self.total_infos,
        );

        let sections: &[(&str, &[ValidationIssue])] = &[
            ("Gaussian Model", &self.model_validation.issues),
            ("Camera", &self.camera_issues),
            ("Raster Config", &self.config_issues),
        ];

        for &(section_name, section_issues) in sections {
            if section_issues.is_empty() {
                continue;
            }
            let _ = writeln!(&mut out, "\n[{section_name}]");
            // Print errors first, then warnings, then infos
            for sev in [
                IssueSeverity::Error,
                IssueSeverity::Warning,
                IssueSeverity::Info,
            ] {
                for issue in section_issues.iter().filter(|i| i.severity == sev) {
                    let _ = writeln!(
                        &mut out,
                        "  [{severity}] {field}: {message}",
                        severity = issue.severity,
                        field = issue.field,
                        message = issue.message,
                    );
                    if let Some(ref suggestion) = issue.suggestion {
                        let _ = writeln!(&mut out, "    Suggestion: {suggestion}");
                    }
                }
            }
        }

        let status = if self.is_safe_to_render() {
            "SAFE TO RENDER"
        } else {
            "UNSAFE — fix errors before rendering"
        };
        let _ = writeln!(&mut out, "\nStatus: {status}");

        out
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gaussian::GaussianAttributes;

    // -----------------------------------------------------------------------
    // Helper builders
    // -----------------------------------------------------------------------

    fn make_model(n: usize) -> GaussianModel {
        let gaussians = (0..n)
            .map(|i| GaussianAttributes {
                position: [i as f32 * 0.1, 0.0, 0.0],
                _pad0: 0.0,
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: [0.0, 0.0, 0.0],
                opacity: 0.0,
            })
            .collect();
        let sh_coeffs = vec![0.0_f32; n * 3]; // degree 0 → 3 coeffs each
        GaussianModel {
            gaussians,
            sh_coeffs,
            sh_degree: 0,
            face_indices: Vec::new(),
            barycentric: Vec::new(),
            local_offsets: Vec::new(),
            is_rigid: Vec::new(),
        }
    }

    fn make_camera() -> RenderCamera {
        // Simple identity-like view, basic perspective projection
        let mut view = [0.0_f32; 16];
        view[0] = 1.0;
        view[5] = 1.0;
        view[10] = 1.0;
        view[15] = 1.0;

        let mut proj = [0.0_f32; 16];
        proj[0] = 1.0;
        proj[5] = 1.0;
        proj[10] = -1.0; // typical perspective
        proj[15] = 1.0;

        RenderCamera {
            view_matrix: view,
            proj_matrix: proj,
            position: [0.0, 0.0, 5.0],
            focal: [525.0, 525.0],
        }
    }

    fn make_config() -> RasterConfig {
        RasterConfig::default()
    }

    // -----------------------------------------------------------------------
    // Model tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_validate_empty_model() {
        let model = make_model(0);
        let result = validate_gaussian_model(&model);
        assert!(!result.is_valid);
        assert_eq!(result.num_gaussians, 0);
        let has_error = result
            .issues
            .iter()
            .any(|i| i.severity == IssueSeverity::Error && i.field == "model");
        assert!(has_error, "expected Error issue on 'model' field");
    }

    #[test]
    fn test_validate_clean_model() {
        let model = make_model(4);
        let result = validate_gaussian_model(&model);
        assert!(result.is_valid);
        assert_eq!(result.num_gaussians, 4);
        let error_count = result
            .issues
            .iter()
            .filter(|i| i.severity == IssueSeverity::Error)
            .count();
        assert_eq!(error_count, 0, "clean model should have no errors");
    }

    #[test]
    fn test_validate_nan_positions() {
        let mut model = make_model(3);
        model.gaussians[1].position[0] = f32::NAN;
        let result = validate_gaussian_model(&model);
        assert!(!result.is_valid);
        let has_pos_error = result
            .issues
            .iter()
            .any(|i| i.severity == IssueSeverity::Error && i.field == "positions");
        assert!(has_pos_error, "expected Error on 'positions'");
    }

    #[test]
    fn test_validate_inf_opacities() {
        let mut model = make_model(5);
        model.gaussians[4].opacity = f32::INFINITY;
        let result = validate_gaussian_model(&model);
        assert!(!result.is_valid);
        let has_opacity_error = result
            .issues
            .iter()
            .any(|i| i.severity == IssueSeverity::Error && i.field == "opacities");
        assert!(has_opacity_error, "expected Error on 'opacities'");
    }

    #[test]
    fn test_validate_non_unit_quaternions() {
        let mut model = make_model(4);
        // Set quaternion to (2, 0, 0, 0) — norm = 2.0, deviates by 1.0 from 1.0
        model.gaussians[2].rotation = [2.0, 0.0, 0.0, 0.0];
        let result = validate_gaussian_model(&model);
        assert!(
            result.is_valid,
            "non-unit quaternions should be Warning, not Error"
        );
        let has_warn = result
            .issues
            .iter()
            .any(|i| i.severity == IssueSeverity::Warning && i.field == "rotations");
        assert!(has_warn, "expected Warning on 'rotations'");
    }

    #[test]
    fn test_validate_extreme_scales() {
        let mut model = make_model(3);
        model.gaussians[0].scale = [10.0, 0.0, 0.0]; // > 5.0 → extreme
        let result = validate_gaussian_model(&model);
        assert!(result.is_valid, "extreme scale should be Warning");
        let has_warn = result
            .issues
            .iter()
            .any(|i| i.severity == IssueSeverity::Warning && i.field == "scales");
        assert!(has_warn, "expected Warning on 'scales'");
    }

    #[test]
    fn test_validate_nan_sh_coeffs() {
        let mut model = make_model(2);
        model.sh_coeffs[1] = f32::NAN;
        let result = validate_gaussian_model(&model);
        // NaN SH should be Warning, not Error
        assert!(result.is_valid);
        let has_warn = result
            .issues
            .iter()
            .any(|i| i.severity == IssueSeverity::Warning && i.field == "sh_coeffs");
        assert!(has_warn, "expected Warning on 'sh_coeffs'");
    }

    #[test]
    fn test_validate_face_indices_length_mismatch() {
        let mut model = make_model(4);
        // Only 2 face indices for 4 Gaussians
        model.face_indices = vec![0, 1];
        let result = validate_gaussian_model(&model);
        assert!(!result.is_valid);
        let has_error = result
            .issues
            .iter()
            .any(|i| i.severity == IssueSeverity::Error && i.field == "face_indices");
        assert!(has_error, "expected Error on 'face_indices'");
    }

    #[test]
    fn test_validate_barycentric_invalid() {
        let mut model = make_model(3);
        model.barycentric = vec![[0.5, 0.5, 0.5], [0.33, 0.33, 0.34], [0.0, 0.0, 0.0]];
        // First: sum = 1.5 (> 1.1) → bad
        // Second: sum ≈ 1.0 → good
        // Third: sum = 0.0 (< 0.9) → bad
        let result = validate_gaussian_model(&model);
        // Should be warning only (not error)
        assert!(result.is_valid);
        let has_warn = result
            .issues
            .iter()
            .any(|i| i.severity == IssueSeverity::Warning && i.field == "barycentric");
        assert!(has_warn, "expected Warning on 'barycentric'");
    }

    // -----------------------------------------------------------------------
    // Camera tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_validate_camera_nan_view() {
        let mut cam = make_camera();
        cam.view_matrix[3] = f32::NAN;
        let issues = validate_camera(&cam);
        let has_error = issues
            .iter()
            .any(|i| i.severity == IssueSeverity::Error && i.field == "view_matrix");
        assert!(has_error, "expected Error on 'view_matrix'");
    }

    #[test]
    fn test_validate_camera_clean() {
        let cam = make_camera();
        let issues = validate_camera(&cam);
        let error_count = issues
            .iter()
            .filter(|i| i.severity == IssueSeverity::Error)
            .count();
        assert_eq!(error_count, 0, "clean camera should have no errors");
    }

    #[test]
    fn test_validate_camera_realistic_perspective_no_spurious_degenerate_warning() {
        // A standard OpenGL-style perspective projection matrix has
        // proj[15] == 0.0 (only an orthographic matrix has proj[15] == 1.0).
        // Before the fix, `validate_camera` included index 15 in its
        // near-zero diagonal check, so every correct perspective camera was
        // flagged as having a "degenerate projection".
        let mut cam = make_camera();
        cam.proj_matrix[15] = 0.0;
        let issues = validate_camera(&cam);
        let has_degenerate_warning = issues
            .iter()
            .any(|i| i.field == "proj_matrix" && i.message.contains("degenerate"));
        assert!(
            !has_degenerate_warning,
            "a standard perspective matrix (proj[15] == 0) must not be flagged as degenerate: {issues:?}"
        );
    }

    // -----------------------------------------------------------------------
    // RasterConfig tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_validate_raster_config_zero_near() {
        let mut config = make_config();
        config.near_plane = 0.0;
        let issues = validate_raster_config(&config);
        let has_error = issues
            .iter()
            .any(|i| i.severity == IssueSeverity::Error && i.field == "near_plane");
        assert!(has_error, "expected Error on 'near_plane'");
    }

    #[test]
    fn test_validate_raster_config_near_exceeds_far() {
        let mut config = make_config();
        config.near_plane = 10.0;
        config.far_plane = 5.0;
        let issues = validate_raster_config(&config);
        let has_error = issues
            .iter()
            .any(|i| i.severity == IssueSeverity::Error && i.field == "far_plane");
        assert!(has_error, "expected Error on 'far_plane'");
    }

    #[test]
    fn test_validate_raster_config_tile_size_not_power_of_2() {
        let mut config = make_config();
        config.tile_size = 15; // not a power of 2
        let issues = validate_raster_config(&config);
        let has_warn = issues
            .iter()
            .any(|i| i.severity == IssueSeverity::Warning && i.field == "tile_size");
        assert!(has_warn, "expected Warning on 'tile_size'");
    }

    #[test]
    fn test_validate_raster_config_default_is_clean() {
        let config = RasterConfig::default();
        let issues = validate_raster_config(&config);
        let error_count = issues
            .iter()
            .filter(|i| i.severity == IssueSeverity::Error)
            .count();
        assert_eq!(error_count, 0, "default RasterConfig should have no errors");
    }

    // -----------------------------------------------------------------------
    // Pipeline validation tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_pipeline_validation_run() {
        let model = make_model(10);
        let camera = make_camera();
        let config = make_config();
        let pv = RenderPipelineValidation::run(&model, &camera, &config);
        assert_eq!(pv.model_validation.num_gaussians, 10);
        // total counts are non-negative (trivially true with usize but let's be explicit)
        assert_eq!(
            pv.total_errors + pv.total_warnings + pv.total_infos,
            pv.model_validation.issues.len() + pv.camera_issues.len() + pv.config_issues.len()
        );
    }

    #[test]
    fn test_pipeline_is_safe_when_no_errors() {
        let model = make_model(5);
        let camera = make_camera();
        let config = make_config();
        let pv = RenderPipelineValidation::run(&model, &camera, &config);
        assert!(
            pv.is_safe_to_render(),
            "clean inputs should be safe to render"
        );
    }

    #[test]
    fn test_pipeline_not_safe_with_errors() {
        let model = make_model(0); // empty → Error
        let camera = make_camera();
        let config = make_config();
        let pv = RenderPipelineValidation::run(&model, &camera, &config);
        assert!(!pv.is_safe_to_render());
        assert!(pv.total_errors > 0);
    }

    // -----------------------------------------------------------------------
    // Severity ordering
    // -----------------------------------------------------------------------

    #[test]
    fn test_issue_severity_ordering() {
        assert!(IssueSeverity::Info < IssueSeverity::Warning);
        assert!(IssueSeverity::Warning < IssueSeverity::Error);
        assert!(IssueSeverity::Info < IssueSeverity::Error);
    }

    // -----------------------------------------------------------------------
    // Format report
    // -----------------------------------------------------------------------

    #[test]
    fn test_format_report() {
        // Make a model with known issues
        let mut model = make_model(3);
        model.gaussians[0].position[1] = f32::NAN; // Error: NaN position
        model.gaussians[1].scale[0] = 99.0; // Warning: extreme scale

        let camera = make_camera();
        let config = make_config();

        let pv = RenderPipelineValidation::run(&model, &camera, &config);
        let report = pv.format_report();

        // Report should contain key structural elements
        assert!(
            report.contains("Render Pipeline Validation Report"),
            "report missing header"
        );
        assert!(report.contains("Errors:"), "report should show error count");
        assert!(
            report.contains("Gaussian Model"),
            "report should have Gaussian Model section"
        );
        assert!(
            report.contains("ERROR") || report.contains("WARN"),
            "report should contain at least one issue"
        );
        assert!(
            report.contains("Status:"),
            "report should contain status line"
        );
    }

    // -----------------------------------------------------------------------
    // ValidationIssue builder helpers
    // -----------------------------------------------------------------------

    #[test]
    fn test_issue_constructors() {
        let e = ValidationIssue::error("field_a", "some error").with_suggestion("fix it this way");
        assert_eq!(e.severity, IssueSeverity::Error);
        assert_eq!(e.field, "field_a");
        assert_eq!(e.message, "some error");
        assert_eq!(e.suggestion.as_deref(), Some("fix it this way"));

        let w = ValidationIssue::warning("field_b", "some warning");
        assert_eq!(w.severity, IssueSeverity::Warning);
        assert!(w.suggestion.is_none());

        let i = ValidationIssue::info("field_c", "some info");
        assert_eq!(i.severity, IssueSeverity::Info);
    }
}
