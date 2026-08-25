//! Gradient verification test module.
//!
//! This module provides utilities for verifying analytical gradients against
//! numerical gradients computed via finite-difference approximation.
//!
//! # Test Strategy
//!
//! 1. **Setup**: Create simple test scene (1-10 Gaussians)
//! 2. **Forward**: Render image, compute loss
//! 3. **Backward**: Compute analytical gradients via the GPU backward pass
//!    (see [`compute_analytical_gradients_sync`]; requires a GPU adapter,
//!    checked at runtime by [`gpu_available`] rather than statically
//!    `#[ignore]`d)
//! 4. **Verify**: Compare with numerical gradients from finite-difference
//! 5. **Assert**: `median_error < MEDIAN_ERROR_THRESHOLD`

use nalgebra as na;
use oxigaf_render::config::RasterConfig;
use oxigaf_render::gaussian::{GaussianAttributes, GaussianModel};
use oxigaf_render::{CpuCamera, RenderError};

pub mod finite_diff;
pub mod test_opacity;
pub mod test_position;
pub mod test_rotation;
pub mod test_scale;
pub mod test_sh;

// Re-export commonly used types
pub use finite_diff::{
    compute_opacity_gradients, compute_position_gradients, compute_relative_error,
    compute_rotation_gradients, compute_scale_gradients, compute_sh_gradients, FiniteDiffConfig,
    MseLoss,
};

/// Median error threshold for gradient verification.
///
/// The median is naturally robust to outliers regardless of sample size.
/// At least 50% of gradient entries must match within this threshold.
pub const MEDIAN_ERROR_THRESHOLD: f32 = 5e-2;

/// Position-specific median error threshold for gradient verification.
///
/// Position gradients through a tiled rasterizer have higher finite-difference error
/// because position perturbation directly affects tile assignment, causing discontinuities
/// in the forward pass that the backward pass (correctly) doesn't model.
pub const POSITION_MEDIAN_ERROR_THRESHOLD: f32 = 2.5e-1;

/// Maximum fraction of entries allowed to be outliers (error > 0.5).
pub const MAX_OUTLIER_FRACTION: f32 = 0.3;

/// Compute median of a (mutable) error vector.
///
/// The median is naturally robust to outliers regardless of sample size.
/// On success the vector is left sorted in-place. Returns `f32::NAN` -
/// without sorting - if `errors` is empty or contains any non-finite (NaN)
/// entry, so a downstream `median_err < THRESHOLD` assertion always fails
/// on either condition instead of silently reporting a passing median.
pub fn median_error(errors: &mut [f32]) -> f32 {
    if errors.is_empty() {
        // Nothing was actually compared. Returning a "clean" 0.0 here would
        // let every call site's `median_err < THRESHOLD` assertion silently
        // pass on an empty comparison, so surface NaN instead (NaN compares
        // `false` against everything, including `<`).
        return f32::NAN;
    }
    if errors.iter().any(|e| e.is_nan()) {
        // `partial_cmp` returns `None` for any comparison involving NaN, so
        // the `unwrap_or(Ordering::Equal)` below treats a NaN as tied with
        // every other value and lets `sort_by` place it anywhere - silently
        // picking an arbitrary "median" instead of surfacing that a
        // backward shader produced a non-finite gradient. Return NaN
        // instead so `median_err < THRESHOLD` assertions at call sites
        // correctly evaluate to `false` and the test fails loudly.
        return f32::NAN;
    }
    errors.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = errors.len();
    if n.is_multiple_of(2) {
        (errors[n / 2 - 1] + errors[n / 2]) / 2.0
    } else {
        errors[n / 2]
    }
}

/// Test scene configuration.
#[derive(Debug, Clone)]
pub struct TestSceneConfig {
    /// Number of Gaussians in the scene.
    pub num_gaussians: usize,
    /// Image resolution (width, height).
    pub resolution: (u32, u32),
    /// SH degree (0-3).
    pub sh_degree: u32,
    /// Random seed for reproducibility.
    pub seed: u64,
}

impl Default for TestSceneConfig {
    fn default() -> Self {
        Self {
            num_gaussians: 5,
            resolution: (128, 128),
            sh_degree: 0,
            seed: 42,
        }
    }
}

/// Create a simple test scene with random Gaussians.
///
/// Gaussians are positioned in front of the camera with random rotations,
/// scales, and colors for gradient testing.
pub fn create_test_scene(config: &TestSceneConfig) -> Result<GaussianModel, RenderError> {
    // Use a simple deterministic pattern based on seed
    let mut gaussians = Vec::new();
    let mut sh_coeffs = Vec::new();
    let sh_coeffs_per_gaussian = ((config.sh_degree + 1) * (config.sh_degree + 1) * 3) as usize;

    for i in 0..config.num_gaussians {
        let offset = (i as f32 + config.seed as f32 * 0.01) * 0.1;

        // Position: in front of camera, slight offset per Gaussian
        let position = [
            (offset * 3.0).sin() * 0.5,
            (offset * 5.0).sin() * 0.5,
            -3.0 - offset, // Behind near plane
        ];

        // Rotation: slight variation per Gaussian
        let angle = offset;
        let axis = na::Vector3::new(0.0, 1.0, 0.0);
        let quat = na::UnitQuaternion::from_axis_angle(&na::Unit::new_normalize(axis), angle);
        let rotation = [quat.coords.x, quat.coords.y, quat.coords.z, quat.coords.w];

        // Scale: log-space, so exp(scale) gives actual scale
        let scale = [
            -1.0 + offset * 0.1,
            -1.0 + offset * 0.1,
            -1.0 + offset * 0.1,
        ];

        // Opacity: sigmoid-inverse space
        let opacity = offset * 0.5;

        gaussians.push(GaussianAttributes {
            position,
            _pad0: 0.0,
            rotation,
            scale,
            opacity,
        });

        // SH coefficients: deterministic colors based on index
        for j in 0..sh_coeffs_per_gaussian {
            sh_coeffs.push(((i * sh_coeffs_per_gaussian + j) as f32 * 0.01).sin() * 0.5);
        }
    }

    let n = gaussians.len();
    let third = 1.0_f32 / 3.0_f32;

    Ok(GaussianModel {
        gaussians,
        sh_coeffs,
        sh_degree: config.sh_degree,
        // These synthetic gradient-test scenes have no real FLAME mesh
        // binding, but every `GaussianModel` invariant is that the binding
        // arrays are parallel to `gaussians` (one entry per Gaussian) - code
        // that indexes them in lockstep (FLAME deform, density control
        // clone/split) would panic or read misaligned data otherwise.
        // Defaults mirror `GaussianModel::load_ply`'s "no binding" case.
        face_indices: vec![0u32; n],
        barycentric: vec![[third, third, third]; n],
        local_offsets: vec![[0.0, 0.0, 0.0]; n],
        is_rigid: vec![false; n],
    })
}

/// Build a [`GaussianModel`] from explicit per-Gaussian attributes.
///
/// Gradient tests that need a *specific* configuration (extreme anisotropy,
/// a chosen opacity, a hand-picked SH coefficient set) cannot use
/// [`create_test_scene`], which only emits its own deterministic pattern.
/// They previously hand-wrote a `GaussianModel` literal with **empty**
/// `face_indices` / `barycentric` / `local_offsets` / `is_rigid` vectors,
/// which violates the type's invariant that every binding array is parallel
/// to `gaussians` - code that indexes them in lockstep (FLAME deform,
/// density-control clone/split) panics or reads misaligned data on such a
/// model.
///
/// This helper builds the same "no FLAME binding" defaults that
/// [`create_test_scene`] and `GaussianModel::load_ply` use, sized to
/// `gaussians`, so a test can pick its own attributes without re-deriving
/// (or forgetting) the invariant.
pub fn model_from_gaussians(
    gaussians: Vec<GaussianAttributes>,
    sh_coeffs: Vec<f32>,
    sh_degree: u32,
) -> GaussianModel {
    let n = gaussians.len();
    let third = 1.0_f32 / 3.0_f32;

    GaussianModel {
        gaussians,
        sh_coeffs,
        sh_degree,
        face_indices: vec![0u32; n],
        barycentric: vec![[third, third, third]; n],
        local_offsets: vec![[0.0, 0.0, 0.0]; n],
        is_rigid: vec![false; n],
    }
}

/// Create a test camera looking at the origin.
pub fn create_test_camera(resolution: (u32, u32)) -> CpuCamera {
    let (width, height) = resolution;

    // Camera positioned at (0, 0, 0) looking down -Z axis
    let view = na::Matrix4::look_at_rh(
        &na::Point3::new(0.0, 0.0, 0.0),
        &na::Point3::new(0.0, 0.0, -1.0),
        &na::Vector3::y(),
    );

    // Simple perspective projection
    let fov_y = 45.0f32.to_radians();
    let aspect = width as f32 / height as f32;
    let near = 0.1;
    let far = 100.0;
    let proj = na::Matrix4::new_perspective(aspect, fov_y, near, far);

    // Focal lengths in pixels
    let focal_y = height as f32 / (2.0 * (fov_y / 2.0).tan());
    let focal_x = focal_y; // Square pixels
    let focal = na::Vector2::new(focal_x, focal_y);

    CpuCamera {
        view,
        proj,
        position: na::Vector3::zeros(),
        focal,
    }
}

/// Create a target image (all black for MSE loss).
pub fn create_target_image(resolution: (u32, u32)) -> Vec<f32> {
    let (width, height) = resolution;
    vec![0.0; (width * height * 4) as usize]
}

/// Gradient verification result.
#[derive(Debug, Clone)]
pub struct GradientVerificationResult {
    /// Maximum relative error across all *finite* gradients (0.0 if none
    /// were finite).
    pub max_error: f32,
    /// Mean relative error. May be non-finite if any input error was
    /// non-finite, or if `errors` was empty.
    pub mean_error: f32,
    /// Number of gradients checked.
    pub num_gradients: usize,
    /// Number of non-finite (NaN or +/-Inf) relative errors encountered.
    /// A backward shader emitting a non-finite gradient is always a bug,
    /// not merely an out-of-tolerance value, so any non-zero count fails
    /// verification regardless of `max_error`.
    pub num_non_finite: usize,
    /// Whether verification passed: `errors` was non-empty, every entry was
    /// finite, and the maximum finite error was below `threshold`.
    pub passed: bool,
}

impl GradientVerificationResult {
    /// Create a new verification result.
    ///
    /// Fails verification (`passed = false`) when `errors` is empty (nothing
    /// was actually compared) or contains any non-finite entry, even though
    /// naively folding with `f32::max` would silently discard NaNs (it
    /// returns the non-NaN operand) and `sum() / 0` on an empty slice would
    /// produce a `mean_error` of NaN while `max_error` stayed `0.0` - both
    /// of which previously left `passed` incorrectly `true`.
    pub fn new(errors: &[f32], threshold: f32) -> Self {
        let num_gradients = errors.len();
        let num_non_finite = errors.iter().filter(|e| !e.is_finite()).count();

        if num_gradients == 0 {
            return Self {
                max_error: f32::INFINITY,
                mean_error: f32::INFINITY,
                num_gradients,
                num_non_finite,
                passed: false,
            };
        }

        let max_error = errors
            .iter()
            .copied()
            .filter(|e| e.is_finite())
            .fold(0.0f32, f32::max);
        let mean_error = errors.iter().sum::<f32>() / num_gradients as f32;
        let passed = num_non_finite == 0 && max_error < threshold;

        Self {
            max_error,
            mean_error,
            num_gradients,
            num_non_finite,
            passed,
        }
    }
}

/// Compare two gradient arrays and compute relative errors.
pub fn compare_gradients_3d(analytical: &[[f32; 3]], numerical: &[[f32; 3]]) -> Vec<f32> {
    analytical
        .iter()
        .zip(numerical.iter())
        .flat_map(|(a, n)| {
            vec![
                compute_relative_error(a[0], n[0]),
                compute_relative_error(a[1], n[1]),
                compute_relative_error(a[2], n[2]),
            ]
        })
        .collect()
}

/// Compare two gradient arrays (4D) and compute relative errors.
pub fn compare_gradients_4d(analytical: &[[f32; 4]], numerical: &[[f32; 4]]) -> Vec<f32> {
    analytical
        .iter()
        .zip(numerical.iter())
        .flat_map(|(a, n)| {
            vec![
                compute_relative_error(a[0], n[0]),
                compute_relative_error(a[1], n[1]),
                compute_relative_error(a[2], n[2]),
                compute_relative_error(a[3], n[3]),
            ]
        })
        .collect()
}

/// Compare two gradient arrays (1D) and compute relative errors.
pub fn compare_gradients_1d(analytical: &[f32], numerical: &[f32]) -> Vec<f32> {
    analytical
        .iter()
        .zip(numerical.iter())
        .map(|(a, n)| compute_relative_error(*a, *n))
        .collect()
}

/// Compute analytical gradients using GPU backward pass.
///
/// This function:
/// 1. Creates a GPU rasterizer
/// 2. Runs forward pass to render the image
/// 3. Computes loss gradient (simple MSE gradient: 2*(rendered - target))
/// 4. Runs backward pass to get per-Gaussian gradients
///
/// Returns analytical gradients that can be compared with numerical gradients.
pub async fn compute_analytical_gradients(
    model: &GaussianModel,
    camera: &CpuCamera,
    target: &[f32],
    config: &RasterConfig,
) -> Result<oxigaf_render::GaussianGradients, RenderError> {
    use oxigaf_render::{Rasterizer, RenderCamera};

    // Create GPU rasterizer
    let mut rasterizer = Rasterizer::new(config.clone()).await?;

    // Convert CpuCamera to RenderCamera
    let view_matrix: [f32; 16] = camera
        .view
        .as_slice()
        .try_into()
        .map_err(|_| RenderError::Rasterize("Failed to convert view matrix".into()))?;
    let proj_matrix: [f32; 16] = camera
        .proj
        .as_slice()
        .try_into()
        .map_err(|_| RenderError::Rasterize("Failed to convert proj matrix".into()))?;

    let render_camera = RenderCamera {
        view_matrix,
        proj_matrix,
        position: [camera.position.x, camera.position.y, camera.position.z],
        focal: [camera.focal.x, camera.focal.y],
    };

    // Upload Gaussians
    rasterizer.upload_gaussians(model);

    // Forward pass
    let output = rasterizer.forward(model, &render_camera)?;

    // Compute loss gradient: ∂L/∂rendered = 2 * (rendered - target) / N for RGB, 0 for alpha
    // RGB-only MSE matches the GPU backward pass which only processes RGB channels
    let num_pixels = target.len() / 4;
    let n_rgb = (num_pixels * 3) as f32;
    let grad_image: Vec<f32> = output
        .color_data
        .chunks(4)
        .zip(target.chunks(4))
        .flat_map(|(rendered, target_chunk)| {
            [
                2.0 * (rendered[0] - target_chunk[0]) / n_rgb,
                2.0 * (rendered[1] - target_chunk[1]) / n_rgb,
                2.0 * (rendered[2] - target_chunk[2]) / n_rgb,
                0.0, // Zero alpha gradient - GPU backward only processes RGB
            ]
        })
        .collect();

    // Backward pass
    let gradients = rasterizer.backward(model, &grad_image)?;

    Ok(gradients)
}

/// Synchronous wrapper for compute_analytical_gradients using pollster.
///
/// This is more convenient for tests that don't want to deal with async.
pub fn compute_analytical_gradients_sync(
    model: &GaussianModel,
    camera: &CpuCamera,
    target: &[f32],
    config: &RasterConfig,
) -> Result<oxigaf_render::GaussianGradients, RenderError> {
    pollster::block_on(compute_analytical_gradients(model, camera, target, config))
}

/// Cached result of probing for a usable GPU adapter (see [`gpu_available`]).
static GPU_AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

/// Whether a compatible GPU adapter is available for the analytical
/// (GPU backward-pass) gradient comparison tests in this environment.
///
/// The comparison tests construct a real [`oxigaf_render::Rasterizer`],
/// which fails with `RenderError::GpuInit`/`AdapterNotFound` on a machine
/// with no compatible adapter (e.g. many headless CI runners). Rather than
/// permanently `#[ignore]`ing those tests - which lets the whole suite
/// report green without a single backward shader ever having run - each
/// comparison test calls this at the top and returns early when it is
/// `false`, so the test actually executes (and gates CI) on any machine
/// that does have a GPU.
///
/// The result is cached in a [`std::sync::OnceLock`] (safe under the test
/// harness's default parallel execution) since probing constructs a full
/// `Rasterizer`; this is only done once per test binary run.
pub fn gpu_available() -> bool {
    *GPU_AVAILABLE.get_or_init(|| {
        let probe_config = RasterConfig::new();
        match pollster::block_on(oxigaf_render::Rasterizer::new(probe_config)) {
            Ok(_) => true,
            Err(err) => {
                eprintln!(
                    "skipping GPU-dependent gradient test: no compatible GPU adapter available ({err})"
                );
                false
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_test_scene() {
        let config = TestSceneConfig::default();
        let scene = create_test_scene(&config);
        assert!(scene.is_ok());

        let model = scene.ok().unwrap_or_else(|| {
            panic!("Failed to create test scene");
        });
        assert_eq!(model.len(), config.num_gaussians);
    }

    /// Regression test: the FLAME binding arrays must be parallel to
    /// `gaussians` (one entry per Gaussian), not left empty. Code that
    /// indexes them in lockstep with `gaussians` (FLAME deform, density
    /// control clone/split) would otherwise panic or read misaligned data.
    #[test]
    fn test_create_test_scene_binding_arrays_match_gaussian_count() {
        let config = TestSceneConfig {
            num_gaussians: 4,
            ..TestSceneConfig::default()
        };
        let model = create_test_scene(&config).expect("Failed to create test scene");

        assert_eq!(model.gaussians.len(), config.num_gaussians);
        assert_eq!(model.face_indices.len(), config.num_gaussians);
        assert_eq!(model.barycentric.len(), config.num_gaussians);
        assert_eq!(model.local_offsets.len(), config.num_gaussians);
        assert_eq!(model.is_rigid.len(), config.num_gaussians);
    }

    /// Regression test: `model_from_gaussians` must size every FLAME binding
    /// array to the Gaussian count. The hand-written `GaussianModel` literals
    /// this helper replaces left them empty, breaking the type's invariant.
    #[test]
    fn test_model_from_gaussians_fills_binding_arrays() {
        let gaussian = GaussianAttributes {
            position: [0.0, 0.0, -3.0],
            _pad0: 0.0,
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [-1.0, -1.0, -1.0],
            opacity: 0.0,
        };
        let model = model_from_gaussians(vec![gaussian; 3], vec![0.5; 9], 0);

        assert_eq!(model.gaussians.len(), 3);
        assert_eq!(model.face_indices.len(), 3);
        assert_eq!(model.barycentric.len(), 3);
        assert_eq!(model.local_offsets.len(), 3);
        assert_eq!(model.is_rigid.len(), 3);
        assert_eq!(model.sh_degree, 0);
    }

    /// An empty Gaussian list must still produce empty (not mismatched)
    /// binding arrays.
    #[test]
    fn test_model_from_gaussians_empty() {
        let model = model_from_gaussians(Vec::new(), Vec::new(), 2);
        assert_eq!(model.len(), 0);
        assert!(model.face_indices.is_empty());
        assert!(model.barycentric.is_empty());
        assert!(model.local_offsets.is_empty());
        assert!(model.is_rigid.is_empty());
    }

    #[test]
    fn test_create_test_camera() {
        let camera = create_test_camera((128, 128));
        assert_eq!(camera.position, na::Vector3::zeros());
    }

    #[test]
    fn test_gradient_verification_result() {
        let errors = vec![0.0001, 0.0002, 0.0003];
        let result = GradientVerificationResult::new(&errors, 1e-3);

        assert!(result.passed);
        assert_eq!(result.max_error, 0.0003);
        assert!((result.mean_error - 0.0002).abs() < 1e-6);
        assert_eq!(result.num_non_finite, 0);
    }

    /// Regression test: an empty error list must not silently report PASS.
    /// Previously, `mean_error` divided by zero (`NaN`) while `max_error`
    /// stayed `0.0` and `passed` stayed `true`.
    #[test]
    fn test_gradient_verification_result_empty_fails() {
        let errors: Vec<f32> = vec![];
        let result = GradientVerificationResult::new(&errors, 1e-3);

        assert!(!result.passed, "an empty gradient comparison must not pass");
        assert_eq!(result.num_gradients, 0);
    }

    /// Regression test: a NaN relative error (e.g. from a backward shader
    /// emitting a non-finite gradient) must fail verification. Previously
    /// `f32::max` silently discarded the NaN (it returns the non-NaN
    /// operand), so `max_error` stayed at the tiny finite value and
    /// `passed` stayed `true`.
    #[test]
    fn test_gradient_verification_result_nan_fails() {
        let errors = vec![0.0001, f32::NAN, 0.0002];
        let result = GradientVerificationResult::new(&errors, 1e-3);

        assert!(!result.passed, "a non-finite error must fail verification");
        assert_eq!(result.num_non_finite, 1);
        // The finite errors are tiny: if the NaN were silently dropped (the
        // pre-fix behavior), max_error would stay under the threshold and
        // this would incorrectly report PASS.
        assert!(result.max_error < 1e-3);
    }

    /// Regression test: `median_error` must not let a NaN entry sort to an
    /// arbitrary position and return a misleadingly "passing" median.
    #[test]
    fn test_median_error_nan_propagates() {
        let mut errors = vec![0.1, f32::NAN, 0.2];
        let median = median_error(&mut errors);

        // NaN compares `false` against everything (a language guarantee),
        // so this is what makes every downstream `median_err < THRESHOLD`
        // assertion at call sites correctly fail.
        assert!(median.is_nan(), "NaN input must yield a NaN median");
    }

    /// Regression test: an empty error list must not report a "passing"
    /// median of `0.0`.
    #[test]
    fn test_median_error_empty_is_not_a_silent_pass() {
        let mut errors: Vec<f32> = vec![];
        let median = median_error(&mut errors);

        assert!(
            median.is_nan(),
            "empty error list must not report a passing median"
        );
    }

    #[test]
    fn test_compare_gradients_3d() {
        let analytical = vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let numerical = vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

        let errors = compare_gradients_3d(&analytical, &numerical);
        assert_eq!(errors.len(), 6);
        assert!(errors.iter().all(|&e| e < 1e-6));
    }
}
