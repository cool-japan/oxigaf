//! Builder patterns, convenience functions, validation utilities, and
//! platform-detection for the OxiGAF pipeline.

use std::path::{Path, PathBuf};

use super::OxigafError;

// ============================================================================
// PipelineConfig + PipelineBuilder
// ============================================================================

/// Validated, ready-to-use pipeline configuration.
///
/// Construct via [`PipelineBuilder`].
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Path to the directory containing FLAME model `.npy` files.
    pub flame_model_path: PathBuf,
    /// Directory where outputs (checkpoints, rendered images) will be written.
    pub output_dir: PathBuf,
    /// Number of novel views to generate during diffusion.
    pub num_views: usize,
    /// Number of 3DGS optimisation iterations.
    pub iterations: u32,
}

/// Fluent builder for the full OxiGAF pipeline configuration.
///
/// # Example
///
/// ```rust,ignore
/// use oxigaf::PipelineBuilder;
///
/// let config = PipelineBuilder::new()
///     .flame_model_path("/data/flame")
///     .output_dir("/tmp/output")
///     .num_views(8)
///     .iterations(30_000)
///     .build()?;
/// ```
pub struct PipelineBuilder {
    flame_model_path: Option<PathBuf>,
    output_dir: Option<PathBuf>,
    num_views: usize,
    iterations: u32,
}

impl PipelineBuilder {
    /// Create a new builder with sensible defaults.
    pub fn new() -> Self {
        Self {
            flame_model_path: None,
            output_dir: None,
            num_views: 8,
            iterations: 30_000,
        }
    }

    /// Set the path to the FLAME model directory.
    pub fn flame_model_path<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.flame_model_path = Some(path.as_ref().to_path_buf());
        self
    }

    /// Set the output directory.
    pub fn output_dir<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.output_dir = Some(path.as_ref().to_path_buf());
        self
    }

    /// Set the number of novel views to generate.
    pub fn num_views(mut self, n: usize) -> Self {
        self.num_views = n;
        self
    }

    /// Set the number of 3DGS optimisation iterations.
    pub fn iterations(mut self, n: u32) -> Self {
        self.iterations = n;
        self
    }

    /// Build and validate the configuration.
    ///
    /// Returns [`OxigafError::InvalidConfig`] if required fields are missing or
    /// values are out of range.
    pub fn build(self) -> Result<PipelineConfig, OxigafError> {
        let flame_model_path = self.flame_model_path.ok_or_else(|| {
            OxigafError::InvalidConfig("flame_model_path is required".to_string())
        })?;

        let output_dir = self
            .output_dir
            .ok_or_else(|| OxigafError::InvalidConfig("output_dir is required".to_string()))?;

        if self.num_views == 0 {
            return Err(OxigafError::InvalidConfig(
                "num_views must be at least 1".to_string(),
            ));
        }

        if self.iterations == 0 {
            return Err(OxigafError::InvalidConfig(
                "iterations must be at least 1".to_string(),
            ));
        }

        Ok(PipelineConfig {
            flame_model_path,
            output_dir,
            num_views: self.num_views,
            iterations: self.iterations,
        })
    }
}

impl Default for PipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Convenience functions
// ============================================================================

/// Quick one-liner training with default configuration.
///
/// Creates a [`PipelineConfig`] from the given paths, validates it, then
/// returns the output directory path on success.
///
/// This function does **not** actually invoke the trainer (which requires GPU
/// resources and heavy dependencies); it validates the configuration and
/// returns the resolved output path. Full training is available via
/// [`oxigaf_trainer::Trainer`].
///
/// # Errors
///
/// Returns [`OxigafError::InvalidConfig`] or [`OxigafError::PathNotFound`] if
/// the configuration is invalid or required paths do not exist.
pub fn quick_train<P: AsRef<Path>>(
    flame_model_path: P,
    output_dir: P,
) -> Result<PathBuf, OxigafError> {
    let config = PipelineBuilder::new()
        .flame_model_path(flame_model_path)
        .output_dir(output_dir)
        .build()?;

    validate_config(&config)?;
    Ok(config.output_dir)
}

/// Render a Gaussian model from file to an output image.
///
/// This is a thin validation wrapper; full rendering requires a live wgpu
/// device available through [`oxigaf_render::Rasterizer`].
///
/// # Errors
///
/// Returns [`OxigafError::PathNotFound`] if `model_path` does not exist, or
/// [`OxigafError::InvalidConfig`] for invalid dimensions.
pub fn render_from_file<P: AsRef<Path>>(
    model_path: P,
    output_path: P,
    width: u32,
    height: u32,
) -> Result<(), OxigafError> {
    let model_path = model_path.as_ref();
    if !model_path.exists() {
        return Err(OxigafError::PathNotFound(model_path.display().to_string()));
    }
    if width == 0 || height == 0 {
        return Err(OxigafError::InvalidConfig(
            "width and height must be greater than 0".to_string(),
        ));
    }
    let _ = output_path.as_ref(); // validated by caller; output created at render time
    Ok(())
}

/// Export format for [`export`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportFormat {
    /// Stanford PLY point cloud.
    Ply,
    /// GL Transmission Format 2.0.
    Gltf,
    /// Wavefront OBJ mesh.
    Obj,
}

/// Export a Gaussian model to a different format.
///
/// This is a thin validation wrapper. Actual serialisation is delegated to the
/// CLI or render subsystems once a runtime is available.
///
/// # Errors
///
/// Returns [`OxigafError::PathNotFound`] if `model_path` does not exist.
pub fn export<P: AsRef<Path>>(
    model_path: P,
    output_path: P,
    format: ExportFormat,
) -> Result<(), OxigafError> {
    let model_path = model_path.as_ref();
    if !model_path.exists() {
        return Err(OxigafError::PathNotFound(model_path.display().to_string()));
    }
    let _ = (output_path.as_ref(), format);
    Ok(())
}

// ============================================================================
// Validation utilities
// ============================================================================

/// GPU information obtained from wgpu adapter enumeration.
#[derive(Debug, Clone)]
pub struct GpuInfo {
    /// wgpu backend name (e.g. `"Metal"`, `"Vulkan"`, `"Dx12"`, `"Gl"`).
    pub backend: String,
    /// Human-readable adapter name from the driver.
    pub name: String,
    /// Adapter device type (`"Discrete"`, `"Integrated"`, `"Virtual"`, `"Cpu"`, `"Other"`).
    pub device_type: String,
}

/// Check available GPUs via synchronous wgpu adapter enumeration.
///
/// Returns `Ok(vec)` of [`GpuInfo`] structs. The vector may be empty on
/// headless or CI systems that expose no hardware adapters.
///
/// # Errors
///
/// Currently infallible at the API level — any driver failure results in an
/// empty list rather than an error. Returns `Err(OxigafError::GpuError)` only
/// if the wgpu instance itself cannot be constructed.
pub fn check_gpu() -> Result<Vec<GpuInfo>, OxigafError> {
    use wgpu::{Backends, Instance, InstanceDescriptor};

    let instance = Instance::new(InstanceDescriptor {
        backends: Backends::all(),
        ..InstanceDescriptor::new_without_display_handle()
    });

    let adapters = pollster::block_on(instance.enumerate_adapters(Backends::all()));
    let infos = adapters
        .into_iter()
        .map(|adapter| {
            let info = adapter.get_info();
            GpuInfo {
                backend: format!("{:?}", info.backend),
                name: info.name.clone(),
                device_type: format!("{:?}", info.device_type),
            }
        })
        .collect();

    Ok(infos)
}

/// Validate a [`PipelineConfig`] — checks paths exist and values are in range.
///
/// # Errors
///
/// Returns [`OxigafError::PathNotFound`] if `flame_model_path` does not exist,
/// or [`OxigafError::InvalidConfig`] for out-of-range values.
pub fn validate_config(config: &PipelineConfig) -> Result<(), OxigafError> {
    if !config.flame_model_path.exists() {
        return Err(OxigafError::PathNotFound(
            config.flame_model_path.display().to_string(),
        ));
    }
    if config.num_views == 0 {
        return Err(OxigafError::InvalidConfig(
            "num_views must be at least 1".to_string(),
        ));
    }
    if config.iterations == 0 {
        return Err(OxigafError::InvalidConfig(
            "iterations must be at least 1".to_string(),
        ));
    }
    Ok(())
}

/// Verify that expected asset files exist in a directory.
///
/// Returns a `Vec<String>` of file names that are **missing** from
/// `asset_dir`. An empty vec means all expected assets are present.
///
/// Currently checks for the canonical FLAME asset set:
/// `"shape_dirs.npy"`, `"exp_dirs.npy"`, `"posedirs.npy"`,
/// `"v_template.npy"`, `"J_regressor.npy"`, `"kintree_table.npy"`,
/// `"faces.npy"`.
pub fn verify_assets<P: AsRef<Path>>(asset_dir: P) -> Vec<String> {
    const EXPECTED: &[&str] = &[
        "shape_dirs.npy",
        "exp_dirs.npy",
        "posedirs.npy",
        "v_template.npy",
        "J_regressor.npy",
        "kintree_table.npy",
        "faces.npy",
    ];

    let dir = asset_dir.as_ref();
    EXPECTED
        .iter()
        .filter(|&&name| !dir.join(name).exists())
        .map(|&s| s.to_string())
        .collect()
}

// ============================================================================
// Platform detection
// ============================================================================

/// Detect the best available wgpu backend for this system.
///
/// On macOS returns `"Metal"`, on Linux `"Vulkan"`, on Windows `"Dx12"`.
/// Falls back to `"Gl"` when none of the primary backends are compiled in,
/// and to `"Unknown"` as a last resort.
///
/// The detection is purely compile-time / target-OS based; no GPU driver
/// enumeration is performed.
pub fn detect_best_backend() -> String {
    #[cfg(target_os = "macos")]
    {
        "Metal".to_string()
    }
    #[cfg(all(target_os = "linux", not(target_os = "macos")))]
    {
        "Vulkan".to_string()
    }
    #[cfg(all(target_os = "windows", not(target_os = "macos")))]
    {
        "Dx12".to_string()
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        "Gl".to_string()
    }
}
