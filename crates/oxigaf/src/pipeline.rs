//! Builder patterns, convenience functions, validation utilities, and
//! platform-detection for the OxiGAF pipeline.

use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use oxigaf_render::camera_path::{keyframe_to_render_camera, CameraKeyframe};
use oxigaf_render::gaussian::GaussianModel;
use oxigaf_render::gltf::GltfError;
use oxigaf_render::{RasterConfig, Rasterizer, RenderCamera};

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
pub fn quick_train<P: AsRef<Path>, Q: AsRef<Path>>(
    flame_model_path: P,
    output_dir: Q,
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
/// Loads `model_path` (a binary 3DGS `.ply` or a `.safetensors` checkpoint),
/// frames a camera around the model's bounding box, runs the wgpu rasterizer
/// through [`Rasterizer`], and writes the result to `output_path`. The image
/// encoding is chosen from the `output_path` extension (`.png`, `.jpg`, …).
///
/// Missing parent directories of `output_path` are created.
///
/// # Camera framing
///
/// The camera looks along `-Z` at the centre of the Gaussians' bounding box
/// from `2.5 ×` the box's half-extent, clamped to `[0.1, 40.0]` so the model
/// stays inside the rasterizer's default `0.01 … 100.0` frustum. Models whose
/// half-extent exceeds roughly `45` world units are therefore partially
/// clipped; use [`Rasterizer`] directly with a custom [`RenderCamera`] for
/// full control.
///
/// # Errors
///
/// Returns [`OxigafError::PathNotFound`] if `model_path` does not exist,
/// [`OxigafError::InvalidConfig`] for invalid dimensions, an unsupported model
/// extension or an empty model, [`OxigafError::Render`] if the model cannot be
/// parsed or no GPU adapter is available, and [`OxigafError::Io`] if the image
/// cannot be written.
pub fn render_from_file<P: AsRef<Path>, Q: AsRef<Path>>(
    model_path: P,
    output_path: Q,
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
    let output_path = output_path.as_ref();

    let model = load_gaussian_model(model_path)?;
    if model.is_empty() {
        return Err(OxigafError::InvalidConfig(format!(
            "model contains no Gaussians: {}",
            model_path.display()
        )));
    }

    let camera = frame_camera(&model, width, height);
    let config = RasterConfig {
        image_width: width,
        image_height: height,
        // Must mirror the model, otherwise the SH buffers are sized wrongly.
        sh_degree: model.sh_degree,
        ..RasterConfig::default()
    };

    let mut rasterizer = pollster::block_on(Rasterizer::new(config))?;
    let output = rasterizer.forward(&model, &camera)?;
    let rendered = rasterizer.download_image(&output);

    create_parent_dir(output_path)?;
    rendered.save(output_path).map_err(|e| {
        OxigafError::Io(std::io::Error::other(format!(
            "failed to write rendered image to {}: {e}",
            output_path.display()
        )))
    })?;

    Ok(())
}

/// Create the parent directory of `path` if it does not already exist.
fn create_parent_dir(path: &Path) -> Result<(), OxigafError> {
    match path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => {
            std::fs::create_dir_all(parent)?;
            Ok(())
        }
        _ => Ok(()),
    }
}

/// Load a [`GaussianModel`], selecting the reader from the file extension.
///
/// `.ply` uses the binary 3DGS reader, `.safetensors` the checkpoint reader.
fn load_gaussian_model(path: &Path) -> Result<GaussianModel, OxigafError> {
    let extension = path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase)
        .unwrap_or_default();

    match extension.as_str() {
        "ply" => Ok(GaussianModel::load_ply(path)?),
        "safetensors" => Ok(GaussianModel::load_safetensors(path)?),
        other => Err(OxigafError::InvalidConfig(format!(
            "unsupported Gaussian model extension {other:?} for {} \
             — expected \"ply\" or \"safetensors\"",
            path.display()
        ))),
    }
}

/// Axis-aligned bounding box over the finite Gaussian centres.
///
/// Returns `None` when the model is empty or every position is non-finite.
fn bounding_box(model: &GaussianModel) -> Option<([f32; 3], [f32; 3])> {
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];

    for gaussian in &model.gaussians {
        let p = gaussian.position;
        if !p[0].is_finite() || !p[1].is_finite() || !p[2].is_finite() {
            continue;
        }
        min[0] = min[0].min(p[0]);
        min[1] = min[1].min(p[1]);
        min[2] = min[2].min(p[2]);
        max[0] = max[0].max(p[0]);
        max[1] = max[1].max(p[1]);
        max[2] = max[2].max(p[2]);
    }

    if min.iter().all(|v| v.is_finite()) && max.iter().all(|v| v.is_finite()) {
        Some((min, max))
    } else {
        None
    }
}

/// Build a camera that frames the whole model.
///
/// `keyframe_to_render_camera` bakes the rasterizer's default near/far planes
/// (`0.01` / `100.0`) into the projection matrix, so the eye distance is
/// clamped to keep the geometry inside that frustum rather than silently
/// rendering an empty image.
fn frame_camera(model: &GaussianModel, width: u32, height: u32) -> RenderCamera {
    let (center, radius) = match bounding_box(model) {
        Some((min, max)) => {
            let center = [
                0.5 * (min[0] + max[0]),
                0.5 * (min[1] + max[1]),
                0.5 * (min[2] + max[2]),
            ];
            let extent = (max[0] - min[0]).max(max[1] - min[1]).max(max[2] - min[2]);
            let radius = 0.5 * extent;
            let radius = if radius.is_finite() && radius > 1e-4 {
                radius
            } else {
                1.0
            };
            (center, radius)
        }
        None => ([0.0, 0.0, 0.0], 1.0),
    };

    let distance = (radius * 2.5).clamp(0.1, 40.0);
    let eye = [center[0], center[1], center[2] + distance];
    let keyframe = CameraKeyframe::look_from_to(0.0, eye, center);
    keyframe_to_render_camera(&keyframe, width as usize, height as usize)
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
/// `model_path` is read with the loader matching its extension (`.ply` or
/// `.safetensors`) and re-serialised to `output_path` in `format`:
///
/// | Format | Output |
/// |--------|--------|
/// | [`ExportFormat::Ply`] | Binary little-endian 3DGS PLY (SIBR-viewer compatible) |
/// | [`ExportFormat::Obj`] | Wavefront OBJ point cloud, one vertex per Gaussian |
/// | [`ExportFormat::Gltf`] | glTF 2.0 JSON at `output_path` plus a `.bin` buffer sidecar |
///
/// Missing parent directories of `output_path` are created.
///
/// # Errors
///
/// Returns [`OxigafError::PathNotFound`] if `model_path` does not exist,
/// [`OxigafError::InvalidConfig`] for an unsupported input extension or an
/// unusable output path, [`OxigafError::Render`] if the model cannot be parsed
/// or written, and [`OxigafError::Io`] for filesystem failures.
pub fn export<P: AsRef<Path>, Q: AsRef<Path>>(
    model_path: P,
    output_path: Q,
    format: ExportFormat,
) -> Result<(), OxigafError> {
    let model_path = model_path.as_ref();
    if !model_path.exists() {
        return Err(OxigafError::PathNotFound(model_path.display().to_string()));
    }
    let output_path = output_path.as_ref();

    let model = load_gaussian_model(model_path)?;
    create_parent_dir(output_path)?;

    match format {
        ExportFormat::Ply => model.save_ply(output_path)?,
        ExportFormat::Obj => write_obj(&model, output_path)?,
        ExportFormat::Gltf => write_gltf(&model, output_path)?,
    }

    Ok(())
}

// ----------------------------------------------------------------------------
// Export writers
// ----------------------------------------------------------------------------

/// Degree-0 spherical-harmonics basis constant, `0.5 / sqrt(pi)`.
const SH_C0: f32 = 0.282_094_79;

/// Number of spherical-harmonics floats stored per Gaussian: `(degree + 1)² × 3`.
fn sh_coeffs_per_gaussian(sh_degree: u32) -> usize {
    let bands = (sh_degree + 1) as usize;
    bands * bands * 3
}

/// Convert the degree-0 SH coefficients starting at `base` into linear RGB.
///
/// Uses the standard 3DGS convention `color = 0.5 + SH_C0 × f_dc`, clamped to
/// `[0, 1]`. Missing coefficients are treated as zero (mid grey).
fn dc_color(sh_coeffs: &[f32], base: usize) -> [f32; 3] {
    let channel = |offset: usize| {
        let dc = sh_coeffs.get(base + offset).copied().unwrap_or(0.0);
        (0.5 + SH_C0 * dc).clamp(0.0, 1.0)
    };
    [channel(0), channel(1), channel(2)]
}

/// Write a Gaussian model as a Wavefront OBJ point cloud.
///
/// One `v` line per Gaussian centre. The trailing `r g b` triple is the
/// widely-supported vertex-colour extension to OBJ rather than part of the
/// original specification; readers that do not understand it ignore the three
/// extra floats.
fn write_obj(model: &GaussianModel, path: &Path) -> Result<(), OxigafError> {
    let file = std::fs::File::create(path)?;
    let mut w = BufWriter::new(file);
    let count = model.len();
    let stride = sh_coeffs_per_gaussian(model.sh_degree);

    writeln!(
        w,
        "# Wavefront OBJ point cloud generated by OxiGAF v{}",
        env!("CARGO_PKG_VERSION")
    )?;
    writeln!(w, "# {count} Gaussians, SH degree {}", model.sh_degree)?;
    writeln!(
        w,
        "# 'v x y z r g b' vertex colours are an OBJ extension, not base OBJ"
    )?;
    writeln!(w, "o OxiGAF_GaussianCloud")?;

    for (index, gaussian) in model.gaussians.iter().enumerate() {
        let [red, green, blue] = dc_color(&model.sh_coeffs, index * stride);
        let [x, y, z] = gaussian.position;
        writeln!(w, "v {x} {y} {z} {red} {green} {blue}")?;
    }

    w.flush()?;
    Ok(())
}

/// Write a Gaussian model as a glTF 2.0 document plus a binary buffer sidecar.
///
/// This is a thin adapter over [`oxigaf_render::gltf::write_gltf`], which is
/// the workspace's single glTF writer — see that module for the file layout,
/// the binary buffer layout, and the specification requirements it satisfies.
///
/// The implementation used to live here, duplicated (incompatibly) by
/// `oxigaf_cli::export_gltf` and again by `oxigaf_cli::export`'s GLB writer.
/// Three emitters sharing one format name meant a consumer written against one
/// silently mis-read the others; hoisting the writer into `oxigaf-render`,
/// which both crates already depend on, is what makes that impossible.
///
/// Only the error type differs from the hoisted writer: a path that collides
/// with its own `.bin` sidecar (or yields no usable buffer file name) is a
/// caller mistake and keeps surfacing as [`OxigafError::InvalidConfig`], while
/// a filesystem failure stays an [`OxigafError::Io`] carrying the original
/// [`std::io::Error`] — the two must not collapse into one message, or a full
/// disk would be reported as a bad output path.
fn write_gltf(model: &GaussianModel, path: &Path) -> Result<(), OxigafError> {
    oxigaf_render::gltf::write_gltf(model, path).map_err(|e| match e {
        GltfError::InvalidOutputPath(message) => {
            OxigafError::InvalidConfig(format!("glTF output path {message}"))
        }
        GltfError::Io {
            action,
            path,
            source,
        } => OxigafError::Io(std::io::Error::new(
            source.kind(),
            format!("cannot {action} {}: {source}", path.display()),
        )),
    })
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
/// empty list rather than an error, so this always returns `Ok`. The
/// `Result` is retained for forward compatibility, so that a future backend
/// that can report enumeration failures does not need a breaking signature
/// change.
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
/// The checked set is not spelled out here: it is
/// [`oxigaf_flame::io::REQUIRED_NPY_FILES`], the same constant
/// `oxigaf_flame::io::load_flame_model` names each of its `.npy` inputs with.
/// Sharing the constant — rather than repeating the names in a second list, as
/// this function used to — is what stops the two from drifting apart, which
/// they already had once (`shape_dirs.npy` / `J_regressor.npy` were checked
/// while the loader opened `shapedirs.npy` / `j_regressor.npy`).
///
/// The names are compared exactly as spelled, so a directory that satisfies
/// this check also loads on case-sensitive filesystems.
pub fn verify_assets<P: AsRef<Path>>(asset_dir: P) -> Vec<String> {
    let dir = asset_dir.as_ref();
    oxigaf_flame::io::REQUIRED_NPY_FILES
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
/// On macOS returns `"Metal"`, on Linux `"Vulkan"`, on Windows `"Dx12"`, and
/// `"Gl"` on every other target. Those four strings are the complete set of
/// possible results.
///
/// The detection is purely compile-time / target-OS based; no GPU driver
/// enumeration is performed.
pub fn detect_best_backend() -> String {
    #[cfg(target_os = "macos")]
    {
        "Metal".to_string()
    }
    #[cfg(target_os = "linux")]
    {
        "Vulkan".to_string()
    }
    #[cfg(target_os = "windows")]
    {
        "Dx12".to_string()
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        "Gl".to_string()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    use oxigaf_render::gaussian::GaussianAttributes;

    /// The exact file set read by `oxigaf_flame::io::load_flame_model`.
    ///
    /// This deliberately aliases the loader's own constant rather than
    /// restating the names: a second literal list here would be a third copy
    /// able to drift, and a test that agrees with a stale copy of the list
    /// proves nothing. `oxigaf_flame` owns the "is every listed name really
    /// opened?" direction (see its `io::tests`); these tests only assert that
    /// `verify_assets` checks exactly that set.
    const LOADER_NPY_FILES: &[&str] = oxigaf_flame::io::REQUIRED_NPY_FILES;

    /// Create a fresh temporary directory dedicated to one test.
    fn temp_subdir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("oxigaf_pipeline_{name}"));
        let _ = std::fs::remove_dir_all(&dir);
        if let Err(e) = std::fs::create_dir_all(&dir) {
            panic!("cannot create temp dir {}: {e}", dir.display());
        }
        dir
    }

    /// A small, deterministic SH-degree-0 Gaussian model.
    fn sample_model(count: usize) -> GaussianModel {
        let gaussians = (0..count)
            .map(|i| {
                let f = i as f32;
                GaussianAttributes {
                    position: [f, f * 0.5, -f],
                    _pad0: 0.0,
                    rotation: [0.0, 0.0, 0.0, 1.0],
                    scale: [-2.0, -2.0, -2.0],
                    opacity: 0.5,
                }
            })
            .collect();

        GaussianModel {
            gaussians,
            sh_coeffs: vec![0.25; count * 3],
            sh_degree: 0,
            face_indices: vec![0; count],
            barycentric: vec![[1.0 / 3.0; 3]; count],
            local_offsets: vec![[0.0; 3]; count],
            is_rigid: vec![false; count],
        }
    }

    /// Write `model` to `path` as PLY, failing the test on error.
    fn write_source_ply(model: &GaussianModel, path: &Path) {
        if let Err(e) = model.save_ply(path) {
            panic!("could not write source PLY {}: {e}", path.display());
        }
    }

    /// Write raw bytes, failing the test on error.
    fn write_bytes(path: &Path, bytes: &[u8]) {
        if let Err(e) = std::fs::write(path, bytes) {
            panic!("could not write {}: {e}", path.display());
        }
    }

    /// Read a whole file as text, failing the test on error.
    fn read_text(path: &Path) -> String {
        match std::fs::read_to_string(path) {
            Ok(text) => text,
            Err(e) => panic!("could not read {}: {e}", path.display()),
        }
    }

    // ---- verify_assets ----

    #[test]
    fn verify_assets_matches_the_flame_loader_file_set() {
        let dir = temp_subdir("verify_assets_loader_set");

        let missing = verify_assets(&dir);
        assert_eq!(
            missing.len(),
            LOADER_NPY_FILES.len(),
            "every loader input must be reported for an empty directory: {missing:?}"
        );
        for name in LOADER_NPY_FILES {
            assert!(
                missing.contains(&(*name).to_string()),
                "{name} is read by the FLAME loader and must be checked"
            );
        }

        for name in LOADER_NPY_FILES {
            write_bytes(&dir.join(name), b"stub");
        }
        assert!(
            verify_assets(&dir).is_empty(),
            "a complete FLAME directory must report nothing missing"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn verify_assets_flags_a_missing_lbs_weights() {
        let dir = temp_subdir("verify_assets_lbs_weights");
        for name in LOADER_NPY_FILES {
            if *name != "lbs_weights.npy" {
                write_bytes(&dir.join(name), b"stub");
            }
        }

        assert_eq!(
            verify_assets(&dir),
            vec!["lbs_weights.npy".to_string()],
            "lbs_weights.npy is required by the loader and must be reported missing"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn verify_assets_ignores_names_the_loader_never_reads() {
        let dir = temp_subdir("verify_assets_legacy_names");
        let missing = verify_assets(&dir);
        for stale in ["shape_dirs.npy", "exp_dirs.npy", "J_regressor.npy"] {
            assert!(
                !missing.contains(&stale.to_string()),
                "{stale} is not read by the FLAME loader and must not be checked"
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn verify_assets_reports_the_loaders_own_list_verbatim() {
        // The drift guard itself: `verify_assets` must derive its file set
        // from `oxigaf_flame::io::REQUIRED_NPY_FILES`, not from a private copy
        // that can fall behind. Against an empty directory the reported
        // missing set is the checked set, so comparing it to the loader's
        // constant — order included — pins the wiring rather than the names.
        let dir = temp_subdir("verify_assets_shared_const");

        let expected: Vec<String> = oxigaf_flame::io::REQUIRED_NPY_FILES
            .iter()
            .map(|&s| s.to_string())
            .collect();
        assert_eq!(
            verify_assets(&dir),
            expected,
            "verify_assets must check exactly oxigaf_flame::io::REQUIRED_NPY_FILES"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ---- render_from_file ----

    #[test]
    fn render_from_file_reports_a_missing_model() {
        let dir = temp_subdir("render_missing_model");
        let result = render_from_file(dir.join("absent.ply"), dir.join("out.png"), 32, 32);
        assert!(
            matches!(result, Err(OxigafError::PathNotFound(_))),
            "got {result:?}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn render_from_file_rejects_zero_dimensions() {
        let dir = temp_subdir("render_zero_dimensions");
        let src = dir.join("model.ply");
        write_source_ply(&sample_model(2), &src);

        let zero_width = render_from_file(&src, dir.join("out.png"), 0, 64);
        assert!(
            matches!(zero_width, Err(OxigafError::InvalidConfig(_))),
            "got {zero_width:?}"
        );
        let zero_height = render_from_file(&src, dir.join("out.png"), 64, 0);
        assert!(
            matches!(zero_height, Err(OxigafError::InvalidConfig(_))),
            "got {zero_height:?}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn render_from_file_rejects_a_malformed_model() {
        // Regression: this used to return Ok(()) without loading, rendering or
        // writing anything at all.
        let dir = temp_subdir("render_malformed_ply");
        let src = dir.join("model.ply");
        let out = dir.join("out.png");
        write_bytes(&src, b"this is not a ply header\n");

        let result = render_from_file(&src, &out, 16, 16);
        assert!(
            matches!(result, Err(OxigafError::Render(_))),
            "a malformed model must be reported, not silently accepted; got {result:?}"
        );
        assert!(
            !out.exists(),
            "no image may be produced from a model that cannot be parsed"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn render_from_file_rejects_an_unknown_model_extension() {
        let dir = temp_subdir("render_unknown_extension");
        let src = dir.join("model.txt");
        write_bytes(&src, b"not a gaussian model");

        let result = render_from_file(&src, dir.join("out.png"), 16, 16);
        assert!(
            matches!(result, Err(OxigafError::InvalidConfig(_))),
            "got {result:?}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn render_from_file_rejects_an_empty_model() {
        let dir = temp_subdir("render_empty_model");
        let src = dir.join("model.ply");
        write_source_ply(&sample_model(0), &src);

        let result = render_from_file(&src, dir.join("out.png"), 16, 16);
        assert!(
            matches!(result, Err(OxigafError::InvalidConfig(_))),
            "got {result:?}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ---- export ----

    #[test]
    fn export_reports_a_missing_model() {
        let dir = temp_subdir("export_missing_model");
        let result = export(
            dir.join("absent.ply"),
            dir.join("out.ply"),
            ExportFormat::Ply,
        );
        assert!(
            matches!(result, Err(OxigafError::PathNotFound(_))),
            "got {result:?}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn export_ply_writes_a_reloadable_file() {
        // Regression: export() used to discard both the destination and the
        // format and return Ok(()) without writing a single byte.
        let dir = temp_subdir("export_ply");
        let src = dir.join("src.ply");
        let dst = dir.join("nested").join("dst.ply");
        write_source_ply(&sample_model(5), &src);

        let result = export(&src, &dst, ExportFormat::Ply);
        assert!(result.is_ok(), "export failed: {result:?}");
        assert!(dst.is_file(), "export must create {}", dst.display());

        match GaussianModel::load_ply(&dst) {
            Ok(reloaded) => {
                assert_eq!(reloaded.len(), 5, "every Gaussian must survive the export");
                assert_eq!(reloaded.sh_degree, 0);
            }
            Err(e) => panic!("the exported PLY does not reload: {e}"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn export_obj_writes_one_vertex_per_gaussian() {
        let dir = temp_subdir("export_obj");
        let src = dir.join("src.ply");
        let dst = dir.join("cloud.obj");
        write_source_ply(&sample_model(3), &src);

        let result = export(&src, &dst, ExportFormat::Obj);
        assert!(result.is_ok(), "export failed: {result:?}");

        let text = read_text(&dst);
        let vertices = text.lines().filter(|l| l.starts_with("v ")).count();
        assert_eq!(vertices, 3, "one 'v' line per Gaussian; got:\n{text}");
        assert!(
            text.contains("o OxiGAF_GaussianCloud"),
            "the OBJ must name its object; got:\n{text}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn export_gltf_writes_a_document_and_its_binary_buffer() {
        let dir = temp_subdir("export_gltf");
        let src = dir.join("src.ply");
        let dst = dir.join("scene.gltf");
        let sidecar = dir.join("scene.bin");
        let count = 4_usize;
        write_source_ply(&sample_model(count), &src);

        let result = export(&src, &dst, ExportFormat::Gltf);
        assert!(result.is_ok(), "export failed: {result:?}");
        assert!(dst.is_file(), "the glTF document must exist");
        assert!(sidecar.is_file(), "the .bin buffer sidecar must exist");

        // 3 position + 4 rotation + 3 scale + 1 opacity + 3 SH floats each.
        let expected_bytes = count * (3 + 4 + 3 + 1 + 3) * 4;
        match std::fs::metadata(&sidecar) {
            Ok(meta) => assert_eq!(
                meta.len() as usize,
                expected_bytes,
                "the sidecar must hold every attribute block"
            ),
            Err(e) => panic!("could not stat {}: {e}", sidecar.display()),
        }

        let text = read_text(&dst);
        assert_eq!(
            text.matches('{').count(),
            text.matches('}').count(),
            "the emitted glTF JSON must have balanced braces:\n{text}"
        );
        assert_eq!(
            text.matches('[').count(),
            text.matches(']').count(),
            "the emitted glTF JSON must have balanced brackets:\n{text}"
        );
        assert!(text.contains(r#""version": "2.0""#), "got:\n{text}");
        assert!(
            text.contains(r#""uri": "scene.bin""#),
            "the document must reference its sidecar by name; got:\n{text}"
        );
        assert!(
            text.contains(&format!(r#""byteLength": {expected_bytes}"#)),
            "the declared buffer length must match the sidecar; got:\n{text}"
        );
        assert!(
            text.contains(r#""min": ["#) && text.contains(r#""max": ["#),
            "the POSITION accessor must carry min/max; got:\n{text}"
        );
        assert!(text.contains("OXIGAF_gaussian_splat"), "got:\n{text}");
        assert_eq!(
            text.matches(r#""bufferView""#).count(),
            5,
            "each accessor owns its own buffer view; got:\n{text}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn export_gltf_of_an_empty_model_omits_the_buffer() {
        // glTF forbids zero-length buffers and empty `nodes` arrays, so an
        // empty model must produce an asset-only document and no sidecar.
        let dir = temp_subdir("export_gltf_empty");
        let src = dir.join("src.ply");
        let dst = dir.join("empty.gltf");
        let sidecar = dir.join("empty.bin");
        write_source_ply(&sample_model(0), &src);

        let result = export(&src, &dst, ExportFormat::Gltf);
        assert!(result.is_ok(), "export failed: {result:?}");
        assert!(dst.is_file(), "the glTF document must exist");
        assert!(
            !sidecar.exists(),
            "an empty model must not produce a zero-length buffer sidecar"
        );

        let text = read_text(&dst);
        assert_eq!(
            text.matches('{').count(),
            text.matches('}').count(),
            "the emitted glTF JSON must have balanced braces:\n{text}"
        );
        assert!(
            !text.contains(r#""nodes""#) && !text.contains(r#""buffers""#),
            "empty `nodes` / zero-length buffers are invalid glTF; got:\n{text}"
        );
        assert!(
            text.contains(r#""gaussianCount": 0"#),
            "the document must still record the model size; got:\n{text}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn export_gltf_refuses_to_clobber_its_own_buffer() {
        let dir = temp_subdir("export_gltf_bin_collision");
        let src = dir.join("src.ply");
        write_source_ply(&sample_model(2), &src);

        let result = export(&src, dir.join("scene.bin"), ExportFormat::Gltf);
        assert!(
            matches!(result, Err(OxigafError::InvalidConfig(_))),
            "a .bin destination would overwrite its own sidecar; got {result:?}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn export_rejects_an_unreadable_input_and_writes_nothing() {
        let dir = temp_subdir("export_unknown_extension");
        let src = dir.join("model.dat");
        let dst = dir.join("out.ply");
        write_bytes(&src, b"raw bytes");

        let result = export(&src, &dst, ExportFormat::Ply);
        assert!(
            matches!(result, Err(OxigafError::InvalidConfig(_))),
            "got {result:?}"
        );
        assert!(
            !dst.exists(),
            "no output may be produced for an unreadable input"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ---- helpers ----

    // `json_escape` moved to `oxigaf_render::gltf` along with the glTF writer
    // that was its only caller; its unit test moved with it. The remaining
    // OBJ writer needs no escaping (it emits no quoted strings).

    #[test]
    fn sh_coeffs_per_gaussian_matches_the_ply_layout() {
        assert_eq!(sh_coeffs_per_gaussian(0), 3);
        assert_eq!(sh_coeffs_per_gaussian(1), 12);
        assert_eq!(sh_coeffs_per_gaussian(2), 27);
        assert_eq!(sh_coeffs_per_gaussian(3), 48);
    }

    #[test]
    fn dc_color_clamps_and_defaults_to_mid_grey() {
        assert_eq!(dc_color(&[0.0, 0.0, 0.0], 0), [0.5, 0.5, 0.5]);
        assert_eq!(dc_color(&[], 12), [0.5, 0.5, 0.5]);
        assert_eq!(dc_color(&[1e6, -1e6, 0.0], 0), [1.0, 0.0, 0.5]);
    }

    #[test]
    fn bounding_box_ignores_non_finite_positions() {
        let mut model = sample_model(2);
        model.gaussians[0].position = [f32::NAN, 1.0, 2.0];
        model.gaussians[1].position = [3.0, 4.0, 5.0];
        match bounding_box(&model) {
            Some((min, max)) => {
                assert_eq!(min, [3.0, 4.0, 5.0]);
                assert_eq!(max, [3.0, 4.0, 5.0]);
            }
            None => panic!("one finite position is enough for a bounding box"),
        }
        assert!(
            bounding_box(&sample_model(0)).is_none(),
            "an empty model has no bounding box"
        );
    }

    #[test]
    fn frame_camera_keeps_the_eye_inside_the_default_frustum() {
        // A model far larger than the baked 100-unit far plane must still be
        // framed at a representable distance rather than rendering black.
        let mut model = sample_model(2);
        model.gaussians[0].position = [-500.0, -500.0, -500.0];
        model.gaussians[1].position = [500.0, 500.0, 500.0];

        let camera = frame_camera(&model, 64, 64);
        let distance = camera.position[2];
        assert!(
            distance > 0.0 && distance <= 40.0,
            "eye distance {distance} must stay inside the 0.01..100 frustum"
        );
        assert!(camera.focal[0] > 0.0 && camera.focal[1] > 0.0);
    }

    // ---- platform detection ----

    #[test]
    fn detect_best_backend_returns_a_known_backend() {
        let backend = detect_best_backend();
        assert!(
            ["Metal", "Vulkan", "Dx12", "Gl"].contains(&backend.as_str()),
            "unexpected backend {backend}"
        );
    }

    // ---- generic path parameters ----

    #[test]
    fn path_arguments_may_have_independent_types() {
        // Regression: a single `P` for both paths forced callers to coerce
        // one argument just to satisfy the type checker.
        let dir = temp_subdir("mixed_path_types");
        let owned = dir.join("absent.ply");

        let rendered = render_from_file(owned.clone(), Path::new("out.png"), 8, 8);
        assert!(matches!(rendered, Err(OxigafError::PathNotFound(_))));

        let exported = export(owned.as_path(), String::from("out.ply"), ExportFormat::Ply);
        assert!(matches!(exported, Err(OxigafError::PathNotFound(_))));

        let trained = quick_train(owned, "some/output/dir");
        assert!(matches!(trained, Err(OxigafError::PathNotFound(_))));

        let _ = std::fs::remove_dir_all(&dir);
    }
}
