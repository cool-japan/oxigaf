//! TOML configuration file loading and validation.
//!
//! Loads the `oxigaf.toml` project configuration and converts it to the
//! internal [`TrainingConfig`] and [`RasterConfig`] types used by the trainer
//! and rasterizer subsystems.

use std::env;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use oxigaf::render::RasterConfig;
// `GradientClipConfig` / `LrScheduleConfig` live only under `trainer::config`;
// the crate root re-exports the other four config types but not these two.
use oxigaf::trainer::config::{GradientClipConfig, LrScheduleConfig};
use oxigaf::trainer::{
    DensityConfig, InitConfig, LossConfig, OptimizerConfig, TensorBoardConfig, TrainingConfig,
    TrainingPrecision,
};

// ---------------------------------------------------------------------------
// Top-level configuration
// ---------------------------------------------------------------------------

/// Top-level project configuration loaded from `oxigaf.toml`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
#[derive(Default)]
pub struct ProjectConfig {
    /// Model file paths.
    pub model: ModelSection,
    /// GPU / backend settings.
    pub device: DeviceSection,
    /// Training hyper-parameters.
    pub training: TrainingSection,
    /// Output settings (checkpointing, logging, export).
    pub output: OutputSection,
}

// ---------------------------------------------------------------------------
// [model]
// ---------------------------------------------------------------------------

/// `[model]` section — paths to pretrained model files.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelSection {
    /// Path to the converted FLAME model directory.
    pub flame_model_path: PathBuf,
    /// Path to the directory containing diffusion model weights.
    pub diffusion_weights_dir: PathBuf,
}

impl Default for ModelSection {
    fn default() -> Self {
        Self {
            flame_model_path: PathBuf::from("~/.cache/oxigaf/flame2023"),
            diffusion_weights_dir: PathBuf::from("~/.cache/oxigaf/weights"),
        }
    }
}

// ---------------------------------------------------------------------------
// [device]
// ---------------------------------------------------------------------------

/// `[device]` section — GPU backend configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DeviceSection {
    /// GPU backend: `vulkan`, `metal`, `dx12`, or `gl`.
    ///
    /// Empty (the default) or `"auto"` means "let wgpu choose", which is the
    /// only portable answer: a concrete default here would be wrong on every
    /// platform that does not implement it.
    pub backend: String,
    /// GPU device index.
    pub gpu_index: usize,
}

impl Default for DeviceSection {
    /// `backend` defaults to the **empty string**, meaning "auto-detect".
    ///
    /// It used to default to the concrete literal `"vulkan"`, which made
    /// "the user asked for Vulkan" and "nobody ever touched `[device]`"
    /// indistinguishable — so [`DeviceSection::to_wgpu_backends`] could not
    /// be wired into GPU selection at all without forcing a Vulkan-only
    /// instance on macOS (Metal-only) and most Windows installs. An empty
    /// default carries that distinction in the value itself, with no
    /// `Option` (which would break the config's public field type), and lets
    /// `pipeline::request_gpu_device` apply the mapping unconditionally.
    fn default() -> Self {
        Self {
            backend: String::new(),
            gpu_index: 0,
        }
    }
}

/// Backend names [`DeviceSection::resolve_backends`] accepts, for error text.
const KNOWN_BACKEND_NAMES: &[&str] = &["vulkan", "metal", "dx12", "d3d12", "gl", "opengl"];

impl DeviceSection {
    /// Resolve [`DeviceSection::backend`] into an explicit backend selection.
    ///
    /// * `Ok(None)` — nothing was configured (empty/whitespace, or the
    ///   explicit `"auto"`): the caller should let wgpu pick, i.e. use
    ///   [`wgpu::Backends::all`].
    /// * `Ok(Some(backends))` — the user named a backend; honour it.
    /// * `Err(..)` — the user named something that is not a backend. A typo
    ///   such as `backend = "vulcan"` must not silently degrade into
    ///   auto-detection, because the whole point of writing the key was to
    ///   pin the backend.
    ///
    /// # Errors
    ///
    /// Returns an error naming the accepted values when `backend` is neither
    /// empty/`"auto"` nor a recognised backend name.
    pub fn resolve_backends(&self) -> Result<Option<wgpu::Backends>> {
        let name = self.backend.trim().to_ascii_lowercase();
        match name.as_str() {
            "" | "auto" => Ok(None),
            "vulkan" => Ok(Some(wgpu::Backends::VULKAN)),
            "metal" => Ok(Some(wgpu::Backends::METAL)),
            "dx12" | "d3d12" => Ok(Some(wgpu::Backends::DX12)),
            "gl" | "opengl" => Ok(Some(wgpu::Backends::GL)),
            other => anyhow::bail!(
                "Unknown [device] backend {other:?}: expected one of {} \
                 (or an empty value / \"auto\" to let wgpu choose)",
                KNOWN_BACKEND_NAMES.join(", "),
            ),
        }
    }

    /// Lenient form of [`DeviceSection::resolve_backends`]: maps "nothing
    /// configured" *and* "unrecognised name" alike to
    /// [`wgpu::Backends::all`].
    ///
    /// Prefer `resolve_backends` on any path that can surface an error to the
    /// user; this exists for callers that must produce a bitflag
    /// unconditionally (diagnostics, display).
    pub fn to_wgpu_backends(&self) -> wgpu::Backends {
        self.resolve_backends()
            .ok()
            .flatten()
            .unwrap_or_else(wgpu::Backends::all)
    }
}

// ---------------------------------------------------------------------------
// [training]
// ---------------------------------------------------------------------------

/// `[training]` section — training hyper-parameters with sub-sections.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TrainingSection {
    pub total_iterations: u32,
    pub views_per_step: usize,
    pub image_size: u32,
    pub guidance_scale_start: f32,
    pub guidance_scale_end: f32,
    pub guidance_anneal_steps: u32,
    pub num_inference_steps: usize,
    pub opacity_reset_interval: u32,

    /// Number of `train_step` calls whose gradients are averaged into one
    /// optimizer update (micro-batching).
    ///
    /// `1` (the default) steps every iteration. Larger values trade update
    /// frequency for a lower-variance gradient at the same VRAM cost, which is
    /// how a bigger effective batch is reached on a small GPU.
    pub gradient_accumulation_steps: u32,

    /// Decay of the EMA shadow copy of the model, or `None` (the default) to
    /// keep no shadow weights.
    ///
    /// Must be in `(0, 1)`; `0.999` is typical. When set, the trainer keeps an
    /// exponential moving average alongside the live model and checkpoints the
    /// **averaged** weights, which usually evaluate better.
    pub ema_decay: Option<f32>,

    /// Master RNG seed for the whole reconstruction run.
    ///
    /// `None` means "use the built-in default seed"
    /// (`pipeline::DEFAULT_SEED`), which keeps runs reproducible by default.
    /// Gaussian initialisation and the trainer's RNG — view sampling and
    /// densification — are derived from this value, each on its own stream.
    ///
    /// The diffusion denoiser's noise is the documented exception: it is
    /// seeded from the iteration counter inside `oxigaf-trainer`, so it is
    /// reproducible but not influenced by this setting. See
    /// `pipeline::DEFAULT_SEED` for the full account.
    ///
    /// This is the field `oxigaf train --seed` feeds; see the followup note
    /// on `cmd_train` in `main.rs`.
    pub seed: Option<u64>,

    /// Run a periodic evaluation pass every *N* iterations.
    ///
    /// `None` disables it. This is the field `oxigaf train --eval-interval`
    /// feeds. See `pipeline::run_evaluation` for exactly what is measured
    /// (and what it is measured *against*).
    pub eval_interval: Option<u32>,

    /// Stop training as soon as the total loss reaches this threshold.
    ///
    /// `None` disables it, leaving only patience-based early stopping
    /// (`--patience`/`--min-delta`). This is the field
    /// `oxigaf train --early-stop-loss` feeds.
    pub early_stop_loss: Option<f32>,

    /// `[training.init]`
    pub init: InitSection,
    /// `[training.optimizer]`
    pub optimizer: OptimizerSection,
    /// `[training.density_control]`
    pub density_control: DensityControlSection,
    /// `[training.loss]`
    pub loss: LossSection,
}

impl Default for TrainingSection {
    fn default() -> Self {
        // The two knobs that forward straight to `TrainingConfig` take their
        // defaults from it rather than repeating a literal, so the TOML schema
        // cannot silently drift from the trainer's own default behaviour.
        let d = TrainingConfig::default();
        Self {
            total_iterations: 15_000,
            views_per_step: 4,
            image_size: 512,
            guidance_scale_start: 7.5,
            guidance_scale_end: 3.0,
            guidance_anneal_steps: 10_000,
            num_inference_steps: 50,
            opacity_reset_interval: 3_000,
            gradient_accumulation_steps: d.gradient_accumulation_steps,
            ema_decay: d.ema_decay,
            seed: None,
            eval_interval: None,
            early_stop_loss: None,
            init: InitSection::default(),
            optimizer: OptimizerSection::default(),
            density_control: DensityControlSection::default(),
            loss: LossSection::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// [training.init]
// ---------------------------------------------------------------------------

/// `[training.init]` — Gaussian initialisation parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct InitSection {
    pub num_rigid_gaussians: usize,
    pub num_flexible_gaussians: usize,
    pub initial_scale: f32,
    pub initial_opacity: f32,
    pub sh_degree: u32,
}

impl Default for InitSection {
    fn default() -> Self {
        let d = InitConfig::default();
        Self {
            num_rigid_gaussians: d.num_rigid,
            num_flexible_gaussians: d.num_flexible,
            initial_scale: d.initial_scale,
            initial_opacity: d.initial_opacity,
            sh_degree: d.sh_degree,
        }
    }
}

// ---------------------------------------------------------------------------
// [training.optimizer]
// ---------------------------------------------------------------------------

/// `[training.optimizer]` — per-parameter-group learning rates.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct OptimizerSection {
    pub position_lr: f32,
    pub position_lr_final: f32,
    pub rotation_lr: f32,
    pub scale_lr: f32,
    pub opacity_lr: f32,
    pub sh_lr: f32,
    pub offset_lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub position_lr_decay_steps: u32,
}

impl Default for OptimizerSection {
    fn default() -> Self {
        let d = OptimizerConfig::default();
        Self {
            position_lr: d.lr_position,
            position_lr_final: d.lr_position_final,
            rotation_lr: d.lr_rotation,
            scale_lr: d.lr_scale,
            opacity_lr: d.lr_opacity,
            sh_lr: d.lr_sh,
            offset_lr: d.lr_offset,
            beta1: d.beta1,
            beta2: d.beta2,
            epsilon: d.epsilon,
            position_lr_decay_steps: d.position_lr_decay_steps,
        }
    }
}

// ---------------------------------------------------------------------------
// [training.density_control]
// ---------------------------------------------------------------------------

/// `[training.density_control]` — adaptive density control parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DensityControlSection {
    pub interval: u32,
    pub start_iteration: u32,
    pub end_iteration: u32,
    pub grad_threshold: f32,
    pub min_opacity: f32,
    pub max_screen_size: f32,
    pub split_scale_threshold: f32,
    pub max_gaussians: usize,
}

impl Default for DensityControlSection {
    fn default() -> Self {
        let d = DensityConfig::default();
        Self {
            interval: 500,
            start_iteration: 1_000,
            end_iteration: 12_000,
            grad_threshold: d.grad_threshold,
            min_opacity: d.min_opacity,
            max_screen_size: d.max_screen_size,
            split_scale_threshold: d.split_scale_threshold,
            max_gaussians: d.max_gaussians,
        }
    }
}

// ---------------------------------------------------------------------------
// [training.loss]
// ---------------------------------------------------------------------------

/// `[training.loss]` — loss function weights.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LossSection {
    pub lambda_l1: f32,
    pub lambda_ssim: f32,
    pub lambda_ms_ssim: f32,
    pub lambda_lpips: f32,
    pub lambda_position_reg: f32,
    pub lambda_scale_reg: f32,
    pub lambda_opacity_reg: f32,
    pub lambda_normal: f32,
    pub lambda_gradient_penalty: f32,
    pub gradient_penalty_threshold: f32,
    /// World-space Gaussian size (post-`exp()`) above which `lambda_scale_reg`
    /// starts to charge.
    ///
    /// A threshold rather than a weight, hence the bare name — same convention
    /// as `gradient_penalty_threshold`. Defaults to the trainer's
    /// `oxigaf_trainer::loss::MAX_REASONABLE_WORLD_SCALE`; raising it tolerates
    /// larger Gaussians, lowering it fights growth sooner.
    pub scale_reg_max_scale: f32,
}

impl Default for LossSection {
    fn default() -> Self {
        let d = LossConfig::default();
        Self {
            lambda_l1: d.w_l1,
            lambda_ssim: d.w_ssim,
            lambda_ms_ssim: d.w_ms_ssim,
            lambda_lpips: d.w_lpips,
            lambda_position_reg: d.w_position_reg,
            lambda_scale_reg: d.w_scale_reg,
            lambda_opacity_reg: d.w_opacity_reg,
            lambda_normal: d.w_normal,
            lambda_gradient_penalty: d.w_gradient_penalty,
            gradient_penalty_threshold: d.gradient_penalty_threshold,
            scale_reg_max_scale: d.w_scale_reg_max_scale,
        }
    }
}

// ---------------------------------------------------------------------------
// [output]
// ---------------------------------------------------------------------------

/// `[output]` section — checkpoint, logging, and export settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct OutputSection {
    pub checkpoint_interval: u32,
    pub log_interval: u32,
    pub export_format: String,
}

impl Default for OutputSection {
    fn default() -> Self {
        Self {
            checkpoint_interval: 1_000,
            log_interval: 50,
            export_format: "ply".to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// Conversion helpers
// ---------------------------------------------------------------------------

impl ProjectConfig {
    /// Convert the user-facing project configuration to the internal
    /// [`TrainingConfig`] consumed by the trainer.
    pub fn to_training_config(&self) -> TrainingConfig {
        let t = &self.training;
        TrainingConfig {
            total_iterations: t.total_iterations,
            views_per_step: t.views_per_step,
            density_control_interval: t.density_control.interval,
            density_control_start: t.density_control.start_iteration,
            density_control_end: t.density_control.end_iteration,
            opacity_reset_interval: t.opacity_reset_interval,
            checkpoint_interval: self.output.checkpoint_interval,
            log_interval: self.output.log_interval,
            guidance_scale_start: t.guidance_scale_start,
            guidance_scale_end: t.guidance_scale_end,
            guidance_anneal_steps: t.guidance_anneal_steps,
            optimizer: OptimizerConfig {
                lr_position: t.optimizer.position_lr,
                lr_position_final: t.optimizer.position_lr_final,
                lr_rotation: t.optimizer.rotation_lr,
                lr_scale: t.optimizer.scale_lr,
                lr_opacity: t.optimizer.opacity_lr,
                lr_sh: t.optimizer.sh_lr,
                lr_offset: t.optimizer.offset_lr,
                beta1: t.optimizer.beta1,
                beta2: t.optimizer.beta2,
                epsilon: t.optimizer.epsilon,
                position_lr_decay_steps: t.optimizer.position_lr_decay_steps,
            },
            loss: LossConfig {
                w_l1: t.loss.lambda_l1,
                w_ssim: t.loss.lambda_ssim,
                w_ms_ssim: t.loss.lambda_ms_ssim,
                w_lpips: t.loss.lambda_lpips,
                w_position_reg: t.loss.lambda_position_reg,
                w_scale_reg: t.loss.lambda_scale_reg,
                w_opacity_reg: t.loss.lambda_opacity_reg,
                w_normal: t.loss.lambda_normal,
                w_gradient_penalty: t.loss.lambda_gradient_penalty,
                gradient_penalty_threshold: t.loss.gradient_penalty_threshold,
                w_scale_reg_max_scale: t.loss.scale_reg_max_scale,
            },
            density: DensityConfig {
                grad_threshold: t.density_control.grad_threshold,
                min_opacity: t.density_control.min_opacity,
                max_screen_size: t.density_control.max_screen_size,
                split_scale_threshold: t.density_control.split_scale_threshold,
                max_gaussians: t.density_control.max_gaussians,
            },
            init: InitConfig {
                num_rigid: t.init.num_rigid_gaussians,
                num_flexible: t.init.num_flexible_gaussians,
                initial_scale: t.init.initial_scale,
                initial_opacity: t.init.initial_opacity,
                sh_degree: t.init.sh_degree,
            },
            tensorboard: TensorBoardConfig::default(),
            precision: TrainingPrecision::Float32,
            enable_profiling: false,
            gradient_accumulation_steps: t.gradient_accumulation_steps,
            ema_decay: t.ema_decay,
            // `lr_schedule` and `gradient_clip` are sum types whose variants
            // each carry their own parameters, and no `[training.*]` key in
            // this schema is enum-valued (`precision` above is hardcoded for
            // the same reason). They therefore take the trainer's own default
            // — `Fixed` / `Disabled`, i.e. exactly the behaviour every run had
            // before those knobs existed. Spelled `::default()` rather than as
            // the literal variants so the CLI follows the trainer if it ever
            // changes its mind about the neutral setting. Giving them a
            // decoupled TOML surface is a schema addition, filed as a followup.
            lr_schedule: LrScheduleConfig::default(),
            gradient_clip: GradientClipConfig::default(),
        }
    }

    /// Build a [`RasterConfig`] from the project configuration.
    pub fn to_raster_config(&self) -> RasterConfig {
        RasterConfig {
            image_width: self.training.image_size,
            image_height: self.training.image_size,
            sh_degree: self.training.init.sh_degree,
            ..RasterConfig::default()
        }
    }

    /// Basic validation of configuration values.
    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            self.training.total_iterations > 0,
            "total_iterations must be > 0"
        );
        anyhow::ensure!(
            self.training.views_per_step > 0,
            "views_per_step must be > 0"
        );
        anyhow::ensure!(self.training.image_size > 0, "image_size must be > 0");
        anyhow::ensure!(
            self.training.guidance_scale_start > 0.0,
            "guidance_scale_start must be > 0"
        );
        anyhow::ensure!(
            self.training.init.num_rigid_gaussians + self.training.init.num_flexible_gaussians > 0,
            "Total Gaussian count must be > 0"
        );
        anyhow::ensure!(self.training.init.sh_degree <= 3, "SH degree must be <= 3");
        // Reaches `DiffusionTargetConfig`, whose own validator rejects 0 —
        // catch it here so the message names the TOML key.
        anyhow::ensure!(
            self.training.num_inference_steps > 0,
            "num_inference_steps must be > 0"
        );
        if let Some(interval) = self.training.eval_interval {
            anyhow::ensure!(
                interval > 0,
                "eval_interval must be > 0 (omit it to disable evaluation)"
            );
        }
        if let Some(threshold) = self.training.early_stop_loss {
            anyhow::ensure!(
                threshold.is_finite(),
                "early_stop_loss must be a finite number, got {threshold}"
            );
        }
        // `Trainer::new` reads `gradient_accumulation_steps <= 1` as "no
        // accumulation", so a `0` written here would silently disable a feature
        // the user was trying to configure. `TrainingConfig::validate` rejects
        // it, but nothing on the CLI path calls that — so reject it here, where
        // the message can name the TOML key.
        anyhow::ensure!(
            self.training.gradient_accumulation_steps > 0,
            "gradient_accumulation_steps must be > 0 (use 1 to step every iteration)"
        );
        // The trainer rejects an out-of-range decay too, but only once the
        // FLAME model, the frame sequence and the GPU are already up; catch it
        // while the failure is still cheap.
        if let Some(decay) = self.training.ema_decay {
            anyhow::ensure!(
                decay > 0.0 && decay < 1.0,
                "ema_decay must be in (0, 1), got {decay} (omit it to keep no shadow weights)"
            );
        }
        // Surface a bad `[device] backend` here rather than at GPU-init time,
        // after the FLAME model and the whole frame sequence have been loaded.
        self.device.resolve_backends()?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Hierarchical Configuration Loading
// ---------------------------------------------------------------------------

/// Load config with hierarchical priority:
/// 1. CLI arguments (highest)
/// 2. Environment variables
/// 3. Project config file (./oxigaf.toml)
/// 4. User config file (~/.config/oxigaf/config.toml)
/// 5. Default values (lowest)
///
/// Layers 1-3 (below) are merged as raw TOML *tables* before anything is
/// deserialised into a [`ProjectConfig`]. This matters: once a layer has
/// been deserialised, "a field was explicitly written to a value that
/// happens to equal the struct default" and "a field was never mentioned"
/// become indistinguishable, because `#[serde(default)]` has already filled
/// in a concrete value either way. A struct-level "does this differ from
/// the default?" merge (as used for layer 5 below, and as this function
/// used to do for every layer) therefore silently keeps the lower-priority
/// layer's value whenever the higher-priority layer's value happens to
/// equal the default -- inverting the documented priority order for that
/// case. Merging TOML tables directly avoids this: a key that no layer
/// mentions simply stays absent from the merged table, and defaults are
/// applied exactly once, in the final deserialisation.
pub fn load_hierarchical_config(
    cli_config_path: Option<&Path>,
    override_values: Option<&ProjectConfig>,
) -> Result<ProjectConfig> {
    let mut merged_toml = toml::Value::Table(toml::map::Map::new());

    // Layer 1: User config (~/.config/oxigaf/config.toml)
    if let Some(user_config_path) = get_user_config_path() {
        if user_config_path.exists() {
            tracing::debug!("Loading user config from: {}", user_config_path.display());
            let user_value = load_toml_value(&user_config_path)?;
            merged_toml = merge_toml_values(merged_toml, user_value);
        }
    }

    // Layer 2: Project config (./oxigaf.toml)
    let project_config_path = PathBuf::from("./oxigaf.toml");
    if project_config_path.exists() {
        tracing::debug!(
            "Loading project config from: {}",
            project_config_path.display()
        );
        let project_value = load_toml_value(&project_config_path)?;
        merged_toml = merge_toml_values(merged_toml, project_value);
    }

    // Layer 3: CLI-specified config file.
    //
    // `TrainArgs::config` (cli.rs) has `#[arg(default_value = "oxigaf.toml")]`
    // rather than being an `Option`, so `cli_config_path` is *always*
    // `Some(..)` in practice, even when the user never passed `--config`.
    // Treat a missing path as "no project config file" (fall back to
    // whatever earlier layers / defaults already produced) only when it is
    // that implicit default name; a missing path the user explicitly named
    // is still a hard error.
    if let Some(path) = cli_config_path {
        if path.exists() {
            tracing::debug!("Loading CLI-specified config from: {}", path.display());
            let cli_value = load_toml_value(path)?;
            merged_toml = merge_toml_values(merged_toml, cli_value);
        } else if is_default_config_name(path) {
            tracing::debug!(
                "CLI-specified config {} not found; using values from earlier layers/defaults",
                path.display()
            );
        } else {
            anyhow::bail!("Config file not found: {}", path.display());
        }
    }

    let mut config: ProjectConfig = merged_toml
        .try_into()
        .context("Failed to materialise merged configuration")?;

    // Layer 4: Environment variables
    config = apply_env_overrides(config)?;

    // Layer 5: CLI arguments (always take priority, regardless of value)
    // Note: For actual CLI usage, it's recommended to apply CLI overrides
    // directly after calling this function with override_values=None.
    //
    // Unlike layers 1-3 above, this layer necessarily still uses the
    // struct-level "differs from default" merge (`merge_configs`): the
    // caller hands us an already-materialised `ProjectConfig`, which -- like
    // any deserialised layer -- has no way left to represent "this field
    // was intentionally left unset". That is a known, narrower limitation
    // of this specific parameter (its own doc note below), not the general
    // bug the TOML-table merge above fixes for file-based layers; it is
    // primarily used by tests; production CLI overrides are applied by the
    // caller field-by-field after this function returns, which has no such
    // ambiguity.
    if let Some(overrides) = override_values {
        // For override_values, we do a simple overlay: any field that's been
        // explicitly set in the override takes priority. Since we can't detect
        // which fields were explicitly set vs. defaulted, this parameter is
        // primarily for testing. In production, apply CLI overrides after calling
        // this function.
        config = merge_configs(config, overrides.clone());
    }

    config.validate()?;
    Ok(config)
}

/// Whether a missing `cli_config_path` in [`load_hierarchical_config`] should
/// be treated as "no CLI-specified config file" (fall back to whatever
/// earlier layers / defaults already produced) rather than a hard error.
///
/// True only for the implicit default file name (`TrainArgs::config`'s
/// `default_value = "oxigaf.toml"`, regardless of which directory it is
/// joined with) -- a path the user explicitly named that turns out to be
/// missing is always an error.
fn is_default_config_name(path: &Path) -> bool {
    path.ends_with("oxigaf.toml")
}

/// Get user config path (~/.config/oxigaf/config.toml)
fn get_user_config_path() -> Option<PathBuf> {
    let mut path = dirs::config_dir()?;
    path.push("oxigaf");
    path.push("config.toml");
    Some(path)
}

/// Load a config file as a raw [`toml::Value`], without deserialising into
/// [`ProjectConfig`]. Used so hierarchical file layers can be deep-merged as
/// TOML tables (see [`merge_toml_values`]) before a single final
/// deserialisation into a concrete config.
fn load_toml_value(path: &Path) -> Result<toml::Value> {
    let contents = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read config file: {}", path.display()))?;
    toml::from_str(&contents)
        .with_context(|| format!("Failed to parse config file: {}", path.display()))
}

/// Deep-merge two TOML values: when both sides are tables, recursively merge
/// key by key with `overlay` winning on any key present in both; otherwise
/// `overlay` replaces `base` outright (this also covers a key present as a
/// table on one side and a scalar/array on the other -- the overlay's shape
/// wins). This lets a higher-priority config layer override a single leaf
/// key (e.g. `training.total_iterations`) without needing to repeat every
/// other key from lower-priority layers.
fn merge_toml_values(base: toml::Value, overlay: toml::Value) -> toml::Value {
    match (base, overlay) {
        (toml::Value::Table(mut base_table), toml::Value::Table(overlay_table)) => {
            for (key, overlay_value) in overlay_table {
                let merged = match base_table.remove(&key) {
                    Some(base_value) => merge_toml_values(base_value, overlay_value),
                    None => overlay_value,
                };
                base_table.insert(key, merged);
            }
            toml::Value::Table(base_table)
        }
        (_, overlay_value) => overlay_value,
    }
}

/// Merge two configs (second takes priority)
fn merge_configs(base: ProjectConfig, override_cfg: ProjectConfig) -> ProjectConfig {
    ProjectConfig {
        model: merge_model_section(base.model, override_cfg.model),
        device: merge_device_section(base.device, override_cfg.device),
        training: merge_training_section(base.training, override_cfg.training),
        output: merge_output_section(base.output, override_cfg.output),
    }
}

/// Merge model sections (override takes priority for non-default values)
fn merge_model_section(base: ModelSection, override_cfg: ModelSection) -> ModelSection {
    let default = ModelSection::default();
    ModelSection {
        flame_model_path: if override_cfg.flame_model_path != default.flame_model_path {
            override_cfg.flame_model_path
        } else {
            base.flame_model_path
        },
        diffusion_weights_dir: if override_cfg.diffusion_weights_dir
            != default.diffusion_weights_dir
        {
            override_cfg.diffusion_weights_dir
        } else {
            base.diffusion_weights_dir
        },
    }
}

/// Merge device sections (override takes priority for non-default values)
fn merge_device_section(base: DeviceSection, override_cfg: DeviceSection) -> DeviceSection {
    let default = DeviceSection::default();
    DeviceSection {
        backend: if override_cfg.backend != default.backend {
            override_cfg.backend
        } else {
            base.backend
        },
        gpu_index: if override_cfg.gpu_index != default.gpu_index {
            override_cfg.gpu_index
        } else {
            base.gpu_index
        },
    }
}

/// Merge training sections (override takes priority for non-default values)
fn merge_training_section(base: TrainingSection, override_cfg: TrainingSection) -> TrainingSection {
    let default = TrainingSection::default();
    TrainingSection {
        total_iterations: if override_cfg.total_iterations != default.total_iterations {
            override_cfg.total_iterations
        } else {
            base.total_iterations
        },
        views_per_step: if override_cfg.views_per_step != default.views_per_step {
            override_cfg.views_per_step
        } else {
            base.views_per_step
        },
        image_size: if override_cfg.image_size != default.image_size {
            override_cfg.image_size
        } else {
            base.image_size
        },
        guidance_scale_start: if (override_cfg.guidance_scale_start - default.guidance_scale_start)
            .abs()
            > f32::EPSILON
        {
            override_cfg.guidance_scale_start
        } else {
            base.guidance_scale_start
        },
        guidance_scale_end: if (override_cfg.guidance_scale_end - default.guidance_scale_end).abs()
            > f32::EPSILON
        {
            override_cfg.guidance_scale_end
        } else {
            base.guidance_scale_end
        },
        guidance_anneal_steps: if override_cfg.guidance_anneal_steps
            != default.guidance_anneal_steps
        {
            override_cfg.guidance_anneal_steps
        } else {
            base.guidance_anneal_steps
        },
        num_inference_steps: if override_cfg.num_inference_steps != default.num_inference_steps {
            override_cfg.num_inference_steps
        } else {
            base.num_inference_steps
        },
        opacity_reset_interval: if override_cfg.opacity_reset_interval
            != default.opacity_reset_interval
        {
            override_cfg.opacity_reset_interval
        } else {
            base.opacity_reset_interval
        },
        gradient_accumulation_steps: if override_cfg.gradient_accumulation_steps
            != default.gradient_accumulation_steps
        {
            override_cfg.gradient_accumulation_steps
        } else {
            base.gradient_accumulation_steps
        },
        // The four `Option` knobs default to `None`, so "differs from the
        // default" is exactly "the override actually set it" — no
        // sentinel-value ambiguity to work around here.
        ema_decay: override_cfg.ema_decay.or(base.ema_decay),
        seed: override_cfg.seed.or(base.seed),
        eval_interval: override_cfg.eval_interval.or(base.eval_interval),
        early_stop_loss: override_cfg.early_stop_loss.or(base.early_stop_loss),
        init: merge_init_section(base.init, override_cfg.init),
        optimizer: merge_optimizer_section(base.optimizer, override_cfg.optimizer),
        density_control: merge_density_control_section(
            base.density_control,
            override_cfg.density_control,
        ),
        loss: merge_loss_section(base.loss, override_cfg.loss),
    }
}

/// Merge init sections (override takes priority for non-default values)
fn merge_init_section(base: InitSection, override_cfg: InitSection) -> InitSection {
    let default = InitSection::default();
    InitSection {
        num_rigid_gaussians: if override_cfg.num_rigid_gaussians != default.num_rigid_gaussians {
            override_cfg.num_rigid_gaussians
        } else {
            base.num_rigid_gaussians
        },
        num_flexible_gaussians: if override_cfg.num_flexible_gaussians
            != default.num_flexible_gaussians
        {
            override_cfg.num_flexible_gaussians
        } else {
            base.num_flexible_gaussians
        },
        initial_scale: if (override_cfg.initial_scale - default.initial_scale).abs() > f32::EPSILON
        {
            override_cfg.initial_scale
        } else {
            base.initial_scale
        },
        initial_opacity: if (override_cfg.initial_opacity - default.initial_opacity).abs()
            > f32::EPSILON
        {
            override_cfg.initial_opacity
        } else {
            base.initial_opacity
        },
        sh_degree: if override_cfg.sh_degree != default.sh_degree {
            override_cfg.sh_degree
        } else {
            base.sh_degree
        },
    }
}

/// Merge optimizer sections (override takes priority for non-default values)
fn merge_optimizer_section(
    base: OptimizerSection,
    override_cfg: OptimizerSection,
) -> OptimizerSection {
    let default = OptimizerSection::default();
    OptimizerSection {
        position_lr: if (override_cfg.position_lr - default.position_lr).abs() > f32::EPSILON {
            override_cfg.position_lr
        } else {
            base.position_lr
        },
        position_lr_final: if (override_cfg.position_lr_final - default.position_lr_final).abs()
            > f32::EPSILON
        {
            override_cfg.position_lr_final
        } else {
            base.position_lr_final
        },
        rotation_lr: if (override_cfg.rotation_lr - default.rotation_lr).abs() > f32::EPSILON {
            override_cfg.rotation_lr
        } else {
            base.rotation_lr
        },
        scale_lr: if (override_cfg.scale_lr - default.scale_lr).abs() > f32::EPSILON {
            override_cfg.scale_lr
        } else {
            base.scale_lr
        },
        opacity_lr: if (override_cfg.opacity_lr - default.opacity_lr).abs() > f32::EPSILON {
            override_cfg.opacity_lr
        } else {
            base.opacity_lr
        },
        sh_lr: if (override_cfg.sh_lr - default.sh_lr).abs() > f32::EPSILON {
            override_cfg.sh_lr
        } else {
            base.sh_lr
        },
        offset_lr: if (override_cfg.offset_lr - default.offset_lr).abs() > f32::EPSILON {
            override_cfg.offset_lr
        } else {
            base.offset_lr
        },
        beta1: if (override_cfg.beta1 - default.beta1).abs() > f32::EPSILON {
            override_cfg.beta1
        } else {
            base.beta1
        },
        beta2: if (override_cfg.beta2 - default.beta2).abs() > f32::EPSILON {
            override_cfg.beta2
        } else {
            base.beta2
        },
        epsilon: if (override_cfg.epsilon - default.epsilon).abs() > f32::EPSILON {
            override_cfg.epsilon
        } else {
            base.epsilon
        },
        position_lr_decay_steps: if override_cfg.position_lr_decay_steps
            != default.position_lr_decay_steps
        {
            override_cfg.position_lr_decay_steps
        } else {
            base.position_lr_decay_steps
        },
    }
}

/// Merge density control sections (override takes priority for non-default values)
fn merge_density_control_section(
    base: DensityControlSection,
    override_cfg: DensityControlSection,
) -> DensityControlSection {
    let default = DensityControlSection::default();
    DensityControlSection {
        interval: if override_cfg.interval != default.interval {
            override_cfg.interval
        } else {
            base.interval
        },
        start_iteration: if override_cfg.start_iteration != default.start_iteration {
            override_cfg.start_iteration
        } else {
            base.start_iteration
        },
        end_iteration: if override_cfg.end_iteration != default.end_iteration {
            override_cfg.end_iteration
        } else {
            base.end_iteration
        },
        grad_threshold: if (override_cfg.grad_threshold - default.grad_threshold).abs()
            > f32::EPSILON
        {
            override_cfg.grad_threshold
        } else {
            base.grad_threshold
        },
        min_opacity: if (override_cfg.min_opacity - default.min_opacity).abs() > f32::EPSILON {
            override_cfg.min_opacity
        } else {
            base.min_opacity
        },
        max_screen_size: if (override_cfg.max_screen_size - default.max_screen_size).abs()
            > f32::EPSILON
        {
            override_cfg.max_screen_size
        } else {
            base.max_screen_size
        },
        split_scale_threshold: if (override_cfg.split_scale_threshold
            - default.split_scale_threshold)
            .abs()
            > f32::EPSILON
        {
            override_cfg.split_scale_threshold
        } else {
            base.split_scale_threshold
        },
        max_gaussians: if override_cfg.max_gaussians != default.max_gaussians {
            override_cfg.max_gaussians
        } else {
            base.max_gaussians
        },
    }
}

/// Merge loss sections (override takes priority for non-default values)
fn merge_loss_section(base: LossSection, override_cfg: LossSection) -> LossSection {
    let default = LossSection::default();
    LossSection {
        lambda_l1: if (override_cfg.lambda_l1 - default.lambda_l1).abs() > f32::EPSILON {
            override_cfg.lambda_l1
        } else {
            base.lambda_l1
        },
        lambda_ssim: if (override_cfg.lambda_ssim - default.lambda_ssim).abs() > f32::EPSILON {
            override_cfg.lambda_ssim
        } else {
            base.lambda_ssim
        },
        lambda_ms_ssim: if (override_cfg.lambda_ms_ssim - default.lambda_ms_ssim).abs()
            > f32::EPSILON
        {
            override_cfg.lambda_ms_ssim
        } else {
            base.lambda_ms_ssim
        },
        lambda_lpips: if (override_cfg.lambda_lpips - default.lambda_lpips).abs() > f32::EPSILON {
            override_cfg.lambda_lpips
        } else {
            base.lambda_lpips
        },
        lambda_position_reg: if (override_cfg.lambda_position_reg - default.lambda_position_reg)
            .abs()
            > f32::EPSILON
        {
            override_cfg.lambda_position_reg
        } else {
            base.lambda_position_reg
        },
        lambda_scale_reg: if (override_cfg.lambda_scale_reg - default.lambda_scale_reg).abs()
            > f32::EPSILON
        {
            override_cfg.lambda_scale_reg
        } else {
            base.lambda_scale_reg
        },
        lambda_opacity_reg: if (override_cfg.lambda_opacity_reg - default.lambda_opacity_reg).abs()
            > f32::EPSILON
        {
            override_cfg.lambda_opacity_reg
        } else {
            base.lambda_opacity_reg
        },
        lambda_normal: if (override_cfg.lambda_normal - default.lambda_normal).abs() > f32::EPSILON
        {
            override_cfg.lambda_normal
        } else {
            base.lambda_normal
        },
        lambda_gradient_penalty: if (override_cfg.lambda_gradient_penalty
            - default.lambda_gradient_penalty)
            .abs()
            > f32::EPSILON
        {
            override_cfg.lambda_gradient_penalty
        } else {
            base.lambda_gradient_penalty
        },
        gradient_penalty_threshold: if (override_cfg.gradient_penalty_threshold
            - default.gradient_penalty_threshold)
            .abs()
            > f32::EPSILON
        {
            override_cfg.gradient_penalty_threshold
        } else {
            base.gradient_penalty_threshold
        },
        scale_reg_max_scale: if (override_cfg.scale_reg_max_scale - default.scale_reg_max_scale)
            .abs()
            > f32::EPSILON
        {
            override_cfg.scale_reg_max_scale
        } else {
            base.scale_reg_max_scale
        },
    }
}

/// Merge output sections (override takes priority for non-default values)
fn merge_output_section(base: OutputSection, override_cfg: OutputSection) -> OutputSection {
    let default = OutputSection::default();
    OutputSection {
        checkpoint_interval: if override_cfg.checkpoint_interval != default.checkpoint_interval {
            override_cfg.checkpoint_interval
        } else {
            base.checkpoint_interval
        },
        log_interval: if override_cfg.log_interval != default.log_interval {
            override_cfg.log_interval
        } else {
            base.log_interval
        },
        export_format: if override_cfg.export_format != default.export_format {
            override_cfg.export_format
        } else {
            base.export_format
        },
    }
}

/// Apply environment variable overrides
fn apply_env_overrides(mut config: ProjectConfig) -> Result<ProjectConfig> {
    // Training parameters
    if let Ok(val) = env::var("OXIGAF_TOTAL_ITERATIONS") {
        config.training.total_iterations = val
            .parse()
            .with_context(|| format!("Invalid OXIGAF_TOTAL_ITERATIONS: {}", val))?;
    }

    if let Ok(val) = env::var("OXIGAF_IMAGE_SIZE") {
        config.training.image_size = val
            .parse()
            .with_context(|| format!("Invalid OXIGAF_IMAGE_SIZE: {}", val))?;
    }

    if let Ok(val) = env::var("OXIGAF_VIEWS_PER_STEP") {
        config.training.views_per_step = val
            .parse()
            .with_context(|| format!("Invalid OXIGAF_VIEWS_PER_STEP: {}", val))?;
    }

    if let Ok(val) = env::var("OXIGAF_GUIDANCE_SCALE_START") {
        config.training.guidance_scale_start = val
            .parse()
            .with_context(|| format!("Invalid OXIGAF_GUIDANCE_SCALE_START: {}", val))?;
    }

    if let Ok(val) = env::var("OXIGAF_GUIDANCE_SCALE_END") {
        config.training.guidance_scale_end = val
            .parse()
            .with_context(|| format!("Invalid OXIGAF_GUIDANCE_SCALE_END: {}", val))?;
    }

    // Optimizer parameters
    if let Ok(val) = env::var("OXIGAF_POSITION_LR") {
        config.training.optimizer.position_lr = val
            .parse()
            .with_context(|| format!("Invalid OXIGAF_POSITION_LR: {}", val))?;
    }

    if let Ok(val) = env::var("OXIGAF_SCALING_LR") {
        config.training.optimizer.scale_lr = val
            .parse()
            .with_context(|| format!("Invalid OXIGAF_SCALING_LR: {}", val))?;
    }

    if let Ok(val) = env::var("OXIGAF_ROTATION_LR") {
        config.training.optimizer.rotation_lr = val
            .parse()
            .with_context(|| format!("Invalid OXIGAF_ROTATION_LR: {}", val))?;
    }

    if let Ok(val) = env::var("OXIGAF_OPACITY_LR") {
        config.training.optimizer.opacity_lr = val
            .parse()
            .with_context(|| format!("Invalid OXIGAF_OPACITY_LR: {}", val))?;
    }

    if let Ok(val) = env::var("OXIGAF_SH_LR") {
        config.training.optimizer.sh_lr = val
            .parse()
            .with_context(|| format!("Invalid OXIGAF_SH_LR: {}", val))?;
    }

    if let Ok(val) = env::var("OXIGAF_SEED") {
        config.training.seed = Some(
            val.parse()
                .with_context(|| format!("Invalid OXIGAF_SEED: {}", val))?,
        );
    }

    if let Ok(val) = env::var("OXIGAF_EVAL_INTERVAL") {
        config.training.eval_interval = Some(
            val.parse()
                .with_context(|| format!("Invalid OXIGAF_EVAL_INTERVAL: {}", val))?,
        );
    }

    if let Ok(val) = env::var("OXIGAF_EARLY_STOP_LOSS") {
        config.training.early_stop_loss = Some(
            val.parse()
                .with_context(|| format!("Invalid OXIGAF_EARLY_STOP_LOSS: {}", val))?,
        );
    }

    // Device parameters
    if let Ok(val) = env::var("OXIGAF_DEVICE_BACKEND") {
        config.device.backend = val;
        // Reject a typo here, where the variable's name is still available
        // for the message, instead of silently degrading to auto-detect.
        config
            .device
            .resolve_backends()
            .context("Invalid OXIGAF_DEVICE_BACKEND")?;
    }

    if let Ok(val) = env::var("OXIGAF_DEVICE_GPU_INDEX") {
        config.device.gpu_index = val
            .parse()
            .with_context(|| format!("Invalid OXIGAF_DEVICE_GPU_INDEX: {}", val))?;
    }

    // Output parameters
    if let Ok(val) = env::var("OXIGAF_OUTPUT_CHECKPOINT_INTERVAL") {
        config.output.checkpoint_interval = val
            .parse()
            .with_context(|| format!("Invalid OXIGAF_OUTPUT_CHECKPOINT_INTERVAL: {}", val))?;
    }

    if let Ok(val) = env::var("OXIGAF_OUTPUT_LOG_INTERVAL") {
        config.output.log_interval = val
            .parse()
            .with_context(|| format!("Invalid OXIGAF_OUTPUT_LOG_INTERVAL: {}", val))?;
    }

    if let Ok(val) = env::var("OXIGAF_OUTPUT_EXPORT_FORMAT") {
        config.output.export_format = val;
    }

    // Init parameters
    if let Ok(val) = env::var("OXIGAF_SH_DEGREE") {
        config.training.init.sh_degree = val
            .parse()
            .with_context(|| format!("Invalid OXIGAF_SH_DEGREE: {}", val))?;
    }

    if let Ok(val) = env::var("OXIGAF_NUM_RIGID_GAUSSIANS") {
        config.training.init.num_rigid_gaussians = val
            .parse()
            .with_context(|| format!("Invalid OXIGAF_NUM_RIGID_GAUSSIANS: {}", val))?;
    }

    if let Ok(val) = env::var("OXIGAF_NUM_FLEXIBLE_GAUSSIANS") {
        config.training.init.num_flexible_gaussians = val
            .parse()
            .with_context(|| format!("Invalid OXIGAF_NUM_FLEXIBLE_GAUSSIANS: {}", val))?;
    }

    Ok(config)
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------
//
// The former standalone `load_config` helper (single-file load with a
// "missing default oxigaf.toml -> defaults" fallback) has been superseded:
// its logic is now inlined directly into `load_hierarchical_config`'s Layer
// 3 handling above, which is the only place that needs it. It had no
// callers anywhere in this crate or its test suites.

/// Generate a default TOML configuration string that can be written to a file.
pub fn generate_default_config() -> Result<String> {
    let config = ProjectConfig::default();
    toml::to_string_pretty(&config).context("Failed to serialize default config")
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

/// Expand a leading `~` in a path to the user's home directory.
///
/// Uses [`dirs::home_dir`] rather than reading `$HOME` directly so that this
/// also works on Windows (`USERPROFILE`), where `$HOME` is not normally set.
pub fn expand_tilde(path: &Path) -> PathBuf {
    let s = path.to_string_lossy();
    if s.starts_with("~/") || s == "~" {
        if let Some(home) = dirs::home_dir() {
            let home_str = home.to_string_lossy();
            return PathBuf::from(s.replacen('~', &home_str, 1));
        }
    }
    path.to_path_buf()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_round_trips() -> Result<()> {
        let config = ProjectConfig::default();
        let toml_str =
            toml::to_string_pretty(&config).context("Failed to serialize default config")?;
        let parsed: ProjectConfig =
            toml::from_str(&toml_str).context("Failed to parse serialized config")?;
        assert_eq!(parsed.training.total_iterations, 15_000);
        assert_eq!(parsed.training.init.num_rigid_gaussians, 50_000);
        Ok(())
    }

    #[test]
    fn partial_config_uses_defaults() -> Result<()> {
        let toml_str = r#"
[training]
total_iterations = 5000
"#;
        let config: ProjectConfig =
            toml::from_str(toml_str).context("Failed to parse partial config")?;
        assert_eq!(config.training.total_iterations, 5000);
        // Unspecified fields use defaults.
        assert_eq!(config.training.views_per_step, 4);
        assert_eq!(config.training.init.num_rigid_gaussians, 50_000);
        Ok(())
    }

    #[test]
    fn validation_catches_zero_iterations() {
        let mut config = ProjectConfig::default();
        config.training.total_iterations = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn expand_tilde_works() {
        // Assert against `dirs::home_dir()` directly (rather than mutating
        // the process-wide `HOME` env var, which is both a data race with
        // other tests running in parallel in this module and not portable
        // to Windows, where `expand_tilde`'s underlying `dirs::home_dir()`
        // consults `USERPROFILE` instead). This exercises exactly what the
        // implementation promises on whatever platform the test runs on.
        let Some(home) = dirs::home_dir() else {
            // No resolvable home directory in this environment (e.g. a
            // minimal/sandboxed CI container) -- nothing to assert.
            return;
        };
        let p = expand_tilde(Path::new("~/.cache/oxigaf"));
        assert_eq!(p, home.join(".cache/oxigaf"));
    }

    #[test]
    fn expand_tilde_leaves_non_tilde_paths_untouched() {
        assert_eq!(
            expand_tilde(Path::new("/abs/path")),
            PathBuf::from("/abs/path")
        );
        assert_eq!(
            expand_tilde(Path::new("relative/x")),
            PathBuf::from("relative/x")
        );
    }

    #[test]
    fn to_wgpu_backends_maps_known_names() {
        let mut device = DeviceSection::default();
        device.backend = "metal".to_string();
        assert_eq!(device.to_wgpu_backends(), wgpu::Backends::METAL);
        device.backend = "VULKAN".to_string();
        assert_eq!(device.to_wgpu_backends(), wgpu::Backends::VULKAN);
        device.backend = "dx12".to_string();
        assert_eq!(device.to_wgpu_backends(), wgpu::Backends::DX12);
        device.backend = "gl".to_string();
        assert_eq!(device.to_wgpu_backends(), wgpu::Backends::GL);
    }

    #[test]
    fn to_wgpu_backends_unknown_falls_back_to_all() {
        let mut device = DeviceSection::default();
        device.backend = "nonsense".to_string();
        assert_eq!(device.to_wgpu_backends(), wgpu::Backends::all());
        device.backend = String::new();
        assert_eq!(device.to_wgpu_backends(), wgpu::Backends::all());
    }

    // -----------------------------------------------------------------------
    // DeviceSection: "auto" default + strict resolution
    //
    // Regression coverage for: `backend` used to default to the literal
    // "vulkan", so `to_wgpu_backends()` could not be wired into GPU selection
    // at all -- doing so would have requested a Vulkan-only instance for
    // every user who never touched `[device]`, which finds no adapters on
    // macOS. The default must therefore stay "unset", and an unrecognised
    // name must be an error rather than silently becoming auto-detect.
    // -----------------------------------------------------------------------

    #[test]
    fn device_backend_defaults_to_auto_not_a_concrete_backend() {
        let device = DeviceSection::default();
        assert!(
            device.backend.is_empty(),
            "a concrete default backend ({:?}) makes 'user chose it' and \
             'never configured' indistinguishable and breaks macOS",
            device.backend
        );
        assert_eq!(
            device.resolve_backends().expect("empty backend is valid"),
            None,
            "the default must resolve to 'let wgpu choose'"
        );
        assert_eq!(device.to_wgpu_backends(), wgpu::Backends::all());
    }

    #[test]
    fn resolve_backends_accepts_auto_and_known_names() {
        let mut device = DeviceSection::default();
        for spelling in ["", "   ", "auto", "AUTO"] {
            device.backend = spelling.to_string();
            assert_eq!(
                device.resolve_backends().expect("auto spelling is valid"),
                None,
                "{spelling:?} must mean auto-detect"
            );
        }
        device.backend = " Metal ".to_string();
        assert_eq!(
            device.resolve_backends().expect("metal is valid"),
            Some(wgpu::Backends::METAL),
            "resolution must be case- and whitespace-insensitive"
        );
        device.backend = "d3d12".to_string();
        assert_eq!(
            device.resolve_backends().expect("d3d12 is valid"),
            Some(wgpu::Backends::DX12)
        );
    }

    #[test]
    fn resolve_backends_rejects_typos() {
        let mut device = DeviceSection::default();
        device.backend = "vulcan".to_string();
        let err = device
            .resolve_backends()
            .expect_err("a typo must not silently become auto-detect");
        let msg = err.to_string();
        assert!(msg.contains("vulcan"), "must quote the bad value: {msg}");
        assert!(msg.contains("vulkan"), "must list valid values: {msg}");
    }

    #[test]
    fn validate_rejects_unknown_backend() {
        let mut config = ProjectConfig::default();
        config.device.backend = "openkl".to_string();
        assert!(
            config.validate().is_err(),
            "a bad backend must fail validation, not GPU init after the \
             FLAME model and every frame have already been loaded"
        );
    }

    // -----------------------------------------------------------------------
    // seed / eval_interval / early_stop_loss
    //
    // Regression coverage for: `--seed`, `--eval-interval` and
    // `--early-stop-loss` were accepted by the CLI and then ignored, because
    // no configuration field carried them as far as the pipeline.
    // -----------------------------------------------------------------------

    #[test]
    fn training_knobs_default_to_unset() {
        let t = TrainingSection::default();
        assert_eq!(t.seed, None);
        assert_eq!(t.eval_interval, None);
        assert_eq!(t.early_stop_loss, None);
    }

    #[test]
    fn training_knobs_round_trip_through_toml() -> Result<()> {
        let toml_str = r#"
[training]
seed = 7
eval_interval = 250
early_stop_loss = 0.015
"#;
        let config: ProjectConfig =
            toml::from_str(toml_str).context("Failed to parse training knobs")?;
        assert_eq!(config.training.seed, Some(7));
        assert_eq!(config.training.eval_interval, Some(250));
        assert_eq!(config.training.early_stop_loss, Some(0.015));

        // And they survive a serialise/deserialise cycle.
        let rendered = toml::to_string_pretty(&config).context("serialize")?;
        let parsed: ProjectConfig = toml::from_str(&rendered).context("re-parse")?;
        assert_eq!(parsed.training.seed, Some(7));
        assert_eq!(parsed.training.eval_interval, Some(250));
        Ok(())
    }

    #[test]
    fn optimisation_knobs_reach_the_training_config() -> Result<()> {
        // `gradient_accumulation_steps` / `ema_decay` / `scale_reg_max_scale`
        // are only worth exposing if they actually arrive at the trainer, so
        // assert the whole TOML → `TrainingConfig` path rather than the field.
        let toml_str = r#"
[training]
gradient_accumulation_steps = 4
ema_decay = 0.999

[training.loss]
scale_reg_max_scale = 0.02
"#;
        let config: ProjectConfig =
            toml::from_str(toml_str).context("Failed to parse optimisation knobs")?;
        let training = config.to_training_config();
        assert_eq!(training.gradient_accumulation_steps, 4);
        assert_eq!(training.ema_decay, Some(0.999));
        assert_eq!(training.loss.w_scale_reg_max_scale, 0.02);

        // The two enum-valued knobs are not part of this schema; they must
        // land on the trainer's own neutral defaults, i.e. the behaviour every
        // run had before those knobs existed.
        assert_eq!(training.lr_schedule, LrScheduleConfig::Fixed);
        assert_eq!(training.gradient_clip, GradientClipConfig::Disabled);

        // Unset leaves the trainer's defaults in place.
        let bare = ProjectConfig::default().to_training_config();
        assert_eq!(bare.gradient_accumulation_steps, 1);
        assert_eq!(bare.ema_decay, None);
        assert_eq!(
            bare.loss.w_scale_reg_max_scale,
            LossConfig::default().w_scale_reg_max_scale
        );
        Ok(())
    }

    #[test]
    fn validate_rejects_out_of_range_optimisation_knobs() {
        // `Trainer::new` reads `<= 1` as "no accumulation", so a `0` would be
        // silently ignored rather than reported.
        let mut config = ProjectConfig::default();
        config.training.gradient_accumulation_steps = 0;
        let err = config
            .validate()
            .expect_err("zero accumulation steps must be rejected");
        assert!(err.to_string().contains("gradient_accumulation_steps"));

        let mut config = ProjectConfig::default();
        config.training.ema_decay = Some(1.0);
        assert!(config.validate().is_err(), "decay of 1.0 never converges");
        config.training.ema_decay = Some(0.0);
        assert!(config.validate().is_err(), "decay of 0.0 keeps no history");
        config.training.ema_decay = Some(0.999);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn validate_rejects_zero_num_inference_steps() {
        // The value now reaches `DiffusionTargetConfig`, which rejects 0 with
        // a message that does not mention `oxigaf.toml` at all.
        let mut config = ProjectConfig::default();
        config.training.num_inference_steps = 0;
        let err = config
            .validate()
            .expect_err("zero inference steps must be rejected");
        assert!(err.to_string().contains("num_inference_steps"));
    }

    #[test]
    fn validate_rejects_zero_eval_interval() {
        let mut config = ProjectConfig::default();
        config.training.eval_interval = Some(0);
        assert!(config.validate().is_err());
        config.training.eval_interval = Some(1);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn validate_rejects_non_finite_early_stop_loss() {
        let mut config = ProjectConfig::default();
        config.training.early_stop_loss = Some(f32::NAN);
        assert!(config.validate().is_err());
        config.training.early_stop_loss = Some(f32::INFINITY);
        assert!(config.validate().is_err());
        config.training.early_stop_loss = Some(0.01);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn merge_training_section_keeps_set_optional_knobs() {
        let mut base = TrainingSection::default();
        base.seed = Some(11);
        base.eval_interval = Some(100);
        let mut overlay = TrainingSection::default();
        overlay.seed = Some(22);
        // `overlay` leaves eval_interval unset, so base's must survive.
        let merged = merge_training_section(base, overlay);
        assert_eq!(merged.seed, Some(22), "the override must win when set");
        assert_eq!(
            merged.eval_interval,
            Some(100),
            "an unset override must not erase the lower-priority layer"
        );
        assert_eq!(merged.early_stop_loss, None);
    }

    // -----------------------------------------------------------------------
    // is_default_config_name / load_hierarchical_config Layer 3 decision
    //
    // Regression coverage for: `oxigaf train` used to hard-fail whenever
    // `./oxigaf.toml` did not exist, because `TrainArgs::config` always
    // supplies `Some("oxigaf.toml")` (its clap default) and the old code
    // called `load_config_from_file` unconditionally for that path. These
    // are pure/hermetic: no filesystem access, no reliance on cwd or
    // `$HOME`, unlike `load_hierarchical_config` itself (see
    // `tests/config_hierarchy_tests.rs` for that end-to-end coverage).
    // -----------------------------------------------------------------------

    #[test]
    fn is_default_config_name_bare() {
        assert!(is_default_config_name(Path::new("oxigaf.toml")));
    }

    #[test]
    fn is_default_config_name_with_directory() {
        assert!(is_default_config_name(Path::new("some/dir/oxigaf.toml")));
        assert!(is_default_config_name(Path::new("./oxigaf.toml")));
    }

    #[test]
    fn is_default_config_name_rejects_custom_names() {
        assert!(!is_default_config_name(Path::new("my-config.toml")));
        assert!(!is_default_config_name(Path::new("nonexistent.toml")));
        assert!(!is_default_config_name(Path::new(
            "configs/oxigaf-prod.toml"
        )));
    }

    // -----------------------------------------------------------------------
    // merge_toml_values
    //
    // Regression coverage for: the old struct-level "does the override
    // differ from `Default::default()`?" merge silently kept a
    // lower-priority layer's value whenever a higher-priority layer
    // explicitly wrote a value that happened to equal the struct default
    // (e.g. re-stating `total_iterations = 15000`), inverting the
    // documented layer priority for that case.
    // -----------------------------------------------------------------------

    #[test]
    fn merge_toml_values_overlay_scalar_wins() {
        let base: toml::Value = toml::from_str("value = 1").unwrap();
        let overlay: toml::Value = toml::from_str("value = 2").unwrap();
        let merged = merge_toml_values(base, overlay);
        assert_eq!(merged["value"].as_integer(), Some(2));
    }

    #[test]
    fn merge_toml_values_keeps_base_keys_not_in_overlay() {
        let base: toml::Value = toml::from_str("a = 1\nb = 2").unwrap();
        let overlay: toml::Value = toml::from_str("b = 20").unwrap();
        let merged = merge_toml_values(base, overlay);
        assert_eq!(merged["a"].as_integer(), Some(1));
        assert_eq!(merged["b"].as_integer(), Some(20));
    }

    #[test]
    fn merge_toml_values_recurses_into_nested_tables() {
        let base: toml::Value = toml::from_str(
            r#"
            [training]
            total_iterations = 15000
            image_size = 512
            "#,
        )
        .unwrap();
        let overlay: toml::Value = toml::from_str(
            r#"
            [training]
            total_iterations = 5000
            "#,
        )
        .unwrap();
        let merged = merge_toml_values(base, overlay);
        // Overridden leaf wins...
        assert_eq!(
            merged["training"]["total_iterations"].as_integer(),
            Some(5000)
        );
        // ...and a leaf the overlay's table never mentioned survives from base.
        assert_eq!(merged["training"]["image_size"].as_integer(), Some(512));
    }

    #[test]
    fn merge_toml_values_explicit_default_value_beats_lower_layer() {
        // This is exactly the scenario from the audit finding: a
        // lower-priority layer (e.g. the user config) sets a non-default
        // value, and a higher-priority layer (e.g. the project config)
        // explicitly re-states the struct default. The higher-priority
        // layer must still win, because it was merged as raw TOML -- there
        // is no "differs from Default::default()" heuristic in the way.
        let user_config: toml::Value =
            toml::from_str("[training]\ntotal_iterations = 5000").unwrap();
        let project_config: toml::Value =
            toml::from_str("[training]\ntotal_iterations = 15000").unwrap(); // == struct default

        let merged = merge_toml_values(toml::Value::Table(toml::map::Map::new()), user_config);
        let merged = merge_toml_values(merged, project_config);

        let config: ProjectConfig = merged.try_into().expect("deserialise merged config");
        assert_eq!(
            config.training.total_iterations, 15_000,
            "higher-priority layer's explicitly-default value must win over \
             the lower-priority layer's non-default value"
        );
    }

    #[test]
    fn merge_toml_values_empty_table_deserialises_to_defaults() {
        let merged = toml::Value::Table(toml::map::Map::new());
        let config: ProjectConfig = merged.try_into().expect("deserialise empty table");
        let default_config = ProjectConfig::default();
        assert_eq!(
            config.training.total_iterations,
            default_config.training.total_iterations
        );
        assert_eq!(config.device.backend, default_config.device.backend);
    }
}
