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
use oxigaf::trainer::{
    DensityConfig, InitConfig, LossConfig, OptimizerConfig, TensorBoardConfig, TrainingConfig,
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
    pub backend: String,
    /// GPU device index.
    pub gpu_index: usize,
}

impl Default for DeviceSection {
    fn default() -> Self {
        Self {
            backend: "vulkan".to_string(),
            gpu_index: 0,
        }
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
        Self {
            total_iterations: 15_000,
            views_per_step: 4,
            image_size: 512,
            guidance_scale_start: 7.5,
            guidance_scale_end: 3.0,
            guidance_anneal_steps: 10_000,
            num_inference_steps: 50,
            opacity_reset_interval: 3_000,
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
pub fn load_hierarchical_config(
    cli_config_path: Option<&Path>,
    override_values: Option<&ProjectConfig>,
) -> Result<ProjectConfig> {
    // Start with defaults
    let mut config = ProjectConfig::default();

    // Layer 1: User config (~/.config/oxigaf/config.toml)
    if let Some(user_config_path) = get_user_config_path() {
        if user_config_path.exists() {
            tracing::debug!("Loading user config from: {}", user_config_path.display());
            let user_config = load_config_from_file(&user_config_path)?;
            config = merge_configs(config, user_config);
        }
    }

    // Layer 2: Project config (./oxigaf.toml)
    let project_config_path = PathBuf::from("./oxigaf.toml");
    if project_config_path.exists() {
        tracing::debug!(
            "Loading project config from: {}",
            project_config_path.display()
        );
        let project_config = load_config_from_file(&project_config_path)?;
        config = merge_configs(config, project_config);
    }

    // Layer 3: CLI-specified config file
    if let Some(path) = cli_config_path {
        tracing::debug!("Loading CLI-specified config from: {}", path.display());
        let cli_file_config = load_config_from_file(path)?;
        config = merge_configs(config, cli_file_config);
    }

    // Layer 4: Environment variables
    config = apply_env_overrides(config)?;

    // Layer 5: CLI arguments (always take priority, regardless of value)
    // Note: For actual CLI usage, it's recommended to apply CLI overrides
    // directly after calling this function with override_values=None
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

/// Get user config path (~/.config/oxigaf/config.toml)
fn get_user_config_path() -> Option<PathBuf> {
    let mut path = dirs::config_dir()?;
    path.push("oxigaf");
    path.push("config.toml");
    Some(path)
}

/// Load config from a file without checking if it's the default oxigaf.toml
fn load_config_from_file(path: &Path) -> Result<ProjectConfig> {
    let contents = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read config file: {}", path.display()))?;
    let config: ProjectConfig = toml::from_str(&contents)
        .with_context(|| format!("Failed to parse config file: {}", path.display()))?;
    Ok(config)
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

    // Device parameters
    if let Ok(val) = env::var("OXIGAF_DEVICE_BACKEND") {
        config.device.backend = val;
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

/// Load and validate a [`ProjectConfig`] from a TOML file.
///
/// If `path` does not exist and is the default `oxigaf.toml`, returns the
/// default configuration instead of erroring.
#[allow(dead_code)]
pub fn load_config(path: &Path) -> Result<ProjectConfig> {
    if path.exists() {
        let contents = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read config file: {}", path.display()))?;
        let config: ProjectConfig = toml::from_str(&contents)
            .with_context(|| format!("Failed to parse config file: {}", path.display()))?;
        config.validate()?;
        Ok(config)
    } else if path.ends_with("oxigaf.toml") {
        tracing::info!(
            "Config file not found at {}, using defaults",
            path.display()
        );
        Ok(ProjectConfig::default())
    } else {
        anyhow::bail!("Config file not found: {}", path.display())
    }
}

/// Generate a default TOML configuration string that can be written to a file.
#[allow(dead_code)]
pub fn generate_default_config() -> Result<String> {
    let config = ProjectConfig::default();
    toml::to_string_pretty(&config).context("Failed to serialize default config")
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

/// Expand a leading `~` in a path to the user's home directory.
pub fn expand_tilde(path: &Path) -> PathBuf {
    let s = path.to_string_lossy();
    if s.starts_with("~/") || s == "~" {
        if let Ok(home) = std::env::var("HOME") {
            return PathBuf::from(s.replacen('~', &home, 1));
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
        // Use a unique env var to avoid conflicts in parallel tests
        std::env::set_var("HOME", "/home/test");
        let p = expand_tilde(Path::new("~/.cache/oxigaf"));
        assert_eq!(p, PathBuf::from("/home/test/.cache/oxigaf"));
    }
}
