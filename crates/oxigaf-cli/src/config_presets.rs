//! Named preset configurations for common OxiGAF training scenarios.
//!
//! Provides a set of [`TrainingPreset`] values that encode sensible defaults
//! for common use-cases, letting users select `"quality"` instead of
//! specifying 40+ individual parameters.

use std::fmt;

use thiserror::Error;

use crate::config::ProjectConfig;

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// Errors that can occur when working with presets.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum PresetError {
    /// The requested preset name is not known.
    #[error(
        "unknown preset '{0}'. Available presets: quick, balanced, quality, research, \
         production, portrait, video (aliases: fast, default, high, debug, prod, headshot, \
         animation)."
    )]
    UnknownPreset(String),

    /// An override string (key=value) could not be parsed or applied.
    #[error(
        "invalid override '{0}'. Expected key=value with one of the supported keys: \
         num_iterations, warmup_iterations, position_lr, color_lr, opacity_lr, scale_lr, \
         rotation_lr, densify_from_iter, densify_until_iter, densify_grad_threshold, \
         opacity_reset_interval, max_gaussians, sh_degree, image_width, image_height, \
         checkpoint_every, keep_last_n_checkpoints, log_every, tensorboard_enabled, verbose, \
         lambda_l1, lambda_ssim, lambda_lpips, lambda_opacity, lambda_scale, batch_size, \
         num_workers."
    )]
    InvalidOverride(String),
}

// ---------------------------------------------------------------------------
// TrainingPresetName
// ---------------------------------------------------------------------------

/// Named variants for built-in training presets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TrainingPresetName {
    /// Fast iteration, low quality.
    Quick,
    /// Default recommendation.
    Balanced,
    /// High quality, slow.
    Quality,
    /// Full logging, detailed metrics.
    Research,
    /// Optimized for deployment.
    Production,
    /// Optimized for portrait/headshot rendering.
    Portrait,
    /// Optimized for video/animation sequences.
    Video,
}

impl TrainingPresetName {
    /// Parse a preset name from a string (case-insensitive, aliases accepted).
    ///
    /// Accepted aliases:
    /// - `"quick"` / `"fast"` → [`Quick`](TrainingPresetName::Quick)
    /// - `"balanced"` / `"default"` → [`Balanced`](TrainingPresetName::Balanced)
    /// - `"quality"` / `"high"` → [`Quality`](TrainingPresetName::Quality)
    /// - `"research"` / `"debug"` → [`Research`](TrainingPresetName::Research)
    /// - `"production"` / `"prod"` → [`Production`](TrainingPresetName::Production)
    /// - `"portrait"` / `"headshot"` → [`Portrait`](TrainingPresetName::Portrait)
    /// - `"video"` / `"animation"` → [`Video`](TrainingPresetName::Video)
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Result<Self, PresetError> {
        match s.to_ascii_lowercase().as_str() {
            "quick" | "fast" => Ok(Self::Quick),
            "balanced" | "default" => Ok(Self::Balanced),
            "quality" | "high" => Ok(Self::Quality),
            "research" | "debug" => Ok(Self::Research),
            "production" | "prod" => Ok(Self::Production),
            "portrait" | "headshot" => Ok(Self::Portrait),
            "video" | "animation" => Ok(Self::Video),
            other => Err(PresetError::UnknownPreset(other.to_string())),
        }
    }

    /// Return the canonical string name for this preset.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Quick => "quick",
            Self::Balanced => "balanced",
            Self::Quality => "quality",
            Self::Research => "research",
            Self::Production => "production",
            Self::Portrait => "portrait",
            Self::Video => "video",
        }
    }

    /// Return all preset name variants in a well-defined order.
    pub fn all() -> &'static [TrainingPresetName] {
        &[
            TrainingPresetName::Quick,
            TrainingPresetName::Balanced,
            TrainingPresetName::Quality,
            TrainingPresetName::Research,
            TrainingPresetName::Production,
            TrainingPresetName::Portrait,
            TrainingPresetName::Video,
        ]
    }
}

impl fmt::Display for TrainingPresetName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

// ---------------------------------------------------------------------------
// PresetDiff
// ---------------------------------------------------------------------------

/// A single parameter difference between two [`TrainingPreset`] values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PresetDiff {
    /// Parameter field name.
    pub parameter: &'static str,
    /// Formatted value from preset A.
    pub value_a: String,
    /// Formatted value from preset B.
    pub value_b: String,
}

// ---------------------------------------------------------------------------
// TrainingPreset
// ---------------------------------------------------------------------------

/// A complete set of training hyper-parameters for a named scenario.
///
/// All fields are plain value types so that a preset can be cloned and
/// mutated to apply user overrides without touching the global defaults.
#[derive(Debug, Clone)]
pub struct TrainingPreset {
    /// Preset identity.
    pub name: TrainingPresetName,
    /// Human-readable description (single sentence).
    pub description: &'static str,

    // --- Training duration ---
    pub num_iterations: u32,
    pub warmup_iterations: u32,

    // --- Learning rates ---
    pub position_lr: f32,
    pub color_lr: f32,
    pub opacity_lr: f32,
    pub scale_lr: f32,
    pub rotation_lr: f32,

    // --- Density control ---
    pub densify_from_iter: u32,
    pub densify_until_iter: u32,
    pub densify_grad_threshold: f32,
    pub opacity_reset_interval: u32,
    pub max_gaussians: u32,

    // --- Rendering ---
    pub sh_degree: u32,
    pub image_width: u32,
    pub image_height: u32,

    // --- Checkpointing ---
    pub checkpoint_every: u32,
    pub keep_last_n_checkpoints: usize,

    // --- Logging ---
    pub log_every: u32,
    pub tensorboard_enabled: bool,
    pub verbose: bool,

    // --- Loss weights ---
    pub lambda_l1: f32,
    pub lambda_ssim: f32,
    pub lambda_lpips: f32,
    /// Opacity regularisation weight.
    pub lambda_opacity: f32,
    /// Scale regularisation weight.
    pub lambda_scale: f32,

    // --- Hardware ---
    pub batch_size: usize,
    pub num_workers: usize,
}

// ---------------------------------------------------------------------------
// Preset constructors
// ---------------------------------------------------------------------------

fn quick_preset() -> TrainingPreset {
    TrainingPreset {
        name: TrainingPresetName::Quick,
        description: "Fast iteration preset for quick prototyping. Lower quality, 50K iterations, 512\u{00d7}512.",
        num_iterations: 50_000,
        warmup_iterations: 500,
        position_lr: 0.00016,
        color_lr: 0.0025,
        opacity_lr: 0.05,
        scale_lr: 0.005,
        rotation_lr: 0.001,
        densify_from_iter: 500,
        densify_until_iter: 15_000,
        densify_grad_threshold: 0.0002,
        opacity_reset_interval: 3_000,
        max_gaussians: 200_000,
        sh_degree: 1,
        image_width: 512,
        image_height: 512,
        checkpoint_every: 5_000,
        keep_last_n_checkpoints: 2,
        log_every: 100,
        tensorboard_enabled: false,
        verbose: false,
        lambda_l1: 0.8,
        lambda_ssim: 0.2,
        lambda_lpips: 0.0,
        lambda_opacity: 0.0,
        lambda_scale: 0.0,
        batch_size: 1,
        num_workers: 2,
    }
}

fn balanced_preset() -> TrainingPreset {
    TrainingPreset {
        name: TrainingPresetName::Balanced,
        description: "Balanced preset \u{2014} recommended default. 100K iterations, 800\u{00d7}800, SH degree 3.",
        num_iterations: 100_000,
        warmup_iterations: 1_000,
        position_lr: 0.00016,
        color_lr: 0.0025,
        opacity_lr: 0.05,
        scale_lr: 0.005,
        rotation_lr: 0.001,
        densify_from_iter: 500,
        densify_until_iter: 20_000,
        densify_grad_threshold: 0.0002,
        opacity_reset_interval: 3_000,
        max_gaussians: 500_000,
        sh_degree: 3,
        image_width: 800,
        image_height: 800,
        checkpoint_every: 5_000,
        keep_last_n_checkpoints: 3,
        log_every: 500,
        tensorboard_enabled: true,
        verbose: false,
        lambda_l1: 0.8,
        lambda_ssim: 0.2,
        lambda_lpips: 0.05,
        lambda_opacity: 0.01,
        lambda_scale: 0.005,
        batch_size: 1,
        num_workers: 4,
    }
}

fn quality_preset() -> TrainingPreset {
    TrainingPreset {
        name: TrainingPresetName::Quality,
        description: "High-quality preset. 300K iterations, 1024\u{00d7}1024, full SH, LPIPS loss.",
        num_iterations: 300_000,
        warmup_iterations: 2_000,
        position_lr: 0.00016,
        color_lr: 0.001,
        opacity_lr: 0.05,
        scale_lr: 0.005,
        rotation_lr: 0.001,
        densify_from_iter: 1_000,
        densify_until_iter: 60_000,
        densify_grad_threshold: 0.0001,
        opacity_reset_interval: 5_000,
        max_gaussians: 1_000_000,
        sh_degree: 3,
        image_width: 1024,
        image_height: 1024,
        checkpoint_every: 10_000,
        keep_last_n_checkpoints: 5,
        log_every: 1_000,
        tensorboard_enabled: true,
        verbose: false,
        lambda_l1: 0.8,
        lambda_ssim: 0.2,
        lambda_lpips: 0.1,
        lambda_opacity: 0.01,
        lambda_scale: 0.005,
        batch_size: 1,
        num_workers: 4,
    }
}

fn research_preset() -> TrainingPreset {
    TrainingPreset {
        name: TrainingPresetName::Research,
        description: "Research preset. Full logging, frequent checkpoints, verbose output.",
        num_iterations: 200_000,
        warmup_iterations: 1_000,
        position_lr: 0.00016,
        color_lr: 0.0025,
        opacity_lr: 0.05,
        scale_lr: 0.005,
        rotation_lr: 0.001,
        densify_from_iter: 500,
        densify_until_iter: 40_000,
        densify_grad_threshold: 0.0002,
        opacity_reset_interval: 3_000,
        max_gaussians: 800_000,
        sh_degree: 3,
        image_width: 800,
        image_height: 800,
        checkpoint_every: 2_000,
        keep_last_n_checkpoints: 10,
        log_every: 100,
        tensorboard_enabled: true,
        verbose: true,
        lambda_l1: 0.8,
        lambda_ssim: 0.2,
        lambda_lpips: 0.1,
        lambda_opacity: 0.01,
        lambda_scale: 0.01,
        batch_size: 1,
        num_workers: 4,
    }
}

fn production_preset() -> TrainingPreset {
    TrainingPreset {
        name: TrainingPresetName::Production,
        description:
            "Production preset. Optimized for deployment: fewer checkpoints, minimal logging.",
        num_iterations: 150_000,
        warmup_iterations: 500,
        position_lr: 0.00016,
        color_lr: 0.001,
        opacity_lr: 0.05,
        scale_lr: 0.005,
        rotation_lr: 0.001,
        densify_from_iter: 500,
        densify_until_iter: 30_000,
        densify_grad_threshold: 0.0002,
        opacity_reset_interval: 3_000,
        max_gaussians: 600_000,
        sh_degree: 3,
        image_width: 800,
        image_height: 800,
        checkpoint_every: 25_000,
        keep_last_n_checkpoints: 2,
        log_every: 5_000,
        tensorboard_enabled: false,
        verbose: false,
        lambda_l1: 0.8,
        lambda_ssim: 0.2,
        lambda_lpips: 0.05,
        lambda_opacity: 0.02,
        lambda_scale: 0.01,
        batch_size: 1,
        num_workers: 4,
    }
}

fn portrait_preset() -> TrainingPreset {
    TrainingPreset {
        name: TrainingPresetName::Portrait,
        description:
            "Portrait/headshot preset. Tuned for close-up face/head rendering, 512\u{00d7}512.",
        num_iterations: 200_000,
        warmup_iterations: 1_000,
        position_lr: 0.00016,
        color_lr: 0.0025,
        opacity_lr: 0.05,
        scale_lr: 0.005,
        rotation_lr: 0.001,
        densify_from_iter: 500,
        densify_until_iter: 40_000,
        densify_grad_threshold: 0.00015,
        opacity_reset_interval: 4_000,
        max_gaussians: 700_000,
        sh_degree: 3,
        image_width: 512,
        image_height: 512,
        checkpoint_every: 5_000,
        keep_last_n_checkpoints: 3,
        log_every: 500,
        tensorboard_enabled: true,
        verbose: false,
        lambda_l1: 0.8,
        lambda_ssim: 0.2,
        lambda_lpips: 0.1,
        lambda_opacity: 0.005,
        lambda_scale: 0.005,
        batch_size: 1,
        num_workers: 4,
    }
}

fn video_preset() -> TrainingPreset {
    TrainingPreset {
        name: TrainingPresetName::Video,
        description: "Video/animation preset. Lower Gaussian count for temporal consistency, frequent checkpoints.",
        num_iterations: 100_000,
        warmup_iterations: 500,
        position_lr: 0.00016,
        color_lr: 0.0025,
        opacity_lr: 0.05,
        scale_lr: 0.003,
        rotation_lr: 0.001,
        densify_from_iter: 500,
        densify_until_iter: 15_000,
        densify_grad_threshold: 0.0002,
        opacity_reset_interval: 2_000,
        max_gaussians: 400_000,
        sh_degree: 2,
        image_width: 512,
        image_height: 512,
        checkpoint_every: 2_000,
        keep_last_n_checkpoints: 5,
        log_every: 200,
        tensorboard_enabled: true,
        verbose: false,
        lambda_l1: 0.8,
        lambda_ssim: 0.2,
        lambda_lpips: 0.05,
        lambda_opacity: 0.02,
        lambda_scale: 0.01,
        batch_size: 1,
        num_workers: 4,
    }
}

// ---------------------------------------------------------------------------
// TrainingPreset impl
// ---------------------------------------------------------------------------

impl TrainingPreset {
    /// Construct the preset for the given name.
    pub fn get(name: &TrainingPresetName) -> TrainingPreset {
        match name {
            TrainingPresetName::Quick => quick_preset(),
            TrainingPresetName::Balanced => balanced_preset(),
            TrainingPresetName::Quality => quality_preset(),
            TrainingPresetName::Research => research_preset(),
            TrainingPresetName::Production => production_preset(),
            TrainingPresetName::Portrait => portrait_preset(),
            TrainingPresetName::Video => video_preset(),
        }
    }

    /// Return all presets.
    pub fn all() -> Vec<TrainingPreset> {
        TrainingPresetName::all()
            .iter()
            .map(TrainingPreset::get)
            .collect()
    }

    /// Return each preset's canonical name paired with its description.
    pub fn list_descriptions() -> Vec<(String, &'static str)> {
        TrainingPreset::all()
            .into_iter()
            .map(|p| (p.name.to_string(), p.description))
            .collect()
    }

    /// Format the preset as a human-readable parameter table.
    pub fn format_table(&self) -> String {
        let mut buf = String::with_capacity(1024);
        buf.push_str(&format!("Preset: {}\n", self.name.as_str()));
        buf.push_str(&format!("  {}\n\n", self.description));
        buf.push_str("  Training\n");
        buf.push_str(&format!(
            "    num_iterations          : {}\n",
            self.num_iterations
        ));
        buf.push_str(&format!(
            "    warmup_iterations       : {}\n",
            self.warmup_iterations
        ));
        buf.push_str("\n  Learning rates\n");
        buf.push_str(&format!(
            "    position_lr             : {}\n",
            self.position_lr
        ));
        buf.push_str(&format!(
            "    color_lr                : {}\n",
            self.color_lr
        ));
        buf.push_str(&format!(
            "    opacity_lr              : {}\n",
            self.opacity_lr
        ));
        buf.push_str(&format!(
            "    scale_lr                : {}\n",
            self.scale_lr
        ));
        buf.push_str(&format!(
            "    rotation_lr             : {}\n",
            self.rotation_lr
        ));
        buf.push_str("\n  Density control\n");
        buf.push_str(&format!(
            "    densify_from_iter       : {}\n",
            self.densify_from_iter
        ));
        buf.push_str(&format!(
            "    densify_until_iter      : {}\n",
            self.densify_until_iter
        ));
        buf.push_str(&format!(
            "    densify_grad_threshold  : {}\n",
            self.densify_grad_threshold
        ));
        buf.push_str(&format!(
            "    opacity_reset_interval  : {}\n",
            self.opacity_reset_interval
        ));
        buf.push_str(&format!(
            "    max_gaussians           : {}\n",
            self.max_gaussians
        ));
        buf.push_str("\n  Rendering\n");
        buf.push_str(&format!(
            "    sh_degree               : {}\n",
            self.sh_degree
        ));
        buf.push_str(&format!(
            "    image_width             : {}\n",
            self.image_width
        ));
        buf.push_str(&format!(
            "    image_height            : {}\n",
            self.image_height
        ));
        buf.push_str("\n  Checkpointing\n");
        buf.push_str(&format!(
            "    checkpoint_every        : {}\n",
            self.checkpoint_every
        ));
        buf.push_str(&format!(
            "    keep_last_n_checkpoints : {}\n",
            self.keep_last_n_checkpoints
        ));
        buf.push_str("\n  Logging\n");
        buf.push_str(&format!(
            "    log_every               : {}\n",
            self.log_every
        ));
        buf.push_str(&format!(
            "    tensorboard_enabled     : {}\n",
            self.tensorboard_enabled
        ));
        buf.push_str(&format!("    verbose                 : {}\n", self.verbose));
        buf.push_str("\n  Loss weights\n");
        buf.push_str(&format!(
            "    lambda_l1               : {}\n",
            self.lambda_l1
        ));
        buf.push_str(&format!(
            "    lambda_ssim             : {}\n",
            self.lambda_ssim
        ));
        buf.push_str(&format!(
            "    lambda_lpips            : {}\n",
            self.lambda_lpips
        ));
        buf.push_str(&format!(
            "    lambda_opacity          : {}\n",
            self.lambda_opacity
        ));
        buf.push_str(&format!(
            "    lambda_scale            : {}\n",
            self.lambda_scale
        ));
        buf.push_str("\n  Hardware\n");
        buf.push_str(&format!(
            "    batch_size              : {}\n",
            self.batch_size
        ));
        buf.push_str(&format!(
            "    num_workers             : {}\n",
            self.num_workers
        ));
        buf.push_str(&format!(
            "\n  Estimates\n    training_time_minutes   : {:.1}\n    vram_mb                 : {}\n    quality_score           : {:.3}\n",
            self.estimated_minutes(),
            self.estimated_vram_mb(),
            self.quality_score(),
        ));
        buf
    }

    /// Compare two presets and return the list of differing parameters.
    pub fn diff(a: &TrainingPreset, b: &TrainingPreset) -> Vec<PresetDiff> {
        let mut diffs: Vec<PresetDiff> = Vec::new();

        macro_rules! cmp_field {
            ($field:ident) => {
                if a.$field != b.$field {
                    diffs.push(PresetDiff {
                        parameter: stringify!($field),
                        value_a: format!("{}", a.$field),
                        value_b: format!("{}", b.$field),
                    });
                }
            };
        }

        // name and description intentionally excluded (they are identity, not parameters)
        cmp_field!(num_iterations);
        cmp_field!(warmup_iterations);
        cmp_field!(position_lr);
        cmp_field!(color_lr);
        cmp_field!(opacity_lr);
        cmp_field!(scale_lr);
        cmp_field!(rotation_lr);
        cmp_field!(densify_from_iter);
        cmp_field!(densify_until_iter);
        cmp_field!(densify_grad_threshold);
        cmp_field!(opacity_reset_interval);
        cmp_field!(max_gaussians);
        cmp_field!(sh_degree);
        cmp_field!(image_width);
        cmp_field!(image_height);
        cmp_field!(checkpoint_every);
        cmp_field!(keep_last_n_checkpoints);
        cmp_field!(log_every);
        cmp_field!(tensorboard_enabled);
        cmp_field!(verbose);
        cmp_field!(lambda_l1);
        cmp_field!(lambda_ssim);
        cmp_field!(lambda_lpips);
        cmp_field!(lambda_opacity);
        cmp_field!(lambda_scale);
        cmp_field!(batch_size);
        cmp_field!(num_workers);

        diffs
    }

    /// Rough estimated training time in minutes.
    ///
    /// Heuristic: `num_iterations / 1000.0 * 2.5`.
    pub fn estimated_minutes(&self) -> f32 {
        self.num_iterations as f32 / 1_000.0 * 2.5
    }

    /// Estimated VRAM usage in MB for training at this preset's settings.
    ///
    /// Delegates to [`crate::memory_estimator::estimate_memory`], which
    /// accounts for Gaussian parameters (position/rotation/scale/opacity/SH,
    /// sized by `sh_degree`), gradients, Adam optimizer state, and render
    /// buffers at `image_width`x`image_height`. The previous heuristic
    /// (`max_gaussians * 64 / 1_000_000`) assumed a flat 64 bytes/Gaussian
    /// regardless of SH degree or resolution, which undercounted a
    /// degree-3 Gaussian's true training footprint (~944 B: parameters +
    /// gradients + two Adam moments) by roughly an order of magnitude.
    pub fn estimated_vram_mb(&self) -> u32 {
        let mem_config = crate::memory_estimator::MemEstimateConfig {
            n_gaussians: self.max_gaussians as usize,
            sh_degree: self.sh_degree.min(3),
            render_width: self.image_width.max(1) as usize,
            render_height: self.image_height.max(1) as usize,
            use_ema: false,
            use_optimizer: true,
            n_render_views: self.batch_size.max(1),
            // This estimate is about the 3DGS scene only, not the (much
            // larger, separately-loaded) multi-view diffusion model.
            diffusion_model_params: 0,
            target_vram_bytes: usize::MAX,
            fp16_optimizer: false,
        };

        match crate::memory_estimator::estimate_memory(&mem_config) {
            Ok(estimate) => (estimate.total_gpu_bytes as u64 / (1024 * 1024)) as u32,
            Err(_) => {
                // Only reachable for a configuration the estimator rejects
                // outright (e.g. sh_degree > 3 from a malformed
                // `apply_override`, or a zero/oversized resolution) --
                // none of the built-in presets hit this. Fall back to a
                // direct per-Gaussian estimate rather than reporting 0.
                let sh_coeffs = crate::memory_estimator::sh_coefficients(self.sh_degree.min(3));
                // positions(3) + rotations(4) + scales(3) + opacity(1) + SH,
                // each an f32 (4 bytes); x4 for parameters + gradients + two
                // Adam moments.
                let bytes_per_gaussian = (3 + 4 + 3 + 1 + sh_coeffs) as u64 * 4 * 4;
                ((self.max_gaussians as u64 * bytes_per_gaussian) / (1024 * 1024)) as u32
            }
        }
    }

    /// Estimated output quality in the range `[0.0, 1.0]`.
    ///
    /// Combines iteration count, SH degree, resolution, and LPIPS weight
    /// into a single normalised score. Satisfies: quality > balanced > quick.
    pub fn quality_score(&self) -> f32 {
        // Normalisation constants chosen so quality ≈ 1.0, quick ≈ 0.2.
        const MAX_ITER: f32 = 300_000.0;
        const MAX_SH: f32 = 3.0;
        const MAX_RES: f32 = 1024.0;
        const MAX_LPIPS: f32 = 0.1;

        let iter_score = (self.num_iterations as f32 / MAX_ITER).min(1.0);
        let sh_score = (self.sh_degree as f32 / MAX_SH).min(1.0);
        let res_score =
            ((self.image_width as f32 + self.image_height as f32) / (2.0 * MAX_RES)).min(1.0);
        let lpips_score = (self.lambda_lpips / MAX_LPIPS).min(1.0);

        // Weighted combination.
        0.40 * iter_score + 0.25 * sh_score + 0.20 * res_score + 0.15 * lpips_score
    }

    /// Apply this preset's hyper-parameters onto an existing
    /// [`ProjectConfig`], overwriting only the fields this preset has an
    /// opinion about.
    ///
    /// Call this on a fresh [`ProjectConfig::default`] (or as a layer
    /// between defaults and the project's `oxigaf.toml`, e.g. from
    /// `TrainArgs::profile`) so that file/env/CLI layers applied afterwards
    /// can still override individual values.
    ///
    /// Not every preset field has a corresponding [`ProjectConfig`] field
    /// yet: `warmup_iterations`, `tensorboard_enabled`, `verbose`,
    /// `keep_last_n_checkpoints`, `batch_size`, and `num_workers` describe
    /// concepts `ProjectConfig` does not currently track (TensorBoard
    /// enablement, CLI verbosity, and dataloader batching are configured
    /// elsewhere, or not at all). Those are intentionally left untouched
    /// rather than force-fit into an unrelated field.
    pub fn apply_to(&self, config: &mut ProjectConfig) {
        let t = &mut config.training;
        t.total_iterations = self.num_iterations;
        // All built-in presets use image_width == image_height; ProjectConfig
        // only models a single square `image_size`, so take the larger side
        // if a future/overridden preset ever has an asymmetric resolution.
        t.image_size = self.image_width.max(self.image_height);
        t.opacity_reset_interval = self.opacity_reset_interval;

        t.init.sh_degree = self.sh_degree;

        t.optimizer.position_lr = self.position_lr;
        t.optimizer.rotation_lr = self.rotation_lr;
        t.optimizer.scale_lr = self.scale_lr;
        t.optimizer.opacity_lr = self.opacity_lr;
        // `color_lr` is the preset's name for the SH/color coefficient
        // learning rate, i.e. ProjectConfig's `sh_lr`.
        t.optimizer.sh_lr = self.color_lr;

        t.density_control.start_iteration = self.densify_from_iter;
        t.density_control.end_iteration = self.densify_until_iter;
        t.density_control.grad_threshold = self.densify_grad_threshold;
        t.density_control.max_gaussians = self.max_gaussians as usize;

        t.loss.lambda_l1 = self.lambda_l1;
        t.loss.lambda_ssim = self.lambda_ssim;
        t.loss.lambda_lpips = self.lambda_lpips;
        t.loss.lambda_opacity_reg = self.lambda_opacity;
        t.loss.lambda_scale_reg = self.lambda_scale;

        config.output.checkpoint_interval = self.checkpoint_every;
        config.output.log_interval = self.log_every;
    }
}

impl From<&TrainingPreset> for ProjectConfig {
    /// Build a [`ProjectConfig`] starting from defaults with this preset
    /// applied via [`TrainingPreset::apply_to`].
    fn from(preset: &TrainingPreset) -> Self {
        let mut config = ProjectConfig::default();
        preset.apply_to(&mut config);
        config
    }
}

// ---------------------------------------------------------------------------
// Override helpers
// ---------------------------------------------------------------------------

/// Parse a single `key=value` override string and return a modified clone of
/// the preset with that field overridden.
///
/// Supported keys correspond to the field names of [`TrainingPreset`].
pub fn apply_override(
    preset: &TrainingPreset,
    key_value: &str,
) -> Result<TrainingPreset, PresetError> {
    let eq_pos = key_value
        .find('=')
        .ok_or_else(|| PresetError::InvalidOverride(key_value.to_string()))?;

    let key = &key_value[..eq_pos];
    let raw = &key_value[eq_pos + 1..];

    if key.is_empty() || raw.is_empty() {
        return Err(PresetError::InvalidOverride(key_value.to_string()));
    }

    let mut p = preset.clone();

    // Helper closures defined inline to keep the match arms concise.
    macro_rules! parse_u32 {
        ($field:ident) => {{
            p.$field = raw
                .parse::<u32>()
                .map_err(|_| PresetError::InvalidOverride(key_value.to_string()))?;
        }};
    }
    macro_rules! parse_f32 {
        ($field:ident) => {{
            p.$field = raw
                .parse::<f32>()
                .map_err(|_| PresetError::InvalidOverride(key_value.to_string()))?;
        }};
    }
    macro_rules! parse_bool {
        ($field:ident) => {{
            p.$field = match raw {
                "true" | "1" | "yes" => true,
                "false" | "0" | "no" => false,
                _ => return Err(PresetError::InvalidOverride(key_value.to_string())),
            };
        }};
    }
    macro_rules! parse_usize {
        ($field:ident) => {{
            p.$field = raw
                .parse::<usize>()
                .map_err(|_| PresetError::InvalidOverride(key_value.to_string()))?;
        }};
    }

    match key {
        "num_iterations" => parse_u32!(num_iterations),
        "warmup_iterations" => parse_u32!(warmup_iterations),
        "sh_degree" => parse_u32!(sh_degree),
        "image_width" => parse_u32!(image_width),
        "image_height" => parse_u32!(image_height),
        "max_gaussians" => parse_u32!(max_gaussians),
        "log_every" => parse_u32!(log_every),
        "densify_from_iter" => parse_u32!(densify_from_iter),
        "densify_until_iter" => parse_u32!(densify_until_iter),
        "opacity_reset_interval" => parse_u32!(opacity_reset_interval),
        "checkpoint_every" => parse_u32!(checkpoint_every),
        "keep_last_n_checkpoints" => parse_usize!(keep_last_n_checkpoints),
        "batch_size" => parse_usize!(batch_size),
        "num_workers" => parse_usize!(num_workers),
        "tensorboard_enabled" => parse_bool!(tensorboard_enabled),
        "verbose" => parse_bool!(verbose),
        "lambda_l1" => parse_f32!(lambda_l1),
        "lambda_ssim" => parse_f32!(lambda_ssim),
        "lambda_lpips" => parse_f32!(lambda_lpips),
        "lambda_opacity" => parse_f32!(lambda_opacity),
        "lambda_scale" => parse_f32!(lambda_scale),
        "densify_grad_threshold" => parse_f32!(densify_grad_threshold),
        "position_lr" => parse_f32!(position_lr),
        "color_lr" => parse_f32!(color_lr),
        "opacity_lr" => parse_f32!(opacity_lr),
        "scale_lr" => parse_f32!(scale_lr),
        "rotation_lr" => parse_f32!(rotation_lr),
        _ => return Err(PresetError::InvalidOverride(key_value.to_string())),
    }

    Ok(p)
}

/// Apply multiple `key=value` override strings in order, returning the
/// resulting [`TrainingPreset`].
pub fn apply_overrides(
    preset: &TrainingPreset,
    overrides: &[&str],
) -> Result<TrainingPreset, PresetError> {
    let mut current = preset.clone();
    for kv in overrides {
        current = apply_override(&current, kv)?;
    }
    Ok(current)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // 1. from_str: "quick" → Quick
    #[test]
    fn test_from_str_quick() {
        assert_eq!(
            TrainingPresetName::from_str("quick").unwrap(),
            TrainingPresetName::Quick
        );
    }

    // 2. from_str: "QUALITY" (case-insensitive) → Quality
    #[test]
    fn test_from_str_quality_uppercase() {
        assert_eq!(
            TrainingPresetName::from_str("QUALITY").unwrap(),
            TrainingPresetName::Quality
        );
    }

    // 3. from_str: "fast" alias → Quick
    #[test]
    fn test_from_str_fast_alias() {
        assert_eq!(
            TrainingPresetName::from_str("fast").unwrap(),
            TrainingPresetName::Quick
        );
    }

    // 4. from_str: "balanced" → Balanced
    #[test]
    fn test_from_str_balanced() {
        assert_eq!(
            TrainingPresetName::from_str("balanced").unwrap(),
            TrainingPresetName::Balanced
        );
    }

    // 5. from_str: "unknown" → Err(UnknownPreset)
    #[test]
    fn test_from_str_unknown() {
        let err = TrainingPresetName::from_str("unknown").unwrap_err();
        assert!(matches!(err, PresetError::UnknownPreset(_)));
    }

    // 6. TrainingPresetName::all() returns all 7 variants
    #[test]
    fn test_all_variants_count() {
        assert_eq!(TrainingPresetName::all().len(), 7);
    }

    // 7. TrainingPreset::get(Quick).num_iterations == 50_000
    #[test]
    fn test_get_quick_iterations() {
        let p = TrainingPreset::get(&TrainingPresetName::Quick);
        assert_eq!(p.num_iterations, 50_000);
    }

    // 8. Quality.num_iterations > Balanced.num_iterations
    #[test]
    fn test_quality_more_iters_than_balanced() {
        let quality = TrainingPreset::get(&TrainingPresetName::Quality);
        let balanced = TrainingPreset::get(&TrainingPresetName::Balanced);
        assert!(quality.num_iterations > balanced.num_iterations);
    }

    // 9. Research.verbose == true
    #[test]
    fn test_research_verbose() {
        let p = TrainingPreset::get(&TrainingPresetName::Research);
        assert!(p.verbose);
    }

    // 10. Production.log_every > Research.log_every
    #[test]
    fn test_production_less_frequent_logging() {
        let prod = TrainingPreset::get(&TrainingPresetName::Production);
        let res = TrainingPreset::get(&TrainingPresetName::Research);
        assert!(prod.log_every > res.log_every);
    }

    // 11. format_table: contains preset name and key field values
    #[test]
    fn test_format_table_contains_name_and_values() {
        let p = TrainingPreset::get(&TrainingPresetName::Balanced);
        let table = p.format_table();
        assert!(table.contains("balanced"));
        assert!(table.contains("100000"));
        assert!(table.contains("800"));
    }

    // 12. diff: Quick vs Quality shows num_iterations difference
    #[test]
    fn test_diff_quick_vs_quality() {
        let quick = TrainingPreset::get(&TrainingPresetName::Quick);
        let quality = TrainingPreset::get(&TrainingPresetName::Quality);
        let diffs = TrainingPreset::diff(&quick, &quality);
        let iters_diff = diffs.iter().find(|d| d.parameter == "num_iterations");
        assert!(iters_diff.is_some(), "expected num_iterations in diffs");
        let d = iters_diff.unwrap();
        assert_eq!(d.value_a, "50000");
        assert_eq!(d.value_b, "300000");
    }

    // 13. diff: same preset vs itself → empty diffs
    #[test]
    fn test_diff_same_preset_empty() {
        let p = TrainingPreset::get(&TrainingPresetName::Balanced);
        let diffs = TrainingPreset::diff(&p, &p);
        assert!(
            diffs.is_empty(),
            "no diffs expected when comparing preset to itself"
        );
    }

    // 14. estimated_minutes: quality > balanced > quick
    #[test]
    fn test_estimated_minutes_ordering() {
        let quick = TrainingPreset::get(&TrainingPresetName::Quick);
        let balanced = TrainingPreset::get(&TrainingPresetName::Balanced);
        let quality = TrainingPreset::get(&TrainingPresetName::Quality);
        assert!(quick.estimated_minutes() < balanced.estimated_minutes());
        assert!(balanced.estimated_minutes() < quality.estimated_minutes());
    }

    // 15. estimated_vram_mb: quality > quick
    #[test]
    fn test_estimated_vram_ordering() {
        let quick = TrainingPreset::get(&TrainingPresetName::Quick);
        let quality = TrainingPreset::get(&TrainingPresetName::Quality);
        assert!(quality.estimated_vram_mb() > quick.estimated_vram_mb());
    }

    // 16. quality_score: quality > balanced > quick
    #[test]
    fn test_quality_score_ordering() {
        let quick = TrainingPreset::get(&TrainingPresetName::Quick);
        let balanced = TrainingPreset::get(&TrainingPresetName::Balanced);
        let quality = TrainingPreset::get(&TrainingPresetName::Quality);
        assert!(
            quick.quality_score() < balanced.quality_score(),
            "quick ({}) should be less than balanced ({})",
            quick.quality_score(),
            balanced.quality_score()
        );
        assert!(
            balanced.quality_score() < quality.quality_score(),
            "balanced ({}) should be less than quality ({})",
            balanced.quality_score(),
            quality.quality_score()
        );
    }

    // 17. apply_override: "num_iterations=42000" modifies num_iterations
    #[test]
    fn test_apply_override_num_iterations() {
        let base = TrainingPreset::get(&TrainingPresetName::Quick);
        let overridden = apply_override(&base, "num_iterations=42000").unwrap();
        assert_eq!(overridden.num_iterations, 42_000);
        // other fields unchanged
        assert_eq!(overridden.sh_degree, base.sh_degree);
    }

    // 18. apply_override: unknown key → Err(InvalidOverride)
    #[test]
    fn test_apply_override_invalid_key() {
        let base = TrainingPreset::get(&TrainingPresetName::Quick);
        let err = apply_override(&base, "invalid_key=123").unwrap_err();
        assert!(matches!(err, PresetError::InvalidOverride(_)));
    }

    // 19. apply_overrides: multiple overrides applied in sequence
    #[test]
    fn test_apply_overrides_multiple() {
        let base = TrainingPreset::get(&TrainingPresetName::Balanced);
        let result = apply_overrides(
            &base,
            &["num_iterations=75000", "sh_degree=1", "verbose=true"],
        )
        .unwrap();
        assert_eq!(result.num_iterations, 75_000);
        assert_eq!(result.sh_degree, 1);
        assert!(result.verbose);
    }

    // 20. list_descriptions returns 7 entries
    #[test]
    fn test_list_descriptions_count() {
        let descs = TrainingPreset::list_descriptions();
        assert_eq!(descs.len(), 7);
        // Each entry has a non-empty name and description.
        for (name, desc) in &descs {
            assert!(!name.is_empty());
            assert!(!desc.is_empty());
        }
    }

    // --- Additional alias coverage ---

    #[test]
    fn test_from_str_default_alias() {
        assert_eq!(
            TrainingPresetName::from_str("default").unwrap(),
            TrainingPresetName::Balanced
        );
    }

    #[test]
    fn test_from_str_prod_alias() {
        assert_eq!(
            TrainingPresetName::from_str("prod").unwrap(),
            TrainingPresetName::Production
        );
    }

    #[test]
    fn test_from_str_headshot_alias() {
        assert_eq!(
            TrainingPresetName::from_str("headshot").unwrap(),
            TrainingPresetName::Portrait
        );
    }

    #[test]
    fn test_from_str_animation_alias() {
        assert_eq!(
            TrainingPresetName::from_str("animation").unwrap(),
            TrainingPresetName::Video
        );
    }

    #[test]
    fn test_apply_override_bool_flag() {
        let base = TrainingPreset::get(&TrainingPresetName::Quick);
        let r = apply_override(&base, "tensorboard_enabled=true").unwrap();
        assert!(r.tensorboard_enabled);
    }

    #[test]
    fn test_apply_override_f32_field() {
        let base = TrainingPreset::get(&TrainingPresetName::Balanced);
        let r = apply_override(&base, "lambda_lpips=0.2").unwrap();
        let expected: f32 = 0.2;
        assert!((r.lambda_lpips - expected).abs() < 1e-6);
    }

    #[test]
    fn test_apply_override_missing_equals() {
        let base = TrainingPreset::get(&TrainingPresetName::Quick);
        let err = apply_override(&base, "num_iterations42000").unwrap_err();
        assert!(matches!(err, PresetError::InvalidOverride(_)));
    }

    #[test]
    fn test_preset_names_round_trip() {
        for name in TrainingPresetName::all() {
            let s = name.as_str();
            let parsed = TrainingPresetName::from_str(s).unwrap();
            assert_eq!(parsed, *name, "round-trip failed for {s}");
        }
    }

    #[test]
    fn test_all_presets_sane_iterations() {
        for p in TrainingPreset::all() {
            assert!(
                p.num_iterations > 0,
                "preset {} has zero iterations",
                p.name
            );
            assert!(
                p.warmup_iterations < p.num_iterations,
                "warmup > total for {}",
                p.name
            );
        }
    }

    #[test]
    fn test_estimated_vram_non_zero() {
        for p in TrainingPreset::all() {
            assert!(
                p.estimated_vram_mb() > 0,
                "vram should be > 0 for {}",
                p.name
            );
        }
    }

    // 21. estimated_vram_mb: the real memory-estimator-backed figure is far
    // above the old flat-64-bytes/Gaussian heuristic for the `quality`
    // preset (1M Gaussians, sh_degree=3), which the audit measured as
    // undercounting VRAM by roughly an order of magnitude (64 MB reported
    // vs. ~1 GB real). Assert against a threshold well above the old
    // heuristic's output rather than an exact figure, since the estimator's
    // internals may reasonably evolve.
    #[test]
    fn test_estimated_vram_reflects_real_footprint() {
        let quality = TrainingPreset::get(&TrainingPresetName::Quality);
        let old_heuristic_mb = (quality.max_gaussians as u64 * 64) / 1_000_000;
        assert!(
            u64::from(quality.estimated_vram_mb()) > old_heuristic_mb * 4,
            "expected the real estimate ({} MB) to be well above the old \
             flat-64-bytes/Gaussian heuristic ({} MB)",
            quality.estimated_vram_mb(),
            old_heuristic_mb
        );
    }

    // -----------------------------------------------------------------------
    // apply_to / From<&TrainingPreset> for ProjectConfig
    // -----------------------------------------------------------------------

    #[test]
    fn test_preset_converts_to_valid_project_config() {
        for preset in TrainingPreset::all() {
            let config: crate::config::ProjectConfig = (&preset).into();
            assert_eq!(config.training.total_iterations, preset.num_iterations);
            assert_eq!(config.training.init.sh_degree, preset.sh_degree);
            assert_eq!(config.training.image_size, preset.image_width);
            assert_eq!(
                config.training.density_control.max_gaussians,
                preset.max_gaussians as usize
            );
            assert_eq!(config.output.checkpoint_interval, preset.checkpoint_every);
            assert_eq!(config.output.log_interval, preset.log_every);
            assert!(
                config.validate().is_ok(),
                "preset {} must convert to a valid ProjectConfig",
                preset.name
            );
        }
    }

    #[test]
    fn test_preset_apply_to_preserves_untouched_sections() {
        // `apply_to` must not disturb [model]/[device], which presets have
        // no opinion about.
        let mut config = crate::config::ProjectConfig::default();
        let default_model_path = config.model.flame_model_path.clone();
        let preset = TrainingPreset::get(&TrainingPresetName::Research);
        preset.apply_to(&mut config);
        assert_eq!(config.model.flame_model_path, default_model_path);
    }

    #[test]
    fn test_preset_apply_to_maps_optimizer_and_loss() {
        let preset = TrainingPreset::get(&TrainingPresetName::Portrait);
        let config: crate::config::ProjectConfig = (&preset).into();
        assert!((config.training.optimizer.position_lr - preset.position_lr).abs() < 1e-9);
        assert!((config.training.optimizer.sh_lr - preset.color_lr).abs() < 1e-9);
        assert!((config.training.loss.lambda_opacity_reg - preset.lambda_opacity).abs() < 1e-9);
        assert!((config.training.loss.lambda_scale_reg - preset.lambda_scale).abs() < 1e-9);
        assert_eq!(
            config.training.density_control.start_iteration,
            preset.densify_from_iter
        );
        assert_eq!(
            config.training.density_control.end_iteration,
            preset.densify_until_iter
        );
    }

    // -----------------------------------------------------------------------
    // apply_override: previously-missing fields
    // -----------------------------------------------------------------------

    #[test]
    fn test_apply_override_previously_missing_u32_fields() {
        let base = TrainingPreset::get(&TrainingPresetName::Quick);
        let r = apply_overrides(
            &base,
            &[
                "warmup_iterations=1234",
                "densify_from_iter=100",
                "densify_until_iter=2000",
                "opacity_reset_interval=999",
                "checkpoint_every=555",
            ],
        )
        .unwrap();
        assert_eq!(r.warmup_iterations, 1234);
        assert_eq!(r.densify_from_iter, 100);
        assert_eq!(r.densify_until_iter, 2000);
        assert_eq!(r.opacity_reset_interval, 999);
        assert_eq!(r.checkpoint_every, 555);
    }

    #[test]
    fn test_apply_override_previously_missing_usize_fields() {
        let base = TrainingPreset::get(&TrainingPresetName::Quick);
        let r = apply_overrides(
            &base,
            &[
                "keep_last_n_checkpoints=7",
                "batch_size=4",
                "num_workers=16",
            ],
        )
        .unwrap();
        assert_eq!(r.keep_last_n_checkpoints, 7);
        assert_eq!(r.batch_size, 4);
        assert_eq!(r.num_workers, 16);
    }

    #[test]
    fn test_apply_override_previously_missing_f32_fields() {
        let base = TrainingPreset::get(&TrainingPresetName::Quick);
        let r = apply_override(&base, "lambda_opacity=0.02").unwrap();
        assert!((r.lambda_opacity - 0.02).abs() < 1e-6);
        let r = apply_override(&r, "lambda_scale=0.03").unwrap();
        assert!((r.lambda_scale - 0.03).abs() < 1e-6);
        let r = apply_override(&r, "densify_grad_threshold=0.00025").unwrap();
        assert!((r.densify_grad_threshold - 0.00025).abs() < 1e-6);
    }

    #[test]
    fn test_invalid_override_error_lists_supported_keys() {
        let base = TrainingPreset::get(&TrainingPresetName::Quick);
        let err = apply_override(&base, "totally_bogus_key=1").unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("num_iterations"),
            "error should list supported keys: {msg}"
        );
        assert!(
            msg.contains("batch_size"),
            "error should list supported keys: {msg}"
        );
    }

    #[test]
    fn test_unknown_preset_error_lists_real_names_not_a_missing_subcommand() {
        let err = TrainingPresetName::from_str("bogus").unwrap_err();
        let msg = err.to_string();
        assert!(
            !msg.contains("oxigaf preset list"),
            "error must not reference a subcommand that does not exist: {msg}"
        );
        assert!(
            msg.contains("quality"),
            "error should list real preset names: {msg}"
        );
    }
}
