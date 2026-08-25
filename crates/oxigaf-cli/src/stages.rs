//! Composable pipeline stages for end-to-end workflows.
//!
//! This module orchestrates the reconstruction workflow as a sequence of
//! independently runnable [`PipelineStage`]s with progress tracking,
//! checkpointing, and error recovery.
//!
//! ## Relationship to the `pipeline` module
//!
//! The `train` subcommand drives the binary-only `pipeline` module, which runs
//! FLAME loading → Gaussian initialisation → training → export as a single
//! function.  This module exposes the same work as *composable* stages so a
//! caller can run tracking, diffusion, training, and export separately and
//! checkpoint between them.
//!
//! ## No fabricated results
//!
//! Every stage either performs real work or fails with an explicit error.
//! Where a step needs an external asset that OxiGAF deliberately does not
//! bundle — a facial-landmark detector for monocular FLAME tracking, or trained
//! multi-view diffusion weights — the stage says exactly what is missing rather
//! than emitting placeholder data that downstream tools would mistake for a
//! real result.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use nalgebra as na;
use serde::{Deserialize, Serialize};

use oxigaf::diffusion::DiffusionConfig;
use oxigaf::render::RasterConfig;
use oxigaf::trainer::diffusion_target::{DiffusionTargetConfig, DiffusionTargetGenerator};
use oxigaf::trainer::{Trainer, TrainingConfig};
use oxigaf_flame::sequence::FlameSequence;
use oxigaf_flame::{Camera, FlameModel, NormalMapRenderer};

/// The Gaussian model carried through the pipeline.
///
/// This is the renderer's model type — the same one [`Trainer`] optimises and
/// [`crate::export`] serialises — re-exported so stage users do not need a
/// direct dependency on `oxigaf-render`.
pub use oxigaf::render::gaussian::GaussianModel;

/// Frame rate assumed when a FLAME sequence carries no `fps` field.
const DEFAULT_FPS: f32 = 30.0;

/// File names searched for when `--input` names a directory of tracking output.
const TRACKING_FILE_CANDIDATES: &[&str] = &[
    "flame_params.json",
    "tracking.json",
    "params.json",
    "sequence.json",
];

/// Abstract pipeline stage that can report progress and execute
pub trait PipelineStage: Send + Sync {
    /// Get the name of this stage for logging and display
    fn name(&self) -> &str;

    /// Execute this stage with the given context
    ///
    /// # Errors
    ///
    /// Returns error if stage execution fails
    fn run(&mut self, ctx: &mut PipelineContext) -> Result<()>;

    /// Get the current progress of this stage (0.0 to 1.0)
    fn progress(&self) -> f32 {
        0.0
    }

    /// Estimate remaining time in seconds (if available)
    fn eta_seconds(&self) -> Option<f64> {
        None
    }
}

/// Context passed between pipeline stages
///
/// Holds all intermediate results and configuration needed by stages
#[derive(Default)]
pub struct PipelineContext {
    /// Input video path (if applicable)
    pub video_path: Option<PathBuf>,

    /// FLAME parameter sequence from tracking
    pub flame_sequence: Option<FlameSequence>,

    /// Generated multi-view images from diffusion
    pub generated_images: Vec<image::RgbImage>,

    /// Generated masks for training
    pub generated_masks: Vec<image::GrayImage>,

    /// Trained Gaussian model
    pub trained_model: Option<GaussianModel>,

    /// Metrics collected during processing
    pub metrics: HashMap<String, f32>,

    /// Checkpoint directory for saving intermediate results
    pub checkpoint_dir: Option<PathBuf>,

    /// Current stage index
    pub current_stage: usize,

    /// Total number of stages
    pub total_stages: usize,
}

impl PipelineContext {
    /// Create a new pipeline context
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the checkpoint directory
    pub fn with_checkpoint_dir(mut self, dir: PathBuf) -> Self {
        self.checkpoint_dir = Some(dir);
        self
    }

    /// Save checkpoint to disk
    ///
    /// # Errors
    ///
    /// Returns error if checkpoint cannot be saved
    pub fn save_checkpoint(&self, stage_name: &str) -> Result<()> {
        if let Some(ref checkpoint_dir) = self.checkpoint_dir {
            std::fs::create_dir_all(checkpoint_dir)
                .context("Failed to create checkpoint directory")?;

            let checkpoint_path = checkpoint_dir.join(format!("stage_{}.json", stage_name));

            let checkpoint = CheckpointData {
                stage_name: stage_name.to_string(),
                current_stage: self.current_stage,
                total_stages: self.total_stages,
                metrics: self.metrics.clone(),
                has_flame_sequence: self.flame_sequence.is_some(),
                num_generated_images: self.generated_images.len(),
                has_trained_model: self.trained_model.is_some(),
            };

            let json = serde_json::to_string_pretty(&checkpoint)
                .context("Failed to serialize checkpoint")?;
            std::fs::write(&checkpoint_path, json).context("Failed to write checkpoint file")?;

            tracing::info!("Saved checkpoint: {}", checkpoint_path.display());
        }

        Ok(())
    }

    /// Load checkpoint from disk
    ///
    /// # Errors
    ///
    /// Returns error if checkpoint cannot be loaded
    pub fn load_checkpoint(checkpoint_dir: &Path, stage_name: &str) -> Result<CheckpointData> {
        let checkpoint_path = checkpoint_dir.join(format!("stage_{}.json", stage_name));

        let json = std::fs::read_to_string(&checkpoint_path)
            .with_context(|| format!("Failed to read checkpoint: {}", checkpoint_path.display()))?;

        serde_json::from_str(&json).context("Failed to parse checkpoint JSON")
    }
}

/// Checkpoint data saved to disk
#[derive(Debug, Serialize, Deserialize)]
pub struct CheckpointData {
    pub stage_name: String,
    pub current_stage: usize,
    pub total_stages: usize,
    pub metrics: HashMap<String, f32>,
    pub has_flame_sequence: bool,
    pub num_generated_images: usize,
    pub has_trained_model: bool,
}

// ---------------------------------------------------------------------------
// Shared image helpers
// ---------------------------------------------------------------------------

/// Convert an 8-bit RGB image to the flat `[H·W·3]` `f32` HWC layout in `[0,1]`
/// used by the trainer and the diffusion target generator.
fn rgb_to_f32(img: &image::RgbImage) -> Vec<f32> {
    let mut out = Vec::with_capacity((img.width() as usize) * (img.height() as usize) * 3);
    for px in img.pixels() {
        out.push(f32::from(px[0]) / 255.0);
        out.push(f32::from(px[1]) / 255.0);
        out.push(f32::from(px[2]) / 255.0);
    }
    out
}

/// Convert a flat `[H·W·3]` `f32` HWC buffer back to an 8-bit RGB image.
///
/// # Errors
///
/// Returns an error when `data` is shorter than `width · height · 3`.
fn f32_to_rgb(data: &[f32], width: u32, height: u32) -> Result<image::RgbImage> {
    let expected = (width as usize) * (height as usize) * 3;
    if data.len() < expected {
        anyhow::bail!(
            "Generated view has {} samples, expected {expected} for {width}×{height} RGB",
            data.len(),
        );
    }
    let mut img = image::RgbImage::new(width, height);
    for (i, px) in img.pixels_mut().enumerate() {
        let base = i * 3;
        for c in 0..3 {
            px[c] = (data[base + c].clamp(0.0, 1.0) * 255.0).round() as u8;
        }
    }
    Ok(img)
}

/// Convert rendered normal maps into the per-view conditioning buffers the
/// diffusion target generator consumes.
///
/// Each map must already be at the pipeline's `width`×`height`; a mismatch is
/// an error rather than something to resample here, because it means the
/// camera set and the diffusion configuration have drifted apart and the
/// resulting conditioning would not correspond to the views being generated.
///
/// # Errors
///
/// Returns an error when `normal_maps` is empty, or when any map's dimensions
/// differ from `width`×`height`.
fn normal_map_conditioning(
    normal_maps: &[image::RgbImage],
    width: u32,
    height: u32,
) -> Result<Vec<Vec<f32>>> {
    if normal_maps.is_empty() {
        anyhow::bail!("No normal maps were rendered — diffusion has nothing to condition on");
    }
    let mut out = Vec::with_capacity(normal_maps.len());
    for (i, map) in normal_maps.iter().enumerate() {
        if map.width() != width || map.height() != height {
            anyhow::bail!(
                "Normal map {i} is {}×{}, expected {width}×{height}",
                map.width(),
                map.height(),
            );
        }
        out.push(rgb_to_f32(map));
    }
    Ok(out)
}

/// Derive a foreground mask from a rendered normal map.
///
/// `NormalMapRenderer` leaves untouched pixels black, so any non-black pixel is
/// covered by the FLAME mesh.
fn coverage_mask(normal_map: &image::RgbImage) -> image::GrayImage {
    let mut mask = image::GrayImage::new(normal_map.width(), normal_map.height());
    for (m, px) in mask.pixels_mut().zip(normal_map.pixels()) {
        m[0] = if px[0] != 0 || px[1] != 0 || px[2] != 0 {
            255
        } else {
            0
        };
    }
    mask
}

/// Build `num_views` pinhole cameras evenly spaced on a horizontal orbit,
/// all looking at the origin.
fn orbit_cameras(num_views: usize, width: u32, height: u32, radius: f32) -> Vec<Camera> {
    (0..num_views)
        .map(|i| {
            let azimuth = if num_views == 0 {
                0.0
            } else {
                (i as f32) * 360.0 / (num_views as f32)
            };
            orbit_camera(azimuth, 10.0, radius, width, height)
        })
        .collect()
}

/// Create a pinhole camera looking at the origin from spherical coordinates.
fn orbit_camera(
    azimuth_deg: f32,
    elevation_deg: f32,
    distance: f32,
    width: u32,
    height: u32,
) -> Camera {
    let az = azimuth_deg.to_radians();
    let el = elevation_deg.to_radians();

    let eye = na::Vector3::new(
        distance * el.cos() * az.sin(),
        distance * el.sin(),
        distance * el.cos() * az.cos(),
    );
    let forward = (-eye).normalize();
    let world_up = na::Vector3::new(0.0, 1.0, 0.0);
    let right = if forward.cross(&world_up).norm() < 1e-6 {
        na::Vector3::new(1.0, 0.0, 0.0)
    } else {
        forward.cross(&world_up).normalize()
    };
    let up = right.cross(&forward);

    let rotation = na::Matrix3::from_columns(&[right, up, -forward]).transpose();
    let translation = -(rotation * eye);
    let focal = width as f32 * 1.5;

    Camera {
        rotation,
        translation,
        focal_x: focal,
        focal_y: focal,
        cx: width as f32 / 2.0,
        cy: height as f32 / 2.0,
        width,
        height,
        near: 0.01,
        far: 10.0,
    }
}

// ---------------------------------------------------------------------------
// Tracking
// ---------------------------------------------------------------------------

/// Tracking stage: resolve the per-frame FLAME parameter sequence for the input.
///
/// ## Why this does not run a tracker on raw video
///
/// Fitting FLAME to monocular footage requires a facial-landmark detector, and
/// every usable detector is a trained neural network.  OxiGAF ships no such
/// weights, so this stage consumes *already tracked* parameters: a FLAME
/// sequence JSON, or a directory containing one (see
/// `TRACKING_FILE_CANDIDATES`).  Raw video or an image folder is rejected
/// with an explicit message instead of silently producing an empty sequence.
pub struct TrackingStage {
    video_path: PathBuf,
    output_path: PathBuf,
    progress: f32,
}

impl TrackingStage {
    /// Create a new tracking stage.
    ///
    /// * `video_path` — the FLAME sequence JSON, or a directory containing one.
    /// * `output_path` — where the tracking manifest is written.
    pub fn new(video_path: PathBuf, output_path: PathBuf) -> Self {
        Self {
            video_path,
            output_path,
            progress: 0.0,
        }
    }

    /// Record what was actually loaded so later runs and downstream tools can
    /// see which parameters produced the avatar.
    fn write_manifest(&self, params_path: &Path, num_frames: usize, fps: f32) -> Result<()> {
        if let Some(parent) = self.output_path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)
                    .context("Failed to create tracking output directory")?;
            }
        }
        let source = self.video_path.display().to_string();
        let flame_params = params_path.display().to_string();
        let manifest = serde_json::json!({
            "source": source,
            "flame_params": flame_params,
            "num_frames": num_frames,
            "fps": fps,
        });
        let json =
            serde_json::to_string_pretty(&manifest).context("Failed to serialize manifest")?;
        std::fs::write(&self.output_path, json).with_context(|| {
            format!(
                "Failed to write tracking manifest: {}",
                self.output_path.display()
            )
        })
    }
}

/// Resolve the FLAME parameter JSON referenced by a tracking input.
///
/// # Errors
///
/// Returns an error when the path is missing, names raw footage, or names a
/// directory without recognised tracking output.
fn resolve_tracking_params(input: &Path) -> Result<PathBuf> {
    if !input.exists() {
        anyhow::bail!("Tracking input does not exist: {}", input.display());
    }

    let is_json = input
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("json"))
        .unwrap_or(false);

    if input.is_file() {
        if is_json {
            return Ok(input.to_path_buf());
        }
        anyhow::bail!(
            "Cannot track {}: FLAME tracking from raw footage needs a facial-landmark \
             detector model, which OxiGAF does not bundle. Supply pre-computed per-frame \
             FLAME parameters as a sequence JSON \
             ({{\"fps\": 30.0, \"frames\": [{{\"shape\": [..], \"expression\": [..], \
             \"pose\": [..]}}]}}) and pass that file (or a directory containing one of: {}).",
            input.display(),
            TRACKING_FILE_CANDIDATES.join(", "),
        );
    }

    for name in TRACKING_FILE_CANDIDATES {
        let candidate = input.join(name);
        if candidate.is_file() {
            return Ok(candidate);
        }
    }

    anyhow::bail!(
        "No FLAME parameter file found in {}: expected one of {}. FLAME tracking from raw \
         frames needs a facial-landmark detector model, which OxiGAF does not bundle.",
        input.display(),
        TRACKING_FILE_CANDIDATES.join(", "),
    )
}

impl PipelineStage for TrackingStage {
    fn name(&self) -> &str {
        "Tracking"
    }

    fn run(&mut self, ctx: &mut PipelineContext) -> Result<()> {
        tracing::info!(
            "Resolving FLAME tracking data from {}",
            self.video_path.display()
        );
        self.progress = 0.0;

        let params_path = resolve_tracking_params(&self.video_path)?;
        self.progress = 0.25;

        tracing::info!(
            "Loading FLAME parameter sequence from {}",
            params_path.display()
        );
        let sequence = FlameSequence::from_json(&params_path)
            .with_context(|| format!("Failed to load FLAME sequence: {}", params_path.display()))?;

        let num_frames = sequence.num_frames();
        if num_frames == 0 {
            anyhow::bail!(
                "FLAME sequence {} contains no frames — downstream stages need at least one",
                params_path.display()
            );
        }
        let fps = sequence.fps().unwrap_or(DEFAULT_FPS);
        self.progress = 0.75;

        self.write_manifest(&params_path, num_frames, fps)?;

        ctx.video_path = Some(self.video_path.clone());
        ctx.flame_sequence = Some(sequence);
        ctx.metrics.insert("tracking_fps".to_string(), fps);
        ctx.metrics
            .insert("tracking_frames".to_string(), num_frames as f32);

        tracing::info!("Tracking resolved: {num_frames} frame(s) at {fps} fps");
        self.progress = 1.0;
        Ok(())
    }

    fn progress(&self) -> f32 {
        self.progress
    }
}

// ---------------------------------------------------------------------------
// Diffusion
// ---------------------------------------------------------------------------

/// Diffusion stage: generate multi-view pseudo ground-truth from FLAME geometry.
///
/// The stage renders normal maps of the tracked FLAME mesh from a ring of
/// cameras and hands them to the multi-view diffusion pipeline as conditioning.
/// Trained diffusion weights are required: without them there is nothing to
/// generate, and this stage reports that rather than pushing blank images.
pub struct DiffusionStage {
    num_views: usize,
    resolution: (u32, u32),
    progress: f32,
    weights_dir: Option<PathBuf>,
    flame_model_path: Option<PathBuf>,
    orbit_radius: f32,
}

impl DiffusionStage {
    /// Create a new diffusion stage.
    ///
    /// `num_views` and `resolution` must match the diffusion model's
    /// configuration ([`DiffusionConfig`] defaults: 4 views at 256×256);
    /// [`PipelineStage::run`] validates this.
    pub fn new(num_views: usize, resolution: (u32, u32)) -> Self {
        Self {
            num_views,
            resolution,
            progress: 0.0,
            weights_dir: None,
            flame_model_path: None,
            orbit_radius: 0.6,
        }
    }

    /// Directory holding the multi-view diffusion safetensors weights.
    #[must_use]
    pub fn with_weights(mut self, weights_dir: PathBuf) -> Self {
        self.weights_dir = Some(weights_dir);
        self
    }

    /// Directory holding the converted FLAME model used to build the mesh.
    #[must_use]
    pub fn with_flame_model(mut self, flame_model_path: PathBuf) -> Self {
        self.flame_model_path = Some(flame_model_path);
        self
    }

    /// Distance of the orbit cameras from the origin (metres).
    #[must_use]
    pub fn with_orbit_radius(mut self, radius: f32) -> Self {
        self.orbit_radius = radius;
        self
    }
}

impl PipelineStage for DiffusionStage {
    fn name(&self) -> &str {
        "Diffusion"
    }

    fn run(&mut self, ctx: &mut PipelineContext) -> Result<()> {
        let (width, height) = self.resolution;
        tracing::info!("Generating {} views at {width}×{height}", self.num_views);
        self.progress = 0.0;

        // 1. Tracked geometry is mandatory conditioning.
        let params = {
            let sequence = ctx.flame_sequence.as_mut().ok_or_else(|| {
                anyhow::anyhow!(
                    "No FLAME sequence available for diffusion — run TrackingStage first"
                )
            })?;
            if sequence.num_frames() == 0 {
                anyhow::bail!("FLAME sequence is empty — diffusion has nothing to condition on");
            }
            sequence
                .get_frame(0)
                .context("Failed to read FLAME frame 0")?
                .clone()
        };

        // 2. The generator is driven by the model's own configuration, so the
        //    requested view count / resolution has to agree with it.
        let diff_config = DiffusionConfig::default();
        if self.num_views != diff_config.num_views {
            anyhow::bail!(
                "DiffusionStage is configured for {} views but the diffusion model expects {} — \
                 construct it with DiffusionStage::new({}, ..)",
                self.num_views,
                diff_config.num_views,
                diff_config.num_views,
            );
        }
        let expected_size = diff_config.image_size as u32;
        if width != expected_size || height != expected_size {
            anyhow::bail!(
                "DiffusionStage is configured for {width}×{height} but the diffusion model \
                 generates {expected_size}×{expected_size}"
            );
        }

        // 3. Weights are an external asset; refuse rather than fake output.
        //    Cloned so `self.progress` stays writable below.
        let weights_dir = self.weights_dir.clone().ok_or_else(|| {
            anyhow::anyhow!(
                "Multi-view diffusion requires trained weights (unet/, vae/, image_encoder/ \
                 safetensors). Call DiffusionStage::with_weights(<dir>) — `oxigaf setup` \
                 downloads them into the asset cache."
            )
        })?;
        if !weights_dir.is_dir() {
            anyhow::bail!(
                "Diffusion weights directory not found: {}",
                weights_dir.display()
            );
        }

        let flame_model_path = self.flame_model_path.clone().ok_or_else(|| {
            anyhow::anyhow!(
                "DiffusionStage needs the FLAME model to render conditioning normal maps: \
                 call DiffusionStage::with_flame_model(<dir>)"
            )
        })?;

        // 4. Build the conditioning normal maps from the tracked mesh.
        let flame = FlameModel::load(&flame_model_path).with_context(|| {
            format!(
                "Failed to load FLAME model from {}",
                flame_model_path.display()
            )
        })?;
        let mesh = flame.forward(&params);
        let cameras = orbit_cameras(self.num_views, width, height, self.orbit_radius);
        let normal_maps: Vec<image::RgbImage> = cameras
            .iter()
            .map(|cam| NormalMapRenderer::render(&mesh, cam))
            .collect();
        let conditioning = normal_map_conditioning(&normal_maps, width, height)?;
        self.progress = 0.35;

        // 5. Run the diffusion pipeline.  `warmup_iterations = 0` so the
        //    generator denoises immediately instead of echoing its input.
        let target_config = DiffusionTargetConfig {
            warmup_iterations: 0,
            ..DiffusionTargetConfig::default()
        };
        let mut generator = DiffusionTargetGenerator::new(target_config);
        generator.load_pipeline(&weights_dir).with_context(|| {
            format!(
                "Failed to load diffusion pipeline from {}",
                weights_dir.display()
            )
        })?;
        self.progress = 0.5;

        // Iteration 1 is past the zero-length warmup, so the generator denoises
        // rather than echoing its conditioning input back.
        //
        // `conditioning` is passed **twice**, deliberately — this is not a
        // copy-paste slip:
        //
        // * as `normal_maps`, it is VAE-encoded into the geometry latents the
        //   U-Net concatenates onto its input. Passing `None` here (as this
        //   call used to) left those channels zero-filled, so the generated
        //   views were untied from the tracked FLAME geometry and the
        //   generator logged its one-shot "geometry conditioning is
        //   zero-filled" warning on every run.
        // * as `rendered`, it supplies the CLIP identity reference (the first
        //   view) and the warmup pass-through. At this point in the pipeline
        //   no Gaussian model exists yet — that is what these targets are
        //   about to be used to fit — so the normal maps are the only
        //   image-space signal available. `TrainingStage` later drives the
        //   generator with genuine renders.
        let targets = generator
            .generate_targets_with_normals(
                &conditioning,
                &cameras,
                Some(&conditioning),
                1,
                width,
                height,
            )
            .context("Multi-view diffusion generation failed")?;
        if targets.len() < self.num_views {
            anyhow::bail!(
                "Diffusion produced {} view(s), expected {}",
                targets.len(),
                self.num_views
            );
        }

        // 6. Publish the generated views and their coverage masks.
        for (i, target) in targets.iter().take(self.num_views).enumerate() {
            let img = f32_to_rgb(target, width, height)
                .with_context(|| format!("Invalid diffusion output for view {i}"))?;
            ctx.generated_images.push(img);
            let mask = normal_maps
                .get(i)
                .map(coverage_mask)
                .unwrap_or_else(|| image::GrayImage::new(width, height));
            ctx.generated_masks.push(mask);
            self.progress = 0.5 + 0.5 * ((i + 1) as f32 / self.num_views as f32);
        }

        ctx.metrics
            .insert("num_views".to_string(), self.num_views as f32);
        self.progress = 1.0;
        Ok(())
    }

    fn progress(&self) -> f32 {
        self.progress
    }
}

// ---------------------------------------------------------------------------
// Training
// ---------------------------------------------------------------------------

/// Everything [`TrainingStage`] needs to run a real optimisation loop.
///
/// 3D Gaussian training is GPU work: it cannot be approximated, so the stage
/// refuses to run without this.
pub struct TrainingSetup {
    /// GPU device backing the rasterizer.
    pub device: wgpu::Device,
    /// Queue paired with `device`.
    pub queue: wgpu::Queue,
    /// Optimiser / density / loss configuration.
    pub training_config: TrainingConfig,
    /// Rasterizer configuration (image size, background, tile size).
    pub raster_config: RasterConfig,
    /// Gaussians to start from (e.g. from `GaussianInitializer::initialize`).
    pub initial_model: GaussianModel,
    /// RNG seed for reproducibility.
    pub seed: u64,
}

/// Training stage: optimise a 3D Gaussian Splatting model.
pub struct TrainingStage {
    num_iterations: usize,
    current_iteration: usize,
    start_time: Option<Instant>,
    setup: Option<TrainingSetup>,
    final_loss: Option<f32>,
}

impl TrainingStage {
    /// Create a new training stage
    pub fn new(num_iterations: usize) -> Self {
        Self {
            num_iterations,
            current_iteration: 0,
            start_time: None,
            setup: None,
            final_loss: None,
        }
    }

    /// Provide the GPU device, configuration, and starting model.
    #[must_use]
    pub fn with_setup(mut self, setup: TrainingSetup) -> Self {
        self.setup = Some(setup);
        self
    }

    /// Loss of the last completed training step, if any.
    pub fn final_loss(&self) -> Option<f32> {
        self.final_loss
    }
}

impl PipelineStage for TrainingStage {
    fn name(&self) -> &str {
        "Training"
    }

    fn run(&mut self, ctx: &mut PipelineContext) -> Result<()> {
        tracing::info!(
            "Training 3D Gaussians for {} iterations",
            self.num_iterations
        );

        if ctx.generated_images.is_empty() {
            anyhow::bail!("No generated images available for training");
        }
        if self.num_iterations == 0 {
            anyhow::bail!("TrainingStage was configured with 0 iterations");
        }

        let setup = self.setup.take().ok_or_else(|| {
            anyhow::anyhow!(
                "TrainingStage has no GPU setup: call TrainingStage::with_setup(TrainingSetup \
                 {{ device, queue, training_config, raster_config, initial_model, seed }}). \
                 Gaussian optimisation runs on the GPU rasterizer and cannot be simulated."
            )
        })?;

        self.start_time = Some(Instant::now());
        let mut trainer = Trainer::new(
            setup.training_config,
            setup.initial_model,
            setup.raster_config,
            setup.device,
            setup.queue,
            setup.seed,
        )
        .context("Failed to create trainer")?;

        let mut last_loss = f32::NAN;
        for i in 0..self.num_iterations {
            let step = trainer
                .train_step()
                .with_context(|| format!("Training step {} failed", i + 1))?;
            self.current_iteration = i + 1;
            last_loss = step.loss.total;

            if i.is_multiple_of(100) {
                ctx.metrics.insert(format!("loss_iter_{}", i), last_loss);
            }
        }

        self.final_loss = Some(last_loss);
        ctx.metrics.insert("final_loss".to_string(), last_loss);
        ctx.metrics
            .insert("iterations".to_string(), self.num_iterations as f32);
        ctx.metrics
            .insert("num_gaussians".to_string(), trainer.model.len() as f32);
        ctx.trained_model = Some(trainer.model.clone());

        tracing::info!(
            "Training complete: {} Gaussians, final loss {last_loss:.6}",
            trainer.model.len()
        );
        Ok(())
    }

    fn progress(&self) -> f32 {
        if self.num_iterations == 0 {
            return 0.0;
        }
        self.current_iteration as f32 / self.num_iterations as f32
    }

    fn eta_seconds(&self) -> Option<f64> {
        if let Some(start) = self.start_time {
            if self.current_iteration > 0 {
                let elapsed = start.elapsed().as_secs_f64();
                let per_iter = elapsed / self.current_iteration as f64;
                let remaining =
                    self.num_iterations.saturating_sub(self.current_iteration) as f64 * per_iter;
                return Some(remaining);
            }
        }
        None
    }
}

// ---------------------------------------------------------------------------
// Export
// ---------------------------------------------------------------------------

/// Export stage: export trained model to various formats
pub struct ExportStage {
    format: ExportFormat,
    output_path: PathBuf,
}

/// Supported export formats
#[derive(Debug, Clone, Copy)]
pub enum ExportFormat {
    /// ASCII PLY in the 3D Gaussian Splatting property layout
    Ply,
    /// glTF 2.0 (GLB) with the `OXIGAF_gaussians` extension
    Gltf,
    /// safetensors — the binary interchange format for Gaussian models
    Binary,
}

impl ExportStage {
    /// Create a new export stage
    pub fn new(format: ExportFormat, output_path: PathBuf) -> Self {
        Self {
            format,
            output_path,
        }
    }
}

impl PipelineStage for ExportStage {
    fn name(&self) -> &str {
        "Export"
    }

    fn run(&mut self, ctx: &mut PipelineContext) -> Result<()> {
        tracing::info!(
            "Exporting model to {:?} format: {}",
            self.format,
            self.output_path.display()
        );

        let Some(model) = ctx.trained_model.as_ref() else {
            anyhow::bail!("No trained model in context");
        };

        // Delegate to the crate's real serialisers so the stage output is byte
        // for byte the same as `oxigaf export`.
        match self.format {
            ExportFormat::Ply => {
                crate::export::export_ply(model, &self.output_path).context("PLY export failed")?
            }
            ExportFormat::Gltf => crate::export::export_gltf(model, &self.output_path, true)
                .context("glTF export failed")?,
            ExportFormat::Binary => crate::export::export_safetensors(model, &self.output_path)
                .context("safetensors export failed")?,
        }

        ctx.metrics
            .insert("num_gaussians".to_string(), model.len() as f32);
        ctx.metrics.insert("exported".to_string(), 1.0);

        Ok(())
    }

    fn progress(&self) -> f32 {
        // Export is typically fast, so either 0 or 1
        1.0
    }
}

// ---------------------------------------------------------------------------
// Executor
// ---------------------------------------------------------------------------

/// Pipeline executor that runs stages sequentially
pub struct PipelineExecutor {
    stages: Vec<Box<dyn PipelineStage>>,
    show_progress: bool,
}

impl PipelineExecutor {
    /// Create a new pipeline executor
    pub fn new() -> Self {
        Self {
            stages: Vec::new(),
            show_progress: true,
        }
    }

    /// Add a stage to the pipeline
    pub fn add_stage(&mut self, stage: Box<dyn PipelineStage>) -> &mut Self {
        self.stages.push(stage);
        self
    }

    /// Set whether to show progress bars
    pub fn show_progress(&mut self, show: bool) -> &mut Self {
        self.show_progress = show;
        self
    }

    /// Execute all stages
    ///
    /// # Errors
    ///
    /// Returns error if any stage fails
    pub fn execute(&mut self, mut ctx: PipelineContext) -> Result<PipelineContext> {
        ctx.total_stages = self.stages.len();

        for (i, stage) in self.stages.iter_mut().enumerate() {
            ctx.current_stage = i;

            let stage_name = stage.name().to_string(); // Clone to avoid borrow issue
            tracing::info!(
                "Executing stage {}/{}: {}",
                i + 1,
                ctx.total_stages,
                stage_name
            );

            let pb = if self.show_progress {
                let pb = ProgressBar::new(100);
                pb.set_style(
                    ProgressStyle::default_bar()
                        .template("[{elapsed_precise}] [{bar:40.cyan/blue}] {pos}% {msg}")
                        .context("Failed to create progress bar template")?
                        .progress_chars("#>-"),
                );
                pb.set_message(stage_name.clone());
                Some(pb)
            } else {
                None
            };

            // Run stage
            stage
                .run(&mut ctx)
                .with_context(|| format!("Stage '{}' failed", stage_name))?;

            if let Some(pb) = pb {
                pb.set_position(100);
                pb.finish_with_message(format!("{} complete", stage_name));
            }

            // Save checkpoint
            ctx.save_checkpoint(&stage_name)?;

            tracing::info!("Stage {} complete", stage_name);
        }

        tracing::info!("Pipeline complete! Executed {} stages", ctx.total_stages);

        Ok(ctx)
    }
}

impl Default for PipelineExecutor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxigaf::render::gaussian::GaussianAttributes;
    use tempfile::TempDir;

    /// Build a real `n`-Gaussian model with distinguishable, non-zero data so
    /// export round-trips assert on content rather than only on vertex count.
    fn test_model(n: usize) -> GaussianModel {
        let gaussians: Vec<GaussianAttributes> = (0..n)
            .map(|i| GaussianAttributes {
                position: [i as f32 * 0.1, i as f32 * 0.2, i as f32 * 0.3],
                _pad0: 0.0,
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: [f32::ln(0.05); 3],
                opacity: 2.0,
            })
            .collect();
        GaussianModel {
            gaussians,
            sh_coeffs: (0..n * 3).map(|i| 0.25 + i as f32 * 0.01).collect(),
            sh_degree: 0,
            face_indices: vec![0u32; n],
            barycentric: vec![[1.0_f32 / 3.0; 3]; n],
            local_offsets: vec![[0.0_f32; 3]; n],
            is_rigid: vec![true; n],
        }
    }

    /// Write a minimal but valid FLAME sequence JSON, returning its path.
    fn write_sequence_json(dir: &Path, name: &str, num_frames: usize) -> PathBuf {
        let frames: Vec<serde_json::Value> = (0..num_frames)
            .map(|i| {
                let shape: Vec<f32> = vec![0.1 * i as f32, -0.2];
                let expression: Vec<f32> = vec![0.0, 0.3];
                let pose: Vec<f32> = vec![0.0; 15];
                let translation: Vec<f32> = vec![0.0, 0.0, 0.0];
                serde_json::json!({
                    "shape": shape,
                    "expression": expression,
                    "pose": pose,
                    "translation": translation,
                })
            })
            .collect();
        let doc = serde_json::json!({ "fps": 30.0, "frames": frames });
        let path = dir.join(name);
        std::fs::write(
            &path,
            serde_json::to_string_pretty(&doc).expect("test: serialize sequence"),
        )
        .expect("test: write sequence json");
        path
    }

    #[test]
    fn test_pipeline_context_creation() {
        let ctx = PipelineContext::new();
        assert_eq!(ctx.current_stage, 0);
        assert_eq!(ctx.total_stages, 0);
        assert!(ctx.flame_sequence.is_none());
        assert!(ctx.trained_model.is_none());
    }

    #[test]
    fn test_checkpoint_save_load() {
        let temp_dir = TempDir::new().expect("test: temp dir creation should succeed");
        let mut ctx = PipelineContext::new().with_checkpoint_dir(temp_dir.path().to_path_buf());
        ctx.current_stage = 1;
        ctx.total_stages = 3;
        ctx.metrics.insert("test".to_string(), 42.0);

        // Save checkpoint
        ctx.save_checkpoint("test_stage")
            .expect("test: checkpoint operation should succeed");

        // Load checkpoint
        let loaded = PipelineContext::load_checkpoint(temp_dir.path(), "test_stage")
            .expect("test: checkpoint operation should succeed");
        assert_eq!(loaded.stage_name, "test_stage");
        assert_eq!(loaded.current_stage, 1);
        assert_eq!(loaded.total_stages, 3);
        assert_eq!(loaded.metrics.get("test"), Some(&42.0));
    }

    #[test]
    fn test_training_stage_progress() {
        let mut stage = TrainingStage::new(100);
        assert_eq!(stage.progress(), 0.0);

        stage.current_iteration = 50;
        assert_eq!(stage.progress(), 0.5);

        stage.current_iteration = 100;
        assert_eq!(stage.progress(), 1.0);
    }

    #[test]
    fn test_pipeline_executor() {
        let mut executor = PipelineExecutor::new();
        executor.show_progress(false);

        let ctx = PipelineContext::new();

        // Execute empty pipeline
        let result = executor.execute(ctx);
        assert!(result.is_ok());
    }

    /// Regression test: `TrackingStage` must publish the loaded sequence into
    /// the context.  It previously built a sequence and dropped it, so
    /// `DiffusionStage` always bailed with "No FLAME sequence available".
    #[test]
    fn test_tracking_stage_publishes_sequence() {
        let dir = std::env::temp_dir().join("oxigaf_stage_tracking_publish");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("test: create temp dir");
        let params = write_sequence_json(&dir, "flame_params.json", 3);
        let manifest = dir.join("tracking_manifest.json");

        let mut stage = TrackingStage::new(params.clone(), manifest.clone());
        let mut ctx = PipelineContext::new();
        stage.run(&mut ctx).expect("test: tracking should succeed");

        let sequence = ctx
            .flame_sequence
            .as_ref()
            .expect("tracking must publish the FLAME sequence into the context");
        assert_eq!(sequence.num_frames(), 3);
        assert_eq!(ctx.metrics.get("tracking_frames"), Some(&3.0));
        assert!(manifest.exists(), "tracking manifest must be written");
        assert!((stage.progress() - 1.0).abs() < f32::EPSILON);

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A directory of tracking output resolves through the candidate list.
    #[test]
    fn test_tracking_stage_resolves_directory() {
        let dir = std::env::temp_dir().join("oxigaf_stage_tracking_dir");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("test: create temp dir");
        write_sequence_json(&dir, "tracking.json", 2);

        let resolved = resolve_tracking_params(&dir).expect("directory must resolve");
        assert!(resolved.ends_with("tracking.json"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Raw footage is rejected with an explanation, not silently accepted.
    #[test]
    fn test_tracking_stage_rejects_raw_footage() {
        let dir = std::env::temp_dir().join("oxigaf_stage_tracking_reject");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("test: create temp dir");
        let clip = dir.join("clip.mp4");
        std::fs::write(&clip, b"not a real video").expect("test: write fixture");

        let err = resolve_tracking_params(&clip).expect_err("raw video must be rejected");
        let msg = err.to_string();
        assert!(
            msg.contains("facial-landmark detector"),
            "error must explain the missing asset: {msg}"
        );

        // Empty directory: also rejected.
        let empty = dir.join("empty");
        std::fs::create_dir_all(&empty).expect("test: create temp dir");
        assert!(resolve_tracking_params(&empty).is_err());

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// `DiffusionStage` must not invent blank images when weights are absent.
    #[test]
    fn test_diffusion_stage_requires_weights() {
        let dir = std::env::temp_dir().join("oxigaf_stage_diffusion_weights");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("test: create temp dir");
        let params = write_sequence_json(&dir, "flame_params.json", 1);

        let mut ctx = PipelineContext::new();
        ctx.flame_sequence = Some(FlameSequence::from_json(&params).expect("test: load sequence"));

        let mut stage = DiffusionStage::new(4, (256, 256));
        let err = stage
            .run(&mut ctx)
            .expect_err("diffusion without weights must fail");
        assert!(
            err.to_string().contains("requires trained weights"),
            "unexpected error: {err}"
        );
        assert!(
            ctx.generated_images.is_empty(),
            "no placeholder images may be produced"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A view-count mismatch is reported before any weights are touched.
    #[test]
    fn test_diffusion_stage_validates_view_count() {
        let dir = std::env::temp_dir().join("oxigaf_stage_diffusion_views");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("test: create temp dir");
        let params = write_sequence_json(&dir, "flame_params.json", 1);

        let mut ctx = PipelineContext::new();
        ctx.flame_sequence = Some(FlameSequence::from_json(&params).expect("test: load sequence"));

        let mut stage = DiffusionStage::new(7, (256, 256));
        let err = stage.run(&mut ctx).expect_err("view mismatch must fail");
        assert!(err.to_string().contains("views"), "unexpected error: {err}");

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// `TrainingStage` must refuse to run without a GPU setup instead of
    /// sleeping and fabricating `loss = 1/(i+1)`.
    #[test]
    fn test_training_stage_requires_setup() {
        let mut stage = TrainingStage::new(10);
        let mut ctx = PipelineContext::new();
        ctx.generated_images.push(image::RgbImage::new(4, 4));

        let err = stage
            .run(&mut ctx)
            .expect_err("training without a GPU setup must fail");
        assert!(
            err.to_string().contains("with_setup"),
            "error must name the missing setup: {err}"
        );
        assert!(
            ctx.trained_model.is_none(),
            "no placeholder model may be produced"
        );
        assert_eq!(ctx.metrics.get("final_loss"), None);
    }

    #[test]
    fn test_export_stage() {
        let temp_dir = TempDir::new().expect("test: temp dir creation should succeed");
        let output_path = temp_dir.path().join("model.ply");

        let mut stage = ExportStage::new(ExportFormat::Ply, output_path.clone());
        let mut ctx = PipelineContext::new();

        // Should fail without model
        assert!(stage.run(&mut ctx).is_err());

        // Add model and retry
        ctx.trained_model = Some(test_model(10));
        assert!(stage.run(&mut ctx).is_ok());

        // Check file was created
        assert!(output_path.exists());
    }

    /// Regression test: the exported PLY must carry the model's real data.
    ///
    /// The stage used to emit `0 0 0 0 0 0 0 0 0 0 -5 -5 -5 1 0 0 0` per
    /// Gaussian, so a vertex-count assertion passed while every splat sat at
    /// the origin with zero colour.
    #[test]
    fn test_export_stage_writes_real_gaussians() {
        let dir = std::env::temp_dir().join("oxigaf_stage_test");
        std::fs::create_dir_all(&dir).expect("test: create temp dir");
        let out = dir.join("test_export_stage_writes_real_gaussians.ply");

        let mut stage = ExportStage::new(ExportFormat::Ply, out.clone());
        let model = test_model(10);
        let mut ctx = PipelineContext::default();
        ctx.trained_model = Some(model);

        stage
            .run(&mut ctx)
            .expect("test: export stage should succeed");
        assert!(out.exists(), "PLY file should have been written");

        let loaded =
            crate::export::load_ply(&out).expect("test: load_ply should parse the written file");
        assert_eq!(loaded.len(), 10, "loaded model must have 10 Gaussians");

        // Positions must round-trip — not collapse to the origin.
        for (i, g) in loaded.gaussians.iter().enumerate() {
            let expected = [i as f32 * 0.1, i as f32 * 0.2, i as f32 * 0.3];
            for (k, exp) in expected.iter().copied().enumerate() {
                assert!(
                    (g.position[k] - exp).abs() < 1e-4,
                    "gaussian {i} axis {k}: got {}, expected {}",
                    g.position[k],
                    exp
                );
            }
        }
        // Opacity and scale must survive too (the placeholder wrote 0 / -5).
        assert!(
            (loaded.gaussians[0].opacity - 2.0).abs() < 1e-4,
            "opacity must round-trip, got {}",
            loaded.gaussians[0].opacity
        );
        assert!(
            (loaded.gaussians[0].scale[0] - f32::ln(0.05)).abs() < 1e-4,
            "scale must round-trip, got {}",
            loaded.gaussians[0].scale[0]
        );
        // Non-zero SH DC means the splats actually have colour.
        assert!(
            loaded.sh_coeffs.iter().any(|c| c.abs() > 1e-6),
            "SH coefficients must not all be zero"
        );

        let _ = std::fs::remove_file(&out);
    }

    /// Every `ExportStage` format must produce a non-empty file.
    #[test]
    fn test_export_stage_writes_nonempty_file() {
        let dir = std::env::temp_dir().join("oxigaf_stage_test_nonempty");
        std::fs::create_dir_all(&dir).expect("test: create temp dir");

        let formats: &[(ExportFormat, &str)] = &[
            (ExportFormat::Ply, "nonempty_model.ply"),
            (ExportFormat::Gltf, "nonempty_model.glb"),
            (ExportFormat::Binary, "nonempty_model.safetensors"),
        ];

        for (fmt, filename) in formats {
            let out = dir.join(filename);
            let mut stage = ExportStage::new(*fmt, out.clone());
            let mut ctx = PipelineContext::default();
            ctx.trained_model = Some(test_model(5));

            stage
                .run(&mut ctx)
                .expect("test: export stage should succeed");

            assert!(out.exists(), "output file should exist for format {fmt:?}");

            let metadata = std::fs::metadata(&out).expect("test: metadata should be readable");
            assert!(
                metadata.len() > 0,
                "output file must be non-empty for format {fmt:?}, got {} bytes",
                metadata.len()
            );

            let _ = std::fs::remove_file(&out);
        }
    }

    #[test]
    fn test_image_conversion_round_trip() {
        let mut img = image::RgbImage::new(2, 2);
        img.put_pixel(0, 0, image::Rgb([255, 0, 0]));
        img.put_pixel(1, 0, image::Rgb([0, 255, 0]));
        img.put_pixel(0, 1, image::Rgb([0, 0, 255]));
        img.put_pixel(1, 1, image::Rgb([0, 0, 0]));

        let flat = rgb_to_f32(&img);
        assert_eq!(flat.len(), 2 * 2 * 3);
        let restored = f32_to_rgb(&flat, 2, 2).expect("test: conversion should succeed");
        assert_eq!(restored.get_pixel(0, 0), &image::Rgb([255, 0, 0]));
        assert_eq!(restored.get_pixel(1, 1), &image::Rgb([0, 0, 0]));

        // A too-short buffer is an error, not a silently truncated image.
        assert!(f32_to_rgb(&flat[..3], 2, 2).is_err());

        // Coverage mask: only the black pixel is background.
        let mask = coverage_mask(&img);
        assert_eq!(mask.get_pixel(0, 0), &image::Luma([255]));
        assert_eq!(mask.get_pixel(1, 1), &image::Luma([0]));
    }

    // -----------------------------------------------------------------------
    // normal_map_conditioning
    //
    // Regression coverage for: the diffusion call site rendered normal maps
    // and then handed them only as the `rendered` argument, passing nothing
    // as `normal_maps` -- so the U-Net's geometry conditioning channels were
    // zero-filled and the generated views were untied from the tracked FLAME
    // mesh. The conversion is extracted here so it is testable without the
    // trained weights the full stage requires.
    // -----------------------------------------------------------------------

    #[test]
    fn test_normal_map_conditioning_converts_every_view() {
        let mut a = image::RgbImage::new(2, 2);
        a.put_pixel(0, 0, image::Rgb([255, 0, 0]));
        let b = image::RgbImage::new(2, 2);

        let conditioning =
            normal_map_conditioning(&[a, b], 2, 2).expect("matching dimensions must convert");
        assert_eq!(
            conditioning.len(),
            2,
            "one conditioning buffer per rendered view"
        );
        for buffer in &conditioning {
            assert_eq!(buffer.len(), 2 * 2 * 3, "flat HWC RGB layout");
        }
        // Real normal data must survive, normalised to [0, 1] -- an
        // all-zeroes buffer here is exactly the bug being guarded against.
        assert!((conditioning[0][0] - 1.0).abs() < 1e-6);
        assert!(
            conditioning[0].iter().any(|v| *v > 0.0),
            "conditioning must not be zero-filled"
        );
    }

    #[test]
    fn test_normal_map_conditioning_rejects_mismatched_sizes() {
        let wrong = image::RgbImage::new(4, 4);
        let err = normal_map_conditioning(&[wrong], 2, 2)
            .expect_err("a size mismatch must not be silently conditioned on");
        let msg = err.to_string();
        assert!(msg.contains("4×4"), "must report the actual size: {msg}");
        assert!(msg.contains("2×2"), "must report the expected size: {msg}");
    }

    #[test]
    fn test_normal_map_conditioning_rejects_empty_input() {
        assert!(normal_map_conditioning(&[], 2, 2).is_err());
    }

    #[test]
    fn test_orbit_cameras_are_evenly_spaced() {
        let cams = orbit_cameras(4, 256, 256, 0.6);
        assert_eq!(cams.len(), 4);
        for cam in &cams {
            assert_eq!(cam.width, 256);
            assert_eq!(cam.height, 256);
            // Camera centre is at distance `radius` from the origin.
            let eye = -(cam.rotation.transpose() * cam.translation);
            assert!(
                (eye.norm() - 0.6).abs() < 1e-3,
                "camera must sit on the orbit, got {}",
                eye.norm()
            );
        }
    }
}
