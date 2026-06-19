//! Pipeline stage abstractions for end-to-end workflows
//!
//! This module provides a flexible framework for orchestrating multi-stage processing pipelines
//! with progress tracking, checkpointing, and error recovery.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};

use oxigaf_flame::sequence::FlameSequence;

// Placeholder for Gaussian model in pipeline context
// NOTE: This is a simplified placeholder for CLI pipeline scaffolding.
// For actual training, use `oxigaf_render::gaussian::GaussianModel`.
// This placeholder allows the pipeline framework to be tested without
// implementing full end-to-end training integration.
#[derive(Debug)]
pub struct GaussianModel {
    #[allow(dead_code)]
    num_gaussians: usize,
}

impl GaussianModel {
    pub fn new(num_gaussians: usize) -> Self {
        Self { num_gaussians }
    }
}

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

/// Tracking stage: Extract FLAME parameters from video
pub struct TrackingStage {
    video_path: PathBuf,
    #[allow(dead_code)]
    output_path: PathBuf,
    progress: f32,
}

impl TrackingStage {
    /// Create a new tracking stage
    pub fn new(video_path: PathBuf, output_path: PathBuf) -> Self {
        Self {
            video_path,
            output_path,
            progress: 0.0,
        }
    }
}

impl PipelineStage for TrackingStage {
    fn name(&self) -> &str {
        "Tracking"
    }

    fn run(&mut self, ctx: &mut PipelineContext) -> Result<()> {
        tracing::info!(
            "Starting FLAME tracking from video: {}",
            self.video_path.display()
        );

        // NOTE: Placeholder implementation for pipeline scaffolding.
        // Production implementation would integrate with FLAME tracking algorithm.
        self.progress = 0.5;

        // Placeholder: Create a simple sequence
        let _sequence = FlameSequence::from_memory(vec![], Some(30.0));

        self.progress = 1.0;

        // Update context
        ctx.video_path = Some(self.video_path.clone());
        ctx.metrics.insert("tracking_fps".to_string(), 30.0);

        Ok(())
    }

    fn progress(&self) -> f32 {
        self.progress
    }
}

/// Diffusion stage: Generate multi-view images from FLAME parameters
pub struct DiffusionStage {
    num_views: usize,
    resolution: (u32, u32),
    progress: f32,
}

impl DiffusionStage {
    /// Create a new diffusion stage
    pub fn new(num_views: usize, resolution: (u32, u32)) -> Self {
        Self {
            num_views,
            resolution,
            progress: 0.0,
        }
    }
}

impl PipelineStage for DiffusionStage {
    fn name(&self) -> &str {
        "Diffusion"
    }

    fn run(&mut self, ctx: &mut PipelineContext) -> Result<()> {
        tracing::info!(
            "Generating {} views at {:?}",
            self.num_views,
            self.resolution
        );

        if ctx.flame_sequence.is_none() {
            anyhow::bail!("No FLAME sequence available for diffusion");
        }

        // NOTE: Placeholder implementation for pipeline scaffolding.
        // Production implementation would integrate with oxigaf-diffusion.
        self.progress = 0.5;

        let (width, height) = self.resolution;
        for i in 0..self.num_views {
            let img = image::RgbImage::new(width, height);
            ctx.generated_images.push(img);

            let mask = image::GrayImage::new(width, height);
            ctx.generated_masks.push(mask);

            self.progress = 0.5 + 0.5 * (i as f32 / self.num_views as f32);
        }

        ctx.metrics
            .insert("num_views".to_string(), self.num_views as f32);

        Ok(())
    }

    fn progress(&self) -> f32 {
        self.progress
    }
}

/// Training stage: Train 3D Gaussian Splatting model
pub struct TrainingStage {
    num_iterations: usize,
    current_iteration: usize,
    start_time: Option<Instant>,
}

impl TrainingStage {
    /// Create a new training stage
    pub fn new(num_iterations: usize) -> Self {
        Self {
            num_iterations,
            current_iteration: 0,
            start_time: None,
        }
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

        self.start_time = Some(Instant::now());

        // NOTE: Placeholder implementation for pipeline scaffolding.
        // Production implementation would integrate with oxigaf-trainer::Trainer.
        for i in 0..self.num_iterations {
            self.current_iteration = i + 1;

            // Simulate training step
            std::thread::sleep(std::time::Duration::from_millis(1));

            if i % 100 == 0 {
                let loss = 1.0 / (i as f32 + 1.0);
                ctx.metrics.insert(format!("loss_iter_{}", i), loss);
            }
        }

        // Create placeholder model for pipeline scaffolding
        let model = GaussianModel::new(1000);
        ctx.trained_model = Some(model);

        ctx.metrics.insert("final_loss".to_string(), 0.01);
        ctx.metrics
            .insert("iterations".to_string(), self.num_iterations as f32);

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
                let remaining = (self.num_iterations - self.current_iteration) as f64 * per_iter;
                return Some(remaining);
            }
        }
        None
    }
}

/// Export stage: Export trained model to various formats
pub struct ExportStage {
    format: ExportFormat,
    output_path: PathBuf,
}

/// Supported export formats
#[derive(Debug, Clone, Copy)]
pub enum ExportFormat {
    /// PLY point cloud
    Ply,
    /// glTF with Gaussian extension
    Gltf,
    /// Custom binary format
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

        if let Some(model) = ctx.trained_model.as_ref() {
            std::fs::create_dir_all(
                self.output_path
                    .parent()
                    .unwrap_or_else(|| std::path::Path::new(".")),
            )
            .context("Failed to create output directory")?;

            match self.format {
                ExportFormat::Ply => {
                    // Write a standards-compliant ASCII PLY file whose property layout
                    // matches the 3D Gaussian Splatting convention understood by
                    // `crate::export::load_ply`.  Each Gaussian is represented as a
                    // zero-initialised vertex with the full property set (position,
                    // normals, SH DC, opacity, scale, rotation) so that a round-trip
                    // through `load_ply` reconstructs the correct vertex count.
                    use std::io::Write as _;
                    let file = std::fs::File::create(&self.output_path)
                        .context("PLY export: failed to create file")?;
                    let mut w = std::io::BufWriter::new(file);
                    writeln!(w, "ply").context("PLY write")?;
                    writeln!(w, "format ascii 1.0").context("PLY write")?;
                    writeln!(w, "comment oxigaf pipeline export").context("PLY write")?;
                    writeln!(w, "element vertex {}", model.num_gaussians).context("PLY write")?;
                    for prop in ["x", "y", "z", "nx", "ny", "nz"] {
                        writeln!(w, "property float {prop}").context("PLY write")?;
                    }
                    // SH DC (degree-0 coefficients for R/G/B)
                    for c in 0..3u32 {
                        writeln!(w, "property float f_dc_{c}").context("PLY write")?;
                    }
                    writeln!(w, "property float opacity").context("PLY write")?;
                    for i in 0..3u32 {
                        writeln!(w, "property float scale_{i}").context("PLY write")?;
                    }
                    // Rotation stored as (w,x,y,z) in PLY convention.
                    for i in 0..4u32 {
                        writeln!(w, "property float rot_{i}").context("PLY write")?;
                    }
                    writeln!(w, "end_header").context("PLY write")?;
                    // Zero-initialised vertex data — one row per Gaussian.
                    // 13 float columns: x y z nx ny nz f_dc_0..2 opacity scale_0..2 rot_0..3
                    for _ in 0..model.num_gaussians {
                        writeln!(w, "0 0 0 0 0 0 0 0 0 0 -5 -5 -5 1 0 0 0").context("PLY write")?;
                    }
                    w.flush().context("PLY export: flush failed")?;
                }
                ExportFormat::Gltf => {
                    // Write a minimal valid glTF 2.0 JSON skeleton that records the
                    // Gaussian count.  A real implementation would delegate to
                    // `crate::export::export_gltf`.
                    let json = serde_json::to_string_pretty(&serde_json::json!({
                        "asset": { "version": "2.0", "generator": "oxigaf-pipeline" },
                        "extensions": {
                            "OXIGAF_gaussians": {
                                "num_gaussians": model.num_gaussians
                            }
                        }
                    }))
                    .context("glTF metadata serialization failed")?;
                    std::fs::write(&self.output_path, json).context("glTF export failed")?;
                }
                ExportFormat::Binary => {
                    // Binary format: write a JSON metadata placeholder so the file
                    // is at least valid and inspectable.
                    let json = serde_json::to_string_pretty(&serde_json::json!({
                        "format": "oxigaf_binary",
                        "num_gaussians": model.num_gaussians,
                    }))
                    .context("JSON metadata serialization failed")?;
                    std::fs::write(&self.output_path, json)
                        .context("Binary metadata write failed")?;
                }
            }

            ctx.metrics
                .insert("num_gaussians".to_string(), model.num_gaussians as f32);
        } else {
            anyhow::bail!("No trained model in context");
        }

        ctx.metrics.insert("exported".to_string(), 1.0);

        Ok(())
    }

    fn progress(&self) -> f32 {
        // Export is typically fast, so either 0 or 1
        1.0
    }
}

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
    use tempfile::TempDir;

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

    #[test]
    fn test_export_stage() {
        let temp_dir = TempDir::new().expect("test: temp dir creation should succeed");
        let output_path = temp_dir.path().join("model.ply");

        let mut stage = ExportStage::new(ExportFormat::Ply, output_path.clone());
        let mut ctx = PipelineContext::new();

        // Should fail without model
        assert!(stage.run(&mut ctx).is_err());

        // Add model and retry
        ctx.trained_model = Some(GaussianModel::new(10));
        assert!(stage.run(&mut ctx).is_ok());

        // Check file was created
        assert!(output_path.exists());
    }

    #[test]
    fn test_export_stage_writes_ply() {
        // ExportStage must produce a valid ASCII PLY that load_ply can round-trip.
        let dir = std::env::temp_dir().join("oxigaf_stage_test");
        std::fs::create_dir_all(&dir).expect("test: create temp dir");
        let out = dir.join("test_export_stage_writes_ply.ply");

        let mut stage = ExportStage::new(ExportFormat::Ply, out.clone());
        let model = GaussianModel::new(10);
        let mut ctx = PipelineContext::default();
        ctx.trained_model = Some(model);

        stage
            .run(&mut ctx)
            .expect("test: export stage should succeed");

        assert!(out.exists(), "PLY file should have been written");

        // Round-trip through load_ply and verify vertex count.
        let loaded =
            crate::export::load_ply(&out).expect("test: load_ply should parse the written file");
        assert_eq!(loaded.len(), 10, "loaded model must have 10 Gaussians");

        // Clean up
        let _ = std::fs::remove_file(&out);
    }

    /// The output file from any `ExportStage` format must be non-empty (> 0 bytes).
    ///
    /// This exercises PLY, glTF, and Binary export paths with a minimal
    /// 5-Gaussian model, asserting the output file both exists and contains
    /// at least one byte.
    #[test]
    fn test_export_stage_writes_nonempty_file() {
        let dir = std::env::temp_dir().join("oxigaf_stage_test_nonempty");
        std::fs::create_dir_all(&dir).expect("test: create temp dir");

        let formats: &[(ExportFormat, &str)] = &[
            (ExportFormat::Ply, "nonempty_model.ply"),
            (ExportFormat::Gltf, "nonempty_model.gltf"),
            (ExportFormat::Binary, "nonempty_model.bin"),
        ];

        for (fmt, filename) in formats {
            let out = dir.join(filename);
            let mut stage = ExportStage::new(*fmt, out.clone());
            let mut ctx = PipelineContext::default();
            ctx.trained_model = Some(GaussianModel::new(5));

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
}
