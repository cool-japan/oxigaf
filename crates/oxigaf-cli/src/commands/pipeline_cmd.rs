//! `oxigaf pipeline` — run the reconstruction workflow one stage at a time.
//!
//! Glue over [`crate::stages`], which exposes the same work `oxigaf train`
//! does as composable [`crate::stages::PipelineStage`]s that checkpoint
//! between steps.
//!
//! | Subcommand | Stage |
//! |------------|-------|
//! | `plan`    | describes the whole sequence, runs nothing |
//! | `track`   | [`crate::stages::TrackingStage`] |
//! | `diffuse` | [`crate::stages::DiffusionStage`] |
//! | `export`  | [`crate::stages::ExportStage`] |
//! | `status`  | reads the [`crate::stages::CheckpointData`] a run left behind |
//!
//! # Why there is no `pipeline train`
//!
//! [`crate::stages::TrainingStage`] needs a [`crate::stages::TrainingSetup`]:
//! a live `wgpu` device and queue, the resolved training and rasterizer
//! configs, and an initialised Gaussian model. The code that assembles all
//! five lives inside `crate::pipeline::run_reconstruction` and its private
//! `request_gpu_device`, so `oxigaf train` is the shipped end-to-end driver
//! and this family covers the stages that can stand alone. Re-deriving that
//! setup here would be a second, divergent copy of the trainer bootstrap
//! rather than glue. See the crate's followups.
//!
//! # External assets
//!
//! `track` consumes *already tracked* FLAME parameters, and `diffuse`
//! requires trained multi-view diffusion weights: neither a landmark
//! detector nor those weights ship with OxiGAF. Both stages say exactly what
//! is missing instead of emitting placeholder data, and this family passes
//! that message through unchanged.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Args, Subcommand, ValueEnum, ValueHint};
use serde_json::{json, Value};

use crate::commands::image_io::input_invalid;
use crate::commands::{emit, prepare_output, CmdContext};
use crate::stages::{
    DiffusionStage, ExportFormat as StageExportFormat, ExportStage, PipelineContext,
    PipelineExecutor, PipelineStage, TrackingStage, TrainingStage,
};

/// `oxigaf pipeline <command>`.
#[derive(Debug, Args)]
pub struct PipelineArgs {
    #[command(subcommand)]
    pub command: PipelineCommand,
}

/// Staged-pipeline subcommands.
#[derive(Debug, Subcommand)]
pub enum PipelineCommand {
    /// Describe the stage sequence and what each stage needs.
    Plan,

    /// Resolve a run's FLAME parameter sequence and write its manifest.
    Track(TrackArgs),

    /// Generate multi-view pseudo ground truth from the tracked geometry.
    Diffuse(DiffuseArgs),

    /// Run the export stage on an existing model.
    Export(StageExportArgs),

    /// Report the stage checkpoints a previous run left in a directory.
    Status(StatusArgs),
}

/// Export encoding for `oxigaf pipeline export`.
#[derive(Debug, Clone, Copy, ValueEnum, Default, PartialEq, Eq)]
pub enum StageFormat {
    /// ASCII PLY in the 3DGS property layout.
    #[default]
    Ply,
    /// glTF 2.0 with the `OXIGAF_gaussians` extension.
    Gltf,
    /// safetensors.
    Binary,
}

impl From<StageFormat> for StageExportFormat {
    fn from(value: StageFormat) -> Self {
        match value {
            StageFormat::Ply => StageExportFormat::Ply,
            StageFormat::Gltf => StageExportFormat::Gltf,
            StageFormat::Binary => StageExportFormat::Binary,
        }
    }
}

/// Arguments for `oxigaf pipeline track`.
#[derive(Debug, Args)]
pub struct TrackArgs {
    /// FLAME sequence JSON, or a directory containing one.
    ///
    /// Raw footage is rejected: fitting FLAME to video needs a
    /// facial-landmark detector, which OxiGAF does not bundle.
    #[arg(short, long, value_hint = ValueHint::AnyPath)]
    pub input: PathBuf,

    /// Where to write the tracking manifest.
    #[arg(short, long, value_hint = ValueHint::FilePath)]
    pub output: PathBuf,

    /// Directory for the per-stage checkpoint JSON.
    #[arg(long, value_hint = ValueHint::DirPath)]
    pub checkpoint_dir: Option<PathBuf>,

    /// Overwrite the manifest if it already exists.
    #[arg(long)]
    pub force: bool,
}

/// Arguments for `oxigaf pipeline diffuse`.
#[derive(Debug, Args)]
pub struct DiffuseArgs {
    /// FLAME sequence JSON, or a directory containing one.
    #[arg(short, long, value_hint = ValueHint::AnyPath)]
    pub input: PathBuf,

    /// Directory holding the multi-view diffusion safetensors weights.
    #[arg(long, value_hint = ValueHint::DirPath)]
    pub weights: PathBuf,

    /// Directory holding the converted FLAME model (`.npy` files).
    #[arg(long, value_hint = ValueHint::DirPath)]
    pub flame_model: PathBuf,

    /// Directory to write the generated views and masks into.
    #[arg(short, long, value_hint = ValueHint::DirPath)]
    pub output: PathBuf,

    /// Number of views to generate; must match the diffusion model's config.
    #[arg(long, default_value = "4")]
    pub views: usize,

    /// Square resolution; must match the diffusion model's config.
    #[arg(long, default_value = "256")]
    pub resolution: u32,

    /// Distance of the conditioning orbit cameras from the origin, in metres.
    #[arg(long, default_value = "0.6")]
    pub orbit_radius: f32,

    /// Directory for the per-stage checkpoint JSON.
    #[arg(long, value_hint = ValueHint::DirPath)]
    pub checkpoint_dir: Option<PathBuf>,
}

/// Arguments for `oxigaf pipeline export`.
#[derive(Debug, Args)]
pub struct StageExportArgs {
    /// Model to export (`.ply`, `.safetensors` or `.json` checkpoint).
    #[arg(short, long, value_hint = ValueHint::FilePath)]
    pub model: PathBuf,

    /// Output file.
    #[arg(short, long, value_hint = ValueHint::FilePath)]
    pub output: PathBuf,

    /// Export encoding.
    #[arg(long, value_enum, default_value = "ply")]
    pub format: StageFormat,

    /// Directory for the per-stage checkpoint JSON.
    #[arg(long, value_hint = ValueHint::DirPath)]
    pub checkpoint_dir: Option<PathBuf>,

    /// Overwrite the output file if it already exists.
    #[arg(long)]
    pub force: bool,
}

/// Arguments for `oxigaf pipeline status`.
#[derive(Debug, Args)]
pub struct StatusArgs {
    /// Directory the stages wrote their checkpoints into.
    #[arg(long, value_hint = ValueHint::DirPath)]
    pub checkpoint_dir: PathBuf,

    /// Report only this stage instead of every known one.
    #[arg(long)]
    pub stage: Option<String>,
}

/// The stage names [`PipelineStage::name`] reports, in execution order.
///
/// Kept as data so `plan` and `status` agree on both the order and the
/// spelling of the checkpoint files (`stage_<name>.json`).
const STAGE_ORDER: [&str; 4] = ["Tracking", "Diffusion", "Training", "Export"];

/// Run the `pipeline` family.
///
/// # Errors
///
/// Propagates missing tracking input, missing diffusion weights, unreadable
/// models and checkpoint I/O failures.
pub fn run(args: PipelineArgs, ctx: CmdContext) -> Result<()> {
    match args.command {
        PipelineCommand::Plan => cmd_plan(&ctx),
        PipelineCommand::Track(track_args) => cmd_track(track_args, &ctx),
        PipelineCommand::Diffuse(diffuse_args) => cmd_diffuse(diffuse_args, &ctx),
        PipelineCommand::Export(export_args) => cmd_export(export_args, &ctx),
        PipelineCommand::Status(status_args) => cmd_status(status_args, &ctx),
    }
}

// ---------------------------------------------------------------------------
// pipeline plan
// ---------------------------------------------------------------------------

/// One row of the plan: the stage, what it needs, and how to drive it.
///
/// `name` is owned because [`PipelineStage::name`] borrows from the stage,
/// and the stages this is built from are dropped when `plan_rows` returns.
struct PlanRow {
    name: String,
    requires: &'static str,
    command: &'static str,
}

/// The plan, built from freshly constructed stages so the names come from
/// [`PipelineStage::name`] rather than from a hand-written list that could
/// drift away from the code.
fn plan_rows() -> Vec<PlanRow> {
    let tracking = TrackingStage::new(PathBuf::new(), PathBuf::new());
    let diffusion = DiffusionStage::new(4, (256, 256));
    let training = TrainingStage::new(1);
    let export = ExportStage::new(StageExportFormat::Ply, PathBuf::new());

    vec![
        PlanRow {
            name: tracking.name().to_string(),
            requires: "a FLAME parameter sequence JSON (no landmark detector ships with OxiGAF)",
            command: "oxigaf pipeline track",
        },
        PlanRow {
            name: diffusion.name().to_string(),
            requires: "the tracked sequence, a converted FLAME model, and diffusion weights",
            command: "oxigaf pipeline diffuse",
        },
        PlanRow {
            name: training.name().to_string(),
            requires: "the generated views plus a GPU device, queue and initialised Gaussians",
            command: "oxigaf train",
        },
        PlanRow {
            name: export.name().to_string(),
            requires: "a trained Gaussian model",
            command: "oxigaf pipeline export",
        },
    ]
}

fn cmd_plan(ctx: &CmdContext) -> Result<()> {
    let rows = plan_rows();
    emit(
        ctx,
        "pipeline plan",
        json!({
            "stages": rows
                .iter()
                .enumerate()
                .map(|(index, row)| json!({
                    "order": index + 1,
                    "name": row.name,
                    "requires": row.requires,
                    "command": row.command,
                }))
                .collect::<Vec<_>>(),
        }),
        &[],
        || {
            println!("Reconstruction pipeline stages");
            println!("------------------------------");
            for (index, row) in rows.iter().enumerate() {
                println!("{}. {}", index + 1, row.name);
                println!("   needs:  {}", row.requires);
                println!("   run:    {}", row.command);
            }
        },
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Shared execution helper
// ---------------------------------------------------------------------------

/// Run one stage through [`PipelineExecutor`] so checkpointing and progress
/// behave exactly as they do in a full pipeline run.
fn execute(
    stage: Box<dyn PipelineStage>,
    checkpoint_dir: Option<&Path>,
    ctx: &CmdContext,
) -> Result<PipelineContext> {
    let mut pipeline_ctx = PipelineContext::new();
    if let Some(dir) = checkpoint_dir {
        pipeline_ctx = pipeline_ctx.with_checkpoint_dir(dir.to_path_buf());
    }

    let mut executor = PipelineExecutor::new();
    executor.add_stage(stage);
    // The executor's own bar goes to stderr, but it is still pointless under
    // `--json` and under `-q`, where nothing else is printed either.
    executor.show_progress(ctx.human() && ctx.verbosity.show_progress());
    executor.execute(pipeline_ctx)
}

/// The metrics a stage recorded, as a JSON object with stable key order.
fn metrics_json(pipeline_ctx: &PipelineContext) -> Value {
    let mut sorted: Vec<(&String, &f32)> = pipeline_ctx.metrics.iter().collect();
    sorted.sort_by(|a, b| a.0.cmp(b.0));
    let mut map = serde_json::Map::new();
    for (key, value) in sorted {
        map.insert(
            key.clone(),
            if value.is_finite() {
                json!(value)
            } else {
                Value::Null
            },
        );
    }
    Value::Object(map)
}

// ---------------------------------------------------------------------------
// pipeline track
// ---------------------------------------------------------------------------

fn cmd_track(args: TrackArgs, ctx: &CmdContext) -> Result<()> {
    if !args.input.exists() {
        return Err(input_invalid(&args.input, "does not exist"));
    }

    if !prepare_output(ctx, &args.output, args.force)? {
        emit(
            ctx,
            "pipeline track",
            json!({
                "dry_run": true,
                "input": args.input.display().to_string(),
                "would_create": [args.output.display().to_string()],
            }),
            &[],
            || println!("Would write tracking manifest: {}", args.output.display()),
        );
        return Ok(());
    }

    let stage = TrackingStage::new(args.input.clone(), args.output.clone());
    let pipeline_ctx = execute(Box::new(stage), args.checkpoint_dir.as_deref(), ctx)?;

    let num_frames = pipeline_ctx
        .flame_sequence
        .as_ref()
        .map(|sequence| sequence.num_frames())
        .unwrap_or(0);

    emit(
        ctx,
        "pipeline track",
        json!({
            "input": args.input.display().to_string(),
            "manifest": args.output.display().to_string(),
            "num_frames": num_frames,
            "metrics": metrics_json(&pipeline_ctx),
        }),
        &[("manifest", args.output.as_path())],
        || {
            println!(
                "Resolved {num_frames} tracked frame(s); manifest written to {}",
                args.output.display()
            );
        },
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// pipeline diffuse
// ---------------------------------------------------------------------------

fn cmd_diffuse(args: DiffuseArgs, ctx: &CmdContext) -> Result<()> {
    if args.views == 0 {
        anyhow::bail!("--views must be at least 1");
    }
    if args.resolution == 0 {
        anyhow::bail!("--resolution must be at least 1");
    }
    if !args.orbit_radius.is_finite() || args.orbit_radius <= 0.0 {
        anyhow::bail!(
            "--orbit-radius must be a finite value above 0 (got {})",
            args.orbit_radius
        );
    }
    if !args.input.exists() {
        return Err(input_invalid(&args.input, "does not exist"));
    }
    if !args.weights.is_dir() {
        return Err(input_invalid(
            &args.weights,
            "is not a directory of diffusion weights; `oxigaf setup` downloads them",
        ));
    }
    if !args.flame_model.is_dir() {
        return Err(input_invalid(
            &args.flame_model,
            "is not a converted FLAME model directory; run `oxigaf convert` first",
        ));
    }

    if ctx.dry_run {
        emit(
            ctx,
            "pipeline diffuse",
            json!({
                "dry_run": true,
                "views": args.views,
                "resolution": args.resolution,
                "would_create": [args.output.display().to_string()],
            }),
            &[],
            || {
                println!(
                    "Would generate {} view(s) at {}×{} into {}",
                    args.views,
                    args.resolution,
                    args.resolution,
                    args.output.display()
                );
            },
        );
        return Ok(());
    }

    // Tracking has to run first: `DiffusionStage` conditions on the FLAME
    // sequence and refuses to run without it.
    let tracking_manifest = args.output.join("tracking.json");
    std::fs::create_dir_all(&args.output).with_context(|| {
        format!(
            "Failed to create output directory: {}",
            args.output.display()
        )
    })?;

    let mut pipeline_ctx = PipelineContext::new();
    if let Some(ref dir) = args.checkpoint_dir {
        pipeline_ctx = pipeline_ctx.with_checkpoint_dir(dir.clone());
    }

    let mut executor = PipelineExecutor::new();
    executor.add_stage(Box::new(TrackingStage::new(
        args.input.clone(),
        tracking_manifest.clone(),
    )));
    executor.add_stage(Box::new(
        DiffusionStage::new(args.views, (args.resolution, args.resolution))
            .with_weights(args.weights.clone())
            .with_flame_model(args.flame_model.clone())
            .with_orbit_radius(args.orbit_radius),
    ));
    executor.show_progress(ctx.human() && ctx.verbosity.show_progress());
    let pipeline_ctx = executor.execute(pipeline_ctx)?;

    let mut written: Vec<String> = Vec::new();
    for (index, image) in pipeline_ctx.generated_images.iter().enumerate() {
        let path = args.output.join(format!("view_{index:03}.png"));
        image
            .save(&path)
            .with_context(|| format!("Failed to write generated view: {}", path.display()))?;
        written.push(path.display().to_string());
    }
    for (index, mask) in pipeline_ctx.generated_masks.iter().enumerate() {
        let path = args.output.join(format!("mask_{index:03}.png"));
        mask.save(&path)
            .with_context(|| format!("Failed to write coverage mask: {}", path.display()))?;
        written.push(path.display().to_string());
    }

    emit(
        ctx,
        "pipeline diffuse",
        json!({
            "input": args.input.display().to_string(),
            "weights": args.weights.display().to_string(),
            "flame_model": args.flame_model.display().to_string(),
            "output_dir": args.output.display().to_string(),
            "views": pipeline_ctx.generated_images.len(),
            "masks": pipeline_ctx.generated_masks.len(),
            "resolution": args.resolution,
            "tracking_manifest": tracking_manifest.display().to_string(),
            "written": written,
            "metrics": metrics_json(&pipeline_ctx),
        }),
        &[("views", args.output.as_path())],
        || {
            println!(
                "Generated {} view(s) and {} mask(s) into {}",
                pipeline_ctx.generated_images.len(),
                pipeline_ctx.generated_masks.len(),
                args.output.display()
            );
        },
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// pipeline export
// ---------------------------------------------------------------------------

fn cmd_export(args: StageExportArgs, ctx: &CmdContext) -> Result<()> {
    if !args.model.is_file() {
        return Err(input_invalid(&args.model, "not an existing model file"));
    }

    if !prepare_output(ctx, &args.output, args.force)? {
        emit(
            ctx,
            "pipeline export",
            json!({
                "dry_run": true,
                "model": args.model.display().to_string(),
                "would_create": [args.output.display().to_string()],
            }),
            &[],
            || println!("Would export to {}", args.output.display()),
        );
        return Ok(());
    }

    let model = crate::export::load_model(&args.model)
        .with_context(|| format!("Failed to load model: {}", args.model.display()))?;
    let num_gaussians = model.len();

    let mut pipeline_ctx = PipelineContext::new();
    if let Some(ref dir) = args.checkpoint_dir {
        pipeline_ctx = pipeline_ctx.with_checkpoint_dir(dir.clone());
    }
    // `ExportStage` reads its input from the context, which is how it is fed
    // in a full run; seeding it with a loaded model is what lets the stage be
    // exercised on its own.
    pipeline_ctx.trained_model = Some(model);

    let mut executor = PipelineExecutor::new();
    executor.add_stage(Box::new(ExportStage::new(
        args.format.into(),
        args.output.clone(),
    )));
    executor.show_progress(ctx.human() && ctx.verbosity.show_progress());
    let pipeline_ctx = executor.execute(pipeline_ctx)?;

    let file_size_bytes = std::fs::metadata(&args.output)
        .map(|m| m.len())
        .unwrap_or(0);

    emit(
        ctx,
        "pipeline export",
        json!({
            "model": args.model.display().to_string(),
            "output": args.output.display().to_string(),
            "format": format!("{:?}", args.format).to_lowercase(),
            "num_gaussians": num_gaussians,
            "file_size_bytes": file_size_bytes,
            "metrics": metrics_json(&pipeline_ctx),
        }),
        &[("export", args.output.as_path())],
        || {
            println!(
                "Exported {num_gaussians} Gaussians to {} ({file_size_bytes} bytes)",
                args.output.display()
            );
        },
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// pipeline status
// ---------------------------------------------------------------------------

fn cmd_status(args: StatusArgs, ctx: &CmdContext) -> Result<()> {
    if !args.checkpoint_dir.is_dir() {
        return Err(input_invalid(
            &args.checkpoint_dir,
            "not an existing directory",
        ));
    }

    let wanted: Vec<String> = match args.stage {
        Some(ref name) => vec![name.clone()],
        None => STAGE_ORDER.iter().map(|name| (*name).to_string()).collect(),
    };

    let mut entries = Vec::with_capacity(wanted.len());
    let mut present = 0usize;
    for name in &wanted {
        // A stage that never ran simply has no checkpoint file; that is a
        // reportable state, not an error.
        match PipelineContext::load_checkpoint(&args.checkpoint_dir, name) {
            Ok(data) => {
                present += 1;
                let mut metrics: Vec<(&String, &f32)> = data.metrics.iter().collect();
                metrics.sort_by(|a, b| a.0.cmp(b.0));
                entries.push(json!({
                    "stage": data.stage_name,
                    "present": true,
                    "current_stage": data.current_stage,
                    "total_stages": data.total_stages,
                    "has_flame_sequence": data.has_flame_sequence,
                    "num_generated_images": data.num_generated_images,
                    "has_trained_model": data.has_trained_model,
                    "metrics": metrics
                        .iter()
                        .map(|(key, value)| json!({ "name": key, "value": value }))
                        .collect::<Vec<_>>(),
                }));
            }
            Err(_) => entries.push(json!({ "stage": name, "present": false })),
        }
    }

    emit(
        ctx,
        "pipeline status",
        json!({
            "checkpoint_dir": args.checkpoint_dir.display().to_string(),
            "checkpoints_found": present,
            "stages": entries,
        }),
        &[],
        || {
            println!("Stage checkpoints in {}", args.checkpoint_dir.display());
            for entry in &entries {
                let stage = entry
                    .get("stage")
                    .and_then(Value::as_str)
                    .unwrap_or("<unknown>");
                if entry.get("present").and_then(Value::as_bool) == Some(true) {
                    let images = entry
                        .get("num_generated_images")
                        .and_then(Value::as_u64)
                        .unwrap_or(0);
                    let model = entry
                        .get("has_trained_model")
                        .and_then(Value::as_bool)
                        .unwrap_or(false);
                    println!("  [done]    {stage}  ({images} view(s), model: {model})");
                } else {
                    println!("  [missing] {stage}");
                }
            }
        },
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::EXIT_IO_ERROR;
    use crate::verbosity::Verbosity;

    fn ctx(dry_run: bool) -> CmdContext {
        CmdContext::new(Verbosity::Quiet, true, dry_run)
    }

    /// The plan's stage names must come from the stages themselves, and must
    /// stay in step with the checkpoint file names `status` looks for.
    #[test]
    fn plan_names_match_the_checkpoint_order() {
        let rows = plan_rows();
        let names: Vec<&str> = rows.iter().map(|row| row.name.as_str()).collect();
        assert_eq!(names, STAGE_ORDER.to_vec());
    }

    #[test]
    fn plan_runs_without_touching_the_filesystem() {
        assert!(cmd_plan(&ctx(false)).is_ok());
    }

    /// Tracking a real sequence must write the manifest and report the frame
    /// count — the stage is the one part of the pipeline that needs no GPU
    /// and no external weights.
    #[test]
    fn track_resolves_a_flame_sequence() {
        let dir = std::env::temp_dir().join("oxigaf_pipeline_track");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("temp dir");
        let params = dir.join("flame_params.json");
        std::fs::write(
            &params,
            r#"{"fps": 24.0, "frames": [
                {"shape": [0.0], "expression": [0.0], "pose": [0.0]},
                {"shape": [0.0], "expression": [0.0], "pose": [0.0]}
            ]}"#,
        )
        .expect("temp sequence write");

        let manifest = dir.join("tracking.json");
        let args = TrackArgs {
            input: params.clone(),
            output: manifest.clone(),
            checkpoint_dir: Some(dir.join("checkpoints")),
            force: true,
        };
        let outcome = cmd_track(args, &ctx(false));
        assert!(outcome.is_ok(), "tracking failed: {:?}", outcome.err());
        assert!(manifest.is_file(), "the manifest must be written");
        assert!(
            dir.join("checkpoints")
                .join("stage_Tracking.json")
                .is_file(),
            "the stage checkpoint must be written"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Raw footage has no tracker, and the refusal has to name the input
    /// rather than fail somewhere inside the sequence loader.
    #[test]
    fn track_refuses_raw_footage() {
        let dir = std::env::temp_dir().join("oxigaf_pipeline_raw");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("temp dir");
        let video = dir.join("clip.mp4");
        std::fs::write(&video, b"not really a video").expect("temp file write");

        let args = TrackArgs {
            input: video,
            output: dir.join("tracking.json"),
            checkpoint_dir: None,
            force: true,
        };
        assert!(cmd_track(args, &ctx(false)).is_err());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_diffusion_weights_are_an_input_error() {
        let dir = std::env::temp_dir().join("oxigaf_pipeline_weights");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("temp dir");
        let params = dir.join("flame_params.json");
        std::fs::write(&params, "{\"fps\": 24.0, \"frames\": []}").expect("temp write");

        let args = DiffuseArgs {
            input: params,
            weights: dir.join("no-such-weights"),
            flame_model: dir.clone(),
            output: dir.join("out"),
            views: 4,
            resolution: 256,
            orbit_radius: 0.6,
            checkpoint_dir: None,
        };
        let err = cmd_diffuse(args, &ctx(false)).expect_err("missing weights must not run");
        assert_eq!(
            crate::commands::runtime::to_cli_error(err).exit_code(),
            EXIT_IO_ERROR
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn status_reports_absent_stages_without_failing() {
        let dir = std::env::temp_dir().join("oxigaf_pipeline_status");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("temp dir");

        let args = StatusArgs {
            checkpoint_dir: dir.clone(),
            stage: None,
        };
        assert!(cmd_status(args, &ctx(false)).is_ok());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn stage_format_maps_onto_the_library_enum() {
        assert!(matches!(
            StageExportFormat::from(StageFormat::Binary),
            StageExportFormat::Binary
        ));
        assert!(matches!(
            StageExportFormat::from(StageFormat::Gltf),
            StageExportFormat::Gltf
        ));
    }
}
