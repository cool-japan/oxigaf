//! OxiGAF CLI — Gaussian Avatar Reconstruction from a frame sequence.
//!
//! This binary is a thin shim over the `oxigaf_cli` library crate: every
//! *library* module lives in `src/lib.rs`, and this file only parses
//! arguments and dispatches. Do **not** re-declare one of those `mod`s
//! here — a module declared in both roots is compiled twice into two
//! unrelated types, which is exactly how forty-odd library modules ended up
//! unreachable from the binary. The single `#[cfg(test)] mod tests;` below
//! is not one of those: it exists only in this binary crate (split into
//! `main_tests.rs` to keep this file under the 2000-line policy), so it has
//! nothing in `lib.rs` to collide with.
//!
//! # Input is frames, not video
//!
//! This header used to advertise reconstruction "from monocular video".
//! [`oxigaf_cli::pipeline::collect_input_frames`] refuses a video container
//! outright — OxiGAF is pure Rust (COOLJAPAN policy) and bundles no demuxer
//! or decoder — so `--input` takes a directory of extracted frames (or a
//! single frame image). Extract them first, e.g. with
//! `ffmpeg -i clip.mp4 frames/%05d.png`.
//!
//! Pipeline subcommands:
//! * `train` (alias `reconstruct`) — end-to-end avatar reconstruction pipeline
//! * `render` — render an existing avatar from novel viewpoints
//! * `export` — export an avatar to PLY, glTF, or safetensors
//! * `convert` — convert FLAME model files (.pkl to .npy format)
//! * `benchmark` — run performance benchmarks
//! * `doctor` — check system configuration and dependencies
//! * `setup` — download and cache required model weights
//! * `cache` — manage cached assets (list, clean, verify, path)
//! * `info` / `compare` / `config` (alias `config-cmd`) / `completions`
//!
//! Tool families (handlers in [`oxigaf_cli::commands`]):
//! * `anim`, `analyze`, `batch`, `camera`, `dataset`, `inspect`, `monitor`,
//!   `perf`, `pipeline`, `preset`, `preview`, `profile`, `quality`, `report`,
//!   `runs`, `scene`, `sweep`, `training`, `video`, `workspace`

#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![warn(clippy::panic)]

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{CommandFactory, Parser};
use clap_complete::generate;

use oxigaf::render::gaussian::GaussianModel;
use oxigaf_cli::cli::{
    CacheCommands, Cli, Command, DoctorCheck, ExportFormat, ImageFormat, RenderMode,
};
use oxigaf_cli::commands::error_report::classify_error;
use oxigaf_cli::commands::final_export::FinalExport;
use oxigaf_cli::commands::gpu_probe;
use oxigaf_cli::commands::model_io::FlatScene;
use oxigaf_cli::commands::runtime::{
    available_disk_mb, check_cache, check_flame_model, default_cache_dir, doctor_runs,
    downsample_sh, get_version_info, init_file_only_logging, init_logging, peak_rss_mb,
    resolve_cache_dir, select_assets, RawModeGuard,
};
use oxigaf_cli::commands::CmdContext;
use oxigaf_cli::error::{CliError, EXIT_INTERRUPTED, EXIT_SUCCESS};
use oxigaf_cli::parallel_render::{FrameTask, ParallelRenderConfig, ParallelRenderer};
use oxigaf_cli::progress_types::BatchProgress;
use oxigaf_cli::verbosity::Verbosity;
use oxigaf_cli::{
    assets, benchmark, cache, cli, commands, compare, config, config_cmd, convert, dry_run, export,
    export_gltf, export_mesh, export_ply, export_pointcloud, info, interactive, json_output,
    output, pipeline, summary,
};

/// Cooperative-shutdown flag published by `train --interactive`.
///
/// A SIGINT sets it so the training loop's step gate can stop at a clean
/// boundary; see [`install_interrupt_handler`].
static INTERRUPT_FLAG: OnceLock<Arc<AtomicBool>> = OnceLock::new();

/// Install a SIGINT handler that leaves the terminal usable and exits 130.
///
/// Without this, Ctrl+C during `--interactive` training killed the process
/// with the terminal still in raw mode, and every interrupt reported exit
/// status 0 to the shell.
///
/// The handler runs on its own OS thread with its own current-thread runtime
/// rather than on the main runtime, so it fires even while a synchronous
/// command body occupies every worker thread.
fn install_interrupt_handler() {
    std::thread::spawn(|| {
        let runtime = match tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
        {
            Ok(runtime) => runtime,
            Err(e) => {
                tracing::warn!("Ctrl+C handling is unavailable: {e}");
                return;
            }
        };
        runtime.block_on(async {
            if tokio::signal::ctrl_c().await.is_err() {
                return;
            }
            // When a cooperative consumer is registered, ask it to stop and
            // then hand control back: the command finishes its current step,
            // writes its outputs, and reports its own exit status. A timer
            // that forced an exit here would race that clean shutdown and
            // make the status a coin flip — exactly the defect the exit-code
            // taxonomy exists to avoid. A *second* Ctrl+C aborts instead.
            if let Some(flag) = INTERRUPT_FLAG.get() {
                flag.store(true, Ordering::SeqCst);
                eprintln!(
                    "\nInterrupted — finishing the current step. Press Ctrl+C again to abort."
                );
                if tokio::signal::ctrl_c().await.is_err() {
                    return;
                }
            }
            // A guard's `Drop` cannot run on this path, so restore the
            // terminal explicitly before exiting.
            let _ = crossterm::terminal::disable_raw_mode();
            eprintln!("Interrupted (SIGINT).");
            std::process::exit(EXIT_INTERRUPTED);
        });
    });
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> ExitCode {
    // Parse CLI args first to get verbosity level
    let cli = Cli::parse();
    let verbosity = cli.verbosity();
    let dry_run = cli.dry_run;
    let json_mode = cli.json;

    // Initialize logging based on verbosity. Under --json only the file
    // sink is attached, so stdout stays a pure JSON stream.
    let log_init = match (json_mode, cli.log_file.clone()) {
        (false, log_file) => init_logging(
            verbosity,
            log_file,
            cli.log_rotation,
            cli.log_max_files,
            cli.log_format,
        ),
        (true, Some(log_file)) => init_file_only_logging(
            verbosity,
            &log_file,
            cli.log_rotation,
            cli.log_max_files,
            cli.log_format,
        ),
        (true, None) => Ok(()),
    };
    if let Err(e) = log_init {
        eprintln!("Failed to initialize logging: {}", e);
        return ExitCode::from(1);
    }

    install_interrupt_handler();

    let ctx = CmdContext::new(verbosity, json_mode, dry_run);

    let result = match cli.command {
        Command::Train(args) => cmd_train(args, verbosity, dry_run, json_mode),
        Command::Render(args) => cmd_render(args, verbosity, dry_run, json_mode),
        Command::Export(args) => cmd_export(args, verbosity, dry_run, json_mode),
        Command::Convert(args) => convert::run_convert(args, verbosity, dry_run, json_mode),
        Command::Benchmark(args) => match (dry_run, args.output.clone()) {
            (true, Some(report_path)) => {
                dry_run_writes("benchmark", &[report_path.as_path()], json_mode)
            }
            _ => benchmark::run_benchmark(args, verbosity, json_mode),
        },
        Command::Doctor(args) => cmd_doctor(args, verbosity, json_mode),
        Command::Setup(args) => cmd_setup(args, verbosity, dry_run, json_mode),
        Command::Cache { command } => cmd_cache(command, dry_run, json_mode),
        // `info` and `config` grew real `--json` renderings (`run_*_json`
        // emits the single document through `commands::emit`), so the global
        // flag is honoured rather than refused. `completions` is the one
        // handler left that cannot: a completion script is shell source.
        Command::Info(args) => {
            if json_mode {
                info::run_info_json(args)
            } else {
                info::run_info(args)
            }
        }
        Command::Compare(mut args) => {
            // `compare` predates the global flag and carries its own
            // `--format json`; map the global one onto it so
            // `oxigaf --json compare a.ply b.ply` is not silently textual.
            if json_mode && args.format == "text" {
                args.format = "json".to_string();
            }
            compare::run_compare(args)
        }
        Command::ConfigCmd { command } => {
            // `config init --output <path>` is the one writing arm; the
            // borrow is dropped before `command` is moved into the handler.
            let init_output = match (&command, dry_run) {
                (cli::ConfigCmdSubcommand::Init { output, .. }, true) => output.clone(),
                _ => None,
            };
            match (init_output, json_mode) {
                (Some(path), _) => dry_run_writes("config", &[path.as_path()], json_mode),
                (None, true) => config_cmd::run_config_cmd_json(command),
                (None, false) => config_cmd::run_config_cmd(command),
            }
        }
        Command::Completions { shell } => reject_json(
            json_mode,
            "completions",
            "A completion script is shell source, not JSON; run it without --json.",
        )
        .and_then(|()| cmd_completions(shell)),

        Command::Anim(args) => commands::anim::run(args, ctx),
        Command::Analyze(args) => commands::analyze::run(args, ctx),
        Command::Batch(args) => commands::batch::run(args, ctx),
        Command::Camera(args) => commands::camera::run(args, ctx),
        Command::Dataset(args) => commands::dataset::run(args, ctx),
        Command::Inspect(args) => commands::inspect::run(args, ctx),
        Command::Monitor(args) => commands::monitor::run(args, ctx),
        Command::Perf(args) => commands::perf::run(args, ctx),
        Command::Pipeline(args) => commands::pipeline_cmd::run(args, ctx),
        Command::Preset(args) => commands::preset::run(args, ctx),
        Command::Preview(args) => commands::preview::run(args, ctx),
        Command::Profile(args) => commands::profile::run(args, ctx),
        Command::Quality(args) => commands::quality::run(args, ctx),
        Command::Report(args) => commands::report::run(args, ctx),
        Command::Runs(args) => commands::runs::run(args, ctx),
        Command::Scene(args) => commands::scene_ops::run(args, ctx),
        Command::Sweep(args) => commands::sweep::run(args, ctx),
        Command::Training(args) => commands::training::run(args, ctx),
        Command::Video(args) => commands::video::run(args, ctx),
        Command::Workspace(args) => commands::workspace::run(args, ctx),
    };

    match result {
        Ok(()) => ExitCode::from(EXIT_SUCCESS as u8),
        Err(err) => {
            let (cli_err, detail) = classify_error(err);

            if json_mode {
                // Several commands print their result document and *then*
                // fail on what it says: `quality check` on a render below
                // its thresholds, `doctor` with a dead GPU, `cache verify`
                // with a missing asset, `setup --offline` with an uncached
                // one. Appending an error document there would put two
                // concatenated JSON values on stdout, which `jq` rejects —
                // and a scripted caller could not tell that from a broken
                // tool. The document already on stdout says what happened,
                // so the message goes to stderr and the exit status carries
                // the failure. See `commands::json_document_emitted`.
                if should_emit_error_document(json_mode, commands::json_document_emitted()) {
                    print_json(json_output::JsonOutput::error("command", detail));
                } else {
                    eprintln!("Error: {detail}");
                }
            } else {
                output::display_error(&cli_err);
                output::flush();
            }

            ExitCode::from(cli_err.exit_code() as u8)
        }
    }
}

/// Write a JSON document to stdout and record that the single document
/// `--json` promises has now been produced.
///
/// Every hand-built [`json_output::JsonOutput`] in this file goes through
/// here rather than calling `print()` directly, so a later failure knows not
/// to append a second document. Handlers under [`oxigaf_cli::commands`] get
/// the same bookkeeping from [`oxigaf_cli::commands::emit`].
fn print_json(output: json_output::JsonOutput) {
    output.print();
    commands::mark_json_document_emitted();
}

/// Whether a failure should be rendered as its own JSON document on stdout.
///
/// Only under `--json`, and only when nothing has written the result
/// document yet. A handler that printed its document and then failed on what
/// it said — `quality check` below threshold, `doctor` with a dead GPU,
/// `cache verify` with a missing asset — has already produced the one value
/// `--json` promises; a second one would make stdout invalid JSON.
///
/// Split out as a pure function so the rule is testable without touching the
/// process-wide flag it is normally fed from.
#[must_use]
fn should_emit_error_document(json_mode: bool, already_emitted: bool) -> bool {
    json_mode && !already_emitted
}

/// Refuse `--json` for a subcommand whose handler cannot emit JSON.
///
/// The global flag promises that stdout carries "only valid JSON" (see
/// [`cli::Cli::json`]), and every handler that can honour it does. `info` and
/// `config` used to be refused here alongside `completions`; both have since
/// grown a real JSON rendering (`run_info_json`, `run_config_cmd_json`) and
/// are dispatched to it instead. `completions` remains: a completion script
/// is shell source, and there is no JSON form of it to produce.
///
/// Letting such a handler run under `--json` hands a scripted caller non-JSON
/// *with exit status 0*, so a `| jq` pipeline fails with no way to tell a
/// broken tool from a broken model. Refusing loudly is the honest path, and
/// it is what `cache clean --json` already does; `alternative` names the
/// command that does have a machine-readable form.
///
/// # Errors
///
/// Returns an error whenever `json_mode` is set.
fn reject_json(json_mode: bool, subcommand: &str, alternative: &str) -> Result<()> {
    if json_mode {
        anyhow::bail!("`oxigaf {subcommand}` has no JSON output. {alternative}");
    }
    Ok(())
}

/// Honour the global `--dry-run` for a handler whose only side effect is the
/// file it was asked to write.
///
/// `--dry-run` is global and documented as reporting "what would be done
/// without executing any modifications". `train`, `render`, `export`,
/// `convert`, `setup` and `cache clean` each implement their own validation
/// phase; `benchmark --output` and `config init --output` had none and so
/// executed and wrote the file anyway. Neither has a phase worth reusing, so
/// the honest implementation is to report the write and stop before making
/// it. The remaining subcommands (`doctor`, `info`, `compare`, `completions`,
/// `cache list/verify/path`) modify nothing, so running them *is* their dry
/// run and they are left alone.
///
/// # Errors
///
/// Never fails today; returns `Result` so the dispatch arms stay uniform.
fn dry_run_writes(command: &str, outputs: &[&Path], json_mode: bool) -> Result<()> {
    let mut report = dry_run::DryRunReport::new();
    for path in outputs {
        if path.exists() {
            report.add_modify(path.display().to_string());
        } else {
            report.add_create(path.display().to_string());
        }
    }
    if json_mode {
        print_json(json_output::JsonOutput::success(
            command,
            serde_json::json!({
                "dry_run": true,
                "would_create": report.would_create,
                "would_modify": report.would_modify,
            }),
        ));
    } else {
        report.print_report();
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// train (alias: reconstruct)
// ---------------------------------------------------------------------------

fn cmd_train(
    args: cli::TrainArgs,
    verbosity: Verbosity,
    dry_run: bool,
    json_mode: bool,
) -> Result<()> {
    let start = Instant::now();
    tracing::info!(?args.input, ?args.output, "Starting training pipeline");

    // Flags the training pipeline does not consume yet. Warn loudly rather
    // than accepting them in silence — a run started with `--seed 7` that
    // is not actually reproducible is worse than one that says so. The
    // messages go to the log sink *and* stderr; see
    // [`commands::flag_warnings`] for why one channel is not enough.
    let warnings = commands::flag_warnings::train(&args);
    commands::flag_warnings::emit(&warnings);

    // 1. Load project configuration with hierarchical loading
    // Priority: CLI args > preset > env vars > project config > user config > defaults
    let mut project_config = config::load_hierarchical_config(Some(&args.config), None)?;
    tracing::info!("Configuration loaded with hierarchical priority");

    // Apply the named training preset *before* the individual CLI
    // overrides, so an explicit `--max-iterations` still wins over the
    // profile it was combined with.
    if let Some(ref profile) = args.profile {
        let preset = commands::preset::resolve(profile, &[])?;
        preset.apply_to(&mut project_config);
        tracing::info!(
            "Applied training preset '{}': {}",
            profile,
            preset.description
        );
    }

    // Apply CLI overrides (highest priority)
    if let Some(max_iter) = args.max_iterations {
        project_config.training.total_iterations = max_iter;
        tracing::debug!("CLI override: total_iterations = {}", max_iter);
    }
    if let Some(ckpt_int) = args.checkpoint_interval {
        project_config.output.checkpoint_interval = ckpt_int;
        tracing::debug!("CLI override: checkpoint_interval = {}", ckpt_int);
    }

    // Dry-run validation (after the config is resolved, so the resource
    // estimates below describe the run that would actually happen).
    if dry_run {
        return train_dry_run(&args, &project_config, &warnings, json_mode);
    }

    // Resolve `[output] export_format` before the configuration is moved into
    // the pipeline. This used to be read by nothing: whatever the config (or
    // `OXIGAF_OUTPUT_EXPORT_FORMAT`) asked for, training always wrote
    // `final_model.ply`, so a run configured for safetensors silently
    // produced a PLY and the format field was decoration.
    let final_export = FinalExport::parse(&project_config.output.export_format)?;

    // Create interactive controller if requested. The guard restores the
    // terminal on every exit path below, including the error paths.
    let (controller, _raw_guard) = if args.interactive {
        let ctrl = interactive::InteractiveController::new();
        ctrl.start_keyboard_listener();
        // Let a SIGINT ask the training loop to stop at its next step
        // boundary through the same flag the `q` key uses, instead of only
        // killing the process.
        let _ = INTERRUPT_FLAG.set(Arc::clone(&ctrl.quit_requested));
        (Some(ctrl), Some(RawModeGuard))
    } else {
        (None, None)
    };

    // 2. Ensure output directory exists
    std::fs::create_dir_all(&args.output)
        .with_context(|| format!("Failed to create output dir: {}", args.output.display()))?;

    // 3. Run the full pipeline
    let pipeline_cfg = pipeline::PipelineConfig {
        flame_model_path: args.flame_model.clone(),
        flame_params_path: args.flame_params.clone(),
        input_path: args.input.clone(),
        output_dir: args.output.clone(),
        resume_checkpoint: args.resume.clone(),
        device_index: args.device,
        project_config,
        patience: args.patience,
        min_delta: args.min_delta,
        metrics_output: args.metrics_output.clone(),
        metrics_format: args.metrics_format,
        tensorboard: args.tensorboard,
        tensorboard_dir: args.tensorboard_dir.clone(),
    };

    let result = pipeline::run_reconstruction(pipeline_cfg, verbosity, controller.as_ref())?;

    // 4. Export the final model in the configured format.
    let final_path = args.output.join(final_export.file_name());
    final_export.write(&result.model, &final_path)?;
    tracing::info!(
        format = final_export.label(),
        path = %final_path.display(),
        "Wrote the final model"
    );

    // 5. Render a preview turntable (unless disabled)
    if !args.no_preview {
        let preview_dir = args.output.join("preview");
        std::fs::create_dir_all(&preview_dir)?;
        let cameras = pipeline::default_orbit_cameras(512, 512);
        for (i, cam) in cameras.iter().enumerate() {
            let img = export::render_point_cloud(&result.model, cam);
            img.save(preview_dir.join(format!("view_{i:03}.png")))?;
        }
        tracing::info!(
            "Saved {} preview images to {}",
            cameras.len(),
            preview_dir.display(),
        );
    }

    let elapsed = start.elapsed();

    // Output based on mode
    if json_mode {
        let mut output = json_output::JsonOutput::success(
            "train",
            serde_json::json!({
                "num_gaussians": result.model.len(),
                "elapsed_seconds": elapsed.as_secs_f64(),
                "export_format": final_export.label(),
                "model_file": final_path.display().to_string(),
            }),
        );

        if final_path.exists() {
            output.add_artifact(final_export.label().to_string(), final_path.clone());
        }
        if let Some(sidecar) = final_export.sidecar(&final_path) {
            if sidecar.exists() {
                output.add_artifact("gltf-buffer".to_string(), sidecar);
            }
        }

        if !args.no_preview {
            let preview_dir = args.output.join("preview");
            if preview_dir.exists() {
                output.add_artifact("preview".to_string(), preview_dir);
            }
        }

        commands::flag_warnings::attach(&mut output, &warnings);

        print_json(output);
    } else {
        let throughput = result.total_iterations as f32 / elapsed.as_secs_f32();

        let checkpoint_path = args.output.join("checkpoints/final.json");
        let preview_dir = if !args.no_preview {
            Some(args.output.join("preview"))
        } else {
            None
        };

        let training_summary = summary::TrainingSummary {
            total_iterations: result.total_iterations,
            final_loss: result.final_loss,
            num_gaussians: result.model.len() as u32,
            num_rigid: result.num_rigid as u32,
            num_flexible: result.num_flexible as u32,
            sh_degree: result.model.sh_degree,
            elapsed,
            throughput_iters_per_sec: throughput,
            checkpoint_path: Some(checkpoint_path.display().to_string()),
            ply_path: Some(final_path.display().to_string()),
            preview_dir: preview_dir.map(|p| p.display().to_string()),
            peak_memory_mb: peak_rss_mb(),
        };

        training_summary.print();
    }

    Ok(())
}

/// `train --dry-run`: validate inputs and report the resources the resolved
/// configuration would actually need.
fn train_dry_run(
    args: &cli::TrainArgs,
    project_config: &config::ProjectConfig,
    warnings: &[String],
    json_mode: bool,
) -> Result<()> {
    let mut report = dry_run::DryRunReport::new();

    if !args.input.exists() {
        anyhow::bail!("Input not found: {}", args.input.display());
    }
    if !args.flame_model.exists() {
        anyhow::bail!("FLAME model not found: {}", args.flame_model.display());
    }

    // Resolved here too, so `--dry-run` fails on an unusable
    // `export_format` at the same point the real run would, and names the
    // file the real run would actually write.
    let final_export = FinalExport::parse(&project_config.output.export_format)?;

    dry_run::check_writable(&args.output)?;
    report.add_create(format!("{}/", args.output.display()));
    report.add_create(format!("{}/checkpoints/", args.output.display()));
    if !args.no_preview {
        report.add_create(format!("{}/preview/", args.output.display()));
    }
    let final_path = args.output.join(final_export.file_name());
    report.add_create(final_path.display().to_string());
    if let Some(sidecar) = final_export.sidecar(&final_path) {
        report.add_create(sidecar.display().to_string());
    }

    dry_run::check_gpu()?;

    // Estimate from the configuration that would actually be used, instead
    // of the fixed "1 hour / 4096 MB / 500 MB" literals this used to print
    // for every run regardless of size.
    let init = &project_config.training.init;
    let num_gaussians = init.num_rigid_gaussians + init.num_flexible_gaussians;
    report.resource_estimates = dry_run::estimate_training_resources(
        num_gaussians,
        init.sh_degree,
        project_config.training.image_size,
        project_config.training.total_iterations,
        // No measured throughput is available before the run starts, so no
        // duration is fabricated: `print_report` renders "(not estimated)".
        None,
    );

    if json_mode {
        let mut output = json_output::JsonOutput::success(
            "train",
            serde_json::json!({
                "dry_run": true,
                "input": args.input.display().to_string(),
                "output": args.output.display().to_string(),
                "would_create": report.would_create,
                "would_modify": report.would_modify,
                "would_delete": report.would_delete,
                "num_gaussians": num_gaussians,
                "total_iterations": project_config.training.total_iterations,
                "image_size": project_config.training.image_size,
                "sh_degree": init.sh_degree,
                "export_format": final_export.label(),
                "estimated_duration_sec": report.resource_estimates.estimated_duration_sec,
                "estimated_vram_mb": report.resource_estimates.estimated_vram_mb,
                "estimated_disk_mb": report.resource_estimates.estimated_disk_mb,
            }),
        );
        commands::flag_warnings::attach(&mut output, warnings);
        print_json(output);
    } else {
        output::success(&format!("Input validated: {}", args.input.display()));
        output::success(&format!(
            "FLAME model validated: {}",
            args.flame_model.display()
        ));
        report.print_report();
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// render
// ---------------------------------------------------------------------------

fn cmd_render(
    args: cli::RenderArgs,
    verbosity: Verbosity,
    dry_run: bool,
    json_mode: bool,
) -> Result<()> {
    let start = Instant::now();
    tracing::info!(?args.model, ?args.output, "Rendering avatar");

    if args.num_frames == 0 {
        anyhow::bail!("--num-frames must be at least 1");
    }
    if !(1..=5).contains(&args.splat_radius) {
        anyhow::bail!(
            "--splat-radius must be within 1..=5 (got {})",
            args.splat_radius
        );
    }

    // 1. Resolution: the quality preset only supplies the values the user
    // did not give. An explicit `--width 512` now wins even though 512 used
    // to be indistinguishable from the default.
    let (preset_width, preset_height) = args.quality.resolution();
    let width = args.width.unwrap_or(preset_width);
    let height = args.height.unwrap_or(preset_height);
    if args.width.is_none() || args.height.is_none() {
        tracing::info!(
            "Quality preset {:?} → resolution {}x{}",
            args.quality,
            width,
            height
        );
    }

    // 2. Load model and apply the preset's SH degree.
    let mut model = export::load_model(&args.model)
        .with_context(|| format!("Failed to load model: {}", args.model.display()))?;
    let original_sh_degree = model.sh_degree;
    let target_sh_degree = args.quality.sh_degree();
    if downsample_sh(&mut model, target_sh_degree) {
        tracing::info!(
            "Quality preset {:?} → SH degree {} (model had {})",
            args.quality,
            model.sh_degree,
            original_sh_degree
        );
    }
    tracing::info!(
        "Model loaded: {} Gaussians, SH degree {}",
        model.len(),
        model.sh_degree
    );

    // Flags the software renderer cannot honour yet. A caller who asked for
    // an animated render with a white background would otherwise get N
    // identical dark static frames, exit status 0, and no signal whatsoever;
    // the same text is attached to the JSON document below for machine
    // consumers. See [`commands::flag_warnings`] for the channel rationale.
    let warnings = commands::flag_warnings::render(&args);
    commands::flag_warnings::emit(&warnings);

    // 3. Build cameras based on mode
    let cameras = match args.mode {
        RenderMode::Frames => {
            if let Some(ref cam_path) = args.cameras {
                let json = std::fs::read_to_string(cam_path).with_context(|| {
                    format!("Failed to read cameras file: {}", cam_path.display())
                })?;
                let specs: Vec<pipeline::CameraSpec> =
                    serde_json::from_str(&json).with_context(|| {
                        format!("Failed to parse cameras JSON: {}", cam_path.display())
                    })?;
                specs
                    .iter()
                    .map(|s| {
                        pipeline::orbit_camera(s.azimuth, s.elevation, s.distance, width, height)
                    })
                    .collect::<Vec<_>>()
            } else {
                pipeline::default_orbit_cameras(width, height)
            }
        }
        RenderMode::Turntable => {
            // 360-degree turntable: the last frame stops one step short of a
            // full revolution so the loop does not duplicate frame 0.
            let step = 360.0 / args.num_frames as f32;
            (0..args.num_frames)
                .map(|i| pipeline::orbit_camera(i as f32 * step, 10.0, 0.6, width, height))
                .collect()
        }
        RenderMode::Orbit => {
            let step = 360.0 / args.num_frames as f32;
            (0..args.num_frames)
                .map(|i| {
                    let az = i as f32 * step;
                    let el = 10.0 + 20.0 * (az.to_radians().sin());
                    pipeline::orbit_camera(az, el, 0.6, width, height)
                })
                .collect()
        }
        RenderMode::Dolly => {
            // Dolly from 0.9 down to 0.3: `t` must reach 1.0 on the final
            // frame, so divide by `num_frames - 1` rather than `num_frames`
            // (which stopped short and never rendered the closest position).
            let last = args.num_frames.saturating_sub(1).max(1) as f32;
            (0..args.num_frames)
                .map(|i| {
                    let t = if args.num_frames > 1 {
                        i as f32 / last
                    } else {
                        0.0
                    };
                    let dist = 0.3 + 0.6 * (1.0 - t);
                    pipeline::orbit_camera(0.0, 10.0, dist, width, height)
                })
                .collect()
        }
    };

    // Determine output extension based on format
    let ext = match args.format {
        ImageFormat::Png => "png",
        ImageFormat::Jpeg => "jpg",
        ImageFormat::Exr => "exr",
    };

    if dry_run {
        let mut report = dry_run::DryRunReport::new();
        dry_run::check_writable(&args.output)?;
        for i in 0..cameras.len() {
            report.add_create(
                args.output
                    .join(format!("view_{i:03}.{ext}"))
                    .display()
                    .to_string(),
            );
        }
        if json_mode {
            let mut out = json_output::JsonOutput::success(
                "render",
                serde_json::json!({
                    "dry_run": true,
                    "num_views": cameras.len(),
                    "width": width,
                    "height": height,
                    "would_create": report.would_create,
                }),
            );
            commands::flag_warnings::attach(&mut out, &warnings);
            print_json(out);
        } else {
            report.print_report();
        }
        return Ok(());
    }

    // 4. Prepare output directory
    std::fs::create_dir_all(&args.output)
        .with_context(|| format!("Failed to create output dir: {}", args.output.display()))?;

    // 5. Render every view. Each frame is a pure function of the model and
    //    its own camera, so `--parallel` only changes *where* the work runs:
    //    0 uses rayon's global pool (all cores), 1 renders sequentially on a
    //    dedicated single-thread pool, and N builds an N-thread pool. The
    //    written images are identical either way.
    let rendered_files: Vec<PathBuf> = (0..cameras.len())
        .map(|index| args.output.join(format!("view_{index:03}.{ext}")))
        .collect();

    let renderer = ParallelRenderer::new(ParallelRenderConfig {
        num_threads: args.parallel,
        chunk_size: 1,
        output_dir: args.output.clone(),
        // Carried for completeness only: every task below names its own
        // destination, which keeps the established `view_NNN` numbering.
        filename_pattern: format!("view_{{:04d}}.{ext}"),
        width,
        height,
    })?;

    let tasks: Vec<FrameTask> = rendered_files
        .iter()
        .enumerate()
        .map(|(index, path)| FrameTask {
            frame_index: index,
            output_path: path.clone(),
            // The render closure indexes `cameras` by `frame_index`, so each
            // mode's own azimuth/elevation/**distance** trajectory is used
            // verbatim. These two summary fields cannot express a dolly or a
            // file-driven camera path, so this path does not read them.
            camera_azimuth_deg: 0.0,
            camera_elevation_deg: 0.0,
        })
        .collect();

    let progress = if !json_mode && verbosity.show_progress() {
        Some(BatchProgress::new(cameras.len() as u64, "views rendered"))
    } else {
        None
    };

    let outcome = renderer.execute(
        &tasks,
        |task| {
            let camera = cameras
                .get(task.frame_index)
                .ok_or_else(|| format!("no camera for frame {}", task.frame_index))?;
            let frame = export::render_point_cloud(&model, camera);
            save_render(&frame, &task.output_path, &args.format).map_err(|e| format!("{e:#}"))
        },
        progress.as_ref(),
    );

    if let Some(ref bar) = progress {
        bar.finish();
    }

    if !outcome.all_succeeded() {
        for (index, message) in &outcome.errors {
            tracing::error!("Frame {index} failed: {message}");
        }
        let first = outcome
            .errors
            .first()
            .map(|(_, message)| message.as_str())
            .unwrap_or("unknown error");
        anyhow::bail!(
            "{} of {} frame(s) failed to render; the first failure was: {first}",
            outcome.failed,
            outcome.total_frames,
        );
    }

    let elapsed = start.elapsed();

    if json_mode {
        let mut output = json_output::JsonOutput::success(
            "render",
            serde_json::json!({
                "num_views": cameras.len(),
                "width": width,
                "height": height,
                "sh_degree": model.sh_degree,
                "mode": format!("{:?}", args.mode),
                "format": format!("{:?}", args.format),
                "output_dir": args.output.display().to_string()
            }),
        );

        for file_path in rendered_files {
            if file_path.exists() {
                output.add_artifact("image".to_string(), file_path);
            }
        }

        commands::flag_warnings::attach(&mut output, &warnings);

        print_json(output);
    } else {
        let fps = if elapsed.as_secs_f32() > 0.0 {
            Some(cameras.len() as f32 / elapsed.as_secs_f32())
        } else {
            None
        };

        let render_summary = summary::RenderSummary {
            num_views: cameras.len() as u32,
            resolution: (width, height),
            format: format!("{:?}", args.format),
            mode: format!("{:?}", args.mode),
            elapsed,
            output_dir: args.output.display().to_string(),
            fps,
        };

        render_summary.print();
    }

    Ok(())
}

/// Encode one rendered frame in the requested image format.
///
/// Split out of the render loop so the same code path serves the sequential
/// and parallel schedulers — the bytes written do not depend on which thread
/// produced them.
fn save_render(frame: &image::RgbImage, path: &Path, format: &ImageFormat) -> Result<()> {
    match format {
        ImageFormat::Png => {
            frame
                .save(path)
                .with_context(|| format!("Failed to save image: {}", path.display()))?;
        }
        ImageFormat::Jpeg => {
            let jpeg_quality = 90u8;
            let file = std::fs::File::create(path)
                .with_context(|| format!("Failed to create file: {}", path.display()))?;
            let mut encoder =
                image::codecs::jpeg::JpegEncoder::new_with_quality(file, jpeg_quality);
            encoder
                .encode_image(frame)
                .with_context(|| format!("Failed to encode JPEG: {}", path.display()))?;
        }
        ImageFormat::Exr => {
            use image::{ImageBuffer, Rgb32FImage};
            let exr_frame: Rgb32FImage =
                ImageBuffer::from_fn(frame.width(), frame.height(), |x, y| {
                    let pixel = frame.get_pixel(x, y);
                    image::Rgb([
                        pixel[0] as f32 / 255.0,
                        pixel[1] as f32 / 255.0,
                        pixel[2] as f32 / 255.0,
                    ])
                });
            exr_frame
                .save(path)
                .with_context(|| format!("Failed to save EXR image: {}", path.display()))?;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// export
// ---------------------------------------------------------------------------

/// Write `model` as the PLY variant `--ply-format` asked for.
///
/// ASCII keeps going through the crate's long-standing writer, so existing
/// output is byte-for-byte unchanged. Binary little-endian is delegated to
/// [`export_ply`], which owns the binary encoder — previously the flag was
/// parsed and then ignored, so `--ply-format binary-le` silently produced a
/// several-times-larger ASCII file that `oxigaf compare` could not analyse in
/// full. Big-endian has no writer here (nor anywhere in the 3DGS ecosystem),
/// so it is refused rather than quietly downgraded.
///
/// # `f_rest` ordering
///
/// Both variants write the higher-order SH coefficients in this crate's
/// in-memory **coefficient-major, RGB-interleaved** order, which is what
/// [`export::export_ply`] has always written and what
/// [`export::load_model`] reads back — so ASCII and binary round-trip
/// identically, and `oxigaf compare` (which locates fields by property
/// *name* and uses `f_rest` only as a count) is unaffected. Note this is a
/// different order from `GaussianModel::save_ply`, which permutes to the
/// channel-major layout of the reference 3DGS implementation. Do not
/// "align" one of the two in isolation: changing either without the other
/// breaks the round-trip.
fn export_ply_variant(
    model: &GaussianModel,
    output: &Path,
    format: &cli::PlyFormat,
) -> Result<&'static str> {
    match format {
        cli::PlyFormat::Ascii => {
            export::export_ply(model, output)?;
            Ok("PLY (ascii)")
        }
        cli::PlyFormat::BinaryLe => {
            let flat = FlatScene::from_model(model)?;
            if flat.n == 0 {
                anyhow::bail!(
                    "Cannot write a binary PLY for a model with no Gaussians; \
                     use --ply-format ascii if an empty file is what you want."
                );
            }
            if let Some(parent) = output.parent() {
                if !parent.as_os_str().is_empty() {
                    std::fs::create_dir_all(parent).with_context(|| {
                        format!("Failed to create directory: {}", parent.display())
                    })?;
                }
            }
            let stats = export_ply::ply_export_scene(
                output,
                export_ply::PlyExportParams {
                    positions: &flat.positions,
                    rotations: &flat.rotations,
                    scales: &flat.log_scales,
                    opacities: &flat.opacity_logits,
                    sh_dc: &flat.sh_dc,
                    sh_rest: &flat.sh_rest,
                    n_rest_per_gaussian: flat.n_rest_per_gaussian,
                    format: export_ply::PlyFormat::BinaryLittleEndian,
                },
            )?;
            tracing::info!(
                "Wrote {} Gaussians (SH degree {}) as binary little-endian PLY, {} bytes",
                stats.n_gaussians,
                stats.sh_degree,
                stats.file_size_bytes,
            );
            Ok("PLY (binary little-endian)")
        }
        cli::PlyFormat::BinaryBe => anyhow::bail!(
            "--ply-format binary-be is not supported: this crate has no big-endian PLY writer, \
             and no 3DGS viewer expects one. Use --ply-format binary-le for a compact file or \
             ascii for a portable one."
        ),
    }
}

/// Filenames [`export::export_all_formats_parallel`] writes under
/// `--output` for `--format all`. One array so the overwrite check,
/// dry-run report and artifact list below cannot drift apart on the names.
const EXPORT_ALL_FILENAMES: [&str; 4] =
    ["model.ply", "model.safetensors", "model.glb", "model.json"];

fn cmd_export(
    args: cli::ExportArgs,
    verbosity: Verbosity,
    dry_run: bool,
    json_mode: bool,
) -> Result<()> {
    let start = Instant::now();
    tracing::info!(?args.model, ?args.output, ?args.format, "Exporting avatar");

    // glTF's `.bin` sidecar needs the same overwrite protection as the
    // named file, and `all`'s four files need it in place of `--output`
    // itself (an already-existing, e.g. freshly `mkdir`ed, empty directory
    // must not demand `--force`). `output_targets` is reused below for the
    // dry-run report and, for `all`, the real run's artifact list.
    let sidecar = matches!(args.format, ExportFormat::Gltf)
        .then(|| args.output.with_extension("bin"))
        .filter(|path| path != &args.output);
    let is_all_format = matches!(args.format, ExportFormat::All);

    let output_targets: Vec<PathBuf> = if is_all_format {
        EXPORT_ALL_FILENAMES
            .iter()
            .map(|name| args.output.join(name))
            .collect()
    } else {
        std::iter::once(args.output.clone())
            .chain(sidecar.clone())
            .collect()
    };

    for path in &output_targets {
        if path.exists() && !args.force {
            anyhow::bail!(
                "Output file already exists: {}. Use --force to overwrite.",
                path.display()
            );
        }
    }

    // Flags no writer consumes. Reported before the dry run so that
    // `export --dry-run` — the command whose whole job is to say what would
    // happen — says that these will not.
    let warnings = commands::flag_warnings::export(&args);
    commands::flag_warnings::emit(&warnings);

    // Dry-run validation
    if dry_run {
        let mut report = dry_run::DryRunReport::new();

        if !args.model.exists() {
            anyhow::bail!("Input model not found: {}", args.model.display());
        }

        dry_run::check_writable(&args.output)?;

        // `all` creates `--output` itself as a directory before the four
        // files land inside it -- report that step too.
        if is_all_format && !args.output.exists() {
            report.add_create(format!("{}/", args.output.display()));
        }
        for path in &output_targets {
            if path.exists() {
                report.add_modify(path.display().to_string());
            } else {
                report.add_create(path.display().to_string());
            }
        }

        // Size the estimate off the real model rather than a flat 100 MB.
        let (num_gaussians, sh_degree) = match export::load_model(&args.model) {
            Ok(model) => (
                model.len(),
                args.sh_degree
                    .unwrap_or(model.sh_degree)
                    .min(model.sh_degree),
            ),
            Err(e) => {
                tracing::warn!("Could not size the export estimate: {e:#}");
                (0, 0)
            }
        };
        // `all` writes four full-precision copies of the model, so the
        // single-format estimate is scaled by four rather than reused as-is.
        report.resource_estimates.estimated_disk_mb =
            dry_run::estimate_export_disk_mb(num_gaussians, sh_degree)
                .map(|mb| mb.saturating_mul(if is_all_format { 4 } else { 1 }));

        if json_mode {
            let mut out = json_output::JsonOutput::success(
                "export",
                serde_json::json!({
                    "dry_run": true,
                    "model": args.model.display().to_string(),
                    "would_create": report.would_create,
                    "would_modify": report.would_modify,
                    "num_gaussians": num_gaussians,
                    "sh_degree": sh_degree,
                    "estimated_disk_mb": report.resource_estimates.estimated_disk_mb,
                }),
            );
            commands::flag_warnings::attach(&mut out, &warnings);
            print_json(out);
        } else {
            output::success(&format!("Input model validated: {}", args.model.display()));
            report.print_report();
        }
        return Ok(());
    }

    // 1. Load model
    let mut model = export::load_model(&args.model)
        .with_context(|| format!("Failed to load model: {}", args.model.display()))?;
    tracing::info!(
        "Loaded model: {} Gaussians, SH degree {}",
        model.len(),
        model.sh_degree,
    );

    // 2. Honour --sh-degree (documented as "downsample if less than the
    // model's degree"). Requesting a higher degree cannot invent
    // coefficients, so that is an error rather than a silent no-op.
    if let Some(requested) = args.sh_degree {
        if requested > 3 {
            anyhow::bail!("--sh-degree must be within 0..=3 (got {requested})");
        }
        if requested > model.sh_degree {
            anyhow::bail!(
                "--sh-degree {requested} exceeds the model's SH degree {}; \
                 higher-order coefficients cannot be synthesised.",
                model.sh_degree
            );
        }
        if downsample_sh(&mut model, requested) {
            tracing::info!("Downsampled SH coefficients to degree {requested}");
        }
    }

    // 3. Export
    let format_name = match args.format {
        ExportFormat::Ply => export_ply_variant(&model, &args.output, &args.ply_format)?,
        ExportFormat::Safetensors => {
            export::export_safetensors(&model, &args.output)?;
            "safetensors"
        }
        ExportFormat::Gltf => {
            // Use the .gltf + .bin file-pair exporter (OXIGAF_gaussian_splat extension).
            export_gltf::export_gltf(&model, &args.output).map_err(anyhow::Error::from)?;
            "glTF 2.0"
        }
        ExportFormat::Json => {
            export::export_json_checkpoint(&model, &args.output)?;
            "JSON checkpoint"
        }
        ExportFormat::PointCloud => {
            export_pointcloud::export_pointcloud(&model, &args.output, args.point_color_mode)
                .map_err(anyhow::Error::from)?;
            "point cloud PLY"
        }
        ExportFormat::Mesh => {
            let mesh_cfg = export_mesh::MeshExportConfig {
                resolution: args.mesh_resolution,
                iso: args.mesh_iso,
                padding: args.mesh_padding,
                opacity_cutoff: 0.01,
            };
            export_mesh::export_mesh(&model, &args.output, &mesh_cfg)
                .map_err(anyhow::Error::from)?;
            "surface mesh PLY"
        }
        ExportFormat::All => {
            // `--json` reserves stdout for the single result document, same
            // as the `--from-hub` path above forcing `Quiet` regardless of
            // the caller's real verbosity.
            let all_verbosity = if json_mode {
                Verbosity::Quiet
            } else {
                verbosity
            };
            export::export_all_formats_parallel(&model, &args.output, all_verbosity)?;
            "all formats (PLY, safetensors, glTF, JSON)"
        }
    };

    let elapsed = start.elapsed();

    // `all` writes a directory, not a file: `args.output`'s own metadata
    // says nothing about the bytes written under it, so size and artifacts
    // come from the four files themselves instead.
    let file_size_mb = if is_all_format {
        output_targets
            .iter()
            .filter_map(|path| std::fs::metadata(path).ok())
            .map(|m| m.len() as f64 / (1024.0 * 1024.0))
            .sum::<f64>()
    } else {
        std::fs::metadata(&args.output)
            .ok()
            .map(|m| m.len() as f64 / (1024.0 * 1024.0))
            .unwrap_or(0.0)
    };

    if json_mode {
        let mut output = json_output::JsonOutput::success(
            "export",
            serde_json::json!({
                "format": format_name,
                "num_gaussians": model.len(),
                "sh_degree": model.sh_degree,
                "output_file": args.output.display().to_string()
            }),
        );

        if is_all_format {
            for path in &output_targets {
                if path.exists() {
                    output.add_artifact("export".to_string(), path.clone());
                }
            }
        } else {
            if args.output.exists() {
                output.add_artifact("export".to_string(), args.output.clone());
            }
            if let Some(ref path) = sidecar {
                if path.exists() {
                    output.add_artifact("export-buffer".to_string(), path.clone());
                }
            }
        }

        commands::flag_warnings::attach(&mut output, &warnings);

        print_json(output);
    } else {
        let export_summary = summary::ExportSummary {
            format: format_name.to_string(),
            input_file: args.model.display().to_string(),
            output_file: args.output.display().to_string(),
            file_size_mb,
            num_gaussians: model.len() as u32,
            elapsed,
        };

        export_summary.print();
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// doctor
// ---------------------------------------------------------------------------

fn cmd_doctor(args: cli::DoctorArgs, verbosity: Verbosity, json_mode: bool) -> Result<()> {
    use serde_json::json;

    if !json_mode {
        println!();
        output::info("OxiGAF System Diagnostics");
        output::separator();
        println!();
    }

    let selector = args.check.as_ref();
    let mut all_ok = true;
    let mut gpu_failure: Option<String> = None;
    let mut diagnostics = serde_json::Map::new();

    // 1. GPU — the adapter `--device <index>` selects, which is the one a
    // run started with the same flag would actually use. Asking wgpu for
    // "the high-performance adapter" instead (as this used to) could report
    // a healthy GPU while the requested one was absent or a CPU fallback.
    if doctor_runs(selector, DoctorCheck::Gpu) {
        if !json_mode {
            output::header("GPU Configuration");
        }
        match gpu_probe::probe_adapter(args.device) {
            Ok(report) => {
                let software = report.is_software_fallback();
                if !json_mode {
                    if software {
                        // A CPU adapter is real to wgpu and useless to a
                        // 3DGS trainer; reporting it as a healthy GPU is the
                        // true-but-useless answer a diagnostic exists to
                        // avoid.
                        output::warning(&format!(
                            "GPU adapter {} is a software rasteriser; training will be \
                             impractically slow. Install GPU drivers, or pick another \
                             adapter with --device <index>.",
                            report.summary()
                        ));
                    } else {
                        output::success(&format!("GPU adapter found: {}", report.summary()));
                    }
                }
                diagnostics.insert(
                    "gpu".to_string(),
                    json!({
                        "status": if software { "warning" } else { "ok" },
                        "adapter": report.summary(),
                        "device_index": report.index,
                        "name": report.name,
                        "backend": report.backend,
                        "device_type": report.device_type,
                        "software_fallback": software,
                    }),
                );
                if software {
                    all_ok = false;
                }
            }
            Err(e) => {
                if !json_mode {
                    output::error(&format!("GPU not available: {:#}", e));
                }
                diagnostics.insert(
                    "gpu".to_string(),
                    json!({
                        "status": "error",
                        "device_index": args.device,
                        "error": format!("{e:#}"),
                    }),
                );
                all_ok = false;
                gpu_failure = Some(format!("{e:#}"));
            }
        }
    }

    // 2. FLAME model (only when a path is provided)
    if doctor_runs(selector, DoctorCheck::Flame) {
        match args.flame_model {
            Some(ref flame_path) => {
                if !json_mode {
                    output::header("FLAME Model");
                }
                match check_flame_model(flame_path) {
                    Ok(info) => {
                        if !json_mode {
                            output::success(&format!("FLAME model valid: {}", info));
                        }
                        diagnostics
                            .insert("flame".to_string(), json!({ "status": "ok", "info": info }));
                    }
                    Err(e) => {
                        if !json_mode {
                            output::error(&format!("FLAME model invalid: {}", e));
                        }
                        diagnostics.insert(
                            "flame".to_string(),
                            json!({ "status": "error", "error": e.to_string() }),
                        );
                        all_ok = false;
                    }
                }
            }
            None if matches!(selector, Some(DoctorCheck::Flame)) => {
                // `--check flame` with nothing to check is a usage error,
                // not a silent pass.
                anyhow::bail!("`doctor --check flame` requires --flame-model <dir>");
            }
            None => {}
        }
    }

    // 3. Cache directory
    let cache_dir = resolve_cache_dir(args.cache_dir.as_ref())?;
    if doctor_runs(selector, DoctorCheck::Cache) {
        if !json_mode {
            output::header("Asset Cache");
        }
        match check_cache(&cache_dir) {
            Ok(info) => {
                if !json_mode {
                    output::success(&format!("Cache directory: {}", info));
                }
                diagnostics.insert("cache".to_string(), json!({ "status": "ok", "info": info }));
            }
            Err(e) => {
                if !json_mode {
                    output::warning(&format!("Cache issue: {}", e));
                }
                diagnostics.insert(
                    "cache".to_string(),
                    json!({ "status": "warning", "warning": e.to_string() }),
                );
            }
        }
    }

    // 4. Version information
    if doctor_runs(selector, DoctorCheck::Version) {
        if !json_mode {
            output::header("Version Information");
        }
        let version_info = get_version_info();
        if !json_mode {
            output::value("OxiGAF", &version_info.oxigaf);
            output::value("Rust", &version_info.rust);
            output::value("Platform", &version_info.platform);
        }
        diagnostics.insert(
            "version".to_string(),
            json!({
                "oxigaf": version_info.oxigaf,
                "rust": version_info.rust,
                "platform": version_info.platform,
            }),
        );
    }

    // 5. Disk space (verbose, or explicitly selected via --check cache)
    if doctor_runs(selector, DoctorCheck::Cache) && (json_mode || verbosity >= Verbosity::Verbose) {
        match available_disk_mb(&cache_dir) {
            Ok(mb) => {
                if !json_mode {
                    output::header("Disk Space");
                    output::value("Available Space", &format!("{} MB", mb));
                }
                diagnostics.insert("disk".to_string(), json!({ "available_mb": mb }));
            }
            Err(e) => {
                if !json_mode {
                    output::warning(&format!("Could not check disk space: {}", e));
                }
                diagnostics.insert(
                    "disk".to_string(),
                    json!({ "status": "warning", "warning": e.to_string() }),
                );
            }
        }
    }

    if json_mode {
        diagnostics.insert("all_ok".to_string(), json!(all_ok));
        let output = json_output::JsonOutput::success("doctor", json!(diagnostics));
        print_json(output);
    } else {
        println!();
        output::separator();
        if all_ok {
            output::success("All checks passed! System is ready.");
        } else {
            output::warning("Some checks failed. See above for details.");
        }
    }

    // A diagnostic command that always exits 0 is useless in CI.
    if !all_ok {
        if let Some(reason) = gpu_failure {
            // Name the adapter that was asked for and why it could not be
            // used. This used to report a fixed `backend: "any"`, so the
            // last line a user saw was "GPU not available: any" — less
            // specific than the message already printed above it, and
            // useless for telling "--device 99 does not exist" apart from
            // "no drivers installed".
            return Err(CliError::GpuNotAvailable {
                backend: format!("device {} — {reason}", args.device),
                fallback: None,
            }
            .into());
        }
        anyhow::bail!("One or more diagnostic checks failed");
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// setup
// ---------------------------------------------------------------------------

fn cmd_setup(
    args: cli::SetupArgs,
    verbosity: Verbosity,
    dry_run: bool,
    json_mode: bool,
) -> Result<()> {
    let cache_dir = resolve_cache_dir(args.cache_dir.as_ref())?;

    // HuggingFace Hub download
    if let Some(hub_spec) = args.from_hub {
        if args.offline {
            anyhow::bail!("--offline conflicts with --from-hub: nothing can be downloaded");
        }
        if dry_run {
            let source = assets::HfModelSource::parse(&hub_spec)
                .context("Failed to parse HuggingFace model specification")?;
            let mut report = dry_run::DryRunReport::new();
            report.add_create(format!(
                "{} (from {}/{})",
                cache_dir.display(),
                source.repo_id,
                source.filename
            ));
            if json_mode {
                print_json(json_output::JsonOutput::success(
                    "setup",
                    serde_json::json!({
                        "dry_run": true,
                        "source": "huggingface_hub",
                        "repo_id": source.repo_id,
                        "filename": source.filename,
                        "cache_dir": cache_dir.display().to_string(),
                    }),
                ));
            } else {
                report.print_report();
            }
            return Ok(());
        }

        let mut source = assets::HfModelSource::parse(&hub_spec)
            .context("Failed to parse HuggingFace model specification")?;

        if let Some(rev) = args.revision {
            source.revision = Some(rev);
        }
        if let Some(filename) = args.filename {
            source = source.with_filename(filename);
        }

        let token = args.hf_token.or_else(assets::get_hf_token);

        // `download_with_progress` writes its banner ("📥 Downloading …",
        // "✓ Cached: …") and its indicatif bar to **stdout**, which under
        // `--json` belongs exclusively to the single result document. It
        // silences both at `Verbosity::Quiet`, so that is what JSON mode
        // asks for regardless of `-v`; the same information still reaches
        // the log sink through the `tracing::info!` below and `--log-file`.
        let download_verbosity = if json_mode {
            Verbosity::Quiet
        } else {
            verbosity
        };

        tracing::info!(?hub_spec, ?source.revision, "Downloading from HuggingFace Hub");
        let downloaded_path = assets::download_with_progress(
            &source.repo_id,
            &source.filename,
            source.revision.as_deref(),
            token.as_deref(),
            download_verbosity,
        )
        .context("Failed to download model from HuggingFace Hub")?;

        if json_mode {
            print_json(json_output::JsonOutput::success(
                "setup",
                serde_json::json!({
                    "source": "huggingface_hub",
                    "repo_id": source.repo_id,
                    "filename": source.filename,
                    "revision": source.revision,
                    "path": downloaded_path.display().to_string()
                }),
            ));
        } else {
            output::success(&format!(
                "Model downloaded successfully from HuggingFace Hub\nPath: {}",
                downloaded_path.display()
            ));
        }
        return Ok(());
    }

    // Offline mode: report what is already cached, download nothing.
    if args.offline {
        let expected = assets::expected_asset_paths(&cache_dir);
        let selected = select_assets(&expected, args.only.as_deref());
        let present: Vec<&PathBuf> = selected.iter().copied().filter(|p| p.exists()).collect();
        let missing: Vec<&PathBuf> = selected.iter().copied().filter(|p| !p.exists()).collect();

        if json_mode {
            print_json(json_output::JsonOutput::success(
                "setup",
                serde_json::json!({
                    "offline": true,
                    "cache_dir": cache_dir.display().to_string(),
                    "present": present.iter().map(|p| p.display().to_string()).collect::<Vec<_>>(),
                    "missing": missing.iter().map(|p| p.display().to_string()).collect::<Vec<_>>(),
                }),
            ));
        } else {
            output::info(&format!(
                "Offline check of {} — {}/{} asset(s) cached",
                cache_dir.display(),
                present.len(),
                selected.len()
            ));
            for path in &missing {
                output::warning(&format!("missing: {}", path.display()));
            }
        }

        if !missing.is_empty() {
            anyhow::bail!(
                "{} asset(s) are not cached; re-run without --offline to download them",
                missing.len()
            );
        }
        return Ok(());
    }

    if dry_run {
        let expected = assets::expected_asset_paths(&cache_dir);
        let selected = select_assets(&expected, args.only.as_deref());
        let mut report = dry_run::DryRunReport::new();
        for path in &selected {
            if path.exists() {
                report.add_modify(path.display().to_string());
            } else {
                report.add_create(path.display().to_string());
            }
        }
        if json_mode {
            print_json(json_output::JsonOutput::success(
                "setup",
                serde_json::json!({
                    "dry_run": true,
                    "cache_dir": cache_dir.display().to_string(),
                    "would_create": report.would_create,
                    "would_modify": report.would_modify,
                }),
            ));
        } else {
            report.print_report();
        }
        return Ok(());
    }

    // Only the downloading path is reached here. `--skip-checksum` and
    // `--only` used to be *warned about* rather than honoured, because
    // `assets::setup_cache` took neither; `assets::setup_cache_with_options`
    // implements both — the filter is resolved through the same
    // `runtime::select_assets` the `--offline` and `--dry-run` paths above
    // use, so the three cannot disagree about what a filter selects — so
    // they are passed through instead of apologised for.
    tracing::info!(
        ?cache_dir,
        skip_checksum = args.skip_checksum,
        only = args.only.as_deref(),
        "Setting up model assets"
    );
    assets::setup_cache_with_options(
        &cache_dir,
        verbosity,
        json_mode,
        args.skip_checksum,
        args.only.as_deref(),
    )?;
    Ok(())
}

// ---------------------------------------------------------------------------
// cache
// ---------------------------------------------------------------------------

/// Manage cached assets.
fn cmd_cache(command: CacheCommands, global_dry_run: bool, json_mode: bool) -> Result<()> {
    let cache_dir = default_cache_dir()?;

    match command {
        CacheCommands::List => {
            if !json_mode {
                return cache::list_cache(&cache_dir);
            }
            let metadata = cache::CacheMetadata::load(&cache_dir)?;
            print_json(json_output::JsonOutput::success(
                "cache list",
                serde_json::json!({
                    "cache_dir": cache_dir.display().to_string(),
                    "total_size_bytes": metadata.total_size(),
                    "entries": serde_json::to_value(&metadata.entries)
                        .unwrap_or(serde_json::Value::Null),
                }),
            ));
            Ok(())
        }

        CacheCommands::Verify => {
            if !json_mode {
                return cache::verify_cache(&cache_dir);
            }
            let metadata = cache::CacheMetadata::load(&cache_dir)?;
            let expected = assets::expected_asset_paths(&cache_dir);
            let assets_json: Vec<serde_json::Value> = expected
                .iter()
                .map(|path| {
                    let size = std::fs::metadata(path).ok().map(|m| m.len());
                    let recorded = metadata
                        .entries
                        .iter()
                        .find(|entry| entry.path == *path)
                        .map(|entry| entry.size_bytes);
                    serde_json::json!({
                        "path": path.display().to_string(),
                        "present": path.exists(),
                        "size_bytes": size,
                        "recorded_size_bytes": recorded,
                        "size_matches": match (size, recorded) {
                            (Some(a), Some(b)) => Some(a == b),
                            _ => None,
                        },
                    })
                })
                .collect();
            let missing = expected.iter().filter(|p| !p.exists()).count();
            print_json(json_output::JsonOutput::success(
                "cache verify",
                serde_json::json!({
                    "cache_dir": cache_dir.display().to_string(),
                    "assets": assets_json,
                    "missing": missing,
                }),
            ));
            if missing > 0 {
                anyhow::bail!("{missing} expected asset(s) are missing from the cache");
            }
            Ok(())
        }

        CacheCommands::Path => {
            if json_mode {
                print_json(json_output::JsonOutput::success(
                    "cache path",
                    serde_json::json!({ "path": cache_dir.display().to_string() }),
                ));
            } else {
                println!("{}", cache_dir.display());
            }
            Ok(())
        }

        CacheCommands::Clean {
            max_age_days,
            dry_run,
        } => {
            if json_mode {
                // `cache::clean_cache` writes its report straight to stdout,
                // which would corrupt the JSON stream. Reimplementing the
                // deletion policy here would duplicate it, so refuse
                // explicitly instead of emitting invalid JSON.
                anyhow::bail!(
                    "`cache clean --json` is not supported yet (the cleaner writes a human \
                     report to stdout). Run it without --json, or use `cache list --json` \
                     to inspect the cache."
                );
            }
            // The global `--dry-run` must protect this destructive command
            // too, not only the subcommand-local flag.
            cache::clean_cache(&cache_dir, max_age_days, dry_run || global_dry_run)
        }
    }
}

// ---------------------------------------------------------------------------
// completions
// ---------------------------------------------------------------------------

/// Generate shell completion scripts.
fn cmd_completions(shell: clap_complete::Shell) -> Result<()> {
    let mut cmd = Cli::command();
    let bin_name = cmd.get_name().to_string();
    generate(shell, &mut cmd, bin_name, &mut std::io::stdout());
    Ok(())
}

// The test module lives in `main_tests.rs`, not inline, purely to keep this
// file under the workspace's 2000-line policy -- see that file's header for
// why `#[path]` rather than the usual bare `mod tests;` `src/tests.rs`
// lookup (it would collide in spelling, though not in namespace, with the
// crate's separate `tests/` integration-test directory).
#[cfg(test)]
#[path = "main_tests.rs"]
mod tests;
