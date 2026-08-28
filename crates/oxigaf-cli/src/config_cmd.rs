//! `oxigaf config-cmd` subcommand — manage OxiGAF configuration files.
//!
//! Subcommands:
//! - `init [--output <path>]` — write a default config TOML to stdout or to a file.
//! - `validate <path>` — parse a config TOML and report errors or "OK".
//! - `show <path>` — parse and pretty-print all configuration fields.
//!
//! # Output modes
//!
//! Every handler here comes in three flavours: a plain-text one
//! ([`run_config_cmd`]), a JSON one ([`run_config_cmd_json`]), and the
//! [`CmdContext`]-driven [`run_config_cmd_with`] the other two delegate to.
//! Under `--json` the TOML body, the validation verdict and the parsed
//! configuration become fields of a single JSON document instead of being
//! printed as text, so `oxigaf --json config-cmd …` puts exactly one JSON
//! value on stdout. Human-readable text — the TOML itself, the wizard's
//! banner, `OK — … is valid.` — is emitted only from the human branch of
//! [`crate::commands::emit`], never alongside the document.

use std::path::Path;

use anyhow::{Context, Result};
use serde_json::json;

use crate::cli::ConfigCmdSubcommand;
use crate::commands::{emit, CmdContext};
use crate::config::ProjectConfig;
use crate::verbosity::Verbosity;

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// The context used by the plain-text entry points.
fn human_context() -> CmdContext {
    CmdContext::new(Verbosity::Normal, false, false)
}

/// The context used by the JSON entry points.
fn json_context() -> CmdContext {
    CmdContext::new(Verbosity::Normal, true, false)
}

/// Run the `config-cmd` subcommand, printing human-readable output.
///
/// # Errors
///
/// Propagates unreadable, unparseable or invalid config files, and a
/// refusal to overwrite an existing `--output` path.
pub fn run_config_cmd(command: ConfigCmdSubcommand) -> Result<()> {
    run_config_cmd_with(command, &human_context())
}

/// Run the `config-cmd` subcommand, emitting a single JSON document.
///
/// # Errors
///
/// Same failures as [`run_config_cmd`].
pub fn run_config_cmd_json(command: ConfigCmdSubcommand) -> Result<()> {
    run_config_cmd_with(command, &json_context())
}

/// Run the `config-cmd` subcommand, rendering according to `ctx`.
///
/// # Errors
///
/// Propagates unreadable, unparseable or invalid config files, and a
/// refusal to overwrite an existing `--output` path.
pub fn run_config_cmd_with(command: ConfigCmdSubcommand, ctx: &CmdContext) -> Result<()> {
    match command {
        ConfigCmdSubcommand::Init {
            output,
            interactive,
        } => {
            if interactive {
                run_config_init_wizard_with(output.as_deref(), ctx)
            } else {
                run_config_init_with(output.as_deref(), ctx)
            }
        }
        ConfigCmdSubcommand::Validate { path } => run_config_validate_with(&path, ctx),
        ConfigCmdSubcommand::Show { path } => run_config_show_with(&path, ctx),
    }
}

/// Artifact list for a handler that either wrote one file or wrote none.
fn optional_artifact(path: Option<&Path>) -> Vec<(&'static str, &Path)> {
    path.map(|path| vec![("config", path)]).unwrap_or_default()
}

/// Build a result document only when one will actually be emitted.
///
/// Serializing a [`ProjectConfig`] to JSON is fallible (a path that is not
/// valid UTF-8 has no JSON string form), and a plain-text run must not
/// start failing on a document nobody asked for. The placeholder handed to
/// [`emit`] in human mode is never read: `emit` runs the human closure
/// instead.
fn payload_for(
    ctx: &CmdContext,
    build: impl FnOnce() -> Result<serde_json::Value>,
) -> Result<serde_json::Value> {
    if ctx.json {
        build()
    } else {
        Ok(serde_json::Value::Null)
    }
}

/// The `config-cmd init` result document.
///
/// `toml` carries the exact bytes written to `output` (or, with no
/// `--output`, the exact text the human run would have printed), and
/// `config` the same configuration as JSON so a caller never has to parse
/// TOML to read one field back.
fn init_payload(
    output: Option<&Path>,
    toml_text: &str,
    config: &ProjectConfig,
    dry_run: bool,
) -> Result<serde_json::Value> {
    let config_json = serde_json::to_value(config).context("Failed to serialize config to JSON")?;
    Ok(json!({
        "action": "init",
        "interactive": false,
        "output": output.map(|path| path.display().to_string()),
        "dry_run": dry_run,
        "toml": toml_text,
        "config": config_json,
    }))
}

/// The `config-cmd init --interactive` result document.
///
/// Adds the hardware signal the wizard acted on, and the four settings it
/// derived from it, to everything [`init_payload`] reports — so a scripted
/// caller can see *why* it chose them, not just what it chose.
fn wizard_payload(
    output: Option<&Path>,
    toml_text: &str,
    config: &ProjectConfig,
    hardware: &str,
    settings: (u32, usize, u32, usize),
    dry_run: bool,
) -> Result<serde_json::Value> {
    let (sh_degree, views_per_step, image_size, max_gaussians) = settings;
    let config_json =
        serde_json::to_value(config).context("Failed to serialize wizard config to JSON")?;
    Ok(json!({
        "action": "init",
        "interactive": true,
        "hardware": hardware,
        "settings": {
            "sh_degree": sh_degree,
            "views_per_step": views_per_step,
            "image_size": image_size,
            "max_gaussians": max_gaussians,
        },
        "output": output.map(|path| path.display().to_string()),
        "dry_run": dry_run,
        "toml": toml_text,
        "config": config_json,
    }))
}

/// The `config-cmd validate` result document.
///
/// Only ever reports success: a config that fails to parse or validate is
/// an `Err`, which `main.rs` renders as the error document with a non-zero
/// exit status.
fn validate_payload(path: &Path) -> serde_json::Value {
    json!({
        "action": "validate",
        "path": path.display().to_string(),
        "valid": true,
    })
}

/// The `config-cmd show` result document.
fn show_payload(path: &Path, config: &ProjectConfig) -> Result<serde_json::Value> {
    let config_json = serde_json::to_value(config).context("Failed to serialize config to JSON")?;
    Ok(json!({
        "action": "show",
        "path": path.display().to_string(),
        "config": config_json,
    }))
}

// ---------------------------------------------------------------------------
// config init
// ---------------------------------------------------------------------------

/// Write the default `ProjectConfig` as TOML to stdout, or to `output` if given.
///
/// # Errors
///
/// Returns an error when the config cannot be serialized, when `output`
/// already exists, or when it cannot be written.
pub fn run_config_init(output: Option<&Path>) -> Result<()> {
    run_config_init_with(output, &human_context())
}

/// [`run_config_init`], rendering according to `ctx`.
///
/// Under `--json` the TOML body is a field of the result document (`toml`)
/// alongside the same configuration as JSON (`config`) rather than being
/// printed to stdout, so the stream stays a single parseable value. Under
/// `--dry-run` the file is not written and the document says so, matching
/// [`crate::commands::prepare_output`]'s contract for every other handler.
///
/// # Errors
///
/// Returns an error when the config cannot be serialized, when `output`
/// already exists, or when it cannot be written.
pub fn run_config_init_with(output: Option<&Path>, ctx: &CmdContext) -> Result<()> {
    let default_config = ProjectConfig::default();

    let toml_string = toml::to_string_pretty(&default_config)
        .context("Failed to serialize default config to TOML")?;

    let written = write_config_file(output, &toml_string, ctx)
        .with_context(|| "Failed to write default config")?;

    let payload = payload_for(ctx, || {
        init_payload(output, &toml_string, &default_config, ctx.dry_run)
    })?;

    emit(
        ctx,
        "config-cmd init",
        payload,
        &optional_artifact(written),
        || match output {
            None => print!("{}", toml_string),
            Some(path) => println!("{}", wrote_line(path, written.is_some())),
        },
    );

    Ok(())
}

/// Write `contents` to `output`, honouring the overwrite guard and
/// `--dry-run`.
///
/// Returns the path actually written, or `None` when there was nothing to
/// write (no `--output`) or when `--dry-run` asked for a rehearsal. The
/// existing-file check runs even under `--dry-run`, so a rehearsal reports
/// the same refusal a real run would hit rather than promising a write that
/// would fail.
///
/// # Errors
///
/// Returns an error when `output` exists, or when the write fails.
fn write_config_file<'a>(
    output: Option<&'a Path>,
    contents: &str,
    ctx: &CmdContext,
) -> Result<Option<&'a Path>> {
    let Some(path) = output else {
        return Ok(None);
    };
    if path.exists() {
        anyhow::bail!(
            "Output file already exists: {}. Remove it first, or re-run with a \
             different -o/--output path.",
            path.display()
        );
    }
    if ctx.dry_run {
        return Ok(None);
    }
    std::fs::write(path, contents)
        .with_context(|| format!("Failed to write config to: {}", path.display()))?;
    Ok(Some(path))
}

/// The human line reporting where a config went — or would have gone.
fn wrote_line(path: &Path, written: bool) -> String {
    if written {
        format!("Default config written to: {}", path.display())
    } else {
        format!(
            "Would write default config to: {} (--dry-run)",
            path.display()
        )
    }
}

// ---------------------------------------------------------------------------
// config validate
// ---------------------------------------------------------------------------

/// Parse a config TOML file and validate its contents.
///
/// Prints "OK" on success, or a detailed error message on failure.
///
/// # Errors
///
/// Returns an error when the file is missing, unreadable, not valid TOML,
/// or fails [`ProjectConfig::validate`].
pub fn run_config_validate(path: &Path) -> Result<()> {
    run_config_validate_with(path, &human_context())
}

/// [`run_config_validate`], rendering according to `ctx`.
///
/// A failed validation stays an `Err` in both modes: the JSON document is
/// the *success* report (`"valid": true`), and the top-level handler in
/// `main.rs` renders a failure as the error document so a scripted caller
/// sees a non-zero exit status rather than a document claiming success.
///
/// # Errors
///
/// Returns an error when the file is missing, unreadable, not valid TOML,
/// or fails [`ProjectConfig::validate`].
pub fn run_config_validate_with(path: &Path, ctx: &CmdContext) -> Result<()> {
    if !path.exists() {
        anyhow::bail!("Config file not found: {}", path.display());
    }

    let content = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read config file: {}", path.display()))?;

    let config: ProjectConfig = toml::from_str(&content)
        .with_context(|| format!("Failed to parse TOML in: {}", path.display()))?;

    config
        .validate()
        .with_context(|| format!("Config validation failed for: {}", path.display()))?;

    emit(
        ctx,
        "config-cmd validate",
        validate_payload(path),
        &[],
        || {
            println!("OK — {} is valid.", path.display());
        },
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// config show
// ---------------------------------------------------------------------------

/// Parse and pretty-print a config TOML file.
///
/// # Errors
///
/// Returns an error when the file is missing, unreadable, or not valid
/// TOML for a [`ProjectConfig`].
pub fn run_config_show(path: &Path) -> Result<()> {
    run_config_show_with(path, &human_context())
}

/// [`run_config_show`], rendering according to `ctx`.
///
/// Under `--json` the whole parsed configuration is the result document's
/// `config` field — the machine-readable equivalent of the field-by-field
/// listing, and a superset of it, since nothing is elided for width.
///
/// # Errors
///
/// Returns an error when the file is missing, unreadable, or not valid
/// TOML for a [`ProjectConfig`].
pub fn run_config_show_with(path: &Path, ctx: &CmdContext) -> Result<()> {
    if !path.exists() {
        anyhow::bail!("Config file not found: {}", path.display());
    }

    let content = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read config file: {}", path.display()))?;

    let config: ProjectConfig = toml::from_str(&content)
        .with_context(|| format!("Failed to parse TOML in: {}", path.display()))?;

    let payload = payload_for(ctx, || show_payload(path, &config))?;
    emit(ctx, "config-cmd show", payload, &[], || {
        print_config_human(&config, path);
    });
    Ok(())
}

/// Print every configuration field, section by section.
fn print_config_human(config: &ProjectConfig, path: &Path) {
    println!("=== OxiGAF Configuration ===");
    println!("File: {}", path.display());
    println!();

    // [model]
    println!("[model]");
    println!(
        "  flame_model_path    = {:?}",
        config.model.flame_model_path
    );
    println!(
        "  diffusion_weights_dir = {:?}",
        config.model.diffusion_weights_dir
    );
    println!();

    // [device]
    println!("[device]");
    println!("  backend   = {:?}", config.device.backend);
    println!("  gpu_index = {}", config.device.gpu_index);
    println!();

    // [training]
    let t = &config.training;
    println!("[training]");
    println!("  total_iterations        = {}", t.total_iterations);
    println!("  views_per_step          = {}", t.views_per_step);
    println!("  image_size              = {}", t.image_size);
    println!("  guidance_scale_start    = {}", t.guidance_scale_start);
    println!("  guidance_scale_end      = {}", t.guidance_scale_end);
    println!("  guidance_anneal_steps   = {}", t.guidance_anneal_steps);
    println!("  num_inference_steps     = {}", t.num_inference_steps);
    println!("  opacity_reset_interval  = {}", t.opacity_reset_interval);
    println!();

    // [training.init]
    let init = &t.init;
    println!("[training.init]");
    println!("  num_rigid_gaussians     = {}", init.num_rigid_gaussians);
    println!(
        "  num_flexible_gaussians  = {}",
        init.num_flexible_gaussians
    );
    println!("  initial_scale           = {}", init.initial_scale);
    println!("  initial_opacity         = {}", init.initial_opacity);
    println!("  sh_degree               = {}", init.sh_degree);
    println!();

    // [training.optimizer]
    let opt = &t.optimizer;
    println!("[training.optimizer]");
    println!("  position_lr             = {:.2e}", opt.position_lr);
    println!("  position_lr_final       = {:.2e}", opt.position_lr_final);
    println!("  rotation_lr             = {:.2e}", opt.rotation_lr);
    println!("  scale_lr                = {:.2e}", opt.scale_lr);
    println!("  opacity_lr              = {:.2e}", opt.opacity_lr);
    println!("  sh_lr                   = {:.2e}", opt.sh_lr);
    println!("  offset_lr               = {:.2e}", opt.offset_lr);
    println!("  beta1                   = {}", opt.beta1);
    println!("  beta2                   = {}", opt.beta2);
    println!("  epsilon                 = {:.2e}", opt.epsilon);
    println!(
        "  position_lr_decay_steps = {}",
        opt.position_lr_decay_steps
    );
    println!();

    // [training.density_control]
    let dc = &t.density_control;
    println!("[training.density_control]");
    println!("  interval                = {}", dc.interval);
    println!("  start_iteration         = {}", dc.start_iteration);
    println!("  end_iteration           = {}", dc.end_iteration);
    println!("  grad_threshold          = {:.2e}", dc.grad_threshold);
    println!("  min_opacity             = {}", dc.min_opacity);
    println!("  max_screen_size         = {}", dc.max_screen_size);
    println!("  split_scale_threshold   = {}", dc.split_scale_threshold);
    println!("  max_gaussians           = {}", dc.max_gaussians);
    println!();

    // [training.loss]
    let loss = &t.loss;
    println!("[training.loss]");
    println!("  lambda_l1               = {}", loss.lambda_l1);
    println!("  lambda_ssim             = {}", loss.lambda_ssim);
    println!("  lambda_ms_ssim          = {}", loss.lambda_ms_ssim);
    println!("  lambda_lpips            = {}", loss.lambda_lpips);
    println!(
        "  lambda_position_reg     = {:.2e}",
        loss.lambda_position_reg
    );
    println!("  lambda_scale_reg        = {:.2e}", loss.lambda_scale_reg);
    println!(
        "  lambda_opacity_reg      = {:.2e}",
        loss.lambda_opacity_reg
    );
    println!("  lambda_normal           = {}", loss.lambda_normal);
    println!(
        "  lambda_gradient_penalty = {}",
        loss.lambda_gradient_penalty
    );
    println!(
        "  gradient_penalty_threshold = {}",
        loss.gradient_penalty_threshold
    );
    println!();

    // [output]
    println!("[output]");
    println!(
        "  checkpoint_interval = {}",
        config.output.checkpoint_interval
    );
    println!("  log_interval        = {}", config.output.log_interval);
    println!("  export_format       = {:?}", config.output.export_format);
}

// ---------------------------------------------------------------------------
// config init --interactive (hardware-detection wizard)
// ---------------------------------------------------------------------------

/// Real-hardware signal used to pick VRAM-bound wizard defaults.
///
/// Populated from an actual `wgpu` adapter query in [`detect_gpu`]. Kept as
/// a plain struct (rather than querying wgpu inline in the wizard) so the
/// tier-selection logic in [`wizard_hardware_profile`] can be unit-tested
/// without touching real hardware.
#[derive(Debug, Clone, Copy)]
struct DetectedGpu {
    device_type: wgpu::DeviceType,
    max_buffer_size: u64,
    max_texture_dimension_2d: u32,
}

/// Query the first available `wgpu` adapter for a coarse hardware profile.
///
/// Returns `None` if no adapter can be reached at all (headless environment,
/// missing drivers, sandboxed CI, ...); callers must handle that case with a
/// conservative fallback rather than erroring, since `config-cmd init
/// --interactive` should still produce a usable config offline.
fn detect_gpu() -> Option<DetectedGpu> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });

    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
        apply_limit_buckets: false,
    }))
    .ok()?;

    let info = adapter.get_info();
    let limits = adapter.limits();
    Some(DetectedGpu {
        device_type: info.device_type,
        max_buffer_size: limits.max_buffer_size,
        max_texture_dimension_2d: limits.max_texture_dimension_2d,
    })
}

/// Derive wizard defaults from a (possibly absent) detected GPU profile.
///
/// `sh_degree`, `views_per_step`, and `max_gaussians` are VRAM-bound, not
/// CPU-bound, in a 3D Gaussian Splatting trainer -- a 32-core workstation
/// with a 6 GB GPU cannot run degree-3 SH at 8 views/step, and a 2-core
/// machine with a 24 GB GPU is needlessly throttled by a CPU-only guess.
/// This derives all four from the GPU's reported capability instead:
/// `device_type` (discrete/integrated/software) sets `sh_degree`, and
/// `max_buffer_size` (the largest single allocation the adapter accepts --
/// not literal total VRAM, since wgpu has no portable API for that, but a
/// real signal that scales with GPU class) sets a VRAM budget used to pick
/// `views_per_step`, `image_size`, and `max_gaussians`.
///
/// Pure function: no I/O, so it can be exercised directly in tests.
fn wizard_hardware_profile(gpu: Option<DetectedGpu>) -> (u32, usize, u32, usize, String) {
    // (sh_degree, views_per_step, image_size, max_gaussians, description)
    match gpu {
        Some(g) => {
            let sh_degree = match g.device_type {
                wgpu::DeviceType::DiscreteGpu => 3,
                wgpu::DeviceType::IntegratedGpu | wgpu::DeviceType::VirtualGpu => 2,
                wgpu::DeviceType::Cpu | wgpu::DeviceType::Other => 1,
            };
            let image_size = g.max_texture_dimension_2d.clamp(512, 1024);
            let (views_per_step, max_gaussians) = match g.max_buffer_size {
                b if b >= 2_000_000_000 => (8, 1_000_000),
                b if b >= 1_000_000_000 => (4, 500_000),
                b if b >= 400_000_000 => (2, 200_000),
                _ => (1, 100_000),
            };
            let desc = format!(
                "{:?} adapter, max_buffer_size={} MB",
                g.device_type,
                g.max_buffer_size / (1024 * 1024)
            );
            (sh_degree, views_per_step, image_size, max_gaussians, desc)
        }
        None => (
            1,
            1,
            512,
            100_000,
            "no GPU adapter detected; using conservative offline defaults".to_string(),
        ),
    }
}

/// Generate and write a configuration based on hardware detection.
///
/// Instead of interactive stdin prompts, this wizard:
/// 1. Detects the available GPU via a real `wgpu` adapter query.
/// 2. Selects VRAM-bound defaults for `sh_degree`, `views_per_step`,
///    `image_size`, and `max_gaussians` from that adapter's reported
///    capability (falling back to conservative values if no adapter is
///    reachable).
/// 3. Prints explanations for each choice.
/// 4. Writes the annotated TOML to `output_path` or stdout.
pub fn run_config_init_wizard(output_path: Option<&Path>) -> Result<()> {
    run_config_init_wizard_with(output_path, &human_context())
}

/// [`run_config_init_wizard`], rendering according to `ctx`.
///
/// Under `--json` the detected hardware profile, the chosen settings and
/// the generated TOML are all fields of the result document; the banner and
/// the TOML body are printed only in human mode, so the JSON stream stays a
/// single parseable value. Under `--dry-run` nothing is written, exactly as
/// in [`run_config_init_with`].
///
/// # Errors
///
/// Returns an error when `output_path` already exists, when the config
/// cannot be serialized, or when the file cannot be written.
pub fn run_config_init_wizard_with(output_path: Option<&Path>, ctx: &CmdContext) -> Result<()> {
    run_wizard_with_profile(output_path, detect_gpu(), ctx)
}

/// Implementation of [`run_config_init_wizard`] parameterised over the
/// detected GPU profile, so tests can exercise both the "real GPU found"
/// and "no adapter reachable" branches deterministically.
fn run_wizard_with_profile(
    output_path: Option<&Path>,
    gpu: Option<DetectedGpu>,
    ctx: &CmdContext,
) -> Result<()> {
    if let Some(path) = output_path {
        if path.exists() {
            anyhow::bail!(
                "Output file already exists: {}. Remove it first, or re-run with a \
                 different -o/--output path.",
                path.display()
            );
        }
    }

    let (sh_degree, views_per_step, image_size, max_gaussians, hw_desc) =
        wizard_hardware_profile(gpu);

    // Build the base default config
    let mut config = ProjectConfig::default();
    config.training.init.sh_degree = sh_degree;
    config.training.views_per_step = views_per_step;
    config.training.image_size = image_size;
    config.training.density_control.max_gaussians = max_gaussians;

    // Serialize to TOML
    let toml_body =
        toml::to_string_pretty(&config).context("Failed to serialize wizard config to TOML")?;

    // Build a comment header that explains the choices
    let header = format!(
        "# OxiGAF Configuration — generated by hardware-detection wizard\n\
         # Hardware            : {hw_desc}\n\
         # sh_degree           : {sh_degree}\n\
         # views_per_step      : {views_per_step}\n\
         # image_size          : {image_size}\n\
         # max_gaussians       : {max_gaussians}\n\
         #\n\
         # Tip: increase sh_degree to 3 for best visual quality (needs more VRAM).\n\
         # Tip: lower views_per_step or max_gaussians if you run out of GPU memory.\n\n",
    );

    let full_output = format!("{header}{toml_body}");

    let written = write_config_file(output_path, &full_output, ctx)
        .with_context(|| "Failed to write wizard config")?;

    let payload = payload_for(ctx, || {
        wizard_payload(
            output_path,
            &full_output,
            &config,
            &hw_desc,
            (sh_degree, views_per_step, image_size, max_gaussians),
            ctx.dry_run,
        )
    })?;

    emit(
        ctx,
        "config-cmd init",
        payload,
        &optional_artifact(written),
        || {
            // Inform the user of the decisions being made
            println!("# OxiGAF Configuration Wizard");
            println!("# ============================");
            println!("# Hardware: {hw_desc}");
            println!(
                "# Using sh_degree = {sh_degree}, views_per_step = {views_per_step}, \
                 image_size = {image_size}, max_gaussians = {max_gaussians}."
            );
            println!();

            match output_path {
                None => print!("{full_output}"),
                Some(path) if written.is_some() => {
                    println!("Wizard config written to: {}", path.display())
                }
                Some(path) => println!(
                    "Would write wizard config to: {} (--dry-run)",
                    path.display()
                ),
            }
        },
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::fs;

    #[test]
    fn test_config_init_to_stdout() -> Result<()> {
        // Should not error — writes to stdout, which is fine in tests.
        run_config_init(None)
    }

    #[test]
    fn test_config_init_to_file() -> Result<()> {
        let tmp_dir = env::temp_dir();
        let path = tmp_dir.join("oxigaf_test_config_init.toml");
        fs::remove_file(&path).ok(); // start from a clean slate: init now refuses to overwrite
        run_config_init(Some(&path))?;
        assert!(path.exists());
        let content = fs::read_to_string(&path)?;
        assert!(content.contains("total_iterations"));
        fs::remove_file(&path).ok();
        Ok(())
    }

    #[test]
    fn test_config_init_refuses_existing_file() -> Result<()> {
        let tmp_dir = env::temp_dir();
        let path = tmp_dir.join("oxigaf_test_config_init_no_overwrite.toml");
        fs::write(
            &path,
            "# pre-existing tuned config, must not be clobbered\n",
        )?;
        let result = run_config_init(Some(&path));
        assert!(
            result.is_err(),
            "init must refuse to overwrite an existing file"
        );
        let content = fs::read_to_string(&path)?;
        assert!(
            content.contains("must not be clobbered"),
            "existing file must be left untouched"
        );
        fs::remove_file(&path).ok();
        Ok(())
    }

    #[test]
    fn test_config_validate_default_config() -> Result<()> {
        let tmp_dir = env::temp_dir();
        let path = tmp_dir.join("oxigaf_test_validate.toml");
        fs::remove_file(&path).ok(); // start from a clean slate: init now refuses to overwrite
                                     // Write default config to file, then validate it.
        run_config_init(Some(&path))?;
        run_config_validate(&path)?;
        fs::remove_file(&path).ok();
        Ok(())
    }

    #[test]
    fn test_config_validate_invalid_toml() {
        let tmp_dir = env::temp_dir();
        let path = tmp_dir.join("oxigaf_test_invalid.toml");
        fs::write(&path, "this is !! not valid [[[ toml").ok();
        let result = run_config_validate(&path);
        assert!(result.is_err());
        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_config_validate_missing_file() {
        let path = env::temp_dir().join("oxigaf_test_missing_config.toml");
        let result = run_config_validate(&path);
        assert!(result.is_err());
        let err_msg = result.err().map(|e| e.to_string()).unwrap_or_default();
        assert!(err_msg.contains("not found"));
    }

    #[test]
    fn test_config_show_default_config() -> Result<()> {
        let tmp_dir = env::temp_dir();
        let path = tmp_dir.join("oxigaf_test_show.toml");
        fs::remove_file(&path).ok(); // start from a clean slate: init now refuses to overwrite
        run_config_init(Some(&path))?;
        // Should not error.
        run_config_show(&path)?;
        fs::remove_file(&path).ok();
        Ok(())
    }

    #[test]
    fn test_config_show_missing_file() {
        let path = env::temp_dir().join("oxigaf_test_show_missing.toml");
        let result = run_config_show(&path);
        assert!(result.is_err());
    }

    #[test]
    fn test_config_roundtrip_serialize_deserialize() -> Result<()> {
        let default_config = ProjectConfig::default();
        let toml_string = toml::to_string_pretty(&default_config).context("serialize error")?;
        let parsed: ProjectConfig = toml::from_str(&toml_string).context("deserialize error")?;
        // Spot-check a few fields.
        assert_eq!(
            parsed.training.total_iterations,
            default_config.training.total_iterations
        );
        assert_eq!(
            parsed.training.init.sh_degree,
            default_config.training.init.sh_degree
        );
        assert_eq!(
            parsed.output.export_format,
            default_config.output.export_format
        );
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Wizard tests
    //
    // These exercise `run_wizard_with_profile(.., None, ..)` rather than the
    // public `run_config_init_wizard`, so they are deterministic and do not
    // spin up a real `wgpu` adapter (slow, and unpredictable on headless
    // CI). `wizard_hardware_profile`'s branch selection is covered directly
    // below; `detect_gpu`/`run_config_init_wizard` themselves are thin,
    // real-hardware wrappers around already-tested pure logic.
    // -----------------------------------------------------------------------

    #[test]
    fn test_wizard_to_stdout_no_error() -> Result<()> {
        // Should complete without error when writing to stdout.
        run_wizard_with_profile(None, None, &human_context())
    }

    #[test]
    fn test_wizard_creates_file() -> Result<()> {
        let path = env::temp_dir().join("oxigaf_wizard_test_create.toml");
        fs::remove_file(&path).ok();
        run_wizard_with_profile(Some(&path), None, &human_context())?;
        assert!(path.exists(), "Wizard should create the output file");
        fs::remove_file(&path).ok();
        Ok(())
    }

    #[test]
    fn test_wizard_refuses_existing_file() -> Result<()> {
        let path = env::temp_dir().join("oxigaf_wizard_test_no_overwrite.toml");
        fs::write(
            &path,
            "# pre-existing tuned config, must not be clobbered\n",
        )?;
        let result = run_wizard_with_profile(Some(&path), None, &human_context());
        assert!(
            result.is_err(),
            "wizard init must refuse to overwrite an existing file"
        );
        let content = fs::read_to_string(&path)?;
        assert!(
            content.contains("must not be clobbered"),
            "existing file must be left untouched"
        );
        fs::remove_file(&path).ok();
        Ok(())
    }

    #[test]
    fn test_wizard_output_is_valid_toml() -> Result<()> {
        let path = env::temp_dir().join("oxigaf_wizard_test_valid_toml.toml");
        fs::remove_file(&path).ok();
        run_wizard_with_profile(Some(&path), None, &human_context())?;
        let content = fs::read_to_string(&path).context("read wizard output")?;
        // Strip comment lines (TOML parsers do handle # comments but let's verify parse)
        let _parsed: toml::Value =
            toml::from_str(&content).context("wizard output is not valid TOML")?;
        fs::remove_file(&path).ok();
        Ok(())
    }

    #[test]
    fn test_wizard_output_contains_expected_keys() -> Result<()> {
        let path = env::temp_dir().join("oxigaf_wizard_test_keys.toml");
        fs::remove_file(&path).ok();
        run_wizard_with_profile(Some(&path), None, &human_context())?;
        let content = fs::read_to_string(&path).context("read wizard output")?;
        // Check that the key training fields are present
        assert!(
            content.contains("total_iterations"),
            "Missing total_iterations in wizard output"
        );
        assert!(
            content.contains("sh_degree"),
            "Missing sh_degree in wizard output"
        );
        assert!(
            content.contains("views_per_step"),
            "Missing views_per_step in wizard output"
        );
        fs::remove_file(&path).ok();
        Ok(())
    }

    // -----------------------------------------------------------------------
    // wizard_hardware_profile: VRAM-bound tier selection (pure, hermetic)
    //
    // Regression coverage for: the wizard used to derive `sh_degree` and
    // `views_per_step` from `available_parallelism()` (CPU cores), which
    // are both VRAM-bound properties of a 3DGS trainer, not CPU-bound ones.
    // -----------------------------------------------------------------------

    #[test]
    fn wizard_profile_no_adapter_is_conservative() {
        let (sh_degree, views_per_step, image_size, max_gaussians, desc) =
            wizard_hardware_profile(None);
        assert_eq!(sh_degree, 1);
        assert_eq!(views_per_step, 1);
        assert_eq!(image_size, 512);
        assert_eq!(max_gaussians, 100_000);
        assert!(desc.contains("no GPU adapter"));
    }

    #[test]
    fn wizard_profile_discrete_high_vram_gpu_gets_top_tier() {
        let gpu = DetectedGpu {
            device_type: wgpu::DeviceType::DiscreteGpu,
            max_buffer_size: 4 * 1024 * 1024 * 1024, // 4 GiB
            max_texture_dimension_2d: 16384,
        };
        let (sh_degree, views_per_step, image_size, max_gaussians, _) =
            wizard_hardware_profile(Some(gpu));
        assert_eq!(sh_degree, 3);
        assert_eq!(views_per_step, 8);
        assert_eq!(image_size, 1024);
        assert_eq!(max_gaussians, 1_000_000);
    }

    #[test]
    fn wizard_profile_integrated_low_vram_gpu_gets_low_tier() {
        let gpu = DetectedGpu {
            device_type: wgpu::DeviceType::IntegratedGpu,
            max_buffer_size: 256 * 1024 * 1024, // 256 MiB
            max_texture_dimension_2d: 8192,
        };
        let (sh_degree, views_per_step, _, max_gaussians, _) = wizard_hardware_profile(Some(gpu));
        assert_eq!(sh_degree, 2); // integrated caps at 2, regardless of VRAM tier
        assert_eq!(views_per_step, 1);
        assert_eq!(max_gaussians, 100_000);
    }

    #[test]
    fn wizard_profile_cpu_software_adapter_gets_lowest_sh_degree() {
        let gpu = DetectedGpu {
            device_type: wgpu::DeviceType::Cpu,
            max_buffer_size: 4 * 1024 * 1024 * 1024,
            max_texture_dimension_2d: 8192,
        };
        let (sh_degree, ..) = wizard_hardware_profile(Some(gpu));
        assert_eq!(
            sh_degree, 1,
            "a CPU/software adapter must not select high SH degree"
        );
    }

    #[test]
    fn wizard_profile_image_size_is_clamped_to_512_1024() {
        let tiny_texture_gpu = DetectedGpu {
            device_type: wgpu::DeviceType::DiscreteGpu,
            max_buffer_size: 4 * 1024 * 1024 * 1024,
            max_texture_dimension_2d: 256, // below the 512 floor
        };
        let (_, _, image_size, ..) = wizard_hardware_profile(Some(tiny_texture_gpu));
        assert_eq!(image_size, 512);

        let huge_texture_gpu = DetectedGpu {
            device_type: wgpu::DeviceType::DiscreteGpu,
            max_buffer_size: 4 * 1024 * 1024 * 1024,
            max_texture_dimension_2d: 16384, // above the 1024 ceiling
        };
        let (_, _, image_size, ..) = wizard_hardware_profile(Some(huge_texture_gpu));
        assert_eq!(image_size, 1024);
    }

    // -----------------------------------------------------------------------
    // JSON entry points: `oxigaf --json config-cmd …` must produce a single
    // machine-readable document instead of TOML/plain text on stdout.
    // -----------------------------------------------------------------------

    #[test]
    fn init_payload_carries_toml_and_parsed_config() -> Result<()> {
        let config = ProjectConfig::default();
        let toml_text = toml::to_string_pretty(&config).context("serialize")?;

        // Without --output the TOML body lives in the document rather than
        // being printed, so the stream stays one parseable value.
        let payload = init_payload(None, &toml_text, &config, false)?;
        assert_eq!(payload["action"], "init");
        assert_eq!(payload["interactive"], false);
        assert!(payload["output"].is_null());
        assert_eq!(payload["toml"], toml_text.as_str());
        assert_eq!(
            payload["config"]["training"]["total_iterations"],
            config.training.total_iterations
        );
        assert_eq!(
            payload["config"]["training"]["init"]["sh_degree"],
            config.training.init.sh_degree
        );

        // With --output the same document names the file it wrote.
        let path = env::temp_dir().join("oxigaf_json_init_payload.toml");
        let payload = init_payload(Some(&path), &toml_text, &config, false)?;
        assert_eq!(payload["output"], path.display().to_string().as_str());
        Ok(())
    }

    #[test]
    fn validate_and_show_payloads_report_path_and_config() -> Result<()> {
        let path = env::temp_dir().join("oxigaf_json_show_payload.toml");
        let mut config = ProjectConfig::default();
        config.training.image_size = 768;

        let payload = validate_payload(&path);
        assert_eq!(payload["action"], "validate");
        assert_eq!(payload["valid"], true);
        assert_eq!(payload["path"], path.display().to_string().as_str());

        let payload = show_payload(&path, &config)?;
        assert_eq!(payload["action"], "show");
        assert_eq!(payload["config"]["training"]["image_size"], 768);
        Ok(())
    }

    #[test]
    fn wizard_payload_reports_the_hardware_it_acted_on() -> Result<()> {
        let config = ProjectConfig::default();
        let payload = wizard_payload(
            None,
            "# toml body\n",
            &config,
            "DiscreteGpu adapter, max_buffer_size=4096 MB",
            (3, 8, 1024, 1_000_000),
            false,
        )?;
        assert_eq!(payload["action"], "init");
        assert_eq!(payload["interactive"], true);
        assert!(
            payload["hardware"]
                .as_str()
                .unwrap_or_default()
                .contains("DiscreteGpu"),
            "hardware: {}",
            payload["hardware"]
        );
        assert_eq!(payload["settings"]["sh_degree"], 3);
        assert_eq!(payload["settings"]["views_per_step"], 8);
        assert_eq!(payload["settings"]["image_size"], 1024);
        assert_eq!(payload["settings"]["max_gaussians"], 1_000_000);
        assert_eq!(payload["toml"], "# toml body\n");
        Ok(())
    }

    #[test]
    fn json_mode_still_writes_a_real_toml_artifact() -> Result<()> {
        // The document goes to stdout; the file on disk must remain the
        // TOML config a later `--config` run can actually load.
        let path = env::temp_dir().join("oxigaf_json_init_writes_toml.toml");
        fs::remove_file(&path).ok();
        run_config_init_with(Some(&path), &json_context())?;
        let written = fs::read_to_string(&path)?;
        let parsed: ProjectConfig =
            toml::from_str(&written).context("JSON-mode init must still write valid TOML")?;
        assert_eq!(
            parsed.training.total_iterations,
            ProjectConfig::default().training.total_iterations
        );
        fs::remove_file(&path).ok();
        Ok(())
    }

    #[test]
    fn json_mode_wizard_still_writes_a_real_toml_artifact() -> Result<()> {
        let path = env::temp_dir().join("oxigaf_json_wizard_writes_toml.toml");
        fs::remove_file(&path).ok();
        run_wizard_with_profile(Some(&path), None, &json_context())?;
        let written = fs::read_to_string(&path)?;
        let _parsed: toml::Value =
            toml::from_str(&written).context("JSON-mode wizard must still write valid TOML")?;
        assert!(
            written.contains("# OxiGAF Configuration"),
            "the annotated header belongs in the file, not only on stdout"
        );
        fs::remove_file(&path).ok();
        Ok(())
    }

    #[test]
    fn json_entry_points_run_every_subcommand() -> Result<()> {
        // End-to-end through `commands::emit` for all three subcommands, in
        // both renderings, so neither path can regress into an error.
        let path = env::temp_dir().join("oxigaf_json_entry_points.toml");
        fs::remove_file(&path).ok();
        run_config_cmd_json(ConfigCmdSubcommand::Init {
            output: Some(path.clone()),
            interactive: false,
        })?;
        run_config_cmd_json(ConfigCmdSubcommand::Validate { path: path.clone() })?;
        run_config_cmd_json(ConfigCmdSubcommand::Show { path: path.clone() })?;
        run_config_cmd(ConfigCmdSubcommand::Validate { path: path.clone() })?;
        run_config_cmd(ConfigCmdSubcommand::Show { path: path.clone() })?;
        fs::remove_file(&path).ok();
        Ok(())
    }

    #[test]
    fn dry_run_context_writes_nothing_and_says_so() -> Result<()> {
        // `--dry-run` is a property of the context, so an entry point that
        // accepts an arbitrary `CmdContext` has to honour it the way every
        // handler going through `commands::prepare_output` does: rehearse,
        // report, write nothing.
        let path = env::temp_dir().join("oxigaf_dry_run_init.toml");
        fs::remove_file(&path).ok();
        let rehearsal = CmdContext::new(Verbosity::Quiet, true, true);
        run_config_init_with(Some(&path), &rehearsal)?;
        assert!(!path.exists(), "--dry-run must not create the output file");

        let wizard_path = env::temp_dir().join("oxigaf_dry_run_wizard.toml");
        fs::remove_file(&wizard_path).ok();
        run_wizard_with_profile(Some(&wizard_path), None, &rehearsal)?;
        assert!(
            !wizard_path.exists(),
            "--dry-run must not create the wizard output file"
        );

        // The document says the write was a rehearsal, and claims no
        // artifact for a file that does not exist.
        let config = ProjectConfig::default();
        let payload = init_payload(Some(&path), "# body\n", &config, true)?;
        assert_eq!(payload["dry_run"], true);
        assert!(optional_artifact(None).is_empty());
        Ok(())
    }

    #[test]
    fn json_entry_points_still_refuse_to_overwrite() {
        // The overwrite guard is a property of the command, not of the
        // rendering: --json must not turn it into a silent clobber.
        let path = env::temp_dir().join("oxigaf_json_no_overwrite.toml");
        fs::write(
            &path,
            "# pre-existing tuned config, must not be clobbered\n",
        )
        .ok();
        let result = run_config_cmd_json(ConfigCmdSubcommand::Init {
            output: Some(path.clone()),
            interactive: false,
        });
        assert!(result.is_err(), "--json init must not overwrite");
        let content = fs::read_to_string(&path).unwrap_or_default();
        assert!(
            content.contains("must not be clobbered"),
            "existing file must be left untouched"
        );
        fs::remove_file(&path).ok();
    }
}
