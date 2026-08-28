//! `oxigaf preset` — named training hyper-parameter presets.
//!
//! Glue over [`crate::config_presets`]. The same presets back the
//! `oxigaf train --profile <name>` flag, so `preset show` and
//! `preset apply` describe exactly what `--profile` will do to a run.
//!
//! This lives as its own top-level family rather than under `config-cmd`
//! because [`crate::cli::ConfigCmdSubcommand`] is matched exhaustively in
//! `config_cmd.rs`; a new variant there would be a breaking change to a
//! module this command does not own.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Args, Subcommand};
use serde_json::json;

use crate::commands::{emit, prepare_output, CmdContext};
use crate::config::ProjectConfig;
use crate::config_presets::{apply_overrides, TrainingPreset, TrainingPresetName};

/// `oxigaf preset <command>`.
#[derive(Debug, Args)]
pub struct PresetArgs {
    #[command(subcommand)]
    pub command: PresetCommand,
}

/// Preset subcommands.
#[derive(Debug, Subcommand)]
pub enum PresetCommand {
    /// List every built-in preset with its one-line description.
    List,

    /// Print every hyper-parameter of one preset.
    Show {
        /// Preset name (see `oxigaf preset list`).
        name: String,

        /// Apply `key=value` overrides before printing.
        #[arg(long = "set", value_name = "KEY=VALUE", num_args = 1..)]
        overrides: Vec<String>,
    },

    /// Report the parameters that differ between two presets.
    Diff {
        /// First preset name.
        first: String,
        /// Second preset name.
        second: String,
    },

    /// Write a project config TOML built from a preset.
    Apply {
        /// Preset name.
        name: String,

        /// Start from this config file instead of the built-in defaults.
        #[arg(long)]
        base: Option<PathBuf>,

        /// Write the resulting TOML here. Omit to print it on stdout.
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Apply `key=value` preset overrides before conversion.
        #[arg(long = "set", value_name = "KEY=VALUE", num_args = 1..)]
        overrides: Vec<String>,

        /// Overwrite the output file if it exists.
        #[arg(long)]
        force: bool,
    },
}

/// Resolve a preset name, applying any `--set key=value` overrides.
///
/// # Errors
///
/// Returns an error for an unknown preset name or a malformed override.
pub fn resolve(name: &str, overrides: &[String]) -> Result<TrainingPreset> {
    let preset_name = TrainingPresetName::from_str(name).with_context(|| {
        let known: Vec<&str> = TrainingPresetName::all()
            .iter()
            .map(TrainingPresetName::as_str)
            .collect();
        format!(
            "Unknown preset {name:?}; available presets: {}",
            known.join(", ")
        )
    })?;
    let base = TrainingPreset::get(&preset_name);
    if overrides.is_empty() {
        return Ok(base);
    }
    let refs: Vec<&str> = overrides.iter().map(String::as_str).collect();
    Ok(apply_overrides(&base, &refs)?)
}

fn preset_json(preset: &TrainingPreset) -> serde_json::Value {
    json!({
        "name": preset.name.as_str(),
        "description": preset.description,
        "num_iterations": preset.num_iterations,
        "warmup_iterations": preset.warmup_iterations,
        "position_lr": preset.position_lr,
        "color_lr": preset.color_lr,
        "opacity_lr": preset.opacity_lr,
        "scale_lr": preset.scale_lr,
        "rotation_lr": preset.rotation_lr,
        "densify_from_iter": preset.densify_from_iter,
        "densify_until_iter": preset.densify_until_iter,
        "densify_grad_threshold": preset.densify_grad_threshold,
        "opacity_reset_interval": preset.opacity_reset_interval,
        "max_gaussians": preset.max_gaussians,
        "sh_degree": preset.sh_degree,
        "image_width": preset.image_width,
        "image_height": preset.image_height,
        "checkpoint_every": preset.checkpoint_every,
        "keep_last_n_checkpoints": preset.keep_last_n_checkpoints,
        "log_every": preset.log_every,
        "tensorboard_enabled": preset.tensorboard_enabled,
        "lambda_l1": preset.lambda_l1,
        "lambda_ssim": preset.lambda_ssim,
        "lambda_lpips": preset.lambda_lpips,
        "lambda_opacity": preset.lambda_opacity,
        "lambda_scale": preset.lambda_scale,
        "batch_size": preset.batch_size,
        "num_workers": preset.num_workers,
        "estimated_minutes": preset.estimated_minutes(),
        "estimated_vram_mb": preset.estimated_vram_mb(),
        "quality_score": preset.quality_score(),
    })
}

/// Run the `preset` family.
///
/// # Errors
///
/// Returns an error for an unknown preset, a malformed override, or a
/// config file that cannot be read or written.
pub fn run(args: PresetArgs, ctx: CmdContext) -> Result<()> {
    match args.command {
        PresetCommand::List => {
            let descriptions = TrainingPreset::list_descriptions();
            let payload = json!({
                "presets": descriptions
                    .iter()
                    .map(|(name, description)| json!({
                        "name": name,
                        "description": description,
                    }))
                    .collect::<Vec<_>>(),
            });
            emit(&ctx, "preset list", payload, &[], || {
                for (name, description) in &descriptions {
                    println!("{name:<12} {description}");
                }
            });
            Ok(())
        }

        PresetCommand::Show { name, overrides } => {
            let preset = resolve(&name, &overrides)?;
            let payload = preset_json(&preset);
            emit(&ctx, "preset show", payload, &[], || {
                println!("{}", preset.format_table());
            });
            Ok(())
        }

        PresetCommand::Diff { first, second } => {
            let a = resolve(&first, &[])?;
            let b = resolve(&second, &[])?;
            let diffs = TrainingPreset::diff(&a, &b);
            let payload = json!({
                "a": a.name.as_str(),
                "b": b.name.as_str(),
                "differences": diffs
                    .iter()
                    .map(|entry| json!({
                        "parameter": entry.parameter,
                        "value_a": entry.value_a,
                        "value_b": entry.value_b,
                    }))
                    .collect::<Vec<_>>(),
            });
            emit(&ctx, "preset diff", payload, &[], || {
                if diffs.is_empty() {
                    println!("{} and {} are identical", a.name, b.name);
                    return;
                }
                println!("{:<28} {:<20} {:<20}", "parameter", first, second);
                for entry in &diffs {
                    println!(
                        "{:<28} {:<20} {:<20}",
                        entry.parameter, entry.value_a, entry.value_b
                    );
                }
            });
            Ok(())
        }

        PresetCommand::Apply {
            name,
            base,
            output,
            overrides,
            force,
        } => cmd_apply(
            &name,
            base.as_deref(),
            output.as_deref(),
            &overrides,
            force,
            &ctx,
        ),
    }
}

fn cmd_apply(
    name: &str,
    base: Option<&Path>,
    output: Option<&Path>,
    overrides: &[String],
    force: bool,
    ctx: &CmdContext,
) -> Result<()> {
    let preset = resolve(name, overrides)?;

    // Read the base file directly rather than through
    // `config::load_hierarchical_config`: `preset apply` must be a pure
    // function of (base file, preset, overrides), with no environment
    // variables or the user-level config silently folded in.
    let mut config = match base {
        Some(path) => {
            let text = std::fs::read_to_string(path)
                .with_context(|| format!("Failed to read base config: {}", path.display()))?;
            toml::from_str::<ProjectConfig>(&text)
                .with_context(|| format!("Failed to parse base config: {}", path.display()))?
        }
        None => ProjectConfig::default(),
    };
    preset.apply_to(&mut config);

    let toml_text = toml::to_string_pretty(&config)
        .context("Failed to serialise the project config as TOML")?;

    let payload = {
        let mut value = preset_json(&preset);
        if let Some(map) = value.as_object_mut() {
            map.insert("config_toml".to_string(), json!(toml_text));
            map.insert(
                "output".to_string(),
                json!(output.map(|p| p.display().to_string())),
            );
        }
        value
    };

    let mut artifacts: Vec<(&str, &Path)> = Vec::new();
    if let Some(path) = output {
        if prepare_output(ctx, path, force)? {
            std::fs::write(path, toml_text.as_bytes())
                .with_context(|| format!("Failed to write config: {}", path.display()))?;
            artifacts.push(("config", path));
        }
    }

    emit(ctx, "preset apply", payload, &artifacts, || match output {
        Some(path) => {
            if ctx.dry_run {
                println!("[dry-run] would write {}", path.display());
            } else {
                println!("Wrote {}", path.display());
            }
        }
        None => println!("{toml_text}"),
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_preset_names_are_reported_with_the_valid_set() {
        let error = resolve("does-not-exist", &[]).expect_err("unknown preset must fail");
        let text = format!("{error:#}");
        assert!(text.contains("available presets"), "got: {text}");
    }

    #[test]
    fn overrides_are_applied_in_order() {
        let preset = resolve(
            "quick",
            &["num_iterations=1234".to_string(), "sh_degree=1".to_string()],
        )
        .expect("valid overrides");
        assert_eq!(preset.num_iterations, 1234);
        assert_eq!(preset.sh_degree, 1);
    }

    #[test]
    fn malformed_overrides_are_rejected() {
        assert!(resolve("quick", &["not-a-pair".to_string()]).is_err());
        assert!(resolve("quick", &["unknown_field=1".to_string()]).is_err());
    }

    #[test]
    fn apply_writes_a_parsable_toml_config() {
        let out = std::env::temp_dir().join("oxigaf_preset_apply.toml");
        let _ = std::fs::remove_file(&out);
        let ctx = CmdContext::new(crate::verbosity::Verbosity::Quiet, true, false);
        cmd_apply("quick", None, Some(&out), &[], true, &ctx).expect("apply succeeds");

        let text = std::fs::read_to_string(&out).expect("config written");
        let parsed: ProjectConfig = toml::from_str(&text).expect("written TOML must reparse");
        let preset = resolve("quick", &[]).expect("preset");
        assert_eq!(parsed.training.total_iterations, preset.num_iterations);

        let _ = std::fs::remove_file(&out);
    }
}
