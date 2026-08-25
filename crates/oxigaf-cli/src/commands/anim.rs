//! `oxigaf anim` — per-frame Gaussian animation sequences.
//!
//! Thin glue over [`crate::animation_export`]: every subcommand loads an
//! animation JSON file, applies exactly one of the module's transforms, and
//! writes the result back out (or reports statistics without writing).

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Args, Subcommand};
use serde_json::json;

use crate::animation_export::{
    compute_animation_stats, compute_frame_stats, concatenate_animations, export_animation_json,
    format_animation_summary, load_animation_json, loop_animation, resample_animation,
    reverse_animation, subsample_animation, trim_animation, AnimExportConfig, AnimationSequence,
    AnimationStats,
};
use crate::commands::{emit, prepare_output, CmdContext};

/// `oxigaf anim <command>`.
#[derive(Debug, Args)]
pub struct AnimArgs {
    #[command(subcommand)]
    pub command: AnimCommand,
}

/// Animation sequence subcommands.
#[derive(Debug, Subcommand)]
pub enum AnimCommand {
    /// Print a one-line summary and aggregate statistics for an animation.
    Info {
        /// Animation sequence JSON file.
        input: PathBuf,

        /// Also print per-frame statistics.
        #[arg(long)]
        per_frame: bool,
    },

    /// Resample an animation to a new frame rate.
    Resample {
        /// Input animation JSON.
        input: PathBuf,
        /// Output animation JSON.
        output: PathBuf,
        /// Target frames per second (must be > 0).
        #[arg(long)]
        fps: f32,
        #[command(flatten)]
        write: WriteOpts,
    },

    /// Keep only frames `[start, end)` of an animation.
    Trim {
        /// Input animation JSON.
        input: PathBuf,
        /// Output animation JSON.
        output: PathBuf,
        /// First frame to keep (inclusive).
        #[arg(long, default_value = "0")]
        start: usize,
        /// Last frame to keep (exclusive).
        #[arg(long)]
        end: usize,
        #[command(flatten)]
        write: WriteOpts,
    },

    /// Reverse the frame order of an animation.
    Reverse {
        /// Input animation JSON.
        input: PathBuf,
        /// Output animation JSON.
        output: PathBuf,
        #[command(flatten)]
        write: WriteOpts,
    },

    /// Repeat an animation end-to-end.
    Loop {
        /// Input animation JSON.
        input: PathBuf,
        /// Output animation JSON.
        output: PathBuf,
        /// Number of repetitions (must be >= 1).
        #[arg(long, default_value = "2")]
        repeats: usize,
        #[command(flatten)]
        write: WriteOpts,
    },

    /// Keep every N-th frame of an animation.
    Subsample {
        /// Input animation JSON.
        input: PathBuf,
        /// Output animation JSON.
        output: PathBuf,
        /// Keep every `stride`-th frame (must be >= 1).
        #[arg(long, default_value = "2")]
        stride: usize,
        #[command(flatten)]
        write: WriteOpts,
    },

    /// Concatenate two or more animations into one sequence.
    Concat {
        /// Output animation JSON.
        #[arg(short, long)]
        output: PathBuf,
        /// Input animation JSON files, joined left to right (at least two).
        #[arg(required = true, num_args = 2..)]
        inputs: Vec<PathBuf>,
        #[command(flatten)]
        write: WriteOpts,
    },
}

/// Options shared by every subcommand that writes an animation file.
#[derive(Debug, Args, Clone)]
pub struct WriteOpts {
    /// Drop spherical-harmonics coefficients from the written file.
    #[arg(long)]
    pub no_sh: bool,

    /// Decimal places retained for floating-point fields.
    #[arg(long, default_value = "6")]
    pub precision: usize,

    /// Overwrite the output file if it already exists.
    #[arg(long)]
    pub force: bool,
}

impl WriteOpts {
    fn to_config(&self, fps: f32) -> AnimExportConfig {
        AnimExportConfig {
            fps,
            include_sh: !self.no_sh,
            precision: self.precision,
        }
    }
}

/// Run the `anim` family.
///
/// # Errors
///
/// Returns an error when an input file cannot be read or parsed, when the
/// requested transform rejects its parameters, or when the output file
/// exists without `--force`.
pub fn run(args: AnimArgs, ctx: CmdContext) -> Result<()> {
    match args.command {
        AnimCommand::Info { input, per_frame } => cmd_info(&input, per_frame, &ctx),
        AnimCommand::Resample {
            input,
            output,
            fps,
            write,
        } => {
            if !(fps.is_finite() && fps > 0.0) {
                anyhow::bail!("--fps must be a positive, finite number (got {fps})");
            }
            let sequence = load(&input)?;
            let result = resample_animation(sequence, fps)?;
            finish("anim resample", &input, &output, result, &write, &ctx)
        }
        AnimCommand::Trim {
            input,
            output,
            start,
            end,
            write,
        } => {
            if start >= end {
                anyhow::bail!("--start ({start}) must be strictly less than --end ({end})");
            }
            let sequence = load(&input)?;
            let result = trim_animation(sequence, start, end)?;
            finish("anim trim", &input, &output, result, &write, &ctx)
        }
        AnimCommand::Reverse {
            input,
            output,
            write,
        } => {
            let sequence = load(&input)?;
            let result = reverse_animation(sequence);
            finish("anim reverse", &input, &output, result, &write, &ctx)
        }
        AnimCommand::Loop {
            input,
            output,
            repeats,
            write,
        } => {
            if repeats == 0 {
                anyhow::bail!("--repeats must be at least 1");
            }
            let sequence = load(&input)?;
            let result = loop_animation(sequence, repeats)?;
            finish("anim loop", &input, &output, result, &write, &ctx)
        }
        AnimCommand::Subsample {
            input,
            output,
            stride,
            write,
        } => {
            if stride == 0 {
                anyhow::bail!("--stride must be at least 1");
            }
            let sequence = load(&input)?;
            let result = subsample_animation(sequence, stride)?;
            finish("anim subsample", &input, &output, result, &write, &ctx)
        }
        AnimCommand::Concat {
            output,
            inputs,
            write,
        } => {
            let mut iter = inputs.iter();
            let first = iter
                .next()
                .ok_or_else(|| anyhow::anyhow!("at least two input animations are required"))?;
            let mut merged = load(first)?;
            for path in iter {
                let next = load(path)?;
                merged = concatenate_animations(merged, next)?;
            }
            finish("anim concat", first, &output, merged, &write, &ctx)
        }
    }
}

fn load(path: &std::path::Path) -> Result<AnimationSequence> {
    load_animation_json(path)
        .with_context(|| format!("Failed to load animation: {}", path.display()))
}

fn stats_json(stats: &AnimationStats) -> serde_json::Value {
    json!({
        "n_frames": stats.n_frames,
        "n_gaussians": stats.n_gaussians,
        "fps": stats.fps,
        "duration_ms": stats.duration_ms,
        "mean_opacity": stats.mean_opacity,
        "opacity_std": stats.opacity_std,
        "position_drift": stats.position_drift,
    })
}

fn cmd_info(input: &std::path::Path, per_frame: bool, ctx: &CmdContext) -> Result<()> {
    let sequence = load(input)?;
    let stats = compute_animation_stats(&sequence);
    let summary = format_animation_summary(&sequence);

    let frames: Vec<serde_json::Value> = if per_frame {
        sequence
            .frames
            .iter()
            .map(|frame| {
                let fs = compute_frame_stats(frame);
                json!({
                    "frame_idx": fs.frame_idx,
                    "n_gaussians": fs.n_gaussians,
                    "mean_opacity": fs.mean_opacity,
                    "mean_scale": fs.mean_scale,
                    "position_centroid": fs.position_centroid,
                })
            })
            .collect()
    } else {
        Vec::new()
    };

    let mut payload = stats_json(&stats);
    if per_frame {
        if let Some(map) = payload.as_object_mut() {
            map.insert("frames".to_string(), serde_json::Value::Array(frames));
        }
    }

    emit(ctx, "anim info", payload, &[], || {
        println!("{summary}");
        println!("  mean opacity  : {:.4}", stats.mean_opacity);
        println!("  opacity stddev: {:.4}", stats.opacity_std);
        println!("  position drift: {:.6}", stats.position_drift);
        if per_frame {
            println!();
            println!("  frame  gaussians  mean_opacity  mean_scale");
            for frame in &sequence.frames {
                let fs = compute_frame_stats(frame);
                println!(
                    "  {:>5}  {:>9}  {:>12.4}  {:>10.6}",
                    fs.frame_idx, fs.n_gaussians, fs.mean_opacity, fs.mean_scale
                );
            }
        }
    });
    Ok(())
}

fn finish(
    command: &str,
    input: &std::path::Path,
    output: &std::path::Path,
    sequence: AnimationSequence,
    write: &WriteOpts,
    ctx: &CmdContext,
) -> Result<()> {
    let stats = compute_animation_stats(&sequence);
    let payload = {
        let mut value = stats_json(&stats);
        if let Some(map) = value.as_object_mut() {
            map.insert("input".to_string(), json!(input.display().to_string()));
            map.insert("output".to_string(), json!(output.display().to_string()));
        }
        value
    };

    if !prepare_output(ctx, output, write.force)? {
        emit(ctx, command, payload, &[], || {
            println!(
                "[dry-run] would write {} ({} frames, {} Gaussians)",
                output.display(),
                stats.n_frames,
                stats.n_gaussians
            );
        });
        return Ok(());
    }

    let config = write.to_config(sequence.meta.fps);
    export_animation_json(&sequence, output, &config)
        .with_context(|| format!("Failed to write animation: {}", output.display()))?;

    let summary = format_animation_summary(&sequence);
    emit(ctx, command, payload, &[("animation", output)], || {
        println!("{summary}");
        println!("Wrote {}", output.display());
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::animation_export::AnimationFrame;
    use crate::verbosity::Verbosity;

    fn sample_sequence(n: usize) -> AnimationSequence {
        let frames: Vec<AnimationFrame> = (0..n)
            .map(|i| {
                AnimationFrame::from_positions_only(
                    i,
                    i as f64 * 1000.0 / 30.0,
                    vec![i as f32, 0.0, 0.0],
                )
            })
            .collect();
        AnimationSequence::new(frames, 30.0).expect("sample sequence is always valid")
    }

    #[test]
    fn write_opts_map_onto_export_config() {
        let opts = WriteOpts {
            no_sh: true,
            precision: 3,
            force: false,
        };
        let config = opts.to_config(24.0);
        assert!(!config.include_sh);
        assert_eq!(config.precision, 3);
        assert!((config.fps - 24.0).abs() < f32::EPSILON);
    }

    #[test]
    fn finish_honours_dry_run_and_writes_nothing() {
        let ctx = CmdContext::new(Verbosity::Quiet, true, true);
        let out = std::env::temp_dir().join("oxigaf_anim_dry_run.json");
        let _ = std::fs::remove_file(&out);
        let opts = WriteOpts {
            no_sh: false,
            precision: 6,
            force: false,
        };
        let res = finish(
            "anim reverse",
            std::path::Path::new("in.json"),
            &out,
            sample_sequence(3),
            &opts,
            &ctx,
        );
        assert!(res.is_ok());
        assert!(!out.exists(), "dry-run must not create the output file");
    }

    #[test]
    fn finish_writes_and_round_trips() {
        let ctx = CmdContext::new(Verbosity::Quiet, true, false);
        let out = std::env::temp_dir().join("oxigaf_anim_round_trip.json");
        let _ = std::fs::remove_file(&out);
        let opts = WriteOpts {
            no_sh: false,
            precision: 6,
            force: true,
        };
        finish(
            "anim reverse",
            std::path::Path::new("in.json"),
            &out,
            sample_sequence(4),
            &opts,
            &ctx,
        )
        .expect("write should succeed");
        let loaded = load(&out).expect("written file must reload");
        assert_eq!(loaded.frames.len(), 4);
        let _ = std::fs::remove_file(&out);
    }
}
