//! `oxigaf scene` — whole-scene geometry operations on model files.
//!
//! Currently wires [`crate::cloud_registration`] (`scene register`). The
//! family is the intended home for the remaining scene-level tools
//! (merge, optimize, filter, dedup, compress, LOD, streaming chunks and the
//! geometry helpers), each as one more variant of [`SceneCommand`].

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Args, Subcommand};
use serde_json::json;

use crate::cloud_registration::{
    apply_registration_transform, compute_initial_rmse, compute_registration_stats,
    format_registration_result, register_point_clouds, subsample_positions, RegistrationConfig,
};
use crate::commands::{emit, load_positions, prepare_output, CmdContext};

/// `oxigaf scene <command>`.
#[derive(Debug, Args)]
pub struct SceneArgs {
    #[command(subcommand)]
    pub command: SceneCommand,
}

/// Scene-level subcommands.
#[derive(Debug, Subcommand)]
pub enum SceneCommand {
    /// Align one scene onto another with iterative closest point (ICP).
    Register(RegisterArgs),
}

/// Arguments for `oxigaf scene register`.
#[derive(Debug, Args)]
pub struct RegisterArgs {
    /// Scene that is moved onto the target.
    #[arg(long)]
    pub source: PathBuf,

    /// Reference scene that stays fixed.
    #[arg(long)]
    pub target: PathBuf,

    /// Write the transformed source scene here (PLY). Omit to only report
    /// the estimated transform.
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Maximum ICP iterations.
    #[arg(long, default_value = "100")]
    pub max_iterations: usize,

    /// Stop when the RMSE improvement drops below this value.
    #[arg(long, default_value = "1e-5")]
    pub tolerance: f32,

    /// Reject correspondences farther apart than this distance.
    #[arg(long)]
    pub max_correspondence_dist: Option<f32>,

    /// Also estimate a uniform scale factor.
    #[arg(long)]
    pub allow_scale: bool,

    /// Use every N-th source point during ICP (1 = use all).
    #[arg(long, default_value = "1")]
    pub subsample: usize,

    /// Discard this fraction of the worst correspondences each iteration.
    #[arg(long, default_value = "0.0")]
    pub outlier_fraction: f32,

    /// Overwrite the output file if it exists.
    #[arg(long)]
    pub force: bool,
}

/// Run the `scene` family.
///
/// # Errors
///
/// Returns an error when a scene cannot be loaded, when ICP rejects its
/// inputs (empty cloud, no correspondences), or when the output file exists
/// without `--force`.
pub fn run(args: SceneArgs, ctx: CmdContext) -> Result<()> {
    match args.command {
        SceneCommand::Register(register_args) => cmd_register(register_args, &ctx),
    }
}

fn cmd_register(args: RegisterArgs, ctx: &CmdContext) -> Result<()> {
    if args.max_iterations == 0 {
        anyhow::bail!("--max-iterations must be at least 1");
    }
    if args.subsample == 0 {
        anyhow::bail!("--subsample must be at least 1");
    }
    if !(0.0..1.0).contains(&args.outlier_fraction) {
        anyhow::bail!(
            "--outlier-fraction must be within [0.0, 1.0) (got {})",
            args.outlier_fraction
        );
    }

    let source = load_positions(&args.source)?;
    let target = load_positions(&args.target)?;

    let config = RegistrationConfig {
        max_iterations: args.max_iterations,
        tolerance: args.tolerance,
        max_correspondence_dist: args.max_correspondence_dist.unwrap_or(f32::MAX),
        allow_scale: args.allow_scale,
        subsample_rate: args.subsample,
        outlier_fraction: args.outlier_fraction,
    };

    // The initial RMSE is measured on the same point set ICP will consume,
    // so `improvement_factor` compares like with like when `--subsample`
    // thins the source cloud.
    let probe = if args.subsample > 1 {
        subsample_positions(&source, args.subsample)?
    } else {
        source.clone()
    };
    let initial_rmse = compute_initial_rmse(&probe, &target)?;

    let result = register_point_clouds(&source, &target, &config)?;
    let stats = compute_registration_stats(&result, initial_rmse);

    let mut payload = json!({
        "source": args.source.display().to_string(),
        "target": args.target.display().to_string(),
        "converged": result.converged,
        "iterations": result.n_iterations,
        "correspondences": result.n_correspondences,
        "initial_rmse": stats.initial_rmse,
        "final_rmse": stats.final_rmse,
        "improvement_factor": stats.improvement_factor,
        "rotation_angle_deg": stats.rotation_angle_deg,
        "translation_magnitude": stats.transform_magnitude,
        "scale_change": stats.scale_change,
        "transform": {
            "rotation": result.transform.rotation,
            "translation": result.transform.translation,
            "scale": result.transform.scale,
        },
        "rmse_history": result.rmse_history,
    });

    let mut artifacts: Vec<(&str, &Path)> = Vec::new();
    if let Some(ref output) = args.output {
        if prepare_output(ctx, output, args.force)? {
            write_transformed(&args.source, &result.transform, output)?;
            artifacts.push(("scene", output.as_path()));
        }
        if let Some(map) = payload.as_object_mut() {
            map.insert("output".to_string(), json!(output.display().to_string()));
            map.insert("written".to_string(), json!(!ctx.dry_run));
        }
    }

    let summary = format_registration_result(&result, &stats);
    emit(ctx, "scene register", payload, &artifacts, || {
        println!("{summary}");
        if let Some(ref output) = args.output {
            if ctx.dry_run {
                println!("[dry-run] would write {}", output.display());
            } else {
                println!("Wrote {}", output.display());
            }
        }
    });
    Ok(())
}

/// Apply `transform` to every Gaussian centre of the source model and write
/// the result as a PLY file.
fn write_transformed(
    source_path: &Path,
    transform: &crate::cloud_registration::RegistrationTransform,
    output: &Path,
) -> Result<()> {
    let mut model = crate::export::load_model(source_path)
        .with_context(|| format!("Failed to load model: {}", source_path.display()))?;

    let mut positions = Vec::with_capacity(model.gaussians.len() * 3);
    for gaussian in &model.gaussians {
        positions.extend_from_slice(&gaussian.position);
    }
    let moved = apply_registration_transform(&positions, transform)?;

    for (gaussian, chunk) in model.gaussians.iter_mut().zip(moved.chunks_exact(3)) {
        gaussian.position = [chunk[0], chunk[1], chunk[2]];
    }

    crate::export::export_ply(&model, output)
        .with_context(|| format!("Failed to write scene: {}", output.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cloud_registration::RegistrationTransform;

    #[test]
    fn identity_transform_leaves_positions_untouched() {
        let positions = vec![0.0, 1.0, 2.0, -3.0, 4.0, 5.0];
        let moved = apply_registration_transform(&positions, &RegistrationTransform::identity())
            .expect("identity transform applies cleanly");
        assert_eq!(moved, positions);
    }

    #[test]
    fn icp_recovers_a_pure_translation() {
        // A small L-shaped cloud translated by (0.1, -0.2, 0.05).
        let base: Vec<f32> = vec![
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0,
        ];
        let shift = [0.1f32, -0.2, 0.05];
        let target: Vec<f32> = base
            .chunks_exact(3)
            .flat_map(|p| [p[0] + shift[0], p[1] + shift[1], p[2] + shift[2]])
            .collect();

        let config = RegistrationConfig::default();
        let result = register_point_clouds(&base, &target, &config)
            .expect("ICP should converge on a pure translation");
        assert!(
            result.final_rmse < 1e-3,
            "final RMSE {} should be near zero",
            result.final_rmse
        );
    }

    #[test]
    fn outlier_fraction_is_range_checked() {
        let args = RegisterArgs {
            source: PathBuf::from("a.ply"),
            target: PathBuf::from("b.ply"),
            output: None,
            max_iterations: 10,
            tolerance: 1e-5,
            max_correspondence_dist: None,
            allow_scale: false,
            subsample: 1,
            outlier_fraction: 1.5,
            force: false,
        };
        let ctx = CmdContext::new(crate::verbosity::Verbosity::Quiet, true, false);
        assert!(cmd_register(args, &ctx).is_err());
    }
}
