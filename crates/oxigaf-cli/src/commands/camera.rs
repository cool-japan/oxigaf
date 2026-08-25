//! `oxigaf camera` — camera path authoring and arcball navigation.
//!
//! Glue over [`crate::camera_path_tool`] and [`crate::arcball`].
//!
//! Paths can be written in two schemas:
//!
//! * `--format path` — the native [`crate::camera_path_tool`] document
//!   (position / target / up / fov per frame), round-trippable through
//!   `camera stats` and `camera blend`.
//! * `--format camera-spec` — the spherical `{azimuth, elevation, distance}`
//!   list that `oxigaf render --cameras` consumes, so a generated path can
//!   be rendered without any further conversion.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Args, Subcommand, ValueEnum};
use serde_json::json;

use crate::arcball::ArcballCamera;
use crate::camera_path_tool::{
    blend_paths, compute_path_stats, figure_eight_path, orbit_path, path_from_json, path_to_json,
    path_velocities, smooth_path, spiral_orbit_path, turntable_preset, zoom_in_path, CameraPath,
    PathConfig, PathStats,
};
use crate::commands::{emit, parse_vec3, prepare_output, CmdContext};
use crate::pipeline::CameraSpec;

/// `oxigaf camera <command>`.
#[derive(Debug, Args)]
pub struct CameraArgs {
    #[command(subcommand)]
    pub command: CameraCommand,
}

/// Camera tooling subcommands.
#[derive(Debug, Subcommand)]
pub enum CameraCommand {
    /// Generate a camera path from an analytic preset.
    Path(PathArgs),

    /// Report statistics (length, speed, FOV) for an existing path file.
    Stats {
        /// Camera path JSON produced by `oxigaf camera path --format path`.
        input: PathBuf,

        /// Fallback frame rate when the document does not carry one.
        #[arg(long, default_value = "30")]
        fps: f32,

        /// Also print the per-frame speed series.
        #[arg(long)]
        velocities: bool,
    },

    /// Blend two equal-length camera paths frame by frame.
    Blend {
        /// First camera path JSON.
        path_a: PathBuf,
        /// Second camera path JSON.
        path_b: PathBuf,
        /// Output camera path JSON.
        #[arg(short, long)]
        output: PathBuf,
        /// Blend factor: 0.0 keeps path A, 1.0 keeps path B.
        #[arg(long, default_value = "0.5")]
        t: f32,
        /// Fallback frame rate when a document does not carry one.
        #[arg(long, default_value = "30")]
        fps: f32,
        /// Overwrite the output file if it exists.
        #[arg(long)]
        force: bool,
    },

    /// Evaluate an arcball camera after applying orbit / dolly / pan deltas.
    Arcball(ArcballArgs),
}

/// Analytic camera path presets.
#[derive(Debug, Clone, Copy, ValueEnum, Default)]
pub enum PathShape {
    /// Circular orbit at a fixed height.
    #[default]
    Orbit,
    /// Orbit with linearly changing radius and height.
    Spiral,
    /// Lemniscate ("figure eight") in the horizontal plane.
    FigureEight,
    /// One full turntable revolution at height 0.
    Turntable,
    /// Straight dolly-in toward the target.
    ZoomIn,
}

/// Output schema for a generated path.
#[derive(Debug, Clone, Copy, ValueEnum, Default)]
pub enum PathFormat {
    /// Native camera-path document (position / target / up / fov).
    #[default]
    Path,
    /// Spherical camera specs consumed by `oxigaf render --cameras`.
    CameraSpec,
}

/// Arguments for `oxigaf camera path`.
#[derive(Debug, Args)]
pub struct PathArgs {
    /// Path preset to generate.
    #[arg(long, value_enum, default_value = "orbit")]
    pub shape: PathShape,

    /// Look-at point as `x,y,z`.
    #[arg(long, default_value = "0,0,0", value_parser = parse_vec3)]
    pub center: [f32; 3],

    /// Orbit radius (start radius for `spiral`, start distance for `zoom-in`).
    #[arg(long, default_value = "0.6")]
    pub radius: f32,

    /// End radius for `spiral`.
    #[arg(long)]
    pub radius_end: Option<f32>,

    /// Camera height above the target (start height for `spiral`).
    #[arg(long, default_value = "0.0")]
    pub height: f32,

    /// End height for `spiral`.
    #[arg(long)]
    pub height_end: Option<f32>,

    /// Number of revolutions for `orbit`.
    #[arg(long, default_value = "1.0")]
    pub turns: f32,

    /// Zoom factor for `zoom-in`: end distance = start distance / factor.
    #[arg(long, default_value = "2.0")]
    pub zoom_factor: f32,

    /// Path duration in seconds.
    #[arg(long, default_value = "5.0")]
    pub duration: f32,

    /// Frames per second.
    #[arg(long, default_value = "30")]
    pub fps: f32,

    /// Moving-average window applied to positions after generation (1 = off).
    #[arg(long, default_value = "1")]
    pub smooth: usize,

    /// Output file. Omit to print the document on stdout.
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Output schema.
    #[arg(long, value_enum, default_value = "path")]
    pub format: PathFormat,

    /// Overwrite the output file if it exists.
    #[arg(long)]
    pub force: bool,
}

/// Arguments for `oxigaf camera arcball`.
#[derive(Debug, Args)]
pub struct ArcballArgs {
    /// Orbit target as `x,y,z`.
    #[arg(long, default_value = "0,0,0", value_parser = parse_vec3)]
    pub target: [f32; 3],

    /// Distance from the target.
    #[arg(long, default_value = "0.6")]
    pub distance: f32,

    /// Initial yaw in degrees.
    #[arg(long, default_value = "0.0")]
    pub yaw: f32,

    /// Initial pitch in degrees.
    #[arg(long, default_value = "10.0")]
    pub pitch: f32,

    /// Apply an orbit delta `yaw_deg,pitch_deg`.
    #[arg(long, value_parser = parse_pair)]
    pub orbit: Option<[f32; 2]>,

    /// Apply a dolly delta (positive moves closer).
    #[arg(long)]
    pub dolly: Option<f32>,

    /// Apply a pan delta `dx,dy` in view-space units.
    #[arg(long, value_parser = parse_pair)]
    pub pan: Option<[f32; 2]>,

    /// Viewport aspect ratio used for the projection matrix.
    #[arg(long, default_value = "1.0")]
    pub aspect: f32,
}

fn parse_pair(raw: &str) -> std::result::Result<[f32; 2], String> {
    let parts: Vec<&str> = raw.split(',').map(str::trim).collect();
    if parts.len() != 2 {
        return Err(format!(
            "expected two comma-separated numbers (a,b), got {} component(s) in {raw:?}",
            parts.len()
        ));
    }
    let mut out = [0.0f32; 2];
    for (slot, text) in out.iter_mut().zip(parts.iter()) {
        let value: f32 = text
            .parse()
            .map_err(|_| format!("{text:?} is not a valid number"))?;
        if !value.is_finite() {
            return Err(format!("{text:?} is not finite"));
        }
        *slot = value;
    }
    Ok(out)
}

/// Run the `camera` family.
///
/// # Errors
///
/// Returns an error when a preset rejects its parameters, when a path file
/// cannot be read or parsed, or when an output file exists without
/// `--force`.
pub fn run(args: CameraArgs, ctx: CmdContext) -> Result<()> {
    match args.command {
        CameraCommand::Path(path_args) => cmd_path(path_args, &ctx),
        CameraCommand::Stats {
            input,
            fps,
            velocities,
        } => cmd_stats(&input, fps, velocities, &ctx),
        CameraCommand::Blend {
            path_a,
            path_b,
            output,
            t,
            fps,
            force,
        } => {
            if !(0.0..=1.0).contains(&t) {
                anyhow::bail!("--t must be within [0.0, 1.0] (got {t})");
            }
            let a = load_path(&path_a, fps)?;
            let b = load_path(&path_b, fps)?;
            let blended = blend_paths(&a, &b, t)?;
            write_path(
                &blended,
                &output,
                PathFormat::Path,
                force,
                "camera blend",
                &ctx,
            )
        }
        CameraCommand::Arcball(arcball_args) => cmd_arcball(arcball_args, &ctx),
    }
}

fn load_path(path: &Path, fps: f32) -> Result<CameraPath> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read camera path: {}", path.display()))?;
    let parsed = path_from_json(&raw, fps)
        .with_context(|| format!("Failed to parse camera path: {}", path.display()))?;
    Ok(parsed)
}

/// Convert a pose to the spherical `{azimuth, elevation, distance}` triple
/// used by [`crate::pipeline::orbit_camera`].
///
/// `orbit_camera` places the eye at
/// `(d·cos(el)·sin(az), d·sin(el), d·cos(el)·cos(az))` and always looks at
/// the world origin, so a pose whose target is not the origin can only be
/// represented relative to that target — the resulting spec renders the same
/// framing about the origin, not about the original target.
fn pose_to_spec(position: [f32; 3], target: [f32; 3]) -> CameraSpec {
    let dx = position[0] - target[0];
    let dy = position[1] - target[1];
    let dz = position[2] - target[2];
    let distance = (dx * dx + dy * dy + dz * dz).sqrt();
    let elevation = if distance > f32::EPSILON {
        (dy / distance).clamp(-1.0, 1.0).asin().to_degrees()
    } else {
        0.0
    };
    let azimuth = dx.atan2(dz).to_degrees();
    CameraSpec {
        azimuth,
        elevation,
        distance,
    }
}

fn render_document(path: &CameraPath, format: PathFormat) -> Result<String> {
    match format {
        PathFormat::Path => Ok(path_to_json(path)),
        PathFormat::CameraSpec => {
            let specs: Vec<CameraSpec> = path
                .frames
                .iter()
                .map(|pose| pose_to_spec(pose.position, pose.target))
                .collect();
            serde_json::to_string_pretty(&specs).context("Failed to serialise camera specs to JSON")
        }
    }
}

fn stats_json(stats: &PathStats) -> serde_json::Value {
    json!({
        "total_frames": stats.total_frames,
        "duration_secs": stats.duration_secs,
        "total_distance": stats.total_distance,
        "mean_speed": stats.mean_speed,
        "max_speed": stats.max_speed,
        "mean_fov": stats.mean_fov,
    })
}

fn write_path(
    path: &CameraPath,
    output: &Path,
    format: PathFormat,
    force: bool,
    command: &str,
    ctx: &CmdContext,
) -> Result<()> {
    let document = render_document(path, format)?;
    let stats = compute_path_stats(path)?;
    let mut payload = stats_json(&stats);
    if let Some(map) = payload.as_object_mut() {
        map.insert("output".to_string(), json!(output.display().to_string()));
    }

    if !prepare_output(ctx, output, force)? {
        emit(ctx, command, payload, &[], || {
            println!(
                "[dry-run] would write {} ({} frames)",
                output.display(),
                stats.total_frames
            );
        });
        return Ok(());
    }

    std::fs::write(output, document.as_bytes())
        .with_context(|| format!("Failed to write camera path: {}", output.display()))?;

    emit(ctx, command, payload, &[("camera-path", output)], || {
        println!(
            "Wrote {} ({} frames, {:.2}s, {:.4} units travelled)",
            output.display(),
            stats.total_frames,
            stats.duration_secs,
            stats.total_distance
        );
    });
    Ok(())
}

fn cmd_path(args: PathArgs, ctx: &CmdContext) -> Result<()> {
    if !(args.fps.is_finite() && args.fps > 0.0) {
        anyhow::bail!("--fps must be a positive, finite number (got {})", args.fps);
    }
    if !(args.duration.is_finite() && args.duration > 0.0) {
        anyhow::bail!(
            "--duration must be a positive, finite number (got {})",
            args.duration
        );
    }
    if args.smooth == 0 {
        anyhow::bail!("--smooth must be at least 1 (1 disables smoothing)");
    }

    let config = PathConfig {
        duration_secs: args.duration,
        fps: args.fps,
        smooth_tangents: false,
    };

    let generated = match args.shape {
        PathShape::Orbit => orbit_path(args.center, args.radius, args.height, args.turns, &config)?,
        PathShape::Spiral => spiral_orbit_path(
            args.center,
            args.radius,
            args.radius_end.unwrap_or(args.radius),
            args.height,
            args.height_end.unwrap_or(args.height),
            &config,
        )?,
        PathShape::FigureEight => {
            figure_eight_path(args.center, args.radius, args.height, &config)?
        }
        PathShape::Turntable => {
            turntable_preset(args.center, args.radius, args.fps, args.duration)?
        }
        PathShape::ZoomIn => {
            let start = [
                args.center[0],
                args.center[1] + args.height,
                args.center[2] + args.radius,
            ];
            zoom_in_path(start, args.center, args.zoom_factor, &config)?
        }
    };

    let path = if args.smooth > 1 {
        smooth_path(&generated, args.smooth)?
    } else {
        generated
    };

    match args.output {
        Some(ref output) => write_path(&path, output, args.format, args.force, "camera path", ctx),
        None => {
            let document = render_document(&path, args.format)?;
            let stats = compute_path_stats(&path)?;
            if ctx.json {
                // stdout must carry a single JSON document: fold the path
                // document into the standard envelope rather than printing
                // it separately.
                let parsed: serde_json::Value = serde_json::from_str(&document)
                    .context("Failed to re-parse generated camera path document")?;
                let mut payload = stats_json(&stats);
                if let Some(map) = payload.as_object_mut() {
                    map.insert("document".to_string(), parsed);
                }
                emit(ctx, "camera path", payload, &[], || {});
            } else {
                println!("{document}");
            }
            Ok(())
        }
    }
}

fn cmd_stats(input: &Path, fps: f32, velocities: bool, ctx: &CmdContext) -> Result<()> {
    let path = load_path(input, fps)?;
    let stats = compute_path_stats(&path)?;
    let speeds = if velocities {
        path_velocities(&path)?
    } else {
        Vec::new()
    };

    let mut payload = stats_json(&stats);
    if velocities {
        if let Some(map) = payload.as_object_mut() {
            map.insert("velocities".to_string(), json!(speeds));
        }
    }

    emit(ctx, "camera stats", payload, &[], || {
        println!("Camera path: {}", input.display());
        println!("  frames        : {}", stats.total_frames);
        println!("  duration      : {:.3}s", stats.duration_secs);
        println!("  total distance: {:.6}", stats.total_distance);
        println!("  mean speed    : {:.6} units/s", stats.mean_speed);
        println!("  max speed     : {:.6} units/s", stats.max_speed);
        println!("  mean FOV      : {:.2}°", stats.mean_fov);
        if velocities {
            println!("  velocities    :");
            for (i, speed) in speeds.iter().enumerate() {
                println!("    {i:>5}: {speed:.6}");
            }
        }
    });
    Ok(())
}

fn cmd_arcball(args: ArcballArgs, ctx: &CmdContext) -> Result<()> {
    if !(args.distance.is_finite() && args.distance > 0.0) {
        anyhow::bail!(
            "--distance must be a positive, finite number (got {})",
            args.distance
        );
    }
    if !(args.aspect.is_finite() && args.aspect > 0.0) {
        anyhow::bail!(
            "--aspect must be a positive, finite number (got {})",
            args.aspect
        );
    }

    let mut camera = ArcballCamera::new(
        args.target,
        args.distance,
        args.yaw.to_radians(),
        args.pitch.to_radians(),
    );
    if let Some([delta_yaw, delta_pitch]) = args.orbit {
        camera.orbit(delta_yaw.to_radians(), delta_pitch.to_radians());
    }
    if let Some(delta) = args.dolly {
        camera.dolly(delta);
    }
    if let Some([dx, dy]) = args.pan {
        camera.pan(dx, dy);
    }

    let position = camera.position();
    let view = camera.view_matrix();
    let view_projection = camera.view_projection(args.aspect);
    let spec = pose_to_spec(position, args.target);

    let payload = json!({
        "position": position,
        "target": args.target,
        "view_matrix": view,
        "view_projection": view_projection,
        "camera_spec": {
            "azimuth": spec.azimuth,
            "elevation": spec.elevation,
            "distance": spec.distance,
        },
    });

    emit(ctx, "camera arcball", payload, &[], || {
        println!(
            "position : [{:.6}, {:.6}, {:.6}]",
            position[0], position[1], position[2]
        );
        println!(
            "target   : [{:.6}, {:.6}, {:.6}]",
            args.target[0], args.target[1], args.target[2]
        );
        println!(
            "spec     : azimuth {:.3}°, elevation {:.3}°, distance {:.6}",
            spec.azimuth, spec.elevation, spec.distance
        );
        // `Mat4` is a flat column-major `[f32; 16]` indexed `mat[col * 4 + row]`,
        // so a printed row gathers one element from each column.
        println!("view matrix (rows of the column-major 4x4):");
        for row in 0..4 {
            println!(
                "  [{:>10.5} {:>10.5} {:>10.5} {:>10.5}]",
                view[row],
                view[4 + row],
                view[8 + row],
                view[12 + row]
            );
        }
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::verbosity::Verbosity;

    #[test]
    fn pose_to_spec_inverts_orbit_camera_convention() {
        // orbit_camera: x = d·cos(el)·sin(az), y = d·sin(el), z = d·cos(el)·cos(az)
        let (az, el, d) = (37.0f32, 12.0f32, 0.75f32);
        let (ar, er) = (az.to_radians(), el.to_radians());
        let position = [
            d * er.cos() * ar.sin(),
            d * er.sin(),
            d * er.cos() * ar.cos(),
        ];
        let spec = pose_to_spec(position, [0.0, 0.0, 0.0]);
        assert!((spec.azimuth - az).abs() < 1e-2, "azimuth {}", spec.azimuth);
        assert!(
            (spec.elevation - el).abs() < 1e-2,
            "elevation {}",
            spec.elevation
        );
        assert!(
            (spec.distance - d).abs() < 1e-4,
            "distance {}",
            spec.distance
        );
    }

    #[test]
    fn parse_pair_rejects_bad_arity() {
        assert!(parse_pair("1").is_err());
        assert!(parse_pair("1,2,3").is_err());
        assert_eq!(parse_pair("1.5, -2"), Ok([1.5, -2.0]));
    }

    #[test]
    fn camera_spec_document_is_render_compatible() {
        let config = PathConfig {
            duration_secs: 0.2,
            fps: 10.0,
            smooth_tangents: false,
        };
        let path = orbit_path([0.0, 0.0, 0.0], 0.6, 0.1, 1.0, &config)
            .expect("orbit path with valid config");
        let document =
            render_document(&path, PathFormat::CameraSpec).expect("camera-spec serialisation");
        let specs: Vec<CameraSpec> =
            serde_json::from_str(&document).expect("render --cameras must be able to parse this");
        assert_eq!(specs.len(), path.frames.len());
        for spec in &specs {
            assert!(
                (spec.distance - 0.6082763).abs() < 1e-3,
                "{}",
                spec.distance
            );
        }
    }

    #[test]
    fn path_dry_run_writes_nothing() {
        let ctx = CmdContext::new(Verbosity::Quiet, true, true);
        let out = std::env::temp_dir().join("oxigaf_camera_path_dry_run.json");
        let _ = std::fs::remove_file(&out);
        let config = PathConfig {
            duration_secs: 0.2,
            fps: 10.0,
            smooth_tangents: false,
        };
        let path = orbit_path([0.0, 0.0, 0.0], 0.6, 0.0, 1.0, &config).expect("orbit path");
        write_path(&path, &out, PathFormat::Path, false, "camera path", &ctx)
            .expect("dry-run must succeed");
        assert!(!out.exists());
    }
}
