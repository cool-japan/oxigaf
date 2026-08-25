//! `oxigaf scene` — whole-scene operations on Gaussian model files.
//!
//! This module owns the `scene` family's top-level parser. `scene register`
//! is delegated verbatim to [`crate::commands::scene`], which owns the ICP
//! handler; every other subcommand is implemented here or in
//! [`crate::commands::scene_reduce`]:
//!
//! | Subcommand | Library module |
//! |------------|----------------|
//! | `register` | [`crate::cloud_registration`] |
//! | `stats` | [`crate::geometry_tools`] + [`crate::filter_gaussians`](mod@crate::filter_gaussians) |
//! | `filter` | [`crate::filter_gaussians`](mod@crate::filter_gaussians) |
//! | `prune` | [`crate::filter_gaussians`](mod@crate::filter_gaussians) |
//! | `transform` | [`crate::geometry_tools`] |
//! | `dedup` | [`crate::gaussian_deduplicator`] |
//! | `compress` | [`crate::gaussian_compressor`] |
//! | `lod` | [`crate::lod_generator`] |
//! | `convert` | [`crate::format_converter`] |
//!
//! Splitting the handlers across two files keeps both under the 2000-line
//! ceiling; the parser stays here so `cli.rs` has a single path to name.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Args, Subcommand};
use serde_json::json;

use crate::commands::model_io::{load_scene, save_scene, subset_model, FlatScene};
use crate::commands::scene_reduce::{CompressArgs, ConvertArgs, DedupArgs, LodArgs};
use crate::commands::{emit, parse_vec3, prepare_output, CmdContext};
use crate::filter_gaussians::{
    compute_scene_stats, filter_gaussians_multi, filter_gaussians_pipeline, prune_gaussians,
    FilterCriterion, PruningConfig,
};
use crate::geometry_tools::{
    center_at_origin, compute_geometry_stats, nearest_neighbor_distances, normalize_to_unit_cube,
    rescale_gaussians, transform_positions, transform_rotations, transform_scales, RigidTransform,
};

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
    Register(crate::commands::scene::RegisterArgs),

    /// Summarise a scene's geometry, opacity and scale distributions.
    Stats(StatsArgs),

    /// Keep only the Gaussians matching a set of criteria.
    Filter(FilterArgs),

    /// Drop degenerate Gaussians with the standard 3DGS pruning heuristics.
    Prune(PruneArgs),

    /// Move, rotate, rescale, centre or normalise a whole scene.
    Transform(TransformArgs),

    /// Merge near-duplicate Gaussians.
    Dedup(DedupArgs),

    /// Quantise, prune and optionally cluster a scene to shrink it.
    Compress(CompressArgs),

    /// Build a level-of-detail chain from a scene.
    Lod(LodArgs),

    /// Convert a scene between PLY/checkpoint and CSV/JSON/binary records.
    Convert(ConvertArgs),
}

/// Run the `scene` family.
///
/// # Errors
///
/// Propagates every handler's failure: unreadable models, invalid argument
/// combinations, and refused overwrites.
pub fn run(args: SceneArgs, ctx: CmdContext) -> Result<()> {
    match args.command {
        SceneCommand::Register(register_args) => crate::commands::scene::run(
            crate::commands::scene::SceneArgs {
                command: crate::commands::scene::SceneCommand::Register(register_args),
            },
            ctx,
        ),
        SceneCommand::Stats(stats_args) => cmd_stats(stats_args, &ctx),
        SceneCommand::Filter(filter_args) => cmd_filter(filter_args, &ctx),
        SceneCommand::Prune(prune_args) => cmd_prune(prune_args, &ctx),
        SceneCommand::Transform(transform_args) => cmd_transform(transform_args, &ctx),
        SceneCommand::Dedup(dedup_args) => {
            crate::commands::scene_reduce::cmd_dedup(dedup_args, &ctx)
        }
        SceneCommand::Compress(compress_args) => {
            crate::commands::scene_reduce::cmd_compress(compress_args, &ctx)
        }
        SceneCommand::Lod(lod_args) => crate::commands::scene_reduce::cmd_lod(lod_args, &ctx),
        SceneCommand::Convert(convert_args) => {
            crate::commands::scene_reduce::cmd_convert(convert_args, &ctx)
        }
    }
}

// ---------------------------------------------------------------------------
// scene stats
// ---------------------------------------------------------------------------

/// Arguments for `oxigaf scene stats`.
#[derive(Debug, Args)]
pub struct StatsArgs {
    /// Model to summarise (`.ply` or `.json` checkpoint).
    pub model: PathBuf,

    /// Also report the mean distance to the k-th nearest neighbour.
    ///
    /// This is an O(n²) scan; leave at 0 to skip it on large scenes.
    #[arg(long, default_value = "0")]
    pub neighbors: usize,
}

fn cmd_stats(args: StatsArgs, ctx: &CmdContext) -> Result<()> {
    let model = load_scene(&args.model)?;
    let flat = FlatScene::from_model(&model)?;
    if flat.n == 0 {
        anyhow::bail!("Scene is empty: {}", args.model.display());
    }

    let geometry = compute_geometry_stats(&flat.positions, &flat.log_scales)?;
    let scene = compute_scene_stats(&flat.gaussian_data())?;

    let neighbor_mean = if args.neighbors > 0 {
        let distances = nearest_neighbor_distances(&flat.positions, args.neighbors)?;
        if distances.is_empty() {
            None
        } else {
            Some(distances.iter().sum::<f32>() / distances.len() as f32)
        }
    } else {
        None
    };

    let payload = json!({
        "model": args.model.display().to_string(),
        "n_gaussians": flat.n,
        "sh_degree": flat.sh_degree,
        "bbox_min": geometry.bbox.min,
        "bbox_max": geometry.bbox.max,
        "centroid": geometry.centroid,
        "bounding_sphere": {
            "center": geometry.bounding_sphere.center,
            "radius": geometry.bounding_sphere.radius,
        },
        "scale": {
            "mean": geometry.mean_scale,
            "min": geometry.min_scale,
            "max": geometry.max_scale,
        },
        "position_std": geometry.std_position,
        "opacity": {
            "mean": scene.mean_opacity,
            "median": scene.median_opacity,
            "histogram": scene.opacity_histogram,
        },
        "volume": {
            "mean": scene.mean_volume,
            "total": scene.total_volume,
        },
        "size_histogram": scene.size_histogram,
        "scene_diameter": scene.scene_diameter,
        "mean_neighbor_distance": neighbor_mean,
    });

    emit(ctx, "scene stats", payload, &[], || {
        println!("{}", geometry.format_summary());
        println!("{}", scene.format_summary());
        if let Some(mean) = neighbor_mean {
            println!(
                "  mean distance to neighbour #{}: {mean:.6}",
                args.neighbors
            );
        }
    });
    Ok(())
}

// ---------------------------------------------------------------------------
// scene filter
// ---------------------------------------------------------------------------

/// Arguments for `oxigaf scene filter`.
///
/// Every threshold is optional; at least one must be given. Opacity bounds
/// are **probabilities** in `[0, 1]` (the module's own convention), not the
/// logits stored in the file.
#[derive(Debug, Args)]
pub struct FilterArgs {
    /// Scene to filter (`.ply` or `.json` checkpoint).
    pub input: PathBuf,

    /// Write the filtered scene here. Omit to only report what would survive.
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Overwrite the output file if it exists.
    #[arg(long)]
    pub force: bool,

    /// Keep Gaussians whose opacity is at least this probability.
    #[arg(long)]
    pub min_opacity: Option<f32>,

    /// Keep Gaussians whose opacity is at most this probability.
    #[arg(long)]
    pub max_opacity: Option<f32>,

    /// Keep Gaussians whose largest world-space extent is at least this.
    #[arg(long)]
    pub min_size: Option<f32>,

    /// Keep Gaussians whose largest world-space extent is at most this.
    #[arg(long)]
    pub max_size: Option<f32>,

    /// Keep Gaussians whose volume is at least this.
    #[arg(long)]
    pub min_volume: Option<f32>,

    /// Keep Gaussians whose volume is at most this.
    #[arg(long)]
    pub max_volume: Option<f32>,

    /// Keep Gaussians no more elongated than this max/min scale ratio.
    #[arg(long)]
    pub max_aspect: Option<f32>,

    /// Keep Gaussians with at least one DC colour channel this bright.
    #[arg(long)]
    pub min_brightness: Option<f32>,

    /// Lower corner of an axis-aligned keep region, as `x,y,z`.
    #[arg(long, value_parser = parse_vec3, requires = "bbox_max")]
    pub bbox_min: Option<[f32; 3]>,

    /// Upper corner of an axis-aligned keep region, as `x,y,z`.
    #[arg(long, value_parser = parse_vec3, requires = "bbox_min")]
    pub bbox_max: Option<[f32; 3]>,

    /// Centre of a spherical keep region, as `x,y,z`.
    #[arg(long, value_parser = parse_vec3, requires = "sphere_radius")]
    pub sphere_center: Option<[f32; 3]>,

    /// Radius of the spherical keep region.
    #[arg(long, requires = "sphere_center")]
    pub sphere_radius: Option<f32>,

    /// Invert the box/sphere region: keep what lies *outside* it.
    #[arg(long)]
    pub invert_region: bool,

    /// Apply the criteria one after another instead of all at once.
    ///
    /// The surviving set is identical for pure conjunctions; the flag exists
    /// so the per-stage behaviour of
    /// [`crate::filter_gaussians::filter_gaussians_pipeline`] is reachable.
    #[arg(long)]
    pub pipeline: bool,
}

impl FilterArgs {
    /// Translate the flags into library criteria.
    fn criteria(&self) -> Result<Vec<FilterCriterion>> {
        let mut criteria = Vec::new();
        if let Some(threshold) = self.min_opacity {
            criteria.push(FilterCriterion::OpacityAbove(threshold));
        }
        if let Some(threshold) = self.max_opacity {
            criteria.push(FilterCriterion::OpacityBelow(threshold));
        }
        if let Some(threshold) = self.min_size {
            criteria.push(FilterCriterion::SizeAbove(threshold));
        }
        if let Some(threshold) = self.max_size {
            criteria.push(FilterCriterion::SizeBelow(threshold));
        }
        if self.min_volume.is_some() || self.max_volume.is_some() {
            criteria.push(FilterCriterion::VolumeRange(
                self.min_volume.unwrap_or(0.0),
                self.max_volume.unwrap_or(f32::MAX),
            ));
        }
        if let Some(ratio) = self.max_aspect {
            criteria.push(FilterCriterion::MaxAspectRatio(ratio));
        }
        if let Some(brightness) = self.min_brightness {
            criteria.push(FilterCriterion::ColorBright(brightness));
        }
        if let (Some(min), Some(max)) = (self.bbox_min, self.bbox_max) {
            for axis in 0..3 {
                if min[axis] > max[axis] {
                    anyhow::bail!(
                        "--bbox-min component {axis} ({}) exceeds --bbox-max ({})",
                        min[axis],
                        max[axis]
                    );
                }
            }
            criteria.push(if self.invert_region {
                FilterCriterion::OutsideAabb { min, max }
            } else {
                FilterCriterion::InsideAabb { min, max }
            });
        }
        if let (Some(center), Some(radius)) = (self.sphere_center, self.sphere_radius) {
            if !(radius.is_finite() && radius > 0.0) {
                anyhow::bail!("--sphere-radius must be finite and positive (got {radius})");
            }
            criteria.push(if self.invert_region {
                FilterCriterion::OutsideSphere { center, radius }
            } else {
                FilterCriterion::InsideSphere { center, radius }
            });
        }
        if criteria.is_empty() {
            anyhow::bail!(
                "No filter criteria given. Pass at least one of --min-opacity, --max-opacity, \
                 --min-size, --max-size, --min-volume, --max-volume, --max-aspect, \
                 --min-brightness, --bbox-min/--bbox-max or --sphere-center/--sphere-radius."
            );
        }
        Ok(criteria)
    }
}

fn cmd_filter(args: FilterArgs, ctx: &CmdContext) -> Result<()> {
    let criteria = args.criteria()?;
    let model = load_scene(&args.input)?;
    let flat = FlatScene::from_model(&model)?;
    let gaussians = flat.gaussian_data();

    let result = if args.pipeline {
        filter_gaussians_pipeline(&gaussians, &criteria)?
    } else {
        filter_gaussians_multi(&gaussians, &criteria)?
    };

    let descriptions: Vec<String> = criteria.iter().map(|c| format!("{c:?}")).collect();
    let kept_fraction = if result.total == 0 {
        0.0
    } else {
        result.num_kept as f64 / result.total as f64
    };

    let mut payload = json!({
        "input": args.input.display().to_string(),
        "criteria": descriptions,
        "mode": if args.pipeline { "pipeline" } else { "all" },
        "total": result.total,
        "kept": result.num_kept,
        "removed": result.num_removed,
        "kept_fraction": kept_fraction,
    });

    let mut artifacts: Vec<(&str, &Path)> = Vec::new();
    let mut written = false;
    if let Some(ref output) = args.output {
        if prepare_output(ctx, output, args.force)? {
            let filtered = subset_model(&model, &result.kept_indices);
            save_scene(&filtered, output)?;
            artifacts.push(("scene", output.as_path()));
            written = true;
        }
        if let Some(map) = payload.as_object_mut() {
            map.insert("output".to_string(), json!(output.display().to_string()));
            map.insert("written".to_string(), json!(written));
        }
    }

    emit(ctx, "scene filter", payload, &artifacts, || {
        println!(
            "Filtered {} → {} Gaussians ({} removed, {:.1}% kept)",
            result.total,
            result.num_kept,
            result.num_removed,
            kept_fraction * 100.0
        );
        report_output(ctx, args.output.as_deref(), written);
    });
    Ok(())
}

// ---------------------------------------------------------------------------
// scene prune
// ---------------------------------------------------------------------------

/// Arguments for `oxigaf scene prune`.
#[derive(Debug, Args)]
pub struct PruneArgs {
    /// Scene to prune (`.ply` or `.json` checkpoint).
    pub input: PathBuf,

    /// Write the pruned scene here. Omit to only report the reduction.
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Overwrite the output file if it exists.
    #[arg(long)]
    pub force: bool,

    /// Drop Gaussians whose opacity probability is below this.
    #[arg(long, default_value = "0.005")]
    pub min_opacity: f32,

    /// Drop Gaussians smaller than this world-space extent.
    #[arg(long, default_value = "0.0")]
    pub min_size: f32,

    /// Drop Gaussians larger than this world-space extent.
    #[arg(long)]
    pub max_size: Option<f32>,

    /// Drop Gaussians more elongated than this max/min scale ratio.
    #[arg(long)]
    pub max_aspect_ratio: Option<f32>,

    /// Among the survivors, keep only the N most opaque.
    #[arg(long)]
    pub keep_top_n: Option<usize>,
}

fn cmd_prune(args: PruneArgs, ctx: &CmdContext) -> Result<()> {
    if let Some(n) = args.keep_top_n {
        if n == 0 {
            anyhow::bail!("--keep-top-n must be at least 1");
        }
    }
    let config = PruningConfig {
        min_opacity: args.min_opacity,
        min_size: args.min_size,
        max_size: args.max_size.unwrap_or(f32::MAX),
        max_aspect_ratio: args.max_aspect_ratio.unwrap_or(f32::MAX),
        keep_top_n: args.keep_top_n,
    };

    let model = load_scene(&args.input)?;
    let flat = FlatScene::from_model(&model)?;
    let result = prune_gaussians(&flat.gaussian_data(), &config)?;

    let mut payload = json!({
        "input": args.input.display().to_string(),
        "min_opacity": config.min_opacity,
        "min_size": config.min_size,
        "max_size": args.max_size,
        "max_aspect_ratio": args.max_aspect_ratio,
        "keep_top_n": args.keep_top_n,
        "total": result.total,
        "kept": result.num_kept,
        "removed": result.num_removed,
    });

    let mut artifacts: Vec<(&str, &Path)> = Vec::new();
    let mut written = false;
    if let Some(ref output) = args.output {
        if prepare_output(ctx, output, args.force)? {
            let pruned = subset_model(&model, &result.kept_indices);
            save_scene(&pruned, output)?;
            artifacts.push(("scene", output.as_path()));
            written = true;
        }
        if let Some(map) = payload.as_object_mut() {
            map.insert("output".to_string(), json!(output.display().to_string()));
            map.insert("written".to_string(), json!(written));
        }
    }

    emit(ctx, "scene prune", payload, &artifacts, || {
        println!(
            "Pruned {} → {} Gaussians ({} removed)",
            result.total, result.num_kept, result.num_removed
        );
        report_output(ctx, args.output.as_deref(), written);
    });
    Ok(())
}

// ---------------------------------------------------------------------------
// scene transform
// ---------------------------------------------------------------------------

/// Parse a comma-separated `qx,qy,qz,qw` quaternion for a clap `value_parser`.
fn parse_quat(raw: &str) -> std::result::Result<[f32; 4], String> {
    let parts: Vec<&str> = raw.split(',').map(str::trim).collect();
    if parts.len() != 4 {
        return Err(format!(
            "expected four comma-separated numbers (qx,qy,qz,qw), got {} component(s) in {raw:?}",
            parts.len()
        ));
    }
    let mut out = [0.0f32; 4];
    for (slot, text) in out.iter_mut().zip(parts.iter()) {
        let value: f32 = text
            .parse()
            .map_err(|_| format!("{text:?} is not a valid number"))?;
        if !value.is_finite() {
            return Err(format!("{text:?} is not finite"));
        }
        *slot = value;
    }
    let norm = (out[0] * out[0] + out[1] * out[1] + out[2] * out[2] + out[3] * out[3]).sqrt();
    if norm < 1e-6 {
        return Err(format!("{raw:?} is a zero quaternion"));
    }
    for slot in &mut out {
        *slot /= norm;
    }
    Ok(out)
}

/// Arguments for `oxigaf scene transform`.
///
/// Operations are applied in a fixed order: rigid transform (scale, then
/// rotation, then translation), then `--center`, then `--normalize`, then
/// `--target-mean-scale`. Per-Gaussian rotations and log-scales are updated
/// alongside the centres, so the Gaussians stay consistent with their new
/// positions rather than only the point cloud moving.
#[derive(Debug, Args)]
pub struct TransformArgs {
    /// Scene to transform (`.ply` or `.json` checkpoint).
    pub input: PathBuf,

    /// Write the transformed scene here.
    #[arg(short, long)]
    pub output: PathBuf,

    /// Overwrite the output file if it exists.
    #[arg(long)]
    pub force: bool,

    /// Translate by `x,y,z` (applied after rotation and scaling).
    #[arg(long, value_parser = parse_vec3)]
    pub translate: Option<[f32; 3]>,

    /// Rotate by the unit quaternion `qx,qy,qz,qw` (normalised on parse).
    #[arg(long, value_parser = parse_quat)]
    pub rotate: Option<[f32; 4]>,

    /// Uniform scale factor applied to positions and Gaussian sizes.
    #[arg(long)]
    pub scale: Option<f32>,

    /// Subtract the centroid so the scene is centred on the origin.
    #[arg(long)]
    pub center: bool,

    /// Centre on the bounding box and rescale to fit `[-0.5, 0.5]³`.
    #[arg(long)]
    pub normalize: bool,

    /// Rescale every Gaussian so the mean world-space size is this value.
    #[arg(long)]
    pub target_mean_scale: Option<f32>,
}

fn cmd_transform(args: TransformArgs, ctx: &CmdContext) -> Result<()> {
    if args.translate.is_none()
        && args.rotate.is_none()
        && args.scale.is_none()
        && !args.center
        && !args.normalize
        && args.target_mean_scale.is_none()
    {
        anyhow::bail!(
            "Nothing to do. Pass at least one of --translate, --rotate, --scale, --center, \
             --normalize or --target-mean-scale."
        );
    }
    if let Some(scale) = args.scale {
        if !(scale.is_finite() && scale > 0.0) {
            anyhow::bail!("--scale must be finite and positive (got {scale})");
        }
    }

    let model = load_scene(&args.input)?;
    let mut flat = FlatScene::from_model(&model)?;
    if flat.n == 0 {
        anyhow::bail!("Scene is empty: {}", args.input.display());
    }

    let mut applied: Vec<String> = Vec::new();

    if args.translate.is_some() || args.rotate.is_some() || args.scale.is_some() {
        let transform = RigidTransform {
            rotation: args.rotate.unwrap_or([0.0, 0.0, 0.0, 1.0]),
            translation: args.translate.unwrap_or([0.0, 0.0, 0.0]),
            scale: args.scale.unwrap_or(1.0),
        };
        transform_positions(&mut flat.positions, &transform)?;
        if args.rotate.is_some() {
            transform_rotations(&mut flat.rotations, &transform)?;
        }
        if args.scale.is_some() {
            transform_scales(&mut flat.log_scales, &transform)?;
        }
        applied.push(format!(
            "rigid(scale={}, rotation={:?}, translation={:?})",
            transform.scale, transform.rotation, transform.translation
        ));
    }

    let mut centroid = None;
    if args.center {
        centroid = Some(center_at_origin(&mut flat.positions)?);
        applied.push("center".to_string());
    }

    let mut normalize_extent = None;
    if args.normalize {
        let extent = normalize_to_unit_cube(&mut flat.positions)?;
        if extent.is_finite() && extent > 0.0 {
            // Positions were divided by `extent`; shrink the Gaussians by the
            // same factor so they do not stay oversized in the new frame.
            transform_scales(
                &mut flat.log_scales,
                &RigidTransform::from_scale(1.0 / extent),
            )?;
        }
        normalize_extent = Some(extent);
        applied.push("normalize".to_string());
    }

    let mut rescale_delta = None;
    if let Some(target) = args.target_mean_scale {
        rescale_delta = Some(rescale_gaussians(&mut flat.log_scales, target)?);
        applied.push(format!("target-mean-scale={target}"));
    }

    let payload = json!({
        "input": args.input.display().to_string(),
        "output": args.output.display().to_string(),
        "n_gaussians": flat.n,
        "operations": applied.clone(),
        "centroid_removed": centroid,
        "normalize_extent": normalize_extent,
        "rescale_log_delta": rescale_delta,
        "written": !ctx.dry_run,
    });

    let mut artifacts: Vec<(&str, &Path)> = Vec::new();
    let mut written = false;
    if prepare_output(ctx, &args.output, args.force)? {
        let mut transformed = model.clone();
        for (index, gaussian) in transformed.gaussians.iter_mut().enumerate() {
            gaussian.position = [
                flat.positions[index * 3],
                flat.positions[index * 3 + 1],
                flat.positions[index * 3 + 2],
            ];
            gaussian.rotation = [
                flat.rotations[index * 4],
                flat.rotations[index * 4 + 1],
                flat.rotations[index * 4 + 2],
                flat.rotations[index * 4 + 3],
            ];
            gaussian.scale = [
                flat.log_scales[index * 3],
                flat.log_scales[index * 3 + 1],
                flat.log_scales[index * 3 + 2],
            ];
        }
        save_scene(&transformed, &args.output)
            .with_context(|| format!("Failed to write {}", args.output.display()))?;
        artifacts.push(("scene", args.output.as_path()));
        written = true;
    }

    emit(ctx, "scene transform", payload, &artifacts, || {
        println!("Applied: {}", applied.join(", "));
        report_output(ctx, Some(args.output.as_path()), written);
    });
    Ok(())
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Print the "wrote / would write" line shared by every writing handler.
pub(crate) fn report_output(ctx: &CmdContext, output: Option<&Path>, written: bool) {
    let Some(path) = output else {
        return;
    };
    if written {
        println!("Wrote {}", path.display());
    } else if ctx.dry_run {
        println!("[dry-run] would write {}", path.display());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::verbosity::Verbosity;

    fn quiet_ctx() -> CmdContext {
        CmdContext::new(Verbosity::Quiet, true, false)
    }

    fn base_filter_args() -> FilterArgs {
        FilterArgs {
            input: PathBuf::from("scene.ply"),
            output: None,
            force: false,
            min_opacity: None,
            max_opacity: None,
            min_size: None,
            max_size: None,
            min_volume: None,
            max_volume: None,
            max_aspect: None,
            min_brightness: None,
            bbox_min: None,
            bbox_max: None,
            sphere_center: None,
            sphere_radius: None,
            invert_region: false,
            pipeline: false,
        }
    }

    #[test]
    fn filter_requires_at_least_one_criterion() {
        assert!(base_filter_args().criteria().is_err());
    }

    #[test]
    fn filter_builds_the_expected_criteria() {
        let mut args = base_filter_args();
        args.min_opacity = Some(0.1);
        args.max_aspect = Some(4.0);
        let criteria = args.criteria().expect("criteria");
        assert_eq!(criteria.len(), 2);
        assert!(matches!(criteria[0], FilterCriterion::OpacityAbove(v) if (v - 0.1).abs() < 1e-6));
        assert!(
            matches!(criteria[1], FilterCriterion::MaxAspectRatio(v) if (v - 4.0).abs() < 1e-6)
        );
    }

    #[test]
    fn filter_inverts_the_region_when_asked() {
        let mut args = base_filter_args();
        args.sphere_center = Some([0.0, 0.0, 0.0]);
        args.sphere_radius = Some(1.0);
        args.invert_region = true;
        let criteria = args.criteria().expect("criteria");
        assert!(matches!(criteria[0], FilterCriterion::OutsideSphere { .. }));
    }

    #[test]
    fn filter_rejects_an_inverted_bounding_box() {
        let mut args = base_filter_args();
        args.bbox_min = Some([1.0, 0.0, 0.0]);
        args.bbox_max = Some([0.0, 1.0, 1.0]);
        assert!(args.criteria().is_err());
    }

    #[test]
    fn filter_rejects_a_non_positive_sphere_radius() {
        let mut args = base_filter_args();
        args.sphere_center = Some([0.0, 0.0, 0.0]);
        args.sphere_radius = Some(0.0);
        assert!(args.criteria().is_err());
    }

    #[test]
    fn parse_quat_normalises_and_validates() {
        let parsed = parse_quat("0,0,0,2").expect("quaternion");
        assert!((parsed[3] - 1.0).abs() < 1e-6);
        assert!(parse_quat("0,0,0").is_err());
        assert!(parse_quat("0,0,0,0").is_err());
        assert!(parse_quat("0,0,0,nan").is_err());
    }

    #[test]
    fn transform_rejects_an_empty_operation_set() {
        let args = TransformArgs {
            input: PathBuf::from("in.ply"),
            output: PathBuf::from("out.ply"),
            force: false,
            translate: None,
            rotate: None,
            scale: None,
            center: false,
            normalize: false,
            target_mean_scale: None,
        };
        assert!(cmd_transform(args, &quiet_ctx()).is_err());
    }

    #[test]
    fn transform_rejects_a_non_positive_scale() {
        let args = TransformArgs {
            input: PathBuf::from("in.ply"),
            output: PathBuf::from("out.ply"),
            force: false,
            translate: None,
            rotate: None,
            scale: Some(-1.0),
            center: false,
            normalize: false,
            target_mean_scale: None,
        };
        assert!(cmd_transform(args, &quiet_ctx()).is_err());
    }

    #[test]
    fn prune_rejects_keep_top_zero() {
        let args = PruneArgs {
            input: PathBuf::from("in.ply"),
            output: None,
            force: false,
            min_opacity: 0.005,
            min_size: 0.0,
            max_size: None,
            max_aspect_ratio: None,
            keep_top_n: Some(0),
        };
        assert!(cmd_prune(args, &quiet_ctx()).is_err());
    }
}
