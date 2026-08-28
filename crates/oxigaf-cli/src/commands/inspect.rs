//! `oxigaf inspect` — read-only interrogation of model and scene files.
//!
//! | Subcommand | Library module |
//! |------------|----------------|
//! | `inspect model` | [`crate::model_inspector`] |
//! | `inspect query` | [`crate::model_inspector`] |
//! | `inspect memory` | [`crate::memory_estimator`] |
//! | `inspect ply` | [`crate::export_ply`] |
//! | `inspect pointcloud` | [`crate::export_pointcloud_stats`] |
//!
//! Nothing here writes to the scene; the only files these commands can
//! produce are the optional CSV dumps, which go through
//! [`crate::commands::prepare_output`] like every other artifact.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Args, Subcommand};
use serde_json::json;

use crate::cli::PointColorMode;
use crate::commands::model_io::{load_scene, FlatScene};
use crate::commands::{emit, parse_vec3, prepare_output, CmdContext};
use crate::export_ply::{
    ply_compute_scene_stats, ply_format_stats, ply_parse_element_count, ply_parse_format,
    ply_parse_properties, ply_read,
};
use crate::export_pointcloud_stats::PointCloudStats;
use crate::memory_estimator::{
    compare_memory_configs, estimate_memory, estimate_model_weights, format_memory_estimate,
    max_gaussians_for_vram, mem_format_bytes, memory_breakdown_percent, MemEstimateConfig,
};
use crate::model_inspector::{
    density_voxel_grid, dump_gaussians_csv, find_high_anisotropy, find_large_gaussians,
    find_low_opacity, find_spatial_outliers, format_inspection_report, histogram, inspect_model,
    percentile, query_aabb, query_knn, query_sphere, BoundingBox3d, InspectableModel, QueryResult,
};

/// `oxigaf inspect <command>`.
#[derive(Debug, Args)]
pub struct InspectArgs {
    #[command(subcommand)]
    pub command: InspectCommand,
}

/// Inspection subcommands.
#[derive(Debug, Subcommand)]
pub enum InspectCommand {
    /// Full statistical report on a Gaussian model.
    Model(ModelArgs),

    /// Select Gaussians by region or by property and report the matches.
    Query(QueryArgs),

    /// Estimate the VRAM and RAM a configuration needs.
    Memory(MemoryArgs),

    /// Report a PLY file's header, format and scene statistics.
    Ply(PlyArgs),

    /// Report what a point-cloud export of a model would contain.
    Pointcloud(PointcloudArgs),
}

/// Run the `inspect` family.
///
/// # Errors
///
/// Propagates unreadable files, empty models and invalid query arguments.
pub fn run(args: InspectArgs, ctx: CmdContext) -> Result<()> {
    match args.command {
        InspectCommand::Model(model_args) => cmd_model(model_args, &ctx),
        InspectCommand::Query(query_args) => cmd_query(query_args, &ctx),
        InspectCommand::Memory(memory_args) => cmd_memory(memory_args, &ctx),
        InspectCommand::Ply(ply_args) => cmd_ply(ply_args, &ctx),
        InspectCommand::Pointcloud(pointcloud_args) => cmd_pointcloud(pointcloud_args, &ctx),
    }
}

/// Load a model and present it in the inspector's view (probability opacity,
/// activated colours).
fn inspectable(path: &Path) -> Result<InspectableModel> {
    let model = load_scene(path)?;
    let flat = FlatScene::from_model(&model)?;
    if flat.n == 0 {
        anyhow::bail!("Model contains no Gaussians: {}", path.display());
    }
    flat.inspectable()
}

// ---------------------------------------------------------------------------
// inspect model
// ---------------------------------------------------------------------------

/// Arguments for `oxigaf inspect model`.
#[derive(Debug, Args)]
pub struct ModelArgs {
    /// Model to inspect (`.ply` or `.json` checkpoint).
    pub model: PathBuf,

    /// Bin count for the opacity histogram (0 disables it).
    #[arg(long, default_value = "10")]
    pub histogram_bins: usize,

    /// Voxels per axis for the density grid (0 disables it).
    #[arg(long, default_value = "0")]
    pub voxels: usize,

    /// Dump the first Gaussians to this CSV file.
    #[arg(long)]
    pub dump_csv: Option<PathBuf>,

    /// How many Gaussians `--dump-csv` writes.
    #[arg(long, default_value = "100")]
    pub dump_limit: usize,

    /// Overwrite the CSV dump if it exists.
    #[arg(long)]
    pub force: bool,
}

fn cmd_model(args: ModelArgs, ctx: &CmdContext) -> Result<()> {
    let model = inspectable(&args.model)?;
    let report = inspect_model(&model)?;

    let opacity_histogram = if args.histogram_bins > 0 {
        let (edges, counts) = histogram(&model.opacities, args.histogram_bins)?;
        Some(json!({ "edges": edges, "counts": counts }))
    } else {
        None
    };

    // Percentiles of the largest per-Gaussian extent, which is what decides
    // how much a scene "blooms" when rasterised. `percentile` sorts in
    // place, so the first call is the only one that does any work — the
    // later two see an already-sorted slice.
    let mut max_scales = Vec::with_capacity(model.n);
    for index in 0..model.n {
        max_scales.push(
            model.activated_scale(index, 0)?.max(
                model
                    .activated_scale(index, 1)?
                    .max(model.activated_scale(index, 2)?),
            ),
        );
    }
    let scale_p50 = percentile(&mut max_scales, 50.0)?;
    let scale_p90 = percentile(&mut max_scales, 90.0)?;
    let scale_p99 = percentile(&mut max_scales, 99.0)?;

    let density = if args.voxels > 0 {
        let (counts, voxel_size) = density_voxel_grid(&model, args.voxels)?;
        let occupied = counts.iter().filter(|&&count| count > 0).count();
        Some(json!({
            "grid": args.voxels,
            "voxel_size": voxel_size,
            "occupied_voxels": occupied,
            "total_voxels": counts.len(),
            "max_voxel_count": counts.iter().copied().max().unwrap_or(0),
        }))
    } else {
        None
    };

    let mut payload = json!({
        "model": args.model.display().to_string(),
        "n_gaussians": report.n_gaussians,
        "bbox_min": report.bounding_box.min,
        "bbox_max": report.bounding_box.max,
        "bbox_diagonal": report.bounding_box.diagonal(),
        "opacity": {
            "mean": report.mean_opacity,
            "std": report.std_opacity,
            "transparent_fraction": report.transparent_fraction,
            "histogram": opacity_histogram,
        },
        "scale": {
            "mean_max": report.mean_max_scale,
            "std_max": report.std_max_scale,
            "p50": scale_p50,
            "p90": scale_p90,
            "p99": scale_p99,
        },
        "anisotropy": {
            "mean": report.mean_anisotropy,
            "high_fraction": report.high_anisotropy_fraction,
        },
        "color": {
            "mean": report.color_distribution.mean,
            "std": report.color_distribution.std,
            "dominant_channel": report.color_distribution.dominant_channel,
            "grayscale_fraction": report.color_distribution.grayscale_fraction,
        },
        "spatial_outlier_fraction": report.spatial_outlier_fraction,
        "density": density,
    });

    let mut artifacts: Vec<(&str, &Path)> = Vec::new();
    let mut dumped = false;
    if let Some(ref csv_path) = args.dump_csv {
        if args.dump_limit == 0 {
            anyhow::bail!("--dump-limit must be at least 1");
        }
        if prepare_output(ctx, csv_path, args.force)? {
            let indices: Vec<usize> = (0..model.n.min(args.dump_limit)).collect();
            let csv = dump_gaussians_csv(&model, &indices)?;
            std::fs::write(csv_path, csv)
                .with_context(|| format!("Failed to write {}", csv_path.display()))?;
            artifacts.push(("csv", csv_path.as_path()));
            dumped = true;
        }
        if let Some(map) = payload.as_object_mut() {
            map.insert(
                "dump_csv".to_string(),
                json!(csv_path.display().to_string()),
            );
            map.insert("dump_written".to_string(), json!(dumped));
        }
    }

    let summary = format_inspection_report(&report);
    emit(ctx, "inspect model", payload, &artifacts, || {
        println!("{summary}");
        println!(
            "  max-scale percentiles: p50={scale_p50:.6} p90={scale_p90:.6} p99={scale_p99:.6}"
        );
        if dumped {
            if let Some(ref csv_path) = args.dump_csv {
                println!("Wrote {}", csv_path.display());
            }
        }
    });
    Ok(())
}

// ---------------------------------------------------------------------------
// inspect query
// ---------------------------------------------------------------------------

/// Arguments for `oxigaf inspect query`.
///
/// Exactly one selector must be given.
#[derive(Debug, Args)]
pub struct QueryArgs {
    /// Model to query (`.ply` or `.json` checkpoint).
    pub model: PathBuf,

    /// Centre of a spherical region, as `x,y,z`.
    #[arg(long, value_parser = parse_vec3, requires = "radius")]
    pub sphere: Option<[f32; 3]>,

    /// Radius of the spherical region.
    #[arg(long, requires = "sphere")]
    pub radius: Option<f32>,

    /// Lower corner of an axis-aligned region, as `x,y,z`.
    #[arg(long, value_parser = parse_vec3, requires = "bbox_max")]
    pub bbox_min: Option<[f32; 3]>,

    /// Upper corner of an axis-aligned region, as `x,y,z`.
    #[arg(long, value_parser = parse_vec3, requires = "bbox_min")]
    pub bbox_max: Option<[f32; 3]>,

    /// Find the nearest Gaussians to this point, as `x,y,z`.
    #[arg(long, value_parser = parse_vec3)]
    pub knn: Option<[f32; 3]>,

    /// How many neighbours `--knn` returns.
    #[arg(long, default_value = "10")]
    pub k: usize,

    /// Select Gaussians whose opacity probability is below this.
    #[arg(long)]
    pub low_opacity: Option<f32>,

    /// Select Gaussians more elongated than this max/min scale ratio.
    #[arg(long)]
    pub high_anisotropy: Option<f32>,

    /// Select Gaussians whose largest extent exceeds this.
    #[arg(long)]
    pub larger_than: Option<f32>,

    /// Select Gaussians farther than this from the centroid.
    #[arg(long)]
    pub outliers: Option<f32>,

    /// Write the matched Gaussians to this CSV file.
    #[arg(long)]
    pub dump_csv: Option<PathBuf>,

    /// Overwrite the CSV dump if it exists.
    #[arg(long)]
    pub force: bool,

    /// How many matched indices to print in human mode.
    #[arg(long, default_value = "20")]
    pub show: usize,
}

fn cmd_query(args: QueryArgs, ctx: &CmdContext) -> Result<()> {
    let selectors = usize::from(args.sphere.is_some())
        + usize::from(args.bbox_min.is_some())
        + usize::from(args.knn.is_some())
        + usize::from(args.low_opacity.is_some())
        + usize::from(args.high_anisotropy.is_some())
        + usize::from(args.larger_than.is_some())
        + usize::from(args.outliers.is_some());
    if selectors != 1 {
        anyhow::bail!(
            "Give exactly one selector: --sphere/--radius, --bbox-min/--bbox-max, --knn, \
             --low-opacity, --high-anisotropy, --larger-than or --outliers (got {selectors})."
        );
    }

    let model = inspectable(&args.model)?;

    // `distances` is only populated by the k-nearest-neighbour selector.
    let mut distances: Option<Vec<(usize, f32)>> = None;
    let result: QueryResult = if let (Some(center), Some(radius)) = (args.sphere, args.radius) {
        if !(radius.is_finite() && radius > 0.0) {
            anyhow::bail!("--radius must be finite and positive (got {radius})");
        }
        query_sphere(&model, center, radius)?
    } else if let (Some(min), Some(max)) = (args.bbox_min, args.bbox_max) {
        for axis in 0..3 {
            if min[axis] > max[axis] {
                anyhow::bail!(
                    "--bbox-min component {axis} ({}) exceeds --bbox-max ({})",
                    min[axis],
                    max[axis]
                );
            }
        }
        query_aabb(&model, &BoundingBox3d::new(min, max))?
    } else if let Some(point) = args.knn {
        if args.k == 0 {
            anyhow::bail!("--k must be at least 1");
        }
        let neighbours = query_knn(&model, point, args.k)?;
        let indices: Vec<usize> = neighbours.iter().map(|(index, _)| *index).collect();
        let count = indices.len();
        distances = Some(neighbours);
        QueryResult {
            indices,
            count,
            fraction: if model.n == 0 {
                0.0
            } else {
                count as f32 / model.n as f32
            },
        }
    } else if let Some(threshold) = args.low_opacity {
        find_low_opacity(&model, threshold)
    } else if let Some(threshold) = args.high_anisotropy {
        find_high_anisotropy(&model, threshold)
    } else if let Some(threshold) = args.larger_than {
        find_large_gaussians(&model, threshold)
    } else if let Some(distance) = args.outliers {
        find_spatial_outliers(&model, distance)?
    } else {
        // Unreachable: the selector count above already rejected this.
        anyhow::bail!("No selector given");
    };

    let shown: Vec<usize> = result.indices.iter().copied().take(args.show).collect();
    let mut payload = json!({
        "model": args.model.display().to_string(),
        "matched": result.count,
        "total": model.n,
        "fraction": result.fraction,
        "indices": shown,
        "neighbour_distances": distances
            .as_ref()
            .map(|pairs| pairs.iter().map(|(_, distance)| *distance).collect::<Vec<_>>()),
    });

    let mut artifacts: Vec<(&str, &Path)> = Vec::new();
    let mut dumped = false;
    if let Some(ref csv_path) = args.dump_csv {
        if prepare_output(ctx, csv_path, args.force)? {
            let csv = dump_gaussians_csv(&model, &result.indices)?;
            std::fs::write(csv_path, csv)
                .with_context(|| format!("Failed to write {}", csv_path.display()))?;
            artifacts.push(("csv", csv_path.as_path()));
            dumped = true;
        }
        if let Some(map) = payload.as_object_mut() {
            map.insert(
                "dump_csv".to_string(),
                json!(csv_path.display().to_string()),
            );
            map.insert("dump_written".to_string(), json!(dumped));
        }
    }

    emit(ctx, "inspect query", payload, &artifacts, || {
        println!(
            "Matched {} of {} Gaussians ({:.2}%)",
            result.count,
            model.n,
            result.fraction * 100.0
        );
        if !shown.is_empty() {
            println!("  first indices: {shown:?}");
        }
        if dumped {
            if let Some(ref csv_path) = args.dump_csv {
                println!("Wrote {}", csv_path.display());
            }
        }
    });
    Ok(())
}

// ---------------------------------------------------------------------------
// inspect memory
// ---------------------------------------------------------------------------

/// Arguments for `oxigaf inspect memory`.
#[derive(Debug, Args)]
pub struct MemoryArgs {
    /// Size the estimate from this model instead of `--gaussians`.
    #[arg(long)]
    pub model: Option<PathBuf>,

    /// Gaussian count to estimate for.
    #[arg(long, default_value = "1000000")]
    pub gaussians: usize,

    /// Spherical-harmonics degree (0–3).
    #[arg(long, default_value = "3")]
    pub sh_degree: u32,

    /// Render width in pixels.
    #[arg(long, default_value = "1024")]
    pub width: usize,

    /// Render height in pixels.
    #[arg(long, default_value = "1024")]
    pub height: usize,

    /// Views rendered per training step.
    #[arg(long, default_value = "4")]
    pub views: usize,

    /// Do not budget for the exponential-moving-average copy.
    #[arg(long)]
    pub no_ema: bool,

    /// Do not budget for optimiser state (inference-only estimate).
    #[arg(long)]
    pub no_optimizer: bool,

    /// Budget half-precision optimiser moments instead of f32.
    #[arg(long)]
    pub fp16_optimizer: bool,

    /// Parameter count of the diffusion prior held on the device.
    #[arg(long, default_value = "860000000")]
    pub diffusion_params: usize,

    /// Bytes per diffusion parameter (2 for fp16, 4 for f32).
    #[arg(long, default_value = "2")]
    pub diffusion_dtype_bytes: usize,

    /// Device VRAM budget in gibibytes.
    #[arg(long, default_value = "24")]
    pub vram_gb: f64,

    /// Also compare against this Gaussian count.
    #[arg(long)]
    pub compare_gaussians: Option<usize>,
}

impl MemoryArgs {
    fn config(
        &self,
        n_gaussians: usize,
        sh_degree: u32,
        target_vram_bytes: usize,
    ) -> MemEstimateConfig {
        MemEstimateConfig {
            n_gaussians,
            sh_degree,
            render_width: self.width,
            render_height: self.height,
            use_ema: !self.no_ema,
            use_optimizer: !self.no_optimizer,
            n_render_views: self.views,
            diffusion_model_params: self.diffusion_params,
            target_vram_bytes,
            fp16_optimizer: self.fp16_optimizer,
        }
    }
}

fn cmd_memory(args: MemoryArgs, ctx: &CmdContext) -> Result<()> {
    if args.sh_degree > 3 {
        anyhow::bail!("--sh-degree must be within 0..=3 (got {})", args.sh_degree);
    }
    if !(args.vram_gb.is_finite() && args.vram_gb > 0.0) {
        anyhow::bail!("--vram-gb must be finite and positive");
    }
    if args.diffusion_dtype_bytes == 0 {
        anyhow::bail!("--diffusion-dtype-bytes must be at least 1");
    }

    // A model, when given, decides the count and degree the estimate is for.
    let (n_gaussians, sh_degree, source) = match args.model {
        Some(ref path) => {
            let model = load_scene(path)?;
            (model.len(), model.sh_degree, path.display().to_string())
        }
        None => (args.gaussians, args.sh_degree, "flags".to_string()),
    };
    if n_gaussians == 0 {
        anyhow::bail!("Nothing to estimate: the Gaussian count is zero");
    }

    let target_vram_bytes = (args.vram_gb * 1024.0 * 1024.0 * 1024.0) as usize;
    let config = args.config(n_gaussians, sh_degree, target_vram_bytes);
    let estimate = estimate_memory(&config)?;
    let breakdown = memory_breakdown_percent(&estimate);

    let model_weights = estimate_model_weights(args.diffusion_params, args.diffusion_dtype_bytes);
    let capacity = max_gaussians_for_vram(
        target_vram_bytes,
        sh_degree,
        args.width,
        args.height,
        !args.no_ema,
        !args.no_optimizer,
        model_weights,
    )?;

    let comparison = match args.compare_gaussians {
        Some(other) if other > 0 => {
            let other_config = args.config(other, sh_degree, target_vram_bytes);
            let delta = compare_memory_configs(&config, &other_config)?;
            Some(json!({
                "n_gaussians": other,
                "a_total_bytes": delta.a_total_bytes,
                "b_total_bytes": delta.b_total_bytes,
                "delta_bytes": delta.delta_bytes,
                "delta_percent": delta.delta_pct,
            }))
        }
        Some(_) => anyhow::bail!("--compare-gaussians must be at least 1"),
        None => None,
    };

    let payload = json!({
        "source": source,
        "n_gaussians": n_gaussians,
        "sh_degree": sh_degree,
        "resolution": [args.width, args.height],
        "gaussian_bytes": estimate.gaussians.total_bytes,
        "render_buffer_bytes": estimate.render_buffers.total_bytes,
        "training_bytes": estimate.training.total_bytes,
        "model_weights_bytes": estimate.model_weights_bytes,
        "total_gpu_bytes": estimate.total_gpu_bytes,
        "total_cpu_bytes": estimate.total_cpu_bytes,
        "recommended_vram_gb": estimate.recommended_vram_gb,
        "recommended_ram_gb": estimate.recommended_ram_gb,
        "fits_in_vram": estimate.fits_in_vram,
        "target_vram_bytes": estimate.target_vram_bytes,
        "breakdown_percent": {
            "gaussians": breakdown.gaussians_pct,
            "render": breakdown.render_pct,
            "training": breakdown.training_pct,
            "model": breakdown.model_pct,
        },
        "max_gaussians_for_vram": capacity,
        "comparison": comparison,
    });

    let summary = format_memory_estimate(&estimate);
    emit(ctx, "inspect memory", payload, &[], || {
        println!("{summary}");
        println!(
            "  budget {} fits at most {capacity} Gaussians",
            mem_format_bytes(target_vram_bytes)
        );
    });
    Ok(())
}

// ---------------------------------------------------------------------------
// inspect ply
// ---------------------------------------------------------------------------

/// Arguments for `oxigaf inspect ply`.
#[derive(Debug, Args)]
pub struct PlyArgs {
    /// PLY file to inspect.
    pub path: PathBuf,

    /// Also list every vertex property declared in the header.
    #[arg(long)]
    pub properties: bool,
}

/// Read a PLY file's ASCII header (everything up to and including
/// `end_header`), so the header parsers can be run against it.
fn read_ply_header(path: &Path) -> Result<String> {
    let raw = std::fs::read(path).with_context(|| format!("Failed to read {}", path.display()))?;
    let marker = b"end_header";
    let end = raw
        .windows(marker.len())
        .position(|window| window == marker)
        .ok_or_else(|| anyhow::anyhow!("{} has no 'end_header' line", path.display()))?;
    Ok(String::from_utf8_lossy(&raw[..end + marker.len()]).into_owned())
}

fn cmd_ply(args: PlyArgs, ctx: &CmdContext) -> Result<()> {
    let header = read_ply_header(&args.path)?;
    let declared_format = ply_parse_format(&header)?;
    let declared_count = ply_parse_element_count(&header, "vertex")?;
    let properties = ply_parse_properties(&header)?;

    let (gaussians, read_stats) = ply_read(&args.path)?;
    let scene_stats = ply_compute_scene_stats(&gaussians);

    let payload = json!({
        "path": args.path.display().to_string(),
        "format": format!("{declared_format:?}"),
        "declared_vertex_count": declared_count,
        "read_vertex_count": read_stats.n_gaussians,
        "n_properties": read_stats.n_properties,
        "sh_degree": read_stats.sh_degree,
        "properties": if args.properties { Some(properties.clone()) } else { None },
        "mean_opacity": scene_stats.mean_opacity,
        "mean_scale": scene_stats.mean_scale,
        "bbox_min": scene_stats.bbox_min,
        "bbox_max": scene_stats.bbox_max,
    });

    let summary = ply_format_stats(&scene_stats);
    emit(ctx, "inspect ply", payload, &[], || {
        println!("Format: {declared_format:?}");
        println!("Declared vertices: {declared_count}");
        println!("{summary}");
        if args.properties {
            println!("Properties ({}):", properties.len());
            for name in &properties {
                println!("  {name}");
            }
        }
    });
    Ok(())
}

// ---------------------------------------------------------------------------
// inspect pointcloud
// ---------------------------------------------------------------------------

/// Arguments for `oxigaf inspect pointcloud`.
#[derive(Debug, Args)]
pub struct PointcloudArgs {
    /// Model whose point-cloud export should be summarised.
    pub model: PathBuf,

    /// Colour mode the export would use.
    #[arg(long, value_enum, default_value = "sh-dc")]
    pub color_mode: PointColorMode,
}

fn cmd_pointcloud(args: PointcloudArgs, ctx: &CmdContext) -> Result<()> {
    let model = load_scene(&args.model)?;
    let stats = PointCloudStats::compute(&model, args.color_mode);

    let payload = json!({
        "model": args.model.display().to_string(),
        "num_points": stats.num_points,
        "bbox_min": stats.bbox_min,
        "bbox_max": stats.bbox_max,
        "mean_opacity": stats.mean_opacity,
        "color_mode": format!("{:?}", stats.color_mode),
    });

    let summary = stats.format_summary();
    emit(ctx, "inspect pointcloud", payload, &[], || {
        println!("{summary}");
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::verbosity::Verbosity;

    fn quiet_ctx() -> CmdContext {
        CmdContext::new(Verbosity::Quiet, true, false)
    }

    fn base_query_args() -> QueryArgs {
        QueryArgs {
            model: PathBuf::from("scene.ply"),
            sphere: None,
            radius: None,
            bbox_min: None,
            bbox_max: None,
            knn: None,
            k: 10,
            low_opacity: None,
            high_anisotropy: None,
            larger_than: None,
            outliers: None,
            dump_csv: None,
            force: false,
            show: 20,
        }
    }

    fn base_memory_args() -> MemoryArgs {
        MemoryArgs {
            model: None,
            gaussians: 1_000,
            sh_degree: 1,
            width: 256,
            height: 256,
            views: 2,
            no_ema: false,
            no_optimizer: false,
            fp16_optimizer: false,
            diffusion_params: 1_000_000,
            diffusion_dtype_bytes: 2,
            vram_gb: 8.0,
            compare_gaussians: None,
        }
    }

    #[test]
    fn query_requires_exactly_one_selector() {
        assert!(cmd_query(base_query_args(), &quiet_ctx()).is_err());

        let mut two = base_query_args();
        two.low_opacity = Some(0.1);
        two.larger_than = Some(1.0);
        assert!(cmd_query(two, &quiet_ctx()).is_err());
    }

    #[test]
    fn memory_rejects_an_out_of_range_sh_degree() {
        let mut args = base_memory_args();
        args.sh_degree = 7;
        assert!(cmd_memory(args, &quiet_ctx()).is_err());
    }

    #[test]
    fn memory_rejects_a_zero_gaussian_count() {
        let mut args = base_memory_args();
        args.gaussians = 0;
        assert!(cmd_memory(args, &quiet_ctx()).is_err());
    }

    #[test]
    fn memory_estimate_grows_with_the_gaussian_count() {
        let small = base_memory_args();
        let target = (small.vram_gb * 1024.0 * 1024.0 * 1024.0) as usize;
        let a = small.config(1_000, 1, target);
        let b = small.config(100_000, 1, target);
        let delta = compare_memory_configs(&a, &b).expect("compare");
        assert!(
            delta.delta_bytes > 0,
            "100k Gaussians must cost more than 1k, got {}",
            delta.delta_bytes
        );
    }

    #[test]
    fn read_ply_header_rejects_a_file_without_end_header() {
        let path = std::env::temp_dir().join("oxigaf_inspect_bad_header.ply");
        std::fs::write(&path, b"ply\nformat ascii 1.0\n").expect("write");
        assert!(read_ply_header(&path).is_err());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn read_ply_header_stops_at_the_marker() {
        let path = std::env::temp_dir().join("oxigaf_inspect_header.ply");
        std::fs::write(
            &path,
            b"ply\nformat ascii 1.0\nelement vertex 2\nproperty float x\nend_header\n0 0 0\n",
        )
        .expect("write");
        let header = read_ply_header(&path).expect("header");
        assert!(header.ends_with("end_header"));
        assert_eq!(
            ply_parse_element_count(&header, "vertex").ok(),
            Some(2usize)
        );
        let _ = std::fs::remove_file(&path);
    }
}
