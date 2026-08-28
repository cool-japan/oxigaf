//! Size-reduction and record-format handlers for the `oxigaf scene` family.
//!
//! Split out of [`crate::commands::scene_tools`] purely to keep both files
//! under the 2000-line ceiling; the parser and dispatch live there, the
//! heavier handlers here:
//!
//! * `scene dedup` — [`crate::gaussian_deduplicator`]
//! * `scene compress` — [`crate::gaussian_compressor`]
//! * `scene lod` — [`crate::lod_generator`]
//! * `scene convert` — [`crate::format_converter`]

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Args, ValueEnum};
use serde_json::json;

use oxigaf::render::gaussian::GaussianModel;

use crate::commands::model_io::{
    load_scene, model_from_arrays, save_scene, sh_total_for_degree, warn_if_binding_dropped,
    FlatScene, SceneArrays,
};
use crate::commands::scene_tools::report_output;
use crate::commands::{emit, prepare_output, CmdContext};
use crate::format_converter::{
    compute_conversion_stats, convert as convert_records, filter_valid, from_binary, from_csv,
    from_json, validate_record, FileFormat, GaussianRecord,
};
use crate::gaussian_compressor::{
    gc_compress, gc_compute_stats, gc_decompress, gc_format_config, gc_format_stats,
    CompressionConfig, GcSceneSlices, KMeansConfig, QuantizationPrecision, ScenePruningConfig,
};
use crate::gaussian_deduplicator::{
    gd_analyze_duplicates, gd_build_report, gd_deduplicate, gd_format_report, DedupConfig,
    DedupKeepPolicy, GdDeduplicateInput,
};
use crate::lod_generator::{
    compute_lod_stats, find_optimal_reduction_ratios, format_lod_stats, generate_lod_chain,
    LodConfig, LodSelector, LodStrategy,
};

// ---------------------------------------------------------------------------
// scene dedup
// ---------------------------------------------------------------------------

/// Which member of a duplicate group survives.
#[derive(Debug, Clone, Copy, ValueEnum, Default)]
pub enum KeepPolicy {
    /// Keep the most opaque Gaussian.
    #[default]
    HighestOpacity,
    /// Keep the physically largest Gaussian.
    LargestScale,
    /// Keep the smallest Gaussian.
    SmallestScale,
    /// Keep the one appearing first in the file.
    First,
    /// Keep the one appearing last in the file.
    Last,
}

impl From<KeepPolicy> for DedupKeepPolicy {
    fn from(value: KeepPolicy) -> Self {
        match value {
            KeepPolicy::HighestOpacity => DedupKeepPolicy::KeepHighestOpacity,
            KeepPolicy::LargestScale => DedupKeepPolicy::KeepLargestScale,
            KeepPolicy::SmallestScale => DedupKeepPolicy::KeepSmallestScale,
            KeepPolicy::First => DedupKeepPolicy::KeepFirst,
            KeepPolicy::Last => DedupKeepPolicy::KeepLast,
        }
    }
}

/// Arguments for `oxigaf scene dedup`.
///
/// Thresholds compare the values as stored in the file: opacity is compared
/// in **logit** space and colour as raw DC spherical-harmonic coefficients.
#[derive(Debug, Args)]
pub struct DedupArgs {
    /// Scene to deduplicate (`.ply` or `.json` checkpoint).
    pub input: PathBuf,

    /// Write the deduplicated scene here. Omit to only report the analysis.
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Overwrite the output file if it exists.
    #[arg(long)]
    pub force: bool,

    /// Maximum centre distance for two Gaussians to count as duplicates.
    #[arg(long, default_value = "0.001")]
    pub position_threshold: f32,

    /// Maximum logit-opacity difference.
    #[arg(long, default_value = "0.05")]
    pub opacity_threshold: f32,

    /// Maximum relative difference in largest scale.
    #[arg(long, default_value = "0.1")]
    pub scale_threshold: f32,

    /// Maximum DC-colour distance.
    #[arg(long, default_value = "0.1")]
    pub color_threshold: f32,

    /// Which member of a duplicate group to keep.
    #[arg(long, value_enum, default_value = "highest-opacity")]
    pub keep: KeepPolicy,

    /// Compare every pair instead of using the spatial hash (O(n²); exact).
    #[arg(long)]
    pub brute_force: bool,

    /// Spatial-hash cell size; ignored with `--brute-force`.
    #[arg(long, default_value = "0.002")]
    pub cell_size: f32,

    /// Report the largest duplicate groups as well as the summary.
    #[arg(long)]
    pub report_groups: bool,
}

/// Run `oxigaf scene dedup`.
///
/// # Errors
///
/// Returns an error for an unreadable scene, an invalid threshold, or a
/// refused overwrite.
pub fn cmd_dedup(args: DedupArgs, ctx: &CmdContext) -> Result<()> {
    if args.position_threshold <= 0.0 {
        anyhow::bail!("--position-threshold must be positive");
    }
    if !args.brute_force && args.cell_size <= 0.0 {
        anyhow::bail!("--cell-size must be positive");
    }

    let model = load_scene(&args.input)?;
    let flat = FlatScene::from_model(&model)?;
    if flat.n == 0 {
        anyhow::bail!("Scene is empty: {}", args.input.display());
    }

    let config = DedupConfig {
        position_threshold: args.position_threshold,
        opacity_threshold: args.opacity_threshold,
        scale_threshold: args.scale_threshold,
        color_threshold: args.color_threshold,
        keep_policy: args.keep.into(),
        use_spatial_hash: !args.brute_force,
        cell_size: args.cell_size,
    };

    let sh_channels = sh_total_for_degree(flat.sh_degree);
    let sh_coeffs = flat.sh_coeffs();

    let result = gd_deduplicate(
        GdDeduplicateInput {
            positions: &flat.positions,
            rotations: &flat.rotations,
            scales: &flat.log_scales,
            opacities: &flat.opacity_logits,
            sh_coefficients: &sh_coeffs,
            sh_channels,
            n_gaussians: flat.n,
        },
        &config,
    )?;

    let groups = if args.report_groups {
        gd_analyze_duplicates(
            &flat.positions,
            &flat.opacity_logits,
            &flat.log_scales,
            &sh_coeffs,
            sh_channels,
            flat.n,
            &config,
        )?
    } else {
        Vec::new()
    };
    let report = gd_build_report(&result, groups, &config, sh_channels);

    let mut payload = json!({
        "input": args.input.display().to_string(),
        "before": report.stats.n_before,
        "after": report.stats.n_after,
        "removed": result.n_removed,
        "reduction_percent": report.stats.reduction_percent,
        "groups": report.stats.n_groups,
        "mean_group_size": report.stats.mean_group_size,
        "max_group_size": report.stats.max_group_size,
        "memory_saved_bytes": report.stats.memory_saved_bytes,
        "largest_groups": report
            .largest_groups
            .iter()
            .map(|group| json!({
                "size": group.indices.len(),
                "centroid": group.centroid,
                "mean_opacity": group.mean_opacity,
                "max_position_spread": group.max_position_spread,
            }))
            .collect::<Vec<_>>(),
    });

    let mut artifacts: Vec<(&str, &Path)> = Vec::new();
    let mut written = false;
    if let Some(ref output) = args.output {
        if prepare_output(ctx, output, args.force)? {
            let deduplicated = model_from_arrays(SceneArrays {
                positions: result.positions.clone(),
                rotations: result.rotations.clone(),
                log_scales: result.scales.clone(),
                opacity_logits: result.opacities.clone(),
                sh_coeffs: result.sh_coefficients.clone(),
                sh_degree: flat.sh_degree,
            })?;
            warn_if_binding_dropped(&model, output);
            save_scene(&deduplicated, output)?;
            artifacts.push(("scene", output.as_path()));
            written = true;
        }
        if let Some(map) = payload.as_object_mut() {
            map.insert("output".to_string(), json!(output.display().to_string()));
            map.insert("written".to_string(), json!(written));
        }
    }

    let summary = gd_format_report(&report);
    emit(ctx, "scene dedup", payload, &artifacts, || {
        println!("{summary}");
        report_output(ctx, args.output.as_deref(), written);
    });
    Ok(())
}

// ---------------------------------------------------------------------------
// scene compress
// ---------------------------------------------------------------------------

/// Scalar quantisation width for one attribute group.
#[derive(Debug, Clone, Copy, ValueEnum, Default)]
pub enum Precision {
    /// No quantisation — values are kept as `f32`.
    Full,
    /// 16-bit integer quantisation.
    #[default]
    Half,
    /// 8-bit integer quantisation.
    Byte,
}

impl From<Precision> for QuantizationPrecision {
    fn from(value: Precision) -> Self {
        match value {
            Precision::Full => QuantizationPrecision::Full,
            Precision::Half => QuantizationPrecision::Half,
            Precision::Byte => QuantizationPrecision::Byte,
        }
    }
}

/// Arguments for `oxigaf scene compress`.
///
/// The compressed representation has no on-disk container in this crate, so
/// `--output` writes the *dequantised* scene: exactly what a decoder would
/// reconstruct, which is what makes the reported error meaningful. The
/// reported ratio describes the in-memory compressed form.
#[derive(Debug, Args)]
pub struct CompressArgs {
    /// Scene to compress (`.ply` or `.json` checkpoint).
    pub input: PathBuf,

    /// Write the round-tripped (pruned and dequantised) scene here.
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Overwrite the output file if it exists.
    #[arg(long)]
    pub force: bool,

    /// Quantisation width for positions.
    #[arg(long, value_enum, default_value = "half")]
    pub position_precision: Precision,

    /// Quantisation width for rotations.
    #[arg(long, value_enum, default_value = "half")]
    pub rotation_precision: Precision,

    /// Quantisation width for log-scales.
    #[arg(long, value_enum, default_value = "half")]
    pub scale_precision: Precision,

    /// Quantisation width for opacities.
    #[arg(long, value_enum, default_value = "half")]
    pub opacity_precision: Precision,

    /// Quantisation width for the DC spherical-harmonic term.
    #[arg(long, value_enum, default_value = "half")]
    pub sh_dc_precision: Precision,

    /// Quantisation width for the higher-order spherical harmonics.
    #[arg(long, value_enum, default_value = "half")]
    pub sh_rest_precision: Precision,

    /// Drop Gaussians whose opacity probability is below this.
    #[arg(long, default_value = "0.01")]
    pub prune_opacity: f32,

    /// Drop Gaussians with any log-scale above this.
    #[arg(long, default_value = "6.0")]
    pub max_log_scale: f32,

    /// Drop Gaussians with every log-scale below this.
    #[arg(long, default_value = "-10.0")]
    pub min_log_scale: f32,

    /// Keep at most this many Gaussians, the most opaque first.
    #[arg(long)]
    pub target_gaussians: Option<usize>,

    /// Keep this fraction of the most opaque Gaussians unconditionally.
    #[arg(long, default_value = "1.0")]
    pub preserve_top_fraction: f32,

    /// Also cluster positions with k-means before quantising.
    #[arg(long)]
    pub cluster: bool,

    /// Number of k-means clusters; only used with `--cluster`.
    #[arg(long, default_value = "256")]
    pub clusters: usize,

    /// k-means iterations; only used with `--cluster`.
    #[arg(long, default_value = "50")]
    pub cluster_iterations: usize,
}

/// Run `oxigaf scene compress`.
///
/// # Errors
///
/// Returns an error for an unreadable scene, an out-of-range threshold, or a
/// refused overwrite.
pub fn cmd_compress(args: CompressArgs, ctx: &CmdContext) -> Result<()> {
    if !(0.0..=1.0).contains(&args.prune_opacity) {
        anyhow::bail!(
            "--prune-opacity is a probability and must lie in [0, 1] (got {})",
            args.prune_opacity
        );
    }
    if !(0.0..=1.0).contains(&args.preserve_top_fraction) {
        anyhow::bail!(
            "--preserve-top-fraction must lie in [0, 1] (got {})",
            args.preserve_top_fraction
        );
    }
    if args.cluster && args.clusters == 0 {
        anyhow::bail!("--clusters must be at least 1");
    }

    let model = load_scene(&args.input)?;
    let flat = FlatScene::from_model(&model)?;
    if flat.n == 0 {
        anyhow::bail!("Scene is empty: {}", args.input.display());
    }

    let config = CompressionConfig {
        position_precision: args.position_precision.into(),
        rotation_precision: args.rotation_precision.into(),
        scale_precision: args.scale_precision.into(),
        opacity_precision: args.opacity_precision.into(),
        sh_dc_precision: args.sh_dc_precision.into(),
        sh_rest_precision: args.sh_rest_precision.into(),
        pruning: ScenePruningConfig {
            opacity_threshold: args.prune_opacity,
            max_log_scale: args.max_log_scale,
            min_log_scale: args.min_log_scale,
            target_n_gaussians: args.target_gaussians,
            preserve_top_fraction: args.preserve_top_fraction,
        },
        use_position_clustering: args.cluster,
        kmeans: KMeansConfig {
            n_clusters: args.clusters,
            n_iterations: args.cluster_iterations,
            tolerance: 1e-4,
        },
    };

    let compressed = gc_compress(
        GcSceneSlices {
            positions: &flat.positions,
            rotations: &flat.rotations,
            scales: &flat.log_scales,
            opacities: &flat.opacity_logits,
            sh_dc: &flat.sh_dc,
            sh_rest: &flat.sh_rest,
            n_rest_per_gaussian: flat.n_rest_per_gaussian,
        },
        &config,
    )?;
    let stats = gc_compute_stats(&flat.positions, &flat.opacity_logits, &compressed)?;

    let mut payload = json!({
        "input": args.input.display().to_string(),
        "before": stats.n_gaussians_before,
        "after": stats.n_gaussians_after,
        "pruned_fraction": stats.pruned_fraction,
        "uncompressed_mb": stats.uncompressed_mb,
        "compressed_mb": stats.compressed_mb,
        "compression_ratio": stats.compression_ratio,
        "position_rmse": stats.position_quantization_rmse,
        "opacity_rmse": stats.opacity_quantization_rmse,
        "clustered": args.cluster,
    });

    let mut artifacts: Vec<(&str, &Path)> = Vec::new();
    let mut written = false;
    if let Some(ref output) = args.output {
        if prepare_output(ctx, output, args.force)? {
            let decompressed = gc_decompress(&compressed)?;
            let n = decompressed.n_gaussians;
            let rest = compressed.n_sh_rest;
            if decompressed.sh_dc.len() != n * 3 || decompressed.sh_rest.len() != n * rest {
                anyhow::bail!(
                    "Decompressed scene has inconsistent SH buffers: {} DC and {} rest values \
                     for {n} Gaussians",
                    decompressed.sh_dc.len(),
                    decompressed.sh_rest.len(),
                );
            }
            let mut sh_coeffs = Vec::with_capacity(n * (3 + rest));
            for i in 0..n {
                sh_coeffs.extend_from_slice(&decompressed.sh_dc[i * 3..i * 3 + 3]);
                if rest > 0 {
                    sh_coeffs.extend_from_slice(&decompressed.sh_rest[i * rest..i * rest + rest]);
                }
            }
            let rebuilt = model_from_arrays(SceneArrays {
                positions: decompressed.positions,
                rotations: decompressed.rotations,
                log_scales: decompressed.scales,
                opacity_logits: decompressed.opacities,
                sh_coeffs,
                sh_degree: flat.sh_degree,
            })?;
            warn_if_binding_dropped(&model, output);
            save_scene(&rebuilt, output)?;
            artifacts.push(("scene", output.as_path()));
            written = true;
        }
        if let Some(map) = payload.as_object_mut() {
            map.insert("output".to_string(), json!(output.display().to_string()));
            map.insert("written".to_string(), json!(written));
        }
    }

    let config_summary = gc_format_config(&config);
    let stats_summary = gc_format_stats(&stats);
    emit(ctx, "scene compress", payload, &artifacts, || {
        println!("{config_summary}");
        println!("{stats_summary}");
        report_output(ctx, args.output.as_deref(), written);
    });
    Ok(())
}

// ---------------------------------------------------------------------------
// scene lod
// ---------------------------------------------------------------------------

/// How lower LOD levels choose which Gaussians to keep.
#[derive(Debug, Clone, Copy, ValueEnum, Default)]
pub enum LodPick {
    /// Keep the most opaque Gaussians.
    #[default]
    TopOpacity,
    /// Keep evenly spaced Gaussians.
    Uniform,
    /// Keep a spatially even spread using grid sampling.
    SpatialGrid,
    /// Keep a deterministic pseudo-random subset.
    Random,
}

impl From<LodPick> for LodStrategy {
    fn from(value: LodPick) -> Self {
        match value {
            LodPick::TopOpacity => LodStrategy::TopOpacity,
            LodPick::Uniform => LodStrategy::Uniform,
            LodPick::SpatialGrid => LodStrategy::SpatialGrid,
            LodPick::Random => LodStrategy::Random,
        }
    }
}

/// Parse a comma-separated list of reduction ratios for a clap `value_parser`.
fn parse_ratios(raw: &str) -> std::result::Result<Vec<f32>, String> {
    let mut out = Vec::new();
    for text in raw.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let value: f32 = text
            .parse()
            .map_err(|_| format!("{text:?} is not a valid number"))?;
        if !(value.is_finite() && value > 0.0 && value <= 1.0) {
            return Err(format!("{text:?} is not a ratio in (0, 1]"));
        }
        out.push(value);
    }
    if out.is_empty() {
        return Err("expected at least one ratio".to_string());
    }
    Ok(out)
}

/// Arguments for `oxigaf scene lod`.
#[derive(Debug, Args)]
pub struct LodArgs {
    /// Scene to build a LOD chain from (`.ply` or `.json` checkpoint).
    pub input: PathBuf,

    /// Directory for the per-level PLY files. Omit to only report the plan.
    #[arg(short, long)]
    pub output_dir: Option<PathBuf>,

    /// File-name stem for the written levels.
    #[arg(long, default_value = "lod")]
    pub prefix: String,

    /// Overwrite existing level files.
    #[arg(long)]
    pub force: bool,

    /// Number of levels, level 0 being the full-resolution scene.
    #[arg(long, default_value = "4")]
    pub levels: usize,

    /// Explicit non-ascending ratios, e.g. `1.0,0.5,0.25`.
    #[arg(long, conflicts_with = "target_memory_mb")]
    pub ratios: Option<String>,

    /// Solve for ratios that fit the whole chain into this memory budget.
    #[arg(long)]
    pub target_memory_mb: Option<f64>,

    /// Selection strategy for the reduced levels.
    #[arg(long, value_enum, default_value = "top-opacity")]
    pub strategy: LodPick,

    /// Pick by storage order instead of ranking by opacity first.
    ///
    /// Only affects the `uniform` and `random` strategies.
    #[arg(long)]
    pub unsorted: bool,

    /// Also report which level a viewer at this distance would select.
    #[arg(long)]
    pub select_distance: Option<f32>,
}

/// Run `oxigaf scene lod`.
///
/// # Errors
///
/// Returns an error for an unreadable scene, an invalid ratio list, or a
/// refused overwrite.
pub fn cmd_lod(args: LodArgs, ctx: &CmdContext) -> Result<()> {
    if args.levels == 0 {
        anyhow::bail!("--levels must be at least 1");
    }

    let model = load_scene(&args.input)?;
    let flat = FlatScene::from_model(&model)?;
    if flat.n == 0 {
        anyhow::bail!("Scene is empty: {}", args.input.display());
    }
    let sh_total = sh_total_for_degree(flat.sh_degree);

    let ratios = match (&args.ratios, args.target_memory_mb) {
        (Some(explicit), _) => parse_ratios(explicit)
            .map_err(|reason| anyhow::anyhow!("--ratios {explicit:?} is invalid: {reason}"))?,
        (None, Some(budget_mb)) => {
            if !(budget_mb.is_finite() && budget_mb > 0.0) {
                anyhow::bail!("--target-memory-mb must be finite and positive");
            }
            let budget_bytes = (budget_mb * 1024.0 * 1024.0) as usize;
            find_optimal_reduction_ratios(flat.n, budget_bytes, args.levels, sh_total)?
        }
        // Halve at every step: 1.0, 0.5, 0.25, … — always non-ascending and
        // inside (0, 1], so `LodConfig::validate` accepts it for any level
        // count the user asks for.
        (None, None) => (0..args.levels)
            .map(|level| 0.5_f32.powi(level as i32))
            .collect(),
    };
    if ratios.len() != args.levels {
        anyhow::bail!(
            "The reduction-ratio list has {} entry/entries but --levels is {}",
            ratios.len(),
            args.levels
        );
    }

    let config = LodConfig {
        n_levels: args.levels,
        reduction_ratios: ratios.clone(),
        strategy: args.strategy.into(),
        sort_by_opacity: !args.unsorted,
    };
    config.validate()?;

    let sh_coeffs = flat.sh_coeffs();
    let chain = generate_lod_chain(
        &flat.positions,
        &flat.rotations,
        &flat.log_scales,
        &flat.opacity_logits,
        &sh_coeffs,
        &config,
    )?;
    let stats = compute_lod_stats(&chain);

    let selected_level = match args.select_distance {
        Some(distance) => Some(
            chain
                .select(distance, &LodSelector::default())
                .map(|level| level.level)?,
        ),
        None => None,
    };

    let mut written_paths: Vec<PathBuf> = Vec::new();
    if let Some(ref dir) = args.output_dir {
        for level in &chain.levels {
            let path = dir.join(format!("{}_{}.ply", args.prefix, level.level));
            if !prepare_output(ctx, &path, args.force)? {
                continue;
            }
            let level_model = model_from_arrays(SceneArrays {
                positions: level.positions.clone(),
                rotations: level.rotations.clone(),
                log_scales: level.scales.clone(),
                opacity_logits: level.opacities.clone(),
                sh_coeffs: level.sh_coefficients.clone(),
                sh_degree: flat.sh_degree,
            })?;
            warn_if_binding_dropped(&model, &path);
            save_scene(&level_model, &path)
                .with_context(|| format!("Failed to write LOD level {}", level.level))?;
            written_paths.push(path);
        }
    }

    let payload = json!({
        "input": args.input.display().to_string(),
        "original_gaussians": stats.original_gaussians,
        "levels": stats.n_levels,
        "ratios": ratios,
        "strategy": format!("{:?}", config.strategy),
        "level_sizes": stats.level_sizes,
        "memory_estimates": stats.memory_estimates,
        "total_memory": stats.total_memory,
        "selected_level": selected_level,
        "written": written_paths
            .iter()
            .map(|p| p.display().to_string())
            .collect::<Vec<_>>(),
    });

    let artifacts: Vec<(&str, &Path)> = written_paths
        .iter()
        .map(|path| ("lod-level", path.as_path()))
        .collect();

    let summary = format_lod_stats(&stats);
    emit(ctx, "scene lod", payload, &artifacts, || {
        println!("{summary}");
        if let Some(level) = selected_level {
            if let Some(distance) = args.select_distance {
                println!("At distance {distance}: level {level}");
            }
        }
        for path in &written_paths {
            println!("Wrote {}", path.display());
        }
        if ctx.dry_run && args.output_dir.is_some() {
            println!("[dry-run] no level files were written");
        }
    });
    Ok(())
}

// ---------------------------------------------------------------------------
// scene convert
// ---------------------------------------------------------------------------

/// Record container formats understood by `scene convert`.
#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum RecordFormat {
    /// Comma-separated values, one row per Gaussian.
    Csv,
    /// A JSON array of Gaussian objects.
    Json,
    /// Compact little-endian binary with an `OXIGAF01` header.
    Binary,
}

impl From<RecordFormat> for FileFormat {
    fn from(value: RecordFormat) -> Self {
        match value {
            RecordFormat::Csv => FileFormat::Csv,
            RecordFormat::Json => FileFormat::Json,
            RecordFormat::Binary => FileFormat::Binary,
        }
    }
}

/// Arguments for `oxigaf scene convert`.
///
/// Converts between the scene formats the renderer reads (`.ply`, `.json`
/// checkpoints) and the flat record containers (`.csv`, `.json`, `.bin`).
/// The direction is inferred from the file extensions; `--to` forces the
/// output container when the extension is ambiguous.
#[derive(Debug, Args)]
pub struct ConvertArgs {
    /// File to read.
    pub input: PathBuf,

    /// File to write.
    #[arg(short, long)]
    pub output: PathBuf,

    /// Force the output container instead of inferring it from the extension.
    #[arg(long, value_enum)]
    pub to: Option<RecordFormat>,

    /// Overwrite the output file if it exists.
    #[arg(long)]
    pub force: bool,

    /// Silently drop records containing NaN or infinite values.
    #[arg(long)]
    pub drop_invalid: bool,
}

/// What the input file turned out to hold.
enum Source {
    /// A renderable scene.
    Model(Box<GaussianModel>),
    /// Flat Gaussian records.
    Records(Vec<GaussianRecord>),
}

/// Read the input, distinguishing a checkpoint from a record file.
///
/// `.json` is ambiguous — a training checkpoint is an object, a record file
/// is an array — so it is resolved by the first non-whitespace byte rather
/// than by guessing from the extension.
fn read_source(path: &Path) -> Result<Source> {
    let extension = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or_default()
        .to_lowercase();

    match extension.as_str() {
        "ply" => Ok(Source::Model(Box::new(load_scene(path)?))),
        "csv" => {
            let data = std::fs::read(path)
                .with_context(|| format!("Failed to read {}", path.display()))?;
            Ok(Source::Records(from_csv(&data)?))
        }
        "bin" | "oxigaf" => {
            let data = std::fs::read(path)
                .with_context(|| format!("Failed to read {}", path.display()))?;
            Ok(Source::Records(from_binary(&data)?))
        }
        "json" | "jsonl" => {
            let data = std::fs::read(path)
                .with_context(|| format!("Failed to read {}", path.display()))?;
            let first = data.iter().find(|byte| !byte.is_ascii_whitespace());
            if first == Some(&b'[') {
                Ok(Source::Records(from_json(&data)?))
            } else {
                Ok(Source::Model(Box::new(load_scene(path)?)))
            }
        }
        other => anyhow::bail!(
            "Unsupported input format {other:?} for {}: expected .ply, .json, .csv or .bin",
            path.display()
        ),
    }
}

/// Flatten a model into records, preserving the raw (logit / log-space) values.
fn model_to_records(model: &GaussianModel) -> Result<Vec<GaussianRecord>> {
    let flat = FlatScene::from_model(model)?;
    let rest = flat.n_rest_per_gaussian;
    let mut records = Vec::with_capacity(flat.n);
    for i in 0..flat.n {
        let record = GaussianRecord::new(
            [
                flat.positions[i * 3],
                flat.positions[i * 3 + 1],
                flat.positions[i * 3 + 2],
            ],
            [
                flat.log_scales[i * 3],
                flat.log_scales[i * 3 + 1],
                flat.log_scales[i * 3 + 2],
            ],
            [
                flat.rotations[i * 4],
                flat.rotations[i * 4 + 1],
                flat.rotations[i * 4 + 2],
                flat.rotations[i * 4 + 3],
            ],
            flat.opacity_logits[i],
            [
                flat.sh_dc[i * 3],
                flat.sh_dc[i * 3 + 1],
                flat.sh_dc[i * 3 + 2],
            ],
        );
        records.push(if rest > 0 {
            record.with_sh_rest(flat.sh_rest[i * rest..i * rest + rest].to_vec())
        } else {
            record
        });
    }
    Ok(records)
}

/// Rebuild a model from records, requiring a single consistent SH degree.
fn records_to_model(records: &[GaussianRecord]) -> Result<GaussianModel> {
    let Some(first) = records.first() else {
        anyhow::bail!("No Gaussian records to convert");
    };
    let degree = first.sh_degree();
    let rest = first.sh_rest.len();
    let sh_degree = u32::try_from(degree)
        .map_err(|_| anyhow::anyhow!("Implausible spherical-harmonics degree {degree}"))?;

    let mut positions = Vec::with_capacity(records.len() * 3);
    let mut rotations = Vec::with_capacity(records.len() * 4);
    let mut log_scales = Vec::with_capacity(records.len() * 3);
    let mut opacity_logits = Vec::with_capacity(records.len());
    let mut sh_coeffs = Vec::with_capacity(records.len() * (3 + rest));

    for (index, record) in records.iter().enumerate() {
        if record.sh_rest.len() != rest {
            anyhow::bail!(
                "Record {index} carries {} SH rest coefficients but record 0 carries {rest}; \
                 a scene must have one uniform SH degree.",
                record.sh_rest.len()
            );
        }
        positions.extend_from_slice(&record.position);
        rotations.extend_from_slice(&record.rotation);
        log_scales.extend_from_slice(&record.log_scale);
        opacity_logits.push(record.opacity);
        sh_coeffs.extend_from_slice(&record.sh_dc);
        sh_coeffs.extend_from_slice(&record.sh_rest);
    }

    model_from_arrays(SceneArrays {
        positions,
        rotations,
        log_scales,
        opacity_logits,
        sh_coeffs,
        sh_degree,
    })
}

/// Run `oxigaf scene convert`.
///
/// # Errors
///
/// Returns an error for an unreadable or malformed input, an unsupported
/// output extension, or a refused overwrite.
pub fn cmd_convert(args: ConvertArgs, ctx: &CmdContext) -> Result<()> {
    let source = read_source(&args.input)?;
    let mut records = match &source {
        Source::Model(model) => model_to_records(model)?,
        Source::Records(records) => records.to_vec(),
    };

    let invalid: usize = records
        .iter()
        .filter(|record| !validate_record(record).is_empty())
        .count();
    if args.drop_invalid {
        records = filter_valid(records);
    } else if invalid > 0 {
        anyhow::bail!(
            "{invalid} of {} record(s) contain non-finite values. Re-run with --drop-invalid \
             to discard them.",
            records.len()
        );
    }
    if records.is_empty() {
        anyhow::bail!("Nothing left to write: the input holds no valid Gaussians");
    }

    let stats = compute_conversion_stats(&records);
    let output_extension = args
        .output
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or_default()
        .to_lowercase();

    // `--to` wins; otherwise `.ply` means "write a scene" and every other
    // known extension names a record container.
    let target_records: Option<FileFormat> = match args.to {
        Some(explicit) => Some(explicit.into()),
        None => {
            if output_extension == "ply" {
                None
            } else {
                match FileFormat::from_extension(&output_extension) {
                    Some(format) => Some(format),
                    None => anyhow::bail!(
                        "Cannot infer the output format for {}: pass --to csv|json|binary, \
                         or use a .ply/.csv/.json/.bin extension.",
                        args.output.display()
                    ),
                }
            }
        }
    };

    let target_name = match target_records {
        Some(format) => format.extension().to_string(),
        None => "ply".to_string(),
    };

    let payload = json!({
        "input": args.input.display().to_string(),
        "output": args.output.display().to_string(),
        "target": target_name,
        "records": stats.num_records,
        "valid": stats.num_valid,
        "dropped": if args.drop_invalid { invalid } else { 0 },
        "nan": stats.num_nan,
        "inf": stats.num_inf,
        "sh_degree": stats.sh_degree,
        "mean_opacity": stats.mean_opacity,
        "mean_max_scale": stats.mean_max_scale,
        "written": !ctx.dry_run,
    });

    let mut artifacts: Vec<(&str, &Path)> = Vec::new();
    let mut written = false;
    if prepare_output(ctx, &args.output, args.force)? {
        match target_records {
            Some(format) => {
                let bytes = convert_records(&records, format)?;
                std::fs::write(&args.output, &bytes)
                    .with_context(|| format!("Failed to write {}", args.output.display()))?;
            }
            None => {
                let model = records_to_model(&records)?;
                save_scene(&model, &args.output)?;
            }
        }
        artifacts.push(("converted", args.output.as_path()));
        written = true;
    }

    emit(ctx, "scene convert", payload, &artifacts, || {
        println!(
            "Converted {} record(s) to {target_name} (SH degree {})",
            stats.num_records, stats.sh_degree
        );
        if args.drop_invalid && invalid > 0 {
            println!("Dropped {invalid} record(s) with non-finite values");
        }
        report_output(ctx, Some(args.output.as_path()), written);
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::verbosity::Verbosity;
    use oxigaf::render::gaussian::GaussianAttributes;

    fn quiet_ctx() -> CmdContext {
        CmdContext::new(Verbosity::Quiet, true, false)
    }

    fn model_with_degree(n: usize, sh_degree: u32) -> GaussianModel {
        let sh_total = sh_total_for_degree(sh_degree);
        GaussianModel {
            gaussians: (0..n)
                .map(|i| GaussianAttributes {
                    position: [i as f32 * 0.1, 0.0, 0.0],
                    _pad0: 0.0,
                    rotation: [0.0, 0.0, 0.0, 1.0],
                    scale: [-2.0, -2.0, -2.0],
                    // Distinct opacities so opacity-ranked selection is
                    // deterministic rather than tie-broken arbitrarily.
                    opacity: 1.0 + i as f32 * 0.01,
                })
                .collect(),
            sh_coeffs: (0..n * sh_total).map(|i| i as f32 * 0.001).collect(),
            sh_degree,
            face_indices: Vec::new(),
            barycentric: Vec::new(),
            local_offsets: Vec::new(),
            is_rigid: Vec::new(),
        }
    }

    fn tiny_model(n: usize) -> GaussianModel {
        model_with_degree(n, 0)
    }

    /// Every LOD level must come back out with the *source's* SH stride.
    ///
    /// `extract_subset` infers its stride from `source.len() / n_gaussians`,
    /// so a level's `sh_coefficients` stays `n × (degree+1)² × 3`. `cmd_lod`
    /// depends on that: it hands each level to `model_from_arrays` with the
    /// source's `sh_degree`, which rejects any other stride. This pins the
    /// coupling — if the LOD generator ever re-strides, this fails here
    /// instead of at the user's first `scene lod --output-dir`.
    #[test]
    fn every_lod_level_keeps_the_sources_sh_stride() {
        let model = model_with_degree(8, 1);
        let flat = FlatScene::from_model(&model).expect("decompose");
        let sh_coeffs = flat.sh_coeffs();
        let config = LodConfig {
            n_levels: 3,
            reduction_ratios: vec![1.0, 0.5, 0.25],
            strategy: LodStrategy::TopOpacity,
            sort_by_opacity: true,
        };
        let chain = generate_lod_chain(
            &flat.positions,
            &flat.rotations,
            &flat.log_scales,
            &flat.opacity_logits,
            &sh_coeffs,
            &config,
        )
        .expect("chain");
        assert_eq!(chain.levels.len(), 3);

        let stride = sh_total_for_degree(flat.sh_degree);
        assert_eq!(stride, 12, "degree 1 stores 12 SH floats per Gaussian");
        for level in &chain.levels {
            assert_eq!(
                level.sh_coefficients.len(),
                level.n_gaussians * stride,
                "level {} changed the SH stride",
                level.level
            );
            let rebuilt = model_from_arrays(SceneArrays {
                positions: level.positions.clone(),
                rotations: level.rotations.clone(),
                log_scales: level.scales.clone(),
                opacity_logits: level.opacities.clone(),
                sh_coeffs: level.sh_coefficients.clone(),
                sh_degree: flat.sh_degree,
            })
            .expect("every LOD level must rebuild into a model");
            assert_eq!(rebuilt.len(), level.n_gaussians);
        }
    }

    #[test]
    fn keep_policy_maps_onto_the_library_enum() {
        assert!(matches!(
            DedupKeepPolicy::from(KeepPolicy::LargestScale),
            DedupKeepPolicy::KeepLargestScale
        ));
        assert!(matches!(
            DedupKeepPolicy::from(KeepPolicy::Last),
            DedupKeepPolicy::KeepLast
        ));
    }

    #[test]
    fn precision_maps_onto_the_library_enum() {
        assert_eq!(
            QuantizationPrecision::from(Precision::Byte),
            QuantizationPrecision::Byte
        );
        assert_eq!(
            QuantizationPrecision::from(Precision::Full),
            QuantizationPrecision::Full
        );
    }

    #[test]
    fn parse_ratios_rejects_out_of_range_values() {
        assert_eq!(parse_ratios("1.0,0.5").ok(), Some(vec![1.0, 0.5]));
        assert!(parse_ratios("1.0,1.5").is_err());
        assert!(parse_ratios("0").is_err());
        assert!(parse_ratios("").is_err());
    }

    #[test]
    fn default_lod_ratios_are_non_ascending_and_valid() {
        for levels in 1..8usize {
            let ratios: Vec<f32> = (0..levels)
                .map(|level| 0.5_f32.powi(level as i32))
                .collect();
            let config = LodConfig {
                n_levels: levels,
                reduction_ratios: ratios,
                strategy: LodStrategy::TopOpacity,
                sort_by_opacity: true,
            };
            assert!(config.validate().is_ok(), "levels={levels}");
        }
    }

    #[test]
    fn records_round_trip_through_a_model() {
        let model = tiny_model(4);
        let records = model_to_records(&model).expect("to records");
        assert_eq!(records.len(), 4);
        let rebuilt = records_to_model(&records).expect("to model");
        assert_eq!(rebuilt.len(), model.len());
        assert_eq!(rebuilt.sh_degree, model.sh_degree);
        assert_eq!(rebuilt.sh_coeffs, model.sh_coeffs);
        assert_eq!(rebuilt.gaussians[2].position, model.gaussians[2].position);
        assert_eq!(rebuilt.gaussians[2].opacity, model.gaussians[2].opacity);
    }

    #[test]
    fn records_to_model_rejects_mixed_sh_degrees() {
        let mut records = model_to_records(&tiny_model(2)).expect("to records");
        records[1] = records[1].clone().with_sh_rest(vec![0.0; 9]);
        assert!(records_to_model(&records).is_err());
    }

    #[test]
    fn read_source_rejects_an_unknown_extension() {
        let path = std::env::temp_dir().join("oxigaf_scene_convert_input.xyz");
        assert!(read_source(&path).is_err());
    }

    #[test]
    fn convert_round_trips_csv_into_binary() {
        let dir = std::env::temp_dir();
        let input = dir.join("oxigaf_scene_convert_in.csv");
        let output = dir.join("oxigaf_scene_convert_out.bin");
        let _ = std::fs::remove_file(&output);

        let records = model_to_records(&tiny_model(3)).expect("to records");
        std::fs::write(&input, crate::format_converter::to_csv(&records)).expect("write input");

        let args = ConvertArgs {
            input: input.clone(),
            output: output.clone(),
            to: None,
            force: true,
            drop_invalid: false,
        };
        cmd_convert(args, &quiet_ctx()).expect("convert");

        let bytes = std::fs::read(&output).expect("read output");
        let round_tripped = from_binary(&bytes).expect("decode binary");
        assert_eq!(round_tripped.len(), 3);
        assert!((round_tripped[1].position[0] - 0.1).abs() < 1e-5);

        let _ = std::fs::remove_file(&input);
        let _ = std::fs::remove_file(&output);
    }

    #[test]
    fn dedup_rejects_a_non_positive_threshold() {
        let args = DedupArgs {
            input: PathBuf::from("in.ply"),
            output: None,
            force: false,
            position_threshold: 0.0,
            opacity_threshold: 0.05,
            scale_threshold: 0.1,
            color_threshold: 0.1,
            keep: KeepPolicy::HighestOpacity,
            brute_force: false,
            cell_size: 0.002,
            report_groups: false,
        };
        assert!(cmd_dedup(args, &quiet_ctx()).is_err());
    }

    #[test]
    fn compress_rejects_an_out_of_range_opacity() {
        let args = CompressArgs {
            input: PathBuf::from("in.ply"),
            output: None,
            force: false,
            position_precision: Precision::Half,
            rotation_precision: Precision::Half,
            scale_precision: Precision::Half,
            opacity_precision: Precision::Half,
            sh_dc_precision: Precision::Half,
            sh_rest_precision: Precision::Half,
            prune_opacity: 1.5,
            max_log_scale: 6.0,
            min_log_scale: -10.0,
            target_gaussians: None,
            preserve_top_fraction: 1.0,
            cluster: false,
            clusters: 256,
            cluster_iterations: 50,
        };
        assert!(cmd_compress(args, &quiet_ctx()).is_err());
    }

    #[test]
    fn lod_rejects_zero_levels() {
        let args = LodArgs {
            input: PathBuf::from("in.ply"),
            output_dir: None,
            prefix: "lod".to_string(),
            force: false,
            levels: 0,
            ratios: None,
            target_memory_mb: None,
            strategy: LodPick::TopOpacity,
            unsorted: false,
            select_distance: None,
        };
        assert!(cmd_lod(args, &quiet_ctx()).is_err());
    }
}
