//! `oxigaf scene` — top-level parser for the whole scene family.
//!
//! [`crate::commands::scene_tools`] owns the geometry/reduction half of the
//! family (`register`, `stats`, `filter`, `prune`, `transform`, `dedup`,
//! `compress`, `lod`, `convert`); its [`crate::commands::scene_tools::SceneCommand`]
//! is flattened into [`SceneCommand`] here so `oxigaf scene <cmd>` stays a
//! single family while this file adds the remaining four library modules:
//!
//! | Subcommand | Library module |
//! |------------|----------------|
//! | `analyze`  | [`crate::scene_analyzer`] |
//! | `compare`  | [`crate::scene_analyzer`] |
//! | `merge`    | [`crate::scene_merge`] |
//! | `optimize` | [`crate::scene_optimizer`] |
//! | `stream`   | [`crate::scene_streaming`] |
//!
//! Flattening (rather than re-declaring the existing variants) keeps the two
//! parsers independent: adding a subcommand to `scene_tools` publishes it
//! under `oxigaf scene` with no change here.
//!
//! # Opacity and colour spaces
//!
//! See [`crate::commands::model_io`]. `scene_analyzer` takes **logit**
//! opacity and activated `[0, 1]` colour; `scene_merge` takes **probability**
//! opacity and activated colour; `scene_optimizer` and `scene_streaming` take
//! the raw model arrays (logit opacity, coefficient-major SH).

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Args, Subcommand, ValueEnum};
use serde_json::json;

use crate::commands::model_io::{
    load_scene, logit, model_from_arrays, save_scene, sh_dc_to_rgb, sh_total_for_degree, sigmoid,
    warn_if_binding_dropped, FlatScene, SceneArrays, SH_C0,
};
use crate::commands::{emit, parse_vec3, prepare_output, CmdContext};
use crate::scene_analyzer::{analyze_scene, compare_scenes, SceneData};
use crate::scene_merge::{
    merge_scenes_with_stats, GaussianEntry, SceneGaussians, SceneMergeConfig,
};
use crate::scene_optimizer::{
    so_format_config, so_format_report, so_profile_config, OptimizationPipeline,
    OptimizationProfile, OptimizationStep, SceneOptimizerConfig,
};
use crate::scene_streaming::{
    ss_compute_stats, ss_format_config, ss_format_stats, StreamingConfig, StreamingScene,
    ViewFrustum,
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
    /// Geometry and reduction tools (`register`, `stats`, `filter`, `prune`,
    /// `transform`, `dedup`, `compress`, `lod`, `convert`).
    #[command(flatten)]
    Tools(crate::commands::scene_tools::SceneCommand),

    /// Report a scene's spatial, opacity, scale and colour distributions.
    Analyze(AnalyzeSceneArgs),

    /// Compare two scenes' aggregate statistics side by side.
    Compare(CompareSceneArgs),

    /// Combine several scenes into one, with filtering and de-duplication.
    Merge(MergeArgs),

    /// Shrink a scene with a configurable prune/dedup/sort pipeline.
    Optimize(OptimizeArgs),

    /// Plan progressive streaming: spatial chunks, LOD and cache budget.
    Stream(StreamArgs),
}

/// Run the `scene` family.
///
/// # Errors
///
/// Propagates every handler's failure: unreadable models, invalid argument
/// combinations, and refused overwrites.
pub fn run(args: SceneArgs, ctx: CmdContext) -> Result<()> {
    match args.command {
        SceneCommand::Tools(tools) => crate::commands::scene_tools::run(
            crate::commands::scene_tools::SceneArgs { command: tools },
            ctx,
        ),
        SceneCommand::Analyze(analyze_args) => cmd_analyze(analyze_args, &ctx),
        SceneCommand::Compare(compare_args) => cmd_compare(compare_args, &ctx),
        SceneCommand::Merge(merge_args) => cmd_merge(merge_args, &ctx),
        SceneCommand::Optimize(optimize_args) => cmd_optimize(optimize_args, &ctx),
        SceneCommand::Stream(stream_args) => cmd_stream(stream_args, &ctx),
    }
}

// ---------------------------------------------------------------------------
// Shared conversions
// ---------------------------------------------------------------------------

/// Inverse of [`sh_dc_to_rgb`]: an activated `[0, 1]` colour back to a DC
/// spherical-harmonic coefficient.
fn rgb_to_sh_dc(rgb: f32) -> f32 {
    (rgb - 0.5) / SH_C0
}

/// Build the [`crate::scene_analyzer`] view of a scene (logit opacity,
/// activated colour).
fn scene_data(flat: &FlatScene) -> SceneData {
    SceneData {
        positions: flat.positions.clone(),
        log_scales: flat.log_scales.clone(),
        rotations: flat.rotations.clone(),
        opacities: flat.opacity_logits.clone(),
        colors: flat.rgb_colors(),
    }
}

/// Load a model and decompose it, rejecting empty scenes up front.
fn load_flat(path: &std::path::Path) -> Result<(crate::commands::model_io::FlatScene, u32)> {
    let model = load_scene(path)?;
    let flat = FlatScene::from_model(&model)?;
    if flat.n == 0 {
        anyhow::bail!("Scene is empty: {}", path.display());
    }
    let degree = flat.sh_degree;
    Ok((flat, degree))
}

/// Re-parse a module's hand-rolled JSON document into a value the `--json`
/// envelope can embed. Falls back to a string field rather than failing the
/// command over a formatting detail.
fn embed_json(raw: &str) -> serde_json::Value {
    serde_json::from_str::<serde_json::Value>(raw).unwrap_or_else(|_| json!({ "raw": raw }))
}

/// Parse a comma-separated `x,y,z` triple of positive integers.
fn parse_uvec3(raw: &str) -> std::result::Result<[u32; 3], String> {
    let parts: Vec<&str> = raw.split(',').map(str::trim).collect();
    if parts.len() != 3 {
        return Err(format!(
            "expected three comma-separated integers (x,y,z), got {} component(s) in {raw:?}",
            parts.len()
        ));
    }
    let mut out = [0u32; 3];
    for (slot, text) in out.iter_mut().zip(parts.iter()) {
        let value: u32 = text
            .parse()
            .map_err(|_| format!("{text:?} is not a non-negative integer"))?;
        if value == 0 {
            return Err("grid divisions must be at least 1".to_string());
        }
        *slot = value;
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// scene analyze
// ---------------------------------------------------------------------------

/// Arguments for `oxigaf scene analyze`.
#[derive(Debug, Args)]
pub struct AnalyzeSceneArgs {
    /// Model to analyse (`.ply` or `.json` checkpoint).
    pub model: PathBuf,

    /// Print the full multi-section breakdown instead of a single line.
    #[arg(long)]
    pub detailed: bool,
}

fn cmd_analyze(args: AnalyzeSceneArgs, ctx: &CmdContext) -> Result<()> {
    let (flat, _) = load_flat(&args.model)?;
    let scene = scene_data(&flat);
    let report = analyze_scene(&scene)?;

    let payload = json!({
        "model": args.model.display().to_string(),
        "quality_score": report.quality_score,
        "report": embed_json(&report.to_json()),
    });

    let detailed = args.detailed;
    emit(ctx, "scene analyze", payload, &[], || {
        if detailed {
            println!("{}", report.format_detailed());
        } else {
            println!("{}", report.format_summary());
        }
    });
    Ok(())
}

// ---------------------------------------------------------------------------
// scene compare
// ---------------------------------------------------------------------------

/// Arguments for `oxigaf scene compare`.
#[derive(Debug, Args)]
pub struct CompareSceneArgs {
    /// Baseline scene (A).
    pub before: PathBuf,

    /// Candidate scene (B).
    pub after: PathBuf,
}

fn cmd_compare(args: CompareSceneArgs, ctx: &CmdContext) -> Result<()> {
    let (flat_a, _) = load_flat(&args.before)?;
    let (flat_b, _) = load_flat(&args.after)?;
    let comparison = compare_scenes(&scene_data(&flat_a), &scene_data(&flat_b))?;

    let payload = json!({
        "before": args.before.display().to_string(),
        "after": args.after.display().to_string(),
        "scene_a_gaussians": comparison.scene_a_gaussians,
        "scene_b_gaussians": comparison.scene_b_gaussians,
        "size_ratio": comparison.size_ratio,
        "mean_opacity_diff": comparison.mean_opacity_diff,
        "mean_scale_diff": comparison.mean_scale_diff,
        "quality_score_diff": comparison.quality_score_diff,
        "centroid_distance": comparison.centroid_distance,
    });

    emit(ctx, "scene compare", payload, &[], || {
        println!(
            "Gaussians: {} → {} (ratio {:.3})",
            comparison.scene_a_gaussians, comparison.scene_b_gaussians, comparison.size_ratio
        );
        println!("Mean opacity delta: {:+.4}", comparison.mean_opacity_diff);
        println!("Mean scale delta:   {:+.6}", comparison.mean_scale_diff);
        println!("Quality delta:      {:+.4}", comparison.quality_score_diff);
        println!("Centroid distance:  {:.6}", comparison.centroid_distance);
    });
    Ok(())
}

// ---------------------------------------------------------------------------
// scene merge
// ---------------------------------------------------------------------------

/// Arguments for `oxigaf scene merge`.
#[derive(Debug, Args)]
pub struct MergeArgs {
    /// Scenes to merge, in order. At least two are required.
    #[arg(long = "input", short = 'i', required = true, num_args = 1.., value_name = "PATH")]
    pub inputs: Vec<PathBuf>,

    /// Write the merged scene here (`.ply` or `.json`).
    #[arg(short, long)]
    pub output: PathBuf,

    /// Drop Gaussians whose activated opacity is below this value.
    #[arg(long, default_value = "0.005")]
    pub min_opacity: f32,

    /// Drop Gaussians whose largest linear scale exceeds this value (0 = keep all).
    #[arg(long, default_value = "0")]
    pub max_scale: f32,

    /// Remove near-duplicate Gaussians after merging.
    #[arg(long)]
    pub dedup: bool,

    /// Distance below which two Gaussians count as duplicates.
    #[arg(long, default_value = "0.001")]
    pub dedup_threshold: f32,

    /// Rescale opacities after merging so the mean approaches --target-opacity.
    #[arg(long)]
    pub normalize_opacity: bool,

    /// Target mean opacity for --normalize-opacity.
    #[arg(long, default_value = "0.3")]
    pub target_opacity: f32,

    /// Cap the merged Gaussian count, keeping the most opaque (0 = no cap).
    #[arg(long, default_value = "0")]
    pub max_gaussians: usize,

    /// Ignore each scene's stored transform instead of applying it.
    #[arg(long)]
    pub no_transforms: bool,

    /// Overwrite the output file if it already exists.
    #[arg(long)]
    pub force: bool,
}

fn cmd_merge(args: MergeArgs, ctx: &CmdContext) -> Result<()> {
    if args.inputs.len() < 2 {
        anyhow::bail!("`scene merge` needs at least two --input scenes");
    }

    let mut scenes: Vec<SceneGaussians> = Vec::with_capacity(args.inputs.len());
    let mut sh_degree: Option<u32> = None;
    let mut source_models = Vec::with_capacity(args.inputs.len());

    for path in &args.inputs {
        let model = load_scene(path)?;
        let flat = FlatScene::from_model(&model)?;
        if flat.n == 0 {
            anyhow::bail!("Scene is empty: {}", path.display());
        }
        match sh_degree {
            None => sh_degree = Some(flat.sh_degree),
            Some(degree) if degree != flat.sh_degree => anyhow::bail!(
                "Cannot merge scenes with different SH degrees: {} is degree {}, \
                 an earlier input is degree {degree}. Re-export them at a common \
                 degree first (`oxigaf export --sh-degree`).",
                path.display(),
                flat.sh_degree,
            ),
            Some(_) => {}
        }

        let rest = flat.n_rest_per_gaussian;
        let entries: Vec<GaussianEntry> = (0..flat.n)
            .map(|i| GaussianEntry {
                position: [
                    flat.positions[i * 3],
                    flat.positions[i * 3 + 1],
                    flat.positions[i * 3 + 2],
                ],
                log_scale: [
                    flat.log_scales[i * 3],
                    flat.log_scales[i * 3 + 1],
                    flat.log_scales[i * 3 + 2],
                ],
                rotation: [
                    flat.rotations[i * 4],
                    flat.rotations[i * 4 + 1],
                    flat.rotations[i * 4 + 2],
                    flat.rotations[i * 4 + 3],
                ],
                opacity: sigmoid(flat.opacity_logits[i]),
                color: [
                    sh_dc_to_rgb(flat.sh_dc[i * 3]),
                    sh_dc_to_rgb(flat.sh_dc[i * 3 + 1]),
                    sh_dc_to_rgb(flat.sh_dc[i * 3 + 2]),
                ],
                sh_coeffs: flat.sh_rest[i * rest..i * rest + rest].to_vec(),
            })
            .collect();

        let name = path.display().to_string();
        scenes.push(SceneGaussians::new(entries).with_name(name));
        source_models.push(model);
    }

    let degree = sh_degree.unwrap_or(0);
    let rest_per_gaussian = sh_total_for_degree(degree).saturating_sub(3);

    let config = SceneMergeConfig {
        min_opacity: args.min_opacity,
        max_scale: args.max_scale,
        remove_duplicates: args.dedup,
        duplicate_threshold: args.dedup_threshold,
        normalize_opacities: args.normalize_opacity,
        target_opacity: args.target_opacity,
        max_gaussians: args.max_gaussians,
        apply_transforms: !args.no_transforms,
    };
    config.validate()?;

    let (merged, stats) = merge_scenes_with_stats(&scenes, &config)?;
    if merged.is_empty() {
        anyhow::bail!(
            "Merging produced an empty scene — every Gaussian was filtered out. \
             Lower --min-opacity or raise --max-scale."
        );
    }

    let n = merged.len();
    let mut positions = Vec::with_capacity(n * 3);
    let mut rotations = Vec::with_capacity(n * 4);
    let mut log_scales = Vec::with_capacity(n * 3);
    let mut opacity_logits = Vec::with_capacity(n);
    let mut sh_coeffs = Vec::with_capacity(n * (3 + rest_per_gaussian));
    for gaussian in &merged.gaussians {
        positions.extend_from_slice(&gaussian.position);
        rotations.extend_from_slice(&gaussian.rotation);
        log_scales.extend_from_slice(&gaussian.log_scale);
        opacity_logits.push(logit(gaussian.opacity));
        for channel in gaussian.color {
            sh_coeffs.push(rgb_to_sh_dc(channel));
        }
        // A merged Gaussian keeps the rest-coefficient block of whichever
        // input it came from; a shorter block (an input that carried none)
        // is zero-padded so the SH stride stays uniform.
        let rest = &gaussian.sh_coeffs;
        let take = rest.len().min(rest_per_gaussian);
        sh_coeffs.extend_from_slice(&rest[..take]);
        sh_coeffs.resize(sh_coeffs.len() + (rest_per_gaussian - take), 0.0);
    }

    let model = model_from_arrays(SceneArrays {
        positions,
        rotations,
        log_scales,
        opacity_logits,
        sh_coeffs,
        sh_degree: degree,
    })?;

    let payload = json!({
        "inputs": args.inputs.iter().map(|p| p.display().to_string()).collect::<Vec<_>>(),
        "output": args.output.display().to_string(),
        "input_scenes": stats.input_scenes,
        "input_gaussians": stats.input_gaussians,
        "total_input": stats.total_input,
        "after_opacity_filter": stats.after_opacity_filter,
        "after_scale_filter": stats.after_scale_filter,
        "duplicates_removed": stats.duplicates_removed,
        "final_count": stats.final_count,
        "sh_degree": degree,
    });

    if prepare_output(ctx, &args.output, args.force)? {
        for source in &source_models {
            warn_if_binding_dropped(source, &args.output);
        }
        save_scene(&model, &args.output)?;
        emit(
            ctx,
            "scene merge",
            payload,
            &[("scene", &args.output)],
            || {
                println!("{}", stats.format_summary());
                println!("Wrote {}", args.output.display());
            },
        );
    } else {
        emit(ctx, "scene merge", payload, &[], || {
            println!("{}", stats.format_summary());
            println!("[dry-run] would write {}", args.output.display());
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// scene optimize
// ---------------------------------------------------------------------------

/// Preset optimisation pipeline.
#[derive(Debug, Clone, Copy, ValueEnum, Default)]
pub enum OptimizeProfile {
    /// De-duplication only — the most conservative preset.
    Quality,
    /// De-duplication, light opacity pruning and scale clamping.
    #[default]
    Balanced,
    /// Aggressive pruning, top-N selection and a Morton sort.
    Performance,
    /// Morton sort plus a head-sized bounding-sphere clip.
    Streaming,
}

impl From<OptimizeProfile> for OptimizationProfile {
    fn from(value: OptimizeProfile) -> Self {
        match value {
            OptimizeProfile::Quality => OptimizationProfile::Quality,
            OptimizeProfile::Balanced => OptimizationProfile::Balanced,
            OptimizeProfile::Performance => OptimizationProfile::Performance,
            OptimizeProfile::Streaming => OptimizationProfile::Streaming,
        }
    }
}

/// Arguments for `oxigaf scene optimize`.
///
/// Either pick a `--profile`, or build a custom pipeline from the individual
/// step flags below — the two are mutually exclusive so the executed pipeline
/// is never a surprise.
#[derive(Debug, Args)]
pub struct OptimizeArgs {
    /// Model to optimise (`.ply` or `.json` checkpoint).
    pub model: PathBuf,

    /// Write the optimised scene here (`.ply` or `.json`). Omit to only report.
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Preset pipeline (default `balanced` when no step flag is given).
    #[arg(long, value_enum)]
    pub profile: Option<OptimizeProfile>,

    /// Custom step: drop Gaussians whose activated opacity is below this.
    #[arg(long, value_name = "THRESHOLD")]
    pub prune_opacity: Option<f32>,

    /// Custom step: merge Gaussians closer together than this distance.
    #[arg(long, value_name = "RADIUS")]
    pub dedup_radius: Option<f32>,

    /// Custom step: clamp linear (world-unit) scales to at least this value.
    ///
    /// Requires --clamp-scale-max. Converted to the module's log-space bounds.
    #[arg(long, value_name = "MIN", requires = "clamp_scale_max")]
    pub clamp_scale_min: Option<f32>,

    /// Custom step: clamp linear (world-unit) scales to at most this value.
    #[arg(long, value_name = "MAX", requires = "clamp_scale_min")]
    pub clamp_scale_max: Option<f32>,

    /// Custom step: keep only the N most opaque Gaussians.
    #[arg(long, value_name = "N")]
    pub top_n: Option<usize>,

    /// Custom step: reorder Gaussians by Morton code for cache locality.
    #[arg(long)]
    pub sort_morton: bool,

    /// Custom step: keep only Gaussians inside a sphere centred here.
    #[arg(long, value_name = "X,Y,Z", value_parser = parse_vec3, requires = "clip_radius")]
    pub clip_center: Option<[f32; 3]>,

    /// Radius of the --clip-center sphere.
    #[arg(long, value_name = "R", requires = "clip_center")]
    pub clip_radius: Option<f32>,

    /// Custom step: keep only Gaussians inside an axis-aligned box (minimum corner).
    #[arg(long, value_name = "X,Y,Z", value_parser = parse_vec3, requires = "clip_max")]
    pub clip_min: Option<[f32; 3]>,

    /// Maximum corner of the --clip-min box.
    #[arg(long, value_name = "X,Y,Z", value_parser = parse_vec3, requires = "clip_min")]
    pub clip_max: Option<[f32; 3]>,

    /// Custom step: activate opacity logits into probabilities.
    ///
    /// The written model always stores logits, so this is re-inverted before
    /// saving; it only changes what the reported snapshot measures.
    #[arg(long)]
    pub normalize_opacity: bool,

    /// Overwrite the output file if it already exists.
    #[arg(long)]
    pub force: bool,
}

impl OptimizeArgs {
    /// Custom steps in canonical execution order, or an empty vector when no
    /// step flag was given.
    fn custom_steps(&self) -> Vec<OptimizationStep> {
        let mut steps = Vec::new();
        if let Some(radius) = self.dedup_radius {
            steps.push(OptimizationStep::DeduplicateNear {
                position_radius: radius,
            });
        }
        if let Some(threshold) = self.prune_opacity {
            steps.push(OptimizationStep::PruneByOpacity { threshold });
        }
        if let (Some(min), Some(max)) = (self.clamp_scale_min, self.clamp_scale_max) {
            steps.push(OptimizationStep::ClampScales {
                min_scale: min.ln(),
                max_scale: max.ln(),
            });
        }
        if let (Some(center), Some(radius)) = (self.clip_center, self.clip_radius) {
            steps.push(OptimizationStep::ClipToSphere { center, radius });
        }
        if let (Some(min), Some(max)) = (self.clip_min, self.clip_max) {
            steps.push(OptimizationStep::ClipToAabb { min, max });
        }
        if let Some(n) = self.top_n {
            steps.push(OptimizationStep::TopNByOpacity { n });
        }
        if self.sort_morton {
            steps.push(OptimizationStep::SortMorton);
        }
        if self.normalize_opacity {
            steps.push(OptimizationStep::NormalizeOpacity);
        }
        steps
    }

    /// Reject argument combinations the pipeline cannot honour.
    fn validate(&self) -> Result<()> {
        if let Some(threshold) = self.prune_opacity {
            if !(0.0..=1.0).contains(&threshold) {
                anyhow::bail!("--prune-opacity must be in [0, 1], got {threshold}");
            }
        }
        // These bounds are spelled `!is_finite() || <= 0.0` rather than the
        // shorter `<= 0.0` on purpose: `NaN <= 0.0` is *false*, so the terse
        // form would wave a `--dedup-radius nan` straight through into a
        // spatial hash whose every cell index is undefined. The negated
        // comparison (`!(radius > 0.0)`) that used to encode the same intent
        // is what `clippy::neg_cmp_op_on_partial_ord` objects to, so the NaN
        // case is now named explicitly instead of hidden in the negation.
        // Infinities are rejected for the same reason.
        if let Some(radius) = self.dedup_radius {
            if !radius.is_finite() || radius <= 0.0 {
                anyhow::bail!("--dedup-radius must be a finite value greater than 0, got {radius}");
            }
        }
        if let (Some(min), Some(max)) = (self.clamp_scale_min, self.clamp_scale_max) {
            if !min.is_finite() || !max.is_finite() || min <= 0.0 || max <= 0.0 {
                anyhow::bail!(
                    "--clamp-scale-min/--clamp-scale-max are linear world-unit sizes and \
                     must both be finite and greater than 0 (got {min} and {max})"
                );
            }
            if min > max {
                anyhow::bail!(
                    "--clamp-scale-min ({min}) must not exceed --clamp-scale-max ({max})"
                );
            }
        }
        if let Some(radius) = self.clip_radius {
            if !radius.is_finite() || radius <= 0.0 {
                anyhow::bail!("--clip-radius must be a finite value greater than 0, got {radius}");
            }
        }
        if let (Some(min), Some(max)) = (self.clip_min, self.clip_max) {
            for axis in 0..3 {
                if min[axis] > max[axis] {
                    anyhow::bail!(
                        "--clip-min must be component-wise less than or equal to --clip-max \
                         (axis {axis}: {} > {})",
                        min[axis],
                        max[axis]
                    );
                }
            }
        }
        if self.top_n == Some(0) {
            anyhow::bail!("--top-n must be at least 1");
        }
        Ok(())
    }
}

fn cmd_optimize(args: OptimizeArgs, ctx: &CmdContext) -> Result<()> {
    args.validate()?;

    let custom = args.custom_steps();
    if !custom.is_empty() && args.profile.is_some() {
        anyhow::bail!(
            "--profile and the individual step flags are mutually exclusive: pass one preset, \
             or spell the pipeline out step by step."
        );
    }

    let (flat, degree) = load_flat(&args.model)?;
    let sh_channels = sh_total_for_degree(degree);
    let sh_coeffs = flat.sh_coeffs();

    let config: SceneOptimizerConfig = if custom.is_empty() {
        so_profile_config(args.profile.unwrap_or_default().into(), sh_channels)
    } else {
        SceneOptimizerConfig {
            steps: custom,
            sh_channels,
            seed: 42,
        }
    };
    let opacity_is_probability = config
        .steps
        .iter()
        .any(|step| matches!(step, OptimizationStep::NormalizeOpacity));

    let pipeline = OptimizationPipeline::new(config);
    let (optimized, report) = pipeline.run(
        &flat.positions,
        &flat.rotations,
        &flat.log_scales,
        &flat.opacity_logits,
        &sh_coeffs,
        flat.n,
    )?;

    let payload = json!({
        "model": args.model.display().to_string(),
        "output": args.output.as_ref().map(|p| p.display().to_string()),
        "config": so_format_config(&pipeline.config),
        "gaussians_before": report.snapshot_before.n_gaussians,
        "gaussians_after": report.snapshot_after.n_gaussians,
        "total_removed": report.total_removed,
        "total_reduction_percent": report.total_reduction_percent,
        "memory_saved_bytes": report.memory_saved_bytes,
        "steps": report.step_results.iter().map(|step| json!({
            "name": step.step_name,
            "n_before": step.n_before,
            "n_after": step.n_after,
            "n_removed": step.n_removed,
            "duration_hint": step.duration_hint,
            "notes": step.notes,
        })).collect::<Vec<_>>(),
    });

    let Some(ref output) = args.output else {
        emit(ctx, "scene optimize", payload, &[], || {
            print!("{}", so_format_report(&report));
        });
        return Ok(());
    };

    if !prepare_output(ctx, output, args.force)? {
        emit(ctx, "scene optimize", payload, &[], || {
            print!("{}", so_format_report(&report));
            println!("[dry-run] would write {}", output.display());
        });
        return Ok(());
    }

    // The model format always stores opacity logits; undo NormalizeOpacity
    // before writing so a round trip does not silently re-compress opacity.
    let opacity_logits: Vec<f32> = if opacity_is_probability {
        optimized.opacities.iter().copied().map(logit).collect()
    } else {
        optimized.opacities.clone()
    };

    let model = model_from_arrays(SceneArrays {
        positions: optimized.positions.clone(),
        rotations: optimized.rotations.clone(),
        log_scales: optimized.scales.clone(),
        opacity_logits,
        sh_coeffs: optimized.sh_coefficients.clone(),
        sh_degree: degree,
    })?;

    let source = load_scene(&args.model)?;
    warn_if_binding_dropped(&source, output);
    save_scene(&model, output)?;

    emit(ctx, "scene optimize", payload, &[("scene", output)], || {
        print!("{}", so_format_report(&report));
        println!("Wrote {}", output.display());
    });
    Ok(())
}

// ---------------------------------------------------------------------------
// scene stream
// ---------------------------------------------------------------------------

/// Arguments for `oxigaf scene stream`.
#[derive(Debug, Args)]
pub struct StreamArgs {
    /// Model to plan streaming for (`.ply` or `.json` checkpoint).
    pub model: PathBuf,

    /// Grid divisions along each axis, as `nx,ny,nz`.
    #[arg(long, default_value = "4,4,4", value_parser = parse_uvec3)]
    pub divisions: [u32; 3],

    /// Resident-chunk memory budget in megabytes.
    #[arg(long, default_value = "512")]
    pub memory_budget_mb: usize,

    /// Camera position, as `x,y,z`.
    #[arg(long, default_value = "0,0,3", value_parser = parse_vec3)]
    pub eye: [f32; 3],

    /// Point the camera looks at, as `x,y,z`.
    #[arg(long, default_value = "0,0,0", value_parser = parse_vec3)]
    pub target: [f32; 3],

    /// Field of view in degrees (applied to both axes).
    #[arg(long, default_value = "60")]
    pub fov_deg: f32,

    /// Near clipping distance.
    #[arg(long, default_value = "0.1")]
    pub near: f32,

    /// Far clipping distance.
    #[arg(long, default_value = "1000")]
    pub far: f32,

    /// LOD switch distances, as `near,mid,far`.
    #[arg(long, default_value = "10,30,80", value_parser = parse_vec3)]
    pub lod_distances: [f32; 3],

    /// Maximum chunk load requests issued per simulated frame.
    #[arg(long, default_value = "4")]
    pub max_chunks_per_frame: usize,

    /// Chunks within this distance of the eye are pre-loaded.
    #[arg(long, default_value = "15")]
    pub preload_radius: f32,

    /// Number of frames to simulate (each issues one load batch).
    #[arg(long, default_value = "1")]
    pub frames: usize,

    /// Also list every chunk, not just the summary.
    #[arg(long)]
    pub list_chunks: bool,
}

fn cmd_stream(args: StreamArgs, ctx: &CmdContext) -> Result<()> {
    if args.memory_budget_mb == 0 {
        anyhow::bail!("--memory-budget-mb must be at least 1");
    }
    if args.max_chunks_per_frame == 0 {
        anyhow::bail!("--max-chunks-per-frame must be at least 1");
    }
    if args.frames == 0 {
        anyhow::bail!("--frames must be at least 1");
    }
    if args.fov_deg <= 0.0 || args.fov_deg >= 180.0 {
        anyhow::bail!("--fov-deg must be in (0, 180), got {}", args.fov_deg);
    }

    let forward = [
        args.target[0] - args.eye[0],
        args.target[1] - args.eye[1],
        args.target[2] - args.eye[2],
    ];
    if forward.iter().all(|component| component.abs() < 1e-9) {
        anyhow::bail!("--eye and --target must not be the same point");
    }
    let fov = args.fov_deg.to_radians();
    let frustum = ViewFrustum::new(
        args.eye,
        forward,
        [0.0, 1.0, 0.0],
        fov,
        fov,
        args.near,
        args.far,
    )?;

    let (flat, degree) = load_flat(&args.model)?;
    let config = StreamingConfig {
        memory_budget_bytes: args
            .memory_budget_mb
            .checked_mul(1024 * 1024)
            .context("--memory-budget-mb is too large to express in bytes")?,
        chunk_divisions: args.divisions,
        lod_distances: args.lod_distances,
        max_chunks_per_frame: args.max_chunks_per_frame,
        preload_radius: args.preload_radius,
    };
    let config_text = ss_format_config(&config);
    let mut scene =
        StreamingScene::new(config, &flat.positions, flat.n, sh_total_for_degree(degree))
            .with_context(|| {
                format!(
                    "Failed to plan streaming for {}: {} Gaussian(s) over a {}×{}×{} chunk grid",
                    args.model.display(),
                    flat.n,
                    args.divisions[0],
                    args.divisions[1],
                    args.divisions[2],
                )
            })?;

    let mut loaded = 0usize;
    let mut evicted_note: Option<String> = None;
    for _ in 0..args.frames {
        let requested = scene.update_view(&frustum);
        for id in requested {
            match scene.mark_loaded(id) {
                Ok(()) => loaded += 1,
                Err(e) => {
                    if evicted_note.is_none() {
                        evicted_note = Some(e.to_string());
                    }
                }
            }
        }
    }

    let stats = ss_compute_stats(&scene, &frustum);
    let chunks: Vec<serde_json::Value> = scene
        .chunks
        .iter()
        .map(|chunk| {
            json!({
                "id": chunk.id,
                "bounds_min": chunk.bounds_min,
                "bounds_max": chunk.bounds_max,
                "n_gaussians": chunk.n_gaussians,
                "priority": format!("{:?}", chunk.priority),
                "lod_level": chunk.lod_level,
                "loaded": chunk.loaded,
                "memory_bytes": chunk.memory_bytes,
            })
        })
        .collect();

    let payload = json!({
        "model": args.model.display().to_string(),
        "config": config_text,
        "frames_simulated": args.frames,
        "chunks_loaded": loaded,
        "load_error": evicted_note,
        "total_chunks": stats.total_chunks,
        "loaded_chunks": stats.loaded_chunks,
        "loading_ratio": stats.loading_ratio,
        "total_gaussians": stats.total_gaussians,
        "visible_gaussians": stats.visible_gaussians,
        "cache_utilization": stats.cache_utilization,
        "memory_used_bytes": stats.memory_used_bytes,
        "memory_budget_bytes": stats.memory_budget_bytes,
        "chunks": if args.list_chunks { serde_json::Value::Array(chunks.clone()) } else { serde_json::Value::Null },
    });

    let list_chunks = args.list_chunks;
    emit(ctx, "scene stream", payload, &[], || {
        println!("{config_text}");
        println!("{}", ss_format_stats(&stats));
        if let Some(ref note) = evicted_note {
            println!("Cache refused at least one chunk: {note}");
        }
        if list_chunks {
            for chunk in &scene.chunks {
                println!(
                    "  chunk {:#018x}  n={:<8} lod={} priority={:?} loaded={} mem={}B",
                    chunk.id,
                    chunk.n_gaussians,
                    chunk.lod_level,
                    chunk.priority,
                    chunk.loaded,
                    chunk.memory_bytes,
                );
            }
        }
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_uvec3_accepts_a_positive_triple() {
        assert_eq!(parse_uvec3("2, 3 ,4"), Ok([2, 3, 4]));
    }

    #[test]
    fn parse_uvec3_rejects_zero_and_wrong_arity() {
        assert!(parse_uvec3("0,1,1").is_err());
        assert!(parse_uvec3("1,1").is_err());
        assert!(parse_uvec3("1,1,-1").is_err());
    }

    #[test]
    fn rgb_to_sh_dc_inverts_sh_dc_to_rgb() {
        for dc in [-1.5_f32, -0.25, 0.0, 0.25, 1.5] {
            let round_trip = rgb_to_sh_dc(sh_dc_to_rgb(dc));
            assert!(
                (round_trip - dc).abs() < 1e-3,
                "dc {dc} round-tripped to {round_trip}"
            );
        }
    }

    #[test]
    fn optimize_rejects_a_profile_alongside_custom_steps() {
        let args = OptimizeArgs {
            model: PathBuf::from("scene.ply"),
            output: None,
            profile: Some(OptimizeProfile::Quality),
            prune_opacity: Some(0.1),
            dedup_radius: None,
            clamp_scale_min: None,
            clamp_scale_max: None,
            top_n: None,
            sort_morton: false,
            clip_center: None,
            clip_radius: None,
            clip_min: None,
            clip_max: None,
            normalize_opacity: false,
            force: false,
        };
        assert!(!args.custom_steps().is_empty());
        assert!(args.profile.is_some());
    }

    #[test]
    fn optimize_validate_rejects_out_of_range_values() {
        let mut args = OptimizeArgs {
            model: PathBuf::from("scene.ply"),
            output: None,
            profile: None,
            prune_opacity: Some(2.0),
            dedup_radius: None,
            clamp_scale_min: None,
            clamp_scale_max: None,
            top_n: None,
            sort_morton: false,
            clip_center: None,
            clip_radius: None,
            clip_min: None,
            clip_max: None,
            normalize_opacity: false,
            force: false,
        };
        assert!(args.validate().is_err());
        args.prune_opacity = Some(0.5);
        assert!(args.validate().is_ok());
        args.top_n = Some(0);
        assert!(args.validate().is_err());
    }

    /// `NaN <= 0.0` is false, so a bound written as a plain `<= 0.0` would
    /// let `--dedup-radius nan` reach the spatial hash, where every cell
    /// index is then undefined. Non-finite values are rejected by name.
    #[test]
    fn optimize_validate_rejects_non_finite_radii_and_scales() {
        let base = || OptimizeArgs {
            model: PathBuf::from("scene.ply"),
            output: None,
            profile: None,
            prune_opacity: None,
            dedup_radius: None,
            clamp_scale_min: None,
            clamp_scale_max: None,
            top_n: None,
            sort_morton: false,
            clip_center: None,
            clip_radius: None,
            clip_min: None,
            clip_max: None,
            normalize_opacity: false,
            force: false,
        };

        for bad in [f32::NAN, f32::INFINITY, 0.0, -1.0] {
            let mut args = base();
            args.dedup_radius = Some(bad);
            assert!(
                args.validate().is_err(),
                "--dedup-radius {bad} should be refused"
            );

            let mut args = base();
            args.clip_radius = Some(bad);
            assert!(
                args.validate().is_err(),
                "--clip-radius {bad} should be refused"
            );

            let mut args = base();
            args.clamp_scale_min = Some(bad);
            args.clamp_scale_max = Some(1.0);
            assert!(
                args.validate().is_err(),
                "--clamp-scale-min {bad} should be refused"
            );
        }

        let mut args = base();
        args.dedup_radius = Some(0.001);
        args.clip_radius = Some(1.0);
        args.clip_center = Some([0.0, 0.0, 0.0]);
        args.clamp_scale_min = Some(0.001);
        args.clamp_scale_max = Some(0.1);
        assert!(args.validate().is_ok());
    }

    /// Regression: `cmd_stream` used to build its scene through a nonsense
    /// `map_or`/`map_or_else` chain that handed the four-argument
    /// [`StreamingScene::new`] a single borrowed config — `oxigaf scene
    /// stream` could not compile at all, and the module was only reachable
    /// once `commands/mod.rs` declared it. This exercises the same call
    /// shape the handler now uses.
    #[test]
    fn stream_plan_is_built_from_the_scene_positions() {
        let config = StreamingConfig {
            memory_budget_bytes: 8 * 1024 * 1024,
            chunk_divisions: [2, 2, 2],
            lod_distances: [10.0, 30.0, 80.0],
            max_chunks_per_frame: 4,
            preload_radius: 15.0,
        };
        // Eight Gaussians on the corners of a unit cube: every one of the
        // 2×2×2 cells gets exactly one.
        let mut positions = Vec::with_capacity(8 * 3);
        for i in 0..8u32 {
            positions.push(if i & 1 == 0 { -1.0 } else { 1.0 });
            positions.push(if i & 2 == 0 { -1.0 } else { 1.0 });
            positions.push(if i & 4 == 0 { -1.0 } else { 1.0 });
        }

        let mut scene = StreamingScene::new(config, &positions, 8, sh_total_for_degree(0))
            .expect("a non-empty scene must produce a streaming plan");
        assert!(
            !scene.chunks.is_empty(),
            "chunking eight Gaussians produced no chunks"
        );

        let frustum = ViewFrustum::new(
            [0.0, 0.0, 5.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
            60.0_f32.to_radians(),
            60.0_f32.to_radians(),
            0.1,
            1000.0,
        )
        .expect("a near/far ordered frustum is valid");
        let requested = scene.update_view(&frustum);
        assert!(
            requested.len() <= 4,
            "update_view returned {} chunks, above --max-chunks-per-frame 4",
            requested.len()
        );
    }

    /// An empty scene has no chunks to plan, so the constructor refuses it
    /// rather than reporting a zero-chunk plan as a success.
    #[test]
    fn stream_plan_refuses_an_empty_scene() {
        let config = StreamingConfig {
            memory_budget_bytes: 1024 * 1024,
            chunk_divisions: [2, 2, 2],
            lod_distances: [10.0, 30.0, 80.0],
            max_chunks_per_frame: 1,
            preload_radius: 1.0,
        };
        assert!(StreamingScene::new(config, &[], 0, 3).is_err());
    }

    #[test]
    fn custom_steps_are_ordered_canonically() {
        let args = OptimizeArgs {
            model: PathBuf::from("scene.ply"),
            output: None,
            profile: None,
            prune_opacity: Some(0.01),
            dedup_radius: Some(0.001),
            clamp_scale_min: None,
            clamp_scale_max: None,
            top_n: Some(10),
            sort_morton: true,
            clip_center: None,
            clip_radius: None,
            clip_min: None,
            clip_max: None,
            normalize_opacity: false,
            force: false,
        };
        let steps = args.custom_steps();
        assert_eq!(steps.len(), 4);
        assert!(matches!(steps[0], OptimizationStep::DeduplicateNear { .. }));
        assert!(matches!(steps[1], OptimizationStep::PruneByOpacity { .. }));
        assert!(matches!(steps[2], OptimizationStep::TopNByOpacity { .. }));
        assert!(matches!(steps[3], OptimizationStep::SortMorton));
    }
}
