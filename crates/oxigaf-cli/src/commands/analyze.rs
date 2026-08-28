//! `oxigaf analyze` — read-only inspection and quality analysis.
//!
//! Wires three library modules:
//!
//! * `analyze color` — [`crate::color_calibration`]: fit a colour-correction
//!   matrix against the Macbeth ColorChecker reference and report ΔE.
//! * `analyze diff`  — [`crate::diff_tool`]: field-by-field comparison of two
//!   Gaussian models.
//! * `analyze eval`  — [`crate::evaluation_suite`]: PSNR / SSIM / MS-SSIM /
//!   LPIPS-approx over a rendered-vs-ground-truth image set.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Args, Subcommand, ValueEnum};
use serde_json::json;

use crate::color_calibration::{
    cal_apply_gamma_inv_channel, cal_evaluate, cal_format_stats, cal_macbeth_patches,
    cal_solve_ccm, CalibrationMatrix, DeltaEMetric, GammaProfile, WhiteBalance,
};
use crate::commands::{emit, CmdContext};
use crate::diff_tool::{
    detect_regression, diff_models, diff_models_variable, format_field_diff,
    format_field_diff_header, format_model_diff, largest_position_changes, DiffConfig, FieldDiff,
    ModelDiff, ModelSnapshot,
};
use crate::evaluation_suite::{
    eval_compare, eval_format_comparison, eval_format_suite_result, eval_psnr_percentiles,
    EvalConfig, EvalSuiteResult, EvalTestItem,
};

/// `oxigaf analyze <command>`.
#[derive(Debug, Args)]
pub struct AnalyzeArgs {
    #[command(subcommand)]
    pub command: AnalyzeCommand,
}

/// Analysis subcommands.
#[derive(Debug, Subcommand)]
pub enum AnalyzeCommand {
    /// Fit and score a colour-correction matrix from a ColorChecker shot.
    Color(ColorArgs),

    /// Compare two Gaussian models field by field.
    Diff(DiffArgs),

    /// Score rendered images against ground truth.
    Eval(EvalArgs),
}

// ---------------------------------------------------------------------------
// analyze color
// ---------------------------------------------------------------------------

/// Perceptual colour-difference metric.
#[derive(Debug, Clone, Copy, ValueEnum, Default)]
pub enum DeltaE {
    /// CIE76 — fast Euclidean distance in Lab.
    Cie76,
    /// CIEDE2000 — the industry standard for chart calibration.
    #[default]
    Cie2000,
}

impl From<DeltaE> for DeltaEMetric {
    fn from(value: DeltaE) -> Self {
        match value {
            DeltaE::Cie76 => DeltaEMetric::Cie76,
            DeltaE::Cie2000 => DeltaEMetric::Cie2000,
        }
    }
}

/// Arguments for `oxigaf analyze color`.
#[derive(Debug, Args)]
pub struct ColorArgs {
    /// Photograph of a ColorChecker chart, cropped to the patch grid.
    #[arg(long, conflicts_with = "measured")]
    pub image: Option<PathBuf>,

    /// JSON array of measured linear-sRGB patches: `[[r,g,b], ...]`.
    #[arg(long, conflicts_with = "image")]
    pub measured: Option<PathBuf>,

    /// JSON array of reference linear-sRGB patches. Defaults to the 24
    /// standard Macbeth ColorChecker values.
    #[arg(long)]
    pub reference: Option<PathBuf>,

    /// Patch grid columns when sampling `--image`.
    #[arg(long, default_value = "6")]
    pub cols: usize,

    /// Patch grid rows when sampling `--image`.
    #[arg(long, default_value = "4")]
    pub rows: usize,

    /// Perceptual metric used for the per-patch difference.
    #[arg(long, value_enum, default_value = "cie2000")]
    pub metric: DeltaE,

    /// Colour temperature (K) used to derive white-balance gains.
    #[arg(long, conflicts_with = "gray_patch")]
    pub temperature: Option<f32>,

    /// Measured neutral patch `r,g,b` used to derive white-balance gains.
    #[arg(long, conflicts_with = "temperature", value_parser = crate::commands::parse_vec3)]
    pub gray_patch: Option<[f32; 3]>,

    /// Treat `--image` samples as already linear instead of sRGB-encoded.
    #[arg(long)]
    pub linear_input: bool,
}

fn load_patches(path: &Path) -> Result<Vec<[f32; 3]>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read patch file: {}", path.display()))?;
    let patches: Vec<[f32; 3]> = serde_json::from_str(&text).with_context(|| {
        format!(
            "Failed to parse patch file as [[r,g,b], ...]: {}",
            path.display()
        )
    })?;
    Ok(patches)
}

/// Sample the mean colour of each cell of a `rows` x `cols` chart grid.
///
/// Only the central 50% of every cell is averaged, so the patch borders and
/// the black separators between them never bias the measurement.
fn sample_chart(
    path: &Path,
    rows: usize,
    cols: usize,
    linear_input: bool,
) -> Result<Vec<[f32; 3]>> {
    let image = image::open(path)
        .with_context(|| format!("Failed to open chart image: {}", path.display()))?
        .to_rgb8();
    let (width, height) = (image.width() as usize, image.height() as usize);
    if width < cols * 2 || height < rows * 2 {
        anyhow::bail!(
            "Chart image {}x{} is too small for a {rows}x{cols} patch grid",
            width,
            height
        );
    }

    let cell_w = width / cols;
    let cell_h = height / rows;
    let mut patches = Vec::with_capacity(rows * cols);

    for row in 0..rows {
        for col in 0..cols {
            let x0 = col * cell_w + cell_w / 4;
            let x1 = col * cell_w + (3 * cell_w) / 4;
            let y0 = row * cell_h + cell_h / 4;
            let y1 = row * cell_h + (3 * cell_h) / 4;
            let mut sum = [0.0f64; 3];
            let mut count = 0u64;
            for y in y0..y1.max(y0 + 1) {
                for x in x0..x1.max(x0 + 1) {
                    let pixel = image.get_pixel(x as u32, y as u32);
                    for channel in 0..3 {
                        sum[channel] += f64::from(pixel[channel]) / 255.0;
                    }
                    count += 1;
                }
            }
            let scale = count.max(1) as f64;
            let mut rgb = [
                (sum[0] / scale) as f32,
                (sum[1] / scale) as f32,
                (sum[2] / scale) as f32,
            ];
            if !linear_input {
                // Chart photographs are sRGB-encoded; the reference patches
                // are linear, so undo the transfer function before fitting.
                for channel in rgb.iter_mut() {
                    *channel = cal_apply_gamma_inv_channel(*channel, &GammaProfile::Srgb);
                }
            }
            patches.push(rgb);
        }
    }
    Ok(patches)
}

fn cmd_color(args: ColorArgs, ctx: &CmdContext) -> Result<()> {
    if args.rows == 0 || args.cols == 0 {
        anyhow::bail!("--rows and --cols must both be at least 1");
    }

    let measured = match (&args.image, &args.measured) {
        (Some(image), _) => sample_chart(image, args.rows, args.cols, args.linear_input)?,
        (None, Some(patches)) => load_patches(patches)?,
        (None, None) => {
            anyhow::bail!("Specify either --image <chart.png> or --measured <patches.json>")
        }
    };

    let reference: Vec<[f32; 3]> = match args.reference {
        Some(ref path) => load_patches(path)?,
        None => cal_macbeth_patches()
            .iter()
            .map(|patch| patch.reference_rgb)
            .collect(),
    };

    if measured.len() != reference.len() {
        anyhow::bail!(
            "Measured patch count ({}) does not match the reference ({}). \
             Adjust --rows/--cols or supply --reference.",
            measured.len(),
            reference.len()
        );
    }

    let ccm: CalibrationMatrix = cal_solve_ccm(&measured, &reference)?;
    let white_balance = match (args.temperature, args.gray_patch) {
        (Some(kelvin), _) => WhiteBalance::from_temperature(kelvin)?,
        (None, Some(gray)) => WhiteBalance::from_gray_patch(gray)?,
        (None, None) => WhiteBalance::daylight(),
    };
    let stats = cal_evaluate(
        &measured,
        &reference,
        &ccm,
        &white_balance,
        args.metric.into(),
    )?;

    let payload = json!({
        "n_patches": measured.len(),
        "ccm": ccm.m,
        "white_balance_gains": white_balance.gains,
        "mean_delta_e": stats.mean_delta_e,
        "max_delta_e": stats.max_delta_e,
        "min_delta_e": stats.min_delta_e,
        "rmse_rgb": stats.rmse_rgb,
        "per_patch_delta_e": stats.per_patch_delta_e,
    });

    emit(ctx, "analyze color", payload, &[], || {
        println!("{}", cal_format_stats(&stats));
        println!("Colour correction matrix (row-major):");
        for row in 0..3 {
            println!(
                "  [{:>9.5} {:>9.5} {:>9.5}]",
                ccm.m[row * 3],
                ccm.m[row * 3 + 1],
                ccm.m[row * 3 + 2]
            );
        }
        println!(
            "White balance gains: [{:.5}, {:.5}, {:.5}]",
            white_balance.gains[0], white_balance.gains[1], white_balance.gains[2]
        );
    });
    Ok(())
}

// ---------------------------------------------------------------------------
// analyze diff
// ---------------------------------------------------------------------------

/// Arguments for `oxigaf analyze diff`.
#[derive(Debug, Args)]
pub struct DiffArgs {
    /// Baseline model (`.ply`, `.safetensors`, or `.json` checkpoint).
    pub before: PathBuf,

    /// Candidate model.
    pub after: PathBuf,

    /// Training step attributed to the baseline snapshot.
    #[arg(long, default_value = "0")]
    pub before_step: usize,

    /// Training step attributed to the candidate snapshot.
    #[arg(long, default_value = "0")]
    pub after_step: usize,

    /// Threshold below which an element counts as unchanged.
    #[arg(long, default_value = "1e-6")]
    pub epsilon: f32,

    /// Normalise differences by the mean magnitude of the baseline.
    #[arg(long)]
    pub normalize: bool,

    /// Exclude Gaussians whose activated opacity is below 0.1.
    #[arg(long)]
    pub skip_inactive: bool,

    /// Nearest-neighbour match radius used when the models differ in size.
    #[arg(long, default_value = "0.5")]
    pub match_radius: f32,

    /// Report the N Gaussians whose centres moved furthest (requires both
    /// models to hold the same number of Gaussians).
    #[arg(long, default_value = "0")]
    pub top_moved: usize,

    /// Run the regression check and fail when it trips.
    #[arg(long)]
    pub check_regression: bool,

    /// Maximum tolerated mean opacity *decrease* for `--check-regression`.
    #[arg(long, default_value = "0.05")]
    pub opacity_threshold: f32,

    /// Maximum tolerated mean scale *increase* for `--check-regression`.
    #[arg(long, default_value = "0.1")]
    pub scale_threshold: f32,

    /// Maximum tolerated RMS position change for `--check-regression`.
    #[arg(long, default_value = "0.01")]
    pub position_threshold: f32,
}

/// Build a diff snapshot from a model file.
fn snapshot(path: &Path, name: &str, step: usize) -> Result<ModelSnapshot> {
    let model = crate::export::load_model(path)
        .with_context(|| format!("Failed to load model: {}", path.display()))?;
    let n = model.gaussians.len();
    let channels = ((model.sh_degree + 1) * (model.sh_degree + 1) * 3) as usize;

    let mut positions = Vec::with_capacity(n * 3);
    let mut opacities = Vec::with_capacity(n);
    let mut scales = Vec::with_capacity(n * 3);
    let mut colors = Vec::with_capacity(n * 3);

    for (index, gaussian) in model.gaussians.iter().enumerate() {
        positions.extend_from_slice(&gaussian.position);
        scales.extend_from_slice(&gaussian.scale);
        opacities.push(gaussian.opacity);
        // The SH DC term is the first three channels of each Gaussian's
        // `[N, C]` row — the same slice `export_ply` writes as `f_dc_*`.
        let base = index * channels;
        for channel in 0..3 {
            colors.push(model.sh_coeffs.get(base + channel).copied().unwrap_or(0.0));
        }
    }

    let snap = ModelSnapshot::new(name, step, positions, opacities, scales, colors)?;
    Ok(snap)
}

fn diff_json(diff: &ModelDiff) -> serde_json::Value {
    json!({
        "name_a": diff.name_a,
        "name_b": diff.name_b,
        "step_a": diff.step_a,
        "step_b": diff.step_b,
        "n_gaussians": diff.n_gaussians,
        "n_compared": diff.n_compared,
        "added_gaussians": diff.added_gaussians,
        "removed_gaussians": diff.removed_gaussians,
        "summary_score": diff.summary_score,
        "position": field_json(&diff.position_diff),
        "opacity": field_json(&diff.opacity_diff),
        "scale": field_json(&diff.scale_diff),
        "color": field_json(&diff.color_diff),
    })
}

fn field_json(field: &FieldDiff) -> serde_json::Value {
    json!({
        "field_name": field.field_name,
        "mean_change": field.mean_change,
        "std_change": field.std_change,
        "max_abs_change": field.max_abs_change,
        "rms_change": field.rms_change,
        "fraction_changed": field.fraction_changed,
        "l2_distance": field.l2_distance,
        "cosine_similarity": field.cosine_similarity,
    })
}

fn cmd_diff(args: DiffArgs, ctx: &CmdContext) -> Result<()> {
    let config = DiffConfig {
        epsilon: args.epsilon,
        normalize: args.normalize,
        include_inactive: !args.skip_inactive,
        match_radius: args.match_radius,
    };
    config.validate()?;

    let before = snapshot(&args.before, "before", args.before_step)?;
    let after = snapshot(&args.after, "after", args.after_step)?;

    // `diff_models` requires equal Gaussian counts; densification and
    // pruning routinely change the count between checkpoints, so fall back
    // to the spatial-matching variant when the sizes differ.
    let variable = before.n_gaussians() != after.n_gaussians();
    let diff = if variable {
        diff_models_variable(&before, &after, &config)?
    } else {
        diff_models(&before, &after, &config)?
    };

    // `largest_position_changes` pairs Gaussians by index, which is only
    // meaningful when both models hold the same number of them.
    let moved = if args.top_moved > 0 && !variable {
        largest_position_changes(&before, &after, args.top_moved)?
    } else {
        Vec::new()
    };
    if args.top_moved > 0 && variable {
        tracing::warn!(
            "--top-moved needs both models to hold the same number of Gaussians \
             ({} vs {}); skipping the per-Gaussian ranking",
            before.n_gaussians(),
            after.n_gaussians(),
        );
    }

    let regression = args.check_regression.then(|| {
        detect_regression(
            &diff,
            args.opacity_threshold,
            args.scale_threshold,
            args.position_threshold,
        )
    });

    let mut payload = diff_json(&diff);
    if let Some(map) = payload.as_object_mut() {
        map.insert("spatial_matching".to_string(), json!(variable));
        if !moved.is_empty() {
            map.insert(
                "largest_position_changes".to_string(),
                json!(moved
                    .iter()
                    .map(|(index, delta)| json!({ "index": index, "distance": delta }))
                    .collect::<Vec<_>>()),
            );
        }
        if let Some(ref report) = regression {
            map.insert(
                "regression".to_string(),
                json!({
                    "overall_regression": report.overall_regression,
                    "opacity_regressed": report.opacity_regressed,
                    "scale_regressed": report.scale_regressed,
                    "position_unstable": report.position_unstable,
                    "details": report.details,
                }),
            );
        }
    }

    let regressed = regression
        .as_ref()
        .map(|report| report.overall_regression)
        .unwrap_or(false);
    let details: Vec<String> = regression.map(|report| report.details).unwrap_or_default();

    emit(ctx, "analyze diff", payload, &[], || {
        println!("{}", format_model_diff(&diff));
        if variable {
            println!(
                "(models differ in size; per-field statistics use spatial matching \
                 within {:.3} units)",
                config.match_radius
            );
        }
        if !moved.is_empty() {
            println!();
            println!("{}", format_field_diff_header());
            println!("{}", format_field_diff(&diff.position_diff));
            println!("Largest position changes:");
            for (index, distance) in &moved {
                println!("  #{index:<8} {distance:.6}");
            }
        }
        if args.check_regression {
            if details.is_empty() {
                println!("Regression check: no regression detected");
            } else {
                println!("Regression check:");
                for detail in &details {
                    println!("  {detail}");
                }
            }
        }
    });

    if regressed {
        anyhow::bail!("Regression detected between the two models");
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// analyze eval
// ---------------------------------------------------------------------------

/// Arguments for `oxigaf analyze eval`.
#[derive(Debug, Args)]
pub struct EvalArgs {
    /// Directory of rendered images.
    #[arg(long)]
    pub pred: PathBuf,

    /// Directory of ground-truth images, matched by file name.
    #[arg(long)]
    pub gt: PathBuf,

    /// Second set of rendered images; when given, the two candidate sets are
    /// compared against the same ground truth.
    #[arg(long)]
    pub compare_to: Option<PathBuf>,

    /// Number of worst-PSNR views reported.
    #[arg(long, default_value = "5")]
    pub n_worst: usize,

    /// Number of best-PSNR views reported.
    #[arg(long, default_value = "5")]
    pub n_best: usize,

    /// Also print the per-view table.
    #[arg(long)]
    pub per_view: bool,
}

/// Load an image as flat interleaved RGB floats in `[0, 1]`.
fn load_rgb(path: &Path) -> Result<(Vec<f32>, usize, usize)> {
    let image = image::open(path)
        .with_context(|| format!("Failed to open image: {}", path.display()))?
        .to_rgb8();
    let (width, height) = (image.width() as usize, image.height() as usize);
    let pixels: Vec<f32> = image
        .into_raw()
        .into_iter()
        .map(|v| f32::from(v) / 255.0)
        .collect();
    Ok((pixels, width, height))
}

fn image_files(dir: &Path) -> Result<Vec<PathBuf>> {
    if !dir.is_dir() {
        anyhow::bail!("Not a directory: {}", dir.display());
    }
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("Failed to read directory: {}", dir.display()))?
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_file()
                && path
                    .extension()
                    .and_then(|e| e.to_str())
                    .map(|e| {
                        matches!(
                            e.to_ascii_lowercase().as_str(),
                            "png" | "jpg" | "jpeg" | "bmp" | "tiff" | "tif"
                        )
                    })
                    .unwrap_or(false)
        })
        .collect();
    files.sort();
    Ok(files)
}

/// Build the evaluation items by pairing files with the same name.
fn build_items(pred_dir: &Path, gt_dir: &Path) -> Result<Vec<EvalTestItem>> {
    let pred_files = image_files(pred_dir)?;
    if pred_files.is_empty() {
        anyhow::bail!("No images found in {}", pred_dir.display());
    }

    let mut items = Vec::with_capacity(pred_files.len());
    for pred_path in pred_files {
        let name = pred_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("<unnamed>")
            .to_string();
        let gt_path = gt_dir.join(&name);
        if !gt_path.exists() {
            anyhow::bail!(
                "No ground-truth counterpart for {name}: expected {}",
                gt_path.display()
            );
        }
        let (pred, pw, ph) = load_rgb(&pred_path)?;
        let (gt, gw, gh) = load_rgb(&gt_path)?;
        if pw != gw || ph != gh {
            anyhow::bail!(
                "Resolution mismatch for {name}: rendered {pw}x{ph} vs ground truth {gw}x{gh}"
            );
        }
        items.push(EvalTestItem {
            view_id: name,
            pred,
            gt,
            width: pw,
            height: ph,
        });
    }
    Ok(items)
}

fn suite_json(result: &EvalSuiteResult) -> serde_json::Value {
    let percentiles = eval_psnr_percentiles(result);
    json!({
        "n_views": result.n_views,
        "mean_psnr": finite_or_null(result.mean_psnr),
        "mean_ssim": result.mean_ssim,
        "mean_lpips": result.mean_lpips,
        "mean_mae": result.mean_mae,
        "std_psnr": result.std_psnr,
        "min_psnr": finite_or_null(result.min_psnr),
        "max_psnr": finite_or_null(result.max_psnr),
        "psnr_percentiles": {
            "p5": finite_or_null(percentiles[0]),
            "p25": finite_or_null(percentiles[1]),
            "p50": finite_or_null(percentiles[2]),
            "p75": finite_or_null(percentiles[3]),
            "p95": finite_or_null(percentiles[4]),
        },
        "worst_views": result.worst_views,
        "best_views": result.best_views,
    })
}

/// JSON has no representation for infinity; a pixel-perfect view yields an
/// infinite PSNR, so report it as `null` rather than emitting invalid JSON.
fn finite_or_null(value: f32) -> serde_json::Value {
    if value.is_finite() {
        json!(value)
    } else {
        serde_json::Value::Null
    }
}

fn cmd_eval(args: EvalArgs, ctx: &CmdContext) -> Result<()> {
    let config = EvalConfig {
        n_worst_views: args.n_worst,
        n_best_views: args.n_best,
        ..EvalConfig::default()
    };

    let items = build_items(&args.pred, &args.gt)?;
    let result = crate::evaluation_suite::eval_suite(&items, &config)?;

    let comparison: Option<EvalSuiteResult> = match args.compare_to {
        Some(ref other_dir) => {
            let other_items = build_items(other_dir, &args.gt)?;
            Some(crate::evaluation_suite::eval_suite(&other_items, &config)?)
        }
        None => None,
    };

    let mut payload = json!({
        "pred": args.pred.display().to_string(),
        "gt": args.gt.display().to_string(),
        "baseline": suite_json(&result),
    });

    let mut comparison_text = String::new();
    if let Some(ref candidate) = comparison {
        let delta = eval_compare(&result, candidate)?;
        comparison_text = eval_format_comparison(&delta);
        if let Some(map) = payload.as_object_mut() {
            map.insert("candidate".to_string(), suite_json(candidate));
            map.insert(
                "comparison".to_string(),
                json!({
                    "delta_psnr": finite_or_null(delta.delta_psnr),
                    "delta_ssim": delta.delta_ssim,
                    "delta_lpips": delta.delta_lpips,
                    "n_views_improved": delta.n_views_improved,
                    "n_views_degraded": delta.n_views_degraded,
                    "is_candidate_better": delta.is_candidate_better,
                }),
            );
        }
    }

    if args.per_view {
        if let Some(map) = payload.as_object_mut() {
            map.insert(
                "per_view".to_string(),
                json!(result
                    .per_view
                    .iter()
                    .map(|view| json!({
                        "view_id": view.view_id,
                        "psnr": finite_or_null(view.psnr),
                        "ssim": view.ssim,
                        "ssim_ms": view.ssim_ms,
                        "lpips_approx": view.lpips_approx,
                        "mae": view.mae,
                        "rmse": view.rmse,
                    }))
                    .collect::<Vec<_>>()),
            );
        }
    }

    emit(ctx, "analyze eval", payload, &[], || {
        println!("{}", eval_format_suite_result(&result));
        if args.per_view {
            println!();
            for view in &result.per_view {
                println!("{}", crate::evaluation_suite::eval_format_view_result(view));
            }
        }
        if !comparison_text.is_empty() {
            println!();
            println!("{comparison_text}");
        }
    });
    Ok(())
}

/// Run the `analyze` family.
///
/// # Errors
///
/// Returns an error when an input file or directory cannot be read, when a
/// library routine rejects its inputs, or (for `analyze diff`) when a
/// regression threshold is exceeded.
pub fn run(args: AnalyzeArgs, ctx: CmdContext) -> Result<()> {
    match args.command {
        AnalyzeCommand::Color(color_args) => cmd_color(color_args, &ctx),
        AnalyzeCommand::Diff(diff_args) => cmd_diff(diff_args, &ctx),
        AnalyzeCommand::Eval(eval_args) => cmd_eval(eval_args, &ctx),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn macbeth_reference_is_the_default_and_has_24_patches() {
        let reference: Vec<[f32; 3]> = cal_macbeth_patches()
            .iter()
            .map(|patch| patch.reference_rgb)
            .collect();
        assert_eq!(reference.len(), 24);
    }

    #[test]
    fn identity_measurement_fits_an_identity_ccm() {
        let reference: Vec<[f32; 3]> = cal_macbeth_patches()
            .iter()
            .map(|patch| patch.reference_rgb)
            .collect();
        let ccm = cal_solve_ccm(&reference, &reference).expect("well-conditioned fit");
        let stats = cal_evaluate(
            &reference,
            &reference,
            &ccm,
            &WhiteBalance::daylight(),
            DeltaEMetric::Cie2000,
        )
        .expect("evaluation succeeds");
        assert!(
            stats.mean_delta_e < 1.0,
            "mean delta-E {} should be near zero for an identity fit",
            stats.mean_delta_e
        );
    }

    #[test]
    fn delta_e_flag_maps_onto_the_library_metric() {
        assert!(matches!(
            DeltaEMetric::from(DeltaE::Cie76),
            DeltaEMetric::Cie76
        ));
        assert!(matches!(
            DeltaEMetric::from(DeltaE::Cie2000),
            DeltaEMetric::Cie2000
        ));
    }

    #[test]
    fn finite_or_null_maps_infinity_to_json_null() {
        assert_eq!(finite_or_null(f32::INFINITY), serde_json::Value::Null);
        assert_eq!(finite_or_null(31.5), json!(31.5));
    }

    #[test]
    fn build_items_rejects_a_missing_ground_truth_dir() {
        let missing = std::env::temp_dir().join("oxigaf_analyze_eval_missing");
        let _ = std::fs::remove_dir_all(&missing);
        assert!(build_items(&missing, &missing).is_err());
    }
}
