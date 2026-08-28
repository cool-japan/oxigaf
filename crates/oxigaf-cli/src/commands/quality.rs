//! `oxigaf quality` — threshold-based quality control for rendered images.
//!
//! Glue over [`crate::quality_checker`].
//!
//! | Subcommand | Library entry point |
//! |------------|---------------------|
//! | `check`     | [`crate::quality_checker::check_quality`] |
//! | `batch`     | [`crate::quality_checker::check_quality_batch`] |
//! | `artifacts` | [`crate::quality_checker::detect_artifacts`] |
//! | `error-map` | [`crate::quality_checker::error_map`] |
//! | `histogram` | [`crate::quality_checker::compute_histogram`] |
//!
//! # Relationship to `oxigaf analyze eval`
//!
//! `analyze eval` ([`crate::evaluation_suite`]) *scores* a render set — PSNR /
//! SSIM / LPIPS percentiles across views, for reporting. This family *gates*
//! one: it applies pass/fail thresholds, hunts for the specific artefacts a
//! 3DGS render goes wrong with (clipping, colour drift, banding, noise), and
//! — unless `--allow-failure` is given — exits non-zero when the render does
//! not meet them, so it can stand in a CI pipeline. The two share no code and
//! answer different questions.
//!
//! # Exit codes
//!
//! An unusable input (missing file, undecodable image, mismatched
//! dimensions) is [`crate::error::CliError::InputInvalid`] →
//! [`crate::error::EXIT_IO_ERROR`]. A render that *loads fine but fails the
//! thresholds* is the catch-all status 1: the command worked, the render did
//! not.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Args, Subcommand, ValueHint};
use serde_json::{json, Value};

use crate::commands::image_io::{image_files, input_invalid, load_rgba, parse_hex_rgb};
use crate::commands::{emit, prepare_output, CmdContext};
use crate::progress_types::BatchProgress;
use crate::quality_checker::{
    check_quality, check_quality_batch, compute_histogram, detect_artifacts, error_map,
    error_map_to_heatmap, histogram_distance, is_blank_image, ArtifactReport, BatchQualityReport,
    ImageQualityMetrics, QualityReport, QualityThresholds,
};

/// `oxigaf quality <command>`.
#[derive(Debug, Args)]
pub struct QualityArgs {
    #[command(subcommand)]
    pub command: QualityCommand,
}

/// Quality-control subcommands.
#[derive(Debug, Subcommand)]
pub enum QualityCommand {
    /// Score one rendered image against its reference and apply thresholds.
    Check(CheckArgs),

    /// Apply the same check to every matching pair in two directories.
    Batch(BatchArgs),

    /// Hunt for rendering artefacts in a single image, with no reference.
    Artifacts(ArtifactsArgs),

    /// Write a colour-coded per-pixel error heatmap between two images.
    ErrorMap(ErrorMapArgs),

    /// Report a per-channel histogram, optionally against a second image.
    Histogram(HistogramArgs),
}

/// The pass/fail thresholds, shared by `check`, `batch` and `artifacts`.
#[derive(Debug, Args)]
pub struct ThresholdArgs {
    /// Minimum acceptable PSNR in dB.
    #[arg(long, default_value = "25.0")]
    pub min_psnr: f32,

    /// Minimum acceptable SSIM in `[0, 1]`.
    #[arg(long, default_value = "0.85")]
    pub min_ssim: f32,

    /// Maximum acceptable fraction of clipped pixels.
    #[arg(long, default_value = "0.05")]
    pub max_clipping: f32,

    /// Maximum acceptable estimated noise level.
    #[arg(long, default_value = "0.05")]
    pub max_noise: f32,

    /// Flat background fill colour as `rrggbb`, excluded from clipping
    /// detection.
    ///
    /// A composited avatar render is often 40–80 % solid background, and a
    /// solid black or white fill saturates every channel — without this the
    /// render is reported as clipped no matter what the subject looks like.
    #[arg(long, value_name = "RRGGBB", value_parser = parse_hex_rgb)]
    pub background: Option<[u8; 3]>,
}

impl ThresholdArgs {
    /// Build the library's threshold struct, validating the ranges first.
    ///
    /// # Errors
    ///
    /// Returns an error when a threshold is outside the range its metric can
    /// ever produce, which would otherwise make the check unconditionally
    /// pass or unconditionally fail.
    fn to_thresholds(&self) -> Result<QualityThresholds> {
        if self.min_psnr <= 0.0 || !self.min_psnr.is_finite() {
            anyhow::bail!(
                "--min-psnr must be a finite value above 0 (got {})",
                self.min_psnr
            );
        }
        if !(0.0..=1.0).contains(&self.min_ssim) {
            anyhow::bail!(
                "--min-ssim must be within 0.0..=1.0 (got {})",
                self.min_ssim
            );
        }
        if !(0.0..=1.0).contains(&self.max_clipping) {
            anyhow::bail!(
                "--max-clipping is a fraction and must be within 0.0..=1.0 (got {})",
                self.max_clipping
            );
        }
        if !(0.0..=1.0).contains(&self.max_noise) {
            anyhow::bail!(
                "--max-noise must be within 0.0..=1.0 (got {})",
                self.max_noise
            );
        }
        Ok(QualityThresholds {
            min_psnr: self.min_psnr,
            min_ssim: self.min_ssim,
            max_clipping_pct: self.max_clipping,
            max_noise_level: self.max_noise,
            background_color: self.background,
        })
    }
}

/// Arguments for `oxigaf quality check`.
#[derive(Debug, Args)]
pub struct CheckArgs {
    /// The rendered image under test.
    #[arg(long, value_hint = ValueHint::FilePath)]
    pub rendered: PathBuf,

    /// The ground-truth image to compare against.
    #[arg(long, value_hint = ValueHint::FilePath)]
    pub reference: PathBuf,

    #[command(flatten)]
    pub thresholds: ThresholdArgs,

    /// Report the result and exit 0 even when the thresholds are not met.
    #[arg(long)]
    pub allow_failure: bool,
}

/// Arguments for `oxigaf quality batch`.
#[derive(Debug, Args)]
pub struct BatchArgs {
    /// Directory of rendered images.
    #[arg(long, value_hint = ValueHint::DirPath)]
    pub rendered_dir: PathBuf,

    /// Directory of reference images, matched to `--rendered-dir` by file name.
    #[arg(long, value_hint = ValueHint::DirPath)]
    pub reference_dir: PathBuf,

    #[command(flatten)]
    pub thresholds: ThresholdArgs,

    /// Minimum fraction of images that must pass, in `[0, 1]`.
    ///
    /// Defaults to 1.0: every image must pass.
    #[arg(long, default_value = "1.0")]
    pub min_pass_rate: f32,

    /// Also print a line per image instead of only the aggregate.
    #[arg(long)]
    pub per_image: bool,

    /// Report the result and exit 0 even when `--min-pass-rate` is not met.
    #[arg(long)]
    pub allow_failure: bool,
}

/// Arguments for `oxigaf quality artifacts`.
#[derive(Debug, Args)]
pub struct ArtifactsArgs {
    /// Image to inspect.
    #[arg(long, value_hint = ValueHint::FilePath)]
    pub image: PathBuf,

    #[command(flatten)]
    pub thresholds: ThresholdArgs,

    /// Per-channel tolerance for the "is this frame blank?" test.
    #[arg(long, default_value = "2")]
    pub blank_tolerance: u8,
}

/// Arguments for `oxigaf quality error-map`.
#[derive(Debug, Args)]
pub struct ErrorMapArgs {
    /// The rendered image under test.
    #[arg(long, value_hint = ValueHint::FilePath)]
    pub rendered: PathBuf,

    /// The ground-truth image to compare against.
    #[arg(long, value_hint = ValueHint::FilePath)]
    pub reference: PathBuf,

    /// Where to write the heatmap (PNG).
    #[arg(short, long, value_hint = ValueHint::FilePath)]
    pub output: PathBuf,

    /// Overwrite the output file if it already exists.
    #[arg(long)]
    pub force: bool,
}

/// Arguments for `oxigaf quality histogram`.
#[derive(Debug, Args)]
pub struct HistogramArgs {
    /// Image to histogram.
    #[arg(long, value_hint = ValueHint::FilePath)]
    pub image: PathBuf,

    /// Second image; when given, the normalised L1 distance is reported.
    #[arg(long, value_hint = ValueHint::FilePath)]
    pub compare: Option<PathBuf>,
}

/// Run the `quality` family.
///
/// # Errors
///
/// Propagates unusable inputs and, unless `--allow-failure` is set, turns a
/// failed quality gate into a non-zero exit status.
pub fn run(args: QualityArgs, ctx: CmdContext) -> Result<()> {
    match args.command {
        QualityCommand::Check(check_args) => cmd_check(check_args, &ctx),
        QualityCommand::Batch(batch_args) => cmd_batch(batch_args, &ctx),
        QualityCommand::Artifacts(artifact_args) => cmd_artifacts(artifact_args, &ctx),
        QualityCommand::ErrorMap(map_args) => cmd_error_map(map_args, &ctx),
        QualityCommand::Histogram(hist_args) => cmd_histogram(hist_args, &ctx),
    }
}

// ---------------------------------------------------------------------------
// JSON helpers
// ---------------------------------------------------------------------------

/// Render an `f32` for JSON, mapping non-finite values to `null`.
///
/// PSNR is [`f32::INFINITY`] for a byte-identical pair, and `serde_json`
/// silently turns any non-finite float into `null` anyway — doing it
/// explicitly keeps that behaviour visible next to the field it affects.
fn json_f32(value: f32) -> Value {
    if value.is_finite() {
        json!(value)
    } else {
        Value::Null
    }
}

/// The JSON shape of [`ImageQualityMetrics`].
fn metrics_json(metrics: &ImageQualityMetrics) -> Value {
    json!({
        "psnr_db": json_f32(metrics.psnr),
        "mse": json_f32(metrics.mse),
        "mae": json_f32(metrics.mae),
        // `null` when the image was too small for SSIM to be evaluated —
        // distinct from a genuine score of 0.
        "ssim": metrics.ssim.map(json_f32).unwrap_or(Value::Null),
        "max_error": json_f32(metrics.max_error),
        "width": metrics.width,
        "height": metrics.height,
    })
}

/// The JSON shape of [`ArtifactReport`].
fn artifacts_json(artifacts: &ArtifactReport) -> Value {
    json!({
        "has_clipping": artifacts.has_clipping,
        "clipping_fraction": json_f32(artifacts.clipping_fraction),
        "has_color_drift": artifacts.has_color_drift,
        "color_drift_magnitude": json_f32(artifacts.color_drift_magnitude),
        "has_excessive_noise": artifacts.has_excessive_noise,
        "noise_level": json_f32(artifacts.noise_level),
        "has_banding": artifacts.has_banding,
        "banding_score": json_f32(artifacts.banding_score),
        "overall_score": json_f32(artifacts.overall_score),
    })
}

/// The JSON shape of a single [`QualityReport`].
fn report_json(report: &QualityReport) -> Value {
    json!({
        "passed": report.passed,
        "issues": report.issues,
        "metrics": metrics_json(&report.metrics),
        "artifacts": artifacts_json(&report.artifacts),
    })
}

/// Human-readable rendering of an [`ArtifactReport`].
///
/// [`crate::quality_checker`] ships formatters for its metrics and its
/// reports but not for a bare artefact scan, so this is the one piece of
/// formatting the family owns.
fn format_artifacts(artifacts: &ArtifactReport) -> String {
    let flag = |set: bool| if set { "FAIL" } else { "ok  " };
    format!(
        "Artifact scan\n\
         -------------\n\
         [{}] clipping      fraction {:.4}\n\
         [{}] colour drift  magnitude {:.4}\n\
         [{}] noise         level {:.4}\n\
         [{}] banding       score {:.4}\n\
         overall score {:.4} (0 = clean, 1 = heavily artefacted)",
        flag(artifacts.has_clipping),
        artifacts.clipping_fraction,
        flag(artifacts.has_color_drift),
        artifacts.color_drift_magnitude,
        flag(artifacts.has_excessive_noise),
        artifacts.noise_level,
        flag(artifacts.has_banding),
        artifacts.banding_score,
        artifacts.overall_score,
    )
}

/// Report a size mismatch as a bad *input* (exit 3) that names both sides.
///
/// A thin, named wrapper over [`input_invalid`] so the three call sites
/// below don't repeat "this is a size-mismatch input error" inline.
/// `CliError::InputInvalid`'s `Display` renders `reason` (`error.rs`), so
/// `sentence` — here the two resolutions, the only thing that tells a
/// caller which side is wrong — reaches both `{}` (this message) and `{:#}`
/// (`commands::error_report::classify_error`, what the process prints)
/// without needing an extra `anyhow` context layer; the typed error
/// underneath still selects the exit code.
fn size_mismatch(path: &Path, sentence: String) -> anyhow::Error {
    input_invalid(path, sentence)
}

/// Load a (reference, rendered) pair and confirm the dimensions agree.
///
/// Returns `(reference_rgba, rendered_rgba, width, height)`.
fn load_pair(reference: &Path, rendered: &Path) -> Result<(Vec<u8>, Vec<u8>, u32, u32)> {
    let (reference_pixels, width, height) = load_rgba(reference)?;
    let (rendered_pixels, rendered_w, rendered_h) = load_rgba(rendered)?;
    if (width, height) != (rendered_w, rendered_h) {
        return Err(size_mismatch(
            rendered,
            format!(
                "{} is {rendered_w}×{rendered_h} but the reference {} is {width}×{height}; \
                 quality metrics need identical dimensions",
                rendered.display(),
                reference.display()
            ),
        ));
    }
    Ok((reference_pixels, rendered_pixels, width, height))
}

// ---------------------------------------------------------------------------
// quality check
// ---------------------------------------------------------------------------

fn cmd_check(args: CheckArgs, ctx: &CmdContext) -> Result<()> {
    let thresholds = args.thresholds.to_thresholds()?;
    let (reference, rendered, width, height) = load_pair(&args.reference, &args.rendered)?;

    let report = check_quality(&reference, &rendered, width, height, &thresholds)?;
    let passed = report.passed;

    emit(
        ctx,
        "quality check",
        json!({
            "rendered": args.rendered.display().to_string(),
            "reference": args.reference.display().to_string(),
            "report": report_json(&report),
        }),
        &[],
        || println!("{}", report.format_report()),
    );

    finish_gate(passed, args.allow_failure, "quality check failed")
}

// ---------------------------------------------------------------------------
// quality batch
// ---------------------------------------------------------------------------

fn cmd_batch(args: BatchArgs, ctx: &CmdContext) -> Result<()> {
    if !(0.0..=1.0).contains(&args.min_pass_rate) {
        anyhow::bail!(
            "--min-pass-rate is a fraction and must be within 0.0..=1.0 (got {})",
            args.min_pass_rate
        );
    }
    let thresholds = args.thresholds.to_thresholds()?;

    let rendered_files = image_files(&args.rendered_dir)?;
    if rendered_files.is_empty() {
        return Err(input_invalid(&args.rendered_dir, "contains no image files"));
    }

    // Decoding is by far the slow part, so the bar tracks the load rather
    // than the (in-memory) scoring pass. indicatif draws on stderr, which
    // keeps stdout a pure JSON stream, but it is still suppressed under
    // `--json` and `-q` so a redirected run stays silent.
    let progress = if ctx.human() && ctx.verbosity.show_progress() {
        Some(BatchProgress::new(
            rendered_files.len() as u64,
            "image pairs loaded",
        ))
    } else {
        None
    };

    let mut names = Vec::with_capacity(rendered_files.len());
    let mut loaded: Vec<(Vec<u8>, Vec<u8>)> = Vec::with_capacity(rendered_files.len());
    let mut dimensions: Option<(u32, u32)> = None;

    for rendered_path in &rendered_files {
        let name = rendered_path
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| input_invalid(rendered_path, "has a non-UTF-8 file name"))?
            .to_string();
        let reference_path = args.reference_dir.join(&name);
        if !reference_path.is_file() {
            return Err(input_invalid(
                &reference_path,
                format!("is missing: no reference counterpart for {name}"),
            ));
        }

        let (reference, rendered, width, height) = load_pair(&reference_path, rendered_path)?;
        match dimensions {
            None => dimensions = Some((width, height)),
            // `check_quality_batch` takes one width/height for the whole
            // batch, so a mixed-resolution directory has to be refused here
            // rather than silently scored against the first frame's size.
            Some((w, h)) if (w, h) != (width, height) => {
                return Err(size_mismatch(
                    rendered_path,
                    format!(
                        "{} is {width}×{height} but earlier frames are {w}×{h}; \
                         a batch must be one resolution",
                        rendered_path.display()
                    ),
                ));
            }
            Some(_) => {}
        }

        names.push(name);
        loaded.push((reference, rendered));
        if let Some(ref bar) = progress {
            bar.increment();
        }
    }
    if let Some(ref bar) = progress {
        bar.finish();
    }

    let (width, height) = dimensions
        .ok_or_else(|| input_invalid(&args.rendered_dir, "produced no loadable image pairs"))?;

    let pairs: Vec<(&[u8], &[u8])> = loaded
        .iter()
        .map(|(reference, rendered)| (reference.as_slice(), rendered.as_slice()))
        .collect();
    let batch = check_quality_batch(&pairs, width, height, &thresholds)?;
    let pass_rate = batch.pass_rate();
    let gate_passed = pass_rate >= args.min_pass_rate;

    let per_image: Vec<Value> = names
        .iter()
        .zip(batch.reports.iter())
        .map(|(name, report)| {
            let mut entry = report_json(report);
            if let Value::Object(ref mut map) = entry {
                map.insert("image".to_string(), json!(name));
            }
            entry
        })
        .collect();

    emit(
        ctx,
        "quality batch",
        json!({
            "rendered_dir": args.rendered_dir.display().to_string(),
            "reference_dir": args.reference_dir.display().to_string(),
            "total_images": batch.total_images,
            "passed": batch.passed_count,
            "failed": batch.failed_count,
            "pass_rate": json_f32(pass_rate),
            "min_pass_rate": json_f32(args.min_pass_rate),
            "gate_passed": gate_passed,
            "mean_psnr_db": json_f32(batch.mean_psnr),
            "min_psnr_db": json_f32(batch.min_psnr),
            "max_psnr_db": json_f32(batch.max_psnr),
            "mean_ssim": json_f32(batch.mean_ssim),
            "images": per_image,
        }),
        &[],
        || print_batch(&args, &names, &batch),
    );

    finish_gate(
        gate_passed,
        args.allow_failure,
        &format!(
            "quality batch failed: pass rate {:.1}% is below the required {:.1}%",
            pass_rate * 100.0,
            args.min_pass_rate * 100.0
        ),
    )
}

/// Human-readable rendering of a batch run.
fn print_batch(args: &BatchArgs, names: &[String], batch: &BatchQualityReport) {
    if args.per_image {
        for (name, report) in names.iter().zip(batch.reports.iter()) {
            let verdict = if report.passed { "PASS" } else { "FAIL" };
            println!("{verdict}  {name}  {}", report.metrics.format_summary());
            for issue in &report.issues {
                println!("        - {issue}");
            }
        }
        println!();
    }
    println!("{}", batch.format_summary());
}

// ---------------------------------------------------------------------------
// quality artifacts
// ---------------------------------------------------------------------------

fn cmd_artifacts(args: ArtifactsArgs, ctx: &CmdContext) -> Result<()> {
    let thresholds = args.thresholds.to_thresholds()?;
    let (pixels, width, height) = load_rgba(&args.image)?;

    let artifacts = detect_artifacts(&pixels, width, height, &thresholds);
    let blank = is_blank_image(&pixels, args.blank_tolerance);

    emit(
        ctx,
        "quality artifacts",
        json!({
            "image": args.image.display().to_string(),
            "width": width,
            "height": height,
            "is_blank": blank,
            "blank_tolerance": args.blank_tolerance,
            "artifacts": artifacts_json(&artifacts),
        }),
        &[],
        || {
            println!("{}", format_artifacts(&artifacts));
            if blank {
                println!(
                    "\nNOTE: every pixel is within ±{} of the first one — this frame is blank.",
                    args.blank_tolerance
                );
            }
        },
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// quality error-map
// ---------------------------------------------------------------------------

fn cmd_error_map(args: ErrorMapArgs, ctx: &CmdContext) -> Result<()> {
    let (reference, rendered, width, height) = load_pair(&args.reference, &args.rendered)?;

    let errors = error_map(&reference, &rendered)?;
    let heatmap = error_map_to_heatmap(&errors);

    // Peak and mean of the scalar (channel-averaged) error, reported so the
    // heatmap's arbitrary normalisation can be interpreted.
    let scalars: Vec<f32> = errors
        .chunks_exact(3)
        .map(|c| (c[0] + c[1] + c[2]) / 3.0)
        .collect();
    let max_error = scalars.iter().copied().fold(0.0f32, f32::max);
    let mean_error = if scalars.is_empty() {
        0.0
    } else {
        scalars.iter().sum::<f32>() / scalars.len() as f32
    };

    if !prepare_output(ctx, &args.output, args.force)? {
        emit(
            ctx,
            "quality error-map",
            json!({
                "dry_run": true,
                "would_create": [args.output.display().to_string()],
                "width": width,
                "height": height,
            }),
            &[],
            || println!("Would write heatmap: {}", args.output.display()),
        );
        return Ok(());
    }

    let buffer: image::RgbaImage = image::ImageBuffer::from_raw(width, height, heatmap)
        .ok_or_else(|| anyhow::anyhow!("heatmap buffer does not match {width}×{height}"))?;
    buffer
        .save(&args.output)
        .with_context(|| format!("Failed to write heatmap: {}", args.output.display()))?;

    emit(
        ctx,
        "quality error-map",
        json!({
            "rendered": args.rendered.display().to_string(),
            "reference": args.reference.display().to_string(),
            "output": args.output.display().to_string(),
            "width": width,
            "height": height,
            "max_error": json_f32(max_error),
            "mean_error": json_f32(mean_error),
        }),
        &[("heatmap", args.output.as_path())],
        || {
            println!(
                "Wrote {width}×{height} heatmap to {}",
                args.output.display()
            );
            println!("  mean error {mean_error:.4}, peak error {max_error:.4}");
        },
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// quality histogram
// ---------------------------------------------------------------------------

fn cmd_histogram(args: HistogramArgs, ctx: &CmdContext) -> Result<()> {
    let (pixels, width, height) = load_rgba(&args.image)?;
    let histogram = compute_histogram(&pixels);
    let n_pixels = width.saturating_mul(height);

    let (comparison, distance) = match args.compare {
        Some(ref other_path) => {
            let (other_pixels, other_w, other_h) = load_rgba(other_path)?;
            if (other_w, other_h) != (width, height) {
                return Err(size_mismatch(
                    other_path,
                    format!(
                        "{} is {other_w}×{other_h} but {} is {width}×{height}; a histogram \
                         distance normalised by pixel count needs both to match",
                        other_path.display(),
                        args.image.display()
                    ),
                ));
            }
            let other = compute_histogram(&other_pixels);
            (
                Some(other_path.display().to_string()),
                Some(histogram_distance(&histogram, &other, n_pixels)),
            )
        }
        None => (None, None),
    };

    // The library returns one flat `[R0, G0, B0, R1, …]` array; splitting it
    // per channel is what makes the JSON usable from a plotting script.
    let channel = |offset: usize| -> Vec<u32> {
        (0..256)
            .map(|bin| histogram.get(bin * 3 + offset).copied().unwrap_or(0))
            .collect()
    };

    emit(
        ctx,
        "quality histogram",
        json!({
            "image": args.image.display().to_string(),
            "width": width,
            "height": height,
            "compare": comparison,
            "l1_distance": distance.map(json_f32).unwrap_or(Value::Null),
            "red": channel(0),
            "green": channel(1),
            "blue": channel(2),
        }),
        &[],
        || {
            println!("{}  {width}×{height}", args.image.display());
            for (label, offset) in [("red", 0usize), ("green", 1), ("blue", 2)] {
                let bins = channel(offset);
                let total: u64 = bins.iter().map(|&v| u64::from(v)).sum();
                let peak_bin = bins
                    .iter()
                    .enumerate()
                    .max_by_key(|(_, &count)| count)
                    .map(|(bin, _)| bin)
                    .unwrap_or(0);
                let mean = if total == 0 {
                    0.0
                } else {
                    bins.iter()
                        .enumerate()
                        .map(|(bin, &count)| bin as f64 * f64::from(count))
                        .sum::<f64>()
                        / total as f64
                };
                println!("  {label:<5}  mean {mean:6.2}  peak bin {peak_bin:3}");
            }
            if let Some(d) = distance {
                println!("  L1 distance to the comparison image: {d:.4}");
            }
        },
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Gate handling
// ---------------------------------------------------------------------------

/// Turn a failed quality gate into a non-zero exit status.
///
/// The report has already been printed (or emitted as JSON) by the time this
/// runs, so the caller sees *why* it failed and the shell sees *that* it
/// failed. `--allow-failure` keeps the report and drops the status change,
/// which is what a "collect the numbers, do not block the build" run wants.
fn finish_gate(passed: bool, allow_failure: bool, message: &str) -> Result<()> {
    if passed || allow_failure {
        return Ok(());
    }
    anyhow::bail!("{message} (pass --allow-failure to report without failing)")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::EXIT_IO_ERROR;
    use crate::verbosity::Verbosity;

    fn ctx() -> CmdContext {
        CmdContext::new(Verbosity::Quiet, true, false)
    }

    fn thresholds(min_psnr: f32, min_ssim: f32) -> ThresholdArgs {
        ThresholdArgs {
            min_psnr,
            min_ssim,
            max_clipping: 0.05,
            max_noise: 0.05,
            background: None,
        }
    }

    /// Write an 8×8 solid-colour PNG and return its path.
    fn write_png(name: &str, rgba: [u8; 4]) -> PathBuf {
        let path = std::env::temp_dir().join(name);
        let buffer: image::RgbaImage = image::ImageBuffer::from_pixel(8, 8, image::Rgba(rgba));
        buffer.save(&path).expect("temp PNG write");
        path
    }

    #[test]
    fn thresholds_reject_out_of_range_values() {
        assert!(thresholds(-1.0, 0.5).to_thresholds().is_err());
        assert!(thresholds(25.0, 1.5).to_thresholds().is_err());
        assert!(thresholds(25.0, 0.85).to_thresholds().is_ok());
    }

    /// Regression: a rendered/reference size mismatch must be reported as a
    /// bad *input* (exit 3), not as the catch-all failure, and must name both
    /// resolutions so the caller can see which side is wrong.
    ///
    /// The resolutions used to be passed as `CliError::InputInvalid`'s
    /// `reason`, which its `Display` never printed — the whole message was
    /// "Invalid input file: <path>" — so `size_mismatch` compensated by
    /// attaching the same sentence again as `anyhow` context. `Display` now
    /// renders `reason` directly (`error.rs`) and that context layer is
    /// gone, so both renderings are checked here: `{}` (this message) and
    /// `{:#}` (what the process actually prints, via `classify_error`) must
    /// each contain the resolutions exactly once, not twice from a
    /// leftover duplicate layer.
    #[test]
    fn mismatched_dimensions_are_input_errors() {
        let reference = write_png("oxigaf_quality_ref_8x8.png", [10, 20, 30, 255]);
        let rendered = std::env::temp_dir().join("oxigaf_quality_rendered_4x4.png");
        let buffer: image::RgbaImage =
            image::ImageBuffer::from_pixel(4, 4, image::Rgba([10, 20, 30, 255]));
        buffer.save(&rendered).expect("temp PNG write");

        let err = load_pair(&reference, &rendered).expect_err("a size mismatch must not load");
        let message = format!("{err}");
        assert!(message.contains("4×4"), "message was: {message}");
        assert!(message.contains("8×8"), "message was: {message}");

        let (cli_err, detail) = crate::commands::error_report::classify_error(err);
        assert!(detail.contains("4×4"), "rendered chain was: {detail}");
        assert!(detail.contains("8×8"), "rendered chain was: {detail}");
        assert_eq!(
            detail.matches("4×4").count(),
            1,
            "resolution must appear exactly once, not duplicated by a \
             leftover context layer: {detail}"
        );
        assert_eq!(cli_err.exit_code(), EXIT_IO_ERROR);

        let _ = std::fs::remove_file(&reference);
        let _ = std::fs::remove_file(&rendered);
    }

    /// The batch walk refuses a mixed-resolution directory, and must say
    /// which frame broke the run and at what size.
    #[test]
    fn a_mixed_resolution_batch_names_the_offending_frame() {
        let rendered_dir = std::env::temp_dir().join("oxigaf_quality_batch_mixed_rendered");
        let reference_dir = std::env::temp_dir().join("oxigaf_quality_batch_mixed_reference");
        let _ = std::fs::remove_dir_all(&rendered_dir);
        let _ = std::fs::remove_dir_all(&reference_dir);
        std::fs::create_dir_all(&rendered_dir).expect("create dir");
        std::fs::create_dir_all(&reference_dir).expect("create dir");

        for (index, size) in [(0usize, 8u32), (1, 4)] {
            let name = format!("frame_{index:03}.png");
            let buffer: image::RgbaImage =
                image::ImageBuffer::from_pixel(size, size, image::Rgba([9, 9, 9, 255]));
            buffer
                .save(rendered_dir.join(&name))
                .expect("temp PNG write");
            buffer
                .save(reference_dir.join(&name))
                .expect("temp PNG write");
        }

        let args = BatchArgs {
            rendered_dir: rendered_dir.clone(),
            reference_dir: reference_dir.clone(),
            thresholds: thresholds(25.0, 0.85),
            min_pass_rate: 0.5,
            per_image: false,
            allow_failure: false,
        };
        let err = cmd_batch(args, &ctx()).expect_err("a mixed-resolution batch must not run");
        let message = format!("{err}");
        assert!(message.contains("frame_001"), "message was: {message}");
        assert!(message.contains("4×4"), "message was: {message}");
        assert!(message.contains("8×8"), "message was: {message}");

        let _ = std::fs::remove_dir_all(&rendered_dir);
        let _ = std::fs::remove_dir_all(&reference_dir);
    }

    /// An identical pair has infinite PSNR; the gate must pass and the JSON
    /// must carry `null` rather than a bogus finite number.
    #[test]
    fn identical_images_pass_the_gate() {
        let reference = write_png("oxigaf_quality_same_a.png", [128, 64, 32, 255]);
        let rendered = write_png("oxigaf_quality_same_b.png", [128, 64, 32, 255]);

        let args = CheckArgs {
            rendered: rendered.clone(),
            reference: reference.clone(),
            thresholds: thresholds(25.0, 0.85),
            allow_failure: false,
        };
        assert!(cmd_check(args, &ctx()).is_ok());
        assert_eq!(json_f32(f32::INFINITY), Value::Null);

        let _ = std::fs::remove_file(&reference);
        let _ = std::fs::remove_file(&rendered);
    }

    /// A render that misses the thresholds must exit non-zero unless
    /// `--allow-failure` is given — a QC gate that always exits 0 is useless
    /// in CI, which is the whole reason this family exists.
    #[test]
    fn failing_gate_exits_non_zero_unless_allowed() {
        assert!(finish_gate(false, false, "boom").is_err());
        assert!(finish_gate(false, true, "boom").is_ok());
        assert!(finish_gate(true, false, "boom").is_ok());
    }

    #[test]
    fn batch_rejects_an_out_of_range_pass_rate() {
        let args = BatchArgs {
            rendered_dir: std::env::temp_dir(),
            reference_dir: std::env::temp_dir(),
            thresholds: thresholds(25.0, 0.85),
            min_pass_rate: 1.5,
            per_image: false,
            allow_failure: false,
        };
        assert!(cmd_batch(args, &ctx()).is_err());
    }
}
