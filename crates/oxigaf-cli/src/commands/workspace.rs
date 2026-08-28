//! `oxigaf workspace` — checkpoint discovery and run bookkeeping.
//!
//! Glue over [`crate::checkpoint_browser`]. The browser module deliberately
//! performs no directory I/O of its own, so the directory walk and the
//! `file_size_bytes` lookup live here and the parsed records are handed to
//! [`crate::checkpoint_browser::CheckpointBrowser`] for filtering, sorting
//! and formatting.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Args, Subcommand, ValueEnum};
use serde_json::json;

use crate::checkpoint_browser::{
    checkpoint_spacing_stats, compare_checkpoints, describe_checkpoint, estimate_steps_to_psnr,
    find_psnr_elbow, format_checkpoint_diff, format_checkpoint_table, format_spacing_stats,
    psnr_trend, BrowserCheckpoint, BrowserConfig, BrowserFilter, BrowserSort, CheckpointBrowser,
};
use crate::commands::{emit, CmdContext};

/// Default file extensions treated as checkpoints.
const DEFAULT_CHECKPOINT_EXTENSIONS: [&str; 3] = ["json", "safetensors", "ply"];

/// `oxigaf workspace <command>`.
#[derive(Debug, Args)]
pub struct WorkspaceArgs {
    #[command(subcommand)]
    pub command: WorkspaceCommand,
}

/// Workspace subcommands.
#[derive(Debug, Subcommand)]
pub enum WorkspaceCommand {
    /// List, filter and rank the checkpoints in a run directory.
    Checkpoints(CheckpointsArgs),

    /// Compare two checkpoint files field by field.
    CheckpointDiff {
        /// Baseline checkpoint path.
        before: PathBuf,
        /// Candidate checkpoint path.
        after: PathBuf,
    },
}

/// Sort order exposed on the command line.
#[derive(Debug, Clone, Copy, ValueEnum, Default)]
pub enum CheckpointSort {
    /// Ascending training step (oldest first).
    #[default]
    Step,
    /// Descending training step (newest first).
    StepDesc,
    /// Descending PSNR (best first).
    Psnr,
    /// Ascending loss (best first).
    Loss,
    /// Descending file size (largest first).
    FileSize,
    /// Descending composite quality score (best first).
    QualityScore,
}

impl From<CheckpointSort> for BrowserSort {
    fn from(value: CheckpointSort) -> Self {
        match value {
            CheckpointSort::Step => BrowserSort::ByStep,
            CheckpointSort::StepDesc => BrowserSort::ByStepDesc,
            CheckpointSort::Psnr => BrowserSort::ByPsnr,
            CheckpointSort::Loss => BrowserSort::ByLoss,
            CheckpointSort::FileSize => BrowserSort::ByFileSize,
            CheckpointSort::QualityScore => BrowserSort::ByQualityScore,
        }
    }
}

/// Arguments for `oxigaf workspace checkpoints`.
#[derive(Debug, Args)]
pub struct CheckpointsArgs {
    /// Run directory containing checkpoint files.
    pub dir: PathBuf,

    /// File extensions treated as checkpoints (without the leading dot).
    #[arg(long, num_args = 1..)]
    pub ext: Vec<String>,

    /// Sort order.
    #[arg(long, value_enum, default_value = "step")]
    pub sort: CheckpointSort,

    /// Maximum number of rows shown.
    #[arg(long, default_value = "20")]
    pub limit: usize,

    /// Only include checkpoints at or after this step.
    #[arg(long)]
    pub min_step: Option<usize>,

    /// Only include checkpoints at or before this step.
    #[arg(long)]
    pub max_step: Option<usize>,

    /// Only include checkpoints whose parsed PSNR is at least this value.
    #[arg(long)]
    pub min_psnr: Option<f32>,

    /// Only include checkpoints whose loss is at most this value.
    #[arg(long)]
    pub max_loss: Option<f32>,

    /// Tags every listed checkpoint must carry.
    #[arg(long = "tag", num_args = 1..)]
    pub tags_required: Vec<String>,

    /// Tags that exclude a checkpoint from the listing.
    #[arg(long = "exclude-tag", num_args = 1..)]
    pub tags_excluded: Vec<String>,

    /// Hide tags in the formatted table.
    #[arg(long)]
    pub no_tags: bool,

    /// Also report the PSNR trend, its elbow, and checkpoint spacing.
    #[arg(long)]
    pub trend: bool,

    /// With `--trend`, estimate how many further steps reach this PSNR.
    #[arg(long)]
    pub target_psnr: Option<f32>,
}

/// Run the `workspace` family.
///
/// # Errors
///
/// Returns an error when the run directory cannot be read or contains no
/// checkpoint files, or when a checkpoint path cannot be parsed.
pub fn run(args: WorkspaceArgs, ctx: CmdContext) -> Result<()> {
    match args.command {
        WorkspaceCommand::Checkpoints(list_args) => cmd_checkpoints(list_args, &ctx),
        WorkspaceCommand::CheckpointDiff { before, after } => {
            let a = checkpoint_from_file(&before);
            let b = checkpoint_from_file(&after);
            let diff = compare_checkpoints(&a, &b);
            let payload = json!({
                "before": describe_checkpoint(&a),
                "after": describe_checkpoint(&b),
                "step_delta": diff.step_delta,
                "psnr_delta": diff.psnr_delta,
                "loss_delta": diff.loss_delta,
                "gaussians_delta": diff.gaussians_delta,
                "size_delta": diff.size_delta,
                "tags_added": diff.tags_added,
                "tags_removed": diff.tags_removed,
            });
            emit(&ctx, "workspace checkpoint-diff", payload, &[], || {
                println!("{}", format_checkpoint_diff(&diff));
            });
            Ok(())
        }
    }
}

/// Build a [`BrowserCheckpoint`] for a single file, filling in the real
/// on-disk size the browser module cannot look up itself.
fn checkpoint_from_file(path: &Path) -> BrowserCheckpoint {
    let text = path.to_string_lossy();
    let mut checkpoint = BrowserCheckpoint::from_path(&text);
    if let Ok(meta) = std::fs::metadata(path) {
        checkpoint.file_size_bytes = usize::try_from(meta.len()).unwrap_or(usize::MAX);
    }
    checkpoint
}

fn collect_checkpoints(dir: &Path, extensions: &[String]) -> Result<Vec<BrowserCheckpoint>> {
    if !dir.is_dir() {
        anyhow::bail!("Not a directory: {}", dir.display());
    }
    let allowed: Vec<String> = if extensions.is_empty() {
        DEFAULT_CHECKPOINT_EXTENSIONS
            .iter()
            .map(|e| (*e).to_string())
            .collect()
    } else {
        extensions.iter().map(|e| e.to_ascii_lowercase()).collect()
    };

    let entries = std::fs::read_dir(dir)
        .with_context(|| format!("Failed to read directory: {}", dir.display()))?;

    let mut found = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let matches = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| allowed.iter().any(|a| a.eq_ignore_ascii_case(e)))
            .unwrap_or(false);
        if !matches {
            continue;
        }
        found.push(checkpoint_from_file(&path));
    }

    if found.is_empty() {
        anyhow::bail!(
            "No checkpoint files ({}) found in {}",
            allowed.join(", "),
            dir.display()
        );
    }
    found.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(found)
}

fn checkpoint_json(checkpoint: &BrowserCheckpoint) -> serde_json::Value {
    json!({
        "path": checkpoint.path,
        "step": checkpoint.step,
        "epoch": checkpoint.epoch,
        "psnr": checkpoint.psnr,
        "loss": checkpoint.loss,
        "n_gaussians": checkpoint.n_gaussians,
        "file_size_bytes": checkpoint.file_size_bytes,
        "tags": checkpoint.tags,
        "quality_score": checkpoint.quality_score(),
        "is_best": checkpoint.is_best(),
        "is_final": checkpoint.is_final(),
    })
}

fn cmd_checkpoints(args: CheckpointsArgs, ctx: &CmdContext) -> Result<()> {
    if args.limit == 0 {
        anyhow::bail!("--limit must be at least 1");
    }
    let all = collect_checkpoints(&args.dir, &args.ext)?;

    let config = BrowserConfig {
        sort_by: args.sort.into(),
        filter: BrowserFilter {
            min_step: args.min_step,
            max_step: args.max_step,
            min_psnr: args.min_psnr,
            max_loss: args.max_loss,
            tags_required: args.tags_required.clone(),
            tags_excluded: args.tags_excluded.clone(),
        },
        max_display: args.limit,
        show_tags: !args.no_tags,
    };

    let browser = CheckpointBrowser::new(all.clone(), config);
    let listed = browser.browse();
    let table = format_checkpoint_table(&listed);

    let mut payload = json!({
        "directory": args.dir.display().to_string(),
        "total_discovered": browser.len(),
        "listed": listed.iter().map(|c| checkpoint_json(c)).collect::<Vec<_>>(),
        "best": browser.find_best().map(checkpoint_json),
        "latest": browser.find_latest().map(checkpoint_json),
        "total_size_bytes": browser.total_size_bytes(),
        "step_range": browser.step_range().map(|(lo, hi)| json!([lo, hi])),
    });

    let spacing = checkpoint_spacing_stats(&all);
    let trend = psnr_trend(&all);
    let elbow = find_psnr_elbow(&all);
    let steps_to_target = args
        .target_psnr
        .and_then(|target| estimate_steps_to_psnr(&all, target));

    if args.trend {
        if let Some(map) = payload.as_object_mut() {
            map.insert(
                "spacing".to_string(),
                json!({
                    "mean_step_gap": spacing.mean_step_gap,
                    "min_step_gap": spacing.min_step_gap,
                    "max_step_gap": spacing.max_step_gap,
                    "is_regular": spacing.is_regular,
                    "total_steps": spacing.total_steps,
                }),
            );
            map.insert("psnr_trend".to_string(), json!(trend));
            map.insert("psnr_elbow_step".to_string(), json!(elbow));
            map.insert("steps_to_target_psnr".to_string(), json!(steps_to_target));
        }
    }

    emit(ctx, "workspace checkpoints", payload, &[], || {
        println!("{table}");
        if let Some(best) = browser.find_best() {
            println!("best  : {}", describe_checkpoint(best));
        }
        if let Some(latest) = browser.find_latest() {
            println!("latest: {}", describe_checkpoint(latest));
        }
        if args.trend {
            println!();
            println!("{}", format_spacing_stats(&spacing));
            match elbow {
                Some(step) => println!("PSNR elbow at step {step}"),
                None => println!("PSNR elbow: not detectable from {} point(s)", trend.len()),
            }
            if let Some(target) = args.target_psnr {
                match steps_to_target {
                    Some(steps) => {
                        println!("~{steps} further step(s) to reach {target:.2} dB")
                    }
                    None => println!(
                        "Cannot extrapolate {target:.2} dB: PSNR trend is flat or declining"
                    ),
                }
            }
        }
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::verbosity::Verbosity;

    #[test]
    fn sort_flag_maps_onto_browser_sort() {
        assert_eq!(BrowserSort::from(CheckpointSort::Psnr), BrowserSort::ByPsnr);
        assert_eq!(
            BrowserSort::from(CheckpointSort::StepDesc),
            BrowserSort::ByStepDesc
        );
    }

    #[test]
    fn collect_checkpoints_reads_sizes_and_rejects_empty_dirs() {
        let dir = std::env::temp_dir().join("oxigaf_workspace_ckpts");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("temp dir");

        // A directory with no matching files must be an error, not an empty list.
        assert!(collect_checkpoints(&dir, &[]).is_err());

        let file = dir.join("ckpt_step_1200_psnr_31.5.json");
        std::fs::write(&file, b"{\"step\":1200}").expect("write checkpoint");
        let found = collect_checkpoints(&dir, &[]).expect("one checkpoint");
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].step, 1200);
        assert!(
            found[0].file_size_bytes > 0,
            "file size must come from disk, not the parser"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn checkpoints_command_rejects_zero_limit() {
        let ctx = CmdContext::new(Verbosity::Quiet, true, false);
        let args = CheckpointsArgs {
            dir: std::env::temp_dir(),
            ext: Vec::new(),
            sort: CheckpointSort::Step,
            limit: 0,
            min_step: None,
            max_step: None,
            min_psnr: None,
            max_loss: None,
            tags_required: Vec::new(),
            tags_excluded: Vec::new(),
            no_tags: false,
            trend: false,
            target_psnr: None,
        };
        assert!(cmd_checkpoints(args, &ctx).is_err());
    }
}
