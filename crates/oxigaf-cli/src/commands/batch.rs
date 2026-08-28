//! `oxigaf batch` — dependency-aware batch conversion of many model files.
//!
//! Glue over [`crate::batch_processor`]. `batch plan` shows the execution
//! waves the scheduler derives from job dependencies; `batch run` executes
//! them through a [`crate::batch_processor::JobExecutor`] that performs one
//! real model conversion per job.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Args, Subcommand, ValueEnum};
use serde_json::json;

use crate::batch_processor::{
    jobs_from_directory, jobs_from_file_list, BatchConfig, BatchJob, BatchProcessor, BatchStats,
    JobExecutor, JobResult,
};
use crate::commands::{emit, CmdContext};

/// `oxigaf batch <command>`.
#[derive(Debug, Args)]
pub struct BatchArgs {
    #[command(subcommand)]
    pub command: BatchCommand,
}

/// Batch processing subcommands.
#[derive(Debug, Subcommand)]
pub enum BatchCommand {
    /// Show the dependency-ordered execution plan without running anything.
    Plan {
        #[command(flatten)]
        source: JobSource,
    },

    /// Execute every job in the batch.
    Run {
        #[command(flatten)]
        source: JobSource,

        /// Operation applied to each input file.
        #[arg(long, value_enum, default_value = "export-ply")]
        op: BatchOp,

        /// Maximum number of jobs executed concurrently within a wave.
        #[arg(long, default_value = "1")]
        max_parallel: usize,

        /// Abort the batch after the first failing job.
        #[arg(long)]
        fail_fast: bool,

        /// Retries attempted for a failing job before giving up.
        #[arg(long, default_value = "0")]
        max_retries: usize,

        /// Base backoff in milliseconds, doubled per retry attempt.
        #[arg(long, default_value = "100")]
        retry_backoff_ms: u64,
    },
}

/// Where the batch's jobs come from.
#[derive(Debug, Args, Clone)]
pub struct JobSource {
    /// Build one job per matching file in this directory.
    #[arg(long, conflicts_with = "files")]
    pub dir: Option<PathBuf>,

    /// File extension matched by `--dir` (without the leading dot).
    #[arg(long, default_value = "ply")]
    pub ext: String,

    /// Explicit list of input files (alternative to `--dir`).
    #[arg(long, num_args = 1.., conflicts_with = "dir")]
    pub files: Vec<PathBuf>,

    /// Directory that receives the produced files.
    #[arg(long)]
    pub output_dir: PathBuf,
}

/// Per-job operation.
#[derive(Debug, Clone, Copy, ValueEnum, Default)]
pub enum BatchOp {
    /// Load the model and re-export it as a 3DGS PLY file.
    #[default]
    ExportPly,
    /// Load the model and export it as a JSON checkpoint.
    ExportJson,
    /// Load the model and export it as safetensors.
    ExportSafetensors,
    /// Load the model and report its Gaussian count without writing anything.
    Validate,
}

impl BatchOp {
    /// File extension produced by this operation, or `None` when it writes
    /// no file.
    fn output_extension(self) -> Option<&'static str> {
        match self {
            Self::ExportPly => Some("ply"),
            Self::ExportJson => Some("json"),
            Self::ExportSafetensors => Some("safetensors"),
            Self::Validate => None,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::ExportPly => "export-ply",
            Self::ExportJson => "export-json",
            Self::ExportSafetensors => "export-safetensors",
            Self::Validate => "validate",
        }
    }
}

impl JobSource {
    fn build(&self) -> Result<Vec<BatchJob>> {
        let output_dir = self.output_dir.to_string_lossy().into_owned();
        if let Some(ref dir) = self.dir {
            let jobs = jobs_from_directory(&dir.to_string_lossy(), &self.ext, &output_dir)?;
            if jobs.is_empty() {
                anyhow::bail!("No *.{} files found in {}", self.ext, dir.display());
            }
            return Ok(jobs);
        }
        if self.files.is_empty() {
            anyhow::bail!("Specify either --dir <directory> or --files <path>...");
        }
        let paths: Vec<String> = self
            .files
            .iter()
            .map(|p| p.to_string_lossy().into_owned())
            .collect();
        Ok(jobs_from_file_list(&paths, &output_dir)?)
    }
}

/// Rewrite each job's output extension to match the requested operation.
fn retarget_outputs(jobs: &mut [BatchJob], op: BatchOp) {
    let Some(ext) = op.output_extension() else {
        return;
    };
    for job in jobs.iter_mut() {
        let path = PathBuf::from(&job.output_path).with_extension(ext);
        job.output_path = path.to_string_lossy().into_owned();
    }
}

/// Build the executor that performs one model conversion per job.
fn make_executor(op: BatchOp) -> JobExecutor {
    Box::new(
        move |job: &BatchJob| -> std::result::Result<usize, String> {
            let input = Path::new(&job.input_path);
            let model = crate::export::load_model(input).map_err(|e| format!("{e:#}"))?;
            let count = model.len();

            if op.output_extension().is_some() {
                let output = PathBuf::from(&job.output_path);
                if let Some(parent) = output.parent() {
                    if !parent.as_os_str().is_empty() {
                        std::fs::create_dir_all(parent)
                            .map_err(|e| format!("failed to create {}: {e}", parent.display()))?;
                    }
                }
                match op {
                    BatchOp::ExportPly => {
                        crate::export::export_ply(&model, &output).map_err(|e| format!("{e:#}"))?
                    }
                    BatchOp::ExportJson => crate::export::export_json_checkpoint(&model, &output)
                        .map_err(|e| format!("{e:#}"))?,
                    BatchOp::ExportSafetensors => {
                        crate::export::export_safetensors(&model, &output)
                            .map_err(|e| format!("{e:#}"))?
                    }
                    BatchOp::Validate => {}
                }
            }
            Ok(count)
        },
    )
}

/// Run the `batch` family.
///
/// # Errors
///
/// Returns an error when no jobs can be built, when the scheduler rejects
/// the job graph (cycles, duplicate ids), or when the worker pool cannot be
/// created.
pub fn run(args: BatchArgs, ctx: CmdContext) -> Result<()> {
    match args.command {
        BatchCommand::Plan { source } => {
            let jobs = source.build()?;
            let processor = BatchProcessor::from_jobs(jobs)?;
            let waves = processor.plan()?;
            let payload = json!({
                "job_count": processor.job_count(),
                "waves": waves,
            });
            emit(&ctx, "batch plan", payload, &[], || {
                println!(
                    "{} job(s) in {} wave(s)",
                    processor.job_count(),
                    waves.len()
                );
                for (index, wave) in waves.iter().enumerate() {
                    println!("  wave {index}:");
                    for id in wave {
                        match processor.get_job(*id) {
                            Some(job) => {
                                println!(
                                    "    [{}] {} -> {}",
                                    job.id, job.input_path, job.output_path
                                )
                            }
                            None => println!("    [{id}] <unknown job>"),
                        }
                    }
                }
            });
            Ok(())
        }
        BatchCommand::Run {
            source,
            op,
            max_parallel,
            fail_fast,
            max_retries,
            retry_backoff_ms,
        } => {
            if max_parallel == 0 {
                anyhow::bail!("--max-parallel must be at least 1");
            }
            let mut jobs = source.build()?;
            retarget_outputs(&mut jobs, op);

            let config = BatchConfig {
                max_parallel,
                fail_fast,
                max_retries,
                retry_backoff_ms,
                verbosity: usize::from(ctx.verbosity >= crate::verbosity::Verbosity::Verbose) + 1,
                // The global `--dry-run` flag maps directly onto the
                // scheduler's own dry-run mode, which reports every job as
                // skipped instead of invoking the executor.
                dry_run: ctx.dry_run,
            };

            if !ctx.dry_run {
                std::fs::create_dir_all(&source.output_dir).with_context(|| {
                    format!(
                        "Failed to create output directory: {}",
                        source.output_dir.display()
                    )
                })?;
            }

            let processor = BatchProcessor::new(jobs, config)?;
            let executor = make_executor(op);
            let (results, stats) = processor.execute(&executor)?;

            let payload = json!({
                "operation": op.label(),
                "dry_run": ctx.dry_run,
                "stats": stats_json(&stats),
                "jobs": results.iter().map(result_json).collect::<Vec<_>>(),
            });

            let failed = stats.failed_jobs;
            emit(&ctx, "batch run", payload, &[], || {
                println!("{}", stats.format_summary());
                for result in &results {
                    let status = if result.skipped {
                        "SKIP"
                    } else if result.success {
                        "OK  "
                    } else {
                        "FAIL"
                    };
                    match result.error_message {
                        Some(ref message) => println!(
                            "  {status} [{}] {} — {message}",
                            result.job_id, result.job_name
                        ),
                        None => println!(
                            "  {status} [{}] {} ({} Gaussians, {} ms)",
                            result.job_id,
                            result.job_name,
                            result.items_processed,
                            result.duration_ms
                        ),
                    }
                }
            });

            if failed > 0 {
                anyhow::bail!("{failed} of {} job(s) failed", stats.total_jobs);
            }
            Ok(())
        }
    }
}

fn stats_json(stats: &BatchStats) -> serde_json::Value {
    json!({
        "total_jobs": stats.total_jobs,
        "successful_jobs": stats.successful_jobs,
        "failed_jobs": stats.failed_jobs,
        "skipped_jobs": stats.skipped_jobs,
        "total_duration_ms": stats.total_duration_ms,
        "mean_job_duration_ms": stats.mean_job_duration_ms,
        "total_items_processed": stats.total_items_processed,
        "success_rate": stats.success_rate(),
    })
}

fn result_json(result: &JobResult) -> serde_json::Value {
    json!({
        "job_id": result.job_id,
        "job_name": result.job_name,
        "success": result.success,
        "skipped": result.skipped,
        "attempts": result.attempts,
        "duration_ms": result.duration_ms,
        "items_processed": result.items_processed,
        "error": result.error_message,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retarget_outputs_rewrites_extension_for_writing_ops() {
        let mut jobs = vec![BatchJob::new(0, "a", "in/a.ply", "out/a.ply")];
        retarget_outputs(&mut jobs, BatchOp::ExportJson);
        assert!(
            jobs[0].output_path.ends_with("a.json"),
            "got {}",
            jobs[0].output_path
        );
    }

    #[test]
    fn retarget_outputs_leaves_validate_alone() {
        let mut jobs = vec![BatchJob::new(0, "a", "in/a.ply", "out/a.ply")];
        retarget_outputs(&mut jobs, BatchOp::Validate);
        assert_eq!(jobs[0].output_path, "out/a.ply");
    }

    #[test]
    fn job_source_requires_a_source() {
        let source = JobSource {
            dir: None,
            ext: "ply".to_string(),
            files: Vec::new(),
            output_dir: std::env::temp_dir(),
        };
        assert!(source.build().is_err());
    }

    #[test]
    fn job_source_builds_from_explicit_files() {
        let out = std::env::temp_dir().join("oxigaf_batch_out");
        let source = JobSource {
            dir: None,
            ext: "ply".to_string(),
            files: vec![PathBuf::from("one.ply"), PathBuf::from("two.ply")],
            output_dir: out,
        };
        let jobs = source.build().expect("explicit file list builds jobs");
        assert_eq!(jobs.len(), 2);
        assert_eq!(jobs[0].id, 0);
        assert_eq!(jobs[1].id, 1);
    }
}
