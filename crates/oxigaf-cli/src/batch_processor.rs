//! Batch processing utilities for running operations across multiple scene files.
//!
//! This module provides tools to configure, plan, and execute a batch of jobs
//! over PLY/JSON Gaussian scene files. Features include:
//! - Dependency ordering via topological sort (Kahn's algorithm)
//! - Priority-based scheduling within execution waves
//! - Configurable parallelism, retries, fail-fast, and dry-run modes
//! - Progress-friendly result aggregation with summary statistics
//!
//! # Example
//! ```rust
//! use oxigaf_cli::batch_processor::{
//!     BatchJob, BatchConfig, BatchProcessor, JobExecutor,
//! };
//!
//! let jobs = vec![
//!     BatchJob::new(0, "scene-a", "input/a.ply", "output/a.ply"),
//!     BatchJob::new(1, "scene-b", "input/b.ply", "output/b.ply")
//!         .with_dependency(0),
//! ];
//! let processor = BatchProcessor::new(jobs, BatchConfig::default())
//!     .expect("failed to create processor");
//! let executor: JobExecutor = Box::new(|job| {
//!     println!("Processing {}", job.name);
//!     Ok(1000)
//! });
//! let (results, stats) = processor.execute(&executor).expect("batch failed");
//! println!("{}", stats.format_summary());
//! ```

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::Instant;

use rayon::prelude::*;
use thiserror::Error;

// ---------------------------------------------------------------------------
// BatchError
// ---------------------------------------------------------------------------

/// Errors that can occur during batch processing operations.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum BatchError {
    /// The batch configuration is invalid.
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// No jobs in the batch when execution was requested.
    #[error("Batch is empty — no jobs to process")]
    EmptyBatch,

    /// A specific job failed during execution.
    #[error("Job {job_id} failed: {reason}")]
    JobFailed { job_id: usize, reason: String },

    /// A cycle was detected in the job dependency graph.
    #[error("Dependency cycle detected — jobs cannot be ordered")]
    DependencyCycle,

    /// An input file path is missing or empty.
    #[error("File not found: {0}")]
    FileNotFound(String),

    /// A job specification is invalid (e.g., empty name or path).
    #[error("Invalid job spec: {0}")]
    InvalidJobSpec(String),
}

// ---------------------------------------------------------------------------
// BatchJob
// ---------------------------------------------------------------------------

/// Specification for a single job within a batch.
///
/// A job describes one file-level operation: the input scene, the desired
/// output path, any key-value parameters, dependency relationships with
/// other jobs in the same batch, and a scheduling priority.
#[derive(Debug, Clone)]
pub struct BatchJob {
    /// Unique job ID within this batch.
    pub id: usize,
    /// Human-readable job name.
    pub name: String,
    /// Input file path (relative or absolute).
    pub input_path: String,
    /// Output file path.
    pub output_path: String,
    /// Job-specific parameters as key-value pairs.
    pub params: Vec<(String, String)>,
    /// IDs of jobs that must complete before this one.
    pub depends_on: Vec<usize>,
    /// Priority (higher = run first within a wave).
    pub priority: i32,
}

impl BatchJob {
    /// Create a new `BatchJob` with the given ID, name, input, and output paths.
    ///
    /// Defaults: `depends_on = []`, `params = []`, `priority = 0`.
    pub fn new(
        id: usize,
        name: impl Into<String>,
        input: impl Into<String>,
        output: impl Into<String>,
    ) -> Self {
        Self {
            id,
            name: name.into(),
            input_path: input.into(),
            output_path: output.into(),
            params: Vec::new(),
            depends_on: Vec::new(),
            priority: 0,
        }
    }

    /// Add a key-value parameter (builder-style).
    pub fn with_param(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.params.push((key.into(), value.into()));
        self
    }

    /// Set the scheduling priority (builder-style).
    ///
    /// Jobs with higher priority values are scheduled first within the same wave.
    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    /// Add a dependency on another job by ID (builder-style).
    ///
    /// This job will not start until the job with `dep_id` has completed.
    pub fn with_dependency(mut self, dep_id: usize) -> Self {
        self.depends_on.push(dep_id);
        self
    }

    /// Look up a parameter value by key.
    ///
    /// Returns the first matching value, or `None` if the key is absent.
    pub fn get_param(&self, key: &str) -> Option<&str> {
        self.params
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }

    /// Validate that the job specification is complete.
    ///
    /// # Errors
    ///
    /// Returns [`BatchError::InvalidJobSpec`] if the name, input path, or output
    /// path is empty.
    pub fn validate(&self) -> Result<(), BatchError> {
        if self.name.is_empty() {
            return Err(BatchError::InvalidJobSpec(format!(
                "job {} has an empty name",
                self.id
            )));
        }
        if self.input_path.is_empty() {
            return Err(BatchError::InvalidJobSpec(format!(
                "job {} '{}' has an empty input_path",
                self.id, self.name
            )));
        }
        if self.output_path.is_empty() {
            return Err(BatchError::InvalidJobSpec(format!(
                "job {} '{}' has an empty output_path",
                self.id, self.name
            )));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// JobResult
// ---------------------------------------------------------------------------

/// Result record for a single completed (or failed) job.
#[derive(Debug, Clone)]
pub struct JobResult {
    /// The ID of the job.
    pub job_id: usize,
    /// The human-readable name of the job.
    pub job_name: String,
    /// Whether the job completed successfully.
    pub success: bool,
    /// Error message if the job failed or was skipped.
    pub error_message: Option<String>,
    /// Wall-clock duration in milliseconds.
    pub duration_ms: u64,
    /// Number of items processed (e.g., number of Gaussians).
    pub items_processed: usize,
    /// Number of execution attempts made (1 = succeeded or failed on the
    /// first try; >1 indicates retries occurred; 0 for a job that was never
    /// executed, e.g. `skipped`).
    pub attempts: usize,
    /// `true` when this job was not executed at all — a dry run, an unmet
    /// dependency, or `fail_fast` having aborted the batch — as opposed to
    /// `success == false` from a genuine execution failure.
    pub skipped: bool,
}

impl JobResult {
    /// Construct a successful job result.
    pub fn success(job_id: usize, name: impl Into<String>, duration_ms: u64, items: usize) -> Self {
        Self {
            job_id,
            job_name: name.into(),
            success: true,
            error_message: None,
            duration_ms,
            items_processed: items,
            attempts: 1,
            skipped: false,
        }
    }

    /// Construct a failed job result (the job ran but did not succeed).
    pub fn failure(
        job_id: usize,
        name: impl Into<String>,
        duration_ms: u64,
        reason: impl Into<String>,
    ) -> Self {
        Self {
            job_id,
            job_name: name.into(),
            success: false,
            error_message: Some(reason.into()),
            duration_ms,
            items_processed: 0,
            attempts: 1,
            skipped: false,
        }
    }

    /// Construct a skipped job result: the job was never executed (dry run,
    /// an unmet dependency, or `fail_fast` aborting the batch). `success` is
    /// `false` since it did not run, but `skipped` distinguishes this from a
    /// genuine execution failure.
    pub fn skipped(job_id: usize, name: impl Into<String>, reason: impl Into<String>) -> Self {
        Self {
            job_id,
            job_name: name.into(),
            success: false,
            error_message: Some(reason.into()),
            duration_ms: 0,
            items_processed: 0,
            attempts: 0,
            skipped: true,
        }
    }

    /// Override the number of execution attempts (builder-style).
    #[must_use]
    pub fn with_attempts(mut self, attempts: usize) -> Self {
        self.attempts = attempts;
        self
    }
}

// ---------------------------------------------------------------------------
// compute_execution_waves
// ---------------------------------------------------------------------------

/// Compute job execution waves using topological sort (Kahn's algorithm).
///
/// Returns a `Vec` of waves; each wave is a `Vec` of job IDs that have all
/// their dependencies satisfied by previous waves, and can therefore run in
/// parallel.  Jobs with higher [`BatchJob::priority`] values are placed first
/// within each wave.
///
/// # Errors
///
/// - Returns `Ok(vec![])` when `jobs` is empty.
/// - Returns [`BatchError::InvalidJobSpec`] if `jobs` contains duplicate IDs
///   (this algorithm relies on IDs being unique keys), or if a dependency
///   references an unknown job ID.
/// - Returns [`BatchError::DependencyCycle`] if the dependency graph contains
///   a cycle.
pub fn compute_execution_waves(jobs: &[BatchJob]) -> Result<Vec<Vec<usize>>, BatchError> {
    if jobs.is_empty() {
        return Ok(vec![]);
    }

    // Build a lookup: id → index in `jobs`, rejecting duplicate ids (the
    // rest of this algorithm treats ids as unique map keys).
    let mut id_to_idx: HashMap<usize, usize> = HashMap::with_capacity(jobs.len());
    for (i, job) in jobs.iter().enumerate() {
        if id_to_idx.insert(job.id, i).is_some() {
            return Err(BatchError::InvalidJobSpec(format!(
                "duplicate job id {}: job ids must be unique",
                job.id
            )));
        }
    }

    // Validate that every depends_on entry references a known job ID.
    for job in jobs {
        for &dep in &job.depends_on {
            if !id_to_idx.contains_key(&dep) {
                return Err(BatchError::InvalidJobSpec(format!(
                    "job {} '{}' depends on unknown job id {}",
                    job.id, job.name, dep
                )));
            }
        }
    }

    // in_degree[id] = number of unsatisfied dependencies
    let mut in_degree: HashMap<usize, usize> = jobs.iter().map(|j| (j.id, 0)).collect();
    // successors[id] = list of job IDs that directly depend on `id`
    let mut successors: HashMap<usize, Vec<usize>> =
        jobs.iter().map(|j| (j.id, Vec::new())).collect();

    for job in jobs {
        for &dep in &job.depends_on {
            match in_degree.get_mut(&job.id) {
                Some(d) => *d += 1,
                None => {
                    // Unreachable given the duplicate-id check above (every
                    // job.id was seeded as a key), but handled explicitly
                    // rather than silently dropping the update.
                    return Err(BatchError::InvalidJobSpec(format!(
                        "internal error: job {} missing from in-degree map",
                        job.id
                    )));
                }
            }
            successors.entry(dep).or_default().push(job.id);
        }
    }

    // Seed: all jobs with no incoming edges
    let mut frontier: Vec<usize> = in_degree
        .iter()
        .filter(|(_, &deg)| deg == 0)
        .map(|(&id, _)| id)
        .collect();

    let mut waves: Vec<Vec<usize>> = Vec::new();
    let mut visited = 0usize;

    while !frontier.is_empty() {
        // Sort descending by priority so higher-priority jobs come first.
        frontier.sort_by(|&a, &b| {
            let pa = id_to_idx
                .get(&a)
                .and_then(|&idx| jobs.get(idx))
                .map_or(0, |j| j.priority);
            let pb = id_to_idx
                .get(&b)
                .and_then(|&idx| jobs.get(idx))
                .map_or(0, |j| j.priority);
            pb.cmp(&pa).then(a.cmp(&b)) // stable tie-break by id
        });

        let wave = frontier.clone();
        visited += wave.len();

        // Collect the next frontier from successors of current wave.
        let mut next: Vec<usize> = Vec::new();
        for &id in &wave {
            for &succ in successors.get(&id).map(Vec::as_slice).unwrap_or(&[]) {
                if let Some(deg) = in_degree.get_mut(&succ) {
                    *deg = deg.saturating_sub(1);
                    if *deg == 0 {
                        next.push(succ);
                    }
                }
            }
        }

        waves.push(wave);
        frontier = next;
    }

    if visited != jobs.len() {
        return Err(BatchError::DependencyCycle);
    }

    Ok(waves)
}

// ---------------------------------------------------------------------------
// BatchConfig
// ---------------------------------------------------------------------------

/// Configuration controlling how the batch is executed.
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Maximum number of parallel jobs within a wave.
    ///
    /// Default: `1` (sequential execution).
    pub max_parallel: usize,
    /// Stop on first failure when `true`; continue with other jobs when `false`.
    ///
    /// Default: `false`.
    pub fail_fast: bool,
    /// Maximum number of retries for a failed job.
    ///
    /// Default: `0` (no retries).
    pub max_retries: usize,
    /// Base delay (ms) before retrying a failed job, doubled per subsequent
    /// attempt (exponential backoff: `retry_backoff_ms * 2^attempt`). `0`
    /// disables the delay while still retrying immediately.
    ///
    /// Default: `100`.
    pub retry_backoff_ms: u64,
    /// Log verbosity level: `0` = quiet, `1` = normal, `2` = verbose.
    ///
    /// Default: `1`.
    pub verbosity: usize,
    /// Dry run: plan what would be done but do not execute.
    ///
    /// Default: `false`.
    pub dry_run: bool,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_parallel: 1,
            fail_fast: false,
            max_retries: 0,
            retry_backoff_ms: 100,
            verbosity: 1,
            dry_run: false,
        }
    }
}

impl BatchConfig {
    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// Returns [`BatchError::InvalidConfig`] when `max_parallel` is `0`.
    pub fn validate(&self) -> Result<(), BatchError> {
        if self.max_parallel == 0 {
            return Err(BatchError::InvalidConfig(
                "max_parallel must be at least 1".to_string(),
            ));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// JobExecutor type alias
// ---------------------------------------------------------------------------

/// A callback that executes a single job.
///
/// Receives a reference to the [`BatchJob`] and returns the number of items
/// processed on success, or a human-readable error string on failure.
pub type JobExecutor = Box<dyn Fn(&BatchJob) -> Result<usize, String> + Send + Sync>;

// ---------------------------------------------------------------------------
// BatchStats
// ---------------------------------------------------------------------------

/// Aggregate statistics for a completed batch run.
#[derive(Debug, Clone, Default)]
pub struct BatchStats {
    /// Total number of jobs in the batch.
    pub total_jobs: usize,
    /// Number of jobs that succeeded.
    pub successful_jobs: usize,
    /// Number of jobs that failed (after all retries).
    pub failed_jobs: usize,
    /// Number of jobs skipped due to `fail_fast` or unmet dependencies.
    pub skipped_jobs: usize,
    /// Combined wall-clock time of all executed jobs in milliseconds.
    pub total_duration_ms: u64,
    /// Mean wall-clock time per executed job in milliseconds.
    pub mean_job_duration_ms: f64,
    /// Combined number of items processed across all successful jobs.
    pub total_items_processed: usize,
}

impl BatchStats {
    /// Fraction of jobs that succeeded, in `[0.0, 1.0]`.
    ///
    /// Returns `0.0` when there are no jobs.
    pub fn success_rate(&self) -> f32 {
        self.successful_jobs as f32 / self.total_jobs.max(1) as f32
    }

    /// Format a one-line human-readable summary of the batch run.
    pub fn format_summary(&self) -> String {
        let total_secs = self.total_duration_ms as f64 / 1000.0;
        format!(
            "Processed {} jobs: {} success, {} failed, {} skipped. Total time: {:.2}s",
            self.total_jobs, self.successful_jobs, self.failed_jobs, self.skipped_jobs, total_secs,
        )
    }
}

// ---------------------------------------------------------------------------
// BatchProcessor
// ---------------------------------------------------------------------------

/// Orchestrator for a collection of [`BatchJob`]s.
///
/// Validates jobs and configuration on construction, computes a dependency-
/// ordered execution plan, and runs each job through a caller-supplied
/// [`JobExecutor`] callback.
pub struct BatchProcessor {
    jobs: Vec<BatchJob>,
    config: BatchConfig,
}

impl BatchProcessor {
    /// Create a new `BatchProcessor`, validating every job and the config.
    ///
    /// An empty job list is accepted here; [`execute`](Self::execute) will
    /// return [`BatchError::EmptyBatch`] if there is nothing to run.
    ///
    /// # Errors
    ///
    /// Returns [`BatchError::InvalidJobSpec`] if any job fails validation or
    /// if `jobs` contains duplicate IDs, or [`BatchError::InvalidConfig`] if
    /// the config is invalid.
    pub fn new(jobs: Vec<BatchJob>, config: BatchConfig) -> Result<Self, BatchError> {
        config.validate()?;
        for job in &jobs {
            job.validate()?;
        }
        let mut seen_ids: HashSet<usize> = HashSet::with_capacity(jobs.len());
        for job in &jobs {
            if !seen_ids.insert(job.id) {
                return Err(BatchError::InvalidJobSpec(format!(
                    "duplicate job id {}: job ids must be unique within a batch",
                    job.id
                )));
            }
        }
        Ok(Self { jobs, config })
    }

    /// Create a new `BatchProcessor` with the default [`BatchConfig`].
    ///
    /// # Errors
    ///
    /// Returns [`BatchError::InvalidJobSpec`] if any job fails validation.
    pub fn from_jobs(jobs: Vec<BatchJob>) -> Result<Self, BatchError> {
        Self::new(jobs, BatchConfig::default())
    }

    /// Add a job to the processor after construction.
    ///
    /// # Errors
    ///
    /// Returns [`BatchError::InvalidJobSpec`] if the job fails validation or
    /// its ID duplicates one already in this batch.
    pub fn add_job(&mut self, job: BatchJob) -> Result<(), BatchError> {
        job.validate()?;
        if self.jobs.iter().any(|j| j.id == job.id) {
            return Err(BatchError::InvalidJobSpec(format!(
                "duplicate job id {}: job ids must be unique within a batch",
                job.id
            )));
        }
        self.jobs.push(job);
        Ok(())
    }

    /// Return the number of jobs registered with this processor.
    pub fn job_count(&self) -> usize {
        self.jobs.len()
    }

    /// Compute the execution waves without running any jobs.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`compute_execution_waves`].
    pub fn plan(&self) -> Result<Vec<Vec<usize>>, BatchError> {
        compute_execution_waves(&self.jobs)
    }

    /// Perform a dry-run: return the planned execution order without running.
    ///
    /// Equivalent to [`plan`](Self::plan).
    pub fn dry_run(&self) -> Result<Vec<Vec<usize>>, BatchError> {
        self.plan()
    }

    /// Look up a job by its ID.
    pub fn get_job(&self, id: usize) -> Option<&BatchJob> {
        self.jobs.iter().find(|j| j.id == id)
    }

    /// Return all job IDs in topological (wave-flattened) order.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`plan`](Self::plan).
    pub fn execution_order(&self) -> Result<Vec<usize>, BatchError> {
        let waves = self.plan()?;
        Ok(waves.into_iter().flatten().collect())
    }

    /// Execute all jobs using the provided executor callback.
    ///
    /// # Execution model
    ///
    /// Jobs are run wave-by-wave in topological order. Within each wave —
    /// all of whose jobs are mutually independent by construction, see
    /// [`compute_execution_waves`] — eligible jobs are dispatched to a
    /// dedicated worker pool in chunks of at most `config.max_parallel`
    /// jobs running concurrently, each with its own `config.max_retries`
    /// retries (exponential backoff via `config.retry_backoff_ms`). Jobs
    /// with an unmet dependency are skipped before any dispatch. `abort`
    /// (from `fail_fast`) is rechecked between chunks, so at most one
    /// chunk's worth of jobs can start after it triggers — with the
    /// default `max_parallel == 1`, each chunk is a single job, exactly
    /// reproducing sequential check-before-every-job `fail_fast` semantics.
    /// `all_results` therefore groups each chunk's skipped-by-dependency or
    /// skipped-by-abort entries with its executed ones in wave/chunk order,
    /// rather than strict original priority order when both occur.
    ///
    /// When `config.dry_run` is `true`, the method returns one
    /// [`JobResult::skipped`] entry per job (`stats.skipped_jobs` equal to
    /// the total job count, all other stats zeroed) instead of invoking the
    /// executor.
    ///
    /// # Errors
    ///
    /// - [`BatchError::EmptyBatch`] when there are no jobs.
    /// - Any planning error from [`plan`](Self::plan).
    /// - [`BatchError::InvalidConfig`] if the worker pool cannot be built.
    pub fn execute(
        &self,
        executor: &JobExecutor,
    ) -> Result<(Vec<JobResult>, BatchStats), BatchError> {
        if self.jobs.is_empty() {
            return Err(BatchError::EmptyBatch);
        }

        // Build an id → job reference map for quick lookup.
        let job_map: HashMap<usize, &BatchJob> = self.jobs.iter().map(|j| (j.id, j)).collect();

        let waves = self.plan()?;

        // For a dry-run, return synthetic "skipped" results for every job.
        if self.config.dry_run {
            let results: Vec<JobResult> = self
                .jobs
                .iter()
                .map(|j| JobResult::skipped(j.id, j.name.clone(), "dry-run: not executed"))
                .collect();
            let stats = BatchStats {
                total_jobs: self.jobs.len(),
                skipped_jobs: self.jobs.len(),
                ..Default::default()
            };
            return Ok((results, stats));
        }

        // Track which jobs have failed (so dependents can be skipped).
        let mut failed_ids: HashSet<usize> = HashSet::new();
        // We also track whether fail_fast has been triggered.
        let mut abort = false;

        let mut all_results: Vec<JobResult> = Vec::with_capacity(self.jobs.len());
        let mut stats = BatchStats {
            total_jobs: self.jobs.len(),
            ..Default::default()
        };

        // Build a per-job set of dependency IDs for quick lookup.
        let dep_map: HashMap<usize, &[usize]> = self
            .jobs
            .iter()
            .map(|j| (j.id, j.depends_on.as_slice()))
            .collect();

        let max_retries = self.config.max_retries;
        let retry_backoff_ms = self.config.retry_backoff_ms;

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(self.config.max_parallel)
            .build()
            .map_err(|e| BatchError::InvalidConfig(format!("failed to build worker pool: {e}")))?;

        for wave in &waves {
            // Resolve skips up front (sequential, cheap): everything left
            // in `to_run` is independent within this wave and safe to
            // dispatch to the worker pool below.
            let mut to_run: Vec<usize> = Vec::with_capacity(wave.len());

            for &job_id in wave {
                let job = match job_map.get(&job_id) {
                    Some(j) => *j,
                    None => continue,
                };

                let unmet_dep = dep_map
                    .get(&job_id)
                    .map(|deps| deps.iter().any(|d| failed_ids.contains(d)))
                    .unwrap_or(false);

                if abort || unmet_dep {
                    all_results.push(JobResult::skipped(
                        job_id,
                        job.name.clone(),
                        if abort {
                            "skipped: fail_fast triggered"
                        } else {
                            "skipped: dependency failed"
                        },
                    ));
                    stats.skipped_jobs += 1;
                } else {
                    to_run.push(job_id);
                }
            }

            // Dispatch `to_run` in chunks of at most `max_parallel` jobs,
            // rechecking `abort` between chunks. This bounds how much extra
            // work can start after a fail_fast trigger to at most one
            // chunk's worth (rather than the whole wave), and — critically
            // — with the default `max_parallel == 1` each chunk holds
            // exactly one job, exactly reproducing sequential
            // check-before-every-job fail_fast semantics.
            for chunk in to_run.chunks(self.config.max_parallel.max(1)) {
                if abort {
                    for &job_id in chunk {
                        let name = job_map
                            .get(&job_id)
                            .map(|j| j.name.clone())
                            .unwrap_or_else(|| format!("job-{job_id}"));
                        all_results.push(JobResult::skipped(
                            job_id,
                            name,
                            "skipped: fail_fast triggered",
                        ));
                        stats.skipped_jobs += 1;
                    }
                    continue;
                }

                let chunk_results: Vec<(usize, JobResult, bool)> = pool.install(|| {
                    chunk
                        .par_iter()
                        .map(|&job_id| {
                            let job = match job_map.get(&job_id) {
                                Some(j) => *j,
                                None => {
                                    return (
                                        job_id,
                                        JobResult::failure(
                                            job_id,
                                            format!("job-{job_id}"),
                                            0,
                                            "internal error: job not found in job map",
                                        ),
                                        false,
                                    );
                                }
                            };

                            let mut attempt_result: Result<usize, String> = Err(String::new());
                            let mut elapsed_ms = 0u64;
                            let mut attempts_made = 0usize;

                            for attempt in 0..=max_retries {
                                let t0 = Instant::now();
                                attempt_result = executor(job);
                                elapsed_ms = t0.elapsed().as_millis() as u64;
                                attempts_made += 1;

                                if attempt_result.is_ok() {
                                    break;
                                }
                                if attempt < max_retries && retry_backoff_ms > 0 {
                                    let backoff_ms =
                                        retry_backoff_ms.saturating_mul(1u64 << attempt.min(20));
                                    std::thread::sleep(std::time::Duration::from_millis(
                                        backoff_ms,
                                    ));
                                }
                            }

                            match attempt_result {
                                Ok(items) => (
                                    job_id,
                                    JobResult::success(job_id, job.name.clone(), elapsed_ms, items)
                                        .with_attempts(attempts_made),
                                    true,
                                ),
                                Err(reason) => (
                                    job_id,
                                    JobResult::failure(
                                        job_id,
                                        job.name.clone(),
                                        elapsed_ms,
                                        reason,
                                    )
                                    .with_attempts(attempts_made),
                                    false,
                                ),
                            }
                        })
                        .collect::<Vec<(usize, JobResult, bool)>>()
                });

                for (job_id, result, succeeded) in chunk_results {
                    if succeeded {
                        stats.successful_jobs += 1;
                        stats.total_duration_ms += result.duration_ms;
                        stats.total_items_processed += result.items_processed;
                    } else {
                        stats.failed_jobs += 1;
                        stats.total_duration_ms += result.duration_ms;
                        failed_ids.insert(job_id);
                        if self.config.fail_fast {
                            abort = true;
                        }
                    }
                    all_results.push(result);
                }
            }
        }

        // Compute mean only over executed (non-skipped) jobs.
        let executed = (stats.successful_jobs + stats.failed_jobs) as f64;
        stats.mean_job_duration_ms = if executed > 0.0 {
            stats.total_duration_ms as f64 / executed
        } else {
            0.0
        };

        Ok((all_results, stats))
    }
}

// ---------------------------------------------------------------------------
// Bulk construction helpers
// ---------------------------------------------------------------------------

/// Build a batch from a directory scan: one job per file matching an extension.
///
/// Scans `dir` (non-recursively) for regular files whose extension matches
/// `extension` (case-insensitively), sorted by file name for a
/// deterministic, reproducible job order, and constructs one [`BatchJob`]
/// per match with sequential IDs starting at `0`.
///
/// # Arguments
///
/// * `dir`        – Source directory path (must be non-empty and exist).
/// * `extension`  – File extension to match without a leading dot (e.g., `"ply"`).
/// * `output_dir` – Directory where output files should be written (must be non-empty).
///
/// # Errors
///
/// - [`BatchError::InvalidJobSpec`] if `dir`, `extension`, or `output_dir`
///   is an empty string.
/// - [`BatchError::FileNotFound`] if `dir` cannot be read (does not exist,
///   is not a directory, or is inaccessible).
pub fn jobs_from_directory(
    dir: &str,
    extension: &str,
    output_dir: &str,
) -> Result<Vec<BatchJob>, BatchError> {
    if dir.is_empty() {
        return Err(BatchError::InvalidJobSpec(
            "source directory must not be empty".to_string(),
        ));
    }
    if extension.is_empty() {
        return Err(BatchError::InvalidJobSpec(
            "extension must not be empty".to_string(),
        ));
    }
    if output_dir.is_empty() {
        return Err(BatchError::InvalidJobSpec(
            "output directory must not be empty".to_string(),
        ));
    }

    let dir_path = Path::new(dir);
    let entries =
        std::fs::read_dir(dir_path).map_err(|_| BatchError::FileNotFound(dir.to_string()))?;

    let mut matches: Vec<(String, PathBuf)> = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let ext_matches = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.eq_ignore_ascii_case(extension))
            .unwrap_or(false);
        if !ext_matches {
            continue;
        }
        if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
            matches.push((name.to_string(), path));
        }
    }
    matches.sort_by(|a, b| a.0.cmp(&b.0));

    let output_root = Path::new(output_dir);
    let jobs = matches
        .into_iter()
        .enumerate()
        .map(|(idx, (name, path))| {
            let input_path = path.to_string_lossy().to_string();
            let output_path = output_root.join(&name).to_string_lossy().to_string();
            BatchJob::new(idx, name, input_path, output_path)
        })
        .collect();

    Ok(jobs)
}

/// Build a batch from an explicit list of input file paths.
///
/// Creates one [`BatchJob`] per path with sequential IDs starting at `0`.  The
/// job name is derived from the file name component of the path.
///
/// # Arguments
///
/// * `input_paths` – Slice of input file path strings.  Each must be non-empty.
/// * `output_dir`  – Directory where output files will be written.
///
/// # Errors
///
/// - [`BatchError::EmptyBatch`] when `input_paths` is empty.
/// - [`BatchError::FileNotFound`] when any path string is empty.
pub fn jobs_from_file_list(
    input_paths: &[String],
    output_dir: &str,
) -> Result<Vec<BatchJob>, BatchError> {
    if input_paths.is_empty() {
        return Err(BatchError::EmptyBatch);
    }

    let mut jobs = Vec::with_capacity(input_paths.len());
    for (idx, path) in input_paths.iter().enumerate() {
        if path.is_empty() {
            return Err(BatchError::FileNotFound(String::new()));
        }
        // Derive a short name from the file-name component (after last '/').
        let filename = path.rsplit('/').next().unwrap_or(path.as_str());
        let output_path = format!("{}/{}", output_dir, filename);
        jobs.push(BatchJob::new(idx, filename, path.clone(), output_path));
    }
    Ok(jobs)
}

/// Merge two [`BatchProcessor`] instances into a single one.
///
/// Job IDs from `b` are offset by `a`'s highest existing ID plus one (not
/// `a.job_count()`, which collides whenever `a`'s IDs are not contiguous
/// from `0` — e.g. `a` holding ids `[0, 2]` has `job_count() == 2`, which
/// would collide with `a`'s own job `2`). Dependency IDs within `b` are
/// offset by the same amount. The resulting processor uses the default
/// [`BatchConfig`] and re-validates all IDs are unique (see
/// [`BatchProcessor::new`]).
///
/// # Errors
///
/// Propagates any validation error from constructing the merged processor.
pub fn merge_batches(a: BatchProcessor, b: BatchProcessor) -> Result<BatchProcessor, BatchError> {
    let offset = a.jobs.iter().map(|j| j.id).max().map_or(0, |m| m + 1);
    let mut merged_jobs: Vec<BatchJob> = a.jobs;

    for job in b.jobs {
        let new_deps: Vec<usize> = job.depends_on.iter().map(|&d| d + offset).collect();
        let new_job = BatchJob {
            id: job.id + offset,
            name: job.name,
            input_path: job.input_path,
            output_path: job.output_path,
            params: job.params,
            depends_on: new_deps,
            priority: job.priority,
        };
        merged_jobs.push(new_job);
    }

    BatchProcessor::new(merged_jobs, BatchConfig::default())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // BatchJob construction and accessors
    // -----------------------------------------------------------------------

    #[test]
    fn test_batch_job_new_basic_fields() {
        let job = BatchJob::new(7, "my-job", "in/scene.ply", "out/scene.ply");
        assert_eq!(job.id, 7);
        assert_eq!(job.name, "my-job");
        assert_eq!(job.input_path, "in/scene.ply");
        assert_eq!(job.output_path, "out/scene.ply");
        assert!(job.params.is_empty());
        assert!(job.depends_on.is_empty());
        assert_eq!(job.priority, 0);
    }

    #[test]
    fn test_batch_job_with_param() {
        let job = BatchJob::new(0, "j", "a", "b")
            .with_param("scale", "1.0")
            .with_param("mode", "fast");
        assert_eq!(job.params.len(), 2);
        assert_eq!(job.params[0], ("scale".to_string(), "1.0".to_string()));
        assert_eq!(job.params[1], ("mode".to_string(), "fast".to_string()));
    }

    #[test]
    fn test_batch_job_get_param_found() {
        let job = BatchJob::new(0, "j", "a", "b").with_param("key", "value");
        assert_eq!(job.get_param("key"), Some("value"));
    }

    #[test]
    fn test_batch_job_get_param_not_found() {
        let job = BatchJob::new(0, "j", "a", "b").with_param("key", "value");
        assert_eq!(job.get_param("missing"), None);
    }

    #[test]
    fn test_batch_job_get_param_first_wins() {
        let job = BatchJob::new(0, "j", "a", "b")
            .with_param("k", "first")
            .with_param("k", "second");
        assert_eq!(job.get_param("k"), Some("first"));
    }

    #[test]
    fn test_batch_job_with_priority() {
        let job = BatchJob::new(0, "j", "a", "b").with_priority(42);
        assert_eq!(job.priority, 42);
    }

    #[test]
    fn test_batch_job_with_dependency() {
        let job = BatchJob::new(1, "j", "a", "b")
            .with_dependency(0)
            .with_dependency(3);
        assert_eq!(job.depends_on, vec![0, 3]);
    }

    // -----------------------------------------------------------------------
    // BatchJob::validate
    // -----------------------------------------------------------------------

    #[test]
    fn test_batch_job_validate_ok() {
        let job = BatchJob::new(0, "valid", "in.ply", "out.ply");
        assert!(job.validate().is_ok());
    }

    #[test]
    fn test_batch_job_validate_empty_name() {
        let job = BatchJob::new(0, "", "in.ply", "out.ply");
        let err = job.validate().unwrap_err();
        assert!(matches!(err, BatchError::InvalidJobSpec(_)));
    }

    #[test]
    fn test_batch_job_validate_empty_input() {
        let job = BatchJob::new(0, "name", "", "out.ply");
        let err = job.validate().unwrap_err();
        assert!(matches!(err, BatchError::InvalidJobSpec(_)));
    }

    #[test]
    fn test_batch_job_validate_empty_output() {
        let job = BatchJob::new(0, "name", "in.ply", "");
        let err = job.validate().unwrap_err();
        assert!(matches!(err, BatchError::InvalidJobSpec(_)));
    }

    // -----------------------------------------------------------------------
    // JobResult
    // -----------------------------------------------------------------------

    #[test]
    fn test_job_result_success_fields() {
        let r = JobResult::success(3, "my-job", 250, 5000);
        assert_eq!(r.job_id, 3);
        assert_eq!(r.job_name, "my-job");
        assert!(r.success);
        assert!(r.error_message.is_none());
        assert_eq!(r.duration_ms, 250);
        assert_eq!(r.items_processed, 5000);
        assert_eq!(r.attempts, 1);
        assert!(!r.skipped);
    }

    #[test]
    fn test_job_result_failure_fields() {
        let r = JobResult::failure(9, "bad-job", 100, "timeout");
        assert_eq!(r.job_id, 9);
        assert_eq!(r.job_name, "bad-job");
        assert!(!r.success);
        assert_eq!(r.error_message.as_deref(), Some("timeout"));
        assert_eq!(r.duration_ms, 100);
        assert_eq!(r.items_processed, 0);
        assert_eq!(r.attempts, 1);
        assert!(!r.skipped);
    }

    #[test]
    fn test_job_result_skipped_fields() {
        let r = JobResult::skipped(2, "skipped-job", "dependency failed");
        assert_eq!(r.job_id, 2);
        assert!(!r.success);
        assert!(r.skipped);
        assert_eq!(r.attempts, 0);
        assert_eq!(r.error_message.as_deref(), Some("dependency failed"));
    }

    #[test]
    fn test_job_result_with_attempts_builder() {
        let r = JobResult::success(0, "job", 10, 1).with_attempts(3);
        assert_eq!(r.attempts, 3);
    }

    // -----------------------------------------------------------------------
    // compute_execution_waves
    // -----------------------------------------------------------------------

    #[test]
    fn test_waves_empty_input() {
        let result = compute_execution_waves(&[]).expect("should be Ok");
        assert!(result.is_empty());
    }

    #[test]
    fn test_waves_no_deps_single_wave() {
        let jobs = vec![
            BatchJob::new(0, "a", "a.ply", "a_out.ply"),
            BatchJob::new(1, "b", "b.ply", "b_out.ply"),
            BatchJob::new(2, "c", "c.ply", "c_out.ply"),
        ];
        let waves = compute_execution_waves(&jobs).expect("should succeed");
        assert_eq!(waves.len(), 1);
        let mut wave0 = waves[0].clone();
        wave0.sort();
        assert_eq!(wave0, vec![0, 1, 2]);
    }

    #[test]
    fn test_waves_linear_chain_multiple_waves() {
        // 0 → 1 → 2 (linear chain: three separate waves)
        let jobs = vec![
            BatchJob::new(0, "a", "a.ply", "a_out.ply"),
            BatchJob::new(1, "b", "b.ply", "b_out.ply").with_dependency(0),
            BatchJob::new(2, "c", "c.ply", "c_out.ply").with_dependency(1),
        ];
        let waves = compute_execution_waves(&jobs).expect("should succeed");
        assert_eq!(waves.len(), 3);
        assert_eq!(waves[0], vec![0]);
        assert_eq!(waves[1], vec![1]);
        assert_eq!(waves[2], vec![2]);
    }

    #[test]
    fn test_waves_diamond_dependency() {
        // 0 → 1, 0 → 2, 1 → 3, 2 → 3 (diamond)
        let jobs = vec![
            BatchJob::new(0, "root", "r.ply", "r_out.ply"),
            BatchJob::new(1, "left", "l.ply", "l_out.ply").with_dependency(0),
            BatchJob::new(2, "right", "rg.ply", "rg_out.ply").with_dependency(0),
            BatchJob::new(3, "sink", "s.ply", "s_out.ply")
                .with_dependency(1)
                .with_dependency(2),
        ];
        let waves = compute_execution_waves(&jobs).expect("should succeed");
        // Wave 0: job 0; Wave 1: jobs 1 and 2; Wave 2: job 3
        assert_eq!(waves.len(), 3);
        assert_eq!(waves[0], vec![0]);
        let mut wave1 = waves[1].clone();
        wave1.sort();
        assert_eq!(wave1, vec![1, 2]);
        assert_eq!(waves[2], vec![3]);
    }

    #[test]
    fn test_waves_cycle_detection() {
        // 0 → 1 → 0 (cycle)
        let jobs = vec![
            BatchJob::new(0, "a", "a.ply", "a_out.ply").with_dependency(1),
            BatchJob::new(1, "b", "b.ply", "b_out.ply").with_dependency(0),
        ];
        let err = compute_execution_waves(&jobs).unwrap_err();
        assert_eq!(err, BatchError::DependencyCycle);
    }

    #[test]
    fn test_waves_duplicate_ids_rejected() {
        let jobs = vec![
            BatchJob::new(0, "a", "a.ply", "a_out.ply"),
            BatchJob::new(0, "b", "b.ply", "b_out.ply"),
        ];
        let err = compute_execution_waves(&jobs).unwrap_err();
        assert!(matches!(err, BatchError::InvalidJobSpec(_)));
    }

    #[test]
    fn test_waves_priority_ordering_within_wave() {
        // No dependencies: all in one wave, but priority should order them.
        let jobs = vec![
            BatchJob::new(0, "low", "l.ply", "l_out.ply").with_priority(1),
            BatchJob::new(1, "high", "h.ply", "h_out.ply").with_priority(10),
            BatchJob::new(2, "mid", "m.ply", "m_out.ply").with_priority(5),
        ];
        let waves = compute_execution_waves(&jobs).expect("should succeed");
        assert_eq!(waves.len(), 1);
        // Higher priority first: 1 (10) → 2 (5) → 0 (1)
        assert_eq!(waves[0], vec![1, 2, 0]);
    }

    // -----------------------------------------------------------------------
    // BatchConfig
    // -----------------------------------------------------------------------

    #[test]
    fn test_batch_config_default_values() {
        let cfg = BatchConfig::default();
        assert_eq!(cfg.max_parallel, 1);
        assert!(!cfg.fail_fast);
        assert_eq!(cfg.max_retries, 0);
        assert_eq!(cfg.retry_backoff_ms, 100);
        assert_eq!(cfg.verbosity, 1);
        assert!(!cfg.dry_run);
    }

    #[test]
    fn test_batch_config_validate_zero_parallel() {
        let cfg = BatchConfig {
            max_parallel: 0,
            ..Default::default()
        };
        let err = cfg.validate().unwrap_err();
        assert!(matches!(err, BatchError::InvalidConfig(_)));
    }

    #[test]
    fn test_batch_config_validate_ok() {
        let cfg = BatchConfig {
            max_parallel: 4,
            ..Default::default()
        };
        assert!(cfg.validate().is_ok());
    }

    // -----------------------------------------------------------------------
    // BatchProcessor construction
    // -----------------------------------------------------------------------

    #[test]
    fn test_batch_processor_new_empty_ok() {
        // Empty job list is allowed; execute() will fail, not new().
        let result = BatchProcessor::new(vec![], BatchConfig::default());
        assert!(result.is_ok());
    }

    #[test]
    fn test_batch_processor_from_jobs_basic() {
        let jobs = vec![BatchJob::new(0, "j", "in.ply", "out.ply")];
        let proc = BatchProcessor::from_jobs(jobs).expect("should succeed");
        assert_eq!(proc.job_count(), 1);
    }

    #[test]
    fn test_batch_processor_job_count() {
        let jobs = (0..5)
            .map(|i| BatchJob::new(i, format!("job-{i}"), "in.ply", "out.ply"))
            .collect();
        let proc = BatchProcessor::from_jobs(jobs).expect("ok");
        assert_eq!(proc.job_count(), 5);
    }

    #[test]
    fn test_batch_processor_add_job() {
        let mut proc = BatchProcessor::from_jobs(vec![]).expect("ok");
        assert_eq!(proc.job_count(), 0);
        proc.add_job(BatchJob::new(0, "j", "in.ply", "out.ply"))
            .expect("ok");
        assert_eq!(proc.job_count(), 1);
    }

    #[test]
    fn test_batch_processor_add_invalid_job_rejected() {
        let mut proc = BatchProcessor::from_jobs(vec![]).expect("ok");
        let err = proc
            .add_job(BatchJob::new(0, "", "in.ply", "out.ply"))
            .unwrap_err();
        assert!(matches!(err, BatchError::InvalidJobSpec(_)));
    }

    #[test]
    fn test_batch_processor_new_rejects_duplicate_ids() {
        let jobs = vec![
            BatchJob::new(0, "a", "a.ply", "a_out.ply"),
            BatchJob::new(0, "b", "b.ply", "b_out.ply"),
        ];
        let err = BatchProcessor::new(jobs, BatchConfig::default()).unwrap_err();
        assert!(matches!(err, BatchError::InvalidJobSpec(_)));
    }

    #[test]
    fn test_batch_processor_add_job_rejects_duplicate_id() {
        let mut proc = BatchProcessor::from_jobs(vec![BatchJob::new(0, "a", "a.ply", "a_out.ply")])
            .expect("ok");
        let err = proc
            .add_job(BatchJob::new(0, "dup", "b.ply", "b_out.ply"))
            .unwrap_err();
        assert!(matches!(err, BatchError::InvalidJobSpec(_)));
    }

    #[test]
    fn test_batch_processor_get_job_found() {
        let jobs = vec![BatchJob::new(42, "found", "in.ply", "out.ply")];
        let proc = BatchProcessor::from_jobs(jobs).expect("ok");
        let job = proc.get_job(42).expect("should find job 42");
        assert_eq!(job.name, "found");
    }

    #[test]
    fn test_batch_processor_get_job_not_found() {
        let proc = BatchProcessor::from_jobs(vec![]).expect("ok");
        assert!(proc.get_job(99).is_none());
    }

    // -----------------------------------------------------------------------
    // BatchProcessor::plan
    // -----------------------------------------------------------------------

    #[test]
    fn test_batch_processor_plan_no_deps() {
        let jobs = vec![
            BatchJob::new(0, "a", "a.ply", "a_out.ply"),
            BatchJob::new(1, "b", "b.ply", "b_out.ply"),
        ];
        let proc = BatchProcessor::from_jobs(jobs).expect("ok");
        let waves = proc.plan().expect("plan ok");
        assert_eq!(waves.len(), 1);
        assert_eq!(waves[0].len(), 2);
    }

    #[test]
    fn test_batch_processor_plan_with_deps() {
        let jobs = vec![
            BatchJob::new(0, "first", "a.ply", "a_out.ply"),
            BatchJob::new(1, "second", "b.ply", "b_out.ply").with_dependency(0),
        ];
        let proc = BatchProcessor::from_jobs(jobs).expect("ok");
        let waves = proc.plan().expect("plan ok");
        assert_eq!(waves.len(), 2);
        assert_eq!(waves[0], vec![0]);
        assert_eq!(waves[1], vec![1]);
    }

    // -----------------------------------------------------------------------
    // BatchProcessor::execute
    // -----------------------------------------------------------------------

    #[test]
    fn test_execute_empty_batch_error() {
        let proc = BatchProcessor::from_jobs(vec![]).expect("ok");
        let executor: JobExecutor = Box::new(|_| Ok(0));
        let err = proc.execute(&executor).unwrap_err();
        assert_eq!(err, BatchError::EmptyBatch);
    }

    #[test]
    fn test_execute_success() {
        let jobs = vec![
            BatchJob::new(0, "scene-0", "in0.ply", "out0.ply"),
            BatchJob::new(1, "scene-1", "in1.ply", "out1.ply"),
        ];
        let proc = BatchProcessor::from_jobs(jobs).expect("ok");
        let executor: JobExecutor = Box::new(|_| Ok(100));
        let (results, stats) = proc.execute(&executor).expect("execute ok");
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| r.success));
        assert_eq!(stats.successful_jobs, 2);
        assert_eq!(stats.failed_jobs, 0);
        assert_eq!(stats.total_items_processed, 200);
    }

    #[test]
    fn test_execute_fail_fast_stops_early() {
        // job 1 depends on job 0; executor always fails job 0.
        let jobs = vec![
            BatchJob::new(0, "fails", "a.ply", "a_out.ply"),
            BatchJob::new(1, "skipped", "b.ply", "b_out.ply").with_dependency(0),
            BatchJob::new(2, "also-skipped", "c.ply", "c_out.ply"),
        ];
        let config = BatchConfig {
            fail_fast: true,
            ..Default::default()
        };
        let proc = BatchProcessor::new(jobs, config).expect("ok");
        let executor: JobExecutor = Box::new(|_| Err("forced failure".to_string()));
        let (_results, stats) = proc
            .execute(&executor)
            .expect("execute returns ok even with failures");
        assert_eq!(stats.failed_jobs, 1); // only the first job in wave 0 ran and failed
                                          // Remaining jobs should be skipped (not failed)
        assert!(stats.skipped_jobs >= 1);
    }

    #[test]
    fn test_execute_dry_run_reports_skipped_not_failed() {
        let jobs = vec![
            BatchJob::new(0, "a", "a.ply", "a_out.ply"),
            BatchJob::new(1, "b", "b.ply", "b_out.ply"),
        ];
        let config = BatchConfig {
            dry_run: true,
            ..Default::default()
        };
        let proc = BatchProcessor::new(jobs, config).expect("ok");
        let executor: JobExecutor = Box::new(|_| Ok(1));
        let (results, stats) = proc.execute(&executor).expect("dry run ok");
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| r.skipped));
        assert!(results.iter().all(|r| !r.success));
        assert_eq!(stats.skipped_jobs, 2);
        assert_eq!(stats.successful_jobs, 0);
        assert_eq!(stats.failed_jobs, 0);
    }

    #[test]
    fn test_execute_fail_fast_default_max_parallel_matches_sequential_semantics() {
        // With the default max_parallel == 1, two independent jobs in the
        // same wave must still be processed one at a time: once the first
        // fails with fail_fast set, the second must be *skipped*, not also
        // executed (and failed) — regression guard for intra-wave
        // parallelism accidentally racing fail_fast checks.
        let jobs = vec![
            BatchJob::new(0, "a", "a.ply", "a_out.ply"),
            BatchJob::new(1, "b", "b.ply", "b_out.ply"),
        ];
        let config = BatchConfig {
            fail_fast: true,
            ..Default::default()
        };
        assert_eq!(config.max_parallel, 1);
        let proc = BatchProcessor::new(jobs, config).expect("ok");
        let executor: JobExecutor = Box::new(|_| Err("boom".to_string()));
        let (_results, stats) = proc.execute(&executor).expect("execute ok");
        assert_eq!(stats.failed_jobs, 1);
        assert_eq!(stats.skipped_jobs, 1);
    }

    #[test]
    fn test_execute_parallel_two_workers_runs_independent_jobs() {
        let jobs = vec![
            BatchJob::new(0, "a", "a.ply", "a_out.ply"),
            BatchJob::new(1, "b", "b.ply", "b_out.ply"),
            BatchJob::new(2, "c", "c.ply", "c_out.ply"),
        ];
        let config = BatchConfig {
            max_parallel: 2,
            ..Default::default()
        };
        let proc = BatchProcessor::new(jobs, config).expect("ok");
        let executor: JobExecutor = Box::new(|_| Ok(1));
        let (results, stats) = proc.execute(&executor).expect("execute ok");
        assert_eq!(results.len(), 3);
        assert_eq!(stats.successful_jobs, 3);
        assert_eq!(stats.failed_jobs, 0);
    }

    #[test]
    fn test_execute_retry_backoff_records_attempt_count() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        let call_count = Arc::new(AtomicUsize::new(0));
        let jobs = vec![BatchJob::new(0, "retry-job", "a.ply", "a_out.ply")];
        let config = BatchConfig {
            max_retries: 2,
            retry_backoff_ms: 1,
            ..Default::default()
        };
        let proc = BatchProcessor::new(jobs, config).expect("ok");
        let cc = Arc::clone(&call_count);
        let executor: JobExecutor = Box::new(move |_| {
            let n = cc.fetch_add(1, Ordering::SeqCst);
            if n < 2 {
                Err("transient".to_string())
            } else {
                Ok(10)
            }
        });
        let (results, _stats) = proc.execute(&executor).expect("ok");
        assert_eq!(call_count.load(Ordering::SeqCst), 3);
        assert!(results[0].success);
        assert_eq!(results[0].attempts, 3);
    }

    #[test]
    fn test_execute_continues_without_fail_fast() {
        let jobs = vec![
            BatchJob::new(0, "fail0", "a.ply", "a_out.ply"),
            BatchJob::new(1, "ok1", "b.ply", "b_out.ply"),
        ];
        let config = BatchConfig {
            fail_fast: false,
            ..Default::default()
        };
        let proc = BatchProcessor::new(jobs, config).expect("ok");
        let executor: JobExecutor = Box::new(|job| {
            if job.id == 0 {
                Err("job 0 failed".to_string())
            } else {
                Ok(50)
            }
        });
        let (results, stats) = proc.execute(&executor).expect("ok");
        assert_eq!(results.len(), 2);
        assert_eq!(stats.failed_jobs, 1);
        assert_eq!(stats.successful_jobs, 1);
        assert_eq!(stats.skipped_jobs, 0);
    }

    #[test]
    fn test_execute_retries() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        let call_count = Arc::new(AtomicUsize::new(0));
        let jobs = vec![BatchJob::new(0, "retry-job", "a.ply", "a_out.ply")];
        let config = BatchConfig {
            max_retries: 2,
            ..Default::default()
        };
        let proc = BatchProcessor::new(jobs, config).expect("ok");
        let cc = Arc::clone(&call_count);
        let executor: JobExecutor = Box::new(move |_| {
            let n = cc.fetch_add(1, Ordering::SeqCst);
            if n < 2 {
                Err("transient".to_string())
            } else {
                Ok(10)
            }
        });
        let (results, stats) = proc.execute(&executor).expect("ok");
        // 3 total calls (attempt 0, 1, 2 → succeeds on attempt 2)
        assert_eq!(call_count.load(Ordering::SeqCst), 3);
        assert!(results[0].success);
        assert_eq!(stats.successful_jobs, 1);
    }

    // -----------------------------------------------------------------------
    // BatchStats
    // -----------------------------------------------------------------------

    #[test]
    fn test_batch_stats_success_rate_full() {
        let stats = BatchStats {
            total_jobs: 10,
            successful_jobs: 10,
            ..Default::default()
        };
        assert!((stats.success_rate() - 1.0_f32).abs() < 1e-6);
    }

    #[test]
    fn test_batch_stats_success_rate_zero() {
        let stats = BatchStats {
            total_jobs: 5,
            successful_jobs: 0,
            ..Default::default()
        };
        assert!((stats.success_rate() - 0.0_f32).abs() < 1e-6);
    }

    #[test]
    fn test_batch_stats_success_rate_no_jobs() {
        let stats = BatchStats::default();
        // total_jobs = 0 → should not divide by zero
        assert_eq!(stats.success_rate(), 0.0_f32);
    }

    #[test]
    fn test_batch_stats_format_summary_smoke() {
        let stats = BatchStats {
            total_jobs: 10,
            successful_jobs: 8,
            failed_jobs: 1,
            skipped_jobs: 1,
            total_duration_ms: 5000,
            ..Default::default()
        };
        let s = stats.format_summary();
        assert!(s.contains("10 jobs"));
        assert!(s.contains("8 success"));
        assert!(s.contains("1 failed"));
        assert!(s.contains("1 skipped"));
        assert!(s.contains("5.00s"));
    }

    // -----------------------------------------------------------------------
    // jobs_from_file_list
    // -----------------------------------------------------------------------

    #[test]
    fn test_jobs_from_file_list_empty_error() {
        let err = jobs_from_file_list(&[], "output/").unwrap_err();
        assert_eq!(err, BatchError::EmptyBatch);
    }

    #[test]
    fn test_jobs_from_file_list_basic() {
        let paths: Vec<String> = vec![
            "scenes/a.ply".to_string(),
            "scenes/b.ply".to_string(),
            "scenes/c.ply".to_string(),
        ];
        let jobs = jobs_from_file_list(&paths, "output/").expect("ok");
        assert_eq!(jobs.len(), 3);
        assert_eq!(jobs[0].id, 0);
        assert_eq!(jobs[1].id, 1);
        assert_eq!(jobs[2].id, 2);
    }

    #[test]
    fn test_jobs_from_file_list_empty_path_error() {
        let paths = vec!["valid.ply".to_string(), String::new()];
        let err = jobs_from_file_list(&paths, "out/").unwrap_err();
        assert!(matches!(err, BatchError::FileNotFound(_)));
    }

    #[test]
    fn test_jobs_from_file_list_filename_as_name() {
        let paths = vec!["dir/subdir/scene_42.ply".to_string()];
        let jobs = jobs_from_file_list(&paths, "out/").expect("ok");
        assert_eq!(jobs[0].name, "scene_42.ply");
    }

    // -----------------------------------------------------------------------
    // jobs_from_directory
    // -----------------------------------------------------------------------

    #[test]
    fn test_jobs_from_directory_empty_dir_error() {
        let err = jobs_from_directory("", "ply", "output/").unwrap_err();
        assert!(matches!(err, BatchError::InvalidJobSpec(_)));
    }

    #[test]
    fn test_jobs_from_directory_empty_extension_error() {
        let err = jobs_from_directory("/some/dir", "", "output/").unwrap_err();
        assert!(matches!(err, BatchError::InvalidJobSpec(_)));
    }

    #[test]
    fn test_jobs_from_directory_empty_output_dir_error() {
        let err = jobs_from_directory("/some/dir", "ply", "").unwrap_err();
        assert!(matches!(err, BatchError::InvalidJobSpec(_)));
    }

    #[test]
    fn test_jobs_from_directory_nonexistent_dir_errors() {
        let result = jobs_from_directory(
            "/oxigaf_test_nonexistent_dir_xyz_should_not_exist",
            "ply",
            "/output",
        );
        assert!(matches!(result, Err(BatchError::FileNotFound(_))));
    }

    #[test]
    fn test_jobs_from_directory_scans_matching_files() {
        let base = std::env::temp_dir().join("oxigaf_batch_processor_test_scan");
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).expect("create test dir");
        std::fs::write(base.join("a.ply"), b"x").expect("write a.ply");
        std::fs::write(base.join("b.ply"), b"x").expect("write b.ply");
        std::fs::write(base.join("c.txt"), b"x").expect("write c.txt (should be ignored)");

        let dir_str = base.to_string_lossy().to_string();
        let jobs = jobs_from_directory(&dir_str, "ply", "/output").expect("scan ok");
        assert_eq!(jobs.len(), 2);
        let mut names: Vec<&str> = jobs.iter().map(|j| j.name.as_str()).collect();
        names.sort();
        assert_eq!(names, vec!["a.ply", "b.ply"]);
        let mut ids: Vec<usize> = jobs.iter().map(|j| j.id).collect();
        ids.sort();
        assert_eq!(ids, vec![0, 1]);

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn test_jobs_from_directory_empty_dir_scans_to_empty_jobs() {
        let base = std::env::temp_dir().join("oxigaf_batch_processor_test_scan_empty");
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).expect("create test dir");

        let dir_str = base.to_string_lossy().to_string();
        let jobs = jobs_from_directory(&dir_str, "ply", "/output").expect("scan ok");
        assert!(jobs.is_empty());

        let _ = std::fs::remove_dir_all(&base);
    }

    // -----------------------------------------------------------------------
    // merge_batches
    // -----------------------------------------------------------------------

    #[test]
    fn test_merge_batches_ids_offset() {
        let a = BatchProcessor::from_jobs(vec![
            BatchJob::new(0, "a0", "a0.ply", "a0_out.ply"),
            BatchJob::new(1, "a1", "a1.ply", "a1_out.ply"),
        ])
        .expect("ok");

        let b = BatchProcessor::from_jobs(vec![
            BatchJob::new(0, "b0", "b0.ply", "b0_out.ply"),
            BatchJob::new(1, "b1", "b1.ply", "b1_out.ply"),
        ])
        .expect("ok");

        let merged = merge_batches(a, b).expect("merge ok");
        assert_eq!(merged.job_count(), 4);

        // IDs in merged should be 0, 1, 2, 3
        let mut ids: Vec<usize> = merged.jobs.iter().map(|j| j.id).collect();
        ids.sort();
        assert_eq!(ids, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_merge_batches_deps_offset() {
        // b job 1 depends on b job 0; after merge with offset=2, dep should be id 2.
        let a = BatchProcessor::from_jobs(vec![
            BatchJob::new(0, "a0", "a0.ply", "a0_out.ply"),
            BatchJob::new(1, "a1", "a1.ply", "a1_out.ply"),
        ])
        .expect("ok");

        let b = BatchProcessor::from_jobs(vec![
            BatchJob::new(0, "b0", "b0.ply", "b0_out.ply"),
            BatchJob::new(1, "b1", "b1.ply", "b1_out.ply").with_dependency(0),
        ])
        .expect("ok");

        let merged = merge_batches(a, b).expect("merge ok");
        // The job originally 'b1' (id=1) is now id=3 and should depend on id=2.
        let b1 = merged.get_job(3).expect("job 3 should exist");
        assert_eq!(b1.depends_on, vec![2]);
    }

    #[test]
    fn test_merge_batches_plan_succeeds() {
        let a = BatchProcessor::from_jobs(vec![BatchJob::new(0, "a", "a.ply", "a_out.ply")])
            .expect("ok");
        let b = BatchProcessor::from_jobs(vec![BatchJob::new(0, "b", "b.ply", "b_out.ply")])
            .expect("ok");
        let merged = merge_batches(a, b).expect("ok");
        // Both jobs are independent, so plan should be a single wave with two jobs.
        let waves = merged.plan().expect("plan ok");
        assert_eq!(waves.len(), 1);
        assert_eq!(waves[0].len(), 2);
    }

    #[test]
    fn test_merge_batches_non_contiguous_ids_no_collision() {
        // `a` has ids [0, 2] (a gap at 1); job_count() would wrongly compute
        // offset = 2, colliding with a's own job id 2. The correct offset is
        // max(id) + 1 = 3.
        let a = BatchProcessor::from_jobs(vec![
            BatchJob::new(0, "a0", "a0.ply", "a0_out.ply"),
            BatchJob::new(2, "a2", "a2.ply", "a2_out.ply"),
        ])
        .expect("ok");
        let b = BatchProcessor::from_jobs(vec![BatchJob::new(0, "b0", "b0.ply", "b0_out.ply")])
            .expect("ok");

        let merged = merge_batches(a, b).expect("merge should not collide");
        let mut ids: Vec<usize> = merged.jobs.iter().map(|j| j.id).collect();
        ids.sort();
        assert_eq!(ids, vec![0, 2, 3]);
    }
}
