//! Parallel frame rendering support for OxiGAF CLI.
//!
//! Provides [`ParallelRenderer`] which distributes frame rendering across CPU
//! threads using rayon. Supports turntable animations and arbitrary frame task
//! lists with configurable thread counts and progress reporting.
//!
//! # Example
//!
//! ```
//! use oxigaf_cli::parallel_render::{ParallelRenderConfig, ParallelRenderer};
//!
//! let config = ParallelRenderConfig {
//!     num_threads: 4,
//!     chunk_size: 2,
//!     width: 512,
//!     height: 512,
//!     ..Default::default()
//! };
//! let renderer = ParallelRenderer::new(config).expect("renderer creation failed");
//! let tasks = ParallelRenderer::turntable_tasks(renderer.config(), 8, 20.0);
//! let result = renderer.execute_mock(&tasks, 1.0);
//! assert!(result.all_succeeded());
//! ```

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use rayon::prelude::*;

use crate::error::CliError;
use crate::progress_types::BatchProgress;

// ---------------------------------------------------------------------------
// ParallelRenderConfig
// ---------------------------------------------------------------------------

/// Configuration for parallel rendering.
#[derive(Debug, Clone)]
pub struct ParallelRenderConfig {
    /// Number of threads to use. 0 = auto (use rayon default = all cores).
    pub num_threads: usize,

    /// Chunk size: how many frames to render per thread per batch.
    pub chunk_size: usize,

    /// Output directory for rendered frames.
    pub output_dir: PathBuf,

    /// Output filename pattern. The `{frame}` placeholder is replaced with the
    /// zero-padded frame index (4 digits). Example: `"frame_{frame}.png"`.
    pub filename_pattern: String,

    /// Image width in pixels.
    pub width: u32,

    /// Image height in pixels.
    pub height: u32,
}

impl Default for ParallelRenderConfig {
    fn default() -> Self {
        Self {
            num_threads: 0,
            chunk_size: 4,
            output_dir: PathBuf::from("."),
            filename_pattern: "frame_{:04d}.png".into(),
            width: 512,
            height: 512,
        }
    }
}

impl ParallelRenderConfig {
    /// Validate the configuration, returning an error if any field is invalid.
    ///
    /// # Errors
    ///
    /// Returns [`CliError::ConfigValidationError`] if:
    /// - `width` is zero
    /// - `height` is zero
    /// - `chunk_size` is zero
    /// - `filename_pattern` is empty
    pub fn validate(&self) -> Result<(), CliError> {
        if self.width == 0 {
            return Err(CliError::ConfigValidationError {
                reason: "render width must be greater than zero".into(),
            });
        }
        if self.height == 0 {
            return Err(CliError::ConfigValidationError {
                reason: "render height must be greater than zero".into(),
            });
        }
        if self.chunk_size == 0 {
            return Err(CliError::ConfigValidationError {
                reason: "chunk_size must be greater than zero".into(),
            });
        }
        if self.filename_pattern.is_empty() {
            return Err(CliError::ConfigValidationError {
                reason: "filename_pattern must not be empty".into(),
            });
        }
        Ok(())
    }

    /// Format a filename for a given frame index.
    ///
    /// The frame index is zero-padded to 4 digits. The `{:04d}` placeholder
    /// in the pattern is replaced with the formatted index. If no placeholder
    /// is present the index is appended before the file extension, or at the
    /// end of the string when no extension is found.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxigaf_cli::parallel_render::ParallelRenderConfig;
    ///
    /// let cfg = ParallelRenderConfig::default();
    /// assert_eq!(cfg.frame_filename(0), "frame_0000.png");
    /// assert_eq!(cfg.frame_filename(42), "frame_0042.png");
    /// assert_eq!(cfg.frame_filename(9999), "frame_9999.png");
    /// ```
    #[must_use]
    pub fn frame_filename(&self, frame_index: usize) -> String {
        let padded = format!("{:04}", frame_index);
        // Replace the Python-style {:04d} placeholder with the formatted index
        if self.filename_pattern.contains("{:04d}") {
            self.filename_pattern.replace("{:04d}", &padded)
        } else if self.filename_pattern.contains("{frame}") {
            self.filename_pattern.replace("{frame}", &padded)
        } else {
            // Fall back: insert before extension or append
            match self.filename_pattern.rfind('.') {
                Some(dot) => {
                    let (stem, ext) = self.filename_pattern.split_at(dot);
                    format!("{}_{}{}", stem, padded, ext)
                }
                None => format!("{}_{}", self.filename_pattern, padded),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ParallelRenderResult
// ---------------------------------------------------------------------------

/// Results from a parallel render job.
#[derive(Debug)]
pub struct ParallelRenderResult {
    /// Total number of frames submitted.
    pub total_frames: usize,
    /// Number of frames that succeeded.
    pub successful: usize,
    /// Number of frames that failed.
    pub failed: usize,
    /// Wall-clock duration of the entire render job.
    pub total_duration: std::time::Duration,
    /// Throughput in frames per second.
    pub frames_per_second: f64,
    /// Per-frame errors: `(frame_index, error_message)`.
    pub errors: Vec<(usize, String)>,
}

impl ParallelRenderResult {
    /// Returns `true` if all frames succeeded.
    #[must_use]
    pub fn all_succeeded(&self) -> bool {
        self.failed == 0
    }

    /// Format a human-readable summary of the render result.
    ///
    /// The summary includes total frames, successful/failed counts, duration,
    /// and throughput. Error details are listed when failures occurred.
    #[must_use]
    pub fn format_summary(&self) -> String {
        let mut lines = Vec::new();

        lines.push(format!(
            "Render complete: {}/{} frames succeeded in {:.2}s ({:.1} frames/s)",
            self.successful,
            self.total_frames,
            self.total_duration.as_secs_f64(),
            self.frames_per_second,
        ));

        if self.failed > 0 {
            lines.push(format!("  {} frames failed:", self.failed));
            for (idx, msg) in &self.errors {
                lines.push(format!("    frame {:04}: {}", idx, msg));
            }
        }

        lines.join("\n")
    }
}

// ---------------------------------------------------------------------------
// FrameTask
// ---------------------------------------------------------------------------

/// A single frame rendering task.
#[derive(Debug, Clone)]
pub struct FrameTask {
    /// Zero-based index of this frame in the sequence.
    pub frame_index: usize,
    /// Destination file path for the rendered frame.
    pub output_path: PathBuf,
    /// Camera azimuth in degrees (horizontal rotation around the subject).
    pub camera_azimuth_deg: f32,
    /// Camera elevation in degrees (vertical angle above the horizontal plane).
    pub camera_elevation_deg: f32,
}

// ---------------------------------------------------------------------------
// ParallelRenderer
// ---------------------------------------------------------------------------

/// Parallel frame renderer.
///
/// Distributes frame tasks across CPU threads using rayon. A custom thread
/// pool is created when `num_threads > 0`; otherwise the rayon global pool
/// (which defaults to the number of logical CPU cores) is used.
pub struct ParallelRenderer {
    config: ParallelRenderConfig,
    thread_pool: Option<rayon::ThreadPool>,
}

impl ParallelRenderer {
    /// Create a new `ParallelRenderer` from the given configuration.
    ///
    /// When `config.num_threads > 0` a dedicated rayon thread pool is created
    /// with that many threads. When `num_threads == 0` the global rayon pool
    /// is used.
    ///
    /// # Errors
    ///
    /// Returns [`CliError::ConfigValidationError`] if the configuration is
    /// invalid, or if thread pool creation fails.
    pub fn new(config: ParallelRenderConfig) -> Result<Self, CliError> {
        config.validate()?;

        let thread_pool = if config.num_threads > 0 {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(config.num_threads)
                .build()
                .map_err(|e| CliError::ConfigValidationError {
                    reason: format!("failed to create thread pool: {}", e),
                })?;
            Some(pool)
        } else {
            None
        };

        Ok(Self {
            config,
            thread_pool,
        })
    }

    /// Access the configuration.
    #[must_use]
    pub fn config(&self) -> &ParallelRenderConfig {
        &self.config
    }

    /// Create a list of [`FrameTask`]s for a turntable (360°) animation.
    ///
    /// Azimuths are evenly distributed across 360° starting at 0°. All frames
    /// share the same elevation angle.
    ///
    /// # Arguments
    ///
    /// * `config` — render configuration (used for output paths and dimensions)
    /// * `n_frames` — total number of frames in the animation
    /// * `elevation_deg` — constant camera elevation in degrees
    ///
    /// # Panics
    ///
    /// Never panics — returns an empty `Vec` when `n_frames == 0`.
    #[must_use]
    pub fn turntable_tasks(
        config: &ParallelRenderConfig,
        n_frames: usize,
        elevation_deg: f32,
    ) -> Vec<FrameTask> {
        (0..n_frames)
            .map(|i| {
                let azimuth = if n_frames > 1 {
                    360.0 * (i as f32) / (n_frames as f32)
                } else {
                    0.0
                };
                let filename = config.frame_filename(i);
                FrameTask {
                    frame_index: i,
                    output_path: config.output_dir.join(&filename),
                    camera_azimuth_deg: azimuth,
                    camera_elevation_deg: elevation_deg,
                }
            })
            .collect()
    }

    /// Execute frame tasks in parallel using the configured thread pool.
    ///
    /// `render_fn` is called concurrently for each task. It must be both
    /// `Send` and `Sync` so that it can be shared across rayon threads.
    ///
    /// Progress is reported through the optional [`BatchProgress`] handle;
    /// it is incremented once per completed task (success or failure).
    ///
    /// # Arguments
    ///
    /// * `tasks` — slice of tasks to process
    /// * `render_fn` — closure returning `Ok(())` on success or `Err(msg)` on failure
    /// * `progress` — optional progress bar to update as tasks complete
    ///
    /// # Returns
    ///
    /// A [`ParallelRenderResult`] summarising success/failure counts, timing,
    /// and per-frame error details.
    pub fn execute<F>(
        &self,
        tasks: &[FrameTask],
        render_fn: F,
        progress: Option<&BatchProgress>,
    ) -> ParallelRenderResult
    where
        F: Fn(&FrameTask) -> Result<(), String> + Send + Sync,
    {
        let total_frames = tasks.len();
        let errors: Arc<Mutex<Vec<(usize, String)>>> = Arc::new(Mutex::new(Vec::new()));
        let successful = Arc::new(std::sync::atomic::AtomicUsize::new(0));

        let start = Instant::now();

        let execute_inner = |tasks: &[FrameTask]| {
            tasks.par_iter().for_each(|task| {
                match render_fn(task) {
                    Ok(()) => {
                        successful.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                    Err(msg) => {
                        if let Ok(mut guard) = errors.lock() {
                            guard.push((task.frame_index, msg));
                        }
                    }
                }
                if let Some(pb) = progress {
                    pb.increment();
                }
            });
        };

        match &self.thread_pool {
            Some(pool) => pool.install(|| execute_inner(tasks)),
            None => execute_inner(tasks),
        }

        let total_duration = start.elapsed();
        let successful_count = successful.load(std::sync::atomic::Ordering::Relaxed);
        let errors_vec = Arc::try_unwrap(errors)
            .unwrap_or_else(|arc| {
                // Fallback: clone the inner value if we can't take ownership
                arc.lock()
                    .map(|g| Mutex::new(g.clone()))
                    .unwrap_or_else(|_| Mutex::new(Vec::new()))
            })
            .into_inner()
            .unwrap_or_default();

        let failed = total_frames - successful_count;
        let secs = total_duration.as_secs_f64();
        let frames_per_second = if secs > 0.0 {
            total_frames as f64 / secs
        } else {
            f64::INFINITY
        };

        ParallelRenderResult {
            total_frames,
            successful: successful_count,
            failed,
            total_duration,
            frames_per_second,
            errors: errors_vec,
        }
    }

    /// Render tasks with a mock render function (useful for scheduling tests).
    ///
    /// Tasks with index `< floor(n * success_rate)` succeed; the rest fail.
    /// This is fully deterministic — no randomness is involved.
    ///
    /// # Arguments
    ///
    /// * `tasks` — slice of tasks to process
    /// * `success_rate` — fraction of tasks that succeed (0.0 = all fail, 1.0 = all succeed)
    pub fn execute_mock(&self, tasks: &[FrameTask], success_rate: f64) -> ParallelRenderResult {
        let n = tasks.len();
        let success_threshold = (n as f64 * success_rate.clamp(0.0, 1.0)).floor() as usize;

        self.execute(
            tasks,
            move |task| {
                if task.frame_index < success_threshold {
                    Ok(())
                } else {
                    Err(format!("mock failure for frame {}", task.frame_index))
                }
            },
            None,
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::thread;

    fn default_renderer() -> ParallelRenderer {
        ParallelRenderer::new(ParallelRenderConfig::default())
            .expect("default config must be valid")
    }

    // -----------------------------------------------------------------------
    // ParallelRenderConfig tests
    // -----------------------------------------------------------------------

    #[test]
    fn config_default_has_correct_fields() {
        let cfg = ParallelRenderConfig::default();
        assert_eq!(cfg.num_threads, 0);
        assert_eq!(cfg.chunk_size, 4);
        assert_eq!(cfg.width, 512);
        assert_eq!(cfg.height, 512);
        assert_eq!(cfg.output_dir, PathBuf::from("."));
        assert_eq!(cfg.filename_pattern, "frame_{:04d}.png");
    }

    #[test]
    fn validate_rejects_zero_width() {
        let cfg = ParallelRenderConfig {
            width: 0,
            ..Default::default()
        };
        let err = cfg.validate();
        assert!(err.is_err(), "validate() should reject width=0");
        let msg = err.unwrap_err().to_string();
        assert!(
            msg.contains("width"),
            "error should mention 'width', got: {}",
            msg
        );
    }

    #[test]
    fn validate_rejects_zero_height() {
        let cfg = ParallelRenderConfig {
            height: 0,
            ..Default::default()
        };
        let err = cfg.validate();
        assert!(err.is_err(), "validate() should reject height=0");
        let msg = err.unwrap_err().to_string();
        assert!(
            msg.contains("height"),
            "error should mention 'height', got: {}",
            msg
        );
    }

    #[test]
    fn validate_rejects_zero_chunk_size() {
        let cfg = ParallelRenderConfig {
            chunk_size: 0,
            ..Default::default()
        };
        assert!(
            cfg.validate().is_err(),
            "validate() should reject chunk_size=0"
        );
    }

    #[test]
    fn validate_rejects_empty_filename_pattern() {
        let cfg = ParallelRenderConfig {
            filename_pattern: String::new(),
            ..Default::default()
        };
        assert!(
            cfg.validate().is_err(),
            "validate() should reject empty filename_pattern"
        );
    }

    #[test]
    fn validate_accepts_valid_config() {
        let cfg = ParallelRenderConfig::default();
        assert!(
            cfg.validate().is_ok(),
            "default config must pass validation"
        );
    }

    #[test]
    fn frame_filename_formats_zero_as_four_digits() {
        let cfg = ParallelRenderConfig::default();
        assert_eq!(cfg.frame_filename(0), "frame_0000.png");
    }

    #[test]
    fn frame_filename_formats_42() {
        let cfg = ParallelRenderConfig::default();
        assert_eq!(cfg.frame_filename(42), "frame_0042.png");
    }

    #[test]
    fn frame_filename_formats_large_index() {
        let cfg = ParallelRenderConfig::default();
        // 9999 still fits in 4 digits
        assert_eq!(cfg.frame_filename(9999), "frame_9999.png");
    }

    // -----------------------------------------------------------------------
    // turntable_tasks tests
    // -----------------------------------------------------------------------

    #[test]
    fn turntable_tasks_returns_correct_count() {
        let cfg = ParallelRenderConfig::default();
        let tasks = ParallelRenderer::turntable_tasks(&cfg, 8, 20.0);
        assert_eq!(tasks.len(), 8);
    }

    #[test]
    fn turntable_tasks_azimuths_are_evenly_spaced() {
        let cfg = ParallelRenderConfig::default();
        let tasks = ParallelRenderer::turntable_tasks(&cfg, 8, 0.0);
        let expected_step = 45.0_f32; // 360 / 8
        for (i, task) in tasks.iter().enumerate() {
            let expected_az = expected_step * i as f32;
            assert!(
                (task.camera_azimuth_deg - expected_az).abs() < 1e-4,
                "frame {}: expected azimuth {}, got {}",
                i,
                expected_az,
                task.camera_azimuth_deg
            );
        }
    }

    #[test]
    fn turntable_tasks_single_frame_at_azimuth_zero() {
        let cfg = ParallelRenderConfig::default();
        let tasks = ParallelRenderer::turntable_tasks(&cfg, 1, 30.0);
        assert_eq!(tasks.len(), 1);
        assert!(
            tasks[0].camera_azimuth_deg.abs() < 1e-4,
            "single-frame azimuth should be 0, got {}",
            tasks[0].camera_azimuth_deg
        );
        assert!(
            (tasks[0].camera_elevation_deg - 30.0).abs() < 1e-4,
            "elevation should be 30, got {}",
            tasks[0].camera_elevation_deg
        );
    }

    #[test]
    fn turntable_tasks_zero_frames_returns_empty() {
        let cfg = ParallelRenderConfig::default();
        let tasks = ParallelRenderer::turntable_tasks(&cfg, 0, 20.0);
        assert!(tasks.is_empty());
    }

    // -----------------------------------------------------------------------
    // execute tests
    // -----------------------------------------------------------------------

    #[test]
    fn execute_with_success_fn_returns_all_successful() {
        let renderer = default_renderer();
        let cfg = ParallelRenderConfig::default();
        let tasks = ParallelRenderer::turntable_tasks(&cfg, 8, 0.0);
        let result = renderer.execute(&tasks, |_task| Ok(()), None);
        assert_eq!(result.total_frames, 8);
        assert_eq!(result.successful, 8);
        assert_eq!(result.failed, 0);
        assert!(result.errors.is_empty());
        assert!(result.all_succeeded());
    }

    #[test]
    fn execute_with_fail_fn_returns_all_failed() {
        let renderer = default_renderer();
        let cfg = ParallelRenderConfig::default();
        let tasks = ParallelRenderer::turntable_tasks(&cfg, 5, 0.0);
        let result = renderer.execute(
            &tasks,
            |task| Err(format!("simulated error for frame {}", task.frame_index)),
            None,
        );
        assert_eq!(result.total_frames, 5);
        assert_eq!(result.successful, 0);
        assert_eq!(result.failed, 5);
        assert_eq!(result.errors.len(), 5);
        assert!(!result.all_succeeded());
    }

    #[test]
    fn execute_accumulates_errors_with_frame_indices() {
        let renderer = default_renderer();
        let cfg = ParallelRenderConfig::default();
        // Only frame index 3 fails
        let tasks = ParallelRenderer::turntable_tasks(&cfg, 5, 0.0);
        let result = renderer.execute(
            &tasks,
            |task| {
                if task.frame_index == 3 {
                    Err("frame 3 is broken".into())
                } else {
                    Ok(())
                }
            },
            None,
        );
        assert_eq!(result.successful, 4);
        assert_eq!(result.failed, 1);
        assert_eq!(result.errors.len(), 1);
        assert_eq!(result.errors[0].0, 3);
        assert!(result.errors[0].1.contains("frame 3"));
    }

    // -----------------------------------------------------------------------
    // execute_mock tests
    // -----------------------------------------------------------------------

    #[test]
    fn execute_mock_success_rate_one_all_succeed() {
        let renderer = default_renderer();
        let cfg = ParallelRenderConfig::default();
        let tasks = ParallelRenderer::turntable_tasks(&cfg, 10, 0.0);
        let result = renderer.execute_mock(&tasks, 1.0);
        assert!(
            result.all_succeeded(),
            "success_rate=1.0 should mean all succeed"
        );
        assert_eq!(result.successful, 10);
        assert_eq!(result.failed, 0);
    }

    #[test]
    fn execute_mock_success_rate_zero_all_fail() {
        let renderer = default_renderer();
        let cfg = ParallelRenderConfig::default();
        let tasks = ParallelRenderer::turntable_tasks(&cfg, 8, 0.0);
        let result = renderer.execute_mock(&tasks, 0.0);
        assert_eq!(result.successful, 0);
        assert_eq!(result.failed, 8);
        assert!(!result.all_succeeded());
    }

    #[test]
    fn execute_mock_partial_success_rate() {
        let renderer = default_renderer();
        let cfg = ParallelRenderConfig::default();
        let n = 10;
        let tasks = ParallelRenderer::turntable_tasks(&cfg, n, 0.0);
        let result = renderer.execute_mock(&tasks, 0.5);
        // 50% of 10 = 5 succeed (indices 0..5)
        assert_eq!(result.successful, 5, "half should succeed");
        assert_eq!(result.failed, 5, "half should fail");
    }

    // -----------------------------------------------------------------------
    // ParallelRenderResult format_summary tests
    // -----------------------------------------------------------------------

    #[test]
    fn format_summary_is_non_empty() {
        let renderer = default_renderer();
        let cfg = ParallelRenderConfig::default();
        let tasks = ParallelRenderer::turntable_tasks(&cfg, 4, 0.0);
        let result = renderer.execute_mock(&tasks, 1.0);
        let summary = result.format_summary();
        assert!(!summary.is_empty(), "format_summary() must not be empty");
    }

    #[test]
    fn format_summary_contains_frames_per_second() {
        let renderer = default_renderer();
        let cfg = ParallelRenderConfig::default();
        let tasks = ParallelRenderer::turntable_tasks(&cfg, 4, 0.0);
        let result = renderer.execute_mock(&tasks, 1.0);
        let summary = result.format_summary();
        assert!(
            summary.contains("frames/s"),
            "format_summary() should contain 'frames/s', got: {}",
            summary
        );
    }

    #[test]
    fn format_summary_mentions_failed_frames_on_failure() {
        let renderer = default_renderer();
        let cfg = ParallelRenderConfig::default();
        let tasks = ParallelRenderer::turntable_tasks(&cfg, 4, 0.0);
        let result = renderer.execute_mock(&tasks, 0.0);
        let summary = result.format_summary();
        assert!(
            summary.contains("failed") || summary.contains("fail"),
            "format_summary() should mention failures, got: {}",
            summary
        );
    }

    // -----------------------------------------------------------------------
    // Thread-count test: verify actual multi-threaded execution
    // -----------------------------------------------------------------------

    #[test]
    fn execute_uses_multiple_threads_for_many_tasks() {
        // Build a renderer with explicit multi-thread pool
        let config = ParallelRenderConfig {
            num_threads: 4,
            chunk_size: 2,
            ..Default::default()
        };
        let renderer = ParallelRenderer::new(config).expect("renderer creation should succeed");

        let base_cfg = ParallelRenderConfig::default();
        // Use plenty of tasks so rayon distributes across threads
        let tasks = ParallelRenderer::turntable_tasks(&base_cfg, 32, 0.0);

        let thread_ids: Arc<Mutex<HashSet<thread::ThreadId>>> =
            Arc::new(Mutex::new(HashSet::new()));
        let thread_ids_clone = Arc::clone(&thread_ids);

        renderer.execute(
            &tasks,
            move |_task| {
                // Record which thread is executing this task.
                if let Ok(mut set) = thread_ids_clone.lock() {
                    set.insert(thread::current().id());
                }
                // Do enough CPU work that rayon must dispatch to multiple threads
                // rather than completing all tasks on one before stealing begins.
                let mut acc: u64 = 0;
                for i in 0..50_000u64 {
                    acc = acc.wrapping_add(i.wrapping_mul(i));
                }
                std::hint::black_box(acc);
                Ok(())
            },
            None,
        );

        let observed_threads = thread_ids.lock().map(|g| g.len()).unwrap_or(0);

        // Only assert multi-threading when the host actually has multiple cores.
        // Single-core CI/container environments legitimately use 1 thread.
        let available_cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        if available_cores >= 2 {
            assert!(
                observed_threads >= 2,
                "expected at least 2 threads to be used, but observed only {}",
                observed_threads
            );
        }
    }

    // -----------------------------------------------------------------------
    // ParallelRenderer::new tests
    // -----------------------------------------------------------------------

    #[test]
    fn renderer_new_succeeds_with_default_config() {
        let result = ParallelRenderer::new(ParallelRenderConfig::default());
        assert!(result.is_ok(), "new() should succeed with default config");
    }

    #[test]
    fn renderer_new_fails_with_invalid_config() {
        let cfg = ParallelRenderConfig {
            width: 0,
            ..Default::default()
        };
        let result = ParallelRenderer::new(cfg);
        assert!(result.is_err(), "new() should fail with width=0");
    }

    #[test]
    fn renderer_new_with_explicit_thread_count() {
        let cfg = ParallelRenderConfig {
            num_threads: 2,
            ..Default::default()
        };
        let result = ParallelRenderer::new(cfg);
        assert!(result.is_ok(), "new() should succeed with num_threads=2");
    }
}
