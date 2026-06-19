//! Data parallel training infrastructure.
//!
//! Multi-GPU training requires gradient synchronization across devices. This
//! module provides the data structures for data-parallel gradient aggregation.
//! Actual multi-GPU dispatch requires wgpu or CUDA; we implement the sync
//! logic without real GPU calls so that the upper training loop can be written
//! against a stable interface.

use crate::TrainerError;

// ---------------------------------------------------------------------------
// Sync mode
// ---------------------------------------------------------------------------

/// Gradient synchronization mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncMode {
    /// Average gradients across all workers.
    AllReduce,
    /// Sum gradients (caller divides manually).
    AllReduceSum,
    /// No synchronization (single device).
    NoSync,
}

// ---------------------------------------------------------------------------
// DataParallelConfig
// ---------------------------------------------------------------------------

/// Configuration for data parallel training.
#[derive(Debug, Clone)]
pub struct DataParallelConfig {
    /// Number of devices (default: 1).
    pub num_workers: usize,
    pub sync_mode: SyncMode,
    /// Enable gradient compression (future: quantize grads before sync).
    pub gradient_compression: bool,
    /// All-reduce bucket size in megabytes (default: 25.0 MB).
    pub bucket_size_mb: f32,
}

impl DataParallelConfig {
    /// Single-device configuration — no gradient sync.
    pub fn single_device() -> Self {
        Self {
            num_workers: 1,
            sync_mode: SyncMode::NoSync,
            gradient_compression: false,
            bucket_size_mb: 25.0,
        }
    }

    /// Multi-GPU configuration with AllReduce (average) sync.
    pub fn multi_gpu(n: usize) -> Self {
        Self {
            num_workers: n,
            sync_mode: SyncMode::AllReduce,
            gradient_compression: false,
            bucket_size_mb: 25.0,
        }
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), TrainerError> {
        if self.num_workers == 0 {
            return Err(TrainerError::InvalidConfig(
                "num_workers must be > 0".to_string(),
            ));
        }
        if !self.bucket_size_mb.is_finite() || self.bucket_size_mb <= 0.0 {
            return Err(TrainerError::InvalidConfig(format!(
                "bucket_size_mb must be a positive finite value, got {}",
                self.bucket_size_mb
            )));
        }
        Ok(())
    }

    /// Returns `true` if this is a multi-worker (distributed) configuration.
    pub fn is_distributed(&self) -> bool {
        self.num_workers > 1
    }

    /// Effective batch scale factor (linear batch scaling law).
    pub fn effective_batch_scale(&self) -> f32 {
        self.num_workers as f32
    }
}

// ---------------------------------------------------------------------------
// GradientAggregator
// ---------------------------------------------------------------------------

/// Tracks gradient aggregation across workers for one optimizer step.
pub struct GradientAggregator {
    pub config: DataParallelConfig,
    /// Accumulated gradients from each worker: `[worker_idx][param_idx]`.
    worker_grads: Vec<Vec<Vec<f32>>>,
    /// Per-worker completion flags.
    worker_done: Vec<bool>,
    completed_workers: usize,
    generation: u64,
}

impl GradientAggregator {
    /// Create a new aggregator.
    ///
    /// Initialises `worker_grads` as `num_workers × num_params` with empty
    /// inner `Vec<f32>` (gradients are supplied via `submit_gradients`).
    pub fn new(config: DataParallelConfig, num_params: usize) -> Self {
        let num_workers = config.num_workers;
        let worker_grads: Vec<Vec<Vec<f32>>> = (0..num_workers)
            .map(|_| (0..num_params).map(|_| Vec::new()).collect())
            .collect();
        let worker_done = vec![false; num_workers];
        Self {
            config,
            worker_grads,
            worker_done,
            completed_workers: 0,
            generation: 0,
        }
    }

    /// Submit gradient buffer for a parameter from a specific worker.
    ///
    /// Errors if `worker_idx >= num_workers` or `param_idx >= num_params`.
    pub fn submit_gradients(
        &mut self,
        worker_idx: usize,
        param_idx: usize,
        grads: Vec<f32>,
    ) -> Result<(), TrainerError> {
        let num_workers = self.config.num_workers;
        let num_params = self.worker_grads.first().map(|w| w.len()).unwrap_or(0);

        if worker_idx >= num_workers {
            return Err(TrainerError::Training(format!(
                "worker_idx {} out of range (num_workers={})",
                worker_idx, num_workers
            )));
        }
        if param_idx >= num_params {
            return Err(TrainerError::Training(format!(
                "param_idx {} out of range (num_params={})",
                param_idx, num_params
            )));
        }

        self.worker_grads[worker_idx][param_idx] = grads;
        Ok(())
    }

    /// Mark a worker as done for this iteration.
    ///
    /// Errors if the worker was already marked done.
    pub fn mark_worker_done(&mut self, worker_idx: usize) -> Result<(), TrainerError> {
        let num_workers = self.config.num_workers;
        if worker_idx >= num_workers {
            return Err(TrainerError::Training(format!(
                "worker_idx {} out of range (num_workers={})",
                worker_idx, num_workers
            )));
        }
        if self.worker_done[worker_idx] {
            return Err(TrainerError::Training(format!(
                "worker {} already marked done for generation {}",
                worker_idx, self.generation
            )));
        }
        self.worker_done[worker_idx] = true;
        self.completed_workers += 1;
        Ok(())
    }

    /// Returns `true` when all workers have called `mark_worker_done`.
    pub fn is_ready(&self) -> bool {
        self.completed_workers == self.config.num_workers
    }

    /// Aggregate gradients across workers according to [`SyncMode`].
    ///
    /// Returns `Vec<Vec<f32>>` indexed `[param_idx][grad_element]`.
    ///
    /// * **AllReduce**: element-wise average.
    /// * **AllReduceSum**: element-wise sum.
    /// * **NoSync**: returns worker 0's gradients unchanged.
    ///
    /// Missing / empty worker grad buffers are treated as zeros matching the
    /// length of non-empty buffers for that parameter.
    pub fn aggregate(&self) -> Result<Vec<Vec<f32>>, TrainerError> {
        if !self.is_ready() {
            return Err(TrainerError::Training(format!(
                "Cannot aggregate: only {}/{} workers done in generation {}",
                self.completed_workers, self.config.num_workers, self.generation
            )));
        }

        let num_params = self.worker_grads.first().map(|w| w.len()).unwrap_or(0);
        let num_workers = self.config.num_workers;

        let mut result: Vec<Vec<f32>> = Vec::with_capacity(num_params);

        match self.config.sync_mode {
            SyncMode::NoSync => {
                // Return worker 0's gradients
                let worker0 = self
                    .worker_grads
                    .first()
                    .ok_or_else(|| TrainerError::Training("no workers registered".to_string()))?;
                for param_grads in worker0 {
                    result.push(param_grads.clone());
                }
            }

            SyncMode::AllReduce | SyncMode::AllReduceSum => {
                for p in 0..num_params {
                    // Determine the length from the first non-empty buffer.
                    let param_len = self
                        .worker_grads
                        .iter()
                        .map(|wg| wg[p].len())
                        .find(|&l| l > 0)
                        .unwrap_or(0);

                    if param_len == 0 {
                        result.push(Vec::new());
                        continue;
                    }

                    let mut acc = vec![0.0f32; param_len];
                    for w in 0..num_workers {
                        let worker_param = &self.worker_grads[w][p];
                        if worker_param.is_empty() {
                            // Treat missing grads as zeros — nothing to add.
                            continue;
                        }
                        for (a, &g) in acc.iter_mut().zip(worker_param.iter()) {
                            *a += g;
                        }
                    }

                    if self.config.sync_mode == SyncMode::AllReduce {
                        let inv = 1.0 / num_workers as f32;
                        for a in acc.iter_mut() {
                            *a *= inv;
                        }
                    }

                    result.push(acc);
                }
            }
        }

        Ok(result)
    }

    /// Reset state for the next iteration and increment the generation counter.
    pub fn reset(&mut self) {
        let num_params = self.worker_grads.first().map(|w| w.len()).unwrap_or(0);
        let num_workers = self.config.num_workers;

        // Clear all gradient buffers
        for wg in self.worker_grads.iter_mut() {
            for pg in wg.iter_mut() {
                pg.clear();
            }
        }
        // Reset per-worker done flags
        for flag in self.worker_done.iter_mut() {
            *flag = false;
        }
        self.completed_workers = 0;
        self.generation += 1;

        // Ensure structure is correct (safety re-init if needed)
        debug_assert_eq!(self.worker_grads.len(), num_workers);
        debug_assert!(self.worker_grads.iter().all(|wg| wg.len() == num_params));
    }

    /// Current generation (number of completed aggregation cycles).
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Build a [`SyncReport`] for observability.
    pub fn sync_report(&self) -> SyncReport {
        let num_params = self.worker_grads.first().map(|w| w.len()).unwrap_or(0);

        let total_gradient_floats: usize = self
            .worker_grads
            .iter()
            .flat_map(|wg| wg.iter())
            .map(|grads| grads.len())
            .sum();

        let estimated_bandwidth_mb = (total_gradient_floats as f32 * 4.0) / (1024.0 * 1024.0);

        SyncReport {
            num_workers: self.config.num_workers,
            sync_mode: self.config.sync_mode,
            total_params: num_params,
            total_gradient_floats,
            estimated_bandwidth_mb,
            generation: self.generation,
        }
    }
}

// ---------------------------------------------------------------------------
// SyncReport
// ---------------------------------------------------------------------------

/// Observability report for one data-parallel synchronization.
#[derive(Debug, Clone)]
pub struct SyncReport {
    pub num_workers: usize,
    pub sync_mode: SyncMode,
    pub total_params: usize,
    pub total_gradient_floats: usize,
    /// `total_gradient_floats * 4 / (1024 * 1024)` — approximate MB transferred.
    pub estimated_bandwidth_mb: f32,
    pub generation: u64,
}

impl SyncReport {
    /// Human-readable summary string.
    pub fn format(&self) -> String {
        format!(
            "SyncReport {{ generation={}, workers={}, sync_mode={:?}, \
             params={}, grad_floats={}, bandwidth_mb={:.3} }}",
            self.generation,
            self.num_workers,
            self.sync_mode,
            self.total_params,
            self.total_gradient_floats,
            self.estimated_bandwidth_mb,
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    // --- Config tests -------------------------------------------------------

    #[test]
    fn test_config_single_device() {
        let cfg = DataParallelConfig::single_device();
        assert_eq!(cfg.num_workers, 1);
        assert_eq!(cfg.sync_mode, SyncMode::NoSync);
        assert!(!cfg.gradient_compression);
        assert_abs_diff_eq!(cfg.bucket_size_mb, 25.0, epsilon = 1e-6);
    }

    #[test]
    fn test_config_multi_gpu() {
        let cfg = DataParallelConfig::multi_gpu(4);
        assert_eq!(cfg.num_workers, 4);
        assert_eq!(cfg.sync_mode, SyncMode::AllReduce);
    }

    #[test]
    fn test_config_validate() {
        assert!(DataParallelConfig::single_device().validate().is_ok());
        assert!(DataParallelConfig::multi_gpu(8).validate().is_ok());

        let mut bad = DataParallelConfig::single_device();
        bad.num_workers = 0;
        assert!(bad.validate().is_err());

        let mut bad2 = DataParallelConfig::single_device();
        bad2.bucket_size_mb = -1.0;
        assert!(bad2.validate().is_err());
    }

    #[test]
    fn test_config_is_distributed() {
        assert!(!DataParallelConfig::single_device().is_distributed());
        assert!(DataParallelConfig::multi_gpu(2).is_distributed());
        assert!(DataParallelConfig::multi_gpu(8).is_distributed());
    }

    #[test]
    fn test_config_effective_batch_scale() {
        let cfg = DataParallelConfig::multi_gpu(4);
        assert_abs_diff_eq!(cfg.effective_batch_scale(), 4.0, epsilon = 1e-6);
    }

    // --- Aggregator tests ---------------------------------------------------

    #[test]
    fn test_aggregator_new() {
        let cfg = DataParallelConfig::multi_gpu(3);
        let agg = GradientAggregator::new(cfg, 10);
        assert_eq!(agg.worker_grads.len(), 3);
        for wg in &agg.worker_grads {
            assert_eq!(wg.len(), 10);
            for pg in wg {
                assert!(pg.is_empty());
            }
        }
        assert_eq!(agg.generation(), 0);
    }

    #[test]
    fn test_submit_gradients() {
        let cfg = DataParallelConfig::multi_gpu(2);
        let mut agg = GradientAggregator::new(cfg, 3);

        let grads = vec![1.0f32, 2.0f32, 3.0f32];
        assert!(agg.submit_gradients(0, 1, grads.clone()).is_ok());
        assert_eq!(agg.worker_grads[0][1], grads);

        // Out-of-range worker
        assert!(agg.submit_gradients(5, 0, vec![]).is_err());
        // Out-of-range param
        assert!(agg.submit_gradients(0, 10, vec![]).is_err());
    }

    #[test]
    fn test_mark_worker_done() {
        let cfg = DataParallelConfig::multi_gpu(2);
        let mut agg = GradientAggregator::new(cfg, 2);

        assert!(agg.mark_worker_done(0).is_ok());
        assert_eq!(agg.completed_workers, 1);

        // Double-mark should error
        assert!(agg.mark_worker_done(0).is_err());

        // Out-of-range
        assert!(agg.mark_worker_done(99).is_err());
    }

    #[test]
    fn test_is_ready_false_before_all_done() {
        let cfg = DataParallelConfig::multi_gpu(3);
        let mut agg = GradientAggregator::new(cfg, 1);

        assert!(!agg.is_ready());
        agg.mark_worker_done(0).expect("mark done");
        assert!(!agg.is_ready());
        agg.mark_worker_done(1).expect("mark done");
        assert!(!agg.is_ready());
    }

    #[test]
    fn test_is_ready_true_when_all_done() {
        let cfg = DataParallelConfig::multi_gpu(2);
        let mut agg = GradientAggregator::new(cfg, 1);

        agg.mark_worker_done(0).expect("done");
        agg.mark_worker_done(1).expect("done");
        assert!(agg.is_ready());
    }

    #[test]
    fn test_aggregate_all_reduce_average() {
        let cfg = DataParallelConfig::multi_gpu(2);
        let mut agg = GradientAggregator::new(cfg, 1);

        agg.submit_gradients(0, 0, vec![1.0f32, 3.0f32])
            .expect("submit");
        agg.submit_gradients(1, 0, vec![3.0f32, 5.0f32])
            .expect("submit");
        agg.mark_worker_done(0).expect("done");
        agg.mark_worker_done(1).expect("done");

        let result = agg.aggregate().expect("aggregate");
        assert_eq!(result.len(), 1);
        assert_abs_diff_eq!(result[0][0], 2.0, epsilon = 1e-6); // (1+3)/2
        assert_abs_diff_eq!(result[0][1], 4.0, epsilon = 1e-6); // (3+5)/2
    }

    #[test]
    fn test_aggregate_sum() {
        let mut cfg = DataParallelConfig::multi_gpu(2);
        cfg.sync_mode = SyncMode::AllReduceSum;
        let mut agg = GradientAggregator::new(cfg, 1);

        agg.submit_gradients(0, 0, vec![1.0f32, 2.0f32])
            .expect("submit");
        agg.submit_gradients(1, 0, vec![3.0f32, 4.0f32])
            .expect("submit");
        agg.mark_worker_done(0).expect("done");
        agg.mark_worker_done(1).expect("done");

        let result = agg.aggregate().expect("aggregate");
        assert_abs_diff_eq!(result[0][0], 4.0, epsilon = 1e-6); // 1+3
        assert_abs_diff_eq!(result[0][1], 6.0, epsilon = 1e-6); // 2+4
    }

    #[test]
    fn test_aggregate_no_sync_single_worker() {
        let cfg = DataParallelConfig::single_device();
        let mut agg = GradientAggregator::new(cfg, 2);

        agg.submit_gradients(0, 0, vec![7.0f32, 8.0f32])
            .expect("submit");
        agg.submit_gradients(0, 1, vec![9.0f32]).expect("submit");
        agg.mark_worker_done(0).expect("done");

        let result = agg.aggregate().expect("aggregate");
        assert_abs_diff_eq!(result[0][0], 7.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result[0][1], 8.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result[1][0], 9.0, epsilon = 1e-6);
    }

    #[test]
    fn test_aggregate_not_ready_error() {
        let cfg = DataParallelConfig::multi_gpu(2);
        let agg = GradientAggregator::new(cfg, 1);
        assert!(agg.aggregate().is_err());
    }

    #[test]
    fn test_reset_clears_state() {
        let cfg = DataParallelConfig::multi_gpu(2);
        let mut agg = GradientAggregator::new(cfg, 1);

        agg.submit_gradients(0, 0, vec![1.0f32]).expect("submit");
        agg.submit_gradients(1, 0, vec![2.0f32]).expect("submit");
        agg.mark_worker_done(0).expect("done");
        agg.mark_worker_done(1).expect("done");
        assert!(agg.is_ready());
        assert_eq!(agg.generation(), 0);

        agg.reset();
        assert!(!agg.is_ready());
        assert_eq!(agg.generation(), 1);
        // Grads should be cleared
        for wg in &agg.worker_grads {
            for pg in wg {
                assert!(pg.is_empty());
            }
        }
    }

    #[test]
    fn test_sync_report() {
        let cfg = DataParallelConfig::multi_gpu(2);
        let mut agg = GradientAggregator::new(cfg, 2);

        agg.submit_gradients(0, 0, vec![1.0f32; 100])
            .expect("submit");
        agg.submit_gradients(1, 0, vec![2.0f32; 100])
            .expect("submit");

        let report = agg.sync_report();
        assert_eq!(report.num_workers, 2);
        assert_eq!(report.sync_mode, SyncMode::AllReduce);
        assert_eq!(report.total_params, 2);
        assert_eq!(report.total_gradient_floats, 200); // 100 + 100
        let expected_mb = 200.0 * 4.0 / (1024.0 * 1024.0);
        assert_abs_diff_eq!(report.estimated_bandwidth_mb, expected_mb, epsilon = 1e-6);
        assert_eq!(report.generation, 0);

        let formatted = report.format();
        assert!(formatted.contains("workers=2"));
        assert!(formatted.contains("generation=0"));
    }
}
