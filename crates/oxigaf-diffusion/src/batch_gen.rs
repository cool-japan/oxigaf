//! Batch generation API for multi-view diffusion inference.
//!
//! Provides a queue-based interface for processing multiple reference images,
//! with optional KV-cache integration for cross-attention acceleration.
//!
//! # Example
//!
//! ```rust
//! use oxigaf_diffusion::batch_gen::{BatchGenerator, BatchGenConfig, GenerationRequest};
//!
//! let config = BatchGenConfig::default();
//! let gen = BatchGenerator::new(config);
//!
//! let request = GenerationRequest {
//!     id: "img-001".into(),
//!     reference_image: vec![128u8; 256 * 256 * 3],
//!     image_width: 256,
//!     image_height: 256,
//!     num_views: 2,
//!     guidance_scale: None,
//!     num_steps: None,
//!     seed: None,
//! };
//!
//! let result = gen.process_one(request).expect("generation failed");
//! assert_eq!(result.views.len(), 2);
//! ```

use std::sync::{Arc, Mutex};

use crate::{kv_cache::KVCache, kv_cache::KVCacheConfig, DiffusionError};

// ---------------------------------------------------------------------------
// GenerationRequest
// ---------------------------------------------------------------------------

/// Request for generating views of one reference image.
#[derive(Debug, Clone)]
pub struct GenerationRequest {
    /// Unique request ID (for matching responses).
    pub id: String,

    /// Reference image data (RGB bytes, flat row-major).
    pub reference_image: Vec<u8>,

    /// Width of the reference image in pixels.
    pub image_width: u32,

    /// Height of the reference image in pixels.
    pub image_height: u32,

    /// Number of output views to generate.
    pub num_views: usize,

    /// Guidance scale (overrides config default when `Some`).
    pub guidance_scale: Option<f32>,

    /// Number of denoising steps (overrides config default when `Some`).
    pub num_steps: Option<usize>,

    /// Random seed for reproducibility.
    pub seed: Option<u64>,
}

// ---------------------------------------------------------------------------
// GeneratedView
// ---------------------------------------------------------------------------

/// Output for one generated view.
#[derive(Debug, Clone)]
pub struct GeneratedView {
    /// Zero-based index of this view in the result set.
    pub view_index: usize,

    /// Pixel data for this view (RGB bytes, flat row-major).
    pub image_data: Vec<u8>,

    /// Width of the generated image in pixels.
    pub width: u32,

    /// Height of the generated image in pixels.
    pub height: u32,

    /// Wall-clock time in milliseconds to generate this view.
    pub generation_time_ms: f64,
}

// ---------------------------------------------------------------------------
// GenerationResult
// ---------------------------------------------------------------------------

/// Result for one [`GenerationRequest`].
#[derive(Debug, Clone)]
pub struct GenerationResult {
    /// Matches the `id` from the corresponding [`GenerationRequest`].
    pub id: String,

    /// All generated views, in order.
    pub views: Vec<GeneratedView>,

    /// Total wall-clock time in milliseconds for the entire request.
    pub total_time_ms: f64,

    /// Number of KV pairs that were served from cache (0 in placeholder mode).
    pub num_cached_kv: usize,
}

impl GenerationResult {
    /// Returns `true` if at least one view was generated.
    pub fn all_views_generated(&self) -> bool {
        !self.views.is_empty()
    }

    /// Throughput in views per second.
    ///
    /// Returns `0.0` when `total_time_ms` is zero (to avoid division by zero).
    pub fn throughput_views_per_sec(&self) -> f64 {
        if self.total_time_ms > 0.0 {
            self.views.len() as f64 / (self.total_time_ms / 1000.0)
        } else {
            0.0
        }
    }
}

// ---------------------------------------------------------------------------
// BatchGenConfig
// ---------------------------------------------------------------------------

/// Configuration for [`BatchGenerator`].
#[derive(Debug, Clone)]
pub struct BatchGenConfig {
    /// Maximum requests to process simultaneously.
    ///
    /// Default: `4`.
    pub max_batch_size: usize,

    /// Maximum output views per request.
    ///
    /// Default: `4`.
    pub max_views_per_request: usize,

    /// Default guidance scale applied when a request omits `guidance_scale`.
    ///
    /// Default: `3.0`.
    pub guidance_scale: f32,

    /// Default number of denoising steps applied when a request omits `num_steps`.
    ///
    /// Default: `20`.
    pub num_steps: usize,

    /// Whether to use KV cache for cross-attention.
    ///
    /// Default: `true`.
    pub use_kv_cache: bool,

    /// Whether to process synchronously (blocking) or queue.
    ///
    /// Default: `true`.
    pub synchronous: bool,
}

impl Default for BatchGenConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 4,
            max_views_per_request: 4,
            guidance_scale: 3.0,
            num_steps: 20,
            use_kv_cache: true,
            synchronous: true,
        }
    }
}

// ---------------------------------------------------------------------------
// BatchStats
// ---------------------------------------------------------------------------

/// Cumulative statistics for a [`BatchGenerator`].
#[derive(Debug, Clone, Default)]
pub struct BatchStats {
    /// Total number of generation requests processed.
    pub total_requests: u64,

    /// Total number of individual views generated across all requests.
    pub total_views_generated: u64,

    /// Cumulative wall-clock time in milliseconds across all requests.
    pub total_time_ms: f64,

    /// Number of cross-attention KV lookups that were served from cache.
    pub cache_hits: u64,

    /// Number of cross-attention KV lookups that were cache misses.
    pub cache_misses: u64,
}

impl BatchStats {
    /// Average wall-clock time per generated view in milliseconds.
    ///
    /// Returns `0.0` when no views have been generated yet.
    pub fn average_time_per_view_ms(&self) -> f64 {
        if self.total_views_generated > 0 {
            self.total_time_ms / self.total_views_generated as f64
        } else {
            0.0
        }
    }

    /// Fraction of cache lookups that were hits.
    ///
    /// Returns `0.0` when no lookups have been performed.
    pub fn cache_hit_rate(&self) -> f64 {
        let total = self.cache_hits + self.cache_misses;
        if total > 0 {
            self.cache_hits as f64 / total as f64
        } else {
            0.0
        }
    }
}

// ---------------------------------------------------------------------------
// BatchGenerator
// ---------------------------------------------------------------------------

/// Batch generator for processing multiple reference images.
///
/// Requests can be queued with [`BatchGenerator::queue`] and flushed in one
/// call via [`BatchGenerator::process_batch`], or processed immediately with
/// [`BatchGenerator::process_one`].
pub struct BatchGenerator {
    config: BatchGenConfig,
    kv_cache: Arc<KVCache>,
    request_queue: Mutex<Vec<GenerationRequest>>,
    stats: Mutex<BatchStats>,
}

impl BatchGenerator {
    /// Create a new batch generator with the given configuration.
    pub fn new(config: BatchGenConfig) -> Self {
        let kv_cache = Arc::new(KVCache::new(KVCacheConfig::default()));
        Self {
            config,
            kv_cache,
            request_queue: Mutex::new(Vec::new()),
            stats: Mutex::new(BatchStats::default()),
        }
    }

    // -----------------------------------------------------------------------
    // Queue management
    // -----------------------------------------------------------------------

    /// Add a request to the processing queue.
    ///
    /// Returns an error if the queue already holds [`BatchGenConfig::max_batch_size`]
    /// pending requests.
    pub fn queue(&self, request: GenerationRequest) -> Result<(), DiffusionError> {
        let mut queue = self.request_queue.lock().unwrap_or_else(|e| e.into_inner());
        if queue.len() >= self.config.max_batch_size {
            return Err(DiffusionError::InvalidConfig(format!(
                "queue is full: {} pending requests (max {})",
                queue.len(),
                self.config.max_batch_size
            )));
        }
        queue.push(request);
        Ok(())
    }

    /// Number of requests currently in the queue.
    pub fn queue_len(&self) -> usize {
        self.request_queue
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .len()
    }

    /// Remove all pending requests from the queue without processing them.
    pub fn clear_queue(&self) {
        self.request_queue
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
    }

    // -----------------------------------------------------------------------
    // Processing
    // -----------------------------------------------------------------------

    /// Process all queued requests and return results in queue order.
    ///
    /// The queue is drained before processing begins, so any requests added
    /// concurrently during processing will not be included in this batch.
    pub fn process_batch(&self) -> Result<Vec<GenerationResult>, DiffusionError> {
        // Drain the queue atomically before any processing.
        let requests: Vec<GenerationRequest> = {
            let mut queue = self.request_queue.lock().unwrap_or_else(|e| e.into_inner());
            std::mem::take(&mut *queue)
        };

        let mut results = Vec::with_capacity(requests.len());
        for req in requests {
            let result = self.process_one(req)?;
            results.push(result);
        }
        Ok(results)
    }

    /// Process a single request immediately, bypassing the queue.
    ///
    /// In the current placeholder implementation, output images are filled with
    /// a uniform grey value — real model weights are not required.
    ///
    /// # Errors
    ///
    /// Returns [`DiffusionError::InvalidConfig`] when:
    /// - `request.num_views == 0`
    /// - `request.num_views > max_views_per_request`
    pub fn process_one(
        &self,
        request: GenerationRequest,
    ) -> Result<GenerationResult, DiffusionError> {
        let start = std::time::Instant::now();

        // --- Validate inputs -----------------------------------------------
        if request.num_views == 0 {
            return Err(DiffusionError::InvalidConfig(
                "num_views must be > 0".into(),
            ));
        }
        if request.num_views > self.config.max_views_per_request {
            return Err(DiffusionError::InvalidConfig(format!(
                "num_views {} exceeds max {}",
                request.num_views, self.config.max_views_per_request
            )));
        }

        // --- Resolve per-request overrides ---------------------------------
        let _guidance = request.guidance_scale.unwrap_or(self.config.guidance_scale);
        let steps = request.num_steps.unwrap_or(self.config.num_steps);

        // Simulated per-step timing (placeholder).
        let simulated_ms_per_step = 10.0_f64;
        let view_time = steps as f64 * simulated_ms_per_step;

        // --- Generate placeholder views ------------------------------------
        let pixel_count = (request.image_width * request.image_height * 3) as usize;
        let views: Vec<GeneratedView> = (0..request.num_views)
            .map(|i| GeneratedView {
                view_index: i,
                image_data: vec![128u8; pixel_count],
                width: request.image_width,
                height: request.image_height,
                generation_time_ms: view_time,
            })
            .collect();

        let total_ms = start.elapsed().as_secs_f64() * 1000.0;

        // --- Check KV cache stats for reporting ----------------------------
        let cache_stats = self.kv_cache.stats();
        let num_cached_kv = cache_stats.hits as usize;

        // --- Update cumulative statistics ----------------------------------
        {
            let mut stats = self.stats.lock().unwrap_or_else(|e| e.into_inner());
            stats.total_requests += 1;
            stats.total_views_generated += request.num_views as u64;
            stats.total_time_ms += total_ms;
        }

        Ok(GenerationResult {
            id: request.id,
            views,
            total_time_ms: total_ms,
            num_cached_kv,
        })
    }

    // -----------------------------------------------------------------------
    // Statistics
    // -----------------------------------------------------------------------

    /// Return a snapshot of cumulative generation statistics.
    pub fn stats(&self) -> BatchStats {
        self.stats.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// Reset cumulative statistics to zero.
    pub fn reset_stats(&self) {
        let mut stats = self.stats.lock().unwrap_or_else(|e| e.into_inner());
        *stats = BatchStats::default();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_request(id: &str, num_views: usize) -> GenerationRequest {
        GenerationRequest {
            id: id.to_string(),
            reference_image: vec![128u8; 64 * 64 * 3],
            image_width: 64,
            image_height: 64,
            num_views,
            guidance_scale: None,
            num_steps: None,
            seed: None,
        }
    }

    // -----------------------------------------------------------------------
    // BatchGenConfig defaults
    // -----------------------------------------------------------------------

    #[test]
    fn test_batch_gen_config_default_max_batch_size() {
        assert_eq!(BatchGenConfig::default().max_batch_size, 4);
    }

    #[test]
    fn test_batch_gen_config_default_max_views_per_request() {
        assert_eq!(BatchGenConfig::default().max_views_per_request, 4);
    }

    #[test]
    fn test_batch_gen_config_default_guidance_scale() {
        let cfg = BatchGenConfig::default();
        assert!((cfg.guidance_scale - 3.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_batch_gen_config_default_num_steps() {
        assert_eq!(BatchGenConfig::default().num_steps, 20);
    }

    #[test]
    fn test_batch_gen_config_default_use_kv_cache() {
        assert!(BatchGenConfig::default().use_kv_cache);
    }

    #[test]
    fn test_batch_gen_config_default_synchronous() {
        assert!(BatchGenConfig::default().synchronous);
    }

    // -----------------------------------------------------------------------
    // GenerationRequest construction
    // -----------------------------------------------------------------------

    #[test]
    fn test_generation_request_can_be_constructed() {
        let req = GenerationRequest {
            id: "req-1".into(),
            reference_image: vec![0u8; 256 * 256 * 3],
            image_width: 256,
            image_height: 256,
            num_views: 4,
            guidance_scale: Some(5.0),
            num_steps: Some(30),
            seed: Some(42),
        };
        assert_eq!(req.id, "req-1");
        assert_eq!(req.num_views, 4);
    }

    // -----------------------------------------------------------------------
    // Queue length
    // -----------------------------------------------------------------------

    #[test]
    fn test_queue_len_starts_at_zero() {
        let gen = BatchGenerator::new(BatchGenConfig::default());
        assert_eq!(gen.queue_len(), 0);
    }

    #[test]
    fn test_queue_increases_queue_length() {
        let gen = BatchGenerator::new(BatchGenConfig::default());
        gen.queue(make_request("a", 1)).expect("queue failed");
        assert_eq!(gen.queue_len(), 1);
        gen.queue(make_request("b", 1)).expect("queue failed");
        assert_eq!(gen.queue_len(), 2);
    }

    #[test]
    fn test_clear_queue_resets_to_zero() {
        let gen = BatchGenerator::new(BatchGenConfig::default());
        gen.queue(make_request("a", 1)).expect("queue failed");
        gen.queue(make_request("b", 1)).expect("queue failed");
        assert_eq!(gen.queue_len(), 2);
        gen.clear_queue();
        assert_eq!(gen.queue_len(), 0);
    }

    // -----------------------------------------------------------------------
    // process_one — validation
    // -----------------------------------------------------------------------

    #[test]
    fn test_process_one_num_views_zero_returns_err() {
        let gen = BatchGenerator::new(BatchGenConfig::default());
        let req = make_request("x", 0);
        let result = gen.process_one(req);
        assert!(result.is_err(), "num_views=0 should be an error");
    }

    #[test]
    fn test_process_one_num_views_exceeds_max_returns_err() {
        let config = BatchGenConfig {
            max_views_per_request: 2,
            ..Default::default()
        };
        let gen = BatchGenerator::new(config);
        let req = make_request("x", 5);
        let result = gen.process_one(req);
        assert!(result.is_err(), "num_views > max should be an error");
    }

    // -----------------------------------------------------------------------
    // process_one — happy path
    // -----------------------------------------------------------------------

    #[test]
    fn test_process_one_valid_returns_ok() {
        let gen = BatchGenerator::new(BatchGenConfig::default());
        let req = make_request("ok", 2);
        let result = gen.process_one(req);
        assert!(result.is_ok(), "valid request should succeed: {:?}", result);
    }

    #[test]
    fn test_process_one_views_count_matches_num_views() {
        let gen = BatchGenerator::new(BatchGenConfig::default());
        let req = make_request("ok", 3);
        let result = gen.process_one(req).expect("generation failed");
        assert_eq!(result.views.len(), 3);
    }

    #[test]
    fn test_process_one_view_dimensions_match_request() {
        let gen = BatchGenerator::new(BatchGenConfig::default());
        let req = GenerationRequest {
            id: "dim".into(),
            reference_image: vec![0u8; 128 * 64 * 3],
            image_width: 128,
            image_height: 64,
            num_views: 1,
            guidance_scale: None,
            num_steps: None,
            seed: None,
        };
        let result = gen.process_one(req).expect("generation failed");
        let view = &result.views[0];
        assert_eq!(view.width, 128);
        assert_eq!(view.height, 64);
        assert_eq!(view.image_data.len(), 128 * 64 * 3);
    }

    #[test]
    fn test_process_one_view_indices_are_sequential() {
        let gen = BatchGenerator::new(BatchGenConfig::default());
        let req = make_request("idx", 3);
        let result = gen.process_one(req).expect("generation failed");
        for (i, view) in result.views.iter().enumerate() {
            assert_eq!(view.view_index, i);
        }
    }

    #[test]
    fn test_process_one_result_id_matches_request() {
        let gen = BatchGenerator::new(BatchGenConfig::default());
        let req = make_request("my-unique-id", 1);
        let result = gen.process_one(req).expect("generation failed");
        assert_eq!(result.id, "my-unique-id");
    }

    // -----------------------------------------------------------------------
    // Statistics
    // -----------------------------------------------------------------------

    #[test]
    fn test_stats_total_requests_increments() {
        let gen = BatchGenerator::new(BatchGenConfig::default());
        assert_eq!(gen.stats().total_requests, 0);
        gen.process_one(make_request("r1", 1)).expect("ok");
        assert_eq!(gen.stats().total_requests, 1);
        gen.process_one(make_request("r2", 2)).expect("ok");
        assert_eq!(gen.stats().total_requests, 2);
    }

    #[test]
    fn test_stats_total_views_generated_increments() {
        let gen = BatchGenerator::new(BatchGenConfig::default());
        gen.process_one(make_request("a", 3)).expect("ok");
        assert_eq!(gen.stats().total_views_generated, 3);
        gen.process_one(make_request("b", 2)).expect("ok");
        assert_eq!(gen.stats().total_views_generated, 5);
    }

    #[test]
    fn test_stats_cache_hit_rate_zero_for_no_hits() {
        let gen = BatchGenerator::new(BatchGenConfig::default());
        gen.process_one(make_request("r", 1)).expect("ok");
        let stats = gen.stats();
        assert!((stats.cache_hit_rate() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_stats_average_time_zero_for_empty() {
        let stats = BatchStats::default();
        assert!((stats.average_time_per_view_ms() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_reset_stats_clears_counters() {
        let gen = BatchGenerator::new(BatchGenConfig::default());
        gen.process_one(make_request("r", 2)).expect("ok");
        assert!(gen.stats().total_requests > 0);
        gen.reset_stats();
        assert_eq!(gen.stats().total_requests, 0);
        assert_eq!(gen.stats().total_views_generated, 0);
    }

    // -----------------------------------------------------------------------
    // process_batch
    // -----------------------------------------------------------------------

    #[test]
    fn test_process_batch_empty_queue_returns_empty_vec() {
        let gen = BatchGenerator::new(BatchGenConfig::default());
        let results = gen.process_batch().expect("process_batch failed");
        assert!(results.is_empty());
    }

    #[test]
    fn test_process_batch_processes_queued_request() {
        let gen = BatchGenerator::new(BatchGenConfig::default());
        gen.queue(make_request("q1", 2)).expect("queue failed");
        let results = gen.process_batch().expect("process_batch failed");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "q1");
        assert_eq!(results[0].views.len(), 2);
    }

    #[test]
    fn test_process_batch_drains_queue() {
        let gen = BatchGenerator::new(BatchGenConfig::default());
        gen.queue(make_request("q1", 1)).expect("queue failed");
        gen.queue(make_request("q2", 1)).expect("queue failed");
        let _ = gen.process_batch().expect("process_batch failed");
        assert_eq!(gen.queue_len(), 0);
    }

    // -----------------------------------------------------------------------
    // GenerationResult helpers
    // -----------------------------------------------------------------------

    #[test]
    fn test_all_views_generated_true_when_views_present() {
        let gen = BatchGenerator::new(BatchGenConfig::default());
        let result = gen.process_one(make_request("r", 1)).expect("ok");
        assert!(result.all_views_generated());
    }

    #[test]
    fn test_throughput_zero_when_time_zero() {
        let result = GenerationResult {
            id: "x".into(),
            views: vec![],
            total_time_ms: 0.0,
            num_cached_kv: 0,
        };
        assert!((result.throughput_views_per_sec() - 0.0).abs() < f64::EPSILON);
    }
}
