//! Streaming inference API for multi-view diffusion.
//!
//! Generates views one denoising step at a time, yielding partial results as
//! they become available. This is useful for progressive UI updates and
//! latency-sensitive pipelines.
//!
//! # Example
//!
//! ```rust
//! use oxigaf_diffusion::streaming::{StreamingInference, StreamingConfig};
//!
//! let config = StreamingConfig::default();
//! let si = StreamingInference::new(config);
//!
//! let steps: Vec<_> = si.step_iter(2).collect();
//! // 2 views × num_steps steps each
//! assert_eq!(steps.len(), 2 * StreamingConfig::default().num_steps);
//! ```

// ---------------------------------------------------------------------------
// StreamingConfig
// ---------------------------------------------------------------------------

/// Configuration for [`StreamingInference`].
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Output image width in pixels.
    ///
    /// Default: `256`.
    pub image_width: u32,

    /// Output image height in pixels.
    ///
    /// Default: `256`.
    pub image_height: u32,

    /// Classifier-free guidance scale.
    ///
    /// Default: `3.0`.
    pub guidance_scale: f32,

    /// Number of denoising steps per view.
    ///
    /// Default: `20`.
    pub num_steps: usize,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            image_width: 256,
            image_height: 256,
            guidance_scale: 3.0,
            num_steps: 20,
        }
    }
}

// ---------------------------------------------------------------------------
// StreamingStep
// ---------------------------------------------------------------------------

/// One step of streaming output from [`StreamingIterator`].
#[derive(Debug, Clone)]
pub struct StreamingStep {
    /// Zero-based view index this step belongs to.
    pub view_index: usize,

    /// Zero-based denoising step index within the current view.
    pub step_index: usize,

    /// Total number of denoising steps per view.
    pub total_steps: usize,

    /// Partial denoising result at this step (may be noisy).
    ///
    /// RGB bytes, flat row-major, length = `width × height × 3`.
    pub partial_image: Vec<u8>,

    /// `true` when this is the last step for the current view.
    pub is_final: bool,
}

impl StreamingStep {
    /// Fraction of denoising complete for the current view, in `[0.0, 1.0)`.
    ///
    /// The last step has `progress_fraction = (num_steps - 1) / num_steps`,
    /// not `1.0`, because `is_final` signals completion instead.
    ///
    /// Returns `0.0` when `total_steps == 0` (degenerate configuration).
    pub fn progress_fraction(&self) -> f32 {
        if self.total_steps == 0 {
            return 0.0;
        }
        self.step_index as f32 / self.total_steps as f32
    }
}

// ---------------------------------------------------------------------------
// StreamingInference
// ---------------------------------------------------------------------------

/// Streaming multi-view inference engine.
///
/// Produces an [`Iterator`] over [`StreamingStep`] values so callers can
/// process partial results as each denoising step completes.
pub struct StreamingInference {
    config: StreamingConfig,
}

impl StreamingInference {
    /// Create a new streaming inference engine with the given configuration.
    pub fn new(config: StreamingConfig) -> Self {
        Self { config }
    }

    /// Build an iterator that yields one [`StreamingStep`] per denoising step,
    /// covering all `num_views` views.
    ///
    /// Total steps yielded = `num_views × config.num_steps`.
    pub fn step_iter(&self, num_views: usize) -> StreamingIterator {
        StreamingIterator {
            config: self.config.clone(),
            num_views,
            current_view: 0,
            current_step: 0,
            seed: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// StreamingIterator
// ---------------------------------------------------------------------------

/// Iterator over streaming denoising steps for multi-view generation.
///
/// Produced by [`StreamingInference::step_iter`].
pub struct StreamingIterator {
    config: StreamingConfig,
    num_views: usize,
    current_view: usize,
    current_step: usize,
    /// Seed kept for future reproducibility support; currently unused.
    #[allow(dead_code)]
    seed: u64,
}

impl Iterator for StreamingIterator {
    type Item = StreamingStep;

    fn next(&mut self) -> Option<Self::Item> {
        // Termination condition: all views exhausted.
        if self.current_view >= self.num_views {
            return None;
        }

        // Guard against zero-step configuration — skip immediately.
        if self.config.num_steps == 0 {
            self.current_view += 1;
            return None;
        }

        let step_index = self.current_step;
        let is_final = step_index + 1 == self.config.num_steps;

        // Placeholder: gradient from black (step 0) to white (step num_steps-1).
        // Use saturating arithmetic to avoid overflow on edge-case configs.
        let pixel_value = (255_usize
            .saturating_mul(step_index)
            .saturating_div(self.config.num_steps.max(1))) as u8;

        let pixel_count =
            (self.config.image_width as usize) * (self.config.image_height as usize) * 3;

        let step = StreamingStep {
            view_index: self.current_view,
            step_index,
            total_steps: self.config.num_steps,
            partial_image: vec![pixel_value; pixel_count],
            is_final,
        };

        // Advance internal state.
        self.current_step += 1;
        if self.current_step >= self.config.num_steps {
            self.current_step = 0;
            self.current_view += 1;
        }

        Some(step)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // StreamingConfig defaults
    // -----------------------------------------------------------------------

    #[test]
    fn test_streaming_config_default_image_width() {
        assert_eq!(StreamingConfig::default().image_width, 256);
    }

    #[test]
    fn test_streaming_config_default_image_height() {
        assert_eq!(StreamingConfig::default().image_height, 256);
    }

    #[test]
    fn test_streaming_config_default_guidance_scale() {
        let cfg = StreamingConfig::default();
        assert!((cfg.guidance_scale - 3.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_streaming_config_default_num_steps() {
        assert_eq!(StreamingConfig::default().num_steps, 20);
    }

    // -----------------------------------------------------------------------
    // StreamingIterator step count
    // -----------------------------------------------------------------------

    #[test]
    fn test_step_iter_total_count_equals_views_times_steps() {
        let config = StreamingConfig::default();
        let si = StreamingInference::new(config.clone());
        let count = si.step_iter(2).count();
        assert_eq!(count, 2 * config.num_steps);
    }

    #[test]
    fn test_step_iter_zero_views_yields_nothing() {
        let si = StreamingInference::new(StreamingConfig::default());
        assert_eq!(si.step_iter(0).count(), 0);
    }

    #[test]
    fn test_step_iter_one_view_yields_num_steps_items() {
        let config = StreamingConfig {
            num_steps: 5,
            ..Default::default()
        };
        let si = StreamingInference::new(config);
        assert_eq!(si.step_iter(1).count(), 5);
    }

    // -----------------------------------------------------------------------
    // First and last step properties
    // -----------------------------------------------------------------------

    #[test]
    fn test_first_step_of_each_view_has_step_index_zero() {
        let config = StreamingConfig {
            num_steps: 4,
            ..Default::default()
        };
        let si = StreamingInference::new(config);
        let steps: Vec<_> = si.step_iter(3).collect();

        // First step of each view is at positions 0, 4, 8.
        for view_idx in 0..3 {
            let first_pos = view_idx * 4;
            assert_eq!(
                steps[first_pos].step_index, 0,
                "view {} first step should have step_index=0",
                view_idx
            );
        }
    }

    #[test]
    fn test_last_step_of_each_view_has_is_final_true() {
        let config = StreamingConfig {
            num_steps: 4,
            ..Default::default()
        };
        let si = StreamingInference::new(config);
        let steps: Vec<_> = si.step_iter(3).collect();

        // Last step of each view is at positions 3, 7, 11.
        for view_idx in 0..3 {
            let last_pos = view_idx * 4 + 3;
            assert!(
                steps[last_pos].is_final,
                "view {} last step should be is_final=true",
                view_idx
            );
        }
    }

    #[test]
    fn test_non_final_steps_have_is_final_false() {
        let config = StreamingConfig {
            num_steps: 5,
            ..Default::default()
        };
        let si = StreamingInference::new(config);
        let steps: Vec<_> = si.step_iter(1).collect();

        for step in steps.iter().take(4) {
            assert!(!step.is_final, "only the last step should be final");
        }
    }

    // -----------------------------------------------------------------------
    // progress_fraction
    // -----------------------------------------------------------------------

    #[test]
    fn test_progress_fraction_first_step_is_zero() {
        let config = StreamingConfig {
            num_steps: 10,
            ..Default::default()
        };
        let si = StreamingInference::new(config);
        let first = si
            .step_iter(1)
            .next()
            .expect("should have at least one step");
        assert!(
            (first.progress_fraction() - 0.0).abs() < f32::EPSILON,
            "step_index=0 should give progress_fraction=0.0"
        );
    }

    #[test]
    fn test_progress_fraction_last_step_is_n_minus_1_over_n() {
        let num_steps = 10;
        let config = StreamingConfig {
            num_steps,
            ..Default::default()
        };
        let si = StreamingInference::new(config);
        let last = si
            .step_iter(1)
            .last()
            .expect("should have at least one step");
        let expected = (num_steps - 1) as f32 / num_steps as f32;
        assert!(
            (last.progress_fraction() - expected).abs() < f32::EPSILON,
            "last step progress_fraction should be {expected}, got {}",
            last.progress_fraction()
        );
    }

    #[test]
    fn test_progress_fraction_zero_total_steps_returns_zero() {
        let step = StreamingStep {
            view_index: 0,
            step_index: 0,
            total_steps: 0,
            partial_image: vec![],
            is_final: true,
        };
        assert!((step.progress_fraction() - 0.0).abs() < f32::EPSILON);
    }

    // -----------------------------------------------------------------------
    // View indices
    // -----------------------------------------------------------------------

    #[test]
    fn test_view_indices_are_correct() {
        let config = StreamingConfig {
            num_steps: 3,
            ..Default::default()
        };
        let si = StreamingInference::new(config);
        let steps: Vec<_> = si.step_iter(2).collect();

        // First 3 steps: view 0.
        for step in steps.iter().take(3) {
            assert_eq!(step.view_index, 0);
        }
        // Last 3 steps: view 1.
        for step in steps.iter().skip(3) {
            assert_eq!(step.view_index, 1);
        }
    }

    // -----------------------------------------------------------------------
    // partial_image pixel gradient
    // -----------------------------------------------------------------------

    #[test]
    fn test_first_step_partial_image_is_black() {
        let config = StreamingConfig {
            num_steps: 10,
            image_width: 4,
            image_height: 4,
            ..Default::default()
        };
        let si = StreamingInference::new(config);
        let first = si.step_iter(1).next().expect("one step");
        // step 0 / 10 = 0 → pixel value 0
        assert_eq!(
            first.partial_image[0], 0,
            "first step should produce black pixels"
        );
    }

    #[test]
    fn test_partial_image_size_matches_config() {
        let config = StreamingConfig {
            num_steps: 5,
            image_width: 8,
            image_height: 8,
            ..Default::default()
        };
        let si = StreamingInference::new(config);
        for step in si.step_iter(1) {
            assert_eq!(
                step.partial_image.len(),
                8 * 8 * 3,
                "partial_image size should be width*height*3"
            );
        }
    }

    // -----------------------------------------------------------------------
    // total_steps in each StreamingStep
    // -----------------------------------------------------------------------

    #[test]
    fn test_total_steps_field_matches_config() {
        let config = StreamingConfig {
            num_steps: 7,
            ..Default::default()
        };
        let si = StreamingInference::new(config);
        for step in si.step_iter(1) {
            assert_eq!(step.total_steps, 7);
        }
    }
}
