//! Video-based frame-by-frame training using FLAME parameter sequences.
//!
//! This module provides [`VideoConfig`] and [`VideoFrameIterator`] for iterating
//! over per-frame FLAME parameters during training, with support for:
//!
//! - Stride-based frame subsampling
//! - Max-frame capping
//! - Looping / one-shot iteration
//! - Deterministic shuffle per epoch
//! - Batched frame access

use std::path::PathBuf;

use oxigaf_flame::{sequence::FlameSequence, FlameParams};
use rand::seq::SliceRandom as _;
use rand::SeedableRng;

use crate::TrainerError;

// ---------------------------------------------------------------------------
// VideoConfig
// ---------------------------------------------------------------------------

/// Configuration for video-based training.
///
/// Controls how [`VideoFrameIterator`] loads and iterates over a
/// [`FlameSequence`] during training.
#[derive(Debug, Clone)]
pub struct VideoConfig {
    /// Path to a JSON sequence file or a directory containing per-frame JSON
    /// parameter files.
    pub sequence_dir: PathBuf,

    /// Maximum number of frames to use after applying stride.
    /// `None` means use all available frames.
    pub max_frames: Option<usize>,

    /// Use every Nth frame (stride=1 uses all frames, stride=2 uses every other
    /// frame, etc.).
    pub frame_stride: usize,

    /// When `true`, after the last frame the iterator wraps back to the first
    /// frame instead of returning `None`.
    pub loop_sequence: bool,

    /// When `true`, the frame order is shuffled at construction time and again
    /// on each call to [`VideoFrameIterator::reset`].
    pub shuffle_frames: bool,

    /// Seed used for the shuffle RNG.  Fixed for reproducibility.
    pub shuffle_seed: u64,
}

impl Default for VideoConfig {
    fn default() -> Self {
        Self {
            sequence_dir: PathBuf::from("."),
            max_frames: None,
            frame_stride: 1,
            loop_sequence: false,
            shuffle_frames: false,
            shuffle_seed: 42,
        }
    }
}

impl VideoConfig {
    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// Returns [`TrainerError::InvalidConfig`] if any field is out of range.
    pub fn validate(&self) -> Result<(), TrainerError> {
        if self.frame_stride == 0 {
            return Err(TrainerError::InvalidConfig(
                "frame_stride must be >= 1".into(),
            ));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// FrameBatch
// ---------------------------------------------------------------------------

/// A collection of consecutive FLAME parameter frames produced by
/// [`VideoFrameIterator::next_batch`].
pub struct FrameBatch {
    /// Owned FLAME parameters for each frame in the batch.
    pub frames: Vec<FlameParams>,
    /// The original sequence indices for each frame (into the full sequence,
    /// not the iteration order).
    pub frame_indices: Vec<usize>,
}

// ---------------------------------------------------------------------------
// VideoFrameIterator
// ---------------------------------------------------------------------------

/// Manages frame iteration for video-based training.
///
/// Create one via [`VideoFrameIterator::from_sequence`] for in-memory use
/// (useful in tests) or via [`VideoFrameIterator::new`] to load from disk.
pub struct VideoFrameIterator {
    sequence: FlameSequence,
    /// Ordered list of sequence indices that will be iterated.
    frame_indices: Vec<usize>,
    current_pos: usize,
    config: VideoConfig,
}

impl VideoFrameIterator {
    // ---- Constructors -------------------------------------------------------

    /// Create from a `FlameSequence` that is already loaded.
    ///
    /// This is the primary constructor used in tests because it avoids any I/O.
    ///
    /// # Errors
    ///
    /// Returns [`TrainerError::InvalidConfig`] if `config` fails validation.
    pub fn from_sequence(
        sequence: FlameSequence,
        config: VideoConfig,
    ) -> Result<Self, TrainerError> {
        config.validate()?;

        let frame_indices = build_frame_indices(&sequence, &config);

        let mut iter = Self {
            sequence,
            frame_indices,
            current_pos: 0,
            config,
        };

        // Shuffle the initial iteration order if requested.
        if iter.config.shuffle_frames {
            iter.shuffle_indices();
        }

        Ok(iter)
    }

    /// Create from a [`VideoConfig`], loading the sequence from disk.
    ///
    /// Supports:
    /// - A `.json` file path (loaded with [`FlameSequence::from_json`]).
    /// - A directory path (loaded with [`FlameSequence::from_directory`] using
    ///   the pattern `"frame_{}.json"` and auto-detected frame count).
    ///
    /// # Errors
    ///
    /// Returns [`TrainerError::InvalidConfig`] if validation fails, or
    /// [`TrainerError::SequenceError`] if the sequence cannot be loaded.
    pub fn new(config: VideoConfig) -> Result<Self, TrainerError> {
        config.validate()?;

        let sequence = load_sequence_from_config(&config)?;
        Self::from_sequence(sequence, config)
    }

    // ---- Accessors ----------------------------------------------------------

    /// Total number of frames that will be iterated (after stride + max_frames
    /// are applied).
    #[must_use]
    pub fn total_frames(&self) -> usize {
        self.frame_indices.len()
    }

    /// Current iteration position (0-based index into the iteration order, not
    /// the raw sequence index).
    #[must_use]
    pub fn current_frame_index(&self) -> usize {
        self.current_pos
    }

    // ---- Iteration ----------------------------------------------------------

    /// Get the next frame's FLAME parameters.
    ///
    /// Returns `None` when the sequence is exhausted and
    /// [`VideoConfig::loop_sequence`] is `false`.
    ///
    /// When looping is enabled the iterator wraps back to position 0
    /// automatically (re-shuffling if configured).
    pub fn next_frame(&mut self) -> Option<Result<FlameParams, TrainerError>> {
        if self.frame_indices.is_empty() {
            return None;
        }

        // Wrap around if looping, otherwise signal exhaustion.
        if self.current_pos >= self.frame_indices.len() {
            if self.config.loop_sequence {
                self.reset();
            } else {
                return None;
            }
        }

        let seq_idx = self.frame_indices[self.current_pos];
        self.current_pos += 1;

        Some(
            self.sequence
                .get_frame(seq_idx)
                .cloned()
                .map_err(|e| TrainerError::SequenceError(e.to_string())),
        )
    }

    /// Collect the next `batch_size` frames into a [`FrameBatch`].
    ///
    /// Returns `None` when the sequence is exhausted and looping is disabled.
    /// A partial batch is returned if fewer than `batch_size` frames remain
    /// (non-looping) — the batch will contain all remaining frames.
    ///
    /// Returns `None` only when the very first call to `next_frame` returns
    /// `None` (i.e., the sequence was already exhausted before this call).
    pub fn next_batch(&mut self, batch_size: usize) -> Option<FrameBatch> {
        let mut frames = Vec::with_capacity(batch_size);
        let mut indices = Vec::with_capacity(batch_size);

        for _ in 0..batch_size {
            match self.next_frame() {
                None => break,
                Some(Err(_e)) => {
                    // Propagate the error by stopping; caller can call next_frame
                    // to observe the error directly.
                    break;
                }
                Some(Ok(params)) => {
                    let seq_idx = if self.current_pos > 0 {
                        self.frame_indices
                            .get(self.current_pos - 1)
                            .copied()
                            .unwrap_or(self.current_pos - 1)
                    } else {
                        0
                    };
                    indices.push(seq_idx);
                    frames.push(params);
                }
            }
        }

        if frames.is_empty() {
            None
        } else {
            Some(FrameBatch {
                frames,
                frame_indices: indices,
            })
        }
    }

    /// Reset the iterator to the beginning of the sequence.
    ///
    /// If [`VideoConfig::shuffle_frames`] is `true` the frame order is
    /// re-shuffled using the configured seed.
    pub fn reset(&mut self) {
        self.current_pos = 0;
        if self.config.shuffle_frames {
            self.shuffle_indices();
        }
    }

    /// Get the FLAME parameters at a specific iteration position.
    ///
    /// Unlike [`next_frame`] this does not advance the current position.
    ///
    /// # Errors
    ///
    /// Returns [`TrainerError::InvalidConfig`] if `pos` is out of bounds, or
    /// [`TrainerError::SequenceError`] if the underlying frame cannot be loaded.
    ///
    /// [`next_frame`]: VideoFrameIterator::next_frame
    pub fn frame_at(&mut self, pos: usize) -> Result<FlameParams, TrainerError> {
        let seq_idx = self.frame_indices.get(pos).copied().ok_or_else(|| {
            TrainerError::InvalidConfig(format!(
                "frame position {pos} is out of bounds (total_frames={})",
                self.frame_indices.len()
            ))
        })?;

        self.sequence
            .get_frame(seq_idx)
            .cloned()
            .map_err(|e| TrainerError::SequenceError(e.to_string()))
    }

    // ---- Private helpers ----------------------------------------------------

    /// Shuffle `frame_indices` in-place using the configured seed.
    fn shuffle_indices(&mut self) {
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.config.shuffle_seed);
        self.frame_indices.shuffle(&mut rng);
    }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Build the ordered list of sequence indices based on stride and max_frames.
fn build_frame_indices(sequence: &FlameSequence, config: &VideoConfig) -> Vec<usize> {
    let total = sequence.num_frames();

    // Step through raw indices with the requested stride.
    let strided: Vec<usize> = (0..total).step_by(config.frame_stride).collect();

    // Optionally cap the number of frames.
    match config.max_frames {
        Some(max) => strided.into_iter().take(max).collect(),
        None => strided,
    }
}

/// Load a [`FlameSequence`] from the path described in `config`.
///
/// - If `sequence_dir` points to a `.json` file, uses [`FlameSequence::from_json`].
/// - If `sequence_dir` is a directory, scans for `frame_*.json` files and uses
///   [`FlameSequence::from_directory`].
fn load_sequence_from_config(config: &VideoConfig) -> Result<FlameSequence, TrainerError> {
    let path = &config.sequence_dir;

    // Detect whether this is a single JSON file or a directory.
    if path.is_file() {
        // Single JSON sequence file.
        FlameSequence::from_json(path).map_err(|e| {
            TrainerError::SequenceError(format!(
                "Failed to load sequence from '{}': {e}",
                path.display()
            ))
        })
    } else if path.is_dir() {
        // Directory of per-frame JSON files.  Discover the frame count by
        // counting files matching `frame_*.json`.
        let (count, pattern) = discover_directory_sequence(path)?;
        FlameSequence::from_directory(path, &pattern, count, None).map_err(|e| {
            TrainerError::SequenceError(format!(
                "Failed to load sequence directory '{}': {e}",
                path.display()
            ))
        })
    } else {
        Err(TrainerError::SequenceError(format!(
            "Sequence path '{}' does not exist or is not accessible",
            path.display()
        )))
    }
}

/// Scan a directory for `frame_NNNN.json` files and return `(count, pattern)`.
fn discover_directory_sequence(dir: &std::path::Path) -> Result<(usize, String), TrainerError> {
    let read_dir = std::fs::read_dir(dir).map_err(|e| {
        TrainerError::SequenceError(format!("Cannot read directory '{}': {e}", dir.display()))
    })?;

    let mut count = 0usize;
    for entry in read_dir {
        let entry = entry
            .map_err(|e| TrainerError::SequenceError(format!("Directory entry error: {e}")))?;
        let file_name = entry.file_name();
        let name = file_name.to_string_lossy();
        if name.starts_with("frame_") && name.ends_with(".json") {
            count += 1;
        }
    }

    if count == 0 {
        return Err(TrainerError::SequenceError(format!(
            "No 'frame_*.json' files found in '{}'",
            dir.display()
        )));
    }

    Ok((count, "frame_{:04}.json".to_string()))
}

// ---------------------------------------------------------------------------
// TrainerError addition — SequenceError
//
// NOTE: This variant is declared here but must be added to the enum in lib.rs.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use oxigaf_flame::FlameParams;

    /// Build a minimal `FlameParams` for frame `i`.
    fn make_params(i: usize) -> FlameParams {
        FlameParams {
            shape: vec![i as f32 * 0.1],
            expression: vec![i as f32 * 0.2],
            pose: vec![0.0; 15],
            translation: [i as f32 * 0.01, 0.0, 0.0],
        }
    }

    /// Build a `FlameSequence` with `n` in-memory frames.
    fn make_sequence(n: usize) -> FlameSequence {
        let frames: Vec<FlameParams> = (0..n).map(make_params).collect();
        FlameSequence::from_memory(frames, Some(30.0))
    }

    /// Default config with no looping / shuffling.
    fn default_config() -> VideoConfig {
        VideoConfig::default()
    }

    // ---- 1. VideoConfig default values -------------------------------------

    #[test]
    fn video_config_default_values() {
        let cfg = VideoConfig::default();
        assert_eq!(cfg.frame_stride, 1);
        assert_eq!(cfg.max_frames, None);
        assert!(!cfg.loop_sequence);
        assert!(!cfg.shuffle_frames);
        assert_eq!(cfg.shuffle_seed, 42);
    }

    // ---- 2. validate rejects frame_stride == 0 -----------------------------

    #[test]
    fn validate_rejects_zero_stride() {
        let mut cfg = default_config();
        cfg.frame_stride = 0;
        assert!(
            cfg.validate().is_err(),
            "frame_stride=0 should fail validation"
        );
    }

    // ---- 3. validate accepts valid config ----------------------------------

    #[test]
    fn validate_accepts_valid_config() {
        let cfg = VideoConfig {
            frame_stride: 2,
            max_frames: Some(10),
            ..Default::default()
        };
        assert!(cfg.validate().is_ok());
    }

    // ---- 4. total_frames with stride=1 -------------------------------------

    #[test]
    fn total_frames_stride_1_all_frames() {
        let seq = make_sequence(10);
        let cfg = VideoConfig {
            frame_stride: 1,
            ..Default::default()
        };
        let iter = VideoFrameIterator::from_sequence(seq, cfg).expect("from_sequence failed");
        assert_eq!(iter.total_frames(), 10);
    }

    // ---- 5. total_frames with stride=2 -------------------------------------

    #[test]
    fn total_frames_stride_2_half_frames() {
        let seq = make_sequence(10);
        let cfg = VideoConfig {
            frame_stride: 2,
            ..Default::default()
        };
        let iter = VideoFrameIterator::from_sequence(seq, cfg).expect("from_sequence failed");
        // 0, 2, 4, 6, 8 → 5 frames
        assert_eq!(iter.total_frames(), 5);
    }

    // ---- 6. total_frames with max_frames cap --------------------------------

    #[test]
    fn total_frames_max_frames_caps_at_5() {
        let seq = make_sequence(20);
        let cfg = VideoConfig {
            max_frames: Some(5),
            ..Default::default()
        };
        let iter = VideoFrameIterator::from_sequence(seq, cfg).expect("from_sequence failed");
        assert_eq!(iter.total_frames(), 5);
    }

    // ---- 7. next_frame returns None after exhaustion (no loop) --------------

    #[test]
    fn next_frame_returns_none_after_exhaustion() {
        let seq = make_sequence(3);
        let cfg = default_config();
        let mut iter = VideoFrameIterator::from_sequence(seq, cfg).expect("from_sequence failed");

        // Consume all 3 frames.
        for _ in 0..3 {
            assert!(iter.next_frame().is_some());
        }
        // Next call must return None.
        assert!(
            iter.next_frame().is_none(),
            "expected None after all frames consumed"
        );
    }

    // ---- 8. next_frame loops when loop_sequence = true ---------------------

    #[test]
    fn next_frame_loops_when_configured() {
        let seq = make_sequence(3);
        let cfg = VideoConfig {
            loop_sequence: true,
            ..Default::default()
        };
        let mut iter = VideoFrameIterator::from_sequence(seq, cfg).expect("from_sequence failed");

        // Consume 3 frames (one full pass).
        for _ in 0..3 {
            assert!(iter.next_frame().is_some());
        }
        // 4th call should wrap and return the first frame again.
        let wrapped = iter.next_frame();
        assert!(
            wrapped.is_some(),
            "expected loop to wrap and return a frame"
        );
        let params = wrapped
            .expect("is_some checked above")
            .expect("frame should load ok");
        // Shape[0] of frame 0 == 0.0.
        assert!(
            (params.shape[0] - 0.0_f32).abs() < 1e-5,
            "wrapped frame should be frame 0 (shape[0]==0.0)"
        );
    }

    // ---- 9. reset restarts iteration ----------------------------------------

    #[test]
    fn reset_restarts_iteration() {
        let seq = make_sequence(5);
        let cfg = default_config();
        let mut iter = VideoFrameIterator::from_sequence(seq, cfg).expect("from_sequence failed");

        // Advance 3 frames.
        for _ in 0..3 {
            iter.next_frame();
        }
        assert_eq!(iter.current_frame_index(), 3);

        iter.reset();
        assert_eq!(iter.current_frame_index(), 0);

        // After reset, next_frame should return the first frame.
        let first = iter
            .next_frame()
            .expect("should have a frame after reset")
            .expect("frame load ok");
        assert!(
            (first.shape[0] - 0.0_f32).abs() < 1e-5,
            "after reset, frame 0 should have shape[0]==0.0"
        );
    }

    // ---- 10. frame_at(0) returns first frame --------------------------------

    #[test]
    fn frame_at_zero_returns_first_frame() {
        let seq = make_sequence(5);
        let cfg = default_config();
        let mut iter = VideoFrameIterator::from_sequence(seq, cfg).expect("from_sequence failed");

        let first = iter.frame_at(0).expect("frame_at(0) should succeed");
        assert!(
            (first.shape[0] - 0.0_f32).abs() < 1e-5,
            "frame 0 should have shape[0]==0.0"
        );
    }

    // ---- 11. next_batch(3) returns FrameBatch with 3 frames -----------------

    #[test]
    fn next_batch_returns_batch_of_3() {
        let seq = make_sequence(10);
        let cfg = default_config();
        let mut iter = VideoFrameIterator::from_sequence(seq, cfg).expect("from_sequence failed");

        let batch = iter
            .next_batch(3)
            .expect("next_batch should return a batch");
        assert_eq!(batch.frames.len(), 3);
        assert_eq!(batch.frame_indices.len(), 3);
    }

    // ---- 12. next_batch returns None when sequence exhausted ----------------

    #[test]
    fn next_batch_returns_none_when_exhausted() {
        let seq = make_sequence(3);
        let cfg = default_config();
        let mut iter = VideoFrameIterator::from_sequence(seq, cfg).expect("from_sequence failed");

        // Consume all.
        let _ = iter.next_batch(3);
        // Now exhausted.
        let result = iter.next_batch(3);
        assert!(
            result.is_none(),
            "next_batch should return None when sequence is exhausted"
        );
    }

    // ---- Bonus: current_frame_index advances correctly ----------------------

    #[test]
    fn current_frame_index_advances() {
        let seq = make_sequence(5);
        let cfg = default_config();
        let mut iter = VideoFrameIterator::from_sequence(seq, cfg).expect("from_sequence failed");

        assert_eq!(iter.current_frame_index(), 0);
        iter.next_frame();
        assert_eq!(iter.current_frame_index(), 1);
        iter.next_frame();
        assert_eq!(iter.current_frame_index(), 2);
    }

    // ---- Bonus: shuffle changes order but not total count -------------------

    #[test]
    fn shuffle_changes_order_preserves_count() {
        let seq_ordered = make_sequence(10);
        let seq_shuffled = make_sequence(10);

        let cfg_ordered = VideoConfig {
            shuffle_frames: false,
            ..Default::default()
        };
        let cfg_shuffled = VideoConfig {
            shuffle_frames: true,
            shuffle_seed: 123,
            ..Default::default()
        };

        let iter_ordered = VideoFrameIterator::from_sequence(seq_ordered, cfg_ordered)
            .expect("from_sequence ordered failed");
        let iter_shuffled = VideoFrameIterator::from_sequence(seq_shuffled, cfg_shuffled)
            .expect("from_sequence shuffled failed");

        // Same count.
        assert_eq!(iter_ordered.total_frames(), iter_shuffled.total_frames());

        // The raw index lists differ (shuffled ≠ 0,1,2,...).
        assert_ne!(
            iter_ordered.frame_indices, iter_shuffled.frame_indices,
            "shuffle should change the iteration order"
        );
    }

    // ---- Bonus: frame_at is non-destructive to current_pos ------------------

    #[test]
    fn frame_at_does_not_advance_position() {
        let seq = make_sequence(5);
        let cfg = default_config();
        let mut iter = VideoFrameIterator::from_sequence(seq, cfg).expect("from_sequence failed");

        let _ = iter.frame_at(2).expect("frame_at(2) ok");
        // current_pos must still be 0.
        assert_eq!(
            iter.current_frame_index(),
            0,
            "frame_at must not advance current_pos"
        );
    }
}
