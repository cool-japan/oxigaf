//! Higher-level multi-clip animation orchestration layer.
//!
//! Builds on top of [`crate::expression_animation`] to provide:
//! - Multiple named animation clips playing simultaneously on **layers**
//! - A global timeline with per-clip start offsets
//! - Layer blending (additive and override modes)
//! - Timeline markers (named time positions)
//! - Per-layer weight/mute control
//!
//! # Quick Start
//!
//! ```rust
//! use oxigaf_flame::expression_animation::{
//!     AnimationClip, EasingFunction, ExpressionKeyframe, ExpressionTimeline, LoopMode,
//! };
//! use oxigaf_flame::timeline::{
//!     AnimationTimeline, BlendMode, TimelineLayer, TimelineMarker, timeline_from_clip,
//! };
//!
//! // Build a clip
//! let names = vec!["smile".to_string()];
//! let clip = AnimationClip::new("base", names.clone());
//!
//! // Wrap it in a single-layer timeline
//! let tl = timeline_from_clip(clip, 1);
//! let weights = tl.evaluate().expect("evaluate");
//! assert_eq!(weights.len(), 1);
//! ```

use crate::expression_animation::{AnimationClip, LoopMode};

// ---------------------------------------------------------------------------
// TimelineError
// ---------------------------------------------------------------------------

/// Errors that can occur in the animation timeline layer.
#[derive(Debug, thiserror::Error)]
pub enum TimelineError {
    /// A layer with the given name was not found.
    #[error("Layer not found: {0}")]
    LayerNotFound(String),
    /// A clip with the given name was not found.
    #[error("Clip not found: {0}")]
    ClipNotFound(String),
    /// A marker with the given name was not found.
    #[error("Marker not found: {0}")]
    MarkerNotFound(String),
    /// A layer with the given name already exists.
    #[error("Duplicate layer: {0}")]
    DuplicateLayer(String),
    /// The supplied time value is invalid.
    #[error("Invalid time: {0}")]
    InvalidTime(f32),
    /// Blend weight is outside `[0, 1]`.
    #[error("Blend weight out of range: {0} (must be [0,1])")]
    BlendWeightOutOfRange(f32),
    /// Expression weight vector dimension mismatch.
    #[error("Expression dimension mismatch: expected {expected}, got {got}")]
    DimMismatch { expected: usize, got: usize },
}

// ---------------------------------------------------------------------------
// BlendMode
// ---------------------------------------------------------------------------

/// How a layer's evaluated weights are composited on top of lower layers.
#[derive(Debug, Clone, PartialEq)]
pub enum BlendMode {
    /// This layer's output replaces lower layers (lerp by weight).
    Override,
    /// This layer's output is added to lower layers (scaled by weight).
    Additive,
    /// Weighted blend — identical to Additive in the implemented algorithm.
    Weighted,
}

// ---------------------------------------------------------------------------
// TimelineLayer
// ---------------------------------------------------------------------------

/// A single named animation layer within an [`AnimationTimeline`].
#[derive(Debug, Clone)]
pub struct TimelineLayer {
    /// Unique name for this layer.
    pub name: String,
    /// The animation clip to play on this layer.
    pub clip: AnimationClip,
    /// Global timeline time (in seconds) at which the clip begins playing.
    pub start_offset: f32,
    /// How this layer's output is composited with lower layers.
    pub blend_mode: BlendMode,
    /// Blend weight in `[0, 1]`.
    pub weight: f32,
    /// When `true` the layer contributes nothing to the final blend.
    pub muted: bool,
}

impl TimelineLayer {
    /// Create a new layer with sensible defaults (`Override`, `weight = 1.0`, not muted).
    #[must_use]
    pub fn new(name: impl Into<String>, clip: AnimationClip) -> Self {
        Self {
            name: name.into(),
            clip,
            start_offset: 0.0,
            blend_mode: BlendMode::Override,
            weight: 1.0,
            muted: false,
        }
    }

    /// Set the start offset and return `self`.
    #[must_use]
    pub fn with_start_offset(mut self, offset: f32) -> Self {
        self.start_offset = offset;
        self
    }

    /// Set the blend mode and return `self`.
    #[must_use]
    pub fn with_blend_mode(mut self, mode: BlendMode) -> Self {
        self.blend_mode = mode;
        self
    }

    /// Set the blend weight and return `self`.
    #[must_use]
    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = weight;
        self
    }

    /// Set whether the layer is muted and return `self`.
    #[must_use]
    pub fn muted(mut self, muted: bool) -> Self {
        self.muted = muted;
        self
    }
}

// ---------------------------------------------------------------------------
// TimelineMarker
// ---------------------------------------------------------------------------

/// A named time position within an [`AnimationTimeline`].
#[derive(Debug, Clone)]
pub struct TimelineMarker {
    /// Unique name for this marker.
    pub name: String,
    /// Position in global timeline seconds.
    pub time: f32,
    /// Optional human-readable description.
    pub label: Option<String>,
}

// ---------------------------------------------------------------------------
// AnimationTimeline
// ---------------------------------------------------------------------------

/// A multi-layer animation timeline with global time, markers, and blending.
///
/// Layers are stored in priority order — lower index = lower priority. The
/// highest-priority layer is the last in the `layers` vec. When iterating
/// for blending we go from index 0 upward, which means higher-indexed layers
/// compose on top of lower-indexed ones.
pub struct AnimationTimeline {
    /// Layers ordered from lowest priority (index 0) to highest (last).
    layers: Vec<TimelineLayer>,
    /// Markers sorted by ascending `time`.
    markers: Vec<TimelineMarker>,
    /// Current global playhead position in seconds.
    global_time: f32,
    /// Expected length of the expression weight vectors produced by each layer.
    expr_dims: usize,
}

impl AnimationTimeline {
    /// Create a new, empty timeline for expressions with `expr_dims` dimensions.
    #[must_use]
    pub fn new(expr_dims: usize) -> Self {
        Self {
            layers: Vec::new(),
            markers: Vec::new(),
            global_time: 0.0,
            expr_dims,
        }
    }

    /// Add a layer at the top (highest priority).
    ///
    /// # Errors
    ///
    /// Returns [`TimelineError::DuplicateLayer`] if a layer with the same name
    /// already exists.
    pub fn add_layer(&mut self, layer: TimelineLayer) -> Result<(), TimelineError> {
        if self.layers.iter().any(|l| l.name == layer.name) {
            return Err(TimelineError::DuplicateLayer(layer.name.clone()));
        }
        self.layers.push(layer);
        Ok(())
    }

    /// Remove and return the layer with the given name.
    ///
    /// # Errors
    ///
    /// Returns [`TimelineError::LayerNotFound`] if no layer with that name
    /// exists.
    pub fn remove_layer(&mut self, name: &str) -> Result<TimelineLayer, TimelineError> {
        let pos = self
            .layers
            .iter()
            .position(|l| l.name == name)
            .ok_or_else(|| TimelineError::LayerNotFound(name.to_string()))?;
        Ok(self.layers.remove(pos))
    }

    /// Return a mutable reference to the layer with the given name, or `None`.
    pub fn layer_mut(&mut self, name: &str) -> Option<&mut TimelineLayer> {
        self.layers.iter_mut().find(|l| l.name == name)
    }

    /// Set the blend weight of a layer.
    ///
    /// The weight is clamped to `[0, 1]` before being applied.
    ///
    /// # Errors
    ///
    /// Returns [`TimelineError::LayerNotFound`] if no layer with that name
    /// exists.
    pub fn set_layer_weight(&mut self, name: &str, weight: f32) -> Result<(), TimelineError> {
        let layer = self
            .layer_mut(name)
            .ok_or_else(|| TimelineError::LayerNotFound(name.to_string()))?;
        layer.weight = weight.clamp(0.0, 1.0);
        Ok(())
    }

    /// Mute or unmute a layer.
    ///
    /// # Errors
    ///
    /// Returns [`TimelineError::LayerNotFound`] if no layer with that name
    /// exists.
    pub fn set_layer_muted(&mut self, name: &str, muted: bool) -> Result<(), TimelineError> {
        let layer = self
            .layer_mut(name)
            .ok_or_else(|| TimelineError::LayerNotFound(name.to_string()))?;
        layer.muted = muted;
        Ok(())
    }

    /// Insert a marker in sorted order by time.
    pub fn add_marker(&mut self, marker: TimelineMarker) {
        let pos = self.markers.partition_point(|m| m.time <= marker.time);
        self.markers.insert(pos, marker);
    }

    /// Move the global playhead to the position of the named marker.
    ///
    /// # Errors
    ///
    /// Returns [`TimelineError::MarkerNotFound`] if no marker with that name
    /// exists.
    pub fn seek_to_marker(&mut self, name: &str) -> Result<(), TimelineError> {
        let time = self
            .markers
            .iter()
            .find(|m| m.name == name)
            .map(|m| m.time)
            .ok_or_else(|| TimelineError::MarkerNotFound(name.to_string()))?;
        self.global_time = time;
        Ok(())
    }

    /// Advance the global playhead by `dt` seconds.
    pub fn advance(&mut self, dt: f32) {
        self.global_time += dt;
    }

    /// Set the global playhead to an absolute time.
    pub fn set_time(&mut self, t: f32) {
        self.global_time = t;
    }

    /// Return the current global playhead position in seconds.
    #[must_use]
    pub fn global_time(&self) -> f32 {
        self.global_time
    }

    /// Evaluate all layers and produce blended expression weights.
    ///
    /// Returns a `Vec<f32>` of length `expr_dims`.
    ///
    /// # Blending algorithm
    ///
    /// 1. Start with a zero vector of length `expr_dims`.
    /// 2. For each layer (lowest to highest priority, i.e., index 0 upward),
    ///    if not muted:
    ///    - Compute `local_time = global_time - start_offset`.
    ///    - If `local_time < 0` **and** the clip is `LoopMode::Once`, skip the
    ///      layer (clip hasn't started yet).
    ///    - Evaluate the clip at `local_time` → `layer_weights`.
    ///    - Apply blend mode:
    ///      - `Override`:  `result = lerp(result, layer_weights, weight)`
    ///      - `Additive`:  `result = result + layer_weights * weight`
    ///      - `Weighted`:  same as Additive
    /// 3. Clamp final result element-wise to `[-3, 3]`.
    ///
    /// # Errors
    ///
    /// Returns [`TimelineError::DimMismatch`] if a layer produces a weight
    /// vector of unexpected length.
    pub fn evaluate(&self) -> Result<Vec<f32>, TimelineError> {
        let mut result = vec![0.0_f32; self.expr_dims];

        for layer in &self.layers {
            if layer.muted {
                continue;
            }

            let local_time = self.global_time - layer.start_offset;

            // Skip Once-mode clips that haven't started yet.
            if local_time < 0.0 && layer.clip.loop_mode == LoopMode::Once {
                continue;
            }

            // For non-Once modes with negative local time, treat as 0.0.
            let eval_time = local_time.max(0.0);

            let layer_weights = layer.clip.evaluate(eval_time);

            if layer_weights.len() != self.expr_dims {
                return Err(TimelineError::DimMismatch {
                    expected: self.expr_dims,
                    got: layer_weights.len(),
                });
            }

            let w = layer.weight;
            match layer.blend_mode {
                BlendMode::Override => {
                    // lerp(result, layer_weights, weight)
                    for (r, &lw) in result.iter_mut().zip(layer_weights.iter()) {
                        *r = *r + (lw - *r) * w;
                    }
                }
                BlendMode::Additive | BlendMode::Weighted => {
                    for (r, &lw) in result.iter_mut().zip(layer_weights.iter()) {
                        *r += lw * w;
                    }
                }
            }
        }

        // Clamp to FLAME expression range [-3, 3].
        for r in &mut result {
            *r = r.clamp(-3.0, 3.0);
        }

        Ok(result)
    }

    /// Return the number of layers.
    #[must_use]
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Return all markers sorted by time.
    #[must_use]
    pub fn markers(&self) -> &[TimelineMarker] {
        &self.markers
    }

    /// Return the total duration of the timeline.
    ///
    /// Computed as the maximum over all layers of
    /// `start_offset + clip.effective_duration()`.  Returns `0.0` if there are
    /// no layers.
    #[must_use]
    pub fn total_duration(&self) -> f32 {
        self.layers
            .iter()
            .map(|l| l.start_offset + l.clip.effective_duration())
            .fold(0.0_f32, f32::max)
    }

    /// Return the next marker after the current global time, or `None`.
    #[must_use]
    pub fn next_marker(&self) -> Option<&TimelineMarker> {
        self.markers.iter().find(|m| m.time > self.global_time)
    }

    /// Return the last marker at or before the current global time, or `None`.
    #[must_use]
    pub fn prev_marker(&self) -> Option<&TimelineMarker> {
        self.markers
            .iter()
            .rev()
            .find(|m| m.time <= self.global_time)
    }
}

// ---------------------------------------------------------------------------
// Convenience constructor
// ---------------------------------------------------------------------------

/// Create a single-layer `AnimationTimeline` from an `AnimationClip`.
///
/// The layer is named `"default"`, uses `BlendMode::Override` at full weight,
/// and has no start offset.
#[must_use]
pub fn timeline_from_clip(clip: AnimationClip, expr_dims: usize) -> AnimationTimeline {
    let mut tl = AnimationTimeline::new(expr_dims);
    let layer = TimelineLayer::new("default", clip);
    // Ignore error — we just created an empty timeline so there can be no duplicate.
    let _ = tl.add_layer(layer);
    tl
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::expression_animation::{
        AnimationClip, EasingFunction, ExpressionKeyframe, ExpressionTimeline, LoopMode,
    };

    // -----------------------------------------------------------------------
    // Helper
    // -----------------------------------------------------------------------

    /// Build a minimal looping clip whose weights are all `value` at every time.
    fn make_clip(expr_dims: usize, value: f32, loop_mode: LoopMode) -> AnimationClip {
        let names: Vec<String> = (0..expr_dims).map(|i| format!("dim_{i}")).collect();
        let mut timeline = ExpressionTimeline::new(names.clone());
        // ExpressionTimeline::new already inserts a zero keyframe at t=0.
        // Add one at t=1 with the desired value.
        let kf1 = ExpressionKeyframe {
            time: 1.0,
            weights: vec![value; expr_dims],
            easing: EasingFunction::Linear,
        };
        timeline.add_keyframe(kf1).unwrap();
        AnimationClip {
            name: "test_clip".to_string(),
            timeline,
            loop_mode,
            playback_rate: 1.0,
        }
    }

    /// A clip that produces all-zeros at every time (neutral face).
    fn make_neutral_clip(expr_dims: usize) -> AnimationClip {
        make_clip(expr_dims, 0.0, LoopMode::Loop)
    }

    /// A clip that produces all-`value` weights at t=1 with `LoopMode::Once`.
    fn make_once_clip(expr_dims: usize, value: f32) -> AnimationClip {
        make_clip(expr_dims, value, LoopMode::Once)
    }

    // -----------------------------------------------------------------------
    // Test 1: new() creates empty timeline
    // -----------------------------------------------------------------------

    #[test]
    fn test_new_empty_timeline() {
        let tl = AnimationTimeline::new(5);
        assert_eq!(tl.num_layers(), 0);
        assert_eq!(tl.markers().len(), 0);
        assert!((tl.global_time() - 0.0).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // Test 2: add_layer adds layers in order
    // -----------------------------------------------------------------------

    #[test]
    fn test_add_layer_order() {
        let mut tl = AnimationTimeline::new(2);
        let clip_a = make_neutral_clip(2);
        let clip_b = make_neutral_clip(2);
        tl.add_layer(TimelineLayer::new("layer_a", clip_a)).unwrap();
        tl.add_layer(TimelineLayer::new("layer_b", clip_b)).unwrap();
        assert_eq!(tl.num_layers(), 2);
        assert_eq!(tl.layers[0].name, "layer_a");
        assert_eq!(tl.layers[1].name, "layer_b");
    }

    // -----------------------------------------------------------------------
    // Test 3: add_layer returns DuplicateLayer on duplicate name
    // -----------------------------------------------------------------------

    #[test]
    fn test_add_layer_duplicate_error() {
        let mut tl = AnimationTimeline::new(2);
        let clip_a = make_neutral_clip(2);
        let clip_b = make_neutral_clip(2);
        tl.add_layer(TimelineLayer::new("dup", clip_a)).unwrap();
        let result = tl.add_layer(TimelineLayer::new("dup", clip_b));
        assert!(matches!(result, Err(TimelineError::DuplicateLayer(n)) if n == "dup"));
    }

    // -----------------------------------------------------------------------
    // Test 4: remove_layer removes and returns the layer
    // -----------------------------------------------------------------------

    #[test]
    fn test_remove_layer_success() {
        let mut tl = AnimationTimeline::new(2);
        let clip = make_neutral_clip(2);
        tl.add_layer(TimelineLayer::new("my_layer", clip)).unwrap();
        assert_eq!(tl.num_layers(), 1);
        let removed = tl.remove_layer("my_layer").unwrap();
        assert_eq!(removed.name, "my_layer");
        assert_eq!(tl.num_layers(), 0);
    }

    // -----------------------------------------------------------------------
    // Test 5: remove_layer returns LayerNotFound for missing name
    // -----------------------------------------------------------------------

    #[test]
    fn test_remove_layer_not_found() {
        let mut tl = AnimationTimeline::new(2);
        let result = tl.remove_layer("ghost");
        assert!(matches!(result, Err(TimelineError::LayerNotFound(n)) if n == "ghost"));
    }

    // -----------------------------------------------------------------------
    // Test 6: set_layer_weight clamps weight to [0, 1]
    // -----------------------------------------------------------------------

    #[test]
    fn test_set_layer_weight_clamps() {
        let mut tl = AnimationTimeline::new(2);
        let clip = make_neutral_clip(2);
        tl.add_layer(TimelineLayer::new("l", clip)).unwrap();

        // Clamp above 1
        tl.set_layer_weight("l", 5.0).unwrap();
        assert!((tl.layers[0].weight - 1.0).abs() < 1e-6);

        // Clamp below 0
        tl.set_layer_weight("l", -2.0).unwrap();
        assert!((tl.layers[0].weight - 0.0).abs() < 1e-6);

        // Normal value stays
        tl.set_layer_weight("l", 0.75).unwrap();
        assert!((tl.layers[0].weight - 0.75).abs() < 1e-4);
    }

    // -----------------------------------------------------------------------
    // Test 7: set_layer_muted toggles mute
    // -----------------------------------------------------------------------

    #[test]
    fn test_set_layer_muted() {
        let mut tl = AnimationTimeline::new(2);
        let clip = make_neutral_clip(2);
        tl.add_layer(TimelineLayer::new("l", clip)).unwrap();

        assert!(!tl.layers[0].muted);
        tl.set_layer_muted("l", true).unwrap();
        assert!(tl.layers[0].muted);
        tl.set_layer_muted("l", false).unwrap();
        assert!(!tl.layers[0].muted);
    }

    // -----------------------------------------------------------------------
    // Test 8: add_marker inserts sorted by time
    // -----------------------------------------------------------------------

    #[test]
    fn test_add_marker_sorted() {
        let mut tl = AnimationTimeline::new(2);
        tl.add_marker(TimelineMarker {
            name: "c".to_string(),
            time: 3.0,
            label: None,
        });
        tl.add_marker(TimelineMarker {
            name: "a".to_string(),
            time: 1.0,
            label: None,
        });
        tl.add_marker(TimelineMarker {
            name: "b".to_string(),
            time: 2.0,
            label: None,
        });

        let times: Vec<f32> = tl.markers().iter().map(|m| m.time).collect();
        assert_eq!(times, vec![1.0, 2.0, 3.0]);
        assert_eq!(tl.markers()[0].name, "a");
        assert_eq!(tl.markers()[1].name, "b");
        assert_eq!(tl.markers()[2].name, "c");
    }

    // -----------------------------------------------------------------------
    // Test 9: seek_to_marker moves global_time to marker's time
    // -----------------------------------------------------------------------

    #[test]
    fn test_seek_to_marker_success() {
        let mut tl = AnimationTimeline::new(2);
        tl.add_marker(TimelineMarker {
            name: "intro".to_string(),
            time: 2.5,
            label: None,
        });
        tl.seek_to_marker("intro").unwrap();
        assert!((tl.global_time() - 2.5).abs() < 1e-4);
    }

    // -----------------------------------------------------------------------
    // Test 10: seek_to_marker returns MarkerNotFound
    // -----------------------------------------------------------------------

    #[test]
    fn test_seek_to_marker_not_found() {
        let mut tl = AnimationTimeline::new(2);
        let result = tl.seek_to_marker("nonexistent");
        assert!(matches!(result, Err(TimelineError::MarkerNotFound(n)) if n == "nonexistent"));
    }

    // -----------------------------------------------------------------------
    // Test 11: advance increases global_time
    // -----------------------------------------------------------------------

    #[test]
    fn test_advance_increases_time() {
        let mut tl = AnimationTimeline::new(2);
        tl.advance(1.5);
        assert!((tl.global_time() - 1.5).abs() < 1e-4);
        tl.advance(0.5);
        assert!((tl.global_time() - 2.0).abs() < 1e-4);
    }

    // -----------------------------------------------------------------------
    // Test 12: set_time sets global_time exactly
    // -----------------------------------------------------------------------

    #[test]
    fn test_set_time_exact() {
        let mut tl = AnimationTimeline::new(2);
        tl.set_time(7.3);
        assert!((tl.global_time() - 7.3).abs() < 1e-4);
        tl.set_time(0.0);
        assert!((tl.global_time() - 0.0).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // Test 13: evaluate with empty timeline returns zero vector
    // -----------------------------------------------------------------------

    #[test]
    fn test_evaluate_empty_timeline_returns_zeros() {
        let tl = AnimationTimeline::new(4);
        let result = tl.evaluate().unwrap();
        assert_eq!(result.len(), 4);
        for &v in &result {
            assert!((v - 0.0).abs() < 1e-6, "expected 0.0, got {v}");
        }
    }

    // -----------------------------------------------------------------------
    // Test 14: evaluate with single Override layer at t=0 returns layer output
    // -----------------------------------------------------------------------

    #[test]
    fn test_evaluate_single_override_layer_at_t0() {
        let mut tl = AnimationTimeline::new(2);
        let clip = make_neutral_clip(2);
        tl.add_layer(TimelineLayer::new("l", clip)).unwrap();
        tl.set_time(0.0);
        let result = tl.evaluate().unwrap();
        assert_eq!(result.len(), 2);
        // neutral clip returns [0, 0] at t=0
        for &v in &result {
            assert!((v - 0.0).abs() < 1e-4);
        }
    }

    // -----------------------------------------------------------------------
    // Test 15: evaluate with muted layer returns zeros
    // -----------------------------------------------------------------------

    #[test]
    fn test_evaluate_muted_layer_returns_zeros() {
        let mut tl = AnimationTimeline::new(2);
        // At t=1 this clip would produce [0.5, 0.5] (between 0 and 1.0)
        let clip = make_once_clip(2, 1.0);
        tl.add_layer(
            TimelineLayer::new("l", clip)
                .with_blend_mode(BlendMode::Override)
                .with_weight(1.0)
                .muted(true),
        )
        .unwrap();
        tl.set_time(0.5);
        let result = tl.evaluate().unwrap();
        for &v in &result {
            assert!(
                (v - 0.0).abs() < 1e-4,
                "muted layer should yield 0, got {v}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Test 16: evaluate with two Override layers — top layer wins
    // -----------------------------------------------------------------------

    #[test]
    fn test_evaluate_two_override_layers_top_wins() {
        let mut tl = AnimationTimeline::new(2);

        // Base layer: all zeros (neutral)
        let base = make_neutral_clip(2);
        tl.add_layer(
            TimelineLayer::new("base", base)
                .with_blend_mode(BlendMode::Override)
                .with_weight(1.0),
        )
        .unwrap();

        // Top layer at t=1 produces [1.0, 1.0]
        let top = make_once_clip(2, 1.0);
        tl.add_layer(
            TimelineLayer::new("top", top)
                .with_blend_mode(BlendMode::Override)
                .with_weight(1.0),
        )
        .unwrap();

        // At t=1 the top layer produces [1.0, 1.0]
        tl.set_time(1.0);
        let result = tl.evaluate().unwrap();
        for &v in &result {
            assert!((v - 1.0).abs() < 1e-4, "top layer should dominate, got {v}");
        }
    }

    // -----------------------------------------------------------------------
    // Test 17: evaluate with Additive layer adds to base
    // -----------------------------------------------------------------------

    #[test]
    fn test_evaluate_additive_layer() {
        let mut tl = AnimationTimeline::new(2);

        // Base layer at t=1 produces [0.5, 0.5] (halfway between 0 and 1)
        // but we will use it as Override to seed the result at 0
        // Actually let's put base at t=1 producing value=0 (neutral)
        let base = make_neutral_clip(2);
        tl.add_layer(
            TimelineLayer::new("base", base)
                .with_blend_mode(BlendMode::Override)
                .with_weight(1.0),
        )
        .unwrap();

        // Additive layer: at t=1 produces [1.0, 1.0] * weight=0.5 = [0.5, 0.5]
        let additive = make_once_clip(2, 1.0);
        tl.add_layer(
            TimelineLayer::new("add", additive)
                .with_blend_mode(BlendMode::Additive)
                .with_weight(0.5),
        )
        .unwrap();

        tl.set_time(1.0);
        let result = tl.evaluate().unwrap();
        // base=0, additive adds 1.0 * 0.5 = 0.5
        for &v in &result {
            assert!((v - 0.5).abs() < 1e-4, "additive should add 0.5, got {v}");
        }
    }

    // -----------------------------------------------------------------------
    // Test 18: evaluate with Weighted blend mode
    // -----------------------------------------------------------------------

    #[test]
    fn test_evaluate_weighted_blend() {
        let mut tl = AnimationTimeline::new(2);

        let base = make_neutral_clip(2);
        tl.add_layer(
            TimelineLayer::new("base", base)
                .with_blend_mode(BlendMode::Override)
                .with_weight(1.0),
        )
        .unwrap();

        let weighted = make_once_clip(2, 1.0);
        tl.add_layer(
            TimelineLayer::new("w", weighted)
                .with_blend_mode(BlendMode::Weighted)
                .with_weight(0.3),
        )
        .unwrap();

        tl.set_time(1.0);
        let result = tl.evaluate().unwrap();
        // base=0, weighted adds 1.0 * 0.3 = 0.3
        for &v in &result {
            assert!((v - 0.3).abs() < 1e-4, "weighted should add 0.3, got {v}");
        }
    }

    // -----------------------------------------------------------------------
    // Test 19: total_duration with single layer with offset
    // -----------------------------------------------------------------------

    #[test]
    fn test_total_duration_with_offset() {
        let mut tl = AnimationTimeline::new(2);
        // Clip has 1.0s duration (playback_rate=1.0 so effective_duration = 1.0)
        let clip = make_once_clip(2, 1.0);
        tl.add_layer(TimelineLayer::new("l", clip).with_start_offset(2.0))
            .unwrap();
        // total = start_offset + effective_duration = 2.0 + 1.0 = 3.0
        assert!((tl.total_duration() - 3.0).abs() < 1e-4);
    }

    // -----------------------------------------------------------------------
    // Test 20: next_marker finds marker after current time
    // -----------------------------------------------------------------------

    #[test]
    fn test_next_marker() {
        let mut tl = AnimationTimeline::new(2);
        tl.add_marker(TimelineMarker {
            name: "m1".to_string(),
            time: 1.0,
            label: None,
        });
        tl.add_marker(TimelineMarker {
            name: "m2".to_string(),
            time: 3.0,
            label: None,
        });
        tl.add_marker(TimelineMarker {
            name: "m3".to_string(),
            time: 5.0,
            label: None,
        });

        tl.set_time(2.0);
        let next = tl.next_marker().unwrap();
        assert_eq!(next.name, "m2");
        assert!((next.time - 3.0).abs() < 1e-4);

        tl.set_time(5.0);
        assert!(tl.next_marker().is_none());
    }

    // -----------------------------------------------------------------------
    // Test 21: prev_marker finds marker at or before current time
    // -----------------------------------------------------------------------

    #[test]
    fn test_prev_marker() {
        let mut tl = AnimationTimeline::new(2);
        tl.add_marker(TimelineMarker {
            name: "m1".to_string(),
            time: 1.0,
            label: None,
        });
        tl.add_marker(TimelineMarker {
            name: "m2".to_string(),
            time: 3.0,
            label: None,
        });

        tl.set_time(3.0);
        let prev = tl.prev_marker().unwrap();
        assert_eq!(prev.name, "m2");

        tl.set_time(0.5);
        assert!(tl.prev_marker().is_none());
    }

    // -----------------------------------------------------------------------
    // Test 22: timeline_from_clip convenience constructor
    // -----------------------------------------------------------------------

    #[test]
    fn test_timeline_from_clip() {
        let clip = make_neutral_clip(3);
        let tl = timeline_from_clip(clip, 3);
        assert_eq!(tl.num_layers(), 1);
        assert_eq!(tl.layers[0].name, "default");
        let result = tl.evaluate().unwrap();
        assert_eq!(result.len(), 3);
    }

    // -----------------------------------------------------------------------
    // Test 23: evaluate after seek_to_marker uses correct time
    // -----------------------------------------------------------------------

    #[test]
    fn test_evaluate_after_seek_to_marker() {
        let mut tl = AnimationTimeline::new(2);
        // At t=1 the once-clip produces [1.0, 1.0]
        let clip = make_once_clip(2, 1.0);
        tl.add_layer(
            TimelineLayer::new("l", clip)
                .with_blend_mode(BlendMode::Override)
                .with_weight(1.0),
        )
        .unwrap();

        tl.add_marker(TimelineMarker {
            name: "peak".to_string(),
            time: 1.0,
            label: None,
        });
        tl.seek_to_marker("peak").unwrap();

        assert!((tl.global_time() - 1.0).abs() < 1e-4);

        let result = tl.evaluate().unwrap();
        for &v in &result {
            assert!(
                (v - 1.0).abs() < 1e-4,
                "at marker=peak t=1: expected 1.0, got {v}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Additional edge-case tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_layer_builder_chain() {
        let clip = make_neutral_clip(2);
        let layer = TimelineLayer::new("chain", clip)
            .with_start_offset(2.5)
            .with_blend_mode(BlendMode::Additive)
            .with_weight(0.7)
            .muted(false);
        assert_eq!(layer.name, "chain");
        assert!((layer.start_offset - 2.5).abs() < 1e-4);
        assert_eq!(layer.blend_mode, BlendMode::Additive);
        assert!((layer.weight - 0.7).abs() < 1e-4);
        assert!(!layer.muted);
    }

    #[test]
    fn test_set_layer_weight_not_found() {
        let mut tl = AnimationTimeline::new(2);
        let result = tl.set_layer_weight("ghost", 0.5);
        assert!(matches!(result, Err(TimelineError::LayerNotFound(_))));
    }

    #[test]
    fn test_set_layer_muted_not_found() {
        let mut tl = AnimationTimeline::new(2);
        let result = tl.set_layer_muted("ghost", true);
        assert!(matches!(result, Err(TimelineError::LayerNotFound(_))));
    }

    #[test]
    fn test_layer_mut_returns_none_for_missing() {
        let mut tl = AnimationTimeline::new(2);
        assert!(tl.layer_mut("missing").is_none());
    }

    #[test]
    fn test_evaluate_once_clip_before_start_offset_skipped() {
        let mut tl = AnimationTimeline::new(2);
        // Clip starts at t=5.0, current time is 1.0 → local = -4.0 → Once → skip
        let clip = make_once_clip(2, 1.0);
        tl.add_layer(
            TimelineLayer::new("l", clip)
                .with_start_offset(5.0)
                .with_blend_mode(BlendMode::Override)
                .with_weight(1.0),
        )
        .unwrap();
        tl.set_time(1.0);
        let result = tl.evaluate().unwrap();
        // Layer skipped → result stays zeros
        for &v in &result {
            assert!(
                (v - 0.0).abs() < 1e-4,
                "clip before start should be skipped, got {v}"
            );
        }
    }

    #[test]
    fn test_evaluate_loop_clip_before_start_offset_not_skipped() {
        let mut tl = AnimationTimeline::new(2);
        // Loop clip with start_offset=5, current t=1 → local=-4 → clamped to 0.0 → evaluate at 0
        let clip = make_clip(2, 1.0, LoopMode::Loop);
        tl.add_layer(
            TimelineLayer::new("l", clip)
                .with_start_offset(5.0)
                .with_blend_mode(BlendMode::Override)
                .with_weight(1.0),
        )
        .unwrap();
        tl.set_time(1.0);
        let result = tl.evaluate().unwrap();
        // evaluate at local=0 → timeline at t=0 → weights = [0, 0]
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_override_partial_weight() {
        let mut tl = AnimationTimeline::new(2);

        // Base seeded at 0 (neutral)
        let base = make_neutral_clip(2);
        tl.add_layer(
            TimelineLayer::new("base", base)
                .with_blend_mode(BlendMode::Override)
                .with_weight(1.0),
        )
        .unwrap();

        // Override top layer with weight=0.5 at t=1 (produces [1.0,1.0])
        // lerp(0, 1, 0.5) = 0.5
        let top = make_once_clip(2, 1.0);
        tl.add_layer(
            TimelineLayer::new("top", top)
                .with_blend_mode(BlendMode::Override)
                .with_weight(0.5),
        )
        .unwrap();

        tl.set_time(1.0);
        let result = tl.evaluate().unwrap();
        for &v in &result {
            assert!(
                (v - 0.5).abs() < 1e-4,
                "override weight=0.5 should lerp to 0.5, got {v}"
            );
        }
    }

    #[test]
    fn test_result_clamped_to_flame_range() {
        let mut tl = AnimationTimeline::new(2);
        // Additive layer at t=1 with value=1.0, weight=1.0, applied many times
        // We add 4 additive layers each contributing 1.0 → sum = 4.0 > 3.0 → clamped to 3.0
        for i in 0..4 {
            let clip = make_once_clip(2, 1.0);
            tl.add_layer(
                TimelineLayer::new(format!("add_{i}"), clip)
                    .with_blend_mode(BlendMode::Additive)
                    .with_weight(1.0),
            )
            .unwrap();
        }
        tl.set_time(1.0);
        let result = tl.evaluate().unwrap();
        for &v in &result {
            assert!(
                (v - 3.0).abs() < 1e-4,
                "result should be clamped to 3.0, got {v}"
            );
        }
    }

    #[test]
    fn test_total_duration_multiple_layers() {
        let mut tl = AnimationTimeline::new(2);
        // Layer 1: offset=0, duration=1 → end=1
        let c1 = make_once_clip(2, 1.0);
        tl.add_layer(TimelineLayer::new("a", c1).with_start_offset(0.0))
            .unwrap();
        // Layer 2: offset=3, duration=1 → end=4
        let c2 = make_once_clip(2, 1.0);
        tl.add_layer(TimelineLayer::new("b", c2).with_start_offset(3.0))
            .unwrap();
        assert!((tl.total_duration() - 4.0).abs() < 1e-4);
    }

    #[test]
    fn test_next_prev_marker_at_boundary() {
        let mut tl = AnimationTimeline::new(2);
        tl.add_marker(TimelineMarker {
            name: "start".to_string(),
            time: 0.0,
            label: None,
        });
        tl.add_marker(TimelineMarker {
            name: "end".to_string(),
            time: 10.0,
            label: None,
        });

        // At t=0, prev_marker should be "start" (time <= 0)
        tl.set_time(0.0);
        let prev = tl.prev_marker().unwrap();
        assert_eq!(prev.name, "start");

        // next should be "end"
        let next = tl.next_marker().unwrap();
        assert_eq!(next.name, "end");
    }
}
