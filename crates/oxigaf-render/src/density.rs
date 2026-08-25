//! Adaptive Density Control (ADC) for 3D Gaussian Splatting training.
//!
//! This module provides [`DensityController`], which wraps a [`GaussianModel`] and
//! implements the clone/split/prune/reset operations used during 3DGS training
//! as described in "3D Gaussian Splatting for Real-Time Radiance Field Rendering"
//! (Kerbl et al., 2023).
//!
//! ## Overview
//!
//! During training the gradient of the rendering loss with respect to the projected
//! Gaussian positions is accumulated over several iterations. Gaussians with a high
//! accumulated gradient are *under-reconstructed* and need to be densified:
//!
//! - **Small** Gaussians (max scale < threshold) are **cloned**: a copy is placed
//!   nearby with a small positional offset in the gradient direction.
//! - **Large** Gaussians (max scale ≥ threshold) are **split** into two smaller
//!   children whose scales are reduced by 1/φ ≈ 0.618 (reciprocal of the golden
//!   ratio), placed symmetrically along the dominant scale axis.
//!
//! Periodically, opacities are **reset** to a small value so that unused Gaussians
//! can be **pruned** on the next densification round.

use crate::gaussian::{GaussianAttributes, GaussianModel};

// ─────────────────────────────────────────────────────────────────────────────
// Math helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Numerically stable sigmoid: 1 / (1 + exp(-x)).
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0_f32 / (1.0_f32 + (-x).exp())
}

/// Logit (inverse sigmoid): ln(p / (1 - p)).
///
/// Clamped so that p is never exactly 0 or 1 to avoid ±inf.
#[inline]
fn logit(p: f32) -> f32 {
    let p = p.clamp(1e-7_f32, 1.0_f32 - 1e-7_f32);
    (p / (1.0_f32 - p)).ln()
}

/// Return the index of the maximum element in a three-element array.
#[inline]
fn argmax3(v: [f32; 3]) -> usize {
    if v[0] >= v[1] && v[0] >= v[2] {
        0
    } else if v[1] >= v[2] {
        1
    } else {
        2
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DensityConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for adaptive density control.
#[derive(Debug, Clone)]
pub struct DensityConfig {
    /// Gradient magnitude threshold for densification (clone or split).
    ///
    /// Gaussians whose *average* accumulated gradient norm exceeds this
    /// threshold are candidates for densification.
    pub grad_threshold: f32,

    /// Scale threshold used to choose between cloning and splitting.
    ///
    /// A Gaussian whose maximum *exponentiated* scale (i.e. `scale[i].exp()`)
    /// is **below** this value is **cloned**; otherwise it is **split**.
    pub scale_split_threshold: f32,

    /// Opacity threshold for pruning.
    ///
    /// Gaussians whose `sigmoid(opacity)` is **below** this value are pruned.
    pub opacity_prune_threshold: f32,

    /// Screen-space size threshold for pruning.
    ///
    /// Gaussians whose projected screen-space size (in normalised screen
    /// coordinates) **exceeds** this value are pruned.
    pub size_prune_threshold: f32,

    /// Opacity value (in [0, 1]) to which all opacities are reset during
    /// periodic opacity sparsification.  The raw stored value is
    /// `logit(opacity_reset_value)`.
    pub opacity_reset_value: f32,
}

impl Default for DensityConfig {
    fn default() -> Self {
        Self {
            grad_threshold: 2e-4_f32,
            scale_split_threshold: 0.01_f32,
            opacity_prune_threshold: 0.005_f32,
            size_prune_threshold: 0.1_f32,
            opacity_reset_value: 0.01_f32,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GradientAccumulator
// ─────────────────────────────────────────────────────────────────────────────

/// Tracks per-Gaussian gradient accumulation for density-control decisions.
///
/// Each training step the magnitude of the 2-D screen-space position gradient
/// is accumulated for every Gaussian that was visible in that frame.  After
/// `densify_and_prune` the accumulator is resized to match the new Gaussian
/// count and all accumulators are reset to zero.
#[derive(Debug, Clone)]
pub struct GradientAccumulator {
    /// Accumulated L2 norms of the screen-space position gradients,
    /// one element per Gaussian.
    pub position_grad_norm: Vec<f32>,

    /// Number of times each Gaussian contributed a gradient observation.
    /// Used to compute the running average.
    pub observation_count: Vec<u32>,
}

impl GradientAccumulator {
    /// Create a new zeroed accumulator for `n` Gaussians.
    pub fn new(n: usize) -> Self {
        Self {
            position_grad_norm: vec![0.0_f32; n],
            observation_count: vec![0_u32; n],
        }
    }

    /// Accumulate gradient norms for one training iteration.
    ///
    /// `grad_norms` must have the same length as the current Gaussian count.
    /// If the lengths differ the accumulation is skipped (safe no-op).
    ///
    /// A Gaussian that was not visible in this frame (had no rasterized
    /// footprint) contributes nothing to the loss and therefore has an
    /// exactly-zero screen-space position gradient *norm* (norms are
    /// non-negative by construction). `observation_count` is only
    /// incremented when `grad_norms[i] > 0.0`, so [`Self::average_grad_norm`]
    /// divides by the number of iterations the Gaussian was actually visible
    /// in rather than by the total number of `accumulate` calls - matching
    /// the struct-level documentation above and the reference 3DGS `denom`
    /// behaviour. This is a cheap proxy for an explicit visibility mask: it
    /// only under-counts a genuinely-visible Gaussian in the (exceedingly
    /// rare) case its gradient happens to land on exactly `0.0`.
    pub fn accumulate(&mut self, grad_norms: &[f32]) {
        if grad_norms.len() != self.position_grad_norm.len() {
            return;
        }
        for ((acc, cnt), &g) in self
            .position_grad_norm
            .iter_mut()
            .zip(self.observation_count.iter_mut())
            .zip(grad_norms.iter())
        {
            *acc += g;
            if g > 0.0_f32 {
                *cnt += 1;
            }
        }
    }

    /// Return the per-Gaussian *average* gradient norm.
    ///
    /// Gaussians that have never been observed return `0.0`.
    pub fn average_grad_norm(&self) -> Vec<f32> {
        self.position_grad_norm
            .iter()
            .zip(self.observation_count.iter())
            .map(
                |(&sum, &cnt)| {
                    if cnt == 0 {
                        0.0_f32
                    } else {
                        sum / cnt as f32
                    }
                },
            )
            .collect()
    }

    /// Reset all accumulators to zero without changing the size.
    pub fn reset(&mut self) {
        for v in self.position_grad_norm.iter_mut() {
            *v = 0.0_f32;
        }
        for c in self.observation_count.iter_mut() {
            *c = 0_u32;
        }
    }

    /// Resize the accumulator to exactly `new_n` Gaussians.
    ///
    /// Extra slots are initialised to zero; existing values within bounds are
    /// preserved.
    pub fn resize(&mut self, new_n: usize) {
        self.position_grad_norm.resize(new_n, 0.0_f32);
        self.observation_count.resize(new_n, 0_u32);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GaussianModel builder helpers (internal)
// ─────────────────────────────────────────────────────────────────────────────

/// Number of SH coefficients for a given degree (total float count).
fn sh_total_for_degree(sh_degree: u32) -> usize {
    ((sh_degree + 1) * (sh_degree + 1) * 3) as usize
}

/// Build an empty [`GaussianModel`] with the same metadata (sh_degree) as a
/// reference model, pre-sized to hold `cap` Gaussians.
///
/// Pre-sizing avoids the repeated reallocate-and-copy cycles that a bare
/// `Vec::new()` followed by a long burst of `push`/`extend_from_slice` calls
/// would otherwise incur - significant for large models, since `sh_coeffs`
/// alone is `cap * sh_total_for_degree(sh_degree)` floats (e.g. ~200 MB for
/// 1M degree-3 Gaussians).
fn model_with_capacity(reference: &GaussianModel, cap: usize) -> GaussianModel {
    let sh_c = sh_total_for_degree(reference.sh_degree);
    GaussianModel {
        gaussians: Vec::with_capacity(cap),
        sh_coeffs: Vec::with_capacity(cap.saturating_mul(sh_c)),
        sh_degree: reference.sh_degree,
        face_indices: Vec::with_capacity(cap),
        barycentric: Vec::with_capacity(cap),
        local_offsets: Vec::with_capacity(cap),
        is_rigid: Vec::with_capacity(cap),
    }
}

/// Append Gaussian `i` from `src` (including all FLAME fields and SH
/// coefficients) to `dst`.
fn append_gaussian(
    dst: &mut GaussianModel,
    src: &GaussianModel,
    i: usize,
    attrs: GaussianAttributes,
) {
    let sh_c = sh_total_for_degree(src.sh_degree);
    let sh_start = i * sh_c;
    let sh_end = sh_start + sh_c;

    dst.gaussians.push(attrs);
    if src.sh_coeffs.len() >= sh_end {
        dst.sh_coeffs
            .extend_from_slice(&src.sh_coeffs[sh_start..sh_end]);
    } else {
        // Pad with zeros if SH is missing (defensive).
        dst.sh_coeffs.extend(std::iter::repeat_n(0.0_f32, sh_c));
    }

    // FLAME binding fields.
    let fi = src.face_indices.get(i).copied().unwrap_or(0);
    let bary =
        src.barycentric
            .get(i)
            .copied()
            .unwrap_or([1.0_f32 / 3.0, 1.0_f32 / 3.0, 1.0_f32 / 3.0]);
    let off = src.local_offsets.get(i).copied().unwrap_or([0.0_f32; 3]);
    let rig = src.is_rigid.get(i).copied().unwrap_or(false);

    dst.face_indices.push(fi);
    dst.barycentric.push(bary);
    dst.local_offsets.push(off);
    dst.is_rigid.push(rig);
}

// ─────────────────────────────────────────────────────────────────────────────
// Gaussian transform helpers (clone / split)
// ─────────────────────────────────────────────────────────────────────────────
//
// Factored out so that `clone_gaussians_with_sources`, `split_gaussians_with_sources`
// and the fused single-pass path in `DensityController::densify_and_prune` all
// compute clone offsets / split children the exact same way - a single
// source of truth instead of three copies of the same arithmetic that could
// silently drift apart.

/// Log-space scale multiplier applied to both children of a split:
/// `ln(1/φ)` where `φ ≈ 1.618` is the golden ratio (so each child's linear
/// scale is ≈0.618× the parent's).
#[inline]
fn split_scale_factor_log() -> f32 {
    (1.0_f32 / 1.618_034_f32).ln() // ≈ -0.4812
}

/// Build the offset clone of a small, high-gradient Gaussian: identical
/// attributes except the position is nudged by `scale[dominant] * 0.01`
/// along the dominant scale axis.
fn make_clone_attrs(g: GaussianAttributes) -> GaussianAttributes {
    let dominant = argmax3(g.scale);
    let offset_mag = g.scale[dominant].exp() * 0.01_f32;
    let mut clone_pos = g.position;
    clone_pos[dominant] += offset_mag;
    GaussianAttributes {
        position: clone_pos,
        ..g
    }
}

/// Build the two children of a large, high-gradient Gaussian being split:
/// scale reduced by [`split_scale_factor_log`] in log-space on both, and
/// positions offset by ±`exp(scale[dominant])` along the dominant axis.
fn make_split_children(g: GaussianAttributes) -> (GaussianAttributes, GaussianAttributes) {
    let dominant = argmax3(g.scale);
    let offset_mag = g.scale[dominant].exp(); // world-space magnitude
    let scale_factor_log = split_scale_factor_log();
    let child_scale = [
        g.scale[0] + scale_factor_log,
        g.scale[1] + scale_factor_log,
        g.scale[2] + scale_factor_log,
    ];

    // Child A: position + offset along dominant axis.
    let mut pos_a = g.position;
    pos_a[dominant] += offset_mag;
    let child_a = GaussianAttributes {
        position: pos_a,
        scale: child_scale,
        ..g
    };

    // Child B: position - offset along dominant axis.
    let mut pos_b = g.position;
    pos_b[dominant] -= offset_mag;
    let child_b = GaussianAttributes {
        position: pos_b,
        scale: child_scale,
        ..g
    };

    (child_a, child_b)
}

// ─────────────────────────────────────────────────────────────────────────────
// DensityController
// ─────────────────────────────────────────────────────────────────────────────

/// Performs adaptive density control (ADC) on a [`GaussianModel`].
///
/// The controller owns a [`GradientAccumulator`] that must be kept in sync
/// with the model's Gaussian count (via [`DensityController::sync_to_model`]
/// after densification).
pub struct DensityController {
    /// Configuration parameters for density control.
    pub config: DensityConfig,
    /// Gradient accumulator tracking per-Gaussian gradients.
    pub accumulator: GradientAccumulator,
}

impl DensityController {
    /// Create a new controller for a model with `n` Gaussians.
    pub fn new(n: usize, config: DensityConfig) -> Self {
        Self {
            accumulator: GradientAccumulator::new(n),
            config,
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // clone_gaussians
    // ─────────────────────────────────────────────────────────────────────────

    /// Clone Gaussians that have high gradient AND small exponentiated scale.
    ///
    /// For each Gaussian `i` where:
    /// - `avg_grad[i] >= config.grad_threshold`, AND
    /// - `max(scale[j].exp()) < config.scale_split_threshold`
    ///
    /// …a copy is added to the model with a small position offset of
    /// `scale[dominant] * 0.01` along the dominant scale axis.
    ///
    /// Returns the new (original + clones) model and an updated accumulator
    /// sized to the new model.
    pub fn clone_gaussians(&self, model: &GaussianModel) -> (GaussianModel, GradientAccumulator) {
        let (new_model, new_acc, _sources) = self.clone_gaussians_with_sources(model);
        (new_model, new_acc)
    }

    /// Same as [`Self::clone_gaussians`], but additionally returns, for every
    /// Gaussian in the returned model, the index into `model` it was derived
    /// from (identity for unmodified copies, the parent's index for clones).
    ///
    /// Used internally by [`Self::densify_and_prune`] to correctly remap a
    /// caller-supplied `screen_sizes` buffer (computed against `model`)
    /// across the clone/split transformations, and to pre-size the output
    /// model's buffers in one allocation instead of growing them via
    /// repeated reallocation.
    fn clone_gaussians_with_sources(
        &self,
        model: &GaussianModel,
    ) -> (GaussianModel, GradientAccumulator, Vec<usize>) {
        let avg_grad = self.accumulator.average_grad_norm();
        let n = model.gaussians.len();
        let mut clone_indices: Vec<usize> = Vec::new();

        for i in 0..n {
            let g = model.gaussians[i];

            // Determine if this Gaussian should be cloned.
            let grad = avg_grad.get(i).copied().unwrap_or(0.0_f32);
            if grad < self.config.grad_threshold {
                continue;
            }

            // Compare exponentiated scales.
            let max_exp_scale = g.scale[0].exp().max(g.scale[1].exp()).max(g.scale[2].exp());
            if max_exp_scale >= self.config.scale_split_threshold {
                continue; // Large → split, not clone.
            }

            clone_indices.push(i);
        }

        // Pre-size the output model/sources for exactly `n + clone_indices.len()`
        // Gaussians so the append loops below never trigger a reallocation.
        let out_len = n + clone_indices.len();
        let mut new_model = model_with_capacity(model, out_len);
        let mut sources: Vec<usize> = Vec::with_capacity(out_len);

        // Copy every original unconditionally, in order.
        for i in 0..n {
            append_gaussian(&mut new_model, model, i, model.gaussians[i]);
            sources.push(i);
        }

        // Append clones.
        for i in clone_indices {
            let clone_attrs = make_clone_attrs(model.gaussians[i]);
            append_gaussian(&mut new_model, model, i, clone_attrs);
            sources.push(i);
        }

        let new_n = new_model.gaussians.len();
        let new_acc = GradientAccumulator::new(new_n);
        (new_model, new_acc, sources)
    }

    // ─────────────────────────────────────────────────────────────────────────
    // split_gaussians
    // ─────────────────────────────────────────────────────────────────────────

    /// Split Gaussians that have high gradient AND large exponentiated scale.
    ///
    /// For each Gaussian `i` where:
    /// - `avg_grad[i] >= config.grad_threshold`, AND
    /// - `max(scale[j].exp()) >= config.scale_split_threshold`
    ///
    /// …the Gaussian is replaced by two children.  Each child has:
    /// - Scale multiplied by 0.618 (in *log-space*: `scale[j] + ln(0.618)`),
    /// - Position offset by ±`exp(scale[dominant])` along the dominant axis.
    ///
    /// Gaussians that do not meet the split criterion are copied unchanged.
    ///
    /// Returns the new model and a fresh accumulator sized to the new count.
    pub fn split_gaussians(&self, model: &GaussianModel) -> (GaussianModel, GradientAccumulator) {
        let (new_model, new_acc, _sources) = self.split_gaussians_with_sources(model);
        (new_model, new_acc)
    }

    /// Same as [`Self::split_gaussians`], but additionally returns, for every
    /// Gaussian in the returned model, the index into `model` it was derived
    /// from. See [`Self::clone_gaussians_with_sources`] for why this exists.
    fn split_gaussians_with_sources(
        &self,
        model: &GaussianModel,
    ) -> (GaussianModel, GradientAccumulator, Vec<usize>) {
        let avg_grad = self.accumulator.average_grad_norm();
        let n = model.gaussians.len();

        // First pass: decide clone/split without touching the output model,
        // so its buffers can be pre-sized in one allocation (see
        // `model_with_capacity`) instead of growing via repeated
        // reallocation across the loop.
        let mut should_split: Vec<bool> = Vec::with_capacity(n);
        let mut out_len = 0usize;
        for i in 0..n {
            let g = model.gaussians[i];
            let grad = avg_grad.get(i).copied().unwrap_or(0.0_f32);
            let max_exp_scale = g.scale[0].exp().max(g.scale[1].exp()).max(g.scale[2].exp());
            let split = grad >= self.config.grad_threshold
                && max_exp_scale >= self.config.scale_split_threshold;
            should_split.push(split);
            out_len += if split { 2 } else { 1 };
        }

        let mut new_model = model_with_capacity(model, out_len);
        let mut sources: Vec<usize> = Vec::with_capacity(out_len);

        for (i, &split) in should_split.iter().enumerate() {
            let g = model.gaussians[i];

            if !split {
                // Copy unchanged.
                append_gaussian(&mut new_model, model, i, g);
                sources.push(i);
                continue;
            }

            let (child_a, child_b) = make_split_children(g);
            append_gaussian(&mut new_model, model, i, child_a);
            sources.push(i);
            append_gaussian(&mut new_model, model, i, child_b);
            sources.push(i);
        }

        let new_n = new_model.gaussians.len();
        let new_acc = GradientAccumulator::new(new_n);
        (new_model, new_acc, sources)
    }

    // ─────────────────────────────────────────────────────────────────────────
    // prune_gaussians
    // ─────────────────────────────────────────────────────────────────────────

    /// Remove Gaussians with low opacity or large screen-space projection.
    ///
    /// A Gaussian at index `i` is **kept** iff:
    /// 1. `sigmoid(opacity) > config.opacity_prune_threshold`, AND
    /// 2. `screen_sizes` is `None` OR `screen_sizes[i] <= config.size_prune_threshold`.
    ///
    /// All FLAME binding fields and SH coefficients are filtered in lockstep.
    pub fn prune_gaussians(
        &self,
        model: &GaussianModel,
        screen_sizes: Option<&[f32]>,
    ) -> GaussianModel {
        // The result can never be larger than the input, so pre-sizing with
        // the input length is a safe upper bound that guarantees zero
        // reallocations regardless of how many Gaussians survive.
        let mut new_model = model_with_capacity(model, model.gaussians.len());

        for i in 0..model.gaussians.len() {
            let g = model.gaussians[i];

            // Opacity check.
            if sigmoid(g.opacity) <= self.config.opacity_prune_threshold {
                continue;
            }

            // Screen-size check (optional).
            if let Some(sizes) = screen_sizes {
                let sz = sizes.get(i).copied().unwrap_or(0.0_f32);
                if sz > self.config.size_prune_threshold {
                    continue;
                }
            }

            append_gaussian(&mut new_model, model, i, g);
        }

        new_model
    }

    // ─────────────────────────────────────────────────────────────────────────
    // densify_and_prune
    // ─────────────────────────────────────────────────────────────────────────

    /// Combined densification step: clone small + split large + prune.
    ///
    /// This is the standard 3DGS densification operation applied once every
    /// `densification_interval` training steps. Conceptually equivalent to
    /// applying [`Self::clone_gaussians`], then [`Self::split_gaussians`] on
    /// the result, then [`Self::prune_gaussians`] - those three remain
    /// available separately for callers that want the individual steps -
    /// but implemented as a single fused pass (see
    /// `densify_and_prune_fused` for the per-Gaussian case analysis
    /// proving the two are equivalent) so one densification interval
    /// performs a single pre-sized output allocation and a single write
    /// pass over the model, instead of chaining three full model copies
    /// (each with its own SH-coefficient `extend_from_slice` traffic)
    /// together.
    ///
    /// `screen_sizes`, if given, must be indexed against `model` (i.e. the
    /// *pre*-densification Gaussian count) - typically the per-Gaussian
    /// screen-space size computed by the caller's most recent render pass.
    /// Every Gaussian produced by cloning or splitting inherits its parent's
    /// opacity and, for the purposes of this size check, its parent's screen
    /// size unchanged - i.e. it is tested against `screen_sizes[parent]`,
    /// never against whatever `screen_sizes` entry happens to share its
    /// *post*-densification index.
    ///
    /// The controller's internal accumulator is updated automatically.
    pub fn densify_and_prune(
        &mut self,
        model: &GaussianModel,
        screen_sizes: Option<&[f32]>,
    ) -> GaussianModel {
        let pruned_model = self.densify_and_prune_fused(model, screen_sizes);
        self.sync_to_model(&pruned_model);
        pruned_model
    }

    // ─────────────────────────────────────────────────────────────────────────
    // densify_and_prune_fused (internal)
    // ─────────────────────────────────────────────────────────────────────────

    /// One-pass fused clone + split + prune, used internally by
    /// [`Self::densify_and_prune`].
    ///
    /// # Why this is semantically identical to the composed pipeline
    ///
    /// `split_gaussians_with_sources` evaluates its grad/scale test against
    /// `self.accumulator`, which - inside `densify_and_prune` - is never
    /// resized between the (conceptual) clone and split steps. So for any
    /// post-clone index at or beyond the pre-clone Gaussian count `n` (i.e.
    /// every clone), `average_grad_norm().get(index)` is `None` and the
    /// split test's gradient defaults to `0.0`, which is always below
    /// `grad_threshold`. Combined with a clone's scale being identical to
    /// its (necessarily *small*, by the clone criterion) parent's, a
    /// freshly-cloned entry can therefore never itself qualify for a split,
    /// and symmetrically a Gaussian large enough to split is never small
    /// enough to have been cloned in the first place. Every entry the
    /// composed pipeline ultimately emits is therefore fully determined by
    /// exactly one original Gaussian `i`, evaluated once against `model`
    /// and `average_grad_norm()` alone (never against a post-clone or
    /// post-split intermediate):
    ///
    /// - `grad < grad_threshold` → **Keep**: the single unchanged original.
    /// - `grad >= grad_threshold` and `max(scale.exp()) < scale_split_threshold`
    ///   → **Clone**: the unchanged original, plus one clone offset along
    ///   the dominant scale axis ([`make_clone_attrs`]; this Gaussian's
    ///   small scale means `split_gaussians` always leaves both the
    ///   original-row copy and the later-appended clone unmodified).
    /// - `grad >= grad_threshold` and `max(scale.exp()) >= scale_split_threshold`
    ///   → **Split**: two children replacing the original
    ///   ([`make_split_children`]; this Gaussian's large scale means
    ///   `clone_gaussians` never clones it, so `split_gaussians` sees it
    ///   unmodified at its original row and splits it there).
    ///
    /// Every entry derived from Gaussian `i` inherits `i`'s opacity
    /// (`GaussianAttributes { ..g }` never overwrites `opacity`) and is
    /// tested against `screen_sizes[i]` per `densify_and_prune`'s
    /// screen-size contract - so the prune decision is identical for every
    /// entry derived from a given `i` and can be made once per `i` rather
    /// than once per output entry.
    ///
    /// Output order matches the composed pipeline exactly: for each
    /// surviving original index in ascending order, its Keep/Clone-original/
    /// Split-children entry (or entries) first, matching
    /// `split_gaussians_with_sources` walking the post-clone model's first
    /// `n` (original-row) positions before its appended-clone tail; then,
    /// in a second pass over ascending original index, each surviving
    /// Clone's tail entry - matching `clone_gaussians_with_sources`
    /// appending all clones after all originals.
    fn densify_and_prune_fused(
        &self,
        model: &GaussianModel,
        screen_sizes: Option<&[f32]>,
    ) -> GaussianModel {
        #[derive(Clone, Copy, PartialEq, Eq)]
        enum Action {
            /// Below the gradient threshold: emit the original unchanged.
            Keep,
            /// High gradient, small scale: emit the original plus one
            /// offset clone.
            Clone,
            /// High gradient, large scale: emit two children in place of
            /// the original.
            Split,
        }

        let avg_grad = self.accumulator.average_grad_norm();
        let n = model.gaussians.len();

        // Decide the action and prune-survival for Gaussian `i` once; every
        // entry `i` contributes shares this same verdict (see this
        // function's doc).
        let decide = |i: usize| -> (Action, bool) {
            let g = model.gaussians[i];
            let grad = avg_grad.get(i).copied().unwrap_or(0.0_f32);
            let max_exp_scale = g.scale[0].exp().max(g.scale[1].exp()).max(g.scale[2].exp());
            let action = if grad < self.config.grad_threshold {
                Action::Keep
            } else if max_exp_scale < self.config.scale_split_threshold {
                Action::Clone
            } else {
                Action::Split
            };

            let mut survives = sigmoid(g.opacity) > self.config.opacity_prune_threshold;
            if survives {
                if let Some(sizes) = screen_sizes {
                    let sz = sizes.get(i).copied().unwrap_or(0.0_f32);
                    if sz > self.config.size_prune_threshold {
                        survives = false;
                    }
                }
            }
            (action, survives)
        };

        // Pass 1: decide once per original Gaussian and count the exact
        // number of surviving output entries, so the destination model's
        // buffers can be pre-sized in a single allocation (`model_with_capacity`)
        // instead of growing via repeated reallocation.
        let mut decisions: Vec<(Action, bool)> = Vec::with_capacity(n);
        let mut out_len = 0usize;
        for i in 0..n {
            let d = decide(i);
            if d.1 {
                out_len += match d.0 {
                    Action::Keep => 1,
                    Action::Clone | Action::Split => 2,
                };
            }
            decisions.push(d);
        }

        let mut new_model = model_with_capacity(model, out_len);

        // Pass 2a: main block, one entry (Keep/Clone-original) or split
        // pair per surviving original index, in ascending index order.
        for (i, &(action, survives)) in decisions.iter().enumerate() {
            if !survives {
                continue;
            }
            let g = model.gaussians[i];
            match action {
                Action::Keep | Action::Clone => {
                    append_gaussian(&mut new_model, model, i, g);
                }
                Action::Split => {
                    let (child_a, child_b) = make_split_children(g);
                    append_gaussian(&mut new_model, model, i, child_a);
                    append_gaussian(&mut new_model, model, i, child_b);
                }
            }
        }

        // Pass 2b: clone tail, in ascending index order - matching
        // `clone_gaussians_with_sources`'s clones-appended-after-originals
        // ordering.
        for (i, &(action, survives)) in decisions.iter().enumerate() {
            if action != Action::Clone || !survives {
                continue;
            }
            let clone_attrs = make_clone_attrs(model.gaussians[i]);
            append_gaussian(&mut new_model, model, i, clone_attrs);
        }

        new_model
    }

    // ─────────────────────────────────────────────────────────────────────────
    // reset_opacity
    // ─────────────────────────────────────────────────────────────────────────

    /// Reset all Gaussian opacities to `config.opacity_reset_value`.
    ///
    /// The stored raw value becomes `logit(opacity_reset_value)`.
    /// This is used for periodic opacity sparsification so that unused
    /// Gaussians can be pruned in the next densification round.
    pub fn reset_opacity(&self, model: &mut GaussianModel) {
        let raw = logit(self.config.opacity_reset_value);
        for g in model.gaussians.iter_mut() {
            g.opacity = raw;
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // sync_to_model
    // ─────────────────────────────────────────────────────────────────────────

    /// Resize and reset the internal accumulator to match `model.len()`.
    ///
    /// Call this after any operation that changes the Gaussian count so that
    /// subsequent `accumulate` calls have the correct length.
    pub fn sync_to_model(&mut self, model: &GaussianModel) {
        let n = model.len();
        self.accumulator.resize(n);
        self.accumulator.reset();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gaussian::{GaussianAttributes, GaussianModel};

    // ── Test helpers ──────────────────────────────────────────────────────────

    /// Create a simple `GaussianModel` with `n` Gaussians.
    ///
    /// - `opacity_logit`: raw stored opacity (logit-space)
    /// - `scale_log`: log-scale applied to all three axes
    fn make_model(n: usize, opacity_logit: f32, scale_log: f32) -> GaussianModel {
        let sh_degree = 0_u32;
        let sh_c = sh_total_for_degree(sh_degree);

        let mut gaussians = Vec::with_capacity(n);
        let mut sh_coeffs = Vec::with_capacity(n * sh_c);

        for idx in 0..n {
            gaussians.push(GaussianAttributes {
                position: [idx as f32 * 0.1_f32, 0.0, 0.0],
                _pad0: 0.0,
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: [scale_log; 3],
                opacity: opacity_logit,
            });
            // SH coefficients — all zeros is fine for tests.
            sh_coeffs.extend(std::iter::repeat_n(0.0_f32, sh_c));
        }

        let third = 1.0_f32 / 3.0_f32;
        GaussianModel {
            gaussians,
            sh_coeffs,
            sh_degree,
            face_indices: vec![0_u32; n],
            barycentric: vec![[third, third, third]; n],
            local_offsets: vec![[0.0_f32; 3]; n],
            is_rigid: vec![false; n],
        }
    }

    /// Create a controller with a given config and set the accumulator so that
    /// all `n` Gaussians have `avg_grad` equal to `grad_value`.
    fn make_controller_with_grad(
        n: usize,
        config: DensityConfig,
        grad_value: f32,
    ) -> DensityController {
        let mut ctrl = DensityController::new(n, config);
        // Set raw sums directly so that avg = grad_value (count = 1).
        for i in 0..n {
            ctrl.accumulator.position_grad_norm[i] = grad_value;
            ctrl.accumulator.observation_count[i] = 1;
        }
        ctrl
    }

    // ── GradientAccumulator tests ─────────────────────────────────────────────

    #[test]
    fn test_accumulator_new_initialises_to_zero() {
        let acc = GradientAccumulator::new(5);
        assert_eq!(acc.position_grad_norm, vec![0.0_f32; 5]);
        assert_eq!(acc.observation_count, vec![0_u32; 5]);
    }

    #[test]
    fn test_accumulate_adds_norms_and_increments_count() {
        let mut acc = GradientAccumulator::new(3);
        acc.accumulate(&[0.1, 0.2, 0.3]);
        assert!((acc.position_grad_norm[0] - 0.1).abs() < 1e-6);
        assert!((acc.position_grad_norm[1] - 0.2).abs() < 1e-6);
        assert!((acc.position_grad_norm[2] - 0.3).abs() < 1e-6);
        assert_eq!(acc.observation_count, vec![1, 1, 1]);
    }

    #[test]
    fn test_accumulate_is_additive() {
        let mut acc = GradientAccumulator::new(2);
        acc.accumulate(&[0.5, 0.5]);
        acc.accumulate(&[0.5, 0.5]);
        assert!((acc.position_grad_norm[0] - 1.0).abs() < 1e-6);
        assert_eq!(acc.observation_count[0], 2);
    }

    #[test]
    fn test_accumulate_length_mismatch_is_noop() {
        let mut acc = GradientAccumulator::new(3);
        acc.accumulate(&[1.0, 2.0]); // length mismatch → no-op
        assert_eq!(acc.position_grad_norm, vec![0.0_f32; 3]);
        assert_eq!(acc.observation_count, vec![0_u32; 3]);
    }

    #[test]
    fn test_accumulate_does_not_count_invisible_gaussians() {
        // A Gaussian with an exactly-zero position-gradient norm did not
        // contribute to the rendered image this iteration (was not
        // visible), so it must not count toward the observation
        // denominator - otherwise `average_grad_norm` dilutes rarely-visible
        // Gaussians by the *total* iteration count instead of the number of
        // iterations they were actually visible in (regression test for the
        // bug where `observation_count` was incremented unconditionally).
        let mut acc = GradientAccumulator::new(3);
        // Gaussian 0 visible every iteration, gaussian 1 visible only on the
        // second iteration, gaussian 2 never visible.
        acc.accumulate(&[1.0, 0.0, 0.0]);
        acc.accumulate(&[1.0, 1.0, 0.0]);
        acc.accumulate(&[1.0, 0.0, 0.0]);

        assert_eq!(
            acc.observation_count[0], 3,
            "gaussian 0 was visible all 3 iterations"
        );
        assert_eq!(
            acc.observation_count[1], 1,
            "gaussian 1 was visible only once"
        );
        assert_eq!(acc.observation_count[2], 0, "gaussian 2 was never visible");

        let avg = acc.average_grad_norm();
        assert!((avg[0] - 1.0).abs() < 1e-6);
        assert!(
            (avg[1] - 1.0).abs() < 1e-6,
            "average should be 1.0/1 = 1.0 (diluted-by-total-count would give 1.0/3), got {}",
            avg[1]
        );
        assert_eq!(avg[2], 0.0);
    }

    #[test]
    fn test_average_grad_norm_divides_by_count() {
        let mut acc = GradientAccumulator::new(2);
        acc.accumulate(&[0.6, 0.4]);
        acc.accumulate(&[0.6, 0.4]);
        let avg = acc.average_grad_norm();
        assert!((avg[0] - 0.6).abs() < 1e-5);
        assert!((avg[1] - 0.4).abs() < 1e-5);
    }

    #[test]
    fn test_average_grad_norm_zero_for_unobserved() {
        let acc = GradientAccumulator::new(3);
        let avg = acc.average_grad_norm();
        for v in avg {
            assert_eq!(v, 0.0_f32);
        }
    }

    #[test]
    fn test_reset_zeroes_everything() {
        let mut acc = GradientAccumulator::new(4);
        acc.accumulate(&[1.0, 2.0, 3.0, 4.0]);
        acc.reset();
        assert_eq!(acc.position_grad_norm, vec![0.0_f32; 4]);
        assert_eq!(acc.observation_count, vec![0_u32; 4]);
    }

    #[test]
    fn test_resize_extends_with_zeros() {
        let mut acc = GradientAccumulator::new(2);
        acc.accumulate(&[0.5, 0.5]);
        acc.resize(5);
        assert_eq!(acc.position_grad_norm.len(), 5);
        assert_eq!(acc.observation_count.len(), 5);
        // New slots are zero.
        assert_eq!(acc.position_grad_norm[4], 0.0_f32);
        assert_eq!(acc.observation_count[4], 0_u32);
    }

    #[test]
    fn test_resize_shrinks_correctly() {
        let mut acc = GradientAccumulator::new(5);
        acc.resize(2);
        assert_eq!(acc.position_grad_norm.len(), 2);
        assert_eq!(acc.observation_count.len(), 2);
    }

    // ── clone_gaussians tests ─────────────────────────────────────────────────

    #[test]
    fn test_clone_gaussians_increases_count() {
        let n = 4;
        // Small scale: exp(log(0.001)) = 0.001 < 0.01 (default threshold).
        let scale_log = 0.001_f32.ln();
        let model = make_model(n, logit(0.5), scale_log);
        // High gradient for all Gaussians.
        let config = DensityConfig::default();
        let ctrl = make_controller_with_grad(n, config, 5e-4_f32);
        let (new_model, _acc) = ctrl.clone_gaussians(&model);
        assert!(
            new_model.gaussians.len() > model.gaussians.len(),
            "Expected more Gaussians after cloning, got {} (was {})",
            new_model.gaussians.len(),
            model.gaussians.len()
        );
    }

    #[test]
    fn test_clone_gaussians_preserves_originals() {
        let n = 3;
        let scale_log = 0.001_f32.ln();
        let model = make_model(n, logit(0.5), scale_log);
        let config = DensityConfig::default();
        let ctrl = make_controller_with_grad(n, config, 5e-4_f32);
        let (new_model, _) = ctrl.clone_gaussians(&model);

        // The first `n` entries must match the originals exactly.
        for i in 0..n {
            let orig = model.gaussians[i];
            let copy = new_model.gaussians[i];
            assert_eq!(orig.position, copy.position);
            assert_eq!(orig.opacity, copy.opacity);
            assert_eq!(orig.scale, copy.scale);
        }
    }

    #[test]
    fn test_clone_gaussians_no_clone_below_threshold() {
        let n = 4;
        let scale_log = 0.001_f32.ln();
        let model = make_model(n, logit(0.5), scale_log);
        let config = DensityConfig::default();
        // Gradient below threshold → no clones.
        let ctrl = make_controller_with_grad(n, config, 1e-5_f32);
        let (new_model, _) = ctrl.clone_gaussians(&model);
        assert_eq!(
            new_model.gaussians.len(),
            n,
            "Expected no clones when gradient is below threshold"
        );
    }

    // ── split_gaussians tests ─────────────────────────────────────────────────

    #[test]
    fn test_split_gaussians_increases_or_maintains_count() {
        let n = 4;
        // Large scale: exp(log(0.05)) = 0.05 > 0.01.
        let scale_log = 0.05_f32.ln();
        let model = make_model(n, logit(0.5), scale_log);
        let config = DensityConfig::default();
        let ctrl = make_controller_with_grad(n, config, 5e-4_f32);
        let (new_model, _) = ctrl.split_gaussians(&model);
        assert!(
            new_model.gaussians.len() >= model.gaussians.len(),
            "Expected split to produce at least as many Gaussians"
        );
    }

    #[test]
    fn test_split_gaussians_reduces_scale() {
        let n = 2;
        let scale_log = 0.05_f32.ln();
        let model = make_model(n, logit(0.5), scale_log);
        let config = DensityConfig::default();
        let ctrl = make_controller_with_grad(n, config, 5e-4_f32);
        let (new_model, _) = ctrl.split_gaussians(&model);

        // All children should have smaller scale than the original.
        let original_max_exp = scale_log.exp();
        for g in &new_model.gaussians {
            let child_max_exp = g.scale[0].exp().max(g.scale[1].exp()).max(g.scale[2].exp());
            assert!(
                child_max_exp < original_max_exp + 1e-6,
                "Child scale {} should be less than original {}",
                child_max_exp,
                original_max_exp
            );
        }
    }

    #[test]
    fn test_split_gaussians_no_split_small_scale() {
        let n = 3;
        // Small scale (below threshold) → no split.
        let scale_log = 0.001_f32.ln();
        let model = make_model(n, logit(0.5), scale_log);
        let config = DensityConfig::default();
        let ctrl = make_controller_with_grad(n, config, 5e-4_f32);
        let (new_model, _) = ctrl.split_gaussians(&model);
        // Small scale → clone (not split), so split_gaussians should just copy them.
        assert_eq!(
            new_model.gaussians.len(),
            n,
            "Expected no splits when scale is below split threshold"
        );
    }

    // ── prune_gaussians tests ─────────────────────────────────────────────────

    #[test]
    fn test_prune_high_opacity_keeps_all() {
        let n = 5;
        // sigmoid(logit(0.9)) ≈ 0.9 >> 0.005
        let model = make_model(n, logit(0.9), 0.0_f32);
        let config = DensityConfig::default();
        let ctrl = DensityController::new(n, config);
        let pruned = ctrl.prune_gaussians(&model, None);
        assert_eq!(
            pruned.gaussians.len(),
            n,
            "High-opacity Gaussians should all be kept"
        );
    }

    #[test]
    fn test_prune_low_opacity_removes_all() {
        let n = 5;
        // sigmoid(logit(0.001)) ≈ 0.001 < 0.005
        let model = make_model(n, logit(0.001), 0.0_f32);
        let config = DensityConfig::default();
        let ctrl = DensityController::new(n, config);
        let pruned = ctrl.prune_gaussians(&model, None);
        assert_eq!(
            pruned.gaussians.len(),
            0,
            "Low-opacity Gaussians should all be pruned"
        );
    }

    #[test]
    fn test_prune_by_screen_size_removes_large() {
        let n = 4;
        let model = make_model(n, logit(0.9), 0.0_f32);
        let config = DensityConfig {
            size_prune_threshold: 0.05_f32,
            ..DensityConfig::default()
        };
        let ctrl = DensityController::new(n, config);
        // First two are small, last two are large.
        let sizes = vec![0.01_f32, 0.03, 0.2, 0.5];
        let pruned = ctrl.prune_gaussians(&model, Some(&sizes));
        assert_eq!(
            pruned.gaussians.len(),
            2,
            "Only Gaussians with screen size <= threshold should be kept"
        );
    }

    #[test]
    fn test_prune_sh_coeffs_stay_in_sync() {
        let n = 4;
        let model = make_model(n, logit(0.001), 0.0_f32);
        let config = DensityConfig::default();
        let ctrl = DensityController::new(n, config);
        let pruned = ctrl.prune_gaussians(&model, None);
        let sh_c = sh_total_for_degree(pruned.sh_degree);
        assert_eq!(
            pruned.sh_coeffs.len(),
            pruned.gaussians.len() * sh_c,
            "SH coefficients must remain in sync after pruning"
        );
    }

    // ── reset_opacity tests ───────────────────────────────────────────────────

    #[test]
    fn test_reset_opacity_sets_correct_value() {
        let n = 5;
        let mut model = make_model(n, logit(0.9), 0.0_f32);
        let config = DensityConfig::default(); // opacity_reset_value = 0.01
        let ctrl = DensityController::new(n, config.clone());
        ctrl.reset_opacity(&mut model);

        for g in &model.gaussians {
            let recovered = sigmoid(g.opacity);
            assert!(
                (recovered - config.opacity_reset_value).abs() < 1e-5,
                "Expected sigmoid(raw) ≈ {}, got {}",
                config.opacity_reset_value,
                recovered
            );
        }
    }

    // ── densify_and_prune tests ───────────────────────────────────────────────

    #[test]
    fn test_densify_and_prune_runs_without_error() {
        let n = 6;
        // High opacity so Gaussians survive pruning; small scale so cloning happens.
        let scale_log = 0.001_f32.ln();
        let model = make_model(n, logit(0.5), scale_log);
        let config = DensityConfig::default();
        let mut ctrl = make_controller_with_grad(n, config, 5e-4_f32);
        let new_model = ctrl.densify_and_prune(&model, None);
        // At minimum the model should be non-empty.
        assert!(
            !new_model.is_empty(),
            "Model should not be empty after densification"
        );
    }

    #[test]
    fn test_densify_and_prune_removes_low_opacity() {
        let n = 4;
        // Very low opacity → pruned after reset would happen; here directly prune.
        let model = make_model(n, logit(0.001), 0.0_f32);
        let config = DensityConfig::default();
        let mut ctrl = make_controller_with_grad(n, config, 5e-4_f32);
        let new_model = ctrl.densify_and_prune(&model, None);
        assert_eq!(
            new_model.gaussians.len(),
            0,
            "All low-opacity Gaussians should be pruned"
        );
    }

    #[test]
    fn test_densify_and_prune_remaps_screen_sizes_through_split() {
        // Regression test: `screen_sizes` must be tested against the
        // ORIGINAL Gaussian a post-split entry derives from, not against
        // whichever index it happens to land on after cloning/splitting
        // shifts everything.
        //
        // Index 0: high gradient + large scale → splits into 2 children.
        // Index 1: zero gradient → left completely untouched (copied as-is).
        let n = 2;
        let scale_log = 0.05_f32.ln(); // exp = 0.05 >= default split threshold 0.01
        let model = make_model(n, logit(0.9), scale_log);
        let config = DensityConfig::default();
        let mut ctrl = DensityController::new(n, config);
        ctrl.accumulator.position_grad_norm[0] = 5e-4_f32; // >= grad_threshold
        ctrl.accumulator.observation_count[0] = 1;
        ctrl.accumulator.position_grad_norm[1] = 0.0_f32; // < grad_threshold
        ctrl.accumulator.observation_count[1] = 1;

        // Screen sizes indexed against the ORIGINAL 2-Gaussian model: index 0
        // is small (its split children should survive), index 1 is over
        // threshold (its untouched copy should be pruned).
        let screen_sizes = [0.01_f32, 0.5_f32];
        let result = ctrl.densify_and_prune(&model, Some(&screen_sizes));

        assert_eq!(
            result.gaussians.len(),
            2,
            "expected both children of index 0 to survive and index 1 to be \
             pruned by its own (correctly-remapped) screen size, got {} Gaussians",
            result.gaussians.len()
        );
        // Both survivors must be children of index 0 (position offset from
        // 0.0 along the dominant axis), never index 1's untouched copy at
        // x=0.1 - if `screen_sizes` were indexed by post-split position
        // instead of by source Gaussian, a child of index 0 would be
        // wrongly pruned and index 1's copy would wrongly survive.
        for g in &result.gaussians {
            assert!(
                (g.position[0] - 0.1_f32).abs() > 1e-6,
                "survivor at x={} looks like index 1's untouched copy, not a split child of index 0",
                g.position[0]
            );
        }
    }

    #[test]
    fn test_densify_and_prune_fused_matches_composed_pipeline() {
        // Regression test proving the one-pass fused implementation
        // (`densify_and_prune_fused`, called by `densify_and_prune`)
        // produces *exactly* the same model - same Gaussians, same SH
        // coefficients, same order - as manually composing
        // clone_gaussians_with_sources -> split_gaussians_with_sources ->
        // prune_gaussians with screen-size remapping, the three-full-model
        // pipeline `densify_and_prune` used to run directly. Exercises
        // every action category (kept, cloned, split) together with both
        // an opacity-based and a screen-size-based prune, so every branch
        // of the fused path is cross-checked in one go.
        let n = 6;
        let mut model = make_model(n, logit(0.9), 0.001_f32.ln()); // small scale by default
                                                                   // Index 0: grad below threshold -> Keep.
                                                                   // Index 1: high grad, small scale -> Clone (survives).
                                                                   // Index 2: high grad, large scale -> Split (survives).
                                                                   // Index 3: high grad, small scale, but low opacity -> pruned entirely.
                                                                   // Index 4: high grad, small scale, oversized screen -> pruned entirely.
                                                                   // Index 5: high grad, large scale -> Split (survives).
        model.gaussians[2].scale = [0.05_f32.ln(); 3];
        model.gaussians[5].scale = [0.05_f32.ln(); 3];
        model.gaussians[3].opacity = logit(0.0001_f32); // below opacity_prune_threshold

        let config = DensityConfig::default();
        let mut ctrl = DensityController::new(n, config);
        for i in [1, 2, 3, 4, 5] {
            ctrl.accumulator.position_grad_norm[i] = 5e-4_f32; // >= grad_threshold
            ctrl.accumulator.observation_count[i] = 1;
        }
        // Index 0 left at grad 0.0 (never accumulated) -> Keep.

        let screen_sizes = [0.01_f32, 0.01, 0.01, 0.01, 0.9, 0.01]; // index 4 oversized

        // Manually compose the three public steps the way `densify_and_prune`
        // used to, remapping `screen_sizes` through clone then split.
        let (cloned_model, _, clone_sources) = ctrl.clone_gaussians_with_sources(&model);
        let cloned_sizes: Vec<f32> = clone_sources.iter().map(|&src| screen_sizes[src]).collect();
        let (split_model, _, split_sources) = ctrl.split_gaussians_with_sources(&cloned_model);
        let split_sizes: Vec<f32> = split_sources.iter().map(|&src| cloned_sizes[src]).collect();
        let composed = ctrl.prune_gaussians(&split_model, Some(&split_sizes));

        let fused = ctrl.densify_and_prune_fused(&model, Some(&screen_sizes));

        assert_eq!(
            composed.gaussians.len(),
            fused.gaussians.len(),
            "fused and composed pipelines must produce the same Gaussian count \
             (composed={}, fused={})",
            composed.gaussians.len(),
            fused.gaussians.len()
        );
        assert_eq!(
            composed.gaussians.len(),
            7,
            "expected 7 survivors: Keep(0), Clone(1)'s 2 entries, Split(2)'s \
             2 children, Split(5)'s 2 children - indices 3 and 4 fully pruned"
        );
        for (i, (a, b)) in composed
            .gaussians
            .iter()
            .zip(fused.gaussians.iter())
            .enumerate()
        {
            assert_eq!(a.position, b.position, "position mismatch at {i}");
            assert_eq!(a.scale, b.scale, "scale mismatch at {i}");
            assert_eq!(a.opacity, b.opacity, "opacity mismatch at {i}");
            assert_eq!(a.rotation, b.rotation, "rotation mismatch at {i}");
        }
        assert_eq!(
            composed.sh_coeffs, fused.sh_coeffs,
            "SH coefficients must match between fused and composed pipelines"
        );
    }

    // ── sync_to_model tests ───────────────────────────────────────────────────

    #[test]
    fn test_sync_to_model_updates_accumulator_size() {
        let n = 4;
        let model_a = make_model(n, logit(0.5), 0.0_f32);
        let config = DensityConfig::default();
        let mut ctrl = DensityController::new(n, config);

        // Simulate the model growing.
        let model_b = make_model(10, logit(0.5), 0.0_f32);
        ctrl.sync_to_model(&model_b);
        assert_eq!(ctrl.accumulator.position_grad_norm.len(), 10);
        assert_eq!(ctrl.accumulator.observation_count.len(), 10);

        // Simulate the model shrinking.
        ctrl.sync_to_model(&model_a);
        assert_eq!(ctrl.accumulator.position_grad_norm.len(), n);
    }

    #[test]
    fn test_sync_to_model_resets_accumulator() {
        let n = 4;
        let config = DensityConfig::default();
        let mut ctrl = make_controller_with_grad(n, config, 1.0_f32);
        // After sync the accumulator should be zeroed.
        let model = make_model(n, logit(0.5), 0.0_f32);
        ctrl.sync_to_model(&model);
        for &v in &ctrl.accumulator.position_grad_norm {
            assert_eq!(v, 0.0_f32);
        }
        for &c in &ctrl.accumulator.observation_count {
            assert_eq!(c, 0_u32);
        }
    }

    // ── FLAME binding field sync tests ────────────────────────────────────────

    #[test]
    fn test_clone_preserves_flame_fields_in_sync() {
        let n = 3;
        let scale_log = 0.001_f32.ln();
        let mut model = make_model(n, logit(0.5), scale_log);
        // Give distinct face_indices so we can verify propagation.
        model.face_indices = vec![1, 2, 3];

        let config = DensityConfig::default();
        let ctrl = make_controller_with_grad(n, config, 5e-4_f32);
        let (new_model, _) = ctrl.clone_gaussians(&model);

        // All FLAME vecs must have the same length as gaussians.
        assert_eq!(new_model.face_indices.len(), new_model.gaussians.len());
        assert_eq!(new_model.barycentric.len(), new_model.gaussians.len());
        assert_eq!(new_model.local_offsets.len(), new_model.gaussians.len());
        assert_eq!(new_model.is_rigid.len(), new_model.gaussians.len());
    }
}
