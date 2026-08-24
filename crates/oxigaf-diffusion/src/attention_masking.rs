//! Attention masks for multi-view diffusion cross-attention layers.
//!
//! This module computes attention masks that control which query-key pairs are
//! allowed to attend to each other in the multi-view diffusion U-Net. Masks
//! enable camera-conditioned, view-consistent generation by restricting or
//! allowing attention across view boundaries.
//!
//! ## Sequence Structure
//!
//! In multi-view diffusion the sequence is structured as:
//! - Total tokens = `num_views × tokens_per_view`
//! - View `v` occupies tokens `[v * tokens_per_view, (v+1) * tokens_per_view)`
//!
//! ## Usage
//!
//! ```rust
//! use oxigaf_diffusion::attention_masking::{
//!     full_mask, causal_mask, self_view_mask, MaskPattern, build_mask,
//! };
//!
//! // Full attention: every token attends to every other token
//! let mask = full_mask(16);
//! assert_eq!(mask.density(), 1.0);
//!
//! // Self-view only: each view is isolated
//! let mask = self_view_mask(4, 16).unwrap();
//! assert!(mask.density() < 1.0);
//!
//! // Build from pattern
//! let mask = build_mask(
//!     &MaskPattern::Full,
//!     4, 16, None,
//! ).unwrap();
//! ```

use thiserror::Error;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur when building or operating on attention masks.
#[derive(Debug, Error, PartialEq)]
pub enum AttentionMaskError {
    /// Sequence length is zero or dimensions are incompatible.
    #[error("Invalid dimensions: {0}")]
    InvalidDimensions(String),

    /// Configuration parameter is invalid (e.g., `num_views = 0`).
    #[error("Invalid config: {0}")]
    InvalidConfig(String),

    /// Unrecognized or unsupported mask pattern.
    #[error("Invalid pattern: {0}")]
    InvalidPattern(String),
}

// ---------------------------------------------------------------------------
// AttentionMask
// ---------------------------------------------------------------------------

/// A boolean attention mask: `true` = allow attention, `false` = mask out.
///
/// Stored as a flat, row-major `[seq_len × seq_len]` vector of booleans.
/// Entry `mask[q * seq_len + k]` indicates whether query position `q` may
/// attend to key position `k`.
#[derive(Debug, Clone, PartialEq)]
pub struct AttentionMask {
    /// Length of both the query and key sequences.
    pub seq_len: usize,
    /// Flat row-major boolean mask of length `seq_len * seq_len`.
    pub mask: Vec<bool>,
}

impl AttentionMask {
    /// Construct a new mask filled with `default` for all entries.
    pub fn new(seq_len: usize, default: bool) -> Self {
        Self {
            seq_len,
            mask: vec![default; seq_len * seq_len],
        }
    }

    /// Return the mask value for query position `query` attending to key
    /// position `key`.
    ///
    /// Returns `false` for any out-of-bounds index, *and* for any in-range
    /// index that still falls outside `mask`'s actual length. Both `seq_len`
    /// and `mask` are public fields with no invariant enforced between them
    /// outside of [`AttentionMask::new`], so a hand-built (or partially
    /// mutated) `AttentionMask` can have `mask.len() != seq_len * seq_len`;
    /// this never panics regardless.
    pub fn get(&self, query: usize, key: usize) -> bool {
        if query >= self.seq_len || key >= self.seq_len {
            return false;
        }
        self.mask
            .get(query * self.seq_len + key)
            .copied()
            .unwrap_or(false)
    }

    /// Set the mask value for query position `query` attending to key
    /// position `key`.
    ///
    /// No-op for any out-of-bounds index, or any in-range index that falls
    /// outside `mask`'s actual length (see [`AttentionMask::get`]).
    pub fn set(&mut self, query: usize, key: usize, value: bool) {
        if query >= self.seq_len || key >= self.seq_len {
            return;
        }
        if let Some(slot) = self.mask.get_mut(query * self.seq_len + key) {
            *slot = value;
        }
    }

    /// Return an inverted copy of this mask (`true` ↔ `false`).
    pub fn invert(&self) -> Self {
        Self {
            seq_len: self.seq_len,
            mask: self.mask.iter().map(|&b| !b).collect(),
        }
    }

    /// Logical AND: both masks must allow for result to allow.
    ///
    /// # Errors
    /// Returns [`AttentionMaskError::InvalidDimensions`] when `other` has a
    /// different `seq_len`.
    pub fn and(&self, other: &AttentionMask) -> Result<AttentionMask, AttentionMaskError> {
        if self.seq_len != other.seq_len {
            return Err(AttentionMaskError::InvalidDimensions(format!(
                "AND: seq_len mismatch: {} vs {}",
                self.seq_len, other.seq_len
            )));
        }
        let mask = self
            .mask
            .iter()
            .zip(other.mask.iter())
            .map(|(&a, &b)| a && b)
            .collect();
        Ok(AttentionMask {
            seq_len: self.seq_len,
            mask,
        })
    }

    /// Logical OR: either mask allowing is sufficient.
    ///
    /// # Errors
    /// Returns [`AttentionMaskError::InvalidDimensions`] when `other` has a
    /// different `seq_len`.
    pub fn or(&self, other: &AttentionMask) -> Result<AttentionMask, AttentionMaskError> {
        if self.seq_len != other.seq_len {
            return Err(AttentionMaskError::InvalidDimensions(format!(
                "OR: seq_len mismatch: {} vs {}",
                self.seq_len, other.seq_len
            )));
        }
        let mask = self
            .mask
            .iter()
            .zip(other.mask.iter())
            .map(|(&a, &b)| a || b)
            .collect();
        Ok(AttentionMask {
            seq_len: self.seq_len,
            mask,
        })
    }

    /// Convert to an additive bias vector for use in softmax.
    ///
    /// - `true`  → `0.0_f32`      (unmasked: no penalty)
    /// - `false` → `-65504.0_f32` (masked: minimum finite f16 value)
    ///
    /// `-65504.0` (rather than e.g. `-1e9` or `f32::NEG_INFINITY`) is chosen
    /// deliberately: it is exactly representable in both f32 and f16, so it
    /// never overflows when this bias is added under mixed-precision
    /// (f16) training. It is also finite, which matters when an entire query
    /// row happens to be fully masked (e.g. a narrow local window at a
    /// sequence boundary) — with a finite bias, `softmax(row - max(row))`
    /// degrades to a uniform distribution over that row instead of computing
    /// `exp(-inf - (-inf)) = exp(NaN)`, which `f32::NEG_INFINITY` would
    /// produce and then propagate through the rest of the network.
    ///
    /// The returned vector has length `seq_len * seq_len`.
    pub fn to_bias(&self) -> Vec<f32> {
        self.mask
            .iter()
            .map(|&b| if b { 0.0_f32 } else { -65504.0_f32 })
            .collect()
    }

    /// Convert to a multiplicative float mask.
    ///
    /// - `true`  → `1.0_f32`
    /// - `false` → `0.0_f32`
    pub fn to_float_mask(&self) -> Vec<f32> {
        self.mask
            .iter()
            .map(|&b| if b { 1.0_f32 } else { 0.0_f32 })
            .collect()
    }

    /// Fraction of entries that are `true` (allowed attention).
    ///
    /// Returns `0.0` when `seq_len == 0`.
    pub fn density(&self) -> f32 {
        let total = self.mask.len();
        if total == 0 {
            return 0.0;
        }
        let allowed = self.mask.iter().filter(|&&b| b).count();
        allowed as f32 / total as f32
    }

    /// Render the mask as an ASCII grid.
    ///
    /// `O` = allowed, `.` = masked.  Only up to 16 × 16 entries are shown;
    /// larger masks are truncated.
    pub fn format_ascii(&self) -> String {
        let display_len = self.seq_len.min(16);
        let mut out = String::new();
        for q in 0..display_len {
            for k in 0..display_len {
                let ch = if self.get(q, k) { 'O' } else { '.' };
                out.push(ch);
            }
            out.push('\n');
        }
        out
    }

    /// Apply a padding mask, zeroing out rows / columns for padding tokens.
    ///
    /// - `padding_mask[q] == false` → query `q` is padding, attends to nothing
    ///   (entire row set to `false`).
    /// - `padding_mask[k] == false` → key `k` is padding, nothing attends to
    ///   it (entire column set to `false`).
    ///
    /// # Errors
    /// Returns [`AttentionMaskError::InvalidDimensions`] when
    /// `padding_mask.len() != seq_len`.
    pub fn apply_padding(
        &self,
        padding_mask: &[bool],
    ) -> Result<AttentionMask, AttentionMaskError> {
        if padding_mask.len() != self.seq_len {
            return Err(AttentionMaskError::InvalidDimensions(format!(
                "apply_padding: padding_mask length {} != seq_len {}",
                padding_mask.len(),
                self.seq_len
            )));
        }
        let mut result = self.clone();
        for (q, &q_valid) in padding_mask.iter().enumerate() {
            if !q_valid {
                // padding query: zero entire row
                for k in 0..self.seq_len {
                    result.set(q, k, false);
                }
            }
        }
        for (k, &k_valid) in padding_mask.iter().enumerate() {
            if !k_valid {
                // padding key: zero entire column
                for q in 0..self.seq_len {
                    result.set(q, k, false);
                }
            }
        }
        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// Simple masks
// ---------------------------------------------------------------------------

/// Create a causal (lower-triangular) attention mask.
///
/// Query `i` may attend to key `j` only when `j <= i`.
///
/// # Errors
/// Returns [`AttentionMaskError::InvalidDimensions`] when `seq_len == 0`.
pub fn causal_mask(seq_len: usize) -> Result<AttentionMask, AttentionMaskError> {
    if seq_len == 0 {
        return Err(AttentionMaskError::InvalidDimensions(
            "causal_mask: seq_len must be > 0".to_string(),
        ));
    }
    let mut m = AttentionMask::new(seq_len, false);
    for q in 0..seq_len {
        for k in 0..=q {
            m.set(q, k, true);
        }
    }
    Ok(m)
}

/// Create a full (all-to-all) attention mask.
///
/// Every query attends to every key; the mask is entirely `true`.
pub fn full_mask(seq_len: usize) -> AttentionMask {
    AttentionMask::new(seq_len, true)
}

/// Create a local (sliding-window) attention mask.
///
/// Query `i` attends to keys in the range `[i - window, i + window]`
/// (clamped to valid indices).
///
/// When `window == 0` only the diagonal (`i == j`) is allowed.
///
/// # Errors
/// Returns [`AttentionMaskError::InvalidDimensions`] when `seq_len == 0`.
pub fn local_mask(seq_len: usize, window: usize) -> Result<AttentionMask, AttentionMaskError> {
    if seq_len == 0 {
        return Err(AttentionMaskError::InvalidDimensions(
            "local_mask: seq_len must be > 0".to_string(),
        ));
    }
    let mut m = AttentionMask::new(seq_len, false);
    for q in 0..seq_len {
        let lo = q.saturating_sub(window);
        let hi = (q + window).min(seq_len - 1);
        for k in lo..=hi {
            m.set(q, k, true);
        }
    }
    Ok(m)
}

// ---------------------------------------------------------------------------
// Multi-view masks
// ---------------------------------------------------------------------------

/// Validate common multi-view parameters, returning `seq_len` on success.
fn validate_multiview(
    num_views: usize,
    tokens_per_view: usize,
) -> Result<usize, AttentionMaskError> {
    if num_views == 0 {
        return Err(AttentionMaskError::InvalidConfig(
            "num_views must be > 0".to_string(),
        ));
    }
    if tokens_per_view == 0 {
        return Err(AttentionMaskError::InvalidConfig(
            "tokens_per_view must be > 0".to_string(),
        ));
    }
    Ok(num_views * tokens_per_view)
}

/// Allow attention within a view's block `[v_start, v_end)` for both Q and K.
fn allow_view_block(
    m: &mut AttentionMask,
    q_start: usize,
    q_end: usize,
    k_start: usize,
    k_end: usize,
) {
    for q in q_start..q_end {
        for k in k_start..k_end {
            m.set(q, k, true);
        }
    }
}

/// Self-attention within each view only.
///
/// Each token can only attend to tokens in the same view; cross-view
/// attention is fully blocked.
///
/// # Errors
/// Returns [`AttentionMaskError::InvalidConfig`] when `num_views == 0` or
/// `tokens_per_view == 0`.
pub fn self_view_mask(
    num_views: usize,
    tokens_per_view: usize,
) -> Result<AttentionMask, AttentionMaskError> {
    let seq_len = validate_multiview(num_views, tokens_per_view)?;
    let mut m = AttentionMask::new(seq_len, false);
    for v in 0..num_views {
        let start = v * tokens_per_view;
        let end = start + tokens_per_view;
        allow_view_block(&mut m, start, end, start, end);
    }
    Ok(m)
}

/// Cross-view (all-to-all) attention across all views.
///
/// Equivalent to `full_mask(num_views * tokens_per_view)`.
///
/// # Errors
/// Returns [`AttentionMaskError::InvalidConfig`] when `num_views == 0` or
/// `tokens_per_view == 0`.
pub fn cross_view_mask(
    num_views: usize,
    tokens_per_view: usize,
) -> Result<AttentionMask, AttentionMaskError> {
    let seq_len = validate_multiview(num_views, tokens_per_view)?;
    Ok(full_mask(seq_len))
}

/// Reference-view attention: every token attends to view 0 plus its own view.
///
/// A token in view `v` can attend to:
/// - All tokens in view 0 (the reference view)
/// - All tokens in view `v` (self-view)
/// - Additionally, tokens in the reference view itself (view 0) attend to
///   *all* tokens in *every* view — the reference view's query row is
///   unrestricted, not just self+reference like the other views.
///
/// # Errors
/// Returns [`AttentionMaskError::InvalidConfig`] when `num_views == 0` or
/// `tokens_per_view == 0`.
pub fn reference_view_mask(
    num_views: usize,
    tokens_per_view: usize,
) -> Result<AttentionMask, AttentionMaskError> {
    let seq_len = validate_multiview(num_views, tokens_per_view)?;
    let mut m = AttentionMask::new(seq_len, false);

    // Reference view (view 0) block
    let ref_start = 0_usize;
    let ref_end = tokens_per_view;

    for v in 0..num_views {
        let v_start = v * tokens_per_view;
        let v_end = v_start + tokens_per_view;

        // Each view attends to its own tokens
        allow_view_block(&mut m, v_start, v_end, v_start, v_end);

        // Each view attends to the reference view tokens
        allow_view_block(&mut m, v_start, v_end, ref_start, ref_end);

        // The reference view tokens attend to every view's tokens
        // (since every view allows reference → this already set, but also set
        // reference row → v_k direction)
        allow_view_block(&mut m, ref_start, ref_end, v_start, v_end);
    }

    Ok(m)
}

/// Nearest-neighbor view attention.
///
/// Each view attends to itself plus each view listed in `view_neighbors[v]`.
///
/// # Errors
/// - [`AttentionMaskError::InvalidConfig`] when `num_views == 0`, `tokens_per_view == 0`,
///   or `view_neighbors.len() != num_views`.
/// - [`AttentionMaskError::InvalidDimensions`] when any neighbor index is `>= num_views`.
pub fn neighbor_view_mask(
    num_views: usize,
    tokens_per_view: usize,
    view_neighbors: &[Vec<usize>],
) -> Result<AttentionMask, AttentionMaskError> {
    let seq_len = validate_multiview(num_views, tokens_per_view)?;
    if view_neighbors.len() != num_views {
        return Err(AttentionMaskError::InvalidConfig(format!(
            "neighbor_view_mask: view_neighbors length {} != num_views {}",
            view_neighbors.len(),
            num_views
        )));
    }
    for (v, neighbors) in view_neighbors.iter().enumerate() {
        for &n in neighbors {
            if n >= num_views {
                return Err(AttentionMaskError::InvalidDimensions(format!(
                    "neighbor_view_mask: neighbor index {} out of range for view {} \
                     (num_views = {})",
                    n, v, num_views
                )));
            }
        }
    }

    let mut m = AttentionMask::new(seq_len, false);
    for (v, neighbors) in view_neighbors.iter().enumerate() {
        let v_start = v * tokens_per_view;
        let v_end = v_start + tokens_per_view;

        // Self-attention
        allow_view_block(&mut m, v_start, v_end, v_start, v_end);

        // Neighbor attention (bidirectional: v→n and n→v)
        for &n in neighbors {
            let n_start = n * tokens_per_view;
            let n_end = n_start + tokens_per_view;
            allow_view_block(&mut m, v_start, v_end, n_start, n_end);
            allow_view_block(&mut m, n_start, n_end, v_start, v_end);
        }
    }
    Ok(m)
}

/// Ring attention: each view attends to itself and its two circular neighbors.
///
/// View `v` attends to views `(v - 1) % N` and `(v + 1) % N`.  With a single
/// view the ring collapses to self-attention.
///
/// # Errors
/// Returns [`AttentionMaskError::InvalidConfig`] when `num_views == 0` or
/// `tokens_per_view == 0`.
pub fn ring_view_mask(
    num_views: usize,
    tokens_per_view: usize,
) -> Result<AttentionMask, AttentionMaskError> {
    let _ = validate_multiview(num_views, tokens_per_view)?;

    // Build neighbor lists for a circular ring
    let neighbors: Vec<Vec<usize>> = (0..num_views)
        .map(|v| {
            if num_views == 1 {
                vec![]
            } else {
                let prev = (v + num_views - 1) % num_views;
                let next = (v + 1) % num_views;
                if prev == next {
                    vec![prev]
                } else {
                    vec![prev, next]
                }
            }
        })
        .collect();

    neighbor_view_mask(num_views, tokens_per_view, &neighbors)
}

// ---------------------------------------------------------------------------
// Camera-conditioned mask
// ---------------------------------------------------------------------------

/// Build a mask based on angular distance between camera views.
///
/// Views within `max_angle_rad` of each other (inclusive) may attend across
/// their entire token blocks.  Self-attention (same view) is always included.
///
/// Angular distance between `(az1, el1)` and `(az2, el2)`:
/// ```text
/// dist = acos(clamp(sin(el1)*sin(el2) + cos(el1)*cos(el2)*cos(az1-az2), -1, 1))
/// ```
///
/// # Errors
/// - [`AttentionMaskError::InvalidConfig`] when `num_views == 0`,
///   `tokens_per_view == 0`, or `view_positions.len() != num_views`.
pub fn angular_proximity_mask(
    num_views: usize,
    tokens_per_view: usize,
    view_positions: &[(f32, f32)],
    max_angle_rad: f32,
) -> Result<AttentionMask, AttentionMaskError> {
    let seq_len = validate_multiview(num_views, tokens_per_view)?;
    if view_positions.len() != num_views {
        return Err(AttentionMaskError::InvalidConfig(format!(
            "angular_proximity_mask: view_positions length {} != num_views {}",
            view_positions.len(),
            num_views
        )));
    }

    let mut m = AttentionMask::new(seq_len, false);

    for (va, &(az1, el1)) in view_positions.iter().enumerate() {
        for (vb, &(az2, el2)) in view_positions.iter().enumerate() {
            let dot = (el1.sin() * el2.sin() + el1.cos() * el2.cos() * (az1 - az2).cos())
                .clamp(-1.0, 1.0);
            let dist = dot.acos();

            if dist <= max_angle_rad {
                let a_start = va * tokens_per_view;
                let a_end = a_start + tokens_per_view;
                let b_start = vb * tokens_per_view;
                let b_end = b_start + tokens_per_view;
                allow_view_block(&mut m, a_start, a_end, b_start, b_end);
            }
        }
    }

    Ok(m)
}

// ---------------------------------------------------------------------------
// MaskPattern & build_mask
// ---------------------------------------------------------------------------

/// Named attention mask patterns.
#[derive(Debug, Clone, PartialEq)]
pub enum MaskPattern {
    /// All-to-all attention (every query attends to every key).
    Full,
    /// Causal (lower-triangular) attention.
    Causal,
    /// Local sliding-window attention with the given half-width.
    Local {
        /// Half-width of the attention window (tokens on each side).
        window: usize,
    },
    /// Self-attention within each view only.
    SelfView,
    /// Full cross-view attention (equivalent to `Full`).
    CrossView,
    /// Reference-view attention: all tokens attend to view 0 plus their own view.
    ReferenceView,
    /// Ring attention: each view attends to its two circular neighbors.
    Ring,
    /// Camera-angle-conditioned attention with the given maximum angle.
    AngularProximity {
        /// Maximum allowed angular distance in radians.
        max_angle_rad: f32,
    },
}

/// Build an attention mask from a named [`MaskPattern`].
///
/// For patterns that require camera positions (`AngularProximity`),
/// `view_positions` must be `Some`.
///
/// # Errors
/// - [`AttentionMaskError::InvalidConfig`] for invalid parameters.
/// - [`AttentionMaskError::InvalidPattern`] when `AngularProximity` is
///   requested but `view_positions` is `None`.
pub fn build_mask(
    pattern: &MaskPattern,
    num_views: usize,
    tokens_per_view: usize,
    view_positions: Option<&[(f32, f32)]>,
) -> Result<AttentionMask, AttentionMaskError> {
    match pattern {
        MaskPattern::Full => {
            let seq_len = validate_multiview(num_views, tokens_per_view)?;
            Ok(full_mask(seq_len))
        }
        MaskPattern::Causal => {
            let seq_len = validate_multiview(num_views, tokens_per_view)?;
            causal_mask(seq_len)
        }
        MaskPattern::Local { window } => {
            let seq_len = validate_multiview(num_views, tokens_per_view)?;
            local_mask(seq_len, *window)
        }
        MaskPattern::SelfView => self_view_mask(num_views, tokens_per_view),
        MaskPattern::CrossView => cross_view_mask(num_views, tokens_per_view),
        MaskPattern::ReferenceView => reference_view_mask(num_views, tokens_per_view),
        MaskPattern::Ring => ring_view_mask(num_views, tokens_per_view),
        MaskPattern::AngularProximity { max_angle_rad } => {
            let positions = view_positions.ok_or_else(|| {
                AttentionMaskError::InvalidPattern(
                    "AngularProximity requires view_positions".to_string(),
                )
            })?;
            angular_proximity_mask(num_views, tokens_per_view, positions, *max_angle_rad)
        }
    }
}

// ---------------------------------------------------------------------------
// LayerMaskConfig & build_layer_mask
// ---------------------------------------------------------------------------

/// Configuration for per-layer attention masks in a multi-view U-Net.
pub struct LayerMaskConfig {
    /// Whether to use cross-view attention at this layer.
    pub cross_view: bool,
    /// Pattern for self-attention (within each view's token block).
    pub self_pattern: MaskPattern,
    /// Pattern for cross-view attention (used when `cross_view` is `true`).
    pub cross_pattern: Option<MaskPattern>,
}

/// Build the combined attention mask for a U-Net layer.
///
/// When `cross_view` is `false`: returns a pure self-view mask, with the
/// `self_pattern` applied within each view block (block-diagonal structure).
///
/// When `cross_view` is `true`: builds the self-view block-diagonal mask
/// (applying `self_pattern` within each block), builds the cross-view mask
/// from `cross_pattern`, then ORs them together.
///
/// `self_pattern` only accepts patterns meaningful *within* a single view's
/// token block: [`MaskPattern::Full`], [`MaskPattern::Causal`],
/// [`MaskPattern::Local`], or [`MaskPattern::AngularProximity`] (which
/// degrades to `Full` at the intra-view level; the angular conditioning
/// itself only applies across views). The multi-view-only patterns
/// ([`MaskPattern::SelfView`], [`MaskPattern::CrossView`],
/// [`MaskPattern::ReferenceView`], [`MaskPattern::Ring`]) describe
/// relationships *between* views and have no meaningful definition as a
/// `self_pattern`; use `cross_pattern` for those instead.
///
/// # Errors
/// - [`AttentionMaskError::InvalidConfig`] for invalid parameters.
/// - [`AttentionMaskError::InvalidPattern`] when `self_pattern` is one of the
///   multi-view-only patterns listed above, or (for `cross_pattern`) when
///   `AngularProximity` is requested without `view_positions`.
pub fn build_layer_mask(
    config: &LayerMaskConfig,
    num_views: usize,
    tokens_per_view: usize,
    view_positions: Option<&[(f32, f32)]>,
) -> Result<AttentionMask, AttentionMaskError> {
    let seq_len = validate_multiview(num_views, tokens_per_view)?;

    // Build the intra-view (block-diagonal) self-attention mask.
    // First compute what the pattern looks like for a single view block,
    // then tile it along the block diagonal.
    let intra_mask = build_intra_view_mask(
        &config.self_pattern,
        num_views,
        tokens_per_view,
        seq_len,
        view_positions,
    )?;

    if !config.cross_view {
        return Ok(intra_mask);
    }

    // Cross-view: build the inter-view mask and OR with the intra mask.
    let cross_pattern = config.cross_pattern.as_ref().ok_or_else(|| {
        AttentionMaskError::InvalidConfig(
            "build_layer_mask: cross_view=true but cross_pattern is None".to_string(),
        )
    })?;

    let inter_mask = build_mask(cross_pattern, num_views, tokens_per_view, view_positions)?;

    intra_mask.or(&inter_mask)
}

/// Build the block-diagonal intra-view mask.
///
/// For each view block `[v*tpv, (v+1)*tpv)` apply `pattern` to the
/// `tokens_per_view × tokens_per_view` sub-block.
fn build_intra_view_mask(
    pattern: &MaskPattern,
    num_views: usize,
    tokens_per_view: usize,
    seq_len: usize,
    view_positions: Option<&[(f32, f32)]>,
) -> Result<AttentionMask, AttentionMaskError> {
    // Compute per-block mask (tokens_per_view × tokens_per_view)
    let block_mask = match pattern {
        MaskPattern::Full => full_mask(tokens_per_view),
        MaskPattern::Causal => causal_mask(tokens_per_view)?,
        MaskPattern::Local { window } => local_mask(tokens_per_view, *window)?,
        // These patterns describe relationships *between* views (which view
        // attends to which other view); they have no meaningful definition
        // within a single view's token block, where there is no "other view"
        // to distinguish. Silently treating them as full self-attention
        // (the previous behavior) made them indistinguishable from
        // `MaskPattern::Full` with no error and no warning, even though this
        // function's caller (`build_layer_mask`) documents
        // `AttentionMaskError::InvalidPattern` as a possible error for
        // exactly this kind of pattern mismatch.
        MaskPattern::SelfView
        | MaskPattern::CrossView
        | MaskPattern::ReferenceView
        | MaskPattern::Ring => {
            return Err(AttentionMaskError::InvalidPattern(format!(
                "{pattern:?} is a multi-view (inter-view) pattern and has no meaningful \
                 definition as a self_pattern (intra-view); use MaskPattern::Full, Causal, \
                 or Local instead"
            )));
        }
        MaskPattern::AngularProximity { .. } => {
            // Angular proximity within a single view: use full mask for the block.
            // The actual angular conditioning applies at the cross-view level.
            let _ = view_positions; // consumed at the inter-view step
            full_mask(tokens_per_view)
        }
    };

    // Tile the block mask onto the block diagonal of the full sequence mask
    let mut full = AttentionMask::new(seq_len, false);
    for v in 0..num_views {
        let offset = v * tokens_per_view;
        for q_local in 0..tokens_per_view {
            for k_local in 0..tokens_per_view {
                if block_mask.get(q_local, k_local) {
                    full.set(offset + q_local, offset + k_local, true);
                }
            }
        }
    }
    Ok(full)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Test 1: full_mask — all entries True
    // -------------------------------------------------------------------------
    #[test]
    fn test_full_mask_all_true() {
        let m = full_mask(4);
        assert_eq!(m.seq_len, 4);
        assert!(m.mask.iter().all(|&b| b));
    }

    // -------------------------------------------------------------------------
    // Test 2: full_mask density = 1.0
    // -------------------------------------------------------------------------
    #[test]
    fn test_full_mask_density() {
        let m = full_mask(8);
        let d = m.density();
        assert!((d - 1.0).abs() < 1e-6, "density should be 1.0, got {d}");
    }

    // -------------------------------------------------------------------------
    // Test 3: causal_mask — diagonal and below True, above False
    // -------------------------------------------------------------------------
    #[test]
    fn test_causal_mask_structure() {
        let m = causal_mask(5).unwrap();
        for q in 0..5 {
            for k in 0..5 {
                if k <= q {
                    assert!(m.get(q, k), "causal: q={q} k={k} should be true");
                } else {
                    assert!(!m.get(q, k), "causal: q={q} k={k} should be false");
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // Test 4: causal_mask(0) → Err
    // -------------------------------------------------------------------------
    #[test]
    fn test_causal_mask_zero_seq_len_err() {
        let result = causal_mask(0);
        assert!(
            matches!(result, Err(AttentionMaskError::InvalidDimensions(_))),
            "expected InvalidDimensions error"
        );
    }

    // -------------------------------------------------------------------------
    // Test 5: local_mask window=0 → diagonal only
    // -------------------------------------------------------------------------
    #[test]
    fn test_local_mask_window_zero_diagonal_only() {
        let m = local_mask(5, 0).unwrap();
        for q in 0..5 {
            for k in 0..5 {
                if q == k {
                    assert!(
                        m.get(q, k),
                        "local w=0: diagonal q={q} k={k} should be true"
                    );
                } else {
                    assert!(
                        !m.get(q, k),
                        "local w=0: off-diagonal q={q} k={k} should be false"
                    );
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // Test 6: local_mask window=1 → one neighbor each side
    // -------------------------------------------------------------------------
    #[test]
    fn test_local_mask_window_one() {
        let m = local_mask(5, 1).unwrap();
        // For q=2 (middle), k in {1,2,3}
        assert!(m.get(2, 1));
        assert!(m.get(2, 2));
        assert!(m.get(2, 3));
        // k=0 and k=4 should be blocked for q=2
        assert!(!m.get(2, 0));
        assert!(!m.get(2, 4));
        // Edge: q=0 can attend k=0 and k=1, but not k=2..4
        assert!(m.get(0, 0));
        assert!(m.get(0, 1));
        assert!(!m.get(0, 2));
    }

    // -------------------------------------------------------------------------
    // Test 7: invert — flips all values
    // -------------------------------------------------------------------------
    #[test]
    fn test_invert_flips_all() {
        let m = causal_mask(4).unwrap();
        let inv = m.invert();
        for q in 0..4 {
            for k in 0..4 {
                assert_ne!(
                    m.get(q, k),
                    inv.get(q, k),
                    "invert: q={q} k={k} should differ"
                );
            }
        }
    }

    // -------------------------------------------------------------------------
    // Test 8: and — only True where both True
    // -------------------------------------------------------------------------
    #[test]
    fn test_and_logic() {
        let a = causal_mask(4).unwrap();
        let b = causal_mask(4).unwrap().invert();
        let result = a.and(&b).unwrap();
        // causal AND ~causal = all false
        assert!(result.mask.iter().all(|&x| !x));
    }

    // -------------------------------------------------------------------------
    // Test 9: or — True where either True
    // -------------------------------------------------------------------------
    #[test]
    fn test_or_logic() {
        let a = causal_mask(4).unwrap();
        let b = a.invert();
        let result = a.or(&b).unwrap();
        // causal OR ~causal = all true
        assert!(result.mask.iter().all(|&x| x));
    }

    // -------------------------------------------------------------------------
    // Test 10: to_bias — True→0.0, False→-65504.0 (finite, f16-safe)
    // -------------------------------------------------------------------------
    #[test]
    fn test_to_bias_values() {
        let mut m = AttentionMask::new(2, false);
        m.set(0, 0, true);
        m.set(1, 1, true);
        let bias = m.to_bias();
        assert!((bias[0] - 0.0).abs() < 1e-7, "True should map to 0.0");
        assert!(
            (bias[1] - (-65504.0)).abs() < 1.0,
            "False should map to -65504.0"
        );
        assert!((bias[3] - 0.0).abs() < 1e-7, "True should map to 0.0");
    }

    #[test]
    fn test_to_bias_is_finite_and_f16_representable() {
        // -65504.0 is f16::MIN (the most negative finite f16 value), so it
        // must never overflow to infinity when narrowed to f16.
        let m = AttentionMask::new(3, false);
        let bias = m.to_bias();
        assert!(bias.iter().all(|v| v.is_finite()), "bias must be finite");
        assert!(bias.iter().all(|&v| v == -65504.0));
    }

    #[test]
    fn test_get_set_never_panics_on_mismatched_mask_length() {
        // Hand-built AttentionMask where mask.len() != seq_len * seq_len.
        // get/set must degrade gracefully instead of panicking.
        let mut m = AttentionMask {
            seq_len: 4,
            mask: vec![true; 3], // way shorter than 4*4=16
        };
        assert!(!m.get(3, 3), "out-of-buffer read should return false");
        assert!(m.get(0, 0), "in-buffer read should still work");
        m.set(3, 3, true); // must not panic
        assert!(!m.get(3, 3), "out-of-buffer write is a no-op");
    }

    // -------------------------------------------------------------------------
    // Test 11: density — correct fraction
    // -------------------------------------------------------------------------
    #[test]
    fn test_density_correct_fraction() {
        let m = causal_mask(4).unwrap();
        // Lower triangle including diagonal: 1+2+3+4 = 10 of 16 entries
        let d = m.density();
        let expected = 10.0 / 16.0;
        assert!(
            (d - expected).abs() < 1e-6,
            "density: expected {expected}, got {d}"
        );
    }

    // -------------------------------------------------------------------------
    // Test 12: self_view_mask — same-view attends, cross-view blocked
    // -------------------------------------------------------------------------
    #[test]
    fn test_self_view_mask_structure() {
        let m = self_view_mask(3, 4).unwrap(); // 3 views, 4 tokens each → seq=12
                                               // Within view 0 (tokens 0–3)
        assert!(m.get(0, 3));
        assert!(m.get(3, 0));
        // Within view 1 (tokens 4–7)
        assert!(m.get(4, 6));
        // Cross-view: view 0 → view 1 should be blocked
        assert!(!m.get(0, 4));
        assert!(!m.get(3, 4));
    }

    // -------------------------------------------------------------------------
    // Test 13: cross_view_mask — equivalent to full_mask
    // -------------------------------------------------------------------------
    #[test]
    fn test_cross_view_mask_equals_full() {
        let cv = cross_view_mask(3, 4).unwrap();
        let full = full_mask(12);
        assert_eq!(cv, full);
    }

    // -------------------------------------------------------------------------
    // Test 14: reference_view_mask — view 0 tokens attend to all views
    // -------------------------------------------------------------------------
    #[test]
    fn test_reference_view_mask_view0_attends_all() {
        let m = reference_view_mask(3, 4).unwrap();
        // view 0 query tokens (0–3) should be able to attend to all views
        for q in 0..4 {
            for k in 0..12 {
                assert!(
                    m.get(q, k),
                    "view 0 query {q} should attend to all keys, failed at k={k}"
                );
            }
        }
    }

    // -------------------------------------------------------------------------
    // Test 15: reference_view_mask — view 1 attends to view 0 + itself
    // -------------------------------------------------------------------------
    #[test]
    fn test_reference_view_mask_view1_attends_ref_and_self() {
        let m = reference_view_mask(3, 4).unwrap(); // views: 0→[0,4), 1→[4,8), 2→[8,12)
                                                    // View 1 queries (4–7) attend to view 0 (0–3)
        for q in 4..8 {
            for k in 0..4 {
                assert!(m.get(q, k), "view 1 query {q} → ref key {k} should be true");
            }
        }
        // View 1 queries attend to their own tokens (4–7)
        for q in 4..8 {
            for k in 4..8 {
                assert!(m.get(q, k), "view 1 query {q} → own key {k} should be true");
            }
        }
        // View 1 queries do NOT attend to view 2 (8–11)
        for q in 4..8 {
            for k in 8..12 {
                assert!(
                    !m.get(q, k),
                    "view 1 query {q} → view 2 key {k} should be false"
                );
            }
        }
    }

    // -------------------------------------------------------------------------
    // Test 16: ring_view_mask — view 0 attends to view 1 and last view
    // -------------------------------------------------------------------------
    #[test]
    fn test_ring_view_mask_view0_attends_neighbors() {
        let n = 4;
        let tpv = 3;
        let m = ring_view_mask(n, tpv).unwrap();
        // View 0 → view 1 (forward neighbor)
        assert!(m.get(0, tpv));
        // View 0 → view 3 (last view, backward neighbor in ring)
        assert!(m.get(0, 3 * tpv));
        // View 0 does NOT directly attend to view 2
        assert!(!m.get(0, 2 * tpv));
    }

    // -------------------------------------------------------------------------
    // Test 17: ring_view_mask num_views=1 → same as self_view
    // -------------------------------------------------------------------------
    #[test]
    fn test_ring_view_mask_single_view_is_self_view() {
        let ring = ring_view_mask(1, 5).unwrap();
        let self_v = self_view_mask(1, 5).unwrap();
        assert_eq!(ring, self_v);
    }

    // -------------------------------------------------------------------------
    // Test 18: angular_proximity_mask — same-position views → full cross-view
    // -------------------------------------------------------------------------
    #[test]
    fn test_angular_proximity_same_position_full_cross() {
        // Two views at the same position → dist = 0 ≤ any positive max_angle
        let positions = vec![(0.0_f32, 0.0_f32), (0.0_f32, 0.0_f32)];
        let m = angular_proximity_mask(2, 4, &positions, 0.1).unwrap();
        // Should be equivalent to full mask
        let full = full_mask(8);
        assert_eq!(m, full);
    }

    // -------------------------------------------------------------------------
    // Test 19: angular_proximity_mask — opposite-hemisphere → no cross-attention
    // -------------------------------------------------------------------------
    #[test]
    fn test_angular_proximity_opposite_hemisphere_no_cross() {
        // North pole and south pole: angular distance = π
        let positions = vec![
            (0.0_f32, std::f32::consts::FRAC_PI_2), // north pole (el = π/2)
            (0.0_f32, -std::f32::consts::FRAC_PI_2), // south pole (el = -π/2)
        ];
        // max angle = π/4 → too small to bridge the poles
        let m = angular_proximity_mask(2, 4, &positions, std::f32::consts::FRAC_PI_4).unwrap();
        // Cross-view blocks should be blocked
        for q in 0..4 {
            for k in 4..8 {
                assert!(
                    !m.get(q, k),
                    "q={q} → k={k} should be masked (opposite poles)"
                );
            }
        }
        // Self-attention should still be allowed
        for q in 0..4 {
            assert!(m.get(q, q), "q={q} → self should be allowed");
        }
        for q in 4..8 {
            assert!(m.get(q, q), "q={q} → self should be allowed");
        }
    }

    // -------------------------------------------------------------------------
    // Test 20: build_mask Full → full_mask
    // -------------------------------------------------------------------------
    #[test]
    fn test_build_mask_full_pattern() {
        let m = build_mask(&MaskPattern::Full, 2, 4, None).unwrap();
        let expected = full_mask(8);
        assert_eq!(m, expected);
    }

    // -------------------------------------------------------------------------
    // Test 21: build_mask SelfView → self_view_mask
    // -------------------------------------------------------------------------
    #[test]
    fn test_build_mask_self_view_pattern() {
        let m = build_mask(&MaskPattern::SelfView, 3, 4, None).unwrap();
        let expected = self_view_mask(3, 4).unwrap();
        assert_eq!(m, expected);
    }

    // -------------------------------------------------------------------------
    // Test 22: build_layer_mask cross_view=false → self-view only
    // -------------------------------------------------------------------------
    #[test]
    fn test_build_layer_mask_no_cross_view() {
        let config = LayerMaskConfig {
            cross_view: false,
            self_pattern: MaskPattern::Full,
            cross_pattern: None,
        };
        let m = build_layer_mask(&config, 3, 4, None).unwrap();
        let expected = self_view_mask(3, 4).unwrap();
        assert_eq!(m, expected, "no cross-view should produce self-view mask");
    }

    // -------------------------------------------------------------------------
    // Test 23: apply_padding — padding tokens excluded from both Q and K
    // -------------------------------------------------------------------------
    #[test]
    fn test_apply_padding_excludes_padding_tokens() {
        let base = full_mask(4);
        // Token 1 and token 3 are padding
        let padding = vec![true, false, true, false];
        let m = base.apply_padding(&padding).unwrap();

        // Real tokens: q=0, q=2; k=0, k=2
        assert!(m.get(0, 0));
        assert!(m.get(0, 2));
        assert!(m.get(2, 0));
        assert!(m.get(2, 2));

        // Padding query (q=1) attends to nothing
        for k in 0..4 {
            assert!(!m.get(1, k), "padding query q=1 should not attend k={k}");
        }
        // Padding key (k=3) should not be attended to
        for q in 0..4 {
            assert!(
                !m.get(q, 3),
                "padding key k=3 should not be attended to by q={q}"
            );
        }
    }

    // -------------------------------------------------------------------------
    // Test 24: neighbor_view_mask — correct connectivity
    // -------------------------------------------------------------------------
    #[test]
    fn test_neighbor_view_mask_connectivity() {
        // 4 views, each with explicit neighbors (not a ring)
        let neighbors = vec![
            vec![1_usize],    // view 0 neighbors: [1]
            vec![0_usize, 2], // view 1 neighbors: [0, 2]
            vec![1_usize, 3], // view 2 neighbors: [1, 3]
            vec![2_usize],    // view 3 neighbors: [2]
        ];
        let tpv = 2;
        let m = neighbor_view_mask(4, tpv, &neighbors).unwrap();

        // view 0 (0,1) ↔ view 1 (2,3): allowed
        assert!(m.get(0, 2));
        assert!(m.get(2, 0));

        // view 0 (0,1) → view 2 (4,5): NOT directly connected
        assert!(!m.get(0, 4));

        // view 1 (2,3) ↔ view 2 (4,5): allowed
        assert!(m.get(2, 4));
        assert!(m.get(4, 2));

        // view 0 (0,1) → view 3 (6,7): NOT connected
        assert!(!m.get(0, 6));

        // Self-attention always enabled
        assert!(m.get(0, 1));
        assert!(m.get(1, 0));
        assert!(m.get(4, 5));
    }

    // -------------------------------------------------------------------------
    // Additional: and/or dimension mismatch errors
    // -------------------------------------------------------------------------
    #[test]
    fn test_and_dimension_mismatch_err() {
        let a = full_mask(3);
        let b = full_mask(4);
        assert!(matches!(
            a.and(&b),
            Err(AttentionMaskError::InvalidDimensions(_))
        ));
    }

    #[test]
    fn test_or_dimension_mismatch_err() {
        let a = full_mask(3);
        let b = full_mask(4);
        assert!(matches!(
            a.or(&b),
            Err(AttentionMaskError::InvalidDimensions(_))
        ));
    }

    // -------------------------------------------------------------------------
    // Additional: format_ascii length check
    // -------------------------------------------------------------------------
    #[test]
    fn test_format_ascii_length() {
        let m = full_mask(4);
        let ascii = m.format_ascii();
        // 4 rows × (4 chars + 1 newline) = 20
        assert_eq!(ascii.len(), 20);
        assert!(ascii.chars().all(|c| c == 'O' || c == '\n'));
    }

    // -------------------------------------------------------------------------
    // Additional: to_float_mask values
    // -------------------------------------------------------------------------
    #[test]
    fn test_to_float_mask_values() {
        let mut m = AttentionMask::new(2, false);
        m.set(0, 0, true);
        let fmask = m.to_float_mask();
        assert!((fmask[0] - 1.0).abs() < 1e-7);
        assert!((fmask[1] - 0.0).abs() < 1e-7);
    }

    // -------------------------------------------------------------------------
    // Additional: build_layer_mask with cross-view OR'd together
    // -------------------------------------------------------------------------
    #[test]
    fn test_build_layer_mask_multiview_self_pattern_errors() {
        // Ring (and SelfView/CrossView/ReferenceView) describe inter-view
        // relationships and are meaningless as a self_pattern (intra-view);
        // build_layer_mask must reject them rather than silently substituting
        // a plain full_mask indistinguishable from MaskPattern::Full.
        for pattern in [
            MaskPattern::SelfView,
            MaskPattern::CrossView,
            MaskPattern::ReferenceView,
            MaskPattern::Ring,
        ] {
            let config = LayerMaskConfig {
                cross_view: false,
                self_pattern: pattern.clone(),
                cross_pattern: None,
            };
            let result = build_layer_mask(&config, 3, 4, None);
            assert!(
                matches!(result, Err(AttentionMaskError::InvalidPattern(_))),
                "{pattern:?} as self_pattern should be InvalidPattern, got {result:?}"
            );
        }
    }

    #[test]
    fn test_build_layer_mask_cross_view_or() {
        let config = LayerMaskConfig {
            cross_view: true,
            self_pattern: MaskPattern::Full,
            cross_pattern: Some(MaskPattern::ReferenceView),
        };
        let m = build_layer_mask(&config, 3, 4, None).unwrap();
        // Should include all of reference_view_mask plus self-view blocks
        let ref_mask = reference_view_mask(3, 4).unwrap();
        let self_v = self_view_mask(3, 4).unwrap();
        let expected = self_v.or(&ref_mask).unwrap();
        assert_eq!(m, expected);
    }
}
