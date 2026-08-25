//! Scene optimization pipeline for 3DGS (3D Gaussian Splatting) scenes.
//!
//! Provides a configurable multi-step pipeline that deduplicates, prunes,
//! sorts, clamps, and clips Gaussians to reduce scene size before deployment.
//!
//! All functions use the `so_` prefix to avoid name conflicts with other CLI modules.
//!
//! # Opacity space
//!
//! Opacity arrays hold raw logits until [`OptimizationStep::NormalizeOpacity`]
//! activates them into `(0, 1)` probabilities — see [`OpacitySpace`], taken
//! explicitly by every opacity-consuming function so sigmoid applies at most once.
//!
//! # Scale space
//!
//! Scale arrays are **log-scale** throughout (matching
//! `oxigaf_render::gaussian::GaussianAttributes::scale`, which callers
//! `.exp()` before use). Nothing here exponentiates. A *linear* bound (e.g.
//! "clamp to [1e-5, 0.1] world units") needs `ln` first: `ln(1e-5)..ln(0.1)`
//! ≈ `-11.51..-2.303`.

use thiserror::Error;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors produced by the scene optimization pipeline.
#[derive(Debug, Error)]
pub enum OptimizerError {
    #[error("Empty scene: no Gaussians to optimize")]
    EmptyScene,

    #[error("Invalid step config: {reason}")]
    InvalidStep { reason: String },

    #[error("Array length mismatch: expected {expected} elements for {field}, got {got}")]
    LengthMismatch {
        expected: usize,
        got: usize,
        field: String,
    },

    #[error("Optimization pipeline failed at step '{step}': {reason}")]
    StepFailed { step: String, reason: String },

    #[error("Invalid threshold: {value} for parameter '{param}'")]
    InvalidThreshold { value: f32, param: String },
}

// ---------------------------------------------------------------------------
// Scene snapshot
// ---------------------------------------------------------------------------

/// Statistical snapshot of a 3DGS scene (positions, scales, opacities).
#[derive(Debug, Clone)]
pub struct SceneSnapshot {
    /// Number of Gaussians.
    pub n_gaussians: usize,
    /// Mean over sigmoid-applied opacity values.
    pub mean_opacity: f32,
    /// Max sigmoid-applied opacity value.
    pub max_opacity: f32,
    /// Mean over all scale components, in log-scale (see module docs) —
    /// `.exp()` before treating as a world-unit length.
    pub mean_scale: f32,
    /// Max scale component, log-scale like [`SceneSnapshot::mean_scale`].
    pub max_scale: f32,
    /// Axis-aligned bounding box minimum corner.
    pub bounds_min: [f32; 3],
    /// Axis-aligned bounding box maximum corner.
    pub bounds_max: [f32; 3],
    /// Estimated memory in bytes: (3+4+3+1+sh_channels)*4*n_gaussians
    pub memory_bytes: usize,
}

/// Compute a snapshot from flat arrays.
///
/// `positions` must have `n*3` elements, `scales` `n*3`, `opacities` `n`.
/// See [`OpacitySpace`] for `opacity_space`.
pub fn so_compute_snapshot(
    positions: &[f32],
    scales: &[f32],
    opacities: &[f32],
    n: usize,
    sh_channels: usize,
    opacity_space: OpacitySpace,
) -> SceneSnapshot {
    if n == 0 {
        return SceneSnapshot {
            n_gaussians: 0,
            mean_opacity: 0.0,
            max_opacity: 0.0,
            mean_scale: 0.0,
            max_scale: 0.0,
            bounds_min: [0.0; 3],
            bounds_max: [0.0; 3],
            memory_bytes: 0,
        };
    }

    // Opacities — activate (apply sigmoid) only if not already activated.
    // `n` is caller-supplied and need not match `opacities.len()`, so clamp
    // exactly as the scale and position loops below do; indexing `0..n`
    // straight into `opacities` panicked on a short array.
    let opacity_count = n.min(opacities.len());
    let mut sum_opacity = 0.0f32;
    let mut max_opacity = f32::NEG_INFINITY;
    for &v in &opacities[..opacity_count] {
        let s = so_activate_opacity(v, opacity_space);
        sum_opacity += s;
        if s > max_opacity {
            max_opacity = s;
        }
    }
    let (mean_opacity, max_opacity) = if opacity_count > 0 {
        (sum_opacity / opacity_count as f32, max_opacity)
    } else {
        // No opacity data at all: report zeros rather than leaking the
        // `-inf` seed of the running maximum into the snapshot.
        (0.0, 0.0)
    };

    // Scales
    let mut sum_scale = 0.0f32;
    let mut max_scale = f32::NEG_INFINITY;
    let scale_count = (n * 3).min(scales.len());
    for &v in &scales[..scale_count] {
        sum_scale += v;
        if v > max_scale {
            max_scale = v;
        }
    }
    let (mean_scale, max_scale) = if scale_count > 0 {
        (sum_scale / scale_count as f32, max_scale)
    } else {
        (0.0, 0.0)
    };

    // Bounds from positions
    let pos_count = (n * 3).min(positions.len());
    let mut bounds_min = [f32::INFINITY; 3];
    let mut bounds_max = [f32::NEG_INFINITY; 3];
    let full_n = pos_count / 3;
    for i in 0..full_n {
        for ax in 0..3 {
            let v = positions[i * 3 + ax];
            if v < bounds_min[ax] {
                bounds_min[ax] = v;
            }
            if v > bounds_max[ax] {
                bounds_max[ax] = v;
            }
        }
    }
    if full_n == 0 {
        bounds_min = [0.0; 3];
        bounds_max = [0.0; 3];
    }

    let memory_bytes = (3 + 4 + 3 + 1 + sh_channels) * 4 * n;

    SceneSnapshot {
        n_gaussians: n,
        mean_opacity,
        max_opacity,
        mean_scale,
        max_scale,
        bounds_min,
        bounds_max,
        memory_bytes,
    }
}

// ---------------------------------------------------------------------------
// Optimization steps (enum)
// ---------------------------------------------------------------------------

/// An individual step in the optimization pipeline.
#[derive(Debug, Clone)]
pub enum OptimizationStep {
    /// Remove Gaussians with sigmoid(opacity_logit) < threshold.
    PruneByOpacity { threshold: f32 },

    /// Remove spatial near-duplicates using spatial hashing (or O(N²) for small N).
    DeduplicateNear { position_radius: f32 },

    /// Clamp per-component scale values to `[min_scale, max_scale]`. Both
    /// bounds are **log-scale** (see module docs) — linear bounds
    /// `[1e-5, 0.1]` need `(1e-5f32).ln()`/`(0.1f32).ln()`.
    ClampScales { min_scale: f32, max_scale: f32 },

    /// Sort Gaussians by 30-bit Morton code for spatial cache locality.
    SortMorton,

    /// Keep only the top-n Gaussians by sigmoid(opacity).
    TopNByOpacity { n: usize },

    /// Convert raw opacity logits to probabilities via sigmoid (in-place).
    NormalizeOpacity,

    /// Remove Gaussians outside a bounding sphere.
    ClipToSphere { center: [f32; 3], radius: f32 },

    /// Remove Gaussians outside an axis-aligned bounding box.
    ClipToAabb { min: [f32; 3], max: [f32; 3] },
}

impl OptimizationStep {
    fn name(&self) -> &'static str {
        match self {
            OptimizationStep::PruneByOpacity { .. } => "PruneByOpacity",
            OptimizationStep::DeduplicateNear { .. } => "DeduplicateNear",
            OptimizationStep::ClampScales { .. } => "ClampScales",
            OptimizationStep::SortMorton => "SortMorton",
            OptimizationStep::TopNByOpacity { .. } => "TopNByOpacity",
            OptimizationStep::NormalizeOpacity => "NormalizeOpacity",
            OptimizationStep::ClipToSphere { .. } => "ClipToSphere",
            OptimizationStep::ClipToAabb { .. } => "ClipToAabb",
        }
    }

    fn duration_hint(&self) -> &'static str {
        match self {
            OptimizationStep::PruneByOpacity { .. } => "fast",
            OptimizationStep::DeduplicateNear { .. } => "moderate",
            OptimizationStep::ClampScales { .. } => "fast",
            OptimizationStep::SortMorton => "moderate",
            OptimizationStep::TopNByOpacity { .. } => "moderate",
            OptimizationStep::NormalizeOpacity => "fast",
            OptimizationStep::ClipToSphere { .. } => "fast",
            OptimizationStep::ClipToAabb { .. } => "fast",
        }
    }
}

// ---------------------------------------------------------------------------
// Per-step result
// ---------------------------------------------------------------------------

/// Result from executing a single optimization step.
#[derive(Debug, Clone)]
pub struct OptimizationStepResult {
    /// Human-readable step name.
    pub step_name: String,
    /// Gaussian count before this step.
    pub n_before: usize,
    /// Gaussian count after this step.
    pub n_after: usize,
    /// Number of Gaussians removed.
    pub n_removed: usize,
    /// Rough timing class: "fast", "moderate", or "slow".
    pub duration_hint: String,
    /// Informational notes about this run.
    pub notes: String,
}

// ---------------------------------------------------------------------------
// Math helpers
// ---------------------------------------------------------------------------

/// Numerically stable logistic sigmoid.
///
/// The textbook form `1 / (1 + exp(-x))` overflows `f32`'s exponent for
/// `x < -88.7` (`exp` saturates to `+inf`), collapsing the result to exactly
/// `0.0` — so a Gaussian with an opacity logit of, say, `-100` looked
/// *completely* transparent instead of merely almost so, and
/// [`so_prune_by_opacity`] deleted it even at `threshold = 0`. Branching on
/// the sign keeps the exponent argument negative in both halves, so the
/// result stays positive down to `x ≈ -103`, where `exp` genuinely
/// underflows past the smallest subnormal `f32`.
#[inline]
fn so_sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Which space an opacity array is currently in (see module docs).
/// Sigmoiding twice silently recompresses an already-activated value, so
/// every opacity-consuming function here takes this explicitly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpacitySpace {
    /// Raw, unbounded opacity logits — needs activation before use.
    Logit,
    /// Already-activated probabilities in `(0, 1)` — used as-is.
    Probability,
}

/// Interpret `v` as a probability, applying [`so_sigmoid`] only if needed.
#[inline]
fn so_activate_opacity(v: f32, space: OpacitySpace) -> f32 {
    match space {
        OpacitySpace::Logit => so_sigmoid(v),
        OpacitySpace::Probability => v,
    }
}

// ---------------------------------------------------------------------------
// Mask application helpers
// ---------------------------------------------------------------------------

/// Apply a keep mask with arbitrary stride; retains rows where `mask[i]` is true.
pub fn so_apply_keep_mask_nd(data: &[f32], mask: &[bool], n: usize, stride: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n * stride);
    let elements = (n * stride).min(data.len());
    for i in 0..n {
        if i >= mask.len() || !mask[i] {
            continue;
        }
        let start = i * stride;
        let end = (start + stride).min(elements);
        if start < elements {
            out.extend_from_slice(&data[start..end]);
        }
    }
    out
}

/// Apply a keep mask, stride=1.
pub fn so_apply_keep_mask_1d(data: &[f32], mask: &[bool], n: usize) -> Vec<f32> {
    so_apply_keep_mask_nd(data, mask, n, 1)
}

/// Apply a keep mask, stride=3.
pub fn so_apply_keep_mask_3d(data: &[f32], mask: &[bool], n: usize) -> Vec<f32> {
    so_apply_keep_mask_nd(data, mask, n, 3)
}

/// Apply a keep mask, stride=4.
pub fn so_apply_keep_mask_4d(data: &[f32], mask: &[bool], n: usize) -> Vec<f32> {
    so_apply_keep_mask_nd(data, mask, n, 4)
}

/// Reorder a flat array by the given index permutation.
pub fn so_reorder_by_indices(data: &[f32], indices: &[usize], n: usize, stride: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n * stride);
    for &idx in indices.iter().take(n) {
        let start = idx * stride;
        let end = start + stride;
        if end <= data.len() {
            out.extend_from_slice(&data[start..end]);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Morton code helpers
// ---------------------------------------------------------------------------

/// Spread the low 10 bits of `x` into bits 0, 3, 6, 9, ..., 27.
pub fn so_morton_interleave(x: u32) -> u32 {
    let mut v = x & 0x000003FF; // keep only 10 bits
    v = (v | (v << 16)) & 0x030000FF;
    v = (v | (v << 8)) & 0x0300F00F;
    v = (v | (v << 4)) & 0x030C30C3;
    v = (v | (v << 2)) & 0x09249249;
    v
}

/// Combine three 10-bit quantised coordinates into a 30-bit Morton code.
///
/// Bit layout: z bits go to positions 2,5,8,...; y to 1,4,7,...; x to 0,3,6,...
pub fn so_morton_code(ix: u32, iy: u32, iz: u32) -> u32 {
    so_morton_interleave(ix) | (so_morton_interleave(iy) << 1) | (so_morton_interleave(iz) << 2)
}

/// Map a 3-D position from [bounds_min, bounds_max] to [0, 2^bits - 1].
pub fn so_quantize_position(
    pos: [f32; 3],
    bounds_min: [f32; 3],
    bounds_max: [f32; 3],
    bits: u32,
) -> [u32; 3] {
    let max_val = ((1u32 << bits) - 1) as f32;
    let mut out = [0u32; 3];
    for ax in 0..3 {
        let range = bounds_max[ax] - bounds_min[ax];
        let t = if range.abs() < 1e-12 {
            0.5
        } else {
            ((pos[ax] - bounds_min[ax]) / range).clamp(0.0, 1.0)
        };
        out[ax] = (t * max_val).round() as u32;
    }
    out
}

// ---------------------------------------------------------------------------
// Core optimization functions
// ---------------------------------------------------------------------------

/// Compute keep mask: activated_opacity > threshold (strict; see
/// [`OpacitySpace`]).
///
/// In [`OpacitySpace::Logit`], `sigmoid` is strictly positive for every
/// finite logit, so `threshold = 0` keeps every Gaussian (down to the
/// `x ≈ -103` point where `exp` underflows to zero in `f32` — see
/// `so_sigmoid`). In [`OpacitySpace::Probability`] a stored value of
/// exactly `0.0` is fully transparent and *is* pruned at `threshold = 0`,
/// which is the point of the strict comparison.
///
/// Entries past `opacities.len()` keep their default `true`.
pub fn so_prune_by_opacity(
    opacities: &[f32],
    n: usize,
    threshold: f32,
    opacity_space: OpacitySpace,
) -> Vec<bool> {
    let mut mask = vec![true; n];
    let count = n.min(opacities.len());
    for (keep, &opacity) in mask[..count].iter_mut().zip(&opacities[..count]) {
        *keep = so_activate_opacity(opacity, opacity_space) > threshold;
    }
    mask
}

/// Compute keep mask: first occurrence within `radius` of any later point is kept.
///
/// Uses O(N²) for N < 1000 and a spatial hash for N >= 1000.
/// radius=0 keeps all (strict `<` comparison).
pub fn so_deduplicate_near(positions: &[f32], n: usize, radius: f32) -> Vec<bool> {
    let mut keep = vec![true; n];

    if radius <= 0.0 {
        return keep;
    }

    if n < 1000 {
        // O(N²) brute force.
        //
        // `split_at_mut` hands the inner loop an exclusive borrow of the
        // *later* entries only, so it can clear duplicates through an
        // iterator while the outer loop still reads `keep[i]`.
        for i in 0..n {
            let (earlier, later) = keep.split_at_mut(i + 1);
            let keep_i = earlier.last().copied().unwrap_or(false);
            if !keep_i {
                continue;
            }
            let xi = positions.get(i * 3).copied().unwrap_or(0.0);
            let yi = positions.get(i * 3 + 1).copied().unwrap_or(0.0);
            let zi = positions.get(i * 3 + 2).copied().unwrap_or(0.0);
            for (offset, keep_j) in later.iter_mut().enumerate() {
                if !*keep_j {
                    continue;
                }
                let j = i + 1 + offset;
                let xj = positions.get(j * 3).copied().unwrap_or(0.0);
                let yj = positions.get(j * 3 + 1).copied().unwrap_or(0.0);
                let zj = positions.get(j * 3 + 2).copied().unwrap_or(0.0);
                let dx = xi - xj;
                let dy = yi - yj;
                let dz = zi - zj;
                let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                if dist < radius {
                    *keep_j = false;
                }
            }
        }
    } else {
        // Spatial hash approach
        // Cell size = radius so any overlapping pair will be in the same or adjacent cell
        let inv_r = 1.0 / radius;

        // Quantize each point to a cell
        let cell_of = |i: usize| -> (i64, i64, i64) {
            let x = positions.get(i * 3).copied().unwrap_or(0.0);
            let y = positions.get(i * 3 + 1).copied().unwrap_or(0.0);
            let z = positions.get(i * 3 + 2).copied().unwrap_or(0.0);
            (
                (x * inv_r).floor() as i64,
                (y * inv_r).floor() as i64,
                (z * inv_r).floor() as i64,
            )
        };

        // Build a simple hash map: cell → list of indices
        let mut cell_map: std::collections::HashMap<(i64, i64, i64), Vec<usize>> =
            std::collections::HashMap::new();
        for i in 0..n {
            cell_map.entry(cell_of(i)).or_default().push(i);
        }

        for i in 0..n {
            if !keep[i] {
                continue;
            }
            let xi = positions.get(i * 3).copied().unwrap_or(0.0);
            let yi = positions.get(i * 3 + 1).copied().unwrap_or(0.0);
            let zi = positions.get(i * 3 + 2).copied().unwrap_or(0.0);
            let (cx, cy, cz) = cell_of(i);

            // Check 3x3x3 neighbourhood
            for nx in -1i64..=1 {
                for ny in -1i64..=1 {
                    for nz in -1i64..=1 {
                        let nc = (cx + nx, cy + ny, cz + nz);
                        if let Some(neighbours) = cell_map.get(&nc) {
                            for &j in neighbours {
                                if j <= i || !keep[j] {
                                    continue;
                                }
                                let xj = positions.get(j * 3).copied().unwrap_or(0.0);
                                let yj = positions.get(j * 3 + 1).copied().unwrap_or(0.0);
                                let zj = positions.get(j * 3 + 2).copied().unwrap_or(0.0);
                                let dx = xi - xj;
                                let dy = yi - yj;
                                let dz = zi - zj;
                                let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                                if dist < radius {
                                    keep[j] = false;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    keep
}

/// Clamp every scale component to `[min_scale, max_scale]` in-place.
/// All three are log-scale (see the [module-level](self) "Scale space" note).
pub fn so_clamp_scales(scales: &mut [f32], n: usize, min_scale: f32, max_scale: f32) {
    let count = (n * 3).min(scales.len());
    for v in scales[..count].iter_mut() {
        *v = v.clamp(min_scale, max_scale);
    }
}

/// Return sorted indices by Morton code (ascending).
pub fn so_sort_morton(positions: &[f32], n: usize) -> Vec<usize> {
    if n == 0 {
        return Vec::new();
    }

    // Compute bounds
    let full_n = (n * 3).min(positions.len()) / 3;
    let mut bounds_min = [f32::INFINITY; 3];
    let mut bounds_max = [f32::NEG_INFINITY; 3];
    for i in 0..full_n {
        for ax in 0..3 {
            let v = positions[i * 3 + ax];
            if v < bounds_min[ax] {
                bounds_min[ax] = v;
            }
            if v > bounds_max[ax] {
                bounds_max[ax] = v;
            }
        }
    }

    // Build (morton_code, index) pairs
    let mut pairs: Vec<(u32, usize)> = (0..n)
        .map(|i| {
            if i < full_n {
                let pos = [positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]];
                let [qx, qy, qz] = so_quantize_position(pos, bounds_min, bounds_max, 10);
                (so_morton_code(qx, qy, qz), i)
            } else {
                (0, i)
            }
        })
        .collect();

    pairs.sort_unstable_by_key(|&(code, _)| code);
    pairs.into_iter().map(|(_, idx)| idx).collect()
}

/// Keep mask: top-n by activated opacity. See [`OpacitySpace`] for
/// `opacity_space`. If n_keep >= n_total, all are kept.
pub fn so_top_n_by_opacity(
    opacities: &[f32],
    n_total: usize,
    n_keep: usize,
    opacity_space: OpacitySpace,
) -> Vec<bool> {
    if n_keep >= n_total {
        return vec![true; n_total];
    }

    // Build (activated_opacity, index) pairs
    let mut indexed: Vec<(f32, usize)> = opacities[..n_total.min(opacities.len())]
        .iter()
        .enumerate()
        .map(|(i, &v)| (so_activate_opacity(v, opacity_space), i))
        .collect();

    // Sort descending by opacity
    indexed.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut mask = vec![false; n_total];
    for (_, idx) in indexed.into_iter().take(n_keep) {
        mask[idx] = true;
    }
    mask
}

/// Apply sigmoid to all opacity logits, returning probabilities in (0, 1).
pub fn so_normalize_opacity(opacities: &[f32], n: usize) -> Vec<f32> {
    opacities[..n.min(opacities.len())]
        .iter()
        .map(|&v| so_sigmoid(v))
        .collect()
}

/// Keep mask: points inside the sphere (distance from center < radius).
pub fn so_clip_to_sphere(positions: &[f32], n: usize, center: [f32; 3], radius: f32) -> Vec<bool> {
    let r2 = radius * radius;
    let mut mask = vec![true; n];
    for (i, inside) in mask.iter_mut().enumerate() {
        let x = positions.get(i * 3).copied().unwrap_or(0.0) - center[0];
        let y = positions.get(i * 3 + 1).copied().unwrap_or(0.0) - center[1];
        let z = positions.get(i * 3 + 2).copied().unwrap_or(0.0) - center[2];
        *inside = x * x + y * y + z * z <= r2;
    }
    mask
}

/// Keep mask: points inside the AABB [min, max] (inclusive on both sides).
pub fn so_clip_to_aabb(positions: &[f32], n: usize, min: [f32; 3], max: [f32; 3]) -> Vec<bool> {
    let mut mask = vec![true; n];
    for (i, keep) in mask.iter_mut().enumerate() {
        let mut inside = true;
        for ax in 0..3 {
            let v = positions.get(i * 3 + ax).copied().unwrap_or(0.0);
            if v < min[ax] || v > max[ax] {
                inside = false;
                break;
            }
        }
        *keep = inside;
    }
    mask
}

// ---------------------------------------------------------------------------
// Configuration and report structs
// ---------------------------------------------------------------------------

/// Configuration for the optimization pipeline.
#[derive(Debug, Clone)]
pub struct SceneOptimizerConfig {
    /// Ordered list of steps to execute.
    pub steps: Vec<OptimizationStep>,
    /// Number of spherical-harmonic channels per Gaussian (used for memory estimation).
    pub sh_channels: usize,
    /// RNG seed (reserved for future use; included for API stability).
    pub seed: u64,
}

/// A per-step summary and before/after scene snapshot.
#[derive(Debug, Clone)]
pub struct OptimizationReport {
    /// Per-step results in execution order.
    pub step_results: Vec<OptimizationStepResult>,
    /// Scene snapshot before the first step.
    pub snapshot_before: SceneSnapshot,
    /// Scene snapshot after the last step.
    pub snapshot_after: SceneSnapshot,
    /// Total Gaussians removed across all steps.
    pub total_removed: usize,
    /// Percent reduction in Gaussian count.
    pub total_reduction_percent: f32,
    /// Memory saved (bytes) between before and after snapshots.
    pub memory_saved_bytes: usize,
}

/// The full pipeline executor.
#[derive(Debug, Clone)]
pub struct OptimizationPipeline {
    /// Configuration used by this pipeline.
    pub config: SceneOptimizerConfig,
}

/// A 3DGS scene returned from the pipeline.
#[derive(Debug, Clone)]
pub struct OptimizedScene {
    /// Flat positions array (n_gaussians * 3).
    pub positions: Vec<f32>,
    /// Flat rotations array (n_gaussians * 4).
    pub rotations: Vec<f32>,
    /// Flat scales array (n_gaussians * 3), log-scale (see module docs).
    pub scales: Vec<f32>,
    /// Flat opacity array (n_gaussians). Raw logits, unless `config.steps`
    /// included [`OptimizationStep::NormalizeOpacity`] (then probabilities).
    pub opacities: Vec<f32>,
    /// Flat SH coefficients array (n_gaussians * sh_channels).
    pub sh_coefficients: Vec<f32>,
    /// Number of Gaussians in the optimised scene.
    pub n_gaussians: usize,
    /// SH channels per Gaussian.
    pub sh_channels: usize,
}

// ---------------------------------------------------------------------------
// Pipeline implementation
// ---------------------------------------------------------------------------

impl OptimizationPipeline {
    /// Create a new pipeline with the given configuration.
    pub fn new(config: SceneOptimizerConfig) -> Self {
        Self { config }
    }

    /// Run the pipeline on the given flat Gaussian arrays.
    ///
    /// Returns the optimised scene and a detailed report.
    pub fn run(
        &self,
        positions: &[f32],
        rotations: &[f32],
        scales: &[f32],
        opacities: &[f32],
        sh_coefficients: &[f32],
        n_gaussians: usize,
    ) -> Result<(OptimizedScene, OptimizationReport), OptimizerError> {
        if n_gaussians == 0 {
            return Err(OptimizerError::EmptyScene);
        }

        // Validate input lengths
        for (expected, got, field) in [
            (n_gaussians * 3, positions.len(), "positions"),
            (n_gaussians * 4, rotations.len(), "rotations"),
            (n_gaussians * 3, scales.len(), "scales"),
            (n_gaussians, opacities.len(), "opacities"),
        ] {
            if got != expected {
                return Err(OptimizerError::LengthMismatch {
                    expected,
                    got,
                    field: field.to_string(),
                });
            }
        }

        let sh_channels = self.config.sh_channels;
        let expected_sh = n_gaussians * sh_channels;
        if !sh_coefficients.is_empty() && sh_coefficients.len() != expected_sh {
            return Err(OptimizerError::LengthMismatch {
                expected: expected_sh,
                got: sh_coefficients.len(),
                field: "sh_coefficients".to_string(),
            });
        }

        // Snapshot before — pipeline input is always raw opacity logits.
        let snapshot_before = so_compute_snapshot(
            positions,
            scales,
            opacities,
            n_gaussians,
            sh_channels,
            OpacitySpace::Logit,
        );

        // Working copies
        let mut cur = WorkingScene {
            positions: positions.to_vec(),
            rotations: rotations.to_vec(),
            scales: scales.to_vec(),
            opacities: opacities.to_vec(),
            sh: sh_coefficients.to_vec(),
            n: n_gaussians,
        };
        // Tracks whether `cur.opacities` currently holds raw logits or
        // already-activated probabilities, so every opacity-consuming step
        // below applies `so_sigmoid` at most once overall (see
        // `OpacitySpace`).
        let mut opacity_space = OpacitySpace::Logit;

        let mut step_results = Vec::with_capacity(self.config.steps.len());

        for step in &self.config.steps {
            let n_before = cur.n;
            let step_name = step.name().to_string();
            let duration_hint = step.duration_hint().to_string();

            let notes = match step {
                OptimizationStep::PruneByOpacity { threshold } => {
                    let mask =
                        so_prune_by_opacity(&cur.opacities, cur.n, *threshold, opacity_space);
                    cur.apply_mask(&mask, sh_channels);
                    format!("threshold={threshold:.4}")
                }

                OptimizationStep::DeduplicateNear { position_radius } => {
                    let mask = so_deduplicate_near(&cur.positions, cur.n, *position_radius);
                    cur.apply_mask(&mask, sh_channels);
                    format!("radius={position_radius:.4}")
                }

                OptimizationStep::ClampScales {
                    min_scale,
                    max_scale,
                } => {
                    if min_scale > max_scale {
                        return Err(OptimizerError::InvalidThreshold {
                            value: *min_scale,
                            param: "min_scale (must be ≤ max_scale)".to_string(),
                        });
                    }
                    so_clamp_scales(&mut cur.scales, cur.n, *min_scale, *max_scale);
                    format!("min={min_scale:.5} max={max_scale:.5}")
                }

                OptimizationStep::SortMorton => {
                    let indices = so_sort_morton(&cur.positions, cur.n);
                    cur.positions = so_reorder_by_indices(&cur.positions, &indices, cur.n, 3);
                    cur.rotations = so_reorder_by_indices(&cur.rotations, &indices, cur.n, 4);
                    cur.scales = so_reorder_by_indices(&cur.scales, &indices, cur.n, 3);
                    cur.opacities = so_reorder_by_indices(&cur.opacities, &indices, cur.n, 1);
                    if !cur.sh.is_empty() {
                        cur.sh = so_reorder_by_indices(&cur.sh, &indices, cur.n, sh_channels);
                    }
                    "Morton code sort".to_string()
                }

                OptimizationStep::TopNByOpacity { n } => {
                    let mask = so_top_n_by_opacity(&cur.opacities, cur.n, *n, opacity_space);
                    cur.apply_mask(&mask, sh_channels);
                    format!("n_keep={n}")
                }

                OptimizationStep::NormalizeOpacity => {
                    if opacity_space == OpacitySpace::Probability {
                        // Already activated by an earlier NormalizeOpacity
                        // step (or by construction) — applying sigmoid again
                        // would recompress an already-(0,1) value instead of
                        // converting a logit, so this is a no-op.
                        "already activated (no-op)".to_string()
                    } else {
                        cur.opacities = so_normalize_opacity(&cur.opacities, cur.n);
                        opacity_space = OpacitySpace::Probability;
                        "sigmoid applied".to_string()
                    }
                }

                OptimizationStep::ClipToSphere { center, radius } => {
                    let mask = so_clip_to_sphere(&cur.positions, cur.n, *center, *radius);
                    cur.apply_mask(&mask, sh_channels);
                    format!(
                        "center=[{:.2},{:.2},{:.2}] r={:.4}",
                        center[0], center[1], center[2], radius
                    )
                }

                OptimizationStep::ClipToAabb { min, max } => {
                    let mask = so_clip_to_aabb(&cur.positions, cur.n, *min, *max);
                    cur.apply_mask(&mask, sh_channels);
                    format!(
                        "min=[{:.2},{:.2},{:.2}] max=[{:.2},{:.2},{:.2}]",
                        min[0], min[1], min[2], max[0], max[1], max[2]
                    )
                }
            };

            let n_after = cur.n;
            step_results.push(OptimizationStepResult {
                step_name,
                n_before,
                n_after,
                n_removed: n_before.saturating_sub(n_after),
                duration_hint,
                notes,
            });
        }

        let snapshot_after = so_compute_snapshot(
            &cur.positions,
            &cur.scales,
            &cur.opacities,
            cur.n,
            sh_channels,
            opacity_space,
        );

        let total_removed = n_gaussians.saturating_sub(cur.n);
        let total_reduction_percent = if n_gaussians > 0 {
            100.0 * total_removed as f32 / n_gaussians as f32
        } else {
            0.0
        };
        let memory_saved_bytes = snapshot_before
            .memory_bytes
            .saturating_sub(snapshot_after.memory_bytes);

        let report = OptimizationReport {
            step_results,
            snapshot_before,
            snapshot_after,
            total_removed,
            total_reduction_percent,
            memory_saved_bytes,
        };

        let optimized = OptimizedScene {
            positions: cur.positions,
            rotations: cur.rotations,
            scales: cur.scales,
            opacities: cur.opacities,
            sh_coefficients: cur.sh,
            n_gaussians: cur.n,
            sh_channels,
        };

        Ok((optimized, report))
    }
}

/// The five flat Gaussian arrays a pipeline run carries from step to step,
/// plus the current Gaussian count.
///
/// The arrays are only ever resized together — dropping a Gaussian from one
/// without dropping it from the other four silently desynchronises the whole
/// scene — so [`OptimizationPipeline::run`] keeps them in one value with a
/// single [`apply_mask`](WorkingScene::apply_mask) entry point rather than as
/// six independent locals threaded through an eight-argument helper.
struct WorkingScene {
    positions: Vec<f32>,
    rotations: Vec<f32>,
    scales: Vec<f32>,
    opacities: Vec<f32>,
    sh: Vec<f32>,
    n: usize,
}

impl WorkingScene {
    /// Apply a boolean keep-mask to all five arrays at once and update `n`.
    ///
    /// `n` is written last: every array is filtered against the count the
    /// mask was built for.
    fn apply_mask(&mut self, mask: &[bool], sh_channels: usize) {
        self.positions = so_apply_keep_mask_3d(&self.positions, mask, self.n);
        self.rotations = so_apply_keep_mask_4d(&self.rotations, mask, self.n);
        self.scales = so_apply_keep_mask_3d(&self.scales, mask, self.n);
        self.opacities = so_apply_keep_mask_1d(&self.opacities, mask, self.n);
        self.sh = if sh_channels > 0 {
            so_apply_keep_mask_nd(&self.sh, mask, self.n, sh_channels)
        } else {
            Vec::new()
        };
        self.n = mask.iter().filter(|&&v| v).count();
    }
}

// ---------------------------------------------------------------------------
// Preset profiles
// ---------------------------------------------------------------------------

/// Pre-built optimization profiles for common use-cases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationProfile {
    /// Maximum quality: only deduplication (very conservative).
    Quality,
    /// Balanced: dedup + light pruning + scale clamping.
    Balanced,
    /// Performance: aggressive pruning + top-N + Morton sort.
    Performance,
    /// Streaming: Morton sort + optional spatial clip (head-sized bounding sphere).
    Streaming,
}

/// Return an appropriate `SceneOptimizerConfig` for the given profile.
pub fn so_profile_config(profile: OptimizationProfile, sh_channels: usize) -> SceneOptimizerConfig {
    let steps = match profile {
        OptimizationProfile::Quality => vec![OptimizationStep::DeduplicateNear {
            position_radius: 0.001,
        }],

        OptimizationProfile::Balanced => vec![
            OptimizationStep::DeduplicateNear {
                position_radius: 0.001,
            },
            OptimizationStep::PruneByOpacity { threshold: 0.01 },
            // Intent: keep linear world-unit sizes within [1e-5, 0.1].
            // `ClampScales` operates in log-scale (see the module-level
            // "Scale space" note), so the bounds are `ln` of those linear
            // values, not the linear values themselves.
            OptimizationStep::ClampScales {
                min_scale: (1e-5f32).ln(),
                max_scale: (0.1f32).ln(),
            },
        ],

        OptimizationProfile::Performance => vec![
            OptimizationStep::PruneByOpacity { threshold: 0.05 },
            OptimizationStep::TopNByOpacity { n: 500_000 },
            OptimizationStep::SortMorton,
        ],

        OptimizationProfile::Streaming => vec![
            OptimizationStep::SortMorton,
            OptimizationStep::ClipToSphere {
                center: [0.0, 0.0, 0.0],
                radius: 2.0,
            },
        ],
    };

    SceneOptimizerConfig {
        steps,
        sh_channels,
        seed: 42,
    }
}

/// Borrowed view of the five flat per-Gaussian arrays that describe a scene.
///
/// The arrays always travel together and are always the same length in
/// Gaussians, so entry points that take a whole scene take one of these
/// rather than five same-typed `&[f32]` parameters that are trivial to pass
/// in the wrong order.
#[derive(Debug, Clone, Copy)]
pub struct GaussianArrays<'a> {
    /// `3 × n` XYZ centres.
    pub positions: &'a [f32],
    /// `4 × n` rotation quaternions.
    pub rotations: &'a [f32],
    /// `3 × n` log-scales (see the module-level "Scale space" note).
    pub scales: &'a [f32],
    /// `n` opacity logits.
    pub opacities: &'a [f32],
    /// `sh_channels × n` spherical-harmonic coefficients, or empty.
    pub sh_coefficients: &'a [f32],
}

/// One-line convenience function: build a pipeline from a profile and run it.
///
/// # Errors
///
/// Propagates every error [`OptimizationPipeline::run`] reports: an empty
/// scene, or an array whose length disagrees with `n`/`sh_channels`.
pub fn so_quick_optimize(
    arrays: GaussianArrays<'_>,
    n: usize,
    sh_channels: usize,
    profile: OptimizationProfile,
) -> Result<(OptimizedScene, OptimizationReport), OptimizerError> {
    let config = so_profile_config(profile, sh_channels);
    let pipeline = OptimizationPipeline::new(config);
    pipeline.run(
        arrays.positions,
        arrays.rotations,
        arrays.scales,
        arrays.opacities,
        arrays.sh_coefficients,
        n,
    )
}

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------

/// Human-readable multi-line report of an entire pipeline run.
pub fn so_format_report(report: &OptimizationReport) -> String {
    let mut s = String::new();
    s.push_str("=== Optimization Report ===\n");
    s.push_str(&format!("Steps: {}\n", report.step_results.len()));
    s.push_str(&format!(
        "Gaussians: {} → {} (removed {}, {:.1}%)\n",
        report.snapshot_before.n_gaussians,
        report.snapshot_after.n_gaussians,
        report.total_removed,
        report.total_reduction_percent
    ));
    s.push_str(&format!(
        "Memory saved: {} bytes\n",
        report.memory_saved_bytes
    ));
    s.push_str("\nBefore:\n");
    s.push_str(&so_format_snapshot(&report.snapshot_before));
    s.push_str("\nAfter:\n");
    s.push_str(&so_format_snapshot(&report.snapshot_after));
    s.push_str("\nStep breakdown:\n");
    for result in &report.step_results {
        s.push_str(&so_format_step_result(result));
        s.push('\n');
    }
    s
}

/// One-line summary of a step result.
pub fn so_format_step_result(result: &OptimizationStepResult) -> String {
    format!(
        "  [{}] {} → {} (removed {}) [{}] {}",
        result.step_name,
        result.n_before,
        result.n_after,
        result.n_removed,
        result.duration_hint,
        result.notes
    )
}

/// Multi-line snapshot summary.
pub fn so_format_snapshot(snapshot: &SceneSnapshot) -> String {
    // "log-scale" labels mean_scale/max_scale explicitly: they are means/
    // maxima of raw (unexponentiated) scale values, not linear world-unit
    // sizes — see the module-level "Scale space" note.
    format!(
        "  n={} opacity(mean={:.4} max={:.4}) log-scale(mean={:.4} max={:.4})\n  bounds=[{:.3},{:.3},{:.3}]→[{:.3},{:.3},{:.3}] mem={}B\n",
        snapshot.n_gaussians,
        snapshot.mean_opacity,
        snapshot.max_opacity,
        snapshot.mean_scale,
        snapshot.max_scale,
        snapshot.bounds_min[0],
        snapshot.bounds_min[1],
        snapshot.bounds_min[2],
        snapshot.bounds_max[0],
        snapshot.bounds_max[1],
        snapshot.bounds_max[2],
        snapshot.memory_bytes
    )
}

/// Human-readable configuration summary.
pub fn so_format_config(config: &SceneOptimizerConfig) -> String {
    let mut s = format!(
        "SceneOptimizerConfig {{ sh_channels={}, seed={}, steps=[\n",
        config.sh_channels, config.seed
    );
    for step in &config.steps {
        s.push_str(&format!("  {:?}\n", step));
    }
    s.push_str("] }");
    s
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests;
