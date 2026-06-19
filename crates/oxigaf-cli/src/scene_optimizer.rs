//! Scene optimization pipeline for 3DGS (3D Gaussian Splatting) scenes.
//!
//! Provides a configurable multi-step pipeline that deduplicates, prunes,
//! sorts, clamps, and clips Gaussians to reduce scene size before deployment.
//!
//! All functions use the `so_` prefix to avoid name conflicts with other CLI modules.

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
    /// Mean over all scale components.
    pub mean_scale: f32,
    /// Max scale component.
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
pub fn so_compute_snapshot(
    positions: &[f32],
    scales: &[f32],
    opacities: &[f32],
    n: usize,
    sh_channels: usize,
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

    // Opacities — apply sigmoid
    let mut sum_opacity = 0.0f32;
    let mut max_opacity = f32::NEG_INFINITY;
    for i in 0..n {
        let s = so_sigmoid(opacities[i]);
        sum_opacity += s;
        if s > max_opacity {
            max_opacity = s;
        }
    }
    let mean_opacity = sum_opacity / n as f32;

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
    let mean_scale = if scale_count > 0 {
        sum_scale / scale_count as f32
    } else {
        0.0
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

    /// Clamp per-component scale values to [min_scale, max_scale].
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

#[inline]
fn so_sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ---------------------------------------------------------------------------
// Mask application helpers
// ---------------------------------------------------------------------------

/// Apply a keep mask with arbitrary stride; retains rows where `mask[i]` is true.
pub fn so_apply_keep_mask_nd(
    data: &[f32],
    mask: &[bool],
    n: usize,
    stride: usize,
) -> Vec<f32> {
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
pub fn so_reorder_by_indices(
    data: &[f32],
    indices: &[usize],
    n: usize,
    stride: usize,
) -> Vec<f32> {
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

/// Compute keep mask: sigmoid(logit) > threshold (strict).
///
/// Note: sigmoid always returns a value in (0, 1), so threshold=0 keeps everything.
pub fn so_prune_by_opacity(opacities: &[f32], n: usize, threshold: f32) -> Vec<bool> {
    let mut mask = vec![true; n];
    for i in 0..n.min(opacities.len()) {
        mask[i] = so_sigmoid(opacities[i]) > threshold;
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
        // O(N²) brute force
        for i in 0..n {
            if !keep[i] {
                continue;
            }
            let xi = positions.get(i * 3).copied().unwrap_or(0.0);
            let yi = positions.get(i * 3 + 1).copied().unwrap_or(0.0);
            let zi = positions.get(i * 3 + 2).copied().unwrap_or(0.0);
            for j in (i + 1)..n {
                if !keep[j] {
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

/// Clamp every scale component to [min_scale, max_scale] in-place.
pub fn so_clamp_scales(scales: &mut Vec<f32>, n: usize, min_scale: f32, max_scale: f32) {
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
                let pos = [
                    positions[i * 3],
                    positions[i * 3 + 1],
                    positions[i * 3 + 2],
                ];
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

/// Keep mask: top-n by sigmoid(opacity).
///
/// If n_keep >= n_total, all are kept.
pub fn so_top_n_by_opacity(opacities: &[f32], n_total: usize, n_keep: usize) -> Vec<bool> {
    if n_keep >= n_total {
        return vec![true; n_total];
    }

    // Build (sigmoid_opacity, index) pairs
    let mut indexed: Vec<(f32, usize)> = opacities[..n_total.min(opacities.len())]
        .iter()
        .enumerate()
        .map(|(i, &v)| (so_sigmoid(v), i))
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
pub fn so_clip_to_sphere(
    positions: &[f32],
    n: usize,
    center: [f32; 3],
    radius: f32,
) -> Vec<bool> {
    let r2 = radius * radius;
    let mut mask = vec![true; n];
    for i in 0..n {
        let x = positions.get(i * 3).copied().unwrap_or(0.0) - center[0];
        let y = positions.get(i * 3 + 1).copied().unwrap_or(0.0) - center[1];
        let z = positions.get(i * 3 + 2).copied().unwrap_or(0.0) - center[2];
        mask[i] = x * x + y * y + z * z <= r2;
    }
    mask
}

/// Keep mask: points inside the AABB [min, max] (inclusive on both sides).
pub fn so_clip_to_aabb(
    positions: &[f32],
    n: usize,
    min: [f32; 3],
    max: [f32; 3],
) -> Vec<bool> {
    let mut mask = vec![true; n];
    for i in 0..n {
        let mut inside = true;
        for ax in 0..3 {
            let v = positions.get(i * 3 + ax).copied().unwrap_or(0.0);
            if v < min[ax] || v > max[ax] {
                inside = false;
                break;
            }
        }
        mask[i] = inside;
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
    /// Flat scales array (n_gaussians * 3).
    pub scales: Vec<f32>,
    /// Flat opacity array (n_gaussians).
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

        // Snapshot before
        let snapshot_before =
            so_compute_snapshot(positions, scales, opacities, n_gaussians, sh_channels);

        // Working copies
        let mut cur_positions = positions.to_vec();
        let mut cur_rotations = rotations.to_vec();
        let mut cur_scales = scales.to_vec();
        let mut cur_opacities = opacities.to_vec();
        let mut cur_sh = sh_coefficients.to_vec();
        let mut cur_n = n_gaussians;

        let mut step_results = Vec::with_capacity(self.config.steps.len());

        for step in &self.config.steps {
            let n_before = cur_n;
            let step_name = step.name().to_string();
            let duration_hint = step.duration_hint().to_string();

            let notes = match step {
                OptimizationStep::PruneByOpacity { threshold } => {
                    let mask = so_prune_by_opacity(&cur_opacities, cur_n, *threshold);
                    let (new_positions, new_rotations, new_scales, new_opacities, new_sh, new_n) =
                        apply_mask_all(
                            &cur_positions,
                            &cur_rotations,
                            &cur_scales,
                            &cur_opacities,
                            &cur_sh,
                            &mask,
                            cur_n,
                            sh_channels,
                        );
                    cur_positions = new_positions;
                    cur_rotations = new_rotations;
                    cur_scales = new_scales;
                    cur_opacities = new_opacities;
                    cur_sh = new_sh;
                    cur_n = new_n;
                    format!("threshold={threshold:.4}")
                }

                OptimizationStep::DeduplicateNear { position_radius } => {
                    let mask = so_deduplicate_near(&cur_positions, cur_n, *position_radius);
                    let (new_positions, new_rotations, new_scales, new_opacities, new_sh, new_n) =
                        apply_mask_all(
                            &cur_positions,
                            &cur_rotations,
                            &cur_scales,
                            &cur_opacities,
                            &cur_sh,
                            &mask,
                            cur_n,
                            sh_channels,
                        );
                    cur_positions = new_positions;
                    cur_rotations = new_rotations;
                    cur_scales = new_scales;
                    cur_opacities = new_opacities;
                    cur_sh = new_sh;
                    cur_n = new_n;
                    format!("radius={position_radius:.4}")
                }

                OptimizationStep::ClampScales { min_scale, max_scale } => {
                    if min_scale > max_scale {
                        return Err(OptimizerError::InvalidThreshold {
                            value: *min_scale,
                            param: "min_scale (must be ≤ max_scale)".to_string(),
                        });
                    }
                    so_clamp_scales(&mut cur_scales, cur_n, *min_scale, *max_scale);
                    format!("min={min_scale:.5} max={max_scale:.5}")
                }

                OptimizationStep::SortMorton => {
                    let indices = so_sort_morton(&cur_positions, cur_n);
                    cur_positions =
                        so_reorder_by_indices(&cur_positions, &indices, cur_n, 3);
                    cur_rotations =
                        so_reorder_by_indices(&cur_rotations, &indices, cur_n, 4);
                    cur_scales = so_reorder_by_indices(&cur_scales, &indices, cur_n, 3);
                    cur_opacities =
                        so_reorder_by_indices(&cur_opacities, &indices, cur_n, 1);
                    if !cur_sh.is_empty() {
                        cur_sh = so_reorder_by_indices(
                            &cur_sh,
                            &indices,
                            cur_n,
                            sh_channels,
                        );
                    }
                    "Morton code sort".to_string()
                }

                OptimizationStep::TopNByOpacity { n } => {
                    let mask = so_top_n_by_opacity(&cur_opacities, cur_n, *n);
                    let (new_positions, new_rotations, new_scales, new_opacities, new_sh, new_n) =
                        apply_mask_all(
                            &cur_positions,
                            &cur_rotations,
                            &cur_scales,
                            &cur_opacities,
                            &cur_sh,
                            &mask,
                            cur_n,
                            sh_channels,
                        );
                    cur_positions = new_positions;
                    cur_rotations = new_rotations;
                    cur_scales = new_scales;
                    cur_opacities = new_opacities;
                    cur_sh = new_sh;
                    cur_n = new_n;
                    format!("n_keep={n}")
                }

                OptimizationStep::NormalizeOpacity => {
                    cur_opacities = so_normalize_opacity(&cur_opacities, cur_n);
                    "sigmoid applied".to_string()
                }

                OptimizationStep::ClipToSphere { center, radius } => {
                    let mask = so_clip_to_sphere(&cur_positions, cur_n, *center, *radius);
                    let (new_positions, new_rotations, new_scales, new_opacities, new_sh, new_n) =
                        apply_mask_all(
                            &cur_positions,
                            &cur_rotations,
                            &cur_scales,
                            &cur_opacities,
                            &cur_sh,
                            &mask,
                            cur_n,
                            sh_channels,
                        );
                    cur_positions = new_positions;
                    cur_rotations = new_rotations;
                    cur_scales = new_scales;
                    cur_opacities = new_opacities;
                    cur_sh = new_sh;
                    cur_n = new_n;
                    format!(
                        "center=[{:.2},{:.2},{:.2}] r={:.4}",
                        center[0], center[1], center[2], radius
                    )
                }

                OptimizationStep::ClipToAabb { min, max } => {
                    let mask = so_clip_to_aabb(&cur_positions, cur_n, *min, *max);
                    let (new_positions, new_rotations, new_scales, new_opacities, new_sh, new_n) =
                        apply_mask_all(
                            &cur_positions,
                            &cur_rotations,
                            &cur_scales,
                            &cur_opacities,
                            &cur_sh,
                            &mask,
                            cur_n,
                            sh_channels,
                        );
                    cur_positions = new_positions;
                    cur_rotations = new_rotations;
                    cur_scales = new_scales;
                    cur_opacities = new_opacities;
                    cur_sh = new_sh;
                    cur_n = new_n;
                    format!(
                        "min=[{:.2},{:.2},{:.2}] max=[{:.2},{:.2},{:.2}]",
                        min[0], min[1], min[2], max[0], max[1], max[2]
                    )
                }
            };

            let n_after = cur_n;
            step_results.push(OptimizationStepResult {
                step_name,
                n_before,
                n_after,
                n_removed: n_before.saturating_sub(n_after),
                duration_hint,
                notes,
            });
        }

        let snapshot_after =
            so_compute_snapshot(&cur_positions, &cur_scales, &cur_opacities, cur_n, sh_channels);

        let total_removed = n_gaussians.saturating_sub(cur_n);
        let total_reduction_percent = if n_gaussians > 0 {
            100.0 * total_removed as f32 / n_gaussians as f32
        } else {
            0.0
        };
        let memory_saved_bytes =
            snapshot_before.memory_bytes.saturating_sub(snapshot_after.memory_bytes);

        let report = OptimizationReport {
            step_results,
            snapshot_before,
            snapshot_after,
            total_removed,
            total_reduction_percent,
            memory_saved_bytes,
        };

        let optimized = OptimizedScene {
            positions: cur_positions,
            rotations: cur_rotations,
            scales: cur_scales,
            opacities: cur_opacities,
            sh_coefficients: cur_sh,
            n_gaussians: cur_n,
            sh_channels,
        };

        Ok((optimized, report))
    }
}

/// Internal helper: apply a boolean keep-mask to all five Gaussian arrays at once.
fn apply_mask_all(
    positions: &[f32],
    rotations: &[f32],
    scales: &[f32],
    opacities: &[f32],
    sh: &[f32],
    mask: &[bool],
    n: usize,
    sh_channels: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, usize) {
    let new_positions = so_apply_keep_mask_3d(positions, mask, n);
    let new_rotations = so_apply_keep_mask_4d(rotations, mask, n);
    let new_scales = so_apply_keep_mask_3d(scales, mask, n);
    let new_opacities = so_apply_keep_mask_1d(opacities, mask, n);
    let new_sh = if sh_channels > 0 {
        so_apply_keep_mask_nd(sh, mask, n, sh_channels)
    } else {
        Vec::new()
    };
    let new_n = mask.iter().filter(|&&v| v).count();
    (new_positions, new_rotations, new_scales, new_opacities, new_sh, new_n)
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
            OptimizationStep::ClampScales {
                min_scale: 1e-5,
                max_scale: 0.1,
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

/// One-line convenience function: build a pipeline from a profile and run it.
pub fn so_quick_optimize(
    positions: &[f32],
    rotations: &[f32],
    scales: &[f32],
    opacities: &[f32],
    sh_coefficients: &[f32],
    n: usize,
    sh_channels: usize,
    profile: OptimizationProfile,
) -> Result<(OptimizedScene, OptimizationReport), OptimizerError> {
    let config = so_profile_config(profile, sh_channels);
    let pipeline = OptimizationPipeline::new(config);
    pipeline.run(positions, rotations, scales, opacities, sh_coefficients, n)
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
    format!(
        "  n={} opacity(mean={:.4} max={:.4}) scale(mean={:.4} max={:.4})\n  bounds=[{:.3},{:.3},{:.3}]→[{:.3},{:.3},{:.3}] mem={}B\n",
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
mod tests {
    use super::*;

    // --- so_compute_snapshot ---

    #[test]
    fn test_snapshot_single_gaussian_values() {
        let positions = vec![1.0f32, 2.0, 3.0];
        let scales = vec![0.1f32, 0.2, 0.3];
        let opacities = vec![0.0f32]; // sigmoid(0) = 0.5
        let snap = so_compute_snapshot(&positions, &scales, &opacities, 1, 3);
        assert_eq!(snap.n_gaussians, 1);
        assert!((snap.mean_opacity - 0.5).abs() < 1e-5, "mean_opacity should be ~0.5");
        assert!((snap.max_opacity - 0.5).abs() < 1e-5, "max_opacity should be ~0.5");
        assert!((snap.mean_scale - 0.2).abs() < 1e-5, "mean_scale should be ~0.2");
        assert!((snap.max_scale - 0.3).abs() < 1e-5, "max_scale should be 0.3");
    }

    #[test]
    fn test_snapshot_memory_bytes() {
        let n = 10;
        let sh_channels = 9;
        let positions = vec![0.0f32; n * 3];
        let scales = vec![1.0f32; n * 3];
        let opacities = vec![0.0f32; n];
        let snap = so_compute_snapshot(&positions, &scales, &opacities, n, sh_channels);
        let expected = (3 + 4 + 3 + 1 + sh_channels) * 4 * n;
        assert_eq!(snap.memory_bytes, expected);
    }

    #[test]
    fn test_snapshot_empty() {
        let snap = so_compute_snapshot(&[], &[], &[], 0, 3);
        assert_eq!(snap.n_gaussians, 0);
        assert_eq!(snap.memory_bytes, 0);
    }

    #[test]
    fn test_snapshot_bounds() {
        let positions = vec![1.0f32, 2.0, 3.0, -1.0, -2.0, -3.0];
        let scales = vec![0.1f32; 6];
        let opacities = vec![0.0f32; 2];
        let snap = so_compute_snapshot(&positions, &scales, &opacities, 2, 0);
        assert!((snap.bounds_min[0] - (-1.0)).abs() < 1e-5);
        assert!((snap.bounds_max[0] - 1.0).abs() < 1e-5);
    }

    // --- so_prune_by_opacity ---

    #[test]
    fn test_prune_high_logit_kept() {
        let mask = so_prune_by_opacity(&[10.0f32], 1, 0.5);
        assert!(mask[0], "logit=10 (sigmoid≈1) should be kept with threshold=0.5");
    }

    #[test]
    fn test_prune_low_logit_removed() {
        let mask = so_prune_by_opacity(&[-10.0f32], 1, 0.5);
        assert!(!mask[0], "logit=-10 (sigmoid≈0) should be removed with threshold=0.5");
    }

    #[test]
    fn test_prune_all_above_threshold() {
        let opacities = vec![5.0f32, 3.0, 2.0];
        let mask = so_prune_by_opacity(&opacities, 3, 0.5);
        assert!(mask.iter().all(|&v| v));
    }

    #[test]
    fn test_prune_threshold_zero_keeps_all() {
        // sigmoid(x) > 0 for all finite x, so threshold=0 keeps everything
        let opacities = vec![-100.0f32, -10.0, 0.0, 10.0];
        let mask = so_prune_by_opacity(&opacities, 4, 0.0);
        assert!(mask.iter().all(|&v| v), "threshold=0 must keep all Gaussians");
    }

    #[test]
    fn test_prune_mixed() {
        let opacities = vec![5.0f32, -5.0]; // sigmoid(5)≈0.993, sigmoid(-5)≈0.007
        let mask = so_prune_by_opacity(&opacities, 2, 0.5);
        assert!(mask[0]);
        assert!(!mask[1]);
    }

    // --- so_deduplicate_near ---

    #[test]
    fn test_dedup_identical_positions_keeps_first() {
        let positions = vec![1.0f32, 2.0, 3.0, 1.0, 2.0, 3.0];
        let mask = so_deduplicate_near(&positions, 2, 0.01);
        assert!(mask[0], "first should be kept");
        assert!(!mask[1], "duplicate should be removed");
    }

    #[test]
    fn test_dedup_distinct_positions_all_kept() {
        let positions = vec![0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0];
        let mask = so_deduplicate_near(&positions, 3, 0.01);
        assert!(mask.iter().all(|&v| v));
    }

    #[test]
    fn test_dedup_radius_zero_keeps_all() {
        let positions = vec![0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0];
        let mask = so_deduplicate_near(&positions, 2, 0.0);
        assert!(mask.iter().all(|&v| v), "radius=0 must keep all");
    }

    #[test]
    fn test_dedup_single_gaussian() {
        let positions = vec![0.0f32, 0.0, 0.0];
        let mask = so_deduplicate_near(&positions, 1, 0.1);
        assert!(mask[0]);
    }

    // --- so_clamp_scales ---

    #[test]
    fn test_clamp_above_max() {
        let mut scales = vec![0.5f32, 1.0, 2.0];
        so_clamp_scales(&mut scales, 1, 0.0, 0.3);
        // Only first Gaussian (3 components), all above max=0.3
        assert!(scales[0] <= 0.3);
        assert!(scales[1] <= 0.3);
        assert!(scales[2] <= 0.3);
    }

    #[test]
    fn test_clamp_below_min() {
        let mut scales = vec![0.001f32, 0.002, 0.003];
        so_clamp_scales(&mut scales, 1, 0.01, 1.0);
        assert!(scales[0] >= 0.01);
    }

    #[test]
    fn test_clamp_in_range_unchanged() {
        let mut scales = vec![0.05f32, 0.06, 0.07];
        so_clamp_scales(&mut scales, 1, 0.01, 0.1);
        assert!((scales[0] - 0.05).abs() < 1e-7);
        assert!((scales[1] - 0.06).abs() < 1e-7);
        assert!((scales[2] - 0.07).abs() < 1e-7);
    }

    #[test]
    fn test_clamp_multiple_gaussians() {
        let mut scales = vec![0.2f32, 0.3, 0.4, 0.001, 0.002, 0.003];
        so_clamp_scales(&mut scales, 2, 0.01, 0.15);
        for &v in &scales {
            assert!(v >= 0.01 && v <= 0.15);
        }
    }

    // --- so_sort_morton ---

    #[test]
    fn test_sort_morton_non_decreasing_codes() {
        let positions = vec![
            0.5f32, 0.5, 0.5,
            0.1, 0.1, 0.1,
            0.9, 0.9, 0.9,
            0.3, 0.3, 0.3,
        ];
        let indices = so_sort_morton(&positions, 4);
        assert_eq!(indices.len(), 4);

        // Compute bounds
        let bounds_min = [0.1f32; 3];
        let bounds_max = [0.9f32; 3];

        let codes: Vec<u32> = indices
            .iter()
            .map(|&i| {
                let pos = [positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]];
                let [qx, qy, qz] = so_quantize_position(pos, bounds_min, bounds_max, 10);
                so_morton_code(qx, qy, qz)
            })
            .collect();

        for w in codes.windows(2) {
            assert!(w[0] <= w[1], "Morton codes must be non-decreasing");
        }
    }

    #[test]
    fn test_sort_morton_single() {
        let indices = so_sort_morton(&[0.0f32, 0.0, 0.0], 1);
        assert_eq!(indices, vec![0]);
    }

    #[test]
    fn test_sort_morton_empty() {
        let indices = so_sort_morton(&[], 0);
        assert!(indices.is_empty());
    }

    // --- so_top_n_by_opacity ---

    #[test]
    fn test_top_n_keeps_highest() {
        let opacities = vec![-5.0f32, 10.0, -3.0]; // sigmoid: ~0.007, ~1.0, ~0.047
        let mask = so_top_n_by_opacity(&opacities, 3, 1);
        assert!(!mask[0]);
        assert!(mask[1], "highest opacity (index 1) must be kept");
        assert!(!mask[2]);
    }

    #[test]
    fn test_top_n_keep_all() {
        let opacities = vec![1.0f32, 2.0, 3.0];
        let mask = so_top_n_by_opacity(&opacities, 3, 3);
        assert!(mask.iter().all(|&v| v));
    }

    #[test]
    fn test_top_n_keep_more_than_total() {
        let opacities = vec![1.0f32, 2.0];
        let mask = so_top_n_by_opacity(&opacities, 2, 100);
        assert!(mask.iter().all(|&v| v));
    }

    #[test]
    fn test_top_n_count_correct() {
        let opacities = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let mask = so_top_n_by_opacity(&opacities, 5, 2);
        let kept = mask.iter().filter(|&&v| v).count();
        assert_eq!(kept, 2);
    }

    // --- so_normalize_opacity ---

    #[test]
    fn test_normalize_zero_logit() {
        let result = so_normalize_opacity(&[0.0f32], 1);
        assert!((result[0] - 0.5).abs() < 1e-5, "sigmoid(0) must be 0.5");
    }

    #[test]
    fn test_normalize_large_positive_logit() {
        let result = so_normalize_opacity(&[100.0f32], 1);
        assert!(result[0] > 0.999, "sigmoid(100) must be ≈1.0");
    }

    #[test]
    fn test_normalize_large_negative_logit() {
        let result = so_normalize_opacity(&[-100.0f32], 1);
        assert!(result[0] < 0.001, "sigmoid(-100) must be ≈0.0");
    }

    #[test]
    fn test_normalize_all_in_range() {
        let opacities = vec![-5.0f32, -1.0, 0.0, 1.0, 5.0];
        let result = so_normalize_opacity(&opacities, 5);
        for &v in &result {
            assert!(v > 0.0 && v < 1.0);
        }
    }

    // --- so_clip_to_sphere ---

    #[test]
    fn test_clip_sphere_center_kept() {
        let positions = vec![0.0f32, 0.0, 0.0];
        let mask = so_clip_to_sphere(&positions, 1, [0.0, 0.0, 0.0], 1.0);
        assert!(mask[0]);
    }

    #[test]
    fn test_clip_sphere_outside_removed() {
        let positions = vec![10.0f32, 0.0, 0.0];
        let mask = so_clip_to_sphere(&positions, 1, [0.0, 0.0, 0.0], 1.0);
        assert!(!mask[0]);
    }

    #[test]
    fn test_clip_sphere_on_boundary_kept() {
        let positions = vec![1.0f32, 0.0, 0.0];
        let mask = so_clip_to_sphere(&positions, 1, [0.0, 0.0, 0.0], 1.0);
        assert!(mask[0], "point exactly on boundary should be kept");
    }

    // --- so_clip_to_aabb ---

    #[test]
    fn test_clip_aabb_inside_kept() {
        let positions = vec![0.5f32, 0.5, 0.5];
        let mask = so_clip_to_aabb(&positions, 1, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        assert!(mask[0]);
    }

    #[test]
    fn test_clip_aabb_outside_removed() {
        let positions = vec![2.0f32, 0.5, 0.5];
        let mask = so_clip_to_aabb(&positions, 1, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        assert!(!mask[0]);
    }

    #[test]
    fn test_clip_aabb_mixed() {
        let positions = vec![
            0.5f32, 0.5, 0.5, // inside
            2.0, 0.5, 0.5, // outside x
        ];
        let mask = so_clip_to_aabb(&positions, 2, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        assert!(mask[0]);
        assert!(!mask[1]);
    }

    // --- mask helpers ---

    #[test]
    fn test_apply_keep_mask_nd_stride3() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mask = vec![true, false];
        let result = so_apply_keep_mask_nd(&data, &mask, 2, 3);
        assert_eq!(result, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_apply_keep_mask_1d_scalar() {
        let data = vec![10.0f32, 20.0, 30.0];
        let mask = vec![true, false, true];
        let result = so_apply_keep_mask_1d(&data, &mask, 3);
        assert_eq!(result, vec![10.0, 30.0]);
    }

    #[test]
    fn test_apply_keep_mask_3d() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let mask = vec![false, true, false];
        let result = so_apply_keep_mask_3d(&data, &mask, 3);
        assert_eq!(result, vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_apply_keep_mask_4d() {
        let data = vec![
            1.0f32, 2.0, 3.0, 4.0, // row 0
            5.0, 6.0, 7.0, 8.0, // row 1
        ];
        let mask = vec![false, true];
        let result = so_apply_keep_mask_4d(&data, &mask, 2);
        assert_eq!(result, vec![5.0, 6.0, 7.0, 8.0]);
    }

    // --- so_reorder_by_indices ---

    #[test]
    fn test_reorder_identity() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let indices = vec![0, 1];
        let result = so_reorder_by_indices(&data, &indices, 2, 3);
        assert_eq!(result, data);
    }

    #[test]
    fn test_reorder_reversal() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let indices = vec![1, 0];
        let result = so_reorder_by_indices(&data, &indices, 2, 3);
        assert_eq!(result, vec![4.0, 5.0, 6.0, 1.0, 2.0, 3.0]);
    }

    // --- so_morton_interleave ---

    #[test]
    fn test_morton_interleave_zero() {
        assert_eq!(so_morton_interleave(0), 0);
    }

    #[test]
    fn test_morton_interleave_one() {
        assert_eq!(so_morton_interleave(1), 1);
    }

    #[test]
    fn test_morton_interleave_two() {
        // 2 = 0b10 → bit 1 spreads to position 3 → 0b1000 = 8
        assert_eq!(so_morton_interleave(2), 8);
    }

    // --- so_morton_code ---

    #[test]
    fn test_morton_code_origin() {
        assert_eq!(so_morton_code(0, 0, 0), 0);
    }

    #[test]
    fn test_morton_code_x1() {
        // (1,0,0): interleave(1)=1, rest=0 → 1
        assert_eq!(so_morton_code(1, 0, 0), 1);
    }

    #[test]
    fn test_morton_code_y1() {
        // (0,1,0): interleave(1)<<1 = 2
        assert_eq!(so_morton_code(0, 1, 0), 2);
    }

    #[test]
    fn test_morton_code_z1() {
        // (0,0,1): interleave(1)<<2 = 4
        assert_eq!(so_morton_code(0, 0, 1), 4);
    }

    // --- so_quantize_position ---

    #[test]
    fn test_quantize_min_maps_to_zero() {
        let [qx, qy, qz] =
            so_quantize_position([0.0, 0.0, 0.0], [0.0; 3], [1.0; 3], 10);
        assert_eq!(qx, 0);
        assert_eq!(qy, 0);
        assert_eq!(qz, 0);
    }

    #[test]
    fn test_quantize_max_maps_to_2pow_bits_minus1() {
        let [qx, qy, qz] =
            so_quantize_position([1.0, 1.0, 1.0], [0.0; 3], [1.0; 3], 10);
        assert_eq!(qx, 1023);
        assert_eq!(qy, 1023);
        assert_eq!(qz, 1023);
    }

    #[test]
    fn test_quantize_midpoint() {
        let [qx, _, _] =
            so_quantize_position([0.5, 0.0, 0.0], [0.0; 3], [1.0; 3], 10);
        // 0.5 * 1023 = 511.5 → rounds to 512
        assert_eq!(qx, 512);
    }

    // --- OptimizationPipeline::run ---

    fn make_scene(n: usize, sh_ch: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let positions: Vec<f32> = (0..n * 3).map(|i| i as f32 * 0.01).collect();
        let rotations: Vec<f32> = (0..n)
            .flat_map(|_| [0.0f32, 0.0, 0.0, 1.0])
            .collect();
        let scales: Vec<f32> = vec![0.05f32; n * 3];
        let opacities: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5 - 2.0).collect();
        let sh: Vec<f32> = vec![0.0f32; n * sh_ch];
        (positions, rotations, scales, opacities, sh)
    }

    #[test]
    fn test_pipeline_empty_scene_error() {
        let config = SceneOptimizerConfig {
            steps: vec![OptimizationStep::PruneByOpacity { threshold: 0.5 }],
            sh_channels: 9,
            seed: 0,
        };
        let pipeline = OptimizationPipeline::new(config);
        let result = pipeline.run(&[], &[], &[], &[], &[], 0);
        assert!(matches!(result, Err(OptimizerError::EmptyScene)));
    }

    #[test]
    fn test_pipeline_single_prune_step() {
        let n = 10;
        let sh_ch = 3;
        let (pos, rot, scl, op, sh) = make_scene(n, sh_ch);
        let config = SceneOptimizerConfig {
            steps: vec![OptimizationStep::PruneByOpacity { threshold: 0.5 }],
            sh_channels: sh_ch,
            seed: 0,
        };
        let pipeline = OptimizationPipeline::new(config);
        let result = pipeline.run(&pos, &rot, &scl, &op, &sh, n);
        assert!(result.is_ok());
        let (scene, report) = result.expect("pipeline should succeed");
        assert!(scene.n_gaussians <= n);
        assert_eq!(report.step_results.len(), 1);
    }

    #[test]
    fn test_pipeline_n_after_le_n_before() {
        let n = 20;
        let sh_ch = 9;
        let (pos, rot, scl, op, sh) = make_scene(n, sh_ch);
        let config = SceneOptimizerConfig {
            steps: vec![
                OptimizationStep::PruneByOpacity { threshold: 0.3 },
                OptimizationStep::DeduplicateNear { position_radius: 0.05 },
            ],
            sh_channels: sh_ch,
            seed: 0,
        };
        let pipeline = OptimizationPipeline::new(config);
        let (scene, report) = pipeline.run(&pos, &rot, &scl, &op, &sh, n)
            .expect("pipeline should succeed");
        for step_result in &report.step_results {
            assert!(step_result.n_after <= step_result.n_before);
        }
        assert!(scene.n_gaussians <= n);
    }

    #[test]
    fn test_pipeline_step_results_count() {
        let n = 5;
        let sh_ch = 3;
        let (pos, rot, scl, op, sh) = make_scene(n, sh_ch);
        let steps = vec![
            OptimizationStep::PruneByOpacity { threshold: 0.1 },
            OptimizationStep::ClampScales { min_scale: 0.01, max_scale: 0.5 },
            OptimizationStep::SortMorton,
        ];
        let config = SceneOptimizerConfig {
            steps: steps.clone(),
            sh_channels: sh_ch,
            seed: 0,
        };
        let pipeline = OptimizationPipeline::new(config);
        let (_, report) = pipeline.run(&pos, &rot, &scl, &op, &sh, n)
            .expect("pipeline should succeed");
        assert_eq!(report.step_results.len(), steps.len());
    }

    #[test]
    fn test_pipeline_snapshot_after_matches_scene() {
        let n = 10;
        let sh_ch = 3;
        let (pos, rot, scl, op, sh) = make_scene(n, sh_ch);
        let config = SceneOptimizerConfig {
            steps: vec![OptimizationStep::PruneByOpacity { threshold: 0.5 }],
            sh_channels: sh_ch,
            seed: 0,
        };
        let pipeline = OptimizationPipeline::new(config);
        let (scene, report) = pipeline.run(&pos, &rot, &scl, &op, &sh, n)
            .expect("pipeline should succeed");
        assert_eq!(report.snapshot_after.n_gaussians, scene.n_gaussians);
    }

    #[test]
    fn test_report_total_removed() {
        let n = 10;
        let sh_ch = 3;
        let (pos, rot, scl, op, sh) = make_scene(n, sh_ch);
        let config = SceneOptimizerConfig {
            steps: vec![OptimizationStep::PruneByOpacity { threshold: 0.5 }],
            sh_channels: sh_ch,
            seed: 0,
        };
        let pipeline = OptimizationPipeline::new(config);
        let (scene, report) = pipeline.run(&pos, &rot, &scl, &op, &sh, n)
            .expect("pipeline should succeed");
        assert_eq!(
            report.total_removed,
            n - scene.n_gaussians,
            "total_removed must equal n_before - n_after"
        );
    }

    #[test]
    fn test_report_memory_saved() {
        let n = 10;
        let sh_ch = 9;
        let (pos, rot, scl, op, sh) = make_scene(n, sh_ch);
        let config = SceneOptimizerConfig {
            steps: vec![OptimizationStep::PruneByOpacity { threshold: 0.5 }],
            sh_channels: sh_ch,
            seed: 0,
        };
        let pipeline = OptimizationPipeline::new(config);
        let (_, report) = pipeline.run(&pos, &rot, &scl, &op, &sh, n)
            .expect("pipeline should succeed");
        let expected_saved = report.snapshot_before.memory_bytes
            .saturating_sub(report.snapshot_after.memory_bytes);
        assert_eq!(report.memory_saved_bytes, expected_saved);
    }

    // --- so_profile_config ---

    #[test]
    fn test_profile_quality_only_dedup() {
        let config = so_profile_config(OptimizationProfile::Quality, 9);
        assert_eq!(config.steps.len(), 1);
        assert!(
            matches!(config.steps[0], OptimizationStep::DeduplicateNear { .. }),
            "Quality profile should only contain DeduplicateNear"
        );
    }

    #[test]
    fn test_profile_performance_includes_topn() {
        let config = so_profile_config(OptimizationProfile::Performance, 9);
        let has_topn = config.steps.iter().any(|s| matches!(s, OptimizationStep::TopNByOpacity { .. }));
        assert!(has_topn, "Performance profile must include TopNByOpacity");
    }

    #[test]
    fn test_profile_streaming_includes_sort_morton() {
        let config = so_profile_config(OptimizationProfile::Streaming, 9);
        let has_sort = config.steps.iter().any(|s| matches!(s, OptimizationStep::SortMorton));
        assert!(has_sort, "Streaming profile must include SortMorton");
    }

    #[test]
    fn test_profile_balanced_has_multiple_steps() {
        let config = so_profile_config(OptimizationProfile::Balanced, 9);
        assert!(config.steps.len() >= 2, "Balanced profile needs multiple steps");
    }

    // --- so_quick_optimize ---

    #[test]
    fn test_quick_optimize_quality_no_error() {
        let n = 5;
        let sh_ch = 3;
        let (pos, rot, scl, op, sh) = make_scene(n, sh_ch);
        let result = so_quick_optimize(&pos, &rot, &scl, &op, &sh, n, sh_ch, OptimizationProfile::Quality);
        assert!(result.is_ok(), "quick_optimize with Quality profile must succeed");
    }

    #[test]
    fn test_quick_optimize_balanced_no_error() {
        let n = 5;
        let sh_ch = 3;
        let (pos, rot, scl, op, sh) = make_scene(n, sh_ch);
        let result = so_quick_optimize(&pos, &rot, &scl, &op, &sh, n, sh_ch, OptimizationProfile::Balanced);
        assert!(result.is_ok());
    }

    // --- formatting ---

    #[test]
    fn test_format_report_nonempty_with_step_count() {
        let n = 5;
        let sh_ch = 3;
        let (pos, rot, scl, op, sh) = make_scene(n, sh_ch);
        let config = SceneOptimizerConfig {
            steps: vec![OptimizationStep::PruneByOpacity { threshold: 0.5 }],
            sh_channels: sh_ch,
            seed: 0,
        };
        let pipeline = OptimizationPipeline::new(config);
        let (_, report) = pipeline.run(&pos, &rot, &scl, &op, &sh, n)
            .expect("pipeline should succeed");
        let s = so_format_report(&report);
        assert!(!s.is_empty());
        assert!(s.contains("Steps:"), "report must mention step count");
    }

    #[test]
    fn test_format_snapshot_nonempty() {
        let snap = so_compute_snapshot(
            &[0.0f32, 0.0, 0.0],
            &[0.1f32, 0.1, 0.1],
            &[0.0f32],
            1,
            3,
        );
        let s = so_format_snapshot(&snap);
        assert!(!s.is_empty());
    }

    #[test]
    fn test_format_config_nonempty() {
        let config = so_profile_config(OptimizationProfile::Performance, 9);
        let s = so_format_config(&config);
        assert!(!s.is_empty());
    }

    #[test]
    fn test_format_step_result_nonempty() {
        let result = OptimizationStepResult {
            step_name: "PruneByOpacity".to_string(),
            n_before: 100,
            n_after: 80,
            n_removed: 20,
            duration_hint: "fast".to_string(),
            notes: "threshold=0.5".to_string(),
        };
        let s = so_format_step_result(&result);
        assert!(!s.is_empty());
    }

    // --- error cases ---

    #[test]
    fn test_clamp_scales_invalid_min_gt_max_pipeline_error() {
        let n = 3;
        let sh_ch = 3;
        let (pos, rot, scl, op, sh) = make_scene(n, sh_ch);
        let config = SceneOptimizerConfig {
            steps: vec![OptimizationStep::ClampScales {
                min_scale: 1.0,
                max_scale: 0.01, // min > max → error
            }],
            sh_channels: sh_ch,
            seed: 0,
        };
        let pipeline = OptimizationPipeline::new(config);
        let result = pipeline.run(&pos, &rot, &scl, &op, &sh, n);
        assert!(
            matches!(result, Err(OptimizerError::InvalidThreshold { .. })),
            "min_scale > max_scale must produce InvalidThreshold"
        );
    }

    // --- additional coverage ---

    #[test]
    fn test_pipeline_sort_morton_preserves_count() {
        let n = 8;
        let sh_ch = 3;
        let (pos, rot, scl, op, sh) = make_scene(n, sh_ch);
        let config = SceneOptimizerConfig {
            steps: vec![OptimizationStep::SortMorton],
            sh_channels: sh_ch,
            seed: 0,
        };
        let pipeline = OptimizationPipeline::new(config);
        let (scene, _) = pipeline.run(&pos, &rot, &scl, &op, &sh, n)
            .expect("Morton sort pipeline should succeed");
        assert_eq!(scene.n_gaussians, n, "SortMorton must not remove any Gaussians");
    }

    #[test]
    fn test_pipeline_normalize_opacity_preserves_count() {
        let n = 5;
        let sh_ch = 3;
        let (pos, rot, scl, op, sh) = make_scene(n, sh_ch);
        let config = SceneOptimizerConfig {
            steps: vec![OptimizationStep::NormalizeOpacity],
            sh_channels: sh_ch,
            seed: 0,
        };
        let pipeline = OptimizationPipeline::new(config);
        let (scene, _) = pipeline.run(&pos, &rot, &scl, &op, &sh, n)
            .expect("NormalizeOpacity pipeline should succeed");
        assert_eq!(scene.n_gaussians, n);
    }

    #[test]
    fn test_pipeline_clip_sphere_removes_distant() {
        // Put 3 Gaussians far from origin, 2 at origin
        let positions = vec![
            0.0f32, 0.0, 0.0, // inside
            0.0, 0.1, 0.0, // inside
            5.0, 0.0, 0.0, // outside
            0.0, 5.0, 0.0, // outside
            0.0, 0.0, 5.0, // outside
        ];
        let n = 5;
        let rotations: Vec<f32> = vec![0.0, 0.0, 0.0, 1.0].into_iter().cycle().take(n * 4).collect();
        let scales = vec![0.05f32; n * 3];
        let opacities = vec![0.0f32; n];
        let sh: Vec<f32> = vec![];
        let config = SceneOptimizerConfig {
            steps: vec![OptimizationStep::ClipToSphere {
                center: [0.0, 0.0, 0.0],
                radius: 1.0,
            }],
            sh_channels: 0,
            seed: 0,
        };
        let pipeline = OptimizationPipeline::new(config);
        let (scene, _) = pipeline.run(&positions, &rotations, &scales, &opacities, &sh, n)
            .expect("ClipToSphere pipeline should succeed");
        assert_eq!(scene.n_gaussians, 2, "only 2 Gaussians should survive sphere clip");
    }

    #[test]
    fn test_pipeline_clip_aabb_removes_outside() {
        let positions = vec![
            0.5f32, 0.5, 0.5,  // inside
            2.0, 0.5, 0.5,     // outside
        ];
        let n = 2;
        let rotations: Vec<f32> = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let scales = vec![0.05f32; 6];
        let opacities = vec![0.0f32; 2];
        let sh: Vec<f32> = vec![];
        let config = SceneOptimizerConfig {
            steps: vec![OptimizationStep::ClipToAabb {
                min: [0.0, 0.0, 0.0],
                max: [1.0, 1.0, 1.0],
            }],
            sh_channels: 0,
            seed: 0,
        };
        let pipeline = OptimizationPipeline::new(config);
        let (scene, _) = pipeline.run(&positions, &rotations, &scales, &opacities, &sh, n)
            .expect("ClipToAabb pipeline should succeed");
        assert_eq!(scene.n_gaussians, 1);
    }
}
