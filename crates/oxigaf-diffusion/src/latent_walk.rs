//! Latent space walk and guided navigation utilities for diffusion models.
//!
//! This module focuses on *guided* navigation through latent space, including:
//!
//! - **Named semantic directions** ([`LatentDirection`]): Pre-computed unit vectors
//!   representing meaningful attributes (e.g., "smile", "age"), which can be applied
//!   to a latent to edit the corresponding attribute.
//! - **Walk modes** ([`WalkMode`]): Linear, spherical, circular, and random walk
//!   parameterizations for generating smooth latent paths.
//! - **Core walk functions**: [`linear_walk`], [`spherical_walk`], [`circular_walk`],
//!   [`random_walk`] — each producing a sequence of latent vectors along a path.
//! - **Directed editing**: [`directional_walk`] and [`multi_direction_walk`] for
//!   simultaneous multi-attribute manipulation.
//! - **Path analysis**: [`analyze_walk_path`], [`sample_path`], [`resample_path`] for
//!   measuring and resampling latent trajectories.
//! - **Interactive exploration**: [`LatentExplorer`] accumulates named direction offsets
//!   for iterative attribute editing sessions.
//!
//! Unlike the basic lerp/slerp in `latent_interp`, this module provides higher-level
//! primitives for semantic editing and manifold exploration.
//!
//! ## Design
//!
//! All operations are pure Rust with no external tensor library dependencies.
//! All fallible functions return `Result<T, LatentWalkError>` — no panics, no unwraps.
//! Random walks use an inline xorshift64 + Box-Muller PRNG for reproducible output.

use std::f32::consts::PI;
use thiserror::Error;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during latent walk operations.
#[derive(Debug, Error, PartialEq)]
pub enum LatentWalkError {
    /// A latent vector or direction vector has an unexpected number of dimensions.
    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    /// A configuration parameter is invalid (e.g., step_size <= 0).
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// A walk path or step slice is empty.
    #[error("Walk path or steps list is empty")]
    EmptyPath,

    /// A step index is out of bounds for the given walk.
    #[error("Invalid step index: {step} (total steps: {total})")]
    InvalidStep { step: usize, total: usize },

    /// A numerical computation failed (e.g., near-zero magnitude normalization).
    #[error("Numerical error: {0}")]
    NumericalError(String),
}

// ---------------------------------------------------------------------------
// Internal math helpers
// ---------------------------------------------------------------------------

/// Compute L2 norm of a slice.
#[inline]
fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Normalize a slice to unit length, returning an error if the norm is too small.
fn normalize(v: &[f32]) -> Result<Vec<f32>, LatentWalkError> {
    let norm = l2_norm(v);
    if norm < 1e-8 {
        return Err(LatentWalkError::NumericalError(format!(
            "Cannot normalize vector with near-zero magnitude: {norm:.2e}"
        )));
    }
    Ok(v.iter().map(|x| x / norm).collect())
}

/// Compute dot product of two equal-length slices.
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Compute L2 distance between two equal-length slices.
#[inline]
fn l2_dist(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

/// Check that two slices have the same length, returning DimensionMismatch otherwise.
#[inline]
fn check_same_dim(expected: usize, got: usize) -> Result<(), LatentWalkError> {
    if expected != got {
        Err(LatentWalkError::DimensionMismatch { expected, got })
    } else {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// xorshift64 + Box-Muller PRNG (inline, no external deps)
// ---------------------------------------------------------------------------

/// Advance an xorshift64 state and return the next pseudo-random u64.
#[inline]
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// Generate a pseudo-random f32 in (0, 1] (never exactly 0 to keep Box-Muller safe).
#[inline]
fn xorshift_f32(state: &mut u64) -> f32 {
    let raw = xorshift64(state);
    // Divide by u64::MAX to get [0, 1], then clamp away from 0.
    let v = raw as f32 / u64::MAX as f32;
    v.max(f32::EPSILON)
}

/// Box-Muller transform: given two independent uniform samples in (0,1],
/// returns one standard-normal variate.
#[inline]
fn box_muller(u1: f32, u2: f32) -> f32 {
    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}

// ---------------------------------------------------------------------------
// LatentDirection
// ---------------------------------------------------------------------------

/// A named, unit-length direction in latent space (e.g., "smile", "age").
///
/// Directions are always stored normalized so that applying them with different
/// `step_size` values produces predictable, magnitude-comparable edits.
#[derive(Debug, Clone)]
pub struct LatentDirection {
    /// Human-readable name for this semantic direction.
    pub name: String,
    /// Unit direction vector in latent space.
    pub direction: Vec<f32>,
}

impl LatentDirection {
    /// Create a new [`LatentDirection`], normalizing `direction` to unit length.
    ///
    /// # Errors
    ///
    /// Returns [`LatentWalkError::NumericalError`] if `direction` has a near-zero
    /// L2 norm (< 1e-8), making normalization ambiguous.
    pub fn new(name: impl Into<String>, direction: Vec<f32>) -> Result<Self, LatentWalkError> {
        let unit = normalize(&direction)?;
        Ok(Self {
            name: name.into(),
            direction: unit,
        })
    }

    /// Number of dimensions in this direction vector.
    #[inline]
    pub fn dim(&self) -> usize {
        self.direction.len()
    }

    /// Apply this direction to a latent vector, returning `latent + step_size * direction`.
    ///
    /// # Errors
    ///
    /// Returns [`LatentWalkError::DimensionMismatch`] if `latent.len() != self.dim()`.
    pub fn apply(&self, latent: &[f32], step_size: f32) -> Result<Vec<f32>, LatentWalkError> {
        check_same_dim(self.dim(), latent.len())?;
        Ok(latent
            .iter()
            .zip(self.direction.iter())
            .map(|(l, d)| l + step_size * d)
            .collect())
    }

    /// Return a new [`LatentDirection`] pointing in the opposite direction.
    ///
    /// The new direction's name is prefixed with `-`.
    pub fn reverse(&self) -> Self {
        Self {
            name: format!("-{}", self.name),
            direction: self.direction.iter().map(|x| -x).collect(),
        }
    }

    /// Compose this direction with `other` by summing and renormalizing.
    ///
    /// The composed direction points "between" the two input directions.
    ///
    /// # Errors
    ///
    /// - [`LatentWalkError::DimensionMismatch`] if the two directions have different dims.
    /// - [`LatentWalkError::NumericalError`] if the sum is near-zero (directions cancel).
    pub fn compose(&self, other: &LatentDirection) -> Result<LatentDirection, LatentWalkError> {
        check_same_dim(self.dim(), other.dim())?;
        let sum: Vec<f32> = self
            .direction
            .iter()
            .zip(other.direction.iter())
            .map(|(a, b)| a + b)
            .collect();
        let unit = normalize(&sum)?;
        Ok(LatentDirection {
            name: format!("{}+{}", self.name, other.name),
            direction: unit,
        })
    }
}

// ---------------------------------------------------------------------------
// WalkMode + WalkConfig
// ---------------------------------------------------------------------------

/// Parameterization of a latent walk path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WalkMode {
    /// Linear interpolation between start and end.
    Linear,
    /// Spherical linear interpolation, preserving latent norm.
    Spherical,
    /// Circular walk around a center point in a 2D subspace.
    Circular,
    /// Random walk with per-step Gaussian perturbations.
    RandomWalk,
}

/// Configuration for a latent walk.
#[derive(Debug, Clone)]
pub struct WalkConfig {
    /// Walk parameterization.
    pub mode: WalkMode,
    /// Number of points in the generated path (including both endpoints for
    /// Linear/Spherical, or the full sequence for Circular/RandomWalk).
    pub num_steps: usize,
    /// For [`WalkMode::RandomWalk`]: standard deviation of per-step Gaussian noise.
    pub step_size: f32,
    /// For [`WalkMode::RandomWalk`]: seed for the internal PRNG.
    pub seed: u64,
}

impl Default for WalkConfig {
    fn default() -> Self {
        Self {
            mode: WalkMode::Linear,
            num_steps: 10,
            step_size: 0.1,
            seed: 42,
        }
    }
}

impl WalkConfig {
    /// Validate that the configuration is self-consistent.
    ///
    /// # Errors
    ///
    /// - [`LatentWalkError::InvalidConfig`] if `num_steps < 2`.
    /// - [`LatentWalkError::InvalidConfig`] if `step_size <= 0`.
    pub fn validate(&self) -> Result<(), LatentWalkError> {
        if self.num_steps < 2 {
            return Err(LatentWalkError::InvalidConfig(format!(
                "num_steps must be >= 2, got {}",
                self.num_steps
            )));
        }
        if self.step_size <= 0.0 {
            return Err(LatentWalkError::InvalidConfig(format!(
                "step_size must be > 0, got {}",
                self.step_size
            )));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Core walk functions
// ---------------------------------------------------------------------------

/// Generate a linear walk from `start` to `end`.
///
/// Returns `num_steps` latent vectors evenly spaced along the straight line
/// from `start` (inclusive, t=0) to `end` (inclusive, t=1).
///
/// # Errors
///
/// - [`LatentWalkError::DimensionMismatch`] if `start` and `end` have different lengths.
/// - [`LatentWalkError::InvalidConfig`] if `num_steps < 2`.
pub fn linear_walk(
    start: &[f32],
    end: &[f32],
    num_steps: usize,
) -> Result<Vec<Vec<f32>>, LatentWalkError> {
    if num_steps < 2 {
        return Err(LatentWalkError::InvalidConfig(format!(
            "linear_walk requires num_steps >= 2, got {num_steps}"
        )));
    }
    check_same_dim(start.len(), end.len())?;

    let n = num_steps - 1;
    (0..num_steps)
        .map(|i| {
            let t = i as f32 / n as f32;
            Ok(start
                .iter()
                .zip(end.iter())
                .map(|(s, e)| s + t * (e - s))
                .collect())
        })
        .collect()
}

/// Generate a spherical linear walk (slerp) from `start` to `end`.
///
/// Normalizes both vectors, computes the angle between them, then interpolates
/// along the great-circle arc.  The output magnitude at each step is linearly
/// interpolated between `|start|` and `|end|`.
///
/// Falls back to linear interpolation when `start` and `end` are nearly parallel
/// (angle < 1e-6 rad).
///
/// # Errors
///
/// - [`LatentWalkError::DimensionMismatch`] if vectors have different lengths.
/// - [`LatentWalkError::InvalidConfig`] if `num_steps < 2`.
/// - [`LatentWalkError::NumericalError`] if `start` or `end` has near-zero norm.
pub fn spherical_walk(
    start: &[f32],
    end: &[f32],
    num_steps: usize,
) -> Result<Vec<Vec<f32>>, LatentWalkError> {
    if num_steps < 2 {
        return Err(LatentWalkError::InvalidConfig(format!(
            "spherical_walk requires num_steps >= 2, got {num_steps}"
        )));
    }
    check_same_dim(start.len(), end.len())?;

    let norm_start = l2_norm(start);
    let norm_end = l2_norm(end);

    let start_n = normalize(start)?;
    let end_n = normalize(end)?;

    let cos_angle = dot(&start_n, &end_n).clamp(-1.0, 1.0);
    let angle = cos_angle.acos();

    let n = num_steps - 1;
    (0..num_steps)
        .map(|i| {
            let t = i as f32 / n as f32;
            let mag = (1.0 - t) * norm_start + t * norm_end;

            let unit_vec: Vec<f32> = if angle < 1e-6 {
                // Nearly parallel — fall back to lerp on unit sphere
                start_n
                    .iter()
                    .zip(end_n.iter())
                    .map(|(a, b)| a + t * (b - a))
                    .collect()
            } else {
                let sin_angle = angle.sin();
                let w_start = ((1.0 - t) * angle).sin() / sin_angle;
                let w_end = (t * angle).sin() / sin_angle;
                start_n
                    .iter()
                    .zip(end_n.iter())
                    .map(|(a, b)| w_start * a + w_end * b)
                    .collect()
            };

            Ok(unit_vec.iter().map(|x| x * mag).collect())
        })
        .collect()
}

/// Generate a circular walk around `center` in the 2-D subspace spanned by
/// `axis1` and `axis2`.
///
/// The walk traces a full circle with the given `radius`, sampling `num_steps`
/// equally-spaced angles from 0 to 2π.
///
/// # Arguments
///
/// * `center`   — Center point of the circle in latent space.
/// * `radius`   — Radius of the circle.  Must be > 0.
/// * `axis1`, `axis2` — Two orthonormal vectors defining the walk plane.
///   They should be orthogonal and of equal dimension to `center`, but the
///   function does not enforce orthogonality for generality.
/// * `num_steps` — Number of points sampled along the circle.
///
/// # Errors
///
/// - [`LatentWalkError::DimensionMismatch`] if `center`, `axis1`, or `axis2` differ in length.
/// - [`LatentWalkError::InvalidConfig`] if `radius <= 0` or `num_steps < 2`.
pub fn circular_walk(
    center: &[f32],
    radius: f32,
    axis1: &[f32],
    axis2: &[f32],
    num_steps: usize,
) -> Result<Vec<Vec<f32>>, LatentWalkError> {
    if num_steps < 2 {
        return Err(LatentWalkError::InvalidConfig(format!(
            "circular_walk requires num_steps >= 2, got {num_steps}"
        )));
    }
    if radius <= 0.0 {
        return Err(LatentWalkError::InvalidConfig(format!(
            "circular_walk requires radius > 0, got {radius}"
        )));
    }
    check_same_dim(center.len(), axis1.len())?;
    check_same_dim(center.len(), axis2.len())?;

    (0..num_steps)
        .map(|i| {
            let theta = 2.0 * PI * i as f32 / num_steps as f32;
            let cos_t = theta.cos();
            let sin_t = theta.sin();
            Ok(center
                .iter()
                .zip(axis1.iter())
                .zip(axis2.iter())
                .map(|((c, a1), a2)| c + radius * (cos_t * a1 + sin_t * a2))
                .collect())
        })
        .collect()
}

/// Generate a random walk starting at `start`.
///
/// Each step adds independent Gaussian noise with standard deviation `step_size`
/// to every dimension.  The walk is deterministic given `seed`.
///
/// Uses an inline xorshift64 PRNG with Box-Muller transform — no external RNG
/// dependencies.
///
/// # Arguments
///
/// * `start`     — Starting latent vector.
/// * `num_steps` — Total number of points (including `start`).
/// * `step_size` — Standard deviation of per-step Gaussian noise.
/// * `seed`      — PRNG seed.  Must be non-zero for xorshift64; if 0, it is set to 1.
///
/// # Errors
///
/// - [`LatentWalkError::InvalidConfig`] if `num_steps < 2` or `step_size <= 0`.
pub fn random_walk(
    start: &[f32],
    num_steps: usize,
    step_size: f32,
    seed: u64,
) -> Result<Vec<Vec<f32>>, LatentWalkError> {
    if num_steps < 2 {
        return Err(LatentWalkError::InvalidConfig(format!(
            "random_walk requires num_steps >= 2, got {num_steps}"
        )));
    }
    if step_size <= 0.0 {
        return Err(LatentWalkError::InvalidConfig(format!(
            "random_walk requires step_size > 0, got {step_size}"
        )));
    }

    let mut path = Vec::with_capacity(num_steps);
    path.push(start.to_vec());

    // xorshift64 requires non-zero state
    let mut rng_state = if seed == 0 { 1u64 } else { seed };

    for step_idx in 1..num_steps {
        // Safety: we always push to `path` before the loop body and at end of
        // each iteration, so path always has exactly `step_idx` elements here.
        let prev: Vec<f32> = path[step_idx - 1].clone();
        let next: Vec<f32> = prev
            .iter()
            .map(|&p| {
                let u1 = xorshift_f32(&mut rng_state);
                let u2 = xorshift_f32(&mut rng_state);
                let gauss = box_muller(u1, u2);
                p + step_size * gauss
            })
            .collect();
        path.push(next);
    }

    Ok(path)
}

// ---------------------------------------------------------------------------
// Directed walk (semantic editing)
// ---------------------------------------------------------------------------

/// Walk along a named semantic direction with an arbitrary set of step sizes.
///
/// Returns one latent per entry in `steps`:
/// `output[i] = latent + steps[i] * direction.direction`
///
/// Step sizes may be negative (moving opposite to the direction) and need not
/// be monotone.
///
/// # Errors
///
/// - [`LatentWalkError::EmptyPath`] if `steps` is empty.
/// - [`LatentWalkError::DimensionMismatch`] if `latent.len() != direction.dim()`.
pub fn directional_walk(
    latent: &[f32],
    direction: &LatentDirection,
    steps: &[f32],
) -> Result<Vec<Vec<f32>>, LatentWalkError> {
    if steps.is_empty() {
        return Err(LatentWalkError::EmptyPath);
    }
    check_same_dim(direction.dim(), latent.len())?;

    steps.iter().map(|&s| direction.apply(latent, s)).collect()
}

/// Walk simultaneously along multiple named directions.
///
/// At step `i` (t = i / (num_steps − 1)):
/// ```text
/// offset = Σ  t * step_size_k * direction_k
/// output[i] = latent + offset
/// ```
///
/// All directions are blended linearly from zero (at step 0) to their full
/// contribution (at step num_steps−1).
///
/// # Errors
///
/// - [`LatentWalkError::InvalidConfig`] if `num_steps < 2`.
/// - [`LatentWalkError::DimensionMismatch`] if any direction has a different dim
///   from `latent`.
pub fn multi_direction_walk(
    latent: &[f32],
    directions: &[(LatentDirection, f32)],
    num_steps: usize,
) -> Result<Vec<Vec<f32>>, LatentWalkError> {
    if num_steps < 2 {
        return Err(LatentWalkError::InvalidConfig(format!(
            "multi_direction_walk requires num_steps >= 2, got {num_steps}"
        )));
    }
    for (dir, _) in directions {
        check_same_dim(latent.len(), dir.dim())?;
    }

    let n = num_steps - 1;
    (0..num_steps)
        .map(|i| {
            let t = i as f32 / n as f32;
            let mut point = latent.to_vec();
            for (dir, step_size) in directions {
                for (p, d) in point.iter_mut().zip(dir.direction.iter()) {
                    *p += t * step_size * d;
                }
            }
            Ok(point)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Path analysis
// ---------------------------------------------------------------------------

/// Summary statistics for a latent walk path.
#[derive(Debug, Clone)]
pub struct WalkPathStats {
    /// Number of points in the path.
    pub num_steps: usize,
    /// Sum of L2 distances between consecutive points.
    pub total_distance: f32,
    /// Mean L2 distance per step.
    pub mean_step_distance: f32,
    /// Deviation from a straight line (0.0 = perfectly straight, higher = more curved).
    ///
    /// Computed as the variance of cosine-similarity values between consecutive
    /// step vectors.  When there are fewer than 3 points (fewer than 2 step vectors),
    /// this is 0.0 (undefined curvature treated as straight).
    pub curvature: f32,
    /// L2 distance from path start to path end.
    pub start_to_end_distance: f32,
    /// Ratio `total_distance / start_to_end_distance`.  A value near 1.0 indicates
    /// a nearly straight path; larger values indicate detours.  Returns 1.0 when
    /// start == end.
    pub tortuosity: f32,
}

/// Compute summary statistics for a latent walk path.
///
/// # Errors
///
/// - [`LatentWalkError::EmptyPath`] if `path` has fewer than 2 points.
/// - [`LatentWalkError::DimensionMismatch`] if any two consecutive points differ in length.
pub fn analyze_walk_path(path: &[Vec<f32>]) -> Result<WalkPathStats, LatentWalkError> {
    if path.len() < 2 {
        return Err(LatentWalkError::EmptyPath);
    }

    let num_steps = path.len();
    let dim = path[0].len();

    // Validate all point dimensions against path[0].
    for pt in path.iter().skip(1) {
        if pt.len() != dim {
            return Err(LatentWalkError::DimensionMismatch {
                expected: dim,
                got: pt.len(),
            });
        }
    }

    // Compute step distances
    let step_distances: Vec<f32> = path.windows(2).map(|w| l2_dist(&w[0], &w[1])).collect();

    let total_distance: f32 = step_distances.iter().sum();
    let mean_step_distance = total_distance / (num_steps - 1) as f32;

    // Curvature: variance of cosines between consecutive step vectors
    let step_vectors: Vec<Vec<f32>> = path
        .windows(2)
        .map(|w| {
            w[0].iter()
                .zip(w[1].iter())
                .map(|(a, b)| b - a)
                .collect::<Vec<f32>>()
        })
        .collect();

    let curvature = if step_vectors.len() < 2 {
        // 2-point path: only 1 step vector → 0 direction changes → curvature = 0
        0.0
    } else {
        // Compute dot products between consecutive step vector *directions*
        let dot_products: Vec<f32> = step_vectors
            .windows(2)
            .filter_map(|sv| {
                let n0 = l2_norm(&sv[0]);
                let n1 = l2_norm(&sv[1]);
                if n0 < 1e-12 || n1 < 1e-12 {
                    None
                } else {
                    Some(dot(&sv[0], &sv[1]) / (n0 * n1))
                }
            })
            .collect();

        if dot_products.is_empty() {
            0.0
        } else {
            // Variance of the dot products
            let mean = dot_products.iter().sum::<f32>() / dot_products.len() as f32;
            let variance = dot_products
                .iter()
                .map(|x| (x - mean) * (x - mean))
                .sum::<f32>()
                / dot_products.len() as f32;
            variance
        }
    };

    let start_to_end_distance = l2_dist(&path[0], &path[num_steps - 1]);

    let tortuosity = if start_to_end_distance < 1e-12 {
        1.0
    } else {
        total_distance / start_to_end_distance
    };

    Ok(WalkPathStats {
        num_steps,
        total_distance,
        mean_step_distance,
        curvature,
        start_to_end_distance,
        tortuosity,
    })
}

/// Sample a point along a multi-waypoint path at fractional position `t ∈ [0, 1]`.
///
/// Uses arc-length parameterization: `t` is the fraction of total path length.
/// The returned point is linearly interpolated within the appropriate segment.
///
/// # Errors
///
/// - [`LatentWalkError::EmptyPath`] if `path` has fewer than 2 points.
/// - [`LatentWalkError::DimensionMismatch`] if consecutive points differ in length.
pub fn sample_path(path: &[Vec<f32>], t: f32) -> Result<Vec<f32>, LatentWalkError> {
    if path.len() < 2 {
        return Err(LatentWalkError::EmptyPath);
    }
    let dim = path[0].len();
    for pt in path.iter().skip(1) {
        if pt.len() != dim {
            return Err(LatentWalkError::DimensionMismatch {
                expected: dim,
                got: pt.len(),
            });
        }
    }

    // Clamp t to [0, 1]
    let t = t.clamp(0.0, 1.0);

    // Compute cumulative arc lengths.
    // cum_lengths always has at least one element (initialized below), so
    // tracking `prev` via a mutable variable avoids any fallible indexing.
    let mut cum_lengths = vec![0.0f32];
    let mut running = 0.0f32;
    for w in path.windows(2) {
        running += l2_dist(&w[0], &w[1]);
        cum_lengths.push(running);
    }
    let total = running;

    // Handle degenerate case: all points coincide
    if total < 1e-12 {
        return Ok(path[0].clone());
    }

    let target = t * total;

    // Find which segment contains `target`
    let seg_idx = cum_lengths
        .iter()
        .enumerate()
        .rev()
        .find(|(_, &l)| l <= target)
        .map(|(i, _)| i)
        .unwrap_or(0);

    // Clamp to valid segment range
    let seg_idx = seg_idx.min(path.len() - 2);

    let seg_start = cum_lengths[seg_idx];
    let seg_end = cum_lengths[seg_idx + 1];
    let seg_len = seg_end - seg_start;

    let local_t = if seg_len < 1e-12 {
        0.0
    } else {
        ((target - seg_start) / seg_len).clamp(0.0, 1.0)
    };

    let p0 = &path[seg_idx];
    let p1 = &path[seg_idx + 1];
    Ok(p0
        .iter()
        .zip(p1.iter())
        .map(|(a, b)| a + local_t * (b - a))
        .collect())
}

/// Resample a latent path to exactly `n_points` points evenly spaced by arc length.
///
/// # Errors
///
/// - [`LatentWalkError::EmptyPath`] if `path` has fewer than 2 points.
/// - [`LatentWalkError::InvalidConfig`] if `n_points < 2`.
/// - [`LatentWalkError::DimensionMismatch`] if consecutive points differ in length.
pub fn resample_path(path: &[Vec<f32>], n_points: usize) -> Result<Vec<Vec<f32>>, LatentWalkError> {
    if path.len() < 2 {
        return Err(LatentWalkError::EmptyPath);
    }
    if n_points < 2 {
        return Err(LatentWalkError::InvalidConfig(format!(
            "resample_path requires n_points >= 2, got {n_points}"
        )));
    }

    let n = n_points - 1;
    (0..n_points)
        .map(|i| {
            let t = i as f32 / n as f32;
            sample_path(path, t)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// LatentExplorer
// ---------------------------------------------------------------------------

/// Interactive latent space explorer that accumulates named direction offsets.
///
/// Maintains a base latent and a set of named directions.  Repeated calls to
/// [`LatentExplorer::step`] accumulate offsets along each direction so that the
/// current edited latent is always `base + Σ offset_i * direction_i`.
///
/// This is the primary interface for iterative semantic attribute editing.
#[derive(Debug, Clone)]
pub struct LatentExplorer {
    /// The starting latent vector (never modified after construction).
    pub base_latent: Vec<f32>,
    directions: Vec<LatentDirection>,
    current_offsets: Vec<f32>,
}

impl LatentExplorer {
    /// Create a new explorer anchored at `base`.
    pub fn new(base: Vec<f32>) -> Self {
        Self {
            base_latent: base,
            directions: Vec::new(),
            current_offsets: Vec::new(),
        }
    }

    /// Register a new semantic direction.
    ///
    /// # Errors
    ///
    /// Returns [`LatentWalkError::DimensionMismatch`] if `dir.dim() != base_latent.len()`.
    pub fn add_direction(&mut self, dir: LatentDirection) -> Result<(), LatentWalkError> {
        check_same_dim(self.base_latent.len(), dir.dim())?;
        self.directions.push(dir);
        self.current_offsets.push(0.0);
        Ok(())
    }

    /// Number of registered directions.
    #[inline]
    pub fn num_directions(&self) -> usize {
        self.directions.len()
    }

    /// Advance in direction `direction_idx` by `step_size` and return the updated latent.
    ///
    /// # Errors
    ///
    /// Returns [`LatentWalkError::InvalidStep`] if `direction_idx >= num_directions()`.
    pub fn step(
        &mut self,
        direction_idx: usize,
        step_size: f32,
    ) -> Result<Vec<f32>, LatentWalkError> {
        let total = self.num_directions();
        if direction_idx >= total {
            return Err(LatentWalkError::InvalidStep {
                step: direction_idx,
                total,
            });
        }
        self.current_offsets[direction_idx] += step_size;
        Ok(self.current_latent())
    }

    /// Compute the current latent: `base + Σ offset_i * direction_i`.
    pub fn current_latent(&self) -> Vec<f32> {
        let mut result = self.base_latent.clone();
        for (dir, &offset) in self.directions.iter().zip(self.current_offsets.iter()) {
            for (r, d) in result.iter_mut().zip(dir.direction.iter()) {
                *r += offset * d;
            }
        }
        result
    }

    /// Reset all offsets to zero, returning to `base_latent`.
    pub fn reset(&mut self) {
        for o in self.current_offsets.iter_mut() {
            *o = 0.0;
        }
    }

    /// Get the current cumulative offset for direction `i`.
    ///
    /// Returns 0.0 for out-of-range indices.
    pub fn offset(&self, i: usize) -> f32 {
        self.current_offsets.get(i).copied().unwrap_or(0.0)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // -----------------------------------------------------------------------
    // LatentDirection tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_direction_new_normalizes() {
        let dir = LatentDirection::new("test", vec![3.0, 4.0]).unwrap();
        let norm: f32 = dir.direction.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert_relative_eq!(norm, 1.0, epsilon = 1e-6);
        assert_relative_eq!(dir.direction[0], 0.6, epsilon = 1e-6);
        assert_relative_eq!(dir.direction[1], 0.8, epsilon = 1e-6);
    }

    #[test]
    fn test_direction_new_near_zero_error() {
        let result = LatentDirection::new("bad", vec![0.0, 0.0, 1e-10]);
        assert!(result.is_err(), "near-zero direction should fail");
        assert!(matches!(
            result.unwrap_err(),
            LatentWalkError::NumericalError(_)
        ));
    }

    #[test]
    fn test_direction_apply_correct_offset() {
        let dir = LatentDirection::new("test", vec![1.0, 0.0]).unwrap();
        let latent = vec![2.0, 3.0];
        let result = dir.apply(&latent, 0.5).unwrap();
        // direction is already [1.0, 0.0] (unit); step = 0.5
        assert_relative_eq!(result[0], 2.5, epsilon = 1e-6);
        assert_relative_eq!(result[1], 3.0, epsilon = 1e-6);
    }

    #[test]
    fn test_direction_apply_dimension_mismatch() {
        let dir = LatentDirection::new("test", vec![1.0, 0.0]).unwrap();
        let latent = vec![1.0, 2.0, 3.0];
        let err = dir.apply(&latent, 1.0).unwrap_err();
        assert!(matches!(err, LatentWalkError::DimensionMismatch { .. }));
    }

    #[test]
    fn test_direction_reverse() {
        let dir = LatentDirection::new("smile", vec![1.0, 0.0]).unwrap();
        let rev = dir.reverse();
        assert_eq!(rev.name, "-smile");
        assert_relative_eq!(rev.direction[0], -1.0, epsilon = 1e-6);
        assert_relative_eq!(rev.direction[1], 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_direction_compose_unit_vector() {
        // Two perpendicular unit vectors: compose should bisect at 45°
        let d1 = LatentDirection::new("d1", vec![1.0, 0.0]).unwrap();
        let d2 = LatentDirection::new("d2", vec![0.0, 1.0]).unwrap();
        let composed = d1.compose(&d2).unwrap();
        let norm: f32 = composed.direction.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert_relative_eq!(norm, 1.0, epsilon = 1e-5);
        // Should be [√2/2, √2/2]
        assert_relative_eq!(composed.direction[0], 2.0f32.sqrt() / 2.0, epsilon = 1e-5);
        assert_relative_eq!(composed.direction[1], 2.0f32.sqrt() / 2.0, epsilon = 1e-5);
    }

    // -----------------------------------------------------------------------
    // WalkConfig tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_walk_config_default_is_valid() {
        let cfg = WalkConfig::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_walk_config_validate_num_steps_lt_2() {
        let cfg = WalkConfig {
            num_steps: 1,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_walk_config_validate_step_size_zero() {
        let cfg = WalkConfig {
            step_size: 0.0,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_walk_config_validate_step_size_negative() {
        let cfg = WalkConfig {
            step_size: -0.1,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    // -----------------------------------------------------------------------
    // linear_walk tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_linear_walk_too_few_steps() {
        let err = linear_walk(&[0.0, 0.0], &[1.0, 1.0], 1).unwrap_err();
        assert!(matches!(err, LatentWalkError::InvalidConfig(_)));
    }

    #[test]
    fn test_linear_walk_first_equals_start() {
        let start = vec![1.0, 2.0, 3.0];
        let end = vec![4.0, 5.0, 6.0];
        let path = linear_walk(&start, &end, 5).unwrap();
        for (a, b) in path[0].iter().zip(start.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_linear_walk_last_equals_end() {
        let start = vec![1.0, 2.0, 3.0];
        let end = vec![4.0, 5.0, 6.0];
        let path = linear_walk(&start, &end, 5).unwrap();
        let last = path.last().unwrap();
        for (a, b) in last.iter().zip(end.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_linear_walk_correct_length() {
        let path = linear_walk(&[0.0; 4], &[1.0; 4], 7).unwrap();
        assert_eq!(path.len(), 7);
    }

    #[test]
    fn test_linear_walk_dimension_mismatch() {
        let err = linear_walk(&[0.0, 0.0], &[1.0, 1.0, 1.0], 3).unwrap_err();
        assert!(matches!(err, LatentWalkError::DimensionMismatch { .. }));
    }

    // -----------------------------------------------------------------------
    // spherical_walk tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_spherical_walk_first_equals_start() {
        let start = vec![1.0, 0.0, 0.0];
        let end = vec![0.0, 1.0, 0.0];
        let path = spherical_walk(&start, &end, 5).unwrap();
        for (a, b) in path[0].iter().zip(start.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_spherical_walk_last_equals_end() {
        let start = vec![1.0, 0.0, 0.0];
        let end = vec![0.0, 1.0, 0.0];
        let path = spherical_walk(&start, &end, 5).unwrap();
        let last = path.last().unwrap();
        for (a, b) in last.iter().zip(end.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_spherical_walk_preserves_norm_approximately() {
        // Both start and end have norm 2.0; intermediate points should have norm ≈ 2.0
        let start = vec![2.0, 0.0, 0.0];
        let end = vec![0.0, 2.0, 0.0];
        let path = spherical_walk(&start, &end, 9).unwrap();
        for pt in &path {
            let norm: f32 = pt.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert_relative_eq!(norm, 2.0, epsilon = 1e-4);
        }
    }

    #[test]
    fn test_spherical_walk_correct_length() {
        let path = spherical_walk(&[1.0, 0.0], &[0.0, 1.0], 11).unwrap();
        assert_eq!(path.len(), 11);
    }

    // -----------------------------------------------------------------------
    // circular_walk tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_circular_walk_correct_length() {
        let center = vec![0.0, 0.0, 0.0];
        let axis1 = vec![1.0, 0.0, 0.0];
        let axis2 = vec![0.0, 1.0, 0.0];
        let path = circular_walk(&center, 1.0, &axis1, &axis2, 8).unwrap();
        assert_eq!(path.len(), 8);
    }

    #[test]
    fn test_circular_walk_center_is_mean() {
        // Mean of points on a circle should be approximately the center
        let center = vec![1.0, 2.0, 3.0];
        let axis1 = vec![1.0, 0.0, 0.0];
        let axis2 = vec![0.0, 1.0, 0.0];
        let path = circular_walk(&center, 1.0, &axis1, &axis2, 100).unwrap();
        let n = path.len() as f32;
        let mean: Vec<f32> = (0..3)
            .map(|i| path.iter().map(|p| p[i]).sum::<f32>() / n)
            .collect();
        for (m, c) in mean.iter().zip(center.iter()) {
            assert_relative_eq!(m, c, epsilon = 0.05);
        }
    }

    #[test]
    fn test_circular_walk_radius_zero_error() {
        let err = circular_walk(&[0.0; 3], 0.0, &[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0], 4).unwrap_err();
        assert!(matches!(err, LatentWalkError::InvalidConfig(_)));
    }

    // -----------------------------------------------------------------------
    // random_walk tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_random_walk_first_equals_start() {
        let start = vec![1.0, 2.0, 3.0];
        let path = random_walk(&start, 5, 0.1, 42).unwrap();
        for (a, b) in path[0].iter().zip(start.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_random_walk_correct_length() {
        let path = random_walk(&[0.0; 4], 10, 0.1, 1).unwrap();
        assert_eq!(path.len(), 10);
    }

    #[test]
    fn test_random_walk_different_seeds_differ() {
        let start = vec![0.0; 8];
        let p1 = random_walk(&start, 5, 0.5, 42).unwrap();
        let p2 = random_walk(&start, 5, 0.5, 99).unwrap();
        // At least one step should differ
        let any_differ = p1
            .iter()
            .zip(p2.iter())
            .any(|(a, b)| a.iter().zip(b.iter()).any(|(x, y)| (x - y).abs() > 1e-6));
        assert!(any_differ, "different seeds should produce different walks");
    }

    #[test]
    fn test_random_walk_too_few_steps() {
        let err = random_walk(&[0.0; 4], 1, 0.1, 42).unwrap_err();
        assert!(matches!(err, LatentWalkError::InvalidConfig(_)));
    }

    // -----------------------------------------------------------------------
    // directional_walk tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_directional_walk_empty_steps_error() {
        let dir = LatentDirection::new("d", vec![1.0, 0.0]).unwrap();
        let err = directional_walk(&[0.0, 0.0], &dir, &[]).unwrap_err();
        assert!(matches!(err, LatentWalkError::EmptyPath));
    }

    #[test]
    fn test_directional_walk_step_zero_equals_latent() {
        let dir = LatentDirection::new("d", vec![1.0, 0.0]).unwrap();
        let latent = vec![3.0, 4.0];
        let path = directional_walk(&latent, &dir, &[0.0]).unwrap();
        for (a, b) in path[0].iter().zip(latent.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_directional_walk_dimension_mismatch() {
        let dir = LatentDirection::new("d", vec![1.0, 0.0]).unwrap();
        let err = directional_walk(&[0.0, 0.0, 0.0], &dir, &[1.0]).unwrap_err();
        assert!(matches!(err, LatentWalkError::DimensionMismatch { .. }));
    }

    // -----------------------------------------------------------------------
    // multi_direction_walk tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_multi_direction_walk_first_equals_latent() {
        let d1 = LatentDirection::new("d1", vec![1.0, 0.0]).unwrap();
        let d2 = LatentDirection::new("d2", vec![0.0, 1.0]).unwrap();
        let latent = vec![5.0, 5.0];
        let path = multi_direction_walk(&latent, &[(d1, 1.0), (d2, 1.0)], 5).unwrap();
        for (a, b) in path[0].iter().zip(latent.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_multi_direction_walk_too_few_steps() {
        let d = LatentDirection::new("d", vec![1.0]).unwrap();
        let err = multi_direction_walk(&[0.0], &[(d, 1.0)], 1).unwrap_err();
        assert!(matches!(err, LatentWalkError::InvalidConfig(_)));
    }

    // -----------------------------------------------------------------------
    // analyze_walk_path tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_analyze_walk_path_too_few_points() {
        let err = analyze_walk_path(&[vec![0.0, 1.0]]).unwrap_err();
        assert!(matches!(err, LatentWalkError::EmptyPath));
    }

    #[test]
    fn test_analyze_walk_path_straight_line_tortuosity() {
        let path = linear_walk(&[0.0, 0.0], &[3.0, 4.0], 10).unwrap();
        let stats = analyze_walk_path(&path).unwrap();
        assert_relative_eq!(stats.tortuosity, 1.0, epsilon = 1e-4);
    }

    #[test]
    fn test_analyze_walk_path_straight_line_curvature_zero() {
        let path = linear_walk(&[0.0, 0.0], &[1.0, 0.0], 5).unwrap();
        let stats = analyze_walk_path(&path).unwrap();
        assert_relative_eq!(stats.curvature, 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_analyze_walk_path_two_point_curvature_zero() {
        let path = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
        let stats = analyze_walk_path(&path).unwrap();
        assert_relative_eq!(stats.curvature, 0.0, epsilon = 1e-6);
    }

    // -----------------------------------------------------------------------
    // sample_path tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_sample_path_t0_equals_start() {
        let path = linear_walk(&[1.0, 2.0], &[3.0, 4.0], 5).unwrap();
        let pt = sample_path(&path, 0.0).unwrap();
        assert_relative_eq!(pt[0], 1.0, epsilon = 1e-5);
        assert_relative_eq!(pt[1], 2.0, epsilon = 1e-5);
    }

    #[test]
    fn test_sample_path_t1_equals_end() {
        let path = linear_walk(&[1.0, 2.0], &[3.0, 4.0], 5).unwrap();
        let pt = sample_path(&path, 1.0).unwrap();
        assert_relative_eq!(pt[0], 3.0, epsilon = 1e-5);
        assert_relative_eq!(pt[1], 4.0, epsilon = 1e-5);
    }

    #[test]
    fn test_sample_path_empty_error() {
        let err = sample_path(&[vec![0.0]], 0.5).unwrap_err();
        assert!(matches!(err, LatentWalkError::EmptyPath));
    }

    // -----------------------------------------------------------------------
    // resample_path tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_resample_path_two_points() {
        let path = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
        let resampled = resample_path(&path, 2).unwrap();
        assert_eq!(resampled.len(), 2);
    }

    #[test]
    fn test_resample_path_correct_length() {
        let path = linear_walk(&[0.0, 0.0], &[10.0, 0.0], 20).unwrap();
        let resampled = resample_path(&path, 7).unwrap();
        assert_eq!(resampled.len(), 7);
    }

    #[test]
    fn test_resample_path_endpoints_preserved() {
        let path = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![2.0, 1.0],
            vec![3.0, 0.0],
        ];
        let resampled = resample_path(&path, 5).unwrap();
        assert_relative_eq!(resampled[0][0], 0.0, epsilon = 1e-4);
        assert_relative_eq!(resampled[0][1], 0.0, epsilon = 1e-4);
        let last = resampled.last().unwrap();
        assert_relative_eq!(last[0], 3.0, epsilon = 1e-4);
        assert_relative_eq!(last[1], 0.0, epsilon = 1e-4);
    }

    // -----------------------------------------------------------------------
    // LatentExplorer tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_latent_explorer_new_reset_flow() {
        let base = vec![1.0, 2.0, 3.0];
        let mut explorer = LatentExplorer::new(base.clone());

        let d = LatentDirection::new("test", vec![1.0, 0.0, 0.0]).unwrap();
        explorer.add_direction(d).unwrap();

        // Step forward
        let after_step = explorer.step(0, 2.0).unwrap();
        assert_relative_eq!(after_step[0], 3.0, epsilon = 1e-5);

        // Offset recorded
        assert_relative_eq!(explorer.offset(0), 2.0, epsilon = 1e-6);

        // Reset
        explorer.reset();
        let after_reset = explorer.current_latent();
        for (a, b) in after_reset.iter().zip(base.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-6);
        }
        assert_relative_eq!(explorer.offset(0), 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_latent_explorer_add_direction_dimension_mismatch() {
        let mut explorer = LatentExplorer::new(vec![0.0; 4]);
        let d = LatentDirection::new("d", vec![1.0, 0.0]).unwrap();
        let err = explorer.add_direction(d).unwrap_err();
        assert!(matches!(err, LatentWalkError::DimensionMismatch { .. }));
    }

    #[test]
    fn test_latent_explorer_step_invalid_index() {
        let mut explorer = LatentExplorer::new(vec![0.0; 4]);
        let err = explorer.step(0, 1.0).unwrap_err();
        assert!(matches!(err, LatentWalkError::InvalidStep { .. }));
    }

    #[test]
    fn test_latent_explorer_num_directions() {
        let mut explorer = LatentExplorer::new(vec![0.0; 3]);
        assert_eq!(explorer.num_directions(), 0);
        explorer
            .add_direction(LatentDirection::new("a", vec![1.0, 0.0, 0.0]).unwrap())
            .unwrap();
        explorer
            .add_direction(LatentDirection::new("b", vec![0.0, 1.0, 0.0]).unwrap())
            .unwrap();
        assert_eq!(explorer.num_directions(), 2);
    }

    #[test]
    fn test_latent_explorer_current_latent_multiple_directions() {
        let base = vec![0.0, 0.0, 0.0];
        let mut explorer = LatentExplorer::new(base);

        let d1 = LatentDirection::new("x", vec![1.0, 0.0, 0.0]).unwrap();
        let d2 = LatentDirection::new("y", vec![0.0, 1.0, 0.0]).unwrap();
        explorer.add_direction(d1).unwrap();
        explorer.add_direction(d2).unwrap();

        explorer.step(0, 3.0).unwrap();
        explorer.step(1, 4.0).unwrap();

        let current = explorer.current_latent();
        assert_relative_eq!(current[0], 3.0, epsilon = 1e-5);
        assert_relative_eq!(current[1], 4.0, epsilon = 1e-5);
        assert_relative_eq!(current[2], 0.0, epsilon = 1e-5);
    }
}
