//! FLAME parameter space sampler.
//!
//! Provides structured sampling strategies for the FLAME parameter space
//! (shape, expression, pose, translation) for training, testing, and data
//! generation.
//!
//! # Example
//!
//! ```rust
//! use oxigaf_flame::param_sampler::{FlameParamsSampler, SamplingStrategy};
//!
//! // Create a sampler with Latin Hypercube strategy
//! let sampler = FlameParamsSampler::lhs(42);
//! let samples = sampler.sample_batch(50);
//! assert_eq!(samples.len(), 50);
//! ```

use crate::error::FlameError;
use crate::params::FlameParams;

// ---------------------------------------------------------------------------
// Xorshift64 PRNG (no rand crate — Pure Rust policy)
// ---------------------------------------------------------------------------

/// Advance xorshift64 state and return the next pseudo-random u64.
///
/// Marsaglia's xorshift64 algorithm. State must be non-zero; the function
/// ensures this by substituting a fixed seed when state is zero.
#[inline]
fn xorshift64(state: &mut u64) -> u64 {
    // Guard against zero state (degenerate case for xorshift).
    if *state == 0 {
        *state = 0x853c_49e6_748f_ea9b;
    }
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// Generate a pseudo-random `f32` in `[0.0, 1.0)` from xorshift64 state.
#[inline]
fn rand_f32(state: &mut u64) -> f32 {
    // Use the top 23 bits (mantissa bits) for uniform float in [1,2), then
    // subtract 1.0 to get [0,1).  This avoids any division by large constants.
    let bits = xorshift64(state);
    // Take upper 23 bits, set exponent to 127 (value 1.0..2.0), subtract 1.0.
    let mantissa = (bits >> 41) as u32; // 23 bits
    let float_bits: u32 = 0x3f80_0000u32 | mantissa;
    f32::from_bits(float_bits) - 1.0_f32
}

/// Generate a pseudo-random `f32` in `[min, max)`.
#[inline]
fn rand_in_range(state: &mut u64, min: f32, max: f32) -> f32 {
    min + rand_f32(state) * (max - min)
}

// ---------------------------------------------------------------------------
// Van der Corput low-discrepancy sequence
// ---------------------------------------------------------------------------

/// Compute the n-th element of the Van der Corput sequence in the given base.
///
/// Returns a value in `[0.0, 1.0)`.
///
/// # Precondition
///
/// `base` must be `>= 2`; `base == 0` would panic (remainder by zero) and
/// `base == 1` would loop forever (`n % 1 == 0` and `n /= 1` never changes
/// `n`), so both are guarded here and return `0.0` instead.
#[must_use]
pub fn van_der_corput(n: usize, base: usize) -> f32 {
    if base < 2 {
        return 0.0;
    }
    let mut q = 0f64;
    let mut bk = 1.0 / base as f64;
    let mut n = n;
    while n > 0 {
        q += (n % base) as f64 * bk;
        n /= base;
        bk /= base as f64;
    }
    q as f32
}

// ---------------------------------------------------------------------------
// ParameterRange
// ---------------------------------------------------------------------------

/// Range `[min, max]` for a single scalar parameter.
#[derive(Debug, Clone, Copy)]
pub struct ParameterRange {
    /// Minimum value (inclusive).
    pub min: f32,
    /// Maximum value (inclusive).
    pub max: f32,
}

impl ParameterRange {
    /// Create a new `ParameterRange`.
    ///
    /// # Errors
    ///
    /// Returns [`FlameError::InvalidParams`] if `min >= max`.
    pub fn new(min: f32, max: f32) -> Result<Self, FlameError> {
        if min >= max {
            return Err(FlameError::InvalidParams(format!(
                "ParameterRange: min ({min}) must be strictly less than max ({max})"
            )));
        }
        Ok(Self { min, max })
    }

    /// Create a symmetric range `[-half_range, half_range]`.
    #[must_use]
    pub fn symmetric(half_range: f32) -> Self {
        Self {
            min: -half_range,
            max: half_range,
        }
    }

    /// Create the unit range `[0.0, 1.0]`.
    #[must_use]
    pub fn unit() -> Self {
        Self { min: 0.0, max: 1.0 }
    }

    /// Return `true` if `v` lies within `[min, max]`.
    #[must_use]
    pub fn contains(&self, v: f32) -> bool {
        v >= self.min && v <= self.max
    }

    /// Clamp `v` to `[min, max]`.
    #[must_use]
    pub fn clamp(&self, v: f32) -> f32 {
        v.clamp(self.min, self.max)
    }

    /// Linear interpolation: `min + t * (max - min)`.
    #[must_use]
    pub fn lerp(&self, t: f32) -> f32 {
        self.min + t * (self.max - self.min)
    }

    /// Width of the range.
    #[must_use]
    #[inline]
    pub fn width(self) -> f32 {
        self.max - self.min
    }
}

// ---------------------------------------------------------------------------
// ParameterDimension
// ---------------------------------------------------------------------------

/// A named parameter dimension with a range and a default value.
#[derive(Debug, Clone)]
pub struct ParameterDimension {
    /// Human-readable name of this dimension.
    pub name: String,
    /// The valid range for this dimension.
    pub range: ParameterRange,
    /// Default (neutral) value; defaults to the midpoint of `range`.
    pub default_value: f32,
}

impl ParameterDimension {
    /// Create a new `ParameterDimension`.
    ///
    /// The `default_value` is initialised to the midpoint `(min + max) / 2`.
    #[must_use]
    pub fn new(name: &str, range: ParameterRange) -> Self {
        let default_value = (range.min + range.max) * 0.5;
        Self {
            name: name.to_string(),
            range,
            default_value,
        }
    }

    /// Builder-pattern method to override `default_value`.
    #[must_use]
    pub fn with_default(mut self, default: f32) -> Self {
        self.default_value = default;
        self
    }
}

// ---------------------------------------------------------------------------
// SamplingStrategy
// ---------------------------------------------------------------------------

/// Strategy used to distribute samples across the parameter space.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SamplingStrategy {
    /// Uniformly random (independent) samples.
    Random,
    /// Uniform grid with specified resolution per dimension.
    Grid,
    /// Latin Hypercube Sampling — better space coverage for a given n.
    LatinHypercube,
    /// Sobol-like sequence using Van der Corput for low discrepancy.
    Sobol,
}

// ---------------------------------------------------------------------------
// FlameParamsSpace
// ---------------------------------------------------------------------------

/// Configuration that defines the FLAME parameter sampling space.
#[derive(Debug, Clone)]
pub struct FlameParamsSpace {
    /// Number of shape (identity) dimensions to sample.  Default: 10.
    pub shape_dims: usize,
    /// Number of expression dimensions to sample.  Default: 10.
    pub expression_dims: usize,
    /// Number of pose dimensions to sample (≤ 15).  Default: 12.
    pub pose_dims: usize,
    /// Number of translation dimensions (must be ≤ 3).  Default: 3.
    pub translation_dims: usize,
    /// Range for shape coefficients.
    pub shape_range: ParameterRange,
    /// Range for expression coefficients.
    pub expression_range: ParameterRange,
    /// Range for jaw angle (pose indices 6..9, only index 6 is opened widely).
    pub jaw_angle_range: ParameterRange,
    /// Range for global rotation (pose indices 0..3).
    pub global_rotation_range: ParameterRange,
    /// Range for neck & eye pose components (pose indices 3..6, 9..15).
    pub neck_eye_range: ParameterRange,
    /// Range for translation.
    pub translation_range: ParameterRange,
}

impl FlameParamsSpace {
    /// Default configuration matching typical FLAME parameter statistics.
    ///
    /// Shape/expression ±2.0, jaw `[0, 0.5]`, global rotation ±0.5, translation ±0.2.
    #[must_use]
    pub fn default_space() -> Self {
        Self {
            shape_dims: 10,
            expression_dims: 10,
            pose_dims: 12,
            translation_dims: 3,
            shape_range: ParameterRange::symmetric(2.0),
            expression_range: ParameterRange::symmetric(2.0),
            jaw_angle_range: ParameterRange { min: 0.0, max: 0.5 },
            global_rotation_range: ParameterRange::symmetric(0.5),
            neck_eye_range: ParameterRange::symmetric(0.3),
            translation_range: ParameterRange::symmetric(0.2),
        }
    }

    /// Strict (narrower) configuration for controlled evaluations.
    ///
    /// Shape ±1.5, expression ±1.5, jaw `[0, 0.3]`.
    #[must_use]
    pub fn strict() -> Self {
        Self {
            shape_dims: 10,
            expression_dims: 10,
            pose_dims: 12,
            translation_dims: 3,
            shape_range: ParameterRange::symmetric(1.5),
            expression_range: ParameterRange::symmetric(1.5),
            jaw_angle_range: ParameterRange { min: 0.0, max: 0.3 },
            global_rotation_range: ParameterRange::symmetric(0.3),
            neck_eye_range: ParameterRange::symmetric(0.15),
            translation_range: ParameterRange::symmetric(0.1),
        }
    }

    /// Expressive (wider) configuration for augmentation.
    ///
    /// Shape ±3.0, expression ±3.0, jaw `[0, 0.7]`.
    #[must_use]
    pub fn expressive() -> Self {
        Self {
            shape_dims: 10,
            expression_dims: 10,
            pose_dims: 12,
            translation_dims: 3,
            shape_range: ParameterRange::symmetric(3.0),
            expression_range: ParameterRange::symmetric(3.0),
            jaw_angle_range: ParameterRange { min: 0.0, max: 0.7 },
            global_rotation_range: ParameterRange::symmetric(0.8),
            neck_eye_range: ParameterRange::symmetric(0.4),
            translation_range: ParameterRange::symmetric(0.4),
        }
    }

    /// Total number of sampled dimensions.
    #[must_use]
    pub fn total_dims(&self) -> usize {
        self.shape_dims + self.expression_dims + self.pose_dims + self.translation_dims
    }

    /// Number of pose dimensions actually sampled from `pose_dims`.
    ///
    /// Pose is filled in 3-dimension blocks (global rotation, neck, jaw,
    /// left eye, right eye — see `sample_pose_random`/`sample_pose_grid`),
    /// so only 0, 3, 6, 9, 12, or 15 of a configured `pose_dims` are ever
    /// actually read: e.g. `pose_dims: 14` behaves exactly like
    /// `pose_dims: 12`. Grid sizing must use this (not raw `pose_dims`), or
    /// the unread dimensions get allocated grid resolution that produces
    /// duplicate samples.
    fn quantized_pose_dims(&self) -> usize {
        let pd = self.pose_dims;
        let mut consumed = 0;
        if pd >= 3 {
            consumed += 3;
        }
        if pd >= 6 {
            consumed += 3;
        }
        if pd >= 9 {
            consumed += 3;
        }
        if pd >= 12 {
            consumed += 3;
        }
        if pd == 15 {
            consumed += 3;
        }
        consumed
    }

    /// Validate the parameter space configuration.
    ///
    /// # Errors
    ///
    /// Returns [`FlameError::InvalidParams`] if any dimension count or range is invalid.
    pub fn validate(&self) -> Result<(), FlameError> {
        if self.shape_dims == 0 {
            return Err(FlameError::InvalidParams(
                "shape_dims must be > 0".to_string(),
            ));
        }
        if self.expression_dims == 0 {
            return Err(FlameError::InvalidParams(
                "expression_dims must be > 0".to_string(),
            ));
        }
        if self.pose_dims > FlameParams::NUM_JOINTS * 3 {
            return Err(FlameError::InvalidParams(format!(
                "pose_dims ({}) exceeds maximum ({})",
                self.pose_dims,
                FlameParams::NUM_JOINTS * 3
            )));
        }
        if self.translation_dims > 3 {
            return Err(FlameError::InvalidParams(format!(
                "translation_dims ({}) exceeds 3",
                self.translation_dims
            )));
        }
        // Validate ranges (min < max).
        let check = |r: &ParameterRange, name: &str| -> Result<(), FlameError> {
            if r.min >= r.max {
                Err(FlameError::InvalidParams(format!(
                    "{name} range is degenerate: min={} >= max={}",
                    r.min, r.max
                )))
            } else {
                Ok(())
            }
        };
        check(&self.shape_range, "shape_range")?;
        check(&self.expression_range, "expression_range")?;
        check(&self.jaw_angle_range, "jaw_angle_range")?;
        check(&self.global_rotation_range, "global_rotation_range")?;
        check(&self.neck_eye_range, "neck_eye_range")?;
        check(&self.translation_range, "translation_range")?;
        Ok(())
    }
}

impl Default for FlameParamsSpace {
    fn default() -> Self {
        Self::default_space()
    }
}

// ---------------------------------------------------------------------------
// FlameParamsSampler
// ---------------------------------------------------------------------------

/// Sampler that generates `FlameParams` instances from a defined parameter space.
pub struct FlameParamsSampler {
    /// The parameter space to sample from.
    pub space: FlameParamsSpace,
    /// Sampling strategy.
    pub strategy: SamplingStrategy,
    /// Seed for the internal xorshift64 PRNG.
    pub seed: u64,
}

impl FlameParamsSampler {
    /// Create a new sampler with the given space, strategy, and seed.
    #[must_use]
    pub fn new(space: FlameParamsSpace, strategy: SamplingStrategy, seed: u64) -> Self {
        Self {
            space,
            strategy,
            seed,
        }
    }

    /// Create a sampler with default space and [`SamplingStrategy::Random`].
    #[must_use]
    pub fn random(seed: u64) -> Self {
        Self::new(
            FlameParamsSpace::default_space(),
            SamplingStrategy::Random,
            seed,
        )
    }

    /// Create a sampler with default space and [`SamplingStrategy::LatinHypercube`].
    #[must_use]
    pub fn lhs(seed: u64) -> Self {
        Self::new(
            FlameParamsSpace::default_space(),
            SamplingStrategy::LatinHypercube,
            seed,
        )
    }

    /// Sample a single `FlameParams` from the parameter space.
    ///
    /// For [`SamplingStrategy::LatinHypercube`] and [`SamplingStrategy::Sobol`],
    /// single-sample semantics fall back to random (LHS/Sobol are batch concepts).
    #[must_use]
    pub fn sample_one(&self) -> FlameParams {
        let mut state = self.seed;
        self.sample_random_one(&mut state)
    }

    /// Sample a batch of `n` `FlameParams` instances.
    ///
    /// Strategy-specific behaviour:
    /// - **Random**: `n` independent uniform samples.
    /// - **Grid**: nearest feasible grid; take up to `n` points.
    /// - **`LatinHypercube`**: proper LHS over shape+expression dims, random for pose/translation.
    /// - **Sobol**: Van der Corput for first 2 dims, random for the rest.
    #[must_use]
    pub fn sample_batch(&self, n: usize) -> Vec<FlameParams> {
        if n == 0 {
            return Vec::new();
        }
        match self.strategy {
            SamplingStrategy::Random => self.sample_batch_random(n),
            SamplingStrategy::Grid => self.sample_batch_grid(n),
            SamplingStrategy::LatinHypercube => self.sample_batch_lhs(n),
            SamplingStrategy::Sobol => self.sample_batch_sobol(n),
        }
    }

    /// Generate `n` near-neutral poses.
    ///
    /// Shape parameters are random within ±0.3; expression and pose are zero;
    /// translation is zero.
    #[must_use]
    pub fn neutral_variants(n: usize) -> Vec<FlameParams> {
        let mut state: u64 = 0xdead_beef_cafe_babe;
        let mut out = Vec::with_capacity(n);
        let narrow = ParameterRange::symmetric(0.3);
        for _ in 0..n {
            let shape = (0..10)
                .map(|_| rand_in_range(&mut state, narrow.min, narrow.max))
                .collect();
            out.push(FlameParams {
                shape,
                expression: vec![0.0; 10],
                pose: vec![0.0; FlameParams::NUM_JOINTS * 3],
                translation: [0.0; 3],
            });
        }
        out
    }

    /// Sweep one expression dimension from its minimum to maximum.
    ///
    /// All other parameters are at their neutral (zero) value.
    ///
    /// # Panics
    ///
    /// Does not panic; if `expression_dim` is out of range the dimension is
    /// clamped to `expression_dims - 1`.
    #[must_use]
    pub fn expression_sweep(&self, expression_dim: usize, n: usize) -> Vec<FlameParams> {
        if n == 0 {
            return Vec::new();
        }
        let dim = expression_dim.min(self.space.expression_dims.saturating_sub(1));
        let range = self.space.expression_range;
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let t = if n == 1 {
                0.5
            } else {
                i as f32 / (n - 1) as f32
            };
            let value = range.lerp(t);
            let mut expression = vec![0.0f32; self.space.expression_dims];
            if self.space.expression_dims > 0 {
                expression[dim] = value;
            }
            out.push(FlameParams {
                shape: vec![0.0; self.space.shape_dims],
                expression,
                pose: vec![0.0; FlameParams::NUM_JOINTS * 3],
                translation: [0.0; 3],
            });
        }
        out
    }

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    /// Sample a single random `FlameParams`, advancing `state` in place.
    fn sample_random_one(&self, state: &mut u64) -> FlameParams {
        let sp = &self.space;

        // Shape coefficients.
        let shape = (0..sp.shape_dims)
            .map(|_| rand_in_range(state, sp.shape_range.min, sp.shape_range.max))
            .collect();

        // Expression coefficients.
        let expression = (0..sp.expression_dims)
            .map(|_| rand_in_range(state, sp.expression_range.min, sp.expression_range.max))
            .collect();

        // Pose — always produce a full 15-element vector (5 joints × 3).
        let pose = self.sample_pose_random(state);

        // Translation.
        let translation = [
            if sp.translation_dims > 0 {
                rand_in_range(state, sp.translation_range.min, sp.translation_range.max)
            } else {
                0.0
            },
            if sp.translation_dims > 1 {
                rand_in_range(state, sp.translation_range.min, sp.translation_range.max)
            } else {
                0.0
            },
            if sp.translation_dims > 2 {
                rand_in_range(state, sp.translation_range.min, sp.translation_range.max)
            } else {
                0.0
            },
        ];

        FlameParams {
            shape,
            expression,
            pose,
            translation,
        }
    }

    /// Sample a full 15-element pose vector according to the space's ranges.
    ///
    /// Layout (axis-angle, 3 per joint):
    /// - `[0..3]`   global rotation  → `global_rotation_range`
    /// - `[3..6]`   neck             → `neck_eye_range`
    /// - `[6..9]`   jaw              → `jaw_angle_range` for index 6; zeros for 7,8
    /// - `[9..12]`  left eye         → `neck_eye_range`
    /// - `[12..15]` right eye        → zeroed (not sampled when `pose_dims ≤ 12`)
    fn sample_pose_random(&self, state: &mut u64) -> Vec<f32> {
        let sp = &self.space;
        let mut pose = vec![0.0f32; FlameParams::NUM_JOINTS * 3];

        // Global rotation (indices 0..3).
        if sp.pose_dims >= 3 {
            for v in &mut pose[0..3] {
                *v = rand_in_range(
                    state,
                    sp.global_rotation_range.min,
                    sp.global_rotation_range.max,
                );
            }
        }
        // Neck (indices 3..6).
        if sp.pose_dims >= 6 {
            for v in &mut pose[3..6] {
                *v = rand_in_range(state, sp.neck_eye_range.min, sp.neck_eye_range.max);
            }
        }
        // Jaw (indices 6..9): primary opening axis is index 6.
        if sp.pose_dims >= 9 {
            pose[6] = rand_in_range(state, sp.jaw_angle_range.min, sp.jaw_angle_range.max);
            // Lateral jaw movement is small.
            pose[7] = rand_in_range(state, -0.05, 0.05);
            pose[8] = rand_in_range(state, -0.05, 0.05);
        }
        // Left eye (indices 9..12).
        if sp.pose_dims >= 12 {
            for v in &mut pose[9..12] {
                *v = rand_in_range(state, sp.neck_eye_range.min, sp.neck_eye_range.max);
            }
        }
        // Right eye (indices 12..15) — left at zero unless pose_dims == 15.
        if sp.pose_dims == 15 {
            for v in &mut pose[12..15] {
                *v = rand_in_range(state, sp.neck_eye_range.min, sp.neck_eye_range.max);
            }
        }

        pose
    }

    // ------------------------------------------------------------------
    // Random batch
    // ------------------------------------------------------------------

    fn sample_batch_random(&self, n: usize) -> Vec<FlameParams> {
        let mut state = self.seed;
        (0..n).map(|_| self.sample_random_one(&mut state)).collect()
    }

    // ------------------------------------------------------------------
    // Grid batch
    // ------------------------------------------------------------------

    fn sample_batch_grid(&self, n: usize) -> Vec<FlameParams> {
        let sp = &self.space;
        // Use the QUANTIZED pose dimension count, not raw `pose_dims`: pose
        // is only ever filled in 0/3/6/9/12/15-dim blocks (see
        // `quantized_pose_dims`), so sizing the grid on the raw count would
        // allocate grid resolution to dimensions nothing ever reads,
        // producing duplicate samples.
        let total =
            sp.shape_dims + sp.expression_dims + sp.quantized_pose_dims() + sp.translation_dims;
        if total == 0 {
            return vec![FlameParams::default(); n];
        }

        // Compute points-per-dimension for an approximately equal grid.
        // We want pts_per_dim^total_dims ≈ n.  Use cube-root heuristic:
        // find pts such that pts^total >= n.
        let pts_per_dim = {
            let raw = (n as f64).powf(1.0 / total as f64).ceil() as usize;
            raw.max(1)
        };

        // Generate grid coordinates in [0,1] for each dimension, then map to range.
        // We enumerate in a flat, lexicographic order.
        let mut out: Vec<FlameParams> = Vec::with_capacity(n);
        let mut grid_index: usize = 0;
        let total_grid = pts_per_dim.saturating_pow(total as u32);
        let step_size = if pts_per_dim == 1 {
            0.5 // midpoint for single-point grid
        } else {
            1.0 / (pts_per_dim - 1) as f64
        };

        let mut rng_state = self.seed;

        while out.len() < n && grid_index < total_grid {
            let mut remaining = grid_index;
            let mut dim_indices = vec![0usize; total];
            for idx_val in dim_indices.iter_mut().take(total) {
                *idx_val = remaining % pts_per_dim;
                remaining /= pts_per_dim;
            }

            // Map dim indices to parameter values.
            let t_for = |d: usize| -> f32 {
                if pts_per_dim == 1 {
                    0.5
                } else {
                    (dim_indices[d] as f64 * step_size) as f32
                }
            };

            let mut d = 0usize;

            // Shape.
            let shape: Vec<f32> = (0..sp.shape_dims)
                .map(|_| {
                    let v = sp.shape_range.lerp(t_for(d));
                    d += 1;
                    v
                })
                .collect();

            // Expression.
            let expression: Vec<f32> = (0..sp.expression_dims)
                .map(|_| {
                    let v = sp.expression_range.lerp(t_for(d));
                    d += 1;
                    v
                })
                .collect();

            // Pose (full 15-element vector; grid samples mapped to the relevant dims).
            // Advance `d` by what was ACTUALLY consumed (0/3/6/9/12/15), not
            // by raw `pose_dims` -- see `quantized_pose_dims`.
            let (pose, consumed) =
                self.sample_pose_grid(&dim_indices, d, pts_per_dim, &mut rng_state);
            d += consumed;

            // Translation.
            let tx = if d < total {
                sp.translation_range.lerp(t_for(d))
            } else {
                0.0
            };
            if d < total {
                d += 1;
            }
            let ty = if d < total {
                sp.translation_range.lerp(t_for(d))
            } else {
                0.0
            };
            if d < total {
                d += 1;
            }
            let tz = if d < total {
                sp.translation_range.lerp(t_for(d))
            } else {
                0.0
            };
            let _ = d;

            out.push(FlameParams {
                shape,
                expression,
                pose,
                translation: [tx, ty, tz],
            });

            grid_index += 1;
        }

        // If grid exhausted before n, pad with random samples.
        if out.len() < n {
            let extra = n - out.len();
            let mut extra_state = rng_state;
            for _ in 0..extra {
                out.push(self.sample_random_one(&mut extra_state));
            }
        }

        out
    }

    /// Build a 15-element pose vector using grid coordinates for the sampled dims.
    /// Returns `(pose, consumed)`, where `consumed` is the number of grid
    /// dimensions actually read starting at `base_dim` (0, 3, 6, 9, 12, or
    /// 15 — see `quantized_pose_dims`). Callers must advance their
    /// dimension cursor by `consumed`, NOT by raw `pose_dims`.
    fn sample_pose_grid(
        &self,
        dim_indices: &[usize],
        base_dim: usize,
        pts_per_dim: usize,
        rng_state: &mut u64,
    ) -> (Vec<f32>, usize) {
        let sp = &self.space;
        let mut pose = vec![0.0f32; FlameParams::NUM_JOINTS * 3];

        let t_of = |d: usize| -> f32 {
            if pts_per_dim == 1 {
                0.5
            } else {
                dim_indices[d] as f32 / (pts_per_dim - 1) as f32
            }
        };

        let mut d = base_dim;

        // Global rotation (0..3).
        if sp.pose_dims >= 3 {
            for v in &mut pose[0..3] {
                *v = sp.global_rotation_range.lerp(t_of(d));
                d += 1;
            }
        }
        // Neck (3..6).
        if sp.pose_dims >= 6 {
            for v in &mut pose[3..6] {
                *v = sp.neck_eye_range.lerp(t_of(d));
                d += 1;
            }
        }
        // Jaw (6..9).
        if sp.pose_dims >= 9 {
            pose[6] = sp.jaw_angle_range.lerp(t_of(d));
            d += 1;
            pose[7] = rand_in_range(rng_state, -0.05, 0.05);
            pose[8] = rand_in_range(rng_state, -0.05, 0.05);
            d += 2;
        }
        // Left eye (9..12).
        if sp.pose_dims >= 12 {
            for v in &mut pose[9..12] {
                *v = sp.neck_eye_range.lerp(t_of(d));
                d += 1;
            }
        }
        // Right eye (12..15).
        if sp.pose_dims == 15 {
            for v in &mut pose[12..15] {
                *v = sp.neck_eye_range.lerp(t_of(d));
                d += 1;
            }
        }
        let consumed = d - base_dim;

        (pose, consumed)
    }

    // ------------------------------------------------------------------
    // Latin Hypercube batch
    // ------------------------------------------------------------------

    fn sample_batch_lhs(&self, n: usize) -> Vec<FlameParams> {
        let sp = &self.space;
        let mut state = self.seed;

        // For each sampled dimension, create n strata and shuffle.
        // We do full LHS for shape + expression dims; pose and translation use random.

        let lhs_dims = sp.shape_dims + sp.expression_dims;

        // Generate LHS matrix: lhs_matrix[d][i] is the i-th sample in dimension d.
        let lhs_matrix: Vec<Vec<f32>> = (0..lhs_dims)
            .map(|d| {
                let range = if d < sp.shape_dims {
                    sp.shape_range
                } else {
                    sp.expression_range
                };
                // Create strata: each in [(i/n), (i+1)/n).
                let mut stratum_vals: Vec<f32> = (0..n)
                    .map(|i| {
                        let lo = i as f32 / n as f32;
                        let hi = (i + 1) as f32 / n as f32;
                        let u = lo + rand_f32(&mut state) * (hi - lo);
                        range.lerp(u)
                    })
                    .collect();
                // Shuffle (Fisher-Yates).
                for i in (1..n).rev() {
                    let j = (xorshift64(&mut state) as usize) % (i + 1);
                    stratum_vals.swap(i, j);
                }
                stratum_vals
            })
            .collect();

        // Assemble FlameParams from the matrix rows.
        (0..n)
            .map(|i| {
                let shape = (0..sp.shape_dims).map(|d| lhs_matrix[d][i]).collect();
                let expression = (0..sp.expression_dims)
                    .map(|d| lhs_matrix[sp.shape_dims + d][i])
                    .collect();
                let pose = self.sample_pose_random(&mut state);
                let translation = [
                    if sp.translation_dims > 0 {
                        rand_in_range(
                            &mut state,
                            sp.translation_range.min,
                            sp.translation_range.max,
                        )
                    } else {
                        0.0
                    },
                    if sp.translation_dims > 1 {
                        rand_in_range(
                            &mut state,
                            sp.translation_range.min,
                            sp.translation_range.max,
                        )
                    } else {
                        0.0
                    },
                    if sp.translation_dims > 2 {
                        rand_in_range(
                            &mut state,
                            sp.translation_range.min,
                            sp.translation_range.max,
                        )
                    } else {
                        0.0
                    },
                ];
                FlameParams {
                    shape,
                    expression,
                    pose,
                    translation,
                }
            })
            .collect()
    }

    // ------------------------------------------------------------------
    // Sobol-like batch (Van der Corput for first 2 dims)
    // ------------------------------------------------------------------

    fn sample_batch_sobol(&self, n: usize) -> Vec<FlameParams> {
        let sp = &self.space;
        let mut state = self.seed;

        (0..n)
            .map(|i| {
                // Van der Corput sequences for first two shape dimensions.
                let vdc0 = van_der_corput(i + 1, 2);
                let vdc1 = van_der_corput(i + 1, 3);

                let shape: Vec<f32> = (0..sp.shape_dims)
                    .map(|d| match d {
                        0 => sp.shape_range.lerp(vdc0),
                        1 => sp.shape_range.lerp(vdc1),
                        _ => rand_in_range(&mut state, sp.shape_range.min, sp.shape_range.max),
                    })
                    .collect();

                let expression: Vec<f32> = (0..sp.expression_dims)
                    .map(|_| {
                        rand_in_range(&mut state, sp.expression_range.min, sp.expression_range.max)
                    })
                    .collect();

                let pose = self.sample_pose_random(&mut state);

                let translation = [
                    if sp.translation_dims > 0 {
                        rand_in_range(
                            &mut state,
                            sp.translation_range.min,
                            sp.translation_range.max,
                        )
                    } else {
                        0.0
                    },
                    if sp.translation_dims > 1 {
                        rand_in_range(
                            &mut state,
                            sp.translation_range.min,
                            sp.translation_range.max,
                        )
                    } else {
                        0.0
                    },
                    if sp.translation_dims > 2 {
                        rand_in_range(
                            &mut state,
                            sp.translation_range.min,
                            sp.translation_range.max,
                        )
                    } else {
                        0.0
                    },
                ];

                FlameParams {
                    shape,
                    expression,
                    pose,
                    translation,
                }
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// SampleSetStats
// ---------------------------------------------------------------------------

/// Per-dimension statistics over a collection of `FlameParams` samples.
#[derive(Debug, Clone)]
pub struct SampleSetStats {
    /// Number of samples.
    pub n: usize,
    /// Mean value for each parameter dimension (flattened order).
    pub mean_per_dim: Vec<f32>,
    /// Standard deviation for each parameter dimension.
    pub std_per_dim: Vec<f32>,
    /// Minimum observed value per dimension.
    pub min_per_dim: Vec<f32>,
    /// Maximum observed value per dimension.
    pub max_per_dim: Vec<f32>,
}

/// Flatten a `FlameParams` into a single `Vec<f32>` in the order
/// `[shape..., expression..., pose..., translation...]`.
#[must_use]
pub fn flatten_params(p: &FlameParams) -> Vec<f32> {
    let mut v =
        Vec::with_capacity(p.shape.len() + p.expression.len() + p.pose.len() + p.translation.len());
    v.extend_from_slice(&p.shape);
    v.extend_from_slice(&p.expression);
    v.extend_from_slice(&p.pose);
    v.extend_from_slice(&p.translation);
    v
}

/// Compute per-dimension statistics across a slice of `FlameParams`.
///
/// Returns a [`SampleSetStats`] over all parameters flattened into a 1-D array.
/// If `samples` is empty, all statistics vectors will be empty.
#[must_use]
pub fn compute_sample_stats(samples: &[FlameParams]) -> SampleSetStats {
    let n = samples.len();
    if n == 0 {
        return SampleSetStats {
            n: 0,
            mean_per_dim: Vec::new(),
            std_per_dim: Vec::new(),
            min_per_dim: Vec::new(),
            max_per_dim: Vec::new(),
        };
    }

    // Determine dimensionality from first sample.
    let dims = flatten_params(&samples[0]).len();

    let mut sums = vec![0.0f64; dims];
    let mut min_per_dim = vec![f32::MAX; dims];
    let mut max_per_dim = vec![f32::MIN; dims];

    // First pass: sum, min, max.
    for p in samples {
        let flat = flatten_params(p);
        for (d, &v) in flat.iter().enumerate().take(dims) {
            sums[d] += f64::from(v);
            if v < min_per_dim[d] {
                min_per_dim[d] = v;
            }
            if v > max_per_dim[d] {
                max_per_dim[d] = v;
            }
        }
    }

    let mean_per_dim: Vec<f32> = sums.iter().map(|&s| (s / n as f64) as f32).collect();

    // Second pass: variance.
    let mut sum_sq = vec![0.0f64; dims];
    for p in samples {
        let flat = flatten_params(p);
        for (d, &v) in flat.iter().enumerate().take(dims) {
            let diff = f64::from(v) - sums[d] / n as f64;
            sum_sq[d] += diff * diff;
        }
    }

    let std_per_dim: Vec<f32> = sum_sq
        .iter()
        .map(|&ss| ((ss / n as f64).sqrt()) as f32)
        .collect();

    SampleSetStats {
        n,
        mean_per_dim,
        std_per_dim,
        min_per_dim,
        max_per_dim,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // ParameterRange
    // -----------------------------------------------------------------------

    #[test]
    fn test_parameter_range_new() {
        let r = ParameterRange::new(-1.0, 1.0).expect("valid range");
        assert!((r.min - (-1.0)).abs() < 1e-9);
        assert!((r.max - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_parameter_range_invalid() {
        assert!(
            ParameterRange::new(1.0, 0.0).is_err(),
            "min > max should fail"
        );
        assert!(
            ParameterRange::new(0.5, 0.5).is_err(),
            "min == max should fail"
        );
    }

    #[test]
    fn test_parameter_range_symmetric() {
        let r = ParameterRange::symmetric(2.0);
        assert!((r.min - (-2.0)).abs() < 1e-9);
        assert!((r.max - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_parameter_range_lerp() {
        let r = ParameterRange {
            min: 0.0,
            max: 10.0,
        };
        assert!((r.lerp(0.0) - 0.0).abs() < 1e-6, "lerp(0) == min");
        assert!((r.lerp(1.0) - 10.0).abs() < 1e-6, "lerp(1) == max");
        assert!((r.lerp(0.5) - 5.0).abs() < 1e-6, "lerp(0.5) == midpoint");
    }

    #[test]
    fn test_parameter_range_contains() {
        let r = ParameterRange {
            min: -1.0,
            max: 1.0,
        };
        assert!(r.contains(0.0));
        assert!(r.contains(-1.0));
        assert!(r.contains(1.0));
        assert!(!r.contains(1.1));
        assert!(!r.contains(-1.1));
    }

    // -----------------------------------------------------------------------
    // ParameterDimension
    // -----------------------------------------------------------------------

    #[test]
    fn test_parameter_dimension_default_midpoint() {
        let range = ParameterRange {
            min: -2.0,
            max: 4.0,
        };
        let dim = ParameterDimension::new("test_dim", range);
        let expected_mid = (-2.0_f32 + 4.0_f32) * 0.5;
        assert!(
            (dim.default_value - expected_mid).abs() < 1e-6,
            "default_value should be midpoint: got {} expected {}",
            dim.default_value,
            expected_mid
        );
    }

    // -----------------------------------------------------------------------
    // FlameParamsSpace
    // -----------------------------------------------------------------------

    #[test]
    fn test_flame_params_space_default() {
        let sp = FlameParamsSpace::default_space();
        assert_eq!(sp.shape_dims, 10);
        assert_eq!(sp.expression_dims, 10);
        assert_eq!(sp.pose_dims, 12);
        assert_eq!(sp.translation_dims, 3);
        assert!((sp.shape_range.min - (-2.0)).abs() < 1e-9);
        assert!((sp.shape_range.max - 2.0).abs() < 1e-9);
        assert!((sp.jaw_angle_range.min - 0.0).abs() < 1e-9);
        assert!((sp.jaw_angle_range.max - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_flame_params_space_validate() {
        let sp = FlameParamsSpace::default_space();
        assert!(sp.validate().is_ok(), "default space should be valid");

        // Invalid: pose_dims too large.
        let mut bad = sp.clone();
        bad.pose_dims = 100;
        assert!(bad.validate().is_err());

        // Invalid: translation_dims too large.
        let mut bad2 = sp.clone();
        bad2.translation_dims = 4;
        assert!(bad2.validate().is_err());

        // Invalid: shape_dims == 0.
        let mut bad3 = sp.clone();
        bad3.shape_dims = 0;
        assert!(bad3.validate().is_err());
    }

    // -----------------------------------------------------------------------
    // FlameParamsSampler construction
    // -----------------------------------------------------------------------

    #[test]
    fn test_sampler_new() {
        let sp = FlameParamsSpace::default_space();
        let sampler = FlameParamsSampler::new(sp, SamplingStrategy::Random, 42);
        assert_eq!(sampler.strategy, SamplingStrategy::Random);
        assert_eq!(sampler.seed, 42);
    }

    // -----------------------------------------------------------------------
    // sample_one
    // -----------------------------------------------------------------------

    #[test]
    fn test_sample_one_in_range() {
        let sampler = FlameParamsSampler::random(123);
        let sp = &sampler.space;
        let p = sampler.sample_one();

        // Shape in range.
        for &v in &p.shape {
            assert!(
                sp.shape_range.contains(v),
                "shape value {} out of range [{}, {}]",
                v,
                sp.shape_range.min,
                sp.shape_range.max
            );
        }
        // Expression in range.
        for &v in &p.expression {
            assert!(
                sp.expression_range.contains(v),
                "expression value {v} out of range"
            );
        }
        // Pose must be 15 elements.
        assert_eq!(p.pose.len(), FlameParams::NUM_JOINTS * 3);
        // Jaw opening (index 6) in jaw range.
        assert!(
            sp.jaw_angle_range.contains(p.pose[6]),
            "jaw angle {} out of range [{}, {}]",
            p.pose[6],
            sp.jaw_angle_range.min,
            sp.jaw_angle_range.max
        );
        // Translation in range.
        for &t in &p.translation {
            assert!(
                sp.translation_range.contains(t),
                "translation {t} out of range"
            );
        }
    }

    // -----------------------------------------------------------------------
    // sample_batch
    // -----------------------------------------------------------------------

    #[test]
    fn test_sample_batch_count() {
        let sampler = FlameParamsSampler::random(99);
        for n in [0, 1, 10, 100] {
            let batch = sampler.sample_batch(n);
            assert_eq!(batch.len(), n, "batch size mismatch for n={n}");
        }
    }

    #[test]
    fn test_sample_batch_unique() {
        // 100 samples should not all be identical.
        let sampler = FlameParamsSampler::random(7);
        let batch = sampler.sample_batch(100);
        assert_eq!(batch.len(), 100);

        // Check that at least two samples differ in the first shape coefficient.
        let first = batch[0].shape.first().copied().unwrap_or(0.0);
        let all_same = batch
            .iter()
            .all(|p| p.shape.first().copied().unwrap_or(0.0) == first);
        assert!(!all_same, "all 100 samples are identical — PRNG broken");
    }

    // -----------------------------------------------------------------------
    // LHS coverage
    // -----------------------------------------------------------------------

    #[test]
    fn test_lhs_batch_coverage() {
        // LHS should cover the first dimension with more uniform spacing than
        // pure random for small n. We check that the min/max spread of the
        // first shape dimension is close to the full range.
        let n = 20;
        let sampler = FlameParamsSampler::lhs(42);
        let batch = sampler.sample_batch(n);
        assert_eq!(batch.len(), n);

        let vals: Vec<f32> = batch
            .iter()
            .filter_map(|p| p.shape.first().copied())
            .collect();
        let min_val = vals.iter().copied().fold(f32::MAX, f32::min);
        let max_val = vals.iter().copied().fold(f32::MIN, f32::max);
        let spread = max_val - min_val;
        let expected = sampler.space.shape_range.width();

        // LHS should cover at least 80% of the range for n=20.
        assert!(
            spread >= 0.8 * expected,
            "LHS spread {spread} < 80% of expected range {expected}"
        );
    }

    // -----------------------------------------------------------------------
    // Grid batch
    // -----------------------------------------------------------------------

    #[test]
    fn test_grid_batch() {
        let sp = FlameParamsSpace::default_space();
        let sampler = FlameParamsSampler::new(sp, SamplingStrategy::Grid, 0);
        let batch = sampler.sample_batch(27);
        assert_eq!(
            batch.len(),
            27,
            "grid batch should return exactly n samples"
        );

        // All params should be within declared ranges.
        let sp = &sampler.space;
        for p in &batch {
            for &v in &p.shape {
                assert!(
                    sp.shape_range.contains(v),
                    "grid shape value {v} out of range"
                );
            }
        }
    }

    #[test]
    fn test_quantized_pose_dims_only_takes_3_6_9_12_15() {
        // pose is filled in 3-dim blocks, so anything between block
        // boundaries must quantize DOWN to the previous block.
        for (pose_dims, expected) in [
            (0, 0),
            (2, 0),
            (3, 3),
            (5, 3),
            (6, 6),
            (8, 6),
            (9, 9),
            (11, 9),
            (12, 12),
            (14, 12), // the finding's example: 14 behaves like 12
            (15, 15),
        ] {
            let mut sp = FlameParamsSpace::default_space();
            sp.pose_dims = pose_dims;
            assert_eq!(
                sp.quantized_pose_dims(),
                expected,
                "pose_dims={pose_dims} should quantize to {expected}"
            );
        }
    }

    #[test]
    fn test_sample_pose_grid_consumed_matches_quantized() {
        // Regression: the caller in `sample_batch_grid` must advance its
        // dimension cursor by what `sample_pose_grid` ACTUALLY consumed,
        // not by raw `pose_dims` -- otherwise unread grid dimensions get
        // allocated resolution that produces duplicate samples.
        let mut sp = FlameParamsSpace::default_space();
        sp.pose_dims = 14;
        let sampler = FlameParamsSampler::new(sp, SamplingStrategy::Grid, 0);
        let dim_indices = vec![0usize; 20];
        let mut state = 1u64;
        let (_pose, consumed) = sampler.sample_pose_grid(&dim_indices, 0, 2, &mut state);
        assert_eq!(
            consumed, 12,
            "pose_dims=14 must consume exactly 12 grid dimensions"
        );
    }

    // -----------------------------------------------------------------------
    // Neutral variants
    // -----------------------------------------------------------------------

    #[test]
    fn test_neutral_variants_near_neutral() {
        let variants = FlameParamsSampler::neutral_variants(20);
        assert_eq!(variants.len(), 20);

        for v in &variants {
            // Shape should be within ±0.3.
            for &s in &v.shape {
                assert!((-0.3..=0.3).contains(&s), "shape {s} outside ±0.3");
            }
            // Expression should be zero.
            for &e in &v.expression {
                assert!((e).abs() < 1e-9, "expression should be zero, got {e}");
            }
            // Pose should be zero.
            for &p in &v.pose {
                assert!((p).abs() < 1e-9, "pose should be zero, got {p}");
            }
            // Translation should be zero.
            for &t in &v.translation {
                assert!((t).abs() < 1e-9, "translation should be zero, got {t}");
            }
        }
    }

    // -----------------------------------------------------------------------
    // Expression sweep
    // -----------------------------------------------------------------------

    #[test]
    fn test_expression_sweep_count() {
        let sampler = FlameParamsSampler::random(0);
        let sweep = sampler.expression_sweep(0, 10);
        assert_eq!(sweep.len(), 10);
    }

    #[test]
    fn test_expression_sweep_range() {
        let sampler = FlameParamsSampler::random(0);
        let n = 11;
        let sweep = sampler.expression_sweep(0, n);
        assert_eq!(sweep.len(), n);

        let sp = &sampler.space;
        // First sample should be at min, last at max.
        let first_val = sweep.first().map_or(0.0, |p| p.expression[0]);
        let last_val = sweep.last().map_or(0.0, |p| p.expression[0]);
        assert!(
            (first_val - sp.expression_range.min).abs() < 1e-5,
            "sweep start {first_val} != expression min {}",
            sp.expression_range.min
        );
        assert!(
            (last_val - sp.expression_range.max).abs() < 1e-5,
            "sweep end {last_val} != expression max {}",
            sp.expression_range.max
        );

        // Non-swept dims should be zero.
        for p in &sweep {
            for (d, &v) in p.expression.iter().enumerate().skip(1) {
                assert!(
                    (v).abs() < 1e-9,
                    "expression dim {d} should be zero, got {v}"
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // compute_sample_stats
    // -----------------------------------------------------------------------

    #[test]
    fn test_compute_sample_stats() {
        // Two identical samples → std = 0, mean = value.
        let p = FlameParams {
            shape: vec![1.0, 2.0],
            expression: vec![0.5],
            pose: vec![0.0; 15],
            translation: [0.1, 0.2, 0.3],
        };
        let samples = vec![p.clone(), p.clone()];
        let stats = compute_sample_stats(&samples);

        assert_eq!(stats.n, 2);
        let flat = flatten_params(&p);
        assert_eq!(stats.mean_per_dim.len(), flat.len());

        for (d, (&m, &expected)) in stats.mean_per_dim.iter().zip(flat.iter()).enumerate() {
            assert!(
                (m - expected).abs() < 1e-5,
                "mean[{d}]: got {m} expected {expected}"
            );
        }
        for (d, &s) in stats.std_per_dim.iter().enumerate() {
            assert!(
                s.abs() < 1e-5,
                "std[{d}] should be 0 for identical samples, got {s}"
            );
        }

        // Empty input.
        let empty_stats = compute_sample_stats(&[]);
        assert_eq!(empty_stats.n, 0);
        assert!(empty_stats.mean_per_dim.is_empty());
    }

    // -----------------------------------------------------------------------
    // Van der Corput
    // -----------------------------------------------------------------------

    #[test]
    fn test_van_der_corput_base2() {
        // Known values: n=1 → 0.5, n=2 → 0.25, n=3 → 0.75, n=4 → 0.125.
        let eps = 1e-6_f32;
        assert!((van_der_corput(1, 2) - 0.5).abs() < eps, "vdc(1,2)");
        assert!((van_der_corput(2, 2) - 0.25).abs() < eps, "vdc(2,2)");
        assert!((van_der_corput(3, 2) - 0.75).abs() < eps, "vdc(3,2)");
        assert!((van_der_corput(4, 2) - 0.125).abs() < eps, "vdc(4,2)");
    }

    #[test]
    fn test_van_der_corput_base_0_returns_promptly() {
        // base=0 must not panic (remainder by zero) -- guarded, returns 0.0.
        assert_eq!(van_der_corput(5, 0), 0.0);
    }

    #[test]
    fn test_van_der_corput_base_1_returns_promptly() {
        // base=1 must not loop forever (n%1==0, n/=1 never changes n) --
        // guarded, returns 0.0. This test itself is the regression check:
        // it hangs under the pre-fix implementation instead of completing.
        assert_eq!(van_der_corput(5, 1), 0.0);
    }

    // -----------------------------------------------------------------------
    // Sobol-like batch stays in range
    // -----------------------------------------------------------------------

    #[test]
    fn test_sobol_batch_in_range() {
        let sp = FlameParamsSpace::default_space();
        let sampler = FlameParamsSampler::new(sp, SamplingStrategy::Sobol, 17);
        let batch = sampler.sample_batch(50);
        assert_eq!(batch.len(), 50);

        let sp = &sampler.space;
        for p in &batch {
            for &v in &p.shape {
                assert!(sp.shape_range.contains(v), "sobol shape {v} out of range");
            }
            for &v in &p.expression {
                assert!(
                    sp.expression_range.contains(v),
                    "sobol expression {v} out of range"
                );
            }
        }
    }
}
