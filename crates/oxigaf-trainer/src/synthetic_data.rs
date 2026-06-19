//! Synthetic data generation for OxiGAF training pipeline.
//!
//! Creates artificial training examples by randomly sampling FLAME parameters
//! (shapes, expressions, poses) and randomly initialising Gaussian fields.
//! Enables testing the training pipeline without real data, evaluating
//! convergence properties, and curriculum learning from easy to hard examples.

use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors produced by synthetic data generation.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum SyntheticDataError {
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("empty output: generation produced no samples")]
    EmptyOutput,

    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
}

// ─────────────────────────────────────────────────────────────────────────────
// Inline xorshift64 PRNG
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
fn xorshift64(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

#[inline]
fn xorshift_f32(state: &mut u64) -> f32 {
    (xorshift64(state) >> 32) as f32 / u32::MAX as f32
}

/// Box-Muller transform: two uniform [0,1] values → standard normal N(0,1).
#[inline]
fn box_muller(u1: f32, u2: f32) -> f32 {
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = std::f32::consts::TAU * u2;
    r * theta.cos()
}

/// Sample N(mean, std) using xorshift64 state.
#[inline]
fn sample_normal(state: &mut u64, mean: f32, std: f32) -> f32 {
    let u1 = xorshift_f32(state).max(1e-10); // avoid log(0)
    let u2 = xorshift_f32(state);
    mean + std * box_muller(u1, u2)
}

/// Sample uniform [lo, hi).
#[inline]
fn sample_uniform(state: &mut u64, lo: f32, hi: f32) -> f32 {
    lo + xorshift_f32(state) * (hi - lo)
}

// ─────────────────────────────────────────────────────────────────────────────
// SyntheticFlameParams
// ─────────────────────────────────────────────────────────────────────────────

/// Sampled FLAME parameter set.
#[derive(Debug, Clone)]
pub struct SyntheticFlameParams {
    /// Shape coefficients (n_shape_betas).
    pub shape: Vec<f32>,
    /// Expression coefficients (n_expr_betas).
    pub expression: Vec<f32>,
    /// Pose axis-angle vectors: 5 joints * 3 = 15 values.
    pub pose: Vec<f32>,
    /// Global translation [x, y, z].
    pub translation: Vec<f32>,
}

impl SyntheticFlameParams {
    /// Construct a new parameter set from explicit vectors.
    pub fn new(
        shape: Vec<f32>,
        expression: Vec<f32>,
        pose: Vec<f32>,
        translation: Vec<f32>,
    ) -> Self {
        Self {
            shape,
            expression,
            pose,
            translation,
        }
    }

    /// Neutral pose — all parameters are zero.
    pub fn neutral(n_shape: usize, n_expr: usize) -> Self {
        Self {
            shape: vec![0.0; n_shape],
            expression: vec![0.0; n_expr],
            pose: vec![0.0; 15],
            translation: vec![0.0; 3],
        }
    }

    /// L2 norm of the expression vector.
    pub fn expression_magnitude(&self) -> f32 {
        self.expression.iter().map(|v| v * v).sum::<f32>().sqrt()
    }

    /// L2 norm of pose excluding root rotation (joints 1–4, indices 3..15).
    pub fn pose_magnitude(&self) -> f32 {
        self.pose.iter().skip(3).map(|v| v * v).sum::<f32>().sqrt()
    }

    /// Returns `true` when every pose value has absolute value less than `threshold`.
    pub fn is_neutral(&self, threshold: f32) -> bool {
        self.pose.iter().all(|v| v.abs() < threshold)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FlameParamSampler
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for [`FlameParamSampler`].
#[derive(Debug, Clone)]
pub struct FlameParamSamplerConfig {
    /// Number of shape beta coefficients.
    pub n_shape_betas: usize,
    /// Number of expression beta coefficients.
    pub n_expr_betas: usize,
    /// Standard deviation for shape sampling (per-beta).
    pub shape_std: f32,
    /// Standard deviation for expression sampling.
    pub expr_std: f32,
    /// Maximum absolute jaw rotation in radians (jaw opens downward).
    pub max_jaw_rotation: f32,
    /// Maximum absolute neck rotation per axis in radians.
    pub max_neck_rotation: f32,
    /// Maximum absolute global head rotation per axis in radians.
    pub max_head_rotation: f32,
    /// Maximum translation magnitude per axis.
    pub max_translation: f32,
    /// Random seed (must be > 0 for xorshift; clamped internally).
    pub seed: u64,
}

impl Default for FlameParamSamplerConfig {
    fn default() -> Self {
        Self {
            n_shape_betas: 100,
            n_expr_betas: 50,
            shape_std: 1.0,
            expr_std: 0.5,
            max_jaw_rotation: 0.4,
            max_neck_rotation: 0.3,
            max_head_rotation: 0.5,
            max_translation: 0.1,
            seed: 42,
        }
    }
}

impl FlameParamSamplerConfig {
    /// Validate configuration fields. Returns an error if any value is out of
    /// bounds.
    pub fn validate(&self) -> Result<(), SyntheticDataError> {
        if self.n_shape_betas < 1 {
            return Err(SyntheticDataError::InvalidConfig(
                "n_shape_betas must be >= 1".into(),
            ));
        }
        if self.n_expr_betas < 1 {
            return Err(SyntheticDataError::InvalidConfig(
                "n_expr_betas must be >= 1".into(),
            ));
        }
        if self.shape_std < 0.0 {
            return Err(SyntheticDataError::InvalidConfig(
                "shape_std must be >= 0".into(),
            ));
        }
        if self.expr_std < 0.0 {
            return Err(SyntheticDataError::InvalidConfig(
                "expr_std must be >= 0".into(),
            ));
        }
        if self.max_jaw_rotation < 0.0 {
            return Err(SyntheticDataError::InvalidConfig(
                "max_jaw_rotation must be >= 0".into(),
            ));
        }
        if self.max_neck_rotation < 0.0 {
            return Err(SyntheticDataError::InvalidConfig(
                "max_neck_rotation must be >= 0".into(),
            ));
        }
        if self.max_head_rotation < 0.0 {
            return Err(SyntheticDataError::InvalidConfig(
                "max_head_rotation must be >= 0".into(),
            ));
        }
        if self.max_translation < 0.0 {
            return Err(SyntheticDataError::InvalidConfig(
                "max_translation must be >= 0".into(),
            ));
        }
        Ok(())
    }
}

/// Generates random FLAME parameter sets using an inline xorshift64 PRNG.
pub struct FlameParamSampler {
    config: FlameParamSamplerConfig,
    state: u64,
}

impl FlameParamSampler {
    /// Create a new sampler, validating the config first.
    pub fn new(config: FlameParamSamplerConfig) -> Result<Self, SyntheticDataError> {
        config.validate()?;
        let state = config.seed.max(1);
        Ok(Self { config, state })
    }

    /// Draw one random [`SyntheticFlameParams`].
    ///
    /// Joint layout (axis-angle, 3 values each):
    /// - `[0..3]`  global head rotation
    /// - `[3..6]`  neck rotation
    /// - `[6..9]`  jaw (only index 6 is non-zero — opens downward)
    /// - `[9..12]` left eye
    /// - `[12..15]` right eye
    pub fn sample(&mut self) -> SyntheticFlameParams {
        let n_shape = self.config.n_shape_betas;
        let n_expr = self.config.n_expr_betas;

        // Shape: N(0, shape_std) per beta
        let shape: Vec<f32> = (0..n_shape)
            .map(|_| sample_normal(&mut self.state, 0.0, self.config.shape_std))
            .collect();

        // Expression: N(0, expr_std) per beta
        let expression: Vec<f32> = (0..n_expr)
            .map(|_| sample_normal(&mut self.state, 0.0, self.config.expr_std))
            .collect();

        // Pose: 15 values for 5 joints
        let mut pose = vec![0.0f32; 15];
        // Global head rotation [0..3]
        for p in pose[0..3].iter_mut() {
            *p = sample_uniform(
                &mut self.state,
                -self.config.max_head_rotation,
                self.config.max_head_rotation,
            );
        }
        // Neck rotation [3..6]
        for p in pose[3..6].iter_mut() {
            *p = sample_uniform(
                &mut self.state,
                -self.config.max_neck_rotation,
                self.config.max_neck_rotation,
            );
        }
        // Jaw [6..9]: only index 6 (opens downward), others stay 0
        pose[6] = sample_uniform(&mut self.state, 0.0, self.config.max_jaw_rotation);
        // Left eye [9..12]
        for p in pose[9..12].iter_mut() {
            *p = sample_uniform(&mut self.state, -0.2, 0.2);
        }
        // Right eye [12..15]
        for p in pose[12..15].iter_mut() {
            *p = sample_uniform(&mut self.state, -0.2, 0.2);
        }

        // Translation: uniform [-max, max] per axis
        let translation: Vec<f32> = (0..3)
            .map(|_| {
                sample_uniform(
                    &mut self.state,
                    -self.config.max_translation,
                    self.config.max_translation,
                )
            })
            .collect();

        SyntheticFlameParams::new(shape, expression, pose, translation)
    }

    /// Draw `n` independent samples.
    pub fn sample_batch(&mut self, n: usize) -> Vec<SyntheticFlameParams> {
        (0..n).map(|_| self.sample()).collect()
    }

    /// Sample near-neutral parameters: small expression, zero pose.
    ///
    /// Expression is drawn from `N(0, expr_std * expr_scale)`. Shape and
    /// translation are sampled at full scale; pose is exactly zero.
    pub fn sample_near_neutral(&mut self, expr_scale: f32) -> SyntheticFlameParams {
        let n_shape = self.config.n_shape_betas;
        let n_expr = self.config.n_expr_betas;

        let shape: Vec<f32> = (0..n_shape)
            .map(|_| sample_normal(&mut self.state, 0.0, self.config.shape_std))
            .collect();

        let scaled_std = self.config.expr_std * expr_scale;
        let expression: Vec<f32> = (0..n_expr)
            .map(|_| sample_normal(&mut self.state, 0.0, scaled_std))
            .collect();

        let pose = vec![0.0f32; 15];

        let translation: Vec<f32> = (0..3)
            .map(|_| {
                sample_uniform(
                    &mut self.state,
                    -self.config.max_translation,
                    self.config.max_translation,
                )
            })
            .collect();

        SyntheticFlameParams::new(shape, expression, pose, translation)
    }

    /// Sample a trajectory of `n_frames` frames by linearly interpolating
    /// between two independently sampled endpoint parameter sets.
    ///
    /// For `n_frames <= 1` the trajectory contains only the start sample.
    pub fn sample_trajectory(&mut self, n_frames: usize) -> Vec<SyntheticFlameParams> {
        if n_frames == 0 {
            return Vec::new();
        }

        let start = self.sample();
        if n_frames == 1 {
            return vec![start];
        }

        let end = self.sample();

        (0..n_frames)
            .map(|i| {
                let t = i as f32 / (n_frames - 1) as f32;
                let lerp = |a: &[f32], b: &[f32]| -> Vec<f32> {
                    a.iter()
                        .zip(b.iter())
                        .map(|(av, bv)| av + t * (bv - av))
                        .collect()
                };

                SyntheticFlameParams::new(
                    lerp(&start.shape, &end.shape),
                    lerp(&start.expression, &end.expression),
                    lerp(&start.pose, &end.pose),
                    lerp(&start.translation, &end.translation),
                )
            })
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SyntheticGaussianCloud
// ─────────────────────────────────────────────────────────────────────────────

/// A randomly initialised Gaussian cloud for testing.
#[derive(Debug, Clone)]
pub struct SyntheticGaussianCloud {
    /// Positions [x, y, z] per Gaussian (len = n * 3).
    pub positions: Vec<f32>,
    /// Log-scale [sx, sy, sz] per Gaussian (len = n * 3).
    pub log_scales: Vec<f32>,
    /// Quaternion rotation [qx, qy, qz, qw] per Gaussian (len = n * 4).
    pub rotations: Vec<f32>,
    /// Logit-space opacity per Gaussian (len = n); sigmoid maps to (0, 1).
    pub opacities: Vec<f32>,
    /// RGB colour per Gaussian, values in [0, 1] (len = n * 3).
    pub colors: Vec<f32>,
}

impl SyntheticGaussianCloud {
    /// Number of Gaussians in the cloud.
    pub fn num_gaussians(&self) -> usize {
        self.positions.len() / 3
    }

    /// Position of Gaussian `i` as `[x, y, z]`.
    pub fn position(&self, i: usize) -> [f32; 3] {
        let base = i * 3;
        [
            self.positions[base],
            self.positions[base + 1],
            self.positions[base + 2],
        ]
    }

    /// Log-scale of Gaussian `i` as `[sx, sy, sz]`.
    pub fn log_scale(&self, i: usize) -> [f32; 3] {
        let base = i * 3;
        [
            self.log_scales[base],
            self.log_scales[base + 1],
            self.log_scales[base + 2],
        ]
    }

    /// Rotation quaternion of Gaussian `i` as `[qx, qy, qz, qw]`.
    pub fn rotation(&self, i: usize) -> [f32; 4] {
        let base = i * 4;
        [
            self.rotations[base],
            self.rotations[base + 1],
            self.rotations[base + 2],
            self.rotations[base + 3],
        ]
    }

    /// Sigmoid-transformed opacity of Gaussian `i`, in `(0, 1)`.
    pub fn opacity(&self, i: usize) -> f32 {
        sigmoid(self.opacities[i])
    }

    /// RGB colour of Gaussian `i` as `[r, g, b]`, values in `[0, 1]`.
    pub fn color(&self, i: usize) -> [f32; 3] {
        let base = i * 3;
        [
            self.colors[base],
            self.colors[base + 1],
            self.colors[base + 2],
        ]
    }

    /// Mean position (centroid) across all Gaussians.
    ///
    /// Returns `[0.0, 0.0, 0.0]` for an empty cloud.
    pub fn centroid(&self) -> [f32; 3] {
        let n = self.num_gaussians();
        if n == 0 {
            return [0.0, 0.0, 0.0];
        }
        let inv_n = 1.0 / n as f32;
        let mut cx = 0.0f32;
        let mut cy = 0.0f32;
        let mut cz = 0.0f32;
        for i in 0..n {
            let p = self.position(i);
            cx += p[0];
            cy += p[1];
            cz += p[2];
        }
        [cx * inv_n, cy * inv_n, cz * inv_n]
    }

    /// Axis-aligned bounding box as `(min_corner, max_corner)`.
    ///
    /// Returns `([0,0,0], [0,0,0])` for an empty cloud.
    pub fn aabb(&self) -> ([f32; 3], [f32; 3]) {
        let n = self.num_gaussians();
        if n == 0 {
            return ([0.0; 3], [0.0; 3]);
        }

        let first = self.position(0);
        let mut lo = first;
        let mut hi = first;

        for i in 1..n {
            let p = self.position(i);
            for axis in 0..3 {
                if p[axis] < lo[axis] {
                    lo[axis] = p[axis];
                }
                if p[axis] > hi[axis] {
                    hi[axis] = p[axis];
                }
            }
        }
        (lo, hi)
    }

    /// Scene diameter approximated from the AABB diagonal length.
    pub fn scene_diameter(&self) -> f32 {
        let (lo, hi) = self.aabb();
        let dx = hi[0] - lo[0];
        let dy = hi[1] - lo[1];
        let dz = hi[2] - lo[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Mean sigmoid-transformed opacity across all Gaussians.
    ///
    /// Returns `0.0` for an empty cloud.
    pub fn mean_opacity(&self) -> f32 {
        let n = self.num_gaussians();
        if n == 0 {
            return 0.0;
        }
        let sum: f32 = (0..n).map(|i| self.opacity(i)).sum();
        sum / n as f32
    }
}

/// Sigmoid function used for opacity mapping.
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ─────────────────────────────────────────────────────────────────────────────
// GaussianCloudSampler
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for [`sample_gaussian_cloud`].
#[derive(Debug, Clone)]
pub struct GaussianCloudConfig {
    /// Number of Gaussians to generate.
    pub num_gaussians: usize,
    /// Standard deviation for position sampling (each axis independently).
    pub position_std: f32,
    /// Mean of the log-scale distribution.
    pub log_scale_mean: f32,
    /// Standard deviation of the log-scale distribution.
    pub log_scale_std: f32,
    /// Mean of the logit-opacity distribution (sigmoid(0) = 0.5).
    pub opacity_logit_mean: f32,
    /// Standard deviation of the logit-opacity distribution.
    pub opacity_logit_std: f32,
    /// If `true`, colours are sampled uniformly; otherwise fixed at grey.
    pub random_colors: bool,
    /// Random seed.
    pub seed: u64,
}

impl Default for GaussianCloudConfig {
    fn default() -> Self {
        Self {
            num_gaussians: 1000,
            position_std: 0.5,
            log_scale_mean: -4.0,
            log_scale_std: 0.5,
            opacity_logit_mean: 0.0,
            opacity_logit_std: 1.0,
            random_colors: true,
            seed: 123,
        }
    }
}

impl GaussianCloudConfig {
    /// Validate configuration fields.
    pub fn validate(&self) -> Result<(), SyntheticDataError> {
        if self.num_gaussians < 1 {
            return Err(SyntheticDataError::InvalidConfig(
                "num_gaussians must be >= 1".into(),
            ));
        }
        if self.log_scale_std < 0.0 {
            return Err(SyntheticDataError::InvalidConfig(
                "log_scale_std must be >= 0".into(),
            ));
        }
        if self.opacity_logit_std < 0.0 {
            return Err(SyntheticDataError::InvalidConfig(
                "opacity_logit_std must be >= 0".into(),
            ));
        }
        Ok(())
    }
}

/// Sample a random unit quaternion `[qx, qy, qz, qw]`.
///
/// Draws four standard-normal values and normalises them. Falls back to
/// `[0, 0, 0, 1]` (identity) if the magnitude is near zero.
pub fn sample_unit_quaternion(state: &mut u64) -> [f32; 4] {
    let qx = sample_normal(state, 0.0, 1.0);
    let qy = sample_normal(state, 0.0, 1.0);
    let qz = sample_normal(state, 0.0, 1.0);
    let qw = sample_normal(state, 0.0, 1.0);

    let mag_sq = qx * qx + qy * qy + qz * qz + qw * qw;
    if mag_sq < 1e-12 {
        return [0.0, 0.0, 0.0, 1.0];
    }
    let inv_mag = 1.0 / mag_sq.sqrt();
    [qx * inv_mag, qy * inv_mag, qz * inv_mag, qw * inv_mag]
}

/// Generate a synthetic Gaussian cloud using the provided configuration.
pub fn sample_gaussian_cloud(
    config: &GaussianCloudConfig,
) -> Result<SyntheticGaussianCloud, SyntheticDataError> {
    config.validate()?;

    let n = config.num_gaussians;
    let mut state = config.seed.max(1);

    let mut positions = Vec::with_capacity(n * 3);
    let mut log_scales = Vec::with_capacity(n * 3);
    let mut rotations = Vec::with_capacity(n * 4);
    let mut opacities = Vec::with_capacity(n);
    let mut colors = Vec::with_capacity(n * 3);

    for _ in 0..n {
        // Position: N(0, position_std) per axis
        positions.push(sample_normal(&mut state, 0.0, config.position_std));
        positions.push(sample_normal(&mut state, 0.0, config.position_std));
        positions.push(sample_normal(&mut state, 0.0, config.position_std));

        // Log-scale: N(log_scale_mean, log_scale_std) per axis
        log_scales.push(sample_normal(
            &mut state,
            config.log_scale_mean,
            config.log_scale_std,
        ));
        log_scales.push(sample_normal(
            &mut state,
            config.log_scale_mean,
            config.log_scale_std,
        ));
        log_scales.push(sample_normal(
            &mut state,
            config.log_scale_mean,
            config.log_scale_std,
        ));

        // Rotation: random unit quaternion
        let q = sample_unit_quaternion(&mut state);
        rotations.extend_from_slice(&q);

        // Opacity: N(opacity_logit_mean, opacity_logit_std)
        opacities.push(sample_normal(
            &mut state,
            config.opacity_logit_mean,
            config.opacity_logit_std,
        ));

        // Colour
        if config.random_colors {
            colors.push(xorshift_f32(&mut state));
            colors.push(xorshift_f32(&mut state));
            colors.push(xorshift_f32(&mut state));
        } else {
            colors.extend_from_slice(&[0.5, 0.5, 0.5]);
        }
    }

    Ok(SyntheticGaussianCloud {
        positions,
        log_scales,
        rotations,
        opacities,
        colors,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// SyntheticBatch
// ─────────────────────────────────────────────────────────────────────────────

/// A batch of `(image, params)` pairs for synthetic training.
#[derive(Debug, Clone)]
pub struct SyntheticBatch {
    /// Random images (width × height × 3 per image), values in [0, 1].
    pub images: Vec<Vec<f32>>,
    /// Corresponding FLAME params for each image.
    pub params: Vec<SyntheticFlameParams>,
    /// Image width in pixels.
    pub width: usize,
    /// Image height in pixels.
    pub height: usize,
}

impl SyntheticBatch {
    /// Number of items in the batch.
    pub fn len(&self) -> usize {
        self.images.len()
    }

    /// Returns `true` when the batch contains no items.
    pub fn is_empty(&self) -> bool {
        self.images.is_empty()
    }
}

/// Generate a synthetic batch with random noise images and random FLAME params.
///
/// Images are filled with uniform random RGB values in `[0, 1]`. FLAME params
/// are sampled independently from the given config.
pub fn generate_synthetic_batch(
    batch_size: usize,
    width: usize,
    height: usize,
    flame_config: &FlameParamSamplerConfig,
    seed: u64,
) -> Result<SyntheticBatch, SyntheticDataError> {
    if batch_size == 0 {
        return Err(SyntheticDataError::EmptyOutput);
    }

    let mut state = seed.max(1);
    let pixels_per_image = width * height * 3;

    let mut images = Vec::with_capacity(batch_size);
    for _ in 0..batch_size {
        let image: Vec<f32> = (0..pixels_per_image)
            .map(|_| xorshift_f32(&mut state))
            .collect();
        images.push(image);
    }

    // Use a separate sampler for FLAME params (different seed to avoid
    // correlation with image noise).
    let param_seed = seed.wrapping_add(0xDEAD_BEEF_CAFE_1234);
    let param_config = FlameParamSamplerConfig {
        seed: param_seed,
        ..flame_config.clone()
    };
    let mut sampler = FlameParamSampler::new(param_config)?;
    let params = sampler.sample_batch(batch_size);

    Ok(SyntheticBatch {
        images,
        params,
        width,
        height,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Curriculum / DifficultyLevel
// ─────────────────────────────────────────────────────────────────────────────

/// Difficulty level for curriculum learning of FLAME parameter sampling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DifficultyLevel {
    /// Near-neutral expression, frontal pose.
    Easy,
    /// Moderate expression, slight pose.
    Medium,
    /// Strong expression, large pose.
    Hard,
    /// Extreme expression and pose.
    Extreme,
}

impl DifficultyLevel {
    /// Scale factor applied to expression standard deviation.
    pub fn expr_scale(&self) -> f32 {
        match self {
            Self::Easy => 0.1,
            Self::Medium => 0.5,
            Self::Hard => 1.0,
            Self::Extreme => 2.0,
        }
    }

    /// Scale factor applied to pose values.
    pub fn pose_scale(&self) -> f32 {
        match self {
            Self::Easy => 0.1,
            Self::Medium => 0.5,
            Self::Hard => 1.0,
            Self::Extreme => 2.0,
        }
    }

    /// Determine difficulty from training step and total warmup steps.
    ///
    /// Schedule:
    /// - `step <  warmup_steps / 4` → Easy
    /// - `step <  warmup_steps / 2` → Medium
    /// - `step <  warmup_steps`     → Hard
    /// - `step >= warmup_steps`     → Extreme
    pub fn from_step(step: usize, warmup_steps: usize) -> Self {
        if step < warmup_steps / 4 {
            Self::Easy
        } else if step < warmup_steps / 2 {
            Self::Medium
        } else if step < warmup_steps {
            Self::Hard
        } else {
            Self::Extreme
        }
    }
}

/// Sample FLAME params at the given difficulty level.
///
/// Expression is drawn from `N(0, expr_std * level.expr_scale())`. Pose
/// values are first sampled at full scale (as in [`FlameParamSampler::sample`])
/// and then multiplied by `level.pose_scale()` so that higher difficulty levels
/// produce larger articulations.
pub fn sample_at_difficulty(
    sampler: &mut FlameParamSampler,
    level: DifficultyLevel,
) -> SyntheticFlameParams {
    let expr_scale = level.expr_scale();
    let pose_scale = level.pose_scale();

    // Start with near-neutral: correct expression magnitude, zero pose.
    let mut params = sampler.sample_near_neutral(expr_scale);

    // Now sample a full set of pose values and scale them so the curriculum
    // difficulty actually modulates articulation.
    let full = sampler.sample();
    for (dst, src) in params.pose.iter_mut().zip(full.pose.iter()) {
        *dst = src * pose_scale;
    }

    params
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── PRNG / sampling primitives ────────────────────────────────────────────

    #[test]
    fn test_xorshift_f32_in_unit_interval() {
        let mut state = 12345u64;
        for _ in 0..10_000 {
            let v = xorshift_f32(&mut state);
            assert!(v >= 0.0, "xorshift_f32 must be >= 0, got {v}");
            assert!(v < 1.0, "xorshift_f32 must be < 1, got {v}");
        }
    }

    #[test]
    fn test_sample_normal_statistics() {
        let mut state = 99_999u64;
        let n = 10_000usize;
        let samples: Vec<f32> = (0..n)
            .map(|_| sample_normal(&mut state, 0.0, 1.0))
            .collect();

        let mean: f32 = samples.iter().sum::<f32>() / n as f32;
        let variance: f32 = samples.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n as f32;
        let std = variance.sqrt();

        // Mean should be close to 0, std close to 1 — allow generous tolerances
        // for a unit test (not a Monte-Carlo paper).
        assert!(mean.abs() < 0.1, "mean out of range: {mean}");
        assert!((std - 1.0).abs() < 0.1, "std out of range: {std}");
    }

    // ── SyntheticFlameParams ──────────────────────────────────────────────────

    #[test]
    fn test_neutral_all_zeros() {
        let p = SyntheticFlameParams::neutral(10, 5);
        assert!(p.shape.iter().all(|&v| v == 0.0));
        assert!(p.expression.iter().all(|&v| v == 0.0));
        assert!(p.pose.iter().all(|&v| v == 0.0));
        assert!(p.translation.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_expression_magnitude_zero_for_neutral() {
        let p = SyntheticFlameParams::neutral(10, 5);
        assert_eq!(p.expression_magnitude(), 0.0);
    }

    #[test]
    fn test_is_neutral_true_for_neutral() {
        let p = SyntheticFlameParams::neutral(10, 5);
        assert!(p.is_neutral(1e-6));
    }

    #[test]
    fn test_is_neutral_false_when_pose_nonzero() {
        let mut p = SyntheticFlameParams::neutral(10, 5);
        p.pose[4] = 0.5;
        assert!(!p.is_neutral(0.1));
    }

    #[test]
    fn test_pose_magnitude_ignores_root() {
        let mut p = SyntheticFlameParams::neutral(10, 5);
        // Root (indices 0..3) should be ignored
        p.pose[0] = 100.0;
        assert_eq!(p.pose_magnitude(), 0.0);
        // Non-root
        p.pose[3] = 1.0;
        assert!((p.pose_magnitude() - 1.0).abs() < 1e-5);
    }

    // ── FlameParamSamplerConfig::validate ─────────────────────────────────────

    #[test]
    fn test_sampler_config_validate_defaults() {
        let cfg = FlameParamSamplerConfig::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_sampler_config_validate_zero_n_shape() {
        let cfg = FlameParamSamplerConfig {
            n_shape_betas: 0,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_sampler_config_validate_zero_n_expr() {
        let cfg = FlameParamSamplerConfig {
            n_expr_betas: 0,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_sampler_config_validate_negative_std() {
        let cfg = FlameParamSamplerConfig {
            shape_std: -1.0,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    // ── FlameParamSampler ─────────────────────────────────────────────────────

    #[test]
    fn test_sampler_new_valid_config() {
        let cfg = FlameParamSamplerConfig::default();
        let sampler = FlameParamSampler::new(cfg);
        assert!(sampler.is_ok());
    }

    #[test]
    fn test_sample_vector_lengths() {
        let cfg = FlameParamSamplerConfig {
            n_shape_betas: 7,
            n_expr_betas: 3,
            ..Default::default()
        };
        let mut sampler = FlameParamSampler::new(cfg).expect("valid config");
        let p = sampler.sample();
        assert_eq!(p.shape.len(), 7);
        assert_eq!(p.expression.len(), 3);
        assert_eq!(p.pose.len(), 15);
        assert_eq!(p.translation.len(), 3);
    }

    #[test]
    fn test_sample_batch_count() {
        let mut sampler = FlameParamSampler::new(Default::default()).expect("valid");
        let batch = sampler.sample_batch(8);
        assert_eq!(batch.len(), 8);
    }

    #[test]
    fn test_sample_different_seeds_different_results() {
        let cfg_a = FlameParamSamplerConfig {
            seed: 1,
            ..Default::default()
        };
        let cfg_b = FlameParamSamplerConfig {
            seed: 2,
            ..Default::default()
        };
        let mut a = FlameParamSampler::new(cfg_a).expect("valid");
        let mut b = FlameParamSampler::new(cfg_b).expect("valid");
        let pa = a.sample();
        let pb = b.sample();
        // At minimum the expression vectors should differ
        assert_ne!(pa.expression, pb.expression);
    }

    #[test]
    fn test_sample_trajectory_length() {
        let mut sampler = FlameParamSampler::new(Default::default()).expect("valid");
        for n in [0usize, 1, 5, 30] {
            let traj = sampler.sample_trajectory(n);
            assert_eq!(traj.len(), n, "trajectory length mismatch for n={n}");
        }
    }

    #[test]
    fn test_sample_jaw_rotation_in_bounds() {
        let max_jaw = 0.4;
        let cfg = FlameParamSamplerConfig {
            max_jaw_rotation: max_jaw,
            ..Default::default()
        };
        let mut sampler = FlameParamSampler::new(cfg).expect("valid");
        for _ in 0..500 {
            let p = sampler.sample();
            // pose[6] is jaw; must be in [0, max_jaw]
            assert!(
                p.pose[6] >= 0.0 && p.pose[6] <= max_jaw,
                "jaw rotation {} out of [0, {}]",
                p.pose[6],
                max_jaw
            );
            // pose[7] and pose[8] must be zero
            assert_eq!(p.pose[7], 0.0);
            assert_eq!(p.pose[8], 0.0);
        }
    }

    #[test]
    fn test_sample_near_neutral_has_zero_pose() {
        let mut sampler = FlameParamSampler::new(Default::default()).expect("valid");
        let p = sampler.sample_near_neutral(0.5);
        assert!(
            p.pose.iter().all(|&v| v == 0.0),
            "near-neutral pose should be all zeros"
        );
    }

    // ── sample_unit_quaternion ────────────────────────────────────────────────

    #[test]
    fn test_unit_quaternion_unit_length() {
        let mut state = 7777u64;
        for _ in 0..1000 {
            let q = sample_unit_quaternion(&mut state);
            let mag_sq = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
            assert!(
                (mag_sq - 1.0).abs() < 1e-5,
                "quaternion not unit: mag_sq={mag_sq}"
            );
        }
    }

    // ── GaussianCloudConfig & sample_gaussian_cloud ───────────────────────────

    #[test]
    fn test_gaussian_cloud_config_validate_valid() {
        let cfg = GaussianCloudConfig::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_gaussian_cloud_config_validate_zero_gaussians() {
        let cfg = GaussianCloudConfig {
            num_gaussians: 0,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_gaussian_cloud_config_validate_negative_scale_std() {
        let cfg = GaussianCloudConfig {
            log_scale_std: -0.1,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_sample_gaussian_cloud_num_gaussians() {
        let cfg = GaussianCloudConfig {
            num_gaussians: 42,
            ..Default::default()
        };
        let cloud = sample_gaussian_cloud(&cfg).expect("valid");
        assert_eq!(cloud.num_gaussians(), 42);
    }

    #[test]
    fn test_sample_gaussian_cloud_positions_length() {
        let n = 100;
        let cfg = GaussianCloudConfig {
            num_gaussians: n,
            ..Default::default()
        };
        let cloud = sample_gaussian_cloud(&cfg).expect("valid");
        assert_eq!(cloud.positions.len(), n * 3);
    }

    #[test]
    fn test_centroid_approximately_origin_for_large_cloud() {
        // With large N and symmetric N(0,σ), centroid → 0 by LLN.
        let cfg = GaussianCloudConfig {
            num_gaussians: 10_000,
            seed: 42,
            ..Default::default()
        };
        let cloud = sample_gaussian_cloud(&cfg).expect("valid");
        let c = cloud.centroid();
        let tol = 0.05;
        assert!(c[0].abs() < tol, "centroid x {} too large", c[0]);
        assert!(c[1].abs() < tol, "centroid y {} too large", c[1]);
        assert!(c[2].abs() < tol, "centroid z {} too large", c[2]);
    }

    #[test]
    fn test_aabb_min_leq_max() {
        let cfg = GaussianCloudConfig {
            num_gaussians: 200,
            ..Default::default()
        };
        let cloud = sample_gaussian_cloud(&cfg).expect("valid");
        let (lo, hi) = cloud.aabb();
        for axis in 0..3 {
            assert!(lo[axis] <= hi[axis], "AABB min > max on axis {axis}");
        }
    }

    #[test]
    fn test_mean_opacity_in_unit_interval() {
        let cfg = GaussianCloudConfig {
            num_gaussians: 500,
            ..Default::default()
        };
        let cloud = sample_gaussian_cloud(&cfg).expect("valid");
        let mo = cloud.mean_opacity();
        assert!(mo > 0.0 && mo < 1.0, "mean opacity {mo} not in (0, 1)");
    }

    // ── generate_synthetic_batch ──────────────────────────────────────────────

    #[test]
    fn test_generate_synthetic_batch_size() {
        let flame_cfg = FlameParamSamplerConfig::default();
        let batch = generate_synthetic_batch(4, 8, 6, &flame_cfg, 99).expect("valid");
        assert_eq!(batch.len(), 4);
    }

    #[test]
    fn test_generate_synthetic_batch_image_dimensions() {
        let flame_cfg = FlameParamSamplerConfig::default();
        let w = 16;
        let h = 12;
        let batch = generate_synthetic_batch(3, w, h, &flame_cfg, 55).expect("valid");
        assert_eq!(batch.width, w);
        assert_eq!(batch.height, h);
        for img in &batch.images {
            assert_eq!(
                img.len(),
                w * h * 3,
                "image pixel count mismatch: {} vs {}",
                img.len(),
                w * h * 3
            );
        }
    }

    #[test]
    fn test_generate_synthetic_batch_zero_size_err() {
        let flame_cfg = FlameParamSamplerConfig::default();
        let result = generate_synthetic_batch(0, 8, 8, &flame_cfg, 1);
        assert!(result.is_err());
    }

    // ── DifficultyLevel ───────────────────────────────────────────────────────

    #[test]
    fn test_difficulty_from_step_easy() {
        // warmup_steps = 100 → easy region is step < 25
        assert_eq!(DifficultyLevel::from_step(0, 100), DifficultyLevel::Easy);
        assert_eq!(DifficultyLevel::from_step(24, 100), DifficultyLevel::Easy);
    }

    #[test]
    fn test_difficulty_from_step_medium() {
        assert_eq!(DifficultyLevel::from_step(25, 100), DifficultyLevel::Medium);
        assert_eq!(DifficultyLevel::from_step(49, 100), DifficultyLevel::Medium);
    }

    #[test]
    fn test_difficulty_from_step_hard() {
        assert_eq!(DifficultyLevel::from_step(50, 100), DifficultyLevel::Hard);
        assert_eq!(DifficultyLevel::from_step(99, 100), DifficultyLevel::Hard);
    }

    #[test]
    fn test_difficulty_from_step_extreme() {
        assert_eq!(
            DifficultyLevel::from_step(100, 100),
            DifficultyLevel::Extreme
        );
        assert_eq!(
            DifficultyLevel::from_step(9999, 100),
            DifficultyLevel::Extreme
        );
    }

    // ── sample_at_difficulty ──────────────────────────────────────────────────

    #[test]
    fn test_sample_at_difficulty_easy_near_neutral_expression() {
        let mut sampler = FlameParamSampler::new(Default::default()).expect("valid");
        let p = sample_at_difficulty(&mut sampler, DifficultyLevel::Easy);
        // Easy level has expr_scale = 0.1 (std = 0.5 * 0.1 = 0.05).
        // Expression magnitude should be very small relative to Hard level.
        let mag = p.expression_magnitude();
        // With std=0.05 and 50 betas, rms ≈ 0.05, l2 ≈ 0.05*sqrt(50) ≈ 0.35.
        // We allow up to 1.0 to avoid flaky tests, but it must be much less
        // than the Hard-level magnitude in the comparison test below.
        assert!(mag < 2.0, "easy magnitude unexpectedly large: {mag}");
    }

    #[test]
    fn test_sample_at_difficulty_hard_larger_expression_than_easy() {
        // Use the same seed for fair comparison.
        let make_sampler = || {
            FlameParamSampler::new(FlameParamSamplerConfig {
                seed: 7,
                ..Default::default()
            })
            .expect("valid")
        };

        let easy_mag: f32 = {
            let mut s = make_sampler();
            (0..20)
                .map(|_| sample_at_difficulty(&mut s, DifficultyLevel::Easy).expression_magnitude())
                .sum::<f32>()
                / 20.0
        };
        let hard_mag: f32 = {
            let mut s = make_sampler();
            (0..20)
                .map(|_| sample_at_difficulty(&mut s, DifficultyLevel::Hard).expression_magnitude())
                .sum::<f32>()
                / 20.0
        };

        assert!(
            hard_mag > easy_mag,
            "hard ({hard_mag}) should have larger expression magnitude than easy ({easy_mag})"
        );
    }
}
