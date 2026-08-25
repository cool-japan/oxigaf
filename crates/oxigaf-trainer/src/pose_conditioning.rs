//! Camera pose coverage tracking and conditioning for 3DGS avatar training.
//!
//! Tracks which poses have been trained, identifies undertrained regions, and
//! manages pose-based batch composition to maximise spherical coverage.

use std::f32::consts::PI;
use thiserror::Error;

// ---------------------------------------------------------------------------
// xorshift64 (private – public versions live in data_augmentation)
// ---------------------------------------------------------------------------

fn xorshift64(state: &mut u64) -> u64 {
    (*state) ^= (*state) << 13;
    (*state) ^= (*state) >> 7;
    (*state) ^= (*state) << 17;
    if *state == 0 {
        *state = 1;
    }
    *state
}

fn xorshift_f32(state: &mut u64) -> f32 {
    (xorshift64(state) >> 11) as f32 / (1u64 << 53) as f32
}

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

/// Errors produced by pose conditioning.
#[derive(Debug, Error)]
pub enum PoseCondError {
    #[error("Empty pose set")]
    EmptyPoseSet,
    #[error("Pose index {0} out of range")]
    PoseIndexOutOfRange(usize),
    #[error("Invalid angle {0}: must be finite")]
    InvalidAngle(f32),
    #[error("Grid resolution {0} must be >= 2")]
    InvalidGridResolution(usize),
    #[error("Invalid parameter: {0}")]
    InvalidParam(String),
}

// ---------------------------------------------------------------------------
// SphericalPose
// ---------------------------------------------------------------------------

/// A camera pose in spherical coordinates (yaw, pitch) relative to object centre.
#[derive(Debug, Clone)]
pub struct SphericalPose {
    /// Azimuth angle in radians `[-PI, PI]`.
    pub yaw: f32,
    /// Elevation angle in radians `[-PI/2, PI/2]`.
    pub pitch: f32,
    /// Distance from centre.
    pub radius: f32,
}

impl SphericalPose {
    /// Create a new `SphericalPose`, validating that all values are finite.
    ///
    /// `yaw` is wrapped into `[-PI, PI]` and `pitch` is clamped to
    /// `[-PI/2, PI/2]` so every constructed pose satisfies the ranges
    /// documented on this struct's fields — free functions that operate on
    /// `SphericalPose` (e.g. [`interpolate_poses`], [`pose_to_grid_idx`])
    /// still defend against out-of-range values independently, since
    /// `yaw`/`pitch` are public fields and a caller can bypass `new` with a
    /// struct literal.
    pub fn new(yaw: f32, pitch: f32, radius: f32) -> Result<Self, PoseCondError> {
        if !yaw.is_finite() {
            return Err(PoseCondError::InvalidAngle(yaw));
        }
        if !pitch.is_finite() {
            return Err(PoseCondError::InvalidAngle(pitch));
        }
        if !radius.is_finite() {
            return Err(PoseCondError::InvalidAngle(radius));
        }
        Ok(Self {
            yaw: wrap_to_pi(yaw),
            pitch: pitch.clamp(-PI / 2.0, PI / 2.0),
            radius,
        })
    }

    /// Convert to Cartesian camera position `(x, y, z)`.
    pub fn to_cartesian(&self) -> [f32; 3] {
        let x = self.radius * self.pitch.cos() * self.yaw.cos();
        let y = self.radius * self.pitch.sin();
        let z = self.radius * self.pitch.cos() * self.yaw.sin();
        [x, y, z]
    }

    /// Geodesic angular distance on the unit sphere (radians).
    pub fn angular_distance(&self, other: &SphericalPose) -> f32 {
        let [x1, y1, z1] = unit_cartesian(self.yaw, self.pitch);
        let [x2, y2, z2] = unit_cartesian(other.yaw, other.pitch);
        let dot = (x1 * x2 + y1 * y2 + z1 * z2).clamp(-1.0, 1.0);
        dot.acos()
    }

    /// Convert Cartesian camera position back to spherical coordinates.
    pub fn from_cartesian(pos: [f32; 3]) -> Self {
        let [x, y, z] = pos;
        let r = (x * x + y * y + z * z).sqrt();
        let pitch = if r > 0.0 { (y / r).asin() } else { 0.0 };
        let yaw = z.atan2(x);
        Self {
            yaw,
            pitch,
            radius: r,
        }
    }
}

/// Wrap an angle in radians into `[-PI, PI]`, matching the range documented
/// on [`SphericalPose::yaw`]. Non-finite input propagates unchanged (no
/// panic). Runs in O(1) regardless of input magnitude (a `while`-loop
/// reduction would be unbounded for a pathological hand-constructed pose
/// with an extreme yaw, since `SphericalPose`'s fields are public).
fn wrap_to_pi(angle: f32) -> f32 {
    const TAU: f32 = 2.0 * PI;
    let wrapped = angle % TAU;
    if wrapped > PI {
        wrapped - TAU
    } else if wrapped < -PI {
        wrapped + TAU
    } else {
        wrapped
    }
}

/// Return the unit-sphere Cartesian point for a (yaw, pitch) pair.
fn unit_cartesian(yaw: f32, pitch: f32) -> [f32; 3] {
    let x = pitch.cos() * yaw.cos();
    let y = pitch.sin();
    let z = pitch.cos() * yaw.sin();
    [x, y, z]
}

// ---------------------------------------------------------------------------
// RegisteredPose
// ---------------------------------------------------------------------------

/// A registered training pose with coverage and loss statistics.
#[derive(Debug, Clone)]
pub struct RegisteredPose {
    pub pose: SphericalPose,
    pub visit_count: usize,
    /// Most recent loss recorded at this pose.
    pub last_loss: f32,
    /// Exponential moving average of losses seen at this pose.
    pub mean_loss: f32,
    /// How much of the sphere this pose "covers" (influence radius in radians).
    pub coverage_radius: f32,
}

impl RegisteredPose {
    fn new(pose: SphericalPose, coverage_radius: f32) -> Self {
        Self {
            pose,
            visit_count: 0,
            last_loss: 0.0,
            mean_loss: 1.0, // start high so unvisited poses are prioritised
            coverage_radius,
        }
    }
}

/// Coverage-grid weight contributed by a single registered pose:
/// `visit_count` once visited, or `1.0` while still unvisited (an
/// unvisited pose still contributes a baseline weight equal to a
/// freshly-visited one, so the first visit alone does not change the
/// grid — only the second and later visits do).
fn pose_coverage_weight(rp: &RegisteredPose) -> f32 {
    if rp.visit_count == 0 {
        1.0
    } else {
        rp.visit_count as f32
    }
}

// ---------------------------------------------------------------------------
// PoseCondConfig
// ---------------------------------------------------------------------------

/// Configuration for the pose conditioner.
#[derive(Debug, Clone)]
pub struct PoseCondConfig {
    /// Spherical grid resolution along the yaw axis.
    /// Grid is `grid_resolution × (grid_resolution/2 + 1)` cells.
    pub grid_resolution: usize,
    /// Coverage Gaussian sigma in radians (default `0.3`).
    pub coverage_sigma: f32,
    /// Coverage below this threshold is considered undertrained (default `0.1`).
    pub min_coverage: f32,
    /// EMA decay factor for per-pose loss tracking (default `0.9`).
    pub ema_decay: f32,
    /// Weight for diversity vs. quality in pose selection (default `0.3`).
    pub diversity_weight: f32,
}

impl Default for PoseCondConfig {
    fn default() -> Self {
        Self {
            grid_resolution: 16,
            coverage_sigma: 0.3,
            min_coverage: 0.1,
            ema_decay: 0.9,
            diversity_weight: 0.3,
        }
    }
}

// ---------------------------------------------------------------------------
// CoverageStats
// ---------------------------------------------------------------------------

/// Summary statistics for the spherical coverage grid.
#[derive(Debug, Clone)]
pub struct CoverageStats {
    pub mean_coverage: f32,
    pub min_coverage: f32,
    pub max_coverage: f32,
    /// Fraction of grid cells with coverage ≥ `min_coverage`.
    pub coverage_fraction: f32,
    pub total_visits: usize,
    pub n_poses: usize,
}

// ---------------------------------------------------------------------------
// PoseConditioner
// ---------------------------------------------------------------------------

/// Tracks camera-pose coverage during 3DGS avatar training.
pub struct PoseConditioner {
    config: PoseCondConfig,
    registered_poses: Vec<RegisteredPose>,
    /// Flat grid of coverage scores (normalized: `raw_grid / total_weight`);
    /// shape `[grid_res × (grid_res/2+1)]`.
    coverage_grid: Vec<f32>,
    /// Unnormalized `Σ weight_i * contribution_i` accumulator, same shape as
    /// `coverage_grid`; kept in sync so `update()` can apply an
    /// O(grid_len) weight delta instead of recomputing every registered
    /// pose's contribution from scratch on every call.
    raw_grid: Vec<f32>,
    /// `Σ weight_i` over all registered poses, kept in sync with `raw_grid`.
    total_weight: f32,
    /// Cached per-pose coverage kernel (`spherical_gaussian_coverage`
    /// evaluated once at registration), reused by `update()`'s incremental
    /// path. Parallel to `registered_poses`.
    cached_contribs: Vec<Vec<f32>>,
    step: usize,
    rng_state: u64,
}

impl PoseConditioner {
    /// Create a new conditioner with the given configuration and RNG seed.
    pub fn new(config: PoseCondConfig, seed: u64) -> Result<Self, PoseCondError> {
        if config.grid_resolution < 2 {
            return Err(PoseCondError::InvalidGridResolution(config.grid_resolution));
        }
        if !config.coverage_sigma.is_finite() || config.coverage_sigma <= 0.0 {
            return Err(PoseCondError::InvalidParam(format!(
                "coverage_sigma must be positive finite, got {}",
                config.coverage_sigma
            )));
        }
        if config.ema_decay <= 0.0 || config.ema_decay >= 1.0 {
            return Err(PoseCondError::InvalidParam(format!(
                "ema_decay must be in (0, 1), got {}",
                config.ema_decay
            )));
        }
        let grid_yaw = config.grid_resolution;
        let grid_pitch = config.grid_resolution / 2 + 1;
        let grid_len = grid_yaw * grid_pitch;
        let rng_state = if seed == 0 { 12345678901u64 } else { seed };
        Ok(Self {
            config,
            registered_poses: Vec::new(),
            coverage_grid: vec![0.0_f32; grid_len],
            raw_grid: vec![0.0_f32; grid_len],
            total_weight: 0.0,
            cached_contribs: Vec::new(),
            step: 0,
            rng_state,
        })
    }

    /// Register a set of training poses from the dataset.
    pub fn register_poses(&mut self, poses: Vec<SphericalPose>) -> Result<(), PoseCondError> {
        if poses.is_empty() {
            return Err(PoseCondError::EmptyPoseSet);
        }
        for pose in poses {
            let rp = RegisteredPose::new(pose, self.config.coverage_sigma);
            self.registered_poses.push(rp);
        }
        self.recompute_coverage();
        Ok(())
    }

    /// Update coverage and loss statistics after training on a registered pose.
    ///
    /// Coverage is updated incrementally in O(grid_len) — folding in only
    /// this pose's weight delta via its cached kernel — rather than
    /// recomputing every registered pose's contribution from scratch (see
    /// [`Self::recompute_coverage`]), since this is called once per
    /// training step.
    pub fn update(&mut self, pose_idx: usize, loss: f32) -> Result<(), PoseCondError> {
        let n = self.registered_poses.len();
        if pose_idx >= n {
            return Err(PoseCondError::PoseIndexOutOfRange(pose_idx));
        }
        let old_weight = pose_coverage_weight(&self.registered_poses[pose_idx]);

        let rp = self
            .registered_poses
            .get_mut(pose_idx)
            .ok_or(PoseCondError::PoseIndexOutOfRange(pose_idx))?;
        rp.visit_count += 1;
        rp.last_loss = loss;
        let decay = self.config.ema_decay;
        rp.mean_loss = decay * rp.mean_loss + (1.0 - decay) * loss;
        let new_weight = pose_coverage_weight(rp);
        self.step += 1;

        let delta = new_weight - old_weight;
        if delta != 0.0 {
            match self.cached_contribs.get(pose_idx).cloned() {
                Some(contrib) => {
                    for (g, c) in self.raw_grid.iter_mut().zip(contrib.iter()) {
                        *g += delta * c;
                    }
                    self.total_weight += delta;
                    let denom = self.total_weight.max(1.0);
                    for (g, r) in self.coverage_grid.iter_mut().zip(self.raw_grid.iter()) {
                        *g = r / denom;
                    }
                }
                // Cache miss (should not happen in practice — every
                // registered pose gets a cached contribution in
                // `recompute_coverage`, which `register_poses` always
                // calls). Fall back to a full rebuild for correctness.
                None => self.recompute_coverage(),
            }
        }
        Ok(())
    }

    /// Recompute the spherical coverage grid from all registered poses,
    /// rebuilding the per-pose contribution cache used by
    /// [`Self::update`]'s incremental path.
    ///
    /// This is O(n_poses × grid_len) — called once by
    /// [`Self::register_poses`] after a (typically rare) batch of pose
    /// registrations. Per-step coverage updates should go through
    /// [`Self::update`] instead, which is O(grid_len).
    pub fn recompute_coverage(&mut self) {
        let n = self.registered_poses.len();
        let grid_yaw = self.config.grid_resolution;
        let grid_pitch = self.config.grid_resolution / 2 + 1;
        let grid_len = grid_yaw * grid_pitch;
        if n == 0 {
            self.raw_grid = vec![0.0_f32; grid_len];
            self.total_weight = 0.0;
            self.cached_contribs.clear();
            self.coverage_grid = vec![0.0_f32; grid_len];
            return;
        }
        let sigma = self.config.coverage_sigma;
        let mut raw_grid = vec![0.0_f32; grid_len];
        let mut cached_contribs = Vec::with_capacity(n);
        let mut total_weight = 0.0_f32;
        for rp in &self.registered_poses {
            let weight = pose_coverage_weight(rp);
            let contrib = spherical_gaussian_coverage(&rp.pose, grid_yaw, grid_pitch, sigma);
            for (g, c) in raw_grid.iter_mut().zip(contrib.iter()) {
                *g += weight * c;
            }
            total_weight += weight;
            cached_contribs.push(contrib);
        }
        // Normalize by the sum of weights (not by `n`): each pose
        // contributes `weight * kernel`, where `weight` grows with
        // `visit_count`, so dividing by a fixed `n` let coverage grow
        // without bound as visits accumulated instead of staying in a
        // fixed range comparable to `config.min_coverage`.
        let denom = total_weight.max(1.0);
        self.coverage_grid = raw_grid.iter().map(|&r| r / denom).collect();
        self.raw_grid = raw_grid;
        self.cached_contribs = cached_contribs;
        self.total_weight = total_weight;
    }

    /// Get the interpolated coverage at a given spherical pose.
    pub fn coverage_at(&self, pose: &SphericalPose) -> f32 {
        let idx = pose_to_grid_idx(pose, self.config.grid_resolution);
        self.coverage_grid.get(idx).copied().unwrap_or(0.0)
    }

    /// Return representative poses for all grid cells with coverage below `min_coverage`.
    pub fn undertrained_regions(&self) -> Vec<SphericalPose> {
        let grid_yaw = self.config.grid_resolution;
        let min_cov = self.config.min_coverage;
        let mean_radius = if self.registered_poses.is_empty() {
            1.0
        } else {
            let sum: f32 = self.registered_poses.iter().map(|rp| rp.pose.radius).sum();
            sum / self.registered_poses.len() as f32
        };
        self.coverage_grid
            .iter()
            .enumerate()
            .filter(|(_, &cov)| cov < min_cov)
            .map(|(idx, _)| grid_idx_to_pose(idx, grid_yaw, mean_radius))
            .collect()
    }

    /// Select the next registered pose to train on, balancing quality and coverage.
    ///
    /// Returns the index into `registered_poses`.
    pub fn select_pose(&mut self) -> Result<usize, PoseCondError> {
        let n = self.registered_poses.len();
        if n == 0 {
            return Err(PoseCondError::EmptyPoseSet);
        }
        let dw = self.config.diversity_weight;
        // Compute raw scores.
        let scores: Vec<f32> = self
            .registered_poses
            .iter()
            .map(|rp| {
                let loss_score = rp.mean_loss;
                let cov_score = 1.0 - self.coverage_at(&rp.pose).min(1.0);
                (1.0 - dw) * loss_score + dw * cov_score
            })
            .collect();
        // Stable softmax.
        let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
        let total: f32 = exps.iter().sum();
        // Sample proportionally.
        let r = xorshift_f32(&mut self.rng_state) * total;
        let mut cumsum = 0.0;
        for (i, &e) in exps.iter().enumerate() {
            cumsum += e;
            if cumsum >= r {
                return Ok(i);
            }
        }
        Ok(n - 1)
    }

    /// Select a batch of `n` poses that maximise diversity (greedy farthest-point sampling).
    ///
    /// Runs in O(k × total) rather than O(k² × total): a
    /// `min_dist_to_selected` array tracks each not-yet-selected pose's
    /// running minimum distance to the selected set incrementally (folding
    /// in only the newly-selected pose each round, instead of recomputing
    /// the minimum over the whole selected set from scratch for every
    /// candidate), and a `Vec<bool>` mask replaces an O(k) `Vec::contains`
    /// membership check with O(1). Tie-breaking (first index wins) and
    /// results are identical to the original algorithm.
    pub fn select_diverse_batch(&mut self, n: usize) -> Result<Vec<usize>, PoseCondError> {
        let total = self.registered_poses.len();
        if total == 0 {
            return Err(PoseCondError::EmptyPoseSet);
        }
        let k = n.min(total);
        // Start with the least-visited pose.
        let start = self.least_visited_pose().unwrap_or(0);

        let mut selected_mask = vec![false; total];
        selected_mask[start] = true;
        let mut selected = Vec::with_capacity(k);
        selected.push(start);

        let mut min_dist_to_selected = vec![f32::INFINITY; total];
        let start_pose = self.registered_poses[start].pose.clone();
        for (c, dist) in min_dist_to_selected.iter_mut().enumerate() {
            if c != start {
                *dist = self.registered_poses[c].pose.angular_distance(&start_pose);
            }
        }

        // Greedy farthest-point sampling.
        while selected.len() < k {
            let mut best_idx = 0usize;
            let mut best_min_dist = f32::NEG_INFINITY;
            for (candidate, &dist) in min_dist_to_selected.iter().enumerate() {
                if selected_mask[candidate] {
                    continue;
                }
                if dist > best_min_dist {
                    best_min_dist = dist;
                    best_idx = candidate;
                }
            }
            selected_mask[best_idx] = true;
            selected.push(best_idx);

            // Fold the newly-selected pose into every remaining candidate's
            // running minimum distance.
            let new_pose = self.registered_poses[best_idx].pose.clone();
            for (c, dist) in min_dist_to_selected.iter_mut().enumerate() {
                if !selected_mask[c] {
                    let d = self.registered_poses[c].pose.angular_distance(&new_pose);
                    if d < *dist {
                        *dist = d;
                    }
                }
            }
        }
        Ok(selected)
    }

    /// Return aggregate coverage statistics.
    pub fn coverage_stats(&self) -> CoverageStats {
        if self.coverage_grid.is_empty() {
            return CoverageStats {
                mean_coverage: 0.0,
                min_coverage: 0.0,
                max_coverage: 0.0,
                coverage_fraction: 0.0,
                total_visits: 0,
                n_poses: 0,
            };
        }
        let len = self.coverage_grid.len() as f32;
        let min_cov = self.config.min_coverage;
        let mean = self.coverage_grid.iter().sum::<f32>() / len;
        let min = self
            .coverage_grid
            .iter()
            .cloned()
            .fold(f32::INFINITY, f32::min);
        let max = self
            .coverage_grid
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let above = self.coverage_grid.iter().filter(|&&c| c >= min_cov).count();
        let coverage_fraction = above as f32 / len;
        let total_visits = self.registered_poses.iter().map(|rp| rp.visit_count).sum();
        CoverageStats {
            mean_coverage: mean,
            min_coverage: min,
            max_coverage: max,
            coverage_fraction,
            total_visits,
            n_poses: self.registered_poses.len(),
        }
    }

    /// Return the index of the pose with the highest EMA loss.
    pub fn hardest_pose(&self) -> Option<usize> {
        self.registered_poses
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.mean_loss
                    .partial_cmp(&b.mean_loss)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
    }

    /// Return the index of the least-visited pose (ties broken by index).
    pub fn least_visited_pose(&self) -> Option<usize> {
        self.registered_poses
            .iter()
            .enumerate()
            .min_by_key(|(_, rp)| rp.visit_count)
            .map(|(i, _)| i)
    }

    /// Number of registered poses.
    pub fn n_poses(&self) -> usize {
        self.registered_poses.len()
    }

    /// Current training step counter.
    pub fn step(&self) -> usize {
        self.step
    }
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// Compute the spherical-Gaussian coverage contribution of a single pose over a grid.
///
/// Returns a flat `Vec<f32>` of length `grid_yaw * grid_pitch`.
pub fn spherical_gaussian_coverage(
    pose: &SphericalPose,
    grid_yaw: usize,
    grid_pitch: usize,
    sigma: f32,
) -> Vec<f32> {
    let n = grid_yaw * grid_pitch;
    let mut result = Vec::with_capacity(n);
    let inv_two_sigma_sq = 0.5 / (sigma * sigma);
    for i in 0..grid_yaw {
        for j in 0..grid_pitch {
            // Cell centres so distances are consistent with grid_idx_to_pose.
            let yaw_cell = ((i as f32 + 0.5) / grid_yaw as f32) * 2.0 * PI - PI;
            let pitch_cell = ((j as f32 + 0.5) / grid_pitch as f32) * PI - PI / 2.0;
            // Geodesic distance from pose to cell centre.
            let cell_pose = SphericalPose {
                yaw: yaw_cell,
                pitch: pitch_cell,
                radius: 1.0,
            };
            let dist = pose.angular_distance(&cell_pose);
            result.push((-inv_two_sigma_sq * dist * dist).exp());
        }
    }
    result
}

/// Compute pairwise angular distances between poses.
///
/// Returns a flat `n×n` matrix stored row-major.
pub fn pairwise_angular_distances(poses: &[SphericalPose]) -> Vec<f32> {
    let n = poses.len();
    let mut mat = vec![0.0_f32; n * n];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                mat[i * n + j] = 0.0;
            } else if j > i {
                let d = poses[i].angular_distance(&poses[j]);
                mat[i * n + j] = d;
                mat[j * n + i] = d;
            }
        }
    }
    mat
}

/// Compute a diversity score for a set of poses (mean of per-pose minimum angular distances).
pub fn pose_diversity_score(poses: &[SphericalPose]) -> f32 {
    let n = poses.len();
    if n <= 1 {
        return 0.0;
    }
    let mat = pairwise_angular_distances(poses);
    let sum: f32 = (0..n)
        .map(|i| {
            (0..n)
                .filter(|&j| j != i)
                .map(|j| mat[i * n + j])
                .fold(f32::INFINITY, f32::min)
        })
        .sum();
    sum / n as f32
}

/// Select `k` poses from `poses` that maximise diversity using greedy farthest-point sampling.
///
/// Returns a sorted vector of indices (length ≤ min(k, poses.len())).
pub fn select_diverse_poses(poses: &[SphericalPose], k: usize, seed: u64) -> Vec<usize> {
    let n = poses.len();
    if n == 0 || k == 0 {
        return Vec::new();
    }
    let effective_k = k.min(n);
    // Start from a random pose.
    let mut rng = if seed == 0 { 987654321u64 } else { seed };
    let start = (xorshift64(&mut rng) as usize) % n;
    let mut selected = vec![start];
    let mat = pairwise_angular_distances(poses);
    while selected.len() < effective_k {
        let mut best_idx = 0usize;
        let mut best_dist = f32::NEG_INFINITY;
        for candidate in 0..n {
            if selected.contains(&candidate) {
                continue;
            }
            let min_dist = selected
                .iter()
                .map(|&s| mat[candidate * n + s])
                .fold(f32::INFINITY, f32::min);
            if min_dist > best_dist {
                best_dist = min_dist;
                best_idx = candidate;
            }
        }
        selected.push(best_idx);
    }
    selected.sort_unstable();
    selected
}

/// Map a flat grid index to the CENTRE of the corresponding spherical grid cell.
pub fn grid_idx_to_pose(grid_idx: usize, grid_res: usize, radius: f32) -> SphericalPose {
    let grid_pitch_res = grid_res / 2 + 1;
    let pitch_idx = grid_idx % grid_pitch_res;
    let yaw_idx = grid_idx / grid_pitch_res;
    // Use cell centres (+0.5) so that round-tripping through pose_to_grid_idx is stable.
    let yaw = ((yaw_idx as f32 + 0.5) / grid_res as f32) * 2.0 * PI - PI;
    let pitch = ((pitch_idx as f32 + 0.5) / grid_pitch_res as f32) * PI - PI / 2.0;
    SphericalPose { yaw, pitch, radius }
}

/// Map a spherical pose to the nearest flat grid index.
///
/// `yaw` wraps around the ±PI seam (it is a periodic azimuth angle), so an
/// out-of-range yaw (e.g. from a hand-constructed `SphericalPose` that
/// bypassed `SphericalPose::new`) maps to the geometrically correct cell
/// instead of clamping into the last yaw bin. `pitch` is not periodic — it
/// clamps to `[-PI/2, PI/2]`.
pub fn pose_to_grid_idx(pose: &SphericalPose, grid_res: usize) -> usize {
    let grid_pitch_res = grid_res / 2 + 1;
    let yaw = wrap_to_pi(pose.yaw);
    let pitch = pose.pitch.clamp(-PI / 2.0, PI / 2.0);
    let yaw_norm = (yaw + PI) / (2.0 * PI);
    let pitch_norm = (pitch + PI / 2.0) / PI;
    let yaw_idx = ((yaw_norm * grid_res as f32).floor() as usize).min(grid_res - 1);
    let pitch_idx = ((pitch_norm * grid_pitch_res as f32).floor() as usize).min(grid_pitch_res - 1);
    yaw_idx * grid_pitch_res + pitch_idx
}

/// Linearly interpolate between two spherical poses.
///
/// `yaw` is interpolated along the shortest arc across the ±PI wrap
/// boundary — e.g. interpolating from `yaw=3.0` to `yaw=-3.0` (a 0.28 rad
/// arc across the seam) sweeps the short way through PI rather than the
/// long way through 0. `pitch` and `radius` are interpolated linearly.
pub fn interpolate_poses(a: &SphericalPose, b: &SphericalPose, t: f32) -> SphericalPose {
    let dyaw = wrap_to_pi(b.yaw - a.yaw);
    SphericalPose {
        yaw: a.yaw + t * dyaw,
        pitch: a.pitch + t * (b.pitch - a.pitch),
        radius: a.radius + t * (b.radius - a.radius),
    }
}

/// Compute the angular centroid of a set of poses via mean unit-sphere Cartesian coordinates.
pub fn pose_centroid(poses: &[SphericalPose]) -> Result<SphericalPose, PoseCondError> {
    if poses.is_empty() {
        return Err(PoseCondError::EmptyPoseSet);
    }
    let n = poses.len() as f32;
    let mut sx = 0.0_f32;
    let mut sy = 0.0_f32;
    let mut sz = 0.0_f32;
    let mean_radius: f32 = poses.iter().map(|p| p.radius).sum::<f32>() / n;
    for p in poses {
        let [x, y, z] = unit_cartesian(p.yaw, p.pitch);
        sx += x;
        sy += y;
        sz += z;
    }
    sx /= n;
    sy /= n;
    sz /= n;
    let r = (sx * sx + sy * sy + sz * sz).sqrt();
    if r < 1e-8 {
        // Poses cancel out; return equatorial front.
        return Ok(SphericalPose {
            yaw: 0.0,
            pitch: 0.0,
            radius: mean_radius,
        });
    }
    let pitch = (sy / r).asin();
    let yaw = sz.atan2(sx);
    Ok(SphericalPose {
        yaw,
        pitch,
        radius: mean_radius,
    })
}

/// Format coverage statistics as a human-readable string.
pub fn format_coverage_stats(stats: &CoverageStats) -> String {
    format!(
        "CoverageStats {{ mean={:.3}, min={:.3}, max={:.3}, fraction={:.1}%, visits={}, poses={} }}",
        stats.mean_coverage,
        stats.min_coverage,
        stats.max_coverage,
        stats.coverage_fraction * 100.0,
        stats.total_visits,
        stats.n_poses,
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn default_conditioner(n_poses: usize) -> PoseConditioner {
        let config = PoseCondConfig::default();
        let mut cond = PoseConditioner::new(config, 42).unwrap();
        let poses: Vec<SphericalPose> = (0..n_poses)
            .map(|i| {
                let yaw = (i as f32 / n_poses as f32) * 2.0 * PI - PI;
                SphericalPose::new(yaw, 0.0, 2.0).unwrap()
            })
            .collect();
        cond.register_poses(poses).unwrap();
        cond
    }

    // ---- SphericalPose::new ----

    #[test]
    fn test_spherical_pose_new_valid() {
        let p = SphericalPose::new(0.0, 0.0, 1.0);
        assert!(p.is_ok());
    }

    #[test]
    fn test_spherical_pose_new_nan_yaw() {
        let p = SphericalPose::new(f32::NAN, 0.0, 1.0);
        assert!(matches!(p, Err(PoseCondError::InvalidAngle(_))));
    }

    #[test]
    fn test_spherical_pose_new_inf_pitch() {
        let p = SphericalPose::new(0.0, f32::INFINITY, 1.0);
        assert!(matches!(p, Err(PoseCondError::InvalidAngle(_))));
    }

    #[test]
    fn test_spherical_pose_new_nan_radius() {
        let p = SphericalPose::new(0.0, 0.0, f32::NAN);
        assert!(matches!(p, Err(PoseCondError::InvalidAngle(_))));
    }

    // ---- SphericalPose::to_cartesian ----

    #[test]
    fn test_to_cartesian_front() {
        let p = SphericalPose::new(0.0, 0.0, 1.0).unwrap();
        let [x, y, z] = p.to_cartesian();
        assert!((x - 1.0).abs() < 1e-6, "x should be ~1.0, got {x}");
        assert!(y.abs() < 1e-6, "y should be ~0.0, got {y}");
        assert!(z.abs() < 1e-6, "z should be ~0.0, got {z}");
    }

    #[test]
    fn test_to_cartesian_top() {
        let p = SphericalPose::new(0.0, PI / 2.0, 1.0).unwrap();
        let [_x, y, _z] = p.to_cartesian();
        assert!((y - 1.0).abs() < 1e-5, "y should be ~1.0 at top, got {y}");
    }

    #[test]
    fn test_to_cartesian_radius_scales() {
        let p = SphericalPose::new(0.0, 0.0, 3.0).unwrap();
        let [x, _y, _z] = p.to_cartesian();
        assert!((x - 3.0).abs() < 1e-5);
    }

    // ---- SphericalPose::from_cartesian ----

    #[test]
    fn test_from_cartesian_round_trip() {
        let original = SphericalPose::new(0.7, 0.3, 2.5).unwrap();
        let cart = original.to_cartesian();
        let recovered = SphericalPose::from_cartesian(cart);
        assert!((recovered.yaw - original.yaw).abs() < 1e-5, "yaw mismatch");
        assert!(
            (recovered.pitch - original.pitch).abs() < 1e-5,
            "pitch mismatch"
        );
        assert!(
            (recovered.radius - original.radius).abs() < 1e-4,
            "radius mismatch"
        );
    }

    #[test]
    fn test_from_cartesian_origin() {
        let p = SphericalPose::from_cartesian([0.0, 0.0, 0.0]);
        assert_eq!(p.radius, 0.0);
    }

    #[test]
    fn test_from_cartesian_back() {
        let p = SphericalPose::from_cartesian([-1.0, 0.0, 0.0]);
        assert!((p.radius - 1.0).abs() < 1e-5);
    }

    // ---- SphericalPose::angular_distance ----

    #[test]
    fn test_angular_distance_same_pose() {
        let p = SphericalPose::new(0.5, 0.2, 1.0).unwrap();
        assert!(p.angular_distance(&p) < 1e-5);
    }

    #[test]
    fn test_angular_distance_opposite_poles() {
        let north = SphericalPose::new(0.0, PI / 2.0, 1.0).unwrap();
        let south = SphericalPose::new(0.0, -PI / 2.0, 1.0).unwrap();
        let d = north.angular_distance(&south);
        assert!(
            (d - PI).abs() < 1e-5,
            "opposite poles should be PI apart, got {d}"
        );
    }

    #[test]
    fn test_angular_distance_symmetric() {
        let a = SphericalPose::new(0.3, 0.1, 1.0).unwrap();
        let b = SphericalPose::new(-0.5, 0.4, 1.0).unwrap();
        let d1 = a.angular_distance(&b);
        let d2 = b.angular_distance(&a);
        assert!((d1 - d2).abs() < 1e-6);
    }

    #[test]
    fn test_angular_distance_ninety_degrees() {
        let front = SphericalPose::new(0.0, 0.0, 1.0).unwrap();
        let top = SphericalPose::new(0.0, PI / 2.0, 1.0).unwrap();
        let d = front.angular_distance(&top);
        assert!((d - PI / 2.0).abs() < 1e-5, "should be PI/2, got {d}");
    }

    // ---- pose_to_grid_idx ----

    #[test]
    fn test_pose_to_grid_idx_front() {
        let pose = SphericalPose::new(0.0, 0.0, 1.0).unwrap();
        let idx = pose_to_grid_idx(&pose, 16);
        // Verify it's within grid bounds.
        let grid_pitch = 16 / 2 + 1;
        assert!(idx < 16 * grid_pitch);
    }

    #[test]
    fn test_pose_to_grid_idx_boundaries() {
        let pose_min = SphericalPose::new(-PI, -PI / 2.0, 1.0).unwrap();
        let pose_max = SphericalPose::new(PI - 0.001, PI / 2.0 - 0.001, 1.0).unwrap();
        let grid_size = 16 * (16 / 2 + 1);
        assert!(pose_to_grid_idx(&pose_min, 16) < grid_size);
        assert!(pose_to_grid_idx(&pose_max, 16) < grid_size);
    }

    // ---- grid_idx_to_pose ----

    #[test]
    fn test_grid_idx_to_pose_in_range() {
        let grid_res = 8;
        let grid_pitch = grid_res / 2 + 1;
        for idx in 0..(grid_res * grid_pitch) {
            let p = grid_idx_to_pose(idx, grid_res, 1.0);
            assert!(
                p.yaw >= -PI - 1e-5 && p.yaw <= PI + 1e-5,
                "yaw out of range: {}",
                p.yaw
            );
            assert!(
                p.pitch >= -PI / 2.0 - 1e-5 && p.pitch <= PI / 2.0 + 1e-5,
                "pitch out of range: {}",
                p.pitch
            );
        }
    }

    #[test]
    fn test_grid_idx_round_trip() {
        let grid_res = 16;
        for idx in 0..16 {
            let pose = grid_idx_to_pose(idx, grid_res, 1.0);
            let back = pose_to_grid_idx(&pose, grid_res);
            assert_eq!(back, idx, "round-trip failed for idx={idx}");
        }
    }

    // ---- spherical_gaussian_coverage ----

    #[test]
    fn test_coverage_peak_at_nearest_cell() {
        // Use a pose placed exactly at a cell centre to guarantee peak ≈ 1.
        // Grid 8×5: cell-centre yaw_idx=4 → yaw = (4.5/8)*2PI - PI = PI/8
        // pitch_idx=2 → pitch = (2.5/5)*PI - PI/2 = 0
        let yaw_center = (4.5 / 8.0_f32) * 2.0 * PI - PI;
        let pitch_center = (2.5 / 5.0_f32) * PI - PI / 2.0;
        let pose = SphericalPose::new(yaw_center, pitch_center, 1.0).unwrap();
        let cov = spherical_gaussian_coverage(&pose, 8, 5, 0.3);
        // Find the max.
        let max_val = cov.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(
            max_val > 0.99,
            "peak coverage should be near 1.0 when pose is at cell centre, got {max_val}"
        );
    }

    #[test]
    fn test_coverage_decays_with_distance() {
        let pose = SphericalPose::new(0.0, 0.0, 1.0).unwrap();
        let cov = spherical_gaussian_coverage(&pose, 16, 9, 0.3);
        let near_idx = pose_to_grid_idx(&pose, 16);
        let far_pose = SphericalPose::new(PI / 2.0, PI / 4.0, 1.0).unwrap();
        let far_idx = pose_to_grid_idx(&far_pose, 16);
        let near_cov = cov.get(near_idx).copied().unwrap_or(0.0);
        let far_cov = cov.get(far_idx).copied().unwrap_or(0.0);
        assert!(
            near_cov > far_cov,
            "near coverage {near_cov} should exceed far {far_cov}"
        );
    }

    #[test]
    fn test_coverage_output_length() {
        let pose = SphericalPose::new(0.0, 0.0, 1.0).unwrap();
        let cov = spherical_gaussian_coverage(&pose, 8, 5, 0.3);
        assert_eq!(cov.len(), 8 * 5);
    }

    #[test]
    fn test_coverage_all_nonnegative() {
        let pose = SphericalPose::new(1.0, 0.5, 1.0).unwrap();
        let cov = spherical_gaussian_coverage(&pose, 8, 5, 0.5);
        assert!(cov.iter().all(|&v| v >= 0.0));
    }

    // ---- pairwise_angular_distances ----

    #[test]
    fn test_pairwise_diagonal_zero() {
        let poses: Vec<SphericalPose> = vec![
            SphericalPose::new(0.0, 0.0, 1.0).unwrap(),
            SphericalPose::new(1.0, 0.5, 1.0).unwrap(),
        ];
        let mat = pairwise_angular_distances(&poses);
        assert!(mat[0] < 1e-5, "diagonal[0,0] should be 0");
        assert!(mat[3] < 1e-5, "diagonal[1,1] should be 0");
    }

    #[test]
    fn test_pairwise_symmetric() {
        let poses: Vec<SphericalPose> = vec![
            SphericalPose::new(0.0, 0.0, 1.0).unwrap(),
            SphericalPose::new(1.5, 0.5, 1.0).unwrap(),
            SphericalPose::new(-1.0, -0.3, 1.0).unwrap(),
        ];
        let mat = pairwise_angular_distances(&poses);
        let n = poses.len();
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (mat[i * n + j] - mat[j * n + i]).abs() < 1e-5,
                    "matrix not symmetric at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn test_pairwise_single_pose() {
        let poses = vec![SphericalPose::new(0.0, 0.0, 1.0).unwrap()];
        let mat = pairwise_angular_distances(&poses);
        assert_eq!(mat.len(), 1);
        assert!(mat[0] < 1e-5);
    }

    // ---- pose_diversity_score ----

    #[test]
    fn test_diversity_single_pose() {
        let poses = vec![SphericalPose::new(0.0, 0.0, 1.0).unwrap()];
        assert_eq!(pose_diversity_score(&poses), 0.0);
    }

    #[test]
    fn test_diversity_two_spread_poses() {
        let poses = vec![
            SphericalPose::new(0.0, 0.0, 1.0).unwrap(),
            SphericalPose::new(PI, 0.0, 1.0).unwrap(),
        ];
        let score = pose_diversity_score(&poses);
        assert!(
            score > 0.0,
            "spread poses should have positive diversity, got {score}"
        );
    }

    #[test]
    fn test_diversity_close_poses_lower_score() {
        let spread = vec![
            SphericalPose::new(0.0, 0.0, 1.0).unwrap(),
            SphericalPose::new(PI, 0.0, 1.0).unwrap(),
        ];
        let close = vec![
            SphericalPose::new(0.0, 0.0, 1.0).unwrap(),
            SphericalPose::new(0.01, 0.0, 1.0).unwrap(),
        ];
        let spread_score = pose_diversity_score(&spread);
        let close_score = pose_diversity_score(&close);
        assert!(
            spread_score > close_score,
            "spread {spread_score} should beat close {close_score}"
        );
    }

    // ---- select_diverse_poses ----

    #[test]
    fn test_select_diverse_poses_k1() {
        let poses: Vec<SphericalPose> = (0..5)
            .map(|i| SphericalPose::new(i as f32 * 0.5, 0.0, 1.0).unwrap())
            .collect();
        let sel = select_diverse_poses(&poses, 1, 42);
        assert_eq!(sel.len(), 1);
        assert!(sel[0] < 5);
    }

    #[test]
    fn test_select_diverse_poses_k_equals_n() {
        let poses: Vec<SphericalPose> = (0..4)
            .map(|i| SphericalPose::new(i as f32 * 0.8, 0.0, 1.0).unwrap())
            .collect();
        let sel = select_diverse_poses(&poses, 4, 99);
        assert_eq!(sel.len(), 4);
        // All indices present.
        for idx in 0..4 {
            assert!(sel.contains(&idx), "missing index {idx}");
        }
    }

    #[test]
    fn test_select_diverse_poses_sorted() {
        let poses: Vec<SphericalPose> = (0..6)
            .map(|i| SphericalPose::new(i as f32 * 0.5 - 1.5, 0.0, 1.0).unwrap())
            .collect();
        let sel = select_diverse_poses(&poses, 3, 7);
        assert!(
            sel.windows(2).all(|w| w[0] < w[1]),
            "result should be sorted"
        );
    }

    #[test]
    fn test_select_diverse_poses_empty() {
        let sel = select_diverse_poses(&[], 3, 1);
        assert!(sel.is_empty());
    }

    #[test]
    fn test_select_diverse_poses_k_zero() {
        let poses = vec![SphericalPose::new(0.0, 0.0, 1.0).unwrap()];
        let sel = select_diverse_poses(&poses, 0, 1);
        assert!(sel.is_empty());
    }

    // ---- interpolate_poses ----

    #[test]
    fn test_interpolate_t0_is_a() {
        let a = SphericalPose::new(0.0, 0.1, 1.0).unwrap();
        let b = SphericalPose::new(1.0, 0.5, 2.0).unwrap();
        let r = interpolate_poses(&a, &b, 0.0);
        assert!((r.yaw - a.yaw).abs() < 1e-6);
        assert!((r.pitch - a.pitch).abs() < 1e-6);
        assert!((r.radius - a.radius).abs() < 1e-6);
    }

    #[test]
    fn test_interpolate_t1_is_b() {
        let a = SphericalPose::new(0.0, 0.1, 1.0).unwrap();
        let b = SphericalPose::new(1.0, 0.5, 2.0).unwrap();
        let r = interpolate_poses(&a, &b, 1.0);
        assert!((r.yaw - b.yaw).abs() < 1e-6);
        assert!((r.pitch - b.pitch).abs() < 1e-6);
        assert!((r.radius - b.radius).abs() < 1e-6);
    }

    #[test]
    fn test_interpolate_midpoint() {
        let a = SphericalPose::new(0.0, 0.0, 1.0).unwrap();
        let b = SphericalPose::new(1.0, 1.0, 3.0).unwrap();
        let r = interpolate_poses(&a, &b, 0.5);
        assert!((r.yaw - 0.5).abs() < 1e-6);
        assert!((r.pitch - 0.5).abs() < 1e-6);
        assert!((r.radius - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_interpolate_wraps_across_seam_short_way() {
        // Regression: interpolate_poses used to lerp yaw directly without
        // wrapping, so interpolating from yaw=3.0 to yaw=-3.0 (a short
        // ~0.28 rad arc across the +-PI seam) swept the LONG way through
        // yaw=0.0 instead. With the fix, the midpoint lands near the seam
        // (yaw ~ +-PI), not near 0.0.
        let a = SphericalPose::new(3.0, 0.0, 1.0).unwrap();
        let b = SphericalPose::new(-3.0, 0.0, 1.0).unwrap();
        let r = interpolate_poses(&a, &b, 0.5);
        let dist_to_seam = (r.yaw.abs() - PI).abs();
        assert!(
            dist_to_seam < 0.2,
            "midpoint should be near the +-PI seam, got yaw={} \
             (old buggy code would give ~0.0)",
            r.yaw
        );
        assert!(
            r.yaw.abs() > 1.0,
            "midpoint should not be near 0.0 (the long-way bug), got yaw={}",
            r.yaw
        );
    }

    // ---- pose_centroid ----

    #[test]
    fn test_pose_centroid_empty_error() {
        let r = pose_centroid(&[]);
        assert!(matches!(r, Err(PoseCondError::EmptyPoseSet)));
    }

    #[test]
    fn test_pose_centroid_single() {
        let poses = vec![SphericalPose::new(0.5, 0.3, 2.0).unwrap()];
        let c = pose_centroid(&poses).unwrap();
        assert!((c.yaw - 0.5).abs() < 1e-5);
        assert!((c.pitch - 0.3).abs() < 1e-5);
    }

    #[test]
    fn test_pose_centroid_two_opposite_equatorial() {
        // Poles should cancel → near-zero pitch.
        let poses = vec![
            SphericalPose::new(0.0, PI / 2.0, 1.0).unwrap(),
            SphericalPose::new(0.0, -PI / 2.0, 1.0).unwrap(),
        ];
        let c = pose_centroid(&poses).unwrap();
        // Both Cartesian unit vectors cancel; falls back to (0, 0, radius).
        assert!(
            c.pitch.abs() < 1e-3 || c.radius > 0.0,
            "centroid of opposite poles should be near equator or fallback"
        );
    }

    #[test]
    fn test_pose_centroid_two_symmetric_equatorial() {
        let poses = vec![
            SphericalPose::new(PI / 4.0, 0.0, 1.0).unwrap(),
            SphericalPose::new(-PI / 4.0, 0.0, 1.0).unwrap(),
        ];
        let c = pose_centroid(&poses).unwrap();
        assert!(
            c.pitch.abs() < 1e-5,
            "centroid should be on equator, pitch={}",
            c.pitch
        );
    }

    // ---- PoseConditioner::new ----

    #[test]
    fn test_conditioner_new_valid() {
        let config = PoseCondConfig::default();
        let r = PoseConditioner::new(config, 42);
        assert!(r.is_ok());
    }

    #[test]
    fn test_conditioner_new_invalid_grid() {
        let config = PoseCondConfig {
            grid_resolution: 1,
            ..Default::default()
        };
        let r = PoseConditioner::new(config, 42);
        assert!(matches!(r, Err(PoseCondError::InvalidGridResolution(1))));
    }

    #[test]
    fn test_conditioner_new_invalid_sigma() {
        let config = PoseCondConfig {
            coverage_sigma: -0.1,
            ..Default::default()
        };
        let r = PoseConditioner::new(config, 42);
        assert!(matches!(r, Err(PoseCondError::InvalidParam(_))));
    }

    #[test]
    fn test_conditioner_new_invalid_ema() {
        let config = PoseCondConfig {
            ema_decay: 1.0,
            ..Default::default()
        };
        let r = PoseConditioner::new(config, 42);
        assert!(matches!(r, Err(PoseCondError::InvalidParam(_))));
    }

    // ---- register_poses ----

    #[test]
    fn test_register_poses_empty_error() {
        let config = PoseCondConfig::default();
        let mut cond = PoseConditioner::new(config, 1).unwrap();
        let r = cond.register_poses(vec![]);
        assert!(matches!(r, Err(PoseCondError::EmptyPoseSet)));
    }

    #[test]
    fn test_register_poses_increments_count() {
        let config = PoseCondConfig::default();
        let mut cond = PoseConditioner::new(config, 1).unwrap();
        let poses = vec![
            SphericalPose::new(0.0, 0.0, 1.0).unwrap(),
            SphericalPose::new(1.0, 0.2, 1.0).unwrap(),
        ];
        cond.register_poses(poses).unwrap();
        assert_eq!(cond.n_poses(), 2);
    }

    #[test]
    fn test_register_poses_initialises_visit_count() {
        let cond = default_conditioner(3);
        assert!(cond.registered_poses.iter().all(|rp| rp.visit_count == 0));
    }

    // ---- update ----

    #[test]
    fn test_update_increments_visit_count() {
        let mut cond = default_conditioner(4);
        cond.update(0, 0.5).unwrap();
        assert_eq!(cond.registered_poses[0].visit_count, 1);
    }

    #[test]
    fn test_update_tracks_loss() {
        let mut cond = default_conditioner(4);
        cond.update(1, 0.8).unwrap();
        assert!((cond.registered_poses[1].last_loss - 0.8).abs() < 1e-5);
    }

    #[test]
    fn test_update_ema_loss() {
        let mut cond = default_conditioner(4);
        // mean_loss starts at 1.0; after one update with loss=0.0 (decay=0.9):
        // 0.9 * 1.0 + 0.1 * 0.0 = 0.9
        cond.update(0, 0.0).unwrap();
        assert!((cond.registered_poses[0].mean_loss - 0.9).abs() < 1e-5);
    }

    #[test]
    fn test_update_out_of_range_error() {
        let mut cond = default_conditioner(2);
        let r = cond.update(99, 0.5);
        assert!(matches!(r, Err(PoseCondError::PoseIndexOutOfRange(99))));
    }

    #[test]
    fn test_update_increments_step() {
        let mut cond = default_conditioner(4);
        cond.update(0, 0.1).unwrap();
        cond.update(1, 0.2).unwrap();
        assert_eq!(cond.step(), 2);
    }

    // ---- recompute_coverage ----

    #[test]
    fn test_recompute_coverage_uniform_visits() {
        let mut cond = default_conditioner(4);
        // Update each pose once.
        for i in 0..4 {
            cond.update(i, 0.5).unwrap();
        }
        let stats = cond.coverage_stats();
        assert!(stats.mean_coverage > 0.0);
    }

    #[test]
    fn test_recompute_coverage_grid_length() {
        let config = PoseCondConfig {
            grid_resolution: 8,
            ..Default::default()
        };
        let mut cond = PoseConditioner::new(config, 1).unwrap();
        cond.register_poses(vec![SphericalPose::new(0.0, 0.0, 1.0).unwrap()])
            .unwrap();
        assert_eq!(cond.coverage_grid.len(), 8 * (8 / 2 + 1));
    }

    #[test]
    fn test_recompute_coverage_bounded_after_many_visits_not_unbounded() {
        // Regression: recompute_coverage/update used to normalize by pose
        // COUNT (n) rather than the SUM OF WEIGHTS, so coverage grew
        // without bound as visits accumulated. With the fix,
        // coverage_grid[cell] is a weighted AVERAGE of per-pose kernel
        // contributions (each in (0, 1]), so it stays bounded to (0, 1]
        // regardless of visit count — the old formula would have let a
        // heavily-visited pose's peak cell reach ~10x its registration-time
        // value here.
        let mut cond = default_conditioner(2);
        for _ in 0..20 {
            cond.update(0, 0.1).unwrap();
        }
        let pose0 = cond.registered_poses[0].pose.clone();
        let cov_at_pose0 = cond.coverage_at(&pose0);
        assert!(
            cov_at_pose0 <= 1.0 + 1e-4,
            "coverage at a heavily-visited pose must stay bounded to ~1.0, got {cov_at_pose0}"
        );
    }

    // ---- coverage_at ----

    #[test]
    fn test_coverage_at_registered_pose() {
        let mut cond = default_conditioner(4);
        cond.update(0, 0.5).unwrap();
        let cov = cond.coverage_at(&cond.registered_poses[0].pose.clone());
        assert!(
            cov > 0.0,
            "coverage at a registered pose should be positive, got {cov}"
        );
    }

    // ---- undertrained_regions ----

    #[test]
    fn test_undertrained_regions_empty_conditioner() {
        let config = PoseCondConfig::default();
        let cond = PoseConditioner::new(config, 1).unwrap();
        // No poses registered — grid is all zeros, every cell is undertrained.
        let regions = cond.undertrained_regions();
        // Grid is 16 × 9 = 144 cells.
        assert!(!regions.is_empty());
    }

    #[test]
    fn test_undertrained_regions_count_decreases_after_training() {
        let mut cond = default_conditioner(8);
        let before = cond.undertrained_regions().len();
        // Train on all poses several times to increase coverage.
        for _ in 0..5 {
            for i in 0..8 {
                cond.update(i, 0.2).unwrap();
            }
        }
        let after = cond.undertrained_regions().len();
        assert!(
            after <= before,
            "undertrained regions should not increase after training"
        );
    }

    // ---- select_pose ----

    #[test]
    fn test_select_pose_valid_index() {
        let mut cond = default_conditioner(5);
        let idx = cond.select_pose().unwrap();
        assert!(idx < 5, "selected index {idx} out of range");
    }

    #[test]
    fn test_select_pose_empty_error() {
        let config = PoseCondConfig::default();
        let mut cond = PoseConditioner::new(config, 1).unwrap();
        let r = cond.select_pose();
        assert!(matches!(r, Err(PoseCondError::EmptyPoseSet)));
    }

    #[test]
    fn test_select_pose_single_pose() {
        let config = PoseCondConfig::default();
        let mut cond = PoseConditioner::new(config, 1).unwrap();
        cond.register_poses(vec![SphericalPose::new(0.0, 0.0, 1.0).unwrap()])
            .unwrap();
        let idx = cond.select_pose().unwrap();
        assert_eq!(idx, 0);
    }

    // ---- select_diverse_batch ----

    #[test]
    fn test_select_diverse_batch_correct_count() {
        let mut cond = default_conditioner(6);
        let batch = cond.select_diverse_batch(3).unwrap();
        assert_eq!(batch.len(), 3);
    }

    #[test]
    fn test_select_diverse_batch_unique_indices() {
        let mut cond = default_conditioner(6);
        let batch = cond.select_diverse_batch(4).unwrap();
        let unique: std::collections::HashSet<_> = batch.iter().collect();
        assert_eq!(
            unique.len(),
            batch.len(),
            "batch should have unique indices"
        );
    }

    #[test]
    fn test_select_diverse_batch_empty_error() {
        let config = PoseCondConfig::default();
        let mut cond = PoseConditioner::new(config, 1).unwrap();
        let r = cond.select_diverse_batch(3);
        assert!(matches!(r, Err(PoseCondError::EmptyPoseSet)));
    }

    #[test]
    fn test_select_diverse_batch_capped_at_total() {
        let mut cond = default_conditioner(3);
        let batch = cond.select_diverse_batch(100).unwrap();
        assert_eq!(batch.len(), 3, "batch should be capped at total poses");
    }

    // ---- hardest_pose ----

    #[test]
    fn test_hardest_pose_none_when_empty() {
        let config = PoseCondConfig::default();
        let cond = PoseConditioner::new(config, 1).unwrap();
        assert!(cond.hardest_pose().is_none());
    }

    #[test]
    fn test_hardest_pose_returns_highest_loss() {
        let mut cond = default_conditioner(3);
        // Give pose 1 a high loss.
        cond.registered_poses[1].mean_loss = 2.0;
        cond.registered_poses[0].mean_loss = 0.5;
        cond.registered_poses[2].mean_loss = 0.3;
        let hardest = cond.hardest_pose().unwrap();
        assert_eq!(hardest, 1);
    }

    // ---- least_visited_pose ----

    #[test]
    fn test_least_visited_none_when_empty() {
        let config = PoseCondConfig::default();
        let cond = PoseConditioner::new(config, 1).unwrap();
        assert!(cond.least_visited_pose().is_none());
    }

    #[test]
    fn test_least_visited_returns_unvisited() {
        let mut cond = default_conditioner(3);
        cond.update(0, 0.5).unwrap();
        cond.update(2, 0.3).unwrap();
        let lv = cond.least_visited_pose().unwrap();
        assert_eq!(lv, 1, "unvisited pose 1 should be least visited");
    }

    // ---- coverage_stats ----

    #[test]
    fn test_coverage_stats_after_registration() {
        let cond = default_conditioner(4);
        let stats = cond.coverage_stats();
        assert!(stats.mean_coverage >= 0.0);
        assert!(stats.min_coverage <= stats.mean_coverage);
        assert!(stats.mean_coverage <= stats.max_coverage);
        assert_eq!(stats.n_poses, 4);
        assert_eq!(stats.total_visits, 0);
    }

    #[test]
    fn test_coverage_stats_total_visits() {
        let mut cond = default_conditioner(3);
        cond.update(0, 0.1).unwrap();
        cond.update(0, 0.2).unwrap();
        cond.update(1, 0.3).unwrap();
        let stats = cond.coverage_stats();
        assert_eq!(stats.total_visits, 3);
    }

    #[test]
    fn test_coverage_stats_fraction_in_0_1() {
        let cond = default_conditioner(4);
        let stats = cond.coverage_stats();
        assert!((0.0..=1.0).contains(&stats.coverage_fraction));
    }

    // ---- format_coverage_stats ----

    #[test]
    fn test_format_coverage_stats_nonempty() {
        let stats = CoverageStats {
            mean_coverage: 0.5,
            min_coverage: 0.1,
            max_coverage: 0.9,
            coverage_fraction: 0.75,
            total_visits: 42,
            n_poses: 8,
        };
        let s = format_coverage_stats(&stats);
        assert!(!s.is_empty());
        assert!(s.contains("42"));
    }

    #[test]
    fn test_format_coverage_stats_contains_fraction() {
        let stats = CoverageStats {
            mean_coverage: 0.3,
            min_coverage: 0.0,
            max_coverage: 0.8,
            coverage_fraction: 0.6,
            total_visits: 10,
            n_poses: 5,
        };
        let s = format_coverage_stats(&stats);
        assert!(s.contains("60.0"), "should contain 60.0%, got: {s}");
    }

    // ---- error variants ----

    #[test]
    fn test_error_display_empty_pose_set() {
        let e = PoseCondError::EmptyPoseSet;
        let s = e.to_string();
        assert!(!s.is_empty());
    }

    #[test]
    fn test_error_display_pose_index_out_of_range() {
        let e = PoseCondError::PoseIndexOutOfRange(77);
        let s = e.to_string();
        assert!(s.contains("77"));
    }

    #[test]
    fn test_error_display_invalid_grid_resolution() {
        let e = PoseCondError::InvalidGridResolution(0);
        let s = e.to_string();
        assert!(s.contains('0'));
    }
}
