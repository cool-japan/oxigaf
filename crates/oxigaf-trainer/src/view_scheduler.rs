//! Adaptive view scheduler for 3D Gaussian Splatting training.
//!
//! Selects which camera views to render/optimize each training step,
//! prioritising high-loss ("hard") views while guaranteeing full coverage.

use thiserror::Error;

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

/// Errors produced by the view scheduler.
#[derive(Debug, Error)]
pub enum ViewSchedulerError {
    #[error("No views registered")]
    NoViews,
    #[error("View index {0} out of range")]
    ViewIndexOutOfRange(usize),
    #[error("Batch size {batch} exceeds total views {total}")]
    BatchExceedsViews { batch: usize, total: usize },
    #[error("Invalid EMA alpha {0}: must be in (0, 1]")]
    InvalidAlpha(f32),
    #[error("Priority exponent must be positive, got {0}")]
    InvalidExponent(f32),
}

// ---------------------------------------------------------------------------
// View metadata
// ---------------------------------------------------------------------------

/// Information about a single camera view.
#[derive(Debug, Clone)]
pub struct ViewInfo {
    pub index: usize,
    /// Camera position in world space.
    pub position: [f32; 3],
    /// Camera orientation (look direction).
    pub direction: [f32; 3],
    /// Elevation angle (radians) from horizontal: positive = looking down.
    pub elevation: f32,
    /// Azimuth angle (radians): 0 = front, increases clockwise.
    pub azimuth: f32,
}

/// Statistics tracked per view.
#[derive(Debug, Clone)]
pub struct ViewStats {
    pub index: usize,
    /// EMA of PSNR (higher = better reconstruction).
    pub psnr_ema: f32,
    /// EMA of loss (lower = better).
    pub loss_ema: f32,
    /// How many times this view has been selected.
    pub visit_count: u64,
    pub last_visit_step: u64,
    /// Current priority weight (higher = selected more often).
    pub priority: f32,
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for the adaptive view scheduler.
#[derive(Debug, Clone)]
pub struct ViewSchedulerConfig {
    /// EMA alpha for loss smoothing. Default: 0.1
    pub loss_ema_alpha: f32,
    /// EMA alpha for PSNR smoothing. Default: 0.1
    pub psnr_ema_alpha: f32,
    /// Priority = loss_ema ^ priority_exponent. Default: 2.0
    pub priority_exponent: f32,
    /// Fraction of batch drawn from high-priority views. Default: 0.7
    pub priority_fraction: f32,
    /// Minimum visits before a view is eligible for priority selection. Default: 2
    pub min_visits_for_priority: u64,
    /// Staleness bonus added when views are not visited recently. Default: 0.2
    pub staleness_bonus: f32,
    /// Steps between full round-robin coverage passes. Default: 100
    pub coverage_period: u64,
}

impl Default for ViewSchedulerConfig {
    fn default() -> Self {
        Self {
            loss_ema_alpha: 0.1,
            psnr_ema_alpha: 0.1,
            priority_exponent: 2.0,
            priority_fraction: 0.7,
            min_visits_for_priority: 2,
            staleness_bonus: 0.2,
            coverage_period: 100,
        }
    }
}

// ---------------------------------------------------------------------------
// PRNG helpers (xorshift64, no rand crate)
// ---------------------------------------------------------------------------

#[inline]
fn xorshift64(s: &mut u64) -> u64 {
    let mut x = *s;
    if x == 0 {
        x = 1;
    }
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *s = x;
    x
}

#[inline]
fn xorshift_f32(s: &mut u64) -> f32 {
    (xorshift64(s) >> 11) as f32 / (1u64 << 53) as f32
}

// ---------------------------------------------------------------------------
// ViewScheduler
// ---------------------------------------------------------------------------

/// Adaptive view scheduler that prioritises high-loss camera views.
pub struct ViewScheduler {
    config: ViewSchedulerConfig,
    views: Vec<ViewInfo>,
    stats: Vec<ViewStats>,
    current_step: u64,
    /// Cursor for round-robin coverage passes.
    round_robin_cursor: usize,
}

impl ViewScheduler {
    // ------------------------------------------------------------------
    // Constructors
    // ------------------------------------------------------------------

    /// Create a scheduler for `num_views` anonymous views (simple indices).
    pub fn with_count(
        num_views: usize,
        config: ViewSchedulerConfig,
    ) -> Result<Self, ViewSchedulerError> {
        if num_views == 0 {
            return Err(ViewSchedulerError::NoViews);
        }
        Self::validate_config(&config)?;

        let views: Vec<ViewInfo> = (0..num_views)
            .map(|i| ViewInfo {
                index: i,
                position: [0.0; 3],
                direction: [0.0, 0.0, -1.0],
                elevation: 0.0,
                azimuth: 0.0,
            })
            .collect();

        let stats = (0..num_views)
            .map(|i| ViewStats {
                index: i,
                psnr_ema: 0.0,
                loss_ema: 0.0,
                visit_count: 0,
                last_visit_step: 0,
                priority: 1.0,
            })
            .collect();

        Ok(Self {
            config,
            views,
            stats,
            current_step: 0,
            round_robin_cursor: 0,
        })
    }

    /// Create with full view metadata.
    pub fn with_views(
        views: Vec<ViewInfo>,
        config: ViewSchedulerConfig,
    ) -> Result<Self, ViewSchedulerError> {
        if views.is_empty() {
            return Err(ViewSchedulerError::NoViews);
        }
        Self::validate_config(&config)?;

        let n = views.len();
        let stats = (0..n)
            .map(|i| ViewStats {
                index: i,
                psnr_ema: 0.0,
                loss_ema: 0.0,
                visit_count: 0,
                last_visit_step: 0,
                priority: 1.0,
            })
            .collect();

        Ok(Self {
            config,
            views,
            stats,
            current_step: 0,
            round_robin_cursor: 0,
        })
    }

    fn validate_config(config: &ViewSchedulerConfig) -> Result<(), ViewSchedulerError> {
        if config.loss_ema_alpha <= 0.0 || config.loss_ema_alpha > 1.0 {
            return Err(ViewSchedulerError::InvalidAlpha(config.loss_ema_alpha));
        }
        if config.psnr_ema_alpha <= 0.0 || config.psnr_ema_alpha > 1.0 {
            return Err(ViewSchedulerError::InvalidAlpha(config.psnr_ema_alpha));
        }
        if config.priority_exponent <= 0.0 {
            return Err(ViewSchedulerError::InvalidExponent(
                config.priority_exponent,
            ));
        }
        Ok(())
    }

    // ------------------------------------------------------------------
    // Observation recording
    // ------------------------------------------------------------------

    /// Record loss and PSNR after rendering a view.
    pub fn record_observation(
        &mut self,
        view_index: usize,
        loss: f32,
        psnr: f32,
    ) -> Result<(), ViewSchedulerError> {
        if view_index >= self.stats.len() {
            return Err(ViewSchedulerError::ViewIndexOutOfRange(view_index));
        }
        let s = &mut self.stats[view_index];

        if s.visit_count == 0 {
            // First observation: initialise EMA directly.
            s.loss_ema = loss;
            s.psnr_ema = psnr;
        } else {
            let la = self.config.loss_ema_alpha;
            let pa = self.config.psnr_ema_alpha;
            s.loss_ema = la * loss + (1.0 - la) * s.loss_ema;
            s.psnr_ema = pa * psnr + (1.0 - pa) * s.psnr_ema;
        }

        s.visit_count += 1;
        s.last_visit_step = self.current_step;

        self.update_priorities();
        Ok(())
    }

    // ------------------------------------------------------------------
    // Step counter
    // ------------------------------------------------------------------

    /// Advance the training step counter.
    pub fn step(&mut self) {
        self.current_step += 1;
    }

    /// Current step.
    pub fn current_step(&self) -> u64 {
        self.current_step
    }

    // ------------------------------------------------------------------
    // Batch selection
    // ------------------------------------------------------------------

    /// Select a batch of view indices for the next training iteration.
    pub fn select_batch(
        &mut self,
        batch_size: usize,
        rng_seed: u64,
    ) -> Result<Vec<usize>, ViewSchedulerError> {
        let total = self.views.len();

        if batch_size == 0 {
            return Ok(Vec::new());
        }
        if batch_size > total {
            return Err(ViewSchedulerError::BatchExceedsViews {
                batch: batch_size,
                total,
            });
        }

        // Update priorities before sampling.
        self.update_priorities();

        // Determine if this is a coverage step.
        let is_coverage = self
            .current_step
            .is_multiple_of(self.config.coverage_period);

        let selected = if is_coverage {
            self.select_round_robin(batch_size)
        } else {
            self.select_priority(batch_size, rng_seed)
        };

        // Record visits.
        for &idx in &selected {
            let s = &mut self.stats[idx];
            s.visit_count += 1;
            s.last_visit_step = self.current_step;
        }

        Ok(selected)
    }

    /// Round-robin selection advancing the cursor.
    fn select_round_robin(&mut self, batch_size: usize) -> Vec<usize> {
        let total = self.views.len();
        let mut result = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            result.push(self.round_robin_cursor % total);
            self.round_robin_cursor = (self.round_robin_cursor + 1) % total;
        }
        result
    }

    /// Priority-based selection with optional uniform fill.
    fn select_priority(&mut self, batch_size: usize, rng_seed: u64) -> Vec<usize> {
        let total = self.views.len();
        let mut rng = rng_seed ^ 0xCAFEBABE;

        let priority_count = (batch_size as f32 * self.config.priority_fraction).round() as usize;
        let priority_count = priority_count.min(batch_size);
        let random_count = batch_size - priority_count;

        // Eligible views: visit_count >= min_visits.
        let eligible: Vec<usize> = (0..total)
            .filter(|&i| self.stats[i].visit_count >= self.config.min_visits_for_priority)
            .collect();

        let mut selected: Vec<usize> = Vec::with_capacity(batch_size);
        let mut used = vec![false; total];

        // Draw priority_count from eligible views by weighted sampling.
        let mut deficit = 0usize;
        if !eligible.is_empty() {
            let drawn =
                self.weighted_sample_without_replacement(&eligible, priority_count, &mut rng);
            for idx in drawn {
                used[idx] = true;
                selected.push(idx);
            }
            if selected.len() < priority_count {
                deficit = priority_count - selected.len();
            }
        } else {
            deficit = priority_count;
        }

        // Fill deficit from round-robin.
        for _ in 0..deficit {
            let rr_idx = self.round_robin_cursor % total;
            self.round_robin_cursor = (self.round_robin_cursor + 1) % total;
            if !used[rr_idx] {
                used[rr_idx] = true;
                selected.push(rr_idx);
            } else {
                // Find next unused in round-robin order.
                let mut found = false;
                for offset in 1..total {
                    let candidate = (rr_idx + offset) % total;
                    if !used[candidate] {
                        used[candidate] = true;
                        selected.push(candidate);
                        found = true;
                        break;
                    }
                }
                if !found {
                    // All views already used — just pick round-robin anyway.
                    selected.push(rr_idx);
                }
            }
        }

        // Fill random_count from uniform sampling over unused views.
        let mut unused: Vec<usize> = (0..total).filter(|&i| !used[i]).collect();
        // Fisher-Yates shuffle on unused, then take first random_count.
        for i in (1..unused.len()).rev() {
            let j = xorshift64(&mut rng) as usize % (i + 1);
            unused.swap(i, j);
        }
        let take = random_count.min(unused.len());
        for &idx in &unused[..take] {
            used[idx] = true;
            selected.push(idx);
        }

        // If still short (shouldn't happen normally), pad from round-robin.
        while selected.len() < batch_size {
            let rr_idx = self.round_robin_cursor % total;
            self.round_robin_cursor = (self.round_robin_cursor + 1) % total;
            selected.push(rr_idx);
        }

        selected
    }

    /// Weighted sampling without replacement using cumulative weights + binary search.
    fn weighted_sample_without_replacement(
        &self,
        candidates: &[usize],
        count: usize,
        rng: &mut u64,
    ) -> Vec<usize> {
        let count = count.min(candidates.len());
        let mut weights: Vec<f32> = candidates
            .iter()
            .map(|&i| self.stats[i].priority.max(0.0))
            .collect();

        let mut result = Vec::with_capacity(count);

        for _ in 0..count {
            let total_w: f32 = weights.iter().sum();
            if total_w <= 0.0 {
                // All weights exhausted — stop.
                break;
            }
            let r = xorshift_f32(rng) * total_w;

            // Build cumsum and find index via scan.
            let mut cumsum = 0.0f32;
            let mut chosen = candidates.len() - 1; // fallback
            for (i, &w) in weights.iter().enumerate() {
                cumsum += w;
                if r < cumsum {
                    chosen = i;
                    break;
                }
            }
            result.push(candidates[chosen]);
            weights[chosen] = 0.0; // remove from future draws
        }

        result
    }

    // ------------------------------------------------------------------
    // Priority update
    // ------------------------------------------------------------------

    /// Recompute priority weights for all views.
    fn update_priorities(&mut self) {
        let exponent = self.config.priority_exponent;
        let staleness_bonus = self.config.staleness_bonus;
        let coverage_period = self.config.coverage_period.max(1) as f32;
        let current_step = self.current_step;

        for s in &mut self.stats {
            let base = s.loss_ema.powf(exponent);
            let staleness_steps = current_step.saturating_sub(s.last_visit_step) as f32;
            let staleness = staleness_bonus * staleness_steps / coverage_period;
            s.priority = (base + staleness).max(0.0);
        }
    }

    // ------------------------------------------------------------------
    // Accessors
    // ------------------------------------------------------------------

    /// Stats for a specific view.
    pub fn view_stats(&self, index: usize) -> Option<&ViewStats> {
        self.stats.get(index)
    }

    /// All view stats.
    pub fn all_stats(&self) -> &[ViewStats] {
        &self.stats
    }

    /// View with highest loss EMA.
    pub fn hardest_view(&self) -> Option<&ViewStats> {
        self.stats
            .iter()
            .filter(|s| s.visit_count > 0)
            .max_by(|a, b| {
                a.loss_ema
                    .partial_cmp(&b.loss_ema)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// View with lowest loss EMA.
    pub fn easiest_view(&self) -> Option<&ViewStats> {
        self.stats
            .iter()
            .filter(|s| s.visit_count > 0)
            .min_by(|a, b| {
                a.loss_ema
                    .partial_cmp(&b.loss_ema)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Mean loss EMA across all visited views.
    pub fn mean_loss(&self) -> f32 {
        let visited: Vec<f32> = self
            .stats
            .iter()
            .filter(|s| s.visit_count > 0)
            .map(|s| s.loss_ema)
            .collect();

        if visited.is_empty() {
            return 0.0;
        }
        visited.iter().sum::<f32>() / visited.len() as f32
    }

    /// Fraction of views visited at least once.
    pub fn coverage_fraction(&self) -> f32 {
        if self.stats.is_empty() {
            return 0.0;
        }
        let visited = self.stats.iter().filter(|s| s.visit_count > 0).count();
        visited as f32 / self.stats.len() as f32
    }

    /// Human-readable scheduling summary.
    pub fn format_summary(&self) -> String {
        let total = self.stats.len();
        let visited = self.stats.iter().filter(|s| s.visit_count > 0).count();
        let mean_loss = self.mean_loss();
        let max_loss = self
            .stats
            .iter()
            .map(|s| s.loss_ema)
            .fold(f32::NEG_INFINITY, f32::max);
        let coverage = self.coverage_fraction() * 100.0;
        let step = self.current_step;

        let hardest = self
            .hardest_view()
            .map(|s| format!("{}", s.index))
            .unwrap_or_else(|| "none".to_string());
        let easiest = self
            .easiest_view()
            .map(|s| format!("{}", s.index))
            .unwrap_or_else(|| "none".to_string());

        format!(
            "ViewScheduler [step={step}] views={total} visited={visited} \
             coverage={coverage:.1}% mean_loss={mean_loss:.4} max_loss={max_loss:.4} \
             hardest={hardest} easiest={easiest}"
        )
    }
}

// ---------------------------------------------------------------------------
// Angular coverage analysis
// ---------------------------------------------------------------------------

/// Summary of how well selected views cover the view sphere.
#[derive(Debug, Clone)]
pub struct CoverageReport {
    pub num_views: usize,
    /// Mean angular distance to nearest neighbour (radians).
    pub mean_nn_distance: f32,
    /// Approximate largest uncovered angular gap (radians).
    pub max_gap: f32,
    /// Coverage uniformity in [0, 1]: 1 = perfectly uniform.
    pub uniformity: f32,
}

/// Great-circle angular distance between two (azimuth, elevation) pairs (radians).
pub fn angular_distance(az1: f32, el1: f32, az2: f32, el2: f32) -> f32 {
    let dot = el1.sin() * el2.sin() + el1.cos() * el2.cos() * (az2 - az1).cos();
    dot.clamp(-1.0, 1.0).acos()
}

/// Analyse how well the given views cover the sphere.
pub fn analyze_view_coverage(views: &[ViewInfo]) -> CoverageReport {
    let n = views.len();
    if n == 0 {
        return CoverageReport {
            num_views: 0,
            mean_nn_distance: 0.0,
            max_gap: 0.0,
            uniformity: 0.0,
        };
    }
    if n == 1 {
        return CoverageReport {
            num_views: 1,
            mean_nn_distance: 0.0,
            max_gap: 0.0,
            uniformity: 1.0,
        };
    }

    // Compute nearest-neighbour distance for each view.
    let mut nn_distances: Vec<f32> = Vec::with_capacity(n);
    for i in 0..n {
        let mut min_dist = f32::INFINITY;
        for j in 0..n {
            if i == j {
                continue;
            }
            let d = angular_distance(
                views[i].azimuth,
                views[i].elevation,
                views[j].azimuth,
                views[j].elevation,
            );
            if d < min_dist {
                min_dist = d;
            }
        }
        nn_distances.push(min_dist);
    }

    let mean_nn: f32 = nn_distances.iter().sum::<f32>() / n as f32;
    let max_gap: f32 = nn_distances
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);

    // Uniformity = 1 / (1 + std(nn_distances)).
    let variance: f32 = nn_distances
        .iter()
        .map(|&d| (d - mean_nn) * (d - mean_nn))
        .sum::<f32>()
        / n as f32;
    let std_dev = variance.sqrt();
    let uniformity = 1.0 / (1.0 + std_dev);

    CoverageReport {
        num_views: n,
        mean_nn_distance: mean_nn,
        max_gap,
        uniformity,
    }
}

// ---------------------------------------------------------------------------
// View importance heuristics
// ---------------------------------------------------------------------------

/// Compute view importance from geometry alone (before any training).
///
/// Importance is based on equatorial proximity and spacing.
/// Values are normalised to sum to 1.
pub fn geometric_importance(views: &[ViewInfo]) -> Vec<f32> {
    let n = views.len();
    if n == 0 {
        return Vec::new();
    }

    // Minimum angular distance to any neighbour, per view.
    let min_dists: Vec<f32> = (0..n)
        .map(|i| {
            if n == 1 {
                return 0.0;
            }
            (0..n)
                .filter(|&j| j != i)
                .map(|j| {
                    angular_distance(
                        views[i].azimuth,
                        views[i].elevation,
                        views[j].azimuth,
                        views[j].elevation,
                    )
                })
                .fold(f32::INFINITY, f32::min)
        })
        .collect();

    let pi = std::f32::consts::PI;
    let mut importances: Vec<f32> = views
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let cos_el = v.elevation.cos();
            1.0 + cos_el * cos_el * (1.0 + min_dists[i] / pi)
        })
        .collect();

    // Normalise to sum to 1.
    let total: f32 = importances.iter().sum();
    if total > 0.0 {
        for imp in &mut importances {
            *imp /= total;
        }
    }

    importances
}

/// Create a grid of uniformly distributed view angles.
///
/// * `azimuth`: `num_azimuth` values spanning [0, 2π) (no endpoint).
/// * `elevation`: `num_elevation` values spanning [-π/4, π/4].
/// * `index`: `i * num_elevation + j`.
pub fn uniform_view_angles(num_azimuth: usize, num_elevation: usize) -> Vec<ViewInfo> {
    if num_azimuth == 0 || num_elevation == 0 {
        return Vec::new();
    }

    let pi = std::f32::consts::PI;
    let two_pi = 2.0 * pi;

    let azimuths: Vec<f32> = if num_azimuth == 1 {
        vec![0.0]
    } else {
        (0..num_azimuth)
            .map(|i| i as f32 * two_pi / num_azimuth as f32)
            .collect()
    };

    let el_start = -pi / 4.0;
    let el_end = pi / 4.0;
    let elevations: Vec<f32> = if num_elevation == 1 {
        vec![(el_start + el_end) / 2.0]
    } else {
        (0..num_elevation)
            .map(|j| el_start + j as f32 * (el_end - el_start) / (num_elevation - 1) as f32)
            .collect()
    };

    let mut views = Vec::with_capacity(num_azimuth * num_elevation);
    for (i, &az) in azimuths.iter().enumerate() {
        for (j, &el) in elevations.iter().enumerate() {
            let look_dir = [(-az).sin() * el.cos(), -el.sin(), (-az).cos() * el.cos()];
            views.push(ViewInfo {
                index: i * num_elevation + j,
                position: [0.0; 3],
                direction: look_dir,
                elevation: el,
                azimuth: az,
            });
        }
    }
    views
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // 1. with_count creates scheduler with correct num_views
    #[test]
    fn test_with_count_creates_correct_num_views() -> Result<(), ViewSchedulerError> {
        let sched = ViewScheduler::with_count(5, ViewSchedulerConfig::default())?;
        assert_eq!(sched.all_stats().len(), 5);
        Ok(())
    }

    // 2. with_count 0 views → NoViews
    #[test]
    fn test_with_count_zero_views_returns_error() {
        let result = ViewScheduler::with_count(0, ViewSchedulerConfig::default());
        assert!(matches!(result, Err(ViewSchedulerError::NoViews)));
    }

    // 3. record_observation first visit: ema = loss
    #[test]
    fn test_record_first_observation_initialises_ema() -> Result<(), ViewSchedulerError> {
        let mut sched = ViewScheduler::with_count(3, ViewSchedulerConfig::default())?;
        sched.record_observation(1, 0.5, 25.0)?;
        let s = sched
            .view_stats(1)
            .ok_or(ViewSchedulerError::ViewIndexOutOfRange(1))?;
        assert!(
            (s.loss_ema - 0.5).abs() < 1e-6,
            "loss_ema should equal loss on first visit"
        );
        assert!(
            (s.psnr_ema - 25.0).abs() < 1e-6,
            "psnr_ema should equal psnr on first visit"
        );
        Ok(())
    }

    // 4. record_observation second visit updates EMA
    #[test]
    fn test_record_second_observation_updates_ema() -> Result<(), ViewSchedulerError> {
        let config = ViewSchedulerConfig {
            loss_ema_alpha: 0.5,
            ..ViewSchedulerConfig::default()
        };
        let mut sched = ViewScheduler::with_count(3, config)?;
        sched.record_observation(0, 1.0, 20.0)?;
        sched.record_observation(0, 0.0, 30.0)?;
        let s = sched
            .view_stats(0)
            .ok_or(ViewSchedulerError::ViewIndexOutOfRange(0))?;
        // EMA: 0.5 * 0.0 + 0.5 * 1.0 = 0.5
        assert!(
            (s.loss_ema - 0.5).abs() < 1e-6,
            "EMA not updated correctly: {}",
            s.loss_ema
        );
        Ok(())
    }

    // 5. record_observation out-of-range → ViewIndexOutOfRange
    #[test]
    fn test_record_observation_out_of_range_returns_error() -> Result<(), ViewSchedulerError> {
        let mut sched = ViewScheduler::with_count(3, ViewSchedulerConfig::default())?;
        let result = sched.record_observation(99, 0.1, 30.0);
        assert!(matches!(
            result,
            Err(ViewSchedulerError::ViewIndexOutOfRange(99))
        ));
        Ok(())
    }

    // 6. step increments current_step
    #[test]
    fn test_step_increments_counter() -> Result<(), ViewSchedulerError> {
        let mut sched = ViewScheduler::with_count(2, ViewSchedulerConfig::default())?;
        assert_eq!(sched.current_step(), 0);
        sched.step();
        assert_eq!(sched.current_step(), 1);
        sched.step();
        assert_eq!(sched.current_step(), 2);
        Ok(())
    }

    // 7. select_batch batch size 1 returns 1 view
    #[test]
    fn test_select_batch_size_one() -> Result<(), ViewSchedulerError> {
        let mut sched = ViewScheduler::with_count(5, ViewSchedulerConfig::default())?;
        let batch = sched.select_batch(1, 42)?;
        assert_eq!(batch.len(), 1);
        assert!(batch[0] < 5);
        Ok(())
    }

    // 8. select_batch batch > total → BatchExceedsViews
    #[test]
    fn test_select_batch_exceeds_total_returns_error() -> Result<(), ViewSchedulerError> {
        let mut sched = ViewScheduler::with_count(3, ViewSchedulerConfig::default())?;
        let result = sched.select_batch(10, 42);
        assert!(matches!(
            result,
            Err(ViewSchedulerError::BatchExceedsViews {
                batch: 10,
                total: 3
            })
        ));
        Ok(())
    }

    // 9. select_batch coverage step (step=0): returns round-robin views
    #[test]
    fn test_select_batch_coverage_step_uses_round_robin() -> Result<(), ViewSchedulerError> {
        let config = ViewSchedulerConfig {
            coverage_period: 100,
            ..ViewSchedulerConfig::default()
        };
        let mut sched = ViewScheduler::with_count(5, config)?;
        // step=0 is a coverage step (0 % 100 == 0)
        assert_eq!(sched.current_step(), 0);
        let batch = sched.select_batch(3, 42)?;
        assert_eq!(batch.len(), 3);
        // Round-robin from cursor=0 should give [0,1,2]
        assert_eq!(batch, vec![0, 1, 2]);
        Ok(())
    }

    // 10. hardest_view returns view with highest loss_ema
    #[test]
    fn test_hardest_view() -> Result<(), ViewSchedulerError> {
        let mut sched = ViewScheduler::with_count(4, ViewSchedulerConfig::default())?;
        sched.record_observation(0, 0.1, 30.0)?;
        sched.record_observation(1, 0.9, 15.0)?;
        sched.record_observation(2, 0.5, 22.0)?;
        let hardest = sched.hardest_view().ok_or(ViewSchedulerError::NoViews)?;
        assert_eq!(hardest.index, 1, "Expected view 1 to be hardest");
        Ok(())
    }

    // 11. easiest_view returns view with lowest loss_ema
    #[test]
    fn test_easiest_view() -> Result<(), ViewSchedulerError> {
        let mut sched = ViewScheduler::with_count(4, ViewSchedulerConfig::default())?;
        sched.record_observation(0, 0.1, 30.0)?;
        sched.record_observation(1, 0.9, 15.0)?;
        sched.record_observation(2, 0.5, 22.0)?;
        let easiest = sched.easiest_view().ok_or(ViewSchedulerError::NoViews)?;
        assert_eq!(easiest.index, 0, "Expected view 0 to be easiest");
        Ok(())
    }

    // 12. mean_loss correct average
    #[test]
    fn test_mean_loss_correct_average() -> Result<(), ViewSchedulerError> {
        let mut sched = ViewScheduler::with_count(3, ViewSchedulerConfig::default())?;
        sched.record_observation(0, 0.2, 28.0)?;
        sched.record_observation(1, 0.6, 18.0)?;
        sched.record_observation(2, 1.0, 10.0)?;
        let mean = sched.mean_loss();
        let expected = (0.2 + 0.6 + 1.0) / 3.0;
        assert!(
            (mean - expected).abs() < 1e-5,
            "mean_loss={mean}, expected={expected}"
        );
        Ok(())
    }

    // 13. coverage_fraction increases as views are visited
    #[test]
    fn test_coverage_fraction_increases() -> Result<(), ViewSchedulerError> {
        let mut sched = ViewScheduler::with_count(4, ViewSchedulerConfig::default())?;
        assert!((sched.coverage_fraction() - 0.0).abs() < 1e-6);
        sched.record_observation(0, 0.5, 20.0)?;
        assert!((sched.coverage_fraction() - 0.25).abs() < 1e-6);
        sched.record_observation(1, 0.5, 20.0)?;
        assert!((sched.coverage_fraction() - 0.5).abs() < 1e-6);
        Ok(())
    }

    // 14. coverage_fraction 100% after visiting all views
    #[test]
    fn test_coverage_fraction_full_after_all_visited() -> Result<(), ViewSchedulerError> {
        let mut sched = ViewScheduler::with_count(3, ViewSchedulerConfig::default())?;
        for i in 0..3 {
            sched.record_observation(i, 0.5, 20.0)?;
        }
        assert!((sched.coverage_fraction() - 1.0).abs() < 1e-6);
        Ok(())
    }

    // 15. angular_distance same position → 0.0
    #[test]
    fn test_angular_distance_same_position() {
        let d = angular_distance(1.0, 0.5, 1.0, 0.5);
        assert!(
            d.abs() < 1e-5,
            "Same position distance should be 0, got {d}"
        );
    }

    // 16. angular_distance opposite hemispheres → π
    #[test]
    fn test_angular_distance_opposite_hemispheres() {
        let pi = std::f32::consts::PI;
        // North pole vs south pole: el = π/2 vs -π/2
        let d = angular_distance(0.0, pi / 2.0, 0.0, -pi / 2.0);
        assert!(
            (d - pi).abs() < 1e-5,
            "Opposite poles distance should be π, got {d}"
        );
    }

    // 17. analyze_view_coverage uniformity = 1 for single view
    #[test]
    fn test_analyze_view_coverage_single_view() {
        let views = vec![ViewInfo {
            index: 0,
            position: [0.0; 3],
            direction: [0.0, 0.0, -1.0],
            elevation: 0.0,
            azimuth: 0.0,
        }];
        let report = analyze_view_coverage(&views);
        assert!(
            (report.uniformity - 1.0).abs() < 1e-6,
            "uniformity={}",
            report.uniformity
        );
        assert_eq!(report.num_views, 1);
    }

    // 18. uniform_view_angles returns num_azimuth * num_elevation views
    #[test]
    fn test_uniform_view_angles_count() {
        let views = uniform_view_angles(4, 3);
        assert_eq!(
            views.len(),
            12,
            "Expected 4*3=12 views, got {}",
            views.len()
        );
    }

    // 19. geometric_importance sums to 1.0
    #[test]
    fn test_geometric_importance_sums_to_one() {
        let views = uniform_view_angles(4, 3);
        let importances = geometric_importance(&views);
        let total: f32 = importances.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-5,
            "Importances sum={total}, expected 1.0"
        );
    }

    // 20. format_summary returns non-empty string
    #[test]
    fn test_format_summary_non_empty() -> Result<(), ViewSchedulerError> {
        let sched = ViewScheduler::with_count(5, ViewSchedulerConfig::default())?;
        let s = sched.format_summary();
        assert!(!s.is_empty());
        Ok(())
    }
}
