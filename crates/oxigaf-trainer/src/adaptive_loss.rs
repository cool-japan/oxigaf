//! Adaptive loss weight adjustment for multi-objective training.
//!
//! Provides dynamic weighting strategies that balance competing objectives
//! (photometric reconstruction, perceptual similarity, regularizers, etc.)
//! during Gaussian avatar training. Strategies include fixed weights,
//! gradient-norm balancing, loss-ratio balancing, and uncertainty weighting.

use std::collections::{HashMap, VecDeque};

// ──────────────────────────────────────────────────────────────────────────────
// LossComponent
// ──────────────────────────────────────────────────────────────────────────────

/// Identifies an individual loss term in the multi-objective training objective.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum LossComponent {
    /// L1 / SSIM image reconstruction loss.
    Photometric,
    /// LPIPS / perceptual similarity loss.
    Perceptual,
    /// Normal map smoothness regularizer.
    NormalConsistency,
    /// Keeps Gaussian positions close to the FLAME mesh surface.
    PositionRegularizer,
    /// Penalizes Gaussian scales that are too large.
    ScaleRegularizer,
    /// Sparsity penalty on opacity values.
    OpacityRegularizer,
    /// Diffusion-model pseudo-GT loss (SDS / distillation).
    DiffusionTarget,
    /// User-defined component identified by a numeric ID.
    Custom(u32),
}

impl LossComponent {
    /// Returns a lower-case human-readable name for the component.
    pub fn name(&self) -> String {
        match self {
            Self::Photometric => "photometric".to_string(),
            Self::Perceptual => "perceptual".to_string(),
            Self::NormalConsistency => "normal_consistency".to_string(),
            Self::PositionRegularizer => "position_regularizer".to_string(),
            Self::ScaleRegularizer => "scale_regularizer".to_string(),
            Self::OpacityRegularizer => "opacity_regularizer".to_string(),
            Self::DiffusionTarget => "diffusion_target".to_string(),
            Self::Custom(id) => format!("custom_{}", id),
        }
    }

    /// Returns `true` for components that act as geometric / sparsity regularizers.
    pub fn is_regularizer(&self) -> bool {
        matches!(
            self,
            Self::NormalConsistency
                | Self::PositionRegularizer
                | Self::ScaleRegularizer
                | Self::OpacityRegularizer
        )
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// LossHistory
// ──────────────────────────────────────────────────────────────────────────────

/// Rolling window + EMA tracker for a single loss component.
#[derive(Debug, Clone)]
pub struct LossHistory {
    /// Rolling window of recent loss values.
    window: VecDeque<f32>,
    /// Maximum number of entries retained in the window.
    pub window_size: usize,
    /// Exponential moving average of the loss.
    pub ema: f32,
    /// Smoothing factor for EMA (α = 0.05 by default).
    pub ema_alpha: f32,
}

impl LossHistory {
    /// Creates a new history with the given window size and default EMA α = 0.05.
    pub fn new(window_size: usize) -> Self {
        Self {
            window: VecDeque::new(),
            window_size,
            ema: 0.0,
            ema_alpha: 0.05,
        }
    }

    /// Appends a new loss sample, evicting the oldest if the window is full.
    pub fn push(&mut self, loss: f32) {
        if self.window.len() >= self.window_size {
            self.window.pop_front();
        }
        self.window.push_back(loss);
        self.ema = self.ema_alpha * loss + (1.0 - self.ema_alpha) * self.ema;
    }

    /// Mean of the current window values; returns 0.0 if the window is empty.
    pub fn mean(&self) -> f32 {
        if self.window.is_empty() {
            return 0.0;
        }
        self.window.iter().sum::<f32>() / self.window.len() as f32
    }

    /// Sample variance of the window; returns 0.0 when there are ≤ 1 samples.
    pub fn variance(&self) -> f32 {
        if self.window.len() <= 1 {
            return 0.0;
        }
        let m = self.mean();
        let sum_sq: f32 = self.window.iter().map(|v| (v - m) * (v - m)).sum();
        sum_sq / (self.window.len() - 1) as f32
    }

    /// Rate of change: `(last − first) / window_size`. Returns 0.0 if fewer
    /// than 2 samples are present.
    pub fn trend(&self) -> f32 {
        let first = match self.window.front() {
            Some(v) => *v,
            None => return 0.0,
        };
        let last = match self.window.back() {
            Some(v) => *v,
            None => return 0.0,
        };
        if self.window.len() < 2 {
            return 0.0;
        }
        (last - first) / self.window_size as f32
    }

    /// Number of samples currently stored in the window.
    pub fn len(&self) -> usize {
        self.window.len()
    }

    /// Returns `true` if the window contains no samples.
    pub fn is_empty(&self) -> bool {
        self.window.is_empty()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// GradNormTracker
// ──────────────────────────────────────────────────────────────────────────────

/// A single recorded gradient-norm measurement.
#[derive(Debug, Clone)]
pub struct GradNormEntry {
    /// Training step at which the norm was recorded.
    pub step: usize,
    /// L2 gradient norm value.
    pub norm: f32,
}

/// Tracks per-component gradient norms across training steps.
pub struct GradNormTracker {
    histories: HashMap<LossComponent, Vec<GradNormEntry>>,
    max_history: usize,
}

impl GradNormTracker {
    /// Creates a new tracker that retains at most `max_history` entries per component.
    pub fn new(max_history: usize) -> Self {
        Self {
            histories: HashMap::new(),
            max_history,
        }
    }

    /// Records a gradient norm for the given component at the given step.
    pub fn record(&mut self, component: LossComponent, step: usize, norm: f32) {
        let entries = self.histories.entry(component).or_default();
        entries.push(GradNormEntry { step, norm });
        // Evict oldest entries when the buffer exceeds the capacity.
        if entries.len() > self.max_history {
            let excess = entries.len() - self.max_history;
            entries.drain(..excess);
        }
    }

    /// Mean gradient norm of the most recent `n` entries for the given component.
    /// Returns `None` if no history exists for the component.
    pub fn recent_mean_norm(&self, component: &LossComponent, n: usize) -> Option<f32> {
        let entries = self.histories.get(component)?;
        if entries.is_empty() {
            return None;
        }
        let start = entries.len().saturating_sub(n);
        let slice = &entries[start..];
        let sum: f32 = slice.iter().map(|e| e.norm).sum();
        Some(sum / slice.len() as f32)
    }

    /// Returns the recent mean norm for every component that has history.
    pub fn all_recent_means(&self, n: usize) -> HashMap<LossComponent, f32> {
        self.histories
            .keys()
            .filter_map(|c| {
                let mean = self.recent_mean_norm(c, n)?;
                Some((c.clone(), mean))
            })
            .collect()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// WeightingStrategy
// ──────────────────────────────────────────────────────────────────────────────

/// Selects the algorithm used to adapt loss weights during training.
#[derive(Debug, Clone)]
pub enum WeightingStrategy {
    /// Fixed weights — no adaptation.
    Fixed(HashMap<LossComponent, f32>),

    /// Normalize weights inversely proportional to recent gradient L2 norm.
    ///
    /// `w_i = (1/g_i) / Σ(1/g_j)` where `g_i` is the mean recent grad norm.
    /// Components without gradient history receive the global mean weight.
    GradNormBalanced {
        base_weights: HashMap<LossComponent, f32>,
        /// EMA blend factor: `w_new = (1−rate)·w_old + rate·w_target` (default 0.01).
        adaptation_rate: f32,
    },

    /// Weights inversely proportional to recent loss magnitude.
    ///
    /// `w_i = (1/l_i) / Σ(1/l_j)`.
    LossRatioBalanced {
        base_weights: HashMap<LossComponent, f32>,
        /// EMA blend factor (default 0.01).
        adaptation_rate: f32,
    },

    /// Uncertainty-based multi-task weighting.
    ///
    /// Models each task's noise as a learnable scalar σ_i.
    /// `w_i = 1 / (2·σ_i²)`, updated so σ_i ≈ sqrt(loss_i / 2).
    Uncertainty {
        /// Per-component uncertainty parameter (initialized to 1.0).
        sigma: HashMap<LossComponent, f32>,
        /// EMA factor for σ updates (default 0.1).
        update_rate: f32,
    },
}

// ──────────────────────────────────────────────────────────────────────────────
// AdaptiveLossWeights
// ──────────────────────────────────────────────────────────────────────────────

/// Holds the current per-component weights together with the adaptation strategy.
#[derive(Debug, Clone)]
pub struct AdaptiveLossWeights {
    /// Current effective weight per component.
    pub weights: HashMap<LossComponent, f32>,
    /// Strategy governing how weights evolve.
    pub strategy: WeightingStrategy,
    /// Current training step (used for bookkeeping).
    pub step: usize,
    /// Minimum weight; prevents any component from being silenced.
    pub min_weight: f32,
    /// Maximum weight; prevents any component from dominating.
    pub max_weight: f32,
}

impl AdaptiveLossWeights {
    /// Creates `AdaptiveLossWeights` initialised from the given strategy.
    pub fn new(strategy: WeightingStrategy) -> Self {
        let weights = Self::init_weights_from_strategy(&strategy);
        Self {
            weights,
            strategy,
            step: 0,
            min_weight: 0.0,
            max_weight: f32::MAX,
        }
    }

    /// Builder: set explicit lower and upper bounds on individual weights.
    pub fn with_bounds(mut self, min: f32, max: f32) -> Self {
        self.min_weight = min;
        self.max_weight = max;
        self
    }

    /// Current weight for `component`; falls back to 1.0 if unknown.
    pub fn get_weight(&self, component: &LossComponent) -> f32 {
        self.weights.get(component).copied().unwrap_or(1.0)
    }

    /// Formats the current weights as a human-readable table.
    pub fn format_weights(&self) -> String {
        let mut lines = vec!["Loss Weights:".to_string()];
        let mut pairs: Vec<(&LossComponent, &f32)> = self.weights.iter().collect();
        pairs.sort_by_key(|a| a.0.name());
        for (component, weight) in pairs {
            lines.push(format!("  {:30} = {:.6}", component.name(), weight));
        }
        lines.join("\n")
    }

    // ---- internal helpers ----

    /// Derives the initial weight map from the chosen strategy.
    fn init_weights_from_strategy(strategy: &WeightingStrategy) -> HashMap<LossComponent, f32> {
        match strategy {
            WeightingStrategy::Fixed(map) => map.clone(),
            WeightingStrategy::GradNormBalanced { base_weights, .. } => base_weights.clone(),
            WeightingStrategy::LossRatioBalanced { base_weights, .. } => base_weights.clone(),
            WeightingStrategy::Uncertainty { sigma, .. } => sigma
                .iter()
                .map(|(c, &s)| {
                    let w = 1.0 / (2.0 * s * s).max(1e-8);
                    (c.clone(), w)
                })
                .collect(),
        }
    }

    /// Clamps `w` into `[self.min_weight, self.max_weight]`.
    fn clamp(&self, w: f32) -> f32 {
        w.clamp(self.min_weight, self.max_weight)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// AdaptiveLossController
// ──────────────────────────────────────────────────────────────────────────────

/// Orchestrates per-step loss recording, weight adaptation, and summary reporting.
pub struct AdaptiveLossController {
    /// Current adaptive weights.
    pub weights: AdaptiveLossWeights,
    /// Rolling loss history per component.
    pub loss_histories: HashMap<LossComponent, LossHistory>,
    /// Gradient norm tracker.
    pub grad_tracker: GradNormTracker,
    /// Window size used when constructing new `LossHistory` instances.
    pub history_window: usize,
    /// Weights are updated once every `update_interval` steps.
    pub update_interval: usize,
}

impl AdaptiveLossController {
    /// Creates a controller tracking the given `components` with the given strategy.
    pub fn new(components: Vec<LossComponent>, strategy: WeightingStrategy) -> Self {
        let history_window = 100;
        let loss_histories = components
            .iter()
            .map(|c| (c.clone(), LossHistory::new(history_window)))
            .collect();

        Self {
            weights: AdaptiveLossWeights::new(strategy),
            loss_histories,
            grad_tracker: GradNormTracker::new(200),
            history_window,
            update_interval: 10,
        }
    }

    /// Records per-component losses for `step`.
    ///
    /// If `step` is a multiple of `update_interval`, the weights are updated
    /// first.  Returns the effective (weighted) total loss.
    pub fn record_losses(&mut self, step: usize, losses: &HashMap<LossComponent, f32>) -> f32 {
        // Push losses into rolling histories.
        for (component, &loss) in losses {
            let hist = self
                .loss_histories
                .entry(component.clone())
                .or_insert_with(|| LossHistory::new(self.history_window));
            hist.push(loss);
        }

        // Periodic weight update.
        if step > 0 && step.is_multiple_of(self.update_interval) {
            self.update_weights(step);
        }

        self.weighted_total(losses)
    }

    /// Records gradient norms for `step`.
    pub fn record_grad_norms(&mut self, step: usize, norms: &HashMap<LossComponent, f32>) {
        for (component, &norm) in norms {
            self.grad_tracker.record(component.clone(), step, norm);
        }
    }

    /// Current weight for a component (defaults to 1.0 if unknown).
    pub fn weight(&self, component: &LossComponent) -> f32 {
        self.weights.get_weight(component)
    }

    /// Computes `Σ w_i · loss_i` using the current weights.
    pub fn weighted_total(&self, losses: &HashMap<LossComponent, f32>) -> f32 {
        losses
            .iter()
            .map(|(c, &l)| self.weights.get_weight(c) * l)
            .sum()
    }

    /// Updates weights according to the current strategy.
    ///
    /// Called automatically by `record_losses` every `update_interval` steps.
    pub fn update_weights(&mut self, _step: usize) {
        match &self.weights.strategy.clone() {
            WeightingStrategy::Fixed(_) => {
                // No-op: fixed weights never change.
            }

            WeightingStrategy::GradNormBalanced {
                adaptation_rate, ..
            } => {
                let rate = *adaptation_rate;
                let recent_means = self.grad_tracker.all_recent_means(10);

                if recent_means.is_empty() {
                    return;
                }

                // Compute inverse norms, clamping to avoid divide-by-zero.
                let inv_norms: HashMap<LossComponent, f32> = recent_means
                    .iter()
                    .map(|(c, &g)| (c.clone(), 1.0 / g.max(1e-8)))
                    .collect();

                let inv_sum: f32 = inv_norms.values().sum();
                if inv_sum < 1e-8 {
                    return;
                }

                // Mean weight for components without grad history.
                let mean_weight = if self.weights.weights.is_empty() {
                    1.0
                } else {
                    self.weights.weights.values().sum::<f32>() / self.weights.weights.len() as f32
                };

                // Blend current weights toward the target.
                let components: Vec<LossComponent> = self.weights.weights.keys().cloned().collect();
                for comp in components {
                    let target = if let Some(&inv) = inv_norms.get(&comp) {
                        inv / inv_sum
                    } else {
                        mean_weight
                    };
                    let old = self.weights.weights.get(&comp).copied().unwrap_or(1.0);
                    let new_w = self.weights.clamp((1.0 - rate) * old + rate * target);
                    self.weights.weights.insert(comp, new_w);
                }
            }

            WeightingStrategy::LossRatioBalanced {
                adaptation_rate, ..
            } => {
                let rate = *adaptation_rate;
                let min_w = self.weights.min_weight;

                // Collect recent means from loss histories.
                let loss_means: HashMap<LossComponent, f32> = self
                    .loss_histories
                    .iter()
                    .map(|(c, h)| (c.clone(), h.mean()))
                    .collect();

                // Compute inverse loss, using min_weight for near-zero losses.
                let inv_losses: HashMap<LossComponent, f32> = loss_means
                    .iter()
                    .map(|(c, &l)| {
                        let inv = if l < 1e-8 { min_w } else { 1.0 / l };
                        (c.clone(), inv)
                    })
                    .collect();

                let inv_sum: f32 = inv_losses.values().sum();
                if inv_sum < 1e-8 {
                    return;
                }

                let components: Vec<LossComponent> = self.weights.weights.keys().cloned().collect();
                for comp in components {
                    let target = if let Some(&inv) = inv_losses.get(&comp) {
                        inv / inv_sum
                    } else {
                        min_w
                    };
                    let old = self.weights.weights.get(&comp).copied().unwrap_or(1.0);
                    let new_w = self.weights.clamp((1.0 - rate) * old + rate * target);
                    self.weights.weights.insert(comp, new_w);
                }
            }

            WeightingStrategy::Uncertainty { update_rate, .. } => {
                let rate = *update_rate;

                // Collect current loss means before taking any mutable borrows.
                let component_losses: Vec<(LossComponent, f32)> = self
                    .loss_histories
                    .iter()
                    .map(|(c, h)| (c.clone(), h.mean()))
                    .collect();

                // Snapshot weight bounds so we can use them inside the
                // mutable borrow of `self.weights.strategy`.
                let min_w = self.weights.min_weight;
                let max_w = self.weights.max_weight;

                // Accumulate (component, new_sigma, new_weight) updates, then
                // apply them in two separate passes to avoid overlapping borrows.
                let mut updates: Vec<(LossComponent, f32, f32)> = Vec::new();

                if let WeightingStrategy::Uncertainty { sigma, .. } = &mut self.weights.strategy {
                    for (comp, loss) in &component_losses {
                        let target_sigma = (loss / 2.0 + 1e-8_f32).sqrt();
                        let old_sigma = sigma.get(comp).copied().unwrap_or(1.0);
                        let new_sigma = (1.0 - rate) * old_sigma + rate * target_sigma;
                        sigma.insert(comp.clone(), new_sigma);

                        let raw_w = 1.0 / (2.0 * new_sigma * new_sigma).max(1e-8);
                        let clamped_w = raw_w.clamp(min_w, max_w);
                        updates.push((comp.clone(), new_sigma, clamped_w));
                    }
                }

                // Apply weight updates now that the mutable borrow is released.
                for (comp, _sigma, w) in updates {
                    self.weights.weights.insert(comp, w);
                }
            }
        }
    }

    /// Formats a human-readable summary of the controller state.
    pub fn format_summary(&self, step: usize) -> String {
        let mut lines = vec![format!("=== AdaptiveLossController @ step {} ===", step)];
        let strategy_name = match &self.weights.strategy {
            WeightingStrategy::Fixed(_) => "Fixed",
            WeightingStrategy::GradNormBalanced { .. } => "GradNormBalanced",
            WeightingStrategy::LossRatioBalanced { .. } => "LossRatioBalanced",
            WeightingStrategy::Uncertainty { .. } => "Uncertainty",
        };
        lines.push(format!("Strategy       : {}", strategy_name));
        lines.push(format!("Update interval: {} steps", self.update_interval));
        lines.push(format!(
            "Weight bounds  : [{:.4}, {:.4}]",
            self.weights.min_weight, self.weights.max_weight
        ));
        lines.push(String::new());
        lines.push(self.weights.format_weights());
        lines.push(String::new());
        lines.push("Loss histories (EMA | mean | trend):".to_string());
        let mut hist_pairs: Vec<(&LossComponent, &LossHistory)> =
            self.loss_histories.iter().collect();
        hist_pairs.sort_by_key(|a| a.0.name());
        for (comp, hist) in hist_pairs {
            lines.push(format!(
                "  {:30} ema={:.4}  mean={:.4}  trend={:.4}  n={}",
                comp.name(),
                hist.ema,
                hist.mean(),
                hist.trend(),
                hist.len()
            ));
        }
        lines.join("\n")
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: build a simple two-component Fixed strategy controller.
    fn fixed_controller() -> AdaptiveLossController {
        let mut base = HashMap::new();
        base.insert(LossComponent::Photometric, 1.0_f32);
        base.insert(LossComponent::Perceptual, 0.5_f32);
        let strategy = WeightingStrategy::Fixed(base);
        AdaptiveLossController::new(
            vec![LossComponent::Photometric, LossComponent::Perceptual],
            strategy,
        )
    }

    #[test]
    fn test_loss_component_name() {
        assert_eq!(LossComponent::Photometric.name(), "photometric");
        assert_eq!(LossComponent::Perceptual.name(), "perceptual");
        assert_eq!(
            LossComponent::NormalConsistency.name(),
            "normal_consistency"
        );
        assert_eq!(
            LossComponent::PositionRegularizer.name(),
            "position_regularizer"
        );
        assert_eq!(LossComponent::ScaleRegularizer.name(), "scale_regularizer");
        assert_eq!(
            LossComponent::OpacityRegularizer.name(),
            "opacity_regularizer"
        );
        assert_eq!(LossComponent::DiffusionTarget.name(), "diffusion_target");
        assert_eq!(LossComponent::Custom(42).name(), "custom_42");
    }

    #[test]
    fn test_loss_component_is_regularizer() {
        assert!(!LossComponent::Photometric.is_regularizer());
        assert!(!LossComponent::Perceptual.is_regularizer());
        assert!(!LossComponent::DiffusionTarget.is_regularizer());
        assert!(LossComponent::NormalConsistency.is_regularizer());
        assert!(LossComponent::PositionRegularizer.is_regularizer());
        assert!(LossComponent::ScaleRegularizer.is_regularizer());
        assert!(LossComponent::OpacityRegularizer.is_regularizer());
        assert!(!LossComponent::Custom(0).is_regularizer());
    }

    #[test]
    fn test_loss_history_push_and_mean() {
        let mut h = LossHistory::new(5);
        assert!(h.is_empty());
        h.push(2.0);
        h.push(4.0);
        // mean = 3.0
        let mean = h.mean();
        assert!((mean - 3.0).abs() < 1e-5, "mean={}", mean);
    }

    #[test]
    fn test_loss_history_variance() {
        let mut h = LossHistory::new(10);
        // single sample: variance = 0
        h.push(5.0);
        assert_eq!(h.variance(), 0.0);

        h.push(7.0);
        // sample variance = ((5-6)^2 + (7-6)^2) / 1 = 2.0
        let v = h.variance();
        assert!((v - 2.0).abs() < 1e-5, "variance={}", v);
    }

    #[test]
    fn test_loss_history_trend() {
        let mut h = LossHistory::new(4);
        assert_eq!(h.trend(), 0.0); // empty

        h.push(1.0);
        assert_eq!(h.trend(), 0.0); // single sample

        h.push(3.0);
        // trend = (3 - 1) / 4 = 0.5
        let t = h.trend();
        assert!((t - 0.5).abs() < 1e-5, "trend={}", t);
    }

    #[test]
    fn test_loss_history_ema() {
        let mut h = LossHistory::new(10);
        // Initial EMA is 0.0.
        h.push(100.0);
        // EMA after first push: 0.05 * 100 + 0.95 * 0 = 5.0
        assert!((h.ema - 5.0).abs() < 1e-5, "ema={}", h.ema);

        h.push(100.0);
        // EMA: 0.05 * 100 + 0.95 * 5.0 = 5 + 4.75 = 9.75
        assert!((h.ema - 9.75).abs() < 1e-5, "ema={}", h.ema);
    }

    #[test]
    fn test_grad_norm_tracker_record_and_query() {
        let mut tracker = GradNormTracker::new(50);
        tracker.record(LossComponent::Photometric, 0, 2.0);
        tracker.record(LossComponent::Photometric, 1, 4.0);
        tracker.record(LossComponent::Photometric, 2, 6.0);

        let mean = tracker
            .recent_mean_norm(&LossComponent::Photometric, 2)
            .expect("should have history");
        // last 2: [4.0, 6.0] → mean = 5.0
        assert!((mean - 5.0).abs() < 1e-5, "mean={}", mean);

        // Component with no history returns None.
        assert!(tracker
            .recent_mean_norm(&LossComponent::Perceptual, 5)
            .is_none());
    }

    #[test]
    fn test_grad_norm_tracker_all_recent_means() {
        let mut tracker = GradNormTracker::new(50);
        tracker.record(LossComponent::Photometric, 0, 1.0);
        tracker.record(LossComponent::Perceptual, 0, 3.0);

        let means = tracker.all_recent_means(10);
        assert_eq!(means.len(), 2);
        let photo = *means.get(&LossComponent::Photometric).expect("present");
        let percep = *means.get(&LossComponent::Perceptual).expect("present");
        assert!((photo - 1.0).abs() < 1e-5);
        assert!((percep - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_adaptive_weights_fixed_strategy() {
        let mut base = HashMap::new();
        base.insert(LossComponent::Photometric, 2.0_f32);
        base.insert(LossComponent::Perceptual, 0.5_f32);
        let w = AdaptiveLossWeights::new(WeightingStrategy::Fixed(base));

        assert!((w.get_weight(&LossComponent::Photometric) - 2.0).abs() < 1e-6);
        assert!((w.get_weight(&LossComponent::Perceptual) - 0.5).abs() < 1e-6);
        // Unknown component defaults to 1.0.
        assert!((w.get_weight(&LossComponent::DiffusionTarget) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_adaptive_weights_loss_ratio_balanced() {
        let mut base = HashMap::new();
        base.insert(LossComponent::Photometric, 1.0_f32);
        base.insert(LossComponent::Perceptual, 1.0_f32);

        let strategy = WeightingStrategy::LossRatioBalanced {
            base_weights: base,
            adaptation_rate: 1.0, // instant update for testing
        };
        let mut ctrl = AdaptiveLossController::new(
            vec![LossComponent::Photometric, LossComponent::Perceptual],
            strategy,
        );

        // Seed loss histories: photometric = 2.0, perceptual = 8.0.
        for _ in 0..20 {
            if let Some(h) = ctrl.loss_histories.get_mut(&LossComponent::Photometric) {
                h.push(2.0)
            }
            if let Some(h) = ctrl.loss_histories.get_mut(&LossComponent::Perceptual) {
                h.push(8.0)
            }
        }

        ctrl.update_weights(10);

        // inv_photo = 0.5, inv_percep = 0.125, sum = 0.625
        // target_photo = 0.5/0.625 = 0.8, target_percep = 0.125/0.625 = 0.2
        let wp = ctrl.weight(&LossComponent::Photometric);
        let we = ctrl.weight(&LossComponent::Perceptual);
        assert!(
            wp > we,
            "photometric weight ({}) should exceed perceptual ({})",
            wp,
            we
        );
        assert!((wp + we - 1.0).abs() < 1e-5, "weights sum={}", wp + we);
    }

    #[test]
    fn test_adaptive_weights_grad_norm_balanced() {
        let mut base = HashMap::new();
        base.insert(LossComponent::Photometric, 1.0_f32);
        base.insert(LossComponent::Perceptual, 1.0_f32);

        let strategy = WeightingStrategy::GradNormBalanced {
            base_weights: base,
            adaptation_rate: 1.0, // instant update
        };
        let mut ctrl = AdaptiveLossController::new(
            vec![LossComponent::Photometric, LossComponent::Perceptual],
            strategy,
        );

        // photometric grad norm = 2.0 (smaller) → should get larger weight
        // perceptual  grad norm = 8.0             → smaller weight
        for step in 0..10_usize {
            ctrl.grad_tracker
                .record(LossComponent::Photometric, step, 2.0);
            ctrl.grad_tracker
                .record(LossComponent::Perceptual, step, 8.0);
        }

        ctrl.update_weights(10);

        let wp = ctrl.weight(&LossComponent::Photometric);
        let we = ctrl.weight(&LossComponent::Perceptual);
        // inv_photo = 0.5, inv_percep = 0.125, sum = 0.625
        // target_photo = 0.8, target_percep = 0.2
        assert!(wp > we, "photo={} percep={}", wp, we);
    }

    #[test]
    fn test_adaptive_weights_uncertainty() {
        let mut sigma = HashMap::new();
        sigma.insert(LossComponent::Photometric, 1.0_f32);
        sigma.insert(LossComponent::Perceptual, 1.0_f32);

        let strategy = WeightingStrategy::Uncertainty {
            sigma,
            update_rate: 1.0, // instant update
        };
        let mut ctrl = AdaptiveLossController::new(
            vec![LossComponent::Photometric, LossComponent::Perceptual],
            strategy,
        );

        // Seed histories.
        for _ in 0..20 {
            if let Some(h) = ctrl.loss_histories.get_mut(&LossComponent::Photometric) {
                h.push(0.02)
            }
            if let Some(h) = ctrl.loss_histories.get_mut(&LossComponent::Perceptual) {
                h.push(0.5)
            }
        }

        ctrl.update_weights(10);

        // Smaller loss → smaller sigma → larger weight.
        let wp = ctrl.weight(&LossComponent::Photometric);
        let we = ctrl.weight(&LossComponent::Perceptual);
        assert!(
            wp > we,
            "photometric ({}) should have larger weight than perceptual ({})",
            wp,
            we
        );
    }

    #[test]
    fn test_controller_record_losses() {
        let mut ctrl = fixed_controller();
        let mut losses = HashMap::new();
        losses.insert(LossComponent::Photometric, 0.1_f32);
        losses.insert(LossComponent::Perceptual, 0.2_f32);

        // Weights: photo=1.0, percep=0.5 → total = 0.1*1 + 0.2*0.5 = 0.2
        let total = ctrl.record_losses(1, &losses);
        assert!((total - 0.2).abs() < 1e-5, "total={}", total);

        // Loss histories should have been updated.
        let ph = ctrl
            .loss_histories
            .get(&LossComponent::Photometric)
            .expect("present");
        assert_eq!(ph.len(), 1);
    }

    #[test]
    fn test_controller_weighted_total() {
        let ctrl = fixed_controller();
        let mut losses = HashMap::new();
        losses.insert(LossComponent::Photometric, 0.4_f32);
        losses.insert(LossComponent::Perceptual, 0.2_f32);

        // photo=1.0*0.4 + percep=0.5*0.2 = 0.4 + 0.1 = 0.5
        let total = ctrl.weighted_total(&losses);
        assert!((total - 0.5).abs() < 1e-5, "total={}", total);
    }

    #[test]
    fn test_controller_update_interval() {
        let mut base = HashMap::new();
        base.insert(LossComponent::Photometric, 1.0_f32);
        base.insert(LossComponent::Perceptual, 1.0_f32);

        let strategy = WeightingStrategy::LossRatioBalanced {
            base_weights: base,
            adaptation_rate: 0.5,
        };
        let mut ctrl = AdaptiveLossController::new(
            vec![LossComponent::Photometric, LossComponent::Perceptual],
            strategy,
        );
        ctrl.update_interval = 5;

        let mut losses = HashMap::new();
        losses.insert(LossComponent::Photometric, 1.0_f32);
        losses.insert(LossComponent::Perceptual, 1.0_f32);

        // Steps 1..4 should not trigger update (no panic / no weight change).
        for step in 1..5_usize {
            ctrl.record_losses(step, &losses);
        }
        // Step 5 triggers an update.
        ctrl.record_losses(5, &losses);
        // Both components have identical losses → weights should remain balanced.
        let wp = ctrl.weight(&LossComponent::Photometric);
        let we = ctrl.weight(&LossComponent::Perceptual);
        assert!((wp - we).abs() < 0.1, "photo={} percep={}", wp, we);
    }

    #[test]
    fn test_controller_format_summary() {
        let ctrl = fixed_controller();
        let summary = ctrl.format_summary(42);
        assert!(summary.contains("step 42"));
        assert!(summary.contains("Fixed"));
        assert!(summary.contains("photometric"));
        assert!(summary.contains("perceptual"));
    }
}
