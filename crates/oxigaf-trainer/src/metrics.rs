//! Image-quality metrics and metric history tracking.
//!
//! * **PSNR** — Peak Signal-to-Noise Ratio.
//! * **SSIM** — reuses the windowed SSIM from [`crate::loss`].
//! * [`MetricTracker`] — rolling log of per-iteration metrics.

use crate::loss;

// ---------------------------------------------------------------------------
// PSNR
// ---------------------------------------------------------------------------

/// Compute PSNR between two images stored as flat `[0, 1]` f32 arrays.
///
/// $\text{PSNR} = -10 \log_{10}(\text{MSE})$  (for peak value = 1).
pub fn psnr(pred: &[f32], target: &[f32]) -> f32 {
    if pred.is_empty() || target.is_empty() {
        return 0.0;
    }
    let mse: f32 = pred
        .iter()
        .zip(target.iter())
        .map(|(p, t)| {
            let d = p - t;
            d * d
        })
        .sum::<f32>()
        / pred.len() as f32;

    if mse < 1e-10 {
        return 100.0; // cap at 100 dB for identical images
    }
    -10.0 * mse.log10()
}

/// Approximate PSNR from a known MSE (or L1→MSE heuristic).
///
/// Useful for quick logging when only the loss value is available.
pub fn psnr_from_mse(mse: f32) -> f32 {
    if mse < 1e-10 {
        return 100.0;
    }
    -10.0 * mse.log10()
}

// ---------------------------------------------------------------------------
// SSIM (delegates to loss module)
// ---------------------------------------------------------------------------

/// Compute mean SSIM ∈ [0, 1] between two HWC float images.
pub fn ssim(pred: &[f32], target: &[f32], width: usize, height: usize) -> f32 {
    let kernel = loss::gaussian_kernel_1d(11, 1.5);
    // ssim_loss returns 1 − SSIM, so invert.
    1.0 - loss::ssim_loss(pred, target, width, height, &kernel)
}

// ---------------------------------------------------------------------------
// MetricTracker
// ---------------------------------------------------------------------------

/// A single metric snapshot.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MetricEntry {
    pub iteration: u32,
    pub psnr: f32,
    pub ssim: f32,
    pub loss: f32,
}

/// Simple rolling log of per-iteration metrics.
#[derive(Debug, Clone)]
pub struct MetricTracker {
    history: Vec<MetricEntry>,
}

impl MetricTracker {
    pub fn new() -> Self {
        Self {
            history: Vec::new(),
        }
    }

    /// Append a new entry.
    pub fn record(&mut self, iteration: u32, psnr: f32, ssim: f32, loss: f32) {
        self.history.push(MetricEntry {
            iteration,
            psnr,
            ssim,
            loss,
        });
    }

    /// Most recent entry (if any).
    pub fn latest(&self) -> Option<&MetricEntry> {
        self.history.last()
    }

    /// Number of recorded entries.
    pub fn len(&self) -> usize {
        self.history.len()
    }

    /// Whether the history is empty.
    pub fn is_empty(&self) -> bool {
        self.history.is_empty()
    }

    /// Mean PSNR over the last `n` entries.
    pub fn mean_psnr(&self, n: usize) -> f32 {
        let slice = &self.history[self.history.len().saturating_sub(n)..];
        if slice.is_empty() {
            return 0.0;
        }
        slice.iter().map(|e| e.psnr).sum::<f32>() / slice.len() as f32
    }

    /// Mean SSIM over the last `n` entries.
    pub fn mean_ssim(&self, n: usize) -> f32 {
        let slice = &self.history[self.history.len().saturating_sub(n)..];
        if slice.is_empty() {
            return 0.0;
        }
        slice.iter().map(|e| e.ssim).sum::<f32>() / slice.len() as f32
    }

    /// Mean loss over the last `n` entries.
    pub fn mean_loss(&self, n: usize) -> f32 {
        let slice = &self.history[self.history.len().saturating_sub(n)..];
        if slice.is_empty() {
            return 0.0;
        }
        slice.iter().map(|e| e.loss).sum::<f32>() / slice.len() as f32
    }

    /// Human-readable summary of recent metrics.
    pub fn summary_string(&self, window: usize) -> String {
        if let Some(last) = self.history.last() {
            format!(
                "iter {:>6} | loss {:.5} | PSNR {:.2} dB | SSIM {:.4} | \
                 avg({window}): loss {:.5} PSNR {:.2} SSIM {:.4}",
                last.iteration,
                last.loss,
                last.psnr,
                last.ssim,
                self.mean_loss(window),
                self.mean_psnr(window),
                self.mean_ssim(window),
            )
        } else {
            "No metrics recorded yet.".into()
        }
    }

    /// Access the full history slice.
    pub fn history(&self) -> &[MetricEntry] {
        &self.history
    }

    /// Checkpoint the metrics history for serialization.
    pub fn checkpoint_state(&self) -> Vec<MetricEntry> {
        self.history.clone()
    }

    /// Restore metrics history from a checkpoint.
    pub fn restore(&mut self, history: Vec<MetricEntry>) {
        self.history = history;
    }

    /// Create a MetricTracker from a saved history.
    pub fn from_history(history: Vec<MetricEntry>) -> Self {
        Self { history }
    }
}

impl Default for MetricTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn psnr_identical_is_high() {
        let img = vec![0.5_f32; 100];
        let val = psnr(&img, &img);
        assert!(val >= 99.0, "expected ≥99, got {val}");
    }

    #[test]
    fn psnr_different_is_finite() {
        let a = vec![0.0_f32; 100];
        let b = vec![1.0_f32; 100];
        let val = psnr(&a, &b);
        assert!(val.is_finite());
        assert!(val < 10.0);
    }

    #[test]
    fn tracker_rolling_mean() {
        let mut tracker = MetricTracker::new();
        for i in 0..10 {
            tracker.record(i, i as f32, 0.0, 0.0);
        }
        let mean = tracker.mean_psnr(5);
        // last 5 entries: 5,6,7,8,9 → mean = 7
        assert!((mean - 7.0).abs() < 1e-5);
    }
}
