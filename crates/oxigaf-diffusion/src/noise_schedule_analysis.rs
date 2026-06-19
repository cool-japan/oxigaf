//! Noise schedule analysis tools for understanding diffusion model training dynamics.
//!
//! These are pure CPU/mathematical utilities — no model weights needed.
//! Covers linear, scaled-linear (SD 2.1), cosine (improved DDPM), and sigmoid
//! schedules, plus DDIM sampling analysis and ASCII visualization.

use thiserror::Error;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during schedule analysis.
#[derive(Debug, Error)]
pub enum ScheduleAnalysisError {
    #[error("Number of timesteps must be > 0, got {0}")]
    InvalidTimesteps(usize),
    #[error("Beta values must be in (0, 1), got {0} at index {1}")]
    InvalidBeta(f32, usize),
    #[error("Alpha cumulative must be strictly decreasing")]
    NonDecreasingAlphaCum,
    #[error("Schedule names must be non-empty")]
    EmptyScheduleName,
    #[error("Cannot compare schedules with different timestep counts: {a} vs {b}")]
    TimestepMismatch { a: usize, b: usize },
}

// ---------------------------------------------------------------------------
// ScheduleType
// ---------------------------------------------------------------------------

/// Identifies the kind of noise schedule.
#[derive(Debug, Clone, PartialEq)]
pub enum ScheduleType {
    /// Linear beta schedule: beta_t = beta_start + t*(beta_end-beta_start)/(T-1)
    Linear { beta_start: f32, beta_end: f32 },
    /// Scaled-linear (SD 2.1): betas linearly spaced in sqrt space, then squared.
    ScaledLinear { beta_start: f32, beta_end: f32 },
    /// Cosine schedule (improved DDPM): alpha_bar_t = cos²(((t/T+s)/(1+s))*π/2)
    Cosine { s: f32 },
    /// Sigmoid schedule: beta_t = sigmoid(beta_start + (beta_end-beta_start)*t/(T-1))
    Sigmoid { beta_start: f32, beta_end: f32 },
    /// Custom: betas supplied directly.
    Custom,
    /// Karras et al. (EDM 2022): log-linear sigma sequence.
    /// σ_i = (σ_max^(1/ρ) + i/(N-1) * (σ_min^(1/ρ) - σ_max^(1/ρ)))^ρ
    Karras {
        sigma_min: f32,
        sigma_max: f32,
        rho: f32,
    },
}

// ---------------------------------------------------------------------------
// NoiseSchedule
// ---------------------------------------------------------------------------

/// A discretized noise schedule with all derived coefficient arrays.
#[derive(Debug, Clone)]
pub struct NoiseSchedule {
    pub schedule_type: ScheduleType,
    pub num_timesteps: usize,
    pub betas: Vec<f32>,
    pub alphas: Vec<f32>,
    pub alphas_cumprod: Vec<f32>,
    pub alphas_cumprod_prev: Vec<f32>,
    pub sqrt_alphas_cumprod: Vec<f32>,
    pub sqrt_one_minus_alphas_cumprod: Vec<f32>,
}

// Helper: evenly spaced values (inclusive on both ends)
fn linspace(start: f32, end: f32, n: usize) -> Vec<f32> {
    if n == 1 {
        return vec![start];
    }
    (0..n)
        .map(|i| start + (end - start) * (i as f32) / ((n - 1) as f32))
        .collect()
}

// Helper: sigmoid
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Compute the Karras sigma at inference step `i` (0-indexed, 0 = noisiest).
/// Uses the formula: σ_i = (σ_max^(1/ρ) + i/(N-1) * (σ_min^(1/ρ) - σ_max^(1/ρ)))^ρ
///
/// # Panics if N < 2 (checked by callers).
pub fn karras_sigma_at(i: usize, n: usize, sigma_min: f32, sigma_max: f32, rho: f32) -> f32 {
    debug_assert!(n >= 2, "need at least 2 steps");
    let rho_inv = 1.0 / rho;
    let max_inv = sigma_max.powf(rho_inv);
    let min_inv = sigma_min.powf(rho_inv);
    let t = i as f32 / (n - 1) as f32;
    (max_inv + t * (min_inv - max_inv)).powf(rho)
}

impl NoiseSchedule {
    /// Build a schedule from an explicit betas vector.
    pub fn from_betas(
        betas: Vec<f32>,
        schedule_type: ScheduleType,
    ) -> Result<Self, ScheduleAnalysisError> {
        let n = betas.len();
        if n == 0 {
            return Err(ScheduleAnalysisError::InvalidTimesteps(0));
        }
        // Validate beta range
        for (i, &b) in betas.iter().enumerate() {
            if !(b > 0.0 && b < 1.0) {
                return Err(ScheduleAnalysisError::InvalidBeta(b, i));
            }
        }

        let alphas: Vec<f32> = betas.iter().map(|&b| 1.0 - b).collect();

        // alphas_cumprod[t] = prod(alpha[0..=t])
        let mut alphas_cumprod = Vec::with_capacity(n);
        let mut cum = 1.0_f32;
        for &a in &alphas {
            cum *= a;
            alphas_cumprod.push(cum);
        }

        // Verify strictly decreasing
        for i in 1..n {
            if alphas_cumprod[i] >= alphas_cumprod[i - 1] {
                return Err(ScheduleAnalysisError::NonDecreasingAlphaCum);
            }
        }

        // alphas_cumprod_prev: [1.0, alphas_cumprod[0], ..., alphas_cumprod[T-2]]
        let mut alphas_cumprod_prev = Vec::with_capacity(n);
        alphas_cumprod_prev.push(1.0_f32);
        alphas_cumprod_prev.extend_from_slice(&alphas_cumprod[..n - 1]);

        let sqrt_alphas_cumprod: Vec<f32> = alphas_cumprod.iter().map(|&a| a.sqrt()).collect();
        let sqrt_one_minus_alphas_cumprod: Vec<f32> =
            alphas_cumprod.iter().map(|&a| (1.0 - a).sqrt()).collect();

        Ok(Self {
            schedule_type,
            num_timesteps: n,
            betas,
            alphas,
            alphas_cumprod,
            alphas_cumprod_prev,
            sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod,
        })
    }

    /// Create a linear beta schedule.
    pub fn linear(
        num_timesteps: usize,
        beta_start: f32,
        beta_end: f32,
    ) -> Result<Self, ScheduleAnalysisError> {
        if num_timesteps == 0 {
            return Err(ScheduleAnalysisError::InvalidTimesteps(0));
        }
        let betas = linspace(beta_start, beta_end, num_timesteps);
        Self::from_betas(
            betas,
            ScheduleType::Linear {
                beta_start,
                beta_end,
            },
        )
    }

    /// Create a scaled-linear (SD 2.1) schedule.
    /// betas = linspace(sqrt(beta_start), sqrt(beta_end), T)²
    pub fn scaled_linear(
        num_timesteps: usize,
        beta_start: f32,
        beta_end: f32,
    ) -> Result<Self, ScheduleAnalysisError> {
        if num_timesteps == 0 {
            return Err(ScheduleAnalysisError::InvalidTimesteps(0));
        }
        let betas: Vec<f32> = linspace(beta_start.sqrt(), beta_end.sqrt(), num_timesteps)
            .into_iter()
            .map(|x| x * x)
            .collect();
        Self::from_betas(
            betas,
            ScheduleType::ScaledLinear {
                beta_start,
                beta_end,
            },
        )
    }

    /// Create a cosine schedule (improved DDPM, Nichol & Dhariwal 2021).
    ///
    /// f(t) = cos²(((t/T + s)/(1+s)) * π/2)
    /// alpha_bar\[t\] = f(t) / f(0)
    /// beta\[t\] = clamp(1 - alpha_bar\[t\] / alpha_bar\[t-1\], 0, 0.999)  for t > 0
    /// beta\[0\] = clamp(1 - alpha_bar\[0\], 0, 0.999)
    pub fn cosine(num_timesteps: usize, s: f32) -> Result<Self, ScheduleAnalysisError> {
        if num_timesteps == 0 {
            return Err(ScheduleAnalysisError::InvalidTimesteps(0));
        }
        let t = num_timesteps as f32;
        let f0 = {
            let arg = (s / (1.0 + s)) * std::f32::consts::FRAC_PI_2;
            arg.cos().powi(2)
        };

        // Build raw alpha_bar array
        let alpha_bar: Vec<f32> = (0..num_timesteps)
            .map(|i| {
                let arg = ((i as f32 / t + s) / (1.0 + s)) * std::f32::consts::FRAC_PI_2;
                arg.cos().powi(2) / f0
            })
            .collect();

        // Derive betas
        let mut betas = Vec::with_capacity(num_timesteps);
        betas.push((1.0 - alpha_bar[0]).clamp(0.0, 0.999));
        for i in 1..num_timesteps {
            let ratio = alpha_bar[i] / alpha_bar[i - 1];
            betas.push((1.0 - ratio).clamp(0.0, 0.999));
        }

        // Validate: all betas should be in (0,1) after clamping at 0.999, but
        // a zero beta would violate from_betas — clamp minimum to a tiny epsilon.
        let betas: Vec<f32> = betas.into_iter().map(|b| b.max(f32::EPSILON)).collect();

        Self::from_betas(betas, ScheduleType::Cosine { s })
    }

    /// Create a sigmoid schedule.
    ///
    /// beta\[t\] = sigmoid(beta_start + (beta_end - beta_start) * t/(T-1))
    pub fn sigmoid_schedule(
        num_timesteps: usize,
        beta_start: f32,
        beta_end: f32,
    ) -> Result<Self, ScheduleAnalysisError> {
        if num_timesteps == 0 {
            return Err(ScheduleAnalysisError::InvalidTimesteps(0));
        }
        let betas: Vec<f32> = (0..num_timesteps)
            .map(|i| {
                let x = if num_timesteps == 1 {
                    beta_start
                } else {
                    beta_start + (beta_end - beta_start) * (i as f32) / ((num_timesteps - 1) as f32)
                };
                sigmoid(x)
            })
            .collect();
        Self::from_betas(
            betas,
            ScheduleType::Sigmoid {
                beta_start,
                beta_end,
            },
        )
    }

    /// Build a Karras EDM sigma schedule (Karras et al. 2022).
    ///
    /// Converts Karras sigmas σ_i to alpha_bar_t values via:
    ///   alpha_bar_t = 1 / (1 + σ_i²)
    /// so that σ = sqrt((1-ᾱ)/ᾱ) (the standard diffusion SNR relationship).
    ///
    /// # Errors
    /// Returns `ScheduleAnalysisError::InvalidTimesteps` if `num_timesteps < 2`.
    /// Returns `ScheduleAnalysisError::InvalidBeta` if `sigma_min <= 0` or `sigma_max <= sigma_min`.
    pub fn karras(
        num_timesteps: usize,
        sigma_min: f32,
        sigma_max: f32,
        rho: f32,
    ) -> Result<Self, ScheduleAnalysisError> {
        if num_timesteps < 2 {
            return Err(ScheduleAnalysisError::InvalidTimesteps(num_timesteps));
        }
        if sigma_min <= 0.0 || !sigma_min.is_finite() {
            return Err(ScheduleAnalysisError::InvalidBeta(sigma_min, 0));
        }
        if sigma_max <= sigma_min {
            return Err(ScheduleAnalysisError::InvalidBeta(sigma_max, 1));
        }
        // Build sigmas from noisiest (i=0) to cleanest (i=N-1)
        let sigmas: Vec<f32> = (0..num_timesteps)
            .map(|i| karras_sigma_at(i, num_timesteps, sigma_min, sigma_max, rho))
            .collect();
        // Convert sigma → alpha_bar: alpha_bar = 1/(1 + sigma²)
        let alphas_cumprod: Vec<f32> = sigmas.iter().map(|&s| 1.0 / (1.0 + s * s)).collect();
        // alpha_t = alpha_bar_t / alpha_bar_{t-1}
        let alphas: Vec<f32> = std::iter::once(alphas_cumprod[0])
            .chain(alphas_cumprod.windows(2).map(|w| w[1] / w[0]))
            .collect();
        // beta_t = 1 - alpha_t
        let betas: Vec<f32> = alphas.iter().map(|&a| 1.0 - a).collect();
        let alphas_cumprod_prev: Vec<f32> = std::iter::once(1.0f32)
            .chain(alphas_cumprod[..alphas_cumprod.len() - 1].iter().copied())
            .collect();
        let sqrt_alphas_cumprod: Vec<f32> = alphas_cumprod.iter().map(|x| x.sqrt()).collect();
        let sqrt_one_minus_alphas_cumprod: Vec<f32> =
            alphas_cumprod.iter().map(|x| (1.0 - x).sqrt()).collect();
        Ok(NoiseSchedule {
            schedule_type: ScheduleType::Karras {
                sigma_min,
                sigma_max,
                rho,
            },
            num_timesteps,
            betas,
            alphas,
            alphas_cumprod,
            alphas_cumprod_prev,
            sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod,
        })
    }

    /// Signal-to-Noise Ratio at timestep t: SNR(t) = ᾱ_t / (1 - ᾱ_t)
    pub fn snr(&self, t: usize) -> f32 {
        let alpha_bar = self.alphas_cumprod[t];
        let denom = (1.0 - alpha_bar).max(f32::EPSILON);
        alpha_bar / denom
    }

    /// Log (base-e) of SNR at timestep t.
    pub fn log_snr(&self, t: usize) -> f32 {
        self.snr(t).max(f32::EPSILON).ln()
    }

    /// Forward process coefficients at timestep t.
    ///
    /// Returns (signal_coeff = √ᾱ_t, noise_coeff = √(1-ᾱ_t)).
    pub fn forward_coefficients(&self, t: usize) -> (f32, f32) {
        (
            self.sqrt_alphas_cumprod[t],
            self.sqrt_one_minus_alphas_cumprod[t],
        )
    }

    /// Noise standard deviation at timestep t: √(1 - ᾱ_t)
    pub fn noise_level(&self, t: usize) -> f32 {
        self.sqrt_one_minus_alphas_cumprod[t]
    }

    /// Find the timestep where SNR is closest to `target_snr` using linear search
    /// (binary search not applicable because SNR can be non-strictly monotone due to
    /// float rounding; a scan is O(T) and T ≤ 10000 in practice).
    pub fn timestep_at_snr(&self, target_snr: f32) -> usize {
        let mut best_t = 0usize;
        let mut best_diff = f32::MAX;
        for t in 0..self.num_timesteps {
            let diff = (self.snr(t) - target_snr).abs();
            if diff < best_diff {
                best_diff = diff;
                best_t = t;
            }
        }
        best_t
    }
}

// ---------------------------------------------------------------------------
// ScheduleAnalysis
// ---------------------------------------------------------------------------

/// Statistics derived from a single noise schedule.
#[derive(Debug, Clone)]
pub struct ScheduleAnalysis {
    pub schedule_name: String,
    pub num_timesteps: usize,
    pub snr_values: Vec<f32>,
    pub signal_fractions: Vec<f32>,
    pub noise_fractions: Vec<f32>,
    /// Timestep t where |SNR(t) - 1.0| is minimised.
    pub transition_timestep: usize,
    /// (first t where SNR < 10, first t where SNR < 0.1), or (0, T-1) when absent.
    pub informative_range: (usize, usize),
    /// Trapezoidal integral of the SNR curve over all timesteps.
    pub snr_auc: f32,
    /// Standard deviation of beta increments (Δβ).
    pub beta_smoothness: f32,
}

/// Analyze a noise schedule and return statistics.
pub fn analyze_schedule(schedule: &NoiseSchedule, name: impl Into<String>) -> ScheduleAnalysis {
    let schedule_name = name.into();
    let n = schedule.num_timesteps;

    let snr_values: Vec<f32> = (0..n).map(|t| schedule.snr(t)).collect();
    let signal_fractions: Vec<f32> = schedule.sqrt_alphas_cumprod.clone();
    let noise_fractions: Vec<f32> = schedule.sqrt_one_minus_alphas_cumprod.clone();

    // Transition timestep: minimise |SNR(t) - 1.0|
    let transition_timestep = snr_values
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            let da = (*a - 1.0_f32).abs();
            let db = (*b - 1.0_f32).abs();
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i)
        .unwrap_or(0);

    // Informative range
    let first_below_10 = snr_values.iter().position(|&s| s < 10.0);
    let first_below_01 = snr_values.iter().position(|&s| s < 0.1);
    let informative_range = match (first_below_10, first_below_01) {
        (Some(a), Some(b)) => (a, b),
        _ => (0, n.saturating_sub(1)),
    };

    // Trapezoidal SNR AUC
    let snr_auc = if n < 2 {
        snr_values.first().copied().unwrap_or(0.0)
    } else {
        snr_values
            .windows(2)
            .map(|w| 0.5 * (w[0] + w[1]))
            .sum::<f32>()
    };

    // Beta smoothness = std dev of consecutive beta differences
    let beta_smoothness = {
        let diffs: Vec<f32> = schedule.betas.windows(2).map(|w| w[1] - w[0]).collect();
        if diffs.is_empty() {
            0.0_f32
        } else {
            let mean = diffs.iter().sum::<f32>() / diffs.len() as f32;
            let variance =
                diffs.iter().map(|&d| (d - mean).powi(2)).sum::<f32>() / diffs.len() as f32;
            variance.sqrt()
        }
    };

    ScheduleAnalysis {
        schedule_name,
        num_timesteps: n,
        snr_values,
        signal_fractions,
        noise_fractions,
        transition_timestep,
        informative_range,
        snr_auc,
        beta_smoothness,
    }
}

// ---------------------------------------------------------------------------
// ScheduleComparison
// ---------------------------------------------------------------------------

/// Result of comparing two noise schedules.
#[derive(Debug, Clone)]
pub struct ScheduleComparison {
    pub name_a: String,
    pub name_b: String,
    /// L2 distance between SNR curves.
    pub snr_l2_distance: f32,
    /// Max absolute difference in alphas_cumprod.
    pub max_alpha_diff: f32,
    /// Difference in transition timestep (signed: a - b).
    pub transition_diff: i64,
    /// Which schedule has higher total SNR (better signal preservation).
    pub higher_snr_schedule: String,
}

/// Compare two noise schedules of equal length.
pub fn compare_schedules(
    a: &NoiseSchedule,
    name_a: impl Into<String>,
    b: &NoiseSchedule,
    name_b: impl Into<String>,
) -> Result<ScheduleComparison, ScheduleAnalysisError> {
    if a.num_timesteps != b.num_timesteps {
        return Err(ScheduleAnalysisError::TimestepMismatch {
            a: a.num_timesteps,
            b: b.num_timesteps,
        });
    }
    let name_a = name_a.into();
    let name_b = name_b.into();
    let n = a.num_timesteps;

    // L2 distance between SNR curves
    let snr_l2_distance = {
        let sq_sum: f32 = (0..n).map(|t| (a.snr(t) - b.snr(t)).powi(2)).sum();
        (sq_sum / n as f32).sqrt()
    };

    // Max absolute difference in alphas_cumprod
    let max_alpha_diff = a
        .alphas_cumprod
        .iter()
        .zip(b.alphas_cumprod.iter())
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0_f32, f32::max);

    // Transition timestep for each
    let analysis_a = analyze_schedule(a, &name_a);
    let analysis_b = analyze_schedule(b, &name_b);
    let transition_diff =
        analysis_a.transition_timestep as i64 - analysis_b.transition_timestep as i64;

    // Higher total SNR
    let total_a: f32 = analysis_a.snr_values.iter().sum();
    let total_b: f32 = analysis_b.snr_values.iter().sum();
    let higher_snr_schedule = if total_a >= total_b {
        name_a.clone()
    } else {
        name_b.clone()
    };

    Ok(ScheduleComparison {
        name_a,
        name_b,
        snr_l2_distance,
        max_alpha_diff,
        transition_diff,
        higher_snr_schedule,
    })
}

// ---------------------------------------------------------------------------
// DDIM sampling analysis
// ---------------------------------------------------------------------------

/// Analysis of a DDIM sampling trajectory over a noise schedule.
#[derive(Debug, Clone)]
pub struct DdimSamplingAnalysis {
    pub num_steps: usize,
    pub timesteps: Vec<usize>,
    pub step_snrs: Vec<f32>,
    pub snr_gaps: Vec<f32>,
    pub max_snr_gap: f32,
    /// 1.0 / (1.0 + max_snr_gap)  — higher is better.
    pub quality_score: f32,
}

/// Compute DDIM timestep schedule: linspace(0, T-1, num_steps), rounded, reversed.
///
/// Returns timesteps in descending order (denoising order: high-noise → low-noise).
pub fn ddim_timesteps(total_timesteps: usize, num_steps: usize) -> Vec<usize> {
    if num_steps == 0 || total_timesteps == 0 {
        return Vec::new();
    }
    let t_max = (total_timesteps - 1) as f32;
    let mut ts: Vec<usize> = if num_steps == 1 {
        vec![0]
    } else {
        (0..num_steps)
            .map(|i| {
                let v = t_max * (i as f32) / ((num_steps - 1) as f32);
                v.round() as usize
            })
            .collect()
    };
    // Reverse for denoising order (high timestep first)
    ts.reverse();
    ts
}

/// Analyze a DDIM sampling trajectory.
pub fn analyze_ddim_sampling(
    schedule: &NoiseSchedule,
    num_inference_steps: usize,
) -> DdimSamplingAnalysis {
    let timesteps = ddim_timesteps(schedule.num_timesteps, num_inference_steps);
    let step_snrs: Vec<f32> = timesteps.iter().map(|&t| schedule.snr(t)).collect();

    let snr_gaps: Vec<f32> = step_snrs.windows(2).map(|w| (w[0] - w[1]).abs()).collect();

    let max_snr_gap = snr_gaps.iter().cloned().fold(0.0_f32, f32::max);
    let quality_score = 1.0 / (1.0 + max_snr_gap);

    DdimSamplingAnalysis {
        num_steps: num_inference_steps,
        timesteps,
        step_snrs,
        snr_gaps,
        max_snr_gap,
        quality_score,
    }
}

// ---------------------------------------------------------------------------
// Visualization helpers
// ---------------------------------------------------------------------------

/// Format the log10(SNR) curve as ASCII art (`width` × `height` characters).
///
/// Y-axis: log10(SNR), X-axis: timestep index.
/// `'*'` marks the curve point for each column, `'.'` fills empty cells.
pub fn format_snr_curve_ascii(schedule: &NoiseSchedule, width: usize, height: usize) -> String {
    if width == 0 || height == 0 || schedule.num_timesteps == 0 {
        return String::new();
    }

    // Compute log10(SNR) at width-sampled timesteps
    let n = schedule.num_timesteps;
    let log_snrs: Vec<f32> = (0..width)
        .map(|col| {
            let t = if width == 1 {
                0
            } else {
                ((col as f32 / (width - 1) as f32) * (n - 1) as f32).round() as usize
            };
            let t = t.min(n - 1);
            let snr = schedule.snr(t).max(f32::EPSILON);
            snr.log10()
        })
        .collect();

    let y_min = log_snrs.iter().cloned().fold(f32::INFINITY, f32::min);
    let y_max = log_snrs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let y_range = (y_max - y_min).max(f32::EPSILON);

    // For each column, compute which row the curve occupies (row 0 = top = high SNR)
    let curve_row: Vec<usize> = log_snrs
        .iter()
        .map(|&v| {
            let frac = (v - y_min) / y_range; // 0 = bottom, 1 = top
            let row = ((1.0 - frac) * (height - 1) as f32).round() as usize;
            row.min(height - 1)
        })
        .collect();

    // Build grid
    let mut grid = vec![vec!['.'; width]; height];
    for (col, &row) in curve_row.iter().enumerate() {
        grid[row][col] = '*';
    }

    grid.iter()
        .map(|row| row.iter().collect::<String>())
        .collect::<Vec<_>>()
        .join("\n")
}

/// Format a ScheduleComparison as a human-readable text table.
pub fn format_comparison_table(comparison: &ScheduleComparison) -> String {
    format!(
        "Schedule Comparison: {} vs {}\n\
         {}\n\
         {:30} {:.6}\n\
         {:30} {:.6}\n\
         {:30} {}\n\
         {:30} {}\n",
        comparison.name_a,
        comparison.name_b,
        "-".repeat(60),
        "SNR L2 distance:",
        comparison.snr_l2_distance,
        "Max alpha_cumprod diff:",
        comparison.max_alpha_diff,
        "Transition timestep diff:",
        comparison.transition_diff,
        "Higher SNR schedule:",
        comparison.higher_snr_schedule,
    )
}

/// Format a ScheduleAnalysis as a human-readable text summary.
pub fn format_schedule_summary(analysis: &ScheduleAnalysis) -> String {
    format!(
        "Schedule: {}\n\
         Timesteps:          {}\n\
         Transition step:    {}\n\
         Informative range:  {}..{}\n\
         SNR AUC:            {:.4}\n\
         Beta smoothness:    {:.6}\n",
        analysis.schedule_name,
        analysis.num_timesteps,
        analysis.transition_timestep,
        analysis.informative_range.0,
        analysis.informative_range.1,
        analysis.snr_auc,
        analysis.beta_smoothness,
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn sd21_schedule() -> NoiseSchedule {
        NoiseSchedule::scaled_linear(1000, 0.00085, 0.012).unwrap()
    }

    fn cosine_schedule() -> NoiseSchedule {
        NoiseSchedule::cosine(1000, 0.008).unwrap()
    }

    // 1. Linear betas are linearly increasing
    #[test]
    fn test_linear_betas_increasing() {
        let sched = NoiseSchedule::linear(100, 0.0001, 0.02).unwrap();
        for i in 1..sched.betas.len() {
            assert!(
                sched.betas[i] >= sched.betas[i - 1],
                "betas should be non-decreasing"
            );
        }
    }

    // 2. Linear alphas_cumprod is strictly decreasing
    #[test]
    fn test_linear_alphas_cumprod_decreasing() {
        let sched = NoiseSchedule::linear(100, 0.0001, 0.02).unwrap();
        for i in 1..sched.alphas_cumprod.len() {
            assert!(
                sched.alphas_cumprod[i] < sched.alphas_cumprod[i - 1],
                "alphas_cumprod should be strictly decreasing"
            );
        }
    }

    // 3. First alpha_cumprod ≈ 1.0
    #[test]
    fn test_linear_first_alpha_cumprod_near_one() {
        let sched = NoiseSchedule::linear(100, 0.0001, 0.02).unwrap();
        assert!(
            (sched.alphas_cumprod[0] - 1.0).abs() < 0.01,
            "first alpha_cumprod should be near 1, got {}",
            sched.alphas_cumprod[0]
        );
    }

    // 4. Last alpha_cumprod ≈ 0.0 (near zero)
    #[test]
    fn test_linear_last_alpha_cumprod_near_zero() {
        let sched = NoiseSchedule::linear(1000, 0.0001, 0.02).unwrap();
        let last = *sched.alphas_cumprod.last().unwrap();
        assert!(
            last < 0.05,
            "last alpha_cumprod should be near 0, got {}",
            last
        );
    }

    // 5. ScaledLinear betas smaller than linear for same params
    #[test]
    fn test_scaled_linear_betas_smaller_than_linear() {
        let linear = NoiseSchedule::linear(1000, 0.00085, 0.012).unwrap();
        let scaled = sd21_schedule();
        // Mean beta should be smaller for scaled-linear (square of mid-range sqrt)
        let mean_linear: f32 = linear.betas.iter().sum::<f32>() / linear.betas.len() as f32;
        let mean_scaled: f32 = scaled.betas.iter().sum::<f32>() / scaled.betas.len() as f32;
        assert!(
            mean_scaled < mean_linear,
            "scaled-linear mean beta ({}) should be < linear ({})",
            mean_scaled,
            mean_linear
        );
    }

    // 6. Cosine alphas_cumprod decreases smoothly (all consecutive diffs < 0)
    #[test]
    fn test_cosine_alphas_cumprod_smooth_decrease() {
        let sched = cosine_schedule();
        for i in 1..sched.alphas_cumprod.len() {
            assert!(
                sched.alphas_cumprod[i] < sched.alphas_cumprod[i - 1],
                "cosine alphas_cumprod not strictly decreasing at i={}",
                i
            );
        }
    }

    // 7. Sigmoid schedule betas all in (0, 1)
    #[test]
    fn test_sigmoid_betas_in_range() {
        let sched = NoiseSchedule::sigmoid_schedule(100, -3.0, 3.0).unwrap();
        for (i, &b) in sched.betas.iter().enumerate() {
            assert!(b > 0.0 && b < 1.0, "beta out of range at i={}: {}", i, b);
        }
    }

    // 8. SNR at t=0 is large (high signal)
    #[test]
    fn test_snr_high_at_t0() {
        let sched = sd21_schedule();
        let snr0 = sched.snr(0);
        assert!(snr0 > 10.0, "SNR at t=0 should be large, got {}", snr0);
    }

    // 9. SNR at t=T-1 is small (mostly noise)
    #[test]
    fn test_snr_low_at_t_final() {
        let sched = sd21_schedule();
        let t_last = sched.num_timesteps - 1;
        let snr_last = sched.snr(t_last);
        assert!(
            snr_last < 1.0,
            "SNR at t=T-1 should be small, got {}",
            snr_last
        );
    }

    // 10. forward_coefficients: s² + n² ≈ 1.0
    #[test]
    fn test_forward_coefficients_unit_preserving() {
        let sched = sd21_schedule();
        for t in [0, 250, 500, 750, 999] {
            let (s, n) = sched.forward_coefficients(t);
            let sum = s * s + n * n;
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "s²+n² should ≈ 1 at t={}, got {}",
                t,
                sum
            );
        }
    }

    // 11. noise_level = sqrt(1 - alpha_cumprod)
    #[test]
    fn test_noise_level_equals_sqrt_one_minus_alpha() {
        let sched = sd21_schedule();
        for t in [0, 500, 999] {
            let expected = (1.0 - sched.alphas_cumprod[t]).sqrt();
            let got = sched.noise_level(t);
            assert!(
                (got - expected).abs() < 1e-6,
                "noise_level mismatch at t={}: expected {}, got {}",
                t,
                expected,
                got
            );
        }
    }

    // 12. timestep_at_snr with SNR=1 returns middle-ish timestep
    #[test]
    fn test_timestep_at_snr_one_is_middle() {
        let sched = sd21_schedule();
        let t = sched.timestep_at_snr(1.0);
        let n = sched.num_timesteps;
        assert!(
            t > n / 4 && t < 3 * n / 4,
            "SNR=1 crossover should be in middle quarter, got t={}",
            t
        );
    }

    // 13. analyze_schedule transition_timestep is reasonable
    #[test]
    fn test_analyze_transition_timestep_reasonable() {
        let sched = sd21_schedule();
        let analysis = analyze_schedule(&sched, "sd21");
        assert!(
            analysis.transition_timestep < sched.num_timesteps,
            "transition timestep out of range: {}",
            analysis.transition_timestep
        );
    }

    // 14. analyze_schedule snr_values length == num_timesteps
    #[test]
    fn test_analyze_snr_values_length() {
        let sched = sd21_schedule();
        let analysis = analyze_schedule(&sched, "sd21");
        assert_eq!(
            analysis.snr_values.len(),
            sched.num_timesteps,
            "snr_values length mismatch"
        );
    }

    // 15. analyze_schedule signal_fractions all in [0, 1]
    #[test]
    fn test_analyze_signal_fractions_in_range() {
        let sched = sd21_schedule();
        let analysis = analyze_schedule(&sched, "sd21");
        for (i, &v) in analysis.signal_fractions.iter().enumerate() {
            assert!(
                (0.0..=1.0).contains(&v),
                "signal_fraction[{}] = {} out of [0,1]",
                i,
                v
            );
        }
    }

    // 16. analyze_schedule snr_auc > 0
    #[test]
    fn test_analyze_snr_auc_positive() {
        let sched = sd21_schedule();
        let analysis = analyze_schedule(&sched, "sd21");
        assert!(
            analysis.snr_auc > 0.0,
            "snr_auc should be positive, got {}",
            analysis.snr_auc
        );
    }

    // 17. compare_schedules snr_l2_distance > 0 for different schedules
    #[test]
    fn test_compare_different_schedules_nonzero_distance() {
        let a = sd21_schedule();
        let b = cosine_schedule();
        let cmp = compare_schedules(&a, "sd21", &b, "cosine").unwrap();
        assert!(
            cmp.snr_l2_distance > 0.0,
            "L2 distance between different schedules should be >0"
        );
    }

    // 18. compare_schedules same schedule vs itself → distance = 0
    #[test]
    fn test_compare_same_schedule_zero_distance() {
        let a = sd21_schedule();
        let b = sd21_schedule();
        let cmp = compare_schedules(&a, "a", &b, "b").unwrap();
        assert!(
            cmp.snr_l2_distance < 1e-6,
            "L2 distance of schedule vs itself should be 0, got {}",
            cmp.snr_l2_distance
        );
    }

    // 19. compare_schedules timestep mismatch → error
    #[test]
    fn test_compare_timestep_mismatch_error() {
        let a = NoiseSchedule::linear(500, 0.0001, 0.02).unwrap();
        let b = NoiseSchedule::linear(1000, 0.0001, 0.02).unwrap();
        let result = compare_schedules(&a, "a", &b, "b");
        assert!(
            matches!(result, Err(ScheduleAnalysisError::TimestepMismatch { .. })),
            "expected TimestepMismatch error"
        );
    }

    // 20. ddim_timesteps length == num_steps
    #[test]
    fn test_ddim_timesteps_length() {
        let ts = ddim_timesteps(1000, 50);
        assert_eq!(ts.len(), 50, "ddim_timesteps length should equal num_steps");
    }

    // 21. ddim_timesteps contains endpoints near 0 and T-1
    #[test]
    fn test_ddim_timesteps_contains_endpoints() {
        let total = 1000usize;
        let ts = ddim_timesteps(total, 50);
        // First element (reversed) should be near T-1
        assert!(
            *ts.first().unwrap() >= total - 5,
            "first ddim timestep should be near T-1, got {}",
            ts.first().unwrap()
        );
        // Last element should be 0
        assert_eq!(*ts.last().unwrap(), 0, "last ddim timestep should be 0");
    }

    // 22. analyze_ddim_sampling step_snrs length == num_steps
    #[test]
    fn test_ddim_step_snrs_length() {
        let sched = sd21_schedule();
        let ddim = analyze_ddim_sampling(&sched, 50);
        assert_eq!(
            ddim.step_snrs.len(),
            50,
            "step_snrs length should equal num_inference_steps"
        );
    }

    // 23. analyze_ddim_sampling quality_score in (0, 1]
    #[test]
    fn test_ddim_quality_score_in_range() {
        let sched = sd21_schedule();
        let ddim = analyze_ddim_sampling(&sched, 50);
        assert!(
            ddim.quality_score > 0.0 && ddim.quality_score <= 1.0,
            "quality_score should be in (0,1], got {}",
            ddim.quality_score
        );
    }

    // 24. format_snr_curve_ascii returns string with correct number of lines
    #[test]
    fn test_ascii_curve_line_count() {
        let sched = sd21_schedule();
        let art = format_snr_curve_ascii(&sched, 60, 20);
        let lines: Vec<&str> = art.lines().collect();
        assert_eq!(
            lines.len(),
            20,
            "ASCII art should have 20 lines, got {}",
            lines.len()
        );
    }

    // 25. format_schedule_summary returns non-empty string
    #[test]
    fn test_format_schedule_summary_nonempty() {
        let sched = sd21_schedule();
        let analysis = analyze_schedule(&sched, "test");
        let summary = format_schedule_summary(&analysis);
        assert!(!summary.is_empty(), "schedule summary should be non-empty");
    }

    // --- Karras EDM tests (26–37) ---

    // 26. karras_sigma_at endpoints: i=0 → sigma_max, i=N-1 → sigma_min
    #[test]
    fn test_karras_sigma_at_endpoints() {
        let n = 20usize;
        let sigma_min = 0.002_f32;
        let sigma_max = 80.0_f32;
        let rho = 7.0_f32;
        let s0 = karras_sigma_at(0, n, sigma_min, sigma_max, rho);
        let s_last = karras_sigma_at(n - 1, n, sigma_min, sigma_max, rho);
        assert!(
            (s0 - sigma_max).abs() < 1e-4,
            "i=0 should give sigma_max={}, got {}",
            sigma_max,
            s0
        );
        assert!(
            (s_last - sigma_min).abs() < 1e-5,
            "i=N-1 should give sigma_min={}, got {}",
            sigma_min,
            s_last
        );
    }

    // 27. karras sigmas are strictly monotone decreasing (noisiest first)
    #[test]
    fn test_karras_sigma_monotone_decreasing() {
        let n = 20usize;
        let sigma_min = 0.002_f32;
        let sigma_max = 80.0_f32;
        let rho = 7.0_f32;
        let sigmas: Vec<f32> = (0..n)
            .map(|i| karras_sigma_at(i, n, sigma_min, sigma_max, rho))
            .collect();
        for i in 1..n {
            assert!(
                sigmas[i] < sigmas[i - 1],
                "sigmas should be strictly decreasing: sigmas[{}]={} >= sigmas[{}]={}",
                i,
                sigmas[i],
                i - 1,
                sigmas[i - 1]
            );
        }
    }

    // 28. alphas_cumprod from Karras schedule are monotone increasing
    //     (because sigma decreases ↓, alpha_bar = 1/(1+σ²) increases ↑)
    #[test]
    fn test_karras_alpha_bar_monotone_increasing() {
        let sched = NoiseSchedule::karras(20, 0.002, 80.0, 7.0).unwrap();
        for i in 1..sched.alphas_cumprod.len() {
            assert!(
                sched.alphas_cumprod[i] > sched.alphas_cumprod[i - 1],
                "karras alphas_cumprod should be strictly increasing at i={}: {} <= {}",
                i,
                sched.alphas_cumprod[i],
                sched.alphas_cumprod[i - 1]
            );
        }
    }

    // 29. All alphas_cumprod values are in (0, 1)
    #[test]
    fn test_karras_alpha_bar_range() {
        let sched = NoiseSchedule::karras(20, 0.002, 80.0, 7.0).unwrap();
        for (i, &ab) in sched.alphas_cumprod.iter().enumerate() {
            assert!(
                ab > 0.0 && ab < 1.0,
                "alphas_cumprod[{}]={} not in (0,1)",
                i,
                ab
            );
        }
    }

    // 30. Default EDM params produce a valid schedule
    #[test]
    fn test_karras_default_params() {
        let sched = NoiseSchedule::karras(20, 0.002, 80.0, 7.0).unwrap();
        assert_eq!(sched.num_timesteps, 20);
        assert_eq!(sched.betas.len(), 20);
        assert_eq!(sched.alphas_cumprod.len(), 20);
        assert_eq!(sched.alphas_cumprod_prev.len(), 20);
        assert_eq!(sched.sqrt_alphas_cumprod.len(), 20);
        assert_eq!(sched.sqrt_one_minus_alphas_cumprod.len(), 20);
    }

    // 31. rho=1 → sigmas linearly spaced between sigma_min and sigma_max
    #[test]
    fn test_karras_rho_1_is_linear() {
        let n = 10usize;
        let sigma_min = 0.1_f32;
        let sigma_max = 10.0_f32;
        let rho = 1.0_f32;
        let sigmas: Vec<f32> = (0..n)
            .map(|i| karras_sigma_at(i, n, sigma_min, sigma_max, rho))
            .collect();
        // With rho=1: σ_i = sigma_max + i/(n-1)*(sigma_min - sigma_max)
        for (i, &sigma_val) in sigmas.iter().enumerate() {
            let expected = sigma_max + (i as f32 / (n - 1) as f32) * (sigma_min - sigma_max);
            assert!(
                (sigma_val - expected).abs() < 1e-5,
                "rho=1 sigma[{}] should be {}, got {}",
                i,
                expected,
                sigma_val
            );
        }
    }

    // 32. n=1 → Err(InvalidTimesteps)
    #[test]
    fn test_karras_invalid_too_few_steps() {
        let result = NoiseSchedule::karras(1, 0.002, 80.0, 7.0);
        assert!(
            matches!(result, Err(ScheduleAnalysisError::InvalidTimesteps(1))),
            "expected InvalidTimesteps(1), got {:?}",
            result
        );
    }

    // 33. sigma_min=0.0 → Err(InvalidBeta)
    #[test]
    fn test_karras_invalid_sigma_min_zero() {
        let result = NoiseSchedule::karras(20, 0.0, 80.0, 7.0);
        assert!(
            matches!(result, Err(ScheduleAnalysisError::InvalidBeta(_, 0))),
            "expected InvalidBeta(_, 0) for sigma_min=0, got {:?}",
            result
        );
    }

    // 34. sigma_max <= sigma_min → Err(InvalidBeta)
    #[test]
    fn test_karras_invalid_sigma_max_too_small() {
        // sigma_max == sigma_min
        let result_eq = NoiseSchedule::karras(20, 5.0, 5.0, 7.0);
        assert!(
            matches!(result_eq, Err(ScheduleAnalysisError::InvalidBeta(_, 1))),
            "expected InvalidBeta(_, 1) for sigma_max == sigma_min, got {:?}",
            result_eq
        );
        // sigma_max < sigma_min
        let result_lt = NoiseSchedule::karras(20, 10.0, 5.0, 7.0);
        assert!(
            matches!(result_lt, Err(ScheduleAnalysisError::InvalidBeta(_, 1))),
            "expected InvalidBeta(_, 1) for sigma_max < sigma_min, got {:?}",
            result_lt
        );
    }

    // 35. Both Karras and cosine produce strictly decreasing alpha_bars
    #[test]
    fn test_karras_vs_cosine_structure() {
        let karras = NoiseSchedule::karras(100, 0.002, 80.0, 7.0).unwrap();
        let cosine = NoiseSchedule::cosine(100, 0.008).unwrap();
        // Karras: increasing alpha_bar (sigma decreases)
        for i in 1..karras.alphas_cumprod.len() {
            assert!(
                karras.alphas_cumprod[i] > karras.alphas_cumprod[i - 1],
                "karras alphas_cumprod not increasing at i={}",
                i
            );
        }
        // Cosine: strictly decreasing alpha_bar (noise increases)
        for i in 1..cosine.alphas_cumprod.len() {
            assert!(
                cosine.alphas_cumprod[i] < cosine.alphas_cumprod[i - 1],
                "cosine alphas_cumprod not decreasing at i={}",
                i
            );
        }
    }

    // 36. schedule_type field matches the Karras variant
    #[test]
    fn test_karras_schedule_type_variant() {
        let sched = NoiseSchedule::karras(20, 0.002, 80.0, 7.0).unwrap();
        assert!(
            matches!(
                sched.schedule_type,
                ScheduleType::Karras { sigma_min, sigma_max, rho }
                    if (sigma_min - 0.002).abs() < 1e-6
                        && (sigma_max - 80.0).abs() < 1e-4
                        && (rho - 7.0).abs() < 1e-6
            ),
            "unexpected schedule_type: {:?}",
            sched.schedule_type
        );
    }

    // 37. SNR > 0 for all timesteps
    #[test]
    fn test_karras_snr_positive() {
        let sched = NoiseSchedule::karras(20, 0.002, 80.0, 7.0).unwrap();
        for t in 0..sched.num_timesteps {
            let snr = sched.snr(t);
            assert!(snr > 0.0, "SNR should be positive at t={}, got {}", t, snr);
        }
    }
}
