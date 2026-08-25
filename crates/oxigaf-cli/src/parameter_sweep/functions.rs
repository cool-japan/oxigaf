//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{ParamSpec, ParamValue, SweepError, SweepSummary, SweepTrial};

pub(super) fn xorshift64(state: &mut u64) -> u64 {
    (*state) ^= (*state) << 13;
    (*state) ^= (*state) >> 7;
    (*state) ^= (*state) << 17;
    if *state == 0 {
        *state = 1;
    }
    *state
}

pub(super) fn xorshift_f64(state: &mut u64) -> f64 {
    (xorshift64(state) >> 11) as f64 / (1u64 << 53) as f64
}

/// Sample a continuous parameter value.
///
/// If `log_scale` is true, samples uniformly in log-space: `exp(uniform(log(low), log(high)))`.
///
/// # Errors
/// Returns `SweepError::InvalidParam` if `low >= high` or if `log_scale`
/// and `low <= 0.0`.
pub fn sweep_sample_continuous(
    low: f64,
    high: f64,
    log_scale: bool,
    state: &mut u64,
) -> Result<f64, SweepError> {
    if low >= high {
        return Err(SweepError::InvalidParam(format!(
            "low ({}) must be less than high ({})",
            low, high
        )));
    }
    if log_scale && low <= 0.0 {
        return Err(SweepError::InvalidParam(format!(
            "log_scale requires low > 0.0, got low={}",
            low
        )));
    }
    if log_scale {
        let log_low = low.ln();
        let log_high = high.ln();
        let t = xorshift_f64(state);
        Ok((log_low + t * (log_high - log_low)).exp())
    } else {
        let t = xorshift_f64(state);
        Ok(low + t * (high - low))
    }
}

/// Sample a value uniformly from a discrete list.
///
/// # Errors
/// Returns `SweepError::EmptyDiscrete` if `values` is empty.
pub fn sweep_sample_discrete(values: &[f64], state: &mut u64) -> Result<f64, SweepError> {
    if values.is_empty() {
        return Err(SweepError::EmptyDiscrete);
    }
    let idx = xorshift64(state) as usize % values.len();
    Ok(values[idx])
}

/// Compute the per-dimension indices for a mixed-radix grid counter.
///
/// Given dimension sizes `dims` and a trial index, returns the index into each
/// dimension using a mixed-radix (row-major) decomposition.
///
/// For dimension `i`, the index is:
/// `(trial_idx / product(dims[i+1..])) % dims[i]`
pub fn sweep_grid_indices(dims: &[usize], trial_idx: usize) -> Vec<usize> {
    let n = dims.len();
    let mut indices = vec![0usize; n];
    let remaining = trial_idx;
    // Compute suffix products: suffix[i] = product of dims[i+1..]
    let mut suffix = vec![1usize; n];
    for i in (0..n.saturating_sub(1)).rev() {
        suffix[i] = suffix[i + 1].saturating_mul(dims[i + 1]);
    }
    for i in 0..n {
        if dims[i] == 0 {
            indices[i] = 0;
        } else {
            indices[i] = (remaining / suffix[i]) % dims[i];
        }
    }
    indices
}

/// Predict the score for a candidate parameter set using inverse-distance weighted KNN.
///
/// Uses K = min(5, number of completed trials).
/// Distance is computed in normalized parameter space (each dimension scaled
/// to \[0,1\] using `specs`' declared bounds; see `param_value_distance`).
/// Weight = 1 / (dist^2 + 1e-8).
/// Returns 0.0 if there are no scored trials.
pub fn sweep_surrogate_predict(
    trials: &[SweepTrial],
    params: &[(String, ParamValue)],
    specs: &[ParamSpec],
) -> f64 {
    let scored: Vec<&SweepTrial> = trials.iter().filter(|t| t.score.is_some()).collect();
    if scored.is_empty() {
        return 0.0;
    }
    let k = scored.len().min(5);

    // Compute distances from candidate to all scored trials.
    let mut dist_scores: Vec<(f64, f64)> = scored
        .iter()
        .map(|trial| {
            let dist = param_distance(params, &trial.params, specs);
            let score = trial.score.unwrap_or(0.0);
            (dist, score)
        })
        .collect();

    // Sort by distance ascending.
    dist_scores.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Take K nearest.
    let nearest = &dist_scores[..k];

    // Inverse-distance weighting.
    let mut weight_sum = 0.0f64;
    let mut weighted_score = 0.0f64;
    for &(dist, score) in nearest {
        let w = 1.0 / (dist * dist + 1e-8);
        weight_sum += w;
        weighted_score += w * score;
    }
    if weight_sum < 1e-15 {
        return scored[0].score.unwrap_or(0.0);
    }
    weighted_score / weight_sum
}

/// Compute the importance of each parameter based on rank correlation with scores.
///
/// Uses Spearman rank correlation: `rho = 1 - 6*sum(d^2) / (n*(n^2-1))`.
/// Returns importances normalized to sum to 1.0.
/// Returns an empty vec if fewer than 2 scored trials exist.
pub fn sweep_param_importance(trials: &[SweepTrial], specs: &[ParamSpec]) -> Vec<(String, f64)> {
    let scored: Vec<&SweepTrial> = trials.iter().filter(|t| t.score.is_some()).collect();
    if scored.len() < 2 {
        return specs.iter().map(|s| (s.name().to_string(), 0.0)).collect();
    }
    let n = scored.len();

    // Score ranks.
    let scores: Vec<f64> = scored.iter().map(|t| t.score.unwrap_or(0.0)).collect();
    let score_ranks = compute_ranks(&scores);

    let mut raw_importances: Vec<(String, f64)> = Vec::with_capacity(specs.len());

    for spec in specs {
        let name = spec.name().to_string();
        // Extract numeric proxy for each trial for this param.
        let param_vals: Vec<f64> = scored
            .iter()
            .map(|trial| numeric_proxy_for_param(trial, spec))
            .collect();
        let param_ranks = compute_ranks(&param_vals);

        // Spearman rank correlation.
        let rho = if n <= 2 {
            0.0
        } else {
            let d_sq_sum: f64 = score_ranks
                .iter()
                .zip(param_ranks.iter())
                .map(|(r1, r2)| (r1 - r2).powi(2))
                .sum();
            let n_f = n as f64;
            1.0 - 6.0 * d_sq_sum / (n_f * (n_f * n_f - 1.0))
        };

        raw_importances.push((name, rho.abs()));
    }

    // Normalize to sum 1.0.
    let total: f64 = raw_importances.iter().map(|(_, v)| v).sum();
    if total < 1e-12 {
        let uniform = if specs.is_empty() {
            0.0
        } else {
            1.0 / specs.len() as f64
        };
        raw_importances.iter_mut().for_each(|(_, v)| *v = uniform);
    } else {
        raw_importances.iter_mut().for_each(|(_, v)| *v /= total);
    }
    raw_importances
}

/// Format a trial as a human-readable string.
pub fn format_sweep_trial(trial: &SweepTrial) -> String {
    let mut parts = Vec::with_capacity(trial.params.len() + 2);
    parts.push(format!("Trial #{}", trial.id));
    for (name, value) in &trial.params {
        parts.push(format!("{}={}", name, value));
    }
    if let Some(score) = trial.score {
        parts.push(format!("score={:.6}", score));
    } else {
        parts.push("score=pending".to_string());
    }
    parts.join(", ")
}

/// Format a sweep summary as a human-readable multi-line string.
pub fn format_sweep_summary(summary: &SweepSummary) -> String {
    let mut lines = Vec::new();
    lines.push(format!(
        "Sweep Summary: {}/{} trials completed",
        summary.completed_trials, summary.total_trials
    ));
    if let Some(b) = summary.best_score {
        lines.push(format!("  Best score:  {:.6}", b));
    }
    if let Some(w) = summary.worst_score {
        lines.push(format!("  Worst score: {:.6}", w));
    }
    if let Some(m) = summary.mean_score {
        lines.push(format!("  Mean score:  {:.6}", m));
    }
    if let Some(s) = summary.std_score {
        lines.push(format!("  Std score:   {:.6}", s));
    }
    if !summary.param_importances.is_empty() {
        lines.push("  Parameter importances:".to_string());
        for (name, imp) in &summary.param_importances {
            lines.push(format!("    {}: {:.4}", name, imp));
        }
    }
    lines.join("\n")
}

/// Compute the Hyperband bracket: list of `(n_configs, budget)` per round.
///
/// Rounds are ordered from the innermost (fewest configs, largest budget)
/// to the outermost (most configs, smallest budget).
///
/// Formula (matches Li et al. 2016's published Hyperband algorithm):
/// - `s_max = floor(log_eta(max_iter))`
/// - Round `r` (0..=s_max): `n_r = ceil((s_max+1) / (s_max-r+1) * eta^(s_max-r))`,
///   `budget_r = floor(max_iter / eta^(s_max-r))`
/// - Returns rounds from `r=s_max` down to `r=0`.
///
/// The `/ (s_max-r+1)` factor was previously omitted, which over-allocated
/// configurations in the wide/cheap rounds: for `max_iter=81, eta=3` this
/// used to yield `5, 15, 45, 135, 405` configs per round instead of the
/// correct `5, 8, 15, 34, 81`, so a caller budgeting total work from these
/// counts planned several times the intended amount.
pub fn hyperband_bracket(max_iter: usize, eta: usize) -> Vec<(usize, usize)> {
    if max_iter == 0 || eta < 2 {
        return Vec::new();
    }
    // Use round-then-floor to avoid floating-point underflow on exact powers.
    let raw_log = (max_iter as f64).log(eta as f64);
    // If raw_log is within 1e-9 of an integer (e.g. log_3(243) ≈ 5.0 - eps),
    // round to that integer first so exact powers are handled correctly.
    let s_max = if (raw_log - raw_log.round()).abs() < 1e-9 {
        raw_log.round() as usize
    } else {
        raw_log.floor() as usize
    };
    let mut rounds = Vec::with_capacity(s_max + 1);

    // r iterates from s_max down to 0 (innermost first).
    for r_rev in 0..=s_max {
        let r = s_max - r_rev; // actual r value
        let power = s_max.saturating_sub(r) as u32;
        let eta_pow = (eta as f64).powi(power as i32);
        // `power` here is `s_max - r`, i.e. the paper's `s` in `eta^s`; the
        // published formula divides by `(s + 1)` = `(power + 1)`, not by
        // `(r + 1)` (this function's own, differently-scoped `r`).
        let n_configs = (((s_max + 1) as f64 / (power + 1) as f64) * eta_pow).ceil() as usize;
        let n_configs = n_configs.max(1);
        let budget = if eta_pow < 1.0 {
            0usize
        } else {
            ((max_iter as f64) / eta_pow).floor() as usize
        };
        rounds.push((n_configs, budget));
    }
    rounds
}

/// Sample a full parameter set randomly for all specs.
pub(super) fn sample_params_random(
    specs: &[ParamSpec],
    rng: &mut u64,
) -> Result<Vec<(String, ParamValue)>, SweepError> {
    let mut params = Vec::with_capacity(specs.len());
    for spec in specs {
        let pv = match spec {
            ParamSpec::Continuous {
                name,
                low,
                high,
                log_scale,
            } => {
                let v = sweep_sample_continuous(*low, *high, *log_scale, rng)?;
                (name.clone(), ParamValue::Float(v))
            }
            ParamSpec::Discrete { name, values } => {
                let v = sweep_sample_discrete(values, rng)?;
                (name.clone(), ParamValue::Float(v))
            }
            ParamSpec::Categorical { name, choices } => {
                if choices.is_empty() {
                    return Err(SweepError::EmptyDiscrete);
                }
                let idx = xorshift64(rng) as usize % choices.len();
                let c = choices[idx].clone();
                (name.clone(), ParamValue::Choice(c))
            }
        };
        params.push(pv);
    }
    Ok(params)
}

/// Compute the L2 distance between two parameter sets in normalized space.
///
/// Categorical: 0.0 if same choice, 1.0 if different.
/// Float: difference normalized to [0,1] using each parameter's declared
/// bounds in `specs` (see [`param_value_distance`]).
fn param_distance(
    a: &[(String, ParamValue)],
    b: &[(String, ParamValue)],
    specs: &[ParamSpec],
) -> f64 {
    let mut sq_sum = 0.0f64;
    for (name_a, val_a) in a {
        if let Some((_, val_b)) = b.iter().find(|(n, _)| n == name_a) {
            let spec = specs.iter().find(|s| s.name() == name_a);
            let d = param_value_distance(spec, val_a, val_b);
            sq_sum += d * d;
        }
    }
    sq_sum.sqrt()
}

/// Distance between two `ParamValue`s in \[0, 1\].
///
/// `spec` (matched by name to the parameter) normalizes a `Float` distance
/// by the parameter's actual declared range instead of the un-normalized
/// `diff / (diff + 1.0)` this used previously -- which put every `Float`
/// parameter on the same soft [0,1) scale regardless of magnitude (a
/// learning rate spanning `1e-4..1e-2` always scored near 0 next to a
/// `0..1000` parameter always near 1, making the KNN surrogate in
/// [`sweep_surrogate_predict`] effectively blind to small-range dimensions).
/// See [`normalized_float_distance`] for the normalization itself, used
/// when `spec` has no usable bounds (no match, or a degenerate range).
pub(super) fn param_value_distance(
    spec: Option<&ParamSpec>,
    a: &ParamValue,
    b: &ParamValue,
) -> f64 {
    match (a, b) {
        (ParamValue::Choice(ca), ParamValue::Choice(cb)) if ca == cb => 0.0,
        (ParamValue::Float(fa), ParamValue::Float(fb)) => normalized_float_distance(spec, *fa, *fb),
        // No `ParamSpec` variant currently produces an `Int` (see
        // `ParamValue`'s doc); kept as a total fallback, not reachable today.
        (ParamValue::Int(ia), ParamValue::Int(ib)) => {
            let diff = (ia - ib).unsigned_abs() as f64;
            diff / (diff + 1.0)
        }
        // Mixed types: treat as maximally different.
        _ => 1.0,
    }
}

/// Normalize the distance between two `Float` parameter values to \[0, 1\]
/// using `spec`'s declared bounds (log-space for a log-scaled
/// [`ParamSpec::Continuous`], matching how [`sweep_sample_continuous`]
/// samples it; linear range for `Discrete`'s `values`), falling back to the
/// spec-agnostic `diff / (diff + 1.0)` when no usable bounds are found.
pub(super) fn normalized_float_distance(spec: Option<&ParamSpec>, a: f64, b: f64) -> f64 {
    let bounds: Option<(f64, f64, bool)> = match spec {
        Some(ParamSpec::Continuous {
            low,
            high,
            log_scale,
            ..
        }) => Some((*low, *high, *log_scale)),
        Some(ParamSpec::Discrete { values, .. }) if !values.is_empty() => {
            let low = values.iter().copied().fold(f64::INFINITY, f64::min);
            let high = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            Some((low, high, false))
        }
        _ => None,
    };

    if let Some((low, high, log_scale)) = bounds {
        if log_scale && low > 0.0 && high > 0.0 {
            let log_low = low.ln();
            let range = high.ln() - log_low;
            if range.abs() > 1e-12 {
                let na = (a.max(f64::MIN_POSITIVE).ln() - log_low) / range;
                let nb = (b.max(f64::MIN_POSITIVE).ln() - log_low) / range;
                return (na - nb).abs().clamp(0.0, 1.0);
            }
        } else {
            let range = high - low;
            if range.abs() > 1e-12 {
                let na = (a - low) / range;
                let nb = (b - low) / range;
                return (na - nb).abs().clamp(0.0, 1.0);
            }
        }
    }

    // Fallback: no matching/usable spec (unmatched name, or a degenerate
    // zero-width / non-positive-for-log-scale range).
    let diff = (a - b).abs();
    diff / (diff + 1.0)
}

/// Extract a numeric proxy for a parameter value given its spec.
///
/// For Categorical, uses the index of the choice in the spec's choices list.
/// For Float/Int, uses the raw value.
fn numeric_proxy_for_param(trial: &SweepTrial, spec: &ParamSpec) -> f64 {
    let name = spec.name();
    let pv = trial.params.iter().find(|(n, _)| n == name).map(|(_, v)| v);
    match (spec, pv) {
        (ParamSpec::Categorical { choices, .. }, Some(ParamValue::Choice(c))) => {
            choices.iter().position(|ch| ch == c).unwrap_or(0) as f64
        }
        (_, Some(ParamValue::Float(f))) => *f,
        (_, Some(ParamValue::Int(i))) => *i as f64,
        _ => 0.0,
    }
}

/// Compute fractional ranks (1-indexed) for a slice of values.
///
/// Ties are broken by taking the average rank.
fn compute_ranks(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    if n == 0 {
        return Vec::new();
    }
    // Create (value, original_index) pairs sorted by value.
    let mut indexed: Vec<(f64, usize)> = values
        .iter()
        .cloned()
        .enumerate()
        .map(|(i, v)| (v, i))
        .collect();
    indexed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut ranks = vec![0.0f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        // Find the run of equal values.
        while j < n && (indexed[j].0 - indexed[i].0).abs() < f64::EPSILON {
            j += 1;
        }
        // Average rank for the tie group (1-indexed).
        let avg_rank = (i + 1 + j) as f64 / 2.0;
        for k in i..j {
            ranks[indexed[k].1] = avg_rank;
        }
        i = j;
    }
    ranks
}
