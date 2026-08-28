//! Low-level One Euro filter math shared by [`super::tracker::HeadTracker`]'s
//! live `OneEuro` mode and [`super::functions::one_euro_filter_sequence`]'s
//! offline batch variant.

use super::tracker::OneEuroState;

// ---------------------------------------------------------------------------
// One Euro filter helper
// ---------------------------------------------------------------------------

/// Low-pass filter cutoff from the derivative magnitude.
pub(super) fn one_euro_cutoff(min_cutoff: f32, beta: f32, dx: f32) -> f32 {
    min_cutoff + beta * dx.abs()
}

/// First-order low-pass filter coefficient.
pub(super) fn alpha_from_cutoff(cutoff_hz: f32, dt_ms: f32) -> f32 {
    let dt_s = dt_ms / 1000.0;
    let tau = 1.0 / (2.0 * std::f32::consts::PI * cutoff_hz);
    1.0 / (1.0 + tau / dt_s.max(f32::EPSILON))
}

/// Advance the One Euro filter by one step given the real inter-sample
/// `dt_ms` (ignored on the first call for a fresh `state`, which just
/// seeds the filter).
///
/// Modifies `state` in place.
pub(super) fn one_euro_step(
    x: f32,
    dt_ms: f32,
    state: &mut OneEuroState,
    min_cutoff: f32,
    beta: f32,
) -> f32 {
    if let (Some(xp), Some(dxp)) = (state.x_prev, state.dx_prev) {
        let dt_ms = dt_ms.max(f32::EPSILON);
        // Derivative low-pass (cutoff 1 Hz).
        let d_cutoff = 1.0f32;
        let a_d = alpha_from_cutoff(d_cutoff, dt_ms);
        let dx = (x - xp) / dt_ms;
        let dx_hat = a_d * dx + (1.0 - a_d) * dxp;

        let cutoff = one_euro_cutoff(min_cutoff, beta, dx_hat);
        let a = alpha_from_cutoff(cutoff, dt_ms);
        let x_hat = a * x + (1.0 - a) * xp;

        state.x_prev = Some(x_hat);
        state.dx_prev = Some(dx_hat);
        x_hat
    } else {
        // First sample: initialize.
        state.x_prev = Some(x);
        state.dx_prev = Some(0.0);
        x
    }
}
