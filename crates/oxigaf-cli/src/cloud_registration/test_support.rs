//! Deterministic fixtures and comparison helpers shared by the registration
//! test modules.
//!
//! Compiled only under `cfg(test)`; nothing here is reachable from a normal
//! build.

use super::math::{vec3_len, vec3_sub};

/// Approximate scalar equality within `tol`.
pub(super) fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() <= tol
}

/// Approximate 3-vector equality (L2 distance within 1e-5).
pub(super) fn close3(a: [f32; 3], b: [f32; 3]) -> bool {
    vec3_len(vec3_sub(a, b)) <= 1e-5
}

/// Determinant of a row-major 3×3 matrix.
pub(super) fn mat3_det(m: [f32; 9]) -> f32 {
    m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
        + m[2] * (m[3] * m[7] - m[4] * m[6])
}

/// Deterministic pseudo-random cloud of `n` points in `[0, span)³`.
///
/// A plain LCG, so the same `seed` always yields the same cloud — tests that
/// compare two code paths must see byte-identical input.
pub(super) fn pseudo_cloud(n: usize, seed: u32, span: f32) -> Vec<f32> {
    let mut state = seed;
    (0..n * 3)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 8) as f32 / 16_777_216.0) * span
        })
        .collect()
}

/// Axis-aligned grid of `n` points with the given `spacing`, filled in
/// x-major order and truncated to exactly `n` points.
pub(super) fn grid_positions(n: usize, spacing: f32) -> Vec<f32> {
    let side = (n as f32).cbrt().ceil() as usize;
    let mut pts = Vec::with_capacity(n * 3);
    'outer: for ix in 0..side {
        for iy in 0..side {
            for iz in 0..side {
                if pts.len() / 3 >= n {
                    break 'outer;
                }
                pts.push(ix as f32 * spacing);
                pts.push(iy as f32 * spacing);
                pts.push(iz as f32 * spacing);
            }
        }
    }
    pts
}
