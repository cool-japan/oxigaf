//! Shared bilinear sampling helper for flat RGB image buffers.
//!
//! Used both by equirectangular (lat-long) panorama projection
//! ([`super::projection`]) and by [`super::cubemap::CubemapProbe`] face sampling.

/// Bilinear sample from an f32 RGB row-major image.
///
/// Returns `[0.0; 3]` for a zero-width or zero-height image instead of
/// underflowing `width - 1` / `height - 1` (which would panic in `.min()`
/// arithmetic on `usize`).
#[inline]
pub(super) fn bilinear_sample_rgb(
    image: &[f32],
    width: u32,
    height: u32,
    u: f32,
    v: f32,
) -> [f32; 3] {
    if width == 0 || height == 0 {
        return [0.0; 3];
    }
    let w = width as usize;
    let h = height as usize;
    let px = (u * width as f32 - 0.5).max(0.0);
    let py = (v * height as f32 - 0.5).max(0.0);
    let x0 = (px.floor() as usize).min(w.saturating_sub(2));
    let y0 = (py.floor() as usize).min(h.saturating_sub(2));
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let tx = px - x0 as f32;
    let ty = py - y0 as f32;

    let idx = |row: usize, col: usize| -> [f32; 3] {
        let base = (row * w + col) * 3;
        [image[base], image[base + 1], image[base + 2]]
    };

    let c00 = idx(y0, x0);
    let c10 = idx(y0, x1);
    let c01 = idx(y1, x0);
    let c11 = idx(y1, x1);

    let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
    [
        lerp(lerp(c00[0], c10[0], tx), lerp(c01[0], c11[0], tx), ty),
        lerp(lerp(c00[1], c10[1], tx), lerp(c01[1], c11[1], tx), ty),
        lerp(lerp(c00[2], c10[2], tx), lerp(c01[2], c11[2], tx), ty),
    ]
}
