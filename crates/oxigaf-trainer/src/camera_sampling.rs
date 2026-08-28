//! Camera view sampling strategies for training.
//!
//! Provides several geometric distributions for drawing camera positions on
//! (or around) a sphere centred at a configurable look-at point:
//!
//! * [`SamplingStrategy::Random`]      — uniform spherical distribution.
//! * [`SamplingStrategy::Spiral`]      — pole-to-pole azimuthal spiral.
//! * [`SamplingStrategy::Hemisphere`]  — uniform grid over upper hemisphere.
//! * [`SamplingStrategy::Turntable`]   — fixed elevation, uniform azimuth.
//!
//! All strategies produce [`CameraView`] structs that expose a ready-to-use
//! column-major 4×4 look-at matrix via [`CameraView::view_matrix`].

use rand::{Rng, RngExt};
use std::f32::consts::{FRAC_PI_2, TAU};

// ---------------------------------------------------------------------------
// Sampling strategy
// ---------------------------------------------------------------------------

/// The geometric distribution used when drawing camera positions.
#[derive(Debug, Clone)]
pub enum SamplingStrategy {
    /// Uniform random distribution over the full sphere.
    Random,

    /// Points spiralling from pole to pole: azimuth sweeps `rounds` full
    /// revolutions while the polar angle traverses the sphere linearly in
    /// `z` (giving equal-area coverage per unit `z`, i.e. per unit height).
    ///
    /// `rounds` controls how many full azimuthal revolutions the spiral
    /// completes over one full pole-to-pole traversal (typical range 1–5).
    /// Both sampling entry points map a traversal parameter `t ∈ [0, 1]` to
    /// a point via the *same* formula, `azimuth = TAU * rounds * t`, so
    /// `rounds` means the same thing whether points come from
    /// [`CameraSampler::sample`] (a batch of `n` points spread evenly across
    /// the traversal) or [`CameraSampler::sample_iter`] (a single point at
    /// the phase implied by the current training iteration).
    Spiral { rounds: f32 },

    /// Regular `num_phi × num_theta` grid over the upper hemisphere
    /// (`theta` ∈ [0, π/2]).
    Hemisphere { num_phi: u32, num_theta: u32 },

    /// Fixed elevation angle with uniform azimuth, producing a "turntable"
    /// rotation around the subject.
    ///
    /// `elevation_deg` is measured from the horizontal plane in degrees;
    /// positive values look down at the subject.
    Turntable { elevation_deg: f32 },
}

// ---------------------------------------------------------------------------
// CameraView
// ---------------------------------------------------------------------------

/// A single camera pose expressed as eye, centre, and up vectors.
#[derive(Debug, Clone)]
pub struct CameraView {
    /// Camera position in world space.
    pub eye: [f32; 3],
    /// The point the camera looks at (usually the scene centre).
    pub center: [f32; 3],
    /// World-space up vector (usually [0, 1, 0]).
    pub up: [f32; 3],
}

impl CameraView {
    /// Compute the look-at **view matrix** (column-major, 4×4 f32).
    ///
    /// The matrix transforms world-space coordinates into camera space.
    /// The convention is right-handed: −Z points toward the viewer.
    pub fn view_matrix(&self) -> [[f32; 4]; 4] {
        let eye = Vec3::from(self.eye);
        let center = Vec3::from(self.center);
        let up = Vec3::from(self.up);

        // Forward (from eye toward center, then negate for right-handed −Z).
        let f = (center - eye).normalize_or_z();
        // Right = forward × up.
        let r = f.cross(up).normalize_or_z();
        // Recomputed up = right × forward.
        let u = r.cross(f);

        // Translation component.
        let tx = -r.dot(eye);
        let ty = -u.dot(eye);
        let tz = f.dot(eye);

        // Column-major storage: column 0, column 1, column 2, column 3.
        [
            [r.x, u.x, -f.x, 0.0],
            [r.y, u.y, -f.y, 0.0],
            [r.z, u.z, -f.z, 0.0],
            [tx, ty, tz, 1.0],
        ]
    }
}

// ---------------------------------------------------------------------------
// CameraSampler
// ---------------------------------------------------------------------------

/// Draws camera positions according to a [`SamplingStrategy`].
#[derive(Debug, Clone)]
pub struct CameraSampler {
    /// Distribution used to generate positions.
    strategy: SamplingStrategy,
    /// Distance of the camera from `center`.
    radius: f32,
    /// The world-space point the camera looks at.
    center: [f32; 3],
}

impl CameraSampler {
    /// Create a new sampler centred at the origin.
    pub fn new(strategy: SamplingStrategy, radius: f32) -> Self {
        Self {
            strategy,
            radius,
            center: [0.0, 0.0, 0.0],
        }
    }

    /// Create a new sampler with an explicit look-at centre.
    pub fn with_center(strategy: SamplingStrategy, radius: f32, center: [f32; 3]) -> Self {
        Self {
            strategy,
            radius,
            center,
        }
    }

    /// Sample `n` camera views.
    ///
    /// RNG usage differs per strategy:
    /// * `Random` draws from the RNG for every sample.
    /// * `Hemisphere` draws from the RNG *once*, to pick a random contiguous
    ///   window's starting offset into the `num_phi × num_theta` grid,
    ///   whenever the grid is larger than `n` (i.e. `num_phi * num_theta >
    ///   n`) — not "when `n` does not divide the count evenly", and the
    ///   result is a window offset, not a phase shift.
    /// * `Spiral` and `Turntable` are fully deterministic and never touch
    ///   the RNG at all, regardless of `n`.
    pub fn sample(&self, n: usize, rng: &mut impl Rng) -> Vec<CameraView> {
        match &self.strategy {
            SamplingStrategy::Random => self.sample_random(n, rng),
            SamplingStrategy::Spiral { rounds } => self.sample_spiral(n, *rounds),
            SamplingStrategy::Hemisphere { num_phi, num_theta } => {
                self.sample_hemisphere(n, *num_phi, *num_theta, rng)
            }
            SamplingStrategy::Turntable { elevation_deg } => {
                self.sample_turntable(n, *elevation_deg)
            }
        }
    }

    /// Sample a **single** view corresponding to training iteration `iter` out
    /// of `total` iterations.
    ///
    /// For deterministic strategies the position smoothly traverses the
    /// distribution as `iter` increases.  For `Random`, a fresh random view is
    /// drawn each call.
    pub fn sample_iter(&self, iter: u32, total: u32, rng: &mut impl Rng) -> CameraView {
        let total = total.max(1);
        let t = (iter % total) as f32 / total as f32; // ∈ [0, 1)
        match &self.strategy {
            SamplingStrategy::Random => {
                self.sample_random(1, rng)
                    .into_iter()
                    .next()
                    // Safety: sample_random always returns exactly `n` items.
                    .unwrap_or_else(|| self.make_view(0.0, 0.0))
            }
            SamplingStrategy::Spiral { rounds } => {
                // Same formula as `sample_spiral`: azimuth = TAU * rounds *
                // t, z linear in t. Keeping these in sync means a given `t`
                // (hence a given `iter`/`total` phase) always lands on the
                // same point of the sphere as the corresponding position in
                // a batched `sample()` call.
                let theta = TAU * rounds * t;
                let phi = (1.0 - 2.0 * t).acos();
                self.make_view(theta, phi)
            }
            SamplingStrategy::Hemisphere { num_phi, num_theta } => {
                let np = (*num_phi).max(1);
                let nt = (*num_theta).max(1);
                // Widen to u64 before multiplying: `np * nt` in u32 can
                // overflow for large caller-supplied grids.
                let total_grid = (np as u64 * nt as u64).max(1) as f64;
                let idx = (t as f64 * total_grid) as u64;
                let pi = (idx / nt as u64) as u32;
                let ti = (idx % nt as u64) as u32;
                // Azimuth (phi) is periodic (phi=0 and phi=TAU are the same
                // direction), so the grid spacing divisor is `num_phi`, not
                // `num_phi - 1` — dividing by `num_phi - 1` would duplicate
                // the phi=0 column. Polar angle (theta) is NOT periodic, so
                // `num_theta - 1` is correct there (endpoints 0 and π/2 are
                // genuinely distinct poles).
                let phi = (pi as f32 / np.max(1) as f32) * TAU;
                let theta = (ti as f32 / (nt - 1).max(1) as f32) * FRAC_PI_2;
                self.make_view(phi, theta)
            }
            SamplingStrategy::Turntable { elevation_deg } => {
                let phi = t * TAU;
                let theta = FRAC_PI_2 - elevation_deg.to_radians();
                self.make_view(phi, theta)
            }
        }
    }

    // ----- per-strategy sampling helpers ------------------------------------

    fn sample_random(&self, n: usize, rng: &mut impl Rng) -> Vec<CameraView> {
        // Marsaglia's uniform sphere sampling (no rejection).
        (0..n)
            .map(|_| {
                let theta = rng.random::<f32>() * TAU;
                // cos(phi) uniform on [-1, 1] ⟹ phi uniform on sphere.
                let cos_phi = rng.random::<f32>() * 2.0 - 1.0;
                let phi = cos_phi.acos();
                self.make_view(theta, phi)
            })
            .collect()
    }

    fn sample_spiral(&self, n: usize, rounds: f32) -> Vec<CameraView> {
        // Azimuth sweeps `rounds` revolutions over the traversal parameter
        // `t`, matching `sample_iter`'s Spiral branch exactly (`azimuth =
        // TAU * rounds * t`) so both entry points agree on what a given
        // `rounds` means and on where a given `t` lands on the sphere.
        //
        // Previously this used `TAU * rounds * (i / golden)` — a genuine
        // golden-angle Fibonacci lattice, but one where the actual number of
        // revolutions scaled with the sample count `n` (via `i` ranging up
        // to `n`), not with `rounds` as the field's own doc comment
        // promised, and where `sample_iter` computed a different, mutually
        // inconsistent azimuth for the "same" strategy. Sampling `t` at
        // `(i + 0.5) / n` (a half-step offset) rather than `i / n` avoids
        // placing the very first point exactly on the pole (t=0 ⟹ phi=0).
        //
        // z-parameter is still linear in `t`, giving equal-area coverage
        // per unit height (uniform density in the z-slab sense), which is
        // what the struct doc's "low discrepancy" claim relies on.
        (0..n)
            .map(|i| {
                let t = (i as f32 + 0.5) / n.max(1) as f32;
                let theta = TAU * rounds * t;
                let phi = (1.0 - 2.0 * t).acos(); // linear z → uniform area
                self.make_view(theta, phi)
            })
            .collect()
    }

    fn sample_hemisphere(
        &self,
        n: usize,
        num_phi: u32,
        num_theta: u32,
        rng: &mut impl Rng,
    ) -> Vec<CameraView> {
        let num_phi = num_phi.max(1);
        let num_theta = num_theta.max(1);
        // Widen to usize before multiplying: `num_phi * num_theta` performed
        // in u32 overflows for large caller-supplied grids (e.g. 100_000 x
        // 100_000), which would previously wrap and drive an incorrect
        // slice/modulo below.
        let total = num_phi as usize * num_theta as usize;

        if n == 0 || total == 0 {
            return Vec::new();
        }

        // Convert a flat grid index directly to a `CameraView`, without
        // materializing the full `num_phi * num_theta` grid — the previous
        // implementation always built the entire grid up front (e.g. 65 536
        // `CameraView`s for a 256×256 grid) even when only a handful of
        // samples were requested.
        let make_cell = |idx: usize| -> CameraView {
            let pi = (idx / num_theta as usize) as u32;
            let ti = (idx % num_theta as usize) as u32;
            // Azimuth (phi) is periodic (phi=0 and phi=TAU are the same
            // direction), so the grid spacing divisor is `num_phi`, not
            // `num_phi - 1` — using `num_phi - 1` duplicates the phi=0
            // column. Polar angle (theta) is NOT periodic (0 and π/2 are
            // genuinely distinct poles), so `num_theta - 1` is correct.
            let phi = (pi as f32 / num_phi as f32) * TAU;
            let theta = (ti as f32 / (num_theta - 1).max(1) as f32) * FRAC_PI_2;
            self.make_view(phi, theta)
        };

        if n <= total {
            // Draw a random contiguous window start, then generate only the
            // `n` requested cells.
            let start = if total > n {
                rng.random_range(0..=(total - n))
            } else {
                0
            };
            (start..start + n).map(make_cell).collect()
        } else {
            // Repeat the grid as needed.
            (0..n).map(|i| make_cell(i % total)).collect()
        }
    }

    fn sample_turntable(&self, n: usize, elevation_deg: f32) -> Vec<CameraView> {
        // Fixed elevation; uniformly spaced azimuth.
        let theta = FRAC_PI_2 - elevation_deg.to_radians();
        (0..n)
            .map(|i| {
                let phi = (i as f32 / n.max(1) as f32) * TAU;
                self.make_view(phi, theta)
            })
            .collect()
    }

    // ----- core geometry ----------------------------------------------------

    /// Convert spherical coordinates (azimuth `phi`, polar `theta`) to a
    /// [`CameraView`] with the eye on the sphere of radius `self.radius`.
    ///
    /// Convention:
    /// * `theta` = 0 → north pole (+Y axis).
    /// * `phi`   = 0 → +X axis in the XZ plane.
    fn make_view(&self, phi: f32, theta: f32) -> CameraView {
        let sin_t = theta.sin();
        let cos_t = theta.cos();
        let eye = [
            self.center[0] + self.radius * sin_t * phi.cos(),
            self.center[1] + self.radius * cos_t,
            self.center[2] + self.radius * sin_t * phi.sin(),
        ];
        // World up: prefer +Y; fall back to +Z when eye is near the poles.
        let up = if sin_t.abs() < 1e-6 {
            [0.0, 0.0, if cos_t >= 0.0 { 1.0 } else { -1.0 }]
        } else {
            [0.0, 1.0, 0.0]
        };
        CameraView {
            eye,
            center: self.center,
            up,
        }
    }
}

// ---------------------------------------------------------------------------
// Internal tiny Vec3 helper (avoids external dependencies)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct Vec3 {
    x: f32,
    y: f32,
    z: f32,
}

impl Vec3 {
    fn from(a: [f32; 3]) -> Self {
        Self {
            x: a[0],
            y: a[1],
            z: a[2],
        }
    }

    fn dot(self, other: Self) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    fn cross(self, other: Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    fn length_sq(self) -> f32 {
        self.dot(self)
    }

    /// Normalise, or return the +Z unit vector if near-zero.
    fn normalize_or_z(self) -> Self {
        let len_sq = self.length_sq();
        if len_sq < 1e-12 {
            Self {
                x: 0.0,
                y: 0.0,
                z: 1.0,
            }
        } else {
            let inv = len_sq.sqrt().recip();
            Self {
                x: self.x * inv,
                y: self.y * inv,
                z: self.z * inv,
            }
        }
    }
}

impl std::ops::Sub for Vec3 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn seeded_rng() -> rand::rngs::StdRng {
        rand::rngs::StdRng::seed_from_u64(42)
    }

    // ----- helper to measure distance from center ---------------------------

    fn dist(view: &CameraView) -> f32 {
        let dx = view.eye[0] - view.center[0];
        let dy = view.eye[1] - view.center[1];
        let dz = view.eye[2] - view.center[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    // ----- radius correctness -----------------------------------------------

    #[test]
    fn random_views_correct_radius() {
        let mut rng = seeded_rng();
        let sampler = CameraSampler::new(SamplingStrategy::Random, 3.0);
        for view in sampler.sample(50, &mut rng) {
            let d = dist(&view);
            assert!(
                (d - 3.0).abs() < 1e-5,
                "random view distance {d} != radius 3.0"
            );
        }
    }

    #[test]
    fn spiral_views_correct_radius() {
        let sampler = CameraSampler::new(SamplingStrategy::Spiral { rounds: 3.0 }, 5.0);
        let mut rng = seeded_rng();
        for view in sampler.sample(40, &mut rng) {
            let d = dist(&view);
            assert!(
                (d - 5.0).abs() < 1e-5,
                "spiral view distance {d} != radius 5.0"
            );
        }
    }

    #[test]
    fn hemisphere_views_correct_radius_and_upper_half() {
        let mut rng = seeded_rng();
        let sampler = CameraSampler::new(
            SamplingStrategy::Hemisphere {
                num_phi: 8,
                num_theta: 4,
            },
            2.0,
        );
        for view in sampler.sample(20, &mut rng) {
            let d = dist(&view);
            assert!(
                (d - 2.0).abs() < 1e-5,
                "hemisphere view distance {d} != radius 2.0"
            );
            // Upper hemisphere: eye.y >= center.y.
            assert!(
                view.eye[1] >= view.center[1] - 1e-5,
                "hemisphere view not in upper half: eye.y={} center.y={}",
                view.eye[1],
                view.center[1]
            );
        }
    }

    #[test]
    fn turntable_views_correct_radius_and_elevation() {
        let elevation = 30.0_f32;
        let radius = 4.0_f32;
        let mut rng = seeded_rng();
        let sampler = CameraSampler::new(
            SamplingStrategy::Turntable {
                elevation_deg: elevation,
            },
            radius,
        );
        for view in sampler.sample(36, &mut rng) {
            let d = dist(&view);
            assert!(
                (d - radius).abs() < 1e-5,
                "turntable view distance {d} != radius {radius}"
            );
            // Eye Y should equal radius * sin(elevation_deg).
            let expected_y = radius * elevation.to_radians().sin();
            assert!(
                (view.eye[1] - expected_y).abs() < 1e-4,
                "turntable eye.y={} expected {expected_y}",
                view.eye[1]
            );
        }
    }

    // ----- sample_iter consistency ------------------------------------------

    #[test]
    fn sample_iter_spiral_covers_sphere() {
        let sampler = CameraSampler::new(SamplingStrategy::Spiral { rounds: 2.0 }, 1.0);
        let mut rng = seeded_rng();
        // Collect Y coordinates over a full revolution.
        let ys: Vec<f32> = (0..100)
            .map(|i| sampler.sample_iter(i, 100, &mut rng).eye[1])
            .collect();
        // Should span roughly [-1, +1].
        let min_y = ys.iter().cloned().fold(f32::MAX, f32::min);
        let max_y = ys.iter().cloned().fold(f32::MIN, f32::max);
        assert!(min_y < -0.5, "spiral min Y={min_y} too high");
        assert!(max_y > 0.5, "spiral max Y={max_y} too low");
    }

    #[test]
    fn sample_iter_random_correct_radius() {
        let sampler = CameraSampler::new(SamplingStrategy::Random, 7.0);
        let mut rng = seeded_rng();
        let view = sampler.sample_iter(0, 100, &mut rng);
        let d = dist(&view);
        assert!(
            (d - 7.0).abs() < 1e-5,
            "random iter view distance {d} != 7.0"
        );
    }

    // ----- view_matrix orthogonality ----------------------------------------

    #[test]
    fn view_matrix_columns_orthonormal() {
        let mut rng = seeded_rng();
        let sampler = CameraSampler::new(SamplingStrategy::Random, 2.0);
        for view in sampler.sample(10, &mut rng) {
            let m = view.view_matrix();
            // First three rows of each of the first three columns give r, u, -f.
            let col = |c: usize| [m[c][0], m[c][1], m[c][2]];
            let dot3 = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
            let r = col(0);
            let u = col(1);
            let nf = col(2);
            // Each column should be unit length.
            assert!(
                (dot3(r, r) - 1.0).abs() < 1e-5,
                "r column not unit: {:?}",
                r
            );
            assert!(
                (dot3(u, u) - 1.0).abs() < 1e-5,
                "u column not unit: {:?}",
                u
            );
            assert!(
                (dot3(nf, nf) - 1.0).abs() < 1e-5,
                "-f column not unit: {:?}",
                nf
            );
            // Columns must be mutually orthogonal.
            assert!(dot3(r, u).abs() < 1e-5, "r·u != 0");
            assert!(dot3(r, nf).abs() < 1e-5, "r·(-f) != 0");
            assert!(dot3(u, nf).abs() < 1e-5, "u·(-f) != 0");
        }
    }

    // ----- with_center offset -----------------------------------------------

    #[test]
    fn with_center_shifts_views() {
        let center = [1.0, 2.0, 3.0];
        let sampler = CameraSampler::with_center(SamplingStrategy::Random, 1.0, center);
        let mut rng = seeded_rng();
        for view in sampler.sample(10, &mut rng) {
            assert_eq!(view.center, center, "center mismatch");
            let d = dist(&view);
            assert!((d - 1.0).abs() < 1e-5, "distance {d} != radius 1.0");
        }
    }

    // ----- edge cases -------------------------------------------------------

    #[test]
    fn sample_zero_returns_empty() {
        let sampler = CameraSampler::new(SamplingStrategy::Random, 1.0);
        let mut rng = seeded_rng();
        assert!(sampler.sample(0, &mut rng).is_empty());
    }

    #[test]
    fn sample_one_returns_single() {
        let sampler = CameraSampler::new(SamplingStrategy::Turntable { elevation_deg: 0.0 }, 1.0);
        let mut rng = seeded_rng();
        assert_eq!(sampler.sample(1, &mut rng).len(), 1);
    }

    // ----- Regression tests --------------------------------------------------

    // `sample_spiral` and `sample_iter`'s Spiral branch previously computed
    // different azimuth formulas for the same `SamplingStrategy::Spiral`
    // (`TAU * rounds * (i / golden)` vs `TAU * rounds * t`), meaning the two
    // entry points produced mutually inconsistent camera distributions, and
    // the `rounds` field did not actually control the number of revolutions
    // in `sample_spiral` (it scaled with the sample count `n` instead).
    // Both now compute `azimuth = TAU * rounds * t` with `t` derived the
    // same way (linearly over the traversal). Verify a `sample_iter` call
    // at the exact `t` that `sample_spiral` would use for index `i` (namely
    // `t = (i + 0.5) / n`) reproduces that same batch element exactly.
    #[test]
    fn spiral_sample_and_sample_iter_use_the_same_azimuth_formula() {
        let sampler = CameraSampler::new(SamplingStrategy::Spiral { rounds: 2.5 }, 3.0);
        let mut rng = seeded_rng();
        let n: u32 = 20;
        let batch = sampler.sample(n as usize, &mut rng);
        assert_eq!(batch.len(), n as usize);

        for (i, batch_view) in batch.iter().enumerate() {
            // sample_spiral's t for index i is (i + 0.5) / n. Reproduce
            // that phase through sample_iter by scaling both numerator and
            // denominator by 2 so it lands on an exact integer `iter`:
            // t = (2*i + 1) / (2*n).
            let iter_val = 2 * i as u32 + 1;
            let total = 2 * n;
            let mut rng2 = seeded_rng(); // sample_iter's Spiral branch ignores rng
            let iter_view = sampler.sample_iter(iter_val, total, &mut rng2);
            assert!(
                (iter_view.eye[0] - batch_view.eye[0]).abs() < 1e-3
                    && (iter_view.eye[1] - batch_view.eye[1]).abs() < 1e-3
                    && (iter_view.eye[2] - batch_view.eye[2]).abs() < 1e-3,
                "index {i}: sample_iter({iter_val}, {total}) = {:?} != sample()[{i}] = {:?}",
                iter_view.eye,
                batch_view.eye
            );
        }
    }

    // `rounds` must control the actual number of azimuthal revolutions:
    // doubling `rounds` must double the total unwrapped azimuth swept from
    // t=0 to t=1, independent of how many samples `n` are requested.
    #[test]
    fn spiral_rounds_controls_revolutions_independent_of_sample_count() {
        let radius = 1.0;
        for &rounds in &[1.0_f32, 3.0] {
            let sampler = CameraSampler::new(SamplingStrategy::Spiral { rounds }, radius);
            // At t=0.25 (well away from any pole so azimuth is well
            // defined), azimuth should equal TAU * rounds * 0.25 (mod TAU).
            let mut rng = seeded_rng();
            let view = sampler.sample_iter(1, 4, &mut rng); // t = 0.25
            let expected_azimuth = (TAU * rounds * 0.25).rem_euclid(TAU);
            // make_view: eye.x = r*sin(theta)*cos(phi), eye.z = r*sin(theta)*sin(phi)
            let actual_azimuth = view.eye[2].atan2(view.eye[0]).rem_euclid(TAU);
            let diff = (actual_azimuth - expected_azimuth).rem_euclid(TAU);
            let diff = diff.min(TAU - diff);
            assert!(
                diff < 1e-3,
                "rounds={rounds}: expected azimuth {expected_azimuth}, got {actual_azimuth}"
            );
        }
    }

    // Hemisphere azimuth must not duplicate the phi=0 column: with
    // `num_phi = 8`, the 8 distinct phi columns should be evenly spaced by
    // 2π/8, and phi index 0 and phi index num_phi-1 must NOT coincide
    // (previously dividing by `num_phi - 1` treated azimuth as
    // non-periodic, duplicating the phi=0 column and producing unequal
    // spacing of 2π/7).
    #[test]
    fn hemisphere_azimuth_grid_has_no_duplicate_column() {
        let sampler = CameraSampler::new(
            SamplingStrategy::Hemisphere {
                num_phi: 8,
                num_theta: 1,
            },
            1.0,
        );
        let mut rng = seeded_rng();
        // theta fixed (num_theta=1 -> theta=0 for every ti), so all 8
        // returned views (n == total) differ only in phi.
        let views = sampler.sample(8, &mut rng);
        assert_eq!(views.len(), 8);

        // Recover azimuth via atan2(z, x) (make_view: eye.x = r*sin(theta)*cos(phi),
        // eye.z = r*sin(theta)*sin(phi); theta=0 makes sin(theta)=0 so this
        // degenerates — use num_theta=2 instead so theta is nonzero for at
        // least one row). Rebuild with num_theta=2 for a meaningful azimuth
        // readout on the ti=1 (theta=pi/2) row.
        let sampler2 = CameraSampler::new(
            SamplingStrategy::Hemisphere {
                num_phi: 8,
                num_theta: 2,
            },
            1.0,
        );
        let mut rng2 = seeded_rng();
        let grid = sampler2.sample(16, &mut rng2); // full grid, no RNG draw needed
                                                   // Row ti=1 (theta=pi/2, equator): indices 1, 3, 5, ..., 15 (pi*num_theta + 1).
        let mut azimuths: Vec<f32> = (0..8)
            .map(|pi| {
                let view = &grid[pi * 2 + 1];
                view.eye[2]
                    .atan2(view.eye[0])
                    .rem_euclid(std::f32::consts::TAU)
            })
            .collect();
        azimuths.sort_by(|a, b| a.partial_cmp(b).unwrap());
        // 8 distinct azimuths, evenly spaced by 2*pi/8 = pi/4, no duplicate.
        for w in azimuths.windows(2) {
            let gap = w[1] - w[0];
            assert!(
                gap > 0.1,
                "adjacent azimuths too close (possible duplicate column): {:?}",
                azimuths
            );
        }
        let expected_spacing = std::f32::consts::TAU / 8.0;
        for w in azimuths.windows(2) {
            let gap = w[1] - w[0];
            assert!(
                (gap - expected_spacing).abs() < 0.05,
                "expected spacing ~{expected_spacing}, got {gap}"
            );
        }
    }

    // num_phi * num_theta must not overflow when multiplied: previously
    // computed as `(num_phi * num_theta) as usize` with the multiplication
    // performed in u32, which overflows (and panics in debug builds) for
    // large caller-supplied grids.
    #[test]
    fn hemisphere_large_grid_does_not_overflow() {
        let sampler = CameraSampler::new(
            SamplingStrategy::Hemisphere {
                num_phi: 100_000,
                num_theta: 100_000,
            },
            1.0,
        );
        let mut rng = seeded_rng();
        // Requesting a small `n` must not panic and must not materialize
        // the full 10^10-entry grid.
        let views = sampler.sample(4, &mut rng);
        assert_eq!(views.len(), 4);
        for view in &views {
            let d = dist(view);
            assert!((d - 1.0).abs() < 1e-4);
        }
    }

    // sample_iter's Hemisphere branch must also not overflow on a large grid.
    #[test]
    fn hemisphere_sample_iter_large_grid_does_not_overflow() {
        let sampler = CameraSampler::new(
            SamplingStrategy::Hemisphere {
                num_phi: 100_000,
                num_theta: 100_000,
            },
            1.0,
        );
        let mut rng = seeded_rng();
        let view = sampler.sample_iter(12345, 1_000_000, &mut rng);
        let d = dist(&view);
        assert!((d - 1.0).abs() < 1e-4);
    }
}
