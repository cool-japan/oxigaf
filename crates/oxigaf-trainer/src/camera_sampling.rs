//! Camera view sampling strategies for training.
//!
//! Provides several geometric distributions for drawing camera positions on
//! (or around) a sphere centred at a configurable look-at point:
//!
//! * [`SamplingStrategy::Random`]      — uniform spherical distribution.
//! * [`SamplingStrategy::Spiral`]      — Fibonacci-lattice spiral.
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

    /// Points arranged along a Fibonacci spiral on the sphere, providing
    /// near-uniform coverage with low discrepancy.
    ///
    /// `rounds` controls how many full azimuthal revolutions the spiral
    /// completes (typical range 1–5).
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
    /// For deterministic strategies (Spiral, Hemisphere, Turntable) the RNG
    /// is only used when `n` does not divide the pre-determined sample count
    /// evenly — a random starting phase is then applied.  For `Random`, the
    /// RNG is used for every sample.
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
                // Use a single Fibonacci-spiral point at parameter `t`.
                let theta = TAU * rounds * t;
                let phi = (1.0 - 2.0 * t).acos();
                self.make_view(theta, phi)
            }
            SamplingStrategy::Hemisphere { num_phi, num_theta } => {
                let np = (*num_phi).max(1);
                let nt = (*num_theta).max(1);
                let total_grid = (np * nt).max(1) as f32;
                let idx = (t * total_grid) as u32;
                let pi = idx / nt;
                let ti = idx % nt;
                let phi = (pi as f32 / (np - 1).max(1) as f32) * TAU;
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
        // Golden-ratio Fibonacci lattice on sphere.
        // Reference: Roberts (2018), "The Unreasonable Effectiveness of Quasirandom Sequences"
        let golden = (1.0 + 5.0_f32.sqrt()) / 2.0;
        (0..n)
            .map(|i| {
                let t = i as f32 / n.max(1) as f32;
                let theta = TAU * rounds * (i as f32 / golden);
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
        let total = (num_phi * num_theta) as usize;

        // Build the full grid, then pick `n` entries (with repetition if n > total).
        let grid: Vec<CameraView> = (0..num_phi)
            .flat_map(|pi| {
                let phi = (pi as f32 / (num_phi - 1).max(1) as f32) * TAU;
                (0..num_theta).map(move |ti| {
                    // theta ∈ [0, π/2] for upper hemisphere.
                    let theta = (ti as f32 / (num_theta - 1).max(1) as f32) * FRAC_PI_2;
                    (phi, theta)
                })
            })
            .map(|(phi, theta)| self.make_view(phi, theta))
            .collect();

        if n <= total {
            // Draw a random contiguous slice.
            let start = if total > n {
                rng.random_range(0..=(total - n))
            } else {
                0
            };
            grid[start..start + n].to_vec()
        } else {
            // Repeat the grid as needed.
            (0..n).map(|i| grid[i % total].clone()).collect()
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
}
