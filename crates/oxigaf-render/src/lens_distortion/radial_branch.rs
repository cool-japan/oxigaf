//! Exact inversion of the radial part of a lens-distortion model.
//!
//! Split out of `lens_distortion.rs` to keep that file under the 2000-line
//! limit; it is a self-contained numeric kernel with its own tests.

use super::LensDistortionError;

/// Upper bound on the undistorted normalised radius considered by
/// [`RadialBranch`] when the forward map has no stationary point.
///
/// A normalised radius of 8 is already 8 focal lengths off-axis (a ~83°
/// half-angle), far outside any real image, so treating the map as monotonic
/// below it costs nothing while keeping the bracket finite.
pub(super) const RADIAL_SEARCH_LIMIT: f32 = 8.0;

/// The monotonic branch of a radial forward map
/// `r_d = r_u * (1 + k1*r_u² + k2*r_u⁴ + k3*r_u⁶)`.
///
/// Built once per model (the stationary-point search is not per-pixel work)
/// and then used to invert distorted radii exactly, by safeguarded
/// Newton iteration inside a bracket that is guaranteed to contain the root.
/// Unlike a bare fixed-point iteration this cannot silently diverge, and it
/// can tell "no solution exists" (see
/// [`LensDistortionError::OutsideDistortionDomain`]) apart from "iteration
/// ran out of steps".
#[derive(Debug, Clone, Copy)]
pub(super) struct RadialBranch {
    pub(super) k1: f32,
    pub(super) k2: f32,
    pub(super) k3: f32,
    /// Undistorted radius at which the forward map stops increasing.
    pub(super) r_u_max: f32,
    /// Largest distorted radius the map can produce (`forward(r_u_max)`).
    pub(super) r_d_max: f32,
}

impl RadialBranch {
    pub(super) fn new(k1: f32, k2: f32, k3: f32) -> Self {
        let r_u_max = radial_stationary_point(k1, k2, k3).unwrap_or(RADIAL_SEARCH_LIMIT);
        let mut branch = Self {
            k1,
            k2,
            k3,
            r_u_max,
            r_d_max: 0.0,
        };
        branch.r_d_max = branch.forward(r_u_max);
        branch
    }

    /// Forward radial map.
    #[inline]
    pub(super) fn forward(&self, r: f32) -> f32 {
        let r2 = r * r;
        r * (1.0 + self.k1 * r2 + self.k2 * r2 * r2 + self.k3 * r2 * r2 * r2)
    }

    /// Derivative of the forward radial map.
    #[inline]
    pub(super) fn forward_derivative(&self, r: f32) -> f32 {
        let r2 = r * r;
        1.0 + 3.0 * self.k1 * r2 + 5.0 * self.k2 * r2 * r2 + 7.0 * self.k3 * r2 * r2 * r2
    }

    /// Solve `forward(r_u) == r_d` for `r_u` on the monotonic branch.
    ///
    /// # Errors
    ///
    /// [`LensDistortionError::OutsideDistortionDomain`] when `r_d` is larger
    /// than any radius the forward map can produce.
    pub(super) fn invert(&self, r_d: f32) -> Result<f32, LensDistortionError> {
        if !r_d.is_finite() {
            return Err(LensDistortionError::OutsideDistortionDomain {
                r_d,
                r_d_max: self.r_d_max,
            });
        }
        if r_d <= 0.0 {
            return Ok(0.0);
        }
        if r_d > self.r_d_max {
            return Err(LensDistortionError::OutsideDistortionDomain {
                r_d,
                r_d_max: self.r_d_max,
            });
        }

        // `forward` is continuous and strictly increasing on [0, r_u_max],
        // with forward(0) = 0 <= r_d <= forward(r_u_max), so the root is
        // bracketed from the start and stays bracketed below.
        let mut lo = 0.0_f32;
        let mut hi = self.r_u_max;
        // `r_d` itself is the exact answer for an undistorted model and a
        // good starting guess otherwise.
        let mut r = r_d.clamp(lo, hi);

        for _ in 0..RADIAL_INVERT_MAX_STEPS {
            let f = self.forward(r) - r_d;
            if f.abs() <= RADIAL_INVERT_TOL {
                return Ok(r);
            }
            if f > 0.0 {
                hi = r;
            } else {
                lo = r;
            }
            if hi - lo <= RADIAL_INVERT_TOL {
                return Ok(0.5 * (lo + hi));
            }
            let d = self.forward_derivative(r);
            let newton = if d.abs() > 1e-12 { r - f / d } else { f32::NAN };
            // Bisect whenever Newton leaves the bracket (or misbehaves).
            r = if newton.is_finite() && newton > lo && newton < hi {
                newton
            } else {
                0.5 * (lo + hi)
            };
        }

        Ok(r)
    }
}

/// Maximum safeguarded-Newton steps for [`RadialBranch::invert`].
///
/// Bisection alone halves the bracket every step, so 64 steps drive any
/// `f32` bracket to its last representable digit; Newton usually gets there
/// in a handful.
const RADIAL_INVERT_MAX_STEPS: usize = 64;

/// Absolute tolerance for [`RadialBranch::invert`], in normalised radius
/// units (~0.001 pixel at a 1000-pixel focal length).
const RADIAL_INVERT_TOL: f32 = 1e-7;

/// Smallest positive `r` where `1 + 3*k1*r² + 5*k2*r⁴ + 7*k3*r⁶` vanishes,
/// i.e. where the radial forward map stops increasing, or `None` when the map
/// is monotonic throughout [`RADIAL_SEARCH_LIMIT`].
///
/// Substituting `s = r²` turns it into a cubic in `s` whose smallest positive
/// root is what is wanted; the linear and quadratic cases (the common ones:
/// `SimpleRadial`, and Brown-Conrady with `k3 = 0`) are solved in closed
/// form, and the full cubic by a sign-change scan plus bisection.
fn radial_stationary_point(k1: f32, k2: f32, k3: f32) -> Option<f32> {
    let a = 7.0 * k3;
    let b = 5.0 * k2;
    let c = 3.0 * k1;
    let s_limit = RADIAL_SEARCH_LIMIT * RADIAL_SEARCH_LIMIT;
    let accept = |s: f32| -> Option<f32> {
        if s > 0.0 && s <= s_limit {
            Some(s.sqrt())
        } else {
            None
        }
    };

    if a == 0.0 && b == 0.0 {
        // 1 + c*s = 0
        if c < 0.0 {
            return accept(-1.0 / c);
        }
        return None;
    }

    if a == 0.0 {
        // b*s² + c*s + 1 = 0
        let disc = c * c - 4.0 * b;
        if disc < 0.0 {
            return None;
        }
        let sq = disc.sqrt();
        let s1 = (-c - sq) / (2.0 * b);
        let s2 = (-c + sq) / (2.0 * b);
        let mut best = f32::INFINITY;
        for s in [s1, s2] {
            if s > 0.0 && s < best {
                best = s;
            }
        }
        return if best.is_finite() { accept(best) } else { None };
    }

    // General cubic: h(0) = 1 > 0, so scan for the first sign change and
    // bisect inside it. The scan resolution is fine enough that a root pair
    // closer together than one step is a numerically irrelevant grazing
    // touch of the axis.
    let h = |s: f32| 1.0 + c * s + b * s * s + a * s * s * s;
    const SCAN_STEPS: usize = 512;
    let mut prev_s = 0.0_f32;
    for i in 1..=SCAN_STEPS {
        let s = s_limit * (i as f32) / (SCAN_STEPS as f32);
        if h(s) <= 0.0 {
            // Root in (prev_s, s]: bisect.
            let mut lo = prev_s;
            let mut hi = s;
            for _ in 0..40 {
                let mid = 0.5 * (lo + hi);
                if h(mid) > 0.0 {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            return accept(0.5 * (lo + hi));
        }
        prev_s = s;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_radial_branch_matches_analytic_fold() {
        // The stationary point of `r*(1 + k1*r²)` is at `r = 1/sqrt(-3*k1)`
        // for k1 < 0; check the solver's closed-form linear case, its
        // quadratic case (k2 != 0), and the monotonic case (k1 > 0).
        let b = RadialBranch::new(-0.1, 0.0, 0.0);
        assert!(
            (b.r_u_max - 1.825_741_9).abs() < 1e-4,
            "r_u_max: {}",
            b.r_u_max
        );
        assert!(
            (b.r_d_max - 1.217_161_2).abs() < 1e-4,
            "r_d_max: {}",
            b.r_d_max
        );

        // k1 = -0.2, k2 = 0.05: 5*k2*s² + 3*k1*s + 1 = 0.25s² - 0.6s + 1 has
        // a negative discriminant (0.36 - 1 < 0) -> no stationary point.
        let mono = RadialBranch::new(-0.2, 0.05, 0.0);
        assert!(
            (mono.r_u_max - RADIAL_SEARCH_LIMIT).abs() < 1e-6,
            "expected a monotonic branch, got r_u_max = {}",
            mono.r_u_max
        );

        // Pincushion (k1 > 0) never folds.
        let pincushion = RadialBranch::new(0.2, 0.0, 0.0);
        assert!((pincushion.r_u_max - RADIAL_SEARCH_LIMIT).abs() < 1e-6);

        // Round-trip right up to the fold.
        let near_fold = b.forward(b.r_u_max * 0.99);
        let inverted = b.invert(near_fold).expect("inside the domain");
        assert!(
            (b.forward(inverted) - near_fold).abs() < 1e-5,
            "forward(invert(r)) should reproduce r: {} vs {near_fold}",
            b.forward(inverted)
        );

        // A cubic (k3 != 0) case goes through the scan-and-bisect path:
        // k1 = -0.3, k2 = 0, k3 = 0 gives r = 1/sqrt(0.9) = 1.05409; adding
        // a small positive k3 pushes the fold outward but must keep one.
        let cubic = RadialBranch::new(-0.3, 0.0, 0.001);
        assert!(
            cubic.r_u_max > 1.0 && cubic.r_u_max < RADIAL_SEARCH_LIMIT,
            "expected a fold from the cubic branch, got {}",
            cubic.r_u_max
        );
        assert!(cubic.forward_derivative(cubic.r_u_max).abs() < 1e-3);
    }
}
