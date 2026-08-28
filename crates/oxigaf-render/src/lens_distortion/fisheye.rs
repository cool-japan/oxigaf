//! Fisheye projection models: the forward map and its Newton inverse.
//!
//! Split out of `lens_distortion.rs` to keep that file under the 2000-line
//! limit.

use super::LensDistortionError;
use std::f32::consts::FRAC_PI_2;

/// Internal fisheye projection types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum FisheyeProjection {
    Equidistant,
    Equisolid,
    Stereographic,
}

/// Angular margin kept clear of the 90° fold when inverting a fisheye model,
/// in radians.
///
/// The pinhole radius is `tan(theta)`, which diverges at `theta = π/2`; at
/// `π/2 - 1e-3` it is already ~1000 focal lengths off-axis, so the margin
/// costs no representable geometry while keeping every returned radius
/// finite (and its `scale = r_u / r_d` far from overflow).
const FISHEYE_THETA_MARGIN: f32 = 1e-3;

/// Largest undistorted ray angle that a pinhole image plane can represent,
/// as used by [`undistort_fisheye`].
///
/// A fisheye lens maps rays at and beyond 90° off-axis into the image (that
/// is the whole point of the projection), but the *undistorted* pinhole model
/// those rays are being mapped back to cannot hold them: `r_u = tan(theta)`
/// diverges at `theta = π/2` and comes back **negative** past it, so an
/// unguarded inverse silently returns a mirrored point instead of reporting
/// that the ray has no pinhole pre-image.
const FISHEYE_MAX_THETA: f32 = FRAC_PI_2 - FISHEYE_THETA_MARGIN;

/// Shared implementation for all fisheye distortion variants.
pub(super) fn fisheye_distort(
    xu: f32,
    yu: f32,
    k1: f32,
    k2: f32,
    proj: FisheyeProjection,
) -> (f32, f32) {
    let r = (xu * xu + yu * yu).sqrt();
    if r < 1e-9 {
        return (xu, yu);
    }
    // theta = angle of incoming ray from optical axis (in the undistorted pinhole model, r = tan(theta))
    let theta = r.atan();
    let theta2 = theta * theta;
    let theta4 = theta2 * theta2;

    // Fisheye distortion polynomial on theta
    let theta_d = theta * (1.0 + k1 * theta2 + k2 * theta4);

    // Map distorted theta to radius in the image plane
    let r_d = match proj {
        FisheyeProjection::Equidistant => theta_d,
        FisheyeProjection::Equisolid => 2.0 * (theta_d / 2.0).sin(),
        FisheyeProjection::Stereographic => 2.0 * (theta_d / 2.0).tan(),
    };

    let scale = r_d / r;
    (xu * scale, yu * scale)
}

/// Largest distorted radius a fisheye model can produce that still has an
/// undistorted (pinhole) pre-image: the image of [`FISHEYE_MAX_THETA`] under
/// the same forward chain [`fisheye_distort`] applies.
///
/// Reported as `r_d_max` in
/// [`LensDistortionError::OutsideDistortionDomain`]. It is a diagnostic
/// boundary value, computed at the domain edge itself; with strongly negative
/// `k1` the polynomial `theta_d(theta)` can fold below `π/2`, in which case
/// the *distorted* radius at the edge is not the largest one the model emits
/// and this is a conservative report rather than an exact supremum. A
/// projection whose closed form blows up there (stereographic with
/// `theta_d ≥ π`) is reported as infinite.
fn fisheye_domain_max_radius(k1: f32, k2: f32, proj: FisheyeProjection) -> f32 {
    let theta = FISHEYE_MAX_THETA;
    let theta2 = theta * theta;
    let theta_d = theta * (1.0 + k1 * theta2 + k2 * theta2 * theta2);
    let r_d_max = match proj {
        FisheyeProjection::Equidistant => theta_d,
        FisheyeProjection::Equisolid => 2.0 * (theta_d / 2.0).sin(),
        FisheyeProjection::Stereographic => 2.0 * (theta_d / 2.0).tan(),
    };
    if r_d_max.is_finite() && r_d_max > 0.0 {
        r_d_max
    } else {
        f32::INFINITY
    }
}

/// Map a solved undistorted ray angle back to normalised pinhole coordinates,
/// refusing the angles a pinhole plane cannot represent.
///
/// Every exit from [`undistort_fisheye`] that produces coordinates goes
/// through here, so the domain guard cannot be skipped at one of them.
///
/// # Errors
///
/// [`LensDistortionError::OutsideDistortionDomain`] when `theta` is at or
/// past [`FISHEYE_MAX_THETA`] (in either direction), i.e. when the distorted
/// point corresponds to a ray 90° or more off-axis and therefore has no
/// pinhole pre-image at all.
fn pinhole_from_theta(
    theta: f32,
    xd: f32,
    yd: f32,
    r_d: f32,
    k1: f32,
    k2: f32,
    proj: FisheyeProjection,
) -> Result<(f32, f32), LensDistortionError> {
    if !theta.is_finite() || theta.abs() >= FISHEYE_MAX_THETA {
        return Err(LensDistortionError::OutsideDistortionDomain {
            r_d,
            r_d_max: fisheye_domain_max_radius(k1, k2, proj),
        });
    }
    let r_u = theta.tan();
    let scale = r_u / r_d;
    Ok((xd * scale, yd * scale))
}

/// Invert a fisheye distortion model by solving for the undistorted ray
/// angle `theta` via 1-D Newton iteration on `theta_d = theta * (1 +
/// k1*theta² + k2*theta⁴)`, then mapping back through the (closed-form,
/// invertible) projection to a radius and finally to `(xu, yu)`.
///
/// # Domain
///
/// The result is a point in the *undistorted pinhole* plane, whose radius is
/// `tan(theta)`. That map only exists for rays less than 90° off-axis, so a
/// distorted radius whose solved angle reaches [`FISHEYE_MAX_THETA`] has no
/// pre-image and is reported as such — the same contract the radial family
/// uses for its fold (see
/// [`LensDistortionError::OutsideDistortionDomain`]), which whole-image
/// passes render as the module's unmapped-pixel colour rather than as a
/// hard failure. Fisheye lenses routinely image such rays (a 180° circular
/// fisheye images exactly up to 90°), so this is ordinary geometry, not a
/// pathological input. The largest representable undistorted radius is
/// therefore `tan(FISHEYE_MAX_THETA)` ≈ 1000 focal lengths.
///
/// # Errors
///
/// - [`LensDistortionError::OutsideDistortionDomain`] when the solved ray
///   angle is at or past 90° off-axis (see above).
/// - [`LensDistortionError::ConvergenceError`] when the Newton iteration
///   stalls (vanishing or non-finite derivative), leaves the reals, or does
///   not reach `tol` within `max_iters`.
pub(super) fn undistort_fisheye(
    xd: f32,
    yd: f32,
    k1: f32,
    k2: f32,
    proj: FisheyeProjection,
    max_iters: usize,
    tol: f32,
) -> Result<(f32, f32), LensDistortionError> {
    let r_d = (xd * xd + yd * yd).sqrt();
    if r_d < 1e-9 {
        return Ok((xd, yd));
    }

    // Invert the closed-form projection r_d = proj(theta_d) for theta_d,
    // the *distorted* ray angle (the inverse of the forward mapping in
    // `fisheye_distort`).
    let theta_d = match proj {
        FisheyeProjection::Equidistant => r_d,
        FisheyeProjection::Equisolid => 2.0 * (r_d * 0.5).clamp(-1.0, 1.0).asin(),
        FisheyeProjection::Stereographic => 2.0 * (r_d * 0.5).atan(),
    };

    // 1-D Newton solve for the undistorted angle theta on
    // f(theta) = theta*(1 + k1*theta² + k2*theta⁴) - theta_d = 0.
    let mut theta = theta_d;
    for iter in 0..max_iters {
        let t2 = theta * theta;
        let t4 = t2 * t2;
        let f = theta * (1.0 + k1 * t2 + k2 * t4) - theta_d;
        let fp = 1.0 + 3.0 * k1 * t2 + 5.0 * k2 * t4;
        if !fp.is_finite() || fp.abs() < 1e-8 {
            return Err(LensDistortionError::ConvergenceError { iterations: iter });
        }
        let step = f / fp;
        theta -= step;
        if !theta.is_finite() {
            return Err(LensDistortionError::ConvergenceError {
                iterations: iter + 1,
            });
        }
        if step.abs() < tol {
            return pinhole_from_theta(theta, xd, yd, r_d, k1, k2, proj);
        }
        if iter + 1 == max_iters {
            return Err(LensDistortionError::ConvergenceError {
                iterations: max_iters,
            });
        }
    }

    // max_iters == 0 edge case: accept theta_d itself if it is already a root.
    let t2 = theta * theta;
    let t4 = t2 * t2;
    let f = theta * (1.0 + k1 * t2 + k2 * t4) - theta_d;
    if f.abs() < tol {
        return pinhole_from_theta(theta, xd, yd, r_d, k1, k2, proj);
    }
    Err(LensDistortionError::ConvergenceError { iterations: 0 })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Solve for the undistorted point with the settings the whole-image
    /// passes use.
    fn undistort(
        xd: f32,
        yd: f32,
        k1: f32,
        k2: f32,
        proj: FisheyeProjection,
    ) -> Result<(f32, f32), LensDistortionError> {
        undistort_fisheye(xd, yd, k1, k2, proj, 40, 1e-6)
    }

    #[test]
    fn test_undistort_fisheye_rejects_rays_past_the_90_degree_fold() {
        // Regression: the inverse ended in a bare `theta.tan()`. For an
        // equidistant model with k1 = k2 = 0 the Newton solve returns
        // `theta = r_d` immediately, so a distorted radius of 2.0 rad (114.6°
        // off-axis — well inside what a real fisheye images) used to be
        // mapped through `tan(2.0)`, which is NEGATIVE: the point came back
        // mirrored through the optical axis, silently, with no diagnostic.
        assert!(
            (2.0_f32).tan() < 0.0,
            "the premise of this regression: tan is negative past π/2"
        );

        match undistort(2.0, 0.0, 0.0, 0.0, FisheyeProjection::Equidistant) {
            Err(LensDistortionError::OutsideDistortionDomain { r_d, r_d_max }) => {
                assert!((r_d - 2.0).abs() < 1e-6, "r_d: {r_d}");
                assert!(
                    (r_d_max - FISHEYE_MAX_THETA).abs() < 1e-5,
                    "an equidistant model's boundary radius is the boundary \
                     angle itself, got {r_d_max}"
                );
            }
            other => panic!("expected OutsideDistortionDomain, got {other:?}"),
        }

        // The same fold, reached through the two other projections.
        for (proj, r_d) in [
            (FisheyeProjection::Equisolid, 2.0_f32),
            (FisheyeProjection::Stereographic, 10.0_f32),
        ] {
            match undistort(r_d, 0.0, 0.0, 0.0, proj) {
                Err(LensDistortionError::OutsideDistortionDomain { r_d_max, .. }) => {
                    assert!(
                        r_d > r_d_max,
                        "{proj:?}: r_d {r_d} should exceed the reported \
                         boundary {r_d_max}"
                    );
                }
                other => panic!("{proj:?}: expected OutsideDistortionDomain, got {other:?}"),
            }
        }
    }

    #[test]
    fn test_undistort_fisheye_boundary_is_exactly_the_fold() {
        // Equidistant with k1 = k2 = 0 makes theta == r_d, so the fold sits
        // at exactly FISHEYE_MAX_THETA and both sides of it can be probed.
        let just_inside = FISHEYE_MAX_THETA - 1e-3;
        let (xu, yu) = undistort(just_inside, 0.0, 0.0, 0.0, FisheyeProjection::Equidistant)
            .expect("just below the fold still has a pre-image");
        assert!(
            xu.is_finite() && xu > 0.0,
            "a radius just inside the domain must stay finite and unmirrored, got {xu}"
        );
        assert!(
            (xu - just_inside.tan()).abs() < 1e-2 * xu.abs(),
            "xu should be tan(theta) = {}, got {xu}",
            just_inside.tan()
        );
        assert!(yu.abs() < 1e-6, "yu: {yu}");

        // Exactly at the fold, and one step past it, there is no pre-image.
        for r_d in [FISHEYE_MAX_THETA, FISHEYE_MAX_THETA + 1e-3, FRAC_PI_2] {
            assert!(
                matches!(
                    undistort(r_d, 0.0, 0.0, 0.0, FisheyeProjection::Equidistant),
                    Err(LensDistortionError::OutsideDistortionDomain { .. })
                ),
                "r_d = {r_d} is at or past the fold"
            );
        }
    }

    #[test]
    fn test_undistort_fisheye_still_round_trips_wide_but_valid_rays() {
        // The guard must not narrow the working domain: 3.0 focal lengths
        // off-axis is 71.6°, a legitimately wide ray that must survive a
        // forward/inverse round trip unchanged.
        for proj in [
            FisheyeProjection::Equidistant,
            FisheyeProjection::Equisolid,
            FisheyeProjection::Stereographic,
        ] {
            let (xu0, yu0) = (3.0_f32, 0.5_f32);
            let (xd, yd) = fisheye_distort(xu0, yu0, -0.05, 0.005, proj);
            let (xu1, yu1) = undistort(xd, yd, -0.05, 0.005, proj)
                .unwrap_or_else(|e| panic!("{proj:?}: a 71.6° ray is inside the domain: {e}"));
            assert!((xu1 - xu0).abs() < 1e-3, "{proj:?}: xu {xu1} vs {xu0}");
            assert!((yu1 - yu0).abs() < 1e-3, "{proj:?}: yu {yu1} vs {yu0}");
        }
    }

    #[test]
    fn test_fisheye_domain_max_radius_bounds_the_forward_map() {
        // The reported boundary must actually bound what the forward map
        // emits for representable rays, otherwise the error message points
        // the caller at the wrong number.
        for proj in [
            FisheyeProjection::Equidistant,
            FisheyeProjection::Equisolid,
            FisheyeProjection::Stereographic,
        ] {
            let r_d_max = fisheye_domain_max_radius(0.0, 0.0, proj);
            assert!(r_d_max.is_finite() && r_d_max > 0.0, "{proj:?}: {r_d_max}");
            let near_edge = (0.99 * FISHEYE_MAX_THETA).tan();
            let (xd, _) = fisheye_distort(near_edge, 0.0, 0.0, 0.0, proj);
            assert!(
                xd < r_d_max,
                "{proj:?}: forward radius {xd} at 0.99 of the boundary angle \
                 must stay under the reported maximum {r_d_max}"
            );
        }
    }
}
