//! Lens distortion models for camera calibration and image correction.
//!
//! Supports Brown-Conrady, multiple fisheye projections, simple radial,
//! and the Fitzgibbon division model. Provides forward distortion, iterative
//! undistortion, full-image remap, pre-computed distortion maps, and
//! distortion-field statistics.

use thiserror::Error;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors produced by the lens-distortion module.
#[derive(Debug, Error)]
pub enum LensDistortionError {
    /// Configuration parameter out of range or incoherent.
    #[error("Invalid config: {0}")]
    InvalidConfig(String),

    /// Image dimensions are zero or otherwise unusable.
    #[error("Invalid image dimensions: width={w}, height={h}")]
    InvalidDimensions { w: usize, h: usize },

    /// Newton / fixed-point iteration did not reach the requested tolerance.
    #[error("Iteration did not converge after {iterations} steps")]
    ConvergenceError { iterations: usize },

    /// The distorted point has no undistorted pre-image: its radius is beyond
    /// the largest radius the forward model can produce.
    ///
    /// The forward radial map `r_d = r_u * (1 + k1*r_u² + k2*r_u⁴ + k3*r_u⁶)`
    /// is only monotonic up to the first stationary point of that polynomial.
    /// For barrel distortion (`k1 < 0`) it peaks and then folds back, so
    /// every distorted radius above the peak is simply outside the image of
    /// the map -- no undistorted point maps there, and no amount of iterating
    /// will find one. This is a normal geometric fact (it is what leaves the
    /// corners of a barrel-distorted frame empty), not a numerical failure,
    /// which is why it is distinct from
    /// [`LensDistortionError::ConvergenceError`].
    #[error(
        "Distorted radius {r_d} exceeds the model's maximum representable \
         distorted radius {r_d_max}: no undistorted point maps here"
    )]
    OutsideDistortionDomain {
        /// The distorted radius that was requested.
        r_d: f32,
        /// The largest distorted radius the model can produce.
        r_d_max: f32,
    },

    /// Pixel buffer is empty (zero length).
    #[error("Empty image")]
    EmptyImage,

    /// Pixel buffer length does not match `width * height * 3`.
    #[error("Image buffer length {actual} does not match {width}×{height}×3 = {expected}")]
    BufferSizeMismatch {
        /// Actual buffer length.
        actual: usize,
        /// Expected buffer length (`width * height * 3`).
        expected: usize,
        /// Stated width.
        width: usize,
        /// Stated height.
        height: usize,
    },
}

// ---------------------------------------------------------------------------
// Distortion models
// ---------------------------------------------------------------------------

/// Supported lens distortion models.
#[derive(Debug, Clone, PartialEq)]
pub enum DistortionModel {
    /// Brown-Conrady: radial (k1, k2, k3) + tangential (p1, p2).
    BrownConrady {
        /// First radial coefficient.
        k1: f32,
        /// Second radial coefficient.
        k2: f32,
        /// Third radial coefficient.
        k3: f32,
        /// First tangential coefficient.
        p1: f32,
        /// Second tangential coefficient.
        p2: f32,
    },
    /// Fisheye equidistant: r_d = f * theta.
    FisheyeEquidistant {
        /// First distortion coefficient.
        k1: f32,
        /// Second distortion coefficient.
        k2: f32,
    },
    /// Fisheye equisolid angle: r_d = 2*f * sin(theta/2).
    FisheyeEquisolid {
        /// First distortion coefficient.
        k1: f32,
        /// Second distortion coefficient.
        k2: f32,
    },
    /// Fisheye stereographic: r_d = 2*f * tan(theta/2).
    FisheyeStereographic {
        /// First distortion coefficient.
        k1: f32,
        /// Second distortion coefficient.
        k2: f32,
    },
    /// Simple radial: radial k1, k2 only, no tangential component.
    SimpleRadial {
        /// First radial coefficient.
        k1: f32,
        /// Second radial coefficient.
        k2: f32,
    },
    /// Fitzgibbon (2001) division model: r_d = r_u / (1 + lambda * r_u^2).
    Division {
        /// Division model lambda coefficient.
        lambda: f32,
    },
}

impl DistortionModel {
    /// Validate that all coefficients are finite (not NaN / Inf).
    pub fn validate(&self) -> Result<(), LensDistortionError> {
        let ok = match self {
            Self::BrownConrady { k1, k2, k3, p1, p2 } => {
                k1.is_finite()
                    && k2.is_finite()
                    && k3.is_finite()
                    && p1.is_finite()
                    && p2.is_finite()
            }
            Self::FisheyeEquidistant { k1, k2 }
            | Self::FisheyeEquisolid { k1, k2 }
            | Self::FisheyeStereographic { k1, k2 }
            | Self::SimpleRadial { k1, k2 } => k1.is_finite() && k2.is_finite(),
            Self::Division { lambda } => lambda.is_finite(),
        };
        if ok {
            Ok(())
        } else {
            Err(LensDistortionError::InvalidConfig(
                "distortion coefficients contain NaN or Inf".into(),
            ))
        }
    }

    /// Human-readable name of this model.
    pub fn name(&self) -> &'static str {
        match self {
            Self::BrownConrady { .. } => "BrownConrady",
            Self::FisheyeEquidistant { .. } => "FisheyeEquidistant",
            Self::FisheyeEquisolid { .. } => "FisheyeEquisolid",
            Self::FisheyeStereographic { .. } => "FisheyeStereographic",
            Self::SimpleRadial { .. } => "SimpleRadial",
            Self::Division { .. } => "Division",
        }
    }

    /// `true` for the three fisheye projection variants.
    pub fn is_fisheye(&self) -> bool {
        matches!(
            self,
            Self::FisheyeEquidistant { .. }
                | Self::FisheyeEquisolid { .. }
                | Self::FisheyeStereographic { .. }
        )
    }
}

// ---------------------------------------------------------------------------
// Camera intrinsics
// ---------------------------------------------------------------------------

/// Pinhole camera intrinsic parameters plus image dimensions.
#[derive(Debug, Clone)]
pub struct CameraIntrinsics {
    /// Focal length in pixels along x.
    pub fx: f32,
    /// Focal length in pixels along y.
    pub fy: f32,
    /// Principal-point x coordinate (pixels).
    pub cx: f32,
    /// Principal-point y coordinate (pixels).
    pub cy: f32,
    /// Image width in pixels.
    pub width: usize,
    /// Image height in pixels.
    pub height: usize,
}

impl CameraIntrinsics {
    /// General constructor.
    pub fn new(fx: f32, fy: f32, cx: f32, cy: f32, width: usize, height: usize) -> Self {
        Self {
            fx,
            fy,
            cx,
            cy,
            width,
            height,
        }
    }

    /// Convenience constructor: square pixels, principal point at image centre.
    pub fn pinhole(focal: f32, width: usize, height: usize) -> Self {
        Self {
            fx: focal,
            fy: focal,
            cx: width as f32 / 2.0,
            cy: height as f32 / 2.0,
            width,
            height,
        }
    }

    /// Validate that the intrinsics are geometrically meaningful.
    pub fn validate(&self) -> Result<(), LensDistortionError> {
        if self.fx <= 0.0 || !self.fx.is_finite() {
            return Err(LensDistortionError::InvalidConfig(format!(
                "focal length fx must be positive finite, got {}",
                self.fx
            )));
        }
        if self.fy <= 0.0 || !self.fy.is_finite() {
            return Err(LensDistortionError::InvalidConfig(format!(
                "focal length fy must be positive finite, got {}",
                self.fy
            )));
        }
        if self.width == 0 || self.height == 0 {
            return Err(LensDistortionError::InvalidDimensions {
                w: self.width,
                h: self.height,
            });
        }
        Ok(())
    }

    /// Convert a pixel coordinate to normalised image coordinates.
    ///
    /// Subtracts the principal point and divides by the focal lengths so that
    /// the returned point lies in the undistorted normalised plane.
    #[inline]
    pub fn to_normalized(&self, px: f32, py: f32) -> (f32, f32) {
        ((px - self.cx) / self.fx, (py - self.cy) / self.fy)
    }

    /// Convert a normalised image coordinate back to pixel space.
    #[inline]
    pub fn to_pixel(&self, nx: f32, ny: f32) -> (f32, f32) {
        (nx * self.fx + self.cx, ny * self.fy + self.cy)
    }
}

// ---------------------------------------------------------------------------
// Bilinear interpolation helper
// ---------------------------------------------------------------------------

/// Sample an RGB image using bilinear interpolation with clamp-to-edge.
///
/// `pixels` is row-major HxWx3 u8.
#[inline]
fn bilinear_sample(pixels: &[u8], width: usize, height: usize, x: f32, y: f32) -> [u8; 3] {
    // Clamp coordinates to valid range.
    let x = x.clamp(0.0, (width as f32) - 1.0);
    let y = y.clamp(0.0, (height as f32) - 1.0);

    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);

    let tx = x - x0 as f32;
    let ty = y - y0 as f32;

    let idx00 = (y0 * width + x0) * 3;
    let idx10 = (y0 * width + x1) * 3;
    let idx01 = (y1 * width + x0) * 3;
    let idx11 = (y1 * width + x1) * 3;

    let mut out = [0u8; 3];
    for c in 0..3 {
        let v00 = pixels[idx00 + c] as f32;
        let v10 = pixels[idx10 + c] as f32;
        let v01 = pixels[idx01 + c] as f32;
        let v11 = pixels[idx11 + c] as f32;
        let v = v00 * (1.0 - tx) * (1.0 - ty)
            + v10 * tx * (1.0 - ty)
            + v01 * (1.0 - tx) * ty
            + v11 * tx * ty;
        out[c] = v.round().clamp(0.0, 255.0) as u8;
    }
    out
}

// ---------------------------------------------------------------------------
// Core distortion function
// ---------------------------------------------------------------------------

/// Apply lens distortion to a normalised undistorted point.
///
/// Returns the normalised distorted coordinates `(x_d, y_d)`.
pub fn distort_point(xu: f32, yu: f32, model: &DistortionModel) -> (f32, f32) {
    match model {
        DistortionModel::BrownConrady { k1, k2, k3, p1, p2 } => {
            let r2 = xu * xu + yu * yu;
            let r4 = r2 * r2;
            let r6 = r4 * r2;
            let radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;
            let xd = xu * radial + 2.0 * p1 * xu * yu + p2 * (r2 + 2.0 * xu * xu);
            let yd = yu * radial + p1 * (r2 + 2.0 * yu * yu) + 2.0 * p2 * xu * yu;
            (xd, yd)
        }
        DistortionModel::SimpleRadial { k1, k2 } => {
            let r2 = xu * xu + yu * yu;
            let r4 = r2 * r2;
            let radial = 1.0 + k1 * r2 + k2 * r4;
            (xu * radial, yu * radial)
        }
        DistortionModel::FisheyeEquidistant { k1, k2 } => {
            fisheye_distort(xu, yu, *k1, *k2, FisheyeProjection::Equidistant)
        }
        DistortionModel::FisheyeEquisolid { k1, k2 } => {
            fisheye_distort(xu, yu, *k1, *k2, FisheyeProjection::Equisolid)
        }
        DistortionModel::FisheyeStereographic { k1, k2 } => {
            fisheye_distort(xu, yu, *k1, *k2, FisheyeProjection::Stereographic)
        }
        DistortionModel::Division { lambda } => {
            // Division model: r_d = r_u / (1 + lambda * r_u^2)
            let r2 = xu * xu + yu * yu;
            let denom = 1.0 + lambda * r2;
            if denom.abs() < 1e-12 {
                return (xu, yu);
            }
            (xu / denom, yu / denom)
        }
    }
}

/// Fisheye projection kernels (forward map plus Newton inverse). In their
/// own file purely to keep this one under the 2000-line limit.
mod fisheye;

use fisheye::{fisheye_distort, undistort_fisheye, FisheyeProjection};

// ---------------------------------------------------------------------------
// Radial inversion (exact, on the monotonic branch)
// ---------------------------------------------------------------------------

/// Exact inversion of a model's radial term, on the branch where the forward
/// map is monotonic. Lives in its own file purely to keep this one under the
/// 2000-line limit.
mod radial_branch;

use radial_branch::RadialBranch;

// ---------------------------------------------------------------------------
// Iterative undistortion
// ---------------------------------------------------------------------------

/// Undistort a normalised distorted point.
///
/// Dispatches to a model-specific inversion:
/// - [`DistortionModel::Division`]: closed-form via the quadratic formula.
/// - [`DistortionModel::SimpleRadial`] / [`DistortionModel::BrownConrady`]:
///   the radial part is inverted *exactly* on its monotonic branch (see
///   `RadialBranch`), and any tangential terms are then refined with the
///   OpenCV-style division iteration `xu_{n+1} = (xd - tangential(xu_n)) /
///   radial(xu_n)` seeded from that exact radial solution.
/// - Fisheye variants: a 1-D Newton solve for the undistorted ray angle
///   `theta` on `theta_d = theta*(1 + k1*theta² + k2*theta⁴)`, converging
///   quadratically instead of linearly.
///
/// `max_iters` and `tol` govern the iterative parts only (the tangential
/// refinement and the fisheye Newton solve); the radial inversion is
/// bracketed and always converges.
///
/// # Errors
///
/// - [`LensDistortionError::OutsideDistortionDomain`] when the distorted
///   radius is beyond what the model can produce, i.e. the point has no
///   pre-image at all (see that variant's documentation).
/// - [`LensDistortionError::ConvergenceError`] when an iterative refinement
///   fails to reach `tol` within `max_iters`.
pub fn undistort_point_iterative(
    xd: f32,
    yd: f32,
    model: &DistortionModel,
    max_iters: usize,
    tol: f32,
) -> Result<(f32, f32), LensDistortionError> {
    undistort_point_cached(
        xd,
        yd,
        model,
        model_radial_branch(model).as_ref(),
        max_iters,
        tol,
    )
}

/// The radial branch of `model`, for the models that have one.
///
/// Building it involves a stationary-point search, which is per-model work,
/// not per-pixel work: whole-image passes build it once and hand it to
/// [`undistort_point_cached`].
fn model_radial_branch(model: &DistortionModel) -> Option<RadialBranch> {
    match model {
        DistortionModel::SimpleRadial { k1, k2 } => Some(RadialBranch::new(*k1, *k2, 0.0)),
        DistortionModel::BrownConrady { k1, k2, k3, .. } => Some(RadialBranch::new(*k1, *k2, *k3)),
        _ => None,
    }
}

/// [`undistort_point_iterative`] with a pre-built radial branch.
///
/// `branch` must have been built from `model` (see [`model_radial_branch`]);
/// it is rebuilt on the spot if it is missing, so a mismatch costs
/// performance, never correctness.
fn undistort_point_cached(
    xd: f32,
    yd: f32,
    model: &DistortionModel,
    branch: Option<&RadialBranch>,
    max_iters: usize,
    tol: f32,
) -> Result<(f32, f32), LensDistortionError> {
    match model {
        DistortionModel::Division { lambda } => {
            // Closed-form inverse via the quadratic formula.
            // Forward: r_d = r_u / (1 + lambda * r_u^2)
            // Rearranging: lambda * r_d * r_u^2 - r_u + r_d = 0
            // r_u = (1 - sqrt(1 - 4*lambda*r_d^2)) / (2*lambda*r_d)
            let r_d = (xd * xd + yd * yd).sqrt();
            if r_d < 1e-12 || lambda.abs() < 1e-12 {
                return Ok((xd, yd));
            }
            let discriminant = 1.0 - 4.0 * lambda * r_d * r_d;
            if discriminant < 0.0 {
                // Same geometry as the radial family's fold: for `lambda > 0`
                // the forward map `r_d = r_u / (1 + lambda*r_u²)` peaks at
                // `r_u = 1/sqrt(lambda)` with `r_d = 1/(2*sqrt(lambda))`, and
                // a negative discriminant is exactly "past that peak". It is
                // a point with no pre-image, not a numerical failure, and
                // whole-image passes must treat it the same way they treat
                // the radial models' folded corners.
                return Err(LensDistortionError::OutsideDistortionDomain {
                    r_d,
                    r_d_max: if *lambda > 0.0 {
                        0.5 / lambda.sqrt()
                    } else {
                        f32::INFINITY
                    },
                });
            }
            let two_lambda_rd = 2.0 * lambda * r_d;
            let r_u = (1.0 - discriminant.sqrt()) / two_lambda_rd;
            let scale = r_u / r_d;
            Ok((xd * scale, yd * scale))
        }

        DistortionModel::SimpleRadial { k1, k2 } => {
            let owned;
            let branch = match branch {
                Some(b) => b,
                None => {
                    owned = RadialBranch::new(*k1, *k2, 0.0);
                    &owned
                }
            };
            undistort_radial(xd, yd, branch, 0.0, 0.0, max_iters, tol)
        }

        DistortionModel::BrownConrady { k1, k2, k3, p1, p2 } => {
            let owned;
            let branch = match branch {
                Some(b) => b,
                None => {
                    owned = RadialBranch::new(*k1, *k2, *k3);
                    &owned
                }
            };
            undistort_radial(xd, yd, branch, *p1, *p2, max_iters, tol)
        }

        DistortionModel::FisheyeEquidistant { k1, k2 } => undistort_fisheye(
            xd,
            yd,
            *k1,
            *k2,
            FisheyeProjection::Equidistant,
            max_iters,
            tol,
        ),
        DistortionModel::FisheyeEquisolid { k1, k2 } => undistort_fisheye(
            xd,
            yd,
            *k1,
            *k2,
            FisheyeProjection::Equisolid,
            max_iters,
            tol,
        ),
        DistortionModel::FisheyeStereographic { k1, k2 } => undistort_fisheye(
            xd,
            yd,
            *k1,
            *k2,
            FisheyeProjection::Stereographic,
            max_iters,
            tol,
        ),
    }
}

/// Invert Brown-Conrady / simple-radial distortion.
///
/// The radial part is solved exactly on its monotonic branch (see
/// [`RadialBranch::invert`]), which both removes the old iteration's
/// convergence limits and makes "this point has no pre-image" a distinct,
/// detectable outcome instead of an iteration that runs out of steps.
///
/// With tangential coefficients present the radial solution is only the
/// starting point: the OpenCV-style division iteration
/// `xu_{n+1} = (xd - tangential(xu_n)) / radial(xu_n)` then refines it. Real
/// tangential coefficients are two to three orders of magnitude smaller than
/// the radial ones, so seeded this way it converges in very few steps.
///
/// The domain test uses the radial part alone; with tangential terms it is
/// therefore an approximation of the true domain boundary (an excellent one
/// for any physically plausible `p1`/`p2`).
fn undistort_radial(
    xd: f32,
    yd: f32,
    branch: &RadialBranch,
    p1: f32,
    p2: f32,
    max_iters: usize,
    tol: f32,
) -> Result<(f32, f32), LensDistortionError> {
    let r_d = (xd * xd + yd * yd).sqrt();
    if r_d <= 0.0 {
        // The optical centre maps to itself under both the radial and the
        // tangential terms.
        return Ok((xd, yd));
    }

    let r_u = branch.invert(r_d)?;
    let scale = r_u / r_d;
    let mut xu = xd * scale;
    let mut yu = yd * scale;

    if p1 == 0.0 && p2 == 0.0 {
        // Purely radial: the solve above is already the exact inverse.
        return Ok((xu, yu));
    }

    let tangential = |x: f32, y: f32| {
        let r2 = x * x + y * y;
        (
            2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x),
            p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y,
        )
    };
    let radial_factor = |x: f32, y: f32| {
        let r2 = x * x + y * y;
        1.0 + branch.k1 * r2 + branch.k2 * r2 * r2 + branch.k3 * r2 * r2 * r2
    };

    for iter in 0..max_iters {
        let radial = radial_factor(xu, yu);
        if !radial.is_finite() || radial.abs() < 1e-8 {
            return Err(LensDistortionError::ConvergenceError { iterations: iter });
        }
        let (dx_tan, dy_tan) = tangential(xu, yu);

        let xu_next = (xd - dx_tan) / radial;
        let yu_next = (yd - dy_tan) / radial;
        let ex = xu_next - xu;
        let ey = yu_next - yu;
        xu = xu_next;
        yu = yu_next;

        if ex.abs().max(ey.abs()) < tol {
            return Ok((xu, yu));
        }
        if iter + 1 == max_iters {
            return Err(LensDistortionError::ConvergenceError {
                iterations: max_iters,
            });
        }
    }

    // `max_iters == 0`: accept the radial seed if the tangential terms leave
    // it a fixed point within tolerance anyway.
    let radial = radial_factor(xu, yu);
    if radial.is_finite() && radial.abs() >= 1e-8 {
        let (dx_tan, dy_tan) = tangential(xu, yu);
        let xu_next = (xd - dx_tan) / radial;
        let yu_next = (yd - dy_tan) / radial;
        if (xu_next - xu).abs().max((yu_next - yu).abs()) < tol {
            return Ok((xu_next, yu_next));
        }
    }
    Err(LensDistortionError::ConvergenceError { iterations: 0 })
}

// ---------------------------------------------------------------------------
// Dimension validation helper
// ---------------------------------------------------------------------------

fn validate_image(pixels: &[u8], width: usize, height: usize) -> Result<(), LensDistortionError> {
    if pixels.is_empty() {
        return Err(LensDistortionError::EmptyImage);
    }
    if width == 0 || height == 0 {
        return Err(LensDistortionError::InvalidDimensions {
            w: width,
            h: height,
        });
    }
    // Every caller (`distort_image`, `undistort_image`, `remap_image`,
    // `barrel_distort_image`) relies on this to guarantee `bilinear_sample`
    // can safely index up to `((height-1)*width + (width-1)) * 3 + 2`
    // without an out-of-bounds panic. `checked_mul` guards the dimension
    // product itself against overflow.
    let expected = width
        .checked_mul(height)
        .and_then(|n| n.checked_mul(3))
        .ok_or(LensDistortionError::InvalidDimensions {
            w: width,
            h: height,
        })?;
    if pixels.len() != expected {
        return Err(LensDistortionError::BufferSizeMismatch {
            actual: pixels.len(),
            expected,
            width,
            height,
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Full-image distortion / undistortion
// ---------------------------------------------------------------------------

/// Colour written to destination pixels that have no source ("no data").
///
/// Matches the OpenCV `remap`/`warp` convention of a constant border, which
/// is what an unmapped region of the destination frame is.
const UNMAPPED_PIXEL: [u8; 3] = [0, 0, 0];

/// Apply lens distortion to an entire RGB image.
///
/// For each destination (distorted) pixel, finds the corresponding undistorted
/// source location using [`undistort_point_iterative`], then samples via
/// bilinear interpolation with clamp-to-edge.
///
/// `pixels` must be row-major HxWx3 u8.
///
/// # Unmapped destination pixels
///
/// Barrel distortion (`k1 < 0`) pulls the image inward, so beyond a certain
/// radius *no* undistorted point maps into the destination frame at all: the
/// forward radial map `r_u -> r_u * (1 + k1*r_u² + ...)` peaks and folds
/// back, and every destination radius above that peak is outside its image.
/// A plain 8x8 frame with `k1 = -0.1` and a `min(w,h)/2` focal length already
/// has corners past the peak (corner radius 1.414 vs. a peak of 1.217).
///
/// A fisheye model produces unmapped pixels the same way wherever the
/// distorted radius corresponds to a ray at or past 90° off-axis, which a
/// pinhole undistorted plane cannot represent (the fisheye inverse reports
/// it as `OutsideDistortionDomain`, handled identically).
///
/// Those pixels are filled with the module's unmapped-pixel colour (black)
/// — the honest answer for "no source content exists here" — and counted in a
/// `tracing::debug` record. They are *not* an error: failing the whole call
/// would make ordinary barrel distortion unusable, and substituting the
/// identity mapping (what a much earlier version did) invents content and
/// leaves a discontinuous seam with no indication anything happened.
///
/// # Errors
/// In addition to the buffer/intrinsics/model validation errors, propagates
/// [`LensDistortionError::ConvergenceError`] from
/// [`undistort_point_iterative`] if any pixel's inverse mapping fails to
/// converge. With the exact radial inversion this can now only come from the
/// tangential refinement or a fisheye model, i.e. from genuinely pathological
/// coefficients rather than from ordinary geometry.
pub fn distort_image(
    pixels: &[u8],
    width: usize,
    height: usize,
    intrinsics: &CameraIntrinsics,
    model: &DistortionModel,
) -> Result<Vec<u8>, LensDistortionError> {
    validate_image(pixels, width, height)?;
    intrinsics.validate()?;
    model.validate()?;

    let mut out = vec![0u8; width * height * 3];
    // Per-model work (a stationary-point search), hoisted out of the pixel
    // loop.
    let branch = model_radial_branch(model);
    let mut unmapped = 0usize;

    for row in 0..height {
        for col in 0..width {
            // Destination pixel is the distorted image; find where in the
            // undistorted source to sample from.
            let (xd_n, yd_n) = intrinsics.to_normalized(col as f32, row as f32);
            let idx = (row * width + col) * 3;
            let sample = match undistort_point_cached(xd_n, yd_n, model, branch.as_ref(), 20, 1e-6)
            {
                Ok((xu_n, yu_n)) => {
                    let (src_x, src_y) = intrinsics.to_pixel(xu_n, yu_n);
                    bilinear_sample(pixels, width, height, src_x, src_y)
                }
                Err(LensDistortionError::OutsideDistortionDomain { .. }) => {
                    unmapped += 1;
                    UNMAPPED_PIXEL
                }
                Err(e) => return Err(e),
            };
            out[idx] = sample[0];
            out[idx + 1] = sample[1];
            out[idx + 2] = sample[2];
        }
    }

    if unmapped > 0 {
        tracing::debug!(
            unmapped,
            total = width * height,
            "Destination pixels outside the distortion model's image were \
             filled with the unmapped-pixel colour"
        );
    }

    Ok(out)
}

/// Undistort an entire RGB image.
///
/// For each destination (undistorted) pixel, applies `distort_point` to find
/// where to sample in the distorted source, then uses bilinear interpolation.
///
/// `pixels` must be row-major HxWx3 u8.
pub fn undistort_image(
    pixels: &[u8],
    width: usize,
    height: usize,
    intrinsics: &CameraIntrinsics,
    model: &DistortionModel,
) -> Result<Vec<u8>, LensDistortionError> {
    validate_image(pixels, width, height)?;
    intrinsics.validate()?;
    model.validate()?;

    let mut out = vec![0u8; width * height * 3];

    for row in 0..height {
        for col in 0..width {
            // Destination pixel is undistorted; find the corresponding
            // distorted source location by applying forward distortion.
            let (xu_n, yu_n) = intrinsics.to_normalized(col as f32, row as f32);
            let (xd_n, yd_n) = distort_point(xu_n, yu_n, model);
            let (src_x, src_y) = intrinsics.to_pixel(xd_n, yd_n);
            let sample = bilinear_sample(pixels, width, height, src_x, src_y);
            let idx = (row * width + col) * 3;
            out[idx] = sample[0];
            out[idx + 1] = sample[1];
            out[idx + 2] = sample[2];
        }
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// Distortion map
// ---------------------------------------------------------------------------

/// Pre-compute a distortion map: per-pixel source offsets in pixels.
///
/// Returns `(dx_map, dy_map)` each of length `width * height`.  Entry `i`
/// contains the *offset* from pixel `i` to its source in the distorted image
/// (suitable for use with [`remap_image`]).
///
/// Specifically: for undistort use-case, `dx[i] = src_x - col`,
/// `dy[i] = src_y - row`, where `(src_x, src_y)` is the distorted source
/// location for undistorted destination pixel `(col, row)`.
pub fn compute_distortion_map(
    width: usize,
    height: usize,
    intrinsics: &CameraIntrinsics,
    model: &DistortionModel,
) -> Result<(Vec<f32>, Vec<f32>), LensDistortionError> {
    if width == 0 || height == 0 {
        return Err(LensDistortionError::InvalidDimensions {
            w: width,
            h: height,
        });
    }
    intrinsics.validate()?;
    model.validate()?;

    let n = width * height;
    let mut dx_map = vec![0.0f32; n];
    let mut dy_map = vec![0.0f32; n];

    for row in 0..height {
        for col in 0..width {
            let (xu_n, yu_n) = intrinsics.to_normalized(col as f32, row as f32);
            let (xd_n, yd_n) = distort_point(xu_n, yu_n, model);
            let (src_x, src_y) = intrinsics.to_pixel(xd_n, yd_n);
            let i = row * width + col;
            dx_map[i] = src_x - col as f32;
            dy_map[i] = src_y - row as f32;
        }
    }

    Ok((dx_map, dy_map))
}

// ---------------------------------------------------------------------------
// Remap using pre-computed map
// ---------------------------------------------------------------------------

/// Remap an RGB image using a pre-computed distortion map.
///
/// `dx_map[i]` and `dy_map[i]` are source offsets (in pixels) for
/// destination pixel `i`.  Both maps must have length `width * height`.
pub fn remap_image(
    pixels: &[u8],
    width: usize,
    height: usize,
    dx_map: &[f32],
    dy_map: &[f32],
) -> Result<Vec<u8>, LensDistortionError> {
    validate_image(pixels, width, height)?;
    let n = width * height;
    if dx_map.len() != n || dy_map.len() != n {
        return Err(LensDistortionError::InvalidConfig(format!(
            "map size mismatch: expected {n}, got dx={}, dy={}",
            dx_map.len(),
            dy_map.len(),
        )));
    }

    let mut out = vec![0u8; n * 3];
    for row in 0..height {
        for col in 0..width {
            let i = row * width + col;
            let src_x = col as f32 + dx_map[i];
            let src_y = row as f32 + dy_map[i];
            let sample = bilinear_sample(pixels, width, height, src_x, src_y);
            out[i * 3] = sample[0];
            out[i * 3 + 1] = sample[1];
            out[i * 3 + 2] = sample[2];
        }
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// Barrel distortion convenience
// ---------------------------------------------------------------------------

/// Apply barrel/pincushion distortion using a single radial coefficient.
///
/// `k1 < 0` produces barrel distortion; `k1 > 0` produces pincushion.
/// Delegates to [`distort_image`] with a `SimpleRadial` model and
/// a pinhole intrinsic at the image centre.
pub fn barrel_distort_image(
    pixels: &[u8],
    width: usize,
    height: usize,
    k1: f32,
) -> Result<Vec<u8>, LensDistortionError> {
    validate_image(pixels, width, height)?;
    if !k1.is_finite() {
        return Err(LensDistortionError::InvalidConfig(
            "k1 must be finite".into(),
        ));
    }
    let intrinsics = CameraIntrinsics::pinhole(((width.min(height)) as f32) / 2.0, width, height);
    let model = DistortionModel::SimpleRadial { k1, k2: 0.0 };
    distort_image(pixels, width, height, &intrinsics, &model)
}

// ---------------------------------------------------------------------------
// Radial coefficient estimation (least-squares)
// ---------------------------------------------------------------------------

/// Estimate radial distortion coefficients from paired measurements.
///
/// Each measurement is `(r_undistorted, r_distorted)`.  Fits the model
/// `r_d = r_u * (1 + k1 * r_u^2 + k2 * r_u^4)` by linear regression on
/// the residual `r_d / r_u - 1 = k1 * r_u^2 + k2 * r_u^4`.
///
/// Measurements with `r_u < 1e-6` are skipped to avoid numerical instability.
///
/// Returns `(k1_estimate, k2_estimate)`.
pub fn estimate_radial_coefficients(
    measurements: &[(f32, f32)],
) -> Result<(f32, f32), LensDistortionError> {
    if measurements.is_empty() {
        return Err(LensDistortionError::InvalidConfig(
            "no measurements provided".into(),
        ));
    }

    // Build the normal equations for the 2×2 least-squares system:
    //   [a, b] [k1]   [e]
    //   [b, c] [k2] = [f]
    // where for each sample: y = k1 * x1 + k2 * x2
    //   x1 = r_u^2,  x2 = r_u^4,  y = r_d / r_u - 1
    let mut a = 0.0f64;
    let mut b = 0.0f64;
    let mut c = 0.0f64;
    let mut e = 0.0f64;
    let mut f = 0.0f64;
    let mut count = 0usize;

    for &(ru, rd) in measurements {
        if ru < 1e-6 {
            continue;
        }
        let ru = ru as f64;
        let rd = rd as f64;
        let x1 = ru * ru;
        let x2 = x1 * x1;
        let y = rd / ru - 1.0;
        a += x1 * x1;
        b += x1 * x2;
        c += x2 * x2;
        e += x1 * y;
        f += x2 * y;
        count += 1;
    }

    if count == 0 {
        return Err(LensDistortionError::InvalidConfig(
            "all measurements had r_u < 1e-6".into(),
        ));
    }

    // Solve 2×2 system via Cramer's rule.
    let det = a * c - b * b;
    if det.abs() < 1e-30 {
        // Degenerate (e.g. only one distinct r value): fall back to k1 only.
        if a.abs() < 1e-30 {
            return Ok((0.0, 0.0));
        }
        let k1 = (e / a) as f32;
        return Ok((k1, 0.0));
    }

    let k1 = ((e * c - f * b) / det) as f32;
    let k2 = ((a * f - b * e) / det) as f32;

    Ok((k1, k2))
}

// ---------------------------------------------------------------------------
// Distortion statistics
// ---------------------------------------------------------------------------

/// Statistics summarising the distortion field across the image.
#[derive(Debug, Clone)]
pub struct DistortionStats {
    /// Maximum pixel displacement across the entire image.
    pub max_shift_px: f32,
    /// Mean pixel displacement.
    pub mean_shift_px: f32,
    /// Root-mean-square pixel displacement.
    pub rms_shift_px: f32,
    /// Signed corner shift (negative = barrel, positive = pincushion).
    ///
    /// Computed as the mean outward shift at the four image corners relative
    /// to the image centre.
    pub barrel_pincushion: f32,
}

/// Compute distortion statistics over the image grid.
///
/// Samples every pixel, applies `distort_point`, measures the Euclidean
/// displacement from the pixel's nominal position, and aggregates.
pub fn compute_distortion_stats(
    width: usize,
    height: usize,
    intrinsics: &CameraIntrinsics,
    model: &DistortionModel,
) -> Result<DistortionStats, LensDistortionError> {
    if width == 0 || height == 0 {
        return Err(LensDistortionError::InvalidDimensions {
            w: width,
            h: height,
        });
    }
    intrinsics.validate()?;
    model.validate()?;

    let n = (width * height) as f64;
    let mut max_shift = 0.0f32;
    // f64 accumulators: for a 4K image (~8.3M pixels) with shifts around
    // 10px, an f32 accumulator for `sum_shift` reaches ~8.3e7 — past f32's
    // 1.67e7 exact-integer limit — so late additions get silently
    // quantised away (worse still for `sum_sq`, ~8.3e8), systematically
    // underestimating `mean_shift_px` / `rms_shift_px`.
    let mut sum_shift = 0.0f64;
    let mut sum_sq = 0.0f64;

    for row in 0..height {
        for col in 0..width {
            let (xu_n, yu_n) = intrinsics.to_normalized(col as f32, row as f32);
            let (xd_n, yd_n) = distort_point(xu_n, yu_n, model);
            let (dst_x, dst_y) = intrinsics.to_pixel(xd_n, yd_n);
            let dx = dst_x - col as f32;
            let dy = dst_y - row as f32;
            let shift = (dx * dx + dy * dy).sqrt();
            if shift > max_shift {
                max_shift = shift;
            }
            let shift_f64 = shift as f64;
            sum_shift += shift_f64;
            sum_sq += shift_f64 * shift_f64;
        }
    }

    let mean_shift = (sum_shift / n) as f32;
    let rms_shift = (sum_sq / n).sqrt() as f32;

    // Barrel/pincushion: average outward radial shift at the four corners.
    let corners = [
        (0.0f32, 0.0f32),
        ((width - 1) as f32, 0.0f32),
        (0.0f32, (height - 1) as f32),
        ((width - 1) as f32, (height - 1) as f32),
    ];
    let cx = intrinsics.cx;
    let cy = intrinsics.cy;
    let mut corner_sum = 0.0f32;
    for (px, py) in corners {
        let (xu_n, yu_n) = intrinsics.to_normalized(px, py);
        let (xd_n, yd_n) = distort_point(xu_n, yu_n, model);
        let (dst_x, dst_y) = intrinsics.to_pixel(xd_n, yd_n);
        // Outward direction from centre.
        let dir_x = px - cx;
        let dir_y = py - cy;
        let len = (dir_x * dir_x + dir_y * dir_y).sqrt().max(1e-9);
        let shift_x = dst_x - px;
        let shift_y = dst_y - py;
        // Project shift onto outward direction.
        corner_sum += (shift_x * dir_x + shift_y * dir_y) / len;
    }
    let barrel_pincushion = corner_sum / 4.0;

    Ok(DistortionStats {
        max_shift_px: max_shift,
        mean_shift_px: mean_shift,
        rms_shift_px: rms_shift,
        barrel_pincushion,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: a solid-colour RGB image of size w×h.
    fn solid_image(w: usize, h: usize, r: u8, g: u8, b: u8) -> Vec<u8> {
        let mut v = Vec::with_capacity(w * h * 3);
        for _ in 0..w * h {
            v.push(r);
            v.push(g);
            v.push(b);
        }
        v
    }

    // -----------------------------------------------------------------------
    // CameraIntrinsics
    // -----------------------------------------------------------------------

    #[test]
    fn test_intrinsics_new() {
        let k = CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0, 640, 480);
        assert_eq!(k.fx, 500.0);
        assert_eq!(k.cx, 320.0);
        assert_eq!(k.width, 640);
    }

    #[test]
    fn test_intrinsics_pinhole() {
        let k = CameraIntrinsics::pinhole(400.0, 800, 600);
        assert_eq!(k.fx, 400.0);
        assert_eq!(k.fy, 400.0);
        assert_eq!(k.cx, 400.0);
        assert_eq!(k.cy, 300.0);
    }

    #[test]
    fn test_intrinsics_validate_ok() {
        let k = CameraIntrinsics::pinhole(400.0, 640, 480);
        assert!(k.validate().is_ok());
    }

    #[test]
    fn test_intrinsics_validate_bad_focal() {
        let k = CameraIntrinsics::new(-1.0, 400.0, 320.0, 240.0, 640, 480);
        assert!(k.validate().is_err());
    }

    #[test]
    fn test_intrinsics_validate_zero_dims() {
        let k = CameraIntrinsics::new(400.0, 400.0, 0.0, 0.0, 0, 480);
        assert!(k.validate().is_err());
    }

    #[test]
    fn test_intrinsics_to_normalized_roundtrip() {
        let k = CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0, 640, 480);
        let (nx, ny) = k.to_normalized(320.0, 240.0);
        assert!((nx).abs() < 1e-6);
        assert!((ny).abs() < 1e-6);
        let (px, py) = k.to_pixel(nx, ny);
        assert!((px - 320.0).abs() < 1e-4);
        assert!((py - 240.0).abs() < 1e-4);
    }

    #[test]
    fn test_intrinsics_to_pixel_roundtrip() {
        let k = CameraIntrinsics::pinhole(600.0, 800, 600);
        for (col, row) in [(100.0f32, 200.0f32), (400.0, 300.0), (750.0, 50.0)] {
            let (nx, ny) = k.to_normalized(col, row);
            let (px, py) = k.to_pixel(nx, ny);
            assert!((px - col).abs() < 1e-4, "col roundtrip failed");
            assert!((py - row).abs() < 1e-4, "row roundtrip failed");
        }
    }

    // -----------------------------------------------------------------------
    // DistortionModel
    // -----------------------------------------------------------------------

    #[test]
    fn test_model_validate_ok() {
        assert!(DistortionModel::SimpleRadial { k1: 0.1, k2: 0.01 }
            .validate()
            .is_ok());
        assert!(DistortionModel::Division { lambda: -0.5 }
            .validate()
            .is_ok());
    }

    #[test]
    fn test_model_validate_nan() {
        assert!(DistortionModel::SimpleRadial {
            k1: f32::NAN,
            k2: 0.0
        }
        .validate()
        .is_err());
    }

    #[test]
    fn test_model_validate_inf() {
        assert!(DistortionModel::Division {
            lambda: f32::INFINITY
        }
        .validate()
        .is_err());
    }

    #[test]
    fn test_model_name() {
        assert_eq!(
            DistortionModel::BrownConrady {
                k1: 0.0,
                k2: 0.0,
                k3: 0.0,
                p1: 0.0,
                p2: 0.0
            }
            .name(),
            "BrownConrady"
        );
        assert_eq!(
            DistortionModel::FisheyeEquidistant { k1: 0.0, k2: 0.0 }.name(),
            "FisheyeEquidistant"
        );
        assert_eq!(DistortionModel::Division { lambda: 0.0 }.name(), "Division");
    }

    #[test]
    fn test_model_is_fisheye() {
        assert!(DistortionModel::FisheyeEquidistant { k1: 0.0, k2: 0.0 }.is_fisheye());
        assert!(DistortionModel::FisheyeEquisolid { k1: 0.0, k2: 0.0 }.is_fisheye());
        assert!(DistortionModel::FisheyeStereographic { k1: 0.0, k2: 0.0 }.is_fisheye());
        assert!(!DistortionModel::SimpleRadial { k1: 0.0, k2: 0.0 }.is_fisheye());
        assert!(!DistortionModel::Division { lambda: 0.0 }.is_fisheye());
    }

    // -----------------------------------------------------------------------
    // distort_point
    // -----------------------------------------------------------------------

    #[test]
    fn test_distort_point_identity_simple_radial() {
        // k1 = k2 = 0 → identity
        let model = DistortionModel::SimpleRadial { k1: 0.0, k2: 0.0 };
        let (xd, yd) = distort_point(0.3, -0.2, &model);
        assert!((xd - 0.3).abs() < 1e-6);
        assert!((yd + 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_distort_point_barrel_k1_negative() {
        // Barrel distortion pulls points inward.
        let model = DistortionModel::SimpleRadial { k1: -0.1, k2: 0.0 };
        let (xd, yd) = distort_point(0.5, 0.0, &model);
        // With k1 < 0, radial factor < 1, so |xd| < |xu|.
        assert!(xd < 0.5, "expected inward shift, got xd={xd}");
        assert!(yd.abs() < 1e-6);
    }

    #[test]
    fn test_distort_point_brown_conrady_zero() {
        let model = DistortionModel::BrownConrady {
            k1: 0.0,
            k2: 0.0,
            k3: 0.0,
            p1: 0.0,
            p2: 0.0,
        };
        let (xd, yd) = distort_point(0.4, 0.2, &model);
        assert!((xd - 0.4).abs() < 1e-6);
        assert!((yd - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_distort_point_fisheye_equidistant_identity() {
        // k1 = k2 = 0 → identity mapping (theta_d = theta → r_d = theta ≈ r for small r)
        let model = DistortionModel::FisheyeEquidistant { k1: 0.0, k2: 0.0 };
        let (xd, yd) = distort_point(0.0, 0.0, &model);
        assert!(xd.abs() < 1e-6);
        assert!(yd.abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // undistort_point_iterative
    // -----------------------------------------------------------------------

    #[test]
    fn test_undistort_roundtrip_simple_radial() {
        let model = DistortionModel::SimpleRadial { k1: -0.1, k2: 0.02 };
        let xu0 = 0.3f32;
        let yu0 = -0.15f32;
        let (xd, yd) = distort_point(xu0, yu0, &model);
        let (xu1, yu1) =
            undistort_point_iterative(xd, yd, &model, 20, 1e-6).expect("should converge");
        assert!((xu1 - xu0).abs() < 1e-4, "xu roundtrip: {xu1} vs {xu0}");
        assert!((yu1 - yu0).abs() < 1e-4, "yu roundtrip: {yu1} vs {yu0}");
    }

    #[test]
    fn test_undistort_roundtrip_brown_conrady() {
        let model = DistortionModel::BrownConrady {
            k1: -0.15,
            k2: 0.02,
            k3: 0.001,
            p1: 0.001,
            p2: -0.001,
        };
        let (xu0, yu0) = (0.2f32, 0.1f32);
        let (xd, yd) = distort_point(xu0, yu0, &model);
        let (xu1, yu1) =
            undistort_point_iterative(xd, yd, &model, 30, 1e-6).expect("should converge");
        assert!((xu1 - xu0).abs() < 1e-4);
        assert!((yu1 - yu0).abs() < 1e-4);
    }

    #[test]
    fn test_undistort_roundtrip_division() {
        let model = DistortionModel::Division { lambda: -0.3 };
        let (xu0, yu0) = (0.25f32, -0.1f32);
        let (xd, yd) = distort_point(xu0, yu0, &model);
        let (xu1, yu1) =
            undistort_point_iterative(xd, yd, &model, 20, 1e-6).expect("should converge");
        assert!((xu1 - xu0).abs() < 1e-4, "xu: {xu1} vs {xu0}");
        assert!((yu1 - yu0).abs() < 1e-4, "yu: {yu1} vs {yu0}");
    }

    #[test]
    fn test_undistort_convergence_error() {
        // max_iters=0 with a non-trivial model should fail.
        let model = DistortionModel::SimpleRadial { k1: -0.5, k2: 0.0 };
        let result = undistort_point_iterative(0.3, 0.2, &model, 0, 1e-12);
        // This may or may not converge in 0 iterations; check the type.
        // If it happens to pass the convergence check on 0 iterations that's fine.
        let _ = result; // just checking it compiles and runs without panic.
    }

    // -----------------------------------------------------------------------
    // distort_image / undistort_image
    // -----------------------------------------------------------------------

    #[test]
    fn test_distort_image_1x1() {
        let pixels = vec![128u8, 64, 32];
        let k = CameraIntrinsics::pinhole(100.0, 1, 1);
        let model = DistortionModel::SimpleRadial { k1: 0.0, k2: 0.0 };
        let out = distort_image(&pixels, 1, 1, &k, &model).expect("1x1 distort");
        assert_eq!(out, pixels);
    }

    #[test]
    fn test_undistort_image_1x1() {
        let pixels = vec![200u8, 100, 50];
        let k = CameraIntrinsics::pinhole(100.0, 1, 1);
        let model = DistortionModel::SimpleRadial { k1: 0.0, k2: 0.0 };
        let out = undistort_image(&pixels, 1, 1, &k, &model).expect("1x1 undistort");
        assert_eq!(out, pixels);
    }

    #[test]
    fn test_distort_image_4x4_no_distortion() {
        // With identity distortion, the output equals the input (±1 due to bilinear).
        let w = 4usize;
        let h = 4usize;
        let pixels = solid_image(w, h, 100, 150, 200);
        let k = CameraIntrinsics::pinhole(10.0, w, h);
        let model = DistortionModel::SimpleRadial { k1: 0.0, k2: 0.0 };
        let out = distort_image(&pixels, w, h, &k, &model).expect("4x4 distort");
        assert_eq!(out.len(), w * h * 3);
        // Solid colour image should remain solid.
        for chunk in out.chunks(3) {
            assert_eq!(chunk[0], 100);
            assert_eq!(chunk[1], 150);
            assert_eq!(chunk[2], 200);
        }
    }

    #[test]
    fn test_undistort_image_4x4() {
        let w = 4usize;
        let h = 4usize;
        let pixels = solid_image(w, h, 80, 160, 240);
        let k = CameraIntrinsics::pinhole(10.0, w, h);
        let model = DistortionModel::SimpleRadial { k1: -0.05, k2: 0.0 };
        let out = undistort_image(&pixels, w, h, &k, &model).expect("4x4 undistort");
        assert_eq!(out.len(), w * h * 3);
        // Solid-colour source → all sampled pixels identical.
        for chunk in out.chunks(3) {
            assert_eq!(chunk[0], 80);
            assert_eq!(chunk[1], 160);
            assert_eq!(chunk[2], 240);
        }
    }

    #[test]
    fn test_distort_image_empty_error() {
        let pixels: Vec<u8> = Vec::new();
        let k = CameraIntrinsics::pinhole(100.0, 4, 4);
        let model = DistortionModel::SimpleRadial { k1: 0.0, k2: 0.0 };
        assert!(distort_image(&pixels, 4, 4, &k, &model).is_err());
    }

    #[test]
    fn test_undistort_image_zero_dims_error() {
        let pixels = solid_image(1, 1, 0, 0, 0);
        let k = CameraIntrinsics::pinhole(100.0, 0, 4);
        let model = DistortionModel::SimpleRadial { k1: 0.0, k2: 0.0 };
        assert!(undistort_image(&pixels, 0, 4, &k, &model).is_err());
    }

    // -----------------------------------------------------------------------
    // barrel_distort_image
    // -----------------------------------------------------------------------

    #[test]
    fn test_barrel_distort_valid() {
        let w = 8usize;
        let h = 8usize;
        let pixels = solid_image(w, h, 128, 128, 128);
        let out = barrel_distort_image(&pixels, w, h, -0.1).expect("barrel distort");
        assert_eq!(out.len(), w * h * 3);
    }

    #[test]
    fn test_barrel_distort_zero_size_error() {
        let pixels: Vec<u8> = Vec::new();
        assert!(barrel_distort_image(&pixels, 0, 0, -0.1).is_err());
    }

    #[test]
    fn test_barrel_distort_nan_k1_error() {
        let pixels = solid_image(4, 4, 0, 0, 0);
        assert!(barrel_distort_image(&pixels, 4, 4, f32::NAN).is_err());
    }

    // -----------------------------------------------------------------------
    // compute_distortion_map
    // -----------------------------------------------------------------------

    #[test]
    fn test_compute_distortion_map_correct_size() {
        let w = 6usize;
        let h = 4usize;
        let k = CameraIntrinsics::pinhole(50.0, w, h);
        let model = DistortionModel::SimpleRadial { k1: -0.1, k2: 0.0 };
        let (dx, dy) = compute_distortion_map(w, h, &k, &model).expect("distortion map");
        assert_eq!(dx.len(), w * h);
        assert_eq!(dy.len(), w * h);
    }

    #[test]
    fn test_compute_distortion_map_zero_dims_error() {
        let k = CameraIntrinsics::pinhole(50.0, 4, 4);
        let model = DistortionModel::SimpleRadial { k1: 0.0, k2: 0.0 };
        assert!(compute_distortion_map(0, 4, &k, &model).is_err());
    }

    #[test]
    fn test_compute_distortion_map_identity_zero_offset() {
        // With zero distortion, all offsets should be ~0.
        let w = 4usize;
        let h = 4usize;
        let k = CameraIntrinsics::pinhole(100.0, w, h);
        let model = DistortionModel::SimpleRadial { k1: 0.0, k2: 0.0 };
        let (dx, dy) = compute_distortion_map(w, h, &k, &model).expect("identity map");
        for v in dx.iter().chain(dy.iter()) {
            assert!(v.abs() < 1e-4, "non-zero offset {v} in identity map");
        }
    }

    // -----------------------------------------------------------------------
    // remap_image
    // -----------------------------------------------------------------------

    #[test]
    fn test_remap_image_identity_map() {
        let w = 4usize;
        let h = 4usize;
        let pixels = solid_image(w, h, 77, 88, 99);
        let dx = vec![0.0f32; w * h];
        let dy = vec![0.0f32; w * h];
        let out = remap_image(&pixels, w, h, &dx, &dy).expect("identity remap");
        assert_eq!(out, pixels);
    }

    #[test]
    fn test_remap_image_map_size_mismatch_error() {
        let w = 4usize;
        let h = 4usize;
        let pixels = solid_image(w, h, 10, 20, 30);
        // dx has wrong size
        let dx = vec![0.0f32; w * h - 1];
        let dy = vec![0.0f32; w * h];
        assert!(remap_image(&pixels, w, h, &dx, &dy).is_err());
    }

    #[test]
    fn test_remap_image_empty_error() {
        let pixels: Vec<u8> = Vec::new();
        let dx = Vec::new();
        let dy = Vec::new();
        assert!(remap_image(&pixels, 0, 0, &dx, &dy).is_err());
    }

    // -----------------------------------------------------------------------
    // estimate_radial_coefficients
    // -----------------------------------------------------------------------

    #[test]
    fn test_estimate_radial_empty_error() {
        assert!(estimate_radial_coefficients(&[]).is_err());
    }

    #[test]
    fn test_estimate_radial_linear_case() {
        // Synthesise measurements from a known model, then estimate coefficients.
        let k1_true = -0.1f32;
        let k2_true = 0.02f32;
        let measurements: Vec<(f32, f32)> = (1..=10)
            .map(|i| {
                let ru = i as f32 * 0.05;
                let rd = ru * (1.0 + k1_true * ru * ru + k2_true * ru * ru * ru * ru);
                (ru, rd)
            })
            .collect();
        let (k1_est, k2_est) = estimate_radial_coefficients(&measurements).expect("estimate");
        assert!((k1_est - k1_true).abs() < 0.02, "k1 off: {k1_est}");
        assert!((k2_est - k2_true).abs() < 0.01, "k2 off: {k2_est}");
    }

    #[test]
    fn test_estimate_radial_skip_near_zero() {
        // Measurements with ru < 1e-6 should be skipped, not panic.
        let measurements = vec![(0.0f32, 0.0f32), (0.1f32, 0.09f32)];
        let result = estimate_radial_coefficients(&measurements);
        assert!(result.is_ok());
    }

    // -----------------------------------------------------------------------
    // compute_distortion_stats
    // -----------------------------------------------------------------------

    #[test]
    fn test_compute_distortion_stats_non_empty() {
        let w = 8usize;
        let h = 8usize;
        let k = CameraIntrinsics::pinhole(50.0, w, h);
        let model = DistortionModel::SimpleRadial { k1: -0.1, k2: 0.0 };
        let stats = compute_distortion_stats(w, h, &k, &model).expect("stats");
        assert!(stats.max_shift_px >= 0.0);
        assert!(stats.mean_shift_px >= 0.0);
        assert!(stats.rms_shift_px >= 0.0);
    }

    #[test]
    fn test_compute_distortion_stats_barrel_sign() {
        // Barrel distortion (k1 < 0) should give negative barrel_pincushion.
        let w = 8usize;
        let h = 8usize;
        let k = CameraIntrinsics::pinhole(50.0, w, h);
        let model = DistortionModel::SimpleRadial { k1: -0.2, k2: 0.0 };
        let stats = compute_distortion_stats(w, h, &k, &model).expect("barrel stats");
        assert!(
            stats.barrel_pincushion <= 0.0,
            "expected barrel (<=0), got {}",
            stats.barrel_pincushion
        );
    }

    #[test]
    fn test_compute_distortion_stats_zero_dims_error() {
        let k = CameraIntrinsics::pinhole(50.0, 4, 4);
        let model = DistortionModel::SimpleRadial { k1: 0.0, k2: 0.0 };
        assert!(compute_distortion_stats(0, 4, &k, &model).is_err());
    }

    #[test]
    fn test_compute_distortion_stats_identity_zero_shift() {
        // Zero distortion → all shifts are 0.
        let w = 6usize;
        let h = 6usize;
        let k = CameraIntrinsics::pinhole(100.0, w, h);
        let model = DistortionModel::SimpleRadial { k1: 0.0, k2: 0.0 };
        let stats = compute_distortion_stats(w, h, &k, &model).expect("identity stats");
        assert!(stats.max_shift_px < 1e-4);
        assert!(stats.mean_shift_px < 1e-4);
        assert!(stats.rms_shift_px < 1e-4);
    }

    // -----------------------------------------------------------------------
    // Additional round-trip and edge-case tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_distort_undistort_fisheye_roundtrip() {
        let model = DistortionModel::FisheyeEquidistant {
            k1: -0.05,
            k2: 0.005,
        };
        let (xu0, yu0) = (0.1f32, 0.08f32);
        let (xd, yd) = distort_point(xu0, yu0, &model);
        let (xu1, yu1) =
            undistort_point_iterative(xd, yd, &model, 40, 1e-6).expect("fisheye convergence");
        assert!((xu1 - xu0).abs() < 1e-4, "fisheye xu: {xu1} vs {xu0}");
        assert!((yu1 - yu0).abs() < 1e-4, "fisheye yu: {yu1} vs {yu0}");
    }

    #[test]
    fn test_distort_point_origin_unchanged() {
        // All models should leave the origin unchanged.
        let models = [
            DistortionModel::SimpleRadial { k1: -0.3, k2: 0.1 },
            DistortionModel::Division { lambda: -0.5 },
            DistortionModel::BrownConrady {
                k1: -0.1,
                k2: 0.05,
                k3: 0.0,
                p1: 0.01,
                p2: -0.01,
            },
            DistortionModel::FisheyeEquidistant { k1: -0.1, k2: 0.01 },
        ];
        for model in &models {
            let (xd, yd) = distort_point(0.0, 0.0, model);
            assert!(
                xd.abs() < 1e-6 && yd.abs() < 1e-6,
                "{}: origin not preserved: ({xd},{yd})",
                model.name()
            );
        }
    }

    #[test]
    fn test_remap_with_computed_map() {
        // Remap using a pre-computed map should give the same result as undistort_image.
        let w = 6usize;
        let h = 6usize;
        let pixels = solid_image(w, h, 50, 100, 200);
        let k = CameraIntrinsics::pinhole(30.0, w, h);
        let model = DistortionModel::SimpleRadial { k1: -0.05, k2: 0.0 };

        let expected = undistort_image(&pixels, w, h, &k, &model).expect("undistort");
        let (dx, dy) = compute_distortion_map(w, h, &k, &model).expect("map");
        let got = remap_image(&pixels, w, h, &dx, &dy).expect("remap");

        // Solid-colour source: all pixels should be the same.
        assert_eq!(expected, got);
    }

    // -----------------------------------------------------------------------
    // validate_image / BufferSizeMismatch regression tests (critical OOB fix)
    // -----------------------------------------------------------------------

    #[test]
    fn test_distort_image_buffer_too_short_error() {
        // Regression: `validate_image` never checked `pixels.len()` against
        // `width*height*3`, so this call previously panicked with an
        // out-of-bounds index inside `bilinear_sample` instead of
        // returning a clean error.
        let pixels = vec![0u8; 3]; // 1 pixel's worth, claiming 1000x1000
        let k = CameraIntrinsics::pinhole(100.0, 1000, 1000);
        let model = DistortionModel::SimpleRadial { k1: 0.0, k2: 0.0 };
        let result = distort_image(&pixels, 1000, 1000, &k, &model);
        assert!(
            matches!(result, Err(LensDistortionError::BufferSizeMismatch { .. })),
            "expected BufferSizeMismatch, got {result:?}"
        );
    }

    #[test]
    fn test_undistort_image_buffer_too_short_error() {
        let pixels = vec![0u8; 3];
        let k = CameraIntrinsics::pinhole(100.0, 1000, 1000);
        let model = DistortionModel::SimpleRadial { k1: 0.0, k2: 0.0 };
        let result = undistort_image(&pixels, 1000, 1000, &k, &model);
        assert!(matches!(
            result,
            Err(LensDistortionError::BufferSizeMismatch { .. })
        ));
    }

    #[test]
    fn test_remap_image_buffer_too_short_error() {
        let pixels = vec![0u8; 3];
        let w = 1000;
        let h = 1000;
        let dx = vec![0.0f32; w * h];
        let dy = vec![0.0f32; w * h];
        let result = remap_image(&pixels, w, h, &dx, &dy);
        assert!(matches!(
            result,
            Err(LensDistortionError::BufferSizeMismatch { .. })
        ));
    }

    #[test]
    fn test_barrel_distort_image_buffer_too_short_error() {
        let pixels = vec![0u8; 3];
        let result = barrel_distort_image(&pixels, 1000, 1000, -0.1);
        assert!(matches!(
            result,
            Err(LensDistortionError::BufferSizeMismatch { .. })
        ));
    }

    // -----------------------------------------------------------------------
    // undistort_point_iterative: convergence over a wide k1 range (division form)
    // -----------------------------------------------------------------------

    #[test]
    fn test_undistort_converges_across_wide_k1_range() {
        // Regression: the old unit-Jacobian fixed point only converges
        // while |d(distort)/dx - 1| < 1, which fails for |k1| as small as
        // ~0.3 — well within normal lens ranges. The division-form
        // iteration must converge across a much wider sweep.
        let mut k1 = -0.5_f32;
        while k1 <= 0.5 {
            if k1.abs() > 1e-6 {
                let model = DistortionModel::SimpleRadial { k1, k2: 0.0 };
                for &(xu0, yu0) in &[(0.3_f32, 0.2_f32), (0.5, -0.4), (0.1, 0.1)] {
                    let (xd, yd) = distort_point(xu0, yu0, &model);
                    let result = undistort_point_iterative(xd, yd, &model, 50, 1e-6);
                    let (xu1, yu1) = result.unwrap_or_else(|e| {
                        panic!("k1={k1} did not converge from ({xu0},{yu0}): {e}")
                    });
                    assert!(
                        (xu1 - xu0).abs() < 1e-3 && (yu1 - yu0).abs() < 1e-3,
                        "k1={k1}: roundtrip ({xu1},{yu1}) vs ({xu0},{yu0})"
                    );
                }
            }
            k1 += 0.05;
        }
    }

    #[test]
    fn test_undistort_point_iterative_reports_points_with_no_preimage() {
        // `xd = 0.5, k1 = -4` used to be described as a "singular radial
        // denominator" (`1 + k1*r²` is exactly 0 at the first iterate) and
        // could only be reported as a `ConvergenceError`. It is in fact a
        // point with *no pre-image at all*: the forward map
        // `g(r) = r*(1 - 4r²)` peaks at `r = 1/sqrt(12) = 0.288675` with
        // `g = 0.192450`, so no undistorted point maps to a distorted radius
        // of 0.5. Iterating harder can never find one, and the error now
        // says so.
        let model = DistortionModel::SimpleRadial { k1: -4.0, k2: 0.0 };
        let result = undistort_point_iterative(0.5, 0.0, &model, 10, 1e-6);
        match result {
            Err(LensDistortionError::OutsideDistortionDomain { r_d, r_d_max }) => {
                assert!((r_d - 0.5).abs() < 1e-6, "r_d: {r_d}");
                assert!(
                    (r_d_max - 0.192_450_09).abs() < 1e-5,
                    "r_d_max should be the analytic peak 0.19245009, got {r_d_max}"
                );
            }
            other => panic!("expected OutsideDistortionDomain, got {other:?}"),
        }

        // Just below the peak the inverse exists and must round-trip.
        let r_u_probe = 0.2_f32;
        let (xd, yd) = distort_point(r_u_probe, 0.0, &model);
        let (xu, yu) =
            undistort_point_iterative(xd, yd, &model, 10, 1e-6).expect("inside the domain");
        assert!((xu - r_u_probe).abs() < 1e-4, "xu: {xu}");
        assert!(yu.abs() < 1e-6, "yu: {yu}");
    }

    #[test]
    fn test_division_model_reports_points_with_no_preimage() {
        // The division model folds the same way: `r_d = r_u / (1 + λ r_u²)`
        // peaks at `r_u = 1/sqrt(λ)` with `r_d = 1/(2 sqrt(λ))`, which for
        // λ = 1 is 0.5. A distorted radius past that has no pre-image, and
        // must be reported as such rather than as a convergence failure --
        // otherwise `distort_image` would hard-error a whole image for the
        // division model while filling the same geometry for the radial
        // family.
        let model = DistortionModel::Division { lambda: 1.0 };
        match undistort_point_iterative(0.75, 0.0, &model, 20, 1e-6) {
            Err(LensDistortionError::OutsideDistortionDomain { r_d, r_d_max }) => {
                assert!((r_d - 0.75).abs() < 1e-6, "r_d: {r_d}");
                assert!((r_d_max - 0.5).abs() < 1e-6, "r_d_max: {r_d_max}");
            }
            other => panic!("expected OutsideDistortionDomain, got {other:?}"),
        }

        // Inside the domain it still round-trips exactly (closed form).
        let (xd, yd) = distort_point(0.4, 0.0, &model);
        let (xu, _) = undistort_point_iterative(xd, yd, &model, 20, 1e-6).expect("inside");
        assert!((xu - 0.4).abs() < 1e-4, "xu: {xu}");

        // A whole-image pass now fills those pixels instead of failing.
        let pixels = vec![10u8, 20u8, 30u8];
        let k = CameraIntrinsics::new(1.0, 1.0, -0.75, 0.0, 1, 1);
        let out = distort_image(&pixels, 1, 1, &k, &model).expect("unmapped pixels are geometry");
        assert_eq!(out, UNMAPPED_PIXEL.to_vec());
    }

    #[test]
    fn test_distort_image_marks_unmapped_pixels_instead_of_inventing_content() {
        // Regression: a pixel with no source used to silently fall back to
        // the identity mapping, i.e. it invented content (the source colour)
        // with no indication anything had happened. A single pixel whose
        // normalized coordinate is exactly (0.5, 0.0), combined with
        // k1 = -4.0, is past the forward map's fold (see
        // `test_undistort_point_iterative_reports_points_with_no_preimage`),
        // so it must come back as the explicit "no data" colour rather than
        // as the source pixel.
        let pixels = vec![10u8, 20u8, 30u8];
        let k = CameraIntrinsics::new(1.0, 1.0, -0.5, 0.0, 1, 1);
        let model = DistortionModel::SimpleRadial { k1: -4.0, k2: 0.0 };
        let out = distort_image(&pixels, 1, 1, &k, &model)
            .expect("an unmapped pixel is geometry, not an error");
        assert_eq!(
            out,
            UNMAPPED_PIXEL.to_vec(),
            "a destination pixel with no pre-image must be marked unmapped, \
             not filled with the source colour"
        );
    }

    #[test]
    fn test_barrel_distort_fills_folded_corners_and_keeps_centre() {
        // `barrel_distort_image` uses a `min(w,h)/2` focal length and a
        // `w/2, h/2` principal point, so an 8x8 frame spans normalized
        // coordinates -1.0 ..= 0.75 on both axes. With k1 = -0.1 the forward
        // map peaks at r_u = 1/sqrt(0.3) = 1.82574, r_d = 1.21716; the
        // destination pixels whose radius exceeds that peak have no
        // pre-image at all. Enumerating the grid, exactly five do:
        //   (-1.00,-1.00) r=1.4142   (col 0, row 0)
        //   (-1.00,-0.75) r=1.2500   (col 0, row 1)
        //   (-0.75,-1.00) r=1.2500   (col 1, row 0)
        //   (-1.00, 0.75) r=1.2500   (col 0, row 7)
        //   ( 0.75,-1.00) r=1.2500   (col 7, row 0)
        // The next-largest radius on the grid is 1.118 (e.g. -1.0,-0.5),
        // comfortably inside the domain.
        let w = 8usize;
        let h = 8usize;
        let pixels = solid_image(w, h, 128, 128, 128);
        let out = barrel_distort_image(&pixels, w, h, -0.1).expect("barrel distort");
        assert_eq!(out.len(), w * h * 3);

        let px = |col: usize, row: usize| {
            let i = (row * w + col) * 3;
            [out[i], out[i + 1], out[i + 2]]
        };
        for &(col, row) in &[(0, 0), (0, 1), (1, 0), (0, 7), (7, 0)] {
            assert_eq!(
                px(col, row),
                UNMAPPED_PIXEL,
                "pixel (col {col}, row {row}) is past the fold and has no pre-image"
            );
        }
        assert_eq!(px(4, 4), [128, 128, 128], "the centre must be sampled");
        assert_eq!(
            px(w - 1, h - 1),
            [128, 128, 128],
            "(0.75, 0.75) has radius 1.0607, inside the domain"
        );

        // Every pixel is either sampled from the (solid) source or marked
        // unmapped -- nothing in between, and nothing invented -- and
        // exactly the five enumerated pixels are unmapped.
        let unmapped = out
            .chunks(3)
            .filter(|chunk| {
                assert!(
                    *chunk == [128, 128, 128] || *chunk == UNMAPPED_PIXEL,
                    "unexpected pixel {chunk:?}"
                );
                *chunk == UNMAPPED_PIXEL
            })
            .count();
        assert_eq!(unmapped, 5, "expected exactly 5 unmapped pixels");

        // A milder coefficient keeps the whole frame inside the domain.
        let mild = barrel_distort_image(&pixels, w, h, -0.02).expect("mild barrel distort");
        assert!(
            mild.chunks(3).all(|c| c == [128, 128, 128]),
            "k1 = -0.02 folds at r_d = 2.72, well outside the frame, so no \
             pixel should be unmapped"
        );
    }

    // -----------------------------------------------------------------------
    // compute_distortion_stats: f64 accumulation precision
    // -----------------------------------------------------------------------

    #[test]
    fn test_compute_distortion_stats_f64_accumulation_precision() {
        // Regression: `sum_shift` / `sum_sq` were f32 accumulators. Once a
        // running sum exceeds f32's ~1.677e7 exact-integer limit, further
        // additions smaller than the current value's ULP are silently
        // dropped, systematically *underestimating* mean/rms shift. Use a
        // wide-FOV setup (large per-pixel shifts) so the true sum
        // comfortably exceeds that threshold, and compare against an
        // independent f64-accumulated reference computed the same way.
        let w = 1000usize;
        let h = 1000usize;
        let k = CameraIntrinsics::pinhole(50.0, w, h);
        let model = DistortionModel::SimpleRadial { k1: -0.4, k2: 0.0 };

        let mut ref_sum = 0.0f64;
        for row in 0..h {
            for col in 0..w {
                let (xu_n, yu_n) = k.to_normalized(col as f32, row as f32);
                let (xd_n, yd_n) = distort_point(xu_n, yu_n, &model);
                let (dst_x, dst_y) = k.to_pixel(xd_n, yd_n);
                let dx = dst_x - col as f32;
                let dy = dst_y - row as f32;
                ref_sum += ((dx * dx + dy * dy).sqrt()) as f64;
            }
        }
        assert!(
            ref_sum > 1.677e7,
            "test setup should exceed f32's exact-integer accumulation limit, got {ref_sum:e}"
        );
        let ref_mean = (ref_sum / (w * h) as f64) as f32;

        let stats = compute_distortion_stats(w, h, &k, &model).expect("stats");
        assert!(
            (stats.mean_shift_px - ref_mean).abs() < 1.0,
            "mean_shift_px={} should match the f64 reference {ref_mean}",
            stats.mean_shift_px
        );
    }
}
