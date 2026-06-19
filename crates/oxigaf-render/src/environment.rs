//! Environment map and spherical harmonics lighting for 3D Gaussian Splatting.
//!
//! This module provides:
//! - Spherical harmonic basis evaluation (up to degree L=2, 9 coefficients)
//! - `SphericalHarmonicsLight`: RGB SH lighting coefficients
//! - `EnvironmentMap`: equirectangular HDR panorama with bilinear sampling
//! - Monte Carlo projection of environment maps to SH coefficients
//! - Per-Gaussian SH colour evaluation (the 3DGS convention with +0.5 offset)

use crate::RenderError;

// ---------------------------------------------------------------------------
// SH constants
// ---------------------------------------------------------------------------

/// L=0 SH normalisation constant: 1 / (2 * sqrt(pi)).
const SH_C0: f32 = 0.282_094_8_f32;

/// L=1 SH normalisation constant: sqrt(3 / (4*pi)).
const SH_C1: f32 = 0.488_602_52_f32;

/// L=2 SH normalisation constants (five, one per m-index).
const SH_C2: [f32; 5] = [
    1.092_548_5_f32,  // m=-2: sqrt(15/(4*pi))
    1.092_548_5_f32,  // m=-1: sqrt(15/(4*pi))
    0.315_391_57_f32, // m= 0: sqrt(5/(16*pi))
    1.092_548_5_f32,  // m=+1: sqrt(15/(4*pi))
    0.546_274_24_f32, // m=+2: sqrt(15/(16*pi))
];

// ---------------------------------------------------------------------------
// SH basis evaluation
// ---------------------------------------------------------------------------

/// Evaluate all 9 spherical-harmonic basis functions through degree L=2.
///
/// `direction` must be a unit vector `[x, y, z]`.
///
/// Returns `[Y_0_0, Y_1_{-1}, Y_1_0, Y_1_1, Y_2_{-2}, Y_2_{-1}, Y_2_0, Y_2_1, Y_2_2]`.
pub fn sh_basis_up_to_l2(direction: [f32; 3]) -> [f32; 9] {
    let x = direction[0];
    let y = direction[1];
    let z = direction[2];

    [
        // L=0
        SH_C0,
        // L=1
        SH_C1 * y,
        SH_C1 * z,
        SH_C1 * x,
        // L=2
        SH_C2[0] * (x * y),
        SH_C2[1] * (y * z),
        SH_C2[2] * (3.0 * z * z - 1.0),
        SH_C2[3] * (x * z),
        SH_C2[4] * (x * x - y * y),
    ]
}

// ---------------------------------------------------------------------------
// SphericalHarmonicsLight
// ---------------------------------------------------------------------------

/// RGB spherical-harmonics lighting coefficients (up to degree L=2).
///
/// Storage layout: `coeffs[basis_i * 3 + channel]` for `basis_i` in `0..9`
/// and `channel` in `{0=R, 1=G, 2=B}`.
#[derive(Debug, Clone, PartialEq)]
pub struct SphericalHarmonicsLight {
    /// `[9 × 3]` flat array of SH coefficients.
    pub coeffs: [f32; 27],
    /// Maximum SH degree stored (0, 1, or 2).
    pub max_degree: u32,
}

impl SphericalHarmonicsLight {
    /// All coefficients zero; `max_degree` = 2.
    pub fn zeros() -> Self {
        Self {
            coeffs: [0.0_f32; 27],
            max_degree: 2,
        }
    }

    /// Ambient (L=0 only) light of the given RGB colour.
    pub fn ambient(color: [f32; 3]) -> Self {
        let mut s = Self::zeros();
        s.max_degree = 0;
        s.coeffs[0] = color[0];
        s.coeffs[1] = color[1];
        s.coeffs[2] = color[2];
        s
    }

    /// Evaluate the SH lighting model for a unit direction vector.
    ///
    /// Returns a clamped `[R, G, B]` in `[0, 1]`.
    pub fn evaluate(&self, direction: [f32; 3]) -> [f32; 3] {
        let num_basis = match self.max_degree {
            0 => 1_usize,
            1 => 4_usize,
            _ => 9_usize,
        };
        let basis = sh_basis_up_to_l2(direction);

        let mut rgb = [0.0_f32; 3];
        for (i, &basis_val) in basis.iter().enumerate().take(num_basis) {
            for (c, rgb_val) in rgb.iter_mut().enumerate() {
                *rgb_val += basis_val * self.coeffs[i * 3 + c];
            }
        }
        [
            rgb[0].clamp(0.0, 1.0),
            rgb[1].clamp(0.0, 1.0),
            rgb[2].clamp(0.0, 1.0),
        ]
    }

    /// Add a constant ambient term to the L=0 coefficient.
    pub fn add_ambient(&mut self, color: [f32; 3]) {
        self.coeffs[0] += color[0];
        self.coeffs[1] += color[1];
        self.coeffs[2] += color[2];
    }

    /// Scale every coefficient by `factor`.
    pub fn scale(&mut self, factor: f32) {
        for c in self.coeffs.iter_mut() {
            *c *= factor;
        }
    }
}

// ---------------------------------------------------------------------------
// EnvironmentMap
// ---------------------------------------------------------------------------

/// Equirectangular (panoramic) HDR environment map stored as row-major f32 RGB.
///
/// Pixel `(px, py)` starts at `data[(py * width + px) * 3]`.
#[derive(Debug, Clone)]
pub struct EnvironmentMap {
    /// Row-major `[height × width × 3]` pixel data (HDR f32 RGB).
    pub data: Vec<f32>,
    /// Map width in pixels.
    pub width: usize,
    /// Map height in pixels.
    pub height: usize,
}

impl EnvironmentMap {
    /// Construct from existing pixel data.
    ///
    /// Returns `Err` if `data.len() != width * height * 3`.
    pub fn new(width: usize, height: usize, data: Vec<f32>) -> Result<Self, RenderError> {
        let expected = width * height * 3;
        if data.len() != expected {
            return Err(RenderError::MismatchedBufferSizes {
                expected,
                actual: data.len(),
            });
        }
        Ok(Self {
            data,
            width,
            height,
        })
    }

    /// 1×1 environment map filled with a single solid colour.
    pub fn solid(color: [f32; 3]) -> Self {
        Self {
            data: vec![color[0], color[1], color[2]],
            width: 1,
            height: 1,
        }
    }

    /// 256×128 vertically-graduated sky environment.
    ///
    /// Row 0 uses `zenith`, row 64 uses `horizon`, row 127 uses `nadir`.
    /// Each channel is linearly interpolated between the anchor rows.
    pub fn gradient_sky(zenith: [f32; 3], horizon: [f32; 3], nadir: [f32; 3]) -> Self {
        const W: usize = 256;
        const H: usize = 128;
        let mut data = Vec::with_capacity(W * H * 3);

        for py in 0..H {
            // Map row index to a colour via two linear segments.
            let color = if py <= 64 {
                let t = py as f32 / 64.0_f32;
                lerp_rgb(zenith, horizon, t)
            } else {
                let t = (py - 64) as f32 / (127 - 64) as f32;
                lerp_rgb(horizon, nadir, t)
            };
            for _px in 0..W {
                data.push(color[0]);
                data.push(color[1]);
                data.push(color[2]);
            }
        }

        Self {
            data,
            width: W,
            height: H,
        }
    }

    /// Read pixel `(px, py)` (no bounds-checking clamp is applied; caller ensures validity).
    #[inline]
    pub fn pixel(&self, px: usize, py: usize) -> [f32; 3] {
        let idx = (py * self.width + px) * 3;
        [self.data[idx], self.data[idx + 1], self.data[idx + 2]]
    }

    /// Sample the environment map in the given direction using bilinear interpolation.
    ///
    /// `direction` is a **unit** vector `[x, y, z]` in a **y-up** world.
    ///
    /// - u wraps horizontally (panoramic).
    /// - v is clamped vertically (poles).
    pub fn sample(&self, direction: [f32; 3]) -> [f32; 3] {
        use std::f32::consts::PI;

        let x = direction[0];
        let y = direction[1];
        let z = direction[2];

        // Polar coordinates (y-up).
        let theta = y.clamp(-1.0, 1.0).acos(); // [0, pi]
        let phi = x.atan2(z); // [-pi, pi]

        // UV in [0,1].
        let u = (phi + PI) / (2.0 * PI); // [0, 1]
        let v = theta / PI; // [0, 1]

        let w = self.width as f32;
        let h = self.height as f32;

        // Continuous pixel coordinates.
        let fx = u * w - 0.5;
        let fy = v * h - 0.5;

        let x0 = fx.floor() as i64;
        let y0 = fy.floor() as i64;
        let tx = fx - fx.floor();
        let ty = fy - fy.floor();

        // Horizontal: wrap; vertical: clamp.
        let px0 = wrap_coord(x0, self.width as i64) as usize;
        let px1 = wrap_coord(x0 + 1, self.width as i64) as usize;
        let py0 = clamp_coord(y0, self.height as i64) as usize;
        let py1 = clamp_coord(y0 + 1, self.height as i64) as usize;

        let c00 = self.pixel(px0, py0);
        let c10 = self.pixel(px1, py0);
        let c01 = self.pixel(px0, py1);
        let c11 = self.pixel(px1, py1);

        let mut out = [0.0_f32; 3];
        for c in 0..3 {
            let top = c00[c] * (1.0 - tx) + c10[c] * tx;
            let bot = c01[c] * (1.0 - tx) + c11[c] * tx;
            out[c] = top * (1.0 - ty) + bot * ty;
        }
        out
    }
}

/// Linear interpolation between two RGB colours.
#[inline]
fn lerp_rgb(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}

/// Wrap an integer coordinate into `[0, size)`.
#[inline]
fn wrap_coord(coord: i64, size: i64) -> i64 {
    ((coord % size) + size) % size
}

/// Clamp an integer coordinate into `[0, size-1]`.
#[inline]
fn clamp_coord(coord: i64, size: i64) -> i64 {
    coord.clamp(0, size - 1)
}

// ---------------------------------------------------------------------------
// SH projection from environment map
// ---------------------------------------------------------------------------

/// Project an environment map to SH coefficients using the Fibonacci sphere
/// Monte Carlo estimator (`num_samples` uniformly distributed directions).
///
/// The weight per sample is `4π / num_samples` (solid-angle integral over S²).
pub fn project_environment_to_sh(
    env: &EnvironmentMap,
    num_samples: usize,
) -> SphericalHarmonicsLight {
    use std::f32::consts::PI;

    let golden_ratio = (1.0_f32 + 5.0_f32.sqrt()) / 2.0_f32;
    let n = num_samples as f32;
    let weight = 4.0_f32 * PI / n;

    let mut light = SphericalHarmonicsLight::zeros();

    for i in 0..num_samples {
        let fi = i as f32;
        let theta = (1.0_f32 - 2.0_f32 * (fi + 0.5_f32) / n)
            .clamp(-1.0, 1.0)
            .acos();
        let phi = 2.0_f32 * PI * fi / golden_ratio;

        // y-up direction from spherical coordinates.
        let sin_theta = theta.sin();
        let direction = [sin_theta * phi.cos(), theta.cos(), sin_theta * phi.sin()];

        let color = env.sample(direction);
        let basis = sh_basis_up_to_l2(direction);

        for (b, &basis_val) in basis.iter().enumerate() {
            for (c, &color_val) in color.iter().enumerate() {
                light.coeffs[b * 3 + c] += color_val * basis_val * weight;
            }
        }
    }

    light.max_degree = 2;
    light
}

// ---------------------------------------------------------------------------
// Per-Gaussian SH evaluation (3DGS convention)
// ---------------------------------------------------------------------------

/// Evaluate view-dependent colour for a single Gaussian given its SH coefficients.
///
/// `sh_coeffs` layout: `[r0,g0,b0, r1,g1,b1, …]` for SH basis indices 0, 1, 2, …
/// (length must be a multiple of 3; the number of basis functions used is `len / 3`).
///
/// The result includes the **+0.5** DC offset used in the 3DGS implementation, and
/// is clamped to `[0, 1]`.
pub fn evaluate_gaussian_sh(sh_coeffs: &[f32], direction: [f32; 3]) -> [f32; 3] {
    let num_basis = sh_coeffs.len() / 3;
    let basis = sh_basis_up_to_l2(direction);

    let mut rgb = [0.5_f32; 3]; // +0.5 DC bias
    for i in 0..num_basis {
        for c in 0..3 {
            rgb[c] += basis[i] * sh_coeffs[i * 3 + c];
        }
    }
    [
        rgb[0].clamp(0.0, 1.0),
        rgb[1].clamp(0.0, 1.0),
        rgb[2].clamp(0.0, 1.0),
    ]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    // Tolerance for floating-point comparisons.
    const EPS: f32 = 1e-4;

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    // -----------------------------------------------------------------------
    // SH basis tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_sh_basis_l0_constant() {
        // Y_0_0 must be the same constant for every direction.
        let dirs = [
            [1.0_f32, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.577_350_3, 0.577_350_3, 0.577_350_3],
        ];
        for d in &dirs {
            let b = sh_basis_up_to_l2(*d);
            assert!(
                approx_eq(b[0], SH_C0, EPS),
                "Y_0_0 = {} != {} for dir {:?}",
                b[0],
                SH_C0,
                d
            );
        }
    }

    #[test]
    fn test_sh_basis_l1_x_direction() {
        // For direction [1, 0, 0]: Y_1_1 = SH_C1 * x = SH_C1.
        let b = sh_basis_up_to_l2([1.0, 0.0, 0.0]);
        assert!(
            approx_eq(b[3], SH_C1, EPS),
            "Y_1_1 should be SH_C1 for [1,0,0], got {}",
            b[3]
        );
        // Y_1_-1 and Y_1_0 should be zero.
        assert!(approx_eq(b[1], 0.0, EPS), "Y_1_-1 != 0 for [1,0,0]");
        assert!(approx_eq(b[2], 0.0, EPS), "Y_1_0 != 0 for [1,0,0]");
    }

    #[test]
    fn test_sh_basis_l2_shape() {
        // Verify the 5 L=2 values for direction [1, 0, 0].
        let b = sh_basis_up_to_l2([1.0, 0.0, 0.0]);
        // Y_2_-2 = SH_C2[0] * x*y = 0
        assert!(approx_eq(b[4], 0.0, EPS));
        // Y_2_-1 = SH_C2[1] * y*z = 0
        assert!(approx_eq(b[5], 0.0, EPS));
        // Y_2_0  = SH_C2[2] * (3z²-1) = SH_C2[2] * (-1)
        let expected_y20 = SH_C2[2] * (3.0 * 0.0 - 1.0);
        assert!(approx_eq(b[6], expected_y20, EPS));
        // Y_2_1  = SH_C2[3] * x*z = 0
        assert!(approx_eq(b[7], 0.0, EPS));
        // Y_2_2  = SH_C2[4] * (x²-y²) = SH_C2[4]
        assert!(approx_eq(b[8], SH_C2[4], EPS));
    }

    #[test]
    fn test_sh_basis_normalization_property() {
        // Monte Carlo estimate of the integral of Y_0_0² over S² should be ≈ 1.
        // ∫ Y_l_m² dΩ = 1 by orthonormality of real SH.
        let num_samples = 100_000_usize;
        let golden_ratio = (1.0_f32 + 5.0_f32.sqrt()) / 2.0_f32;
        let n = num_samples as f32;

        let mut sum = 0.0_f32;
        for i in 0..num_samples {
            let fi = i as f32;
            let theta = (1.0_f32 - 2.0_f32 * (fi + 0.5_f32) / n)
                .clamp(-1.0, 1.0)
                .acos();
            let phi = 2.0_f32 * PI * fi / golden_ratio;
            let direction = [
                theta.sin() * phi.cos(),
                theta.cos(),
                theta.sin() * phi.sin(),
            ];
            let b = sh_basis_up_to_l2(direction);
            sum += b[0] * b[0]; // Y_0_0²
        }
        let integral = sum * (4.0_f32 * PI / n);
        // Expected: 1.0 (orthonormality). Tolerance 1% for Monte Carlo.
        assert!(
            approx_eq(integral, 1.0, 0.01),
            "∫ Y_0_0² dΩ ≈ {} (expected 1.0)",
            integral
        );
    }

    // -----------------------------------------------------------------------
    // SphericalHarmonicsLight tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_sh_light_zeros() {
        let s = SphericalHarmonicsLight::zeros();
        for c in &s.coeffs {
            assert_eq!(*c, 0.0_f32);
        }
        assert_eq!(s.max_degree, 2);
    }

    #[test]
    fn test_sh_light_ambient() {
        let color = [0.3_f32, 0.5_f32, 0.7_f32];
        let s = SphericalHarmonicsLight::ambient(color);
        assert_eq!(s.max_degree, 0);
        assert!(approx_eq(s.coeffs[0], color[0], EPS));
        assert!(approx_eq(s.coeffs[1], color[1], EPS));
        assert!(approx_eq(s.coeffs[2], color[2], EPS));
        // All higher coefficients should be zero.
        for i in 3..27 {
            assert_eq!(s.coeffs[i], 0.0_f32);
        }
    }

    #[test]
    fn test_sh_light_evaluate_ambient() {
        // Ambient-only: evaluate([any dir]) = clamp(C0 * SH_C0)
        let raw_intensity = 1.5_f32; // > 1 to trigger clamping.
        let s = SphericalHarmonicsLight::ambient([raw_intensity, 0.4, 0.2]);
        let result = s.evaluate([1.0, 0.0, 0.0]);
        let expected_r = (raw_intensity * SH_C0).clamp(0.0, 1.0);
        let expected_g = (0.4_f32 * SH_C0).clamp(0.0, 1.0);
        let expected_b = (0.2_f32 * SH_C0).clamp(0.0, 1.0);
        assert!(approx_eq(result[0], expected_r, EPS));
        assert!(approx_eq(result[1], expected_g, EPS));
        assert!(approx_eq(result[2], expected_b, EPS));
    }

    #[test]
    fn test_sh_light_add_ambient() {
        let mut s = SphericalHarmonicsLight::zeros();
        s.add_ambient([0.1, 0.2, 0.3]);
        s.add_ambient([0.4, 0.5, 0.6]);
        assert!(approx_eq(s.coeffs[0], 0.5, EPS));
        assert!(approx_eq(s.coeffs[1], 0.7, EPS));
        assert!(approx_eq(s.coeffs[2], 0.9, EPS));
    }

    #[test]
    fn test_sh_light_scale() {
        let mut s = SphericalHarmonicsLight::zeros();
        s.coeffs[0] = 2.0;
        s.coeffs[5] = 3.0;
        s.scale(0.5);
        assert!(approx_eq(s.coeffs[0], 1.0, EPS));
        assert!(approx_eq(s.coeffs[5], 1.5, EPS));
    }

    // -----------------------------------------------------------------------
    // EnvironmentMap tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_environment_map_new_valid() {
        let data = vec![0.0_f32; 4 * 2 * 3];
        let env = EnvironmentMap::new(4, 2, data).expect("should succeed");
        assert_eq!(env.width, 4);
        assert_eq!(env.height, 2);
    }

    #[test]
    fn test_environment_map_invalid_size() {
        let data = vec![0.0_f32; 5]; // Wrong length.
        let result = EnvironmentMap::new(4, 2, data);
        assert!(result.is_err());
    }

    #[test]
    fn test_environment_map_solid_sample() {
        let color = [0.6_f32, 0.3_f32, 0.9_f32];
        let env = EnvironmentMap::solid(color);
        // Any direction must return the same colour.
        let dirs = [[1.0_f32, 0.0, 0.0], [0.0, 1.0, 0.0], [-0.707, 0.0, 0.707]];
        for d in &dirs {
            let s = env.sample(*d);
            for c in 0..3 {
                assert!(
                    approx_eq(s[c], color[c], EPS),
                    "channel {c}: got {}, expected {}",
                    s[c],
                    color[c]
                );
            }
        }
    }

    #[test]
    fn test_environment_map_gradient_sky() {
        let zenith = [0.0_f32, 0.0_f32, 1.0_f32];
        let horizon = [0.5_f32, 0.5_f32, 0.5_f32];
        let nadir = [1.0_f32, 0.0_f32, 0.0_f32];
        let env = EnvironmentMap::gradient_sky(zenith, horizon, nadir);

        assert_eq!(env.width, 256);
        assert_eq!(env.height, 128);
        assert_eq!(env.data.len(), 256 * 128 * 3);

        // Row 0 should be zenith.
        let top = env.pixel(0, 0);
        for c in 0..3 {
            assert!(approx_eq(top[c], zenith[c], EPS), "zenith mismatch");
        }

        // Row 64 should be horizon.
        let mid = env.pixel(0, 64);
        for c in 0..3 {
            assert!(approx_eq(mid[c], horizon[c], EPS), "horizon mismatch");
        }

        // Row 127 should be nadir.
        let bot = env.pixel(0, 127);
        for c in 0..3 {
            assert!(approx_eq(bot[c], nadir[c], EPS), "nadir mismatch");
        }
    }

    #[test]
    fn test_environment_map_sample_top_direction() {
        // Direction pointing straight up [0,1,0] → theta=0 → v=0 → top row → zenith colour.
        let zenith = [0.1_f32, 0.5_f32, 0.9_f32];
        let horizon = [0.5_f32, 0.5_f32, 0.5_f32];
        let nadir = [0.9_f32, 0.5_f32, 0.1_f32];
        let env = EnvironmentMap::gradient_sky(zenith, horizon, nadir);

        let s = env.sample([0.0, 1.0, 0.0]);
        // Should be close to the zenith colour (bilinear may blend edge pixels).
        for c in 0..3 {
            assert!(
                approx_eq(s[c], zenith[c], 0.05),
                "channel {c}: got {}, expected zenith {}",
                s[c],
                zenith[c]
            );
        }
    }

    // -----------------------------------------------------------------------
    // SH projection tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_project_to_sh_ambient_env() {
        // A constant environment maps to a pure L=0 SH signal.
        let color = [0.8_f32, 0.6_f32, 0.4_f32];
        let env = EnvironmentMap::solid(color);
        let sh = project_environment_to_sh(&env, 1000);

        // The L=0 coefficient reconstructs the average radiance:
        // integral of constant env * Y_0_0 over S² = color * Y_0_0 * 4π
        // SH coefficient c_0 = color * 4π / N * Σ Y_0_0 ≈ color * 4π * Y_0_0
        // Evaluating sh at any direction = c_0 * Y_0_0 ≈ color * 4π * Y_0_0²
        // = color * 1.0 (by orthonormality).  So evaluated result ≈ color, clamped.
        let evaluated = sh.evaluate([0.0, 1.0, 0.0]);
        for c in 0..3 {
            assert!(
                approx_eq(evaluated[c], color[c].clamp(0.0, 1.0), 0.05),
                "channel {c}: got {}, expected {}",
                evaluated[c],
                color[c]
            );
        }
    }

    #[test]
    fn test_project_to_sh_preserves_color() {
        // After projection, re-evaluating a white environment at many directions
        // should give roughly [1,1,1] (clamped).
        let env = EnvironmentMap::solid([1.0, 1.0, 1.0]);
        let sh = project_environment_to_sh(&env, 2000);

        let dirs = [
            [1.0_f32, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ];
        for d in &dirs {
            let result = sh.evaluate(*d);
            for (c, &val) in result.iter().enumerate() {
                // Allow generous tolerance due to Monte Carlo and SH truncation.
                assert!(
                    val > 0.7,
                    "direction {:?} channel {c}: value {} too low",
                    d,
                    val
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // evaluate_gaussian_sh tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_evaluate_gaussian_sh_degree_0() {
        // Single SH term (L=0): coeffs = [r, g, b], result = 0.5 + r*SH_C0, etc.
        let r = 0.2_f32;
        let g = -0.3_f32; // Negative to test clamping.
        let b = 0.4_f32;
        let coeffs = [r, g, b];
        let result = evaluate_gaussian_sh(&coeffs, [1.0, 0.0, 0.0]);

        let expected_r = (0.5 + r * SH_C0).clamp(0.0, 1.0);
        let expected_g = (0.5 + g * SH_C0).clamp(0.0, 1.0);
        let expected_b = (0.5 + b * SH_C0).clamp(0.0, 1.0);
        assert!(approx_eq(result[0], expected_r, EPS));
        assert!(approx_eq(result[1], expected_g, EPS));
        assert!(approx_eq(result[2], expected_b, EPS));
    }

    #[test]
    fn test_evaluate_gaussian_sh_degree_1() {
        // 4-basis (L=0 + L=1): 12 coefficients.
        let mut coeffs = [0.0_f32; 12];
        // Set L=0 term: coeffs[0..3]
        coeffs[0] = 0.5;
        coeffs[1] = 0.5;
        coeffs[2] = 0.5;
        // Set Y_1_1 (index 3): coeffs[9..12]
        coeffs[9] = 1.0; // r
        coeffs[10] = 0.0;
        coeffs[11] = 0.0;

        let direction = [1.0_f32, 0.0, 0.0]; // x=1 → Y_1_1 = SH_C1
        let result = evaluate_gaussian_sh(&coeffs, direction);

        // R = clamp(0.5 + 0.5*SH_C0 + 1.0*SH_C1)
        let basis = sh_basis_up_to_l2(direction);
        let expected_r = (0.5 + 0.5 * basis[0] + 1.0 * basis[3]).clamp(0.0, 1.0);
        assert!(
            approx_eq(result[0], expected_r, EPS),
            "R: got {}, expected {}",
            result[0],
            expected_r
        );
    }

    // -----------------------------------------------------------------------
    // Fibonacci sphere coverage test
    // -----------------------------------------------------------------------

    #[test]
    fn test_fibonacci_sphere_coverage() {
        // Generate 500 Fibonacci-sphere samples and verify each lies on the unit sphere.
        let num_samples = 500_usize;
        let golden_ratio = (1.0_f32 + 5.0_f32.sqrt()) / 2.0_f32;
        let n = num_samples as f32;

        for i in 0..num_samples {
            let fi = i as f32;
            let theta = (1.0_f32 - 2.0_f32 * (fi + 0.5_f32) / n)
                .clamp(-1.0, 1.0)
                .acos();
            let phi = 2.0_f32 * PI * fi / golden_ratio;
            let direction = [
                theta.sin() * phi.cos(),
                theta.cos(),
                theta.sin() * phi.sin(),
            ];
            let norm_sq = direction[0] * direction[0]
                + direction[1] * direction[1]
                + direction[2] * direction[2];
            assert!(
                approx_eq(norm_sq, 1.0, 1e-5),
                "sample {i}: norm² = {norm_sq}, direction = {direction:?}"
            );
        }

        // Additionally verify that all sample y-coordinates span [-1, 1].
        let ys: Vec<f32> = (0..num_samples)
            .map(|i| {
                let fi = i as f32;
                let theta = (1.0_f32 - 2.0_f32 * (fi + 0.5_f32) / n)
                    .clamp(-1.0, 1.0)
                    .acos();
                theta.cos()
            })
            .collect();
        let y_min = ys.iter().cloned().fold(f32::INFINITY, f32::min);
        let y_max = ys.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(y_min < -0.9, "y_min {} not near -1", y_min);
        assert!(y_max > 0.9, "y_max {} not near +1", y_max);
    }
}
