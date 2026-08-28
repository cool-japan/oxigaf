//! CPU reference rasterizer for gradient verification.
//!
//! This module provides a pure-CPU implementation of 3D Gaussian Splatting
//! rasterization. It matches the GPU implementation logic exactly but runs
//! entirely on the CPU for easier debugging and gradient verification.
//!
//! The CPU rasterizer serves as ground truth for validating GPU gradients
//! through finite-difference approximation.

use nalgebra as na;
use rayon::prelude::*;

use crate::config::RasterConfig;
use crate::gaussian::GaussianModel;
use crate::RenderError;

/// Spherical Harmonics constants (matching WGSL shader)
const SH_C0: f32 = 0.282_094_8;
const SH_C1: f32 = 0.488_602_52;
const SH_C2_0: f32 = 1.092_548_5;
const SH_C2_1: f32 = -1.092_548_5;
const SH_C2_2: f32 = 0.315_391_57;
const SH_C2_3: f32 = -1.092_548_5;
const SH_C2_4: f32 = 0.546_274_24;
const SH_C3_0: f32 = -0.590_043_6;
const SH_C3_1: f32 = 2.890_611_4;
const SH_C3_2: f32 = -0.457_045_8;
const SH_C3_3: f32 = 0.373_176_34;
const SH_C3_4: f32 = -0.457_045_8;
const SH_C3_5: f32 = 1.445_305_7;
const SH_C3_6: f32 = -0.590_043_6;

/// Camera parameters for CPU rasterization.
#[derive(Debug, Clone)]
pub struct CpuCamera {
    /// View matrix (world to camera).
    pub view: na::Matrix4<f32>,
    /// Projection matrix.
    pub proj: na::Matrix4<f32>,
    /// Camera position in world space.
    pub position: na::Vector3<f32>,
    /// Focal lengths (fx, fy) in pixels.
    pub focal: na::Vector2<f32>,
}

/// Projected Gaussian in 2D screen space.
///
/// Deliberately carries no source-Gaussian index. One was stored here and read
/// by nobody, held alive by `#[allow(dead_code)]`; the ordering it looked like
/// it existed for is supplied by *position* in the projected vector instead —
/// see the depth sort in [`CpuRasterizer::render`], which relies on these being
/// pushed in ascending model index.
#[derive(Debug, Clone)]
struct ProjectedGaussian {
    /// 2D mean (pixel coordinates).
    mean2d: na::Vector2<f32>,
    /// 2D conic matrix (inverse of covariance): [a, b, c] for [[a, b], [b, c]].
    conic: na::Vector3<f32>,
    /// RGB color (evaluated from SH).
    color: na::Vector3<f32>,
    /// Opacity (after sigmoid).
    opacity: f32,
    /// Depth (distance from camera).
    depth: f32,
}

/// CPU reference rasterizer output.
#[derive(Debug, Clone)]
pub struct CpuRenderOutput {
    /// RGBA color data [H * W * 4] as f32.
    pub color_data: Vec<f32>,
    /// Depth data [H * W].
    pub depth_data: Vec<f32>,
    /// Image width.
    pub width: u32,
    /// Image height.
    pub height: u32,
}

/// Pure-CPU rasterizer for gradient verification.
pub struct CpuRasterizer {
    config: RasterConfig,
}

impl CpuRasterizer {
    /// Create a new CPU rasterizer.
    pub fn new(config: RasterConfig) -> Self {
        Self { config }
    }

    /// Render Gaussians to an image using CPU.
    ///
    /// This implements the same algorithm as the GPU rasterizer:
    /// 1. Project 3D Gaussians to 2D screen space
    /// 2. Sort by depth (painter's algorithm)
    /// 3. For each pixel, blend Gaussians front-to-back
    /// 4. Accumulate RGB with alpha blending
    pub fn render(
        &self,
        model: &GaussianModel,
        camera: &CpuCamera,
    ) -> Result<CpuRenderOutput, RenderError> {
        let width = self.config.image_width;
        let height = self.config.image_height;
        let num_pixels = (width * height) as usize;

        // Step 1: Project all Gaussians to 2D
        let mut projected = Vec::new();
        for i in 0..model.len() {
            if let Some(g) = self.project_gaussian(model, i, camera)? {
                projected.push(g);
            }
        }

        // Step 2: Sort by depth (front-to-back).
        //
        // GPU parity for equal depths: `shaders/tile_assign.wgsl` builds the
        // key as `(tile_id, bitcast<u32>(depth))` with no index component, so
        // two Gaussians at exactly the same depth in the same tile are ordered
        // purely by the stability of the radix sort — i.e. by the order their
        // pairs were emitted, which is ascending Gaussian index. `sort_by` is
        // likewise a stable sort and `projected` is filled by the ascending
        // `0..model.len()` loop above, so ties break by ascending model index
        // on both sides. Do not switch this to `sort_unstable_by`: that would
        // silently diverge from the GPU exactly on the coincident-depth case
        // the CPU path exists to cross-check.
        projected.sort_by(|a, b| {
            a.depth
                .partial_cmp(&b.depth)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Step 3: Rasterize per-pixel (parallel)
        let pixel_results: Vec<([f32; 4], f32)> = (0..num_pixels)
            .into_par_iter()
            .map(|pixel_idx| {
                let px = (pixel_idx % width as usize) as f32;
                let py = (pixel_idx / width as usize) as f32;
                self.blend_pixel(px, py, &projected)
            })
            .collect();

        // Convert to output format
        let mut color_data = vec![0.0; num_pixels * 4];
        let mut depth_data = vec![0.0; num_pixels];

        for (i, (pixel_color, pixel_depth)) in pixel_results.iter().enumerate() {
            color_data[i * 4] = pixel_color[0];
            color_data[i * 4 + 1] = pixel_color[1];
            color_data[i * 4 + 2] = pixel_color[2];
            color_data[i * 4 + 3] = pixel_color[3];
            depth_data[i] = *pixel_depth;
        }

        Ok(CpuRenderOutput {
            color_data,
            depth_data,
            width,
            height,
        })
    }

    /// Project a single Gaussian from 3D to 2D.
    ///
    /// Returns `None` if Gaussian is culled (behind camera, off-screen, etc.).
    fn project_gaussian(
        &self,
        model: &GaussianModel,
        idx: usize,
        camera: &CpuCamera,
    ) -> Result<Option<ProjectedGaussian>, RenderError> {
        let gaussian = &model.gaussians[idx];

        // Get position
        let pos = na::Vector3::new(
            gaussian.position[0],
            gaussian.position[1],
            gaussian.position[2],
        );

        // Transform to camera space
        let pos_cam = camera.view.transform_point(&na::Point3::from(pos));

        // Cull against near/far planes, mirroring shaders/preprocess.wgsl
        // exactly (`let depth = -p_view.z; if depth <= near_plane || depth
        // >= far_plane { cull }`), including its boundary strictness — a
        // Gaussian sitting exactly on a clip plane is culled just like on
        // the GPU, and Gaussians beyond `far_plane` are dropped here too
        // (previously only the near plane was checked).
        let depth = -pos_cam.z;
        if depth <= self.config.near_plane || depth >= self.config.far_plane {
            return Ok(None);
        }

        // Project to screen space
        let pos_ndc = camera.proj.transform_point(&pos_cam);

        // Convert to pixel coordinates
        let mean2d = na::Vector2::new(
            (pos_ndc.x * 0.5 + 0.5) * self.config.image_width as f32,
            (pos_ndc.y * 0.5 + 0.5) * self.config.image_height as f32,
        );

        // Build rotation matrix directly from raw quaternion (matching GPU's quat_to_mat
        // which does NOT normalize the quaternion)
        let qx = gaussian.rotation[0];
        let qy = gaussian.rotation[1];
        let qz = gaussian.rotation[2];
        let qw = gaussian.rotation[3];
        let x2 = 2.0 * qx;
        let y2 = 2.0 * qy;
        let z2 = 2.0 * qz;
        let xx = qx * x2;
        let xy = qx * y2;
        let xz = qx * z2;
        let yy = qy * y2;
        let yz = qy * z2;
        let zz = qz * z2;
        let wx = qw * x2;
        let wy = qw * y2;
        let wz = qw * z2;
        let rotation_matrix = na::Matrix3::new(
            1.0 - (yy + zz),
            xy - wz,
            xz + wy,
            xy + wz,
            1.0 - (xx + zz),
            yz - wx,
            xz - wy,
            yz + wx,
            1.0 - (xx + yy),
        );

        let scale = na::Vector3::new(
            gaussian.scale[0].exp(),
            gaussian.scale[1].exp(),
            gaussian.scale[2].exp(),
        );

        // Compute 3D covariance matrix
        let s = na::Matrix3::from_diagonal(&scale);
        let m = rotation_matrix * s;
        let cov3d = m * m.transpose();

        // Project covariance to 2D (using Jacobian of projection)
        let focal = camera.focal;
        let z = pos_cam.z.abs();
        let z2 = z * z;

        // Jacobian of perspective projection
        let j = na::Matrix2x3::new(
            focal.x / z,
            0.0,
            focal.x * pos_cam.x / z2,
            0.0,
            focal.y / z,
            focal.y * pos_cam.y / z2,
        );

        // View matrix rotation part
        let w = camera.view.fixed_view::<3, 3>(0, 0);

        // Transform covariance: cov2d = J * W * cov3d * W^T * J^T
        let t = w * cov3d;
        let cov2d_full = j * t * w.transpose() * j.transpose();

        // Add small constant for numerical stability
        let cov2d = na::Matrix2::new(
            cov2d_full[(0, 0)] + 0.3,
            cov2d_full[(0, 1)],
            cov2d_full[(1, 0)],
            cov2d_full[(1, 1)] + 0.3,
        );

        // Compute conic (inverse of covariance)
        let det = cov2d[(0, 0)] * cov2d[(1, 1)] - cov2d[(0, 1)] * cov2d[(1, 0)];
        if det.abs() < 1e-10 {
            return Ok(None); // Singular covariance
        }

        let conic = na::Vector3::new(
            cov2d[(1, 1)] / det,
            -cov2d[(0, 1)] / det,
            cov2d[(0, 0)] / det,
        );

        // Evaluate Spherical Harmonics for color
        let view_dir = (pos - camera.position).normalize();
        let color = self.eval_sh(model, idx, view_dir)?;

        // Get opacity (apply sigmoid)
        let opacity = sigmoid(gaussian.opacity);

        Ok(Some(ProjectedGaussian {
            mean2d,
            conic,
            color,
            opacity,
            depth,
        }))
    }

    /// Evaluate Spherical Harmonics for a Gaussian.
    fn eval_sh(
        &self,
        model: &GaussianModel,
        idx: usize,
        dir: na::Vector3<f32>,
    ) -> Result<na::Vector3<f32>, RenderError> {
        // `GaussianModel::sh_degree` is an unvalidated plain `u32`. Silently
        // clamping it to 3 while indexing with the *clamped* value would
        // compute a stride (`sh_coeffs_per_gaussian`) that is smaller than
        // the data's actual per-Gaussian layout whenever `sh_degree > 3`,
        // so every Gaussian after the first would read coefficients
        // belonging to a different Gaussian. Reject it outright instead.
        if model.sh_degree > 3 {
            return Err(RenderError::Rasterize(format!(
                "unsupported SH degree {}: only degrees 0-3 are implemented",
                model.sh_degree
            )));
        }
        let sh_degree = model.sh_degree;
        let sh_coeffs_per_gaussian = ((sh_degree + 1) * (sh_degree + 1) * 3) as usize;
        let sh_start = idx * sh_coeffs_per_gaussian;

        if sh_start + sh_coeffs_per_gaussian > model.sh_coeffs.len() {
            return Err(RenderError::Rasterize(format!(
                "SH coefficient index out of bounds: {} + {} > {}",
                sh_start,
                sh_coeffs_per_gaussian,
                model.sh_coeffs.len()
            )));
        }

        let sh = &model.sh_coeffs[sh_start..sh_start + sh_coeffs_per_gaussian];

        let result = match sh_degree {
            0 => eval_sh_degree0(sh),
            1 => eval_sh_degree1(dir, sh),
            2 => eval_sh_degree2(dir, sh),
            3 => eval_sh_degree3(dir, sh),
            _ => {
                return Err(RenderError::Rasterize(format!(
                    "Invalid SH degree: {}",
                    sh_degree
                )))
            }
        };

        Ok(result)
    }

    /// Blend Gaussians for a single pixel (front-to-back alpha compositing).
    ///
    /// Returns `([r, g, b, alpha], depth)`. `depth` is the alpha-weighted
    /// accumulated depth, mirroring `shaders/rasterize_fwd.wgsl` exactly:
    /// the same per-Gaussian `weight = T * alpha` used for color is also
    /// applied to `g.depth` inside the same loop
    /// (`depth_acc += weight * wg_depth[b]` on the GPU side), with no
    /// background contribution added to depth afterward — the GPU writes
    /// `out_depth[pixel_idx] = depth_acc` directly, unlike color which gets
    /// `color += T * background`.
    fn blend_pixel(&self, px: f32, py: f32, gaussians: &[ProjectedGaussian]) -> ([f32; 4], f32) {
        let pixel_f = na::Vector2::new(px + 0.5, py + 0.5);

        let mut t = 1.0f32; // transmittance
        let mut color = na::Vector3::zeros();
        let mut depth_acc = 0.0f32;

        for g in gaussians {
            if t < 1.0 / 255.0 {
                break; // Early termination
            }

            // Evaluate 2D Gaussian
            let d = pixel_f - g.mean2d;
            let power = -0.5
                * (g.conic.x * d.x * d.x + 2.0 * g.conic.y * d.x * d.y + g.conic.z * d.y * d.y);

            if !(-4.0..=0.0).contains(&power) {
                continue; // Outside valid range
            }

            let alpha_raw = g.opacity * power.exp();
            let alpha = alpha_raw.min(0.99);

            if alpha < 1.0 / 255.0 {
                continue; // Too transparent
            }

            let weight = t * alpha;
            color += weight * g.color;
            depth_acc += weight * g.depth;
            t *= 1.0 - alpha;
        }

        // Add background (color only — depth has no background term, as on
        // the GPU).
        let bg = na::Vector3::new(
            self.config.background[0],
            self.config.background[1],
            self.config.background[2],
        );
        color += t * bg;

        ([color.x, color.y, color.z, 1.0 - t], depth_acc)
    }
}

/// Sigmoid function.
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Evaluate SH degree 0 (DC term only).
fn eval_sh_degree0(sh: &[f32]) -> na::Vector3<f32> {
    let dc = na::Vector3::new(sh[0], sh[1], sh[2]);
    (SH_C0 * dc + na::Vector3::new(0.5, 0.5, 0.5)).sup(&na::Vector3::zeros())
}

/// Evaluate SH degree 1 (DC + linear).
fn eval_sh_degree1(dir: na::Vector3<f32>, sh: &[f32]) -> na::Vector3<f32> {
    let x = dir.x;
    let y = dir.y;
    let z = dir.z;

    // DC term
    let mut result = SH_C0 * na::Vector3::new(sh[0], sh[1], sh[2]);

    // Linear terms: coefficients [3..12]
    // Y_1^{-1} = -y, Y_1^0 = z, Y_1^1 = -x
    result += SH_C1 * (-y) * na::Vector3::new(sh[3], sh[4], sh[5]);
    result += SH_C1 * z * na::Vector3::new(sh[6], sh[7], sh[8]);
    result += SH_C1 * (-x) * na::Vector3::new(sh[9], sh[10], sh[11]);

    (result + na::Vector3::new(0.5, 0.5, 0.5)).sup(&na::Vector3::zeros())
}

/// Evaluate SH degree 2 (DC + linear + quadratic).
///
/// All terms are accumulated without intermediate clamping to match the GPU
/// shader, which evaluates every SH band in a single pass and only clamps the
/// final result.
fn eval_sh_degree2(dir: na::Vector3<f32>, sh: &[f32]) -> na::Vector3<f32> {
    let x = dir.x;
    let y = dir.y;
    let z = dir.z;

    // DC term (degree 0)
    let mut result = SH_C0 * na::Vector3::new(sh[0], sh[1], sh[2]);

    // Linear terms (degree 1): same as in eval_sh_degree1 but without clamping
    result += SH_C1 * (-y) * na::Vector3::new(sh[3], sh[4], sh[5]);
    result += SH_C1 * z * na::Vector3::new(sh[6], sh[7], sh[8]);
    result += SH_C1 * (-x) * na::Vector3::new(sh[9], sh[10], sh[11]);

    // Quadratic terms (degree 2)
    let xx = x * x;
    let yy = y * y;
    let zz = z * z;
    let xy = x * y;
    let yz = y * z;
    let xz = x * z;

    result += SH_C2_0 * xy * na::Vector3::new(sh[12], sh[13], sh[14]);
    result += SH_C2_1 * yz * na::Vector3::new(sh[15], sh[16], sh[17]);
    result += SH_C2_2 * (2.0 * zz - xx - yy) * na::Vector3::new(sh[18], sh[19], sh[20]);
    result += SH_C2_3 * xz * na::Vector3::new(sh[21], sh[22], sh[23]);
    result += SH_C2_4 * (xx - yy) * na::Vector3::new(sh[24], sh[25], sh[26]);

    // Apply clamp ONCE at the end (matches GPU behavior)
    (result + na::Vector3::new(0.5, 0.5, 0.5)).sup(&na::Vector3::zeros())
}

/// Evaluate SH degree 3 (DC + linear + quadratic + cubic).
///
/// All terms are accumulated without intermediate clamping to match the GPU
/// shader, which evaluates every SH band in a single pass and only clamps the
/// final result.
fn eval_sh_degree3(dir: na::Vector3<f32>, sh: &[f32]) -> na::Vector3<f32> {
    let x = dir.x;
    let y = dir.y;
    let z = dir.z;

    // DC term (degree 0)
    let mut result = SH_C0 * na::Vector3::new(sh[0], sh[1], sh[2]);

    // Linear terms (degree 1)
    result += SH_C1 * (-y) * na::Vector3::new(sh[3], sh[4], sh[5]);
    result += SH_C1 * z * na::Vector3::new(sh[6], sh[7], sh[8]);
    result += SH_C1 * (-x) * na::Vector3::new(sh[9], sh[10], sh[11]);

    // Quadratic terms (degree 2)
    let xx = x * x;
    let yy = y * y;
    let zz = z * z;
    let xy = x * y;
    let yz = y * z;
    let xz = x * z;

    result += SH_C2_0 * xy * na::Vector3::new(sh[12], sh[13], sh[14]);
    result += SH_C2_1 * yz * na::Vector3::new(sh[15], sh[16], sh[17]);
    result += SH_C2_2 * (2.0 * zz - xx - yy) * na::Vector3::new(sh[18], sh[19], sh[20]);
    result += SH_C2_3 * xz * na::Vector3::new(sh[21], sh[22], sh[23]);
    result += SH_C2_4 * (xx - yy) * na::Vector3::new(sh[24], sh[25], sh[26]);

    // Cubic terms (degree 3)
    result += SH_C3_0 * y * (3.0 * xx - yy) * na::Vector3::new(sh[27], sh[28], sh[29]);
    result += SH_C3_1 * x * y * z * na::Vector3::new(sh[30], sh[31], sh[32]);
    result += SH_C3_2 * y * (4.0 * zz - xx - yy) * na::Vector3::new(sh[33], sh[34], sh[35]);
    result +=
        SH_C3_3 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * na::Vector3::new(sh[36], sh[37], sh[38]);
    result += SH_C3_4 * x * (4.0 * zz - xx - yy) * na::Vector3::new(sh[39], sh[40], sh[41]);
    result += SH_C3_5 * z * (xx - yy) * na::Vector3::new(sh[42], sh[43], sh[44]);
    result += SH_C3_6 * x * (xx - 3.0 * yy) * na::Vector3::new(sh[45], sh[46], sh[47]);

    // Apply clamp ONCE at the end (matches GPU behavior)
    (result + na::Vector3::new(0.5, 0.5, 0.5)).sup(&na::Vector3::zeros())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(10.0) > 0.99);
        assert!(sigmoid(-10.0) < 0.01);
    }

    #[test]
    fn test_sh_degree0() {
        let sh = vec![1.0, 0.5, 0.25];
        let result = eval_sh_degree0(&sh);

        // Expected: SH_C0 * [1.0, 0.5, 0.25] + 0.5
        let expected = na::Vector3::new(SH_C0 * 1.0 + 0.5, SH_C0 * 0.5 + 0.5, SH_C0 * 0.25 + 0.5);

        assert!((result - expected).norm() < 1e-5);
    }

    #[test]
    fn test_cpu_rasterizer_creation() {
        let config = RasterConfig::new().with_resolution(512, 512);
        let _rasterizer = CpuRasterizer::new(config);
    }

    #[test]
    fn test_blend_pixel_empty() {
        let config = RasterConfig::new().with_resolution(512, 512);
        let rasterizer = CpuRasterizer::new(config);

        let gaussians = vec![];
        let (pixel, depth) = rasterizer.blend_pixel(0.0, 0.0, &gaussians);

        // Should return background color (black by default) and zero
        // accumulated depth (no Gaussians contributed).
        assert_eq!(pixel, [0.0, 0.0, 0.0, 0.0]);
        assert_eq!(depth, 0.0);
    }

    #[test]
    fn test_blend_pixel_depth_matches_weighted_sum() {
        // Two exactly-overlapping, fully-opaque (post-clamp) Gaussians at
        // the pixel center: depth_acc must equal sum(weight_i * depth_i)
        // with weight_i = T_i * alpha_i, exactly mirroring
        // shaders/rasterize_fwd.wgsl's `depth_acc += weight * wg_depth[b]`.
        let config = RasterConfig::new().with_resolution(64, 64);
        let rasterizer = CpuRasterizer::new(config);

        let near = ProjectedGaussian {
            mean2d: na::Vector2::new(10.5, 10.5),
            conic: na::Vector3::new(10.0, 0.0, 10.0),
            color: na::Vector3::new(1.0, 0.0, 0.0),
            opacity: 1.0,
            depth: 1.0,
        };
        let far = ProjectedGaussian {
            mean2d: na::Vector2::new(10.5, 10.5),
            conic: na::Vector3::new(10.0, 0.0, 10.0),
            color: na::Vector3::new(0.0, 1.0, 0.0),
            opacity: 1.0,
            depth: 10.0,
        };

        let (_pixel, depth) = rasterizer.blend_pixel(10.0, 10.0, &[near, far]);

        // Both Gaussians sit dead-center (power = 0) with opacity clamped
        // to 0.99: weight_near = 1.0 * 0.99, weight_far = (1.0 - 0.99) * 0.99.
        let expected = 0.99 * 1.0 + (0.01 * 0.99) * 10.0;
        assert!(
            (depth - expected).abs() < 1e-4,
            "expected depth={expected}, got {depth}"
        );
        // The nearer Gaussian dominates because its weight (0.99) far
        // exceeds the farther one's residual-transmittance weight (0.0099).
        let near_contribution = 0.99 * 1.0;
        assert!(
            depth < near_contribution * 1.2,
            "nearer Gaussian should dominate accumulated depth: depth={depth}"
        );
    }

    #[test]
    fn test_blend_pixel_depth_zero_when_pixel_uncovered() {
        // A Gaussian far from the queried pixel contributes ~0 weight, so
        // depth should stay at (approximately) zero rather than picking up
        // a stray, uncontributed value.
        let config = RasterConfig::new().with_resolution(64, 64);
        let rasterizer = CpuRasterizer::new(config);

        let g = ProjectedGaussian {
            mean2d: na::Vector2::new(1000.5, 1000.5), // far outside the pixel
            conic: na::Vector3::new(10.0, 0.0, 10.0),
            color: na::Vector3::new(1.0, 0.0, 0.0),
            opacity: 1.0,
            depth: 42.0,
        };

        let (_pixel, depth) = rasterizer.blend_pixel(10.0, 10.0, &[g]);
        assert_eq!(depth, 0.0, "uncovered pixel should have zero depth");
    }

    #[test]
    fn test_eval_sh_rejects_degree_above_three() {
        // A model claiming sh_degree > 3 must error out rather than
        // silently clamp to 3 and read coefficients at the wrong stride.
        let config = RasterConfig::new().with_resolution(4, 4);
        let rasterizer = CpuRasterizer::new(config);
        let model = GaussianModel {
            gaussians: Vec::new(),
            // Give it *some* coefficients so the failure is unambiguously
            // about the degree check, not an out-of-bounds slice: degree-4
            // stride would be (4+1)^2*3 = 75.
            sh_coeffs: vec![0.0f32; 75],
            sh_degree: 4,
            face_indices: Vec::new(),
            barycentric: Vec::new(),
            local_offsets: Vec::new(),
            is_rigid: Vec::new(),
        };
        let dir = na::Vector3::new(0.0, 0.0, 1.0);
        let result = rasterizer.eval_sh(&model, 0, dir);
        assert!(result.is_err(), "sh_degree > 3 must be rejected");
    }
}
