//! Batch multi-view rendering of 3D Gaussian Splatting models.
//!
//! This module provides [`MultiViewRenderer`], which renders the same
//! [`GaussianModel`] from multiple camera positions efficiently by reusing
//! the GPU-uploaded Gaussian data across all views.
//!
//! # Design Rationale
//!
//! Within a single [`MultiViewRenderer::render_views`] /
//! [`MultiViewRenderer::render_views_stacked`] call, the Gaussian data is
//! uploaded once and every camera in the slice reuses it — only the
//! per-frame uniform update (camera matrices) plus the rasterization
//! dispatch repeat per camera.
//!
//! Across *separate* calls, `MultiViewRenderer` also keeps a snapshot of
//! the most recently uploaded model and skips the re-upload (and the GPU
//! buffer set it reallocates) when the incoming model compares equal to
//! it, so re-rendering the same static scene from a new batch of cameras
//! does not repeatedly reallocate `GaussianBuffers`, `IntermediateBuffers`,
//! `OutputBuffers` and `GradientBuffers`. Passing a genuinely different
//! model (by content) always triggers a fresh upload.
//!
//! Future work: true parallel multi-view could issue all N rasterization
//! dispatches into a single command encoder, but sequential dispatch is
//! already GPU-pipelined and handles the practical training/inference case.
//!
//! # Example
//!
//! ```rust,no_run
//! # use oxigaf_render::{MultiViewRenderer, MultiViewConfig, RenderCamera, RasterConfig};
//! # use oxigaf_render::gaussian::GaussianModel;
//! # async fn example() -> Result<(), oxigaf_render::RenderError> {
//! let config = MultiViewConfig {
//!     width: 512,
//!     height: 512,
//!     background: [0.0, 0.0, 0.0],
//!     ..Default::default()
//! };
//! let mut renderer = MultiViewRenderer::new(config).await?;
//! let model: GaussianModel = todo!("load model");
//! let images = renderer.render_turntable(&model, 8, 2.0, 15.0, [0.0, 0.0, 0.0], 60.0)?;
//! # Ok(())
//! # }
//! ```

use std::f32::consts::TAU;

use crate::config::RasterConfig;
use crate::gaussian::{GaussianAttributes, GaussianModel};
use crate::rasterizer::{Rasterizer, RenderCamera};
use crate::RenderError;

/// Configuration for multi-view batch rendering.
///
/// Used to construct a [`MultiViewRenderer`]. Image dimensions and background
/// colour are shared across all views in a batch — cameras supply only the
/// extrinsic/intrinsic transform matrices.
#[derive(Debug, Clone)]
pub struct MultiViewConfig {
    /// Output image width in pixels. Must be > 0.
    pub width: u32,
    /// Output image height in pixels. Must be > 0.
    pub height: u32,
    /// Background colour (linear RGB). Default black.
    pub background: [f32; 3],
    /// Spherical harmonics degree used during rendering (0–3). Default 3.
    pub sh_degree: u32,
    /// Near clipping plane. Default 0.01.
    pub near_plane: f32,
    /// Far clipping plane. Default 100.0.
    pub far_plane: f32,
}

impl Default for MultiViewConfig {
    fn default() -> Self {
        Self {
            width: 512,
            height: 512,
            background: [0.0, 0.0, 0.0],
            sh_degree: 3,
            near_plane: 0.01,
            far_plane: 100.0,
        }
    }
}

impl MultiViewConfig {
    /// Validate that the configuration is well-formed.
    ///
    /// Returns `Err` if `width` or `height` is zero.
    pub fn validate(&self) -> Result<(), RenderError> {
        if self.width == 0 {
            return Err(RenderError::Rasterize(
                "MultiViewConfig: width must be > 0".into(),
            ));
        }
        if self.height == 0 {
            return Err(RenderError::Rasterize(
                "MultiViewConfig: height must be > 0".into(),
            ));
        }
        if self.sh_degree > 3 {
            return Err(RenderError::Rasterize(format!(
                "MultiViewConfig: sh_degree {} > 3 is not supported",
                self.sh_degree
            )));
        }
        Ok(())
    }

    /// Convert to the underlying [`RasterConfig`].
    fn to_raster_config(&self) -> RasterConfig {
        RasterConfig::new()
            .with_resolution(self.width, self.height)
            .with_sh_degree(self.sh_degree)
            .with_background(self.background)
    }
}

/// Renders the same [`GaussianModel`] from multiple camera positions.
///
/// The internal [`Rasterizer`] lazily uploads Gaussian data to the GPU the
/// first time [`render_views`] is called for a given model, and caches the
/// GPU buffers across cameras in the same batch.
///
/// # Multi-view API
///
/// - [`render_views`]: returns one `Vec<u8>` (RGBA u8) per camera.
/// - [`render_views_stacked`]: returns all images as a flat `[N, H, W, 4]` u8 buffer.
/// - [`render_turntable`]: convenience wrapper generating evenly-spaced orbital cameras.
///
/// [`render_views`]: MultiViewRenderer::render_views
/// [`render_views_stacked`]: MultiViewRenderer::render_views_stacked
/// [`render_turntable`]: MultiViewRenderer::render_turntable
pub struct MultiViewRenderer {
    rasterizer: Rasterizer,
    config: MultiViewConfig,
    /// Snapshot of the most recently uploaded model, used to skip a
    /// redundant `upload_gaussians` GPU re-allocation when the same model
    /// (by full data equality) is rendered again — see the module docs.
    /// This trades memory (one extra full `GaussianModel` clone, held for
    /// the renderer's lifetime) for avoiding repeated GPU buffer
    /// reallocation; a model containing NaN coefficients always compares
    /// unequal to itself (`f32` `PartialEq`), which is the safe direction —
    /// it just means such a model always re-uploads.
    uploaded_model: Option<GaussianModel>,
}

impl MultiViewRenderer {
    /// Create a new `MultiViewRenderer` by requesting a GPU device.
    ///
    /// This is `async` because wgpu device creation is asynchronous.
    pub async fn new(config: MultiViewConfig) -> Result<Self, RenderError> {
        config.validate()?;
        let raster_config = config.to_raster_config();
        let rasterizer = Rasterizer::new(raster_config).await?;
        Ok(Self {
            rasterizer,
            config,
            uploaded_model: None,
        })
    }

    /// Create a `MultiViewRenderer` from an already-initialised [`Rasterizer`].
    ///
    /// Use this when you want to share GPU device setup with other code.
    /// The `Rasterizer`'s resolution must match `config.width` / `config.height`.
    pub fn from_rasterizer(
        rasterizer: Rasterizer,
        config: MultiViewConfig,
    ) -> Result<Self, RenderError> {
        config.validate()?;
        Ok(Self {
            rasterizer,
            config,
            uploaded_model: None,
        })
    }

    /// Upload `model` to the GPU unless it is identical (by full data
    /// equality) to the model already resident from a previous call.
    fn ensure_uploaded(&mut self, model: &GaussianModel) {
        let already_current = match &self.uploaded_model {
            Some(cached) => gaussian_model_data_eq(cached, model),
            None => false,
        };
        if already_current {
            return;
        }
        self.rasterizer.upload_gaussians(model);
        self.uploaded_model = Some(model.clone());
    }

    /// Render the model from each camera in `cameras`.
    ///
    /// Returns a `Vec` of RGBA images, one per camera, each as `Vec<u8>`
    /// with length `width * height * 4` in row-major order.
    ///
    /// The Gaussian model is uploaded to the GPU lazily (once per unique
    /// model, by full data equality — see the module docs). All cameras in
    /// the slice share that single GPU upload.
    ///
    /// # Errors
    ///
    /// Propagates any [`RenderError`] from the underlying rasterizer.
    pub fn render_views(
        &mut self,
        model: &GaussianModel,
        cameras: &[RenderCamera],
    ) -> Result<Vec<Vec<u8>>, RenderError> {
        // Fast-path: no cameras → empty result.
        if cameras.is_empty() {
            return Ok(Vec::new());
        }

        self.ensure_uploaded(model);

        let mut results = Vec::with_capacity(cameras.len());
        for camera in cameras {
            let output = self.rasterizer.forward(model, camera)?;
            let rgba_u8 = f32_rgba_to_u8(&output.color_data);
            results.push(rgba_u8);
        }
        Ok(results)
    }

    /// Render N views and return them as a single stacked buffer: `[N, H, W, 4]` as flat `Vec<u8>`.
    ///
    /// This is equivalent to `render_views` followed by `concat`, but avoids
    /// an extra allocation by pre-sizing the output buffer.
    ///
    /// # Errors
    ///
    /// Propagates any [`RenderError`] from the underlying rasterizer.
    pub fn render_views_stacked(
        &mut self,
        model: &GaussianModel,
        cameras: &[RenderCamera],
    ) -> Result<Vec<u8>, RenderError> {
        let n = cameras.len();
        let frame_bytes = (self.config.width * self.config.height * 4) as usize;
        let total_bytes = n * frame_bytes;

        let mut out = Vec::with_capacity(total_bytes);

        // Upload once before iterating (skipped if `model` is unchanged
        // from a previous call — see `ensure_uploaded`).
        if !cameras.is_empty() {
            self.ensure_uploaded(model);
        }

        for camera in cameras {
            let render_output = self.rasterizer.forward(model, camera)?;
            let rgba_u8 = f32_rgba_to_u8(&render_output.color_data);
            out.extend_from_slice(&rgba_u8);
        }

        Ok(out)
    }

    /// Convenience: render from evenly-spaced turntable (orbital) positions.
    ///
    /// Creates `n_views` cameras evenly distributed on a horizontal circle of
    /// the given `radius` around `target`, at `elevation_deg` degrees above the
    /// horizontal plane. All cameras look toward `target` with a perspective
    /// projection parameterised by `fov_deg` (vertical field-of-view).
    ///
    /// # Parameters
    ///
    /// - `model`: Gaussian scene to render.
    /// - `n_views`: Number of equally-spaced views (must be ≥ 1).
    /// - `radius`: Orbital radius in world units.
    /// - `elevation_deg`: Camera elevation above the XZ-plane, in degrees.
    /// - `target`: Look-at target point in world space.
    /// - `fov_deg`: Vertical field-of-view in degrees.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::Rasterize`] if `n_views == 0`.
    /// Propagates GPU errors from the underlying rasterizer.
    pub fn render_turntable(
        &mut self,
        model: &GaussianModel,
        n_views: usize,
        radius: f32,
        elevation_deg: f32,
        target: [f32; 3],
        fov_deg: f32,
    ) -> Result<Vec<Vec<u8>>, RenderError> {
        if n_views == 0 {
            return Err(RenderError::Rasterize(
                "render_turntable: n_views must be >= 1".into(),
            ));
        }

        let cameras = build_turntable_cameras(
            n_views,
            radius,
            elevation_deg,
            target,
            fov_deg,
            self.config.width,
            self.config.height,
        )?;

        self.render_views(model, &cameras)
    }

    /// Width configured for this renderer (pixels).
    pub fn width(&self) -> u32 {
        self.config.width
    }

    /// Height configured for this renderer (pixels).
    pub fn height(&self) -> u32 {
        self.config.height
    }

    /// Access the underlying configuration.
    pub fn config(&self) -> &MultiViewConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Returns `true` when `a` and `b` contain identical Gaussian data — the
/// full set of fields that influence what gets uploaded to the GPU by
/// [`Rasterizer::upload_gaussians`]. Used to decide whether a re-upload can
/// be skipped (see [`MultiViewRenderer::ensure_uploaded`]).
fn gaussian_model_data_eq(a: &GaussianModel, b: &GaussianModel) -> bool {
    a.sh_degree == b.sh_degree
        && a.face_indices == b.face_indices
        && a.barycentric == b.barycentric
        && a.local_offsets == b.local_offsets
        && a.is_rigid == b.is_rigid
        && a.sh_coeffs == b.sh_coeffs
        && bytemuck::cast_slice::<GaussianAttributes, u8>(&a.gaussians)
            == bytemuck::cast_slice::<GaussianAttributes, u8>(&b.gaussians)
}

/// Convert a slice of linear-f32 RGBA values to `u8` RGBA (clamped, gamma-encoded as sRGB).
///
/// The GPU rasterizer stores colour in **linear** float format. We apply a
/// simple gamma-2.2 approximation (raise to `1/2.2`) before converting to u8
/// so the images look perceptually correct when saved as PNG/JPEG.
///
/// Values outside `[0, 1]` are clamped before encoding.
pub(crate) fn f32_rgba_to_u8(f32_data: &[f32]) -> Vec<u8> {
    f32_data
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let clamped = v.clamp(0.0, 1.0);
            // Apply sRGB gamma only to the RGB channels (indices 0,1,2 of each
            // 4-tuple), leave alpha (index 3) linear.
            let encoded = if i % 4 == 3 {
                // Alpha: keep linear.
                clamped
            } else {
                // RGB: linearise → sRGB (approximate with gamma 2.2).
                clamped.powf(1.0 / 2.2)
            };
            (encoded * 255.0 + 0.5) as u8
        })
        .collect()
}

/// Build `n_views` orbital [`RenderCamera`] instances evenly distributed on a
/// horizontal circle.
///
/// Azimuth angles are spaced by `TAU / n_views`, starting from azimuth 0
/// (positive X-axis). The Y-axis is up.
pub(crate) fn build_turntable_cameras(
    n_views: usize,
    radius: f32,
    elevation_deg: f32,
    target: [f32; 3],
    fov_deg: f32,
    width: u32,
    height: u32,
) -> Result<Vec<RenderCamera>, RenderError> {
    if n_views == 0 {
        return Err(RenderError::Rasterize(
            "build_turntable_cameras: n_views must be >= 1".into(),
        ));
    }
    if radius <= 0.0 {
        return Err(RenderError::Rasterize(
            "build_turntable_cameras: radius must be positive".into(),
        ));
    }
    if fov_deg <= 0.0 || fov_deg >= 180.0 {
        return Err(RenderError::Rasterize(format!(
            "build_turntable_cameras: fov_deg {} is out of range (0, 180)",
            fov_deg
        )));
    }
    if width == 0 || height == 0 {
        return Err(RenderError::Rasterize(
            "build_turntable_cameras: width and height must be > 0".into(),
        ));
    }

    let elev_rad = elevation_deg.to_radians();
    let cos_e = elev_rad.cos();
    let sin_e = elev_rad.sin();

    let tgt = glam::Vec3::from(target);

    // Build a right-hand perspective matrix with vertical FoV.
    let aspect = width as f32 / height as f32;
    let fov_y_rad = fov_deg.to_radians();

    // For RenderCamera.focal we need (fx, fy) in pixels:
    //   fy = H / (2 * tan(fov_y/2))
    //   fx = fy * aspect  (for square pixels)
    let fy = (height as f32) / (2.0 * (fov_y_rad * 0.5).tan());
    let fx = fy * aspect;

    // Near/far planes — use sensible defaults; callers needing custom values
    // should build RenderCamera directly.
    let near = 0.01_f32;
    let far = 1000.0_f32;

    // glam projection (right-hand, depth 0→1, clip-space y-down, matching WGSL)
    let proj_mat = glam::camera::rh::proj::directx::perspective(fov_y_rad, aspect, near, far);
    let proj_matrix: [f32; 16] = proj_mat.to_cols_array();

    let mut cameras = Vec::with_capacity(n_views);

    for i in 0..n_views {
        let azimuth = TAU * (i as f32) / (n_views as f32);
        let cos_a = azimuth.cos();
        let sin_a = azimuth.sin();

        // Camera position on the sphere.
        let eye = tgt
            + glam::Vec3::new(
                radius * cos_e * cos_a,
                radius * sin_e,
                radius * cos_e * sin_a,
            );

        // "Up" vector: world Y, unless the camera is looking straight up/down.
        let up = if cos_e.abs() < 1e-4 {
            // Gimbal-lock guard: use Z as up when looking along Y.
            glam::Vec3::Z
        } else {
            glam::Vec3::Y
        };

        let view_mat = glam::camera::rh::view::look_at_mat4(eye, tgt, up);
        let view_matrix: [f32; 16] = view_mat.to_cols_array();

        cameras.push(RenderCamera {
            view_matrix,
            proj_matrix,
            position: eye.to_array(),
            focal: [fx, fy],
        });
    }

    Ok(cameras)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // MultiViewConfig validation
    // ------------------------------------------------------------------

    #[test]
    fn test_config_default() {
        let cfg = MultiViewConfig::default();
        assert_eq!(cfg.width, 512);
        assert_eq!(cfg.height, 512);
        assert_eq!(cfg.background, [0.0, 0.0, 0.0]);
        assert_eq!(cfg.sh_degree, 3);
    }

    #[test]
    fn test_config_validate_ok() {
        let cfg = MultiViewConfig {
            width: 256,
            height: 128,
            ..Default::default()
        };
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_validate_zero_width() {
        let cfg = MultiViewConfig {
            width: 0,
            height: 128,
            ..Default::default()
        };
        let err = cfg.validate();
        assert!(err.is_err());
        let msg = err.err().map(|e| e.to_string()).unwrap_or_default();
        assert!(msg.contains("width"), "expected 'width' in error: {msg}");
    }

    #[test]
    fn test_config_validate_zero_height() {
        let cfg = MultiViewConfig {
            width: 256,
            height: 0,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_validate_sh_degree_too_high() {
        let cfg = MultiViewConfig {
            sh_degree: 4,
            ..Default::default()
        };
        let err = cfg.validate();
        assert!(err.is_err());
        let msg = err.err().map(|e| e.to_string()).unwrap_or_default();
        assert!(
            msg.contains("sh_degree"),
            "expected 'sh_degree' in error: {msg}"
        );
    }

    #[test]
    fn test_config_to_raster_config() {
        let cfg = MultiViewConfig {
            width: 320,
            height: 240,
            background: [0.5, 0.5, 0.5],
            sh_degree: 1,
            ..Default::default()
        };
        let rc = cfg.to_raster_config();
        assert_eq!(rc.image_width, 320);
        assert_eq!(rc.image_height, 240);
        assert_eq!(rc.background, [0.5, 0.5, 0.5]);
        assert_eq!(rc.sh_degree, 1);
    }

    // ------------------------------------------------------------------
    // f32_rgba_to_u8 helper
    // ------------------------------------------------------------------

    #[test]
    fn test_f32_rgba_to_u8_black() {
        let data = vec![0.0f32; 4];
        let out = f32_rgba_to_u8(&data);
        assert_eq!(out.len(), 4);
        // All channels: 0.0 → 0
        assert_eq!(out, vec![0u8, 0, 0, 0]);
    }

    #[test]
    fn test_f32_rgba_to_u8_white() {
        let data = vec![1.0f32; 4];
        let out = f32_rgba_to_u8(&data);
        assert_eq!(out.len(), 4);
        // All channels: 1.0 → 255 (after gamma, still 1.0)
        assert_eq!(out[3], 255, "alpha should be 255");
        // RGB channels with gamma 2.2: 1.0^(1/2.2) = 1.0 → 255
        assert_eq!(out[0], 255);
        assert_eq!(out[1], 255);
        assert_eq!(out[2], 255);
    }

    #[test]
    fn test_f32_rgba_to_u8_alpha_linear() {
        // Alpha (index 3) should NOT have gamma applied.
        // 0.5 linear alpha → ~127, while 0.5 RGB with gamma 2.2 → ~186.
        let data = vec![0.5f32, 0.5, 0.5, 0.5];
        let out = f32_rgba_to_u8(&data);
        assert_eq!(out.len(), 4);
        // RGB with gamma: 0.5^(1/2.2) ≈ 0.7297 → ~186
        let rgb_expected = (0.5f32.powf(1.0 / 2.2) * 255.0 + 0.5) as u8;
        // Alpha linear: 0.5 * 255 + 0.5 = 128
        let alpha_expected = (0.5f32 * 255.0 + 0.5) as u8;
        assert_eq!(out[0], rgb_expected, "R channel gamma mismatch");
        assert_eq!(out[3], alpha_expected, "alpha should be linear");
        assert!(
            rgb_expected > alpha_expected,
            "gamma-encoded R should be brighter than linear alpha at 0.5"
        );
    }

    #[test]
    fn test_f32_rgba_to_u8_clamp() {
        // Values outside [0, 1] are clamped.
        let data = vec![-1.0f32, 2.0, 0.5, -0.5];
        let out = f32_rgba_to_u8(&data);
        assert_eq!(out.len(), 4);
        assert_eq!(out[0], 0, "negative should clamp to 0");
        assert_eq!(out[1], 255, "2.0 should clamp to 255");
    }

    #[test]
    fn test_f32_rgba_to_u8_length() {
        let n_pixels = 100usize;
        let data = vec![0.5f32; n_pixels * 4];
        let out = f32_rgba_to_u8(&data);
        assert_eq!(out.len(), n_pixels * 4);
    }

    // ------------------------------------------------------------------
    // Turntable camera generation
    // ------------------------------------------------------------------

    #[test]
    fn test_build_turntable_cameras_count() {
        let cams = build_turntable_cameras(8, 2.0, 0.0, [0.0, 0.0, 0.0], 60.0, 512, 512)
            .expect("should succeed");
        assert_eq!(cams.len(), 8);
    }

    #[test]
    fn test_build_turntable_cameras_radii() {
        // All cameras should be at the requested radius from the target.
        let radius = 3.0f32;
        let target = [1.0f32, 0.5, -0.5];
        let cams = build_turntable_cameras(6, radius, 0.0, target, 45.0, 256, 256)
            .expect("should succeed");
        for cam in &cams {
            let pos = glam::Vec3::from(cam.position);
            let tgt = glam::Vec3::from(target);
            let dist = (pos - tgt).length();
            assert!(
                (dist - radius).abs() < 1e-4,
                "camera distance {dist} != radius {radius}"
            );
        }
    }

    #[test]
    fn test_build_turntable_cameras_evenly_spaced() {
        // For N cameras at elevation 0, azimuth should be evenly spaced.
        // The cameras at elevation 0 lie on a horizontal circle; compare XZ angles.
        let n = 4usize;
        let target = [0.0f32, 0.0, 0.0];
        let cams =
            build_turntable_cameras(n, 1.0, 0.0, target, 60.0, 512, 512).expect("should succeed");
        assert_eq!(cams.len(), n);

        // Extract azimuth angles from camera positions.
        let azimuths: Vec<f32> = cams
            .iter()
            .map(|c| c.position[2].atan2(c.position[0]))
            .collect();

        // Expected: 0, π/2, π, 3π/2 (mod 2π)
        let expected_step = TAU / (n as f32);
        for i in 1..n {
            let delta = (azimuths[i] - azimuths[0] - expected_step * i as f32 + TAU * 10.0) % TAU;
            // Within a small floating-point tolerance
            assert!(
                delta.abs() < 1e-5 || (delta - TAU).abs() < 1e-5,
                "azimuth spacing wrong at view {i}: delta={delta}"
            );
        }
    }

    #[test]
    fn test_build_turntable_cameras_elevation() {
        // At elevation 90°, camera should be directly above the target.
        let target = [0.0f32, 0.0, 0.0];
        let radius = 5.0f32;
        let cams = build_turntable_cameras(4, radius, 90.0, target, 60.0, 256, 256)
            .expect("should succeed");
        for cam in &cams {
            let pos = glam::Vec3::from(cam.position);
            // At 90° elevation, XZ components should be near zero.
            let horizontal_dist = (pos.x * pos.x + pos.z * pos.z).sqrt();
            assert!(
                horizontal_dist < 1e-4,
                "at 90° elevation, horizontal dist {horizontal_dist} should be near 0"
            );
            // Y component should be near ±radius.
            assert!(
                (pos.y.abs() - radius).abs() < 1e-4,
                "at 90° elevation, |y|={} should equal radius {}",
                pos.y.abs(),
                radius
            );
        }
    }

    #[test]
    fn test_build_turntable_cameras_focal_from_fov() {
        // Verify focal lengths match expected formula.
        let width = 640u32;
        let height = 480u32;
        let fov_deg = 60.0f32;
        let fov_rad = fov_deg.to_radians();
        let fy_expected = (height as f32) / (2.0 * (fov_rad * 0.5).tan());
        let fx_expected = fy_expected * (width as f32 / height as f32);

        let cams = build_turntable_cameras(1, 2.0, 0.0, [0.0, 0.0, 0.0], fov_deg, width, height)
            .expect("should succeed");
        assert_eq!(cams.len(), 1);
        assert!(
            (cams[0].focal[0] - fx_expected).abs() < 0.1,
            "fx: got {}, expected {}",
            cams[0].focal[0],
            fx_expected
        );
        assert!(
            (cams[0].focal[1] - fy_expected).abs() < 0.1,
            "fy: got {}, expected {}",
            cams[0].focal[1],
            fy_expected
        );
    }

    #[test]
    fn test_build_turntable_cameras_zero_views_error() {
        let result = build_turntable_cameras(0, 1.0, 0.0, [0.0, 0.0, 0.0], 60.0, 512, 512);
        assert!(result.is_err());
    }

    #[test]
    fn test_build_turntable_cameras_negative_radius_error() {
        let result = build_turntable_cameras(4, -1.0, 0.0, [0.0, 0.0, 0.0], 60.0, 512, 512);
        assert!(result.is_err());
    }

    #[test]
    fn test_build_turntable_cameras_invalid_fov_error() {
        // fov = 0 is invalid
        let r1 = build_turntable_cameras(4, 1.0, 0.0, [0.0, 0.0, 0.0], 0.0, 512, 512);
        assert!(r1.is_err());
        // fov >= 180 is invalid
        let r2 = build_turntable_cameras(4, 1.0, 0.0, [0.0, 0.0, 0.0], 180.0, 512, 512);
        assert!(r2.is_err());
    }

    // ------------------------------------------------------------------
    // Stacked output size arithmetic
    // ------------------------------------------------------------------

    #[test]
    fn test_stacked_output_size_formula() {
        // N * H * W * 4 = expected byte count.
        let (n, w, h) = (7usize, 32u32, 24u32);
        let expected = n * (w as usize) * (h as usize) * 4;
        // Simulate stacking n frames of size w*h*4.
        let frame = vec![128u8; (w * h * 4) as usize];
        let stacked: Vec<u8> = std::iter::repeat_n(frame, n).flatten().collect();
        assert_eq!(stacked.len(), expected);
    }

    #[test]
    fn test_stacked_output_empty_cameras() {
        // Zero cameras → 0 bytes.
        let stacked: Vec<u8> = Vec::new();
        assert_eq!(stacked.len(), 0);
    }

    // ------------------------------------------------------------------
    // gaussian_model_data_eq (pure function, no GPU required)
    // ------------------------------------------------------------------

    #[test]
    fn test_gaussian_model_data_eq_identical_models() {
        let a = minimal_gaussian_model();
        let b = minimal_gaussian_model();
        assert!(gaussian_model_data_eq(&a, &b));
    }

    #[test]
    fn test_gaussian_model_data_eq_detects_position_change() {
        let a = minimal_gaussian_model();
        let mut b = minimal_gaussian_model();
        b.gaussians[0].position[0] += 0.001;
        assert!(!gaussian_model_data_eq(&a, &b));
    }

    #[test]
    fn test_gaussian_model_data_eq_detects_sh_coeffs_change() {
        let a = minimal_gaussian_model();
        let mut b = minimal_gaussian_model();
        b.sh_coeffs[0] += 0.001;
        assert!(!gaussian_model_data_eq(&a, &b));
    }

    #[test]
    fn test_gaussian_model_data_eq_detects_gaussian_count_change() {
        let a = minimal_gaussian_model();
        let mut b = minimal_gaussian_model();
        b.gaussians.push(b.gaussians[0]);
        b.face_indices.push(0);
        b.barycentric.push([1.0 / 3.0; 3]);
        b.local_offsets.push([0.0; 3]);
        b.is_rigid.push(true);
        b.sh_coeffs.extend_from_slice(&[0.0; 3]);
        assert!(!gaussian_model_data_eq(&a, &b));
    }

    // ------------------------------------------------------------------
    // GPU integration tests (require real GPU, marked #[ignore])
    // ------------------------------------------------------------------

    /// Smoke-test: single camera renders without panic.
    #[test]
    #[ignore = "requires GPU"]
    fn test_gpu_single_view() {
        let rt = tokio_test_runtime();
        let config = MultiViewConfig {
            width: 64,
            height: 64,
            sh_degree: 0,
            ..Default::default()
        };
        let mut renderer = rt
            .block_on(MultiViewRenderer::new(config.clone()))
            .expect("renderer creation should succeed");

        let model = minimal_gaussian_model();
        let cam = build_turntable_cameras(1, 2.0, 0.0, [0.0, 0.0, 0.0], 60.0, 64, 64)
            .expect("camera build");

        let images = renderer.render_views(&model, &cam).expect("render");
        assert_eq!(images.len(), 1);
        assert_eq!(images[0].len(), 64 * 64 * 4);
    }

    /// Smoke-test: N cameras return N images.
    #[test]
    #[ignore = "requires GPU"]
    fn test_gpu_multi_view_count() {
        let rt = tokio_test_runtime();
        let config = MultiViewConfig {
            width: 32,
            height: 32,
            sh_degree: 0,
            ..Default::default()
        };
        let mut renderer = rt
            .block_on(MultiViewRenderer::new(config))
            .expect("renderer");
        let model = minimal_gaussian_model();
        let cameras =
            build_turntable_cameras(5, 2.0, 10.0, [0.0, 0.0, 0.0], 45.0, 32, 32).expect("cams");

        let images = renderer.render_views(&model, &cameras).expect("render");
        assert_eq!(images.len(), 5);
        for img in &images {
            assert_eq!(img.len(), 32 * 32 * 4);
        }
    }

    /// Empty cameras list returns empty Vec without touching GPU.
    #[test]
    #[ignore = "requires GPU"]
    fn test_gpu_empty_cameras() {
        let rt = tokio_test_runtime();
        let mut renderer = rt
            .block_on(MultiViewRenderer::new(MultiViewConfig::default()))
            .expect("renderer");
        let model = minimal_gaussian_model();
        let images = renderer.render_views(&model, &[]).expect("render");
        assert!(images.is_empty());
    }

    /// Stacked output has the right total byte count.
    #[test]
    #[ignore = "requires GPU"]
    fn test_gpu_stacked_length() {
        let rt = tokio_test_runtime();
        let (w, h) = (16u32, 16u32);
        let n = 3usize;
        let config = MultiViewConfig {
            width: w,
            height: h,
            sh_degree: 0,
            ..Default::default()
        };
        let mut renderer = rt
            .block_on(MultiViewRenderer::new(config))
            .expect("renderer");
        let model = minimal_gaussian_model();
        let cams = build_turntable_cameras(n, 2.0, 0.0, [0.0, 0.0, 0.0], 60.0, w, h).expect("cams");
        let stacked = renderer
            .render_views_stacked(&model, &cams)
            .expect("render");
        assert_eq!(stacked.len(), n * (w as usize) * (h as usize) * 4);
    }

    /// Turntable renders N images.
    #[test]
    #[ignore = "requires GPU"]
    fn test_gpu_turntable() {
        let rt = tokio_test_runtime();
        let config = MultiViewConfig {
            width: 32,
            height: 32,
            sh_degree: 0,
            ..Default::default()
        };
        let mut renderer = rt
            .block_on(MultiViewRenderer::new(config))
            .expect("renderer");
        let model = minimal_gaussian_model();
        let images = renderer
            .render_turntable(&model, 4, 2.0, 0.0, [0.0, 0.0, 0.0], 60.0)
            .expect("turntable render");
        assert_eq!(images.len(), 4);
    }

    // ------------------------------------------------------------------
    // Test utilities (GPU tests only)
    // ------------------------------------------------------------------

    #[cfg(test)]
    fn tokio_test_runtime() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("tokio runtime")
    }

    #[cfg(test)]
    fn minimal_gaussian_model() -> GaussianModel {
        use crate::gaussian::{GaussianAttributes, GaussianModel};
        let g = GaussianAttributes {
            position: [0.0, 0.0, 0.0],
            _pad0: 0.0,
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [0.0, 0.0, 0.0],
            opacity: 0.0,
        };
        GaussianModel {
            gaussians: vec![g],
            sh_coeffs: vec![0.0; 3], // sh_degree=0
            sh_degree: 0,
            face_indices: vec![0],
            barycentric: vec![[1.0 / 3.0; 3]],
            local_offsets: vec![[0.0; 3]],
            is_rigid: vec![true],
        }
    }
}
