//! Software point-cloud renderer with a configurable background and splat
//! footprint.
//!
//! Each Gaussian is splatted as a small filled disc at its projected pixel
//! position, coloured by its SH DC coefficient, with a Z-buffer for depth
//! ordering.  This is intentionally a lightweight preview renderer — the full
//! GPU rasteriser lives in `oxigaf-render`.
//!
//! # Why the options live here
//!
//! `oxigaf render` accepts `--background` and `--splat-radius`.  Both used to
//! be compile-time constants inside the renderer (`Rgb([30, 30, 30])` and
//! `let splat_radius: i32 = 2;`), so the flags could only be reported as
//! unimplemented by [`crate::commands::flag_warnings`].  [`PointCloudRenderOptions`]
//! is the parameter object that lets a caller pass those two values straight
//! through, and [`Background::parse`] turns the CLI's `--background` string
//! (`"ffffff"`, `"#1e1e1e"`, `"transparent"`, …) into a value the renderer can
//! honour.
//!
//! [`PointCloudRenderOptions::default()`] reproduces the previous hard-coded
//! behaviour byte for byte, so callers that do not care are unaffected.

use anyhow::{Context, Result};
use image::{Rgb, RgbImage, Rgba, RgbaImage};
use nalgebra as na;

use oxigaf::flame::Camera;
use oxigaf::render::gaussian::GaussianModel;

/// SH band-0 normalisation constant: `0.5 / Y₀⁰` where `Y₀⁰ = 0.5·√(1/π)`.
pub(crate) const SH_C0: f32 = 0.282_094_8;

/// The `--background` default declared by `oxigaf render`, as RGB bytes.
///
/// `0x1e1e1e` == `[30, 30, 30]`, which is exactly the clear colour the
/// renderer used to hard-code.
pub const DEFAULT_BACKGROUND_RGB: [u8; 3] = [0x1e, 0x1e, 0x1e];

/// The `--splat-radius` default declared by `oxigaf render`.
pub const DEFAULT_SPLAT_RADIUS: u32 = 2;

/// Smallest `--splat-radius` the CLI accepts.
pub const MIN_SPLAT_RADIUS: u32 = 1;

/// Largest `--splat-radius` the CLI accepts.
pub const MAX_SPLAT_RADIUS: u32 = 5;

// ---------------------------------------------------------------------------
// Background
// ---------------------------------------------------------------------------

/// The colour the frame is cleared to before any Gaussian is splatted.
///
/// `alpha` is only representable in the RGBA output of
/// [`render_point_cloud_rgba`]; [`render_point_cloud_with_options`] returns an
/// [`RgbImage`], where a transparent background is written as its plain `rgb`
/// value (black, for [`Background::TRANSPARENT`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Background {
    /// Fill colour, as straight (non-premultiplied) RGB bytes.
    pub rgb: [u8; 3],
    /// Fill alpha: `255` fully opaque, `0` fully transparent.
    pub alpha: u8,
}

impl Background {
    /// A fully transparent background (black with alpha 0).
    pub const TRANSPARENT: Self = Self {
        rgb: [0, 0, 0],
        alpha: 0,
    };

    /// An opaque background of the given RGB colour.
    #[must_use]
    pub const fn opaque(rgb: [u8; 3]) -> Self {
        Self { rgb, alpha: 255 }
    }

    /// Whether this background is fully transparent.
    #[must_use]
    pub const fn is_transparent(self) -> bool {
        self.alpha == 0
    }

    /// Parse a `--background` specification.
    ///
    /// Accepted forms (case-insensitive, surrounding whitespace ignored):
    ///
    /// | Form | Example | Meaning |
    /// |------|---------|---------|
    /// | `transparent` / `none` / `clear` | `transparent` | alpha 0 |
    /// | 3 hex digits | `fff`, `#fff` | RGB, each nibble doubled |
    /// | 4 hex digits | `#fff8` | RGBA, each nibble doubled |
    /// | 6 hex digits | `1e1e1e`, `0x1e1e1e` | RGB |
    /// | 8 hex digits | `1e1e1eff` | RGBA |
    ///
    /// # Errors
    ///
    /// Returns an error naming every accepted form when `spec` is neither a
    /// recognised keyword nor a hex string of a supported length.
    pub fn parse(spec: &str) -> Result<Self> {
        let trimmed = spec.trim();
        let lowered = trimmed.to_ascii_lowercase();

        match lowered.as_str() {
            "transparent" | "none" | "clear" => return Ok(Self::TRANSPARENT),
            "black" => return Ok(Self::opaque([0, 0, 0])),
            "white" => return Ok(Self::opaque([255, 255, 255])),
            _ => {}
        }

        let digits = lowered
            .strip_prefix('#')
            .or_else(|| lowered.strip_prefix("0x"))
            .unwrap_or(lowered.as_str());

        anyhow::ensure!(
            !digits.is_empty() && digits.chars().all(|c| c.is_ascii_hexdigit()),
            "invalid --background '{spec}': expected a hex colour (e.g. ffffff, #1e1e1e, \
             1e1e1eff) or one of transparent/none/clear/black/white"
        );

        // `from_str_radix` on an ASCII-hex slice cannot fail, but the error is
        // still propagated rather than unwrapped.
        let nibble = |i: usize| -> Result<u8> {
            let s = digits
                .get(i..=i)
                .with_context(|| format!("--background '{spec}' is shorter than expected"))?;
            u8::from_str_radix(s, 16)
                .with_context(|| format!("invalid hex digit in --background '{spec}'"))
        };
        let byte = |i: usize| -> Result<u8> {
            let s = digits
                .get(i..i + 2)
                .with_context(|| format!("--background '{spec}' is shorter than expected"))?;
            u8::from_str_radix(s, 16)
                .with_context(|| format!("invalid hex byte in --background '{spec}'"))
        };
        // A doubled nibble: `f` → `0xff`, `3` → `0x33`.
        let expand = |v: u8| v * 17;

        match digits.len() {
            3 => Ok(Self {
                rgb: [expand(nibble(0)?), expand(nibble(1)?), expand(nibble(2)?)],
                alpha: 255,
            }),
            4 => Ok(Self {
                rgb: [expand(nibble(0)?), expand(nibble(1)?), expand(nibble(2)?)],
                alpha: expand(nibble(3)?),
            }),
            6 => Ok(Self {
                rgb: [byte(0)?, byte(2)?, byte(4)?],
                alpha: 255,
            }),
            8 => Ok(Self {
                rgb: [byte(0)?, byte(2)?, byte(4)?],
                alpha: byte(6)?,
            }),
            other => anyhow::bail!(
                "invalid --background '{spec}': {other} hex digits, expected 3, 4, 6 or 8 \
                 (or one of transparent/none/clear/black/white)"
            ),
        }
    }

    /// This background as an opaque RGB pixel (alpha discarded).
    #[must_use]
    pub const fn rgb_pixel(self) -> Rgb<u8> {
        Rgb(self.rgb)
    }

    /// This background as an RGBA pixel.
    #[must_use]
    pub const fn rgba_pixel(self) -> Rgba<u8> {
        Rgba([self.rgb[0], self.rgb[1], self.rgb[2], self.alpha])
    }
}

impl Default for Background {
    fn default() -> Self {
        Self::opaque(DEFAULT_BACKGROUND_RGB)
    }
}

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

/// Tunables for the software point-cloud renderer.
///
/// [`Default`] reproduces the renderer's historical hard-coded behaviour: a
/// `0x1e1e1e` clear colour and a 2-pixel splat radius.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PointCloudRenderOptions {
    /// Colour the frame is cleared to.
    pub background: Background,
    /// Splat disc radius in pixels; `0` draws a single pixel per Gaussian.
    pub splat_radius: u32,
}

impl Default for PointCloudRenderOptions {
    fn default() -> Self {
        Self {
            background: Background::default(),
            splat_radius: DEFAULT_SPLAT_RADIUS,
        }
    }
}

impl PointCloudRenderOptions {
    /// Build options from a `--background` string and a `--splat-radius`
    /// value, validating the radius against the CLI's documented range.
    ///
    /// # Errors
    ///
    /// Returns an error when `background` is not a recognised colour
    /// specification (see [`Background::parse`]) or when `splat_radius` is
    /// outside [`MIN_SPLAT_RADIUS`]`..=`[`MAX_SPLAT_RADIUS`].
    pub fn from_cli(background: &str, splat_radius: u32) -> Result<Self> {
        anyhow::ensure!(
            (MIN_SPLAT_RADIUS..=MAX_SPLAT_RADIUS).contains(&splat_radius),
            "--splat-radius must be within {MIN_SPLAT_RADIUS}..={MAX_SPLAT_RADIUS} \
             (got {splat_radius})"
        );
        Ok(Self {
            background: Background::parse(background)?,
            splat_radius,
        })
    }
}

// ---------------------------------------------------------------------------
// Rasterisation
// ---------------------------------------------------------------------------

/// Project and splat every visible Gaussian, handing each written pixel to
/// `set_pixel` as `(x, y, rgb)`.
///
/// The Z-buffer lives here so both the RGB and RGBA entry points share exactly
/// one implementation of the visibility rules (near/far clipping, the
/// `sigmoid(opacity) < 0.01` cull, and per-pixel depth ordering).
fn rasterize<F>(model: &GaussianModel, camera: &Camera, splat_radius: u32, mut set_pixel: F)
where
    F: FnMut(u32, u32, Rgb<u8>),
{
    let w = camera.width as usize;
    let h = camera.height as usize;
    if w == 0 || h == 0 {
        return;
    }
    let mut depth_buf = vec![f32::INFINITY; w * h];

    let sh_channels = ((model.sh_degree + 1).pow(2) * 3) as usize;
    // `MAX_SPLAT_RADIUS` is 5, so the `i32` conversion cannot overflow for any
    // value the CLI accepts; a caller passing something larger is clamped to
    // `i32::MAX` rather than wrapping.
    let radius = i32::try_from(splat_radius).unwrap_or(i32::MAX);
    let r2 = radius.saturating_mul(radius);

    for (i, g) in model.gaussians.iter().enumerate() {
        // Skip nearly-transparent Gaussians (sigmoid(opacity) < 0.01)
        if g.opacity < -4.6 {
            continue;
        }

        let p = na::Point3::new(g.position[0], g.position[1], g.position[2]);
        let p_cam = camera.world_to_cam(&p);

        if p_cam.z <= camera.near || p_cam.z >= camera.far {
            continue;
        }

        let [px, py] = camera.project(&p_cam);
        let cx = px.round() as i32;
        let cy = py.round() as i32;

        // SH DC → colour
        let sh_base = i * sh_channels;
        let r_f =
            (model.sh_coeffs.get(sh_base).copied().unwrap_or(0.0) * SH_C0 + 0.5).clamp(0.0, 1.0);
        let g_f = (model.sh_coeffs.get(sh_base + 1).copied().unwrap_or(0.0) * SH_C0 + 0.5)
            .clamp(0.0, 1.0);
        let b_f = (model.sh_coeffs.get(sh_base + 2).copied().unwrap_or(0.0) * SH_C0 + 0.5)
            .clamp(0.0, 1.0);

        let color = Rgb([
            (r_f * 255.0) as u8,
            (g_f * 255.0) as u8,
            (b_f * 255.0) as u8,
        ]);

        // Draw a filled disc at the projected position.
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                if dx * dx + dy * dy > r2 {
                    continue;
                }
                let ix = cx + dx;
                let iy = cy + dy;
                if ix < 0 || ix >= w as i32 || iy < 0 || iy >= h as i32 {
                    continue;
                }
                let pidx = iy as usize * w + ix as usize;
                if p_cam.z < depth_buf[pidx] {
                    depth_buf[pidx] = p_cam.z;
                    set_pixel(ix as u32, iy as u32, color);
                }
            }
        }
    }
}

/// Render a point-cloud preview honouring `options`.
///
/// A transparent [`Background`] cannot be expressed in an [`RgbImage`]: its
/// `rgb` component is used as the clear colour (black, for
/// [`Background::TRANSPARENT`]).  Use [`render_point_cloud_rgba`] when the
/// alpha channel matters.
#[must_use]
pub fn render_point_cloud_with_options(
    model: &GaussianModel,
    camera: &Camera,
    options: &PointCloudRenderOptions,
) -> RgbImage {
    let mut img = RgbImage::from_pixel(camera.width, camera.height, options.background.rgb_pixel());
    rasterize(model, camera, options.splat_radius, |x, y, color| {
        img.put_pixel(x, y, color);
    });
    img
}

/// Render a point-cloud preview into an RGBA image, preserving background
/// transparency.
///
/// Splatted pixels are always fully opaque; only the cleared background
/// carries [`Background::alpha`].
#[must_use]
pub fn render_point_cloud_rgba(
    model: &GaussianModel,
    camera: &Camera,
    options: &PointCloudRenderOptions,
) -> RgbaImage {
    let mut img =
        RgbaImage::from_pixel(camera.width, camera.height, options.background.rgba_pixel());
    rasterize(model, camera, options.splat_radius, |x, y, color| {
        img.put_pixel(x, y, Rgba([color[0], color[1], color[2], 255]));
    });
    img
}

/// Render a point-cloud preview with the default background and splat radius.
///
/// Equivalent to [`render_point_cloud_with_options`] with
/// [`PointCloudRenderOptions::default()`].
#[must_use]
pub fn render_point_cloud(model: &GaussianModel, camera: &Camera) -> RgbImage {
    render_point_cloud_with_options(model, camera, &PointCloudRenderOptions::default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxigaf::render::gaussian::GaussianAttributes;

    /// One mid-grey Gaussian at the world origin, which
    /// [`Camera::default_front`] projects to the exact centre of the frame.
    fn centre_model() -> GaussianModel {
        GaussianModel {
            gaussians: vec![GaussianAttributes {
                position: [0.0, 0.0, 0.0],
                _pad0: 0.0,
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: [-3.0; 3],
                opacity: 2.0,
            }],
            sh_coeffs: vec![0.0, 0.0, 0.0],
            sh_degree: 0,
            face_indices: vec![0],
            barycentric: vec![[1.0 / 3.0; 3]],
            local_offsets: vec![[0.0; 3]],
            is_rigid: vec![true],
        }
    }

    fn count_non_background(img: &RgbImage, background: Rgb<u8>) -> usize {
        img.pixels().filter(|p| **p != background).count()
    }

    #[test]
    fn default_options_reproduce_the_legacy_clear_colour() {
        // Regression: the renderer used to hard-code `Rgb([30, 30, 30])`.
        // `PointCloudRenderOptions::default()` must be byte-identical, or
        // every existing caller silently changes its output.
        let model = centre_model();
        let camera = Camera::default_front(32, 32);
        let img = render_point_cloud(&model, &camera);
        assert_eq!(*img.get_pixel(0, 0), Rgb([30, 30, 30]));
        assert_eq!(Background::default().rgb, [30, 30, 30]);
        assert_eq!(DEFAULT_SPLAT_RADIUS, 2);
    }

    #[test]
    fn background_option_reaches_the_rendered_frame() {
        // Regression for `--background` being warn-only: a caller-supplied
        // colour must actually clear the frame.
        let model = centre_model();
        let camera = Camera::default_front(32, 32);
        let options = PointCloudRenderOptions::from_cli("ffffff", 2).expect("options parse");
        let img = render_point_cloud_with_options(&model, &camera, &options);
        assert_eq!(*img.get_pixel(0, 0), Rgb([255, 255, 255]));
        assert_eq!(*img.get_pixel(31, 31), Rgb([255, 255, 255]));
    }

    #[test]
    fn splat_radius_option_changes_the_footprint() {
        // Regression for `--splat-radius` being warn-only: a larger radius
        // must cover strictly more pixels.
        let model = centre_model();
        let camera = Camera::default_front(64, 64);
        let background = Rgb(DEFAULT_BACKGROUND_RGB);

        let mut previous = 0usize;
        for radius in MIN_SPLAT_RADIUS..=MAX_SPLAT_RADIUS {
            let options =
                PointCloudRenderOptions::from_cli("1e1e1e", radius).expect("options parse");
            let img = render_point_cloud_with_options(&model, &camera, &options);
            let covered = count_non_background(&img, background);
            assert!(
                covered > previous,
                "radius {radius} covered {covered} pixels, radius {} covered {previous}",
                radius - 1
            );
            previous = covered;
        }
    }

    #[test]
    fn rgba_output_preserves_background_transparency() {
        let model = centre_model();
        let camera = Camera::default_front(32, 32);
        let options = PointCloudRenderOptions {
            background: Background::TRANSPARENT,
            splat_radius: 2,
        };
        let img = render_point_cloud_rgba(&model, &camera, &options);
        assert_eq!(
            img.get_pixel(0, 0)[3],
            0,
            "background must stay transparent"
        );
        assert_eq!(
            img.get_pixel(16, 16)[3],
            255,
            "splatted pixels must stay opaque"
        );
    }

    #[test]
    fn background_parses_every_documented_form() {
        assert_eq!(
            Background::parse("1e1e1e").expect("6 digits"),
            Background::opaque([30, 30, 30])
        );
        assert_eq!(
            Background::parse("#FFF").expect("3 digits"),
            Background::opaque([255, 255, 255])
        );
        assert_eq!(
            Background::parse("0x1e1e1eff").expect("8 digits"),
            Background::opaque([30, 30, 30])
        );
        assert_eq!(
            Background::parse("#fff8").expect("4 digits").alpha,
            0x88,
            "4-digit form must expand its alpha nibble"
        );
        assert!(Background::parse("  TRANSPARENT ")
            .expect("keyword")
            .is_transparent());
        assert_eq!(
            Background::parse("black").expect("keyword"),
            Background::opaque([0, 0, 0])
        );
    }

    #[test]
    fn background_rejects_nonsense_instead_of_defaulting() {
        for bad in ["", "ff", "menthol", "#12345", "zzzzzz", "#"] {
            assert!(
                Background::parse(bad).is_err(),
                "'{bad}' must be an explicit error, not a silent fallback"
            );
        }
    }

    #[test]
    fn from_cli_rejects_out_of_range_splat_radius() {
        assert!(PointCloudRenderOptions::from_cli("ffffff", 0).is_err());
        assert!(PointCloudRenderOptions::from_cli("ffffff", 6).is_err());
        assert!(PointCloudRenderOptions::from_cli("ffffff", 1).is_ok());
        assert!(PointCloudRenderOptions::from_cli("ffffff", 5).is_ok());
    }

    #[test]
    fn zero_sized_camera_renders_without_panicking() {
        let model = centre_model();
        let mut camera = Camera::default_front(1, 1);
        camera.width = 0;
        camera.height = 0;
        let img = render_point_cloud(&model, &camera);
        assert_eq!(img.width(), 0);
        assert_eq!(img.height(), 0);
    }
}
