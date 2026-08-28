//! UV texture map loading and sampling for FLAME meshes.
//!
//! This module provides:
//! - [`TextureMap`]: An RGBA/RGB/grayscale texture stored as f32 pixels in \[0,1\].
//! - [`FilterMode`]: Nearest or bilinear sampling.
//! - [`WrapMode`]: Clamp, repeat, or mirror UV wrap modes.
//! - [`UvTextureSampler`]: Configurable sampler that queries a texture at UV coordinates.
//! - [`TextureMeshExt`]: Extension trait for [`Mesh`] to sample textures at barycentric
//!   surface points or all vertex UV positions.

use crate::{error::FlameError, mesh::Mesh};

// ---------------------------------------------------------------------------
// TextureMap
// ---------------------------------------------------------------------------

/// An RGBA, RGB, or grayscale texture map stored as f32 pixels in `[0.0, 1.0]`.
///
/// Pixels are stored in row-major order: `data[row * width * channels + col * channels]`
/// is the first component of pixel `(col, row)`, where row 0 is the top of the image.
///
/// UV convention: `u=0` is the left edge, `u=1` is the right edge, `v=0` is the top,
/// `v=1` is the bottom (standard image / texture convention).
#[derive(Debug, Clone)]
pub struct TextureMap {
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
    /// Number of channels: 1 (grayscale), 3 (RGB), or 4 (RGBA).
    pub channels: usize,
    /// Raw pixel data in row-major order. Length == `width * height * channels`.
    data: Vec<f32>,
}

impl TextureMap {
    /// Create a new `TextureMap` from raw pixel data.
    ///
    /// # Arguments
    ///
    /// * `width` – image width in pixels (must be > 0)
    /// * `height` – image height in pixels (must be > 0)
    /// * `channels` – number of channels per pixel; must be 1, 3, or 4
    /// * `data` – raw f32 pixel values, length must equal `width * height * channels`
    ///
    /// # Errors
    ///
    /// Returns [`FlameError::InvalidParams`] if `width` or `height` is 0, if the
    /// channel count is not 1, 3, or 4, or if `data.len()` does not equal
    /// `width * height * channels`.
    pub fn new(
        width: usize,
        height: usize,
        channels: usize,
        data: Vec<f32>,
    ) -> Result<Self, FlameError> {
        if width == 0 || height == 0 {
            return Err(FlameError::InvalidParams(format!(
                "TextureMap dimensions must be > 0; got {width}x{height}"
            )));
        }
        if channels != 1 && channels != 3 && channels != 4 {
            return Err(FlameError::InvalidParams(format!(
                "TextureMap channels must be 1, 3, or 4; got {channels}"
            )));
        }
        let expected = width * height * channels;
        if data.len() != expected {
            return Err(FlameError::InvalidParams(format!(
                "TextureMap data length mismatch: expected {expected} ({width}×{height}×{channels}), got {}",
                data.len()
            )));
        }
        Ok(Self {
            width,
            height,
            channels,
            data,
        })
    }

    /// Create a 3-channel RGB texture from raw f32 data.
    ///
    /// # Errors
    ///
    /// Returns [`FlameError::InvalidParams`] if `data.len() != width * height * 3`.
    #[inline]
    pub fn from_rgb(width: usize, height: usize, data: Vec<f32>) -> Result<Self, FlameError> {
        Self::new(width, height, 3, data)
    }

    /// Create a 1-channel grayscale texture from raw f32 data.
    ///
    /// # Errors
    ///
    /// Returns [`FlameError::InvalidParams`] if `data.len() != width * height`.
    #[inline]
    pub fn from_grayscale(width: usize, height: usize, data: Vec<f32>) -> Result<Self, FlameError> {
        Self::new(width, height, 1, data)
    }

    /// Create a 1×1 texture filled with `color`.
    ///
    /// The length of `color` determines the channel count. Must be 1, 3, or 4.
    /// If the slice length is not 1, 3, or 4, falls back to a black 1-channel pixel.
    #[must_use]
    pub fn solid_color(color: &[f32]) -> Self {
        let channels = match color.len() {
            1 | 3 | 4 => color.len(),
            _ => 1,
        };
        let data: Vec<f32> = if color.len() == channels {
            color.to_vec()
        } else {
            vec![0.0; channels]
        };
        Self {
            width: 1,
            height: 1,
            channels,
            data,
        }
    }

    /// Create a `size × size` RGB checkerboard texture.
    ///
    /// Each pixel `(x, y)` is assigned `color_a` if `(x + y) % 2 == 0`, else `color_b`.
    /// `size` must be greater than 0; if 0 is passed, a 1×1 black texture is returned.
    #[must_use]
    pub fn checkerboard(size: usize, color_a: [f32; 3], color_b: [f32; 3]) -> Self {
        if size == 0 {
            return Self {
                width: 1,
                height: 1,
                channels: 3,
                data: vec![0.0; 3],
            };
        }
        let channels = 3usize;
        let mut data = Vec::with_capacity(size * size * channels);
        for y in 0..size {
            for x in 0..size {
                let color = if (x + y) % 2 == 0 { color_a } else { color_b };
                data.push(color[0]);
                data.push(color[1]);
                data.push(color[2]);
            }
        }
        Self {
            width: size,
            height: size,
            channels,
            data,
        }
    }

    /// Return a slice of `channels` f32 values for the pixel at column `x`, row `y`.
    ///
    /// Row 0 is the top of the image. The returned slice has exactly `self.channels` elements.
    ///
    /// # Panics
    ///
    /// Panics if `x >= self.width` or `y >= self.height`. Callers (such as internal sampling
    /// code) are responsible for clamping indices to valid ranges before calling.
    #[inline]
    #[must_use]
    pub fn pixel(&self, x: usize, y: usize) -> &[f32] {
        let offset = (y * self.width + x) * self.channels;
        &self.data[offset..offset + self.channels]
    }

    /// Image width in pixels.
    #[inline]
    #[must_use]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Image height in pixels.
    #[inline]
    #[must_use]
    pub fn height(&self) -> usize {
        self.height
    }

    /// Number of channels per pixel.
    #[inline]
    #[must_use]
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Raw pixel data slice.
    #[inline]
    #[must_use]
    pub fn data(&self) -> &[f32] {
        &self.data
    }
}

// ---------------------------------------------------------------------------
// FilterMode / WrapMode
// ---------------------------------------------------------------------------

/// Texture filtering mode used during sampling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FilterMode {
    /// Pick the nearest pixel (no interpolation).
    Nearest,
    /// Bilinear interpolation from the four surrounding pixels.
    #[default]
    Bilinear,
}

/// UV wrap mode applied when UV coordinates fall outside `[0, 1]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WrapMode {
    /// Clamp UV to `[0.0, 1.0]`.
    #[default]
    Clamp,
    /// UV wraps around with `coord.rem_euclid(1.0)`.
    Repeat,
    /// UV is mirrored: each unit interval reflects the image.
    Mirror,
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Apply the given wrap mode to a single UV component, returning a value in `[0, 1]`.
#[inline]
fn apply_wrap(coord: f32, mode: WrapMode) -> f32 {
    match mode {
        WrapMode::Clamp => coord.clamp(0.0, 1.0),
        WrapMode::Repeat => coord.rem_euclid(1.0),
        WrapMode::Mirror => {
            let t = coord.rem_euclid(2.0);
            if t > 1.0 {
                2.0 - t
            } else {
                t
            }
        }
    }
}

/// Bilinear interpolation across four corner pixel slices.
///
/// `tl` = top-left, `tr` = top-right, `bl` = bottom-left, `br` = bottom-right.
/// `fx` is the fractional x offset in `[0, 1]`, `fy` is the fractional y offset in `[0, 1]`.
/// All four slices must have the same length.
#[inline]
fn bilinear_interp(tl: &[f32], tr: &[f32], bl: &[f32], br: &[f32], fx: f32, fy: f32) -> Vec<f32> {
    let channels = tl.len();
    let mut result = Vec::with_capacity(channels);
    let inv_fx = 1.0 - fx;
    let inv_fy = 1.0 - fy;
    for c in 0..channels {
        // top row blend, bottom row blend, then vertical blend
        let top = inv_fx * tl[c] + fx * tr[c];
        let bot = inv_fx * bl[c] + fx * br[c];
        result.push(inv_fy * top + fy * bot);
    }
    result
}

// ---------------------------------------------------------------------------
// UvTextureSampler
// ---------------------------------------------------------------------------

/// Samples a [`TextureMap`] at given UV coordinates with configurable filtering and wrapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UvTextureSampler {
    /// Filtering mode (nearest or bilinear).
    pub filter: FilterMode,
    /// UV wrap mode (clamp, repeat, or mirror).
    pub wrap: WrapMode,
}

impl UvTextureSampler {
    /// Create a new sampler with explicit filter and wrap modes.
    #[must_use]
    #[inline]
    pub fn new(filter: FilterMode, wrap: WrapMode) -> Self {
        Self { filter, wrap }
    }

    /// Convenience constructor: bilinear filtering with clamp wrap.
    #[must_use]
    #[inline]
    pub fn bilinear_clamp() -> Self {
        Self::new(FilterMode::Bilinear, WrapMode::Clamp)
    }

    /// Convenience constructor: nearest filtering with repeat wrap.
    #[must_use]
    #[inline]
    pub fn nearest_repeat() -> Self {
        Self::new(FilterMode::Nearest, WrapMode::Repeat)
    }

    /// Sample the texture at UV coordinates `uv`, returning `channels` f32 values.
    ///
    /// UV convention: `u=0` is left, `u=1` is right, `v=0` is top, `v=1` is bottom.
    ///
    /// The wrap mode is applied to both components before sampling. For a 1×1 texture
    /// both modes reduce to the single pixel.
    #[must_use]
    pub fn sample(&self, texture: &TextureMap, uv: [f32; 2]) -> Vec<f32> {
        let w = texture.width;
        let h = texture.height;

        // Apply wrap mode to both UV components.
        let u = apply_wrap(uv[0], self.wrap);
        let v = apply_wrap(uv[1], self.wrap);

        match self.filter {
            FilterMode::Nearest => {
                // Map [0,1] to pixel indices: texel i covers u in [i/W, (i+1)/W),
                // so px = floor(u*W). The clamp bound is saturating so a
                // zero-sized texture (which `TextureMap::new` now rejects,
                // but a hand-constructed value could still smuggle in) gives
                // `clamp(0, 0)` instead of the inverted `clamp(0, -1)` range,
                // which used to panic on the clamp's own precondition.
                let max_x = w.saturating_sub(1).cast_signed();
                let max_y = h.saturating_sub(1).cast_signed();
                let px = ((u * w as f32).floor() as isize).clamp(0, max_x) as usize;
                let py = ((v * h as f32).floor() as isize).clamp(0, max_y) as usize;
                texture.pixel(px, py).to_vec()
            }

            FilterMode::Bilinear => {
                // GPU-standard texel-center convention, matching Nearest
                // above: texel i's center sits at u = (i + 0.5) / W, so the
                // continuous texel-space position is `u*W - 0.5`. Previously
                // this branch used `u*(W-1)` instead, which agreed with
                // Nearest only at u=0, u=1, and u=0.5 and disagreed by up to
                // half a texel elsewhere, so switching `FilterMode` shifted
                // the sampled image. `x0`/`y0` are floored (and `fx`/`fy`
                // computed) BEFORE clamping, so a UV exactly at the border
                // still blends toward the true edge texel with the correct
                // weight instead of being pulled fully onto it.
                let x_f = u * w as f32 - 0.5;
                let y_f = v * h as f32 - 0.5;

                let x0_raw = x_f.floor() as isize;
                let y0_raw = y_f.floor() as isize;
                let fx = x_f - x0_raw as f32;
                let fy = y_f - y0_raw as f32;

                let max_x = w.saturating_sub(1).cast_signed();
                let max_y = h.saturating_sub(1).cast_signed();
                let x0 = x0_raw.clamp(0, max_x) as usize;
                let y0 = y0_raw.clamp(0, max_y) as usize;
                let x1 = (x0_raw + 1).clamp(0, max_x) as usize;
                let y1 = (y0_raw + 1).clamp(0, max_y) as usize;

                // tl=top-left, tr=top-right, bl=bottom-left, br=bottom-right
                // row y0 = top, row y1 = bottom (v=0 is top in image convention)
                let tl = texture.pixel(x0, y0);
                let tr = texture.pixel(x1, y0);
                let bl = texture.pixel(x0, y1);
                let br = texture.pixel(x1, y1);

                bilinear_interp(tl, tr, bl, br, fx, fy)
            }
        }
    }

    /// Sample the texture and return an RGB triple `[r, g, b]`.
    ///
    /// - Grayscale (1-channel): the single value is broadcast to all three channels.
    /// - RGB (3-channel): returned as-is.
    /// - RGBA (4-channel): alpha is dropped.
    /// - Any other channel count: zero-padded or truncated to 3.
    #[must_use]
    pub fn sample_rgb(&self, texture: &TextureMap, uv: [f32; 2]) -> [f32; 3] {
        let raw = self.sample(texture, uv);
        match raw.len() {
            0 => [0.0, 0.0, 0.0],
            1 => [raw[0], raw[0], raw[0]],
            2 => [raw[0], raw[1], 0.0],
            _ => [raw[0], raw[1], raw[2]], // 3 channels, RGBA, or more
        }
    }

    /// Sample the texture at every UV coordinate in `uvs`, returning one `Vec<f32>` per UV.
    #[must_use]
    pub fn sample_all(&self, texture: &TextureMap, uvs: &[[f32; 2]]) -> Vec<Vec<f32>> {
        uvs.iter().map(|&uv| self.sample(texture, uv)).collect()
    }
}

// ---------------------------------------------------------------------------
// TextureMeshExt
// ---------------------------------------------------------------------------

/// Extension trait for [`Mesh`] that enables texture sampling at surface positions.
pub trait TextureMeshExt {
    /// Sample `texture` at the surface point on face `face_index` defined by
    /// barycentric coordinates `barycentric = [w0, w1, w2]`.
    ///
    /// The interpolated UV is computed as:
    /// `uv = w0 * uv[v0] + w1 * uv[v1] + w2 * uv[v2]`
    ///
    /// Returns `None` if:
    /// - The mesh has no UV coordinates.
    /// - `face_index` is out of range.
    /// - Any face vertex index is out of the UV coordinate array range.
    fn sample_texture_at_barycentric(
        &self,
        face_index: u32,
        barycentric: [f32; 3],
        texture: &TextureMap,
        sampler: &UvTextureSampler,
    ) -> Option<Vec<f32>>;

    /// Sample `texture` at every vertex UV coordinate, one result per vertex.
    ///
    /// Returns `None` if the mesh has no UV coordinates.
    fn sample_texture_at_all_vertices(
        &self,
        texture: &TextureMap,
        sampler: &UvTextureSampler,
    ) -> Option<Vec<Vec<f32>>>;
}

impl TextureMeshExt for Mesh {
    fn sample_texture_at_barycentric(
        &self,
        face_index: u32,
        barycentric: [f32; 3],
        texture: &TextureMap,
        sampler: &UvTextureSampler,
    ) -> Option<Vec<f32>> {
        if self.uv_coords.is_empty() {
            return None;
        }

        let face = self.faces.get(face_index as usize)?;
        let i0 = face[0] as usize;
        let i1 = face[1] as usize;
        let i2 = face[2] as usize;

        let uv0 = self.uv_coords.get(i0)?;
        let uv1 = self.uv_coords.get(i1)?;
        let uv2 = self.uv_coords.get(i2)?;

        let [w0, w1, w2] = barycentric;
        let u = w0 * uv0[0] + w1 * uv1[0] + w2 * uv2[0];
        let v = w0 * uv0[1] + w1 * uv1[1] + w2 * uv2[1];

        Some(sampler.sample(texture, [u, v]))
    }

    fn sample_texture_at_all_vertices(
        &self,
        texture: &TextureMap,
        sampler: &UvTextureSampler,
    ) -> Option<Vec<Vec<f32>>> {
        if self.uv_coords.is_empty() {
            return None;
        }
        Some(sampler.sample_all(texture, &self.uv_coords))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::uv::UvMeshExt;
    use nalgebra as na;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// A 2×2 RGB texture:
    /// ```text
    /// TL=(1,0,0)  TR=(0,1,0)
    /// BL=(0,0,1)  BR=(1,1,0)
    /// ```
    fn two_by_two_rgb() -> TextureMap {
        let data: Vec<f32> = vec![
            1.0, 0.0, 0.0, // (0,0) TL = red
            0.0, 1.0, 0.0, // (1,0) TR = green
            0.0, 0.0, 1.0, // (0,1) BL = blue
            1.0, 1.0, 0.0, // (1,1) BR = yellow
        ];
        TextureMap::from_rgb(2, 2, data).expect("valid 2x2 RGB texture")
    }

    /// A 2×2 grayscale texture:
    /// ```text
    /// TL=0.0  TR=1.0
    /// BL=0.5  BR=0.25
    /// ```
    fn two_by_two_gray() -> TextureMap {
        let data: Vec<f32> = vec![0.0, 1.0, 0.5, 0.25];
        TextureMap::from_grayscale(2, 2, data).expect("valid 2x2 grayscale texture")
    }

    /// A simple triangle mesh with per-vertex UV coordinates.
    ///
    /// Vertices: (0,0,0), (1,0,0), (0,1,0)
    /// UVs:      [0,0],   [1,0],   [0,1]
    fn triangle_mesh_with_uvs() -> Mesh {
        let vertices = vec![
            na::Point3::new(0.0f32, 0.0, 0.0),
            na::Point3::new(1.0f32, 0.0, 0.0),
            na::Point3::new(0.0f32, 1.0, 0.0),
        ];
        let faces = vec![[0u32, 1, 2]];
        let uvs: Vec<[f32; 2]> = vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        Mesh::new(vertices, faces)
            .with_uv_coords(uvs)
            .expect("UV count matches vertex count")
    }

    /// A triangle mesh with **no** UV coordinates.
    fn triangle_mesh_no_uvs() -> Mesh {
        let vertices = vec![
            na::Point3::new(0.0f32, 0.0, 0.0),
            na::Point3::new(1.0f32, 0.0, 0.0),
            na::Point3::new(0.0f32, 1.0, 0.0),
        ];
        let faces = vec![[0u32, 1, 2]];
        Mesh::new(vertices, faces)
    }

    // -----------------------------------------------------------------------
    // TextureMap construction
    // -----------------------------------------------------------------------

    #[test]
    fn test_texture_map_new_valid() {
        let data = vec![0.5f32; 4 * 4 * 3];
        let tex = TextureMap::new(4, 4, 3, data.clone());
        assert!(tex.is_ok(), "valid 4×4 RGB texture should succeed");
        let tex = tex.expect("valid");
        assert_eq!(tex.width(), 4);
        assert_eq!(tex.height(), 4);
        assert_eq!(tex.channels(), 3);
        assert_eq!(tex.data().len(), data.len());
    }

    #[test]
    fn test_texture_map_data_size_mismatch() {
        let data = vec![0.0f32; 3 * 3 * 3 - 1]; // one too short
        let result = TextureMap::new(3, 3, 3, data);
        assert!(result.is_err(), "mismatched data length should return Err");
        let err_msg = format!("{}", result.expect_err("should be Err"));
        assert!(
            err_msg.contains("mismatch"),
            "error should mention 'mismatch': {err_msg}"
        );
    }

    #[test]
    fn test_texture_map_invalid_channels() {
        let data = vec![0.0f32; 2]; // channels=2 is invalid
        let result = TextureMap::new(1, 1, 2, data);
        assert!(result.is_err(), "channel count 2 should return Err");
    }

    #[test]
    fn test_texture_map_zero_width_or_height_errors() {
        // A 0×0 texture used to pass construction (0 == 0*0*channels) and
        // then panic inside `sample()`'s clamp precondition.
        assert!(
            TextureMap::new(0, 0, 3, Vec::new()).is_err(),
            "0x0 texture must be rejected at construction"
        );
        assert!(
            TextureMap::from_rgb(0, 4, Vec::new()).is_err(),
            "zero width with nonzero height must be rejected"
        );
        assert!(
            TextureMap::from_rgb(4, 0, Vec::new()).is_err(),
            "zero height with nonzero width must be rejected"
        );
    }

    #[test]
    fn test_solid_color() {
        let color = [0.2, 0.5, 0.8f32];
        let tex = TextureMap::solid_color(&color);
        assert_eq!(tex.width(), 1);
        assert_eq!(tex.height(), 1);
        assert_eq!(tex.channels(), 3);
        let px = tex.pixel(0, 0);
        assert!((px[0] - 0.2).abs() < 1e-6);
        assert!((px[1] - 0.5).abs() < 1e-6);
        assert!((px[2] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_checkerboard_pattern() {
        let white = [1.0f32; 3];
        let black = [0.0f32; 3];
        let tex = TextureMap::checkerboard(4, white, black);
        assert_eq!(tex.width(), 4);
        assert_eq!(tex.height(), 4);
        assert_eq!(tex.channels(), 3);

        // (0,0): (0+0)%2==0 → white
        let px00 = tex.pixel(0, 0);
        assert!((px00[0] - 1.0).abs() < 1e-6, "pixel (0,0) should be white");

        // (1,0): (1+0)%2==1 → black
        let px10 = tex.pixel(1, 0);
        assert!((px10[0] - 0.0).abs() < 1e-6, "pixel (1,0) should be black");

        // (2,1): (2+1)%2==1 → black
        let px21 = tex.pixel(2, 1);
        assert!((px21[0] - 0.0).abs() < 1e-6, "pixel (2,1) should be black");

        // (3,1): (3+1)%2==0 → white
        let px31 = tex.pixel(3, 1);
        assert!((px31[0] - 1.0).abs() < 1e-6, "pixel (3,1) should be white");
    }

    #[test]
    fn test_pixel_access() {
        let tex = two_by_two_rgb();
        // TL pixel
        let tl = tex.pixel(0, 0);
        assert_eq!(tl, &[1.0f32, 0.0, 0.0], "TL should be red");
        // TR pixel
        let tr = tex.pixel(1, 0);
        assert_eq!(tr, &[0.0f32, 1.0, 0.0], "TR should be green");
        // BL pixel
        let bl = tex.pixel(0, 1);
        assert_eq!(bl, &[0.0f32, 0.0, 1.0], "BL should be blue");
        // BR pixel
        let br = tex.pixel(1, 1);
        assert_eq!(br, &[1.0f32, 1.0, 0.0], "BR should be yellow");
    }

    // -----------------------------------------------------------------------
    // Nearest-neighbour sampling
    // -----------------------------------------------------------------------

    #[test]
    fn test_nearest_sample_center() {
        // Sample at the center of a 1×1 solid red texture.
        let tex = TextureMap::solid_color(&[1.0f32, 0.0, 0.0]);
        let sampler = UvTextureSampler::new(FilterMode::Nearest, WrapMode::Clamp);
        let result = sampler.sample(&tex, [0.5, 0.5]);
        assert_eq!(result.len(), 3);
        assert!((result[0] - 1.0).abs() < 1e-6, "R should be 1.0");
        assert!((result[1] - 0.0).abs() < 1e-6, "G should be 0.0");
    }

    #[test]
    fn test_nearest_sample_clamp_wrap() {
        // UV outside [0,1] should clamp to the nearest border pixel.
        let tex = two_by_two_rgb();
        let sampler = UvTextureSampler::new(FilterMode::Nearest, WrapMode::Clamp);

        // u=1.5, v=0.5 → clamped to (1.0, 0.5) → right column, top half
        // → nearest is pixel (1, 0) = green
        let result = sampler.sample(&tex, [1.5, 0.0]);
        assert!(
            (result[1] - 1.0).abs() < 1e-6,
            "clamped UV should land on green pixel"
        );

        // u=-0.5, v=0.0 → clamped to (0, 0) = red
        let result2 = sampler.sample(&tex, [-0.5, 0.0]);
        assert!(
            (result2[0] - 1.0).abs() < 1e-6,
            "clamped negative UV should land on red pixel"
        );
    }

    #[test]
    fn test_nearest_sample_repeat_wrap() {
        let tex = two_by_two_rgb();
        let sampler = UvTextureSampler::new(FilterMode::Nearest, WrapMode::Repeat);

        // u=1.1 → rem_euclid(1.0) = 0.1 → left half → column 0 → red (for v near 0)
        let result = sampler.sample(&tex, [1.1, 0.0]);
        // floor(0.1 * 2) = 0 → pixel (0, 0) = red
        assert!(
            (result[0] - 1.0).abs() < 1e-6,
            "repeated UV should sample red"
        );

        // u=-0.1 → rem_euclid(1.0) = 0.9 → right half → column 1
        let result2 = sampler.sample(&tex, [-0.1, 0.0]);
        // floor(0.9 * 2) = 1 → pixel (1, 0) = green
        assert!(
            (result2[1] - 1.0).abs() < 1e-6,
            "negative repeat UV should sample green"
        );
    }

    #[test]
    fn test_nearest_sample_mirror_wrap() {
        let tex = two_by_two_rgb();
        let sampler = UvTextureSampler::new(FilterMode::Nearest, WrapMode::Mirror);

        // u=1.2 → rem_euclid(2) = 1.2 > 1.0 → 2 - 1.2 = 0.8 → right half → column 1
        let result = sampler.sample(&tex, [1.2, 0.0]);
        // floor(0.8 * 2) = 1 → pixel (1, 0) = green
        assert!(
            (result[1] - 1.0).abs() < 1e-6,
            "mirrored UV at 1.2 should sample green"
        );

        // u=1.8 → rem_euclid(2) = 1.8 > 1.0 → 2 - 1.8 = 0.2 → left half → column 0
        let result2 = sampler.sample(&tex, [1.8, 0.0]);
        // floor(0.2 * 2) = 0 → pixel (0, 0) = red
        assert!(
            (result2[0] - 1.0).abs() < 1e-6,
            "mirrored UV at 1.8 should sample red"
        );
    }

    // -----------------------------------------------------------------------
    // Bilinear sampling
    // -----------------------------------------------------------------------

    #[test]
    fn test_bilinear_sample_center() {
        // Sample a 2×2 RGBA texture at the exact center (u=0.5, v=0.5).
        // With the (W-1) mapping: x_f = 0.5 * 1 = 0.5, x0=0, x1=1, fx=0.5 (same for y).
        // Result = average of all 4 corners.
        let tex = two_by_two_rgb();
        let sampler = UvTextureSampler::bilinear_clamp();

        let result = sampler.sample(&tex, [0.5, 0.5]);
        // R: TL=1, TR=0, BL=0, BR=1 → average = 0.5
        // G: TL=0, TR=1, BL=0, BR=1 → average = 0.5
        // B: TL=0, TR=0, BL=1, BR=0 → average = 0.25
        assert!(
            (result[0] - 0.5).abs() < 1e-5,
            "R should be 0.5, got {}",
            result[0]
        );
        assert!(
            (result[1] - 0.5).abs() < 1e-5,
            "G should be 0.5, got {}",
            result[1]
        );
        assert!(
            (result[2] - 0.25).abs() < 1e-5,
            "B should be 0.25, got {}",
            result[2]
        );
    }

    #[test]
    fn test_bilinear_sample_interpolation() {
        // Sample at the exact top-left pixel (u=0, v=0) → should return TL value.
        let tex = two_by_two_rgb();
        let sampler = UvTextureSampler::bilinear_clamp();

        let tl = sampler.sample(&tex, [0.0, 0.0]);
        assert!((tl[0] - 1.0).abs() < 1e-6, "TL corner should be red R=1");
        assert!((tl[1] - 0.0).abs() < 1e-6, "TL corner G=0");

        // Sample at top-right (u=1, v=0) → should return TR value.
        let tr = sampler.sample(&tex, [1.0, 0.0]);
        assert!((tr[0] - 0.0).abs() < 1e-6, "TR corner R=0");
        assert!((tr[1] - 1.0).abs() < 1e-6, "TR corner should be green G=1");
    }

    #[test]
    fn test_bilinear_clamp_boundary() {
        // UV outside [0,1] should clamp and still return a valid pixel color.
        let tex = two_by_two_rgb();
        let sampler = UvTextureSampler::bilinear_clamp();

        // u=2.0 → clamped to 1.0 → right column → TR=(0,1,0)
        let result = sampler.sample(&tex, [2.0, 0.0]);
        assert!(
            (result[1] - 1.0).abs() < 1e-6,
            "clamped u=2 should give green"
        );

        // v=-1.0 → clamped to 0.0 → top row
        let result2 = sampler.sample(&tex, [0.0, -1.0]);
        assert!(
            (result2[0] - 1.0).abs() < 1e-6,
            "clamped v=-1 should give red TL"
        );
    }

    #[test]
    fn test_nearest_and_bilinear_agree_at_texel_centers() {
        // Nearest and Bilinear must use the same UV-to-texel convention. At a
        // texel's exact center (u = (i+0.5)/W), Bilinear's fractional weight
        // is exactly 0 and it should collapse to that same texel, matching
        // Nearest exactly. Before the fix, Bilinear used `u*(W-1)` instead of
        // `u*W - 0.5`, which only coincided with this convention at u=0, 0.5,
        // and 1 — texel 0's center (u=0.125 for W=4) diverged by 0.375 of a
        // texel.
        let w = 4usize;
        let data: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4]; // one distinct value per column
        let tex = TextureMap::from_grayscale(w, 1, data.clone()).expect("valid texture");

        let nearest = UvTextureSampler::new(FilterMode::Nearest, WrapMode::Clamp);
        let bilinear = UvTextureSampler::new(FilterMode::Bilinear, WrapMode::Clamp);

        for (i, &expected) in data.iter().enumerate() {
            let u = (i as f32 + 0.5) / w as f32;
            let n = nearest.sample(&tex, [u, 0.5]);
            let b = bilinear.sample(&tex, [u, 0.5]);
            assert!(
                (n[0] - expected).abs() < 1e-6,
                "nearest at texel {i}'s center should read texel {i} exactly, got {}",
                n[0]
            );
            assert!(
                (b[0] - expected).abs() < 1e-5,
                "bilinear at texel {i}'s center should also read texel {i} exactly \
                 (fx=0), got {}",
                b[0]
            );
        }
    }

    // -----------------------------------------------------------------------
    // sample_rgb helpers
    // -----------------------------------------------------------------------

    #[test]
    fn test_sample_rgb_expands_grayscale() {
        let tex = two_by_two_gray();
        let sampler = UvTextureSampler::new(FilterMode::Nearest, WrapMode::Clamp);

        // Pixel (0,0) = 0.0 grayscale → RGB should be (0.0, 0.0, 0.0)
        let rgb = sampler.sample_rgb(&tex, [0.0, 0.0]);
        assert!((rgb[0] - 0.0).abs() < 1e-6, "grayscale 0.0 → R=0");
        assert!((rgb[1] - 0.0).abs() < 1e-6, "grayscale 0.0 → G=0");
        assert!((rgb[2] - 0.0).abs() < 1e-6, "grayscale 0.0 → B=0");

        // Pixel (1,0) = 1.0 grayscale → RGB should be (1.0, 1.0, 1.0)
        let rgb2 = sampler.sample_rgb(&tex, [1.0, 0.0]);
        assert!((rgb2[0] - 1.0).abs() < 1e-6, "grayscale 1.0 → R=1");
        assert!((rgb2[1] - 1.0).abs() < 1e-6, "grayscale 1.0 → G=1");
        assert!((rgb2[2] - 1.0).abs() < 1e-6, "grayscale 1.0 → B=1");
    }

    #[test]
    fn test_sample_rgb_truncates_rgba() {
        // Build a 1×1 RGBA texture with alpha=0.5
        let data = vec![0.2f32, 0.4, 0.6, 0.5];
        let tex = TextureMap::new(1, 1, 4, data).expect("valid RGBA texture");
        let sampler = UvTextureSampler::bilinear_clamp();

        let rgb = sampler.sample_rgb(&tex, [0.5, 0.5]);
        assert!((rgb[0] - 0.2).abs() < 1e-6, "R should be 0.2");
        assert!((rgb[1] - 0.4).abs() < 1e-6, "G should be 0.4");
        assert!((rgb[2] - 0.6).abs() < 1e-6, "B should be 0.6");
        // Alpha channel is silently dropped
    }

    // -----------------------------------------------------------------------
    // Batch sampling
    // -----------------------------------------------------------------------

    #[test]
    fn test_batch_sampling() {
        let tex = two_by_two_rgb();
        let sampler = UvTextureSampler::new(FilterMode::Nearest, WrapMode::Clamp);

        let uvs: Vec<[f32; 2]> = vec![
            [0.0, 0.0], // → TL = red
            [1.0, 0.0], // → TR = green
            [0.0, 1.0], // → BL = blue
            [1.0, 1.0], // → BR = yellow
        ];

        let results = sampler.sample_all(&tex, &uvs);
        assert_eq!(results.len(), 4, "should return one result per UV");

        // TL: red
        assert!((results[0][0] - 1.0).abs() < 1e-6 && results[0][1].abs() < 1e-6);
        // TR: green
        assert!(results[1][0].abs() < 1e-6 && (results[1][1] - 1.0).abs() < 1e-6);
        // BL: blue
        assert!(
            results[2][0].abs() < 1e-6
                && results[2][1].abs() < 1e-6
                && (results[2][2] - 1.0).abs() < 1e-6
        );
        // BR: yellow
        assert!((results[3][0] - 1.0).abs() < 1e-6 && (results[3][1] - 1.0).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // TextureMeshExt
    // -----------------------------------------------------------------------

    #[test]
    fn test_texture_at_barycentric() {
        // Mesh UVs: v0=[0,0], v1=[1,0], v2=[0,1]
        // Barycentric [1,0,0] → UV [0,0] → TL pixel of 2×2 RGB = red
        let mesh = triangle_mesh_with_uvs();
        let tex = two_by_two_rgb();
        let sampler = UvTextureSampler::new(FilterMode::Nearest, WrapMode::Clamp);

        let result = mesh.sample_texture_at_barycentric(0, [1.0, 0.0, 0.0], &tex, &sampler);
        assert!(result.is_some(), "should return Some for valid face");
        let color = result.expect("some");
        assert!(
            (color[0] - 1.0).abs() < 1e-6,
            "barycentric [1,0,0] → UV [0,0] → red, got {color:?}"
        );

        // Barycentric [0,1,0] → UV [1,0] → nearest with clamp → pixel (1,0) = green
        let result2 = mesh.sample_texture_at_barycentric(0, [0.0, 1.0, 0.0], &tex, &sampler);
        let color2 = result2.expect("some");
        assert!(
            (color2[1] - 1.0).abs() < 1e-6,
            "barycentric [0,1,0] → UV [1,0] → green, got {color2:?}"
        );
    }

    #[test]
    fn test_texture_at_barycentric_no_uv_returns_none() {
        let mesh = triangle_mesh_no_uvs();
        let tex = two_by_two_rgb();
        let sampler = UvTextureSampler::bilinear_clamp();

        let result = mesh.sample_texture_at_barycentric(0, [1.0, 0.0, 0.0], &tex, &sampler);
        assert!(result.is_none(), "mesh without UV coords must return None");
    }

    #[test]
    fn test_sample_all_vertices() {
        // Mesh with 3 vertices, UVs at corners of [0,1]²
        // Sampling with nearest on 2×2 RGB texture should return one color per vertex.
        let mesh = triangle_mesh_with_uvs();
        let tex = two_by_two_rgb();
        let sampler = UvTextureSampler::new(FilterMode::Nearest, WrapMode::Clamp);

        let results = mesh
            .sample_texture_at_all_vertices(&tex, &sampler)
            .expect("mesh has UV coords");

        assert_eq!(results.len(), 3, "one result per vertex");

        // Vertex 0 UV [0,0] → TL = red
        assert!(
            (results[0][0] - 1.0).abs() < 1e-6,
            "vertex 0 should sample red"
        );
        // Vertex 1 UV [1,0] → TR = green
        assert!(
            (results[1][1] - 1.0).abs() < 1e-6,
            "vertex 1 should sample green"
        );
        // Vertex 2 UV [0,1] → BL = blue
        assert!(
            (results[2][2] - 1.0).abs() < 1e-6,
            "vertex 2 should sample blue"
        );
    }

    // -----------------------------------------------------------------------
    // apply_wrap helper
    // -----------------------------------------------------------------------

    #[test]
    fn test_apply_wrap_clamp() {
        assert!((apply_wrap(-0.5, WrapMode::Clamp) - 0.0).abs() < 1e-7);
        assert!((apply_wrap(0.5, WrapMode::Clamp) - 0.5).abs() < 1e-7);
        assert!((apply_wrap(1.5, WrapMode::Clamp) - 1.0).abs() < 1e-7);
    }

    #[test]
    fn test_apply_wrap_repeat() {
        // 1.3 → 0.3
        assert!((apply_wrap(1.3, WrapMode::Repeat) - 0.3).abs() < 1e-6);
        // -0.2 → 0.8
        assert!((apply_wrap(-0.2, WrapMode::Repeat) - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_apply_wrap_mirror() {
        // 1.2 → 2 - 1.2 = 0.8
        assert!((apply_wrap(1.2, WrapMode::Mirror) - 0.8).abs() < 1e-6);
        // 0.3 → 0.3
        assert!((apply_wrap(0.3, WrapMode::Mirror) - 0.3).abs() < 1e-6);
        // -0.2 → rem_euclid(2) = 1.8 > 1 → 2 - 1.8 = 0.2
        assert!((apply_wrap(-0.2, WrapMode::Mirror) - 0.2).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // bilinear_interp helper
    // -----------------------------------------------------------------------

    #[test]
    fn test_bilinear_interp_corners() {
        // At fx=0, fy=0 → pure tl
        let tl = &[1.0f32, 0.0, 0.0];
        let tr = &[0.0f32, 1.0, 0.0];
        let bl = &[0.0f32, 0.0, 1.0];
        let br = &[1.0f32, 1.0, 1.0];

        let result = bilinear_interp(tl, tr, bl, br, 0.0, 0.0);
        assert!((result[0] - 1.0).abs() < 1e-6, "fx=0,fy=0 → tl");

        let result2 = bilinear_interp(tl, tr, bl, br, 1.0, 0.0);
        assert!((result2[1] - 1.0).abs() < 1e-6, "fx=1,fy=0 → tr");

        let result3 = bilinear_interp(tl, tr, bl, br, 0.0, 1.0);
        assert!((result3[2] - 1.0).abs() < 1e-6, "fx=0,fy=1 → bl");
    }
}
