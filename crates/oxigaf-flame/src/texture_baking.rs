//! Texture baking for FLAME meshes.
//!
//! This module provides triangle-rasterisation-based baking of per-vertex
//! attributes (colours, normals, positions, or custom data) onto a 2-D UV
//! texture map.  The main entry points are:
//!
//! - [`bake_attribute`] – general per-vertex attribute → UV texture
//! - [`bake_vertex_colors`] – convenience wrapper for RGB colours
//! - [`bake_normals`] – normals encoded to `[0, 1]`
//! - [`bake_positions`] – raw 3-D positions
//! - [`bake`] – attribute-enum dispatcher
//!
//! Padding / dilation is applied after rasterisation via [`apply_uv_padding`].

use thiserror::Error;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during texture baking.
#[derive(Debug, Error)]
pub enum BakeError {
    /// Vertex and UV arrays have different lengths.
    #[error("Mesh has {n_verts} vertices but UV has {n_uv} entries")]
    UvCountMismatch { n_verts: usize, n_uv: usize },

    /// Face and UV-face arrays have different lengths.
    #[error("Mesh has {n_faces} faces but UV faces has {n_uv_faces} entries")]
    FaceCountMismatch { n_faces: usize, n_uv_faces: usize },

    /// Texture resolution is not a power of two.
    #[error("Invalid texture resolution {0}: must be power of two")]
    InvalidResolution(usize),

    /// Channel count must be 1, 3, or 4.
    #[error("Channel count {0} must be 1, 3, or 4")]
    InvalidChannels(usize),

    /// No vertices or faces provided.
    #[error("Empty mesh")]
    EmptyMesh,

    /// Generic parameter validation failure.
    #[error("Invalid parameter: {0}")]
    InvalidParam(String),
}

// ---------------------------------------------------------------------------
// BakedTexture
// ---------------------------------------------------------------------------

/// A baked 2-D texture stored as flat `f32` values in row-major order
/// (`height × width × channels`).
///
/// Optionally tracks which texels have been written so that padding and
/// coverage statistics work correctly.
#[derive(Debug, Clone)]
pub struct BakedTexture {
    /// Raw f32 pixel data.  Length == `width * height * channels`.
    pub data: Vec<f32>,
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
    /// Number of channels per texel (1, 3, or 4).
    pub channels: usize,
    /// `true` for every texel that has been explicitly written.
    filled: Vec<bool>,
}

impl BakedTexture {
    /// Create a new `BakedTexture` whose data is all zeros and no texel is
    /// marked as filled.
    #[must_use]
    pub fn new(width: usize, height: usize, channels: usize) -> Self {
        let n = width * height;
        Self {
            data: vec![0.0f32; n * channels],
            width,
            height,
            channels,
            filled: vec![false; n],
        }
    }

    /// Return an immutable slice over the `channels` values of texel `(x, y)`.
    ///
    /// # Panics
    ///
    /// Panics if `x >= self.width` or `y >= self.height` -- this is an
    /// ordinary slice-range index, so the bounds check is not
    /// debug-only; it applies in release builds too.
    #[inline]
    #[must_use]
    pub fn get_pixel(&self, x: usize, y: usize) -> &[f32] {
        let base = (y * self.width + x) * self.channels;
        &self.data[base..base + self.channels]
    }

    /// Write `value` into texel `(x, y)` and mark it as filled.
    ///
    /// If `value` has fewer elements than `self.channels`, the missing channels
    /// are left unchanged.  If it has more, the excess is ignored.
    #[inline]
    pub fn set_pixel(&mut self, x: usize, y: usize, value: &[f32]) {
        let base = (y * self.width + x) * self.channels;
        let count = value.len().min(self.channels);
        self.data[base..(count + base)].copy_from_slice(&value[..count]);
        self.filled[y * self.width + x] = true;
    }

    /// Convert to 8-bit per channel by clamping each value to `[0, 1]` and
    /// multiplying by 255.  The output length is `width * height * channels`.
    #[must_use]
    pub fn to_u8(&self) -> Vec<u8> {
        self.data
            .iter()
            .map(|&v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
            .collect()
    }

    /// Fraction of texels that have been written (i.e. marked as filled).
    #[must_use]
    pub fn coverage(&self) -> f32 {
        if self.filled.is_empty() {
            return 0.0;
        }
        let count = self.filled.iter().filter(|&&f| f).count();
        count as f32 / self.filled.len() as f32
    }

    /// Return `true` if texel `(x, y)` has been written.
    ///
    /// # Panics
    ///
    /// Panics if `x >= self.width` or `y >= self.height` (ordinary slice
    /// indexing, checked in release builds too).
    #[inline]
    #[must_use]
    pub fn is_filled(&self, x: usize, y: usize) -> bool {
        self.filled[y * self.width + x]
    }
}

// ---------------------------------------------------------------------------
// BakeConfig
// ---------------------------------------------------------------------------

/// Configuration controlling texture baking behaviour.
#[derive(Debug, Clone)]
pub struct BakeConfig {
    /// Output texture width in pixels (must be a power of two).
    pub width: usize,
    /// Output texture height in pixels (must be a power of two).
    pub height: usize,
    /// Number of channels (1, 3, or 4).
    pub channels: usize,
    /// Number of dilation iterations applied after rasterisation (0 = none).
    pub padding: usize,
    /// Background colour written to unfilled texels.  Length must equal
    /// `channels`.
    pub background: Vec<f32>,
}

impl Default for BakeConfig {
    fn default() -> Self {
        Self {
            width: 512,
            height: 512,
            channels: 3,
            padding: 2,
            background: vec![0.0, 0.0, 0.0],
        }
    }
}

// ---------------------------------------------------------------------------
// BakeAttribute
// ---------------------------------------------------------------------------

/// Selects which per-vertex attribute to bake.
pub enum BakeAttribute {
    /// Per-vertex RGB colour in `[0, 1]`.
    VertexColor,
    /// Per-vertex surface normal (encoded as `(n + 1) / 2` so values lie in `[0, 1]`).
    VertexNormal,
    /// Raw per-vertex 3-D position (not normalised).
    VertexPosition,
    /// Arbitrary per-vertex data — `n_verts × channels` layout.
    Custom(Vec<Vec<f32>>),
}

// ---------------------------------------------------------------------------
// UV coordinate types
// ---------------------------------------------------------------------------

/// UV coordinate pair in `[0, 1]²`.
pub type UvCoord = [f32; 2];

/// UV coordinates for one triangle (may differ from per-vertex UVs when the
/// mesh uses a split UV unwrap).
pub struct TriangleUv {
    /// UV coords for the three vertices of this triangle.
    pub uv: [[f32; 2]; 3],
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Signed 2-D edge function.  Returns a positive value when `p` is to the
/// left of the directed edge `a → b`.
#[inline]
fn edge_fn_2d(a: [f32; 2], b: [f32; 2], p: [f32; 2]) -> f32 {
    (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
}

/// Validate a `BakeConfig`.
fn validate_config(config: &BakeConfig) -> Result<(), BakeError> {
    if !config.width.is_power_of_two() || config.width == 0 {
        return Err(BakeError::InvalidResolution(config.width));
    }
    if !config.height.is_power_of_two() || config.height == 0 {
        return Err(BakeError::InvalidResolution(config.height));
    }
    if config.channels != 1 && config.channels != 3 && config.channels != 4 {
        return Err(BakeError::InvalidChannels(config.channels));
    }
    if config.background.len() != config.channels {
        return Err(BakeError::InvalidParam(format!(
            "background length {} != channels {}",
            config.background.len(),
            config.channels
        )));
    }
    Ok(())
}

/// Validate that UV count matches vertex count.
fn validate_uv_count(n_verts: usize, n_uv: usize) -> Result<(), BakeError> {
    if n_uv != n_verts {
        return Err(BakeError::UvCountMismatch { n_verts, n_uv });
    }
    Ok(())
}

/// Rasterise one triangle into `texture`.
///
/// `p0`, `p1`, `p2` are the pixel-space positions (f32) of the three corners.
/// `a0`, `a1`, `a2` are the attribute vectors (length == channels) at each corner.
/// `channels` must equal `a0.len()`.
fn rasterize_triangle(
    texture: &mut BakedTexture,
    p0: [f32; 2],
    p1: [f32; 2],
    p2: [f32; 2],
    a0: &[f32],
    a1: &[f32],
    a2: &[f32],
) {
    let width = texture.width as f32;
    let height = texture.height as f32;
    let channels = texture.channels;
    // `validate_config` restricts channels to 1, 3, or 4 on every public
    // entry point that reaches here; the fixed-size scratch buffer below
    // relies on that.
    debug_assert!(
        channels <= 4,
        "rasterize_triangle only supports up to 4 channels, got {channels}"
    );

    // Signed area (×2) — skip degenerate triangles.
    let area = edge_fn_2d(p0, p1, p2);
    if area.abs() < 1e-7 {
        return;
    }

    // Bounding box in pixel coordinates, clamped to texture extent.
    let min_x = p0[0].min(p1[0]).min(p2[0]).max(0.0) as usize;
    let max_x = (p0[0].max(p1[0]).max(p2[0]).min(width - 1.0)) as usize;
    let min_y = p0[1].min(p1[1]).min(p2[1]).max(0.0) as usize;
    let max_y = (p0[1].max(p1[1]).max(p2[1]).min(height - 1.0)) as usize;

    let inv_area = 1.0 / area;

    // Interpolated-attribute scratch buffer, reused across every covered
    // texel instead of heap-allocated per texel.
    let mut interp = [0.0f32; 4];

    for py in min_y..=max_y {
        for px in min_x..=max_x {
            let p = [px as f32 + 0.5, py as f32 + 0.5];

            // Barycentric weights via edge functions.
            let w0 = edge_fn_2d(p1, p2, p) * inv_area;
            let w1 = edge_fn_2d(p2, p0, p) * inv_area;
            let w2 = edge_fn_2d(p0, p1, p) * inv_area;

            // Point is inside the triangle when all weights are non-negative.
            if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 {
                continue;
            }

            // Interpolate attribute.
            for c in 0..channels {
                let v0 = if c < a0.len() { a0[c] } else { 0.0 };
                let v1 = if c < a1.len() { a1[c] } else { 0.0 };
                let v2 = if c < a2.len() { a2[c] } else { 0.0 };
                interp[c] = w0 * v0 + w1 * v1 + w2 * v2;
            }

            texture.set_pixel(px, py, &interp[..channels]);
        }
    }
}

/// Convert a UV coordinate to pixel position (f32) in a texture of the given
/// dimensions.
///
/// `v = 0` → top row, `v = 1` → bottom row (the image/texture convention
/// documented on [`crate::uv_texture::TextureMap`] and used by
/// [`crate::uv_texture::UvTextureSampler::sample`], so a texture baked
/// here and sampled there round-trips without a vertical flip).
#[inline]
fn uv_to_pixel(uv: [f32; 2], width: usize, height: usize) -> [f32; 2] {
    let px = uv[0] * (width as f32 - 1.0);
    let py = uv[1] * (height as f32 - 1.0);
    [px, py]
}

// ---------------------------------------------------------------------------
// Public baking functions
// ---------------------------------------------------------------------------

/// Bake per-vertex attribute data to a UV texture using triangle rasterisation.
///
/// `per_vertex_attrs` must have the same length as `vertices`; each inner
/// `Vec<f32>` must have length `config.channels`.
///
/// # Errors
///
/// Returns [`BakeError`] if any validation fails.
pub fn bake_attribute(
    vertices: &[[f32; 3]],
    faces: &[[u32; 3]],
    per_vertex_attrs: &[Vec<f32>],
    uv_coords: &[UvCoord],
    config: &BakeConfig,
) -> Result<BakedTexture, BakeError> {
    validate_config(config)?;

    if vertices.is_empty() || faces.is_empty() {
        return Err(BakeError::EmptyMesh);
    }

    let n_verts = vertices.len();
    validate_uv_count(n_verts, uv_coords.len())?;

    if per_vertex_attrs.len() != n_verts {
        return Err(BakeError::UvCountMismatch {
            n_verts,
            n_uv: per_vertex_attrs.len(),
        });
    }

    let mut texture = BakedTexture::new(config.width, config.height, config.channels);

    // Fill background.
    for y in 0..config.height {
        for x in 0..config.width {
            let base = (y * config.width + x) * config.channels;
            for c in 0..config.channels {
                texture.data[base + c] = config.background[c];
            }
        }
    }

    // Rasterise each triangle.
    for face in faces {
        let i0 = face[0] as usize;
        let i1 = face[1] as usize;
        let i2 = face[2] as usize;

        if i0 >= n_verts || i1 >= n_verts || i2 >= n_verts {
            continue;
        }

        let p0 = uv_to_pixel(uv_coords[i0], config.width, config.height);
        let p1 = uv_to_pixel(uv_coords[i1], config.width, config.height);
        let p2 = uv_to_pixel(uv_coords[i2], config.width, config.height);

        rasterize_triangle(
            &mut texture,
            p0,
            p1,
            p2,
            &per_vertex_attrs[i0],
            &per_vertex_attrs[i1],
            &per_vertex_attrs[i2],
        );
    }

    // Dilation padding.
    if config.padding > 0 {
        let padded = apply_uv_padding(&texture, config.padding);
        return Ok(padded);
    }

    Ok(texture)
}

/// Bake per-vertex RGB colours to a UV texture.
///
/// # Errors
///
/// Returns [`BakeError`] if validation fails.
pub fn bake_vertex_colors(
    vertices: &[[f32; 3]],
    faces: &[[u32; 3]],
    colors: &[[f32; 3]],
    uv_coords: &[UvCoord],
    config: &BakeConfig,
) -> Result<BakedTexture, BakeError> {
    if colors.len() != vertices.len() {
        return Err(BakeError::UvCountMismatch {
            n_verts: vertices.len(),
            n_uv: colors.len(),
        });
    }
    let attrs: Vec<Vec<f32>> = colors.iter().map(|c| c.to_vec()).collect();
    bake_attribute(vertices, faces, &attrs, uv_coords, config)
}

/// Bake per-vertex normals encoded as `(n + 1) / 2` so values lie in `[0, 1]`.
///
/// # Errors
///
/// Returns [`BakeError`] if validation fails.
pub fn bake_normals(
    vertices: &[[f32; 3]],
    faces: &[[u32; 3]],
    normals: &[[f32; 3]],
    uv_coords: &[UvCoord],
    config: &BakeConfig,
) -> Result<BakedTexture, BakeError> {
    if normals.len() != vertices.len() {
        return Err(BakeError::UvCountMismatch {
            n_verts: vertices.len(),
            n_uv: normals.len(),
        });
    }
    let attrs: Vec<Vec<f32>> = normals
        .iter()
        .map(|n| vec![(n[0] + 1.0) * 0.5, (n[1] + 1.0) * 0.5, (n[2] + 1.0) * 0.5])
        .collect();
    bake_attribute(vertices, faces, &attrs, uv_coords, config)
}

/// Bake raw per-vertex 3-D positions to a UV texture.
///
/// # Errors
///
/// Returns [`BakeError`] if validation fails.
pub fn bake_positions(
    vertices: &[[f32; 3]],
    faces: &[[u32; 3]],
    uv_coords: &[UvCoord],
    config: &BakeConfig,
) -> Result<BakedTexture, BakeError> {
    let attrs: Vec<Vec<f32>> = vertices.iter().map(|v| v.to_vec()).collect();
    bake_attribute(vertices, faces, &attrs, uv_coords, config)
}

/// Bake the selected attribute using the [`BakeAttribute`] enum.
///
/// # Errors
///
/// Returns [`BakeError`] if validation fails or a required array is missing.
pub fn bake(
    vertices: &[[f32; 3]],
    faces: &[[u32; 3]],
    normals: &[[f32; 3]],
    colors: Option<&[[f32; 3]]>,
    uv_coords: &[UvCoord],
    attribute: &BakeAttribute,
    config: &BakeConfig,
) -> Result<BakedTexture, BakeError> {
    match attribute {
        BakeAttribute::VertexColor => {
            let c = colors.ok_or_else(|| {
                BakeError::InvalidParam("VertexColor requires colors array".to_string())
            })?;
            bake_vertex_colors(vertices, faces, c, uv_coords, config)
        }
        BakeAttribute::VertexNormal => bake_normals(vertices, faces, normals, uv_coords, config),
        BakeAttribute::VertexPosition => bake_positions(vertices, faces, uv_coords, config),
        BakeAttribute::Custom(data) => bake_attribute(vertices, faces, data, uv_coords, config),
    }
}

// ---------------------------------------------------------------------------
// Padding / dilation
// ---------------------------------------------------------------------------

/// Dilate filled texels into neighbouring empty texels over `iterations` passes.
///
/// Each pass uses 4-connected neighbours (up, down, left, right).
/// Returns a new `BakedTexture` — the input is not modified.
#[must_use]
pub fn apply_uv_padding(texture: &BakedTexture, iterations: usize) -> BakedTexture {
    let mut current = texture.clone();
    let (width, height, channels) = (current.width, current.height, current.channels);

    for _ in 0..iterations {
        // Only the filled mask needs double-buffering (to avoid cascading
        // dilation within a single pass); pixel *data* is read directly
        // from `current.data` below. That's safe because dilation only
        // ever reads a texel that was already filled at the start of this
        // pass (`prev_filled[n_idx]`), and such a texel is never written
        // to during this same pass (the loop below only writes texels
        // where `prev_filled[idx]` is false) -- so its data cannot have
        // changed since the snapshot was taken. This avoids a full
        // `BakedTexture` clone (data + mask) every iteration.
        let prev_filled = current.filled.clone();

        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                if prev_filled[idx] {
                    continue;
                }
                // 4-connected neighbours: left, right, up, down.
                let neighbours: [(isize, isize); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
                for (dx, dy) in neighbours {
                    let nx: isize = x.cast_signed() + dx;
                    let ny: isize = y.cast_signed() + dy;
                    if nx < 0 || ny < 0 || nx >= width.cast_signed() || ny >= height.cast_signed() {
                        continue;
                    }
                    let nx = nx as usize;
                    let ny = ny as usize;
                    let n_idx = ny * width + nx;
                    if prev_filled[n_idx] {
                        let src_base = n_idx * channels;
                        let dst_base = idx * channels;
                        current
                            .data
                            .copy_within(src_base..src_base + channels, dst_base);
                        current.filled[idx] = true;
                        break;
                    }
                }
            }
        }
    }

    current
}

// ---------------------------------------------------------------------------
// UV mask
// ---------------------------------------------------------------------------

/// Compute a UV validity mask: `1.0` where any triangle covers the texel,
/// `0.0` otherwise.  Returns a flat `Vec<f32>` of length `width * height`.
#[must_use]
pub fn compute_uv_mask(
    faces: &[[u32; 3]],
    uv_coords: &[UvCoord],
    width: usize,
    height: usize,
) -> Vec<f32> {
    let n = width * height;
    let mut mask = vec![0.0f32; n];

    // Use a scratch BakedTexture with 1 channel.
    let mut scratch = BakedTexture::new(width, height, 1);

    for face in faces {
        let i0 = face[0] as usize;
        let i1 = face[1] as usize;
        let i2 = face[2] as usize;

        if i0 >= uv_coords.len() || i1 >= uv_coords.len() || i2 >= uv_coords.len() {
            continue;
        }

        let p0 = uv_to_pixel(uv_coords[i0], width, height);
        let p1 = uv_to_pixel(uv_coords[i1], width, height);
        let p2 = uv_to_pixel(uv_coords[i2], width, height);

        rasterize_triangle(&mut scratch, p0, p1, p2, &[1.0], &[1.0], &[1.0]);
    }

    for (mask_val, &filled) in mask.iter_mut().take(n).zip(scratch.filled.iter()) {
        *mask_val = if filled { 1.0 } else { 0.0 };
    }

    mask
}

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

/// Convert a baked texture to packed RGB u8 bytes.
///
/// If the texture has 3 channels the output is a direct byte conversion.
/// If it has 1 channel the single channel is replicated to R, G, B.
/// If it has 4 channels, only the first 3 are used.
///
/// # Errors
///
/// Returns [`BakeError::InvalidChannels`] if the channel count is unsupported.
pub fn baked_texture_to_rgb(texture: &BakedTexture) -> Result<Vec<u8>, BakeError> {
    let n_pixels = texture.width * texture.height;
    let mut out = Vec::with_capacity(n_pixels * 3);

    match texture.channels {
        1 => {
            for i in 0..n_pixels {
                let v = (texture.data[i].clamp(0.0, 1.0) * 255.0).round() as u8;
                out.push(v);
                out.push(v);
                out.push(v);
            }
        }
        3 => {
            for i in 0..n_pixels {
                let base = i * 3;
                out.push((texture.data[base].clamp(0.0, 1.0) * 255.0).round() as u8);
                out.push((texture.data[base + 1].clamp(0.0, 1.0) * 255.0).round() as u8);
                out.push((texture.data[base + 2].clamp(0.0, 1.0) * 255.0).round() as u8);
            }
        }
        4 => {
            for i in 0..n_pixels {
                let base = i * 4;
                out.push((texture.data[base].clamp(0.0, 1.0) * 255.0).round() as u8);
                out.push((texture.data[base + 1].clamp(0.0, 1.0) * 255.0).round() as u8);
                out.push((texture.data[base + 2].clamp(0.0, 1.0) * 255.0).round() as u8);
            }
        }
        other => return Err(BakeError::InvalidChannels(other)),
    }

    Ok(out)
}

/// Compute the UV-space area of each face (triangle) using the provided UV
/// coordinates.  Degenerate triangles have area 0.
///
/// Returns a `Vec<f32>` of length `faces.len()`.
#[must_use]
pub fn compute_face_uv_areas(faces: &[[u32; 3]], uv_coords: &[UvCoord]) -> Vec<f32> {
    let mut areas = Vec::with_capacity(faces.len());

    for face in faces {
        let i0 = face[0] as usize;
        let i1 = face[1] as usize;
        let i2 = face[2] as usize;

        if i0 >= uv_coords.len() || i1 >= uv_coords.len() || i2 >= uv_coords.len() {
            areas.push(0.0);
            continue;
        }

        let u0 = uv_coords[i0];
        let u1 = uv_coords[i1];
        let u2 = uv_coords[i2];

        // Half the absolute value of the 2-D cross product.
        let cross = (u1[0] - u0[0]) * (u2[1] - u0[1]) - (u1[1] - u0[1]) * (u2[0] - u0[0]);
        areas.push(cross.abs() * 0.5);
    }

    areas
}

/// Return a human-readable summary of bake statistics.
#[must_use]
pub fn format_bake_stats(texture: &BakedTexture) -> String {
    let filled = texture.filled.iter().filter(|&&f| f).count();
    let total = texture.width * texture.height;
    let coverage = texture.coverage() * 100.0;
    format!(
        "BakedTexture {}×{} ch={} — {}/{} texels filled ({:.1}%)",
        texture.width, texture.height, texture.channels, filled, total, coverage,
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Single-triangle mesh centred around UV (0.5, 0.5).
    fn single_triangle() -> ([[f32; 3]; 3], [[u32; 3]; 1], [UvCoord; 3]) {
        let verts = [[0.0f32, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]];
        let faces = [[0u32, 1, 2]];
        let uvs = [[0.0f32, 0.0], [1.0, 0.0], [0.5, 1.0]];
        (verts, faces, uvs)
    }

    /// `BakeConfig` with a small resolution to keep tests fast.
    fn small_config() -> BakeConfig {
        BakeConfig {
            width: 8,
            height: 8,
            channels: 3,
            padding: 0,
            background: vec![0.0, 0.0, 0.0],
        }
    }

    // -----------------------------------------------------------------------
    // BakedTexture construction and accessors
    // -----------------------------------------------------------------------

    #[test]
    fn test_baked_texture_new_zeros() {
        let t = BakedTexture::new(4, 4, 3);
        assert_eq!(t.data.len(), 4 * 4 * 3);
        assert!(
            t.data.iter().all(|&v| v == 0.0),
            "initial data must be zero"
        );
    }

    #[test]
    fn test_baked_texture_new_dimensions() {
        let t = BakedTexture::new(16, 32, 4);
        assert_eq!(t.width, 16);
        assert_eq!(t.height, 32);
        assert_eq!(t.channels, 4);
    }

    #[test]
    fn test_baked_texture_no_filled_initially() {
        let t = BakedTexture::new(4, 4, 3);
        for y in 0..4 {
            for x in 0..4 {
                assert!(!t.is_filled(x, y), "({x},{y}) must not be filled initially");
            }
        }
    }

    #[test]
    fn test_get_set_pixel_roundtrip() {
        let mut t = BakedTexture::new(4, 4, 3);
        let val = [0.1f32, 0.5, 0.9];
        t.set_pixel(2, 3, &val);
        let got = t.get_pixel(2, 3);
        for i in 0..3 {
            assert!((got[i] - val[i]).abs() < 1e-7, "channel {i} mismatch");
        }
    }

    #[test]
    fn test_set_pixel_marks_filled() {
        let mut t = BakedTexture::new(4, 4, 3);
        assert!(!t.is_filled(1, 2));
        t.set_pixel(1, 2, &[0.5, 0.5, 0.5]);
        assert!(t.is_filled(1, 2));
    }

    #[test]
    fn test_set_pixel_does_not_mark_others() {
        let mut t = BakedTexture::new(4, 4, 3);
        t.set_pixel(0, 0, &[1.0, 0.0, 0.0]);
        assert!(!t.is_filled(1, 0));
        assert!(!t.is_filled(0, 1));
    }

    #[test]
    fn test_to_u8_clamping() {
        let mut t = BakedTexture::new(1, 1, 3);
        // Over-range value.
        t.set_pixel(0, 0, &[2.0, -1.0, 0.5]);
        let bytes = t.to_u8();
        assert_eq!(bytes[0], 255, "2.0 clamps to 255");
        assert_eq!(bytes[1], 0, "-1.0 clamps to 0");
        assert_eq!(bytes[2], 128, "0.5 * 255 rounds to 128");
    }

    #[test]
    fn test_to_u8_length() {
        let t = BakedTexture::new(8, 8, 3);
        assert_eq!(t.to_u8().len(), 8 * 8 * 3);
    }

    #[test]
    fn test_coverage_empty_is_zero() {
        let t = BakedTexture::new(4, 4, 3);
        assert!((t.coverage() - 0.0).abs() < 1e-7);
    }

    #[test]
    fn test_coverage_after_fill() {
        let mut t = BakedTexture::new(4, 4, 3);
        t.set_pixel(0, 0, &[1.0, 0.0, 0.0]);
        // 1 out of 16.
        let expected = 1.0 / 16.0;
        assert!(
            (t.coverage() - expected).abs() < 1e-6,
            "coverage = {} expected ≈ {}",
            t.coverage(),
            expected
        );
    }

    #[test]
    fn test_coverage_full() {
        let mut t = BakedTexture::new(2, 2, 1);
        for y in 0..2 {
            for x in 0..2 {
                t.set_pixel(x, y, &[1.0]);
            }
        }
        assert!((t.coverage() - 1.0).abs() < 1e-7);
    }

    // -----------------------------------------------------------------------
    // BakeConfig
    // -----------------------------------------------------------------------

    #[test]
    fn test_default_config_sensible() {
        let cfg = BakeConfig::default();
        assert_eq!(cfg.width, 512);
        assert_eq!(cfg.height, 512);
        assert_eq!(cfg.channels, 3);
        assert_eq!(cfg.padding, 2);
        assert_eq!(cfg.background.len(), 3);
    }

    // -----------------------------------------------------------------------
    // edge_fn_2d
    // -----------------------------------------------------------------------

    #[test]
    fn test_edge_fn_2d_inside() {
        let a = [0.0f32, 0.0];
        let b = [1.0, 0.0];
        let c = [0.5, 1.0];
        // Centroid should be inside → all edge functions positive.
        let cx = (a[0] + b[0] + c[0]) / 3.0;
        let cy = (a[1] + b[1] + c[1]) / 3.0;
        let p = [cx, cy];
        assert!(
            edge_fn_2d(a, b, p) > 0.0 || edge_fn_2d(a, b, p) < 0.0,
            "centroid should produce a non-zero edge fn"
        );
        // For a CCW triangle the centroid should give positive for all edges.
        assert!(edge_fn_2d(a, b, p) >= 0.0);
        assert!(edge_fn_2d(b, c, p) >= 0.0);
        assert!(edge_fn_2d(c, a, p) >= 0.0);
    }

    #[test]
    fn test_edge_fn_2d_outside() {
        let a = [0.0f32, 0.0];
        let b = [1.0, 0.0];
        let p_outside = [0.5f32, -1.0]; // below the x-axis edge
                                        // For edge a→b the point below should give a negative value.
        assert!(edge_fn_2d(a, b, p_outside) < 0.0);
    }

    #[test]
    fn test_edge_fn_2d_on_edge() {
        let a = [0.0f32, 0.0];
        let b = [1.0, 0.0];
        let p = [0.5f32, 0.0];
        assert!((edge_fn_2d(a, b, p)).abs() < 1e-7, "point on edge → 0");
    }

    // -----------------------------------------------------------------------
    // compute_face_uv_areas
    // -----------------------------------------------------------------------

    #[test]
    fn test_face_uv_areas_known_triangle() {
        // Right triangle with legs of length 1 in UV space.
        let uvs: &[UvCoord] = &[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let faces: &[[u32; 3]] = &[[0, 1, 2]];
        let areas = compute_face_uv_areas(faces, uvs);
        assert_eq!(areas.len(), 1);
        assert!((areas[0] - 0.5).abs() < 1e-6, "area = {}", areas[0]);
    }

    #[test]
    fn test_face_uv_areas_unit_square_two_tris() {
        // Unit square split into two triangles: total area = 1.
        let uvs: &[UvCoord] = &[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let faces: &[[u32; 3]] = &[[0, 1, 2], [0, 2, 3]];
        let areas = compute_face_uv_areas(faces, uvs);
        let total: f32 = areas.iter().sum();
        assert!((total - 1.0).abs() < 1e-5, "total area = {total}");
    }

    #[test]
    fn test_face_uv_areas_degenerate() {
        // Three identical UV points → area 0.
        let uvs: &[UvCoord] = &[[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]];
        let faces: &[[u32; 3]] = &[[0, 1, 2]];
        let areas = compute_face_uv_areas(faces, uvs);
        assert!((areas[0]).abs() < 1e-7);
    }

    #[test]
    fn test_face_uv_areas_small_triangle() {
        // Equilateral-ish small triangle.
        let uvs: &[UvCoord] = &[[0.0, 0.0], [0.2, 0.0], [0.1, 0.1]];
        let faces: &[[u32; 3]] = &[[0, 1, 2]];
        let areas = compute_face_uv_areas(faces, uvs);
        // base=0.2, height=0.1 → area = 0.5 * 0.2 * 0.1 = 0.01.
        assert!((areas[0] - 0.01).abs() < 1e-6, "area = {}", areas[0]);
    }

    // -----------------------------------------------------------------------
    // compute_uv_mask
    // -----------------------------------------------------------------------

    #[test]
    fn test_uv_mask_empty_faces() {
        let mask = compute_uv_mask(&[], &[], 8, 8);
        assert_eq!(mask.len(), 64);
        assert!(mask.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_uv_mask_single_triangle_fills_texels() {
        // Triangle that covers the bottom-left quarter of UV space.
        let uvs: &[UvCoord] = &[[0.0, 0.0], [0.5, 0.0], [0.0, 0.5]];
        let faces: &[[u32; 3]] = &[[0, 1, 2]];
        let mask = compute_uv_mask(faces, uvs, 8, 8);
        // At least one texel must be 1.
        assert!(
            mask.iter().any(|&v| v > 0.0),
            "triangle should fill at least one texel"
        );
    }

    #[test]
    fn test_uv_mask_length() {
        let mask = compute_uv_mask(&[], &[], 16, 16);
        assert_eq!(mask.len(), 16 * 16);
    }

    // -----------------------------------------------------------------------
    // bake_attribute
    // -----------------------------------------------------------------------

    #[test]
    fn test_bake_attribute_basic() {
        let (verts, faces, uvs) = single_triangle();
        let attrs: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let cfg = small_config();
        let tex = bake_attribute(&verts, &faces, &attrs, &uvs, &cfg).expect("bake must succeed");
        assert_eq!(tex.width, 8);
        assert_eq!(tex.height, 8);
        assert!(tex.coverage() > 0.0, "some texels must be filled");
    }

    #[test]
    fn test_bake_attribute_centroid_interpolation() {
        // Corner colours: red, green, blue.
        // Centroid should be (1/3, 1/3, 1/3).
        let verts = [[0.0f32; 3]; 3];
        let faces = [[0u32, 1, 2]];
        // UV occupies the full texture.
        let uvs: [UvCoord; 3] = [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]];
        let attrs: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let cfg = BakeConfig {
            width: 64,
            height: 64,
            channels: 3,
            padding: 0,
            background: vec![0.0, 0.0, 0.0],
        };
        let tex = bake_attribute(&verts, &faces, &attrs, &uvs, &cfg).expect("bake must succeed");
        // The centroid in UV space is roughly (0.5, 0.33).
        // Check that the texture has values in it.
        assert!(tex.coverage() > 0.0);
    }

    #[test]
    fn test_bake_attribute_uv_mismatch_error() {
        let verts = [[0.0f32; 3]; 3];
        let faces = [[0u32, 1, 2]];
        let uvs: [UvCoord; 2] = [[0.0, 0.0], [1.0, 0.0]]; // only 2 UV for 3 verts
        let attrs: Vec<Vec<f32>> = vec![vec![0.0; 3]; 3];
        let cfg = small_config();
        let res = bake_attribute(&verts, &faces, &attrs, &uvs, &cfg);
        assert!(matches!(res, Err(BakeError::UvCountMismatch { .. })));
    }

    #[test]
    fn test_bake_attribute_empty_mesh_error() {
        let verts: &[[f32; 3]] = &[];
        let faces: &[[u32; 3]] = &[];
        let uvs: &[UvCoord] = &[];
        let attrs: Vec<Vec<f32>> = vec![];
        let cfg = small_config();
        let res = bake_attribute(verts, faces, &attrs, uvs, &cfg);
        assert!(matches!(res, Err(BakeError::EmptyMesh)));
    }

    #[test]
    fn test_bake_attribute_invalid_resolution_error() {
        let verts = [[0.0f32; 3]; 3];
        let faces = [[0u32, 1, 2]];
        let uvs: [UvCoord; 3] = [[0.0, 0.0]; 3];
        let attrs: Vec<Vec<f32>> = vec![vec![0.0; 3]; 3];
        let mut cfg = small_config();
        cfg.width = 10; // not power of two
        let res = bake_attribute(&verts, &faces, &attrs, &uvs, &cfg);
        assert!(matches!(res, Err(BakeError::InvalidResolution(10))));
    }

    #[test]
    fn test_bake_attribute_invalid_channels_error() {
        let verts = [[0.0f32; 3]; 3];
        let faces = [[0u32, 1, 2]];
        let uvs: [UvCoord; 3] = [[0.0, 0.0]; 3];
        let attrs: Vec<Vec<f32>> = vec![vec![0.0; 2]; 3];
        let cfg = BakeConfig {
            width: 8,
            height: 8,
            channels: 2, // invalid
            padding: 0,
            background: vec![0.0, 0.0],
        };
        let res = bake_attribute(&verts, &faces, &attrs, &uvs, &cfg);
        assert!(matches!(res, Err(BakeError::InvalidChannels(2))));
    }

    // -----------------------------------------------------------------------
    // bake_vertex_colors
    // -----------------------------------------------------------------------

    #[test]
    fn test_bake_vertex_colors_basic() {
        let (verts, faces, uvs) = single_triangle();
        let colors = [[1.0f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let cfg = small_config();
        let tex = bake_vertex_colors(&verts, &faces, &colors, &uvs, &cfg)
            .expect("bake_vertex_colors must succeed");
        assert!(tex.coverage() > 0.0);
    }

    #[test]
    fn test_bake_vertex_colors_all_red() {
        let (verts, faces, uvs) = single_triangle();
        let colors = [[1.0f32, 0.0, 0.0]; 3];
        let cfg = BakeConfig {
            width: 16,
            height: 16,
            channels: 3,
            padding: 0,
            background: vec![0.0, 0.0, 0.0],
        };
        let tex =
            bake_vertex_colors(&verts, &faces, &colors, &uvs, &cfg).expect("bake must succeed");
        // Every filled texel should be red.
        let total_pixels = tex.width * tex.height;
        for i in 0..total_pixels {
            if tex.filled[i] {
                let base = i * 3;
                assert!(
                    (tex.data[base] - 1.0).abs() < 1e-5,
                    "red channel must be 1.0 at pixel {i}"
                );
                assert!(
                    tex.data[base + 1].abs() < 1e-5,
                    "green must be 0 at pixel {i}"
                );
                assert!(
                    tex.data[base + 2].abs() < 1e-5,
                    "blue must be 0 at pixel {i}"
                );
            }
        }
    }

    #[test]
    fn test_bake_vertex_colors_count_mismatch() {
        let (verts, faces, uvs) = single_triangle();
        let colors = [[1.0f32, 0.0, 0.0]; 2]; // only 2 colours for 3 verts
        let cfg = small_config();
        let res = bake_vertex_colors(&verts, &faces, &colors, &uvs, &cfg);
        assert!(matches!(res, Err(BakeError::UvCountMismatch { .. })));
    }

    // -----------------------------------------------------------------------
    // bake_normals
    // -----------------------------------------------------------------------

    #[test]
    fn test_bake_normals_encoding_range() {
        let (verts, faces, uvs) = single_triangle();
        let normals = [[0.0f32, 0.0, 1.0]; 3]; // all Z normals
        let cfg = BakeConfig {
            width: 16,
            height: 16,
            channels: 3,
            padding: 0,
            background: vec![0.0, 0.0, 0.0],
        };
        let tex =
            bake_normals(&verts, &faces, &normals, &uvs, &cfg).expect("bake_normals must succeed");
        // Z normal (0,0,1) encodes to (0.5, 0.5, 1.0).
        let total_pixels = tex.width * tex.height;
        for i in 0..total_pixels {
            if tex.filled[i] {
                let base = i * 3;
                // All encoded values must be in [0,1].
                for c in 0..3 {
                    assert!(
                        tex.data[base + c] >= -1e-5 && tex.data[base + c] <= 1.0 + 1e-5,
                        "encoded normal out of range at pixel {i} channel {c}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_bake_normals_known_encoding() {
        // Normal (1,0,0) → encoded (1.0, 0.5, 0.5).
        let verts = [[0.0f32; 3]; 3];
        let faces = [[0u32, 1, 2]];
        // UV covers most of the texture.
        let uvs: [UvCoord; 3] = [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]];
        let normals = [[1.0f32, 0.0, 0.0]; 3];
        let cfg = BakeConfig {
            width: 16,
            height: 16,
            channels: 3,
            padding: 0,
            background: vec![0.0, 0.0, 0.0],
        };
        let tex =
            bake_normals(&verts, &faces, &normals, &uvs, &cfg).expect("bake_normals must succeed");
        let total_pixels = tex.width * tex.height;
        for i in 0..total_pixels {
            if tex.filled[i] {
                let base = i * 3;
                assert!((tex.data[base] - 1.0).abs() < 1e-5, "R must be 1.0");
                assert!((tex.data[base + 1] - 0.5).abs() < 1e-5, "G must be 0.5");
                assert!((tex.data[base + 2] - 0.5).abs() < 1e-5, "B must be 0.5");
            }
        }
    }

    #[test]
    fn test_bake_normals_mismatch_error() {
        let (verts, faces, uvs) = single_triangle();
        let normals = [[0.0f32; 3]; 2];
        let cfg = small_config();
        let res = bake_normals(&verts, &faces, &normals, &uvs, &cfg);
        assert!(matches!(res, Err(BakeError::UvCountMismatch { .. })));
    }

    // -----------------------------------------------------------------------
    // bake_positions
    // -----------------------------------------------------------------------

    #[test]
    fn test_bake_positions_basic() {
        let (verts, faces, uvs) = single_triangle();
        let cfg = BakeConfig {
            width: 16,
            height: 16,
            channels: 3,
            padding: 0,
            background: vec![0.0, 0.0, 0.0],
        };
        let tex = bake_positions(&verts, &faces, &uvs, &cfg).expect("bake_positions must succeed");
        assert!(tex.coverage() > 0.0);
    }

    #[test]
    fn test_bake_positions_empty_mesh_error() {
        let verts: &[[f32; 3]] = &[];
        let faces: &[[u32; 3]] = &[];
        let uvs: &[UvCoord] = &[];
        let cfg = small_config();
        let res = bake_positions(verts, faces, uvs, &cfg);
        assert!(matches!(res, Err(BakeError::EmptyMesh)));
    }

    // -----------------------------------------------------------------------
    // bake (enum dispatch)
    // -----------------------------------------------------------------------

    #[test]
    fn test_bake_vertex_color_enum() {
        let (verts, faces, uvs) = single_triangle();
        let normals = [[0.0f32; 3]; 3];
        let colors = [[0.8f32, 0.2, 0.0]; 3];
        let cfg = small_config();
        let tex = bake(
            &verts,
            &faces,
            &normals,
            Some(&colors),
            &uvs,
            &BakeAttribute::VertexColor,
            &cfg,
        )
        .expect("bake must succeed");
        assert!(tex.coverage() > 0.0);
    }

    #[test]
    fn test_bake_vertex_normal_enum() {
        let (verts, faces, uvs) = single_triangle();
        let normals = [[0.0f32, 1.0, 0.0]; 3];
        let cfg = small_config();
        let tex = bake(
            &verts,
            &faces,
            &normals,
            None,
            &uvs,
            &BakeAttribute::VertexNormal,
            &cfg,
        )
        .expect("bake must succeed");
        assert!(tex.coverage() > 0.0);
    }

    #[test]
    fn test_bake_vertex_position_enum() {
        let (verts, faces, uvs) = single_triangle();
        let normals = [[0.0f32; 3]; 3];
        let cfg = small_config();
        let tex = bake(
            &verts,
            &faces,
            &normals,
            None,
            &uvs,
            &BakeAttribute::VertexPosition,
            &cfg,
        )
        .expect("bake must succeed");
        assert!(tex.coverage() > 0.0);
    }

    #[test]
    fn test_bake_custom_enum() {
        let (verts, faces, uvs) = single_triangle();
        let normals = [[0.0f32; 3]; 3];
        let custom: Vec<Vec<f32>> = vec![
            vec![0.9, 0.1, 0.2],
            vec![0.1, 0.9, 0.3],
            vec![0.2, 0.3, 0.9],
        ];
        let cfg = small_config();
        let tex = bake(
            &verts,
            &faces,
            &normals,
            None,
            &uvs,
            &BakeAttribute::Custom(custom),
            &cfg,
        )
        .expect("bake must succeed");
        assert!(tex.coverage() > 0.0);
    }

    #[test]
    fn test_bake_vertex_color_no_colors_error() {
        let (verts, faces, uvs) = single_triangle();
        let normals = [[0.0f32; 3]; 3];
        let cfg = small_config();
        let res = bake(
            &verts,
            &faces,
            &normals,
            None, // no colors!
            &uvs,
            &BakeAttribute::VertexColor,
            &cfg,
        );
        assert!(matches!(res, Err(BakeError::InvalidParam(_))));
    }

    // -----------------------------------------------------------------------
    // apply_uv_padding
    // -----------------------------------------------------------------------

    #[test]
    fn test_padding_fills_adjacent_empty_texel() {
        // 3×1 texture: pixel 0 is filled red, pixels 1 and 2 are empty.
        let mut tex = BakedTexture::new(4, 1, 3);
        tex.set_pixel(0, 0, &[1.0, 0.0, 0.0]);

        let padded = apply_uv_padding(&tex, 1);

        // Pixel 1 should now have the value of pixel 0.
        assert!(
            padded.is_filled(1, 0),
            "pixel 1 should be filled after 1 dilation"
        );
        let p1 = padded.get_pixel(1, 0);
        assert!((p1[0] - 1.0).abs() < 1e-6, "pixel 1 R must be 1.0");
    }

    #[test]
    fn test_padding_two_iterations_propagates_further() {
        let mut tex = BakedTexture::new(4, 1, 3);
        tex.set_pixel(0, 0, &[0.5, 0.0, 0.0]);

        let padded = apply_uv_padding(&tex, 2);
        // After 2 iterations, pixel 2 should be filled.
        assert!(
            padded.is_filled(2, 0),
            "pixel 2 should be filled after 2 iterations"
        );
    }

    #[test]
    fn test_padding_zero_iterations_is_identity() {
        let mut tex = BakedTexture::new(4, 4, 3);
        tex.set_pixel(0, 0, &[1.0, 1.0, 1.0]);
        let padded = apply_uv_padding(&tex, 0);
        assert_eq!(padded.coverage(), tex.coverage());
    }

    #[test]
    fn test_padding_does_not_overwrite_filled_texels() {
        let mut tex = BakedTexture::new(4, 1, 3);
        tex.set_pixel(0, 0, &[1.0, 0.0, 0.0]);
        tex.set_pixel(1, 0, &[0.0, 1.0, 0.0]);

        let padded = apply_uv_padding(&tex, 1);
        // Pixel 0 was filled — it must keep its value.
        let p0 = padded.get_pixel(0, 0);
        assert!((p0[0] - 1.0).abs() < 1e-6, "pixel 0 R must stay 1.0");
        assert!((p0[1]).abs() < 1e-6, "pixel 0 G must stay 0.0");
    }

    // -----------------------------------------------------------------------
    // baked_texture_to_rgb
    // -----------------------------------------------------------------------

    #[test]
    fn test_baked_texture_to_rgb_3ch() {
        let mut t = BakedTexture::new(2, 2, 3);
        t.set_pixel(0, 0, &[1.0, 0.5, 0.0]);
        let rgb = baked_texture_to_rgb(&t).expect("must succeed for 3ch");
        assert_eq!(rgb.len(), 2 * 2 * 3);
        assert_eq!(rgb[0], 255);
        assert_eq!(rgb[1], 128);
        assert_eq!(rgb[2], 0);
    }

    #[test]
    fn test_baked_texture_to_rgb_1ch() {
        let mut t = BakedTexture::new(1, 1, 1);
        t.set_pixel(0, 0, &[0.5]);
        let rgb = baked_texture_to_rgb(&t).expect("must succeed for 1ch");
        assert_eq!(rgb.len(), 3, "1ch → replicate to RGB");
        assert_eq!(rgb[0], rgb[1], "R and G must be equal");
        assert_eq!(rgb[1], rgb[2], "G and B must be equal");
    }

    #[test]
    fn test_baked_texture_to_rgb_4ch() {
        let mut t = BakedTexture::new(1, 1, 4);
        t.set_pixel(0, 0, &[1.0, 0.5, 0.0, 1.0]);
        let rgb = baked_texture_to_rgb(&t).expect("must succeed for 4ch");
        assert_eq!(rgb.len(), 3, "4ch → only first 3 used");
        assert_eq!(rgb[0], 255);
        assert_eq!(rgb[1], 128);
        assert_eq!(rgb[2], 0);
    }

    #[test]
    fn test_baked_texture_to_rgb_invalid_channels() {
        let t = BakedTexture::new(2, 2, 2);
        let res = baked_texture_to_rgb(&t);
        assert!(matches!(res, Err(BakeError::InvalidChannels(2))));
    }

    // -----------------------------------------------------------------------
    // format_bake_stats
    // -----------------------------------------------------------------------

    #[test]
    fn test_format_bake_stats_non_empty() {
        let t = BakedTexture::new(8, 8, 3);
        let s = format_bake_stats(&t);
        assert!(!s.is_empty(), "stats string must not be empty");
        assert!(s.contains("8×8"), "must contain dimensions");
    }

    #[test]
    fn test_format_bake_stats_coverage_zero() {
        let t = BakedTexture::new(8, 8, 3);
        let s = format_bake_stats(&t);
        assert!(s.contains("0.0%"), "coverage must show 0%: {s}");
    }

    // -----------------------------------------------------------------------
    // Error cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_error_face_count_mismatch_display() {
        let err = BakeError::FaceCountMismatch {
            n_faces: 10,
            n_uv_faces: 5,
        };
        let s = format!("{err}");
        assert!(s.contains("10") && s.contains('5'));
    }

    #[test]
    fn test_error_invalid_param_display() {
        let err = BakeError::InvalidParam("test reason".to_string());
        let s = format!("{err}");
        assert!(s.contains("test reason"));
    }

    #[test]
    fn test_error_empty_mesh_display() {
        let err = BakeError::EmptyMesh;
        assert!(!format!("{err}").is_empty());
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_single_face_mesh_succeeds() {
        let verts = [[0.0f32, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]];
        let faces = [[0u32, 1, 2]];
        let uvs: [UvCoord; 3] = [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]];
        let attrs: Vec<Vec<f32>> = vec![vec![0.5; 3]; 3];
        let cfg = BakeConfig {
            width: 16,
            height: 16,
            channels: 3,
            padding: 0,
            background: vec![0.0; 3],
        };
        let tex =
            bake_attribute(&verts, &faces, &attrs, &uvs, &cfg).expect("single face must succeed");
        assert!(tex.coverage() > 0.0);
    }

    #[test]
    fn test_uv_at_corners_does_not_panic() {
        let verts = [[0.0f32; 3]; 4];
        let faces = [[0u32, 1, 2], [0, 2, 3]];
        let uvs: [UvCoord; 4] = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let attrs: Vec<Vec<f32>> = vec![vec![1.0; 3]; 4];
        let cfg = BakeConfig {
            width: 8,
            height: 8,
            channels: 3,
            padding: 0,
            background: vec![0.0; 3],
        };
        let res = bake_attribute(&verts, &faces, &attrs, &uvs, &cfg);
        assert!(res.is_ok());
    }

    #[test]
    fn test_degenerate_triangle_skipped() {
        // All three UV coords are the same → degenerate, no panic.
        let verts = [[0.0f32; 3]; 3];
        let faces = [[0u32, 1, 2]];
        let uvs: [UvCoord; 3] = [[0.5, 0.5]; 3];
        let attrs: Vec<Vec<f32>> = vec![vec![1.0; 3]; 3];
        let cfg = small_config();
        let tex = bake_attribute(&verts, &faces, &attrs, &uvs, &cfg)
            .expect("degenerate triangle must not panic");
        // Degenerate → nothing filled.
        assert_eq!(tex.coverage(), 0.0);
    }

    #[test]
    fn test_background_written_to_unfilled_texels() {
        let verts = [[0.0f32; 3]; 3];
        let faces = [[0u32, 1, 2]];
        // UV concentrated in one corner so most texels stay empty.
        let uvs: [UvCoord; 3] = [[0.0, 0.0], [0.1, 0.0], [0.05, 0.1]];
        let attrs: Vec<Vec<f32>> = vec![vec![1.0; 3]; 3];
        let cfg = BakeConfig {
            width: 16,
            height: 16,
            channels: 3,
            padding: 0,
            background: vec![0.3, 0.5, 0.7],
        };
        let tex = bake_attribute(&verts, &faces, &attrs, &uvs, &cfg).expect("bake must succeed");
        // Pixel (15, 0) is at UV (1, 0) — far from the triangle, which sits
        // near UV (0,0)-(0.1,0.1).
        let p = tex.get_pixel(15, 0);
        assert!(
            (p[0] - 0.3).abs() < 1e-6,
            "R background = 0.3, got {}",
            p[0]
        );
        assert!(
            (p[1] - 0.5).abs() < 1e-6,
            "G background = 0.5, got {}",
            p[1]
        );
        assert!(
            (p[2] - 0.7).abs() < 1e-6,
            "B background = 0.7, got {}",
            p[2]
        );
    }

    // Regression test for the uv_to_pixel V-flip bug: bakes attribute
    // value = v itself, so bake and `uv_texture::UvTextureSampler` must
    // agree on which row corresponds to which v to recover it. A
    // vertical-flip mismatch between the two would sample v=0.1 from the
    // row baked for v=0.9 (and vice versa) -- a large, obvious discrepancy.
    #[test]
    fn test_bake_sample_roundtrip_no_vertical_flip() {
        use crate::uv_texture::{FilterMode, TextureMap, UvTextureSampler, WrapMode};

        let verts = [[0.0f32; 3]; 4];
        let faces = [[0u32, 1, 2], [0, 2, 3]];
        let uvs: [UvCoord; 4] = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let attrs: Vec<Vec<f32>> = uvs.iter().map(|uv| vec![uv[1]]).collect();
        let cfg = BakeConfig {
            width: 32,
            height: 32,
            channels: 1,
            padding: 0,
            background: vec![0.0],
        };
        let tex = bake_attribute(&verts, &faces, &attrs, &uvs, &cfg).expect("bake must succeed");
        let map = TextureMap::new(tex.width, tex.height, tex.channels, tex.data.clone())
            .expect("texture map must be valid");
        let sampler = UvTextureSampler::new(FilterMode::Bilinear, WrapMode::Clamp);

        let low = sampler.sample(&map, [0.5, 0.1])[0];
        let high = sampler.sample(&map, [0.5, 0.9])[0];
        assert!(
            (low - 0.1).abs() < 0.15,
            "sampling at v=0.1 should recover ~0.1, got {low}"
        );
        assert!(
            (high - 0.9).abs() < 0.15,
            "sampling at v=0.9 should recover ~0.9, got {high}"
        );
    }

    #[test]
    fn test_bake_attribute_background_mismatch_error() {
        let verts = [[0.0f32; 3]; 3];
        let faces = [[0u32, 1, 2]];
        let uvs: [UvCoord; 3] = [[0.0, 0.0]; 3];
        let attrs: Vec<Vec<f32>> = vec![vec![0.0; 3]; 3];
        let cfg = BakeConfig {
            width: 8,
            height: 8,
            channels: 3,
            padding: 0,
            background: vec![0.0, 0.0], // wrong length
        };
        let res = bake_attribute(&verts, &faces, &attrs, &uvs, &cfg);
        assert!(matches!(res, Err(BakeError::InvalidParam(_))));
    }

    #[test]
    fn test_padding_config_applied_in_bake() {
        let (verts, faces, uvs) = single_triangle();
        let attrs: Vec<Vec<f32>> = vec![vec![0.5; 3]; 3];
        let cfg = BakeConfig {
            width: 16,
            height: 16,
            channels: 3,
            padding: 3,
            background: vec![0.0; 3],
        };
        let tex_with_pad = bake_attribute(&verts, &faces, &attrs, &uvs, &cfg)
            .expect("bake with padding must succeed");
        let cfg_no_pad = BakeConfig {
            padding: 0,
            ..cfg.clone()
        };
        let tex_no_pad = bake_attribute(&verts, &faces, &attrs, &uvs, &cfg_no_pad)
            .expect("bake without padding must succeed");
        // Padded texture should have at least as much coverage as unpadded.
        assert!(
            tex_with_pad.coverage() >= tex_no_pad.coverage(),
            "padding should increase or equal coverage: {} vs {}",
            tex_with_pad.coverage(),
            tex_no_pad.coverage()
        );
    }

    #[test]
    fn test_face_uv_areas_multiple_faces() {
        // Two identical right triangles.
        let uvs: &[UvCoord] = &[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let faces: &[[u32; 3]] = &[[0, 1, 2], [1, 3, 2]];
        let areas = compute_face_uv_areas(faces, uvs);
        assert_eq!(areas.len(), 2);
        assert!((areas[0] - 0.5).abs() < 1e-5);
        assert!((areas[1] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_baked_texture_clone() {
        let mut t = BakedTexture::new(4, 4, 3);
        t.set_pixel(0, 0, &[1.0, 0.5, 0.0]);
        let t2 = t.clone();
        assert_eq!(t2.data, t.data);
        assert!(t2.is_filled(0, 0));
    }

    #[test]
    fn test_triangle_uv_struct() {
        let tri = TriangleUv {
            uv: [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]],
        };
        assert!((tri.uv[0][0]).abs() < 1e-7);
        assert!((tri.uv[1][0] - 1.0).abs() < 1e-7);
    }

    #[test]
    fn test_compute_uv_mask_full_coverage() {
        // Two triangles that cover the full UV square.
        let uvs: &[UvCoord] = &[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let faces: &[[u32; 3]] = &[[0, 1, 2], [0, 2, 3]];
        let mask = compute_uv_mask(faces, uvs, 8, 8);
        let covered = mask.iter().filter(|&&v| v > 0.0).count();
        // Most pixels should be covered (two triangles tiling the unit square
        // covers ≥ 45 of the 64 texels even accounting for sub-pixel gaps at
        // edges).
        assert!(
            covered >= 45,
            "full-square triangles should cover most texels, got {covered}"
        );
    }
}
