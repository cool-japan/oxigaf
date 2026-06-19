//! Face atlas packing for FLAME head model regions.
//!
//! This module provides tools to pack multiple face-region textures into a
//! single square texture atlas for efficient GPU rendering. It is distinct from
//! per-mesh UV coordinates (see [`crate::uv`]) — it handles atlas layout and
//! sampling.
//!
//! ## Highlights
//!
//! - [`FaceAtlas`]: A packed atlas mapping region IDs to [`AtlasRect`]s.
//! - [`AtlasConfig`]: Configuration for atlas size, padding, and power-of-two
//!   constraints.
//! - [`create_flame_face_atlas`]: Convenience constructor for the nine standard
//!   FLAME face regions.
//! - [`pack_regions`]: Pack arbitrary `(width, height, id, name)` tuples via a
//!   first-fit decreasing-height shelf algorithm.
//! - Pixel-buffer helpers: [`rasterize_atlas_layout`], [`blit_into_atlas`],
//!   [`extract_from_atlas`].

use thiserror::Error;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur when working with a face atlas.
#[derive(Debug, Error, PartialEq)]
pub enum FaceAtlasError {
    /// The configuration parameters are invalid.
    #[error("Invalid atlas config: {0}")]
    InvalidConfig(String),

    /// The atlas is too small to hold the requested regions.
    #[error("Packing failed: atlas too small for {count} regions")]
    PackingFailed {
        /// Number of regions that could not be packed.
        count: usize,
    },

    /// No region with the given ID exists.
    #[error("Region not found: {id}")]
    RegionNotFound {
        /// The requested region ID.
        id: usize,
    },

    /// Source texture dimensions do not match the region dimensions.
    #[error("Texture size mismatch: expected {expected}x{expected}, got {w}x{h}")]
    SizeMismatch {
        /// Expected square side length.
        expected: usize,
        /// Actual width.
        w: usize,
        /// Actual height.
        h: usize,
    },

    /// A dimension-related computation error.
    #[error("Dimension error: {0}")]
    DimensionError(String),

    /// The atlas contains no regions.
    #[error("Empty atlas")]
    EmptyAtlas,
}

// ---------------------------------------------------------------------------
// AtlasRect
// ---------------------------------------------------------------------------

/// A rectangular region within the atlas, in pixel coordinates.
///
/// `x` and `y` are the top-left corner; `width` and `height` describe the
/// inner extent of the region (excluding any padding).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AtlasRect {
    /// Left edge (pixels).
    pub x: usize,
    /// Top edge (pixels).
    pub y: usize,
    /// Width (pixels, inner).
    pub width: usize,
    /// Height (pixels, inner).
    pub height: usize,
}

impl AtlasRect {
    /// Create a new [`AtlasRect`].
    #[inline]
    #[must_use]
    pub fn new(x: usize, y: usize, width: usize, height: usize) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// Area in pixels (inner extent).
    #[inline]
    #[must_use]
    pub fn area(&self) -> usize {
        self.width * self.height
    }

    /// Returns `true` if pixel `(px, py)` lies strictly inside this rect.
    #[inline]
    #[must_use]
    pub fn contains(&self, px: usize, py: usize) -> bool {
        px >= self.x && px < self.right() && py >= self.y && py < self.bottom()
    }

    /// Minimum UV coordinates for this rect within an atlas of `atlas_w × atlas_h`.
    ///
    /// Returns `(u_min, v_min)` where `u = x / atlas_w`, `v = y / atlas_h`.
    #[inline]
    #[must_use]
    pub fn uv_min(&self, atlas_w: usize, atlas_h: usize) -> (f32, f32) {
        (
            self.x as f32 / atlas_w as f32,
            self.y as f32 / atlas_h as f32,
        )
    }

    /// Maximum UV coordinates for this rect within an atlas of `atlas_w × atlas_h`.
    ///
    /// Returns `(u_max, v_max)` where `u = (x+w) / atlas_w`, `v = (y+h) / atlas_h`.
    #[inline]
    #[must_use]
    pub fn uv_max(&self, atlas_w: usize, atlas_h: usize) -> (f32, f32) {
        (
            (self.x + self.width) as f32 / atlas_w as f32,
            (self.y + self.height) as f32 / atlas_h as f32,
        )
    }

    /// Returns `true` if this rect overlaps (touches or intersects) `other`.
    #[inline]
    #[must_use]
    pub fn intersects(&self, other: &AtlasRect) -> bool {
        self.x < other.right()
            && other.x < self.right()
            && self.y < other.bottom()
            && other.y < self.bottom()
    }

    /// Right edge (`x + width`).
    #[inline]
    #[must_use]
    pub fn right(&self) -> usize {
        self.x + self.width
    }

    /// Bottom edge (`y + height`).
    #[inline]
    #[must_use]
    pub fn bottom(&self) -> usize {
        self.y + self.height
    }
}

// ---------------------------------------------------------------------------
// AtlasRegion
// ---------------------------------------------------------------------------

/// A named region packed into the atlas (e.g., `"left_eye"`, `"nose"`).
#[derive(Debug, Clone)]
pub struct AtlasRegion {
    /// Unique identifier for the region.
    pub id: usize,
    /// Human-readable name (e.g., `"forehead"`, `"left_eye"`).
    pub name: String,
    /// Inner rectangle in atlas pixel space.
    pub rect: AtlasRect,
    /// Padding (pixels) that surrounds this region on all sides.
    pub padding: usize,
}

// ---------------------------------------------------------------------------
// AtlasConfig
// ---------------------------------------------------------------------------

/// Configuration for a [`FaceAtlas`].
#[derive(Debug, Clone)]
pub struct AtlasConfig {
    /// Side length of the square atlas in pixels.  Must be a power of two and
    /// greater than zero. Default: `1024`.
    pub atlas_size: usize,
    /// Padding (pixels) between regions. Default: `4`.
    pub default_padding: usize,
    /// When `true`, each region's width and height are rounded up to the next
    /// power of two before packing. Default: `true`.
    pub power_of_two: bool,
}

impl Default for AtlasConfig {
    fn default() -> Self {
        Self {
            atlas_size: 1024,
            default_padding: 4,
            power_of_two: true,
        }
    }
}

impl AtlasConfig {
    /// Validate the configuration, returning an error if any field is invalid.
    ///
    /// Rules:
    /// - `atlas_size` must be > 0 and a power of two.
    /// - `default_padding` must be < `atlas_size / 4`.
    ///
    /// # Errors
    ///
    /// Returns an error if the operation fails.
    pub fn validate(&self) -> Result<(), FaceAtlasError> {
        if self.atlas_size == 0 || !self.atlas_size.is_power_of_two() {
            return Err(FaceAtlasError::InvalidConfig(format!(
                "atlas_size must be a positive power of two, got {}",
                self.atlas_size
            )));
        }
        if self.default_padding >= self.atlas_size / 4 {
            return Err(FaceAtlasError::InvalidConfig(format!(
                "default_padding ({}) must be < atlas_size/4 ({})",
                self.default_padding,
                self.atlas_size / 4
            )));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// FaceAtlas (shelf packer state)
// ---------------------------------------------------------------------------

/// A packed face atlas: maps region IDs to atlas rectangles.
///
/// Regions are placed using a simple left-to-right shelf packer. When a region
/// does not fit on the current shelf it starts a new one below.
#[derive(Debug, Clone)]
pub struct FaceAtlas {
    /// Side length of the square atlas (pixels).
    pub atlas_size: usize,
    /// All packed regions.
    pub regions: Vec<AtlasRegion>,
    /// Configuration used to create this atlas.
    pub config: AtlasConfig,

    // --- internal shelf-packer state ---
    /// Horizontal cursor on the current shelf (pixels).
    current_x: usize,
    /// Top of the current shelf (pixels).
    current_y: usize,
    /// Height of the tallest region on the current shelf (pixels, padded).
    shelf_height: usize,
}

impl FaceAtlas {
    /// Create a new, empty [`FaceAtlas`] from `config`.
    ///
    /// Returns [`FaceAtlasError::InvalidConfig`] if the configuration fails
    /// validation.
    ///
    /// # Errors
    ///
    /// Returns an error if the operation fails.
    pub fn new(config: AtlasConfig) -> Result<Self, FaceAtlasError> {
        config.validate()?;
        let atlas_size = config.atlas_size;
        Ok(Self {
            atlas_size,
            regions: Vec::new(),
            config,
            current_x: 0,
            current_y: 0,
            shelf_height: 0,
        })
    }

    /// Add a region of `width × height` pixels to the atlas, returning its
    /// [`AtlasRect`] on success.
    ///
    /// If `config.power_of_two` is set, both dimensions are rounded up to the
    /// next power of two before placement. The stored [`AtlasRect`] reflects
    /// the (possibly rounded) inner dimensions; padding expands around it.
    ///
    /// Returns [`FaceAtlasError::PackingFailed`] when the region does not fit.
    ///
    /// # Errors
    ///
    /// Returns an error if the operation fails.
    pub fn add_region(
        &mut self,
        id: usize,
        name: impl Into<String>,
        width: usize,
        height: usize,
    ) -> Result<AtlasRect, FaceAtlasError> {
        let pad = self.config.default_padding;

        // Optionally round dimensions up to next power of two.
        let inner_w = if self.config.power_of_two {
            next_power_of_two(width)
        } else {
            width
        };
        let inner_h = if self.config.power_of_two {
            next_power_of_two(height)
        } else {
            height
        };

        // Padded (slot) dimensions.
        let slot_w = inner_w + 2 * pad;
        let slot_h = inner_h + 2 * pad;

        // Sanity: region must fit at all.
        if slot_w > self.atlas_size || slot_h > self.atlas_size {
            return Err(FaceAtlasError::PackingFailed { count: 1 });
        }

        // Try to place on the current shelf.
        if self.current_x + slot_w > self.atlas_size {
            // Start a new shelf below the current one.
            self.current_y += self.shelf_height;
            self.current_x = 0;
            self.shelf_height = 0;
        }

        // Check vertical overflow.
        if self.current_y + slot_h > self.atlas_size {
            return Err(FaceAtlasError::PackingFailed {
                count: self.regions.len() + 1,
            });
        }

        // Place the region (inner rect is inset by `pad`).
        let rect = AtlasRect::new(self.current_x + pad, self.current_y + pad, inner_w, inner_h);

        // Advance cursor and update shelf height.
        self.current_x += slot_w;
        if slot_h > self.shelf_height {
            self.shelf_height = slot_h;
        }

        self.regions.push(AtlasRegion {
            id,
            name: name.into(),
            rect,
            padding: pad,
        });

        Ok(rect)
    }

    /// Look up a region by ID.
    #[must_use]
    pub fn get_region(&self, id: usize) -> Option<&AtlasRegion> {
        self.regions.iter().find(|r| r.id == id)
    }

    /// Number of packed regions.
    #[inline]
    #[must_use]
    pub fn num_regions(&self) -> usize {
        self.regions.len()
    }

    /// Fraction of atlas area used by packed regions (`0.0 ..= 1.0`).
    #[must_use]
    pub fn coverage(&self) -> f32 {
        let total = (self.atlas_size * self.atlas_size) as f32;
        let used: usize = self.regions.iter().map(|r| r.rect.area()).sum();
        used as f32 / total
    }

    /// Return `(uv_min, uv_max)` for the region with `id`, or `None` if not
    /// found.
    #[must_use]
    pub fn uv_for_region(&self, id: usize) -> Option<((f32, f32), (f32, f32))> {
        let region = self.get_region(id)?;
        let aw = self.atlas_size;
        let ah = self.atlas_size;
        Some((region.rect.uv_min(aw, ah), region.rect.uv_max(aw, ah)))
    }

    /// Transform a local UV in `[0, 1]²` within a region to a global atlas UV
    /// in `[0, 1]²`.
    ///
    /// Returns `None` if `id` is not found.
    #[must_use]
    pub fn local_to_atlas_uv(&self, id: usize, local_u: f32, local_v: f32) -> Option<(f32, f32)> {
        let ((u0, v0), (u1, v1)) = self.uv_for_region(id)?;
        let au = u0 + local_u * (u1 - u0);
        let av = v0 + local_v * (v1 - v0);
        Some((au, av))
    }

    /// Determine which region an atlas UV `(atlas_u, atlas_v) ∈ [0, 1]²` falls
    /// in, and return `(region_id, local_u, local_v)`.
    ///
    /// Returns `None` if the UV does not fall inside any packed region.
    #[must_use]
    pub fn atlas_to_local_uv(&self, atlas_u: f32, atlas_v: f32) -> Option<(usize, f32, f32)> {
        let px = (atlas_u * self.atlas_size as f32) as usize;
        let py = (atlas_v * self.atlas_size as f32) as usize;

        for region in &self.regions {
            if region.rect.contains(px, py) {
                let r = &region.rect;
                let local_u = (px - r.x) as f32 / r.width as f32;
                let local_v = (py - r.y) as f32 / r.height as f32;
                return Some((region.id, local_u, local_v));
            }
        }
        None
    }
}

// ---------------------------------------------------------------------------
// create_flame_face_atlas
// ---------------------------------------------------------------------------

/// Create a standard FLAME face atlas with nine predefined regions.
///
/// Regions (ID → name):
/// 0 → `forehead`, 1 → `left_eye`, 2 → `right_eye`, 3 → `nose`,
/// 4 → `left_cheek`, 5 → `right_cheek`, 6 → `upper_lip`, 7 → `lower_lip`,
/// 8 → `chin`.
///
/// Each region is 128 × 128 pixels with the atlas's default padding.
///
/// # Errors
///
/// Returns an error if the operation fails.
pub fn create_flame_face_atlas(config: AtlasConfig) -> Result<FaceAtlas, FaceAtlasError> {
    let mut atlas = FaceAtlas::new(config)?;

    let regions: &[(usize, &str)] = &[
        (0, "forehead"),
        (1, "left_eye"),
        (2, "right_eye"),
        (3, "nose"),
        (4, "left_cheek"),
        (5, "right_cheek"),
        (6, "upper_lip"),
        (7, "lower_lip"),
        (8, "chin"),
    ];

    for &(id, name) in regions {
        atlas.add_region(id, name, 128, 128)?;
    }

    Ok(atlas)
}

// ---------------------------------------------------------------------------
// rasterize_atlas_layout
// ---------------------------------------------------------------------------

/// Rasterize the atlas layout to an RGBA pixel buffer for visualization.
///
/// Each region is filled with a unique colour drawn from a fixed palette;
/// unoccupied pixels are black. Returns an RGBA buffer of
/// `atlas_size × atlas_size × 4` bytes.
#[must_use]
pub fn rasterize_atlas_layout(atlas: &FaceAtlas) -> Vec<u8> {
    // 9-colour palette (RGBA).
    const PALETTE: &[(u8, u8, u8)] = &[
        (220, 80, 80),
        (80, 220, 80),
        (80, 80, 220),
        (220, 220, 80),
        (220, 80, 220),
        (80, 220, 220),
        (200, 140, 80),
        (140, 80, 200),
        (80, 200, 140),
    ];

    let size = atlas.atlas_size;
    let mut pixels = vec![0u8; size * size * 4];

    for (palette_idx, region) in atlas.regions.iter().enumerate() {
        let (r, g, b) = PALETTE[palette_idx % PALETTE.len()];
        let rect = &region.rect;

        let x_end = rect.right().min(size);
        let y_end = rect.bottom().min(size);

        for py in rect.y..y_end {
            for px in rect.x..x_end {
                let offset = (py * size + px) * 4;
                pixels[offset] = r;
                pixels[offset + 1] = g;
                pixels[offset + 2] = b;
                pixels[offset + 3] = 255;
            }
        }
    }

    pixels
}

// ---------------------------------------------------------------------------
// blit_into_atlas
// ---------------------------------------------------------------------------

/// Blit an RGBA source texture into an atlas pixel buffer at the region's rect.
///
/// `src` must be an RGBA buffer of `src_w × src_h × 4` bytes.  If `src_w` or
/// `src_h` do not match `region.rect.width` / `region.rect.height` the
/// function returns [`FaceAtlasError::SizeMismatch`].
///
/// # Errors
///
/// Returns an error if the operation fails.
pub fn blit_into_atlas(
    atlas_pixels: &mut [u8],
    atlas_size: usize,
    region: &AtlasRegion,
    src: &[u8],
    src_w: usize,
    src_h: usize,
) -> Result<(), FaceAtlasError> {
    let rect = &region.rect;

    if src_w != rect.width || src_h != rect.height {
        return Err(FaceAtlasError::SizeMismatch {
            expected: rect.width,
            w: src_w,
            h: src_h,
        });
    }

    let expected_len = src_w * src_h * 4;
    if src.len() != expected_len {
        return Err(FaceAtlasError::DimensionError(format!(
            "src buffer length {} != expected {}",
            src.len(),
            expected_len
        )));
    }

    for row in 0..src_h {
        let src_start = row * src_w * 4;
        let dst_y = rect.y + row;
        if dst_y >= atlas_size {
            break;
        }
        let dst_start = (dst_y * atlas_size + rect.x) * 4;
        let dst_end = dst_start + src_w * 4;
        if dst_end > atlas_pixels.len() {
            return Err(FaceAtlasError::DimensionError(format!(
                "blit would exceed atlas buffer at row {row}"
            )));
        }
        atlas_pixels[dst_start..dst_end].copy_from_slice(&src[src_start..src_start + src_w * 4]);
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// extract_from_atlas
// ---------------------------------------------------------------------------

/// Extract the RGBA pixels for a region from an atlas pixel buffer.
///
/// Returns an RGBA buffer of `region.rect.width × region.rect.height × 4`
/// bytes, or a [`FaceAtlasError::DimensionError`] if the atlas buffer is
/// smaller than expected.
///
/// # Errors
///
/// Returns an error if the operation fails.
pub fn extract_from_atlas(
    atlas_pixels: &[u8],
    atlas_size: usize,
    region: &AtlasRegion,
) -> Result<Vec<u8>, FaceAtlasError> {
    let rect = &region.rect;
    let expected_atlas_len = atlas_size * atlas_size * 4;
    if atlas_pixels.len() < expected_atlas_len {
        return Err(FaceAtlasError::DimensionError(format!(
            "atlas buffer length {} < expected {}",
            atlas_pixels.len(),
            expected_atlas_len
        )));
    }

    let mut out = Vec::with_capacity(rect.width * rect.height * 4);
    for row in 0..rect.height {
        let src_y = rect.y + row;
        let src_start = (src_y * atlas_size + rect.x) * 4;
        let src_end = src_start + rect.width * 4;
        if src_end > atlas_pixels.len() {
            return Err(FaceAtlasError::DimensionError(format!(
                "extract would read past atlas buffer at row {row}"
            )));
        }
        out.extend_from_slice(&atlas_pixels[src_start..src_end]);
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// pack_regions
// ---------------------------------------------------------------------------

/// Pack an arbitrary collection of `(width, height, id, name)` tuples into a
/// new [`FaceAtlas`].
///
/// Regions are sorted by **decreasing height** (first-fit decreasing-height
/// strategy) before being handed to the shelf packer, which improves
/// utilisation.
///
/// Returns [`FaceAtlasError::PackingFailed`] if any region does not fit.
///
/// # Errors
///
/// Returns an error if the operation fails.
pub fn pack_regions(
    regions: &[(usize, usize, usize, String)],
    config: AtlasConfig,
) -> Result<FaceAtlas, FaceAtlasError> {
    // Sort by decreasing height for better bin-packing efficiency.
    let mut sorted: Vec<(usize, usize, usize, &String)> = regions
        .iter()
        .map(|(w, h, id, name)| (*w, *h, *id, name))
        .collect();
    sorted.sort_by_key(|b| std::cmp::Reverse(b.1));

    let mut atlas = FaceAtlas::new(config)?;
    for (w, h, id, name) in sorted {
        atlas.add_region(id, name.clone(), w, h)?;
    }
    Ok(atlas)
}

// ---------------------------------------------------------------------------
// next_power_of_two
// ---------------------------------------------------------------------------

/// Return the smallest power of two that is ≥ `n`.
///
/// `next_power_of_two(0)` returns `1`.
#[inline]
#[must_use]
pub fn next_power_of_two(n: usize) -> usize {
    if n <= 1 {
        return 1;
    }
    n.next_power_of_two()
}

// ---------------------------------------------------------------------------
// AtlasStats / compute_atlas_stats
// ---------------------------------------------------------------------------

/// Statistics for a packed [`FaceAtlas`].
#[derive(Debug, Clone)]
pub struct AtlasStats {
    /// Number of packed regions.
    pub num_regions: usize,
    /// Sum of all inner region areas (pixels²).
    pub total_area: usize,
    /// Fraction of atlas area occupied by regions (`0.0 ..= 1.0`).
    pub coverage_fraction: f32,
    /// Atlas area not covered by any region.
    pub wasted_area: usize,
    /// Area of the largest region.
    pub largest_region_area: usize,
    /// Area of the smallest region.
    pub smallest_region_area: usize,
}

/// Compute statistics for a [`FaceAtlas`].
#[must_use]
pub fn compute_atlas_stats(atlas: &FaceAtlas) -> AtlasStats {
    let num_regions = atlas.num_regions();
    let atlas_total = atlas.atlas_size * atlas.atlas_size;

    if num_regions == 0 {
        return AtlasStats {
            num_regions: 0,
            total_area: 0,
            coverage_fraction: 0.0,
            wasted_area: atlas_total,
            largest_region_area: 0,
            smallest_region_area: 0,
        };
    }

    let mut total_area = 0usize;
    let mut largest = 0usize;
    let mut smallest = usize::MAX;

    for region in &atlas.regions {
        let area = region.rect.area();
        total_area += area;
        if area > largest {
            largest = area;
        }
        if area < smallest {
            smallest = area;
        }
    }

    let coverage_fraction = total_area as f32 / atlas_total as f32;
    let wasted_area = atlas_total.saturating_sub(total_area);

    AtlasStats {
        num_regions,
        total_area,
        coverage_fraction,
        wasted_area,
        largest_region_area: largest,
        smallest_region_area: smallest,
    }
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

    fn default_config() -> AtlasConfig {
        AtlasConfig::default()
    }

    // -----------------------------------------------------------------------
    // next_power_of_two
    // -----------------------------------------------------------------------

    #[test]
    fn test_next_power_of_two_zero() {
        assert_eq!(next_power_of_two(0), 1);
    }

    #[test]
    fn test_next_power_of_two_one() {
        assert_eq!(next_power_of_two(1), 1);
    }

    #[test]
    fn test_next_power_of_two_exact_powers() {
        for exp in 1u32..12 {
            let n = 1usize << exp;
            assert_eq!(next_power_of_two(n), n, "failed for {n}");
        }
    }

    #[test]
    fn test_next_power_of_two_non_power() {
        assert_eq!(next_power_of_two(3), 4);
        assert_eq!(next_power_of_two(5), 8);
        assert_eq!(next_power_of_two(100), 128);
        assert_eq!(next_power_of_two(127), 128);
        assert_eq!(next_power_of_two(129), 256);
    }

    // -----------------------------------------------------------------------
    // AtlasRect
    // -----------------------------------------------------------------------

    #[test]
    fn test_atlas_rect_area() {
        let r = AtlasRect::new(0, 0, 10, 20);
        assert_eq!(r.area(), 200);
    }

    #[test]
    fn test_atlas_rect_area_zero() {
        let r = AtlasRect::new(0, 0, 0, 5);
        assert_eq!(r.area(), 0);
    }

    #[test]
    fn test_atlas_rect_right_bottom() {
        let r = AtlasRect::new(3, 7, 10, 5);
        assert_eq!(r.right(), 13);
        assert_eq!(r.bottom(), 12);
    }

    #[test]
    fn test_atlas_rect_contains() {
        let r = AtlasRect::new(10, 10, 20, 20);
        assert!(r.contains(10, 10));
        assert!(r.contains(15, 15));
        assert!(r.contains(29, 29));
        assert!(!r.contains(30, 10)); // right boundary
        assert!(!r.contains(10, 30)); // bottom boundary
        assert!(!r.contains(9, 10)); // left of rect
    }

    #[test]
    fn test_atlas_rect_uv_min_max() {
        let r = AtlasRect::new(0, 0, 512, 512);
        let (u0, v0) = r.uv_min(1024, 1024);
        let (u1, v1) = r.uv_max(1024, 1024);
        assert!((u0 - 0.0).abs() < 1e-6);
        assert!((v0 - 0.0).abs() < 1e-6);
        assert!((u1 - 0.5).abs() < 1e-6);
        assert!((v1 - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_atlas_rect_uv_offset() {
        let r = AtlasRect::new(256, 0, 128, 128);
        let (u0, _v0) = r.uv_min(1024, 1024);
        let (u1, _v1) = r.uv_max(1024, 1024);
        assert!((u0 - 0.25).abs() < 1e-6);
        assert!((u1 - 0.375).abs() < 1e-6);
    }

    #[test]
    fn test_atlas_rect_intersects_overlap() {
        let a = AtlasRect::new(0, 0, 10, 10);
        let b = AtlasRect::new(5, 5, 10, 10);
        assert!(a.intersects(&b));
        assert!(b.intersects(&a));
    }

    #[test]
    fn test_atlas_rect_intersects_adjacent() {
        // Touching at the edge — NOT overlapping (exclusive right/bottom).
        let a = AtlasRect::new(0, 0, 10, 10);
        let b = AtlasRect::new(10, 0, 10, 10);
        assert!(!a.intersects(&b));
    }

    #[test]
    fn test_atlas_rect_intersects_disjoint() {
        let a = AtlasRect::new(0, 0, 5, 5);
        let b = AtlasRect::new(10, 10, 5, 5);
        assert!(!a.intersects(&b));
    }

    // -----------------------------------------------------------------------
    // AtlasConfig::validate
    // -----------------------------------------------------------------------

    #[test]
    fn test_config_validate_ok() {
        let cfg = AtlasConfig {
            atlas_size: 1024,
            default_padding: 4,
            power_of_two: true,
        };
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_validate_zero_size() {
        let cfg = AtlasConfig {
            atlas_size: 0,
            ..Default::default()
        };
        assert!(matches!(
            cfg.validate(),
            Err(FaceAtlasError::InvalidConfig(_))
        ));
    }

    #[test]
    fn test_config_validate_not_power_of_two() {
        let cfg = AtlasConfig {
            atlas_size: 1000,
            ..Default::default()
        };
        assert!(matches!(
            cfg.validate(),
            Err(FaceAtlasError::InvalidConfig(_))
        ));
    }

    #[test]
    fn test_config_validate_padding_too_large() {
        // atlas_size=256, atlas_size/4=64, padding must be < 64.
        let cfg = AtlasConfig {
            atlas_size: 256,
            default_padding: 64,
            power_of_two: true,
        };
        assert!(matches!(
            cfg.validate(),
            Err(FaceAtlasError::InvalidConfig(_))
        ));
    }

    #[test]
    fn test_config_validate_padding_at_limit() {
        // padding = atlas_size/4 - 1 is valid; atlas_size/4 is not.
        let cfg_ok = AtlasConfig {
            atlas_size: 256,
            default_padding: 63,
            power_of_two: true,
        };
        assert!(cfg_ok.validate().is_ok());
        let cfg_bad = AtlasConfig {
            atlas_size: 256,
            default_padding: 64,
            power_of_two: true,
        };
        assert!(cfg_bad.validate().is_err());
    }

    // -----------------------------------------------------------------------
    // FaceAtlas::new
    // -----------------------------------------------------------------------

    #[test]
    fn test_face_atlas_new_ok() {
        let atlas = FaceAtlas::new(default_config());
        assert!(atlas.is_ok());
        let atlas = atlas.expect("just checked");
        assert_eq!(atlas.num_regions(), 0);
        assert_eq!(atlas.atlas_size, 1024);
    }

    #[test]
    fn test_face_atlas_new_invalid_config() {
        let cfg = AtlasConfig {
            atlas_size: 0,
            ..Default::default()
        };
        assert!(FaceAtlas::new(cfg).is_err());
    }

    // -----------------------------------------------------------------------
    // FaceAtlas::add_region
    // -----------------------------------------------------------------------

    #[test]
    fn test_add_single_region() {
        let mut atlas = FaceAtlas::new(default_config()).expect("valid config");
        let rect = atlas.add_region(0, "test", 64, 64).expect("should fit");
        assert_eq!(atlas.num_regions(), 1);
        // Inner rect must be positive and within atlas bounds.
        assert!(rect.width > 0);
        assert!(rect.right() <= atlas.atlas_size);
        assert!(rect.bottom() <= atlas.atlas_size);
    }

    #[test]
    fn test_add_multiple_regions_no_overlap() {
        let mut atlas = FaceAtlas::new(default_config()).expect("valid config");
        let mut rects = Vec::new();
        for i in 0..5 {
            let r = atlas
                .add_region(i, format!("region_{i}"), 64, 64)
                .expect("should fit");
            rects.push(r);
        }
        // No two rects should intersect.
        for i in 0..rects.len() {
            for j in (i + 1)..rects.len() {
                assert!(
                    !rects[i].intersects(&rects[j]),
                    "regions {i} and {j} overlap"
                );
            }
        }
    }

    #[test]
    fn test_add_region_fills_atlas_then_fails() {
        // Use a tiny atlas that can hold exactly one large region.
        let cfg = AtlasConfig {
            atlas_size: 64,
            default_padding: 0,
            power_of_two: false,
        };
        let mut atlas = FaceAtlas::new(cfg).expect("valid config");
        atlas
            .add_region(0, "first", 64, 64)
            .expect("first should fit");
        let result = atlas.add_region(1, "second", 64, 64);
        assert!(matches!(result, Err(FaceAtlasError::PackingFailed { .. })));
    }

    #[test]
    fn test_add_region_power_of_two_rounding() {
        let cfg = AtlasConfig {
            atlas_size: 1024,
            default_padding: 0,
            power_of_two: true,
        };
        let mut atlas = FaceAtlas::new(cfg).expect("valid config");
        let rect = atlas.add_region(0, "r", 100, 100).expect("should fit");
        // 100 rounds up to 128.
        assert_eq!(rect.width, 128);
        assert_eq!(rect.height, 128);
    }

    #[test]
    fn test_add_region_no_power_of_two() {
        let cfg = AtlasConfig {
            atlas_size: 1024,
            default_padding: 0,
            power_of_two: false,
        };
        let mut atlas = FaceAtlas::new(cfg).expect("valid config");
        let rect = atlas.add_region(0, "r", 100, 100).expect("should fit");
        assert_eq!(rect.width, 100);
        assert_eq!(rect.height, 100);
    }

    // -----------------------------------------------------------------------
    // FaceAtlas::get_region
    // -----------------------------------------------------------------------

    #[test]
    fn test_get_region_found() {
        let mut atlas = FaceAtlas::new(default_config()).expect("valid config");
        atlas.add_region(42, "test", 32, 32).expect("should fit");
        let region = atlas.get_region(42);
        assert!(region.is_some());
        assert_eq!(region.expect("just checked").name, "test");
    }

    #[test]
    fn test_get_region_not_found() {
        let atlas = FaceAtlas::new(default_config()).expect("valid config");
        assert!(atlas.get_region(999).is_none());
    }

    // -----------------------------------------------------------------------
    // FaceAtlas::coverage
    // -----------------------------------------------------------------------

    #[test]
    fn test_coverage_empty() {
        let atlas = FaceAtlas::new(default_config()).expect("valid config");
        assert_eq!(atlas.coverage(), 0.0);
    }

    #[test]
    fn test_coverage_partial() {
        let cfg = AtlasConfig {
            atlas_size: 512,
            default_padding: 0,
            power_of_two: false,
        };
        let mut atlas = FaceAtlas::new(cfg).expect("valid config");
        atlas.add_region(0, "r", 256, 256).expect("should fit");
        let cov = atlas.coverage();
        assert!(cov > 0.0 && cov <= 1.0);
        // 256*256 / (512*512) = 0.25
        let expected = 256.0 * 256.0 / (512.0 * 512.0);
        assert!((cov - expected).abs() < 1e-5, "coverage was {cov}");
    }

    // -----------------------------------------------------------------------
    // uv_for_region / local_to_atlas_uv / atlas_to_local_uv
    // -----------------------------------------------------------------------

    #[test]
    fn test_uv_for_region_exists() {
        let cfg = AtlasConfig {
            atlas_size: 512,
            default_padding: 0,
            power_of_two: false,
        };
        let mut atlas = FaceAtlas::new(cfg).expect("valid config");
        atlas.add_region(0, "r", 256, 256).expect("should fit");
        let uvs = atlas.uv_for_region(0);
        assert!(uvs.is_some());
        let ((u0, v0), (u1, v1)) = uvs.expect("just checked");
        assert!(u0 >= 0.0 && u1 <= 1.0);
        assert!(v0 >= 0.0 && v1 <= 1.0);
        assert!(u1 > u0);
        assert!(v1 > v0);
    }

    #[test]
    fn test_uv_for_region_not_found() {
        let atlas = FaceAtlas::new(default_config()).expect("valid config");
        assert!(atlas.uv_for_region(99).is_none());
    }

    #[test]
    fn test_local_to_atlas_uv_corners() {
        let cfg = AtlasConfig {
            atlas_size: 512,
            default_padding: 0,
            power_of_two: false,
        };
        let mut atlas = FaceAtlas::new(cfg).expect("valid config");
        atlas.add_region(0, "r", 256, 256).expect("should fit");
        let ((u0, v0), (u1, v1)) = atlas.uv_for_region(0).expect("region present");

        let tl = atlas
            .local_to_atlas_uv(0, 0.0, 0.0)
            .expect("region present");
        let br = atlas
            .local_to_atlas_uv(0, 1.0, 1.0)
            .expect("region present");

        assert!((tl.0 - u0).abs() < 1e-6);
        assert!((tl.1 - v0).abs() < 1e-6);
        assert!((br.0 - u1).abs() < 1e-6);
        assert!((br.1 - v1).abs() < 1e-6);
    }

    #[test]
    fn test_local_to_atlas_uv_not_found() {
        let atlas = FaceAtlas::new(default_config()).expect("valid config");
        assert!(atlas.local_to_atlas_uv(99, 0.5, 0.5).is_none());
    }

    #[test]
    fn test_atlas_to_local_uv_hit() {
        let cfg = AtlasConfig {
            atlas_size: 512,
            default_padding: 0,
            power_of_two: false,
        };
        let mut atlas = FaceAtlas::new(cfg).expect("valid config");
        atlas.add_region(0, "r", 256, 256).expect("should fit");

        // Atlas UV at centre of the region (x=128/512=0.25, y=128/512=0.25).
        let result = atlas.atlas_to_local_uv(0.25, 0.25);
        assert!(result.is_some());
        let (id, lu, lv) = result.expect("just checked");
        assert_eq!(id, 0);
        // local UV should be ~0.5 (centre of region).
        assert!((0.0..=1.0).contains(&lu), "lu={lu}");
        assert!((0.0..=1.0).contains(&lv), "lv={lv}");
    }

    #[test]
    fn test_atlas_to_local_uv_miss() {
        let cfg = AtlasConfig {
            atlas_size: 512,
            default_padding: 0,
            power_of_two: false,
        };
        let mut atlas = FaceAtlas::new(cfg).expect("valid config");
        atlas.add_region(0, "r", 10, 10).expect("should fit");
        // UV near bottom-right corner, well outside the small region.
        let result = atlas.atlas_to_local_uv(0.99, 0.99);
        assert!(result.is_none());
    }

    // -----------------------------------------------------------------------
    // create_flame_face_atlas
    // -----------------------------------------------------------------------

    #[test]
    fn test_create_flame_face_atlas_region_count() {
        let atlas = create_flame_face_atlas(default_config()).expect("valid config");
        assert_eq!(atlas.num_regions(), 9);
    }

    #[test]
    fn test_create_flame_face_atlas_all_ids() {
        let atlas = create_flame_face_atlas(default_config()).expect("valid config");
        for id in 0..9 {
            assert!(atlas.get_region(id).is_some(), "region {id} missing");
        }
    }

    #[test]
    fn test_create_flame_face_atlas_coverage_positive() {
        let atlas = create_flame_face_atlas(default_config()).expect("valid config");
        assert!(atlas.coverage() > 0.0);
    }

    #[test]
    fn test_create_flame_face_atlas_region_names() {
        let atlas = create_flame_face_atlas(default_config()).expect("valid config");
        let expected: &[&str] = &[
            "forehead",
            "left_eye",
            "right_eye",
            "nose",
            "left_cheek",
            "right_cheek",
            "upper_lip",
            "lower_lip",
            "chin",
        ];
        for (i, &name) in expected.iter().enumerate() {
            let region = atlas.get_region(i).expect("region present");
            assert_eq!(region.name, name);
        }
    }

    // -----------------------------------------------------------------------
    // rasterize_atlas_layout
    // -----------------------------------------------------------------------

    #[test]
    fn test_rasterize_atlas_layout_buffer_size() {
        let atlas = create_flame_face_atlas(default_config()).expect("valid config");
        let pixels = rasterize_atlas_layout(&atlas);
        let size = atlas.atlas_size;
        assert_eq!(pixels.len(), size * size * 4);
    }

    #[test]
    fn test_rasterize_atlas_layout_has_non_zero() {
        let atlas = create_flame_face_atlas(default_config()).expect("valid config");
        let pixels = rasterize_atlas_layout(&atlas);
        assert!(pixels.iter().any(|&b| b != 0));
    }

    #[test]
    fn test_rasterize_atlas_layout_empty_atlas() {
        let atlas = FaceAtlas::new(default_config()).expect("valid config");
        let pixels = rasterize_atlas_layout(&atlas);
        let size = atlas.atlas_size;
        assert_eq!(pixels.len(), size * size * 4);
        // Empty atlas → all black.
        assert!(pixels.iter().all(|&b| b == 0));
    }

    // -----------------------------------------------------------------------
    // blit_into_atlas
    // -----------------------------------------------------------------------

    #[test]
    fn test_blit_into_atlas_valid() {
        let cfg = AtlasConfig {
            atlas_size: 64,
            default_padding: 0,
            power_of_two: false,
        };
        let mut atlas = FaceAtlas::new(cfg).expect("valid config");
        atlas.add_region(0, "r", 8, 8).expect("should fit");
        let region = atlas.get_region(0).expect("just added").clone();

        let src = vec![255u8; 8 * 8 * 4];
        let mut atlas_pixels = vec![0u8; 64 * 64 * 4];
        blit_into_atlas(&mut atlas_pixels, 64, &region, &src, 8, 8).expect("blit should succeed");

        // Check that some bytes at the region location are 255.
        let offset = (region.rect.y * 64 + region.rect.x) * 4;
        assert_eq!(atlas_pixels[offset], 255);
    }

    #[test]
    fn test_blit_into_atlas_size_mismatch() {
        let cfg = AtlasConfig {
            atlas_size: 64,
            default_padding: 0,
            power_of_two: false,
        };
        let mut atlas = FaceAtlas::new(cfg).expect("valid config");
        atlas.add_region(0, "r", 8, 8).expect("should fit");
        let region = atlas.get_region(0).expect("just added").clone();

        let src = vec![255u8; 16 * 16 * 4];
        let mut atlas_pixels = vec![0u8; 64 * 64 * 4];
        let result = blit_into_atlas(&mut atlas_pixels, 64, &region, &src, 16, 16);
        assert!(matches!(result, Err(FaceAtlasError::SizeMismatch { .. })));
    }

    // -----------------------------------------------------------------------
    // extract_from_atlas
    // -----------------------------------------------------------------------

    #[test]
    fn test_extract_from_atlas_correct_size() {
        let cfg = AtlasConfig {
            atlas_size: 64,
            default_padding: 0,
            power_of_two: false,
        };
        let mut atlas = FaceAtlas::new(cfg).expect("valid config");
        atlas.add_region(0, "r", 8, 8).expect("should fit");
        let region = atlas.get_region(0).expect("just added").clone();

        let atlas_pixels = vec![128u8; 64 * 64 * 4];
        let extracted =
            extract_from_atlas(&atlas_pixels, 64, &region).expect("extract should succeed");

        assert_eq!(extracted.len(), 8 * 8 * 4);
    }

    #[test]
    fn test_extract_from_atlas_roundtrip() {
        let cfg = AtlasConfig {
            atlas_size: 64,
            default_padding: 0,
            power_of_two: false,
        };
        let mut atlas = FaceAtlas::new(cfg).expect("valid config");
        atlas.add_region(0, "r", 8, 8).expect("should fit");
        let region = atlas.get_region(0).expect("just added").clone();

        // Create a distinct source pattern.
        let src: Vec<u8> = (0..(8 * 8 * 4)).map(|i| (i % 256) as u8).collect();
        let mut atlas_pixels = vec![0u8; 64 * 64 * 4];
        blit_into_atlas(&mut atlas_pixels, 64, &region, &src, 8, 8).expect("blit should succeed");
        let extracted =
            extract_from_atlas(&atlas_pixels, 64, &region).expect("extract should succeed");

        assert_eq!(extracted, src);
    }

    // -----------------------------------------------------------------------
    // pack_regions
    // -----------------------------------------------------------------------

    #[test]
    fn test_pack_regions_normal() {
        let regions = vec![
            (32usize, 32usize, 0usize, "a".to_string()),
            (64, 64, 1, "b".to_string()),
            (16, 16, 2, "c".to_string()),
        ];
        let atlas = pack_regions(&regions, default_config()).expect("should fit");
        assert_eq!(atlas.num_regions(), 3);
        // All regions present.
        assert!(atlas.get_region(0).is_some());
        assert!(atlas.get_region(1).is_some());
        assert!(atlas.get_region(2).is_some());
    }

    #[test]
    fn test_pack_regions_too_many_fails() {
        // Fill a tiny atlas with many large regions.
        let cfg = AtlasConfig {
            atlas_size: 64,
            default_padding: 0,
            power_of_two: false,
        };
        // Each 64×64 region takes the whole atlas; more than one should fail.
        let regions: Vec<_> = (0..10)
            .map(|i| (64usize, 64usize, i, format!("r{i}")))
            .collect();
        let result = pack_regions(&regions, cfg);
        assert!(matches!(result, Err(FaceAtlasError::PackingFailed { .. })));
    }

    #[test]
    fn test_pack_regions_sorted_by_height() {
        // The tallest region should be packed first (lowest id in sorted output
        // may differ from input order — verify no overlap).
        let regions = vec![
            (10usize, 10usize, 0usize, "small".to_string()),
            (10, 50, 1, "tall".to_string()),
            (10, 30, 2, "medium".to_string()),
        ];
        let atlas = pack_regions(&regions, default_config()).expect("should fit");
        assert_eq!(atlas.num_regions(), 3);
        // Verify no rects overlap.
        let rects: Vec<AtlasRect> = atlas.regions.iter().map(|r| r.rect).collect();
        for i in 0..rects.len() {
            for j in (i + 1)..rects.len() {
                assert!(!rects[i].intersects(&rects[j]), "rects {i} and {j} overlap");
            }
        }
    }

    // -----------------------------------------------------------------------
    // compute_atlas_stats
    // -----------------------------------------------------------------------

    #[test]
    fn test_compute_atlas_stats_empty() {
        let atlas = FaceAtlas::new(default_config()).expect("valid config");
        let stats = compute_atlas_stats(&atlas);
        assert_eq!(stats.num_regions, 0);
        assert_eq!(stats.total_area, 0);
        assert_eq!(stats.coverage_fraction, 0.0);
    }

    #[test]
    fn test_compute_atlas_stats_flame_atlas() {
        let atlas = create_flame_face_atlas(default_config()).expect("valid config");
        let stats = compute_atlas_stats(&atlas);
        assert_eq!(stats.num_regions, 9);
        assert!(stats.total_area > 0);
        assert!(stats.coverage_fraction > 0.0 && stats.coverage_fraction <= 1.0);
        assert!(stats.largest_region_area >= stats.smallest_region_area);
    }

    #[test]
    fn test_compute_atlas_stats_coverage_matches_atlas_coverage() {
        let atlas = create_flame_face_atlas(default_config()).expect("valid config");
        let stats = compute_atlas_stats(&atlas);
        let direct = atlas.coverage();
        assert!((stats.coverage_fraction - direct).abs() < 1e-6);
    }

    #[test]
    fn test_compute_atlas_stats_wasted_area() {
        let atlas = create_flame_face_atlas(default_config()).expect("valid config");
        let stats = compute_atlas_stats(&atlas);
        let total = atlas.atlas_size * atlas.atlas_size;
        assert_eq!(stats.wasted_area + stats.total_area, total);
    }
}
