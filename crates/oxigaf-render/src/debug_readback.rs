//! GPU→CPU intermediate buffer readback debug utilities.
//!
//! This module provides CPU-side snapshots of GPU pipeline intermediate state
//! for debugging 3D Gaussian Splatting rasterization. Since GPU hardware may
//! not be available in tests, all core logic runs on the CPU path; GPU-side
//! readback would feed the same data structures.
//!
//! ## Usage
//!
//! ```rust
//! use oxigaf_render::debug_readback::DebugReadbackBuilder;
//!
//! let builder = DebugReadbackBuilder::new(512, 512)
//!     .with_tile_size(16)
//!     .with_capacity(256);
//!
//! // Feed CPU-side Gaussian data (positions, depths, radii).
//! let positions = vec![(256.0f32, 256.0f32)];
//! let depths    = vec![1.0f32];
//! let radii     = vec![8.0f32];
//!
//! let snapshot = builder.compute_snapshot(&positions, &depths, &radii).unwrap();
//! println!("Tiles: {}×{}", snapshot.num_tiles_x, snapshot.num_tiles_y);
//! println!("Stats: {:?}", snapshot.stats);
//! ```

use crate::RenderError;

// ── Statistics ───────────────────────────────────────────────────────────────

/// Per-tile and per-Gaussian rasterization statistics.
#[derive(Debug, Clone, PartialEq)]
pub struct RasterizationStats {
    /// Total number of Gaussians submitted.
    pub total_gaussians: u32,
    /// Gaussians with depth > near-plane threshold (0.001).
    pub visible_gaussians: u32,
    /// Maximum Gaussians in any single tile.
    pub max_tile_occupancy: u32,
    /// Average Gaussians per tile across all tiles.
    pub mean_tile_occupancy: f32,
    /// Tiles that contain zero Gaussians.
    pub empty_tiles: u32,
    /// Tiles whose count exceeds `capacity_per_tile`.
    pub overflow_tiles: u32,
    /// Configured maximum Gaussians per tile.
    pub capacity_per_tile: u32,
}

impl RasterizationStats {
    /// Compute tile-level statistics from raw tile counts.
    ///
    /// `total_gaussians` and `visible_gaussians` must be supplied by the caller
    /// because they cannot be recovered from tile counts (a single Gaussian may
    /// contribute to multiple tiles).
    pub fn compute(
        tile_counts: &[u32],
        capacity: u32,
        total_gaussians: u32,
        visible_gaussians: u32,
    ) -> Self {
        let num_tiles = tile_counts.len() as u32;

        if num_tiles == 0 {
            return Self {
                total_gaussians,
                visible_gaussians,
                max_tile_occupancy: 0,
                mean_tile_occupancy: 0.0,
                empty_tiles: 0,
                overflow_tiles: 0,
                capacity_per_tile: capacity,
            };
        }

        let mut max_occupancy: u32 = 0;
        let mut sum: u64 = 0;
        let mut empty_tiles: u32 = 0;
        let mut overflow_tiles: u32 = 0;

        for &count in tile_counts {
            if count > max_occupancy {
                max_occupancy = count;
            }
            sum += u64::from(count);
            if count == 0 {
                empty_tiles += 1;
            }
            if count > capacity {
                overflow_tiles += 1;
            }
        }

        let mean_tile_occupancy = sum as f32 / num_tiles as f32;

        Self {
            total_gaussians,
            visible_gaussians,
            max_tile_occupancy: max_occupancy,
            mean_tile_occupancy,
            empty_tiles,
            overflow_tiles,
            capacity_per_tile: capacity,
        }
    }
}

// ── Snapshot ─────────────────────────────────────────────────────────────────

/// A CPU-side snapshot of rasterization pipeline intermediate state.
///
/// This is produced by [`DebugReadbackBuilder::compute_snapshot`] from
/// CPU-side Gaussian data, or from GPU buffer readback after a render pass.
#[derive(Debug, Clone)]
pub struct RasterizationSnapshot {
    /// Per-tile Gaussian counts (number of Gaussians touching each tile).
    ///
    /// Length: `num_tiles_x * num_tiles_y`.
    /// Layout: row-major, `tile[y * num_tiles_x + x]`.
    pub tile_gaussian_counts: Vec<u32>,

    /// Tile grid width (number of tiles along X).
    pub num_tiles_x: u32,
    /// Tile grid height (number of tiles along Y).
    pub num_tiles_y: u32,
    /// Tile size in pixels.
    pub tile_size: u32,

    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,

    /// Minimum depth value across all Gaussians.
    /// `f32::INFINITY` when no Gaussians are present.
    pub min_depth: f32,
    /// Maximum depth value across all Gaussians.
    /// `f32::NEG_INFINITY` when no Gaussians are present.
    pub max_depth: f32,

    /// Screen-space radius (pixels) per Gaussian.  Length: `num_gaussians`.
    pub gaussian_screen_sizes: Vec<f32>,

    /// Summary statistics derived from the tile counts and Gaussian data.
    pub stats: RasterizationStats,
}

impl RasterizationSnapshot {
    // ── Visualization helpers ─────────────────────────────────────────────

    /// Render tile occupancy as a greyscale image (one `u8` per pixel).
    ///
    /// The tile with `max_tile_occupancy` maps to 255; empty tiles map to 0.
    /// The returned `Vec<u8>` has `width * height` elements in row-major order,
    /// one byte per pixel.
    pub fn tile_occupancy_image(&self) -> Vec<u8> {
        let num_pixels = (self.width * self.height) as usize;
        let max_occ = self.stats.max_tile_occupancy;

        if max_occ == 0 {
            // Avoid division by zero: all-black for an empty scene.
            return vec![0u8; num_pixels];
        }

        let mut image = vec![0u8; num_pixels];
        for py in 0..self.height {
            for px in 0..self.width {
                let pixel_idx = (py * self.width + px) as usize;
                let tile_x = px / self.tile_size;
                let tile_y = py / self.tile_size;
                let tile_idx = (tile_y * self.num_tiles_x + tile_x) as usize;
                let count = self
                    .tile_gaussian_counts
                    .get(tile_idx)
                    .copied()
                    .unwrap_or(0);
                // Normalise to [0, 255], rounding.
                let value = ((count as f32 / max_occ as f32) * 255.0 + 0.5) as u8;
                image[pixel_idx] = value;
            }
        }
        image
    }

    /// Return the flat tile index for a pixel coordinate.
    ///
    /// Returns `None` if `(px, py)` is outside the image boundaries.
    pub fn tile_for_pixel(&self, px: u32, py: u32) -> Option<usize> {
        if px >= self.width || py >= self.height {
            return None;
        }
        let tile_x = px / self.tile_size;
        let tile_y = py / self.tile_size;
        Some((tile_y * self.num_tiles_x + tile_x) as usize)
    }

    /// Return the Gaussian count for the tile that contains pixel `(px, py)`.
    ///
    /// Returns `None` if the pixel is out of bounds.
    pub fn tile_count_at_pixel(&self, px: u32, py: u32) -> Option<u32> {
        let tile_idx = self.tile_for_pixel(px, py)?;
        self.tile_gaussian_counts.get(tile_idx).copied()
    }

    /// Find all tiles whose Gaussian count meets or exceeds `threshold`.
    ///
    /// Returns a list of `(tile_x, tile_y, count)` tuples, sorted by count
    /// descending so the hottest tile appears first.
    pub fn hotspot_tiles(&self, threshold: u32) -> Vec<(u32, u32, u32)> {
        let mut hotspots: Vec<(u32, u32, u32)> = Vec::new();
        for tile_y in 0..self.num_tiles_y {
            for tile_x in 0..self.num_tiles_x {
                let idx = (tile_y * self.num_tiles_x + tile_x) as usize;
                let count = self.tile_gaussian_counts.get(idx).copied().unwrap_or(0);
                if count >= threshold {
                    hotspots.push((tile_x, tile_y, count));
                }
            }
        }
        hotspots.sort_by_key(|&(_, _, count)| std::cmp::Reverse(count));
        hotspots
    }

    /// Depth range as `(min_depth, max_depth)`.
    ///
    /// Returns `(f32::INFINITY, f32::NEG_INFINITY)` when no Gaussians are present.
    #[inline]
    pub fn depth_range(&self) -> (f32, f32) {
        (self.min_depth, self.max_depth)
    }
}

// ── Validation helpers ────────────────────────────────────────────────────────

/// Check all values in a CPU buffer for NaN or Inf.
///
/// Returns `Ok(())` when every element is finite, or an error describing
/// the first problematic value found.
///
/// # Feature gate
///
/// This function is always compiled, but is intended to be called only under
/// `#[cfg(feature = "gpu_debug")]` guard sites in the pipeline so that the
/// validation overhead is zero in production builds.
///
/// # Errors
///
/// Returns [`RenderError`] when any element is NaN or non-finite.
pub fn validate_no_nan_inf(data: &[f32], label: &str) -> Result<(), RenderError> {
    for (i, &v) in data.iter().enumerate() {
        if v.is_nan() {
            return Err(RenderError::ValidationError(format!(
                "{label}: NaN at index {i}"
            )));
        }
        if v.is_infinite() {
            return Err(RenderError::ValidationError(format!(
                "{label}: Inf ({v}) at index {i}"
            )));
        }
    }
    Ok(())
}

/// Check that a count matches the expected value.
///
/// # Errors
///
/// Returns [`RenderError`] when `actual != expected`.
pub fn validate_buffer_count(
    actual: usize,
    expected: usize,
    label: &str,
) -> Result<(), RenderError> {
    if actual != expected {
        return Err(RenderError::ValidationError(format!(
            "{label}: expected {expected} elements but got {actual}"
        )));
    }
    Ok(())
}

// ── Builder ───────────────────────────────────────────────────────────────────

/// Near-plane threshold for classifying a Gaussian as "visible".
const NEAR_PLANE_THRESHOLD: f32 = 0.001;

/// Builder for computing a [`RasterizationSnapshot`] from CPU-side data.
///
/// GPU-side readback would call the same logic after copying GPU buffers back to
/// CPU memory.
///
/// # Example
///
/// ```rust
/// use oxigaf_render::debug_readback::DebugReadbackBuilder;
///
/// let snap = DebugReadbackBuilder::new(64, 64)
///     .compute_snapshot(
///         &[(32.0, 32.0)],  // screen positions
///         &[1.0],           // depths
///         &[4.0],           // radii
///     )
///     .expect("snapshot");
///
/// assert_eq!(snap.stats.total_gaussians, 1);
/// ```
pub struct DebugReadbackBuilder {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Tile size in pixels (default: 16).
    pub tile_size: u32,
    /// Maximum Gaussians per tile before a tile is counted as overflowing
    /// (default: 256).
    pub capacity_per_tile: u32,
}

impl DebugReadbackBuilder {
    /// Create a new builder for an image of the given dimensions.
    ///
    /// Defaults: `tile_size = 16`, `capacity_per_tile = 256`.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            tile_size: 16,
            capacity_per_tile: 256,
        }
    }

    /// Override the tile size in pixels.
    #[must_use]
    pub fn with_tile_size(mut self, tile_size: u32) -> Self {
        self.tile_size = tile_size;
        self
    }

    /// Override the per-tile capacity used for overflow detection.
    #[must_use]
    pub fn with_capacity(mut self, capacity: u32) -> Self {
        self.capacity_per_tile = capacity;
        self
    }

    /// Number of tiles along X.
    #[inline]
    fn num_tiles_x(&self) -> u32 {
        self.width.div_ceil(self.tile_size)
    }

    /// Number of tiles along Y.
    #[inline]
    fn num_tiles_y(&self) -> u32 {
        self.height.div_ceil(self.tile_size)
    }

    /// Compute a [`RasterizationSnapshot`] from CPU-side Gaussian data.
    ///
    /// # Parameters
    ///
    /// - `gaussian_positions_2d`: Screen-space `(x, y)` in pixel coordinates.
    /// - `gaussian_depths`: Depth value per Gaussian (positive = in front of camera).
    /// - `gaussian_radii`: Screen-space radius in pixels per Gaussian.
    ///
    /// All three slices must have the same length.
    ///
    /// # Tile assignment
    ///
    /// For each Gaussian at `(cx, cy)` with radius `r`, its axis-aligned bounding
    /// box `[cx-r, cx+r] × [cy-r, cy+r]` is clipped to the image boundary, then
    /// every tile overlapping that AABB has its count incremented.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::Rasterize`] if `width`, `height`, or `tile_size` is
    /// zero, and [`RenderError::MismatchedBufferSizes`] if the three input slices
    /// have different lengths.
    pub fn compute_snapshot(
        &self,
        gaussian_positions_2d: &[(f32, f32)],
        gaussian_depths: &[f32],
        gaussian_radii: &[f32],
    ) -> Result<RasterizationSnapshot, RenderError> {
        // ── Validate config ───────────────────────────────────────────────
        if self.width == 0 || self.height == 0 {
            return Err(RenderError::Rasterize(
                "Image dimensions must be non-zero".to_string(),
            ));
        }
        if self.tile_size == 0 {
            return Err(RenderError::Rasterize(
                "Tile size must be non-zero".to_string(),
            ));
        }

        // ── Validate input lengths ────────────────────────────────────────
        let n = gaussian_positions_2d.len();
        if gaussian_depths.len() != n {
            return Err(RenderError::MismatchedBufferSizes {
                expected: n,
                actual: gaussian_depths.len(),
            });
        }
        if gaussian_radii.len() != n {
            return Err(RenderError::MismatchedBufferSizes {
                expected: n,
                actual: gaussian_radii.len(),
            });
        }

        let num_tiles_x = self.num_tiles_x();
        let num_tiles_y = self.num_tiles_y();
        let total_tiles = (num_tiles_x * num_tiles_y) as usize;

        let mut tile_gaussian_counts = vec![0u32; total_tiles];
        let mut min_depth = f32::INFINITY;
        let mut max_depth = f32::NEG_INFINITY;
        let mut visible_gaussians: u32 = 0;

        // ── Build tile counts ─────────────────────────────────────────────
        for i in 0..n {
            let (cx, cy) = gaussian_positions_2d[i];
            let depth = gaussian_depths[i];
            let radius = gaussian_radii[i];

            // Track depth range.
            if depth < min_depth {
                min_depth = depth;
            }
            if depth > max_depth {
                max_depth = depth;
            }

            // Classify visibility.
            if depth > NEAR_PLANE_THRESHOLD {
                visible_gaussians += 1;
            }

            // Compute AABB in pixel space.
            let x_min = (cx - radius).max(0.0);
            let y_min = (cy - radius).max(0.0);
            // Exclusive upper bounds: clamp to [0, width/height).
            let x_max = (cx + radius).min(self.width as f32 - 1.0);
            let y_max = (cy + radius).min(self.height as f32 - 1.0);

            // Skip Gaussians fully outside the image.
            if x_max < 0.0
                || y_max < 0.0
                || x_min >= self.width as f32
                || y_min >= self.height as f32
            {
                continue;
            }

            // Convert pixel AABB to tile range.
            let tile_x_min = (x_min as u32) / self.tile_size;
            let tile_y_min = (y_min as u32) / self.tile_size;
            // Ceil division for the upper bound tile.
            let tile_x_max = (x_max as u32) / self.tile_size;
            let tile_y_max = (y_max as u32) / self.tile_size;

            // Clamp to valid tile indices.
            let tile_x_max = tile_x_max.min(num_tiles_x - 1);
            let tile_y_max = tile_y_max.min(num_tiles_y - 1);

            for ty in tile_y_min..=tile_y_max {
                for tx in tile_x_min..=tile_x_max {
                    let tile_idx = (ty * num_tiles_x + tx) as usize;
                    if let Some(cell) = tile_gaussian_counts.get_mut(tile_idx) {
                        *cell = cell.saturating_add(1);
                    }
                }
            }
        }

        // ── Compute statistics ────────────────────────────────────────────
        let total_gaussians = n as u32;
        let stats = RasterizationStats::compute(
            &tile_gaussian_counts,
            self.capacity_per_tile,
            total_gaussians,
            visible_gaussians,
        );

        // Collect radii.
        let gaussian_screen_sizes = gaussian_radii.to_vec();

        Ok(RasterizationSnapshot {
            tile_gaussian_counts,
            num_tiles_x,
            num_tiles_y,
            tile_size: self.tile_size,
            width: self.width,
            height: self.height,
            min_depth,
            max_depth,
            gaussian_screen_sizes,
            stats,
        })
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Builder construction ──────────────────────────────────────────────

    #[test]
    fn test_builder_tile_dimensions_default() {
        let b = DebugReadbackBuilder::new(512, 256);
        // 512 / 16 = 32, 256 / 16 = 16
        assert_eq!(b.num_tiles_x(), 32);
        assert_eq!(b.num_tiles_y(), 16);
    }

    #[test]
    fn test_builder_tile_dimensions_non_divisible() {
        // 100 / 16 = 6.25 → ceil → 7
        let b = DebugReadbackBuilder::new(100, 100);
        assert_eq!(b.num_tiles_x(), 7);
        assert_eq!(b.num_tiles_y(), 7);
    }

    #[test]
    fn test_builder_with_tile_size() {
        let b = DebugReadbackBuilder::new(64, 64).with_tile_size(8);
        assert_eq!(b.tile_size, 8);
        assert_eq!(b.num_tiles_x(), 8);
        assert_eq!(b.num_tiles_y(), 8);
    }

    #[test]
    fn test_builder_with_capacity() {
        let b = DebugReadbackBuilder::new(64, 64).with_capacity(512);
        assert_eq!(b.capacity_per_tile, 512);
    }

    // ── Empty Gaussian list ───────────────────────────────────────────────

    #[test]
    fn test_empty_gaussians_all_zero_tile_counts() {
        let snap = DebugReadbackBuilder::new(64, 64)
            .compute_snapshot(&[], &[], &[])
            .expect("empty snapshot");

        assert!(snap.tile_gaussian_counts.iter().all(|&c| c == 0));
        assert_eq!(snap.stats.total_gaussians, 0);
        assert_eq!(snap.stats.visible_gaussians, 0);
    }

    #[test]
    fn test_empty_gaussians_depth_range_sentinels() {
        let snap = DebugReadbackBuilder::new(32, 32)
            .compute_snapshot(&[], &[], &[])
            .expect("empty snapshot");

        assert_eq!(snap.min_depth, f32::INFINITY);
        assert_eq!(snap.max_depth, f32::NEG_INFINITY);
    }

    // ── Single Gaussian ───────────────────────────────────────────────────

    #[test]
    fn test_single_gaussian_center_tile_has_count_one() {
        // 64×64 image, tile_size=16 → 4×4 tiles.
        // Gaussian at (24, 24) radius=4 → AABB [20,28]×[20,28].
        // 20/16=1, 28/16=1 → only tile (1,1) is touched.
        let snap = DebugReadbackBuilder::new(64, 64)
            .with_tile_size(16)
            .compute_snapshot(&[(24.0, 24.0)], &[1.0], &[4.0])
            .expect("single Gaussian");

        // num_tiles_x = 4, tile (1,1) → index 1*4+1 = 5.
        let center_tile_idx = 4 + 1;
        assert_eq!(
            snap.tile_gaussian_counts[center_tile_idx], 1,
            "tile (1,1) should have count 1"
        );

        let other_sum: u32 = snap
            .tile_gaussian_counts
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != center_tile_idx)
            .map(|(_, &c)| c)
            .sum();
        assert_eq!(other_sum, 0, "all other tiles should be 0");
    }

    #[test]
    fn test_single_gaussian_with_zero_depth_not_visible() {
        let snap = DebugReadbackBuilder::new(64, 64)
            .compute_snapshot(&[(32.0, 32.0)], &[0.0], &[4.0])
            .expect("snapshot");

        // depth == 0.0 <= NEAR_PLANE_THRESHOLD → not visible.
        assert_eq!(snap.stats.visible_gaussians, 0);
        assert_eq!(snap.stats.total_gaussians, 1);
    }

    // ── Gaussian spanning 4 tiles ─────────────────────────────────────────

    #[test]
    fn test_gaussian_spanning_four_tiles() {
        // 32×32 image, tile_size=16 → 2×2 tiles.
        // Gaussian at (16, 16) radius=4 → AABB [12,20]×[12,20].
        //   tile_x range: 12/16=0 to 20/16=1 (both tiles)
        //   tile_y range: 0 to 1 (both tiles)
        // All 4 tiles touched.
        let snap = DebugReadbackBuilder::new(32, 32)
            .with_tile_size(16)
            .compute_snapshot(&[(16.0, 16.0)], &[1.0], &[4.0])
            .expect("4-tile Gaussian");

        assert_eq!(snap.tile_gaussian_counts.len(), 4, "2×2 = 4 tiles");
        for (i, &count) in snap.tile_gaussian_counts.iter().enumerate() {
            assert_eq!(count, 1, "tile {i} should have count 1");
        }
    }

    // ── Accumulation ─────────────────────────────────────────────────────

    #[test]
    fn test_multiple_gaussians_counts_accumulate() {
        // 16×16 image, tile_size=16 → 1×1 tile.
        // Three Gaussians all in the single tile.
        let positions = vec![(8.0f32, 8.0f32); 3];
        let depths = vec![1.0f32; 3];
        let radii = vec![2.0f32; 3];

        let snap = DebugReadbackBuilder::new(16, 16)
            .with_tile_size(16)
            .compute_snapshot(&positions, &depths, &radii)
            .expect("multi-Gaussian");

        assert_eq!(snap.tile_gaussian_counts[0], 3);
        assert_eq!(snap.stats.total_gaussians, 3);
    }

    // ── tile_for_pixel ────────────────────────────────────────────────────

    #[test]
    fn test_tile_for_pixel_correct_index() {
        let snap = DebugReadbackBuilder::new(64, 64)
            .with_tile_size(16)
            .compute_snapshot(&[], &[], &[])
            .expect("snap");

        // num_tiles_x = 4
        // Pixel (20, 20) → tile_x = 20/16 = 1, tile_y = 20/16 = 1 → idx = 1*4+1 = 5
        assert_eq!(snap.tile_for_pixel(20, 20), Some(5));
        // Pixel (0, 0) → tile (0,0) → idx 0
        assert_eq!(snap.tile_for_pixel(0, 0), Some(0));
        // Pixel (63, 63) → tile_x=3, tile_y=3 → idx = 3*4+3 = 15
        assert_eq!(snap.tile_for_pixel(63, 63), Some(15));
    }

    #[test]
    fn test_tile_for_pixel_out_of_bounds() {
        let snap = DebugReadbackBuilder::new(64, 64)
            .compute_snapshot(&[], &[], &[])
            .expect("snap");

        assert_eq!(snap.tile_for_pixel(64, 0), None);
        assert_eq!(snap.tile_for_pixel(0, 64), None);
        assert_eq!(snap.tile_for_pixel(100, 100), None);
    }

    // ── tile_count_at_pixel ───────────────────────────────────────────────

    #[test]
    fn test_tile_count_at_pixel_correct() {
        // Single Gaussian at (8, 8) radius=2 → tile (0,0) in a 16×16/tile-16 setup.
        let snap = DebugReadbackBuilder::new(16, 16)
            .with_tile_size(16)
            .compute_snapshot(&[(8.0, 8.0)], &[1.0], &[2.0])
            .expect("snap");

        assert_eq!(snap.tile_count_at_pixel(8, 8), Some(1));
    }

    #[test]
    fn test_tile_count_at_pixel_out_of_bounds_returns_none() {
        let snap = DebugReadbackBuilder::new(16, 16)
            .compute_snapshot(&[], &[], &[])
            .expect("snap");

        assert_eq!(snap.tile_count_at_pixel(16, 0), None);
        assert_eq!(snap.tile_count_at_pixel(0, 16), None);
    }

    // ── tile_occupancy_image ──────────────────────────────────────────────

    #[test]
    fn test_tile_occupancy_image_correct_dimensions() {
        let snap = DebugReadbackBuilder::new(32, 32)
            .compute_snapshot(&[], &[], &[])
            .expect("snap");

        let img = snap.tile_occupancy_image();
        assert_eq!(img.len(), 32 * 32);
    }

    #[test]
    fn test_tile_occupancy_image_all_zero_for_empty_scene() {
        let snap = DebugReadbackBuilder::new(32, 32)
            .compute_snapshot(&[], &[], &[])
            .expect("snap");

        let img = snap.tile_occupancy_image();
        assert!(img.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_tile_occupancy_image_max_tile_is_255() {
        // Single tile (16×16 image, tile_size=16), one Gaussian → max_occ=1.
        // Every pixel in that tile should map to 255.
        let snap = DebugReadbackBuilder::new(16, 16)
            .with_tile_size(16)
            .compute_snapshot(&[(8.0, 8.0)], &[1.0], &[2.0])
            .expect("snap");

        let img = snap.tile_occupancy_image();
        assert!(img.iter().all(|&v| v == 255), "all pixels should be 255");
    }

    // ── hotspot_tiles ─────────────────────────────────────────────────────

    #[test]
    fn test_hotspot_tiles_only_above_threshold() {
        // 2×1 tile grid (32×16 image, tile_size=16).
        // Left tile: 3 Gaussians; right tile: 1 Gaussian.
        let positions = vec![
            (8.0f32, 8.0f32), // left tile
            (8.0f32, 8.0f32),
            (8.0f32, 8.0f32),
            (24.0f32, 8.0f32), // right tile
        ];
        let depths = vec![1.0f32; 4];
        let radii = vec![2.0f32; 4];

        let snap = DebugReadbackBuilder::new(32, 16)
            .with_tile_size(16)
            .compute_snapshot(&positions, &depths, &radii)
            .expect("snap");

        let hotspots = snap.hotspot_tiles(2);
        assert_eq!(hotspots.len(), 1, "only left tile exceeds threshold 2");
        let (tx, ty, count) = hotspots[0];
        assert_eq!(tx, 0);
        assert_eq!(ty, 0);
        assert_eq!(count, 3);
    }

    #[test]
    fn test_hotspot_tiles_empty_above_threshold() {
        let snap = DebugReadbackBuilder::new(32, 32)
            .compute_snapshot(&[], &[], &[])
            .expect("snap");

        let hotspots = snap.hotspot_tiles(1);
        assert!(hotspots.is_empty());
    }

    // ── RasterizationStats::compute ───────────────────────────────────────

    #[test]
    fn test_stats_all_zero_empty_tiles_equals_total() {
        let counts = vec![0u32; 6]; // 6 tiles, all empty.
        let stats = RasterizationStats::compute(&counts, 256, 0, 0);

        assert_eq!(stats.empty_tiles, 6);
        assert_eq!(stats.max_tile_occupancy, 0);
        assert_eq!(stats.overflow_tiles, 0);
        assert_eq!(stats.mean_tile_occupancy, 0.0);
    }

    #[test]
    fn test_stats_total_gaussians_correct() {
        let counts = vec![1u32, 2u32, 0u32];
        let stats = RasterizationStats::compute(&counts, 256, 42, 40);

        assert_eq!(stats.total_gaussians, 42);
        assert_eq!(stats.visible_gaussians, 40);
    }

    #[test]
    fn test_stats_max_tile_occupancy_correct() {
        let counts = vec![0u32, 5u32, 3u32, 1u32];
        let stats = RasterizationStats::compute(&counts, 256, 0, 0);

        assert_eq!(stats.max_tile_occupancy, 5);
    }

    #[test]
    fn test_stats_mean_tile_occupancy_correct() {
        // Counts: [4, 2] → mean = 3.0
        let counts = vec![4u32, 2u32];
        let stats = RasterizationStats::compute(&counts, 256, 0, 0);

        assert!((stats.mean_tile_occupancy - 3.0).abs() < 1e-5);
    }

    // ── depth_range ───────────────────────────────────────────────────────

    #[test]
    fn test_depth_range_min_max_correct() {
        let positions = vec![(8.0f32, 8.0f32); 3];
        let depths = vec![1.0f32, 0.5f32, 2.5f32];
        let radii = vec![2.0f32; 3];

        let snap = DebugReadbackBuilder::new(16, 16)
            .with_tile_size(16)
            .compute_snapshot(&positions, &depths, &radii)
            .expect("snap");

        let (min_d, max_d) = snap.depth_range();
        assert!((min_d - 0.5).abs() < 1e-6);
        assert!((max_d - 2.5).abs() < 1e-6);
    }

    // ── Overflow detection ────────────────────────────────────────────────

    #[test]
    fn test_overflow_detection_tile_count_exceeds_capacity() {
        // Single tile, capacity=2, insert 3 Gaussians → overflow_tiles = 1.
        let positions = vec![(8.0f32, 8.0f32); 3];
        let depths = vec![1.0f32; 3];
        let radii = vec![2.0f32; 3];

        let snap = DebugReadbackBuilder::new(16, 16)
            .with_tile_size(16)
            .with_capacity(2)
            .compute_snapshot(&positions, &depths, &radii)
            .expect("snap");

        assert_eq!(snap.stats.overflow_tiles, 1);
    }

    // ── Input validation ──────────────────────────────────────────────────

    #[test]
    fn test_mismatched_depths_returns_error() {
        let result = DebugReadbackBuilder::new(64, 64).compute_snapshot(&[(1.0, 1.0)], &[], &[1.0]);

        assert!(result.is_err());
    }

    #[test]
    fn test_mismatched_radii_returns_error() {
        let result = DebugReadbackBuilder::new(64, 64).compute_snapshot(&[(1.0, 1.0)], &[1.0], &[]);

        assert!(result.is_err());
    }

    #[test]
    fn test_zero_tile_size_returns_error() {
        let result = DebugReadbackBuilder::new(64, 64)
            .with_tile_size(0)
            .compute_snapshot(&[], &[], &[]);

        assert!(result.is_err());
    }

    // ── gaussian_screen_sizes ─────────────────────────────────────────────

    #[test]
    fn test_gaussian_screen_sizes_stored_correctly() {
        let radii = vec![3.0f32, 7.5f32, 12.0f32];
        let positions = vec![(8.0f32, 8.0f32); 3];
        let depths = vec![1.0f32; 3];

        let snap = DebugReadbackBuilder::new(32, 32)
            .compute_snapshot(&positions, &depths, &radii)
            .expect("snap");

        assert_eq!(snap.gaussian_screen_sizes, radii);
    }
}
