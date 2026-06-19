//! Per-tile rendering statistics for adaptive rendering decisions.
//!
//! This module provides [`TileStatsGrid`], a compact representation of how
//! projected 2D Gaussians are distributed across the render tile grid.
//! The statistics drive adaptive decisions such as tile-size tuning, LOD
//! selection, and load-balancing in the GPU tile rasterizer.
//!
//! ## Quick start
//!
//! ```rust
//! use oxigaf_render::tile_stats::{compute_tile_stats, TileAnalysisReport, HeatmapMode, TileHeatmap};
//!
//! let positions = vec![[128.0f32, 128.0f32], [64.0, 64.0]];
//! let radii     = vec![8.0f32, 16.0f32];
//! let depths    = vec![1.0f32, 2.0f32];
//!
//! let grid = compute_tile_stats(&positions, &radii, &depths, 256, 256, 16)
//!     .expect("compute_tile_stats");
//!
//! let report = TileAnalysisReport::compute(&grid, 32);
//! println!("{}", report.format_summary());
//!
//! let heatmap = TileHeatmap::from_grid(&grid, HeatmapMode::GaussianDensity);
//! println!("{}", heatmap.to_ascii());
//! ```

use thiserror::Error;

// ── Constants ────────────────────────────────────────────────────────────────

/// Memory consumed per active Gaussian in the rasterizer (position, conic,
/// color, alpha — four 16-byte blocks).
const BYTES_PER_GAUSSIAN_ACTIVATION: usize = 64;

// ── TileStatsError ───────────────────────────────────────────────────────────

/// Errors returned by [`compute_tile_stats`].
#[derive(Debug, Error)]
pub enum TileStatsError {
    /// No Gaussians were provided.
    #[error("Empty input: positions_2d, screen_radii, and depths are all empty")]
    EmptyInput,

    /// Input slices have different lengths.
    #[error("Length mismatch: expected {expected} elements but got {got}")]
    LengthMismatch {
        /// The reference (positions_2d) length.
        expected: usize,
        /// The length of the mismatched slice.
        got: usize,
    },

    /// tile_size was set to zero.
    #[error("Tile size must be non-zero")]
    ZeroTileSize,

    /// image_width or image_height was set to zero.
    #[error("Image dimensions must be non-zero")]
    ZeroImageDimension,
}

// ── TileStats ────────────────────────────────────────────────────────────────

/// Per-tile rendering statistics.
///
/// Produced by [`compute_tile_stats`] and stored in a [`TileStatsGrid`].
#[derive(Debug, Clone, Default)]
pub struct TileStats {
    /// Number of Gaussians whose 2D projection overlaps this tile.
    pub gaussian_count: usize,

    /// Fraction of tile pixels estimated to be covered by Gaussians.
    ///
    /// Computed as `min(1.0, gaussian_count / tile_area_pixels)` after all
    /// Gaussians have been processed.
    pub coverage_fraction: f32,

    /// Minimum depth of Gaussians in this tile.
    ///
    /// `f32::INFINITY` when the tile is empty.
    pub min_depth: f32,

    /// Maximum depth of Gaussians in this tile.
    ///
    /// `0.0` when the tile is empty (matches the `Default` value).
    pub max_depth: f32,

    /// Estimated GPU memory pressure for this tile in bytes.
    ///
    /// Computed as `gaussian_count × BYTES_PER_GAUSSIAN_ACTIVATION`.
    pub memory_pressure_bytes: usize,
}

impl TileStats {
    /// Returns `true` when no Gaussians overlap this tile.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.gaussian_count == 0
    }

    /// Depth extent of the Gaussians in this tile (`max_depth - min_depth`).
    ///
    /// Returns `0.0` for empty tiles (since `max_depth - min_depth = 0 - 0`).
    #[inline]
    pub fn depth_range(&self) -> f32 {
        self.max_depth - self.min_depth
    }
}

// ── TileStatsGrid ────────────────────────────────────────────────────────────

/// A grid of [`TileStats`], one entry per render tile.
///
/// Tiles are stored in row-major order: `tiles[y * width_tiles + x]`.
#[derive(Debug, Clone)]
pub struct TileStatsGrid {
    /// Number of tiles along the X axis.
    pub width_tiles: usize,
    /// Number of tiles along the Y axis.
    pub height_tiles: usize,
    /// Side length of each tile in pixels.
    pub tile_size: usize,
    /// Image width in pixels.
    pub image_width: usize,
    /// Image height in pixels.
    pub image_height: usize,
    /// Row-major tile data: index = `y * width_tiles + x`.
    tiles: Vec<TileStats>,
}

impl TileStatsGrid {
    /// Create an empty [`TileStatsGrid`] for the given image and tile size.
    ///
    /// All tiles start with `gaussian_count = 0`, `min_depth = f32::INFINITY`,
    /// and other fields at their `Default` values.
    pub fn new(image_width: usize, image_height: usize, tile_size: usize) -> Self {
        let width_tiles = image_width.div_ceil(tile_size);
        let height_tiles = image_height.div_ceil(tile_size);
        let total = width_tiles * height_tiles;

        // Initialise min_depth to INFINITY so the first real depth wins.
        let tiles = (0..total)
            .map(|_| TileStats {
                min_depth: f32::INFINITY,
                ..TileStats::default()
            })
            .collect();

        Self {
            width_tiles,
            height_tiles,
            tile_size,
            image_width,
            image_height,
            tiles,
        }
    }

    /// Shared reference to the stats for tile `(tile_x, tile_y)`.
    ///
    /// Returns `None` when the coordinates are out of bounds.
    pub fn tile_at(&self, tile_x: usize, tile_y: usize) -> Option<&TileStats> {
        if tile_x >= self.width_tiles || tile_y >= self.height_tiles {
            return None;
        }
        self.tiles.get(tile_y * self.width_tiles + tile_x)
    }

    /// Mutable reference to the stats for tile `(tile_x, tile_y)`.
    ///
    /// Returns `None` when the coordinates are out of bounds.
    pub fn tile_at_mut(&mut self, tile_x: usize, tile_y: usize) -> Option<&mut TileStats> {
        if tile_x >= self.width_tiles || tile_y >= self.height_tiles {
            return None;
        }
        let idx = tile_y * self.width_tiles + tile_x;
        self.tiles.get_mut(idx)
    }

    /// Return the `(tile_x, tile_y)` tile coordinates for pixel `(px, py)`.
    ///
    /// No bounds check is performed — callers that may supply out-of-range
    /// pixels should validate separately.
    #[inline]
    pub fn tile_for_pixel(&self, px: usize, py: usize) -> (usize, usize) {
        (px / self.tile_size, py / self.tile_size)
    }

    /// Total number of tiles in the grid.
    #[inline]
    pub fn total_tiles(&self) -> usize {
        self.width_tiles * self.height_tiles
    }

    /// Number of tiles containing zero Gaussians.
    pub fn empty_tile_count(&self) -> usize {
        self.tiles.iter().filter(|t| t.is_empty()).count()
    }

    /// Maximum `gaussian_count` across all tiles.
    pub fn max_gaussian_count(&self) -> usize {
        self.tiles
            .iter()
            .map(|t| t.gaussian_count)
            .max()
            .unwrap_or(0)
    }

    /// Sum of `gaussian_count` across all tiles (each Gaussian may be counted
    /// multiple times if it spans several tiles).
    pub fn total_gaussian_count(&self) -> usize {
        self.tiles.iter().map(|t| t.gaussian_count).sum()
    }

    /// Return `(tile_x, tile_y)` pairs for the `top_n` tiles ordered by
    /// `gaussian_count` descending.
    pub fn hotspot_tile_indices(&self, top_n: usize) -> Vec<(usize, usize)> {
        // Build (count, index) pairs.
        let mut indexed: Vec<(usize, usize)> = self
            .tiles
            .iter()
            .enumerate()
            .map(|(idx, t)| (t.gaussian_count, idx))
            .collect();

        // Sort descending by count; stable to keep deterministic ordering on ties.
        indexed.sort_by_key(|&(count, _)| std::cmp::Reverse(count));

        indexed
            .into_iter()
            .take(top_n)
            .map(|(_, idx)| {
                let ty = idx / self.width_tiles;
                let tx = idx % self.width_tiles;
                (tx, ty)
            })
            .collect()
    }
}

// ── compute_tile_stats ───────────────────────────────────────────────────────

/// Compute per-tile statistics from 2D-projected Gaussian data.
///
/// For each Gaussian the function computes its axis-aligned bounding box
/// (AABB) in tile space and increments all overlapping tiles.  After all
/// Gaussians are processed a second pass finalises `coverage_fraction` and
/// `memory_pressure_bytes`.
///
/// # Parameters
///
/// - `positions_2d` – screen-space `[x, y]` in pixels for each Gaussian.
/// - `screen_radii` – screen-space radius in pixels for each Gaussian.
/// - `depths`       – depth value for each Gaussian.
/// - `image_width`, `image_height` – in pixels.
/// - `tile_size`    – tile side length in pixels (e.g. `16`).
///
/// # Errors
///
/// Returns [`TileStatsError::EmptyInput`] if all slices are empty,
/// [`TileStatsError::LengthMismatch`] if lengths differ,
/// [`TileStatsError::ZeroTileSize`] or [`TileStatsError::ZeroImageDimension`]
/// for degenerate configuration values.
pub fn compute_tile_stats(
    positions_2d: &[[f32; 2]],
    screen_radii: &[f32],
    depths: &[f32],
    image_width: usize,
    image_height: usize,
    tile_size: usize,
) -> Result<TileStatsGrid, TileStatsError> {
    // ── Input validation ──────────────────────────────────────────────────
    if tile_size == 0 {
        return Err(TileStatsError::ZeroTileSize);
    }
    if image_width == 0 || image_height == 0 {
        return Err(TileStatsError::ZeroImageDimension);
    }

    let n = positions_2d.len();

    if n == 0 {
        return Err(TileStatsError::EmptyInput);
    }

    if screen_radii.len() != n {
        return Err(TileStatsError::LengthMismatch {
            expected: n,
            got: screen_radii.len(),
        });
    }
    if depths.len() != n {
        return Err(TileStatsError::LengthMismatch {
            expected: n,
            got: depths.len(),
        });
    }

    // ── Build empty grid ──────────────────────────────────────────────────
    let mut grid = TileStatsGrid::new(image_width, image_height, tile_size);

    let width_tiles = grid.width_tiles;
    let height_tiles = grid.height_tiles;

    // ── Per-Gaussian pass ─────────────────────────────────────────────────
    for i in 0..n {
        let cx = positions_2d[i][0];
        let cy = positions_2d[i][1];
        let radius = screen_radii[i];
        let depth = depths[i];

        // AABB in pixel space — clamp negative to 0.0 before casting.
        let x_lo = (cx - radius).max(0.0);
        let y_lo = (cy - radius).max(0.0);
        let x_hi = (cx + radius).min((image_width as f32) - 1.0);
        let y_hi = (cy + radius).min((image_height as f32) - 1.0);

        // Skip Gaussians fully outside the image.
        if x_hi < 0.0 || y_hi < 0.0 || x_lo >= image_width as f32 || y_lo >= image_height as f32 {
            continue;
        }

        // Tile AABB (inclusive on both ends).
        let tile_x_min = (x_lo as usize) / tile_size;
        let tile_y_min = (y_lo as usize) / tile_size;
        let tile_x_max = ((x_hi as usize) / tile_size).min(width_tiles.saturating_sub(1));
        let tile_y_max = ((y_hi as usize) / tile_size).min(height_tiles.saturating_sub(1));

        for ty in tile_y_min..=tile_y_max {
            for tx in tile_x_min..=tile_x_max {
                if let Some(tile) = grid.tile_at_mut(tx, ty) {
                    tile.gaussian_count = tile.gaussian_count.saturating_add(1);

                    if depth < tile.min_depth {
                        tile.min_depth = depth;
                    }
                    if depth > tile.max_depth {
                        tile.max_depth = depth;
                    }
                }
            }
        }
    }

    // ── Post-processing pass: finalise derived fields ─────────────────────
    let tile_area = tile_size * tile_size;
    for tile in &mut grid.tiles {
        if tile.is_empty() {
            // Leave min_depth as INFINITY (sentinel), max_depth stays 0.0.
            tile.coverage_fraction = 0.0;
            tile.memory_pressure_bytes = 0;
        } else {
            tile.coverage_fraction = (tile.gaussian_count as f32 / tile_area as f32).min(1.0);
            tile.memory_pressure_bytes = tile.gaussian_count * BYTES_PER_GAUSSIAN_ACTIVATION;
        }
    }

    Ok(grid)
}

// ── Heatmap ──────────────────────────────────────────────────────────────────

/// Which statistic the heatmap colours should represent.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HeatmapMode {
    /// Colour by normalised Gaussian count (`gaussian_count / max_count`).
    GaussianDensity,
    /// Colour by `coverage_fraction`.
    Coverage,
    /// Colour by depth range normalised to the maximum depth range across tiles.
    DepthRange,
    /// Colour by memory pressure normalised to the maximum memory pressure.
    MemoryPressure,
}

/// A per-tile heatmap rendered as RGBA pixels using the Jet colourmap.
pub struct TileHeatmap {
    /// The statistic that was used to generate the heatmap.
    pub mode: HeatmapMode,
    /// Number of tiles along the X axis (= `grid.width_tiles`).
    pub width: usize,
    /// Number of tiles along the Y axis (= `grid.height_tiles`).
    pub height: usize,
    /// RGBA pixel data, one `[u8; 4]` per tile, in row-major order.
    pub pixels: Vec<[u8; 4]>,
}

impl TileHeatmap {
    /// Generate a heatmap from a [`TileStatsGrid`] using the given mode.
    ///
    /// The Jet colourmap maps a normalised value in `[0, 1]` to an RGB colour
    /// using the standard piecewise-linear approximation:
    ///
    /// ```text
    /// r = clamp(1.5 - |4v - 3|, 0, 1)
    /// g = clamp(1.5 - |4v - 2|, 0, 1)
    /// b = clamp(1.5 - |4v - 1|, 0, 1)
    /// ```
    pub fn from_grid(grid: &TileStatsGrid, mode: HeatmapMode) -> Self {
        // Gather raw values per tile.
        let raw_values: Vec<f32> = match mode {
            HeatmapMode::GaussianDensity => {
                grid.tiles.iter().map(|t| t.gaussian_count as f32).collect()
            }
            HeatmapMode::Coverage => grid.tiles.iter().map(|t| t.coverage_fraction).collect(),
            HeatmapMode::DepthRange => grid
                .tiles
                .iter()
                .map(|t| if t.is_empty() { 0.0 } else { t.depth_range() })
                .collect(),
            HeatmapMode::MemoryPressure => grid
                .tiles
                .iter()
                .map(|t| t.memory_pressure_bytes as f32)
                .collect(),
        };

        // Normalise to [0, 1].
        let max_val = raw_values.iter().cloned().fold(0.0f32, f32::max);

        let pixels = if max_val <= 0.0 {
            // All zeros → all-blue (cold end of Jet).
            vec![[0u8, 0u8, 128u8, 255u8]; grid.tiles.len()]
        } else {
            raw_values
                .iter()
                .map(|&v| {
                    let norm = (v / max_val).clamp(0.0, 1.0);
                    let [r, g, b] = jet_rgb(norm);
                    [r, g, b, 255u8]
                })
                .collect()
        };

        Self {
            mode,
            width: grid.width_tiles,
            height: grid.height_tiles,
            pixels,
        }
    }

    /// Render the heatmap as a compact ASCII art string.
    ///
    /// Each tile maps to one character from the ramp
    /// `' ' . : ; ! | = # % @` (10 levels).  Rows are separated by `'\n'`.
    pub fn to_ascii(&self) -> String {
        const RAMP: &[u8] = b" .:;!|=#%@";
        let n_levels = RAMP.len();

        let mut out = String::with_capacity(self.height * (self.width + 1));

        for ty in 0..self.height {
            for tx in 0..self.width {
                let pixel = self.pixels[ty * self.width + tx];
                // Reconstruct approximate brightness from R channel of Jet.
                // A simpler approach: use the stored pixel's luminance.
                let r = pixel[0] as f32;
                let g = pixel[1] as f32;
                let b = pixel[2] as f32;
                // Perceptual luminance (BT.601 coefficients).
                let lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0;
                let level = (lum * (n_levels as f32 - 1.0) + 0.5) as usize;
                let level = level.min(n_levels - 1);
                out.push(RAMP[level] as char);
            }
            out.push('\n');
        }

        out
    }
}

/// Jet colourmap: maps `v ∈ [0, 1]` to `[r, g, b]` each in `0..=255`.
#[inline]
fn jet_rgb(v: f32) -> [u8; 3] {
    let r = (1.5 - (4.0 * v - 3.0).abs()).clamp(0.0, 1.0);
    let g = (1.5 - (4.0 * v - 2.0).abs()).clamp(0.0, 1.0);
    let b = (1.5 - (4.0 * v - 1.0).abs()).clamp(0.0, 1.0);
    [
        (r * 255.0 + 0.5) as u8,
        (g * 255.0 + 0.5) as u8,
        (b * 255.0 + 0.5) as u8,
    ]
}

// ── TileAnalysisReport ───────────────────────────────────────────────────────

/// Summary analysis of a [`TileStatsGrid`] relative to an overload threshold.
#[derive(Debug, Clone)]
pub struct TileAnalysisReport {
    /// Total number of tiles.
    pub total_tiles: usize,
    /// Tiles with `gaussian_count == 0`.
    pub empty_tiles: usize,
    /// Tiles whose `gaussian_count > overload_threshold`.
    pub overloaded_tiles: usize,
    /// Tiles between empty and overloaded (neither empty nor overloaded).
    pub balanced_tiles: usize,
    /// The threshold used to classify tiles as overloaded.
    pub overload_threshold: usize,
    /// `(tile_x, tile_y)` coordinates of the top-5 most loaded tiles.
    pub hotspot_locations: Vec<(usize, usize)>,
    /// Sum of `gaussian_count` across all tiles (a Gaussian spanning N tiles
    /// is counted N times).
    pub total_gaussians_projected: usize,
    /// Mean `gaussian_count` per tile.
    pub mean_density: f32,
    /// Maximum `gaussian_count` across all tiles.
    pub max_density: usize,
    /// Load imbalance: standard deviation of `gaussian_count` / mean.
    ///
    /// `0.0` when mean is zero (all tiles empty).
    pub load_imbalance: f32,
}

impl TileAnalysisReport {
    /// Compute the analysis report for `grid` using `overload_threshold` as
    /// the cut-off for "overloaded" tiles.
    pub fn compute(grid: &TileStatsGrid, overload_threshold: usize) -> Self {
        let total_tiles = grid.total_tiles();

        if total_tiles == 0 {
            return Self {
                total_tiles: 0,
                empty_tiles: 0,
                overloaded_tiles: 0,
                balanced_tiles: 0,
                overload_threshold,
                hotspot_locations: Vec::new(),
                total_gaussians_projected: 0,
                mean_density: 0.0,
                max_density: 0,
                load_imbalance: 0.0,
            };
        }

        let mut empty_tiles = 0usize;
        let mut overloaded_tiles = 0usize;
        let mut total_gaussians = 0usize;
        let mut max_density = 0usize;

        for tile in &grid.tiles {
            let c = tile.gaussian_count;
            total_gaussians = total_gaussians.saturating_add(c);
            if c == 0 {
                empty_tiles += 1;
            } else if c > overload_threshold {
                overloaded_tiles += 1;
            }
            if c > max_density {
                max_density = c;
            }
        }

        let balanced_tiles = total_tiles - empty_tiles - overloaded_tiles;
        let mean_density = total_gaussians as f32 / total_tiles as f32;

        // Population standard deviation.
        let load_imbalance = if mean_density <= 0.0 {
            0.0
        } else {
            let variance = grid
                .tiles
                .iter()
                .map(|t| {
                    let diff = t.gaussian_count as f32 - mean_density;
                    diff * diff
                })
                .sum::<f32>()
                / total_tiles as f32;
            let std_dev = variance.sqrt();
            std_dev / mean_density
        };

        let hotspot_locations = grid.hotspot_tile_indices(5);

        Self {
            total_tiles,
            empty_tiles,
            overloaded_tiles,
            balanced_tiles,
            overload_threshold,
            hotspot_locations,
            total_gaussians_projected: total_gaussians,
            mean_density,
            max_density,
            load_imbalance,
        }
    }

    /// Format a human-readable one-block summary of this report.
    pub fn format_summary(&self) -> String {
        let mut s = String::new();
        s.push_str("=== Tile Analysis Report ===\n");
        s.push_str(&format!("  Total tiles     : {}\n", self.total_tiles));
        s.push_str(&format!(
            "  Empty tiles     : {} ({:.1}%)\n",
            self.empty_tiles,
            100.0 * self.empty_tiles as f32 / self.total_tiles.max(1) as f32,
        ));
        s.push_str(&format!("  Balanced tiles  : {}\n", self.balanced_tiles));
        s.push_str(&format!(
            "  Overloaded tiles: {} (threshold > {})\n",
            self.overloaded_tiles, self.overload_threshold
        ));
        s.push_str(&format!(
            "  Total projected : {}\n",
            self.total_gaussians_projected
        ));
        s.push_str(&format!("  Mean density    : {:.2}\n", self.mean_density));
        s.push_str(&format!("  Max density     : {}\n", self.max_density));
        s.push_str(&format!(
            "  Load imbalance  : {:.4} (σ/μ)\n",
            self.load_imbalance
        ));
        if !self.hotspot_locations.is_empty() {
            s.push_str("  Top hotspots    :");
            for (tx, ty) in &self.hotspot_locations {
                s.push_str(&format!(" ({},{})", tx, ty));
            }
            s.push('\n');
        }
        s.push_str("============================\n");
        s
    }

    /// Suggest a better tile size based on density patterns.
    ///
    /// - If `max_density > overload_threshold * 2`: halve the tile size
    ///   (minimum 8).
    /// - If `mean_density < 5.0`: double the tile size (maximum 64).
    /// - Otherwise: keep `current_tile_size`.
    pub fn suggest_tile_size(&self, current_tile_size: usize) -> usize {
        if self.max_density > self.overload_threshold.saturating_mul(2) {
            (current_tile_size / 2).max(8)
        } else if self.mean_density < 5.0 {
            (current_tile_size * 2).min(64)
        } else {
            current_tile_size
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── TileStats default ─────────────────────────────────────────────────

    #[test]
    fn test_tile_stats_default() {
        let t = TileStats::default();
        assert_eq!(t.gaussian_count, 0);
        assert_eq!(t.coverage_fraction, 0.0);
        assert_eq!(t.min_depth, 0.0);
        assert_eq!(t.max_depth, 0.0);
        assert_eq!(t.memory_pressure_bytes, 0);
        assert!(t.is_empty());
        assert_eq!(t.depth_range(), 0.0);
    }

    // ── TileStatsGrid construction ────────────────────────────────────────

    #[test]
    fn test_tile_stats_grid_new() {
        let grid = TileStatsGrid::new(64, 64, 16);
        assert_eq!(grid.width_tiles, 4);
        assert_eq!(grid.height_tiles, 4);
        assert_eq!(grid.total_tiles(), 16);
        assert_eq!(grid.tiles.len(), 16);

        // All tiles start empty.
        assert_eq!(grid.empty_tile_count(), 16);

        // All min_depths are INFINITY.
        for tile in &grid.tiles {
            assert_eq!(tile.min_depth, f32::INFINITY);
        }
    }

    #[test]
    fn test_tile_stats_grid_new_non_divisible() {
        // 100 / 16 = 6.25 → ceil → 7 tiles along each axis.
        let grid = TileStatsGrid::new(100, 100, 16);
        assert_eq!(grid.width_tiles, 7);
        assert_eq!(grid.height_tiles, 7);
        assert_eq!(grid.total_tiles(), 49);
    }

    // ── tile_for_pixel ────────────────────────────────────────────────────

    #[test]
    fn test_tile_stats_grid_tile_for_pixel() {
        let grid = TileStatsGrid::new(64, 64, 16);
        assert_eq!(grid.tile_for_pixel(0, 0), (0, 0));
        assert_eq!(grid.tile_for_pixel(15, 15), (0, 0));
        assert_eq!(grid.tile_for_pixel(16, 16), (1, 1));
        assert_eq!(grid.tile_for_pixel(63, 63), (3, 3));
        // Pixel at exact boundary.
        assert_eq!(grid.tile_for_pixel(32, 0), (2, 0));
    }

    // ── tile_at and tile_at_mut ───────────────────────────────────────────

    #[test]
    fn test_tile_stats_grid_tile_at() {
        let grid = TileStatsGrid::new(32, 32, 16);
        // Valid coordinates.
        assert!(grid.tile_at(0, 0).is_some());
        assert!(grid.tile_at(1, 1).is_some());
        // Out of bounds.
        assert!(grid.tile_at(2, 0).is_none()); // only 2×2 tiles
        assert!(grid.tile_at(0, 2).is_none());
    }

    // ── compute_tile_stats error cases ────────────────────────────────────

    #[test]
    fn test_compute_tile_stats_empty_input_error() {
        let result = compute_tile_stats(&[], &[], &[], 64, 64, 16);
        assert!(matches!(result, Err(TileStatsError::EmptyInput)));
    }

    #[test]
    fn test_compute_tile_stats_zero_tile_size_error() {
        let result = compute_tile_stats(&[[0.0, 0.0]], &[1.0], &[1.0], 64, 64, 0);
        assert!(matches!(result, Err(TileStatsError::ZeroTileSize)));
    }

    #[test]
    fn test_compute_tile_stats_zero_image_dimension_error() {
        let result = compute_tile_stats(&[[0.0, 0.0]], &[1.0], &[1.0], 0, 64, 16);
        assert!(matches!(result, Err(TileStatsError::ZeroImageDimension)));
    }

    #[test]
    fn test_compute_tile_stats_length_mismatch_radii() {
        let result = compute_tile_stats(
            &[[8.0, 8.0]],
            &[], // wrong length
            &[1.0],
            64,
            64,
            16,
        );
        assert!(matches!(
            result,
            Err(TileStatsError::LengthMismatch {
                expected: 1,
                got: 0
            })
        ));
    }

    #[test]
    fn test_compute_tile_stats_length_mismatch_depths() {
        let result = compute_tile_stats(
            &[[8.0, 8.0]],
            &[4.0],
            &[], // wrong length
            64,
            64,
            16,
        );
        assert!(matches!(
            result,
            Err(TileStatsError::LengthMismatch {
                expected: 1,
                got: 0
            })
        ));
    }

    // ── compute_tile_stats single Gaussian ────────────────────────────────

    #[test]
    fn test_compute_tile_stats_single_gaussian_center() {
        // 64×64 image, tile_size=16 → 4×4 grid.
        // Gaussian at (24, 24) radius=4 → AABB [20,28]×[20,28].
        // Tile: 20/16=1, 28/16=1 → only tile (1,1).
        let grid = compute_tile_stats(&[[24.0, 24.0]], &[4.0], &[1.5], 64, 64, 16)
            .expect("single Gaussian");

        let t = grid.tile_at(1, 1).expect("tile (1,1)");
        assert_eq!(t.gaussian_count, 1);
        assert!((t.min_depth - 1.5).abs() < 1e-6);
        assert!((t.max_depth - 1.5).abs() < 1e-6);
        assert!(t.coverage_fraction > 0.0 && t.coverage_fraction <= 1.0);
        assert_eq!(t.memory_pressure_bytes, BYTES_PER_GAUSSIAN_ACTIVATION);

        // All other tiles empty.
        assert_eq!(grid.empty_tile_count(), 15);
    }

    // ── compute_tile_stats multiple Gaussians ─────────────────────────────

    #[test]
    fn test_compute_tile_stats_multiple_gaussians() {
        // 16×16 image, tile_size=16 → single tile.
        // Three Gaussians all at (8,8) radius 2 → all land in tile (0,0).
        let positions = vec![[8.0f32, 8.0f32]; 3];
        let radii = vec![2.0f32; 3];
        let depths = vec![0.5f32, 1.0f32, 1.5f32];

        let grid =
            compute_tile_stats(&positions, &radii, &depths, 16, 16, 16).expect("three Gaussians");

        assert_eq!(grid.total_tiles(), 1);
        let t = grid.tile_at(0, 0).expect("only tile");
        assert_eq!(t.gaussian_count, 3);
        assert!((t.min_depth - 0.5).abs() < 1e-6);
        assert!((t.max_depth - 1.5).abs() < 1e-6);
        assert_eq!(t.memory_pressure_bytes, 3 * BYTES_PER_GAUSSIAN_ACTIVATION);
    }

    // ── Gaussian spanning multiple tiles ──────────────────────────────────

    #[test]
    fn test_compute_tile_stats_gaussian_spanning_tiles() {
        // 32×32 image, tile_size=16 → 2×2 tiles.
        // Gaussian at (16,16) radius=4 → AABB [12,20]×[12,20].
        // tile_x: 12/16=0 .. 20/16=1 → both tiles along X
        // tile_y: same → all 4 tiles touched.
        let grid = compute_tile_stats(&[[16.0, 16.0]], &[4.0], &[2.0], 32, 32, 16)
            .expect("spanning Gaussian");

        assert_eq!(grid.total_tiles(), 4);
        assert_eq!(grid.empty_tile_count(), 0);
        for ty in 0..2 {
            for tx in 0..2 {
                let t = grid.tile_at(tx, ty).expect("tile");
                assert_eq!(t.gaussian_count, 1, "tile ({tx},{ty})");
            }
        }
    }

    // ── hotspot_tile_indices ──────────────────────────────────────────────

    #[test]
    fn test_tile_stats_grid_hotspot_indices() {
        // 32×16 image (2 tiles wide, 1 tile tall).
        // Left tile gets 3 Gaussians, right gets 1.
        let positions = vec![[8.0f32, 8.0f32], [8.0, 8.0], [8.0, 8.0], [24.0, 8.0]];
        let radii = vec![2.0f32; 4];
        let depths = vec![1.0f32; 4];

        let grid =
            compute_tile_stats(&positions, &radii, &depths, 32, 16, 16).expect("hotspot grid");

        let hotspots = grid.hotspot_tile_indices(2);
        assert_eq!(hotspots.len(), 2);
        // First hotspot should be the left (denser) tile (0,0).
        assert_eq!(hotspots[0], (0, 0));
        // Second is the right tile (1,0).
        assert_eq!(hotspots[1], (1, 0));
    }

    // ── empty_tile_count ──────────────────────────────────────────────────

    #[test]
    fn test_tile_stats_grid_empty_tile_count() {
        // 64×16 image = 4 tiles wide × 1 tile tall.
        // One Gaussian at (8,8) → only tile (0,0) is non-empty.
        let grid = compute_tile_stats(&[[8.0, 8.0]], &[2.0], &[1.0], 64, 16, 16).expect("grid");

        assert_eq!(grid.empty_tile_count(), 3); // 3 out of 4 are empty
    }

    // ── Heatmap ───────────────────────────────────────────────────────────

    #[test]
    fn test_heatmap_from_grid_density() {
        let grid = compute_tile_stats(&[[8.0, 8.0]], &[2.0], &[1.0], 32, 32, 16).expect("grid");

        let heatmap = TileHeatmap::from_grid(&grid, HeatmapMode::GaussianDensity);
        assert_eq!(heatmap.mode, HeatmapMode::GaussianDensity);
        assert_eq!(heatmap.width, grid.width_tiles);
        assert_eq!(heatmap.height, grid.height_tiles);
        assert_eq!(heatmap.pixels.len(), grid.total_tiles());

        // Every pixel has alpha 255.
        for px in &heatmap.pixels {
            assert_eq!(px[3], 255);
        }
    }

    // ── ASCII heatmap length ──────────────────────────────────────────────

    #[test]
    fn test_heatmap_to_ascii_length() {
        let grid = compute_tile_stats(&[[8.0, 8.0]], &[2.0], &[1.0], 32, 32, 16).expect("grid");

        let heatmap = TileHeatmap::from_grid(&grid, HeatmapMode::Coverage);
        let ascii = heatmap.to_ascii();

        // Each row: width_tiles chars + '\n'.
        let expected_len = grid.height_tiles * (grid.width_tiles + 1);
        assert_eq!(ascii.len(), expected_len);
    }

    // ── TileAnalysisReport ────────────────────────────────────────────────

    #[test]
    fn test_analysis_report_compute() {
        // 32×16 image → 2×1 tiles.
        // Left: 3 Gaussians, right: 1 Gaussian. Overload threshold = 2.
        let positions = vec![[8.0f32, 8.0f32], [8.0, 8.0], [8.0, 8.0], [24.0, 8.0]];
        let radii = vec![2.0f32; 4];
        let depths = vec![1.0f32; 4];

        let grid = compute_tile_stats(&positions, &radii, &depths, 32, 16, 16).expect("grid");

        let report = TileAnalysisReport::compute(&grid, 2);

        assert_eq!(report.total_tiles, 2);
        assert_eq!(report.empty_tiles, 0);
        assert_eq!(report.overloaded_tiles, 1); // left tile count=3 > threshold=2
        assert_eq!(report.balanced_tiles, 1); // right tile count=1 ≤ threshold=2
        assert_eq!(report.overload_threshold, 2);
        assert_eq!(report.max_density, 3);
        assert!((report.mean_density - 2.0).abs() < 1e-4); // (3+1)/2 = 2.0
        assert!(report.load_imbalance >= 0.0);
    }

    // ── suggest_tile_size: halve ──────────────────────────────────────────

    #[test]
    fn test_analysis_report_suggest_tile_size_halve() {
        // overload_threshold = 2, overload_threshold*2 = 4
        // max_density = 5 > 4 → halve.
        let mut report = TileAnalysisReport {
            total_tiles: 4,
            empty_tiles: 0,
            overloaded_tiles: 1,
            balanced_tiles: 3,
            overload_threshold: 2,
            hotspot_locations: Vec::new(),
            total_gaussians_projected: 20,
            mean_density: 5.0,
            max_density: 5,
            load_imbalance: 0.2,
        };
        // max_density(5) > overload_threshold*2(4) → halve.
        assert_eq!(report.suggest_tile_size(16), 8);
        // Minimum is 8.
        assert_eq!(report.suggest_tile_size(8), 8);
        // Corner: threshold=0 → 0*2=0, max_density=5>0 → halve.
        report.overload_threshold = 0;
        assert_eq!(report.suggest_tile_size(16), 8);
    }

    // ── suggest_tile_size: double ─────────────────────────────────────────

    #[test]
    fn test_analysis_report_suggest_tile_size_double() {
        // mean_density < 5.0 → double.
        let report = TileAnalysisReport {
            total_tiles: 100,
            empty_tiles: 90,
            overloaded_tiles: 0,
            balanced_tiles: 10,
            overload_threshold: 64,
            hotspot_locations: Vec::new(),
            total_gaussians_projected: 10,
            mean_density: 0.1,
            max_density: 1,
            load_imbalance: 0.0,
        };
        assert_eq!(report.suggest_tile_size(16), 32);
        // Maximum is 64.
        assert_eq!(report.suggest_tile_size(64), 64);
        assert_eq!(report.suggest_tile_size(32), 64);
    }

    // ── format_summary ────────────────────────────────────────────────────

    #[test]
    fn test_analysis_report_format_summary() {
        let grid = compute_tile_stats(
            &[[8.0, 8.0], [24.0, 8.0]],
            &[2.0, 2.0],
            &[1.0, 2.0],
            32,
            16,
            16,
        )
        .expect("grid");

        let report = TileAnalysisReport::compute(&grid, 5);
        let summary = report.format_summary();

        // Must contain the section header.
        assert!(summary.contains("Tile Analysis Report"));
        // Must mention total tiles.
        assert!(summary.contains("Total tiles"));
        // Must mention mean density.
        assert!(summary.contains("Mean density"));
        // Must mention load imbalance.
        assert!(summary.contains("Load imbalance"));
    }

    // ── depth_range helper ────────────────────────────────────────────────

    #[test]
    fn test_tile_stats_depth_range() {
        let grid = compute_tile_stats(
            &[[8.0, 8.0], [8.0, 8.0], [8.0, 8.0]],
            &[2.0, 2.0, 2.0],
            &[0.5, 1.5, 3.0],
            16,
            16,
            16,
        )
        .expect("grid");

        let t = grid.tile_at(0, 0).expect("tile");
        assert!((t.min_depth - 0.5).abs() < 1e-6);
        assert!((t.max_depth - 3.0).abs() < 1e-6);
        assert!((t.depth_range() - 2.5).abs() < 1e-5);
    }

    // ── Jet colormap extremes ─────────────────────────────────────────────

    #[test]
    fn test_jet_rgb_extremes() {
        // At v=0 (cold): blue dominant.
        let cold = jet_rgb(0.0);
        assert!(cold[2] > cold[0], "blue > red at v=0");

        // At v=1 (hot): red dominant.
        let hot = jet_rgb(1.0);
        assert!(hot[0] > hot[2], "red > blue at v=1");
    }
}
