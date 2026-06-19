//! Scene merging for 3D Gaussian Splatting reconstructions.
//!
//! This module provides tools to merge multiple 3DGS scenes into a single
//! combined scene. This is useful for:
//! - Compositing objects from separate reconstructions
//! - Stitching scenes with limited FOV
//! - Combining foreground and background elements
//!
//! # Example
//! ```rust,no_run
//! use oxigaf_cli::scene_merge::{
//!     GaussianEntry, SceneGaussians, SceneMergeConfig, merge_scenes,
//! };
//!
//! let gaussians = vec![
//!     GaussianEntry {
//!         position: [0.0, 0.0, 0.0],
//!         log_scale: [-2.0; 3],
//!         rotation: [0.0, 0.0, 0.0, 1.0],
//!         opacity: 0.8,
//!         color: [0.5; 3],
//!         sh_coeffs: vec![],
//!     },
//! ];
//! let scene = SceneGaussians::new(gaussians);
//! let config = SceneMergeConfig::default();
//! let merged = merge_scenes(&[scene], &config).expect("merge failed");
//! println!("Merged {} Gaussians", merged.len());
//! ```

use std::collections::HashMap;

use thiserror::Error;

// ---------------------------------------------------------------------------
// MergeError
// ---------------------------------------------------------------------------

/// Errors that can occur during scene merging operations.
#[derive(Debug, Error)]
pub enum MergeError {
    /// No scenes were provided to merge.
    #[error("No scenes to merge")]
    EmptyScenes,

    /// The merge configuration is invalid.
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Inconsistent field lengths within a scene.
    #[error("Length mismatch for field '{field}': expected {expected}, got {actual}")]
    LengthMismatch {
        field: String,
        expected: usize,
        actual: usize,
    },
}

// ---------------------------------------------------------------------------
// GaussianEntry
// ---------------------------------------------------------------------------

/// One Gaussian's parameters packed flat.
#[derive(Debug, Clone)]
pub struct GaussianEntry {
    /// World-space center [x, y, z].
    pub position: [f32; 3],
    /// Log-space scales [log_sx, log_sy, log_sz].
    pub log_scale: [f32; 3],
    /// Quaternion rotation [qx, qy, qz, qw].
    pub rotation: [f32; 4],
    /// Actual opacity in [0, 1].
    pub opacity: f32,
    /// DC SH color [r, g, b] in [0, 1].
    pub color: [f32; 3],
    /// Higher-order spherical harmonics coefficients (can be empty).
    pub sh_coeffs: Vec<f32>,
}

impl GaussianEntry {
    /// Maximum scale (exp of max log_scale component).
    #[must_use]
    pub fn max_scale(&self) -> f32 {
        let max_log = self
            .log_scale
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        max_log.exp()
    }

    /// Volume approximation: product of scales = exp(log_sx + log_sy + log_sz).
    #[must_use]
    pub fn volume(&self) -> f32 {
        let sum_log = self.log_scale[0] + self.log_scale[1] + self.log_scale[2];
        sum_log.exp()
    }

    /// Euclidean distance from a given world-space point.
    #[must_use]
    pub fn distance_from(&self, point: [f32; 3]) -> f32 {
        let dx = self.position[0] - point[0];
        let dy = self.position[1] - point[1];
        let dz = self.position[2] - point[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

// ---------------------------------------------------------------------------
// SceneGaussians
// ---------------------------------------------------------------------------

/// A collection of Gaussians forming a scene.
#[derive(Debug, Clone)]
pub struct SceneGaussians {
    /// All Gaussians in this scene.
    pub gaussians: Vec<GaussianEntry>,
    /// Optional 4×4 row-major homogeneous transform applied to positions during merging.
    pub transform: Option<[f32; 16]>,
    /// Optional scene name / label.
    pub name: Option<String>,
}

impl SceneGaussians {
    /// Create a new scene from a list of Gaussians.
    #[must_use]
    pub fn new(gaussians: Vec<GaussianEntry>) -> Self {
        Self {
            gaussians,
            transform: None,
            name: None,
        }
    }

    /// Number of Gaussians in this scene.
    #[must_use]
    pub fn len(&self) -> usize {
        self.gaussians.len()
    }

    /// Returns `true` if this scene contains no Gaussians.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.gaussians.is_empty()
    }

    /// Set the scene name (builder style).
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set a 4×4 row-major transform (builder style).
    #[must_use]
    pub fn with_transform(mut self, transform: [f32; 16]) -> Self {
        self.transform = Some(transform);
        self
    }

    /// Compute the axis-aligned bounding box of all Gaussian positions.
    ///
    /// Returns `None` if the scene is empty.
    #[must_use]
    pub fn bounding_box(&self) -> Option<([f32; 3], [f32; 3])> {
        if self.gaussians.is_empty() {
            return None;
        }
        let first = self.gaussians[0].position;
        let (min_xyz, max_xyz) =
            self.gaussians
                .iter()
                .map(|g| g.position)
                .fold((first, first), |(mn, mx), p| {
                    (
                        [mn[0].min(p[0]), mn[1].min(p[1]), mn[2].min(p[2])],
                        [mx[0].max(p[0]), mx[1].max(p[1]), mx[2].max(p[2])],
                    )
                });
        Some((min_xyz, max_xyz))
    }

    /// Centroid of all Gaussian positions. Returns `[0,0,0]` if empty.
    #[must_use]
    pub fn centroid(&self) -> [f32; 3] {
        if self.gaussians.is_empty() {
            return [0.0; 3];
        }
        let n = self.gaussians.len() as f32;
        let sum = self.gaussians.iter().fold([0.0f32; 3], |acc, g| {
            [
                acc[0] + g.position[0],
                acc[1] + g.position[1],
                acc[2] + g.position[2],
            ]
        });
        [sum[0] / n, sum[1] / n, sum[2] / n]
    }

    /// Mean opacity across all Gaussians. Returns 0.0 if empty.
    #[must_use]
    pub fn mean_opacity(&self) -> f32 {
        if self.gaussians.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.gaussians.iter().map(|g| g.opacity).sum();
        sum / (self.gaussians.len() as f32)
    }

    /// Mean max-scale across all Gaussians. Returns 0.0 if empty.
    #[must_use]
    pub fn mean_max_scale(&self) -> f32 {
        if self.gaussians.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.gaussians.iter().map(|g| g.max_scale()).sum();
        sum / (self.gaussians.len() as f32)
    }
}

// ---------------------------------------------------------------------------
// SceneMergeConfig
// ---------------------------------------------------------------------------

/// Configuration controlling how scenes are merged.
#[derive(Debug, Clone)]
pub struct SceneMergeConfig {
    /// Remove Gaussians below this opacity from each scene before merging.
    pub min_opacity: f32,
    /// Remove Gaussians above this max scale (0 = no limit).
    pub max_scale: f32,
    /// Whether to remove near-duplicate Gaussians after merging.
    pub remove_duplicates: bool,
    /// Distance threshold for duplicate detection.
    pub duplicate_threshold: f32,
    /// Re-normalize opacities after merge so `mean_opacity == target_opacity`.
    pub normalize_opacities: bool,
    /// Target mean opacity when `normalize_opacities` is `true`.
    pub target_opacity: f32,
    /// Maximum total Gaussians in output (0 = no limit). Excess pruned by opacity.
    pub max_gaussians: usize,
    /// Whether to apply scene transforms during merge.
    pub apply_transforms: bool,
}

impl Default for SceneMergeConfig {
    fn default() -> Self {
        Self {
            min_opacity: 0.005,
            max_scale: 0.0,
            remove_duplicates: false,
            duplicate_threshold: 0.001,
            normalize_opacities: false,
            target_opacity: 0.3,
            max_gaussians: 0,
            apply_transforms: true,
        }
    }
}

impl SceneMergeConfig {
    /// Validate the configuration, returning an error on invalid combinations.
    pub fn validate(&self) -> Result<(), MergeError> {
        if !(0.0..=1.0).contains(&self.min_opacity) {
            return Err(MergeError::InvalidConfig(format!(
                "min_opacity must be in [0,1], got {}",
                self.min_opacity
            )));
        }
        if self.remove_duplicates && self.duplicate_threshold <= 0.0 {
            return Err(MergeError::InvalidConfig(format!(
                "duplicate_threshold must be > 0 when remove_duplicates is true, got {}",
                self.duplicate_threshold
            )));
        }
        if self.normalize_opacities && (self.target_opacity <= 0.0 || self.target_opacity > 1.0) {
            return Err(MergeError::InvalidConfig(format!(
                "target_opacity must be in (0,1] when normalize_opacities is true, got {}",
                self.target_opacity
            )));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Transform application
// ---------------------------------------------------------------------------

/// Apply a 4×4 row-major homogeneous transform to a 3D point.
///
/// The matrix layout is:
/// ```text
/// [ m[0]  m[1]  m[2]  m[3]  ]   row 0
/// [ m[4]  m[5]  m[6]  m[7]  ]   row 1
/// [ m[8]  m[9]  m[10] m[11] ]   row 2
/// [ m[12] m[13] m[14] m[15] ]   row 3 (homogeneous)
/// ```
#[must_use]
pub fn apply_transform(mat: &[f32; 16], point: [f32; 3]) -> [f32; 3] {
    let x = point[0];
    let y = point[1];
    let z = point[2];
    let out_x = mat[0] * x + mat[1] * y + mat[2] * z + mat[3];
    let out_y = mat[4] * x + mat[5] * y + mat[6] * z + mat[7];
    let out_z = mat[8] * x + mat[9] * y + mat[10] * z + mat[11];
    let w = mat[12] * x + mat[13] * y + mat[14] * z + mat[15];
    if w.abs() > 1e-10 {
        [out_x / w, out_y / w, out_z / w]
    } else {
        [out_x, out_y, out_z]
    }
}

/// Apply the rotation part of a 4×4 row-major transform to a quaternion.
///
/// Extracts the 3×3 rotation sub-matrix from `mat` (columns normalized to
/// remove any embedded scale), computes the composition `R_mat * R_quat`,
/// and returns the result as a unit quaternion `[qx, qy, qz, qw]`.
#[must_use]
pub fn apply_transform_rotation(mat: &[f32; 16], quat: [f32; 4]) -> [f32; 4] {
    // Extract raw 3×3 from upper-left of the row-major 4×4.
    // mat layout (row-major):
    //   row0: mat[0..4], row1: mat[4..8], row2: mat[8..12]
    // element (r,c) = mat[r*4 + c]
    let mut col0 = [mat[0], mat[4], mat[8]];
    let mut col1 = [mat[1], mat[5], mat[9]];
    let mut col2 = [mat[2], mat[6], mat[10]];

    // Normalize columns to strip scale.
    col0 = normalize3_safe(col0);
    col1 = normalize3_safe(col1);
    col2 = normalize3_safe(col2);

    // Build normalized rotation matrix (row-major): mat_r[row][col]
    let mat_r = [
        [col0[0], col1[0], col2[0]], // row 0
        [col0[1], col1[1], col2[1]], // row 1
        [col0[2], col1[2], col2[2]], // row 2
    ];

    // Convert quat → rotation matrix R_q, then compose R_out = mat_r * R_q.
    let r_q = quat_to_rotation_matrix(quat);
    let r_out = mat3_mul(mat_r, r_q);

    // Convert back to quaternion and renormalize.
    rotation_matrix_to_quat(r_out)
}

// ---------------------------------------------------------------------------
// Quaternion / rotation matrix helpers
// ---------------------------------------------------------------------------

/// Convert a quaternion `[qx, qy, qz, qw]` to a 3×3 rotation matrix.
///
/// The returned matrix is in row-major order: `r[row][col]`.
fn quat_to_rotation_matrix(q: [f32; 4]) -> [[f32; 3]; 3] {
    let (qx, qy, qz, qw) = (q[0], q[1], q[2], q[3]);

    // Normalize first to be safe.
    let norm = (qx * qx + qy * qy + qz * qz + qw * qw).sqrt();
    let (qx, qy, qz, qw) = if norm > 1e-10 {
        (qx / norm, qy / norm, qz / norm, qw / norm)
    } else {
        (0.0, 0.0, 0.0, 1.0)
    };

    [
        [
            1.0 - 2.0 * (qy * qy + qz * qz),
            2.0 * (qx * qy - qz * qw),
            2.0 * (qx * qz + qy * qw),
        ],
        [
            2.0 * (qx * qy + qz * qw),
            1.0 - 2.0 * (qx * qx + qz * qz),
            2.0 * (qy * qz - qx * qw),
        ],
        [
            2.0 * (qx * qz - qy * qw),
            2.0 * (qy * qz + qx * qw),
            1.0 - 2.0 * (qx * qx + qy * qy),
        ],
    ]
}

/// Convert a 3×3 rotation matrix (row-major) to a unit quaternion `[qx, qy, qz, qw]`
/// using the Shepperd method.
fn rotation_matrix_to_quat(m: [[f32; 3]; 3]) -> [f32; 4] {
    let trace = m[0][0] + m[1][1] + m[2][2];

    let (x, y, z, w) = if trace > 0.0 {
        let s = 0.5 / (trace + 1.0).sqrt();
        let w = 0.25 / s;
        let x = (m[2][1] - m[1][2]) * s;
        let y = (m[0][2] - m[2][0]) * s;
        let z = (m[1][0] - m[0][1]) * s;
        (x, y, z, w)
    } else if m[0][0] > m[1][1] && m[0][0] > m[2][2] {
        let s = 2.0 * (1.0 + m[0][0] - m[1][1] - m[2][2]).sqrt();
        let w = (m[2][1] - m[1][2]) / s;
        let x = 0.25 * s;
        let y = (m[0][1] + m[1][0]) / s;
        let z = (m[0][2] + m[2][0]) / s;
        (x, y, z, w)
    } else if m[1][1] > m[2][2] {
        let s = 2.0 * (1.0 + m[1][1] - m[0][0] - m[2][2]).sqrt();
        let w = (m[0][2] - m[2][0]) / s;
        let x = (m[0][1] + m[1][0]) / s;
        let y = 0.25 * s;
        let z = (m[1][2] + m[2][1]) / s;
        (x, y, z, w)
    } else {
        let s = 2.0 * (1.0 + m[2][2] - m[0][0] - m[1][1]).sqrt();
        let w = (m[1][0] - m[0][1]) / s;
        let x = (m[0][2] + m[2][0]) / s;
        let y = (m[1][2] + m[2][1]) / s;
        let z = 0.25 * s;
        (x, y, z, w)
    };

    // Renormalize to unit quaternion.
    let norm = (x * x + y * y + z * z + w * w).sqrt();
    if norm > 1e-10 {
        [x / norm, y / norm, z / norm, w / norm]
    } else {
        [0.0, 0.0, 0.0, 1.0]
    }
}

/// Multiply two 3×3 row-major matrices: `a * b`.
fn mat3_mul(a: [[f32; 3]; 3], b: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let mut out = [[0.0f32; 3]; 3];
    for row in 0..3 {
        for col in 0..3 {
            for k in 0..3 {
                out[row][col] += a[row][k] * b[k][col];
            }
        }
    }
    out
}

/// Normalize a 3-vector, returning the input unchanged if its norm is near zero.
#[inline]
fn normalize3_safe(v: [f32; 3]) -> [f32; 3] {
    let norm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if norm > 1e-10 {
        [v[0] / norm, v[1] / norm, v[2] / norm]
    } else {
        v
    }
}

// ---------------------------------------------------------------------------
// Duplicate removal
// ---------------------------------------------------------------------------

/// Find duplicate Gaussians using a grid-based approach.
///
/// Groups Gaussians into cells of size `threshold × threshold × threshold`.
/// For each cell, only the highest-opacity Gaussian is kept.
///
/// Returns a mask where `true` means "keep" and `false` means "duplicate to remove".
#[must_use]
pub fn find_duplicates(gaussians: &[GaussianEntry], threshold: f32) -> Vec<bool> {
    let n = gaussians.len();
    if n == 0 {
        return Vec::new();
    }

    // cell index for a coordinate component: floor(pos / threshold) as i64
    let cell_of = |pos: f32| -> i64 { (pos / threshold).floor() as i64 };

    // Map: cell_key → index of the best (highest opacity) Gaussian in that cell.
    let mut best: HashMap<(i64, i64, i64), usize> = HashMap::new();

    for (idx, g) in gaussians.iter().enumerate() {
        let key = (
            cell_of(g.position[0]),
            cell_of(g.position[1]),
            cell_of(g.position[2]),
        );
        match best.get(&key) {
            None => {
                best.insert(key, idx);
            }
            Some(&prev_idx) => {
                if g.opacity > gaussians[prev_idx].opacity {
                    best.insert(key, idx);
                }
            }
        }
    }

    // Build mask: only the winner per cell is kept.
    let winners: std::collections::HashSet<usize> = best.into_values().collect();
    (0..n).map(|i| winners.contains(&i)).collect()
}

// ---------------------------------------------------------------------------
// Boundary blending helper
// ---------------------------------------------------------------------------

/// Smoothstep function: 3t² - 2t³ where t = clamp((x - lo) / (hi - lo), 0, 1).
#[inline]
fn smoothstep(lo: f32, hi: f32, x: f32) -> f32 {
    if hi <= lo {
        return if x < lo { 0.0 } else { 1.0 };
    }
    let t = ((x - lo) / (hi - lo)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

// ---------------------------------------------------------------------------
// Core merge functions
// ---------------------------------------------------------------------------

/// Merge multiple scenes into a single `SceneGaussians`, applying filtering
/// and post-processing according to `config`.
pub fn merge_scenes(
    scenes: &[SceneGaussians],
    config: &SceneMergeConfig,
) -> Result<SceneGaussians, MergeError> {
    if scenes.is_empty() {
        return Err(MergeError::EmptyScenes);
    }
    config.validate()?;

    let mut merged: Vec<GaussianEntry> = Vec::new();

    for scene in scenes {
        let has_transform = config.apply_transforms && scene.transform.is_some();
        let mat = scene.transform;

        for g in &scene.gaussians {
            // Filter by min_opacity.
            if g.opacity < config.min_opacity {
                continue;
            }
            // Filter by max_scale (0 = no limit).
            if config.max_scale > 0.0 && g.max_scale() > config.max_scale {
                continue;
            }

            let mut entry = g.clone();

            // Apply scene transform if requested.
            if has_transform {
                if let Some(ref m) = mat {
                    entry.position = apply_transform(m, g.position);
                    entry.rotation = apply_transform_rotation(m, g.rotation);
                }
            }

            merged.push(entry);
        }
    }

    // Duplicate removal.
    if config.remove_duplicates && !merged.is_empty() {
        let mask = find_duplicates(&merged, config.duplicate_threshold);
        merged = merged
            .into_iter()
            .zip(mask)
            .filter_map(|(g, keep)| if keep { Some(g) } else { None })
            .collect();
    }

    // Opacity normalization.
    if config.normalize_opacities && !merged.is_empty() {
        let mean: f32 = merged.iter().map(|g| g.opacity).sum::<f32>() / (merged.len() as f32);
        if mean > 1e-10 {
            let scale = (config.target_opacity / mean).min(1.0 / mean);
            for g in &mut merged {
                g.opacity = (g.opacity * scale).clamp(0.0, 1.0);
            }
        }
    }

    // Prune to max_gaussians by opacity (descending).
    if config.max_gaussians > 0 && merged.len() > config.max_gaussians {
        // Partial sort: sort descending by opacity, keep top N.
        merged.sort_unstable_by(|a, b| {
            b.opacity
                .partial_cmp(&a.opacity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        merged.truncate(config.max_gaussians);
    }

    Ok(SceneGaussians::new(merged))
}

/// Simple concatenation of scenes without any filtering.
///
/// Returns `Err(EmptyScenes)` if `scenes` is empty.
pub fn concatenate_scenes(scenes: &[SceneGaussians]) -> Result<SceneGaussians, MergeError> {
    if scenes.is_empty() {
        return Err(MergeError::EmptyScenes);
    }
    let total: usize = scenes.iter().map(|s| s.len()).sum();
    let mut merged = Vec::with_capacity(total);
    for scene in scenes {
        merged.extend(scene.gaussians.iter().cloned());
    }
    Ok(SceneGaussians::new(merged))
}

/// Merge two scenes with spatial blending at their boundary.
///
/// - `boundary_axis`: axis along which the boundary runs (0=X, 1=Y, 2=Z).
/// - `boundary_pos`: world-space coordinate of the boundary.
/// - `transition_width`: width of the blending zone.
///
/// Scene A fades out approaching `boundary_pos` from the negative side.
/// Scene B fades out approaching `boundary_pos` from the positive side.
pub fn merge_at_boundary(
    scene_a: &SceneGaussians,
    scene_b: &SceneGaussians,
    boundary_axis: usize,
    boundary_pos: f32,
    transition_width: f32,
) -> Result<SceneGaussians, MergeError> {
    if boundary_axis > 2 {
        return Err(MergeError::InvalidConfig(format!(
            "boundary_axis must be 0, 1, or 2, got {}",
            boundary_axis
        )));
    }
    if transition_width < 0.0 {
        return Err(MergeError::InvalidConfig(format!(
            "transition_width must be >= 0, got {}",
            transition_width
        )));
    }

    let mut merged = Vec::with_capacity(scene_a.len() + scene_b.len());

    // Scene A: fades out from (boundary_pos - transition_width) → boundary_pos.
    // blend goes from 0 → 1 in that zone; final opacity = opacity * (1 - blend).
    let a_lo = boundary_pos - transition_width;
    let a_hi = boundary_pos;
    for g in &scene_a.gaussians {
        let pos_axis = g.position[boundary_axis];
        let blend = smoothstep(a_lo, a_hi, pos_axis);
        let new_opacity = g.opacity * (1.0 - blend);
        let mut entry = g.clone();
        entry.opacity = new_opacity;
        merged.push(entry);
    }

    // Scene B: fades out from boundary_pos → (boundary_pos + transition_width).
    // blend goes from 0 → 1 in that zone; final opacity = opacity * blend (reversed: 1→0 in zone).
    // We want B to fade out *approaching* boundary from positive side, so it fades in as pos
    // increases beyond boundary_pos. We use smoothstep(boundary_pos, boundary_pos+width, pos)
    // which gives 0 at boundary_pos and 1 at boundary_pos+width — that's the "far" region.
    // At the boundary (pos ≈ boundary_pos), blend ≈ 0, so opacity * blend ≈ 0.
    // At boundary_pos + width, blend = 1, opacity = full.
    let b_lo = boundary_pos;
    let b_hi = boundary_pos + transition_width;
    for g in &scene_b.gaussians {
        let pos_axis = g.position[boundary_axis];
        let blend = smoothstep(b_lo, b_hi, pos_axis);
        let new_opacity = g.opacity * blend;
        let mut entry = g.clone();
        entry.opacity = new_opacity;
        merged.push(entry);
    }

    Ok(SceneGaussians::new(merged))
}

// ---------------------------------------------------------------------------
// MergeStats
// ---------------------------------------------------------------------------

/// Statistics gathered during a merge operation.
#[derive(Debug, Clone)]
pub struct MergeStats {
    /// Number of input scenes.
    pub input_scenes: usize,
    /// Gaussian count per input scene.
    pub input_gaussians: Vec<usize>,
    /// Total Gaussians across all input scenes.
    pub total_input: usize,
    /// Count after opacity filtering.
    pub after_opacity_filter: usize,
    /// Count after scale filtering (subset of after_opacity_filter).
    pub after_scale_filter: usize,
    /// Number of duplicates removed.
    pub duplicates_removed: usize,
    /// Final Gaussian count in the merged scene.
    pub final_count: usize,
    /// Bounding box of the final scene, if non-empty.
    pub final_bbox: Option<([f32; 3], [f32; 3])>,
}

impl MergeStats {
    /// Format a human-readable summary of the merge statistics.
    #[must_use]
    pub fn format_summary(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!("Merge summary: {} input scenes", self.input_scenes));
        for (i, count) in self.input_gaussians.iter().enumerate() {
            lines.push(format!("  Scene {}: {} Gaussians", i, count));
        }
        lines.push(format!("  Total input:          {}", self.total_input));
        lines.push(format!(
            "  After opacity filter: {}",
            self.after_opacity_filter
        ));
        lines.push(format!(
            "  After scale filter:   {}",
            self.after_scale_filter
        ));
        lines.push(format!(
            "  Duplicates removed:   {}",
            self.duplicates_removed
        ));
        lines.push(format!("  Final count:          {}", self.final_count));
        if let Some((mn, mx)) = &self.final_bbox {
            lines.push(format!(
                "  Bounding box: [{:.3}, {:.3}, {:.3}] → [{:.3}, {:.3}, {:.3}]",
                mn[0], mn[1], mn[2], mx[0], mx[1], mx[2],
            ));
        } else {
            lines.push("  Bounding box: (empty)".to_string());
        }
        lines.join("\n")
    }
}

/// Merge scenes and collect detailed statistics about the process.
pub fn merge_scenes_with_stats(
    scenes: &[SceneGaussians],
    config: &SceneMergeConfig,
) -> Result<(SceneGaussians, MergeStats), MergeError> {
    if scenes.is_empty() {
        return Err(MergeError::EmptyScenes);
    }
    config.validate()?;

    let input_scenes = scenes.len();
    let input_gaussians: Vec<usize> = scenes.iter().map(|s| s.len()).collect();
    let total_input: usize = input_gaussians.iter().sum();

    // Step-by-step filtering to track counts.
    let mut after_opacity: Vec<GaussianEntry> = Vec::new();
    let mut after_scale: Vec<GaussianEntry> = Vec::new();

    for scene in scenes {
        let has_transform = config.apply_transforms && scene.transform.is_some();
        let mat = scene.transform;

        for g in &scene.gaussians {
            // Opacity filter.
            if g.opacity < config.min_opacity {
                continue;
            }
            after_opacity.push(g.clone());

            // Scale filter.
            if config.max_scale > 0.0 && g.max_scale() > config.max_scale {
                continue;
            }

            let mut entry = g.clone();
            if has_transform {
                if let Some(ref m) = mat {
                    entry.position = apply_transform(m, g.position);
                    entry.rotation = apply_transform_rotation(m, g.rotation);
                }
            }
            after_scale.push(entry);
        }
    }

    let after_opacity_filter = after_opacity.len();
    let after_scale_filter = after_scale.len();

    let mut merged = after_scale;

    // Duplicate removal.
    let duplicates_removed;
    if config.remove_duplicates && !merged.is_empty() {
        let mask = find_duplicates(&merged, config.duplicate_threshold);
        let before = merged.len();
        merged = merged
            .into_iter()
            .zip(mask)
            .filter_map(|(g, keep)| if keep { Some(g) } else { None })
            .collect();
        duplicates_removed = before - merged.len();
    } else {
        duplicates_removed = 0;
    }

    // Opacity normalization.
    if config.normalize_opacities && !merged.is_empty() {
        let mean: f32 = merged.iter().map(|g| g.opacity).sum::<f32>() / (merged.len() as f32);
        if mean > 1e-10 {
            let scale = (config.target_opacity / mean).min(1.0 / mean);
            for g in &mut merged {
                g.opacity = (g.opacity * scale).clamp(0.0, 1.0);
            }
        }
    }

    // Prune to max_gaussians by opacity.
    if config.max_gaussians > 0 && merged.len() > config.max_gaussians {
        merged.sort_unstable_by(|a, b| {
            b.opacity
                .partial_cmp(&a.opacity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        merged.truncate(config.max_gaussians);
    }

    let final_count = merged.len();
    let result_scene = SceneGaussians::new(merged);
    let final_bbox = result_scene.bounding_box();

    let stats = MergeStats {
        input_scenes,
        input_gaussians,
        total_input,
        after_opacity_filter,
        after_scale_filter,
        duplicates_removed,
        final_count,
        final_bbox,
    };

    Ok((result_scene, stats))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_gaussian(pos: [f32; 3], opacity: f32) -> GaussianEntry {
        GaussianEntry {
            position: pos,
            log_scale: [-2.0; 3],
            rotation: [0.0, 0.0, 0.0, 1.0],
            opacity,
            color: [0.5; 3],
            sh_coeffs: vec![],
        }
    }

    fn make_scene(gaussians: Vec<GaussianEntry>) -> SceneGaussians {
        SceneGaussians::new(gaussians)
    }

    // -----------------------------------------------------------------------
    // Test 1: apply_transform with identity matrix → same point
    // -----------------------------------------------------------------------
    #[test]
    fn test_apply_transform_identity() {
        let identity = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0f32,
        ];
        let point = [1.0, 2.0, 3.0];
        let result = apply_transform(&identity, point);
        assert!((result[0] - 1.0).abs() < 1e-6);
        assert!((result[1] - 2.0).abs() < 1e-6);
        assert!((result[2] - 3.0).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // Test 2: apply_transform with translation matrix
    // -----------------------------------------------------------------------
    #[test]
    fn test_apply_transform_translation() {
        let translation = [
            1.0, 0.0, 0.0, 5.0, 0.0, 1.0, 0.0, -3.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 1.0f32,
        ];
        let point = [1.0, 0.0, 0.0];
        let result = apply_transform(&translation, point);
        assert!((result[0] - 6.0).abs() < 1e-6, "x={}", result[0]);
        assert!((result[1] - (-3.0)).abs() < 1e-6, "y={}", result[1]);
        assert!((result[2] - 2.0).abs() < 1e-6, "z={}", result[2]);
    }

    // -----------------------------------------------------------------------
    // Test 3: SceneGaussians::bounding_box with single Gaussian
    // -----------------------------------------------------------------------
    #[test]
    fn test_bounding_box_single() {
        let g = make_gaussian([1.0, 2.0, 3.0], 0.8);
        let scene = make_scene(vec![g]);
        let bbox = scene.bounding_box().expect("should have bbox");
        assert_eq!(bbox.0, [1.0, 2.0, 3.0]);
        assert_eq!(bbox.1, [1.0, 2.0, 3.0]);
    }

    // -----------------------------------------------------------------------
    // Test 4: SceneGaussians::bounding_box with empty scene → None
    // -----------------------------------------------------------------------
    #[test]
    fn test_bounding_box_empty() {
        let scene = make_scene(vec![]);
        assert!(scene.bounding_box().is_none());
    }

    // -----------------------------------------------------------------------
    // Test 5: SceneGaussians::centroid of two Gaussians → average position
    // -----------------------------------------------------------------------
    #[test]
    fn test_centroid_two_gaussians() {
        let g1 = make_gaussian([0.0, 0.0, 0.0], 0.5);
        let g2 = make_gaussian([2.0, 4.0, 6.0], 0.5);
        let scene = make_scene(vec![g1, g2]);
        let centroid = scene.centroid();
        assert!((centroid[0] - 1.0).abs() < 1e-6);
        assert!((centroid[1] - 2.0).abs() < 1e-6);
        assert!((centroid[2] - 3.0).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // Test 6: concatenate_scenes combines counts
    // -----------------------------------------------------------------------
    #[test]
    fn test_concatenate_scenes_count() {
        let s1 = make_scene(vec![
            make_gaussian([0.0; 3], 0.5),
            make_gaussian([1.0; 3], 0.5),
        ]);
        let s2 = make_scene(vec![make_gaussian([2.0; 3], 0.5)]);
        let merged = concatenate_scenes(&[s1, s2]).expect("concat failed");
        assert_eq!(merged.len(), 3);
    }

    // -----------------------------------------------------------------------
    // Test 7: concatenate_scenes with empty input → Err
    // -----------------------------------------------------------------------
    #[test]
    fn test_concatenate_scenes_empty_err() {
        let result = concatenate_scenes(&[]);
        assert!(matches!(result, Err(MergeError::EmptyScenes)));
    }

    // -----------------------------------------------------------------------
    // Test 8: merge_scenes with no scenes → Err
    // -----------------------------------------------------------------------
    #[test]
    fn test_merge_scenes_empty_err() {
        let config = SceneMergeConfig::default();
        let result = merge_scenes(&[], &config);
        assert!(matches!(result, Err(MergeError::EmptyScenes)));
    }

    // -----------------------------------------------------------------------
    // Test 9: merge_scenes min_opacity filters low-opacity Gaussians
    // -----------------------------------------------------------------------
    #[test]
    fn test_merge_scenes_min_opacity_filter() {
        let g_high = make_gaussian([0.0; 3], 0.8);
        let g_low = make_gaussian([1.0; 3], 0.001); // below default min_opacity 0.005
        let scene = make_scene(vec![g_high, g_low]);
        let config = SceneMergeConfig {
            min_opacity: 0.01,
            ..Default::default()
        };
        let merged = merge_scenes(&[scene], &config).expect("merge failed");
        assert_eq!(merged.len(), 1);
        assert!((merged.gaussians[0].opacity - 0.8).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // Test 10: merge_scenes max_scale filters large Gaussians
    // -----------------------------------------------------------------------
    #[test]
    fn test_merge_scenes_max_scale_filter() {
        let mut g_big = make_gaussian([0.0; 3], 0.8);
        g_big.log_scale = [2.0; 3]; // exp(2) ≈ 7.39 > 1.0
        let g_small = make_gaussian([1.0; 3], 0.8); // log_scale = [-2; 3], exp(-2) ≈ 0.135

        let scene = make_scene(vec![g_big, g_small]);
        let config = SceneMergeConfig {
            max_scale: 1.0,
            ..Default::default()
        };
        let merged = merge_scenes(&[scene], &config).expect("merge failed");
        assert_eq!(merged.len(), 1, "only small Gaussian should remain");
    }

    // -----------------------------------------------------------------------
    // Test 11: merge_scenes remove_duplicates removes co-located Gaussians
    // -----------------------------------------------------------------------
    #[test]
    fn test_merge_scenes_remove_duplicates() {
        // Two Gaussians at nearly the same position.
        let g1 = make_gaussian([0.0, 0.0, 0.0], 0.3);
        let g2 = make_gaussian([0.0001, 0.0001, 0.0001], 0.8); // higher opacity, same cell

        let scene = make_scene(vec![g1, g2]);
        let config = SceneMergeConfig {
            remove_duplicates: true,
            duplicate_threshold: 0.01, // cell size 0.01 → both in cell (0,0,0)
            ..Default::default()
        };
        let merged = merge_scenes(&[scene], &config).expect("merge failed");
        assert_eq!(merged.len(), 1, "duplicate should be removed");
        // The higher-opacity one should be kept.
        assert!((merged.gaussians[0].opacity - 0.8).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // Test 12: merge_scenes max_gaussians limits output size
    // -----------------------------------------------------------------------
    #[test]
    fn test_merge_scenes_max_gaussians() {
        let gaussians: Vec<GaussianEntry> = (0..10)
            .map(|i| make_gaussian([i as f32, 0.0, 0.0], 0.1 * (i as f32 + 1.0)))
            .collect();
        let scene = make_scene(gaussians);
        let config = SceneMergeConfig {
            max_gaussians: 3,
            ..Default::default()
        };
        let merged = merge_scenes(&[scene], &config).expect("merge failed");
        assert_eq!(merged.len(), 3);
        // Should keep the 3 highest-opacity ones.
        for g in &merged.gaussians {
            assert!(g.opacity >= 0.8, "expected high opacity, got {}", g.opacity);
        }
    }

    // -----------------------------------------------------------------------
    // Test 13: merge_scenes normalize_opacities adjusts mean
    // -----------------------------------------------------------------------
    #[test]
    fn test_merge_scenes_normalize_opacities() {
        let gaussians: Vec<GaussianEntry> = vec![
            make_gaussian([0.0; 3], 0.1),
            make_gaussian([1.0; 3], 0.2),
            make_gaussian([2.0; 3], 0.3),
        ];
        let scene = make_scene(gaussians);
        let config = SceneMergeConfig {
            normalize_opacities: true,
            target_opacity: 0.3,
            min_opacity: 0.0,
            ..Default::default()
        };
        let merged = merge_scenes(&[scene], &config).expect("merge failed");
        let mean: f32 = merged.gaussians.iter().map(|g| g.opacity).sum::<f32>()
            / (merged.gaussians.len() as f32);
        // Mean should be at target or clamped by [0,1].
        assert!(mean > 0.0);
        // All opacities should remain in [0, 1].
        for g in &merged.gaussians {
            assert!(g.opacity >= 0.0 && g.opacity <= 1.0);
        }
    }

    // -----------------------------------------------------------------------
    // Test 14: merge_scenes apply_transforms moves positions
    // -----------------------------------------------------------------------
    #[test]
    fn test_merge_scenes_apply_transforms() {
        let g = make_gaussian([0.0, 0.0, 0.0], 0.8);
        let translation = [
            1.0, 0.0, 0.0, 10.0, 0.0, 1.0, 0.0, 20.0, 0.0, 0.0, 1.0, 30.0, 0.0, 0.0, 0.0, 1.0f32,
        ];
        let scene = make_scene(vec![g]).with_transform(translation);
        let config = SceneMergeConfig {
            apply_transforms: true,
            ..Default::default()
        };
        let merged = merge_scenes(&[scene], &config).expect("merge failed");
        assert_eq!(merged.len(), 1);
        let pos = merged.gaussians[0].position;
        assert!((pos[0] - 10.0).abs() < 1e-5, "x={}", pos[0]);
        assert!((pos[1] - 20.0).abs() < 1e-5, "y={}", pos[1]);
        assert!((pos[2] - 30.0).abs() < 1e-5, "z={}", pos[2]);
    }

    // -----------------------------------------------------------------------
    // Test 15: find_duplicates — identical positions → one kept
    // -----------------------------------------------------------------------
    #[test]
    fn test_find_duplicates_identical() {
        let g1 = make_gaussian([0.0, 0.0, 0.0], 0.3);
        let g2 = make_gaussian([0.0, 0.0, 0.0], 0.8);
        let gaussians = vec![g1, g2];
        let mask = find_duplicates(&gaussians, 0.01);
        let kept: Vec<usize> = mask
            .iter()
            .enumerate()
            .filter(|(_, &k)| k)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(kept.len(), 1, "only one should be kept");
        // Winner is the one with higher opacity (index 1).
        assert_eq!(kept[0], 1);
    }

    // -----------------------------------------------------------------------
    // Test 16: find_duplicates — distant positions → all kept
    // -----------------------------------------------------------------------
    #[test]
    fn test_find_duplicates_distant() {
        let g1 = make_gaussian([0.0, 0.0, 0.0], 0.5);
        let g2 = make_gaussian([1.0, 1.0, 1.0], 0.5);
        let g3 = make_gaussian([-1.0, -1.0, -1.0], 0.5);
        let gaussians = vec![g1, g2, g3];
        let mask = find_duplicates(&gaussians, 0.01);
        let kept_count = mask.iter().filter(|&&k| k).count();
        assert_eq!(kept_count, 3, "all should be kept");
    }

    // -----------------------------------------------------------------------
    // Test 17: merge_at_boundary splits scene across axis
    // -----------------------------------------------------------------------
    #[test]
    fn test_merge_at_boundary_split() {
        // Scene A: two points, one clearly left of boundary, one in zone.
        let a1 = make_gaussian([-5.0, 0.0, 0.0], 1.0); // far left: should keep full opacity
        let a2 = make_gaussian([0.0, 0.0, 0.0], 1.0); // at boundary: blend = 1 → opacity = 0
                                                      // Scene B: one point well to the right.
        let b1 = make_gaussian([5.0, 0.0, 0.0], 1.0); // far right: blend = 1 → full opacity

        let scene_a = make_scene(vec![a1, a2]);
        let scene_b = make_scene(vec![b1]);

        let merged =
            merge_at_boundary(&scene_a, &scene_b, 0, 0.0, 1.0).expect("boundary merge failed");

        assert_eq!(merged.len(), 3);
        // a1 (pos=-5): smoothstep(-1, 0, -5) = 0 → opacity = 1*(1-0) = 1
        assert!(
            (merged.gaussians[0].opacity - 1.0).abs() < 1e-5,
            "a1 opacity={}",
            merged.gaussians[0].opacity
        );
        // a2 (pos=0): smoothstep(-1, 0, 0) = 1 → opacity = 1*(1-1) = 0
        assert!(
            (merged.gaussians[1].opacity - 0.0).abs() < 1e-5,
            "a2 opacity={}",
            merged.gaussians[1].opacity
        );
        // b1 (pos=5): smoothstep(0, 1, 5) = 1 → opacity = 1*1 = 1
        assert!(
            (merged.gaussians[2].opacity - 1.0).abs() < 1e-5,
            "b1 opacity={}",
            merged.gaussians[2].opacity
        );
    }

    // -----------------------------------------------------------------------
    // Test 18: merge_at_boundary — transition zone reduces opacity
    // -----------------------------------------------------------------------
    #[test]
    fn test_merge_at_boundary_transition_zone() {
        // A point exactly at the midpoint of scene A's transition zone.
        // Zone: [boundary - width, boundary] = [-1, 0], midpoint = -0.5
        // smoothstep(-1, 0, -0.5) → t = 0.5 → 3*(0.25) - 2*(0.125) = 0.75 - 0.25 = 0.5
        // opacity = 1.0 * (1 - 0.5) = 0.5
        let a_mid = make_gaussian([-0.5, 0.0, 0.0], 1.0);
        let scene_a = make_scene(vec![a_mid]);
        let scene_b = make_scene(vec![]);

        let merged =
            merge_at_boundary(&scene_a, &scene_b, 0, 0.0, 1.0).expect("boundary merge failed");
        assert_eq!(merged.len(), 1);
        let expected = 0.5f32;
        assert!(
            (merged.gaussians[0].opacity - expected).abs() < 1e-5,
            "expected {}, got {}",
            expected,
            merged.gaussians[0].opacity
        );
    }

    // -----------------------------------------------------------------------
    // Test 19: MergeStats::format_summary returns non-empty string
    // -----------------------------------------------------------------------
    #[test]
    fn test_merge_stats_format_summary() {
        let stats = MergeStats {
            input_scenes: 2,
            input_gaussians: vec![10, 20],
            total_input: 30,
            after_opacity_filter: 25,
            after_scale_filter: 22,
            duplicates_removed: 3,
            final_count: 19,
            final_bbox: Some(([0.0; 3], [1.0; 3])),
        };
        let summary = stats.format_summary();
        assert!(!summary.is_empty());
        assert!(summary.contains("2 input scenes"));
        assert!(summary.contains("19"));
    }

    // -----------------------------------------------------------------------
    // Test 20: merge_scenes_with_stats — stats counts are correct
    // -----------------------------------------------------------------------
    #[test]
    fn test_merge_scenes_with_stats_counts() {
        let gaussians: Vec<GaussianEntry> = vec![
            make_gaussian([0.0; 3], 0.8),
            make_gaussian([1.0; 3], 0.001), // below min_opacity 0.005
            make_gaussian([2.0; 3], 0.7),
        ];
        let scene = make_scene(gaussians);
        let config = SceneMergeConfig::default(); // min_opacity = 0.005
        let (merged, stats) =
            merge_scenes_with_stats(&[scene], &config).expect("merge_with_stats failed");

        assert_eq!(stats.input_scenes, 1);
        assert_eq!(stats.total_input, 3);
        assert_eq!(stats.after_opacity_filter, 2);
        assert_eq!(stats.after_scale_filter, 2);
        assert_eq!(stats.duplicates_removed, 0);
        assert_eq!(stats.final_count, 2);
        assert_eq!(merged.len(), 2);
    }

    // -----------------------------------------------------------------------
    // Test 21: SceneMergeConfig::validate — duplicate_threshold=0 with
    //          remove_duplicates → Err
    // -----------------------------------------------------------------------
    #[test]
    fn test_validate_duplicate_threshold_zero() {
        let config = SceneMergeConfig {
            remove_duplicates: true,
            duplicate_threshold: 0.0,
            ..Default::default()
        };
        let result = config.validate();
        assert!(matches!(result, Err(MergeError::InvalidConfig(_))));
    }

    // -----------------------------------------------------------------------
    // Test 22: GaussianEntry::max_scale — correct exp of max log_scale
    // -----------------------------------------------------------------------
    #[test]
    fn test_max_scale_correct() {
        let mut g = make_gaussian([0.0; 3], 0.5);
        g.log_scale = [-1.0, 2.0, 0.5]; // max log = 2.0 → exp(2.0) ≈ 7.389
        let expected = 2.0f32.exp();
        assert!((g.max_scale() - expected).abs() < 1e-5);
    }

    // -----------------------------------------------------------------------
    // Additional test: apply_transform_rotation with identity → same quat
    // -----------------------------------------------------------------------
    #[test]
    fn test_apply_transform_rotation_identity() {
        let identity = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0f32,
        ];
        let quat = [0.0, 0.0, 0.0, 1.0f32];
        let result = apply_transform_rotation(&identity, quat);
        // Result should be unit quat [0,0,0,1] (or close to it up to sign).
        let norm = (result[0] * result[0]
            + result[1] * result[1]
            + result[2] * result[2]
            + result[3] * result[3])
            .sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "not unit: norm={}", norm);
    }

    // -----------------------------------------------------------------------
    // Additional test: SceneMergeConfig::validate invalid min_opacity
    // -----------------------------------------------------------------------
    #[test]
    fn test_validate_min_opacity_out_of_range() {
        let config = SceneMergeConfig {
            min_opacity: 1.5, // > 1.0
            ..Default::default()
        };
        assert!(matches!(
            config.validate(),
            Err(MergeError::InvalidConfig(_))
        ));
    }

    // -----------------------------------------------------------------------
    // Additional test: SceneGaussians::centroid with empty scene → [0,0,0]
    // -----------------------------------------------------------------------
    #[test]
    fn test_centroid_empty() {
        let scene = make_scene(vec![]);
        assert_eq!(scene.centroid(), [0.0, 0.0, 0.0]);
    }

    // -----------------------------------------------------------------------
    // Additional test: concatenate preserves all positions
    // -----------------------------------------------------------------------
    #[test]
    fn test_concatenate_preserves_positions() {
        let s1 = make_scene(vec![make_gaussian([1.0, 2.0, 3.0], 0.5)]);
        let s2 = make_scene(vec![make_gaussian([4.0, 5.0, 6.0], 0.5)]);
        let merged = concatenate_scenes(&[s1, s2]).expect("concat failed");
        assert_eq!(merged.gaussians[0].position, [1.0, 2.0, 3.0]);
        assert_eq!(merged.gaussians[1].position, [4.0, 5.0, 6.0]);
    }
}
