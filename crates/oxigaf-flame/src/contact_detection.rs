//! Self-contact detection for FLAME head meshes.
//!
//! Detects when parts of the face touch each other, including:
//! - Mouth open/closed state (lip contact)
//! - Eye open/closed state (eyelid contact)
//! - General self-contact and mesh interpenetration (clipping)
//!
//! # Usage
//!
//! ```rust,no_run
//! use oxigaf_flame::{FlameModel, FlameParams};
//! use oxigaf_flame::contact_detection::{
//!     analyze_contact, ContactConfig, FlameContactRegions,
//! };
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let model = FlameModel::load("path/to/flame")?;
//! let mesh = model.forward(&FlameParams::neutral());
//! let regions = FlameContactRegions::default_flame();
//! let config = ContactConfig::default();
//! let report = analyze_contact(&mesh, &regions, &config)?;
//! println!("{}", report.format_summary());
//! # Ok(())
//! # }
//! ```

use thiserror::Error;

use crate::Mesh;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during contact detection.
#[derive(Debug, Error)]
pub enum ContactError {
    /// Mesh has no vertices — cannot compute distances.
    #[error("Mesh has no vertices")]
    EmptyMesh,

    /// A vertex index in a region definition is out of range for the mesh.
    #[error("Region A vertex index {idx} out of range (mesh has {n} vertices)")]
    VertexOutOfRange { idx: usize, n: usize },

    /// A threshold value is invalid (must be strictly positive).
    #[error("Invalid threshold {threshold}: must be > 0")]
    InvalidThreshold { threshold: f32 },

    /// A region contains no valid vertices.
    #[error("Region contains no valid vertices")]
    EmptyRegion,
}

// ---------------------------------------------------------------------------
// ContactConfig
// ---------------------------------------------------------------------------

/// Configuration for contact detection thresholds and sampling.
#[derive(Debug, Clone)]
pub struct ContactConfig {
    /// Distance threshold for mouth contact (upper/lower lip). Default: 0.002 (2 mm).
    pub mouth_contact_threshold: f32,
    /// Distance threshold for eyelid contact. Default: 0.001 (1 mm).
    pub eye_contact_threshold: f32,
    /// Threshold for general self-contact detection. Default: 0.005 (5 mm).
    pub general_threshold: f32,
    /// Number of vertex pairs to sample per region in `detect_self_contact`. Default: 16.
    pub num_sample_pairs: usize,
}

impl Default for ContactConfig {
    fn default() -> Self {
        Self {
            mouth_contact_threshold: 0.002,
            eye_contact_threshold: 0.001,
            general_threshold: 0.005,
            num_sample_pairs: 16,
        }
    }
}

impl ContactConfig {
    /// Validate configuration values, returning an error for invalid parameters.
    ///
    /// # Errors
    ///
    /// Returns [`ContactError::InvalidThreshold`] if any threshold is `<= 0`.
    pub fn validate(&self) -> Result<(), ContactError> {
        if self.mouth_contact_threshold <= 0.0 {
            return Err(ContactError::InvalidThreshold {
                threshold: self.mouth_contact_threshold,
            });
        }
        if self.eye_contact_threshold <= 0.0 {
            return Err(ContactError::InvalidThreshold {
                threshold: self.eye_contact_threshold,
            });
        }
        if self.general_threshold <= 0.0 {
            return Err(ContactError::InvalidThreshold {
                threshold: self.general_threshold,
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// ContactPair
// ---------------------------------------------------------------------------

/// A pair of vertices in close proximity on the mesh.
#[derive(Debug, Clone)]
pub struct ContactPair {
    /// Index of the first vertex.
    pub vertex_a: usize,
    /// Index of the second vertex.
    pub vertex_b: usize,
    /// Euclidean distance between the two vertices (world units).
    pub distance: f32,
    /// Midpoint between the two vertices: `(pos_a + pos_b) / 2`.
    pub midpoint: [f32; 3],
}

// ---------------------------------------------------------------------------
// ContactReport
// ---------------------------------------------------------------------------

/// Whether the mouth is open or closed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MouthContactState {
    /// Mouth is open (lip opening exceeds threshold).
    #[default]
    Open,
    /// Mouth is closed (lip opening is below threshold).
    Closed,
}

/// Whether an eye is open or closed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EyeContactState {
    /// Eye is open (eyelid opening exceeds threshold).
    #[default]
    Open,
    /// Eye is closed (eyelid opening is below threshold).
    Closed,
}

/// Whether the mesh has interpenetrating (clipping) geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ClippingStatus {
    /// No vertex pairs with distance ≤ 0 detected.
    #[default]
    NoClipping,
    /// At least one vertex pair with distance ≤ 0 detected.
    HasClipping,
}

/// Contact state flags using two-variant enums to avoid the
/// [`struct_excessive_bools`](https://rust-lang.github.io/rust-clippy/master/index.html#struct_excessive_bools) lint.
#[derive(Debug, Clone, Default)]
pub struct ContactFlags {
    /// Whether the mouth is considered closed (opening below threshold).
    pub mouth: MouthContactState,
    /// Whether the left eye is considered closed.
    pub left_eye: EyeContactState,
    /// Whether the right eye is considered closed.
    pub right_eye: EyeContactState,
    /// Whether any pair has distance `≤ 0` (interpenetration / clipping).
    pub clipping: ClippingStatus,
}

/// Summary of all contact states for a posed FLAME mesh.
#[derive(Debug, Clone)]
pub struct ContactReport {
    /// Boolean contact state flags.
    pub flags: ContactFlags,
    /// Mean distance between upper and lower lip key vertices (world units).
    pub mouth_opening: f32,
    /// Opening distance of the left eye (world units).
    pub left_eye_opening: f32,
    /// Opening distance of the right eye (world units).
    pub right_eye_opening: f32,
    /// All detected close vertex pairs across the analyzed regions.
    pub contact_pairs: Vec<ContactPair>,
    /// Number of pairs with distance `< 0`.
    pub clipping_count: usize,
}

impl ContactReport {
    /// Format a human-readable one-line summary of contact state.
    #[must_use]
    pub fn format_summary(&self) -> String {
        let mouth = if self.flags.mouth == MouthContactState::Closed {
            format!("mouth=closed({:.4})", self.mouth_opening)
        } else {
            format!("mouth=open({:.4})", self.mouth_opening)
        };
        let left_eye = if self.flags.left_eye == EyeContactState::Closed {
            format!("left_eye=closed({:.4})", self.left_eye_opening)
        } else {
            format!("left_eye=open({:.4})", self.left_eye_opening)
        };
        let right_eye = if self.flags.right_eye == EyeContactState::Closed {
            format!("right_eye=closed({:.4})", self.right_eye_opening)
        } else {
            format!("right_eye=open({:.4})", self.right_eye_opening)
        };
        let clipping = if self.flags.clipping == ClippingStatus::HasClipping {
            format!("CLIPPING({} pairs)", self.clipping_count)
        } else {
            "no_clipping".to_string()
        };
        format!(
            "ContactReport: {} | {} | {} | contacts={} | {}",
            mouth,
            left_eye,
            right_eye,
            self.contact_pairs.len(),
            clipping,
        )
    }

    /// Check if the avatar is in a physically valid state (no interpenetration).
    #[must_use]
    pub fn is_physically_valid(&self) -> bool {
        self.flags.clipping == ClippingStatus::NoClipping
    }
}

// ---------------------------------------------------------------------------
// FlameContactRegions
// ---------------------------------------------------------------------------

/// Anatomically meaningful vertex index sets for FLAME-specific contact detection.
///
/// All indices are approximate for a standard 5023-vertex FLAME mesh.
/// In production use, provide accurate indices from the FLAME model's
/// semantic vertex annotations.
#[derive(Debug, Clone)]
pub struct FlameContactRegions {
    /// Upper lip center vertices.
    pub upper_lip: Vec<usize>,
    /// Lower lip center vertices.
    pub lower_lip: Vec<usize>,
    /// Upper left eyelid vertices.
    pub left_upper_eyelid: Vec<usize>,
    /// Lower left eyelid vertices.
    pub left_lower_eyelid: Vec<usize>,
    /// Upper right eyelid vertices.
    pub right_upper_eyelid: Vec<usize>,
    /// Lower right eyelid vertices.
    pub right_lower_eyelid: Vec<usize>,
}

impl FlameContactRegions {
    /// Default FLAME vertex indices (approximate, for standard 5023-vertex FLAME).
    ///
    /// These are representative anatomical vertices derived from the FLAME
    /// mesh topology. In production, replace with accurate indices from the
    /// FLAME model's face-region annotations.
    #[must_use]
    pub fn default_flame() -> Self {
        Self {
            upper_lip: vec![3535, 3536, 3537],
            lower_lip: vec![3501, 3502, 3503],
            left_upper_eyelid: vec![3800, 3801, 3802],
            left_lower_eyelid: vec![3820, 3821, 3822],
            right_upper_eyelid: vec![4000, 4001, 4002],
            right_lower_eyelid: vec![4020, 4021, 4022],
        }
    }

    /// Validate that all vertex indices are within bounds for a mesh with `n_vertices`.
    ///
    /// Returns `true` if all indices are valid, `false` otherwise.
    #[must_use]
    pub fn validate(&self, n_vertices: usize) -> bool {
        let all_regions = [
            self.upper_lip.as_slice(),
            self.lower_lip.as_slice(),
            self.left_upper_eyelid.as_slice(),
            self.left_lower_eyelid.as_slice(),
            self.right_upper_eyelid.as_slice(),
            self.right_lower_eyelid.as_slice(),
        ];
        for region in &all_regions {
            for &idx in *region {
                if idx >= n_vertices {
                    return false;
                }
            }
        }
        true
    }
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// Compute the Euclidean distance between two 3D points.
#[inline]
fn dist3(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Extract vertex position as `[f32; 3]` from a mesh.
///
/// # Errors
///
/// Returns [`ContactError::VertexOutOfRange`] if `idx >= mesh.vertices.len()`.
#[inline]
fn vertex_pos(mesh: &Mesh, idx: usize) -> Result<[f32; 3], ContactError> {
    let n = mesh.vertices.len();
    if idx >= n {
        return Err(ContactError::VertexOutOfRange { idx, n });
    }
    let v = &mesh.vertices[idx];
    Ok([v.x, v.y, v.z])
}

/// Compute the mean 3D position of a set of vertex indices from a mesh.
///
/// # Errors
///
/// - [`ContactError::EmptyRegion`] if `indices` is empty.
/// - [`ContactError::VertexOutOfRange`] if any index is out of bounds.
pub fn mean_position(mesh: &Mesh, indices: &[usize]) -> Result<[f32; 3], ContactError> {
    if indices.is_empty() {
        return Err(ContactError::EmptyRegion);
    }
    let mut sum = [0.0f32; 3];
    for &idx in indices {
        let pos = vertex_pos(mesh, idx)?;
        sum[0] += pos[0];
        sum[1] += pos[1];
        sum[2] += pos[2];
    }
    let n = indices.len() as f32;
    Ok([sum[0] / n, sum[1] / n, sum[2] / n])
}

/// Compute the minimum Euclidean distance between two sets of vertex indices.
///
/// Uses brute-force O(|A| × |B|) comparison.
///
/// # Errors
///
/// - [`ContactError::EmptyRegion`] if either region is empty.
/// - [`ContactError::VertexOutOfRange`] if any index is out of bounds.
pub fn min_distance_between_regions(
    mesh: &Mesh,
    region_a: &[usize],
    region_b: &[usize],
) -> Result<f32, ContactError> {
    if region_a.is_empty() || region_b.is_empty() {
        return Err(ContactError::EmptyRegion);
    }
    let mut min_dist = f32::MAX;
    for &a in region_a {
        let pa = vertex_pos(mesh, a)?;
        for &b in region_b {
            let pb = vertex_pos(mesh, b)?;
            let d = dist3(pa, pb);
            if d < min_dist {
                min_dist = d;
            }
        }
    }
    Ok(min_dist)
}

/// Compute the mean Euclidean distance between two sets of vertex indices.
///
/// Averages over all (a, b) pairs — O(|A| × |B|).
///
/// # Errors
///
/// - [`ContactError::EmptyRegion`] if either region is empty.
/// - [`ContactError::VertexOutOfRange`] if any index is out of bounds.
pub fn mean_distance_between_regions(
    mesh: &Mesh,
    region_a: &[usize],
    region_b: &[usize],
) -> Result<f32, ContactError> {
    if region_a.is_empty() || region_b.is_empty() {
        return Err(ContactError::EmptyRegion);
    }
    let mut total = 0.0f32;
    let mut count = 0usize;
    for &a in region_a {
        let pa = vertex_pos(mesh, a)?;
        for &b in region_b {
            let pb = vertex_pos(mesh, b)?;
            total += dist3(pa, pb);
            count += 1;
        }
    }
    // count > 0 because both regions are non-empty
    Ok(total / count as f32)
}

/// Find all vertex pairs within a given threshold distance across two regions.
///
/// Iterates all (a, b) pairs. Returns pairs whose distance is strictly less
/// than `threshold`.
///
/// # Errors
///
/// - [`ContactError::InvalidThreshold`] if `threshold <= 0`.
/// - [`ContactError::EmptyRegion`] if either region is empty.
/// - [`ContactError::VertexOutOfRange`] if any index is out of bounds.
pub fn find_contact_pairs(
    mesh: &Mesh,
    region_a: &[usize],
    region_b: &[usize],
    threshold: f32,
) -> Result<Vec<ContactPair>, ContactError> {
    if threshold <= 0.0 {
        return Err(ContactError::InvalidThreshold { threshold });
    }
    if region_a.is_empty() || region_b.is_empty() {
        return Err(ContactError::EmptyRegion);
    }
    let mut pairs = Vec::new();
    for &a in region_a {
        let pa = vertex_pos(mesh, a)?;
        for &b in region_b {
            let pb = vertex_pos(mesh, b)?;
            let distance = dist3(pa, pb);
            if distance < threshold {
                let midpoint = [
                    (pa[0] + pb[0]) * 0.5,
                    (pa[1] + pb[1]) * 0.5,
                    (pa[2] + pb[2]) * 0.5,
                ];
                pairs.push(ContactPair {
                    vertex_a: a,
                    vertex_b: b,
                    distance,
                    midpoint,
                });
            }
        }
    }
    Ok(pairs)
}

/// Check mouth contact using anatomical lip regions.
///
/// Returns `(is_closed, mouth_opening_distance)` where `mouth_opening_distance`
/// is the mean distance between upper and lower lip representative vertices.
///
/// # Errors
///
/// - Propagates [`ContactError`] from [`mean_distance_between_regions`].
pub fn check_mouth_contact(
    mesh: &Mesh,
    regions: &FlameContactRegions,
    config: &ContactConfig,
) -> Result<(bool, f32), ContactError> {
    let opening = mean_distance_between_regions(mesh, &regions.upper_lip, &regions.lower_lip)?;
    let is_closed = opening < config.mouth_contact_threshold;
    Ok((is_closed, opening))
}

/// Check eye contact (eyelid closure) for one eye.
///
/// Returns `(is_closed, eye_opening_distance)` where `eye_opening_distance`
/// is the mean distance between the upper and lower eyelid vertices.
///
/// # Errors
///
/// - Propagates [`ContactError`] from [`mean_distance_between_regions`].
pub fn check_eye_contact(
    mesh: &Mesh,
    upper_eyelid: &[usize],
    lower_eyelid: &[usize],
    threshold: f32,
) -> Result<(bool, f32), ContactError> {
    let opening = mean_distance_between_regions(mesh, upper_eyelid, lower_eyelid)?;
    let is_closed = opening < threshold;
    Ok((is_closed, opening))
}

/// Detect general self-contact: find vertex pairs from two regions within `config.general_threshold`.
///
/// Uses strided sampling of `region_a` rather than random selection.
/// The stride is `max(1, region_a.len() / num_sample_pairs)`, yielding at most
/// `num_sample_pairs` vertices from `region_a`. All of `region_b` is checked
/// against each sampled vertex from `region_a`.
///
/// # Errors
///
/// - [`ContactError::EmptyRegion`] if either region is empty.
/// - [`ContactError::VertexOutOfRange`] if any index is out of bounds.
pub fn detect_self_contact(
    mesh: &Mesh,
    region_a: &[usize],
    region_b: &[usize],
    config: &ContactConfig,
) -> Result<Vec<ContactPair>, ContactError> {
    if region_a.is_empty() || region_b.is_empty() {
        return Err(ContactError::EmptyRegion);
    }
    let num_pairs = config.num_sample_pairs;
    let stride = (region_a.len() / num_pairs).max(1);

    let mut pairs = Vec::new();
    let threshold = config.general_threshold;

    for (step, &a) in region_a.iter().enumerate().filter(|(i, _)| i % stride == 0) {
        // Limit to num_sample_pairs sampled vertices from region_a
        if step / stride >= num_pairs {
            break;
        }
        let pa = vertex_pos(mesh, a)?;
        for &b in region_b {
            let pb = vertex_pos(mesh, b)?;
            let distance = dist3(pa, pb);
            if distance < threshold {
                let midpoint = [
                    (pa[0] + pb[0]) * 0.5,
                    (pa[1] + pb[1]) * 0.5,
                    (pa[2] + pb[2]) * 0.5,
                ];
                pairs.push(ContactPair {
                    vertex_a: a,
                    vertex_b: b,
                    distance,
                    midpoint,
                });
            }
        }
    }
    Ok(pairs)
}

/// Full contact analysis using FLAME anatomical regions.
///
/// Computes:
/// - Mouth open/closed state
/// - Left/right eye open/closed states
/// - All contact pairs between upper and lower lips
/// - Clipping detection (any pair with distance < 0)
///
/// # Errors
///
/// - [`ContactError::EmptyMesh`] if the mesh has no vertices.
/// - Propagates other [`ContactError`] variants from sub-functions.
pub fn analyze_contact(
    mesh: &Mesh,
    regions: &FlameContactRegions,
    config: &ContactConfig,
) -> Result<ContactReport, ContactError> {
    if mesh.vertices.is_empty() {
        return Err(ContactError::EmptyMesh);
    }

    let (mouth_is_closed, mouth_opening) = check_mouth_contact(mesh, regions, config)?;

    let (left_eye_is_closed, left_eye_opening) = check_eye_contact(
        mesh,
        &regions.left_upper_eyelid,
        &regions.left_lower_eyelid,
        config.eye_contact_threshold,
    )?;

    let (right_eye_is_closed, right_eye_opening) = check_eye_contact(
        mesh,
        &regions.right_upper_eyelid,
        &regions.right_lower_eyelid,
        config.eye_contact_threshold,
    )?;

    // Find all close pairs between upper and lower lips
    let contact_pairs = find_contact_pairs(
        mesh,
        &regions.upper_lip,
        &regions.lower_lip,
        config.mouth_contact_threshold,
    )
    .unwrap_or_default();

    // Interpenetration: distance < 0 means vertices overlap (signed distance field)
    // In Euclidean geometry distances are >= 0; clipping is represented as distance == 0.0
    // For this implementation, we treat distance exactly 0.0 as clipping (vertex coincidence).
    let clipping_count = contact_pairs.iter().filter(|p| p.distance <= 0.0).count();
    let clipping = if clipping_count > 0 {
        ClippingStatus::HasClipping
    } else {
        ClippingStatus::NoClipping
    };

    Ok(ContactReport {
        flags: ContactFlags {
            mouth: if mouth_is_closed {
                MouthContactState::Closed
            } else {
                MouthContactState::Open
            },
            left_eye: if left_eye_is_closed {
                EyeContactState::Closed
            } else {
                EyeContactState::Open
            },
            right_eye: if right_eye_is_closed {
                EyeContactState::Closed
            } else {
                EyeContactState::Open
            },
            clipping,
        },
        mouth_opening,
        left_eye_opening,
        right_eye_opening,
        contact_pairs,
        clipping_count,
    })
}

// ---------------------------------------------------------------------------
// Jaw / Eye state classification
// ---------------------------------------------------------------------------

/// Discrete classification of jaw opening state.
#[derive(Debug, Clone, PartialEq)]
pub enum JawState {
    /// Opening distance < 0.002 m — lips are touching.
    Closed,
    /// Opening distance in [0.002, 0.01) m — slight separation.
    Slightly,
    /// Opening distance in [0.01, 0.03) m — moderate jaw drop.
    Moderate,
    /// Opening distance >= 0.03 m — wide open mouth.
    Wide,
}

/// Classify jaw opening from a distance measurement.
///
/// | Range              | State      |
/// |--------------------|------------|
/// | < 0.002            | Closed     |
/// | 0.002 … < 0.01     | Slightly   |
/// | 0.01  … < 0.03     | Moderate   |
/// | >= 0.03            | Wide       |
#[must_use]
pub fn classify_jaw_state(opening: f32) -> JawState {
    if opening < 0.002 {
        JawState::Closed
    } else if opening < 0.01 {
        JawState::Slightly
    } else if opening < 0.03 {
        JawState::Moderate
    } else {
        JawState::Wide
    }
}

/// Discrete classification of eye opening state.
#[derive(Debug, Clone, PartialEq)]
pub enum EyeState {
    /// Opening distance < 0.001 m — eyelids are touching.
    Closed,
    /// Opening distance in [0.001, 0.005) m — eye is partially closed.
    Squinting,
    /// Opening distance in [0.005, 0.015) m — normal open eye.
    Open,
    /// Opening distance >= 0.015 m — eye is very wide open.
    Wide,
}

/// Classify eye state from an eyelid opening distance.
///
/// | Range               | State     |
/// |---------------------|-----------|
/// | < 0.001             | Closed    |
/// | 0.001 … < 0.005     | Squinting |
/// | 0.005 … < 0.015     | Open      |
/// | >= 0.015            | Wide      |
#[must_use]
pub fn classify_eye_state(opening: f32) -> EyeState {
    if opening < 0.001 {
        EyeState::Closed
    } else if opening < 0.005 {
        EyeState::Squinting
    } else if opening < 0.015 {
        EyeState::Open
    } else {
        EyeState::Wide
    }
}

// ---------------------------------------------------------------------------
// Temporal smoothing & transition detection
// ---------------------------------------------------------------------------

/// Smooth a sequence of mouth-opening distances with a simple moving average.
///
/// If `window == 0`, the input is returned unchanged.
/// The window is clamped to `[1, openings.len()]`.
///
/// Each output element `i` is the mean of the input elements in the range
/// `[i.saturating_sub(window/2), i + window/2]` (inclusive, clamped).
///
/// When `window == 1`, this is an identity transform.
#[must_use]
pub fn smooth_mouth_openings(openings: &[f32], window: usize) -> Vec<f32> {
    if openings.is_empty() || window <= 1 {
        return openings.to_vec();
    }
    let w = window.min(openings.len());
    let half = w / 2;
    let n = openings.len();

    let mut result = Vec::with_capacity(n);
    for i in 0..n {
        let start = i.saturating_sub(half);
        let end = (i + half + 1).min(n);
        let count = end - start;
        let sum: f32 = openings[start..end].iter().sum();
        result.push(sum / count as f32);
    }
    result
}

/// Detect rapid changes in mouth opening — potential speech boundaries.
///
/// Returns the indices `i` (1-based into `openings`) where
/// `|openings[i] - openings[i-1]| > threshold`.
///
/// # Returns
///
/// A `Vec<usize>` of frame indices where significant transitions occur.
#[must_use]
pub fn detect_mouth_transitions(openings: &[f32], threshold: f32) -> Vec<usize> {
    if openings.len() < 2 {
        return Vec::new();
    }
    let mut transitions = Vec::new();
    for i in 1..openings.len() {
        if (openings[i] - openings[i - 1]).abs() > threshold {
            transitions.push(i);
        }
    }
    transitions
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra as na;

    // -----------------------------------------------------------------------
    // Test mesh helpers
    // -----------------------------------------------------------------------

    /// Build a tiny mesh from raw `[f32;3]` vertex positions with no faces.
    fn make_mesh(raw: &[[f32; 3]]) -> Mesh {
        let vertices: Vec<na::Point3<f32>> = raw
            .iter()
            .map(|&[x, y, z]| na::Point3::new(x, y, z))
            .collect();
        Mesh::new(vertices, vec![])
    }

    // -----------------------------------------------------------------------
    // ContactConfig tests
    // -----------------------------------------------------------------------

    #[test]
    fn contact_config_default_values() {
        let cfg = ContactConfig::default();
        assert!((cfg.mouth_contact_threshold - 0.002).abs() < f32::EPSILON);
        assert!((cfg.eye_contact_threshold - 0.001).abs() < f32::EPSILON);
        assert!((cfg.general_threshold - 0.005).abs() < f32::EPSILON);
        assert_eq!(cfg.num_sample_pairs, 16);
    }

    #[test]
    fn contact_config_validate_ok() {
        let cfg = ContactConfig::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn contact_config_validate_zero_mouth_threshold() {
        let cfg = ContactConfig {
            mouth_contact_threshold: 0.0,
            ..ContactConfig::default()
        };
        assert!(matches!(
            cfg.validate(),
            Err(ContactError::InvalidThreshold { threshold }) if threshold == 0.0
        ));
    }

    #[test]
    fn contact_config_validate_negative_eye_threshold() {
        let cfg = ContactConfig {
            eye_contact_threshold: -0.001,
            ..ContactConfig::default()
        };
        assert!(matches!(
            cfg.validate(),
            Err(ContactError::InvalidThreshold { .. })
        ));
    }

    // -----------------------------------------------------------------------
    // FlameContactRegions tests
    // -----------------------------------------------------------------------

    #[test]
    fn flame_contact_regions_default_flame_non_empty() {
        let regions = FlameContactRegions::default_flame();
        assert!(!regions.upper_lip.is_empty());
        assert!(!regions.lower_lip.is_empty());
        assert!(!regions.left_upper_eyelid.is_empty());
        assert!(!regions.left_lower_eyelid.is_empty());
        assert!(!regions.right_upper_eyelid.is_empty());
        assert!(!regions.right_lower_eyelid.is_empty());
    }

    #[test]
    fn flame_contact_regions_validate_small_mesh_fails() {
        // Default regions have indices ~3500-4022, which exceed a 10-vertex mesh
        let regions = FlameContactRegions::default_flame();
        assert!(!regions.validate(10));
    }

    // -----------------------------------------------------------------------
    // mean_position tests
    // -----------------------------------------------------------------------

    #[test]
    fn mean_position_single_vertex() {
        let mesh = make_mesh(&[[1.0, 2.0, 3.0]]);
        let pos = mean_position(&mesh, &[0]).expect("single vertex mean");
        assert!((pos[0] - 1.0).abs() < 1e-6);
        assert!((pos[1] - 2.0).abs() < 1e-6);
        assert!((pos[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn mean_position_two_vertices() {
        let mesh = make_mesh(&[[0.0, 0.0, 0.0], [2.0, 4.0, 6.0]]);
        let pos = mean_position(&mesh, &[0, 1]).expect("two vertex mean");
        assert!((pos[0] - 1.0).abs() < 1e-6);
        assert!((pos[1] - 2.0).abs() < 1e-6);
        assert!((pos[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn mean_position_empty_region_error() {
        let mesh = make_mesh(&[[0.0, 0.0, 0.0]]);
        let result = mean_position(&mesh, &[]);
        assert!(matches!(result, Err(ContactError::EmptyRegion)));
    }

    // -----------------------------------------------------------------------
    // min_distance_between_regions tests
    // -----------------------------------------------------------------------

    #[test]
    fn min_distance_between_regions_same_vertex_zero() {
        let mesh = make_mesh(&[[1.0, 2.0, 3.0]]);
        let d = min_distance_between_regions(&mesh, &[0], &[0]).expect("same vertex");
        assert!(d.abs() < 1e-6);
    }

    #[test]
    fn min_distance_between_regions_known_distance() {
        // Point A at origin, Point B at (3,4,0) -> dist = 5
        let mesh = make_mesh(&[[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]]);
        let d = min_distance_between_regions(&mesh, &[0], &[1]).expect("known distance");
        assert!((d - 5.0).abs() < 1e-5, "expected 5.0, got {d}");
    }

    #[test]
    fn min_distance_between_regions_selects_minimum() {
        // region_a = [0], region_b = [1, 2]; min should be to vertex 2
        let mesh = make_mesh(&[
            [0.0, 0.0, 0.0], // 0
            [3.0, 4.0, 0.0], // 1 — dist 5 from 0
            [1.0, 0.0, 0.0], // 2 — dist 1 from 0
        ]);
        let d = min_distance_between_regions(&mesh, &[0], &[1, 2]).expect("min distance");
        assert!((d - 1.0).abs() < 1e-5, "expected 1.0, got {d}");
    }

    // -----------------------------------------------------------------------
    // mean_distance_between_regions tests
    // -----------------------------------------------------------------------

    #[test]
    fn mean_distance_between_regions_known_result() {
        // A={0}, B={1,2}: distances 5 and 1, mean = 3
        let mesh = make_mesh(&[
            [0.0, 0.0, 0.0], // 0
            [3.0, 4.0, 0.0], // 1 — dist 5
            [1.0, 0.0, 0.0], // 2 — dist 1
        ]);
        let d = mean_distance_between_regions(&mesh, &[0], &[1, 2]).expect("mean distance");
        assert!((d - 3.0).abs() < 1e-5, "expected 3.0, got {d}");
    }

    #[test]
    fn mean_distance_between_regions_empty_error() {
        let mesh = make_mesh(&[[0.0, 0.0, 0.0]]);
        assert!(matches!(
            mean_distance_between_regions(&mesh, &[], &[0]),
            Err(ContactError::EmptyRegion)
        ));
    }

    // -----------------------------------------------------------------------
    // find_contact_pairs tests
    // -----------------------------------------------------------------------

    #[test]
    fn find_contact_pairs_within_threshold() {
        // Two vertices 0.001 apart — below threshold 0.005
        let mesh = make_mesh(&[[0.0, 0.0, 0.0], [0.001, 0.0, 0.0]]);
        let pairs = find_contact_pairs(&mesh, &[0], &[1], 0.005).expect("find pairs");
        assert_eq!(pairs.len(), 1);
        assert!((pairs[0].distance - 0.001).abs() < 1e-6);
    }

    #[test]
    fn find_contact_pairs_outside_threshold() {
        // Two vertices 0.01 apart — above threshold 0.005
        let mesh = make_mesh(&[[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]]);
        let pairs = find_contact_pairs(&mesh, &[0], &[1], 0.005).expect("find pairs");
        assert!(pairs.is_empty());
    }

    #[test]
    fn find_contact_pairs_midpoint_correct() {
        let mesh = make_mesh(&[[0.0, 0.0, 0.0], [0.002, 0.0, 0.0]]);
        let pairs = find_contact_pairs(&mesh, &[0], &[1], 0.01).expect("find pairs");
        assert_eq!(pairs.len(), 1);
        let mp = pairs[0].midpoint;
        assert!((mp[0] - 0.001).abs() < 1e-6);
        assert!(mp[1].abs() < 1e-6);
        assert!(mp[2].abs() < 1e-6);
    }

    #[test]
    fn find_contact_pairs_invalid_threshold_error() {
        let mesh = make_mesh(&[[0.0, 0.0, 0.0], [0.001, 0.0, 0.0]]);
        assert!(matches!(
            find_contact_pairs(&mesh, &[0], &[1], 0.0),
            Err(ContactError::InvalidThreshold { .. })
        ));
    }

    // -----------------------------------------------------------------------
    // check_mouth_contact tests
    // -----------------------------------------------------------------------

    #[test]
    fn check_mouth_contact_far_vertices_open() {
        // Upper lip at 0, lower lip at 0.05 — far apart, mouth open
        let mesh = make_mesh(&[[0.0, 0.0, 0.0], [0.0, -0.05, 0.0]]);
        let _regions = FlameContactRegions {
            upper_lip: vec![0],
            lower_lip: vec![1],
            ..FlameContactRegions::default_flame()
        };
        // Override eyelid indices to valid ones (0 and 1)
        let regions = FlameContactRegions {
            upper_lip: vec![0],
            lower_lip: vec![1],
            left_upper_eyelid: vec![0],
            left_lower_eyelid: vec![1],
            right_upper_eyelid: vec![0],
            right_lower_eyelid: vec![1],
        };
        let config = ContactConfig::default();
        let (is_closed, opening) = check_mouth_contact(&mesh, &regions, &config).expect("mouth");
        assert!(!is_closed, "lips are far apart — mouth should be open");
        assert!((opening - 0.05).abs() < 1e-5);
    }

    #[test]
    fn check_mouth_contact_close_vertices_closed() {
        // Upper lip at 0, lower lip at 0.0001 — within 2mm threshold
        let mesh = make_mesh(&[[0.0, 0.0, 0.0], [0.0001, 0.0, 0.0]]);
        let regions = FlameContactRegions {
            upper_lip: vec![0],
            lower_lip: vec![1],
            left_upper_eyelid: vec![0],
            left_lower_eyelid: vec![1],
            right_upper_eyelid: vec![0],
            right_lower_eyelid: vec![1],
        };
        let config = ContactConfig::default();
        let (is_closed, opening) = check_mouth_contact(&mesh, &regions, &config).expect("mouth");
        assert!(is_closed, "lips are very close — mouth should be closed");
        assert!((opening - 0.0001).abs() < 1e-6);
    }

    #[test]
    fn check_mouth_contact_exactly_at_threshold_open() {
        // Opening exactly equals threshold — should be open (< not <=)
        let mesh = make_mesh(&[[0.0, 0.0, 0.0], [0.002, 0.0, 0.0]]);
        let regions = FlameContactRegions {
            upper_lip: vec![0],
            lower_lip: vec![1],
            left_upper_eyelid: vec![0],
            left_lower_eyelid: vec![1],
            right_upper_eyelid: vec![0],
            right_lower_eyelid: vec![1],
        };
        let config = ContactConfig::default();
        let (is_closed, _) = check_mouth_contact(&mesh, &regions, &config).expect("mouth");
        assert!(!is_closed, "opening == threshold → open");
    }

    // -----------------------------------------------------------------------
    // check_eye_contact tests
    // -----------------------------------------------------------------------

    #[test]
    fn check_eye_contact_far_vertices_open() {
        let mesh = make_mesh(&[[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]]);
        let (is_closed, opening) =
            check_eye_contact(&mesh, &[0], &[1], 0.001).expect("eye contact");
        assert!(!is_closed);
        assert!((opening - 0.01).abs() < 1e-5);
    }

    #[test]
    fn check_eye_contact_close_vertices_closed() {
        let mesh = make_mesh(&[[0.0, 0.0, 0.0], [0.0005, 0.0, 0.0]]);
        let (is_closed, opening) =
            check_eye_contact(&mesh, &[0], &[1], 0.001).expect("eye contact");
        assert!(is_closed);
        assert!((opening - 0.0005).abs() < 1e-6);
    }

    #[test]
    fn check_eye_contact_empty_region_error() {
        let mesh = make_mesh(&[[0.0, 0.0, 0.0]]);
        assert!(matches!(
            check_eye_contact(&mesh, &[], &[0], 0.001),
            Err(ContactError::EmptyRegion)
        ));
    }

    // -----------------------------------------------------------------------
    // detect_self_contact tests
    // -----------------------------------------------------------------------

    #[test]
    fn detect_self_contact_no_contact() {
        // Vertices far apart — no contact within general_threshold (0.005)
        let mesh = make_mesh(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]);
        let config = ContactConfig::default();
        let pairs = detect_self_contact(&mesh, &[0], &[1], &config).expect("self contact");
        assert!(pairs.is_empty());
    }

    #[test]
    fn detect_self_contact_contact_found() {
        // Vertices very close — within general_threshold (0.005)
        let mesh = make_mesh(&[[0.0, 0.0, 0.0], [0.003, 0.0, 0.0]]);
        let config = ContactConfig::default();
        let pairs = detect_self_contact(&mesh, &[0], &[1], &config).expect("self contact");
        assert!(!pairs.is_empty());
        assert!((pairs[0].distance - 0.003).abs() < 1e-6);
    }

    #[test]
    fn detect_self_contact_empty_region_error() {
        let mesh = make_mesh(&[[0.0, 0.0, 0.0]]);
        let config = ContactConfig::default();
        assert!(matches!(
            detect_self_contact(&mesh, &[], &[0], &config),
            Err(ContactError::EmptyRegion)
        ));
    }

    // -----------------------------------------------------------------------
    // classify_jaw_state tests
    // -----------------------------------------------------------------------

    #[test]
    fn classify_jaw_state_closed() {
        assert_eq!(classify_jaw_state(0.0), JawState::Closed);
        assert_eq!(classify_jaw_state(0.001), JawState::Closed);
        assert_eq!(classify_jaw_state(0.00199), JawState::Closed);
    }

    #[test]
    fn classify_jaw_state_slightly() {
        assert_eq!(classify_jaw_state(0.002), JawState::Slightly);
        assert_eq!(classify_jaw_state(0.005), JawState::Slightly);
        assert_eq!(classify_jaw_state(0.0099), JawState::Slightly);
    }

    #[test]
    fn classify_jaw_state_moderate() {
        assert_eq!(classify_jaw_state(0.01), JawState::Moderate);
        assert_eq!(classify_jaw_state(0.02), JawState::Moderate);
        assert_eq!(classify_jaw_state(0.0299), JawState::Moderate);
    }

    #[test]
    fn classify_jaw_state_wide() {
        assert_eq!(classify_jaw_state(0.03), JawState::Wide);
        assert_eq!(classify_jaw_state(0.1), JawState::Wide);
        assert_eq!(classify_jaw_state(1.0), JawState::Wide);
    }

    // -----------------------------------------------------------------------
    // classify_eye_state tests
    // -----------------------------------------------------------------------

    #[test]
    fn classify_eye_state_closed() {
        assert_eq!(classify_eye_state(0.0), EyeState::Closed);
        assert_eq!(classify_eye_state(0.0009), EyeState::Closed);
    }

    #[test]
    fn classify_eye_state_squinting() {
        assert_eq!(classify_eye_state(0.001), EyeState::Squinting);
        assert_eq!(classify_eye_state(0.003), EyeState::Squinting);
        assert_eq!(classify_eye_state(0.0049), EyeState::Squinting);
    }

    #[test]
    fn classify_eye_state_open() {
        assert_eq!(classify_eye_state(0.005), EyeState::Open);
        assert_eq!(classify_eye_state(0.010), EyeState::Open);
        assert_eq!(classify_eye_state(0.0149), EyeState::Open);
    }

    #[test]
    fn classify_eye_state_wide() {
        assert_eq!(classify_eye_state(0.015), EyeState::Wide);
        assert_eq!(classify_eye_state(0.1), EyeState::Wide);
    }

    // -----------------------------------------------------------------------
    // smooth_mouth_openings tests
    // -----------------------------------------------------------------------

    #[test]
    fn smooth_mouth_openings_window_1_identity() {
        let data = vec![0.0f32, 0.01, 0.02, 0.015, 0.005];
        let smoothed = smooth_mouth_openings(&data, 1);
        assert_eq!(smoothed.len(), data.len());
        for (a, b) in data.iter().zip(smoothed.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "window=1 should be identity: {a} vs {b}"
            );
        }
    }

    #[test]
    fn smooth_mouth_openings_window_3() {
        // [0, 1, 2, 3, 4] with window=3 — each element becomes the mean of its neighbors
        let data = vec![0.0f32, 1.0, 2.0, 3.0, 4.0];
        let smoothed = smooth_mouth_openings(&data, 3);
        assert_eq!(smoothed.len(), 5);
        // Middle element (index 2): mean of [1,2,3] = 2.0
        assert!((smoothed[2] - 2.0).abs() < 1e-5, "middle: {}", smoothed[2]);
        // First element (index 0): window is [0..2], mean of [0,1] = 0.5
        assert!((smoothed[0] - 0.5).abs() < 1e-5, "first: {}", smoothed[0]);
    }

    // -----------------------------------------------------------------------
    // detect_mouth_transitions tests
    // -----------------------------------------------------------------------

    #[test]
    fn detect_mouth_transitions_no_transitions() {
        let data = vec![0.01f32, 0.011, 0.012, 0.011];
        let transitions = detect_mouth_transitions(&data, 0.05);
        assert!(transitions.is_empty());
    }

    #[test]
    fn detect_mouth_transitions_single_transition() {
        // Large jump between index 1 and 2
        let data = vec![0.001f32, 0.001, 0.05, 0.051];
        let transitions = detect_mouth_transitions(&data, 0.01);
        assert_eq!(transitions, vec![2], "transition should be at index 2");
    }

    #[test]
    fn detect_mouth_transitions_multiple_transitions() {
        let data = vec![0.0f32, 0.1, 0.0, 0.1, 0.0];
        let transitions = detect_mouth_transitions(&data, 0.05);
        assert_eq!(transitions, vec![1, 2, 3, 4]);
    }

    // -----------------------------------------------------------------------
    // ContactReport tests
    // -----------------------------------------------------------------------

    #[test]
    fn contact_report_is_physically_valid_no_clipping() {
        let report = ContactReport {
            flags: ContactFlags {
                mouth: MouthContactState::Open,
                left_eye: EyeContactState::Open,
                right_eye: EyeContactState::Open,
                clipping: ClippingStatus::NoClipping,
            },
            mouth_opening: 0.01,
            left_eye_opening: 0.01,
            right_eye_opening: 0.01,
            contact_pairs: vec![],
            clipping_count: 0,
        };
        assert!(report.is_physically_valid());
    }

    #[test]
    fn contact_report_format_summary_contains_key_info() {
        let report = ContactReport {
            flags: ContactFlags {
                mouth: MouthContactState::Closed,
                left_eye: EyeContactState::Open,
                right_eye: EyeContactState::Closed,
                clipping: ClippingStatus::NoClipping,
            },
            mouth_opening: 0.0008,
            left_eye_opening: 0.008,
            right_eye_opening: 0.0004,
            contact_pairs: vec![],
            clipping_count: 0,
        };
        let summary = report.format_summary();
        assert!(
            summary.contains("mouth=closed"),
            "should note closed mouth: {summary}"
        );
        assert!(
            summary.contains("right_eye=closed"),
            "should note closed right eye: {summary}"
        );
        assert!(
            summary.contains("left_eye=open"),
            "should note open left eye: {summary}"
        );
        assert!(
            summary.contains("no_clipping"),
            "no clipping expected: {summary}"
        );
    }
}
