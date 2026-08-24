//! Vertex mask (face region segmentation) for FLAME meshes.
//!
//! FLAME meshes have semantic face regions (eyes, mouth, scalp, …).  Two
//! constructors are available, in order of preference:
//!
//! 1. **Real masks** — [`VertexMask::from_region_map`] (and its JSON sibling
//!    [`VertexMask::from_region_map_json`]) take the region → vertex-index lists
//!    shipped with the FLAME model.  Exporting those index arrays once, offline,
//!    needs no Python at run time and is exact.
//! 2. **Geometric approximation** — [`VertexMask::from_vertices`] classifies by
//!    coordinate thresholds.  It is a fallback: the thresholds are hand-tuned
//!    approximations, and they are only meaningful for a mesh in the FLAME
//!    canonical pose.  [`VertexMask::from_vertices_checked`] enforces that
//!    precondition instead of silently misclassifying a posed or translated
//!    mesh.
//!
//! ## Coordinate System
//!
//! All threshold values are expressed in FLAME canonical-pose metric units
//! (right-handed, +X right, +Y up, +Z forward):
//! - `y` axis: up (+) → scalp, down (−) → neck
//! - `x` axis: left−right (from subject perspective)
//! - `z` axis: forward (+) → nose/lips protrude outward

use std::collections::HashMap;

use nalgebra as na;
use thiserror::Error;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors produced by the [`VertexMask`] constructors.
#[derive(Debug, Error)]
pub enum VertexMaskError {
    /// A region name in the supplied map is not a known [`FaceRegion`].
    #[error("Unknown face region '{name}'")]
    UnknownRegion { name: String },

    /// A region index list references a vertex beyond the end of the mesh.
    #[error("Vertex index {index} out of range (num_vertices = {num_vertices})")]
    IndexOutOfRange { index: usize, num_vertices: usize },

    /// The same vertex was assigned to two different regions.
    #[error("Vertex {index} assigned to both '{first}' and '{second}'")]
    DuplicateAssignment {
        index: usize,
        first: &'static str,
        second: &'static str,
    },

    /// The region map could not be parsed.
    #[error("Invalid region-map JSON: {0}")]
    InvalidJson(String),

    /// The mesh is not in the FLAME canonical pose, so the geometric
    /// thresholds would be evaluated in the wrong frame.
    #[error("Mesh is not in the FLAME canonical pose: {detail}")]
    NonCanonicalPose { detail: String },
}

// ---------------------------------------------------------------------------
// FaceRegion
// ---------------------------------------------------------------------------

/// Semantic regions of the FLAME face mesh.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum FaceRegion {
    /// Central face area (forehead, nose, cheeks).
    Face,
    /// Left eye region (from the subject's perspective: positive X side).
    LeftEye,
    /// Right eye region (from the subject's perspective: negative X side).
    RightEye,
    /// Mouth / lip region.
    Mouth,
    /// Neck area (below the jaw).
    Neck,
    /// Left ear (positive X side in canonical pose).
    LeftEar,
    /// Right ear (negative X side in canonical pose).
    RightEar,
    /// Top of head / hair area.
    Scalp,
}

impl FaceRegion {
    /// Every region, in a fixed order.
    pub const ALL: [Self; 8] = [
        Self::Face,
        Self::LeftEye,
        Self::RightEye,
        Self::Mouth,
        Self::Neck,
        Self::LeftEar,
        Self::RightEar,
        Self::Scalp,
    ];

    /// Return a human-readable name for this region.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::Face => "face",
            Self::LeftEye => "left_eye",
            Self::RightEye => "right_eye",
            Self::Mouth => "mouth",
            Self::Neck => "neck",
            Self::LeftEar => "left_ear",
            Self::RightEar => "right_ear",
            Self::Scalp => "scalp",
        }
    }

    /// Parse a region from its [`FaceRegion::name`] string.
    ///
    /// A few spellings used by the published FLAME mask files are accepted as
    /// aliases (`"face_region"`, `"left_eyeball"`, `"lips"`, …).
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "face" | "face_region" => Some(Self::Face),
            "left_eye" | "left_eye_region" | "left_eyeball" => Some(Self::LeftEye),
            "right_eye" | "right_eye_region" | "right_eyeball" => Some(Self::RightEye),
            "mouth" | "lips" => Some(Self::Mouth),
            "neck" => Some(Self::Neck),
            "left_ear" => Some(Self::LeftEar),
            "right_ear" => Some(Self::RightEar),
            "scalp" => Some(Self::Scalp),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Geometric classification thresholds
// ---------------------------------------------------------------------------

/// Vertices with y above this value belong to the scalp.
const SCALP_Y_MIN: f32 = 0.15;

/// Vertices with y below this value belong to the neck.
const NECK_Y_MAX: f32 = -0.12;

/// Ear region: |x| greater than this value and y in the middle band.
const EAR_X_MIN: f32 = 0.08;
/// Ear region: y lower bound.
const EAR_Y_MIN: f32 = -0.05;
/// Ear region: y upper bound.
const EAR_Y_MAX: f32 = 0.05;

/// Eye region: y lower bound.
const EYE_Y_MIN: f32 = 0.02;
/// Eye region: y upper bound.
const EYE_Y_MAX: f32 = 0.10;
/// Eye region: |x| upper bound (eyes are near the center in x).
const EYE_X_MAX: f32 = 0.06;
/// Eye region: |x| lower bound.
///
/// The eyes straddle the midline at roughly half the interocular distance
/// (≈ 0.063 m on a canonical FLAME head, so ±0.03 m); without a lower bound the
/// band `|x| < EYE_X_MAX ∧ z > EYE_Z_MIN` is dominated by the nose bridge,
/// which sits on the midline and protrudes furthest forward.
const EYE_X_MIN: f32 = 0.015;
/// Eye region: z minimum (eyes protrude forward).
const EYE_Z_MIN: f32 = 0.06;

/// Mouth region: y lower bound.
const MOUTH_Y_MIN: f32 = -0.10;
/// Mouth region: y upper bound.
const MOUTH_Y_MAX: f32 = 0.00;
/// Mouth region: z minimum (lips protrude forward).
const MOUTH_Z_MIN: f32 = 0.06;

// ---------------------------------------------------------------------------
// Canonical-pose acceptance bounds (preconditions of the geometric fallback)
// ---------------------------------------------------------------------------

/// Maximum distance of the vertex centroid from the origin, in metres.
///
/// A canonical FLAME head is centred near the origin; a posed or translated
/// mesh (e.g. the output of `FlameModel::forward` with a non-zero
/// `FlameParams::translation`) is not, and every threshold above would then be
/// evaluated in the wrong frame.
///
/// The bound is deliberately loose — roughly the half-height of a canonical
/// head, i.e. the distance at which the scalp/neck thresholds
/// (`SCALP_Y_MIN`/`NECK_Y_MAX`, ±0.12…0.15) start swallowing whole regions.
/// A tighter value risks rejecting legitimately canonical meshes whose exact
/// centroid has never been measured here; this one still catches the
/// translations that break the classification.
const CANONICAL_CENTROID_MAX: f32 = 0.15;

/// Maximum half-extent of a canonical FLAME head along any axis, in metres.
const CANONICAL_EXTENT_MAX: f32 = 0.5;

/// Minimum half-extent of a canonical FLAME head along any axis, in metres.
///
/// Guards against a mesh expressed in different units (e.g. millimetres scaled
/// down, or a unit-normalized mesh).
const CANONICAL_EXTENT_MIN: f32 = 0.02;

// ---------------------------------------------------------------------------
// VertexMask
// ---------------------------------------------------------------------------

/// A vertex mask: which semantic region each vertex belongs to.
#[derive(Debug, Clone)]
pub struct VertexMask {
    /// Region assignment per vertex. Length equals `num_vertices`.
    pub regions: Vec<FaceRegion>,
    /// Total number of vertices in the mesh.
    pub num_vertices: usize,
}

impl VertexMask {
    /// Build a mask from real FLAME region index arrays.
    ///
    /// `regions` maps a region name (see [`FaceRegion::from_name`]) to the
    /// vertex indices belonging to it — exactly the contents of the FLAME mask
    /// arrays, which can be exported once offline and embedded or shipped
    /// alongside the model, with no Python at run time.  Vertices that appear in
    /// no list default to [`FaceRegion::Face`].
    ///
    /// This is the exact constructor; prefer it over the geometric
    /// approximation in [`VertexMask::from_vertices`].
    ///
    /// # Errors
    ///
    /// [`VertexMaskError::UnknownRegion`] for an unrecognized region name,
    /// [`VertexMaskError::IndexOutOfRange`] when an index exceeds
    /// `num_vertices`, and [`VertexMaskError::DuplicateAssignment`] when two
    /// regions claim the same vertex.
    pub fn from_region_map(
        num_vertices: usize,
        regions: &HashMap<String, Vec<u32>>,
    ) -> Result<Self, VertexMaskError> {
        let mut assigned: Vec<Option<FaceRegion>> = vec![None; num_vertices];

        // Sort by region name so that error reporting is deterministic even
        // though `HashMap` iteration order is not.
        let mut entries: Vec<(&String, &Vec<u32>)> = regions.iter().collect();
        entries.sort_by(|a, b| a.0.cmp(b.0));

        for (name, indices) in entries {
            let region = FaceRegion::from_name(name)
                .ok_or_else(|| VertexMaskError::UnknownRegion { name: name.clone() })?;
            for &raw_index in indices {
                let index = raw_index as usize;
                let slot =
                    assigned
                        .get_mut(index)
                        .ok_or_else(|| VertexMaskError::IndexOutOfRange {
                            index,
                            num_vertices,
                        })?;
                if let Some(existing) = slot {
                    if *existing != region {
                        return Err(VertexMaskError::DuplicateAssignment {
                            index,
                            first: existing.name(),
                            second: region.name(),
                        });
                    }
                }
                *slot = Some(region.clone());
            }
        }

        let regions: Vec<FaceRegion> = assigned
            .into_iter()
            .map(|slot| slot.unwrap_or(FaceRegion::Face))
            .collect();
        Ok(Self {
            regions,
            num_vertices,
        })
    }

    /// Build a mask from a JSON region map: `{"left_eye": [12, 13, …], …}`.
    ///
    /// Thin wrapper over [`VertexMask::from_region_map`] for masks exported to
    /// JSON; the caller reads the file (or embeds the string) itself.
    ///
    /// # Errors
    ///
    /// [`VertexMaskError::InvalidJson`] when the document is not an object of
    /// integer arrays, plus every error of [`VertexMask::from_region_map`].
    pub fn from_region_map_json(num_vertices: usize, json: &str) -> Result<Self, VertexMaskError> {
        let parsed: HashMap<String, Vec<u32>> =
            serde_json::from_str(json).map_err(|e| VertexMaskError::InvalidJson(e.to_string()))?;
        Self::from_region_map(num_vertices, &parsed)
    }

    /// Compute vertex masks geometrically from vertex positions.
    ///
    /// **Approximation, not the real FLAME masks.** The thresholds below are
    /// hand-tuned coordinate bounds and are only meaningful for a mesh in the
    /// FLAME canonical pose — applying them to a posed or translated mesh
    /// classifies in the wrong frame.  Prefer
    /// [`VertexMask::from_region_map`] when the real masks are available, and
    /// [`VertexMask::from_vertices_checked`] when the caller can handle an error
    /// instead of the `tracing` warning this function emits for a mesh that
    /// fails the canonical-pose check.
    ///
    /// Classification order (highest priority first):
    ///
    /// 1. **Scalp**: `y > SCALP_Y_MIN`
    /// 2. **Neck**: `y < NECK_Y_MAX`
    /// 3. **Ears**: `|x| > EAR_X_MIN` with `y ∈ [EAR_Y_MIN, EAR_Y_MAX]`
    ///    - Left ear: `x > 0`
    ///    - Right ear: `x < 0`
    /// 4. **Eyes**: `y ∈ [EYE_Y_MIN, EYE_Y_MAX]`,
    ///    `EYE_X_MIN < |x| < EYE_X_MAX`, `z > EYE_Z_MIN`
    ///    - Left eye: `x > 0` (positive x is the subject's left side)
    ///    - Right eye: `x < 0`
    /// 5. **Mouth**: `y ∈ [MOUTH_Y_MIN, MOUTH_Y_MAX]`, `z > MOUTH_Z_MIN`
    /// 6. **Face**: everything else
    #[must_use]
    pub fn from_vertices(vertices: &[na::Point3<f32>]) -> Self {
        if let Some(detail) = canonical_pose_violation(vertices) {
            tracing::warn!(
                "VertexMask::from_vertices: {}; the geometric thresholds assume \
                 the FLAME canonical pose, so this classification is unreliable — \
                 use VertexMask::from_region_map with the real FLAME masks, or \
                 canonicalize the mesh first",
                detail
            );
        }
        Self::classify_all(vertices)
    }

    /// Same as [`VertexMask::from_vertices`], but rejects a mesh that is not in
    /// the FLAME canonical pose instead of warning.
    ///
    /// # Errors
    ///
    /// [`VertexMaskError::NonCanonicalPose`] when the vertex centroid is farther
    /// than `CANONICAL_CENTROID_MAX` from the origin, or the bounding box does
    /// not match canonical FLAME extents.
    pub fn from_vertices_checked(vertices: &[na::Point3<f32>]) -> Result<Self, VertexMaskError> {
        match canonical_pose_violation(vertices) {
            Some(detail) => Err(VertexMaskError::NonCanonicalPose { detail }),
            None => Ok(Self::classify_all(vertices)),
        }
    }

    /// Threshold classification of every vertex (no precondition check).
    fn classify_all(vertices: &[na::Point3<f32>]) -> Self {
        let regions: Vec<FaceRegion> = vertices
            .iter()
            .map(|v| classify_vertex(v.x, v.y, v.z))
            .collect();

        let num_vertices = regions.len();
        Self {
            regions,
            num_vertices,
        }
    }

    /// Get the indices of vertices assigned to `region`.
    #[must_use]
    pub fn region_indices(&self, region: &FaceRegion) -> Vec<usize> {
        self.regions
            .iter()
            .enumerate()
            .filter_map(|(i, r)| if r == region { Some(i) } else { None })
            .collect()
    }

    /// Get a boolean mask where `true` means the vertex belongs to `region`.
    ///
    /// The returned `Vec` has the same length as `num_vertices`.
    #[must_use]
    pub fn region_mask(&self, region: &FaceRegion) -> Vec<bool> {
        self.regions.iter().map(|r| r == region).collect()
    }

    /// Get the region for a single vertex by index.
    ///
    /// Returns `None` if `idx >= num_vertices`.
    #[must_use]
    pub fn vertex_region(&self, idx: usize) -> Option<&FaceRegion> {
        self.regions.get(idx)
    }

    /// Count vertices per region.
    ///
    /// Returns a `HashMap` from region name string to vertex count.
    /// All eight regions appear in the map (with zero counts for empty regions).
    #[must_use]
    pub fn region_counts(&self) -> HashMap<String, usize> {
        let mut counts: HashMap<String, usize> = FaceRegion::ALL
            .iter()
            .map(|r| (r.name().to_string(), 0usize))
            .collect();

        for region in &self.regions {
            *counts.entry(region.name().to_string()).or_insert(0) += 1;
        }

        counts
    }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Describe why `vertices` cannot be in the FLAME canonical pose, or `None`
/// when the mesh passes the check.
///
/// Two cheap necessary conditions: the centroid sits near the origin, and the
/// bounding box has head-like extents.  A mesh that fails either one would be
/// classified against thresholds expressed in a different frame — e.g. a head
/// translated by `y = +0.2` has *every* vertex above `SCALP_Y_MIN`.
fn canonical_pose_violation(vertices: &[na::Point3<f32>]) -> Option<String> {
    if vertices.is_empty() {
        return None;
    }

    let n = vertices.len() as f32;
    let mut centroid = [0.0f32; 3];
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for v in vertices {
        for (axis, value) in [v.x, v.y, v.z].into_iter().enumerate() {
            centroid[axis] += value / n;
            min[axis] = min[axis].min(value);
            max[axis] = max[axis].max(value);
        }
    }

    let offset =
        (centroid[0] * centroid[0] + centroid[1] * centroid[1] + centroid[2] * centroid[2]).sqrt();
    if !offset.is_finite() || offset > CANONICAL_CENTROID_MAX {
        return Some(format!(
            "vertex centroid is {offset:.3} from the origin (limit {CANONICAL_CENTROID_MAX})"
        ));
    }

    let half_extent = (0..3)
        .map(|axis| 0.5 * (max[axis] - min[axis]))
        .fold(0.0f32, f32::max);
    if !half_extent.is_finite()
        || half_extent > CANONICAL_EXTENT_MAX
        || half_extent < CANONICAL_EXTENT_MIN
    {
        return Some(format!(
            "largest half-extent is {half_extent:.3}, outside the canonical FLAME range \
             [{CANONICAL_EXTENT_MIN}, {CANONICAL_EXTENT_MAX}]"
        ));
    }

    None
}

/// Classify a single vertex `(x, y, z)` into a [`FaceRegion`].
///
/// Rules are evaluated in priority order; the first matching rule wins.
#[inline]
fn classify_vertex(x: f32, y: f32, z: f32) -> FaceRegion {
    // 1. Scalp: top of head
    if y > SCALP_Y_MIN {
        return FaceRegion::Scalp;
    }

    // 2. Neck: below the jaw
    if y < NECK_Y_MAX {
        return FaceRegion::Neck;
    }

    // 3. Ears: far out on the x axis in the middle y band
    if x.abs() > EAR_X_MIN && (EAR_Y_MIN..=EAR_Y_MAX).contains(&y) {
        if x > 0.0 {
            return FaceRegion::LeftEar;
        }
        return FaceRegion::RightEar;
    }

    // 4. Eyes: upper face, off the midline but still central, protruding forward
    if (EYE_Y_MIN..=EYE_Y_MAX).contains(&y)
        && (EYE_X_MIN..EYE_X_MAX).contains(&x.abs())
        && z > EYE_Z_MIN
    {
        if x > 0.0 {
            return FaceRegion::LeftEye;
        }
        return FaceRegion::RightEye;
    }

    // 5. Mouth: lower face, protruding forward
    if (MOUTH_Y_MIN..=MOUTH_Y_MAX).contains(&y) && z > MOUTH_Z_MIN {
        return FaceRegion::Mouth;
    }

    // 6. Default: central face area
    FaceRegion::Face
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Helper: build a VertexMask from raw [f32;3] arrays
    // -----------------------------------------------------------------------

    fn make_mask(raw: &[[f32; 3]]) -> VertexMask {
        let pts: Vec<na::Point3<f32>> = raw
            .iter()
            .map(|&[x, y, z]| na::Point3::new(x, y, z))
            .collect();
        VertexMask::from_vertices(&pts)
    }

    // -----------------------------------------------------------------------
    // Basic classification tests
    // -----------------------------------------------------------------------

    #[test]
    fn scalp_region_for_high_y_vertices() {
        // All vertices have y well above the SCALP_Y_MIN threshold (0.15)
        let raw: &[[f32; 3]] = &[[0.0, 0.20, 0.0], [0.01, 0.25, 0.01], [-0.02, 0.30, -0.01]];
        let mask = make_mask(raw);
        for (i, r) in mask.regions.iter().enumerate() {
            assert_eq!(
                *r,
                FaceRegion::Scalp,
                "vertex {i} (y={}) should be Scalp",
                raw[i][1]
            );
        }
    }

    #[test]
    fn neck_region_for_low_y_vertices() {
        // All vertices have y well below NECK_Y_MAX (-0.12)
        let raw: &[[f32; 3]] = &[
            [0.0, -0.15, 0.0],
            [0.01, -0.20, 0.01],
            [-0.02, -0.25, -0.01],
        ];
        let mask = make_mask(raw);
        for (i, r) in mask.regions.iter().enumerate() {
            assert_eq!(
                *r,
                FaceRegion::Neck,
                "vertex {i} (y={}) should be Neck",
                raw[i][1]
            );
        }
    }

    #[test]
    fn from_vertices_correct_region_assignment() {
        // Build a synthetic vertex set that covers each major region
        let raw: &[[f32; 3]] = &[
            [0.0, 0.25, 0.0],    // idx 0: Scalp (y > 0.15)
            [0.0, -0.15, 0.0],   // idx 1: Neck  (y < -0.12)
            [0.10, 0.00, 0.0],   // idx 2: LeftEar (|x|>0.08, y in ear band)
            [-0.10, 0.00, 0.0],  // idx 3: RightEar
            [0.02, 0.05, 0.08],  // idx 4: LeftEye (x>0, y in eye band, z>0.06)
            [-0.02, 0.05, 0.08], // idx 5: RightEye
            [0.0, -0.05, 0.08],  // idx 6: Mouth (y in mouth band, z>0.06)
            [0.0, 0.05, 0.0],    // idx 7: Face (nothing else matches)
        ];
        let mask = make_mask(raw);

        assert_eq!(mask.regions[0], FaceRegion::Scalp, "idx 0 should be Scalp");
        assert_eq!(mask.regions[1], FaceRegion::Neck, "idx 1 should be Neck");
        assert_eq!(
            mask.regions[2],
            FaceRegion::LeftEar,
            "idx 2 should be LeftEar"
        );
        assert_eq!(
            mask.regions[3],
            FaceRegion::RightEar,
            "idx 3 should be RightEar"
        );
        assert_eq!(
            mask.regions[4],
            FaceRegion::LeftEye,
            "idx 4 should be LeftEye"
        );
        assert_eq!(
            mask.regions[5],
            FaceRegion::RightEye,
            "idx 5 should be RightEye"
        );
        assert_eq!(mask.regions[6], FaceRegion::Mouth, "idx 6 should be Mouth");
        assert_eq!(mask.regions[7], FaceRegion::Face, "idx 7 should be Face");
    }

    // -----------------------------------------------------------------------
    // All vertices assigned some region
    // -----------------------------------------------------------------------

    #[test]
    fn all_vertices_assigned_some_region() {
        // Use a random spread of vertices across the canonical FLAME coordinate space
        let raw: &[[f32; 3]] = &[
            [0.0, 0.0, 0.0],
            [0.05, 0.05, 0.05],
            [-0.05, -0.05, -0.05],
            [0.1, 0.2, 0.1],
            [-0.1, -0.2, -0.1],
            [0.0, 0.3, 0.0],
            [0.0, -0.3, 0.0],
        ];
        let mask = make_mask(raw);
        // Every vertex must be assigned (no panics, all 7 regions covered)
        assert_eq!(mask.regions.len(), raw.len());
        // Verify region_counts sums to total
        let counts = mask.region_counts();
        let total: usize = counts.values().sum();
        assert_eq!(total, raw.len(), "region counts must sum to vertex count");
    }

    // -----------------------------------------------------------------------
    // region_indices / region_mask
    // -----------------------------------------------------------------------

    #[test]
    fn region_indices_returns_subset_of_all_indices() {
        let raw: &[[f32; 3]] = &[
            [0.0, 0.25, 0.0],  // Scalp
            [0.0, -0.15, 0.0], // Neck
            [0.0, 0.05, 0.0],  // Face
        ];
        let mask = make_mask(raw);

        let scalp_idx = mask.region_indices(&FaceRegion::Scalp);
        assert_eq!(scalp_idx, vec![0], "only vertex 0 is scalp");

        let neck_idx = mask.region_indices(&FaceRegion::Neck);
        assert_eq!(neck_idx, vec![1], "only vertex 1 is neck");

        let face_idx = mask.region_indices(&FaceRegion::Face);
        assert_eq!(face_idx, vec![2], "only vertex 2 is face");

        // Combined size equals total vertices
        assert_eq!(scalp_idx.len() + neck_idx.len() + face_idx.len(), 3);
    }

    #[test]
    fn region_mask_has_same_length_as_num_vertices() {
        let raw: &[[f32; 3]] = &[
            [0.0, 0.25, 0.0],
            [0.0, -0.15, 0.0],
            [0.0, 0.05, 0.0],
            [0.0, 0.10, 0.0],
            [0.0, 0.00, 0.0],
        ];
        let mask = make_mask(raw);
        let n = mask.num_vertices;

        for region in &[
            FaceRegion::Face,
            FaceRegion::Scalp,
            FaceRegion::Neck,
            FaceRegion::Mouth,
            FaceRegion::LeftEye,
            FaceRegion::RightEye,
            FaceRegion::LeftEar,
            FaceRegion::RightEar,
        ] {
            let m = mask.region_mask(region);
            assert_eq!(
                m.len(),
                n,
                "region_mask for {:?} has length {} != num_vertices {}",
                region,
                m.len(),
                n
            );
        }
    }

    // -----------------------------------------------------------------------
    // region_counts
    // -----------------------------------------------------------------------

    #[test]
    fn region_counts_sums_to_num_vertices() {
        // Use 20 vertices with diverse positions
        let mut raw: Vec<[f32; 3]> = Vec::new();
        for i in 0..20i32 {
            let y = (i as f32 - 10.0) * 0.03;
            raw.push([0.0, y, 0.04]);
        }
        let mask = make_mask(&raw);
        let counts = mask.region_counts();
        let total: usize = counts.values().sum();
        assert_eq!(
            total, mask.num_vertices,
            "region_counts total {total} != num_vertices {}",
            mask.num_vertices
        );
    }

    // -----------------------------------------------------------------------
    // vertex_region
    // -----------------------------------------------------------------------

    #[test]
    fn vertex_region_returns_none_for_out_of_bounds() {
        let raw: &[[f32; 3]] = &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let mask = make_mask(raw);

        // In-bounds
        assert!(mask.vertex_region(0).is_some(), "index 0 should be Some");
        assert!(mask.vertex_region(1).is_some(), "index 1 should be Some");
        // Out-of-bounds
        assert!(
            mask.vertex_region(2).is_none(),
            "index 2 should be None (out of bounds)"
        );
        assert!(
            mask.vertex_region(1000).is_none(),
            "index 1000 should be None (out of bounds)"
        );
    }

    // -----------------------------------------------------------------------
    // Boolean mask count consistency
    // -----------------------------------------------------------------------

    #[test]
    fn boolean_mask_true_count_matches_indices_count() {
        let raw: &[[f32; 3]] = &[
            [0.0, 0.25, 0.0],  // Scalp
            [0.0, 0.25, 0.0],  // Scalp
            [0.0, -0.15, 0.0], // Neck
            [0.0, 0.05, 0.0],  // Face
        ];
        let mask = make_mask(raw);

        for region in &[FaceRegion::Scalp, FaceRegion::Neck, FaceRegion::Face] {
            let indices = mask.region_indices(region);
            let bool_mask = mask.region_mask(region);
            let true_count = bool_mask.iter().filter(|&&b| b).count();
            assert_eq!(
                indices.len(),
                true_count,
                "region {:?}: indices count {} != bool mask true count {}",
                region,
                indices.len(),
                true_count
            );
        }
    }

    // -----------------------------------------------------------------------
    // Integration with Mesh::vertex_mask
    // -----------------------------------------------------------------------

    #[test]
    fn mesh_vertex_mask_works_on_test_mesh() {
        use crate::Mesh;

        let vertices = vec![
            na::Point3::new(0.0f32, 0.25, 0.0),  // Scalp
            na::Point3::new(0.0f32, -0.15, 0.0), // Neck
            na::Point3::new(0.0f32, 0.05, 0.0),  // Face
        ];
        let faces = vec![[0u32, 1, 2]];
        let mesh = Mesh::new(vertices, faces);

        let vm = mesh.vertex_mask();
        assert_eq!(vm.num_vertices, 3, "vertex_mask num_vertices mismatch");
        assert_eq!(vm.regions[0], FaceRegion::Scalp, "idx 0 should be Scalp");
        assert_eq!(vm.regions[1], FaceRegion::Neck, "idx 1 should be Neck");
        assert_eq!(vm.regions[2], FaceRegion::Face, "idx 2 should be Face");
    }

    // -----------------------------------------------------------------------
    // Empty vertex set
    // -----------------------------------------------------------------------

    #[test]
    fn from_vertices_empty_produces_empty_mask() {
        let mask = make_mask(&[]);
        assert_eq!(mask.num_vertices, 0);
        assert!(mask.regions.is_empty());
        let counts = mask.region_counts();
        let total: usize = counts.values().sum();
        assert_eq!(total, 0, "empty mask should have zero total count");
    }

    // -----------------------------------------------------------------------
    // Real-mask constructors
    // -----------------------------------------------------------------------

    #[test]
    fn from_region_map_assigns_listed_indices() {
        let mut regions: HashMap<String, Vec<u32>> = HashMap::new();
        regions.insert("left_eye".to_string(), vec![0, 1]);
        regions.insert("mouth".to_string(), vec![3]);

        let mask = VertexMask::from_region_map(5, &regions).expect("valid region map");
        assert_eq!(mask.num_vertices, 5);
        assert_eq!(mask.regions[0], FaceRegion::LeftEye);
        assert_eq!(mask.regions[1], FaceRegion::LeftEye);
        assert_eq!(mask.regions[2], FaceRegion::Face, "unlisted vertex → Face");
        assert_eq!(mask.regions[3], FaceRegion::Mouth);
        assert_eq!(mask.region_indices(&FaceRegion::LeftEye), vec![0, 1]);
    }

    #[test]
    fn from_region_map_rejects_bad_input() {
        let single = |name: &str, indices: Vec<u32>| {
            let mut map: HashMap<String, Vec<u32>> = HashMap::new();
            map.insert(name.to_string(), indices);
            map
        };

        assert!(matches!(
            VertexMask::from_region_map(2, &single("nose", vec![0])),
            Err(VertexMaskError::UnknownRegion { .. })
        ));
        assert!(matches!(
            VertexMask::from_region_map(2, &single("scalp", vec![7])),
            Err(VertexMaskError::IndexOutOfRange { .. })
        ));

        let mut duplicate = single("scalp", vec![0]);
        duplicate.insert("neck".to_string(), vec![0]);
        assert!(matches!(
            VertexMask::from_region_map(2, &duplicate),
            Err(VertexMaskError::DuplicateAssignment { .. })
        ));
    }

    #[test]
    fn from_region_map_json_parses_and_reports_errors() {
        let json = r#"{"scalp": [0, 1], "neck": [2]}"#;
        let mask = VertexMask::from_region_map_json(3, json).expect("valid json");
        assert_eq!(
            mask.regions,
            vec![FaceRegion::Scalp, FaceRegion::Scalp, FaceRegion::Neck]
        );
        assert!(matches!(
            VertexMask::from_region_map_json(3, "not json"),
            Err(VertexMaskError::InvalidJson(_))
        ));
    }

    // -----------------------------------------------------------------------
    // Canonical-pose precondition
    // -----------------------------------------------------------------------

    fn canonical_head() -> Vec<na::Point3<f32>> {
        vec![
            na::Point3::new(-0.08_f32, -0.15, -0.08),
            na::Point3::new(0.08_f32, 0.15, 0.08),
            na::Point3::new(0.0_f32, 0.0, 0.05),
        ]
    }

    #[test]
    fn from_vertices_checked_accepts_canonical_mesh() {
        assert!(VertexMask::from_vertices_checked(&canonical_head()).is_ok());
    }

    #[test]
    fn from_vertices_checked_rejects_translated_mesh() {
        let canonical = canonical_head();
        // The same head translated by y = +0.2 — what `FlameModel::forward`
        // returns for a non-zero `FlameParams::translation`.
        let translated: Vec<na::Point3<f32>> = canonical
            .iter()
            .map(|p| na::Point3::new(p.x, p.y + 0.2, p.z))
            .collect();

        assert!(matches!(
            VertexMask::from_vertices_checked(&translated),
            Err(VertexMaskError::NonCanonicalPose { .. })
        ));
        // The infallible path still returns a mask (so `Mesh::vertex_mask`
        // keeps working) but the classification differs from the canonical one —
        // exactly the silent misclassification the checked variant reports.
        assert_ne!(
            VertexMask::from_vertices(&translated).regions,
            VertexMask::from_vertices(&canonical).regions
        );
    }

    #[test]
    fn from_vertices_checked_rejects_wrongly_scaled_mesh() {
        let huge: Vec<na::Point3<f32>> = canonical_head()
            .iter()
            .map(|p| na::Point3::new(p.x * 100.0, p.y * 100.0, p.z * 100.0))
            .collect();
        assert!(matches!(
            VertexMask::from_vertices_checked(&huge),
            Err(VertexMaskError::NonCanonicalPose { .. })
        ));
    }

    // -----------------------------------------------------------------------
    // Eye rule excludes the midline (nose bridge)
    // -----------------------------------------------------------------------

    #[test]
    fn nose_bridge_is_not_classified_as_eye() {
        // Midline, protruding forward, inside the eye y-band: the nose bridge.
        assert_eq!(classify_vertex(0.0, 0.05, 0.08), FaceRegion::Face);
        // Real eyes sit roughly half an interocular distance off the midline.
        assert_eq!(classify_vertex(0.03, 0.05, 0.08), FaceRegion::LeftEye);
        assert_eq!(classify_vertex(-0.03, 0.05, 0.08), FaceRegion::RightEye);
    }

    #[test]
    fn face_region_name_roundtrip() {
        for region in FaceRegion::ALL {
            assert_eq!(FaceRegion::from_name(region.name()), Some(region.clone()));
        }
        assert_eq!(FaceRegion::from_name("not_a_region"), None);
    }
}
