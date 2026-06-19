//! 68-point facial landmark extraction for FLAME meshes.
//!
//! Implements the iBUG 68-point facial landmark convention using hardcoded
//! vertex indices from the canonical FLAME mesh topology (5023 vertices).
//!
//! ## Landmark Groups
//!
//! | Group         | Indices | Count |
//! |---------------|---------|-------|
//! | Jaw line      | 0–16    | 17    |
//! | Left eyebrow  | 17–21   | 5     |
//! | Right eyebrow | 22–26   | 5     |
//! | Nose          | 27–35   | 9     |
//! | Left eye      | 36–41   | 6     |
//! | Right eye     | 42–47   | 6     |
//! | Outer lip     | 48–59   | 12    |
//! | Inner lip     | 60–67   | 8     |

use crate::{error::FlameError, mesh::Mesh};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// FLAME vertex indices for the 68 iBUG landmarks.
///
/// Based on FLAME 2020 canonical mesh topology (5023 vertices).
/// These map the 68 standard iBUG facial landmark positions to their
/// corresponding vertex indices in the FLAME mesh.
const FLAME_68_VERTEX_INDICES: [u32; 68] = [
    // Jaw line (0–16)
    2306, 2304, 2303, 2302, 2301, 2300, 1910, 1908, 1907, 1906, 1905, 2448, 2449, 2450, 2444, 2443,
    2446, // Left eyebrow (17–21)
    3543, 3545, 3547, 3549, 3551, // Right eyebrow (22–26)
    778, 782, 786, 790, 792, // Nose (27–35)
    1662, 1661, 1660, 1659, 1658, 1666, 1665, 1664, 1663, // Left eye (36–41)
    3572, 3574, 3576, 3578, 3579, 3577, // Right eye (42–47)
    815, 817, 819, 821, 822, 820, // Outer lip (48–59)
    2774, 2775, 2776, 2777, 2778, 2779, 2780, 2781, 2782, 2783, 2784, 2785,
    // Inner lip (60–67)
    2790, 2791, 2792, 2793, 2794, 2795, 2796, 2797,
];

/// Total number of canonical FLAME landmarks.
pub const NUM_LANDMARKS: usize = 68;

// ---------------------------------------------------------------------------
// LandmarkGroup
// ---------------------------------------------------------------------------

/// Semantic group a facial landmark belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LandmarkGroup {
    /// Jaw-line contour points (indices 0–16, 17 points).
    JawLine,
    /// Left eyebrow (indices 17–21, 5 points).
    LeftEyebrow,
    /// Right eyebrow (indices 22–26, 5 points).
    RightEyebrow,
    /// Nose bridge and tip (indices 27–35, 9 points).
    Nose,
    /// Left eye outline (indices 36–41, 6 points).
    LeftEye,
    /// Right eye outline (indices 42–47, 6 points).
    RightEye,
    /// Outer lip contour (indices 48–59, 12 points).
    OuterLip,
    /// Inner lip / mouth opening (indices 60–67, 8 points).
    InnerLip,
}

impl LandmarkGroup {
    /// Determine the semantic group from an iBUG landmark index (0-based).
    ///
    /// Any index outside 0–67 falls back to [`LandmarkGroup::JawLine`].
    #[must_use]
    pub fn from_index(idx: usize) -> Self {
        match idx {
            17..=21 => Self::LeftEyebrow,
            22..=26 => Self::RightEyebrow,
            27..=35 => Self::Nose,
            36..=41 => Self::LeftEye,
            42..=47 => Self::RightEye,
            48..=59 => Self::OuterLip,
            60..=67 => Self::InnerLip,
            _ => Self::JawLine, // fallback
        }
    }

    /// Human-readable name for this group.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::JawLine => "jaw_line",
            Self::LeftEyebrow => "left_eyebrow",
            Self::RightEyebrow => "right_eyebrow",
            Self::Nose => "nose",
            Self::LeftEye => "left_eye",
            Self::RightEye => "right_eye",
            Self::OuterLip => "outer_lip",
            Self::InnerLip => "inner_lip",
        }
    }

    /// Number of landmarks belonging to this group.
    #[must_use]
    pub fn count(&self) -> usize {
        match self {
            Self::JawLine => 17,
            Self::LeftEyebrow | Self::RightEyebrow => 5,
            Self::Nose => 9,
            Self::LeftEye | Self::RightEye => 6,
            Self::OuterLip => 12,
            Self::InnerLip => 8,
        }
    }

    /// Inclusive range of iBUG indices that belong to this group.
    #[must_use]
    pub fn index_range(&self) -> std::ops::RangeInclusive<usize> {
        match self {
            Self::JawLine => 0..=16,
            Self::LeftEyebrow => 17..=21,
            Self::RightEyebrow => 22..=26,
            Self::Nose => 27..=35,
            Self::LeftEye => 36..=41,
            Self::RightEye => 42..=47,
            Self::OuterLip => 48..=59,
            Self::InnerLip => 60..=67,
        }
    }
}

// ---------------------------------------------------------------------------
// Landmark
// ---------------------------------------------------------------------------

/// A single facial landmark extracted from a FLAME mesh.
#[derive(Debug, Clone, PartialEq)]
pub struct Landmark {
    /// 3D position in mesh space (same units as the mesh vertices).
    pub position: [f32; 3],
    /// Zero-based index in the canonical 68-point landmark set.
    pub index: usize,
    /// Semantic group this landmark belongs to.
    pub group: LandmarkGroup,
}

// ---------------------------------------------------------------------------
// LandmarkExtractor
// ---------------------------------------------------------------------------

/// Extracts static facial landmarks from a FLAME mesh using precomputed
/// vertex indices.
///
/// By default the extractor uses the canonical 68-point iBUG indices for the
/// FLAME 2020 mesh topology.  Custom indices can be supplied via
/// [`LandmarkExtractor::with_indices`] for fine-tuning or alternative mesh
/// versions.
pub struct LandmarkExtractor {
    /// Vertex indices corresponding to each landmark point.
    vertex_indices: Vec<u32>,
}

impl LandmarkExtractor {
    /// Create an extractor using the canonical FLAME 68-point landmark indices.
    #[must_use]
    pub fn new() -> Self {
        Self {
            vertex_indices: FLAME_68_VERTEX_INDICES.to_vec(),
        }
    }

    /// Create an extractor with custom vertex indices.
    ///
    /// # Errors
    ///
    /// Returns [`FlameError::InvalidParams`] if `indices` is empty.
    pub fn with_indices(indices: Vec<u32>) -> Result<Self, FlameError> {
        if indices.is_empty() {
            return Err(FlameError::InvalidParams(
                "landmark vertex indices must not be empty".to_string(),
            ));
        }
        Ok(Self {
            vertex_indices: indices,
        })
    }

    /// Total number of landmarks this extractor will produce.
    #[inline]
    #[must_use]
    pub fn num_landmarks(&self) -> usize {
        self.vertex_indices.len()
    }

    /// Extract all landmarks from `mesh`.
    ///
    /// # Errors
    ///
    /// Returns [`FlameError::IndexOutOfBounds`] if any stored vertex index
    /// is greater than or equal to the number of vertices in `mesh`.
    pub fn extract(&self, mesh: &Mesh) -> Result<Vec<Landmark>, FlameError> {
        let num_verts = mesh.vertices.len();
        let mut landmarks = Vec::with_capacity(self.vertex_indices.len());

        for (landmark_idx, &vertex_idx) in self.vertex_indices.iter().enumerate() {
            let vi = vertex_idx as usize;
            if vi >= num_verts {
                return Err(FlameError::index_out_of_bounds(
                    format!("landmark {landmark_idx} vertex index"),
                    vi,
                    num_verts,
                ));
            }
            let v = &mesh.vertices[vi];
            landmarks.push(Landmark {
                position: [v.x, v.y, v.z],
                index: landmark_idx,
                group: LandmarkGroup::from_index(landmark_idx),
            });
        }

        Ok(landmarks)
    }

    /// Extract only landmarks belonging to `group`.
    ///
    /// # Errors
    ///
    /// Returns [`FlameError::IndexOutOfBounds`] if any vertex index in the
    /// requested group is out of bounds for `mesh`.
    pub fn extract_group(
        &self,
        mesh: &Mesh,
        group: LandmarkGroup,
    ) -> Result<Vec<Landmark>, FlameError> {
        let range = group.index_range();
        let num_verts = mesh.vertices.len();
        let mut landmarks = Vec::with_capacity(group.count());

        for landmark_idx in range {
            // Indices beyond the extractor's own list are silently skipped.
            let Some(&vertex_idx) = self.vertex_indices.get(landmark_idx) else {
                break;
            };
            let vi = vertex_idx as usize;
            if vi >= num_verts {
                return Err(FlameError::index_out_of_bounds(
                    format!(
                        "landmark {landmark_idx} vertex index for group '{}'",
                        group.name()
                    ),
                    vi,
                    num_verts,
                ));
            }
            let v = &mesh.vertices[vi];
            landmarks.push(Landmark {
                position: [v.x, v.y, v.z],
                index: landmark_idx,
                group,
            });
        }

        Ok(landmarks)
    }

    /// Return the raw vertex indices for a specific landmark group.
    ///
    /// The returned slice covers only the portion of `self.vertex_indices`
    /// that falls within the group's index range and within the extractor's
    /// stored indices.
    #[must_use]
    pub fn group_indices(&self, group: LandmarkGroup) -> &[u32] {
        let range = group.index_range();
        let start = *range.start();
        let end = (*range.end() + 1).min(self.vertex_indices.len());
        if start >= self.vertex_indices.len() {
            return &[];
        }
        &self.vertex_indices[start..end]
    }

    /// Compute the centroid of all landmarks in `group`.
    ///
    /// # Errors
    ///
    /// Returns an error if any vertex index is out of bounds, or if the group
    /// produces zero landmarks (extractor has fewer indices than the group requires).
    pub fn group_centroid(
        &self,
        mesh: &Mesh,
        group: LandmarkGroup,
    ) -> Result<[f32; 3], FlameError> {
        let members = self.extract_group(mesh, group)?;
        if members.is_empty() {
            return Err(FlameError::landmark(
                0,
                format!("group '{}' produced no landmarks", group.name()),
            ));
        }
        let n = members.len() as f32;
        let mut cx = 0.0_f32;
        let mut cy = 0.0_f32;
        let mut cz = 0.0_f32;
        for lm in &members {
            cx += lm.position[0];
            cy += lm.position[1];
            cz += lm.position[2];
        }
        Ok([cx / n, cy / n, cz / n])
    }
}

impl Default for LandmarkExtractor {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Mesh extension
// ---------------------------------------------------------------------------

impl Mesh {
    /// Extract 68 facial landmarks using canonical FLAME vertex indices.
    ///
    /// Convenience wrapper around `LandmarkExtractor::new().extract(self)`.
    ///
    /// # Errors
    ///
    /// Returns [`FlameError::IndexOutOfBounds`] if the mesh has fewer than
    /// the maximum vertex index required by the canonical landmark table.
    pub fn extract_landmarks(&self) -> Result<Vec<Landmark>, FlameError> {
        LandmarkExtractor::new().extract(self)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra as na;

    /// Minimum vertex count required to satisfy all canonical landmark indices.
    const MIN_VERTS: usize = 5023;

    /// Build a synthetic mesh with `n` vertices, all positioned on the unit sphere
    /// surface (deterministically) so bounding-box checks are straightforward.
    fn synthetic_mesh(n: usize) -> Mesh {
        let vertices: Vec<na::Point3<f32>> = (0..n)
            .map(|i| {
                let t = (i as f32) / (n.max(1) as f32) * std::f32::consts::TAU;
                na::Point3::new(t.cos(), t.sin(), (i as f32) * 0.001)
            })
            .collect();
        // Minimal face list to avoid degenerate normal computation
        let faces = if n >= 3 {
            (0..(n as u32 - 2)).map(|i| [i, i + 1, i + 2]).collect()
        } else {
            vec![]
        };
        Mesh::new(vertices, faces)
    }

    // -----------------------------------------------------------------------
    // LandmarkGroup::from_index
    // -----------------------------------------------------------------------

    #[test]
    fn test_from_index_jaw_line() {
        assert_eq!(LandmarkGroup::from_index(0), LandmarkGroup::JawLine);
        assert_eq!(LandmarkGroup::from_index(16), LandmarkGroup::JawLine);
    }

    #[test]
    fn test_from_index_left_eyebrow() {
        assert_eq!(LandmarkGroup::from_index(17), LandmarkGroup::LeftEyebrow);
        assert_eq!(LandmarkGroup::from_index(21), LandmarkGroup::LeftEyebrow);
    }

    #[test]
    fn test_from_index_right_eyebrow() {
        assert_eq!(LandmarkGroup::from_index(22), LandmarkGroup::RightEyebrow);
        assert_eq!(LandmarkGroup::from_index(26), LandmarkGroup::RightEyebrow);
    }

    #[test]
    fn test_from_index_nose() {
        assert_eq!(LandmarkGroup::from_index(27), LandmarkGroup::Nose);
        assert_eq!(LandmarkGroup::from_index(35), LandmarkGroup::Nose);
    }

    #[test]
    fn test_from_index_left_eye() {
        assert_eq!(LandmarkGroup::from_index(36), LandmarkGroup::LeftEye);
        assert_eq!(LandmarkGroup::from_index(41), LandmarkGroup::LeftEye);
    }

    #[test]
    fn test_from_index_right_eye() {
        assert_eq!(LandmarkGroup::from_index(42), LandmarkGroup::RightEye);
        assert_eq!(LandmarkGroup::from_index(47), LandmarkGroup::RightEye);
    }

    #[test]
    fn test_from_index_outer_lip() {
        assert_eq!(LandmarkGroup::from_index(48), LandmarkGroup::OuterLip);
        assert_eq!(LandmarkGroup::from_index(59), LandmarkGroup::OuterLip);
    }

    #[test]
    fn test_from_index_inner_lip() {
        assert_eq!(LandmarkGroup::from_index(60), LandmarkGroup::InnerLip);
        assert_eq!(LandmarkGroup::from_index(67), LandmarkGroup::InnerLip);
    }

    #[test]
    fn test_from_index_out_of_range_falls_back_to_jaw() {
        assert_eq!(LandmarkGroup::from_index(68), LandmarkGroup::JawLine);
        assert_eq!(LandmarkGroup::from_index(999), LandmarkGroup::JawLine);
    }

    // -----------------------------------------------------------------------
    // LandmarkGroup::count / name
    // -----------------------------------------------------------------------

    #[test]
    fn test_group_counts() {
        assert_eq!(LandmarkGroup::JawLine.count(), 17);
        assert_eq!(LandmarkGroup::LeftEyebrow.count(), 5);
        assert_eq!(LandmarkGroup::RightEyebrow.count(), 5);
        assert_eq!(LandmarkGroup::Nose.count(), 9);
        assert_eq!(LandmarkGroup::LeftEye.count(), 6);
        assert_eq!(LandmarkGroup::RightEye.count(), 6);
        assert_eq!(LandmarkGroup::OuterLip.count(), 12);
        assert_eq!(LandmarkGroup::InnerLip.count(), 8);

        // Sum must equal 68
        let total = 17 + 5 + 5 + 9 + 6 + 6 + 12 + 8;
        assert_eq!(total, NUM_LANDMARKS);
    }

    #[test]
    fn test_group_names_are_non_empty() {
        let groups = [
            LandmarkGroup::JawLine,
            LandmarkGroup::LeftEyebrow,
            LandmarkGroup::RightEyebrow,
            LandmarkGroup::Nose,
            LandmarkGroup::LeftEye,
            LandmarkGroup::RightEye,
            LandmarkGroup::OuterLip,
            LandmarkGroup::InnerLip,
        ];
        for g in &groups {
            assert!(!g.name().is_empty(), "name for {g:?} is empty");
        }
    }

    // -----------------------------------------------------------------------
    // LandmarkExtractor::new / num_landmarks
    // -----------------------------------------------------------------------

    #[test]
    fn test_new_produces_68_landmarks() {
        let extractor = LandmarkExtractor::new();
        assert_eq!(extractor.num_landmarks(), 68);
    }

    #[test]
    fn test_default_produces_68_landmarks() {
        let extractor = LandmarkExtractor::default();
        assert_eq!(extractor.num_landmarks(), 68);
    }

    // -----------------------------------------------------------------------
    // LandmarkExtractor::with_indices
    // -----------------------------------------------------------------------

    #[test]
    fn test_with_indices_rejects_empty() {
        let result = LandmarkExtractor::with_indices(vec![]);
        assert!(result.is_err(), "empty index list should produce an error");
    }

    #[test]
    fn test_with_indices_accepts_custom() {
        let indices = vec![0u32, 1, 2, 3];
        let extractor =
            LandmarkExtractor::with_indices(indices).expect("non-empty index list should succeed");
        assert_eq!(extractor.num_landmarks(), 4);
    }

    // -----------------------------------------------------------------------
    // LandmarkExtractor::extract
    // -----------------------------------------------------------------------

    #[test]
    fn test_extract_returns_68_landmarks_on_large_mesh() {
        let mesh = synthetic_mesh(MIN_VERTS);
        let extractor = LandmarkExtractor::new();
        let landmarks = extractor
            .extract(&mesh)
            .expect("should succeed on large mesh");
        assert_eq!(landmarks.len(), 68);
    }

    #[test]
    fn test_extract_fails_on_small_mesh() {
        // Mesh with 100 vertices cannot satisfy indices up to ~3800.
        let mesh = synthetic_mesh(100);
        let extractor = LandmarkExtractor::new();
        let result = extractor.extract(&mesh);
        assert!(result.is_err(), "should fail when vertices < max index");
    }

    #[test]
    fn test_extract_landmark_group_matches_from_index() {
        let mesh = synthetic_mesh(MIN_VERTS);
        let extractor = LandmarkExtractor::new();
        let landmarks = extractor.extract(&mesh).expect("should succeed");
        for lm in &landmarks {
            assert_eq!(
                lm.group,
                LandmarkGroup::from_index(lm.index),
                "landmark {} group mismatch",
                lm.index
            );
        }
    }

    #[test]
    fn test_extract_indices_are_sequential() {
        let mesh = synthetic_mesh(MIN_VERTS);
        let extractor = LandmarkExtractor::new();
        let landmarks = extractor.extract(&mesh).expect("should succeed");
        for (i, lm) in landmarks.iter().enumerate() {
            assert_eq!(lm.index, i, "landmark indices must be sequential");
        }
    }

    // -----------------------------------------------------------------------
    // LandmarkExtractor::extract_group
    // -----------------------------------------------------------------------

    #[test]
    fn test_extract_group_nose_returns_9() {
        let mesh = synthetic_mesh(MIN_VERTS);
        let extractor = LandmarkExtractor::new();
        let nose = extractor
            .extract_group(&mesh, LandmarkGroup::Nose)
            .expect("should succeed");
        assert_eq!(nose.len(), 9, "nose group must have 9 landmarks");
    }

    #[test]
    fn test_extract_group_jaw_returns_17() {
        let mesh = synthetic_mesh(MIN_VERTS);
        let extractor = LandmarkExtractor::new();
        let jaw = extractor
            .extract_group(&mesh, LandmarkGroup::JawLine)
            .expect("should succeed");
        assert_eq!(jaw.len(), 17, "jaw line group must have 17 landmarks");
    }

    #[test]
    fn test_extract_group_outer_lip_returns_12() {
        let mesh = synthetic_mesh(MIN_VERTS);
        let extractor = LandmarkExtractor::new();
        let lip = extractor
            .extract_group(&mesh, LandmarkGroup::OuterLip)
            .expect("should succeed");
        assert_eq!(lip.len(), 12);
    }

    #[test]
    fn test_extract_group_inner_lip_returns_8() {
        let mesh = synthetic_mesh(MIN_VERTS);
        let extractor = LandmarkExtractor::new();
        let lip = extractor
            .extract_group(&mesh, LandmarkGroup::InnerLip)
            .expect("should succeed");
        assert_eq!(lip.len(), 8);
    }

    #[test]
    fn test_extract_group_all_have_correct_group_field() {
        let mesh = synthetic_mesh(MIN_VERTS);
        let extractor = LandmarkExtractor::new();
        let groups = [
            LandmarkGroup::JawLine,
            LandmarkGroup::LeftEyebrow,
            LandmarkGroup::RightEyebrow,
            LandmarkGroup::Nose,
            LandmarkGroup::LeftEye,
            LandmarkGroup::RightEye,
            LandmarkGroup::OuterLip,
            LandmarkGroup::InnerLip,
        ];
        for group in &groups {
            let members = extractor
                .extract_group(&mesh, *group)
                .expect("should succeed");
            for lm in &members {
                assert_eq!(
                    lm.group, *group,
                    "landmark {}: expected group {:?}, got {:?}",
                    lm.index, group, lm.group
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // LandmarkExtractor::group_indices
    // -----------------------------------------------------------------------

    #[test]
    fn test_group_indices_jaw_has_17_elements() {
        let extractor = LandmarkExtractor::new();
        let slice = extractor.group_indices(LandmarkGroup::JawLine);
        assert_eq!(slice.len(), 17);
    }

    #[test]
    fn test_group_indices_nose_has_9_elements() {
        let extractor = LandmarkExtractor::new();
        let slice = extractor.group_indices(LandmarkGroup::Nose);
        assert_eq!(slice.len(), 9);
    }

    // -----------------------------------------------------------------------
    // LandmarkExtractor::group_centroid
    // -----------------------------------------------------------------------

    #[test]
    fn test_group_centroid_is_within_bounding_box() {
        let mesh = synthetic_mesh(MIN_VERTS);
        let extractor = LandmarkExtractor::new();

        // Compute the bounding box of all vertices.
        let mut min_x = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut min_y = f32::INFINITY;
        let mut max_y = f32::NEG_INFINITY;
        let mut min_z = f32::INFINITY;
        let mut max_z = f32::NEG_INFINITY;
        for v in &mesh.vertices {
            min_x = min_x.min(v.x);
            max_x = max_x.max(v.x);
            min_y = min_y.min(v.y);
            max_y = max_y.max(v.y);
            min_z = min_z.min(v.z);
            max_z = max_z.max(v.z);
        }

        let centroid = extractor
            .group_centroid(&mesh, LandmarkGroup::Nose)
            .expect("should succeed");

        assert!(
            centroid[0] >= min_x && centroid[0] <= max_x,
            "centroid x={} out of bounding box [{}, {}]",
            centroid[0],
            min_x,
            max_x
        );
        assert!(
            centroid[1] >= min_y && centroid[1] <= max_y,
            "centroid y={} out of bounding box [{}, {}]",
            centroid[1],
            min_y,
            max_y
        );
        assert!(
            centroid[2] >= min_z && centroid[2] <= max_z,
            "centroid z={} out of bounding box [{}, {}]",
            centroid[2],
            min_z,
            max_z
        );
    }

    #[test]
    fn test_group_centroid_fails_on_small_mesh() {
        let mesh = synthetic_mesh(100);
        let extractor = LandmarkExtractor::new();
        let result = extractor.group_centroid(&mesh, LandmarkGroup::Nose);
        assert!(result.is_err(), "should fail when mesh too small");
    }

    // -----------------------------------------------------------------------
    // Mesh::extract_landmarks
    // -----------------------------------------------------------------------

    #[test]
    fn test_mesh_extract_landmarks_returns_68() {
        let mesh = synthetic_mesh(MIN_VERTS);
        let landmarks = mesh.extract_landmarks().expect("should succeed");
        assert_eq!(landmarks.len(), 68);
    }

    #[test]
    fn test_mesh_extract_landmarks_fails_on_tiny_mesh() {
        let mesh = synthetic_mesh(10);
        let result = mesh.extract_landmarks();
        assert!(result.is_err());
    }

    #[test]
    fn test_landmark_position_is_finite() {
        let mesh = synthetic_mesh(MIN_VERTS);
        let landmarks = mesh.extract_landmarks().expect("should succeed");
        for lm in &landmarks {
            assert!(
                lm.position.iter().all(|c| c.is_finite()),
                "landmark {} has non-finite position: {:?}",
                lm.index,
                lm.position
            );
        }
    }

    // -----------------------------------------------------------------------
    // Cross-check: group field vs from_index
    // -----------------------------------------------------------------------

    #[test]
    fn test_all_landmark_group_fields_match_from_index() {
        let mesh = synthetic_mesh(MIN_VERTS);
        let landmarks = mesh.extract_landmarks().expect("should succeed");
        for lm in &landmarks {
            assert_eq!(
                lm.group,
                LandmarkGroup::from_index(lm.index),
                "landmark {}: group field does not match from_index({})",
                lm.index,
                lm.index
            );
        }
    }
}
