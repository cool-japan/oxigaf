//! Geometric vertex mask (face region segmentation) for FLAME meshes.
//!
//! FLAME meshes have semantic face regions (eyes, mouth, scalp, etc.).
//! Since loading the original FLAME masks requires Python, this module
//! implements a geometric approximation based on vertex positions in the
//! FLAME canonical pose (right-handed, +X right, +Y up, +Z forward).
//!
//! ## Coordinate System
//!
//! All threshold values are expressed in FLAME canonical-pose metric units:
//! - `y` axis: up (+) → scalp, down (−) → neck
//! - `x` axis: left−right (from subject perspective)
//! - `z` axis: forward (+) → nose/lips protrude outward

use std::collections::HashMap;

use nalgebra as na;

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
/// Eye region: z minimum (eyes protrude forward).
const EYE_Z_MIN: f32 = 0.06;

/// Mouth region: y lower bound.
const MOUTH_Y_MIN: f32 = -0.10;
/// Mouth region: y upper bound.
const MOUTH_Y_MAX: f32 = 0.00;
/// Mouth region: z minimum (lips protrude forward).
const MOUTH_Z_MIN: f32 = 0.06;

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
    /// Compute vertex masks geometrically from vertex positions.
    ///
    /// Uses approximate y/z/x coordinate thresholds for FLAME's canonical pose.
    /// Classification order (highest priority first):
    ///
    /// 1. **Scalp**: `y > SCALP_Y_MIN`
    /// 2. **Neck**: `y < NECK_Y_MAX`
    /// 3. **Ears**: `|x| > EAR_X_MIN` with `y ∈ [EAR_Y_MIN, EAR_Y_MAX]`
    ///    - Left ear: `x > 0`
    ///    - Right ear: `x < 0`
    /// 4. **Eyes**: `y ∈ [EYE_Y_MIN, EYE_Y_MAX]`, `|x| < EYE_X_MAX`, `z > EYE_Z_MIN`
    ///    - Left eye: `x > 0` (positive x is the subject's left side)
    ///    - Right eye: `x <= 0`
    /// 5. **Mouth**: `y ∈ [MOUTH_Y_MIN, MOUTH_Y_MAX]`, `z > MOUTH_Z_MIN`
    /// 6. **Face**: everything else
    #[must_use]
    pub fn from_vertices(vertices: &[na::Point3<f32>]) -> Self {
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
        let all_regions = [
            FaceRegion::Face,
            FaceRegion::LeftEye,
            FaceRegion::RightEye,
            FaceRegion::Mouth,
            FaceRegion::Neck,
            FaceRegion::LeftEar,
            FaceRegion::RightEar,
            FaceRegion::Scalp,
        ];

        let mut counts: HashMap<String, usize> = all_regions
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
// Private classification helper
// ---------------------------------------------------------------------------

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

    // 4. Eyes: upper face, close to the centre, protruding forward
    if (EYE_Y_MIN..=EYE_Y_MAX).contains(&y) && x.abs() < EYE_X_MAX && z > EYE_Z_MIN {
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
}
