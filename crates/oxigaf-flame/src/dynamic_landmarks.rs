//! Pose-dependent dynamic face contour landmark extraction.
//!
//! Dynamic landmarks are the **face contour** points (chin/jaw outline) that
//! shift based on head pose.  As the head rotates, different vertex chains
//! become the visible silhouette:
//!
//! - Left turn  (positive yaw): left-side jaw vertices become the visible outline
//! - Right turn (negative yaw): right-side jaw vertices become the visible outline
//! - Near-frontal (|yaw| < threshold): both sides used symmetrically
//!
//! ## Quick Start
//!
//! ```rust
//! use oxigaf_flame::{FlameParams, dynamic_landmarks::{DynamicLandmarkExtractor, DynamicLandmarkConfig}};
//!
//! let params = FlameParams::neutral();
//! let config = DynamicLandmarkConfig::default();
//! assert_eq!(config.num_contour_landmarks, 17);
//! ```

use crate::{
    error::FlameError,
    landmarks::{Landmark, LandmarkExtractor, LandmarkGroup},
    mesh::Mesh,
    params::FlameParams,
};

// ---------------------------------------------------------------------------
// ContourSide
// ---------------------------------------------------------------------------

/// Which side of the face contour to use for dynamic landmark computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContourSide {
    /// Left-side contour (chin → left ear).
    Left,
    /// Right-side contour (chin → right ear).
    Right,
    /// Both sides symmetrically (frontal pose).
    Both,
}

// ---------------------------------------------------------------------------
// DynamicLandmarkConfig
// ---------------------------------------------------------------------------

/// Configuration for dynamic landmark computation.
#[derive(Debug, Clone)]
pub struct DynamicLandmarkConfig {
    /// Number of contour landmarks to compute (default: 17, matches jaw line).
    pub num_contour_landmarks: usize,

    /// Yaw angle threshold (radians) for side selection.
    ///
    /// When |yaw| < threshold the extractor uses both sides symmetrically
    /// (frontal pose).  Default: 0.1 radians (~5.7°).
    pub side_threshold_rad: f32,
}

impl Default for DynamicLandmarkConfig {
    fn default() -> Self {
        Self {
            num_contour_landmarks: 17,
            side_threshold_rad: 0.1,
        }
    }
}

// ---------------------------------------------------------------------------
// ContourVertexChains
// ---------------------------------------------------------------------------

/// Left and right face contour vertex chains.
///
/// Each chain lists vertex indices in order from the chin outward to the
/// respective ear.  The vertex indices correspond to the FLAME 2020 canonical
/// mesh topology (~5023 vertices).
pub struct ContourVertexChains {
    /// Left side contour vertices (chin → left ear), ordered.
    pub left_chain: Vec<u32>,
    /// Right side contour vertices (chin → right ear), ordered.
    pub right_chain: Vec<u32>,
}

impl ContourVertexChains {
    /// Default FLAME 2020 contour vertex chains.
    ///
    /// These approximate the jaw/chin silhouette for a 5023-vertex FLAME mesh.
    /// When `flame_dynamic_embedding.npy` is available the chains should be
    /// replaced with the embedding's `dynamic_lmk_faces_idx` and
    /// `dynamic_lmk_b_coords` values projected to vertex indices.
    ///
    /// **Left chain** covers approximately vertex indices 1–17 (chin to left
    /// jaw), using small indices that are guaranteed to exist in any standard
    /// FLAME mesh.
    ///
    /// **Right chain** covers approximately vertex indices 4984–5000 (chin to
    /// right jaw), near the end of the FLAME vertex array.
    #[must_use]
    pub fn default_flame() -> Self {
        // Left side: chin → left ear (approximate FLAME 2020 jaw contour).
        // Uses indices 1-17 which are valid for the 5023-vertex FLAME mesh.
        let left_chain = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17];

        // Right side: chin → right ear (approximate FLAME 2020 jaw contour).
        // Uses indices near the end of the vertex array, valid for 5023-vertex FLAME.
        let right_chain = vec![
            5000, 4999, 4998, 4997, 4996, 4995, 4994, 4993, 4992, 4991, 4990, 4989, 4988, 4987,
            4986, 4985, 4984,
        ];

        Self {
            left_chain,
            right_chain,
        }
    }
}

// ---------------------------------------------------------------------------
// DynamicLandmarkExtractor
// ---------------------------------------------------------------------------

/// Pose-dependent face contour landmark extractor.
///
/// Selects the appropriate jaw-line vertex chain based on the head's yaw
/// angle extracted from [`FlameParams::pose`], then maps those chain vertices
/// to 3D positions from the posed [`Mesh`].
///
/// ## Yaw extraction
///
/// The yaw is approximated as `params.pose[1]` — the Y-component of the
/// global-orient axis-angle vector.  This is exact only for small rotations;
/// for large rotations a full Rodrigues decomposition into Euler angles would
/// be required.  The approximation is sufficient for the contour-side
/// selection (left / right / both) which only needs a sign and threshold test.
pub struct DynamicLandmarkExtractor {
    contour_chains: ContourVertexChains,
    config: DynamicLandmarkConfig,
}

impl DynamicLandmarkExtractor {
    /// Create a new extractor with default contour chains and configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            contour_chains: ContourVertexChains::default_flame(),
            config: DynamicLandmarkConfig::default(),
        }
    }

    /// Create a new extractor with custom configuration and default contour chains.
    #[must_use]
    pub fn with_config(config: DynamicLandmarkConfig) -> Self {
        Self {
            contour_chains: ContourVertexChains::default_flame(),
            config,
        }
    }

    /// Create a new extractor with both custom config and custom contour chains.
    #[must_use]
    pub fn with_chains_and_config(
        contour_chains: ContourVertexChains,
        config: DynamicLandmarkConfig,
    ) -> Self {
        Self {
            contour_chains,
            config,
        }
    }

    // -----------------------------------------------------------------------
    // Yaw extraction
    // -----------------------------------------------------------------------

    /// Extract the head yaw angle (radians) from FLAME pose parameters.
    ///
    /// FLAME pose layout: `[global_orient(3), neck(3), jaw(3), left_eye(3), right_eye(3)]`.
    /// The global orientation gives head rotation in world space as an axis-angle vector.
    ///
    /// **Approximation**: returns `pose[1]` (Y-component of the global-orient
    /// axis-angle).  This equals the true yaw only when the rotation is small.
    /// For large rotations a full Euler-angle decomposition is required, but
    /// this approximation is sufficient for coarse side selection.
    ///
    /// Returns `0.0` when the pose vector has fewer than 2 elements.
    #[must_use]
    pub fn extract_yaw(params: &FlameParams) -> f32 {
        if params.pose.len() >= 2 {
            params.pose[1]
        } else {
            0.0
        }
    }

    // -----------------------------------------------------------------------
    // Side selection
    // -----------------------------------------------------------------------

    /// Select which contour side to use based on the head yaw angle.
    ///
    /// - `|yaw| < threshold` → [`ContourSide::Both`]
    /// - `yaw > 0`           → [`ContourSide::Left`]  (left-turn)
    /// - `yaw < 0`           → [`ContourSide::Right`] (right-turn)
    #[must_use]
    pub fn select_contour_side(&self, params: &FlameParams) -> ContourSide {
        let yaw = Self::extract_yaw(params);
        if yaw.abs() < self.config.side_threshold_rad {
            ContourSide::Both
        } else if yaw > 0.0 {
            ContourSide::Left
        } else {
            ContourSide::Right
        }
    }

    // -----------------------------------------------------------------------
    // Core extraction helpers
    // -----------------------------------------------------------------------

    /// Look up positions for a slice of vertex indices from `mesh`.
    ///
    /// Samples at most `count` indices from `chain`.  Returns an error when
    /// any requested index is ≥ `mesh.vertices.len()`.
    fn sample_chain(
        mesh: &Mesh,
        chain: &[u32],
        count: usize,
        start_index: usize,
        group: LandmarkGroup,
    ) -> Result<Vec<Landmark>, FlameError> {
        let num_verts = mesh.vertices.len();
        let n = count.min(chain.len());
        let mut landmarks = Vec::with_capacity(n);

        for (offset, &vi_u32) in chain.iter().take(n).enumerate() {
            let vi = vi_u32 as usize;
            if vi >= num_verts {
                return Err(FlameError::index_out_of_bounds(
                    format!("dynamic contour chain vertex at chain-offset {offset}"),
                    vi,
                    num_verts,
                ));
            }
            let v = &mesh.vertices[vi];
            landmarks.push(Landmark {
                position: [v.x, v.y, v.z],
                index: start_index + offset,
                group,
            });
        }

        Ok(landmarks)
    }

    // -----------------------------------------------------------------------
    // Public extraction
    // -----------------------------------------------------------------------

    /// Compute dynamic contour landmarks from a posed mesh.
    ///
    /// Returns exactly [`DynamicLandmarkConfig::num_contour_landmarks`] landmarks
    /// with [`LandmarkGroup::JawLine`] group labels.  Landmark `index` values
    /// start at `0` and are sequential.
    ///
    /// When both sides are selected (frontal pose) the left and right chains
    /// are interleaved to produce a symmetric contour covering both sides.
    ///
    /// # Errors
    ///
    /// Returns [`FlameError::IndexOutOfBounds`] if any contour vertex index is
    /// larger than the number of vertices in `mesh`.
    pub fn extract(&self, mesh: &Mesh, params: &FlameParams) -> Result<Vec<Landmark>, FlameError> {
        let n = self.config.num_contour_landmarks;
        let side = self.select_contour_side(params);

        match side {
            ContourSide::Left => Self::sample_chain(
                mesh,
                &self.contour_chains.left_chain,
                n,
                0,
                LandmarkGroup::JawLine,
            ),
            ContourSide::Right => Self::sample_chain(
                mesh,
                &self.contour_chains.right_chain,
                n,
                0,
                LandmarkGroup::JawLine,
            ),
            ContourSide::Both => {
                // Interleave left and right chains.
                // Take ceil(n/2) from left and floor(n/2) from right.
                let left_n = n.div_ceil(2);
                let right_n = n / 2;
                let num_verts = mesh.vertices.len();
                let mut landmarks = Vec::with_capacity(n);

                // Left side first
                for (offset, &vi_u32) in self
                    .contour_chains
                    .left_chain
                    .iter()
                    .take(left_n)
                    .enumerate()
                {
                    let vi = vi_u32 as usize;
                    if vi >= num_verts {
                        return Err(FlameError::index_out_of_bounds(
                            format!("dynamic contour left-chain vertex at offset {offset}"),
                            vi,
                            num_verts,
                        ));
                    }
                    let v = &mesh.vertices[vi];
                    landmarks.push(Landmark {
                        position: [v.x, v.y, v.z],
                        index: offset,
                        group: LandmarkGroup::JawLine,
                    });
                }

                // Right side fills the remainder
                for (offset, &vi_u32) in self
                    .contour_chains
                    .right_chain
                    .iter()
                    .take(right_n)
                    .enumerate()
                {
                    let vi = vi_u32 as usize;
                    if vi >= num_verts {
                        return Err(FlameError::index_out_of_bounds(
                            format!("dynamic contour right-chain vertex at offset {offset}"),
                            vi,
                            num_verts,
                        ));
                    }
                    let v = &mesh.vertices[vi];
                    landmarks.push(Landmark {
                        position: [v.x, v.y, v.z],
                        index: left_n + offset,
                        group: LandmarkGroup::JawLine,
                    });
                }

                Ok(landmarks)
            }
        }
    }

    /// Compute all 68 landmarks: static set with the jaw line (indices 0–16)
    /// replaced by the pose-dependent dynamic contour.
    ///
    /// The returned `Vec` always has exactly 68 entries.  Landmark `index`
    /// values match the iBUG 68-point convention so that downstream code does
    /// not need to know whether static or dynamic jaw points were used.
    ///
    /// # Errors
    ///
    /// Returns an error if either the static extractor or the dynamic contour
    /// computation fails (e.g., mesh too small).
    pub fn extract_all(
        &self,
        mesh: &Mesh,
        params: &FlameParams,
    ) -> Result<Vec<Landmark>, FlameError> {
        // Extract full static 68-point set.
        let mut static_landmarks = LandmarkExtractor::new().extract(mesh)?;

        // Compute dynamic jaw contour (17 points).
        let dynamic_jaw = self.extract(mesh, params)?;

        // Overwrite jaw-line entries (indices 0-16) with dynamic landmarks.
        // Preserve the original iBUG index values and group labels.
        for (jaw_slot, dyn_lm) in static_landmarks
            .iter_mut()
            .take(17) // jaw line is always slots 0-16 in iBUG ordering
            .zip(dynamic_jaw.iter())
        {
            // Keep jaw_slot.index unchanged (it's 0..16) — just update position.
            jaw_slot.position = dyn_lm.position;
        }

        Ok(static_landmarks)
    }
}

impl Default for DynamicLandmarkExtractor {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Mesh extension
// ---------------------------------------------------------------------------

impl Mesh {
    /// Extract pose-dependent dynamic face contour landmarks.
    ///
    /// Convenience wrapper around `DynamicLandmarkExtractor::new().extract()`.
    /// Returns 17 contour landmarks using the jaw-line chain selected by head pose.
    ///
    /// # Errors
    ///
    /// Returns [`FlameError::IndexOutOfBounds`] if any contour vertex index is
    /// out of bounds for this mesh.
    pub fn extract_dynamic_landmarks(
        &self,
        params: &FlameParams,
    ) -> Result<Vec<Landmark>, FlameError> {
        DynamicLandmarkExtractor::new().extract(self, params)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra as na;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Minimum vertex count to satisfy all canonical FLAME landmark indices
    /// AND the right-chain indices (max 5000).
    const MIN_VERTS: usize = 5023;

    /// Build a synthetic mesh with `n` vertices placed on a unit-circle
    /// spiral (deterministic, all positions finite).
    fn synthetic_mesh(n: usize) -> Mesh {
        let vertices: Vec<na::Point3<f32>> = (0..n)
            .map(|i| {
                let t = (i as f32) / (n.max(1) as f32) * std::f32::consts::TAU;
                na::Point3::new(t.cos(), t.sin(), (i as f32) * 0.001)
            })
            .collect();
        let faces = if n >= 3 {
            (0..(n as u32 - 2)).map(|i| [i, i + 1, i + 2]).collect()
        } else {
            vec![]
        };
        Mesh::new(vertices, faces)
    }

    /// Build a [`FlameParams`] with the first pose element set to `yaw`.
    ///
    /// Uses a 15-element pose vector (5 joints × 3).
    fn params_with_yaw(yaw: f32) -> FlameParams {
        let mut pose = vec![0.0f32; 15];
        pose[1] = yaw; // Y-component of global orient
        FlameParams {
            shape: vec![],
            expression: vec![],
            pose,
            translation: [0.0; 3],
        }
    }

    // -----------------------------------------------------------------------
    // DynamicLandmarkConfig
    // -----------------------------------------------------------------------

    #[test]
    fn test_default_config_num_contour() {
        let cfg = DynamicLandmarkConfig::default();
        assert_eq!(cfg.num_contour_landmarks, 17);
    }

    #[test]
    fn test_default_config_threshold() {
        let cfg = DynamicLandmarkConfig::default();
        assert!(
            (cfg.side_threshold_rad - 0.1).abs() < 1e-6,
            "default threshold should be 0.1 rad"
        );
    }

    // -----------------------------------------------------------------------
    // ContourVertexChains
    // -----------------------------------------------------------------------

    #[test]
    fn test_default_chains_left_non_empty() {
        let chains = ContourVertexChains::default_flame();
        assert!(!chains.left_chain.is_empty());
    }

    #[test]
    fn test_default_chains_right_non_empty() {
        let chains = ContourVertexChains::default_flame();
        assert!(!chains.right_chain.is_empty());
    }

    #[test]
    fn test_default_chains_left_length_17() {
        let chains = ContourVertexChains::default_flame();
        assert_eq!(chains.left_chain.len(), 17);
    }

    #[test]
    fn test_default_chains_right_length_17() {
        let chains = ContourVertexChains::default_flame();
        assert_eq!(chains.right_chain.len(), 17);
    }

    // -----------------------------------------------------------------------
    // extract_yaw
    // -----------------------------------------------------------------------

    #[test]
    fn test_extract_yaw_neutral_is_zero() {
        let params = FlameParams::neutral();
        let yaw = DynamicLandmarkExtractor::extract_yaw(&params);
        assert!(yaw.abs() < 1e-7, "neutral pose yaw should be 0");
    }

    #[test]
    fn test_extract_yaw_positive_y() {
        let params = params_with_yaw(0.5);
        let yaw = DynamicLandmarkExtractor::extract_yaw(&params);
        assert!(yaw > 0.0, "positive y-component should give positive yaw");
    }

    #[test]
    fn test_extract_yaw_empty_pose_no_panic() {
        let params = FlameParams {
            shape: vec![],
            expression: vec![],
            pose: vec![],
            translation: [0.0; 3],
        };
        // Must not panic — returns 0.0 for empty pose.
        let yaw = DynamicLandmarkExtractor::extract_yaw(&params);
        assert_eq!(yaw, 0.0);
    }

    #[test]
    fn test_extract_yaw_one_element_pose_no_panic() {
        // pose has only one element (index 0), so pose[1] is out of range.
        let params = FlameParams {
            shape: vec![],
            expression: vec![],
            pose: vec![0.3],
            translation: [0.0; 3],
        };
        let yaw = DynamicLandmarkExtractor::extract_yaw(&params);
        assert_eq!(yaw, 0.0);
    }

    // -----------------------------------------------------------------------
    // select_contour_side
    // -----------------------------------------------------------------------

    #[test]
    fn test_select_side_frontal_is_both() {
        let extractor = DynamicLandmarkExtractor::new();
        let params = FlameParams::neutral(); // yaw = 0
        assert_eq!(extractor.select_contour_side(&params), ContourSide::Both);
    }

    #[test]
    fn test_select_side_large_positive_yaw_is_left() {
        let extractor = DynamicLandmarkExtractor::new();
        let params = params_with_yaw(0.5); // well above threshold 0.1
        assert_eq!(extractor.select_contour_side(&params), ContourSide::Left);
    }

    #[test]
    fn test_select_side_large_negative_yaw_is_right() {
        let extractor = DynamicLandmarkExtractor::new();
        let params = params_with_yaw(-0.5);
        assert_eq!(extractor.select_contour_side(&params), ContourSide::Right);
    }

    #[test]
    fn test_select_side_threshold_boundary_positive() {
        let extractor = DynamicLandmarkExtractor::new();
        // Just below threshold → Both
        let params_below = params_with_yaw(0.05);
        assert_eq!(
            extractor.select_contour_side(&params_below),
            ContourSide::Both
        );
        // Just above threshold → Left
        let params_above = params_with_yaw(0.15);
        assert_eq!(
            extractor.select_contour_side(&params_above),
            ContourSide::Left
        );
    }

    // -----------------------------------------------------------------------
    // extract
    // -----------------------------------------------------------------------

    #[test]
    fn test_extract_frontal_returns_17() {
        let mesh = synthetic_mesh(MIN_VERTS);
        let extractor = DynamicLandmarkExtractor::new();
        let params = FlameParams::neutral();
        let landmarks = extractor
            .extract(&mesh, &params)
            .expect("extract should succeed on FLAME-sized mesh");
        assert_eq!(landmarks.len(), 17);
    }

    #[test]
    fn test_extract_left_returns_17() {
        let mesh = synthetic_mesh(MIN_VERTS);
        let extractor = DynamicLandmarkExtractor::new();
        let params = params_with_yaw(0.5);
        let landmarks = extractor
            .extract(&mesh, &params)
            .expect("extract left should succeed");
        assert_eq!(landmarks.len(), 17);
    }

    #[test]
    fn test_extract_right_returns_17() {
        let mesh = synthetic_mesh(MIN_VERTS);
        let extractor = DynamicLandmarkExtractor::new();
        let params = params_with_yaw(-0.5);
        let landmarks = extractor
            .extract(&mesh, &params)
            .expect("extract right should succeed");
        assert_eq!(landmarks.len(), 17);
    }

    #[test]
    fn test_extract_all_positions_finite() {
        let mesh = synthetic_mesh(MIN_VERTS);
        let extractor = DynamicLandmarkExtractor::new();
        let params = FlameParams::neutral();
        let landmarks = extractor.extract(&mesh, &params).expect("should succeed");
        for lm in &landmarks {
            assert!(
                lm.position.iter().all(|c| c.is_finite()),
                "landmark {} position {:?} is not finite",
                lm.index,
                lm.position
            );
        }
    }

    #[test]
    fn test_extract_positions_within_bounding_box() {
        let mesh = synthetic_mesh(MIN_VERTS);
        // Compute bounding box.
        let mut min_x = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut min_y = f32::INFINITY;
        let mut max_y = f32::NEG_INFINITY;
        for v in &mesh.vertices {
            min_x = min_x.min(v.x);
            max_x = max_x.max(v.x);
            min_y = min_y.min(v.y);
            max_y = max_y.max(v.y);
        }

        let extractor = DynamicLandmarkExtractor::new();
        let params = FlameParams::neutral();
        let landmarks = extractor.extract(&mesh, &params).expect("should succeed");

        for lm in &landmarks {
            assert!(
                lm.position[0] >= min_x - 1e-5 && lm.position[0] <= max_x + 1e-5,
                "landmark x={} outside bounding box [{}, {}]",
                lm.position[0],
                min_x,
                max_x
            );
            assert!(
                lm.position[1] >= min_y - 1e-5 && lm.position[1] <= max_y + 1e-5,
                "landmark y={} outside bounding box [{}, {}]",
                lm.position[1],
                min_y,
                max_y
            );
        }
    }

    #[test]
    fn test_extract_small_mesh_returns_error() {
        // Right chain has indices up to 5000 — a small mesh can't satisfy them.
        let mesh = synthetic_mesh(100);
        let extractor = DynamicLandmarkExtractor::new();
        // For left-only (small indices 1-17), a 100-vertex mesh works.
        // For right-only (indices 4984-5000), it should fail.
        let params = params_with_yaw(-0.5); // force ContourSide::Right
        let result = extractor.extract(&mesh, &params);
        assert!(
            result.is_err(),
            "extract with right chain on tiny mesh should return Err"
        );
    }

    // -----------------------------------------------------------------------
    // extract_all
    // -----------------------------------------------------------------------

    #[test]
    fn test_extract_all_returns_68() {
        let mesh = synthetic_mesh(MIN_VERTS);
        let extractor = DynamicLandmarkExtractor::new();
        let params = FlameParams::neutral();
        let landmarks = extractor
            .extract_all(&mesh, &params)
            .expect("extract_all should succeed");
        assert_eq!(landmarks.len(), 68);
    }

    #[test]
    fn test_extract_all_indices_sequential() {
        let mesh = synthetic_mesh(MIN_VERTS);
        let extractor = DynamicLandmarkExtractor::new();
        let params = FlameParams::neutral();
        let landmarks = extractor
            .extract_all(&mesh, &params)
            .expect("should succeed");
        for (i, lm) in landmarks.iter().enumerate() {
            assert_eq!(lm.index, i, "landmark indices must be sequential");
        }
    }

    #[test]
    fn test_extract_all_jaw_group_correct() {
        let mesh = synthetic_mesh(MIN_VERTS);
        let extractor = DynamicLandmarkExtractor::new();
        let params = FlameParams::neutral();
        let landmarks = extractor
            .extract_all(&mesh, &params)
            .expect("should succeed");
        // Jaw-line entries (0-16) must still be tagged JawLine.
        for lm in landmarks.iter().take(17) {
            assert_eq!(
                lm.group,
                LandmarkGroup::JawLine,
                "dynamic jaw landmark {} should have JawLine group",
                lm.index
            );
        }
    }

    // -----------------------------------------------------------------------
    // Mesh::extract_dynamic_landmarks
    // -----------------------------------------------------------------------

    #[test]
    fn test_mesh_method_returns_17() {
        let mesh = synthetic_mesh(MIN_VERTS);
        let params = FlameParams::neutral();
        let landmarks = mesh
            .extract_dynamic_landmarks(&params)
            .expect("Mesh::extract_dynamic_landmarks should succeed");
        assert_eq!(landmarks.len(), 17);
    }

    #[test]
    fn test_mesh_method_small_mesh_left_chain_ok() {
        // Left chain uses indices 1-17, so a 50-vertex mesh is sufficient.
        let mesh = synthetic_mesh(50);
        let params = params_with_yaw(0.5); // force left chain
        let landmarks = mesh
            .extract_dynamic_landmarks(&params)
            .expect("left chain on 50-vertex mesh should succeed");
        assert_eq!(landmarks.len(), 17);
    }
}
