//! Pose-dependent dynamic face contour landmark extraction.
//!
//! Dynamic landmarks are the **face contour** points (chin/jaw outline) that
//! shift based on head pose.  As the head rotates, one side of the jaw recedes
//! behind the cheek and its contour has to slide inward, so a different vertex
//! chain traces the silhouette.
//!
//! ## Yaw sign convention
//!
//! This crate uses a right-handed frame with `+X` = the subject's right,
//! `+Y` = up and `+Z` = out of the face (see the crate-level docs), so a
//! positive rotation about `+Y` swings the face toward `+X`: **positive yaw is
//! a turn to the subject's right**.  That matches
//! [`crate::canonical::HeadOrientation::yaw`], the `head_turn` rig control
//! (`+1 = turn right`) and [`crate::pose_estimation::estimate_yaw_from_symmetry`].
//!
//! Under a right turn the subject's right ear rotates away from a camera on
//! `+Z`, so the right side is the one that recedes and needs the pose-dependent
//! contour:
//!
//! - Right turn (positive yaw): the right-side chain traces the contour
//! - Left turn  (negative yaw): the left-side chain traces the contour
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
///
/// The two chains are **disjoint**: the chin vertex belongs to `left_chain`
/// only, so `left_chain` followed by `right_chain` walks the whole jaw contour
/// without repeating a point.  [`DynamicLandmarkExtractor::extract`] relies on
/// that when it builds a symmetric frontal contour from both chains.
#[derive(Debug, Clone)]
pub struct ContourVertexChains {
    /// Left side contour vertices (chin → left ear), ordered.
    pub left_chain: Vec<u32>,
    /// Right side contour vertices (just past the chin → right ear), ordered.
    pub right_chain: Vec<u32>,
}

impl ContourVertexChains {
    /// Contour chains derived from the canonical FLAME 68-point jaw line.
    ///
    /// The two chains are the two halves of the jaw contour that
    /// [`LandmarkExtractor`] already models — real jaw vertices, split at the
    /// chin — rather than runs of consecutive vertex indices, which are
    /// unrelated to one another in the FLAME topology.
    ///
    /// # This is a pose-*independent* fallback
    ///
    /// A genuinely pose-dependent contour needs FLAME's dynamic landmark
    /// embedding (`flame_dynamic_embedding.npy`), which is a licensed model
    /// asset this crate does not ship.  Without it the contour vertex *set*
    /// cannot change with head pose, so an extractor built on these chains
    /// reports [`DynamicLandmarkExtractor::is_pose_dependent`] `== false` and
    /// [`DynamicLandmarkExtractor::extract_all`] deliberately leaves the static
    /// 68-point set alone instead of overwriting its jaw slots with a contour
    /// that carries no extra information.  Load the real embedding with
    /// [`ContourVertexChains::from_dynamic_embedding`] to get pose-dependent
    /// behaviour.
    ///
    /// # Side assignment
    ///
    /// Which end of the jaw table is the subject's left is inherited from
    /// [`landmarks`](crate::landmarks): that module labels the low-numbered
    /// iBUG ranges (17–21, 36–41) as the *left* eyebrow and eye, so the
    /// low-numbered end of the jaw line (slot 0) is taken to be the subject's
    /// left and slot 16 the subject's right.  Correcting the landmark table
    /// therefore corrects this too, in one place.
    #[must_use]
    pub fn default_flame() -> Self {
        // Bind the extractor: `group_indices` borrows from it.
        let extractor = LandmarkExtractor::new();
        Self::from_jaw_contour(extractor.group_indices(LandmarkGroup::JawLine))
    }

    /// Split an ordered jaw contour into a left and a right chain.
    ///
    /// `jaw` is expected in iBUG jaw order, walking continuously from one ear
    /// to the other; the midpoint entry is the chin.  The chin is assigned to
    /// `left_chain`, keeping the two chains disjoint.
    ///
    /// An empty `jaw` yields two empty chains rather than an error — the
    /// extraction functions surface that as an empty landmark list.
    #[must_use]
    pub fn from_jaw_contour(jaw: &[u32]) -> Self {
        if jaw.is_empty() {
            return Self {
                left_chain: Vec::new(),
                right_chain: Vec::new(),
            };
        }
        // `chin <= jaw.len() - 1` for every non-empty slice, so both slices below
        // are in range.
        let chin = jaw.len() / 2;
        // Left chain runs chin-outward, so the first half is walked in reverse.
        let left_chain: Vec<u32> = jaw[..=chin].iter().rev().copied().collect();
        let right_chain: Vec<u32> = jaw[chin + 1..].to_vec();
        Self {
            left_chain,
            right_chain,
        }
    }

    /// Build contour chains from FLAME's dynamic landmark embedding.
    ///
    /// `flame_dynamic_embedding.npy` stores, for each of `n_bins` quantised head
    /// yaw angles, `n_contour` contour landmarks as a triangle index
    /// (`dynamic_lmk_faces_idx`, shape `[n_bins, n_contour]`) plus barycentric
    /// weights within that triangle (`dynamic_lmk_b_coords`, shape
    /// `[n_bins, n_contour, 3]`).  Both are passed here flattened in row-major
    /// order; `n_bins` is inferred from `lmk_faces_idx.len() / n_contour`.
    ///
    /// A barycentric point generally lies *inside* a triangle, while a chain can
    /// only name vertices, so each contour point is collapsed to the triangle
    /// corner carrying the largest barycentric weight — the nearest true vertex.
    ///
    /// `left_bin` and `right_bin` select the two yaw rows to keep.  They are
    /// explicit parameters because the row ordering (which end of the table is
    /// maximum left yaw) is a property of the file being loaded and cannot be
    /// inferred from the data.
    ///
    /// # Errors
    ///
    /// - [`FlameError::InvalidParams`] if `n_contour` is zero, if
    ///   `lmk_faces_idx` is empty or not a whole number of bins, or if
    ///   `lmk_b_coords` does not hold `n_bins * n_contour * 3` weights.
    /// - [`FlameError::IndexOutOfBounds`] if a bin index is beyond `n_bins`, or
    ///   if a stored triangle index is beyond `faces`.
    pub fn from_dynamic_embedding(
        lmk_faces_idx: &[u32],
        lmk_b_coords: &[f32],
        faces: &[[u32; 3]],
        n_contour: usize,
        left_bin: usize,
        right_bin: usize,
    ) -> Result<Self, FlameError> {
        if n_contour == 0 {
            return Err(FlameError::InvalidParams(
                "dynamic landmark embedding: n_contour must be > 0".to_string(),
            ));
        }
        if lmk_faces_idx.is_empty() || !lmk_faces_idx.len().is_multiple_of(n_contour) {
            return Err(FlameError::InvalidParams(format!(
                "dynamic landmark embedding: dynamic_lmk_faces_idx has {} entries, which is not a positive multiple of n_contour {n_contour}",
                lmk_faces_idx.len()
            )));
        }
        let n_bins = lmk_faces_idx.len() / n_contour;
        let expected_coords = n_bins * n_contour * 3;
        if lmk_b_coords.len() != expected_coords {
            return Err(FlameError::InvalidParams(format!(
                "dynamic landmark embedding: dynamic_lmk_b_coords has {} entries, expected {expected_coords}",
                lmk_b_coords.len()
            )));
        }
        for bin in [left_bin, right_bin] {
            if bin >= n_bins {
                return Err(FlameError::index_out_of_bounds(
                    "dynamic landmark yaw bin",
                    bin,
                    n_bins,
                ));
            }
        }
        Ok(Self {
            left_chain: chain_from_yaw_bin(
                lmk_faces_idx,
                lmk_b_coords,
                faces,
                n_contour,
                left_bin,
            )?,
            right_chain: chain_from_yaw_bin(
                lmk_faces_idx,
                lmk_b_coords,
                faces,
                n_contour,
                right_bin,
            )?,
        })
    }
}

/// Collapse one yaw row of a dynamic landmark embedding to a vertex chain.
///
/// Callers must have validated `n_contour`, the array lengths and `bin`; every
/// index computed here is then in range except the stored triangle index, which
/// is checked against `faces`.
fn chain_from_yaw_bin(
    lmk_faces_idx: &[u32],
    lmk_b_coords: &[f32],
    faces: &[[u32; 3]],
    n_contour: usize,
    bin: usize,
) -> Result<Vec<u32>, FlameError> {
    let mut chain = Vec::with_capacity(n_contour);
    for point in 0..n_contour {
        let flat = bin * n_contour + point;
        let face_index = lmk_faces_idx[flat] as usize;
        let face = faces.get(face_index).ok_or_else(|| {
            FlameError::index_out_of_bounds(
                format!("dynamic landmark triangle for contour point {point}"),
                face_index,
                faces.len(),
            )
        })?;
        let base = flat * 3;
        let weights = [
            lmk_b_coords[base],
            lmk_b_coords[base + 1],
            lmk_b_coords[base + 2],
        ];
        // Nearest true vertex = the corner with the largest barycentric weight.
        let mut best = 0_usize;
        for (corner, &weight) in weights.iter().enumerate().skip(1) {
            if weight > weights[best] {
                best = corner;
            }
        }
        chain.push(face[best]);
    }
    Ok(chain)
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
    /// Whether `contour_chains` genuinely varies with head pose.
    ///
    /// `false` for the [`ContourVertexChains::default_flame`] fallback, which is
    /// the static jaw contour split in two; `true` for caller-supplied chains.
    pose_dependent: bool,
}

impl DynamicLandmarkExtractor {
    /// Create a new extractor with the fallback contour chains and default
    /// configuration.
    ///
    /// The chains come from [`ContourVertexChains::default_flame`], so the
    /// extractor is **not** pose-dependent — see
    /// [`DynamicLandmarkExtractor::is_pose_dependent`].
    #[must_use]
    pub fn new() -> Self {
        Self {
            contour_chains: ContourVertexChains::default_flame(),
            config: DynamicLandmarkConfig::default(),
            pose_dependent: false,
        }
    }

    /// Create a new extractor with custom configuration and the fallback
    /// contour chains (not pose-dependent).
    #[must_use]
    pub fn with_config(config: DynamicLandmarkConfig) -> Self {
        Self {
            contour_chains: ContourVertexChains::default_flame(),
            config,
            pose_dependent: false,
        }
    }

    /// Create a new extractor with both custom config and custom contour chains.
    ///
    /// Chains supplied here are treated as **authoritative pose-dependent
    /// contours** — typically the output of
    /// [`ContourVertexChains::from_dynamic_embedding`] — so
    /// [`DynamicLandmarkExtractor::extract_all`] will use them to replace the
    /// static jaw landmarks.  Passing [`ContourVertexChains::default_flame`]
    /// here therefore opts back into that overwrite, which
    /// [`DynamicLandmarkExtractor::new`] deliberately avoids.
    #[must_use]
    pub fn with_chains_and_config(
        contour_chains: ContourVertexChains,
        config: DynamicLandmarkConfig,
    ) -> Self {
        Self {
            contour_chains,
            config,
            pose_dependent: true,
        }
    }

    /// Whether this extractor carries a contour that really does vary with head
    /// pose.
    ///
    /// `false` means the extractor is running on the
    /// [`ContourVertexChains::default_flame`] fallback: `extract` still returns
    /// jaw contour points, but they come from the static jaw line and the
    /// selected side only changes which half is walked.
    /// [`DynamicLandmarkExtractor::extract_all`] uses this to decide whether
    /// replacing the static jaw landmarks would add information or destroy it.
    #[inline]
    #[must_use]
    pub fn is_pose_dependent(&self) -> bool {
        self.pose_dependent
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
    /// Follows the crate-wide convention (positive yaw = turn to the subject's
    /// right) and picks the side that *recedes* under that turn, because that
    /// is the side whose contour moves:
    ///
    /// - `|yaw| < threshold` → [`ContourSide::Both`]
    /// - `yaw > 0`           → [`ContourSide::Right`] (right turn — the right
    ///   side of the jaw rotates away from the camera)
    /// - `yaw < 0`           → [`ContourSide::Left`]  (left turn)
    #[must_use]
    pub fn select_contour_side(&self, params: &FlameParams) -> ContourSide {
        let yaw = Self::extract_yaw(params);
        if yaw.abs() < self.config.side_threshold_rad {
            ContourSide::Both
        } else if yaw > 0.0 {
            ContourSide::Right
        } else {
            ContourSide::Left
        }
    }

    // -----------------------------------------------------------------------
    // Core extraction helpers
    // -----------------------------------------------------------------------

    /// Look up positions for a slice of vertex indices from `mesh`.
    ///
    /// Samples at most `count` indices from `chain`; `label` names the chain in
    /// any error message.  Returns an error when any requested index is
    /// ≥ `mesh.vertices.len()`.
    fn sample_chain(
        mesh: &Mesh,
        chain: &[u32],
        count: usize,
        start_index: usize,
        label: &str,
        group: LandmarkGroup,
    ) -> Result<Vec<Landmark>, FlameError> {
        let num_verts = mesh.vertices.len();
        let n = count.min(chain.len());
        let mut landmarks = Vec::with_capacity(n);

        for (offset, &vi_u32) in chain.iter().take(n).enumerate() {
            let vi = vi_u32 as usize;
            if vi >= num_verts {
                return Err(FlameError::index_out_of_bounds(
                    format!("dynamic contour {label}-chain vertex at chain-offset {offset}"),
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
    /// Returns **at most** [`DynamicLandmarkConfig::num_contour_landmarks`]
    /// landmarks with [`LandmarkGroup::JawLine`] group labels; landmark `index`
    /// values start at `0` and are sequential.
    ///
    /// The count is capped by the chains actually available:
    ///
    /// - [`ContourSide::Left`] / [`ContourSide::Right`] yield at most
    ///   `left_chain.len()` / `right_chain.len()` landmarks — with the default
    ///   chains that is one half of the jaw contour, not all 17 points.
    /// - [`ContourSide::Both`] (frontal pose) concatenates `ceil(n / 2)` points
    ///   from the left chain and `floor(n / 2)` from the right chain, producing
    ///   the full contour when both chains are long enough.  The chains are
    ///   disjoint, so no vertex is emitted twice.
    ///
    /// Callers that need a fixed-length result must check `len()`;
    /// [`DynamicLandmarkExtractor::extract_all`] logs a warning when the
    /// contour is shorter than the iBUG jaw line.
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
                "left",
                LandmarkGroup::JawLine,
            ),
            ContourSide::Right => Self::sample_chain(
                mesh,
                &self.contour_chains.right_chain,
                n,
                0,
                "right",
                LandmarkGroup::JawLine,
            ),
            ContourSide::Both => {
                // Concatenate the two disjoint chains: ceil(n/2) points from the
                // left chain (chin outward) then floor(n/2) from the right.
                let left_n = n.div_ceil(2);
                let right_n = n / 2;

                let mut landmarks = Self::sample_chain(
                    mesh,
                    &self.contour_chains.left_chain,
                    left_n,
                    0,
                    "left",
                    LandmarkGroup::JawLine,
                )?;
                // Continue numbering from the points actually emitted, so a
                // short left chain leaves no gap in the landmark indices.
                let right_start = landmarks.len();
                let right = Self::sample_chain(
                    mesh,
                    &self.contour_chains.right_chain,
                    right_n,
                    right_start,
                    "right",
                    LandmarkGroup::JawLine,
                )?;
                landmarks.extend(right);

                Ok(landmarks)
            }
        }
    }

    /// Compute all 68 landmarks: the static set with the jaw line
    /// (indices 0–16) replaced by the pose-dependent dynamic contour.
    ///
    /// The returned `Vec` always has exactly 68 entries.  Landmark `index`
    /// values match the iBUG 68-point convention so that downstream code does
    /// not need to know whether static or dynamic jaw points were used.
    ///
    /// # Without a pose-dependent contour
    ///
    /// When [`DynamicLandmarkExtractor::is_pose_dependent`] is `false` the jaw
    /// slots are **left untouched** and the static 68-point set is returned
    /// verbatim.  The fallback chains are two halves of the very jaw contour
    /// those static landmarks already carry, so overwriting slot *i* with
    /// contour point *i* would only scramble which vertex lands in which iBUG
    /// slot — corrupting an otherwise correct landmark set while advertising a
    /// pose dependence that is not there.  Build the extractor with
    /// [`DynamicLandmarkExtractor::with_chains_and_config`] from a real
    /// embedding to enable the replacement.
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

        if !self.pose_dependent {
            tracing::debug!(
                "dynamic_landmarks: no pose-dependent contour loaded (default chains); \
                 keeping the static iBUG jaw line untouched"
            );
            return Ok(static_landmarks);
        }

        // Compute the dynamic jaw contour.
        let dynamic_jaw = self.extract(mesh, params)?;
        let jaw_slots = LandmarkGroup::JawLine.count();

        // Overwrite jaw-line entries (indices 0-16) with dynamic landmarks.
        // Preserve the original iBUG index values and group labels.
        let mut replaced = 0_usize;
        for (jaw_slot, dyn_lm) in static_landmarks
            .iter_mut()
            .take(jaw_slots)
            .zip(dynamic_jaw.iter())
        {
            // Keep jaw_slot.index unchanged (it's 0..16) — just update position.
            jaw_slot.position = dyn_lm.position;
            replaced += 1;
        }

        if replaced < jaw_slots {
            tracing::warn!(
                replaced,
                expected = jaw_slots,
                "dynamic_landmarks: contour is shorter than the iBUG jaw line; \
                 the remaining jaw slots keep their static positions"
            );
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
    /// Convenience wrapper around `DynamicLandmarkExtractor::new().extract()`,
    /// which uses the pose-independent fallback chains — see
    /// [`ContourVertexChains::default_flame`].  Returns the jaw contour points
    /// for the side selected by head pose: the full 17-point contour for a
    /// near-frontal pose, one half of it otherwise.
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

    /// Minimum vertex count to satisfy all canonical FLAME landmark indices,
    /// which the contour chains are now derived from (max index 2450).
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
    fn test_default_chains_split_the_jaw_line_at_the_chin() {
        // 17 jaw points split at the chin: 9 on the left (chin included), 8 on
        // the right, disjoint and together covering the whole contour.
        let chains = ContourVertexChains::default_flame();
        assert_eq!(chains.left_chain.len(), 9);
        assert_eq!(chains.right_chain.len(), 8);
        assert_eq!(chains.left_chain.len() + chains.right_chain.len(), 17);
    }

    #[test]
    fn test_default_chains_are_the_canonical_jaw_vertices() {
        // Regression: the chains used to be fabricated runs (1..17 and
        // 4984..5000) picked for existing, not for tracing the jaw.
        let extractor = LandmarkExtractor::new();
        let jaw = extractor.group_indices(LandmarkGroup::JawLine);
        let chains = ContourVertexChains::default_flame();
        let mut covered: Vec<u32> = chains
            .left_chain
            .iter()
            .chain(chains.right_chain.iter())
            .copied()
            .collect();
        covered.sort_unstable();
        let mut expected = jaw.to_vec();
        expected.sort_unstable();
        assert_eq!(
            covered, expected,
            "the two chains must partition the canonical jaw contour"
        );
    }

    #[test]
    fn test_default_chains_are_disjoint() {
        let chains = ContourVertexChains::default_flame();
        for vertex in &chains.left_chain {
            assert!(
                !chains.right_chain.contains(vertex),
                "vertex {vertex} appears in both chains"
            );
        }
    }

    #[test]
    fn test_default_chains_start_at_the_chin() {
        // The chin is the midpoint of the jaw contour and belongs to the left
        // chain, which walks chin-outward.
        let extractor = LandmarkExtractor::new();
        let jaw = extractor.group_indices(LandmarkGroup::JawLine);
        let chains = ContourVertexChains::default_flame();
        assert_eq!(chains.left_chain[0], jaw[jaw.len() / 2]);
        assert_eq!(chains.right_chain[0], jaw[jaw.len() / 2 + 1]);
    }

    #[test]
    fn test_from_jaw_contour_empty_is_empty() {
        let chains = ContourVertexChains::from_jaw_contour(&[]);
        assert!(chains.left_chain.is_empty());
        assert!(chains.right_chain.is_empty());
    }

    #[test]
    fn test_from_jaw_contour_single_point() {
        let chains = ContourVertexChains::from_jaw_contour(&[42]);
        assert_eq!(chains.left_chain, vec![42]);
        assert!(chains.right_chain.is_empty());
    }

    // -----------------------------------------------------------------------
    // ContourVertexChains::from_dynamic_embedding
    // -----------------------------------------------------------------------

    #[test]
    fn test_from_dynamic_embedding_collapses_to_the_dominant_corner() {
        // Two yaw bins × two contour points, over three triangles.
        let faces = vec![[10_u32, 11, 12], [20, 21, 22], [30, 31, 32]];
        let lmk_faces_idx = vec![0_u32, 1, 2, 0];
        let lmk_b_coords = vec![
            // bin 0, point 0 → corner 2 of face 0 → vertex 12
            0.1_f32, 0.2, 0.7, // bin 0, point 1 → corner 0 of face 1 → vertex 20
            0.8, 0.1, 0.1, // bin 1, point 0 → corner 1 of face 2 → vertex 31
            0.2, 0.6, 0.2, // bin 1, point 1 → corner 1 of face 0 → vertex 11
            0.3, 0.5, 0.2,
        ];
        let chains = ContourVertexChains::from_dynamic_embedding(
            &lmk_faces_idx,
            &lmk_b_coords,
            &faces,
            2,
            0,
            1,
        )
        .expect("well-formed embedding");
        assert_eq!(chains.left_chain, vec![12, 20]);
        assert_eq!(chains.right_chain, vec![31, 11]);
    }

    #[test]
    fn test_from_dynamic_embedding_rejects_ragged_face_table() {
        let faces = vec![[0_u32, 1, 2]];
        // 3 entries cannot be split into whole bins of 2 contour points.
        assert!(ContourVertexChains::from_dynamic_embedding(
            &[0_u32, 0, 0],
            &[1.0_f32; 9],
            &faces,
            2,
            0,
            0
        )
        .is_err());
    }

    #[test]
    fn test_from_dynamic_embedding_rejects_bad_bary_length() {
        let faces = vec![[0_u32, 1, 2]];
        assert!(ContourVertexChains::from_dynamic_embedding(
            &[0_u32, 0],
            &[1.0_f32; 5],
            &faces,
            2,
            0,
            0
        )
        .is_err());
    }

    #[test]
    fn test_from_dynamic_embedding_rejects_out_of_range_bin() {
        let faces = vec![[0_u32, 1, 2]];
        assert!(matches!(
            ContourVertexChains::from_dynamic_embedding(
                &[0_u32, 0],
                &[1.0_f32; 6],
                &faces,
                2,
                0,
                7
            ),
            Err(FlameError::IndexOutOfBounds { .. })
        ));
    }

    #[test]
    fn test_from_dynamic_embedding_rejects_out_of_range_face() {
        let faces = vec![[0_u32, 1, 2]];
        assert!(matches!(
            ContourVertexChains::from_dynamic_embedding(
                &[0_u32, 9],
                &[1.0_f32; 6],
                &faces,
                2,
                0,
                0
            ),
            Err(FlameError::IndexOutOfBounds { .. })
        ));
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
    fn test_select_side_positive_yaw_is_the_receding_right_side() {
        // Regression: this module used to read positive yaw as a LEFT turn,
        // inverting the convention the rest of the crate documents
        // (canonical::HeadOrientation::yaw, the `head_turn` rig control which
        // drives root pose index 1, pose_estimation::estimate_yaw_from_symmetry
        // — all "positive = turn right").  A right turn swings the subject's
        // right side away from the camera, so the right chain is the one that
        // traces the pose-dependent contour.
        let extractor = DynamicLandmarkExtractor::new();
        let params = params_with_yaw(0.5); // well above threshold 0.1
        assert_eq!(extractor.select_contour_side(&params), ContourSide::Right);
    }

    #[test]
    fn test_select_side_negative_yaw_is_the_receding_left_side() {
        let extractor = DynamicLandmarkExtractor::new();
        let params = params_with_yaw(-0.5);
        assert_eq!(extractor.select_contour_side(&params), ContourSide::Left);
    }

    #[test]
    fn test_select_side_is_antisymmetric_in_yaw() {
        // Whatever the labelling, mirroring the yaw must mirror the side.
        let extractor = DynamicLandmarkExtractor::new();
        let positive = extractor.select_contour_side(&params_with_yaw(0.4));
        let negative = extractor.select_contour_side(&params_with_yaw(-0.4));
        assert_ne!(positive, negative);
        assert_ne!(positive, ContourSide::Both);
        assert_ne!(negative, ContourSide::Both);
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
        // Just above threshold → Right (positive yaw = turn right)
        let params_above = params_with_yaw(0.15);
        assert_eq!(
            extractor.select_contour_side(&params_above),
            ContourSide::Right
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
    fn test_extract_right_side_returns_the_right_half_chain() {
        let mesh = synthetic_mesh(MIN_VERTS);
        let extractor = DynamicLandmarkExtractor::new();
        let params = params_with_yaw(0.5); // right turn
        let landmarks = extractor
            .extract(&mesh, &params)
            .expect("extract right should succeed");
        // Documented as "at most num_contour_landmarks": the right half of the
        // jaw contour is 8 points, fewer than the 17 requested.
        assert_eq!(landmarks.len(), 8);
    }

    #[test]
    fn test_extract_left_side_returns_the_left_half_chain() {
        let mesh = synthetic_mesh(MIN_VERTS);
        let extractor = DynamicLandmarkExtractor::new();
        let params = params_with_yaw(-0.5); // left turn
        let landmarks = extractor
            .extract(&mesh, &params)
            .expect("extract left should succeed");
        assert_eq!(landmarks.len(), 9);
    }

    #[test]
    fn test_extract_frontal_contour_has_no_duplicate_vertices() {
        // The chains are disjoint, so the frontal contour must not repeat the
        // chin (or any other point).
        let mesh = synthetic_mesh(MIN_VERTS);
        let extractor = DynamicLandmarkExtractor::new();
        let landmarks = extractor
            .extract(&mesh, &FlameParams::neutral())
            .expect("frontal extraction");
        for (i, lm) in landmarks.iter().enumerate() {
            for other in landmarks.iter().skip(i + 1) {
                assert!(
                    (lm.position[0] - other.position[0]).abs() > 1e-9
                        || (lm.position[1] - other.position[1]).abs() > 1e-9
                        || (lm.position[2] - other.position[2]).abs() > 1e-9,
                    "contour point {i} is duplicated"
                );
            }
        }
    }

    #[test]
    fn test_extract_shorter_chain_is_capped_not_padded() {
        // A caller asking for more landmarks than the chain holds gets the
        // chain length back, never a panic and never fabricated points.
        let mesh = synthetic_mesh(MIN_VERTS);
        let config = DynamicLandmarkConfig {
            num_contour_landmarks: 100,
            ..DynamicLandmarkConfig::default()
        };
        let extractor = DynamicLandmarkExtractor::with_config(config);
        let landmarks = extractor
            .extract(&mesh, &params_with_yaw(-0.5))
            .expect("extraction");
        assert_eq!(landmarks.len(), 9);
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
        // The chains are canonical jaw vertices (indices ~1900-2450), so a
        // 100-vertex mesh cannot satisfy either side.
        let mesh = synthetic_mesh(100);
        let extractor = DynamicLandmarkExtractor::new();
        for params in [params_with_yaw(-0.5), params_with_yaw(0.5)] {
            assert!(
                matches!(
                    extractor.extract(&mesh, &params),
                    Err(FlameError::IndexOutOfBounds { .. })
                ),
                "extract on a tiny mesh must return IndexOutOfBounds, never index out of range"
            );
        }
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
    fn test_mesh_method_small_mesh_reports_the_missing_vertex() {
        // Canonical jaw vertices live around index 1900-2450, so a 50-vertex
        // mesh is reported as out of bounds rather than silently truncated.
        let mesh = synthetic_mesh(50);
        let params = params_with_yaw(0.5);
        assert!(matches!(
            mesh.extract_dynamic_landmarks(&params),
            Err(FlameError::IndexOutOfBounds { .. })
        ));
    }

    // -----------------------------------------------------------------------
    // Pose dependence: extract_all must not corrupt the static jaw line
    // -----------------------------------------------------------------------

    #[test]
    fn test_default_extractor_is_not_pose_dependent() {
        assert!(!DynamicLandmarkExtractor::new().is_pose_dependent());
        assert!(
            !DynamicLandmarkExtractor::with_config(DynamicLandmarkConfig::default())
                .is_pose_dependent()
        );
    }

    #[test]
    fn test_supplied_chains_are_pose_dependent() {
        let extractor = DynamicLandmarkExtractor::with_chains_and_config(
            ContourVertexChains {
                left_chain: vec![100, 101],
                right_chain: vec![200, 201],
            },
            DynamicLandmarkConfig::default(),
        );
        assert!(extractor.is_pose_dependent());
    }

    #[test]
    fn test_extract_all_keeps_the_static_jaw_without_a_real_embedding() {
        // Regression: extract_all used to overwrite iBUG slots 0-16 with the
        // fallback contour, scrambling an otherwise correct landmark set with
        // no error signal.  With no pose-dependent contour loaded the static
        // 68-point set must come back untouched.
        let mesh = synthetic_mesh(MIN_VERTS);
        let extractor = DynamicLandmarkExtractor::new();
        let params = params_with_yaw(0.5);
        let dynamic = extractor
            .extract_all(&mesh, &params)
            .expect("extract_all should succeed");
        let static_only = LandmarkExtractor::new()
            .extract(&mesh)
            .expect("static extraction");
        assert_eq!(dynamic.len(), static_only.len());
        for (got, expected) in dynamic.iter().zip(static_only.iter()) {
            assert_eq!(
                got.position, expected.position,
                "landmark {} must keep its static position",
                got.index
            );
        }
    }

    #[test]
    fn test_extract_all_uses_supplied_pose_dependent_chains() {
        let mesh = synthetic_mesh(MIN_VERTS);
        // 9 + 8 disjoint vertices so a frontal pose fills all 17 jaw slots.
        let left_chain: Vec<u32> = (100..109).collect();
        let right_chain: Vec<u32> = (200..208).collect();
        let extractor = DynamicLandmarkExtractor::with_chains_and_config(
            ContourVertexChains {
                left_chain: left_chain.clone(),
                right_chain: right_chain.clone(),
            },
            DynamicLandmarkConfig::default(),
        );
        let landmarks = extractor
            .extract_all(&mesh, &FlameParams::neutral())
            .expect("extract_all with custom chains");
        assert_eq!(landmarks.len(), 68);

        let expected: Vec<u32> = left_chain.into_iter().chain(right_chain).collect();
        for (slot, &vertex) in expected.iter().enumerate() {
            let v = &mesh.vertices[vertex as usize];
            assert_eq!(
                landmarks[slot].position,
                [v.x, v.y, v.z],
                "jaw slot {slot} should carry the supplied contour vertex {vertex}"
            );
            assert_eq!(
                landmarks[slot].index, slot,
                "iBUG indices must be preserved"
            );
        }
    }

    #[test]
    fn test_extract_all_short_contour_leaves_remaining_slots_static() {
        let mesh = synthetic_mesh(MIN_VERTS);
        let extractor = DynamicLandmarkExtractor::with_chains_and_config(
            ContourVertexChains {
                left_chain: vec![100, 101],
                right_chain: vec![200],
            },
            DynamicLandmarkConfig::default(),
        );
        let landmarks = extractor
            .extract_all(&mesh, &FlameParams::neutral())
            .expect("extract_all with a short contour");
        assert_eq!(landmarks.len(), 68);
        let static_only = LandmarkExtractor::new()
            .extract(&mesh)
            .expect("static extraction");
        // Slots beyond the supplied contour keep their static positions rather
        // than being dropped or zeroed.
        for slot in 3..LandmarkGroup::JawLine.count() {
            assert_eq!(landmarks[slot].position, static_only[slot].position);
        }
    }
}
