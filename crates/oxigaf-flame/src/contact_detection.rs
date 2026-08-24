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

use crate::landmarks::{LandmarkExtractor, LandmarkGroup};
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

    /// [`ContactConfig::num_sample_pairs`] is zero.
    #[error("Invalid sample count: num_sample_pairs must be >= 1")]
    InvalidSampleCount,
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
    /// - [`ContactError::InvalidThreshold`] if any threshold is `<= 0`.
    /// - [`ContactError::InvalidSampleCount`] if `num_sample_pairs` is `0`.
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
        if self.num_sample_pairs == 0 {
            return Err(ContactError::InvalidSampleCount);
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
    /// **Signed** separation between the two vertices (world units).
    ///
    /// The magnitude is the Euclidean distance.  The value is **negative** when
    /// the two surfaces interpenetrate — when `vertex_b` lies behind the
    /// outward-facing surface at `vertex_a` (and, where both normals are
    /// usable, vice versa).  Vertices that merely touch keep a positive
    /// separation, so `distance < 0` is the clipping test.
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
    /// No vertex pair with a negative signed separation was detected.
    #[default]
    NoClipping,
    /// At least one vertex pair has a negative signed separation, i.e. the two
    /// surfaces have driven through each other.
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
    /// Whether any pair has a negative signed separation (interpenetration / clipping).
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
    /// Number of pairs whose [signed separation](ContactPair::distance) is `< 0`.
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

    /// Whether no interpenetration was found **in the regions that were
    /// analysed**.
    ///
    /// This is a report-scoped predicate, not a whole-mesh guarantee: for a
    /// report produced by [`analyze_contact`] it covers the lip and eyelid
    /// regions of [`FlameContactRegions`], and nothing else.  Interpenetration
    /// elsewhere on the head — a nose through a cheek, a tongue through a
    /// palate — is outside what the report observed, so `true` means "no
    /// clipping was detected in the analysed regions", not "the mesh is
    /// globally self-intersection free".
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

/// Pick the entries of `indices` at the given offsets, skipping offsets that
/// fall outside the slice.
fn pick_landmarks(indices: &[u32], offsets: &[usize]) -> Vec<usize> {
    offsets
        .iter()
        .filter_map(|&offset| indices.get(offset).map(|&v| v as usize))
        .collect()
}

impl FlameContactRegions {
    /// Derive contact regions from a [`LandmarkExtractor`]'s vertex indices.
    ///
    /// The contacting surfaces are read off the iBUG 68-point layout that the
    /// extractor already models, rather than being written out as literals:
    ///
    /// | Region              | iBUG points          | Meaning                    |
    /// |---------------------|----------------------|----------------------------|
    /// | `upper_lip`         | 61, 62, 63           | inner upper lip margin     |
    /// | `lower_lip`         | 65, 66, 67           | inner lower lip margin     |
    /// | `*_upper_eyelid`    | 37, 38 / 43, 44      | upper lid margin           |
    /// | `*_lower_eyelid`    | 40, 41 / 46, 47      | lower lid margin           |
    ///
    /// The *inner* lip contour is used for the mouth because it is the surface
    /// that actually meets when the lips close; the outer contour (48–59) never
    /// touches.  Likewise the lid margins, not the eye corners, are what meet
    /// when an eye shuts — the corners are shared by both lids and would
    /// report a permanently closed eye.
    ///
    /// Regions are only as good as the extractor's indices: an extractor built
    /// with [`LandmarkExtractor::with_indices`] and fewer than 68 entries
    /// yields correspondingly shorter (possibly empty) regions, which the
    /// analysis functions then reject with [`ContactError::EmptyRegion`].
    #[must_use]
    pub fn from_landmark_extractor(extractor: &LandmarkExtractor) -> Self {
        let inner_lip = extractor.group_indices(LandmarkGroup::InnerLip);
        let left_eye = extractor.group_indices(LandmarkGroup::LeftEye);
        let right_eye = extractor.group_indices(LandmarkGroup::RightEye);
        Self {
            // Inner lip group is iBUG 60-67; offsets are relative to point 60.
            upper_lip: pick_landmarks(inner_lip, &[1, 2, 3]),
            lower_lip: pick_landmarks(inner_lip, &[5, 6, 7]),
            // Eye groups are 6 points: 0 = outer corner, 1-2 = upper lid,
            // 3 = inner corner, 4-5 = lower lid.
            left_upper_eyelid: pick_landmarks(left_eye, &[1, 2]),
            left_lower_eyelid: pick_landmarks(left_eye, &[4, 5]),
            right_upper_eyelid: pick_landmarks(right_eye, &[1, 2]),
            right_lower_eyelid: pick_landmarks(right_eye, &[4, 5]),
        }
    }

    /// Default FLAME contact regions (approximate, for standard 5023-vertex FLAME).
    ///
    /// Derived from the canonical FLAME 68-point landmark table via
    /// [`FlameContactRegions::from_landmark_extractor`], so the regions are at
    /// least anatomically *placed*: the lip regions sit on the inner lip
    /// margins and the eyelid regions on the lid margins, instead of on runs of
    /// consecutive vertex indices, which are unrelated to each other in the
    /// FLAME topology.
    ///
    /// These indices remain **approximate**.  The landmark table they come from
    /// is itself an approximation of the FLAME semantic annotations, and each
    /// region is only two or three vertices wide, so the absolute distances
    /// feeding [`classify_jaw_state`] and [`classify_eye_state`] should be
    /// re-calibrated against a known-good mesh before being trusted at the
    /// millimetre scale.  When accurate per-region annotations are available
    /// (`FLAME_masks.pkl` and friends), build a [`LandmarkExtractor`] from them
    /// and call [`FlameContactRegions::from_landmark_extractor`], or construct
    /// [`FlameContactRegions`] directly.
    #[must_use]
    pub fn default_flame() -> Self {
        Self::from_landmark_extractor(&LandmarkExtractor::new())
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

/// Minimum projection depth (world units) before a vertex pair is called
/// interpenetrating.
///
/// Vertex normals along a seam that is pressed shut (closed lips, shut eyelids)
/// are numerically ambiguous, so a bare sign test on the projection would flip
/// at random for perfectly healthy contact.  Requiring the projection to clear
/// 10 µm — two orders of magnitude below the 1 mm eyelid threshold — keeps
/// legitimate contact positive while still catching real interpenetration.
const PENETRATION_EPS: f32 = 1e-5;

/// Minimum share of the separation that must lie *along* the surface normal
/// before a pair counts as interpenetrating.
///
/// Without this the sign test would be scale-free: two vertices sitting on
/// back-to-back surfaces would read as "interpenetrating" however far apart
/// they are, because a millimetre of normal-aligned offset looks the same
/// whether the vertices are 2 mm or 2 m apart.  Requiring half the separation
/// to lie along the normal (i.e. the offset within 60° of the normal) demands
/// that the offset run *through* the surface rather than along it.
const MIN_PENETRATION_ALIGNMENT: f32 = 0.5;

/// Squared length below which a vertex normal is treated as unusable.
///
/// [`Mesh::recompute_normals`] leaves the normal of a vertex with no incident
/// faces at exactly zero, and such a vertex carries no orientation information.
const MIN_NORMAL_LEN_SQ: f32 = 1e-12;

/// Compute the Euclidean distance between two 3D points.
#[inline]
fn dist3(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Dot product of two 3D vectors.
#[inline]
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Unit-length outward normal at `idx`, or `None` when the mesh carries no
/// usable orientation there (missing entry, or a degenerate zero normal).
#[inline]
fn unit_normal(mesh: &Mesh, idx: usize) -> Option<[f32; 3]> {
    let n = mesh.normals.get(idx)?;
    let len_sq = n.x * n.x + n.y * n.y + n.z * n.z;
    if len_sq < MIN_NORMAL_LEN_SQ {
        return None;
    }
    let inv_len = len_sq.sqrt().recip();
    Some([n.x * inv_len, n.y * inv_len, n.z * inv_len])
}

/// Signed separation between vertex `a` and vertex `b`.
///
/// The **magnitude** is the plain Euclidean distance.  The **sign** is negative
/// when the two surfaces have passed through each other, which is what makes
/// interpenetration (clipping) detectable at all: a Euclidean distance alone is
/// never negative and reaches `0.0` only for bit-identical coincident vertices.
///
/// A vertex pair is interpenetrating when `b` sits *behind* the outward-facing
/// surface at `a`, i.e. `(p_b − p_a) · n_a < −depth`, where `depth` scales with
/// the separation (see [`MIN_PENETRATION_ALIGNMENT`]) and never falls below
/// [`PENETRATION_EPS`].  When both vertices carry a usable normal the mirrored
/// test `(p_a − p_b) · n_b < −depth` must agree, which rejects the false
/// positives that a single noisy normal would produce.  When neither vertex has
/// a usable normal the pair is reported as separated — unknown orientation is
/// never treated as evidence of clipping.
///
/// The distance-proportional gate is what keeps the test honest for vertices
/// that merely happen to lie on back-to-back surfaces: two points a metre apart
/// need a metre-scale normal-aligned offset to qualify, not a millimetre.
fn signed_separation(mesh: &Mesh, a: usize, pa: [f32; 3], b: usize, pb: [f32; 3]) -> f32 {
    let distance = dist3(pa, pb);
    let a_to_b = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
    // Required normal-aligned depth: a fixed share of the separation, floored so
    // that a seam pressed shut to numerical noise never flips sign.
    let min_depth = (distance * MIN_PENETRATION_ALIGNMENT).max(PENETRATION_EPS);
    // `(p_a − p_b) · n_b < −depth` is `a_to_b · n_b > depth` with the sign folded in.
    let penetrating = match (unit_normal(mesh, a), unit_normal(mesh, b)) {
        (Some(na), Some(nb)) => dot3(a_to_b, na) < -min_depth && dot3(a_to_b, nb) > min_depth,
        (Some(na), None) => dot3(a_to_b, na) < -min_depth,
        (None, Some(nb)) => dot3(a_to_b, nb) > min_depth,
        (None, None) => false,
    };
    if penetrating {
        -distance
    } else {
        distance
    }
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
/// Iterates all (a, b) pairs and returns those whose signed separation
/// ([`ContactPair::distance`]) is strictly less than `threshold`.  Because
/// interpenetrating pairs carry a *negative* separation they always pass the
/// test, however deep the interpenetration is — a mesh whose lips have driven
/// 5 mm through each other is still reported under a 2 mm threshold.
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
            let distance = signed_separation(mesh, a, pa, b, pb);
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
/// Pair distances are [signed](ContactPair::distance), so interpenetrating
/// pairs are always reported regardless of `config.general_threshold`.
///
/// # Errors
///
/// - [`ContactError::InvalidThreshold`] / [`ContactError::InvalidSampleCount`]
///   if `config` fails [`ContactConfig::validate`].
/// - [`ContactError::EmptyRegion`] if either region is empty.
/// - [`ContactError::VertexOutOfRange`] if any index is out of bounds.
pub fn detect_self_contact(
    mesh: &Mesh,
    region_a: &[usize],
    region_b: &[usize],
    config: &ContactConfig,
) -> Result<Vec<ContactPair>, ContactError> {
    config.validate()?;
    if region_a.is_empty() || region_b.is_empty() {
        return Err(ContactError::EmptyRegion);
    }
    // `validate` already rejects zero, but clamp anyway: this division must
    // never be reachable with a zero divisor.
    let num_pairs = config.num_sample_pairs.max(1);
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
            let distance = signed_separation(mesh, a, pa, b, pb);
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
/// - Contact pairs across every opposing region pair: upper/lower lip (at
///   `mouth_contact_threshold`) and both eyelid pairs (at
///   `eye_contact_threshold`)
/// - Clipping detection (any pair whose [signed separation](ContactPair::distance)
///   is negative), which therefore covers lips driven through each other *and*
///   an eyelid driven through its counterpart
///
/// # Errors
///
/// - [`ContactError::EmptyMesh`] if the mesh has no vertices.
/// - [`ContactError::InvalidThreshold`] / [`ContactError::InvalidSampleCount`]
///   if `config` fails [`ContactConfig::validate`].
/// - Propagates other [`ContactError`] variants from sub-functions.  In
///   particular a failure to extract the lip contact pairs is **not** silently
///   downgraded to "no contact": an empty region or an out-of-range vertex
///   index surfaces as an error rather than as a clean, clipping-free report.
pub fn analyze_contact(
    mesh: &Mesh,
    regions: &FlameContactRegions,
    config: &ContactConfig,
) -> Result<ContactReport, ContactError> {
    if mesh.vertices.is_empty() {
        return Err(ContactError::EmptyMesh);
    }
    config.validate()?;

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

    // Find all close pairs across every opposing region pair.  Errors propagate:
    // a region that could not be analysed must not masquerade as a clean mesh.
    let mut contact_pairs = find_contact_pairs(
        mesh,
        &regions.upper_lip,
        &regions.lower_lip,
        config.mouth_contact_threshold,
    )?;
    for (upper, lower) in [
        (&regions.left_upper_eyelid, &regions.left_lower_eyelid),
        (&regions.right_upper_eyelid, &regions.right_lower_eyelid),
    ] {
        contact_pairs.extend(find_contact_pairs(
            mesh,
            upper,
            lower,
            config.eye_contact_threshold,
        )?);
    }

    // Interpenetration: `ContactPair::distance` is a *signed* separation, so a
    // negative value means the two surfaces have driven through each other.
    let clipping_count = contact_pairs.iter().filter(|p| p.distance < 0.0).count();
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

    // -----------------------------------------------------------------------
    // Regression: num_sample_pairs == 0 used to divide by zero
    // -----------------------------------------------------------------------

    #[test]
    fn contact_config_validate_zero_sample_pairs() {
        let cfg = ContactConfig {
            num_sample_pairs: 0,
            ..ContactConfig::default()
        };
        assert!(matches!(
            cfg.validate(),
            Err(ContactError::InvalidSampleCount)
        ));
    }

    #[test]
    fn detect_self_contact_zero_sample_pairs_errors_instead_of_dividing_by_zero() {
        let mesh = make_mesh(&[[0.0, 0.0, 0.0], [0.003, 0.0, 0.0]]);
        let config = ContactConfig {
            num_sample_pairs: 0,
            ..ContactConfig::default()
        };
        assert!(matches!(
            detect_self_contact(&mesh, &[0], &[1], &config),
            Err(ContactError::InvalidSampleCount)
        ));
    }

    #[test]
    fn analyze_contact_zero_sample_pairs_errors() {
        let mesh = seam_mesh(-0.01);
        let config = ContactConfig {
            num_sample_pairs: 0,
            ..ContactConfig::default()
        };
        assert!(matches!(
            analyze_contact(&mesh, &seam_regions(), &config),
            Err(ContactError::InvalidSampleCount)
        ));
    }

    // -----------------------------------------------------------------------
    // Regression: analyze_contact used to swallow sub-function errors
    // -----------------------------------------------------------------------

    #[test]
    fn analyze_contact_rejects_non_positive_mouth_threshold() {
        // With `.unwrap_or_default()` this produced a clean, contact-free,
        // clipping-free report even though the lip analysis never ran.
        let mesh = seam_mesh(-0.01);
        let config = ContactConfig {
            mouth_contact_threshold: 0.0,
            ..ContactConfig::default()
        };
        assert!(matches!(
            analyze_contact(&mesh, &seam_regions(), &config),
            Err(ContactError::InvalidThreshold { .. })
        ));
    }

    #[test]
    fn analyze_contact_rejects_out_of_range_lip_vertex() {
        let mesh = seam_mesh(-0.01);
        let regions = FlameContactRegions {
            upper_lip: vec![99],
            ..seam_regions()
        };
        assert!(matches!(
            analyze_contact(&mesh, &regions, &ContactConfig::default()),
            Err(ContactError::VertexOutOfRange { .. })
        ));
    }

    // -----------------------------------------------------------------------
    // Regression: clipping detection could never fire (signed separation)
    // -----------------------------------------------------------------------

    /// Two facing triangle patches modelling a lip seam.
    ///
    /// Vertices 0–2 form the upper patch in the `y = 0` plane, wound so its
    /// normal points at `−Y` (down, into the gap).  Vertices 3–5 form the lower
    /// patch at `y = other_y`, wound so its normal points at `+Y` (up, into the
    /// gap).  Vertex 0 and vertex 3 share the `(x, z) = (0, 0)` column, so
    /// their separation is exactly `|other_y|`.
    ///
    /// A **negative** `other_y` is a healthy open/closed mouth; a **positive**
    /// `other_y` means the lower patch has driven through the upper one.
    fn seam_mesh(other_y: f32) -> Mesh {
        let vertices = vec![
            na::Point3::new(0.0, 0.0, 0.0),
            na::Point3::new(1.0, 0.0, 0.0),
            na::Point3::new(0.0, 0.0, 1.0),
            na::Point3::new(0.0, other_y, 0.0),
            na::Point3::new(0.0, other_y, 1.0),
            na::Point3::new(1.0, other_y, 0.0),
        ];
        Mesh::new(vertices, vec![[0, 1, 2], [3, 4, 5]])
    }

    /// Contact regions addressing the two patches of [`seam_mesh`].
    fn seam_regions() -> FlameContactRegions {
        FlameContactRegions {
            upper_lip: vec![0],
            lower_lip: vec![3],
            left_upper_eyelid: vec![1],
            left_lower_eyelid: vec![4],
            right_upper_eyelid: vec![2],
            right_lower_eyelid: vec![5],
        }
    }

    #[test]
    fn seam_mesh_normals_face_each_other() {
        // Guards the fixture itself: the rest of the signed-separation tests
        // are meaningless if the winding does not produce ∓Y normals.
        let mesh = seam_mesh(-0.01);
        assert!(mesh.normals[0].y < -0.9, "upper patch must face −Y");
        assert!(mesh.normals[3].y > 0.9, "lower patch must face +Y");
    }

    #[test]
    fn find_contact_pairs_separated_surfaces_stay_positive() {
        let mesh = seam_mesh(-0.01);
        let pairs = find_contact_pairs(&mesh, &[0], &[3], 0.02).expect("find pairs");
        assert_eq!(pairs.len(), 1);
        assert!(
            pairs[0].distance > 0.0,
            "surfaces that merely approach must keep a positive separation, got {}",
            pairs[0].distance
        );
        assert!((pairs[0].distance - 0.01).abs() < 1e-6);
    }

    #[test]
    fn find_contact_pairs_micro_overlap_is_not_clipping() {
        // 1 µm of numerical overlap is inside the PENETRATION_EPS guard band:
        // a seam pressed shut must not be reported as interpenetrating.
        let mesh = seam_mesh(1e-6);
        let pairs = find_contact_pairs(&mesh, &[0], &[3], 0.002).expect("find pairs");
        assert_eq!(pairs.len(), 1);
        assert!(
            pairs[0].distance > 0.0,
            "sub-epsilon overlap must not flip the sign, got {}",
            pairs[0].distance
        );
    }

    #[test]
    fn find_contact_pairs_signs_real_interpenetration() {
        let mesh = seam_mesh(0.002);
        let pairs = find_contact_pairs(&mesh, &[0], &[3], 0.002).expect("find pairs");
        assert_eq!(pairs.len(), 1);
        assert!(
            (pairs[0].distance + 0.002).abs() < 1e-6,
            "interpenetration must be reported as −0.002, got {}",
            pairs[0].distance
        );
    }

    #[test]
    fn find_contact_pairs_deep_interpenetration_beats_the_threshold() {
        // 20 mm of interpenetration is far outside the 2 mm contact threshold;
        // a signed separation is negative, so the pair is still reported.
        let mesh = seam_mesh(0.02);
        let pairs = find_contact_pairs(&mesh, &[0], &[3], 0.002).expect("find pairs");
        assert_eq!(pairs.len(), 1);
        assert!((pairs[0].distance + 0.02).abs() < 1e-5);
    }

    #[test]
    fn find_contact_pairs_distant_back_to_back_vertices_are_not_clipping() {
        // Vertices 1 and 4 of the seam mesh sit on the two facing patches but
        // are ~1.4 world units apart, with only 2 mm of that offset running
        // along the normals.  A scale-free sign test would call that
        // interpenetration; the alignment gate must not.
        let mesh = seam_mesh(0.002);
        let pairs = find_contact_pairs(&mesh, &[1], &[4], 10.0).expect("find pairs");
        assert_eq!(pairs.len(), 1);
        assert!(
            pairs[0].distance > 0.0,
            "a mostly-tangential offset is not interpenetration, got {}",
            pairs[0].distance
        );
    }

    #[test]
    fn find_contact_pairs_without_normals_never_reports_clipping() {
        // A mesh with no faces carries zero normals: orientation is unknown, and
        // unknown orientation must never be read as evidence of clipping.
        let mesh = make_mesh(&[[0.0, 0.0, 0.0], [0.001, 0.0, 0.0]]);
        let pairs = find_contact_pairs(&mesh, &[0], &[1], 0.005).expect("find pairs");
        assert_eq!(pairs.len(), 1);
        assert!(pairs[0].distance > 0.0);
    }

    #[test]
    fn analyze_contact_reports_clipping_for_an_interpenetrating_mesh() {
        let mesh = seam_mesh(0.002);
        let report = analyze_contact(&mesh, &seam_regions(), &ContactConfig::default())
            .expect("analysis of a valid mesh");
        assert_eq!(report.flags.clipping, ClippingStatus::HasClipping);
        assert_eq!(report.clipping_count, 1);
        assert!(
            !report.is_physically_valid(),
            "an interpenetrating mesh must not be reported as physically valid"
        );
        assert!(report.format_summary().contains("CLIPPING"));
    }

    #[test]
    fn analyze_contact_reports_no_clipping_for_a_healthy_mesh() {
        let mesh = seam_mesh(-0.01);
        let report = analyze_contact(&mesh, &seam_regions(), &ContactConfig::default())
            .expect("analysis of a valid mesh");
        assert_eq!(report.flags.clipping, ClippingStatus::NoClipping);
        assert_eq!(report.clipping_count, 0);
        assert!(report.is_physically_valid());
    }

    #[test]
    fn analyze_contact_detects_an_eyelid_driven_through_its_counterpart() {
        // Lips healthy, left eyelids interpenetrating: clipping must still fire,
        // because analyze_contact examines the eyelid regions too.
        let vertices = vec![
            // Lip patch A (normal −Y) at y = 0
            na::Point3::new(0.0, 0.0, 0.0),
            na::Point3::new(1.0, 0.0, 0.0),
            na::Point3::new(0.0, 0.0, 1.0),
            // Lip patch B (normal +Y) 10 mm below — healthy separation
            na::Point3::new(0.0, -0.01, 0.0),
            na::Point3::new(0.0, -0.01, 1.0),
            na::Point3::new(1.0, -0.01, 0.0),
            // Eyelid patch A (normal −Y) at y = 1.0
            na::Point3::new(5.0, 1.0, 0.0),
            na::Point3::new(6.0, 1.0, 0.0),
            na::Point3::new(5.0, 1.0, 1.0),
            // Eyelid patch B (normal +Y) driven 0.4 mm *above* patch A
            na::Point3::new(5.0, 1.0004, 0.0),
            na::Point3::new(5.0, 1.0004, 1.0),
            na::Point3::new(6.0, 1.0004, 0.0),
        ];
        let mesh = Mesh::new(vertices, vec![[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]);
        let regions = FlameContactRegions {
            upper_lip: vec![0],
            lower_lip: vec![3],
            left_upper_eyelid: vec![6],
            left_lower_eyelid: vec![9],
            right_upper_eyelid: vec![2],
            right_lower_eyelid: vec![5],
        };
        let report = analyze_contact(&mesh, &regions, &ContactConfig::default())
            .expect("analysis of a valid mesh");
        assert_eq!(report.flags.clipping, ClippingStatus::HasClipping);
        assert_eq!(report.clipping_count, 1);
        assert!(!report.is_physically_valid());
    }

    #[test]
    fn detect_self_contact_signs_interpenetration() {
        let mesh = seam_mesh(0.002);
        let pairs = detect_self_contact(&mesh, &[0], &[3], &ContactConfig::default())
            .expect("self contact");
        assert_eq!(pairs.len(), 1);
        assert!(pairs[0].distance < 0.0);
    }

    // -----------------------------------------------------------------------
    // Regression: default_flame() used to fabricate consecutive vertex indices
    // -----------------------------------------------------------------------

    #[test]
    fn default_flame_regions_come_from_the_canonical_landmark_table() {
        let extractor = LandmarkExtractor::new();
        let inner_lip = extractor.group_indices(LandmarkGroup::InnerLip);
        let left_eye = extractor.group_indices(LandmarkGroup::LeftEye);
        let regions = FlameContactRegions::default_flame();

        assert_eq!(regions.upper_lip.len(), 3);
        assert_eq!(regions.lower_lip.len(), 3);
        assert_eq!(regions.upper_lip[0], inner_lip[1] as usize);
        assert_eq!(regions.lower_lip[2], inner_lip[7] as usize);

        // Lid margins, never the shared eye corners (offsets 0 and 3).
        assert_eq!(regions.left_upper_eyelid.len(), 2);
        assert_eq!(regions.left_lower_eyelid.len(), 2);
        assert!(!regions.left_upper_eyelid.contains(&(left_eye[0] as usize)));
        assert!(!regions.left_upper_eyelid.contains(&(left_eye[3] as usize)));
    }

    #[test]
    fn default_flame_opposing_regions_are_disjoint() {
        // Regions that shared a vertex would measure an opening of zero and
        // report a permanently closed mouth or eye.
        let regions = FlameContactRegions::default_flame();
        for pair in [
            (&regions.upper_lip, &regions.lower_lip),
            (&regions.left_upper_eyelid, &regions.left_lower_eyelid),
            (&regions.right_upper_eyelid, &regions.right_lower_eyelid),
        ] {
            for idx in pair.0 {
                assert!(
                    !pair.1.contains(idx),
                    "vertex {idx} appears in both opposing regions"
                );
            }
        }
    }

    #[test]
    fn default_flame_regions_fit_a_standard_flame_mesh() {
        let regions = FlameContactRegions::default_flame();
        assert!(
            regions.validate(5023),
            "all regions must index into the 5023-vertex FLAME mesh"
        );
    }
}
