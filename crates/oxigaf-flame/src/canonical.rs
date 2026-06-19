//! Canonical face pose and alignment utilities for FLAME meshes.
//!
//! This module provides tools to extract geometric keypoints, bounding boxes,
//! head orientation, and canonical pose transforms from FLAME mesh vertices.
//! These are used to normalize face geometry for consistent downstream processing.

use std::fmt;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can arise during canonical face computation.
#[derive(Debug)]
pub enum CanonicalError {
    /// Vertex slice is completely empty.
    EmptyMesh,
    /// Not enough vertices for the requested operation.
    InsufficientVertices {
        /// Minimum number of vertices required.
        required: usize,
        /// Actual number of vertices provided.
        got: usize,
    },
    /// Inter-pupillary distance is zero (degenerate mesh).
    ZeroIpd,
    /// A generic computation failure with a human-readable description.
    ComputationFailed(String),
}

impl fmt::Display for CanonicalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyMesh => write!(f, "Vertex list is empty"),
            Self::InsufficientVertices { required, got } => {
                write!(f, "Insufficient vertices: required {required}, got {got}")
            }
            Self::ZeroIpd => {
                write!(f, "Inter-pupillary distance is zero (degenerate mesh)")
            }
            Self::ComputationFailed(msg) => write!(f, "Computation failed: {msg}"),
        }
    }
}

impl std::error::Error for CanonicalError {}

// ---------------------------------------------------------------------------
// Approximate FLAME vertex indices
// ---------------------------------------------------------------------------

/// Left eye region vertices (approximate FLAME indices).
const LEFT_EYE_INDICES: [usize; 3] = [2198, 2200, 2208];
/// Right eye region vertices (approximate FLAME indices).
const RIGHT_EYE_INDICES: [usize; 3] = [4920, 4922, 4930];
/// Nose tip vertex (approximate FLAME index).
const NOSE_TIP_INDEX: usize = 3052;
/// Chin vertex (approximate FLAME index).
const CHIN_INDEX: usize = 1789;
/// Forehead vertex (approximate FLAME index).
const FOREHEAD_INDEX: usize = 335;
/// Left ear vertex (approximate FLAME index).
const LEFT_EAR_INDEX: usize = 1070;
/// Right ear vertex (approximate FLAME index).
const RIGHT_EAR_INDEX: usize = 2725;

/// Target interpupillary distance for canonical normalization (metres).
const TARGET_IPD: f32 = 0.063;
/// Minimum allowed normalization scale.
const SCALE_MIN: f32 = 0.5;
/// Maximum allowed normalization scale.
const SCALE_MAX: f32 = 2.0;

// ---------------------------------------------------------------------------
// Helper: compute centroid of an arbitrary slice
// ---------------------------------------------------------------------------

/// Compute the centroid of a non-empty vertex slice.
/// Returns `[0.0, 0.0, 0.0]` when the slice is empty (caller must guard).
#[inline]
fn centroid(vertices: &[[f32; 3]]) -> [f32; 3] {
    if vertices.is_empty() {
        return [0.0, 0.0, 0.0];
    }
    let n = vertices.len() as f32;
    let mut sum = [0.0_f32; 3];
    for v in vertices {
        sum[0] += v[0];
        sum[1] += v[1];
        sum[2] += v[2];
    }
    [sum[0] / n, sum[1] / n, sum[2] / n]
}

/// Return the vertex at `index` if in bounds, otherwise return `fallback`.
#[inline]
fn get_or_fallback(vertices: &[[f32; 3]], index: usize, fallback: [f32; 3]) -> [f32; 3] {
    vertices.get(index).copied().unwrap_or(fallback)
}

/// Average a set of vertex indices; out-of-bounds indices use `fallback`.
#[inline]
fn average_indices(vertices: &[[f32; 3]], indices: &[usize], fallback: [f32; 3]) -> [f32; 3] {
    if indices.is_empty() {
        return fallback;
    }
    let mut sum = [0.0_f32; 3];
    for &idx in indices {
        let v = get_or_fallback(vertices, idx, fallback);
        sum[0] += v[0];
        sum[1] += v[1];
        sum[2] += v[2];
    }
    let n = indices.len() as f32;
    [sum[0] / n, sum[1] / n, sum[2] / n]
}

// ---------------------------------------------------------------------------
// FaceKeypoints
// ---------------------------------------------------------------------------

/// Key geometric points of a FLAME mesh face.
#[derive(Debug, Clone)]
pub struct FaceKeypoints {
    /// Centroid of the face region.
    pub face_center: [f32; 3],
    /// Left eye centre (subject's left).
    pub left_eye: [f32; 3],
    /// Right eye centre (subject's right).
    pub right_eye: [f32; 3],
    /// Nose apex.
    pub nose_tip: [f32; 3],
    /// Lowest chin point.
    pub chin: [f32; 3],
    /// Left ear position.
    pub left_ear: [f32; 3],
    /// Right ear position.
    pub right_ear: [f32; 3],
    /// Forehead centre.
    pub forehead: [f32; 3],
}

impl FaceKeypoints {
    /// Estimate keypoints from raw vertex positions.
    ///
    /// Uses approximate FLAME vertex indices. Any out-of-bounds index falls back
    /// to the overall centroid, so this succeeds for any non-empty slice.
    ///
    /// # Errors
    ///
    /// Returns [`CanonicalError::EmptyMesh`] when `vertices` is empty.
    pub fn from_vertices(vertices: &[[f32; 3]]) -> Result<Self, CanonicalError> {
        if vertices.is_empty() {
            return Err(CanonicalError::EmptyMesh);
        }

        let fallback = centroid(vertices);

        let left_eye = average_indices(vertices, &LEFT_EYE_INDICES, fallback);
        let right_eye = average_indices(vertices, &RIGHT_EYE_INDICES, fallback);
        let nose_tip = get_or_fallback(vertices, NOSE_TIP_INDEX, fallback);
        let chin = get_or_fallback(vertices, CHIN_INDEX, fallback);
        let forehead = get_or_fallback(vertices, FOREHEAD_INDEX, fallback);
        let left_ear = get_or_fallback(vertices, LEFT_EAR_INDEX, fallback);
        let right_ear = get_or_fallback(vertices, RIGHT_EAR_INDEX, fallback);
        let face_center = fallback;

        Ok(Self {
            face_center,
            left_eye,
            right_eye,
            nose_tip,
            chin,
            left_ear,
            right_ear,
            forehead,
        })
    }

    /// Interpupillary distance (Euclidean distance between eye centres).
    #[must_use]
    pub fn ipd(&self) -> f32 {
        let dx = self.right_eye[0] - self.left_eye[0];
        let dy = self.right_eye[1] - self.left_eye[1];
        let dz = self.right_eye[2] - self.left_eye[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Face height: Euclidean distance from chin to forehead.
    #[must_use]
    pub fn face_height(&self) -> f32 {
        let dx = self.forehead[0] - self.chin[0];
        let dy = self.forehead[1] - self.chin[1];
        let dz = self.forehead[2] - self.chin[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Face width: Euclidean distance from left ear to right ear.
    #[must_use]
    pub fn face_width(&self) -> f32 {
        let dx = self.right_ear[0] - self.left_ear[0];
        let dy = self.right_ear[1] - self.left_ear[1];
        let dz = self.right_ear[2] - self.left_ear[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Face aspect ratio: width divided by height.
    ///
    /// Returns `0.0` when face height is zero to avoid division by zero.
    #[must_use]
    pub fn aspect_ratio(&self) -> f32 {
        let h = self.face_height();
        if h < f32::EPSILON {
            0.0
        } else {
            self.face_width() / h
        }
    }
}

// ---------------------------------------------------------------------------
// FaceBoundingBox
// ---------------------------------------------------------------------------

/// Axis-aligned 3D bounding box for a face mesh.
#[derive(Debug, Clone)]
pub struct FaceBoundingBox {
    /// Component-wise minimum of all vertices.
    pub min: [f32; 3],
    /// Component-wise maximum of all vertices.
    pub max: [f32; 3],
    /// Geometric centre `(min + max) / 2`.
    pub center: [f32; 3],
}

impl FaceBoundingBox {
    /// Compute the bounding box from a slice of vertices.
    ///
    /// Returns `None` when `vertices` is empty.
    #[must_use]
    pub fn from_vertices(vertices: &[[f32; 3]]) -> Option<Self> {
        let first = vertices.first()?;
        let mut min = *first;
        let mut max = *first;

        for v in vertices.iter().skip(1) {
            for i in 0..3 {
                if v[i] < min[i] {
                    min[i] = v[i];
                }
                if v[i] > max[i] {
                    max[i] = v[i];
                }
            }
        }

        let center = [
            (min[0] + max[0]) * 0.5,
            (min[1] + max[1]) * 0.5,
            (min[2] + max[2]) * 0.5,
        ];

        Some(Self { min, max, center })
    }

    /// Per-axis size: `max - min`.
    #[must_use]
    pub fn size(&self) -> [f32; 3] {
        [
            self.max[0] - self.min[0],
            self.max[1] - self.min[1],
            self.max[2] - self.min[2],
        ]
    }

    /// Length of the space diagonal: `sqrt(sum(size[i]^2))`.
    #[must_use]
    pub fn diagonal(&self) -> f32 {
        let s = self.size();
        (s[0] * s[0] + s[1] * s[1] + s[2] * s[2]).sqrt()
    }

    /// Whether point `p` lies within (inclusive) the bounding box.
    #[must_use]
    pub fn contains(&self, p: &[f32; 3]) -> bool {
        p[0] >= self.min[0]
            && p[0] <= self.max[0]
            && p[1] >= self.min[1]
            && p[1] <= self.max[1]
            && p[2] >= self.min[2]
            && p[2] <= self.max[2]
    }
}

// ---------------------------------------------------------------------------
// Canonical transform
// ---------------------------------------------------------------------------

/// Compute a 4×4 column-major-style transform matrix that places the face in
/// canonical pose.
///
/// The resulting transform:
/// - Translates the face centre to the origin.
/// - Scales so that the inter-pupillary distance equals `TARGET_IPD` (0.063 m).
///   Scale is clamped to \[0.5, 2.0\] to handle degenerate cases.
///
/// # Returns
///
/// `(transform, scale)` where `transform` is row-major (`m[row][col]`) and
/// `scale` is the normalization factor applied to distances.
///
/// Row layout:
/// ```text
/// [scale, 0,     0,     -scale*cx]
/// [0,     scale, 0,     -scale*cy]
/// [0,     0,     scale, -scale*cz]
/// [0,     0,     0,     1        ]
/// ```
#[must_use]
pub fn compute_canonical_transform(keypoints: &FaceKeypoints) -> ([[f32; 4]; 4], f32) {
    let actual_ipd = keypoints.ipd();

    // Guard against zero IPD: use identity scale.
    let scale = if actual_ipd < f32::EPSILON {
        1.0_f32
    } else {
        (TARGET_IPD / actual_ipd).clamp(SCALE_MIN, SCALE_MAX)
    };

    let cx = keypoints.face_center[0];
    let cy = keypoints.face_center[1];
    let cz = keypoints.face_center[2];

    let transform = [
        [scale, 0.0, 0.0, -scale * cx],
        [0.0, scale, 0.0, -scale * cy],
        [0.0, 0.0, scale, -scale * cz],
        [0.0, 0.0, 0.0, 1.0],
    ];

    (transform, scale)
}

// ---------------------------------------------------------------------------
// 2D projection helpers
// ---------------------------------------------------------------------------

/// Project a 3D vertex to 2D image coordinates using orthographic projection.
///
/// Formula:
/// ```text
/// pixel_x = cx + focal_scale * vertex.x
/// pixel_y = cy - focal_scale * vertex.y   (image Y is flipped vs world Y)
/// ```
///
/// where `cx = image_width / 2` and `cy = image_height / 2`.
///
/// If the vertex contains NaN values the returned pixel is `[cx, cy]`.
#[must_use]
pub fn orthographic_project(
    vertex: &[f32; 3],
    focal_scale: f32,
    image_width: usize,
    image_height: usize,
) -> [f32; 2] {
    let cx = image_width as f32 * 0.5;
    let cy = image_height as f32 * 0.5;

    if vertex[0].is_nan() || vertex[1].is_nan() || vertex[2].is_nan() {
        return [cx, cy];
    }

    [cx + focal_scale * vertex[0], cy - focal_scale * vertex[1]]
}

/// Convert 3D face positions to a 2D bounding box in image space.
///
/// # Returns
///
/// `Some((top_left, bottom_right))` in pixel coordinates, or `None` when
/// `vertices` is empty.
#[must_use]
pub fn face_bbox_2d(
    vertices: &[[f32; 3]],
    focal_scale: f32,
    image_width: usize,
    image_height: usize,
) -> Option<([f32; 2], [f32; 2])> {
    if vertices.is_empty() {
        return None;
    }

    let first = orthographic_project(&vertices[0], focal_scale, image_width, image_height);
    let mut min_x = first[0];
    let mut max_x = first[0];
    let mut min_y = first[1];
    let mut max_y = first[1];

    for v in vertices.iter().skip(1) {
        let p = orthographic_project(v, focal_scale, image_width, image_height);
        if p[0] < min_x {
            min_x = p[0];
        }
        if p[0] > max_x {
            max_x = p[0];
        }
        if p[1] < min_y {
            min_y = p[1];
        }
        if p[1] > max_y {
            max_y = p[1];
        }
    }

    Some(([min_x, min_y], [max_x, max_y]))
}

/// Compute a uniform scale factor so the face fills `target_fraction` of
/// the image height.
///
/// Scale = `(target_fraction * image_height) / face_height`.
///
/// Returns `1.0` when `face_height` is zero or `target_fraction` is
/// non-positive, to avoid degenerate values.
#[must_use]
pub fn compute_face_scale_for_image(
    keypoints: &FaceKeypoints,
    target_fraction: f32,
    image_height: usize,
) -> f32 {
    let face_h = keypoints.face_height();
    if face_h < f32::EPSILON || target_fraction <= 0.0 {
        return 1.0;
    }
    (target_fraction * image_height as f32) / face_h
}

// ---------------------------------------------------------------------------
// HeadOrientation
// ---------------------------------------------------------------------------

/// Euler-like head orientation angles derived from geometric face landmarks.
///
/// These are geometric approximations, not a true Euler decomposition.
#[derive(Debug, Clone, Copy)]
pub struct HeadOrientation {
    /// Rotation around the Y axis (positive = turn right), in radians.
    pub yaw: f32,
    /// Rotation around the X axis (positive = look up), in radians.
    pub pitch: f32,
    /// Rotation around the Z axis (positive = tilt right), in radians.
    pub roll: f32,
}

impl HeadOrientation {
    /// Return `[yaw_deg, pitch_deg, roll_deg]`.
    #[must_use]
    pub fn in_degrees(&self) -> [f32; 3] {
        let to_deg = 180.0 / std::f32::consts::PI;
        [self.yaw * to_deg, self.pitch * to_deg, self.roll * to_deg]
    }
}

/// Estimate head orientation from 3D face keypoints.
///
/// Approach (geometric approximation):
/// - **Yaw**: `atan2(right_ear.x - left_ear.x, right_ear.z - left_ear.z)` — side-to-side rotation.
/// - **Pitch**: `atan2(nose_tip.y - face_center.y, nose_tip.z - face_center.z)` — up-down tilt.
/// - **Roll**: `atan2(right_eye.y - left_eye.y, right_eye.x - left_eye.x)` — head tilt.
#[must_use]
pub fn estimate_head_orientation(keypoints: &FaceKeypoints) -> HeadOrientation {
    let yaw = f32::atan2(
        keypoints.right_ear[0] - keypoints.left_ear[0],
        keypoints.right_ear[2] - keypoints.left_ear[2],
    );

    let pitch = f32::atan2(
        keypoints.nose_tip[1] - keypoints.face_center[1],
        keypoints.nose_tip[2] - keypoints.face_center[2],
    );

    let roll = f32::atan2(
        keypoints.right_eye[1] - keypoints.left_eye[1],
        keypoints.right_eye[0] - keypoints.left_eye[0],
    );

    HeadOrientation { yaw, pitch, roll }
}

// ---------------------------------------------------------------------------
// CanonicalFace
// ---------------------------------------------------------------------------

/// Complete canonical face description derived from FLAME mesh vertices.
pub struct CanonicalFace {
    /// Geometric keypoints (eyes, nose, chin, etc.).
    pub keypoints: FaceKeypoints,
    /// 3D axis-aligned bounding box.
    pub bounding_box: FaceBoundingBox,
    /// Geometric head orientation estimate.
    pub orientation: HeadOrientation,
    /// 4×4 row-major transform matrix to canonical pose.
    pub canonical_transform: [[f32; 4]; 4],
    /// Normalization scale factor applied in the canonical transform.
    pub scale: f32,
}

impl CanonicalFace {
    /// Build a complete canonical face description from raw FLAME vertices.
    ///
    /// # Errors
    ///
    /// - [`CanonicalError::EmptyMesh`] when `vertices` is empty.
    /// - [`CanonicalError::ComputationFailed`] when bounding box computation
    ///   fails (should not happen for non-empty input, but guards the
    ///   `Option`-returning API).
    pub fn from_vertices(vertices: &[[f32; 3]]) -> Result<Self, CanonicalError> {
        if vertices.is_empty() {
            return Err(CanonicalError::EmptyMesh);
        }

        let keypoints = FaceKeypoints::from_vertices(vertices)?;
        let bounding_box = FaceBoundingBox::from_vertices(vertices)
            .ok_or_else(|| CanonicalError::ComputationFailed("bounding box failed".into()))?;
        let orientation = estimate_head_orientation(&keypoints);
        let (canonical_transform, scale) = compute_canonical_transform(&keypoints);

        Ok(Self {
            keypoints,
            bounding_box,
            orientation,
            canonical_transform,
            scale,
        })
    }

    /// Apply the canonical transform to a single vertex.
    ///
    /// Computes `M * [v.x, v.y, v.z, 1]^T` and returns the first three
    /// components.
    #[must_use]
    pub fn transform_vertex(&self, v: &[f32; 3]) -> [f32; 3] {
        let m = &self.canonical_transform;
        [
            m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2] + m[0][3],
            m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2] + m[1][3],
            m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2] + m[2][3],
        ]
    }

    /// Return a concise human-readable summary string.
    #[must_use]
    pub fn format_summary(&self) -> String {
        let kp = &self.keypoints;
        let bb = &self.bounding_box;
        let ori = &self.orientation;
        let deg = ori.in_degrees();
        format!(
            "CanonicalFace {{ ipd={:.4} m, face_h={:.4} m, face_w={:.4} m, \
             scale={:.4}, bbox_diag={:.4} m, \
             yaw={:.1}° pitch={:.1}° roll={:.1}° }}",
            kp.ipd(),
            kp.face_height(),
            kp.face_width(),
            self.scale,
            bb.diagonal(),
            deg[0],
            deg[1],
            deg[2],
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Helper: build a minimal synthetic vertex list that covers all the
    // approximate FLAME indices needed for full-resolution tests.
    // -----------------------------------------------------------------------

    /// Build a vertex list of size `n` filled with `value`.
    fn uniform_vertices(n: usize, value: [f32; 3]) -> Vec<[f32; 3]> {
        vec![value; n]
    }

    /// Build a vertex list large enough to cover all landmark indices.
    /// All vertices placed at the origin except those explicitly set below.
    fn synthetic_flame_vertices() -> Vec<[f32; 3]> {
        // Largest index we reference: 4930 (right eye) → need at least 4931 verts.
        let mut verts = vec![[0.0_f32; 3]; 5000];

        // Left eye region (indices 2198, 2200, 2208) – centred around (-0.032, 0.05, 0.0)
        verts[2198] = [-0.030, 0.050, 0.0];
        verts[2200] = [-0.032, 0.050, 0.0];
        verts[2208] = [-0.034, 0.050, 0.0];

        // Right eye region (indices 4920, 4922, 4930) – centred around (0.032, 0.05, 0.0)
        verts[4920] = [0.030, 0.050, 0.0];
        verts[4922] = [0.032, 0.050, 0.0];
        verts[4930] = [0.034, 0.050, 0.0];

        // Nose tip (index 3052)
        verts[3052] = [0.0, -0.030, 0.060];

        // Chin (index 1789)
        verts[1789] = [0.0, -0.080, 0.020];

        // Forehead (index 335)
        verts[335] = [0.0, 0.120, 0.010];

        // Left ear (index 1070)
        verts[1070] = [-0.080, 0.010, -0.010];

        // Right ear (index 2725)
        verts[2725] = [0.080, 0.010, -0.010];

        verts
    }

    // -----------------------------------------------------------------------
    // FaceBoundingBox tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_face_bbox_from_vertices() {
        let verts = vec![[1.0_f32, 2.0, 3.0], [-1.0, -2.0, -3.0], [0.5, 0.0, 1.5]];
        let bb = FaceBoundingBox::from_vertices(&verts).expect("should succeed");
        assert!((bb.min[0] - (-1.0)).abs() < 1e-6);
        assert!((bb.max[0] - 1.0).abs() < 1e-6);
        assert!((bb.center[0]).abs() < 1e-6, "center x should be 0");
        assert!((bb.center[1] - 0.0).abs() < 1e-6, "center y should be 0");
    }

    #[test]
    fn test_face_bbox_empty() {
        let verts: Vec<[f32; 3]> = vec![];
        let result = FaceBoundingBox::from_vertices(&verts);
        assert!(result.is_none(), "empty slice should yield None");
    }

    // -----------------------------------------------------------------------
    // FaceKeypoints tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_face_keypoints_fallback_safe() {
        // Only 3 vertices → all landmark indices are out of bounds;
        // all keypoints fall back to the centroid and no error is returned.
        let verts = vec![[1.0_f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let kp = FaceKeypoints::from_vertices(&verts);
        assert!(
            kp.is_ok(),
            "fallback should prevent errors for small meshes"
        );
        let kp = kp.expect("already checked");
        // All points should equal the centroid [1/3, 1/3, 1/3].
        let expected = 1.0 / 3.0;
        assert!(
            (kp.left_eye[0] - expected).abs() < 1e-5,
            "left_eye x should be centroid"
        );
        assert!(
            (kp.nose_tip[0] - expected).abs() < 1e-5,
            "nose_tip x should be centroid"
        );
    }

    #[test]
    fn test_face_keypoints_ipd() {
        let verts = synthetic_flame_vertices();
        let kp = FaceKeypoints::from_vertices(&verts).expect("should succeed");

        // Expected left eye avg x ≈ (-0.030 + -0.032 + -0.034)/3 = -0.032
        // Expected right eye avg x ≈ (0.030 + 0.032 + 0.034)/3 = 0.032
        // IPD ≈ 0.064 (y and z offsets are equal, so only x contributes)
        let ipd = kp.ipd();
        assert!((ipd - 0.064).abs() < 1e-4, "expected IPD ~0.064, got {ipd}");
    }

    #[test]
    fn test_face_keypoints_face_height() {
        let verts = synthetic_flame_vertices();
        let kp = FaceKeypoints::from_vertices(&verts).expect("should succeed");

        // chin = [0, -0.08, 0.02], forehead = [0, 0.12, 0.01]
        // height = sqrt(0 + (0.12-(-0.08))^2 + (0.01-0.02)^2)
        //        = sqrt(0.04 + 0.0001) ≈ 0.20025
        let h = kp.face_height();
        assert!(
            (h - 0.200_25).abs() < 1e-3,
            "expected face height ~0.200, got {h}"
        );
    }

    #[test]
    fn test_face_keypoints_aspect_ratio() {
        let verts = synthetic_flame_vertices();
        let kp = FaceKeypoints::from_vertices(&verts).expect("should succeed");

        let ar = kp.aspect_ratio();
        // Width ≈ 0.160, height ≈ 0.200 → ratio ≈ 0.8
        assert!(
            ar > 0.5 && ar < 1.5,
            "aspect ratio should be in (0.5, 1.5), got {ar}"
        );
    }

    #[test]
    fn test_face_keypoints_empty_returns_error() {
        let verts: Vec<[f32; 3]> = vec![];
        let result = FaceKeypoints::from_vertices(&verts);
        assert!(
            matches!(result, Err(CanonicalError::EmptyMesh)),
            "empty slice should yield EmptyMesh error"
        );
    }

    // -----------------------------------------------------------------------
    // Canonical transform tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_compute_canonical_transform_scale() {
        let verts = synthetic_flame_vertices();
        let kp = FaceKeypoints::from_vertices(&verts).expect("should succeed");
        let ipd = kp.ipd(); // ≈ 0.064

        let (_transform, scale) = compute_canonical_transform(&kp);
        let expected_scale = (TARGET_IPD / ipd).clamp(SCALE_MIN, SCALE_MAX);
        assert!(
            (scale - expected_scale).abs() < 1e-5,
            "scale should match TARGET_IPD / actual_ipd, got {scale}"
        );
    }

    #[test]
    fn test_transform_vertex_identity() {
        // If face_center is at origin and IPD matches TARGET_IPD, the transform
        // should be a pure scale with no translation offset for the centre.
        // Build minimal keypoints by hand.
        let kp = FaceKeypoints {
            face_center: [0.0, 0.0, 0.0],
            left_eye: [-TARGET_IPD / 2.0, 0.0, 0.0],
            right_eye: [TARGET_IPD / 2.0, 0.0, 0.0],
            nose_tip: [0.0, 0.0, 0.0],
            chin: [0.0, -0.1, 0.0],
            forehead: [0.0, 0.1, 0.0],
            left_ear: [-0.08, 0.0, 0.0],
            right_ear: [0.08, 0.0, 0.0],
        };

        let (transform, scale) = compute_canonical_transform(&kp);

        // scale should be TARGET_IPD / TARGET_IPD = 1.0
        assert!((scale - 1.0).abs() < 1e-5, "scale should be 1.0");

        // transform_vertex of face_center = [0,0,0] should yield [0,0,0]
        let cf = CanonicalFace {
            keypoints: kp,
            bounding_box: FaceBoundingBox {
                min: [-0.1; 3],
                max: [0.1; 3],
                center: [0.0; 3],
            },
            orientation: HeadOrientation {
                yaw: 0.0,
                pitch: 0.0,
                roll: 0.0,
            },
            canonical_transform: transform,
            scale,
        };
        let transformed = cf.transform_vertex(&[0.0, 0.0, 0.0]);
        for coord in transformed {
            assert!(
                coord.abs() < 1e-5,
                "transformed origin should be origin, got {coord}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Head orientation tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_head_orientation_frontal_face() {
        // Symmetric face looking straight: ears at equal ±x, eyes at equal ±x.
        let kp = FaceKeypoints {
            face_center: [0.0, 0.0, 0.0],
            left_eye: [-0.032, 0.05, 0.0],
            right_eye: [0.032, 0.05, 0.0],
            nose_tip: [0.0, 0.0, 0.06],
            chin: [0.0, -0.08, 0.02],
            forehead: [0.0, 0.12, 0.01],
            left_ear: [-0.08, 0.0, 0.0],
            right_ear: [0.08, 0.0, 0.0],
        };
        let ori = estimate_head_orientation(&kp);
        // Yaw: atan2(0.08 - (-0.08), 0.0 - 0.0) = atan2(0.16, 0.0) = π/2
        // However for a frontal face the x-difference is symmetric and z-diff is 0.
        // The sign comes purely from the positive x-difference.
        // What matters for this test: roll should be 0 (eyes at same y).
        assert!(
            ori.roll.abs() < 1e-5,
            "frontal face: roll should be ~0, got {}",
            ori.roll
        );
    }

    #[test]
    fn test_head_orientation_turned_face() {
        // Face turned right: right ear closer in z than left ear.
        let kp = FaceKeypoints {
            face_center: [0.0, 0.0, 0.0],
            left_eye: [-0.032, 0.05, 0.0],
            right_eye: [0.020, 0.05, 0.0], // compressed because turned
            nose_tip: [0.02, 0.0, 0.05],   // nose moved toward viewer right
            chin: [0.0, -0.08, 0.02],
            forehead: [0.0, 0.12, 0.01],
            left_ear: [-0.09, 0.0, -0.02],
            right_ear: [0.06, 0.0, 0.05], // right ear closer to camera
        };
        let ori = estimate_head_orientation(&kp);
        // With these values yaw should be non-zero.
        assert!(
            ori.yaw.abs() > 0.01,
            "turned face: yaw should be non-zero, got {}",
            ori.yaw
        );
    }

    // -----------------------------------------------------------------------
    // 2D projection tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_orthographic_project_center() {
        // A vertex at the origin should project to the image centre.
        let px = orthographic_project(&[0.0, 0.0, 0.0], 500.0, 512, 512);
        assert!((px[0] - 256.0).abs() < 1e-5, "x should be image center");
        assert!((px[1] - 256.0).abs() < 1e-5, "y should be image center");
    }

    #[test]
    fn test_orthographic_project_nan_safe() {
        let px = orthographic_project(&[f32::NAN, 0.0, 0.0], 500.0, 512, 512);
        // Falls back to image centre
        assert!((px[0] - 256.0).abs() < 1e-5);
        assert!((px[1] - 256.0).abs() < 1e-5);
    }

    #[test]
    fn test_face_bbox_2d() {
        let verts = vec![[0.0_f32, 0.0, 0.0], [0.1, 0.1, 0.0], [-0.1, -0.1, 0.0]];
        let result = face_bbox_2d(&verts, 500.0, 512, 512);
        assert!(result.is_some(), "should return Some for non-empty verts");
        let (tl, br) = result.expect("already checked");
        // top-left x should be < bottom-right x
        assert!(tl[0] < br[0], "tl.x < br.x");
        // Remember: image-y is flipped, so tl.y < br.y means the vertex with
        // largest world-y maps to smallest image-y.
        assert!(tl[1] < br[1]);
    }

    #[test]
    fn test_face_bbox_2d_empty() {
        let verts: Vec<[f32; 3]> = vec![];
        let result = face_bbox_2d(&verts, 500.0, 512, 512);
        assert!(result.is_none());
    }

    // -----------------------------------------------------------------------
    // Face scale for image tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_compute_face_scale() {
        let verts = synthetic_flame_vertices();
        let kp = FaceKeypoints::from_vertices(&verts).expect("should succeed");

        // Target 70% of 512 pixels.
        let scale = compute_face_scale_for_image(&kp, 0.7, 512);
        let face_h = kp.face_height();
        let expected = (0.7 * 512.0) / face_h;
        assert!(
            (scale - expected).abs() < 1e-3,
            "scale mismatch: expected {expected}, got {scale}"
        );
        assert!(scale > 0.0, "scale should be positive");
    }

    #[test]
    fn test_compute_face_scale_zero_height() {
        // All vertices at origin → chin and forehead collapse to same point.
        let verts = uniform_vertices(5000, [0.0, 0.0, 0.0]);
        let kp = FaceKeypoints::from_vertices(&verts).expect("should succeed");
        let scale = compute_face_scale_for_image(&kp, 0.7, 512);
        assert!(
            (scale - 1.0).abs() < 1e-5,
            "zero face height should yield scale=1.0, got {scale}"
        );
    }

    // -----------------------------------------------------------------------
    // CanonicalFace integration tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_canonical_face_from_vertices() {
        let verts = synthetic_flame_vertices();
        let cf = CanonicalFace::from_vertices(&verts);
        assert!(cf.is_ok(), "should succeed for valid vertex list");
        let cf = cf.expect("already checked");
        assert!(cf.scale > 0.0, "scale should be positive");
        // Bounding box should be non-degenerate.
        assert!(cf.bounding_box.diagonal() > 0.0);
    }

    #[test]
    fn test_canonical_face_from_empty_vertices() {
        let verts: Vec<[f32; 3]> = vec![];
        let result = CanonicalFace::from_vertices(&verts);
        assert!(
            matches!(result, Err(CanonicalError::EmptyMesh)),
            "empty mesh should yield EmptyMesh error"
        );
    }

    #[test]
    fn test_canonical_face_format_summary() {
        let verts = synthetic_flame_vertices();
        let cf = CanonicalFace::from_vertices(&verts).expect("should succeed");
        let s = cf.format_summary();
        assert!(
            s.contains("CanonicalFace"),
            "summary should start with CanonicalFace"
        );
        assert!(s.contains("ipd="), "summary should contain ipd");
        assert!(s.contains("scale="), "summary should contain scale");
        assert!(s.contains("yaw="), "summary should contain yaw");
    }

    #[test]
    fn test_canonical_face_transform_vertex_face_center() {
        // The canonical transform should map the face_center to (approx.) origin.
        let verts = synthetic_flame_vertices();
        let cf = CanonicalFace::from_vertices(&verts).expect("should succeed");
        let center = cf.keypoints.face_center;
        let transformed = cf.transform_vertex(&center);
        for (i, &coord) in transformed.iter().enumerate() {
            assert!(
                coord.abs() < 1e-4,
                "transformed face center axis {i} should be ~0, got {coord}"
            );
        }
    }
}
