//! Face normalization: canonical pose and scale utilities for FLAME head meshes.
//!
//! This module provides mesh-level normalization into a standard canonical pose and scale,
//! distinct from `canonical.rs` (which handles camera transforms). Use these utilities to
//! produce consistent training data, compare faces across poses, and prepare inputs for
//! diffusion conditioning pipelines.
//!
//! # Overview
//!
//! - [`normalize_mesh`]: Full pipeline — center, rotate, scale a mesh to canonical form.
//! - [`align_eye_line`]: Rotate mesh so the eye line is horizontal.
//! - [`align_frontal`]: PCA-based frontal alignment (nose toward +Z).
//! - [`inter_pupil_distance`]: Estimate IPD from eye landmark vertices.
//! - [`pca_axes`]: Compute principal axes via power-iteration deflation.
//! - [`rotation_align`]: Build a rotation that maps one unit vector to another.
//! - [`axis_angle_rotation`]: Build a rotation matrix from an axis and angle.
//! - [`NormTransform`]: Composable rigid + scale transform.

use thiserror::Error;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can arise during face normalization.
#[derive(Debug, Error)]
pub enum FaceNormError {
    /// Mesh has no vertices.
    #[error("Empty mesh: {0}")]
    EmptyMesh(String),
    /// Landmark index is out of range for the given mesh.
    #[error("Invalid landmark index {idx} for mesh with {n_verts} vertices")]
    InvalidLandmark { idx: usize, n_verts: usize },
    /// Matrix is singular and cannot be inverted.
    #[error("Singular matrix: cannot normalize")]
    SingularMatrix,
    /// A parameter value is invalid.
    #[error("Invalid parameter: {0}")]
    InvalidParam(String),
}

// ---------------------------------------------------------------------------
// Math primitives  (all private; public aliases are provided where the spec demands)
// ---------------------------------------------------------------------------

/// Compute dot product of two 3-vectors.
#[inline]
#[must_use]
pub fn vec3_dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Compute cross product of two 3-vectors.
#[inline]
#[must_use]
pub fn vec3_cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Normalize a 3-vector to unit length. Returns the zero vector if already zero.
#[inline]
#[must_use]
pub fn vec3_normalize(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len < 1e-30 {
        return [0.0, 0.0, 0.0];
    }
    [v[0] / len, v[1] / len, v[2] / len]
}

/// Compute the angle (radians) between two 3-vectors.
#[inline]
#[must_use]
pub fn vec3_angle(a: [f32; 3], b: [f32; 3]) -> f32 {
    let na = vec3_normalize(a);
    let nb = vec3_normalize(b);
    let d = vec3_dot(na, nb).clamp(-1.0, 1.0);
    d.acos()
}

/// L2 length of a 3-vector.
#[inline]
fn vec3_len(v: [f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// Scale a 3-vector by a scalar.
#[inline]
fn vec3_scale(v: [f32; 3], s: f32) -> [f32; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}

/// Add two 3-vectors.
#[inline]
fn vec3_add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// Subtract two 3-vectors.
#[inline]
fn vec3_sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Multiply two 3×3 rotation matrices (row-major): result = a * b.
#[inline]
#[must_use]
pub fn mat3_mul(a: [[f32; 3]; 3], b: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let mut out = [[0.0_f32; 3]; 3];
    for row in 0..3 {
        for col in 0..3 {
            out[row][col] = a[row][0] * b[0][col] + a[row][1] * b[1][col] + a[row][2] * b[2][col];
        }
    }
    out
}

/// Transpose a 3×3 matrix (row-major).
#[inline]
#[must_use]
pub fn mat3_transpose(m: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ]
}

/// Apply a 3×3 matrix to a 3-vector: result = m * v.
#[inline]
#[must_use]
pub fn mat3_apply(m: [[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

/// 3×3 identity matrix.
#[inline]
fn mat3_identity() -> [[f32; 3]; 3] {
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
}

/// Determinant of a 3×3 matrix (used for sign checks).
#[inline]
fn mat3_det(m: [[f32; 3]; 3]) -> f32 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// A 3D rigid + uniform scale transform: `y = scale * R * x + t`.
///
/// The rotation matrix is stored row-major: `rotation[row][col]`.
#[derive(Debug, Clone, PartialEq)]
pub struct NormTransform {
    /// Row-major 3×3 rotation matrix.
    pub rotation: [[f32; 3]; 3],
    /// Translation vector (applied after rotation and scale).
    pub translation: [f32; 3],
    /// Uniform scale factor applied before rotation.
    pub scale: f32,
}

impl NormTransform {
    /// Identity transform: no rotation, no translation, scale = 1.
    #[must_use]
    pub fn identity() -> Self {
        Self {
            rotation: mat3_identity(),
            translation: [0.0, 0.0, 0.0],
            scale: 1.0,
        }
    }

    /// Apply this transform to a single point: `y = scale * R * x + t`.
    #[inline]
    #[must_use]
    pub fn apply(&self, point: &[f32; 3]) -> [f32; 3] {
        let rotated = mat3_apply(self.rotation, *point);
        let scaled = vec3_scale(rotated, self.scale);
        vec3_add(scaled, self.translation)
    }

    /// Apply this transform to a slice of points.
    #[must_use]
    pub fn apply_batch(&self, points: &[[f32; 3]]) -> Vec<[f32; 3]> {
        points.iter().map(|p| self.apply(p)).collect()
    }

    /// Compute the inverse transform.
    ///
    /// For `y = scale * R * x + t`, the inverse is:
    /// `x = (1/scale) * R^T * (y - t)`
    ///
    /// Which corresponds to: `scale' = 1/scale`, `R' = R^T`, `t' = -(1/scale) * R^T * t`.
    ///
    /// Returns `SingularMatrix` if scale ≈ 0.
    ///
    /// # Errors
    ///
    /// Returns [`FaceNormError::SingularMatrix`] if `self.scale` is near zero.
    pub fn inverse(&self) -> Result<NormTransform, FaceNormError> {
        if self.scale.abs() < 1e-30 {
            return Err(FaceNormError::SingularMatrix);
        }
        let inv_scale = 1.0 / self.scale;
        let r_t = mat3_transpose(self.rotation);
        // t' = -(1/scale) * R^T * t
        let rt_t = mat3_apply(r_t, self.translation);
        let new_t = vec3_scale(rt_t, -inv_scale);
        Ok(NormTransform {
            rotation: r_t,
            translation: new_t,
            scale: inv_scale,
        })
    }

    /// Compose: first apply `self`, then apply `other`.
    ///
    /// If `y = s_self * R_self * x + t_self` and `z = s_other * R_other * y + t_other`, then:
    /// `z = (s_self * s_other) * (R_other * R_self) * x + (s_other * R_other * t_self + t_other)`
    #[must_use]
    pub fn compose(&self, other: &NormTransform) -> NormTransform {
        let new_scale = self.scale * other.scale;
        let new_rotation = mat3_mul(other.rotation, self.rotation);
        // new_t = s_other * R_other * t_self + t_other
        let r_other_t_self = mat3_apply(other.rotation, self.translation);
        let scaled_part = vec3_scale(r_other_t_self, other.scale);
        let new_t = vec3_add(scaled_part, other.translation);
        NormTransform {
            rotation: new_rotation,
            translation: new_t,
            scale: new_scale,
        }
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// How to determine the centering point of the normalization.
pub enum CenterMode {
    /// Center on the vertex centroid (mean of all vertices).
    Centroid,
    /// Center on the mean of the given landmark vertex indices.
    LandmarkMean(Vec<usize>),
    /// Do not shift (origin stays at origin).
    Origin,
}

/// How to rotationally align the mesh.
pub enum AlignMode {
    /// No rotation applied.
    None,
    /// Rotate so the nose points toward +Z (frontal face, PCA-based).
    FrontalFace,
    /// Rotate so the eye line is horizontal (roll correction only).
    EyeLine,
}

/// Configuration for the [`normalize_mesh`] pipeline.
pub struct NormConfig {
    /// Scale so that inter-pupil distance equals this value (default 1.0).
    pub target_scale: f32,
    /// How to center the mesh.
    pub center_on: CenterMode,
    /// How to rotationally align the mesh.
    pub align_to: AlignMode,
}

impl Default for NormConfig {
    fn default() -> Self {
        Self {
            target_scale: 1.0,
            center_on: CenterMode::Centroid,
            align_to: AlignMode::None,
        }
    }
}

/// Result of a [`normalize_mesh`] call.
pub struct NormResult {
    /// The transform that was applied.
    pub transform: NormTransform,
    /// The normalized vertex positions.
    pub normalized_vertices: Vec<[f32; 3]>,
    /// Estimated inter-pupil distance (before normalization), or 0.0 if unavailable.
    pub inter_pupil_distance: f32,
    /// Estimated face scale (bounding-box diagonal) before normalization.
    pub face_scale: f32,
}

// ---------------------------------------------------------------------------
// Public utility functions
// ---------------------------------------------------------------------------

/// Compute the centroid of a vertex cloud.
///
/// # Errors
///
/// Returns [`FaceNormError::EmptyMesh`] if `vertices` is empty.
pub fn vertex_centroid(vertices: &[[f32; 3]]) -> Result<[f32; 3], FaceNormError> {
    if vertices.is_empty() {
        return Err(FaceNormError::EmptyMesh("vertex slice is empty".into()));
    }
    let n = vertices.len() as f32;
    let mut sum = [0.0_f32; 3];
    for v in vertices {
        sum[0] += v[0];
        sum[1] += v[1];
        sum[2] += v[2];
    }
    Ok([sum[0] / n, sum[1] / n, sum[2] / n])
}

/// Compute the bounding-box diagonal of a vertex cloud (as a face-scale estimate).
///
/// # Errors
///
/// Returns [`FaceNormError::EmptyMesh`] if `vertices` is empty.
pub fn face_diagonal(vertices: &[[f32; 3]]) -> Result<f32, FaceNormError> {
    if vertices.is_empty() {
        return Err(FaceNormError::EmptyMesh("vertex slice is empty".into()));
    }
    let first = vertices[0];
    let mut mn = first;
    let mut mx = first;
    for v in vertices.iter().skip(1) {
        mn[0] = mn[0].min(v[0]);
        mn[1] = mn[1].min(v[1]);
        mn[2] = mn[2].min(v[2]);
        mx[0] = mx[0].max(v[0]);
        mx[1] = mx[1].max(v[1]);
        mx[2] = mx[2].max(v[2]);
    }
    let d = vec3_sub(mx, mn);
    Ok(vec3_len(d))
}

/// Estimate inter-pupil distance from two eye landmark vertices.
///
/// # Errors
///
/// Returns [`FaceNormError::InvalidLandmark`] if either index is out of range,
/// or [`FaceNormError::EmptyMesh`] if `vertices` is empty.
pub fn inter_pupil_distance(
    vertices: &[[f32; 3]],
    left_eye_idx: usize,
    right_eye_idx: usize,
) -> Result<f32, FaceNormError> {
    if vertices.is_empty() {
        return Err(FaceNormError::EmptyMesh("vertex slice is empty".into()));
    }
    let n = vertices.len();
    if left_eye_idx >= n {
        return Err(FaceNormError::InvalidLandmark {
            idx: left_eye_idx,
            n_verts: n,
        });
    }
    if right_eye_idx >= n {
        return Err(FaceNormError::InvalidLandmark {
            idx: right_eye_idx,
            n_verts: n,
        });
    }
    let le = vertices[left_eye_idx];
    let re = vertices[right_eye_idx];
    Ok(vec3_len(vec3_sub(re, le)))
}

/// Apply a [`NormTransform`] to every vertex in a slice.
#[must_use]
pub fn apply_norm_transform(vertices: &[[f32; 3]], transform: &NormTransform) -> Vec<[f32; 3]> {
    transform.apply_batch(vertices)
}

/// Build a rotation matrix that rotates by `angle` radians around `axis` (unit vector).
///
/// Uses Rodrigues' formula: `R = I*cos(θ) + (1-cos(θ))*k⊗k + sin(θ)*K`.
#[must_use]
pub fn axis_angle_rotation(axis: [f32; 3], angle: f32) -> [[f32; 3]; 3] {
    let k = vec3_normalize(axis);
    let c = angle.cos();
    let s = angle.sin();
    let t = 1.0 - c;
    let (kx, ky, kz) = (k[0], k[1], k[2]);
    [
        [c + t * kx * kx, t * kx * ky - s * kz, t * kx * kz + s * ky],
        [t * ky * kx + s * kz, c + t * ky * ky, t * ky * kz - s * kx],
        [t * kz * kx - s * ky, t * kz * ky + s * kx, c + t * kz * kz],
    ]
}

/// Build a rotation matrix that aligns unit vector `from` to unit vector `to`.
///
/// - If `from ≈ to` (dot > 0.9999): returns identity.
/// - If `from ≈ -to` (dot < -0.9999): returns 180° rotation around a perpendicular axis.
/// - Otherwise: Rodrigues formula with `axis = cross(from, to)`.
#[must_use]
pub fn rotation_align(from: [f32; 3], to: [f32; 3]) -> [[f32; 3]; 3] {
    let f = vec3_normalize(from);
    let t = vec3_normalize(to);
    let d = vec3_dot(f, t);
    if d > 0.9999 {
        return mat3_identity();
    }
    if d < -0.9999 {
        // 180° rotation around a perpendicular axis
        // Find an axis perpendicular to `f`
        let perp = if f[0].abs() < 0.9 {
            vec3_normalize(vec3_cross(f, [1.0, 0.0, 0.0]))
        } else {
            vec3_normalize(vec3_cross(f, [0.0, 1.0, 0.0]))
        };
        return axis_angle_rotation(perp, std::f32::consts::PI);
    }
    let axis = vec3_normalize(vec3_cross(f, t));
    let angle = d.clamp(-1.0, 1.0).acos();
    axis_angle_rotation(axis, angle)
}

/// Compute 3 principal axes (eigenvectors) of a vertex cloud via power-iteration deflation.
///
/// The first axis is seeded deterministically at `[1,0,0]`. The second is
/// seeded with whichever world axis is *least* aligned with the first,
/// Gram-Schmidt-orthogonalised against it — never with a fixed `[0,1,0]`
/// regardless of where the first axis converged. That fixed-seed approach
/// is what let the second axis collapse onto the first: if axis 0
/// converges to (say) `[0,1,0]` — the common case for a head-shaped cloud
/// whose dominant variance is vertical — the once-deflated data has no
/// remaining energy along `[0,1,0]`, so seeding axis 1 there made power
/// iteration break on its very first step and return the untouched,
/// non-orthogonal seed as if it were a converged axis. Seeding orthogonally
/// to axis 0 up front makes that failure mode structurally impossible. The
/// third axis is never estimated by power iteration at all: it is always
/// the exact `cross(axis0, axis1)`, which is both cheaper and guarantees an
/// exactly orthonormal, right-handed triple.
///
/// Returns `[axis0, axis1, axis2]`. When the (deflated) data has genuine
/// variance along a given axis, that axis is a real eigenvector and the
/// triple is in descending order of explained variance; when the deflated
/// data is degenerate in some direction (e.g. a perfectly planar or linear
/// point cloud), the corresponding axis instead falls back to an
/// orthogonal completion of the frame with no variance information. Either
/// way the three returned vectors are always mutually orthonormal, which is
/// what callers like [`align_frontal`] rely on to build a valid rotation
/// matrix.
///
/// # Errors
///
/// Returns [`FaceNormError::EmptyMesh`] if `vertices` is empty.
pub fn pca_axes(vertices: &[[f32; 3]]) -> Result<[[f32; 3]; 3], FaceNormError> {
    if vertices.is_empty() {
        return Err(FaceNormError::EmptyMesh(
            "cannot compute PCA of empty mesh".into(),
        ));
    }
    // Center the vertices
    let centroid = vertex_centroid(vertices)?;
    let mut centered: Vec<[f32; 3]> = vertices.iter().map(|v| vec3_sub(*v, centroid)).collect();

    let axis0 = power_iterate_axis(&centered, [1.0, 0.0, 0.0]);
    deflate_along(&mut centered, axis0);

    let seed1 = orthogonal_seed(axis0);
    let axis1_raw = power_iterate_axis(&centered, seed1);
    // Re-orthogonalise defensively: every point used to derive `axis1_raw`
    // already had its axis0 component removed by `deflate_along`, so it
    // should already be orthogonal to axis0 up to floating-point error —
    // this just cleans up drift accumulated over up to 100 iterations.
    let axis1 = gram_schmidt_unit(axis1_raw, axis0).unwrap_or(seed1);
    deflate_along(&mut centered, axis1);

    // Axis 2 is derived exactly, never estimated: the one direction power
    // iteration would otherwise have to recover from data that (after two
    // deflations) may carry almost no signal at all.
    let axis2_raw = vec3_cross(axis0, axis1);
    let axis2 = if vec3_len(axis2_raw) < 1e-6 {
        // Should not happen given the orthogonal seeding above, but fall
        // back to *some* vector orthogonal to axis0 rather than returning
        // a non-unit (or zero) vector.
        orthogonal_seed(axis0)
    } else {
        vec3_normalize(axis2_raw)
    };

    Ok([axis0, axis1, axis2])
}

/// Scatter-form power iteration for the dominant eigenvector of the
/// covariance of `centered` (already-centered points), starting from
/// `seed` (assumed unit length).
///
/// `Cov * v = Σ_i x_i * dot(x_i, v)` avoids forming an explicit 3×3 matrix.
/// If the data carries no energy reachable from `seed` — including on the
/// very first step, e.g. because deflation already removed all variance in
/// every direction `seed` can reach — `seed` itself is returned unchanged,
/// which is why callers must pass a `seed` that is already a reasonable
/// axis candidate (orthogonal to any previously-found axes) rather than an
/// arbitrary vector.
fn power_iterate_axis(centered: &[[f32; 3]], seed: [f32; 3]) -> [f32; 3] {
    let mut v = seed;
    for _ in 0..100 {
        let mut cv = [0.0_f32; 3];
        for xi in centered {
            let d = vec3_dot(*xi, v);
            cv[0] += xi[0] * d;
            cv[1] += xi[1] * d;
            cv[2] += xi[2] * d;
        }
        if vec3_len(cv) < 1e-30 {
            break;
        }
        v = vec3_normalize(cv);
    }
    v
}

/// Remove the component along unit vector `axis` from every point in
/// `centered`, in place (one step of PCA deflation).
fn deflate_along(centered: &mut [[f32; 3]], axis: [f32; 3]) {
    for xi in centered.iter_mut() {
        let proj = vec3_dot(*xi, axis);
        xi[0] -= proj * axis[0];
        xi[1] -= proj * axis[1];
        xi[2] -= proj * axis[2];
    }
}

/// Pick the world coordinate axis (`[1,0,0]`, `[0,1,0]`, or `[0,0,1]`)
/// least aligned with unit vector `axis`, then Gram-Schmidt it against
/// `axis` so the result is exactly orthogonal to `axis`.
///
/// Because `axis` is a unit vector, its 3 squared components sum to 1, so
/// the smallest is at most `1/3` — the chosen coordinate axis's component
/// along `axis` therefore has magnitude at most `sqrt(1/3) ≈ 0.577`, and
/// the orthogonalised result always has length at least
/// `sqrt(1 - 1/3) ≈ 0.816`, safely clear of the degenerate zero-length case
/// regardless of what `axis` turns out to be.
fn orthogonal_seed(axis: [f32; 3]) -> [f32; 3] {
    let candidates = [[1.0_f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let best = candidates
        .into_iter()
        .min_by(|a, b| {
            let da = vec3_dot(*a, axis).abs();
            let db = vec3_dot(*b, axis).abs();
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or([1.0, 0.0, 0.0]);
    gram_schmidt_unit(best, axis).unwrap_or(best)
}

/// Remove the component of `v` along unit vector `axis` and re-normalise.
///
/// Returns `None` if the residual is too small to normalise reliably,
/// leaving the caller to pick a fallback.
fn gram_schmidt_unit(v: [f32; 3], axis: [f32; 3]) -> Option<[f32; 3]> {
    let proj = vec3_dot(v, axis);
    let residual = vec3_sub(v, vec3_scale(axis, proj));
    if vec3_len(residual) < 1e-6 {
        None
    } else {
        Some(vec3_normalize(residual))
    }
}

/// Align a mesh so the eye line (vector from left to right eye) is horizontal (no Z component).
///
/// This corrects roll only (rotation around the face-forward axis).
///
/// Returns `(aligned_vertices, transform)`.
///
/// # Errors
///
/// Returns [`FaceNormError::InvalidLandmark`] if either eye index is out of range,
/// or [`FaceNormError::EmptyMesh`] if `vertices` is empty.
pub fn align_eye_line(
    vertices: &[[f32; 3]],
    left_eye_idx: usize,
    right_eye_idx: usize,
) -> Result<(Vec<[f32; 3]>, NormTransform), FaceNormError> {
    if vertices.is_empty() {
        return Err(FaceNormError::EmptyMesh("vertex slice is empty".into()));
    }
    let n = vertices.len();
    if left_eye_idx >= n {
        return Err(FaceNormError::InvalidLandmark {
            idx: left_eye_idx,
            n_verts: n,
        });
    }
    if right_eye_idx >= n {
        return Err(FaceNormError::InvalidLandmark {
            idx: right_eye_idx,
            n_verts: n,
        });
    }

    // Eye-line vector (left → right)
    let le = vertices[left_eye_idx];
    let re = vertices[right_eye_idx];
    let eye_vec = vec3_sub(re, le);

    // Roll correction only: rotate about the face-forward (Z) axis by the
    // angle needed to zero the eye line's Y component. `rotation_align`
    // (the minimal-arc rotation between `current_eye` and its XZ-plane
    // projection) is NOT equivalent to this: its rotation axis is
    // `cross(current_eye, target_eye)`, which — since the two vectors
    // differ only in their Y component — lies in the XZ plane, i.e. a
    // HORIZONTAL axis, not the Z (face-forward) axis. Rotating about a
    // horizontal axis zeroes the eye line's Y component too, but it also
    // injects unwanted pitch/yaw into the rest of the head. An explicit Z
    // roll is the only rotation that corrects roll alone, matching this
    // function's documented contract.
    let roll_angle = -eye_vec[1].atan2(eye_vec[0]);
    let rot = axis_angle_rotation([0.0, 0.0, 1.0], roll_angle);
    let transform = NormTransform {
        rotation: rot,
        translation: [0.0, 0.0, 0.0],
        scale: 1.0,
    };
    let aligned = apply_norm_transform(vertices, &transform);
    Ok((aligned, transform))
}

/// Assign each PCA axis a semantic role (left-right / up / front-back) by
/// which *original* coordinate axis it is most aligned with, rather than by
/// variance rank.
///
/// For a real FLAME head the largest-variance axis is vertical (head height
/// including the neck: extents are roughly X≈0.15m width, Y≈0.25m height,
/// Z≈0.20m depth), not left-right — a rank-based assignment silently swaps
/// roles. `align_frontal` refines a mesh that is already roughly frontal
/// (that is what "frontal alignment" means for a PCA-based refinement
/// rather than a from-scratch frame construction), so each PCA axis should
/// already be close to one of the original X/Y/Z directions; whichever
/// original axis a PCA axis best matches (by absolute dot product)
/// determines its role. Conflicts are resolved by greedily assigning the
/// single best-matching (axis, role) pair first, then the next-best among
/// what remains, since the 3 PCA axes returned by [`pca_axes`] are always
/// mutually orthonormal.
///
/// Returns `(right, forward)` — the PCA axes assigned to the left-right and
/// front-back roles (signs not yet fixed; `up` is not returned because
/// callers re-derive it from `cross(right, forward)` for orthogonality).
fn pick_right_and_forward(axes: [[f32; 3]; 3]) -> ([f32; 3], [f32; 3]) {
    // Role indices: 0 = right (world +X), 1 = up (world +Y), 2 = forward
    // (world +Z).
    let world = [[1.0_f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    let mut axis_for_role: [usize; 3] = [0, 1, 2];
    let mut role_taken = [false; 3];
    let mut axis_taken = [false; 3];

    for _ in 0..3 {
        let mut best = (0usize, 0usize, -1.0_f32);
        for (a, axis) in axes.iter().enumerate() {
            if axis_taken[a] {
                continue;
            }
            for (r, role_axis) in world.iter().enumerate() {
                if role_taken[r] {
                    continue;
                }
                let score = vec3_dot(*axis, *role_axis).abs();
                if score > best.2 {
                    best = (a, r, score);
                }
            }
        }
        axis_for_role[best.1] = best.0;
        role_taken[best.1] = true;
        axis_taken[best.0] = true;
    }

    (axes[axis_for_role[0]], axes[axis_for_role[2]])
}

/// Align a mesh to a frontal face pose using PCA.
///
/// Each PCA axis is assigned to the left-right or front-back role by which
/// original coordinate axis it is closest to (see
/// [`pick_right_and_forward`]) — NOT by variance rank, which is wrong for
/// head-shaped meshes (their largest-variance axis is vertical).
/// Result: front-back → +Z, left-right → +X, up = cross(right, forward).
///
/// `nose_tip_idx`, if given, pins the forward direction's sign using actual
/// geometry (the nose tip must end up on the `+forward` side of the
/// centroid) instead of the coarser `forward.z >= 0` convention used when
/// it is `None`.
///
/// Returns `(aligned_vertices, transform)`.
///
/// # Errors
///
/// Returns [`FaceNormError::EmptyMesh`] if `vertices` is empty, or
/// [`FaceNormError::InvalidLandmark`] if `nose_tip_idx` is out of range.
pub fn align_frontal(
    vertices: &[[f32; 3]],
    nose_tip_idx: Option<usize>,
) -> Result<(Vec<[f32; 3]>, NormTransform), FaceNormError> {
    if vertices.is_empty() {
        return Err(FaceNormError::EmptyMesh("vertex slice is empty".into()));
    }
    if let Some(idx) = nose_tip_idx {
        if idx >= vertices.len() {
            return Err(FaceNormError::InvalidLandmark {
                idx,
                n_verts: vertices.len(),
            });
        }
    }

    let axes = pca_axes(vertices)?;
    let (right_raw, forward_raw) = pick_right_and_forward(axes);

    // Fix signs by convention:
    // - forward should have positive Z (face looks toward +Z)
    // - right should have positive X
    let forward = if forward_raw[2] < 0.0 {
        vec3_scale(forward_raw, -1.0)
    } else {
        forward_raw
    };
    let right = if right_raw[0] < 0.0 {
        vec3_scale(right_raw, -1.0)
    } else {
        right_raw
    };

    // A nose-tip landmark, when available, is a far more reliable way to
    // fix which way is "forward" than the coarse `forward.z >= 0`
    // convention above (which only canonicalises an otherwise-arbitrary
    // PCA sign, and has no notion of which side the face actually looks
    // toward): the nose tip should end up on the `+forward` side of the
    // centroid.
    let forward = if let Some(idx) = nose_tip_idx {
        let centroid = vertex_centroid(vertices)?;
        let nose_dir = vec3_sub(vertices[idx], centroid);
        if vec3_dot(nose_dir, forward) < 0.0 {
            vec3_scale(forward, -1.0)
        } else {
            forward
        }
    } else {
        forward
    };

    // up = cross(right, forward), normalized
    let up_raw = vec3_cross(right, forward);
    let up = vec3_normalize(up_raw);

    // Recompute right to ensure orthogonality
    let right_orth = vec3_normalize(vec3_cross(forward, up));

    // Build the rotation matrix: rows = [right, up, forward] (standard camera convention)
    // R maps world coords to the aligned frame
    // The rotation that takes [1,0,0]→right, [0,1,0]→up, [0,0,1]→forward
    // is: columns are right, up, forward.  We store row-major.
    // col-major: R = [right | up | forward]
    // row-major: R[i][j] = component j of basis vector i
    let rot_raw = [
        [right_orth[0], right_orth[1], right_orth[2]],
        [up[0], up[1], up[2]],
        [forward[0], forward[1], forward[2]],
    ];

    // Ensure proper rotation (det = +1)
    let rot = if mat3_det(rot_raw) < 0.0 {
        [
            [right_orth[0], right_orth[1], right_orth[2]],
            [
                vec3_scale(up, -1.0)[0],
                vec3_scale(up, -1.0)[1],
                vec3_scale(up, -1.0)[2],
            ],
            [forward[0], forward[1], forward[2]],
        ]
    } else {
        rot_raw
    };

    let transform = NormTransform {
        rotation: rot,
        translation: [0.0, 0.0, 0.0],
        scale: 1.0,
    };
    let aligned = apply_norm_transform(vertices, &transform);
    Ok((aligned, transform))
}

/// Normalize a mesh to canonical pose and scale.
///
/// Pipeline (in order):
/// 1. Compute centering point per `config.center_on`.
/// 2. Apply rotational alignment per `config.align_to`
///    (for `EyeLine`, both eye indices must be `Some`; for `FrontalFace`,
///    `nose_tip_idx` is used to pin the forward sign when `Some`).
/// 3. Scale so inter-pupil distance = `config.target_scale`
///    (if eye indices are `None`, skip scaling by IPD — use bbox diagonal instead).
///
/// If `left_eye_idx`/`right_eye_idx` are both given (regardless of
/// `config.align_to`), they must be valid indices into `vertices` — an
/// invalid index is always reported as [`FaceNormError::InvalidLandmark`],
/// never silently downgraded to bounding-box scaling.
///
/// # Errors
///
/// Returns [`FaceNormError`] on empty mesh, missing eye indices for `EyeLine`, invalid
/// landmark indices, or other geometric failures.
pub fn normalize_mesh(
    vertices: &[[f32; 3]],
    config: &NormConfig,
    left_eye_idx: Option<usize>,
    right_eye_idx: Option<usize>,
    nose_tip_idx: Option<usize>,
) -> Result<NormResult, FaceNormError> {
    if vertices.is_empty() {
        return Err(FaceNormError::EmptyMesh(
            "cannot normalize empty mesh".into(),
        ));
    }
    if config.target_scale <= 0.0 {
        return Err(FaceNormError::InvalidParam(format!(
            "target_scale must be positive, got {}",
            config.target_scale
        )));
    }

    // Step 1: compute centering translation
    let center = match &config.center_on {
        CenterMode::Centroid => vertex_centroid(vertices)?,
        CenterMode::LandmarkMean(indices) => {
            if indices.is_empty() {
                return Err(FaceNormError::InvalidParam(
                    "LandmarkMean with empty index list".into(),
                ));
            }
            let n = vertices.len();
            let mut sum = [0.0_f32; 3];
            for &idx in indices {
                if idx >= n {
                    return Err(FaceNormError::InvalidLandmark { idx, n_verts: n });
                }
                sum = vec3_add(sum, vertices[idx]);
            }
            let k = indices.len() as f32;
            [sum[0] / k, sum[1] / k, sum[2] / k]
        }
        CenterMode::Origin => [0.0, 0.0, 0.0],
    };

    // Center: translate by -center
    let centered_t = NormTransform {
        rotation: mat3_identity(),
        translation: vec3_scale(center, -1.0),
        scale: 1.0,
    };
    let centered_verts: Vec<[f32; 3]> = centered_t.apply_batch(vertices);

    // Step 2: rotational alignment
    let align_t = match &config.align_to {
        AlignMode::None => NormTransform::identity(),
        AlignMode::FrontalFace => {
            let (_, t) = align_frontal(&centered_verts, nose_tip_idx)?;
            t
        }
        AlignMode::EyeLine => {
            let li = left_eye_idx.ok_or_else(|| {
                FaceNormError::InvalidParam("EyeLine alignment requires left_eye_idx".into())
            })?;
            let ri = right_eye_idx.ok_or_else(|| {
                FaceNormError::InvalidParam("EyeLine alignment requires right_eye_idx".into())
            })?;
            let (_, t) = align_eye_line(&centered_verts, li, ri)?;
            t
        }
    };

    // Step 3: compute scale
    let face_scale_val = face_diagonal(vertices).unwrap_or(1.0);

    // Propagate `InvalidLandmark` rather than swallowing it: a bad index
    // used to silently zero the IPD, which made `scale_factor` fall back to
    // bounding-box-diagonal scaling — a completely different normalisation
    // — with no error, even though `AlignMode::EyeLine` validates these
    // same indices and does error on a bad one. Falling back to bbox
    // scaling is still supported, but only when eye indices are genuinely
    // absent (`None`), never when they were supplied but wrong.
    let ipd = if let (Some(li), Some(ri)) = (left_eye_idx, right_eye_idx) {
        inter_pupil_distance(vertices, li, ri)?
    } else {
        0.0
    };

    let scale_factor = if ipd > 1e-10 {
        config.target_scale / ipd
    } else if face_scale_val > 1e-10 {
        config.target_scale / face_scale_val
    } else {
        1.0
    };

    let scale_t = NormTransform {
        rotation: mat3_identity(),
        translation: [0.0, 0.0, 0.0],
        scale: scale_factor,
    };

    // Compose: centered_t → align_t → scale_t
    let total_t = centered_t.compose(&align_t).compose(&scale_t);
    let normalized_vertices = total_t.apply_batch(vertices);

    Ok(NormResult {
        transform: total_t,
        normalized_vertices,
        inter_pupil_distance: ipd,
        face_scale: face_scale_val,
    })
}

/// Format a [`NormResult`] as a human-readable summary string.
#[must_use]
pub fn format_norm_result(result: &NormResult) -> String {
    let t = &result.transform;
    format!(
        "NormResult {{ scale={:.4}, translation=[{:.4},{:.4},{:.4}], ipd={:.4}, face_scale={:.4}, n_verts={} }}",
        t.scale,
        t.translation[0], t.translation[1], t.translation[2],
        result.inter_pupil_distance,
        result.face_scale,
        result.normalized_vertices.len(),
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::{FRAC_PI_2, PI};

    // ─── Helpers ────────────────────────────────────────────────────────────

    fn approx(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol
    }

    fn approx3(a: [f32; 3], b: [f32; 3], tol: f32) -> bool {
        approx(a[0], b[0], tol) && approx(a[1], b[1], tol) && approx(a[2], b[2], tol)
    }

    fn approx_mat(a: [[f32; 3]; 3], b: [[f32; 3]; 3], tol: f32) -> bool {
        (0..3).all(|r| approx3(a[r], b[r], tol))
    }

    fn unit_cube_vertices() -> Vec<[f32; 3]> {
        let s = 0.5_f32;
        vec![
            [-s, -s, -s],
            [s, -s, -s],
            [-s, s, -s],
            [s, s, -s],
            [-s, -s, s],
            [s, -s, s],
            [-s, s, s],
            [s, s, s],
        ]
    }

    fn elongated_mesh(n: usize) -> Vec<[f32; 3]> {
        // Elongated along X by factor 10
        (0..n)
            .map(|i| {
                let t = i as f32 / (n as f32 - 1.0);
                [t * 10.0 - 5.0, t * 0.1, t * 0.1]
            })
            .collect()
    }

    /// A dense point cloud on the surface of an axis-aligned ellipsoid with
    /// semi-axes matching a real FLAME head's rough proportions: narrowest
    /// in X (left-right), tallest in Y (up, including the neck), and
    /// mid-sized in Z (front-back depth). Variance order is therefore
    /// Y > Z > X — the LARGEST-variance axis is vertical, not left-right,
    /// which is exactly the case that trips up a variance-rank-based axis
    /// assignment.
    fn head_like_mesh(n_theta: usize, n_phi: usize) -> Vec<[f32; 3]> {
        let (a, b, c) = (0.15_f32, 0.25_f32, 0.20_f32); // X, Y, Z semi-axes
        let mut pts = Vec::with_capacity(n_theta * n_phi);
        for i in 0..n_theta {
            let theta = std::f32::consts::PI * (i as f32 + 0.5) / n_theta as f32; // (0, PI)
            for j in 0..n_phi {
                let phi = 2.0 * std::f32::consts::PI * j as f32 / n_phi as f32; // [0, 2*PI)
                let x = a * theta.sin() * phi.cos();
                let y = b * theta.cos();
                let z = c * theta.sin() * phi.sin();
                pts.push([x, y, z]);
            }
        }
        pts
    }

    // ─── vec3_dot ───────────────────────────────────────────────────────────

    #[test]
    fn test_vec3_dot_orthogonal_is_zero() {
        assert!(approx(
            vec3_dot([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
            0.0,
            1e-7
        ));
    }

    #[test]
    fn test_vec3_dot_parallel_is_magnitude_squared() {
        let v = [3.0_f32, 4.0, 0.0];
        assert!(approx(vec3_dot(v, v), 25.0, 1e-5));
    }

    #[test]
    fn test_vec3_dot_known_value() {
        assert!(approx(
            vec3_dot([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]),
            32.0,
            1e-6
        ));
    }

    // ─── vec3_cross ─────────────────────────────────────────────────────────

    #[test]
    fn test_vec3_cross_x_cross_y_eq_z() {
        let r = vec3_cross([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        assert!(approx3(r, [0.0, 0.0, 1.0], 1e-6));
    }

    #[test]
    fn test_vec3_cross_y_cross_z_eq_x() {
        let r = vec3_cross([0.0, 1.0, 0.0], [0.0, 0.0, 1.0]);
        assert!(approx3(r, [1.0, 0.0, 0.0], 1e-6));
    }

    #[test]
    fn test_vec3_cross_z_cross_x_eq_y() {
        let r = vec3_cross([0.0, 0.0, 1.0], [1.0, 0.0, 0.0]);
        assert!(approx3(r, [0.0, 1.0, 0.0], 1e-6));
    }

    #[test]
    fn test_vec3_cross_anticommutative() {
        let a = [1.0_f32, 2.0, 3.0];
        let b = [4.0_f32, 5.0, 6.0];
        let ab = vec3_cross(a, b);
        let ba = vec3_cross(b, a);
        assert!(approx3(ab, vec3_scale(ba, -1.0), 1e-6));
    }

    #[test]
    fn test_vec3_cross_parallel_is_zero() {
        let r = vec3_cross([1.0, 0.0, 0.0], [2.0, 0.0, 0.0]);
        assert!(approx(vec3_len(r), 0.0, 1e-6));
    }

    // ─── vec3_normalize ─────────────────────────────────────────────────────

    #[test]
    fn test_vec3_normalize_unit_length() {
        let n = vec3_normalize([3.0, 4.0, 0.0]);
        assert!(approx(vec3_len(n), 1.0, 1e-6));
    }

    #[test]
    fn test_vec3_normalize_zero_vector_safe() {
        let n = vec3_normalize([0.0, 0.0, 0.0]);
        assert!(approx(vec3_len(n), 0.0, 1e-6));
    }

    #[test]
    fn test_vec3_normalize_direction_preserved() {
        let n = vec3_normalize([0.0, 5.0, 0.0]);
        assert!(approx3(n, [0.0, 1.0, 0.0], 1e-6));
    }

    // ─── vec3_angle ─────────────────────────────────────────────────────────

    #[test]
    fn test_vec3_angle_parallel_is_zero() {
        let a = vec3_angle([1.0, 0.0, 0.0], [2.0, 0.0, 0.0]);
        assert!(approx(a, 0.0, 1e-5));
    }

    #[test]
    fn test_vec3_angle_perpendicular_is_half_pi() {
        let a = vec3_angle([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        assert!(approx(a, FRAC_PI_2, 1e-5));
    }

    #[test]
    fn test_vec3_angle_antiparallel_is_pi() {
        let a = vec3_angle([1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]);
        assert!(approx(a, PI, 1e-5));
    }

    // ─── mat3_mul ───────────────────────────────────────────────────────────

    #[test]
    fn test_mat3_mul_identity_times_identity() {
        let id = mat3_identity();
        assert!(approx_mat(mat3_mul(id, id), id, 1e-7));
    }

    #[test]
    fn test_mat3_mul_known_result() {
        // A = [[1,2,0],[0,1,0],[0,0,1]]  B = [[2,0,0],[1,1,0],[0,0,1]]
        // A*B = [[4,2,0],[1,1,0],[0,0,1]]
        let a = [[1.0, 2.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let b = [[2.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let c = mat3_mul(a, b);
        let expected = [[4.0, 2.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        assert!(approx_mat(c, expected, 1e-6));
    }

    #[test]
    fn test_mat3_mul_rotation_orthogonality() {
        let r = axis_angle_rotation([0.0, 0.0, 1.0], FRAC_PI_2);
        let rt = mat3_transpose(r);
        let rrt = mat3_mul(r, rt);
        assert!(approx_mat(rrt, mat3_identity(), 1e-5));
    }

    // ─── mat3_transpose ─────────────────────────────────────────────────────

    #[test]
    fn test_mat3_transpose_known() {
        let m = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let mt = mat3_transpose(m);
        let expected = [[1.0, 4.0, 7.0], [2.0, 5.0, 8.0], [3.0, 6.0, 9.0]];
        assert!(approx_mat(mt, expected, 1e-7));
    }

    #[test]
    fn test_mat3_transpose_double_is_identity_op() {
        let m = [[1.0, 2.0, 3.0], [0.0, 4.0, 5.0], [0.0, 0.0, 6.0]];
        assert!(approx_mat(mat3_transpose(mat3_transpose(m)), m, 1e-7));
    }

    // ─── mat3_apply ─────────────────────────────────────────────────────────

    #[test]
    fn test_mat3_apply_identity_unchanged() {
        let v = [1.0_f32, 2.0, 3.0];
        assert!(approx3(mat3_apply(mat3_identity(), v), v, 1e-7));
    }

    #[test]
    fn test_mat3_apply_known_rotation() {
        // 90° around Z: [1,0,0] → [0,1,0]
        let r = axis_angle_rotation([0.0, 0.0, 1.0], FRAC_PI_2);
        let v = mat3_apply(r, [1.0, 0.0, 0.0]);
        assert!(approx3(v, [0.0, 1.0, 0.0], 1e-5));
    }

    #[test]
    fn test_mat3_apply_scaling_matrix() {
        // Diagonal scale matrix
        let m = [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]];
        let v = mat3_apply(m, [1.0, 1.0, 1.0]);
        assert!(approx3(v, [2.0, 3.0, 4.0], 1e-6));
    }

    // ─── axis_angle_rotation ────────────────────────────────────────────────

    #[test]
    fn test_axis_angle_rotation_90_around_z() {
        let r = axis_angle_rotation([0.0, 0.0, 1.0], FRAC_PI_2);
        let v = mat3_apply(r, [1.0, 0.0, 0.0]);
        // 90° around Z: (1,0,0) → (0,1,0)
        assert!(approx3(v, [0.0, 1.0, 0.0], 1e-5));
    }

    #[test]
    fn test_axis_angle_rotation_90_around_x() {
        let r = axis_angle_rotation([1.0, 0.0, 0.0], FRAC_PI_2);
        let v = mat3_apply(r, [0.0, 1.0, 0.0]);
        // 90° around X: (0,1,0) → (0,0,1)
        assert!(approx3(v, [0.0, 0.0, 1.0], 1e-5));
    }

    #[test]
    fn test_axis_angle_rotation_det_is_one() {
        let r = axis_angle_rotation([1.0, 1.0, 0.0], PI / 3.0);
        assert!(approx(mat3_det(r), 1.0, 1e-5));
    }

    #[test]
    fn test_axis_angle_rotation_zero_angle_is_identity() {
        let r = axis_angle_rotation([0.0, 1.0, 0.0], 0.0);
        assert!(approx_mat(r, mat3_identity(), 1e-6));
    }

    // ─── rotation_align ─────────────────────────────────────────────────────

    #[test]
    fn test_rotation_align_from_eq_to_is_identity() {
        let r = rotation_align([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
        assert!(approx_mat(r, mat3_identity(), 1e-6));
    }

    #[test]
    fn test_rotation_align_antiparallel_gives_180_rotation() {
        let r = rotation_align([1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]);
        let v = mat3_apply(r, [1.0, 0.0, 0.0]);
        // Should map to approximately (-1, 0, 0)
        assert!(approx(v[0], -1.0, 1e-4));
    }

    #[test]
    fn test_rotation_align_x_to_y() {
        let r = rotation_align([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        let v = mat3_apply(r, [1.0, 0.0, 0.0]);
        assert!(approx3(v, [0.0, 1.0, 0.0], 1e-5));
    }

    #[test]
    fn test_rotation_align_det_is_one() {
        let r = rotation_align([0.6, 0.8, 0.0], [0.0, 0.0, 1.0]);
        assert!(approx(mat3_det(r), 1.0, 1e-5));
    }

    // ─── NormTransform::identity ─────────────────────────────────────────────

    #[test]
    fn test_norm_transform_identity_apply_unchanged() {
        let t = NormTransform::identity();
        let p = [1.0_f32, 2.0, 3.0];
        assert!(approx3(t.apply(&p), p, 1e-7));
    }

    #[test]
    fn test_norm_transform_identity_fields() {
        let t = NormTransform::identity();
        assert!(approx(t.scale, 1.0, 1e-7));
        assert!(approx3(t.translation, [0.0, 0.0, 0.0], 1e-7));
        assert!(approx_mat(t.rotation, mat3_identity(), 1e-7));
    }

    // ─── NormTransform::apply / apply_batch ──────────────────────────────────

    #[test]
    fn test_norm_transform_apply_translation() {
        let t = NormTransform {
            rotation: mat3_identity(),
            translation: [1.0, 2.0, 3.0],
            scale: 1.0,
        };
        let v = t.apply(&[0.0, 0.0, 0.0]);
        assert!(approx3(v, [1.0, 2.0, 3.0], 1e-6));
    }

    #[test]
    fn test_norm_transform_apply_scale() {
        let t = NormTransform {
            rotation: mat3_identity(),
            translation: [0.0, 0.0, 0.0],
            scale: 2.0,
        };
        let v = t.apply(&[1.0, 1.0, 1.0]);
        assert!(approx3(v, [2.0, 2.0, 2.0], 1e-6));
    }

    #[test]
    fn test_norm_transform_apply_batch_length() {
        let t = NormTransform::identity();
        let pts = vec![[0.0_f32; 3]; 5];
        assert_eq!(t.apply_batch(&pts).len(), 5);
    }

    // ─── NormTransform::inverse ──────────────────────────────────────────────

    #[test]
    fn test_norm_transform_inverse_roundtrip() {
        let t = NormTransform {
            rotation: axis_angle_rotation([0.0, 1.0, 0.0], PI / 4.0),
            translation: [1.0, 2.0, 3.0],
            scale: 2.0,
        };
        let inv = t.inverse().expect("inverse should succeed");
        let p = [5.0_f32, -3.0, 2.0];
        let forward = t.apply(&p);
        let back = inv.apply(&forward);
        assert!(approx3(back, p, 1e-4));
    }

    #[test]
    fn test_norm_transform_inverse_singular_error() {
        let t = NormTransform {
            rotation: mat3_identity(),
            translation: [0.0; 3],
            scale: 0.0,
        };
        assert!(t.inverse().is_err());
    }

    // ─── NormTransform::compose ──────────────────────────────────────────────

    #[test]
    fn test_norm_transform_compose_identity() {
        let t = NormTransform {
            rotation: axis_angle_rotation([0.0, 0.0, 1.0], FRAC_PI_2),
            translation: [1.0, 0.0, 0.0],
            scale: 3.0,
        };
        let id = NormTransform::identity();
        let composed = t.compose(&id);
        let p = [1.0, 0.0, 0.0];
        assert!(approx3(composed.apply(&p), t.apply(&p), 1e-5));
    }

    #[test]
    fn test_norm_transform_compose_scale_multiplied() {
        let t1 = NormTransform {
            rotation: mat3_identity(),
            translation: [0.0; 3],
            scale: 2.0,
        };
        let t2 = NormTransform {
            rotation: mat3_identity(),
            translation: [0.0; 3],
            scale: 3.0,
        };
        let c = t1.compose(&t2);
        assert!(approx(c.scale, 6.0, 1e-6));
    }

    #[test]
    fn test_norm_transform_compose_associativity() {
        let mk = |angle: f32, s: f32, tx: f32| NormTransform {
            rotation: axis_angle_rotation([0.0, 1.0, 0.0], angle),
            translation: [tx, 0.0, 0.0],
            scale: s,
        };
        let a = mk(0.3, 2.0, 1.0);
        let b = mk(0.5, 1.5, -0.5);
        let c = mk(0.1, 0.8, 2.0);
        let abc = a.compose(&b).compose(&c);
        let abc2 = a.compose(&b.compose(&c));
        let p = [1.0, 0.5, -0.3];
        assert!(approx3(abc.apply(&p), abc2.apply(&p), 1e-4));
    }

    // ─── vertex_centroid ────────────────────────────────────────────────────

    #[test]
    fn test_vertex_centroid_cube_at_origin() {
        let verts = unit_cube_vertices();
        let c = vertex_centroid(&verts).expect("centroid failed");
        assert!(approx3(c, [0.0, 0.0, 0.0], 1e-6));
    }

    #[test]
    fn test_vertex_centroid_known_triangle() {
        let verts = vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]];
        let c = vertex_centroid(&verts).expect("centroid failed");
        let expected = [2.0 / 3.0, 2.0 / 3.0, 0.0];
        assert!(approx3(c, expected, 1e-5));
    }

    #[test]
    fn test_vertex_centroid_empty_error() {
        assert!(vertex_centroid(&[]).is_err());
    }

    // ─── face_diagonal ──────────────────────────────────────────────────────

    #[test]
    fn test_face_diagonal_unit_cube() {
        let verts = unit_cube_vertices();
        let d = face_diagonal(&verts).expect("diagonal failed");
        assert!(approx(d, 3.0_f32.sqrt(), 1e-5));
    }

    #[test]
    fn test_face_diagonal_two_points() {
        let verts = vec![[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]];
        let d = face_diagonal(&verts).expect("diagonal failed");
        assert!(approx(d, 5.0, 1e-5));
    }

    #[test]
    fn test_face_diagonal_empty_error() {
        assert!(face_diagonal(&[]).is_err());
    }

    // ─── inter_pupil_distance ───────────────────────────────────────────────

    #[test]
    fn test_inter_pupil_distance_known() {
        let verts = vec![[0.0, 0.0, 0.0], [6.0, 0.0, 0.0], [3.0, 3.0, 0.0]];
        let d = inter_pupil_distance(&verts, 0, 1).expect("ipd failed");
        assert!(approx(d, 6.0, 1e-5));
    }

    #[test]
    fn test_inter_pupil_distance_3d() {
        let verts = vec![[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]];
        let d = inter_pupil_distance(&verts, 0, 1).expect("ipd failed");
        assert!(approx(d, 5.0, 1e-5));
    }

    #[test]
    fn test_inter_pupil_distance_invalid_idx_error() {
        let verts = vec![[0.0; 3], [1.0, 0.0, 0.0]];
        assert!(inter_pupil_distance(&verts, 0, 5).is_err());
    }

    #[test]
    fn test_inter_pupil_distance_empty_error() {
        assert!(inter_pupil_distance(&[], 0, 1).is_err());
    }

    // ─── pca_axes ───────────────────────────────────────────────────────────

    #[test]
    fn test_pca_axes_elongated_mesh_first_axis_aligns_x() {
        let verts = elongated_mesh(20);
        let axes = pca_axes(&verts).expect("pca failed");
        // First axis should align with X (the long direction)
        let alignment = vec3_dot(vec3_normalize(axes[0]), [1.0, 0.0, 0.0]).abs();
        assert!(
            alignment > 0.98,
            "first axis should align with X, got dot={alignment}"
        );
    }

    #[test]
    fn test_pca_axes_orthogonal() {
        let verts = unit_cube_vertices();
        let axes = pca_axes(&verts).expect("pca failed");
        // All axis pairs should be approximately orthogonal
        let d01 = vec3_dot(axes[0], axes[1]).abs();
        let d02 = vec3_dot(axes[0], axes[2]).abs();
        let d12 = vec3_dot(axes[1], axes[2]).abs();
        assert!(d01 < 0.1, "axes 0,1 should be orthogonal, dot={d01}");
        assert!(d02 < 0.1, "axes 0,2 should be orthogonal, dot={d02}");
        assert!(d12 < 0.1, "axes 1,2 should be orthogonal, dot={d12}");
    }

    #[test]
    fn test_pca_axes_empty_error() {
        assert!(pca_axes(&[]).is_err());
    }

    #[test]
    fn test_pca_axes_returns_three_axes() {
        let verts = unit_cube_vertices();
        let axes = pca_axes(&verts).expect("pca failed");
        assert_eq!(axes.len(), 3);
    }

    // Regression test for the duplicate-axis bug: seeding axis 1 with a
    // fixed `[0,1,0]` collapsed it onto axis 0 whenever axis 0 itself
    // converged near vertical — the common case for a head-shaped (or, as
    // here, Y-elongated) point cloud.
    #[test]
    fn test_pca_axes_second_axis_not_duplicate_when_dominant_variance_is_vertical() {
        let verts: Vec<[f32; 3]> = (0..20)
            .map(|i| {
                let t = i as f32 / 19.0;
                // Elongated along Y by factor 100 vs. the small X/Z wobble,
                // matching `elongated_mesh`'s shape but rotated so the
                // dominant direction is vertical instead of horizontal.
                [t * 0.1, t * 10.0 - 5.0, t * 0.1]
            })
            .collect();
        let axes = pca_axes(&verts).expect("pca failed");

        let axis0_vertical = vec3_dot(vec3_normalize(axes[0]), [0.0, 1.0, 0.0]).abs();
        assert!(
            axis0_vertical > 0.98,
            "axis0 should align with Y (the dominant direction), got dot={axis0_vertical}"
        );

        let d01 = vec3_dot(axes[0], axes[1]).abs();
        assert!(
            d01 < 0.1,
            "axis1 must not collapse onto axis0 (old bug), got |dot|={d01}"
        );
        let d02 = vec3_dot(axes[0], axes[2]).abs();
        let d12 = vec3_dot(axes[1], axes[2]).abs();
        assert!(d02 < 0.1, "axis0,2 should be orthogonal, got |dot|={d02}");
        assert!(d12 < 0.1, "axis1,2 should be orthogonal, got |dot|={d12}");
    }

    #[test]
    fn test_pca_axes_all_unit_length() {
        // Every returned axis must be a unit vector, including in the
        // degenerate (rank-deficient) fallback paths.
        let verts = elongated_mesh(20); // exactly rank-1 data
        let axes = pca_axes(&verts).expect("pca failed");
        for (i, axis) in axes.iter().enumerate() {
            let len = vec3_len(*axis);
            assert!(
                (len - 1.0).abs() < 1e-3,
                "axis {i} should be unit length, got {len}"
            );
        }
    }

    // ─── align_eye_line ─────────────────────────────────────────────────────

    #[test]
    fn test_align_eye_line_tilted_becomes_horizontal() {
        // Tilted eye line: left eye below-left, right eye above-right
        let verts = vec![
            [-1.0_f32, -1.0, 0.0], // left eye (idx 0)
            [1.0_f32, 1.0, 0.0],   // right eye (idx 1)
            [0.0_f32, 0.0, 1.0],   // nose tip
        ];
        let (aligned, _) = align_eye_line(&verts, 0, 1).expect("align failed");
        let le = aligned[0];
        let re = aligned[1];
        let dy = (re[1] - le[1]).abs();
        assert!(
            dy < 1e-4,
            "after align, Y difference should be ~0, got {dy}"
        );
    }

    #[test]
    fn test_align_eye_line_already_horizontal_unchanged() {
        // Already horizontal: no rotation expected
        let verts = vec![[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        let (_, t) = align_eye_line(&verts, 0, 1).expect("align failed");
        // Should be identity (or near-identity) rotation
        let p = [1.0_f32, 2.0, 3.0];
        assert!(approx3(t.apply(&p), p, 1e-4));
    }

    #[test]
    fn test_align_eye_line_invalid_idx_error() {
        let verts = vec![[0.0; 3], [1.0, 0.0, 0.0]];
        assert!(align_eye_line(&verts, 0, 5).is_err());
    }

    #[test]
    fn test_align_eye_line_empty_error() {
        assert!(align_eye_line(&[], 0, 1).is_err());
    }

    // ─── align_frontal ───────────────────────────────────────────────────────

    #[test]
    fn test_align_frontal_rotation_is_orthogonal() {
        let verts = elongated_mesh(30);
        let (_, t) = align_frontal(&verts, None).expect("frontal failed");
        let rrt = mat3_mul(t.rotation, mat3_transpose(t.rotation));
        assert!(approx_mat(rrt, mat3_identity(), 1e-4));
    }

    #[test]
    fn test_align_frontal_det_is_positive() {
        let verts = unit_cube_vertices();
        let (_, t) = align_frontal(&verts, None).expect("frontal failed");
        assert!(
            mat3_det(t.rotation) > 0.0,
            "rotation must have positive det"
        );
    }

    #[test]
    fn test_align_frontal_empty_error() {
        assert!(align_frontal(&[], None).is_err());
    }

    #[test]
    fn test_align_frontal_preserves_vertex_count() {
        let verts = unit_cube_vertices();
        let (aligned, _) = align_frontal(&verts, None).expect("frontal failed");
        assert_eq!(aligned.len(), verts.len());
    }

    // Regression test for the variance-rank bug: for a head-shaped mesh
    // (dominant variance is vertical, least variance is left-right — the
    // OPPOSITE of what the old `axes[0] = right` assumption required),
    // `right` must still end up matching the mesh's true narrow (X) axis.
    #[test]
    fn test_align_frontal_identifies_right_axis_by_alignment_not_variance_rank() {
        let verts = head_like_mesh(24, 48);
        let (aligned, transform) = align_frontal(&verts, None).expect("frontal failed");

        // `transform.rotation[0]` ("right") is the direction whose dot
        // product with an original-frame point gives the point's aligned
        // X-coordinate; it must match the mesh's true narrow direction
        // (original world X), not its tall direction (original world Y).
        let right_row = transform.rotation[0];
        let alignment_with_true_right = vec3_dot(vec3_normalize(right_row), [1.0, 0.0, 0.0]).abs();
        assert!(
            alignment_with_true_right > 0.9,
            "right axis should align with the mesh's true left-right (X) \
             direction, got rotation row0={right_row:?} (|dot| with X = \
             {alignment_with_true_right})"
        );

        // Equivalently: the aligned X extent should match the mesh's true
        // (narrow) width, not its (tall) height.
        let xs: Vec<f32> = aligned.iter().map(|v| v[0]).collect();
        let x_extent = xs.iter().copied().fold(f32::MIN, f32::max)
            - xs.iter().copied().fold(f32::MAX, f32::min);
        assert!(
            x_extent < 0.35,
            "aligned X extent should match the mesh's narrow width (~0.3), got {x_extent}"
        );
    }

    #[test]
    fn test_align_frontal_nose_tip_landmark_fixes_forward_sign() {
        // A head-like mesh whose "nose" landmark protrudes along -Z in the
        // input frame. Regardless of whichever sign the raw PCA/heuristic
        // path would otherwise pick, passing `nose_tip_idx` must place the
        // nose on the +forward side after alignment.
        let mut verts = head_like_mesh(24, 48);
        let nose_idx = verts.len();
        verts.push([0.0, 0.0, -0.35]); // nose tip protrudes along -Z

        let (aligned, _) = align_frontal(&verts, Some(nose_idx)).expect("frontal failed");
        assert!(
            aligned[nose_idx][2] > 0.0,
            "nose landmark should end up on the +forward side after alignment, got z={}",
            aligned[nose_idx][2]
        );
    }

    #[test]
    fn test_align_frontal_invalid_nose_tip_idx_error() {
        let verts = unit_cube_vertices();
        assert!(align_frontal(&verts, Some(9999)).is_err());
    }

    // ─── normalize_mesh ──────────────────────────────────────────────────────

    #[test]
    fn test_normalize_mesh_centroid_none_mode() {
        let verts = unit_cube_vertices();
        let config = NormConfig {
            target_scale: 1.0,
            center_on: CenterMode::Centroid,
            align_to: AlignMode::None,
        };
        let result = normalize_mesh(&verts, &config, None, None, None).expect("normalize failed");
        // Centroid of normalized vertices should be near origin
        let c = vertex_centroid(&result.normalized_vertices).expect("centroid failed");
        assert!(approx3(c, [0.0, 0.0, 0.0], 1e-4));
    }

    #[test]
    fn test_normalize_mesh_landmark_mean_center() {
        let verts = vec![
            [0.0_f32, 0.0, 0.0],
            [2.0_f32, 0.0, 0.0],
            [0.0_f32, 2.0, 0.0],
            [2.0_f32, 2.0, 0.0],
        ];
        let config = NormConfig {
            target_scale: 1.0,
            center_on: CenterMode::LandmarkMean(vec![0, 3]),
            align_to: AlignMode::None,
        };
        let result = normalize_mesh(&verts, &config, None, None, None).expect("normalize failed");
        // Mean of verts[0] and verts[3] should map to origin
        let m = vertex_centroid(&[result.normalized_vertices[0], result.normalized_vertices[3]])
            .expect("centroid");
        assert!(approx3(m, [0.0, 0.0, 0.0], 1e-4));
    }

    #[test]
    fn test_normalize_mesh_eye_line_alignment() {
        let verts = vec![
            [-1.0_f32, -0.5, 0.0],
            [1.0_f32, 0.5, 0.0],
            [0.0_f32, 0.0, 1.0],
        ];
        let config = NormConfig {
            target_scale: 1.0,
            center_on: CenterMode::Origin,
            align_to: AlignMode::EyeLine,
        };
        let result =
            normalize_mesh(&verts, &config, Some(0), Some(1), None).expect("normalize failed");
        let le = result.normalized_vertices[0];
        let re = result.normalized_vertices[1];
        assert!(
            approx((re[1] - le[1]).abs(), 0.0, 1e-3),
            "eye line should be horizontal"
        );
    }

    #[test]
    fn test_normalize_mesh_frontal_alignment() {
        let verts = elongated_mesh(20);
        let config = NormConfig {
            target_scale: 1.0,
            center_on: CenterMode::Centroid,
            align_to: AlignMode::FrontalFace,
        };
        let result = normalize_mesh(&verts, &config, None, None, None).expect("normalize failed");
        assert!(!result.normalized_vertices.is_empty());
    }

    #[test]
    fn test_normalize_mesh_ipd_scaling() {
        let verts = vec![
            [-2.0_f32, 0.0, 0.0],
            [2.0_f32, 0.0, 0.0],
            [0.0_f32, 0.0, 1.0],
        ];
        // IPD is 4.0; target 2.0 ⇒ scale = 0.5
        let config = NormConfig {
            target_scale: 2.0,
            center_on: CenterMode::Centroid,
            align_to: AlignMode::None,
        };
        let result =
            normalize_mesh(&verts, &config, Some(0), Some(1), None).expect("normalize failed");
        let nv = &result.normalized_vertices;
        let d = vec3_len(vec3_sub(nv[1], nv[0]));
        assert!(
            approx(d, 2.0, 1e-4),
            "normalized IPD should be target_scale=2.0, got {d}"
        );
    }

    #[test]
    fn test_normalize_mesh_empty_error() {
        let config = NormConfig::default();
        assert!(normalize_mesh(&[], &config, None, None, None).is_err());
    }

    #[test]
    fn test_normalize_mesh_invalid_target_scale_error() {
        let verts = unit_cube_vertices();
        let config = NormConfig {
            target_scale: -1.0,
            center_on: CenterMode::Centroid,
            align_to: AlignMode::None,
        };
        assert!(normalize_mesh(&verts, &config, None, None, None).is_err());
    }

    #[test]
    fn test_normalize_mesh_eyeline_missing_idx_error() {
        let verts = unit_cube_vertices();
        let config = NormConfig {
            target_scale: 1.0,
            center_on: CenterMode::Centroid,
            align_to: AlignMode::EyeLine,
        };
        // No eye indices provided → error
        assert!(normalize_mesh(&verts, &config, None, None, None).is_err());
    }

    // Regression test: an out-of-range eye index used to be silently
    // swallowed by `.unwrap_or(0.0)`, which zeroed the IPD and made
    // `normalize_mesh` fall back to bounding-box scaling instead of
    // reporting `InvalidLandmark` — even though the same indices DO error
    // under `AlignMode::EyeLine`. The error must now surface consistently
    // regardless of `align_to`.
    #[test]
    fn test_normalize_mesh_invalid_eye_idx_errors_even_without_eyeline_align() {
        let verts = unit_cube_vertices();
        let config = NormConfig {
            target_scale: 1.0,
            center_on: CenterMode::Centroid,
            align_to: AlignMode::None,
        };
        let result = normalize_mesh(&verts, &config, Some(0), Some(9999), None);
        assert!(
            result.is_err(),
            "an out-of-range eye index must error, not silently fall back to bbox scaling"
        );
    }

    #[test]
    fn test_normalize_mesh_origin_center_no_shift() {
        let verts = vec![[5.0_f32, 5.0, 5.0], [6.0, 5.0, 5.0], [5.5, 6.0, 5.0]];
        let config = NormConfig {
            target_scale: 1.0,
            center_on: CenterMode::Origin,
            align_to: AlignMode::None,
        };
        let result = normalize_mesh(&verts, &config, None, None, None).expect("normalize failed");
        // Origin mode: no translation — just scaling
        // All vertices should be scaled but not shifted
        assert!(!result.normalized_vertices.is_empty());
    }

    // ─── format_norm_result ──────────────────────────────────────────────────

    #[test]
    fn test_format_norm_result_nonempty() {
        let verts = unit_cube_vertices();
        let config = NormConfig::default();
        let result = normalize_mesh(&verts, &config, None, None, None).expect("normalize failed");
        let s = format_norm_result(&result);
        assert!(!s.is_empty(), "format result must not be empty");
        assert!(s.contains("NormResult"), "should contain 'NormResult'");
    }

    #[test]
    fn test_format_norm_result_contains_fields() {
        let result = NormResult {
            transform: NormTransform::identity(),
            normalized_vertices: vec![[0.0; 3]],
            inter_pupil_distance: 0.5,
            face_scale: 1.2,
        };
        let s = format_norm_result(&result);
        assert!(s.contains("scale="), "should mention scale");
        assert!(s.contains("ipd="), "should mention ipd");
    }

    // ─── apply_norm_transform ───────────────────────────────────────────────

    #[test]
    fn test_apply_norm_transform_identity_unchanged() {
        let verts = unit_cube_vertices();
        let t = NormTransform::identity();
        let out = apply_norm_transform(&verts, &t);
        for (a, b) in verts.iter().zip(out.iter()) {
            assert!(approx3(*a, *b, 1e-6));
        }
    }

    #[test]
    fn test_apply_norm_transform_count_preserved() {
        let verts = unit_cube_vertices();
        let t = NormTransform {
            rotation: mat3_identity(),
            translation: [1.0; 3],
            scale: 2.0,
        };
        assert_eq!(apply_norm_transform(&verts, &t).len(), verts.len());
    }

    // ─── Error display ───────────────────────────────────────────────────────

    #[test]
    fn test_face_norm_error_empty_mesh_display() {
        let e = FaceNormError::EmptyMesh("test context".into());
        let s = e.to_string();
        assert!(s.contains("test context"));
    }

    #[test]
    fn test_face_norm_error_invalid_landmark_display() {
        let e = FaceNormError::InvalidLandmark {
            idx: 99,
            n_verts: 10,
        };
        let s = e.to_string();
        assert!(s.contains("99") && s.contains("10"));
    }

    #[test]
    fn test_face_norm_error_singular_matrix_display() {
        let e = FaceNormError::SingularMatrix;
        assert!(!e.to_string().is_empty());
    }

    #[test]
    fn test_face_norm_error_invalid_param_display() {
        let e = FaceNormError::InvalidParam("bad value".into());
        assert!(e.to_string().contains("bad value"));
    }
}
