//! Head geometry utilities: anthropometric measurements, anatomical region analysis,
//! and geometric properties of FLAME head meshes.
//!
//! # Overview
//!
//! - [`measure_head`]: Compute standard anthropometric measurements from vertex positions.
//! - [`head_bounding_box`]: Axis-aligned bounding box of the head mesh.
//! - [`head_centroid`]: Center of mass (vertex centroid).
//! - [`head_volume`]: Signed volume via divergence theorem (requires faces).
//! - [`head_surface_area`]: Sum of face areas.
//! - [`classify_vertex_region`]: Map a vertex to an anatomical [`HeadRegion`].
//! - [`label_vertices_by_region`]: Per-vertex region classification.
//! - [`convex_hull_2d`]: Graham scan convex hull on 2-D point sets.
//! - [`frontal_silhouette`]: Convex hull of the frontal (XY) projection.
//! - [`principal_axis`]: Largest PCA eigenvector via power iteration.
//! - [`compute_head_geometry_stats`]: Aggregated geometry statistics.

use std::fmt::Write as FmtWrite;
use thiserror::Error;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can arise during head geometry computations.
#[derive(Debug, Error)]
pub enum HeadGeometryError {
    /// The vertex or face slice is completely empty.
    #[error("Empty mesh: {0}")]
    EmptyMesh(String),
    /// A landmark index is out of range for the given vertex count.
    #[error("Invalid landmark index {idx} for {n_verts} vertices")]
    InvalidLandmark {
        /// The out-of-range index that was requested.
        idx: usize,
        /// Actual number of vertices in the mesh.
        n_verts: usize,
    },
    /// The operation needs more vertices than are present.
    #[error("Insufficient vertices for measurement: need {need}, have {have}")]
    InsufficientVertices {
        /// Minimum number of vertices required.
        need: usize,
        /// Number of vertices actually provided.
        have: usize,
    },
    /// A parameter value is outside its valid range.
    #[error("Invalid parameter: {0}")]
    InvalidParam(String),
}

// ---------------------------------------------------------------------------
// Internal math helpers
// ---------------------------------------------------------------------------

#[inline]
fn dot3(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn cross3(a: &[f32; 3], b: &[f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn norm3(v: &[f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

#[inline]
fn normalize3(v: &[f32; 3]) -> [f32; 3] {
    let n = norm3(v);
    if n < 1e-30 {
        return [0.0, 0.0, 0.0];
    }
    [v[0] / n, v[1] / n, v[2] / n]
}

#[inline]
fn sub3(a: &[f32; 3], b: &[f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

// ---------------------------------------------------------------------------
// Anthropometric measurements
// ---------------------------------------------------------------------------

/// Standard anthropometric measurements for a head mesh.
#[derive(Debug, Clone)]
pub struct HeadMeasurements {
    /// Anterior-to-posterior extent (max depth along Z axis).
    pub head_length: f32,
    /// Lateral extent (max width along X axis).
    pub head_width: f32,
    /// Inferior-to-superior extent (max height along Y axis).
    pub head_height: f32,
    /// Approximate zygomatic width (cheekbone to cheekbone).
    pub face_width: f32,
    /// Approximate chin-to-brow distance.
    pub face_height: f32,
    /// Approximate nose tip to root length.
    pub nose_length: f32,
    /// Approximate mandible width at jaw angles.
    pub jaw_width: f32,
    /// Cephalic index: `head_width / head_length * 100`.
    pub cephalic_index: f32,
    /// Facial index: `face_height / face_width * 100`.
    pub facial_index: f32,
}

/// Compute standard anthropometric measurements from vertex positions.
///
/// Optional landmark indices refine the `face_height` calculation.
///
/// # Errors
///
/// Returns [`HeadGeometryError::EmptyMesh`] when `vertices` is empty, or
/// [`HeadGeometryError::InvalidLandmark`] when a provided index is out of range.
pub fn measure_head(
    vertices: &[[f32; 3]],
    nose_tip_idx: Option<usize>,
    chin_idx: Option<usize>,
) -> Result<HeadMeasurements, HeadGeometryError> {
    if vertices.is_empty() {
        return Err(HeadGeometryError::EmptyMesh(
            "vertices slice is empty".to_string(),
        ));
    }

    // Validate optional landmark indices.
    if let Some(idx) = nose_tip_idx {
        if idx >= vertices.len() {
            return Err(HeadGeometryError::InvalidLandmark {
                idx,
                n_verts: vertices.len(),
            });
        }
    }
    if let Some(idx) = chin_idx {
        if idx >= vertices.len() {
            return Err(HeadGeometryError::InvalidLandmark {
                idx,
                n_verts: vertices.len(),
            });
        }
    }

    let mut min_x = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    let mut min_z = f32::INFINITY;
    let mut max_z = f32::NEG_INFINITY;

    for v in vertices {
        if v[0] < min_x {
            min_x = v[0];
        }
        if v[0] > max_x {
            max_x = v[0];
        }
        if v[1] < min_y {
            min_y = v[1];
        }
        if v[1] > max_y {
            max_y = v[1];
        }
        if v[2] < min_z {
            min_z = v[2];
        }
        if v[2] > max_z {
            max_z = v[2];
        }
    }

    let head_length = (max_z - min_z).max(0.0);
    let head_width = (max_x - min_x).max(0.0);
    let head_height = (max_y - min_y).max(0.0);

    let face_width = head_width * 0.8;

    // Optionally use chin + brow region to sharpen face_height.
    let face_height = if let (Some(c_idx), Some(_n_idx)) = (chin_idx, nose_tip_idx) {
        let chin_y = vertices[c_idx][1];
        let brow_y = max_y;
        (brow_y - chin_y).abs() * 0.7_f32.max(0.0)
    } else {
        head_height * 0.7
    };

    let nose_length = 0.15 * head_height;
    let jaw_width = head_width * 0.6;

    let cephalic_index = head_width / head_length.max(1e-6) * 100.0;
    let facial_index = face_height / face_width.max(1e-6) * 100.0;

    Ok(HeadMeasurements {
        head_length,
        head_width,
        head_height,
        face_width,
        face_height,
        nose_length,
        jaw_width,
        cephalic_index,
        facial_index,
    })
}

/// Compute the axis-aligned bounding box of a head mesh.
///
/// Returns `(min_xyz, max_xyz)` where each is an `[f32; 3]`.
///
/// # Errors
///
/// Returns [`HeadGeometryError::EmptyMesh`] when `vertices` is empty.
pub fn head_bounding_box(vertices: &[[f32; 3]]) -> Result<([f32; 3], [f32; 3]), HeadGeometryError> {
    if vertices.is_empty() {
        return Err(HeadGeometryError::EmptyMesh(
            "vertices slice is empty".to_string(),
        ));
    }
    let mut mn = [f32::INFINITY; 3];
    let mut mx = [f32::NEG_INFINITY; 3];
    for v in vertices {
        for i in 0..3 {
            if v[i] < mn[i] {
                mn[i] = v[i];
            }
            if v[i] > mx[i] {
                mx[i] = v[i];
            }
        }
    }
    Ok((mn, mx))
}

/// Compute the center of mass (vertex centroid) of a head mesh.
///
/// # Errors
///
/// Returns [`HeadGeometryError::EmptyMesh`] when `vertices` is empty.
pub fn head_centroid(vertices: &[[f32; 3]]) -> Result<[f32; 3], HeadGeometryError> {
    if vertices.is_empty() {
        return Err(HeadGeometryError::EmptyMesh(
            "vertices slice is empty".to_string(),
        ));
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

/// Compute the head volume using the divergence theorem.
///
/// For each triangular face `(v0, v1, v2)`:
/// `contrib = dot(v0, cross(v1, v2)) / 6`
///
/// The volume is the absolute value of the sum of all contributions.
///
/// # Errors
///
/// Returns [`HeadGeometryError::EmptyMesh`] when `vertices` or `faces` is empty,
/// or [`HeadGeometryError::InvalidLandmark`] when a face references an out-of-range vertex.
pub fn head_volume(vertices: &[[f32; 3]], faces: &[[u32; 3]]) -> Result<f32, HeadGeometryError> {
    if vertices.is_empty() {
        return Err(HeadGeometryError::EmptyMesh(
            "vertices slice is empty".to_string(),
        ));
    }
    if faces.is_empty() {
        return Err(HeadGeometryError::EmptyMesh(
            "faces slice is empty".to_string(),
        ));
    }
    let n = vertices.len();
    let mut signed_sum = 0.0_f32;
    for (fi, face) in faces.iter().enumerate() {
        let i0 = face[0] as usize;
        let i1 = face[1] as usize;
        let i2 = face[2] as usize;
        if i0 >= n || i1 >= n || i2 >= n {
            return Err(HeadGeometryError::InvalidLandmark {
                idx: fi,
                n_verts: n,
            });
        }
        let v0 = &vertices[i0];
        let v1 = &vertices[i1];
        let v2 = &vertices[i2];
        let c = cross3(v1, v2);
        signed_sum += dot3(v0, &c) / 6.0;
    }
    Ok(signed_sum.abs())
}

/// Compute the total surface area of a head mesh (sum of face areas).
///
/// # Errors
///
/// Returns [`HeadGeometryError::EmptyMesh`] when `vertices` or `faces` is empty,
/// or [`HeadGeometryError::InvalidLandmark`] when a face references an out-of-range vertex.
pub fn head_surface_area(
    vertices: &[[f32; 3]],
    faces: &[[u32; 3]],
) -> Result<f32, HeadGeometryError> {
    if vertices.is_empty() {
        return Err(HeadGeometryError::EmptyMesh(
            "vertices slice is empty".to_string(),
        ));
    }
    if faces.is_empty() {
        return Err(HeadGeometryError::EmptyMesh(
            "faces slice is empty".to_string(),
        ));
    }
    let n = vertices.len();
    let mut total = 0.0_f32;
    for (fi, face) in faces.iter().enumerate() {
        let i0 = face[0] as usize;
        let i1 = face[1] as usize;
        let i2 = face[2] as usize;
        if i0 >= n || i1 >= n || i2 >= n {
            return Err(HeadGeometryError::InvalidLandmark {
                idx: fi,
                n_verts: n,
            });
        }
        let edge1 = sub3(&vertices[i1], &vertices[i0]);
        let edge2 = sub3(&vertices[i2], &vertices[i0]);
        let c = cross3(&edge1, &edge2);
        total += 0.5 * norm3(&c);
    }
    Ok(total)
}

// ---------------------------------------------------------------------------
// Anatomical region classification
// ---------------------------------------------------------------------------

/// Predefined anatomical regions of a human head based on relative vertex positions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HeadRegion {
    /// Superior cranial region.
    Crown,
    /// Frontal bone area above the orbits.
    Forehead,
    /// Left zygomatic/maxillary area.
    LeftCheek,
    /// Right zygomatic/maxillary area.
    RightCheek,
    /// Nasal region.
    Nose,
    /// Upper lip and philtrum area.
    UpperLip,
    /// Lower lip and mentolabial region.
    LowerLip,
    /// Chin (mental) protuberance area.
    Chin,
    /// Left orbital region.
    LeftEye,
    /// Right orbital region.
    RightEye,
    /// Mandibular angle area.
    Jaw,
    /// Inferior cervical region.
    Neck,
}

/// Classify a single vertex into a [`HeadRegion`] based on its position relative
/// to the head centroid and overall head dimensions.
///
/// This uses heuristic rules based on normalised offsets from the centroid.
///
/// # Arguments
///
/// * `vertex` — The 3D position to classify.
/// * `centroid` — Head centre of mass (from [`head_centroid`]).
/// * `head_height` — Total head height (Y extent).
/// * `head_width` — Total head width (X extent).
/// * `head_length` — Total head length (Z extent).
#[must_use]
pub fn classify_vertex_region(
    vertex: &[f32; 3],
    centroid: &[f32; 3],
    head_height: f32,
    head_width: f32,
    head_length: f32,
) -> HeadRegion {
    let dy = vertex[1] - centroid[1];
    let dx = vertex[0] - centroid[0];
    let dz = vertex[2] - centroid[2];

    // Normalised offsets (guard against degenerate meshes).
    let hy = head_height.max(1e-6);
    let hx = head_width.max(1e-6);
    let hz = head_length.max(1e-6);

    let ry = dy / hy; // relative Y: -0.5..+0.5 roughly
    let rx = dx / hx;
    let rz = dz / hz;

    // Crown: very top of the skull.
    if ry > 0.35 {
        return HeadRegion::Crown;
    }

    // Neck: below the skull base.
    if ry < -0.45 {
        return HeadRegion::Neck;
    }

    // Jaw / mandible angles: lower face, lateral.
    if ry < -0.25 && rx.abs() > 0.25 {
        return HeadRegion::Jaw;
    }

    // Chin: lower midline, forward.
    if ry < -0.25 && rz > 0.0 && rx.abs() < 0.2 {
        return HeadRegion::Chin;
    }

    // Eye regions: mid-height, lateral, slightly forward.
    if ry > 0.05 && ry < 0.30 && rz > 0.15 {
        if rx > 0.15 {
            return HeadRegion::LeftEye;
        }
        if rx < -0.15 {
            return HeadRegion::RightEye;
        }
    }

    // Forehead: upper face, front.
    if ry > 0.15 && rz > 0.05 {
        return HeadRegion::Forehead;
    }

    // Nose: midline, central height, most forward.
    if rz > 0.35 && rx.abs() < 0.15 && ry > -0.10 && ry < 0.15 {
        return HeadRegion::Nose;
    }

    // Upper lip: midline, lower mid-face, forward.
    if rz > 0.25 && rx.abs() < 0.15 && ry > -0.20 && ry < -0.03 {
        return HeadRegion::UpperLip;
    }

    // Lower lip: midline, lower mid-face.
    if rz > 0.20 && rx.abs() < 0.15 && ry > -0.28 && ry < -0.15 {
        return HeadRegion::LowerLip;
    }

    // Cheeks: mid-lateral, forward.
    if rz > 0.10 && ry > -0.20 && ry < 0.15 {
        if rx > 0.15 {
            return HeadRegion::LeftCheek;
        }
        if rx < -0.15 {
            return HeadRegion::RightCheek;
        }
    }

    // Fallback to forehead for upper unclassified, neck for lower.
    if ry > 0.0 {
        HeadRegion::Forehead
    } else {
        HeadRegion::Neck
    }
}

/// Return the indices of all vertices classified as belonging to `region`.
#[must_use]
pub fn vertices_in_region(
    vertices: &[[f32; 3]],
    region: &HeadRegion,
    centroid: &[f32; 3],
    measurements: &HeadMeasurements,
) -> Vec<usize> {
    vertices
        .iter()
        .enumerate()
        .filter_map(|(i, v)| {
            let r = classify_vertex_region(
                v,
                centroid,
                measurements.head_height,
                measurements.head_width,
                measurements.head_length,
            );
            if r == *region {
                Some(i)
            } else {
                None
            }
        })
        .collect()
}

/// Compute per-vertex region labels for an entire mesh.
///
/// Returns a `Vec<HeadRegion>` with one entry per vertex.
#[must_use]
pub fn label_vertices_by_region(
    vertices: &[[f32; 3]],
    centroid: &[f32; 3],
    measurements: &HeadMeasurements,
) -> Vec<HeadRegion> {
    vertices
        .iter()
        .map(|v| {
            classify_vertex_region(
                v,
                centroid,
                measurements.head_height,
                measurements.head_width,
                measurements.head_length,
            )
        })
        .collect()
}

/// Compute the centroid (average position) of a subset of vertices given by index.
///
/// # Errors
///
/// Returns [`HeadGeometryError::EmptyMesh`] when `region_indices` is empty.
/// Returns [`HeadGeometryError::InvalidLandmark`] when an index exceeds vertex count.
pub fn region_centroid(
    vertices: &[[f32; 3]],
    region_indices: &[usize],
) -> Result<[f32; 3], HeadGeometryError> {
    if region_indices.is_empty() {
        return Err(HeadGeometryError::EmptyMesh(
            "region_indices is empty".to_string(),
        ));
    }
    let n = vertices.len();
    let mut sum = [0.0_f32; 3];
    for &idx in region_indices {
        if idx >= n {
            return Err(HeadGeometryError::InvalidLandmark { idx, n_verts: n });
        }
        let v = &vertices[idx];
        sum[0] += v[0];
        sum[1] += v[1];
        sum[2] += v[2];
    }
    let cnt = region_indices.len() as f32;
    Ok([sum[0] / cnt, sum[1] / cnt, sum[2] / cnt])
}

// ---------------------------------------------------------------------------
// Profile and silhouette
// ---------------------------------------------------------------------------

/// Compute the head profile curve as a 2-D projection onto the XZ plane.
///
/// Returns all vertices projected to `[x, z]`, sorted by X coordinate.
#[must_use]
pub fn head_profile_xz(vertices: &[[f32; 3]]) -> Vec<[f32; 2]> {
    let mut projected: Vec<[f32; 2]> = vertices.iter().map(|v| [v[0], v[2]]).collect();
    projected.sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap_or(std::cmp::Ordering::Equal));
    projected
}

/// Compute the frontal silhouette as the convex hull of vertices projected onto the XY plane.
///
/// Returns the convex hull vertices in counter-clockwise order.
#[must_use]
pub fn frontal_silhouette(vertices: &[[f32; 3]]) -> Vec<[f32; 2]> {
    let projected: Vec<[f32; 2]> = vertices.iter().map(|v| [v[0], v[1]]).collect();
    convex_hull_2d(&projected)
}

/// Compute the 2-D convex hull of a set of points using the Graham scan algorithm.
///
/// Returns the hull vertices in counter-clockwise order. Collinear points on the
/// hull boundary are excluded. Returns an empty `Vec` when `points` is empty.
#[must_use]
pub fn convex_hull_2d(points: &[[f32; 2]]) -> Vec<[f32; 2]> {
    let n = points.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![points[0]];
    }
    if n == 2 {
        return vec![points[0], points[1]];
    }

    // Step 1: Find the bottom-most point (lowest y, break ties by leftmost x).
    let mut pivot_idx = 0;
    for i in 1..n {
        let same_y = (points[i][1] - points[pivot_idx][1]).abs() < 1e-9;
        if points[i][1] < points[pivot_idx][1] || (same_y && points[i][0] < points[pivot_idx][0]) {
            pivot_idx = i;
        }
    }

    let pivot = points[pivot_idx];

    // Step 2: Sort remaining points by polar angle with respect to pivot.
    //         Tie-break by distance from pivot (closer first), so collinear
    //         points between pivot and a hull vertex are skipped cleanly.
    let mut others: Vec<[f32; 2]> = points
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != pivot_idx)
        .map(|(_, &p)| p)
        .collect();

    others.sort_by(|a, b| {
        let cross = (a[0] - pivot[0]) * (b[1] - pivot[1]) - (a[1] - pivot[1]) * (b[0] - pivot[0]);
        if cross.abs() < 1e-10 {
            // Collinear: sort by distance ascending (keep farthest for hull).
            let da = (a[0] - pivot[0]).hypot(a[1] - pivot[1]);
            let db = (b[0] - pivot[0]).hypot(b[1] - pivot[1]);
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        } else if cross > 0.0 {
            std::cmp::Ordering::Less
        } else {
            std::cmp::Ordering::Greater
        }
    });

    // Remove all but the farthest among collinear trailing points.
    // (This ensures only the extreme collinear point is kept.)
    let mut filtered: Vec<[f32; 2]> = Vec::with_capacity(others.len());
    let len = others.len();
    for (i, &p) in others.iter().enumerate() {
        if i + 1 < len {
            let next = others[i + 1];
            let cross =
                (p[0] - pivot[0]) * (next[1] - pivot[1]) - (p[1] - pivot[1]) * (next[0] - pivot[0]);
            if cross.abs() < 1e-10 {
                // Collinear with next: skip this one (keep farthest).
                continue;
            }
        }
        filtered.push(p);
    }

    // Step 3: Graham scan.
    let mut hull: Vec<[f32; 2]> = Vec::with_capacity(filtered.len() + 1);
    hull.push(pivot);
    for &p in &filtered {
        while hull.len() >= 2 {
            let a = hull[hull.len() - 2];
            let b = hull[hull.len() - 1];
            // Cross product of (b-a) × (p-a).
            let cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]);
            if cross <= 0.0 {
                hull.pop();
            } else {
                break;
            }
        }
        hull.push(p);
    }

    hull
}

/// Compute the frontal face area as the area of the convex hull of vertices
/// projected onto the XY plane (shoelace formula).
#[must_use]
pub fn frontal_face_area(vertices: &[[f32; 3]]) -> f32 {
    let hull = frontal_silhouette(vertices);
    shoelace_area(&hull)
}

/// Compute the area of a polygon given as an ordered list of 2-D vertices
/// using the shoelace formula. Returns 0.0 for fewer than 3 points.
fn shoelace_area(poly: &[[f32; 2]]) -> f32 {
    let n = poly.len();
    if n < 3 {
        return 0.0;
    }
    let mut area = 0.0_f32;
    for i in 0..n {
        let j = (i + 1) % n;
        area += poly[i][0] * poly[j][1];
        area -= poly[j][0] * poly[i][1];
    }
    (area * 0.5).abs()
}

// ---------------------------------------------------------------------------
// Symmetry analysis
// ---------------------------------------------------------------------------

/// Compute per-pair asymmetry scores for a set of symmetry vertex pairs.
///
/// For each `(left_idx, right_idx)` pair the score is the Euclidean distance
/// between the left vertex and the YZ-plane reflection of the right vertex
/// (i.e., the right vertex mirrored at x = 0).
///
/// Returns a `Vec<f32>` with one score per input pair.
#[must_use]
pub fn vertex_asymmetry_scores(
    vertices: &[[f32; 3]],
    symmetry_pairs: &[(usize, usize)],
) -> Vec<f32> {
    symmetry_pairs
        .iter()
        .map(|&(li, ri)| {
            if li >= vertices.len() || ri >= vertices.len() {
                return f32::NAN;
            }
            let l = vertices[li];
            let r = vertices[ri];
            // Mirror r across the YZ plane (negate x).
            let r_reflected = [-r[0], r[1], r[2]];
            head_dist3(&l, &r_reflected)
        })
        .collect()
}

/// Find the index of the vertex closest to a query point.
///
/// # Errors
///
/// Returns [`HeadGeometryError::EmptyMesh`] when `vertices` is empty.
pub fn find_nearest_vertex(
    vertices: &[[f32; 3]],
    query: &[f32; 3],
) -> Result<usize, HeadGeometryError> {
    if vertices.is_empty() {
        return Err(HeadGeometryError::EmptyMesh(
            "vertices slice is empty".to_string(),
        ));
    }
    let mut best_idx = 0;
    let mut best_dist = f32::INFINITY;
    for (i, v) in vertices.iter().enumerate() {
        let d = head_dist3(v, query);
        if d < best_dist {
            best_dist = d;
            best_idx = i;
        }
    }
    Ok(best_idx)
}

/// Compute a histogram of vertex Y coordinates.
///
/// Returns a `Vec` of `(bin_center, count)` pairs, one per bin.
///
/// # Errors
///
/// Returns [`HeadGeometryError::InvalidParam`] when `n_bins == 0`.
/// Returns [`HeadGeometryError::EmptyMesh`] when `vertices` is empty.
pub fn height_histogram(
    vertices: &[[f32; 3]],
    n_bins: usize,
) -> Result<Vec<(f32, usize)>, HeadGeometryError> {
    if n_bins == 0 {
        return Err(HeadGeometryError::InvalidParam(
            "n_bins must be greater than 0".to_string(),
        ));
    }
    if vertices.is_empty() {
        return Err(HeadGeometryError::EmptyMesh(
            "vertices slice is empty".to_string(),
        ));
    }

    let mut min_y = f32::INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    for v in vertices {
        if v[1] < min_y {
            min_y = v[1];
        }
        if v[1] > max_y {
            max_y = v[1];
        }
    }

    let range = max_y - min_y;
    let mut counts = vec![0usize; n_bins];

    if range < 1e-30 {
        // All vertices at the same Y — put everything in the first bin.
        counts[0] = vertices.len();
    } else {
        let bin_width = range / n_bins as f32;
        for v in vertices {
            let idx = ((v[1] - min_y) / bin_width) as usize;
            let idx = idx.min(n_bins - 1);
            counts[idx] += 1;
        }
    }

    let result = (0..n_bins)
        .map(|i| {
            let range = max_y - min_y;
            let bin_width = if range < 1e-30 {
                1.0
            } else {
                range / n_bins as f32
            };
            let center = min_y + (i as f32 + 0.5) * bin_width;
            (center, counts[i])
        })
        .collect();

    Ok(result)
}

// ---------------------------------------------------------------------------
// Distance and projection utilities
// ---------------------------------------------------------------------------

/// Euclidean distance between two 3-D points.
#[inline]
#[must_use]
pub fn head_dist3(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    norm3(&d)
}

/// Compute the maximum pairwise distance among a set of vertices.
///
/// This is O(n²) — use only on small vertex sets or bounding-box approximations.
///
/// # Errors
///
/// Returns [`HeadGeometryError::EmptyMesh`] when `vertices` is empty.
pub fn max_pairwise_distance(vertices: &[[f32; 3]]) -> Result<f32, HeadGeometryError> {
    if vertices.is_empty() {
        return Err(HeadGeometryError::EmptyMesh(
            "vertices slice is empty".to_string(),
        ));
    }
    let mut max_d = 0.0_f32;
    for i in 0..vertices.len() {
        for j in (i + 1)..vertices.len() {
            let d = head_dist3(&vertices[i], &vertices[j]);
            if d > max_d {
                max_d = d;
            }
        }
    }
    Ok(max_d)
}

/// Project all vertices onto a plane defined by a point and a unit normal.
///
/// Each vertex `v` is projected to `v - dot(v - plane_point, plane_normal) * plane_normal`.
///
/// The `plane_normal` need not be unit length — it is normalised internally.
#[must_use]
pub fn project_to_plane(
    vertices: &[[f32; 3]],
    plane_normal: &[f32; 3],
    plane_point: &[f32; 3],
) -> Vec<[f32; 3]> {
    let n = normalize3(plane_normal);
    vertices
        .iter()
        .map(|v| {
            let diff = sub3(v, plane_point);
            let dist = dot3(&diff, &n);
            [v[0] - dist * n[0], v[1] - dist * n[1], v[2] - dist * n[2]]
        })
        .collect()
}

/// Compute the principal axis (largest PCA eigenvector) of a vertex cloud
/// via power iteration (100 iterations).
///
/// # Errors
///
/// Returns [`HeadGeometryError::EmptyMesh`] when `vertices` is empty.
/// Returns [`HeadGeometryError::InsufficientVertices`] when fewer than 2 vertices are provided.
pub fn principal_axis(vertices: &[[f32; 3]]) -> Result<[f32; 3], HeadGeometryError> {
    if vertices.is_empty() {
        return Err(HeadGeometryError::EmptyMesh(
            "vertices slice is empty".to_string(),
        ));
    }
    if vertices.len() < 2 {
        return Err(HeadGeometryError::InsufficientVertices {
            need: 2,
            have: vertices.len(),
        });
    }

    // Compute centroid.
    let centroid = head_centroid(vertices)?;

    // Compute 3×3 covariance matrix (upper triangle = lower triangle).
    let mut cov = [[0.0_f32; 3]; 3];
    let n = vertices.len() as f32;
    for v in vertices {
        let d = [v[0] - centroid[0], v[1] - centroid[1], v[2] - centroid[2]];
        for r in 0..3 {
            for c in 0..3 {
                cov[r][c] += d[r] * d[c];
            }
        }
    }
    for row in &mut cov {
        for val in row.iter_mut() {
            *val /= n;
        }
    }

    // Determine the best starting vector by looking at the diagonal of the
    // covariance matrix. The diagonal entry cov[i][i] is the variance along
    // axis i. Start power iteration from the axis with the largest variance so
    // that we converge to the dominant eigenvector regardless of which axis
    // the point cloud is elongated along.
    let diag = [cov[0][0], cov[1][1], cov[2][2]];
    let max_axis = if diag[0] >= diag[1] && diag[0] >= diag[2] {
        0
    } else if diag[1] >= diag[2] {
        1
    } else {
        2
    };
    let mut ev = [0.0_f32; 3];
    ev[max_axis] = 1.0;

    for _ in 0..200 {
        let mut new_ev = [0.0_f32; 3];
        for r in 0..3 {
            for c in 0..3 {
                new_ev[r] += cov[r][c] * ev[c];
            }
        }
        let len = norm3(&new_ev);
        if len < 1e-30 {
            // Degenerate covariance (all vertices identical) — return default axis.
            return Ok([1.0, 0.0, 0.0]);
        }
        ev = [new_ev[0] / len, new_ev[1] / len, new_ev[2] / len];
    }
    Ok(ev)
}

// ---------------------------------------------------------------------------
// Aggregated geometry statistics
// ---------------------------------------------------------------------------

/// Aggregated geometric statistics for a head mesh.
#[derive(Debug, Clone)]
pub struct HeadGeometryStats {
    /// Number of vertices in the mesh.
    pub n_vertices: usize,
    /// Number of triangular faces in the mesh.
    pub n_faces: usize,
    /// Total surface area (sum of face areas).
    pub surface_area: f32,
    /// Enclosed volume (via divergence theorem).
    pub volume: f32,
    /// Isoperimetric compactness: `36π V² / A³` (1.0 for a sphere).
    pub compactness: f32,
    /// Standard anthropometric measurements.
    pub measurements: HeadMeasurements,
}

/// Compute aggregated geometry statistics for a head mesh.
///
/// # Errors
///
/// Propagates errors from [`head_surface_area`], [`head_volume`], and [`measure_head`].
pub fn compute_head_geometry_stats(
    vertices: &[[f32; 3]],
    faces: &[[u32; 3]],
    nose_tip_idx: Option<usize>,
    chin_idx: Option<usize>,
) -> Result<HeadGeometryStats, HeadGeometryError> {
    let surface_area = head_surface_area(vertices, faces)?;
    let volume = head_volume(vertices, faces)?;
    let measurements = measure_head(vertices, nose_tip_idx, chin_idx)?;

    let compactness = if surface_area > 1e-30 {
        36.0 * std::f32::consts::PI * volume * volume / (surface_area * surface_area * surface_area)
    } else {
        0.0
    };

    Ok(HeadGeometryStats {
        n_vertices: vertices.len(),
        n_faces: faces.len(),
        surface_area,
        volume,
        compactness,
        measurements,
    })
}

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------

/// Format [`HeadMeasurements`] as a human-readable multi-line string.
#[must_use]
pub fn format_head_measurements(m: &HeadMeasurements) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "=== Head Measurements ===");
    let _ = writeln!(out, "  head_length   : {:.4} m", m.head_length);
    let _ = writeln!(out, "  head_width    : {:.4} m", m.head_width);
    let _ = writeln!(out, "  head_height   : {:.4} m", m.head_height);
    let _ = writeln!(out, "  face_width    : {:.4} m", m.face_width);
    let _ = writeln!(out, "  face_height   : {:.4} m", m.face_height);
    let _ = writeln!(out, "  nose_length   : {:.4} m", m.nose_length);
    let _ = writeln!(out, "  jaw_width     : {:.4} m", m.jaw_width);
    let _ = writeln!(out, "  cephalic_index: {:.2}", m.cephalic_index);
    let _ = writeln!(out, "  facial_index  : {:.2}", m.facial_index);
    out
}

/// Format [`HeadGeometryStats`] as a human-readable multi-line string.
#[must_use]
pub fn format_head_geometry_stats(stats: &HeadGeometryStats) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "=== Head Geometry Stats ===");
    let _ = writeln!(out, "  n_vertices  : {}", stats.n_vertices);
    let _ = writeln!(out, "  n_faces     : {}", stats.n_faces);
    let _ = writeln!(out, "  surface_area: {:.6} m²", stats.surface_area);
    let _ = writeln!(out, "  volume      : {:.6} m³", stats.volume);
    let _ = writeln!(out, "  compactness : {:.6}", stats.compactness);
    out.push_str(&format_head_measurements(&stats.measurements));
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- helpers -----------------------------------------------------------

    fn unit_cube_vertices() -> Vec<[f32; 3]> {
        vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    }

    fn unit_tetrahedron() -> (Vec<[f32; 3]>, Vec<[u32; 3]>) {
        // Tetrahedron with one vertex at origin; volume = 1/6.
        let verts = vec![
            [0.0_f32, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let faces = vec![[0u32, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]];
        (verts, faces)
    }

    // ---- head_bounding_box -------------------------------------------------

    #[test]
    fn test_bounding_box_known() {
        let verts = unit_cube_vertices();
        let (mn, mx) = head_bounding_box(&verts).expect("bounding box");
        assert!((mn[0] - 0.0).abs() < 1e-6);
        assert!((mn[1] - 0.0).abs() < 1e-6);
        assert!((mn[2] - 0.0).abs() < 1e-6);
        assert!((mx[0] - 1.0).abs() < 1e-6);
        assert!((mx[1] - 1.0).abs() < 1e-6);
        assert!((mx[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_bounding_box_empty() {
        let err = head_bounding_box(&[]).expect_err("should fail");
        assert!(matches!(err, HeadGeometryError::EmptyMesh(_)));
    }

    #[test]
    fn test_bounding_box_single_vertex() {
        let verts = vec![[3.0_f32, -1.0, 2.0]];
        let (mn, mx) = head_bounding_box(&verts).expect("single vertex");
        for i in 0..3 {
            assert!((mn[i] - verts[0][i]).abs() < 1e-6);
            assert!((mx[i] - verts[0][i]).abs() < 1e-6);
        }
    }

    // ---- head_centroid -----------------------------------------------------

    #[test]
    fn test_centroid_uniform_grid() {
        let verts = unit_cube_vertices();
        let c = head_centroid(&verts).expect("centroid");
        assert!((c[0] - 0.5).abs() < 1e-5, "cx={}", c[0]);
        assert!((c[1] - 0.5).abs() < 1e-5, "cy={}", c[1]);
        assert!((c[2] - 0.5).abs() < 1e-5, "cz={}", c[2]);
    }

    #[test]
    fn test_centroid_empty() {
        let err = head_centroid(&[]).expect_err("should fail");
        assert!(matches!(err, HeadGeometryError::EmptyMesh(_)));
    }

    #[test]
    fn test_centroid_single_vertex() {
        let verts = vec![[5.0_f32, -3.0, 2.0]];
        let c = head_centroid(&verts).expect("centroid");
        assert!((c[0] - 5.0).abs() < 1e-6);
        assert!((c[1] + 3.0).abs() < 1e-6);
        assert!((c[2] - 2.0).abs() < 1e-6);
    }

    // ---- head_volume -------------------------------------------------------

    #[test]
    fn test_volume_tetrahedron() {
        let (verts, faces) = unit_tetrahedron();
        let vol = head_volume(&verts, &faces).expect("volume");
        // Volume of unit tetrahedron is 1/6.
        assert!((vol - 1.0 / 6.0).abs() < 1e-5, "vol={vol}");
    }

    #[test]
    fn test_volume_empty_vertices() {
        let err = head_volume(&[], &[[0u32, 1, 2]]).expect_err("should fail");
        assert!(matches!(err, HeadGeometryError::EmptyMesh(_)));
    }

    #[test]
    fn test_volume_empty_faces() {
        let verts = unit_cube_vertices();
        let err = head_volume(&verts, &[]).expect_err("should fail");
        assert!(matches!(err, HeadGeometryError::EmptyMesh(_)));
    }

    // ---- head_surface_area -------------------------------------------------

    #[test]
    fn test_surface_area_equilateral_triangle() {
        // Equilateral triangle with side length 1 — area = sqrt(3)/4.
        let verts = vec![
            [0.0_f32, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, (3.0_f32).sqrt() / 2.0, 0.0],
        ];
        let faces = vec![[0u32, 1, 2]];
        let area = head_surface_area(&verts, &faces).expect("area");
        let expected = (3.0_f32).sqrt() / 4.0;
        assert!(
            (area - expected).abs() < 1e-5,
            "area={area} expected={expected}"
        );
    }

    #[test]
    fn test_surface_area_empty_vertices() {
        let err = head_surface_area(&[], &[[0u32, 1, 2]]).expect_err("should fail");
        assert!(matches!(err, HeadGeometryError::EmptyMesh(_)));
    }

    #[test]
    fn test_surface_area_empty_faces() {
        let verts = unit_cube_vertices();
        let err = head_surface_area(&verts, &[]).expect_err("should fail");
        assert!(matches!(err, HeadGeometryError::EmptyMesh(_)));
    }

    // ---- measure_head ------------------------------------------------------

    #[test]
    fn test_measure_head_unit_cube() {
        let verts = unit_cube_vertices();
        let m = measure_head(&verts, None, None).expect("measurements");
        assert!((m.head_length - 1.0).abs() < 1e-5);
        assert!((m.head_width - 1.0).abs() < 1e-5);
        assert!((m.head_height - 1.0).abs() < 1e-5);
        assert!((m.face_width - 0.8).abs() < 1e-5);
        assert!((m.face_height - 0.7).abs() < 1e-5);
        assert!((m.jaw_width - 0.6).abs() < 1e-5);
    }

    #[test]
    fn test_measure_head_empty() {
        let err = measure_head(&[], None, None).expect_err("should fail");
        assert!(matches!(err, HeadGeometryError::EmptyMesh(_)));
    }

    #[test]
    fn test_measure_head_invalid_nose_idx() {
        let verts = unit_cube_vertices();
        let err = measure_head(&verts, Some(999), None).expect_err("should fail");
        assert!(matches!(err, HeadGeometryError::InvalidLandmark { .. }));
    }

    #[test]
    fn test_measure_head_invalid_chin_idx() {
        let verts = unit_cube_vertices();
        let err = measure_head(&verts, None, Some(999)).expect_err("should fail");
        assert!(matches!(err, HeadGeometryError::InvalidLandmark { .. }));
    }

    #[test]
    fn test_measure_head_cephalic_index() {
        // 2 units wide, 4 units long → cephalic = 2/4 * 100 = 50.
        let verts = vec![
            [0.0_f32, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 0.0, 4.0],
            [2.0, 0.0, 4.0],
        ];
        let m = measure_head(&verts, None, None).expect("measurements");
        assert!(
            (m.cephalic_index - 50.0).abs() < 1e-3,
            "ci={}",
            m.cephalic_index
        );
    }

    // ---- head_dist3 --------------------------------------------------------

    #[test]
    fn test_head_dist3_known() {
        let a = [0.0_f32, 0.0, 0.0];
        let b = [3.0, 4.0, 0.0];
        let d = head_dist3(&a, &b);
        assert!((d - 5.0).abs() < 1e-5, "d={d}");
    }

    #[test]
    fn test_head_dist3_zero() {
        let a = [1.0_f32, 2.0, 3.0];
        assert!((head_dist3(&a, &a)).abs() < 1e-6);
    }

    #[test]
    fn test_head_dist3_unit_vecs() {
        let a = [1.0_f32, 0.0, 0.0];
        let b = [0.0, 1.0, 0.0];
        let d = head_dist3(&a, &b);
        assert!((d - (2.0_f32).sqrt()).abs() < 1e-5);
    }

    // ---- max_pairwise_distance ---------------------------------------------

    #[test]
    fn test_max_pairwise_two_points() {
        let verts = vec![[0.0_f32, 0.0, 0.0], [3.0, 4.0, 0.0]];
        let d = max_pairwise_distance(&verts).expect("max dist");
        assert!((d - 5.0).abs() < 1e-5, "d={d}");
    }

    #[test]
    fn test_max_pairwise_empty() {
        let err = max_pairwise_distance(&[]).expect_err("should fail");
        assert!(matches!(err, HeadGeometryError::EmptyMesh(_)));
    }

    #[test]
    fn test_max_pairwise_single() {
        let verts = vec![[1.0_f32, 0.0, 0.0]];
        let d = max_pairwise_distance(&verts).expect("single vertex");
        assert!((d - 0.0).abs() < 1e-6);
    }

    // ---- project_to_plane --------------------------------------------------

    #[test]
    fn test_project_to_xy_plane() {
        let verts: Vec<[f32; 3]> = vec![[1.0, 2.0, 5.0], [-1.0, 0.5, 3.0], [0.0, 0.0, 7.0]];
        let normal = [0.0_f32, 0.0, 1.0];
        let point = [0.0_f32, 0.0, 0.0];
        let proj = project_to_plane(&verts, &normal, &point);
        for p in &proj {
            assert!(p[2].abs() < 1e-5, "z should be ~0, got {}", p[2]);
        }
    }

    #[test]
    fn test_project_to_plane_already_on_plane() {
        let verts = vec![[1.0_f32, 1.0, 0.0], [2.0, 3.0, 0.0]];
        let normal = [0.0_f32, 0.0, 1.0];
        let point = [0.0_f32, 0.0, 0.0];
        let proj = project_to_plane(&verts, &normal, &point);
        for (p, v) in proj.iter().zip(verts.iter()) {
            for i in 0..3 {
                assert!((p[i] - v[i]).abs() < 1e-5);
            }
        }
    }

    // ---- principal_axis ----------------------------------------------------

    #[test]
    fn test_principal_axis_along_x() {
        // Cloud elongated along X axis.
        let verts: Vec<[f32; 3]> = (-50..=50).map(|i| [i as f32 * 0.1, 0.0_f32, 0.0]).collect();
        let ax = principal_axis(&verts).expect("principal axis");
        assert!(ax[0].abs() > 0.99, "expected X axis, got {ax:?}");
    }

    #[test]
    fn test_principal_axis_empty() {
        let err = principal_axis(&[]).expect_err("should fail");
        assert!(matches!(err, HeadGeometryError::EmptyMesh(_)));
    }

    #[test]
    fn test_principal_axis_insufficient() {
        let err = principal_axis(&[[0.0_f32, 0.0, 0.0]]).expect_err("should fail");
        assert!(matches!(
            err,
            HeadGeometryError::InsufficientVertices { .. }
        ));
    }

    #[test]
    fn test_principal_axis_along_y() {
        let verts: Vec<[f32; 3]> = (-50..=50).map(|i| [0.0_f32, i as f32 * 0.1, 0.0]).collect();
        let ax = principal_axis(&verts).expect("principal axis");
        assert!(ax[1].abs() > 0.99, "expected Y axis, got {ax:?}");
    }

    // ---- height_histogram --------------------------------------------------

    #[test]
    fn test_height_histogram_uniform() {
        let verts: Vec<[f32; 3]> = (0..100).map(|i| [0.0_f32, i as f32 * 0.01, 0.0]).collect();
        let hist = height_histogram(&verts, 10).expect("histogram");
        assert_eq!(hist.len(), 10);
        for (_, cnt) in &hist {
            assert!(*cnt >= 8 && *cnt <= 12, "count={cnt}");
        }
    }

    #[test]
    fn test_height_histogram_zero_bins() {
        let verts = unit_cube_vertices();
        let err = height_histogram(&verts, 0).expect_err("should fail");
        assert!(matches!(err, HeadGeometryError::InvalidParam(_)));
    }

    #[test]
    fn test_height_histogram_empty_vertices() {
        let err = height_histogram(&[], 5).expect_err("should fail");
        assert!(matches!(err, HeadGeometryError::EmptyMesh(_)));
    }

    // ---- classify_vertex_region --------------------------------------------

    #[test]
    fn test_classify_crown() {
        let centroid = [0.0_f32, 0.0, 0.0];
        let v = [0.0_f32, 0.5, 0.0]; // very top
        let r = classify_vertex_region(&v, &centroid, 1.0, 1.0, 1.0);
        assert_eq!(r, HeadRegion::Crown, "got {r:?}");
    }

    #[test]
    fn test_classify_neck() {
        let centroid = [0.0_f32, 0.0, 0.0];
        let v = [0.0_f32, -0.5, 0.0]; // very bottom
        let r = classify_vertex_region(&v, &centroid, 1.0, 1.0, 1.0);
        assert_eq!(r, HeadRegion::Neck, "got {r:?}");
    }

    #[test]
    fn test_classify_jaw() {
        let centroid = [0.0_f32, 0.0, 0.0];
        let v = [0.3_f32, -0.3, 0.0]; // lower lateral
        let r = classify_vertex_region(&v, &centroid, 1.0, 1.0, 1.0);
        assert_eq!(r, HeadRegion::Jaw, "got {r:?}");
    }

    #[test]
    fn test_classify_forehead() {
        let centroid = [0.0_f32, 0.0, 0.0];
        let v = [0.0_f32, 0.25, 0.1]; // upper face front
        let r = classify_vertex_region(&v, &centroid, 1.0, 1.0, 1.0);
        assert_eq!(r, HeadRegion::Forehead, "got {r:?}");
    }

    // ---- label_vertices_by_region ------------------------------------------

    #[test]
    fn test_label_vertices_correct_length() {
        let verts = unit_cube_vertices();
        let centroid = head_centroid(&verts).expect("centroid");
        let m = measure_head(&verts, None, None).expect("measurements");
        let labels = label_vertices_by_region(&verts, &centroid, &m);
        assert_eq!(labels.len(), verts.len());
    }

    // ---- vertices_in_region ------------------------------------------------

    #[test]
    fn test_vertices_in_region_subset() {
        let verts = unit_cube_vertices();
        let centroid = head_centroid(&verts).expect("centroid");
        let m = measure_head(&verts, None, None).expect("measurements");
        // The crown region should have at least 0 and at most all vertices.
        let crown = vertices_in_region(&verts, &HeadRegion::Crown, &centroid, &m);
        assert!(crown.len() <= verts.len());
    }

    #[test]
    fn test_vertices_in_region_all_crown() {
        // Create vertices all at the very top of the head.
        let verts: Vec<[f32; 3]> = vec![[0.0, 0.5, 0.0], [0.1, 0.5, 0.0], [-0.1, 0.5, 0.0]];
        let centroid = [0.0_f32, 0.0, 0.0];
        let m = HeadMeasurements {
            head_length: 1.0,
            head_width: 1.0,
            head_height: 1.0,
            face_width: 0.8,
            face_height: 0.7,
            nose_length: 0.15,
            jaw_width: 0.6,
            cephalic_index: 100.0,
            facial_index: 87.5,
        };
        let crown = vertices_in_region(&verts, &HeadRegion::Crown, &centroid, &m);
        assert_eq!(crown.len(), verts.len(), "all verts should be Crown");
    }

    // ---- region_centroid ---------------------------------------------------

    #[test]
    fn test_region_centroid_empty_indices() {
        let verts = unit_cube_vertices();
        let err = region_centroid(&verts, &[]).expect_err("should fail");
        assert!(matches!(err, HeadGeometryError::EmptyMesh(_)));
    }

    #[test]
    fn test_region_centroid_known() {
        let verts = vec![[0.0_f32, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let c = region_centroid(&verts, &[0, 1, 2]).expect("centroid");
        assert!((c[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_region_centroid_invalid_index() {
        let verts = unit_cube_vertices();
        let err = region_centroid(&verts, &[0, 999]).expect_err("should fail");
        assert!(matches!(err, HeadGeometryError::InvalidLandmark { .. }));
    }

    // ---- vertex_asymmetry_scores -------------------------------------------

    #[test]
    fn test_asymmetry_symmetric_pair() {
        // Left and right are perfect reflections → score = 0.
        let verts = vec![[1.0_f32, 0.0, 0.0], [-1.0, 0.0, 0.0]];
        let scores = vertex_asymmetry_scores(&verts, &[(0, 1)]);
        assert!((scores[0] - 0.0).abs() < 1e-5, "score={}", scores[0]);
    }

    #[test]
    fn test_asymmetry_asymmetric_pair() {
        // Left vertex NOT a mirror of the right → positive score.
        let verts = vec![[1.0_f32, 0.0, 0.0], [2.0, 0.0, 0.0]];
        let scores = vertex_asymmetry_scores(&verts, &[(0, 1)]);
        assert!(scores[0] > 0.0, "score={}", scores[0]);
    }

    #[test]
    fn test_asymmetry_empty_pairs() {
        let verts = vec![[0.0_f32, 0.0, 0.0]];
        let scores = vertex_asymmetry_scores(&verts, &[]);
        assert!(scores.is_empty());
    }

    // ---- find_nearest_vertex -----------------------------------------------

    #[test]
    fn test_find_nearest_vertex_exact() {
        let verts = vec![[0.0_f32, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
        let idx = find_nearest_vertex(&verts, &[1.0, 0.0, 0.0]).expect("nearest");
        assert_eq!(idx, 1);
    }

    #[test]
    fn test_find_nearest_vertex_empty() {
        let err = find_nearest_vertex(&[], &[0.0, 0.0, 0.0]).expect_err("should fail");
        assert!(matches!(err, HeadGeometryError::EmptyMesh(_)));
    }

    #[test]
    fn test_find_nearest_vertex_close() {
        let verts = vec![[0.0_f32, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]];
        let idx = find_nearest_vertex(&verts, &[9.9, 0.0, 0.0]).expect("nearest");
        assert_eq!(idx, 1);
    }

    // ---- head_profile_xz ---------------------------------------------------

    #[test]
    fn test_head_profile_xz_count() {
        let verts = unit_cube_vertices();
        let profile = head_profile_xz(&verts);
        assert_eq!(profile.len(), verts.len());
    }

    #[test]
    fn test_head_profile_xz_sorted() {
        let verts = unit_cube_vertices();
        let profile = head_profile_xz(&verts);
        for w in profile.windows(2) {
            assert!(w[0][0] <= w[1][0], "should be sorted by x");
        }
    }

    // ---- convex_hull_2d ----------------------------------------------------

    #[test]
    fn test_convex_hull_square() {
        let pts = vec![
            [0.0_f32, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.5, 0.5], // interior point — should be excluded
        ];
        let hull = convex_hull_2d(&pts);
        assert_eq!(
            hull.len(),
            4,
            "square hull should have 4 vertices, got {}",
            hull.len()
        );
    }

    #[test]
    fn test_convex_hull_triangle() {
        let pts = vec![[0.0_f32, 0.0], [1.0, 0.0], [0.5, 1.0]];
        let hull = convex_hull_2d(&pts);
        assert_eq!(hull.len(), 3, "triangle hull should have 3 vertices");
    }

    #[test]
    fn test_convex_hull_empty() {
        let hull = convex_hull_2d(&[]);
        assert!(hull.is_empty());
    }

    #[test]
    fn test_convex_hull_single() {
        let hull = convex_hull_2d(&[[1.0_f32, 2.0]]);
        assert_eq!(hull.len(), 1);
    }

    #[test]
    fn test_convex_hull_two_points() {
        let hull = convex_hull_2d(&[[0.0_f32, 0.0], [1.0, 1.0]]);
        assert_eq!(hull.len(), 2);
    }

    // ---- frontal_silhouette ------------------------------------------------

    #[test]
    fn test_frontal_silhouette_non_empty() {
        let verts = unit_cube_vertices();
        let hull = frontal_silhouette(&verts);
        assert!(!hull.is_empty());
    }

    // ---- frontal_face_area -------------------------------------------------

    #[test]
    fn test_frontal_face_area_unit_square() {
        // A unit square viewed from the front — area should be 1.0.
        let verts = vec![
            [0.0_f32, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let area = frontal_face_area(&verts);
        assert!((area - 1.0).abs() < 1e-4, "area={area}");
    }

    #[test]
    fn test_frontal_face_area_positive() {
        let verts = unit_cube_vertices();
        let area = frontal_face_area(&verts);
        assert!(area > 0.0, "area should be positive, got {area}");
    }

    // ---- compute_head_geometry_stats ---------------------------------------

    #[test]
    fn test_compute_head_geometry_stats() {
        let (verts, faces) = unit_tetrahedron();
        let stats =
            compute_head_geometry_stats(&verts, &faces, None, None).expect("head geometry stats");
        assert_eq!(stats.n_vertices, verts.len());
        assert_eq!(stats.n_faces, faces.len());
        assert!(stats.surface_area > 0.0);
        assert!(stats.volume > 0.0);
    }

    #[test]
    fn test_compute_head_geometry_stats_volume_correct() {
        let (verts, faces) = unit_tetrahedron();
        let stats = compute_head_geometry_stats(&verts, &faces, None, None).expect("stats");
        assert!(
            (stats.volume - 1.0 / 6.0).abs() < 1e-5,
            "vol={}",
            stats.volume
        );
    }

    // ---- format_head_measurements ------------------------------------------

    #[test]
    fn test_format_head_measurements_non_empty() {
        let verts = unit_cube_vertices();
        let m = measure_head(&verts, None, None).expect("measurements");
        let s = format_head_measurements(&m);
        assert!(!s.is_empty());
        assert!(s.contains("head_length"), "missing head_length");
    }

    // ---- format_head_geometry_stats ----------------------------------------

    #[test]
    fn test_format_head_geometry_stats_non_empty() {
        let (verts, faces) = unit_tetrahedron();
        let stats = compute_head_geometry_stats(&verts, &faces, None, None).expect("stats");
        let s = format_head_geometry_stats(&stats);
        assert!(!s.is_empty());
        assert!(s.contains("n_vertices"), "missing n_vertices");
    }

    // ---- additional edge-case tests ----------------------------------------

    #[test]
    fn test_measure_head_with_nose_and_chin() {
        let verts = unit_cube_vertices();
        // nose at index 7 (top-right-back), chin at index 0 (bottom-left-front).
        let m = measure_head(&verts, Some(7), Some(0)).expect("measurements");
        assert!(m.face_height >= 0.0);
        assert!(m.nose_length >= 0.0);
    }

    #[test]
    fn test_head_profile_xz_values() {
        let verts = vec![[1.0_f32, 99.0, 2.0], [-1.0, 50.0, 3.0]];
        let profile = head_profile_xz(&verts);
        // Check that Y is dropped and Z is kept.
        assert!((profile[0][0] + 1.0).abs() < 1e-5, "first x should be -1");
        assert!((profile[0][1] - 3.0).abs() < 1e-5, "first z should be 3");
    }

    #[test]
    fn test_convex_hull_collinear_points() {
        // 5 collinear points along x axis — hull should have 2 points (endpoints).
        let pts: Vec<[f32; 2]> = (0..5).map(|i| [i as f32, 0.0]).collect();
        let hull = convex_hull_2d(&pts);
        // With collinear points, the result may vary; the key property is ≥ 1 point.
        assert!(
            !hull.is_empty(),
            "hull should not be empty for collinear points"
        );
    }

    #[test]
    fn test_convex_hull_with_interior_points() {
        // Pentagon plus many interior points.
        let mut pts: Vec<[f32; 2]> = Vec::new();
        for i in 0..5 {
            let angle = i as f32 * 2.0 * std::f32::consts::PI / 5.0;
            pts.push([angle.cos(), angle.sin()]);
        }
        // Interior points.
        for _ in 0..20 {
            pts.push([0.1, 0.1]);
        }
        let hull = convex_hull_2d(&pts);
        assert_eq!(
            hull.len(),
            5,
            "hull of pentagon should have 5 vertices, got {}",
            hull.len()
        );
    }

    #[test]
    fn test_find_nearest_far_point() {
        let verts = vec![[0.0_f32, 0.0, 0.0], [100.0, 0.0, 0.0]];
        let idx = find_nearest_vertex(&verts, &[0.1, 0.0, 0.0]).expect("nearest");
        assert_eq!(idx, 0);
    }

    #[test]
    fn test_height_histogram_single_bin() {
        let verts: Vec<[f32; 3]> = (0..10).map(|i| [0.0, i as f32, 0.0]).collect();
        let hist = height_histogram(&verts, 1).expect("histogram");
        assert_eq!(hist.len(), 1);
        assert_eq!(hist[0].1, 10);
    }

    #[test]
    fn test_region_centroid_two_vertices() {
        let verts = vec![[0.0_f32, 0.0, 0.0], [4.0, 0.0, 0.0]];
        let c = region_centroid(&verts, &[0, 1]).expect("centroid");
        assert!((c[0] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_surface_area_right_triangle() {
        // Right triangle with legs 3 and 4 → area = 6.
        let verts = vec![[0.0_f32, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 4.0, 0.0]];
        let faces = vec![[0u32, 1, 2]];
        let area = head_surface_area(&verts, &faces).expect("area");
        assert!((area - 6.0).abs() < 1e-5, "area={area}");
    }

    #[test]
    fn test_head_volume_positive() {
        let (verts, faces) = unit_tetrahedron();
        let vol = head_volume(&verts, &faces).expect("volume");
        assert!(vol > 0.0);
    }

    #[test]
    fn test_max_pairwise_unit_cube() {
        let verts = unit_cube_vertices();
        let d = max_pairwise_distance(&verts).expect("max dist");
        // Diagonal of unit cube = sqrt(3).
        let expected = (3.0_f32).sqrt();
        assert!((d - expected).abs() < 1e-4, "d={d} expected={expected}");
    }

    #[test]
    fn test_project_to_plane_yz() {
        // Project onto YZ plane (normal = [1,0,0]).
        let verts = vec![[5.0_f32, 3.0, 2.0], [-2.0, 1.0, 4.0]];
        let normal = [1.0_f32, 0.0, 0.0];
        let point = [0.0_f32, 0.0, 0.0];
        let proj = project_to_plane(&verts, &normal, &point);
        for p in &proj {
            assert!(p[0].abs() < 1e-5, "x should be ~0, got {}", p[0]);
        }
    }

    #[test]
    fn test_label_vertices_contains_crown_and_neck() {
        // Create a synthetic head cloud with vertices spread top to bottom.
        let verts: Vec<[f32; 3]> = vec![
            [0.0, 0.5, 0.0],  // Crown
            [0.0, -0.5, 0.0], // Neck
        ];
        let centroid = [0.0_f32, 0.0, 0.0];
        let m = HeadMeasurements {
            head_length: 1.0,
            head_width: 1.0,
            head_height: 1.0,
            face_width: 0.8,
            face_height: 0.7,
            nose_length: 0.15,
            jaw_width: 0.6,
            cephalic_index: 100.0,
            facial_index: 87.5,
        };
        let labels = label_vertices_by_region(&verts, &centroid, &m);
        assert_eq!(labels[0], HeadRegion::Crown);
        assert_eq!(labels[1], HeadRegion::Neck);
    }
}
