//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use nalgebra as na;

use super::type_aliases::ObbResult;
use super::types::{BoundingSphere, GaussianBBox, GeometryError, GeometryStats, RigidTransform};

/// Dot product of two 3-vectors.
#[cfg(test)]
#[inline]
pub(super) fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
/// Euclidean length of a 3-vector.
#[cfg(test)]
#[inline]
fn len3(v: [f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}
/// L2-normalise a 3-vector. Returns the zero vector if the input has (near-)zero length.
#[cfg(test)]
#[inline]
pub(super) fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let l = len3(v);
    if l < f32::EPSILON {
        [0.0, 0.0, 0.0]
    } else {
        [v[0] / l, v[1] / l, v[2] / l]
    }
}
/// Cross product of two 3-vectors: `a × b`.
#[inline]
pub(super) fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
/// Rotate vector `v` by unit quaternion `q` (w-last convention: `[qx, qy, qz, qw]`).
///
/// Uses the formula:  `v' = v + 2*qw*(qxyz × v) + 2*(qxyz × (qxyz × v))`
#[inline]
pub(super) fn quat_rotate(q: [f32; 4], v: [f32; 3]) -> [f32; 3] {
    let qv = [q[0], q[1], q[2]];
    let qw = q[3];
    let c1 = cross3(qv, v);
    let c2 = cross3(qv, c1);
    [
        v[0] + 2.0 * qw * c1[0] + 2.0 * c2[0],
        v[1] + 2.0 * qw * c1[1] + 2.0 * c2[1],
        v[2] + 2.0 * qw * c1[2] + 2.0 * c2[2],
    ]
}
/// Inverse of a unit quaternion (w-last): `[-qx, -qy, -qz, qw]`.
#[inline]
pub(super) fn quat_inverse(q: [f32; 4]) -> [f32; 4] {
    [-q[0], -q[1], -q[2], q[3]]
}
/// Hamilton product `a * b` for quaternions in w-last `[qx, qy, qz, qw]` form.
#[inline]
pub(super) fn quat_mul(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    let (ax, ay, az, aw) = (a[0], a[1], a[2], a[3]);
    let (bx, by, bz, bw) = (b[0], b[1], b[2], b[3]);
    [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]
}
fn validate_positions(positions: &[f32]) -> Result<usize, GeometryError> {
    if !positions.len().is_multiple_of(3) {
        return Err(GeometryError::InvalidPositionLength {
            len: positions.len(),
        });
    }
    let n = positions.len() / 3;
    if n == 0 {
        return Err(GeometryError::EmptyCloud);
    }
    Ok(n)
}
fn validate_rotations(rotations: &[f32]) -> Result<usize, GeometryError> {
    if !rotations.len().is_multiple_of(4) {
        return Err(GeometryError::InvalidRotationLength {
            len: rotations.len(),
        });
    }
    Ok(rotations.len() / 4)
}
fn validate_scales(scales: &[f32]) -> Result<usize, GeometryError> {
    if !scales.len().is_multiple_of(3) {
        return Err(GeometryError::InvalidScaleLength { len: scales.len() });
    }
    Ok(scales.len() / 3)
}
/// Compute the axis-aligned bounding box of flat position data.
pub fn compute_gaussian_bbox(positions: &[f32]) -> Result<GaussianBBox, GeometryError> {
    let n = validate_positions(positions)?;
    let mut min = [f32::MAX; 3];
    let mut max = [f32::MIN; 3];
    for i in 0..n {
        let base = i * 3;
        for axis in 0..3 {
            let v = positions[base + axis];
            if v < min[axis] {
                min[axis] = v;
            }
            if v > max[axis] {
                max[axis] = v;
            }
        }
    }
    Ok(GaussianBBox { min, max })
}
/// Compute an approximate bounding sphere using Ritter's algorithm.
///
/// Steps:
/// 1. Find the pair of points with the greatest separation along any single axis.
/// 2. Use that pair as the initial sphere diameter.
/// 3. Expand the sphere to enclose any remaining points.
pub fn compute_bounding_sphere(positions: &[f32]) -> Result<BoundingSphere, GeometryError> {
    let n = validate_positions(positions)?;
    let mut min_idx = [0usize; 3];
    let mut max_idx = [0usize; 3];
    let mut min_val = [f32::MAX; 3];
    let mut max_val = [f32::MIN; 3];
    for i in 0..n {
        let base = i * 3;
        for axis in 0..3 {
            let v = positions[base + axis];
            if v < min_val[axis] {
                min_val[axis] = v;
                min_idx[axis] = i;
            }
            if v > max_val[axis] {
                max_val[axis] = v;
                max_idx[axis] = i;
            }
        }
    }
    let mut best_axis = 0;
    let mut best_span2 = -1.0f32;
    for axis in 0..3 {
        let p_min = &positions[min_idx[axis] * 3..min_idx[axis] * 3 + 3];
        let p_max = &positions[max_idx[axis] * 3..max_idx[axis] * 3 + 3];
        let span2 = (p_max[0] - p_min[0]).powi(2)
            + (p_max[1] - p_min[1]).powi(2)
            + (p_max[2] - p_min[2]).powi(2);
        if span2 > best_span2 {
            best_span2 = span2;
            best_axis = axis;
        }
    }
    let p_a = &positions[min_idx[best_axis] * 3..min_idx[best_axis] * 3 + 3];
    let p_b = &positions[max_idx[best_axis] * 3..max_idx[best_axis] * 3 + 3];
    let mut center = [
        (p_a[0] + p_b[0]) * 0.5,
        (p_a[1] + p_b[1]) * 0.5,
        (p_a[2] + p_b[2]) * 0.5,
    ];
    let mut radius = best_span2.sqrt() * 0.5;
    for i in 0..n {
        let base = i * 3;
        let dx = positions[base] - center[0];
        let dy = positions[base + 1] - center[1];
        let dz = positions[base + 2] - center[2];
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();
        if dist > radius {
            let excess = dist - radius;
            let ratio = excess / (2.0 * dist);
            center[0] += dx * ratio;
            center[1] += dy * ratio;
            center[2] += dz * ratio;
            radius = (radius + dist) * 0.5;
        }
    }
    Ok(BoundingSphere { center, radius })
}
/// Compute the centroid (mean position) of the cloud.
pub fn compute_centroid(positions: &[f32]) -> Result<[f32; 3], GeometryError> {
    let n = validate_positions(positions)?;
    let mut sum = [0.0f32; 3];
    for i in 0..n {
        let base = i * 3;
        sum[0] += positions[base];
        sum[1] += positions[base + 1];
        sum[2] += positions[base + 2];
    }
    let nf = n as f32;
    Ok([sum[0] / nf, sum[1] / nf, sum[2] / nf])
}
/// Compute full geometry statistics for a Gaussian cloud.
///
/// `positions` is N×3 flat; `scales` is N×3 log-scale flat.
pub fn compute_geometry_stats(
    positions: &[f32],
    scales: &[f32],
) -> Result<GeometryStats, GeometryError> {
    let n_pos = validate_positions(positions)?;
    let n_scales = validate_scales(scales)?;
    if n_pos != n_scales {
        return Err(GeometryError::CountMismatch {
            n_pos,
            n_other: n_scales,
            name: "scales".to_string(),
        });
    }
    let bbox = compute_gaussian_bbox(positions)?;
    let bounding_sphere = compute_bounding_sphere(positions)?;
    let centroid = compute_centroid(positions)?;
    let mut var_sum = 0.0f32;
    let mut count = 0usize;
    for i in 0..n_pos {
        let base = i * 3;
        for axis in 0..3 {
            let diff = positions[base + axis] - centroid[axis];
            var_sum += diff * diff;
            count += 1;
        }
    }
    let std_position = if count > 1 {
        (var_sum / count as f32).sqrt()
    } else {
        0.0
    };
    let mut scale_sum = 0.0f32;
    let mut min_scale = f32::MAX;
    let mut max_scale = f32::MIN;
    let total_axes = scales.len();
    for &ls in scales {
        let s = ls.exp();
        scale_sum += s;
        if s < min_scale {
            min_scale = s;
        }
        if s > max_scale {
            max_scale = s;
        }
    }
    let mean_scale = if total_axes > 0 {
        scale_sum / total_axes as f32
    } else {
        0.0
    };
    Ok(GeometryStats {
        n_gaussians: n_pos,
        bbox,
        bounding_sphere,
        centroid,
        mean_scale,
        max_scale: if max_scale == f32::MIN {
            0.0
        } else {
            max_scale
        },
        min_scale: if min_scale == f32::MAX {
            0.0
        } else {
            min_scale
        },
        std_position,
    })
}
/// Apply a rigid transform to all positions in-place.
pub fn transform_positions(
    positions: &mut [f32],
    transform: &RigidTransform,
) -> Result<(), GeometryError> {
    let n = validate_positions(positions)?;
    for i in 0..n {
        let base = i * 3;
        let p = [positions[base], positions[base + 1], positions[base + 2]];
        let tp = transform.apply_to_point(p);
        positions[base] = tp[0];
        positions[base + 1] = tp[1];
        positions[base + 2] = tp[2];
    }
    Ok(())
}
/// Apply a rigid transform to all quaternion rotations in-place.
///
/// The new rotation for each Gaussian is `transform.rotation * original_rotation`.
pub fn transform_rotations(
    rotations: &mut [f32],
    transform: &RigidTransform,
) -> Result<(), GeometryError> {
    let n = validate_rotations(rotations)?;
    for i in 0..n {
        let base = i * 4;
        let orig = [
            rotations[base],
            rotations[base + 1],
            rotations[base + 2],
            rotations[base + 3],
        ];
        let combined = quat_mul(transform.rotation, orig);
        rotations[base] = combined[0];
        rotations[base + 1] = combined[1];
        rotations[base + 2] = combined[2];
        rotations[base + 3] = combined[3];
    }
    Ok(())
}
/// Subtract the centroid from all positions, centring the cloud at the origin.
///
/// Returns the centroid that was subtracted.
pub fn center_at_origin(positions: &mut [f32]) -> Result<[f32; 3], GeometryError> {
    let centroid = compute_centroid(positions)?;
    let n = positions.len() / 3;
    for i in 0..n {
        let base = i * 3;
        positions[base] -= centroid[0];
        positions[base + 1] -= centroid[1];
        positions[base + 2] -= centroid[2];
    }
    Ok(centroid)
}
/// Centre the cloud at the origin (using bbox centre, not centroid) and scale positions
/// to fit within `[-0.5, 0.5]³`.
///
/// Returns the scale factor applied (positions were divided by this value).
pub fn normalize_to_unit_cube(positions: &mut [f32]) -> Result<f32, GeometryError> {
    let bbox = compute_gaussian_bbox(positions)?;
    let bbox_center = bbox.center();
    let n = validate_positions(positions)?;
    for i in 0..n {
        let base = i * 3;
        positions[base] -= bbox_center[0];
        positions[base + 1] -= bbox_center[1];
        positions[base + 2] -= bbox_center[2];
    }
    let s = bbox.size();
    let max_extent = s[0].max(s[1]).max(s[2]);
    if max_extent < f32::EPSILON {
        return Ok(1.0);
    }
    for i in 0..n {
        let base = i * 3;
        positions[base] /= max_extent;
        positions[base + 1] /= max_extent;
        positions[base + 2] /= max_extent;
    }
    Ok(max_extent)
}
/// Return the indices of Gaussians whose position lies within the given bounding box.
pub fn filter_by_bbox(positions: &[f32], bbox: &GaussianBBox) -> Result<Vec<usize>, GeometryError> {
    let n = validate_positions(positions)?;
    let mut kept = Vec::new();
    for i in 0..n {
        let base = i * 3;
        let p = [positions[base], positions[base + 1], positions[base + 2]];
        if bbox.contains(p) {
            kept.push(i);
        }
    }
    Ok(kept)
}
/// Return the indices of Gaussians whose position lies within the given sphere.
pub fn filter_by_sphere(
    positions: &[f32],
    center: [f32; 3],
    radius: f32,
) -> Result<Vec<usize>, GeometryError> {
    let n = validate_positions(positions)?;
    let r2 = radius * radius;
    let mut kept = Vec::new();
    for i in 0..n {
        let base = i * 3;
        let dx = positions[base] - center[0];
        let dy = positions[base + 1] - center[1];
        let dz = positions[base + 2] - center[2];
        if dx * dx + dy * dy + dz * dz <= r2 {
            kept.push(i);
        }
    }
    Ok(kept)
}
/// Euclidean distance between the centroids of two Gaussian clouds.
pub fn cloud_distance(positions_a: &[f32], positions_b: &[f32]) -> Result<f32, GeometryError> {
    let ca = compute_centroid(positions_a)?;
    let cb = compute_centroid(positions_b)?;
    let dx = ca[0] - cb[0];
    let dy = ca[1] - cb[1];
    let dz = ca[2] - cb[2];
    Ok((dx * dx + dy * dy + dz * dz).sqrt())
}
/// Compute a PCA-based oriented bounding box (OBB).
///
/// Returns `(center, half_extents, rotation_quaternion [x,y,z,w])`.
///
/// ## Algorithm
///
/// 1. Compute the centroid and centre all points.
/// 2. Build the 3×3 covariance matrix `C = (1/n) Σ p·pᵀ`.
/// 3. Eigendecompose `C` with [`nalgebra::SymmetricEigen`].  If decomposition
///    fails (e.g. degenerate point set) fall back to the AABB result.
/// 4. The eigenvectors form an orthonormal rotation matrix; the principal axes
///    are sorted by descending eigenvalue so the longest extent is first.
/// 5. Project all centred points onto each principal axis to find the
///    per-axis half-extent.
/// 6. Return the centroid, the three half-extents (largest first), and the
///    rotation as a unit quaternion `[x, y, z, w]`.
pub fn compute_obb(positions: &[f32]) -> Result<ObbResult, GeometryError> {
    let centroid = compute_centroid(positions)?;
    let n = positions.len() / 3;
    let mut centered: Vec<[f32; 3]> = Vec::with_capacity(n);
    for i in 0..n {
        centered.push([
            positions[i * 3] - centroid[0],
            positions[i * 3 + 1] - centroid[1],
            positions[i * 3 + 2] - centroid[2],
        ]);
    }
    let mut cov = na::Matrix3::<f32>::zeros();
    for p in &centered {
        let col = na::Vector3::new(p[0], p[1], p[2]);
        cov += col * col.transpose();
    }
    cov /= n as f32;
    let eigen_result = na::SymmetricEigen::try_new(cov, 1e-7, 100);
    let (half_extents, quaternion) = match eigen_result {
        None => {
            let bbox = compute_gaussian_bbox(positions)?;
            let s = bbox.size();
            let mut he = [s[0] * 0.5, s[1] * 0.5, s[2] * 0.5];
            he.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            return Ok((centroid, he, [0.0f32, 0.0, 0.0, 1.0]));
        }
        Some(eig) => {
            let eigenvalues = eig.eigenvalues;
            let eigenvectors = eig.eigenvectors;
            let mut order = [0usize, 1, 2];
            order.sort_by(|&a, &b| {
                eigenvalues[b]
                    .partial_cmp(&eigenvalues[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let axes: Vec<na::Vector3<f32>> = order
                .iter()
                .map(|&i| eigenvectors.column(i).into_owned())
                .collect();
            let mut half_extents = [0.0f32; 3];
            for p in &centered {
                let pv = na::Vector3::new(p[0], p[1], p[2]);
                for (k, axis) in axes.iter().enumerate() {
                    let proj = axis.dot(&pv).abs();
                    if proj > half_extents[k] {
                        half_extents[k] = proj;
                    }
                }
            }
            let mut rot = na::Matrix3::<f32>::zeros();
            for (k, axis) in axes.iter().enumerate() {
                rot.set_column(k, axis);
            }
            if rot.determinant() < 0.0 {
                let col = -rot.column(2).into_owned();
                rot.set_column(2, &col);
            }
            let quaternion = match na::Rotation3::from_matrix_unchecked(rot).axis_angle() {
                Some((axis, angle)) => {
                    let s = (angle * 0.5).sin();
                    let c = (angle * 0.5).cos();
                    [axis.x * s, axis.y * s, axis.z * s, c]
                }
                None => [0.0f32, 0.0, 0.0, 1.0],
            };
            (half_extents, quaternion)
        }
    };
    Ok((centroid, half_extents, quaternion))
}
/// Mean of `exp(log_scale)` across all entries in the scales array.
pub fn mean_gaussian_scale(scales: &[f32]) -> Result<f32, GeometryError> {
    if !scales.len().is_multiple_of(3) {
        return Err(GeometryError::InvalidScaleLength { len: scales.len() });
    }
    if scales.is_empty() {
        return Err(GeometryError::EmptyCloud);
    }
    let sum: f32 = scales.iter().map(|&ls| ls.exp()).sum();
    Ok(sum / scales.len() as f32)
}
/// Adjust all log-scale values so the mean actual scale equals `target_mean_scale`.
///
/// Adds `ln(target / current_mean)` to every entry.  Returns the additive delta applied.
pub fn rescale_gaussians(scales: &mut [f32], target_mean_scale: f32) -> Result<f32, GeometryError> {
    if target_mean_scale <= 0.0 {
        return Err(GeometryError::InvalidTransform {
            reason: format!("target_mean_scale must be positive, got {target_mean_scale}"),
        });
    }
    let current_mean = mean_gaussian_scale(scales)?;
    if current_mean < f32::EPSILON {
        return Err(GeometryError::InvalidTransform {
            reason: "current mean scale is (near-)zero; cannot rescale".to_string(),
        });
    }
    let delta = (target_mean_scale / current_mean).ln();
    for ls in scales.iter_mut() {
        *ls += delta;
    }
    Ok(delta)
}
/// For each Gaussian, compute the distance to its k-th nearest neighbour (1-indexed).
///
/// Uses brute-force O(n²) computation — acceptable for CLI tooling.
///
/// # Panics (never)
/// All errors are returned as `GeometryError`.
pub fn nearest_neighbor_distances(positions: &[f32], k: usize) -> Result<Vec<f32>, GeometryError> {
    let n = validate_positions(positions)?;
    if k == 0 {
        return Err(GeometryError::InvalidTransform {
            reason: "k must be >= 1".to_string(),
        });
    }
    if k >= n {
        return Err(GeometryError::InvalidTransform {
            reason: format!("k ({k}) must be less than number of points ({n})"),
        });
    }
    let mut result = Vec::with_capacity(n);
    for i in 0..n {
        let bi = i * 3;
        let pi = [positions[bi], positions[bi + 1], positions[bi + 2]];
        let mut dists: Vec<f32> = (0..n)
            .filter(|&j| j != i)
            .map(|j| {
                let bj = j * 3;
                let dx = positions[bj] - pi[0];
                let dy = positions[bj + 1] - pi[1];
                let dz = positions[bj + 2] - pi[2];
                dx * dx + dy * dy + dz * dz
            })
            .collect();
        dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        result.push(dists[k - 1].sqrt());
    }
    Ok(result)
}
/// Compute what fraction of cells in a voxel grid covering `reference_bbox` are occupied by
/// at least one Gaussian position.
///
/// `grid_resolution` is the number of cells along each axis (total cells = `grid_resolution³`).
/// Returns a value in `[0.0, 1.0]`.
pub fn spatial_coverage(
    positions: &[f32],
    reference_bbox: &GaussianBBox,
    grid_resolution: u32,
) -> Result<f32, GeometryError> {
    let n = validate_positions(positions)?;
    if grid_resolution == 0 {
        return Err(GeometryError::InvalidTransform {
            reason: "grid_resolution must be >= 1".to_string(),
        });
    }
    let res = grid_resolution as usize;
    let total_cells = res * res * res;
    let mut grid = vec![false; total_cells];
    let size = reference_bbox.size();
    let inv_x = if size[0] > f32::EPSILON {
        res as f32 / size[0]
    } else {
        0.0
    };
    let inv_y = if size[1] > f32::EPSILON {
        res as f32 / size[1]
    } else {
        0.0
    };
    let inv_z = if size[2] > f32::EPSILON {
        res as f32 / size[2]
    } else {
        0.0
    };
    for i in 0..n {
        let base = i * 3;
        let x = positions[base] - reference_bbox.min[0];
        let y = positions[base + 1] - reference_bbox.min[1];
        let z = positions[base + 2] - reference_bbox.min[2];
        let gx = ((x * inv_x) as usize).min(res - 1);
        let gy = ((y * inv_y) as usize).min(res - 1);
        let gz = ((z * inv_z) as usize).min(res - 1);
        let idx = gx + gy * res + gz * res * res;
        grid[idx] = true;
    }
    let occupied = grid.iter().filter(|&&v| v).count();
    Ok(occupied as f32 / total_cells as f32)
}
