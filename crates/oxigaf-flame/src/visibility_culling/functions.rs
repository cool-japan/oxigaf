//! Public visibility algorithms: per-face and per-vertex visibility,
//! multi-view aggregation, view coverage/selection, and formatting helpers.

use crate::mesh::Mesh;
use crate::normal_map::Camera;

use super::raster::{
    camera_direction, camera_world_position, compute_face_normal, compute_face_screen_area,
    is_front_facing, is_in_frustum, is_occluded, project_vertex, rasterize_depth_buffer,
};
use super::types::{
    FaceVisibility, MultiViewVisibility, VertexVisibility, VisibilityCullerConfig, VisibilityError,
    VisibilityStats,
};

// ---------------------------------------------------------------------------
// Core Public API
// ---------------------------------------------------------------------------

/// Compute per-face visibility from a single camera viewpoint.
///
/// For every face the function computes:
/// - Whether the face normal points toward the camera (front-facing test), and
///   — when `config.use_depth_test` is set — whether the face centroid survives
///   the depth-occlusion test.
/// - The projected screen-space area of the triangle.
///
/// # Errors
///
/// Returns [`VisibilityError::NoFaces`] for meshes with no triangles, or
/// [`VisibilityError::VertexIndexOutOfRange`] if a face references an invalid
/// vertex index.
pub fn compute_face_visibility(
    mesh: &Mesh,
    camera: &Camera,
    config: &VisibilityCullerConfig,
) -> Result<FaceVisibility, VisibilityError> {
    if mesh.faces.is_empty() {
        return Err(VisibilityError::NoFaces);
    }

    let n_faces = mesh.faces.len();
    let n_verts = mesh.vertices.len();
    let mut visible = vec![false; n_faces];
    let mut screen_area = vec![0.0_f32; n_faces];

    // Constant per camera — computed once instead of once per face.
    let cam_world = camera_world_position(camera);
    let depth_buffer = if config.use_depth_test {
        Some(rasterize_depth_buffer(mesh, camera))
    } else {
        None
    };

    for (face_idx, face) in mesh.faces.iter().enumerate() {
        let i0 = face[0] as usize;
        let i1 = face[1] as usize;
        let i2 = face[2] as usize;

        // Validate indices
        if i0 >= n_verts {
            return Err(VisibilityError::VertexIndexOutOfRange {
                idx: i0,
                n: n_verts,
            });
        }
        if i1 >= n_verts {
            return Err(VisibilityError::VertexIndexOutOfRange {
                idx: i1,
                n: n_verts,
            });
        }
        if i2 >= n_verts {
            return Err(VisibilityError::VertexIndexOutOfRange {
                idx: i2,
                n: n_verts,
            });
        }

        let v0 = mesh.vertices[i0];
        let v1 = mesh.vertices[i1];
        let v2 = mesh.vertices[i2];

        let v0a = [v0.x, v0.y, v0.z];
        let v1a = [v1.x, v1.y, v1.z];
        let v2a = [v2.x, v2.y, v2.z];

        // Face normal from cross product of edges (world space)
        let face_normal = compute_face_normal(v0a, v1a, v2a);

        // View direction from face centroid toward camera
        let centroid = [
            (v0.x + v1.x + v2.x) / 3.0,
            (v0.y + v1.y + v2.y) / 3.0,
            (v0.z + v1.z + v2.z) / 3.0,
        ];
        let view_dir = camera_direction(centroid, cam_world);

        visible[face_idx] = is_front_facing(face_normal, view_dir, config.backface_threshold)
            && !depth_buffer
                .as_deref()
                .is_some_and(|buf| is_occluded(centroid, camera, buf, config.depth_bias));

        // Projected screen-space area
        let s0 = project_vertex(v0a, camera);
        let s1 = project_vertex(v1a, camera);
        let s2 = project_vertex(v2a, camera);

        screen_area[face_idx] = match (s0, s1, s2) {
            (Some(a), Some(b), Some(c)) => compute_face_screen_area(a, b, c),
            _ => 0.0,
        };
    }

    Ok(FaceVisibility {
        visible,
        screen_area,
        n_faces,
    })
}

/// Compute per-vertex visibility from a single camera viewpoint.
///
/// A vertex is considered **visible** when it is inside the camera frustum, has
/// a front-facing per-vertex normal, and — when `config.use_depth_test` is set —
/// is not hidden behind other geometry in the rasterized depth buffer.
///
/// # Errors
///
/// Returns [`VisibilityError::EmptyMesh`] for meshes with no vertices, or
/// [`VisibilityError::NormalCountMismatch`] when the normal buffer length
/// differs from the vertex count.
pub fn compute_vertex_visibility(
    mesh: &Mesh,
    camera: &Camera,
    config: &VisibilityCullerConfig,
) -> Result<VertexVisibility, VisibilityError> {
    if mesh.vertices.is_empty() {
        return Err(VisibilityError::EmptyMesh);
    }
    if mesh.normals.len() != mesh.vertices.len() {
        return Err(VisibilityError::NormalCountMismatch {
            normals: mesh.normals.len(),
            vertices: mesh.vertices.len(),
        });
    }

    let n_vertices = mesh.vertices.len();
    let mut in_frustum_buf = vec![false; n_vertices];
    let mut front_facing_buf = vec![false; n_vertices];
    let mut visible_buf = vec![false; n_vertices];

    // Constant per camera — computed once instead of once per vertex.
    let cam_world = camera_world_position(camera);
    let depth_buffer = if config.use_depth_test {
        Some(rasterize_depth_buffer(mesh, camera))
    } else {
        None
    };

    for (i, (vertex, normal)) in mesh.vertices.iter().zip(mesh.normals.iter()).enumerate() {
        let va = [vertex.x, vertex.y, vertex.z];

        // Frustum test: project vertex and check screen bounds
        let in_f = project_vertex(va, camera)
            .is_some_and(|sp| is_in_frustum(sp, camera, config.frustum_margin));
        in_frustum_buf[i] = in_f;

        // Front-facing test using per-vertex normal
        let view_dir = camera_direction(va, cam_world);
        let na_arr = [normal.x, normal.y, normal.z];
        let ff = is_front_facing(na_arr, view_dir, config.backface_threshold);
        front_facing_buf[i] = ff;

        // Optional occlusion test against the rasterized depth buffer
        visible_buf[i] = in_f
            && ff
            && !depth_buffer
                .as_deref()
                .is_some_and(|buf| is_occluded(va, camera, buf, config.depth_bias));
    }

    Ok(VertexVisibility {
        visible: visible_buf,
        in_frustum: in_frustum_buf,
        front_facing: front_facing_buf,
        n_vertices,
    })
}

/// Compute summary statistics for a [`VertexVisibility`] result.
///
/// All fractions are in `[0.0, 1.0]`. For an empty mesh every fraction is
/// `0.0`.
#[must_use]
pub fn compute_visibility_stats(visibility: &VertexVisibility) -> VisibilityStats {
    let n = visibility.n_vertices;

    let n_visible = visibility.visible.iter().filter(|&&v| v).count();
    let n_front = visibility.front_facing.iter().filter(|&&v| v).count();
    let n_frustum = visibility.in_frustum.iter().filter(|&&v| v).count();

    let (vis_frac, ff_frac, inf_frac) = if n == 0 {
        (0.0, 0.0, 0.0)
    } else {
        let nf = n as f32;
        (
            n_visible as f32 / nf,
            n_front as f32 / nf,
            n_frustum as f32 / nf,
        )
    };

    VisibilityStats {
        n_vertices: n,
        n_visible_vertices: n_visible,
        n_front_facing: n_front,
        n_in_frustum: n_frustum,
        visible_fraction: vis_frac,
        front_facing_fraction: ff_frac,
        in_frustum_fraction: inf_frac,
    }
}

/// Aggregate vertex visibility across multiple camera viewpoints.
///
/// Runs [`compute_vertex_visibility`] for each camera and accumulates results.
///
/// # Errors
///
/// Returns [`VisibilityError::NoCameras`] when `cameras` is empty, or
/// propagates any error from [`compute_vertex_visibility`].
pub fn compute_multi_view_visibility(
    mesh: &Mesh,
    cameras: &[Camera],
    config: &VisibilityCullerConfig,
) -> Result<MultiViewVisibility, VisibilityError> {
    if cameras.is_empty() {
        return Err(VisibilityError::NoCameras);
    }

    let n_vertices = mesh.vertices.len();
    let n_cameras = cameras.len();

    let mut any_visible = vec![false; n_vertices];
    let mut view_count = vec![0usize; n_vertices];

    for camera in cameras {
        let vis = compute_vertex_visibility(mesh, camera, config)?;
        for (i, &v) in vis.visible.iter().enumerate() {
            if v {
                any_visible[i] = true;
                view_count[i] += 1;
            }
        }
    }

    let all_visible: Vec<bool> = view_count.iter().map(|&c| c == n_cameras).collect();

    Ok(MultiViewVisibility {
        any_visible,
        all_visible,
        view_count,
        n_vertices,
        n_cameras,
    })
}

/// Find vertices that are visible from some but not all cameras.
///
/// Returns the indices of vertices where `any_visible[i] == true` but
/// `all_visible[i] == false`.  These are "view-dependent" vertices that
/// require more care during training because they are only partially observed.
#[must_use]
pub fn find_view_dependent_vertices(multi_view: &MultiViewVisibility) -> Vec<usize> {
    multi_view
        .any_visible
        .iter()
        .zip(multi_view.all_visible.iter())
        .enumerate()
        .filter_map(|(i, (&any, &all))| if any && !all { Some(i) } else { None })
        .collect()
}

/// Compute per-camera coverage: fraction of mesh vertices visible from each camera.
///
/// Returns a `Vec<f32>` of length `cameras.len()`, where each entry is the
/// fraction of vertices visible from the corresponding camera.  The values are
/// independent per camera — nothing here optimizes the *set* of views; see
/// [`compute_greedy_view_selection`] for that.
///
/// # Errors
///
/// Returns [`VisibilityError::NoCameras`] when `cameras` is empty, or any
/// error from [`compute_vertex_visibility`].
pub fn compute_per_view_coverage(
    mesh: &Mesh,
    cameras: &[Camera],
    config: &VisibilityCullerConfig,
) -> Result<Vec<f32>, VisibilityError> {
    Ok(compute_per_view_visibility(mesh, cameras, config)?
        .iter()
        .map(coverage_fraction)
        .collect())
}

/// Per-camera [`VertexVisibility`], one entry per camera.
///
/// # Errors
///
/// Returns [`VisibilityError::NoCameras`] when `cameras` is empty, or any
/// error from [`compute_vertex_visibility`].
pub fn compute_per_view_visibility(
    mesh: &Mesh,
    cameras: &[Camera],
    config: &VisibilityCullerConfig,
) -> Result<Vec<VertexVisibility>, VisibilityError> {
    if cameras.is_empty() {
        return Err(VisibilityError::NoCameras);
    }
    cameras
        .iter()
        .map(|camera| compute_vertex_visibility(mesh, camera, config))
        .collect()
}

/// Fraction of vertices marked visible in a single-view result.
#[must_use]
pub(super) fn coverage_fraction(visibility: &VertexVisibility) -> f32 {
    if visibility.n_vertices == 0 {
        return 0.0;
    }
    let visible_count = visibility.visible.iter().filter(|&&v| v).count();
    visible_count as f32 / visibility.n_vertices as f32
}

/// Deprecated name for [`compute_per_view_coverage`], kept for API stability.
///
/// The function computes no optimum: it returns one coverage fraction per
/// camera.  Prefer [`compute_per_view_coverage`] or, for actual view selection,
/// [`compute_greedy_view_selection`].
///
/// # Errors
///
/// Same as [`compute_per_view_coverage`].
pub fn compute_optimal_view_coverage(
    mesh: &Mesh,
    cameras: &[Camera],
    config: &VisibilityCullerConfig,
) -> Result<Vec<f32>, VisibilityError> {
    compute_per_view_coverage(mesh, cameras, config)
}

/// Select the `k` cameras with the highest **individual** coverage fractions.
///
/// Returns camera indices sorted by coverage descending, ties broken by index;
/// if `k` exceeds the number of cameras the full list is returned.  This ignores
/// overlap: the top-`k` views of a head cluster around the frontal direction and
/// see almost the same vertices, so their union can cover far less of the mesh
/// than a greedy choice — use [`select_greedy_covering_views`] for the union.
#[must_use]
pub fn select_top_coverage_views(coverage: &[f32], k: usize) -> Vec<usize> {
    let mut indexed: Vec<(usize, f32)> =
        coverage.iter().enumerate().map(|(i, &c)| (i, c)).collect();

    // Sort descending by coverage; break ties by index (ascending) for determinism
    indexed.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });

    indexed.truncate(k);
    indexed.into_iter().map(|(i, _)| i).collect()
}

/// Deprecated name for [`select_top_coverage_views`], kept for API stability.
///
/// Despite the name it performs no maximal-coverage selection — it is a plain
/// top-`k` by individual coverage.  Prefer [`select_greedy_covering_views`].
#[must_use]
pub fn select_maximally_covering_views(coverage: &[f32], k: usize) -> Vec<usize> {
    select_top_coverage_views(coverage, k)
}

/// Greedy set-cover selection over per-camera visibility masks.
///
/// On each of at most `k` rounds the camera adding the most **newly** covered
/// vertices is appended and its mask OR-ed into the covered set; ties go to the
/// lowest camera index, so the result is deterministic.  Selection stops early —
/// returning fewer than `k` indices — once no remaining camera contributes a new
/// vertex, rather than padding the result with arbitrary cameras.
#[must_use]
pub fn select_greedy_covering_views(visibilities: &[VertexVisibility], k: usize) -> Vec<usize> {
    let n_vertices = visibilities
        .iter()
        .map(|v| v.visible.len())
        .max()
        .unwrap_or(0);

    let mut covered = vec![false; n_vertices];
    let mut used = vec![false; visibilities.len()];
    let mut selected: Vec<usize> = Vec::new();

    while selected.len() < k {
        let mut best: Option<(usize, usize)> = None; // (gain, camera index)
        for (cam_idx, vis) in visibilities.iter().enumerate() {
            if used[cam_idx] {
                continue;
            }
            let gain = vis
                .visible
                .iter()
                .enumerate()
                .filter(|&(v_idx, &v)| v && !covered[v_idx])
                .count();
            if best.is_none_or(|(best_gain, _)| gain > best_gain) {
                best = Some((gain, cam_idx));
            }
        }

        let Some((gain, cam_idx)) = best else {
            break; // every camera already used
        };
        if gain == 0 {
            break; // no marginal coverage left
        }

        for (v_idx, &v) in visibilities[cam_idx].visible.iter().enumerate() {
            if v {
                covered[v_idx] = true;
            }
        }
        used[cam_idx] = true;
        selected.push(cam_idx);
    }

    selected
}

/// Compute per-camera visibility and greedily select `k` views maximizing the
/// union of covered vertices ([`compute_per_view_visibility`] +
/// [`select_greedy_covering_views`]).
///
/// # Errors
///
/// Returns [`VisibilityError::NoCameras`] when `cameras` is empty, or any
/// error from [`compute_vertex_visibility`].
pub fn compute_greedy_view_selection(
    mesh: &Mesh,
    cameras: &[Camera],
    config: &VisibilityCullerConfig,
    k: usize,
) -> Result<Vec<usize>, VisibilityError> {
    let visibilities = compute_per_view_visibility(mesh, cameras, config)?;
    Ok(select_greedy_covering_views(&visibilities, k))
}

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------

/// Format a [`VisibilityStats`] report as a human-readable one-liner.
///
/// Example output:
/// `"Visibility: 3421/5023 vertices visible (68.1%), 4200 front-facing (83.6%), 3800 in frustum (75.7%)"`
#[must_use]
pub fn format_visibility_stats(stats: &VisibilityStats) -> String {
    format!(
        "Visibility: {}/{} vertices visible ({:.1}%), {} front-facing ({:.1}%), {} in frustum ({:.1}%)",
        stats.n_visible_vertices,
        stats.n_vertices,
        stats.visible_fraction * 100.0,
        stats.n_front_facing,
        stats.front_facing_fraction * 100.0,
        stats.n_in_frustum,
        stats.in_frustum_fraction * 100.0,
    )
}

/// Format a [`MultiViewVisibility`] summary as a human-readable one-liner.
///
/// Example output:
/// `"MultiView[4 cams]: any_visible=4890 (97.4%), all_visible=1234 (24.6%), view-dependent=3656 (72.8%)"`
#[must_use]
pub fn format_multi_view_stats(mv: &MultiViewVisibility) -> String {
    let n = mv.n_vertices as f32;
    let any_count = mv.any_visible.iter().filter(|&&v| v).count();
    let all_count = mv.all_visible.iter().filter(|&&v| v).count();
    let dep_count = any_count.saturating_sub(all_count);

    let (any_pct, all_pct, dep_pct) = if mv.n_vertices == 0 {
        (0.0_f32, 0.0_f32, 0.0_f32)
    } else {
        (
            any_count as f32 / n * 100.0,
            all_count as f32 / n * 100.0,
            dep_count as f32 / n * 100.0,
        )
    };

    format!(
        "MultiView[{} cams]: any_visible={} ({:.1}%), all_visible={} ({:.1}%), view-dependent={} ({:.1}%)",
        mv.n_cameras, any_count, any_pct, all_count, all_pct, dep_count, dep_pct,
    )
}
