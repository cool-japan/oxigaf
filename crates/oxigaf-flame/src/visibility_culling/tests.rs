// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

use super::raster::*;
use super::*;
use crate::mesh::Mesh;
use crate::normal_map::Camera;
use nalgebra as na;

// -----------------------------------------------------------------------
// Test helpers
// -----------------------------------------------------------------------

/// Single-triangle mesh in the XY plane with normals pointing +Z.
///
/// Vertices are scaled to ±0.3 so they project well within the 256×256
/// image when using `front_camera()` (focal=256, cx=cy=128, depth≈2).
/// At depth 2, a vertex displaced by 0.3 in X or Y maps to screen offset
/// 256 × 0.3 / 2 = 38.4 px from the principal point — well inside the image.
fn simple_mesh() -> Mesh {
    let vertices = vec![
        na::Point3::new(0.0_f32, 0.0, 0.0),
        na::Point3::new(0.3_f32, 0.0, 0.0),
        na::Point3::new(0.0_f32, 0.3, 0.0),
    ];
    let normals = vec![
        na::Vector3::new(0.0_f32, 0.0, 1.0),
        na::Vector3::new(0.0_f32, 0.0, 1.0),
        na::Vector3::new(0.0_f32, 0.0, 1.0),
    ];
    let faces = vec![[0u32, 1, 2]];
    Mesh {
        vertices,
        normals,
        faces,
        uv_coords: Vec::new(),
    }
}

/// Camera located on the +Z side, looking toward −Z (i.e. looking at the
/// mesh from the front, so vertex normals pointing +Z are front-facing).
///
/// R is a 180° rotation around Y: maps +Z world → −Z camera.
/// The camera world position is −R^T * t = −R * t (R symmetric here) = [0,0,2].
fn front_camera() -> Camera {
    // 180° around Y: R = [[-1,0,0],[0,1,0],[0,0,-1]]
    let rotation = na::Matrix3::new(-1.0_f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0);
    Camera {
        rotation,
        translation: na::Vector3::new(0.0_f32, 0.0, 2.0),
        focal_x: 256.0,
        focal_y: 256.0,
        cx: 128.0,
        cy: 128.0,
        width: 256,
        height: 256,
        near: 0.01,
        far: 10.0,
    }
}

/// Camera located on the −Z side, looking toward +Z (back-facing to +Z normals).
fn back_camera() -> Camera {
    Camera {
        rotation: na::Matrix3::identity(),
        translation: na::Vector3::new(0.0_f32, 0.0, 1.0),
        focal_x: 256.0,
        focal_y: 256.0,
        cx: 128.0,
        cy: 128.0,
        width: 256,
        height: 256,
        near: 0.01,
        far: 10.0,
    }
}

// -----------------------------------------------------------------------
// compute_face_normal
// -----------------------------------------------------------------------

#[test]
fn test_face_normal_orientation_and_winding() {
    let origin = [0.0_f32, 0.0, 0.0];
    // Triangle in the XY plane → +Z; reversed winding → −Z.
    let n = compute_face_normal(origin, [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
    assert!((n[2] - 1.0).abs() < 1e-5, "z should be 1, got {}", n[2]);
    assert!(n[0].abs() < 1e-5 && n[1].abs() < 1e-5, "{n:?}");
    let r = compute_face_normal(origin, [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]);
    assert!((r[2] + 1.0).abs() < 1e-5, "z should be -1, got {}", r[2]);
    // Triangle in the XZ plane → e1×e2 = [1,0,0]×[0,0,1] = [0,-1,0].
    let y = compute_face_normal(origin, [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
    assert!((y[1] + 1.0).abs() < 1e-5, "y should be -1, got {}", y[1]);
}

#[test]
fn test_face_normal_degenerate_and_unit_length() {
    let origin = [0.0_f32, 0.0, 0.0];
    assert_eq!(compute_face_normal(origin, origin, origin), [0.0, 0.0, 0.0]);
    let n = compute_face_normal(origin, [2.0, 0.0, 0.0], [0.0, 3.0, 0.0]);
    let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    assert!((len - 1.0).abs() < 1e-5, "not unit length, len={len}");
}

// -----------------------------------------------------------------------
// is_front_facing
// -----------------------------------------------------------------------

#[test]
fn test_is_front_facing_aligned_normals_true() {
    // Normal and view_dir point same direction
    assert!(is_front_facing([0.0, 0.0, 1.0], [0.0, 0.0, 1.0], 0.0));
}

#[test]
fn test_is_front_facing_opposite_normals_false() {
    assert!(!is_front_facing([0.0, 0.0, 1.0], [0.0, 0.0, -1.0], 0.0));
}

#[test]
fn test_is_front_facing_perpendicular_at_boundary() {
    // Dot = 0, threshold = 0 → NOT strictly greater
    assert!(!is_front_facing([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 0.0));
}

#[test]
fn test_is_front_facing_negative_threshold_accepts_some_backfaces() {
    // dot([1,0,0], [0.5, 0, 0]) would be 0.5 with normalized [0.5,0,0]=[1,0,0].
    // Use a glancing back-face: normal=[0,0,1], view=[0.5, 0, -0.866] (30° behind)
    // dot = 0*0.5 + 0*0 + 1*(-0.866) = -0.866
    // threshold = -0.9 → -0.866 > -0.9 → true
    let n = [0.0_f32, 0.0, 1.0];
    let v = [0.5_f32, 0.0, -0.866]; // normalized by hand (0.5^2+0.866^2≈1)
    assert!(
        is_front_facing(n, v, -0.9),
        "slightly behind should pass threshold -0.9"
    );
    // But same normal with threshold 0.0 → false (still back-facing)
    assert!(!is_front_facing(n, v, 0.0));
}

#[test]
fn test_is_front_facing_high_threshold_rejects_glancing() {
    // dot = 0.1, threshold = 0.5 → false (glancing angle rejected)
    let n = [0.0_f32, 0.0, 1.0];
    let v = [0.995_f32, 0.0, 0.1]; // nearly perpendicular
    let len = (v[0] * v[0] + v[2] * v[2]).sqrt();
    let v_norm = [v[0] / len, 0.0, v[2] / len];
    assert!(!is_front_facing(n, v_norm, 0.5));
}

// -----------------------------------------------------------------------
// is_in_frustum
// -----------------------------------------------------------------------

#[test]
fn test_is_in_frustum_image_bounds() {
    let camera = front_camera(); // 256×256
    assert!(is_in_frustum([128.0, 128.0], &camera, 0.0), "centre");
    assert!(is_in_frustum([0.0, 0.0], &camera, 0.0), "corner pixel");
    assert!(!is_in_frustum([257.0, 128.0], &camera, 0.0), "past right");
    assert!(!is_in_frustum([-1.0, 128.0], &camera, 0.0), "past left");
}

#[test]
fn test_is_in_frustum_margin_extends_bounds() {
    let camera = front_camera();
    // 1 px outside on either side, but within a 2 px margin.
    assert!(is_in_frustum([257.0, 128.0], &camera, 2.0));
    assert!(is_in_frustum([-1.0, 128.0], &camera, 2.0));
}

// -----------------------------------------------------------------------
// compute_face_screen_area
// -----------------------------------------------------------------------

#[test]
fn test_face_screen_area_right_triangle_half() {
    // Right triangle with legs 1×1 → area = 0.5
    let s0 = [0.0_f32, 0.0];
    let s1 = [1.0, 0.0];
    let s2 = [0.0, 1.0];
    let area = compute_face_screen_area(s0, s1, s2);
    assert!((area - 0.5).abs() < 1e-5, "area should be 0.5, got {area}");
}

#[test]
fn test_face_screen_area_degenerate_zero() {
    let s0 = [0.0_f32, 0.0];
    let s1 = [0.0, 0.0];
    let s2 = [0.0, 0.0];
    let area = compute_face_screen_area(s0, s1, s2);
    assert!(area.abs() < 1e-5, "degenerate triangle should have area ~0");
}

#[test]
fn test_face_screen_area_unit_square_triangle() {
    // Triangle occupying half of a 2×2 square → area = 2.0
    let s0 = [0.0_f32, 0.0];
    let s1 = [2.0, 0.0];
    let s2 = [0.0, 2.0];
    let area = compute_face_screen_area(s0, s1, s2);
    assert!((area - 2.0).abs() < 1e-5, "area should be 2.0, got {area}");
}

#[test]
fn test_face_screen_area_absolute_value() {
    // Reversed winding should still give positive area
    let s0 = [0.0_f32, 0.0];
    let s1 = [0.0, 1.0]; // reversed
    let s2 = [1.0, 0.0];
    let area = compute_face_screen_area(s0, s1, s2);
    assert!(area > 0.0, "area must be positive regardless of winding");
}

// -----------------------------------------------------------------------
// compute_face_visibility
// -----------------------------------------------------------------------

#[test]
fn test_face_visibility_empty_faces_error() {
    let mut mesh = simple_mesh();
    mesh.faces.clear();
    let config = VisibilityCullerConfig::default();
    let camera = front_camera();
    let result = compute_face_visibility(&mesh, &camera, &config);
    assert!(matches!(result, Err(VisibilityError::NoFaces)));
}

#[test]
fn test_face_visibility_front_facing_visible() {
    let mesh = simple_mesh();
    let camera = front_camera();
    let config = VisibilityCullerConfig::default();
    let fv = compute_face_visibility(&mesh, &camera, &config).expect("should succeed");
    assert_eq!(fv.n_faces, 1);
    assert!(fv.visible[0], "front-facing face should be visible");
}

#[test]
fn test_face_visibility_back_facing_not_visible() {
    let mesh = simple_mesh();
    let camera = back_camera(); // looking from -Z, face points +Z → back-facing
    let config = VisibilityCullerConfig::default();
    let fv = compute_face_visibility(&mesh, &camera, &config).expect("should succeed");
    assert!(!fv.visible[0], "back-facing face should not be visible");
}

#[test]
fn test_face_visibility_screen_area_positive_for_visible() {
    let mesh = simple_mesh();
    let camera = front_camera();
    let config = VisibilityCullerConfig::default();
    let fv = compute_face_visibility(&mesh, &camera, &config).expect("should succeed");
    assert!(
        fv.screen_area[0] > 0.0,
        "visible face must have positive screen area"
    );
}

#[test]
fn test_face_visibility_invalid_index_error() {
    let mut mesh = simple_mesh();
    mesh.faces = vec![[0, 1, 99]]; // index 99 out of range for 3 vertices
    let camera = front_camera();
    let config = VisibilityCullerConfig::default();
    let result = compute_face_visibility(&mesh, &camera, &config);
    assert!(matches!(
        result,
        Err(VisibilityError::VertexIndexOutOfRange { .. })
    ));
}

// -----------------------------------------------------------------------
// compute_vertex_visibility
// -----------------------------------------------------------------------

#[test]
fn test_vertex_visibility_empty_mesh_error() {
    let mesh = Mesh {
        vertices: Vec::new(),
        normals: Vec::new(),
        faces: Vec::new(),
        uv_coords: Vec::new(),
    };
    let camera = front_camera();
    let config = VisibilityCullerConfig::default();
    let result = compute_vertex_visibility(&mesh, &camera, &config);
    assert!(matches!(result, Err(VisibilityError::EmptyMesh)));
}

#[test]
fn test_vertex_visibility_normal_mismatch_error() {
    let mut mesh = simple_mesh();
    mesh.normals.pop(); // remove one normal → mismatch
    let camera = front_camera();
    let config = VisibilityCullerConfig::default();
    let result = compute_vertex_visibility(&mesh, &camera, &config);
    assert!(matches!(
        result,
        Err(VisibilityError::NormalCountMismatch { .. })
    ));
}

#[test]
fn test_vertex_visibility_front_facing_camera_all_visible() {
    let mesh = simple_mesh();
    let camera = front_camera();
    let config = VisibilityCullerConfig::default();
    let vv = compute_vertex_visibility(&mesh, &camera, &config).expect("should succeed");
    assert_eq!(vv.n_vertices, 3);
    // All normals point +Z, camera is on +Z side → all front-facing and in frustum
    for i in 0..3 {
        assert!(vv.front_facing[i], "vertex {i} should be front-facing");
        assert!(vv.in_frustum[i], "vertex {i} should be in-frustum");
        assert!(vv.visible[i], "vertex {i} should be visible");
    }
}

#[test]
fn test_vertex_visibility_back_camera_not_front_facing() {
    let mesh = simple_mesh();
    let camera = back_camera(); // normals point +Z, camera on -Z → back-facing
    let config = VisibilityCullerConfig::default();
    let vv = compute_vertex_visibility(&mesh, &camera, &config).expect("should succeed");
    for i in 0..3 {
        assert!(
            !vv.front_facing[i],
            "vertex {i} should be back-facing from this camera"
        );
        assert!(!vv.visible[i]);
    }
}

#[test]
fn test_vertex_visibility_fields_consistent() {
    let mesh = simple_mesh();
    let camera = front_camera();
    let config = VisibilityCullerConfig::default();
    let vv = compute_vertex_visibility(&mesh, &camera, &config).expect("should succeed");
    assert_eq!(vv.visible.len(), vv.n_vertices);
    assert_eq!(vv.in_frustum.len(), vv.n_vertices);
    assert_eq!(vv.front_facing.len(), vv.n_vertices);
    // visible must be in_frustum AND front_facing
    for i in 0..vv.n_vertices {
        assert_eq!(vv.visible[i], vv.in_frustum[i] && vv.front_facing[i]);
    }
}

// -----------------------------------------------------------------------
// compute_visibility_stats
// -----------------------------------------------------------------------

#[test]
fn test_visibility_stats_counts_match() {
    let mesh = simple_mesh();
    let camera = front_camera();
    let config = VisibilityCullerConfig::default();
    let vv = compute_vertex_visibility(&mesh, &camera, &config).expect("should succeed");
    let stats = compute_visibility_stats(&vv);
    assert_eq!(stats.n_vertices, 3);
    assert_eq!(stats.n_visible_vertices, 3);
    assert_eq!(stats.n_front_facing, 3);
    assert_eq!(stats.n_in_frustum, 3);
}

#[test]
fn test_visibility_stats_fractions_in_range() {
    let mesh = simple_mesh();
    let camera = front_camera();
    let config = VisibilityCullerConfig::default();
    let vv = compute_vertex_visibility(&mesh, &camera, &config).expect("should succeed");
    let stats = compute_visibility_stats(&vv);
    assert!(stats.visible_fraction >= 0.0 && stats.visible_fraction <= 1.0);
    assert!(stats.front_facing_fraction >= 0.0 && stats.front_facing_fraction <= 1.0);
    assert!(stats.in_frustum_fraction >= 0.0 && stats.in_frustum_fraction <= 1.0);
}

#[test]
fn test_visibility_stats_all_visible_fraction_one() {
    let mesh = simple_mesh();
    let camera = front_camera();
    let config = VisibilityCullerConfig::default();
    let vv = compute_vertex_visibility(&mesh, &camera, &config).expect("should succeed");
    let stats = compute_visibility_stats(&vv);
    assert!((stats.visible_fraction - 1.0).abs() < 1e-5);
}

#[test]
fn test_visibility_stats_none_visible() {
    let mesh = simple_mesh();
    let camera = back_camera();
    let config = VisibilityCullerConfig::default();
    let vv = compute_vertex_visibility(&mesh, &camera, &config).expect("should succeed");
    let stats = compute_visibility_stats(&vv);
    assert_eq!(stats.n_visible_vertices, 0);
    assert!((stats.visible_fraction - 0.0).abs() < 1e-5);
}

// -----------------------------------------------------------------------
// compute_multi_view_visibility
// -----------------------------------------------------------------------

#[test]
fn test_multi_view_no_cameras_error() {
    let mesh = simple_mesh();
    let config = VisibilityCullerConfig::default();
    let result = compute_multi_view_visibility(&mesh, &[], &config);
    assert!(matches!(result, Err(VisibilityError::NoCameras)));
}

#[test]
fn test_multi_view_single_camera() {
    let mesh = simple_mesh();
    let cameras = vec![front_camera()];
    let config = VisibilityCullerConfig::default();
    let mv = compute_multi_view_visibility(&mesh, &cameras, &config).expect("should succeed");
    assert_eq!(mv.n_cameras, 1);
    assert_eq!(mv.n_vertices, 3);
    // All visible from front camera
    for i in 0..3 {
        assert!(mv.any_visible[i]);
        assert!(mv.all_visible[i]);
        assert_eq!(mv.view_count[i], 1);
    }
}

#[test]
fn test_multi_view_two_cameras_any_but_not_all() {
    let mesh = simple_mesh();
    let cameras = vec![front_camera(), back_camera()];
    let config = VisibilityCullerConfig::default();
    let mv = compute_multi_view_visibility(&mesh, &cameras, &config).expect("should succeed");
    assert_eq!(mv.n_cameras, 2);
    // Front camera sees all, back camera sees none
    // → any_visible = true for all, all_visible = false for all (back cam sees 0)
    for i in 0..3 {
        assert!(mv.any_visible[i], "vertex {i} any_visible should be true");
        assert!(!mv.all_visible[i], "vertex {i} all_visible should be false");
        assert_eq!(mv.view_count[i], 1, "vertex {i} view_count should be 1");
    }
}

#[test]
fn test_multi_view_view_count_accumulates() {
    let mesh = simple_mesh();
    let cameras = vec![front_camera(), front_camera()]; // two identical cameras
    let config = VisibilityCullerConfig::default();
    let mv = compute_multi_view_visibility(&mesh, &cameras, &config).expect("should succeed");
    for i in 0..3 {
        assert_eq!(
            mv.view_count[i], 2,
            "two identical cameras both see vertex {i}"
        );
        assert!(mv.all_visible[i]);
    }
}

#[test]
fn test_multi_view_fields_lengths() {
    let mesh = simple_mesh();
    let cameras = vec![front_camera(), back_camera()];
    let config = VisibilityCullerConfig::default();
    let mv = compute_multi_view_visibility(&mesh, &cameras, &config).expect("should succeed");
    assert_eq!(mv.any_visible.len(), 3);
    assert_eq!(mv.all_visible.len(), 3);
    assert_eq!(mv.view_count.len(), 3);
}

// -----------------------------------------------------------------------
// find_view_dependent_vertices
// -----------------------------------------------------------------------

#[test]
fn test_find_view_dependent_two_cameras() {
    let mesh = simple_mesh();
    let cameras = vec![front_camera(), back_camera()];
    let config = VisibilityCullerConfig::default();
    let mv = compute_multi_view_visibility(&mesh, &cameras, &config).expect("should succeed");
    let deps = find_view_dependent_vertices(&mv);
    // All 3 vertices are view-dependent (any=true, all=false)
    assert_eq!(deps.len(), 3);
}

#[test]
fn test_find_view_dependent_all_visible_from_all_empty() {
    let mesh = simple_mesh();
    let cameras = vec![front_camera(), front_camera()];
    let config = VisibilityCullerConfig::default();
    let mv = compute_multi_view_visibility(&mesh, &cameras, &config).expect("should succeed");
    let deps = find_view_dependent_vertices(&mv);
    // All visible from all views → no view-dependent vertices
    assert!(deps.is_empty(), "should be empty, got {} items", deps.len());
}

#[test]
fn test_find_view_dependent_returns_indices_in_range() {
    let mesh = simple_mesh();
    let cameras = vec![front_camera(), back_camera()];
    let config = VisibilityCullerConfig::default();
    let mv = compute_multi_view_visibility(&mesh, &cameras, &config).expect("should succeed");
    let deps = find_view_dependent_vertices(&mv);
    for &idx in &deps {
        assert!(idx < mv.n_vertices, "index {idx} out of range");
    }
}

// -----------------------------------------------------------------------
// compute_optimal_view_coverage
// -----------------------------------------------------------------------

#[test]
fn test_optimal_coverage_no_cameras_error() {
    let mesh = simple_mesh();
    let config = VisibilityCullerConfig::default();
    let result = compute_optimal_view_coverage(&mesh, &[], &config);
    assert!(matches!(result, Err(VisibilityError::NoCameras)));
}

#[test]
fn test_optimal_coverage_front_camera_full() {
    let mesh = simple_mesh();
    let cameras = vec![front_camera()];
    let config = VisibilityCullerConfig::default();
    let cov = compute_optimal_view_coverage(&mesh, &cameras, &config).expect("should succeed");
    assert_eq!(cov.len(), 1);
    assert!(
        (cov[0] - 1.0).abs() < 1e-5,
        "all vertices visible, coverage should be 1.0"
    );
}

#[test]
fn test_optimal_coverage_back_camera_zero() {
    let mesh = simple_mesh();
    let cameras = vec![back_camera()];
    let config = VisibilityCullerConfig::default();
    let cov = compute_optimal_view_coverage(&mesh, &cameras, &config).expect("should succeed");
    assert_eq!(cov.len(), 1);
    assert!(
        (cov[0] - 0.0).abs() < 1e-5,
        "no vertices visible, coverage should be 0.0"
    );
}

#[test]
fn test_optimal_coverage_two_cameras_length() {
    let mesh = simple_mesh();
    let cameras = vec![front_camera(), back_camera()];
    let config = VisibilityCullerConfig::default();
    let cov = compute_optimal_view_coverage(&mesh, &cameras, &config).expect("should succeed");
    assert_eq!(cov.len(), 2);
}

#[test]
fn test_optimal_coverage_values_in_01() {
    let mesh = simple_mesh();
    let cameras = vec![front_camera(), back_camera()];
    let config = VisibilityCullerConfig::default();
    let cov = compute_optimal_view_coverage(&mesh, &cameras, &config).expect("should succeed");
    for &c in &cov {
        assert!((0.0..=1.0).contains(&c), "coverage {c} out of [0,1]");
    }
}

// -----------------------------------------------------------------------
// select_maximally_covering_views
// -----------------------------------------------------------------------

#[test]
fn test_select_top_k_basic() {
    let coverage = vec![0.3, 0.9, 0.5, 0.7];
    let top = select_maximally_covering_views(&coverage, 2);
    assert_eq!(top.len(), 2);
    assert_eq!(top[0], 1, "index 1 has highest coverage 0.9");
    assert_eq!(top[1], 3, "index 3 has second-highest coverage 0.7");
}

#[test]
fn test_select_top_k_exceeds_length_returns_all() {
    let coverage = vec![0.3, 0.8];
    let top = select_maximally_covering_views(&coverage, 10);
    assert_eq!(top.len(), 2, "k > n should return all cameras");
}

#[test]
fn test_select_top_k_zero_returns_empty() {
    let coverage = vec![0.5, 0.9];
    let top = select_maximally_covering_views(&coverage, 0);
    assert!(top.is_empty());
}

#[test]
fn test_select_top_k_single() {
    let coverage = vec![0.5];
    let top = select_maximally_covering_views(&coverage, 1);
    assert_eq!(top, vec![0]);
}

#[test]
fn test_select_top_k_deterministic_tie_break() {
    // Identical coverage: tie broken by index ascending
    let coverage = vec![0.5, 0.5, 0.5];
    let top = select_maximally_covering_views(&coverage, 2);
    assert_eq!(top, vec![0, 1]);
}

// -----------------------------------------------------------------------
// format_visibility_stats
// -----------------------------------------------------------------------

#[test]
fn test_format_visibility_stats_not_empty() {
    let mesh = simple_mesh();
    let camera = front_camera();
    let config = VisibilityCullerConfig::default();
    let vv = compute_vertex_visibility(&mesh, &camera, &config).expect("should succeed");
    let stats = compute_visibility_stats(&vv);
    let s = format_visibility_stats(&stats);
    assert!(!s.is_empty());
    assert!(
        s.contains("Visibility:"),
        "expected 'Visibility:' prefix in: {s}"
    );
}

#[test]
fn test_format_visibility_stats_contains_numbers() {
    let mesh = simple_mesh();
    let camera = front_camera();
    let config = VisibilityCullerConfig::default();
    let vv = compute_vertex_visibility(&mesh, &camera, &config).expect("should succeed");
    let stats = compute_visibility_stats(&vv);
    let s = format_visibility_stats(&stats);
    // Should contain vertex counts and fractions
    assert!(s.contains("3/3"), "should contain '3/3' in: {s}");
}

// -----------------------------------------------------------------------
// format_multi_view_stats
// -----------------------------------------------------------------------

#[test]
fn test_format_multi_view_stats_not_empty() {
    let mesh = simple_mesh();
    let cameras = vec![front_camera(), back_camera()];
    let config = VisibilityCullerConfig::default();
    let mv = compute_multi_view_visibility(&mesh, &cameras, &config).expect("should succeed");
    let s = format_multi_view_stats(&mv);
    assert!(!s.is_empty());
    assert!(
        s.contains("MultiView"),
        "expected 'MultiView' prefix in: {s}"
    );
}

#[test]
fn test_format_multi_view_stats_contains_cam_count() {
    let mesh = simple_mesh();
    let cameras = vec![front_camera(), back_camera()];
    let config = VisibilityCullerConfig::default();
    let mv = compute_multi_view_visibility(&mesh, &cameras, &config).expect("should succeed");
    let s = format_multi_view_stats(&mv);
    assert!(s.contains("2 cams"), "should mention camera count in: {s}");
}

// -----------------------------------------------------------------------
// VisibilityError variants
// -----------------------------------------------------------------------

#[test]
fn test_error_empty_mesh_display() {
    let e = VisibilityError::EmptyMesh;
    assert!(!e.to_string().is_empty());
}

#[test]
fn test_error_no_faces_display() {
    let e = VisibilityError::NoFaces;
    assert!(!e.to_string().is_empty());
}

#[test]
fn test_error_vertex_index_out_of_range_display() {
    let e = VisibilityError::VertexIndexOutOfRange { idx: 99, n: 3 };
    let s = e.to_string();
    assert!(s.contains("99"), "should mention idx 99 in: {s}");
    assert!(s.contains('3'), "should mention n=3 in: {s}");
}

#[test]
fn test_error_no_cameras_display() {
    let e = VisibilityError::NoCameras;
    assert!(!e.to_string().is_empty());
}

#[test]
fn test_error_normal_count_mismatch_display() {
    let e = VisibilityError::NormalCountMismatch {
        normals: 5,
        vertices: 3,
    };
    let s = e.to_string();
    assert!(s.contains('5'), "should mention normals=5 in: {s}");
    assert!(s.contains('3'), "should mention vertices=3 in: {s}");
}

// -----------------------------------------------------------------------
// VisibilityCullerConfig default values
// -----------------------------------------------------------------------

#[test]
fn test_config_default_values() {
    let config = VisibilityCullerConfig::default();
    assert_eq!(config.backface_threshold, 0.0);
    assert_eq!(config.frustum_margin, 0.0);
    assert!(!config.use_depth_test);
    assert!((config.depth_bias - 1e-4).abs() < 1e-7);
}

// -----------------------------------------------------------------------
// VertexVisibility and FaceVisibility field checks
// -----------------------------------------------------------------------

#[test]
fn test_vertex_visibility_n_vertices_correct() {
    let mesh = simple_mesh();
    let camera = front_camera();
    let config = VisibilityCullerConfig::default();
    let vv = compute_vertex_visibility(&mesh, &camera, &config).expect("should succeed");
    assert_eq!(vv.n_vertices, mesh.vertices.len());
}

#[test]
fn test_face_visibility_n_faces_correct() {
    let mesh = simple_mesh();
    let camera = front_camera();
    let config = VisibilityCullerConfig::default();
    let fv = compute_face_visibility(&mesh, &camera, &config).expect("should succeed");
    assert_eq!(fv.n_faces, mesh.faces.len());
}

#[test]
fn test_face_visibility_screen_area_length() {
    let mesh = simple_mesh();
    let camera = front_camera();
    let config = VisibilityCullerConfig::default();
    let fv = compute_face_visibility(&mesh, &camera, &config).expect("should succeed");
    assert_eq!(fv.screen_area.len(), fv.n_faces);
    assert_eq!(fv.visible.len(), fv.n_faces);
}

// -----------------------------------------------------------------------
// MultiViewVisibility field checks
// -----------------------------------------------------------------------

#[test]
fn test_multi_view_n_cameras_correct() {
    let mesh = simple_mesh();
    let cameras = vec![front_camera(), back_camera()];
    let config = VisibilityCullerConfig::default();
    let mv = compute_multi_view_visibility(&mesh, &cameras, &config).expect("should succeed");
    assert_eq!(mv.n_cameras, 2);
}

#[test]
fn test_multi_view_n_vertices_correct() {
    let mesh = simple_mesh();
    let cameras = vec![front_camera()];
    let config = VisibilityCullerConfig::default();
    let mv = compute_multi_view_visibility(&mesh, &cameras, &config).expect("should succeed");
    assert_eq!(mv.n_vertices, 3);
}

// -----------------------------------------------------------------------
// project_vertex edge cases
// -----------------------------------------------------------------------

#[test]
fn test_project_vertex_behind_camera_returns_none() {
    let camera = front_camera();
    // Camera is at world [0,0,2] looking toward -Z.
    // R * [0,0,3] + t = [-1,0,0]*0 + [-1,0,0]*0 + R*[0,0,3] ...
    // With R = [[-1,0,0],[0,1,0],[0,0,-1]] and t=[0,0,2]:
    // cam_pos = R*[0,0,3] + [0,0,2] = [0,0,-3] + [0,0,2] = [0,0,-1]
    // z = -1 ≤ near=0.01 → None
    let result = project_vertex([0.0, 0.0, 3.0], &camera);
    assert!(result.is_none(), "vertex behind camera should return None");
}

#[test]
fn test_project_vertex_in_front_of_camera_returns_some() {
    let camera = front_camera();
    // Vertex at origin: cam_pos = R*[0,0,0]+t = [0,0,2], z=2 > near → Some
    let result = project_vertex([0.0, 0.0, 0.0], &camera);
    assert!(
        result.is_some(),
        "vertex in front of camera should project successfully"
    );
}

// -----------------------------------------------------------------------
// camera_world_position / camera_direction
// -----------------------------------------------------------------------

#[test]
fn test_camera_world_position_and_direction() {
    // front_camera: R = diag(-1, 1, -1), t = [0, 0, 2] → −Rᵀt = [0, 0, 2]
    let cam_world = camera_world_position(&front_camera());
    assert!(
        cam_world[0].abs() < 1e-5 && cam_world[1].abs() < 1e-5,
        "{cam_world:?}"
    );
    assert!((cam_world[2] - 2.0).abs() < 1e-5, "{cam_world:?}");

    let dir = camera_direction([0.0, 0.0, 0.0], cam_world);
    assert!(
        (dir[2] - 1.0).abs() < 1e-5,
        "should point toward +Z: {dir:?}"
    );
    let len = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
    assert!(
        (len - 1.0).abs() < 1e-5,
        "direction must be unit, len={len}"
    );
}

// -----------------------------------------------------------------------
// Depth-occlusion test (use_depth_test / depth_bias)
// -----------------------------------------------------------------------

/// Small triangle at world z = 0 hidden behind a large triangle at z = 0.5
/// (which is nearer to `front_camera`, located at world [0, 0, 2]).
fn occluded_mesh() -> Mesh {
    let vertices = vec![
        na::Point3::new(0.0_f32, 0.0, 0.0),
        na::Point3::new(0.1_f32, 0.0, 0.0),
        na::Point3::new(0.0_f32, 0.1, 0.0),
        na::Point3::new(-0.3_f32, -0.3, 0.5),
        na::Point3::new(0.3_f32, -0.3, 0.5),
        na::Point3::new(0.0_f32, 0.3, 0.5),
    ];
    let normals = vec![na::Vector3::new(0.0_f32, 0.0, 1.0); 6];
    Mesh {
        vertices,
        normals,
        faces: vec![[0u32, 1, 2], [3u32, 4, 5]],
        uv_coords: Vec::new(),
    }
}

#[test]
fn test_depth_test_culls_occluded_vertices_only() {
    let mesh = occluded_mesh();
    let plain = VisibilityCullerConfig::default();
    let baseline =
        compute_vertex_visibility(&mesh, &front_camera(), &plain).expect("should succeed");
    assert_eq!(
        baseline.visible.iter().filter(|&&v| v).count(),
        6,
        "without the depth test every front-facing vertex counts as visible"
    );

    let config = VisibilityCullerConfig {
        use_depth_test: true,
        ..plain
    };
    let vv = compute_vertex_visibility(&mesh, &front_camera(), &config).expect("should succeed");

    for i in 0..3 {
        assert!(
            !vv.visible[i],
            "vertex {i} sits behind the occluder and must be culled"
        );
        // Only the occlusion test changed — it is still front-facing.
        assert!(vv.front_facing[i] && vv.in_frustum[i]);
    }
    // The occluder must not occlude itself (slope-scaled bias / uncovered
    // pixel rule), otherwise the test would be vacuous.
    for i in 3..6 {
        assert!(vv.visible[i], "occluder vertex {i} must stay visible");
    }
}

/// Steeply slanted surface (`z = −x`, camera-space depth 1.76 → 2.24 across
/// ~62 px) tessellated into a strip, plus probe vertices lying exactly on it
/// whose own pixels the strip covers.  Returns the probe vertex indices.
fn slanted_mesh() -> (Mesh, Vec<usize>) {
    let mut vertices: Vec<na::Point3<f32>> = Vec::new();
    let mut faces: Vec<[u32; 3]> = Vec::new();
    let columns = 8u32;
    for i in 0..=columns {
        let x = -0.24 + 0.48 * (i as f32) / (columns as f32);
        vertices.push(na::Point3::new(x, -0.24, -x));
        vertices.push(na::Point3::new(x, 0.24, -x));
    }
    for i in 0..columns {
        let base = 2 * i;
        faces.push([base, base + 2, base + 1]);
        faces.push([base + 1, base + 2, base + 3]);
    }

    let first_probe = vertices.len();
    for k in 0..5 {
        let x = -0.15 + 0.075 * k as f32;
        vertices.push(na::Point3::new(x, 0.013 * k as f32, -x));
    }
    let normal = na::Vector3::new(1.0_f32, 0.0, 1.0).normalize();
    let normals = vec![normal; vertices.len()];
    let probes = (first_probe..vertices.len()).collect();

    (
        Mesh {
            vertices,
            normals,
            faces,
            uv_coords: Vec::new(),
        },
        probes,
    )
}

#[test]
fn test_depth_test_keeps_slanted_surface_visible() {
    let (mesh, probes) = slanted_mesh();
    let camera = front_camera();
    let config = VisibilityCullerConfig {
        use_depth_test: true,
        ..Default::default()
    };
    let depth = rasterize_depth_buffer(&mesh, &camera);
    let width = camera.width as usize;

    for &probe in &probes {
        let v = mesh.vertices[probe];
        let screen = project_vertex([v.x, v.y, v.z], &camera).expect("probe must project");
        let pixel = (screen[1] as usize) * width + (screen[0] as usize);
        let stored = depth[pixel];
        // Non-vacuous: the probe's own pixel is covered, so the comparison
        // really runs instead of taking the "empty pixel" shortcut …
        assert!(stored.is_finite(), "probe {probe} landed on an empty pixel");
        // … and the surface recedes far faster per pixel than `depth_bias`,
        // so a fixed bias alone would report the probe as self-occluded.
        let step = (depth[pixel + 1] - stored).abs();
        assert!(
            step > 10.0 * config.depth_bias,
            "probe {probe}: one-pixel step {step} is too flat to be a test"
        );
    }

    let vv = compute_vertex_visibility(&mesh, &camera, &config).expect("should succeed");
    for &probe in &probes {
        assert!(
            vv.visible[probe],
            "probe {probe} lies on the surface and must not occlude itself"
        );
    }
}

#[test]
fn test_depth_test_culls_occluded_face() {
    let config = VisibilityCullerConfig {
        use_depth_test: true,
        ..Default::default()
    };
    let mesh = occluded_mesh();
    let fv = compute_face_visibility(&mesh, &front_camera(), &config).expect("should succeed");
    assert!(!fv.visible[0], "hidden face must be culled");
    assert!(fv.visible[1], "occluding face must stay visible");
}

#[test]
fn test_rasterize_depth_buffer_records_nearest_surface() {
    let camera = front_camera();
    let depth = rasterize_depth_buffer(&occluded_mesh(), &camera);
    let centre = depth[128 * camera.width as usize + 128];
    assert!(
        (centre - 1.5).abs() < 1e-3,
        "centre pixel should hold the nearer surface (z=1.5), got {centre}"
    );
    // A corner pixel is covered by no triangle.
    assert!(depth[0].is_infinite(), "uncovered pixel must stay infinite");
}

// -----------------------------------------------------------------------
// Greedy (set-cover) view selection
// -----------------------------------------------------------------------

fn visibility_from_mask(mask: &[bool]) -> VertexVisibility {
    VertexVisibility {
        visible: mask.to_vec(),
        in_frustum: mask.to_vec(),
        front_facing: mask.to_vec(),
        n_vertices: mask.len(),
    }
}

#[test]
fn test_greedy_selection_beats_top_k_on_overlapping_views() {
    // Views 0 and 1 are identical (coverage 4/6); view 2 covers the rest.
    let views = [
        visibility_from_mask(&[true, true, true, true, false, false]),
        visibility_from_mask(&[true, true, true, true, false, false]),
        visibility_from_mask(&[false, false, false, false, true, true]),
    ];
    let coverage: Vec<f32> = views.iter().map(coverage_fraction).collect();

    let top = select_top_coverage_views(&coverage, 2);
    assert_eq!(top, vec![0, 1], "top-k picks the two identical views");

    let greedy = select_greedy_covering_views(&views, 2);
    assert_eq!(greedy, vec![0, 2], "greedy must add marginal coverage");

    let union = |sel: &[usize]| {
        (0..6)
            .filter(|&v| sel.iter().any(|&c| views[c].visible[v]))
            .count()
    };
    assert!(
        union(&greedy) > union(&top),
        "greedy union {} should beat top-k union {}",
        union(&greedy),
        union(&top)
    );
}

#[test]
fn test_greedy_stops_when_no_marginal_coverage() {
    let views = [
        visibility_from_mask(&[true, true, false]),
        visibility_from_mask(&[true, false, false]), // subset of view 0
    ];
    let selected = select_greedy_covering_views(&views, 5);
    assert_eq!(
        selected,
        vec![0],
        "selection must stop instead of padding to k"
    );
}

#[test]
fn test_compute_greedy_view_selection_end_to_end() {
    let mesh = simple_mesh();
    let cameras = vec![back_camera(), front_camera()];
    let config = VisibilityCullerConfig::default();
    let selected =
        compute_greedy_view_selection(&mesh, &cameras, &config, 2).expect("should succeed");
    assert_eq!(selected, vec![1], "only the front camera covers anything");
    assert!(matches!(
        compute_greedy_view_selection(&mesh, &[], &config, 2),
        Err(VisibilityError::NoCameras)
    ));
}

#[test]
fn test_renamed_coverage_helpers_match_legacy_names() {
    let mesh = simple_mesh();
    let cameras = vec![front_camera(), back_camera()];
    let config = VisibilityCullerConfig::default();
    let renamed = compute_per_view_coverage(&mesh, &cameras, &config).expect("should succeed");
    let legacy = compute_optimal_view_coverage(&mesh, &cameras, &config).expect("should succeed");
    assert_eq!(renamed, legacy);
    assert_eq!(
        select_top_coverage_views(&renamed, 1),
        select_maximally_covering_views(&legacy, 1)
    );
}
