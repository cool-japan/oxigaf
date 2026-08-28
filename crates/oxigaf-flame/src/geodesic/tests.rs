//! Unit tests for the geodesic module.
//!
//! `use super::*` also pulls in the parent module's own imports (notably
//! `std::collections::HashSet`), which several adjacency assertions rely on.

use super::*;

// -----------------------------------------------------------------------
// Shared test meshes
// -----------------------------------------------------------------------

/// Single triangle mesh: vertices at (0,0,0), (1,0,0), (0,1,0).
fn triangle_mesh() -> GeodesicMesh {
    GeodesicMesh::new(
        vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        vec![[0, 1, 2]],
    )
    .expect("valid triangle mesh")
}

/// Linear chain mesh: 4 vertices connected as 0-1-2-3 via triangles
/// (using degenerate-ish strips). We create two triangles: [0,1,2] and [1,2,3].
fn chain_mesh() -> GeodesicMesh {
    // 0--1--2--3 as a strip of two triangles
    GeodesicMesh::new(
        vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ],
        vec![[0, 1, 2], [1, 2, 3]],
    )
    .expect("valid chain mesh")
}

/// Two-triangle mesh with 4 vertices forming a square split diagonally.
fn square_mesh() -> GeodesicMesh {
    GeodesicMesh::new(
        vec![
            [0.0, 0.0, 0.0], // 0
            [1.0, 0.0, 0.0], // 1
            [1.0, 1.0, 0.0], // 2
            [0.0, 1.0, 0.0], // 3
        ],
        vec![[0, 1, 2], [0, 2, 3]],
    )
    .expect("valid square mesh")
}

// -----------------------------------------------------------------------
// GeodesicMesh::new
// -----------------------------------------------------------------------

#[test]
fn test_mesh_new_valid() {
    let mesh = triangle_mesh();
    assert_eq!(mesh.n_vertices(), 3);
    assert_eq!(mesh.n_faces(), 1);
}

#[test]
fn test_mesh_new_empty_vertices() {
    let result = GeodesicMesh::new(vec![], vec![[0, 1, 2]]);
    assert!(matches!(result, Err(GeodesicError::EmptyMesh)));
}

#[test]
fn test_mesh_new_empty_faces() {
    let result = GeodesicMesh::new(
        vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        vec![],
    );
    assert!(matches!(result, Err(GeodesicError::EmptyFaces)));
}

#[test]
fn test_mesh_new_out_of_bounds_face() {
    let result = GeodesicMesh::new(vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], vec![[0, 1, 99]]);
    assert!(matches!(
        result,
        Err(GeodesicError::VertexOutOfBounds { idx: 99, n: 2 })
    ));
}

// -----------------------------------------------------------------------
// GeodesicMesh::build_adjacency
// -----------------------------------------------------------------------

#[test]
fn test_build_adjacency_triangle() {
    let mesh = triangle_mesh();
    let adj = mesh.build_adjacency();
    assert_eq!(adj.len(), 3);
    // Each vertex should have exactly 2 neighbors in a single triangle
    for (i, neighbors) in adj.iter().enumerate() {
        assert_eq!(neighbors.len(), 2, "vertex {i} should have 2 neighbors");
    }
    // Vertex 0 should be adjacent to 1 and 2
    let adj0: HashSet<usize> = adj[0].iter().copied().collect();
    assert!(adj0.contains(&1));
    assert!(adj0.contains(&2));
}

#[test]
fn test_build_adjacency_chain() {
    let mesh = chain_mesh();
    let adj = mesh.build_adjacency();
    // Vertex 0: connected to 1, 2 (from face [0,1,2])
    let adj0: HashSet<usize> = adj[0].iter().copied().collect();
    assert!(adj0.contains(&1));
    assert!(adj0.contains(&2));
    // Vertex 3: connected to 1, 2 (from face [1,2,3])
    let adj3: HashSet<usize> = adj[3].iter().copied().collect();
    assert!(adj3.contains(&1));
    assert!(adj3.contains(&2));
    // Vertex 1 and 2 are interior; connected to each other and to both 0 and 3
    let adj1: HashSet<usize> = adj[1].iter().copied().collect();
    assert!(adj1.contains(&0) && adj1.contains(&2) && adj1.contains(&3));
}

// -----------------------------------------------------------------------
// GeodesicMesh::edge_length
// -----------------------------------------------------------------------

#[test]
fn test_edge_length_unit() {
    let mesh = triangle_mesh();
    let len = mesh.edge_length(0, 1);
    assert!(
        (len - 1.0).abs() < 1e-6,
        "distance 0→1 should be 1.0, got {len}"
    );
}

#[test]
fn test_edge_length_diagonal() {
    let mesh = square_mesh();
    // Diagonal from (0,0,0) to (1,1,0)
    let len = mesh.edge_length(0, 2);
    assert!((len - std::f32::consts::SQRT_2).abs() < 1e-6);
}

// -----------------------------------------------------------------------
// GeodesicMesh::face_center
// -----------------------------------------------------------------------

#[test]
fn test_face_center_triangle() {
    let mesh = triangle_mesh();
    let center = mesh.face_center(0).expect("valid face");
    // Centroid of (0,0,0), (1,0,0), (0,1,0) = (1/3, 1/3, 0)
    assert!((center[0] - 1.0 / 3.0).abs() < 1e-6);
    assert!((center[1] - 1.0 / 3.0).abs() < 1e-6);
    assert!(center[2].abs() < 1e-6);
}

#[test]
fn test_face_center_out_of_bounds() {
    let mesh = triangle_mesh();
    assert!(matches!(
        mesh.face_center(99),
        Err(GeodesicError::VertexOutOfBounds { .. })
    ));
}

// -----------------------------------------------------------------------
// dijkstra: basic correctness
// -----------------------------------------------------------------------

#[test]
fn test_dijkstra_triangle_distances() {
    let mesh = triangle_mesh();
    let config = GeodesicConfig::default();
    let field = dijkstra(&mesh, &[0], &config).expect("dijkstra ok");
    assert_eq!(field.n_vertices, 3);
    // Source vertex has distance 0
    assert_eq!(field.distance_to(0).expect("ok"), 0.0);
    // Distance 0→1 = 1.0 (along x-axis)
    let d01 = field.distance_to(1).expect("ok");
    assert!((d01 - 1.0).abs() < 1e-6, "d01={d01}");
    // Distance 0→2 = 1.0 (along y-axis)
    let d02 = field.distance_to(2).expect("ok");
    assert!((d02 - 1.0).abs() < 1e-6, "d02={d02}");
}

#[test]
fn test_dijkstra_invalid_source() {
    let mesh = triangle_mesh();
    let config = GeodesicConfig::default();
    let result = dijkstra(&mesh, &[99], &config);
    assert!(matches!(
        result,
        Err(GeodesicError::VertexOutOfBounds { idx: 99, n: 3 })
    ));
}

#[test]
fn test_dijkstra_empty_sources() {
    let mesh = triangle_mesh();
    let config = GeodesicConfig::default();
    let result = dijkstra(&mesh, &[], &config);
    assert!(matches!(result, Err(GeodesicError::InvalidConfig(_))));
}

#[test]
fn test_dijkstra_chain_distances() {
    let mesh = chain_mesh();
    let config = GeodesicConfig::default();
    let field = dijkstra(&mesh, &[0], &config).expect("ok");
    // Source at 0
    assert_eq!(field.distance_to(0).expect("ok"), 0.0);
    // Distance 0→1 = 1.0
    let d01 = field.distance_to(1).expect("ok");
    assert!((d01 - 1.0).abs() < 1e-5, "d01={d01}");
    // Distance 0→3 ≥ 2.0 (shortest path through intermediate vertices)
    let d03 = field.distance_to(3).expect("ok");
    assert!(d03 >= 2.0 - 1e-5, "d03={d03} should be >= 2.0");
}

// -----------------------------------------------------------------------
// GeodesicField methods
// -----------------------------------------------------------------------

#[test]
fn test_field_distance_to_out_of_bounds() {
    let mesh = triangle_mesh();
    let config = GeodesicConfig::default();
    let field = dijkstra(&mesh, &[0], &config).expect("ok");
    assert!(matches!(
        field.distance_to(99),
        Err(GeodesicError::VertexOutOfBounds { idx: 99, n: 3 })
    ));
}

#[test]
fn test_field_is_reachable() {
    let mesh = triangle_mesh();
    let config = GeodesicConfig::default();
    let field = dijkstra(&mesh, &[0], &config).expect("ok");
    assert!(field.is_reachable(0));
    assert!(field.is_reachable(1));
    assert!(field.is_reachable(2));
    assert!(!field.is_reachable(99)); // out-of-bounds treated as unreachable
}

#[test]
fn test_field_shortest_path_one_hop() {
    let mesh = triangle_mesh();
    let config = GeodesicConfig::default();
    let field = dijkstra(&mesh, &[0], &config).expect("ok");
    let path = field.shortest_path(1).expect("ok");
    assert_eq!(path.first(), Some(&0));
    assert_eq!(path.last(), Some(&1));
}

#[test]
fn test_field_shortest_path_source_itself() {
    let mesh = triangle_mesh();
    let config = GeodesicConfig::default();
    let field = dijkstra(&mesh, &[0], &config).expect("ok");
    let path = field.shortest_path(0).expect("ok");
    assert_eq!(path, vec![0]);
}

#[test]
fn test_field_shortest_path_unreachable() {
    // Manually craft a field with infinite distance
    let field = GeodesicField {
        distances: vec![0.0, f32::INFINITY],
        predecessors: vec![None, None],
        n_vertices: 2,
    };
    assert!(matches!(
        field.shortest_path(1),
        Err(GeodesicError::Unreachable { idx: 1 })
    ));
}

#[test]
fn test_field_within_radius() {
    let mesh = triangle_mesh();
    let config = GeodesicConfig::default();
    let field = dijkstra(&mesh, &[0], &config).expect("ok");
    // Radius 0.0: only source
    let r0 = field.within_radius(0.0);
    assert_eq!(r0, vec![0]);
    // Radius 1.5: all vertices (distances are 0, 1, 1)
    let r15 = field.within_radius(1.5);
    assert_eq!(r15.len(), 3);
}

#[test]
fn test_field_normalize() {
    let mesh = triangle_mesh();
    let config = GeodesicConfig::default();
    let field = dijkstra(&mesh, &[0], &config).expect("ok");
    let normalized = field.normalize();
    assert_eq!(normalized.len(), 3);
    // Source is 0.0, max becomes 1.0
    assert!((normalized[0] - 0.0).abs() < 1e-6);
    // Both vertex 1 and 2 have distance 1.0 = max, so normalized = 1.0
    assert!((normalized[1] - 1.0).abs() < 1e-6);
    assert!((normalized[2] - 1.0).abs() < 1e-6);
}

#[test]
fn test_field_max_mean_distance() {
    let mesh = chain_mesh();
    let config = GeodesicConfig::default();
    let field = dijkstra(&mesh, &[0], &config).expect("ok");
    let max = field.max_distance();
    assert!(max > 0.0);
    let mean = field.mean_distance();
    assert!(mean > 0.0 && mean <= max);
}

// -----------------------------------------------------------------------
// multi_source_dijkstra
// -----------------------------------------------------------------------

#[test]
fn test_multi_source_dijkstra_two_sources() {
    let mesh = chain_mesh();
    let config = GeodesicConfig::default();
    // Sources at both ends: 0 and 3
    let field = multi_source_dijkstra(&mesh, &[0, 3], &config).expect("ok");
    // Both sources should have distance 0
    assert_eq!(field.distance_to(0).expect("ok"), 0.0);
    assert_eq!(field.distance_to(3).expect("ok"), 0.0);
    // Interior vertices should be closer than if only one source
    let d1 = field.distance_to(1).expect("ok");
    let d2 = field.distance_to(2).expect("ok");
    assert!(d1 < 2.0);
    assert!(d2 < 2.0);
}

#[test]
fn test_multi_source_dijkstra_nearest_source() {
    let mesh = chain_mesh();
    let config = GeodesicConfig::default();
    let field = multi_source_dijkstra(&mesh, &[0, 3], &config).expect("ok");
    // nearest_source should be a source vertex (distance 0)
    let nearest = field.nearest_source();
    assert!(nearest == 0 || nearest == 3);
}

// -----------------------------------------------------------------------
// pairwise_geodesic
// -----------------------------------------------------------------------

#[test]
fn test_pairwise_geodesic_single_landmark() {
    let mesh = triangle_mesh();
    let config = GeodesicConfig::default();
    let matrix = pairwise_geodesic(&mesh, &[0], &config).expect("ok");
    assert_eq!(matrix.len(), 1);
    assert_eq!(matrix[0], 0.0);
}

#[test]
fn test_pairwise_geodesic_two_landmarks() {
    let mesh = triangle_mesh();
    let config = GeodesicConfig::default();
    let matrix = pairwise_geodesic(&mesh, &[0, 1], &config).expect("ok");
    // 2x2 matrix, row-major
    assert_eq!(matrix.len(), 4);
    // Diagonal is 0
    assert_eq!(matrix[0], 0.0); // [0,0]
    assert_eq!(matrix[3], 0.0); // [1,1]
                                // Off-diagonal should be symmetric
    assert!(
        (matrix[1] - matrix[2]).abs() < 1e-5,
        "symmetric: {}",
        (matrix[1] - matrix[2]).abs()
    );
}

#[test]
fn test_pairwise_geodesic_symmetric() {
    let mesh = square_mesh();
    let config = GeodesicConfig::default();
    let landmarks = vec![0, 1, 2, 3];
    let matrix = pairwise_geodesic(&mesh, &landmarks, &config).expect("ok");
    let k = 4;
    for i in 0..k {
        for j in 0..k {
            let d_ij = matrix[i * k + j];
            let d_ji = matrix[j * k + i];
            assert!(
                (d_ij - d_ji).abs() < 1e-5,
                "asymmetric: d[{i},{j}]={d_ij} vs d[{j},{i}]={d_ji}"
            );
        }
    }
}

// -----------------------------------------------------------------------
// geodesic_voronoi
// -----------------------------------------------------------------------

#[test]
fn test_geodesic_voronoi_single_source() {
    let mesh = triangle_mesh();
    let config = GeodesicConfig::default();
    let labels = geodesic_voronoi(&mesh, &[0], &config).expect("ok");
    assert_eq!(labels.len(), 3);
    assert!(labels.iter().all(|&l| l == 0));
}

#[test]
fn test_geodesic_voronoi_two_sources() {
    let mesh = chain_mesh();
    let config = GeodesicConfig::default();
    let labels = geodesic_voronoi(&mesh, &[0, 3], &config).expect("ok");
    assert_eq!(labels.len(), 4);
    // Each vertex gets one of two labels (0 or 1)
    assert!(labels.iter().all(|&l| l == 0 || l == 1));
}

// -----------------------------------------------------------------------
// geodesic_weights
// -----------------------------------------------------------------------

#[test]
fn test_geodesic_weights_large_sigma() {
    let mesh = triangle_mesh();
    let config = GeodesicConfig::default();
    let field = dijkstra(&mesh, &[0], &config).expect("ok");
    // Very large sigma → weights approach 1.0 uniformly
    let weights = geodesic_weights(&field, 1000.0);
    for w in &weights {
        assert!((w - 1.0).abs() < 0.01, "w={w}");
    }
}

#[test]
fn test_geodesic_weights_source_highest() {
    let mesh = triangle_mesh();
    let config = GeodesicConfig::default();
    let field = dijkstra(&mesh, &[0], &config).expect("ok");
    let weights = geodesic_weights(&field, 0.5);
    // Source (dist=0) → weight=1.0, others lower
    assert!(
        (weights[0] - 1.0).abs() < 1e-6,
        "source weight={}",
        weights[0]
    );
    assert!(weights[0] >= weights[1]);
    assert!(weights[0] >= weights[2]);
}

// -----------------------------------------------------------------------
// geodesic_ball
// -----------------------------------------------------------------------

#[test]
fn test_geodesic_ball_source_included() {
    let mesh = triangle_mesh();
    let config = GeodesicConfig::default();
    let ball = geodesic_ball(&mesh, 0, 0.5, &config).expect("ok");
    assert!(ball.contains(&0), "source must be in ball");
}

#[test]
fn test_geodesic_ball_radius_zero() {
    let mesh = triangle_mesh();
    let config = GeodesicConfig::default();
    let ball = geodesic_ball(&mesh, 0, 0.0, &config).expect("ok");
    assert_eq!(ball, vec![0]);
}

#[test]
fn test_geodesic_ball_all_vertices() {
    let mesh = triangle_mesh();
    let config = GeodesicConfig::default();
    // Radius larger than diameter → all 3 vertices
    let ball = geodesic_ball(&mesh, 0, 10.0, &config).expect("ok");
    assert_eq!(ball.len(), 3);
}

// -----------------------------------------------------------------------
// geodesic_diameter
// -----------------------------------------------------------------------

#[test]
fn test_geodesic_diameter_triangle() {
    let mesh = triangle_mesh();
    let config = GeodesicConfig::default();
    let diam = geodesic_diameter(&mesh, &config).expect("ok");
    // Max distance in equilateral-ish triangle is 1.0
    assert!(diam > 0.0);
    assert!(diam <= 2.0);
}

#[test]
fn test_geodesic_diameter_chain() {
    let mesh = chain_mesh();
    let config = GeodesicConfig::default();
    let diam = geodesic_diameter(&mesh, &config).expect("ok");
    // Chain 0-1-2-3, min path from 0 to 3 is at least 2.0
    assert!(diam >= 2.0 - 1e-5, "diameter={diam}");
}

// -----------------------------------------------------------------------
// geodesic_center
// -----------------------------------------------------------------------

#[test]
fn test_geodesic_center_triangle() {
    let mesh = triangle_mesh();
    let config = GeodesicConfig::default();
    let center = geodesic_center(&mesh, &[], &config).expect("ok");
    // All vertices are symmetric in the triangle — any is valid
    assert!(center < 3);
}

#[test]
fn test_geodesic_center_with_sample() {
    let mesh = square_mesh();
    let config = GeodesicConfig::default();
    // Restrict to vertices 0 and 2
    let center = geodesic_center(&mesh, &[0, 2], &config).expect("ok");
    assert!(center == 0 || center == 2);
}

// -----------------------------------------------------------------------
// smooth_geodesic_path
// -----------------------------------------------------------------------

#[test]
fn test_smooth_geodesic_path_empty() {
    let mesh = triangle_mesh();
    let result = smooth_geodesic_path(&mesh, &[], 5).expect("ok");
    assert!(result.is_empty());
}

#[test]
fn test_smooth_geodesic_path_single_vertex() {
    let mesh = triangle_mesh();
    let result = smooth_geodesic_path(&mesh, &[1], 3).expect("ok");
    assert_eq!(result.len(), 1);
    assert_eq!(result[0], mesh.vertices[1]);
}

#[test]
fn test_smooth_geodesic_path_endpoints_pinned() {
    let mesh = chain_mesh();
    let path = vec![0, 1, 2, 3];
    let result = smooth_geodesic_path(&mesh, &path, 5).expect("ok");
    assert_eq!(result.len(), 4);
    // Endpoints should remain unchanged
    assert_eq!(result[0], mesh.vertices[0]);
    assert_eq!(result[3], mesh.vertices[3]);
}

#[test]
fn test_smooth_geodesic_path_out_of_bounds() {
    let mesh = triangle_mesh();
    let result = smooth_geodesic_path(&mesh, &[0, 99], 1);
    assert!(matches!(
        result,
        Err(GeodesicError::VertexOutOfBounds { idx: 99, n: 3 })
    ));
}

// -----------------------------------------------------------------------
// compute_geodesic_stats
// -----------------------------------------------------------------------

#[test]
fn test_compute_geodesic_stats_known_distances() {
    let field = GeodesicField {
        distances: vec![0.0, 1.0, 2.0],
        predecessors: vec![None, Some(0), Some(1)],
        n_vertices: 3,
    };
    let stats = compute_geodesic_stats(&field);
    assert_eq!(stats.n_reachable, 3);
    assert!((stats.min_distance - 0.0).abs() < 1e-6);
    assert!((stats.max_distance - 2.0).abs() < 1e-6);
    assert!((stats.mean_distance - 1.0).abs() < 1e-6);
    // std of [0, 1, 2] = sqrt(2/3) ≈ 0.8165
    assert!((stats.std_distance - (2.0_f32 / 3.0).sqrt()).abs() < 1e-5);
}

#[test]
fn test_compute_geodesic_stats_with_inf() {
    let field = GeodesicField {
        distances: vec![0.0, 1.0, f32::INFINITY],
        predecessors: vec![None, Some(0), None],
        n_vertices: 3,
    };
    let stats = compute_geodesic_stats(&field);
    assert_eq!(stats.n_reachable, 2);
    assert!((stats.max_distance - 1.0).abs() < 1e-6);
}

#[test]
fn test_compute_geodesic_stats_empty_field() {
    let field = GeodesicField {
        distances: vec![f32::INFINITY],
        predecessors: vec![None],
        n_vertices: 1,
    };
    let stats = compute_geodesic_stats(&field);
    assert_eq!(stats.n_reachable, 0);
    assert_eq!(stats.mean_distance, 0.0);
}

// -----------------------------------------------------------------------
// heat_geodesic
// -----------------------------------------------------------------------

#[test]
fn test_heat_geodesic_output_size() {
    let mesh = triangle_mesh();
    let field = heat_geodesic(&mesh, 0, 100, 0.1).expect("ok");
    assert_eq!(field.n_vertices, 3);
    assert_eq!(field.distances.len(), 3);
}

#[test]
fn test_heat_geodesic_source_min_distance() {
    let mesh = triangle_mesh();
    let field = heat_geodesic(&mesh, 0, 200, 0.05).expect("ok");
    // Source should have the minimum distance (≈ 0)
    let d0 = field.distance_to(0).expect("ok");
    assert!(d0 < 1e-4, "source distance should be ~0, got {d0}");
    // Other vertices should have non-negative distances
    for i in 0..3 {
        let d = field.distance_to(i).expect("ok");
        assert!(d >= 0.0, "distance to {i} should be >= 0, got {d}");
    }
}

#[test]
fn test_heat_geodesic_invalid_dt() {
    let mesh = triangle_mesh();
    let result = heat_geodesic(&mesh, 0, 10, -0.1);
    assert!(matches!(result, Err(GeodesicError::InvalidConfig(_))));
}

#[test]
fn test_heat_geodesic_zero_iters() {
    let mesh = triangle_mesh();
    let result = heat_geodesic(&mesh, 0, 0, 0.1);
    assert!(matches!(result, Err(GeodesicError::InvalidConfig(_))));
}

// -----------------------------------------------------------------------
// heat_geodesic: regression — output must be a real distance
// -----------------------------------------------------------------------

/// A flat regular grid, `res × res` vertices spanning `[0, extent]²` in the
/// z = 0 plane, split into two triangles per cell.  Planar, so the true
/// geodesic distance is exactly the Euclidean distance.
fn grid_mesh(res: usize, extent: f32) -> GeodesicMesh {
    let mut vertices = Vec::with_capacity(res * res);
    for j in 0..res {
        for i in 0..res {
            let x = extent * i as f32 / (res - 1) as f32;
            let y = extent * j as f32 / (res - 1) as f32;
            vertices.push([x, y, 0.0]);
        }
    }
    let mut faces = Vec::new();
    for j in 0..res - 1 {
        for i in 0..res - 1 {
            let a = j * res + i;
            let b = a + 1;
            let c = a + res;
            let d = c + 1;
            faces.push([a, b, d]);
            faces.push([a, d, c]);
        }
    }
    GeodesicMesh::new(vertices, faces).expect("valid grid mesh")
}

#[test]
fn test_heat_geodesic_scales_with_geometry() {
    // The same connectivity at two different scales must give distances
    // that differ by exactly that scale factor.  The old combinatorial
    // implementation never read vertex positions and returned identical
    // numbers for both.
    let small = grid_mesh(9, 1.0);
    let large = grid_mesh(9, 3.0);

    let dt_small = heat_time_step(&small);
    let dt_large = heat_time_step(&large);
    let f_small = heat_geodesic(&small, 0, 400, dt_small).expect("small ok");
    let f_large = heat_geodesic(&large, 0, 400, dt_large).expect("large ok");

    let corner = 9 * 9 - 1;
    let d_small = f_small.distance_to(corner).expect("ok");
    let d_large = f_large.distance_to(corner).expect("ok");

    assert!(d_small > 0.0, "distance must be positive, got {d_small}");
    let ratio = d_large / d_small;
    assert!(
        (ratio - 3.0).abs() < 0.25,
        "tripling the mesh must triple the distance, got ratio {ratio}"
    );
}

#[test]
fn test_heat_geodesic_approximates_planar_distance() {
    // On a flat grid the geodesic distance is the Euclidean distance, so
    // the heat method must reproduce it to within its approximation error.
    let res = 11;
    let extent = 1.0f32;
    let mesh = grid_mesh(res, extent);
    let dt = heat_time_step(&mesh);
    let field = heat_geodesic(&mesh, 0, 600, dt).expect("heat ok");

    let source = mesh.vertices[0];
    for &target in &[res - 1, res * (res - 1), res * res - 1, res * 5 + 5] {
        let p = mesh.vertices[target];
        let truth =
            ((p[0] - source[0]).powi(2) + (p[1] - source[1]).powi(2) + (p[2] - source[2]).powi(2))
                .sqrt();
        let got = field.distance_to(target).expect("ok");
        let rel_err = (got - truth).abs() / truth;
        assert!(
            rel_err < 0.20,
            "vertex {target}: heat distance {got} vs true {truth} (rel err {rel_err})"
        );
    }
}

#[test]
fn test_heat_geodesic_is_monotone_from_source() {
    // Distance must grow with ring index on a regular grid — the old
    // `-ln(h)` conversion saturated at a constant for every vertex the
    // diffusion had not yet reached, destroying monotonicity.
    let res = 9;
    let mesh = grid_mesh(res, 1.0);
    let dt = heat_time_step(&mesh);
    let field = heat_geodesic(&mesh, 0, 600, dt).expect("heat ok");

    let mut prev = -1.0f32;
    for step in 0..res {
        let v = step * res + step; // diagonal from the source corner
        let d = field.distance_to(v).expect("ok");
        assert!(
            d > prev,
            "distance must increase along the diagonal: step {step} gave {d} after {prev}"
        );
        prev = d;
    }
}

#[test]
fn test_heat_geodesic_multi_source() {
    let res = 9;
    let mesh = grid_mesh(res, 1.0);
    let dt = heat_time_step(&mesh);
    let corner = res * res - 1;
    let field = heat_geodesic_multi(&mesh, &[0, corner], 600, Some(dt)).expect("multi-source ok");

    assert!(field.distance_to(0).expect("ok").abs() < 1e-5);
    assert!(field.distance_to(corner).expect("ok").abs() < 1e-5);
    // The centre is roughly equidistant from both corners and strictly
    // farther than either source.
    let centre = (res / 2) * res + res / 2;
    assert!(field.distance_to(centre).expect("ok") > 0.0);
}

#[test]
fn test_heat_geodesic_multi_rejects_empty_sources() {
    let mesh = triangle_mesh();
    let result = heat_geodesic_multi(&mesh, &[], 100, None);
    assert!(matches!(result, Err(GeodesicError::InvalidConfig(_))));
}

#[test]
fn test_heat_geodesic_out_of_bounds_source() {
    let mesh = triangle_mesh();
    let result = heat_geodesic(&mesh, 99, 100, 0.1);
    assert!(matches!(
        result,
        Err(GeodesicError::VertexOutOfBounds { idx: 99, n: 3 })
    ));
}

#[test]
fn test_heat_time_step_is_squared_mean_edge() {
    // Unit right triangle: edges 1, 1, sqrt(2) → mean = (2 + sqrt2)/3.
    let mesh = triangle_mesh();
    let expected = ((2.0 + std::f32::consts::SQRT_2) / 3.0).powi(2);
    let got = heat_time_step(&mesh);
    assert!((got - expected).abs() < 1e-5, "got {got}, want {expected}");
}

// -----------------------------------------------------------------------
// use_face_graph
// -----------------------------------------------------------------------

#[test]
fn test_face_graph_changes_distances() {
    // On a grid, the edge graph must detour around triangles while the
    // face graph may cut through them, so face-graph distances are never
    // longer and are strictly shorter somewhere.
    let res = 7;
    let mesh = grid_mesh(res, 1.0);

    let edge_cfg = GeodesicConfig {
        use_face_graph: false,
        ..GeodesicConfig::default()
    };
    let face_cfg = GeodesicConfig {
        use_face_graph: true,
        ..GeodesicConfig::default()
    };

    let edge_field = dijkstra(&mesh, &[0], &edge_cfg).expect("edge ok");
    let face_field = dijkstra(&mesh, &[0], &face_cfg).expect("face ok");

    assert_eq!(face_field.distances.len(), mesh.n_vertices());

    let mut any_shorter = false;
    for v in 0..mesh.n_vertices() {
        let de = edge_field.distance_to(v).expect("ok");
        let df = face_field.distance_to(v).expect("ok");
        assert!(
            df <= de + 1e-5,
            "face graph must never be longer at {v}: {df} vs {de}"
        );
        if df < de - 1e-4 {
            any_shorter = true;
        }
    }
    assert!(
        any_shorter,
        "use_face_graph = true must actually change some distance"
    );
}

#[test]
fn test_face_graph_paths_are_mesh_vertices() {
    // Every node of the face-augmented graph is a real mesh vertex, so a
    // reconstructed path must consist entirely of valid vertex indices.
    let res = 6;
    let mesh = grid_mesh(res, 1.0);
    let cfg = GeodesicConfig {
        use_face_graph: true,
        ..GeodesicConfig::default()
    };
    let field = dijkstra(&mesh, &[0], &cfg).expect("ok");
    assert_eq!(field.n_vertices, mesh.n_vertices());

    let path = field.shortest_path(res * res - 1).expect("path ok");
    assert_eq!(path.first(), Some(&0));
    assert_eq!(path.last(), Some(&(res * res - 1)));
    for &v in &path {
        assert!(
            v < mesh.n_vertices(),
            "path contains invalid node {v} (mesh has {} vertices)",
            mesh.n_vertices()
        );
    }
}

#[test]
fn test_face_graph_is_more_accurate_on_a_plane() {
    // On a flat grid the true geodesic distance is the Euclidean distance.
    // Edge-graph paths must zig-zag along grid edges and over-estimate it;
    // unfolding across triangle pairs must measurably close that gap.
    let res = 9;
    let mesh = grid_mesh(res, 1.0);
    let edge_cfg = GeodesicConfig::default();
    let face_cfg = GeodesicConfig {
        use_face_graph: true,
        ..GeodesicConfig::default()
    };

    let edge_field = dijkstra(&mesh, &[0], &edge_cfg).expect("edge ok");
    let face_field = dijkstra(&mesh, &[0], &face_cfg).expect("face ok");

    let source = mesh.vertices[0];
    let mean_err = |field: &GeodesicField| -> f32 {
        let mut total = 0.0f32;
        for v in 1..mesh.n_vertices() {
            let p = mesh.vertices[v];
            let truth = ((p[0] - source[0]).powi(2)
                + (p[1] - source[1]).powi(2)
                + (p[2] - source[2]).powi(2))
            .sqrt();
            let got = field.distance_to(v).unwrap_or(f32::INFINITY);
            total += (got - truth).abs() / truth;
        }
        total / (mesh.n_vertices() - 1) as f32
    };

    let edge_err = mean_err(&edge_field);
    let face_err = mean_err(&face_field);
    assert!(
        face_err < edge_err * 0.5,
        "unfolding must at least halve the edge-graph error: face {face_err} vs edge {edge_err}"
    );
}

#[test]
fn test_face_graph_matches_edge_graph_on_a_single_triangle() {
    // A lone triangle has no interior edge, so there is nothing to unfold and
    // both modes must agree exactly.
    let mesh = triangle_mesh();
    let edge_cfg = GeodesicConfig::default();
    let face_cfg = GeodesicConfig {
        use_face_graph: true,
        ..GeodesicConfig::default()
    };
    let edge_field = dijkstra(&mesh, &[0], &edge_cfg).expect("edge ok");
    let face_field = dijkstra(&mesh, &[0], &face_cfg).expect("face ok");
    for v in 0..mesh.n_vertices() {
        let de = edge_field.distance_to(v).expect("ok");
        let df = face_field.distance_to(v).expect("ok");
        assert!((de - df).abs() < 1e-6, "vertex {v}: {de} vs {df}");
    }
}

#[test]
fn test_face_graph_voronoi_labels_sized_to_vertices() {
    let mesh = chain_mesh();
    let cfg = GeodesicConfig {
        use_face_graph: true,
        ..GeodesicConfig::default()
    };
    let labels = geodesic_voronoi(&mesh, &[0, 3], &cfg).expect("ok");
    assert_eq!(labels.len(), mesh.n_vertices());
    assert!(labels.iter().all(|&l| l < 2));
}

#[test]
fn test_face_graph_pairwise_symmetric() {
    let mesh = square_mesh();
    let cfg = GeodesicConfig {
        use_face_graph: true,
        ..GeodesicConfig::default()
    };
    let matrix = pairwise_geodesic(&mesh, &[0, 1, 2, 3], &cfg).expect("ok");
    let k = 4;
    for i in 0..k {
        for j in 0..k {
            assert!((matrix[i * k + j] - matrix[j * k + i]).abs() < 1e-5);
        }
    }
}

// -----------------------------------------------------------------------
// geodesic_center: sampled default
// -----------------------------------------------------------------------

#[test]
fn test_geodesic_center_empty_samples_is_bounded_work() {
    // The empty-slice default must sample rather than run one Dijkstra per
    // vertex.  With more vertices than DEFAULT_CENTER_SAMPLES the routine
    // still returns promptly and yields a valid vertex.
    let mesh = grid_mesh(15, 1.0); // 225 vertices > 64 samples
    let cfg = GeodesicConfig::default();
    let center = geodesic_center(&mesh, &[], &cfg).expect("ok");
    assert!(center < mesh.n_vertices());
}

#[test]
fn test_geodesic_center_sampled_respects_budget() {
    let mesh = grid_mesh(9, 1.0);
    let cfg = GeodesicConfig::default();
    // A budget of 1 can only ever return the farthest-point seed, vertex 0.
    let center = geodesic_center_sampled(&mesh, &[], 1, &cfg).expect("ok");
    assert_eq!(center, 0);
}

#[test]
fn test_geodesic_center_sampled_rejects_zero_budget() {
    let mesh = triangle_mesh();
    let cfg = GeodesicConfig::default();
    let result = geodesic_center_sampled(&mesh, &[], 0, &cfg);
    assert!(matches!(result, Err(GeodesicError::InvalidConfig(_))));
}

#[test]
fn test_geodesic_center_explicit_samples_ignore_budget() {
    let mesh = square_mesh();
    let cfg = GeodesicConfig::default();
    let center = geodesic_center_sampled(&mesh, &[1, 3], 1, &cfg).expect("ok");
    assert!(center == 1 || center == 3, "got {center}");
}

#[test]
fn test_geodesic_center_sampled_finds_interior_of_grid() {
    // On a symmetric grid the exhaustive centre is an interior vertex; the
    // sampled default must land near it, not on a corner.
    let res = 11;
    let mesh = grid_mesh(res, 1.0);
    let cfg = GeodesicConfig::default();
    let center = geodesic_center(&mesh, &[], &cfg).expect("ok");

    let (row, col) = (center / res, center % res);
    assert!(
        row > 0 && row < res - 1 && col > 0 && col < res - 1,
        "centre {center} (row {row}, col {col}) should be interior"
    );
}

#[test]
fn test_geodesic_center_out_of_bounds_candidate() {
    let mesh = triangle_mesh();
    let cfg = GeodesicConfig::default();
    let result = geodesic_center(&mesh, &[0, 99], &cfg);
    assert!(matches!(
        result,
        Err(GeodesicError::VertexOutOfBounds { idx: 99, n: 3 })
    ));
}
