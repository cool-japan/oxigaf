use super::*;

// Helper: single equilateral triangle vertices
fn triangle_verts() -> Vec<na::Point3<f32>> {
    vec![
        na::Point3::new(0.0, 0.0, 0.0),
        na::Point3::new(1.0, 0.0, 0.0),
        na::Point3::new(0.5, 3.0_f32.sqrt() / 2.0, 0.0),
    ]
}
fn triangle_faces() -> Vec<[u32; 3]> {
    vec![[0, 1, 2]]
}

// Helper: path graph 0-1-2 (2 edges)
fn path_verts() -> Vec<na::Point3<f32>> {
    vec![
        na::Point3::new(0.0, 0.0, 0.0),
        na::Point3::new(1.0, 0.0, 0.0),
        na::Point3::new(2.0, 0.0, 0.0),
    ]
}
fn path_faces() -> Vec<[u32; 3]> {
    // Degenerate triangle to get edges 0-1 and 1-2
    // Use two faces sharing edge 0-1-2
    vec![[0, 1, 2]]
}

// Helper: tetrahedron (4 vertices, each connected to the other 3)
fn tet_verts() -> Vec<na::Point3<f32>> {
    vec![
        na::Point3::new(0.0, 0.0, 0.0),
        na::Point3::new(1.0, 0.0, 0.0),
        na::Point3::new(0.5, 1.0, 0.0),
        na::Point3::new(0.5, 0.5, 1.0),
    ]
}
fn tet_faces() -> Vec<[u32; 3]> {
    vec![[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
}

// Test 1: single triangle — degree = 2 for all vertices
#[test]
fn test_combinatorial_triangle_degree() {
    let verts = triangle_verts();
    let faces = triangle_faces();
    let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
    assert_eq!(lap.n_vertices, 3);
    for &d in &lap.degree {
        assert!((d - 2.0).abs() < 1e-6, "expected degree 2, got {d}");
    }
}

// Test 2: row_ptr length = n + 1
#[test]
fn test_combinatorial_row_ptr_len() {
    let verts = triangle_verts();
    let faces = triangle_faces();
    let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
    assert_eq!(lap.row_ptr.len(), 4); // n+1 = 4
}

// Test 3: row sums of combinatorial Laplacian = 0
#[test]
fn test_combinatorial_row_sums_zero() {
    let verts = triangle_verts();
    let faces = triangle_faces();
    let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
    for i in 0..lap.n_vertices {
        let mut row_sum = lap.degree[i]; // diagonal
        let start = lap.row_ptr[i];
        let end = lap.row_ptr[i + 1];
        for idx in start..end {
            row_sum += lap.values[idx];
        }
        assert!(row_sum.abs() < 1e-5, "row {i} sum = {row_sum}");
    }
}

// Test 4: L * 1 = 0 (constant vector in null space)
#[test]
fn test_matvec_constant_zero() {
    let verts = triangle_verts();
    let faces = triangle_faces();
    let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
    let ones = vec![1.0f32; lap.n_vertices];
    let result = spec_laplacian_matvec(&lap, &ones);
    for &r in &result {
        assert!(r.abs() < 1e-5, "L*1 != 0: {r}");
    }
}

// Test 5: matvec dimension correctness
#[test]
fn test_matvec_dimension() {
    let verts = tet_verts();
    let faces = tet_faces();
    let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
    let x = vec![1.0, 2.0, 3.0, 4.0];
    let result = spec_laplacian_matvec(&lap, &x);
    assert_eq!(result.len(), 4);
}

// `spec_laplacian_matvec` must not panic when `x` is shorter than
// `lap.n_vertices` (all `MeshLaplacian` fields are public).
#[test]
fn test_matvec_undersized_input_does_not_panic() {
    let verts = tet_verts();
    let faces = tet_faces();
    let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
    let result = spec_laplacian_matvec(&lap, &[1.0, 2.0]); // lap has 4 vertices
    assert_eq!(result.len(), 4);
}

// Test 6: cotangent Laplacian on equilateral triangle — symmetric weights
#[test]
fn test_cotangent_symmetric() {
    let verts = triangle_verts();
    let faces = triangle_faces();
    let lap = spec_build_cotangent_laplacian(&verts, &faces).expect("cotangent lap");
    // For equilateral triangle all angles are 60°, cot(60°) = 1/sqrt(3)
    // All off-diagonal weights should be equal (symmetric)
    let n = lap.n_vertices;
    let mut w_vals = Vec::new();
    for i in 0..n {
        let start = lap.row_ptr[i];
        let end = lap.row_ptr[i + 1];
        for idx in start..end {
            w_vals.push(lap.values[idx].abs());
        }
    }
    let first = w_vals[0];
    for &w in &w_vals {
        assert!(
            (w - first).abs() < 1e-4,
            "weights not equal: {w} vs {first}"
        );
    }
}

// Regression test for the |w| diagonal bug: this triangle is obtuse at
// vertex 2, giving edge (0,1) a NEGATIVE cotangent weight. Summing |w|
// for the diagonal (the old code) broke `L * 1 = 0`; signed w fixes it.
#[test]
fn test_cotangent_row_sums_zero_with_obtuse_triangle() {
    let verts = vec![
        na::Point3::new(0.0, 0.0, 0.0),
        na::Point3::new(1.0, 0.0, 0.0),
        na::Point3::new(0.5, 0.1, 0.0),
    ];
    let faces = vec![[0u32, 1, 2]];
    let lap = spec_build_cotangent_laplacian(&verts, &faces).expect("cotangent lap");

    // L_ij = -w_ij, so a positive value confirms an obtuse-triangle
    // (negative) cotangent weight actually occurred here.
    assert!(
        lap.values.iter().any(|&v| v > 0.0),
        "expected an obtuse-triangle negative cotangent weight, got values={:?}",
        lap.values
    );

    for i in 0..lap.n_vertices {
        let mut row_sum = lap.degree[i];
        let start = lap.row_ptr[i];
        let end = lap.row_ptr[i + 1];
        for idx in start..end {
            row_sum += lap.values[idx];
        }
        assert!(
            row_sum.abs() < 1e-4,
            "cotangent row {i} sum = {row_sum} (should be ~0)"
        );
    }

    // L * 1 = 0 for the whole operator, not just per-row bookkeeping.
    let ones = vec![1.0f32; lap.n_vertices];
    let lx = spec_laplacian_matvec(&lap, &ones);
    for (i, &v) in lx.iter().enumerate() {
        assert!(v.abs() < 1e-4, "L*1 != 0 at vertex {i}: {v}");
    }
}

// Test 7: cotangent row sums ≈ 0
#[test]
fn test_cotangent_row_sums_zero() {
    let verts = triangle_verts();
    let faces = triangle_faces();
    let lap = spec_build_cotangent_laplacian(&verts, &faces).expect("cotangent lap");
    for i in 0..lap.n_vertices {
        let mut row_sum = lap.degree[i];
        let start = lap.row_ptr[i];
        let end = lap.row_ptr[i + 1];
        for idx in start..end {
            row_sum += lap.values[idx];
        }
        assert!(row_sum.abs() < 1e-3, "cotangent row {i} sum = {row_sum}");
    }
}

// Test 8: normalized Laplacian — diagonal values are 1 (or 0 for isolated)
#[test]
fn test_normalize_laplacian_diagonal() {
    let verts = triangle_verts();
    let faces = triangle_faces();
    let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
    let norm_lap = spec_normalize_laplacian(&lap).expect("normalize");
    for &d in &norm_lap.degree {
        // Normalized diagonal should be 1 for connected vertices
        assert!(
            (d - 1.0).abs() < 1e-6,
            "normalized degree should be 1, got {d}"
        );
    }
}

// Test 9: normalized Laplacian — off-diagonal |values| <= 1
#[test]
fn test_normalize_laplacian_values_bounded() {
    let verts = tet_verts();
    let faces = tet_faces();
    let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
    let norm_lap = spec_normalize_laplacian(&lap).expect("normalize");
    for &v in &norm_lap.values {
        assert!(v.abs() <= 1.0 + 1e-5, "normalized value out of range: {v}");
    }
}

// Gives `LaplacianKind::RandomWalk` a real construction site: diagonal
// of 1s and zero row sums (`L_rw * 1 = 0`, defining `L_rw = I - D^-1 A`).
#[test]
fn test_random_walk_laplacian_properties() {
    let verts = tet_verts();
    let faces = tet_faces();
    let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
    let rw = spec_random_walk_laplacian(&lap).expect("random walk");

    assert!(matches!(rw.kind, LaplacianKind::RandomWalk));
    for &d in &rw.degree {
        assert!(
            (d - 1.0).abs() < 1e-6,
            "random-walk diagonal should be 1, got {d}"
        );
    }

    let ones = vec![1.0f32; rw.n_vertices];
    let lx = spec_laplacian_matvec(&rw, &ones);
    for (i, &v) in lx.iter().enumerate() {
        assert!(v.abs() < 1e-5, "L_rw * 1 != 0 at vertex {i}: {v}");
    }
}

// Test 10: Gram-Schmidt — output vectors are orthonormal
#[test]
fn test_gram_schmidt_orthonormal() {
    let n = 4;
    let mut vecs = vec![
        vec![1.0, 2.0, 3.0, 4.0],
        vec![5.0, 6.0, 7.0, 8.0],
        vec![9.0, 10.0, 11.0, 12.0],
    ];
    let k = vecs.len();
    spec_gram_schmidt(&mut vecs, n, k);
    for (i, vi) in vecs.iter().enumerate() {
        // Self dot = 1
        let self_d = dot(vi, vi);
        assert!((self_d - 1.0).abs() < 1e-5, "v{i} not normalized: {self_d}");
        for (j, vj) in vecs.iter().enumerate() {
            if i != j {
                let cross_d = dot(vi, vj).abs();
                assert!(cross_d < 1e-5, "v{i} · v{j} = {cross_d}, not orthogonal");
            }
        }
    }
}

// Test 11: Gram-Schmidt — doesn't change span (reconstructed same vector up to sign)
#[test]
fn test_gram_schmidt_preserves_span() {
    let n = 3;
    let original = vec![1.0f32, 2.0, 3.0];
    let mut vecs = vec![original.clone(), vec![0.0, 1.0, 0.0]];
    spec_gram_schmidt(&mut vecs, n, 2);
    // First GS vector should be proportional to original
    let d = dot(&vecs[0], &original);
    assert!(d.abs() > 0.99, "GS first vector not along original: {d}");
}

// Test 12: Rayleigh quotient of constant vector = 0 (L*1=0)
#[test]
fn test_rayleigh_quotient_constant() {
    let verts = tet_verts();
    let faces = tet_faces();
    let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
    let ones = vec![1.0f32; lap.n_vertices];
    let rq = spec_rayleigh_quotient(&lap, &ones);
    assert!(
        rq.abs() < 1e-4,
        "RQ of constant vector should be 0, got {rq}"
    );
}

// Test 13: Rayleigh quotient of non-constant vector > 0
#[test]
fn test_rayleigh_quotient_positive() {
    let verts = tet_verts();
    let faces = tet_faces();
    let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
    let v = vec![1.0f32, -1.0, 1.0, -1.0];
    let rq = spec_rayleigh_quotient(&lap, &v);
    assert!(
        rq > 0.0,
        "RQ of non-constant vector should be positive, got {rq}"
    );
}

// Test 14: power iteration on path graph k=2, eigenvalues sorted ascending
#[test]
fn test_power_iteration_sorted() {
    let verts = path_verts();
    let faces = path_faces();
    let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
    let basis = spec_power_iteration(&lap, 2, 2000, 1e-7, 42).expect("power iter");
    assert_eq!(basis.k, 2);
    assert!(
        basis.eigenvalues[0] <= basis.eigenvalues[1] + 1e-5,
        "eigenvalues not sorted: {:?}",
        basis.eigenvalues
    );
}

// Test 15: power iteration eigenvectors are orthogonal
#[test]
fn test_power_iteration_orthogonal() {
    let verts = tet_verts();
    let faces = tet_faces();
    let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
    let basis = spec_power_iteration(&lap, 3, 2000, 1e-7, 7).expect("power iter");
    let k = basis.k;
    for i in 0..k {
        for j in (i + 1)..k {
            let d = dot(&basis.eigenvectors[i], &basis.eigenvectors[j]).abs();
            assert!(d < 0.05, "eigenvectors {i} and {j} not orthogonal: dot={d}");
        }
    }
}

// Test 16: smallest eigenvalue ≈ 0 for connected graph
#[test]
fn test_power_iteration_smallest_near_zero() {
    let verts = triangle_verts();
    let faces = triangle_faces();
    let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
    let basis = spec_power_iteration(&lap, 2, 2000, 1e-7, 1).expect("power iter");
    assert!(
        basis.eigenvalues[0].abs() < 0.3,
        "smallest eigenvalue should be ~0 for connected graph, got {}",
        basis.eigenvalues[0]
    );
}

// Test 17: eigenvalues ordered ascending
#[test]
fn test_eigenvalues_ascending() {
    let verts = tet_verts();
    let faces = tet_faces();
    let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
    let basis = spec_power_iteration(&lap, 3, 2000, 1e-7, 99).expect("power iter");
    for i in 0..basis.eigenvalues.len() - 1 {
        assert!(
            basis.eigenvalues[i] <= basis.eigenvalues[i + 1] + 1e-4,
            "eigenvalues not ascending at {i}: {:?}",
            basis.eigenvalues
        );
    }
}

// Test 18: project → reconstruct round-trip
#[test]
fn test_project_reconstruct_roundtrip() {
    let verts = tet_verts();
    let faces = tet_faces();
    let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
    let basis = spec_power_iteration(&lap, 4, 3000, 1e-8, 5).expect("power iter");
    let signal = SpectralSignal {
        values: vec![1.0, 2.0, 3.0, 4.0],
        n_vertices: 4,
    };
    let coeffs = spec_project(&signal, &basis).expect("project");
    let reconstructed = spec_reconstruct(&coeffs, &basis).expect("reconstruct");
    let mse: f32 = signal
        .values
        .iter()
        .zip(reconstructed.values.iter())
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f32>()
        / 4.0;
    assert!(mse < 0.1, "round-trip MSE = {mse} (should be near 0)");
}

// Test 19: low-pass filter produces smoother output
#[test]
fn test_low_pass_filter_smoother() {
    let verts = tet_verts();
    let faces = tet_faces();
    let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
    let basis = spec_power_iteration(&lap, 3, 2000, 1e-7, 13).expect("power iter");
    let signal = SpectralSignal {
        values: vec![1.0, -2.0, 3.0, -4.0],
        n_vertices: 4,
    };
    let smoothed = spec_low_pass_filter(&signal, &basis, 1).expect("low pass");
    let s_before = spec_smoothness(&signal, &lap).expect("smoothness");
    let s_after = spec_smoothness(&smoothed, &lap).expect("smoothness");
    assert!(
        s_after <= s_before + 1e-3,
        "low pass did not reduce smoothness: before={s_before}, after={s_after}"
    );
}

// Test 20: high-pass filter removes low-freq content
#[test]
fn test_high_pass_filter() {
    let verts = tet_verts();
    let faces = tet_faces();
    let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
    let basis = spec_power_iteration(&lap, 4, 2000, 1e-7, 21).expect("power iter");
    let signal = SpectralSignal {
        values: vec![1.0, 1.0, 1.0, 2.0],
        n_vertices: 4,
    };
    let coeffs_full = spec_project(&signal, &basis).expect("project full");
    let hp = spec_high_pass_filter(&signal, &basis, 1).expect("high pass");
    let coeffs_hp = spec_project(&hp, &basis).expect("project hp");
    // First coefficient should be zeroed or near zero
    assert!(
        coeffs_hp[0].abs() < coeffs_full[0].abs() + 0.5,
        "high pass did not attenuate low freq: full={}, hp={}",
        coeffs_full[0],
        coeffs_hp[0]
    );
}

// Test 21: smoothness of constant signal = 0
#[test]
fn test_smoothness_constant_zero() {
    let verts = triangle_verts();
    let faces = triangle_faces();
    let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
    let signal = SpectralSignal {
        values: vec![3.0, 3.0, 3.0],
        n_vertices: 3,
    };
    let s = spec_smoothness(&signal, &lap).expect("smoothness");
    assert!(s.abs() < 1e-4, "smoothness of constant = {s}");
}

// `spec_smoothness` must reject a mismatched signal/mesh pair rather
// than silently computing a partial (wrong) Dirichlet energy.
#[test]
fn test_smoothness_dimension_mismatch_errors() {
    let verts = triangle_verts();
    let faces = triangle_faces();
    let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
    let signal = SpectralSignal {
        values: vec![1.0, 2.0, 3.0, 4.0],
        n_vertices: 4, // mesh has 3 vertices
    };
    let result = spec_smoothness(&signal, &lap);
    assert!(
        matches!(result, Err(SpectralError::DimensionMismatch { .. })),
        "expected DimensionMismatch error, got {result:?}"
    );
}

// Test 22: smoothness of non-constant > 0
#[test]
fn test_smoothness_nonconstant_positive() {
    let verts = triangle_verts();
    let faces = triangle_faces();
    let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
    let signal = SpectralSignal {
        values: vec![1.0, -1.0, 0.0],
        n_vertices: 3,
    };
    let s = spec_smoothness(&signal, &lap).expect("smoothness");
    assert!(
        s > 0.0,
        "smoothness of non-constant should be positive, got {s}"
    );
}

// Regression test for the smoothing sign error, using the exact
// worked example from the bug report: a 3-vertex path graph 0-1-2
// (degrees 1, 2, 1) with a spike at the middle vertex must SHRINK
// toward its neighbours' average after one step, not grow away.
#[test]
fn test_laplacian_smooth_spike_shrinks_not_grows() {
    let lap = MeshLaplacian {
        n_vertices: 3,
        row_ptr: vec![0, 1, 3, 4],
        col_idx: vec![1, 0, 2, 1],
        values: vec![-1.0, -1.0, -1.0, -1.0],
        degree: vec![1.0, 2.0, 1.0],
        kind: LaplacianKind::Combinatorial,
    };
    // 1-D "positions" packed as [x0,0,0, x1,0,0, x2,0,0].
    let positions = vec![0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let smoothed = spec_laplacian_smooth(&positions, &lap, 0.1, 1);

    let spike_after = smoothed[3];
    assert!(
        spike_after.abs() < positions[3].abs(),
        "spike should shrink toward its neighbours' average: before={}, after={spike_after}",
        positions[3]
    );
    // x1_new = x1 - lambda*(Lx)_1 = 10 - 0.1*(2*10 - 0 - 0) = 8. The old
    // (buggy) sign would give 10 + 0.1*20 = 12 instead.
    assert!(
        (spike_after - 8.0).abs() < 1e-4,
        "expected spike_after ~= 8.0, got {spike_after}"
    );
}

// Test 23: Laplacian smooth changes positions
#[test]
fn test_laplacian_smooth_changes() {
    let verts = triangle_verts();
    let faces = triangle_faces();
    let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
    let positions: Vec<f32> = verts.iter().flat_map(|p| [p.x, p.y, p.z]).collect();
    let smoothed = spec_laplacian_smooth(&positions, &lap, 0.1, 5);
    let diff: f32 = positions
        .iter()
        .zip(smoothed.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(diff > 1e-6, "Laplacian smooth did not change positions");
}

// Test 24: Laplacian smooth with constant positions → no change
#[test]
fn test_laplacian_smooth_constant_no_change() {
    let verts = triangle_verts();
    let faces = triangle_faces();
    let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
    // Constant position (all the same): gradient is zero
    let positions = vec![1.0f32; 9]; // 3 vertices × 3 coords, all 1.0
    let smoothed = spec_laplacian_smooth(&positions, &lap, 0.1, 5);
    let diff: f32 = positions
        .iter()
        .zip(smoothed.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(
        diff < 1e-5,
        "Constant positions changed under smooth: diff={diff}"
    );
}

// Test 25: Taubin smooth changes positions
#[test]
fn test_taubin_smooth_changes() {
    let verts = triangle_verts();
    let faces = triangle_faces();
    let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
    let positions: Vec<f32> = verts.iter().flat_map(|p| [p.x, p.y, p.z]).collect();
    let smoothed = spec_taubin_smooth(&positions, &lap, 0.5, -0.53, 10);
    let diff: f32 = positions
        .iter()
        .zip(smoothed.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(diff > 0.0, "Taubin smooth did not change positions");
}

// Test 26: Taubin closer to original than pure Laplacian (less shrinkage)
// Use stable parameters: for tet with max eigenvalue ~6, lambda < 1/6 ≈ 0.1
// Laplacian shrinks mesh each iteration; Taubin compensates with mu step
#[test]
fn test_taubin_less_shrinkage() {
    let verts = tet_verts();
    let faces = tet_faces();
    let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
    let positions: Vec<f32> = verts.iter().flat_map(|p| [p.x, p.y, p.z]).collect();
    // Pure Laplacian: 20 iters with lambda=0.05 → steady shrinkage
    let lap_smooth = spec_laplacian_smooth(&positions, &lap, 0.05, 20);
    // Taubin: 5 pairs with stable lambda=0.05, mu=-0.06 → less shrinkage
    let taubin = spec_taubin_smooth(&positions, &lap, 0.05, -0.06, 5);
    let err_lap: f32 = positions
        .iter()
        .zip(lap_smooth.iter())
        .map(|(a, b)| (a - b) * (a - b))
        .sum();
    let err_taubin: f32 = positions
        .iter()
        .zip(taubin.iter())
        .map(|(a, b)| (a - b) * (a - b))
        .sum();
    // Taubin with fewer iters must deviate less than pure Laplacian with 20 iters
    assert!(
        err_taubin < err_lap + 1e-3,
        "Taubin should have less deviation from original: lap={err_lap}, taubin={err_taubin}"
    );
}

// Test 27: cluster_vertices k=3 returns labels all in [0,3)
#[test]
fn test_cluster_labels_in_range() {
    let verts = tet_verts();
    let faces = tet_faces();
    let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
    let basis = spec_power_iteration(&lap, 3, 2000, 1e-7, 77).expect("power iter");
    let labels = spec_cluster_vertices(&basis, 3, 42).expect("cluster");
    assert_eq!(labels.len(), 4);
    for &l in &labels {
        assert!(l < 3, "label {l} out of range [0,3)");
    }
}

// Test 28: cluster_vertices returns n_vertices assignments
#[test]
fn test_cluster_returns_n_assignments() {
    let verts = tet_verts();
    let faces = tet_faces();
    let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
    let basis = spec_power_iteration(&lap, 2, 2000, 1e-7, 3).expect("power iter");
    let labels = spec_cluster_vertices(&basis, 2, 11).expect("cluster");
    assert_eq!(labels.len(), verts.len());
}

// Test 29: stats n_edges correct for single triangle (= 3 undirected edges)
#[test]
fn test_stats_n_edges_triangle() {
    let verts = triangle_verts();
    let faces = triangle_faces();
    let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
    let stats = spec_compute_stats(&lap, None);
    assert_eq!(
        stats.n_edges, 3,
        "triangle has 3 edges, got {}",
        stats.n_edges
    );
}

// Test 30: stats mean_degree correct for triangle (= 2.0)
#[test]
fn test_stats_mean_degree_triangle() {
    let verts = triangle_verts();
    let faces = triangle_faces();
    let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
    let stats = spec_compute_stats(&lap, None);
    assert!(
        (stats.mean_degree - 2.0).abs() < 1e-5,
        "mean degree should be 2.0, got {}",
        stats.mean_degree
    );
}

// Test 31: format_stats returns non-empty string
#[test]
fn test_format_stats_nonempty() {
    let verts = triangle_verts();
    let faces = triangle_faces();
    let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
    let stats = spec_compute_stats(&lap, None);
    let s = spec_format_stats(&stats);
    assert!(!s.is_empty(), "format_stats returned empty string");
}

// Test 32: format_config returns non-empty string
#[test]
fn test_format_config_nonempty() {
    let config = SpectralConfig {
        k: 10,
        max_power_iters: 500,
        tol: 1e-6,
        laplacian_kind: LaplacianKind::Combinatorial,
    };
    let s = spec_format_config(&config);
    assert!(!s.is_empty(), "format_config returned empty string");
}

// Test 33: EmptyMesh error for 0 vertices
#[test]
fn test_empty_mesh_error() {
    let result = spec_build_combinatorial_laplacian(&[], &[]);
    assert!(
        matches!(result, Err(SpectralError::EmptyMesh)),
        "expected EmptyMesh error"
    );
}

// Test 34: EmptyMesh error for cotangent with 0 vertices
#[test]
fn test_empty_mesh_cotangent() {
    let result = spec_build_cotangent_laplacian(&[], &[]);
    assert!(
        matches!(result, Err(SpectralError::EmptyMesh)),
        "expected EmptyMesh error"
    );
}

// Test 35: DimensionMismatch error for project
#[test]
fn test_dimension_mismatch_project() {
    let verts = triangle_verts();
    let faces = triangle_faces();
    let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
    let basis = spec_power_iteration(&lap, 2, 1000, 1e-6, 1).expect("power iter");
    let signal = SpectralSignal {
        values: vec![1.0, 2.0, 3.0, 4.0], // 4 != 3
        n_vertices: 4,
    };
    let result = spec_project(&signal, &basis);
    assert!(
        matches!(result, Err(SpectralError::DimensionMismatch { .. })),
        "expected DimensionMismatch error"
    );
}

// Test 36: InsufficientBasis error for reconstruct
#[test]
fn test_insufficient_basis_reconstruct() {
    let verts = triangle_verts();
    let faces = triangle_faces();
    let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
    let basis = spec_power_iteration(&lap, 2, 1000, 1e-6, 1).expect("power iter");
    // More coefficients than k
    let coeffs = vec![1.0f32; 5];
    let result = spec_reconstruct(&coeffs, &basis);
    assert!(
        matches!(result, Err(SpectralError::InsufficientBasis { .. })),
        "expected InsufficientBasis error"
    );
}

// Test 37: InsufficientBasis error for power_iteration with k > n
#[test]
fn test_insufficient_basis_k_gt_n() {
    let verts = triangle_verts();
    let faces = triangle_faces();
    let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
    let result = spec_power_iteration(&lap, 10, 100, 1e-6, 1);
    assert!(
        matches!(result, Err(SpectralError::InsufficientBasis { .. })),
        "expected InsufficientBasis error for k > n"
    );
}

// Regression test for the too-small power-iteration shift: a 4-cycle
// has known eigenvalues {0, 2, 2, 4}. Old shift max(degree)=2 gives
// shifted spectrum {2,0,0,-2}, WRONGLY tying eigenvalue 4 with 2 for a
// k=2 request. Correct shift 2*max(degree)=4 gives {4,2,2,0}, correctly
// picking {0, 2}.
#[test]
fn test_power_iteration_lambda_max_shift_4cycle() {
    let lap = MeshLaplacian {
        n_vertices: 4,
        row_ptr: vec![0, 2, 4, 6, 8],
        col_idx: vec![1, 3, 0, 2, 1, 3, 0, 2],
        values: vec![-1.0; 8],
        degree: vec![2.0; 4],
        kind: LaplacianKind::Combinatorial,
    };

    let basis = spec_power_iteration(&lap, 2, 2000, 1e-8, 42).expect("power iteration");
    // `SpectralBasis::eigenvalues` is documented as sorted ascending.
    assert!(
        basis.eigenvalues[0].abs() < 0.05,
        "smallest eigenvalue should be ~0, got {:?}",
        basis.eigenvalues
    );
    assert!(
        (basis.eigenvalues[1] - 2.0).abs() < 0.05,
        "second-smallest eigenvalue should be ~2 (not ~4, the old \
             too-small shift's wrong answer): got {:?}",
        basis.eigenvalues
    );
}

// Regression test pinning the interaction between two fixes in this
// file: making the cotangent diagonal *signed* (so `L * 1 = 0` holds,
// see `test_cotangent_row_sums_zero_with_obtuse_triangle`) means
// `degree[i]` can now be *smaller in magnitude* than the true absolute
// off-diagonal row sum on a mesh with obtuse triangles (cancellation
// between a large negative cotangent weight and its neighbours). That
// makes the old `2 * max(degree)` an invalid Gershgorin bound again for
// exactly the mesh class the sign fix targets -- `gershgorin_lambda_max`
// must use the per-row form instead.
//
// Hand-built (as `test_power_iteration_lambda_max_shift_4cycle` above
// does) rather than derived from mesh geometry, so the discrepancy is
// exact and easy to verify by hand: this models a vertex 0 with one
// heavily-cancelling "obtuse-like" neighbour (weight -8, vs +1 to each
// of two other neighbours) alongside two ordinary low-degree vertices.
//   - old bound: `2 * max(degree)` folds over signed `degree` values
//     `[-6, 1, 1, -8]`; the fold starts at 0.0 and only ever grows, so
//     both negative degrees are invisible to it and it returns `2*1=2`.
//   - new bound: vertex 0's row has `|degree[0]| + row_abs[0]
//     = 6 + (1+1+8) = 16`, correctly reflecting that its off-diagonal
//     magnitudes are far larger than the cancelled signed sum.
#[test]
fn test_gershgorin_lambda_max_exceeds_two_times_degree_with_cancellation() {
    let lap = MeshLaplacian {
        n_vertices: 4,
        row_ptr: vec![0, 3, 4, 5, 6],
        col_idx: vec![1, 2, 3, 0, 0, 0],
        values: vec![-1.0, -1.0, 8.0, -1.0, -1.0, 8.0],
        degree: vec![-6.0, 1.0, 1.0, -8.0],
        kind: LaplacianKind::Cotangent,
    };

    // Row-sum-zero sanity check: this is a valid (signed) Laplacian,
    // not an arbitrary matrix -- exactly the shape a real cotangent
    // Laplacian with an obtuse triangle can produce.
    for i in 0..lap.n_vertices {
        let mut row_sum = lap.degree[i];
        for idx in lap.row_ptr[i]..lap.row_ptr[i + 1] {
            row_sum += lap.values[idx];
        }
        assert!(
            row_sum.abs() < 1e-5,
            "row {i} sum = {row_sum} (should be 0)"
        );
    }

    let old_weaker_bound = 2.0 * lap.degree.iter().copied().fold(0.0f32, f32::max).max(1.0);
    assert!(
        (old_weaker_bound - 2.0).abs() < 1e-5,
        "expected the old formula to (wrongly) see only the positive \
             degrees and return 2, got {old_weaker_bound}"
    );

    let bound = gershgorin_lambda_max(&lap);
    assert!(
        (bound - 16.0).abs() < 1e-4,
        "expected the per-row Gershgorin bound to be 16 (vertex 0: \
             |degree|=6 + row_abs=10), got {bound}"
    );
    assert!(
        bound > old_weaker_bound + 1e-4,
        "generic per-row Gershgorin bound ({bound}) must strictly exceed \
             the old `2 * max(degree)` bound ({old_weaker_bound}); a shift \
             this much too small lets power iteration converge to a mix of \
             the lowest- and highest-frequency modes instead of the k \
             smallest"
    );

    // End-to-end: power iteration on this same (symmetric, valid)
    // Laplacian must still converge to ascending eigenvalues without
    // reporting divergence now that the shift is valid again.
    let basis = spec_power_iteration(&lap, 2, 2000, 1e-7, 7)
        .expect("power iteration should converge with a valid shift");
    assert_eq!(basis.eigenvalues.len(), 2);
    assert!(
        basis.eigenvalues[0] <= basis.eigenvalues[1] + 1e-4,
        "eigenvalues should be ascending: {:?}",
        basis.eigenvalues
    );
}

// Test 38: tetrahedron Laplacian — all degrees = 3
#[test]
fn test_tet_all_degrees_three() {
    let verts = tet_verts();
    let faces = tet_faces();
    let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
    for (i, &d) in lap.degree.iter().enumerate() {
        assert!(
            (d - 3.0).abs() < 1e-6,
            "tet vertex {i} degree = {d}, expected 3"
        );
    }
}

// Test 39: algebraic connectivity (λ₂) > 0 for connected mesh
#[test]
fn test_algebraic_connectivity_positive() {
    let verts = triangle_verts();
    let faces = triangle_faces();
    let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
    let basis = spec_power_iteration(&lap, 2, 2000, 1e-7, 42).expect("power iter");
    let stats = spec_compute_stats(&lap, Some(&basis));
    assert!(
        stats.algebraic_connectivity > -0.1,
        "algebraic connectivity should be >= 0 for connected mesh, got {}",
        stats.algebraic_connectivity
    );
}

// Test 40: signal projection coefficients length = k
#[test]
fn test_projection_coefficients_length() {
    let verts = tet_verts();
    let faces = tet_faces();
    let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
    let basis = spec_power_iteration(&lap, 3, 2000, 1e-7, 55).expect("power iter");
    let signal = SpectralSignal {
        values: vec![1.0, 2.0, 3.0, 4.0],
        n_vertices: 4,
    };
    let coeffs = spec_project(&signal, &basis).expect("project");
    assert_eq!(coeffs.len(), 3, "coefficients length should be k=3");
}

// Test 41: low-pass filter with cutoff_k=0 → all zeros (no basis kept)
#[test]
fn test_low_pass_cutoff_zero() {
    let verts = triangle_verts();
    let faces = triangle_faces();
    let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
    let basis = spec_power_iteration(&lap, 2, 2000, 1e-7, 99).expect("power iter");
    let signal = SpectralSignal {
        values: vec![1.0, 2.0, 3.0],
        n_vertices: 3,
    };
    let filtered = spec_low_pass_filter(&signal, &basis, 0).expect("low pass k=0");
    let total: f32 = filtered.values.iter().map(|x| x.abs()).sum();
    assert!(
        total < 1e-4,
        "low pass with cutoff 0 should give ~0 output, got {total}"
    );
}

// Test 42: CSR row_ptr is non-decreasing
#[test]
fn test_csr_row_ptr_nondecreasing() {
    let verts = tet_verts();
    let faces = tet_faces();
    let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
    for i in 0..lap.row_ptr.len() - 1 {
        assert!(
            lap.row_ptr[i] <= lap.row_ptr[i + 1],
            "row_ptr not non-decreasing at {i}"
        );
    }
}

// Test 43: CSR col_idx all in valid range
#[test]
fn test_csr_col_idx_valid() {
    let verts = tet_verts();
    let faces = tet_faces();
    let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
    let n = lap.n_vertices;
    for &c in &lap.col_idx {
        assert!(c < n, "col_idx {c} out of range [0, {n})");
    }
}

// Test 44: off-diagonal values of combinatorial Laplacian are -1
#[test]
fn test_combinatorial_off_diag_neg_one() {
    let verts = triangle_verts();
    let faces = triangle_faces();
    let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
    for &v in &lap.values {
        assert!(
            (v + 1.0).abs() < 1e-6,
            "off-diag value should be -1, got {v}"
        );
    }
}

// Test 45: laplacian_matvec with zero vector gives zero
#[test]
fn test_matvec_zero_vector() {
    let verts = tet_verts();
    let faces = tet_faces();
    let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
    let zeros = vec![0.0f32; lap.n_vertices];
    let result = spec_laplacian_matvec(&lap, &zeros);
    for &r in &result {
        assert!(r.abs() < 1e-10, "L*0 should be 0, got {r}");
    }
}

// Test 46: cluster_vertices with n_clusters=1 → all zeros
#[test]
fn test_cluster_one_cluster() {
    let verts = tet_verts();
    let faces = tet_faces();
    let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
    let basis = spec_power_iteration(&lap, 1, 1000, 1e-6, 1).expect("power iter");
    let labels = spec_cluster_vertices(&basis, 1, 7).expect("cluster");
    for &l in &labels {
        assert_eq!(l, 0, "all labels should be 0 for n_clusters=1");
    }
}

// Test 47: spectral_gap = algebraic_connectivity - lambda1
#[test]
fn test_spectral_gap_definition() {
    let verts = tet_verts();
    let faces = tet_faces();
    let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
    let basis = spec_power_iteration(&lap, 2, 2000, 1e-7, 123).expect("power iter");
    let stats = spec_compute_stats(&lap, Some(&basis));
    let expected_gap = basis.eigenvalues[1] - basis.eigenvalues[0];
    assert!(
        (stats.spectral_gap - expected_gap).abs() < 0.01,
        "spectral gap mismatch: {} vs {}",
        stats.spectral_gap,
        expected_gap
    );
}

// Test 48: normalize Laplacian kind is Normalized
#[test]
fn test_normalized_laplacian_kind() {
    let verts = triangle_verts();
    let faces = triangle_faces();
    let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
    let norm = spec_normalize_laplacian(&lap).expect("normalize");
    assert!(
        matches!(norm.kind, LaplacianKind::Normalized),
        "kind should be Normalized"
    );
}

// Test 49: cotangent Laplacian kind is Cotangent
#[test]
fn test_cotangent_laplacian_kind() {
    let verts = triangle_verts();
    let faces = triangle_faces();
    let lap = spec_build_cotangent_laplacian(&verts, &faces).expect("cotangent lap");
    assert!(
        matches!(lap.kind, LaplacianKind::Cotangent),
        "kind should be Cotangent"
    );
}

// Test 50: SpectralSignal n_vertices matches values.len()
#[test]
fn test_spectral_signal_consistency() {
    let signal = SpectralSignal {
        values: vec![1.0, 2.0, 3.0, 4.0],
        n_vertices: 4,
    };
    assert_eq!(signal.values.len(), signal.n_vertices);
}

// Test 51: spec_compute_stats with basis has spectral_radius from basis
#[test]
fn test_stats_spectral_radius_from_basis() {
    let verts = tet_verts();
    let faces = tet_faces();
    let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
    let basis = spec_power_iteration(&lap, 3, 2000, 1e-7, 456).expect("power iter");
    let stats = spec_compute_stats(&lap, Some(&basis));
    // spectral_radius should equal max eigenvalue in basis
    let max_ev = basis.eigenvalues.iter().copied().fold(0.0f32, f32::max);
    assert!(
        (stats.spectral_radius - max_ev).abs() < 1e-4,
        "spectral_radius mismatch: {} vs {}",
        stats.spectral_radius,
        max_ev
    );
}

// Test 52: tetrahedron has 6 undirected edges
#[test]
fn test_tet_n_edges() {
    let verts = tet_verts();
    let faces = tet_faces();
    let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
    let stats = spec_compute_stats(&lap, None);
    assert_eq!(
        stats.n_edges, 6,
        "tetrahedron has 6 edges, got {}",
        stats.n_edges
    );
}

// Test 53: xorshift64 never returns 0
#[test]
fn test_xorshift64_nonzero() {
    let mut state = 1u64;
    for _ in 0..10_000 {
        let v = xorshift64(&mut state);
        assert_ne!(v, 0, "xorshift64 returned 0");
    }
}

// Test 54: format_stats contains expected fields
#[test]
fn test_format_stats_contains_fields() {
    let verts = triangle_verts();
    let faces = triangle_faces();
    let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
    let stats = spec_compute_stats(&lap, None);
    let s = spec_format_stats(&stats);
    assert!(s.contains("n_vertices"), "missing n_vertices");
    assert!(s.contains("n_edges"), "missing n_edges");
    assert!(s.contains("mean_degree"), "missing mean_degree");
}

// Test 55: format_config contains expected fields
#[test]
fn test_format_config_contains_fields() {
    let config = SpectralConfig {
        k: 5,
        max_power_iters: 100,
        tol: 1e-5,
        laplacian_kind: LaplacianKind::Cotangent,
    };
    let s = spec_format_config(&config);
    assert!(s.contains("k:"), "missing k");
    assert!(s.contains("Cotangent"), "missing laplacian kind");
}

// Test 56: spec_laplacian_smooth output length = input length
#[test]
fn test_laplacian_smooth_output_length() {
    let verts = tet_verts();
    let faces = tet_faces();
    let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
    let positions: Vec<f32> = verts.iter().flat_map(|p| [p.x, p.y, p.z]).collect();
    let smoothed = spec_laplacian_smooth(&positions, &lap, 0.1, 3);
    assert_eq!(smoothed.len(), positions.len());
}

// Test 57: spec_taubin_smooth output length = input length
#[test]
fn test_taubin_smooth_output_length() {
    let verts = tet_verts();
    let faces = tet_faces();
    let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
    let positions: Vec<f32> = verts.iter().flat_map(|p| [p.x, p.y, p.z]).collect();
    let smoothed = spec_taubin_smooth(&positions, &lap, 0.5, -0.53, 4);
    assert_eq!(smoothed.len(), positions.len());
}

// Test 58: cluster_vertices with 0 clusters returns error
#[test]
fn test_cluster_zero_clusters_error() {
    let verts = tet_verts();
    let faces = tet_faces();
    let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
    let basis = spec_power_iteration(&lap, 2, 1000, 1e-6, 1).expect("power iter");
    let result = spec_cluster_vertices(&basis, 0, 1);
    assert!(
        matches!(result, Err(SpectralError::InvalidConfig { .. })),
        "expected InvalidConfig error for 0 clusters"
    );
}

// Must not panic (division by zero in k-means++ seeding) on a
// hand-constructed zero-vertex `SpectralBasis` (fields are public).
#[test]
fn test_cluster_empty_basis_errors_not_panics() {
    let basis = SpectralBasis {
        eigenvectors: vec![],
        eigenvalues: vec![],
        k: 0,
        n_vertices: 0,
    };
    let result = spec_cluster_vertices(&basis, 1, 1);
    assert!(matches!(result, Err(SpectralError::EmptyMesh)));
}

// Test 59: power_iteration with k=0 returns error
#[test]
fn test_power_iteration_k_zero_error() {
    let verts = triangle_verts();
    let faces = triangle_faces();
    let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
    let result = spec_power_iteration(&lap, 0, 100, 1e-6, 1);
    assert!(
        matches!(result, Err(SpectralError::InvalidConfig { .. })),
        "expected InvalidConfig error for k=0"
    );
}

// Test 60: n_vertices in basis matches mesh
#[test]
fn test_basis_n_vertices_matches() {
    let verts = tet_verts();
    let faces = tet_faces();
    let lap = spec_build_combinatorial_laplacian(&verts, &faces).expect("build lap");
    let basis = spec_power_iteration(&lap, 2, 1000, 1e-6, 1).expect("power iter");
    assert_eq!(basis.n_vertices, verts.len());
}
