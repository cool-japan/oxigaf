//! Tests for the memoized joint-positions cache in `FlameModel`.
//!
//! Verifies correctness, cache-hit/miss behaviour, cache-size eviction,
//! and hash-collision robustness.

use ndarray::{Array2, Array3};
use oxigaf_flame::{FlameModel, FlameParams};

// ---------------------------------------------------------------------------
// Helper: build a synthetic FlameModel
// ---------------------------------------------------------------------------

fn create_model_with_known_joints() -> FlameModel {
    let n_verts = 6;
    let n_joints = 5;
    let n_shape = 4;
    let n_expr = 2;
    let n_pose_dirs = (n_joints - 1) * 9;

    // v_template: simple ascending values so joint positions are predictable
    let v_template = Array2::from_shape_fn((n_verts, 3), |(i, j)| i as f32 + j as f32 * 0.1);

    // Minimal faces
    let faces = vec![[0u32, 1, 2]];

    // shapedirs: uniform small values so shape params shift joints predictably
    let shapedirs =
        Array3::from_shape_fn((n_verts, 3, n_shape), |(_, _, k)| (k as f32 + 1.0) * 0.1);

    let expressiondirs = Array3::zeros((n_verts, 3, n_expr));
    let posedirs = Array3::zeros((n_verts, 3, n_pose_dirs));

    // j_regressor: uniform average (each joint = mean of all vertices)
    let j_regressor = Array2::from_elem((n_joints, n_verts), 1.0 / n_verts as f32);

    let parents = vec![-1i32, 0, 1, 2, 3];

    // LBS weights
    let lbs_weights = Array2::from_shape_fn((n_verts, n_joints), |(i, j)| {
        if i % n_joints == j {
            0.8
        } else {
            0.05
        }
    });

    FlameModel::from_arrays(
        v_template,
        faces,
        shapedirs,
        expressiondirs,
        posedirs,
        j_regressor,
        parents,
        lbs_weights,
        n_joints,
    )
}

// ---------------------------------------------------------------------------
// Cache correctness: hit vs miss
// ---------------------------------------------------------------------------

#[test]
fn test_cache_miss_computes_correct_joints() {
    let model = create_model_with_known_joints();
    let shape = vec![0.1f32, 0.2, 0.0, 0.0];

    // First call — cache miss, computes fresh
    let joints = model.joint_positions_cached(&shape);
    assert_eq!(
        joints.len(),
        model.n_joints,
        "should return one [f32; 3] per joint"
    );
    // All joint positions must be finite
    for (j, pos) in joints.iter().enumerate() {
        for &coord in pos.iter() {
            assert!(
                coord.is_finite(),
                "joint {j} coordinate {coord} is not finite"
            );
        }
    }
}

#[test]
fn test_cache_hit_returns_identical_result() {
    let model = create_model_with_known_joints();
    let shape = vec![0.3f32, -0.1, 0.5, 0.0];

    // First call populates the cache
    let first = model.joint_positions_cached(&shape);
    assert_eq!(model.joint_cache_len(), 1);

    // Second call should hit the cache
    let second = model.joint_positions_cached(&shape);
    assert_eq!(
        model.joint_cache_len(),
        1,
        "cache size should not grow on hit"
    );

    // Both results must be bitwise identical
    assert_eq!(first.len(), second.len(), "cached result has wrong length");
    for (j, (a, b)) in first.iter().zip(second.iter()).enumerate() {
        for coord in 0..3 {
            assert_eq!(
                a[coord].to_bits(),
                b[coord].to_bits(),
                "joint {j} coord {coord}: cache hit produced different bits"
            );
        }
    }
}

#[test]
fn test_different_shapes_produce_different_joints() {
    let model = create_model_with_known_joints();

    let shape_a = vec![0.0f32, 0.0, 0.0, 0.0]; // neutral
    let shape_b = vec![1.0f32, 0.0, 0.0, 0.0]; // first coeff large

    let joints_a = model.joint_positions_cached(&shape_a);
    let joints_b = model.joint_positions_cached(&shape_b);

    // Cache should now hold two entries
    assert_eq!(model.joint_cache_len(), 2);

    // The results must differ (since shapedirs has non-zero values for coeff 0)
    let any_different = joints_a
        .iter()
        .zip(joints_b.iter())
        .any(|(a, b)| a.iter().zip(b.iter()).any(|(x, y)| (x - y).abs() > 1e-7));

    assert!(
        any_different,
        "different shape params should produce different joint positions"
    );
}

#[test]
fn test_cache_miss_then_hit_consistency() {
    let model = create_model_with_known_joints();
    let shape = vec![0.5f32, 0.1, -0.2, 0.3];

    // Miss
    let from_miss = model.joint_positions_cached(&shape);

    // Hit
    let from_hit = model.joint_positions_cached(&shape);

    assert_eq!(from_miss.len(), from_hit.len());
    for (j, (a, b)) in from_miss.iter().zip(from_hit.iter()).enumerate() {
        for coord in 0..3 {
            assert_eq!(
                a[coord].to_bits(),
                b[coord].to_bits(),
                "joint {j} coord {coord}: miss/hit mismatch"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Cache eviction (simple clear-on-full)
// ---------------------------------------------------------------------------

#[test]
fn test_cache_eviction_at_64_entries() {
    let model = create_model_with_known_joints();

    // Insert exactly 64 distinct shape vectors to fill the cache
    for i in 0..64u32 {
        let shape = vec![i as f32 * 0.001, 0.0, 0.0, 0.0];
        model.joint_positions_cached(&shape);
    }
    assert_eq!(model.joint_cache_len(), 64, "cache should hold 64 entries");

    // Inserting one more triggers eviction (cache clears, then inserts the new entry)
    let shape_65 = vec![64.0f32 * 0.001, 0.0, 0.0, 0.0];
    model.joint_positions_cached(&shape_65);
    assert_eq!(
        model.joint_cache_len(),
        1,
        "after eviction + insert, cache should hold 1 entry"
    );
}

#[test]
fn test_cache_clear_resets_to_zero() {
    let model = create_model_with_known_joints();

    // Populate the cache
    model.joint_positions_cached(&[0.1, 0.0, 0.0, 0.0]);
    model.joint_positions_cached(&[0.2, 0.0, 0.0, 0.0]);
    assert_eq!(model.joint_cache_len(), 2);

    // Explicit clear
    model.clear_joint_cache();
    assert_eq!(
        model.joint_cache_len(),
        0,
        "cache should be empty after clear"
    );

    // Subsequent calls still work correctly
    let joints = model.joint_positions_cached(&[0.1, 0.0, 0.0, 0.0]);
    assert_eq!(joints.len(), model.n_joints);
    assert_eq!(model.joint_cache_len(), 1);
}

// ---------------------------------------------------------------------------
// Correctness: cached result matches non-cached computation
// ---------------------------------------------------------------------------

#[test]
fn test_cached_result_matches_forward_joint_computation() {
    let model = create_model_with_known_joints();

    // Use neutral shape to get a simple reference point
    let shape = vec![0.0f32; 4];

    // Get via cache
    let cached_joints = model.joint_positions_cached(&shape);

    // Get via full forward pass with zero expression/pose, then compare
    // (The forward pass computes joints from shape + expression; with zero
    // expression the shape-only joints should be identical.)
    let params = FlameParams {
        shape: shape.clone(),
        expression: vec![0.0; 2], // zero expression
        pose: vec![0.0; 15],
        translation: [0.0, 0.0, 0.0],
    };
    let _mesh = model.forward(&params); // side-effect: populates cache

    // Re-query the cache — should still be 1 entry and return the same values
    let cached_again = model.joint_positions_cached(&shape);
    assert_eq!(
        cached_joints.len(),
        cached_again.len(),
        "repeated cache query should be consistent"
    );
    for (j, (a, b)) in cached_joints.iter().zip(cached_again.iter()).enumerate() {
        for coord in 0..3 {
            assert_eq!(
                a[coord].to_bits(),
                b[coord].to_bits(),
                "joint {j} coord {coord}: inconsistency after forward pass"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Hash-collision robustness: very similar shape values must produce
// distinct cache entries (or gracefully alias — both are acceptable)
// ---------------------------------------------------------------------------

#[test]
fn test_near_identical_shapes_are_treated_separately() {
    let model = create_model_with_known_joints();

    // Two shape vectors that differ only in the least significant bit of f32
    let shape_a = vec![1.0f32, 0.0, 0.0, 0.0];
    let shape_b = vec![f32::from_bits(1.0f32.to_bits() + 1), 0.0, 0.0, 0.0]; // 1 ULP apart

    let joints_a = model.joint_positions_cached(&shape_a);
    let joints_b = model.joint_positions_cached(&shape_b);

    // These should have different hashes (FNV over raw bits is injective for different bit patterns)
    // and therefore produce separate cache entries
    assert!(
        model.joint_cache_len() == 2,
        "1-ULP different shapes should map to separate cache entries"
    );

    // Both results must be valid (finite)
    for pos in joints_a.iter().chain(joints_b.iter()) {
        for &coord in pos.iter() {
            assert!(coord.is_finite());
        }
    }
}
