//! Thread-safety integration tests for `FlameModel`.
//!
//! These tests verify that `FlameModel` is `Send + Sync` and can be safely
//! shared across threads via `Arc`.

use ndarray::{Array2, Array3};
use oxigaf_flame::{FlameModel, FlameParams};
use std::sync::Arc;
use std::thread;

// ---------------------------------------------------------------------------
// Test helper: minimal synthetic FlameModel (no .npy files needed)
// ---------------------------------------------------------------------------

/// Build a small but fully-valid synthetic `FlameModel` for tests.
///
/// Uses minimal dimensions (8 vertices, 5 joints) to keep tests fast.
fn create_minimal_model() -> FlameModel {
    let n_verts = 8;
    let n_joints = 5;
    let n_shape = 4;
    let n_expr = 3;
    let n_pose_dirs = (n_joints - 1) * 9; // 36

    // v_template: small grid of vertices
    let v_template =
        Array2::from_shape_fn((n_verts, 3), |(i, j)| (i as f32 * 0.1) + (j as f32 * 0.01));

    // Triangle faces within vertex count
    let faces = vec![[0u32, 1, 2], [1, 2, 3], [2, 3, 4]];

    // Blend-shape directions (small non-zero values for realism)
    let shapedirs = Array3::from_shape_fn((n_verts, 3, n_shape), |(i, j, k)| {
        ((i + j + k) as f32 * 0.001).sin()
    });
    let expressiondirs = Array3::from_shape_fn((n_verts, 3, n_expr), |(i, j, k)| {
        ((i + j + k) as f32 * 0.001).cos()
    });
    let posedirs = Array3::from_shape_fn((n_verts, 3, n_pose_dirs), |(i, j, k)| {
        ((i + j + k) as f32 * 0.0001).sin()
    });

    // Joint regressor: uniform average over vertices (sums to 1 per joint)
    let j_regressor = Array2::from_shape_fn((n_joints, n_verts), |_| 1.0 / n_verts as f32);

    // Kinematic chain: root → neck → jaw → left-eye → right-eye
    let parents = vec![-1i32, 0, 1, 2, 3];

    // LBS weights: each vertex belongs primarily to one joint
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

/// Create a set of `FlameParams` with distinct shape coefficients.
fn make_params(shape_seed: f32, expr_seed: f32) -> FlameParams {
    FlameParams {
        shape: vec![shape_seed, -shape_seed * 0.5, shape_seed * 0.25, 0.0],
        expression: vec![expr_seed, -expr_seed],
        pose: vec![0.0; 15],
        translation: [0.0, 0.0, 0.0],
    }
}

// ---------------------------------------------------------------------------
// Compile-time Send + Sync checks
// ---------------------------------------------------------------------------

/// Compile-time assertion that `FlameModel` implements `Send`.
fn assert_send<T: Send>() {}

/// Compile-time assertion that `FlameModel` implements `Sync`.
fn assert_sync<T: Sync>() {}

#[test]
fn flame_model_is_send() {
    assert_send::<FlameModel>();
}

#[test]
fn flame_model_is_sync() {
    assert_sync::<FlameModel>();
}

#[test]
fn flame_model_is_send_sync() {
    // Unified check — ensures both bounds hold simultaneously
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<FlameModel>();
}

// ---------------------------------------------------------------------------
// Runtime thread-safety tests
// ---------------------------------------------------------------------------

#[test]
fn flame_model_shared_across_threads_read_only() {
    // Wrap the model in Arc so all threads share ownership.
    let model = Arc::new(create_minimal_model());

    let handles: Vec<_> = (0..4)
        .map(|i| {
            let model_ref = Arc::clone(&model);
            thread::spawn(move || {
                let params = make_params(i as f32 * 0.1, i as f32 * 0.05);
                // Each thread calls forward() independently
                let mesh = model_ref.forward(&params);
                assert!(
                    mesh.num_vertices() > 0,
                    "thread {i}: mesh should have vertices"
                );
                assert!(mesh.num_faces() > 0, "thread {i}: mesh should have faces");
                mesh.num_vertices()
            })
        })
        .collect();

    // Join and verify all threads succeeded
    let vertex_counts: Vec<_> = handles
        .into_iter()
        .map(|h| h.join().expect("thread should not panic"))
        .collect();

    // All threads should produce the same number of vertices
    let first = vertex_counts[0];
    for (i, &count) in vertex_counts.iter().enumerate() {
        assert_eq!(
            count, first,
            "thread {i} produced {count} vertices; expected {first}"
        );
    }
}

#[test]
fn flame_model_shared_across_threads_with_cache() {
    // Specifically tests the `joint_positions_cached` path under concurrent access.
    let model = Arc::new(create_minimal_model());

    // Use a shared shape — all threads will hit the same cache entry after the
    // first insert, exercising the read-locked fast path.
    let shared_shape = vec![0.5f32, -0.3, 0.1, 0.2];

    let handles: Vec<_> = (0..8)
        .map(|_| {
            let model_ref = Arc::clone(&model);
            let shape = shared_shape.clone();
            thread::spawn(move || {
                // Call the cached method directly
                let joints = model_ref.joint_positions_cached(&shape);
                joints.len()
            })
        })
        .collect();

    let n_joints_expected = create_minimal_model().n_joints;
    for h in handles {
        let n = h.join().expect("thread should not panic");
        assert_eq!(
            n, n_joints_expected,
            "cached joint count must match n_joints"
        );
    }
}

#[test]
fn flame_model_arc_clone_across_thread_boundary() {
    // Verifies that Arc<FlameModel> can be sent to another thread and used.
    let model = Arc::new(create_minimal_model());
    let model_clone = Arc::clone(&model);

    let handle = thread::spawn(move || {
        let params = make_params(0.2, 0.1);
        let mesh = model_clone.forward(&params);
        (mesh.num_vertices(), mesh.num_faces())
    });

    let (n_verts, n_faces) = handle.join().expect("thread should not panic");
    assert!(n_verts > 0);
    assert!(n_faces > 0);
}

#[test]
fn flame_model_concurrent_batch_forward() {
    // Multiple threads each process a batch — simulates a real video pipeline.
    let model = Arc::new(create_minimal_model());

    // Each thread processes a 3-frame batch with different shape/expression
    let handles: Vec<_> = (0..4)
        .map(|thread_id| {
            let model_ref = Arc::clone(&model);
            thread::spawn(move || {
                let params_batch: Vec<FlameParams> = (0..3)
                    .map(|frame| {
                        make_params(
                            thread_id as f32 * 0.1 + frame as f32 * 0.05,
                            frame as f32 * 0.02,
                        )
                    })
                    .collect();

                let meshes = model_ref.forward_batch(&params_batch);
                assert_eq!(meshes.len(), 3, "should get 3 meshes back");
                for (frame, mesh) in meshes.iter().enumerate() {
                    assert!(
                        mesh.num_vertices() > 0,
                        "thread {thread_id} frame {frame}: expected vertices"
                    );
                }
                meshes.len()
            })
        })
        .collect();

    for h in handles {
        let count = h.join().expect("thread should not panic");
        assert_eq!(count, 3);
    }
}
