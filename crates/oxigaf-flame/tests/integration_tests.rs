//! Integration tests for oxigaf-flame.

use nalgebra as na;
use oxigaf_flame::{Camera, FlameError, FlameModel, FlameParams, Mesh, NormalMapRenderer};
use std::path::PathBuf;

/// Helper to get test data directory.
fn test_data_dir() -> PathBuf {
    // In real usage, this would point to actual FLAME model files.
    // For now, we test without real model files.
    PathBuf::from("test_data")
}

#[test]
fn test_flame_model_creation() {
    // Test that FlameModel can be constructed (requires actual .npy files)
    // Skip if test data not available
    let model_dir = test_data_dir();
    if !model_dir.exists() {
        eprintln!("Skipping test_flame_model_creation: test data not available");
        return;
    }

    match FlameModel::load(&model_dir) {
        Ok(model) => {
            assert!(model.num_vertices() > 0);
            assert_eq!(model.n_joints, 5); // FLAME has 5 joints
        }
        Err(FlameError::Io(_)) => {
            eprintln!("Skipping test: model files not found");
        }
        Err(e) => panic!("Unexpected error: {e:?}"),
    }
}

#[test]
fn test_flame_params_default() {
    let params = FlameParams::default();

    // Default should have zero or empty shape and expression
    // Pose should be initialized for 5 joints (15 values)
    assert_eq!(params.pose.len(), 15); // 5 joints * 3

    // Default should have identity global rotation (joint 0 is root)
    let [rx, ry, rz] = params.joint_pose(0);
    assert!((rx.abs() < 1e-8) && (ry.abs() < 1e-8) && (rz.abs() < 1e-8));

    // Default should have zero translation
    let [tx, ty, tz] = params.translation;
    assert!((tx.abs() < 1e-8) && (ty.abs() < 1e-8) && (tz.abs() < 1e-8));
}

#[test]
fn test_flame_params_builder() {
    let params = FlameParams {
        shape: vec![0.5, -0.3, 0.1],
        expression: vec![0.2, 0.4],
        pose: vec![0.0; 15], // 5 joints * 3 = 15
        translation: [1.0, 2.0, 3.0],
    };

    assert_eq!(params.shape.len(), 3);
    assert_eq!(params.expression.len(), 2);
    assert_eq!(params.pose.len(), 15);
    assert_eq!(params.translation, [1.0, 2.0, 3.0]);
}

#[test]
fn test_mesh_normal_computation() {
    // Create a simple triangle mesh
    let vertices = vec![
        na::Point3::new(0.0, 0.0, 0.0),
        na::Point3::new(1.0, 0.0, 0.0),
        na::Point3::new(0.0, 1.0, 0.0),
    ];
    let faces = vec![[0, 1, 2]];

    let mesh = Mesh::new(vertices, faces);

    assert_eq!(mesh.vertices.len(), 3);
    assert_eq!(mesh.faces.len(), 1);
    assert_eq!(mesh.normals.len(), 3);

    // For a triangle on XY plane, normals should point in +Z direction
    for normal in &mesh.normals {
        assert!(normal.x.abs() < 1e-5, "normal.x = {}", normal.x);
        assert!(normal.y.abs() < 1e-5, "normal.y = {}", normal.y);
        assert!((normal.z - 1.0).abs() < 1e-3, "normal.z = {}", normal.z); // Allow some tolerance
    }
}

#[test]
fn test_mesh_empty() {
    let mesh = Mesh::new(vec![], vec![]);
    assert_eq!(mesh.vertices.len(), 0);
    assert_eq!(mesh.faces.len(), 0);
    assert_eq!(mesh.normals.len(), 0);
}

#[test]
fn test_camera_construction() {
    let camera = Camera::default_front(256, 256);

    assert_eq!(camera.width, 256);
    assert_eq!(camera.height, 256);
    assert!(camera.focal_x > 0.0);
    assert!(camera.focal_y > 0.0);

    // The rotation is NOT identity.  Per the `Camera` doc comment the
    // world-to-camera convention is `p_cam = rotation · p_world + translation`
    // with `+Z_cam` forward and `+Y_cam` DOWN the screen, while FLAME meshes
    // are `+Y` up and `+Z` out of the face; `Camera::default_front` therefore
    // documents `rotation = diag(1, -1, -1)`.  Identity would put the camera
    // centre at `-Rᵀt = (0, 0, -0.6)` — behind the head, looking at the back
    // of the skull — and mirror the image vertically.
    //
    // Assert the two properties that derivation exists to guarantee, rather
    // than the matrix literal, so this test tracks the contract and not the
    // implementation.

    // 1. `+Z_cam` forward: the world origin (the centre of the head) is in
    //    front of the camera, i.e. on the visible side of the near plane.
    let origin_cam = camera.world_to_cam(&na::Point3::origin());
    assert!(
        origin_cam.z > 0.0,
        "world origin must sit in front of the default_front camera, got z = {}",
        origin_cam.z
    );

    // 2. `+Y_cam` down: world "up" must land higher on screen (a SMALLER
    //    pixel row) than world "down".  This is the assertion that actually
    //    exercises the `-1` in the Y slot — with identity rotation the image
    //    would be flipped and this comparison would reverse.
    let above = camera.project(&camera.world_to_cam(&na::Point3::new(0.0, 0.1, 0.0)));
    let below = camera.project(&camera.world_to_cam(&na::Point3::new(0.0, -0.1, 0.0)));
    assert!(
        above[1] < below[1],
        "world +Y (up) must project to a smaller pixel row than world -Y (down); \
         got v_up = {}, v_down = {}",
        above[1],
        below[1]
    );

    // 3. The camera sits at world `(0, 0, 0.6)` looking back along world `-Z`.
    //    FLAME's `+Z` is out of the face, so a point in front of the face must
    //    come out NEARER (smaller `z_cam`) than the mirrored point behind it.
    //    Assertion 1 cannot see this — the world origin has `z = 0`, so the
    //    matrix's Z entry is multiplied out of it — and this is the only
    //    check that would catch a `diag(1, -1, +1)`, which would put the
    //    camera behind the head with the depth order reversed.
    let front_of_face = camera.world_to_cam(&na::Point3::new(0.0, 0.0, 0.2)).z;
    let back_of_head = camera.world_to_cam(&na::Point3::new(0.0, 0.0, -0.2)).z;
    assert!(
        front_of_face < back_of_head,
        "the face (world +Z) must be nearer the camera than the back of the head; \
         got z_front = {front_of_face}, z_back = {back_of_head}"
    );

    // 4. The camera is centred on the image, so a point on the world Y axis
    //    stays on the vertical centre line.
    let cx = camera.cx;
    assert!(
        (above[0] - cx).abs() < 1e-3,
        "a point on the world Y axis must project to the principal x, got {}",
        above[0]
    );
}

#[test]
fn test_normal_map_renderer_static() {
    // NormalMapRenderer has static methods only
    // Verify we can import it
    let _renderer = NormalMapRenderer;
    // Just ensure the type exists
}

#[test]
fn test_normal_map_rendering() {
    // Create a simple mesh
    let vertices = vec![
        na::Point3::new(-1.0, -1.0, 0.0),
        na::Point3::new(1.0, -1.0, 0.0),
        na::Point3::new(0.0, 1.0, 0.0),
    ];
    let faces = vec![[0, 1, 2]];
    let mesh = Mesh::new(vertices, faces);

    let camera = Camera::default_front(64, 64);

    let image = NormalMapRenderer::render(&mesh, &camera);

    assert_eq!(image.width(), 64);
    assert_eq!(image.height(), 64);

    // Image should have RGB channels (normals are encoded as RGB)
    // Verify we got a valid image buffer
    assert_eq!(image.pixels().count(), 64 * 64);
}

#[test]
fn test_rodrigues_edge_cases() {
    use approx::assert_relative_eq;

    // Test zero rotation
    let r = oxigaf_flame::model::rodrigues(0.0, 0.0, 0.0);
    assert_relative_eq!(r, na::Matrix3::identity(), epsilon = 1e-6);

    // Test very small rotation
    let r = oxigaf_flame::model::rodrigues(1e-10, 0.0, 0.0);
    assert_relative_eq!(r, na::Matrix3::identity(), epsilon = 1e-6);

    // Test 180 degree rotation around X axis
    let r = oxigaf_flame::model::rodrigues(std::f32::consts::PI, 0.0, 0.0);
    let v = na::Vector3::new(0.0, 1.0, 0.0);
    let rv = r * v;
    assert_relative_eq!(rv.y, -1.0, epsilon = 1e-5);

    // Test combined rotations
    let r1 = oxigaf_flame::model::rodrigues(0.1, 0.2, 0.3);
    let r2 = oxigaf_flame::model::rodrigues(0.2, -0.1, 0.4);
    let combined = r1 * r2;

    // Combined rotation should still be orthogonal
    let should_be_identity = combined * combined.transpose();
    assert_relative_eq!(should_be_identity, na::Matrix3::identity(), epsilon = 1e-4);
}

#[test]
fn test_mesh_sampling() {
    use rand::SeedableRng;

    // Create a larger mesh
    let vertices = vec![
        na::Point3::new(0.0, 0.0, 0.0),
        na::Point3::new(1.0, 0.0, 0.0),
        na::Point3::new(1.0, 1.0, 0.0),
        na::Point3::new(0.0, 1.0, 0.0),
    ];
    let faces = vec![[0, 1, 2], [0, 2, 3]];
    let mesh = Mesh::new(vertices, faces);

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let samples = oxigaf_flame::sample_mesh_surface(&mesh, 1000, &mut rng)
        .expect("a mesh built by Mesh::new is well-formed");

    assert_eq!(samples.len(), 1000);

    // All samples should be within the unit square
    for sample in &samples {
        assert!(sample.position.x >= -1e-5 && sample.position.x <= 1.0 + 1e-5);
        assert!(sample.position.y >= -1e-5 && sample.position.y <= 1.0 + 1e-5);
        assert!(sample.position.z.abs() < 1e-5);

        // Barycentric coordinates should be valid
        for &bc in &sample.barycentric {
            assert!((-1e-5..=1.0 + 1e-5).contains(&bc));
        }
        let sum: f32 = sample.barycentric.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4);
    }
}

#[test]
fn test_mesh_sampling_empty() {
    use rand::SeedableRng;

    let mesh = Mesh::new(vec![], vec![]);
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    // A face-less mesh is well-formed, just empty: that is a successful
    // sample of zero points, not a malformed-mesh error.
    let samples = oxigaf_flame::sample_mesh_surface(&mesh, 100, &mut rng)
        .expect("an empty mesh is well-formed");

    // Should return empty for empty mesh
    assert_eq!(samples.len(), 0);
}

#[test]
fn test_mesh_sampling_deterministic() {
    use rand::SeedableRng;

    let vertices = vec![
        na::Point3::new(0.0, 0.0, 0.0),
        na::Point3::new(1.0, 0.0, 0.0),
        na::Point3::new(0.0, 1.0, 0.0),
    ];
    let faces = vec![[0, 1, 2]];
    let mesh = Mesh::new(vertices, faces);

    // Same seed should give same results
    let mut rng1 = rand::rngs::StdRng::seed_from_u64(123);
    let samples1 = oxigaf_flame::sample_mesh_surface(&mesh, 50, &mut rng1)
        .expect("a mesh built by Mesh::new is well-formed");

    let mut rng2 = rand::rngs::StdRng::seed_from_u64(123);
    let samples2 = oxigaf_flame::sample_mesh_surface(&mesh, 50, &mut rng2)
        .expect("a mesh built by Mesh::new is well-formed");

    assert_eq!(samples1.len(), samples2.len());
    for (s1, s2) in samples1.iter().zip(samples2.iter()) {
        assert!((s1.position - s2.position).norm() < 1e-6);
    }
}
