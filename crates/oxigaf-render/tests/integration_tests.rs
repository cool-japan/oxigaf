//! Integration tests for oxigaf-render.

use oxigaf_render::gaussian::{GaussianAttributes, GaussianModel};
use oxigaf_render::{DeviceLostReason, RasterConfig, RenderError};

#[test]
fn test_raster_config_creation() {
    let config = RasterConfig {
        image_width: 512,
        image_height: 512,
        tile_size: 16,
        sh_degree: 0,
        near_plane: 0.01,
        far_plane: 100.0,
        background: [0.0, 0.0, 0.0],
        ..Default::default()
    };

    assert_eq!(config.image_width, 512);
    assert_eq!(config.image_height, 512);
    assert_eq!(config.tile_size, 16);
    assert_eq!(config.tiles_x(), 512 / 16);
    assert_eq!(config.tiles_y(), 512 / 16);
    assert_eq!(config.num_pixels(), 512 * 512);
}

#[test]
fn test_gaussian_model_empty() {
    let model = GaussianModel {
        gaussians: vec![],
        sh_coeffs: vec![],
        sh_degree: 0,
        face_indices: vec![],
        barycentric: vec![],
        local_offsets: vec![],
        is_rigid: vec![],
    };

    assert_eq!(model.len(), 0);
    assert!(model.is_empty());
}

#[test]
fn test_gaussian_model_single() {
    let gaussian = GaussianAttributes {
        position: [0.0, 0.0, 0.0],
        _pad0: 0.0,
        rotation: [1.0, 0.0, 0.0, 0.0], // Identity quaternion (w, x, y, z)
        scale: [0.0, 0.0, 0.0],         // Log-scale, so exp(0) = 1
        opacity: 0.0,                   // Sigmoid-inverse, so sigmoid(0) = 0.5
    };

    let model = GaussianModel {
        gaussians: vec![gaussian],
        sh_coeffs: vec![0.5, 0.5, 0.5], // SH degree 0 (3 coeffs)
        sh_degree: 0,
        face_indices: vec![0],
        barycentric: vec![[0.33, 0.33, 0.34]],
        local_offsets: vec![[0.0; 3]],
        is_rigid: vec![false],
    };

    assert_eq!(model.len(), 1);
    assert!(!model.is_empty());
    assert_eq!(model.gaussians.len(), 1);
    assert_eq!(model.sh_coeffs.len(), 3);
}

#[test]
fn test_gaussian_model_validation() {
    // Test that model with consistent sizes works
    let n = 100;
    let gaussian = GaussianAttributes {
        position: [0.0; 3],
        _pad0: 0.0,
        rotation: [1.0, 0.0, 0.0, 0.0],
        scale: [0.0; 3],
        opacity: 0.0,
    };

    let model = GaussianModel {
        gaussians: vec![gaussian; n],
        sh_coeffs: vec![0.5; n * 3], // SH degree 0
        sh_degree: 0,
        face_indices: vec![0; n],
        barycentric: vec![[0.33, 0.33, 0.34]; n],
        local_offsets: vec![[0.0; 3]; n],
        is_rigid: vec![false; n],
    };

    assert_eq!(model.len(), n);
    assert!(!model.is_empty());
}

#[test]
fn test_gaussian_identity_quaternion() {
    // Test that identity quaternion is [1, 0, 0, 0]  (w, x, y, z)
    let quat = [1.0, 0.0, 0.0, 0.0];
    assert_eq!(quat[0], 1.0); // w component
    assert_eq!(quat[1], 0.0); // x component
    assert_eq!(quat[2], 0.0); // y component
    assert_eq!(quat[3], 0.0); // z component
}

#[test]
fn test_config_tile_calculations() {
    let config = RasterConfig {
        image_width: 640,
        image_height: 480,
        tile_size: 16,
        sh_degree: 0,
        near_plane: 0.01,
        far_plane: 100.0,
        background: [0.0, 0.0, 0.0],
        ..Default::default()
    };

    assert_eq!(config.tiles_x(), 40); // 640 / 16
    assert_eq!(config.tiles_y(), 30); // 480 / 16
    assert_eq!(config.num_tiles(), 40 * 30);
    assert_eq!(config.num_pixels(), 640 * 480);
}

#[test]
fn test_sh_coeffs_calculation() {
    let config = RasterConfig {
        image_width: 256,
        image_height: 256,
        tile_size: 16,
        sh_degree: 0,
        near_plane: 0.01,
        far_plane: 100.0,
        background: [0.0, 0.0, 0.0],
        ..Default::default()
    };

    // SH degree 0: 1 coefficient per channel, 3 channels = 3 total
    assert_eq!(config.sh_coeffs_per_gaussian(), 3);

    let config_deg1 = RasterConfig {
        sh_degree: 1,
        ..config
    };
    // SH degree 1: (1+1)^2 = 4 coefficients per channel, 3 channels = 12 total
    assert_eq!(config_deg1.sh_coeffs_per_gaussian(), 12);

    let config_deg2 = RasterConfig {
        sh_degree: 2,
        ..config
    };
    // SH degree 2: (2+1)^2 = 9 coefficients per channel, 3 channels = 27 total
    assert_eq!(config_deg2.sh_coeffs_per_gaussian(), 27);
}

#[test]
fn test_config_default() {
    let config = RasterConfig::default();

    assert_eq!(config.image_width, 512);
    assert_eq!(config.image_height, 512);
    assert_eq!(config.tile_size, 16);
    assert_eq!(config.sh_degree, 3);
    assert!(config.near_plane > 0.0);
    assert!(config.far_plane > config.near_plane);
    assert_eq!(config.background, [0.0, 0.0, 0.0]);
}

// ============================================================================
// Configuration Validation Tests
// ============================================================================

#[test]
fn test_config_non_power_of_two_dimensions() {
    // Test non-power-of-two image dimensions
    let config = RasterConfig {
        image_width: 1920,
        image_height: 1080,
        tile_size: 16,
        sh_degree: 0,
        near_plane: 0.01,
        far_plane: 100.0,
        background: [0.0, 0.0, 0.0],
        ..Default::default()
    };

    // 1920 / 16 = 120, 1080 / 16 = 67.5 -> 68 (ceil)
    assert_eq!(config.tiles_x(), 120);
    assert_eq!(config.tiles_y(), 68); // ceil(1080/16) = 68
    assert_eq!(config.num_tiles(), 120 * 68);
    assert_eq!(config.num_pixels(), 1920 * 1080);
}

#[test]
fn test_config_small_tile_size() {
    // Test with tile size = 8
    let config = RasterConfig {
        image_width: 64,
        image_height: 64,
        tile_size: 8,
        sh_degree: 0,
        near_plane: 0.1,
        far_plane: 50.0,
        background: [1.0, 1.0, 1.0],
        ..Default::default()
    };

    assert_eq!(config.tiles_x(), 8);
    assert_eq!(config.tiles_y(), 8);
    assert_eq!(config.num_tiles(), 64);
}

#[test]
fn test_config_large_tile_size() {
    // Tile size larger than image
    let config = RasterConfig {
        image_width: 8,
        image_height: 8,
        tile_size: 16,
        sh_degree: 0,
        near_plane: 0.01,
        far_plane: 100.0,
        background: [0.0, 0.0, 0.0],
        ..Default::default()
    };

    // ceil(8/16) = 1
    assert_eq!(config.tiles_x(), 1);
    assert_eq!(config.tiles_y(), 1);
    assert_eq!(config.num_tiles(), 1);
}

#[test]
fn test_config_sh_degree_max() {
    // SH degree 3 is maximum
    let config = RasterConfig {
        sh_degree: 3,
        ..RasterConfig::default()
    };

    // SH degree 3: (3+1)^2 = 16 coefficients per channel, 3 channels = 48 total
    assert_eq!(config.sh_coeffs_per_gaussian(), 48);
}

#[test]
fn test_config_background_colors() {
    let config = RasterConfig {
        background: [0.5, 0.3, 0.7],
        ..RasterConfig::default()
    };

    assert!((config.background[0] - 0.5).abs() < 1e-6);
    assert!((config.background[1] - 0.3).abs() < 1e-6);
    assert!((config.background[2] - 0.7).abs() < 1e-6);
}

#[test]
fn test_config_near_far_plane() {
    let config = RasterConfig {
        near_plane: 0.001,
        far_plane: 1000.0,
        ..RasterConfig::default()
    };

    assert!((config.near_plane - 0.001).abs() < 1e-9);
    assert!((config.far_plane - 1000.0).abs() < 1e-6);
    assert!(config.far_plane > config.near_plane);
}

// ============================================================================
// Error Type Tests
// ============================================================================

#[test]
fn test_device_lost_reason_display() {
    assert_eq!(DeviceLostReason::Destroyed.to_string(), "device destroyed");
    assert_eq!(DeviceLostReason::Unknown.to_string(), "unknown reason");
    assert_eq!(
        DeviceLostReason::DeviceDisconnected.to_string(),
        "device disconnected"
    );
    assert_eq!(DeviceLostReason::DriverUpdate.to_string(), "driver update");
    assert_eq!(DeviceLostReason::OutOfMemory.to_string(), "out of memory");
}

#[test]
fn test_render_error_gpu_init() {
    let err = RenderError::GpuInit("test error".to_string());
    assert!(err.to_string().contains("GPU initialization error"));
    assert!(err.to_string().contains("test error"));
}

#[test]
fn test_render_error_adapter_not_found() {
    let err = RenderError::AdapterNotFound;
    assert!(err.to_string().contains("No suitable GPU adapter found"));
}

#[test]
fn test_render_error_device_creation_failed() {
    let err = RenderError::DeviceCreationFailed("creation failed".to_string());
    assert!(err.to_string().contains("Failed to create GPU device"));
}

#[test]
fn test_render_error_device_lost() {
    let err = RenderError::DeviceLost {
        reason: DeviceLostReason::OutOfMemory,
        message: "OOM during render".to_string(),
    };
    let msg = err.to_string();
    assert!(msg.contains("GPU device lost"));
    assert!(msg.contains("out of memory"));
    assert!(msg.contains("OOM during render"));
}

#[test]
fn test_render_error_shader_compilation() {
    let err = RenderError::ShaderCompilation {
        shader_name: "preprocess.wgsl".to_string(),
        error: "syntax error".to_string(),
    };
    let msg = err.to_string();
    assert!(msg.contains("Shader compilation failed"));
    assert!(msg.contains("preprocess.wgsl"));
    assert!(msg.contains("syntax error"));
}

#[test]
fn test_render_error_buffer_allocation() {
    let err = RenderError::BufferAllocation {
        buffer_name: "sort_keys".to_string(),
        requested_size: 1024 * 1024 * 100,
    };
    let msg = err.to_string();
    assert!(msg.contains("Buffer allocation failed"));
    assert!(msg.contains("sort_keys"));
}

#[test]
fn test_render_error_buffer_overflow() {
    let err = RenderError::BufferOverflow {
        buffer_name: "tile_pairs".to_string(),
        max_size: 1000,
        requested: 2000,
    };
    let msg = err.to_string();
    assert!(msg.contains("Buffer overflow"));
    assert!(msg.contains("max: 1000"));
    assert!(msg.contains("requested: 2000"));
}

#[test]
fn test_render_error_too_many_gaussians() {
    let err = RenderError::TooManyGaussians {
        count: 2_000_000,
        max: 1_000_000,
    };
    let msg = err.to_string();
    assert!(msg.contains("Too many Gaussians"));
    assert!(msg.contains("2000000"));
    assert!(msg.contains("1000000"));
}

#[test]
fn test_render_error_invalid_quaternion() {
    let err = RenderError::InvalidQuaternion {
        index: 42,
        norm: 0.0,
    };
    let msg = err.to_string();
    assert!(msg.contains("Invalid quaternion"));
    assert!(msg.contains("index 42"));
    assert!(msg.contains("norm = 0"));
}

#[test]
fn test_render_error_mismatched_buffer_sizes() {
    let err = RenderError::MismatchedBufferSizes {
        expected: 1000,
        actual: 500,
    };
    let msg = err.to_string();
    assert!(msg.contains("Mismatched buffer sizes"));
    assert!(msg.contains("expected 1000"));
    assert!(msg.contains("got 500"));
}

#[test]
fn test_render_error_from_recv_error() {
    let (tx, rx) = std::sync::mpsc::channel::<i32>();
    drop(tx); // Close sender
    let recv_err = rx.recv().unwrap_err();
    let render_err: RenderError = recv_err.into();
    assert!(matches!(render_err, RenderError::ChannelRecvError(_)));
}

// ============================================================================
// Gaussian Model Tests
// ============================================================================

#[test]
fn test_gaussian_attributes_layout() {
    // Verify GaussianAttributes is correctly padded for GPU
    let gaussian = GaussianAttributes {
        position: [1.0, 2.0, 3.0],
        _pad0: 0.0,
        rotation: [0.0, 0.0, 0.0, 1.0],
        scale: [-1.0, -1.0, -1.0], // log-scale
        opacity: 2.0,              // sigmoid-inverse
    };

    assert_eq!(gaussian.position[0], 1.0);
    assert_eq!(gaussian.position[1], 2.0);
    assert_eq!(gaussian.position[2], 3.0);
    assert_eq!(gaussian.rotation[3], 1.0); // w component
    assert_eq!(gaussian.scale[0], -1.0);
    assert_eq!(gaussian.opacity, 2.0);
}

#[test]
fn test_gaussian_model_large() {
    // Test with a larger number of Gaussians
    let n = 10000;
    let gaussian = GaussianAttributes {
        position: [0.0; 3],
        _pad0: 0.0,
        rotation: [0.0, 0.0, 0.0, 1.0],
        scale: [0.0; 3],
        opacity: 0.0,
    };

    let model = GaussianModel {
        gaussians: vec![gaussian; n],
        sh_coeffs: vec![0.5; n * 3], // SH degree 0
        sh_degree: 0,
        face_indices: vec![0; n],
        barycentric: vec![[0.33, 0.33, 0.34]; n],
        local_offsets: vec![[0.0; 3]; n],
        is_rigid: vec![false; n],
    };

    assert_eq!(model.len(), n);
    assert!(!model.is_empty());
    assert_eq!(model.gaussians.len(), n);
    assert_eq!(model.sh_coeffs.len(), n * 3);
}

#[test]
fn test_gaussian_model_with_sh_degree_3() {
    // Test with SH degree 3 (maximum)
    let n = 100;
    let sh_coeffs_per_gaussian = 48; // (3+1)^2 * 3
    let gaussian = GaussianAttributes {
        position: [0.0; 3],
        _pad0: 0.0,
        rotation: [0.0, 0.0, 0.0, 1.0],
        scale: [0.0; 3],
        opacity: 0.0,
    };

    let model = GaussianModel {
        gaussians: vec![gaussian; n],
        sh_coeffs: vec![0.0; n * sh_coeffs_per_gaussian],
        sh_degree: 3,
        face_indices: vec![0; n],
        barycentric: vec![[0.33, 0.33, 0.34]; n],
        local_offsets: vec![[0.0; 3]; n],
        is_rigid: vec![true; n],
    };

    assert_eq!(model.len(), n);
    assert_eq!(model.sh_degree, 3);
    assert_eq!(model.sh_coeffs.len(), n * sh_coeffs_per_gaussian);
}

#[test]
fn test_gaussian_model_mixed_rigid_flexible() {
    let n = 10;
    let gaussian = GaussianAttributes {
        position: [0.0; 3],
        _pad0: 0.0,
        rotation: [0.0, 0.0, 0.0, 1.0],
        scale: [0.0; 3],
        opacity: 0.0,
    };

    let is_rigid: Vec<bool> = (0..n).map(|i| i % 2 == 0).collect();

    let model = GaussianModel {
        gaussians: vec![gaussian; n],
        sh_coeffs: vec![0.0; n * 3],
        sh_degree: 0,
        face_indices: vec![0; n],
        barycentric: vec![[0.33, 0.33, 0.34]; n],
        local_offsets: vec![[0.0; 3]; n],
        is_rigid,
    };

    assert!(model.is_rigid[0]); // even -> rigid
    assert!(!model.is_rigid[1]); // odd -> flexible
    assert!(model.is_rigid[2]); // even -> rigid
}

// ============================================================================
// Radix Sort Conceptual Tests (CPU verification of sort logic)
// ============================================================================

#[test]
fn test_radix_sort_concept_empty() {
    let keys: Vec<u64> = vec![];
    let values: Vec<u32> = vec![];

    assert!(keys.is_empty());
    assert!(values.is_empty());
}

#[test]
fn test_radix_sort_concept_single() {
    let keys = [42u64];
    let values = [0u32];

    assert_eq!(keys.len(), 1);
    assert_eq!(values.len(), 1);
    assert_eq!(keys[0], 42);
    assert_eq!(values[0], 0);
}

#[test]
fn test_radix_sort_concept_already_sorted() {
    let keys = [1u64, 2, 3, 4, 5];
    let _values: Vec<u32> = (0..5).collect();

    // Verify already sorted
    for i in 1..keys.len() {
        assert!(keys[i] >= keys[i - 1]);
    }
}

#[test]
fn test_radix_sort_concept_reverse_sorted() {
    let keys = vec![5u64, 4, 3, 2, 1];
    let values: Vec<u32> = (0..5).collect();

    // After sorting, should be [1, 2, 3, 4, 5]
    let mut sorted: Vec<(u64, u32)> = keys.into_iter().zip(values).collect();
    sorted.sort_by_key(|&(k, _)| k);

    assert_eq!(sorted[0].0, 1);
    assert_eq!(sorted[4].0, 5);
}

#[test]
fn test_radix_sort_concept_duplicates() {
    let keys = vec![3u64, 1, 3, 2, 1];
    let values: Vec<u32> = (0..5).collect();

    let mut sorted: Vec<(u64, u32)> = keys.into_iter().zip(values).collect();
    sorted.sort_by_key(|&(k, _)| k);

    // Keys should be sorted
    for i in 1..sorted.len() {
        assert!(sorted[i].0 >= sorted[i - 1].0);
    }
}

#[test]
fn test_radix_sort_concept_tile_depth_key() {
    // Test the tile_id << 32 | depth_bits key format
    let tile_id_0: u64 = 100; // tile 0, depth 100
    let tile_id_1: u64 = 200; // tile 0, depth 200
    let tile_id_2: u64 = 1 << 32 | 50; // tile 1, depth 50

    // Sorting by key should group by tile first, then by depth
    let mut keys = [tile_id_1, tile_id_2, tile_id_0];
    keys.sort();

    // tile 0, depth 100 should come before tile 0, depth 200
    // tile 1 should come after both
    assert_eq!(keys[0], tile_id_0); // tile 0, depth 100
    assert_eq!(keys[1], tile_id_1); // tile 0, depth 200
    assert_eq!(keys[2], tile_id_2); // tile 1, depth 50
}

// ============================================================================
// Buffer Size Calculation Tests
// ============================================================================

#[test]
fn test_buffer_size_calculations() {
    let n: u32 = 1000;
    let config = RasterConfig::default();

    // Intermediate buffer sizes
    let cov2d_size = n as u64 * 3 * 4; // [f32; 3] per Gaussian
    let means2d_size = n as u64 * 2 * 4; // [f32; 2] per Gaussian
    let depths_size = n as u64 * 4; // f32 per Gaussian
    let radii_size = n as u64 * 4; // i32 per Gaussian

    assert_eq!(cov2d_size, 12000);
    assert_eq!(means2d_size, 8000);
    assert_eq!(depths_size, 4000);
    assert_eq!(radii_size, 4000);

    // Output buffer sizes
    let npx = config.num_pixels() as u64;
    let color_size = npx * 4 * 4; // RGBA f32
    let depth_output_size = npx * 4; // f32

    assert_eq!(color_size, 512 * 512 * 16);
    assert_eq!(depth_output_size, 512 * 512 * 4);
}

#[test]
fn test_max_pairs_heuristic() {
    let n: u32 = 10000;
    let avg_tiles_per_gaussian: u32 = 4;
    let max_pairs = n.saturating_mul(avg_tiles_per_gaussian).max(1024);

    assert_eq!(max_pairs, 40000);

    // Test with small n
    let n_small: u32 = 100;
    let max_pairs_small = n_small.saturating_mul(avg_tiles_per_gaussian).max(1024);
    assert_eq!(max_pairs_small, 1024); // min 1024
}

#[test]
fn test_workgroup_calculations() {
    let n: u32 = 10000;
    let wg_size: u32 = 256;

    let num_wg = n.div_ceil(wg_size);
    assert_eq!(num_wg, 40); // ceil(10000/256) = 40

    // Test exact multiple
    let n_exact: u32 = 512;
    let num_wg_exact = n_exact.div_ceil(wg_size);
    assert_eq!(num_wg_exact, 2);

    // Test not exact
    let n_not_exact: u32 = 513;
    let num_wg_not_exact = n_not_exact.div_ceil(wg_size);
    assert_eq!(num_wg_not_exact, 3); // ceil(513/256) = 3
}

// ============================================================================
// Config Serialization Tests
// ============================================================================

#[test]
fn test_config_serde_roundtrip() {
    let config = RasterConfig {
        image_width: 1024,
        image_height: 768,
        tile_size: 16,
        sh_degree: 2,
        near_plane: 0.05,
        far_plane: 200.0,
        background: [0.1, 0.2, 0.3],
        ..Default::default()
    };

    // Serialize to JSON
    let json = serde_json::to_string(&config).expect("Failed to serialize");

    // Deserialize back
    let deserialized: RasterConfig = serde_json::from_str(&json).expect("Failed to deserialize");

    assert_eq!(deserialized.image_width, config.image_width);
    assert_eq!(deserialized.image_height, config.image_height);
    assert_eq!(deserialized.tile_size, config.tile_size);
    assert_eq!(deserialized.sh_degree, config.sh_degree);
    assert!((deserialized.near_plane - config.near_plane).abs() < 1e-6);
    assert!((deserialized.far_plane - config.far_plane).abs() < 1e-6);
    assert_eq!(deserialized.background, config.background);
}

// ============================================================================
// Buffer Pool Tests
// ============================================================================

use oxigaf_render::pool::{BufferPool, SIZE_CLASSES};

#[test]
fn test_buffer_pool_creation() {
    let pool = BufferPool::new(512 * 1024 * 1024); // 512 MB
    let stats = pool.stats();

    assert_eq!(stats.total_allocated_bytes, 0);
    assert_eq!(stats.available_bytes, 0);
    assert_eq!(stats.available_count, 0);
    assert_eq!(stats.in_use_count, 0);
    assert_eq!(stats.total_allocations, 0);
    assert_eq!(stats.total_acquisitions, 0);
}

#[test]
fn test_buffer_pool_budget_setting() {
    let pool = BufferPool::new(100 * 1024 * 1024); // 100 MB
    let stats = pool.stats();
    assert_eq!(stats.total_allocated_bytes, 0);

    // Update budget
    pool.set_budget(200 * 1024 * 1024); // 200 MB

    // Stats should be unchanged (budget doesn't affect existing allocations)
    let stats = pool.stats();
    assert_eq!(stats.total_allocated_bytes, 0);
}

#[test]
fn test_buffer_pool_clear() {
    let pool = BufferPool::new(512 * 1024 * 1024);

    // Clear should not panic on empty pool
    pool.clear();

    let stats = pool.stats();
    assert_eq!(stats.available_count, 0);
    assert_eq!(stats.total_allocated_bytes, 0);
}

#[test]
fn test_size_classes_valid() {
    // Verify size classes are in expected order and values.
    //
    // The table was extended from 8 classes (1KB..16MB) to 10 (1KB..256MB):
    // every request above the old 16MB ceiling used to fall outside the
    // table, so the pool was inert for any full-resolution readback (a
    // 1920x1080 RGBA f32 readback is ~33MB, a 2560x1440 one ~59MB). See
    // `SIZE_CLASSES` and `BufferPoolInner::size_class_index` in
    // `crates/oxigaf-render/src/pool.rs`.
    assert_eq!(SIZE_CLASSES.len(), 10);
    assert_eq!(SIZE_CLASSES[0], 1024); // 1KB
    assert_eq!(SIZE_CLASSES[1], 4 * 1024); // 4KB
    assert_eq!(SIZE_CLASSES[2], 16 * 1024); // 16KB
    assert_eq!(SIZE_CLASSES[3], 64 * 1024); // 64KB
    assert_eq!(SIZE_CLASSES[4], 256 * 1024); // 256KB
    assert_eq!(SIZE_CLASSES[5], 1024 * 1024); // 1MB
    assert_eq!(SIZE_CLASSES[6], 4 * 1024 * 1024); // 4MB
    assert_eq!(SIZE_CLASSES[7], 16 * 1024 * 1024); // 16MB
    assert_eq!(SIZE_CLASSES[8], 64 * 1024 * 1024); // 64MB
    assert_eq!(SIZE_CLASSES[9], 256 * 1024 * 1024); // 256MB
}

#[test]
fn test_size_classes_are_strictly_increasing() {
    for i in 1..SIZE_CLASSES.len() {
        assert!(
            SIZE_CLASSES[i] > SIZE_CLASSES[i - 1],
            "Size class {} ({}) should be greater than {} ({})",
            i,
            SIZE_CLASSES[i],
            i - 1,
            SIZE_CLASSES[i - 1]
        );
    }
}

#[test]
fn test_size_classes_4x_progression() {
    // Each size class should be 4x the previous
    for i in 1..SIZE_CLASSES.len() {
        assert_eq!(
            SIZE_CLASSES[i],
            SIZE_CLASSES[i - 1] * 4,
            "Size class {} should be 4x the previous",
            i
        );
    }
}

#[test]
fn test_config_memory_budget_bytes() {
    let config = RasterConfig::new().with_max_gpu_memory_mb(256);
    assert_eq!(config.memory_budget_bytes(), 256 * 1024 * 1024);

    let config = RasterConfig::new().with_max_gpu_memory_mb(0);
    assert_eq!(config.memory_budget_bytes(), 0);

    let config = RasterConfig::new().with_max_gpu_memory_mb(1024);
    assert_eq!(config.memory_budget_bytes(), 1024 * 1024 * 1024);
}

#[test]
fn test_config_buffer_pooling_enabled() {
    // Default: pooling enabled
    let config = RasterConfig::default();
    assert!(config.enable_buffer_pooling);
    assert_eq!(config.max_gpu_memory_mb, 512);

    // Disable pooling
    let config = RasterConfig::new().with_buffer_pooling(false);
    assert!(!config.enable_buffer_pooling);

    // Enable with custom budget
    let config = RasterConfig::new()
        .with_buffer_pooling(true)
        .with_max_gpu_memory_mb(256);
    assert!(config.enable_buffer_pooling);
    assert_eq!(config.max_gpu_memory_mb, 256);
}

#[test]
fn test_pool_stats_default() {
    let stats = oxigaf_render::PoolStats::default();
    assert_eq!(stats.total_allocated_bytes, 0);
    assert_eq!(stats.available_bytes, 0);
    assert_eq!(stats.available_count, 0);
    assert_eq!(stats.in_use_count, 0);
    assert_eq!(stats.total_allocations, 0);
    assert_eq!(stats.total_acquisitions, 0);
    assert!((stats.hit_rate - 0.0).abs() < f64::EPSILON);
}

#[test]
fn test_pool_stats_debug() {
    // Verify PoolStats implements Debug correctly
    let stats = oxigaf_render::PoolStats {
        total_allocated_bytes: 1024 * 1024,
        available_bytes: 512 * 1024,
        available_count: 2,
        in_use_count: 1,
        total_allocations: 3,
        total_acquisitions: 5,
        hit_rate: 0.4,
    };

    let debug_str = format!("{:?}", stats);
    assert!(debug_str.contains("PoolStats"));
    assert!(debug_str.contains("total_allocated_bytes"));
}

// ============================================================================
// Output Flags Tests (Depth and Normals)
// ============================================================================

#[test]
fn test_config_output_flags_depth_only() {
    let config = RasterConfig::new()
        .with_depth_output(true)
        .with_normal_output(false);

    // Bit 0: depth, Bit 1: normals
    assert_eq!(config.output_flags(), 0b01);
    assert!(config.output_depth);
    assert!(!config.output_normals);
}

#[test]
fn test_config_output_flags_normals_only() {
    let config = RasterConfig::new()
        .with_depth_output(false)
        .with_normal_output(true);

    // Bit 0: depth, Bit 1: normals
    assert_eq!(config.output_flags(), 0b10);
    assert!(!config.output_depth);
    assert!(config.output_normals);
}

#[test]
fn test_config_output_flags_both() {
    let config = RasterConfig::new()
        .with_depth_output(true)
        .with_normal_output(true);

    // Both bits set
    assert_eq!(config.output_flags(), 0b11);
    assert!(config.output_depth);
    assert!(config.output_normals);
}

#[test]
fn test_config_output_flags_neither() {
    let config = RasterConfig::new()
        .with_depth_output(false)
        .with_normal_output(false);

    // No bits set
    assert_eq!(config.output_flags(), 0b00);
    assert!(!config.output_depth);
    assert!(!config.output_normals);
}

#[test]
fn test_config_default_output_flags() {
    let config = RasterConfig::default();

    // Default: depth enabled, normals disabled
    assert!(config.output_depth);
    assert!(!config.output_normals);
    assert_eq!(config.output_flags(), 0b01);
}

#[test]
fn test_config_builder_output_flags() {
    let config = RasterConfig::new()
        .with_resolution(1024, 768)
        .with_normal_output(true)
        .with_depth_output(true);

    assert_eq!(config.image_width, 1024);
    assert_eq!(config.image_height, 768);
    assert!(config.output_depth);
    assert!(config.output_normals);
    assert_eq!(config.output_flags(), 0b11);
}

#[test]
fn test_render_output_with_normals() {
    use oxigaf_render::RenderOutput;

    // Test RenderOutput with normals
    let output = RenderOutput {
        color_data: vec![0.0; 4 * 512 * 512],
        depth_data: vec![1.0; 512 * 512],
        normals: Some(vec![[0.0, 0.0, 1.0]; 512 * 512]),
        width: 512,
        height: 512,
    };

    assert_eq!(output.width, 512);
    assert_eq!(output.height, 512);
    assert_eq!(output.color_data.len(), 4 * 512 * 512);
    assert_eq!(output.depth_data.len(), 512 * 512);
    assert!(output.normals.is_some());

    if let Some(ref normals) = output.normals {
        assert_eq!(normals.len(), 512 * 512);
        // Check that default normal is [0, 0, 1]
        assert_eq!(normals[0], [0.0, 0.0, 1.0]);
    }
}

#[test]
fn test_render_output_without_normals() {
    use oxigaf_render::RenderOutput;

    // Test RenderOutput without normals
    let output = RenderOutput {
        color_data: vec![0.0; 4 * 256 * 256],
        depth_data: vec![1.0; 256 * 256],
        normals: None,
        width: 256,
        height: 256,
    };

    assert_eq!(output.width, 256);
    assert_eq!(output.height, 256);
    assert!(output.normals.is_none());
}

#[test]
fn test_normal_vector_format() {
    // Test that normal vectors are stored as [f32; 3]
    let normal: [f32; 3] = [0.0, 1.0, 0.0];
    assert_eq!(normal.len(), 3);
    assert_eq!(normal[0], 0.0);
    assert_eq!(normal[1], 1.0);
    assert_eq!(normal[2], 0.0);

    // Test normalized vector
    let len = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
    assert!((len - 1.0).abs() < 1e-6);
}

#[test]
fn test_depth_range_validation() {
    let config = RasterConfig::new().with_resolution(512, 512);

    // Verify depth values should be in [near_plane, far_plane]
    assert!(config.near_plane > 0.0);
    assert!(config.far_plane > config.near_plane);

    // Test custom near/far planes
    let config = RasterConfig {
        near_plane: 0.1,
        far_plane: 1000.0,
        ..RasterConfig::default()
    };

    assert!((config.near_plane - 0.1).abs() < 1e-6);
    assert!((config.far_plane - 1000.0).abs() < 1e-6);
}
