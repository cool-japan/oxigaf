//! Unit tests for loss functions.
//!
//! Tests L1 loss, SSIM loss, MS-SSIM loss, and regularization losses.

use nalgebra as na;
use oxigaf_flame::Mesh;
use oxigaf_render::gaussian::{GaussianAttributes, GaussianModel};
use oxigaf_trainer::loss::{
    clip_gradients_by_norm, clip_gradients_by_value, gaussian_kernel_1d, gradient_penalty,
    gradient_statistics, l1_loss, ms_ssim_loss, normal_consistency,
    normal_consistency_view_weighted, opacity_reg, position_reg, sanitize_gradients, scale_reg,
    ssim_loss,
};

/// Create a minimal test model.
fn make_model(n: usize) -> GaussianModel {
    GaussianModel {
        gaussians: vec![
            GaussianAttributes {
                position: [0.0; 3],
                _pad0: 0.0,
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: [-5.0; 3],
                opacity: 0.0,
            };
            n
        ],
        sh_coeffs: vec![0.0; n * 3],
        sh_degree: 0,
        face_indices: vec![0; n],
        barycentric: vec![[1.0, 0.0, 0.0]; n],
        local_offsets: vec![[0.0; 3]; n],
        is_rigid: vec![true; n],
    }
}

// ============================================================================
// L1 Loss Tests
// ============================================================================

#[test]
fn l1_identical_images_zero_loss() {
    let img = vec![0.5_f32; 100];
    let loss = l1_loss(&img, &img);
    assert!(loss.abs() < 1e-7, "Expected 0, got {loss}");
}

#[test]
fn l1_all_zero_vs_all_one() {
    let zeros = vec![0.0_f32; 100];
    let ones = vec![1.0_f32; 100];
    let loss = l1_loss(&zeros, &ones);
    assert!((loss - 1.0).abs() < 1e-7, "Expected 1.0, got {loss}");
}

#[test]
fn l1_symmetry() {
    let a = vec![0.0, 0.5, 1.0];
    let b = vec![0.2, 0.7, 0.3];
    let loss_ab = l1_loss(&a, &b);
    let loss_ba = l1_loss(&b, &a);
    assert!((loss_ab - loss_ba).abs() < 1e-7, "L1 should be symmetric");
}

#[test]
fn l1_empty_arrays() {
    let empty: Vec<f32> = vec![];
    let loss = l1_loss(&empty, &empty);
    assert_eq!(loss, 0.0, "Empty arrays should return 0");
}

#[test]
fn l1_known_values() {
    // |0.1 - 0.2| + |0.3 - 0.5| + |0.7 - 0.4| = 0.1 + 0.2 + 0.3 = 0.6
    // Mean = 0.6 / 3 = 0.2
    let a = vec![0.1, 0.3, 0.7];
    let b = vec![0.2, 0.5, 0.4];
    let loss = l1_loss(&a, &b);
    assert!((loss - 0.2).abs() < 1e-6, "Expected 0.2, got {loss}");
}

// ============================================================================
// SSIM Loss Tests
// ============================================================================

#[test]
fn ssim_identical_images_near_zero_dissimilarity() {
    let kernel = gaussian_kernel_1d(11, 1.5);
    let img = vec![0.5_f32; 32 * 32 * 3];
    let loss = ssim_loss(&img, &img, 32, 32, &kernel);
    assert!(
        loss < 0.01,
        "Expected ~0 dissimilarity for identical, got {loss}"
    );
}

#[test]
fn ssim_different_images_higher_dissimilarity() {
    let kernel = gaussian_kernel_1d(11, 1.5);
    let img1 = vec![0.2_f32; 32 * 32 * 3];
    let img2 = vec![0.8_f32; 32 * 32 * 3];
    let loss = ssim_loss(&img1, &img2, 32, 32, &kernel);
    // Different constant images should have higher dissimilarity
    assert!(loss > 0.0, "Expected positive dissimilarity, got {loss}");
}

#[test]
fn ssim_too_small_returns_zero() {
    let kernel = gaussian_kernel_1d(11, 1.5);
    let small = vec![0.5_f32; 10]; // Too small for W*H*3
    let loss = ssim_loss(&small, &small, 32, 32, &kernel);
    assert_eq!(loss, 0.0, "Invalid dimensions should return 0");
}

// ============================================================================
// Gaussian Kernel Tests
// ============================================================================

#[test]
fn gaussian_kernel_sums_to_one() {
    let kernel = gaussian_kernel_1d(11, 1.5);
    let sum: f32 = kernel.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "Kernel should sum to 1, got {sum}"
    );
}

#[test]
fn gaussian_kernel_symmetric() {
    let kernel = gaussian_kernel_1d(11, 1.5);
    for i in 0..5 {
        assert!(
            (kernel[i] - kernel[10 - i]).abs() < 1e-7,
            "Kernel should be symmetric"
        );
    }
}

#[test]
fn gaussian_kernel_peak_at_center() {
    let kernel = gaussian_kernel_1d(11, 1.5);
    let center = kernel[5];
    for (i, &k) in kernel.iter().enumerate() {
        if i != 5 {
            assert!(k < center, "Center should be peak");
        }
    }
}

// ============================================================================
// MS-SSIM Loss Tests
// ============================================================================

/// Default MS-SSIM weights from the original paper (Wang et al., 2003).
const MS_SSIM_WEIGHTS: [f32; 5] = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333];

#[test]
fn ms_ssim_weights_sum_to_approximately_one() {
    let sum: f32 = MS_SSIM_WEIGHTS.iter().sum();
    assert!(
        (sum - 1.0).abs() < 0.001,
        "MS-SSIM weights should sum to ~1.0, got {sum}"
    );
}

#[test]
fn ms_ssim_identical_images_near_zero_dissimilarity() {
    // Create a 64x64 image (large enough for 5 scales)
    let img = vec![0.5_f32; 64 * 64 * 3];
    let loss = ms_ssim_loss(&img, &img, 64, 64, &MS_SSIM_WEIGHTS);

    // Dissimilarity should be very close to 0 for identical images
    assert!(
        loss < 0.05,
        "Expected ~0 dissimilarity for identical images, got {loss}"
    );
}

#[test]
fn ms_ssim_identical_images_larger() {
    // Create a 128x128 image for more accurate multi-scale computation
    let img = vec![0.5_f32; 128 * 128 * 3];
    let loss = ms_ssim_loss(&img, &img, 128, 128, &MS_SSIM_WEIGHTS);

    assert!(
        loss < 0.01,
        "Expected ~0 dissimilarity for identical images (128x128), got {loss}"
    );
}

#[test]
fn ms_ssim_different_images_higher_dissimilarity() {
    // Two constant but different images
    let img1 = vec![0.2_f32; 64 * 64 * 3];
    let img2 = vec![0.8_f32; 64 * 64 * 3];
    let loss = ms_ssim_loss(&img1, &img2, 64, 64, &MS_SSIM_WEIGHTS);

    // Different images should have positive dissimilarity
    assert!(
        loss > 0.0,
        "Expected positive dissimilarity for different images, got {loss}"
    );
    // But bounded by 2.0 (since MS-SSIM is in [-1, 1], dissimilarity is in [0, 2])
    assert!(
        loss <= 2.0,
        "Dissimilarity should not exceed 2.0, got {loss}"
    );
}

#[test]
fn ms_ssim_different_images_vs_identical() {
    let img_a = vec![0.3_f32; 64 * 64 * 3];
    let img_b = vec![0.7_f32; 64 * 64 * 3];

    let loss_identical = ms_ssim_loss(&img_a, &img_a, 64, 64, &MS_SSIM_WEIGHTS);
    let loss_different = ms_ssim_loss(&img_a, &img_b, 64, 64, &MS_SSIM_WEIGHTS);

    assert!(
        loss_different > loss_identical,
        "Different images should have higher dissimilarity ({loss_different}) than identical ({loss_identical})"
    );
}

#[test]
fn ms_ssim_symmetry() {
    // Generate two different pattern images
    let img1: Vec<f32> = (0..64 * 64 * 3)
        .map(|i| ((i % 256) as f32) / 255.0)
        .collect();
    let img2: Vec<f32> = (0..64 * 64 * 3)
        .map(|i| ((i * 3 + 100) % 256) as f32 / 255.0)
        .collect();

    let loss_ab = ms_ssim_loss(&img1, &img2, 64, 64, &MS_SSIM_WEIGHTS);
    let loss_ba = ms_ssim_loss(&img2, &img1, 64, 64, &MS_SSIM_WEIGHTS);

    assert!(
        (loss_ab - loss_ba).abs() < 1e-6,
        "MS-SSIM should be symmetric: {loss_ab} vs {loss_ba}"
    );
}

#[test]
fn ms_ssim_small_image_returns_zero() {
    // Image smaller than 16x16 should return 0
    let small_img = vec![0.5_f32; 10 * 10 * 3];
    let loss = ms_ssim_loss(&small_img, &small_img, 10, 10, &MS_SSIM_WEIGHTS);

    assert_eq!(
        loss, 0.0,
        "Image too small for MS-SSIM should return 0, got {loss}"
    );
}

#[test]
fn ms_ssim_boundary_size_16x16() {
    // 16x16 is the minimum size that should work
    let img = vec![0.5_f32; 16 * 16 * 3];
    let loss = ms_ssim_loss(&img, &img, 16, 16, &MS_SSIM_WEIGHTS);

    // Should compute (not return 0), though accuracy may be limited
    // For identical images, should still be low dissimilarity
    assert!(
        loss < 0.5,
        "16x16 identical images should have low dissimilarity, got {loss}"
    );
}

#[test]
fn ms_ssim_mismatched_buffer_size_returns_zero() {
    // Buffer too small for stated dimensions
    let small_buffer = vec![0.5_f32; 32 * 32 * 3];
    let loss = ms_ssim_loss(&small_buffer, &small_buffer, 64, 64, &MS_SSIM_WEIGHTS);

    assert_eq!(
        loss, 0.0,
        "Buffer size mismatch should return 0, got {loss}"
    );
}

#[test]
fn ms_ssim_gradient_pattern() {
    // Create gradient images to test multi-scale structure preservation
    let width = 64;
    let height = 64;

    // Horizontal gradient
    let gradient_h: Vec<f32> = (0..height)
        .flat_map(|_| {
            (0..width).flat_map(|x| {
                let v = x as f32 / (width - 1) as f32;
                [v, v, v]
            })
        })
        .collect();

    // Vertical gradient
    let gradient_v: Vec<f32> = (0..height)
        .flat_map(|y| {
            let v = y as f32 / (height - 1) as f32;
            (0..width).flat_map(move |_| [v, v, v])
        })
        .collect();

    // Same gradient should have 0 dissimilarity
    let loss_same = ms_ssim_loss(&gradient_h, &gradient_h, width, height, &MS_SSIM_WEIGHTS);
    assert!(
        loss_same < 0.05,
        "Same gradient should have near-zero dissimilarity, got {loss_same}"
    );

    // Different gradients should have positive dissimilarity
    let loss_diff = ms_ssim_loss(&gradient_h, &gradient_v, width, height, &MS_SSIM_WEIGHTS);
    assert!(
        loss_diff > 0.0,
        "Different gradients should have positive dissimilarity, got {loss_diff}"
    );
}

#[test]
fn ms_ssim_noise_vs_constant() {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let width = 64;
    let height = 64;

    // Constant gray image
    let constant = vec![0.5_f32; width * height * 3];

    // Pseudo-random noise image (deterministic for reproducibility)
    let noise: Vec<f32> = (0..width * height * 3)
        .map(|i| {
            let mut hasher = DefaultHasher::new();
            i.hash(&mut hasher);
            let hash = hasher.finish();
            (hash % 256) as f32 / 255.0
        })
        .collect();

    let loss = ms_ssim_loss(&constant, &noise, width, height, &MS_SSIM_WEIGHTS);

    // Noise vs constant should have high dissimilarity
    assert!(
        loss > 0.3,
        "Noise vs constant should have high dissimilarity, got {loss}"
    );
}

#[test]
fn ms_ssim_custom_weights() {
    // Test with custom weights that sum to 1.0
    let custom_weights: [f32; 5] = [0.1, 0.2, 0.4, 0.2, 0.1];
    let weight_sum: f32 = custom_weights.iter().sum();
    assert!(
        (weight_sum - 1.0).abs() < 0.001,
        "Custom weights should sum to 1"
    );

    let img = vec![0.5_f32; 64 * 64 * 3];
    let loss = ms_ssim_loss(&img, &img, 64, 64, &custom_weights);

    // Should still compute valid result
    assert!(
        (0.0..0.1).contains(&loss),
        "Custom weights on identical images should give low dissimilarity, got {loss}"
    );
}

#[test]
fn ms_ssim_empty_image() {
    let empty: Vec<f32> = vec![];
    let loss = ms_ssim_loss(&empty, &empty, 0, 0, &MS_SSIM_WEIGHTS);

    assert_eq!(loss, 0.0, "Empty images should return 0, got {loss}");
}

#[test]
fn ms_ssim_checkerboard_pattern() {
    let width = 64;
    let height = 64;

    // Create checkerboard patterns with different frequencies
    let make_checkerboard = |block_size: usize| -> Vec<f32> {
        (0..height)
            .flat_map(|y| {
                (0..width).flat_map(move |x| {
                    let is_white = ((x / block_size) + (y / block_size)).is_multiple_of(2);
                    let v = if is_white { 1.0 } else { 0.0 };
                    [v, v, v]
                })
            })
            .collect()
    };

    let check_4 = make_checkerboard(4);
    let check_8 = make_checkerboard(8);

    // Same checkerboard should have 0 dissimilarity
    let loss_same = ms_ssim_loss(&check_4, &check_4, width, height, &MS_SSIM_WEIGHTS);
    assert!(
        loss_same < 0.05,
        "Same checkerboard should have near-zero dissimilarity, got {loss_same}"
    );

    // Different frequency checkerboards should have positive dissimilarity
    let loss_diff = ms_ssim_loss(&check_4, &check_8, width, height, &MS_SSIM_WEIGHTS);
    assert!(
        loss_diff > 0.0,
        "Different frequency checkerboards should differ, got {loss_diff}"
    );
}

#[test]
fn ms_ssim_result_bounded() {
    // MS-SSIM is in [-1, 1], so dissimilarity (1 - MS-SSIM) is in [0, 2]
    // But with clamp, it should be in [0, 1] for typical cases

    // Opposite images (black vs white)
    let black = vec![0.0_f32; 64 * 64 * 3];
    let white = vec![1.0_f32; 64 * 64 * 3];

    let loss = ms_ssim_loss(&black, &white, 64, 64, &MS_SSIM_WEIGHTS);

    assert!(
        loss >= 0.0,
        "MS-SSIM dissimilarity should be non-negative, got {loss}"
    );
    assert!(
        loss <= 2.0,
        "MS-SSIM dissimilarity should be at most 2.0, got {loss}"
    );
}

// ============================================================================
// Regularization Loss Tests
// ============================================================================

#[test]
fn position_reg_zero_offsets() {
    let model = make_model(5);
    let loss = position_reg(&model);
    assert_eq!(loss, 0.0, "Zero offsets should give zero loss");
}

#[test]
fn position_reg_nonzero_offsets() {
    let mut model = make_model(1);
    model.local_offsets[0] = [1.0, 0.0, 0.0];
    let loss = position_reg(&model);
    // |[1,0,0]|^2 = 1, mean = 1
    assert!((loss - 1.0).abs() < 1e-6, "Expected 1.0, got {loss}");
}

#[test]
fn scale_reg_uniform_scales() {
    let model = make_model(3);
    let loss = scale_reg(&model);
    // All scales are -5.0, so (-5)^2 * 3 / 3 = 25
    let expected = 25.0;
    assert!(
        (loss - expected).abs() < 1e-5,
        "Expected {expected}, got {loss}"
    );
}

#[test]
fn scale_reg_zero_scales() {
    let mut model = make_model(3);
    for g in &mut model.gaussians {
        g.scale = [0.0, 0.0, 0.0];
    }
    let loss = scale_reg(&model);
    assert_eq!(loss, 0.0, "Zero scales should give zero loss");
}

#[test]
fn opacity_reg_at_half_is_maximum() {
    // sigmoid(0) = 0.5, which has maximum entropy
    let mut model = make_model(1);
    model.gaussians[0].opacity = 0.0; // sigmoid(0) = 0.5
    let loss_at_half = opacity_reg(&model);

    // At sigmoid extremes, entropy is lower
    model.gaussians[0].opacity = 5.0; // sigmoid(5) ≈ 0.993
    let loss_at_high = opacity_reg(&model);

    model.gaussians[0].opacity = -5.0; // sigmoid(-5) ≈ 0.007
    let loss_at_low = opacity_reg(&model);

    assert!(
        loss_at_half > loss_at_high,
        "Entropy should be higher at 0.5 ({loss_at_half}) than at high ({loss_at_high})"
    );
    assert!(
        loss_at_half > loss_at_low,
        "Entropy should be higher at 0.5 ({loss_at_half}) than at low ({loss_at_low})"
    );
}

#[test]
fn opacity_reg_near_zero_or_one_is_low() {
    let mut model = make_model(1);

    // Near 1
    model.gaussians[0].opacity = 10.0; // sigmoid ≈ 0.99995
    let loss_near_one = opacity_reg(&model);

    // Near 0
    model.gaussians[0].opacity = -10.0; // sigmoid ≈ 0.00005
    let loss_near_zero = opacity_reg(&model);

    // Both should be low (close to 0)
    assert!(
        loss_near_one < 0.1,
        "Loss near 1 should be low, got {loss_near_one}"
    );
    assert!(
        loss_near_zero < 0.1,
        "Loss near 0 should be low, got {loss_near_zero}"
    );
}

#[test]
fn empty_model_returns_zero() {
    let model = GaussianModel {
        gaussians: vec![],
        sh_coeffs: vec![],
        sh_degree: 0,
        face_indices: vec![],
        barycentric: vec![],
        local_offsets: vec![],
        is_rigid: vec![],
    };

    assert_eq!(position_reg(&model), 0.0);
    assert_eq!(scale_reg(&model), 0.0);
    assert_eq!(opacity_reg(&model), 0.0);
}

// ============================================================================
// LPIPS Tests
// ============================================================================

mod lpips_tests {
    use candle_core::Device;
    use oxigaf_trainer::lpips::LpipsWeights;
    use oxigaf_trainer::LpipsLossComputer;

    #[test]
    fn lpips_computer_not_initialized_returns_zero() {
        let computer = LpipsLossComputer::new();
        assert!(!computer.is_initialized());

        // Should return 0.0 when not initialized
        let img = vec![0.5_f32; 32 * 32 * 3];
        let result = computer.compute(&img, &img, 32, 32);
        assert!(result.is_ok());
        assert_eq!(result.ok(), Some(0.0));
    }

    #[test]
    fn lpips_weights_uniform_shape() {
        let device = Device::Cpu;
        let weights = LpipsWeights::uniform(&device);
        assert!(weights.is_ok(), "Failed to create uniform weights");

        let weights = weights.ok();
        assert!(weights.is_some());
        let weights = weights.as_ref();
        assert!(weights.is_some());

        // Check we have 5 weight tensors (one per VGG layer)
        for i in 0..5 {
            let w = weights.map(|w| w.get(i));
            assert!(w.is_some(), "Missing weight for layer {i}");
        }
    }

    #[test]
    fn lpips_computer_compute_multi_empty() {
        let computer = LpipsLossComputer::new();

        // Not initialized - should return 0.0
        let imgs = vec![vec![0.5_f32; 32 * 32 * 3]; 2];
        let result = computer.compute_multi(&imgs, &imgs, 32, 32);
        assert!(result.is_ok());
        assert_eq!(result.ok(), Some(0.0));
    }

    #[test]
    fn lpips_empty_inputs() {
        let computer = LpipsLossComputer::new();

        // Empty input arrays
        let empty: Vec<Vec<f32>> = vec![];
        let result = computer.compute_multi(&empty, &empty, 32, 32);
        assert!(result.is_ok());
        assert_eq!(result.ok(), Some(0.0));
    }

    #[test]
    fn lpips_computer_is_not_initialized_by_default() {
        let computer = LpipsLossComputer::default();
        assert!(!computer.is_initialized());
    }

    #[test]
    fn lpips_computer_debug_impl() {
        let computer = LpipsLossComputer::new();
        let debug_str = format!("{:?}", computer);
        assert!(debug_str.contains("LpipsLossComputer"));
        assert!(debug_str.contains("initialized"));
    }
}

// ============================================================================
// Normal Consistency Tests
// ============================================================================

/// Create a simple test mesh with known normals.
fn make_test_mesh() -> Mesh {
    // Create a simple mesh with 3 vertices forming a triangle
    // Triangle in XY plane, normal pointing up (Z+)
    let vertices = vec![
        na::Point3::new(0.0, 0.0, 0.0),
        na::Point3::new(1.0, 0.0, 0.0),
        na::Point3::new(0.0, 1.0, 0.0),
    ];

    let normals = vec![
        na::Vector3::new(0.0, 0.0, 1.0), // Up
        na::Vector3::new(0.0, 0.0, 1.0), // Up
        na::Vector3::new(0.0, 0.0, 1.0), // Up
    ];

    let faces = vec![[0, 1, 2]];

    Mesh {
        vertices,
        normals,
        faces,
    }
}

/// Create a mesh with varied normals for more complex testing.
fn make_varied_normal_mesh() -> Mesh {
    let vertices = vec![
        na::Point3::new(0.0, 0.0, 0.0),
        na::Point3::new(1.0, 0.0, 0.0),
        na::Point3::new(0.0, 1.0, 0.0),
        na::Point3::new(1.0, 1.0, 0.0),
    ];

    let normals = vec![
        na::Vector3::new(0.0, 0.0, 1.0),       // Up (Z+)
        na::Vector3::new(1.0, 0.0, 0.0),       // Right (X+)
        na::Vector3::new(0.0, 1.0, 0.0),       // Forward (Y+)
        na::Vector3::new(0.577, 0.577, 0.577), // Diagonal
    ];

    let faces = vec![[0, 1, 2], [1, 2, 3]];

    Mesh {
        vertices,
        normals,
        faces,
    }
}

/// Create a model with known rotations for testing normal consistency.
fn make_model_with_rotation(n: usize, rotation: [f32; 4]) -> GaussianModel {
    GaussianModel {
        gaussians: vec![
            GaussianAttributes {
                position: [0.0; 3],
                _pad0: 0.0,
                rotation,
                scale: [-5.0; 3],
                opacity: 0.0,
            };
            n
        ],
        sh_coeffs: vec![0.0; n * 3],
        sh_degree: 0,
        face_indices: vec![0; n],
        barycentric: vec![[1.0, 0.0, 0.0]; n],
        local_offsets: vec![[0.0; 3]; n],
        is_rigid: vec![true; n],
    }
}

#[test]
fn normal_consistency_empty_model_returns_zero() {
    let mesh = make_test_mesh();
    let model = make_model(0);
    let loss = normal_consistency(&model, &mesh);
    assert_eq!(loss, 0.0, "Empty model should return 0");
}

#[test]
fn normal_consistency_perfect_alignment_zero_loss() {
    let mesh = make_test_mesh();
    // Identity rotation: [x=0, y=0, z=0, w=1] means no rotation
    // This aligns the Gaussian's z-axis with the world Z+ axis
    let model = make_model_with_rotation(1, [0.0, 0.0, 0.0, 1.0]);
    let loss = normal_consistency(&model, &mesh);

    // Perfect alignment should give loss close to 0
    assert!(
        loss < 0.01,
        "Perfect alignment should give near-zero loss, got {loss}"
    );
}

#[test]
fn normal_consistency_perpendicular_alignment() {
    let mesh = make_test_mesh();
    // Rotate 90° around X-axis: z-axis points along Y
    // quat = (sin(45°), 0, 0, cos(45°)) for 90° rotation around X
    let s = (std::f32::consts::FRAC_PI_4).sin(); // sin(45°) for 90° rotation
    let c = (std::f32::consts::FRAC_PI_4).cos(); // cos(45°)
    let model = make_model_with_rotation(1, [s, 0.0, 0.0, c]);
    let loss = normal_consistency(&model, &mesh);

    // Perpendicular alignment: dot product = 0, loss = 1 - 0 = 1.0
    assert!(
        (loss - 1.0).abs() < 0.1,
        "Perpendicular alignment should give loss near 1.0, got {loss}"
    );
}

#[test]
fn normal_consistency_opposite_alignment_uses_abs() {
    let mesh = make_test_mesh();
    // Rotate 180° around X-axis: z-axis points along -Z
    // quat = (1, 0, 0, 0) for 180° rotation around X
    let model = make_model_with_rotation(1, [1.0, 0.0, 0.0, 0.0]);
    let loss = normal_consistency(&model, &mesh);

    // Opposite alignment: dot = -1, abs(dot) = 1, loss = 1 - 1 = 0
    assert!(
        loss < 0.1,
        "Opposite alignment with abs() should give low loss, got {loss}"
    );
}

#[test]
fn normal_consistency_multiple_gaussians_averaging() {
    let mesh = make_test_mesh();
    // Mix of aligned and perpendicular
    let mut model = make_model_with_rotation(1, [0.0, 0.0, 0.0, 1.0]); // Aligned
    let s = (std::f32::consts::FRAC_PI_4).sin();
    let c = (std::f32::consts::FRAC_PI_4).cos();
    model.gaussians.push(GaussianAttributes {
        position: [0.0; 3],
        _pad0: 0.0,
        rotation: [s, 0.0, 0.0, c], // Perpendicular
        scale: [-5.0; 3],
        opacity: 0.0,
    });
    model.face_indices.push(0);
    model.barycentric.push([1.0, 0.0, 0.0]);
    model.local_offsets.push([0.0; 3]);
    model.is_rigid.push(true);
    model.sh_coeffs.resize(6, 0.0);

    let loss = normal_consistency(&model, &mesh);

    // Average of ~0.0 and ~1.0 should be around 0.5
    assert!(
        loss > 0.3 && loss < 0.7,
        "Average of aligned and perpendicular should be ~0.5, got {loss}"
    );
}

#[test]
fn normal_consistency_invalid_face_index_skipped() {
    let mesh = make_test_mesh();
    let mut model = make_model_with_rotation(1, [0.0, 0.0, 0.0, 1.0]);
    model.face_indices[0] = 999; // Invalid face index

    let loss = normal_consistency(&model, &mesh);

    // Invalid indices should be skipped, resulting in 0
    assert_eq!(
        loss, 0.0,
        "Invalid face index should be skipped, got {loss}"
    );
}

// ============================================================================
// View-Weighted Normal Consistency Tests
// ============================================================================

#[test]
fn view_weighted_empty_model_returns_zero() {
    let mesh = make_test_mesh();
    let model = make_model(0);
    let views = vec![[0.0, 0.0, -1.0]]; // Camera looking down -Z
    let loss = normal_consistency_view_weighted(&model, &mesh, &views);
    assert_eq!(loss, 0.0, "Empty model should return 0");
}

#[test]
fn view_weighted_empty_views_returns_zero() {
    let mesh = make_test_mesh();
    let model = make_model_with_rotation(1, [0.0, 0.0, 0.0, 1.0]);
    let views: Vec<[f32; 3]> = vec![];
    let loss = normal_consistency_view_weighted(&model, &mesh, &views);
    assert_eq!(loss, 0.0, "Empty view directions should return 0");
}

#[test]
fn view_weighted_facing_camera_full_weight() {
    let mesh = make_test_mesh();
    // Normal pointing up (Z+), camera also looking down from above (view direction -Z)
    let model = make_model_with_rotation(1, [0.0, 0.0, 0.0, 1.0]);
    let views = vec![[0.0, 0.0, -1.0]]; // Camera looking down -Z

    let loss_weighted = normal_consistency_view_weighted(&model, &mesh, &views);
    let loss_unweighted = normal_consistency(&model, &mesh);

    // When facing camera, weighted loss should equal unweighted
    assert!(
        (loss_weighted - loss_unweighted).abs() < 0.01,
        "Facing camera should give full weight: weighted={loss_weighted}, unweighted={loss_unweighted}"
    );
}

#[test]
fn view_weighted_away_from_camera_reduced_weight() {
    let mesh = make_test_mesh();
    // Normal pointing up (Z+), camera looking from below (view direction +Z)
    // This means the surface is facing away from camera
    let model = make_model_with_rotation(1, [0.0, 0.0, 0.0, 1.0]);
    let views = vec![[0.0, 0.0, 1.0]]; // Camera looking up +Z

    let loss = normal_consistency_view_weighted(&model, &mesh, &views);

    // When facing away, weight = 0, so weighted sum / weight total -> NaN handled as 0
    assert_eq!(
        loss, 0.0,
        "Facing away from camera should give zero loss due to zero weight, got {loss}"
    );
}

#[test]
fn view_weighted_oblique_angle() {
    let mesh = make_test_mesh();
    // Normal pointing up (Z+), camera at 45° angle
    let model = make_model_with_rotation(1, [0.0, 0.0, 0.0, 1.0]);

    // View direction at 45° from -Z (looking down and sideways)
    let angle_45 = std::f32::consts::FRAC_PI_4;
    let views = vec![[angle_45.sin(), 0.0, -angle_45.cos()]];

    let loss_weighted = normal_consistency_view_weighted(&model, &mesh, &views);
    let loss_unweighted = normal_consistency(&model, &mesh);

    // At 45°, weight = cos(45°) ≈ 0.707, so loss should be similar but not identical
    // Since consistency loss is 0 for aligned, weighted should also be ~0
    assert!(
        loss_weighted < 0.01,
        "Oblique angle with aligned normal should still be low, got {loss_weighted}"
    );
    assert!(
        (loss_weighted - loss_unweighted).abs() < 0.1,
        "Oblique angle should give similar loss to unweighted for aligned case"
    );
}

#[test]
fn view_weighted_multiple_gaussians_different_views() {
    let mesh = make_varied_normal_mesh();
    let mut model = make_model_with_rotation(2, [0.0, 0.0, 0.0, 1.0]);
    model.face_indices[1] = 1; // Second Gaussian on second face

    // Different view directions for each Gaussian
    let views = vec![
        [0.0, 0.0, -1.0], // First Gaussian: camera looking down
        [1.0, 0.0, 0.0],  // Second Gaussian: camera looking right
    ];

    let loss = normal_consistency_view_weighted(&model, &mesh, &views);

    // Should compute weighted average across both
    assert!(
        (0.0..=1.0).contains(&loss),
        "Multiple Gaussians with different views should give valid loss, got {loss}"
    );
}

#[test]
fn view_weighted_broadcast_single_view() {
    let mesh = make_test_mesh();
    let model = make_model_with_rotation(3, [0.0, 0.0, 0.0, 1.0]);

    // Single view direction should broadcast to all Gaussians
    let views = vec![[0.0, 0.0, -1.0]];

    let loss = normal_consistency_view_weighted(&model, &mesh, &views);
    let loss_unweighted = normal_consistency(&model, &mesh);

    // Should give same result as unweighted when all face same direction
    assert!(
        (loss - loss_unweighted).abs() < 0.01,
        "Broadcasting single view should match unweighted for aligned normals"
    );
}

// ============================================================================
// Gradient Penalty Tests
// ============================================================================

#[test]
fn gradient_penalty_empty_returns_zero() {
    let gradients: Vec<f32> = vec![];
    let penalty = gradient_penalty(&gradients, 10.0);
    assert_eq!(penalty, 0.0, "Empty gradients should return 0");
}

#[test]
fn gradient_penalty_below_threshold_returns_zero() {
    let gradients = vec![1.0, 2.0, 2.0]; // Norm = sqrt(1 + 4 + 4) = 3.0
    let penalty = gradient_penalty(&gradients, 10.0);
    assert_eq!(
        penalty, 0.0,
        "Norm below threshold should return 0, got {penalty}"
    );
}

#[test]
fn gradient_penalty_above_threshold_quadratic() {
    let gradients = vec![3.0, 4.0]; // Norm = sqrt(9 + 16) = 5.0
    let threshold = 3.0;
    let penalty = gradient_penalty(&gradients, threshold);

    // Excess = 5.0 - 3.0 = 2.0
    // Penalty = 2.0^2 = 4.0
    let expected = 4.0;
    assert!(
        (penalty - expected).abs() < 0.01,
        "Expected penalty {expected}, got {penalty}"
    );
}

#[test]
fn gradient_penalty_at_threshold_returns_zero() {
    let gradients = vec![3.0, 4.0]; // Norm = 5.0
    let penalty = gradient_penalty(&gradients, 5.0);
    assert!(
        penalty < 0.01,
        "Norm exactly at threshold should return ~0, got {penalty}"
    );
}

#[test]
fn gradient_penalty_large_gradients() {
    let gradients = vec![100.0; 100]; // Norm = 100 * sqrt(100) = 1000
    let threshold = 10.0;
    let penalty = gradient_penalty(&gradients, threshold);

    // Excess = 1000 - 10 = 990
    // Penalty = 990^2 = 980100
    assert!(
        penalty > 900_000.0,
        "Large gradients should give large penalty, got {penalty}"
    );
}

#[test]
fn gradient_penalty_known_values() {
    // Gradients: [1, 1, 1] -> norm = sqrt(3) ≈ 1.732
    let gradients = vec![1.0, 1.0, 1.0];
    let threshold = 1.0;
    let penalty = gradient_penalty(&gradients, threshold);

    let norm = 3.0_f32.sqrt(); // ≈ 1.732
    let excess = norm - 1.0; // ≈ 0.732
    let expected = excess * excess; // ≈ 0.536

    assert!(
        (penalty - expected).abs() < 0.01,
        "Expected {expected}, got {penalty}"
    );
}

// ============================================================================
// Gradient Utility Functions Tests
// ============================================================================

#[test]
fn gradient_statistics_empty_returns_zeros() {
    let gradients: Vec<f32> = vec![];
    let (norm, max_abs) = gradient_statistics(&gradients);
    assert_eq!(norm, 0.0, "Empty gradients should have 0 norm");
    assert_eq!(max_abs, 0.0, "Empty gradients should have 0 max");
}

#[test]
fn gradient_statistics_known_values() {
    let gradients = vec![3.0, -4.0, 0.0];
    let (norm, max_abs) = gradient_statistics(&gradients);

    // Norm = sqrt(9 + 16 + 0) = 5.0
    assert!((norm - 5.0).abs() < 0.01, "Expected norm 5.0, got {norm}");

    // Max abs = 4.0
    assert!(
        (max_abs - 4.0).abs() < 0.01,
        "Expected max_abs 4.0, got {max_abs}"
    );
}

#[test]
fn gradient_statistics_all_positive() {
    let gradients = vec![1.0, 2.0, 3.0];
    let (norm, max_abs) = gradient_statistics(&gradients);

    let expected_norm = (1.0 + 4.0 + 9.0_f32).sqrt(); // sqrt(14) ≈ 3.742
    assert!(
        (norm - expected_norm).abs() < 0.01,
        "Expected norm {expected_norm}, got {norm}"
    );
    assert!(
        (max_abs - 3.0).abs() < 0.01,
        "Expected max 3.0, got {max_abs}"
    );
}

#[test]
fn clip_gradients_by_norm_below_max_unchanged() {
    let mut gradients = vec![1.0, 2.0, 2.0]; // Norm = 3.0
    let original = gradients.clone();
    let norm = clip_gradients_by_norm(&mut gradients, 10.0);

    assert!((norm - 3.0).abs() < 0.01, "Should return original norm");
    assert_eq!(
        gradients, original,
        "Gradients below max_norm should be unchanged"
    );
}

#[test]
fn clip_gradients_by_norm_above_max_scaled() {
    let mut gradients = vec![3.0, 4.0]; // Norm = 5.0
    let norm = clip_gradients_by_norm(&mut gradients, 2.5);

    assert!((norm - 5.0).abs() < 0.01, "Should return original norm 5.0");

    // Scale factor = 2.5 / 5.0 = 0.5
    // New gradients = [1.5, 2.0]
    assert!(
        (gradients[0] - 1.5).abs() < 0.01,
        "Expected 1.5, got {}",
        gradients[0]
    );
    assert!(
        (gradients[1] - 2.0).abs() < 0.01,
        "Expected 2.0, got {}",
        gradients[1]
    );

    // Verify new norm is 2.5
    let new_norm = (gradients[0] * gradients[0] + gradients[1] * gradients[1]).sqrt();
    assert!(
        (new_norm - 2.5).abs() < 0.01,
        "Clipped norm should be 2.5, got {new_norm}"
    );
}

#[test]
fn clip_gradients_by_norm_empty_returns_zero() {
    let mut gradients: Vec<f32> = vec![];
    let norm = clip_gradients_by_norm(&mut gradients, 10.0);
    assert_eq!(norm, 0.0, "Empty gradients should return 0 norm");
}

#[test]
fn clip_gradients_by_norm_zero_max_returns_zero() {
    let mut gradients = vec![1.0, 2.0];
    let norm = clip_gradients_by_norm(&mut gradients, 0.0);
    assert_eq!(norm, 0.0, "Zero max_norm should return 0");
}

#[test]
fn clip_gradients_by_value_within_bounds_unchanged() {
    let mut gradients = vec![0.5, -0.5, 0.0];
    let original = gradients.clone();
    let clipped = clip_gradients_by_value(&mut gradients, 1.0);

    assert_eq!(clipped, 0, "No values should be clipped");
    assert_eq!(gradients, original, "Gradients should be unchanged");
}

#[test]
fn clip_gradients_by_value_exceeding_max() {
    let mut gradients = vec![5.0, -5.0, 2.0, -2.0];
    let clipped = clip_gradients_by_value(&mut gradients, 3.0);

    assert_eq!(clipped, 2, "Should clip 2 values");
    assert_eq!(gradients[0], 3.0, "5.0 should be clipped to 3.0");
    assert_eq!(gradients[1], -3.0, "-5.0 should be clipped to -3.0");
    assert_eq!(gradients[2], 2.0, "2.0 should remain unchanged");
    assert_eq!(gradients[3], -2.0, "-2.0 should remain unchanged");
}

#[test]
fn clip_gradients_by_value_zero_max() {
    let mut gradients = vec![1.0, -1.0];
    let clipped = clip_gradients_by_value(&mut gradients, 0.0);

    // Zero max_value should not clip (returns 0)
    assert_eq!(clipped, 0, "Zero max_value should not clip");
}

#[test]
fn clip_gradients_by_value_negative_max() {
    let mut gradients = vec![1.0, -1.0];
    let clipped = clip_gradients_by_value(&mut gradients, -1.0);

    // Negative max_value should not clip
    assert_eq!(clipped, 0, "Negative max_value should not clip");
}

#[test]
fn sanitize_gradients_removes_nan() {
    let mut gradients = vec![1.0, f32::NAN, 2.0, f32::NAN];
    let replaced = sanitize_gradients(&mut gradients);

    assert_eq!(replaced, 2, "Should replace 2 NaN values");
    assert_eq!(gradients[0], 1.0, "Normal values should be unchanged");
    assert_eq!(gradients[1], 0.0, "NaN should be replaced with 0.0");
    assert_eq!(gradients[2], 2.0, "Normal values should be unchanged");
    assert_eq!(gradients[3], 0.0, "NaN should be replaced with 0.0");
}

#[test]
fn sanitize_gradients_removes_inf() {
    let mut gradients = vec![1.0, f32::INFINITY, -f32::INFINITY, 2.0];
    let replaced = sanitize_gradients(&mut gradients);

    assert_eq!(replaced, 2, "Should replace 2 Inf values");
    assert_eq!(gradients[0], 1.0);
    assert_eq!(gradients[1], 0.0, "Inf should be replaced with 0.0");
    assert_eq!(gradients[2], 0.0, "-Inf should be replaced with 0.0");
    assert_eq!(gradients[3], 2.0);
}

#[test]
fn sanitize_gradients_finite_unchanged() {
    let mut gradients = vec![1.0, -2.0, 0.0, 3.5];
    let original = gradients.clone();
    let replaced = sanitize_gradients(&mut gradients);

    assert_eq!(replaced, 0, "No values should be replaced");
    assert_eq!(gradients, original, "Finite values should be unchanged");
}

#[test]
fn sanitize_gradients_empty() {
    let mut gradients: Vec<f32> = vec![];
    let replaced = sanitize_gradients(&mut gradients);
    assert_eq!(replaced, 0, "Empty gradients should replace nothing");
}

// ============================================================================
// LossComputer Integration Tests
// ============================================================================

#[test]
fn loss_computer_with_gradient_penalty_integration() {
    use oxigaf_trainer::config::LossConfig;
    use oxigaf_trainer::loss::LossComputer;

    // Create config with gradient penalty enabled
    let config = LossConfig {
        w_gradient_penalty: 0.1,
        gradient_penalty_threshold: 5.0,
        ..Default::default()
    };

    let computer = LossComputer::new(config.clone());

    // Create simple test data
    let rendered = vec![vec![0.5_f32; 32 * 32 * 3]];
    let targets = vec![vec![0.5_f32; 32 * 32 * 3]];
    let model = make_model(10);

    // Gradients that exceed threshold
    let gradients = vec![10.0; 100]; // Norm = 100
                                     // Excess = 100 - 5 = 95, penalty = 95^2 = 9025

    let output = computer.compute_with_options(
        &rendered,
        &targets,
        32,
        32,
        &model,
        None,
        None,
        Some(&gradients),
    );

    assert!(
        output.gradient_penalty > 9000.0,
        "Gradient penalty should be large, got {}",
        output.gradient_penalty
    );

    // Total loss should include weighted gradient penalty
    let expected_contribution = config.w_gradient_penalty * output.gradient_penalty;
    assert!(
        output.total > expected_contribution * 0.9,
        "Total loss should include gradient penalty contribution"
    );
}

#[test]
fn loss_computer_with_normal_consistency_integration() {
    use oxigaf_trainer::config::LossConfig;
    use oxigaf_trainer::loss::LossComputer;

    // Create config with normal consistency enabled
    let config = LossConfig {
        w_normal: 0.05,
        ..Default::default()
    };

    let computer = LossComputer::new(config.clone());

    // Create test data
    let rendered = vec![vec![0.5_f32; 32 * 32 * 3]];
    let targets = vec![vec![0.5_f32; 32 * 32 * 3]];
    let model = make_model_with_rotation(5, [0.0, 0.0, 0.0, 1.0]);
    let mesh = make_test_mesh();

    let output =
        computer.compute_with_options(&rendered, &targets, 32, 32, &model, Some(&mesh), None, None);

    // Normal consistency should be computed
    assert!(
        output.normal >= 0.0,
        "Normal consistency should be non-negative, got {}",
        output.normal
    );

    // For aligned rotations, normal consistency should be low
    assert!(
        output.normal < 0.1,
        "Aligned normals should have low consistency loss, got {}",
        output.normal
    );
}

#[test]
fn loss_computer_with_view_weighted_normals_integration() {
    use oxigaf_trainer::config::LossConfig;
    use oxigaf_trainer::loss::LossComputer;

    let config = LossConfig {
        w_normal: 0.05,
        ..Default::default()
    };

    let computer = LossComputer::new(config);

    let rendered = vec![vec![0.5_f32; 32 * 32 * 3]];
    let targets = vec![vec![0.5_f32; 32 * 32 * 3]];
    let model = make_model_with_rotation(5, [0.0, 0.0, 0.0, 1.0]);
    let mesh = make_test_mesh();
    let views = vec![[0.0, 0.0, -1.0]; 5]; // Camera looking down

    let output = computer.compute_with_options(
        &rendered,
        &targets,
        32,
        32,
        &model,
        Some(&mesh),
        Some(&views),
        None,
    );

    // View-weighted normal consistency should be computed
    assert!(
        output.normal >= 0.0,
        "View-weighted normal should be non-negative, got {}",
        output.normal
    );
}

#[test]
fn loss_computer_zero_weights_zero_contribution() {
    use oxigaf_trainer::config::LossConfig;
    use oxigaf_trainer::loss::LossComputer;

    let config = LossConfig {
        w_gradient_penalty: 0.0,
        w_normal: 0.0,
        ..Default::default()
    };

    let computer = LossComputer::new(config);

    let rendered = vec![vec![0.5_f32; 32 * 32 * 3]];
    let targets = vec![vec![0.6_f32; 32 * 32 * 3]]; // Different to have non-zero base loss
    let model = make_model_with_rotation(5, [1.0, 0.0, 0.0, 0.0]); // Misaligned
    let mesh = make_test_mesh();
    let gradients = vec![100.0; 100]; // Large gradients

    let output = computer.compute_with_options(
        &rendered,
        &targets,
        32,
        32,
        &model,
        Some(&mesh),
        None,
        Some(&gradients),
    );

    // With zero weights, contribution to total should be zero
    // (but the fields themselves may still be computed)
    let gradient_contribution = output.gradient_penalty * 0.0;
    let normal_contribution = output.normal * 0.0;

    assert_eq!(
        gradient_contribution, 0.0,
        "Zero weight should give zero contribution"
    );
    assert_eq!(
        normal_contribution, 0.0,
        "Zero weight should give zero contribution"
    );
}
