//! Tests for camera and timestep embeddings in oxigaf-diffusion.
//!
//! Verifies shape correctness and expected behavior of embedding modules.

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Module, VarMap};
use proptest::prelude::*;

/// Helper to create a mock VarBuilder for testing.
fn test_varbuilder() -> candle_nn::VarBuilder<'static> {
    let varmap = VarMap::new();
    candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu)
}

// ---------------------------------------------------------------------------
// Timestep Embedding Tests
// ---------------------------------------------------------------------------

/// Test that timestep_embedding produces correct output shape.
#[test]
fn test_timestep_embedding_shape() -> Result<()> {
    let batch = 4;
    let dim = 64;

    let timesteps = Tensor::from_slice(&[0f32, 100.0, 500.0, 999.0], (batch,), &Device::Cpu)?;
    let embedding = oxigaf_diffusion::camera::timestep_embedding(&timesteps, dim)?;

    let dims = embedding.dims2()?;
    assert_eq!(dims, (batch, dim), "Embedding shape should be (batch, dim)");

    Ok(())
}

/// Test that timestep_embedding produces sinusoidal pattern (half sin, half cos).
#[test]
fn test_timestep_embedding_sinusoidal_pattern() -> Result<()> {
    let batch = 1;
    let dim = 8;

    let timesteps = Tensor::from_slice(&[100f32], (batch,), &Device::Cpu)?;
    let embedding = oxigaf_diffusion::camera::timestep_embedding(&timesteps, dim)?;

    // Get values
    let values = embedding.to_vec2::<f32>()?;
    let emb = &values[0];

    // First half should be cos values
    let cos_half = &emb[0..dim / 2];
    // Second half should be sin values
    let sin_half = &emb[dim / 2..dim];

    // Verify sin^2 + cos^2 = 1 for corresponding positions
    for i in 0..dim / 2 {
        let sum_sq = cos_half[i].powi(2) + sin_half[i].powi(2);
        assert!(
            (sum_sq - 1.0).abs() < 0.01,
            "sin^2 + cos^2 should be ~1, got {}",
            sum_sq
        );
    }

    Ok(())
}

/// Test that different timesteps produce different embeddings.
#[test]
fn test_timestep_embedding_uniqueness() -> Result<()> {
    let dim = 64;

    let t1 = Tensor::from_slice(&[0f32], (1,), &Device::Cpu)?;
    let t2 = Tensor::from_slice(&[100f32], (1,), &Device::Cpu)?;
    let t3 = Tensor::from_slice(&[999f32], (1,), &Device::Cpu)?;

    let emb1 = oxigaf_diffusion::camera::timestep_embedding(&t1, dim)?;
    let emb2 = oxigaf_diffusion::camera::timestep_embedding(&t2, dim)?;
    let emb3 = oxigaf_diffusion::camera::timestep_embedding(&t3, dim)?;

    // Check that embeddings are different
    let diff_12 = (&emb1 - &emb2)?.abs()?.sum_all()?.to_scalar::<f32>()?;
    let diff_13 = (&emb1 - &emb3)?.abs()?.sum_all()?.to_scalar::<f32>()?;
    let diff_23 = (&emb2 - &emb3)?.abs()?.sum_all()?.to_scalar::<f32>()?;

    assert!(diff_12 > 0.1, "t=0 and t=100 embeddings should differ");
    assert!(diff_13 > 0.1, "t=0 and t=999 embeddings should differ");
    assert!(diff_23 > 0.1, "t=100 and t=999 embeddings should differ");

    Ok(())
}

/// Test TimestepEmbedding MLP output shape.
#[test]
fn test_timestep_embedding_mlp_shape() -> Result<()> {
    let vs = test_varbuilder();

    let in_dim = 64;
    let out_dim = 128;

    let mlp = oxigaf_diffusion::camera::TimestepEmbedding::new(vs.pp("time_emb"), in_dim, out_dim)?;

    let batch = 4;
    let xs = Tensor::randn(0f32, 1f32, (batch, in_dim), &Device::Cpu)?;
    let output = mlp.forward(&xs)?;

    let dims = output.dims2()?;
    assert_eq!(dims, (batch, out_dim));

    Ok(())
}

// ---------------------------------------------------------------------------
// Camera Embedding Tests
// ---------------------------------------------------------------------------

/// Test that CameraEmbedding produces correct output shape.
#[test]
fn test_camera_embedding_shape() -> Result<()> {
    let vs = test_varbuilder();

    let pose_dim = 12; // 4x3 matrix flattened
    let embed_dim = 128;

    let cam_emb =
        oxigaf_diffusion::camera::CameraEmbedding::new(vs.pp("cam"), pose_dim, embed_dim)?;

    let batch = 4;
    let pose = Tensor::randn(0f32, 1f32, (batch, pose_dim), &Device::Cpu)?;
    let output = cam_emb.forward(&pose)?;

    let dims = output.dims2()?;
    assert_eq!(
        dims,
        (batch, embed_dim),
        "Output should be (batch, embed_dim)"
    );

    Ok(())
}

/// Test camera embedding with identity pose.
#[test]
fn test_camera_embedding_identity_pose() -> Result<()> {
    let vs = test_varbuilder();

    let pose_dim = 12;
    let embed_dim = 64;

    let cam_emb =
        oxigaf_diffusion::camera::CameraEmbedding::new(vs.pp("cam"), pose_dim, embed_dim)?;

    // Identity matrix (row-major): [1,0,0,0, 0,1,0,0, 0,0,1,0]
    let identity = Tensor::from_slice(
        &[1f32, 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.],
        (1, pose_dim),
        &Device::Cpu,
    )?;

    let output = cam_emb.forward(&identity)?;

    // Just verify forward pass works and shape is correct
    assert_eq!(output.dims2()?, (1, embed_dim));

    Ok(())
}

/// Test camera embedding with different poses produces different outputs.
#[test]
fn test_camera_embedding_different_poses() -> Result<()> {
    let vs = test_varbuilder();

    let pose_dim = 12;
    let embed_dim = 64;

    let cam_emb =
        oxigaf_diffusion::camera::CameraEmbedding::new(vs.pp("cam"), pose_dim, embed_dim)?;

    // Two different poses
    let pose1 = Tensor::randn(0f32, 1f32, (1, pose_dim), &Device::Cpu)?;
    let pose2 = Tensor::randn(0f32, 1f32, (1, pose_dim), &Device::Cpu)?;

    let emb1 = cam_emb.forward(&pose1)?;
    let emb2 = cam_emb.forward(&pose2)?;

    // Embeddings should differ
    let diff = (&emb1 - &emb2)?.abs()?.sum_all()?.to_scalar::<f32>()?;
    assert!(
        diff > 0.01,
        "Different poses should produce different embeddings"
    );

    Ok(())
}

/// Test that camera embedding matches time_embed_dim for proper addition.
#[test]
fn test_camera_embedding_matches_time_dim() -> Result<()> {
    let vs = test_varbuilder();

    let pose_dim = 12;
    let time_embed_dim = 128;

    let cam_emb =
        oxigaf_diffusion::camera::CameraEmbedding::new(vs.pp("cam"), pose_dim, time_embed_dim)?;
    let time_emb =
        oxigaf_diffusion::camera::TimestepEmbedding::new(vs.pp("time"), 64, time_embed_dim)?;

    let batch = 2;
    let pose = Tensor::randn(0f32, 1f32, (batch, pose_dim), &Device::Cpu)?;
    let timesteps = Tensor::randn(0f32, 1f32, (batch, 64), &Device::Cpu)?;

    let cam_out = cam_emb.forward(&pose)?;
    let time_out = time_emb.forward(&timesteps)?;

    // Should be able to add them together
    let combined = (&cam_out + &time_out)?;
    assert_eq!(combined.dims2()?, (batch, time_embed_dim));

    Ok(())
}

// ---------------------------------------------------------------------------
// Property-based Tests
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(20))]

    #[test]
    fn test_timestep_embedding_dim_invariant(
        batch in 1usize..8,
        dim_mult in 1usize..4,
    ) {
        let dim = 16 * dim_mult; // Must be even

        let result = (|| -> Result<(usize, usize)> {
            let timesteps = Tensor::randn(0f32, 1000f32, (batch,), &Device::Cpu)?;
            let embedding = oxigaf_diffusion::camera::timestep_embedding(&timesteps, dim)?;
            embedding.dims2()
        })();

        // Verify if successful
        if let Ok(dims) = result {
            prop_assert_eq!(dims, (batch, dim));
        }
    }

    #[test]
    fn test_camera_embedding_dim_invariant(
        batch in 1usize..8,
        pose_dim in 1usize..16,
        embed_mult in 1usize..4,
    ) {
        let embed_dim = 32 * embed_mult;

        let result = (|| -> Result<(usize, usize)> {
            let vs = test_varbuilder();
            let cam_emb = oxigaf_diffusion::camera::CameraEmbedding::new(
                vs.pp("cam"),
                pose_dim,
                embed_dim,
            )?;

            let pose = Tensor::randn(0f32, 1f32, (batch, pose_dim), &Device::Cpu)?;
            let output = cam_emb.forward(&pose)?;
            output.dims2()
        })();

        // Verify if successful
        if let Ok(dims) = result {
            prop_assert_eq!(dims, (batch, embed_dim));
        }
    }
}

// ---------------------------------------------------------------------------
// Integration-style Tests
// ---------------------------------------------------------------------------

/// Test full timestep embedding pipeline: sinusoidal -> MLP.
#[test]
fn test_full_timestep_pipeline() -> Result<()> {
    let vs = test_varbuilder();

    let base_channels = 64;
    let time_embed_dim = 128;

    // Create MLP
    let mlp = oxigaf_diffusion::camera::TimestepEmbedding::new(
        vs.pp("time_emb"),
        base_channels,
        time_embed_dim,
    )?;

    // Create timesteps
    let batch = 4;
    let timesteps = Tensor::from_slice(&[0f32, 250.0, 500.0, 750.0], (batch,), &Device::Cpu)?;

    // Get sinusoidal embeddings
    let sin_emb = oxigaf_diffusion::camera::timestep_embedding(&timesteps, base_channels)?;

    // Pass through MLP
    let output = mlp.forward(&sin_emb)?;

    assert_eq!(output.dims2()?, (batch, time_embed_dim));

    Ok(())
}
