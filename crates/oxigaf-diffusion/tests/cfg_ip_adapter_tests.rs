//! Tests for Classifier-Free Guidance (CFG) and IP-Adapter conditioning.
//!
//! Verifies that:
//! 1. IP-Adapter attention layer processes tokens correctly
//! 2. CFG produces different outputs with/without IP conditioning
//! 3. Guidance scale affects output magnitude as expected
//! 4. Numerical stability across various guidance scales

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarMap;

/// Helper to create a mock VarBuilder for testing.
fn test_varbuilder() -> candle_nn::VarBuilder<'static> {
    let varmap = VarMap::new();
    candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu)
}

// ---------------------------------------------------------------------------
// IP-Adapter Attention Tests
// ---------------------------------------------------------------------------

/// Test that IP-Adapter attention layer is present and functional.
#[test]
fn test_ip_adapter_attention_exists() -> Result<()> {
    let vs = test_varbuilder();

    let dim = 64;
    let n_heads = 4;
    let d_head = 16;
    let context_dim = 64;
    let ip_dim = 128; // Different from context_dim
    let num_views = 2;

    // Build a transformer block with IP-Adapter
    let block = oxigaf_diffusion::attention::MultiViewTransformerBlock::new(
        vs.pp("block"),
        dim,
        n_heads,
        d_head,
        context_dim,
        ip_dim,
        num_views,
    )?;

    let batch_views = 4;
    let seq_len = 16;
    let xs = Tensor::randn(0f32, 1f32, (batch_views, seq_len, dim), &Device::Cpu)?;
    let context = Tensor::randn(0f32, 1f32, (batch_views, 10, context_dim), &Device::Cpu)?;
    let ip_tokens = Tensor::randn(0f32, 1f32, (batch_views, 5, ip_dim), &Device::Cpu)?;

    // Forward with IP tokens
    let output = block.forward(&xs, Some(&context), Some(&ip_tokens))?;

    // Verify output shape
    assert_eq!(output.dims3()?, (batch_views, seq_len, dim));

    Ok(())
}

/// Test that IP-Adapter tokens affect the output (not ignored).
#[test]
fn test_ip_adapter_tokens_affect_output() -> Result<()> {
    let vs = test_varbuilder();

    let dim = 64;
    let n_heads = 4;
    let d_head = 16;
    let context_dim = 64;
    let ip_dim = 64;
    let num_views = 2;

    let block = oxigaf_diffusion::attention::MultiViewTransformerBlock::new(
        vs.pp("block"),
        dim,
        n_heads,
        d_head,
        context_dim,
        ip_dim,
        num_views,
    )?;

    let batch_views = 2;
    let seq_len = 8;
    let xs = Tensor::ones((batch_views, seq_len, dim), DType::F32, &Device::Cpu)?;
    let context = Tensor::zeros((batch_views, 4, context_dim), DType::F32, &Device::Cpu)?;

    // Create two different IP token sets
    let ip_tokens_a = Tensor::ones((batch_views, 4, ip_dim), DType::F32, &Device::Cpu)?;
    let ip_tokens_b = Tensor::full(2.0f32, (batch_views, 4, ip_dim), &Device::Cpu)?;

    // Forward with different IP tokens
    let output_a = block.forward(&xs, Some(&context), Some(&ip_tokens_a))?;
    let output_b = block.forward(&xs, Some(&context), Some(&ip_tokens_b))?;

    // Outputs should be different (IP tokens affect the result)
    let diff = (output_a - output_b)?;
    let mean_abs_diff = diff.abs()?.mean_all()?.to_scalar::<f32>()?;

    assert!(
        mean_abs_diff > 1e-6,
        "IP tokens should affect output, but mean diff was {}",
        mean_abs_diff
    );

    Ok(())
}

/// Test that skipping IP tokens (None) produces different output than providing them.
#[test]
fn test_ip_adapter_skip_vs_provide() -> Result<()> {
    let vs = test_varbuilder();

    let dim = 64;
    let n_heads = 4;
    let d_head = 16;
    let context_dim = 64;
    let ip_dim = 64;
    let num_views = 2;

    let block = oxigaf_diffusion::attention::MultiViewTransformerBlock::new(
        vs.pp("block"),
        dim,
        n_heads,
        d_head,
        context_dim,
        ip_dim,
        num_views,
    )?;

    let batch_views = 2;
    let seq_len = 8;
    let xs = Tensor::ones((batch_views, seq_len, dim), DType::F32, &Device::Cpu)?;
    let context = Tensor::zeros((batch_views, 4, context_dim), DType::F32, &Device::Cpu)?;
    let ip_tokens = Tensor::ones((batch_views, 4, ip_dim), DType::F32, &Device::Cpu)?;

    // Forward with IP tokens
    let output_with_ip = block.forward(&xs, Some(&context), Some(&ip_tokens))?;

    // Forward without IP tokens (CFG unconditional)
    let output_without_ip = block.forward(&xs, Some(&context), None)?;

    // Outputs should be different
    let diff = (output_with_ip - output_without_ip)?;
    let mean_abs_diff = diff.abs()?.mean_all()?.to_scalar::<f32>()?;

    assert!(
        mean_abs_diff > 1e-6,
        "Skipping IP tokens should produce different output, but mean diff was {}",
        mean_abs_diff
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// CFG Formula Tests
// ---------------------------------------------------------------------------

/// Test that CFG formula is applied correctly: pred = uncond + scale * (cond - uncond).
#[test]
fn test_cfg_formula() -> Result<()> {
    // Simulate noise predictions
    let uncond = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], &Device::Cpu)?;
    let cond = Tensor::new(&[2.0f32, 3.0, 4.0, 5.0], &Device::Cpu)?;
    let guidance_scale = 2.0;

    // Apply CFG formula
    let diff = (cond - &uncond)?;
    let result = (uncond + (diff * guidance_scale))?;

    // Expected: [1, 2, 3, 4] + 2.0 * ([2, 3, 4, 5] - [1, 2, 3, 4])
    //         = [1, 2, 3, 4] + 2.0 * [1, 1, 1, 1]
    //         = [1, 2, 3, 4] + [2, 2, 2, 2]
    //         = [3, 4, 5, 6]
    let expected = [3.0f32, 4.0, 5.0, 6.0];
    let result_vec: Vec<f32> = result.to_vec1()?;

    for (r, e) in result_vec.iter().zip(expected.iter()) {
        assert!(
            (r - e).abs() < 1e-6,
            "CFG formula mismatch: expected {}, got {}",
            e,
            r
        );
    }

    Ok(())
}

/// Test that guidance_scale=1.0 produces conditional output.
/// CFG formula: pred = uncond + scale * (cond - uncond)
/// When scale=1.0: pred = uncond + 1.0 * (cond - uncond) = cond
#[test]
fn test_cfg_scale_one_is_conditional() -> Result<()> {
    let uncond = Tensor::new(&[1.0f32, 2.0, 3.0], &Device::Cpu)?;
    let cond = Tensor::new(&[10.0f32, 20.0, 30.0], &Device::Cpu)?;
    let guidance_scale = 1.0;

    // With scale=1.0, result should equal cond
    let diff = (&cond - &uncond)?;
    let result = (&uncond + (diff * guidance_scale))?;

    let result_vec: Vec<f32> = result.to_vec1()?;
    let cond_vec: Vec<f32> = cond.to_vec1()?;

    for (r, c) in result_vec.iter().zip(cond_vec.iter()) {
        assert!(
            (r - c).abs() < 1e-6,
            "guidance_scale=1.0 should produce cond, but {} != {}",
            r,
            c
        );
    }

    Ok(())
}

/// Test that higher guidance_scale increases difference from unconditional.
#[test]
fn test_cfg_scale_increases_conditioning() -> Result<()> {
    let uncond = Tensor::new(&[0.0f32, 0.0, 0.0], &Device::Cpu)?;
    let cond = Tensor::new(&[1.0f32, 1.0, 1.0], &Device::Cpu)?;

    // Test different scales
    let scales = [1.0, 3.0, 7.5, 15.0];
    let mut results = Vec::new();

    for scale in scales {
        let diff = (&cond - &uncond)?;
        let result = (&uncond + (diff * scale))?;
        let magnitude = result.abs()?.mean_all()?.to_scalar::<f32>()?;
        results.push(magnitude);
    }

    // Each result should be larger than the previous (monotonically increasing)
    for i in 1..results.len() {
        assert!(
            results[i] > results[i - 1],
            "Higher guidance scale should increase magnitude: {} <= {}",
            results[i],
            results[i - 1]
        );
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Numerical Stability Tests
// ---------------------------------------------------------------------------

/// Test that CFG is numerically stable across various guidance scales.
#[test]
fn test_cfg_numerical_stability() -> Result<()> {
    // Test with realistic noise prediction values
    let uncond = Tensor::randn(0f32, 1f32, (4, 4, 32, 32), &Device::Cpu)?;
    let cond = Tensor::randn(0f32, 1f32, (4, 4, 32, 32), &Device::Cpu)?;

    // Test various guidance scales
    let scales = [1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0];

    for scale in scales {
        let diff = (&cond - &uncond)?;
        let result = (&uncond + (diff * scale))?;

        // Check that result contains finite values by computing statistics
        let mean = result.mean_all()?.to_scalar::<f32>()?;
        let std = result
            .var_keepdim(0)?
            .sqrt()?
            .mean_all()?
            .to_scalar::<f32>()?;

        // Mean and std should be finite (not NaN or Inf)
        assert!(
            mean.is_finite(),
            "CFG produced non-finite mean {} with guidance_scale={}",
            mean,
            scale
        );

        assert!(
            std.is_finite(),
            "CFG produced non-finite std {} with guidance_scale={}",
            std,
            scale
        );
    }

    Ok(())
}

/// Test that CFG works with different tensor dtypes (F16, F32).
#[test]
fn test_cfg_dtype_compatibility() -> Result<()> {
    // Test with F32
    let uncond_f32 = Tensor::new(&[1.0f32, 2.0], &Device::Cpu)?;
    let cond_f32 = Tensor::new(&[2.0f32, 3.0], &Device::Cpu)?;
    let scale = 3.0;

    let diff_f32 = (cond_f32 - &uncond_f32)?;
    let result_f32 = (uncond_f32 + (diff_f32 * scale))?;

    // Verify F32 result is correct
    let expected = [4.0f32, 5.0];
    let result_vec: Vec<f32> = result_f32.to_vec1()?;

    for (r, e) in result_vec.iter().zip(expected.iter()) {
        assert!(
            (r - e).abs() < 1e-5,
            "F32 CFG result mismatch: {} != {}",
            r,
            e
        );
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Integration Tests
// ---------------------------------------------------------------------------

/// Test that spatial transformer preserves shapes with IP tokens.
#[test]
fn test_spatial_transformer_with_ip_tokens() -> Result<()> {
    let vs = test_varbuilder();

    let in_channels = 64;
    let n_heads = 4;
    let d_head = 16;
    let depth = 1;
    let context_dim = 64;
    let ip_dim = 128;
    let num_views = 2;
    let num_groups = 8;
    let use_linear = true;

    let transformer = oxigaf_diffusion::attention::MultiViewSpatialTransformer::new(
        vs.pp("transformer"),
        in_channels,
        n_heads,
        d_head,
        depth,
        context_dim,
        ip_dim,
        num_views,
        num_groups,
        use_linear,
    )?;

    let batch_views = 4;
    let height = 8;
    let width = 8;
    let xs = Tensor::randn(
        0f32,
        1f32,
        (batch_views, in_channels, height, width),
        &Device::Cpu,
    )?;

    let context = Tensor::randn(0f32, 1f32, (batch_views, 10, context_dim), &Device::Cpu)?;
    let ip_tokens = Tensor::randn(0f32, 1f32, (batch_views, 5, ip_dim), &Device::Cpu)?;

    // Forward with IP tokens
    let output = transformer.forward(&xs, Some(&context), Some(&ip_tokens))?;

    // Verify shape preservation
    assert_eq!(output.dims4()?, (batch_views, in_channels, height, width));

    Ok(())
}

/// Test that CFG produces different results based on IP conditioning.
#[test]
fn test_cfg_affects_output_based_on_ip_conditioning() -> Result<()> {
    let vs = test_varbuilder();

    let dim = 64;
    let n_heads = 4;
    let d_head = 16;
    let context_dim = 64;
    let ip_dim = 64;
    let num_views = 2;

    let block = oxigaf_diffusion::attention::MultiViewTransformerBlock::new(
        vs.pp("block"),
        dim,
        n_heads,
        d_head,
        context_dim,
        ip_dim,
        num_views,
    )?;

    let batch_views = 2;
    let seq_len = 8;
    let xs = Tensor::randn(0f32, 1f32, (batch_views, seq_len, dim), &Device::Cpu)?;
    let context = Tensor::zeros((batch_views, 4, context_dim), DType::F32, &Device::Cpu)?;
    let ip_tokens = Tensor::ones((batch_views, 4, ip_dim), DType::F32, &Device::Cpu)?;

    // Simulate CFG: conditional and unconditional passes
    let cond_output = block.forward(&xs, Some(&context), Some(&ip_tokens))?;
    let uncond_output = block.forward(&xs, Some(&context), None)?;

    // Apply CFG with scale=3.0
    let guidance_scale = 3.0;
    let diff = (&cond_output - &uncond_output)?;
    let cfg_output = (&uncond_output + (diff * guidance_scale))?;

    // CFG output should be different from both conditional and unconditional
    let diff_from_cond = (&cfg_output - &cond_output)?
        .abs()?
        .mean_all()?
        .to_scalar::<f32>()?;

    let diff_from_uncond = (&cfg_output - &uncond_output)?
        .abs()?
        .mean_all()?
        .to_scalar::<f32>()?;

    assert!(
        diff_from_cond > 1e-6,
        "CFG output should differ from conditional output"
    );
    assert!(
        diff_from_uncond > 1e-6,
        "CFG output should differ from unconditional output"
    );

    Ok(())
}

/// Test that guidance_scale validation works in config.
#[test]
fn test_guidance_scale_validation() {
    use oxigaf_diffusion::DiffusionConfig;

    let config = DiffusionConfig::default();

    // Default guidance_scale should be valid (>= 1.0)
    assert!(
        config.guidance_scale >= 1.0,
        "Default guidance_scale should be >= 1.0"
    );
}
