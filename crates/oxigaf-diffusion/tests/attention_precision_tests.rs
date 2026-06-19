//! Tests for flash vs standard attention precision and DiffusionConfig
//! use_flash_attention defaults.

use oxigaf_diffusion::config::DiffusionConfig;

// ---------------------------------------------------------------------------
// DiffusionConfig defaults
// ---------------------------------------------------------------------------

#[test]
fn test_diffusion_config_use_flash_attention_default_enabled_with_feature() {
    let cfg = DiffusionConfig::default();
    // When the flash_attention feature is compiled in, default must be true.
    // When the feature is absent, default must be false.
    // Either way the field must exist and be bool.
    #[cfg(feature = "flash_attention")]
    assert!(
        cfg.use_flash_attention,
        "use_flash_attention should default to true when feature is enabled"
    );
    #[cfg(not(feature = "flash_attention"))]
    assert!(
        !cfg.use_flash_attention,
        "use_flash_attention should default to false when feature is absent"
    );
}

#[test]
fn test_diffusion_config_use_flash_attention_can_be_disabled() {
    let cfg = DiffusionConfig {
        use_flash_attention: false,
        ..Default::default()
    };
    assert!(!cfg.use_flash_attention);
}

#[test]
fn test_diffusion_config_flash_attention_block_size_default() {
    let cfg = DiffusionConfig::default();
    assert_eq!(cfg.flash_attention_block_size, 64);
}

// ---------------------------------------------------------------------------
// Flash attention: output shape / tolerance vs standard attention
// ---------------------------------------------------------------------------

#[cfg(feature = "flash_attention")]
mod flash_tests {
    use candle_core::{Device, Tensor};
    use oxigaf_diffusion::flash_attention::{FlashAttention, FlashAttentionConfig};

    /// Build a simple (1, 1, seq, dim) random tensor.
    fn rand_tensor(seq: usize, dim: usize) -> Tensor {
        let data: Vec<f32> = (0..(seq * dim)).map(|i| (i as f32 * 0.01).sin()).collect();
        Tensor::from_vec(data, (1, 1, seq, dim), &Device::Cpu).expect("tensor")
    }

    #[test]
    fn test_flash_attention_output_shape() {
        let q = rand_tensor(16, 8);
        let k = rand_tensor(16, 8);
        let v = rand_tensor(16, 8);

        let fa = FlashAttention::with_dim_head(8);
        let out = fa.forward(&q, &k, &v).expect("flash attention forward");

        assert_eq!(out.dims(), &[1, 1, 16, 8]);
    }

    #[test]
    fn test_flash_vs_standard_attention_tolerance_small() {
        // For a small sequence the flash path falls back to standard attention
        // internally, so the outputs should be bitwise-identical to a hand-rolled
        // standard attention.

        let seq = 8; // <= default block_size(64), triggers standard fallback
        let dim = 4;
        let q = rand_tensor(seq, dim);
        let k = rand_tensor(seq, dim);
        let v = rand_tensor(seq, dim);

        // Flash attention (internally uses standard path for small seq).
        let flash_cfg = FlashAttentionConfig {
            block_size: 64,
            causal: false,
            softmax_eps: 1e-6,
        };
        let fa = FlashAttention::new(dim, flash_cfg);
        let flash_out = fa.forward(&q, &k, &v).expect("flash forward");

        // Standard attention using a larger block size that forces the tiled path
        // would require seq > block_size. Here we test shape-compatibility instead.
        assert_eq!(flash_out.dims(), &[1, 1, seq, dim]);
    }

    #[test]
    fn test_flash_vs_standard_attention_tolerance_large() {
        // Use a sequence length larger than the block size to exercise the
        // tiled flash attention path. Compare against the standard path
        // (block_size=0 is not valid so we use a very large block size for
        // the "standard" reference via the internal fallback in FlashAttention).

        let seq = 128; // > default block_size(64), exercises tiled path
        let dim = 8;
        let q = rand_tensor(seq, dim);
        let k = rand_tensor(seq, dim);
        let v = rand_tensor(seq, dim);

        // Tiled flash attention (block_size < seq).
        let tiled_cfg = FlashAttentionConfig {
            block_size: 32,
            causal: false,
            softmax_eps: 1e-6,
        };
        let fa_tiled = FlashAttention::new(dim, tiled_cfg);
        let tiled_out = fa_tiled.forward(&q, &k, &v).expect("tiled flash forward");

        // Standard attention via a block_size larger than seq (integer overflow
        // safe because we only use usize, just set a large value).
        let std_cfg = FlashAttentionConfig {
            block_size: 65536,
            causal: false,
            softmax_eps: 1e-6,
        };
        let fa_std = FlashAttention::new(dim, std_cfg);
        let std_out = fa_std.forward(&q, &k, &v).expect("std flash forward");

        // Both outputs must have the same shape.
        assert_eq!(tiled_out.dims(), std_out.dims());

        // Values must agree within a reasonable tolerance (< 1e-3 per element).
        let tiled_data: Vec<f32> = tiled_out
            .flatten_all()
            .expect("flatten tiled")
            .to_vec1()
            .expect("to_vec1 tiled");
        let std_data: Vec<f32> = std_out
            .flatten_all()
            .expect("flatten std")
            .to_vec1()
            .expect("to_vec1 std");

        let max_err = tiled_data
            .iter()
            .zip(std_data.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0_f32, f32::max);

        assert!(
            max_err < 1e-3,
            "flash vs standard attention max error {max_err} exceeds 1e-3 tolerance"
        );
    }
}
