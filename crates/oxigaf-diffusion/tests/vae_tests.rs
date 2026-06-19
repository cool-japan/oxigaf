//! Tests for encode_sequential and decode_sequential: shape round-trips,
//! latent statistics, and chunk-size independence.

use oxigaf_diffusion::sequential_vae::{
    decode_sequential, encode_sequential, EncodedViews, SequentialVaeConfig,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a small config: 64×64 images → 8×8 latents, 4 latent channels.
fn small_cfg(chunk_size: usize) -> SequentialVaeConfig {
    SequentialVaeConfig::new(chunk_size, 4, 64, 64, 0.18215)
}

/// Synthetic RGB image for the given config (all values in [0, 1]).
fn synthetic_image(cfg: &SequentialVaeConfig) -> Vec<f32> {
    let len = cfg.image_element_count();
    (0..len).map(|i| (i as f32) / (len as f32)).collect()
}

// ---------------------------------------------------------------------------
// Shape round-trip tests
// ---------------------------------------------------------------------------

#[test]
fn test_encode_sequential_output_shape_chunk1() {
    let cfg = small_cfg(1);
    let images: Vec<Vec<f32>> = (0..4).map(|_| synthetic_image(&cfg)).collect();

    let encoded = encode_sequential(&images, &cfg).expect("encode should succeed");

    assert_eq!(encoded.num_views, 4);
    assert_eq!(encoded.latent_height, 8); // 64 / 8
    assert_eq!(encoded.latent_width, 8);
    assert_eq!(encoded.latent_channels, 4);

    let expected_latent_len = 4 * 8 * 8;
    for (i, lat) in encoded.latents.iter().enumerate() {
        assert_eq!(
            lat.len(),
            expected_latent_len,
            "view {i} latent length mismatch"
        );
    }
}

#[test]
fn test_decode_sequential_output_shape_chunk1() {
    let cfg = small_cfg(1);
    let images: Vec<Vec<f32>> = (0..4).map(|_| synthetic_image(&cfg)).collect();

    let encoded = encode_sequential(&images, &cfg).expect("encode should succeed");
    let decoded = decode_sequential(&encoded, &cfg).expect("decode should succeed");

    assert_eq!(decoded.num_views, 4);
    assert_eq!(decoded.height, 64);
    assert_eq!(decoded.width, 64);
    assert_eq!(decoded.channels, 3); // always RGB

    let expected_image_len = 3 * 64 * 64;
    for (i, img) in decoded.images.iter().enumerate() {
        assert_eq!(
            img.len(),
            expected_image_len,
            "view {i} decoded image length mismatch"
        );
    }
}

#[test]
fn test_encode_decode_shape_same_with_chunk2() {
    let cfg1 = small_cfg(1);
    let cfg2 = small_cfg(2);
    let images: Vec<Vec<f32>> = (0..4).map(|_| synthetic_image(&cfg1)).collect();

    let enc1 = encode_sequential(&images, &cfg1).expect("encode chunk=1");
    let enc2 = encode_sequential(&images, &cfg2).expect("encode chunk=2");
    let dec1 = decode_sequential(&enc1, &cfg1).expect("decode chunk=1");
    let dec2 = decode_sequential(&enc2, &cfg2).expect("decode chunk=2");

    // Both chunk sizes must produce the same output shape.
    assert_eq!(enc1.num_views, enc2.num_views);
    assert_eq!(enc1.latent_height, enc2.latent_height);
    assert_eq!(enc1.latent_width, enc2.latent_width);
    assert_eq!(dec1.height, dec2.height);
    assert_eq!(dec1.width, dec2.width);
    assert_eq!(dec1.channels, dec2.channels);
}

#[test]
fn test_encode_decode_values_same_chunk1_vs_chunk2() {
    let cfg1 = small_cfg(1);
    let cfg2 = small_cfg(2);
    let n_views = 4;
    let images: Vec<Vec<f32>> = (0..n_views).map(|_| synthetic_image(&cfg1)).collect();

    let enc1 = encode_sequential(&images, &cfg1).expect("encode chunk=1");
    let enc2 = encode_sequential(&images, &cfg2).expect("encode chunk=2");

    // Latent values must be identical regardless of chunk size.
    for v in 0..n_views {
        for (j, (&a, &b)) in enc1.latents[v]
            .iter()
            .zip(enc2.latents[v].iter())
            .enumerate()
        {
            assert!(
                (a - b).abs() < 1e-6,
                "latent mismatch at view={v} elem={j}: chunk1={a} chunk2={b}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Latent statistics tests
// ---------------------------------------------------------------------------

#[test]
fn test_latent_statistics_mean_reasonable() {
    let cfg = small_cfg(1);
    let images: Vec<Vec<f32>> = (0..4).map(|_| synthetic_image(&cfg)).collect();

    let encoded = encode_sequential(&images, &cfg).expect("encode should succeed");

    for (i, lat) in encoded.latents.iter().enumerate() {
        let mean: f32 = lat.iter().sum::<f32>() / lat.len() as f32;
        // With latent_scale=0.18215 and pixel values in [0,1], mean should be << 1.
        assert!(
            mean.abs() < 0.5,
            "view {i}: latent mean {mean} is unreasonably large"
        );
    }
}

#[test]
fn test_latent_statistics_not_all_zero() {
    let cfg = small_cfg(1);
    let images: Vec<Vec<f32>> = (0..4).map(|_| synthetic_image(&cfg)).collect();

    let encoded = encode_sequential(&images, &cfg).expect("encode should succeed");

    for (i, lat) in encoded.latents.iter().enumerate() {
        let std: f32 = {
            let mean = lat.iter().sum::<f32>() / lat.len() as f32;
            let var = lat.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / lat.len() as f32;
            var.sqrt()
        };
        assert!(
            std > 0.0,
            "view {i}: latent std is zero — latents appear to be all-identical"
        );
    }
}

// ---------------------------------------------------------------------------
// Error propagation
// ---------------------------------------------------------------------------

#[test]
fn test_encode_empty_image_slice_fails() {
    let cfg = small_cfg(1);
    let empty: Vec<Vec<f32>> = vec![];
    let err = encode_sequential(&empty, &cfg).expect_err("empty slice should fail");
    assert!(
        err.to_string().contains("empty"),
        "expected 'empty' in error, got: {err}"
    );
}

#[test]
fn test_encode_wrong_image_size_fails() {
    let cfg = small_cfg(1);
    // Provide an image with one extra element.
    let bad_image = vec![0.0f32; cfg.image_element_count() + 1];
    let err = encode_sequential(&[bad_image], &cfg).expect_err("wrong size should fail");
    assert!(
        err.to_string().contains("elements"),
        "expected 'elements' in error, got: {err}"
    );
}

#[test]
fn test_decode_wrong_latent_size_fails() {
    let cfg = small_cfg(1);
    // Build valid EncodedViews but with a wrong latent length.
    let encoded = EncodedViews {
        latents: vec![vec![0.0f32; cfg.latent_element_count() + 1]],
        num_views: 1,
        latent_height: cfg.latent_height(),
        latent_width: cfg.latent_width(),
        latent_channels: cfg.latent_channels,
    };
    let err = decode_sequential(&encoded, &cfg).expect_err("wrong latent size should fail");
    assert!(
        err.to_string().contains("elements"),
        "expected 'elements' in error, got: {err}"
    );
}
