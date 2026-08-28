use super::*;

fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
    (a - b).abs() < eps
}

/// A flat plane facing the camera. Under this crate's depth convention
/// (positive, increasing away from the camera), a surface facing the
/// camera has a normal pointing toward *smaller* depth, i.e. `(0, 0, -1)`.
fn make_flat_scene(w: usize, h: usize, depth: f32) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = w * h;
    let depth_buf = vec![depth; n];
    let mut normal_buf = Vec::with_capacity(n * 3);
    for _ in 0..n {
        normal_buf.push(0.0_f32);
        normal_buf.push(0.0_f32);
        normal_buf.push(-1.0_f32);
    }
    let image = vec![0.5_f32; n * 3];
    (depth_buf, normal_buf, image)
}

// ── AoConfig tests ────────────────────────────────────────────────────────

#[test]
fn test_aoconfig_default_values() {
    let cfg = AoConfig::default();
    assert_eq!(cfg.n_samples, 16);
    assert!(approx_eq(cfg.radius, 0.5, 1e-6));
    assert!(approx_eq(cfg.bias, 0.025, 1e-6));
    assert!(approx_eq(cfg.power, 1.0, 1e-6));
    assert_eq!(cfg.blur_radius, 2);
    assert!(approx_eq(cfg.blur_sigma_space, 2.0, 1e-6));
    assert!(approx_eq(cfg.blur_sigma_depth, 0.1, 1e-6));
    assert!(approx_eq(cfg.strength, 1.0, 1e-6));
}

// ── AoProjParams tests ────────────────────────────────────────────────────

#[test]
fn test_ao_proj_params_new_field_names() {
    let p = AoProjParams::new(1234.5, 987.6);
    assert!(approx_eq(p.focal_px_x, 1234.5, 1e-3));
    assert!(approx_eq(p.focal_px_y, 987.6, 1e-3));
}

#[test]
fn test_ao_proj_params_from_fov_matches_manual_formula() {
    // 90-degree horizontal FOV at 1920x1080: focal_px_x = width / (2*tan(45deg)) = width/2.
    let fov_x = core::f32::consts::FRAC_PI_2;
    let fov_y = 2.0 * (1080.0f32 / 1920.0 * (fov_x * 0.5).tan()).atan();
    let p = AoProjParams::from_fov(fov_x, fov_y, 1920.0, 1080.0);
    assert!(
        approx_eq(p.focal_px_x, 960.0, 1.0),
        "focal_px_x should be ~960, got {}",
        p.focal_px_x
    );
    // Symmetric aspect-matched FOV should give the same focal length on both axes.
    assert!(
        approx_eq(p.focal_px_x, p.focal_px_y, 1.0),
        "focal_px_x={} focal_px_y={}",
        p.focal_px_x,
        p.focal_px_y
    );
}

// ── ao_sample_kernel tests ─────────────────────────────────────────────────

#[test]
fn test_ao_sample_kernel_length() {
    let kernel = ao_sample_kernel(16).expect("kernel generation failed");
    assert_eq!(kernel.len(), 16 * 3);
}

#[test]
fn test_ao_sample_kernel_various_sizes() {
    for n in [1, 4, 8, 32, 64] {
        let kernel = ao_sample_kernel(n).expect("kernel generation failed");
        assert_eq!(kernel.len(), n * 3, "n={n}");
    }
}

#[test]
fn test_ao_sample_kernel_within_unit_sphere() {
    let kernel = ao_sample_kernel(64).expect("kernel generation failed");
    let n = kernel.len() / 3;
    for i in 0..n {
        let x = kernel[i * 3];
        let y = kernel[i * 3 + 1];
        let z = kernel[i * 3 + 2];
        let len = (x * x + y * y + z * z).sqrt();
        assert!(
            len <= 1.001,
            "Sample {i} length {len:.4} exceeds unit sphere"
        );
    }
}

#[test]
fn test_ao_sample_kernel_z_nonnegative() {
    let kernel = ao_sample_kernel(32).expect("kernel generation failed");
    let n = kernel.len() / 3;
    for i in 0..n {
        let z = kernel[i * 3 + 2];
        assert!(z >= 0.0, "Sample {i} has negative z={z}");
    }
}

#[test]
fn test_ao_sample_kernel_zero_samples_error() {
    let result = ao_sample_kernel(0);
    assert!(
        matches!(result, Err(AoError::InvalidConfig(_))),
        "Expected InvalidConfig, got {result:?}"
    );
}

#[test]
fn test_ao_sample_kernel_deterministic() {
    let k1 = ao_sample_kernel(16).expect("kernel 1 failed");
    let k2 = ao_sample_kernel(16).expect("kernel 2 failed");
    assert_eq!(k1, k2, "Kernel generation must be deterministic");
}

#[test]
fn test_ao_sample_kernel_different_sizes_differ() {
    let k8 = ao_sample_kernel(8).expect("kernel 8 failed");
    let k16 = ao_sample_kernel(16).expect("kernel 16 failed");
    // Different seeds (based on n_samples), so values should differ
    assert_ne!(
        &k8[..8.min(k8.len())],
        &k16[..8.min(k16.len())],
        "Kernels with different n_samples should differ"
    );
}

#[test]
fn test_ao_sample_kernel_is_cosine_weighted() {
    // For a cosine-weighted hemisphere, z (unit-direction) has density
    // proportional to z itself over [0, 1], so z^2 is *uniformly*
    // distributed with Var[z^2] = 1/12 ≈ 0.0833. A naive
    // uniform-in-polar-angle sampler instead gives Var[cos^2(phi)] with
    // phi ~ Uniform(0, pi/2), which works out to 0.125 — noticeably
    // higher. 0.105 sits roughly halfway between the two and is a safe
    // discriminator (verified empirically over many sample counts).
    let n = 20_000;
    let kernel = ao_sample_kernel(n).expect("kernel generation failed");
    let count = kernel.len() / 3;

    let mut sum_z2 = 0.0_f64;
    let mut sum_z4 = 0.0_f64;
    let mut used = 0usize;
    for i in 0..count {
        let x = kernel[i * 3] as f64;
        let y = kernel[i * 3 + 1] as f64;
        let z = kernel[i * 3 + 2] as f64;
        let len = (x * x + y * y + z * z).sqrt();
        if len < 1e-9 {
            continue;
        }
        let zn = z / len;
        let z2 = zn * zn;
        sum_z2 += z2;
        sum_z4 += z2 * z2;
        used += 1;
    }

    let mean_z2 = sum_z2 / used as f64;
    let var_z2 = sum_z4 / used as f64 - mean_z2 * mean_z2;
    assert!(
        var_z2 < 0.105,
        "Kernel z-distribution should be cosine-weighted \
         (Var[z^2] ~= 0.083), got Var[z^2]={var_z2:.4} (used {used} samples); \
         a uniform-in-polar-angle sampler would give ~0.125"
    );
}

// ── ao_noise_texture tests ────────────────────────────────────────────────

#[test]
fn test_ao_noise_texture_length() {
    let noise = ao_noise_texture();
    assert_eq!(
        noise.len(),
        32,
        "Noise texture should have 32 values (16 pairs)"
    );
}

#[test]
fn test_ao_noise_texture_values_in_range() {
    let noise = ao_noise_texture();
    for (i, &v) in noise.iter().enumerate() {
        assert!(
            (-1.0..=1.0).contains(&v),
            "Noise value {i} = {v} out of [-1, 1]"
        );
    }
}

#[test]
fn test_ao_noise_texture_deterministic() {
    let n1 = ao_noise_texture();
    let n2 = ao_noise_texture();
    assert_eq!(n1, n2, "Noise texture must be deterministic");
}

#[test]
fn test_ao_noise_texture_has_variation() {
    let noise = ao_noise_texture();
    let first = noise[0];
    let has_different = noise.iter().any(|&v| (v - first).abs() > 1e-6);
    assert!(has_different, "Noise texture should have variation");
}

// ── ao_smoothstep tests ───────────────────────────────────────────────────

#[test]
fn test_ao_smoothstep_below_edge0() {
    let v = ao_smoothstep(0.0, 1.0, -0.5);
    assert!(approx_eq(v, 0.0, 1e-6), "Below edge0 should be 0, got {v}");
}

#[test]
fn test_ao_smoothstep_above_edge1() {
    let v = ao_smoothstep(0.0, 1.0, 1.5);
    assert!(approx_eq(v, 1.0, 1e-6), "Above edge1 should be 1, got {v}");
}

#[test]
fn test_ao_smoothstep_at_edge0() {
    let v = ao_smoothstep(0.0, 1.0, 0.0);
    assert!(approx_eq(v, 0.0, 1e-6), "At edge0 should be 0, got {v}");
}

#[test]
fn test_ao_smoothstep_at_edge1() {
    let v = ao_smoothstep(0.0, 1.0, 1.0);
    assert!(approx_eq(v, 1.0, 1e-6), "At edge1 should be 1, got {v}");
}

#[test]
fn test_ao_smoothstep_midpoint() {
    let v = ao_smoothstep(0.0, 1.0, 0.5);
    assert!(approx_eq(v, 0.5, 1e-5), "Midpoint should be 0.5, got {v}");
}

#[test]
fn test_ao_smoothstep_monotone() {
    let mut prev = 0.0_f32;
    for i in 0..=20 {
        let x = i as f32 * 0.05;
        let v = ao_smoothstep(0.0, 1.0, x);
        assert!(v >= prev - 1e-6, "Smoothstep not monotone at x={x}");
        prev = v;
    }
}

// ── ao_cross tests ────────────────────────────────────────────────────────

#[test]
fn test_ao_cross_x_cross_y_equals_z() {
    let r = ao_cross([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
    assert!(approx_eq(r[0], 0.0, 1e-6));
    assert!(approx_eq(r[1], 0.0, 1e-6));
    assert!(approx_eq(r[2], 1.0, 1e-6));
}

#[test]
fn test_ao_cross_y_cross_z_equals_x() {
    let r = ao_cross([0.0, 1.0, 0.0], [0.0, 0.0, 1.0]);
    assert!(approx_eq(r[0], 1.0, 1e-6));
    assert!(approx_eq(r[1], 0.0, 1e-6));
    assert!(approx_eq(r[2], 0.0, 1e-6));
}

#[test]
fn test_ao_cross_z_cross_x_equals_y() {
    let r = ao_cross([0.0, 0.0, 1.0], [1.0, 0.0, 0.0]);
    assert!(approx_eq(r[0], 0.0, 1e-6));
    assert!(approx_eq(r[1], 1.0, 1e-6));
    assert!(approx_eq(r[2], 0.0, 1e-6));
}

#[test]
fn test_ao_cross_anticommutative() {
    let a = [1.0_f32, 2.0, 3.0];
    let b = [4.0_f32, 5.0, 6.0];
    let ab = ao_cross(a, b);
    let ba = ao_cross(b, a);
    for i in 0..3 {
        assert!(
            approx_eq(ab[i], -ba[i], 1e-5),
            "Not anticommutative at component {i}"
        );
    }
}

#[test]
fn test_ao_cross_parallel_vectors_zero() {
    let r = ao_cross([1.0, 0.0, 0.0], [2.0, 0.0, 0.0]);
    for v in r {
        assert!(
            approx_eq(v, 0.0, 1e-6),
            "Parallel cross product should be zero, got {v}"
        );
    }
}

// ── ao_dot tests ──────────────────────────────────────────────────────────

#[test]
fn test_ao_dot_orthogonal() {
    let d = ao_dot([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
    assert!(approx_eq(d, 0.0, 1e-6), "Orthogonal dot should be 0");
}

#[test]
fn test_ao_dot_parallel() {
    let d = ao_dot([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]);
    let expected = 1.0 + 4.0 + 9.0;
    assert!(approx_eq(d, expected, 1e-5));
}

#[test]
fn test_ao_dot_antiparallel() {
    let d = ao_dot([1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]);
    assert!(approx_eq(d, -1.0, 1e-6));
}

// ── ao_normalize tests ────────────────────────────────────────────────────

#[test]
fn test_ao_normalize_unit_result() {
    let v = ao_normalize([3.0, 4.0, 0.0]);
    let len = ao_dot(v, v).sqrt();
    assert!(
        approx_eq(len, 1.0, 1e-5),
        "Normalized length should be 1, got {len}"
    );
}

#[test]
fn test_ao_normalize_zero_vector_fallback() {
    let v = ao_normalize([0.0, 0.0, 0.0]);
    assert_eq!(v, [0.0, 0.0, 1.0], "Zero vector fallback should be (0,0,1)");
}

#[test]
fn test_ao_normalize_already_unit() {
    let v = ao_normalize([1.0, 0.0, 0.0]);
    assert!(approx_eq(v[0], 1.0, 1e-6));
    assert!(approx_eq(v[1], 0.0, 1e-6));
    assert!(approx_eq(v[2], 0.0, 1e-6));
}

#[test]
fn test_ao_normalize_direction_preserved() {
    let input = [3.0_f32, 0.0, 0.0];
    let v = ao_normalize(input);
    assert!(approx_eq(v[0], 1.0, 1e-6), "Direction should be preserved");
    assert!(approx_eq(v[1], 0.0, 1e-6));
    assert!(approx_eq(v[2], 0.0, 1e-6));
}

// ── ao_sample_depth tests ─────────────────────────────────────────────────

#[test]
fn test_ao_sample_depth_exact_pixel() {
    let depth_buf = vec![1.0_f32, 2.0, 3.0, 4.0];
    let d = ao_sample_depth(&depth_buf, 2, 2, 0.0, 0.0);
    assert!(
        approx_eq(d, 1.0, 1e-5),
        "Top-left pixel should be 1.0, got {d}"
    );
}

#[test]
fn test_ao_sample_depth_center_pixel() {
    let depth_buf = vec![5.0_f32; 9]; // 3x3 all 5.0
    let d = ao_sample_depth(&depth_buf, 3, 3, 1.0, 1.0);
    assert!(
        approx_eq(d, 5.0, 1e-5),
        "Center pixel should be 5.0, got {d}"
    );
}

#[test]
fn test_ao_sample_depth_clamp_boundary() {
    let depth_buf = vec![1.0_f32, 2.0, 3.0, 4.0];
    // Out of bounds coordinates should clamp
    let d_neg = ao_sample_depth(&depth_buf, 2, 2, -1.0, -1.0);
    assert!(
        approx_eq(d_neg, 1.0, 1e-5),
        "Negative coords should clamp to top-left"
    );
    let d_over = ao_sample_depth(&depth_buf, 2, 2, 10.0, 10.0);
    assert!(
        approx_eq(d_over, 4.0, 1e-5),
        "Overflow coords should clamp to bottom-right"
    );
}

#[test]
fn test_ao_sample_depth_bilinear_center() {
    // 2x2 with values [1, 2; 3, 4]
    // At (0.5, 0.5) the bilinear interpolation should give 2.5
    let depth_buf = vec![1.0_f32, 2.0, 3.0, 4.0];
    let d = ao_sample_depth(&depth_buf, 2, 2, 0.5, 0.5);
    assert!(
        approx_eq(d, 2.5, 1e-5),
        "Bilinear center should be 2.5, got {d}"
    );
}

#[test]
fn test_ao_sample_depth_empty() {
    let d = ao_sample_depth(&[], 0, 0, 0.0, 0.0);
    assert!(approx_eq(d, 0.0, 1e-6), "Empty buffer should return 0.0");
}

// ── ao_compute tests ──────────────────────────────────────────────────────

#[test]
fn test_ao_compute_all_background_returns_ones() {
    let w = 8;
    let h = 8;
    let depth_buf = vec![0.0_f32; w * h]; // all background
    let normal_buf = vec![0.0_f32; w * h * 3];
    let kernel = ao_sample_kernel(8).expect("kernel failed");
    let noise = ao_noise_texture();
    let config = AoConfig::default();

    let ao = ao_compute(
        &depth_buf,
        &normal_buf,
        w,
        h,
        AoSamplingBuffers {
            kernel: &kernel,
            noise: &noise,
        },
        AoProjParams::new(1.0, 1.0),
        &config,
    )
    .expect("ao_compute failed");

    for (i, &v) in ao.iter().enumerate() {
        assert!(
            approx_eq(v, 1.0, 1e-6),
            "Background pixel {i} should be 1.0, got {v}"
        );
    }
}

#[test]
fn test_ao_compute_output_dimensions() {
    let w = 12;
    let h = 8;
    let (depth_buf, normal_buf, _) = make_flat_scene(w, h, 2.0);
    let kernel = ao_sample_kernel(8).expect("kernel failed");
    let noise = ao_noise_texture();
    let config = AoConfig::default();

    let ao = ao_compute(
        &depth_buf,
        &normal_buf,
        w,
        h,
        AoSamplingBuffers {
            kernel: &kernel,
            noise: &noise,
        },
        AoProjParams::new(1.0, 1.0),
        &config,
    )
    .expect("ao_compute failed");

    assert_eq!(ao.len(), w * h);
}

#[test]
fn test_ao_compute_dimension_mismatch_depth() {
    let w = 4;
    let h = 4;
    let depth_buf = vec![1.0_f32; w * h + 1]; // wrong size
    let normal_buf = vec![0.0_f32; w * h * 3];
    let kernel = ao_sample_kernel(8).expect("kernel failed");
    let noise = ao_noise_texture();
    let config = AoConfig::default();

    let result = ao_compute(
        &depth_buf,
        &normal_buf,
        w,
        h,
        AoSamplingBuffers {
            kernel: &kernel,
            noise: &noise,
        },
        AoProjParams::new(1.0, 1.0),
        &config,
    );
    assert!(
        matches!(result, Err(AoError::DimensionMismatch { .. })),
        "Expected DimensionMismatch, got {result:?}"
    );
}

#[test]
fn test_ao_compute_dimension_mismatch_normal() {
    let w = 4;
    let h = 4;
    let depth_buf = vec![1.0_f32; w * h];
    let normal_buf = vec![0.0_f32; w * h * 3 + 1]; // wrong size
    let kernel = ao_sample_kernel(8).expect("kernel failed");
    let noise = ao_noise_texture();
    let config = AoConfig::default();

    let result = ao_compute(
        &depth_buf,
        &normal_buf,
        w,
        h,
        AoSamplingBuffers {
            kernel: &kernel,
            noise: &noise,
        },
        AoProjParams::new(1.0, 1.0),
        &config,
    );
    assert!(
        matches!(result, Err(AoError::DimensionMismatch { .. })),
        "Expected DimensionMismatch, got {result:?}"
    );
}

#[test]
fn test_ao_compute_empty_input_error() {
    let kernel = ao_sample_kernel(8).expect("kernel failed");
    let noise = ao_noise_texture();
    let config = AoConfig::default();
    let result = ao_compute(
        &[],
        &[],
        0,
        0,
        AoSamplingBuffers {
            kernel: &kernel,
            noise: &noise,
        },
        AoProjParams::new(1.0, 1.0),
        &config,
    );
    assert!(matches!(result, Err(AoError::EmptyInput)));
}

#[test]
fn test_ao_compute_invalid_radius_error() {
    let w = 4;
    let h = 4;
    let (depth_buf, normal_buf, _) = make_flat_scene(w, h, 1.0);
    let kernel = ao_sample_kernel(8).expect("kernel failed");
    let noise = ao_noise_texture();
    let config = AoConfig {
        radius: -0.1,
        ..Default::default()
    };

    let result = ao_compute(
        &depth_buf,
        &normal_buf,
        w,
        h,
        AoSamplingBuffers {
            kernel: &kernel,
            noise: &noise,
        },
        AoProjParams::new(1.0, 1.0),
        &config,
    );
    assert!(matches!(result, Err(AoError::InvalidRadius(_))));
}

#[test]
fn test_ao_compute_empty_noise_error_not_panic() {
    // A regression guard for a modulo-by-zero panic: `noise_idx * 2 %
    // noise.len()` must never execute with an empty noise slice.
    let w = 4;
    let h = 4;
    let (depth_buf, normal_buf, _) = make_flat_scene(w, h, 1.0);
    let kernel = ao_sample_kernel(8).expect("kernel failed");
    let config = AoConfig::default();

    let result = ao_compute(
        &depth_buf,
        &normal_buf,
        w,
        h,
        AoSamplingBuffers {
            kernel: &kernel,
            noise: &[],
        },
        AoProjParams::new(1.0, 1.0),
        &config,
    );
    assert!(
        matches!(result, Err(AoError::InvalidConfig(_))),
        "Expected InvalidConfig for empty noise, got {result:?}"
    );
}

#[test]
fn test_ao_compute_short_noise_error() {
    let w = 4;
    let h = 4;
    let (depth_buf, normal_buf, _) = make_flat_scene(w, h, 1.0);
    let kernel = ao_sample_kernel(8).expect("kernel failed");
    let config = AoConfig::default();

    // Fewer than 32 values (below the 16-pair noise tile size).
    let short_noise = vec![0.5_f32; 8];
    let result = ao_compute(
        &depth_buf,
        &normal_buf,
        w,
        h,
        AoSamplingBuffers {
            kernel: &kernel,
            noise: &short_noise,
        },
        AoProjParams::new(1.0, 1.0),
        &config,
    );
    assert!(
        matches!(result, Err(AoError::InvalidConfig(_))),
        "Expected InvalidConfig for short noise, got {result:?}"
    );
}

#[test]
fn test_ao_compute_empty_kernel_error_not_silent_noop() {
    // An empty kernel must be rejected rather than silently producing
    // AO = 1.0 everywhere (n_samples = 0 would otherwise be a silent
    // no-op indistinguishable from "nothing occluded").
    let w = 4;
    let h = 4;
    let (depth_buf, normal_buf, _) = make_flat_scene(w, h, 1.0);
    let noise = ao_noise_texture();
    let config = AoConfig::default();

    let result = ao_compute(
        &depth_buf,
        &normal_buf,
        w,
        h,
        AoSamplingBuffers {
            kernel: &[],
            noise: &noise,
        },
        AoProjParams::new(1.0, 1.0),
        &config,
    );
    assert!(
        matches!(result, Err(AoError::InvalidConfig(_))),
        "Expected InvalidConfig for empty kernel, got {result:?}"
    );
}

#[test]
fn test_ao_compute_kernel_len_not_multiple_of_3_error() {
    let w = 4;
    let h = 4;
    let (depth_buf, normal_buf, _) = make_flat_scene(w, h, 1.0);
    let noise = ao_noise_texture();
    let config = AoConfig::default();
    let bad_kernel = vec![0.1_f32; 7]; // not a multiple of 3

    let result = ao_compute(
        &depth_buf,
        &normal_buf,
        w,
        h,
        AoSamplingBuffers {
            kernel: &bad_kernel,
            noise: &noise,
        },
        AoProjParams::new(1.0, 1.0),
        &config,
    );
    assert!(
        matches!(result, Err(AoError::InvalidConfig(_))),
        "Expected InvalidConfig for kernel length not a multiple of 3, got {result:?}"
    );
}

#[test]
fn test_ao_compute_flat_surface_high_ao() {
    let w = 16;
    let h = 16;
    let (depth_buf, normal_buf, _) = make_flat_scene(w, h, 2.0);
    let kernel = ao_sample_kernel(16).expect("kernel failed");
    let noise = ao_noise_texture();
    let config = AoConfig::default();

    let ao = ao_compute(
        &depth_buf,
        &normal_buf,
        w,
        h,
        AoSamplingBuffers {
            kernel: &kernel,
            noise: &noise,
        },
        AoProjParams::new(500.0, 500.0),
        &config,
    )
    .expect("ao_compute failed");

    let mean = ao.iter().sum::<f32>() / ao.len() as f32;
    // A perfectly flat plane has no geometry to occlude any of its own
    // hemisphere samples, so it must come back (almost) fully unoccluded.
    assert!(
        mean > 0.99,
        "Perfectly flat surface must be ~fully unoccluded, got mean={mean:.3}"
    );
    for (i, &v) in ao.iter().enumerate() {
        assert!(
            v > 0.99,
            "Flat surface pixel {i} should be ~fully unoccluded, got {v:.3}"
        );
    }
}

#[test]
fn test_ao_compute_values_in_range() {
    let w = 8;
    let h = 8;
    let (depth_buf, normal_buf, _) = make_flat_scene(w, h, 1.5);
    let kernel = ao_sample_kernel(16).expect("kernel failed");
    let noise = ao_noise_texture();
    let config = AoConfig::default();

    let ao = ao_compute(
        &depth_buf,
        &normal_buf,
        w,
        h,
        AoSamplingBuffers {
            kernel: &kernel,
            noise: &noise,
        },
        AoProjParams::new(1.0, 1.0),
        &config,
    )
    .expect("ao_compute failed");

    for (i, &v) in ao.iter().enumerate() {
        assert!((0.0..=1.0).contains(&v), "AO value {i} = {v} out of [0,1]");
    }
}

#[test]
fn test_ao_compute_depth_disparity_causes_occlusion() {
    // A plane at depth 5.0 with a 4x4 "bump" at depth 0.1 (much closer
    // to the camera) in the middle. All surfaces face the camera, i.e.
    // normal = (0, 0, -1) under this crate's depth/normal convention.
    let w = 8;
    let h = 8;
    let n = w * h;
    let mut depth_buf = vec![5.0_f32; n];
    for py in 2..6 {
        for px in 2..6 {
            depth_buf[py * w + px] = 0.1;
        }
    }
    let mut normal_buf = vec![0.0_f32; n * 3];
    for i in 0..n {
        normal_buf[i * 3 + 2] = -1.0;
    }
    let kernel = ao_sample_kernel(16).expect("kernel failed");
    let noise = ao_noise_texture();
    let config = AoConfig::default();

    let ao = ao_compute(
        &depth_buf,
        &normal_buf,
        w,
        h,
        AoSamplingBuffers {
            kernel: &kernel,
            noise: &noise,
        },
        AoProjParams::new(10.0, 10.0),
        &config,
    )
    .expect("ao_compute failed");

    // Check that values are in range
    for (i, &v) in ao.iter().enumerate() {
        assert!((0.0..=1.0).contains(&v), "AO value {i} = {v} out of [0,1]");
    }

    let idx = |px: usize, py: usize| py * w + px;

    // Plane pixels far from the bump (grid corners): max hemisphere
    // reach at depth 5.0 with radius 0.5 and focal_px 10.0 is
    // 0.5 * 10.0 / 5.0 = 1.0 pixel, so a corner (>= 2 pixels from the
    // 2..6 block) can never sample the bump and must be fully
    // unoccluded on this flat region.
    let far_pixels = [(0, 0), (7, 0), (0, 7), (7, 7)];
    let far_mean: f32 =
        far_pixels.iter().map(|&(x, y)| ao[idx(x, y)]).sum::<f32>() / far_pixels.len() as f32;
    assert!(
        far_mean > 0.99,
        "Plane pixels far from the bump should be unoccluded, got mean={far_mean:.3}"
    );

    // Plane pixels immediately bordering the bump: some hemisphere
    // samples land on (or bilinearly blend toward) the much-closer bump
    // depth, so these must show measurable occlusion relative to the
    // far pixels.
    let mut near_ring = Vec::new();
    for px in 2..6 {
        near_ring.push((px, 1));
        near_ring.push((px, 6));
    }
    for py in 2..6 {
        near_ring.push((1, py));
        near_ring.push((6, py));
    }
    let near_mean: f32 =
        near_ring.iter().map(|&(x, y)| ao[idx(x, y)]).sum::<f32>() / near_ring.len() as f32;
    assert!(
        near_mean < far_mean - 0.1,
        "Plane pixels adjacent to the near bump should be measurably more \
         occluded than pixels far from it: near_mean={near_mean:.3}, far_mean={far_mean:.3}"
    );

    // The bump's own interior: its neighbors are either the same close
    // depth (no occlusion) or the much farther plane (also not
    // occluding, since farther means free space under this
    // convention), so the interior must stay unoccluded too.
    let bump_interior = [(3, 3), (4, 4), (2, 2), (5, 5)];
    for &(x, y) in &bump_interior {
        let v = ao[idx(x, y)];
        assert!(
            v > 0.9,
            "Bump interior pixel ({x},{y}) should be ~unoccluded, got {v:.3}"
        );
    }
}

#[test]
fn test_ao_compute_higher_power_darkens_partial_occlusion() {
    // `power` is documented as an "AO intensity exponent": raising it
    // should darken (further occlude) partially-occluded pixels. Uses
    // the same near-bump scene as the disparity test, whose ring
    // pixels are all partially (not fully) occluded.
    let w = 8;
    let h = 8;
    let n = w * h;
    let mut depth_buf = vec![5.0_f32; n];
    for py in 2..6 {
        for px in 2..6 {
            depth_buf[py * w + px] = 0.1;
        }
    }
    let mut normal_buf = vec![0.0_f32; n * 3];
    for i in 0..n {
        normal_buf[i * 3 + 2] = -1.0;
    }
    let kernel = ao_sample_kernel(16).expect("kernel failed");
    let noise = ao_noise_texture();

    let idx = |px: usize, py: usize| py * w + px;
    let mut near_ring = Vec::new();
    for px in 2..6 {
        near_ring.push((px, 1));
        near_ring.push((px, 6));
    }
    for py in 2..6 {
        near_ring.push((1, py));
        near_ring.push((6, py));
    }

    let ring_mean_at = |power: f32| -> f32 {
        let config = AoConfig {
            power,
            ..Default::default()
        };
        let ao = ao_compute(
            &depth_buf,
            &normal_buf,
            w,
            h,
            AoSamplingBuffers {
                kernel: &kernel,
                noise: &noise,
            },
            AoProjParams::new(10.0, 10.0),
            &config,
        )
        .expect("ao_compute failed");
        near_ring.iter().map(|&(x, y)| ao[idx(x, y)]).sum::<f32>() / near_ring.len() as f32
    };

    let mean_power_low = ring_mean_at(0.5);
    let mean_power_default = ring_mean_at(1.0);
    let mean_power_high = ring_mean_at(3.0);

    assert!(
        mean_power_default < mean_power_low - 0.05,
        "power=1.0 should darken more than power=0.5: {mean_power_default:.3} vs {mean_power_low:.3}"
    );
    assert!(
        mean_power_high < mean_power_default - 0.05,
        "power=3.0 should darken more than power=1.0: {mean_power_high:.3} vs {mean_power_default:.3}"
    );
}

#[test]
fn test_ao_compute_infinite_depth_treated_as_background() {
    let w = 4;
    let h = 4;
    let depth_buf = vec![f32::INFINITY; w * h];
    let normal_buf = vec![0.0_f32; w * h * 3];
    let kernel = ao_sample_kernel(8).expect("kernel failed");
    let noise = ao_noise_texture();
    let config = AoConfig::default();

    let ao = ao_compute(
        &depth_buf,
        &normal_buf,
        w,
        h,
        AoSamplingBuffers {
            kernel: &kernel,
            noise: &noise,
        },
        AoProjParams::new(1.0, 1.0),
        &config,
    )
    .expect("ao_compute failed");

    for (i, &v) in ao.iter().enumerate() {
        assert!(
            approx_eq(v, 1.0, 1e-6),
            "Infinite depth pixel {i} should be 1.0"
        );
    }
}

// ── ao_bilateral_blur tests ───────────────────────────────────────────────

#[test]
fn test_ao_bilateral_blur_flat_unchanged() {
    let w = 8;
    let h = 8;
    let ao_buf = vec![1.0_f32; w * h];
    let depth_buf = vec![1.0_f32; w * h];
    let config = AoConfig::default();

    let blurred =
        ao_bilateral_blur(&ao_buf, &depth_buf, w, h, &config).expect("bilateral blur failed");

    for (i, (&a, &b)) in ao_buf.iter().zip(blurred.iter()).enumerate() {
        assert!(approx_eq(a, b, 1e-5), "Flat AO pixel {i}: {a} != {b}");
    }
}

#[test]
fn test_ao_bilateral_blur_output_dimensions() {
    let w = 10;
    let h = 6;
    let ao_buf = vec![0.8_f32; w * h];
    let depth_buf = vec![2.0_f32; w * h];
    let config = AoConfig::default();

    let blurred =
        ao_bilateral_blur(&ao_buf, &depth_buf, w, h, &config).expect("bilateral blur failed");

    assert_eq!(blurred.len(), w * h);
}

#[test]
fn test_ao_bilateral_blur_dimension_mismatch_ao() {
    let w = 4;
    let h = 4;
    let ao_buf = vec![1.0_f32; w * h + 1];
    let depth_buf = vec![1.0_f32; w * h];
    let config = AoConfig::default();

    let result = ao_bilateral_blur(&ao_buf, &depth_buf, w, h, &config);
    assert!(matches!(result, Err(AoError::DimensionMismatch { .. })));
}

#[test]
fn test_ao_bilateral_blur_dimension_mismatch_depth() {
    let w = 4;
    let h = 4;
    let ao_buf = vec![1.0_f32; w * h];
    let depth_buf = vec![1.0_f32; w * h + 1];
    let config = AoConfig::default();

    let result = ao_bilateral_blur(&ao_buf, &depth_buf, w, h, &config);
    assert!(matches!(result, Err(AoError::DimensionMismatch { .. })));
}

#[test]
fn test_ao_bilateral_blur_empty_input() {
    let config = AoConfig::default();
    let result = ao_bilateral_blur(&[], &[], 0, 0, &config);
    assert!(matches!(result, Err(AoError::EmptyInput)));
}

#[test]
fn test_ao_bilateral_blur_values_in_range() {
    let w = 8;
    let h = 8;
    let ao_buf: Vec<f32> = (0..w * h).map(|i| (i % 10) as f32 / 10.0).collect();
    let depth_buf = vec![1.0_f32; w * h];
    let config = AoConfig::default();

    let blurred =
        ao_bilateral_blur(&ao_buf, &depth_buf, w, h, &config).expect("bilateral blur failed");

    for (i, &v) in blurred.iter().enumerate() {
        assert!(
            (0.0..=1.0).contains(&v),
            "Blurred value {i} = {v} out of [0,1]"
        );
    }
}

#[test]
fn test_ao_bilateral_blur_preserves_background() {
    let w = 4;
    let h = 4;
    // Mix background (depth=0) and foreground
    let mut depth_buf = vec![1.0_f32; w * h];
    depth_buf[0] = 0.0; // background pixel
    let mut ao_buf = vec![0.5_f32; w * h];
    ao_buf[0] = 1.0; // background AO value
    let config = AoConfig::default();

    let blurred =
        ao_bilateral_blur(&ao_buf, &depth_buf, w, h, &config).expect("bilateral blur failed");

    // Background pixel should pass through unchanged
    assert!(
        approx_eq(blurred[0], 1.0, 1e-5),
        "Background AO should be unchanged"
    );
}

// ── ao_apply_to_image tests ───────────────────────────────────────────────

#[test]
fn test_ao_apply_strength_zero_unchanged() {
    let w = 4;
    let h = 4;
    let image: Vec<f32> = (0..w * h * 3).map(|i| (i % 10) as f32 / 10.0).collect();
    let ao_buf = vec![0.0_f32; w * h]; // fully occluded
    let result = ao_apply_to_image(&image, &ao_buf, w, h, 0.0).expect("apply failed");

    for (i, (&a, &b)) in image.iter().zip(result.iter()).enumerate() {
        assert!(
            approx_eq(a, b, 1e-5),
            "strength=0 changed pixel {i}: {a} != {b}"
        );
    }
}

#[test]
fn test_ao_apply_full_ao_one_unchanged() {
    let w = 4;
    let h = 4;
    let image = vec![0.7_f32; w * h * 3];
    let ao_buf = vec![1.0_f32; w * h]; // no occlusion
    let result = ao_apply_to_image(&image, &ao_buf, w, h, 1.0).expect("apply failed");

    for (i, (&a, &b)) in image.iter().zip(result.iter()).enumerate() {
        assert!(
            approx_eq(a, b, 1e-5),
            "AO=1.0 changed pixel {i}: {a} != {b}"
        );
    }
}

#[test]
fn test_ao_apply_full_ao_zero_darkens() {
    let w = 2;
    let h = 2;
    let image = vec![1.0_f32; w * h * 3];
    let ao_buf = vec![0.0_f32; w * h];
    let result = ao_apply_to_image(&image, &ao_buf, w, h, 1.0).expect("apply failed");

    for (i, &v) in result.iter().enumerate() {
        assert!(approx_eq(v, 0.0, 1e-5), "Pixel {i} should be 0, got {v}");
    }
}

#[test]
fn test_ao_apply_dimension_mismatch_ao() {
    let w = 4;
    let h = 4;
    let image = vec![0.5_f32; w * h * 3];
    let ao_buf = vec![1.0_f32; w * h + 1];
    let result = ao_apply_to_image(&image, &ao_buf, w, h, 1.0);
    assert!(matches!(result, Err(AoError::DimensionMismatch { .. })));
}

#[test]
fn test_ao_apply_dimension_mismatch_image() {
    let w = 4;
    let h = 4;
    let image = vec![0.5_f32; w * h * 3 + 1];
    let ao_buf = vec![1.0_f32; w * h];
    let result = ao_apply_to_image(&image, &ao_buf, w, h, 1.0);
    assert!(matches!(result, Err(AoError::DimensionMismatch { .. })));
}

#[test]
fn test_ao_apply_empty_input() {
    let result = ao_apply_to_image(&[], &[], 0, 0, 1.0);
    assert!(matches!(result, Err(AoError::EmptyInput)));
}

#[test]
fn test_ao_apply_output_length() {
    let w = 4;
    let h = 4;
    let image = vec![0.5_f32; w * h * 3];
    let ao_buf = vec![0.8_f32; w * h];
    let result = ao_apply_to_image(&image, &ao_buf, w, h, 1.0).expect("apply failed");
    assert_eq!(result.len(), image.len());
}

#[test]
fn test_ao_apply_partial_strength() {
    // With strength=0.5 and ao=0.0, pixel should be 0.5 (lerp between 1.0 and 0.0)
    let w = 1;
    let h = 1;
    let image = vec![1.0_f32, 1.0, 1.0];
    let ao_buf = vec![0.0_f32; 1];
    let result = ao_apply_to_image(&image, &ao_buf, w, h, 0.5).expect("apply failed");
    for &v in &result {
        assert!(
            approx_eq(v, 0.5, 1e-5),
            "Half strength should give 0.5, got {v}"
        );
    }
}

// ── apply_ssao tests ──────────────────────────────────────────────────────

#[test]
fn test_apply_ssao_all_background_image_unchanged() {
    let w = 8;
    let h = 8;
    let n = w * h;
    let depth_buf = vec![0.0_f32; n]; // all background
    let normal_buf = vec![0.0_f32; n * 3];
    let image = vec![0.6_f32; n * 3];
    let config = AoConfig::default();

    let result = apply_ssao(
        &image,
        &depth_buf,
        &normal_buf,
        w,
        h,
        AoProjParams::new(1.0, 1.0),
        &config,
    )
    .expect("apply_ssao failed");

    // All background → AO = 1.0 → image unchanged
    for (i, (&a, &b)) in image.iter().zip(result.image.iter()).enumerate() {
        assert!(
            approx_eq(a, b, 1e-4),
            "Background: image changed at pixel {i}: {a} != {b}"
        );
    }
}

#[test]
fn test_apply_ssao_valid_input_correct_dimensions() {
    let w = 8;
    let h = 8;
    let (depth_buf, normal_buf, image) = make_flat_scene(w, h, 2.0);
    let config = AoConfig::default();

    let result = apply_ssao(
        &image,
        &depth_buf,
        &normal_buf,
        w,
        h,
        AoProjParams::new(1.0, 1.0),
        &config,
    )
    .expect("apply_ssao failed");

    assert_eq!(result.image.len(), w * h * 3);
    assert_eq!(result.ao_map.len(), w * h);
    assert_eq!(result.ao_map_blurred.len(), w * h);
}

#[test]
fn test_apply_ssao_ao_maps_in_range() {
    let w = 8;
    let h = 8;
    let (depth_buf, normal_buf, image) = make_flat_scene(w, h, 1.5);
    let config = AoConfig::default();

    let result = apply_ssao(
        &image,
        &depth_buf,
        &normal_buf,
        w,
        h,
        AoProjParams::new(1.0, 1.0),
        &config,
    )
    .expect("apply_ssao failed");

    for (i, &v) in result.ao_map.iter().enumerate() {
        assert!((0.0..=1.0).contains(&v), "ao_map[{i}] = {v} out of [0,1]");
    }
    for (i, &v) in result.ao_map_blurred.iter().enumerate() {
        assert!(
            (0.0..=1.0).contains(&v),
            "ao_map_blurred[{i}] = {v} out of [0,1]"
        );
    }
}

#[test]
fn test_apply_ssao_invalid_config_error() {
    let w = 4;
    let h = 4;
    let (depth_buf, normal_buf, image) = make_flat_scene(w, h, 1.0);
    let config = AoConfig {
        n_samples: 0,
        ..Default::default()
    }; // invalid

    let result = apply_ssao(
        &image,
        &depth_buf,
        &normal_buf,
        w,
        h,
        AoProjParams::new(1.0, 1.0),
        &config,
    );
    assert!(result.is_err(), "Zero n_samples should fail");
}

// ── ao_compute_stats tests ────────────────────────────────────────────────

#[test]
fn test_ao_compute_stats_all_ones() {
    let ao_buf = vec![1.0_f32; 16];
    let depth_buf = vec![1.0_f32; 16];
    let stats = ao_compute_stats(&ao_buf, &depth_buf);

    assert!(approx_eq(stats.mean_ao, 1.0, 1e-5));
    assert!(approx_eq(stats.occlusion_fraction, 0.0, 1e-6));
}

#[test]
fn test_ao_compute_stats_all_zero_ao() {
    let n = 16;
    let ao_buf = vec![0.0_f32; n];
    let depth_buf = vec![1.0_f32; n]; // all foreground
    let stats = ao_compute_stats(&ao_buf, &depth_buf);

    assert!(
        approx_eq(stats.occlusion_fraction, 1.0, 1e-5),
        "All zero AO should have 100% occlusion fraction, got {}",
        stats.occlusion_fraction
    );
}

#[test]
fn test_ao_compute_stats_background_fraction() {
    let n = 8;
    let mut depth_buf = vec![1.0_f32; n];
    // Set half as background
    for d in depth_buf.iter_mut().take(n / 2) {
        *d = 0.0;
    }
    let ao_buf = vec![1.0_f32; n];
    let stats = ao_compute_stats(&ao_buf, &depth_buf);

    assert!(
        approx_eq(stats.background_fraction, 0.5, 1e-5),
        "Half background, got {}",
        stats.background_fraction
    );
}

#[test]
fn test_ao_compute_stats_empty() {
    let stats = ao_compute_stats(&[], &[]);
    assert!(approx_eq(stats.mean_ao, 1.0, 1e-6));
    assert!(approx_eq(stats.occlusion_fraction, 0.0, 1e-6));
    assert!(approx_eq(stats.background_fraction, 0.0, 1e-6));
}

#[test]
fn test_ao_compute_stats_min_max() {
    let ao_buf = vec![0.2_f32, 0.5, 0.8, 0.9];
    let depth_buf = vec![1.0_f32; 4];
    let stats = ao_compute_stats(&ao_buf, &depth_buf);

    assert!(approx_eq(stats.min_ao, 0.2, 1e-5));
    assert!(approx_eq(stats.max_ao, 0.9, 1e-5));
}

// ── format helpers tests ──────────────────────────────────────────────────

#[test]
fn test_format_ao_config_non_empty() {
    let config = AoConfig::default();
    let s = format_ao_config(&config);
    assert!(
        !s.is_empty(),
        "format_ao_config should return non-empty string"
    );
    assert!(s.contains("n_samples"), "Should contain n_samples");
    assert!(s.contains("radius"), "Should contain radius");
}

#[test]
fn test_format_ao_stats_non_empty() {
    let stats = AoStats {
        mean_ao: 0.8,
        min_ao: 0.3,
        max_ao: 1.0,
        occlusion_fraction: 0.2,
        background_fraction: 0.1,
    };
    let s = format_ao_stats(&stats);
    assert!(
        !s.is_empty(),
        "format_ao_stats should return non-empty string"
    );
    assert!(s.contains("mean"), "Should contain mean");
}

// ── Single pixel edge cases ───────────────────────────────────────────────

#[test]
fn test_ao_compute_single_pixel() {
    let depth_buf = vec![1.0_f32];
    let normal_buf = vec![0.0_f32, 0.0, -1.0];
    let kernel = ao_sample_kernel(4).expect("kernel failed");
    let noise = ao_noise_texture();
    let config = AoConfig::default();

    let ao = ao_compute(
        &depth_buf,
        &normal_buf,
        1,
        1,
        AoSamplingBuffers {
            kernel: &kernel,
            noise: &noise,
        },
        AoProjParams::new(1.0, 1.0),
        &config,
    )
    .expect("ao_compute failed");

    assert_eq!(ao.len(), 1);
    assert!(ao[0] >= 0.0 && ao[0] <= 1.0);
}

#[test]
fn test_apply_ssao_1x1_image() {
    let depth_buf = vec![1.0_f32];
    let normal_buf = vec![0.0_f32, 0.0, -1.0];
    let image = vec![0.5_f32, 0.5, 0.5];
    let config = AoConfig::default();

    let result = apply_ssao(
        &image,
        &depth_buf,
        &normal_buf,
        1,
        1,
        AoProjParams::new(1.0, 1.0),
        &config,
    )
    .expect("apply_ssao on 1x1 failed");

    assert_eq!(result.image.len(), 3);
    assert_eq!(result.ao_map.len(), 1);
}
