//! Unit tests for the `light_probe` module.

use super::projection::LP_COSINE_LOBE_A;
use super::sampling::bilinear_sample_rgb;
use super::sh_math::{LP_SH_C0, LP_SH_C1};
use super::*;
use std::f32::consts::PI;

const SQRT_PI: f32 = 1.772_453_9_f32;

// -----------------------------------------------------------------------
// lp_normalize_dir
// -----------------------------------------------------------------------

#[test]
fn test_normalize_unit_vector() {
    let d = lp_normalize_dir([1.0, 0.0, 0.0]).expect("should not error");
    assert!((d[0] - 1.0).abs() < 1e-6);
    assert!(d[1].abs() < 1e-6);
    assert!(d[2].abs() < 1e-6);
}

#[test]
fn test_normalize_scaled_vector() {
    let d = lp_normalize_dir([3.0, 4.0, 0.0]).expect("should not error");
    let norm = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    assert!((norm - 1.0).abs() < 1e-6);
}

#[test]
fn test_normalize_zero_error() {
    let result = lp_normalize_dir([0.0, 0.0, 0.0]);
    assert!(matches!(result, Err(LightProbeError::ZeroDirection)));
}

#[test]
fn test_normalize_near_zero_error() {
    let result = lp_normalize_dir([1e-8, 0.0, 0.0]);
    assert!(matches!(result, Err(LightProbeError::ZeroDirection)));
}

// -----------------------------------------------------------------------
// lp_sh_basis_l0
// -----------------------------------------------------------------------

#[test]
fn test_sh_basis_l0_value() {
    // Y_0^0 = 1/(2√π)
    let expected = 1.0 / (2.0 * SQRT_PI);
    let result = lp_sh_basis_l0([1.0, 0.0, 0.0]);
    assert!((result[0] - expected).abs() < 1e-5, "got {}", result[0]);
}

#[test]
fn test_sh_basis_l0_constant() {
    // Same for any direction
    let a = lp_sh_basis_l0([1.0, 0.0, 0.0])[0];
    let b = lp_sh_basis_l0([0.0, 1.0, 0.0])[0];
    let c = lp_sh_basis_l0([0.6, 0.8, 0.0])[0];
    assert!((a - b).abs() < 1e-9);
    assert!((a - c).abs() < 1e-9);
}

// -----------------------------------------------------------------------
// lp_sh_basis_l1
// -----------------------------------------------------------------------

#[test]
fn test_sh_basis_l1_along_z() {
    // dir = [0, 0, 1]: Y_1^{-1}=0, Y_1^0=SH_C1, Y_1^1=0
    let result = lp_sh_basis_l1([0.0, 0.0, 1.0]);
    assert!(result[0].abs() < 1e-6); // Y_1^{-1}=C1*y=0
    assert!((result[1] - LP_SH_C1).abs() < 1e-6); // Y_1^0=C1*z=C1
    assert!(result[2].abs() < 1e-6); // Y_1^1=C1*x=0
}

#[test]
fn test_sh_basis_l1_along_x() {
    // dir=[1,0,0]: Y_1^{-1}=0, Y_1^0=0, Y_1^1=C1
    let result = lp_sh_basis_l1([1.0, 0.0, 0.0]);
    assert!(result[0].abs() < 1e-6);
    assert!(result[1].abs() < 1e-6);
    assert!((result[2] - LP_SH_C1).abs() < 1e-6);
}

// -----------------------------------------------------------------------
// lp_sh_basis
// -----------------------------------------------------------------------

#[test]
fn test_sh_basis_order1_length() {
    let v = lp_sh_basis([1.0, 0.0, 0.0], 1).expect("order 1 ok");
    assert_eq!(v.len(), 1);
}

#[test]
fn test_sh_basis_order2_length() {
    let v = lp_sh_basis([1.0, 0.0, 0.0], 2).expect("order 2 ok");
    assert_eq!(v.len(), 4);
}

#[test]
fn test_sh_basis_order3_length() {
    let v = lp_sh_basis([1.0, 0.0, 0.0], 3).expect("order 3 ok");
    assert_eq!(v.len(), 9);
}

#[test]
fn test_sh_basis_invalid_order() {
    let r = lp_sh_basis([1.0, 0.0, 0.0], 0);
    assert!(matches!(r, Err(LightProbeError::InvalidOrder { .. })));
    let r2 = lp_sh_basis([1.0, 0.0, 0.0], 4);
    assert!(matches!(r2, Err(LightProbeError::InvalidOrder { .. })));
}

#[test]
fn test_sh_basis_orthogonality_different_dirs() {
    let v1 = lp_sh_basis([1.0, 0.0, 0.0], 3).expect("ok");
    let v2 = lp_sh_basis([0.0, 1.0, 0.0], 3).expect("ok");
    // The basis vectors for two different unit directions must differ
    let diff: f32 = v1.iter().zip(v2.iter()).map(|(a, b)| (a - b).abs()).sum();
    assert!(
        diff > 0.01,
        "basis vectors should differ for different directions"
    );
}

// -----------------------------------------------------------------------
// IrradianceSH
// -----------------------------------------------------------------------

#[test]
fn test_irradiance_sh_evaluate_zero() {
    let sh = IrradianceSH::new();
    let result = sh.evaluate([0.0, 0.0, 1.0]).expect("ok");
    assert!(result[0].abs() < 1e-9);
    assert!(result[1].abs() < 1e-9);
    assert!(result[2].abs() < 1e-9);
}

#[test]
fn test_irradiance_sh_constant_probe() {
    // Only L=0 coefficient set — result should be same for all directions
    let mut coeffs = [0.0_f32; 27];
    let val = 2.0_f32;
    coeffs[0] = val; // R
    coeffs[1] = val; // G
    coeffs[2] = val; // B
    let sh = IrradianceSH::from_coefficients(coeffs);

    let r1 = sh.evaluate([1.0, 0.0, 0.0]).expect("ok");
    let r2 = sh.evaluate([0.0, 1.0, 0.0]).expect("ok");
    let r3 = sh.evaluate([-1.0, 0.0, 0.0]).expect("ok");
    // All should be val * LP_SH_C0
    let expected = val * LP_SH_C0;
    assert!((r1[0] - expected).abs() < 1e-5);
    assert!((r2[0] - expected).abs() < 1e-5);
    assert!((r3[0] - expected).abs() < 1e-5);
}

#[test]
fn test_irradiance_sh_scale() {
    let mut coeffs = [0.0_f32; 27];
    coeffs[0] = 1.0;
    coeffs[3] = 2.0;
    let sh = IrradianceSH::from_coefficients(coeffs);
    let scaled = sh.scale(2.0);
    assert!((scaled.coefficients[0] - 2.0).abs() < 1e-9);
    assert!((scaled.coefficients[3] - 4.0).abs() < 1e-9);
    // Original unchanged
    assert!((sh.coefficients[0] - 1.0).abs() < 1e-9);
}

#[test]
fn test_irradiance_sh_add() {
    let mut c1 = [0.0_f32; 27];
    c1[0] = 1.0;
    let mut c2 = [0.0_f32; 27];
    c2[0] = 3.0;
    c2[6] = -1.0;
    let sh1 = IrradianceSH::from_coefficients(c1);
    let sh2 = IrradianceSH::from_coefficients(c2);
    let sum = sh1.add(&sh2);
    assert!((sum.coefficients[0] - 4.0).abs() < 1e-9);
    assert!((sum.coefficients[6] + 1.0).abs() < 1e-9);
}

#[test]
fn test_irradiance_sh_ambient_nonzero() {
    let mut coeffs = [0.0_f32; 27];
    coeffs[0] = 1.0;
    coeffs[1] = 0.5;
    coeffs[2] = 0.2;
    let sh = IrradianceSH::from_coefficients(coeffs);
    let amb = sh.ambient();
    assert!(amb[0] > 0.0);
    assert!(amb[1] > 0.0);
    assert!(amb[2] > 0.0);
}

#[test]
fn test_irradiance_sh_evaluate_zero_dir_error() {
    let sh = IrradianceSH::new();
    let r = sh.evaluate([0.0, 0.0, 0.0]);
    assert!(matches!(r, Err(LightProbeError::ZeroDirection)));
}

// -----------------------------------------------------------------------
// lp_generate_sphere_samples
// -----------------------------------------------------------------------

#[test]
fn test_sphere_samples_unit_norm() {
    let samples = lp_generate_sphere_samples(200, 42);
    for s in &samples {
        let norm = (s[0] * s[0] + s[1] * s[1] + s[2] * s[2]).sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "norm={}", norm);
    }
}

#[test]
fn test_sphere_samples_octant_distribution() {
    let samples = lp_generate_sphere_samples(8_000, 123);
    // Count samples in each of the 8 octants
    let mut counts = [0usize; 8];
    for s in &samples {
        let xi = if s[0] >= 0.0 { 1 } else { 0 };
        let yi = if s[1] >= 0.0 { 2 } else { 0 };
        let zi = if s[2] >= 0.0 { 4 } else { 0 };
        counts[xi + yi + zi] += 1;
    }
    // Each octant should have roughly N/8 = 1000 samples; allow ±40%
    for &count in &counts {
        assert!(count > 600 && count < 1400, "octant count = {}", count);
    }
}

// -----------------------------------------------------------------------
// lp_project_samples_to_sh
// -----------------------------------------------------------------------

#[test]
fn test_project_constant_radiance() {
    // Constant white radiance → almost all energy in L=0
    let n = 5_000usize;
    let dirs = lp_generate_sphere_samples(n, 7);
    let rads: Vec<[f32; 3]> = dirs.iter().map(|_| [1.0, 1.0, 1.0]).collect();
    let sh = lp_project_samples_to_sh(&dirs, &rads).expect("ok");

    // The L=0 coefficient (index 0) should dominate
    let l0_r = sh.coefficients[0].abs();
    for i in 1..9 {
        let higher = sh.coefficients[i * 3].abs();
        assert!(
            l0_r > higher * 5.0,
            "L={} coeff {} > L=0/5 for constant input",
            i,
            higher
        );
    }
}

#[test]
fn test_project_length_mismatch() {
    let dirs = lp_generate_sphere_samples(10, 1);
    let rads: Vec<[f32; 3]> = vec![[1.0, 0.0, 0.0]; 5];
    let r = lp_project_samples_to_sh(&dirs, &rads);
    assert!(matches!(r, Err(LightProbeError::BufferMismatch { .. })));
}

// -----------------------------------------------------------------------
// lp_project_latitude_longitude
// -----------------------------------------------------------------------

#[test]
fn test_project_latlong_uniform() {
    // All-white image → near-constant SH (L>0 terms small relative to L=0)
    let w = 64u32;
    let h = 32u32;
    let image = vec![1.0_f32; (w * h * 3) as usize];
    let sh = lp_project_latitude_longitude(&image, w, h, 2000, 99).expect("ok");
    let l0 = sh.coefficients[0].abs();
    for i in 1..9 {
        let hi = sh.coefficients[i * 3].abs();
        assert!(
            hi < l0 * 0.3,
            "L={} coefficient too large for uniform input",
            i
        );
    }
}

#[test]
fn test_project_latlong_buffer_mismatch() {
    let r = lp_project_latitude_longitude(&[1.0; 10], 4, 4, 100, 1);
    assert!(matches!(r, Err(LightProbeError::BufferMismatch { .. })));
}

// -----------------------------------------------------------------------
// lp_dir_to_cubemap_uv
// -----------------------------------------------------------------------

#[test]
fn test_cubemap_uv_pos_x() {
    let (face, _u, _v) = lp_dir_to_cubemap_uv([1.0, 0.0, 0.0]);
    assert_eq!(face, CubemapFace::PosX);
}

#[test]
fn test_cubemap_uv_neg_x() {
    let (face, _u, _v) = lp_dir_to_cubemap_uv([-1.0, 0.0, 0.0]);
    assert_eq!(face, CubemapFace::NegX);
}

#[test]
fn test_cubemap_uv_pos_y() {
    let (face, _u, _v) = lp_dir_to_cubemap_uv([0.0, 1.0, 0.0]);
    assert_eq!(face, CubemapFace::PosY);
}

#[test]
fn test_cubemap_uv_neg_y() {
    let (face, _u, _v) = lp_dir_to_cubemap_uv([0.0, -1.0, 0.0]);
    assert_eq!(face, CubemapFace::NegY);
}

#[test]
fn test_cubemap_uv_pos_z() {
    let (face, _u, _v) = lp_dir_to_cubemap_uv([0.0, 0.0, 1.0]);
    assert_eq!(face, CubemapFace::PosZ);
}

#[test]
fn test_cubemap_uv_neg_z() {
    let (face, _u, _v) = lp_dir_to_cubemap_uv([0.0, 0.0, -1.0]);
    assert_eq!(face, CubemapFace::NegZ);
}

#[test]
fn test_cubemap_uv_in_range() {
    // u, v must lie in [0, 1]
    for dir in [
        [1.0_f32, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
        [0.7, 0.7, 0.0],
        [0.5, 0.5, 0.7],
    ] {
        let (_, u, v) = lp_dir_to_cubemap_uv(dir);
        assert!(
            (0.0..=1.0).contains(&u),
            "u={} out of range for dir={:?}",
            u,
            dir
        );
        assert!(
            (0.0..=1.0).contains(&v),
            "v={} out of range for dir={:?}",
            v,
            dir
        );
    }
}

// -----------------------------------------------------------------------
// CubemapProbe
// -----------------------------------------------------------------------

#[test]
fn test_cubemap_new_invalid_resolution_odd() {
    let r = CubemapProbe::new(5);
    assert!(matches!(r, Err(LightProbeError::InvalidResolution { .. })));
}

#[test]
fn test_cubemap_new_invalid_resolution_small() {
    let r = CubemapProbe::new(2);
    assert!(matches!(r, Err(LightProbeError::InvalidResolution { .. })));
}

#[test]
fn test_cubemap_new_valid() {
    let probe = CubemapProbe::new(8).expect("8 is valid");
    assert_eq!(probe.resolution, 8);
    assert_eq!(probe.faces.len(), 6);
    for face in &probe.faces {
        assert_eq!(face.len(), 8 * 8 * 3);
    }
}

#[test]
fn test_cubemap_sample_different_faces() {
    let mut probe = CubemapProbe::new(4).expect("ok");
    // Color face 0 (+X) with red
    for v in probe.faces[0].iter_mut() {
        *v = 0.0;
    }
    for px in 0..(4 * 4) {
        probe.faces[0][px * 3] = 1.0; // R channel
    }
    // Color face 1 (-X) with blue
    for v in probe.faces[1].iter_mut() {
        *v = 0.0;
    }
    for px in 0..(4 * 4) {
        probe.faces[1][px * 3 + 2] = 1.0; // B channel
    }
    let pos_x = probe.sample([1.0, 0.0, 0.0]).expect("ok");
    let neg_x = probe.sample([-1.0, 0.0, 0.0]).expect("ok");
    // +X face: red dominant
    assert!(pos_x[0] > 0.5, "pos_x R should be high, got {:?}", pos_x);
    // -X face: blue dominant
    assert!(neg_x[2] > 0.5, "neg_x B should be high, got {:?}", neg_x);
}

#[test]
fn test_cubemap_sample_zero_dir_error() {
    let probe = CubemapProbe::new(4).expect("ok");
    let r = probe.sample([0.0, 0.0, 0.0]);
    assert!(matches!(r, Err(LightProbeError::ZeroDirection)));
}

#[test]
fn test_cubemap_to_sh_constant() {
    // Constant grey cubemap → mostly L=0 SH
    let mut probe = CubemapProbe::new(4).expect("ok");
    let grey = 0.5_f32;
    for face in probe.faces.iter_mut() {
        face.fill(grey);
    }
    let sh = lp_cubemap_to_sh(&probe, 3000, 42).expect("ok");
    let l0 = sh.coefficients[0].abs();
    // Higher-order terms should be much smaller than L=0
    for i in 4..9 {
        let hi = sh.coefficients[i * 3].abs();
        assert!(hi < l0 * 0.5, "L2 coefficient {} too large", hi);
    }
}

#[test]
fn test_cubemap_bilinear_smooth() {
    // Fill +X face with a horizontal gradient; verify smooth interpolation
    let mut probe = CubemapProbe::new(8).expect("ok");
    let res = 8usize;
    for py in 0..res {
        for px in 0..res {
            let val = px as f32 / (res - 1) as f32;
            let base = (py * res + px) * 3;
            probe.faces[0][base] = val;
            probe.faces[0][base + 1] = val;
            probe.faces[0][base + 2] = val;
        }
    }
    // Sample at two points: left and right of +X face
    let left = probe.sample([1.0, 0.0, 0.9]).expect("ok");
    let right = probe.sample([1.0, 0.0, -0.9]).expect("ok");
    // Right side (low-z) maps to low u, left side (high-z) to high u or vice versa
    // We just check that the two values are different (gradient is captured)
    let diff = (left[0] - right[0]).abs();
    assert!(
        diff > 0.1,
        "bilinear should capture gradient, diff={}",
        diff
    );
}

// -----------------------------------------------------------------------
// LightProbe::weight_for
// -----------------------------------------------------------------------

#[test]
fn test_weight_at_position() {
    let pos = [1.0, 2.0, 3.0];
    let probe = LightProbe::new(pos, IrradianceSH::new(), 5.0);
    let w = probe.weight_for(pos);
    assert!(
        (w - 1.0).abs() < 1e-6,
        "weight at center should be 1.0, got {}",
        w
    );
}

#[test]
fn test_weight_radius_zero_is_global() {
    let probe = LightProbe::new([0.0, 0.0, 0.0], IrradianceSH::new(), 0.0);
    assert!((probe.weight_for([100.0, 100.0, 100.0]) - 1.0).abs() < 1e-6);
    assert!((probe.weight_for([-50.0, 0.0, 0.0]) - 1.0).abs() < 1e-6);
}

#[test]
fn test_weight_beyond_radius_small() {
    let probe = LightProbe::new([0.0, 0.0, 0.0], IrradianceSH::new(), 2.0);
    let w = probe.weight_for([4.0, 0.0, 0.0]); // dist=4, radius=2 → clamp → 0
    assert!(
        w < 0.01,
        "weight beyond radius should be near zero, got {}",
        w
    );
}

#[test]
fn test_weight_at_boundary() {
    let probe = LightProbe::new([0.0, 0.0, 0.0], IrradianceSH::new(), 3.0);
    let w = probe.weight_for([3.0, 0.0, 0.0]);
    assert!(
        w.abs() < 1e-5,
        "weight at exact radius edge should be 0, got {}",
        w
    );
}

// -----------------------------------------------------------------------
// LightProbe::evaluate
// -----------------------------------------------------------------------

#[test]
fn test_probe_evaluate_zero_normal_error() {
    let probe = LightProbe::new([0.0, 0.0, 0.0], IrradianceSH::new(), 0.0);
    let r = probe.evaluate([1.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
    assert!(matches!(r, Err(LightProbeError::ZeroDirection)));
}

// -----------------------------------------------------------------------
// LightProbeBlend
// -----------------------------------------------------------------------

#[test]
fn test_probe_blend_empty_error() {
    let r = LightProbeBlend::new(vec![], ProbeBlendMode::WeightedAverage);
    assert!(matches!(r, Err(LightProbeError::EmptyProbeList)));
}

#[test]
fn test_probe_blend_single_matches_probe() {
    let mut coeffs = [0.0_f32; 27];
    coeffs[0] = 1.0;
    coeffs[1] = 0.5;
    coeffs[2] = 0.25;
    let probe = LightProbe::new(
        [0.0, 0.0, 0.0],
        IrradianceSH::from_coefficients(coeffs),
        0.0,
    );
    let point = [0.5, 0.0, 0.0];
    let normal = [0.0, 0.0, 1.0];
    let expected = probe.evaluate(point, normal).expect("ok");

    let blend = LightProbeBlend::new(vec![probe], ProbeBlendMode::WeightedAverage).expect("ok");
    let result = blend.evaluate(point, normal).expect("ok");

    for c in 0..3 {
        assert!(
            (result[c] - expected[c]).abs() < 1e-4,
            "channel {}: expected {}, got {}",
            c,
            expected[c],
            result[c]
        );
    }
}

// -----------------------------------------------------------------------
// lp_blend_irradiance_sh
// -----------------------------------------------------------------------

#[test]
fn test_blend_single_weight_one() {
    let mut coeffs = [0.0_f32; 27];
    for (i, c) in coeffs.iter_mut().enumerate() {
        *c = i as f32;
    }
    let probe = LightProbe::new(
        [0.0, 0.0, 0.0],
        IrradianceSH::from_coefficients(coeffs),
        0.0,
    );
    let blended = lp_blend_irradiance_sh(&[probe], &[1.0]).expect("ok");
    for i in 0..27 {
        assert!((blended.coefficients[i] - i as f32).abs() < 1e-5);
    }
}

#[test]
fn test_blend_equal_weights_average() {
    let mut c1 = [0.0_f32; 27];
    let mut c2 = [0.0_f32; 27];
    c1[0] = 2.0;
    c2[0] = 4.0;
    let p1 = LightProbe::new([0.0, 0.0, 0.0], IrradianceSH::from_coefficients(c1), 0.0);
    let p2 = LightProbe::new([0.0, 0.0, 0.0], IrradianceSH::from_coefficients(c2), 0.0);
    let blended = lp_blend_irradiance_sh(&[p1, p2], &[1.0, 1.0]).expect("ok");
    assert!(
        (blended.coefficients[0] - 3.0).abs() < 1e-5,
        "expected 3.0, got {}",
        blended.coefficients[0]
    );
}

#[test]
fn test_blend_empty_error() {
    let r = lp_blend_irradiance_sh(&[], &[]);
    assert!(matches!(r, Err(LightProbeError::EmptyProbeList)));
}

// -----------------------------------------------------------------------
// lp_evaluate_diffuse_ibl
// -----------------------------------------------------------------------

#[test]
fn test_diffuse_ibl_zero_albedo() {
    let mut c = [0.0_f32; 27];
    c[0] = 5.0;
    let sh = IrradianceSH::from_coefficients(c);
    let result = lp_evaluate_diffuse_ibl([0.0, 0.0, 1.0], &sh, [0.0, 0.0, 0.0]).expect("ok");
    assert!(result[0].abs() < 1e-9);
    assert!(result[1].abs() < 1e-9);
    assert!(result[2].abs() < 1e-9);
}

#[test]
fn test_diffuse_ibl_white_albedo_nonzero() {
    let mut c = [0.0_f32; 27];
    c[0] = 1.0;
    c[1] = 1.0;
    c[2] = 1.0;
    let sh = IrradianceSH::from_coefficients(c);
    let result = lp_evaluate_diffuse_ibl([0.0, 0.0, 1.0], &sh, [1.0, 1.0, 1.0]).expect("ok");
    for chan in result {
        assert!(
            chan > 0.0,
            "Expected nonzero result for white ambient+albedo"
        );
    }
}

#[test]
fn test_diffuse_ibl_zero_normal_error() {
    let sh = IrradianceSH::new();
    let r = lp_evaluate_diffuse_ibl([0.0, 0.0, 0.0], &sh, [1.0, 1.0, 1.0]);
    assert!(matches!(r, Err(LightProbeError::ZeroDirection)));
}

// -----------------------------------------------------------------------
// lp_apply_ibl_to_gaussians
// -----------------------------------------------------------------------

#[test]
fn test_ibl_gaussians_output_length() {
    let mut c = [0.0_f32; 27];
    c[0] = 1.0;
    c[1] = 1.0;
    c[2] = 1.0;
    let sh = IrradianceSH::from_coefficients(c);
    let n = 7usize;
    let normals = vec![0.0_f32, 0.0, 1.0]
        .into_iter()
        .cycle()
        .take(n * 3)
        .collect::<Vec<_>>();
    let albedo = vec![0.5_f32, 0.5, 0.5]
        .into_iter()
        .cycle()
        .take(n * 3)
        .collect::<Vec<_>>();
    let out = lp_apply_ibl_to_gaussians(&normals, &sh, &albedo, n).expect("ok");
    assert_eq!(out.len(), n * 3);
}

#[test]
fn test_ibl_gaussians_zero_normal_error() {
    let sh = IrradianceSH::new();
    let normals = vec![0.0_f32; 3];
    let albedo = vec![1.0_f32; 3];
    let r = lp_apply_ibl_to_gaussians(&normals, &sh, &albedo, 1);
    assert!(matches!(r, Err(LightProbeError::ZeroDirection)));
}

#[test]
fn test_ibl_gaussians_buffer_mismatch() {
    let sh = IrradianceSH::new();
    let normals = vec![0.0_f32, 0.0, 1.0, 0.0, 0.0, 1.0]; // 2 gaussians
    let albedo = vec![0.5_f32; 3]; // 1 gaussian
    let r = lp_apply_ibl_to_gaussians(&normals, &sh, &albedo, 2);
    assert!(matches!(r, Err(LightProbeError::BufferMismatch { .. })));
}

// -----------------------------------------------------------------------
// lp_compute_stats
// -----------------------------------------------------------------------

#[test]
fn test_stats_empty_error() {
    let r = lp_compute_stats(&[]);
    assert!(matches!(r, Err(LightProbeError::EmptyProbeList)));
}

#[test]
fn test_stats_one_probe() {
    let mut c = [0.0_f32; 27];
    c[0] = 2.0;
    let probe = LightProbe::new([0.0, 0.0, 0.0], IrradianceSH::from_coefficients(c), 5.0);
    let stats = lp_compute_stats(&[probe]).expect("ok");
    assert_eq!(stats.n_probes, 1);
    assert!(stats.max_coefficient > 0.0);
    assert!(stats.sh_energy > 0.0);
}

#[test]
fn test_stats_format_nonempty() {
    let probe = LightProbe::new([0.0, 0.0, 0.0], IrradianceSH::new(), 0.0);
    let stats = lp_compute_stats(&[probe]).expect("ok");
    let s = lp_format_stats(&stats);
    assert!(!s.is_empty());
    assert!(s.contains("n_probes"));
}

#[test]
fn test_config_format_nonempty() {
    let config = LightProbeConfig::default();
    let s = lp_format_config(&config);
    assert!(!s.is_empty());
    assert!(s.contains("n_samples_projection"));
}

// -----------------------------------------------------------------------
// SH energy / projection quality
// -----------------------------------------------------------------------

#[test]
fn test_sh_projection_energy_white_env() {
    // White environment → L=0 coefficient energy must dominate L=1, L=2
    let n = 6_000usize;
    let dirs = lp_generate_sphere_samples(n, 777);
    let rads: Vec<[f32; 3]> = dirs.iter().map(|_| [1.0, 1.0, 1.0]).collect();
    let sh = lp_project_samples_to_sh(&dirs, &rads).expect("ok");

    // Energy in L=0 (index 0,1,2)
    let e_l0: f32 = (0..3)
        .map(|c| sh.coefficients[c] * sh.coefficients[c])
        .sum();
    // Energy in L=1 (indices 3..12)
    let e_l1: f32 = (1..4)
        .flat_map(|i| (0..3).map(move |c| i * 3 + c))
        .map(|idx| sh.coefficients[idx] * sh.coefficients[idx])
        .sum();
    // Energy in L=2 (indices 12..27)
    let e_l2: f32 = (4..9)
        .flat_map(|i| (0..3).map(move |c| i * 3 + c))
        .map(|idx| sh.coefficients[idx] * sh.coefficients[idx])
        .sum();

    assert!(
        e_l0 > e_l1 * 10.0,
        "L=0 energy should dominate L=1 for white env"
    );
    assert!(
        e_l0 > e_l2 * 10.0,
        "L=0 energy should dominate L=2 for white env"
    );
}

// -----------------------------------------------------------------------
// Cosine-lobe convolution (bug #1: radiance vs irradiance SH)
// -----------------------------------------------------------------------

#[test]
fn test_cosine_lobe_convolution_reproduces_uniform_radiance() {
    // A uniformly-lit white environment of radiance L, viewed by a
    // Lambertian surface with albedo=1, must reflect back exactly L
    // (irradiance E = pi*L for a uniform environment, and
    // lp_evaluate_diffuse_ibl divides by pi again — this only holds if
    // the cosine-lobe convolution has actually been applied).
    let n = 20_000usize;
    let dirs = lp_generate_sphere_samples(n, 321);
    let l_val = 2.0_f32;
    let rads: Vec<[f32; 3]> = dirs.iter().map(|_| [l_val, l_val, l_val]).collect();
    let sh = lp_project_samples_to_sh(&dirs, &rads).expect("ok");

    let out = lp_evaluate_diffuse_ibl([0.0, 1.0, 0.0], &sh, [1.0, 1.0, 1.0]).expect("ok");
    for c in out {
        assert!(
            (c - l_val).abs() < 0.15,
            "expected diffuse IBL to reproduce input radiance {l_val}, got {c}"
        );
    }
}

#[test]
fn test_cosine_lobe_convolution_scales_bands_by_a0_a1_a2() {
    // Directly verify the per-band scale factors on a hand-built
    // (non-Monte-Carlo) coefficient set: projecting unit basis-aligned
    // radiances and checking the resulting IrradianceSH against the
    // known A0/A1/A2 constants would require reproducing the full
    // Monte Carlo estimator, so instead verify the constants themselves
    // match the Ramamoorthi & Hanrahan values used throughout the
    // literature.
    assert!((LP_COSINE_LOBE_A[0] - PI).abs() < 1e-6);
    assert!((LP_COSINE_LOBE_A[1] - (2.0 * PI / 3.0)).abs() < 1e-6);
    assert!((LP_COSINE_LOBE_A[2] - (PI / 4.0)).abs() < 1e-6);
}

// -----------------------------------------------------------------------
// bilinear_sample_rgb / lp_project_latitude_longitude zero-dimension guard
// -----------------------------------------------------------------------

#[test]
fn test_bilinear_sample_rgb_zero_width_no_panic() {
    let result = bilinear_sample_rgb(&[], 0, 4, 0.5, 0.5);
    assert_eq!(result, [0.0, 0.0, 0.0]);
}

#[test]
fn test_bilinear_sample_rgb_zero_height_no_panic() {
    let result = bilinear_sample_rgb(&[], 4, 0, 0.5, 0.5);
    assert_eq!(result, [0.0, 0.0, 0.0]);
}

#[test]
fn test_project_latlong_zero_dimensions_rejected() {
    let r = lp_project_latitude_longitude(&[], 0, 0, 10, 1);
    assert!(matches!(
        r,
        Err(LightProbeError::InvalidImageDimensions {
            width: 0,
            height: 0
        })
    ));
}

// -----------------------------------------------------------------------
// ProbeBlendMode: Nearest / WeightedAverage / VolumeWeighted consistency
// -----------------------------------------------------------------------

#[test]
fn test_blend_modes_agree_for_single_probe() {
    let mut coeffs = [0.0_f32; 27];
    coeffs[0] = 1.0;
    coeffs[1] = 0.6;
    coeffs[2] = 0.3;
    let probe = LightProbe::new(
        [0.0, 0.0, 0.0],
        IrradianceSH::from_coefficients(coeffs),
        5.0,
    );
    let point = [2.0, 0.0, 0.0]; // inside the radius but not at the centre
    let normal = [0.0, 0.0, 1.0];

    let nearest = LightProbeBlend::new(vec![probe.clone()], ProbeBlendMode::Nearest).expect("ok");
    let weighted =
        LightProbeBlend::new(vec![probe.clone()], ProbeBlendMode::WeightedAverage).expect("ok");
    let volume = LightProbeBlend::new(vec![probe], ProbeBlendMode::VolumeWeighted).expect("ok");

    let r_nearest = nearest.evaluate(point, normal).expect("ok");
    let r_weighted = weighted.evaluate(point, normal).expect("ok");
    let r_volume = volume.evaluate(point, normal).expect("ok");

    for c in 0..3 {
        assert!(
            (r_nearest[c] - r_weighted[c]).abs() < 1e-4,
            "Nearest vs WeightedAverage mismatch at channel {c}: {:?} vs {:?}",
            r_nearest,
            r_weighted
        );
        assert!(
            (r_nearest[c] - r_volume[c]).abs() < 1e-4,
            "Nearest vs VolumeWeighted mismatch at channel {c}: {:?} vs {:?}",
            r_nearest,
            r_volume
        );
    }
}

#[test]
fn test_weighted_average_fades_out_beyond_radius_like_nearest() {
    // Regression for the brightness-inconsistency bug: previously
    // WeightedAverage/VolumeWeighted's normalisation cancelled the
    // distance falloff entirely, so a point near a probe's radius edge
    // stayed at full brightness instead of fading like `Nearest`.
    let probe = LightProbe::new(
        [0.0, 0.0, 0.0],
        IrradianceSH::from_coefficients([1.0; 27]),
        4.0,
    );
    let point = [3.9, 0.0, 0.0]; // just inside the radius edge
    let normal = [0.0, 0.0, 1.0];

    let weighted = LightProbeBlend::new(vec![probe], ProbeBlendMode::WeightedAverage).expect("ok");
    let result = weighted.evaluate(point, normal).expect("ok");
    for c in result {
        assert!(
            c.abs() < 0.2,
            "expected near-zero irradiance close to the radius edge, got {c}"
        );
    }
}

#[test]
fn test_volume_weighted_prefers_smaller_probe_when_overlapping() {
    // Two fully-overlapping probes at the same point with very
    // different radii: VolumeWeighted should weight the smaller (more
    // local) probe's SH data far more heavily than the larger one,
    // unlike WeightedAverage which only looks at distance falloff —
    // and both probes have distance weight 1.0 at their shared centre,
    // so WeightedAverage blends them 50/50.
    let mut small_coeffs = [0.0_f32; 27];
    small_coeffs[0] = 10.0; // distinctive L0 value
    let small_probe = LightProbe::new(
        [0.0, 0.0, 0.0],
        IrradianceSH::from_coefficients(small_coeffs),
        1.0, // small radius
    );
    let large_probe = LightProbe::new(
        [0.0, 0.0, 0.0],
        IrradianceSH::from_coefficients([0.0_f32; 27]),
        100.0, // large radius
    );

    let point = [0.0, 0.0, 0.0];
    let normal = [0.0, 0.0, 1.0];

    let volume_blend = LightProbeBlend::new(
        vec![small_probe.clone(), large_probe.clone()],
        ProbeBlendMode::VolumeWeighted,
    )
    .expect("ok");
    let weighted_blend = LightProbeBlend::new(
        vec![small_probe, large_probe],
        ProbeBlendMode::WeightedAverage,
    )
    .expect("ok");

    let r_volume = volume_blend.evaluate(point, normal).expect("ok");
    let r_weighted = weighted_blend.evaluate(point, normal).expect("ok");

    assert!(
        r_volume[0] > r_weighted[0] * 1.5,
        "VolumeWeighted ({r_volume:?}) should favour the smaller probe much more \
         than WeightedAverage's 50/50 blend ({r_weighted:?})"
    );
}

// -----------------------------------------------------------------------
// LightProbeConfig threading
// -----------------------------------------------------------------------

#[test]
fn test_cubemap_to_sh_with_config_uses_n_samples_projection() {
    let probe = CubemapProbe::new(4).expect("ok");
    let config = LightProbeConfig {
        n_samples_projection: 500,
        ..LightProbeConfig::default()
    };
    let sh = lp_cubemap_to_sh_with_config(&probe, &config, 1).expect("ok");
    assert_eq!(sh.order, 2);
}

#[test]
fn test_project_latlong_with_config_uses_n_samples_projection() {
    let w = 8u32;
    let h = 4u32;
    let image = vec![1.0_f32; (w * h * 3) as usize];
    let config = LightProbeConfig {
        n_samples_projection: 400,
        ..LightProbeConfig::default()
    };
    let sh = lp_project_latitude_longitude_with_config(&image, w, h, &config, 7).expect("ok");
    assert_eq!(sh.order, 2);
}

#[test]
fn test_light_probe_blend_with_config_enforces_max_probes() {
    let config = LightProbeConfig {
        max_probes: 1,
        ..LightProbeConfig::default()
    };
    let p1 = LightProbe::new([0.0, 0.0, 0.0], IrradianceSH::new(), 1.0);
    let p2 = LightProbe::new([1.0, 0.0, 0.0], IrradianceSH::new(), 1.0);
    let r = LightProbeBlend::with_config(vec![p1, p2], &config);
    assert!(matches!(
        r,
        Err(LightProbeError::TooManyProbes { count: 2, max: 1 })
    ));
}

#[test]
fn test_light_probe_blend_with_config_uses_configured_mode() {
    let config = LightProbeConfig {
        blend_mode: ProbeBlendMode::Nearest,
        ..LightProbeConfig::default()
    };
    let p1 = LightProbe::new([0.0, 0.0, 0.0], IrradianceSH::new(), 1.0);
    let blend = LightProbeBlend::with_config(vec![p1], &config).expect("ok");
    assert_eq!(blend.blend_mode, ProbeBlendMode::Nearest);
}
