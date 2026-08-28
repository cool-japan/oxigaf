//! Auto-generated test module (consolidated from inline `#[cfg(test)] mod` blocks)

use super::*;

#[cfg(test)]
mod tests_2 {
    use super::*;

    const EPS: f32 = 1e-5;

    // ── 1. reinhard: 0 → 0, large x → approaches 1 ──────────────────────────
    #[test]
    fn test_reinhard_zero() {
        assert!((reinhard(0.0) - 0.0).abs() < EPS);
    }

    #[test]
    fn test_reinhard_large_approaches_one() {
        let out = reinhard(1_000_000.0_f32);
        assert!(out > 0.999_99, "expected > 0.99999, got {out}");
        assert!(out <= 1.0, "expected <= 1.0, got {out}");
    }

    // ── 2. reinhard: 1.0 → 0.5 ───────────────────────────────────────────────
    #[test]
    fn test_reinhard_one_half() {
        let out = reinhard(1.0);
        assert!((out - 0.5).abs() < EPS, "expected 0.5, got {out}");
    }

    // ── 3. reinhard_extended: white point semantics ───────────────────────────
    #[test]
    fn test_reinhard_extended_at_white() {
        // At x=white the extended Reinhard gives (w*(1+1))/(1+w) = 2w/(1+w).
        // For w=1: 2/2 = 1.0 exactly (maps white → 1).
        let out = reinhard_extended(1.0, 1.0);
        assert!(out <= 1.0 + EPS, "expected <= 1, got {out}");
        assert!(out > 0.0, "expected > 0, got {out}");
    }

    // ── 4. aces_filmic: 0 → 0, 1 → reasonable ────────────────────────────────
    #[test]
    fn test_aces_filmic_zero() {
        assert!((aces_filmic(0.0) - 0.0).abs() < EPS);
    }

    #[test]
    fn test_aces_filmic_one_in_range() {
        let out = aces_filmic(1.0);
        assert!(out > 0.0 && out <= 1.0, "expected in (0,1], got {out}");
    }

    // ── 5. aces_filmic: output always clamped to [0,1] ───────────────────────
    #[test]
    fn test_aces_filmic_clamped() {
        for &v in &[-1.0_f32, 0.0, 0.5, 1.0, 10.0, 100.0] {
            let out = aces_filmic(v);
            assert!(
                (0.0..=1.0).contains(&out),
                "aces_filmic({v}) = {out} out of [0,1]"
            );
        }
    }

    // ── 6. hable: 0 → 0, large → approaches 1 ────────────────────────────────
    #[test]
    fn test_hable_zero() {
        assert!((hable(0.0, 2.0) - 0.0).abs() < EPS);
    }

    #[test]
    fn test_hable_large_approaches_one() {
        let out = hable(1000.0, 2.0);
        assert!(out > 0.9, "expected > 0.9, got {out}");
        assert!(out <= 1.0, "expected <= 1.0, got {out}");
    }

    // ── 7. filmic: negative input → 0 ────────────────────────────────────────
    #[test]
    fn test_filmic_negative_is_zero() {
        assert!((filmic(-5.0) - 0.0).abs() < EPS);
        assert!((filmic(-0.001) - 0.0).abs() < EPS);
    }

    // ── 8. gamma_correct: x=0 → 0, x=1 → 1 ──────────────────────────────────
    #[test]
    fn test_gamma_correct_endpoints() {
        assert!((gamma_correct(0.0, 2.2) - 0.0).abs() < EPS);
        assert!((gamma_correct(1.0, 2.2) - 1.0).abs() < EPS);
    }

    // ── 9. gamma_correct: gamma=2 → sqrt ─────────────────────────────────────
    #[test]
    fn test_gamma_correct_sqrt() {
        let out = gamma_correct(0.25, 2.0);
        let expected = 0.5_f32;
        assert!(
            (out - expected).abs() < EPS,
            "expected {expected}, got {out}"
        );
    }

    // ── 10. srgb_gamma: 0 → 0, 1 → 1 ────────────────────────────────────────
    #[test]
    fn test_srgb_gamma_endpoints() {
        assert!((srgb_gamma(0.0) - 0.0).abs() < EPS);
        assert!((srgb_gamma(1.0) - 1.0).abs() < EPS);
    }

    // ── 11. srgb_gamma: piecewise threshold at 0.0031308 ─────────────────────
    #[test]
    fn test_srgb_gamma_piecewise() {
        // Below threshold: linear branch
        let x_low = 0.001_f32;
        let out_low = srgb_gamma(x_low);
        let expected_low = 12.92 * x_low;
        assert!(
            (out_low - expected_low).abs() < EPS,
            "linear branch: expected {expected_low}, got {out_low}"
        );

        // Above threshold: power branch
        let x_high = 0.5_f32;
        let out_high = srgb_gamma(x_high);
        let expected_high = 1.055 * x_high.powf(1.0 / 2.4) - 0.055;
        assert!(
            (out_high - expected_high).abs() < EPS,
            "power branch: expected {expected_high}, got {out_high}"
        );
    }

    // ── 12. inverse_srgb_gamma: roundtrip ────────────────────────────────────
    #[test]
    fn test_inverse_srgb_gamma_roundtrip() {
        for &x in &[0.0_f32, 0.001, 0.02, 0.2, 0.5, 0.9, 1.0] {
            let encoded = srgb_gamma(x);
            let decoded = inverse_srgb_gamma(encoded);
            assert!(
                (decoded - x).abs() < 1e-4,
                "roundtrip failed for {x}: encoded={encoded}, decoded={decoded}"
            );
        }
    }

    // ── 13. apply_exposure: stops=0 → unchanged ───────────────────────────────
    #[test]
    fn test_apply_exposure_zero_stops() {
        let img = vec![0.1_f32, 0.5, 0.9, 0.3, 0.7, 1.5];
        let out = apply_exposure(&img, 0.0);
        for (o, i) in out.iter().zip(img.iter()) {
            assert!((o - i).abs() < EPS, "expected {i}, got {o}");
        }
    }

    // ── 14. apply_exposure: stops=1 → doubled ────────────────────────────────
    #[test]
    fn test_apply_exposure_one_stop() {
        let img = vec![0.25_f32, 0.5, 1.0];
        let out = apply_exposure(&img, 1.0);
        let expected = [0.5_f32, 1.0, 2.0];
        for (o, e) in out.iter().zip(expected.iter()) {
            assert!((o - e).abs() < EPS, "expected {e}, got {o}");
        }
    }

    // ── 15. tone_map_image: all-zero HDR → all-zero LDR ──────────────────────
    #[test]
    fn test_tone_map_image_all_zero() {
        let img = vec![0.0_f32; 12]; // 4 pixels
        let config = ToneMappingConfig::default();
        let out = tone_map_image(&img, &config).expect("tone mapping should succeed");
        for v in &out {
            assert!(v.abs() < EPS, "expected 0.0, got {v}");
        }
    }

    // ── 16. tone_map_image: HDR > 1 → clamped to [0,1] ───────────────────────
    #[test]
    fn test_tone_map_image_hdr_clamped() {
        let img = vec![5.0_f32, 10.0, 20.0]; // one bright pixel
        let config = ToneMappingConfig::default();
        let out = tone_map_image(&img, &config).expect("tone mapping should succeed");
        for v in &out {
            assert!(
                *v >= 0.0 && *v <= 1.0,
                "output {v} not in [0,1] after tone mapping"
            );
        }
    }

    // ── 17. tone_map_image: wrong length → Err ───────────────────────────────
    #[test]
    fn test_tone_map_image_wrong_length() {
        let img = vec![0.5_f32, 0.5]; // 2 values — not multiple of 3
        let config = ToneMappingConfig::default();
        let result = tone_map_image(&img, &config);
        assert!(
            matches!(result, Err(ToneMappingError::InvalidImage(_))),
            "expected InvalidImage error"
        );
    }

    // ── 18. ToneMappingConfig::validate: gamma=0 → Err ────────────────────────
    #[test]
    fn test_config_validate_gamma_zero() {
        let config = ToneMappingConfig {
            gamma: 0.0,
            ..Default::default()
        };
        let result = config.validate();
        assert!(
            matches!(result, Err(ToneMappingError::InvalidConfig(_))),
            "expected InvalidConfig error for gamma=0"
        );
    }

    // ── 19. ToneMappingConfig::validate: saturation=-1 → Err ─────────────────
    #[test]
    fn test_config_validate_saturation_negative() {
        let config = ToneMappingConfig {
            saturation: -1.0,
            ..Default::default()
        };
        let result = config.validate();
        assert!(
            matches!(result, Err(ToneMappingError::InvalidConfig(_))),
            "expected InvalidConfig error for saturation=-1"
        );
    }

    // ── 19b. ToneMappingConfig::validate: Custom operator parameter checks ───

    #[test]
    fn test_config_validate_custom_shadow_gamma_invalid() {
        let config = ToneMappingConfig {
            operator: ToneMappingOperator::Custom {
                shadow_gamma: 0.0,
                midtone_scale: 1.0,
                highlight_rolloff: 1.0,
            },
            ..Default::default()
        };
        assert!(matches!(
            config.validate(),
            Err(ToneMappingError::InvalidConfig(_))
        ));
    }

    #[test]
    fn test_config_validate_custom_midtone_scale_invalid() {
        let config = ToneMappingConfig {
            operator: ToneMappingOperator::Custom {
                shadow_gamma: 1.0,
                midtone_scale: -1.0,
                highlight_rolloff: 1.0,
            },
            ..Default::default()
        };
        assert!(matches!(
            config.validate(),
            Err(ToneMappingError::InvalidConfig(_))
        ));
    }

    #[test]
    fn test_config_validate_custom_highlight_rolloff_invalid() {
        let config = ToneMappingConfig {
            operator: ToneMappingOperator::Custom {
                shadow_gamma: 1.0,
                midtone_scale: 1.0,
                highlight_rolloff: 0.0,
            },
            ..Default::default()
        };
        assert!(matches!(
            config.validate(),
            Err(ToneMappingError::InvalidConfig(_))
        ));
    }

    #[test]
    fn test_config_validate_custom_valid_params_ok() {
        let config = ToneMappingConfig {
            operator: ToneMappingOperator::Custom {
                shadow_gamma: 2.2,
                midtone_scale: 1.0,
                highlight_rolloff: 2.0,
            },
            ..Default::default()
        };
        assert!(config.validate().is_ok());
    }

    // ── 19c. Custom operator: highlights no longer collapse toward black ─────

    #[test]
    fn test_custom_operator_highlights_never_drop_below_one() {
        // Regression: the previous highlight formula was strictly
        // *decreasing* for x_adj > 1 (bright HDR input rendered darker
        // than midtones, eventually clamping to 0/black). The fixed
        // roll-off must never drop below the shoulder value of 1.0.
        let op = ToneMappingOperator::Custom {
            shadow_gamma: 1.0,
            midtone_scale: 1.0,
            highlight_rolloff: 1.0,
        };
        for &x in &[1.0_f32, 2.0, 4.0, 10.0, 100.0, 1000.0] {
            let y = op.apply_channel(x);
            assert!(
                y >= 1.0 - EPS,
                "Custom highlight at x={x} should stay at/above 1.0 (white), got {y}"
            );
        }
    }

    // ── 19d. Custom operator: monotonicity across a wide input range ─────────

    #[test]
    fn test_custom_operator_monotonic_property() {
        for &(shadow_gamma, midtone_scale, highlight_rolloff) in &[
            (1.0_f32, 1.0_f32, 1.0_f32),
            (2.2, 1.0, 2.0),
            (0.5, 2.0, 0.5),
            (1.0, 0.5, 5.0),
        ] {
            let op = ToneMappingOperator::Custom {
                shadow_gamma,
                midtone_scale,
                highlight_rolloff,
            };
            let mut prev = f32::NEG_INFINITY;
            let mut x = 0.0_f32;
            while x <= 100.0 {
                let y = op.apply_channel(x);
                assert!(
                    y >= prev - EPS,
                    "Custom({shadow_gamma},{midtone_scale},{highlight_rolloff}) not monotonic at x={x}: y={y}, prev={prev}"
                );
                prev = y;
                x += 0.5;
            }
        }
    }

    // ── 20. compute_hdr_stats: all-zero → min=max=0 ───────────────────────────
    #[test]
    fn test_compute_hdr_stats_all_zero() {
        let img = vec![0.0_f32; 9]; // 3 pixels
        let stats = compute_hdr_stats(&img).expect("stats should succeed");
        assert!((stats.min_value - 0.0).abs() < EPS);
        assert!((stats.max_value - 0.0).abs() < EPS);
        assert!((stats.mean_luminance - 0.0).abs() < EPS);
    }

    // ── 21. compute_hdr_stats: mixed values → correct percentile ──────────────
    #[test]
    fn test_compute_hdr_stats_percentile() {
        // 100 pixels, luminance 0 through 0.99
        let n = 100_usize;
        let mut img = Vec::with_capacity(n * 3);
        for i in 0..n {
            let v = i as f32 / n as f32;
            img.push(v);
            img.push(v);
            img.push(v);
        }
        let stats = compute_hdr_stats(&img).expect("stats should succeed");
        // 99th percentile index = 99 * 99 / 100 = 98 → luma ≈ 0.98
        assert!(
            stats.percentile_99 >= 0.95 && stats.percentile_99 <= 1.0,
            "unexpected percentile_99 = {}",
            stats.percentile_99
        );
    }

    // ── 22. recommend_exposure: high luminance → negative stops ───────────────
    #[test]
    fn test_recommend_exposure_high_luminance() {
        // Build a bright image so log_mean_luminance >> 0.18
        let img: Vec<f32> = (0..30)
            .map(|i| if i % 3 == 0 { 10.0 } else { 9.0 })
            .collect();
        let stats = compute_hdr_stats(&img).expect("stats should succeed");
        let stops = recommend_exposure(&stats);
        assert!(
            stops < 0.0,
            "expected negative stops for bright image, got {stops}"
        );
    }

    // ── 23. preset_aces: produces a valid config ───────────────────────────────
    #[test]
    fn test_preset_aces_valid() {
        let config = preset_aces();
        config
            .validate()
            .expect("preset_aces config should be valid");
        assert!(config.use_srgb_gamma);
        assert!(matches!(config.operator, ToneMappingOperator::AcesFilmic));
    }

    // ── 24. tone_map_rgba_image: alpha channel is passed through unchanged ─────
    #[test]
    fn test_tone_map_rgba_alpha_passthrough() {
        // Pixels with distinctive alpha values
        let img = vec![
            0.5_f32, 0.3, 0.8, 0.42, // pixel 0, alpha = 0.42
            2.0, 5.0, 0.1, 0.99, // pixel 1, alpha = 0.99 (HDR rgb)
        ];
        let config = ToneMappingConfig::default();
        let out = tone_map_rgba_image(&img, &config).expect("rgba tone mapping should succeed");
        assert_eq!(out.len(), 8);
        // Check alphas are exactly preserved
        assert!((out[3] - 0.42).abs() < EPS, "alpha[0] changed: {}", out[3]);
        assert!((out[7] - 0.99).abs() < EPS, "alpha[1] changed: {}", out[7]);
        // RGB values for pixel 1 must be in [0,1] after tone mapping
        for &v in &[out[4], out[5], out[6]] {
            assert!((0.0..=1.0).contains(&v), "HDR rgb not clamped: {v}");
        }
    }

    // ── 25. tone_map_image: all presets validate and produce LDR output ────────
    #[test]
    fn test_all_presets_produce_ldr() {
        let img: Vec<f32> = (0..30)
            .map(|i| if i % 3 == 0 { 3.5 } else { 0.7 })
            .collect();
        let presets = [
            preset_reinhard(),
            preset_aces(),
            preset_filmic(),
            preset_photography(),
        ];
        for preset in &presets {
            preset.validate().expect("preset should be valid");
            let out = tone_map_image(&img, preset)
                .unwrap_or_else(|e| panic!("preset '{}' failed: {e}", preset.operator.name()));
            for (i, &v) in out.iter().enumerate() {
                assert!(
                    (0.0..=1.0).contains(&v),
                    "preset '{}' output[{i}] = {v} not in [0,1]",
                    preset.operator.name()
                );
            }
        }
    }

    // ── 26. custom operator: continuity at x_adj = 1 ─────────────────────────
    #[test]
    fn test_custom_operator_continuity() {
        let op = ToneMappingOperator::Custom {
            shadow_gamma: 1.0,
            midtone_scale: 1.0,
            highlight_rolloff: 1.0,
        };
        // x = 1 → x_adj = 1 → highlight branch (x_adj >= 1.0): 1 + ln(1)/h = 1
        let at_one = op.apply_channel(1.0);
        // x slightly > 1 → x_adj slightly > 1 → highlight branch, barely above 1.0
        let just_above = op.apply_channel(1.0001);
        assert!(
            (at_one - 1.0).abs() < EPS,
            "at x=1 expected 1.0, got {at_one}"
        );
        assert!(
            (just_above - 1.0).abs() < 0.01,
            "just above 1 expected close to 1.0, got {just_above}"
        );
    }

    // ── 27. dynamic_range_ev: positive for non-uniform image ─────────────────
    #[test]
    fn test_dynamic_range_ev_positive() {
        let img = vec![0.001_f32, 0.001, 0.001, 10.0, 10.0, 10.0];
        let stats = compute_hdr_stats(&img).expect("stats");
        assert!(
            stats.dynamic_range_ev > 0.0,
            "expected positive dynamic range, got {}",
            stats.dynamic_range_ev
        );
    }

    // ── 28. tone_map_image: empty → EmptyImage error ─────────────────────────
    #[test]
    fn test_tone_map_image_empty() {
        let config = ToneMappingConfig::default();
        let result = tone_map_image(&[], &config);
        assert!(matches!(result, Err(ToneMappingError::EmptyImage)));
    }

    // ── 29. ToneMappingOperator::name() returns correct strings ───────────────
    #[test]
    fn test_operator_names() {
        assert_eq!(ToneMappingOperator::Reinhard.name(), "reinhard");
        assert_eq!(ToneMappingOperator::AcesFilmic.name(), "aces_filmic");
        assert_eq!(ToneMappingOperator::Filmic.name(), "filmic");
        assert_eq!(ToneMappingOperator::Linear.name(), "linear");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // New tests (30–64): covering the new functions added in this pass
    // ─────────────────────────────────────────────────────────────────────────

    // ── 30. tone_luminance: pure red → 0.2126 ────────────────────────────────
    #[test]
    fn test_tone_luminance_pure_red() {
        let l = tone_luminance(1.0, 0.0, 0.0);
        assert!((l - 0.2126).abs() < EPS, "expected 0.2126, got {l}");
    }

    // ── 31. tone_luminance: pure green → 0.7152 ──────────────────────────────
    #[test]
    fn test_tone_luminance_pure_green() {
        let l = tone_luminance(0.0, 1.0, 0.0);
        assert!((l - 0.7152).abs() < EPS, "expected 0.7152, got {l}");
    }

    // ── 32. tone_luminance: white → 1.0 ──────────────────────────────────────
    #[test]
    fn test_tone_luminance_white() {
        let l = tone_luminance(1.0, 1.0, 1.0);
        assert!((l - 1.0).abs() < EPS, "expected 1.0, got {l}");
    }

    // ── 33. apply_gamma: value=1.0 → 1.0 regardless of gamma ─────────────────
    #[test]
    fn test_apply_gamma_one() {
        let out = apply_gamma(1.0, 2.2);
        assert!((out - 1.0).abs() < EPS, "expected 1.0, got {out}");
    }

    // ── 34. apply_gamma: value=0.0 → 0.0 ─────────────────────────────────────
    #[test]
    fn test_apply_gamma_zero() {
        let out = apply_gamma(0.0, 2.2);
        assert!(out.abs() < EPS, "expected 0.0, got {out}");
    }

    // ── 35. apply_gamma: gamma=2.0, value=0.25 → 0.5 ────────────────────────
    #[test]
    fn test_apply_gamma_half() {
        let out = apply_gamma(0.25, 2.0);
        assert!((out - 0.5).abs() < EPS, "expected 0.5, got {out}");
    }

    // ── 36. hdr_linear_to_srgb: 0 → 0, 1 → 1 ────────────────────────────────
    #[test]
    fn test_hdr_linear_to_srgb_endpoints() {
        assert!(hdr_linear_to_srgb(0.0).abs() < EPS);
        assert!((hdr_linear_to_srgb(1.0) - 1.0).abs() < EPS);
    }

    // ── 37. hdr_linear_to_srgb: piecewise boundary at 0.0031308 ──────────────
    #[test]
    fn test_hdr_linear_to_srgb_piecewise() {
        let x_low = 0.001_f32;
        let out_low = hdr_linear_to_srgb(x_low);
        let expected_low = 12.92 * x_low;
        assert!(
            (out_low - expected_low).abs() < EPS,
            "linear branch: expected {expected_low}, got {out_low}"
        );

        let x_high = 0.5_f32;
        let out_high = hdr_linear_to_srgb(x_high);
        let expected_high = (1.055 * x_high.powf(1.0 / 2.4) - 0.055).clamp(0.0, 1.0);
        assert!(
            (out_high - expected_high).abs() < EPS,
            "power branch: expected {expected_high}, got {out_high}"
        );
    }

    // ── 38. srgb_to_linear_hdr round-trip ─────────────────────────────────────
    #[test]
    fn test_srgb_to_linear_hdr_roundtrip() {
        for &x in &[0.0_f32, 0.001, 0.02, 0.2, 0.5, 0.9, 1.0] {
            let encoded = hdr_linear_to_srgb(x);
            let decoded = srgb_to_linear_hdr(encoded);
            assert!(
                (decoded - x).abs() < 1e-4,
                "round-trip failed for {x}: encoded={encoded}, decoded={decoded}"
            );
        }
    }

    // ── 39. apply_operator Reinhard: black stays black ────────────────────────
    #[test]
    fn test_apply_operator_reinhard_black() {
        let (r, g, b) = apply_operator(0.0, 0.0, 0.0, &ToneMapOperator::Reinhard);
        assert!(r.abs() < EPS && g.abs() < EPS && b.abs() < EPS);
    }

    // ── 40. apply_operator Reinhard: large → approaches (1,1,1) ──────────────
    #[test]
    fn test_apply_operator_reinhard_large() {
        let (r, g, b) = apply_operator(1000.0, 1000.0, 1000.0, &ToneMapOperator::Reinhard);
        assert!(r > 0.999 && g > 0.999 && b > 0.999, "r={r} g={g} b={b}");
    }

    // ── 41. apply_operator ReinhardExtended: max_luminance controls limit ─────
    #[test]
    fn test_apply_operator_reinhard_extended() {
        let op = ToneMapOperator::ReinhardExtended { max_luminance: 2.0 };
        let (r, _g, _b) = apply_operator(2.0, 2.0, 2.0, &op);
        // At lum = max_lum: extended Reinhard gives ≤ 1.0 and > reinhard(1)
        assert!(r <= 1.0 + EPS, "expected <= 1, got {r}");
        assert!(r > 0.0, "expected > 0, got {r}");
    }

    // ── 42. apply_operator Filmic: in plausible [0,1] range for positive inputs
    #[test]
    fn test_apply_operator_filmic_range() {
        for v in [0.0_f32, 0.5, 1.0, 2.0, 5.0] {
            let (r, g, b) = apply_operator(v, v, v, &ToneMapOperator::Filmic);
            assert!((0.0..=1.0).contains(&r), "filmic r={r} for input {v}");
            assert!((0.0..=1.0).contains(&g), "filmic g={g} for input {v}");
            assert!((0.0..=1.0).contains(&b), "filmic b={b} for input {v}");
        }
    }

    // ── 43. apply_operator Aces: clamped to [0,1] ────────────────────────────
    #[test]
    fn test_apply_operator_aces_clamped() {
        for v in [-1.0_f32, 0.0, 0.5, 1.0, 10.0, 100.0] {
            let (r, g, b) = apply_operator(v, v, v, &ToneMapOperator::Aces);
            assert!((0.0..=1.0).contains(&r), "aces r={r} for input {v}");
            assert!((0.0..=1.0).contains(&g), "aces g={g} for input {v}");
            assert!((0.0..=1.0).contains(&b), "aces b={b} for input {v}");
        }
    }

    // ── 44. apply_operator Lottes: monotonically non-decreasing for positive ──
    #[test]
    fn test_apply_operator_lottes_monotonic() {
        let mut prev = 0.0_f32;
        let lottes_op = ToneMapOperator::Lottes(LottesParams::default());
        for i in 0..=20u32 {
            let v = i as f32 * 0.5;
            let (r, _g, _b) = apply_operator(v, v, v, &lottes_op);
            assert!(
                r >= prev - EPS,
                "Lottes not monotonic at {v}: r={r}, prev={prev}"
            );
            prev = r;
        }
    }

    // ── 45. apply_operator Exposure stops=0 → identity (no clamp applied) ─────
    #[test]
    fn test_apply_operator_exposure_zero_stops() {
        let (r, g, b) = apply_operator(0.5, 0.3, 0.7, &ToneMapOperator::Exposure { stops: 0.0 });
        assert!((r - 0.5).abs() < EPS && (g - 0.3).abs() < EPS && (b - 0.7).abs() < EPS);
    }

    // ── 46. apply_operator Exposure stops=1 → doubled ────────────────────────
    #[test]
    fn test_apply_operator_exposure_one_stop() {
        let (r, g, b) = apply_operator(0.5, 0.25, 0.1, &ToneMapOperator::Exposure { stops: 1.0 });
        assert!((r - 1.0).abs() < EPS, "r={r}");
        assert!((g - 0.5).abs() < EPS, "g={g}");
        assert!((b - 0.2).abs() < EPS, "b={b}");
    }

    // ── 47. apply_operator Linear min=0 max=1 → identity ─────────────────────
    #[test]
    fn test_apply_operator_linear_identity() {
        let op = ToneMapOperator::Linear { min: 0.0, max: 1.0 };
        let (r, g, b) = apply_operator(0.5, 0.3, 0.8, &op);
        assert!((r - 0.5).abs() < 1e-4, "r={r}");
        assert!((g - 0.3).abs() < 1e-4, "g={g}");
        assert!((b - 0.8).abs() < 1e-4, "b={b}");
    }

    // ── 48. apply_operator Linear min=0 max=2 → halves ───────────────────────
    #[test]
    fn test_apply_operator_linear_halves() {
        let op = ToneMapOperator::Linear { min: 0.0, max: 2.0 };
        let (r, _g, _b) = apply_operator(1.0, 0.0, 0.0, &op);
        // (1.0 - 0.0) / (2.0 + 1e-7) ≈ 0.5
        assert!((r - 0.5).abs() < 1e-4, "expected ~0.5, got {r}");
    }

    // ── 49. tone_map: size mismatch error ─────────────────────────────────────
    #[test]
    fn test_tone_map_size_mismatch() {
        let img = vec![0.5_f32; 7]; // 7 ≠ 2*2*3 = 12
        let config = ToneMapConfig::default();
        let result = tone_map(&img, 2, 2, &config);
        assert!(matches!(result, Err(ToneMappingError::SizeMismatch { .. })));
    }

    // ── 50. tone_map: empty image error ──────────────────────────────────────
    #[test]
    fn test_tone_map_empty() {
        let config = ToneMapConfig::default();
        let result = tone_map(&[], 0, 0, &config);
        assert!(matches!(result, Err(ToneMappingError::EmptyImage)));
    }

    // ── 51. tone_map: HDR values → LDR after clipping ────────────────────────
    #[test]
    fn test_tone_map_hdr_to_ldr() {
        let img = vec![5.0_f32, 10.0, 20.0, 3.0, 0.5, 0.1];
        let config = ToneMapConfig::default();
        let out = tone_map(&img, 2, 1, &config).expect("tone_map should succeed");
        for &v in &out {
            assert!((0.0..=1.0).contains(&v), "output {v} not in [0,1]");
        }
    }

    // ── 52. tone_map_inplace: same result as tone_map ─────────────────────────
    #[test]
    fn test_tone_map_inplace_matches_tone_map() {
        let img = vec![0.5_f32, 0.8, 2.0, 0.1, 3.0, 0.2];
        let config = ToneMapConfig::default();
        let expected = tone_map(&img, 2, 1, &config).expect("tone_map");
        let mut buf = img.clone();
        tone_map_inplace(&mut buf, 2, 1, &config).expect("tone_map_inplace");
        for (a, b) in buf.iter().zip(expected.iter()) {
            assert!((a - b).abs() < EPS, "inplace={a}, expected={b}");
        }
    }

    // ── 53. apply_gamma_image: all pixels correctly adjusted ──────────────────
    #[test]
    fn test_apply_gamma_image() {
        let img = vec![0.0_f32, 0.25, 1.0];
        let out = apply_gamma_image(&img, 2.0);
        assert!(out[0].abs() < EPS, "expected 0, got {}", out[0]);
        assert!((out[1] - 0.5).abs() < EPS, "expected 0.5, got {}", out[1]);
        assert!((out[2] - 1.0).abs() < EPS, "expected 1.0, got {}", out[2]);
    }

    // ── 54. image_hdr_linear_to_srgb: all pixels processed ───────────────────
    #[test]
    fn test_image_hdr_linear_to_srgb() {
        let img = vec![0.0_f32, 0.5, 1.0, 0.002];
        let out = image_hdr_linear_to_srgb(&img);
        assert_eq!(out.len(), img.len());
        for (i, (&v, &o)) in img.iter().zip(out.iter()).enumerate() {
            let expected = hdr_linear_to_srgb(v);
            assert!(
                (o - expected).abs() < EPS,
                "pixel {i}: expected {expected}, got {o}"
            );
        }
    }

    // ── 55. hdr_white_balance: scales channels correctly ─────────────────────
    #[test]
    fn test_hdr_white_balance_channels() {
        let img = vec![1.0_f32, 1.0, 1.0, 0.5, 0.5, 0.5];
        let out = hdr_white_balance(&img, 1.5, 0.8, 1.0);
        assert!((out[0] - 1.5).abs() < EPS, "R scale: {}", out[0]);
        assert!((out[1] - 0.8).abs() < EPS, "G scale: {}", out[1]);
        assert!((out[2] - 1.0).abs() < EPS, "B scale: {}", out[2]);
        assert!((out[3] - 0.75).abs() < EPS, "R2: {}", out[3]);
        assert!((out[4] - 0.4).abs() < EPS, "G2: {}", out[4]);
        assert!((out[5] - 0.5).abs() < EPS, "B2: {}", out[5]);
    }

    // ── 56. luminance_histogram: all-gray image fills one bin ────────────────
    #[test]
    fn test_luminance_histogram_single_bin() {
        // Gray pixel (0.5, 0.5, 0.5) → lum = 0.5 → bin = round(0.5 * 255) = 128
        let n_pixels = 10_usize;
        let img = vec![0.5_f32; n_pixels * 3];
        let hist = luminance_histogram(&img, n_pixels, 1).expect("histogram");
        assert_eq!(hist.len(), 256);
        let total: u32 = hist.iter().sum();
        assert_eq!(
            total as usize, n_pixels,
            "total bin count should equal n_pixels"
        );
        // Only one bin should be non-zero
        let non_zero: Vec<usize> = hist
            .iter()
            .enumerate()
            .filter(|(_, &v)| v > 0)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(non_zero.len(), 1, "expected exactly 1 non-zero bin");
    }

    // ── 57. estimate_scene_key: constant luminance image ─────────────────────
    #[test]
    fn test_estimate_scene_key_constant() {
        let lum = 0.5_f32;
        // gray image: r=g=b=v where v gives lum = 0.5
        // lum = 0.2126*v + 0.7152*v + 0.0722*v = v → v = 0.5
        let img = vec![lum; 3 * 4]; // 4 pixels
        let key = estimate_scene_key(&img, 4, 1).expect("scene key");
        let expected = (lum + 1e-6).ln().exp();
        assert!(
            (key - expected).abs() < 1e-4,
            "expected ~{expected}, got {key}"
        );
    }

    // ── 58. hdr_image_stats: known image ─────────────────────────────────────
    #[test]
    fn test_hdr_image_stats_known() {
        // Two pixels: (1,0,0) lum=0.2126, (0,1,0) lum=0.7152
        let img = vec![1.0_f32, 0.0, 0.0, 0.0, 1.0, 0.0];
        let stats = hdr_image_stats(&img, 2, 1).expect("stats");
        assert!(
            (stats.min_luminance - 0.2126).abs() < EPS,
            "min_lum={}",
            stats.min_luminance
        );
        assert!(
            (stats.max_luminance - 0.7152).abs() < EPS,
            "max_lum={}",
            stats.max_luminance
        );
        let expected_mean = (0.2126 + 0.7152) / 2.0;
        assert!(
            (stats.mean_luminance - expected_mean).abs() < EPS,
            "mean_lum={}",
            stats.mean_luminance
        );
    }

    // ── 59. auto_exposure: scene with lum≈0.18 → stops≈0 ────────────────────
    #[test]
    fn test_auto_exposure_middle_gray() {
        // Build image where luminance ≈ 0.18 = 0.2126*r+0.7152*g+0.0722*b
        // Use gray: r=g=b=v, lum = v. Want lum ≈ 0.18, so v ≈ 0.18
        let img = vec![0.18_f32; 3 * 4];
        let stops = auto_exposure(&img, 4, 1).expect("auto_exposure");
        // scene_key ≈ 0.18 + 1e-6, target = 0.18 → stops ≈ 0
        assert!(stops.abs() < 0.01, "expected stops≈0, got {stops}");
    }

    // ── 60. format_tone_config: non-empty string ──────────────────────────────
    #[test]
    fn test_format_tone_config_nonempty() {
        let config = ToneMapConfig::default();
        let s = format_tone_config(&config);
        assert!(!s.is_empty(), "format_tone_config returned empty string");
        assert!(s.contains("aces"), "expected 'aces' in output: {s}");
    }

    // ── 61. tone_map all-black image ──────────────────────────────────────────
    #[test]
    fn test_tone_map_all_black() {
        let img = vec![0.0_f32; 12]; // 2x2x3
        let config = ToneMapConfig::default();
        let out = tone_map(&img, 2, 2, &config).expect("tone_map");
        for &v in &out {
            assert!(v.abs() < EPS, "expected 0 for black image, got {v}");
        }
    }

    // ── 62. tone_map all-white image → all ones ───────────────────────────────
    #[test]
    fn test_tone_map_all_white() {
        let img = vec![1.0_f32; 3];
        let config = ToneMapConfig {
            operator: ToneMapOperator::Reinhard,
            gamma: 1.0,
            apply_gamma: false,
            clip: true,
        };
        let out = tone_map(&img, 1, 1, &config).expect("tone_map");
        // Reinhard(1.0) = 0.5, not 1.0 — just check in [0,1]
        for &v in &out {
            assert!((0.0..=1.0).contains(&v), "v={v}");
        }
    }

    // ── 63. tone_map extreme HDR value=100 ───────────────────────────────────
    #[test]
    fn test_tone_map_extreme_hdr() {
        let img = vec![100.0_f32, 100.0, 100.0];
        let config = ToneMapConfig::default();
        let out = tone_map(&img, 1, 1, &config).expect("tone_map");
        for &v in &out {
            assert!(
                (0.0..=1.0).contains(&v),
                "extreme HDR not mapped to LDR: {v}"
            );
        }
    }

    // ── 64. generalized_reinhard: monotonically non-decreasing ───────────────
    #[test]
    fn test_generalized_reinhard_monotonic() {
        let mut prev = 0.0_f32;
        for i in 0..=30u32 {
            let x = i as f32 * 0.3;
            let y = generalized_reinhard(x);
            assert!(
                y >= prev - EPS,
                "generalized_reinhard not monotonic at {x}: y={y}, prev={prev}"
            );
            prev = y;
        }
    }

    // ── 66. lottes: hits its defining anchors (mid-grey and hdr_max) ─────────
    #[test]
    fn test_lottes_anchors() {
        let params = LottesParams::default();
        assert!(
            (lottes(0.0, &params) - 0.0).abs() < EPS,
            "lottes(0) should be 0"
        );
        let mid = lottes(params.mid_in, &params);
        assert!(
            (mid - params.mid_out).abs() < 1e-2,
            "expected mid-grey anchor {}, got {mid}",
            params.mid_out
        );
        let top = lottes(params.hdr_max, &params);
        assert!(
            (top - 1.0).abs() < 1e-2,
            "expected hdr_max to map to 1.0, got {top}"
        );
    }

    // ── 67. lottes: monotonic and bounded to [0, 1] well past hdr_max ────────
    #[test]
    fn test_lottes_monotonic_and_bounded() {
        let params = LottesParams::default();
        let mut prev = 0.0_f32;
        for i in 0..=100u32 {
            let x = i as f32 * 0.2; // 0.0 .. 20.0, well past hdr_max = 8.0
            let y = lottes(x, &params);
            assert!((0.0..=1.0).contains(&y), "lottes({x}) = {y} out of [0,1]");
            assert!(
                y >= prev - EPS,
                "lottes not monotonic at {x}: y={y}, prev={prev}"
            );
            prev = y;
        }
    }

    // ── 68. hdr_image_stats: size mismatch error ──────────────────────────────
    #[test]
    fn test_hdr_image_stats_size_mismatch() {
        let img = vec![0.5_f32; 7]; // 7 ≠ 2*2*3 = 12
        let result = hdr_image_stats(&img, 2, 2);
        assert!(matches!(result, Err(ToneMappingError::SizeMismatch { .. })));
    }
}
