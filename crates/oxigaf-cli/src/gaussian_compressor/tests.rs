//! Auto-generated test module (consolidated from inline `#[cfg(test)] mod` blocks)

use super::*;

#[cfg(test)]
mod tests_2 {
    use super::*;
    type SceneTuple = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);
    fn make_scene(n: usize) -> SceneTuple {
        let positions: Vec<f32> = (0..n * 3).map(|i| i as f32 * 0.01).collect();
        let rotations: Vec<f32> = (0..n * 4)
            .map(|i| if i % 4 == 3 { 1.0f32 } else { 0.0 })
            .collect();
        let scales: Vec<f32> = vec![-1.0f32; n * 3];
        let opacities: Vec<f32> = (0..n).map(|i| 1.0 + i as f32 * 0.01).collect();
        let sh_dc: Vec<f32> = vec![0.5f32; n * 3];
        let sh_rest: Vec<f32> = vec![];
        (positions, rotations, scales, opacities, sh_dc, sh_rest)
    }
    fn compress_default(n: usize) -> CompressedScene {
        let (pos, rot, scl, op, shd, shr) = make_scene(n);
        let config = CompressionConfig::default();
        gc_compress(
            GcSceneSlices {
                positions: &pos,
                rotations: &rot,
                scales: &scl,
                opacities: &op,
                sh_dc: &shd,
                sh_rest: &shr,
                n_rest_per_gaussian: 0,
            },
            &config,
        )
        .expect("compress failed")
    }
    #[test]
    fn test_prec_bits_full() {
        assert_eq!(QuantizationPrecision::Full.bits(), 32);
    }
    #[test]
    fn test_prec_bits_half() {
        assert_eq!(QuantizationPrecision::Half.bits(), 16);
    }
    #[test]
    fn test_prec_bits_byte() {
        assert_eq!(QuantizationPrecision::Byte.bits(), 8);
    }
    #[test]
    fn test_prec_as_str_nonempty() {
        assert!(!QuantizationPrecision::Full.as_str().is_empty());
        assert!(!QuantizationPrecision::Half.as_str().is_empty());
        assert!(!QuantizationPrecision::Byte.as_str().is_empty());
    }
    #[test]
    fn test_quantize_full_passthrough() {
        let vals: Vec<f32> = vec![1.0, 2.0, 3.0, -1.5];
        let qa = QuantizedAttribute::quantize(&vals, QuantizationPrecision::Full)
            .expect("quantize failed");
        assert_eq!(qa.data_f32, vals);
        assert!(qa.data_i16.is_empty());
        assert!(qa.data_i8.is_empty());
        assert_eq!(qa.n_elements, 4);
    }
    #[test]
    fn test_quantize_half_round_trip() {
        let vals: Vec<f32> = (0..64).map(|i| i as f32 * 0.1 - 3.0).collect();
        let qa = QuantizedAttribute::quantize(&vals, QuantizationPrecision::Half)
            .expect("quantize half");
        let deq = qa.dequantize();
        assert_eq!(deq.len(), vals.len());
        for (a, b) in deq.iter().zip(vals.iter()) {
            assert!((a - b).abs() < 0.01, "half round-trip error: {a} vs {b}");
        }
    }
    #[test]
    fn test_quantize_byte_round_trip() {
        let vals: Vec<f32> = (0..32).map(|i| i as f32 / 31.0).collect();
        let qa = QuantizedAttribute::quantize(&vals, QuantizationPrecision::Byte)
            .expect("quantize byte");
        let deq = qa.dequantize();
        assert_eq!(deq.len(), vals.len());
        let range = 1.0f32;
        let max_err = range / 127.0;
        for (a, b) in deq.iter().zip(vals.iter()) {
            assert!(
                (a - b).abs() <= max_err + f32::EPSILON * 10.0,
                "byte round-trip error: {a} vs {b}, max_err={max_err}"
            );
        }
    }
    #[test]
    fn test_quantize_byte_constant_input() {
        let test_val = std::f32::consts::PI;
        let vals = vec![test_val; 10];
        let qa = QuantizedAttribute::quantize(&vals, QuantizationPrecision::Byte)
            .expect("quantize constant");
        let deq = qa.dequantize();
        for v in &deq {
            assert!((v - test_val).abs() < 1e-5, "constant not preserved: {v}");
        }
    }
    #[test]
    fn test_quantize_half_constant_input() {
        let vals = vec![-7.0f32; 20];
        let qa = QuantizedAttribute::quantize(&vals, QuantizationPrecision::Half)
            .expect("quantize constant half");
        let deq = qa.dequantize();
        for v in &deq {
            assert!((v + 7.0f32).abs() < 1e-5, "constant not preserved: {v}");
        }
    }
    #[test]
    fn test_byte_size_full() {
        let qa =
            QuantizedAttribute::quantize(&[1.0f32; 10], QuantizationPrecision::Full).expect("q");
        assert_eq!(qa.byte_size(), 40);
    }
    #[test]
    fn test_byte_size_half() {
        let qa =
            QuantizedAttribute::quantize(&[1.0f32; 10], QuantizationPrecision::Half).expect("q");
        assert_eq!(qa.byte_size(), 20);
    }
    #[test]
    fn test_byte_size_byte() {
        let qa =
            QuantizedAttribute::quantize(&[1.0f32; 10], QuantizationPrecision::Byte).expect("q");
        assert_eq!(qa.byte_size(), 10);
    }
    #[test]
    fn test_dequantize_length_preserved_half() {
        let vals: Vec<f32> = (0..50).map(|i| i as f32).collect();
        let qa = QuantizedAttribute::quantize(&vals, QuantizationPrecision::Half).expect("q");
        assert_eq!(qa.dequantize().len(), 50);
    }
    #[test]
    fn test_dequantize_length_preserved_byte() {
        let vals: Vec<f32> = (0..50).map(|i| i as f32).collect();
        let qa = QuantizedAttribute::quantize(&vals, QuantizationPrecision::Byte).expect("q");
        assert_eq!(qa.dequantize().len(), 50);
    }
    #[test]
    fn test_compressed_bytes_lt_uncompressed_half() {
        let scene = compress_default(100);
        assert!(scene.compressed_bytes() < scene.uncompressed_bytes());
    }
    #[test]
    fn test_compression_ratio_gt_one_half() {
        let scene = compress_default(100);
        assert!(scene.compression_ratio() > 1.0);
    }
    #[test]
    fn test_compression_ratio_full_is_one() {
        let (pos, rot, scl, op, shd, shr) = make_scene(20);
        let config = CompressionConfig {
            position_precision: QuantizationPrecision::Full,
            rotation_precision: QuantizationPrecision::Full,
            scale_precision: QuantizationPrecision::Full,
            opacity_precision: QuantizationPrecision::Full,
            sh_dc_precision: QuantizationPrecision::Full,
            sh_rest_precision: QuantizationPrecision::Full,
            ..CompressionConfig::default()
        };
        let scene = gc_compress(
            GcSceneSlices {
                positions: &pos,
                rotations: &rot,
                scales: &scl,
                opacities: &op,
                sh_dc: &shd,
                sh_rest: &shr,
                n_rest_per_gaussian: 0,
            },
            &config,
        )
        .expect("compress full");
        assert!((scene.compression_ratio() - 1.0).abs() < f32::EPSILON * 10.0);
    }
    #[test]
    fn test_compression_ratio_byte_gt_half() {
        let (pos, rot, scl, op, shd, shr) = make_scene(100);
        let config_byte = CompressionConfig {
            position_precision: QuantizationPrecision::Byte,
            rotation_precision: QuantizationPrecision::Byte,
            scale_precision: QuantizationPrecision::Byte,
            opacity_precision: QuantizationPrecision::Byte,
            sh_dc_precision: QuantizationPrecision::Byte,
            sh_rest_precision: QuantizationPrecision::Byte,
            ..CompressionConfig::default()
        };
        let scene_byte = gc_compress(
            GcSceneSlices {
                positions: &pos,
                rotations: &rot,
                scales: &scl,
                opacities: &op,
                sh_dc: &shd,
                sh_rest: &shr,
                n_rest_per_gaussian: 0,
            },
            &config_byte,
        )
        .expect("compress byte");
        let scene_half = compress_default(100);
        assert!(scene_byte.compression_ratio() > scene_half.compression_ratio());
    }
    #[test]
    fn test_prune_mask_high_opacity_all_kept() {
        let ops = vec![5.0f32; 10];
        let scales = vec![0.0f32; 30];
        let config = ScenePruningConfig::default();
        let mask = gc_compute_prune_mask(&ops, &scales, &config).expect("mask");
        assert!(mask.iter().all(|&b| b));
    }
    #[test]
    fn test_prune_mask_zero_opacity_all_pruned() {
        let ops = vec![-10.0f32; 10];
        let scales = vec![0.0f32; 30];
        let config = ScenePruningConfig::default();
        let mask = gc_compute_prune_mask(&ops, &scales, &config).expect("mask");
        assert!(mask.iter().all(|&b| !b));
    }
    #[test]
    fn test_prune_mask_empty_error() {
        let ops: Vec<f32> = vec![];
        let scales: Vec<f32> = vec![];
        let config = ScenePruningConfig::default();
        let result = gc_compute_prune_mask(&ops, &scales, &config);
        assert!(matches!(result, Err(CompressorError::EmptyScene)));
    }
    #[test]
    fn test_prune_mask_large_scale_pruned() {
        let ops = vec![5.0f32; 5];
        let mut scales = vec![0.0f32; 15];
        scales[0] = 100.0;
        let config = ScenePruningConfig::default();
        let mask = gc_compute_prune_mask(&ops, &scales, &config).expect("mask");
        assert!(!mask[0], "Gaussian with huge scale should be pruned");
        assert!(mask[1..].iter().all(|&b| b));
    }
    #[test]
    fn test_prune_mask_all_tiny_scale_pruned() {
        let ops = vec![5.0f32; 3];
        let scales = vec![-20.0f32; 9];
        let config = ScenePruningConfig::default();
        let mask = gc_compute_prune_mask(&ops, &scales, &config).expect("mask");
        assert!(mask.iter().all(|&b| !b));
    }
    #[test]
    fn test_prune_mask_dimension_mismatch_error() {
        let ops = vec![5.0f32; 5];
        let scales = vec![0.0f32; 10];
        let config = ScenePruningConfig::default();
        let result = gc_compute_prune_mask(&ops, &scales, &config);
        assert!(matches!(
            result,
            Err(CompressorError::DimensionMismatch { .. })
        ));
    }
    #[test]
    fn test_prune_mask_target_n_gaussians() {
        let ops = vec![5.0, 4.0, 3.0, 2.0, 1.0f32];
        let scales = vec![0.0f32; 15];
        let config = ScenePruningConfig {
            opacity_threshold: 0.0,
            target_n_gaussians: Some(3),
            ..Default::default()
        };
        let mask = gc_compute_prune_mask(&ops, &scales, &config).expect("mask");
        let kept = mask.iter().filter(|&&b| b).count();
        assert_eq!(kept, 3);
    }
    #[test]
    fn test_apply_mask_all_true() {
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let mask = vec![true; 4];
        let result = gc_apply_mask_flat(&data, &mask, 3).expect("apply");
        assert_eq!(result, data);
    }
    #[test]
    fn test_apply_mask_all_false() {
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let mask = vec![false; 4];
        let result = gc_apply_mask_flat(&data, &mask, 3).expect("apply");
        assert!(result.is_empty());
    }
    #[test]
    fn test_apply_mask_mixed() {
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mask = vec![true, false];
        let result = gc_apply_mask_flat(&data, &mask, 3).expect("apply");
        assert_eq!(result, vec![1.0, 2.0, 3.0]);
    }
    #[test]
    fn test_apply_mask_dimension_mismatch_error() {
        let data: Vec<f32> = vec![1.0, 2.0, 3.0];
        let mask = vec![true, false];
        let result = gc_apply_mask_flat(&data, &mask, 3);
        assert!(matches!(
            result,
            Err(CompressorError::DimensionMismatch { .. })
        ));
    }
    #[test]
    fn test_prune_topn_all_when_n_ge_len() {
        let ops = vec![1.0f32, 2.0, 3.0, 4.0];
        let result = gc_prune_to_topn(&ops, 10).expect("topn");
        assert_eq!(result.len(), 4);
    }
    #[test]
    fn test_prune_topn_sorted_desc_by_opacity() {
        let ops = vec![1.0f32, 5.0, 3.0];
        let result = gc_prune_to_topn(&ops, 3).expect("topn");
        assert_eq!(result[0], 1);
        assert_eq!(result[1], 2);
        assert_eq!(result[2], 0);
    }
    #[test]
    fn test_prune_topn_n_zero_returns_empty() {
        let ops = vec![1.0f32, 2.0];
        let result = gc_prune_to_topn(&ops, 0).expect("topn");
        assert!(result.is_empty());
    }
    #[test]
    fn test_prune_topn_n1_highest_opacity() {
        let ops = vec![1.0f32, 5.0, 3.0];
        let result = gc_prune_to_topn(&ops, 1).expect("topn");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], 1);
    }
    #[test]
    fn test_prune_topn_empty_error() {
        let ops: Vec<f32> = vec![];
        let result = gc_prune_to_topn(&ops, 1);
        assert!(matches!(result, Err(CompressorError::EmptyScene)));
    }
    #[test]
    fn test_kmeans_pp_init_distinct_indices() {
        let positions: Vec<f32> = (0..30).map(|i| i as f32).collect();
        let mut rng = 42u64;
        let result = gc_kmeans_plus_plus_init(&positions, 5, &mut rng).expect("init");
        assert_eq!(result.len(), 5);
        let mut sorted = result.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), 5);
    }
    #[test]
    fn test_kmeans_pp_init_k_gt_n_error() {
        let positions: Vec<f32> = (0..9).map(|i| i as f32).collect();
        let mut rng = 1u64;
        let result = gc_kmeans_plus_plus_init(&positions, 10, &mut rng);
        assert!(matches!(result, Err(CompressorError::InvalidConfig(_))));
    }
    #[test]
    fn test_kmeans_pp_init_k_equals_n() {
        let positions: Vec<f32> = (0..9).map(|i| i as f32).collect();
        let mut rng = 7u64;
        let result = gc_kmeans_plus_plus_init(&positions, 3, &mut rng).expect("init");
        assert_eq!(result.len(), 3);
    }
    #[test]
    fn test_kmeans_positions_assignments_valid() {
        let n = 50usize;
        let positions: Vec<f32> = (0..n * 3).map(|i| (i % 10) as f32).collect();
        let config = KMeansConfig {
            n_clusters: 5,
            n_iterations: 20,
            tolerance: 1e-4,
        };
        let mut rng = 99u64;
        let (centers, assignments) =
            gc_kmeans_positions(&positions, &config, &mut rng).expect("kmeans");
        assert_eq!(centers.len(), 5 * 3);
        assert_eq!(assignments.len(), n);
        assert!(assignments.iter().all(|&a| a < 5));
    }
    #[test]
    fn test_kmeans_positions_single_cluster() {
        let positions: Vec<f32> = vec![1.0, 2.0, 3.0, 1.1, 2.1, 3.1, 0.9, 1.9, 2.9];
        let config = KMeansConfig {
            n_clusters: 1,
            n_iterations: 10,
            tolerance: 1e-4,
        };
        let mut rng = 1u64;
        let (centers, assignments) =
            gc_kmeans_positions(&positions, &config, &mut rng).expect("kmeans single");
        assert_eq!(centers.len(), 3);
        assert!(assignments.iter().all(|&a| a == 0));
    }
    #[test]
    fn test_kmeans_positions_k_gt_n_error() {
        let positions: Vec<f32> = vec![1.0, 2.0, 3.0];
        let config = KMeansConfig {
            n_clusters: 5,
            n_iterations: 10,
            tolerance: 1e-4,
        };
        let mut rng = 1u64;
        let result = gc_kmeans_positions(&positions, &config, &mut rng);
        assert!(matches!(result, Err(CompressorError::InvalidConfig(_))));
    }
    #[test]
    fn test_kmeans_positions_centers_count() {
        let n = 30usize;
        let positions: Vec<f32> = (0..n * 3).map(|i| i as f32 * 0.1).collect();
        let config = KMeansConfig {
            n_clusters: 3,
            n_iterations: 30,
            tolerance: 1e-5,
        };
        let mut rng = 12345u64;
        let (centers, _) = gc_kmeans_positions(&positions, &config, &mut rng).expect("kmeans");
        assert_eq!(centers.len(), 3 * 3);
    }
    #[test]
    fn test_cluster_residuals_shape() {
        let positions: Vec<f32> = (0..15).map(|i| i as f32).collect();
        let assignments = vec![0, 1, 0, 1, 0usize];
        let centers = vec![0.0f32; 6];
        let residuals = gc_cluster_residuals(&positions, &assignments, &centers).expect("res");
        assert_eq!(residuals.len(), 15);
    }
    #[test]
    fn test_cluster_residuals_zero_when_center_equals_point() {
        let positions = vec![1.0f32, 2.0, 3.0];
        let assignments = vec![0usize];
        let centers = vec![1.0f32, 2.0, 3.0];
        let residuals = gc_cluster_residuals(&positions, &assignments, &centers).expect("res");
        for v in &residuals {
            assert!(v.abs() < f32::EPSILON * 10.0);
        }
    }
    #[test]
    fn test_compress_empty_error() {
        let config = CompressionConfig::default();
        let result = gc_compress(
            GcSceneSlices {
                positions: &[],
                rotations: &[],
                scales: &[],
                opacities: &[],
                sh_dc: &[],
                sh_rest: &[],
                n_rest_per_gaussian: 0,
            },
            &config,
        );
        assert!(matches!(result, Err(CompressorError::EmptyScene)));
    }
    #[test]
    fn test_compress_dimension_mismatch_rotations() {
        let (pos, _rot, scl, op, shd, shr) = make_scene(10);
        let bad_rot = vec![1.0f32; 30];
        let config = CompressionConfig::default();
        let result = gc_compress(
            GcSceneSlices {
                positions: &pos,
                rotations: &bad_rot,
                scales: &scl,
                opacities: &op,
                sh_dc: &shd,
                sh_rest: &shr,
                n_rest_per_gaussian: 0,
            },
            &config,
        );
        assert!(matches!(
            result,
            Err(CompressorError::DimensionMismatch { .. })
        ));
    }
    #[test]
    fn test_compress_dimension_mismatch_scales() {
        let (pos, rot, _scl, op, shd, shr) = make_scene(10);
        let bad_scl = vec![0.0f32; 20];
        let config = CompressionConfig::default();
        let result = gc_compress(
            GcSceneSlices {
                positions: &pos,
                rotations: &rot,
                scales: &bad_scl,
                opacities: &op,
                sh_dc: &shd,
                sh_rest: &shr,
                n_rest_per_gaussian: 0,
            },
            &config,
        );
        assert!(matches!(
            result,
            Err(CompressorError::DimensionMismatch { .. })
        ));
    }
    #[test]
    fn test_compress_dimension_mismatch_opacities() {
        let (pos, rot, scl, _op, shd, shr) = make_scene(10);
        let bad_op = vec![1.0f32; 5];
        let config = CompressionConfig::default();
        let result = gc_compress(
            GcSceneSlices {
                positions: &pos,
                rotations: &rot,
                scales: &scl,
                opacities: &bad_op,
                sh_dc: &shd,
                sh_rest: &shr,
                n_rest_per_gaussian: 0,
            },
            &config,
        );
        assert!(matches!(
            result,
            Err(CompressorError::DimensionMismatch { .. })
        ));
    }
    #[test]
    fn test_compress_identical_rotations() {
        let n = 20usize;
        let (pos, _rot, scl, op, shd, shr) = make_scene(n);
        let rot_identity = [0.0f32, 0.0, 0.0, 1.0].repeat(n);
        let config = CompressionConfig::default();
        let scene = gc_compress(
            GcSceneSlices {
                positions: &pos,
                rotations: &rot_identity,
                scales: &scl,
                opacities: &op,
                sh_dc: &shd,
                sh_rest: &shr,
                n_rest_per_gaussian: 0,
            },
            &config,
        )
        .expect("compress");
        let deq = scene.rotations.dequantize();
        for chunk in deq.chunks(4) {
            for (i, &v) in chunk.iter().enumerate() {
                let expected = if i == 3 { 1.0f32 } else { 0.0f32 };
                assert!(
                    (v - expected).abs() < 1e-4,
                    "rotation mismatch: {v} vs {expected}"
                );
            }
        }
    }
    #[test]
    fn test_decompress_n_gaussians_correct() {
        let scene = compress_default(50);
        let n = scene.n_gaussians;
        let decompressed = gc_decompress(&scene).expect("decompress");
        assert_eq!(decompressed.n_gaussians, n);
    }
    #[test]
    fn test_decompress_positions_length() {
        let scene = compress_default(30);
        let n = scene.n_gaussians;
        let decompressed = gc_decompress(&scene).expect("decompress");
        assert_eq!(decompressed.positions.len(), n * 3);
    }
    #[test]
    fn test_decompress_rotations_length() {
        let scene = compress_default(30);
        let n = scene.n_gaussians;
        let decompressed = gc_decompress(&scene).expect("decompress");
        assert_eq!(decompressed.rotations.len(), n * 4);
    }
    #[test]
    fn test_round_trip_positions_within_tolerance() {
        let n = 20usize;
        let (pos, rot, scl, op, shd, shr) = make_scene(n);
        let config = CompressionConfig::default();
        let scene = gc_compress(
            GcSceneSlices {
                positions: &pos,
                rotations: &rot,
                scales: &scl,
                opacities: &op,
                sh_dc: &shd,
                sh_rest: &shr,
                n_rest_per_gaussian: 0,
            },
            &config,
        )
        .expect("compress");
        let decomp = gc_decompress(&scene).expect("decompress");
        let max_err = (n as f32 * 3.0 * 0.01) / 65534.0 + 1e-4;
        for (a, b) in decomp.positions.iter().zip(pos.iter()) {
            assert!(
                (a - b).abs() < max_err + 0.01,
                "position round-trip too large: {a} vs {b}"
            );
        }
    }
    #[test]
    fn test_round_trip_full_precision_exact() {
        let n = 10usize;
        let (pos, rot, scl, op, shd, shr) = make_scene(n);
        let config = CompressionConfig {
            position_precision: QuantizationPrecision::Full,
            rotation_precision: QuantizationPrecision::Full,
            scale_precision: QuantizationPrecision::Full,
            opacity_precision: QuantizationPrecision::Full,
            sh_dc_precision: QuantizationPrecision::Full,
            sh_rest_precision: QuantizationPrecision::Full,
            ..CompressionConfig::default()
        };
        let scene = gc_compress(
            GcSceneSlices {
                positions: &pos,
                rotations: &rot,
                scales: &scl,
                opacities: &op,
                sh_dc: &shd,
                sh_rest: &shr,
                n_rest_per_gaussian: 0,
            },
            &config,
        )
        .expect("compress");
        let decomp = gc_decompress(&scene).expect("decompress");
        let n_kept = decomp.n_gaussians;
        assert_eq!(n_kept, n);
        for (a, b) in decomp.positions.iter().zip(pos.iter()) {
            assert!((a - b).abs() < f32::EPSILON * 10.0, "not exact: {a} vs {b}");
        }
    }
    #[test]
    fn test_compute_stats_pruned_fraction_in_range() {
        let n = 100usize;
        let (pos, rot, scl, op, shd, shr) = make_scene(n);
        let config = CompressionConfig::default();
        let scene = gc_compress(
            GcSceneSlices {
                positions: &pos,
                rotations: &rot,
                scales: &scl,
                opacities: &op,
                sh_dc: &shd,
                sh_rest: &shr,
                n_rest_per_gaussian: 0,
            },
            &config,
        )
        .expect("compress");
        let stats = gc_compute_stats(&pos, &op, &scene).expect("stats");
        assert!(stats.pruned_fraction >= 0.0 && stats.pruned_fraction <= 1.0);
    }
    #[test]
    fn test_compute_stats_compression_ratio_positive() {
        let n = 50usize;
        let (pos, rot, scl, op, shd, shr) = make_scene(n);
        let config = CompressionConfig::default();
        let scene = gc_compress(
            GcSceneSlices {
                positions: &pos,
                rotations: &rot,
                scales: &scl,
                opacities: &op,
                sh_dc: &shd,
                sh_rest: &shr,
                n_rest_per_gaussian: 0,
            },
            &config,
        )
        .expect("compress");
        let stats = gc_compute_stats(&pos, &op, &scene).expect("stats");
        assert!(stats.compression_ratio > 0.0);
    }
    #[test]
    fn test_compute_stats_half_approx_2x() {
        let n = 200usize;
        let (pos, rot, scl, op, shd, shr) = make_scene(n);
        let config = CompressionConfig::default();
        let scene = gc_compress(
            GcSceneSlices {
                positions: &pos,
                rotations: &rot,
                scales: &scl,
                opacities: &op,
                sh_dc: &shd,
                sh_rest: &shr,
                n_rest_per_gaussian: 0,
            },
            &config,
        )
        .expect("compress");
        let stats = gc_compute_stats(&pos, &op, &scene).expect("stats");
        assert!(
            stats.compression_ratio > 1.5 && stats.compression_ratio < 3.0,
            "Expected ~2x ratio for half precision, got {}",
            stats.compression_ratio
        );
    }
    #[test]
    fn test_compute_stats_n_gaussians_before_after() {
        let n = 50usize;
        let (pos, rot, scl, op, shd, shr) = make_scene(n);
        let config = CompressionConfig::default();
        let scene = gc_compress(
            GcSceneSlices {
                positions: &pos,
                rotations: &rot,
                scales: &scl,
                opacities: &op,
                sh_dc: &shd,
                sh_rest: &shr,
                n_rest_per_gaussian: 0,
            },
            &config,
        )
        .expect("compress");
        let stats = gc_compute_stats(&pos, &op, &scene).expect("stats");
        assert_eq!(stats.n_gaussians_before, n);
        assert!(stats.n_gaussians_after <= n);
    }
    #[test]
    fn test_format_stats_nonempty() {
        let scene = compress_default(20);
        let (pos, _, _, op, _, _) = make_scene(20);
        let stats = gc_compute_stats(&pos, &op, &scene).expect("stats");
        let s = gc_format_stats(&stats);
        assert!(!s.is_empty());
        assert!(s.contains("ratio") || s.contains("Ratio") || s.contains("ratio"));
    }
    #[test]
    fn test_format_stats_contains_compression_ratio() {
        let scene = compress_default(30);
        let (pos, _, _, op, _, _) = make_scene(30);
        let stats = gc_compute_stats(&pos, &op, &scene).expect("stats");
        let s = gc_format_stats(&stats);
        assert!(s.contains("ratio") || s.contains("x"));
    }
    #[test]
    fn test_format_config_nonempty() {
        let config = CompressionConfig::default();
        let s = gc_format_config(&config);
        assert!(!s.is_empty());
    }
    #[test]
    fn test_format_config_contains_precision_info() {
        let config = CompressionConfig::default();
        let s = gc_format_config(&config);
        assert!(s.contains("half") || s.contains("Half") || s.contains("i16"));
    }
    #[test]
    fn test_full_pipeline_compress_decompress() {
        let n = 100usize;
        let (pos, rot, scl, op, shd, shr) = make_scene(n);
        let config = CompressionConfig::default();
        let scene = gc_compress(
            GcSceneSlices {
                positions: &pos,
                rotations: &rot,
                scales: &scl,
                opacities: &op,
                sh_dc: &shd,
                sh_rest: &shr,
                n_rest_per_gaussian: 0,
            },
            &config,
        )
        .expect("compress");
        let decomp = gc_decompress(&scene).expect("decompress");
        assert_eq!(decomp.n_gaussians, scene.n_gaussians);
        assert_eq!(decomp.positions.len(), scene.n_gaussians * 3);
    }
    #[test]
    fn test_full_pipeline_with_sh_rest() {
        let n = 20usize;
        let (pos, rot, scl, op, shd, _) = make_scene(n);
        let n_rest = 9usize;
        let sh_rest: Vec<f32> = vec![0.1f32; n * n_rest];
        let config = CompressionConfig::default();
        let scene = gc_compress(
            GcSceneSlices {
                positions: &pos,
                rotations: &rot,
                scales: &scl,
                opacities: &op,
                sh_dc: &shd,
                sh_rest: &sh_rest,
                n_rest_per_gaussian: n_rest,
            },
            &config,
        )
        .expect("compress with sh_rest");
        let decomp = gc_decompress(&scene).expect("decompress");
        assert_eq!(decomp.sh_rest.len(), scene.n_gaussians * n_rest);
    }
    #[test]
    fn test_full_pipeline_byte_precision() {
        let n = 50usize;
        let (pos, rot, scl, op, shd, shr) = make_scene(n);
        let config = CompressionConfig {
            position_precision: QuantizationPrecision::Byte,
            rotation_precision: QuantizationPrecision::Byte,
            scale_precision: QuantizationPrecision::Byte,
            opacity_precision: QuantizationPrecision::Byte,
            sh_dc_precision: QuantizationPrecision::Byte,
            sh_rest_precision: QuantizationPrecision::Byte,
            ..CompressionConfig::default()
        };
        let scene = gc_compress(
            GcSceneSlices {
                positions: &pos,
                rotations: &rot,
                scales: &scl,
                opacities: &op,
                sh_dc: &shd,
                sh_rest: &shr,
                n_rest_per_gaussian: 0,
            },
            &config,
        )
        .expect("compress byte");
        assert!(scene.compression_ratio() > 1.0);
        let decomp = gc_decompress(&scene).expect("decompress byte");
        assert_eq!(decomp.n_gaussians, scene.n_gaussians);
    }
    #[test]
    fn test_full_pipeline_with_position_clustering() {
        let n = 100usize;
        let (pos, rot, scl, op, shd, shr) = make_scene(n);
        let config = CompressionConfig {
            use_position_clustering: true,
            kmeans: KMeansConfig {
                n_clusters: 10,
                n_iterations: 20,
                tolerance: 1e-4,
            },
            ..CompressionConfig::default()
        };
        let scene = gc_compress(
            GcSceneSlices {
                positions: &pos,
                rotations: &rot,
                scales: &scl,
                opacities: &op,
                sh_dc: &shd,
                sh_rest: &shr,
                n_rest_per_gaussian: 0,
            },
            &config,
        )
        .expect("compress with clustering");
        let decomp = gc_decompress(&scene).expect("decompress");
        assert_eq!(decomp.n_gaussians, scene.n_gaussians);
        // Clustering is residual coding, not a no-op: the codebook must be
        // persisted, and decompression must add it back so world positions
        // survive the round trip (they used to collapse toward the origin).
        let clustering = scene
            .position_clustering
            .as_ref()
            .expect("codebook must be persisted");
        assert_eq!(clustering.n_clusters(), 10);
        assert_eq!(clustering.assignments.len(), scene.n_gaussians);
        assert_eq!(decomp.positions.len(), pos.len());
        for (a, b) in decomp.positions.iter().zip(pos.iter()) {
            assert!(
                (a - b).abs() < 1e-3,
                "clustered position round-trip failed: {a} vs {b}"
            );
        }
        // Survivor provenance is recorded even when nothing is pruned.
        assert_eq!(scene.kept_indices.len(), scene.n_gaussians);
    }
    #[test]
    fn test_compress_with_pruning_reduces_gaussians() {
        let n = 50usize;
        let mut opacities: Vec<f32> = vec![5.0f32; n];
        for op in opacities.iter_mut().take(n / 2) {
            *op = -10.0;
        }
        let positions: Vec<f32> = (0..n * 3).map(|i| i as f32 * 0.01).collect();
        let rotations: Vec<f32> = (0..n * 4)
            .map(|i| if i % 4 == 3 { 1.0f32 } else { 0.0 })
            .collect();
        let scales = vec![-1.0f32; n * 3];
        let sh_dc = vec![0.0f32; n * 3];
        let config = CompressionConfig::default();
        let scene = gc_compress(
            GcSceneSlices {
                positions: &positions,
                rotations: &rotations,
                scales: &scales,
                opacities: &opacities,
                sh_dc: &sh_dc,
                sh_rest: &[],
                n_rest_per_gaussian: 0,
            },
            &config,
        )
        .expect("compress");
        assert!(
            scene.n_gaussians < n,
            "Pruning should reduce Gaussian count"
        );
    }
    #[test]
    fn test_format_stats_all_stats_mentioned() {
        let n = 20usize;
        let (pos, _, _, op, _, _) = make_scene(n);
        let scene = compress_default(n);
        let stats = gc_compute_stats(&pos, &op, &scene).expect("stats");
        let s = gc_format_stats(&stats);
        assert!(s.chars().any(|c| c.is_ascii_digit()));
    }
}
