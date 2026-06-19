//! Auto-generated test module (consolidated from inline `#[cfg(test)] mod` blocks)

use super::*;

#[cfg(test)]
mod tests_2 {
    use super::*;
    fn temp_ply_path(name: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("oxigaf_ply_{}.ply", name));
        p
    }
    /// Build a slice of N test Gaussians with a given SH degree.
    fn make_test_gaussians(n: usize, sh_degree: usize) -> Vec<PlyGaussian> {
        let n_rest = PlyGaussian::n_rest_coeffs(sh_degree);
        (0..n)
            .map(|i| {
                let fi = i as f32;
                PlyGaussian {
                    x: fi * 0.1,
                    y: fi * 0.2,
                    z: fi * 0.3,
                    nx: 0.0,
                    ny: 0.0,
                    nz: 0.0,
                    f_dc: [fi * 0.01, fi * 0.02, fi * 0.03],
                    f_rest: (0..n_rest)
                        .map(|j| fi * 0.001 + j as f32 * 0.0001)
                        .collect(),
                    opacity: fi * 0.1 - 2.0,
                    scale: [fi.ln_1p() - 1.0, fi.ln_1p() - 0.5, fi.ln_1p()],
                    rot: [1.0, fi * 0.001, fi * 0.002, fi * 0.003],
                }
            })
            .collect()
    }
    /// Compare two f32 values with a relative tolerance.
    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        if a == b {
            return true;
        }
        let diff = (a - b).abs();
        let base = a.abs().max(b.abs()).max(1e-10);
        diff / base <= tol
    }
    #[test]
    fn test_ply_format_as_ply_str_ascii() {
        assert_eq!(PlyFormat::Ascii.as_ply_str(), "ascii");
    }
    #[test]
    fn test_ply_format_as_ply_str_binary_le() {
        assert_eq!(
            PlyFormat::BinaryLittleEndian.as_ply_str(),
            "binary_little_endian"
        );
    }
    #[test]
    fn test_ply_format_as_ply_str_binary_be() {
        assert_eq!(PlyFormat::BinaryBigEndian.as_ply_str(), "binary_big_endian");
    }
    #[test]
    fn test_n_rest_coeffs_degree_0() {
        assert_eq!(PlyGaussian::n_rest_coeffs(0), 0);
    }
    #[test]
    fn test_n_rest_coeffs_degree_1() {
        assert_eq!(PlyGaussian::n_rest_coeffs(1), 9);
    }
    #[test]
    fn test_n_rest_coeffs_degree_2() {
        assert_eq!(PlyGaussian::n_rest_coeffs(2), 24);
    }
    #[test]
    fn test_n_rest_coeffs_degree_3() {
        assert_eq!(PlyGaussian::n_rest_coeffs(3), 45);
    }
    #[test]
    fn test_identity_position_zero() {
        let g = PlyGaussian::identity();
        assert_eq!(g.x, 0.0);
        assert_eq!(g.y, 0.0);
        assert_eq!(g.z, 0.0);
    }
    #[test]
    fn test_identity_normals_zero() {
        let g = PlyGaussian::identity();
        assert_eq!(g.nx, 0.0);
        assert_eq!(g.ny, 0.0);
        assert_eq!(g.nz, 0.0);
    }
    #[test]
    fn test_identity_opacity_zero() {
        let g = PlyGaussian::identity();
        assert_eq!(g.opacity, 0.0);
    }
    #[test]
    fn test_identity_rot_is_unit_quaternion() {
        let g = PlyGaussian::identity();
        assert_eq!(g.rot[0], 1.0);
        assert_eq!(g.rot[1], 0.0);
        assert_eq!(g.rot[2], 0.0);
        assert_eq!(g.rot[3], 0.0);
    }
    #[test]
    fn test_identity_no_rest_coeffs() {
        let g = PlyGaussian::identity();
        assert!(g.f_rest.is_empty());
    }
    #[test]
    fn test_real_opacity_zero_input() {
        let mut g = PlyGaussian::identity();
        g.opacity = 0.0;
        let op = g.real_opacity();
        assert!(
            approx_eq(op, 0.5, 1e-6),
            "sigmoid(0) should be ~0.5, got {}",
            op
        );
    }
    #[test]
    fn test_real_opacity_large_positive() {
        let mut g = PlyGaussian::identity();
        g.opacity = 20.0;
        let op = g.real_opacity();
        assert!(op > 0.99, "sigmoid(20) should be near 1.0, got {}", op);
    }
    #[test]
    fn test_real_opacity_large_negative() {
        let mut g = PlyGaussian::identity();
        g.opacity = -20.0;
        let op = g.real_opacity();
        assert!(op < 0.01, "sigmoid(-20) should be near 0.0, got {}", op);
    }
    #[test]
    fn test_real_scale_zero_is_one() {
        let g = PlyGaussian::identity();
        let rs = g.real_scale();
        assert!(approx_eq(rs[0], 1.0, 1e-6));
        assert!(approx_eq(rs[1], 1.0, 1e-6));
        assert!(approx_eq(rs[2], 1.0, 1e-6));
    }
    #[test]
    fn test_real_scale_known_value() {
        let mut g = PlyGaussian::identity();
        g.scale = [1.0, 2.0, 3.0];
        let rs = g.real_scale();
        assert!(approx_eq(rs[0], 1.0f32.exp(), 1e-5));
        assert!(approx_eq(rs[1], 2.0f32.exp(), 1e-5));
        assert!(approx_eq(rs[2], 3.0f32.exp(), 1e-5));
    }
    #[test]
    fn test_from_flat_basic() {
        let positions = vec![1.0f32, 2.0, 3.0];
        let rotations = vec![0.1f32, 0.2, 0.3, 0.9];
        let scales = vec![0.5f32, -0.5, 0.0];
        let opacities = vec![0.0f32];
        let sh_dc = vec![0.1f32, 0.2, 0.3];
        let sh_rest: Vec<f32> = vec![];
        let g = PlyGaussian::from_flat(
            PlyFlatSlices {
                positions: &positions,
                rotations: &rotations,
                scales: &scales,
                opacities: &opacities,
                sh_dc: &sh_dc,
                sh_rest: &sh_rest,
                n_rest: 0,
            },
            0,
        )
        .expect("from_flat should succeed");
        assert!(approx_eq(g.x, 1.0, 1e-6));
        assert!(approx_eq(g.y, 2.0, 1e-6));
        assert!(approx_eq(g.z, 3.0, 1e-6));
    }
    #[test]
    fn test_from_flat_quaternion_reorder() {
        let positions = vec![0.0f32; 3];
        let rotations = vec![0.1f32, 0.2, 0.3, 0.9];
        let scales = vec![0.0f32; 3];
        let opacities = vec![0.0f32];
        let sh_dc = vec![0.0f32; 3];
        let sh_rest: Vec<f32> = vec![];
        let g = PlyGaussian::from_flat(
            PlyFlatSlices {
                positions: &positions,
                rotations: &rotations,
                scales: &scales,
                opacities: &opacities,
                sh_dc: &sh_dc,
                sh_rest: &sh_rest,
                n_rest: 0,
            },
            0,
        )
        .expect("from_flat should succeed");
        assert!(approx_eq(g.rot[0], 0.9, 1e-6), "w={}", g.rot[0]);
        assert!(approx_eq(g.rot[1], 0.1, 1e-6), "x={}", g.rot[1]);
        assert!(approx_eq(g.rot[2], 0.2, 1e-6), "y={}", g.rot[2]);
        assert!(approx_eq(g.rot[3], 0.3, 1e-6), "z={}", g.rot[3]);
    }
    #[test]
    fn test_from_flat_out_of_bounds() {
        let positions = vec![1.0f32, 2.0, 3.0];
        let rotations = vec![0.0f32; 4];
        let scales = vec![0.0f32; 3];
        let opacities = vec![0.0f32];
        let sh_dc = vec![0.0f32; 3];
        let sh_rest: Vec<f32> = vec![];
        let result = PlyGaussian::from_flat(
            PlyFlatSlices {
                positions: &positions,
                rotations: &rotations,
                scales: &scales,
                opacities: &opacities,
                sh_dc: &sh_dc,
                sh_rest: &sh_rest,
                n_rest: 0,
            },
            1,
        );
        assert!(result.is_err(), "out-of-bounds idx should return error");
    }
    #[test]
    fn test_from_flat_with_sh_rest() {
        let positions = vec![0.0f32; 3];
        let rotations = vec![0.0f32, 0.0, 0.0, 1.0];
        let scales = vec![0.0f32; 3];
        let opacities = vec![0.0f32];
        let sh_dc = vec![0.1f32, 0.2, 0.3];
        let sh_rest: Vec<f32> = (0..9).map(|i| i as f32 * 0.01).collect();
        let g = PlyGaussian::from_flat(
            PlyFlatSlices {
                positions: &positions,
                rotations: &rotations,
                scales: &scales,
                opacities: &opacities,
                sh_dc: &sh_dc,
                sh_rest: &sh_rest,
                n_rest: 9,
            },
            0,
        )
        .expect("from_flat with rest should succeed");
        assert_eq!(g.f_rest.len(), 9);
        assert!(approx_eq(g.f_rest[0], 0.0, 1e-6));
        assert!(approx_eq(g.f_rest[8], 0.08, 1e-5));
    }
    #[test]
    fn test_build_header_starts_with_ply() {
        let h = ply_build_header(100, 0, PlyFormat::Ascii);
        assert!(h.starts_with("ply\n"));
    }
    #[test]
    fn test_build_header_element_vertex_count() {
        let h = ply_build_header(42, 0, PlyFormat::Ascii);
        assert!(h.contains("element vertex 42"), "header:\n{}", h);
    }
    #[test]
    fn test_build_header_contains_format() {
        let h = ply_build_header(1, 0, PlyFormat::BinaryLittleEndian);
        assert!(h.contains("format binary_little_endian 1.0"));
    }
    #[test]
    fn test_build_header_property_count_no_rest() {
        let h = ply_build_header(1, 0, PlyFormat::Ascii);
        let prop_count = h
            .lines()
            .filter(|l| l.starts_with("property float"))
            .count();
        assert_eq!(prop_count, 17, "expected 17 properties, got {}", prop_count);
    }
    #[test]
    fn test_build_header_property_count_with_rest_9() {
        let h = ply_build_header(1, 9, PlyFormat::Ascii);
        let prop_count = h
            .lines()
            .filter(|l| l.starts_with("property float"))
            .count();
        assert_eq!(prop_count, 26);
    }
    #[test]
    fn test_build_header_ends_with_end_header() {
        let h = ply_build_header(1, 0, PlyFormat::Ascii);
        assert!(h.ends_with("end_header\n"));
    }
    #[test]
    fn test_parse_element_count_vertex() {
        let header = "ply\nformat ascii 1.0\nelement vertex 1234\nend_header\n";
        let n = ply_parse_element_count(header, "vertex").expect("should parse");
        assert_eq!(n, 1234);
    }
    #[test]
    fn test_parse_element_count_missing_element() {
        let header = "ply\nformat ascii 1.0\nend_header\n";
        let result = ply_parse_element_count(header, "vertex");
        assert!(result.is_err());
    }
    #[test]
    fn test_parse_format_ascii() {
        let header = "ply\nformat ascii 1.0\nelement vertex 1\nend_header\n";
        let fmt = ply_parse_format(header).expect("should parse");
        assert_eq!(fmt, PlyFormat::Ascii);
    }
    #[test]
    fn test_parse_format_binary_le() {
        let header = "ply\nformat binary_little_endian 1.0\nelement vertex 1\nend_header\n";
        let fmt = ply_parse_format(header).expect("should parse");
        assert_eq!(fmt, PlyFormat::BinaryLittleEndian);
    }
    #[test]
    fn test_parse_format_binary_be() {
        let header = "ply\nformat binary_big_endian 1.0\nelement vertex 1\nend_header\n";
        let fmt = ply_parse_format(header).expect("should parse");
        assert_eq!(fmt, PlyFormat::BinaryBigEndian);
    }
    #[test]
    fn test_parse_format_unknown() {
        let header = "ply\nformat zstd_compressed 1.0\nelement vertex 1\nend_header\n";
        let result = ply_parse_format(header);
        assert!(matches!(result, Err(PlyError::UnsupportedFormat(_))));
    }
    #[test]
    fn test_parse_format_missing_line() {
        let header = "ply\nelement vertex 1\nend_header\n";
        let result = ply_parse_format(header);
        assert!(matches!(result, Err(PlyError::InvalidHeader(_))));
    }
    #[test]
    fn test_parse_properties_contains_xyz() {
        let header = ply_build_header(1, 0, PlyFormat::Ascii);
        let props = ply_parse_properties(&header).expect("should parse");
        assert!(props.contains(&"x".to_owned()));
        assert!(props.contains(&"y".to_owned()));
        assert!(props.contains(&"z".to_owned()));
    }
    #[test]
    fn test_parse_properties_contains_opacity() {
        let header = ply_build_header(1, 0, PlyFormat::Ascii);
        let props = ply_parse_properties(&header).expect("should parse");
        assert!(props.contains(&"opacity".to_owned()));
    }
    #[test]
    fn test_parse_properties_order_xyz_first() {
        let header = ply_build_header(1, 0, PlyFormat::Ascii);
        let props = ply_parse_properties(&header).expect("should parse");
        assert_eq!(&props[0], "x");
        assert_eq!(&props[1], "y");
        assert_eq!(&props[2], "z");
    }
    #[test]
    fn test_ascii_roundtrip_single() {
        let path = temp_ply_path("ascii_single");
        let gaussians = make_test_gaussians(1, 0);
        ply_write_ascii(&path, &gaussians).expect("write failed");
        let (read_back, _) = ply_read(&path).expect("read failed");
        let _ = std::fs::remove_file(&path);
        assert_eq!(read_back.len(), 1);
        assert!(approx_eq(read_back[0].x, gaussians[0].x, 1e-5));
        assert!(approx_eq(read_back[0].opacity, gaussians[0].opacity, 1e-5));
    }
    #[test]
    fn test_ascii_roundtrip_ten() {
        let path = temp_ply_path("ascii_ten");
        let gaussians = make_test_gaussians(10, 0);
        ply_write_ascii(&path, &gaussians).expect("write failed");
        let (read_back, stats) = ply_read(&path).expect("read failed");
        let _ = std::fs::remove_file(&path);
        assert_eq!(read_back.len(), 10);
        assert_eq!(stats.n_gaussians, 10);
        for i in 0..10 {
            assert!(
                approx_eq(read_back[i].x, gaussians[i].x, 1e-5),
                "x[{}] mismatch",
                i
            );
            assert!(
                approx_eq(read_back[i].y, gaussians[i].y, 1e-5),
                "y[{}] mismatch",
                i
            );
        }
    }
    #[test]
    fn test_binary_roundtrip_single() {
        let path = temp_ply_path("binary_single");
        let gaussians = make_test_gaussians(1, 0);
        ply_write_binary(&path, &gaussians).expect("write failed");
        let (read_back, _) = ply_read(&path).expect("read failed");
        let _ = std::fs::remove_file(&path);
        assert_eq!(read_back.len(), 1);
        assert_eq!(read_back[0].x, gaussians[0].x);
        assert_eq!(read_back[0].opacity, gaussians[0].opacity);
    }
    #[test]
    fn test_binary_roundtrip_ten() {
        let path = temp_ply_path("binary_ten");
        let gaussians = make_test_gaussians(10, 0);
        ply_write_binary(&path, &gaussians).expect("write failed");
        let (read_back, stats) = ply_read(&path).expect("read failed");
        let _ = std::fs::remove_file(&path);
        assert_eq!(read_back.len(), 10);
        assert_eq!(stats.format, PlyFormat::BinaryLittleEndian);
        for i in 0..10 {
            assert_eq!(read_back[i].x, gaussians[i].x, "x[{}]", i);
            assert_eq!(read_back[i].opacity, gaussians[i].opacity, "opacity[{}]", i);
        }
    }
    #[test]
    fn test_write_empty_returns_empty_scene_error() {
        let path = temp_ply_path("empty_scene");
        let result = ply_write(&path, &[], PlyFormat::Ascii);
        assert!(matches!(result, Err(PlyError::EmptyScene)));
    }
    #[test]
    fn test_write_big_endian_returns_unsupported_error() {
        let path = temp_ply_path("big_endian_write");
        let gaussians = make_test_gaussians(1, 0);
        let result = ply_write(&path, &gaussians, PlyFormat::BinaryBigEndian);
        assert!(matches!(result, Err(PlyError::UnsupportedFormat(_))));
    }
    #[test]
    fn test_read_nonexistent_file() {
        let path = temp_ply_path("definitely_does_not_exist_xyz");
        let result = ply_read(&path);
        assert!(matches!(result, Err(PlyError::Io(_))));
    }
    #[test]
    fn test_export_import_positions_roundtrip() {
        let path = temp_ply_path("scene_positions");
        let n = 5usize;
        let positions: Vec<f32> = (0..n * 3).map(|i| i as f32 * 0.1).collect();
        let rotations: Vec<f32> = (0..n).flat_map(|_| [0.0f32, 0.0, 0.0, 1.0]).collect();
        let scales: Vec<f32> = vec![0.0; n * 3];
        let opacities: Vec<f32> = vec![0.0; n];
        let sh_dc: Vec<f32> = vec![0.0; n * 3];
        let sh_rest: Vec<f32> = vec![];
        ply_export_scene(
            &path,
            PlyExportParams {
                positions: &positions,
                rotations: &rotations,
                scales: &scales,
                opacities: &opacities,
                sh_dc: &sh_dc,
                sh_rest: &sh_rest,
                n_rest_per_gaussian: 0,
                format: PlyFormat::BinaryLittleEndian,
            },
        )
        .expect("export failed");
        let data = ply_import_scene(&path).expect("import failed");
        let _ = std::fs::remove_file(&path);
        assert_eq!(data.n_gaussians, n);
        for (i, (&got, &expected)) in data
            .positions
            .iter()
            .zip(positions.iter())
            .enumerate()
            .take(n * 3)
        {
            assert!(
                approx_eq(got, expected, 1e-6),
                "pos[{}]: {} vs {}",
                i,
                got,
                expected
            );
        }
    }
    #[test]
    fn test_export_import_quaternion_roundtrip() {
        let path = temp_ply_path("scene_quaternion");
        let positions: Vec<f32> = vec![0.0; 3];
        let rotations: Vec<f32> = vec![0.1, 0.2, 0.3, 0.9274_f32];
        let scales: Vec<f32> = vec![0.0; 3];
        let opacities: Vec<f32> = vec![0.0];
        let sh_dc: Vec<f32> = vec![0.0; 3];
        let sh_rest: Vec<f32> = vec![];
        ply_export_scene(
            &path,
            PlyExportParams {
                positions: &positions,
                rotations: &rotations,
                scales: &scales,
                opacities: &opacities,
                sh_dc: &sh_dc,
                sh_rest: &sh_rest,
                n_rest_per_gaussian: 0,
                format: PlyFormat::BinaryLittleEndian,
            },
        )
        .expect("export failed");
        let data = ply_import_scene(&path).expect("import failed");
        let _ = std::fs::remove_file(&path);
        assert!(
            approx_eq(data.rotations[0], 0.1, 1e-5),
            "qx={}",
            data.rotations[0]
        );
        assert!(
            approx_eq(data.rotations[1], 0.2, 1e-5),
            "qy={}",
            data.rotations[1]
        );
        assert!(
            approx_eq(data.rotations[2], 0.3, 1e-5),
            "qz={}",
            data.rotations[2]
        );
        assert!(
            approx_eq(data.rotations[3], 0.9274, 1e-5),
            "qw={}",
            data.rotations[3]
        );
    }
    #[test]
    fn test_export_dimension_mismatch_positions() {
        let path = temp_ply_path("scene_dim_mismatch");
        let positions: Vec<f32> = vec![0.0; 2];
        let rotations: Vec<f32> = vec![0.0, 0.0, 0.0, 1.0];
        let scales: Vec<f32> = vec![0.0; 3];
        let opacities: Vec<f32> = vec![0.0];
        let sh_dc: Vec<f32> = vec![0.0; 3];
        let sh_rest: Vec<f32> = vec![];
        let result = ply_export_scene(
            &path,
            PlyExportParams {
                positions: &positions,
                rotations: &rotations,
                scales: &scales,
                opacities: &opacities,
                sh_dc: &sh_dc,
                sh_rest: &sh_rest,
                n_rest_per_gaussian: 0,
                format: PlyFormat::Ascii,
            },
        );
        assert!(
            matches!(result, Err(PlyError::DimensionMismatch { .. })),
            "expected DimensionMismatch"
        );
    }
    #[test]
    fn test_import_populates_scene_data_correctly() {
        let path = temp_ply_path("scene_populate");
        let n = 3usize;
        let positions: Vec<f32> = (0..n * 3).map(|i| i as f32).collect();
        let rotations: Vec<f32> = (0..n).flat_map(|_| [0.0f32, 0.0, 0.0, 1.0]).collect();
        let scales: Vec<f32> = vec![0.5; n * 3];
        let opacities: Vec<f32> = vec![-1.0, 0.0, 1.0];
        let sh_dc: Vec<f32> = vec![0.1; n * 3];
        let sh_rest: Vec<f32> = vec![];
        ply_export_scene(
            &path,
            PlyExportParams {
                positions: &positions,
                rotations: &rotations,
                scales: &scales,
                opacities: &opacities,
                sh_dc: &sh_dc,
                sh_rest: &sh_rest,
                n_rest_per_gaussian: 0,
                format: PlyFormat::Ascii,
            },
        )
        .expect("export failed");
        let data = ply_import_scene(&path).expect("import failed");
        let _ = std::fs::remove_file(&path);
        assert_eq!(data.n_gaussians, n);
        assert_eq!(data.n_rest_per_gaussian, 0);
        assert_eq!(data.positions.len(), n * 3);
        assert_eq!(data.rotations.len(), n * 4);
        assert_eq!(data.scales.len(), n * 3);
        assert_eq!(data.opacities.len(), n);
        assert!(approx_eq(data.opacities[0], -1.0, 1e-5));
        assert!(approx_eq(data.opacities[2], 1.0, 1e-5));
    }
    #[test]
    fn test_scene_stats_bbox_from_known_positions() {
        let gaussians = vec![
            PlyGaussian {
                x: 1.0,
                y: 2.0,
                z: 3.0,
                ..PlyGaussian::identity()
            },
            PlyGaussian {
                x: -1.0,
                y: 5.0,
                z: 0.5,
                ..PlyGaussian::identity()
            },
            PlyGaussian {
                x: 3.0,
                y: -1.0,
                z: 2.0,
                ..PlyGaussian::identity()
            },
        ];
        let stats = ply_compute_scene_stats(&gaussians);
        assert!(approx_eq(stats.bbox_min[0], -1.0, 1e-6));
        assert!(approx_eq(stats.bbox_min[1], -1.0, 1e-6));
        assert!(approx_eq(stats.bbox_min[2], 0.5, 1e-6));
        assert!(approx_eq(stats.bbox_max[0], 3.0, 1e-6));
        assert!(approx_eq(stats.bbox_max[1], 5.0, 1e-6));
        assert!(approx_eq(stats.bbox_max[2], 3.0, 1e-6));
    }
    #[test]
    fn test_scene_stats_mean_opacity_sigmoid_applied() {
        let gaussians = vec![
            PlyGaussian {
                opacity: 0.0,
                ..PlyGaussian::identity()
            },
            PlyGaussian {
                opacity: 0.0,
                ..PlyGaussian::identity()
            },
        ];
        let stats = ply_compute_scene_stats(&gaussians);
        assert!(approx_eq(stats.mean_opacity, 0.5, 1e-5));
    }
    #[test]
    fn test_scene_stats_empty_slice() {
        let stats = ply_compute_scene_stats(&[]);
        assert_eq!(stats.n_gaussians, 0);
        assert_eq!(stats.sh_degree, 0);
        assert!(approx_eq(stats.mean_opacity, 0.0, 1e-6));
    }
    #[test]
    fn test_scene_stats_n_gaussians_count() {
        let gaussians = make_test_gaussians(7, 0);
        let stats = ply_compute_scene_stats(&gaussians);
        assert_eq!(stats.n_gaussians, 7);
    }
    #[test]
    fn test_format_stats_non_empty() {
        let gaussians = make_test_gaussians(5, 0);
        let stats = ply_compute_scene_stats(&gaussians);
        let s = ply_format_stats(&stats);
        assert!(!s.is_empty());
        assert!(s.contains("Gaussians: 5"));
    }
    #[test]
    fn test_format_write_stats_contains_n_gaussians() {
        let path = temp_ply_path("format_write_stats");
        let gaussians = make_test_gaussians(3, 0);
        let write_stats = ply_write(&path, &gaussians, PlyFormat::Ascii).expect("write failed");
        let _ = std::fs::remove_file(&path);
        let s = ply_format_write_stats(&write_stats);
        assert!(s.contains('3'), "should contain n_gaussians=3, got: {}", s);
    }
    #[test]
    fn test_write_stats_n_gaussians_field() {
        let path = temp_ply_path("write_stats_n");
        let gaussians = make_test_gaussians(4, 0);
        let stats = ply_write(&path, &gaussians, PlyFormat::Ascii).expect("write failed");
        let _ = std::fs::remove_file(&path);
        assert_eq!(stats.n_gaussians, 4);
    }
    #[test]
    fn test_write_stats_file_size_nonzero() {
        let path = temp_ply_path("write_stats_size");
        let gaussians = make_test_gaussians(2, 0);
        let stats =
            ply_write(&path, &gaussians, PlyFormat::BinaryLittleEndian).expect("write failed");
        let _ = std::fs::remove_file(&path);
        assert!(stats.file_size_bytes > 0);
    }
    #[test]
    fn test_sh_degree_0_roundtrip() {
        let path = temp_ply_path("sh0");
        let gaussians = make_test_gaussians(3, 0);
        ply_write_binary(&path, &gaussians).expect("write");
        let (read_back, stats) = ply_read(&path).expect("read");
        let _ = std::fs::remove_file(&path);
        assert_eq!(stats.sh_degree, 0);
        assert_eq!(read_back[0].f_rest.len(), 0);
    }
    #[test]
    fn test_sh_degree_1_roundtrip() {
        let path = temp_ply_path("sh1");
        let gaussians = make_test_gaussians(2, 1);
        ply_write_binary(&path, &gaussians).expect("write");
        let (read_back, stats) = ply_read(&path).expect("read");
        let _ = std::fs::remove_file(&path);
        assert_eq!(stats.sh_degree, 1);
        assert_eq!(read_back[0].f_rest.len(), 9);
        for i in 0..2 {
            for j in 0..9 {
                assert!(
                    approx_eq(read_back[i].f_rest[j], gaussians[i].f_rest[j], 1e-6),
                    "f_rest[{}][{}] mismatch",
                    i,
                    j
                );
            }
        }
    }
    #[test]
    fn test_sh_degree_3_roundtrip() {
        let path = temp_ply_path("sh3");
        let gaussians = make_test_gaussians(2, 3);
        ply_write_binary(&path, &gaussians).expect("write");
        let (read_back, stats) = ply_read(&path).expect("read");
        let _ = std::fs::remove_file(&path);
        assert_eq!(stats.sh_degree, 3);
        assert_eq!(read_back[0].f_rest.len(), 45);
    }
    #[test]
    fn test_ascii_sh_degree_1_roundtrip() {
        let path = temp_ply_path("ascii_sh1");
        let gaussians = make_test_gaussians(3, 1);
        ply_write_ascii(&path, &gaussians).expect("write");
        let (read_back, stats) = ply_read(&path).expect("read");
        let _ = std::fs::remove_file(&path);
        assert_eq!(stats.sh_degree, 1);
        for i in 0..3 {
            assert!(approx_eq(read_back[i].x, gaussians[i].x, 1e-5));
            assert_eq!(read_back[i].f_rest.len(), 9);
        }
    }
    #[test]
    fn test_ascii_roundtrip_rotation_values() {
        let path = temp_ply_path("ascii_rot");
        let mut g = PlyGaussian::identity();
        g.rot = [
            std::f32::consts::FRAC_1_SQRT_2,
            std::f32::consts::FRAC_1_SQRT_2,
            0.0,
            0.0,
        ];
        ply_write_ascii(&path, &[g.clone()]).expect("write");
        let (read_back, _) = ply_read(&path).expect("read");
        let _ = std::fs::remove_file(&path);
        for k in 0..4 {
            assert!(
                approx_eq(read_back[0].rot[k], g.rot[k], 1e-5),
                "rot[{}]: {} vs {}",
                k,
                read_back[0].rot[k],
                g.rot[k]
            );
        }
    }
    #[test]
    fn test_binary_roundtrip_scale_values() {
        let path = temp_ply_path("bin_scale");
        let mut g = PlyGaussian::identity();
        g.scale = [-1.5, 0.5, 2.3];
        ply_write_binary(&path, &[g.clone()]).expect("write");
        let (read_back, _) = ply_read(&path).expect("read");
        let _ = std::fs::remove_file(&path);
        for k in 0..3 {
            assert_eq!(read_back[0].scale[k], g.scale[k], "scale[{}]", k);
        }
    }
}
