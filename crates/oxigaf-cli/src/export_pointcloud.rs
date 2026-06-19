//! Point cloud export for 3D Gaussian Splatting models.
//!
//! Converts a [`GaussianModel`] to a colored PLY point cloud in binary
//! little-endian format, extracting Gaussian center positions and colors
//! derived from SH DC coefficients (degree-0 spherical harmonics).
//!
//! # Output format
//!
//! Binary little-endian PLY with the following properties per vertex:
//!
//! | Property | Type   | Description                        |
//! |----------|--------|------------------------------------|
//! | x        | float  | World-space X position             |
//! | y        | float  | World-space Y position             |
//! | z        | float  | World-space Z position             |
//! | nx       | float  | Normal X (always 0.0)              |
//! | ny       | float  | Normal Y (always 0.0)              |
//! | nz       | float  | Normal Z (always 0.0)              |
//! | red      | uchar  | Red channel [0, 255]               |
//! | green    | uchar  | Green channel [0, 255]             |
//! | blue     | uchar  | Blue channel [0, 255]              |
//!
//! # SH DC to color conversion
//!
//! ```text
//! SH_C0 = 0.28209479177387814
//! color_linear = clamp(0.5 + SH_C0 * f_dc, 0.0, 1.0)
//! color_u8     = round(color_linear * 255) as u8
//! ```
//!
//! The DC component for Gaussian `i` is stored at:
//! `sh_coeffs[i * C + 0]`, `[i * C + 1]`, `[i * C + 2]`
//! where `C = (sh_degree + 1)² × 3`.

use std::io::{BufWriter, Write};
use std::path::Path;

use oxigaf::render::gaussian::GaussianModel;

use crate::cli::PointColorMode;
use crate::error::CliError;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// SH C0 coefficient (degree 0, mode 0): 1 / (2 * sqrt(pi)).
const SH_C0: f32 = 0.282_094_79;

// ---------------------------------------------------------------------------
// Color helpers
// ---------------------------------------------------------------------------

/// Convert a single SH DC coefficient to a sRGB u8 value.
///
/// Formula: `round(clamp(0.5 + SH_C0 * dc, 0.0, 1.0) * 255)`.
#[must_use]
pub fn sh_dc_to_u8(dc: f32) -> u8 {
    let linear = (0.5_f32 + SH_C0 * dc).clamp(0.0, 1.0);
    (linear * 255.0).round() as u8
}

/// Compute the sigmoid function: `1 / (1 + exp(-x))`.
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0_f32 / (1.0_f32 + (-x).exp())
}

/// Map a value in `[0, 1]` to an RGB triplet using a simple HSV rainbow sweep.
///
/// Hue goes from blue (0.0) → cyan → green → yellow → red (1.0).
fn rainbow_rgb(t: f32) -> [u8; 3] {
    let t = t.clamp(0.0, 1.0);
    // Hue in [0, 240] degrees (blue to red sweep through cyan/green/yellow).
    let hue = (1.0 - t) * 240.0;
    let h = hue / 60.0;
    let i = h.floor() as u32;
    let f = h - h.floor();
    let q = (1.0 - f) * 255.0;
    let p = f * 255.0;
    match i {
        0 => [255, p as u8, 0],
        1 => [q as u8, 255, 0],
        2 => [0, 255, p as u8],
        3 => [0, q as u8, 255],
        _ => [p as u8, 0, 255],
    }
}

// ---------------------------------------------------------------------------
// Public API: gaussian_colors
// ---------------------------------------------------------------------------

/// Compute per-Gaussian RGB colors according to the requested [`PointColorMode`].
///
/// Returns a `Vec<[u8; 3]>` with one entry per Gaussian (same length as
/// `model.len()`). Returns an empty vector for empty models.
#[must_use]
pub fn gaussian_colors(model: &GaussianModel, mode: PointColorMode) -> Vec<[u8; 3]> {
    let n = model.len();
    if n == 0 {
        return Vec::new();
    }

    let sh_channels = ((model.sh_degree + 1).pow(2) * 3) as usize;

    match mode {
        PointColorMode::ShDc => (0..n)
            .map(|i| {
                let base = i * sh_channels;
                let r = model.sh_coeffs.get(base).copied().unwrap_or(0.0);
                let g = model.sh_coeffs.get(base + 1).copied().unwrap_or(0.0);
                let b = model.sh_coeffs.get(base + 2).copied().unwrap_or(0.0);
                [sh_dc_to_u8(r), sh_dc_to_u8(g), sh_dc_to_u8(b)]
            })
            .collect(),
        PointColorMode::White => vec![[255u8, 255u8, 255u8]; n],
        PointColorMode::Opacity => model
            .gaussians
            .iter()
            .map(|g| {
                let alpha = sigmoid(g.opacity);
                let v = (alpha * 255.0).round() as u8;
                [v, v, v]
            })
            .collect(),
        PointColorMode::Scale => {
            // Compute average scale magnitudes to normalize across all Gaussians.
            let magnitudes: Vec<f32> = model
                .gaussians
                .iter()
                .map(|g| {
                    // Scale is stored as log-scale; exponentiate to get actual scale.
                    let sx = g.scale[0].exp();
                    let sy = g.scale[1].exp();
                    let sz = g.scale[2].exp();
                    (sx + sy + sz) / 3.0
                })
                .collect();

            let max_mag = magnitudes.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let min_mag = magnitudes.iter().copied().fold(f32::INFINITY, f32::min);
            let range = (max_mag - min_mag).max(f32::EPSILON);

            magnitudes
                .iter()
                .map(|&m| rainbow_rgb((m - min_mag) / range))
                .collect()
        }
    }
}

// ---------------------------------------------------------------------------
// Public API: export_pointcloud
// ---------------------------------------------------------------------------

/// Export a [`GaussianModel`] as a colored binary little-endian PLY point cloud.
///
/// The output file uses the following header:
/// ```text
/// ply
/// format binary_little_endian 1.0
/// element vertex N
/// property float x
/// property float y
/// property float z
/// property float nx
/// property float ny
/// property float nz
/// property uchar red
/// property uchar green
/// property uchar blue
/// end_header
/// ```
///
/// Binary payload per point: 6 × f32 (xyz + normals) + 3 × u8 (rgb) = 27 bytes.
pub fn export_pointcloud(
    model: &GaussianModel,
    output_path: &Path,
    color_mode: PointColorMode,
) -> Result<(), CliError> {
    // Ensure parent directory exists.
    if let Some(parent) = output_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).map_err(|e| {
                CliError::PointCloudExport(format!(
                    "Failed to create output directory '{}': {e}",
                    parent.display()
                ))
            })?;
        }
    }

    let file = std::fs::File::create(output_path).map_err(|e| {
        CliError::PointCloudExport(format!(
            "Failed to create output file '{}': {e}",
            output_path.display()
        ))
    })?;
    let mut writer = BufWriter::new(file);

    let n = model.len();
    let colors = gaussian_colors(model, color_mode);

    // ----- Write PLY header (ASCII) -----
    write_header(&mut writer, n)
        .map_err(|e| CliError::PointCloudExport(format!("Failed to write PLY header: {e}")))?;

    // ----- Write binary payload -----
    for (i, g) in model.gaussians.iter().enumerate() {
        let [r, green, b] = colors.get(i).copied().unwrap_or([128u8, 128u8, 128u8]);

        // xyz (3 × f32 little-endian)
        writer
            .write_all(&g.position[0].to_le_bytes())
            .map_err(|e| CliError::PointCloudExport(format!("Write error (x): {e}")))?;
        writer
            .write_all(&g.position[1].to_le_bytes())
            .map_err(|e| CliError::PointCloudExport(format!("Write error (y): {e}")))?;
        writer
            .write_all(&g.position[2].to_le_bytes())
            .map_err(|e| CliError::PointCloudExport(format!("Write error (z): {e}")))?;

        // normals (3 × f32 = 0.0 each)
        writer
            .write_all(&0.0_f32.to_le_bytes())
            .map_err(|e| CliError::PointCloudExport(format!("Write error (nx): {e}")))?;
        writer
            .write_all(&0.0_f32.to_le_bytes())
            .map_err(|e| CliError::PointCloudExport(format!("Write error (ny): {e}")))?;
        writer
            .write_all(&0.0_f32.to_le_bytes())
            .map_err(|e| CliError::PointCloudExport(format!("Write error (nz): {e}")))?;

        // rgb (3 × u8)
        writer
            .write_all(&[r, green, b])
            .map_err(|e| CliError::PointCloudExport(format!("Write error (rgb): {e}")))?;
    }

    writer.flush().map_err(|e| {
        CliError::PointCloudExport(format!(
            "Failed to flush output file '{}': {e}",
            output_path.display()
        ))
    })?;

    tracing::info!(
        "Wrote point cloud: {} Gaussians ({:?}) → {}",
        n,
        color_mode,
        output_path.display(),
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Write the ASCII PLY header to `writer`.
fn write_header<W: Write>(writer: &mut W, num_points: usize) -> std::io::Result<()> {
    writeln!(writer, "ply")?;
    writeln!(writer, "format binary_little_endian 1.0")?;
    writeln!(writer, "comment Generated by OxiGAF")?;
    writeln!(writer, "element vertex {num_points}")?;
    writeln!(writer, "property float x")?;
    writeln!(writer, "property float y")?;
    writeln!(writer, "property float z")?;
    writeln!(writer, "property float nx")?;
    writeln!(writer, "property float ny")?;
    writeln!(writer, "property float nz")?;
    writeln!(writer, "property uchar red")?;
    writeln!(writer, "property uchar green")?;
    writeln!(writer, "property uchar blue")?;
    writeln!(writer, "end_header")?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use oxigaf::render::gaussian::{GaussianAttributes, GaussianModel};

    // -----------------------------------------------------------------------
    // Test helpers
    // -----------------------------------------------------------------------

    /// Construct a minimal `GaussianModel` with `n` Gaussians.
    fn make_model(n: usize, sh_degree: u32) -> GaussianModel {
        let sh_channels = ((sh_degree + 1).pow(2) * 3) as usize;
        let gaussians: Vec<GaussianAttributes> = (0..n)
            .map(|i| {
                let f = i as f32 * 0.1;
                GaussianAttributes {
                    position: [f, f + 0.1, f + 0.2],
                    _pad0: 0.0,
                    rotation: [0.0, 0.0, 0.0, 1.0],
                    scale: [0.01_f32.ln(), 0.01_f32.ln(), 0.01_f32.ln()],
                    opacity: 0.0, // sigmoid(0) = 0.5
                }
            })
            .collect();
        let sh_coeffs = vec![0.0_f32; n * sh_channels];
        GaussianModel {
            gaussians,
            sh_coeffs,
            sh_degree,
            face_indices: vec![0u32; n],
            barycentric: vec![[1.0_f32 / 3.0; 3]; n],
            local_offsets: vec![[0.0_f32; 3]; n],
            is_rigid: vec![true; n],
        }
    }

    /// Return a unique temp directory path for each test.
    fn temp_dir() -> std::path::PathBuf {
        let base = std::env::temp_dir();
        let id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0);
        base.join(format!("oxigaf_pc_test_{id}"))
    }

    // -----------------------------------------------------------------------
    // Test 1: sh_dc_to_u8(0.0) == 128
    // -----------------------------------------------------------------------
    #[test]
    fn test_sh_dc_to_u8_zero_gives_128() {
        // 0.5 + SH_C0 * 0.0 = 0.5 → round(0.5 * 255) = 128 (rounds toward even or up).
        let result = sh_dc_to_u8(0.0);
        // 0.5 * 255.0 = 127.5 → rounds to 128
        assert_eq!(result, 128, "sh_dc_to_u8(0.0) must equal 128");
    }

    // -----------------------------------------------------------------------
    // Test 2: sh_dc_to_u8(large positive) → clamped to 255
    // -----------------------------------------------------------------------
    #[test]
    fn test_sh_dc_to_u8_large_positive_gives_255() {
        // 1.0 / SH_C0 ≈ 3.544 → 0.5 + 1.0 = 1.5 → clamp → 1.0 → 255
        let big = 1.0 / SH_C0;
        assert_eq!(sh_dc_to_u8(big), 255, "large positive DC must clamp to 255");
    }

    // -----------------------------------------------------------------------
    // Test 3: sh_dc_to_u8(large negative) → clamped to 0
    // -----------------------------------------------------------------------
    #[test]
    fn test_sh_dc_to_u8_large_negative_gives_0() {
        // -1.0 / SH_C0 → 0.5 - 1.0 = -0.5 → clamp → 0.0 → 0
        let big_neg = -1.0 / SH_C0;
        assert_eq!(sh_dc_to_u8(big_neg), 0, "large negative DC must clamp to 0");
    }

    // -----------------------------------------------------------------------
    // Test 4: gaussian_colors for empty model returns empty vec
    // -----------------------------------------------------------------------
    #[test]
    fn test_gaussian_colors_empty_model() {
        let model = make_model(0, 1);
        let colors = gaussian_colors(&model, PointColorMode::ShDc);
        assert!(colors.is_empty(), "empty model must produce empty colors");
    }

    // -----------------------------------------------------------------------
    // Test 5: gaussian_colors ShDc mode returns correct colors
    // -----------------------------------------------------------------------
    #[test]
    fn test_gaussian_colors_sh_dc_correct() {
        // Build a model with known sh_coeffs: dc = 0.0 → color 128 each
        let model = make_model(3, 0); // sh_degree=0, C=3, dc at [0..3)
        let colors = gaussian_colors(&model, PointColorMode::ShDc);
        assert_eq!(colors.len(), 3);
        for c in &colors {
            // sh_dc_to_u8(0.0) = 128
            assert_eq!(c[0], 128);
            assert_eq!(c[1], 128);
            assert_eq!(c[2], 128);
        }
    }

    // -----------------------------------------------------------------------
    // Test 6: gaussian_colors White mode → all (255, 255, 255)
    // -----------------------------------------------------------------------
    #[test]
    fn test_gaussian_colors_white_mode() {
        let model = make_model(5, 1);
        let colors = gaussian_colors(&model, PointColorMode::White);
        assert_eq!(colors.len(), 5);
        for c in colors {
            assert_eq!(
                c,
                [255u8, 255u8, 255u8],
                "White mode must return pure white"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Test 7: gaussian_colors Opacity mode → proportional to sigmoid(opacity)
    // -----------------------------------------------------------------------
    #[test]
    fn test_gaussian_colors_opacity_mode() {
        // opacity = 0.0 → sigmoid(0.0) = 0.5 → v = 128
        let model = make_model(4, 1);
        let colors = gaussian_colors(&model, PointColorMode::Opacity);
        assert_eq!(colors.len(), 4);
        for c in colors {
            let expected = (sigmoid(0.0) * 255.0).round() as u8;
            assert_eq!(c[0], expected, "Opacity mode grayscale incorrect");
            assert_eq!(c[1], expected);
            assert_eq!(c[2], expected);
        }
    }

    // -----------------------------------------------------------------------
    // Test 8: export_pointcloud creates a file
    // -----------------------------------------------------------------------
    #[test]
    fn test_export_pointcloud_creates_file() {
        let dir = temp_dir();
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let out = dir.join("cloud.ply");
        let model = make_model(10, 1);

        let result = export_pointcloud(&model, &out, PointColorMode::ShDc);
        assert!(result.is_ok(), "export must succeed: {:?}", result);
        assert!(out.exists(), "output file must be created");
    }

    // -----------------------------------------------------------------------
    // Test 9: created file starts with "ply" magic bytes
    // -----------------------------------------------------------------------
    #[test]
    fn test_export_pointcloud_starts_with_ply() {
        let dir = temp_dir();
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let out = dir.join("magic_check.ply");
        let model = make_model(5, 0);

        export_pointcloud(&model, &out, PointColorMode::White).expect("export must succeed");

        let bytes = std::fs::read(&out).expect("must be able to read output");
        assert!(
            bytes.starts_with(b"ply\n"),
            "file must start with PLY magic"
        );
    }

    // -----------------------------------------------------------------------
    // Test 10: File size = header_size + N * 27 bytes
    // -----------------------------------------------------------------------
    #[test]
    fn test_export_pointcloud_file_size_correct() {
        let dir = temp_dir();
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let out = dir.join("size_check.ply");
        let n = 20usize;
        let model = make_model(n, 1);

        export_pointcloud(&model, &out, PointColorMode::ShDc).expect("export must succeed");

        let bytes = std::fs::read(&out).expect("must be able to read output");

        // Find header length (everything up to and including "end_header\n").
        let header_marker = b"end_header\n";
        let header_end = bytes
            .windows(header_marker.len())
            .position(|w| w == header_marker)
            .expect("header end marker must exist");
        let header_len = header_end + header_marker.len();

        // Per point: 6 * 4 (floats) + 3 (u8) = 27 bytes.
        let expected_total = header_len + n * 27;
        assert_eq!(
            bytes.len(),
            expected_total,
            "file size must be header_len + N * 27"
        );
    }

    // -----------------------------------------------------------------------
    // Test 11: export_pointcloud works for an empty model (0 Gaussians)
    // -----------------------------------------------------------------------
    #[test]
    fn test_export_pointcloud_empty_model() {
        let dir = temp_dir();
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let out = dir.join("empty_cloud.ply");
        let model = make_model(0, 1);

        let result = export_pointcloud(&model, &out, PointColorMode::White);
        assert!(
            result.is_ok(),
            "empty model export must succeed: {:?}",
            result
        );
        assert!(out.exists(), "output file must exist even for empty model");
    }

    // -----------------------------------------------------------------------
    // Test 14: Scale color mode produces colors without panic
    // -----------------------------------------------------------------------
    #[test]
    fn test_gaussian_colors_scale_mode_no_panic() {
        let model = make_model(6, 1);
        let colors = gaussian_colors(&model, PointColorMode::Scale);
        assert_eq!(
            colors.len(),
            6,
            "Scale mode must produce one color per Gaussian"
        );
    }
}
