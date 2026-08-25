//! glTF 2.0 export for 3D Gaussian Splatting models.
//!
//! This module is a thin CLI-facing adapter over
//! [`oxigaf::render::gltf::write_gltf`], the workspace's single glTF writer.
//! See that module for the file layout, the binary buffer layout, and the
//! specification requirements the writer satisfies.
//!
//! # Why this is an adapter and not an implementation
//!
//! This file used to carry its own emitter, and it was the *third* one in the
//! workspace to nominally produce "glTF":
//!
//! | Caller | Output | Extension name |
//! |--------|--------|----------------|
//! | `oxigaf export --format gltf` | `.gltf` + `.bin` | `OXIGAF_gaussian_splat` |
//! | `oxigaf::export(.., ExportFormat::Gltf)` | `.gltf` + `.bin` | `OXIGAF_gaussian_splat` |
//! | `ExportStage` (`crate::export::export_gltf`) | self-contained `.glb` | `OXIGAF_gaussians` |
//!
//! Three emitters behind one format name meant a consumer written against one
//! silently mis-read the others. The version kept is the one that was actually
//! spec-conformant: this file's old writer put all five accessors onto a
//! single buffer view with no `byteStride`, which glTF 2.0 forbids for
//! accessors of differing element size.
//!
//! [`export_gltf`] therefore keeps its signature and its
//! [`CliError::GltfExport`] error type — no call site changes — but the bytes
//! it writes now come from the shared writer.

use std::path::Path;

use oxigaf::render::gaussian::GaussianModel;
use oxigaf::render::gltf::{write_gltf, GltfError};

use crate::error::CliError;

/// Export a [`GaussianModel`] to a glTF 2.0 `.gltf` + `.bin` file pair.
///
/// Given `output_path`, the function writes:
/// - `<stem>.gltf` — the JSON glTF document,
/// - `<stem>.bin`  — the binary attribute buffer it references.
///
/// Unlike the shared writer, this wrapper normalises the document's extension
/// to `.gltf` and creates the parent directory, because a CLI takes its output
/// path from the user and `--output model.glb` must not silently produce a
/// file that is not GLB.
///
/// # Errors
///
/// Returns [`CliError::GltfExport`] if the output directory cannot be created,
/// or if either file cannot be created, written or flushed.
pub fn export_gltf(model: &GaussianModel, output_path: &Path) -> Result<(), CliError> {
    // Derive stem and output file paths. `.gltf` is forced rather than taken
    // from the user: the writer emits a JSON document, so any other extension
    // would misdescribe the file — and a `.bin` extension would make the
    // document collide with its own buffer sidecar.
    let stem = output_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model");
    let parent = output_path.parent().unwrap_or_else(|| Path::new("."));
    let gltf_path = parent.join(format!("{stem}.gltf"));

    // Create parent directory if needed.
    if !parent.as_os_str().is_empty() {
        std::fs::create_dir_all(parent).map_err(|e| {
            CliError::GltfExport(format!(
                "Failed to create output directory '{}': {e}",
                parent.display()
            ))
        })?;
    }

    write_gltf(model, &gltf_path).map_err(|e| match e {
        GltfError::InvalidOutputPath(message) => {
            CliError::GltfExport(format!("Invalid output path: {message}"))
        }
        GltfError::Io {
            action,
            path,
            source,
        } => CliError::GltfExport(format!("Failed to {action} '{}': {source}", path.display())),
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use oxigaf::render::gaussian::{GaussianAttributes, GaussianModel};

    /// Create a small GaussianModel with `n` Gaussians and given `sh_degree`.
    fn make_model(n: usize, sh_degree: u32) -> GaussianModel {
        let sh_channels = ((sh_degree + 1).pow(2) * 3) as usize;
        let gaussians: Vec<GaussianAttributes> = (0..n)
            .map(|i| {
                let f = i as f32;
                GaussianAttributes {
                    position: [f, f + 0.1, f + 0.2],
                    _pad0: 0.0,
                    rotation: [0.0, 0.0, 0.0, 1.0],
                    scale: [0.01, 0.01, 0.01],
                    opacity: -1.0,
                }
            })
            .collect();
        let sh_coeffs = vec![0.5_f32; n * sh_channels];
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

    /// Return a fresh, empty temp directory dedicated to one test.
    fn temp_subdir(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("oxigaf_cli_gltf_{name}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("test: temp dir creation should succeed");
        dir
    }

    /// Read a whole file as text, failing the test on error.
    fn read_text(path: &Path) -> String {
        std::fs::read_to_string(path).expect("test: output file should be readable")
    }

    // -----------------------------------------------------------------------
    // File pair
    // -----------------------------------------------------------------------

    #[test]
    fn export_creates_the_document_and_its_buffer() {
        let dir = temp_subdir("two_files");
        export_gltf(&make_model(10, 1), &dir.join("model.gltf"))
            .expect("test: export should succeed");

        assert!(dir.join("model.gltf").is_file(), ".gltf must be created");
        assert!(dir.join("model.bin").is_file(), ".bin must be created");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn export_creates_a_missing_parent_directory() {
        // A CLI takes its output path from the user, so an absent parent is a
        // routine case rather than an error — unlike the shared writer, which
        // deliberately leaves directory creation to its caller.
        let dir = temp_subdir("missing_parent");
        let nested = dir.join("a").join("b");
        export_gltf(&make_model(3, 0), &nested.join("model.gltf"))
            .expect("test: export should create the parent directory");

        assert!(nested.join("model.gltf").is_file());
        assert!(nested.join("model.bin").is_file());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_non_gltf_output_extension_is_normalised() {
        // `--output model.glb` must not yield a JSON document named `.glb`:
        // the writer emits `.gltf` + `.bin`, so the extension is forced to
        // match what is actually written.
        let dir = temp_subdir("normalised_extension");
        export_gltf(&make_model(4, 0), &dir.join("model.glb"))
            .expect("test: export should succeed");

        assert!(dir.join("model.gltf").is_file(), ".gltf must be written");
        assert!(dir.join("model.bin").is_file(), ".bin must be written");
        assert!(
            !dir.join("model.glb").exists(),
            "no file may be written under the misleading .glb name"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_bin_output_path_does_not_clobber_the_buffer() {
        // Regression for the collision the shared writer rejects: because the
        // extension is normalised to `.gltf` first, `--output model.bin` is a
        // *usable* request here rather than an error, and the document and
        // buffer still end up in separate files.
        let dir = temp_subdir("bin_output");
        export_gltf(&make_model(4, 0), &dir.join("model.bin"))
            .expect("test: a .bin output path must still produce a valid pair");

        let doc = read_text(&dir.join("model.gltf"));
        assert!(
            doc.trim_start().starts_with('{'),
            "the document must be JSON, not the binary buffer:\n{doc}"
        );
        assert!(dir.join("model.bin").is_file(), "the buffer must exist");

        let _ = std::fs::remove_dir_all(&dir);
    }

    // -----------------------------------------------------------------------
    // Format unification
    // -----------------------------------------------------------------------

    #[test]
    fn output_is_byte_identical_to_the_shared_writer() {
        // The whole point of this module now: `oxigaf export --format gltf`
        // and `oxigaf::export(.., ExportFormat::Gltf)` must produce the same
        // bytes, because they run the same writer. Any future divergence here
        // reintroduces the incompatible-emitters bug.
        let dir = temp_subdir("byte_identical");
        let model = make_model(6, 2);

        export_gltf(&model, &dir.join("viacli.gltf")).expect("test: CLI export should succeed");
        write_gltf(&model, &dir.join("direct.gltf")).expect("test: direct export should succeed");

        let via_cli = read_text(&dir.join("viacli.gltf")).replace("viacli.bin", "BUF");
        let direct = read_text(&dir.join("direct.gltf")).replace("direct.bin", "BUF");
        assert_eq!(
            via_cli, direct,
            "the CLI adapter must not alter the document beyond its file name"
        );

        let cli_bin = std::fs::read(dir.join("viacli.bin")).expect("test: CLI buffer should exist");
        let direct_bin =
            std::fs::read(dir.join("direct.bin")).expect("test: direct buffer should exist");
        assert_eq!(cli_bin, direct_bin, "the binary buffers must be identical");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn every_accessor_owns_its_own_buffer_view() {
        // Regression for this file's superseded writer, which put all five
        // accessors on ONE strideless buffer view — forbidden by glTF 2.0.
        let dir = temp_subdir("one_view_each");
        export_gltf(&make_model(8, 1), &dir.join("model.gltf"))
            .expect("test: export should succeed");

        let text = read_text(&dir.join("model.gltf"));
        assert_eq!(
            text.matches(r#""buffer": 0"#).count(),
            5,
            "expected five distinct buffer views:\n{text}"
        );
        for view in 0..5 {
            assert!(
                text.contains(&format!(r#""bufferView": {view},"#)),
                "accessor for buffer view {view} is missing:\n{text}"
            );
        }
        assert!(
            !text.contains("byteStride"),
            "tightly-packed per-accessor views need no byteStride"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn empty_model_writes_no_zero_length_buffer() {
        // The old writer emitted a 0-byte `.bin` plus `count: 0` accessors and
        // a `byteLength: 0` buffer view — all forbidden by the glTF 2.0 schema
        // (`minimum: 1`), so every conforming loader rejected the result.
        let dir = temp_subdir("empty");
        export_gltf(&make_model(0, 1), &dir.join("empty.gltf"))
            .expect("test: empty model export should succeed");

        assert!(dir.join("empty.gltf").is_file(), ".gltf must be created");
        assert!(
            !dir.join("empty.bin").exists(),
            "no zero-length .bin may be written"
        );

        let text = read_text(&dir.join("empty.gltf"));
        for forbidden in [
            "\"accessors\"",
            "\"bufferViews\"",
            "\"buffers\"",
            "\"nodes\"",
        ] {
            assert!(
                !text.contains(forbidden),
                "an empty model must omit {forbidden} entirely:\n{text}"
            );
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    // -----------------------------------------------------------------------
    // Metadata and layout
    // -----------------------------------------------------------------------

    #[test]
    fn metadata_reports_the_models_count_and_sh_degree() {
        let dir = temp_subdir("metadata");
        let n = 42usize;
        let sh_degree = 2u32;
        export_gltf(&make_model(n, sh_degree), &dir.join("model.gltf"))
            .expect("test: export should succeed");

        let text = read_text(&dir.join("model.gltf"));
        assert!(
            text.contains(&format!(r#""gaussianCount": {n}"#)),
            "gaussianCount must match the model size:\n{text}"
        );
        assert!(
            text.contains(&format!(r#""shDegree": {sh_degree}"#)),
            "shDegree must match the model:\n{text}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn bin_file_size_is_n_times_44_plus_the_sh_block() {
        let dir = temp_subdir("bin_size");
        let n = 20usize;
        let sh_degree = 3u32;
        let sh_channels = ((sh_degree + 1).pow(2) * 3) as usize;
        export_gltf(&make_model(n, sh_degree), &dir.join("model.gltf"))
            .expect("test: export should succeed");

        // positions(12) + rotations(16) + scales(12) + opacities(4) = 44 B
        let expected = n * 44 + n * sh_channels * 4;
        let actual = std::fs::metadata(dir.join("model.bin"))
            .expect("test: bin file should exist")
            .len();
        assert_eq!(actual as usize, expected, "bin size must be N*44 + N*C*4");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn position_accessor_carries_min_and_max() {
        let dir = temp_subdir("min_max");
        // make_model places Gaussian i at [i, i+0.1, i+0.2].
        export_gltf(&make_model(5, 1), &dir.join("model.gltf"))
            .expect("test: export should succeed");

        let text = read_text(&dir.join("model.gltf"));
        assert!(
            text.contains(r#""min": [0, 0.1, 0.2]"#),
            "POSITION accessor must carry the real min:\n{text}"
        );
        assert!(
            text.contains(r#""max": [4, 4.1, 4.2]"#),
            "POSITION accessor must carry the real max:\n{text}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn emitted_json_is_balanced() {
        for (count, sh_degree) in [(0usize, 1u32), (1, 0), (12, 3)] {
            let dir = temp_subdir(&format!("balanced_{count}_{sh_degree}"));
            export_gltf(&make_model(count, sh_degree), &dir.join("model.gltf"))
                .expect("test: export should succeed");

            let text = read_text(&dir.join("model.gltf"));
            assert_eq!(
                text.matches('{').count(),
                text.matches('}').count(),
                "unbalanced braces for count={count}:\n{text}"
            );
            assert_eq!(
                text.matches('[').count(),
                text.matches(']').count(),
                "unbalanced brackets for count={count}:\n{text}"
            );

            let _ = std::fs::remove_dir_all(&dir);
        }
    }
}
