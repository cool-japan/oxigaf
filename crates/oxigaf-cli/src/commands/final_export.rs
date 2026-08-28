//! The writer `oxigaf train` uses for its final model.
//!
//! `[output] export_format` in the project configuration (and its
//! `OXIGAF_OUTPUT_EXPORT_FORMAT` environment override) documents four
//! values — `ply`, `safetensors`, `gltf`, `json` — and for a long time
//! nothing read any of them: `cmd_train` hard-coded `final_model.ply`, so a
//! run configured for safetensors produced a PLY and the field was
//! decoration.
//!
//! [`FinalExport`] is the resolved form of that field. Parsing it *before*
//! the run starts means an unusable value fails in the first second rather
//! than after an hour of training with the trained model in memory and
//! nowhere to put it, and the file name then follows the format instead of
//! claiming `.ply` for safetensors bytes.
//!
//! This is not the same surface as `oxigaf export`: that command takes a
//! [`crate::cli::ExportFormat`] from the command line and can also write
//! point clouds and meshes. This is the narrower set a *training run* can be
//! configured to leave behind, which is why the two enumerations differ.

use std::path::{Path, PathBuf};

use anyhow::Result;

use oxigaf::render::gaussian::GaussianModel;

/// The format `train` writes its final model in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinalExport {
    /// Standard 3DGS ASCII PLY — the default and the one every viewer reads.
    Ply,
    /// Safetensors tensor bundle.
    Safetensors,
    /// glTF 2.0, written as a `.gltf` + `.bin` pair.
    Gltf,
    /// JSON checkpoint.
    Json,
}

impl FinalExport {
    /// Resolve a configured `export_format` string.
    ///
    /// Case and surrounding whitespace are ignored: the value arrives from
    /// hand-edited TOML and from environment variables, and neither is worth
    /// failing a run over.
    ///
    /// # Errors
    ///
    /// Returns an error naming the accepted values when `raw` is not one of
    /// them, rather than silently falling back to PLY.
    pub fn parse(raw: &str) -> Result<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "ply" => Ok(Self::Ply),
            "safetensors" => Ok(Self::Safetensors),
            "gltf" | "glb" => Ok(Self::Gltf),
            "json" => Ok(Self::Json),
            other => anyhow::bail!(
                "Unsupported [output] export_format {other:?}: expected one of \
                 ply, safetensors, gltf, json. Set it in the config TOML or via \
                 OXIGAF_OUTPUT_EXPORT_FORMAT."
            ),
        }
    }

    /// File name written inside the run's output directory.
    #[must_use]
    pub fn file_name(self) -> &'static str {
        match self {
            Self::Ply => "final_model.ply",
            Self::Safetensors => "final_model.safetensors",
            Self::Gltf => "final_model.gltf",
            Self::Json => "final_model.json",
        }
    }

    /// Short name used in log lines and in the `--json` document.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Ply => "ply",
            Self::Safetensors => "safetensors",
            Self::Gltf => "gltf",
            Self::Json => "json",
        }
    }

    /// The second file this format writes, if any.
    ///
    /// Only glTF has one: the exporter emits a `.bin` buffer next to the
    /// `.gltf` document, so a caller collecting artefacts — or a `--dry-run`
    /// listing what would be written — needs both.
    #[must_use]
    pub fn sidecar(self, path: &Path) -> Option<PathBuf> {
        match self {
            Self::Gltf => Some(path.with_extension("bin")),
            Self::Ply | Self::Safetensors | Self::Json => None,
        }
    }

    /// Write `model` to `path` with this format's writer.
    ///
    /// # Errors
    ///
    /// Propagates the writer's failure.
    pub fn write(self, model: &GaussianModel, path: &Path) -> Result<()> {
        match self {
            Self::Ply => crate::export::export_ply(model, path),
            Self::Safetensors => crate::export::export_safetensors(model, path),
            Self::Gltf => crate::export_gltf::export_gltf(model, path).map_err(anyhow::Error::from),
            Self::Json => crate::export::export_json_checkpoint(model, path),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression: `[output] export_format` (and its
    /// `OXIGAF_OUTPUT_EXPORT_FORMAT` override) was read by nothing — `train`
    /// always wrote `final_model.ply`, so a run configured for safetensors
    /// produced a PLY under a name that did not match its bytes.
    #[test]
    fn parse_follows_the_configured_format() {
        for (configured, expected_file, expected_label) in [
            ("ply", "final_model.ply", "ply"),
            ("safetensors", "final_model.safetensors", "safetensors"),
            ("gltf", "final_model.gltf", "gltf"),
            ("json", "final_model.json", "json"),
            // Case and stray whitespace come from hand-edited TOML and from
            // environment variables; neither should be a hard failure.
            ("  PLY  ", "final_model.ply", "ply"),
            ("GLB", "final_model.gltf", "gltf"),
        ] {
            let resolved = FinalExport::parse(configured);
            assert!(
                resolved.is_ok(),
                "export_format {configured:?} was rejected"
            );
            if let Ok(export) = resolved {
                assert_eq!(export.file_name(), expected_file);
                assert_eq!(export.label(), expected_label);
            }
        }
    }

    /// An unusable `export_format` must fail before the run, not after it:
    /// the alternative is discovering it an hour into training, with the
    /// trained model still in memory and nowhere to put it.
    #[test]
    fn parse_refuses_an_unknown_format() {
        for bad in ["obj", "", "pl y", "safetensor"] {
            let refused = FinalExport::parse(bad);
            assert!(refused.is_err(), "export_format {bad:?} should be refused");
            if let Err(e) = refused {
                let message = format!("{e}");
                assert!(
                    message.contains("safetensors"),
                    "the message must name the alternatives, was: {message}"
                );
            }
        }
    }

    /// Every file name has to carry the extension its writer produces, or a
    /// viewer picking a loader by extension reads the wrong bytes.
    #[test]
    fn file_name_extension_matches_the_format() {
        for (export, extension) in [
            (FinalExport::Ply, "ply"),
            (FinalExport::Safetensors, "safetensors"),
            (FinalExport::Gltf, "gltf"),
            (FinalExport::Json, "json"),
        ] {
            assert!(
                export.file_name().ends_with(&format!(".{extension}")),
                "{} does not end with .{extension}",
                export.file_name()
            );
        }
    }

    /// glTF is a two-file format; a caller collecting artefacts (or a dry
    /// run listing what would be written) needs the `.bin` buffer as well.
    #[test]
    fn only_gltf_reports_a_sidecar() {
        let path = std::env::temp_dir().join("final_model.gltf");
        assert_eq!(
            FinalExport::Gltf.sidecar(&path),
            Some(path.with_extension("bin"))
        );
        assert_eq!(FinalExport::Ply.sidecar(&path), None);
        assert_eq!(FinalExport::Safetensors.sidecar(&path), None);
        assert_eq!(FinalExport::Json.sidecar(&path), None);
    }
}
