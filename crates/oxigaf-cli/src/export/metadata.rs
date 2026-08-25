//! Export metadata: model statistics plus training provenance read back from a
//! checkpoint.
//!
//! `oxigaf export --include-metadata` and `--checkpoint <file>` both used to be
//! accepted and then ignored: the glTF and JSON writers embedded a fixed
//! generator block whether or not the flag was passed, the PLY and safetensors
//! writers had nowhere to put one, and no writer ever opened the
//! `--checkpoint` file.  [`ExportMetadata`] is the block every writer can now
//! carry, and [`load_training_metadata`] is the reader that fills its
//! `training` half.
//!
//! # Checkpoint shapes
//!
//! Two JSON layouts are read, because two writers produce them:
//!
//! * `oxigaf_trainer::checkpoint::CheckpointData` — the file the trainer saves
//!   during a run.  It carries `iteration`, a numeric `version`, the model
//!   arrays, the optimiser moments, and a `metrics_history` array of
//!   `{iteration, psnr, ssim, loss}` entries.
//! * [`crate::export::JsonCheckpoint`] — what `oxigaf export --format json`
//!   writes.  It carries a string `version`, `num_gaussians`, `sh_degree`, and
//!   a free-form `metadata` map.
//!
//! Rather than deserialising into either struct (which would reject the other,
//! and would reject any future field), the reader walks a
//! [`serde_json::Value`] and takes what it recognises.  A checkpoint missing a
//! field yields `None` for it instead of failing the whole export — but a file
//! that is not JSON at all, or is not a JSON object, is a hard error rather
//! than a silently empty metadata block.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use oxigaf::render::gaussian::GaussianModel;

/// Statistics describing the model actually being written.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelStats {
    /// Number of Gaussians in the exported model.
    pub num_gaussians: usize,
    /// Spherical-harmonics degree of the exported model.
    pub sh_degree: u32,
    /// Gaussians flagged rigid (head-bone only).
    pub rigid_count: usize,
    /// Gaussians flagged flexible (expression / jaw driven).
    pub flexible_count: usize,
}

impl ModelStats {
    /// Summarise `model`.
    #[must_use]
    pub fn for_model(model: &GaussianModel) -> Self {
        Self {
            num_gaussians: model.len(),
            sh_degree: model.sh_degree,
            rigid_count: model.is_rigid.iter().filter(|&&r| r).count(),
            flexible_count: model.is_rigid.iter().filter(|&&r| !r).count(),
        }
    }
}

/// Training provenance recovered from a checkpoint file.
///
/// Every field is optional: the two checkpoint layouts overlap only partly,
/// and a field a given checkpoint does not carry is reported as absent rather
/// than invented.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct TrainingMetadata {
    /// File name of the checkpoint the values came from.
    ///
    /// Only the file name, never the full path — an exported artefact must not
    /// leak the directory layout of the machine that produced it.
    pub source: String,
    /// Checkpoint format version, as written (numeric or string).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub checkpoint_version: Option<String>,
    /// Training iteration the checkpoint was taken at.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub iteration: Option<u64>,
    /// Number of Gaussians recorded in the checkpoint.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_gaussians: Option<usize>,
    /// SH degree recorded in the checkpoint.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sh_degree: Option<u32>,
    /// PSNR of the last recorded metrics entry.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub final_psnr: Option<f32>,
    /// SSIM of the last recorded metrics entry.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub final_ssim: Option<f32>,
    /// Loss of the last recorded metrics entry.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub final_loss: Option<f32>,
    /// Number of entries in the checkpoint's metrics history.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metrics_entries: Option<usize>,
    /// Free-form string entries from a `metadata` object, if the checkpoint
    /// carries one.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extra: BTreeMap<String, String>,
}

/// A metadata block a writer can embed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExportMetadata {
    /// Version of the tool that wrote the file.
    pub version: String,
    /// Human-readable generator string.
    pub generator: String,
    /// RFC 3339 timestamp of the export.
    pub export_time: String,
    /// Statistics of the model written.
    pub model_stats: ModelStats,
    /// Training provenance, when a `--checkpoint` was supplied.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub training: Option<TrainingMetadata>,
}

impl ExportMetadata {
    /// Build a metadata block describing `model`, with no training provenance.
    #[must_use]
    pub fn for_model(model: &GaussianModel) -> Self {
        Self {
            version: env!("CARGO_PKG_VERSION").to_string(),
            generator: format!("OxiGAF v{}", env!("CARGO_PKG_VERSION")),
            export_time: chrono::Utc::now().to_rfc3339(),
            model_stats: ModelStats::for_model(model),
            training: None,
        }
    }

    /// Attach training provenance.
    #[must_use]
    pub fn with_training(mut self, training: TrainingMetadata) -> Self {
        self.training = Some(training);
        self
    }

    /// Flatten this block to `key = value` pairs, for the formats whose only
    /// metadata channel is a flat string map (safetensors) or a comment block
    /// (PLY).
    ///
    /// Keys are stable and namespaced so a consumer can tell an OxiGAF key
    /// from a tensor name; values never contain a newline or a carriage
    /// return, which would corrupt a PLY header.
    #[must_use]
    pub fn to_key_values(&self) -> Vec<(String, String)> {
        let mut pairs = vec![
            ("oxigaf_version".to_string(), self.version.clone()),
            ("generator".to_string(), self.generator.clone()),
            ("export_time".to_string(), self.export_time.clone()),
            (
                "num_gaussians".to_string(),
                self.model_stats.num_gaussians.to_string(),
            ),
            (
                "sh_degree".to_string(),
                self.model_stats.sh_degree.to_string(),
            ),
            (
                "rigid_count".to_string(),
                self.model_stats.rigid_count.to_string(),
            ),
            (
                "flexible_count".to_string(),
                self.model_stats.flexible_count.to_string(),
            ),
        ];

        if let Some(ref training) = self.training {
            pairs.push(("training_source".to_string(), training.source.clone()));
            let mut push_opt = |key: &str, value: Option<String>| {
                if let Some(value) = value {
                    pairs.push((key.to_string(), value));
                }
            };
            push_opt(
                "training_checkpoint_version",
                training.checkpoint_version.clone(),
            );
            push_opt(
                "training_iteration",
                training.iteration.map(|v| v.to_string()),
            );
            push_opt(
                "training_num_gaussians",
                training.num_gaussians.map(|v| v.to_string()),
            );
            push_opt(
                "training_sh_degree",
                training.sh_degree.map(|v| v.to_string()),
            );
            push_opt("training_final_psnr", training.final_psnr.map(fmt_f32));
            push_opt("training_final_ssim", training.final_ssim.map(fmt_f32));
            push_opt("training_final_loss", training.final_loss.map(fmt_f32));
            push_opt(
                "training_metrics_entries",
                training.metrics_entries.map(|v| v.to_string()),
            );
            for (key, value) in &training.extra {
                pairs.push((format!("training_{key}"), value.clone()));
            }
        }

        for (_, value) in &mut pairs {
            sanitize_single_line(value);
        }
        pairs
    }
}

/// Format a metric without an exponent or a trailing pile of digits.
fn fmt_f32(value: f32) -> String {
    format!("{value:.6}")
}

/// Replace every character that would break a single-line container (PLY
/// comment, safetensors metadata value) with a space.
fn sanitize_single_line(value: &mut String) {
    if value
        .chars()
        .any(|c| c == '\n' || c == '\r' || c.is_control())
    {
        *value = value
            .chars()
            .map(|c| {
                if c == '\n' || c == '\r' || c.is_control() {
                    ' '
                } else {
                    c
                }
            })
            .collect();
    }
}

// ---------------------------------------------------------------------------
// Checkpoint reading
// ---------------------------------------------------------------------------

/// Read a number out of a JSON value, accepting both integer and string forms.
fn as_u64(value: Option<&serde_json::Value>) -> Option<u64> {
    match value? {
        serde_json::Value::Number(n) => n.as_u64(),
        serde_json::Value::String(s) => s.parse().ok(),
        _ => None,
    }
}

fn as_usize(value: Option<&serde_json::Value>) -> Option<usize> {
    usize::try_from(as_u64(value)?).ok()
}

fn as_u32(value: Option<&serde_json::Value>) -> Option<u32> {
    u32::try_from(as_u64(value)?).ok()
}

fn as_f32(value: Option<&serde_json::Value>) -> Option<f32> {
    match value? {
        serde_json::Value::Number(n) => n.as_f64().map(|v| v as f32),
        serde_json::Value::String(s) => s.parse().ok(),
        _ => None,
    }
}

/// A `version` field is numeric in the trainer's checkpoint and a string in the
/// CLI's JSON export; both are rendered as a string here.
fn version_string(value: Option<&serde_json::Value>) -> Option<String> {
    match value? {
        serde_json::Value::Number(n) => Some(n.to_string()),
        serde_json::Value::String(s) => Some(s.clone()),
        _ => None,
    }
}

/// Read training metadata out of a checkpoint JSON file.
///
/// Understands both the trainer's `CheckpointData` and the CLI's
/// [`crate::export::JsonCheckpoint`]; see the module docs.
///
/// # Errors
///
/// Returns an error when the file cannot be read, does not parse as JSON, or
/// is not a JSON object.  A *recognised* object missing individual fields is
/// not an error — the corresponding [`TrainingMetadata`] fields stay `None`.
pub fn load_training_metadata(path: &Path) -> Result<TrainingMetadata> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read checkpoint: {}", path.display()))?;
    let value: serde_json::Value = serde_json::from_str(&text)
        .with_context(|| format!("Failed to parse checkpoint as JSON: {}", path.display()))?;
    let object = value.as_object().with_context(|| {
        format!(
            "Checkpoint {} is not a JSON object; --checkpoint expects a .json checkpoint \
             written by `oxigaf train` or `oxigaf export --format json`",
            path.display()
        )
    })?;

    // Only the file name: an exported artefact must not carry the directory
    // layout of the machine that produced it.
    let source = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string());

    // `num_gaussians` is explicit in the CLI's JSON export; the trainer's
    // checkpoint only implies it through the length of its `positions` array.
    let num_gaussians = as_usize(object.get("num_gaussians")).or_else(|| {
        object
            .get("positions")
            .and_then(|v| v.as_array())
            .map(|a| a.len())
    });

    let history = object.get("metrics_history").and_then(|v| v.as_array());
    let last = history
        .and_then(|entries| entries.last())
        .and_then(|v| v.as_object());

    let extra = object
        .get("metadata")
        .and_then(|v| v.as_object())
        .map(|map| {
            map.iter()
                .map(|(k, v)| {
                    let rendered = match v {
                        serde_json::Value::String(s) => s.clone(),
                        other => other.to_string(),
                    };
                    (k.clone(), rendered)
                })
                .collect::<BTreeMap<_, _>>()
        })
        .unwrap_or_default();

    let metadata = TrainingMetadata {
        source,
        checkpoint_version: version_string(object.get("version")),
        iteration: as_u64(object.get("iteration")),
        num_gaussians,
        sh_degree: as_u32(object.get("sh_degree")),
        final_psnr: last.and_then(|e| as_f32(e.get("psnr"))),
        final_ssim: last.and_then(|e| as_f32(e.get("ssim"))),
        final_loss: last.and_then(|e| as_f32(e.get("loss"))),
        metrics_entries: history.map(|entries| entries.len()),
        extra,
    };

    tracing::info!(
        "Read training metadata from {} (iteration {:?}, {:?} Gaussians)",
        path.display(),
        metadata.iteration,
        metadata.num_gaussians,
    );
    Ok(metadata)
}

/// Decide what metadata block, if any, an export should embed.
///
/// * `--include-metadata` alone embeds the model statistics.
/// * `--checkpoint <file>` also embeds that checkpoint's training provenance,
///   whether or not `--include-metadata` was passed — asking for a specific
///   metadata source is asking for metadata.
/// * Neither flag yields `None`, and the writers fall back to the minimal
///   block they have always written.
///
/// # Errors
///
/// Propagates [`load_training_metadata`] — a `--checkpoint` that cannot be
/// read is a failed export, not a silently dropped block.
pub fn resolve_export_metadata(
    model: &GaussianModel,
    include_metadata: bool,
    checkpoint: Option<&Path>,
) -> Result<Option<ExportMetadata>> {
    match (include_metadata, checkpoint) {
        (false, None) => Ok(None),
        (_, Some(path)) => {
            let training = load_training_metadata(path)?;
            Ok(Some(
                ExportMetadata::for_model(model).with_training(training),
            ))
        }
        (true, None) => Ok(Some(ExportMetadata::for_model(model))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxigaf::render::gaussian::GaussianAttributes;

    fn model(n: usize, rigid: usize) -> GaussianModel {
        GaussianModel {
            gaussians: (0..n)
                .map(|_| GaussianAttributes {
                    position: [0.0; 3],
                    _pad0: 0.0,
                    rotation: [0.0, 0.0, 0.0, 1.0],
                    scale: [-1.0; 3],
                    opacity: 0.0,
                })
                .collect(),
            sh_coeffs: vec![0.0; n * 3],
            sh_degree: 0,
            face_indices: vec![0; n],
            barycentric: vec![[1.0 / 3.0; 3]; n],
            local_offsets: vec![[0.0; 3]; n],
            is_rigid: (0..n).map(|i| i < rigid).collect(),
        }
    }

    fn temp_json(name: &str, body: &str) -> std::path::PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "oxigaf_export_metadata_{name}_{}.json",
            std::process::id()
        ));
        std::fs::write(&path, body).expect("write temp checkpoint");
        path
    }

    #[test]
    fn trainer_checkpoint_metrics_history_is_read() {
        let path = temp_json(
            "trainer",
            r#"{"version":2,"iteration":7000,"sh_degree":3,
                "positions":[[0,0,0],[1,1,1]],
                "metrics_history":[{"iteration":100,"psnr":22.0,"ssim":0.7,"loss":0.05},
                                   {"iteration":7000,"psnr":31.5,"ssim":0.93,"loss":0.011}]}"#,
        );
        let meta = load_training_metadata(&path).expect("read ok");
        let _ = std::fs::remove_file(&path);

        assert_eq!(meta.iteration, Some(7000));
        assert_eq!(meta.sh_degree, Some(3));
        assert_eq!(meta.num_gaussians, Some(2), "inferred from `positions`");
        assert_eq!(meta.checkpoint_version.as_deref(), Some("2"));
        assert_eq!(meta.metrics_entries, Some(2));
        assert!(meta.final_psnr.map(|v| (v - 31.5).abs() < 1e-5) == Some(true));
        assert!(meta.final_loss.map(|v| (v - 0.011).abs() < 1e-6) == Some(true));
        assert!(
            !meta.source.contains(std::path::MAIN_SEPARATOR),
            "only the file name may be recorded, got {:?}",
            meta.source
        );
    }

    #[test]
    fn cli_json_export_checkpoint_is_read() {
        let path = temp_json(
            "cli",
            r#"{"version":"1.0","iteration":0,"num_gaussians":5,"sh_degree":1,
                "metadata":{"generator":"OxiGAF v0.1.2","export_time":"2026-08-25T00:00:00Z"}}"#,
        );
        let meta = load_training_metadata(&path).expect("read ok");
        let _ = std::fs::remove_file(&path);

        assert_eq!(meta.num_gaussians, Some(5));
        assert_eq!(meta.checkpoint_version.as_deref(), Some("1.0"));
        assert_eq!(
            meta.extra.get("generator").map(String::as_str),
            Some("OxiGAF v0.1.2")
        );
        assert_eq!(meta.metrics_entries, None);
    }

    #[test]
    fn a_non_object_or_unparsable_checkpoint_is_an_error() {
        let array = temp_json("array", "[1, 2, 3]");
        assert!(load_training_metadata(&array).is_err());
        let _ = std::fs::remove_file(&array);

        let junk = temp_json("junk", "not json at all");
        assert!(load_training_metadata(&junk).is_err());
        let _ = std::fs::remove_file(&junk);

        let missing = std::env::temp_dir().join("oxigaf_export_metadata_absent_file.json");
        let _ = std::fs::remove_file(&missing);
        assert!(load_training_metadata(&missing).is_err());
    }

    #[test]
    fn resolve_returns_none_only_when_neither_flag_is_given() {
        let model = model(4, 1);
        assert!(resolve_export_metadata(&model, false, None)
            .expect("resolve ok")
            .is_none());

        let plain = resolve_export_metadata(&model, true, None)
            .expect("resolve ok")
            .expect("metadata present");
        assert_eq!(plain.model_stats.num_gaussians, 4);
        assert_eq!(plain.model_stats.rigid_count, 1);
        assert_eq!(plain.model_stats.flexible_count, 3);
        assert!(plain.training.is_none());

        // `--checkpoint` alone must still produce a block: naming a metadata
        // source is asking for metadata.
        let path = temp_json("resolve", r#"{"iteration":42}"#);
        let with_training = resolve_export_metadata(&model, false, Some(&path))
            .expect("resolve ok")
            .expect("metadata present");
        let _ = std::fs::remove_file(&path);
        assert_eq!(
            with_training.training.and_then(|t| t.iteration),
            Some(42),
            "--checkpoint must actually be read"
        );
    }

    #[test]
    fn resolve_propagates_an_unreadable_checkpoint() {
        let model = model(1, 0);
        let missing = std::env::temp_dir().join("oxigaf_export_metadata_no_such.json");
        let _ = std::fs::remove_file(&missing);
        assert!(
            resolve_export_metadata(&model, true, Some(&missing)).is_err(),
            "a --checkpoint that cannot be read must fail the export, not be dropped"
        );
    }

    #[test]
    fn key_values_are_single_line() {
        // Regression: a metadata value carrying a newline would corrupt a PLY
        // header, where each comment is one line.
        let mut training = TrainingMetadata {
            source: "ckpt.json".to_string(),
            ..TrainingMetadata::default()
        };
        training
            .extra
            .insert("note".to_string(), "line one\nline two\r\tend".to_string());
        let metadata = ExportMetadata::for_model(&model(2, 2)).with_training(training);

        let pairs = metadata.to_key_values();
        for (key, value) in &pairs {
            assert!(
                !value.contains('\n') && !value.contains('\r') && !value.contains('\t'),
                "value for {key} is not single-line: {value:?}"
            );
        }
        assert!(pairs.iter().any(|(k, _)| k == "training_note"));
        assert!(pairs.iter().any(|(k, v)| k == "num_gaussians" && v == "2"));
    }
}
