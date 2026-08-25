//! JSON output format for scripting and automation.
//!
//! Provides structured JSON output mode that:
//! - Emits valid JSON only (no extraneous output)
//! - Includes version, command, status in all outputs
//! - Tracks artifacts (generated files) with paths and sizes
//! - Collects warnings and errors
//! - Supports optional metadata for command-specific data

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// JSON Output Structure
// ---------------------------------------------------------------------------

/// Top-level JSON output structure for all commands.
#[derive(Debug, Serialize, Deserialize)]
pub struct JsonOutput {
    /// OxiGAF version (from CARGO_PKG_VERSION).
    pub version: String,

    /// Command name that was executed.
    pub command: String,

    /// Execution status.
    pub status: Status,

    /// Command-specific result data (optional).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,

    /// Generated artifacts (files) with paths and sizes.
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub artifacts: Vec<Artifact>,

    /// Warning messages (non-fatal issues).
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub warnings: Vec<String>,

    /// Error messages (fatal issues).
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub errors: Vec<String>,

    /// Additional metadata (command-specific).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

/// Execution status.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Status {
    /// Command completed successfully.
    Success,
    /// Command failed with errors.
    Error,
    /// Command completed with warnings.
    Warning,
}

/// File artifact (generated output file).
#[derive(Debug, Serialize, Deserialize)]
pub struct Artifact {
    /// Artifact type (e.g., "ply", "checkpoint", "image").
    #[serde(rename = "type")]
    pub artifact_type: String,

    /// Path to the artifact file.
    pub path: PathBuf,

    /// File size in bytes (if available).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size_bytes: Option<u64>,
}

// ---------------------------------------------------------------------------
// Implementation
// ---------------------------------------------------------------------------

impl JsonOutput {
    /// Create a new JSON output with default success status.
    #[must_use]
    pub fn new(command: &str) -> Self {
        Self {
            version: env!("CARGO_PKG_VERSION").to_string(),
            command: command.to_string(),
            status: Status::Success,
            result: None,
            artifacts: Vec::new(),
            warnings: Vec::new(),
            errors: Vec::new(),
            metadata: None,
        }
    }

    /// Create a success JSON output with result data.
    #[must_use]
    pub fn success(command: &str, result: serde_json::Value) -> Self {
        let mut output = Self::new(command);
        output.result = Some(result);
        output
    }

    /// Create an error JSON output with error message.
    #[must_use]
    pub fn error(command: &str, error: String) -> Self {
        let mut output = Self::new(command);
        output.status = Status::Error;
        output.errors.push(error);
        output
    }

    /// Add an artifact (generated file) to the output.
    ///
    /// Automatically reads file size if the file exists.
    pub fn add_artifact(&mut self, artifact_type: String, path: PathBuf) {
        let size_bytes = std::fs::metadata(&path).ok().map(|m| m.len());
        self.artifacts.push(Artifact {
            artifact_type,
            path,
            size_bytes,
        });
    }

    /// Add a warning message.
    ///
    /// This is the machine-readable half of
    /// [`crate::commands::flag_warnings`]: a handler that emits warnings on
    /// stderr should attach the same strings here so a `--json` consumer sees
    /// them without having to scrape stderr.
    ///
    /// A warning never downgrades an already-failed document: once
    /// [`Self::add_error`] has set [`Status::Error`], the status stays
    /// `Error` and the warning is recorded alongside it.
    pub fn add_warning(&mut self, warning: String) {
        self.warnings.push(warning);
        if self.errors.is_empty() {
            self.status = Status::Warning;
        }
    }

    /// Add an error message.
    ///
    /// Sets the document status to [`Status::Error`] unconditionally — an
    /// error outranks any warning already recorded.
    pub fn add_error(&mut self, error: String) {
        self.errors.push(error);
        self.status = Status::Error;
    }

    /// Print the JSON output to stdout.
    ///
    /// Uses pretty-printing for human readability.
    /// On serialization error, prints error JSON to stderr.
    pub fn print(&self) {
        match serde_json::to_string_pretty(self) {
            Ok(json) => {
                println!("{}", json);
            }
            Err(e) => {
                // Fallback error output to stderr
                eprintln!(
                    "{{\"error\": \"Failed to serialize JSON: {}\"}}",
                    e.to_string().replace('"', "\\\"")
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_json_output_new() {
        let output = JsonOutput::new("test");
        assert_eq!(output.command, "test");
        assert_eq!(output.version, env!("CARGO_PKG_VERSION"));
        assert!(matches!(output.status, Status::Success));
        assert!(output.result.is_none());
        assert!(output.artifacts.is_empty());
        assert!(output.warnings.is_empty());
        assert!(output.errors.is_empty());
    }

    #[test]
    fn test_json_output_success() {
        let result = serde_json::json!({"iterations": 1000});
        let output = JsonOutput::success("train", result.clone());
        assert_eq!(output.command, "train");
        assert!(matches!(output.status, Status::Success));
        assert_eq!(output.result, Some(result));
    }

    #[test]
    fn test_json_output_error() {
        let output = JsonOutput::error("export", "File not found".to_string());
        assert_eq!(output.command, "export");
        assert!(matches!(output.status, Status::Error));
        assert_eq!(output.errors.len(), 1);
        assert_eq!(output.errors[0], "File not found");
    }

    #[test]
    fn test_add_warning() {
        let mut output = JsonOutput::new("test");
        output.add_warning("Low memory".to_string());
        assert!(matches!(output.status, Status::Warning));
        assert_eq!(output.warnings.len(), 1);
    }

    #[test]
    fn test_add_error() {
        let mut output = JsonOutput::new("test");
        output.add_error("Fatal error".to_string());
        assert!(matches!(output.status, Status::Error));
        assert_eq!(output.errors.len(), 1);
    }

    /// Regression: `add_warning` guarded its status update with an empty
    /// `if !errors.is_empty() { /* keep error status */ } else { .. }`. The
    /// intent — a warning must never downgrade a failed document back to
    /// `Warning` — was only expressed by that comment and was untested.
    #[test]
    fn test_warning_does_not_downgrade_error_status() {
        let mut output = JsonOutput::new("test");
        output.add_error("Fatal error".to_string());
        output.add_warning("Low memory".to_string());

        assert!(
            matches!(output.status, Status::Error),
            "a warning must not downgrade an already-failed document"
        );
        assert_eq!(output.errors.len(), 1);
        assert_eq!(output.warnings.len(), 1, "the warning is still recorded");
    }

    /// The reverse order: an error raised after a warning wins.
    #[test]
    fn test_error_overrides_warning_status() {
        let mut output = JsonOutput::new("test");
        output.add_warning("Low memory".to_string());
        assert!(matches!(output.status, Status::Warning));

        output.add_error("Fatal error".to_string());
        assert!(matches!(output.status, Status::Error));
    }

    #[test]
    #[allow(clippy::expect_used)]
    fn test_serialization() {
        let output = JsonOutput::success(
            "test",
            serde_json::json!({
                "key": "value"
            }),
        );

        let json = serde_json::to_string(&output).expect("Serialization should succeed");
        assert!(json.contains("\"command\":\"test\""));
        assert!(json.contains("\"status\":\"success\""));
    }

    #[test]
    #[allow(clippy::expect_used)]
    fn test_skip_serializing_empty_fields() {
        let output = JsonOutput::new("test");
        let json = serde_json::to_string(&output).expect("Serialization should succeed");

        // Empty arrays should be omitted
        assert!(!json.contains("\"artifacts\""));
        assert!(!json.contains("\"warnings\""));
        assert!(!json.contains("\"errors\""));
        // None fields should be omitted
        assert!(!json.contains("\"result\""));
        assert!(!json.contains("\"metadata\""));
    }
}
