//! Subcommand handler glue.
//!
//! Most modules under `commands/` own exactly one subcommand *family*:
//! the clap `Args`/`Subcommand` definitions plus the `run` handler that
//! adapts the family's arguments onto the public API of a library module
//! (`crate::animation_export`, `crate::dataset_tools`, …).
//!
//! The rest are the shared plumbing those handlers — and `main.rs` — run on:
//! [`runtime`] (device, cache and logging setup), [`model_io`] and
//! [`image_io`] (the array/pixel conversions the tool modules take),
//! [`flag_warnings`] (honest reporting for flags nothing consumes yet),
//! [`final_export`] (the format a training run leaves its model in),
//! [`gpu_probe`] (inspecting the adapter `--device <index>` selects) and
//! [`error_report`] (turning a failure into an exit code and a message).
//! They live here rather than at the crate root because they exist to serve
//! the dispatch layer and nothing else imports them.
//!
//! # Why the clap types live here and not in [`crate::cli`]
//!
//! `cli.rs` holds only the top-level [`crate::cli::Cli`] parser and the
//! [`crate::cli::Command`] enum. Forty-one tool modules' worth of argument
//! structs would push that single file far past the 2000-line ceiling, so
//! each family carries its own args next to the handler that consumes them
//! and `cli.rs` refers to them by path. Adding a family is therefore:
//!
//! 1. add `commands/<family>.rs` with `pub struct <Family>Args` (or a
//!    `#[derive(Subcommand)] enum`) and `pub fn run(args, ctx) -> Result<()>`,
//! 2. declare it in this file,
//! 3. add one variant to [`crate::cli::Command`],
//! 4. add one match arm in `main.rs`.
//!
//! # Output contract
//!
//! Handlers must print nothing on stdout other than the JSON document when
//! [`CmdContext::json`] is set — see [`emit`]. Human-readable rendering is
//! delegated to the library module's own `format_*` helpers wherever one
//! exists, so the CLI never re-implements formatting that already has tests.

pub mod analyze;
pub mod anim;
pub mod batch;
pub mod camera;
pub mod dataset;
pub mod error_report;
pub mod final_export;
pub mod flag_warnings;
pub mod gpu_probe;
pub mod image_io;
pub mod inspect;
pub mod model_io;
pub mod monitor;
pub mod perf;
pub mod pipeline_cmd;
pub mod preset;
pub mod preview;
pub mod profile;
pub mod quality;
pub mod report;
pub mod runs;
pub mod runtime;
pub mod scene;
pub mod scene_ops;
pub mod scene_reduce;
pub mod scene_tools;
pub mod sweep;
pub mod training;
pub mod video;
pub mod workspace;

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result};

use crate::json_output::JsonOutput;
use crate::verbosity::Verbosity;

/// Set once something has written the result document to stdout under
/// `--json`.
///
/// The `--json` contract is that stdout carries **exactly one** JSON value.
/// A handler that prints its document and *then* fails — `quality check` on
/// a render below its thresholds, `doctor` with a dead GPU, `cache verify`
/// with a missing asset — would otherwise have the top-level error arm in
/// `main.rs` append a second document, and `… | jq` would choke on two
/// concatenated values with no way to tell that from a broken tool.
static JSON_DOCUMENT_EMITTED: AtomicBool = AtomicBool::new(false);

/// Whether the single `--json` result document has already been written.
///
/// `main.rs` consults this before rendering a failure as JSON: when it is
/// set, the error goes to stderr instead and only the exit status carries
/// the failure.
#[must_use]
pub fn json_document_emitted() -> bool {
    JSON_DOCUMENT_EMITTED.load(Ordering::SeqCst)
}

/// Record that the `--json` result document has been written to stdout.
///
/// [`emit`] calls this for every handler that goes through it; the few
/// places that build a [`JsonOutput`] by hand must call it themselves.
pub fn mark_json_document_emitted() {
    JSON_DOCUMENT_EMITTED.store(true, Ordering::SeqCst);
}

/// Execution context shared by every subcommand handler.
///
/// Carries the three global flags (`-v/-q`, `--json`, `--dry-run`) so a
/// handler never has to reach back into the parsed [`crate::cli::Cli`].
#[derive(Debug, Clone, Copy)]
pub struct CmdContext {
    /// Effective verbosity from `-v`/`-q`.
    pub verbosity: Verbosity,
    /// `--json`: emit exactly one JSON document on stdout and nothing else.
    pub json: bool,
    /// `--dry-run`: validate and report, but do not write anything.
    pub dry_run: bool,
}

impl CmdContext {
    /// Build a context from the global flags.
    #[must_use]
    pub fn new(verbosity: Verbosity, json: bool, dry_run: bool) -> Self {
        Self {
            verbosity,
            json,
            dry_run,
        }
    }

    /// `true` when human-readable output should be written to stdout.
    #[must_use]
    pub fn human(&self) -> bool {
        !self.json
    }
}

/// Emit the result of a command: a JSON document under `--json`, otherwise
/// whatever `human` prints.
///
/// `artifacts` lists `(kind, path)` pairs for files the command produced;
/// they are attached to the JSON document (with sizes) and ignored in
/// human mode, where the handler is expected to mention them itself.
pub fn emit<F: FnOnce()>(
    ctx: &CmdContext,
    command: &str,
    result: serde_json::Value,
    artifacts: &[(&str, &Path)],
    human: F,
) {
    if ctx.json {
        let mut out = JsonOutput::success(command, result);
        for (kind, path) in artifacts {
            out.add_artifact((*kind).to_string(), path.to_path_buf());
        }
        out.print();
        mark_json_document_emitted();
    } else {
        human();
    }
}

/// Parse a comma-separated `x,y,z` triple for a clap `value_parser`.
///
/// # Errors
///
/// Returns a human-readable message when the input is not exactly three
/// comma-separated finite numbers.
pub fn parse_vec3(raw: &str) -> std::result::Result<[f32; 3], String> {
    let parts: Vec<&str> = raw.split(',').map(str::trim).collect();
    if parts.len() != 3 {
        return Err(format!(
            "expected three comma-separated numbers (x,y,z), got {} component(s) in {raw:?}",
            parts.len()
        ));
    }
    let mut out = [0.0f32; 3];
    for (slot, text) in out.iter_mut().zip(parts.iter()) {
        let value: f32 = text
            .parse()
            .map_err(|_| format!("{text:?} is not a valid number"))?;
        if !value.is_finite() {
            return Err(format!("{text:?} is not finite"));
        }
        *slot = value;
    }
    Ok(out)
}

/// Guard an output path against accidental overwrite and `--dry-run`.
///
/// Returns `Ok(false)` when the command must stop before writing (dry-run);
/// `Ok(true)` when writing may proceed.
///
/// # Errors
///
/// Returns an error when the file exists and `force` is not set, or when
/// the parent directory cannot be created.
pub fn prepare_output(ctx: &CmdContext, path: &Path, force: bool) -> Result<bool> {
    if path.exists() && !force {
        anyhow::bail!(
            "Output file already exists: {}. Use --force to overwrite.",
            path.display()
        );
    }
    if ctx.dry_run {
        return Ok(false);
    }
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).with_context(|| {
                format!("Failed to create output directory: {}", parent.display())
            })?;
        }
    }
    Ok(true)
}

/// Load a Gaussian model and return its flat `xyz` position array.
///
/// Used by every geometry-oriented tool command; the library modules all
/// take positions as a flat `[x0, y0, z0, x1, …]` slice.
///
/// # Errors
///
/// Propagates model-loading failures from [`crate::export::load_model`].
pub fn load_positions(path: &Path) -> Result<Vec<f32>> {
    let model = crate::export::load_model(path)
        .with_context(|| format!("Failed to load model: {}", path.display()))?;
    let mut positions = Vec::with_capacity(model.gaussians.len() * 3);
    for gaussian in &model.gaussians {
        positions.extend_from_slice(&gaussian.position);
    }
    Ok(positions)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression: `commands/scene_ops.rs` sat on disk — wiring
    /// `scene_analyzer`, `scene_merge`, `scene_optimizer` and
    /// `scene_streaming` onto `oxigaf scene` — while this file's `pub mod`
    /// list did not name it. Rust does not compile an undeclared file, so it
    /// was not merely unreachable: nothing in it was ever type-checked, and
    /// it had silently rotted into code that could not build.
    ///
    /// Every `.rs` file in this directory must therefore be declared here.
    /// The check reads the directory rather than comparing against a list
    /// that would itself have to be maintained.
    #[test]
    fn every_handler_file_is_declared_as_a_module() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/commands");
        let source =
            std::fs::read_to_string(dir.join("mod.rs")).expect("commands/mod.rs is readable");

        let entries = std::fs::read_dir(&dir).expect("commands/ is readable");
        let mut checked = 0usize;
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("rs") {
                continue;
            }
            let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            if stem == "mod" {
                continue;
            }
            assert!(
                source.contains(&format!("pub mod {stem};")),
                "commands/{stem}.rs exists but `pub mod {stem};` is missing from \
                 commands/mod.rs — the file is not compiled at all"
            );
            checked += 1;
        }
        assert!(
            checked >= 20,
            "only {checked} handler file(s) found; the directory scan looked in the wrong place"
        );
    }

    #[test]
    fn parse_vec3_accepts_spaces_and_signs() {
        assert_eq!(parse_vec3("1, -2 ,3.5"), Ok([1.0, -2.0, 3.5]));
    }

    #[test]
    fn parse_vec3_rejects_wrong_arity() {
        assert!(parse_vec3("1,2").is_err());
        assert!(parse_vec3("1,2,3,4").is_err());
    }

    #[test]
    fn parse_vec3_rejects_non_finite() {
        assert!(parse_vec3("1,2,nan").is_err());
        assert!(parse_vec3("1,2,inf").is_err());
    }

    #[test]
    fn prepare_output_reports_dry_run() {
        let ctx = CmdContext::new(Verbosity::Normal, false, true);
        let path = std::env::temp_dir().join("oxigaf_prepare_output_dry_run.json");
        let _ = std::fs::remove_file(&path);
        assert_eq!(prepare_output(&ctx, &path, false).ok(), Some(false));
        assert!(!path.exists());
    }

    #[test]
    fn prepare_output_refuses_existing_file_without_force() {
        let ctx = CmdContext::new(Verbosity::Normal, false, false);
        let path = std::env::temp_dir().join("oxigaf_prepare_output_existing.json");
        std::fs::write(&path, b"{}").expect("temp file write");
        assert!(prepare_output(&ctx, &path, false).is_err());
        assert!(prepare_output(&ctx, &path, true).is_ok());
        let _ = std::fs::remove_file(&path);
    }
}
