//! Honest reporting for pipeline flags whose behaviour is not implemented yet.
//!
//! `train`, `render`, `export` and `setup` each accept a handful of flags the
//! code behind them cannot honour: `--seed` cannot reach the trainer's RNG
//! until [`crate::pipeline::PipelineConfig`] carries it, `--background` and
//! `--splat-radius` cannot reach a software rasteriser whose clear colour and
//! splat footprint are compile-time constants, `--only` and `--skip-checksum`
//! cannot reach an asset downloader that takes neither, and so on. Accepting
//! such a flag in silence is the worst of the three options — the run looks
//! like it did what was asked and did not — so every one of them produces a
//! message here.
//!
//! # Why the messages are pure functions
//!
//! The wording is the whole feature: it is what tells a user their `--seed 7`
//! run is *not* reproducible. Building each string in a pure function keeps it
//! under test (a flag that stops warning, or starts warning at its default
//! value, fails a test here) and keeps the four `cmd_*` bodies in `main.rs`
//! short.
//!
//! # Why both channels
//!
//! [`emit`] writes each message twice on purpose:
//!
//! * `tracing::warn!` puts it in `--log-file`, where a CI run looks for it —
//!   but it is filtered out entirely by `-q` ([`crate::verbosity::Verbosity`]
//!   maps Quiet to `ERROR`) and, under `--json` with no `--log-file`, no
//!   subscriber is installed at all.
//! * [`crate::output::warning`] writes to **stderr**, so it survives both
//!   cases without putting a single byte on stdout — which under `--json`
//!   belongs exclusively to the result document.
//!
//! Handlers that build a JSON document should additionally attach the same
//! strings to it via [`attach`], so machine consumers see them too instead
//! of having to scrape stderr.

use crate::cli::{ExportArgs, ExportFormat, PlyFormat, RenderArgs, TrainArgs};
use crate::json_output::JsonOutput;

/// Report `warnings` on the log sink and on stderr.
///
/// See the module docs for why both channels are used.
pub fn emit(warnings: &[String]) {
    for warning in warnings {
        tracing::warn!("{warning}");
        crate::output::warning(warning);
    }
}

/// Attach `warnings` to a `--json` result document.
///
/// The single call site every `--json` branch of `train`/`render`/`export`
/// routes through, dry-run and real run alike, so a machine consumer of the
/// document sees exactly the same messages [`emit`] put on stderr for a
/// human. See the module docs for why the messages exist in the first
/// place, and [`JsonOutput::add_warning`] for the status-precedence rule
/// this feeds into.
pub fn attach(output: &mut JsonOutput, warnings: &[String]) {
    for warning in warnings {
        output.add_warning(warning.clone());
    }
}

/// Flags `oxigaf train` accepts but the reconstruction pipeline ignores.
#[must_use]
pub fn train(args: &TrainArgs) -> Vec<String> {
    let mut warnings = Vec::new();
    if args.seed.is_some() {
        warnings.push(
            "--seed is accepted but not yet threaded into Gaussian initialisation, view \
             sampling, or diffusion noise; this run is NOT reproducible."
                .to_string(),
        );
    }
    if args.eval_interval.is_some() {
        warnings.push(
            "--eval-interval is accepted but the pipeline has no periodic validation pass \
             yet; the value is ignored."
                .to_string(),
        );
    }
    if args.early_stop_loss.is_some() {
        warnings.push(
            "--early-stop-loss is accepted but only patience-based early stopping \
             (--patience/--min-delta) is implemented; the threshold is ignored."
                .to_string(),
        );
    }
    warnings
}

/// Flags `oxigaf render` accepts but the software point-cloud renderer ignores.
///
/// Both `--background` and `--splat-radius` are compared against their clap
/// defaults, so a user who never passes them is never warned.
#[must_use]
pub fn render(args: &RenderArgs) -> Vec<String> {
    let mut warnings = Vec::new();
    if args.background != DEFAULT_BACKGROUND {
        warnings.push(format!(
            "--background {} is accepted but the software point-cloud renderer has a fixed \
             clear colour; the rendered background is unchanged.",
            args.background,
        ));
    }
    if args.splat_radius != DEFAULT_SPLAT_RADIUS {
        warnings.push(format!(
            "--splat-radius {} is accepted but the software point-cloud renderer uses a fixed \
             splat footprint; the value is ignored.",
            args.splat_radius,
        ));
    }
    if let Some(ref flame_params_path) = args.flame_params {
        warnings.push(format!(
            "FLAME-driven animation (--flame-params {}) is not yet supported in the \
             software renderer; static Gaussian positions are rendered instead.",
            flame_params_path.display(),
        ));
    }
    warnings
}

/// The `--background` default declared in [`crate::cli::RenderArgs`].
const DEFAULT_BACKGROUND: &str = "1e1e1e";

/// The `--splat-radius` default declared in [`crate::cli::RenderArgs`].
const DEFAULT_SPLAT_RADIUS: u32 = 2;

/// Flags `oxigaf export` accepts but no writer consumes.
///
/// `--include-metadata` is the subtle one: the glTF and JSON writers embed
/// their metadata block *unconditionally* (generator string, format version,
/// export timestamp) and the remaining writers have nowhere to put one, so the
/// flag selects nothing in either case. Warning only for the formats without a
/// block — as this used to — implied it did something for the other two.
#[must_use]
pub fn export(args: &ExportArgs) -> Vec<String> {
    let mut warnings = Vec::new();
    if args.include_metadata {
        match args.format {
            ExportFormat::Gltf | ExportFormat::Json => {
                warnings.push(format!(
                    "--include-metadata is redundant for {:?}: that writer always embeds its \
                     metadata block, with or without the flag.",
                    args.format,
                ));
            }
            // `All` writes all four sibling formats at once, and only two of
            // them carry a metadata block -- neither "redundant" (true for
            // just the glTF/JSON pair) nor the general "no effect" message
            // (which would wrongly imply *no* component embeds one) is
            // accurate on its own.
            ExportFormat::All => {
                warnings.push(
                    "--include-metadata has no effect for All: its glTF and JSON components \
                     always embed their metadata block regardless of the flag, and its PLY and \
                     safetensors components never carry one."
                        .to_string(),
                );
            }
            _ => {
                warnings.push(format!(
                    "--include-metadata has no effect for {:?}: only the glTF and JSON writers \
                     carry a metadata block.",
                    args.format,
                ));
            }
        }
    }
    if let Some(ref checkpoint) = args.checkpoint {
        warnings.push(format!(
            "--checkpoint {} is accepted but no exporter reads training metadata from a \
             separate checkpoint yet; the file is not consulted.",
            checkpoint.display(),
        ));
    }
    if !matches!(args.format, ExportFormat::Ply) && !matches!(args.ply_format, PlyFormat::Ascii) {
        warnings.push(format!(
            "--ply-format only applies to --format ply; it is ignored for {:?}.",
            args.format,
        ));
    }
    warnings
}

/// Flags `oxigaf setup` accepts.
///
/// Currently unreachable from `cmd_setup`'s downloading path: both flags
/// found a real implementation to thread into
/// (`crate::assets::setup_cache_with_options`, which resolves `--only`
/// through the same [`crate::commands::runtime::select_assets`] the
/// `--offline`/`--dry-run` paths use and always honours `--skip-checksum`
/// as given), so there is nothing left for either flag to be warned about.
/// Kept, rather than deleted, as the honest-reporting fallback for any
/// future flag this crate accepts on `setup` but a downloader still can't
/// honour.
///
/// Takes the two values rather than `&SetupArgs` because by the time
/// `cmd_setup` reaches the downloading path it has already moved
/// `--from-hub` out of the struct, which rules out a whole-struct borrow.
#[must_use]
pub fn setup(skip_checksum: bool, only: Option<&str>) -> Vec<String> {
    let mut warnings = Vec::new();
    if skip_checksum {
        warnings.push(
            "--skip-checksum is accepted but `assets::setup_cache` always verifies downloads; \
             verification was NOT skipped."
                .to_string(),
        );
    }
    if only.is_some() {
        warnings.push(
            "--only is accepted but `assets::setup_cache` fetches the full manifest; \
             the filter was NOT applied."
                .to_string(),
        );
    }
    warnings
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::{Cli, Command, SetupArgs};
    use clap::Parser;

    fn parse(argv: &[&str]) -> Command {
        let cli = Cli::try_parse_from(argv.iter().copied()).expect("command line parses");
        cli.command
    }

    fn train_args(extra: &[&str]) -> TrainArgs {
        let mut argv: Vec<&str> = vec![
            "oxigaf",
            "train",
            "-i",
            "in.mp4",
            "-o",
            "out",
            "--flame-model",
            "flame",
        ];
        argv.extend_from_slice(extra);
        match parse(&argv) {
            Command::Train(args) => Some(args),
            _ => None,
        }
        .expect("train subcommand")
    }

    fn render_args(extra: &[&str]) -> RenderArgs {
        let mut argv: Vec<&str> = vec!["oxigaf", "render", "-m", "model.ply", "-o", "out"];
        argv.extend_from_slice(extra);
        match parse(&argv) {
            Command::Render(args) => Some(args),
            _ => None,
        }
        .expect("render subcommand")
    }

    fn export_args(extra: &[&str]) -> ExportArgs {
        let mut argv: Vec<&str> = vec!["oxigaf", "export", "-m", "model.ply", "-o", "out.ply"];
        argv.extend_from_slice(extra);
        match parse(&argv) {
            Command::Export(args) => Some(args),
            _ => None,
        }
        .expect("export subcommand")
    }

    fn setup_args(extra: &[&str]) -> SetupArgs {
        let mut argv: Vec<&str> = vec!["oxigaf", "setup"];
        argv.extend_from_slice(extra);
        match parse(&argv) {
            Command::Setup(args) => Some(args),
            _ => None,
        }
        .expect("setup subcommand")
    }

    /// Apply [`setup`] to a parsed `SetupArgs`, mirroring `cmd_setup`.
    fn setup_warnings(extra: &[&str]) -> Vec<String> {
        let args = setup_args(extra);
        setup(args.skip_checksum, args.only.as_deref())
    }

    #[test]
    fn default_invocations_warn_about_nothing() {
        assert!(train(&train_args(&[])).is_empty());
        assert!(render(&render_args(&[])).is_empty());
        assert!(export(&export_args(&[])).is_empty());
        assert!(setup_warnings(&[]).is_empty());
    }

    /// The defaults this module compares against must stay in step with the
    /// ones `cli.rs` declares, or every default render would warn.
    #[test]
    fn render_defaults_match_the_parser() {
        let args = render_args(&[]);
        assert_eq!(args.background, DEFAULT_BACKGROUND);
        assert_eq!(args.splat_radius, DEFAULT_SPLAT_RADIUS);
    }

    #[test]
    fn seed_is_reported_as_not_reproducible() {
        let warnings = train(&train_args(&["--seed", "7"]));
        assert_eq!(warnings.len(), 1);
        assert!(
            warnings[0].contains("NOT reproducible"),
            "message was: {}",
            warnings[0]
        );
    }

    #[test]
    fn train_reports_each_unwired_flag_once() {
        let warnings = train(&train_args(&[
            "--seed",
            "7",
            "--eval-interval",
            "100",
            "--early-stop-loss",
            "0.01",
        ]));
        assert_eq!(warnings.len(), 3);
    }

    #[test]
    fn render_reports_background_and_splat_radius_only_when_overridden() {
        assert_eq!(render(&render_args(&["--background", "ffffff"])).len(), 1);
        assert_eq!(render(&render_args(&["--splat-radius", "4"])).len(), 1);
        assert_eq!(
            render(&render_args(&[
                "--background",
                "1e1e1e",
                "--splat-radius",
                "2"
            ]))
            .len(),
            0,
            "explicitly passing the default values must not warn"
        );
    }

    #[test]
    fn render_reports_flame_params() {
        let warnings = render(&render_args(&["--flame-params", "params.json"]));
        assert_eq!(warnings.len(), 1);
        assert!(
            warnings[0].contains("params.json"),
            "message was: {}",
            warnings[0]
        );
    }

    /// Regression: `--include-metadata` used to warn only for the formats
    /// *without* a metadata block, which read as "it works for glTF/JSON".
    /// It works for neither — both embed their block unconditionally.
    ///
    /// `all` is included in the sweep deliberately: the name and doc comment
    /// both say "every format", so a format added later that this loop
    /// forgets is exactly the regression the loop exists to catch.
    #[test]
    fn include_metadata_warns_for_every_format() {
        for format in [
            "ply",
            "gltf",
            "json",
            "safetensors",
            "point-cloud",
            "mesh",
            "all",
        ] {
            let warnings = export(&export_args(&["--format", format, "--include-metadata"]));
            assert!(
                warnings.iter().any(|w| w.contains("--include-metadata")),
                "no --include-metadata warning for --format {format}"
            );
        }
        let redundant = export(&export_args(&["--format", "gltf", "--include-metadata"]));
        assert!(
            redundant.iter().any(|w| w.contains("redundant")),
            "glTF should be reported as redundant, got: {redundant:?}"
        );
    }

    /// `all` writes both formats that always embed metadata (glTF, JSON) and
    /// formats that never do (PLY, safetensors); the message must say so
    /// instead of reusing either single-format template verbatim.
    #[test]
    fn include_metadata_for_all_names_the_mixed_components() {
        let warnings = export(&export_args(&["--format", "all", "--include-metadata"]));
        assert_eq!(warnings.len(), 1);
        assert!(
            warnings[0].contains("glTF") && warnings[0].contains("JSON"),
            "message should name the components that always embed metadata: {}",
            warnings[0]
        );
        assert!(
            warnings[0].contains("PLY") && warnings[0].contains("safetensors"),
            "message should name the components that never do: {}",
            warnings[0]
        );
    }

    #[test]
    fn ply_format_warns_only_for_other_formats() {
        assert!(export(&export_args(&["--ply-format", "binary-le"])).is_empty());
        for format in ["gltf", "all"] {
            let warnings = export(&export_args(&[
                "--format",
                format,
                "--ply-format",
                "binary-le",
            ]));
            assert_eq!(warnings.len(), 1, "--format {format}");
            assert!(
                warnings[0].contains("--ply-format"),
                "message was: {}",
                warnings[0]
            );
        }
    }

    #[test]
    fn export_checkpoint_is_reported_as_unread() {
        let warnings = export(&export_args(&["--checkpoint", "ckpt.json"]));
        assert_eq!(warnings.len(), 1);
        assert!(
            warnings[0].contains("ckpt.json"),
            "message was: {}",
            warnings[0]
        );
    }

    #[test]
    fn setup_reports_skip_checksum_and_only() {
        assert_eq!(setup_warnings(&["--skip-checksum"]).len(), 1);
        assert_eq!(setup_warnings(&["--only", "flame"]).len(), 1);
        assert_eq!(
            setup_warnings(&["--skip-checksum", "--only", "flame"]).len(),
            2
        );
    }

    /// Regression for the module docs' promise that a `--json` handler
    /// attaches these strings to its result document (via [`attach`], the
    /// one call site every `--json` branch of `cmd_train`/`cmd_render`/
    /// `cmd_export` routes through) rather than leaving them only on
    /// stderr, where a machine consumer would have to scrape a log file to
    /// learn a flag was silently ignored.
    #[test]
    fn attach_puts_warning_text_in_the_json_document() {
        let warnings = render(&render_args(&["--background", "ffffff"]));
        assert_eq!(warnings.len(), 1);

        let mut output = JsonOutput::success("render", serde_json::json!({}));
        attach(&mut output, &warnings);

        assert_eq!(output.warnings, warnings);
        assert!(matches!(output.status, crate::json_output::Status::Warning));

        let json = serde_json::to_string(&output).expect("JsonOutput serializes");
        assert!(
            json.contains("ffffff"),
            "serialized --json document is missing the flag warning: {json}"
        );
    }
}
