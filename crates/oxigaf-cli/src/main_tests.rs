//! Test module for `main.rs`, split into its own file so `main.rs` itself
//! stays under the workspace's 2000-line file-size policy. Included via
//! `#[path = "main_tests.rs"] mod tests;` at the bottom of `main.rs` — every
//! item here runs against the binary crate root's private functions through
//! `use super::*;` exactly as it would if the module were still inline.
//!
//! Do **not** add a `[[bin]]`/`[lib]` entry or a `mod main_tests;` anywhere
//! else for this file: it is reachable only as `main.rs`'s `tests` child,
//! the same "one file, one place it's declared" rule `main.rs`'s own header
//! comment states for the library modules it re-exports.

use super::*;

/// clap's own consistency check over the whole command tree: duplicate
/// argument ids, conflicting short flags, a `#[command(flatten)]`ed
/// subcommand enum that collides with its host, and so on.
#[test]
fn cli_definition_is_internally_consistent() {
    Cli::command().debug_assert();
}

/// `Command::Scene` was switched from `commands::scene_tools::SceneArgs`
/// to `commands::scene_ops::SceneArgs`, which flattens the former and
/// adds five more subcommands. Both halves must remain reachable — a
/// flatten that silently dropped the nine geometry/reduction commands
/// would be a regression no type error catches.
#[test]
fn scene_family_exposes_both_tools_and_ops_subcommands() {
    let cli = Cli::command();
    let scene = cli.find_subcommand("scene");
    assert!(scene.is_some(), "`oxigaf scene` is not registered");
    if let Some(scene) = scene {
        for name in [
            // flattened in from `scene_tools`
            "register",
            "stats",
            "filter",
            "prune",
            "transform",
            "dedup",
            "compress",
            "lod",
            "convert",
            // added by `scene_ops`
            "analyze",
            "compare",
            "merge",
            "optimize",
            "stream",
        ] {
            assert!(
                scene.find_subcommand(name).is_some(),
                "`oxigaf scene {name}` is not registered"
            );
        }
    }
}

/// Every tool family declared in `cli::Command` must be reachable by
/// name, so a module wired into `commands/` but forgotten in the parser
/// fails the test rather than shipping unreachable.
#[test]
fn every_tool_family_is_reachable_by_name() {
    let cli = Cli::command();
    for name in [
        "anim",
        "analyze",
        "batch",
        "camera",
        "dataset",
        "inspect",
        "monitor",
        "perf",
        "pipeline",
        "preset",
        "preview",
        "profile",
        "quality",
        "report",
        "runs",
        "scene",
        "sweep",
        "training",
        "video",
        "workspace",
    ] {
        assert!(
            cli.find_subcommand(name).is_some(),
            "`oxigaf {name}` is not registered"
        );
    }
}

/// The families added in the third wiring stage must expose every
/// subcommand their handler dispatches on. A `Subcommand` variant that
/// is added to the handler's `match` but forgotten in the enum compiles
/// fine and simply ships unreachable, which is exactly the defect this
/// whole exercise exists to remove.
#[test]
fn stage_three_families_expose_every_subcommand() {
    let cli = Cli::command();
    for (family, subcommands) in [
        (
            "quality",
            &["check", "batch", "artifacts", "error-map", "histogram"][..],
        ),
        (
            "training",
            &["summary", "smooth", "report", "resume", "telemetry"][..],
        ),
        (
            "runs",
            &[
                "new", "list", "show", "status", "rename", "delete", "prune", "stats",
            ][..],
        ),
        ("video", &["build", "viewer"][..]),
        (
            "pipeline",
            &["plan", "track", "diffuse", "export", "status"][..],
        ),
    ] {
        let parent = cli.find_subcommand(family);
        assert!(parent.is_some(), "`oxigaf {family}` is not registered");
        if let Some(parent) = parent {
            for name in subcommands {
                assert!(
                    parent.find_subcommand(name).is_some(),
                    "`oxigaf {family} {name}` is not registered"
                );
            }
        }
    }
}

/// `oxigaf completions <shell>` renders whatever `Cli::command()`
/// describes, so completion coverage is a property of the parser: every
/// top-level subcommand has to carry a `long_about`/`about` line, or it
/// shows up in the generated script with no description at all.
#[test]
fn every_subcommand_documents_itself_for_completions() {
    fn check(command: &clap::Command, path: &str) {
        for sub in command.get_subcommands() {
            let name = sub.get_name();
            let full = if path.is_empty() {
                name.to_string()
            } else {
                format!("{path} {name}")
            };
            assert!(
                sub.get_about().is_some(),
                "`oxigaf {full}` has no help text; shell completion would offer it blank"
            );
            check(sub, &full);
        }
    }
    check(&Cli::command(), "");
}

/// The global `--dry-run` must stop `benchmark`/`config init` before
/// they write, not merely narrate what they already did.
#[test]
fn dry_run_writes_reports_without_creating_the_file() {
    let path = std::env::temp_dir().join("oxigaf_main_dry_run_writes.toml");
    let _ = std::fs::remove_file(&path);
    assert!(dry_run_writes("config", &[path.as_path()], false).is_ok());
    assert!(!path.exists(), "dry run must not create {}", path.display());
}

/// Regression: `--json` promises stdout carries exactly one JSON value.
/// Commands that print their document and *then* fail on what it says
/// (`quality check` below threshold, `doctor` with a dead GPU, `cache
/// verify` with a missing asset, `setup --offline` with an uncached one)
/// used to have a second, error document appended, which `jq` rejects —
/// and a scripted caller could not distinguish that from a broken tool.
#[test]
fn a_second_json_document_is_never_appended() {
    // Nothing printed yet: the failure is the document.
    assert!(should_emit_error_document(true, false));
    // The handler already printed one: the failure goes to stderr and
    // only the exit status carries it.
    assert!(!should_emit_error_document(true, true));
    // Without --json the question does not arise.
    assert!(!should_emit_error_document(false, false));
    assert!(!should_emit_error_document(false, true));
}

#[test]
fn reject_json_only_fires_in_json_mode() {
    assert!(reject_json(false, "info", "use inspect").is_ok());
    let refused = reject_json(true, "info", "use inspect");
    assert!(refused.is_err());
    if let Err(e) = refused {
        let message = format!("{e}");
        assert!(message.contains("info"), "message was: {message}");
        assert!(message.contains("use inspect"), "message was: {message}");
    }
}

/// Handler-level regression for `ExportFormat::All`:
/// `export::export_all_formats_parallel` used to be complete, tested,
/// documented public API that nothing in the binary could reach — no clap
/// value selected it and `cmd_export`'s match had no arm for it. This drives
/// `cmd_export` itself (not just the parser) with `--format all` and checks
/// every artifact `export_all_formats_parallel` promises actually lands on
/// disk under `--output` treated as a directory.
///
/// `main.rs` denies `clippy::unwrap_used`/`clippy::expect_used` crate-wide,
/// tests included (unlike the sibling library crate's), so fallible setup
/// here goes through `assert!` + `let-else` rather than `.expect()`.
#[test]
fn format_all_writes_every_artifact() {
    use oxigaf::render::gaussian::GaussianAttributes;
    use std::ffi::OsString;

    let dir = std::env::temp_dir().join(format!("oxigaf_cli_format_all_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        std::fs::create_dir_all(&dir).is_ok(),
        "failed to create temp dir {}",
        dir.display()
    );

    let model_path = dir.join("in.ply");
    let model = GaussianModel {
        gaussians: vec![GaussianAttributes {
            position: [0.0, 1.0, 2.0],
            _pad0: 0.0,
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [-1.0, -1.0, -1.0],
            opacity: 0.0,
        }],
        sh_coeffs: vec![0.1, 0.2, 0.3],
        sh_degree: 0,
        face_indices: vec![0],
        barycentric: vec![[1.0 / 3.0; 3]],
        local_offsets: vec![[0.0; 3]],
        is_rigid: vec![true],
    };
    assert!(
        export::export_ply(&model, &model_path).is_ok(),
        "failed to write fixture PLY"
    );

    let out_dir = dir.join("out");
    let argv: Vec<OsString> = vec![
        "oxigaf".into(),
        "export".into(),
        "-m".into(),
        model_path.clone().into(),
        "-o".into(),
        out_dir.clone().into(),
        "--format".into(),
        "all".into(),
    ];
    let parsed = Cli::try_parse_from(argv);
    assert!(parsed.is_ok(), "command line should parse: {parsed:?}");
    let Ok(cli) = parsed else { return };

    let export_args = match cli.command {
        Command::Export(args) => Some(args),
        _ => None,
    };
    assert!(export_args.is_some(), "expected Command::Export");
    let Some(args) = export_args else { return };

    let result = cmd_export(args, Verbosity::Quiet, false, false);
    assert!(result.is_ok(), "cmd_export failed: {result:?}");

    for name in EXPORT_ALL_FILENAMES {
        assert!(
            out_dir.join(name).exists(),
            "`--format all` did not write {name}"
        );
    }

    let _ = std::fs::remove_dir_all(&dir);
}
