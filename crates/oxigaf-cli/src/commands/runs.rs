//! `oxigaf runs` — lifecycle management for training run workspaces.
//!
//! Glue over [`crate::workspace_manager`]. A *run workspace* is a directory
//! holding `checkpoints/`, `logs/`, `renders/`, `exports/` and a
//! `workspace.cfg` describing the run (name, description, tags, status).
//! This family creates, lists, renames, prunes and retires them.
//!
//! # Relationship to `oxigaf workspace`
//!
//! `oxigaf workspace` ([`crate::checkpoint_browser`]) looks *inside* one run
//! directory and ranks the checkpoints it finds. `oxigaf runs` manages the
//! *collection* of run directories under a root. They compose:
//!
//! ```bash
//! oxigaf runs new nightly-2026-08-25
//! oxigaf train -o $(oxigaf runs show nightly-2026-08-25 --json | jq -r .result.root) …
//! oxigaf workspace checkpoints <that root>/checkpoints
//! ```
//!
//! # Exit codes
//!
//! A missing or malformed run directory is
//! [`crate::error::CliError::InputInvalid`] → [`crate::error::EXIT_IO_ERROR`],
//! so a script can tell "no such run" from "the command itself failed".
//!
//! # `--dry-run`
//!
//! Every mutating subcommand (`new`, `rename`, `delete`, `prune`,
//! `status --set`) reports what it would change and stops before touching
//! the filesystem.

use std::path::PathBuf;

use anyhow::Result;
use clap::{Args, Subcommand, ValueEnum, ValueHint};
use serde_json::{json, Value};

use crate::commands::image_io::input_invalid;
use crate::commands::{emit, CmdContext};
use crate::progress_types::OperationSpinner;
use crate::workspace_manager::{
    ws_checkpoint_size, ws_compute_stats, ws_format_stats, ws_format_status_counts,
    ws_format_summary, ws_format_table, ws_list_checkpoints, ws_prune_checkpoints,
    ws_timestamped_name, Workspace, WorkspaceConfig, WorkspaceManager, WorkspaceStats,
    WorkspaceStatus,
};

/// `oxigaf runs <command>`.
#[derive(Debug, Args)]
pub struct RunsArgs {
    /// Directory holding the run workspaces.
    ///
    /// Every subcommand resolves names relative to this root, so a project
    /// that keeps its runs in `./experiments` passes `--root experiments`
    /// once per invocation.
    #[arg(long, global = true, default_value = ".", value_hint = ValueHint::DirPath)]
    pub root: PathBuf,

    #[command(subcommand)]
    pub command: RunsCommand,
}

/// Run-workspace subcommands.
#[derive(Debug, Subcommand)]
pub enum RunsCommand {
    /// Create a new run workspace with its standard subdirectories.
    New(NewArgs),

    /// List the run workspaces under the root.
    List(ListArgs),

    /// Report one run's configuration, status, checkpoints and disk usage.
    Show(ShowArgs),

    /// Read or set a run's lifecycle status.
    Status(StatusArgs),

    /// Rename a run workspace and its recorded name.
    Rename(RenameArgs),

    /// Delete a run workspace and everything in it.
    Delete(DeleteArgs),

    /// Keep only the newest N checkpoints of a run.
    Prune(PruneArgs),

    /// Count the runs under the root by status.
    Stats,
}

/// Lifecycle status, as spelled on the command line.
///
/// [`WorkspaceStatus`] cannot derive clap's `ValueEnum` (the library module
/// has no clap dependency), so the mapping lives here.
#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
pub enum RunStatus {
    /// Created, no training started.
    NotStarted,
    /// A lock file says training is in progress.
    Running,
    /// A completion marker is present.
    Completed,
    /// A failure marker is present.
    Failed,
    /// Retired and kept only for reference.
    Archived,
}

impl From<RunStatus> for WorkspaceStatus {
    fn from(value: RunStatus) -> Self {
        match value {
            RunStatus::NotStarted => WorkspaceStatus::NotStarted,
            RunStatus::Running => WorkspaceStatus::Running,
            RunStatus::Completed => WorkspaceStatus::Completed,
            RunStatus::Failed => WorkspaceStatus::Failed,
            RunStatus::Archived => WorkspaceStatus::Archived,
        }
    }
}

/// Arguments for `oxigaf runs new`.
#[derive(Debug, Args)]
pub struct NewArgs {
    /// Name of the run (letters, digits, `-` and `_`).
    pub name: String,

    /// Free-text description stored in `workspace.cfg`.
    #[arg(long, default_value = "")]
    pub description: String,

    /// Tag to attach; repeat the flag for several.
    #[arg(long = "tag")]
    pub tags: Vec<String>,

    /// Model type recorded in the config.
    #[arg(long, default_value = "3dgs_avatar")]
    pub model_type: String,

    /// How many checkpoints `oxigaf runs prune` keeps by default.
    #[arg(long, default_value = "5")]
    pub keep_checkpoints: usize,

    /// Append a UTC timestamp to the name, making it unique per invocation.
    #[arg(long)]
    pub timestamped: bool,
}

/// Arguments for `oxigaf runs list`.
#[derive(Debug, Args)]
pub struct ListArgs {
    /// Show only runs with this status.
    #[arg(long, value_enum)]
    pub status: Option<RunStatus>,

    /// Show only runs carrying this tag.
    #[arg(long)]
    pub tag: Option<String>,
}

/// Arguments for `oxigaf runs show`.
#[derive(Debug, Args)]
pub struct ShowArgs {
    /// Name of the run.
    pub name: String,

    /// Also list the checkpoint file names.
    #[arg(long)]
    pub checkpoints: bool,
}

/// Arguments for `oxigaf runs status`.
#[derive(Debug, Args)]
pub struct StatusArgs {
    /// Name of the run.
    pub name: String,

    /// Set the status instead of reporting it.
    #[arg(long, value_enum)]
    pub set: Option<RunStatus>,

    /// Rewrite the recorded status to match the marker files on disk.
    #[arg(long, conflicts_with = "set")]
    pub reconcile: bool,
}

/// Arguments for `oxigaf runs rename`.
#[derive(Debug, Args)]
pub struct RenameArgs {
    /// Current name.
    pub old_name: String,
    /// New name.
    pub new_name: String,
}

/// Arguments for `oxigaf runs delete`.
#[derive(Debug, Args)]
pub struct DeleteArgs {
    /// Name of the run to delete.
    pub name: String,

    /// Delete without the "are you sure" guard.
    ///
    /// Without it the command refuses, because the deletion is recursive and
    /// removes every checkpoint, log and render the run produced.
    #[arg(long)]
    pub force: bool,
}

/// Arguments for `oxigaf runs prune`.
#[derive(Debug, Args)]
pub struct PruneArgs {
    /// Name of the run.
    pub name: String,

    /// Number of newest checkpoints to keep.
    ///
    /// Defaults to the run's own `max_checkpoints_to_keep`.
    #[arg(long)]
    pub keep: Option<usize>,
}

/// Run the `runs` family.
///
/// # Errors
///
/// Propagates missing roots, unknown run names, invalid names, and refusals
/// to delete a running workspace.
pub fn run(args: RunsArgs, ctx: CmdContext) -> Result<()> {
    let manager = WorkspaceManager::new(args.root.clone());
    match args.command {
        RunsCommand::New(new_args) => cmd_new(&manager, new_args, &ctx),
        RunsCommand::List(list_args) => cmd_list(&manager, list_args, &ctx),
        RunsCommand::Show(show_args) => cmd_show(&manager, show_args, &ctx),
        RunsCommand::Status(status_args) => cmd_status(&manager, status_args, &ctx),
        RunsCommand::Rename(rename_args) => cmd_rename(&manager, rename_args, &ctx),
        RunsCommand::Delete(delete_args) => cmd_delete(&manager, delete_args, &ctx),
        RunsCommand::Prune(prune_args) => cmd_prune(&manager, prune_args, &ctx),
        RunsCommand::Stats => cmd_stats(&manager, &ctx),
    }
}

// ---------------------------------------------------------------------------
// JSON helpers
// ---------------------------------------------------------------------------

/// The JSON shape of one workspace.
fn workspace_json(workspace: &Workspace) -> Value {
    let config = &workspace.config;
    json!({
        "name": config.name,
        "root": workspace.root.display().to_string(),
        "description": config.description,
        "tags": config.tags,
        "model_type": config.model_type,
        // The recorded status and the one implied by the marker files can
        // disagree — `runs status --reconcile` is what fixes that — so both
        // are reported rather than silently preferring one.
        "recorded_status": config.status.as_str(),
        "detected_status": workspace.detect_status().as_str(),
        "created_at": config.created_at,
        "modified_at": config.modified_at,
        "max_checkpoints_to_keep": config.max_checkpoints_to_keep,
        "checkpoint_count": workspace.checkpoint_count(),
        "checkpoint_bytes": ws_checkpoint_size(workspace),
        "disk_usage_bytes": workspace.disk_usage(),
    })
}

/// The JSON shape of [`WorkspaceStats`].
fn stats_json(stats: &WorkspaceStats) -> Value {
    json!({
        "name": stats.name,
        "status": stats.status.as_str(),
        "disk_usage_bytes": stats.disk_usage_bytes,
        "checkpoint_count": stats.checkpoint_count,
        "age_seconds": stats.age_seconds,
        "tags": stats.tags,
    })
}

/// Load a run, mapping "no such run" onto the I/O exit status.
fn load(manager: &WorkspaceManager, name: &str) -> Result<Workspace> {
    manager.load(name).map_err(|e| {
        input_invalid(
            &manager.root.join(name),
            format!("is not a usable run workspace: {e}"),
        )
    })
}

// ---------------------------------------------------------------------------
// runs new
// ---------------------------------------------------------------------------

fn cmd_new(manager: &WorkspaceManager, args: NewArgs, ctx: &CmdContext) -> Result<()> {
    if args.keep_checkpoints == 0 {
        anyhow::bail!("--keep-checkpoints must be at least 1");
    }

    let name = if args.timestamped {
        ws_timestamped_name(&args.name)?
    } else {
        args.name.clone()
    };

    // `WorkspaceConfig::new` validates the name, so a bad one fails here
    // rather than after the directory tree has been half created.
    let mut config = WorkspaceConfig::new(&name)?;
    config.description = args.description.clone();
    config.tags = args.tags.clone();
    config.model_type = args.model_type.clone();
    config.max_checkpoints_to_keep = args.keep_checkpoints;

    if manager.exists(&name) {
        anyhow::bail!(
            "a run named '{name}' already exists under {}",
            manager.root.display()
        );
    }

    let root = manager.root.join(&name);
    if ctx.dry_run {
        emit(
            ctx,
            "runs new",
            json!({
                "dry_run": true,
                "name": name,
                "would_create": [
                    format!("{}/", root.display()),
                    format!("{}/checkpoints/", root.display()),
                    format!("{}/logs/", root.display()),
                    format!("{}/renders/", root.display()),
                    format!("{}/exports/", root.display()),
                    format!("{}/workspace.cfg", root.display()),
                ],
            }),
            &[],
            || println!("Would create run workspace: {}", root.display()),
        );
        return Ok(());
    }

    let workspace = manager.create_with_config(config)?;
    emit(ctx, "runs new", workspace_json(&workspace), &[], || {
        println!("Created run workspace: {}", workspace.root.display());
        println!("{}", ws_format_summary(&workspace));
    });
    Ok(())
}

// ---------------------------------------------------------------------------
// runs list
// ---------------------------------------------------------------------------

fn cmd_list(manager: &WorkspaceManager, args: ListArgs, ctx: &CmdContext) -> Result<()> {
    if !manager.root.is_dir() {
        return Err(input_invalid(&manager.root, "not an existing directory"));
    }

    // Listing reads and parses every `workspace.cfg` under the root, which
    // is slow enough on a machine with hundreds of runs to be worth saying
    // something about.
    let spinner = if ctx.human() && ctx.verbosity.show_progress() {
        Some(OperationSpinner::new(format!(
            "Scanning {} for run workspaces…",
            manager.root.display()
        )))
    } else {
        None
    };

    let listed = match (args.status, args.tag.as_deref()) {
        (Some(status), _) => manager.list_by_status(&WorkspaceStatus::from(status)),
        (None, Some(tag)) => manager.list_by_tag(tag),
        (None, None) => manager.list(),
    };
    let workspaces = match listed {
        Ok(workspaces) => {
            if let Some(ref spinner) = spinner {
                spinner.finish_ok();
            }
            workspaces
        }
        Err(e) => {
            if let Some(ref spinner) = spinner {
                spinner.fail(e.to_string());
            }
            return Err(e.into());
        }
    };

    // `--status` and `--tag` are independent filters; the manager exposes one
    // helper per filter, so combining them is done here.
    let workspaces: Vec<Workspace> = match (args.status, args.tag.as_deref()) {
        (Some(_), Some(tag)) => workspaces
            .into_iter()
            .filter(|workspace| workspace.config.tags.iter().any(|t| t == tag))
            .collect(),
        _ => workspaces,
    };

    emit(
        ctx,
        "runs list",
        json!({
            "root": manager.root.display().to_string(),
            "count": workspaces.len(),
            "runs": workspaces.iter().map(workspace_json).collect::<Vec<_>>(),
        }),
        &[],
        || {
            if workspaces.is_empty() {
                println!("No run workspaces under {}", manager.root.display());
            } else {
                println!("{}", ws_format_table(&workspaces));
            }
        },
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// runs show
// ---------------------------------------------------------------------------

fn cmd_show(manager: &WorkspaceManager, args: ShowArgs, ctx: &CmdContext) -> Result<()> {
    let workspace = load(manager, &args.name)?;
    // Report structural problems (a missing `checkpoints/`, say) instead of
    // presenting a half-built directory as healthy.
    let validation = workspace.validate().err().map(|e| e.to_string());
    let stats = ws_compute_stats(&workspace);
    let checkpoints = if args.checkpoints {
        ws_list_checkpoints(&workspace)
    } else {
        Vec::new()
    };

    let mut document = workspace_json(&workspace);
    if let Value::Object(ref mut map) = document {
        map.insert("stats".to_string(), stats_json(&stats));
        map.insert(
            "validation_error".to_string(),
            validation
                .as_ref()
                .map(|message| json!(message))
                .unwrap_or(Value::Null),
        );
        if args.checkpoints {
            map.insert("checkpoints".to_string(), json!(checkpoints));
        }
    }

    emit(ctx, "runs show", document, &[], || {
        println!("{}", ws_format_summary(&workspace));
        println!();
        println!("{}", ws_format_stats(&stats));
        if let Some(ref message) = validation {
            println!("\nWARNING: {message}");
        }
        if args.checkpoints {
            println!("\nCheckpoints ({}):", checkpoints.len());
            for name in &checkpoints {
                println!("  {name}");
            }
        }
    });
    Ok(())
}

// ---------------------------------------------------------------------------
// runs status
// ---------------------------------------------------------------------------

fn cmd_status(manager: &WorkspaceManager, args: StatusArgs, ctx: &CmdContext) -> Result<()> {
    let mut workspace = load(manager, &args.name)?;
    let before = workspace.config.status.clone();
    let detected = workspace.detect_status();

    let (action, changed) = match (args.set, args.reconcile) {
        (Some(status), _) => {
            let target = WorkspaceStatus::from(status);
            if ctx.dry_run {
                ("would-set", before != target)
            } else {
                workspace.set_status(target.clone())?;
                ("set", before != target)
            }
        }
        (None, true) => {
            if ctx.dry_run {
                ("would-reconcile", before != detected)
            } else {
                let changed = workspace.reconcile_status()?;
                ("reconcile", changed)
            }
        }
        (None, false) => ("report", false),
    };

    emit(
        ctx,
        "runs status",
        json!({
            "name": workspace.config.name,
            "root": workspace.root.display().to_string(),
            "action": action,
            "changed": changed,
            "previous_status": before.as_str(),
            "recorded_status": workspace.config.status.as_str(),
            "detected_status": detected.as_str(),
            "dry_run": ctx.dry_run,
        }),
        &[],
        || {
            println!(
                "{}: {}",
                workspace.config.name,
                workspace.config.status.as_str()
            );
            if detected != workspace.config.status {
                println!(
                    "  marker files on disk say '{}' — run `oxigaf runs status {} --reconcile`",
                    detected.as_str(),
                    workspace.config.name
                );
            }
            if changed {
                println!("  changed from '{}'", before.as_str());
            }
        },
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// runs rename
// ---------------------------------------------------------------------------

fn cmd_rename(manager: &WorkspaceManager, args: RenameArgs, ctx: &CmdContext) -> Result<()> {
    // Confirm the source exists before reporting a rename that cannot happen.
    let _ = load(manager, &args.old_name)?;

    if ctx.dry_run {
        emit(
            ctx,
            "runs rename",
            json!({
                "dry_run": true,
                "from": manager.root.join(&args.old_name).display().to_string(),
                "to": manager.root.join(&args.new_name).display().to_string(),
            }),
            &[],
            || {
                println!(
                    "Would rename {} → {}",
                    manager.root.join(&args.old_name).display(),
                    manager.root.join(&args.new_name).display()
                );
            },
        );
        return Ok(());
    }

    let workspace = manager.rename(&args.old_name, &args.new_name)?;
    emit(ctx, "runs rename", workspace_json(&workspace), &[], || {
        println!(
            "Renamed '{}' → '{}' ({})",
            args.old_name,
            args.new_name,
            workspace.root.display()
        );
    });
    Ok(())
}

// ---------------------------------------------------------------------------
// runs delete
// ---------------------------------------------------------------------------

fn cmd_delete(manager: &WorkspaceManager, args: DeleteArgs, ctx: &CmdContext) -> Result<()> {
    let workspace = load(manager, &args.name)?;
    let bytes = workspace.disk_usage();
    let checkpoints = workspace.checkpoint_count();
    let root = workspace.root.clone();

    // The guard applies to `--dry-run` too. A dry run is meant to report the
    // command that *would* run, and `runs delete NAME` without `--force`
    // would not run at all — printing "would delete …" for it would send a
    // caller off to run a command that then refuses.
    if !args.force {
        anyhow::bail!(
            "refusing to delete '{}' ({} checkpoint(s), {} bytes) without --force; \
             the deletion is recursive and cannot be undone",
            args.name,
            checkpoints,
            bytes
        );
    }

    if ctx.dry_run {
        emit(
            ctx,
            "runs delete",
            json!({
                "dry_run": true,
                "name": args.name,
                "would_delete": [root.display().to_string()],
                "checkpoint_count": checkpoints,
                "disk_usage_bytes": bytes,
            }),
            &[],
            || {
                println!(
                    "Would delete {} ({checkpoints} checkpoint(s), {bytes} bytes)",
                    root.display()
                );
            },
        );
        return Ok(());
    }

    // `delete` refuses a workspace whose lock file says training is live.
    manager.delete(&args.name)?;
    emit(
        ctx,
        "runs delete",
        json!({
            "name": args.name,
            "deleted": root.display().to_string(),
            "checkpoint_count": checkpoints,
            "disk_usage_bytes": bytes,
        }),
        &[],
        || println!("Deleted {} ({bytes} bytes)", root.display()),
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// runs prune
// ---------------------------------------------------------------------------

fn cmd_prune(manager: &WorkspaceManager, args: PruneArgs, ctx: &CmdContext) -> Result<()> {
    let workspace = load(manager, &args.name)?;
    let keep = args
        .keep
        .unwrap_or(workspace.config.max_checkpoints_to_keep);
    let before = ws_list_checkpoints(&workspace);
    let would_remove = before.len().saturating_sub(keep);

    if ctx.dry_run {
        let doomed: Vec<&String> = before.iter().take(would_remove).collect();
        emit(
            ctx,
            "runs prune",
            json!({
                "dry_run": true,
                "name": args.name,
                "keep": keep,
                "checkpoints_before": before.len(),
                "would_delete": doomed,
            }),
            &[],
            || {
                println!(
                    "Would remove {would_remove} of {} checkpoint(s), keeping the newest {keep}",
                    before.len()
                );
                for name in &doomed {
                    println!("  {name}");
                }
            },
        );
        return Ok(());
    }

    let removed = ws_prune_checkpoints(&workspace, keep)?;
    let after = ws_list_checkpoints(&workspace);
    emit(
        ctx,
        "runs prune",
        json!({
            "name": args.name,
            "keep": keep,
            "checkpoints_before": before.len(),
            "checkpoints_after": after.len(),
            "removed": removed,
            "checkpoint_bytes": ws_checkpoint_size(&workspace),
        }),
        &[],
        || {
            println!(
                "Removed {removed} checkpoint(s) from '{}'; {} remain",
                args.name,
                after.len()
            );
        },
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// runs stats
// ---------------------------------------------------------------------------

fn cmd_stats(manager: &WorkspaceManager, ctx: &CmdContext) -> Result<()> {
    if !manager.root.is_dir() {
        return Err(input_invalid(&manager.root, "not an existing directory"));
    }
    let counts = manager.status_counts()?;

    emit(
        ctx,
        "runs stats",
        json!({
            "root": manager.root.display().to_string(),
            "total": counts.total,
            "not_started": counts.not_started,
            "running": counts.running,
            "completed": counts.completed,
            "failed": counts.failed,
            "archived": counts.archived,
        }),
        &[],
        || println!("{}", ws_format_status_counts(&counts)),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::EXIT_IO_ERROR;
    use crate::verbosity::Verbosity;

    fn ctx(dry_run: bool) -> CmdContext {
        CmdContext::new(Verbosity::Quiet, true, dry_run)
    }

    /// A fresh, empty root directory under the system temp dir.
    fn temp_root(name: &str) -> PathBuf {
        let root = std::env::temp_dir().join(name);
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).expect("temp root");
        root
    }

    fn new_args(name: &str) -> NewArgs {
        NewArgs {
            name: name.to_string(),
            description: "test run".to_string(),
            tags: vec!["ci".to_string()],
            model_type: "3dgs_avatar".to_string(),
            keep_checkpoints: 5,
            timestamped: false,
        }
    }

    #[test]
    fn new_then_show_round_trips() {
        let root = temp_root("oxigaf_runs_round_trip");
        let manager = WorkspaceManager::new(root.clone());

        assert!(cmd_new(&manager, new_args("alpha"), &ctx(false)).is_ok());
        assert!(manager.exists("alpha"));
        assert!(root.join("alpha").join("checkpoints").is_dir());

        let show = ShowArgs {
            name: "alpha".to_string(),
            checkpoints: true,
        };
        assert!(cmd_show(&manager, show, &ctx(false)).is_ok());

        let _ = std::fs::remove_dir_all(&root);
    }

    /// Regression: `--dry-run` is global and documented as making no
    /// modifications, so it must stop `runs new` *before* the directory tree
    /// exists — not narrate a tree it already created.
    #[test]
    fn dry_run_new_creates_nothing() {
        let root = temp_root("oxigaf_runs_dry_run");
        let manager = WorkspaceManager::new(root.clone());

        assert!(cmd_new(&manager, new_args("beta"), &ctx(true)).is_ok());
        assert!(
            !root.join("beta").exists(),
            "dry run must not create {}",
            root.join("beta").display()
        );

        let _ = std::fs::remove_dir_all(&root);
    }

    /// A recursive delete must not happen just because the name parsed.
    #[test]
    fn delete_requires_force() {
        let root = temp_root("oxigaf_runs_delete_guard");
        let manager = WorkspaceManager::new(root.clone());
        assert!(cmd_new(&manager, new_args("gamma"), &ctx(false)).is_ok());

        let guarded = DeleteArgs {
            name: "gamma".to_string(),
            force: false,
        };
        assert!(cmd_delete(&manager, guarded, &ctx(false)).is_err());
        assert!(root.join("gamma").exists(), "the run must still be there");

        let forced = DeleteArgs {
            name: "gamma".to_string(),
            force: true,
        };
        assert!(cmd_delete(&manager, forced, &ctx(false)).is_ok());
        assert!(!root.join("gamma").exists());

        let _ = std::fs::remove_dir_all(&root);
    }

    /// Regression: `--dry-run` used to bypass the `--force` guard and print
    /// "would delete …" for a command that, run for real, refuses — sending
    /// the caller off to a command that does not work.
    #[test]
    fn dry_run_delete_still_requires_force() {
        let root = temp_root("oxigaf_runs_dry_delete");
        let manager = WorkspaceManager::new(root.clone());
        assert!(cmd_new(&manager, new_args("delta"), &ctx(false)).is_ok());

        let unguarded = DeleteArgs {
            name: "delta".to_string(),
            force: false,
        };
        assert!(cmd_delete(&manager, unguarded, &ctx(true)).is_err());

        let guarded = DeleteArgs {
            name: "delta".to_string(),
            force: true,
        };
        assert!(cmd_delete(&manager, guarded, &ctx(true)).is_ok());
        assert!(
            root.join("delta").exists(),
            "a dry run must not delete the workspace"
        );

        let _ = std::fs::remove_dir_all(&root);
    }

    /// An unknown run name is a bad *input*, so it has to carry the I/O exit
    /// status rather than collapsing into the catch-all 1.
    #[test]
    fn unknown_run_is_an_input_error() {
        let root = temp_root("oxigaf_runs_unknown");
        let manager = WorkspaceManager::new(root.clone());
        let err = load(&manager, "nope")
            .err()
            .expect("an unknown run must not load");
        assert_eq!(
            crate::commands::runtime::to_cli_error(err).exit_code(),
            EXIT_IO_ERROR
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn invalid_names_are_refused_before_anything_is_written() {
        let root = temp_root("oxigaf_runs_bad_name");
        let manager = WorkspaceManager::new(root.clone());
        assert!(cmd_new(&manager, new_args("has spaces"), &ctx(false)).is_err());
        assert!(cmd_new(&manager, new_args("../escape"), &ctx(false)).is_err());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn status_maps_onto_the_library_enum() {
        assert_eq!(
            WorkspaceStatus::from(RunStatus::Completed),
            WorkspaceStatus::Completed
        );
        assert_eq!(
            WorkspaceStatus::from(RunStatus::NotStarted),
            WorkspaceStatus::NotStarted
        );
    }
}
