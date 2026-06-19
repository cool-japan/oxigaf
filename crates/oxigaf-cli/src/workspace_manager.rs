//! Workspace management for OxiGAF training runs.
//!
//! This module manages training workspaces — directory structures that hold
//! training runs (configs, checkpoints, logs, renders, exports). A workspace
//! tracks everything about a single training experiment.

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use thiserror::Error;

// ---------------------------------------------------------------------------
// WorkspaceError
// ---------------------------------------------------------------------------

/// Errors produced by workspace management operations.
#[derive(Debug, Error)]
pub enum WorkspaceError {
    #[error("workspace not found: {0}")]
    NotFound(String),
    #[error("workspace already exists: {0}")]
    AlreadyExists(String),
    #[error("invalid workspace: {0}")]
    InvalidWorkspace(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid config: {0}")]
    InvalidConfig(String),
    #[error("invalid name: names must be alphanumeric with hyphens/underscores, got '{0}'")]
    InvalidName(String),
}

// ---------------------------------------------------------------------------
// WorkspaceStatus
// ---------------------------------------------------------------------------

/// The lifecycle status of a training workspace.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkspaceStatus {
    /// Created but no training run has started yet.
    NotStarted,
    /// Has a lock file indicating active training.
    Running,
    /// Has a completion marker file.
    Completed,
    /// Has a failure marker file.
    Failed,
    /// Has an archive marker file.
    Archived,
}

impl WorkspaceStatus {
    /// Return the canonical string representation of this status.
    pub fn as_str(&self) -> &'static str {
        match self {
            WorkspaceStatus::NotStarted => "not_started",
            WorkspaceStatus::Running => "running",
            WorkspaceStatus::Completed => "completed",
            WorkspaceStatus::Failed => "failed",
            WorkspaceStatus::Archived => "archived",
        }
    }

    /// Parse a status from its canonical string representation.
    pub fn parse_status(s: &str) -> Option<WorkspaceStatus> {
        match s {
            "not_started" => Some(WorkspaceStatus::NotStarted),
            "running" => Some(WorkspaceStatus::Running),
            "completed" => Some(WorkspaceStatus::Completed),
            "failed" => Some(WorkspaceStatus::Failed),
            "archived" => Some(WorkspaceStatus::Archived),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// WorkspaceConfig
// ---------------------------------------------------------------------------

/// Metadata configuration stored in `workspace.cfg` as plain key=value text.
#[derive(Debug, Clone)]
pub struct WorkspaceConfig {
    pub name: String,
    pub description: String,
    pub tags: Vec<String>,
    /// Unix timestamp (seconds) when workspace was created.
    pub created_at: u64,
    /// Unix timestamp (seconds) of most recent modification.
    pub modified_at: u64,
    pub status: WorkspaceStatus,
    /// Model type, e.g. `"3dgs_avatar"`.
    pub model_type: String,
    /// Maximum number of checkpoints to retain.
    pub max_checkpoints_to_keep: usize,
    /// Extra arbitrary key-value metadata.
    pub custom_fields: Vec<(String, String)>,
}

impl WorkspaceConfig {
    /// Create a new config with default values. Validates `name`.
    pub fn new(name: &str) -> Result<Self, WorkspaceError> {
        ws_validate_name(name)?;
        let now = ws_current_timestamp();
        Ok(WorkspaceConfig {
            name: name.to_owned(),
            description: String::new(),
            tags: Vec::new(),
            created_at: now,
            modified_at: now,
            status: WorkspaceStatus::NotStarted,
            model_type: "3dgs_avatar".to_owned(),
            max_checkpoints_to_keep: 5,
            custom_fields: Vec::new(),
        })
    }

    /// Serialize the config to a `key=value` text format.
    #[allow(clippy::inherent_to_string)]
    pub fn to_string(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!("name={}", self.name));
        lines.push(format!("description={}", self.description));
        lines.push(format!("tags={}", self.tags.join(",")));
        lines.push(format!("created_at={}", self.created_at));
        lines.push(format!("modified_at={}", self.modified_at));
        lines.push(format!("status={}", self.status.as_str()));
        lines.push(format!("model_type={}", self.model_type));
        lines.push(format!(
            "max_checkpoints_to_keep={}",
            self.max_checkpoints_to_keep
        ));
        for (k, v) in &self.custom_fields {
            lines.push(format!("custom.{}={}", k, v));
        }
        lines.join("\n")
    }

    /// Parse a config from a `key=value` text representation.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Result<Self, WorkspaceError> {
        let mut name: Option<String> = None;
        let mut description = String::new();
        let mut tags: Vec<String> = Vec::new();
        let mut created_at: Option<u64> = None;
        let mut modified_at: Option<u64> = None;
        let mut status = WorkspaceStatus::NotStarted;
        let mut model_type = "3dgs_avatar".to_owned();
        let mut max_checkpoints_to_keep: usize = 5;
        let mut custom_fields: Vec<(String, String)> = Vec::new();

        for line in s.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut parts = line.splitn(2, '=');
            let key = parts.next().unwrap_or("").trim();
            let val = parts.next().unwrap_or("").trim();

            if key.starts_with("custom.") {
                let field_name = key.trim_start_matches("custom.").to_owned();
                if !field_name.is_empty() {
                    custom_fields.push((field_name, val.to_owned()));
                }
                continue;
            }

            match key {
                "name" => name = Some(val.to_owned()),
                "description" => description = val.to_owned(),
                "tags" => {
                    if val.is_empty() {
                        tags = Vec::new();
                    } else {
                        tags = val.split(',').map(|t| t.trim().to_owned()).collect();
                    }
                }
                "created_at" => {
                    created_at = val.parse::<u64>().ok();
                }
                "modified_at" => {
                    modified_at = val.parse::<u64>().ok();
                }
                "status" => {
                    if let Some(s) = WorkspaceStatus::parse_status(val) {
                        status = s;
                    }
                }
                "model_type" => model_type = val.to_owned(),
                "max_checkpoints_to_keep" => {
                    if let Ok(n) = val.parse::<usize>() {
                        max_checkpoints_to_keep = n;
                    }
                }
                _ => {} // unknown keys are silently ignored
            }
        }

        let name =
            name.ok_or_else(|| WorkspaceError::InvalidConfig("missing 'name' field".into()))?;
        ws_validate_name(&name)?;

        let now = ws_current_timestamp();
        Ok(WorkspaceConfig {
            name,
            description,
            tags,
            created_at: created_at.unwrap_or(now),
            modified_at: modified_at.unwrap_or(now),
            status,
            model_type,
            max_checkpoints_to_keep,
            custom_fields,
        })
    }
}

// ---------------------------------------------------------------------------
// Workspace
// ---------------------------------------------------------------------------

/// Represents a training workspace directory.
pub struct Workspace {
    /// The root directory of this workspace.
    pub root: PathBuf,
    /// The configuration loaded from `workspace.cfg`.
    pub config: WorkspaceConfig,
}

impl Workspace {
    /// Standard subdirectory for checkpoints.
    pub const CHECKPOINTS_DIR: &'static str = "checkpoints";
    /// Standard subdirectory for logs.
    pub const LOGS_DIR: &'static str = "logs";
    /// Standard subdirectory for renders.
    pub const RENDERS_DIR: &'static str = "renders";
    /// Standard subdirectory for exports.
    pub const EXPORTS_DIR: &'static str = "exports";
    /// Config file name.
    pub const CONFIG_FILE: &'static str = "workspace.cfg";
    /// Lock file name (indicates active training).
    pub const LOCK_FILE: &'static str = ".training.lock";
    /// Completion marker file name.
    pub const DONE_FILE: &'static str = ".done";
    /// Failure marker file name.
    pub const FAILED_FILE: &'static str = ".failed";
    /// Archive marker file name.
    pub const ARCHIVE_FILE: &'static str = ".archived";

    /// Path to the `checkpoints` subdirectory.
    pub fn checkpoints_dir(&self) -> PathBuf {
        self.root.join(Self::CHECKPOINTS_DIR)
    }

    /// Path to the `logs` subdirectory.
    pub fn logs_dir(&self) -> PathBuf {
        self.root.join(Self::LOGS_DIR)
    }

    /// Path to the `renders` subdirectory.
    pub fn renders_dir(&self) -> PathBuf {
        self.root.join(Self::RENDERS_DIR)
    }

    /// Path to the `exports` subdirectory.
    pub fn exports_dir(&self) -> PathBuf {
        self.root.join(Self::EXPORTS_DIR)
    }

    /// Path to `workspace.cfg`.
    pub fn config_path(&self) -> PathBuf {
        self.root.join(Self::CONFIG_FILE)
    }

    /// Verify that the workspace directory structure is valid.
    pub fn validate(&self) -> Result<(), WorkspaceError> {
        if !self.root.is_dir() {
            return Err(WorkspaceError::InvalidWorkspace(format!(
                "root directory does not exist: {}",
                self.root.display()
            )));
        }
        if !self.config_path().is_file() {
            return Err(WorkspaceError::InvalidWorkspace(format!(
                "missing config file: {}",
                self.config_path().display()
            )));
        }
        for dir in &[
            self.checkpoints_dir(),
            self.logs_dir(),
            self.renders_dir(),
            self.exports_dir(),
        ] {
            if !dir.is_dir() {
                return Err(WorkspaceError::InvalidWorkspace(format!(
                    "missing subdirectory: {}",
                    dir.display()
                )));
            }
        }
        Ok(())
    }

    /// Re-read the config from disk, updating `self.config`.
    pub fn reload_config(&mut self) -> Result<(), WorkspaceError> {
        let text = std::fs::read_to_string(self.config_path())?;
        self.config = WorkspaceConfig::from_str(&text)?;
        Ok(())
    }

    /// Write the current config to disk.
    pub fn save_config(&self) -> Result<(), WorkspaceError> {
        std::fs::write(self.config_path(), self.config.to_string())?;
        Ok(())
    }

    /// Detect the current status by inspecting marker files on disk.
    ///
    /// Priority: archived > failed > completed > running > not_started.
    pub fn detect_status(&self) -> WorkspaceStatus {
        if self.root.join(Self::ARCHIVE_FILE).exists() {
            return WorkspaceStatus::Archived;
        }
        if self.root.join(Self::FAILED_FILE).exists() {
            return WorkspaceStatus::Failed;
        }
        if self.root.join(Self::DONE_FILE).exists() {
            return WorkspaceStatus::Completed;
        }
        if self.root.join(Self::LOCK_FILE).exists() {
            return WorkspaceStatus::Running;
        }
        WorkspaceStatus::NotStarted
    }

    /// Update the workspace status by managing marker files.
    ///
    /// Removes all stale marker files before writing the new one.
    pub fn set_status(&mut self, status: WorkspaceStatus) -> Result<(), WorkspaceError> {
        // Remove all existing marker files (ignore missing-file errors).
        for marker in &[
            Self::LOCK_FILE,
            Self::DONE_FILE,
            Self::FAILED_FILE,
            Self::ARCHIVE_FILE,
        ] {
            let path = self.root.join(marker);
            if path.exists() {
                std::fs::remove_file(&path)?;
            }
        }

        // Write the new marker (NotStarted has no marker).
        match &status {
            WorkspaceStatus::Running => {
                std::fs::write(self.root.join(Self::LOCK_FILE), b"")?;
            }
            WorkspaceStatus::Completed => {
                std::fs::write(self.root.join(Self::DONE_FILE), b"")?;
            }
            WorkspaceStatus::Failed => {
                std::fs::write(self.root.join(Self::FAILED_FILE), b"")?;
            }
            WorkspaceStatus::Archived => {
                std::fs::write(self.root.join(Self::ARCHIVE_FILE), b"")?;
            }
            WorkspaceStatus::NotStarted => {}
        }

        self.config.status = status;
        self.config.modified_at = ws_current_timestamp();
        self.save_config()?;
        Ok(())
    }

    /// Count checkpoint files in the checkpoints directory.
    pub fn checkpoint_count(&self) -> usize {
        let dir = self.checkpoints_dir();
        if !dir.is_dir() {
            return 0;
        }
        std::fs::read_dir(&dir)
            .map(|entries| {
                entries
                    .filter_map(|e| e.ok())
                    .filter(|e| e.path().is_file())
                    .count()
            })
            .unwrap_or(0)
    }

    /// Compute the total disk usage of the workspace in bytes.
    pub fn disk_usage(&self) -> u64 {
        dir_size_recursive(&self.root)
    }
}

// ---------------------------------------------------------------------------
// WorkspaceManager
// ---------------------------------------------------------------------------

/// Manages a collection of workspaces under a single root directory.
pub struct WorkspaceManager {
    /// Root directory that contains all workspace subdirectories.
    pub root: PathBuf,
}

impl WorkspaceManager {
    /// Create a new manager for the given root directory.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        WorkspaceManager { root: root.into() }
    }

    /// Create a new workspace with default config.
    pub fn create(&self, name: &str) -> Result<Workspace, WorkspaceError> {
        let config = WorkspaceConfig::new(name)?;
        self.create_with_config(config)
    }

    /// Create a new workspace with a custom config.
    pub fn create_with_config(&self, config: WorkspaceConfig) -> Result<Workspace, WorkspaceError> {
        let ws_root = self.root.join(&config.name);
        if ws_root.exists() {
            return Err(WorkspaceError::AlreadyExists(config.name.clone()));
        }

        // Create root and all standard subdirectories.
        std::fs::create_dir_all(&ws_root)?;
        for sub in &[
            Workspace::CHECKPOINTS_DIR,
            Workspace::LOGS_DIR,
            Workspace::RENDERS_DIR,
            Workspace::EXPORTS_DIR,
        ] {
            std::fs::create_dir_all(ws_root.join(sub))?;
        }

        let workspace = Workspace {
            root: ws_root,
            config,
        };
        workspace.save_config()?;
        Ok(workspace)
    }

    /// Load an existing workspace by name.
    pub fn load(&self, name: &str) -> Result<Workspace, WorkspaceError> {
        let ws_root = self.root.join(name);
        if !ws_root.is_dir() {
            return Err(WorkspaceError::NotFound(name.to_owned()));
        }
        let cfg_path = ws_root.join(Workspace::CONFIG_FILE);
        if !cfg_path.is_file() {
            return Err(WorkspaceError::InvalidWorkspace(format!(
                "workspace '{}' is missing workspace.cfg",
                name
            )));
        }
        let text = std::fs::read_to_string(&cfg_path)?;
        let config = WorkspaceConfig::from_str(&text)?;
        Ok(Workspace {
            root: ws_root,
            config,
        })
    }

    /// List all workspaces, sorted by `modified_at` descending (newest first).
    pub fn list(&self) -> Result<Vec<Workspace>, WorkspaceError> {
        let mut workspaces = Vec::new();

        if !self.root.is_dir() {
            return Ok(workspaces);
        }

        let entries = std::fs::read_dir(&self.root)?;
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            let cfg = path.join(Workspace::CONFIG_FILE);
            if !cfg.is_file() {
                continue;
            }
            if let Ok(text) = std::fs::read_to_string(&cfg) {
                if let Ok(config) = WorkspaceConfig::from_str(&text) {
                    workspaces.push(Workspace { root: path, config });
                }
            }
        }

        workspaces.sort_by_key(|w| std::cmp::Reverse(w.config.modified_at));
        Ok(workspaces)
    }

    /// List workspaces that have the given status.
    pub fn list_by_status(
        &self,
        status: &WorkspaceStatus,
    ) -> Result<Vec<Workspace>, WorkspaceError> {
        let all = self.list()?;
        Ok(all
            .into_iter()
            .filter(|ws| &ws.config.status == status)
            .collect())
    }

    /// List workspaces that contain the given tag.
    pub fn list_by_tag(&self, tag: &str) -> Result<Vec<Workspace>, WorkspaceError> {
        let all = self.list()?;
        Ok(all
            .into_iter()
            .filter(|ws| ws.config.tags.iter().any(|t| t == tag))
            .collect())
    }

    /// Delete a workspace by removing its directory.
    ///
    /// Returns an error if the workspace has `Running` status (safety check).
    pub fn delete(&self, name: &str) -> Result<(), WorkspaceError> {
        let ws = self.load(name)?;
        if ws.detect_status() == WorkspaceStatus::Running {
            return Err(WorkspaceError::InvalidWorkspace(format!(
                "workspace '{}' is currently running; stop training before deleting",
                name
            )));
        }
        std::fs::remove_dir_all(&ws.root)?;
        Ok(())
    }

    /// Rename a workspace directory and update its config.
    pub fn rename(&self, old_name: &str, new_name: &str) -> Result<Workspace, WorkspaceError> {
        ws_validate_name(new_name)?;

        let old_root = self.root.join(old_name);
        if !old_root.is_dir() {
            return Err(WorkspaceError::NotFound(old_name.to_owned()));
        }
        let new_root = self.root.join(new_name);
        if new_root.exists() {
            return Err(WorkspaceError::AlreadyExists(new_name.to_owned()));
        }

        // Load existing config and update the name field.
        let cfg_path = old_root.join(Workspace::CONFIG_FILE);
        let text = std::fs::read_to_string(&cfg_path)?;
        let mut config = WorkspaceConfig::from_str(&text)?;

        // Move directory.
        std::fs::rename(&old_root, &new_root)?;

        config.name = new_name.to_owned();
        config.modified_at = ws_current_timestamp();

        let mut ws = Workspace {
            root: new_root,
            config,
        };
        ws.save_config()?;

        // Reload to ensure round-trip consistency.
        ws.reload_config()?;
        Ok(ws)
    }

    /// Return `true` if a workspace with the given name exists.
    pub fn exists(&self, name: &str) -> bool {
        self.root.join(name).join(Workspace::CONFIG_FILE).is_file()
    }

    /// Count workspaces by status.
    pub fn status_counts(&self) -> Result<WorkspaceStatusCounts, WorkspaceError> {
        let all = self.list()?;
        let mut counts = WorkspaceStatusCounts {
            total: all.len(),
            not_started: 0,
            running: 0,
            completed: 0,
            failed: 0,
            archived: 0,
        };
        for ws in &all {
            match ws.config.status {
                WorkspaceStatus::NotStarted => counts.not_started += 1,
                WorkspaceStatus::Running => counts.running += 1,
                WorkspaceStatus::Completed => counts.completed += 1,
                WorkspaceStatus::Failed => counts.failed += 1,
                WorkspaceStatus::Archived => counts.archived += 1,
            }
        }
        Ok(counts)
    }
}

/// Counts of workspaces grouped by status.
pub struct WorkspaceStatusCounts {
    pub total: usize,
    pub not_started: usize,
    pub running: usize,
    pub completed: usize,
    pub failed: usize,
    pub archived: usize,
}

// ---------------------------------------------------------------------------
// Checkpoint helpers
// ---------------------------------------------------------------------------

/// List checkpoint filenames in the workspace's `checkpoints` directory,
/// sorted by modification time (oldest first).
pub fn ws_list_checkpoints(workspace: &Workspace) -> Vec<String> {
    let dir = workspace.checkpoints_dir();
    if !dir.is_dir() {
        return Vec::new();
    }

    let mut entries: Vec<(u64, String)> = std::fs::read_dir(&dir)
        .map(|rd| {
            rd.flatten()
                .filter(|e| e.path().is_file())
                .map(|e| {
                    let name = e.file_name().to_string_lossy().into_owned();
                    let mtime = e
                        .metadata()
                        .ok()
                        .and_then(|m| m.modified().ok())
                        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                        .map(|d| d.as_secs())
                        .unwrap_or(0);
                    (mtime, name)
                })
                .collect()
        })
        .unwrap_or_default();

    entries.sort_by_key(|(mtime, name)| (*mtime, name.clone()));
    entries.into_iter().map(|(_, name)| name).collect()
}

/// Remove old checkpoints, keeping only the `keep_n` most recent.
///
/// Returns the number of deleted checkpoint files.
pub fn ws_prune_checkpoints(workspace: &Workspace, keep_n: usize) -> Result<usize, WorkspaceError> {
    // Sorted oldest-first.
    let all = ws_list_checkpoints(workspace);
    let total = all.len();
    if total <= keep_n {
        return Ok(0);
    }
    let to_remove = total - keep_n;
    let dir = workspace.checkpoints_dir();
    let mut removed = 0;
    for name in all.into_iter().take(to_remove) {
        let path = dir.join(&name);
        std::fs::remove_file(&path)?;
        removed += 1;
    }
    Ok(removed)
}

/// Total size in bytes of all checkpoint files.
pub fn ws_checkpoint_size(workspace: &Workspace) -> u64 {
    let dir = workspace.checkpoints_dir();
    if !dir.is_dir() {
        return 0;
    }
    std::fs::read_dir(&dir)
        .map(|rd| {
            rd.flatten()
                .filter(|e| e.path().is_file())
                .map(|e| e.metadata().map(|m| m.len()).unwrap_or(0))
                .sum()
        })
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Name validation
// ---------------------------------------------------------------------------

/// Validate a workspace name: 1–64 chars, alphanumeric + hyphens/underscores.
pub fn ws_validate_name(name: &str) -> Result<(), WorkspaceError> {
    if name.is_empty() || name.len() > 64 {
        return Err(WorkspaceError::InvalidName(name.to_owned()));
    }
    if !name
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    {
        return Err(WorkspaceError::InvalidName(name.to_owned()));
    }
    Ok(())
}

/// Generate a timestamped workspace name from a base name.
///
/// Format: `{base}_{last_6_digits_of_unix_secs}`.
pub fn ws_timestamped_name(base: &str) -> Result<String, WorkspaceError> {
    ws_validate_name(base)?;
    let ts = ws_current_timestamp();
    let suffix = ts % 1_000_000; // last 6 digits
    let candidate = format!("{}_{:06}", base, suffix);
    ws_validate_name(&candidate)?;
    Ok(candidate)
}

// ---------------------------------------------------------------------------
// Statistics & formatting
// ---------------------------------------------------------------------------

/// Aggregated statistics for a single workspace.
pub struct WorkspaceStats {
    pub name: String,
    pub status: WorkspaceStatus,
    pub disk_usage_bytes: u64,
    pub checkpoint_count: usize,
    /// Seconds elapsed since `created_at`.
    pub age_seconds: u64,
    pub tags: Vec<String>,
}

/// Compute statistics for a workspace.
pub fn ws_compute_stats(workspace: &Workspace) -> WorkspaceStats {
    let now = ws_current_timestamp();
    let age_seconds = now.saturating_sub(workspace.config.created_at);
    WorkspaceStats {
        name: workspace.config.name.clone(),
        status: workspace.config.status.clone(),
        disk_usage_bytes: workspace.disk_usage(),
        checkpoint_count: workspace.checkpoint_count(),
        age_seconds,
        tags: workspace.config.tags.clone(),
    }
}

/// Format a single workspace as a one-line summary.
pub fn ws_format_summary(workspace: &Workspace) -> String {
    let status = workspace.detect_status();
    let ckpts = workspace.checkpoint_count();
    let usage = ws_format_bytes(workspace.disk_usage());
    format!(
        "[{}] {} | {} | {} checkpoints | {}",
        status.as_str(),
        workspace.config.name,
        workspace.config.model_type,
        ckpts,
        usage,
    )
}

/// Format a table of workspaces with a header row.
pub fn ws_format_table(workspaces: &[Workspace]) -> String {
    let header = format!(
        "{:<30} {:<12} {:<10} {:<10} {}",
        "NAME", "STATUS", "CHECKPTS", "DISK", "TAGS"
    );
    let separator = "-".repeat(header.len());
    let mut lines = vec![header, separator];
    for ws in workspaces {
        let status = ws.detect_status();
        let usage = ws_format_bytes(ws.disk_usage());
        let tags = ws.config.tags.join(",");
        lines.push(format!(
            "{:<30} {:<12} {:<10} {:<10} {}",
            ws.config.name,
            status.as_str(),
            ws.checkpoint_count(),
            usage,
            tags,
        ));
    }
    lines.join("\n")
}

/// Format workspace stats as a multi-line string.
pub fn ws_format_stats(stats: &WorkspaceStats) -> String {
    let tags = if stats.tags.is_empty() {
        "(none)".to_owned()
    } else {
        stats.tags.join(", ")
    };
    format!(
        "name:        {}\nstatus:      {}\ndisk_usage:  {}\ncheckpoints: {}\nage:         {}s\ntags:        {}",
        stats.name,
        stats.status.as_str(),
        ws_format_bytes(stats.disk_usage_bytes),
        stats.checkpoint_count,
        stats.age_seconds,
        tags,
    )
}

/// Format workspace status counts as a multi-line string.
pub fn ws_format_status_counts(counts: &WorkspaceStatusCounts) -> String {
    format!(
        "total:       {}\nnot_started: {}\nrunning:     {}\ncompleted:   {}\nfailed:      {}\narchived:    {}",
        counts.total,
        counts.not_started,
        counts.running,
        counts.completed,
        counts.failed,
        counts.archived,
    )
}

// ---------------------------------------------------------------------------
// Timestamp helper
// ---------------------------------------------------------------------------

/// Return the current Unix timestamp in whole seconds.
pub fn ws_current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Recursively sum file sizes under `path`.
fn dir_size_recursive(path: &Path) -> u64 {
    if path.is_file() {
        return path.metadata().map(|m| m.len()).unwrap_or(0);
    }
    if !path.is_dir() {
        return 0;
    }
    std::fs::read_dir(path)
        .map(|rd| rd.flatten().map(|e| dir_size_recursive(&e.path())).sum())
        .unwrap_or(0)
}

/// Human-readable byte size.
fn ws_format_bytes(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{}B", bytes)
    } else if bytes < 1024 * 1024 {
        format!("{:.1}KB", bytes as f64 / 1024.0)
    } else if bytes < 1024 * 1024 * 1024 {
        format!("{:.1}MB", bytes as f64 / (1024.0 * 1024.0))
    } else {
        format!("{:.1}GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::fs;

    // -----------------------------------------------------------------------
    // Drop-guard for temp directories
    // -----------------------------------------------------------------------

    struct TempDir {
        path: PathBuf,
    }

    impl TempDir {
        fn new(label: &str) -> Self {
            let mut p = env::temp_dir();
            p.push(format!(
                "oxigaf_ws_test_{}_{}",
                label,
                ws_current_timestamp()
            ));
            fs::create_dir_all(&p).expect("create temp dir");
            TempDir { path: p }
        }

        fn path(&self) -> &Path {
            &self.path
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    // -----------------------------------------------------------------------
    // ws_validate_name
    // -----------------------------------------------------------------------

    #[test]
    fn test_validate_name_valid_hyphen() {
        assert!(ws_validate_name("my-run").is_ok());
    }

    #[test]
    fn test_validate_name_valid_underscore() {
        assert!(ws_validate_name("run_01").is_ok());
    }

    #[test]
    fn test_validate_name_valid_alphanumeric() {
        assert!(ws_validate_name("run01abc").is_ok());
    }

    #[test]
    fn test_validate_name_valid_single_char() {
        assert!(ws_validate_name("a").is_ok());
    }

    #[test]
    fn test_validate_name_valid_max_length() {
        let name = "a".repeat(64);
        assert!(ws_validate_name(&name).is_ok());
    }

    #[test]
    fn test_validate_name_empty() {
        assert!(matches!(
            ws_validate_name(""),
            Err(WorkspaceError::InvalidName(_))
        ));
    }

    #[test]
    fn test_validate_name_too_long() {
        let name = "a".repeat(65);
        assert!(matches!(
            ws_validate_name(&name),
            Err(WorkspaceError::InvalidName(_))
        ));
    }

    #[test]
    fn test_validate_name_space() {
        assert!(matches!(
            ws_validate_name("my run"),
            Err(WorkspaceError::InvalidName(_))
        ));
    }

    #[test]
    fn test_validate_name_slash() {
        assert!(matches!(
            ws_validate_name("my/run"),
            Err(WorkspaceError::InvalidName(_))
        ));
    }

    #[test]
    fn test_validate_name_dot() {
        assert!(matches!(
            ws_validate_name("run.01"),
            Err(WorkspaceError::InvalidName(_))
        ));
    }

    // -----------------------------------------------------------------------
    // ws_timestamped_name
    // -----------------------------------------------------------------------

    #[test]
    fn test_timestamped_name_returns_valid() {
        let name = ws_timestamped_name("run").expect("timestamped name");
        assert!(ws_validate_name(&name).is_ok());
        assert!(name.starts_with("run_"));
    }

    #[test]
    fn test_timestamped_name_invalid_base() {
        assert!(ws_timestamped_name("bad name").is_err());
    }

    // -----------------------------------------------------------------------
    // WorkspaceStatus
    // -----------------------------------------------------------------------

    #[test]
    fn test_status_round_trip_not_started() {
        let s = WorkspaceStatus::NotStarted;
        assert_eq!(
            WorkspaceStatus::parse_status(s.as_str()),
            Some(WorkspaceStatus::NotStarted)
        );
    }

    #[test]
    fn test_status_round_trip_running() {
        let s = WorkspaceStatus::Running;
        assert_eq!(
            WorkspaceStatus::parse_status(s.as_str()),
            Some(WorkspaceStatus::Running)
        );
    }

    #[test]
    fn test_status_round_trip_completed() {
        let s = WorkspaceStatus::Completed;
        assert_eq!(
            WorkspaceStatus::parse_status(s.as_str()),
            Some(WorkspaceStatus::Completed)
        );
    }

    #[test]
    fn test_status_round_trip_failed() {
        let s = WorkspaceStatus::Failed;
        assert_eq!(
            WorkspaceStatus::parse_status(s.as_str()),
            Some(WorkspaceStatus::Failed)
        );
    }

    #[test]
    fn test_status_round_trip_archived() {
        let s = WorkspaceStatus::Archived;
        assert_eq!(
            WorkspaceStatus::parse_status(s.as_str()),
            Some(WorkspaceStatus::Archived)
        );
    }

    #[test]
    fn test_status_from_str_unknown() {
        assert_eq!(WorkspaceStatus::parse_status("bogus"), None);
    }

    // -----------------------------------------------------------------------
    // WorkspaceConfig
    // -----------------------------------------------------------------------

    #[test]
    fn test_config_new_valid() {
        let cfg = WorkspaceConfig::new("my-run").expect("new config");
        assert_eq!(cfg.name, "my-run");
        assert_eq!(cfg.model_type, "3dgs_avatar");
        assert_eq!(cfg.max_checkpoints_to_keep, 5);
        assert_eq!(cfg.status, WorkspaceStatus::NotStarted);
    }

    #[test]
    fn test_config_new_invalid_name() {
        assert!(WorkspaceConfig::new("bad name").is_err());
    }

    #[test]
    fn test_config_round_trip_basic() {
        let mut cfg = WorkspaceConfig::new("round-trip").expect("new config");
        cfg.description = "A test run".to_owned();
        cfg.tags = vec!["tag1".to_owned(), "tag2".to_owned()];
        cfg.model_type = "custom_model".to_owned();
        cfg.max_checkpoints_to_keep = 10;

        let s = cfg.to_string();
        let parsed = WorkspaceConfig::from_str(&s).expect("parse config");

        assert_eq!(parsed.name, cfg.name);
        assert_eq!(parsed.description, cfg.description);
        assert_eq!(parsed.tags, cfg.tags);
        assert_eq!(parsed.created_at, cfg.created_at);
        assert_eq!(parsed.modified_at, cfg.modified_at);
        assert_eq!(parsed.model_type, cfg.model_type);
        assert_eq!(parsed.max_checkpoints_to_keep, cfg.max_checkpoints_to_keep);
        assert_eq!(parsed.status, cfg.status);
    }

    #[test]
    fn test_config_round_trip_custom_fields() {
        let mut cfg = WorkspaceConfig::new("custom-ws").expect("new config");
        cfg.custom_fields = vec![
            ("lr".to_owned(), "0.001".to_owned()),
            ("batch_size".to_owned(), "8".to_owned()),
        ];
        let s = cfg.to_string();
        let parsed = WorkspaceConfig::from_str(&s).expect("parse config");
        assert_eq!(parsed.custom_fields.len(), 2);
        assert!(parsed
            .custom_fields
            .contains(&("lr".to_owned(), "0.001".to_owned())));
    }

    #[test]
    fn test_config_from_str_missing_optional_fields_get_defaults() {
        // Only provide name; everything else should be default.
        let s = "name=minimal-run\n";
        let cfg = WorkspaceConfig::from_str(s).expect("parse minimal config");
        assert_eq!(cfg.name, "minimal-run");
        assert_eq!(cfg.model_type, "3dgs_avatar");
        assert_eq!(cfg.max_checkpoints_to_keep, 5);
        assert_eq!(cfg.status, WorkspaceStatus::NotStarted);
        assert!(cfg.tags.is_empty());
    }

    #[test]
    fn test_config_from_str_missing_name_is_error() {
        let s = "description=no name here\n";
        assert!(WorkspaceConfig::from_str(s).is_err());
    }

    #[test]
    fn test_config_from_str_ignores_comments() {
        let s = "# comment line\nname=commented-ws\n# another comment\n";
        let cfg = WorkspaceConfig::from_str(s).expect("parse config with comments");
        assert_eq!(cfg.name, "commented-ws");
    }

    #[test]
    fn test_config_from_str_empty_tags() {
        let s = "name=empty-tags\ntags=\n";
        let cfg = WorkspaceConfig::from_str(s).expect("parse config");
        assert!(cfg.tags.is_empty());
    }

    #[test]
    fn test_config_status_serialization() {
        let mut cfg = WorkspaceConfig::new("status-ws").expect("new config");
        cfg.status = WorkspaceStatus::Completed;
        let s = cfg.to_string();
        let parsed = WorkspaceConfig::from_str(&s).expect("parse config");
        assert_eq!(parsed.status, WorkspaceStatus::Completed);
    }

    // -----------------------------------------------------------------------
    // WorkspaceManager::create / load
    // -----------------------------------------------------------------------

    #[test]
    fn test_create_creates_directory_structure() {
        let td = TempDir::new("create");
        let mgr = WorkspaceManager::new(td.path());
        let ws = mgr.create("my-workspace").expect("create workspace");

        assert!(ws.root.is_dir());
        assert!(ws.checkpoints_dir().is_dir());
        assert!(ws.logs_dir().is_dir());
        assert!(ws.renders_dir().is_dir());
        assert!(ws.exports_dir().is_dir());
        assert!(ws.config_path().is_file());
    }

    #[test]
    fn test_create_duplicate_name_returns_already_exists() {
        let td = TempDir::new("dup");
        let mgr = WorkspaceManager::new(td.path());
        mgr.create("dup-ws").expect("first create");
        let result = mgr.create("dup-ws");
        assert!(matches!(result, Err(WorkspaceError::AlreadyExists(_))));
    }

    #[test]
    fn test_load_after_create() {
        let td = TempDir::new("load");
        let mgr = WorkspaceManager::new(td.path());
        mgr.create("load-ws").expect("create");
        let ws = mgr.load("load-ws").expect("load");
        assert_eq!(ws.config.name, "load-ws");
    }

    #[test]
    fn test_load_missing_returns_not_found() {
        let td = TempDir::new("notfound");
        let mgr = WorkspaceManager::new(td.path());
        let result = mgr.load("missing");
        assert!(matches!(result, Err(WorkspaceError::NotFound(_))));
    }

    // -----------------------------------------------------------------------
    // WorkspaceManager::list
    // -----------------------------------------------------------------------

    #[test]
    fn test_list_empty_dir_returns_empty() {
        let td = TempDir::new("empty");
        let mgr = WorkspaceManager::new(td.path());
        let list = mgr.list().expect("list");
        assert!(list.is_empty());
    }

    #[test]
    fn test_list_multiple_workspaces() {
        let td = TempDir::new("multi");
        let mgr = WorkspaceManager::new(td.path());
        mgr.create("ws-a").expect("create ws-a");
        mgr.create("ws-b").expect("create ws-b");
        mgr.create("ws-c").expect("create ws-c");
        let list = mgr.list().expect("list");
        assert_eq!(list.len(), 3);
        let names: Vec<&str> = list.iter().map(|w| w.config.name.as_str()).collect();
        assert!(names.contains(&"ws-a"));
        assert!(names.contains(&"ws-b"));
        assert!(names.contains(&"ws-c"));
    }

    // -----------------------------------------------------------------------
    // WorkspaceManager::list_by_status
    // -----------------------------------------------------------------------

    #[test]
    fn test_list_by_status_filters_correctly() {
        let td = TempDir::new("bystatus");
        let mgr = WorkspaceManager::new(td.path());

        mgr.create("ws-not-started").expect("create");
        let mut ws_done = mgr.create("ws-done").expect("create");
        ws_done
            .set_status(WorkspaceStatus::Completed)
            .expect("set status");

        let completed = mgr
            .list_by_status(&WorkspaceStatus::Completed)
            .expect("list");
        assert_eq!(completed.len(), 1);
        assert_eq!(completed[0].config.name, "ws-done");

        let not_started = mgr
            .list_by_status(&WorkspaceStatus::NotStarted)
            .expect("list");
        assert_eq!(not_started.len(), 1);
        assert_eq!(not_started[0].config.name, "ws-not-started");
    }

    // -----------------------------------------------------------------------
    // WorkspaceManager::list_by_tag
    // -----------------------------------------------------------------------

    #[test]
    fn test_list_by_tag_filters_correctly() {
        let td = TempDir::new("bytag");
        let mgr = WorkspaceManager::new(td.path());

        let mut cfg_a = WorkspaceConfig::new("tagged-ws").expect("new config");
        cfg_a.tags = vec!["production".to_owned()];
        mgr.create_with_config(cfg_a).expect("create");

        let mut cfg_b = WorkspaceConfig::new("untagged-ws").expect("new config");
        cfg_b.tags = vec!["dev".to_owned()];
        mgr.create_with_config(cfg_b).expect("create");

        let found = mgr.list_by_tag("production").expect("list by tag");
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].config.name, "tagged-ws");

        let empty = mgr.list_by_tag("nonexistent").expect("list by tag");
        assert!(empty.is_empty());
    }

    // -----------------------------------------------------------------------
    // WorkspaceManager::delete
    // -----------------------------------------------------------------------

    #[test]
    fn test_delete_removes_directory() {
        let td = TempDir::new("delete");
        let mgr = WorkspaceManager::new(td.path());
        let ws = mgr.create("del-ws").expect("create");
        let ws_root = ws.root.clone();
        mgr.delete("del-ws").expect("delete");
        assert!(!ws_root.exists());
    }

    #[test]
    fn test_delete_running_workspace_returns_error() {
        let td = TempDir::new("delrun");
        let mgr = WorkspaceManager::new(td.path());
        let mut ws = mgr.create("running-ws").expect("create");
        ws.set_status(WorkspaceStatus::Running)
            .expect("set running");
        let result = mgr.delete("running-ws");
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // WorkspaceManager::rename
    // -----------------------------------------------------------------------

    #[test]
    fn test_rename_succeeds() {
        let td = TempDir::new("rename");
        let mgr = WorkspaceManager::new(td.path());
        mgr.create("old-name").expect("create");
        let renamed = mgr.rename("old-name", "new-name").expect("rename");
        assert_eq!(renamed.config.name, "new-name");
        assert!(!td.path().join("old-name").exists());
        assert!(td.path().join("new-name").is_dir());
    }

    #[test]
    fn test_rename_missing_old_returns_not_found() {
        let td = TempDir::new("renamenotfound");
        let mgr = WorkspaceManager::new(td.path());
        let result = mgr.rename("nonexistent", "new-name");
        assert!(matches!(result, Err(WorkspaceError::NotFound(_))));
    }

    #[test]
    fn test_rename_to_existing_name_returns_already_exists() {
        let td = TempDir::new("renameconflict");
        let mgr = WorkspaceManager::new(td.path());
        mgr.create("ws-one").expect("create one");
        mgr.create("ws-two").expect("create two");
        let result = mgr.rename("ws-one", "ws-two");
        assert!(matches!(result, Err(WorkspaceError::AlreadyExists(_))));
    }

    // -----------------------------------------------------------------------
    // WorkspaceManager::exists
    // -----------------------------------------------------------------------

    #[test]
    fn test_exists_true_after_create() {
        let td = TempDir::new("exists");
        let mgr = WorkspaceManager::new(td.path());
        mgr.create("exists-ws").expect("create");
        assert!(mgr.exists("exists-ws"));
    }

    #[test]
    fn test_exists_false_for_missing() {
        let td = TempDir::new("existsmissing");
        let mgr = WorkspaceManager::new(td.path());
        assert!(!mgr.exists("ghost"));
    }

    // -----------------------------------------------------------------------
    // WorkspaceManager::status_counts
    // -----------------------------------------------------------------------

    #[test]
    fn test_status_counts_correct() {
        let td = TempDir::new("counts");
        let mgr = WorkspaceManager::new(td.path());
        mgr.create("ws1").expect("create");
        let mut ws2 = mgr.create("ws2").expect("create");
        ws2.set_status(WorkspaceStatus::Completed).expect("set");
        let mut ws3 = mgr.create("ws3").expect("create");
        ws3.set_status(WorkspaceStatus::Failed).expect("set");

        let counts = mgr.status_counts().expect("counts");
        assert_eq!(counts.total, 3);
        assert_eq!(counts.not_started, 1);
        assert_eq!(counts.completed, 1);
        assert_eq!(counts.failed, 1);
        assert_eq!(counts.running, 0);
        assert_eq!(counts.archived, 0);
    }

    // -----------------------------------------------------------------------
    // Workspace::detect_status
    // -----------------------------------------------------------------------

    #[test]
    fn test_detect_status_not_started() {
        let td = TempDir::new("detectns");
        let mgr = WorkspaceManager::new(td.path());
        let ws = mgr.create("detect-ns").expect("create");
        assert_eq!(ws.detect_status(), WorkspaceStatus::NotStarted);
    }

    #[test]
    fn test_detect_status_running() {
        let td = TempDir::new("detectrun");
        let mgr = WorkspaceManager::new(td.path());
        let ws = mgr.create("detect-run").expect("create");
        fs::write(ws.root.join(Workspace::LOCK_FILE), b"").expect("write lock");
        assert_eq!(ws.detect_status(), WorkspaceStatus::Running);
    }

    #[test]
    fn test_detect_status_completed() {
        let td = TempDir::new("detectdone");
        let mgr = WorkspaceManager::new(td.path());
        let ws = mgr.create("detect-done").expect("create");
        fs::write(ws.root.join(Workspace::DONE_FILE), b"").expect("write done");
        assert_eq!(ws.detect_status(), WorkspaceStatus::Completed);
    }

    #[test]
    fn test_detect_status_failed() {
        let td = TempDir::new("detectfail");
        let mgr = WorkspaceManager::new(td.path());
        let ws = mgr.create("detect-fail").expect("create");
        fs::write(ws.root.join(Workspace::FAILED_FILE), b"").expect("write failed");
        assert_eq!(ws.detect_status(), WorkspaceStatus::Failed);
    }

    #[test]
    fn test_detect_status_archived() {
        let td = TempDir::new("detectarch");
        let mgr = WorkspaceManager::new(td.path());
        let ws = mgr.create("detect-arch").expect("create");
        fs::write(ws.root.join(Workspace::ARCHIVE_FILE), b"").expect("write archive");
        assert_eq!(ws.detect_status(), WorkspaceStatus::Archived);
    }

    #[test]
    fn test_detect_status_archived_priority_over_failed() {
        let td = TempDir::new("detectprio");
        let mgr = WorkspaceManager::new(td.path());
        let ws = mgr.create("detect-prio").expect("create");
        // Write both failed and archived markers — archived should win.
        fs::write(ws.root.join(Workspace::FAILED_FILE), b"").expect("write failed");
        fs::write(ws.root.join(Workspace::ARCHIVE_FILE), b"").expect("write archive");
        assert_eq!(ws.detect_status(), WorkspaceStatus::Archived);
    }

    // -----------------------------------------------------------------------
    // Workspace::set_status
    // -----------------------------------------------------------------------

    #[test]
    fn test_set_status_running_creates_lock_file() {
        let td = TempDir::new("setrun");
        let mgr = WorkspaceManager::new(td.path());
        let mut ws = mgr.create("set-run").expect("create");
        ws.set_status(WorkspaceStatus::Running)
            .expect("set running");
        assert!(ws.root.join(Workspace::LOCK_FILE).exists());
    }

    #[test]
    fn test_set_status_completed_creates_done_file() {
        let td = TempDir::new("setdone");
        let mgr = WorkspaceManager::new(td.path());
        let mut ws = mgr.create("set-done").expect("create");
        ws.set_status(WorkspaceStatus::Completed).expect("set done");
        assert!(ws.root.join(Workspace::DONE_FILE).exists());
        assert!(!ws.root.join(Workspace::LOCK_FILE).exists());
    }

    #[test]
    fn test_set_status_cleans_up_old_markers() {
        let td = TempDir::new("setclean");
        let mgr = WorkspaceManager::new(td.path());
        let mut ws = mgr.create("set-clean").expect("create");
        // First set running.
        ws.set_status(WorkspaceStatus::Running)
            .expect("set running");
        assert!(ws.root.join(Workspace::LOCK_FILE).exists());
        // Then set completed — lock file must be gone.
        ws.set_status(WorkspaceStatus::Completed)
            .expect("set completed");
        assert!(!ws.root.join(Workspace::LOCK_FILE).exists());
        assert!(ws.root.join(Workspace::DONE_FILE).exists());
    }

    #[test]
    fn test_set_status_not_started_removes_all_markers() {
        let td = TempDir::new("setns");
        let mgr = WorkspaceManager::new(td.path());
        let mut ws = mgr.create("set-ns").expect("create");
        ws.set_status(WorkspaceStatus::Running)
            .expect("set running");
        ws.set_status(WorkspaceStatus::NotStarted)
            .expect("set not started");
        assert!(!ws.root.join(Workspace::LOCK_FILE).exists());
        assert!(!ws.root.join(Workspace::DONE_FILE).exists());
    }

    // -----------------------------------------------------------------------
    // Workspace::checkpoint_count
    // -----------------------------------------------------------------------

    #[test]
    fn test_checkpoint_count_zero_initially() {
        let td = TempDir::new("ckpt0");
        let mgr = WorkspaceManager::new(td.path());
        let ws = mgr.create("ckpt-zero").expect("create");
        assert_eq!(ws.checkpoint_count(), 0);
    }

    #[test]
    fn test_checkpoint_count_with_files() {
        let td = TempDir::new("ckptfiles");
        let mgr = WorkspaceManager::new(td.path());
        let ws = mgr.create("ckpt-files").expect("create");
        fs::write(ws.checkpoints_dir().join("step_1000.ckpt"), b"data").expect("write ckpt");
        fs::write(ws.checkpoints_dir().join("step_2000.ckpt"), b"data").expect("write ckpt");
        assert_eq!(ws.checkpoint_count(), 2);
    }

    #[test]
    fn test_checkpoint_count_five() {
        let td = TempDir::new("ckpt5");
        let mgr = WorkspaceManager::new(td.path());
        let ws = mgr.create("ckpt-five").expect("create");
        for i in 0..5 {
            fs::write(
                ws.checkpoints_dir().join(format!("step_{}.ckpt", i * 1000)),
                b"d",
            )
            .expect("write ckpt");
        }
        assert_eq!(ws.checkpoint_count(), 5);
    }

    // -----------------------------------------------------------------------
    // Workspace::disk_usage
    // -----------------------------------------------------------------------

    #[test]
    fn test_disk_usage_nonzero_after_creation() {
        let td = TempDir::new("diskusage");
        let mgr = WorkspaceManager::new(td.path());
        let ws = mgr.create("disk-ws").expect("create");
        // Config file exists, so disk usage > 0.
        assert!(ws.disk_usage() > 0);
    }

    // -----------------------------------------------------------------------
    // Workspace::validate
    // -----------------------------------------------------------------------

    #[test]
    fn test_validate_passes_for_valid_workspace() {
        let td = TempDir::new("valid");
        let mgr = WorkspaceManager::new(td.path());
        let ws = mgr.create("valid-ws").expect("create");
        assert!(ws.validate().is_ok());
    }

    #[test]
    fn test_validate_fails_when_subdir_missing() {
        let td = TempDir::new("invalidsub");
        let mgr = WorkspaceManager::new(td.path());
        let ws = mgr.create("invalid-sub").expect("create");
        // Remove one required subdirectory.
        fs::remove_dir_all(ws.checkpoints_dir()).expect("remove dir");
        assert!(ws.validate().is_err());
    }

    // -----------------------------------------------------------------------
    // Workspace::reload_config
    // -----------------------------------------------------------------------

    #[test]
    fn test_reload_config_picks_up_changes() {
        let td = TempDir::new("reload");
        let mgr = WorkspaceManager::new(td.path());
        let mut ws = mgr.create("reload-ws").expect("create");

        // Modify config on disk directly.
        let mut cfg = ws.config.clone();
        cfg.description = "updated description".to_owned();
        fs::write(ws.config_path(), cfg.to_string()).expect("write config");

        ws.reload_config().expect("reload");
        assert_eq!(ws.config.description, "updated description");
    }

    // -----------------------------------------------------------------------
    // ws_list_checkpoints
    // -----------------------------------------------------------------------

    #[test]
    fn test_list_checkpoints_empty() {
        let td = TempDir::new("lstckpt0");
        let mgr = WorkspaceManager::new(td.path());
        let ws = mgr.create("lst-empty").expect("create");
        let list = ws_list_checkpoints(&ws);
        assert!(list.is_empty());
    }

    #[test]
    fn test_list_checkpoints_with_files() {
        let td = TempDir::new("lstckptf");
        let mgr = WorkspaceManager::new(td.path());
        let ws = mgr.create("lst-files").expect("create");
        fs::write(ws.checkpoints_dir().join("step_100.ckpt"), b"a").expect("write");
        fs::write(ws.checkpoints_dir().join("step_200.ckpt"), b"b").expect("write");
        let list = ws_list_checkpoints(&ws);
        assert_eq!(list.len(), 2);
    }

    // -----------------------------------------------------------------------
    // ws_prune_checkpoints
    // -----------------------------------------------------------------------

    #[test]
    fn test_prune_keeps_correct_number() {
        let td = TempDir::new("prune");
        let mgr = WorkspaceManager::new(td.path());
        let ws = mgr.create("prune-ws").expect("create");
        for i in 0..5u32 {
            fs::write(
                ws.checkpoints_dir().join(format!("step_{:04}.ckpt", i)),
                b"d",
            )
            .expect("write ckpt");
        }
        let removed = ws_prune_checkpoints(&ws, 3).expect("prune");
        assert_eq!(removed, 2);
        assert_eq!(ws.checkpoint_count(), 3);
    }

    #[test]
    fn test_prune_no_op_when_under_limit() {
        let td = TempDir::new("pruneno");
        let mgr = WorkspaceManager::new(td.path());
        let ws = mgr.create("prune-noop").expect("create");
        fs::write(ws.checkpoints_dir().join("step_100.ckpt"), b"d").expect("write ckpt");
        let removed = ws_prune_checkpoints(&ws, 5).expect("prune");
        assert_eq!(removed, 0);
        assert_eq!(ws.checkpoint_count(), 1);
    }

    // -----------------------------------------------------------------------
    // ws_checkpoint_size
    // -----------------------------------------------------------------------

    #[test]
    fn test_checkpoint_size_zero_for_empty_dir() {
        let td = TempDir::new("ckptsize0");
        let mgr = WorkspaceManager::new(td.path());
        let ws = mgr.create("ckpt-size-zero").expect("create");
        assert_eq!(ws_checkpoint_size(&ws), 0);
    }

    #[test]
    fn test_checkpoint_size_nonzero_with_files() {
        let td = TempDir::new("ckptsizef");
        let mgr = WorkspaceManager::new(td.path());
        let ws = mgr.create("ckpt-size-files").expect("create");
        fs::write(ws.checkpoints_dir().join("step_100.ckpt"), b"hello").expect("write");
        assert_eq!(ws_checkpoint_size(&ws), 5);
    }

    // -----------------------------------------------------------------------
    // ws_compute_stats
    // -----------------------------------------------------------------------

    #[test]
    fn test_compute_stats_fields_present() {
        let td = TempDir::new("stats");
        let mgr = WorkspaceManager::new(td.path());
        let ws = mgr.create("stats-ws").expect("create");
        let stats = ws_compute_stats(&ws);
        assert_eq!(stats.name, "stats-ws");
        assert_eq!(stats.status, WorkspaceStatus::NotStarted);
        assert_eq!(stats.checkpoint_count, 0);
        assert!(stats.disk_usage_bytes > 0);
    }

    // -----------------------------------------------------------------------
    // Formatting functions
    // -----------------------------------------------------------------------

    #[test]
    fn test_format_summary_contains_name() {
        let td = TempDir::new("fmtsummary");
        let mgr = WorkspaceManager::new(td.path());
        let ws = mgr.create("fmt-ws").expect("create");
        let summary = ws_format_summary(&ws);
        assert!(!summary.is_empty());
        assert!(summary.contains("fmt-ws"));
    }

    #[test]
    fn test_format_table_contains_header_and_names() {
        let td = TempDir::new("fmttable");
        let mgr = WorkspaceManager::new(td.path());
        mgr.create("table-ws-a").expect("create a");
        mgr.create("table-ws-b").expect("create b");
        let workspaces = mgr.list().expect("list");
        let table = ws_format_table(&workspaces);
        assert!(table.contains("NAME"));
        assert!(table.contains("table-ws-a"));
        assert!(table.contains("table-ws-b"));
    }

    #[test]
    fn test_format_stats_nonempty() {
        let td = TempDir::new("fmtstats");
        let mgr = WorkspaceManager::new(td.path());
        let ws = mgr.create("fmtstats-ws").expect("create");
        let stats = ws_compute_stats(&ws);
        let s = ws_format_stats(&stats);
        assert!(!s.is_empty());
        assert!(s.contains("fmtstats-ws"));
    }

    #[test]
    fn test_format_status_counts_nonempty() {
        let td = TempDir::new("fmtcounts");
        let mgr = WorkspaceManager::new(td.path());
        mgr.create("cnt-ws").expect("create");
        let counts = mgr.status_counts().expect("counts");
        let s = ws_format_status_counts(&counts);
        assert!(!s.is_empty());
        assert!(s.contains("total"));
    }

    // -----------------------------------------------------------------------
    // ws_current_timestamp
    // -----------------------------------------------------------------------

    #[test]
    fn test_current_timestamp_positive() {
        let ts = ws_current_timestamp();
        assert!(ts > 0);
    }

    // -----------------------------------------------------------------------
    // Full lifecycle test
    // -----------------------------------------------------------------------

    #[test]
    fn test_full_lifecycle() {
        let td = TempDir::new("lifecycle");
        let mgr = WorkspaceManager::new(td.path());

        // 1. Create.
        let mut ws = mgr.create("lifecycle-ws").expect("create");
        assert_eq!(ws.detect_status(), WorkspaceStatus::NotStarted);
        assert!(mgr.exists("lifecycle-ws"));

        // 2. Transition to running.
        ws.set_status(WorkspaceStatus::Running)
            .expect("set running");
        assert_eq!(ws.detect_status(), WorkspaceStatus::Running);

        // Add a checkpoint.
        fs::write(ws.checkpoints_dir().join("step_1000.ckpt"), b"weights").expect("write ckpt");
        assert_eq!(ws.checkpoint_count(), 1);

        // 3. Complete training.
        ws.set_status(WorkspaceStatus::Completed)
            .expect("set completed");
        assert_eq!(ws.detect_status(), WorkspaceStatus::Completed);
        assert!(!ws.root.join(Workspace::LOCK_FILE).exists());

        // 4. List — should appear in listing.
        let list = mgr.list().expect("list");
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].config.name, "lifecycle-ws");

        // 5. Delete should fail on running (already completed, so it must succeed).
        mgr.delete("lifecycle-ws").expect("delete completed ws");
        assert!(!mgr.exists("lifecycle-ws"));
    }
}
