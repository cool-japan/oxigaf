//! Checkpoint management system for OxiGAF training runs.
//!
//! Provides rotation policies, best-checkpoint tracking, and resume support.
//!
//! # Quick Start
//!
//! ```rust
//! use oxigaf_trainer::checkpoint_manager::{CheckpointManager, CheckpointPolicy, CheckpointRecord};
//! use std::path::PathBuf;
//!
//! let dir = std::env::temp_dir().join("ckpt_quickstart");
//! let policy = CheckpointPolicy::default();
//! let mut mgr = CheckpointManager::new(dir.clone(), "run", policy)
//!     .expect("failed to create manager");
//!
//! let record = CheckpointRecord::new(1000, dir.join("run_iter_00001000.json"), 25.3, 0.87, 0.05, 5000, 1024);
//! let _to_delete = mgr.add_checkpoint(record);
//! println!("{}", mgr.format_summary());
//! ```

use std::fmt::Write as FmtWrite;
use std::path::PathBuf;

use crate::TrainerError;

// ---------------------------------------------------------------------------
// CheckpointRecord
// ---------------------------------------------------------------------------

/// Record of a saved checkpoint.
#[derive(Debug, Clone)]
pub struct CheckpointRecord {
    /// Training iteration this checkpoint was saved at.
    pub iteration: u32,
    /// Path to the checkpoint file on disk.
    pub path: PathBuf,
    /// Peak signal-to-noise ratio (higher = better).
    pub psnr: f32,
    /// Structural similarity index (higher = better, range \[0,1\]).
    pub ssim: f32,
    /// Training loss at this checkpoint.
    pub loss: f32,
    /// Number of Gaussians in the model at this checkpoint.
    pub num_gaussians: u32,
    /// Unix timestamp (seconds) when this checkpoint was written.  Use 0 in tests.
    pub saved_at_unix_secs: u64,
    /// Size of the checkpoint file in bytes.
    pub size_bytes: u64,
    /// Whether this is the best checkpoint seen so far (highest PSNR).
    pub is_best: bool,
}

impl CheckpointRecord {
    /// Construct a new record.  `saved_at_unix_secs` is set to 0; call the
    /// field directly if you need a real timestamp.
    pub fn new(
        iteration: u32,
        path: PathBuf,
        psnr: f32,
        ssim: f32,
        loss: f32,
        num_gaussians: u32,
        size_bytes: u64,
    ) -> Self {
        Self {
            iteration,
            path,
            psnr,
            ssim,
            loss,
            num_gaussians,
            saved_at_unix_secs: 0,
            size_bytes,
            is_best: false,
        }
    }

    /// Returns `true` if this checkpoint has a higher PSNR than `other`.
    pub fn is_better_than(&self, other: &CheckpointRecord) -> bool {
        self.psnr > other.psnr
    }

    /// Format a human-readable single-line summary of this record.
    ///
    /// Example output:
    /// `iter=1000 psnr=25.3 ssim=0.87 gaussians=5000 is_best=true`
    pub fn format_line(&self) -> String {
        format!(
            "iter={} psnr={:.1} ssim={:.2} gaussians={} is_best={}",
            self.iteration, self.psnr, self.ssim, self.num_gaussians, self.is_best
        )
    }
}

// ---------------------------------------------------------------------------
// CheckpointPolicy
// ---------------------------------------------------------------------------

/// Policy controlling which checkpoints to keep on disk.
pub struct CheckpointPolicy {
    /// Always keep the last N checkpoints (by iteration order).
    pub keep_last_n: usize,
    /// Always keep the best N checkpoints (by PSNR).
    pub keep_best_n: usize,
    /// Always keep any checkpoint whose iteration is divisible by this value.
    pub keep_every_n: u32,
    /// Skip checkpoints whose PSNR is below this threshold.
    pub min_psnr_to_save: f32,
}

impl Default for CheckpointPolicy {
    fn default() -> Self {
        Self {
            keep_last_n: 5,
            keep_best_n: 3,
            keep_every_n: 10_000,
            min_psnr_to_save: 0.0,
        }
    }
}

impl CheckpointPolicy {
    /// Aggressive rotation: keep very few checkpoints to conserve disk space.
    pub fn aggressive() -> Self {
        Self {
            keep_last_n: 3,
            keep_best_n: 1,
            keep_every_n: 50_000,
            min_psnr_to_save: 0.0,
        }
    }

    /// Conservative rotation: keep many checkpoints for safety.
    pub fn conservative() -> Self {
        Self {
            keep_last_n: 10,
            keep_best_n: 5,
            keep_every_n: 5_000,
            min_psnr_to_save: 0.0,
        }
    }

    /// Validate that policy parameters are internally consistent.
    ///
    /// Returns [`TrainerError::InvalidConfig`] if any constraint is violated.
    pub fn validate(&self) -> Result<(), TrainerError> {
        if self.keep_last_n == 0 {
            return Err(TrainerError::InvalidConfig(
                "keep_last_n must be > 0".to_string(),
            ));
        }
        if self.keep_best_n == 0 {
            return Err(TrainerError::InvalidConfig(
                "keep_best_n must be > 0".to_string(),
            ));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// CheckpointManager
// ---------------------------------------------------------------------------

/// Manages checkpoint rotation, best-checkpoint tracking, and resume.
pub struct CheckpointManager {
    /// Rotation and save policy.
    pub policy: CheckpointPolicy,
    /// Directory where checkpoint files are stored.
    pub checkpoint_dir: PathBuf,
    /// Prefix prepended to checkpoint file names.
    pub prefix: String,
    records: Vec<CheckpointRecord>,
    best_record: Option<CheckpointRecord>,
}

impl CheckpointManager {
    /// Create a new manager.  Creates `checkpoint_dir` if it does not exist.
    pub fn new(
        checkpoint_dir: PathBuf,
        prefix: &str,
        policy: CheckpointPolicy,
    ) -> Result<Self, TrainerError> {
        std::fs::create_dir_all(&checkpoint_dir)?;
        Ok(Self {
            policy,
            checkpoint_dir,
            prefix: prefix.to_string(),
            records: Vec::new(),
            best_record: None,
        })
    }

    /// Add a checkpoint record and apply the rotation policy.
    ///
    /// Returns the list of file paths that should be deleted by the caller.
    /// The internal record list is also pruned to match (deleted records are
    /// removed so future rotation decisions are accurate).
    pub fn add_checkpoint(&mut self, mut record: CheckpointRecord) -> Vec<PathBuf> {
        // Update best tracking.
        let this_is_best = match &self.best_record {
            None => true,
            Some(prev) => record.is_better_than(prev),
        };
        if this_is_best {
            // Clear is_best flag on any existing record in the vec.
            for r in self.records.iter_mut() {
                r.is_best = false;
            }
            record.is_best = true;
            self.best_record = Some(record.clone());
        }

        self.records.push(record);

        // Determine which records to keep.
        let keep_set = self.compute_keep_set();

        // Collect paths to delete and remove matching records.
        let mut to_delete: Vec<PathBuf> = Vec::new();
        let mut kept: Vec<CheckpointRecord> = Vec::new();
        for r in self.records.drain(..) {
            if keep_set.contains(&r.iteration) {
                kept.push(r);
            } else {
                to_delete.push(r.path.clone());
            }
        }
        self.records = kept;

        // Sync best_record with the stored records so it always points to
        // one of the surviving entries (or None if all were purged).
        self.best_record = self.records.iter().rfind(|r| r.is_best).cloned();

        to_delete
    }

    // -----------------------------------------------------------------------
    // Queries
    // -----------------------------------------------------------------------

    /// Return the best checkpoint record (highest PSNR seen so far).
    pub fn best(&self) -> Option<&CheckpointRecord> {
        self.best_record.as_ref()
    }

    /// Return the most recently added checkpoint (highest iteration number).
    pub fn latest(&self) -> Option<&CheckpointRecord> {
        self.records.iter().max_by_key(|r| r.iteration)
    }

    /// Return all surviving checkpoint records.
    pub fn all_records(&self) -> &[CheckpointRecord] {
        &self.records
    }

    /// Return the checkpoint to resume from (the one with the highest
    /// iteration number among all surviving records).
    pub fn find_resume_checkpoint(&self) -> Option<&CheckpointRecord> {
        self.latest()
    }

    /// Build the canonical path for a checkpoint at the given iteration.
    ///
    /// Format: `{checkpoint_dir}/{prefix}_iter_{iteration:08}.json`
    pub fn checkpoint_path_for(&self, iteration: u32) -> PathBuf {
        self.checkpoint_dir
            .join(format!("{}_iter_{:08}.json", self.prefix, iteration))
    }

    /// Return `true` if the current PSNR meets the minimum threshold set by
    /// the policy.  The rotation policy handles which checkpoints survive;
    /// this is the gate for whether a checkpoint is even worth writing.
    pub fn should_save(&self, _iteration: u32, psnr: f32) -> bool {
        psnr >= self.policy.min_psnr_to_save
    }

    /// Generate a compact multi-line summary of the current checkpoint state.
    pub fn format_summary(&self) -> String {
        let mut out = String::new();
        let total = self.records.len();
        let best_psnr = self
            .best_record
            .as_ref()
            .map(|r| format!("{:.2}", r.psnr))
            .unwrap_or_else(|| "none".to_string());
        let latest_iter = self
            .latest()
            .map(|r| r.iteration.to_string())
            .unwrap_or_else(|| "none".to_string());
        let disk_bytes: u64 = self.records.iter().map(|r| r.size_bytes).sum();
        let disk_kb = disk_bytes / 1024;

        let _ = writeln!(out, "CheckpointManager summary:");
        let _ = writeln!(out, "  total checkpoints : {total}");
        let _ = writeln!(out, "  best PSNR         : {best_psnr}");
        let _ = writeln!(out, "  latest iteration  : {latest_iter}");
        let _ = writeln!(out, "  disk usage (est.) : {disk_kb} KB");
        out
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Build the set of iterations that should be retained according to the
    /// current policy.
    fn compute_keep_set(&self) -> std::collections::HashSet<u32> {
        let mut keep = std::collections::HashSet::new();

        // 1. Keep last N (by insertion order; records are pushed in order).
        let last_n = self.records.len().min(self.policy.keep_last_n);
        for r in self.records.iter().rev().take(last_n) {
            keep.insert(r.iteration);
        }

        // 2. Keep best N by PSNR.
        let mut by_psnr: Vec<&CheckpointRecord> = self.records.iter().collect();
        by_psnr.sort_by(|a, b| {
            b.psnr
                .partial_cmp(&a.psnr)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for r in by_psnr.iter().take(self.policy.keep_best_n) {
            keep.insert(r.iteration);
        }

        // 3. Keep periodic checkpoints (iteration % keep_every_n == 0),
        //    excluding iteration 0 to avoid always-keep-first behaviour.
        for r in &self.records {
            if r.iteration > 0
                && self.policy.keep_every_n > 0
                && r.iteration % self.policy.keep_every_n == 0
            {
                keep.insert(r.iteration);
            }
        }

        keep
    }
}

// ---------------------------------------------------------------------------
// CheckpointIndex — optional JSON persistence
// ---------------------------------------------------------------------------

/// Serialisable index of all checkpoint records for a run.
pub struct CheckpointIndex {
    /// Prefix used by the owning [`CheckpointManager`].
    pub prefix: String,
    /// Snapshot of all checkpoint records at time of saving.
    pub records: Vec<CheckpointRecord>,
}

impl CheckpointIndex {
    /// Serialise to a simple JSON string (hand-written, no serde).
    ///
    /// Output format is a JSON object containing a `prefix` string and a
    /// `records` array of record objects.
    pub fn to_json(&self) -> String {
        let mut out = String::from("{\n");
        let _ = writeln!(
            out,
            "  \"prefix\": \"{}\",",
            escape_json_string(&self.prefix)
        );
        out.push_str("  \"records\": [\n");
        for (i, r) in self.records.iter().enumerate() {
            let comma = if i + 1 < self.records.len() { "," } else { "" };
            let _ = writeln!(
                out,
                "    {{\"iteration\":{},\"path\":\"{}\",\"psnr\":{},\"ssim\":{},\"loss\":{},\"num_gaussians\":{},\"saved_at_unix_secs\":{},\"size_bytes\":{},\"is_best\":{}}}{}",
                r.iteration,
                escape_json_string(&r.path.to_string_lossy()),
                r.psnr,
                r.ssim,
                r.loss,
                r.num_gaussians,
                r.saved_at_unix_secs,
                r.size_bytes,
                r.is_best,
                comma
            );
        }
        out.push_str("  ]\n}");
        out
    }

    /// Parse a JSON string previously produced by `to_json`.
    pub fn from_json(s: &str) -> Result<Self, TrainerError> {
        // Extract prefix.
        let prefix = extract_json_string_field(s, "prefix").ok_or_else(|| {
            TrainerError::CheckpointCorrupted("missing 'prefix' field in index JSON".to_string())
        })?;

        // Find the records array content between the outermost '[' and ']'.
        let records_start = s.find("\"records\"").ok_or_else(|| {
            TrainerError::CheckpointCorrupted("missing 'records' key".to_string())
        })?;
        let array_start = s[records_start..].find('[').ok_or_else(|| {
            TrainerError::CheckpointCorrupted("missing '[' for records array".to_string())
        })? + records_start;
        let array_end = find_matching_bracket(s, array_start, '[', ']').ok_or_else(|| {
            TrainerError::CheckpointCorrupted("unmatched '[' in records array".to_string())
        })?;

        let array_content = &s[array_start + 1..array_end];

        let records = parse_record_objects(array_content)?;

        Ok(Self { prefix, records })
    }

    /// Write the index to a file at `path`.
    pub fn save(&self, path: &std::path::Path) -> Result<(), TrainerError> {
        let json = self.to_json();
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load an index from a file at `path`.
    pub fn load(path: &std::path::Path) -> Result<Self, TrainerError> {
        let contents = std::fs::read_to_string(path)?;
        Self::from_json(&contents)
    }
}

// ---------------------------------------------------------------------------
// JSON helpers (private)
// ---------------------------------------------------------------------------

/// Escape special characters for JSON string values.
fn escape_json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out
}

/// Unescape a JSON escape sequence in a string value.
fn unescape_json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\\' {
            match chars.next() {
                Some('"') => out.push('"'),
                Some('\\') => out.push('\\'),
                Some('n') => out.push('\n'),
                Some('r') => out.push('\r'),
                Some('t') => out.push('\t'),
                Some(c) => {
                    out.push('\\');
                    out.push(c);
                }
                None => out.push('\\'),
            }
        } else {
            out.push(ch);
        }
    }
    out
}

/// Extract the value of a simple top-level JSON string field.
fn extract_json_string_field(json: &str, field: &str) -> Option<String> {
    let needle = format!("\"{}\"", field);
    let field_start = json.find(&needle)?;
    let after_key = &json[field_start + needle.len()..];
    // Skip whitespace and colon.
    let colon_pos = after_key.find(':')?;
    let after_colon = after_key[colon_pos + 1..].trim_start();
    if !after_colon.starts_with('"') {
        return None;
    }
    let inner = &after_colon[1..];
    let mut value = String::new();
    let mut escaped = false;
    for ch in inner.chars() {
        if escaped {
            match ch {
                '"' => value.push('"'),
                '\\' => value.push('\\'),
                'n' => value.push('\n'),
                'r' => value.push('\r'),
                't' => value.push('\t'),
                c => {
                    value.push('\\');
                    value.push(c);
                }
            }
            escaped = false;
        } else if ch == '\\' {
            escaped = true;
        } else if ch == '"' {
            break;
        } else {
            value.push(ch);
        }
    }
    Some(value)
}

/// Find the index of the matching closing bracket for an opening bracket at
/// `start`.
fn find_matching_bracket(s: &str, start: usize, open: char, close: char) -> Option<usize> {
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    for (i, ch) in s[start..].char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        if in_string {
            if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }
        if ch == '"' {
            in_string = true;
        } else if ch == open {
            depth += 1;
        } else if ch == close {
            if depth == 0 {
                return Some(start + i);
            }
            depth -= 1;
            if depth == 0 {
                return Some(start + i);
            }
        }
    }
    None
}

/// Parse all JSON record objects from the content between `[` and `]`.
fn parse_record_objects(content: &str) -> Result<Vec<CheckpointRecord>, TrainerError> {
    let mut records = Vec::new();
    let mut pos = 0;
    let bytes = content.as_bytes();

    while pos < bytes.len() {
        // Skip whitespace and commas.
        while pos < bytes.len()
            && (bytes[pos] == b' '
                || bytes[pos] == b'\n'
                || bytes[pos] == b'\r'
                || bytes[pos] == b'\t'
                || bytes[pos] == b',')
        {
            pos += 1;
        }
        if pos >= bytes.len() {
            break;
        }
        if bytes[pos] != b'{' {
            break;
        }
        // Find the matching '}'.
        let obj_end = find_matching_bracket(content, pos, '{', '}').ok_or_else(|| {
            TrainerError::CheckpointCorrupted("unmatched '{' in record object".to_string())
        })?;
        let obj_str = &content[pos..=obj_end];
        records.push(parse_single_record(obj_str)?);
        pos = obj_end + 1;
    }
    Ok(records)
}

/// Parse one JSON record object into a [`CheckpointRecord`].
fn parse_single_record(obj: &str) -> Result<CheckpointRecord, TrainerError> {
    let iteration = extract_json_u32(obj, "iteration").ok_or_else(|| {
        TrainerError::CheckpointCorrupted("record missing 'iteration'".to_string())
    })?;
    let path_str = extract_json_string_field(obj, "path")
        .ok_or_else(|| TrainerError::CheckpointCorrupted("record missing 'path'".to_string()))?;
    let path = PathBuf::from(unescape_json_string(&path_str));
    let psnr = extract_json_f32(obj, "psnr")
        .ok_or_else(|| TrainerError::CheckpointCorrupted("record missing 'psnr'".to_string()))?;
    let ssim = extract_json_f32(obj, "ssim")
        .ok_or_else(|| TrainerError::CheckpointCorrupted("record missing 'ssim'".to_string()))?;
    let loss = extract_json_f32(obj, "loss")
        .ok_or_else(|| TrainerError::CheckpointCorrupted("record missing 'loss'".to_string()))?;
    let num_gaussians = extract_json_u32(obj, "num_gaussians").ok_or_else(|| {
        TrainerError::CheckpointCorrupted("record missing 'num_gaussians'".to_string())
    })?;
    let saved_at_unix_secs = extract_json_u64(obj, "saved_at_unix_secs").ok_or_else(|| {
        TrainerError::CheckpointCorrupted("record missing 'saved_at_unix_secs'".to_string())
    })?;
    let size_bytes = extract_json_u64(obj, "size_bytes").ok_or_else(|| {
        TrainerError::CheckpointCorrupted("record missing 'size_bytes'".to_string())
    })?;
    let is_best = extract_json_bool(obj, "is_best")
        .ok_or_else(|| TrainerError::CheckpointCorrupted("record missing 'is_best'".to_string()))?;

    Ok(CheckpointRecord {
        iteration,
        path,
        psnr,
        ssim,
        loss,
        num_gaussians,
        saved_at_unix_secs,
        size_bytes,
        is_best,
    })
}

/// Extract a u32 field from a JSON object string.
fn extract_json_u32(obj: &str, field: &str) -> Option<u32> {
    let raw = extract_json_number_raw(obj, field)?;
    raw.trim().parse::<u32>().ok()
}

/// Extract a u64 field from a JSON object string.
fn extract_json_u64(obj: &str, field: &str) -> Option<u64> {
    let raw = extract_json_number_raw(obj, field)?;
    raw.trim().parse::<u64>().ok()
}

/// Extract an f32 field from a JSON object string.
fn extract_json_f32(obj: &str, field: &str) -> Option<f32> {
    let raw = extract_json_number_raw(obj, field)?;
    raw.trim().parse::<f32>().ok()
}

/// Extract a boolean field from a JSON object string.
fn extract_json_bool(obj: &str, field: &str) -> Option<bool> {
    let needle = format!("\"{}\"", field);
    let field_start = obj.find(&needle)?;
    let after_key = &obj[field_start + needle.len()..];
    let colon_pos = after_key.find(':')?;
    let after_colon = after_key[colon_pos + 1..].trim_start();
    if after_colon.starts_with("true") {
        Some(true)
    } else if after_colon.starts_with("false") {
        Some(false)
    } else {
        None
    }
}

/// Extract the raw text of a JSON number value for the given field.
fn extract_json_number_raw<'a>(obj: &'a str, field: &str) -> Option<&'a str> {
    let needle = format!("\"{}\"", field);
    let field_start = obj.find(&needle)?;
    let after_key = &obj[field_start + needle.len()..];
    let colon_pos = after_key.find(':')?;
    let after_colon = after_key[colon_pos + 1..].trim_start();
    // A number ends at ',', '}', whitespace, or newline.
    let end = after_colon
        .find([',', '}', '\n', '\r'])
        .unwrap_or(after_colon.len());
    Some(after_colon[..end].trim())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn temp_dir(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("oxigaf_ckpt_test_{name}"))
    }

    fn make_record(iter: u32, psnr: f32) -> CheckpointRecord {
        let path = PathBuf::from(format!("/tmp/ckpt_{iter:08}.json"));
        CheckpointRecord::new(iter, path, psnr, 0.85, 0.05, 5000, 1024)
    }

    // -----------------------------------------------------------------------
    // Policy tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_checkpoint_policy_default() {
        let p = CheckpointPolicy::default();
        assert_eq!(p.keep_last_n, 5);
        assert_eq!(p.keep_best_n, 3);
        assert_eq!(p.keep_every_n, 10_000);
        assert!((p.min_psnr_to_save - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_checkpoint_policy_validate() -> Result<(), TrainerError> {
        CheckpointPolicy::default().validate()?;
        CheckpointPolicy::aggressive().validate()?;
        CheckpointPolicy::conservative().validate()?;
        Ok(())
    }

    #[test]
    fn test_checkpoint_policy_validate_zero_keep_last_error() {
        let p = CheckpointPolicy {
            keep_last_n: 0,
            ..CheckpointPolicy::default()
        };
        assert!(p.validate().is_err(), "keep_last_n = 0 should be invalid");
    }

    #[test]
    fn test_checkpoint_policy_validate_zero_keep_best_error() {
        let p = CheckpointPolicy {
            keep_best_n: 0,
            ..CheckpointPolicy::default()
        };
        assert!(p.validate().is_err(), "keep_best_n = 0 should be invalid");
    }

    // -----------------------------------------------------------------------
    // Record tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_checkpoint_record_is_better_than() {
        let good = make_record(1000, 30.0);
        let bad = make_record(2000, 25.0);
        assert!(
            good.is_better_than(&bad),
            "30 psnr should be better than 25"
        );
        assert!(
            !bad.is_better_than(&good),
            "25 psnr should not be better than 30"
        );
    }

    #[test]
    fn test_checkpoint_record_format_line() {
        let mut r = make_record(1000, 25.3);
        r.ssim = 0.87;
        r.num_gaussians = 5000;
        r.is_best = true;
        let line = r.format_line();
        assert!(line.contains("iter=1000"), "line: {line}");
        assert!(line.contains("psnr=25.3"), "line: {line}");
        assert!(line.contains("ssim=0.87"), "line: {line}");
        assert!(line.contains("gaussians=5000"), "line: {line}");
        assert!(line.contains("is_best=true"), "line: {line}");
    }

    // -----------------------------------------------------------------------
    // Manager construction
    // -----------------------------------------------------------------------

    #[test]
    fn test_manager_new_creates_dir() -> Result<(), TrainerError> {
        let dir = temp_dir("new_creates_dir");
        // Ensure it doesn't exist beforehand.
        let _ = std::fs::remove_dir_all(&dir);
        let _mgr = CheckpointManager::new(dir.clone(), "run", CheckpointPolicy::default())?;
        assert!(dir.exists(), "checkpoint_dir must be created");
        let _ = std::fs::remove_dir_all(&dir);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // add_checkpoint / rotation
    // -----------------------------------------------------------------------

    #[test]
    fn test_add_checkpoint_single() -> Result<(), TrainerError> {
        let dir = temp_dir("add_single");
        let _ = std::fs::remove_dir_all(&dir);
        let mut mgr = CheckpointManager::new(dir.clone(), "run", CheckpointPolicy::default())?;
        let r = make_record(1000, 25.0);
        let to_delete = mgr.add_checkpoint(r);
        assert!(
            to_delete.is_empty(),
            "only one checkpoint: nothing should be deleted"
        );
        assert_eq!(mgr.all_records().len(), 1);
        let _ = std::fs::remove_dir_all(&dir);
        Ok(())
    }

    #[test]
    fn test_add_checkpoint_updates_best() -> Result<(), TrainerError> {
        let dir = temp_dir("updates_best");
        let _ = std::fs::remove_dir_all(&dir);
        let mut mgr = CheckpointManager::new(dir.clone(), "run", CheckpointPolicy::default())?;
        mgr.add_checkpoint(make_record(1000, 25.0));
        mgr.add_checkpoint(make_record(2000, 30.0));
        let best = mgr
            .best()
            .ok_or_else(|| TrainerError::Checkpoint("no best".to_string()))?;
        assert!((best.psnr - 30.0).abs() < 1e-5, "best psnr should be 30.0");
        assert!(best.is_best, "is_best must be set on best record");
        let _ = std::fs::remove_dir_all(&dir);
        Ok(())
    }

    #[test]
    fn test_rotation_keeps_last_n() -> Result<(), TrainerError> {
        let dir = temp_dir("keeps_last_n");
        let _ = std::fs::remove_dir_all(&dir);
        let policy = CheckpointPolicy {
            keep_last_n: 3,
            keep_best_n: 1,
            keep_every_n: 1_000_000, // effectively never
            min_psnr_to_save: 0.0,
        };
        let mut mgr = CheckpointManager::new(dir.clone(), "run", policy)?;
        // Add 6 checkpoints with declining PSNR so "best" is always the first.
        for i in 1..=6u32 {
            mgr.add_checkpoint(make_record(i * 1000, 30.0 - i as f32));
        }
        // Keep last 3 + best 1 (iter 1000 with psnr 29.0) — possibly overlap.
        let iters: Vec<u32> = mgr.all_records().iter().map(|r| r.iteration).collect();
        // The last 3 are 4000, 5000, 6000.
        assert!(iters.contains(&4000), "iter 4000 should be kept: {iters:?}");
        assert!(iters.contains(&5000), "iter 5000 should be kept: {iters:?}");
        assert!(iters.contains(&6000), "iter 6000 should be kept: {iters:?}");
        let _ = std::fs::remove_dir_all(&dir);
        Ok(())
    }

    #[test]
    fn test_rotation_keeps_best_n() -> Result<(), TrainerError> {
        let dir = temp_dir("keeps_best_n");
        let _ = std::fs::remove_dir_all(&dir);
        let policy = CheckpointPolicy {
            keep_last_n: 1,
            keep_best_n: 2,
            keep_every_n: 1_000_000,
            min_psnr_to_save: 0.0,
        };
        let mut mgr = CheckpointManager::new(dir.clone(), "run", policy)?;
        // Checkpoints with specific PSNRs.
        mgr.add_checkpoint(make_record(1000, 20.0));
        mgr.add_checkpoint(make_record(2000, 28.0)); // 2nd best
        mgr.add_checkpoint(make_record(3000, 25.0));
        mgr.add_checkpoint(make_record(4000, 30.0)); // best + latest
        let iters: Vec<u32> = mgr.all_records().iter().map(|r| r.iteration).collect();
        // Must keep: iter 4000 (latest & best psnr=30) and iter 2000 (psnr=28, 2nd best).
        assert!(iters.contains(&4000), "iter 4000 must be kept: {iters:?}");
        assert!(iters.contains(&2000), "iter 2000 must be kept: {iters:?}");
        let _ = std::fs::remove_dir_all(&dir);
        Ok(())
    }

    #[test]
    fn test_rotation_keeps_every_n() -> Result<(), TrainerError> {
        let dir = temp_dir("keeps_every_n");
        let _ = std::fs::remove_dir_all(&dir);
        let policy = CheckpointPolicy {
            keep_last_n: 1,
            keep_best_n: 1,
            keep_every_n: 10_000,
            min_psnr_to_save: 0.0,
        };
        let mut mgr = CheckpointManager::new(dir.clone(), "run", policy)?;
        // A periodic checkpoint at exactly 10_000.
        mgr.add_checkpoint(make_record(10_000, 22.0));
        mgr.add_checkpoint(make_record(11_000, 28.0)); // latest + best
        mgr.add_checkpoint(make_record(12_000, 25.0)); // latest after this
        let iters: Vec<u32> = mgr.all_records().iter().map(|r| r.iteration).collect();
        // iter 10_000 is divisible by keep_every_n so it must survive.
        assert!(
            iters.contains(&10_000),
            "periodic iter 10000 must be kept: {iters:?}"
        );
        let _ = std::fs::remove_dir_all(&dir);
        Ok(())
    }

    #[test]
    fn test_best_record_updated_after_better() -> Result<(), TrainerError> {
        let dir = temp_dir("best_updated");
        let _ = std::fs::remove_dir_all(&dir);
        let mut mgr = CheckpointManager::new(dir.clone(), "run", CheckpointPolicy::default())?;
        mgr.add_checkpoint(make_record(1000, 25.0));
        assert!((mgr.best().map(|r| r.psnr).unwrap_or(0.0) - 25.0).abs() < 1e-5);
        mgr.add_checkpoint(make_record(2000, 30.0));
        assert!((mgr.best().map(|r| r.psnr).unwrap_or(0.0) - 30.0).abs() < 1e-5);
        mgr.add_checkpoint(make_record(3000, 28.0));
        // Best should remain 30.0.
        assert!((mgr.best().map(|r| r.psnr).unwrap_or(0.0) - 30.0).abs() < 1e-5);
        let _ = std::fs::remove_dir_all(&dir);
        Ok(())
    }

    #[test]
    fn test_latest_record() -> Result<(), TrainerError> {
        let dir = temp_dir("latest");
        let _ = std::fs::remove_dir_all(&dir);
        let mut mgr = CheckpointManager::new(dir.clone(), "run", CheckpointPolicy::default())?;
        mgr.add_checkpoint(make_record(1000, 25.0));
        mgr.add_checkpoint(make_record(3000, 26.0));
        mgr.add_checkpoint(make_record(2000, 27.0));
        let latest = mgr
            .latest()
            .ok_or_else(|| TrainerError::Checkpoint("no latest".to_string()))?;
        assert_eq!(latest.iteration, 3000, "latest should be iter 3000");
        let _ = std::fs::remove_dir_all(&dir);
        Ok(())
    }

    #[test]
    fn test_find_resume_checkpoint() -> Result<(), TrainerError> {
        let dir = temp_dir("resume");
        let _ = std::fs::remove_dir_all(&dir);
        let mut mgr = CheckpointManager::new(dir.clone(), "run", CheckpointPolicy::default())?;
        assert!(
            mgr.find_resume_checkpoint().is_none(),
            "empty manager has no resume point"
        );
        mgr.add_checkpoint(make_record(5000, 25.0));
        mgr.add_checkpoint(make_record(10_000, 26.0));
        let resume = mgr
            .find_resume_checkpoint()
            .ok_or_else(|| TrainerError::Checkpoint("no resume checkpoint".to_string()))?;
        assert_eq!(resume.iteration, 10_000);
        let _ = std::fs::remove_dir_all(&dir);
        Ok(())
    }

    #[test]
    fn test_checkpoint_path_format() -> Result<(), TrainerError> {
        let dir = temp_dir("path_format");
        let _ = std::fs::remove_dir_all(&dir);
        let mgr = CheckpointManager::new(dir.clone(), "gaf", CheckpointPolicy::default())?;
        let path = mgr.checkpoint_path_for(1234);
        let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        assert_eq!(
            file_name, "gaf_iter_00001234.json",
            "unexpected filename: {file_name}"
        );
        let _ = std::fs::remove_dir_all(&dir);
        Ok(())
    }

    #[test]
    fn test_should_save() -> Result<(), TrainerError> {
        let dir = temp_dir("should_save");
        let _ = std::fs::remove_dir_all(&dir);
        let policy = CheckpointPolicy {
            min_psnr_to_save: 20.0,
            ..CheckpointPolicy::default()
        };
        let mgr = CheckpointManager::new(dir.clone(), "run", policy)?;
        assert!(
            mgr.should_save(1000, 25.0),
            "psnr 25 >= min 20 → should save"
        );
        assert!(
            !mgr.should_save(1000, 15.0),
            "psnr 15 < min 20 → should not save"
        );
        assert!(mgr.should_save(1000, 20.0), "psnr == min → should save");
        let _ = std::fs::remove_dir_all(&dir);
        Ok(())
    }

    #[test]
    fn test_format_summary() -> Result<(), TrainerError> {
        let dir = temp_dir("summary");
        let _ = std::fs::remove_dir_all(&dir);
        let mut mgr = CheckpointManager::new(dir.clone(), "run", CheckpointPolicy::default())?;
        mgr.add_checkpoint(make_record(1000, 25.0));
        mgr.add_checkpoint(make_record(2000, 28.0));
        let summary = mgr.format_summary();
        assert!(
            summary.contains("CheckpointManager summary"),
            "summary: {summary}"
        );
        assert!(summary.contains("total checkpoints"), "summary: {summary}");
        assert!(summary.contains("best PSNR"), "summary: {summary}");
        assert!(summary.contains("latest iteration"), "summary: {summary}");
        assert!(summary.contains("disk usage"), "summary: {summary}");
        let _ = std::fs::remove_dir_all(&dir);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // JSON roundtrip
    // -----------------------------------------------------------------------

    #[test]
    fn test_checkpoint_index_json_roundtrip() -> Result<(), TrainerError> {
        let dir = temp_dir("json_roundtrip");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir)?;

        let mut records = Vec::new();
        for i in 1..=3u32 {
            let mut r = make_record(i * 1000, 20.0 + i as f32 * 2.5);
            r.ssim = 0.80 + i as f32 * 0.05;
            r.loss = 0.1 / i as f32;
            r.saved_at_unix_secs = 0;
            r.size_bytes = 2048 * i as u64;
            if i == 3 {
                r.is_best = true;
            }
            records.push(r);
        }

        let index = CheckpointIndex {
            prefix: "gaf_run".to_string(),
            records,
        };

        let index_path = dir.join("checkpoint_index.json");
        index.save(&index_path)?;

        let loaded = CheckpointIndex::load(&index_path)?;

        assert_eq!(loaded.prefix, "gaf_run");
        assert_eq!(loaded.records.len(), 3);

        for (orig, loaded) in index.records.iter().zip(loaded.records.iter()) {
            assert_eq!(orig.iteration, loaded.iteration, "iteration mismatch");
            assert!(
                (orig.psnr - loaded.psnr).abs() < 1e-3,
                "psnr mismatch: {} vs {}",
                orig.psnr,
                loaded.psnr
            );
            assert!(
                (orig.ssim - loaded.ssim).abs() < 1e-4,
                "ssim mismatch: {} vs {}",
                orig.ssim,
                loaded.ssim
            );
            assert_eq!(orig.is_best, loaded.is_best, "is_best mismatch");
            assert_eq!(orig.size_bytes, loaded.size_bytes, "size_bytes mismatch");
        }

        let _ = std::fs::remove_dir_all(&dir);
        Ok(())
    }
}
