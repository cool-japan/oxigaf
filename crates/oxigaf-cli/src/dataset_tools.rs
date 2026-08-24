//! Dataset management utilities for OxiGAF training pipelines.
//!
//! This module provides tools for scanning training data directories,
//! creating reproducible train/val/test splits, computing dataset statistics,
//! and validating dataset structure.
//!
//! # Example
//! ```rust,no_run
//! use oxigaf_cli::dataset_tools::{
//!     DatasetScanner, SplitConfig, split_dataset, compute_dataset_stats,
//!     apply_split, validate_dataset,
//! };
//! use std::path::Path;
//!
//! let dir = Path::new("/data/avatars");
//! let stats = validate_dataset(dir, 10).expect("valid dataset");
//! println!("{}", stats.format_summary());
//! ```

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use thiserror::Error;

// ---------------------------------------------------------------------------
// DatasetError
// ---------------------------------------------------------------------------

/// Errors produced by dataset management operations.
#[derive(Debug, Error)]
pub enum DatasetError {
    /// The specified dataset directory was not found.
    #[error("Dataset directory not found: {path}")]
    DirectoryNotFound { path: String },

    /// No files were found in the dataset directory.
    #[error("Empty dataset: no files found in {path}")]
    EmptyDataset { path: String },

    /// The provided split ratios do not sum to 1.0.
    #[error(
        "Invalid split ratios: train({train:.2}) + val({val:.2}) + test({test:.2}) = {sum:.2}, must equal 1.0"
    )]
    InvalidSplitRatios {
        train: f32,
        val: f32,
        test: f32,
        sum: f32,
    },

    /// Split validation encountered a logical error.
    #[error("Split validation failed: {reason}")]
    SplitValidationFailed { reason: String },

    /// An underlying I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// A JSON serialization/deserialization error.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// The dataset has fewer files than the required minimum.
    #[error("Dataset too small: need at least {needed} files, got {actual}")]
    TooSmall { needed: usize, actual: usize },
}

// ---------------------------------------------------------------------------
// FileType
// ---------------------------------------------------------------------------

/// Classification of a file by its extension.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum FileType {
    /// Image files: `.png`, `.jpg`, `.jpeg`, `.exr`, `.hdr`
    Image,
    /// 3D model / weight files: `.ply`, `.safetensors`, `.bin`
    Model,
    /// Configuration files: `.json`, `.toml`, `.yaml`
    Config,
    /// Video files: `.mp4`, `.avi`, `.mov`
    Video,
    /// Any extension not matched by the above categories.
    Unknown,
}

impl FileType {
    /// Classify a file extension (lowercase, without leading dot).
    pub fn from_extension(ext: &str) -> Self {
        match ext {
            "png" | "jpg" | "jpeg" | "exr" | "hdr" => FileType::Image,
            "ply" | "safetensors" | "bin" => FileType::Model,
            "json" | "toml" | "yaml" => FileType::Config,
            "mp4" | "avi" | "mov" => FileType::Video,
            _ => FileType::Unknown,
        }
    }
}

// ---------------------------------------------------------------------------
// FileEntry
// ---------------------------------------------------------------------------

/// Metadata record for a single file found in a dataset scan.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FileEntry {
    /// Absolute path to the file.
    pub path: PathBuf,
    /// Classified type based on extension.
    pub file_type: FileType,
    /// File size in bytes.
    pub size_bytes: u64,
    /// Filename (without parent directories).
    pub name: String,
}

impl FileEntry {
    /// Build a `FileEntry` from a path, reading metadata from the filesystem.
    pub fn from_path(path: PathBuf) -> Result<Self, DatasetError> {
        let meta = std::fs::metadata(&path)?;
        let size_bytes = meta.len();
        let name = path
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        let ext = path
            .extension()
            .map(|s| s.to_string_lossy().to_lowercase())
            .unwrap_or_default();
        let file_type = FileType::from_extension(&ext);
        Ok(FileEntry {
            path,
            file_type,
            size_bytes,
            name,
        })
    }

    /// Return the lowercase file extension (empty string if none).
    ///
    /// Returns an owned `String` (rather than borrowing from `self.path`)
    /// because lowercasing an extension that contains uppercase bytes
    /// necessarily allocates; `entry.extension() == "png"` still works via
    /// `PartialEq<str>`/`PartialEq<&str>` on `String`.
    pub fn extension(&self) -> String {
        self.path
            .extension()
            .and_then(|s| s.to_str())
            .map(str::to_lowercase)
            .unwrap_or_default()
    }
}

// ---------------------------------------------------------------------------
// DatasetStats
// ---------------------------------------------------------------------------

/// Aggregate statistics computed over a collection of [`FileEntry`] records.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DatasetStats {
    /// Total number of files.
    pub total_files: usize,
    /// Count of image files.
    pub image_count: usize,
    /// Count of model/weight files.
    pub model_count: usize,
    /// Count of configuration files.
    pub config_count: usize,
    /// Count of video files.
    pub video_count: usize,
    /// Count of files with unrecognized extensions.
    pub unknown_count: usize,
    /// Sum of all file sizes in bytes.
    pub total_bytes: u64,
    /// Mean file size in bytes (0 if no files).
    pub mean_file_size_bytes: u64,
    /// Largest individual file in bytes (0 if no files).
    pub largest_file_bytes: u64,
    /// Smallest individual file in bytes (0 if no files).
    pub smallest_file_bytes: u64,
}

impl DatasetStats {
    /// Return a concise human-readable summary string.
    pub fn format_summary(&self) -> String {
        format!(
            "Dataset: {} files ({:.2} MB) | images={} models={} configs={} videos={} unknown={}",
            self.total_files,
            self.total_mb(),
            self.image_count,
            self.model_count,
            self.config_count,
            self.video_count,
            self.unknown_count,
        )
    }

    /// Total size in megabytes.
    pub fn total_mb(&self) -> f64 {
        self.total_bytes as f64 / 1_000_000.0
    }
}

// ---------------------------------------------------------------------------
// DatasetSplit
// ---------------------------------------------------------------------------

/// Describes how a dataset of `total_items` entries is partitioned into
/// training, validation, and test subsets via index lists.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DatasetSplit {
    /// Indices of items assigned to the training set.
    pub train_indices: Vec<usize>,
    /// Indices of items assigned to the validation set.
    pub val_indices: Vec<usize>,
    /// Indices of items assigned to the test set.
    pub test_indices: Vec<usize>,
    /// Total number of items in the dataset.
    pub total_items: usize,
}

impl DatasetSplit {
    /// Number of items in the training set.
    pub fn train_count(&self) -> usize {
        self.train_indices.len()
    }

    /// Number of items in the validation set.
    pub fn val_count(&self) -> usize {
        self.val_indices.len()
    }

    /// Number of items in the test set.
    pub fn test_count(&self) -> usize {
        self.test_indices.len()
    }

    /// Returns `true` if the split covers all items with no duplicates and all
    /// indices are within `[0, total_items)`.
    pub fn is_valid(&self) -> bool {
        validate_split(self).is_ok()
    }

    /// Return a concise human-readable summary of the split.
    pub fn format_summary(&self) -> String {
        let total = self.total_items;
        let t = self.train_count();
        let v = self.val_count();
        let te = self.test_count();
        let tf = if total > 0 {
            t as f32 / total as f32
        } else {
            0.0
        };
        let vf = if total > 0 {
            v as f32 / total as f32
        } else {
            0.0
        };
        let tef = if total > 0 {
            te as f32 / total as f32
        } else {
            0.0
        };
        format!(
            "Split: total={} | train={} ({:.1}%) | val={} ({:.1}%) | test={} ({:.1}%)",
            total,
            t,
            tf * 100.0,
            v,
            vf * 100.0,
            te,
            tef * 100.0,
        )
    }
}

// ---------------------------------------------------------------------------
// SplitConfig
// ---------------------------------------------------------------------------

/// Configuration controlling how a dataset is split.
#[derive(Debug, Clone)]
pub struct SplitConfig {
    /// Fraction of data assigned to training (default 0.8).
    pub train_ratio: f32,
    /// Fraction of data assigned to validation (default 0.1).
    pub val_ratio: f32,
    /// Fraction of data assigned to testing (default 0.1).
    pub test_ratio: f32,
    /// Seed for the xorshift64 PRNG used when shuffling.
    pub seed: u64,
    /// Whether to shuffle indices before splitting. Ignored when `strategy`
    /// is [`DatasetSplitStrategy::Sequential`] (original order is kept).
    pub shuffle: bool,
    /// How indices are ordered before slicing into train/val/test ranges.
    /// See [`split_dataset`] for the exact semantics of each variant.
    pub strategy: DatasetSplitStrategy,
}

impl Default for SplitConfig {
    fn default() -> Self {
        SplitConfig {
            train_ratio: 0.8,
            val_ratio: 0.1,
            test_ratio: 0.1,
            seed: 42,
            shuffle: true,
            strategy: DatasetSplitStrategy::Random,
        }
    }
}

// ---------------------------------------------------------------------------
// DatasetSplitStrategy
// ---------------------------------------------------------------------------

/// Strategy controlling how entries are assigned to splits.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum DatasetSplitStrategy {
    /// Shuffle the dataset then assign contiguous ranges to each split.
    Random,
    /// Assign the first N% to train, next M% to val, remainder to test, in
    /// original scan order (`SplitConfig::shuffle` is ignored).
    Sequential,
    /// Stratified split: hold each group of a caller-supplied key together
    /// within its own train/val/test slice, so no single key is spread
    /// unevenly across splits.
    ///
    /// [`split_dataset`] takes only an item *count* (`n: usize`), with no
    /// per-item labels to stratify on; requesting this strategy through
    /// that entry point logs a warning and falls back to the same behaviour
    /// as [`DatasetSplitStrategy::Random`]. Callers that have a stratum key
    /// per item should call [`split_dataset_stratified`] directly instead,
    /// which performs a real per-group split.
    Stratified,
}

// ---------------------------------------------------------------------------
// DatasetScanner
// ---------------------------------------------------------------------------

/// Maximum directory recursion depth for [`DatasetScanner::scan`].
///
/// A backstop against pathological/adversarial directory trees; the primary
/// cycle guard is the canonicalised visited-path set in `scan_dir`, which
/// alone is sufficient to stop a genuine symlink cycle.
const MAX_SCAN_DEPTH: usize = 128;

/// Configurable directory scanner that produces [`FileEntry`] lists.
pub struct DatasetScanner {
    /// Extension filter (lowercase, without dot). Empty means accept all files.
    pub extensions: Vec<String>,
    /// Whether to recurse into subdirectories.
    pub recursive: bool,
    /// Minimum file size in bytes (inclusive, 0 = no lower bound).
    pub min_size_bytes: u64,
    /// Maximum file size in bytes (inclusive, 0 = no upper bound).
    pub max_size_bytes: u64,
}

impl Default for DatasetScanner {
    fn default() -> Self {
        DatasetScanner {
            extensions: vec![],
            recursive: true,
            min_size_bytes: 0,
            max_size_bytes: 0,
        }
    }
}

impl DatasetScanner {
    /// Create a scanner with default settings (recursive, no size limits, all files).
    pub fn new() -> Self {
        Self::default()
    }

    /// Restrict scanning to files with one of the given extensions (lowercase, no dot).
    pub fn with_extensions(mut self, exts: Vec<&str>) -> Self {
        self.extensions = exts.iter().map(|s| s.to_lowercase()).collect();
        self
    }

    /// Set whether subdirectories are scanned recursively.
    pub fn with_recursive(mut self, recursive: bool) -> Self {
        self.recursive = recursive;
        self
    }

    /// Scan `dir` and return all matching [`FileEntry`] records sorted by path.
    ///
    /// Symlinked files and directories are resolved and scanned like real
    /// entries (a common way to assemble a dataset without copying files);
    /// recursion into symlinked directories is guarded against cycles by a
    /// canonicalised visited-path set plus [`MAX_SCAN_DEPTH`].
    pub fn scan(&self, dir: &Path) -> Result<Vec<FileEntry>, DatasetError> {
        if !dir.exists() || !dir.is_dir() {
            return Err(DatasetError::DirectoryNotFound {
                path: dir.to_string_lossy().into_owned(),
            });
        }
        let mut entries = Vec::new();
        let mut visited = HashSet::new();
        self.scan_dir(dir, &mut entries, &mut visited, 0)?;
        entries.sort_by(|a, b| a.path.cmp(&b.path));
        Ok(entries)
    }

    /// Scan `dir` and return only entries of the specified [`FileType`].
    pub fn scan_by_type(
        &self,
        dir: &Path,
        file_type: FileType,
    ) -> Result<Vec<FileEntry>, DatasetError> {
        let all = self.scan(dir)?;
        Ok(all
            .into_iter()
            .filter(|e| e.file_type == file_type)
            .collect())
    }

    // Internal recursive scan implementation.
    //
    // `DirEntry::metadata` does not follow symlinks (on Unix it is
    // equivalent to `symlink_metadata`), so a symlink entry's `meta.is_dir()`
    // and `meta.is_file()` are both `false` and it used to be silently
    // dropped without recursing into it or recording it. Resolve the link
    // target explicitly via `std::fs::metadata` (which *does* follow
    // symlinks) and treat it as whatever it points to. `visited` (populated
    // with canonicalised directory paths) plus `MAX_SCAN_DEPTH` guard
    // against a symlink cycle causing an infinite loop or stack overflow.
    fn scan_dir(
        &self,
        dir: &Path,
        out: &mut Vec<FileEntry>,
        visited: &mut HashSet<PathBuf>,
        depth: usize,
    ) -> Result<(), DatasetError> {
        if depth > MAX_SCAN_DEPTH {
            return Ok(());
        }
        if let Ok(canonical) = dir.canonicalize() {
            if !visited.insert(canonical) {
                return Ok(()); // already visited this real directory: cycle
            }
        }

        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            let mut meta = entry.metadata()?;

            if meta.file_type().is_symlink() {
                match std::fs::metadata(&path) {
                    Ok(target_meta) => meta = target_meta,
                    Err(_) => continue, // broken symlink target: skip
                }
            }

            if meta.is_dir() {
                if self.recursive {
                    self.scan_dir(&path, out, visited, depth + 1)?;
                }
                continue;
            }

            if !meta.is_file() {
                continue;
            }

            // Extension filter
            if !self.extensions.is_empty() {
                let ext = path
                    .extension()
                    .map(|s| s.to_string_lossy().to_lowercase())
                    .unwrap_or_default();
                if !self.extensions.contains(&ext) {
                    continue;
                }
            }

            // Size filters
            let size = meta.len();
            if size < self.min_size_bytes {
                continue;
            }
            if self.max_size_bytes > 0 && size > self.max_size_bytes {
                continue;
            }

            if let Ok(fe) = FileEntry::from_path(path) {
                out.push(fe); // skip unreadable files on Err
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// xorshift64 PRNG
// ---------------------------------------------------------------------------

/// Inline xorshift64 pseudo-random number generator.
///
/// The state must be non-zero; if zero is passed the function uses 1 instead.
fn xorshift64(state: &mut u64) -> u64 {
    if *state == 0 {
        *state = 1;
    }
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// Compute aggregate statistics over a slice of [`FileEntry`] records.
///
/// Returns zeroed-out stats when `entries` is empty.
pub fn compute_dataset_stats(entries: &[FileEntry]) -> DatasetStats {
    let total_files = entries.len();
    if total_files == 0 {
        return DatasetStats {
            total_files: 0,
            image_count: 0,
            model_count: 0,
            config_count: 0,
            video_count: 0,
            unknown_count: 0,
            total_bytes: 0,
            mean_file_size_bytes: 0,
            largest_file_bytes: 0,
            smallest_file_bytes: 0,
        };
    }

    let mut image_count = 0usize;
    let mut model_count = 0usize;
    let mut config_count = 0usize;
    let mut video_count = 0usize;
    let mut unknown_count = 0usize;
    let mut total_bytes: u64 = 0;
    let mut largest_file_bytes: u64 = 0;
    let mut smallest_file_bytes: u64 = u64::MAX;

    for entry in entries {
        match entry.file_type {
            FileType::Image => image_count += 1,
            FileType::Model => model_count += 1,
            FileType::Config => config_count += 1,
            FileType::Video => video_count += 1,
            FileType::Unknown => unknown_count += 1,
        }
        total_bytes = total_bytes.saturating_add(entry.size_bytes);
        if entry.size_bytes > largest_file_bytes {
            largest_file_bytes = entry.size_bytes;
        }
        if entry.size_bytes < smallest_file_bytes {
            smallest_file_bytes = entry.size_bytes;
        }
    }

    let mean_file_size_bytes = total_bytes / total_files as u64;

    DatasetStats {
        total_files,
        image_count,
        model_count,
        config_count,
        video_count,
        unknown_count,
        total_bytes,
        mean_file_size_bytes,
        largest_file_bytes,
        smallest_file_bytes,
    }
}

/// Shuffle `indices` in-place using a Fisher-Yates shuffle driven by xorshift64.
pub fn shuffle_indices(indices: &mut [usize], seed: u64) {
    let n = indices.len();
    if n < 2 {
        return;
    }
    let mut state: u64 = if seed == 0 { 1 } else { seed };
    for i in (1..n).rev() {
        let j = (xorshift64(&mut state) as usize) % (i + 1);
        indices.swap(i, j);
    }
}

/// Partition `n` items into train/val/test subsets using the supplied [`SplitConfig`].
///
/// The split ratios must sum to approximately 1.0 (within 1e-4).
/// [`SplitConfig::strategy`] selects how indices are ordered before slicing
/// into contiguous train/val/test ranges:
/// - [`DatasetSplitStrategy::Random`] (default): shuffle with
///   [`shuffle_indices`] when `config.shuffle` is `true`.
/// - [`DatasetSplitStrategy::Sequential`]: keep the original `0..n` order;
///   `config.shuffle` is ignored.
/// - [`DatasetSplitStrategy::Stratified`]: this entry point has no per-item
///   labels to stratify on, so it logs a warning and behaves like `Random`.
///   Use [`split_dataset_stratified`] for a real per-group split.
pub fn split_dataset(n: usize, config: &SplitConfig) -> Result<DatasetSplit, DatasetError> {
    let sum = config.train_ratio + config.val_ratio + config.test_ratio;
    if (sum - 1.0_f32).abs() >= 1e-4 {
        return Err(DatasetError::InvalidSplitRatios {
            train: config.train_ratio,
            val: config.val_ratio,
            test: config.test_ratio,
            sum,
        });
    }

    if n == 0 {
        return Ok(DatasetSplit {
            train_indices: vec![],
            val_indices: vec![],
            test_indices: vec![],
            total_items: 0,
        });
    }

    let mut indices: Vec<usize> = (0..n).collect();

    match config.strategy {
        DatasetSplitStrategy::Sequential => {
            // Keep the original 0..n order; config.shuffle is ignored.
        }
        DatasetSplitStrategy::Random => {
            if config.shuffle {
                shuffle_indices(&mut indices, config.seed);
            }
        }
        DatasetSplitStrategy::Stratified => {
            tracing::warn!(
                "DatasetSplitStrategy::Stratified was requested via split_dataset(), which \
                 has no per-item labels to stratify on; falling back to Random. Use \
                 split_dataset_stratified() for a real per-group stratified split."
            );
            if config.shuffle {
                shuffle_indices(&mut indices, config.seed);
            }
        }
    }

    let (train_indices, val_indices, test_indices) = slice_into_splits(&indices, config);

    Ok(DatasetSplit {
        train_indices,
        val_indices,
        test_indices,
        total_items: n,
    })
}

/// Split a caller-ordered slice of indices into contiguous train/val/test
/// ranges using `config`'s ratios. Shared by [`split_dataset`] and
/// [`split_dataset_stratified`] so both apply ratios identically.
fn slice_into_splits(
    indices: &[usize],
    config: &SplitConfig,
) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    let n = indices.len();
    let train_end = (n as f32 * config.train_ratio).floor() as usize;
    let val_end = train_end + (n as f32 * config.val_ratio).floor() as usize;

    (
        indices[..train_end].to_vec(),
        indices[train_end..val_end].to_vec(),
        indices[val_end..].to_vec(),
    )
}

/// Partition items into train/val/test subsets while keeping each `key`
/// group (e.g. a subject/scene identifier) from being spread unevenly
/// across splits: [`SplitConfig`]'s ratios are applied independently within
/// every distinct key, and the resulting index sets are concatenated.
///
/// `keys[i]` is the stratification label for item `i`; `keys.len()`
/// determines `n` (there is no separate item count to pass). Within each
/// group, indices are shuffled with a group-specific derivative of
/// `config.seed` when `config.shuffle` is `true` (ignored for
/// [`DatasetSplitStrategy::Sequential`], same as [`split_dataset`]).
///
/// # Errors
/// Returns [`DatasetError::InvalidSplitRatios`] under the same condition as
/// [`split_dataset`].
pub fn split_dataset_stratified<K: std::hash::Hash + Eq + Ord + Clone>(
    keys: &[K],
    config: &SplitConfig,
) -> Result<DatasetSplit, DatasetError> {
    let sum = config.train_ratio + config.val_ratio + config.test_ratio;
    if (sum - 1.0_f32).abs() >= 1e-4 {
        return Err(DatasetError::InvalidSplitRatios {
            train: config.train_ratio,
            val: config.val_ratio,
            test: config.test_ratio,
            sum,
        });
    }

    let n = keys.len();
    if n == 0 {
        return Ok(DatasetSplit {
            train_indices: vec![],
            val_indices: vec![],
            test_indices: vec![],
            total_items: 0,
        });
    }

    // Group item indices by key. A `BTreeMap` (rather than a `HashMap`)
    // keeps group iteration order deterministic across runs regardless of
    // `K`'s hash implementation, which keeps the group-specific seed
    // derivation below reproducible.
    let mut groups: std::collections::BTreeMap<K, Vec<usize>> = std::collections::BTreeMap::new();
    for (idx, key) in keys.iter().enumerate() {
        groups.entry(key.clone()).or_default().push(idx);
    }

    let shuffle = !matches!(config.strategy, DatasetSplitStrategy::Sequential) && config.shuffle;

    let mut train_indices = Vec::new();
    let mut val_indices = Vec::new();
    let mut test_indices = Vec::new();

    for (group_idx, (_key, mut group)) in groups.into_iter().enumerate() {
        if shuffle {
            // Derive a distinct-but-reproducible seed per group so that
            // groups don't all shuffle identically (which would otherwise
            // correlate their train/val/test boundaries).
            let group_seed = config
                .seed
                .wrapping_add((group_idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            shuffle_indices(&mut group, group_seed);
        }
        let (mut g_train, mut g_val, mut g_test) = slice_into_splits(&group, config);
        train_indices.append(&mut g_train);
        val_indices.append(&mut g_val);
        test_indices.append(&mut g_test);
    }

    Ok(DatasetSplit {
        train_indices,
        val_indices,
        test_indices,
        total_items: n,
    })
}

/// Validate that a [`DatasetSplit`] is internally consistent:
///
/// - Every index is in `[0, total_items)`.
/// - No index appears in more than one subset.
/// - All `total_items` items are covered by the union of the three subsets.
pub fn validate_split(split: &DatasetSplit) -> Result<(), DatasetError> {
    let n = split.total_items;

    let mut train_set = HashSet::with_capacity(split.train_indices.len());
    let mut val_set = HashSet::with_capacity(split.val_indices.len());
    let mut test_set = HashSet::with_capacity(split.test_indices.len());

    for &idx in &split.train_indices {
        if idx >= n {
            return Err(DatasetError::SplitValidationFailed {
                reason: format!("train index {} out of range (total_items={})", idx, n),
            });
        }
        if !train_set.insert(idx) {
            return Err(DatasetError::SplitValidationFailed {
                reason: format!("duplicate index {} in train set", idx),
            });
        }
    }

    for &idx in &split.val_indices {
        if idx >= n {
            return Err(DatasetError::SplitValidationFailed {
                reason: format!("val index {} out of range (total_items={})", idx, n),
            });
        }
        if train_set.contains(&idx) {
            return Err(DatasetError::SplitValidationFailed {
                reason: format!("index {} appears in both train and val sets", idx),
            });
        }
        if !val_set.insert(idx) {
            return Err(DatasetError::SplitValidationFailed {
                reason: format!("duplicate index {} in val set", idx),
            });
        }
    }

    for &idx in &split.test_indices {
        if idx >= n {
            return Err(DatasetError::SplitValidationFailed {
                reason: format!("test index {} out of range (total_items={})", idx, n),
            });
        }
        if train_set.contains(&idx) || val_set.contains(&idx) {
            return Err(DatasetError::SplitValidationFailed {
                reason: format!(
                    "index {} appears in test set and also in train or val set",
                    idx
                ),
            });
        }
        if !test_set.insert(idx) {
            return Err(DatasetError::SplitValidationFailed {
                reason: format!("duplicate index {} in test set", idx),
            });
        }
    }

    let covered = train_set.len() + val_set.len() + test_set.len();
    if covered != n {
        return Err(DatasetError::SplitValidationFailed {
            reason: format!("split covers {} items but total_items={}", covered, n),
        });
    }

    Ok(())
}

/// Serialize a [`DatasetSplit`] to a JSON file at `path`.
pub fn save_split(split: &DatasetSplit, path: &Path) -> Result<(), DatasetError> {
    let json = serde_json::to_string_pretty(split)?;
    std::fs::write(path, json)?;
    Ok(())
}

/// Deserialize a [`DatasetSplit`] from a JSON file at `path`.
pub fn load_split(path: &Path) -> Result<DatasetSplit, DatasetError> {
    let json = std::fs::read_to_string(path)?;
    let split = serde_json::from_str(&json)?;
    Ok(split)
}

/// Split result for [`apply_split`]: `(train, val, test)` slices of [`FileEntry`] references.
type SplitResult<'a> = (Vec<&'a FileEntry>, Vec<&'a FileEntry>, Vec<&'a FileEntry>);

/// Apply a [`DatasetSplit`] to a list of [`FileEntry`] records.
///
/// Returns `(train_entries, val_entries, test_entries)` where each is a
/// sorted `Vec` of references into `entries`.
pub fn apply_split<'a>(
    entries: &'a [FileEntry],
    split: &DatasetSplit,
) -> Result<SplitResult<'a>, DatasetError> {
    let n = entries.len();
    if n != split.total_items {
        return Err(DatasetError::SplitValidationFailed {
            reason: format!(
                "entries length {} does not match split total_items {}",
                n, split.total_items
            ),
        });
    }

    // Build sorted index vectors so output order is deterministic.
    let mut train_idxs = split.train_indices.clone();
    let mut val_idxs = split.val_indices.clone();
    let mut test_idxs = split.test_indices.clone();
    train_idxs.sort_unstable();
    val_idxs.sort_unstable();
    test_idxs.sort_unstable();

    let resolve = |idxs: &[usize]| -> Result<Vec<&'a FileEntry>, DatasetError> {
        idxs.iter()
            .map(|&i| {
                entries
                    .get(i)
                    .ok_or_else(|| DatasetError::SplitValidationFailed {
                        reason: format!("index {} out of range for entries (len={})", i, n),
                    })
            })
            .collect()
    };

    let train = resolve(&train_idxs)?;
    let val = resolve(&val_idxs)?;
    let test = resolve(&test_idxs)?;

    Ok((train, val, test))
}

/// Check that `dir` exists and contains at least `min_files` files, returning
/// [`DatasetStats`] on success.
pub fn validate_dataset(dir: &Path, min_files: usize) -> Result<DatasetStats, DatasetError> {
    if !dir.exists() || !dir.is_dir() {
        return Err(DatasetError::DirectoryNotFound {
            path: dir.to_string_lossy().into_owned(),
        });
    }

    let scanner = DatasetScanner::new();
    let entries = scanner.scan(dir)?;

    if entries.is_empty() {
        return Err(DatasetError::EmptyDataset {
            path: dir.to_string_lossy().into_owned(),
        });
    }

    if entries.len() < min_files {
        return Err(DatasetError::TooSmall {
            needed: min_files,
            actual: entries.len(),
        });
    }

    Ok(compute_dataset_stats(&entries))
}

// ---------------------------------------------------------------------------
// SplitStats
// ---------------------------------------------------------------------------

/// Fractional statistics computed from a [`DatasetSplit`].
#[derive(Debug, Clone)]
pub struct SplitStats {
    /// Number of items in the training split.
    pub train_count: usize,
    /// Number of items in the validation split.
    pub val_count: usize,
    /// Number of items in the test split.
    pub test_count: usize,
    /// Fraction of total items in the training split.
    pub train_fraction: f32,
    /// Fraction of total items in the validation split.
    pub val_fraction: f32,
    /// Fraction of total items in the test split.
    pub test_fraction: f32,
}

/// Compute fractional split statistics from a [`DatasetSplit`].
pub fn compute_split_stats(split: &DatasetSplit) -> SplitStats {
    let total = split.total_items;
    let t = split.train_count();
    let v = split.val_count();
    let te = split.test_count();
    let (tf, vf, tef) = if total > 0 {
        (
            t as f32 / total as f32,
            v as f32 / total as f32,
            te as f32 / total as f32,
        )
    } else {
        (0.0, 0.0, 0.0)
    };
    SplitStats {
        train_count: t,
        val_count: v,
        test_count: te,
        train_fraction: tf,
        val_fraction: vf,
        test_fraction: tef,
    }
}

/// Format a [`DatasetStats`] as a multi-line ASCII table.
pub fn format_stats_table(stats: &DatasetStats) -> String {
    let rows = [
        ("Images", stats.image_count),
        ("Models", stats.model_count),
        ("Configs", stats.config_count),
        ("Videos", stats.video_count),
        ("Unknown", stats.unknown_count),
    ];
    let mut table = String::new();
    table.push_str("+----------+---------+\n");
    table.push_str("| Type     | Count   |\n");
    table.push_str("+----------+---------+\n");
    for (label, count) in &rows {
        table.push_str(&format!("| {:<8} | {:>7} |\n", label, count));
    }
    table.push_str("+----------+---------+\n");
    table.push_str(&format!("| Total    | {:>7} |\n", stats.total_files));
    table.push_str("+----------+---------+\n");
    table.push_str(&format!("| Size     | {:>5.2} MB|\n", stats.total_mb()));
    table.push_str("+----------+---------+\n");
    table
}

/// Group file entry indices by their `size_bytes`, returning groups that
/// contain more than one entry (potential duplicates by size).
pub fn find_size_duplicates(entries: &[FileEntry]) -> Vec<Vec<usize>> {
    let mut by_size: HashMap<u64, Vec<usize>> = HashMap::new();
    for (i, entry) in entries.iter().enumerate() {
        by_size.entry(entry.size_bytes).or_default().push(i);
    }
    let mut result: Vec<Vec<usize>> = by_size
        .into_values()
        .filter(|group| group.len() > 1)
        .collect();
    // Sort groups for deterministic output (by first index in each group).
    for group in &mut result {
        group.sort_unstable();
    }
    result.sort_by_key(|g| g[0]);
    result
}

/// Return references to entries whose `size_bytes >= min_bytes`.
pub fn filter_by_size(entries: &[FileEntry], min_bytes: u64) -> Vec<&FileEntry> {
    entries
        .iter()
        .filter(|e| e.size_bytes >= min_bytes)
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    // ------------------------------------------------------------------
    // Helper: create a temporary file with given content
    // ------------------------------------------------------------------
    fn write_temp_file(dir: &Path, name: &str, content: &[u8]) -> PathBuf {
        let path = dir.join(name);
        fs::write(&path, content).expect("write temp file");
        path
    }

    fn make_temp_dir() -> PathBuf {
        let mut dir = std::env::temp_dir();
        dir.push(format!("oxigaf_dataset_test_{}_{}", std::process::id(), {
            use std::time::{SystemTime, UNIX_EPOCH};
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.subsec_nanos())
                .unwrap_or(0)
        }));
        fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    // ------------------------------------------------------------------
    // 1. FileType::from_extension
    // ------------------------------------------------------------------

    #[test]
    fn test_filetype_image_png() {
        assert_eq!(FileType::from_extension("png"), FileType::Image);
    }

    #[test]
    fn test_filetype_image_jpg() {
        assert_eq!(FileType::from_extension("jpg"), FileType::Image);
    }

    #[test]
    fn test_filetype_image_jpeg() {
        assert_eq!(FileType::from_extension("jpeg"), FileType::Image);
    }

    #[test]
    fn test_filetype_image_exr() {
        assert_eq!(FileType::from_extension("exr"), FileType::Image);
    }

    #[test]
    fn test_filetype_image_hdr() {
        assert_eq!(FileType::from_extension("hdr"), FileType::Image);
    }

    #[test]
    fn test_filetype_model_ply() {
        assert_eq!(FileType::from_extension("ply"), FileType::Model);
    }

    #[test]
    fn test_filetype_model_safetensors() {
        assert_eq!(FileType::from_extension("safetensors"), FileType::Model);
    }

    #[test]
    fn test_filetype_model_bin() {
        assert_eq!(FileType::from_extension("bin"), FileType::Model);
    }

    #[test]
    fn test_filetype_config_json() {
        assert_eq!(FileType::from_extension("json"), FileType::Config);
    }

    #[test]
    fn test_filetype_config_toml() {
        assert_eq!(FileType::from_extension("toml"), FileType::Config);
    }

    #[test]
    fn test_filetype_config_yaml() {
        assert_eq!(FileType::from_extension("yaml"), FileType::Config);
    }

    #[test]
    fn test_filetype_video_mp4() {
        assert_eq!(FileType::from_extension("mp4"), FileType::Video);
    }

    #[test]
    fn test_filetype_video_avi() {
        assert_eq!(FileType::from_extension("avi"), FileType::Video);
    }

    #[test]
    fn test_filetype_video_mov() {
        assert_eq!(FileType::from_extension("mov"), FileType::Video);
    }

    #[test]
    fn test_filetype_unknown() {
        assert_eq!(FileType::from_extension("xyz"), FileType::Unknown);
    }

    #[test]
    fn test_filetype_unknown_empty() {
        assert_eq!(FileType::from_extension(""), FileType::Unknown);
    }

    // ------------------------------------------------------------------
    // 2. FileEntry::from_path
    // ------------------------------------------------------------------

    #[test]
    fn test_file_entry_from_path_valid() {
        let dir = make_temp_dir();
        let path = write_temp_file(&dir, "image.png", b"PNG_CONTENT_HERE");
        let entry = FileEntry::from_path(path.clone()).expect("from_path ok");
        assert_eq!(entry.file_type, FileType::Image);
        assert_eq!(entry.name, "image.png");
        assert_eq!(entry.size_bytes, 16);
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_file_entry_from_path_model() {
        let dir = make_temp_dir();
        let path = write_temp_file(&dir, "model.ply", b"HEADER\n");
        let entry = FileEntry::from_path(path).expect("from_path ok");
        assert_eq!(entry.file_type, FileType::Model);
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_file_entry_extension() {
        let dir = make_temp_dir();
        let path = write_temp_file(&dir, "config.json", b"{}");
        let entry = FileEntry::from_path(path).expect("from_path ok");
        assert_eq!(entry.extension(), "json");
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_file_entry_from_path_missing() {
        let path = PathBuf::from("/nonexistent_oxigaf_path/file.png");
        assert!(FileEntry::from_path(path).is_err());
    }

    // ------------------------------------------------------------------
    // 3. compute_dataset_stats
    // ------------------------------------------------------------------

    fn make_entry(name: &str, size: u64) -> FileEntry {
        let ext = name.rsplit('.').next().unwrap_or("");
        let file_type = FileType::from_extension(ext);
        FileEntry {
            path: PathBuf::from(name),
            file_type,
            size_bytes: size,
            name: name.to_string(),
        }
    }

    #[test]
    fn test_stats_empty() {
        let stats = compute_dataset_stats(&[]);
        assert_eq!(stats.total_files, 0);
        assert_eq!(stats.total_bytes, 0);
        assert_eq!(stats.mean_file_size_bytes, 0);
    }

    #[test]
    fn test_stats_all_images() {
        let entries = vec![
            make_entry("a.png", 100),
            make_entry("b.jpg", 200),
            make_entry("c.exr", 300),
        ];
        let stats = compute_dataset_stats(&entries);
        assert_eq!(stats.total_files, 3);
        assert_eq!(stats.image_count, 3);
        assert_eq!(stats.model_count, 0);
        assert_eq!(stats.total_bytes, 600);
        assert_eq!(stats.largest_file_bytes, 300);
        assert_eq!(stats.smallest_file_bytes, 100);
        assert_eq!(stats.mean_file_size_bytes, 200);
    }

    #[test]
    fn test_stats_mixed_types() {
        let entries = vec![
            make_entry("img.png", 50),
            make_entry("model.ply", 1000),
            make_entry("cfg.json", 20),
            make_entry("clip.mp4", 5000),
            make_entry("data.xyz", 10),
        ];
        let stats = compute_dataset_stats(&entries);
        assert_eq!(stats.image_count, 1);
        assert_eq!(stats.model_count, 1);
        assert_eq!(stats.config_count, 1);
        assert_eq!(stats.video_count, 1);
        assert_eq!(stats.unknown_count, 1);
    }

    #[test]
    fn test_stats_size_bounds() {
        let entries = vec![make_entry("a.png", 10), make_entry("b.png", 90)];
        let stats = compute_dataset_stats(&entries);
        assert_eq!(stats.smallest_file_bytes, 10);
        assert_eq!(stats.largest_file_bytes, 90);
    }

    #[test]
    fn test_stats_single_file() {
        let entries = vec![make_entry("solo.png", 42)];
        let stats = compute_dataset_stats(&entries);
        assert_eq!(stats.mean_file_size_bytes, 42);
        assert_eq!(stats.largest_file_bytes, 42);
        assert_eq!(stats.smallest_file_bytes, 42);
    }

    // ------------------------------------------------------------------
    // 4. split_dataset
    // ------------------------------------------------------------------

    #[test]
    fn test_split_basic() {
        let config = SplitConfig::default();
        let split = split_dataset(100, &config).expect("split ok");
        assert!(split.is_valid());
        assert_eq!(split.total_items, 100);
        // train=80, val=10, test=10
        assert_eq!(split.train_count(), 80);
        assert_eq!(split.val_count(), 10);
        assert_eq!(split.test_count(), 10);
    }

    #[test]
    fn test_split_invalid_ratios() {
        let config = SplitConfig {
            train_ratio: 0.7,
            val_ratio: 0.2,
            test_ratio: 0.2,
            seed: 1,
            shuffle: true,
            strategy: DatasetSplitStrategy::Random,
        };
        assert!(matches!(
            split_dataset(10, &config),
            Err(DatasetError::InvalidSplitRatios { .. })
        ));
    }

    #[test]
    fn test_split_n_zero() {
        let config = SplitConfig::default();
        let split = split_dataset(0, &config).expect("split ok");
        assert_eq!(split.train_count(), 0);
        assert_eq!(split.val_count(), 0);
        assert_eq!(split.test_count(), 0);
        assert_eq!(split.total_items, 0);
    }

    #[test]
    fn test_split_n_one() {
        let config = SplitConfig::default();
        let split = split_dataset(1, &config).expect("split ok");
        assert_eq!(split.total_items, 1);
        // Total should always cover all items.
        let covered = split.train_count() + split.val_count() + split.test_count();
        assert_eq!(covered, 1);
    }

    #[test]
    fn test_split_n_100_all_covered() {
        let config = SplitConfig::default();
        let split = split_dataset(100, &config).expect("split ok");
        let covered = split.train_count() + split.val_count() + split.test_count();
        assert_eq!(covered, 100);
    }

    #[test]
    fn test_split_deterministic() {
        let config = SplitConfig::default();
        let a = split_dataset(50, &config).expect("split ok");
        let b = split_dataset(50, &config).expect("split ok");
        assert_eq!(a.train_indices, b.train_indices);
        assert_eq!(a.val_indices, b.val_indices);
        assert_eq!(a.test_indices, b.test_indices);
    }

    #[test]
    fn test_split_different_seeds() {
        let config_a = SplitConfig {
            seed: 1,
            ..Default::default()
        };
        let config_b = SplitConfig {
            seed: 999,
            ..Default::default()
        };
        let a = split_dataset(100, &config_a).expect("split ok");
        let b = split_dataset(100, &config_b).expect("split ok");
        // With different seeds, train sets should differ (extremely high probability).
        assert_ne!(a.train_indices, b.train_indices);
    }

    #[test]
    fn test_split_sequential_no_shuffle() {
        let config = SplitConfig {
            shuffle: false,
            ..Default::default()
        };
        let split = split_dataset(10, &config).expect("split ok");
        // With no shuffle, train gets indices 0..7 (floor(10*0.8)=8)
        assert_eq!(split.train_indices, vec![0, 1, 2, 3, 4, 5, 6, 7]);
    }

    // Regression coverage for: `DatasetSplitStrategy` was defined and
    // re-exported but `SplitConfig` had no field for it, so it could never
    // be read -- selecting `Sequential` had no effect at all.
    #[test]
    fn test_split_sequential_strategy_ignores_shuffle_flag() {
        let config = SplitConfig {
            strategy: DatasetSplitStrategy::Sequential,
            shuffle: true, // must be ignored by Sequential
            seed: 12345,
            ..Default::default()
        };
        let split = split_dataset(10, &config).expect("split ok");
        assert_eq!(split.train_indices, vec![0, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(split.val_indices, vec![8]);
        assert_eq!(split.test_indices, vec![9]);
    }

    #[test]
    fn test_split_random_strategy_is_default_and_shuffles() {
        let config = SplitConfig {
            strategy: DatasetSplitStrategy::Random,
            shuffle: true,
            seed: 7,
            ..Default::default()
        };
        let split = split_dataset(20, &config).expect("split ok");
        assert_ne!(
            split.train_indices,
            vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            "Random with shuffle=true should (almost certainly) not be sequential order"
        );
    }

    // Regression coverage for: the `Stratified` variant's doc claimed it
    // "falls back to Random", but no code read the enum at all, so
    // selecting it silently produced no fallback and no error. It must now
    // actually behave like the documented fallback when driven through
    // `split_dataset` (which has no per-item labels to truly stratify on).
    #[test]
    fn test_split_stratified_via_split_dataset_falls_back_sanely() {
        let config = SplitConfig {
            strategy: DatasetSplitStrategy::Stratified,
            shuffle: false,
            ..Default::default()
        };
        let split = split_dataset(10, &config).expect("split ok");
        let covered = split.train_count() + split.val_count() + split.test_count();
        assert_eq!(covered, 10, "fallback must still produce a complete split");
        assert!(validate_split(&split).is_ok());
    }

    // -----------------------------------------------------------------------
    // split_dataset_stratified: real per-group stratified splitting
    // -----------------------------------------------------------------------

    #[test]
    fn test_split_dataset_stratified_keeps_ratio_within_each_group() {
        // Two groups of 10 items each ("a" and "b"); each group's 80/10/10
        // split should be computed independently, not against the pooled 20.
        let keys: Vec<&str> = (0..10).map(|_| "a").chain((0..10).map(|_| "b")).collect();
        let config = SplitConfig::default();
        let split = split_dataset_stratified(&keys, &config).expect("split ok");

        assert_eq!(split.total_items, 20);
        assert_eq!(split.train_count(), 16); // 8 from each group
        assert_eq!(split.val_count(), 2); // 1 from each group
        assert_eq!(split.test_count(), 2); // 1 from each group
        assert!(validate_split(&split).is_ok());

        // Every train index must resolve to a real key, and both groups
        // must be represented in the train set (not just the larger-index
        // group), proving the split was computed per-group.
        let group_a_in_train = split
            .train_indices
            .iter()
            .filter(|&&i| keys[i] == "a")
            .count();
        let group_b_in_train = split
            .train_indices
            .iter()
            .filter(|&&i| keys[i] == "b")
            .count();
        assert_eq!(group_a_in_train, 8);
        assert_eq!(group_b_in_train, 8);
    }

    #[test]
    fn test_split_dataset_stratified_empty_keys() {
        let keys: Vec<&str> = vec![];
        let config = SplitConfig::default();
        let split = split_dataset_stratified(&keys, &config).expect("split ok");
        assert_eq!(split.total_items, 0);
    }

    #[test]
    fn test_split_dataset_stratified_invalid_ratios() {
        let keys = vec!["a", "b", "c"];
        let config = SplitConfig {
            train_ratio: 0.5,
            val_ratio: 0.5,
            test_ratio: 0.5,
            ..Default::default()
        };
        assert!(matches!(
            split_dataset_stratified(&keys, &config),
            Err(DatasetError::InvalidSplitRatios { .. })
        ));
    }

    #[test]
    fn test_split_dataset_stratified_sequential_ignores_shuffle() {
        let keys: Vec<&str> = (0..10).map(|_| "a").collect();
        let config = SplitConfig {
            strategy: DatasetSplitStrategy::Sequential,
            shuffle: true,
            ..Default::default()
        };
        let split = split_dataset_stratified(&keys, &config).expect("split ok");
        assert_eq!(split.train_indices, vec![0, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn test_split_valid_after_creation() {
        let config = SplitConfig::default();
        let split = split_dataset(73, &config).expect("split ok");
        assert!(validate_split(&split).is_ok());
    }

    // ------------------------------------------------------------------
    // 5. shuffle_indices
    // ------------------------------------------------------------------

    #[test]
    fn test_shuffle_same_seed_same_result() {
        let mut a: Vec<usize> = (0..20).collect();
        let mut b: Vec<usize> = (0..20).collect();
        shuffle_indices(&mut a, 42);
        shuffle_indices(&mut b, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn test_shuffle_different_seeds() {
        let mut a: Vec<usize> = (0..20).collect();
        let mut b: Vec<usize> = (0..20).collect();
        shuffle_indices(&mut a, 1);
        shuffle_indices(&mut b, 9999);
        assert_ne!(a, b);
    }

    #[test]
    fn test_shuffle_preserves_elements() {
        let mut indices: Vec<usize> = (0..50).collect();
        shuffle_indices(&mut indices, 77);
        let mut sorted = indices.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..50).collect::<Vec<_>>());
    }

    // ------------------------------------------------------------------
    // 6. validate_split
    // ------------------------------------------------------------------

    #[test]
    fn test_validate_split_valid() {
        let split = DatasetSplit {
            train_indices: vec![0, 1, 2],
            val_indices: vec![3, 4],
            test_indices: vec![5],
            total_items: 6,
        };
        assert!(validate_split(&split).is_ok());
    }

    #[test]
    fn test_validate_split_duplicate_in_train() {
        let split = DatasetSplit {
            train_indices: vec![0, 0, 1],
            val_indices: vec![2],
            test_indices: vec![3],
            total_items: 4,
        };
        assert!(matches!(
            validate_split(&split),
            Err(DatasetError::SplitValidationFailed { .. })
        ));
    }

    #[test]
    fn test_validate_split_out_of_range() {
        let split = DatasetSplit {
            train_indices: vec![0, 1, 99],
            val_indices: vec![2],
            test_indices: vec![3],
            total_items: 4,
        };
        assert!(matches!(
            validate_split(&split),
            Err(DatasetError::SplitValidationFailed { .. })
        ));
    }

    #[test]
    fn test_validate_split_overlap() {
        let split = DatasetSplit {
            train_indices: vec![0, 1],
            val_indices: vec![1, 2],
            test_indices: vec![3],
            total_items: 4,
        };
        assert!(matches!(
            validate_split(&split),
            Err(DatasetError::SplitValidationFailed { .. })
        ));
    }

    // ------------------------------------------------------------------
    // 7. save_split / load_split
    // ------------------------------------------------------------------

    #[test]
    fn test_split_round_trip() {
        let dir = make_temp_dir();
        let split_path = dir.join("split.json");

        let split = DatasetSplit {
            train_indices: vec![0, 1, 2, 3],
            val_indices: vec![4, 5],
            test_indices: vec![6, 7],
            total_items: 8,
        };
        save_split(&split, &split_path).expect("save ok");
        let loaded = load_split(&split_path).expect("load ok");

        assert_eq!(loaded.train_indices, split.train_indices);
        assert_eq!(loaded.val_indices, split.val_indices);
        assert_eq!(loaded.test_indices, split.test_indices);
        assert_eq!(loaded.total_items, split.total_items);
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_load_split_missing_file() {
        assert!(load_split(Path::new("/nonexistent/split.json")).is_err());
    }

    #[test]
    fn test_split_round_trip_large() {
        let dir = make_temp_dir();
        let split_path = dir.join("large_split.json");

        let config = SplitConfig::default();
        let split = split_dataset(1000, &config).expect("split ok");
        save_split(&split, &split_path).expect("save ok");
        let loaded = load_split(&split_path).expect("load ok");
        assert_eq!(loaded.total_items, 1000);
        assert_eq!(loaded.train_indices.len(), split.train_indices.len());
        fs::remove_dir_all(&dir).ok();
    }

    // ------------------------------------------------------------------
    // 8. DatasetSplit methods
    // ------------------------------------------------------------------

    #[test]
    fn test_dataset_split_counts() {
        let split = DatasetSplit {
            train_indices: (0..8).collect(),
            val_indices: vec![8, 9],
            test_indices: vec![10],
            total_items: 11,
        };
        assert_eq!(split.train_count(), 8);
        assert_eq!(split.val_count(), 2);
        assert_eq!(split.test_count(), 1);
    }

    #[test]
    fn test_dataset_split_is_valid_true() {
        let config = SplitConfig::default();
        let split = split_dataset(20, &config).expect("split ok");
        assert!(split.is_valid());
    }

    #[test]
    fn test_dataset_split_is_valid_false() {
        let split = DatasetSplit {
            train_indices: vec![0, 0],
            val_indices: vec![1],
            test_indices: vec![2],
            total_items: 3,
        };
        assert!(!split.is_valid());
    }

    #[test]
    fn test_dataset_split_format_summary() {
        let split = DatasetSplit {
            train_indices: vec![0, 1, 2, 3, 4, 5, 6, 7],
            val_indices: vec![8, 9],
            test_indices: vec![10],
            total_items: 11,
        };
        let summary = split.format_summary();
        assert!(summary.contains("11"));
        assert!(summary.contains("train=8"));
    }

    // ------------------------------------------------------------------
    // 9. SplitConfig defaults
    // ------------------------------------------------------------------

    #[test]
    fn test_split_config_default_ratios() {
        let cfg = SplitConfig::default();
        assert!((cfg.train_ratio - 0.8).abs() < 1e-6);
        assert!((cfg.val_ratio - 0.1).abs() < 1e-6);
        assert!((cfg.test_ratio - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_split_config_default_seed() {
        let cfg = SplitConfig::default();
        assert_eq!(cfg.seed, 42);
        assert!(cfg.shuffle);
    }

    // ------------------------------------------------------------------
    // 10. compute_split_stats
    // ------------------------------------------------------------------

    #[test]
    fn test_compute_split_stats_fractions_sum_to_one() {
        let config = SplitConfig::default();
        let split = split_dataset(100, &config).expect("split ok");
        let stats = compute_split_stats(&split);
        let sum = stats.train_fraction + stats.val_fraction + stats.test_fraction;
        assert!((sum - 1.0).abs() < 1e-4, "fractions sum={}", sum);
    }

    #[test]
    fn test_compute_split_stats_counts_match() {
        let config = SplitConfig::default();
        let split = split_dataset(50, &config).expect("split ok");
        let stats = compute_split_stats(&split);
        assert_eq!(stats.train_count, split.train_count());
        assert_eq!(stats.val_count, split.val_count());
        assert_eq!(stats.test_count, split.test_count());
    }

    // ------------------------------------------------------------------
    // 11. find_size_duplicates
    // ------------------------------------------------------------------

    #[test]
    fn test_find_size_duplicates_none() {
        let entries = vec![
            make_entry("a.png", 10),
            make_entry("b.png", 20),
            make_entry("c.png", 30),
        ];
        let dupes = find_size_duplicates(&entries);
        assert!(dupes.is_empty());
    }

    #[test]
    fn test_find_size_duplicates_one_group() {
        let entries = vec![
            make_entry("a.png", 100),
            make_entry("b.png", 100),
            make_entry("c.png", 200),
        ];
        let dupes = find_size_duplicates(&entries);
        assert_eq!(dupes.len(), 1);
        assert_eq!(dupes[0].len(), 2);
        assert!(dupes[0].contains(&0));
        assert!(dupes[0].contains(&1));
    }

    #[test]
    fn test_find_size_duplicates_multiple_groups() {
        let entries = vec![
            make_entry("a.png", 100),
            make_entry("b.png", 100),
            make_entry("c.png", 200),
            make_entry("d.png", 200),
            make_entry("e.png", 300),
        ];
        let dupes = find_size_duplicates(&entries);
        assert_eq!(dupes.len(), 2);
    }

    // ------------------------------------------------------------------
    // 12. filter_by_size
    // ------------------------------------------------------------------

    #[test]
    fn test_filter_by_size_basic() {
        let entries = vec![
            make_entry("a.png", 10),
            make_entry("b.png", 50),
            make_entry("c.png", 100),
        ];
        let filtered = filter_by_size(&entries, 50);
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn test_filter_by_size_zero_threshold() {
        let entries = vec![make_entry("a.png", 0), make_entry("b.png", 1)];
        let filtered = filter_by_size(&entries, 0);
        assert_eq!(filtered.len(), 2);
    }

    // ------------------------------------------------------------------
    // 13. DatasetScanner
    // ------------------------------------------------------------------

    #[test]
    fn test_scanner_basic_scan() {
        let dir = make_temp_dir();
        write_temp_file(&dir, "img.png", b"PNG");
        write_temp_file(&dir, "model.ply", b"PLY");
        write_temp_file(&dir, "cfg.json", b"{}");

        let scanner = DatasetScanner::new();
        let entries = scanner.scan(&dir).expect("scan ok");
        assert_eq!(entries.len(), 3);
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_scanner_extension_filter() {
        let dir = make_temp_dir();
        write_temp_file(&dir, "img.png", b"PNG");
        write_temp_file(&dir, "model.ply", b"PLY");

        let scanner = DatasetScanner::new().with_extensions(vec!["png"]);
        let entries = scanner.scan(&dir).expect("scan ok");
        assert_eq!(entries.len(), 1);
        assert!(entries[0].name.ends_with(".png"));
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_scanner_recursive() {
        let dir = make_temp_dir();
        let sub = dir.join("sub");
        fs::create_dir_all(&sub).expect("create sub");
        write_temp_file(&dir, "top.png", b"T");
        write_temp_file(&sub, "nested.png", b"N");

        let scanner = DatasetScanner::new().with_recursive(true);
        let entries = scanner.scan(&dir).expect("scan ok");
        assert_eq!(entries.len(), 2);
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_scanner_non_recursive() {
        let dir = make_temp_dir();
        let sub = dir.join("sub");
        fs::create_dir_all(&sub).expect("create sub");
        write_temp_file(&dir, "top.png", b"T");
        write_temp_file(&sub, "nested.png", b"N");

        let scanner = DatasetScanner::new().with_recursive(false);
        let entries = scanner.scan(&dir).expect("scan ok");
        assert_eq!(entries.len(), 1);
        fs::remove_dir_all(&dir).ok();
    }

    // Regression coverage for: `DirEntry::metadata` does not follow
    // symlinks, so a symlinked file used to be silently dropped (matched
    // neither `is_dir()` nor `is_file()`) instead of being scanned like a
    // real file.
    #[cfg(unix)]
    #[test]
    fn test_scanner_follows_symlinked_file() {
        use std::os::unix::fs::symlink;

        let dir = make_temp_dir();
        let real_dir = make_temp_dir();
        let real_file = write_temp_file(&real_dir, "real.png", b"real");
        symlink(&real_file, dir.join("linked.png")).expect("create symlink");

        let scanner = DatasetScanner::new();
        let entries = scanner.scan(&dir).expect("scan ok");
        assert_eq!(
            entries.len(),
            1,
            "symlinked file must be scanned, not silently skipped"
        );
        assert_eq!(entries[0].name, "linked.png");

        fs::remove_dir_all(&dir).ok();
        fs::remove_dir_all(&real_dir).ok();
    }

    // Regression coverage for the same bug, but for a symlinked *directory*
    // (a common way to assemble a dataset from frames/subdirs living
    // elsewhere without copying them).
    #[cfg(unix)]
    #[test]
    fn test_scanner_follows_symlinked_directory() {
        use std::os::unix::fs::symlink;

        let dir = make_temp_dir();
        let real_dir = make_temp_dir();
        write_temp_file(&real_dir, "nested.png", b"nested");
        symlink(&real_dir, dir.join("linked_dir")).expect("create symlink");

        let scanner = DatasetScanner::new().with_recursive(true);
        let entries = scanner.scan(&dir).expect("scan ok");
        assert_eq!(
            entries.len(),
            1,
            "file inside a symlinked directory must be found via recursion"
        );

        fs::remove_dir_all(&dir).ok();
        fs::remove_dir_all(&real_dir).ok();
    }

    // A symlink cycle (a directory symlinking back to one of its own
    // ancestors) must terminate instead of recursing forever.
    #[cfg(unix)]
    #[test]
    fn test_scanner_symlink_cycle_terminates() {
        use std::os::unix::fs::symlink;

        let dir = make_temp_dir();
        write_temp_file(&dir, "top.png", b"top");
        // `dir/self_link` -> `dir` (a direct cycle back to the scan root).
        symlink(&dir, dir.join("self_link")).expect("create symlink cycle");

        let scanner = DatasetScanner::new().with_recursive(true);
        let result = scanner.scan(&dir);
        assert!(result.is_ok(), "a symlink cycle must not error the scan");
        let entries = result.expect("scan ok");
        assert_eq!(
            entries.len(),
            1,
            "the cycle must not be traversed more than once (found: {:?})",
            entries.iter().map(|e| &e.name).collect::<Vec<_>>()
        );

        fs::remove_dir_all(&dir).ok();
    }

    // ------------------------------------------------------------------
    // 14. format_stats_table
    // ------------------------------------------------------------------

    #[test]
    fn test_format_stats_table_non_empty() {
        let entries = vec![make_entry("a.png", 100), make_entry("b.ply", 2000)];
        let stats = compute_dataset_stats(&entries);
        let table = format_stats_table(&stats);
        assert!(!table.is_empty());
        assert!(table.contains("Images"));
        assert!(table.contains("Models"));
    }

    // ------------------------------------------------------------------
    // 15. apply_split
    // ------------------------------------------------------------------

    #[test]
    fn test_apply_split_basic() {
        let dir = make_temp_dir();
        let paths: Vec<PathBuf> = (0..10)
            .map(|i| {
                let p = write_temp_file(&dir, &format!("{}.png", i), b"DATA");
                p
            })
            .collect();
        let entries: Vec<FileEntry> = paths
            .into_iter()
            .map(|p| FileEntry::from_path(p).expect("from_path"))
            .collect();

        let config = SplitConfig {
            shuffle: false,
            ..Default::default()
        };
        let split = split_dataset(10, &config).expect("split ok");
        let (train, val, test) = apply_split(&entries, &split).expect("apply ok");

        assert_eq!(train.len(), 8);
        assert_eq!(val.len(), 1);
        assert_eq!(test.len(), 1);
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_apply_split_total_items_mismatch() {
        let entry = make_entry("a.png", 10);
        let entries = [entry];
        let split = DatasetSplit {
            train_indices: vec![0],
            val_indices: vec![],
            test_indices: vec![],
            total_items: 5, // mismatch: entries has 1 item
        };
        let result = apply_split(&entries, &split);
        assert!(result.is_err());
    }

    // ------------------------------------------------------------------
    // 16. validate_dataset
    // ------------------------------------------------------------------

    #[test]
    fn test_validate_dataset_ok() {
        let dir = make_temp_dir();
        write_temp_file(&dir, "a.png", b"AAA");
        write_temp_file(&dir, "b.png", b"BBB");
        let stats = validate_dataset(&dir, 1).expect("validate ok");
        assert_eq!(stats.total_files, 2);
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_validate_dataset_too_small() {
        let dir = make_temp_dir();
        write_temp_file(&dir, "a.png", b"AAA");
        let result = validate_dataset(&dir, 5);
        assert!(matches!(result, Err(DatasetError::TooSmall { .. })));
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_validate_dataset_missing_dir() {
        let result = validate_dataset(Path::new("/nonexistent_oxigaf_dir"), 1);
        assert!(matches!(
            result,
            Err(DatasetError::DirectoryNotFound { .. })
        ));
    }

    #[test]
    fn test_validate_dataset_empty() {
        let dir = make_temp_dir();
        let result = validate_dataset(&dir, 0);
        assert!(matches!(result, Err(DatasetError::EmptyDataset { .. })));
        fs::remove_dir_all(&dir).ok();
    }

    // ------------------------------------------------------------------
    // 17. DatasetStats::format_summary / total_mb
    // ------------------------------------------------------------------

    #[test]
    fn test_stats_format_summary_contains_total() {
        let entries = vec![make_entry("a.png", 1_000_000)];
        let stats = compute_dataset_stats(&entries);
        let s = stats.format_summary();
        assert!(s.contains("1 files"));
        assert!(s.contains("1.00 MB"));
    }

    #[test]
    fn test_stats_total_mb() {
        let entries = vec![make_entry("a.png", 2_500_000)];
        let stats = compute_dataset_stats(&entries);
        assert!((stats.total_mb() - 2.5).abs() < 0.001);
    }

    // ------------------------------------------------------------------
    // 18. scan_by_type
    // ------------------------------------------------------------------

    #[test]
    fn test_scan_by_type_images_only() {
        let dir = make_temp_dir();
        write_temp_file(&dir, "img.png", b"PNG");
        write_temp_file(&dir, "model.ply", b"PLY");
        write_temp_file(&dir, "vid.mp4", b"MP4");

        let scanner = DatasetScanner::new();
        let entries = scanner
            .scan_by_type(&dir, FileType::Image)
            .expect("scan ok");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].file_type, FileType::Image);
        fs::remove_dir_all(&dir).ok();
    }
}
