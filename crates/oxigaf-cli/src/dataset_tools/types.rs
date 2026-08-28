//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use thiserror::Error;

use super::constants::MAX_SCAN_DEPTH;
use super::functions::validate_split;

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
    /// See [`split_dataset`](crate::dataset_tools::split_dataset) for the
    /// exact semantics of each variant.
    pub strategy: DatasetSplitStrategy,
}
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
    /// [`split_dataset`](crate::dataset_tools::split_dataset) takes only an
    /// item *count* (`n: usize`), with no per-item labels to stratify on;
    /// requesting this strategy through that entry point logs a warning and
    /// falls back to the same behaviour as [`DatasetSplitStrategy::Random`].
    /// Callers that have a stratum key per item should call
    /// [`split_dataset_stratified`](crate::dataset_tools::split_dataset_stratified)
    /// directly instead, which performs a real per-group split.
    Stratified,
}
/// Configurable directory scanner that produces [`FileEntry`] lists.
#[derive(Debug)]
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
    /// canonicalised visited-path set plus `MAX_SCAN_DEPTH`.
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
    pub(super) fn scan_dir(
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
                return Ok(());
            }
        }
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            let mut meta = entry.metadata()?;
            if meta.file_type().is_symlink() {
                match std::fs::metadata(&path) {
                    Ok(target_meta) => meta = target_meta,
                    Err(_) => continue,
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
            if !self.extensions.is_empty() {
                let ext = path
                    .extension()
                    .map(|s| s.to_string_lossy().to_lowercase())
                    .unwrap_or_default();
                if !self.extensions.contains(&ext) {
                    continue;
                }
            }
            let size = meta.len();
            if size < self.min_size_bytes {
                continue;
            }
            if self.max_size_bytes > 0 && size > self.max_size_bytes {
                continue;
            }
            if let Ok(fe) = FileEntry::from_path(path) {
                out.push(fe);
            }
        }
        Ok(())
    }
}
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
