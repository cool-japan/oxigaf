//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::{HashMap, HashSet};
use std::path::Path;

use super::type_aliases::SplitResult;
use super::types::{
    DatasetError, DatasetScanner, DatasetSplit, DatasetSplitStrategy, DatasetStats, FileEntry,
    FileType, SplitConfig, SplitStats,
};

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
