//! Auto-generated test module (consolidated from inline `#[cfg(test)] mod` blocks)

use std::path::{Path, PathBuf};

use super::*;

#[cfg(test)]
mod tests_2 {
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

    /// Per-process counter appended to every [`make_temp_dir`] path on top
    /// of the pid+nanos suffix.
    ///
    /// Regression: `SystemTime` resolution is not guaranteed finer than the
    /// gap between two back-to-back calls in the same process — several
    /// tests (e.g. `test_scanner_follows_symlinked_file` and
    /// `test_scanner_follows_symlinked_directory`) call `make_temp_dir`
    /// twice in a row to get a `dir` and a `real_dir`. If both calls landed
    /// in the same `subsec_nanos()` tick, they collided on the identical
    /// path — one directory instead of two, and the symlink created inside
    /// it pointed at itself. A monotonic counter can't repeat within a
    /// process no matter how coarse the clock is.
    static TEMP_DIR_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

    fn make_temp_dir() -> PathBuf {
        let mut dir = std::env::temp_dir();
        let counter = TEMP_DIR_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        dir.push(format!(
            "oxigaf_dataset_test_{}_{}_{counter}",
            std::process::id(),
            {
                use std::time::{SystemTime, UNIX_EPOCH};
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|d| d.subsec_nanos())
                    .unwrap_or(0)
            }
        ));
        fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    /// Regression: two calls made back-to-back must never return the same
    /// path, even if the system clock's resolution is coarser than the gap
    /// between them — see [`TEMP_DIR_COUNTER`]. Runs a generous number of
    /// consecutive calls (not just two) since a clock-only nonce is more
    /// likely to repeat the tighter the calls are packed together.
    #[test]
    fn make_temp_dir_never_repeats_across_consecutive_calls() {
        let dirs: Vec<PathBuf> = (0..50).map(|_| make_temp_dir()).collect();
        let mut unique = dirs.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(
            unique.len(),
            dirs.len(),
            "make_temp_dir produced a duplicate path among: {dirs:?}"
        );
        for dir in dirs {
            fs::remove_dir_all(&dir).ok();
        }
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
