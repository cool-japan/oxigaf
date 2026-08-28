//! `oxigaf dataset` — dataset scanning, statistics, splitting and validation.
//!
//! Glue over [`crate::dataset_tools`].

use std::path::{Path, PathBuf};

use anyhow::Result;
use clap::{Args, Subcommand, ValueEnum};
use serde_json::json;

use crate::commands::{emit, prepare_output, CmdContext};
use crate::dataset_tools::{
    apply_split, compute_dataset_stats, compute_split_stats, filter_by_size, find_size_duplicates,
    format_stats_table, load_split, save_split, split_dataset, validate_dataset, validate_split,
    DatasetScanner, DatasetSplitStrategy, DatasetStats, FileEntry, SplitConfig,
};

/// `oxigaf dataset <command>`.
#[derive(Debug, Args)]
pub struct DatasetArgs {
    #[command(subcommand)]
    pub command: DatasetCommand,
}

/// Dataset subcommands.
#[derive(Debug, Subcommand)]
pub enum DatasetCommand {
    /// List every file the scanner finds, with type and size.
    Scan(ScanArgs),

    /// Summarise a dataset directory (counts, sizes, extremes).
    Stats(ScanArgs),

    /// Fail unless the directory holds at least `--min-files` files.
    Validate {
        /// Dataset directory.
        dir: PathBuf,
        /// Minimum acceptable file count.
        #[arg(long, default_value = "1")]
        min_files: usize,
    },

    /// Partition a dataset into train / validation / test index lists.
    Split(SplitArgs),

    /// Report groups of files that share an identical byte size.
    Duplicates(ScanArgs),
}

/// Scan options shared by `scan`, `stats` and `duplicates`.
#[derive(Debug, Args, Clone)]
pub struct ScanArgs {
    /// Dataset directory.
    pub dir: PathBuf,

    /// Only keep files with these extensions (without the leading dot).
    #[arg(long, num_args = 1..)]
    pub ext: Vec<String>,

    /// Do not descend into subdirectories.
    #[arg(long)]
    pub no_recursive: bool,

    /// Ignore files smaller than this many bytes.
    #[arg(long, default_value = "0")]
    pub min_bytes: u64,
}

/// Ordering strategy exposed on the command line.
#[derive(Debug, Clone, Copy, ValueEnum, Default)]
pub enum SplitStrategyArg {
    /// Shuffle, then take contiguous ranges.
    #[default]
    Random,
    /// Keep scan order and take contiguous ranges.
    Sequential,
}

impl From<SplitStrategyArg> for DatasetSplitStrategy {
    fn from(value: SplitStrategyArg) -> Self {
        match value {
            SplitStrategyArg::Random => DatasetSplitStrategy::Random,
            SplitStrategyArg::Sequential => DatasetSplitStrategy::Sequential,
        }
    }
}

/// Arguments for `oxigaf dataset split`.
#[derive(Debug, Args)]
pub struct SplitArgs {
    #[command(flatten)]
    pub scan: ScanArgs,

    /// Fraction assigned to the training set.
    #[arg(long, default_value = "0.8")]
    pub train: f32,

    /// Fraction assigned to the validation set.
    #[arg(long, default_value = "0.1")]
    pub val: f32,

    /// Fraction assigned to the test set.
    #[arg(long, default_value = "0.1")]
    pub test: f32,

    /// PRNG seed used when shuffling.
    #[arg(long, default_value = "42")]
    pub seed: u64,

    /// Ordering strategy.
    #[arg(long, value_enum, default_value = "random")]
    pub strategy: SplitStrategyArg,

    /// Write the split index lists here (JSON).
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Verify an existing split file against the scanned dataset instead of
    /// creating a new one. The ratio flags are ignored in this mode.
    #[arg(long, conflicts_with = "output")]
    pub verify: Option<PathBuf>,

    /// Also list the file names assigned to each subset.
    #[arg(long)]
    pub list_files: bool,

    /// Overwrite the output file if it exists.
    #[arg(long)]
    pub force: bool,
}

fn scan(args: &ScanArgs) -> Result<Vec<FileEntry>> {
    if !args.dir.is_dir() {
        anyhow::bail!("Not a directory: {}", args.dir.display());
    }
    let mut scanner = DatasetScanner::new().with_recursive(!args.no_recursive);
    if !args.ext.is_empty() {
        let refs: Vec<&str> = args.ext.iter().map(String::as_str).collect();
        scanner = scanner.with_extensions(refs);
    }
    let entries = scanner.scan(&args.dir)?;
    if args.min_bytes > 0 {
        Ok(filter_by_size(&entries, args.min_bytes)
            .into_iter()
            .cloned()
            .collect())
    } else {
        Ok(entries)
    }
}

fn stats_json(stats: &DatasetStats) -> serde_json::Value {
    json!({
        "total_files": stats.total_files,
        "image_count": stats.image_count,
        "model_count": stats.model_count,
        "config_count": stats.config_count,
        "video_count": stats.video_count,
        "unknown_count": stats.unknown_count,
        "total_bytes": stats.total_bytes,
        "total_mb": stats.total_mb(),
        "mean_file_size_bytes": stats.mean_file_size_bytes,
        "largest_file_bytes": stats.largest_file_bytes,
        "smallest_file_bytes": stats.smallest_file_bytes,
    })
}

/// Run the `dataset` family.
///
/// # Errors
///
/// Returns an error when the directory cannot be scanned, when validation
/// fails, or when a split file cannot be read or written.
pub fn run(args: DatasetArgs, ctx: CmdContext) -> Result<()> {
    match args.command {
        DatasetCommand::Scan(scan_args) => {
            let entries = scan(&scan_args)?;
            let payload = json!({
                "directory": scan_args.dir.display().to_string(),
                "files": entries
                    .iter()
                    .map(|entry| json!({
                        "path": entry.path.display().to_string(),
                        "name": entry.name,
                        "file_type": format!("{:?}", entry.file_type),
                        "size_bytes": entry.size_bytes,
                    }))
                    .collect::<Vec<_>>(),
            });
            emit(&ctx, "dataset scan", payload, &[], || {
                for entry in &entries {
                    println!(
                        "{:>12}  {:<8}  {}",
                        entry.size_bytes,
                        format!("{:?}", entry.file_type),
                        entry.path.display()
                    );
                }
                println!("{} file(s)", entries.len());
            });
            Ok(())
        }

        DatasetCommand::Stats(scan_args) => {
            let entries = scan(&scan_args)?;
            let stats = compute_dataset_stats(&entries);
            let payload = stats_json(&stats);
            emit(&ctx, "dataset stats", payload, &[], || {
                println!("{}", format_stats_table(&stats));
            });
            Ok(())
        }

        DatasetCommand::Validate { dir, min_files } => {
            let stats = validate_dataset(&dir, min_files)?;
            let payload = stats_json(&stats);
            emit(&ctx, "dataset validate", payload, &[], || {
                println!("{}", stats.format_summary());
                println!("OK: at least {min_files} file(s) present");
            });
            Ok(())
        }

        DatasetCommand::Split(split_args) => cmd_split(split_args, &ctx),

        DatasetCommand::Duplicates(scan_args) => {
            let entries = scan(&scan_args)?;
            let groups = find_size_duplicates(&entries);
            let payload = json!({
                "directory": scan_args.dir.display().to_string(),
                "groups": groups
                    .iter()
                    .map(|group| json!({
                        "size_bytes": group.first().map(|i| entries[*i].size_bytes),
                        "files": group
                            .iter()
                            .map(|i| entries[*i].path.display().to_string())
                            .collect::<Vec<_>>(),
                    }))
                    .collect::<Vec<_>>(),
            });
            emit(&ctx, "dataset duplicates", payload, &[], || {
                if groups.is_empty() {
                    println!(
                        "No same-size file groups found among {} file(s)",
                        entries.len()
                    );
                }
                for group in &groups {
                    let size = group.first().map(|i| entries[*i].size_bytes).unwrap_or(0);
                    println!("{} file(s) at {size} bytes:", group.len());
                    for index in group {
                        println!("  {}", entries[*index].path.display());
                    }
                }
            });
            Ok(())
        }
    }
}

fn cmd_split(args: SplitArgs, ctx: &CmdContext) -> Result<()> {
    let entries = scan(&args.scan)?;
    if entries.is_empty() {
        anyhow::bail!("No files found in {}", args.scan.dir.display());
    }

    let split = match args.verify {
        Some(ref path) => {
            let loaded = load_split(path)?;
            if loaded.total_items != entries.len() {
                anyhow::bail!(
                    "Split file covers {} item(s) but the directory holds {}",
                    loaded.total_items,
                    entries.len()
                );
            }
            validate_split(&loaded)?;
            loaded
        }
        None => {
            let config = SplitConfig {
                train_ratio: args.train,
                val_ratio: args.val,
                test_ratio: args.test,
                seed: args.seed,
                shuffle: matches!(args.strategy, SplitStrategyArg::Random),
                strategy: args.strategy.into(),
            };
            let produced = split_dataset(entries.len(), &config)?;
            validate_split(&produced)?;
            produced
        }
    };

    let split_stats = compute_split_stats(&split);
    let (train, val, test) = apply_split(&entries, &split)?;

    let mut payload = json!({
        "directory": args.scan.dir.display().to_string(),
        "total_items": split.total_items,
        "train_count": split_stats.train_count,
        "val_count": split_stats.val_count,
        "test_count": split_stats.test_count,
        "train_fraction": split_stats.train_fraction,
        "val_fraction": split_stats.val_fraction,
        "test_fraction": split_stats.test_fraction,
    });

    if args.list_files {
        if let Some(map) = payload.as_object_mut() {
            map.insert("train".to_string(), json!(names(&train)));
            map.insert("val".to_string(), json!(names(&val)));
            map.insert("test".to_string(), json!(names(&test)));
        }
    }

    let mut artifacts: Vec<(&str, &Path)> = Vec::new();
    if let Some(ref output) = args.output {
        if prepare_output(ctx, output, args.force)? {
            save_split(&split, output)?;
            artifacts.push(("split", output.as_path()));
        }
    }

    emit(ctx, "dataset split", payload, &artifacts, || {
        println!("{}", split.format_summary());
        if args.list_files {
            print_subset("train", &train);
            print_subset("val", &val);
            print_subset("test", &test);
        }
        if let Some(ref output) = args.output {
            if ctx.dry_run {
                println!("[dry-run] would write {}", output.display());
            } else {
                println!("Wrote {}", output.display());
            }
        }
    });
    Ok(())
}

fn names(entries: &[&FileEntry]) -> Vec<String> {
    entries.iter().map(|entry| entry.name.clone()).collect()
}

fn print_subset(label: &str, entries: &[&FileEntry]) {
    println!("{label} ({}):", entries.len());
    for entry in entries {
        println!("  {}", entry.name);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dataset(name: &str, files: usize) -> PathBuf {
        let dir = std::env::temp_dir().join(name);
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("temp dataset dir");
        for index in 0..files {
            let path = dir.join(format!("frame_{index:03}.png"));
            std::fs::write(&path, vec![index as u8; 16 + index]).expect("write sample file");
        }
        dir
    }

    #[test]
    fn scan_respects_min_bytes() {
        let dir = temp_dataset("oxigaf_dataset_scan", 4);
        let mut args = ScanArgs {
            dir: dir.clone(),
            ext: Vec::new(),
            no_recursive: true,
            min_bytes: 0,
        };
        assert_eq!(scan(&args).expect("scan").len(), 4);

        args.min_bytes = 18;
        // Files are 16, 17, 18, 19 bytes: only the last two survive.
        assert_eq!(scan(&args).expect("filtered scan").len(), 2);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn split_command_writes_and_verifies_a_split_file() {
        let dir = temp_dataset("oxigaf_dataset_split", 10);
        let out = std::env::temp_dir().join("oxigaf_dataset_split.json");
        let _ = std::fs::remove_file(&out);
        let ctx = CmdContext::new(crate::verbosity::Verbosity::Quiet, true, false);

        let args = SplitArgs {
            scan: ScanArgs {
                dir: dir.clone(),
                ext: Vec::new(),
                no_recursive: true,
                min_bytes: 0,
            },
            train: 0.6,
            val: 0.2,
            test: 0.2,
            seed: 7,
            strategy: SplitStrategyArg::Sequential,
            output: Some(out.clone()),
            verify: None,
            list_files: false,
            force: true,
        };
        cmd_split(args, &ctx).expect("split should succeed");
        assert!(out.exists(), "split file must be written");

        let loaded = load_split(&out).expect("split file reloads");
        assert_eq!(loaded.total_items, 10);
        assert_eq!(loaded.train_count(), 6);

        let _ = std::fs::remove_file(&out);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn strategy_flag_maps_onto_library_strategy() {
        assert_eq!(
            DatasetSplitStrategy::from(SplitStrategyArg::Sequential),
            DatasetSplitStrategy::Sequential
        );
    }
}
