use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::assets;

#[derive(Debug, Serialize, Deserialize)]
pub struct CacheMetadata {
    pub version: String,
    pub entries: Vec<CacheEntry>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct CacheEntry {
    pub name: String,
    pub path: PathBuf,
    pub size_bytes: u64,
    pub downloaded_at: u64,
    pub last_accessed: u64,
    pub access_count: u64,
    pub checksum: Option<String>,
}

impl CacheMetadata {
    /// Load cache metadata from cache directory
    pub fn load(cache_dir: &Path) -> Result<Self> {
        let metadata_path = cache_dir.join("cache.json");

        if !metadata_path.exists() {
            return Ok(Self {
                version: env!("CARGO_PKG_VERSION").to_string(),
                entries: Vec::new(),
            });
        }

        let data = std::fs::read_to_string(&metadata_path).with_context(|| {
            format!("Failed to read cache metadata: {}", metadata_path.display())
        })?;

        serde_json::from_str(&data).with_context(|| "Failed to parse cache metadata")
    }

    /// Save cache metadata to cache directory
    pub fn save(&self, cache_dir: &Path) -> Result<()> {
        // Ensure cache directory exists
        std::fs::create_dir_all(cache_dir).with_context(|| {
            format!("Failed to create cache directory: {}", cache_dir.display())
        })?;

        let metadata_path = cache_dir.join("cache.json");
        let data = serde_json::to_string_pretty(self)
            .with_context(|| "Failed to serialize cache metadata")?;

        std::fs::write(&metadata_path, data).with_context(|| {
            format!(
                "Failed to write cache metadata: {}",
                metadata_path.display()
            )
        })?;

        Ok(())
    }

    /// Update access timestamp and count for a cache entry
    pub fn update_access(&mut self, name: &str) {
        if let Some(entry) = self.entries.iter_mut().find(|e| e.name == name) {
            entry.last_accessed = current_timestamp();
            entry.access_count += 1;
        }
    }

    /// Calculate total size of all cached assets
    pub fn total_size(&self) -> u64 {
        self.entries.iter().map(|e| e.size_bytes).sum()
    }

    /// Insert a new entry, or overwrite the existing entry for the same
    /// `path` in place (preserving position/order for everything else).
    ///
    /// The incoming entry describes a *download*, so it carries no usage
    /// history: its `access_count` is 0 and its `last_accessed` is "now".
    /// Any history already recorded for that path is carried across rather
    /// than clobbered — re-downloading a corrupted file must not silently
    /// reset the access count that `cache clean`'s age policy and `cache
    /// list`'s usage column are derived from. `last_accessed` keeps
    /// whichever timestamp is later, since a fresh download *is* a fresh
    /// touch of the file but must never move a newer access backwards.
    fn upsert(&mut self, entry: CacheEntry) {
        if let Some(existing) = self.entries.iter_mut().find(|e| e.path == entry.path) {
            let access_count = existing.access_count.max(entry.access_count);
            let last_accessed = existing.last_accessed.max(entry.last_accessed);
            *existing = entry;
            existing.access_count = access_count;
            existing.last_accessed = last_accessed;
        } else {
            self.entries.push(entry);
        }
    }
}

/// Get current Unix timestamp in seconds
fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Record (or refresh) a [`CacheEntry`] for a downloaded asset.
///
/// The download paths in [`crate::assets`] call this immediately after each
/// successful transfer ([`crate::assets::setup_cache_with_options`] for
/// manifest assets, [`crate::assets::download_with_progress`] for
/// HuggingFace Hub files), so `cache.json` carries a precise
/// `downloaded_at` timestamp — and, when the asset had a published checksum
/// that was actually verified, that checksum — instead of having to infer
/// both from filesystem metadata later.
///
/// `discover_untracked_assets` remains as the fallback for asset files
/// that appeared in the cache directory by some other route (a manual copy,
/// a download by an older build).
///
/// Entries are keyed by `path`, so re-recording the same file overwrites its
/// entry in place rather than accumulating duplicates — while preserving the
/// usage history already recorded for it (`CacheMetadata::upsert` carries
/// `access_count` and the later `last_accessed` across).
pub fn record_download(
    cache_dir: &Path,
    name: &str,
    path: &Path,
    checksum: Option<String>,
) -> Result<()> {
    let mut metadata = CacheMetadata::load(cache_dir)?;
    let size_bytes = std::fs::metadata(path)
        .with_context(|| format!("Failed to read metadata for {}", path.display()))?
        .len();
    let now = current_timestamp();

    metadata.upsert(CacheEntry {
        name: name.to_string(),
        path: path.to_path_buf(),
        size_bytes,
        downloaded_at: now,
        last_accessed: now,
        access_count: 0,
        checksum,
    });

    metadata.save(cache_dir)
}

/// Adopt asset files that already exist on disk but have no matching
/// [`CacheEntry`] in `metadata`.
///
/// The download paths do call [`record_download`] now, but asset files can
/// still reach the cache directory by other routes — a manual copy of a
/// hand-downloaded release artifact, or a download performed by an older
/// build that predates the recording. Relying solely on `cache.json` would
/// make every `cache` subcommand report an empty cache in those cases, no
/// matter how many gigabytes were actually on disk. This scans the
/// well-known asset paths from [`crate::assets`] and adopts any that are
/// present.
///
/// `last_accessed` is seeded to *now*, not to the file's mtime: we have no
/// real access history for a file we are only just noticing, and mtime is
/// frequently much older than "now" (e.g. an asset downloaded weeks ago).
/// Seeding `last_accessed` from mtime would make a first-ever `cache clean`
/// immediately eligible to delete files that were never actually identified
/// as stale — "now" is the only honest lower bound we have. `downloaded_at`
/// still uses mtime as a best-effort historical marker, since it is only
/// ever surfaced informationally.
///
/// Returns `true` if any entries were added, so callers know whether
/// `metadata` needs to be persisted.
fn discover_untracked_assets(cache_dir: &Path, metadata: &mut CacheMetadata) -> Result<bool> {
    let mut changed = false;

    for path in assets::expected_asset_paths(cache_dir) {
        if !path.exists() {
            continue;
        }
        if metadata.entries.iter().any(|e| e.path == path) {
            continue;
        }

        let file_meta = std::fs::metadata(&path)
            .with_context(|| format!("Failed to read metadata for {}", path.display()))?;
        let downloaded_at = file_meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map(|d| d.as_secs())
            .unwrap_or_else(current_timestamp);
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.display().to_string());

        metadata.entries.push(CacheEntry {
            name,
            path,
            size_bytes: file_meta.len(),
            downloaded_at,
            last_accessed: current_timestamp(),
            access_count: 0,
            checksum: None,
        });
        changed = true;
    }

    Ok(changed)
}

/// List all cached assets with details.
///
/// Assets that exist on disk but are not yet recorded in `cache.json` are
/// adopted for this listing (see `discover_untracked_assets`) so a cache
/// directory populated outside the recorded download path — a manual copy,
/// or an older build — doesn't report as empty. `list` is read-only and does
/// not persist the discovery itself; run `cache verify` (or `cache clean`)
/// to make it durable.
pub fn list_cache(cache_dir: &Path) -> Result<()> {
    let mut metadata = CacheMetadata::load(cache_dir)?;
    discover_untracked_assets(cache_dir, &mut metadata)?;

    if metadata.entries.is_empty() {
        println!("Cache is empty");
        return Ok(());
    }

    println!(
        "\n📦 Cached Assets ({} items, {} MB total)\n",
        metadata.entries.len(),
        metadata.total_size() / 1_000_000
    );

    println!(
        "{:<40} {:>12} {:>15} {:>10}",
        "Name", "Size", "Last Accessed", "Count"
    );
    println!("{}", "─".repeat(80));

    for entry in &metadata.entries {
        let size_mb = entry.size_bytes as f64 / 1_000_000.0;
        let days_ago = (current_timestamp().saturating_sub(entry.last_accessed)) / 86400;

        println!(
            "{:<40} {:>10.1} MB {:>12} days {:>10}",
            entry.name, size_mb, days_ago, entry.access_count
        );
    }

    println!();
    Ok(())
}

/// Machine-readable outcome of a [`clean_cache_report`] run.
///
/// The cleaner returns this instead of writing straight to stdout so a
/// caller can either render it as a human report (via [`clean_cache`]) or
/// serialize it for `--json` — the report is the single source of truth for
/// both, so the two can never drift apart.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CleanReport {
    /// Cache directory the run operated on.
    pub cache_dir: PathBuf,
    /// Age threshold the run was given, in days.
    pub max_age_days: u64,
    /// Whether this was a dry run: nothing deleted, nothing persisted.
    pub dry_run: bool,
    /// Entries that were removed — or, for a dry run, that would be.
    pub removed: Vec<CacheEntry>,
    /// Stale entries deliberately *kept* because their file lives outside
    /// `cache_dir` — see [`clean_cache_report`] for why.
    pub retained_external: Vec<CacheEntry>,
    /// Bytes freed — or, for a dry run, that would be freed.
    pub bytes_freed: u64,
}

impl CleanReport {
    /// Number of entries removed (or that would be removed).
    #[must_use]
    pub fn removed_count(&self) -> usize {
        self.removed.len()
    }
}

/// Whether `path` resolves to a location inside `root`.
///
/// When both sides resolve, the *canonical* forms are compared: on macOS
/// `std::env::temp_dir()` alone is enough to make the literal forms
/// (`/var/...` vs `/private/var/...`) disagree, and a symlink inside the
/// cache pointing at a file elsewhere must not count as contained just
/// because its literal path has the right prefix. When either side cannot be
/// resolved — most often because the file has already been deleted — the
/// literal, component-wise comparison is the fallback.
///
/// This gates a destructive operation, so it errs toward *not* contained:
/// mistakenly retaining a cache file wastes disk, mistakenly deleting one
/// outside the cache destroys another tool's data.
fn is_contained(root: &Path, path: &Path) -> bool {
    match (std::fs::canonicalize(root), std::fs::canonicalize(path)) {
        (Ok(canonical_root), Ok(canonical_path)) => canonical_path.starts_with(canonical_root),
        _ => path.starts_with(root),
    }
}

/// Compute — and, unless `dry_run`, apply — a cache cleaning pass.
///
/// Newly discovered (previously unrecorded) assets are adopted first — see
/// `discover_untracked_assets` — with `last_accessed` seeded to *now*, so
/// a file this process is only just noticing can never be immediately swept
/// up as stale by the age check.
///
/// Entries whose file lives outside `cache_dir` are never deleted, even when
/// stale: `record_download` can legitimately register a path in another
/// cache root (a HuggingFace Hub download lands in `~/.cache/huggingface`),
/// and a cache cleaner must not reach outside its own directory to delete
/// another tool's files. Those entries are returned in
/// [`CleanReport::retained_external`] instead.
///
/// Nothing is printed; see [`clean_cache`] for the human-facing wrapper.
///
/// # Errors
///
/// Propagates metadata load/save failures and any file removal that fails.
pub fn clean_cache_report(
    cache_dir: &Path,
    max_age_days: u64,
    dry_run: bool,
) -> Result<CleanReport> {
    let mut metadata = CacheMetadata::load(cache_dir)?;
    let discovered = discover_untracked_assets(cache_dir, &mut metadata)?;

    // `max_age_days` comes straight from `--max-age-days` with no upper
    // bound; saturate the conversion to seconds instead of overflowing (a
    // huge value should mean "keep everything", not panic or wrap around to
    // a tiny cutoff that deletes everything).
    let cutoff = current_timestamp().saturating_sub(max_age_days.saturating_mul(86_400));

    let (to_remove, retained_external): (Vec<CacheEntry>, Vec<CacheEntry>) = metadata
        .entries
        .iter()
        .filter(|e| e.last_accessed < cutoff)
        .cloned()
        .partition(|e| is_contained(cache_dir, &e.path));

    let bytes_freed: u64 = to_remove.iter().map(|e| e.size_bytes).sum();

    let report = CleanReport {
        cache_dir: cache_dir.to_path_buf(),
        max_age_days,
        dry_run,
        removed: to_remove,
        retained_external,
        bytes_freed,
    };

    if dry_run {
        // A dry run must leave the filesystem exactly as it found it —
        // including `cache.json`, so a discovery made only to compute this
        // plan is not silently persisted.
        return Ok(report);
    }

    for entry in &report.removed {
        if entry.path.exists() {
            std::fs::remove_file(&entry.path)
                .with_context(|| format!("Failed to remove file: {}", entry.path.display()))?;
        }
    }
    if !report.removed.is_empty() {
        // Drop the removed entries by `path`, the key `upsert` uses: two
        // distinct assets can share a display name, and retaining by name
        // would evict the survivor along with the casualty.
        metadata
            .entries
            .retain(|e| !report.removed.iter().any(|removed| removed.path == e.path));
    }
    if discovered || !report.removed.is_empty() {
        metadata.save(cache_dir)?;
    }

    // The removal loop tolerates an already-missing file, so the report
    // describes exactly the entries that are gone from the cache now.
    Ok(report)
}

/// Clean old cached assets, printing a human-readable report to stdout.
///
/// Thin wrapper over [`clean_cache_report`]; see that function for the
/// deletion policy.
///
/// # Errors
///
/// Propagates [`clean_cache_report`].
pub fn clean_cache(cache_dir: &Path, max_age_days: u64, dry_run: bool) -> Result<()> {
    let report = clean_cache_report(cache_dir, max_age_days, dry_run)?;
    print_clean_report(&report);
    Ok(())
}

/// Render a [`CleanReport`] as the human-readable `cache clean` output.
fn print_clean_report(report: &CleanReport) {
    for entry in &report.retained_external {
        println!(
            "⚠️  {} is stale but lives outside the cache directory ({}); left in place.",
            entry.name,
            entry.path.display()
        );
    }

    if report.removed.is_empty() {
        println!(
            "✅ No assets to clean (all accessed within {} days)",
            report.max_age_days
        );
        return;
    }

    if report.dry_run {
        println!("🗑️  Will remove {} assets:", report.removed.len());
    } else {
        println!("🗑️  Removed {} assets:", report.removed.len());
    }

    for entry in &report.removed {
        println!(
            "  - {} ({:.1} MB)",
            entry.name,
            entry.size_bytes as f64 / 1_000_000.0
        );
    }

    println!(
        "\nTotal space {}: {:.1} MB",
        if report.dry_run { "to free" } else { "freed" },
        report.bytes_freed as f64 / 1_000_000.0
    );

    if report.dry_run {
        println!("\n(Dry run - no files deleted)");
    } else {
        println!("\n✅ Cleaned {} assets", report.removed.len());
    }
}

/// Verify cache integrity.
///
/// Newly discovered assets are adopted (see `discover_untracked_assets`),
/// and any entry without a recorded checksum has one computed and saved now
/// ("trust on first verify"), so a *subsequent* run can actually detect
/// corruption instead of only ever checking existence and size.
pub fn verify_cache(cache_dir: &Path) -> Result<()> {
    let mut metadata = CacheMetadata::load(cache_dir)?;
    let mut dirty = discover_untracked_assets(cache_dir, &mut metadata)?;

    println!("🔍 Verifying cache integrity...\n");

    let mut issues = 0;
    for idx in 0..metadata.entries.len() {
        let name = metadata.entries[idx].name.clone();
        let path = metadata.entries[idx].path.clone();
        let expected_size = metadata.entries[idx].size_bytes;

        print!("  {} ... ", name);

        if !path.exists() {
            println!("❌ MISSING");
            issues += 1;
            continue;
        }

        let actual_size = std::fs::metadata(&path)
            .with_context(|| format!("Failed to read metadata for {}", path.display()))?
            .len();
        if actual_size != expected_size {
            println!(
                "❌ SIZE MISMATCH (expected {}, got {})",
                expected_size, actual_size
            );
            issues += 1;
            continue;
        }

        match metadata.entries[idx].checksum.clone() {
            Some(expected_checksum) => {
                let actual_checksum = compute_sha256(&path)?;
                // Case-insensitive, matching `assets::is_cached` and
                // `assets::finalize_download`: a recorded checksum can
                // originate from the `ASSETS` manifest, whose hex digests are
                // hand-entered from a published release and may be
                // upper-case. Comparing those byte-for-byte against
                // `compute_sha256`'s always-lower-case output would report
                // CHECKSUM MISMATCH on a perfectly intact file.
                if !actual_checksum.eq_ignore_ascii_case(&expected_checksum) {
                    println!("❌ CHECKSUM MISMATCH");
                    issues += 1;
                    continue;
                }
            }
            None => {
                let computed = compute_sha256(&path)?;
                metadata.entries[idx].checksum = Some(computed);
                dirty = true;
            }
        }

        println!("✅ OK");
    }

    if dirty {
        metadata.save(cache_dir)?;
    }

    println!();
    if issues == 0 {
        println!(
            "✅ All {} assets verified successfully",
            metadata.entries.len()
        );
    } else {
        println!("⚠️  Found {} issues", issues);
    }

    Ok(())
}

/// Compute the lowercase-hex SHA-256 checksum of a file.
///
/// Streams the file through a fixed-size buffer instead of reading it
/// entirely into memory — the manifest's largest asset is a ~1.7 GB
/// diffusion U-Net, and `Sha256::digest(&std::fs::read(path)?)` would
/// resident that whole file just to hash it.
///
/// This is the crate's single checksum implementation: [`crate::assets`]
/// used to carry a `sha256_hex` twin that differed only by slurping the
/// whole file, and now calls this instead.
///
/// # Errors
///
/// Returns an error if the file cannot be opened or read.
pub(crate) fn compute_sha256(path: &Path) -> Result<String> {
    use sha2::{Digest, Sha256};
    use std::fmt::Write as _;
    use std::io::{BufReader, Read};

    let file = std::fs::File::open(path)
        .with_context(|| format!("Failed to open file for checksum: {}", path.display()))?;
    let mut reader = BufReader::new(file);
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 65536];

    loop {
        let n = reader
            .read(&mut buf)
            .with_context(|| format!("Failed to read file for checksum: {}", path.display()))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }

    let hash = hasher.finalize();
    let mut checksum = String::with_capacity(hash.len() * 2);
    for byte in hash {
        // Writing a byte to a `String` is infallible, so the result is discarded.
        let _ = write!(checksum, "{byte:02x}");
    }
    Ok(checksum)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn unique_temp_dir(tag: &str) -> PathBuf {
        let id = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
        let dir = std::env::temp_dir().join(format!(
            "oxigaf_cache_unit_{tag}_{}_{}",
            std::process::id(),
            id
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("failed to create temp dir for test");
        dir
    }

    #[test]
    fn compute_sha256_matches_in_memory_digest_for_a_multi_chunk_file() {
        use sha2::{Digest, Sha256};

        let dir = unique_temp_dir("sha256");
        let path = dir.join("payload.bin");

        // Larger than the 64 KiB streaming buffer so the read loop runs
        // more than once.
        let data: Vec<u8> = (0..200_000u32).map(|i| (i % 251) as u8).collect();
        std::fs::write(&path, &data).expect("failed to write test payload");

        let expected = {
            let digest = Sha256::digest(&data);
            let mut s = String::with_capacity(digest.len() * 2);
            for byte in digest {
                use std::fmt::Write as _;
                let _ = write!(s, "{byte:02x}");
            }
            s
        };

        let actual = compute_sha256(&path).expect("compute_sha256 should succeed");
        assert_eq!(actual, expected);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn clean_cache_with_huge_max_age_days_does_not_delete_a_just_discovered_asset() {
        let dir = unique_temp_dir("overflow");
        let asset_path = dir.join("thing.bin");
        std::fs::write(&asset_path, b"keep me").expect("failed to write asset");
        record_download(&dir, "Thing", &asset_path, None).expect("record_download should succeed");

        // Before the `saturating_mul` fix, `max_age_days * 86_400` with
        // `max_age_days = u64::MAX` panics on overflow in a debug build.
        // After the fix it must also not wrap around to a tiny cutoff that
        // would incorrectly delete an asset recorded moments ago.
        clean_cache(&dir, u64::MAX, false).expect("clean_cache should not error");

        assert!(
            asset_path.exists(),
            "an asset recorded moments ago must survive an (effectively infinite) max-age clean"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn clean_cache_does_not_delete_a_freshly_discovered_but_old_on_disk_file() {
        // Regression test for a destructive first run: a file's mtime can
        // be far in the past (it was downloaded long ago) even though this
        // is the *first* time this process has ever seen it. `clean` must
        // not treat "just discovered" as "long overdue for deletion".
        let dir = unique_temp_dir("discover_clean");
        let expected_paths = assets::expected_asset_paths(&dir);
        std::fs::write(&expected_paths[0], b"old but still wanted")
            .expect("failed to write fake asset");

        // Default `--max-age-days` is 30; use the same order of magnitude.
        clean_cache(&dir, 30, false).expect("clean_cache should not error");

        assert!(
            expected_paths[0].exists(),
            "a file discovered for the first time must not be deleted by the same run"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn discover_untracked_assets_adopts_files_present_on_disk() {
        let dir = unique_temp_dir("discover");

        // Create one of the well-known asset files without going through
        // `assets::setup_cache` or `record_download`.
        let expected_paths = assets::expected_asset_paths(&dir);
        assert!(
            !expected_paths.is_empty(),
            "asset manifest should not be empty"
        );
        std::fs::write(&expected_paths[0], b"fake asset bytes")
            .expect("failed to write fake asset");

        let mut metadata = CacheMetadata::load(&dir).expect("load should succeed on empty dir");
        assert!(metadata.entries.is_empty());

        let changed = discover_untracked_assets(&dir, &mut metadata)
            .expect("discover_untracked_assets should succeed");
        assert!(changed);
        assert_eq!(metadata.entries.len(), 1);
        assert_eq!(metadata.entries[0].path, expected_paths[0]);
        assert_eq!(
            metadata.entries[0].size_bytes,
            b"fake asset bytes".len() as u64
        );

        // Running it again with the entry already present must be a no-op.
        let changed_again = discover_untracked_assets(&dir, &mut metadata)
            .expect("second discover_untracked_assets should succeed");
        assert!(!changed_again);
        assert_eq!(metadata.entries.len(), 1);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn list_cache_shows_assets_discovered_on_disk_but_does_not_persist_them() {
        let dir = unique_temp_dir("list");

        let expected_paths = assets::expected_asset_paths(&dir);
        std::fs::write(&expected_paths[0], b"fake asset bytes")
            .expect("failed to write fake asset");

        // `list` is read-only: it must show newly discovered assets in this
        // invocation, but must not write cache.json as a side effect.
        list_cache(&dir).expect("list_cache should succeed");
        assert!(
            !dir.join("cache.json").exists(),
            "list_cache must not persist discovered entries"
        );

        // `verify` is expected to be the one that makes discovery durable.
        verify_cache(&dir).expect("verify_cache should succeed");
        let metadata = CacheMetadata::load(&dir).expect("cache.json should now be readable");
        assert_eq!(metadata.entries.len(), 1);
        assert_eq!(metadata.entries[0].path, expected_paths[0]);
        assert!(metadata.entries[0].checksum.is_some());

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Write a `cache.json` whose single entry was last accessed
    /// `days_ago` days ago, so age-based cleaning has something to bite on.
    fn seed_stale_entry(cache_dir: &Path, name: &str, path: &Path, days_ago: u64) {
        let size_bytes = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
        let stale = current_timestamp().saturating_sub(days_ago * 86_400);
        let metadata = CacheMetadata {
            version: env!("CARGO_PKG_VERSION").to_string(),
            entries: vec![CacheEntry {
                name: name.to_string(),
                path: path.to_path_buf(),
                size_bytes,
                downloaded_at: stale,
                last_accessed: stale,
                access_count: 0,
                checksum: None,
            }],
        };
        metadata.save(cache_dir).expect("seed metadata should save");
    }

    #[test]
    fn verify_cache_accepts_an_uppercase_recorded_checksum() {
        // A checksum recorded from the `ASSETS` manifest is hand-entered hex
        // and may be upper-case, while `compute_sha256` always emits
        // lower-case. A byte-for-byte comparison would report CHECKSUM
        // MISMATCH on an intact file.
        let dir = unique_temp_dir("case_checksum");
        let path = dir.join("payload.bin");
        std::fs::write(&path, b"hello world").expect("failed to write asset");
        let digest = compute_sha256(&path).expect("compute_sha256 should succeed");
        assert_eq!(digest, digest.to_lowercase(), "digests are lower-case hex");

        record_download(&dir, "Payload", &path, Some(digest.to_uppercase()))
            .expect("record_download should succeed");
        verify_cache(&dir).expect("verify_cache should succeed");

        // An intact file must keep its recorded checksum: a mismatch would
        // have left it untouched *and* counted an issue, so the surviving
        // value being the original upper-case one proves the OK branch ran.
        let metadata = CacheMetadata::load(&dir).expect("metadata should reload");
        assert_eq!(
            metadata.entries[0].checksum.as_deref(),
            Some(digest.to_uppercase().as_str())
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn record_download_preserves_existing_usage_history() {
        // Re-downloading a file (say, after `verify` found it corrupt) must
        // not reset the access history that `cache clean`'s age policy and
        // `cache list`'s usage column read.
        let dir = unique_temp_dir("history");
        let path = dir.join("thing.bin");
        std::fs::write(&path, b"v1").expect("failed to write asset");
        record_download(&dir, "Thing", &path, None).expect("record_download should succeed");

        // Simulate real usage accumulating against the entry.
        let mut metadata = CacheMetadata::load(&dir).expect("load should succeed");
        for _ in 0..5 {
            metadata.update_access("Thing");
        }
        let future = current_timestamp() + 3_600;
        metadata.entries[0].last_accessed = future;
        metadata.save(&dir).expect("save should succeed");

        std::fs::write(&path, b"v2-longer").expect("failed to rewrite asset");
        record_download(&dir, "Thing", &path, Some("abc".to_string()))
            .expect("second record_download should succeed");

        let metadata = CacheMetadata::load(&dir).expect("load should succeed");
        assert_eq!(metadata.entries.len(), 1);
        assert_eq!(
            metadata.entries[0].access_count, 5,
            "a re-download must not reset the access count"
        );
        assert_eq!(
            metadata.entries[0].last_accessed, future,
            "a re-download must not move a newer access timestamp backwards"
        );
        // The download-derived fields still refresh.
        assert_eq!(metadata.entries[0].checksum.as_deref(), Some("abc"));
        assert_eq!(metadata.entries[0].size_bytes, b"v2-longer".len() as u64);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn is_contained_resolves_symlinked_roots_and_rejects_outsiders() {
        let root = unique_temp_dir("contained_root");
        let outside = unique_temp_dir("contained_outside");

        let inside_path = root.join("inside.bin");
        std::fs::write(&inside_path, b"in").expect("failed to write inside file");
        let outside_path = outside.join("outside.bin");
        std::fs::write(&outside_path, b"out").expect("failed to write outside file");

        assert!(is_contained(&root, &inside_path));
        assert!(!is_contained(&root, &outside_path));

        // On macOS `std::env::temp_dir()` is itself a symlink
        // (/var -> /private/var), so a literal-prefix comparison of a
        // canonicalized entry path against a non-canonical root would call
        // an in-cache file "external" and silently stop cleaning it.
        let canonical_inside =
            std::fs::canonicalize(&inside_path).expect("inside file should canonicalize");
        assert!(is_contained(&root, &canonical_inside));

        // A deleted file (nothing left to canonicalize) still classifies by
        // its literal path, so a stale entry for an already-removed cache
        // file is not misfiled as external.
        std::fs::remove_file(&inside_path).expect("failed to remove inside file");
        assert!(is_contained(&root, &inside_path));
        assert!(!is_contained(&root, &outside_path.join("gone.bin")));

        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&outside);
    }

    #[test]
    fn clean_cache_report_returns_the_removed_entries() {
        let dir = unique_temp_dir("report");
        let asset_path = dir.join("stale.bin");
        std::fs::write(&asset_path, vec![7u8; 2048]).expect("failed to write asset");
        seed_stale_entry(&dir, "Stale", &asset_path, 90);

        let report =
            clean_cache_report(&dir, 30, false).expect("clean_cache_report should succeed");

        assert!(!report.dry_run);
        assert_eq!(report.removed_count(), 1);
        assert_eq!(report.removed[0].name, "Stale");
        assert_eq!(report.bytes_freed, 2048);
        assert!(report.retained_external.is_empty());
        assert!(
            !asset_path.exists(),
            "a stale asset must actually be removed"
        );

        // The report is what `cache clean --json` serializes, so it must be
        // serializable without losing the removal list.
        let value = serde_json::to_value(&report).expect("report should serialize");
        assert_eq!(value["removed"].as_array().map(Vec::len), Some(1));
        assert_eq!(value["bytes_freed"].as_u64(), Some(2048));

        let metadata = CacheMetadata::load(&dir).expect("metadata should reload");
        assert!(
            metadata.entries.is_empty(),
            "the removed entry must be dropped from cache.json"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn clean_cache_dry_run_deletes_nothing_and_writes_nothing() {
        let dir = unique_temp_dir("dryrun");
        // A discoverable asset with no cache.json at all: the dry run has to
        // adopt it to build a plan, and must not persist that adoption.
        let expected_paths = assets::expected_asset_paths(&dir);
        std::fs::write(&expected_paths[0], b"untracked").expect("failed to write fake asset");

        let report = clean_cache_report(&dir, 30, true).expect("dry run should succeed");
        assert!(report.dry_run);
        assert!(expected_paths[0].exists(), "dry run must not delete");
        assert!(
            !dir.join("cache.json").exists(),
            "dry run must not persist discovered entries"
        );

        // …and with something genuinely stale, the plan lists it without
        // touching it.
        let stale_path = dir.join("stale.bin");
        std::fs::write(&stale_path, b"old").expect("failed to write asset");
        seed_stale_entry(&dir, "Stale", &stale_path, 90);

        let report = clean_cache_report(&dir, 30, true).expect("dry run should succeed");
        assert_eq!(report.removed_count(), 1);
        assert!(stale_path.exists(), "dry run must not delete");

        let metadata = CacheMetadata::load(&dir).expect("metadata should reload");
        assert_eq!(
            metadata.entries.len(),
            1,
            "dry run must not rewrite cache.json"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn clean_cache_never_deletes_files_outside_the_cache_directory() {
        // `record_download` legitimately registers paths in other cache
        // roots (a HuggingFace Hub download lands under ~/.cache/huggingface),
        // and OxiGAF must not sweep another tool's files out from under it.
        let dir = unique_temp_dir("external_cache");
        let outside = unique_temp_dir("external_home");
        let external_path = outside.join("someone_elses_model.safetensors");
        std::fs::write(&external_path, b"not ours to delete").expect("failed to write asset");
        seed_stale_entry(&dir, "external", &external_path, 900);

        let report =
            clean_cache_report(&dir, 30, false).expect("clean_cache_report should succeed");

        assert!(
            report.removed.is_empty(),
            "nothing inside the cache is stale"
        );
        assert_eq!(report.retained_external.len(), 1);
        assert_eq!(report.retained_external[0].name, "external");
        assert_eq!(report.bytes_freed, 0);
        assert!(
            external_path.exists(),
            "a stale entry outside the cache directory must be left on disk"
        );

        let metadata = CacheMetadata::load(&dir).expect("metadata should reload");
        assert_eq!(
            metadata.entries.len(),
            1,
            "a retained external entry stays tracked"
        );

        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_dir_all(&outside);
    }

    #[test]
    fn clean_cache_removes_by_path_not_by_display_name() {
        // `upsert` keys on `path`, so two distinct files can share a name.
        // Removing the stale one must not evict the fresh namesake.
        let dir = unique_temp_dir("samename");
        let stale_path = dir.join("stale").join("model.bin");
        let fresh_path = dir.join("fresh").join("model.bin");
        for path in [&stale_path, &fresh_path] {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).expect("failed to create asset dir");
            }
            std::fs::write(path, b"payload").expect("failed to write asset");
        }

        let stale = current_timestamp().saturating_sub(90 * 86_400);
        let now = current_timestamp();
        let metadata = CacheMetadata {
            version: env!("CARGO_PKG_VERSION").to_string(),
            entries: vec![
                CacheEntry {
                    name: "model.bin".to_string(),
                    path: stale_path.clone(),
                    size_bytes: 7,
                    downloaded_at: stale,
                    last_accessed: stale,
                    access_count: 0,
                    checksum: None,
                },
                CacheEntry {
                    name: "model.bin".to_string(),
                    path: fresh_path.clone(),
                    size_bytes: 7,
                    downloaded_at: now,
                    last_accessed: now,
                    access_count: 0,
                    checksum: None,
                },
            ],
        };
        metadata.save(&dir).expect("seed metadata should save");

        let report =
            clean_cache_report(&dir, 30, false).expect("clean_cache_report should succeed");
        assert_eq!(report.removed_count(), 1);
        assert!(!stale_path.exists());
        assert!(fresh_path.exists(), "the fresh namesake must survive");

        let metadata = CacheMetadata::load(&dir).expect("metadata should reload");
        assert_eq!(
            metadata.entries.len(),
            1,
            "only the stale entry may be dropped from cache.json"
        );
        assert_eq!(metadata.entries[0].path, fresh_path);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn record_download_upserts_by_path() {
        let dir = unique_temp_dir("record");
        let asset_path = dir.join("thing.bin");
        std::fs::write(&asset_path, b"v1").expect("failed to write asset");

        record_download(&dir, "Thing", &asset_path, Some("abc".to_string()))
            .expect("record_download should succeed");

        std::fs::write(&asset_path, b"v2-longer").expect("failed to rewrite asset");
        record_download(&dir, "Thing", &asset_path, Some("def".to_string()))
            .expect("second record_download should succeed");

        let metadata = CacheMetadata::load(&dir).expect("load should succeed");
        assert_eq!(metadata.entries.len(), 1, "same path must upsert in place");
        assert_eq!(metadata.entries[0].checksum.as_deref(), Some("def"));
        assert_eq!(metadata.entries[0].size_bytes, b"v2-longer".len() as u64);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
