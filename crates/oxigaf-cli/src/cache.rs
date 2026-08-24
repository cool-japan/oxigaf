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
    #[allow(dead_code)]
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
    fn upsert(&mut self, entry: CacheEntry) {
        if let Some(existing) = self.entries.iter_mut().find(|e| e.path == entry.path) {
            *existing = entry;
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
/// `assets::setup_cache` currently writes asset files straight into the
/// cache directory without registering them here (that is why `cache.json`
/// otherwise stays empty forever — see [`discover_untracked_assets`] for the
/// directory-scan fallback that covers that gap in the meantime). Once the
/// download path is wired to call this function right after each successful
/// download, `cache list`/`verify`/`clean` get a precise `downloaded_at`
/// timestamp and an immediately-known checksum instead of having to infer
/// them from filesystem metadata on next use.
#[allow(dead_code)]
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
/// Nothing in the download path currently calls [`record_download`] (or
/// `CacheMetadata::save` at all, outside of [`clean_cache`]), so relying
/// solely on `cache.json` would make every `cache` subcommand report an
/// empty cache no matter how many gigabytes had actually been downloaded.
/// This scans the well-known asset paths from [`crate::assets`] and adopts
/// any that are present on disk.
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
/// adopted for this listing (see [`discover_untracked_assets`]) so a cache
/// directory populated by `assets::setup_cache` doesn't always report as
/// empty — but `list` is read-only and does not persist the discovery
/// itself; run `cache verify` (or `cache clean`) to make it durable.
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

/// Clean old cached assets.
///
/// Newly discovered (previously unrecorded) assets are adopted first — see
/// [`discover_untracked_assets`] — with `last_accessed` seeded to *now*, so
/// a file this process is only just noticing can never be immediately swept
/// up as stale by the age check below.
pub fn clean_cache(cache_dir: &Path, max_age_days: u64, dry_run: bool) -> Result<()> {
    let mut metadata = CacheMetadata::load(cache_dir)?;
    let discovered = discover_untracked_assets(cache_dir, &mut metadata)?;

    // `max_age_days` comes straight from `--max-age-days` with no upper
    // bound; saturate the conversion to seconds instead of overflowing (a
    // huge value should mean "keep everything", not panic or wrap around to
    // a tiny cutoff that deletes everything).
    let cutoff = current_timestamp().saturating_sub(max_age_days.saturating_mul(86_400));

    let to_remove: Vec<_> = metadata
        .entries
        .iter()
        .filter(|e| e.last_accessed < cutoff)
        .cloned()
        .collect();

    if to_remove.is_empty() {
        if discovered && !dry_run {
            metadata.save(cache_dir)?;
        }
        println!(
            "✅ No assets to clean (all accessed within {} days)",
            max_age_days
        );
        return Ok(());
    }

    println!("🗑️  Will remove {} assets:", to_remove.len());

    let mut total_freed = 0u64;
    for entry in &to_remove {
        println!(
            "  - {} ({:.1} MB)",
            entry.name,
            entry.size_bytes as f64 / 1_000_000.0
        );
        total_freed += entry.size_bytes;
    }

    println!(
        "\nTotal space to free: {:.1} MB",
        total_freed as f64 / 1_000_000.0
    );

    if dry_run {
        println!("\n(Dry run - no files deleted)");
        return Ok(());
    }

    for entry in &to_remove {
        if entry.path.exists() {
            std::fs::remove_file(&entry.path)
                .with_context(|| format!("Failed to remove file: {}", entry.path.display()))?;
        }
        metadata.entries.retain(|e| e.name != entry.name);
    }

    metadata.save(cache_dir)?;
    println!("\n✅ Cleaned {} assets", to_remove.len());

    Ok(())
}

/// Verify cache integrity.
///
/// Newly discovered assets are adopted (see [`discover_untracked_assets`]),
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
                if actual_checksum != expected_checksum {
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

/// Compute the SHA-256 checksum of a file.
///
/// Streams the file through a fixed-size buffer instead of reading it
/// entirely into memory — the manifest's largest asset is a ~1.7 GB
/// diffusion U-Net, and `Sha256::digest(&std::fs::read(path)?)` would
/// resident that whole file just to hash it.
fn compute_sha256(path: &Path) -> Result<String> {
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
