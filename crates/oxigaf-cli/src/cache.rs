use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

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
}

/// Get current Unix timestamp in seconds
fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// List all cached assets with details
pub fn list_cache(cache_dir: &Path) -> Result<()> {
    let metadata = CacheMetadata::load(cache_dir)?;

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

/// Clean old cached assets
pub fn clean_cache(cache_dir: &Path, max_age_days: u64, dry_run: bool) -> Result<()> {
    let mut metadata = CacheMetadata::load(cache_dir)?;
    let cutoff = current_timestamp().saturating_sub(max_age_days * 86400);

    let to_remove: Vec<_> = metadata
        .entries
        .iter()
        .filter(|e| e.last_accessed < cutoff)
        .cloned()
        .collect();

    if to_remove.is_empty() {
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

/// Verify cache integrity
pub fn verify_cache(cache_dir: &Path) -> Result<()> {
    let metadata = CacheMetadata::load(cache_dir)?;

    println!("🔍 Verifying cache integrity...\n");

    let mut issues = 0;
    for entry in &metadata.entries {
        print!("  {} ... ", entry.name);

        if !entry.path.exists() {
            println!("❌ MISSING");
            issues += 1;
            continue;
        }

        let actual_size = std::fs::metadata(&entry.path)
            .with_context(|| format!("Failed to read metadata for {}", entry.path.display()))?
            .len();
        if actual_size != entry.size_bytes {
            println!(
                "❌ SIZE MISMATCH (expected {}, got {})",
                entry.size_bytes, actual_size
            );
            issues += 1;
            continue;
        }

        if let Some(ref expected_checksum) = entry.checksum {
            let actual_checksum = compute_sha256(&entry.path)?;
            if &actual_checksum != expected_checksum {
                println!("❌ CHECKSUM MISMATCH");
                issues += 1;
                continue;
            }
        }

        println!("✅ OK");
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

/// Compute SHA-256 checksum of a file
fn compute_sha256(path: &Path) -> Result<String> {
    use sha2::{Digest, Sha256};
    use std::fmt::Write as _;

    let data = std::fs::read(path)
        .with_context(|| format!("Failed to read file for checksum: {}", path.display()))?;
    let hash = Sha256::digest(&data);
    let mut checksum = String::with_capacity(hash.len() * 2);
    for byte in hash {
        // Writing a byte to a `String` is infallible, so the result is discarded.
        let _ = write!(checksum, "{byte:02x}");
    }
    Ok(checksum)
}
