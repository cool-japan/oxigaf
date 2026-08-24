//! Model download and cache management.
//!
//! Provides the `setup` command implementation: downloads required model weights
//! (FLAME, diffusion U-Net, VAE, CLIP) into a local cache directory and
//! verifies file integrity — via SHA-256 when a checksum has been
//! published for the asset, otherwise via a size-floor sanity check.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::progress;
use crate::verbosity::Verbosity;

// ---------------------------------------------------------------------------
// Asset manifest
// ---------------------------------------------------------------------------

/// Metadata for a single downloadable model asset.
struct Asset {
    /// Human-readable name shown in progress output.
    name: &'static str,
    /// Download URL.
    url: &'static str,
    /// Filename within the cache directory.
    filename: &'static str,
    /// Expected file size in bytes (used for progress reporting and, when
    /// `sha256` is empty, as a size-floor sanity check). Set to 0 to skip
    /// the size check.
    expected_bytes: u64,
    /// SHA-256 hex digest of the published artifact, or empty when no
    /// artifact has been published yet. When empty, `setup_cache` treats
    /// this asset as unpublished and fails fast (see the `ASSETS` doc)
    /// instead of attempting a download that is known to fail; once
    /// populated, downloads for that asset flow through the normal
    /// `download_file` → `finalize_download` path and are verified exactly
    /// via SHA-256 rather than the size-floor heuristic.
    sha256: &'static str,
}

/// HuggingFace Hub model source specification.
///
/// Supports parsing model identifiers like:
/// - "cool-japan/oxigaf-flame-2023" (default revision)
/// - "cool-japan/oxigaf-flame-2023:main" (branch/tag)
/// - "cool-japan/oxigaf-flame-2023@v1.0" (specific revision)
pub struct HfModelSource {
    /// Repository identifier (e.g., "cool-japan/oxigaf-flame")
    pub repo_id: String,
    /// Model filename within the repository
    pub filename: String,
    /// Optional revision (branch, tag, or commit SHA)
    pub revision: Option<String>,
}

impl HfModelSource {
    /// Parse a HuggingFace model specification string.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use oxigaf_cli::assets::HfModelSource;
    /// let source = HfModelSource::parse("cool-japan/oxigaf-flame").unwrap();
    /// assert_eq!(source.repo_id, "cool-japan/oxigaf-flame");
    /// assert!(source.revision.is_none());
    ///
    /// let source = HfModelSource::parse("cool-japan/oxigaf-flame:main").unwrap();
    /// assert_eq!(source.revision, Some("main".to_string()));
    /// ```
    pub fn parse(spec: &str) -> Result<Self> {
        if spec.is_empty() {
            anyhow::bail!("Model specification cannot be empty");
        }

        // Split on ':' or '@' to extract revision
        let (repo_part, revision) = if let Some(pos) = spec.find(':') {
            let (repo, rev) = spec.split_at(pos);
            (repo, Some(rev[1..].to_string()))
        } else if let Some(pos) = spec.find('@') {
            let (repo, rev) = spec.split_at(pos);
            (repo, Some(rev[1..].to_string()))
        } else {
            (spec, None)
        };

        // Validate repository format (should be "org/repo")
        if !repo_part.contains('/') {
            anyhow::bail!(
                "Invalid repository format: '{}'. Expected format: 'organization/repository'",
                repo_part
            );
        }

        // Validate revision is not empty if specified
        if let Some(ref rev) = revision {
            if rev.is_empty() {
                anyhow::bail!("Revision cannot be empty");
            }
        }

        Ok(Self {
            repo_id: repo_part.to_string(),
            filename: "model.safetensors".to_string(),
            revision,
        })
    }

    /// Set a custom filename instead of the default "model.safetensors".
    pub fn with_filename(mut self, filename: String) -> Self {
        self.filename = filename;
        self
    }

    // Note: a `download` method used to live here as a thin(-ish) wrapper
    // around HuggingFace Hub's API. It was never called (the CLI always
    // goes through the free function `download_with_progress` instead),
    // duplicated that function's logic almost line for line, and printed
    // unconditionally regardless of verbosity/json-mode — so it was deleted
    // rather than kept as unreachable, buggy dead code. Use
    // `download_with_progress(&source.repo_id, &source.filename,
    // source.revision.as_deref(), token, verbosity)` instead.
}

/// Placeholder asset manifest.
///
/// The URLs below point at the project's GitHub releases page, but the
/// referenced release artifacts do not exist yet (verified: they 404) and
/// every `sha256` is empty. `setup_cache` treats an empty `sha256` as "this
/// asset has not been published" and fails fast with a message pointing at
/// `oxigaf setup --from-hub`, rather than attempting a download that is
/// known to fail.
///
/// Once a real artifact is published: replace its `url` with the real
/// download URL, fill in `sha256` with the file's real SHA-256 digest (not
/// an estimate — `expected_bytes` is only ever used as a size *floor* when
/// no checksum is available, never as an exact match, since it is a rough
/// estimate), and the fail-fast branch for that entry stops firing
/// automatically: `setup_cache` will download and cryptographically verify
/// it through the normal `download_file` → `finalize_download` path like
/// any other asset.
static ASSETS: &[Asset] = &[
    Asset {
        name: "FLAME 2023 Head Model",
        url: "https://github.com/cool-japan/oxigaf/releases/download/v0.1.0/flame2023.tar.gz",
        filename: "flame2023.tar.gz",
        expected_bytes: 250_000_000,
        sha256: "",
    },
    Asset {
        name: "Multi-View Diffusion U-Net",
        url: "https://github.com/cool-japan/oxigaf/releases/download/v0.1.0/diffusion_unet.safetensors",
        filename: "diffusion_unet.safetensors",
        expected_bytes: 1_700_000_000,
        sha256: "",
    },
    Asset {
        name: "VAE Decoder",
        url: "https://github.com/cool-japan/oxigaf/releases/download/v0.1.0/vae_decoder.safetensors",
        filename: "vae_decoder.safetensors",
        expected_bytes: 200_000_000,
        sha256: "",
    },
    Asset {
        name: "CLIP Image Encoder",
        url: "https://github.com/cool-japan/oxigaf/releases/download/v0.1.0/clip_image_encoder.safetensors",
        filename: "clip_image_encoder.safetensors",
        expected_bytes: 600_000_000,
        sha256: "",
    },
];

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Ensure all required model assets are present in `cache_dir`.
///
/// For each asset:
/// 1. If the file already exists and passes verification (SHA-256 when
///    published, otherwise a size-floor sanity check), skip it.
/// 2. If it has no published checksum yet (an unpublished placeholder entry
///    — see the `ASSETS` doc), fail fast with a message pointing at
///    `oxigaf setup --from-hub` instead of attempting a download known to
///    fail.
/// 3. Otherwise, download it (via `curl`, or `wget` as a fallback) to a
///    `.part` staging file, verify it, and only then make it visible at its
///    final path.
///
/// This function prints user-facing progress directly to stdout (suppressed
/// when `json_mode` is set).
pub fn setup_cache(cache_dir: &Path, verbosity: Verbosity, json_mode: bool) -> Result<()> {
    let cache_dir = ensure_cache_dir(cache_dir)?;

    if !json_mode {
        println!();
        println!("📦  OxiGAF Model Setup");
        println!("    Cache directory: {}", cache_dir.display());
        println!();
    }

    let mut downloaded = 0usize;
    let mut skipped = 0usize;
    let mut downloaded_assets = Vec::new();

    for asset in ASSETS {
        let dest = cache_dir.join(asset.filename);

        if is_cached(&dest, asset.expected_bytes, asset.sha256) {
            if !json_mode {
                println!("  ✓  {} (cached)", asset.name);
            }
            skipped += 1;
            continue;
        }

        if asset.sha256.is_empty() {
            anyhow::bail!(
                "{} is not yet published (no release artifact or checksum is \
                 available for it). Try `oxigaf setup --from-hub <org/repo>` to \
                 fetch model weights from HuggingFace Hub instead, or place \
                 `{}` in {} manually.",
                asset.name,
                asset.filename,
                cache_dir.display()
            );
        }

        if !json_mode {
            println!("  ⬇  Downloading {} …", asset.name);
        }
        download_file(
            asset.url,
            &dest,
            asset.expected_bytes,
            asset.sha256,
            verbosity,
            json_mode,
        )
        .with_context(|| format!("Failed to download {}", asset.name))?;
        downloaded += 1;
        downloaded_assets.push(dest.clone());
    }

    // Output based on mode
    if json_mode {
        let mut output = crate::json_output::JsonOutput::success(
            "setup",
            serde_json::json!({
                "cache_dir": cache_dir.display().to_string(),
                "downloaded": downloaded,
                "skipped": skipped,
                "total_assets": ASSETS.len()
            }),
        );

        // Add downloaded files as artifacts
        for path in downloaded_assets {
            if path.exists() {
                output.add_artifact("model".to_string(), path);
            }
        }

        output.print();
    } else {
        println!();
        if downloaded > 0 {
            println!(
                "✅  Setup complete — downloaded {downloaded} asset(s), {skipped} already cached."
            );
        } else {
            println!("✅  All assets already cached. Nothing to download.");
        }
    }

    Ok(())
}

/// Return the list of expected asset file paths within the cache directory.
///
/// Useful for tooling that needs to verify the cache without downloading.
#[allow(dead_code)]
pub fn expected_asset_paths(cache_dir: &Path) -> Vec<PathBuf> {
    ASSETS.iter().map(|a| cache_dir.join(a.filename)).collect()
}

/// Get HuggingFace authentication token from environment or config file.
///
/// Checks sources in the following order:
/// 1. HF_TOKEN environment variable
/// 2. ~/.huggingface/token file
///
/// # Returns
///
/// The authentication token if found, `None` otherwise.
pub fn get_hf_token() -> Option<String> {
    // Check environment variable first
    if let Ok(token) = std::env::var("HF_TOKEN") {
        if !token.trim().is_empty() {
            return Some(token.trim().to_string());
        }
    }

    // Fall back to reading from ~/.huggingface/token
    if let Some(home) = dirs::home_dir() {
        let token_path = home.join(".huggingface").join("token");
        if let Ok(token) = std::fs::read_to_string(token_path) {
            let token = token.trim();
            if !token.is_empty() {
                return Some(token.to_string());
            }
        }
    }

    None
}

/// Download a model from HuggingFace Hub with progress tracking.
///
/// # Arguments
///
/// * `repo_id` - HuggingFace repository identifier (e.g., "cool-japan/oxigaf-flame")
/// * `filename` - Filename within the repository
/// * `revision` - Optional revision (branch, tag, or commit SHA)
/// * `token` - Optional authentication token
/// * `verbosity` - Controls progress display
///
/// # Returns
///
/// The path to the downloaded model file in the HuggingFace cache directory.
///
/// # Errors
///
/// Returns an error if:
/// - The API client cannot be initialized
/// - The repository or file is not found
/// - Network errors occur during download
/// - Authentication fails for private models
pub fn download_with_progress(
    repo_id: &str,
    filename: &str,
    revision: Option<&str>,
    token: Option<&str>,
    verbosity: Verbosity,
) -> Result<PathBuf> {
    use hf_hub::api::sync::ApiBuilder;

    if verbosity != Verbosity::Quiet {
        println!("📥 Downloading from HuggingFace Hub: {}", repo_id);
        if let Some(rev) = revision {
            println!("   Revision: {}", rev);
        }
        println!("   File: {}", filename);
        println!();
    }

    // Build API client with optional token and progress tracking
    let mut api_builder = ApiBuilder::new();

    if let Some(token_str) = token {
        api_builder = api_builder.with_token(Some(token_str.to_string()));
    }

    // Enable progress based on verbosity
    if verbosity.show_progress() {
        api_builder = api_builder.with_progress(true);
    }

    let api = api_builder
        .build()
        .context("Failed to initialize HuggingFace Hub API client")?;

    // Create Repo with revision
    use hf_hub::{Repo, RepoType};
    let repo_obj = if let Some(rev) = revision {
        Repo::with_revision(repo_id.to_string(), RepoType::Model, rev.to_string())
    } else {
        Repo::new(repo_id.to_string(), RepoType::Model)
    };

    // Get the repository handle from the API
    let repo = api.repo(repo_obj);

    // Download the file (hf-hub handles caching and resumable downloads)
    let file_path = repo.get(filename).with_context(|| {
        format!(
            "Failed to download '{}' from repository '{}'{}",
            filename,
            repo_id,
            revision
                .map(|r| format!(" (revision: {})", r))
                .unwrap_or_default()
        )
    })?;

    if verbosity != Verbosity::Quiet {
        println!();
        println!("✓ Downloaded to: {}", file_path.display());
    }

    Ok(file_path)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Create the cache directory (expanding `~` if needed).
fn ensure_cache_dir(cache_dir: &Path) -> Result<PathBuf> {
    let expanded = crate::config::expand_tilde(cache_dir);
    std::fs::create_dir_all(&expanded)
        .with_context(|| format!("Failed to create cache directory: {}", expanded.display()))?;
    Ok(expanded)
}

/// Compute the lowercase-hex SHA-256 digest of a file's contents.
fn sha256_hex(path: &Path) -> Result<String> {
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

/// Check whether a file exists and is verified complete.
///
/// If `expected_sha256` is non-empty, this recomputes the file's SHA-256
/// digest and requires an exact (case-insensitive) match — the
/// authoritative check. Otherwise (no published checksum yet) this falls
/// back to a size floor: the file must be at least 90% of `expected_bytes`
/// (compressed archives can vary slightly, and `expected_bytes` is only an
/// estimate — not the true byte count — so exact equality is not
/// meaningful here).
///
/// This size floor alone is a weak integrity signal (it cannot catch
/// corruption of a similar length). The actual guarantee against a
/// truncated or interrupted download being mistaken for a complete, cached
/// file comes from `download_file`/`finalize_download`, which only ever
/// renames a `.part` staging file to `path` after the transfer completed
/// successfully and (when a checksum is available) passed verification.
fn is_cached(path: &Path, expected_bytes: u64, expected_sha256: &str) -> bool {
    let meta = match std::fs::metadata(path) {
        Ok(meta) => meta,
        Err(_) => return false,
    };
    if !expected_sha256.is_empty() {
        return sha256_hex(path)
            .map(|digest| digest.eq_ignore_ascii_case(expected_sha256))
            .unwrap_or(false);
    }
    if expected_bytes == 0 {
        meta.len() > 0
    } else {
        meta.len() >= expected_bytes * 9 / 10
    }
}

/// Verify a downloaded `.part` staging file and, on success, rename it to
/// its final `dest` path. On failure, the staging file is removed and an
/// error is returned — `dest` is only ever created by the rename on the
/// success path, so a caller that sees `Ok(())` knows the file at `dest` is
/// verified and a caller that sees `Err` knows no file was left at `dest`.
///
/// When `expected_sha256` is non-empty the downloaded bytes must hash to it
/// exactly. Otherwise, when `expected_bytes > 0`, the file must be at least
/// that many bytes (a truncated-transfer sanity check, not full integrity
/// verification — see [`is_cached`]).
fn finalize_download(
    part_path: &Path,
    dest: &Path,
    expected_bytes: u64,
    expected_sha256: &str,
) -> Result<()> {
    if !expected_sha256.is_empty() {
        let digest = sha256_hex(part_path)?;
        if !digest.eq_ignore_ascii_case(expected_sha256) {
            let _ = std::fs::remove_file(part_path);
            anyhow::bail!(
                "Checksum mismatch for {}: expected {expected_sha256}, got {digest}",
                dest.display()
            );
        }
    } else if expected_bytes > 0 {
        let actual = std::fs::metadata(part_path).map(|m| m.len()).unwrap_or(0);
        if actual < expected_bytes {
            let _ = std::fs::remove_file(part_path);
            anyhow::bail!(
                "Downloaded file for {} is smaller than expected ({actual} < \
                 {expected_bytes} bytes); the transfer may have been truncated",
                dest.display()
            );
        }
    }

    std::fs::rename(part_path, dest)
        .with_context(|| format!("Failed to finalize download at {}", dest.display()))?;
    Ok(())
}

/// Spawn `program` with `args` and poll `part_path`'s growing size to drive
/// real progress-bar updates while it runs, instead of jumping straight to
/// 100% only after the process exits.
///
/// Returns `None` if `program` could not be spawned at all (e.g. not
/// installed), or `Some(true/false)` for whether it exited successfully.
fn run_download_command<I, S>(
    program: &str,
    args: I,
    pb: &indicatif::ProgressBar,
    part_path: &Path,
    expected_bytes: u64,
) -> Option<bool>
where
    I: IntoIterator<Item = S>,
    S: AsRef<std::ffi::OsStr>,
{
    let mut child = std::process::Command::new(program)
        .args(args)
        .spawn()
        .ok()?;
    loop {
        if expected_bytes > 0 {
            if let Ok(meta) = std::fs::metadata(part_path) {
                pb.set_position(meta.len().min(expected_bytes));
            }
        }
        match child.try_wait() {
            Ok(Some(status)) => return Some(status.success()),
            Ok(None) => std::thread::sleep(std::time::Duration::from_millis(150)),
            Err(_) => return Some(false),
        }
    }
}

/// Download a file from `url` to `dest` using `curl` or `wget`, verifying
/// its integrity before it becomes visible at `dest` (see
/// [`finalize_download`]).
///
/// A progress bar is shown via `indicatif` while the download runs; its
/// position is refreshed by polling the growing `.part` staging file's size
/// so it reflects real transfer progress. Human-readable status lines are
/// suppressed when `json_mode` is set, so stdout stays valid JSON-only.
fn download_file(
    url: &str,
    dest: &Path,
    expected_bytes: u64,
    expected_sha256: &str,
    verbosity: Verbosity,
    json_mode: bool,
) -> Result<()> {
    // Ensure parent directory exists.
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let mut part_name = dest
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();
    part_name.push_str(".part");
    let part_path = dest.with_file_name(part_name);
    // Remove any stale partial file from a previous interrupted run so its
    // size can't be misread as progress from this attempt.
    let _ = std::fs::remove_file(&part_path);
    let part_str = part_path.to_string_lossy().to_string();

    // Progress bar (indeterminate or sized).
    let pb = if expected_bytes > 0 {
        progress::download_progress(expected_bytes, verbosity)
    } else {
        progress::spinner("Downloading...", verbosity)
    };

    // Try curl first (most common on Linux / macOS), falling back to wget.
    let (success, binary_missing) = match run_download_command(
        "curl",
        [
            "--fail",
            "--location",
            "--output",
            part_str.as_str(),
            "--silent",
            "--show-error",
            url,
        ],
        &pb,
        &part_path,
        expected_bytes,
    ) {
        Some(ok) => (ok, false),
        None => {
            tracing::debug!("curl not available, trying wget");
            match run_download_command(
                "wget",
                ["--quiet", "--output-document", part_str.as_str(), url],
                &pb,
                &part_path,
                expected_bytes,
            ) {
                Some(ok) => (ok, false),
                None => (false, true),
            }
        }
    };

    if !success {
        pb.abandon();
        let _ = std::fs::remove_file(&part_path);
        let dest_str = dest.to_string_lossy();
        if binary_missing {
            anyhow::bail!(
                "Download failed: neither `curl` nor `wget` is available on PATH. \
                 Please install one of them, or download manually:\n\
                 \n\
                 \x20  URL:  {url}\n\
                 \x20  Save: {dest_str}\n"
            )
        }
        anyhow::bail!(
            "Download failed (curl/wget exited with an error — the URL may be \
             unreachable or returned a non-success HTTP status). Please verify \
             the URL, or download manually:\n\
             \n\
             \x20  URL:  {url}\n\
             \x20  Save: {dest_str}\n"
        )
    }

    if let Err(e) = finalize_download(&part_path, dest, expected_bytes, expected_sha256) {
        pb.abandon();
        return Err(e);
    }

    if expected_bytes > 0 {
        if let Ok(meta) = std::fs::metadata(dest) {
            pb.set_position(meta.len());
        }
    }
    pb.finish_and_clear();
    if !json_mode {
        println!("     ✓  Saved to {}", dest.display());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn asset_manifest_has_entries() {
        assert!(!ASSETS.is_empty());
        for asset in ASSETS {
            assert!(!asset.name.is_empty());
            assert!(!asset.url.is_empty());
            assert!(!asset.filename.is_empty());
        }
    }

    #[test]
    fn expected_paths_match_manifest() {
        let cache_path = std::env::temp_dir().join("oxigaf_cache_test");
        let paths = expected_asset_paths(&cache_path);
        assert_eq!(paths.len(), ASSETS.len());
        assert!(paths[0].ends_with("flame2023.tar.gz"));
    }

    #[test]
    fn setup_cache_fails_fast_for_unpublished_assets() {
        // Every entry in ASSETS currently has an empty sha256, so setup_cache
        // must fail immediately (no network attempt) with an actionable
        // message rather than trying — and failing on — a doomed download.
        let cache_path = std::env::temp_dir().join("oxigaf_test_setup_unpublished");
        let _ = std::fs::remove_dir_all(&cache_path);

        let result = setup_cache(&cache_path, Verbosity::Quiet, false);
        assert!(result.is_err());
        let msg = format!("{:#}", result.unwrap_err());
        assert!(
            msg.contains("--from-hub") || msg.contains("not yet published"),
            "unexpected message: {msg}"
        );

        let _ = std::fs::remove_dir_all(&cache_path);
    }

    #[test]
    fn is_cached_missing_file_is_false() {
        let path = std::env::temp_dir().join("oxigaf_test_is_cached_missing.bin");
        let _ = std::fs::remove_file(&path);
        assert!(!is_cached(&path, 100, ""));
    }

    #[test]
    fn is_cached_size_floor_without_checksum() {
        let path = std::env::temp_dir().join("oxigaf_test_is_cached_size.bin");
        std::fs::write(&path, vec![0u8; 100]).expect("write test file");

        assert!(
            is_cached(&path, 100, ""),
            "exact size should pass the floor"
        );
        assert!(is_cached(&path, 50, ""), "larger-than-required should pass");
        assert!(
            is_cached(&path, 105, ""),
            "within-90%-tolerance should still pass (100 >= 105*9/10=94)"
        );
        assert!(
            !is_cached(&path, 200, ""),
            "well below the floor should fail"
        );

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn is_cached_checksum_exact_match_required() {
        let path = std::env::temp_dir().join("oxigaf_test_is_cached_checksum.bin");
        std::fs::write(&path, b"hello world").expect("write test file");
        let digest = sha256_hex(&path).expect("compute checksum");

        assert!(is_cached(&path, 0, &digest));
        assert!(
            is_cached(&path, 0, &digest.to_uppercase()),
            "match should be case-insensitive"
        );
        assert!(!is_cached(
            &path,
            0,
            "0000000000000000000000000000000000000000000000000000000000000000"
        ));

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn finalize_download_renames_on_success() {
        let part = std::env::temp_dir().join("oxigaf_test_finalize_ok.part");
        let dest = std::env::temp_dir().join("oxigaf_test_finalize_ok.bin");
        let _ = std::fs::remove_file(&part);
        let _ = std::fs::remove_file(&dest);
        std::fs::write(&part, vec![1u8; 10]).expect("write part file");

        finalize_download(&part, &dest, 10, "").expect("finalize should succeed");
        assert!(dest.exists());
        assert!(!part.exists());

        let _ = std::fs::remove_file(&dest);
    }

    #[test]
    fn finalize_download_rejects_truncated_file_without_checksum() {
        let part = std::env::temp_dir().join("oxigaf_test_finalize_short.part");
        let dest = std::env::temp_dir().join("oxigaf_test_finalize_short.bin");
        let _ = std::fs::remove_file(&part);
        let _ = std::fs::remove_file(&dest);
        std::fs::write(&part, vec![1u8; 5]).expect("write part file");

        let result = finalize_download(&part, &dest, 10, "");
        assert!(result.is_err());
        assert!(
            !dest.exists(),
            "truncated file must not be promoted to dest"
        );
        assert!(!part.exists(), "part file should be cleaned up on failure");
    }

    #[test]
    fn finalize_download_rejects_checksum_mismatch() {
        let part = std::env::temp_dir().join("oxigaf_test_finalize_badsum.part");
        let dest = std::env::temp_dir().join("oxigaf_test_finalize_badsum.bin");
        let _ = std::fs::remove_file(&part);
        let _ = std::fs::remove_file(&dest);
        std::fs::write(&part, b"hello world").expect("write part file");

        let result = finalize_download(
            &part,
            &dest,
            0,
            "0000000000000000000000000000000000000000000000000000000000000000",
        );
        assert!(result.is_err());
        assert!(!dest.exists());
        assert!(!part.exists());
    }

    #[test]
    fn finalize_download_accepts_valid_checksum() {
        let part = std::env::temp_dir().join("oxigaf_test_finalize_goodsum.part");
        let dest = std::env::temp_dir().join("oxigaf_test_finalize_goodsum.bin");
        let _ = std::fs::remove_file(&part);
        let _ = std::fs::remove_file(&dest);
        std::fs::write(&part, b"hello world").expect("write part file");
        let digest = sha256_hex(&part).expect("compute checksum");

        finalize_download(&part, &dest, 0, &digest).expect("finalize should succeed");
        assert!(dest.exists());

        let _ = std::fs::remove_file(&dest);
    }

    #[test]
    fn sha256_hex_is_deterministic_and_matches_known_vector() {
        let path = std::env::temp_dir().join("oxigaf_test_sha256_known.bin");
        std::fs::write(&path, b"hello world").expect("write test file");
        let digest = sha256_hex(&path).expect("compute checksum");
        // Known SHA-256("hello world")
        assert_eq!(
            digest,
            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde"
        );
        let _ = std::fs::remove_file(&path);
    }
}
