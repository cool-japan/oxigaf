//! Model download and cache management.
//!
//! Provides the `setup` command implementation: downloads required model weights
//! (FLAME, diffusion U-Net, VAE, CLIP) into a local cache directory and
//! verifies file integrity by size.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::progress;
use crate::verbosity::Verbosity;

// ---------------------------------------------------------------------------
// Asset manifest
// ---------------------------------------------------------------------------

/// Metadata for a single downloadable model asset.
#[allow(dead_code)]
struct Asset {
    /// Human-readable name shown in progress output.
    name: &'static str,
    /// Download URL.
    url: &'static str,
    /// Filename within the cache directory.
    filename: &'static str,
    /// Expected file size in bytes (used for progress reporting and simple
    /// integrity checks). Set to 0 to skip the size check.
    expected_bytes: u64,
    /// Optional SHA-256 hex digest. Left empty for now — a future release will
    /// add checksums for all official model bundles.
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

    /// Download the model from HuggingFace Hub.
    ///
    /// # Arguments
    ///
    /// * `token` - Optional HuggingFace authentication token for private models
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
    #[allow(dead_code)]
    pub fn download(&self, token: Option<&str>) -> Result<PathBuf> {
        use hf_hub::api::sync::ApiBuilder;

        println!("📥 Downloading from HuggingFace Hub: {}", self.repo_id);
        if let Some(ref rev) = self.revision {
            println!("   Revision: {}", rev);
        }
        println!("   File: {}", self.filename);

        // Build API client with optional token
        let mut api_builder = ApiBuilder::new();

        if let Some(token_str) = token {
            api_builder = api_builder.with_token(Some(token_str.to_string()));
        }

        let api = api_builder
            .build()
            .context("Failed to initialize HuggingFace Hub API client")?;

        // Create Repo with revision
        use hf_hub::{Repo, RepoType};
        let repo_obj = if let Some(ref rev) = self.revision {
            Repo::with_revision(self.repo_id.clone(), RepoType::Model, rev.clone())
        } else {
            Repo::new(self.repo_id.clone(), RepoType::Model)
        };

        // Get the repository handle from the API
        let repo = api.repo(repo_obj);

        // Download the file (hf-hub handles caching and resumable downloads)
        let file_path = repo.get(&self.filename).with_context(|| {
            format!(
                "Failed to download '{}' from repository '{}'{}",
                self.filename,
                self.repo_id,
                self.revision
                    .as_ref()
                    .map(|r| format!(" (revision: {})", r))
                    .unwrap_or_default()
            )
        })?;

        println!("✓ Downloaded to: {}", file_path.display());

        Ok(file_path)
    }
}

/// Placeholder asset manifest.
///
/// The URLs below point at the project's GitHub releases page. Replace them
/// with real artifact URLs once the model weights are published.
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
/// 1. If the file already exists (and has a reasonable size), skip it.
/// 2. Otherwise, attempt to download it via `curl` (or `wget` as a fallback).
///
/// This function prints user-facing progress directly to stdout.
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

        if is_cached(&dest, asset.expected_bytes) {
            if !json_mode {
                println!("  ✓  {} (cached)", asset.name);
            }
            skipped += 1;
            continue;
        }

        if !json_mode {
            println!("  ⬇  Downloading {} …", asset.name);
        }
        download_file(asset.url, &dest, asset.expected_bytes, verbosity)
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

/// Check whether a file exists and looks complete (non-zero, at least 90% of
/// the expected size).
fn is_cached(path: &Path, expected_bytes: u64) -> bool {
    match std::fs::metadata(path) {
        Ok(meta) => {
            let size = meta.len();
            if expected_bytes == 0 {
                size > 0
            } else {
                // Accept if within 90% — compressed archives may vary slightly.
                size >= expected_bytes * 9 / 10
            }
        }
        Err(_) => false,
    }
}

/// Download a file from `url` to `dest` using `curl` or `wget`.
///
/// A progress bar is shown via `indicatif` while the download runs.
fn download_file(url: &str, dest: &Path, expected_bytes: u64, verbosity: Verbosity) -> Result<()> {
    // Ensure parent directory exists.
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let dest_str = dest.to_string_lossy().to_string();

    // Progress bar (indeterminate or sized).
    let pb = if expected_bytes > 0 {
        progress::download_progress(expected_bytes, verbosity)
    } else {
        progress::spinner("Downloading...", verbosity)
    };

    // Try curl first (most common on Linux / macOS).
    let curl_result = std::process::Command::new("curl")
        .args([
            "--fail",
            "--location",
            "--output",
            &dest_str,
            "--silent",
            "--show-error",
            url,
        ])
        .status();

    let success = match curl_result {
        Ok(status) if status.success() => true,
        _ => {
            // Fall back to wget.
            tracing::debug!("curl not available or failed, trying wget");
            let wget_result = std::process::Command::new("wget")
                .args(["--quiet", "--output-document", &dest_str, url])
                .status();

            matches!(wget_result, Ok(status) if status.success())
        }
    };

    if success {
        // Update progress bar to completion.
        if expected_bytes > 0 {
            if let Ok(meta) = std::fs::metadata(dest) {
                pb.set_position(meta.len());
            }
        }
        pb.finish_and_clear();
        println!("     ✓  Saved to {}", dest.display());
        Ok(())
    } else {
        pb.abandon();
        // Clean up partial download.
        let _ = std::fs::remove_file(dest);
        anyhow::bail!(
            "Download failed. Please install `curl` or `wget`, or download manually:\n\
             \n\
             \x20  URL:  {url}\n\
             \x20  Save: {dest_str}\n"
        )
    }
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
}
