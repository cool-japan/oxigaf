//! Model download and cache management.
//!
//! Provides the `setup` command implementation: downloads required model weights
//! (FLAME, diffusion U-Net, VAE, CLIP) into a local cache directory and
//! verifies file integrity — via SHA-256 when a checksum has been
//! published for the asset, otherwise via a size-floor sanity check.
//!
//! [`setup_cache_with_options`] backs the command's `--skip-checksum` and
//! `--only` flags; [`setup_cache`] is the no-options shorthand. Every
//! completed download is registered in `cache.json` through
//! [`crate::cache::record_download`], and checksums are computed by
//! `crate::cache::compute_sha256` — this module deliberately carries no
//! digest implementation of its own.

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

/// Ensure all required model assets are present in `cache_dir`, downloading
/// the full manifest with checksum verification enabled.
///
/// Equivalent to [`setup_cache_with_options`] with `skip_checksum = false`
/// and `only = None`; see that function for the per-asset policy.
///
/// # Errors
///
/// Propagates [`setup_cache_with_options`].
pub fn setup_cache(cache_dir: &Path, verbosity: Verbosity, json_mode: bool) -> Result<()> {
    setup_cache_with_options(cache_dir, verbosity, json_mode, false, None)
}

/// Ensure the selected model assets are present in `cache_dir`.
///
/// For each *selected* asset (see `only` below):
/// 1. If the file already exists and passes verification (SHA-256 when
///    published and not skipped, otherwise a size-floor sanity check), skip
///    it.
/// 2. If it has no published checksum yet (an unpublished placeholder entry
///    — see the `ASSETS` doc), fail fast with a message pointing at
///    `oxigaf setup --from-hub` instead of attempting a download known to
///    fail.
/// 3. Otherwise, download it (via the built-in pure-Rust HTTP client) to a
///    `.part` staging file, verify it, make it visible at its final path,
///    and register it in `cache.json` via
///    [`crate::cache::record_download`] so `oxigaf cache list`/`verify`
///    have a precise download timestamp — and, when the checksum was
///    actually verified, that checksum — without re-deriving both from
///    filesystem metadata.
///
/// # Arguments
///
/// * `skip_checksum` — backs `oxigaf setup --skip-checksum`. Downgrades
///   verification for assets that *do* carry a published digest to the same
///   size floor used for assets that do not; it never disables verification
///   entirely, and it deliberately does **not** bypass the step-2 fail-fast:
///   an unpublished asset's URL 404s whether or not the caller wants its
///   digest checked, so pretending otherwise would only trade a precise
///   error for a network one.
/// * `only` — backs `oxigaf setup --only`: a comma-separated list of
///   substrings matched against asset file names, resolved through
///   [`crate::commands::runtime::select_assets`] so this path and the
///   `--offline` / `--dry-run` paths cannot disagree about what a filter
///   selects. A filter that matches nothing is an error rather than a
///   silent "nothing to do".
///
/// This function prints user-facing progress directly to stdout (suppressed
/// when `json_mode` is set).
///
/// # Errors
///
/// Returns an error if the cache directory cannot be created, if `only`
/// selects no assets, if a selected asset is unpublished, or if a download
/// or its verification fails.
pub fn setup_cache_with_options(
    cache_dir: &Path,
    verbosity: Verbosity,
    json_mode: bool,
    skip_checksum: bool,
    only: Option<&str>,
) -> Result<()> {
    let cache_dir = ensure_cache_dir(cache_dir)?;

    let expected = expected_asset_paths(&cache_dir);
    let selected = crate::commands::runtime::select_assets(&expected, only);
    if selected.is_empty() {
        let filter = only.unwrap_or_default();
        anyhow::bail!(
            "--only '{filter}' matched none of the {} known assets ({}). \
             Filters are matched as substrings of the asset file name.",
            ASSETS.len(),
            ASSETS
                .iter()
                .map(|a| a.filename)
                .collect::<Vec<_>>()
                .join(", ")
        );
    }

    if !json_mode {
        println!();
        println!("📦  OxiGAF Model Setup");
        println!("    Cache directory: {}", cache_dir.display());
        if selected.len() < ASSETS.len() {
            println!(
                "    Selected: {} of {} assets (--only)",
                selected.len(),
                ASSETS.len()
            );
        }
        if skip_checksum {
            println!("    Checksum verification: SKIPPED (--skip-checksum)");
        }
        println!();
    }

    let mut downloaded = 0usize;
    let mut skipped = 0usize;
    let mut downloaded_assets = Vec::new();

    for (asset, dest) in ASSETS.iter().zip(expected.iter()) {
        // `selected` borrows out of `expected`, which is built from `ASSETS`
        // in order, so membership is an identity check on the same `PathBuf`.
        if !selected.contains(&dest) {
            continue;
        }

        if is_cached(dest, asset.expected_bytes, asset.sha256, skip_checksum) {
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
            dest,
            asset.expected_bytes,
            asset.sha256,
            skip_checksum,
            verbosity,
            json_mode,
        )
        .with_context(|| format!("Failed to download {}", asset.name))?;
        downloaded += 1;

        // Only claim a checksum that was actually verified: under
        // `--skip-checksum` the digest was never compared, so recording it
        // would let a later `cache verify` "confirm" a value this run never
        // checked. Leaving it `None` makes verify do its own
        // trust-on-first-verify pass instead.
        let verified_checksum = if skip_checksum {
            None
        } else {
            Some(asset.sha256.to_string())
        };
        record_downloaded_asset(&cache_dir, asset.name, dest, verified_checksum);
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
                "selected_assets": selected.len(),
                "total_assets": ASSETS.len(),
                "only": only,
                "skip_checksum": skip_checksum,
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
#[must_use]
pub fn expected_asset_paths(cache_dir: &Path) -> Vec<PathBuf> {
    ASSETS.iter().map(|a| cache_dir.join(a.filename)).collect()
}

/// Register a freshly downloaded file in `cache_dir`'s `cache.json`.
///
/// Best-effort by construction: a multi-gigabyte download that completed and
/// verified must never be reported as a failure because its bookkeeping
/// entry could not be written (a read-only cache directory, a corrupt
/// `cache.json`). The failure is logged and the download stands — the
/// directory-scan fallback in [`crate::cache`] still finds the file.
fn record_downloaded_asset(cache_dir: &Path, name: &str, path: &Path, checksum: Option<String>) {
    if let Err(err) = crate::cache::record_download(cache_dir, name, path, checksum) {
        tracing::warn!(
            path = %path.display(),
            error = %format!("{err:#}"),
            "Downloaded asset could not be recorded in cache.json"
        );
    }
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
/// Implemented directly over the pure-Rust HTTP client (see the
/// HTTP / TLS plumbing section below) against the Hub's `resolve`
/// endpoint: `{endpoint}/{repo_id}/resolve/{revision}/{filename}`,
/// following redirects to the CDN. Downloads land in the
/// HuggingFace-compatible cache (`~/.cache/huggingface/hub`, or
/// `HF_HUB_CACHE` / `HF_HOME` when set) via a `.part` staging file that is
/// only renamed into place after the transfer completes, so an interrupted
/// download is never mistaken for a cached file.
///
/// # Arguments
///
/// * `repo_id` - HuggingFace repository identifier (e.g., "cool-japan/oxigaf-flame")
/// * `filename` - Filename within the repository
/// * `revision` - Optional revision (branch, tag, or commit SHA); defaults to "main"
/// * `token` - Optional authentication token (for private/gated repositories)
/// * `verbosity` - Controls progress display
///
/// # Returns
///
/// The path to the downloaded model file in the cache directory.
///
/// # Errors
///
/// Returns an error if:
/// - `repo_id`, `revision`, or `filename` is malformed (path traversal, wrong shape)
/// - The repository or file is not found (HTTP 404)
/// - Authentication fails for private models (HTTP 401/403)
/// - Network errors occur, or the transfer is truncated mid-stream
pub fn download_with_progress(
    repo_id: &str,
    filename: &str,
    revision: Option<&str>,
    token: Option<&str>,
    verbosity: Verbosity,
) -> Result<PathBuf> {
    let revision = revision.unwrap_or("main");
    validate_remote_component("repository id", repo_id)?;
    validate_remote_component("revision", revision)?;
    validate_remote_component("filename", filename)?;
    if repo_id.split('/').count() != 2 {
        anyhow::bail!("Invalid repository id '{repo_id}': expected 'organization/repository'");
    }

    let dest = hf_cache_path_in(&hf_cache_base(), repo_id, revision, filename);
    if let Ok(meta) = std::fs::metadata(&dest) {
        if meta.len() > 0 {
            if verbosity != Verbosity::Quiet {
                println!("✓ Cached: {}", dest.display());
            }
            return Ok(dest);
        }
    }

    if verbosity != Verbosity::Quiet {
        println!("📥 Downloading from HuggingFace Hub: {repo_id}");
        println!("   Revision: {revision}");
        println!("   File: {filename}");
        println!();
    }

    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create cache directory {}", parent.display()))?;
    }
    let mut part_name = dest
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();
    part_name.push_str(".part");
    let part_path = dest.with_file_name(part_name);
    let _ = std::fs::remove_file(&part_path);

    let url = hf_resolve_url(&hf_endpoint(), repo_id, revision, filename);
    // Sized once the response's Content-Length arrives (http_get_to_file
    // calls set_length); until then an indeterminate spinner.
    let pb = progress::spinner("Downloading...", verbosity);

    let agent = http_agent();
    let downloaded = match http_get_to_file(&agent, &url, token, &part_path, &pb) {
        Ok(bytes) => bytes,
        Err(err) => {
            pb.abandon();
            let _ = std::fs::remove_file(&part_path);
            return Err(err.context(format!(
                "Failed to download '{filename}' from repository '{repo_id}' \
                 (revision: {revision})"
            )));
        }
    };

    // No published checksum is available for arbitrary Hub files; integrity
    // against truncation is enforced by http_get_to_file's Content-Length
    // check, and the rename-only-on-success staging protocol.
    finalize_download(&part_path, &dest, 0, "", false)?;
    pb.finish_and_clear();

    // Register the download in OxiGAF's own cache metadata — not in the
    // HuggingFace cache root the file lives in — so `oxigaf cache
    // list`/`verify` can see a Hub download at all (they read
    // `default_cache_dir()`), instead of reporting an empty cache after a
    // multi-gigabyte fetch. The entry's `path` points outside that
    // directory, which `cache::clean_cache_report` explicitly refuses to
    // delete: OxiGAF must not sweep files out of another tool's cache.
    match crate::commands::runtime::default_cache_dir() {
        Ok(oxigaf_cache_dir) => record_downloaded_asset(
            &oxigaf_cache_dir,
            &format!("{repo_id}@{revision}/{filename}"),
            &dest,
            None,
        ),
        Err(err) => tracing::warn!(
            error = %format!("{err:#}"),
            "Could not determine the OxiGAF cache directory; Hub download not recorded"
        ),
    }

    if verbosity != Verbosity::Quiet {
        println!();
        println!("✓ Downloaded to: {} ({downloaded} bytes)", dest.display());
    }

    Ok(dest)
}

// ---------------------------------------------------------------------------
// HTTP / TLS plumbing (Pure Rust)
// ---------------------------------------------------------------------------

/// Install the process-wide rustls `CryptoProvider` exactly once.
///
/// ureq is built with `rustls-no-provider`, so rustls carries no built-in
/// crypto backend (this is what keeps `ring`'s C/asm out of the build); the
/// pure-Rust RustCrypto provider from `oxitls-rustcrypto-provider` is
/// installed as the process default before the first TLS connection. If the
/// embedding process already installed a provider, `install_default` returns
/// `Err` and that provider is respected instead.
fn ensure_tls_provider() {
    static INSTALL: std::sync::Once = std::sync::Once::new();
    INSTALL.call_once(|| {
        let _ =
            rustls::crypto::CryptoProvider::install_default(oxitls_rustcrypto_provider::provider());
    });
}

/// Build the HTTP agent used for all asset downloads (redirects followed,
/// bounded connect timeout, no timeout on the transfer itself — multi-GB
/// model downloads must not race a wall clock).
fn http_agent() -> ureq::Agent {
    ensure_tls_provider();
    ureq::Agent::config_builder()
        .timeout_connect(Some(std::time::Duration::from_secs(30)))
        .build()
        .new_agent()
}

/// HuggingFace endpoint (override with `HF_ENDPOINT`, e.g. for a mirror).
fn hf_endpoint() -> String {
    std::env::var("HF_ENDPOINT")
        .ok()
        .map(|s| s.trim().trim_end_matches('/').to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "https://huggingface.co".to_string())
}

/// `resolve` download URL for a file in a HuggingFace model repository.
///
/// The revision may itself contain slashes (e.g. `refs/pr/1`), which must be
/// percent-encoded so the Hub sees a single path segment.
fn hf_resolve_url(endpoint: &str, repo_id: &str, revision: &str, filename: &str) -> String {
    let encoded_rev = revision.replace('/', "%2F");
    format!("{endpoint}/{repo_id}/resolve/{encoded_rev}/{filename}")
}

/// Root of the HuggingFace-compatible download cache.
///
/// Honors `HF_HUB_CACHE`, then `HF_HOME` (as `$HF_HOME/hub`), then falls
/// back to `~/.cache/huggingface/hub` — the same order and default location
/// the HF ecosystem uses, so downloads can be shared with other tools.
fn hf_cache_base() -> PathBuf {
    if let Ok(dir) = std::env::var("HF_HUB_CACHE") {
        let dir = dir.trim();
        if !dir.is_empty() {
            return PathBuf::from(dir);
        }
    }
    if let Ok(dir) = std::env::var("HF_HOME") {
        let dir = dir.trim();
        if !dir.is_empty() {
            return PathBuf::from(dir).join("hub");
        }
    }
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".cache")
        .join("huggingface")
        .join("hub")
}

/// Local cache path for a repository file, under `base`.
///
/// Layout: `models--{org}--{name}/snapshots/{revision}/{filename}`,
/// mirroring the HF hub cache naming for repositories. Revisions are kept
/// under their given name rather than resolved to commit SHAs, so the cache
/// is self-consistent across runs of this tool without extra API round
/// trips.
fn hf_cache_path_in(base: &Path, repo_id: &str, revision: &str, filename: &str) -> PathBuf {
    let repo_dir = format!("models--{}", repo_id.replace('/', "--"));
    let rev_dir = revision.replace('/', "--");
    base.join(repo_dir)
        .join("snapshots")
        .join(rev_dir)
        .join(filename)
}

/// Reject path traversal and other malformed shapes in user-supplied
/// repo-id / revision / filename parts before they are joined into cache
/// paths or URLs. Multi-segment values ("org/repo", "unet/model.st",
/// "refs/pr/1") are legitimate; empty, `.`, or `..` segments, absolute
/// paths, and backslashes are not.
fn validate_remote_component(kind: &str, value: &str) -> Result<()> {
    let malformed = value.is_empty()
        || value.starts_with('/')
        || value.contains('\\')
        || value
            .split('/')
            .any(|seg| seg.is_empty() || seg == "." || seg == "..");
    if malformed {
        anyhow::bail!(
            "Invalid {kind} '{value}': must be a relative path without empty, \
             '.' or '..' segments"
        );
    }
    Ok(())
}

/// Stream an HTTP GET of `url` into `part_path`, driving `pb` with real byte
/// counts. Returns the number of bytes written.
///
/// When the server reports a Content-Length, `pb` is resized to it and a
/// short read is an error (truncated transfer); the staging file is removed
/// on that path. Other error paths leave `part_path` for the caller to clean
/// up.
fn http_get_to_file(
    agent: &ureq::Agent,
    url: &str,
    token: Option<&str>,
    part_path: &Path,
    pb: &indicatif::ProgressBar,
) -> Result<u64> {
    use std::io::{Read, Write};

    let mut request = agent.get(url);
    if let Some(token) = token {
        request = request.header("Authorization", &format!("Bearer {token}"));
    }
    let response = match request.call() {
        Ok(response) => response,
        Err(ureq::Error::StatusCode(code)) => {
            let hint = match code {
                401 | 403 => {
                    "the repository may be private or gated — set HF_TOKEN or \
                     place a token in ~/.huggingface/token"
                }
                404 => "the repository, revision, or file was not found",
                _ => "the server returned an unexpected status",
            };
            anyhow::bail!("HTTP {code} fetching {url} ({hint})");
        }
        Err(err) => {
            return Err(anyhow::Error::new(err).context(format!("Request to {url} failed")));
        }
    };

    let content_length = response
        .headers()
        .get(ureq::http::header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok());
    if let Some(len) = content_length {
        pb.set_length(len);
    }

    let (_, body) = response.into_parts();
    let mut reader = body.into_reader();
    let mut file = std::fs::File::create(part_path)
        .with_context(|| format!("Failed to create staging file {}", part_path.display()))?;
    let mut buf = [0u8; 64 * 1024];
    let mut total: u64 = 0;
    loop {
        let n = reader
            .read(&mut buf)
            .with_context(|| format!("Network read failed while downloading {url}"))?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])
            .with_context(|| format!("Failed writing to {}", part_path.display()))?;
        total += n as u64;
        pb.set_position(total);
    }
    file.flush()
        .with_context(|| format!("Failed flushing {}", part_path.display()))?;
    drop(file);

    if let Some(expected) = content_length {
        if total != expected {
            let _ = std::fs::remove_file(part_path);
            anyhow::bail!("Truncated transfer from {url}: got {total} of {expected} bytes");
        }
    }
    Ok(total)
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

/// Check whether a file exists and is verified complete.
///
/// If `expected_sha256` is non-empty and `skip_checksum` is not set, this
/// recomputes the file's SHA-256 digest and requires an exact
/// (case-insensitive) match — the authoritative check. Otherwise (no
/// published checksum yet, or the user passed `--skip-checksum`) this falls
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
/// successfully and (when a checksum is available and not skipped) passed
/// verification.
fn is_cached(path: &Path, expected_bytes: u64, expected_sha256: &str, skip_checksum: bool) -> bool {
    let meta = match std::fs::metadata(path) {
        Ok(meta) => meta,
        Err(_) => return false,
    };
    if !skip_checksum && !expected_sha256.is_empty() {
        return crate::cache::compute_sha256(path)
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
/// When `expected_sha256` is non-empty and `skip_checksum` is not set, the
/// downloaded bytes must hash to it exactly. Otherwise, when
/// `expected_bytes > 0`, the file must be at least that many bytes (a
/// truncated-transfer sanity check, not full integrity verification — see
/// [`is_cached`]). Note that `--skip-checksum` downgrades to that floor, it
/// does not disable verification altogether.
fn finalize_download(
    part_path: &Path,
    dest: &Path,
    expected_bytes: u64,
    expected_sha256: &str,
    skip_checksum: bool,
) -> Result<()> {
    if !skip_checksum && !expected_sha256.is_empty() {
        let digest = crate::cache::compute_sha256(part_path)?;
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

/// Download a file from `url` to `dest` using the built-in pure-Rust HTTP
/// client, verifying its integrity before it becomes visible at `dest` (see
/// [`finalize_download`]).
///
/// A progress bar is shown via `indicatif` while the download runs, driven
/// by real transferred-byte counts (and resized to the server's
/// Content-Length when reported, superseding the `expected_bytes`
/// estimate). Human-readable status lines are suppressed when `json_mode`
/// is set, so stdout stays valid JSON-only.
///
/// `skip_checksum` is forwarded to [`finalize_download`]; it downgrades
/// verification to the size floor rather than removing it.
fn download_file(
    url: &str,
    dest: &Path,
    expected_bytes: u64,
    expected_sha256: &str,
    skip_checksum: bool,
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

    // Progress bar (indeterminate or sized).
    let pb = if expected_bytes > 0 {
        progress::download_progress(expected_bytes, verbosity)
    } else {
        progress::spinner("Downloading...", verbosity)
    };

    let agent = http_agent();
    if let Err(err) = http_get_to_file(&agent, url, None, &part_path, &pb) {
        pb.abandon();
        let _ = std::fs::remove_file(&part_path);
        let dest_str = dest.to_string_lossy();
        return Err(err.context(format!(
            "Download failed — the URL may be unreachable or returned a \
             non-success HTTP status. Please verify the URL, or download \
             manually:\n\
             \n\
             \x20  URL:  {url}\n\
             \x20  Save: {dest_str}\n"
        )));
    }

    if let Err(e) = finalize_download(
        &part_path,
        dest,
        expected_bytes,
        expected_sha256,
        skip_checksum,
    ) {
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
    fn setup_cache_only_filter_selects_which_assets_are_attempted() {
        // Regression test for `--only`: with every manifest entry still
        // unpublished, the *first selected* asset is the one that trips the
        // fail-fast — so the name in the error proves the filter was applied
        // rather than warned about and dropped.
        let cache_path = std::env::temp_dir().join("oxigaf_test_setup_only_filter");
        let _ = std::fs::remove_dir_all(&cache_path);

        let unfiltered =
            setup_cache_with_options(&cache_path, Verbosity::Quiet, false, false, None)
                .expect_err("unpublished manifest must fail");
        assert!(
            format!("{unfiltered:#}").contains("FLAME"),
            "without a filter the first manifest entry is attempted first"
        );

        let filtered = setup_cache_with_options(
            &cache_path,
            Verbosity::Quiet,
            false,
            false,
            Some("vae_decoder"),
        )
        .expect_err("the selected asset is unpublished too");
        let msg = format!("{filtered:#}");
        assert!(
            msg.contains("VAE Decoder"),
            "--only must skip straight to the selected asset: {msg}"
        );
        assert!(
            !msg.contains("FLAME"),
            "--only must not attempt deselected assets: {msg}"
        );

        let _ = std::fs::remove_dir_all(&cache_path);
    }

    #[test]
    fn setup_cache_only_filter_matching_nothing_is_an_error() {
        // A filter that selects no asset must not report success ("all
        // assets already cached"), which would be a lie.
        let cache_path = std::env::temp_dir().join("oxigaf_test_setup_only_empty");
        let _ = std::fs::remove_dir_all(&cache_path);

        let err = setup_cache_with_options(
            &cache_path,
            Verbosity::Quiet,
            false,
            false,
            Some("no-such-asset"),
        )
        .expect_err("an unmatched filter must be an error");
        let msg = format!("{err:#}");
        assert!(msg.contains("--only"), "unexpected message: {msg}");
        assert!(
            msg.contains("no-such-asset"),
            "the message should echo the filter: {msg}"
        );

        let _ = std::fs::remove_dir_all(&cache_path);
    }

    #[test]
    fn setup_cache_blank_only_filter_means_no_filter() {
        // `select_assets` treats a whitespace-only `--only` as "no filter"
        // (it selects everything). Since the downloading path shares that
        // resolver, a blank filter must behave like `--only` was never
        // passed — the full manifest — not like a filter that matched
        // nothing.
        let cache_path = std::env::temp_dir().join("oxigaf_test_setup_only_blank");
        let _ = std::fs::remove_dir_all(&cache_path);

        let err = setup_cache_with_options(&cache_path, Verbosity::Quiet, false, false, Some("  "))
            .expect_err("unpublished manifest must fail");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("FLAME"),
            "a blank filter must select the whole manifest: {msg}"
        );
        assert!(
            !msg.contains("--only"),
            "a blank filter is not an unmatched filter: {msg}"
        );

        let _ = std::fs::remove_dir_all(&cache_path);
    }

    #[test]
    fn is_cached_missing_file_is_false() {
        let path = std::env::temp_dir().join("oxigaf_test_is_cached_missing.bin");
        let _ = std::fs::remove_file(&path);
        assert!(!is_cached(&path, 100, "", false));
    }

    #[test]
    fn is_cached_size_floor_without_checksum() {
        let path = std::env::temp_dir().join("oxigaf_test_is_cached_size.bin");
        std::fs::write(&path, vec![0u8; 100]).expect("write test file");

        assert!(
            is_cached(&path, 100, "", false),
            "exact size should pass the floor"
        );
        assert!(
            is_cached(&path, 50, "", false),
            "larger-than-required should pass"
        );
        assert!(
            is_cached(&path, 105, "", false),
            "within-90%-tolerance should still pass (100 >= 105*9/10=94)"
        );
        assert!(
            !is_cached(&path, 200, "", false),
            "well below the floor should fail"
        );

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn is_cached_checksum_exact_match_required() {
        let path = std::env::temp_dir().join("oxigaf_test_is_cached_checksum.bin");
        std::fs::write(&path, b"hello world").expect("write test file");
        let digest = crate::cache::compute_sha256(&path).expect("compute checksum");

        assert!(is_cached(&path, 0, &digest, false));
        assert!(
            is_cached(&path, 0, &digest.to_uppercase(), false),
            "match should be case-insensitive"
        );
        assert!(!is_cached(
            &path,
            0,
            "0000000000000000000000000000000000000000000000000000000000000000",
            false
        ));

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn is_cached_skip_checksum_falls_back_to_the_size_floor() {
        // Regression test for `--skip-checksum`: with a *published* digest
        // present, skipping must stop the digest comparison from running and
        // fall back to the size floor — not silently accept anything, and
        // not keep verifying.
        let path = std::env::temp_dir().join("oxigaf_test_is_cached_skipsum.bin");
        std::fs::write(&path, b"hello world").expect("write test file");
        let wrong = "0000000000000000000000000000000000000000000000000000000000000000";

        assert!(
            !is_cached(&path, 11, wrong, false),
            "a mismatching digest must fail while verification is on"
        );
        assert!(
            is_cached(&path, 11, wrong, true),
            "--skip-checksum must accept the file on its size alone"
        );
        assert!(
            !is_cached(&path, 10_000, wrong, true),
            "--skip-checksum still enforces the size floor"
        );

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn finalize_download_renames_on_success() {
        let part = std::env::temp_dir().join("oxigaf_test_finalize_ok.part");
        let dest = std::env::temp_dir().join("oxigaf_test_finalize_ok.bin");
        let _ = std::fs::remove_file(&part);
        let _ = std::fs::remove_file(&dest);
        std::fs::write(&part, vec![1u8; 10]).expect("write part file");

        finalize_download(&part, &dest, 10, "", false).expect("finalize should succeed");
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

        let result = finalize_download(&part, &dest, 10, "", false);
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
            false,
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
        let digest = crate::cache::compute_sha256(&part).expect("compute checksum");

        finalize_download(&part, &dest, 0, &digest, false).expect("finalize should succeed");
        assert!(dest.exists());

        let _ = std::fs::remove_file(&dest);
    }

    #[test]
    fn finalize_download_skip_checksum_accepts_a_mismatching_digest() {
        // `--skip-checksum` must actually reach the verification step: with
        // a published-but-wrong digest the file is promoted anyway…
        let part = std::env::temp_dir().join("oxigaf_test_finalize_skipsum.part");
        let dest = std::env::temp_dir().join("oxigaf_test_finalize_skipsum.bin");
        let _ = std::fs::remove_file(&part);
        let _ = std::fs::remove_file(&dest);
        std::fs::write(&part, b"hello world").expect("write part file");
        let wrong = "0000000000000000000000000000000000000000000000000000000000000000";

        finalize_download(&part, &dest, 0, wrong, true)
            .expect("skip_checksum should bypass the digest comparison");
        assert!(dest.exists());
        assert!(!part.exists());

        let _ = std::fs::remove_file(&dest);
    }

    #[test]
    fn finalize_download_skip_checksum_still_enforces_the_size_floor() {
        // …but skipping the digest must not disable verification wholesale:
        // a truncated transfer is still rejected on size.
        let part = std::env::temp_dir().join("oxigaf_test_finalize_skipsum_short.part");
        let dest = std::env::temp_dir().join("oxigaf_test_finalize_skipsum_short.bin");
        let _ = std::fs::remove_file(&part);
        let _ = std::fs::remove_file(&dest);
        std::fs::write(&part, b"hello world").expect("write part file");
        let wrong = "0000000000000000000000000000000000000000000000000000000000000000";

        let result = finalize_download(&part, &dest, 10_000, wrong, true);
        assert!(result.is_err());
        assert!(!dest.exists(), "truncated file must not be promoted");
        assert!(!part.exists());
    }

    #[test]
    fn hf_resolve_url_encodes_revision_slashes() {
        let url = hf_resolve_url(
            "https://huggingface.co",
            "org/repo",
            "refs/pr/1",
            "model.safetensors",
        );
        assert_eq!(
            url,
            "https://huggingface.co/org/repo/resolve/refs%2Fpr%2F1/model.safetensors"
        );

        let url = hf_resolve_url(
            "https://huggingface.co",
            "cool-japan/oxigaf-flame",
            "main",
            "unet/model.safetensors",
        );
        assert_eq!(
            url,
            "https://huggingface.co/cool-japan/oxigaf-flame/resolve/main/unet/model.safetensors"
        );
    }

    #[test]
    fn hf_cache_path_mirrors_hub_layout() {
        let base = std::env::temp_dir().join("oxigaf_test_hf_cache");
        let path = hf_cache_path_in(
            &base,
            "cool-japan/oxigaf-flame",
            "main",
            "model.safetensors",
        );
        assert_eq!(
            path,
            base.join("models--cool-japan--oxigaf-flame")
                .join("snapshots")
                .join("main")
                .join("model.safetensors")
        );

        // Slashed revisions must not create extra directory levels beyond
        // the snapshot dir.
        let path = hf_cache_path_in(&base, "org/repo", "refs/pr/1", "m.safetensors");
        assert_eq!(
            path,
            base.join("models--org--repo")
                .join("snapshots")
                .join("refs--pr--1")
                .join("m.safetensors")
        );
    }

    #[test]
    fn remote_component_validation_rejects_traversal() {
        assert!(validate_remote_component("filename", "../secrets").is_err());
        assert!(validate_remote_component("filename", "a/../b").is_err());
        assert!(validate_remote_component("filename", "/abs").is_err());
        assert!(validate_remote_component("filename", "a\\b").is_err());
        assert!(validate_remote_component("filename", "a//b").is_err());
        assert!(validate_remote_component("filename", ".").is_err());
        assert!(validate_remote_component("filename", "").is_err());

        assert!(validate_remote_component("filename", "model.safetensors").is_ok());
        assert!(validate_remote_component("filename", "unet/model.safetensors").is_ok());
        assert!(validate_remote_component("repository id", "cool-japan/oxigaf-flame").is_ok());
        assert!(validate_remote_component("revision", "refs/pr/1").is_ok());
    }

    #[test]
    fn download_with_progress_rejects_malformed_repo_id() {
        let result = download_with_progress(
            "no-slash-repo/extra/segments",
            "model.safetensors",
            None,
            None,
            Verbosity::Quiet,
        );
        assert!(result.is_err());

        let result =
            download_with_progress("../evil", "model.safetensors", None, None, Verbosity::Quiet);
        assert!(result.is_err());
    }

    #[test]
    fn shared_sha256_is_deterministic_and_matches_known_vector() {
        // `assets` no longer carries its own `sha256_hex`: it defers to the
        // streaming implementation in `cache`, which this pins to the same
        // known-answer vector the deleted twin was pinned to.
        let path = std::env::temp_dir().join("oxigaf_test_sha256_known.bin");
        std::fs::write(&path, b"hello world").expect("write test file");
        let digest = crate::cache::compute_sha256(&path).expect("compute checksum");
        // Known SHA-256("hello world"), independently verified via both
        // `shasum -a 256` and Python's hashlib during test authoring.
        assert_eq!(digest.len(), 64, "SHA-256 hex digest must be 64 chars");
        assert_eq!(
            digest,
            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        );
        let _ = std::fs::remove_file(&path);
    }
}
