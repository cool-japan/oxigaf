//! Image loading and input-validation helpers shared by the image-oriented
//! command families (`quality`, `video`).
//!
//! # Why the error type matters here
//!
//! Every failure in this file is "the caller named a file or directory that is
//! not usable". [`crate::error::CliError::InputInvalid`] maps that to
//! [`crate::error::EXIT_IO_ERROR`] (3), so a script can tell a bad path from a
//! genuine processing failure without parsing the message. Returning a bare
//! `anyhow::bail!` here would collapse it into the catch-all exit status 1,
//! which is the defect the exit-code taxonomy exists to prevent — see
//! [`crate::commands::runtime::to_cli_error`], which downcasts whatever a
//! handler returns back to a [`crate::error::CliError`].

use std::path::{Path, PathBuf};

use anyhow::Result;

use crate::error::CliError;

/// File extensions treated as still images by the directory scanners.
const IMAGE_EXTENSIONS: [&str; 6] = ["png", "jpg", "jpeg", "bmp", "tiff", "tif"];

/// Build the "this input cannot be used" error for `path`.
///
/// Exposed so handlers can report a bad *argument* (an empty directory, a
/// mismatched pair) with the same exit status as a bad *file*.
#[must_use]
pub fn input_invalid(path: &Path, reason: impl Into<String>) -> anyhow::Error {
    CliError::InputInvalid {
        path: path.to_path_buf(),
        reason: reason.into(),
    }
    .into()
}

/// Load an image as flat interleaved RGBA bytes plus its dimensions.
///
/// RGBA (not RGB) is the layout every [`crate::quality_checker`] and
/// [`crate::video_export`] entry point expects.
///
/// # Errors
///
/// Returns [`CliError::InputInvalid`] when the file is missing or is not a
/// decodable image.
pub fn load_rgba(path: &Path) -> Result<(Vec<u8>, u32, u32)> {
    if !path.is_file() {
        return Err(input_invalid(path, "not an existing file"));
    }
    let image = image::open(path)
        .map_err(|e| input_invalid(path, format!("could not be decoded as an image: {e}")))?
        .to_rgba8();
    let (width, height) = (image.width(), image.height());
    Ok((image.into_raw(), width, height))
}

/// List the image files in `dir`, sorted by file name.
///
/// The sort is what pairs a rendered frame with its reference: both
/// directories are walked with the same ordering rule, so `frame_003.png`
/// lines up with `frame_003.png` and not with whatever the filesystem
/// happened to return third.
///
/// # Errors
///
/// Returns [`CliError::InputInvalid`] when `dir` is not a directory, and an
/// I/O error when it cannot be read.
pub fn image_files(dir: &Path) -> Result<Vec<PathBuf>> {
    if !dir.is_dir() {
        return Err(input_invalid(dir, "not an existing directory"));
    }
    let entries = std::fs::read_dir(dir).map_err(|e| CliError::IoError {
        context: format!("Failed to read directory: {}", dir.display()),
        source: e,
    })?;
    let mut files: Vec<PathBuf> = entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.is_file() && has_image_extension(path))
        .collect();
    files.sort();
    Ok(files)
}

/// Whether `path` carries one of [`IMAGE_EXTENSIONS`], case-insensitively.
fn has_image_extension(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| {
            let lower = e.to_ascii_lowercase();
            IMAGE_EXTENSIONS.contains(&lower.as_str())
        })
        .unwrap_or(false)
}

/// Parse an `rrggbb` (or `#rrggbb`) hex colour for a clap `value_parser`.
///
/// # Errors
///
/// Returns a human-readable message when the input is not six hex digits.
pub fn parse_hex_rgb(raw: &str) -> std::result::Result<[u8; 3], String> {
    let text = raw.strip_prefix('#').unwrap_or(raw);
    if text.len() != 6 || !text.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(format!(
            "expected six hexadecimal digits (rrggbb), got {raw:?}"
        ));
    }
    let mut out = [0u8; 3];
    for (index, slot) in out.iter_mut().enumerate() {
        let start = index * 2;
        let pair = text.get(start..start + 2).ok_or_else(|| {
            // Unreachable given the length check above, but the no-panic
            // policy rules out indexing a &str directly.
            format!("{raw:?} is not a six-digit hex colour")
        })?;
        *slot = u8::from_str_radix(pair, 16)
            .map_err(|_| format!("{pair:?} is not a hexadecimal byte"))?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_hex_rgb_accepts_both_spellings() {
        assert_eq!(parse_hex_rgb("ff8000"), Ok([255, 128, 0]));
        assert_eq!(parse_hex_rgb("#00FF7F"), Ok([0, 255, 127]));
    }

    #[test]
    fn parse_hex_rgb_rejects_bad_input() {
        assert!(parse_hex_rgb("fff").is_err());
        assert!(parse_hex_rgb("gggggg").is_err());
        assert!(parse_hex_rgb("ff80000").is_err());
    }

    /// A missing input must surface as `InputInvalid`, not as the catch-all
    /// `Other`, or `oxigaf quality check` would exit 1 for a typo'd path
    /// while `oxigaf train` exits 3 for the same mistake.
    #[test]
    fn load_rgba_reports_a_missing_file_as_input_invalid() {
        let path = std::env::temp_dir().join("oxigaf_image_io_missing.png");
        let _ = std::fs::remove_file(&path);
        let err = load_rgba(&path).expect_err("a missing file must not load");
        let cli_err = crate::commands::runtime::to_cli_error(err);
        assert_eq!(cli_err.exit_code(), crate::error::EXIT_IO_ERROR);
    }

    #[test]
    fn image_files_rejects_a_non_directory() {
        let path = std::env::temp_dir().join("oxigaf_image_io_not_a_dir.txt");
        std::fs::write(&path, b"x").expect("temp file write");
        assert!(image_files(&path).is_err());
        let _ = std::fs::remove_file(&path);
    }
}
