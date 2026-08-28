//! Unified progress bar utilities with consistent styling.
//!
//! Provides reusable progress bar helpers for various CLI operations:
//! - Training iterations
//! - File downloads
//! - Rendering frames
//! - Export operations
//! - Indeterminate spinners
//! - Multi-progress for parallel operations
//!
//! All progress bars respect verbosity settings and use a consistent
//! color scheme (green spinner, cyan/blue bar).

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};

use crate::verbosity::Verbosity;

// ---------------------------------------------------------------------------
// Progress Bar Helpers
// ---------------------------------------------------------------------------

/// Create a progress bar for training iterations.
///
/// Shows iteration count, loss value, and estimated time to completion.
///
/// # Arguments
///
/// * `total` - Total number of training iterations
/// * `verbosity` - Verbosity level to respect
///
/// # Returns
///
/// A configured progress bar, or hidden bar if progress shouldn't be shown.
///
/// # Examples
///
/// ```no_run
/// use oxigaf_cli::progress;
/// use oxigaf_cli::verbosity::Verbosity;
///
/// let pb = progress::training_progress(1000, Verbosity::Normal);
/// for i in 0..1000 {
///     pb.set_message(format!("0.{:04}", 1000 - i));
///     pb.inc(1);
/// }
/// pb.finish_with_message("Training complete");
/// ```
#[must_use]
pub fn training_progress(total: u64, verbosity: Verbosity) -> ProgressBar {
    if !verbosity.show_progress() {
        return ProgressBar::hidden();
    }

    let pb = ProgressBar::new(total);
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} | loss: {msg} | ETA: {eta}",
        )
        .unwrap_or_else(|_| ProgressStyle::default_bar())
        .progress_chars("=>-"),
    );
    pb
}

/// Create a progress bar for file downloads.
///
/// Shows bytes downloaded, download speed, and estimated time remaining.
///
/// # Arguments
///
/// * `total_bytes` - Total size of download in bytes
/// * `verbosity` - Verbosity level to respect
///
/// # Returns
///
/// A configured progress bar, or hidden bar if progress shouldn't be shown.
///
/// # Examples
///
/// ```no_run
/// use oxigaf_cli::progress;
/// use oxigaf_cli::verbosity::Verbosity;
///
/// let pb = progress::download_progress(1024 * 1024 * 100, Verbosity::Normal);
/// // Update as bytes are received
/// pb.inc(4096);
/// pb.finish_with_message("Download complete");
/// ```
#[must_use]
pub fn download_progress(total_bytes: u64, verbosity: Verbosity) -> ProgressBar {
    if !verbosity.show_progress() {
        return ProgressBar::hidden();
    }

    let pb = ProgressBar::new(total_bytes);
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}) | ETA: {eta}",
        )
        .unwrap_or_else(|_| ProgressStyle::default_bar())
        .progress_chars("=>-"),
    );
    pb
}

/// Create a progress bar for rendering frames.
///
/// Shows frame count and rendering progress with ETA.
///
/// `oxigaf render` schedules its frames through
/// [`crate::parallel_render::ParallelRenderer`], whose `execute` takes a
/// [`crate::progress_types::BatchProgress`] rather than a bare
/// [`ProgressBar`], so it drives that type instead. This helper stays the
/// styling for any caller that renders frames in its own loop; it used to
/// carry an `#[allow(dead_code)]` for that reason, which was never needed —
/// it is public API of the library crate, not a private binary item.
///
/// # Arguments
///
/// * `num_frames` - Total number of frames to render
/// * `verbosity` - Verbosity level to respect
///
/// # Returns
///
/// A configured progress bar, or hidden bar if progress shouldn't be shown.
///
/// # Examples
///
/// ```no_run
/// use oxigaf_cli::progress;
/// use oxigaf_cli::verbosity::Verbosity;
///
/// let pb = progress::render_progress(120, Verbosity::Normal);
/// for i in 0..120 {
///     pb.set_message(format!("frame {:03}", i));
///     pb.inc(1);
/// }
/// pb.finish_with_message("Rendering complete");
/// ```
#[must_use]
pub fn render_progress(num_frames: u64, verbosity: Verbosity) -> ProgressBar {
    if !verbosity.show_progress() {
        return ProgressBar::hidden();
    }

    let pb = ProgressBar::new(num_frames);
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} frames | {msg} | ETA: {eta}",
        )
        .unwrap_or_else(|_| ProgressStyle::default_bar())
        .progress_chars("=>-"),
    );
    pb
}

/// Create a progress bar for export operations.
///
/// Shows item count with custom message support.
///
/// # Arguments
///
/// * `num_items` - Total number of items to export
/// * `verbosity` - Verbosity level to respect
///
/// # Returns
///
/// A configured progress bar, or hidden bar if progress shouldn't be shown.
///
/// # Examples
///
/// ```no_run
/// use oxigaf_cli::progress;
/// use oxigaf_cli::verbosity::Verbosity;
///
/// let pb = progress::export_progress(50, Verbosity::Normal);
/// pb.set_message("extracting");
/// for i in 0..50 {
///     pb.inc(1);
/// }
/// pb.finish_with_message("Export complete");
/// ```
#[must_use]
pub fn export_progress(num_items: u64, verbosity: Verbosity) -> ProgressBar {
    if !verbosity.show_progress() {
        return ProgressBar::hidden();
    }

    let pb = ProgressBar::new(num_items);
    pb.set_style(
        ProgressStyle::with_template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} | {msg}")
            .unwrap_or_else(|_| ProgressStyle::default_bar())
            .progress_chars("=>-"),
    );
    pb
}

/// Create a spinner for indeterminate operations.
///
/// Use when the total work amount is unknown or for quick operations.
///
/// # Arguments
///
/// * `message` - Message to display next to the spinner
/// * `verbosity` - Verbosity level to respect
///
/// # Returns
///
/// A configured spinner, or hidden spinner if progress shouldn't be shown.
///
/// # Examples
///
/// ```no_run
/// use oxigaf_cli::progress;
/// use oxigaf_cli::verbosity::Verbosity;
///
/// let pb = progress::spinner("Downloading...", Verbosity::Normal);
/// // Do work
/// pb.finish_with_message("Done!");
/// ```
#[must_use]
pub fn spinner(message: &str, verbosity: Verbosity) -> ProgressBar {
    if !verbosity.show_progress() {
        return ProgressBar::hidden();
    }

    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::with_template("{spinner:.green} {msg}")
            .unwrap_or_else(|_| ProgressStyle::default_spinner()),
    );
    pb.set_message(message.to_string());
    pb
}

/// Create a multi-progress container for parallel operations.
///
/// Allows multiple progress bars to be displayed simultaneously.
///
/// # Arguments
///
/// * `verbosity` - Verbosity level to respect
///
/// # Returns
///
/// A multi-progress container if progress should be shown, None otherwise.
///
/// # Examples
///
/// ```no_run
/// use oxigaf_cli::progress;
/// use oxigaf_cli::verbosity::Verbosity;
///
/// if let Some(multi) = progress::multi_progress(Verbosity::Normal) {
///     let pb1 = multi.add(progress::render_progress(100, Verbosity::Normal));
///     let pb2 = multi.add(progress::render_progress(100, Verbosity::Normal));
///     // Use pb1 and pb2 for parallel work
/// }
/// ```
#[must_use]
pub fn multi_progress(verbosity: Verbosity) -> Option<MultiProgress> {
    if verbosity.show_progress() {
        Some(MultiProgress::new())
    } else {
        None
    }
}

/// Create a generic progress bar with custom template.
///
/// For specialized use cases not covered by the predefined helpers.
///
/// # Arguments
///
/// * `total` - Total number of items/iterations
/// * `template` - Custom template string (see indicatif documentation)
/// * `verbosity` - Verbosity level to respect
///
/// # Returns
///
/// A configured progress bar, or hidden bar if progress shouldn't be shown.
///
/// # Examples
///
/// ```no_run
/// use oxigaf_cli::progress;
/// use oxigaf_cli::verbosity::Verbosity;
///
/// let pb = progress::custom_progress(
///     1000,
///     "{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} iterations | {msg}",
///     Verbosity::Normal,
/// );
/// ```
#[must_use]
pub fn custom_progress(total: u64, template: &str, verbosity: Verbosity) -> ProgressBar {
    if !verbosity.show_progress() {
        return ProgressBar::hidden();
    }

    let pb = ProgressBar::new(total);
    pb.set_style(
        ProgressStyle::with_template(template)
            .unwrap_or_else(|_| ProgressStyle::default_bar())
            .progress_chars("=>-"),
    );
    pb
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn training_progress_hidden_in_quiet_mode() {
        let pb = training_progress(100, Verbosity::Quiet);
        assert!(pb.is_hidden());
    }

    #[test]
    fn training_progress_hidden_in_debug_mode() {
        let pb = training_progress(100, Verbosity::Debug);
        assert!(pb.is_hidden());
    }

    #[test]
    fn training_progress_hidden_in_trace_mode() {
        let pb = training_progress(100, Verbosity::Trace);
        assert!(pb.is_hidden());
    }

    #[test]
    fn training_progress_visible_in_normal_mode() {
        let pb = training_progress(100, Verbosity::Normal);
        // Check that progress bar has the expected length
        assert_eq!(pb.length(), Some(100));
    }

    #[test]
    fn training_progress_visible_in_verbose_mode() {
        let pb = training_progress(100, Verbosity::Verbose);
        // Check that progress bar has the expected length
        assert_eq!(pb.length(), Some(100));
    }

    #[test]
    fn download_progress_hidden_in_quiet_mode() {
        let pb = download_progress(1000, Verbosity::Quiet);
        assert!(pb.is_hidden());
    }

    #[test]
    fn download_progress_visible_in_normal_mode() {
        let pb = download_progress(1000, Verbosity::Normal);
        // Check that progress bar has the expected length
        assert_eq!(pb.length(), Some(1000));
    }

    #[test]
    fn render_progress_respects_verbosity() {
        assert!(render_progress(100, Verbosity::Quiet).is_hidden());
        assert_eq!(render_progress(100, Verbosity::Normal).length(), Some(100));
        assert_eq!(render_progress(100, Verbosity::Verbose).length(), Some(100));
        assert!(render_progress(100, Verbosity::Debug).is_hidden());
        assert!(render_progress(100, Verbosity::Trace).is_hidden());
    }

    #[test]
    fn export_progress_respects_verbosity() {
        assert!(export_progress(50, Verbosity::Quiet).is_hidden());
        assert_eq!(export_progress(50, Verbosity::Normal).length(), Some(50));
        assert_eq!(export_progress(50, Verbosity::Verbose).length(), Some(50));
        assert!(export_progress(50, Verbosity::Debug).is_hidden());
    }

    #[test]
    fn spinner_respects_verbosity() {
        let pb = spinner("test", Verbosity::Quiet);
        assert!(pb.is_hidden());

        let pb = spinner("test", Verbosity::Normal);
        // Spinners have no length, just check it's not hidden by checking it has a style
        assert!(!pb.is_finished());

        let pb = spinner("test", Verbosity::Verbose);
        assert!(!pb.is_finished());

        let pb = spinner("test", Verbosity::Debug);
        assert!(pb.is_hidden());
    }

    #[test]
    fn multi_progress_respects_verbosity() {
        assert!(multi_progress(Verbosity::Quiet).is_none());
        assert!(multi_progress(Verbosity::Normal).is_some());
        assert!(multi_progress(Verbosity::Verbose).is_some());
        assert!(multi_progress(Verbosity::Debug).is_none());
    }

    #[test]
    fn custom_progress_respects_verbosity() {
        let template = "{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len}";

        assert!(custom_progress(100, template, Verbosity::Quiet).is_hidden());
        assert_eq!(
            custom_progress(100, template, Verbosity::Normal).length(),
            Some(100)
        );
        assert_eq!(
            custom_progress(100, template, Verbosity::Verbose).length(),
            Some(100)
        );
        assert!(custom_progress(100, template, Verbosity::Debug).is_hidden());
    }

    #[test]
    fn progress_bars_have_correct_length() {
        let pb = training_progress(1000, Verbosity::Normal);
        assert_eq!(pb.length(), Some(1000));

        let pb = download_progress(2048, Verbosity::Normal);
        assert_eq!(pb.length(), Some(2048));

        let pb = render_progress(120, Verbosity::Normal);
        assert_eq!(pb.length(), Some(120));

        let pb = export_progress(50, Verbosity::Normal);
        assert_eq!(pb.length(), Some(50));
    }
}
