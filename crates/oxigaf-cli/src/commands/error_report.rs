//! Turning a failed command into an exit code and a message a user can act on.
//!
//! [`crate::commands::runtime::to_cli_error`] classifies a failure by
//! downcasting to the *innermost* concrete error, which is what selects the
//! process exit status from [`crate::error`]'s taxonomy. That downcast is
//! also lossy: its `std::io::Error` arm rebuilds
//! [`crate::error::CliError::IoError`] from `io_err.to_string()` alone, so
//! every `anyhow` context layer above it is discarded.
//!
//! The effect was that `oxigaf render -m missing.ply` — whose handler adds
//! the context "Failed to load model: missing.ply" — reported nothing but
//! "I/O error: No such file or directory (os error 2)". The operation and
//! the path, the only two things that tell a user what to fix, never reached
//! them.
//!
//! [`classify_error`] renders the chain *before* the conversion consumes the
//! error and re-attaches it, so [`crate::output::display_error`] remains the
//! single place that decides how an error looks.

use crate::commands::runtime::to_cli_error;
use crate::error::CliError;

/// Classify a failure for its exit code without losing its context chain.
///
/// Returns the classified [`CliError`] — whose [`CliError::exit_code`] the
/// process exits with — and the fully rendered `anyhow` chain, which is what
/// both the human renderer and the `--json` error document should show.
#[must_use]
pub fn classify_error(err: anyhow::Error) -> (CliError, String) {
    let detail = format!("{err:#}");
    let classified = match to_cli_error(err) {
        CliError::IoError { source, .. } => CliError::IoError {
            context: detail.clone(),
            source,
        },
        // The remaining variants were either constructed with their own
        // message or wrap the `anyhow::Error` whole, so their context
        // survives the classification.
        other => other,
    };
    (classified, detail)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::{EXIT_GENERAL_ERROR, EXIT_GPU_ERROR, EXIT_IO_ERROR};

    /// Regression: classifying a failure for its exit code used to throw
    /// away the `anyhow` context above the innermost `std::io::Error`, so
    /// `oxigaf render -m /nonexistent/model.ply` reported "I/O error: No
    /// such file or directory (os error 2)" and never named the file or the
    /// operation.
    #[test]
    fn keeps_the_context_an_io_downcast_would_drop() {
        let io = std::io::Error::new(std::io::ErrorKind::NotFound, "No such file or directory");
        let err = anyhow::Error::new(io).context("Failed to load model: /nonexistent/model.ply");

        let (cli_err, detail) = classify_error(err);

        assert!(
            detail.contains("Failed to load model: /nonexistent/model.ply"),
            "rendered chain lost the context: {detail}"
        );
        assert_eq!(
            cli_err.exit_code(),
            EXIT_IO_ERROR,
            "an io::Error must still be classified as an I/O failure"
        );
        let rendered = cli_err.to_string();
        assert!(
            rendered.contains("Failed to load model: /nonexistent/model.ply"),
            "the classified error still hides the path: {rendered}"
        );
    }

    /// A chain several layers deep must survive whole: `{:#}` renders every
    /// context, where `Display` plus one `source` line would show two.
    #[test]
    fn keeps_every_layer_of_a_deep_chain() {
        let io = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "Permission denied");
        let err = anyhow::Error::new(io)
            .context("Failed to open frames/000.png")
            .context("Invalid --input frames/");

        let (_, detail) = classify_error(err);

        for expected in [
            "Invalid --input frames/",
            "Failed to open frames/000.png",
            "Permission denied",
        ] {
            assert!(
                detail.contains(expected),
                "{expected:?} missing from {detail}"
            );
        }
    }

    /// Classification must not turn a non-I/O failure into one: the exit
    /// code taxonomy is what a CI pipeline branches on.
    #[test]
    fn preserves_other_failure_classes() {
        let (cli_err, detail) = classify_error(
            CliError::GpuNotAvailable {
                backend: "any".to_string(),
                fallback: None,
            }
            .into(),
        );
        assert_eq!(
            cli_err.exit_code(),
            EXIT_GPU_ERROR,
            "a GPU failure was reclassified"
        );
        assert!(detail.contains("GPU not available"), "detail was: {detail}");

        let (plain, detail) = classify_error(anyhow::anyhow!("something broke"));
        assert_eq!(plain.exit_code(), EXIT_GENERAL_ERROR);
        assert_eq!(detail, "something broke");
    }
}
