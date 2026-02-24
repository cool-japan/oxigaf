//! Verbosity level configuration for CLI output.
//!
//! Controls logging, progress bars, and timing information based on
//! command-line flags (-v, -vv, -vvv, -q).

use tracing::Level;

/// Verbosity levels for CLI output.
///
/// Determines logging detail, progress bar visibility, and timing information.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Verbosity {
    /// Only errors (-q).
    ///
    /// Suppresses all output except errors. No progress bars or informational messages.
    Quiet,

    /// Progress + results (default).
    ///
    /// Shows progress bars, informational messages, and results.
    Normal,

    /// Debug info (-v).
    ///
    /// Includes timing information and debug-level logging.
    Verbose,

    /// Trace-level logging (-vv).
    ///
    /// Enables trace-level logging with file and line information.
    Debug,

    /// All internal details (-vvv).
    ///
    /// Maximum verbosity with all internal details and trace logging.
    Trace,
}

impl Verbosity {
    /// Create verbosity level from command-line flags.
    ///
    /// # Arguments
    ///
    /// * `verbose` - Number of `-v` flags (0 = normal, 1 = verbose, 2 = debug, 3+ = trace)
    /// * `quiet` - Whether `-q` flag was specified (overrides verbose)
    ///
    /// # Examples
    ///
    /// ```
    /// use oxigaf_cli::verbosity::Verbosity;
    ///
    /// let quiet = Verbosity::from_flags(0, true);
    /// assert_eq!(quiet, Verbosity::Quiet);
    ///
    /// let normal = Verbosity::from_flags(0, false);
    /// assert_eq!(normal, Verbosity::Normal);
    ///
    /// let verbose = Verbosity::from_flags(1, false);
    /// assert_eq!(verbose, Verbosity::Verbose);
    ///
    /// let trace = Verbosity::from_flags(3, false);
    /// assert_eq!(trace, Verbosity::Trace);
    /// ```
    #[must_use]
    pub fn from_flags(verbose: u8, quiet: bool) -> Self {
        if quiet {
            Self::Quiet
        } else {
            match verbose {
                0 => Self::Normal,
                1 => Self::Verbose,
                2 => Self::Debug,
                _ => Self::Trace,
            }
        }
    }

    /// Get the tracing level for this verbosity.
    ///
    /// Maps verbosity to appropriate `tracing::Level`:
    /// - Quiet → ERROR
    /// - Normal → INFO
    /// - Verbose → DEBUG
    /// - Debug/Trace → TRACE
    #[must_use]
    pub fn tracing_level(&self) -> Level {
        match self {
            Self::Quiet => Level::ERROR,
            Self::Normal => Level::INFO,
            Self::Verbose => Level::DEBUG,
            Self::Debug | Self::Trace => Level::TRACE,
        }
    }

    /// Whether progress bars should be shown.
    ///
    /// Progress bars are shown in Normal and Verbose modes, but hidden
    /// in Quiet, Debug, and Trace modes (where detailed logging is preferred).
    #[must_use]
    pub fn show_progress(&self) -> bool {
        matches!(self, Self::Normal | Self::Verbose)
    }

    /// Whether detailed timing information should be shown.
    ///
    /// Timing information is shown in Verbose, Debug, and Trace modes.
    #[must_use]
    pub fn show_timing(&self) -> bool {
        *self >= Self::Verbose
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_flags_quiet() {
        let v = Verbosity::from_flags(0, true);
        assert_eq!(v, Verbosity::Quiet);

        // Quiet overrides verbose flag
        let v = Verbosity::from_flags(3, true);
        assert_eq!(v, Verbosity::Quiet);
    }

    #[test]
    fn test_from_flags_normal() {
        let v = Verbosity::from_flags(0, false);
        assert_eq!(v, Verbosity::Normal);
    }

    #[test]
    fn test_from_flags_verbose_levels() {
        let v1 = Verbosity::from_flags(1, false);
        assert_eq!(v1, Verbosity::Verbose);

        let v2 = Verbosity::from_flags(2, false);
        assert_eq!(v2, Verbosity::Debug);

        let v3 = Verbosity::from_flags(3, false);
        assert_eq!(v3, Verbosity::Trace);

        let v4 = Verbosity::from_flags(10, false);
        assert_eq!(v4, Verbosity::Trace);
    }

    #[test]
    fn test_tracing_level() {
        assert_eq!(Verbosity::Quiet.tracing_level(), Level::ERROR);
        assert_eq!(Verbosity::Normal.tracing_level(), Level::INFO);
        assert_eq!(Verbosity::Verbose.tracing_level(), Level::DEBUG);
        assert_eq!(Verbosity::Debug.tracing_level(), Level::TRACE);
        assert_eq!(Verbosity::Trace.tracing_level(), Level::TRACE);
    }

    #[test]
    fn test_show_progress() {
        assert!(!Verbosity::Quiet.show_progress());
        assert!(Verbosity::Normal.show_progress());
        assert!(Verbosity::Verbose.show_progress());
        assert!(!Verbosity::Debug.show_progress());
        assert!(!Verbosity::Trace.show_progress());
    }

    #[test]
    fn test_show_timing() {
        assert!(!Verbosity::Quiet.show_timing());
        assert!(!Verbosity::Normal.show_timing());
        assert!(Verbosity::Verbose.show_timing());
        assert!(Verbosity::Debug.show_timing());
        assert!(Verbosity::Trace.show_timing());
    }

    #[test]
    fn test_ordering() {
        assert!(Verbosity::Quiet < Verbosity::Normal);
        assert!(Verbosity::Normal < Verbosity::Verbose);
        assert!(Verbosity::Verbose < Verbosity::Debug);
        assert!(Verbosity::Debug < Verbosity::Trace);
    }
}
