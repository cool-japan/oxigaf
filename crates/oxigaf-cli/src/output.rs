//! Color-coded terminal output utilities.
//!
//! Provides helper functions for consistent, user-friendly CLI output with:
//! - Color-coded status messages (success, error, warning, info)
//! - Respect for `NO_COLOR` environment variable
//! - Automatic TTY detection

use std::io::{IsTerminal, Write};
use std::sync::OnceLock;

use owo_colors::OwoColorize;

// ---------------------------------------------------------------------------
// Color Support Detection
// ---------------------------------------------------------------------------

/// Whether the `NO_COLOR`/`TERM=dumb` environment convention forces color
/// off, independent of which output stream is being written to.
///
/// Takes its inputs explicitly rather than reading `std::env` itself, so the
/// NO_COLOR/TERM decision logic is unit-testable deterministically without
/// mutating process-global environment state from a test.
fn colors_forced_off(no_color_env_set: bool, term_env: Option<&str>) -> bool {
    // Respect NO_COLOR convention (https://no-color.org/)
    no_color_env_set || term_env == Some("dumb")
}

fn env_forces_colors_off() -> bool {
    colors_forced_off(
        std::env::var("NO_COLOR").is_ok(),
        std::env::var("TERM").ok().as_deref(),
    )
}

/// Check if color output is enabled for messages written to **stdout**
/// (`success`, `info`, `hint`, `value`, `path_value`, `header`, `separator`).
///
/// Colors are disabled if:
/// - `NO_COLOR` environment variable is set (any value)
/// - stdout is not a terminal (pipe, redirect)
/// - `TERM=dumb` is set
///
/// The result is cached after the first call: `NO_COLOR`/`TERM` and stdout's
/// terminal-ness are not expected to change during a single process's
/// lifetime, so re-checking on every printed message (an `env::var` lookup
/// plus an `is_terminal` syscall) was pure overhead.
#[must_use]
pub fn colors_enabled_stdout() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| !env_forces_colors_off() && std::io::stdout().is_terminal())
}

/// Check if color output is enabled for messages written to **stderr**
/// (`error`, `warning`, and [`display_error`]'s "Caused by" line).
///
/// Same rules as [`colors_enabled_stdout`], but checks stderr's
/// terminal-ness instead of stdout's. Previously every message -- including
/// these stderr ones -- was gated on `colors_enabled()`'s stdout check, so
/// `oxigaf ... > out.txt` on a TTY (stdout redirected, stderr still a TTY)
/// lost color on stderr diagnostics that should have kept it, and
/// `oxigaf ... 2> err.txt` from a TTY (stderr redirected, stdout still a
/// TTY) wrote raw ANSI escapes into the redirected error file.
#[must_use]
pub fn colors_enabled_stderr() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| !env_forces_colors_off() && std::io::stderr().is_terminal())
}

/// Check if color output is enabled for stdout.
///
/// Alias for [`colors_enabled_stdout`], kept for existing callers (e.g.
/// `summary.rs`'s report printers) that only ever print to stdout.
#[must_use]
pub fn colors_enabled() -> bool {
    colors_enabled_stdout()
}

// ---------------------------------------------------------------------------
// Output Functions
// ---------------------------------------------------------------------------

/// Print a success message with a green checkmark.
pub fn success(message: &str) {
    if colors_enabled() {
        println!("{} {}", "[OK]".green().bold(), message);
    } else {
        println!("[OK] {}", message);
    }
}

/// Print an error message with a red X.
pub fn error(message: &str) {
    if colors_enabled_stderr() {
        eprintln!("{} {}", "[ERROR]".red().bold(), message.red());
    } else {
        eprintln!("[ERROR] {}", message);
    }
}

/// Print a warning message with a yellow exclamation.
pub fn warning(message: &str) {
    if colors_enabled_stderr() {
        eprintln!("{} {}", "[WARN]".yellow().bold(), message.yellow());
    } else {
        eprintln!("[WARN] {}", message);
    }
}

/// Print an info message with a blue indicator.
pub fn info(message: &str) {
    if colors_enabled() {
        println!("{} {}", "[INFO]".blue().bold(), message);
    } else {
        println!("[INFO] {}", message);
    }
}

/// Print a hint/suggestion message with cyan color.
pub fn hint(message: &str) {
    if colors_enabled() {
        println!("{} {}", "[HINT]".cyan().bold(), message.cyan());
    } else {
        println!("[HINT] {}", message);
    }
}

/// Print a value with its label (label in dim, value in bold).
pub fn value(label: &str, val: &str) {
    if colors_enabled() {
        println!("  {} {}", format!("{}:", label).dimmed(), val.bold());
    } else {
        println!("  {}: {}", label, val);
    }
}

/// Print a path with special formatting (underlined blue).
pub fn path_value(label: &str, path: &std::path::Path) {
    if colors_enabled() {
        println!(
            "  {} {}",
            format!("{}:", label).dimmed(),
            path.display().to_string().blue().underline()
        );
    } else {
        println!("  {}: {}", label, path.display());
    }
}

/// Print a section header.
pub fn header(text: &str) {
    if colors_enabled() {
        println!();
        println!("{}", text.bold().underline());
    } else {
        println!();
        println!("{}", text);
    }
}

/// Print a horizontal separator line.
pub fn separator() {
    if colors_enabled() {
        println!("{}", "─".repeat(60).dimmed());
    } else {
        println!("{}", "─".repeat(60));
    }
}

// ---------------------------------------------------------------------------
// Error Display with Suggestions
// ---------------------------------------------------------------------------

use crate::error::CliError;

/// Display a CLI error with formatting and suggestions.
pub fn display_error(err: &CliError) {
    error(&err.to_string());

    // Print the error chain if available
    if let Some(source) = std::error::Error::source(err) {
        if colors_enabled_stderr() {
            eprintln!("  {} {}", "Caused by:".dimmed(), source);
        } else {
            eprintln!("  Caused by: {}", source);
        }
    }

    // Print suggestion if available
    if let Some(suggestion) = err.suggestion() {
        println!();
        hint(suggestion);
    }
}

/// Flush stdout to ensure all output is visible.
pub fn flush() {
    let _ = std::io::stdout().flush();
    let _ = std::io::stderr().flush();
}

#[cfg(test)]
mod tests {
    use super::*;

    // `colors_enabled_stdout`/`colors_enabled_stderr` themselves are not
    // meaningfully unit-testable: they cache a `OnceLock` for the lifetime
    // of the test process (multiple tests would observe each other's first
    // result) and `is_terminal()` is essentially always `false` under
    // `cargo test` (stdout/stderr are captured), so a test could only ever
    // observe "false" regardless of whether the stdout/stderr split is
    // correct. `colors_forced_off` is the actual NO_COLOR/TERM decision
    // logic, factored out to take its inputs explicitly so it can be tested
    // deterministically without touching process-global environment state.

    #[test]
    fn test_colors_forced_off_when_no_color_set() {
        assert!(colors_forced_off(true, None));
        assert!(colors_forced_off(true, Some("xterm-256color")));
    }

    #[test]
    fn test_colors_forced_off_when_term_dumb() {
        assert!(colors_forced_off(false, Some("dumb")));
    }

    #[test]
    fn test_colors_not_forced_off_for_normal_terminal() {
        assert!(!colors_forced_off(false, Some("xterm-256color")));
        assert!(!colors_forced_off(false, None));
    }

    #[test]
    fn test_stdout_and_stderr_variants_are_stable_across_calls() {
        // Regression: `colors_enabled_stdout`/`colors_enabled_stderr` used
        // to be a single `colors_enabled()` that only ever checked stdout,
        // so stderr diagnostics (`error`/`warning`) were gated on stdout's
        // terminal-ness. This asserts both are separate, independently
        // callable entry points whose (now-cached) result is stable across
        // repeated calls within the same process -- true regardless of the
        // actual TTY/NO_COLOR state under whatever harness runs this test,
        // so it stays deterministic without asserting a specific value.
        let stdout_first = colors_enabled_stdout();
        let stdout_second = colors_enabled_stdout();
        assert_eq!(
            stdout_first, stdout_second,
            "stdout result must be stable (cached)"
        );

        let stderr_first = colors_enabled_stderr();
        let stderr_second = colors_enabled_stderr();
        assert_eq!(
            stderr_first, stderr_second,
            "stderr result must be stable (cached)"
        );
    }
}
