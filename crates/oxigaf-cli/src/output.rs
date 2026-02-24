//! Color-coded terminal output utilities.
//!
//! Provides helper functions for consistent, user-friendly CLI output with:
//! - Color-coded status messages (success, error, warning, info)
//! - Respect for `NO_COLOR` environment variable
//! - Automatic TTY detection

use std::io::{IsTerminal, Write};

use owo_colors::OwoColorize;

// ---------------------------------------------------------------------------
// Color Support Detection
// ---------------------------------------------------------------------------

/// Check if color output is enabled.
///
/// Colors are disabled if:
/// - `NO_COLOR` environment variable is set (any value)
/// - stdout is not a terminal (pipe, redirect)
/// - `TERM=dumb` is set
#[must_use]
pub fn colors_enabled() -> bool {
    // Respect NO_COLOR convention (https://no-color.org/)
    if std::env::var("NO_COLOR").is_ok() {
        return false;
    }

    // Check if stdout is a terminal
    if !std::io::stdout().is_terminal() {
        return false;
    }

    // Check for dumb terminal
    if let Ok(term) = std::env::var("TERM") {
        if term == "dumb" {
            return false;
        }
    }

    true
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
    if colors_enabled() {
        eprintln!("{} {}", "[ERROR]".red().bold(), message.red());
    } else {
        eprintln!("[ERROR] {}", message);
    }
}

/// Print a warning message with a yellow exclamation.
pub fn warning(message: &str) {
    if colors_enabled() {
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
        if colors_enabled() {
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
