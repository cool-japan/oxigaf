//! Terminal training dashboard using raw ANSI escape codes.
//!
//! Provides a rich terminal UI for monitoring 3D Gaussian Splatting training
//! without external terminal libraries. Uses Unicode block characters for
//! spark lines and box-drawing characters for panel borders.
//!
//! # Example
//!
//! ```
//! use oxigaf_cli::dashboard::{DashboardConfig, DashboardRenderer, DashboardState};
//!
//! let config = DashboardConfig::default();
//! let mut renderer = DashboardRenderer::new(config);
//! let mut state = DashboardState::new(30_000);
//! state.update_step(1000, 28.3, 0.012, 0.009, 0.003, 60_000);
//! state.iter_per_sec = 12.5;
//! state.elapsed_secs = 80.0;
//!
//! let frame = renderer.render_frame_no_cursor(&state);
//! assert!(!frame.is_empty());
//! ```

// ---------------------------------------------------------------------------
// ANSI utilities
// ---------------------------------------------------------------------------

/// ANSI escape code utilities for terminal control.
pub mod ansi {
    /// Reset all terminal attributes.
    pub const RESET: &str = "\x1b[0m";
    /// Bold text.
    pub const BOLD: &str = "\x1b[1m";
    /// Dim text.
    pub const DIM: &str = "\x1b[2m";
    /// Clear current line.
    pub const CLEAR_LINE: &str = "\x1b[2K";
    /// Move cursor to top-left of screen (home position).
    pub const CURSOR_HOME: &str = "\x1b[H";
    /// Hide cursor.
    pub const HIDE_CURSOR: &str = "\x1b[?25l";
    /// Show cursor.
    pub const SHOW_CURSOR: &str = "\x1b[?25h";
    /// Clear entire screen.
    pub const CLEAR_SCREEN: &str = "\x1b[2J";

    // Color codes
    /// Foreground red.
    pub const FG_RED: &str = "\x1b[31m";
    /// Foreground green.
    pub const FG_GREEN: &str = "\x1b[32m";
    /// Foreground yellow.
    pub const FG_YELLOW: &str = "\x1b[33m";
    /// Foreground blue.
    pub const FG_BLUE: &str = "\x1b[34m";
    /// Foreground cyan.
    pub const FG_CYAN: &str = "\x1b[36m";
    /// Foreground white.
    pub const FG_WHITE: &str = "\x1b[37m";
    /// Foreground bright white.
    pub const FG_BRIGHT_WHITE: &str = "\x1b[97m";

    /// Move cursor up `n` lines.
    #[must_use]
    pub fn cursor_up(n: usize) -> String {
        format!("\x1b[{}A", n)
    }

    /// Move cursor to column `col` (1-indexed).
    #[must_use]
    pub fn cursor_col(col: usize) -> String {
        format!("\x1b[{}G", col)
    }

    /// Set foreground color using 24-bit RGB (true color).
    #[must_use]
    pub fn fg_rgb(r: u8, g: u8, b: u8) -> String {
        format!("\x1b[38;2;{};{};{}m", r, g, b)
    }
}

// ---------------------------------------------------------------------------
// Spark line
// ---------------------------------------------------------------------------

/// Unicode block characters for spark lines (8 levels, low to high).
const BLOCKS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

/// The middle block character used when all values are equal.
const BLOCK_MID: char = '▄';

/// Generate a Unicode spark line from a series of values.
///
/// Uses block characters `▁▂▃▄▅▆▇█` (8 levels) to represent value trends.
///
/// # Parameters
///
/// - `values`: Series of `f32` values to visualize.
/// - `width`: Number of characters in the output string.
///
/// # Returns
///
/// A `String` of `width` Unicode block characters representing the trend.
/// Returns spaces if `values` is empty or `width` is zero.
#[must_use]
pub fn format_spark_line(values: &[f32], width: usize) -> String {
    if width == 0 {
        return String::new();
    }
    if values.is_empty() {
        return " ".repeat(width);
    }

    // Sample values to exactly `width` points using evenly spaced indices.
    let sampled: Vec<f32> = if width == 1 {
        vec![values[values.len() / 2]]
    } else {
        (0..width)
            .map(|i| {
                let idx = if values.len() == 1 {
                    0
                } else {
                    // Map [0, width-1] -> [0, values.len()-1]
                    (i * (values.len() - 1) + (width - 1) / 2) / (width - 1)
                };
                values[idx.min(values.len() - 1)]
            })
            .collect()
    };

    let min_val = sampled.iter().copied().fold(f32::INFINITY, f32::min);
    let max_val = sampled.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    // If all values are equal, use the mid block character.
    if (max_val - min_val).abs() < f32::EPSILON {
        return sampled.iter().map(|_| BLOCK_MID).collect();
    }

    let range = max_val - min_val;
    sampled
        .iter()
        .map(|&v| {
            let normalized = (v - min_val) / range;
            // Map [0.0, 1.0] -> [0, 7] with clamping
            let idx = ((normalized * 7.0).round() as usize).min(7);
            BLOCKS[idx]
        })
        .collect()
}

// ---------------------------------------------------------------------------
// MetricBar
// ---------------------------------------------------------------------------

/// Visual progress bar with label, value, and color theme.
///
/// Renders as a single line:
/// ```text
/// PSNR    [████████░░░░░░░] 28.3 dB
/// ```
#[derive(Debug, Clone)]
pub struct MetricBar {
    /// Display label for this metric.
    pub label: String,
    /// Current value of the metric.
    pub value: f32,
    /// Minimum expected value (left end of bar).
    pub min: f32,
    /// Maximum expected value (right end of bar).
    pub max: f32,
    /// Number of characters in the bar fill region.
    pub bar_width: usize,
    /// Unit string displayed after the value (e.g., "dB").
    pub unit: String,
    /// Color theme: "green", "yellow", "red", "blue", "cyan".
    pub color: String,
}

impl MetricBar {
    /// Create a new metric bar with default settings.
    #[must_use]
    pub fn new(label: impl Into<String>, value: f32, min: f32, max: f32) -> Self {
        Self {
            label: label.into(),
            value,
            min,
            max,
            bar_width: 20,
            unit: String::new(),
            color: "green".to_string(),
        }
    }

    /// Set the unit string (e.g., "dB").
    #[must_use]
    pub fn with_unit(mut self, unit: impl Into<String>) -> Self {
        self.unit = unit.into();
        self
    }

    /// Set the color theme.
    ///
    /// Valid values: "green", "yellow", "red", "blue", "cyan".
    /// Defaults to white for unknown color names.
    #[must_use]
    pub fn with_color(mut self, color: impl Into<String>) -> Self {
        self.color = color.into();
        self
    }

    /// Set the bar fill width (number of characters).
    #[must_use]
    pub fn with_width(mut self, width: usize) -> Self {
        self.bar_width = width;
        self
    }

    /// Compute the fill fraction `(value - min) / (max - min)`, clamped to `[0.0, 1.0]`.
    #[must_use]
    pub fn fraction(&self) -> f32 {
        if (self.max - self.min).abs() < f32::EPSILON {
            return 0.0;
        }
        ((self.value - self.min) / (self.max - self.min)).clamp(0.0, 1.0)
    }

    /// Resolve the ANSI color escape for the configured color name.
    fn color_code(&self) -> &'static str {
        match self.color.as_str() {
            "green" => ansi::FG_GREEN,
            "yellow" => ansi::FG_YELLOW,
            "red" => ansi::FG_RED,
            "blue" => ansi::FG_BLUE,
            "cyan" => ansi::FG_CYAN,
            _ => ansi::FG_WHITE,
        }
    }

    /// Format the bar as a single colored line.
    ///
    /// Output: `"PSNR    [████████░░░░░░░] 28.3 dB"`
    #[must_use]
    pub fn render(&self) -> String {
        let frac = self.fraction();
        let filled = (frac * self.bar_width as f32).round() as usize;
        let empty = self.bar_width.saturating_sub(filled);

        let color = self.color_code();
        let filled_str: String = std::iter::repeat_n('█', filled).collect();
        let empty_str: String = std::iter::repeat_n('░', empty).collect();

        let label_padded = format!("{:<8}", self.label);
        let value_str = if self.unit.is_empty() {
            format!("{:.4}", self.value)
        } else {
            format!("{:.1} {}", self.value, self.unit)
        };

        format!(
            "{}{}{} [{}{}{}{}{}] {}",
            ansi::BOLD,
            label_padded,
            ansi::RESET,
            color,
            filled_str,
            ansi::RESET,
            ansi::DIM,
            empty_str,
            ansi::RESET,
            // Note: value_str appended after RESET
        ) + &value_str
    }
}

// ---------------------------------------------------------------------------
// DashboardPanel
// ---------------------------------------------------------------------------

/// A bordered panel containing multiple metric bars.
///
/// Renders with box-drawing characters:
/// ```text
/// ╔══════════════════════════════╗
/// ║  OxiGAF Training Dashboard   ║
/// ╠══════════════════════════════╣
/// ║ PSNR    [████████░░░] 28.3dB ║
/// ║ Loss    [██░░░░░░░░░] 0.012  ║
/// ╚══════════════════════════════╝
/// ```
pub struct DashboardPanel {
    /// Panel title displayed in the header.
    pub title: String,
    /// Metric bars to display.
    pub bars: Vec<MetricBar>,
    /// Footer text displayed at the bottom.
    pub footer: String,
}

impl DashboardPanel {
    /// Create a new empty panel with the given title.
    #[must_use]
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            bars: Vec::new(),
            footer: String::new(),
        }
    }

    /// Add a metric bar to the panel (builder-style).
    #[must_use]
    pub fn add_bar(mut self, bar: MetricBar) -> Self {
        self.bars.push(bar);
        self
    }

    /// Set the footer text (builder-style).
    #[must_use]
    pub fn with_footer(mut self, footer: impl Into<String>) -> Self {
        self.footer = footer.into();
        self
    }

    /// Render the full panel as a multi-line string.
    ///
    /// `width` is the total display width including borders.
    #[must_use]
    pub fn render(&self, width: usize) -> String {
        // Inner content width: total width minus 2 border chars, 2 space padding each side.
        // ║ <content> ║  → inner = width - 4
        let inner_width = width.saturating_sub(4);
        let horizontal: String = std::iter::repeat_n('═', width.saturating_sub(2)).collect();

        let mut lines: Vec<String> = Vec::new();

        // Top border
        lines.push(format!("╔{}╗", horizontal));

        // Title row: center within inner_width
        let title_display = if self.title.chars().count() > inner_width {
            self.title.chars().take(inner_width).collect::<String>()
        } else {
            self.title.clone()
        };
        let title_len = title_display.chars().count();
        let title_padding_total = inner_width.saturating_sub(title_len);
        let title_left = title_padding_total / 2;
        let title_right = title_padding_total - title_left;
        lines.push(format!(
            "║ {}{}{} ║",
            " ".repeat(title_left),
            format!("{}{}{}", ansi::BOLD, ansi::FG_BRIGHT_WHITE, title_display,) + ansi::RESET,
            " ".repeat(title_right),
        ));

        // Separator
        lines.push(format!("╠{}╣", horizontal));

        // Metric bars
        for bar in &self.bars {
            let rendered = bar.render();
            // Truncate or pad to inner_width display columns.
            // Strip ANSI for length measurement then pad/truncate.
            let stripped_len = strip_ansi_len(&rendered);
            let line_content = if stripped_len <= inner_width {
                let padding = inner_width.saturating_sub(stripped_len);
                format!("{}{}", rendered, " ".repeat(padding))
            } else {
                // Truncate is complex with ANSI -- just use rendered as-is
                rendered
            };
            lines.push(format!("║ {} ║", line_content));
        }

        // Footer (if non-empty)
        if !self.footer.is_empty() {
            lines.push(format!("╠{}╣", horizontal));
            let footer_display = if self.footer.chars().count() > inner_width {
                self.footer.chars().take(inner_width).collect::<String>()
            } else {
                self.footer.clone()
            };
            let footer_len = footer_display.chars().count();
            let footer_pad = inner_width.saturating_sub(footer_len);
            lines.push(
                format!("║ {}{}{} ║", ansi::DIM, footer_display, ansi::RESET,)
                    + &" ".repeat(footer_pad),
            );
        }

        // Bottom border
        lines.push(format!("╚{}╝", horizontal));

        lines.join("\n")
    }
}

/// Count display columns in a string, ignoring ANSI escape sequences.
///
/// Each non-ANSI character is counted as 1 display column (ASCII assumption).
fn strip_ansi_len(s: &str) -> usize {
    let mut count = 0usize;
    let mut in_escape = false;
    for ch in s.chars() {
        if in_escape {
            if ch == 'm'
                || ch == 'A'
                || ch == 'G'
                || ch == 'H'
                || ch == 'J'
                || ch == 'l'
                || ch == 'h'
            {
                in_escape = false;
            }
        } else if ch == '\x1b' {
            in_escape = true;
        } else {
            count += 1;
        }
    }
    count
}

// ---------------------------------------------------------------------------
// DashboardState
// ---------------------------------------------------------------------------

/// Snapshot of the training run state for dashboard rendering.
#[derive(Debug, Clone)]
pub struct DashboardState {
    /// Current training step (0-indexed, or 1-indexed as preferred).
    pub step: usize,
    /// Total number of training steps.
    pub total_steps: usize,
    /// Peak signal-to-noise ratio in dB.
    pub psnr: f32,
    /// Combined total training loss.
    pub total_loss: f32,
    /// Photometric (L1/L2) component of the loss.
    pub photometric_loss: f32,
    /// Perceptual (LPIPS/SSIM) component of the loss.
    pub perceptual_loss: f32,
    /// Current number of Gaussian primitives in the scene.
    pub num_gaussians: usize,
    /// Training speed in iterations per second.
    pub iter_per_sec: f32,
    /// Wall-clock elapsed training time in seconds.
    pub elapsed_secs: f64,
    /// PSNR history ring-buffer (last ≤ 50 values) for spark line.
    pub psnr_history: Vec<f32>,
    /// Loss history ring-buffer (last ≤ 50 values) for spark line.
    pub loss_history: Vec<f32>,
}

impl DashboardState {
    /// Create a new state for a run with `total_steps` iterations.
    #[must_use]
    pub fn new(total_steps: usize) -> Self {
        Self {
            step: 0,
            total_steps,
            psnr: 0.0,
            total_loss: 0.0,
            photometric_loss: 0.0,
            perceptual_loss: 0.0,
            num_gaussians: 0,
            iter_per_sec: 0.0,
            elapsed_secs: 0.0,
            psnr_history: Vec::new(),
            loss_history: Vec::new(),
        }
    }

    /// Update the dashboard state with results from a completed training step.
    ///
    /// Histories are capped at 50 entries each (oldest removed first).
    pub fn update_step(
        &mut self,
        step: usize,
        psnr: f32,
        total_loss: f32,
        photometric: f32,
        perceptual: f32,
        num_gaussians: usize,
    ) {
        self.step = step;
        self.psnr = psnr;
        self.total_loss = total_loss;
        self.photometric_loss = photometric;
        self.perceptual_loss = perceptual;
        self.num_gaussians = num_gaussians;

        if self.psnr_history.len() >= 50 {
            self.psnr_history.remove(0);
        }
        self.psnr_history.push(psnr);

        if self.loss_history.len() >= 50 {
            self.loss_history.remove(0);
        }
        self.loss_history.push(total_loss);
    }

    /// Fraction of training completed: `step / total_steps`, clamped to `[0.0, 1.0]`.
    #[must_use]
    pub fn progress_fraction(&self) -> f32 {
        if self.total_steps == 0 {
            return 0.0;
        }
        (self.step as f32 / self.total_steps as f32).min(1.0)
    }

    /// Estimated time remaining in seconds, based on current `iter_per_sec`.
    ///
    /// Returns `0.0` if speed is effectively zero.
    #[must_use]
    pub fn eta_secs(&self) -> f64 {
        if self.iter_per_sec < 1e-6 {
            return 0.0;
        }
        let remaining = self.total_steps.saturating_sub(self.step) as f64;
        remaining / self.iter_per_sec as f64
    }
}

// ---------------------------------------------------------------------------
// DashboardConfig
// ---------------------------------------------------------------------------

/// Configuration for the dashboard renderer.
#[derive(Debug, Clone)]
pub struct DashboardConfig {
    /// Total panel width in terminal columns (default: 80).
    pub panel_width: usize,
    /// Width of the fill area inside metric bars (default: 25).
    pub bar_width: usize,
    /// Width of spark line characters (default: 20).
    pub spark_width: usize,
    /// Number of lines emitted per `render_frame_no_cursor` call.
    ///
    /// Used to calculate the cursor-up escape on subsequent frames.
    pub num_lines: usize,
}

impl Default for DashboardConfig {
    fn default() -> Self {
        Self {
            panel_width: 80,
            bar_width: 25,
            spark_width: 20,
            // Lines breakdown:
            //  1 - blank separator after header
            //  1 - header line (OxiGAF Training  step X/Y)
            //  1 - overall progress bar
            //  1 - blank
            //  1 - PSNR bar line
            //  1 - PSNR spark
            //  1 - Loss bar line
            //  1 - Loss spark
            //  1 - Gaussians bar
            //  1 - Speed / ETA line
            //  1 - blank footer
            // = 11 total
            num_lines: 11,
        }
    }
}

// ---------------------------------------------------------------------------
// DashboardRenderer
// ---------------------------------------------------------------------------

/// Stateful renderer that tracks how many frames have been drawn.
///
/// On subsequent calls to `render_frame` the output is prefixed with
/// enough cursor-up escapes to overwrite the previous frame in-place.
pub struct DashboardRenderer {
    /// Rendering configuration.
    pub config: DashboardConfig,
    /// Number of times `render_frame` has been called.
    pub num_renders: usize,
}

impl DashboardRenderer {
    /// Create a new renderer with the given configuration.
    #[must_use]
    pub fn new(config: DashboardConfig) -> Self {
        Self {
            config,
            num_renders: 0,
        }
    }

    /// Render a complete dashboard frame with cursor control.
    ///
    /// On the first call, renders the full frame.
    /// On subsequent calls, prepends cursor-up to overwrite the previous frame.
    pub fn render_frame(&mut self, state: &DashboardState) -> String {
        let content = self.render_frame_no_cursor(state);
        let result = if self.num_renders == 0 {
            content
        } else {
            format!("{}{}", ansi::cursor_up(self.config.num_lines), content)
        };
        self.num_renders += 1;
        result
    }

    /// Render the dashboard without cursor control codes.
    ///
    /// Suitable for testing and for piped output.
    ///
    /// # Lines produced (matching `num_lines`):
    ///
    /// 1. Header: `"OxiGAF Training  step 12345/30000"`
    /// 2. Overall progress bar
    /// 3. Blank
    /// 4. PSNR bar (metric bar line)
    /// 5. PSNR spark line
    /// 6. Loss bar (metric bar line)
    /// 7. Loss spark line
    /// 8. Gaussians bar
    /// 9. Speed and ETA line
    /// 10. Blank
    /// 11. (blank/trailing newline represented by final join)
    #[must_use]
    pub fn render_frame_no_cursor(&self, state: &DashboardState) -> String {
        let width = self.config.panel_width;
        let bar_w = self.config.bar_width;
        let spark_w = self.config.spark_width;

        let mut lines: Vec<String> = Vec::with_capacity(self.config.num_lines + 1);

        // ------------------------------------------------------------------
        // Line 1: Header
        // ------------------------------------------------------------------
        let header = format!(
            "{}{}OxiGAF Training{}  step {}/{}",
            ansi::BOLD,
            ansi::FG_BRIGHT_WHITE,
            ansi::RESET,
            state.step,
            state.total_steps,
        );
        lines.push(header);

        // ------------------------------------------------------------------
        // Line 2: Overall progress bar
        // ------------------------------------------------------------------
        let frac = state.progress_fraction();
        let prog_filled = (frac * bar_w as f32).round() as usize;
        let prog_empty = bar_w.saturating_sub(prog_filled);
        let filled_str: String = std::iter::repeat_n('█', prog_filled).collect();
        let empty_str: String = std::iter::repeat_n('░', prog_empty).collect();
        let pct = frac * 100.0;
        let progress_line = format!(
            "{}Progress{} [{}{}{}{}{}] {}{:.1}%{}",
            ansi::BOLD,
            ansi::RESET,
            ansi::FG_CYAN,
            filled_str,
            ansi::RESET,
            ansi::DIM,
            empty_str,
            ansi::RESET,
            pct,
            ansi::RESET,
        );
        lines.push(progress_line);

        // ------------------------------------------------------------------
        // Line 3: Blank separator
        // ------------------------------------------------------------------
        lines.push(String::new());

        // ------------------------------------------------------------------
        // Line 4: PSNR bar
        // ------------------------------------------------------------------
        let psnr_bar = MetricBar::new("PSNR", state.psnr, 0.0, 45.0)
            .with_unit("dB")
            .with_color("green")
            .with_width(bar_w);
        lines.push(psnr_bar.render());

        // ------------------------------------------------------------------
        // Line 5: PSNR spark line
        // ------------------------------------------------------------------
        let psnr_spark = format_spark_line(&state.psnr_history, spark_w);
        lines.push(format!(
            "  {}{}PSNR trend:{} {}{}{}",
            ansi::DIM,
            ansi::FG_GREEN,
            ansi::RESET,
            ansi::FG_GREEN,
            psnr_spark,
            ansi::RESET,
        ));

        // ------------------------------------------------------------------
        // Line 6: Loss bar
        // ------------------------------------------------------------------
        let loss_bar = MetricBar::new("Loss", state.total_loss, 0.0, 0.1)
            .with_color("yellow")
            .with_width(bar_w);
        lines.push(loss_bar.render());

        // ------------------------------------------------------------------
        // Line 7: Loss spark line
        // ------------------------------------------------------------------
        let loss_spark = format_spark_line(&state.loss_history, spark_w);
        lines.push(format!(
            "  {}{}Loss trend:{} {}{}{}",
            ansi::DIM,
            ansi::FG_YELLOW,
            ansi::RESET,
            ansi::FG_YELLOW,
            loss_spark,
            ansi::RESET,
        ));

        // ------------------------------------------------------------------
        // Line 8: Gaussians bar
        // ------------------------------------------------------------------
        let gauss_bar = MetricBar::new("Gaussians", state.num_gaussians as f32, 0.0, 300_000.0)
            .with_color("blue")
            .with_width(bar_w);
        lines.push(gauss_bar.render());

        // ------------------------------------------------------------------
        // Line 9: Speed and ETA
        // ------------------------------------------------------------------
        let eta = self.format_duration(state.eta_secs());
        let elapsed = self.format_duration(state.elapsed_secs);
        let gauss_fmt = self.format_number(state.num_gaussians);
        let speed_line = format!(
            "  {}Speed:{} {:.1} it/s  {}ETA:{} {}  {}Elapsed:{} {}  {}Gaussians:{} {}",
            ansi::BOLD,
            ansi::RESET,
            state.iter_per_sec,
            ansi::BOLD,
            ansi::RESET,
            eta,
            ansi::BOLD,
            ansi::RESET,
            elapsed,
            ansi::BOLD,
            ansi::RESET,
            gauss_fmt,
        );
        // Pad to panel width for clean display
        let speed_stripped = strip_ansi_len(&speed_line);
        let speed_padded = if speed_stripped < width {
            format!("{}{}", speed_line, " ".repeat(width - speed_stripped))
        } else {
            speed_line
        };
        lines.push(speed_padded);

        // ------------------------------------------------------------------
        // Line 10: Photo / perceptual loss breakdown
        // ------------------------------------------------------------------
        let breakdown_line = format!(
            "  {}Photo:{} {:.4}  {}Percep:{} {:.4}",
            ansi::DIM,
            ansi::RESET,
            state.photometric_loss,
            ansi::DIM,
            ansi::RESET,
            state.perceptual_loss,
        );
        lines.push(breakdown_line);

        // ------------------------------------------------------------------
        // Line 11: Bottom separator
        // ------------------------------------------------------------------
        lines.push(std::iter::repeat_n('─', width).collect::<String>());

        lines.join("\n")
    }

    /// Format a duration in seconds as `"H:MM:SS"`, `"MM:SS"`, or `"Xs"`.
    ///
    /// - `< 60 s`: `"45s"`
    /// - `< 3600 s`: `"23:45"`
    /// - `≥ 3600 s`: `"1:23:45"`
    #[must_use]
    pub fn format_duration(&self, secs: f64) -> String {
        if secs < 0.0 {
            return "0s".to_string();
        }
        let total_secs = secs as u64;
        let h = total_secs / 3600;
        let m = (total_secs % 3600) / 60;
        let s = total_secs % 60;

        if h > 0 {
            format!("{}:{:02}:{:02}", h, m, s)
        } else if m > 0 {
            format!("{:02}:{:02}", m, s)
        } else {
            format!("{}s", s)
        }
    }

    /// Format a large integer with K/M suffix and one decimal place.
    ///
    /// - `< 1 000`: plain integer, e.g., `"999"`
    /// - `< 1 000 000`: K suffix, e.g., `"123.5K"`
    /// - `≥ 1 000 000`: M suffix, e.g., `"1.2M"`
    #[must_use]
    pub fn format_number(&self, n: usize) -> String {
        if n >= 1_000_000 {
            format!("{:.1}M", n as f64 / 1_000_000.0)
        } else if n >= 1_000 {
            format!("{:.1}K", n as f64 / 1_000.0)
        } else {
            format!("{}", n)
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Spark line tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_spark_line_empty() {
        let result = format_spark_line(&[], 10);
        assert_eq!(
            result.chars().count(),
            10,
            "should return spaces for empty input"
        );
        assert!(
            result.chars().all(|c| c == ' '),
            "all chars should be spaces"
        );
    }

    #[test]
    fn test_spark_line_uniform() {
        let values: Vec<f32> = vec![5.0; 15];
        let result = format_spark_line(&values, 10);
        assert_eq!(result.chars().count(), 10);
        // When all values equal, all chars should be the middle block
        assert!(
            result.chars().all(|c| c == BLOCK_MID),
            "uniform values should produce all '▄' but got: {}",
            result
        );
    }

    #[test]
    fn test_spark_line_increasing() {
        let values: Vec<f32> = (0..20).map(|i| i as f32).collect();
        let result = format_spark_line(&values, 8);
        assert_eq!(result.chars().count(), 8);

        let chars: Vec<char> = result.chars().collect();
        let first = chars.first().copied().unwrap_or('▁');
        let last = chars.last().copied().unwrap_or('█');

        // For monotonically increasing input, last block >= first block
        let first_level = BLOCKS.iter().position(|&c| c == first).unwrap_or(0);
        let last_level = BLOCKS.iter().position(|&c| c == last).unwrap_or(7);
        assert!(
            last_level >= first_level,
            "increasing values should produce ascending blocks: first={} last={}",
            first,
            last
        );
    }

    #[test]
    fn test_spark_line_width() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        for width in [1, 5, 10, 20, 50] {
            let result = format_spark_line(&values, width);
            assert_eq!(
                result.chars().count(),
                width,
                "output width should be exactly {} chars",
                width
            );
        }
    }

    #[test]
    fn test_spark_line_zero_width() {
        let result = format_spark_line(&[1.0, 2.0, 3.0], 0);
        assert!(result.is_empty(), "zero width should return empty string");
    }

    #[test]
    fn test_spark_line_single_value() {
        let result = format_spark_line(&[std::f32::consts::PI], 5);
        assert_eq!(result.chars().count(), 5);
    }

    // -----------------------------------------------------------------------
    // MetricBar tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_metric_bar_fraction() {
        let bar = MetricBar::new("Test", 5.0, 0.0, 10.0);
        let frac = bar.fraction();
        assert!(
            (frac - 0.5).abs() < 1e-6,
            "fraction of 5/10 should be 0.5, got {}",
            frac
        );
    }

    #[test]
    fn test_metric_bar_fraction_min_max_equal() {
        let bar = MetricBar::new("Test", 5.0, 5.0, 5.0);
        assert_eq!(bar.fraction(), 0.0, "equal min/max should return 0.0");
    }

    #[test]
    fn test_metric_bar_render_not_empty() {
        let bar = MetricBar::new("PSNR", 28.3, 0.0, 45.0)
            .with_unit("dB")
            .with_color("green");
        let rendered = bar.render();
        assert!(
            !rendered.is_empty(),
            "render should produce non-empty output"
        );
    }

    #[test]
    fn test_metric_bar_render_contains_label() {
        let bar = MetricBar::new("PSNR", 28.3, 0.0, 45.0).with_unit("dB");
        let rendered = bar.render();
        assert!(
            rendered.contains("PSNR"),
            "rendered bar should contain the label"
        );
    }

    #[test]
    fn test_metric_bar_clamps_fraction() {
        // Value above max
        let bar_over = MetricBar::new("Over", 50.0, 0.0, 45.0);
        assert_eq!(
            bar_over.fraction(),
            1.0,
            "fraction above max should clamp to 1.0"
        );

        // Value below min
        let bar_under = MetricBar::new("Under", -5.0, 0.0, 45.0);
        assert_eq!(
            bar_under.fraction(),
            0.0,
            "fraction below min should clamp to 0.0"
        );
    }

    #[test]
    fn test_metric_bar_render_contains_blocks() {
        let bar = MetricBar::new("Test", 7.5, 0.0, 10.0).with_width(10);
        let rendered = bar.render();
        // Should contain filled or empty block chars
        assert!(
            rendered.contains('█') || rendered.contains('░'),
            "render should contain block characters"
        );
    }

    // -----------------------------------------------------------------------
    // DashboardPanel tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_dashboard_panel_render_contains_title() {
        let panel = DashboardPanel::new("OxiGAF Training Dashboard");
        let rendered = panel.render(60);
        assert!(
            rendered.contains("OxiGAF Training Dashboard"),
            "panel render should contain the title"
        );
    }

    #[test]
    fn test_dashboard_panel_render_has_borders() {
        let panel = DashboardPanel::new("Test Panel");
        let rendered = panel.render(40);
        assert!(rendered.contains('╔'), "panel should have top-left corner");
        assert!(
            rendered.contains('╝'),
            "panel should have bottom-right corner"
        );
        assert!(rendered.contains('║'), "panel should have vertical border");
    }

    #[test]
    fn test_dashboard_panel_render_with_bars() {
        let bar = MetricBar::new("PSNR", 28.3, 0.0, 45.0).with_unit("dB");
        let panel = DashboardPanel::new("Test").add_bar(bar);
        let rendered = panel.render(60);
        assert!(rendered.contains("PSNR"), "panel should contain bar label");
    }

    // -----------------------------------------------------------------------
    // DashboardState tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_dashboard_state_update_step() {
        let mut state = DashboardState::new(30_000);
        state.update_step(1000, 28.3, 0.012, 0.009, 0.003, 60_000);

        assert_eq!(state.step, 1000);
        assert!((state.psnr - 28.3).abs() < 1e-5);
        assert!((state.total_loss - 0.012).abs() < 1e-6);
        assert_eq!(state.num_gaussians, 60_000);
        assert_eq!(state.psnr_history.len(), 1);
        assert_eq!(state.loss_history.len(), 1);
    }

    #[test]
    fn test_dashboard_state_progress_fraction() {
        let mut state = DashboardState::new(10_000);
        state.step = 2_500;
        let frac = state.progress_fraction();
        assert!(
            (frac - 0.25).abs() < 1e-6,
            "progress fraction should be 0.25, got {}",
            frac
        );
    }

    #[test]
    fn test_dashboard_state_progress_fraction_zero_total() {
        let state = DashboardState::new(0);
        assert_eq!(state.progress_fraction(), 0.0);
    }

    #[test]
    fn test_dashboard_state_eta_secs() {
        let mut state = DashboardState::new(10_000);
        state.step = 5_000;
        state.iter_per_sec = 10.0;
        let eta = state.eta_secs();
        // remaining = 5000, speed = 10 → 500 s
        assert!(
            (eta - 500.0).abs() < 1e-6,
            "ETA should be 500 s, got {}",
            eta
        );
    }

    #[test]
    fn test_dashboard_state_eta_secs_zero_speed() {
        let mut state = DashboardState::new(10_000);
        state.step = 5_000;
        state.iter_per_sec = 0.0;
        assert_eq!(state.eta_secs(), 0.0, "zero speed should return 0.0 ETA");
    }

    #[test]
    fn test_dashboard_state_history_cap() {
        let mut state = DashboardState::new(10_000);
        // Push 60 updates — history should never exceed 50
        for i in 0..60 {
            state.update_step(i, i as f32 * 0.5, i as f32 * 0.001, 0.0, 0.0, 1000);
        }
        assert!(
            state.psnr_history.len() <= 50,
            "psnr_history should be capped at 50, got {}",
            state.psnr_history.len()
        );
        assert!(
            state.loss_history.len() <= 50,
            "loss_history should be capped at 50, got {}",
            state.loss_history.len()
        );
    }

    // -----------------------------------------------------------------------
    // DashboardRenderer tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_renderer_render_frame_no_cursor_not_empty() {
        let config = DashboardConfig::default();
        let renderer = DashboardRenderer::new(config);
        let mut state = DashboardState::new(30_000);
        state.update_step(1000, 28.3, 0.012, 0.009, 0.003, 60_000);
        state.iter_per_sec = 12.5;
        state.elapsed_secs = 80.0;

        let frame = renderer.render_frame_no_cursor(&state);
        assert!(
            !frame.is_empty(),
            "render_frame_no_cursor should not return empty string"
        );
    }

    #[test]
    fn test_renderer_render_frame_contains_step() {
        let config = DashboardConfig::default();
        let renderer = DashboardRenderer::new(config);
        let mut state = DashboardState::new(30_000);
        state.update_step(12345, 28.3, 0.012, 0.009, 0.003, 60_000);

        let frame = renderer.render_frame_no_cursor(&state);
        assert!(
            frame.contains("12345"),
            "frame should contain current step number"
        );
        assert!(frame.contains("30000"), "frame should contain total steps");
    }

    #[test]
    fn test_renderer_render_frame_cursor_control() {
        let config = DashboardConfig::default();
        let mut renderer = DashboardRenderer::new(config);
        let state = DashboardState::new(1000);

        // First render: no cursor-up prefix
        let frame1 = renderer.render_frame(&state);
        assert!(
            !frame1.contains("\x1b[11A"),
            "first frame should not contain cursor-up"
        );

        // Second render: should contain cursor-up
        let frame2 = renderer.render_frame(&state);
        assert!(
            frame2.contains("\x1b["),
            "second frame should contain cursor control escape"
        );
    }

    #[test]
    fn test_format_duration_seconds() {
        let renderer = DashboardRenderer::new(DashboardConfig::default());
        assert_eq!(renderer.format_duration(45.0), "45s");
        assert_eq!(renderer.format_duration(0.0), "0s");
        assert_eq!(renderer.format_duration(59.9), "59s");
    }

    #[test]
    fn test_format_duration_minutes() {
        let renderer = DashboardRenderer::new(DashboardConfig::default());
        assert_eq!(renderer.format_duration(90.0), "01:30");
        assert_eq!(renderer.format_duration(60.0), "01:00");
        assert_eq!(renderer.format_duration(3599.0), "59:59");
    }

    #[test]
    fn test_format_duration_hours() {
        let renderer = DashboardRenderer::new(DashboardConfig::default());
        assert_eq!(renderer.format_duration(3600.0), "1:00:00");
        assert_eq!(renderer.format_duration(5025.0), "1:23:45");
    }

    #[test]
    fn test_format_number_k_suffix() {
        let renderer = DashboardRenderer::new(DashboardConfig::default());
        assert_eq!(renderer.format_number(123_456), "123.5K");
        assert_eq!(renderer.format_number(1_000), "1.0K");
        assert_eq!(renderer.format_number(999), "999");
        assert_eq!(renderer.format_number(0), "0");
    }

    #[test]
    fn test_format_number_m_suffix() {
        let renderer = DashboardRenderer::new(DashboardConfig::default());
        assert_eq!(renderer.format_number(1_200_000), "1.2M");
        assert_eq!(renderer.format_number(1_000_000), "1.0M");
    }

    #[test]
    fn test_strip_ansi_len() {
        // Plain text
        assert_eq!(strip_ansi_len("hello"), 5);
        // With ANSI code
        let with_ansi = format!("{}hello{}", ansi::FG_GREEN, ansi::RESET);
        assert_eq!(strip_ansi_len(&with_ansi), 5);
    }
}
