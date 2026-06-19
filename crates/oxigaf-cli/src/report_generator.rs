//! General-purpose report generation for OxiGAF.
//!
//! This module provides:
//! - [`ReportPage`] — a structured report page with typed sections
//! - [`ReportBuilder`] — builder pattern for constructing pages
//! - [`render_text_report`] — plain text output with ASCII tables
//! - [`render_markdown_report`] — Markdown output with pipe tables
//! - [`render_html_report`] — self-contained HTML with embedded SVG charts
//! - [`svg_line_chart`] — inline SVG line chart generation
//! - [`generate_training_report`] — convenience wrapper for loss/lr histories
//! - [`compute_trend`] / [`series_stats`] — lightweight statistical helpers

use std::fmt::Write as FmtWrite;
use std::path::Path;
use thiserror::Error;

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

/// Errors produced by the report generator.
#[derive(Debug, Error)]
pub enum GeneratorError {
    /// I/O error encountered when writing a report file.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// A rendering step failed (e.g. invalid numeric data).
    #[error("Render error: {0}")]
    RenderError(String),

    /// The supplied data is structurally invalid.
    #[error("Invalid data: {0}")]
    InvalidData(String),

    /// A template expansion failed.
    #[error("Template error: {0}")]
    TemplateError(String),

    /// The data set is empty where non-empty content is required.
    #[error("Empty data: {0}")]
    EmptyData(String),
}

// ---------------------------------------------------------------------------
// ReportSection
// ---------------------------------------------------------------------------

/// Logical category of a section within a [`ReportPage`].
#[derive(Debug, Clone, PartialEq)]
pub enum ReportSection {
    /// High-level executive summary.
    Summary,
    /// Quantitative metrics table or prose.
    Metrics,
    /// Visual charts (rendered as SVG in HTML mode).
    Charts,
    /// Configuration parameters.
    Config,
    /// Diagnostic information (errors, warnings, system info).
    Diagnostics,
}

// ---------------------------------------------------------------------------
// MetricTrend / MetricSummary
// ---------------------------------------------------------------------------

/// Direction of movement for a metric over time.
#[derive(Debug, Clone, PartialEq)]
pub enum MetricTrend {
    /// Metric is increasing.
    Up,
    /// Metric is decreasing.
    Down,
    /// Metric is approximately flat.
    Stable,
}

/// One row in a metrics summary table.
#[derive(Debug, Clone)]
pub struct MetricSummary {
    /// Display name of the metric (e.g. `"PSNR"`).
    pub name: String,
    /// Scalar value at the time of the report.
    pub value: f32,
    /// Physical unit string (e.g. `"dB"`, `"%"`, `""`).
    pub unit: String,
    /// Observed trend direction.
    pub trend: MetricTrend,
    /// Whether `Up` means improvement (e.g. `true` for accuracy, `false` for loss).
    pub is_good_trend: bool,
}

// ---------------------------------------------------------------------------
// ChartData / ChartSeries
// ---------------------------------------------------------------------------

/// One data series within a [`ChartData`].
#[derive(Debug, Clone)]
pub struct ChartSeries {
    /// Legend label for this series.
    pub label: String,
    /// Y-axis values, one per x-axis tick.
    pub values: Vec<f32>,
}

/// Input data for a line chart.
#[derive(Debug, Clone)]
pub struct ChartData {
    /// Chart heading text.
    pub title: String,
    /// Label for the x-axis.
    pub x_label: String,
    /// Label for the y-axis.
    pub y_label: String,
    /// X-axis tick values (same length as each [`ChartSeries::values`] or used for display only).
    pub x_values: Vec<f32>,
    /// One or more data series.
    pub series: Vec<ChartSeries>,
}

// ---------------------------------------------------------------------------
// SectionContent
// ---------------------------------------------------------------------------

/// Typed content payload for a report section.
#[derive(Debug, Clone)]
pub enum SectionContent {
    /// Unformatted prose text.
    Text(String),
    /// Tabular data with header row and body rows.
    Table {
        /// Column headings.
        headers: Vec<String>,
        /// Data rows; each inner Vec has the same length as `headers`.
        rows: Vec<Vec<String>>,
    },
    /// A line chart.
    Chart(ChartData),
    /// Raw HTML fragment (only embedded as-is in HTML output; escaped otherwise).
    Html(String),
}

// ---------------------------------------------------------------------------
// ReportPage
// ---------------------------------------------------------------------------

/// A complete report page composed of typed sections.
#[derive(Debug, Clone)]
pub struct ReportPage {
    /// Main title displayed at the top of the report.
    pub title: String,
    /// Optional subtitle or date stamp.
    pub subtitle: String,
    /// Ordered list of `(section_category, heading_text, content)` triples.
    pub sections: Vec<(ReportSection, String, SectionContent)>,
}

// ---------------------------------------------------------------------------
// ReportFormat / ReportGeneratorConfig
// ---------------------------------------------------------------------------

/// Output format for a rendered report.
#[derive(Debug, Clone, PartialEq)]
pub enum ReportFormat {
    /// Self-contained HTML document with embedded CSS and SVG.
    Html,
    /// Plain text with ASCII table borders.
    PlainText,
    /// GitHub-flavoured Markdown with pipe tables.
    Markdown,
}

/// Configuration for the report generator.
#[derive(Debug, Clone)]
pub struct ReportGeneratorConfig {
    /// Output format (default: [`ReportFormat::Html`]).
    pub format: ReportFormat,
    /// Document title injected into the report header.
    pub title: String,
    /// Whether to render charts (SVG in HTML; skipped in plain text/Markdown).
    pub show_charts: bool,
    /// Downsample series longer than this to exactly `max_series_points` points
    /// using uniform stride selection (default: 500).
    pub max_series_points: usize,
    /// SVG chart width in pixels (default: 800).
    pub chart_width: u32,
    /// SVG chart height in pixels (default: 300).
    pub chart_height: u32,
}

impl Default for ReportGeneratorConfig {
    fn default() -> Self {
        Self {
            format: ReportFormat::Html,
            title: String::from("Report"),
            show_charts: true,
            max_series_points: 500,
            chart_width: 800,
            chart_height: 300,
        }
    }
}

impl ReportGeneratorConfig {
    /// Validate the configuration, returning an error if any field is out of range.
    pub fn validate(&self) -> Result<(), GeneratorError> {
        if self.max_series_points < 1 {
            return Err(GeneratorError::InvalidData(
                "max_series_points must be at least 1".into(),
            ));
        }
        if self.chart_width == 0 {
            return Err(GeneratorError::InvalidData(
                "chart_width must be greater than 0".into(),
            ));
        }
        if self.chart_height == 0 {
            return Err(GeneratorError::InvalidData(
                "chart_height must be greater than 0".into(),
            ));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Color palette
// ---------------------------------------------------------------------------

const SERIES_COLORS: &[&str] = &["#4f86c6", "#e07b54", "#5dba7d", "#c45bca", "#d4b44a"];

fn series_color(idx: usize) -> &'static str {
    SERIES_COLORS[idx % SERIES_COLORS.len()]
}

// ---------------------------------------------------------------------------
// HTML escape helper
// ---------------------------------------------------------------------------

fn escape_html(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for ch in s.chars() {
        match ch {
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '&' => out.push_str("&amp;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            _ => out.push(ch),
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Statistical helpers
// ---------------------------------------------------------------------------

/// Compute the direction of movement of a series.
///
/// Compares the mean of the last quarter of values against the mean of the first
/// quarter. Returns [`MetricTrend::Stable`] if fewer than 2 values are present
/// or if the relative change is within ±1 %.
pub fn compute_trend(values: &[f32]) -> MetricTrend {
    if values.len() < 2 {
        return MetricTrend::Stable;
    }
    let quarter = (values.len() / 4).max(1);
    let first: f32 = values[..quarter].iter().sum::<f32>() / quarter as f32;
    let last: f32 = values[values.len() - quarter..].iter().sum::<f32>() / quarter as f32;
    if first == 0.0 && last == 0.0 {
        return MetricTrend::Stable;
    }
    let base = first.abs().max(last.abs());
    if base == 0.0 {
        return MetricTrend::Stable;
    }
    let rel = (last - first) / base;
    if rel > 0.01 {
        MetricTrend::Up
    } else if rel < -0.01 {
        MetricTrend::Down
    } else {
        MetricTrend::Stable
    }
}

/// Compute basic statistics `(mean, min, max, std)` for a slice.
///
/// Returns `None` for an empty slice. Standard deviation uses population
/// formula (divides by N).
pub fn series_stats(values: &[f32]) -> Option<(f32, f32, f32, f32)> {
    if values.is_empty() {
        return None;
    }
    let n = values.len() as f32;
    let mean = values.iter().sum::<f32>() / n;
    let min = values.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
    Some((mean, min, max, variance.sqrt()))
}

// ---------------------------------------------------------------------------
// Downsample
// ---------------------------------------------------------------------------

/// Downsample `values` to at most `max_points` elements using uniform stride.
///
/// If `values.len() <= max_points` or `max_points == 0`, returns a copy of the
/// input. The first and last elements are always included when downsampling.
pub fn downsample_series(values: &[f32], max_points: usize) -> Vec<f32> {
    if max_points == 0 || values.len() <= max_points {
        return values.to_vec();
    }
    let mut out = Vec::with_capacity(max_points);
    let last = values.len() - 1;
    for i in 0..max_points {
        // Map index in output space to index in input space.
        let src = if max_points == 1 {
            0
        } else {
            (i * last) / (max_points - 1)
        };
        out.push(values[src]);
    }
    out
}

// ---------------------------------------------------------------------------
// ASCII text table helpers
// ---------------------------------------------------------------------------

fn ascii_table(headers: &[String], rows: &[Vec<String>]) -> String {
    let ncols = headers.len();
    if ncols == 0 {
        return String::new();
    }
    // Calculate column widths.
    let mut widths: Vec<usize> = headers.iter().map(|h| h.len()).collect();
    for row in rows {
        for (ci, cell) in row.iter().enumerate() {
            if ci < ncols {
                widths[ci] = widths[ci].max(cell.len());
            }
        }
    }

    let mut out = String::new();
    // Top border.
    let border = border_line(&widths);
    out.push_str(&border);
    out.push('\n');
    // Header row.
    out.push_str(&data_line(headers, &widths));
    out.push('\n');
    // Separator.
    let sep = separator_line(&widths);
    out.push_str(&sep);
    out.push('\n');
    // Data rows.
    for row in rows {
        out.push_str(&data_line(row, &widths));
        out.push('\n');
    }
    // Bottom border.
    out.push_str(&border);
    out.push('\n');
    out
}

fn border_line(widths: &[usize]) -> String {
    let mut s = String::from("+");
    for &w in widths {
        for _ in 0..w + 2 {
            s.push('-');
        }
        s.push('+');
    }
    s
}

fn separator_line(widths: &[usize]) -> String {
    let mut s = String::from("+");
    for &w in widths {
        for _ in 0..w + 2 {
            s.push('=');
        }
        s.push('+');
    }
    s
}

fn data_line(cells: &[String], widths: &[usize]) -> String {
    let mut s = String::from("|");
    for (i, w) in widths.iter().enumerate() {
        let cell = cells.get(i).map(String::as_str).unwrap_or("");
        let _ = write!(s, " {cell:<w$} |", w = w);
    }
    s
}

// ---------------------------------------------------------------------------
// Markdown table helpers
// ---------------------------------------------------------------------------

fn markdown_table(headers: &[String], rows: &[Vec<String>]) -> String {
    if headers.is_empty() {
        return String::new();
    }
    let ncols = headers.len();
    let mut out = String::new();
    // Header row.
    out.push('|');
    for h in headers {
        let _ = write!(out, " {h} |");
    }
    out.push('\n');
    // Separator row.
    out.push('|');
    for _ in 0..ncols {
        out.push_str(" --- |");
    }
    out.push('\n');
    // Data rows.
    for row in rows {
        out.push('|');
        for ci in 0..ncols {
            let cell = row.get(ci).map(String::as_str).unwrap_or("");
            let _ = write!(out, " {cell} |");
        }
        out.push('\n');
    }
    out
}

// ---------------------------------------------------------------------------
// render_text_report
// ---------------------------------------------------------------------------

/// Render a [`ReportPage`] as plain text with `=== heading ===` separators
/// and ASCII table borders.
pub fn render_text_report(page: &ReportPage) -> String {
    let mut out = String::new();
    let title_line = format!("=== {} ===", page.title);
    out.push_str(&title_line);
    out.push('\n');
    if !page.subtitle.is_empty() {
        let _ = writeln!(out, "    {}", page.subtitle);
    }
    out.push('\n');

    for (_, heading, content) in &page.sections {
        let _ = writeln!(out, "--- {heading} ---");
        match content {
            SectionContent::Text(t) => {
                out.push_str(t);
                out.push('\n');
            }
            SectionContent::Table { headers, rows } => {
                out.push_str(&ascii_table(headers, rows));
            }
            SectionContent::Chart(chart) => {
                let _ = writeln!(out, "[Chart: {}]", chart.title);
                let _ = writeln!(out, "  X: {}  Y: {}", chart.x_label, chart.y_label);
                for s in &chart.series {
                    let _ = writeln!(out, "  Series '{}': {} points", s.label, s.values.len());
                }
            }
            SectionContent::Html(raw) => {
                // Strip tags naively for plain text.
                let stripped = strip_html_tags(raw);
                out.push_str(&stripped);
                out.push('\n');
            }
        }
        out.push('\n');
    }
    out
}

/// Very simple HTML tag stripper for plain-text fallback.
fn strip_html_tags(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut in_tag = false;
    for ch in s.chars() {
        match ch {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => out.push(ch),
            _ => {}
        }
    }
    out
}

// ---------------------------------------------------------------------------
// render_markdown_report
// ---------------------------------------------------------------------------

/// Render a [`ReportPage`] as Markdown.
pub fn render_markdown_report(page: &ReportPage) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "# {}", page.title);
    if !page.subtitle.is_empty() {
        let _ = writeln!(out, "\n_{}_", page.subtitle);
    }
    out.push('\n');

    for (_, heading, content) in &page.sections {
        let _ = writeln!(out, "## {heading}");
        out.push('\n');
        match content {
            SectionContent::Text(t) => {
                out.push_str(t);
                out.push('\n');
            }
            SectionContent::Table { headers, rows } => {
                out.push_str(&markdown_table(headers, rows));
            }
            SectionContent::Chart(chart) => {
                let _ = writeln!(out, "**Chart:** {}", chart.title);
                let _ = writeln!(out, "- X axis: {}", chart.x_label);
                let _ = writeln!(out, "- Y axis: {}", chart.y_label);
                for s in &chart.series {
                    let _ = writeln!(out, "- Series `{}`: {} points", s.label, s.values.len());
                }
            }
            SectionContent::Html(raw) => {
                // Embed as an HTML block in Markdown.
                out.push_str(raw);
                out.push('\n');
            }
        }
        out.push('\n');
    }
    out
}

// ---------------------------------------------------------------------------
// svg_line_chart
// ---------------------------------------------------------------------------

/// Generate an SVG line chart from [`ChartData`].
///
/// The SVG uses a fixed `viewBox` with `width`/`height` pixel dimensions.
/// It includes:
/// - A grid (5 horizontal, 5 vertical lines)
/// - Polyline paths for each series in the color palette
/// - Axis labels and tick values
/// - A legend box in the top-right corner
pub fn svg_line_chart(data: &ChartData, width: u32, height: u32) -> Result<String, GeneratorError> {
    if data.series.is_empty() {
        return Err(GeneratorError::EmptyData("ChartData has no series".into()));
    }
    if width == 0 || height == 0 {
        return Err(GeneratorError::InvalidData(
            "SVG dimensions must be > 0".into(),
        ));
    }

    // Layout constants (as f32 for arithmetic).
    let w = width as f32;
    let h = height as f32;
    let pad_left: f32 = 60.0;
    let pad_right: f32 = 20.0;
    let pad_top: f32 = 30.0;
    let pad_bottom: f32 = 50.0;

    let plot_w = w - pad_left - pad_right;
    let plot_h = h - pad_top - pad_bottom;

    if plot_w <= 0.0 || plot_h <= 0.0 {
        return Err(GeneratorError::InvalidData(
            "Chart dimensions too small for content".into(),
        ));
    }

    // Collect all y-values across all series.
    let all_y: Vec<f32> = data
        .series
        .iter()
        .flat_map(|s| s.values.iter().cloned())
        .collect();

    let y_min = all_y.iter().cloned().fold(f32::INFINITY, f32::min);
    let y_max = all_y.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    // Guard against degenerate ranges.
    let (y_min, y_max) = if (y_max - y_min).abs() < 1e-10 {
        (y_min - 1.0, y_max + 1.0)
    } else {
        (y_min, y_max)
    };

    // Determine x range.
    let n_points = data
        .series
        .iter()
        .map(|s| s.values.len())
        .max()
        .unwrap_or(0);
    let x_count = if !data.x_values.is_empty() {
        data.x_values.len()
    } else {
        n_points
    };
    let x_count = x_count.max(1);

    // Helper closures mapping data coordinates → SVG coordinates.
    let to_sx =
        |xi: usize| -> f32 { pad_left + (xi as f32 / (x_count - 1).max(1) as f32) * plot_w };
    let to_sy = |y: f32| -> f32 {
        let t = (y - y_min) / (y_max - y_min);
        pad_top + plot_h - t * plot_h
    };

    let mut svg = String::new();

    // SVG header — build via push_str to avoid format-string conflicts with # colors.
    svg.push_str(&format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {} {}\" width=\"{}\" height=\"{}\" font-family=\"sans-serif\" font-size=\"11\">\n",
        width, height, width, height
    ));

    // Background.
    svg.push_str(&format!(
        "  <rect x=\"0\" y=\"0\" width=\"{}\" height=\"{}\" fill=\"#fafafa\" stroke=\"#ccc\" stroke-width=\"1\"/>\n",
        width, height
    ));

    // Grid lines (5 horizontal, 5 vertical).
    let grid_count = 5usize;
    for i in 0..=grid_count {
        // Horizontal grid line.
        let gy = pad_top + (i as f32 / grid_count as f32) * plot_h;
        svg.push_str(&format!(
            "  <line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" stroke=\"#e0e0e0\" stroke-width=\"1\"/>\n",
            pad_left, gy, pad_left + plot_w, gy
        ));
        // Y-axis tick label.
        let yval = y_max - (i as f32 / grid_count as f32) * (y_max - y_min);
        svg.push_str(&format!(
            "  <text x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"end\" fill=\"#555\">{:.2}</text>\n",
            pad_left - 4.0,
            gy + 4.0,
            yval
        ));

        // Vertical grid line.
        let gx = pad_left + (i as f32 / grid_count as f32) * plot_w;
        svg.push_str(&format!(
            "  <line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" stroke=\"#e0e0e0\" stroke-width=\"1\"/>\n",
            gx, pad_top, gx, pad_top + plot_h
        ));
        // X-axis tick label.
        let xi = (i * (x_count - 1).max(1)) / grid_count;
        let xval = data.x_values.get(xi).cloned().unwrap_or(xi as f32);
        svg.push_str(&format!(
            "  <text x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"middle\" fill=\"#555\">{:.1}</text>\n",
            gx,
            pad_top + plot_h + 16.0,
            xval
        ));
    }

    // Axes border box.
    svg.push_str(&format!(
        "  <rect x=\"{:.1}\" y=\"{:.1}\" width=\"{:.1}\" height=\"{:.1}\" fill=\"none\" stroke=\"#999\" stroke-width=\"1\"/>\n",
        pad_left, pad_top, plot_w, plot_h
    ));

    // X-axis label.
    svg.push_str(&format!(
        "  <text x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"middle\" fill=\"#333\" font-size=\"12\">{}</text>\n",
        pad_left + plot_w / 2.0,
        h - 6.0,
        escape_html(&data.x_label)
    ));

    // Y-axis label (rotated).
    let yl_x = 14.0;
    let yl_y = pad_top + plot_h / 2.0;
    svg.push_str(&format!(
        "  <text x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"middle\" fill=\"#333\" font-size=\"12\" transform=\"rotate(-90,{:.1},{:.1})\">{}</text>\n",
        yl_x, yl_y, yl_x, yl_y,
        escape_html(&data.y_label)
    ));

    // Chart title.
    svg.push_str(&format!(
        "  <text x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"middle\" fill=\"#111\" font-size=\"13\" font-weight=\"bold\">{}</text>\n",
        pad_left + plot_w / 2.0,
        pad_top - 10.0,
        escape_html(&data.title)
    ));

    // Series polylines.
    for (si, series) in data.series.iter().enumerate() {
        if series.values.is_empty() {
            continue;
        }
        let color = series_color(si);
        let points: String = series
            .values
            .iter()
            .enumerate()
            .map(|(xi, &y)| format!("{:.1},{:.1}", to_sx(xi), to_sy(y)))
            .collect::<Vec<_>>()
            .join(" ");
        svg.push_str(&format!(
            "  <polyline points=\"{}\" fill=\"none\" stroke=\"{}\" stroke-width=\"2\" stroke-linejoin=\"round\" stroke-linecap=\"round\"/>\n",
            points, color
        ));
    }

    // Legend box.
    let legend_x = pad_left + plot_w - 10.0 - 120.0;
    let legend_y = pad_top + 10.0;
    let legend_row_h = 18.0;
    let legend_h = data.series.len() as f32 * legend_row_h + 8.0;
    svg.push_str(&format!(
        "  <rect x=\"{:.1}\" y=\"{:.1}\" width=\"120\" height=\"{:.1}\" fill=\"white\" stroke=\"#bbb\" stroke-width=\"1\" rx=\"3\"/>\n",
        legend_x, legend_y, legend_h
    ));
    for (si, series) in data.series.iter().enumerate() {
        let color = series_color(si);
        let ly = legend_y + 8.0 + si as f32 * legend_row_h + legend_row_h / 2.0;
        svg.push_str(&format!(
            "  <line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" stroke=\"{}\" stroke-width=\"2\"/>\n",
            legend_x + 6.0, ly, legend_x + 22.0, ly, color
        ));
        // Truncate label at 14 chars for legend readability.
        let label = if series.label.len() > 14 {
            format!("{}…", &series.label[..13])
        } else {
            series.label.clone()
        };
        svg.push_str(&format!(
            "  <text x=\"{:.1}\" y=\"{:.1}\" fill=\"#333\" font-size=\"10\">{}</text>\n",
            legend_x + 26.0,
            ly + 4.0,
            escape_html(&label)
        ));
    }

    svg.push_str("</svg>");
    Ok(svg)
}

// ---------------------------------------------------------------------------
// HTML report rendering
// ---------------------------------------------------------------------------

/// Embedded CSS for the HTML report.
fn html_css() -> &'static str {
    r#"
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',Arial,sans-serif;font-size:14px;color:#222;background:#fff;line-height:1.5;padding:24px}
h1{font-size:2em;margin-bottom:4px;color:#1a1a2e}
.subtitle{color:#666;font-size:0.95em;margin-bottom:24px}
h2{font-size:1.3em;margin:28px 0 10px;color:#16213e;border-bottom:2px solid #eee;padding-bottom:4px}
p{margin-bottom:12px}
table{border-collapse:collapse;width:100%;margin-bottom:16px}
th{background:#16213e;color:#fff;text-align:left;padding:8px 12px;font-weight:600}
td{padding:7px 12px;border-bottom:1px solid #eee}
tr:nth-child(even) td{background:#f6f8fa}
.chart-wrap{overflow-x:auto;margin-bottom:16px}
.section{margin-bottom:28px}
pre{background:#f4f4f4;padding:12px;border-radius:4px;overflow-x:auto;font-size:13px}
"#
}

/// Render a [`ReportPage`] as a self-contained HTML document.
///
/// Charts within [`SectionContent::Chart`] sections are rendered as inline SVG
/// if `config.show_charts` is `true`.
pub fn render_html_report(
    page: &ReportPage,
    config: &ReportGeneratorConfig,
) -> Result<String, GeneratorError> {
    config.validate()?;

    let mut html = String::new();
    let _ = writeln!(html, "<!DOCTYPE html>");
    let _ = writeln!(html, r#"<html lang="en"><head><meta charset="UTF-8"/>"#);
    let _ = writeln!(
        html,
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"/>"
    );
    let _ = writeln!(html, "<title>{}</title>", escape_html(&page.title));
    let _ = writeln!(html, "<style>{}</style>", html_css());
    let _ = writeln!(html, "</head><body>");
    let _ = writeln!(html, "<h1>{}</h1>", escape_html(&page.title));
    if !page.subtitle.is_empty() {
        let _ = writeln!(
            html,
            r#"<p class="subtitle">{}</p>"#,
            escape_html(&page.subtitle)
        );
    }

    for (_, heading, content) in &page.sections {
        let _ = writeln!(html, r#"<div class="section">"#);
        let _ = writeln!(html, "<h2>{}</h2>", escape_html(heading));

        match content {
            SectionContent::Text(t) => {
                for line in t.lines() {
                    if line.is_empty() {
                        let _ = writeln!(html, "<br/>");
                    } else {
                        let _ = writeln!(html, "<p>{}</p>", escape_html(line));
                    }
                }
            }
            SectionContent::Table { headers, rows } => {
                let _ = writeln!(html, "<table><thead><tr>");
                for h in headers {
                    let _ = writeln!(html, "<th>{}</th>", escape_html(h));
                }
                let _ = writeln!(html, "</tr></thead><tbody>");
                for row in rows {
                    let _ = writeln!(html, "<tr>");
                    for ci in 0..headers.len() {
                        let cell = row.get(ci).map(String::as_str).unwrap_or("");
                        let _ = writeln!(html, "<td>{}</td>", escape_html(cell));
                    }
                    let _ = writeln!(html, "</tr>");
                }
                let _ = writeln!(html, "</tbody></table>");
            }
            SectionContent::Chart(chart_data) => {
                if config.show_charts {
                    // Downsample each series before rendering.
                    let mut downsampled = chart_data.clone();
                    for s in &mut downsampled.series {
                        s.values = downsample_series(&s.values, config.max_series_points);
                    }
                    let svg = svg_line_chart(&downsampled, config.chart_width, config.chart_height)
                        .map_err(|e| {
                            GeneratorError::RenderError(format!(
                                "SVG generation failed for '{}': {e}",
                                chart_data.title
                            ))
                        })?;
                    let _ = writeln!(html, r#"<div class="chart-wrap">{svg}</div>"#);
                } else {
                    let _ = writeln!(
                        html,
                        "<p><em>[Chart: {}]</em></p>",
                        escape_html(&chart_data.title)
                    );
                }
            }
            SectionContent::Html(raw) => {
                html.push_str(raw);
                html.push('\n');
            }
        }

        let _ = writeln!(html, "</div>");
    }

    let _ = writeln!(html, "</body></html>");
    Ok(html)
}

// ---------------------------------------------------------------------------
// format_metric_table
// ---------------------------------------------------------------------------

/// Format a slice of [`MetricSummary`] values as an aligned plain-text table.
///
/// Columns: Name | Value | Unit | Trend | Good?
pub fn format_metric_table(metrics: &[MetricSummary]) -> String {
    if metrics.is_empty() {
        return String::from("(no metrics)\n");
    }
    let headers: Vec<String> = ["Name", "Value", "Unit", "Trend", "Direction"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let rows: Vec<Vec<String>> = metrics
        .iter()
        .map(|m| {
            let trend_str = match m.trend {
                MetricTrend::Up => "↑ Up",
                MetricTrend::Down => "↓ Down",
                MetricTrend::Stable => "→ Stable",
            };
            let good_str = if m.is_good_trend {
                "Up is good"
            } else {
                "Down is good"
            };
            vec![
                m.name.clone(),
                format!("{:.4}", m.value),
                m.unit.clone(),
                trend_str.to_string(),
                good_str.to_string(),
            ]
        })
        .collect();
    ascii_table(&headers, &rows)
}

// ---------------------------------------------------------------------------
// generate_training_report
// ---------------------------------------------------------------------------

/// Generate a formatted training report from `(step, loss, lr)` history.
///
/// Returns an error if `history` is empty.
pub fn generate_training_report(
    history: &[(usize, f32, f32)],
    config: &ReportGeneratorConfig,
) -> Result<String, GeneratorError> {
    if history.is_empty() {
        return Err(GeneratorError::EmptyData(
            "training history is empty".into(),
        ));
    }
    config.validate()?;

    let steps: Vec<f32> = history.iter().map(|&(s, _, _)| s as f32).collect();
    let losses: Vec<f32> = history.iter().map(|&(_, l, _)| l).collect();
    let lrs: Vec<f32> = history.iter().map(|&(_, _, lr)| lr).collect();

    // Build summary metrics.
    let loss_trend = compute_trend(&losses);
    let loss_stats = series_stats(&losses).unwrap_or((0.0, 0.0, 0.0, 0.0));
    let metrics = vec![
        MetricSummary {
            name: "Loss".into(),
            value: losses.last().cloned().unwrap_or(0.0),
            unit: "".into(),
            trend: loss_trend,
            is_good_trend: false,
        },
        MetricSummary {
            name: "Loss (mean)".into(),
            value: loss_stats.0,
            unit: "".into(),
            trend: MetricTrend::Stable,
            is_good_trend: false,
        },
        MetricSummary {
            name: "Loss (min)".into(),
            value: loss_stats.1,
            unit: "".into(),
            trend: MetricTrend::Stable,
            is_good_trend: false,
        },
        MetricSummary {
            name: "LR (final)".into(),
            value: lrs.last().cloned().unwrap_or(0.0),
            unit: "".into(),
            trend: compute_trend(&lrs),
            is_good_trend: false,
        },
    ];

    let loss_chart = ChartData {
        title: "Training Loss".into(),
        x_label: "Step".into(),
        y_label: "Loss".into(),
        x_values: steps.clone(),
        series: vec![ChartSeries {
            label: "loss".into(),
            values: losses.clone(),
        }],
    };
    let lr_chart = ChartData {
        title: "Learning Rate".into(),
        x_label: "Step".into(),
        y_label: "LR".into(),
        x_values: steps,
        series: vec![ChartSeries {
            label: "lr".into(),
            values: lrs,
        }],
    };

    let page = ReportBuilder::new(&config.title)
        .subtitle(format!(
            "Steps: {}  |  Final loss: {:.6}",
            history.last().map(|h| h.0).unwrap_or(0),
            losses.last().cloned().unwrap_or(0.0)
        ))
        .config(config.clone())
        .add_metrics("Metrics Summary", metrics)
        .add_chart(ReportSection::Charts, "Loss Curve", loss_chart)
        .add_chart(ReportSection::Charts, "LR Schedule", lr_chart)
        .render()?;

    Ok(page)
}

// ---------------------------------------------------------------------------
// write_report
// ---------------------------------------------------------------------------

/// Write report content to a file at `path`.
pub fn write_report(content: &str, path: &Path) -> Result<(), GeneratorError> {
    std::fs::write(path, content).map_err(GeneratorError::Io)
}

// ---------------------------------------------------------------------------
// ReportBuilder
// ---------------------------------------------------------------------------

/// Builder for constructing a [`ReportPage`] and rendering it.
pub struct ReportBuilder {
    page: ReportPage,
    config: ReportGeneratorConfig,
}

impl ReportBuilder {
    /// Create a new builder with the given report title.
    pub fn new(title: impl Into<String>) -> Self {
        let title = title.into();
        Self {
            page: ReportPage {
                title: title.clone(),
                subtitle: String::new(),
                sections: Vec::new(),
            },
            config: ReportGeneratorConfig {
                title,
                ..Default::default()
            },
        }
    }

    /// Set the subtitle.
    pub fn subtitle(mut self, subtitle: impl Into<String>) -> Self {
        self.page.subtitle = subtitle.into();
        self
    }

    /// Override the generator configuration.
    pub fn config(mut self, config: ReportGeneratorConfig) -> Self {
        self.config = config;
        self
    }

    /// Add a plain-text section.
    pub fn add_text(
        mut self,
        section: ReportSection,
        heading: impl Into<String>,
        text: impl Into<String>,
    ) -> Self {
        self.page
            .sections
            .push((section, heading.into(), SectionContent::Text(text.into())));
        self
    }

    /// Add a tabular section.
    pub fn add_table(
        mut self,
        section: ReportSection,
        heading: impl Into<String>,
        headers: Vec<String>,
        rows: Vec<Vec<String>>,
    ) -> Self {
        self.page.sections.push((
            section,
            heading.into(),
            SectionContent::Table { headers, rows },
        ));
        self
    }

    /// Add a chart section.
    pub fn add_chart(
        mut self,
        section: ReportSection,
        heading: impl Into<String>,
        chart: ChartData,
    ) -> Self {
        self.page
            .sections
            .push((section, heading.into(), SectionContent::Chart(chart)));
        self
    }

    /// Add a metrics table section rendered via [`format_metric_table`].
    pub fn add_metrics(mut self, heading: impl Into<String>, metrics: Vec<MetricSummary>) -> Self {
        let table_text = format_metric_table(&metrics);
        self.page.sections.push((
            ReportSection::Metrics,
            heading.into(),
            SectionContent::Text(table_text),
        ));
        self
    }

    /// Consume the builder, returning the constructed [`ReportPage`].
    pub fn build(self) -> ReportPage {
        self.page
    }

    /// Consume the builder, render the page according to `config.format`, and
    /// return the formatted string.
    pub fn render(self) -> Result<String, GeneratorError> {
        let format = self.config.format.clone();
        match format {
            ReportFormat::Html => render_html_report(&self.page, &self.config),
            ReportFormat::PlainText => Ok(render_text_report(&self.page)),
            ReportFormat::Markdown => Ok(render_markdown_report(&self.page)),
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    // -----------------------------------------------------------------------
    // ReportGeneratorConfig::validate
    // -----------------------------------------------------------------------

    #[test]
    fn test_config_validate_ok() {
        let cfg = ReportGeneratorConfig::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_validate_zero_max_series_points() {
        let cfg = ReportGeneratorConfig {
            max_series_points: 0,
            ..Default::default()
        };
        let err = cfg.validate().unwrap_err();
        assert!(matches!(err, GeneratorError::InvalidData(_)));
    }

    #[test]
    fn test_config_validate_zero_chart_width() {
        let cfg = ReportGeneratorConfig {
            chart_width: 0,
            ..Default::default()
        };
        assert!(matches!(
            cfg.validate().unwrap_err(),
            GeneratorError::InvalidData(_)
        ));
    }

    #[test]
    fn test_config_validate_zero_chart_height() {
        let cfg = ReportGeneratorConfig {
            chart_height: 0,
            ..Default::default()
        };
        assert!(matches!(
            cfg.validate().unwrap_err(),
            GeneratorError::InvalidData(_)
        ));
    }

    // -----------------------------------------------------------------------
    // downsample_series
    // -----------------------------------------------------------------------

    #[test]
    fn test_downsample_empty() {
        let result = downsample_series(&[], 10);
        assert!(result.is_empty());
    }

    #[test]
    fn test_downsample_less_than_max() {
        let v: Vec<f32> = (0..5).map(|i| i as f32).collect();
        let result = downsample_series(&v, 10);
        assert_eq!(result.len(), 5);
    }

    #[test]
    fn test_downsample_more_than_max() {
        let v: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        let result = downsample_series(&v, 50);
        assert_eq!(result.len(), 50);
        // First and last must be preserved.
        assert_eq!(result[0], 0.0);
        assert_eq!(result[49], 999.0);
    }

    #[test]
    fn test_downsample_exact_max() {
        let v: Vec<f32> = (0..500).map(|i| i as f32).collect();
        let result = downsample_series(&v, 500);
        assert_eq!(result.len(), 500);
    }

    #[test]
    fn test_downsample_max_points_zero() {
        // Zero max_points → return as-is (graceful, not an error).
        let v = vec![1.0f32, 2.0, 3.0];
        let result = downsample_series(&v, 0);
        assert_eq!(result, v);
    }

    // -----------------------------------------------------------------------
    // compute_trend
    // -----------------------------------------------------------------------

    #[test]
    fn test_compute_trend_ascending() {
        let v: Vec<f32> = (0..20).map(|i| i as f32).collect();
        assert_eq!(compute_trend(&v), MetricTrend::Up);
    }

    #[test]
    fn test_compute_trend_descending() {
        let v: Vec<f32> = (0..20).map(|i| (20 - i) as f32).collect();
        assert_eq!(compute_trend(&v), MetricTrend::Down);
    }

    #[test]
    fn test_compute_trend_flat() {
        let v = vec![5.0f32; 20];
        assert_eq!(compute_trend(&v), MetricTrend::Stable);
    }

    #[test]
    fn test_compute_trend_single_element() {
        assert_eq!(compute_trend(&[42.0]), MetricTrend::Stable);
    }

    #[test]
    fn test_compute_trend_empty() {
        assert_eq!(compute_trend(&[]), MetricTrend::Stable);
    }

    // -----------------------------------------------------------------------
    // series_stats
    // -----------------------------------------------------------------------

    #[test]
    fn test_series_stats_empty() {
        assert!(series_stats(&[]).is_none());
    }

    #[test]
    fn test_series_stats_single() {
        let (mean, min, max, std) = series_stats(&[3.0]).unwrap();
        assert!((mean - 3.0).abs() < 1e-6);
        assert!((min - 3.0).abs() < 1e-6);
        assert!((max - 3.0).abs() < 1e-6);
        assert!(std.abs() < 1e-6);
    }

    #[test]
    fn test_series_stats_multiple() {
        let v = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let (mean, min, max, std) = series_stats(&v).unwrap();
        assert!((mean - 3.0).abs() < 1e-5);
        assert!((min - 1.0).abs() < 1e-5);
        assert!((max - 5.0).abs() < 1e-5);
        // Population std of [1,2,3,4,5] = sqrt(2).
        assert!((std - 2.0f32.sqrt()).abs() < 1e-4);
    }

    // -----------------------------------------------------------------------
    // format_metric_table
    // -----------------------------------------------------------------------

    #[test]
    fn test_format_metric_table_empty() {
        let result = format_metric_table(&[]);
        assert!(result.contains("no metrics"));
    }

    #[test]
    fn test_format_metric_table_single() {
        let m = MetricSummary {
            name: "PSNR".into(),
            value: 32.5,
            unit: "dB".into(),
            trend: MetricTrend::Up,
            is_good_trend: true,
        };
        let result = format_metric_table(&[m]);
        assert!(result.contains("PSNR"));
        assert!(result.contains("dB"));
        assert!(result.contains("Up"));
    }

    #[test]
    fn test_format_metric_table_multiple() {
        let metrics = vec![
            MetricSummary {
                name: "Loss".into(),
                value: 0.05,
                unit: "".into(),
                trend: MetricTrend::Down,
                is_good_trend: false,
            },
            MetricSummary {
                name: "Accuracy".into(),
                value: 0.98,
                unit: "%".into(),
                trend: MetricTrend::Up,
                is_good_trend: true,
            },
        ];
        let result = format_metric_table(&metrics);
        assert!(result.contains("Loss"));
        assert!(result.contains("Accuracy"));
    }

    // -----------------------------------------------------------------------
    // render_text_report
    // -----------------------------------------------------------------------

    #[test]
    fn test_render_text_report_basic() {
        let page = ReportPage {
            title: "Test Report".into(),
            subtitle: "subtitle here".into(),
            sections: vec![(
                ReportSection::Summary,
                "Overview".into(),
                SectionContent::Text("Hello world.".into()),
            )],
        };
        let out = render_text_report(&page);
        assert!(out.contains("Test Report"));
        assert!(out.contains("Overview"));
        assert!(out.contains("Hello world."));
    }

    #[test]
    fn test_render_text_report_table_section() {
        let page = ReportPage {
            title: "Table Test".into(),
            subtitle: String::new(),
            sections: vec![(
                ReportSection::Metrics,
                "Data".into(),
                SectionContent::Table {
                    headers: vec!["A".into(), "B".into()],
                    rows: vec![vec!["1".into(), "2".into()]],
                },
            )],
        };
        let out = render_text_report(&page);
        assert!(out.contains('A'));
        assert!(out.contains('B'));
        assert!(out.contains('1'));
    }

    // -----------------------------------------------------------------------
    // render_markdown_report
    // -----------------------------------------------------------------------

    #[test]
    fn test_render_markdown_title() {
        let page = ReportPage {
            title: "My MD Report".into(),
            subtitle: "2026".into(),
            sections: vec![],
        };
        let out = render_markdown_report(&page);
        assert!(out.starts_with("# My MD Report"));
        assert!(out.contains("2026"));
    }

    #[test]
    fn test_render_markdown_table_formatting() {
        let page = ReportPage {
            title: "T".into(),
            subtitle: String::new(),
            sections: vec![(
                ReportSection::Metrics,
                "M".into(),
                SectionContent::Table {
                    headers: vec!["Col1".into(), "Col2".into()],
                    rows: vec![vec!["val1".into(), "val2".into()]],
                },
            )],
        };
        let out = render_markdown_report(&page);
        assert!(out.contains("| Col1 | Col2 |"));
        assert!(out.contains("| --- |"));
        assert!(out.contains("| val1 |"));
    }

    // -----------------------------------------------------------------------
    // render_html_report
    // -----------------------------------------------------------------------

    #[test]
    fn test_render_html_report_structure() {
        let page = ReportPage {
            title: "HTML Test".into(),
            subtitle: "sub".into(),
            sections: vec![(
                ReportSection::Summary,
                "Intro".into(),
                SectionContent::Text("Content here.".into()),
            )],
        };
        let cfg = ReportGeneratorConfig::default();
        let html = render_html_report(&page, &cfg).unwrap();
        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("HTML Test"));
        assert!(html.contains("Content here."));
        assert!(html.contains("</html>"));
    }

    #[test]
    fn test_render_html_report_contains_title() {
        let page = ReportPage {
            title: "Unique Title XYZ".into(),
            subtitle: String::new(),
            sections: vec![],
        };
        let cfg = ReportGeneratorConfig::default();
        let html = render_html_report(&page, &cfg).unwrap();
        assert!(html.contains("Unique Title XYZ"));
    }

    #[test]
    fn test_render_html_report_invalid_config() {
        let page = ReportPage {
            title: "T".into(),
            subtitle: String::new(),
            sections: vec![],
        };
        let cfg = ReportGeneratorConfig {
            chart_width: 0,
            ..Default::default()
        };
        assert!(render_html_report(&page, &cfg).is_err());
    }

    // -----------------------------------------------------------------------
    // svg_line_chart
    // -----------------------------------------------------------------------

    #[test]
    fn test_svg_line_chart_empty_series_error() {
        let data = ChartData {
            title: "Empty".into(),
            x_label: "X".into(),
            y_label: "Y".into(),
            x_values: vec![],
            series: vec![],
        };
        let err = svg_line_chart(&data, 800, 300).unwrap_err();
        assert!(matches!(err, GeneratorError::EmptyData(_)));
    }

    #[test]
    fn test_svg_line_chart_single_point() {
        let data = ChartData {
            title: "Single".into(),
            x_label: "X".into(),
            y_label: "Y".into(),
            x_values: vec![0.0],
            series: vec![ChartSeries {
                label: "s1".into(),
                values: vec![1.0],
            }],
        };
        let svg = svg_line_chart(&data, 800, 300).unwrap();
        assert!(svg.contains("<svg"));
        assert!(svg.contains("</svg>"));
    }

    #[test]
    fn test_svg_line_chart_multiple_series() {
        let data = ChartData {
            title: "Multi".into(),
            x_label: "Step".into(),
            y_label: "Val".into(),
            x_values: (0..10).map(|i| i as f32).collect(),
            series: vec![
                ChartSeries {
                    label: "A".into(),
                    values: (0..10).map(|i| i as f32).collect(),
                },
                ChartSeries {
                    label: "B".into(),
                    values: (0..10).map(|i| (10 - i) as f32).collect(),
                },
            ],
        };
        let svg = svg_line_chart(&data, 800, 300).unwrap();
        assert!(svg.contains("polyline"));
        // Both color palette entries should appear.
        assert!(svg.contains("#4f86c6"));
        assert!(svg.contains("#e07b54"));
    }

    #[test]
    fn test_svg_line_chart_zero_dimensions() {
        let data = ChartData {
            title: "T".into(),
            x_label: "X".into(),
            y_label: "Y".into(),
            x_values: vec![1.0],
            series: vec![ChartSeries {
                label: "s".into(),
                values: vec![1.0],
            }],
        };
        assert!(svg_line_chart(&data, 0, 300).is_err());
        assert!(svg_line_chart(&data, 800, 0).is_err());
    }

    // -----------------------------------------------------------------------
    // generate_training_report
    // -----------------------------------------------------------------------

    #[test]
    fn test_generate_training_report_empty_error() {
        let cfg = ReportGeneratorConfig::default();
        let err = generate_training_report(&[], &cfg).unwrap_err();
        assert!(matches!(err, GeneratorError::EmptyData(_)));
    }

    #[test]
    fn test_generate_training_report_short() {
        let history = vec![(0, 1.0f32, 0.001f32), (10, 0.8, 0.001), (20, 0.6, 0.0005)];
        let cfg = ReportGeneratorConfig {
            format: ReportFormat::PlainText,
            ..Default::default()
        };
        let result = generate_training_report(&history, &cfg).unwrap();
        assert!(!result.is_empty());
        assert!(result.contains("Loss"));
    }

    #[test]
    fn test_generate_training_report_long() {
        let history: Vec<(usize, f32, f32)> = (0..1000)
            .map(|i| (i, 1.0 / (1.0 + i as f32), 0.001))
            .collect();
        let cfg = ReportGeneratorConfig::default();
        let result = generate_training_report(&history, &cfg).unwrap();
        assert!(result.contains("html") || result.contains("HTML") || result.contains("<!DOCTYPE"));
    }

    // -----------------------------------------------------------------------
    // write_report
    // -----------------------------------------------------------------------

    #[test]
    fn test_write_report_to_temp_file() {
        let tmp = env::temp_dir().join("oxigaf_test_report.txt");
        let content = "hello report content";
        write_report(content, &tmp).unwrap();
        let read_back = std::fs::read_to_string(&tmp).unwrap();
        assert_eq!(read_back, content);
        // Clean up.
        let _ = std::fs::remove_file(&tmp);
    }

    // -----------------------------------------------------------------------
    // ReportBuilder
    // -----------------------------------------------------------------------

    #[test]
    fn test_report_builder_chaining() {
        let page = ReportBuilder::new("Builder Test")
            .subtitle("sub")
            .add_text(ReportSection::Summary, "Intro", "Some intro text.")
            .build();
        assert_eq!(page.title, "Builder Test");
        assert_eq!(page.subtitle, "sub");
        assert_eq!(page.sections.len(), 1);
    }

    #[test]
    fn test_report_builder_render_plaintext() {
        let result = ReportBuilder::new("PT Report")
            .config(ReportGeneratorConfig {
                format: ReportFormat::PlainText,
                ..Default::default()
            })
            .add_text(ReportSection::Summary, "S1", "text here")
            .render()
            .unwrap();
        assert!(result.contains("PT Report"));
        assert!(result.contains("text here"));
    }

    #[test]
    fn test_report_builder_render_markdown() {
        let result = ReportBuilder::new("MD Report")
            .config(ReportGeneratorConfig {
                format: ReportFormat::Markdown,
                ..Default::default()
            })
            .add_table(
                ReportSection::Metrics,
                "Stats",
                vec!["Key".into(), "Val".into()],
                vec![vec!["foo".into(), "42".into()]],
            )
            .render()
            .unwrap();
        assert!(result.contains("# MD Report"));
        assert!(result.contains("| Key |"));
    }

    #[test]
    fn test_report_builder_render_html() {
        let result = ReportBuilder::new("HTML Report")
            .add_text(ReportSection::Summary, "H", "content")
            .render()
            .unwrap();
        assert!(result.contains("<!DOCTYPE html>"));
    }

    #[test]
    fn test_report_builder_add_metrics() {
        let metrics = vec![MetricSummary {
            name: "PSNR".into(),
            value: 30.0,
            unit: "dB".into(),
            trend: MetricTrend::Up,
            is_good_trend: true,
        }];
        let page = ReportBuilder::new("R").add_metrics("M", metrics).build();
        assert_eq!(page.sections.len(), 1);
        assert_eq!(page.sections[0].0, ReportSection::Metrics);
    }

    // -----------------------------------------------------------------------
    // ChartData and SectionContent construction
    // -----------------------------------------------------------------------

    #[test]
    fn test_chart_data_construction() {
        let cd = ChartData {
            title: "T".into(),
            x_label: "X".into(),
            y_label: "Y".into(),
            x_values: vec![1.0, 2.0, 3.0],
            series: vec![ChartSeries {
                label: "s".into(),
                values: vec![10.0, 20.0, 30.0],
            }],
        };
        assert_eq!(cd.series.len(), 1);
        assert_eq!(cd.x_values.len(), 3);
    }

    #[test]
    fn test_section_content_variants() {
        let text = SectionContent::Text("hi".into());
        let table = SectionContent::Table {
            headers: vec!["H".into()],
            rows: vec![vec!["R".into()]],
        };
        let html = SectionContent::Html("<b>bold</b>".into());
        // Just verify they can be constructed and cloned.
        let _ = text.clone();
        let _ = table.clone();
        let _ = html.clone();
    }

    #[test]
    fn test_metric_trend_equality() {
        assert_eq!(MetricTrend::Up, MetricTrend::Up);
        assert_ne!(MetricTrend::Up, MetricTrend::Down);
    }

    #[test]
    fn test_report_section_equality() {
        assert_eq!(ReportSection::Summary, ReportSection::Summary);
        assert_ne!(ReportSection::Charts, ReportSection::Diagnostics);
    }
}
