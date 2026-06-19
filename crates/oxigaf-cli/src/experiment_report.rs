//! Experiment comparison report generation for OxiGAF training runs.
//!
//! This module provides:
//! - [`ExperimentMetrics`] — per-experiment recorded training metrics
//! - [`ExperimentComparison`] — multi-experiment comparison with rankings
//! - [`generate_svg_line_chart`] — SVG chart generation without external dependencies
//! - [`HtmlReportGenerator`] — self-contained HTML report (no JS frameworks, no templating crates)

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

/// Errors that can occur during report generation.
#[derive(Debug)]
pub enum ReportError {
    /// No experiments provided to `ExperimentComparison::new`.
    EmptyExperiments,
    /// I/O error when saving the report.
    IoError(std::io::Error),
    /// Invalid or inconsistent data.
    InvalidData(String),
}

impl fmt::Display for ReportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReportError::EmptyExperiments => write!(f, "No experiments provided"),
            ReportError::IoError(e) => write!(f, "I/O error: {e}"),
            ReportError::InvalidData(s) => write!(f, "Invalid data: {s}"),
        }
    }
}

impl std::error::Error for ReportError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ReportError::IoError(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for ReportError {
    fn from(e: std::io::Error) -> Self {
        ReportError::IoError(e)
    }
}

// ---------------------------------------------------------------------------
// HTML escaping helper
// ---------------------------------------------------------------------------

/// Escape special HTML characters in a string.
fn escape_html(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '&' => out.push_str("&amp;"),
            '"' => out.push_str("&quot;"),
            _ => out.push(ch),
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Experiment color palette
// ---------------------------------------------------------------------------

/// Color palette for experiment series (8 colors, cycled for >8 experiments).
const COLORS: &[&str] = &[
    "#ff6b6b", "#4ecdc4", "#45b7d1", "#f7b731", "#5f27cd", "#48dbfb", "#ff9ff3", "#54a0ff",
];

/// Return the color for experiment at `index`, cycling through the palette.
fn color_for(index: usize) -> &'static str {
    COLORS[index % COLORS.len()]
}

// ---------------------------------------------------------------------------
// ExperimentMetrics
// ---------------------------------------------------------------------------

/// Per-experiment training metrics recorded at discrete steps.
#[derive(Debug, Clone)]
pub struct ExperimentMetrics {
    /// Short display name for the experiment.
    pub name: String,
    /// Human-readable description.
    pub description: String,
    /// PSNR values at each recorded step.
    pub psnr_curve: Vec<f32>,
    /// Total loss values at each recorded step.
    pub loss_curve: Vec<f32>,
    /// Training steps corresponding to `psnr_curve` / `loss_curve`.
    pub steps: Vec<usize>,
    /// Number of Gaussians at each recorded step.
    pub num_gaussians_curve: Vec<usize>,
    /// Total wall-clock training time in seconds.
    pub training_time_secs: f64,
    /// Best (maximum) PSNR observed across all steps; set by [`finalize`](Self::finalize).
    pub best_psnr: f32,
    /// Final loss value; set by [`finalize`](Self::finalize).
    pub final_loss: f32,
    /// Arbitrary hyperparameter key-value pairs.
    pub hyperparams: HashMap<String, String>,
}

impl ExperimentMetrics {
    /// Create a new experiment with the given name and default empty fields.
    pub fn new(name: impl Into<String>) -> Self {
        ExperimentMetrics {
            name: name.into(),
            description: String::new(),
            psnr_curve: Vec::new(),
            loss_curve: Vec::new(),
            steps: Vec::new(),
            num_gaussians_curve: Vec::new(),
            training_time_secs: 0.0,
            best_psnr: 0.0,
            final_loss: 0.0,
            hyperparams: HashMap::new(),
        }
    }

    /// Set the description (builder pattern).
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Append a recorded step with PSNR, total loss, and Gaussian count.
    pub fn add_step(&mut self, step: usize, psnr: f32, loss: f32, num_gaussians: usize) {
        self.steps.push(step);
        self.psnr_curve.push(psnr);
        self.loss_curve.push(loss);
        self.num_gaussians_curve.push(num_gaussians);
    }

    /// Compute derived fields (`best_psnr`, `final_loss`) from the recorded curves.
    ///
    /// Must be called after all steps have been added and before accessing
    /// `best_psnr` or `final_loss`.
    pub fn finalize(&mut self) {
        self.best_psnr = self
            .psnr_curve
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        if self.best_psnr.is_infinite() {
            self.best_psnr = 0.0;
        }
        self.final_loss = self.loss_curve.last().cloned().unwrap_or(0.0);
    }

    /// Return the first training step at which PSNR exceeded 95 % of `best_psnr`.
    ///
    /// Returns `None` if the curve is empty or the threshold was never reached.
    pub fn convergence_step(&self) -> Option<usize> {
        if self.psnr_curve.is_empty() {
            return None;
        }
        // Compute best inline so this works before finalize() is called.
        let peak = self
            .psnr_curve
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        if peak.is_infinite() || peak <= 0.0 {
            return None;
        }
        let threshold = 0.95 * peak;
        self.psnr_curve
            .iter()
            .zip(self.steps.iter())
            .find(|(&psnr, _)| psnr > threshold)
            .map(|(_, &step)| step)
    }

    /// Area under the PSNR curve using the trapezoidal rule.
    ///
    /// Uses `steps` as x-coordinates so intervals of unequal width are handled
    /// correctly. Returns 0.0 if fewer than two points are recorded.
    pub fn auc_psnr(&self) -> f32 {
        if self.psnr_curve.len() < 2 || self.steps.len() < 2 {
            return 0.0;
        }
        let mut area = 0.0_f32;
        let n = self.psnr_curve.len().min(self.steps.len());
        for i in 0..n - 1 {
            let dx = (self.steps[i + 1] as f32) - (self.steps[i] as f32);
            let avg_y = (self.psnr_curve[i] + self.psnr_curve[i + 1]) * 0.5;
            area += dx * avg_y;
        }
        area
    }

    /// Stability score based on the variance of PSNR in the final 25 % of steps.
    ///
    /// Defined as `1 / (1 + variance)`. Returns 1.0 if fewer than 2 points are
    /// available in the last quarter.
    pub fn stability_score(&self) -> f32 {
        let n = self.psnr_curve.len();
        if n < 2 {
            return 1.0;
        }
        // Take at least 1 sample; integer ceiling of n/4.
        let quarter = n.div_ceil(4).max(1);
        let start = n - quarter;
        let window = &self.psnr_curve[start..];
        if window.len() < 2 {
            return 1.0;
        }
        let mean = window.iter().sum::<f32>() / (window.len() as f32);
        let variance = window
            .iter()
            .map(|&v| {
                let d = v - mean;
                d * d
            })
            .sum::<f32>()
            / (window.len() as f32);
        1.0 / (1.0 + variance)
    }
}

// ---------------------------------------------------------------------------
// ExperimentComparison
// ---------------------------------------------------------------------------

/// Multi-experiment comparison with aggregate rankings.
#[derive(Debug, Clone)]
pub struct ExperimentComparison {
    /// All experiments included in this comparison.
    pub experiments: Vec<ExperimentMetrics>,
    /// Index into `experiments` of the experiment with the best final PSNR.
    pub best_psnr_idx: usize,
    /// Index into `experiments` of the experiment that converged earliest.
    pub fastest_converge_idx: usize,
    /// Index into `experiments` of the experiment with the highest stability score.
    pub most_stable_idx: usize,
}

impl ExperimentComparison {
    /// Build a comparison from a non-empty list of experiments.
    ///
    /// # Errors
    ///
    /// Returns [`ReportError::EmptyExperiments`] if `experiments` is empty.
    pub fn new(experiments: Vec<ExperimentMetrics>) -> Result<Self, ReportError> {
        if experiments.is_empty() {
            return Err(ReportError::EmptyExperiments);
        }

        // Best PSNR index
        let best_psnr_idx = experiments
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.best_psnr
                    .partial_cmp(&b.best_psnr)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0);

        // Fastest convergence: smallest step value; experiments that never converge are ranked last.
        let fastest_converge_idx = {
            let mut best_idx = 0;
            let mut best_step = usize::MAX;
            for (i, exp) in experiments.iter().enumerate() {
                if let Some(step) = exp.convergence_step() {
                    if step < best_step {
                        best_step = step;
                        best_idx = i;
                    }
                }
            }
            best_idx
        };

        // Most stable: highest stability score
        let most_stable_idx = experiments
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.stability_score()
                    .partial_cmp(&b.stability_score())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0);

        Ok(ExperimentComparison {
            experiments,
            best_psnr_idx,
            fastest_converge_idx,
            most_stable_idx,
        })
    }

    /// Ranking of experiments by best PSNR, descending.
    ///
    /// Returns a vector of `(rank, name, psnr)` where rank 1 is highest PSNR.
    pub fn psnr_ranking(&self) -> Vec<(usize, &str, f32)> {
        let mut indexed: Vec<(usize, f32)> = self
            .experiments
            .iter()
            .enumerate()
            .map(|(i, e)| (i, e.best_psnr))
            .collect();
        indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        indexed
            .into_iter()
            .enumerate()
            .map(|(rank, (idx, psnr))| (rank + 1, self.experiments[idx].name.as_str(), psnr))
            .collect()
    }

    /// Render a plain-text ASCII summary table.
    pub fn format_text_table(&self) -> String {
        let col_name = 24_usize;
        let col_psnr = 10_usize;
        let col_loss = 10_usize;
        let col_time = 12_usize;
        let col_conv = 12_usize;
        let col_stab = 10_usize;

        let sep = format!(
            "+{:-<col_name$}+{:-<col_psnr$}+{:-<col_loss$}+{:-<col_time$}+{:-<col_conv$}+{:-<col_stab$}+",
            "", "", "", "", "", "",
            col_name = col_name + 2,
            col_psnr = col_psnr + 2,
            col_loss = col_loss + 2,
            col_time = col_time + 2,
            col_conv = col_conv + 2,
            col_stab = col_stab + 2,
        );

        let header = format!(
            "| {:<col_name$} | {:<col_psnr$} | {:<col_loss$} | {:<col_time$} | {:<col_conv$} | {:<col_stab$} |",
            "Experiment", "Best PSNR", "Final Loss", "Time (s)", "Conv Step", "Stability",
            col_name = col_name,
            col_psnr = col_psnr,
            col_loss = col_loss,
            col_time = col_time,
            col_conv = col_conv,
            col_stab = col_stab,
        );

        let mut lines = vec![sep.clone(), header, sep.clone()];

        for exp in &self.experiments {
            let conv = exp
                .convergence_step()
                .map(|s| s.to_string())
                .unwrap_or_else(|| "N/A".to_string());
            let stab = format!("{:.4}", exp.stability_score());
            let row = format!(
                "| {:<col_name$} | {:<col_psnr$} | {:<col_loss$} | {:<col_time$} | {:<col_conv$} | {:<col_stab$} |",
                // Truncate name if longer than column
                if exp.name.len() > col_name { &exp.name[..col_name] } else { &exp.name },
                format!("{:.4}", exp.best_psnr),
                format!("{:.6}", exp.final_loss),
                format!("{:.2}", exp.training_time_secs),
                conv,
                stab,
                col_name = col_name,
                col_psnr = col_psnr,
                col_loss = col_loss,
                col_time = col_time,
                col_conv = col_conv,
                col_stab = col_stab,
            );
            lines.push(row);
        }
        lines.push(sep);
        lines.join("\n")
    }
}

// ---------------------------------------------------------------------------
// SVG chart generation
// ---------------------------------------------------------------------------

/// Padding constants for the SVG viewport (pixels).
const PAD_LEFT: f64 = 60.0;
const PAD_TOP: f64 = 30.0;
const PAD_RIGHT: f64 = 30.0;
const PAD_BOTTOM: f64 = 40.0;

/// Legend item height (pixels).
const LEGEND_LINE_HEIGHT: f64 = 18.0;

/// Generate a simple SVG line chart from multiple named data series.
///
/// # Parameters
/// - `series`: slice of `(name, color_hex, data_points)` tuples
/// - `title`: chart title rendered at the top
/// - `y_label`: label for the y-axis
/// - `width` / `height`: SVG viewport dimensions in pixels
///
/// # Returns
/// Complete `<svg>` element as a UTF-8 string.
pub fn generate_svg_line_chart(
    series: &[(&str, &str, &[f32])],
    title: &str,
    y_label: &str,
    width: usize,
    height: usize,
) -> String {
    let w = width as f64;
    let h = height as f64;

    // Plot area boundaries
    let plot_left = PAD_LEFT;
    let plot_right = w - PAD_RIGHT;
    let plot_top = PAD_TOP;
    let plot_bottom = h - PAD_BOTTOM;
    let plot_w = plot_right - plot_left;
    let plot_h = plot_bottom - plot_top;

    // Compute global data bounds across all series
    let mut y_min = f64::INFINITY;
    let mut y_max = f64::NEG_INFINITY;
    let mut x_count: usize = 0;
    for (_, _, pts) in series {
        x_count = x_count.max(pts.len());
        for &v in *pts {
            let fv = v as f64;
            if fv < y_min {
                y_min = fv;
            }
            if fv > y_max {
                y_max = fv;
            }
        }
    }
    if x_count == 0 {
        y_min = 0.0;
        y_max = 1.0;
    }
    if (y_max - y_min).abs() < 1e-9 {
        // All values equal — center vertically
        y_min -= 1.0;
        y_max += 1.0;
    }

    // Helper: map data coordinates → SVG coordinates
    let to_svg_x = |xi: usize, total: usize| -> f64 {
        if total <= 1 {
            return plot_left + plot_w * 0.5;
        }
        plot_left + (xi as f64 / (total as f64 - 1.0)) * plot_w
    };

    let to_svg_y = |v: f64| -> f64 { plot_bottom - ((v - y_min) / (y_max - y_min)) * plot_h };

    let mut svg = String::with_capacity(8192);

    // SVG opening tag — use concat to keep # colors out of format braces
    svg.push_str(&format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {width} {height}\" \
         width=\"{width}\" height=\"{height}\" \
         style=\"background:#1a1a1a;font-family:monospace;\">",
        width = width,
        height = height,
    ));
    svg.push('\n');

    // Title
    svg.push_str(&format!(
        "  <text x=\"{x:.2}\" y=\"20\" fill=\"#88aaff\" font-size=\"14\" \
         text-anchor=\"middle\">{title}</text>",
        x = w / 2.0,
        title = escape_html(title),
    ));
    svg.push('\n');

    // Y-axis label (rotated)
    let y_mid = (plot_top + plot_bottom) / 2.0;
    svg.push_str(&format!(
        "  <text x=\"12\" y=\"{y:.2}\" fill=\"#aaa\" font-size=\"11\" \
         text-anchor=\"middle\" transform=\"rotate(-90,12,{y:.2})\">{label}</text>",
        y = y_mid,
        label = escape_html(y_label),
    ));
    svg.push('\n');

    // Axes
    svg.push_str(&format!(
        "  <line x1=\"{x1:.2}\" y1=\"{y1:.2}\" x2=\"{x2:.2}\" y2=\"{y2:.2}\" \
         stroke=\"#555\" stroke-width=\"1\"/>",
        x1 = plot_left,
        y1 = plot_top,
        x2 = plot_left,
        y2 = plot_bottom,
    ));
    svg.push('\n');
    svg.push_str(&format!(
        "  <line x1=\"{x1:.2}\" y1=\"{y1:.2}\" x2=\"{x2:.2}\" y2=\"{y2:.2}\" \
         stroke=\"#555\" stroke-width=\"1\"/>",
        x1 = plot_left,
        y1 = plot_bottom,
        x2 = plot_right,
        y2 = plot_bottom,
    ));
    svg.push('\n');

    // Y-axis min/max labels
    svg.push_str(&format!(
        "  <text x=\"{x:.2}\" y=\"{y:.2}\" fill=\"#aaa\" font-size=\"10\" \
         text-anchor=\"end\">{val:.2}</text>",
        x = plot_left - 4.0,
        y = plot_bottom + 4.0,
        val = y_min,
    ));
    svg.push('\n');
    svg.push_str(&format!(
        "  <text x=\"{x:.2}\" y=\"{y:.2}\" fill=\"#aaa\" font-size=\"10\" \
         text-anchor=\"end\">{val:.2}</text>",
        x = plot_left - 4.0,
        y = plot_top + 4.0,
        val = y_max,
    ));
    svg.push('\n');

    // X-axis labels (first and last step index)
    if x_count > 0 {
        svg.push_str(&format!(
            "  <text x=\"{x:.2}\" y=\"{y:.2}\" fill=\"#aaa\" font-size=\"10\" \
             text-anchor=\"middle\">0</text>",
            x = plot_left,
            y = plot_bottom + 14.0,
        ));
        svg.push('\n');
        svg.push_str(&format!(
            "  <text x=\"{x:.2}\" y=\"{y:.2}\" fill=\"#aaa\" font-size=\"10\" \
             text-anchor=\"middle\">{val}</text>",
            x = plot_right,
            y = plot_bottom + 14.0,
            val = x_count - 1,
        ));
        svg.push('\n');
    }

    // Polylines for each series
    for (name, color, pts) in series.iter() {
        if pts.is_empty() {
            continue;
        }
        let total = pts.len();
        let points: String = pts
            .iter()
            .enumerate()
            .map(|(xi, &v)| format!("{:.2},{:.2}", to_svg_x(xi, total), to_svg_y(v as f64)))
            .collect::<Vec<_>>()
            .join(" ");
        svg.push_str(&format!(
            "  <polyline points=\"{points}\" fill=\"none\" stroke=\"{color}\" \
             stroke-width=\"2\" stroke-linejoin=\"round\" stroke-linecap=\"round\"/>",
            points = points,
            color = color,
        ));
        svg.push('\n');
        svg.push_str(&format!("  <!-- series: {} -->\n", escape_html(name),));
    }

    // Legend
    let legend_x = plot_right - 160.0;
    let legend_y_start = plot_top + 10.0;
    for (i, (name, color, _)) in series.iter().enumerate() {
        let ly = legend_y_start + i as f64 * LEGEND_LINE_HEIGHT;
        svg.push_str(&format!(
            "  <line x1=\"{x1:.2}\" y1=\"{y:.2}\" x2=\"{x2:.2}\" y2=\"{y:.2}\" \
             stroke=\"{color}\" stroke-width=\"2\"/>",
            x1 = legend_x,
            x2 = legend_x + 20.0,
            y = ly,
            color = color,
        ));
        svg.push('\n');
        svg.push_str(&format!(
            "  <text x=\"{x:.2}\" y=\"{y:.2}\" fill=\"#e0e0e0\" font-size=\"11\">{name}</text>",
            x = legend_x + 24.0,
            y = ly + 4.0,
            name = escape_html(name),
        ));
        svg.push('\n');
    }

    svg.push_str("</svg>");
    svg
}

// ---------------------------------------------------------------------------
// HTML Report
// ---------------------------------------------------------------------------

/// Configuration for the HTML experiment report.
#[derive(Debug, Clone)]
pub struct HtmlReportConfig {
    /// Title shown in `<title>` and `<h1>`.
    pub title: String,
    /// Whether to include the Gaussian count over time chart.
    pub include_gaussians_chart: bool,
    /// Whether to include the hyperparameters comparison table.
    pub include_hyperparams_table: bool,
    /// SVG chart width in pixels.
    pub chart_width: usize,
    /// SVG chart height in pixels.
    pub chart_height: usize,
}

impl Default for HtmlReportConfig {
    fn default() -> Self {
        HtmlReportConfig {
            title: "OxiGAF Experiment Comparison".to_string(),
            include_gaussians_chart: true,
            include_hyperparams_table: true,
            chart_width: 900,
            chart_height: 350,
        }
    }
}

/// Generator for a self-contained HTML experiment comparison report.
pub struct HtmlReportGenerator {
    comparison: ExperimentComparison,
    config: HtmlReportConfig,
}

impl HtmlReportGenerator {
    /// Create a new report generator.
    pub fn new(comparison: ExperimentComparison, config: HtmlReportConfig) -> Self {
        HtmlReportGenerator { comparison, config }
    }

    /// Generate the complete self-contained HTML report as a `String`.
    pub fn generate(&self) -> String {
        let exps = &self.comparison.experiments;

        // Build per-experiment color lists
        let series_psnr: Vec<(&str, &str, &[f32])> = exps
            .iter()
            .enumerate()
            .map(|(i, e)| (e.name.as_str(), color_for(i), e.psnr_curve.as_slice()))
            .collect();

        let series_loss: Vec<(&str, &str, &[f32])> = exps
            .iter()
            .enumerate()
            .map(|(i, e)| (e.name.as_str(), color_for(i), e.loss_curve.as_slice()))
            .collect();

        let psnr_svg = generate_svg_line_chart(
            &series_psnr,
            "PSNR over Training Steps",
            "PSNR (dB)",
            self.config.chart_width,
            self.config.chart_height,
        );

        let loss_svg = generate_svg_line_chart(
            &series_loss,
            "Loss over Training Steps",
            "Loss",
            self.config.chart_width,
            self.config.chart_height,
        );

        // Summary table HTML
        let summary_table = self.build_summary_table_html();

        // Gaussians chart (optional)
        let gaussians_section = if self.config.include_gaussians_chart {
            let series_gauss: Vec<(String, &str, Vec<f32>)> = exps
                .iter()
                .enumerate()
                .map(|(i, e)| {
                    let pts: Vec<f32> = e.num_gaussians_curve.iter().map(|&n| n as f32).collect();
                    (e.name.clone(), color_for(i), pts)
                })
                .collect();
            let series_refs: Vec<(&str, &str, &[f32])> = series_gauss
                .iter()
                .map(|(name, color, pts)| (name.as_str(), *color, pts.as_slice()))
                .collect();
            let gauss_svg = generate_svg_line_chart(
                &series_refs,
                "Gaussian Count over Training Steps",
                "# Gaussians",
                self.config.chart_width,
                self.config.chart_height,
            );
            format!("<h2>Gaussian Count</h2>\n{}\n", gauss_svg)
        } else {
            String::new()
        };

        // Hyperparameters section (optional)
        let hyperparams_section = if self.config.include_hyperparams_table {
            self.build_hyperparams_section_html()
        } else {
            String::new()
        };

        // Rankings section
        let rankings_section = self.build_rankings_html();

        // Assemble full HTML
        format!(
            r#"<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{ font-family: monospace; max-width: 1200px; margin: auto; padding: 20px; background: #1a1a1a; color: #e0e0e0; }}
    table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
    th, td {{ border: 1px solid #444; padding: 8px; text-align: left; }}
    th {{ background: #333; }}
    .best {{ background: #1a3a1a; }}
    h1, h2 {{ color: #88aaff; }}
    pre {{ background: #222; padding: 12px; border-radius: 4px; overflow-x: auto; }}
    footer {{ margin-top: 40px; color: #666; border-top: 1px solid #333; padding-top: 12px; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <h2>Summary</h2>
{summary_table}
{rankings_section}
  <h2>PSNR Curves</h2>
{psnr_svg}
  <h2>Loss Curves</h2>
{loss_svg}
{gaussians_section}
{hyperparams_section}
  <footer><small>Generated by OxiGAF oxigaf-cli</small></footer>
</body>
</html>"#,
            title = escape_html(&self.config.title),
            summary_table = summary_table,
            rankings_section = rankings_section,
            psnr_svg = psnr_svg,
            loss_svg = loss_svg,
            gaussians_section = gaussians_section,
            hyperparams_section = hyperparams_section,
        )
    }

    /// Build the HTML summary `<table>`.
    fn build_summary_table_html(&self) -> String {
        let exps = &self.comparison.experiments;
        let best_psnr_idx = self.comparison.best_psnr_idx;

        let mut html = String::from(
            "  <table>\n    <tr>\
             <th>#</th><th>Experiment</th><th>Best PSNR</th><th>Final Loss</th>\
             <th>Training Time (s)</th><th>Conv. Step</th><th>Stability</th><th>AUC-PSNR</th>\
             </tr>\n",
        );

        for (i, exp) in exps.iter().enumerate() {
            let row_class = if i == best_psnr_idx {
                " class=\"best\""
            } else {
                ""
            };
            let conv = exp
                .convergence_step()
                .map(|s| s.to_string())
                .unwrap_or_else(|| "N/A".to_string());

            html.push_str(&format!(
                "    <tr{cls}><td>{rank}</td><td>{name}</td><td>{psnr:.4}</td>\
                 <td>{loss:.6}</td><td>{time:.2}</td><td>{conv}</td>\
                 <td>{stab:.4}</td><td>{auc:.2}</td></tr>\n",
                cls = row_class,
                rank = i + 1,
                name = escape_html(&exp.name),
                psnr = exp.best_psnr,
                loss = exp.final_loss,
                time = exp.training_time_secs,
                conv = conv,
                stab = exp.stability_score(),
                auc = exp.auc_psnr(),
            ));
        }
        html.push_str("  </table>");
        html
    }

    /// Build the PSNR rankings section HTML.
    fn build_rankings_html(&self) -> String {
        let ranking = self.comparison.psnr_ranking();
        let mut html = String::from("  <h2>PSNR Rankings</h2>\n  <table>\n    <tr><th>Rank</th><th>Experiment</th><th>Best PSNR</th></tr>\n");
        for (rank, name, psnr) in &ranking {
            let best_class = if *rank == 1 { " class=\"best\"" } else { "" };
            html.push_str(&format!(
                "    <tr{cls}><td>{rank}</td><td>{name}</td><td>{psnr:.4}</td></tr>\n",
                cls = best_class,
                rank = rank,
                name = escape_html(name),
                psnr = psnr,
            ));
        }
        html.push_str("  </table>");
        html
    }

    /// Build the hyperparameters comparison section HTML.
    fn build_hyperparams_section_html(&self) -> String {
        let exps = &self.comparison.experiments;

        // Collect all unique keys (sorted for determinism)
        let mut all_keys: Vec<String> = exps
            .iter()
            .flat_map(|e| e.hyperparams.keys().cloned())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        all_keys.sort();

        if all_keys.is_empty() {
            return String::new();
        }

        let mut html =
            String::from("  <h2>Hyperparameters</h2>\n  <table>\n    <tr><th>Parameter</th>");
        for exp in exps {
            html.push_str(&format!("<th>{}</th>", escape_html(&exp.name)));
        }
        html.push_str("</tr>\n");

        for key in &all_keys {
            html.push_str(&format!(
                "    <tr><td><strong>{}</strong></td>",
                escape_html(key)
            ));
            for exp in exps {
                let val = exp.hyperparams.get(key).map(String::as_str).unwrap_or("-");
                html.push_str(&format!("<td>{}</td>", escape_html(val)));
            }
            html.push_str("</tr>\n");
        }
        html.push_str("  </table>");
        html
    }

    /// Save the generated HTML report to `path`.
    ///
    /// # Errors
    ///
    /// Returns [`ReportError::IoError`] if the file cannot be written.
    pub fn save(&self, path: &std::path::Path) -> Result<(), ReportError> {
        let html = self.generate();
        std::fs::write(path, html.as_bytes())?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- Helpers ---

    fn make_experiment(name: &str, steps: usize) -> ExperimentMetrics {
        let mut exp = ExperimentMetrics::new(name);
        for i in 0..steps {
            let s = i * 100;
            let psnr = 20.0 + (i as f32) * 0.5;
            let loss = 1.0 / (1.0 + i as f32);
            exp.add_step(s, psnr, loss, 1000 + i * 10);
        }
        exp.finalize();
        exp
    }

    // -----------------------------------------------------------------------
    // ExperimentMetrics tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_experiment_metrics_new() {
        let exp = ExperimentMetrics::new("baseline");
        assert_eq!(exp.name, "baseline");
        assert!(exp.psnr_curve.is_empty());
        assert!(exp.loss_curve.is_empty());
        assert!(exp.steps.is_empty());
        assert_eq!(exp.best_psnr, 0.0);
        assert_eq!(exp.final_loss, 0.0);
    }

    #[test]
    fn test_experiment_add_step_and_finalize() {
        let mut exp = ExperimentMetrics::new("test");
        exp.add_step(0, 18.0, 1.0, 500);
        exp.add_step(100, 22.0, 0.5, 600);
        exp.add_step(200, 25.0, 0.2, 700);
        assert_eq!(exp.psnr_curve.len(), 3);
        assert_eq!(exp.steps, vec![0, 100, 200]);
        exp.finalize();
        assert!((exp.best_psnr - 25.0).abs() < 1e-5);
        assert!((exp.final_loss - 0.2).abs() < 1e-5);
    }

    #[test]
    fn test_experiment_best_psnr() {
        let mut exp = ExperimentMetrics::new("psnr_test");
        exp.add_step(0, 10.0, 2.0, 100);
        exp.add_step(1, 30.0, 1.0, 200);
        exp.add_step(2, 25.0, 0.5, 300);
        exp.finalize();
        assert!((exp.best_psnr - 30.0).abs() < 1e-5);
    }

    #[test]
    fn test_experiment_auc_psnr() {
        let mut exp = ExperimentMetrics::new("auc");
        // steps: 0, 100, 200; psnr: 20, 30, 40
        exp.add_step(0, 20.0, 1.0, 100);
        exp.add_step(100, 30.0, 0.5, 200);
        exp.add_step(200, 40.0, 0.2, 300);
        // AUC trapezoidal:
        //   (100 * (20+30)/2) + (100 * (30+40)/2) = 2500 + 3500 = 6000
        let auc = exp.auc_psnr();
        assert!((auc - 6000.0).abs() < 1.0, "AUC = {auc}");
    }

    #[test]
    fn test_experiment_auc_psnr_empty() {
        let exp = ExperimentMetrics::new("empty");
        assert_eq!(exp.auc_psnr(), 0.0);
    }

    #[test]
    fn test_experiment_stability_score() {
        let mut exp = ExperimentMetrics::new("stable");
        // All last-quarter values are equal → variance ≈ 0 → score ≈ 1
        for i in 0..20 {
            exp.add_step(i * 10, 30.0, 0.1, 1000);
        }
        let score = exp.stability_score();
        assert!(score > 0.99, "stability = {score}");
    }

    #[test]
    fn test_experiment_stability_score_noisy() {
        let mut exp = ExperimentMetrics::new("noisy");
        // Highly oscillating values → low score
        for i in 0..40 {
            let v = if i % 2 == 0 { 10.0_f32 } else { 40.0_f32 };
            exp.add_step(i * 10, v, 0.5, 1000);
        }
        let score = exp.stability_score();
        assert!(score < 0.5, "stability = {score}");
    }

    #[test]
    fn test_experiment_convergence_step() {
        let mut exp = ExperimentMetrics::new("conv");
        // peak PSNR is 40.0; 95% threshold = 38.0
        // step 0: 10 (below), step 100: 35 (below), step 200: 39 (above) → converges at 200
        exp.add_step(0, 10.0, 1.0, 100);
        exp.add_step(100, 35.0, 0.5, 200);
        exp.add_step(200, 39.0, 0.2, 300);
        exp.add_step(300, 40.0, 0.1, 400);
        let conv = exp.convergence_step();
        assert_eq!(conv, Some(200));
    }

    #[test]
    fn test_experiment_convergence_step_empty() {
        let exp = ExperimentMetrics::new("empty");
        assert_eq!(exp.convergence_step(), None);
    }

    // -----------------------------------------------------------------------
    // ExperimentComparison tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_comparison_empty_error() {
        let result = ExperimentComparison::new(vec![]);
        assert!(matches!(result, Err(ReportError::EmptyExperiments)));
    }

    #[test]
    fn test_comparison_new() {
        let exps = vec![make_experiment("exp_a", 10), make_experiment("exp_b", 10)];
        let cmp = ExperimentComparison::new(exps).expect("comparison should succeed");
        assert_eq!(cmp.experiments.len(), 2);
        // exp_b has same PSNR progression as exp_a — both end at same PSNR
        // best_psnr_idx can be 0 or 1 depending on tie-breaking; just ensure valid index
        assert!(cmp.best_psnr_idx < 2);
    }

    #[test]
    fn test_comparison_psnr_ranking() {
        let mut exp_a = ExperimentMetrics::new("low_psnr");
        exp_a.add_step(0, 20.0, 1.0, 100);
        exp_a.finalize();

        let mut exp_b = ExperimentMetrics::new("high_psnr");
        exp_b.add_step(0, 35.0, 0.5, 200);
        exp_b.finalize();

        let cmp = ExperimentComparison::new(vec![exp_a, exp_b]).expect("ok");
        let ranking = cmp.psnr_ranking();
        assert_eq!(ranking.len(), 2);
        // Rank 1 should be the high-PSNR experiment
        assert_eq!(ranking[0].0, 1);
        assert_eq!(ranking[0].1, "high_psnr");
        assert!(ranking[0].2 > ranking[1].2);
    }

    #[test]
    fn test_comparison_format_text_table() {
        let exps = vec![make_experiment("alpha", 5), make_experiment("beta", 5)];
        let cmp = ExperimentComparison::new(exps).expect("ok");
        let table = cmp.format_text_table();
        assert!(table.contains("alpha"));
        assert!(table.contains("beta"));
        assert!(table.contains("Best PSNR"));
        assert!(table.contains("Stability"));
    }

    // -----------------------------------------------------------------------
    // SVG chart tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_svg_chart_not_empty() {
        let data = vec![1.0_f32, 2.0, 3.0, 2.5];
        let svg = generate_svg_line_chart(
            &[("series1", "#ff0000", &data)],
            "Test Chart",
            "Value",
            600,
            300,
        );
        assert!(!svg.is_empty());
        assert!(svg.starts_with("<svg"));
        assert!(svg.ends_with("</svg>"));
    }

    #[test]
    fn test_svg_chart_contains_polyline() {
        let data = vec![10.0_f32, 20.0, 15.0];
        let svg =
            generate_svg_line_chart(&[("s1", "#4ecdc4", &data)], "Polyline Test", "Y", 400, 200);
        assert!(svg.contains("<polyline"));
        assert!(svg.contains("#4ecdc4"));
    }

    #[test]
    fn test_svg_chart_multiple_series() {
        let d1 = vec![1.0_f32, 2.0, 3.0];
        let d2 = vec![3.0_f32, 2.0, 1.0];
        let svg = generate_svg_line_chart(
            &[("up", "#ff6b6b", &d1), ("down", "#4ecdc4", &d2)],
            "Two Series",
            "Val",
            500,
            250,
        );
        assert!(svg.contains("up"));
        assert!(svg.contains("down"));
        let count = svg.matches("<polyline").count();
        assert_eq!(count, 2);
    }

    #[test]
    fn test_svg_chart_empty_series() {
        // No data points — should still produce a valid SVG without panic
        let svg = generate_svg_line_chart(&[], "Empty", "Y", 400, 200);
        assert!(svg.starts_with("<svg"));
        assert!(svg.ends_with("</svg>"));
    }

    // -----------------------------------------------------------------------
    // HTML report tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_html_report_generate() {
        let exps = vec![make_experiment("run1", 8), make_experiment("run2", 8)];
        let cmp = ExperimentComparison::new(exps).expect("ok");
        let config = HtmlReportConfig::default();
        let gen = HtmlReportGenerator::new(cmp, config);
        let html = gen.generate();
        assert!(!html.is_empty());
        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("</html>"));
    }

    #[test]
    fn test_html_report_contains_title() {
        let exps = vec![make_experiment("t1", 4)];
        let cmp = ExperimentComparison::new(exps).expect("ok");
        let config = HtmlReportConfig {
            title: "My Special Report".to_string(),
            ..Default::default()
        };
        let gen = HtmlReportGenerator::new(cmp, config);
        let html = gen.generate();
        assert!(html.contains("My Special Report"));
        assert!(html.contains("<h1>"));
    }

    #[test]
    fn test_html_report_save_to_file() {
        let exps = vec![make_experiment("save_test", 5)];
        let cmp = ExperimentComparison::new(exps).expect("ok");
        let gen = HtmlReportGenerator::new(cmp, HtmlReportConfig::default());

        let mut path = std::env::temp_dir();
        path.push(format!(
            "oxigaf_report_test_{}.html",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));

        let result = gen.save(&path);
        assert!(result.is_ok(), "save failed: {:?}", result.err());
        assert!(path.exists());

        // Clean up
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_html_report_two_experiments() {
        let mut exp_a = ExperimentMetrics::new("ExperimentA");
        exp_a
            .hyperparams
            .insert("lr".to_string(), "0.001".to_string());
        for i in 0..10_usize {
            exp_a.add_step(i * 50, 20.0 + i as f32, 1.0 / (i + 1) as f32, 1000 + i * 50);
        }
        exp_a.training_time_secs = 120.0;
        exp_a.finalize();

        let mut exp_b = ExperimentMetrics::new("ExperimentB<>&\"");
        exp_b
            .hyperparams
            .insert("lr".to_string(), "0.01".to_string());
        for i in 0..10_usize {
            exp_b.add_step(
                i * 50,
                18.0 + i as f32 * 1.2,
                0.8 / (i + 1) as f32,
                800 + i * 40,
            );
        }
        exp_b.training_time_secs = 95.0;
        exp_b.finalize();

        let cmp = ExperimentComparison::new(vec![exp_a, exp_b]).expect("ok");
        let gen = HtmlReportGenerator::new(cmp, HtmlReportConfig::default());
        let html = gen.generate();

        // Escaped HTML
        assert!(html.contains("ExperimentA"));
        assert!(html.contains("ExperimentB&lt;&gt;&amp;&quot;"));
        // Hyperparams table
        assert!(html.contains("Hyperparameters"));
        assert!(html.contains("0.001"));
        assert!(html.contains("0.01"));
    }

    #[test]
    fn test_html_report_escape_in_svg() {
        let exps = vec![{
            let mut e = ExperimentMetrics::new("run<1>&\"test\"");
            e.add_step(0, 25.0, 0.5, 1000);
            e.finalize();
            e
        }];
        let cmp = ExperimentComparison::new(exps).expect("ok");
        let gen = HtmlReportGenerator::new(cmp, HtmlReportConfig::default());
        let html = gen.generate();
        // Raw unescaped < or > from the experiment name must not appear inside SVG tags
        // The SVG legend text should be escaped
        assert!(html.contains("run&lt;1&gt;&amp;&quot;test&quot;"));
    }
}
