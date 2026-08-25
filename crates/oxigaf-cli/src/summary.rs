//! Human-readable summary formatting for CLI command outputs.
//!
//! Provides formatted summary output at command completion with box drawing
//! characters, statistics, resource usage, output paths, and actionable next steps.

use std::time::Duration;

use owo_colors::OwoColorize;

use crate::output;

// ---------------------------------------------------------------------------
// Training Summary
// ---------------------------------------------------------------------------

/// Summary information displayed after training completion.
pub struct TrainingSummary {
    pub total_iterations: u32,
    pub final_loss: f32,
    pub num_gaussians: u32,
    pub num_rigid: u32,
    pub num_flexible: u32,
    pub sh_degree: u32,
    pub elapsed: Duration,
    pub throughput_iters_per_sec: f32,
    pub checkpoint_path: Option<String>,
    pub ply_path: Option<String>,
    pub preview_dir: Option<String>,
    pub peak_memory_mb: Option<u64>,
}

impl TrainingSummary {
    /// Print formatted summary to stdout.
    pub fn print(&self) {
        let separator = "═".repeat(65);

        if output::colors_enabled() {
            println!("\n{}", separator.bright_cyan());
            println!("{}", "  OxiGAF Training Complete".bright_green().bold());
            println!("{}\n", separator.bright_cyan());

            // Model Statistics
            println!("{}", "Model Statistics:".bright_yellow().bold());
            println!(
                "  Gaussians    : {} ({} rigid / {} flexible)",
                format!("{:>8}", self.num_gaussians).bright_white(),
                self.num_rigid.to_string().bright_cyan(),
                self.num_flexible.to_string().bright_cyan()
            );
            println!(
                "  SH Degree    : {}",
                format!("{}", self.sh_degree).bright_white()
            );
            println!(
                "  Final Loss   : {}",
                format!("{:.6}", self.final_loss).bright_white()
            );

            // Training Statistics
            println!("\n{}", "Training:".bright_yellow().bold());
            println!(
                "  Iterations   : {}",
                format!("{:>8}", self.total_iterations).bright_white()
            );
            println!(
                "  Duration     : {}",
                format_duration(self.elapsed).bright_white()
            );
            println!(
                "  Throughput   : {} iter/s",
                format!("{:.2}", self.throughput_iters_per_sec).bright_white()
            );

            if let Some(mem) = self.peak_memory_mb {
                println!("  Peak Memory  : {} MB", format!("{}", mem).bright_white());
            }

            // Output Files
            println!("\n{}", "Outputs:".bright_yellow().bold());

            if let Some(ref path) = self.checkpoint_path {
                println!("  {}  {}", "💾".bright_green(), path.bright_cyan());
            }

            if let Some(ref path) = self.ply_path {
                println!("  {}  {}", "📦".bright_green(), path.bright_cyan());
            }

            if let Some(ref dir) = self.preview_dir {
                let count = count_files_in_dir(dir).unwrap_or(0);
                println!(
                    "  {}  {} ({} images)",
                    "🖼️ ".bright_green(),
                    dir.bright_cyan(),
                    count.to_string().bright_white()
                );
            }

            // Next Steps
            println!("\n{}", "Next Steps:".bright_yellow().bold());

            if let Some(ref ply) = self.ply_path {
                println!("  • View in 3D viewer:");
                println!(
                    "    {}",
                    format!("oxigaf render --model {} --output renders/", ply).bright_cyan()
                );
            }

            if let Some(ref checkpoint) = self.checkpoint_path {
                println!("  • Continue training:");
                println!(
                    "    {}",
                    format!(
                        "oxigaf train --resume {} --input <video> --output <dir> --flame-model <dir>",
                        checkpoint
                    )
                    .bright_cyan()
                );
            }

            println!("  • Export to glTF:");
            if let Some(ref ply) = self.ply_path {
                println!(
                    "    {}",
                    format!(
                        "oxigaf export --model {} --output model.gltf --format gltf",
                        ply
                    )
                    .bright_cyan()
                );
            }

            println!("\n{}", separator.bright_cyan());
        } else {
            // No color output
            println!("\n{}", separator);
            println!("  OxiGAF Training Complete");
            println!("{}\n", separator);

            println!("Model Statistics:");
            println!(
                "  Gaussians    : {:>8} ({} rigid / {} flexible)",
                self.num_gaussians, self.num_rigid, self.num_flexible
            );
            println!("  SH Degree    : {}", self.sh_degree);
            println!("  Final Loss   : {:.6}", self.final_loss);

            println!("\nTraining:");
            println!("  Iterations   : {:>8}", self.total_iterations);
            println!("  Duration     : {}", format_duration(self.elapsed));
            println!(
                "  Throughput   : {:.2} iter/s",
                self.throughput_iters_per_sec
            );

            if let Some(mem) = self.peak_memory_mb {
                println!("  Peak Memory  : {} MB", mem);
            }

            println!("\nOutputs:");

            if let Some(ref path) = self.checkpoint_path {
                println!("  [CKPT] {}", path);
            }

            if let Some(ref path) = self.ply_path {
                println!("  [PLY]  {}", path);
            }

            if let Some(ref dir) = self.preview_dir {
                let count = count_files_in_dir(dir).unwrap_or(0);
                println!("  [IMG]  {} ({} images)", dir, count);
            }

            println!("\nNext Steps:");

            if let Some(ref ply) = self.ply_path {
                println!("  • View in 3D viewer:");
                println!("    oxigaf render --model {} --output renders/", ply);
            }

            if let Some(ref checkpoint) = self.checkpoint_path {
                println!("  • Continue training:");
                println!(
                    "    oxigaf train --resume {} --input <video> --output <dir> --flame-model <dir>",
                    checkpoint
                );
            }

            println!("  • Export to glTF:");
            if let Some(ref ply) = self.ply_path {
                println!(
                    "    oxigaf export --model {} --output model.gltf --format gltf",
                    ply
                );
            }

            println!("\n{}", separator);
        }
    }
}

// ---------------------------------------------------------------------------
// Render Summary
// ---------------------------------------------------------------------------

/// Summary information displayed after rendering completion.
pub struct RenderSummary {
    pub num_views: u32,
    pub resolution: (u32, u32),
    pub format: String,
    pub mode: String,
    pub elapsed: Duration,
    pub output_dir: String,
    pub fps: Option<f32>,
}

impl RenderSummary {
    /// Print formatted summary to stdout.
    pub fn print(&self) {
        let separator = "═".repeat(65);

        if output::colors_enabled() {
            println!("\n{}", separator.bright_cyan());
            println!("{}", "  Rendering Complete".bright_green().bold());
            println!("{}\n", separator.bright_cyan());

            println!("{}", "Render Settings:".bright_yellow().bold());
            println!(
                "  Views        : {}",
                format!("{}", self.num_views).bright_white()
            );
            println!(
                "  Resolution   : {}x{}",
                self.resolution.0.to_string().bright_white(),
                self.resolution.1.to_string().bright_white()
            );
            println!("  Format       : {}", self.format.bright_white());
            println!("  Mode         : {}", self.mode.bright_white());

            println!("\n{}", "Performance:".bright_yellow().bold());
            println!(
                "  Duration     : {}",
                format_duration(self.elapsed).bright_white()
            );

            if let Some(fps) = self.fps {
                println!(
                    "  Throughput   : {} frames/s",
                    format!("{:.2}", fps).bright_white()
                );
            }

            println!("\n{}", "Output:".bright_yellow().bold());
            println!(
                "  {}  {}",
                "📁".bright_green(),
                self.output_dir.bright_cyan()
            );

            let file_count = count_files_in_dir(&self.output_dir).unwrap_or(0);
            if file_count > 0 {
                println!(
                    "  {} files generated",
                    file_count.to_string().bright_white()
                );
            }

            println!("\n{}", separator.bright_cyan());
        } else {
            // No color output
            println!("\n{}", separator);
            println!("  Rendering Complete");
            println!("{}\n", separator);

            println!("Render Settings:");
            println!("  Views        : {}", self.num_views);
            println!(
                "  Resolution   : {}x{}",
                self.resolution.0, self.resolution.1
            );
            println!("  Format       : {}", self.format);
            println!("  Mode         : {}", self.mode);

            println!("\nPerformance:");
            println!("  Duration     : {}", format_duration(self.elapsed));

            if let Some(fps) = self.fps {
                println!("  Throughput   : {:.2} frames/s", fps);
            }

            println!("\nOutput:");
            println!("  [DIR] {}", self.output_dir);

            let file_count = count_files_in_dir(&self.output_dir).unwrap_or(0);
            if file_count > 0 {
                println!("  {} files generated", file_count);
            }

            println!("\n{}", separator);
        }
    }
}

// ---------------------------------------------------------------------------
// Export Summary
// ---------------------------------------------------------------------------

/// Width the label of a `Label       : value` detail row is padded to.
///
/// Every summary block in this module hand-wrote its labels as literals
/// (`"  Format       : "`, `"  Peak Memory  : "`, …) that all place the colon
/// at column 15 — two leading spaces plus a 13-wide label. [`ExportSummary`]
/// builds its rows from data instead, so the padding has to be explicit to
/// keep its colons in the same column as the blocks around it.
const DETAIL_LABEL_WIDTH: usize = 13;

/// Render one uncolored `  Label       : value` detail row.
fn detail_line(label: &str, value: &str) -> String {
    format!("  {label:<DETAIL_LABEL_WIDTH$}: {value}")
}

/// Summary information displayed after export completion.
pub struct ExportSummary {
    pub format: String,
    /// Model the export read from; shown as the `Source` detail line.
    pub input_file: String,
    pub output_file: String,
    pub file_size_mb: f64,
    pub num_gaussians: u32,
    pub elapsed: Duration,
}

impl ExportSummary {
    /// The `Export Details` rows, as `(label, value)` pairs.
    ///
    /// Both the colored and the plain branch of [`Self::print`] render this
    /// one list, so a detail can never again appear in one and be missing
    /// from the other — which is exactly how `input_file` came to be
    /// collected by the export command, stored here, and then printed
    /// nowhere at all (it was carrying an `#[allow(dead_code)]`).
    ///
    /// `Source` is listed first: an export summary that names only the file
    /// it wrote leaves the reader unable to tell *which* model produced it,
    /// which matters as soon as more than one checkpoint is in play.
    fn detail_rows(&self) -> Vec<(&'static str, String)> {
        vec![
            ("Source", self.input_file.clone()),
            ("Format", self.format.to_uppercase()),
            ("Gaussians", self.num_gaussians.to_string()),
            (
                "File Size",
                format!("{} MB", format_file_size_mb(self.file_size_mb)),
            ),
            ("Duration", format_duration(self.elapsed)),
        ]
    }

    /// Print formatted summary to stdout.
    pub fn print(&self) {
        let separator = "═".repeat(65);
        let rows = self.detail_rows();

        if output::colors_enabled() {
            println!("\n{}", separator.bright_cyan());
            println!("{}", "  Export Complete".bright_green().bold());
            println!("{}\n", separator.bright_cyan());

            println!("{}", "Export Details:".bright_yellow().bold());
            for (label, value) in &rows {
                println!(
                    "  {:<width$}: {}",
                    label,
                    value.bright_white(),
                    width = DETAIL_LABEL_WIDTH
                );
            }

            println!("\n{}", "Output:".bright_yellow().bold());
            println!(
                "  {}  {}",
                "💾".bright_green(),
                self.output_file.bright_cyan()
            );

            println!("\n{}", "Usage:".bright_yellow().bold());

            for line in export_usage_lines(&self.format) {
                println!("{line}");
            }

            println!("\n{}", separator.bright_cyan());
        } else {
            // No color output
            println!("\n{}", separator);
            println!("  Export Complete");
            println!("{}\n", separator);

            println!("Export Details:");
            for (label, value) in &rows {
                println!("{}", detail_line(label, value));
            }

            println!("\nOutput:");
            println!("  [FILE] {}", self.output_file);

            println!("\nUsage:");

            for line in export_usage_lines(&self.format) {
                println!("{line}");
            }

            println!("\n{}", separator);
        }
    }
}

// ---------------------------------------------------------------------------
// Helper Functions
// ---------------------------------------------------------------------------

/// Format a duration in a human-readable form.
///
/// Examples:
/// - 45s → "45s"
/// - 90s → "1m 30s"
/// - 3661s → "1h 1m 1s"
fn format_duration(duration: Duration) -> String {
    let total_secs = duration.as_secs();
    let hours = total_secs / 3600;
    let minutes = (total_secs % 3600) / 60;
    let seconds = total_secs % 60;

    if hours > 0 {
        format!("{}h {}m {}s", hours, minutes, seconds)
    } else if minutes > 0 {
        format!("{}m {}s", minutes, seconds)
    } else {
        format!("{}s", seconds)
    }
}

/// Format a file size in megabytes to one decimal place, e.g. `42.5`.
///
/// Kept as a standalone helper (rather than inlined at each call site) so
/// the numeric formatting can be unit-tested independently of whether it is
/// subsequently wrapped in `owo_colors` styling — applying a second `{:.1}`
/// precision specifier to an already-colorized `String` truncates it as text
/// (precision on `String`/`Display` means "max characters", not "decimal
/// places"), so callers must format the number first and print the result
/// with a plain `{}`.
fn format_file_size_mb(mb: f64) -> String {
    format!("{:.1}", mb)
}

/// Return the "Usage" hint lines to show for a given export format string.
///
/// `format` may be a short slug (`"ply"`, `"gltf"`) or one of the
/// human-readable descriptions produced by the export command (e.g.
/// `"glTF 2.0"`, `"point cloud PLY"`, `"surface mesh PLY"`, `"JSON
/// checkpoint"`). Matching is case-insensitive and substring-based so any
/// of those forms resolve to the right advice; returns an empty slice for
/// unrecognized formats.
fn export_usage_lines(format: &str) -> &'static [&'static str] {
    let fmt_lower = format.to_lowercase();
    if fmt_lower.contains("gltf") || fmt_lower.contains("glb") {
        &[
            "  • View in Blender, Three.js, or other 3D tools",
            "  • Compatible with web viewers",
        ]
    } else if fmt_lower.contains("safetensors") {
        &[
            "  • Load in Python with safetensors library",
            "  • Resume training or fine-tuning",
        ]
    } else if fmt_lower.contains("json") {
        &[
            "  • Inspect with any JSON tool or text editor",
            "  • Resume training with `oxigaf train --resume`",
        ]
    } else if fmt_lower.contains("mesh") {
        &[
            "  • View with MeshLab, Blender, or CloudCompare",
            "  • Use for physical simulation or 3D printing",
        ]
    } else if fmt_lower.contains("point cloud") || fmt_lower.contains("pointcloud") {
        &[
            "  • View with MeshLab or CloudCompare",
            "  • Use for point-cloud processing pipelines",
        ]
    } else if fmt_lower.contains("ply") {
        &[
            "  • View with MeshLab or CloudCompare",
            "  • Use for further processing",
        ]
    } else {
        &[]
    }
}

/// Count the number of files (not directories) in a directory.
///
/// Returns `None` if the directory cannot be read.
fn count_files_in_dir(dir: &str) -> Option<usize> {
    std::fs::read_dir(dir)
        .ok()?
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            entry
                .file_type()
                .ok()
                .map(|ft| ft.is_file())
                .unwrap_or(false)
        })
        .count()
        .into()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_duration_seconds() {
        let d = Duration::from_secs(45);
        assert_eq!(format_duration(d), "45s");
    }

    #[test]
    fn test_format_duration_minutes() {
        let d = Duration::from_secs(90);
        assert_eq!(format_duration(d), "1m 30s");
    }

    #[test]
    fn test_format_duration_hours() {
        let d = Duration::from_secs(3661);
        assert_eq!(format_duration(d), "1h 1m 1s");
    }

    #[test]
    fn test_format_duration_hours_only() {
        let d = Duration::from_secs(7200);
        assert_eq!(format_duration(d), "2h 0m 0s");
    }

    #[test]
    fn test_training_summary_no_panic() {
        let summary = TrainingSummary {
            total_iterations: 1000,
            final_loss: 0.0123,
            num_gaussians: 150000,
            num_rigid: 50000,
            num_flexible: 100000,
            sh_degree: 3,
            elapsed: Duration::from_secs(3661),
            throughput_iters_per_sec: 3.5,
            checkpoint_path: Some("/path/to/checkpoint.json".to_string()),
            ply_path: Some("/path/to/model.ply".to_string()),
            preview_dir: Some(
                std::env::temp_dir()
                    .join("oxigaf_nonexistent_preview")
                    .display()
                    .to_string(),
            ),
            peak_memory_mb: Some(4096),
        };

        // Just ensure it doesn't panic
        summary.print();
    }

    #[test]
    fn test_render_summary_no_panic() {
        let summary = RenderSummary {
            num_views: 36,
            resolution: (1920, 1080),
            format: "PNG".to_string(),
            mode: "Turntable".to_string(),
            elapsed: Duration::from_secs(120),
            output_dir: std::env::temp_dir()
                .join("oxigaf_nonexistent_render_out")
                .display()
                .to_string(),
            fps: Some(0.3),
        };

        summary.print();
    }

    #[test]
    fn test_export_summary_no_panic() {
        let summary = ExportSummary {
            format: "gltf".to_string(),
            input_file: "/path/to/input.ply".to_string(),
            output_file: "/path/to/output.gltf".to_string(),
            file_size_mb: 42.5,
            num_gaussians: 100000,
            elapsed: Duration::from_secs(15),
        };

        summary.print();
    }

    // -----------------------------------------------------------------------
    // ExportSummary detail rows
    //
    // Regression: `input_file` was populated by the export command and then
    // printed by neither branch of `print`, so an "Export Complete" block
    // never said which model it had read. The field was suppressed with
    // `#[allow(dead_code)]` instead of being wired up.
    // -----------------------------------------------------------------------

    fn sample_export_summary() -> ExportSummary {
        ExportSummary {
            format: "gltf".to_string(),
            input_file: "/models/subject_042.ply".to_string(),
            output_file: "/exports/subject_042.glb".to_string(),
            file_size_mb: 42.5,
            num_gaussians: 100_000,
            elapsed: Duration::from_secs(75),
        }
    }

    #[test]
    fn test_export_summary_shows_source_file() {
        let summary = sample_export_summary();
        let rows = summary.detail_rows();
        let source = rows
            .iter()
            .find(|(label, _)| *label == "Source")
            .map(|(_, value)| value.clone());
        assert_eq!(
            source,
            Some("/models/subject_042.ply".to_string()),
            "the export summary must name the model it read: {rows:?}"
        );
    }

    #[test]
    fn test_export_summary_rows_cover_every_field() {
        let summary = sample_export_summary();
        let rows = summary.detail_rows();
        let rendered: Vec<String> = rows
            .iter()
            .map(|(label, value)| format!("{label}: {value}"))
            .collect();
        let joined = rendered.join("\n");

        assert!(joined.contains("Source: /models/subject_042.ply"));
        assert!(joined.contains("Format: GLTF"));
        assert!(joined.contains("Gaussians: 100000"));
        assert!(joined.contains("File Size: 42.5 MB"));
        assert!(joined.contains("Duration: 1m 15s"));
    }

    #[test]
    fn test_export_summary_labels_fit_the_column() {
        // `print` pads labels to `DETAIL_LABEL_WIDTH` so the colons line up;
        // a longer label would push its colon out of the column.
        for (label, _) in sample_export_summary().detail_rows() {
            assert!(
                label.len() <= DETAIL_LABEL_WIDTH,
                "detail label {label:?} does not fit the {DETAIL_LABEL_WIDTH}-char label column"
            );
        }
    }

    /// The data-driven detail rows must land their colon in exactly the
    /// column the hand-written literals elsewhere in this module use
    /// (`"  Peak Memory  : "` and friends: two spaces + a 13-wide label, so
    /// the colon sits at index 15). Asserting on `detail_rows` alone would
    /// not catch a padding width that shifts every export detail line
    /// relative to the `Output:` and `Usage:` blocks below it.
    #[test]
    fn test_export_detail_line_colon_column_matches_other_summaries() {
        // Reference literal taken verbatim from `TrainingSummary::print`.
        let reference = "  Peak Memory  : ";
        let colon_column = reference
            .find(':')
            .expect("test: the reference literal contains a colon");
        assert_eq!(colon_column, 15, "reference literal changed shape");

        for (label, value) in sample_export_summary().detail_rows() {
            let line = detail_line(label, &value);
            assert_eq!(
                line.find(':'),
                Some(colon_column),
                "detail line {line:?} puts its colon out of column {colon_column}"
            );
        }
    }

    #[test]
    fn test_export_detail_line_renders_label_and_value() {
        assert_eq!(
            detail_line("Source", "/models/subject_042.ply"),
            "  Source       : /models/subject_042.ply"
        );
    }

    #[test]
    fn test_count_files_in_nonexistent_dir() {
        let result = count_files_in_dir("/nonexistent/directory/path");
        assert_eq!(result, None);
    }

    // -----------------------------------------------------------------------
    // format_file_size_mb
    //
    // Regression test for a bug where the colored branch applied a second
    // `{:.1}` precision specifier to the already-formatted size string,
    // which truncates the string to 1 *character* instead of formatting a
    // number, turning "42.5 MB" into "4 MB".
    // -----------------------------------------------------------------------

    #[test]
    fn test_format_file_size_mb_one_decimal() {
        assert_eq!(format_file_size_mb(42.5), "42.5");
    }

    #[test]
    fn test_format_file_size_mb_rounds() {
        assert_eq!(format_file_size_mb(42.549), "42.5");
        assert_eq!(format_file_size_mb(42.96), "43.0");
    }

    #[test]
    fn test_format_file_size_mb_not_truncated_to_one_char() {
        // The historical bug produced "4" (String precision = max chars)
        // instead of "42.5" (numeric precision = decimal places).
        let formatted = format_file_size_mb(42.5);
        assert_ne!(formatted, "4");
        assert_eq!(formatted.len(), 4);
    }

    // -----------------------------------------------------------------------
    // export_usage_lines
    //
    // Regression tests: main.rs (export subcommand) passes human-readable
    // strings like "glTF 2.0", "point cloud PLY", "surface mesh PLY", and
    // "JSON checkpoint" into ExportSummary.format, not the bare format
    // slugs. Every one of them must resolve to non-empty usage advice.
    // -----------------------------------------------------------------------

    #[test]
    fn test_export_usage_lines_gltf_human_readable() {
        assert!(!export_usage_lines("glTF 2.0").is_empty());
    }

    #[test]
    fn test_export_usage_lines_ply_exact() {
        assert!(!export_usage_lines("PLY").is_empty());
    }

    #[test]
    fn test_export_usage_lines_safetensors() {
        assert!(!export_usage_lines("safetensors").is_empty());
    }

    #[test]
    fn test_export_usage_lines_json_checkpoint() {
        assert!(!export_usage_lines("JSON checkpoint").is_empty());
    }

    #[test]
    fn test_export_usage_lines_point_cloud_ply() {
        assert!(!export_usage_lines("point cloud PLY").is_empty());
    }

    #[test]
    fn test_export_usage_lines_surface_mesh_ply() {
        assert!(!export_usage_lines("surface mesh PLY").is_empty());
    }

    #[test]
    fn test_export_usage_lines_unknown_format_is_empty() {
        assert!(export_usage_lines("totally-unknown-format").is_empty());
    }

    // -----------------------------------------------------------------------
    // Next-step command suggestions must actually parse.
    //
    // Regression tests for suggested commands that referenced a nonexistent
    // `--input` flag on `export` (the flag is `--model`) and omitted
    // required flags on `render`/`train --resume`.
    // -----------------------------------------------------------------------

    #[test]
    fn test_suggested_export_command_parses() {
        // Note: this module is declared once, in `lib.rs`, which denies
        // `clippy::expect_used` under `cfg_attr(not(test), ..)`. Test code is
        // therefore exempt, but asserting on the `Result` states the actual
        // claim ("this command line parses") better than unwrapping it would.
        use crate::cli::Cli;
        use clap::Parser;

        let ply = "/path/to/model.ply";
        let cmd = format!(
            "oxigaf export --model {} --output model.gltf --format gltf",
            ply
        );
        let args: Vec<&str> = cmd.split_whitespace().collect();
        let result = Cli::try_parse_from(args);
        assert!(
            result.is_ok(),
            "suggested `oxigaf export` next-step command must parse, got: {result:?}"
        );
    }

    #[test]
    fn test_suggested_render_command_parses() {
        use crate::cli::Cli;
        use clap::Parser;

        let ply = "/path/to/model.ply";
        let cmd = format!("oxigaf render --model {} --output renders/", ply);
        let args: Vec<&str> = cmd.split_whitespace().collect();
        let result = Cli::try_parse_from(args);
        assert!(
            result.is_ok(),
            "suggested `oxigaf render` next-step command must parse, got: {result:?}"
        );
    }

    #[test]
    fn test_suggested_train_resume_command_parses() {
        use crate::cli::Cli;
        use clap::Parser;

        let checkpoint = "/path/to/checkpoint.json";
        let cmd = format!(
            "oxigaf train --resume {} --input <video> --output <dir> --flame-model <dir>",
            checkpoint
        );
        let args: Vec<&str> = cmd.split_whitespace().collect();
        let result = Cli::try_parse_from(args);
        assert!(
            result.is_ok(),
            "suggested `oxigaf train --resume` next-step command must parse, got: {result:?}"
        );
    }
}
