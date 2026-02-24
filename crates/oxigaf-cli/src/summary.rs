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
                    format!("oxigaf render --model {}", ply).bright_cyan()
                );
            }

            if let Some(ref checkpoint) = self.checkpoint_path {
                println!("  • Continue training:");
                println!(
                    "    {}",
                    format!("oxigaf train --resume {}", checkpoint).bright_cyan()
                );
            }

            println!("  • Export to glTF:");
            if let Some(ref ply) = self.ply_path {
                println!(
                    "    {}",
                    format!("oxigaf export --input {} --output model.gltf", ply).bright_cyan()
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
                println!("    oxigaf render --model {}", ply);
            }

            if let Some(ref checkpoint) = self.checkpoint_path {
                println!("  • Continue training:");
                println!("    oxigaf train --resume {}", checkpoint);
            }

            println!("  • Export to glTF:");
            if let Some(ref ply) = self.ply_path {
                println!("    oxigaf export --input {} --output model.gltf", ply);
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

/// Summary information displayed after export completion.
pub struct ExportSummary {
    pub format: String,
    #[allow(dead_code)]
    pub input_file: String,
    pub output_file: String,
    pub file_size_mb: f64,
    pub num_gaussians: u32,
    pub elapsed: Duration,
}

impl ExportSummary {
    /// Print formatted summary to stdout.
    pub fn print(&self) {
        let separator = "═".repeat(65);

        if output::colors_enabled() {
            println!("\n{}", separator.bright_cyan());
            println!("{}", "  Export Complete".bright_green().bold());
            println!("{}\n", separator.bright_cyan());

            println!("{}", "Export Details:".bright_yellow().bold());
            println!(
                "  Format       : {}",
                self.format.to_uppercase().bright_white()
            );
            println!(
                "  Gaussians    : {}",
                format!("{}", self.num_gaussians).bright_white()
            );
            println!(
                "  File Size    : {:.1} MB",
                format!("{:.1}", self.file_size_mb).bright_white()
            );
            println!(
                "  Duration     : {}",
                format_duration(self.elapsed).bright_white()
            );

            println!("\n{}", "Output:".bright_yellow().bold());
            println!(
                "  {}  {}",
                "💾".bright_green(),
                self.output_file.bright_cyan()
            );

            println!("\n{}", "Usage:".bright_yellow().bold());

            match self.format.to_lowercase().as_str() {
                "gltf" | "glb" => {
                    println!("  • View in Blender, Three.js, or other 3D tools");
                    println!("  • Compatible with web viewers");
                }
                "ply" => {
                    println!("  • View with MeshLab or CloudCompare");
                    println!("  • Use for further processing");
                }
                "safetensors" => {
                    println!("  • Load in Python with safetensors library");
                    println!("  • Resume training or fine-tuning");
                }
                _ => {}
            }

            println!("\n{}", separator.bright_cyan());
        } else {
            // No color output
            println!("\n{}", separator);
            println!("  Export Complete");
            println!("{}\n", separator);

            println!("Export Details:");
            println!("  Format       : {}", self.format.to_uppercase());
            println!("  Gaussians    : {}", self.num_gaussians);
            println!("  File Size    : {:.1} MB", self.file_size_mb);
            println!("  Duration     : {}", format_duration(self.elapsed));

            println!("\nOutput:");
            println!("  [FILE] {}", self.output_file);

            println!("\nUsage:");

            match self.format.to_lowercase().as_str() {
                "gltf" | "glb" => {
                    println!("  • View in Blender, Three.js, or other 3D tools");
                    println!("  • Compatible with web viewers");
                }
                "ply" => {
                    println!("  • View with MeshLab or CloudCompare");
                    println!("  • Use for further processing");
                }
                "safetensors" => {
                    println!("  • Load in Python with safetensors library");
                    println!("  • Resume training or fine-tuning");
                }
                _ => {}
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

    #[test]
    fn test_count_files_in_nonexistent_dir() {
        let result = count_files_in_dir("/nonexistent/directory/path");
        assert_eq!(result, None);
    }
}
