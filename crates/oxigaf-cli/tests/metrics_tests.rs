//! Integration tests for metrics export functionality.

use oxigaf_cli::metrics::{MetricsFormat, MetricsWriter, TrainingMetrics};
use serial_test::serial;
use std::env;
use std::fs;
use std::path::PathBuf;

/// Get a unique temporary file path for testing.
fn temp_metrics_path(suffix: &str) -> PathBuf {
    env::temp_dir().join(format!("oxigaf_metrics_test_{}", suffix))
}

/// Clean up test file if it exists.
fn cleanup_test_file(path: &PathBuf) {
    let _ = fs::remove_file(path);
}

#[test]
#[serial]
fn test_csv_format_creates_header() {
    let path = temp_metrics_path("csv_header.csv");
    cleanup_test_file(&path);

    let result = MetricsWriter::new(&path, MetricsFormat::Csv);
    assert!(
        result.is_ok(),
        "Failed to create CSV writer: {:?}",
        result.err()
    );

    // Drop the writer to ensure flush
    drop(result);

    // Read the file and check for header
    let content = fs::read_to_string(&path).expect("Failed to read metrics file");
    assert!(
        content.contains("iteration,loss_total"),
        "CSV header not found in content: {}",
        content
    );
    assert!(
        content.contains("num_gaussians"),
        "num_gaussians column not found in header"
    );
    assert!(
        content.contains("elapsed_seconds"),
        "elapsed_seconds column not found in header"
    );

    cleanup_test_file(&path);
}

#[test]
#[serial]
fn test_csv_format_writes_data() {
    let path = temp_metrics_path("csv_data.csv");
    cleanup_test_file(&path);

    let mut writer =
        MetricsWriter::new(&path, MetricsFormat::Csv).expect("Failed to create CSV writer");

    let metrics = TrainingMetrics {
        iteration: 1,
        loss_total: 1.234,
        loss_l1: 0.5,
        loss_ssim: 0.3,
        loss_lpips: Some(0.1),
        loss_reg: 0.334,
        num_gaussians: 50000,
        lr_position: 0.00016,
        lr_scaling: 0.005,
        lr_rotation: 0.001,
        memory_mb: Some(4096),
        elapsed_seconds: 120.5,
    };

    let result = writer.write_metrics(&metrics);
    assert!(
        result.is_ok(),
        "Failed to write metrics: {:?}",
        result.err()
    );

    // Drop the writer to ensure flush
    drop(writer);

    // Verify the content
    let content = fs::read_to_string(&path).expect("Failed to read metrics file");
    assert!(content.contains("1,1.234"), "Iteration and loss not found");
    assert!(content.contains("50000"), "Gaussian count not found");
    assert!(content.contains("120.5"), "Elapsed time not found");

    // Verify CSV structure (should have 2 lines: header + data)
    let lines: Vec<&str> = content.lines().collect();
    assert_eq!(lines.len(), 2, "Expected header + 1 data line");

    cleanup_test_file(&path);
}

#[test]
#[serial]
fn test_json_lines_format_writes_valid_json() {
    let path = temp_metrics_path("jsonl.jsonl");
    cleanup_test_file(&path);

    let mut writer = MetricsWriter::new(&path, MetricsFormat::JsonLines)
        .expect("Failed to create JSON Lines writer");

    let metrics = TrainingMetrics {
        iteration: 1,
        loss_total: 1.234,
        loss_l1: 0.5,
        loss_ssim: 0.3,
        loss_lpips: Some(0.1),
        loss_reg: 0.334,
        num_gaussians: 50000,
        lr_position: 0.00016,
        lr_scaling: 0.005,
        lr_rotation: 0.001,
        memory_mb: Some(4096),
        elapsed_seconds: 120.5,
    };

    let result = writer.write_metrics(&metrics);
    assert!(
        result.is_ok(),
        "Failed to write metrics: {:?}",
        result.err()
    );

    // Drop the writer to ensure flush
    drop(writer);

    // Verify the content is valid JSON
    let content = fs::read_to_string(&path).expect("Failed to read metrics file");
    let lines: Vec<&str> = content.lines().collect();
    assert_eq!(lines.len(), 1, "Expected exactly 1 JSON line");

    let parsed: Result<serde_json::Value, _> = serde_json::from_str(lines[0]);
    assert!(parsed.is_ok(), "Failed to parse JSON: {:?}", parsed.err());

    let json = parsed.expect("JSON parsing failed");
    assert_eq!(json["iteration"], 1, "Iteration field incorrect");
    assert_eq!(
        json["num_gaussians"], 50000,
        "num_gaussians field incorrect"
    );
    assert_eq!(
        json["loss_total"]
            .as_f64()
            .expect("loss_total not a number"),
        1.234,
        "loss_total field incorrect"
    );

    cleanup_test_file(&path);
}

#[test]
#[serial]
fn test_multiple_writes_csv() {
    let path = temp_metrics_path("csv_multiple.csv");
    cleanup_test_file(&path);

    let mut writer =
        MetricsWriter::new(&path, MetricsFormat::Csv).expect("Failed to create CSV writer");

    // Write 5 iterations
    for i in 0..5 {
        let metrics = TrainingMetrics {
            iteration: i,
            loss_total: 1.0 - (i as f32 * 0.1),
            loss_l1: 0.5,
            loss_ssim: 0.3,
            loss_lpips: None,
            loss_reg: 0.2,
            num_gaussians: 50000 + i * 100,
            lr_position: 0.00016,
            lr_scaling: 0.005,
            lr_rotation: 0.001,
            memory_mb: None,
            elapsed_seconds: i as f32 * 10.0,
        };

        let result = writer.write_metrics(&metrics);
        assert!(
            result.is_ok(),
            "Failed to write metrics at iteration {}: {:?}",
            i,
            result.err()
        );
    }

    // Drop the writer to ensure flush
    drop(writer);

    // Verify we have header + 5 data lines
    let content = fs::read_to_string(&path).expect("Failed to read metrics file");
    let lines: Vec<&str> = content.lines().collect();
    assert_eq!(
        lines.len(),
        6,
        "Expected header + 5 data lines, got {}",
        lines.len()
    );

    // Verify first data line
    assert!(lines[1].starts_with("0,1"), "First data line incorrect");

    // Verify last data line
    assert!(lines[5].starts_with("4,0.6"), "Last data line incorrect");

    cleanup_test_file(&path);
}

#[test]
#[serial]
fn test_multiple_writes_json_lines() {
    let path = temp_metrics_path("jsonl_multiple.jsonl");
    cleanup_test_file(&path);

    let mut writer = MetricsWriter::new(&path, MetricsFormat::JsonLines)
        .expect("Failed to create JSON Lines writer");

    // Write 3 iterations
    for i in 0..3 {
        let metrics = TrainingMetrics {
            iteration: i * 100,
            loss_total: 1.0 - (i as f32 * 0.2),
            loss_l1: 0.5,
            loss_ssim: 0.3,
            loss_lpips: Some(0.1),
            loss_reg: 0.1,
            num_gaussians: 50000,
            lr_position: 0.00016,
            lr_scaling: 0.005,
            lr_rotation: 0.001,
            memory_mb: Some(4096),
            elapsed_seconds: i as f32 * 30.0,
        };

        let result = writer.write_metrics(&metrics);
        assert!(
            result.is_ok(),
            "Failed to write metrics at iteration {}: {:?}",
            i,
            result.err()
        );
    }

    // Drop the writer to ensure flush
    drop(writer);

    // Verify we have 3 JSON lines
    let content = fs::read_to_string(&path).expect("Failed to read metrics file");
    let lines: Vec<&str> = content.lines().collect();
    assert_eq!(lines.len(), 3, "Expected 3 JSON lines, got {}", lines.len());

    // Verify each line is valid JSON
    for (idx, line) in lines.iter().enumerate() {
        let parsed: Result<serde_json::Value, _> = serde_json::from_str(line);
        assert!(
            parsed.is_ok(),
            "Failed to parse JSON at line {}: {:?}",
            idx,
            parsed.err()
        );

        let json = parsed.expect("JSON parsing failed");
        assert_eq!(
            json["iteration"].as_u64().expect("iteration not a number"),
            (idx * 100) as u64,
            "Iteration incorrect at line {}",
            idx
        );
    }

    cleanup_test_file(&path);
}

#[test]
#[serial]
fn test_manual_flush() {
    let path = temp_metrics_path("csv_flush.csv");
    cleanup_test_file(&path);

    let mut writer =
        MetricsWriter::new(&path, MetricsFormat::Csv).expect("Failed to create CSV writer");

    let metrics = TrainingMetrics {
        iteration: 1,
        loss_total: 1.234,
        loss_l1: 0.5,
        loss_ssim: 0.3,
        loss_lpips: Some(0.1),
        loss_reg: 0.334,
        num_gaussians: 50000,
        lr_position: 0.00016,
        lr_scaling: 0.005,
        lr_rotation: 0.001,
        memory_mb: Some(4096),
        elapsed_seconds: 120.5,
    };

    writer
        .write_metrics(&metrics)
        .expect("Failed to write metrics");

    // Manual flush should not fail
    let result = writer.flush();
    assert!(result.is_ok(), "Manual flush failed: {:?}", result.err());

    drop(writer);
    cleanup_test_file(&path);
}

#[test]
#[serial]
fn test_optional_fields() {
    let path = temp_metrics_path("csv_optional.csv");
    cleanup_test_file(&path);

    let mut writer =
        MetricsWriter::new(&path, MetricsFormat::Csv).expect("Failed to create CSV writer");

    // Test with optional fields set to None
    let metrics = TrainingMetrics {
        iteration: 1,
        loss_total: 1.0,
        loss_l1: 0.5,
        loss_ssim: 0.3,
        loss_lpips: None, // Optional
        loss_reg: 0.2,
        num_gaussians: 50000,
        lr_position: 0.00016,
        lr_scaling: 0.005,
        lr_rotation: 0.001,
        memory_mb: None, // Optional
        elapsed_seconds: 120.0,
    };

    let result = writer.write_metrics(&metrics);
    assert!(
        result.is_ok(),
        "Failed to write metrics with None fields: {:?}",
        result.err()
    );

    drop(writer);

    // Verify the file was written with 0 for optional fields
    let content = fs::read_to_string(&path).expect("Failed to read metrics file");
    let lines: Vec<&str> = content.lines().collect();
    assert_eq!(lines.len(), 2, "Expected header + 1 data line");

    // Check that None values are written as 0
    let data_line = lines[1];
    let fields: Vec<&str> = data_line.split(',').collect();

    // loss_lpips should be 0 (field index 4)
    assert_eq!(fields[4], "0", "loss_lpips should be 0 when None");

    // memory_mb should be 0 (field index 10)
    assert_eq!(fields[10], "0", "memory_mb should be 0 when None");

    cleanup_test_file(&path);
}

#[test]
#[serial]
fn test_file_creation_error() {
    // Try to create a metrics file in a non-existent directory
    let path = PathBuf::from("/nonexistent_directory_12345/metrics.csv");

    let result = MetricsWriter::new(&path, MetricsFormat::Csv);
    assert!(
        result.is_err(),
        "Should fail to create writer in non-existent directory"
    );
}
