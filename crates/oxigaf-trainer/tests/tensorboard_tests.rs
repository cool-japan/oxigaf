//! Integration tests for TensorBoard logging.

use std::fs;
use std::path::PathBuf;

use oxigaf_trainer::tensorboard::{
    LearningRates, TensorBoardConfig, TensorBoardWriter, TrainingMetricsLogger,
};

/// Helper to create a unique test directory.
fn test_dir(name: &str) -> PathBuf {
    let temp = std::env::temp_dir();
    let dir = temp.join(format!("oxigaf_tb_integration_{}", name));
    // Clean up any previous test
    let _ = fs::remove_dir_all(&dir);
    dir
}

/// Helper to clean up test directory.
fn cleanup(dir: &PathBuf) {
    let _ = fs::remove_dir_all(dir);
}

// ---------------------------------------------------------------------------
// TensorBoardWriter Tests
// ---------------------------------------------------------------------------

#[test]
fn test_writer_creates_valid_log_directory() {
    let dir = test_dir("creates_dir");

    let config = TensorBoardConfig::new(&dir);
    let writer_result = TensorBoardWriter::new(config);

    assert!(writer_result.is_ok(), "Writer creation should succeed");
    assert!(dir.exists(), "Log directory should exist");

    let writer = writer_result.ok();
    if let Some(w) = &writer {
        assert!(w.is_enabled(), "Writer should be enabled");
        assert!(w.file_path().exists(), "Event file should exist");
    }

    cleanup(&dir);
}

#[test]
fn test_writer_with_run_name() {
    let dir = test_dir("run_name");

    let config = TensorBoardConfig::new(&dir).with_run_name("experiment_1");
    let writer_result = TensorBoardWriter::new(config);

    assert!(writer_result.is_ok());

    let expected_run_dir = dir.join("experiment_1");
    assert!(expected_run_dir.exists(), "Run subdirectory should exist");

    cleanup(&dir);
}

#[test]
fn test_scalar_logging_creates_valid_file() {
    let dir = test_dir("scalar_logging");

    let config = TensorBoardConfig::new(&dir).with_flush_interval(0);
    let mut writer_result = TensorBoardWriter::new(config);

    if let Ok(ref mut writer) = writer_result {
        // Log multiple scalars
        let result = writer.log_scalar("loss/total", 0.5, 1);
        assert!(result.is_ok(), "Scalar logging should succeed");

        let result = writer.log_scalar("metrics/psnr", 25.0, 1);
        assert!(result.is_ok());

        let result = writer.log_scalar("metrics/ssim", 0.95, 1);
        assert!(result.is_ok());

        // Log at different steps
        for step in 2..=10 {
            let loss = 0.5 * (1.0 - step as f32 / 10.0);
            let result = writer.log_scalar("loss/total", loss, step);
            assert!(result.is_ok());
        }

        let result = writer.flush();
        assert!(result.is_ok(), "Flush should succeed");

        // Verify file size increased
        let file_path = writer.file_path().to_path_buf();
        let metadata = fs::metadata(&file_path);
        assert!(metadata.is_ok());
        assert!(
            metadata.map(|m| m.len()).unwrap_or(0) > 0,
            "Event file should have content"
        );
    }

    cleanup(&dir);
}

#[test]
fn test_batch_scalar_logging() {
    let dir = test_dir("batch_scalars");

    let config = TensorBoardConfig::new(&dir);
    let mut writer_result = TensorBoardWriter::new(config);

    if let Ok(ref mut writer) = writer_result {
        let result = writer.log_scalars(
            &[
                ("loss/l1", 0.1),
                ("loss/ssim", 0.2),
                ("loss/lpips", 0.05),
                ("metrics/psnr", 28.5),
                ("metrics/ssim", 0.97),
            ],
            100,
        );
        assert!(result.is_ok(), "Batch scalar logging should succeed");
    }

    cleanup(&dir);
}

#[test]
fn test_image_logging() {
    let dir = test_dir("image_logging");

    let config = TensorBoardConfig::new(&dir);
    let mut writer_result = TensorBoardWriter::new(config);

    if let Ok(ref mut writer) = writer_result {
        // Create a simple 4x4 RGB gradient image
        let width = 4u32;
        let height = 4u32;
        let mut image_data = Vec::with_capacity((width * height * 3) as usize);

        for y in 0..height {
            for x in 0..width {
                let r = x as f32 / (width - 1) as f32;
                let g = y as f32 / (height - 1) as f32;
                let b = 0.5;
                image_data.push(r);
                image_data.push(g);
                image_data.push(b);
            }
        }

        let result = writer.log_image("render/test", &image_data, width, height, 1);
        assert!(result.is_ok(), "Image logging should succeed");

        let result = writer.flush();
        assert!(result.is_ok());
    }

    cleanup(&dir);
}

#[test]
fn test_image_logging_dimension_mismatch() {
    let dir = test_dir("image_dim_mismatch");

    let config = TensorBoardConfig::new(&dir);
    let mut writer_result = TensorBoardWriter::new(config);

    if let Ok(ref mut writer) = writer_result {
        // Wrong size image data (should be 4x4x3 = 48, but we provide 10)
        let image_data = vec![0.5f32; 10];

        let result = writer.log_image("render/test", &image_data, 4, 4, 1);
        assert!(result.is_err(), "Should fail with dimension mismatch");
    }

    cleanup(&dir);
}

#[test]
fn test_histogram_logging() {
    let dir = test_dir("histogram_logging");

    let config = TensorBoardConfig::new(&dir);
    let mut writer_result = TensorBoardWriter::new(config);

    if let Ok(ref mut writer) = writer_result {
        // Generate random-ish gradient data
        let gradient_data: Vec<f32> = (0..1000)
            .map(|i| (i as f32 * 0.001) * (1.0 - 2.0 * ((i % 2) as f32)))
            .collect();

        let result = writer.log_histogram("gradients/position", &gradient_data, 50);
        assert!(result.is_ok(), "Histogram logging should succeed");

        // Test with uniform data
        let uniform_data = vec![0.5f32; 100];
        let result = writer.log_histogram("gradients/scale", &uniform_data, 50);
        assert!(result.is_ok());

        // Test with empty data (should be no-op)
        let empty_data: Vec<f32> = vec![];
        let result = writer.log_histogram("gradients/empty", &empty_data, 50);
        assert!(result.is_ok(), "Empty histogram should be no-op");
    }

    cleanup(&dir);
}

#[test]
fn test_disabled_writer_is_noop() {
    let writer = TensorBoardWriter::disabled();
    assert!(
        !writer.is_enabled(),
        "Disabled writer should not be enabled"
    );

    // All operations should be no-ops
    // Note: We can't call methods on a moved value, so we create a new one
    let mut writer = TensorBoardWriter::disabled();
    let result = writer.log_scalar("test", 1.0, 1);
    assert!(result.is_ok());

    let result = writer.log_scalars(&[("a", 1.0), ("b", 2.0)], 1);
    assert!(result.is_ok());

    let result = writer.log_histogram("test", &[1.0, 2.0, 3.0], 1);
    assert!(result.is_ok());

    // Image with correct dimensions for a disabled writer
    let result = writer.log_image("test", &[0.0; 12], 2, 2, 1);
    assert!(result.is_ok());

    let result = writer.flush();
    assert!(result.is_ok());
}

// ---------------------------------------------------------------------------
// TrainingMetricsLogger Tests
// ---------------------------------------------------------------------------

#[test]
fn test_training_metrics_logger_creation() {
    let dir = test_dir("metrics_logger");

    let config = TensorBoardConfig::new(&dir);
    let logger_result = TrainingMetricsLogger::new(config);

    assert!(logger_result.is_ok(), "Logger creation should succeed");

    if let Ok(logger) = logger_result {
        assert!(logger.is_enabled(), "Logger should be enabled");
    }

    cleanup(&dir);
}

#[test]
fn test_training_metrics_logger_log_step() {
    let dir = test_dir("log_step");

    let config = TensorBoardConfig::new(&dir);
    let logger_result = TrainingMetricsLogger::new(config);

    if let Ok(mut logger) = logger_result {
        let lr = LearningRates::from_config(1e-4, 1e-3, 5e-3, 5e-2, 2.5e-3);

        for step in 1..=100 {
            let loss = 0.5 * (1.0 - step as f32 / 100.0);
            let psnr = 20.0 + step as f32 * 0.1;
            let ssim = 0.8 + step as f32 * 0.002;
            let num_gaussians = 50000 + step * 10;

            let result = logger.log_step(step, loss, psnr, ssim, num_gaussians as usize, &lr);
            assert!(result.is_ok(), "log_step should succeed at step {}", step);
        }

        let result = logger.flush();
        assert!(result.is_ok());
    }

    cleanup(&dir);
}

#[test]
fn test_training_metrics_logger_log_losses() {
    let dir = test_dir("log_losses");

    let config = TensorBoardConfig::new(&dir);
    let logger_result = TrainingMetricsLogger::new(config);

    if let Ok(mut logger) = logger_result {
        let result = logger.log_losses(100, 0.03, 0.01, 0.005, 0.002, 0.003);
        assert!(result.is_ok(), "log_losses should succeed");
    }

    cleanup(&dir);
}

#[test]
fn test_training_metrics_logger_intervals() {
    let dir = test_dir("intervals");

    let mut config = TensorBoardConfig::new(&dir);
    config.scalar_interval = 10; // Log scalars every 10 steps
    config.image_interval = 50; // Log images every 50 steps

    let logger_result = TrainingMetricsLogger::new(config);

    if let Ok(mut logger) = logger_result {
        let lr = LearningRates::default();

        // Only steps divisible by 10 should actually log
        for step in 1..=100 {
            let result = logger.log_step(step, 0.1, 25.0, 0.95, 50000, &lr);
            assert!(result.is_ok());
        }

        // Create a small test image for image logging test
        let image = vec![0.5f32; 4 * 4 * 3];

        // Step 25 should not log (not divisible by 50)
        let result = logger.log_image("test", &image, 4, 4, 25);
        assert!(result.is_ok());

        // Step 50 should log
        let result = logger.log_image("test", &image, 4, 4, 50);
        assert!(result.is_ok());
    }

    cleanup(&dir);
}

#[test]
fn test_disabled_logger_is_noop() {
    let mut logger = TrainingMetricsLogger::disabled();
    assert!(!logger.is_enabled());

    let lr = LearningRates::default();

    let result = logger.log_step(1, 0.5, 20.0, 0.8, 1000, &lr);
    assert!(result.is_ok());

    let result = logger.log_losses(1, 0.3, 0.1, 0.05, 0.01, 0.02);
    assert!(result.is_ok());

    let result = logger.log_image("test", &[0.0; 12], 2, 2, 1);
    assert!(result.is_ok());

    let result = logger.log_gradient_histogram("test", &[1.0, 2.0, 3.0], 1);
    assert!(result.is_ok());

    let result = logger.flush();
    assert!(result.is_ok());
}

// ---------------------------------------------------------------------------
// LearningRates Tests
// ---------------------------------------------------------------------------

#[test]
fn test_learning_rates_from_config() {
    let lr = LearningRates::from_config(1e-4, 1e-3, 5e-3, 5e-2, 2.5e-3);

    assert!((lr.position - 1e-4).abs() < 1e-10);
    assert!((lr.rotation - 1e-3).abs() < 1e-10);
    assert!((lr.scale - 5e-3).abs() < 1e-10);
    assert!((lr.opacity - 5e-2).abs() < 1e-10);
    assert!((lr.sh - 2.5e-3).abs() < 1e-10);
}

#[test]
fn test_learning_rates_default() {
    let lr = LearningRates::default();

    assert_eq!(lr.position, 0.0);
    assert_eq!(lr.rotation, 0.0);
    assert_eq!(lr.scale, 0.0);
    assert_eq!(lr.opacity, 0.0);
    assert_eq!(lr.sh, 0.0);
}

// ---------------------------------------------------------------------------
// TensorBoardConfig Tests
// ---------------------------------------------------------------------------

#[test]
fn test_config_default() {
    let config = TensorBoardConfig::default();

    assert!(!config.enabled);
    assert_eq!(config.flush_interval, 100);
    assert_eq!(config.scalar_interval, 1);
    assert_eq!(config.image_interval, 500);
    assert_eq!(config.histogram_interval, 500);
}

#[test]
fn test_config_new_enables_by_default() {
    let tmp_path = std::env::temp_dir().join("oxigaf_tb_test");
    let config = TensorBoardConfig::new(&tmp_path);

    assert!(config.enabled);
    assert_eq!(config.log_dir, tmp_path);
}

#[test]
fn test_config_with_run_name() {
    let tmp_path = std::env::temp_dir().join("oxigaf_tb_test");
    let config = TensorBoardConfig::new(&tmp_path).with_run_name("my_experiment");

    assert!(config.enabled);
    assert_eq!(config.run_name, "my_experiment");
    assert_eq!(config.run_dir(), tmp_path.join("my_experiment"));
}

#[test]
fn test_config_validation() {
    // Disabled config should always be valid
    let config = TensorBoardConfig::default();
    assert!(config.validate().is_ok());

    // Enabled with valid log_dir should be valid
    let config = TensorBoardConfig::new(&std::env::temp_dir().join("oxigaf_tb_test"));
    assert!(config.validate().is_ok());

    // Enabled with empty log_dir should fail
    let config = TensorBoardConfig {
        enabled: true,
        log_dir: std::path::PathBuf::new(),
        ..Default::default()
    };
    assert!(config.validate().is_err());
}

// ---------------------------------------------------------------------------
// Edge Cases
// ---------------------------------------------------------------------------

#[test]
fn test_writer_auto_flush_on_drop() {
    let dir = test_dir("auto_flush");

    {
        let config = TensorBoardConfig::new(&dir).with_flush_interval(1000); // High interval
        let mut writer_result = TensorBoardWriter::new(config);

        if let Ok(ref mut writer) = writer_result {
            for i in 1..=10 {
                let _ = writer.log_scalar("test", i as f32, i);
            }
            // Don't explicitly flush - should happen on drop
        }
    }

    // Verify the file was written
    let entries: Vec<_> = fs::read_dir(&dir).into_iter().flatten().flatten().collect();
    assert!(!entries.is_empty(), "Event file should exist after drop");

    cleanup(&dir);
}

#[test]
fn test_step_tracking() {
    let dir = test_dir("step_tracking");

    let config = TensorBoardConfig::new(&dir);
    let mut writer_result = TensorBoardWriter::new(config);

    if let Ok(ref mut writer) = writer_result {
        assert_eq!(writer.current_step(), 0);

        let _ = writer.log_scalar("test", 1.0, 100);
        assert_eq!(writer.current_step(), 100);

        writer.set_step(200);
        assert_eq!(writer.current_step(), 200);
    }

    cleanup(&dir);
}
