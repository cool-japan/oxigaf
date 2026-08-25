//! Pure Rust TensorBoard event logging.
//!
//! Provides a lightweight TFEvents writer for logging training metrics
//! to TensorBoard without external dependencies.
//!
//! ## TFEvents Format
//!
//! Each record in a TFEvents file consists of:
//! 1. 8 bytes: `uint64` little-endian length of data
//! 2. 4 bytes: masked CRC32C of length bytes
//! 3. N bytes: serialized Event protobuf
//! 4. 4 bytes: masked CRC32C of data bytes
//!
//! ## Usage
//!
//! ```ignore
//! use oxigaf_trainer::tensorboard::{TensorBoardWriter, TensorBoardConfig};
//!
//! let config = TensorBoardConfig::new("/tmp/logs");
//! let mut writer = TensorBoardWriter::new(config)?;
//! writer.log_scalar("loss", 0.5, 100)?;
//! writer.log_scalar("psnr", 25.0, 100)?;
//! writer.flush()?;
//! ```

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::TrainerError;

// ---------------------------------------------------------------------------
// CRC32C Implementation (Pure Rust)
// ---------------------------------------------------------------------------

/// CRC32C polynomial (Castagnoli).
const CRC32C_POLY: u32 = 0x82F6_3B78;

/// Precomputed CRC32C lookup table.
fn crc32c_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    for (i, entry) in table.iter_mut().enumerate() {
        let mut crc = i as u32;
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ CRC32C_POLY;
            } else {
                crc >>= 1;
            }
        }
        *entry = crc;
    }
    table
}

/// Compute CRC32C checksum of data.
///
/// The 256-entry lookup table is built once (2048 shift/xor iterations) and
/// cached process-wide, rather than being regenerated on every call —
/// `write_record` calls this twice per TFEvents record, and a training run
/// logs at least two records per step.
fn crc32c(data: &[u8]) -> u32 {
    static TABLE: OnceLock<[u32; 256]> = OnceLock::new();
    let table = TABLE.get_or_init(crc32c_table);

    let mut crc = !0u32;
    for byte in data {
        let idx = ((crc ^ (*byte as u32)) & 0xFF) as usize;
        crc = (crc >> 8) ^ table[idx];
    }
    !crc
}

/// Mask CRC32C value as per TensorBoard spec.
///
/// `masked = ((crc >> 15) | (crc << 17)) + 0xa282ead8`
fn mask_crc(crc: u32) -> u32 {
    crc.rotate_right(15).wrapping_add(0xa282_ead8)
}

// ---------------------------------------------------------------------------
// Protobuf Encoding Helpers (minimal, hand-rolled)
// ---------------------------------------------------------------------------

/// Protobuf wire types.
#[derive(Clone, Copy)]
#[repr(u8)]
enum WireType {
    Varint = 0,
    Fixed64 = 1,
    LengthDelimited = 2,
    Fixed32 = 5,
}

/// Encode a varint to a buffer.
fn encode_varint(value: u64, buf: &mut Vec<u8>) {
    let mut v = value;
    loop {
        let byte = (v & 0x7F) as u8;
        v >>= 7;
        if v == 0 {
            buf.push(byte);
            break;
        } else {
            buf.push(byte | 0x80);
        }
    }
}

/// Encode a field tag (field number + wire type).
fn encode_tag(field_number: u32, wire_type: WireType, buf: &mut Vec<u8>) {
    let tag = (field_number << 3) | (wire_type as u32);
    encode_varint(tag as u64, buf);
}

/// Encode a double (fixed64).
fn encode_double(field_number: u32, value: f64, buf: &mut Vec<u8>) {
    encode_tag(field_number, WireType::Fixed64, buf);
    buf.extend_from_slice(&value.to_le_bytes());
}

/// Encode a float (fixed32).
fn encode_float(field_number: u32, value: f32, buf: &mut Vec<u8>) {
    encode_tag(field_number, WireType::Fixed32, buf);
    buf.extend_from_slice(&value.to_le_bytes());
}

/// Encode a string field.
fn encode_string(field_number: u32, value: &str, buf: &mut Vec<u8>) {
    encode_tag(field_number, WireType::LengthDelimited, buf);
    encode_varint(value.len() as u64, buf);
    buf.extend_from_slice(value.as_bytes());
}

/// Encode bytes field.
fn encode_bytes(field_number: u32, value: &[u8], buf: &mut Vec<u8>) {
    encode_tag(field_number, WireType::LengthDelimited, buf);
    encode_varint(value.len() as u64, buf);
    buf.extend_from_slice(value);
}

/// Encode an int64 varint field.
fn encode_int64(field_number: u32, value: i64, buf: &mut Vec<u8>) {
    encode_tag(field_number, WireType::Varint, buf);
    encode_varint(value as u64, buf);
}

/// Encode a submessage.
fn encode_submessage(field_number: u32, submessage: &[u8], buf: &mut Vec<u8>) {
    encode_tag(field_number, WireType::LengthDelimited, buf);
    encode_varint(submessage.len() as u64, buf);
    buf.extend_from_slice(submessage);
}

// ---------------------------------------------------------------------------
// TensorBoard Protobuf Messages
// ---------------------------------------------------------------------------

/// Build a Summary.Value protobuf for a scalar.
///
/// ```protobuf
/// message Value {
///   string tag = 1;
///   oneof value {
///     float simple_value = 2;
///   }
/// }
/// ```
fn build_scalar_value(tag: &str, value: f32) -> Vec<u8> {
    let mut buf = Vec::new();
    encode_string(1, tag, &mut buf); // tag field
    encode_float(2, value, &mut buf); // simple_value field
    buf
}

/// Build a Summary.Value protobuf for an image.
///
/// ```protobuf
/// message Image {
///   int32 height = 1;
///   int32 width = 2;
///   int32 colorspace = 3;
///   bytes encoded_image_string = 4;
/// }
/// message Value {
///   string tag = 1;
///   Image image = 4;
/// }
/// ```
fn build_image_value(tag: &str, width: u32, height: u32, png_data: &[u8]) -> Vec<u8> {
    // Build Image submessage
    let mut image_buf = Vec::new();
    encode_int64(1, height as i64, &mut image_buf); // height
    encode_int64(2, width as i64, &mut image_buf); // width
    encode_int64(3, 3, &mut image_buf); // colorspace = 3 (RGB)
    encode_bytes(4, png_data, &mut image_buf); // encoded_image_string

    // Build Value message
    let mut buf = Vec::new();
    encode_string(1, tag, &mut buf); // tag field
    encode_submessage(4, &image_buf, &mut buf); // image field
    buf
}

/// Build a Summary.Value protobuf for a histogram.
///
/// ```protobuf
/// message HistogramProto {
///   double min = 1;
///   double max = 2;
///   double num = 3;
///   double sum = 4;
///   double sum_squares = 5;
///   repeated double bucket_limit = 6 [packed = true];
///   repeated double bucket = 7 [packed = true];
/// }
/// message Value {
///   string tag = 1;
///   HistogramProto histo = 5;
/// }
/// ```
#[allow(clippy::too_many_arguments)]
fn build_histogram_value(
    tag: &str,
    min: f64,
    max: f64,
    count: f64,
    sum: f64,
    sum_squares: f64,
    bucket_limits: &[f64],
    bucket_counts: &[f64],
) -> Vec<u8> {
    // Build HistogramProto submessage
    let mut histo_buf = Vec::new();
    encode_double(1, min, &mut histo_buf);
    encode_double(2, max, &mut histo_buf);
    encode_double(3, count, &mut histo_buf);
    encode_double(4, sum, &mut histo_buf);
    encode_double(5, sum_squares, &mut histo_buf);

    // Encode packed bucket_limit (field 6)
    if !bucket_limits.is_empty() {
        let mut packed_limits = Vec::new();
        for &limit in bucket_limits {
            packed_limits.extend_from_slice(&limit.to_le_bytes());
        }
        encode_bytes(6, &packed_limits, &mut histo_buf);
    }

    // Encode packed bucket (field 7)
    if !bucket_counts.is_empty() {
        let mut packed_counts = Vec::new();
        for &count in bucket_counts {
            packed_counts.extend_from_slice(&count.to_le_bytes());
        }
        encode_bytes(7, &packed_counts, &mut histo_buf);
    }

    // Build Value message
    let mut buf = Vec::new();
    encode_string(1, tag, &mut buf); // tag field
    encode_submessage(5, &histo_buf, &mut buf); // histo field
    buf
}

/// Build a Summary protobuf containing multiple values.
///
/// ```protobuf
/// message Summary {
///   repeated Value value = 1;
/// }
/// ```
fn build_summary(values: &[Vec<u8>]) -> Vec<u8> {
    let mut buf = Vec::new();
    for value in values {
        encode_submessage(1, value, &mut buf);
    }
    buf
}

/// Build an Event protobuf.
///
/// ```protobuf
/// message Event {
///   double wall_time = 1;
///   int64 step = 2;
///   oneof what {
///     string file_version = 3;
///     Summary summary = 5;
///   }
/// }
/// ```
fn build_event(
    wall_time: f64,
    step: i64,
    summary: Option<&[u8]>,
    file_version: Option<&str>,
) -> Vec<u8> {
    let mut buf = Vec::new();
    encode_double(1, wall_time, &mut buf); // wall_time
    encode_int64(2, step, &mut buf); // step

    if let Some(version) = file_version {
        encode_string(3, version, &mut buf); // file_version
    }

    if let Some(summary_data) = summary {
        encode_submessage(5, summary_data, &mut buf); // summary
    }

    buf
}

// ---------------------------------------------------------------------------
// TFEvents Record Writer
// ---------------------------------------------------------------------------

/// Write a single TFEvents record to a writer.
fn write_record<W: Write>(writer: &mut W, data: &[u8]) -> Result<(), TrainerError> {
    let len = data.len() as u64;
    let len_bytes = len.to_le_bytes();

    // Length + masked CRC of length
    let len_crc = mask_crc(crc32c(&len_bytes));
    writer.write_all(&len_bytes)?;
    writer.write_all(&len_crc.to_le_bytes())?;

    // Data + masked CRC of data
    let data_crc = mask_crc(crc32c(data));
    writer.write_all(data)?;
    writer.write_all(&data_crc.to_le_bytes())?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Histogram bucketing
// ---------------------------------------------------------------------------

/// TensorBoard's canonical default histogram bucket edges: fixed, symmetric,
/// geometrically-spaced limits covering the full range of `f64` magnitudes
/// (mirrors `tensorflow::histogram::Histogram`'s default bucketer in
/// `tensorflow/core/lib/histogram/histogram.cc`). Computed once and cached
/// process-wide, since the edges never depend on the data being
/// histogrammed.
///
/// Layout (sorted ascending): `[-f64::MAX, ..., -v, ..., -1e-12, 1e-12,
/// ..., v, ..., f64::MAX]` where each `v` steps `*= 1.1` starting from
/// `1e-12` while `v < 1e20`. Every finite value falls into exactly one
/// bucket on the correct side of zero — including data with `min < 0 <
/// max`, which the old `ln|min|..ln|max|` scheme (negated only when `max <
/// 0`) put almost entirely into bucket 0.
fn default_bucket_limits() -> &'static [f64] {
    static LIMITS: OnceLock<Vec<f64>> = OnceLock::new();
    LIMITS
        .get_or_init(|| {
            let mut positive = Vec::new();
            let mut v = 1.0e-12_f64;
            while v < 1.0e20 {
                positive.push(v);
                v *= 1.1;
            }
            positive.push(f64::MAX);

            let mut limits = Vec::with_capacity(positive.len() * 2);
            limits.extend(positive.iter().rev().map(|&p| -p)); // -MAX .. -1e-12
            limits.extend(positive.iter().copied()); //            1e-12 .. MAX
            limits
        })
        .as_slice()
}

// ---------------------------------------------------------------------------
// TensorBoardConfig
// ---------------------------------------------------------------------------

/// Configuration for TensorBoard logging.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorBoardConfig {
    /// Directory where TensorBoard event files are written.
    pub log_dir: PathBuf,
    /// Prefix for the run name (subdirectory under log_dir).
    pub run_name: String,
    /// Interval for flushing to disk (in steps). 0 = manual flush only.
    pub flush_interval: u32,
    /// Whether TensorBoard logging is enabled.
    pub enabled: bool,
    /// Log scalars every N steps (0 = every step when logging).
    pub scalar_interval: u32,
    /// Log images every N steps (0 = disabled).
    pub image_interval: u32,
    /// Log histograms every N steps (0 = disabled).
    pub histogram_interval: u32,
}

impl Default for TensorBoardConfig {
    fn default() -> Self {
        Self {
            log_dir: PathBuf::from("runs"),
            run_name: String::new(), // Will be auto-generated if empty
            flush_interval: 100,
            enabled: false,
            scalar_interval: 1,
            image_interval: 500,
            histogram_interval: 500,
        }
    }
}

impl TensorBoardConfig {
    /// Create a new TensorBoard configuration with the specified log directory.
    pub fn new<P: AsRef<Path>>(log_dir: P) -> Self {
        Self {
            log_dir: log_dir.as_ref().to_path_buf(),
            enabled: true,
            ..Default::default()
        }
    }

    /// Enable TensorBoard logging with a specific run name.
    pub fn with_run_name(mut self, name: impl Into<String>) -> Self {
        self.run_name = name.into();
        self.enabled = true;
        self
    }

    /// Set the flush interval.
    pub fn with_flush_interval(mut self, interval: u32) -> Self {
        self.flush_interval = interval;
        self
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), TrainerError> {
        if self.enabled && self.log_dir.as_os_str().is_empty() {
            return Err(TrainerError::InvalidConfig(
                "TensorBoard log_dir cannot be empty when enabled".into(),
            ));
        }
        Ok(())
    }

    /// Get the full path to the run directory.
    pub fn run_dir(&self) -> PathBuf {
        if self.run_name.is_empty() {
            self.log_dir.clone()
        } else {
            self.log_dir.join(&self.run_name)
        }
    }
}

// ---------------------------------------------------------------------------
// TensorBoardWriter
// ---------------------------------------------------------------------------

/// TensorBoard event file writer.
///
/// Writes TFEvents files compatible with TensorBoard visualization.
/// Supports logging scalars, images, and histograms with automatic step tracking.
pub struct TensorBoardWriter {
    config: TensorBoardConfig,
    writer: Option<BufWriter<File>>,
    file_path: PathBuf,
    current_step: i64,
    steps_since_flush: u32,
}

impl TensorBoardWriter {
    /// Create a new TensorBoard writer with the given configuration.
    ///
    /// Creates the log directory if it doesn't exist and writes the
    /// initial file version event.
    pub fn new(config: TensorBoardConfig) -> Result<Self, TrainerError> {
        if !config.enabled {
            return Ok(Self {
                config,
                writer: None,
                file_path: PathBuf::new(),
                current_step: 0,
                steps_since_flush: 0,
            });
        }

        config.validate()?;

        let run_dir = config.run_dir();
        fs::create_dir_all(&run_dir)?;

        // Generate unique filename with timestamp
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let hostname = std::env::var("HOSTNAME")
            .or_else(|_| std::env::var("HOST"))
            .unwrap_or_else(|_| "localhost".to_string());

        let filename = format!("events.out.tfevents.{}.{}", timestamp, hostname);
        let file_path = run_dir.join(&filename);

        let file = File::create(&file_path)?;
        let mut writer = BufWriter::new(file);

        // Write file version event
        let wall_time = Self::wall_time();
        let event = build_event(wall_time, 0, None, Some("brain.Event:2"));
        write_record(&mut writer, &event)?;

        tracing::info!("TensorBoard writer initialized: {:?}", file_path);

        Ok(Self {
            config,
            writer: Some(writer),
            file_path,
            current_step: 0,
            steps_since_flush: 0,
        })
    }

    /// Create a disabled writer (no-op for all operations).
    pub fn disabled() -> Self {
        Self {
            config: TensorBoardConfig::default(),
            writer: None,
            file_path: PathBuf::new(),
            current_step: 0,
            steps_since_flush: 0,
        }
    }

    /// Check if the writer is enabled.
    pub fn is_enabled(&self) -> bool {
        self.config.enabled && self.writer.is_some()
    }

    /// Get the path to the event file.
    pub fn file_path(&self) -> &Path {
        &self.file_path
    }

    /// Get the current step.
    pub fn current_step(&self) -> i64 {
        self.current_step
    }

    /// Set the current step for subsequent logging.
    pub fn set_step(&mut self, step: i64) {
        self.current_step = step;
    }

    /// Get the current wall time as seconds since epoch.
    fn wall_time() -> f64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0)
    }

    /// Log a scalar value at the current step.
    pub fn log_scalar(&mut self, tag: &str, value: f32, step: i64) -> Result<(), TrainerError> {
        let writer = match self.writer.as_mut() {
            Some(w) => w,
            None => return Ok(()), // Disabled, no-op
        };

        self.current_step = step;

        let scalar_value = build_scalar_value(tag, value);
        let summary = build_summary(&[scalar_value]);
        let event = build_event(Self::wall_time(), step, Some(&summary), None);
        write_record(writer, &event)?;

        self.steps_since_flush += 1;
        self.maybe_auto_flush()?;

        Ok(())
    }

    /// Log multiple scalar values at the current step (more efficient).
    pub fn log_scalars(&mut self, values: &[(&str, f32)], step: i64) -> Result<(), TrainerError> {
        let writer = match self.writer.as_mut() {
            Some(w) => w,
            None => return Ok(()), // Disabled, no-op
        };

        self.current_step = step;

        let scalar_values: Vec<Vec<u8>> = values
            .iter()
            .map(|(tag, value)| build_scalar_value(tag, *value))
            .collect();

        let summary = build_summary(&scalar_values);
        let event = build_event(Self::wall_time(), step, Some(&summary), None);
        write_record(writer, &event)?;

        self.steps_since_flush += 1;
        self.maybe_auto_flush()?;

        Ok(())
    }

    /// Log an image at the current step.
    ///
    /// The image data should be in RGB format (HWC, values 0.0-1.0).
    pub fn log_image(
        &mut self,
        tag: &str,
        image_data: &[f32],
        width: u32,
        height: u32,
        step: i64,
    ) -> Result<(), TrainerError> {
        // Check if disabled first
        if self.writer.is_none() {
            return Ok(()); // Disabled, no-op
        }

        self.current_step = step;

        // Convert f32 RGB to u8 and encode as PNG (before mutable borrow of writer)
        let png_data = self.encode_image_as_png(image_data, width, height)?;

        // Now get the writer for writing
        let writer = match self.writer.as_mut() {
            Some(w) => w,
            None => return Ok(()), // Should not happen, but handle anyway
        };

        let image_value = build_image_value(tag, width, height, &png_data);
        let summary = build_summary(&[image_value]);
        let event = build_event(Self::wall_time(), step, Some(&summary), None);
        write_record(writer, &event)?;

        self.steps_since_flush += 1;
        self.maybe_auto_flush()?;

        Ok(())
    }

    /// Encode image data as PNG.
    fn encode_image_as_png(
        &self,
        data: &[f32],
        width: u32,
        height: u32,
    ) -> Result<Vec<u8>, TrainerError> {
        let expected_len = (width * height * 3) as usize;
        if data.len() != expected_len {
            return Err(TrainerError::ImageDimensionMismatch {
                expected: expected_len,
                actual: data.len(),
            });
        }

        // Convert f32 [0, 1] to u8 [0, 255]
        let pixels: Vec<u8> = data
            .iter()
            .map(|&v| (v.clamp(0.0, 1.0) * 255.0) as u8)
            .collect();

        // Use the image crate to encode as PNG
        let img = image::RgbImage::from_raw(width, height, pixels).ok_or_else(|| {
            TrainerError::InvalidConfig("Failed to create image from pixel data".into())
        })?;

        let mut png_data = Vec::new();
        let mut cursor = std::io::Cursor::new(&mut png_data);
        img.write_to(&mut cursor, image::ImageFormat::Png)
            .map_err(|e| TrainerError::InvalidConfig(format!("PNG encoding failed: {}", e)))?;

        Ok(png_data)
    }

    /// Log a histogram at the current step.
    ///
    /// Automatically computes histogram statistics and bins from the data.
    pub fn log_histogram(
        &mut self,
        tag: &str,
        data: &[f32],
        step: i64,
    ) -> Result<(), TrainerError> {
        // Check if disabled first
        if self.writer.is_none() {
            return Ok(()); // Disabled, no-op
        }

        if data.is_empty() {
            return Ok(());
        }

        self.current_step = step;

        // Compute histogram statistics (before mutable borrow of writer)
        let (min, max, sum, sum_squares, bucket_limits, bucket_counts) =
            self.compute_histogram(data);

        // Now get the writer for writing
        let writer = match self.writer.as_mut() {
            Some(w) => w,
            None => return Ok(()), // Should not happen, but handle anyway
        };

        let histo_value = build_histogram_value(
            tag,
            min,
            max,
            data.len() as f64,
            sum,
            sum_squares,
            bucket_limits,
            &bucket_counts,
        );
        let summary = build_summary(&[histo_value]);
        let event = build_event(Self::wall_time(), step, Some(&summary), None);
        write_record(writer, &event)?;

        self.steps_since_flush += 1;
        self.maybe_auto_flush()?;

        Ok(())
    }

    /// Compute histogram bins and statistics.
    ///
    /// Bucket edges use the fixed, symmetric [`default_bucket_limits`]
    /// scheme rather than deriving them from this call's `min`/`max`: the
    /// previous scheme built limits from `ln|min|..ln|max|` and negated them
    /// only when `max < 0`, which put nearly all samples of mixed-sign data
    /// (the common case for gradients, `min < 0 < max`) into bucket 0. Using
    /// fixed edges also means `bucket_limits.len() == bucket_counts.len()`
    /// always holds, with no special-casing needed for constant-valued data.
    fn compute_histogram(&self, data: &[f32]) -> (f64, f64, f64, f64, &'static [f64], Vec<f64>) {
        let mut min = f64::MAX;
        let mut max = f64::MIN;
        let mut sum = 0.0f64;
        let mut sum_squares = 0.0f64;

        for &v in data {
            let vf = v as f64;
            if vf < min {
                min = vf;
            }
            if vf > max {
                max = vf;
            }
            sum += vf;
            sum_squares += vf * vf;
        }

        let bucket_limits = default_bucket_limits();
        let mut bucket_counts = vec![0.0f64; bucket_limits.len()];
        for &v in data {
            let vf = v as f64;
            // First bucket whose limit is >= vf (limits are sorted
            // ascending, so this is exactly `partition_point`'s contract).
            // `+inf` is the only finite-or-not value that can exceed even
            // the final `f64::MAX` limit, hence the clamp.
            let idx = bucket_limits
                .partition_point(|&limit| limit < vf)
                .min(bucket_limits.len() - 1);
            bucket_counts[idx] += 1.0;
        }

        (min, max, sum, sum_squares, bucket_limits, bucket_counts)
    }

    /// Flush pending writes to disk.
    pub fn flush(&mut self) -> Result<(), TrainerError> {
        if let Some(writer) = self.writer.as_mut() {
            writer.flush()?;
            self.steps_since_flush = 0;
        }
        Ok(())
    }

    /// Auto-flush if flush_interval steps have passed.
    fn maybe_auto_flush(&mut self) -> Result<(), TrainerError> {
        if self.config.flush_interval > 0 && self.steps_since_flush >= self.config.flush_interval {
            self.flush()?;
        }
        Ok(())
    }
}

impl Drop for TensorBoardWriter {
    fn drop(&mut self) {
        if let Err(e) = self.flush() {
            tracing::warn!("Failed to flush TensorBoard writer on drop: {}", e);
        }
    }
}

// ---------------------------------------------------------------------------
// TrainingMetricsLogger
// ---------------------------------------------------------------------------

/// High-level training metrics logger that wraps TensorBoardWriter.
///
/// Provides convenient methods for logging common training metrics
/// with automatic interval handling.
pub struct TrainingMetricsLogger {
    writer: TensorBoardWriter,
    config: TensorBoardConfig,
}

impl TrainingMetricsLogger {
    /// Create a new training metrics logger.
    pub fn new(config: TensorBoardConfig) -> Result<Self, TrainerError> {
        let writer = TensorBoardWriter::new(config.clone())?;
        Ok(Self { writer, config })
    }

    /// Create a disabled logger.
    pub fn disabled() -> Self {
        Self {
            writer: TensorBoardWriter::disabled(),
            config: TensorBoardConfig::default(),
        }
    }

    /// Check if logging is enabled.
    pub fn is_enabled(&self) -> bool {
        self.writer.is_enabled()
    }

    /// Log training step metrics.
    ///
    /// Logs all relevant metrics based on the configured intervals.
    pub fn log_step(
        &mut self,
        step: u32,
        loss: f32,
        psnr: f32,
        ssim: f32,
        num_gaussians: usize,
        learning_rates: &LearningRates,
    ) -> Result<(), TrainerError> {
        if !self.is_enabled() {
            return Ok(());
        }

        let step_i64 = step as i64;

        // Log scalars at scalar_interval
        if self.config.scalar_interval == 0 || step.is_multiple_of(self.config.scalar_interval) {
            self.writer.log_scalars(
                &[
                    ("loss/total", loss),
                    ("metrics/psnr", psnr),
                    ("metrics/ssim", ssim),
                    ("model/num_gaussians", num_gaussians as f32),
                    ("lr/position", learning_rates.position),
                    ("lr/rotation", learning_rates.rotation),
                    ("lr/scale", learning_rates.scale),
                    ("lr/opacity", learning_rates.opacity),
                    ("lr/sh", learning_rates.sh),
                ],
                step_i64,
            )?;
        }

        Ok(())
    }

    /// Log individual loss components.
    pub fn log_losses(
        &mut self,
        step: u32,
        l1_loss: f32,
        ssim_loss: f32,
        lpips_loss: f32,
        sds_loss: f32,
        regularization: f32,
    ) -> Result<(), TrainerError> {
        if !self.is_enabled() {
            return Ok(());
        }

        if self.config.scalar_interval == 0 || step.is_multiple_of(self.config.scalar_interval) {
            self.writer.log_scalars(
                &[
                    ("loss/l1", l1_loss),
                    ("loss/ssim", ssim_loss),
                    ("loss/lpips", lpips_loss),
                    ("loss/sds", sds_loss),
                    ("loss/regularization", regularization),
                ],
                step as i64,
            )?;
        }

        Ok(())
    }

    /// Log a rendered image.
    pub fn log_image(
        &mut self,
        tag: &str,
        image_data: &[f32],
        width: u32,
        height: u32,
        step: u32,
    ) -> Result<(), TrainerError> {
        if !self.is_enabled() {
            return Ok(());
        }

        if self.config.image_interval > 0 && step.is_multiple_of(self.config.image_interval) {
            self.writer
                .log_image(tag, image_data, width, height, step as i64)?;
        }

        Ok(())
    }

    /// Log a gradient histogram.
    pub fn log_gradient_histogram(
        &mut self,
        tag: &str,
        gradients: &[f32],
        step: u32,
    ) -> Result<(), TrainerError> {
        if !self.is_enabled() {
            return Ok(());
        }

        if self.config.histogram_interval > 0 && step.is_multiple_of(self.config.histogram_interval)
        {
            self.writer.log_histogram(tag, gradients, step as i64)?;
        }

        Ok(())
    }

    /// Flush pending writes to disk.
    pub fn flush(&mut self) -> Result<(), TrainerError> {
        self.writer.flush()
    }

    /// Get the underlying writer.
    pub fn writer(&self) -> &TensorBoardWriter {
        &self.writer
    }

    /// Get the underlying writer mutably.
    pub fn writer_mut(&mut self) -> &mut TensorBoardWriter {
        &mut self.writer
    }
}

// ---------------------------------------------------------------------------
// LearningRates
// ---------------------------------------------------------------------------

/// Current learning rates for logging.
#[derive(Debug, Clone, Copy, Default)]
pub struct LearningRates {
    pub position: f32,
    pub rotation: f32,
    pub scale: f32,
    pub opacity: f32,
    pub sh: f32,
}

impl LearningRates {
    /// Create learning rates from optimizer config values.
    pub fn from_config(position: f32, rotation: f32, scale: f32, opacity: f32, sh: f32) -> Self {
        Self {
            position,
            rotation,
            scale,
            opacity,
            sh,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crc32c_empty() {
        let crc = crc32c(&[]);
        assert_eq!(crc, 0);
    }

    #[test]
    fn test_crc32c_known() {
        // Test with known CRC32C values
        let data = b"123456789";
        let crc = crc32c(data);
        // CRC32C of "123456789" should be 0xE3069283
        assert_eq!(crc, 0xE306_9283);
    }

    #[test]
    fn test_crc32c_stable_across_repeated_calls() {
        // Regression: `crc32c` now caches its lookup table in a `OnceLock`
        // instead of rebuilding it every call. Repeated calls — including
        // interleaved with different inputs, which would surface any
        // staleness from a badly-cached table — must keep matching the
        // freshly-computed table's results.
        let fresh_table = crc32c_table();
        for i in 0..5u8 {
            let data = vec![i; 16];
            let mut crc = !0u32;
            for &byte in &data {
                let idx = ((crc ^ (byte as u32)) & 0xFF) as usize;
                crc = (crc >> 8) ^ fresh_table[idx];
            }
            crc = !crc;
            assert_eq!(crc32c(&data), crc, "mismatch on iteration {i}");
        }
        assert_eq!(crc32c(b"123456789"), 0xE306_9283);
    }

    #[test]
    fn test_mask_crc() {
        let crc = 0xDEAD_BEEF;
        let masked = mask_crc(crc);
        // Verify masking formula
        let expected = crc.rotate_right(15).wrapping_add(0xa282_ead8);
        assert_eq!(masked, expected);
    }

    #[test]
    fn test_varint_encoding() {
        let mut buf = Vec::new();
        encode_varint(0, &mut buf);
        assert_eq!(buf, vec![0]);

        buf.clear();
        encode_varint(127, &mut buf);
        assert_eq!(buf, vec![127]);

        buf.clear();
        encode_varint(128, &mut buf);
        assert_eq!(buf, vec![0x80, 0x01]);

        buf.clear();
        encode_varint(300, &mut buf);
        assert_eq!(buf, vec![0xAC, 0x02]);
    }

    #[test]
    fn test_tensorboard_config_default() {
        let config = TensorBoardConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.flush_interval, 100);
    }

    #[test]
    fn test_tensorboard_config_new() {
        let tmp_path = std::env::temp_dir().join("oxigaf_tb_test");
        let config = TensorBoardConfig::new(&tmp_path);
        assert!(config.enabled);
        assert_eq!(config.log_dir, tmp_path);
    }

    #[test]
    fn test_tensorboard_config_validation() {
        let mut config = TensorBoardConfig::default();
        assert!(config.validate().is_ok()); // Disabled is OK

        config.enabled = true;
        config.log_dir = PathBuf::new();
        assert!(config.validate().is_err()); // Empty log_dir with enabled is error
    }

    #[test]
    fn test_tensorboard_writer_disabled() {
        let writer = TensorBoardWriter::disabled();
        assert!(!writer.is_enabled());
    }

    #[test]
    fn test_tensorboard_writer_creates_directory() {
        let temp_dir = std::env::temp_dir().join("oxigaf_tb_test_1");
        let _ = fs::remove_dir_all(&temp_dir); // Clean up any previous test

        let config = TensorBoardConfig::new(&temp_dir);
        let result = TensorBoardWriter::new(config);

        assert!(result.is_ok());
        assert!(temp_dir.exists());

        // Clean up
        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_scalar_logging() {
        let temp_dir = std::env::temp_dir().join("oxigaf_tb_test_2");
        let _ = fs::remove_dir_all(&temp_dir);

        let config = TensorBoardConfig::new(&temp_dir).with_flush_interval(0);
        let mut writer = TensorBoardWriter::new(config).ok();

        if let Some(ref mut w) = writer {
            let result = w.log_scalar("test/loss", 0.5, 1);
            assert!(result.is_ok());

            let result = w.log_scalar("test/accuracy", 0.9, 2);
            assert!(result.is_ok());

            assert!(w.flush().is_ok());
        }

        // Clean up
        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_scalars_batch_logging() {
        let temp_dir = std::env::temp_dir().join("oxigaf_tb_test_3");
        let _ = fs::remove_dir_all(&temp_dir);

        let config = TensorBoardConfig::new(&temp_dir);
        let mut writer = TensorBoardWriter::new(config).ok();

        if let Some(ref mut w) = writer {
            let result = w.log_scalars(
                &[("loss/l1", 0.1), ("loss/ssim", 0.2), ("metrics/psnr", 25.0)],
                10,
            );
            assert!(result.is_ok());
        }

        // Clean up
        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_histogram_logging() {
        let temp_dir = std::env::temp_dir().join("oxigaf_tb_test_4");
        let _ = fs::remove_dir_all(&temp_dir);

        let config = TensorBoardConfig::new(&temp_dir);
        let mut writer = TensorBoardWriter::new(config).ok();

        if let Some(ref mut w) = writer {
            let data: Vec<f32> = (0..100).map(|i| i as f32 * 0.01).collect();
            let result = w.log_histogram("gradients/position", &data, 5);
            assert!(result.is_ok());
        }

        // Clean up
        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_default_bucket_limits_sorted_symmetric_and_bounded() {
        let limits = default_bucket_limits();
        assert!(limits.len() >= 4);
        assert_eq!(limits.len() % 2, 0, "symmetric scheme needs an even count");
        for w in limits.windows(2) {
            assert!(
                w[0] < w[1],
                "limits must be strictly ascending: {} >= {}",
                w[0],
                w[1]
            );
        }
        let half = limits.len() / 2;
        for i in 0..half {
            assert_eq!(
                limits[i],
                -limits[limits.len() - 1 - i],
                "not symmetric at {i}"
            );
        }
        assert_eq!(limits[half - 1], -1.0e-12);
        assert_eq!(limits[half], 1.0e-12);
        assert_eq!(*limits.first().unwrap(), -f64::MAX);
        assert_eq!(*limits.last().unwrap(), f64::MAX);
    }

    #[test]
    fn test_compute_histogram_limits_and_counts_same_length() {
        let writer = TensorBoardWriter::disabled();
        for data in [
            vec![-0.5_f32, -0.2, 0.0, 0.3, 0.5], // mixed sign
            vec![1.0_f32; 10],                   // constant (the old degenerate branch)
            vec![-3.0_f32, -1.0, -0.1],          // all negative
            vec![0.1_f32, 1.0, 100.0],           // all positive
        ] {
            let (_, _, _, _, bucket_limits, bucket_counts) = writer.compute_histogram(&data);
            assert_eq!(
                bucket_limits.len(),
                bucket_counts.len(),
                "mismatch for {data:?}"
            );
            let total: f64 = bucket_counts.iter().sum();
            assert_eq!(
                total,
                data.len() as f64,
                "counts must sum to n for {data:?}"
            );
        }
    }

    #[test]
    fn test_compute_histogram_mixed_sign_not_all_in_one_bucket() {
        // Regression: gradients uniform in [-0.5, 0.5] used to put ~99% of
        // samples in bucket 0, because bucket limits were built from
        // `ln|min|..ln|max|` and negated only when `max < 0` — every limit
        // ended up positive and near `ln(0.5)`, so every negative value
        // (half the data) plus most positives fell below the smallest one.
        let writer = TensorBoardWriter::disabled();
        let n = 1000;
        let data: Vec<f32> = (0..n).map(|i| -0.5 + i as f32 / (n - 1) as f32).collect();
        let (min, max, _, _, _, bucket_counts) = writer.compute_histogram(&data);
        assert!(min < 0.0 && max > 0.0, "fixture must be mixed-sign");

        let total: f64 = bucket_counts.iter().sum();
        assert_eq!(total, n as f64);

        let max_single_bucket = bucket_counts.iter().copied().fold(0.0_f64, f64::max);
        assert!(
            max_single_bucket < 0.5 * n as f64,
            "no bucket should absorb the majority of mixed-sign samples: {max_single_bucket} of {n}"
        );
        // The extreme overflow buckets stay empty for ordinary bounded data.
        assert_eq!(bucket_counts[0], 0.0);
        assert_eq!(*bucket_counts.last().unwrap(), 0.0);
    }

    #[test]
    fn test_image_logging() {
        let temp_dir = std::env::temp_dir().join("oxigaf_tb_test_5");
        let _ = fs::remove_dir_all(&temp_dir);

        let config = TensorBoardConfig::new(&temp_dir);
        let mut writer = TensorBoardWriter::new(config).ok();

        if let Some(ref mut w) = writer {
            // Create a simple 2x2 RGB image
            let image_data = vec![
                1.0, 0.0, 0.0, // Red pixel
                0.0, 1.0, 0.0, // Green pixel
                0.0, 0.0, 1.0, // Blue pixel
                1.0, 1.0, 1.0, // White pixel
            ];
            let result = w.log_image("images/render", &image_data, 2, 2, 1);
            assert!(result.is_ok());
        }

        // Clean up
        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_training_metrics_logger() {
        let temp_dir = std::env::temp_dir().join("oxigaf_tb_test_6");
        let _ = fs::remove_dir_all(&temp_dir);

        let config = TensorBoardConfig::new(&temp_dir);
        let result = TrainingMetricsLogger::new(config);

        if let Ok(mut logger) = result {
            assert!(logger.is_enabled());

            let lr = LearningRates::from_config(1e-4, 1e-3, 5e-3, 5e-2, 2.5e-3);

            let result = logger.log_step(100, 0.05, 25.0, 0.95, 50000, &lr);
            assert!(result.is_ok());

            let result = logger.log_losses(100, 0.03, 0.01, 0.005, 0.002, 0.003);
            assert!(result.is_ok());
        }

        // Clean up
        let _ = fs::remove_dir_all(&temp_dir);
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
    }

    #[test]
    fn test_learning_rates() {
        let lr = LearningRates::from_config(1e-4, 1e-3, 5e-3, 5e-2, 2.5e-3);
        assert!((lr.position - 1e-4).abs() < 1e-10);
        assert!((lr.rotation - 1e-3).abs() < 1e-10);
        assert!((lr.scale - 5e-3).abs() < 1e-10);
        assert!((lr.opacity - 5e-2).abs() < 1e-10);
        assert!((lr.sh - 2.5e-3).abs() < 1e-10);
    }
}
