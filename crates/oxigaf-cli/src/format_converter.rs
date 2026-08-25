//! Format conversion utilities for 3D Gaussian Splatting scene files.
//!
//! Supports converting between PLY, JSON, CSV, and a custom binary format for
//! Gaussian scene data. This module handles the pure data transformation logic
//! without file I/O — callers are responsible for reading and writing bytes.
//!
//! # Supported Formats
//! - **CSV** — human-readable, one Gaussian per line, 14 columns
//! - **JSON** — pretty-printed array of Gaussian objects
//! - **Binary** — compact little-endian binary with magic header (`OXIGAF01`)
//!
//! # Example
//! ```rust
//! use oxigaf_cli::format_converter::{
//!     GaussianRecord, FileFormat, to_binary, from_binary,
//! };
//!
//! let record = GaussianRecord::new(
//!     [1.0, 2.0, 3.0],
//!     [-1.0, -1.0, -1.0],
//!     [0.0, 0.0, 0.0, 1.0],
//!     0.0,
//!     [0.0, 0.0, 0.0],
//! );
//! let bytes = to_binary(&[record]).expect("encode failed");
//! let parsed = from_binary(&bytes).expect("roundtrip failed");
//! assert_eq!(parsed.len(), 1);
//! ```

use thiserror::Error;

// ---------------------------------------------------------------------------
// ConvertError
// ---------------------------------------------------------------------------

/// Errors that can occur during format conversion.
#[derive(Debug, Error)]
pub enum ConvertError {
    /// Unknown or malformed format string.
    #[error("invalid format: {0}")]
    InvalidFormat(String),

    /// A required field is absent in the source data.
    #[error("missing required field: {0}")]
    FieldMissing(String),

    /// A field's type does not match expectations.
    #[error("type mismatch for field '{field}': expected {expected}, got {got}")]
    FieldTypeMismatch {
        field: String,
        expected: String,
        got: String,
    },

    /// A field's dimension (length) does not match expectations.
    #[error("dimension error for field '{field}': expected {expected}, got {got}")]
    DimensionError {
        field: String,
        expected: usize,
        got: usize,
    },

    /// A generic parse failure.
    #[error("parse error: {0}")]
    ParseError(String),
}

// ---------------------------------------------------------------------------
// GaussianRecord
// ---------------------------------------------------------------------------

/// A single Gaussian with all required fields for format conversion.
///
/// Fields are stored in their raw (log-space / logit-space) forms so that
/// serialisation is lossless. Convenience methods expose activated values.
#[derive(Debug, Clone)]
pub struct GaussianRecord {
    /// World-space centre [x, y, z].
    pub position: [f32; 3],
    /// Log-space scales [log_sx, log_sy, log_sz].
    pub log_scale: [f32; 3],
    /// Quaternion rotation [qx, qy, qz, qw].
    pub rotation: [f32; 4],
    /// Opacity in logit space.
    pub opacity: f32,
    /// DC spherical harmonics (colour base) — not the linear RGB colour.
    pub sh_dc: [f32; 3],
    /// Remaining higher-order SH coefficients (empty for sh_degree = 0).
    pub sh_rest: Vec<f32>,
}

impl GaussianRecord {
    /// Create a new record with empty `sh_rest`.
    pub fn new(
        position: [f32; 3],
        log_scale: [f32; 3],
        rotation: [f32; 4],
        opacity: f32,
        sh_dc: [f32; 3],
    ) -> Self {
        Self {
            position,
            log_scale,
            rotation,
            opacity,
            sh_dc,
            sh_rest: vec![],
        }
    }

    /// Builder method to attach higher-order SH coefficients.
    pub fn with_sh_rest(mut self, sh_rest: Vec<f32>) -> Self {
        self.sh_rest = sh_rest;
        self
    }

    /// Activated opacity in [0, 1] via sigmoid: `1 / (1 + exp(-opacity))`.
    #[must_use]
    pub fn activated_opacity(&self) -> f32 {
        1.0_f32 / (1.0_f32 + (-self.opacity).exp())
    }

    /// Scale in linear space, derived from `log_scale` via element-wise `exp`.
    #[must_use]
    pub fn scale(&self) -> [f32; 3] {
        [
            self.log_scale[0].exp(),
            self.log_scale[1].exp(),
            self.log_scale[2].exp(),
        ]
    }

    /// RGB colour derived from DC SH coefficients using the C0 factor (0.2820947917).
    ///
    /// Each channel is clamped to [0, 1].
    #[must_use]
    pub fn base_color(&self) -> [f32; 3] {
        const C0: f32 = 0.282_094_8_f32;
        let r = (0.5_f32 + C0 * self.sh_dc[0]).clamp(0.0, 1.0);
        let g = (0.5_f32 + C0 * self.sh_dc[1]).clamp(0.0, 1.0);
        let b = (0.5_f32 + C0 * self.sh_dc[2]).clamp(0.0, 1.0);
        [r, g, b]
    }

    /// Maximum scale dimension (exp of the largest log_scale component).
    #[must_use]
    pub fn max_scale(&self) -> f32 {
        let max_log = self
            .log_scale
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        max_log.exp()
    }

    /// Volume approximation: product of activated scales.
    ///
    /// Equivalent to `exp(log_sx + log_sy + log_sz)`.
    #[must_use]
    pub fn volume(&self) -> f32 {
        (self.log_scale[0] + self.log_scale[1] + self.log_scale[2]).exp()
    }

    /// Number of SH degrees inferred from `sh_rest.len()`:
    /// 0 → degree 0, 9 → degree 1, 24 → degree 2, 45 → degree 3.
    /// Unrecognised lengths map to degree 0.
    #[must_use]
    pub fn sh_degree(&self) -> usize {
        degree_for_rest_len(self.sh_rest.len()).unwrap_or(0)
    }
}

// ---------------------------------------------------------------------------
// Helper: SH rest coefficient count from degree
// ---------------------------------------------------------------------------

/// Number of `sh_rest` coefficients for a given SH degree.
///
/// 3DGS convention: `((degree+1)² − 1) × 3` (DC term excluded — that's
/// `sh_dc`). Matches `export_ply::PlyGaussian::n_rest_coeffs` and
/// `info::sh_degree_from_rest`.
///
/// | degree | coefficients |
/// |--------|-------------|
/// | 0      | 0           |
/// | 1      | 9           |
/// | 2      | 24          |
/// | 3      | 45          |
/// | other  | 0           |
///
/// Only degrees 0–3 exist in the 3DGS convention (and in this format's
/// header, whose `sh_degree` field is validated by [`from_binary`]), so any
/// other degree maps to 0 rather than extrapolating the formula. Evaluating
/// `((degree+1)² − 1) × 3` unguarded overflowed `usize` for a large
/// attacker- or corruption-supplied header value (`degree = u32::MAX`
/// squares to exactly 2⁶⁴), which panics in debug builds; the upper-bound
/// guard below keeps this function total for every `usize`.
#[must_use]
pub fn sh_rest_size(degree: usize) -> usize {
    if degree == 0 || degree > MAX_SH_DEGREE {
        0
    } else {
        ((degree + 1) * (degree + 1) - 1) * 3
    }
}

/// Highest SH degree the 3DGS convention (and hence this module's binary
/// header and [`degree_for_rest_len`]) represents.
const MAX_SH_DEGREE: usize = 3;

/// Recognised SH degree (0–3) for an exact `sh_rest` length, or `None` if the
/// length does not correspond to any degree in the 3DGS convention.
///
/// Inverse of [`sh_rest_size`], restricted to the four degrees this format
/// recognises. Used internally to detect (rather than silently coerce to
/// degree 0) an `sh_rest` length that doesn't fit the convention.
fn degree_for_rest_len(len: usize) -> Option<usize> {
    match len {
        0 => Some(0),
        9 => Some(1),
        24 => Some(2),
        45 => Some(3),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// CSV format
// ---------------------------------------------------------------------------

/// CSV header written by [`to_csv`].
const CSV_HEADER: &str = "x,y,z,sx,sy,sz,qx,qy,qz,qw,opacity_logit,r,g,b\n";
/// Number of columns expected in every data row.
const CSV_COLUMNS: usize = 14;

/// Convert a slice of [`GaussianRecord`] to CSV bytes.
///
/// The first line is the column header `CSV_HEADER`.  Each subsequent line
/// contains one Gaussian with values formatted to 6 decimal places.
/// The colour columns (`r`, `g`, `b`) contain [`GaussianRecord::base_color`]
/// values — **not** raw SH coefficients — because CSV does not store full SH.
pub fn to_csv(records: &[GaussianRecord]) -> Vec<u8> {
    let mut out = String::with_capacity(CSV_HEADER.len() + records.len() * 120);
    out.push_str(CSV_HEADER);

    for r in records {
        let [rx, ry, rz] = r.base_color();
        let [px, py, pz] = r.position;
        let [sx, sy, sz] = r.log_scale;
        let [qx, qy, qz, qw] = r.rotation;

        let line = format!(
            "{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
            px, py, pz, sx, sy, sz, qx, qy, qz, qw, r.opacity, rx, ry, rz,
        );
        out.push_str(&line);
    }

    out.into_bytes()
}

/// Parse CSV bytes into a [`Vec<GaussianRecord>`].
///
/// The first (header) line is skipped.  Exactly `CSV_COLUMNS` (14) columns
/// are required per row.  Since CSV does not encode full SH coefficients,
/// `sh_rest` is always set to an empty vector; `sh_dc` is back-calculated from
/// the stored RGB colour.
///
/// # Errors
/// Returns [`ConvertError::DimensionError`] if a row has the wrong number of
/// columns, or [`ConvertError::ParseError`] if a value cannot be parsed as f32.
pub fn from_csv(data: &[u8]) -> Result<Vec<GaussianRecord>, ConvertError> {
    let text = std::str::from_utf8(data)
        .map_err(|e| ConvertError::ParseError(format!("UTF-8 decode failed: {e}")))?;

    const C0: f32 = 0.282_094_8_f32;

    let mut records = Vec::new();

    for (line_idx, line) in text.lines().enumerate() {
        // Skip header (first line) and blank lines.
        if line_idx == 0 || line.trim().is_empty() {
            continue;
        }

        let cols: Vec<&str> = line.split(',').collect();
        if cols.len() != CSV_COLUMNS {
            return Err(ConvertError::DimensionError {
                field: format!("row {}", line_idx + 1),
                expected: CSV_COLUMNS,
                got: cols.len(),
            });
        }

        let mut vals = [0.0_f32; CSV_COLUMNS];
        for (i, col) in cols.iter().enumerate() {
            vals[i] = col.trim().parse::<f32>().map_err(|e| {
                ConvertError::ParseError(format!(
                    "row {}, column {}: '{}' is not a valid f32: {e}",
                    line_idx + 1,
                    i + 1,
                    col.trim()
                ))
            })?;
        }

        // Columns: x y z sx sy sz qx qy qz qw opacity r g b
        let position = [vals[0], vals[1], vals[2]];
        let log_scale = [vals[3], vals[4], vals[5]];
        let rotation = [vals[6], vals[7], vals[8], vals[9]];
        let opacity = vals[10];
        // Invert base_color: color = 0.5 + C0 * sh_dc → sh_dc = (color - 0.5) / C0
        let sh_dc = [
            (vals[11] - 0.5) / C0,
            (vals[12] - 0.5) / C0,
            (vals[13] - 0.5) / C0,
        ];

        records.push(GaussianRecord::new(
            position, log_scale, rotation, opacity, sh_dc,
        ));
    }

    Ok(records)
}

// ---------------------------------------------------------------------------
// JSON format (hand-rolled, no serde dependency in logic)
// ---------------------------------------------------------------------------

/// Format an f32 slice as a JSON array string with 6 decimal places.
fn fmt_f32_array(vals: &[f32]) -> String {
    let inner: Vec<String> = vals.iter().map(|v| format!("{v:.6}")).collect();
    format!("[{}]", inner.join(", "))
}

/// Convert a slice of [`GaussianRecord`] to pretty-printed JSON bytes.
///
/// The output is a top-level JSON array of objects.  Each object contains the
/// fields `position`, `log_scale`, `rotation`, `opacity`, `sh_dc`, and
/// `sh_rest`.  Values are formatted to 6 decimal places.
pub fn to_json(records: &[GaussianRecord]) -> Vec<u8> {
    let mut out = String::with_capacity(records.len() * 256 + 4);
    out.push_str("[\n");

    for (idx, r) in records.iter().enumerate() {
        out.push_str("  {\n");
        out.push_str(&format!(
            "    \"position\": {},\n",
            fmt_f32_array(&r.position)
        ));
        out.push_str(&format!(
            "    \"log_scale\": {},\n",
            fmt_f32_array(&r.log_scale)
        ));
        out.push_str(&format!(
            "    \"rotation\": {},\n",
            fmt_f32_array(&r.rotation)
        ));
        out.push_str(&format!("    \"opacity\": {:.6},\n", r.opacity));
        out.push_str(&format!("    \"sh_dc\": {},\n", fmt_f32_array(&r.sh_dc)));
        out.push_str(&format!("    \"sh_rest\": {}\n", fmt_f32_array(&r.sh_rest)));
        if idx + 1 < records.len() {
            out.push_str("  },\n");
        } else {
            out.push_str("  }\n");
        }
    }

    out.push_str("]\n");
    out.into_bytes()
}

// ---------------------------------------------------------------------------
// Minimal JSON parser for the exact format produced by `to_json`
// ---------------------------------------------------------------------------

/// Extract the value string for a given JSON key within one object's text.
///
/// Searches for `"key":` and returns everything after the `:` (trimmed) up to
/// the next key pattern or end of the object block.
fn json_extract_value<'a>(obj: &'a str, key: &str) -> Option<&'a str> {
    let needle = format!("\"{}\":", key);
    let start = obj.find(needle.as_str())?;
    let after_colon = obj[start + needle.len()..].trim_start();
    // Value ends at the next `"key":` pattern or at `}`.
    // Find the first newline that is followed by whitespace + `"` (start of next key)
    // or the closing `}`.  We look for `,\n` or `\n` after the value.
    let value_end = after_colon
        .find('\n')
        .map(|n| {
            // Trim trailing comma from the value if present.
            let candidate = after_colon[..n].trim_end_matches(',').trim();
            candidate
        })
        .unwrap_or_else(|| after_colon.trim_end_matches([',', '}', '\n', ' ']).trim());
    Some(value_end)
}

/// Parse a JSON array of f32 values, e.g. `[1.0, 2.0, 3.0]`.
fn parse_f32_array(s: &str) -> Result<Vec<f32>, ConvertError> {
    let trimmed = s.trim();
    if !trimmed.starts_with('[') || !trimmed.ends_with(']') {
        // Truncate by chars, not bytes: a byte-index slice can land inside a
        // multibyte UTF-8 sequence and panic on non-ASCII input (this is a
        // public parser over untrusted JSON, so that input is expected).
        let snippet: String = trimmed.chars().take(40).collect();
        return Err(ConvertError::ParseError(format!(
            "expected JSON array, got: {snippet}"
        )));
    }
    let inner = &trimmed[1..trimmed.len() - 1];
    if inner.trim().is_empty() {
        return Ok(vec![]);
    }
    inner
        .split(',')
        .map(|tok| {
            tok.trim().parse::<f32>().map_err(|e| {
                ConvertError::ParseError(format!("f32 parse error: '{}': {e}", tok.trim()))
            })
        })
        .collect()
}

/// Parse a JSON f32 scalar, e.g. `0.500000`.
fn parse_f32_scalar(s: &str) -> Result<f32, ConvertError> {
    s.trim()
        .trim_end_matches(',')
        .parse::<f32>()
        .map_err(|e| ConvertError::ParseError(format!("f32 scalar parse error: '{s}': {e}")))
}

/// Parse JSON bytes into a [`Vec<GaussianRecord>`].
///
/// Only the exact format produced by [`to_json`] is supported: a top-level
/// array of objects each containing `position`, `log_scale`, `rotation`,
/// `opacity`, `sh_dc`, and `sh_rest`.  A minimal line-based tokeniser is used
/// rather than a general JSON parser.
///
/// # Errors
/// Returns [`ConvertError`] if the data cannot be decoded as UTF-8, if the
/// top-level structure is not an array, or if any required field is missing or
/// malformed.
pub fn from_json(data: &[u8]) -> Result<Vec<GaussianRecord>, ConvertError> {
    let text = std::str::from_utf8(data)
        .map_err(|e| ConvertError::ParseError(format!("UTF-8 decode failed: {e}")))?;

    let trimmed = text.trim();
    if !trimmed.starts_with('[') {
        return Err(ConvertError::InvalidFormat(
            "JSON must start with '['".to_string(),
        ));
    }

    // Split the text into object blocks by finding `{` … `}` pairs at the top level.
    let object_chunks = split_json_objects(trimmed)?;

    let mut records = Vec::with_capacity(object_chunks.len());

    for (obj_idx, chunk) in object_chunks.iter().enumerate() {
        // --- position ---
        let pos_str = json_extract_value(chunk, "position")
            .ok_or_else(|| ConvertError::FieldMissing(format!("object {obj_idx}: position")))?;
        let pos_vals = parse_f32_array(pos_str)?;
        if pos_vals.len() != 3 {
            return Err(ConvertError::DimensionError {
                field: format!("object {obj_idx}: position"),
                expected: 3,
                got: pos_vals.len(),
            });
        }
        let position = [pos_vals[0], pos_vals[1], pos_vals[2]];

        // --- log_scale ---
        let ls_str = json_extract_value(chunk, "log_scale")
            .ok_or_else(|| ConvertError::FieldMissing(format!("object {obj_idx}: log_scale")))?;
        let ls_vals = parse_f32_array(ls_str)?;
        if ls_vals.len() != 3 {
            return Err(ConvertError::DimensionError {
                field: format!("object {obj_idx}: log_scale"),
                expected: 3,
                got: ls_vals.len(),
            });
        }
        let log_scale = [ls_vals[0], ls_vals[1], ls_vals[2]];

        // --- rotation ---
        let rot_str = json_extract_value(chunk, "rotation")
            .ok_or_else(|| ConvertError::FieldMissing(format!("object {obj_idx}: rotation")))?;
        let rot_vals = parse_f32_array(rot_str)?;
        if rot_vals.len() != 4 {
            return Err(ConvertError::DimensionError {
                field: format!("object {obj_idx}: rotation"),
                expected: 4,
                got: rot_vals.len(),
            });
        }
        let rotation = [rot_vals[0], rot_vals[1], rot_vals[2], rot_vals[3]];

        // --- opacity ---
        let op_str = json_extract_value(chunk, "opacity")
            .ok_or_else(|| ConvertError::FieldMissing(format!("object {obj_idx}: opacity")))?;
        let opacity = parse_f32_scalar(op_str)?;

        // --- sh_dc ---
        let shdc_str = json_extract_value(chunk, "sh_dc")
            .ok_or_else(|| ConvertError::FieldMissing(format!("object {obj_idx}: sh_dc")))?;
        let shdc_vals = parse_f32_array(shdc_str)?;
        if shdc_vals.len() != 3 {
            return Err(ConvertError::DimensionError {
                field: format!("object {obj_idx}: sh_dc"),
                expected: 3,
                got: shdc_vals.len(),
            });
        }
        let sh_dc = [shdc_vals[0], shdc_vals[1], shdc_vals[2]];

        // --- sh_rest ---
        let shrest_str = json_extract_value(chunk, "sh_rest")
            .ok_or_else(|| ConvertError::FieldMissing(format!("object {obj_idx}: sh_rest")))?;
        let sh_rest = parse_f32_array(shrest_str)?;

        records.push(
            GaussianRecord::new(position, log_scale, rotation, opacity, sh_dc)
                .with_sh_rest(sh_rest),
        );
    }

    Ok(records)
}

/// Split a JSON text (starting with `[`) into a list of object body strings.
///
/// Each returned string is the text of one `{…}` block.  Only handles the
/// simple two-level nesting produced by [`to_json`] (no nested objects).
fn split_json_objects(text: &str) -> Result<Vec<String>, ConvertError> {
    let mut chunks = Vec::new();
    let mut depth = 0_i32;
    let mut obj_start: Option<usize> = None;

    for (byte_idx, ch) in text.char_indices() {
        match ch {
            '{' => {
                depth += 1;
                if depth == 1 {
                    obj_start = Some(byte_idx);
                }
            }
            '}' => {
                depth -= 1;
                if depth == 0 {
                    if let Some(start) = obj_start.take() {
                        chunks.push(text[start..=byte_idx].to_string());
                    }
                }
            }
            _ => {}
        }
    }

    if depth != 0 {
        return Err(ConvertError::InvalidFormat(
            "unmatched braces in JSON input".to_string(),
        ));
    }

    Ok(chunks)
}

// ---------------------------------------------------------------------------
// Binary format
// ---------------------------------------------------------------------------

/// Magic bytes identifying an OxiGAF binary scene file.
pub const BINARY_MAGIC: [u8; 8] = *b"OXIGAF01";

/// Current version of the binary format.
pub const BINARY_VERSION: u32 = 1;

/// Header occupying the first 24 bytes of a binary scene file.
///
/// Layout (all integers in little-endian):
/// - bytes  0–7:  `magic`  — 8 ASCII bytes
/// - bytes  8–11: `version` — u32
/// - bytes 12–15: `num_gaussians` — u32
/// - bytes 16–19: `sh_degree` — u32
/// - bytes 20–23: `flags` — u32 (reserved, always 0)
#[derive(Debug, Clone)]
pub struct BinaryHeader {
    /// Magic bytes — must equal [`BINARY_MAGIC`].
    pub magic: [u8; 8],
    /// Format version — must equal [`BINARY_VERSION`].
    pub version: u32,
    /// Number of Gaussians stored in the file.
    pub num_gaussians: u32,
    /// SH degree for all records.
    pub sh_degree: u32,
    /// Reserved flags (always 0).
    pub flags: u32,
}

/// Total size of the serialised [`BinaryHeader`] in bytes.
const HEADER_SIZE: usize = 24;

impl BinaryHeader {
    /// Create a new valid header.
    pub fn new(num_gaussians: usize, sh_degree: usize) -> Self {
        Self {
            magic: BINARY_MAGIC,
            version: BINARY_VERSION,
            num_gaussians: num_gaussians as u32,
            sh_degree: sh_degree as u32,
            flags: 0,
        }
    }

    /// Returns `true` if the magic and version fields are valid.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.magic == BINARY_MAGIC && self.version == BINARY_VERSION
    }
}

/// Write a `BinaryHeader` into an output buffer as little-endian bytes.
fn write_header(header: &BinaryHeader, out: &mut Vec<u8>) {
    out.extend_from_slice(&header.magic);
    out.extend_from_slice(&header.version.to_le_bytes());
    out.extend_from_slice(&header.num_gaussians.to_le_bytes());
    out.extend_from_slice(&header.sh_degree.to_le_bytes());
    out.extend_from_slice(&header.flags.to_le_bytes());
}

/// Read a little-endian u32 from `data` starting at `offset`.
fn read_u32_le(data: &[u8], offset: usize) -> Result<u32, ConvertError> {
    data.get(offset..offset + 4)
        .and_then(|b| b.try_into().ok())
        .map(u32::from_le_bytes)
        .ok_or_else(|| {
            ConvertError::ParseError(format!(
                "truncated binary: cannot read u32 at offset {offset}"
            ))
        })
}

/// Read a little-endian f32 from `data` starting at `offset`.
fn read_f32_le(data: &[u8], offset: usize) -> Result<f32, ConvertError> {
    data.get(offset..offset + 4)
        .and_then(|b| b.try_into().ok())
        .map(f32::from_le_bytes)
        .ok_or_else(|| {
            ConvertError::ParseError(format!(
                "truncated binary: cannot read f32 at offset {offset}"
            ))
        })
}

/// Write a single f32 as little-endian bytes into `out`.
#[inline]
fn push_f32(out: &mut Vec<u8>, v: f32) {
    out.extend_from_slice(&v.to_le_bytes());
}

/// Convert a slice of [`GaussianRecord`] to a binary blob.
///
/// Layout:
/// - 24-byte header ([`BinaryHeader`])
/// - Per-Gaussian records, each:
///   - position:  3 × f32 (12 bytes)
///   - log_scale: 3 × f32 (12 bytes)
///   - rotation:  4 × f32 (16 bytes)
///   - opacity:   1 × f32 (4 bytes)
///   - sh_dc:     3 × f32 (12 bytes)
///   - sh_rest:   n × f32 (n determined by sh_degree in header)
///
/// All values are stored as little-endian IEEE 754 f32.
///
/// The on-disk `sh_degree` is derived from the records themselves: every
/// record must carry the same `sh_rest.len()`, and that length must
/// correspond to a recognised SH degree (0, 9, 24, or 45 coefficients — see
/// [`sh_rest_size`]). Earlier code derived the degree from `records.first()`
/// alone and silently truncated-or-zero-padded every other record to match,
/// which corrupted (dropped SH data from) any heterogeneous record set with
/// no error and no warning.
///
/// # Errors
/// Returns [`ConvertError::DimensionError`] if records disagree on
/// `sh_rest.len()`, or [`ConvertError::InvalidFormat`] if the (consistent)
/// length does not correspond to a supported SH degree.
pub fn to_binary(records: &[GaussianRecord]) -> Result<Vec<u8>, ConvertError> {
    let degree = match records.first() {
        None => 0,
        Some(first) => {
            let expected_len = first.sh_rest.len();
            for (idx, r) in records.iter().enumerate() {
                if r.sh_rest.len() != expected_len {
                    return Err(ConvertError::DimensionError {
                        field: format!(
                            "record {idx}: sh_rest.len() (must match record 0's length)"
                        ),
                        expected: expected_len,
                        got: r.sh_rest.len(),
                    });
                }
            }
            degree_for_rest_len(expected_len).ok_or_else(|| {
                ConvertError::InvalidFormat(format!(
                    "sh_rest length {expected_len} does not correspond to a supported SH \
                     degree (expected 0, 9, 24, or 45 coefficients)"
                ))
            })?
        }
    };
    let sh_extra = sh_rest_size(degree);
    // Fixed bytes per Gaussian (excluding sh_rest): 3+3+4+1+3 = 14 f32s = 56 bytes
    let bytes_per_record = 56 + sh_extra * 4;
    let total = HEADER_SIZE + records.len() * bytes_per_record;

    let mut out = Vec::with_capacity(total);

    let header = BinaryHeader::new(records.len(), degree);
    write_header(&header, &mut out);

    for r in records {
        for v in &r.position {
            push_f32(&mut out, *v);
        }
        for v in &r.log_scale {
            push_f32(&mut out, *v);
        }
        for v in &r.rotation {
            push_f32(&mut out, *v);
        }
        push_f32(&mut out, r.opacity);
        for v in &r.sh_dc {
            push_f32(&mut out, *v);
        }
        // Every record was validated above to have exactly `sh_extra`
        // sh_rest coefficients, so `unwrap_or` here is an unreachable
        // safety net, not a silent truncation/pad.
        for i in 0..sh_extra {
            let v = r.sh_rest.get(i).copied().unwrap_or(0.0);
            push_f32(&mut out, v);
        }
    }

    Ok(out)
}

/// Parse a binary blob produced by [`to_binary`] into a [`Vec<GaussianRecord>`].
///
/// # Errors
/// Returns [`ConvertError::InvalidFormat`] if the magic bytes or version are
/// wrong, or [`ConvertError::ParseError`] if the data is truncated.
pub fn from_binary(data: &[u8]) -> Result<Vec<GaussianRecord>, ConvertError> {
    if data.len() < HEADER_SIZE {
        return Err(ConvertError::ParseError(format!(
            "binary data too short for header: {} bytes",
            data.len()
        )));
    }

    // Validate magic.
    let magic: [u8; 8] = data[0..8]
        .try_into()
        .map_err(|_| ConvertError::ParseError("cannot read magic bytes".to_string()))?;
    if magic != BINARY_MAGIC {
        return Err(ConvertError::InvalidFormat(format!(
            "invalid magic: {:?}",
            &data[0..8.min(data.len())]
        )));
    }

    // Parse header fields.
    let version = read_u32_le(data, 8)?;
    if version != BINARY_VERSION {
        return Err(ConvertError::InvalidFormat(format!(
            "unsupported binary version: {version}"
        )));
    }

    let num_gaussians = read_u32_le(data, 12)? as usize;
    let sh_degree = read_u32_le(data, 16)? as usize;
    // Reject an out-of-range degree explicitly. `sh_rest_size` maps any
    // unrecognised degree to 0, so without this check a header claiming
    // e.g. degree 7 would be parsed as "no sh_rest coefficients at all" —
    // every record silently stripped of its SH data and the record stride
    // computed wrong, with no error.
    if sh_degree > MAX_SH_DEGREE {
        return Err(ConvertError::InvalidFormat(format!(
            "unsupported SH degree in header: {sh_degree} (supported: 0..={MAX_SH_DEGREE})"
        )));
    }
    // flags at bytes 20–23 are reserved; read but ignore.
    let _flags = read_u32_le(data, 20)?;

    let sh_extra = sh_rest_size(sh_degree);
    // Fixed f32 fields per record: 3+3+4+1+3 = 14 → 56 bytes + sh_rest.
    let bytes_per_record = 56 + sh_extra * 4;
    // Saturating: `num_gaussians` comes straight off disk, so on a 32-bit
    // target a corrupt count could otherwise wrap this product to a small
    // number and wave a truncated file through the length check below.
    let expected_total = HEADER_SIZE.saturating_add(num_gaussians.saturating_mul(bytes_per_record));

    if data.len() < expected_total {
        return Err(ConvertError::ParseError(format!(
            "binary data truncated: expected {expected_total} bytes, got {}",
            data.len()
        )));
    }

    let mut records = Vec::with_capacity(num_gaussians);
    let mut cursor = HEADER_SIZE;

    for _ in 0..num_gaussians {
        let px = read_f32_le(data, cursor)?;
        let py = read_f32_le(data, cursor + 4)?;
        let pz = read_f32_le(data, cursor + 8)?;
        cursor += 12;

        let sx = read_f32_le(data, cursor)?;
        let sy = read_f32_le(data, cursor + 4)?;
        let sz = read_f32_le(data, cursor + 8)?;
        cursor += 12;

        let qx = read_f32_le(data, cursor)?;
        let qy = read_f32_le(data, cursor + 4)?;
        let qz = read_f32_le(data, cursor + 8)?;
        let qw = read_f32_le(data, cursor + 12)?;
        cursor += 16;

        let opacity = read_f32_le(data, cursor)?;
        cursor += 4;

        let dr = read_f32_le(data, cursor)?;
        let dg = read_f32_le(data, cursor + 4)?;
        let db = read_f32_le(data, cursor + 8)?;
        cursor += 12;

        let mut sh_rest = Vec::with_capacity(sh_extra);
        for _ in 0..sh_extra {
            sh_rest.push(read_f32_le(data, cursor)?);
            cursor += 4;
        }

        records.push(
            GaussianRecord::new(
                [px, py, pz],
                [sx, sy, sz],
                [qx, qy, qz, qw],
                opacity,
                [dr, dg, db],
            )
            .with_sh_rest(sh_rest),
        );
    }

    Ok(records)
}

// ---------------------------------------------------------------------------
// Format detection and dispatch
// ---------------------------------------------------------------------------

/// Supported in-memory scene file formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileFormat {
    /// Comma-separated values — human-readable, 14 columns per Gaussian.
    Csv,
    /// JSON array of Gaussian objects — pretty-printed.
    Json,
    /// Compact little-endian binary with an `OXIGAF01` magic header.
    Binary,
}

impl FileFormat {
    /// Detect format from a file extension string (case-insensitive).
    ///
    /// | Extension      | Format              |
    /// |----------------|---------------------|
    /// | `"csv"`        | [`FileFormat::Csv`] |
    /// | `"json"`, `"jsonl"` | [`FileFormat::Json`] |
    /// | `"bin"`, `"oxigaf"` | [`FileFormat::Binary`] |
    #[must_use]
    pub fn from_extension(ext: &str) -> Option<FileFormat> {
        match ext.to_ascii_lowercase().as_str() {
            "csv" => Some(FileFormat::Csv),
            "json" | "jsonl" => Some(FileFormat::Json),
            "bin" | "oxigaf" => Some(FileFormat::Binary),
            _ => None,
        }
    }

    /// Detect format by inspecting the first bytes of the data.
    ///
    /// - Starts with [`BINARY_MAGIC`] → [`FileFormat::Binary`]
    /// - Starts with `[` or `{` (after whitespace) → [`FileFormat::Json`]
    /// - Anything else → [`FileFormat::Csv`]
    #[must_use]
    pub fn from_magic(data: &[u8]) -> FileFormat {
        if data.starts_with(&BINARY_MAGIC) {
            return FileFormat::Binary;
        }
        // Find first non-whitespace byte.
        for &byte in data {
            match byte {
                b' ' | b'\t' | b'\r' | b'\n' => continue,
                b'[' | b'{' => return FileFormat::Json,
                _ => return FileFormat::Csv,
            }
        }
        FileFormat::Csv
    }

    /// Canonical file extension for this format (without leading dot).
    #[must_use]
    pub fn extension(&self) -> &'static str {
        match self {
            FileFormat::Csv => "csv",
            FileFormat::Json => "json",
            FileFormat::Binary => "bin",
        }
    }
}

/// Convert a slice of [`GaussianRecord`] to bytes in the specified target format.
///
/// This is a convenience dispatcher that calls [`to_csv`], [`to_json`], or
/// [`to_binary`] depending on `target_format`.
///
/// # Errors
/// Returns [`ConvertError`] if `target_format` is [`FileFormat::Binary`] and
/// the records do not share a consistent, recognised `sh_rest` layout (see
/// [`to_binary`]). CSV and JSON encoding never fail.
pub fn convert(
    records: &[GaussianRecord],
    target_format: FileFormat,
) -> Result<Vec<u8>, ConvertError> {
    match target_format {
        FileFormat::Csv => Ok(to_csv(records)),
        FileFormat::Json => Ok(to_json(records)),
        FileFormat::Binary => to_binary(records),
    }
}

// ---------------------------------------------------------------------------
// Statistics and validation
// ---------------------------------------------------------------------------

/// Summary statistics computed from a collection of [`GaussianRecord`]s.
pub struct ConversionStats {
    /// Total number of records.
    pub num_records: usize,
    /// Records with all-finite position values.
    pub num_valid: usize,
    /// Records containing at least one NaN in their position.
    pub num_nan: usize,
    /// Records containing at least one Inf in their position.
    pub num_inf: usize,
    /// Mean of [`GaussianRecord::activated_opacity`] over valid records.
    pub mean_opacity: f32,
    /// Mean of [`GaussianRecord::max_scale`] over valid records.
    pub mean_max_scale: f32,
    /// SH degree inferred from the first record (0 if empty).
    pub sh_degree: usize,
}

/// Compute summary statistics for a slice of [`GaussianRecord`]s.
///
/// "Valid" means all three position components are finite (neither NaN nor Inf).
/// Mean values are computed exclusively over valid records.
pub fn compute_conversion_stats(records: &[GaussianRecord]) -> ConversionStats {
    let num_records = records.len();
    let sh_degree = records.first().map(|r| r.sh_degree()).unwrap_or(0);

    let mut num_nan = 0_usize;
    let mut num_inf = 0_usize;
    let mut num_valid = 0_usize;
    let mut sum_opacity = 0.0_f32;
    let mut sum_max_scale = 0.0_f32;

    for r in records {
        let has_nan = r.position.iter().any(|v| v.is_nan());
        let has_inf = r.position.iter().any(|v| v.is_infinite());

        if has_nan {
            num_nan += 1;
        } else if has_inf {
            num_inf += 1;
        } else {
            num_valid += 1;
            sum_opacity += r.activated_opacity();
            sum_max_scale += r.max_scale();
        }
    }

    let mean_opacity = if num_valid > 0 {
        sum_opacity / num_valid as f32
    } else {
        0.0
    };
    let mean_max_scale = if num_valid > 0 {
        sum_max_scale / num_valid as f32
    } else {
        0.0
    };

    ConversionStats {
        num_records,
        num_valid,
        num_nan,
        num_inf,
        mean_opacity,
        mean_max_scale,
        sh_degree,
    }
}

/// Validate a single [`GaussianRecord`] and return a list of human-readable issues.
///
/// An empty list means the record is valid.  Checks performed:
/// - All position components are finite.
/// - All log_scale components are finite.
/// - All rotation components are finite.
/// - The quaternion is approximately unit-length (|q|² ∈ [0.99, 1.01]).
/// - Opacity is finite.
/// - All sh_dc components are finite.
pub fn validate_record(record: &GaussianRecord) -> Vec<String> {
    let mut issues = Vec::new();

    // Position
    for (i, &v) in record.position.iter().enumerate() {
        if v.is_nan() {
            issues.push(format!("position[{i}] is NaN"));
        } else if v.is_infinite() {
            issues.push(format!("position[{i}] is Inf"));
        }
    }

    // Log-scale
    for (i, &v) in record.log_scale.iter().enumerate() {
        if !v.is_finite() {
            issues.push(format!("log_scale[{i}] is not finite ({v})"));
        }
    }

    // Rotation
    for (i, &v) in record.rotation.iter().enumerate() {
        if !v.is_finite() {
            issues.push(format!("rotation[{i}] is not finite ({v})"));
        }
    }

    // Unit quaternion check (|q|² should be ≈ 1 ± 0.01).
    let qx = record.rotation[0] as f64;
    let qy = record.rotation[1] as f64;
    let qz = record.rotation[2] as f64;
    let qw = record.rotation[3] as f64;
    let norm_sq = qx * qx + qy * qy + qz * qz + qw * qw;
    if (norm_sq - 1.0).abs() > 0.01 {
        issues.push(format!(
            "quaternion is not unit-length: |q|² = {norm_sq:.6}"
        ));
    }

    // Opacity
    if !record.opacity.is_finite() {
        issues.push(format!("opacity is not finite ({})", record.opacity));
    }

    // sh_dc
    for (i, &v) in record.sh_dc.iter().enumerate() {
        if !v.is_finite() {
            issues.push(format!("sh_dc[{i}] is not finite ({v})"));
        }
    }

    issues
}

/// Filter a [`Vec<GaussianRecord>`] to retain only records that pass
/// [`validate_record`] (i.e., those for which the issue list is empty).
pub fn filter_valid(records: Vec<GaussianRecord>) -> Vec<GaussianRecord> {
    records
        .into_iter()
        .filter(|r| validate_record(r).is_empty())
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: a clean, unit-quaternion Gaussian at the origin.
    fn make_record(x: f32, y: f32, z: f32) -> GaussianRecord {
        GaussianRecord::new(
            [x, y, z],
            [-1.0, -2.0, -0.5],
            [0.0, 0.0, 0.0, 1.0],
            0.0,
            [0.1, -0.2, 0.3],
        )
    }

    // -----------------------------------------------------------------------
    // GaussianRecord methods
    // -----------------------------------------------------------------------

    #[test]
    fn activated_opacity_at_zero_is_half() {
        let r = make_record(0.0, 0.0, 0.0);
        let act = r.activated_opacity();
        assert!(
            (act - 0.5).abs() < 1e-6,
            "sigmoid(0) should be 0.5, got {act}"
        );
    }

    #[test]
    fn activated_opacity_at_large_positive_is_near_one() {
        let mut r = make_record(0.0, 0.0, 0.0);
        r.opacity = 100.0;
        let act = r.activated_opacity();
        assert!(
            (act - 1.0).abs() < 1e-4,
            "sigmoid(100) should be ≈1.0, got {act}"
        );
    }

    #[test]
    fn scale_from_log_scale() {
        let mut r = make_record(0.0, 0.0, 0.0);
        r.log_scale = [0.0, 1.0_f32.ln(), 2.0_f32.ln()];
        let s = r.scale();
        assert!((s[0] - 1.0).abs() < 1e-5);
        assert!((s[1] - 1.0).abs() < 1e-5);
        assert!((s[2] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn base_color_from_sh_dc() {
        let mut r = make_record(0.0, 0.0, 0.0);
        // sh_dc = 0 → base_color = 0.5 + 0 = 0.5
        r.sh_dc = [0.0, 0.0, 0.0];
        let c = r.base_color();
        assert!((c[0] - 0.5).abs() < 1e-5);
        assert!((c[1] - 0.5).abs() < 1e-5);
        assert!((c[2] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn base_color_clamped_to_unit_interval() {
        let mut r = make_record(0.0, 0.0, 0.0);
        // Large positive sh_dc → should clamp to 1.0.
        r.sh_dc = [10.0, -10.0, 0.0];
        let c = r.base_color();
        assert!((c[0] - 1.0).abs() < 1e-5, "expected 1.0, got {}", c[0]);
        assert!((c[1] - 0.0).abs() < 1e-5, "expected 0.0, got {}", c[1]);
    }

    #[test]
    fn sh_degree_from_sh_rest_len() {
        let base = make_record(0.0, 0.0, 0.0);
        assert_eq!(base.sh_degree(), 0);

        let d1 = base.clone().with_sh_rest(vec![0.0; 9]);
        assert_eq!(d1.sh_degree(), 1);

        let d2 = base.clone().with_sh_rest(vec![0.0; 24]);
        assert_eq!(d2.sh_degree(), 2);

        let d3 = base.clone().with_sh_rest(vec![0.0; 45]);
        assert_eq!(d3.sh_degree(), 3);

        // Unrecognised length → degree 0.
        let dunknown = base.with_sh_rest(vec![0.0; 5]);
        assert_eq!(dunknown.sh_degree(), 0);
    }

    #[test]
    fn volume_is_positive() {
        let r = make_record(1.0, 2.0, 3.0);
        let vol = r.volume();
        assert!(vol > 0.0, "volume should be positive, got {vol}");
    }

    #[test]
    fn max_scale_picks_largest() {
        let mut r = make_record(0.0, 0.0, 0.0);
        // log_scales: 0, 1, 2 → scales: 1, e, e² → max is e²
        r.log_scale = [0.0, 1.0, 2.0];
        let ms = r.max_scale();
        assert!((ms - 2.0_f32.exp()).abs() < 1e-4, "expected e^2, got {ms}");
    }

    // -----------------------------------------------------------------------
    // sh_rest_size helper
    // -----------------------------------------------------------------------

    #[test]
    fn sh_rest_size_degree_0_is_0() {
        assert_eq!(sh_rest_size(0), 0);
    }

    #[test]
    fn sh_rest_size_degree_1_is_9() {
        assert_eq!(sh_rest_size(1), 9);
    }

    #[test]
    fn sh_rest_size_degree_2_is_24() {
        assert_eq!(sh_rest_size(2), 24);
    }

    #[test]
    fn sh_rest_size_degree_3_is_45() {
        assert_eq!(sh_rest_size(3), 45);
    }

    #[test]
    fn sh_rest_size_unrecognised_degree_is_0() {
        assert_eq!(sh_rest_size(4), 0);
        assert_eq!(sh_rest_size(99), 0);
    }

    #[test]
    fn sh_rest_size_is_total_over_every_usize() {
        // Regression: the formula `((degree+1)² − 1) × 3` used to be applied
        // to *any* non-zero degree. A header-supplied `degree` near
        // `u32::MAX` squares past `usize::MAX`, which panics on overflow in
        // debug builds instead of returning a value at all.
        assert_eq!(sh_rest_size(u32::MAX as usize), 0);
        assert_eq!(sh_rest_size(usize::MAX), 0);
        assert_eq!(sh_rest_size(usize::MAX / 2), 0);
    }

    #[test]
    fn from_binary_rejects_an_out_of_range_sh_degree_header() {
        // Regression: `sh_rest_size` maps an unrecognised degree to 0, so a
        // header claiming degree 4+ used to parse as "zero sh_rest
        // coefficients" — silently dropping SH data and mis-striding every
        // record — rather than reporting the unsupported degree.
        let bytes = to_binary(&[make_record(1.0, 2.0, 3.0)]).expect("degree-0 record is valid");
        for bad_degree in [4u32, 7, 99, u32::MAX] {
            let mut corrupt = bytes.clone();
            corrupt[16..20].copy_from_slice(&bad_degree.to_le_bytes());
            let err = from_binary(&corrupt)
                .expect_err("an out-of-range SH degree in the header must be rejected");
            assert!(
                matches!(err, ConvertError::InvalidFormat(ref m) if m.contains("SH degree")),
                "unexpected error for degree {bad_degree}: {err:?}"
            );
        }
        // The untouched header still round-trips.
        assert_eq!(from_binary(&bytes).expect("valid blob").len(), 1);
    }

    #[test]
    fn degree_for_rest_len_round_trips_sh_rest_size() {
        for degree in 0..=3usize {
            let len = sh_rest_size(degree);
            assert_eq!(degree_for_rest_len(len), Some(degree));
        }
        assert_eq!(
            degree_for_rest_len(5),
            None,
            "5 is not a valid SH rest length"
        );
    }

    // -----------------------------------------------------------------------
    // CSV
    // -----------------------------------------------------------------------

    #[test]
    fn to_csv_header_and_data() {
        let records = vec![make_record(1.0, 2.0, 3.0)];
        let bytes = to_csv(&records);
        let text = std::str::from_utf8(&bytes).expect("valid UTF-8");
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines[0], "x,y,z,sx,sy,sz,qx,qy,qz,qw,opacity_logit,r,g,b");
        assert_eq!(lines.len(), 2, "header + one data row");
        // Confirm 14 columns in the data row.
        assert_eq!(lines[1].split(',').count(), 14);
    }

    #[test]
    fn from_csv_positions_match() {
        let records = vec![make_record(1.0, 2.0, 3.0), make_record(-1.0, 0.0, 5.5)];
        let bytes = to_csv(&records);
        let parsed = from_csv(&bytes).expect("parse should succeed");
        assert_eq!(parsed.len(), 2);
        assert!((parsed[0].position[0] - 1.0).abs() < 1e-4);
        assert!((parsed[0].position[1] - 2.0).abs() < 1e-4);
        assert!((parsed[1].position[2] - 5.5).abs() < 1e-4);
    }

    #[test]
    fn from_csv_wrong_column_count_is_error() {
        // Provide a malformed CSV with only 5 columns.
        let bad = b"x,y,z,sx,sy,sz,qx,qy,qz,qw,opacity_logit,r,g,b\n1.0,2.0,3.0,4.0,5.0\n";
        let result = from_csv(bad);
        assert!(
            matches!(result, Err(ConvertError::DimensionError { .. })),
            "expected DimensionError, got {result:?}"
        );
    }

    // -----------------------------------------------------------------------
    // JSON
    // -----------------------------------------------------------------------

    #[test]
    fn to_json_valid_utf8_starts_with_bracket() {
        let records = vec![make_record(0.0, 1.0, 2.0)];
        let bytes = to_json(&records);
        let text = std::str::from_utf8(&bytes).expect("valid UTF-8");
        let trimmed = text.trim();
        assert!(trimmed.starts_with('['), "JSON should start with '['");
        assert!(trimmed.ends_with(']'), "JSON should end with ']'");
    }

    #[test]
    fn from_json_smoke_roundtrip() {
        let records = vec![make_record(1.5, -2.5, 3.0)];
        let bytes = to_json(&records);
        let parsed = from_json(&bytes).expect("from_json should succeed");
        assert_eq!(parsed.len(), 1);
        assert!((parsed[0].position[0] - 1.5).abs() < 1e-4);
        assert!((parsed[0].position[1] - (-2.5)).abs() < 1e-4);
    }

    #[test]
    fn from_json_empty_array() {
        let bytes = b"[]\n";
        let parsed = from_json(bytes).expect("empty array is valid");
        assert!(parsed.is_empty());
    }

    #[test]
    fn from_json_multiple_records() {
        let records = vec![make_record(0.0, 1.0, 2.0), make_record(3.0, 4.0, 5.0)];
        let bytes = to_json(&records);
        let parsed = from_json(&bytes).expect("from_json should succeed");
        assert_eq!(parsed.len(), 2);
        assert!((parsed[1].position[2] - 5.0).abs() < 1e-4);
    }

    #[test]
    fn from_json_multibyte_non_array_value_does_not_panic() {
        // Regression: `parse_f32_array`'s error path used to byte-slice the
        // input at index 40 for its error snippet, which panics if byte 40
        // lands inside a multibyte UTF-8 character. "position" here is a
        // long non-array (non-`[...]`) multibyte string, which used to
        // trigger exactly that panic; now it must return a clean ParseError.
        let bad = "[\n  {\n    \"position\": \"X日本語テストです日本語テストです日本語テストです\",\n    \"log_scale\": [0.0, 0.0, 0.0],\n    \"rotation\": [0.0, 0.0, 0.0, 1.0],\n    \"opacity\": 0.0,\n    \"sh_dc\": [0.0, 0.0, 0.0],\n    \"sh_rest\": []\n  }\n]\n";
        let result = from_json(bad.as_bytes());
        assert!(
            matches!(result, Err(ConvertError::ParseError(_))),
            "expected ParseError (not a panic), got {result:?}"
        );
    }

    // -----------------------------------------------------------------------
    // Binary
    // -----------------------------------------------------------------------

    #[test]
    fn binary_header_new_is_valid() {
        let h = BinaryHeader::new(100, 1);
        assert!(h.is_valid());
        assert_eq!(h.num_gaussians, 100);
        assert_eq!(h.sh_degree, 1);
        assert_eq!(h.version, BINARY_VERSION);
    }

    #[test]
    fn binary_header_wrong_magic_is_invalid() {
        let mut h = BinaryHeader::new(10, 0);
        h.magic = *b"BADMAGIC";
        assert!(!h.is_valid());
    }

    #[test]
    fn to_binary_size_is_correct_degree_0() {
        let records = vec![make_record(0.0, 0.0, 0.0), make_record(1.0, 2.0, 3.0)];
        let bytes = to_binary(&records).expect("degree-0 records are valid");
        // 24 header + 2 * 56 bytes (14 f32s, no sh_rest)
        assert_eq!(bytes.len(), 24 + 2 * 56);
    }

    #[test]
    fn to_binary_size_is_correct_degree_1() {
        let r = make_record(0.0, 0.0, 0.0).with_sh_rest(vec![0.0; 9]);
        let bytes = to_binary(&[r]).expect("degree-1 record is valid");
        // 24 header + 1 * (56 + 9*4) = 24 + 92
        assert_eq!(bytes.len(), 24 + 92);
    }

    #[test]
    fn to_binary_inconsistent_sh_rest_len_is_error() {
        let r0 = make_record(0.0, 0.0, 0.0).with_sh_rest(vec![0.0; 9]);
        let r1 = make_record(1.0, 1.0, 1.0).with_sh_rest(vec![0.0; 24]);
        let result = to_binary(&[r0, r1]);
        assert!(
            matches!(result, Err(ConvertError::DimensionError { .. })),
            "expected DimensionError, got {result:?}"
        );
    }

    #[test]
    fn to_binary_unrecognised_sh_rest_len_is_error() {
        // 5 is not 0/9/24/45 for any SH degree.
        let r = make_record(0.0, 0.0, 0.0).with_sh_rest(vec![0.0; 5]);
        let result = to_binary(&[r]);
        assert!(
            matches!(result, Err(ConvertError::InvalidFormat(_))),
            "expected InvalidFormat, got {result:?}"
        );
    }

    #[test]
    fn to_binary_never_silently_drops_sh_rest_coefficients() {
        // Regression for the original bug: a real degree-3 record (45 rest
        // coefficients) used to compute sh_degree()==0 (48 was the wrongly
        // expected length) and silently write zero sh_rest floats. Now the
        // full 45 coefficients must round-trip exactly.
        let r = make_record(0.0, 0.0, 0.0).with_sh_rest((0..45).map(|i| i as f32 * 0.01).collect());
        let bytes = to_binary(std::slice::from_ref(&r)).expect("degree-3 record is valid");
        let parsed = from_binary(&bytes).expect("roundtrip should succeed");
        assert_eq!(parsed[0].sh_rest.len(), 45);
        for (a, b) in parsed[0].sh_rest.iter().zip(r.sh_rest.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn from_binary_roundtrip_positions() {
        let records = vec![make_record(1.0, 2.0, 3.0), make_record(-4.0, 5.0, -6.0)];
        let bytes = to_binary(&records).expect("degree-0 records are valid");
        let parsed = from_binary(&bytes).expect("roundtrip should succeed");
        assert_eq!(parsed.len(), 2);
        assert!((parsed[0].position[0] - 1.0).abs() < 1e-5);
        assert!((parsed[1].position[1] - 5.0).abs() < 1e-5);
        assert!((parsed[1].position[2] - (-6.0)).abs() < 1e-5);
    }

    #[test]
    fn from_binary_roundtrip_full_fields() {
        let r = GaussianRecord::new(
            [1.5, -2.5, 0.5],
            [-0.5, -1.0, -1.5],
            [0.1, 0.2, 0.3, 0.9],
            -0.7,
            [0.5, -0.5, 0.1],
        );
        let bytes = to_binary(std::slice::from_ref(&r)).expect("degree-0 record is valid");
        let parsed = from_binary(&bytes).expect("roundtrip");
        let p = &parsed[0];
        assert!((p.opacity - r.opacity).abs() < 1e-5);
        assert!((p.log_scale[2] - r.log_scale[2]).abs() < 1e-5);
        assert!((p.sh_dc[1] - r.sh_dc[1]).abs() < 1e-5);
    }

    #[test]
    fn from_binary_invalid_magic_is_error() {
        let mut bytes = to_binary(&[make_record(0.0, 0.0, 0.0)]).expect("degree-0 record is valid");
        // Corrupt the magic.
        bytes[0] = b'X';
        let result = from_binary(&bytes);
        assert!(
            matches!(result, Err(ConvertError::InvalidFormat(_))),
            "expected InvalidFormat, got {result:?}"
        );
    }

    #[test]
    fn from_binary_degree_0_no_sh_rest() {
        let r = make_record(1.0, 2.0, 3.0);
        assert_eq!(r.sh_rest.len(), 0);
        let bytes = to_binary(&[r]).expect("degree-0 record is valid");
        let parsed = from_binary(&bytes).expect("should parse");
        assert_eq!(parsed[0].sh_rest.len(), 0);
    }

    // -----------------------------------------------------------------------
    // FileFormat
    // -----------------------------------------------------------------------

    #[test]
    fn file_format_from_extension_known() {
        assert_eq!(FileFormat::from_extension("csv"), Some(FileFormat::Csv));
        assert_eq!(FileFormat::from_extension("json"), Some(FileFormat::Json));
        assert_eq!(FileFormat::from_extension("jsonl"), Some(FileFormat::Json));
        assert_eq!(FileFormat::from_extension("bin"), Some(FileFormat::Binary));
        assert_eq!(
            FileFormat::from_extension("oxigaf"),
            Some(FileFormat::Binary)
        );
    }

    #[test]
    fn file_format_from_extension_unknown() {
        assert_eq!(FileFormat::from_extension("ply"), None);
        assert_eq!(FileFormat::from_extension("txt"), None);
        assert_eq!(FileFormat::from_extension(""), None);
    }

    #[test]
    fn file_format_from_magic_detects_binary() {
        let bytes = to_binary(&[make_record(0.0, 0.0, 0.0)]).expect("degree-0 record is valid");
        assert_eq!(FileFormat::from_magic(&bytes), FileFormat::Binary);
    }

    #[test]
    fn file_format_from_magic_detects_json() {
        let bytes = to_json(&[make_record(0.0, 0.0, 0.0)]);
        assert_eq!(FileFormat::from_magic(&bytes), FileFormat::Json);
    }

    #[test]
    fn file_format_from_magic_detects_csv() {
        let bytes = to_csv(&[make_record(0.0, 0.0, 0.0)]);
        assert_eq!(FileFormat::from_magic(&bytes), FileFormat::Csv);
    }

    #[test]
    fn convert_binary_format_starts_with_magic() {
        let records = vec![make_record(0.0, 1.0, 2.0)];
        let out = convert(&records, FileFormat::Binary).expect("degree-0 records are valid");
        assert!(
            out.starts_with(&BINARY_MAGIC),
            "binary output should start with OXIGAF01 magic"
        );
    }

    // -----------------------------------------------------------------------
    // Validation and filtering
    // -----------------------------------------------------------------------

    #[test]
    fn validate_record_clean_is_empty() {
        let r = make_record(0.0, 1.0, 2.0);
        let issues = validate_record(&r);
        assert!(
            issues.is_empty(),
            "clean record should have no issues: {issues:?}"
        );
    }

    #[test]
    fn validate_record_nan_position_has_issue() {
        let mut r = make_record(0.0, 0.0, 0.0);
        r.position[1] = f32::NAN;
        let issues = validate_record(&r);
        assert!(
            issues.iter().any(|s| s.contains("NaN")),
            "expected NaN issue, got {issues:?}"
        );
    }

    #[test]
    fn validate_record_non_unit_quaternion_has_issue() {
        let mut r = make_record(0.0, 0.0, 0.0);
        // Set quaternion to (0, 0, 0, 2) — norm² = 4, far from 1.
        r.rotation = [0.0, 0.0, 0.0, 2.0];
        let issues = validate_record(&r);
        assert!(
            issues.iter().any(|s| s.contains("unit-length")),
            "expected quaternion issue, got {issues:?}"
        );
    }

    #[test]
    fn filter_valid_removes_invalid_records() {
        let good = make_record(1.0, 2.0, 3.0);
        let mut bad = make_record(0.0, 0.0, 0.0);
        bad.position[0] = f32::NAN;
        let records = vec![good, bad];
        let filtered = filter_valid(records);
        assert_eq!(filtered.len(), 1);
        assert!(filtered[0].position[0].is_finite());
    }

    #[test]
    fn compute_conversion_stats_counts_correct() {
        let good = make_record(0.0, 0.0, 0.0);
        let mut bad_nan = make_record(0.0, 0.0, 0.0);
        bad_nan.position[0] = f32::NAN;
        let mut bad_inf = make_record(0.0, 0.0, 0.0);
        bad_inf.position[1] = f32::INFINITY;

        let records = vec![good, bad_nan, bad_inf];
        let stats = compute_conversion_stats(&records);

        assert_eq!(stats.num_records, 3);
        assert_eq!(stats.num_valid, 1);
        assert_eq!(stats.num_nan, 1);
        assert_eq!(stats.num_inf, 1);
    }

    #[test]
    fn compute_conversion_stats_mean_opacity() {
        // sigmoid(0) = 0.5; with two identical records, mean should be 0.5.
        let records = vec![make_record(0.0, 0.0, 0.0), make_record(1.0, 0.0, 0.0)];
        let stats = compute_conversion_stats(&records);
        assert!((stats.mean_opacity - 0.5).abs() < 1e-5);
    }
}
