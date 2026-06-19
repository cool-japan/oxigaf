//! Quantized compressed Gaussian storage for memory reduction.
//!
//! Provides per-field quantization (positions → i16, rotations → i8,
//! scales → u8, opacities → u8) and LBG vector quantization for
//! spherical-harmonics coefficients.

use thiserror::Error;

// ────────────────────────────────────────────────────────────────
// Error type
// ────────────────────────────────────────────────────────────────

/// Errors that can occur during Gaussian compression / decompression.
#[derive(Debug, Error)]
pub enum CompressionError {
    /// Input slice is empty when at least one element is required.
    #[error("empty input: at least one Gaussian is required")]
    EmptyInput,

    /// Two arrays have different lengths when they must match.
    #[error("length mismatch: expected {expected}, got {got}")]
    LengthMismatch { expected: usize, got: usize },

    /// Codebook size is invalid.
    #[error("invalid codebook size {size}: {reason}")]
    InvalidCodebookSize { size: usize, reason: &'static str },
}

// ────────────────────────────────────────────────────────────────
// Scene bounds
// ────────────────────────────────────────────────────────────────

/// Axis-aligned bounding box of the scene, used for position quantization.
#[derive(Debug, Clone)]
pub struct SceneBounds {
    /// Minimum corner [x_min, y_min, z_min].
    pub min: [f32; 3],
    /// Maximum corner [x_max, y_max, z_max].
    pub max: [f32; 3],
}

impl SceneBounds {
    /// Compute bounds from a slice of positions.  Returns `None` if empty.
    pub fn from_positions(positions: &[[f32; 3]]) -> Option<Self> {
        if positions.is_empty() {
            return None;
        }
        let mut mn = positions[0];
        let mut mx = positions[0];
        for p in positions.iter().skip(1) {
            for i in 0..3 {
                if p[i] < mn[i] {
                    mn[i] = p[i];
                }
                if p[i] > mx[i] {
                    mx[i] = p[i];
                }
            }
        }
        Some(Self { min: mn, max: mx })
    }

    /// Expand each bound outward by `margin` (useful to avoid clipping).
    pub fn expand(mut self, margin: f32) -> Self {
        for i in 0..3 {
            self.min[i] -= margin;
            self.max[i] += margin;
        }
        self
    }

    /// Returns `true` if `p` lies within [min, max] (inclusive).
    pub fn contains(&self, p: &[f32; 3]) -> bool {
        for ((&pv, &mn), &mx) in p.iter().zip(self.min.iter()).zip(self.max.iter()) {
            if pv < mn || pv > mx {
                return false;
            }
        }
        true
    }
}

// ────────────────────────────────────────────────────────────────
// Scalar quantization helpers
// ────────────────────────────────────────────────────────────────

/// Map `v` in `[lo, hi]` to `i16` in `[i16::MIN, i16::MAX]`.
///
/// When `lo == hi` the range is degenerate; returns 0.
fn quantize_i16(v: f32, lo: f32, hi: f32) -> i16 {
    if (hi - lo).abs() < f32::EPSILON {
        return 0;
    }
    let t = ((v - lo) / (hi - lo)).clamp(0.0, 1.0);
    // Map [0,1] → [i16::MIN, i16::MAX]
    let scaled = t * 65535.0 - 32768.0;
    scaled.clamp(i16::MIN as f32, i16::MAX as f32) as i16
}

/// Dequantize `i16` back to `f32` in `[lo, hi]`.
fn dequantize_i16(q: i16, lo: f32, hi: f32) -> f32 {
    let t = (q as f32 + 32768.0) / 65535.0;
    lo + t * (hi - lo)
}

/// Map `v` in `[-1, 1]` to `i8` in `[i8::MIN, i8::MAX]`.
///
/// Values outside `[-1, 1]` are clamped.
fn quantize_i8(v: f32) -> i8 {
    let t = (v.clamp(-1.0, 1.0) + 1.0) / 2.0; // [0, 1]
    let scaled = t * 255.0 - 128.0;
    scaled.clamp(i8::MIN as f32, i8::MAX as f32) as i8
}

/// Dequantize `i8` back to `f32` in `[-1, 1]`.
fn dequantize_i8(q: i8) -> f32 {
    let t = (q as f32 + 128.0) / 255.0; // [0, 1]
    t * 2.0 - 1.0
}

/// Map `v` in `[lo, hi]` to `u8` in `[0, 255]`.
///
/// When `lo == hi` the range is degenerate; returns 128.
fn quantize_u8(v: f32, lo: f32, hi: f32) -> u8 {
    if (hi - lo).abs() < f32::EPSILON {
        return 128;
    }
    let t = ((v - lo) / (hi - lo)).clamp(0.0, 1.0);
    (t * 255.0).clamp(0.0, 255.0) as u8
}

/// Dequantize `u8` back to `f32` in `[lo, hi]`.
fn dequantize_u8(q: u8, lo: f32, hi: f32) -> f32 {
    let t = q as f32 / 255.0;
    lo + t * (hi - lo)
}

// ────────────────────────────────────────────────────────────────
// RMS error helper
// ────────────────────────────────────────────────────────────────

fn rms_error_f32(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() {
        return 0.0;
    }
    let mse = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        / a.len() as f32;
    mse.sqrt()
}

// ────────────────────────────────────────────────────────────────
// Compressed positions
// ────────────────────────────────────────────────────────────────

/// Positions quantized to 3 × i16 per Gaussian.
#[derive(Debug, Clone)]
pub struct CompressedPositions {
    /// Quantized position data.
    pub data: Vec<[i16; 3]>,
    /// Scene bounds used for quantization.
    pub bounds: SceneBounds,
}

impl CompressedPositions {
    /// Compress positions using scene bounds.
    pub fn compress(positions: &[[f32; 3]], bounds: &SceneBounds) -> Self {
        let data = positions
            .iter()
            .map(|p| {
                [
                    quantize_i16(p[0], bounds.min[0], bounds.max[0]),
                    quantize_i16(p[1], bounds.min[1], bounds.max[1]),
                    quantize_i16(p[2], bounds.min[2], bounds.max[2]),
                ]
            })
            .collect();
        Self {
            data,
            bounds: bounds.clone(),
        }
    }

    /// Decompress back to f32 positions.
    pub fn decompress(&self) -> Vec<[f32; 3]> {
        self.data
            .iter()
            .map(|q| {
                [
                    dequantize_i16(q[0], self.bounds.min[0], self.bounds.max[0]),
                    dequantize_i16(q[1], self.bounds.min[1], self.bounds.max[1]),
                    dequantize_i16(q[2], self.bounds.min[2], self.bounds.max[2]),
                ]
            })
            .collect()
    }

    /// Number of compressed Gaussians.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` if no Gaussians are stored.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Total bytes used by the quantized data.
    pub fn bytes(&self) -> usize {
        self.data.len() * 6
    }
}

// ────────────────────────────────────────────────────────────────
// Compressed rotations
// ────────────────────────────────────────────────────────────────

/// Unit quaternions quantized to 4 × i8 per Gaussian.
#[derive(Debug, Clone)]
pub struct CompressedRotations {
    /// Quantized quaternion data (components in [-1, 1]).
    pub data: Vec<[i8; 4]>,
}

impl CompressedRotations {
    /// Compress quaternion rotations.
    pub fn compress(rotations: &[[f32; 4]]) -> Self {
        let data = rotations
            .iter()
            .map(|r| {
                [
                    quantize_i8(r[0]),
                    quantize_i8(r[1]),
                    quantize_i8(r[2]),
                    quantize_i8(r[3]),
                ]
            })
            .collect();
        Self { data }
    }

    /// Decompress back to f32 quaternions.
    pub fn decompress(&self) -> Vec<[f32; 4]> {
        self.data
            .iter()
            .map(|q| {
                [
                    dequantize_i8(q[0]),
                    dequantize_i8(q[1]),
                    dequantize_i8(q[2]),
                    dequantize_i8(q[3]),
                ]
            })
            .collect()
    }

    /// Number of compressed Gaussians.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` if no Gaussians are stored.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Total bytes used by the quantized data.
    pub fn bytes(&self) -> usize {
        self.data.len() * 4
    }
}

// ────────────────────────────────────────────────────────────────
// Compressed scales
// ────────────────────────────────────────────────────────────────

/// Log-scales quantized to 3 × u8 per Gaussian.
#[derive(Debug, Clone)]
pub struct CompressedScales {
    /// Quantized log-scale data.
    pub data: Vec<[u8; 3]>,
    /// Minimum log-scale value (across all dimensions and Gaussians).
    pub log_min: f32,
    /// Maximum log-scale value.
    pub log_max: f32,
}

impl CompressedScales {
    /// Compress log-scale values.
    pub fn compress(scales: &[[f32; 3]]) -> Self {
        if scales.is_empty() {
            return Self {
                data: Vec::new(),
                log_min: 0.0,
                log_max: 0.0,
            };
        }
        // Find global min/max across all axes.
        let mut log_min = f32::INFINITY;
        let mut log_max = f32::NEG_INFINITY;
        for s in scales {
            for &v in s {
                if v < log_min {
                    log_min = v;
                }
                if v > log_max {
                    log_max = v;
                }
            }
        }
        let data = scales
            .iter()
            .map(|s| {
                [
                    quantize_u8(s[0], log_min, log_max),
                    quantize_u8(s[1], log_min, log_max),
                    quantize_u8(s[2], log_min, log_max),
                ]
            })
            .collect();
        Self {
            data,
            log_min,
            log_max,
        }
    }

    /// Decompress back to f32 log-scales.
    pub fn decompress(&self) -> Vec<[f32; 3]> {
        self.data
            .iter()
            .map(|q| {
                [
                    dequantize_u8(q[0], self.log_min, self.log_max),
                    dequantize_u8(q[1], self.log_min, self.log_max),
                    dequantize_u8(q[2], self.log_min, self.log_max),
                ]
            })
            .collect()
    }

    /// Number of compressed Gaussians.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` if no Gaussians are stored.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Total bytes used by the quantized data.
    pub fn bytes(&self) -> usize {
        self.data.len() * 3
    }
}

// ────────────────────────────────────────────────────────────────
// Compressed opacities
// ────────────────────────────────────────────────────────────────

/// Logit-space opacities quantized to 1 × u8 per Gaussian.
#[derive(Debug, Clone)]
pub struct CompressedOpacities {
    /// Quantized opacity data.
    pub data: Vec<u8>,
    /// Minimum logit value.
    pub logit_min: f32,
    /// Maximum logit value.
    pub logit_max: f32,
}

impl CompressedOpacities {
    /// Compress logit-space opacities.
    pub fn compress(opacities: &[f32]) -> Self {
        if opacities.is_empty() {
            return Self {
                data: Vec::new(),
                logit_min: 0.0,
                logit_max: 0.0,
            };
        }
        let mut logit_min = f32::INFINITY;
        let mut logit_max = f32::NEG_INFINITY;
        for &v in opacities {
            if v < logit_min {
                logit_min = v;
            }
            if v > logit_max {
                logit_max = v;
            }
        }
        let data = opacities
            .iter()
            .map(|&v| quantize_u8(v, logit_min, logit_max))
            .collect();
        Self {
            data,
            logit_min,
            logit_max,
        }
    }

    /// Decompress back to f32 logit-space opacities.
    pub fn decompress(&self) -> Vec<f32> {
        self.data
            .iter()
            .map(|&q| dequantize_u8(q, self.logit_min, self.logit_max))
            .collect()
    }

    /// Number of compressed Gaussians.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` if no Gaussians are stored.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Total bytes used by the quantized data.
    pub fn bytes(&self) -> usize {
        self.data.len()
    }
}

// ────────────────────────────────────────────────────────────────
// SH codebook (LBG vector quantization)
// ────────────────────────────────────────────────────────────────

/// LBG-style codebook for spherical-harmonics coefficients.
#[derive(Debug, Clone)]
pub struct ShCodebook {
    /// Codebook entries; each has length `sh_len`.
    pub entries: Vec<Vec<f32>>,
    /// Number of SH coefficients per Gaussian.
    pub sh_len: usize,
}

impl ShCodebook {
    /// Build a codebook using simplified LBG (binary split + k-means).
    ///
    /// `k` must be a power of two between 64 and 256 (inclusive).
    /// Returns `None` if `sh_coeffs` is empty.
    pub fn build(sh_coeffs: &[Vec<f32>], k: usize) -> Option<Self> {
        if sh_coeffs.is_empty() {
            return None;
        }

        // Validate k is power of 2 and in [2, 256].
        if !(2..=256).contains(&k) || (k & (k - 1)) != 0 {
            return None;
        }

        let sh_len = sh_coeffs[0].len();
        let n = sh_coeffs.len();

        // Cap effective k to number of inputs to avoid degenerate codebooks.
        let effective_k = k.min(n);

        // Start: single centroid = mean of all vectors.
        let mean = compute_mean(sh_coeffs, sh_len);
        let mut codebook: Vec<Vec<f32>> = vec![mean];

        // Binary split until we reach effective_k.
        while codebook.len() < effective_k {
            let current_len = codebook.len();
            let mut new_entries: Vec<Vec<f32>> = Vec::with_capacity(current_len * 2);
            for entry in &codebook {
                let (a, b) = split_entry(entry);
                new_entries.push(a);
                new_entries.push(b);
            }
            codebook = new_entries;
            // Trim if we overshoot (shouldn't happen with power-of-2 k).
            codebook.truncate(effective_k);
            // Refine with k-means.
            codebook = kmeans_refine(sh_coeffs, codebook, sh_len, 20);
        }

        Some(Self {
            entries: codebook,
            sh_len,
        })
    }

    /// Encode each SH vector as an index into the codebook.
    pub fn encode(&self, sh_coeffs: &[Vec<f32>]) -> Vec<u8> {
        sh_coeffs
            .iter()
            .map(|v| {
                let mut best_idx = 0usize;
                let mut best_dist = f32::INFINITY;
                for (i, entry) in self.entries.iter().enumerate() {
                    let d = sq_dist(v, entry);
                    if d < best_dist {
                        best_dist = d;
                        best_idx = i;
                    }
                }
                // Safe: entries.len() <= 256 per construction.
                (best_idx & 0xFF) as u8
            })
            .collect()
    }

    /// Decode indices back to SH vectors.
    pub fn decode(&self, indices: &[u8]) -> Vec<Vec<f32>> {
        indices
            .iter()
            .map(|&idx| {
                let i = (idx as usize).min(self.entries.len().saturating_sub(1));
                self.entries[i].clone()
            })
            .collect()
    }

    /// Total bytes used by the codebook (f32 storage).
    pub fn bytes(&self) -> usize {
        self.entries.len() * self.sh_len * 4
    }
}

// ── LBG internals ──────────────────────────────────────────────

/// Compute the mean vector of a slice of equal-length vectors.
fn compute_mean(vecs: &[Vec<f32>], sh_len: usize) -> Vec<f32> {
    let mut sum = vec![0.0f32; sh_len];
    for v in vecs {
        for (s, &x) in sum.iter_mut().zip(v.iter()) {
            *s += x;
        }
    }
    let n = vecs.len() as f32;
    sum.iter().map(|&s| s / n).collect()
}

/// Split a codebook entry into two by nudging ± epsilon on every dimension.
fn split_entry(entry: &[f32]) -> (Vec<f32>, Vec<f32>) {
    const EPS: f32 = 1e-4;
    let a = entry.iter().map(|&x| x + EPS).collect();
    let b = entry.iter().map(|&x| x - EPS).collect();
    (a, b)
}

/// Squared Euclidean distance between two equal-length vectors.
fn sq_dist(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y) * (x - y))
        .sum()
}

/// Assign each data vector to its nearest centroid.
/// Returns `(assignments, per-centroid total distortion)`.
fn assign_to_centroids(data: &[Vec<f32>], codebook: &[Vec<f32>]) -> (Vec<usize>, Vec<f32>) {
    let k = codebook.len();
    let mut assignments = vec![0usize; data.len()];
    let mut distortions = vec![0.0f32; k];
    for (idx, v) in data.iter().enumerate() {
        let mut best_idx = 0usize;
        let mut best_dist = f32::INFINITY;
        for (i, entry) in codebook.iter().enumerate() {
            let d = sq_dist(v, entry);
            if d < best_dist {
                best_dist = d;
                best_idx = i;
            }
        }
        assignments[idx] = best_idx;
        distortions[best_idx] += best_dist;
    }
    (assignments, distortions)
}

/// Run at most `max_iter` iterations of k-means refinement.
///
/// Dead centroids (no assigned vectors) are reinitialized from the data
/// point that is farthest from its assigned centroid in the cluster with
/// the highest total distortion.  This ensures the codebook never
/// collapses to fewer entries than requested.
fn kmeans_refine(
    data: &[Vec<f32>],
    mut codebook: Vec<Vec<f32>>,
    sh_len: usize,
    max_iter: usize,
) -> Vec<Vec<f32>> {
    let k = codebook.len();
    if k == 0 || data.is_empty() {
        return codebook;
    }

    for _iter in 0..max_iter {
        let (assignments, distortions) = assign_to_centroids(data, &codebook);

        // Recompute centroids from assignments.
        let mut sums: Vec<Vec<f32>> = vec![vec![0.0f32; sh_len]; k];
        let mut counts: Vec<usize> = vec![0usize; k];
        for (idx, &c) in assignments.iter().enumerate() {
            for (s, &x) in sums[c].iter_mut().zip(data[idx].iter()) {
                *s += x;
            }
            counts[c] += 1;
        }

        let mut changed = false;

        // Handle dead centroids: reinitialize from largest-distortion cluster.
        let dead_centroid_indices: Vec<usize> = (0..k).filter(|&i| counts[i] == 0).collect();

        for dead_idx in dead_centroid_indices {
            // Find the cluster with the highest total distortion to steal from.
            let donor_cluster = (0..k)
                .filter(|&i| counts[i] > 1) // must have at least 2 points
                .max_by(|&a, &b| {
                    distortions[a]
                        .partial_cmp(&distortions[b])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            if let Some(donor) = donor_cluster {
                // Find the farthest data point from the donor centroid.
                let farthest_idx = assignments
                    .iter()
                    .enumerate()
                    .filter(|(_, &c)| c == donor)
                    .max_by(|(ia, _), (ib, _)| {
                        sq_dist(&data[*ia], &codebook[donor])
                            .partial_cmp(&sq_dist(&data[*ib], &codebook[donor]))
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(i, _)| i);
                if let Some(fi) = farthest_idx {
                    codebook[dead_idx] = data[fi].clone();
                    changed = true;
                }
            }
        }

        // Update live centroids.
        for i in 0..k {
            if counts[i] == 0 {
                continue;
            }
            let new_centroid: Vec<f32> = sums[i].iter().map(|&s| s / counts[i] as f32).collect();
            let delta = sq_dist(&new_centroid, &codebook[i]);
            if delta > 1e-10 {
                changed = true;
            }
            codebook[i] = new_centroid;
        }

        if !changed {
            break;
        }
    }

    codebook
}

// ────────────────────────────────────────────────────────────────
// SH compression enum
// ────────────────────────────────────────────────────────────────

/// Compressed representation of SH coefficients.
#[derive(Debug, Clone)]
pub enum ShCompressed {
    /// No compression — raw f32 per Gaussian.
    Raw(Vec<Vec<f32>>),
    /// Codebook-compressed with per-Gaussian indices.
    Codebook {
        /// The LBG codebook.
        codebook: ShCodebook,
        /// Per-Gaussian codebook index.
        indices: Vec<u8>,
    },
}

// ────────────────────────────────────────────────────────────────
// Compressed Gaussian model
// ────────────────────────────────────────────────────────────────

/// A fully compressed Gaussian model ready for storage or streaming.
#[derive(Debug, Clone)]
pub struct CompressedGaussianModel {
    /// Quantized positions.
    pub positions: CompressedPositions,
    /// Quantized rotations.
    pub rotations: CompressedRotations,
    /// Quantized log-scales.
    pub scales: CompressedScales,
    /// Quantized logit-opacities.
    pub opacities: CompressedOpacities,
    /// Compressed SH coefficients (raw or codebook).
    pub sh_compressed: ShCompressed,
    /// SH degree used (0–3).
    pub sh_degree: u32,
    /// Number of Gaussians.
    pub num_gaussians: usize,
}

impl CompressedGaussianModel {
    /// Total bytes occupied by all compressed fields.
    pub fn total_bytes(&self) -> usize {
        let sh_bytes = match &self.sh_compressed {
            ShCompressed::Raw(raw) => raw.iter().map(|v| v.len() * 4).sum::<usize>(),
            ShCompressed::Codebook { codebook, indices } => codebook.bytes() + indices.len(),
        };
        self.positions.bytes()
            + self.rotations.bytes()
            + self.scales.bytes()
            + self.opacities.bytes()
            + sh_bytes
    }
}

// ────────────────────────────────────────────────────────────────
// Decompressed arrays
// ────────────────────────────────────────────────────────────────

/// All Gaussian arrays after decompression.
pub struct DecompressedArrays {
    /// Positions.
    pub positions: Vec<[f32; 3]>,
    /// Rotation quaternions.
    pub rotations: Vec<[f32; 4]>,
    /// Log-scales.
    pub scales: Vec<[f32; 3]>,
    /// Logit-opacities.
    pub opacities: Vec<f32>,
    /// SH coefficients per Gaussian.
    pub sh_coeffs: Vec<Vec<f32>>,
}

// ────────────────────────────────────────────────────────────────
// Compression config
// ────────────────────────────────────────────────────────────────

/// Configuration for Gaussian compression.
#[derive(Debug, Clone)]
pub struct CompressionConfig {
    /// How much to expand scene bounds before quantizing (default: 0.1).
    pub position_margin: f32,
    /// Whether to use codebook for SH coefficients (default: true).
    pub use_sh_codebook: bool,
    /// Number of codebook entries; must be a power of two (default: 256).
    pub sh_codebook_size: usize,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            position_margin: 0.1,
            use_sh_codebook: true,
            sh_codebook_size: 256,
        }
    }
}

// ────────────────────────────────────────────────────────────────
// Compression statistics
// ────────────────────────────────────────────────────────────────

/// Statistics reported after compression.
#[derive(Debug, Clone)]
pub struct CompressionStats {
    /// Uncompressed size in bytes (f32 storage).
    pub original_bytes: usize,
    /// Compressed size in bytes.
    pub compressed_bytes: usize,
    /// `original_bytes / compressed_bytes`.
    pub compression_ratio: f32,
    /// RMS error from position quantization.
    pub position_error_rms: f32,
    /// RMS error from rotation quantization.
    pub rotation_error_rms: f32,
    /// RMS error from scale quantization.
    pub scale_error_rms: f32,
    /// RMS error from opacity quantization.
    pub opacity_error_rms: f32,
    /// RMS error from SH compression.
    pub sh_error_rms: f32,
}

impl CompressionStats {
    /// Format a human-readable summary.
    pub fn format_summary(&self) -> String {
        format!(
            "Compression: {:.1}× ({} → {} bytes)\n\
             RMS errors — pos: {:.6}, rot: {:.6}, scale: {:.6}, opacity: {:.6}, sh: {:.6}",
            self.compression_ratio,
            self.original_bytes,
            self.compressed_bytes,
            self.position_error_rms,
            self.rotation_error_rms,
            self.scale_error_rms,
            self.opacity_error_rms,
            self.sh_error_rms,
        )
    }
}

// ────────────────────────────────────────────────────────────────
// Main compression / decompression functions
// ────────────────────────────────────────────────────────────────

/// Compress all Gaussian arrays into a `CompressedGaussianModel`.
///
/// Returns the compressed model and compression statistics.
pub fn compress_gaussians(
    positions: &[[f32; 3]],
    rotations: &[[f32; 4]],
    scales: &[[f32; 3]],
    opacities: &[f32],
    sh_coeffs: &[Vec<f32>],
    sh_degree: u32,
    config: &CompressionConfig,
) -> Result<(CompressedGaussianModel, CompressionStats), CompressionError> {
    let n = positions.len();
    if n == 0 {
        return Err(CompressionError::EmptyInput);
    }
    // Validate lengths.
    for (name, got) in [
        ("rotations", rotations.len()),
        ("scales", scales.len()),
        ("opacities", opacities.len()),
    ] {
        if got != n {
            return Err(CompressionError::LengthMismatch { expected: n, got });
        }
        let _ = name;
    }
    if !sh_coeffs.is_empty() && sh_coeffs.len() != n {
        return Err(CompressionError::LengthMismatch {
            expected: n,
            got: sh_coeffs.len(),
        });
    }

    // Validate codebook size.
    if config.use_sh_codebook && !sh_coeffs.is_empty() {
        let k = config.sh_codebook_size;
        if !(2..=256).contains(&k) || (k & (k - 1)) != 0 {
            return Err(CompressionError::InvalidCodebookSize {
                size: k,
                reason: "must be a power of two between 2 and 256",
            });
        }
    }

    // Uncompressed size: f32 per element.
    let sh_len_per = if sh_coeffs.is_empty() {
        0
    } else {
        sh_coeffs[0].len()
    };
    let original_bytes = (n * 3 + n * 4 + n * 3 + n + n * sh_len_per) * 4;

    // ── Compress positions ───────────────────────────────────────
    let bounds = SceneBounds::from_positions(positions)
        .ok_or(CompressionError::EmptyInput)?
        .expand(config.position_margin);
    let comp_pos = CompressedPositions::compress(positions, &bounds);

    // ── Compress rotations ───────────────────────────────────────
    let comp_rot = CompressedRotations::compress(rotations);

    // ── Compress scales ──────────────────────────────────────────
    let comp_scales = CompressedScales::compress(scales);

    // ── Compress opacities ───────────────────────────────────────
    let comp_opacities = CompressedOpacities::compress(opacities);

    // ── Compress SH ─────────────────────────────────────────────
    let sh_compressed = if sh_coeffs.is_empty() {
        ShCompressed::Raw(Vec::new())
    } else if config.use_sh_codebook {
        match ShCodebook::build(sh_coeffs, config.sh_codebook_size) {
            Some(codebook) => {
                let indices = codebook.encode(sh_coeffs);
                ShCompressed::Codebook { codebook, indices }
            }
            None => ShCompressed::Raw(sh_coeffs.to_vec()),
        }
    } else {
        ShCompressed::Raw(sh_coeffs.to_vec())
    };

    // ── RMS errors ───────────────────────────────────────────────
    let pos_recon = comp_pos.decompress();
    let rot_recon = comp_rot.decompress();
    let scale_recon = comp_scales.decompress();
    let opacity_recon = comp_opacities.decompress();

    let pos_flat_orig: Vec<f32> = positions.iter().flat_map(|p| p.iter().copied()).collect();
    let pos_flat_recon: Vec<f32> = pos_recon.iter().flat_map(|p| p.iter().copied()).collect();
    let rot_flat_orig: Vec<f32> = rotations.iter().flat_map(|r| r.iter().copied()).collect();
    let rot_flat_recon: Vec<f32> = rot_recon.iter().flat_map(|r| r.iter().copied()).collect();
    let scale_flat_orig: Vec<f32> = scales.iter().flat_map(|s| s.iter().copied()).collect();
    let scale_flat_recon: Vec<f32> = scale_recon.iter().flat_map(|s| s.iter().copied()).collect();

    let position_error_rms = rms_error_f32(&pos_flat_orig, &pos_flat_recon);
    let rotation_error_rms = rms_error_f32(&rot_flat_orig, &rot_flat_recon);
    let scale_error_rms = rms_error_f32(&scale_flat_orig, &scale_flat_recon);
    let opacity_error_rms = rms_error_f32(opacities, &opacity_recon);

    let sh_error_rms = if sh_coeffs.is_empty() {
        0.0
    } else {
        let sh_recon = match &sh_compressed {
            ShCompressed::Raw(raw) => raw.clone(),
            ShCompressed::Codebook { codebook, indices } => codebook.decode(indices),
        };
        let orig_flat: Vec<f32> = sh_coeffs.iter().flat_map(|v| v.iter().copied()).collect();
        let recon_flat: Vec<f32> = sh_recon.iter().flat_map(|v| v.iter().copied()).collect();
        rms_error_f32(&orig_flat, &recon_flat)
    };

    // ── Build model ──────────────────────────────────────────────
    let model = CompressedGaussianModel {
        positions: comp_pos,
        rotations: comp_rot,
        scales: comp_scales,
        opacities: comp_opacities,
        sh_compressed,
        sh_degree,
        num_gaussians: n,
    };

    let compressed_bytes = model.total_bytes();
    let compression_ratio = if compressed_bytes > 0 {
        original_bytes as f32 / compressed_bytes as f32
    } else {
        1.0
    };

    let stats = CompressionStats {
        original_bytes,
        compressed_bytes,
        compression_ratio,
        position_error_rms,
        rotation_error_rms,
        scale_error_rms,
        opacity_error_rms,
        sh_error_rms,
    };

    Ok((model, stats))
}

/// Decompress a `CompressedGaussianModel` back to raw arrays.
pub fn decompress_gaussians(
    model: &CompressedGaussianModel,
) -> Result<DecompressedArrays, CompressionError> {
    let positions = model.positions.decompress();
    let rotations = model.rotations.decompress();
    let scales = model.scales.decompress();
    let opacities = model.opacities.decompress();

    let sh_coeffs = match &model.sh_compressed {
        ShCompressed::Raw(raw) => raw.clone(),
        ShCompressed::Codebook { codebook, indices } => codebook.decode(indices),
    };

    // Validate consistency.
    let n = model.num_gaussians;
    if positions.len() != n {
        return Err(CompressionError::LengthMismatch {
            expected: n,
            got: positions.len(),
        });
    }

    Ok(DecompressedArrays {
        positions,
        rotations,
        scales,
        opacities,
        sh_coeffs,
    })
}

// ────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Scalar quantization roundtrips ──────────────────────────

    #[test]
    fn test_quantize_i16_roundtrip() {
        let lo = -5.0f32;
        let hi = 5.0f32;
        for &v in &[-5.0f32, -2.5, 0.0, 2.5, 5.0] {
            let q = quantize_i16(v, lo, hi);
            let r = dequantize_i16(q, lo, hi);
            let max_err = (hi - lo) / 65535.0;
            assert!(
                (v - r).abs() <= max_err + 1e-6,
                "i16 roundtrip: orig={v}, recon={r}, max_err={max_err}"
            );
        }
    }

    #[test]
    fn test_quantize_i8_roundtrip() {
        for &v in &[-1.0f32, -0.5, 0.0, 0.5, 1.0] {
            let q = quantize_i8(v);
            let r = dequantize_i8(q);
            // i8 has 256 levels over range [-1, 1], so step ≈ 2/255 ≈ 0.00784
            assert!((v - r).abs() < 0.009, "i8 roundtrip: orig={v}, recon={r}");
        }
    }

    #[test]
    fn test_quantize_u8_roundtrip() {
        let lo = 0.0f32;
        let hi = 1.0f32;
        for &v in &[0.0f32, 0.25, 0.5, 0.75, 1.0] {
            let q = quantize_u8(v, lo, hi);
            let r = dequantize_u8(q, lo, hi);
            let max_err = (hi - lo) / 255.0;
            assert!(
                (v - r).abs() <= max_err + 1e-6,
                "u8 roundtrip: orig={v}, recon={r}"
            );
        }
    }

    // ── SceneBounds ─────────────────────────────────────────────

    #[test]
    fn test_scene_bounds_from_positions() {
        let positions = vec![[1.0f32, 2.0, 3.0], [-1.0, 0.0, 5.0], [0.5, -2.0, 4.0]];
        let bounds = SceneBounds::from_positions(&positions).unwrap();
        assert!((bounds.min[0] - (-1.0)).abs() < 1e-6);
        assert!((bounds.max[0] - 1.0).abs() < 1e-6);
        assert!((bounds.min[1] - (-2.0)).abs() < 1e-6);
        assert!((bounds.max[1] - 2.0).abs() < 1e-6);
        assert!((bounds.min[2] - 3.0).abs() < 1e-6);
        assert!((bounds.max[2] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_scene_bounds_from_empty() {
        let bounds = SceneBounds::from_positions(&[]);
        assert!(bounds.is_none());
    }

    // ── CompressedPositions ─────────────────────────────────────

    #[test]
    fn test_compress_positions_roundtrip() {
        let positions = vec![
            [0.0f32, 0.0, 0.0],
            [1.0, 2.0, 3.0],
            [-1.0, -2.0, -3.0],
            [0.5, 1.5, 2.5],
        ];
        let bounds = SceneBounds::from_positions(&positions).unwrap().expand(0.1);
        let comp = CompressedPositions::compress(&positions, &bounds);
        let recon = comp.decompress();
        let range = 10.0f32; // roughly 2 * (max 3.0) + margin
        let max_err = range / 65535.0 + 1e-4;
        for (orig, rec) in positions.iter().zip(recon.iter()) {
            for i in 0..3 {
                assert!(
                    (orig[i] - rec[i]).abs() < max_err,
                    "pos roundtrip axis {i}: orig={}, recon={}",
                    orig[i],
                    rec[i]
                );
            }
        }
    }

    // ── CompressedRotations ─────────────────────────────────────

    #[test]
    fn test_compress_rotations_roundtrip() {
        let rotations = vec![
            [0.0f32, 0.0, 0.0, 1.0],
            [0.5, 0.5, 0.5, 0.5],
            [-0.5, 0.5, -0.5, 0.5],
            [1.0f32 / 2.0f32.sqrt(), 0.0, 0.0, 1.0f32 / 2.0f32.sqrt()],
        ];
        let comp = CompressedRotations::compress(&rotations);
        let recon = comp.decompress();
        for (orig, rec) in rotations.iter().zip(recon.iter()) {
            for i in 0..4 {
                assert!(
                    (orig[i] - rec[i]).abs() < 0.01,
                    "rot roundtrip axis {i}: orig={}, recon={}",
                    orig[i],
                    rec[i]
                );
            }
        }
    }

    // ── CompressedScales ────────────────────────────────────────

    #[test]
    fn test_compress_scales_roundtrip() {
        let scales = vec![[-3.0f32, -2.0, -1.0], [-0.5, 0.0, 0.5], [1.0, 2.0, 3.0]];
        let comp = CompressedScales::compress(&scales);
        let recon = comp.decompress();
        let range = comp.log_max - comp.log_min;
        let max_err = range / 255.0 + 1e-5;
        for (orig, rec) in scales.iter().zip(recon.iter()) {
            for i in 0..3 {
                assert!(
                    (orig[i] - rec[i]).abs() < max_err,
                    "scale roundtrip axis {i}: orig={}, recon={}",
                    orig[i],
                    rec[i]
                );
            }
        }
    }

    // ── CompressedOpacities ─────────────────────────────────────

    #[test]
    fn test_compress_opacities_roundtrip() {
        let opacities = vec![-5.0f32, -2.0, 0.0, 2.0, 5.0];
        let comp = CompressedOpacities::compress(&opacities);
        let recon = comp.decompress();
        let range = comp.logit_max - comp.logit_min;
        let max_err = range / 255.0 + 1e-5;
        for (orig, rec) in opacities.iter().zip(recon.iter()) {
            assert!(
                (orig - rec).abs() < max_err,
                "opacity roundtrip: orig={orig}, recon={rec}"
            );
        }
    }

    // ── ShCodebook ──────────────────────────────────────────────

    #[test]
    fn test_sh_codebook_build_simple() {
        // Two very different vectors, codebook of size 2.
        let sh_coeffs = vec![vec![1.0f32, 0.0, 0.0], vec![0.0f32, 0.0, 1.0]];
        let codebook = ShCodebook::build(&sh_coeffs, 2);
        assert!(codebook.is_some());
        let cb = codebook.unwrap();
        assert_eq!(cb.entries.len(), 2);
        assert_eq!(cb.sh_len, 3);
    }

    #[test]
    fn test_sh_codebook_encode_decode() {
        // Create two clusters of 4 vectors each.
        let cluster_a = vec![1.0f32, 0.0];
        let cluster_b = vec![0.0f32, 1.0];
        let mut sh_coeffs: Vec<Vec<f32>> = Vec::new();
        for _ in 0..4 {
            sh_coeffs.push(cluster_a.clone());
        }
        for _ in 0..4 {
            sh_coeffs.push(cluster_b.clone());
        }
        let codebook = ShCodebook::build(&sh_coeffs, 2).unwrap();
        let indices = codebook.encode(&sh_coeffs);
        let decoded = codebook.decode(&indices);
        // All decoded vectors should be close to the original cluster center.
        for (orig, dec) in sh_coeffs.iter().zip(decoded.iter()) {
            let dist: f32 = orig
                .iter()
                .zip(dec.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum();
            assert!(dist.sqrt() < 0.05, "SH decode error too large: {dist}");
        }
    }

    // ── Full pipeline ───────────────────────────────────────────

    #[test]
    fn test_compress_gaussians_full_pipeline() {
        let n = 10usize;
        let positions: Vec<[f32; 3]> = (0..n)
            .map(|i| [i as f32 * 0.1, i as f32 * 0.2, i as f32 * 0.3])
            .collect();
        let rotations: Vec<[f32; 4]> = (0..n).map(|_| [0.0, 0.0, 0.0, 1.0]).collect();
        let scales: Vec<[f32; 3]> = (0..n).map(|i| [-1.0 + i as f32 * 0.1; 3]).collect();
        let opacities: Vec<f32> = (0..n).map(|i| i as f32 * 0.1 - 0.5).collect();
        let sh_len = 9usize;
        let sh_coeffs: Vec<Vec<f32>> = (0..n).map(|_| vec![0.0f32; sh_len]).collect();

        let config = CompressionConfig {
            use_sh_codebook: false,
            ..Default::default()
        };

        let result = compress_gaussians(
            &positions, &rotations, &scales, &opacities, &sh_coeffs, 1, &config,
        );
        assert!(
            result.is_ok(),
            "compress_gaussians failed: {:?}",
            result.err()
        );
        let (model, stats) = result.unwrap();
        assert_eq!(model.num_gaussians, n);
        assert!(stats.original_bytes > 0);
        assert!(stats.compressed_bytes > 0);
    }

    // ── CompressionStats ────────────────────────────────────────

    #[test]
    fn test_compression_stats_ratio() {
        let n = 100usize;
        let positions: Vec<[f32; 3]> = (0..n).map(|i| [i as f32; 3]).collect();
        let rotations: Vec<[f32; 4]> = (0..n).map(|_| [0.0, 0.0, 0.0, 1.0]).collect();
        let scales: Vec<[f32; 3]> = (0..n).map(|_| [-1.0, -1.5, -2.0]).collect();
        let opacities: Vec<f32> = (0..n).map(|i| i as f32 * 0.02 - 1.0).collect();
        let sh_len = 27usize;
        let sh_coeffs: Vec<Vec<f32>> = (0..n).map(|i| vec![(i as f32) * 0.01; sh_len]).collect();

        let config = CompressionConfig {
            sh_codebook_size: 64,
            ..Default::default()
        };
        let (_, stats) = compress_gaussians(
            &positions, &rotations, &scales, &opacities, &sh_coeffs, 2, &config,
        )
        .unwrap();

        assert!(
            stats.compressed_bytes < stats.original_bytes,
            "compressed ({}) should be smaller than original ({})",
            stats.compressed_bytes,
            stats.original_bytes
        );
        assert!(stats.compression_ratio > 1.0, "ratio should be > 1");
    }

    #[test]
    fn test_compression_stats_format_summary() {
        let stats = CompressionStats {
            original_bytes: 1000,
            compressed_bytes: 250,
            compression_ratio: 4.0,
            position_error_rms: 0.001,
            rotation_error_rms: 0.005,
            scale_error_rms: 0.002,
            opacity_error_rms: 0.003,
            sh_error_rms: 0.010,
        };
        let s = stats.format_summary();
        assert!(s.contains("4.0"), "summary should contain ratio: {s}");
        assert!(
            s.contains("1000"),
            "summary should contain original bytes: {s}"
        );
        assert!(
            s.contains("250"),
            "summary should contain compressed bytes: {s}"
        );
    }

    // ── RMS error helper ────────────────────────────────────────

    #[test]
    fn test_rms_error_helper() {
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![1.1f32, 2.0, 2.9];
        let rms = rms_error_f32(&a, &b);
        // errors: 0.1, 0.0, 0.1 → mse = 0.02/3 → rms ≈ 0.0816
        assert!((rms - (0.02f32 / 3.0).sqrt()).abs() < 1e-5, "rms={rms}");
    }

    #[test]
    fn test_rms_error_helper_empty() {
        let rms = rms_error_f32(&[], &[]);
        assert_eq!(rms, 0.0);
    }

    // ── Error cases ─────────────────────────────────────────────

    #[test]
    fn test_empty_input_error() {
        let result = compress_gaussians(&[], &[], &[], &[], &[], 0, &Default::default());
        assert!(matches!(result, Err(CompressionError::EmptyInput)));
    }

    #[test]
    fn test_length_mismatch_error() {
        let positions = vec![[0.0f32; 3]; 3];
        let rotations = vec![[0.0f32, 0.0, 0.0, 1.0]; 2]; // wrong length
        let scales = vec![[-1.0f32; 3]; 3];
        let opacities = vec![0.0f32; 3];
        let result = compress_gaussians(
            &positions,
            &rotations,
            &scales,
            &opacities,
            &[],
            0,
            &Default::default(),
        );
        assert!(matches!(
            result,
            Err(CompressionError::LengthMismatch { .. })
        ));
    }

    // ── Decompress roundtrip RMS ────────────────────────────────

    #[test]
    fn test_decompress_roundtrip_rms_error_small() {
        // Use values in typical ranges; expect RMS error < 0.01 for all fields.
        let n = 50usize;
        let positions: Vec<[f32; 3]> = (0..n)
            .map(|i| {
                let t = i as f32 / n as f32;
                [t * 2.0 - 1.0, t * 3.0 - 1.5, t * 4.0 - 2.0]
            })
            .collect();
        // Normalize rotations to unit quaternions.
        let rotations: Vec<[f32; 4]> = (0..n)
            .map(|i| {
                let t = i as f32 / n as f32;
                let q = [t - 0.5, 0.3, -0.2, (1.0f32 - t * 0.1).max(0.1)];
                let norm = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
                [q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm]
            })
            .collect();
        let scales: Vec<[f32; 3]> = (0..n)
            .map(|i| {
                let t = i as f32 / n as f32;
                [-3.0 + t * 3.0; 3]
            })
            .collect();
        let opacities: Vec<f32> = (0..n).map(|i| (i as f32 / n as f32) * 6.0 - 3.0).collect();
        let sh_len = 9usize;
        let sh_coeffs: Vec<Vec<f32>> = (0..n)
            .map(|i| (0..sh_len).map(|j| (i + j) as f32 * 0.01).collect())
            .collect();

        let config = CompressionConfig {
            use_sh_codebook: false, // raw SH for exact roundtrip
            ..Default::default()
        };
        let (model, stats) = compress_gaussians(
            &positions, &rotations, &scales, &opacities, &sh_coeffs, 1, &config,
        )
        .unwrap();

        let arrays = decompress_gaussians(&model).unwrap();
        assert_eq!(arrays.positions.len(), n);
        assert_eq!(arrays.rotations.len(), n);

        // Position RMS < 0.01 (range ~ 6 / 65535 ≈ very small, but bounds add margin).
        assert!(
            stats.position_error_rms < 0.01,
            "position RMS too large: {}",
            stats.position_error_rms
        );
        // Rotation RMS < 0.01 (i8 over [-1,1]: step ≈ 0.008).
        assert!(
            stats.rotation_error_rms < 0.01,
            "rotation RMS too large: {}",
            stats.rotation_error_rms
        );
        // Scale and opacity RMS < 0.05 (u8 over ~6 unit range).
        assert!(
            stats.scale_error_rms < 0.05,
            "scale RMS too large: {}",
            stats.scale_error_rms
        );
        assert!(
            stats.opacity_error_rms < 0.05,
            "opacity RMS too large: {}",
            stats.opacity_error_rms
        );
        // SH is raw (no compression), so error is zero.
        assert!(
            stats.sh_error_rms < 1e-6,
            "sh RMS should be near zero: {}",
            stats.sh_error_rms
        );
    }
}
