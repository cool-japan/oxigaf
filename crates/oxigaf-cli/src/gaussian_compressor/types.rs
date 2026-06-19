//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use thiserror::Error;

use super::functions::{compute_scale_offset, quantize_to_i16, quantize_to_i8};

/// Errors that can occur during Gaussian scene compression.
#[derive(Debug, Error)]
pub enum CompressorError {
    /// The scene contains no Gaussians.
    #[error("empty scene: no Gaussians")]
    EmptyScene,
    /// Array dimension does not match the expected size.
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
    /// Configuration parameter is invalid.
    #[error("invalid config: {0}")]
    InvalidConfig(String),
    /// Quantization failed.
    #[error("quantization error: {0}")]
    Quantization(String),
    /// Bit depth is not 8 or 16.
    #[error("invalid bit depth: must be 8 or 16, got {0}")]
    InvalidBitDepth(u8),
    /// K-means algorithm did not converge.
    #[error("k-means did not converge")]
    KMeansNoConvergence,
}
/// A quantized 1D attribute stored as i16, i8, or f32 (Full pass-through).
///
/// Dequantize: `value = data[i] * scale + offset`
pub struct QuantizedAttribute {
    /// Data for Half (16-bit) precision; empty otherwise.
    pub data_i16: Vec<i16>,
    /// Data for Byte (8-bit) precision; empty otherwise.
    pub data_i8: Vec<i8>,
    /// Data for Full (32-bit) precision pass-through; empty otherwise.
    pub data_f32: Vec<f32>,
    /// Precision level used.
    pub precision: QuantizationPrecision,
    /// Multiplier for dequantization: value = quantized * scale + offset.
    pub scale: f32,
    /// Offset for dequantization.
    pub offset: f32,
    /// Number of scalar elements.
    pub n_elements: usize,
}
impl QuantizedAttribute {
    /// Quantize a slice of f32 values to the requested precision.
    ///
    /// For `Full`, the values are stored as-is and the round-trip is lossless.
    /// For `Half` and `Byte`, scalar min-max quantization is applied.
    pub fn quantize(
        values: &[f32],
        precision: QuantizationPrecision,
    ) -> Result<Self, CompressorError> {
        let n = values.len();
        match precision {
            QuantizationPrecision::Full => Ok(Self {
                data_i16: Vec::new(),
                data_i8: Vec::new(),
                data_f32: values.to_vec(),
                precision,
                scale: 1.0,
                offset: 0.0,
                n_elements: n,
            }),
            QuantizationPrecision::Half => {
                let (scale, offset) = compute_scale_offset(values, 65534.0);
                let data_i16 = quantize_to_i16(values, scale, offset)?;
                Ok(Self {
                    data_i16,
                    data_i8: Vec::new(),
                    data_f32: Vec::new(),
                    precision,
                    scale,
                    offset,
                    n_elements: n,
                })
            }
            QuantizationPrecision::Byte => {
                let (scale, offset) = compute_scale_offset(values, 254.0);
                let data_i8 = quantize_to_i8(values, scale, offset)?;
                Ok(Self {
                    data_i16: Vec::new(),
                    data_i8,
                    data_f32: Vec::new(),
                    precision,
                    scale,
                    offset,
                    n_elements: n,
                })
            }
        }
    }
    /// Dequantize back to f32 values.
    pub fn dequantize(&self) -> Vec<f32> {
        match self.precision {
            QuantizationPrecision::Full => self.data_f32.clone(),
            QuantizationPrecision::Half => self
                .data_i16
                .iter()
                .map(|&v| v as f32 * self.scale + self.offset)
                .collect(),
            QuantizationPrecision::Byte => self
                .data_i8
                .iter()
                .map(|&v| v as f32 * self.scale + self.offset)
                .collect(),
        }
    }
    /// Memory size of the quantized data in bytes.
    pub fn byte_size(&self) -> usize {
        match self.precision {
            QuantizationPrecision::Full => self.n_elements * 4,
            QuantizationPrecision::Half => self.n_elements * 2,
            QuantizationPrecision::Byte => self.n_elements,
        }
    }
}
/// Configuration for k-means position clustering.
#[derive(Debug, Clone)]
pub struct KMeansConfig {
    /// Number of clusters.
    pub n_clusters: usize,
    /// Maximum number of Lloyd iterations.
    pub n_iterations: usize,
    /// Convergence tolerance (L2 shift of centers).
    pub tolerance: f32,
}
/// Full compression pipeline configuration.
#[derive(Debug, Clone)]
pub struct CompressionConfig {
    /// Precision for position data.
    pub position_precision: QuantizationPrecision,
    /// Precision for rotation quaternions.
    pub rotation_precision: QuantizationPrecision,
    /// Precision for log-space scales.
    pub scale_precision: QuantizationPrecision,
    /// Precision for opacity logits.
    pub opacity_precision: QuantizationPrecision,
    /// Precision for SH DC coefficients.
    pub sh_dc_precision: QuantizationPrecision,
    /// Precision for SH rest coefficients.
    pub sh_rest_precision: QuantizationPrecision,
    /// Pruning parameters.
    pub pruning: ScenePruningConfig,
    /// Apply k-means clustering to positions before quantizing.
    pub use_position_clustering: bool,
    /// K-means configuration (used when `use_position_clustering` is true).
    pub kmeans: KMeansConfig,
}
/// Summary statistics for a compression run.
pub struct CompressionStats {
    /// Number of Gaussians before compression (before pruning).
    pub n_gaussians_before: usize,
    /// Number of Gaussians after compression (after pruning).
    pub n_gaussians_after: usize,
    /// Fraction of Gaussians pruned: `(before - after) / before`.
    pub pruned_fraction: f32,
    /// Compressed size in megabytes.
    pub compressed_mb: f32,
    /// Uncompressed size in megabytes.
    pub uncompressed_mb: f32,
    /// Compression ratio: `uncompressed / compressed`.
    pub compression_ratio: f32,
    /// RMSE of position dequantization vs. original positions.
    pub position_quantization_rmse: f32,
    /// RMSE of opacity dequantization vs. original opacities.
    pub opacity_quantization_rmse: f32,
}
/// A fully decompressed Gaussian scene with all attributes as f32.
pub struct DecompressedScene {
    /// Positions N×3.
    pub positions: Vec<f32>,
    /// Rotation quaternions N×4.
    pub rotations: Vec<f32>,
    /// Log-space scales N×3.
    pub scales: Vec<f32>,
    /// Opacity logits N.
    pub opacities: Vec<f32>,
    /// SH DC coefficients N×3.
    pub sh_dc: Vec<f32>,
    /// SH rest coefficients N×C.
    pub sh_rest: Vec<f32>,
    /// Number of Gaussians.
    pub n_gaussians: usize,
}
/// Flat per-attribute slices for a Gaussian scene, passed to `gc_compress`.
pub struct GcSceneSlices<'a> {
    /// Positions, flat N×3.
    pub positions: &'a [f32],
    /// Rotations (quaternions), flat N×4.
    pub rotations: &'a [f32],
    /// Log-scales, flat N×3.
    pub scales: &'a [f32],
    /// Logit-space opacities, length N.
    pub opacities: &'a [f32],
    /// DC SH coefficients, flat N×3.
    pub sh_dc: &'a [f32],
    /// Rest SH coefficients, flat N×(n_rest_per_gaussian).
    pub sh_rest: &'a [f32],
    /// Number of rest SH coefficients per Gaussian (0 if unused).
    pub n_rest_per_gaussian: usize,
}
/// Configuration for opacity- and scale-based Gaussian pruning.
#[derive(Debug, Clone)]
pub struct ScenePruningConfig {
    /// Remove Gaussians whose real opacity (sigmoid of logit) < threshold.
    pub opacity_threshold: f32,
    /// Remove Gaussians where any log_scale component > this value.
    pub max_log_scale: f32,
    /// Remove Gaussians where ALL log_scale components < this value.
    pub min_log_scale: f32,
    /// If set, prune scene to at most this many Gaussians.
    pub target_n_gaussians: Option<usize>,
    /// Keep the top fraction by opacity (1.0 = keep all passing threshold).
    pub preserve_top_fraction: f32,
}
/// A fully compressed Gaussian scene with all attributes quantized.
pub struct CompressedScene {
    /// Positions N×3, quantized.
    pub positions: QuantizedAttribute,
    /// Rotations N×4 (quaternions), quantized.
    pub rotations: QuantizedAttribute,
    /// Log-space scales N×3, quantized.
    pub scales: QuantizedAttribute,
    /// Opacity logits N, quantized.
    pub opacities: QuantizedAttribute,
    /// SH DC coefficients N×3, quantized.
    pub sh_dc: QuantizedAttribute,
    /// SH rest coefficients N×C, quantized.
    pub sh_rest: QuantizedAttribute,
    /// Number of Gaussians after pruning.
    pub n_gaussians: usize,
    /// Number of SH rest coefficients per Gaussian.
    pub n_sh_rest: usize,
    /// The compression configuration used.
    pub compression_config: CompressionConfig,
}
impl CompressedScene {
    /// Total compressed size in bytes (all quantized attributes).
    pub fn compressed_bytes(&self) -> usize {
        self.positions.byte_size()
            + self.rotations.byte_size()
            + self.scales.byte_size()
            + self.opacities.byte_size()
            + self.sh_dc.byte_size()
            + self.sh_rest.byte_size()
    }
    /// Equivalent uncompressed size in bytes (all as f32).
    pub fn uncompressed_bytes(&self) -> usize {
        let n = self.n_gaussians;
        let n_sh_rest = self.n_sh_rest;
        (n * 3 + n * 4 + n * 3 + n + n * 3 + n * n_sh_rest) * 4
    }
    /// Compression ratio: uncompressed / compressed. Higher = better.
    pub fn compression_ratio(&self) -> f32 {
        let c = self.compressed_bytes();
        let u = self.uncompressed_bytes();
        if c == 0 || u == 0 {
            1.0
        } else {
            u as f32 / c as f32
        }
    }
}
/// Specifies the precision level for scalar quantization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationPrecision {
    /// 32-bit float — no quantization; data passed through as-is.
    Full,
    /// 16-bit integer quantization (i16).
    Half,
    /// 8-bit integer quantization (i8).
    Byte,
}
impl QuantizationPrecision {
    /// Number of bits used for this precision level.
    pub fn bits(&self) -> u8 {
        match self {
            Self::Full => 32,
            Self::Half => 16,
            Self::Byte => 8,
        }
    }
    /// Human-readable name for this precision.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Full => "full (f32)",
            Self::Half => "half (i16)",
            Self::Byte => "byte (i8)",
        }
    }
}
