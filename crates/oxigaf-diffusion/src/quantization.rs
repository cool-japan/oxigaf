//! INT8 weight quantization for the diffusion pipeline.
//!
//! This module provides symmetric (absmax) and asymmetric (min-max) INT8
//! quantization of `f32` tensors, with per-tensor or per-channel granularity.
//! The primary use-case is compressing model weights (~1.7 GB fp32 → ~425 MB
//! INT8) with minimal quality loss at inference time.
//!
//! ## Quantization modes
//!
//! | Mode       | Formula                                      | Best for     |
//! |------------|----------------------------------------------|--------------|
//! | Symmetric  | `scale = max_abs / 127`, `q = round(x/s).clamp(-127, 127)` | Weights |
//! | Asymmetric | `scale = (max−min)/255`, `zp = round(−min/s).clamp(0,255)` | Activations |
//!
//! ## Usage
//!
//! ```rust
//! use oxigaf_diffusion::quantization::{AbsmaxQuantizer, QuantizationScope};
//!
//! let data = vec![0.1_f32, -0.5, 0.3, 0.8, -0.2, 0.0];
//! let shape = vec![2, 3];
//! let quantizer = AbsmaxQuantizer::per_tensor();
//! let qt = quantizer.quantize(&data, &shape).expect("quantize");
//! let recovered = qt.dequantize().expect("dequantize");
//! ```

use crate::DiffusionError;

// ---------------------------------------------------------------------------
// QuantizationMode
// ---------------------------------------------------------------------------

/// How scale and zero-point are derived.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationMode {
    /// Symmetric: `zero_point = 0`, `scale = max_abs / 127.0`.
    ///
    /// Values are stored as `i8` in `[-127, 127]`.  Zero maps to zero.
    /// Preferred for weights whose distributions are roughly zero-centred.
    Symmetric,

    /// Asymmetric: `scale = (max − min) / 255.0`, `zero_point = round(−min / scale)`.
    ///
    /// Values are stored as `i8` but represent u8 (0..=255) bit-cast to i8
    /// via `x as u8 as i8`.  Preferred for activations with non-zero minimum.
    Asymmetric,
}

// ---------------------------------------------------------------------------
// QuantizationScope
// ---------------------------------------------------------------------------

/// Granularity at which scale/zero-point parameters are computed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationScope {
    /// A single scale + zero-point covers the whole tensor.
    PerTensor,
    /// One scale + zero-point per slice along the given axis (output channels).
    PerChannel(usize),
}

// ---------------------------------------------------------------------------
// QuantizedTensor
// ---------------------------------------------------------------------------

/// An INT8-quantized tensor with associated calibration parameters.
///
/// For **symmetric** quantization the raw `i8` values are used directly:
/// ```text
///   f32 ≈ i8 * scale
/// ```
///
/// For **asymmetric** quantization the bytes are semantically `u8` stored as
/// `i8` (bit-cast).  Dequantization uses:
/// ```text
///   f32 ≈ (u8 − zero_point) * scale
///       = ((raw_i8 as u8) as i32 − zero_point) as f32 * scale
/// ```
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    /// Quantized data — one byte per element.
    pub data: Vec<i8>,
    /// Scale factor(s).  Length is 1 for `PerTensor`, or the channel count for
    /// `PerChannel`.
    pub scales: Vec<f32>,
    /// Zero-point(s).  Length mirrors `scales`.
    pub zero_points: Vec<i32>,
    /// Original tensor shape (defines how `data` is indexed).
    pub shape: Vec<usize>,
    /// Quantization mode used during quantization.
    pub mode: QuantizationMode,
    /// Quantization scope used during quantization.
    pub scope: QuantizationScope,
}

impl QuantizedTensor {
    /// Total number of scalar elements (`product of shape`).
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// Memory occupied by this struct in bytes.
    ///
    /// Counts:
    /// * 1 byte per quantized element (`data`)
    /// * 4 bytes per scale (`f32`)
    /// * 4 bytes per zero-point (`i32`)
    /// * 8 bytes per shape dimension (`usize` on 64-bit targets)
    pub fn memory_bytes(&self) -> usize {
        self.data.len() + self.scales.len() * 4 + self.zero_points.len() * 4 + self.shape.len() * 8
    }

    /// Compression ratio compared to storing the same data as `f32`.
    ///
    /// `= (num_elements * 4) / memory_bytes`
    pub fn compression_ratio_vs_f32(&self) -> f32 {
        let fp32_bytes = self.num_elements() * 4;
        fp32_bytes as f32 / self.memory_bytes() as f32
    }

    /// Validate internal consistency of this tensor's metadata.
    ///
    /// Every field of `QuantizedTensor` is `pub`, so one can be hand-built
    /// (not just produced by [`AbsmaxQuantizer::quantize`] /
    /// [`MinMaxQuantizer::quantize`], which always produce a tensor that
    /// passes this check) with a `scales`/`zero_points`/`shape` combination
    /// that would otherwise make [`Self::dequantize`] panic on an
    /// out-of-bounds index. [`Self::dequantize`] calls this first.
    ///
    /// # Errors
    ///
    /// Returns [`DiffusionError::InvalidConfig`] if:
    /// * `shape` is empty, or `data.len()` does not equal `product(shape)`.
    /// * `scope` is `PerTensor` and `scales`/`zero_points` do not each have
    ///   exactly 1 entry.
    /// * `scope` is `PerChannel(axis)` and `axis >= shape.len()`, or
    ///   `scales`/`zero_points` do not each have exactly `shape[axis]`
    ///   entries.
    pub fn validate(&self) -> Result<(), DiffusionError> {
        validate_shape_data(&self.data, &self.shape)?;

        let expected_params = match self.scope {
            QuantizationScope::PerTensor => 1,
            QuantizationScope::PerChannel(axis) => {
                let (_, channel_size, _) = channel_dims(&self.shape, axis)?;
                channel_size
            }
        };
        if self.scales.len() != expected_params {
            return Err(DiffusionError::InvalidConfig(format!(
                "quantization: scales has {} entries, expected {} for scope {:?}",
                self.scales.len(),
                expected_params,
                self.scope,
            )));
        }
        if self.zero_points.len() != expected_params {
            return Err(DiffusionError::InvalidConfig(format!(
                "quantization: zero_points has {} entries, expected {} for scope {:?}",
                self.zero_points.len(),
                expected_params,
                self.scope,
            )));
        }
        Ok(())
    }

    /// Dequantize back to `f32`.
    ///
    /// Returns a flat `Vec<f32>` in the same element order as the original
    /// input used during quantization.
    ///
    /// # Errors
    ///
    /// Returns whatever [`Self::validate`] returns if this tensor's
    /// metadata is internally inconsistent. This can only happen for a
    /// hand-built `QuantizedTensor` — every `AbsmaxQuantizer`/
    /// `MinMaxQuantizer::quantize` output always validates.
    pub fn dequantize(&self) -> Result<Vec<f32>, DiffusionError> {
        self.validate()?;
        let n = self.data.len();
        let mut out = vec![0.0_f32; n];

        match self.scope {
            QuantizationScope::PerTensor => {
                let scale = self.scales[0];
                let zp = self.zero_points[0];
                for (i, &raw) in self.data.iter().enumerate() {
                    out[i] = dequantize_one(raw, scale, zp, self.mode);
                }
            }
            QuantizationScope::PerChannel(axis) => {
                let inner_size = self.shape[axis + 1..].iter().product::<usize>().max(1);
                let channel_size = self.shape[axis];

                for (i, &raw) in self.data.iter().enumerate() {
                    let channel = (i / inner_size) % channel_size;
                    let scale = self.scales[channel];
                    let zp = self.zero_points[channel];
                    out[i] = dequantize_one(raw, scale, zp, self.mode);
                }
            }
        }

        Ok(out)
    }
}

/// Single-element dequantization (shared by both scopes).
#[inline]
fn dequantize_one(raw: i8, scale: f32, zero_point: i32, mode: QuantizationMode) -> f32 {
    match mode {
        QuantizationMode::Symmetric => (raw as f32) * scale,
        QuantizationMode::Asymmetric => {
            // raw is a u8 value stored as i8 via bit-cast
            let unsigned = (raw as u8) as i32;
            (unsigned - zero_point) as f32 * scale
        }
    }
}

// ---------------------------------------------------------------------------
// AbsmaxQuantizer (Symmetric)
// ---------------------------------------------------------------------------

/// Symmetric absolute-maximum quantizer — recommended for weight tensors.
///
/// Finds the maximum absolute value in each quantization group and sets:
/// ```text
///   scale = max_abs / 127.0
///   q     = clamp(round(x / scale), -127, 127)
/// ```
#[derive(Debug, Clone)]
pub struct AbsmaxQuantizer {
    /// Always `QuantizationMode::Symmetric`.
    pub mode: QuantizationMode,
    /// Granularity for scale computation.
    pub scope: QuantizationScope,
}

impl AbsmaxQuantizer {
    /// Create a new symmetric quantizer with the given scope.
    pub fn new(scope: QuantizationScope) -> Self {
        Self {
            mode: QuantizationMode::Symmetric,
            scope,
        }
    }

    /// Convenience: per-tensor symmetric quantizer.
    pub fn per_tensor() -> Self {
        Self::new(QuantizationScope::PerTensor)
    }

    /// Convenience: per-channel symmetric quantizer along the given axis.
    pub fn per_channel(axis: usize) -> Self {
        Self::new(QuantizationScope::PerChannel(axis))
    }

    /// Quantize `data` with the given `shape`.
    ///
    /// # Errors
    ///
    /// Returns [`DiffusionError::InvalidConfig`] if:
    /// * `shape` is empty
    /// * `data.len()` does not equal the product of `shape`
    pub fn quantize(
        &self,
        data: &[f32],
        shape: &[usize],
    ) -> Result<QuantizedTensor, DiffusionError> {
        validate_shape_data(data, shape)?;
        let n = data.len();

        match self.scope {
            QuantizationScope::PerTensor => {
                let max_abs = data.iter().copied().map(f32::abs).fold(0.0_f32, f32::max);
                let scale = if max_abs == 0.0 {
                    1.0_f32
                } else {
                    max_abs / 127.0
                };
                let quantized: Vec<i8> =
                    data.iter().map(|&x| symmetric_quantize(x, scale)).collect();
                Ok(QuantizedTensor {
                    data: quantized,
                    scales: vec![scale],
                    zero_points: vec![0],
                    shape: shape.to_vec(),
                    mode: QuantizationMode::Symmetric,
                    scope: self.scope,
                })
            }
            QuantizationScope::PerChannel(axis) => {
                let (outer_size, channel_size, inner_size) = channel_dims(shape, axis)?;
                let _ = outer_size; // used implicitly through n
                let _ = n;

                // Single pass over `data`, accumulating each channel's
                // max-abs directly instead of rescanning the whole tensor
                // once per channel (filtering by channel and folding per
                // channel was O(n * channel_size); this is O(n)).
                let mut max_abs_per_channel = vec![0.0_f32; channel_size];
                for (i, &x) in data.iter().enumerate() {
                    let c = (i / inner_size) % channel_size;
                    let ax = x.abs();
                    if ax > max_abs_per_channel[c] {
                        max_abs_per_channel[c] = ax;
                    }
                }
                let scales: Vec<f32> = max_abs_per_channel
                    .iter()
                    .map(|&max_abs| {
                        if max_abs == 0.0 {
                            1.0_f32
                        } else {
                            max_abs / 127.0
                        }
                    })
                    .collect();
                let zero_points = vec![0_i32; channel_size];

                // Quantize
                let quantized: Vec<i8> = data
                    .iter()
                    .enumerate()
                    .map(|(i, &x)| {
                        let channel = (i / inner_size) % channel_size;
                        symmetric_quantize(x, scales[channel])
                    })
                    .collect();

                Ok(QuantizedTensor {
                    data: quantized,
                    scales,
                    zero_points,
                    shape: shape.to_vec(),
                    mode: QuantizationMode::Symmetric,
                    scope: self.scope,
                })
            }
        }
    }
}

// ---------------------------------------------------------------------------
// MinMaxQuantizer (Asymmetric)
// ---------------------------------------------------------------------------

/// Asymmetric min-max quantizer — recommended for activations.
///
/// For each quantization group:
/// ```text
///   scale      = (max − min) / 255.0
///   zero_point = clamp(round(−min / scale), 0, 255)
///   q_u8       = clamp(round(x / scale) + zero_point, 0, 255)
/// ```
/// The `u8` result is bit-cast to `i8` for storage in [`QuantizedTensor::data`].
#[derive(Debug, Clone)]
pub struct MinMaxQuantizer {
    /// Always `QuantizationMode::Asymmetric`.
    pub mode: QuantizationMode,
    /// Granularity for scale computation.
    pub scope: QuantizationScope,
}

impl MinMaxQuantizer {
    /// Create a new asymmetric quantizer with the given scope.
    pub fn new(scope: QuantizationScope) -> Self {
        Self {
            mode: QuantizationMode::Asymmetric,
            scope,
        }
    }

    /// Convenience: per-tensor asymmetric quantizer.
    pub fn per_tensor() -> Self {
        Self::new(QuantizationScope::PerTensor)
    }

    /// Quantize `data` with the given `shape`.
    ///
    /// # Errors
    ///
    /// Returns [`DiffusionError::InvalidConfig`] if:
    /// * `shape` is empty
    /// * `data.len()` does not equal the product of `shape`
    pub fn quantize(
        &self,
        data: &[f32],
        shape: &[usize],
    ) -> Result<QuantizedTensor, DiffusionError> {
        validate_shape_data(data, shape)?;

        match self.scope {
            QuantizationScope::PerTensor => {
                let (scale, zp) = minmax_scale_zp(data);
                let quantized: Vec<i8> = data
                    .iter()
                    .map(|&x| asymmetric_quantize(x, scale, zp))
                    .collect();
                Ok(QuantizedTensor {
                    data: quantized,
                    scales: vec![scale],
                    zero_points: vec![zp],
                    shape: shape.to_vec(),
                    mode: QuantizationMode::Asymmetric,
                    scope: self.scope,
                })
            }
            QuantizationScope::PerChannel(axis) => {
                let (_, channel_size, inner_size) = channel_dims(shape, axis)?;

                // Single pass over `data`, accumulating each channel's
                // (min, max) directly instead of collecting every channel's
                // values into a fresh Vec and rescanning the whole tensor
                // once per channel (was O(n * channel_size) plus per-channel
                // allocations; this is a single O(n) pass with none).
                let mut min_per_channel = vec![f32::MAX; channel_size];
                let mut max_per_channel = vec![f32::MIN; channel_size];
                for (i, &x) in data.iter().enumerate() {
                    let c = (i / inner_size) % channel_size;
                    if x < min_per_channel[c] {
                        min_per_channel[c] = x;
                    }
                    if x > max_per_channel[c] {
                        max_per_channel[c] = x;
                    }
                }

                let mut scales = vec![1.0_f32; channel_size];
                let mut zero_points = vec![0_i32; channel_size];
                for c in 0..channel_size {
                    // A channel with no elements at all (only possible when
                    // some other shape dimension is 0, which also forces
                    // `data` to be empty) keeps the same (1.0, 0) degenerate
                    // default `minmax_scale_zp`'s own empty-input branch
                    // would have produced.
                    if min_per_channel[c] == f32::MAX {
                        continue;
                    }
                    let (s, zp) =
                        minmax_scale_zp_from_bounds(min_per_channel[c], max_per_channel[c]);
                    scales[c] = s;
                    zero_points[c] = zp;
                }

                let quantized: Vec<i8> = data
                    .iter()
                    .enumerate()
                    .map(|(i, &x)| {
                        let channel = (i / inner_size) % channel_size;
                        asymmetric_quantize(x, scales[channel], zero_points[channel])
                    })
                    .collect();

                Ok(QuantizedTensor {
                    data: quantized,
                    scales,
                    zero_points,
                    shape: shape.to_vec(),
                    mode: QuantizationMode::Asymmetric,
                    scope: self.scope,
                })
            }
        }
    }
}

// ---------------------------------------------------------------------------
// QuantizationMetrics
// ---------------------------------------------------------------------------

/// Quality metrics for a quantize-then-dequantize roundtrip.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantizationMetrics {
    /// Maximum |original − dequantized| across all elements.
    pub max_abs_error: f32,
    /// Mean |original − dequantized|.
    pub mean_abs_error: f32,
    /// Root-mean-squared error.
    pub root_mean_sq_error: f32,
    /// Signal-to-noise ratio in decibels: `20 * log10(signal_rms / noise_rms)`.
    ///
    /// `f32::INFINITY` when noise is zero (perfect reconstruction).
    pub signal_to_noise_ratio_db: f32,
    /// Compression ratio vs fp32: `(n*4) / memory_bytes`.
    pub compression_ratio: f32,
}

/// Compute quality metrics for an INT8 quantization round-trip.
///
/// Dequantizes `quantized_tensor` and compares with `original`.
///
/// # Errors
///
/// Returns [`DiffusionError::InvalidConfig`] if:
/// * `original` is empty (every metric here is a mean or ratio over
///   `original.len()` elements, which is otherwise a silent `0.0 / 0.0 =
///   NaN`).
/// * `original.len()` does not equal `quantized_tensor.num_elements()`
///   (otherwise the `zip`s below would silently truncate to the shorter of
///   the two while the divisor stayed `original.len()`, under-reporting
///   error for a mismatched pair).
/// * `quantized_tensor` itself fails [`QuantizedTensor::validate`].
pub fn compute_quantization_metrics(
    original: &[f32],
    quantized_tensor: &QuantizedTensor,
) -> Result<QuantizationMetrics, DiffusionError> {
    if original.is_empty() {
        return Err(DiffusionError::InvalidConfig(
            "compute_quantization_metrics: original must not be empty".to_owned(),
        ));
    }
    if original.len() != quantized_tensor.num_elements() {
        return Err(DiffusionError::InvalidConfig(format!(
            "compute_quantization_metrics: original has {} elements, quantized_tensor has {}",
            original.len(),
            quantized_tensor.num_elements(),
        )));
    }
    let recovered = quantized_tensor.dequantize()?;
    let n = original.len();

    let max_abs_error = original
        .iter()
        .zip(recovered.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0_f32, f32::max);

    let mean_abs_error = original
        .iter()
        .zip(recovered.iter())
        .map(|(&a, &b)| (a - b).abs())
        .sum::<f32>()
        / n as f32;

    let mse = original
        .iter()
        .zip(recovered.iter())
        .map(|(&a, &b)| (a - b).powi(2))
        .sum::<f32>()
        / n as f32;
    let root_mean_sq_error = mse.sqrt();

    // SNR: signal_rms = sqrt(mean(x^2)), noise_rms = sqrt(mean((x-x')^2))
    let signal_rms = (original.iter().map(|&x| x.powi(2)).sum::<f32>() / n as f32).sqrt();
    let noise_rms = root_mean_sq_error; // same as sqrt(mse)

    let signal_to_noise_ratio_db = if noise_rms == 0.0 {
        f32::INFINITY
    } else if signal_rms == 0.0 {
        // Signal is zero, noise is non-zero → 0 dB
        0.0
    } else {
        20.0 * (signal_rms / noise_rms).log10()
    };

    let compression_ratio = quantized_tensor.compression_ratio_vs_f32();

    Ok(QuantizationMetrics {
        max_abs_error,
        mean_abs_error,
        root_mean_sq_error,
        signal_to_noise_ratio_db,
        compression_ratio,
    })
}

// ---------------------------------------------------------------------------
// LayerQuantizationPlan
// ---------------------------------------------------------------------------

/// Recommended quantization strategy and size estimate for one model layer.
#[derive(Debug, Clone)]
pub struct LayerQuantizationPlan {
    /// Human-readable layer name (e.g. `"encoder.conv1.weight"`).
    pub layer_name: String,
    /// Number of scalar elements in the layer.
    pub num_elements: usize,
    /// Storage size in fp32: `num_elements * 4` bytes.
    pub original_bytes: usize,
    /// Estimated storage in INT8: `num_elements * 1` + scale/zero-point overhead.
    pub quantized_bytes: usize,
    /// Recommended quantization mode.
    pub recommended_mode: QuantizationMode,
    /// Recommended quantization scope.
    pub recommended_scope: QuantizationScope,
}

/// Plan INT8 quantization for a single layer.
///
/// Heuristics:
/// * Weights (2-D or higher): symmetric, per-channel along axis 0.
/// * Weights (1-D, bias): symmetric, per-tensor.
/// * Activations: asymmetric, per-tensor.
///
/// Size estimate for `quantized_bytes`:
/// * 1 byte per element
/// * 4 bytes per scale (f32)
/// * 4 bytes per zero-point (i32)
/// * 8 bytes per shape dimension (metadata)
pub fn plan_layer_quantization(
    layer_name: &str,
    shape: &[usize],
    is_weight: bool,
) -> LayerQuantizationPlan {
    let num_elements: usize = shape.iter().product();
    let original_bytes = num_elements * 4;

    let (recommended_mode, recommended_scope) = if is_weight {
        if shape.len() >= 2 {
            (
                QuantizationMode::Symmetric,
                QuantizationScope::PerChannel(0),
            )
        } else {
            (QuantizationMode::Symmetric, QuantizationScope::PerTensor)
        }
    } else {
        (QuantizationMode::Asymmetric, QuantizationScope::PerTensor)
    };

    // Estimate number of channels (scale/zp pairs)
    let num_channels = match recommended_scope {
        QuantizationScope::PerTensor => 1,
        QuantizationScope::PerChannel(axis) => {
            if shape.is_empty() {
                1
            } else {
                shape[axis]
            }
        }
    };

    // data + scale storage + zero_point storage + shape metadata
    let quantized_bytes = num_elements          // 1 byte per element (i8)
        + num_channels * 4                       // f32 scales
        + num_channels * 4                       // i32 zero_points
        + shape.len() * 8; // shape Vec<usize>

    LayerQuantizationPlan {
        layer_name: layer_name.to_owned(),
        num_elements,
        original_bytes,
        quantized_bytes,
        recommended_mode,
        recommended_scope,
    }
}

/// Aggregate compression estimate across all planned layers.
///
/// Returns `(total_original_bytes, total_quantized_bytes, compression_ratio)`.
pub fn estimate_model_compression(plans: &[LayerQuantizationPlan]) -> (usize, usize, f32) {
    let total_original: usize = plans.iter().map(|p| p.original_bytes).sum();
    let total_quantized: usize = plans.iter().map(|p| p.quantized_bytes).sum();
    let ratio = if total_quantized == 0 {
        0.0
    } else {
        total_original as f32 / total_quantized as f32
    };
    (total_original, total_quantized, ratio)
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Validate that `shape` is non-empty and `data.len() == product(shape)`.
///
/// Generic over the element type so it can guard both the `f32` source buffers
/// handed to the quantizers and the `i8` payload of a hand-built
/// [`QuantizedTensor`]; only the element *count* is ever inspected.
fn validate_shape_data<T>(data: &[T], shape: &[usize]) -> Result<(), DiffusionError> {
    if shape.is_empty() {
        return Err(DiffusionError::InvalidConfig(
            "quantization: shape must not be empty".to_owned(),
        ));
    }
    let expected: usize = shape.iter().product();
    if data.len() != expected {
        return Err(DiffusionError::InvalidConfig(format!(
            "quantization: data length {} does not match shape {:?} (expected {} elements)",
            data.len(),
            shape,
            expected,
        )));
    }
    Ok(())
}

/// Decompose `shape` around `axis` into `(outer, channel, inner)` sizes.
///
/// The element at flat index `i` belongs to channel `(i / inner) % channel`.
fn channel_dims(shape: &[usize], axis: usize) -> Result<(usize, usize, usize), DiffusionError> {
    if axis >= shape.len() {
        return Err(DiffusionError::InvalidConfig(format!(
            "quantization: axis {} is out of bounds for shape {:?}",
            axis, shape
        )));
    }
    let outer_size: usize = shape[..axis].iter().product();
    let channel_size = shape[axis];
    let inner_size: usize = if axis + 1 < shape.len() {
        shape[axis + 1..].iter().product()
    } else {
        1
    };
    // outer_size used by caller to know total element count implicitly
    let _ = outer_size;
    Ok((outer_size, channel_size, inner_size))
}

/// Symmetric quantization of a single `f32` value.
///
/// `q = clamp(round(x / scale), -127, 127)`
#[inline]
fn symmetric_quantize(x: f32, scale: f32) -> i8 {
    let q = (x / scale).round();
    q.clamp(-127.0, 127.0) as i8
}

/// Compute `(scale, zero_point)` for asymmetric min-max quantization.
///
/// `scale = (max − min) / 255.0` (epsilon-guarded against division by zero).
/// `zero_point = clamp(round(−min / scale), 0, 255)`
fn minmax_scale_zp(data: &[f32]) -> (f32, i32) {
    let mut min_val = f32::MAX;
    let mut max_val = f32::MIN;
    for &x in data {
        if x < min_val {
            min_val = x;
        }
        if x > max_val {
            max_val = x;
        }
    }
    if min_val == f32::MAX {
        // empty — degenerate
        return (1.0, 0);
    }
    minmax_scale_zp_from_bounds(min_val, max_val)
}

/// Compute `(scale, zero_point)` from already-known `(min, max)` bounds.
///
/// Shared by [`minmax_scale_zp`] (which derives the bounds from a slice) and
/// [`MinMaxQuantizer::quantize`]'s per-channel path (which accumulates the
/// bounds for every channel in a single pass over the whole tensor, rather
/// than calling [`minmax_scale_zp`] once per channel on a freshly collected
/// per-channel `Vec`).
fn minmax_scale_zp_from_bounds(min_val: f32, max_val: f32) -> (f32, i32) {
    let range = max_val - min_val;
    let scale = if range == 0.0 { 1.0_f32 } else { range / 255.0 };
    let zp_f = (-min_val / scale).round();
    let zero_point = zp_f.clamp(0.0, 255.0) as i32;
    (scale, zero_point)
}

/// Asymmetric quantization of a single `f32` value.
///
/// The result is a `u8` in `[0, 255]` bit-cast to `i8`.
#[inline]
fn asymmetric_quantize(x: f32, scale: f32, zero_point: i32) -> i8 {
    let q = (x / scale).round() + zero_point as f32;
    let q_u8 = q.clamp(0.0, 255.0) as u8;
    q_u8 as i8
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Symmetric / AbsmaxQuantizer
    // -------------------------------------------------------------------------

    #[test]
    fn test_symmetric_quantize_simple_values() {
        // max_abs = 1.0 → scale = 1/127 ≈ 0.00787
        let data = vec![1.0_f32, -1.0, 0.5, -0.5, 0.0];
        let shape = vec![5];
        let qt = AbsmaxQuantizer::per_tensor()
            .quantize(&data, &shape)
            .expect("quantize");
        // 1.0 / (1.0/127) = 127, -1.0 → -127, 0.5 → 63 or 64
        assert_eq!(qt.data[0], 127);
        assert_eq!(qt.data[1], -127);
        assert!(qt.data[2] == 63 || qt.data[2] == 64);
        assert_eq!(qt.data[4], 0);
        assert_eq!(qt.mode, QuantizationMode::Symmetric);
        assert_eq!(qt.zero_points[0], 0);
    }

    #[test]
    fn test_symmetric_dequantize_roundtrip() {
        let data: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) / 32.0).collect();
        let shape = vec![64];
        let qt = AbsmaxQuantizer::per_tensor()
            .quantize(&data, &shape)
            .expect("quantize");
        let recovered = qt.dequantize().expect("dequantize");
        assert_eq!(recovered.len(), data.len());
        // Symmetric INT8: max error ≤ scale/2 ≈ max_abs/127/2
        let max_err = data
            .iter()
            .zip(recovered.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        let scale = qt.scales[0];
        assert!(
            max_err <= scale / 2.0 + f32::EPSILON,
            "max_err={max_err}, scale/2={}",
            scale / 2.0
        );
    }

    #[test]
    fn test_symmetric_all_zeros() {
        let data = vec![0.0_f32; 8];
        let shape = vec![8];
        let qt = AbsmaxQuantizer::per_tensor()
            .quantize(&data, &shape)
            .expect("quantize");
        // Scale defaults to 1.0 when max_abs==0
        assert_eq!(qt.scales[0], 1.0);
        assert!(qt.data.iter().all(|&x| x == 0));
        let recovered = qt.dequantize().expect("dequantize");
        assert!(recovered.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_symmetric_max_abs_scale() {
        let data = vec![3.0_f32, -3.0, 1.5];
        let shape = vec![3];
        let qt = AbsmaxQuantizer::per_tensor()
            .quantize(&data, &shape)
            .expect("quantize");
        let expected_scale = 3.0 / 127.0;
        assert!(
            (qt.scales[0] - expected_scale).abs() < 1e-7,
            "scale={} expected={}",
            qt.scales[0],
            expected_scale
        );
        assert_eq!(qt.data[0], 127);
        assert_eq!(qt.data[1], -127);
    }

    // -------------------------------------------------------------------------
    // Asymmetric / MinMaxQuantizer
    // -------------------------------------------------------------------------

    #[test]
    fn test_asymmetric_quantize_positive_range() {
        // All positive: [0, 1]
        let data = vec![0.0_f32, 0.25, 0.5, 0.75, 1.0];
        let shape = vec![5];
        let qt = MinMaxQuantizer::per_tensor()
            .quantize(&data, &shape)
            .expect("quantize");
        assert_eq!(qt.mode, QuantizationMode::Asymmetric);
        // scale = 1.0/255, zp=0
        let scale = qt.scales[0];
        let zp = qt.zero_points[0];
        assert!((scale - 1.0 / 255.0).abs() < 1e-7, "scale={scale}");
        assert_eq!(zp, 0);
        // First element should dequantize to ~0
        let dq = qt.dequantize().expect("dequantize");
        assert!(dq[0].abs() < scale, "dq[0]={}", dq[0]);
    }

    #[test]
    fn test_asymmetric_quantize_mixed_range() {
        // Mixed: [-1, 1] → range=2, scale=2/255, zp=round(1/(2/255))=127
        let data = vec![-1.0_f32, 0.0, 1.0];
        let shape = vec![3];
        let qt = MinMaxQuantizer::per_tensor()
            .quantize(&data, &shape)
            .expect("quantize");
        let scale = qt.scales[0];
        let zp = qt.zero_points[0];
        assert!((scale - 2.0 / 255.0).abs() < 1e-6, "scale={scale}");
        // zp = round(1.0 / scale) clamped to [0,255]
        let expected_zp = (1.0_f32 / scale).round().clamp(0.0, 255.0) as i32;
        assert_eq!(zp, expected_zp, "zp={zp} expected={expected_zp}");
    }

    #[test]
    fn test_asymmetric_dequantize_roundtrip() {
        let data: Vec<f32> = (0..64).map(|i| i as f32 * 0.01 - 0.3).collect();
        let shape = vec![64];
        let qt = MinMaxQuantizer::per_tensor()
            .quantize(&data, &shape)
            .expect("quantize");
        let recovered = qt.dequantize().expect("dequantize");
        let max_err = data
            .iter()
            .zip(recovered.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        let scale = qt.scales[0];
        // Asymmetric: max error ≤ scale (rounding)
        assert!(max_err <= scale + 1e-5, "max_err={max_err}, scale={scale}");
    }

    // -------------------------------------------------------------------------
    // Scope tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_per_tensor_scope() {
        let data = vec![0.1_f32, -0.5, 0.3, 0.8, -0.2, 0.0];
        let shape = vec![2, 3];
        let qt = AbsmaxQuantizer::per_tensor()
            .quantize(&data, &shape)
            .expect("quantize");
        assert_eq!(qt.scales.len(), 1);
        assert_eq!(qt.zero_points.len(), 1);
        assert!(matches!(qt.scope, QuantizationScope::PerTensor));
    }

    #[test]
    fn test_per_channel_scope_2d() {
        // shape [3, 4], axis 0 → 3 channels, each with 4 elements
        let data: Vec<f32> = (0..12).map(|i| (i as f32 - 5.5) / 6.0).collect();
        let shape = vec![3, 4];
        let qt = AbsmaxQuantizer::per_channel(0)
            .quantize(&data, &shape)
            .expect("quantize");
        assert_eq!(qt.scales.len(), 3, "should have 3 channel scales");
        assert_eq!(qt.zero_points.len(), 3);
        assert!(matches!(qt.scope, QuantizationScope::PerChannel(0)));

        // Each channel's scale should be computed from its own 4 values
        // Channel 0: elements 0-3
        let ch0_max_abs = data[0..4].iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
        let expected_s0 = ch0_max_abs / 127.0;
        assert!(
            (qt.scales[0] - expected_s0).abs() < 1e-7,
            "ch0 scale={} expected={}",
            qt.scales[0],
            expected_s0
        );

        // Roundtrip should be accurate
        let recovered = qt.dequantize().expect("dequantize");
        let max_err = data
            .iter()
            .zip(recovered.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(max_err < 0.02, "per-channel roundtrip max_err={max_err}");
    }

    // Regression test for the per-channel O(n * channel_size) -> O(n)
    // refactor: MinMaxQuantizer's PerChannel path had no prior test
    // coverage at all. Verify each channel's (scale, zero_point) is
    // computed from *its own* values only, not the whole tensor.
    #[test]
    fn test_minmax_per_channel_scope_2d() {
        // shape [2, 3], axis 0 -> 2 channels of 3 elements each, with very
        // different ranges so a cross-channel mix-up would be detectable.
        let data = vec![0.0_f32, 0.5, 1.0, -10.0, -5.0, 0.0];
        let shape = vec![2, 3];
        let qt = MinMaxQuantizer::new(QuantizationScope::PerChannel(0))
            .quantize(&data, &shape)
            .expect("quantize");
        assert_eq!(qt.scales.len(), 2);
        assert_eq!(qt.zero_points.len(), 2);

        // Channel 0: [0.0, 0.5, 1.0] -> range 1.0 -> scale = 1.0/255.
        assert!(
            (qt.scales[0] - 1.0 / 255.0).abs() < 1e-6,
            "ch0 scale={}",
            qt.scales[0]
        );
        // Channel 1: [-10.0, -5.0, 0.0] -> range 10.0 -> scale = 10.0/255.
        assert!(
            (qt.scales[1] - 10.0 / 255.0).abs() < 1e-6,
            "ch1 scale={}",
            qt.scales[1]
        );

        let recovered = qt.dequantize().expect("dequantize");
        let max_err = data
            .iter()
            .zip(recovered.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(max_err < 0.1, "per-channel roundtrip max_err={max_err}");
    }

    // Cross-check: per-channel results must be identical to (slow)
    // per-channel-at-a-time computation, for both quantizers, guarding
    // against the single-pass refactor silently mixing channels together.
    #[test]
    fn test_per_channel_matches_naive_per_channel_computation() {
        let data: Vec<f32> = (0..24).map(|i| ((i * 37) % 23) as f32 - 11.0).collect();
        let shape = vec![4, 6]; // axis 0 -> 4 channels of 6 elements each.

        let absmax_qt = AbsmaxQuantizer::per_channel(0)
            .quantize(&data, &shape)
            .expect("quantize");
        let minmax_qt = MinMaxQuantizer::new(QuantizationScope::PerChannel(0))
            .quantize(&data, &shape)
            .expect("quantize");

        for c in 0..4 {
            let channel_slice = &data[c * 6..(c + 1) * 6];

            let naive_max_abs = channel_slice
                .iter()
                .map(|x| x.abs())
                .fold(0.0_f32, f32::max);
            let naive_absmax_scale = if naive_max_abs == 0.0 {
                1.0
            } else {
                naive_max_abs / 127.0
            };
            assert!(
                (absmax_qt.scales[c] - naive_absmax_scale).abs() < 1e-6,
                "absmax channel {c}: got {}, expected {naive_absmax_scale}",
                absmax_qt.scales[c]
            );

            let naive_min = channel_slice.iter().cloned().fold(f32::MAX, f32::min);
            let naive_max = channel_slice.iter().cloned().fold(f32::MIN, f32::max);
            let naive_range = naive_max - naive_min;
            let naive_minmax_scale = if naive_range == 0.0 {
                1.0
            } else {
                naive_range / 255.0
            };
            assert!(
                (minmax_qt.scales[c] - naive_minmax_scale).abs() < 1e-6,
                "minmax channel {c}: got {}, expected {naive_minmax_scale}",
                minmax_qt.scales[c]
            );
        }
    }

    // -------------------------------------------------------------------------
    // Memory / compression
    // -------------------------------------------------------------------------

    #[test]
    fn test_compression_ratio_4x() {
        // Large flat tensor: overhead is negligible → ratio ≈ 4.0
        let n = 10_000_usize;
        let data = vec![0.5_f32; n];
        let shape = vec![n];
        let qt = AbsmaxQuantizer::per_tensor()
            .quantize(&data, &shape)
            .expect("quantize");
        let ratio = qt.compression_ratio_vs_f32();
        // overhead = 1*4 (scale) + 1*4 (zp) + 1*8 (shape) = 16 bytes
        // memory_bytes = n + 16
        // ratio = n*4 / (n + 16) → for n=10000 → 40000/10016 ≈ 3.99
        assert!(ratio > 3.9, "ratio={ratio}");
        assert!(ratio < 4.1, "ratio={ratio}");
    }

    #[test]
    fn test_memory_bytes() {
        let data = vec![1.0_f32, 2.0, 3.0, 4.0];
        let shape = vec![2, 2];
        let qt = AbsmaxQuantizer::per_tensor()
            .quantize(&data, &shape)
            .expect("quantize");
        // data=4, scales=1*4=4, zero_points=1*4=4, shape=2*8=16 → total=28
        let expected = 4 + 4 + 4 + 16;
        assert_eq!(qt.memory_bytes(), expected, "memory_bytes mismatch");
    }

    // -------------------------------------------------------------------------
    // Metrics
    // -------------------------------------------------------------------------

    #[test]
    fn test_metrics_zero_error() {
        // If we quantize and dequantize the same data, noise should be very small
        let data: Vec<f32> = (0..64).map(|i| i as f32 / 64.0).collect();
        let shape = vec![64];
        let qt = AbsmaxQuantizer::per_tensor()
            .quantize(&data, &shape)
            .expect("quantize");
        let metrics = compute_quantization_metrics(&data, &qt).expect("metrics");
        // INT8 should have low MAE
        assert!(
            metrics.mean_abs_error < 0.01,
            "mae={}",
            metrics.mean_abs_error
        );
        // SNR should be high (> 40 dB)
        assert!(
            metrics.signal_to_noise_ratio_db > 40.0,
            "snr={}",
            metrics.signal_to_noise_ratio_db
        );
    }

    #[test]
    fn test_metrics_snr_high_for_fp32() {
        // Quantizing a smooth signal should give SNR > 40 dB
        let data: Vec<f32> = (0..256)
            .map(|i| ((i as f32) * std::f32::consts::PI / 128.0).sin())
            .collect();
        let shape = vec![256];
        let qt = AbsmaxQuantizer::per_tensor()
            .quantize(&data, &shape)
            .expect("quantize");
        let metrics = compute_quantization_metrics(&data, &qt).expect("metrics");
        assert!(
            metrics.signal_to_noise_ratio_db > 40.0,
            "snr={}",
            metrics.signal_to_noise_ratio_db
        );
    }

    #[test]
    fn test_metrics_noise_zero_gives_infinity() {
        // All-zero tensor: quantize → all zeros → noise_rms=0 → SNR=Inf
        let data = vec![0.0_f32; 16];
        let shape = vec![16];
        let qt = AbsmaxQuantizer::per_tensor()
            .quantize(&data, &shape)
            .expect("quantize");
        let metrics = compute_quantization_metrics(&data, &qt).expect("metrics");
        assert!(
            metrics.signal_to_noise_ratio_db.is_infinite(),
            "expected Inf SNR, got {}",
            metrics.signal_to_noise_ratio_db
        );
    }

    // -------------------------------------------------------------------------
    // Planning
    // -------------------------------------------------------------------------

    #[test]
    fn test_plan_weights_symmetric() {
        // 2-D weight: should get Symmetric + PerChannel(0)
        let plan = plan_layer_quantization("conv1.weight", &[64, 128], true);
        assert_eq!(plan.recommended_mode, QuantizationMode::Symmetric);
        assert!(
            matches!(plan.recommended_scope, QuantizationScope::PerChannel(0)),
            "scope={:?}",
            plan.recommended_scope
        );
        assert_eq!(plan.num_elements, 64 * 128);
        assert_eq!(plan.original_bytes, 64 * 128 * 4);
        // quantized_bytes < original_bytes (meaningful compression)
        assert!(plan.quantized_bytes < plan.original_bytes);
    }

    #[test]
    fn test_plan_weights_1d_symmetric_per_tensor() {
        // 1-D bias: should get Symmetric + PerTensor
        let plan = plan_layer_quantization("conv1.bias", &[64], true);
        assert_eq!(plan.recommended_mode, QuantizationMode::Symmetric);
        assert!(
            matches!(plan.recommended_scope, QuantizationScope::PerTensor),
            "scope={:?}",
            plan.recommended_scope
        );
    }

    #[test]
    fn test_plan_activations_asymmetric() {
        // Activation: Asymmetric + PerTensor
        let plan = plan_layer_quantization("relu_out", &[1, 256, 64, 64], false);
        assert_eq!(plan.recommended_mode, QuantizationMode::Asymmetric);
        assert!(
            matches!(plan.recommended_scope, QuantizationScope::PerTensor),
            "scope={:?}",
            plan.recommended_scope
        );
    }

    #[test]
    fn test_estimate_model_compression() {
        let plans = vec![
            plan_layer_quantization("layer1.weight", &[512, 512], true),
            plan_layer_quantization("layer1.bias", &[512], true),
            plan_layer_quantization("layer2.weight", &[256, 512], true),
        ];
        let (orig, quant, ratio) = estimate_model_compression(&plans);
        assert!(
            orig > quant,
            "original should be larger: orig={orig}, quant={quant}"
        );
        assert!(ratio > 1.0, "ratio should be > 1: ratio={ratio}");
        // Verify totals match sum
        let expected_orig: usize = plans.iter().map(|p| p.original_bytes).sum();
        assert_eq!(orig, expected_orig);
    }

    // -------------------------------------------------------------------------
    // Error handling
    // -------------------------------------------------------------------------

    #[test]
    fn test_invalid_shape_data_mismatch() {
        let data = vec![1.0_f32, 2.0, 3.0];
        let shape = vec![2, 3]; // expects 6 elements, only 3 provided
        let result = AbsmaxQuantizer::per_tensor().quantize(&data, &shape);
        assert!(result.is_err(), "should fail on shape/data mismatch");
        match result {
            Err(DiffusionError::InvalidConfig(_)) => {}
            Err(other) => panic!("wrong error variant: {other:?}"),
            Ok(_) => panic!("expected error"),
        }
    }

    #[test]
    fn test_invalid_empty_shape() {
        let data: Vec<f32> = vec![];
        let shape: Vec<usize> = vec![];
        let result = AbsmaxQuantizer::per_tensor().quantize(&data, &shape);
        assert!(result.is_err());
        match result {
            Err(DiffusionError::InvalidConfig(_)) => {}
            Err(other) => panic!("wrong error variant: {other:?}"),
            Ok(_) => panic!("expected error"),
        }
    }

    // -------------------------------------------------------------------------
    // QuantizedTensor::validate / dequantize error handling
    // -------------------------------------------------------------------------

    // Regression tests: every `QuantizedTensor` field is `pub`, so a
    // hand-built tensor with inconsistent metadata used to make
    // `dequantize()` panic (out-of-bounds `scales[0]` / `shape[axis+1..]` /
    // `scales[channel]`) instead of returning a typed error.

    #[test]
    fn test_validate_rejects_empty_shape() {
        let qt = QuantizedTensor {
            data: vec![],
            scales: vec![1.0],
            zero_points: vec![0],
            shape: vec![],
            mode: QuantizationMode::Symmetric,
            scope: QuantizationScope::PerTensor,
        };
        assert!(matches!(
            qt.validate(),
            Err(DiffusionError::InvalidConfig(_))
        ));
    }

    #[test]
    fn test_validate_rejects_data_shape_mismatch() {
        let qt = QuantizedTensor {
            data: vec![1, 2, 3], // 3 elements
            scales: vec![1.0],
            zero_points: vec![0],
            shape: vec![4], // declares 4
            mode: QuantizationMode::Symmetric,
            scope: QuantizationScope::PerTensor,
        };
        assert!(matches!(
            qt.validate(),
            Err(DiffusionError::InvalidConfig(_))
        ));
    }

    #[test]
    fn test_validate_rejects_wrong_scales_len_per_tensor() {
        let qt = QuantizedTensor {
            data: vec![1, 2, 3, 4],
            scales: vec![1.0, 2.0], // PerTensor needs exactly 1
            zero_points: vec![0],
            shape: vec![4],
            mode: QuantizationMode::Symmetric,
            scope: QuantizationScope::PerTensor,
        };
        assert!(matches!(
            qt.validate(),
            Err(DiffusionError::InvalidConfig(_))
        ));
    }

    #[test]
    fn test_validate_rejects_wrong_scales_len_per_channel() {
        // shape [4, 2], axis 0 -> needs 4 scales, only 2 given.
        let qt = QuantizedTensor {
            data: vec![1, 2, 3, 4, 5, 6, 7, 8],
            scales: vec![1.0, 2.0],
            zero_points: vec![0, 0],
            shape: vec![4, 2],
            mode: QuantizationMode::Symmetric,
            scope: QuantizationScope::PerChannel(0),
        };
        assert!(matches!(
            qt.validate(),
            Err(DiffusionError::InvalidConfig(_))
        ));
    }

    #[test]
    fn test_validate_rejects_axis_out_of_bounds() {
        let qt = QuantizedTensor {
            data: vec![1, 2, 3, 4],
            scales: vec![1.0],
            zero_points: vec![0],
            shape: vec![4],
            mode: QuantizationMode::Symmetric,
            scope: QuantizationScope::PerChannel(5), // shape has only 1 dim
        };
        assert!(matches!(
            qt.validate(),
            Err(DiffusionError::InvalidConfig(_))
        ));
    }

    #[test]
    fn test_validate_accepts_well_formed_tensor() {
        let data = vec![0.1_f32, -0.5, 0.3, 0.8];
        let shape = vec![4];
        let qt = AbsmaxQuantizer::per_tensor()
            .quantize(&data, &shape)
            .expect("quantize");
        assert!(qt.validate().is_ok());
    }

    #[test]
    fn test_dequantize_returns_error_not_panic_for_malformed_tensor() {
        let qt = QuantizedTensor {
            data: vec![1, 2, 3],
            scales: vec![], // wrong: PerTensor needs 1
            zero_points: vec![],
            shape: vec![3],
            mode: QuantizationMode::Symmetric,
            scope: QuantizationScope::PerTensor,
        };
        // Must return Err, not panic on `self.scales[0]`.
        assert!(matches!(
            qt.dequantize(),
            Err(DiffusionError::InvalidConfig(_))
        ));
    }

    #[test]
    fn test_dequantize_returns_error_not_panic_for_out_of_bounds_axis() {
        let qt = QuantizedTensor {
            data: vec![1, 2, 3, 4],
            scales: vec![1.0],
            zero_points: vec![0],
            shape: vec![4],
            mode: QuantizationMode::Symmetric,
            scope: QuantizationScope::PerChannel(9), // wrong: out of bounds
        };
        // Must return Err, not panic on `self.shape[axis]`.
        assert!(matches!(
            qt.dequantize(),
            Err(DiffusionError::InvalidConfig(_))
        ));
    }

    // -------------------------------------------------------------------------
    // compute_quantization_metrics error handling
    // -------------------------------------------------------------------------

    // Regression test: an empty `original` used to silently produce
    // `0.0 / 0.0 = NaN` metrics instead of an error.
    #[test]
    fn test_metrics_rejects_empty_original() {
        let qt = QuantizedTensor {
            data: vec![],
            scales: vec![1.0],
            zero_points: vec![0],
            shape: vec![0],
            mode: QuantizationMode::Symmetric,
            scope: QuantizationScope::PerTensor,
        };
        let result = compute_quantization_metrics(&[], &qt);
        assert!(matches!(result, Err(DiffusionError::InvalidConfig(_))));
    }

    // Regression test: a length mismatch between `original` and
    // `quantized_tensor` used to silently truncate the comparison to the
    // shorter of the two (via `zip`) while still dividing by
    // `original.len()`, under-reporting error instead of failing loudly.
    #[test]
    fn test_metrics_rejects_length_mismatch() {
        let data = vec![0.1_f32, -0.5, 0.3, 0.8];
        let shape = vec![4];
        let qt = AbsmaxQuantizer::per_tensor()
            .quantize(&data, &shape)
            .expect("quantize");
        let shorter_original = vec![0.1_f32, -0.5]; // only 2, tensor has 4
        let result = compute_quantization_metrics(&shorter_original, &qt);
        assert!(matches!(result, Err(DiffusionError::InvalidConfig(_))));
    }

    // -------------------------------------------------------------------------
    // Large tensor
    // -------------------------------------------------------------------------

    #[test]
    fn test_large_tensor_quantization() {
        // 1000 elements, mixed range
        let data: Vec<f32> = (0..1000)
            .map(|i| ((i as f32) / 500.0 - 1.0) * std::f32::consts::PI)
            .collect();
        let shape = vec![1000];
        let qt = AbsmaxQuantizer::per_tensor()
            .quantize(&data, &shape)
            .expect("quantize");
        assert_eq!(qt.data.len(), 1000);
        assert_eq!(qt.num_elements(), 1000);

        let recovered = qt.dequantize().expect("dequantize");
        assert_eq!(recovered.len(), 1000);

        let metrics = compute_quantization_metrics(&data, &qt).expect("metrics");
        // Large smooth signal: SNR should be well above 40 dB
        assert!(
            metrics.signal_to_noise_ratio_db > 40.0,
            "snr={}",
            metrics.signal_to_noise_ratio_db
        );
    }
}
