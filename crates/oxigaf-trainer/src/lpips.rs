//! LPIPS (Learned Perceptual Image Patch Similarity) loss using VGG features.
//!
//! Pure Rust implementation using Candle for VGG16 feature extraction.
//! LPIPS measures perceptual similarity by comparing features from pretrained
//! VGG network layers.
//!
//! # Architecture
//!
//! Uses VGG16 layers:
//! - `conv1_2` (layer 1, relu output) - 64 channels
//! - `conv2_2` (layer 4, relu output) - 128 channels
//! - `conv3_3` (layer 7, relu output) - 256 channels
//! - `conv4_3` (layer 9, relu output) - 512 channels
//! - `conv5_3` (layer 11, relu output) - 512 channels
//!
//! Features are normalized and compared using learned linear weights.

use candle_core::{DType, Device, Result as CandleResult, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, Module, VarBuilder};
use std::path::Path;

use crate::TrainerError;

// ---------------------------------------------------------------------------
// VGG16 Feature Extractor
// ---------------------------------------------------------------------------

/// VGG16 block with configurable number of convolution layers.
struct VggBlock {
    convs: Vec<Conv2d>,
}

impl VggBlock {
    fn new(
        vb: VarBuilder<'_>,
        num_convs: usize,
        in_channels: usize,
        out_channels: usize,
    ) -> CandleResult<Self> {
        let config = Conv2dConfig {
            padding: 1,
            stride: 1,
            ..Default::default()
        };

        let mut convs = Vec::with_capacity(num_convs);
        for i in 0..num_convs {
            let c_in = if i == 0 { in_channels } else { out_channels };
            let conv = Conv2d::new(
                vb.get((out_channels, c_in, 3, 3), &format!("conv{i}.weight"))?,
                Some(vb.get(out_channels, &format!("conv{i}.bias"))?),
                config,
            );
            convs.push(conv);
        }

        Ok(Self { convs })
    }

    /// Forward pass through block, returning all relu outputs.
    fn forward_with_intermediates(&self, x: &Tensor) -> CandleResult<Vec<Tensor>> {
        let mut outputs = Vec::with_capacity(self.convs.len());
        let mut current = x.clone();

        for conv in &self.convs {
            current = conv.forward(&current)?;
            current = current.relu()?;
            outputs.push(current.clone());
        }

        Ok(outputs)
    }
}

/// VGG16 feature extractor for LPIPS computation.
///
/// Extracts features from 5 layers (relu outputs after conv1_2, conv2_2,
/// conv3_3, conv4_3, conv5_3).
pub struct VggFeatureExtractor {
    block1: VggBlock,
    block2: VggBlock,
    block3: VggBlock,
    block4: VggBlock,
    block5: VggBlock,
    device: Device,
}

impl VggFeatureExtractor {
    /// Create a new VGG feature extractor with weights loaded from safetensors.
    ///
    /// # Safety
    ///
    /// This memory-maps `weights_path` for the lifetime of the returned
    /// extractor's internal tensors. The caller must ensure the file is not
    /// truncated, modified, or removed for as long as this extractor (or any
    /// tensor derived from it) is alive — a racing write to a memory-mapped
    /// file is undefined behaviour (it can surface as `SIGBUS` or silently
    /// torn/garbage reads), not a catchable `Err`. For weight files of
    /// unknown provenance or files that may be concurrently written, prefer
    /// loading via [`from_varbuilder`](Self::from_varbuilder) with a
    /// `VarBuilder` built from `candle_core::safetensors::load` (which reads
    /// the file fully into memory) instead of this mmap-based path.
    pub fn from_safetensors(weights_path: &Path, device: &Device) -> Result<Self, TrainerError> {
        // SAFETY: `VarBuilder::from_mmaped_safetensors` memory-maps
        // `weights_path`. Per this function's `# Safety` contract (above),
        // the caller guarantees the file is not mutated or truncated for the
        // lifetime of the resulting `VarBuilder`/`VggFeatureExtractor`; under
        // that invariant the mapped bytes stay a stable, valid safetensors
        // image for the duration of this call and of subsequent tensor
        // reads through it.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, device)
                .map_err(|e| TrainerError::Loss(format!("Failed to load VGG weights: {e}")))?
        };

        Self::from_varbuilder(vb, device)
    }

    /// Create from a VarBuilder (for testing or custom weight loading).
    pub fn from_varbuilder(vb: VarBuilder<'_>, device: &Device) -> Result<Self, TrainerError> {
        let block1 = VggBlock::new(vb.pp("block1"), 2, 3, 64)
            .map_err(|e| TrainerError::Loss(format!("Failed to build VGG block1: {e}")))?;
        let block2 = VggBlock::new(vb.pp("block2"), 2, 64, 128)
            .map_err(|e| TrainerError::Loss(format!("Failed to build VGG block2: {e}")))?;
        let block3 = VggBlock::new(vb.pp("block3"), 3, 128, 256)
            .map_err(|e| TrainerError::Loss(format!("Failed to build VGG block3: {e}")))?;
        let block4 = VggBlock::new(vb.pp("block4"), 3, 256, 512)
            .map_err(|e| TrainerError::Loss(format!("Failed to build VGG block4: {e}")))?;
        let block5 = VggBlock::new(vb.pp("block5"), 3, 512, 512)
            .map_err(|e| TrainerError::Loss(format!("Failed to build VGG block5: {e}")))?;

        Ok(Self {
            block1,
            block2,
            block3,
            block4,
            block5,
            device: device.clone(),
        })
    }

    /// Create with random weights (for testing).
    #[cfg(test)]
    pub fn random(device: &Device) -> Result<Self, TrainerError> {
        // Create blocks directly with random convolutions
        let block1 = Self::create_random_block(2, 3, 64, device)?;
        let block2 = Self::create_random_block(2, 64, 128, device)?;
        let block3 = Self::create_random_block(3, 128, 256, device)?;
        let block4 = Self::create_random_block(3, 256, 512, device)?;
        let block5 = Self::create_random_block(3, 512, 512, device)?;

        Ok(Self {
            block1,
            block2,
            block3,
            block4,
            block5,
            device: device.clone(),
        })
    }

    #[cfg(test)]
    fn create_random_block(
        num_convs: usize,
        in_channels: usize,
        out_channels: usize,
        device: &Device,
    ) -> Result<VggBlock, TrainerError> {
        use candle_nn::VarMap;

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);

        let mut convs = Vec::with_capacity(num_convs);
        for i in 0..num_convs {
            let c_in = if i == 0 { in_channels } else { out_channels };
            let conv_config = candle_nn::Conv2dConfig {
                padding: 1,
                ..Default::default()
            };
            // Create conv with random weights using init
            let conv = candle_nn::conv2d(
                c_in,
                out_channels,
                3,
                conv_config,
                vb.pp(format!("conv{i}")),
            )
            .map_err(|e| TrainerError::Loss(format!("Failed to create conv: {e}")))?;
            convs.push(conv);
        }

        Ok(VggBlock { convs })
    }

    /// Extract features from an image tensor.
    ///
    /// # Arguments
    /// * `image` - Tensor of shape `[B, C, H, W]` with values in `[0, 1]`.
    ///
    /// # Returns
    /// Five feature tensors from layers 1, 4, 7, 9, 11 (relu outputs).
    pub fn extract_features(&self, image: &Tensor) -> Result<Vec<Tensor>, TrainerError> {
        // Normalize with ImageNet mean/std
        let mean = Tensor::new(&[0.485_f32, 0.456, 0.406], &self.device)
            .map_err(|e| TrainerError::Loss(format!("Failed to create mean tensor: {e}")))?
            .reshape((1, 3, 1, 1))
            .map_err(|e| TrainerError::Loss(format!("Failed to reshape mean: {e}")))?;

        let std = Tensor::new(&[0.229_f32, 0.224, 0.225], &self.device)
            .map_err(|e| TrainerError::Loss(format!("Failed to create std tensor: {e}")))?
            .reshape((1, 3, 1, 1))
            .map_err(|e| TrainerError::Loss(format!("Failed to reshape std: {e}")))?;

        let normalized = image
            .broadcast_sub(&mean)
            .map_err(|e| TrainerError::Loss(format!("Failed to subtract mean: {e}")))?
            .broadcast_div(&std)
            .map_err(|e| TrainerError::Loss(format!("Failed to divide by std: {e}")))?;

        let mut features = Vec::with_capacity(5);

        // Block 1: extract last relu output
        let block1_outs = self
            .block1
            .forward_with_intermediates(&normalized)
            .map_err(|e| TrainerError::Loss(format!("Block1 forward failed: {e}")))?;
        let last_idx = block1_outs.len().saturating_sub(1);
        let block1_out = block1_outs
            .get(last_idx)
            .ok_or_else(|| TrainerError::Loss("Block1 produced no outputs".into()))?;

        // Max pool (uses reference, avoids clone)
        let pooled1 = max_pool_2x2(block1_out)?;
        features.push(block1_out.clone());

        // Block 2
        let block2_outs = self
            .block2
            .forward_with_intermediates(&pooled1)
            .map_err(|e| TrainerError::Loss(format!("Block2 forward failed: {e}")))?;
        let last_idx = block2_outs.len().saturating_sub(1);
        let block2_out = block2_outs
            .get(last_idx)
            .ok_or_else(|| TrainerError::Loss("Block2 produced no outputs".into()))?;

        let pooled2 = max_pool_2x2(block2_out)?;
        features.push(block2_out.clone());

        // Block 3
        let block3_outs = self
            .block3
            .forward_with_intermediates(&pooled2)
            .map_err(|e| TrainerError::Loss(format!("Block3 forward failed: {e}")))?;
        let last_idx = block3_outs.len().saturating_sub(1);
        let block3_out = block3_outs
            .get(last_idx)
            .ok_or_else(|| TrainerError::Loss("Block3 produced no outputs".into()))?;

        let pooled3 = max_pool_2x2(block3_out)?;
        features.push(block3_out.clone());

        // Block 4
        let block4_outs = self
            .block4
            .forward_with_intermediates(&pooled3)
            .map_err(|e| TrainerError::Loss(format!("Block4 forward failed: {e}")))?;
        let last_idx = block4_outs.len().saturating_sub(1);
        let block4_out = block4_outs
            .get(last_idx)
            .ok_or_else(|| TrainerError::Loss("Block4 produced no outputs".into()))?;

        let pooled4 = max_pool_2x2(block4_out)?;
        features.push(block4_out.clone());

        // Block 5
        let block5_outs = self
            .block5
            .forward_with_intermediates(&pooled4)
            .map_err(|e| TrainerError::Loss(format!("Block5 forward failed: {e}")))?;
        let last_idx = block5_outs.len().saturating_sub(1);
        let block5_out = block5_outs
            .get(last_idx)
            .cloned()
            .ok_or_else(|| TrainerError::Loss("Block5 produced no outputs".into()))?;
        features.push(block5_out);

        Ok(features)
    }

    /// Get the device this extractor runs on.
    pub fn device(&self) -> &Device {
        &self.device
    }
}

/// 2x2 max pooling with stride 2.
fn max_pool_2x2(x: &Tensor) -> Result<Tensor, TrainerError> {
    x.max_pool2d_with_stride(2, 2)
        .map_err(|e| TrainerError::Loss(format!("Max pool failed: {e}")))
}

// ---------------------------------------------------------------------------
// LPIPS Linear Weights
// ---------------------------------------------------------------------------

/// Learned linear weights for LPIPS distance computation.
pub struct LpipsWeights {
    /// Weights for each VGG layer (5 layers).
    /// Each weight is a 1x1 conv that maps channels to 1.
    weights: Vec<Tensor>,
    device: Device,
}

impl LpipsWeights {
    /// Channel counts for each VGG layer.
    const CHANNELS: [usize; 5] = [64, 128, 256, 512, 512];

    /// Create LPIPS weights from safetensors file.
    pub fn from_safetensors(weights_path: &Path, device: &Device) -> Result<Self, TrainerError> {
        let tensors = load_safetensors(weights_path, device)?;

        let mut weights = Vec::with_capacity(5);
        for i in 0..5 {
            let key = format!("lin{i}.weight");
            let w = tensors
                .get(&key)
                .ok_or_else(|| TrainerError::Loss(format!("Missing LPIPS weight: {key}")))?
                .clone();
            weights.push(w);
        }

        Ok(Self {
            weights,
            device: device.clone(),
        })
    }

    /// Create with uniform weights (equal contribution from each layer).
    ///
    /// This is a non-standard fallback for when no learned LPIPS linear
    /// weights are available (see [`Self::from_safetensors`] for the
    /// reference-accurate path). Each channel is weighted by `1/channels`,
    /// so — combined with [`LpipsDistance::compute`]'s spatial-only
    /// normalization — this variant computes a *mean* squared feature
    /// difference across channels for each layer, rather than the learned
    /// weighted *sum* the reference LPIPS formula uses. That keeps layers
    /// with very different channel counts (64 vs. 512) from dominating the
    /// total distance purely due to their width.
    pub fn uniform(device: &Device) -> Result<Self, TrainerError> {
        let mut weights = Vec::with_capacity(5);
        for &channels in &Self::CHANNELS {
            // 1x1 conv: [1, C, 1, 1]
            let w = Tensor::ones((1, channels, 1, 1), DType::F32, device)
                .map_err(|e| TrainerError::Loss(format!("Failed to create uniform weight: {e}")))?;
            // Scale by 1/channels so the per-layer contribution is a mean
            // (not a sum) over channels — see the doc comment above.
            let w = (w / channels as f64)
                .map_err(|e| TrainerError::Loss(format!("Failed to scale weight: {e}")))?;
            weights.push(w);
        }

        Ok(Self {
            weights,
            device: device.clone(),
        })
    }

    /// Get the weights for a specific layer.
    pub fn get(&self, layer_idx: usize) -> Option<&Tensor> {
        self.weights.get(layer_idx)
    }

    /// Get the device.
    pub fn device(&self) -> &Device {
        &self.device
    }
}

// ---------------------------------------------------------------------------
// LPIPS Distance
// ---------------------------------------------------------------------------

/// LPIPS perceptual distance computer.
pub struct LpipsDistance {
    vgg: VggFeatureExtractor,
    weights: LpipsWeights,
}

impl LpipsDistance {
    /// Create LPIPS distance with pretrained weights.
    pub fn new(
        vgg_weights_path: &Path,
        lpips_weights_path: &Path,
        device: &Device,
    ) -> Result<Self, TrainerError> {
        let vgg = VggFeatureExtractor::from_safetensors(vgg_weights_path, device)?;
        let weights = LpipsWeights::from_safetensors(lpips_weights_path, device)?;

        Ok(Self { vgg, weights })
    }

    /// Create with uniform weights (no learned linear layer).
    pub fn with_uniform_weights(
        vgg_weights_path: &Path,
        device: &Device,
    ) -> Result<Self, TrainerError> {
        let vgg = VggFeatureExtractor::from_safetensors(vgg_weights_path, device)?;
        let weights = LpipsWeights::uniform(device)?;

        Ok(Self { vgg, weights })
    }

    /// Create with random VGG and uniform weights (for testing).
    #[cfg(test)]
    pub fn random(device: &Device) -> Result<Self, TrainerError> {
        let vgg = VggFeatureExtractor::random(device)?;
        let weights = LpipsWeights::uniform(device)?;

        Ok(Self { vgg, weights })
    }

    /// Compute LPIPS distance between two images.
    ///
    /// # Arguments
    /// * `pred` - Predicted image, shape `[B, C, H, W]` with values in `[0, 1]`.
    /// * `target` - Target image, shape `[B, C, H, W]` with values in `[0, 1]`.
    ///
    /// # Returns
    /// Scalar LPIPS distance (lower = more similar).
    pub fn compute(&self, pred: &Tensor, target: &Tensor) -> Result<f32, TrainerError> {
        let pred_features = self.vgg.extract_features(pred)?;
        let target_features = self.vgg.extract_features(target)?;

        let mut total_dist = 0.0_f32;

        for (i, (pf, tf)) in pred_features.iter().zip(target_features.iter()).enumerate() {
            // Unit normalize along channel dimension
            let pf_norm = unit_normalize(pf)?;
            let tf_norm = unit_normalize(tf)?;

            // Squared difference
            let diff = pf_norm
                .sub(&tf_norm)
                .map_err(|e| TrainerError::Loss(format!("Feature diff failed: {e}")))?;
            let diff_sq = diff
                .sqr()
                .map_err(|e| TrainerError::Loss(format!("Feature sqr failed: {e}")))?;

            // Apply learned weights (1x1 conv)
            let weight = self
                .weights
                .get(i)
                .ok_or_else(|| TrainerError::Loss(format!("Missing weight for layer {i}")))?;

            // Weighted sum: sum over channels
            let weighted = diff_sq
                .broadcast_mul(weight)
                .map_err(|e| TrainerError::Loss(format!("Weight mul failed: {e}")))?;

            // Sum over C, then mean over H, W, B
            let layer_dist = weighted
                .sum_all()
                .map_err(|e| TrainerError::Loss(format!("Sum failed: {e}")))?
                .to_scalar::<f32>()
                .map_err(|e| TrainerError::Loss(format!("To scalar failed: {e}")))?;

            // Normalize by batch × spatial dimensions only (B×H×W), matching
            // the reference LPIPS formula: the weighted squared difference is
            // *summed* over channels (that sum already happened via the
            // `broadcast_mul` + `sum_all` above, folding in the per-channel
            // weight) and only *averaged* over the spatial extent. Dividing
            // by the channel count too (i.e. by the full B×C×H×W element
            // count) would silently rescale every layer by 1/C, and C
            // differs per VGG layer (64, 128, 256, 512, 512), so the
            // relative weighting between layers would be wrong.
            let dims = diff_sq.dims();
            let denom = if dims.len() == 4 {
                // NCHW: batch × height × width, excluding the channel axis.
                (dims[0] * dims[2] * dims[3]).max(1)
            } else {
                dims.iter().product::<usize>().max(1)
            };
            total_dist += layer_dist / denom as f32;
        }

        Ok(total_dist)
    }

    /// Get the underlying device.
    pub fn device(&self) -> &Device {
        self.vgg.device()
    }
}

/// Unit normalize tensor along channel dimension (dim=1).
fn unit_normalize(x: &Tensor) -> Result<Tensor, TrainerError> {
    // L2 norm along channel dimension
    let x_sq = x
        .sqr()
        .map_err(|e| TrainerError::Loss(format!("Sqr failed: {e}")))?;

    let norm = x_sq
        .sum_keepdim(1)
        .map_err(|e| TrainerError::Loss(format!("Sum failed: {e}")))?
        .sqrt()
        .map_err(|e| TrainerError::Loss(format!("Sqrt failed: {e}")))?;

    // Add epsilon for stability
    let eps = 1e-10_f64;
    let norm_safe = (norm + eps).map_err(|e| TrainerError::Loss(format!("Add eps failed: {e}")))?;

    x.broadcast_div(&norm_safe)
        .map_err(|e| TrainerError::Loss(format!("Normalize failed: {e}")))
}

// ---------------------------------------------------------------------------
// Convenience function for CPU computation
// ---------------------------------------------------------------------------

/// Compute LPIPS loss from flat f32 images (HWC layout).
///
/// This is a convenience wrapper for use in the loss computation pipeline.
/// For repeated calls, prefer using `LpipsDistance` directly.
///
/// # Arguments
/// * `pred` - Predicted image as flat f32 array in HWC layout, values in `[0, 1]`.
/// * `target` - Target image as flat f32 array in HWC layout, values in `[0, 1]`.
/// * `width` - Image width.
/// * `height` - Image height.
/// * `lpips` - Pre-initialized LPIPS distance computer.
///
/// # Returns
/// LPIPS distance (lower = more similar).
pub fn lpips_loss(
    pred: &[f32],
    target: &[f32],
    width: usize,
    height: usize,
    lpips: &LpipsDistance,
) -> Result<f32, TrainerError> {
    let device = lpips.device();
    let channels = 3;
    let expected_len = width * height * channels;

    if pred.len() < expected_len || target.len() < expected_len {
        return Err(TrainerError::ImageDimensionMismatch {
            expected: expected_len,
            actual: pred.len().min(target.len()),
        });
    }

    // Convert HWC to NCHW
    let pred_nchw = hwc_to_nchw(pred, width, height, channels);
    let target_nchw = hwc_to_nchw(target, width, height, channels);

    // Create tensors
    let pred_tensor = Tensor::from_slice(&pred_nchw, (1, channels, height, width), device)
        .map_err(|e| TrainerError::Loss(format!("Failed to create pred tensor: {e}")))?;

    let target_tensor = Tensor::from_slice(&target_nchw, (1, channels, height, width), device)
        .map_err(|e| TrainerError::Loss(format!("Failed to create target tensor: {e}")))?;

    lpips.compute(&pred_tensor, &target_tensor)
}

/// Convert HWC layout to NCHW layout.
fn hwc_to_nchw(data: &[f32], width: usize, height: usize, channels: usize) -> Vec<f32> {
    let mut nchw = vec![0.0_f32; channels * height * width];

    for c in 0..channels {
        for y in 0..height {
            for x in 0..width {
                let hwc_idx = (y * width + x) * channels + c;
                let nchw_idx = c * height * width + y * width + x;
                if hwc_idx < data.len() {
                    nchw[nchw_idx] = data[hwc_idx];
                }
            }
        }
    }

    nchw
}

// ---------------------------------------------------------------------------
// Safetensors Loading
// ---------------------------------------------------------------------------

/// Load tensors from a safetensors file into a HashMap.
fn load_safetensors(
    path: &Path,
    device: &Device,
) -> Result<std::collections::HashMap<String, Tensor>, TrainerError> {
    let data = std::fs::read(path)
        .map_err(|e| TrainerError::Loss(format!("Failed to read safetensors file: {e}")))?;

    let tensors = safetensors::SafeTensors::deserialize(&data)
        .map_err(|e| TrainerError::Loss(format!("Failed to deserialize safetensors: {e}")))?;

    let mut result = std::collections::HashMap::new();
    for (name, view) in tensors.tensors() {
        let shape: Vec<usize> = view.shape().to_vec();

        // Build the tensor via candle's own raw-buffer loader instead of
        // `bytemuck::cast_slice`: `view.data()`'s alignment depends on the
        // file-controlled header length, so a malformed or merely
        // differently-laid-out `.safetensors` file can hand back a slice
        // that is not 4-byte aligned (or not a multiple-of-4 length), which
        // `bytemuck::cast_slice` turns into a process panic instead of an
        // `Err`. `Tensor::from_raw_buffer` checks alignment itself and falls
        // back to a safe copy when needed, and (unlike the old "F32 only"
        // path) natively understands every dtype safetensors can carry
        // (F16/BF16 included), so a half-precision checkpoint loads
        // correctly instead of silently vanishing from the weight map.
        let dtype = DType::try_from(view.dtype()).map_err(|e| {
            TrainerError::Loss(format!(
                "Tensor {name}: unsupported safetensors dtype {:?}: {e}",
                view.dtype()
            ))
        })?;
        let tensor = Tensor::from_raw_buffer(view.data(), dtype, &shape, device)
            .and_then(|t| t.to_dtype(DType::F32))
            .map_err(|e| TrainerError::Loss(format!("Failed to create tensor {name}: {e}")))?;

        result.insert(name.to_string(), tensor);
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::expect_used)]
    fn test_load_safetensors_f32_roundtrip() {
        let device = Device::Cpu;
        let tensor = Tensor::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], (2, 2), &device)
            .expect("failed to build tensor");
        let path = std::env::temp_dir().join(format!(
            "oxigaf_lpips_test_f32_{}_{}.safetensors",
            std::process::id(),
            "roundtrip"
        ));
        tensor
            .save_safetensors("w", &path)
            .expect("failed to save safetensors");

        let loaded = load_safetensors(&path, &device)
            .expect("load_safetensors must not panic or error on a valid F32 file");
        let t = loaded.get("w").expect("missing tensor 'w'");
        assert_eq!(t.dims(), &[2, 2]);
        let data = t.to_vec2::<f32>().expect("to_vec2 failed");
        assert_eq!(data, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    #[allow(clippy::expect_used)]
    fn test_load_safetensors_f16_converts_to_f32() {
        // Regression test: `load_safetensors` used to silently `continue`
        // past any non-F32 tensor (see the "Only support F32 for now"
        // comment this replaced), so an F16 checkpoint would load as a
        // silently incomplete weight map. It should now convert instead.
        let device = Device::Cpu;
        let values = [0.5_f32, -1.25, 2.0, 4.0];
        let tensor = Tensor::from_slice(&values, (4,), &device)
            .expect("failed to build tensor")
            .to_dtype(DType::F16)
            .expect("failed to cast to f16");
        let path = std::env::temp_dir().join(format!(
            "oxigaf_lpips_test_f16_{}_{}.safetensors",
            std::process::id(),
            "convert"
        ));
        tensor
            .save_safetensors("w", &path)
            .expect("failed to save safetensors");

        let loaded = load_safetensors(&path, &device)
            .expect("load_safetensors must convert F16 tensors instead of silently dropping them");
        let t = loaded
            .get("w")
            .expect("F16 tensor should not have been silently skipped");
        assert_eq!(t.dtype(), DType::F32, "loaded tensor should be F32");
        let data = t.to_vec1::<f32>().expect("to_vec1 failed");
        for (a, b) in data.iter().zip(values.iter()) {
            assert!((a - b).abs() < 1e-2, "got {a}, expected {b}");
        }

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_hwc_to_nchw_conversion() {
        // 2x2 RGB image in HWC format
        #[rustfmt::skip]
        let hwc: Vec<f32> = vec![
            // y=0, x=0: RGB
            0.1, 0.2, 0.3,
            // y=0, x=1: RGB
            0.4, 0.5, 0.6,
            // y=1, x=0: RGB
            0.7, 0.8, 0.9,
            // y=1, x=1: RGB
            1.0, 1.1, 1.2,
        ];

        let nchw = hwc_to_nchw(&hwc, 2, 2, 3);

        // Expected NCHW: channel-first
        // R channel: [[0.1, 0.4], [0.7, 1.0]]
        // G channel: [[0.2, 0.5], [0.8, 1.1]]
        // B channel: [[0.3, 0.6], [0.9, 1.2]]
        let expected: Vec<f32> = vec![
            // R
            0.1, 0.4, 0.7, 1.0, // G
            0.2, 0.5, 0.8, 1.1, // B
            0.3, 0.6, 0.9, 1.2,
        ];

        for (i, (got, exp)) in nchw.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-6,
                "Mismatch at index {i}: got {got}, expected {exp}"
            );
        }
    }

    #[test]
    #[allow(clippy::expect_used)]
    fn test_lpips_uniform_weights() {
        let device = Device::Cpu;
        let weights = LpipsWeights::uniform(&device).expect("Failed to create uniform weights");

        assert_eq!(weights.weights.len(), 5);

        // Check each weight has correct shape
        for (i, &channels) in LpipsWeights::CHANNELS.iter().enumerate() {
            let w = weights.get(i).expect("Missing weight");
            let dims = w.dims();
            assert_eq!(dims, &[1, channels, 1, 1]);
        }
    }

    #[test]
    #[ignore = "slow: LPIPS VGG inference on CPU takes ~95s"]
    #[allow(clippy::expect_used)]
    fn test_lpips_with_random_vgg() {
        let device = Device::Cpu;
        let lpips = LpipsDistance::random(&device).expect("Failed to create LPIPS");

        // Create two different 32x32 images
        let img1 = vec![0.5_f32; 32 * 32 * 3];
        let img2 = vec![0.7_f32; 32 * 32 * 3];

        // Compute distance
        let dist = lpips_loss(&img1, &img2, 32, 32, &lpips).expect("LPIPS failed");

        // Distance should be non-negative
        assert!(dist >= 0.0, "LPIPS distance should be non-negative: {dist}");
    }

    #[test]
    #[ignore = "slow: LPIPS VGG inference on CPU takes ~95s"]
    #[allow(clippy::expect_used)]
    fn test_lpips_identical_images() {
        let device = Device::Cpu;
        let lpips = LpipsDistance::random(&device).expect("Failed to create LPIPS");

        // Same image
        let img = vec![0.5_f32; 32 * 32 * 3];

        let dist = lpips_loss(&img, &img, 32, 32, &lpips).expect("LPIPS failed");

        // Distance for identical images should be very small
        assert!(
            dist < 0.01,
            "LPIPS for identical images should be ~0: {dist}"
        );
    }

    #[test]
    #[ignore = "slow: LPIPS VGG inference on CPU takes ~95s"]
    #[allow(clippy::expect_used)]
    fn test_lpips_different_images_nonzero() {
        let device = Device::Cpu;
        let lpips = LpipsDistance::random(&device).expect("Failed to create LPIPS");

        // Very different images
        let img1 = vec![0.0_f32; 32 * 32 * 3];
        let img2 = vec![1.0_f32; 32 * 32 * 3];

        let dist = lpips_loss(&img1, &img2, 32, 32, &lpips).expect("LPIPS failed");

        // Different images should have non-zero distance
        assert!(
            dist > 0.0,
            "LPIPS for different images should be > 0: {dist}"
        );
    }
}
