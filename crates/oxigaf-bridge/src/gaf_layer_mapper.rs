//! Comprehensive layer name mapping for GAF (Generative Avatar Face) models.
//!
//! This module provides bidirectional mapping between ToRSh (safetensors with "/" separators)
//! and OxiGAF (Candle VarBuilder with "." separators) naming conventions for all GAF model
//! components:
//!
//! - **Multi-View U-Net**: ~1000 layers with time/camera embeddings, down/mid/up blocks,
//!   multi-view attention transformers (attn1, attn_cv, attn2, attn_ip)
//! - **VAE**: ~200 layers with encoder/decoder, mid-blocks, quantization layers
//! - **CLIP Image Encoder**: ~300 layers with ViT-H/14 architecture (32 transformer layers)
//! - **Latent Upsampler**: ~100 layers with simplified U-Net for 32×32→64×64 upsampling
//!
//! # Architecture Overview
//!
//! ## Multi-View U-Net Structure
//!
//! ```text
//! Input (4ch latent)
//!   ↓
//! conv_in → time_embedding + camera_embedding
//!   ↓
//! down_blocks[0..3]: ResNet + MultiViewSpatialTransformer + Downsample
//!   ├─ Each transformer block has:
//!   │  - attn1 (self-attention within view)
//!   │  - attn_cv (cross-view attention across N views)
//!   │  - attn2 (text cross-attention, unused in GAF)
//!   │  - attn_ip (IP-Adapter: CLIP image conditioning)
//!   │  - ff (GeGLU feed-forward network)
//!   ↓
//! mid_block: ResNet + Attention + ResNet
//!   ↓
//! up_blocks[0..3]: ResNet + MultiViewSpatialTransformer + Upsample
//!   ↓
//! conv_norm_out + conv_out → Output (4ch latent prediction)
//! ```
//!
//! ## Naming Convention Examples
//!
//! | Component | ToRSh (safetensors) | OxiGAF (Candle) |
//! |-----------|---------------------|------------------|
//! | U-Net time emb | `time_embedding/linear_1/weight` | `time_embedding.linear_1.weight` |
//! | Down block ResNet | `down_blocks/0/resnets/0/norm1/weight` | `down_blocks.0.resnets.0.norm1.weight` |
//! | Self-attention Q | `down_blocks/0/attentions/0/transformer_blocks/0/attn1/to_q/weight` | `down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight` |
//! | VAE encoder | `encoder/down_blocks/0/resnets/0/conv1/weight` | `encoder.down_blocks.0.resnets.0.conv1.weight` |
//! | CLIP encoder | `encoder/layers/0/self_attn/q_proj/weight` | `encoder.layers.0.self_attn.q_proj.weight` |
//!
//! # Usage
//!
//! ```rust
//! use oxigaf_bridge::GafLayerMapper;
//!
//! let mapper = GafLayerMapper::new();
//!
//! // Convert ToRSh → OxiGAF
//! let oxigaf_name = mapper.map_torsh_to_oxigaf("time_embedding/linear_1/weight")?;
//! assert_eq!(oxigaf_name, "time_embedding.linear_1.weight");
//!
//! // Convert OxiGAF → ToRSh
//! let torsh_name = mapper.map_oxigaf_to_torsh("down_blocks.0.resnets.0.norm1.weight")?;
//! assert_eq!(torsh_name, "down_blocks/0/resnets/0/norm1/weight");
//!
//! // Validate coverage
//! let keys = vec!["time_embedding/linear_1/weight".to_string()];
//! mapper.validate_coverage(&keys)?;
//! # Ok::<(), oxigaf_bridge::BridgeError>(())
//! ```

use crate::error::{BridgeError, Result};
use std::collections::HashMap;

/// Comprehensive layer name mapper for GAF models.
///
/// Provides bidirectional mapping between ToRSh (safetensors) and OxiGAF (Candle)
/// naming conventions for all ~2000 layers across U-Net, VAE, CLIP, and Upsampler.
#[derive(Debug, Clone)]
pub struct GafLayerMapper {
    /// ToRSh name → OxiGAF name mapping
    torsh_to_oxigaf: HashMap<String, String>,
    /// OxiGAF name → ToRSh name mapping
    oxigaf_to_torsh: HashMap<String, String>,
}

impl GafLayerMapper {
    /// Create a new GAF layer mapper with all mappings populated.
    ///
    /// This initializes ~2000 bidirectional mappings for:
    /// - Multi-View U-Net (~1000 layers)
    /// - VAE (~200 layers)
    /// - CLIP Image Encoder (~300 layers)
    /// - Latent Upsampler (~100 layers)
    pub fn new() -> Self {
        let mut mapper = Self {
            torsh_to_oxigaf: HashMap::new(),
            oxigaf_to_torsh: HashMap::new(),
        };

        // Populate all mappings
        mapper.add_unet_mappings();
        mapper.add_vae_mappings();
        mapper.add_clip_mappings();
        mapper.add_upsampler_mappings();

        mapper
    }

    /// Add a bidirectional mapping between ToRSh and OxiGAF names.
    fn add_mapping(&mut self, torsh_name: &str, oxigaf_name: &str) {
        self.torsh_to_oxigaf
            .insert(torsh_name.to_string(), oxigaf_name.to_string());
        self.oxigaf_to_torsh
            .insert(oxigaf_name.to_string(), torsh_name.to_string());
    }

    /// Map ToRSh name to OxiGAF name.
    ///
    /// # Arguments
    ///
    /// * `torsh_name` - Layer name in ToRSh format (e.g., `"time_embedding/linear_1/weight"`)
    ///
    /// # Returns
    ///
    /// OxiGAF name (e.g., `"time_embedding.linear_1.weight"`)
    ///
    /// # Errors
    ///
    /// Returns `BridgeError::LayerMapping` if the layer name is not found in the mapping.
    pub fn map_torsh_to_oxigaf(&self, torsh_name: &str) -> Result<String> {
        self.torsh_to_oxigaf
            .get(torsh_name)
            .cloned()
            .ok_or_else(|| {
                BridgeError::LayerMapping(format!(
                    "Unmapped ToRSh layer: '{}'. This layer is not present in the GAF model mapping.",
                    torsh_name
                ))
            })
    }

    /// Map OxiGAF name to ToRSh name.
    ///
    /// # Arguments
    ///
    /// * `oxigaf_name` - Layer name in OxiGAF format (e.g., `"time_embedding.linear_1.weight"`)
    ///
    /// # Returns
    ///
    /// ToRSh name (e.g., `"time_embedding/linear_1/weight"`)
    ///
    /// # Errors
    ///
    /// Returns `BridgeError::LayerMapping` if the layer name is not found in the mapping.
    pub fn map_oxigaf_to_torsh(&self, oxigaf_name: &str) -> Result<String> {
        self.oxigaf_to_torsh
            .get(oxigaf_name)
            .cloned()
            .ok_or_else(|| {
                BridgeError::LayerMapping(format!(
                    "Unmapped OxiGAF layer: '{}'. This layer is not present in the GAF model mapping.",
                    oxigaf_name
                ))
            })
    }

    /// Validate that all provided keys have mappings.
    ///
    /// # Arguments
    ///
    /// * `keys` - List of ToRSh layer names to validate
    ///
    /// # Errors
    ///
    /// Returns `BridgeError::LayerMapping` with a list of unmapped keys if any are missing.
    pub fn validate_coverage(&self, keys: &[String]) -> Result<()> {
        let unmapped: Vec<&String> = keys
            .iter()
            .filter(|k| !self.torsh_to_oxigaf.contains_key(*k))
            .collect();

        if !unmapped.is_empty() {
            return Err(BridgeError::LayerMapping(format!(
                "Found {} unmapped layer(s): {:?}",
                unmapped.len(),
                unmapped
            )));
        }

        Ok(())
    }

    /// Validate bidirectional consistency of all mappings.
    ///
    /// Ensures that for every ToRSh→OxiGAF mapping, there exists a corresponding
    /// OxiGAF→ToRSh reverse mapping.
    ///
    /// # Errors
    ///
    /// Returns `BridgeError::LayerMapping` if any inconsistency is detected.
    pub fn validate_bidirectional(&self) -> Result<()> {
        // Check ToRSh → OxiGAF → ToRSh round-trip
        for (torsh_name, oxigaf_name) in &self.torsh_to_oxigaf {
            let reverse = self.oxigaf_to_torsh.get(oxigaf_name).ok_or_else(|| {
                BridgeError::LayerMapping(format!(
                    "Bidirectional inconsistency: ToRSh '{}' maps to OxiGAF '{}', \
                     but reverse mapping is missing",
                    torsh_name, oxigaf_name
                ))
            })?;

            if reverse != torsh_name {
                return Err(BridgeError::LayerMapping(format!(
                    "Bidirectional inconsistency: ToRSh '{}' maps to OxiGAF '{}', \
                     which maps back to ToRSh '{}' (expected '{}')",
                    torsh_name, oxigaf_name, reverse, torsh_name
                )));
            }
        }

        // Check counts match
        if self.torsh_to_oxigaf.len() != self.oxigaf_to_torsh.len() {
            return Err(BridgeError::LayerMapping(format!(
                "Bidirectional count mismatch: {} ToRSh→OxiGAF mappings but {} OxiGAF→ToRSh mappings",
                self.torsh_to_oxigaf.len(),
                self.oxigaf_to_torsh.len()
            )));
        }

        Ok(())
    }

    /// Get total number of mapped layers.
    pub fn num_mappings(&self) -> usize {
        self.torsh_to_oxigaf.len()
    }

    /// Add all Multi-View U-Net layer mappings.
    ///
    /// Covers ~1000 layers including:
    /// - Time and camera embeddings
    /// - 4 down blocks with ResNet + MultiViewSpatialTransformer
    /// - Mid block with ResNet + Attention + ResNet
    /// - 4 up blocks with ResNet + MultiViewSpatialTransformer
    /// - Input/output convolutions
    fn add_unet_mappings(&mut self) {
        // Input convolution
        self.add_mapping("conv_in/weight", "conv_in.weight");
        self.add_mapping("conv_in/bias", "conv_in.bias");

        // Time embedding
        self.add_mapping(
            "time_embedding/linear_1/weight",
            "time_embedding.linear_1.weight",
        );
        self.add_mapping(
            "time_embedding/linear_1/bias",
            "time_embedding.linear_1.bias",
        );
        self.add_mapping(
            "time_embedding/linear_2/weight",
            "time_embedding.linear_2.weight",
        );
        self.add_mapping(
            "time_embedding/linear_2/bias",
            "time_embedding.linear_2.bias",
        );

        // Camera embedding
        self.add_mapping(
            "camera_embedding/linear_1/weight",
            "camera_embedding.linear_1.weight",
        );
        self.add_mapping(
            "camera_embedding/linear_1/bias",
            "camera_embedding.linear_1.bias",
        );
        self.add_mapping(
            "camera_embedding/linear_2/weight",
            "camera_embedding.linear_2.weight",
        );
        self.add_mapping(
            "camera_embedding/linear_2/bias",
            "camera_embedding.linear_2.bias",
        );

        // Down blocks (4 stages, 2 resnets each, transformer_layers_per_block varies)
        // transformer_layers_per_block = [1, 2, 10, 10]
        let transformer_depths = [1, 2, 10, 10];
        for (i, &depth) in transformer_depths.iter().enumerate() {
            self.add_down_block_mappings(i, depth);
        }

        // Mid block (transformer depth = 10)
        self.add_mid_block_mappings(10);

        // Up blocks (4 stages, 3 resnets each)
        // Reversed depths: [10, 10, 2, 1]
        let up_depths = [10, 10, 2, 1];
        for (i, &depth) in up_depths.iter().enumerate() {
            self.add_up_block_mappings(i, depth);
        }

        // Output convolution
        self.add_mapping("conv_norm_out/weight", "conv_norm_out.weight");
        self.add_mapping("conv_norm_out/bias", "conv_norm_out.bias");
        self.add_mapping("conv_out/weight", "conv_out.weight");
        self.add_mapping("conv_out/bias", "conv_out.bias");
    }

    /// Add down block mappings for a single stage.
    fn add_down_block_mappings(&mut self, stage: usize, transformer_depth: usize) {
        // 2 ResNet blocks per down stage
        for j in 0..2 {
            self.add_resnet_block_mappings(&format!("down_blocks/{}/resnets/{}", stage, j));

            // MultiViewSpatialTransformer
            self.add_spatial_transformer_mappings(
                &format!("down_blocks/{}/attentions/{}", stage, j),
                transformer_depth,
            );
        }

        // Downsampler (only for stages 0, 1, 2)
        if stage < 3 {
            self.add_mapping(
                &format!("down_blocks/{}/downsamplers/0/conv/weight", stage),
                &format!("down_blocks.{}.downsamplers.0.conv.weight", stage),
            );
            self.add_mapping(
                &format!("down_blocks/{}/downsamplers/0/conv/bias", stage),
                &format!("down_blocks.{}.downsamplers.0.conv.bias", stage),
            );
        }
    }

    /// Add mid block mappings.
    fn add_mid_block_mappings(&mut self, transformer_depth: usize) {
        // ResNet block 0
        self.add_resnet_block_mappings("mid_block/resnets/0");

        // Attention
        self.add_spatial_transformer_mappings("mid_block/attentions/0", transformer_depth);

        // ResNet block 1
        self.add_resnet_block_mappings("mid_block/resnets/1");
    }

    /// Add up block mappings for a single stage.
    fn add_up_block_mappings(&mut self, stage: usize, transformer_depth: usize) {
        // 3 ResNet blocks per up stage (layers_per_block + 1)
        for j in 0..3 {
            self.add_resnet_block_mappings(&format!("up_blocks/{}/resnets/{}", stage, j));

            // MultiViewSpatialTransformer
            self.add_spatial_transformer_mappings(
                &format!("up_blocks/{}/attentions/{}", stage, j),
                transformer_depth,
            );
        }

        // Upsampler (only for stages 0, 1, 2)
        if stage < 3 {
            self.add_mapping(
                &format!("up_blocks/{}/upsamplers/0/conv/weight", stage),
                &format!("up_blocks.{}.upsamplers.0.conv.weight", stage),
            );
            self.add_mapping(
                &format!("up_blocks/{}/upsamplers/0/conv/bias", stage),
                &format!("up_blocks.{}.upsamplers.0.conv.bias", stage),
            );
        }
    }

    /// Add ResNet block mappings (used in U-Net down/mid/up blocks).
    fn add_resnet_block_mappings(&mut self, prefix: &str) {
        let oxigaf_prefix = prefix.replace('/', ".");

        // norm1, conv1, time_emb_proj, norm2, conv2
        for layer in &["norm1", "conv1", "time_emb_proj", "norm2", "conv2"] {
            for param in &["weight", "bias"] {
                self.add_mapping(
                    &format!("{}/{}/{}", prefix, layer, param),
                    &format!("{}.{}.{}", oxigaf_prefix, layer, param),
                );
            }
        }

        // conv_shortcut (optional, present when in_ch != out_ch)
        self.add_mapping(
            &format!("{}/conv_shortcut/weight", prefix),
            &format!("{}.conv_shortcut.weight", oxigaf_prefix),
        );
        self.add_mapping(
            &format!("{}/conv_shortcut/bias", prefix),
            &format!("{}.conv_shortcut.bias", oxigaf_prefix),
        );
    }

    /// Add MultiViewSpatialTransformer mappings.
    fn add_spatial_transformer_mappings(&mut self, prefix: &str, depth: usize) {
        let oxigaf_prefix = prefix.replace('/', ".");

        // GroupNorm + proj_in/proj_out
        self.add_mapping(
            &format!("{}/norm/weight", prefix),
            &format!("{}.norm.weight", oxigaf_prefix),
        );
        self.add_mapping(
            &format!("{}/norm/bias", prefix),
            &format!("{}.norm.bias", oxigaf_prefix),
        );
        self.add_mapping(
            &format!("{}/proj_in/weight", prefix),
            &format!("{}.proj_in.weight", oxigaf_prefix),
        );
        self.add_mapping(
            &format!("{}/proj_in/bias", prefix),
            &format!("{}.proj_in.bias", oxigaf_prefix),
        );
        self.add_mapping(
            &format!("{}/proj_out/weight", prefix),
            &format!("{}.proj_out.weight", oxigaf_prefix),
        );
        self.add_mapping(
            &format!("{}/proj_out/bias", prefix),
            &format!("{}.proj_out.bias", oxigaf_prefix),
        );

        // Transformer blocks
        for k in 0..depth {
            self.add_transformer_block_mappings(&format!("{}/transformer_blocks/{}", prefix, k));
        }
    }

    /// Add MultiViewTransformerBlock mappings.
    ///
    /// Each block has:
    /// - norm1 + attn1 (self-attention)
    /// - norm_cv + attn_cv (cross-view attention)
    /// - norm2 + attn2 (text cross-attention)
    /// - norm_ip + attn_ip (IP-Adapter cross-attention)
    /// - norm3 + ff (feed-forward)
    fn add_transformer_block_mappings(&mut self, prefix: &str) {
        let oxigaf_prefix = prefix.replace('/', ".");

        // Self-attention (attn1)
        self.add_mapping(
            &format!("{}/norm1/weight", prefix),
            &format!("{}.norm1.weight", oxigaf_prefix),
        );
        self.add_mapping(
            &format!("{}/norm1/bias", prefix),
            &format!("{}.norm1.bias", oxigaf_prefix),
        );
        self.add_cross_attention_mappings(&format!("{}/attn1", prefix));

        // Cross-view attention (attn_cv)
        self.add_mapping(
            &format!("{}/norm_cv/weight", prefix),
            &format!("{}.norm_cv.weight", oxigaf_prefix),
        );
        self.add_mapping(
            &format!("{}/norm_cv/bias", prefix),
            &format!("{}.norm_cv.bias", oxigaf_prefix),
        );
        self.add_cross_attention_mappings(&format!("{}/attn_cv", prefix));

        // Text cross-attention (attn2)
        self.add_mapping(
            &format!("{}/norm2/weight", prefix),
            &format!("{}.norm2.weight", oxigaf_prefix),
        );
        self.add_mapping(
            &format!("{}/norm2/bias", prefix),
            &format!("{}.norm2.bias", oxigaf_prefix),
        );
        self.add_cross_attention_mappings(&format!("{}/attn2", prefix));

        // IP-Adapter cross-attention (attn_ip)
        self.add_mapping(
            &format!("{}/norm_ip/weight", prefix),
            &format!("{}.norm_ip.weight", oxigaf_prefix),
        );
        self.add_mapping(
            &format!("{}/norm_ip/bias", prefix),
            &format!("{}.norm_ip.bias", oxigaf_prefix),
        );
        self.add_cross_attention_mappings(&format!("{}/attn_ip", prefix));

        // Feed-forward (ff)
        self.add_mapping(
            &format!("{}/norm3/weight", prefix),
            &format!("{}.norm3.weight", oxigaf_prefix),
        );
        self.add_mapping(
            &format!("{}/norm3/bias", prefix),
            &format!("{}.norm3.bias", oxigaf_prefix),
        );
        self.add_feed_forward_mappings(&format!("{}/ff", prefix));
    }

    /// Add CrossAttention mappings (to_q, to_k, to_v, to_out).
    fn add_cross_attention_mappings(&mut self, prefix: &str) {
        let oxigaf_prefix = prefix.replace('/', ".");

        // Q, K, V projections (no bias for linear_no_bias)
        for proj in &["to_q", "to_k", "to_v"] {
            self.add_mapping(
                &format!("{}/{}/weight", prefix, proj),
                &format!("{}.{}.weight", oxigaf_prefix, proj),
            );
        }

        // Output projection (has bias)
        self.add_mapping(
            &format!("{}/to_out/0/weight", prefix),
            &format!("{}.to_out.0.weight", oxigaf_prefix),
        );
        self.add_mapping(
            &format!("{}/to_out/0/bias", prefix),
            &format!("{}.to_out.0.bias", oxigaf_prefix),
        );
    }

    /// Add FeedForward (GeGLU) mappings.
    fn add_feed_forward_mappings(&mut self, prefix: &str) {
        let oxigaf_prefix = prefix.replace('/', ".");

        // net.0.proj (GeGLU projection)
        self.add_mapping(
            &format!("{}/net/0/proj/weight", prefix),
            &format!("{}.net.0.proj.weight", oxigaf_prefix),
        );
        self.add_mapping(
            &format!("{}/net/0/proj/bias", prefix),
            &format!("{}.net.0.proj.bias", oxigaf_prefix),
        );

        // net.2 (linear out)
        self.add_mapping(
            &format!("{}/net/2/weight", prefix),
            &format!("{}.net.2.weight", oxigaf_prefix),
        );
        self.add_mapping(
            &format!("{}/net/2/bias", prefix),
            &format!("{}.net.2.bias", oxigaf_prefix),
        );
    }

    /// Add all VAE layer mappings.
    ///
    /// Covers ~200 layers including:
    /// - Encoder with 4 down blocks
    /// - Decoder with 4 up blocks
    /// - Mid blocks with attention
    /// - Quantization layers
    fn add_vae_mappings(&mut self) {
        // Encoder
        self.add_mapping("encoder/conv_in/weight", "encoder.conv_in.weight");
        self.add_mapping("encoder/conv_in/bias", "encoder.conv_in.bias");

        // Encoder down blocks (4 stages)
        for i in 0..4 {
            self.add_vae_down_block_mappings(i);
        }

        // Encoder mid block
        self.add_vae_mid_block_mappings("encoder");

        // Encoder output
        self.add_mapping(
            "encoder/conv_norm_out/weight",
            "encoder.conv_norm_out.weight",
        );
        self.add_mapping("encoder/conv_norm_out/bias", "encoder.conv_norm_out.bias");
        self.add_mapping("encoder/conv_out/weight", "encoder.conv_out.weight");
        self.add_mapping("encoder/conv_out/bias", "encoder.conv_out.bias");

        // Quantization layers
        self.add_mapping("quant_conv/weight", "quant_conv.weight");
        self.add_mapping("quant_conv/bias", "quant_conv.bias");
        self.add_mapping("post_quant_conv/weight", "post_quant_conv.weight");
        self.add_mapping("post_quant_conv/bias", "post_quant_conv.bias");

        // Decoder
        self.add_mapping("decoder/conv_in/weight", "decoder.conv_in.weight");
        self.add_mapping("decoder/conv_in/bias", "decoder.conv_in.bias");

        // Decoder mid block
        self.add_vae_mid_block_mappings("decoder");

        // Decoder up blocks (4 stages)
        for i in 0..4 {
            self.add_vae_up_block_mappings(i);
        }

        // Decoder output
        self.add_mapping(
            "decoder/conv_norm_out/weight",
            "decoder.conv_norm_out.weight",
        );
        self.add_mapping("decoder/conv_norm_out/bias", "decoder.conv_norm_out.bias");
        self.add_mapping("decoder/conv_out/weight", "decoder.conv_out.weight");
        self.add_mapping("decoder/conv_out/bias", "decoder.conv_out.bias");
    }

    /// Add VAE encoder down block mappings.
    fn add_vae_down_block_mappings(&mut self, stage: usize) {
        // 2 ResNet blocks per stage
        for j in 0..2 {
            self.add_vae_resnet_block_mappings(&format!(
                "encoder/down_blocks/{}/resnets/{}",
                stage, j
            ));
        }

        // Downsampler (only for stages 0, 1, 2)
        if stage < 3 {
            self.add_mapping(
                &format!("encoder/down_blocks/{}/downsamplers/0/conv/weight", stage),
                &format!("encoder.down_blocks.{}.downsamplers.0.conv.weight", stage),
            );
            self.add_mapping(
                &format!("encoder/down_blocks/{}/downsamplers/0/conv/bias", stage),
                &format!("encoder.down_blocks.{}.downsamplers.0.conv.bias", stage),
            );
        }
    }

    /// Add VAE decoder up block mappings.
    fn add_vae_up_block_mappings(&mut self, stage: usize) {
        // 3 ResNet blocks per stage
        for j in 0..3 {
            self.add_vae_resnet_block_mappings(&format!(
                "decoder/up_blocks/{}/resnets/{}",
                stage, j
            ));
        }

        // Upsampler (only for stages 0, 1, 2)
        if stage < 3 {
            self.add_mapping(
                &format!("decoder/up_blocks/{}/upsamplers/0/conv/weight", stage),
                &format!("decoder.up_blocks.{}.upsamplers.0.conv.weight", stage),
            );
            self.add_mapping(
                &format!("decoder/up_blocks/{}/upsamplers/0/conv/bias", stage),
                &format!("decoder.up_blocks.{}.upsamplers.0.conv.bias", stage),
            );
        }
    }

    /// Add VAE ResNet block mappings.
    fn add_vae_resnet_block_mappings(&mut self, prefix: &str) {
        let oxigaf_prefix = prefix.replace('/', ".");

        // norm1, conv1, norm2, conv2
        for layer in &["norm1", "conv1", "norm2", "conv2"] {
            for param in &["weight", "bias"] {
                self.add_mapping(
                    &format!("{}/{}/{}", prefix, layer, param),
                    &format!("{}.{}.{}", oxigaf_prefix, layer, param),
                );
            }
        }

        // nin_shortcut (optional, for channel mismatch)
        self.add_mapping(
            &format!("{}/nin_shortcut/weight", prefix),
            &format!("{}.nin_shortcut.weight", oxigaf_prefix),
        );
        self.add_mapping(
            &format!("{}/nin_shortcut/bias", prefix),
            &format!("{}.nin_shortcut.bias", oxigaf_prefix),
        );
    }

    /// Add VAE mid block mappings (encoder or decoder).
    fn add_vae_mid_block_mappings(&mut self, parent: &str) {
        let prefix = format!("{}/mid_block", parent);
        let oxigaf_prefix = prefix.replace('/', ".");

        // ResNet block 0
        self.add_vae_resnet_block_mappings(&format!("{}/resnets/0", prefix));

        // Attention
        self.add_mapping(
            &format!("{}/attentions/0/group_norm/weight", prefix),
            &format!("{}.attentions.0.group_norm.weight", oxigaf_prefix),
        );
        self.add_mapping(
            &format!("{}/attentions/0/group_norm/bias", prefix),
            &format!("{}.attentions.0.group_norm.bias", oxigaf_prefix),
        );
        self.add_mapping(
            &format!("{}/attentions/0/to_qkv/weight", prefix),
            &format!("{}.attentions.0.to_qkv.weight", oxigaf_prefix),
        );
        self.add_mapping(
            &format!("{}/attentions/0/to_qkv/bias", prefix),
            &format!("{}.attentions.0.to_qkv.bias", oxigaf_prefix),
        );
        self.add_mapping(
            &format!("{}/attentions/0/to_out/weight", prefix),
            &format!("{}.attentions.0.to_out.weight", oxigaf_prefix),
        );
        self.add_mapping(
            &format!("{}/attentions/0/to_out/bias", prefix),
            &format!("{}.attentions.0.to_out.bias", oxigaf_prefix),
        );

        // ResNet block 1
        self.add_vae_resnet_block_mappings(&format!("{}/resnets/1", prefix));
    }

    /// Add all CLIP Image Encoder layer mappings.
    ///
    /// Covers ~300 layers including:
    /// - Patch and position embeddings
    /// - 32 ViT encoder layers
    /// - Pre/post layer norms
    /// - Optional IP projection
    fn add_clip_mappings(&mut self) {
        // Embeddings
        self.add_mapping(
            "embeddings/patch_embedding/weight",
            "embeddings.patch_embedding.weight",
        );
        self.add_mapping(
            "embeddings/patch_embedding/bias",
            "embeddings.patch_embedding.bias",
        );
        self.add_mapping(
            "embeddings/position_embedding/weight",
            "embeddings.position_embedding.weight",
        );
        self.add_mapping("embeddings/class_embedding", "embeddings.class_embedding");

        // Pre-layernorm
        self.add_mapping("pre_layrnorm/weight", "pre_layrnorm.weight");
        self.add_mapping("pre_layrnorm/bias", "pre_layrnorm.bias");

        // Encoder layers (32 layers: 0..31)
        for i in 0..32 {
            self.add_clip_encoder_layer_mappings(i);
        }

        // Post-layernorm
        self.add_mapping("post_layernorm/weight", "post_layernorm.weight");
        self.add_mapping("post_layernorm/bias", "post_layernorm.bias");

        // IP projection (optional)
        self.add_mapping("ip_projection/weight", "ip_projection.weight");
        self.add_mapping("ip_projection/bias", "ip_projection.bias");
    }

    /// Add CLIP encoder layer mappings.
    fn add_clip_encoder_layer_mappings(&mut self, layer_idx: usize) {
        let prefix = format!("encoder/layers/{}", layer_idx);
        let oxigaf_prefix = prefix.replace('/', ".");

        // LayerNorm 1
        self.add_mapping(
            &format!("{}/layer_norm1/weight", prefix),
            &format!("{}.layer_norm1.weight", oxigaf_prefix),
        );
        self.add_mapping(
            &format!("{}/layer_norm1/bias", prefix),
            &format!("{}.layer_norm1.bias", oxigaf_prefix),
        );

        // Self-attention
        for proj in &["q_proj", "k_proj", "v_proj", "out_proj"] {
            for param in &["weight", "bias"] {
                self.add_mapping(
                    &format!("{}/self_attn/{}/{}", prefix, proj, param),
                    &format!("{}.self_attn.{}.{}", oxigaf_prefix, proj, param),
                );
            }
        }

        // LayerNorm 2
        self.add_mapping(
            &format!("{}/layer_norm2/weight", prefix),
            &format!("{}.layer_norm2.weight", oxigaf_prefix),
        );
        self.add_mapping(
            &format!("{}/layer_norm2/bias", prefix),
            &format!("{}.layer_norm2.bias", oxigaf_prefix),
        );

        // MLP
        for layer in &["fc1", "fc2"] {
            for param in &["weight", "bias"] {
                self.add_mapping(
                    &format!("{}/mlp/{}/{}", prefix, layer, param),
                    &format!("{}.mlp.{}.{}", oxigaf_prefix, layer, param),
                );
            }
        }
    }

    /// Add all Latent Upsampler layer mappings.
    ///
    /// Covers ~100 layers for the sd-x2-latent-upscaler U-Net.
    fn add_upsampler_mappings(&mut self) {
        // Input convolution
        self.add_mapping("conv_in/weight", "conv_in.weight");
        self.add_mapping("conv_in/bias", "conv_in.bias");

        // Time embedding
        self.add_mapping(
            "time_embedding/linear_1/weight",
            "time_embedding.linear_1.weight",
        );
        self.add_mapping(
            "time_embedding/linear_1/bias",
            "time_embedding.linear_1.bias",
        );
        self.add_mapping(
            "time_embedding/linear_2/weight",
            "time_embedding.linear_2.weight",
        );
        self.add_mapping(
            "time_embedding/linear_2/bias",
            "time_embedding.linear_2.bias",
        );

        // Down block 0
        self.add_upsampler_down_block_mappings(0);

        // Mid block
        self.add_upsampler_mid_block_mappings();

        // Up block 0
        self.add_upsampler_up_block_mappings(0);

        // Output convolution
        self.add_mapping("conv_norm_out/weight", "conv_norm_out.weight");
        self.add_mapping("conv_norm_out/bias", "conv_norm_out.bias");
        self.add_mapping("conv_out/weight", "conv_out.weight");
        self.add_mapping("conv_out/bias", "conv_out.bias");
    }

    /// Add upsampler down block mappings.
    fn add_upsampler_down_block_mappings(&mut self, stage: usize) {
        // 2 ResNet blocks
        for j in 0..2 {
            self.add_resnet_block_mappings(&format!("down_blocks/{}/resnets/{}", stage, j));

            // Attention (only for block 0)
            if j == 0 {
                self.add_upsampler_attention_mappings(&format!(
                    "down_blocks/{}/attentions/{}",
                    stage, j
                ));
            }
        }

        // Downsampler
        self.add_mapping(
            &format!("down_blocks/{}/downsamplers/0/conv/weight", stage),
            &format!("down_blocks.{}.downsamplers.0.conv.weight", stage),
        );
        self.add_mapping(
            &format!("down_blocks/{}/downsamplers/0/conv/bias", stage),
            &format!("down_blocks.{}.downsamplers.0.conv.bias", stage),
        );
    }

    /// Add upsampler mid block mappings.
    fn add_upsampler_mid_block_mappings(&mut self) {
        // ResNet block 0
        self.add_resnet_block_mappings("mid_block/resnets/0");

        // Attention
        self.add_upsampler_attention_mappings("mid_block/attentions/0");

        // ResNet block 1
        self.add_resnet_block_mappings("mid_block/resnets/1");
    }

    /// Add upsampler up block mappings.
    fn add_upsampler_up_block_mappings(&mut self, stage: usize) {
        // Upsampler
        self.add_mapping(
            &format!("up_blocks/{}/upsamplers/0/conv/weight", stage),
            &format!("up_blocks.{}.upsamplers.0.conv.weight", stage),
        );
        self.add_mapping(
            &format!("up_blocks/{}/upsamplers/0/conv/bias", stage),
            &format!("up_blocks.{}.upsamplers.0.conv.bias", stage),
        );

        // 2 ResNet blocks
        for j in 0..2 {
            self.add_resnet_block_mappings(&format!("up_blocks/{}/resnets/{}", stage, j));

            // Attention (only for block 0)
            if j == 0 {
                self.add_upsampler_attention_mappings(&format!(
                    "up_blocks/{}/attentions/{}",
                    stage, j
                ));
            }
        }
    }

    /// Add upsampler self-attention mappings.
    fn add_upsampler_attention_mappings(&mut self, prefix: &str) {
        let oxigaf_prefix = prefix.replace('/', ".");

        self.add_mapping(
            &format!("{}/group_norm/weight", prefix),
            &format!("{}.group_norm.weight", oxigaf_prefix),
        );
        self.add_mapping(
            &format!("{}/group_norm/bias", prefix),
            &format!("{}.group_norm.bias", oxigaf_prefix),
        );
        self.add_mapping(
            &format!("{}/to_qkv/weight", prefix),
            &format!("{}.to_qkv.weight", oxigaf_prefix),
        );
        self.add_mapping(
            &format!("{}/to_qkv/bias", prefix),
            &format!("{}.to_qkv.bias", oxigaf_prefix),
        );
        self.add_mapping(
            &format!("{}/to_out/weight", prefix),
            &format!("{}.to_out.weight", oxigaf_prefix),
        );
        self.add_mapping(
            &format!("{}/to_out/bias", prefix),
            &format!("{}.to_out.bias", oxigaf_prefix),
        );
    }
}

impl Default for GafLayerMapper {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mapper_creation() {
        let mapper = GafLayerMapper::new();
        eprintln!("Total GAF layer mappings: {}", mapper.num_mappings());
        assert!(mapper.num_mappings() > 1000, "Should have ~2000 mappings");
    }

    #[test]
    fn test_time_embedding_mapping() {
        let mapper = GafLayerMapper::new();

        let oxigaf = mapper
            .map_torsh_to_oxigaf("time_embedding/linear_1/weight")
            .expect("Failed to map time embedding");
        assert_eq!(oxigaf, "time_embedding.linear_1.weight");

        let torsh = mapper
            .map_oxigaf_to_torsh("time_embedding.linear_1.weight")
            .expect("Failed to reverse map");
        assert_eq!(torsh, "time_embedding/linear_1/weight");
    }

    #[test]
    fn test_camera_embedding_mapping() {
        let mapper = GafLayerMapper::new();

        let oxigaf = mapper
            .map_torsh_to_oxigaf("camera_embedding/linear_1/weight")
            .expect("Failed to map camera embedding");
        assert_eq!(oxigaf, "camera_embedding.linear_1.weight");
    }

    #[test]
    fn test_down_block_resnet_mapping() {
        let mapper = GafLayerMapper::new();

        let oxigaf = mapper
            .map_torsh_to_oxigaf("down_blocks/0/resnets/0/norm1/weight")
            .expect("Failed to map down block");
        assert_eq!(oxigaf, "down_blocks.0.resnets.0.norm1.weight");
    }

    #[test]
    fn test_attention_self_attn_mapping() {
        let mapper = GafLayerMapper::new();

        let oxigaf = mapper
            .map_torsh_to_oxigaf(
                "down_blocks/0/attentions/0/transformer_blocks/0/attn1/to_q/weight",
            )
            .expect("Failed to map self-attention");
        assert_eq!(
            oxigaf,
            "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight"
        );
    }

    #[test]
    fn test_attention_cross_view_mapping() {
        let mapper = GafLayerMapper::new();

        let oxigaf = mapper
            .map_torsh_to_oxigaf(
                "down_blocks/0/attentions/0/transformer_blocks/0/attn_cv/to_k/weight",
            )
            .expect("Failed to map cross-view attention");
        assert_eq!(
            oxigaf,
            "down_blocks.0.attentions.0.transformer_blocks.0.attn_cv.to_k.weight"
        );
    }

    #[test]
    fn test_attention_ip_adapter_mapping() {
        let mapper = GafLayerMapper::new();

        let oxigaf = mapper
            .map_torsh_to_oxigaf(
                "down_blocks/0/attentions/0/transformer_blocks/0/attn_ip/to_v/weight",
            )
            .expect("Failed to map IP-Adapter attention");
        assert_eq!(
            oxigaf,
            "down_blocks.0.attentions.0.transformer_blocks.0.attn_ip.to_v.weight"
        );
    }

    #[test]
    fn test_feed_forward_mapping() {
        let mapper = GafLayerMapper::new();

        let oxigaf = mapper
            .map_torsh_to_oxigaf(
                "down_blocks/0/attentions/0/transformer_blocks/0/ff/net/0/proj/weight",
            )
            .expect("Failed to map feed-forward");
        assert_eq!(
            oxigaf,
            "down_blocks.0.attentions.0.transformer_blocks.0.ff.net.0.proj.weight"
        );
    }

    #[test]
    fn test_mid_block_mapping() {
        let mapper = GafLayerMapper::new();

        let oxigaf = mapper
            .map_torsh_to_oxigaf("mid_block/resnets/0/conv1/weight")
            .expect("Failed to map mid block");
        assert_eq!(oxigaf, "mid_block.resnets.0.conv1.weight");
    }

    #[test]
    fn test_up_block_mapping() {
        let mapper = GafLayerMapper::new();

        let oxigaf = mapper
            .map_torsh_to_oxigaf("up_blocks/0/resnets/0/norm2/bias")
            .expect("Failed to map up block");
        assert_eq!(oxigaf, "up_blocks.0.resnets.0.norm2.bias");
    }

    #[test]
    fn test_vae_encoder_mapping() {
        let mapper = GafLayerMapper::new();

        let oxigaf = mapper
            .map_torsh_to_oxigaf("encoder/down_blocks/0/resnets/0/conv1/weight")
            .expect("Failed to map VAE encoder");
        assert_eq!(oxigaf, "encoder.down_blocks.0.resnets.0.conv1.weight");
    }

    #[test]
    fn test_vae_decoder_mapping() {
        let mapper = GafLayerMapper::new();

        let oxigaf = mapper
            .map_torsh_to_oxigaf("decoder/up_blocks/0/resnets/0/norm1/weight")
            .expect("Failed to map VAE decoder");
        assert_eq!(oxigaf, "decoder.up_blocks.0.resnets.0.norm1.weight");
    }

    #[test]
    fn test_vae_quant_mapping() {
        let mapper = GafLayerMapper::new();

        let oxigaf = mapper
            .map_torsh_to_oxigaf("quant_conv/weight")
            .expect("Failed to map quant_conv");
        assert_eq!(oxigaf, "quant_conv.weight");

        let oxigaf = mapper
            .map_torsh_to_oxigaf("post_quant_conv/bias")
            .expect("Failed to map post_quant_conv");
        assert_eq!(oxigaf, "post_quant_conv.bias");
    }

    #[test]
    fn test_clip_embeddings_mapping() {
        let mapper = GafLayerMapper::new();

        let oxigaf = mapper
            .map_torsh_to_oxigaf("embeddings/patch_embedding/weight")
            .expect("Failed to map CLIP patch embedding");
        assert_eq!(oxigaf, "embeddings.patch_embedding.weight");

        let oxigaf = mapper
            .map_torsh_to_oxigaf("embeddings/class_embedding")
            .expect("Failed to map CLIP class embedding");
        assert_eq!(oxigaf, "embeddings.class_embedding");
    }

    #[test]
    fn test_clip_encoder_layer_mapping() {
        let mapper = GafLayerMapper::new();

        let oxigaf = mapper
            .map_torsh_to_oxigaf("encoder/layers/0/self_attn/q_proj/weight")
            .expect("Failed to map CLIP encoder layer");
        assert_eq!(oxigaf, "encoder.layers.0.self_attn.q_proj.weight");

        let oxigaf = mapper
            .map_torsh_to_oxigaf("encoder/layers/31/mlp/fc2/bias")
            .expect("Failed to map CLIP last layer");
        assert_eq!(oxigaf, "encoder.layers.31.mlp.fc2.bias");
    }

    #[test]
    fn test_clip_ip_projection_mapping() {
        let mapper = GafLayerMapper::new();

        let oxigaf = mapper
            .map_torsh_to_oxigaf("ip_projection/weight")
            .expect("Failed to map IP projection");
        assert_eq!(oxigaf, "ip_projection.weight");
    }

    #[test]
    fn test_unmapped_layer_error() {
        let mapper = GafLayerMapper::new();

        let result = mapper.map_torsh_to_oxigaf("nonexistent/layer/weight");
        assert!(result.is_err());

        if let Err(BridgeError::LayerMapping(msg)) = result {
            assert!(msg.contains("Unmapped ToRSh layer"));
            assert!(msg.contains("nonexistent/layer/weight"));
        } else {
            panic!("Expected LayerMapping error");
        }
    }

    #[test]
    fn test_bidirectional_consistency() {
        let mapper = GafLayerMapper::new();

        // Validate all mappings are bidirectional
        mapper
            .validate_bidirectional()
            .expect("Bidirectional validation failed");
    }

    #[test]
    fn test_round_trip_conversion() {
        let mapper = GafLayerMapper::new();

        let test_cases = vec![
            "time_embedding/linear_1/weight",
            "down_blocks/0/resnets/0/norm1/weight",
            "down_blocks/0/attentions/0/transformer_blocks/0/attn1/to_q/weight",
            "mid_block/resnets/0/conv1/weight",
            "up_blocks/0/resnets/0/norm2/bias",
            "encoder/down_blocks/0/resnets/0/conv1/weight",
            "decoder/up_blocks/0/resnets/0/norm1/weight",
            "encoder/layers/0/self_attn/q_proj/weight",
        ];

        for torsh_name in test_cases {
            let oxigaf_name = mapper
                .map_torsh_to_oxigaf(torsh_name)
                .unwrap_or_else(|_| panic!("Failed to map: {}", torsh_name));
            let round_trip = mapper
                .map_oxigaf_to_torsh(&oxigaf_name)
                .unwrap_or_else(|_| panic!("Failed to reverse map: {}", oxigaf_name));
            assert_eq!(
                round_trip, torsh_name,
                "Round-trip failed for {}",
                torsh_name
            );
        }
    }

    #[test]
    fn test_validate_coverage() {
        let mapper = GafLayerMapper::new();

        // Valid keys
        let valid_keys = vec![
            "time_embedding/linear_1/weight".to_string(),
            "conv_in/weight".to_string(),
        ];
        assert!(mapper.validate_coverage(&valid_keys).is_ok());

        // Invalid keys
        let invalid_keys = vec![
            "time_embedding/linear_1/weight".to_string(),
            "nonexistent/layer".to_string(),
        ];
        assert!(mapper.validate_coverage(&invalid_keys).is_err());
    }

    #[test]
    fn test_conv_shortcut_optional_mapping() {
        let mapper = GafLayerMapper::new();

        // conv_shortcut should be mapped (it's optional in the model but we map it)
        let oxigaf = mapper
            .map_torsh_to_oxigaf("down_blocks/0/resnets/0/conv_shortcut/weight")
            .expect("Failed to map conv_shortcut");
        assert_eq!(oxigaf, "down_blocks.0.resnets.0.conv_shortcut.weight");
    }
}
