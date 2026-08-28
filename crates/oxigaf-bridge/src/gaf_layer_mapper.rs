//! Layer name mapping for GAF (Generative Avatar Face) models.
//!
//! This module provides bidirectional mapping between ToRSh (safetensors with "/" separators)
//! and OxiGAF (Candle VarBuilder with "." separators) naming conventions for all GAF model
//! components: the Multi-View U-Net, VAE, CLIP Image Encoder, and Latent Upsampler.
//!
//! # Design
//!
//! Every GAF layer name pair used by this bridge is related by a purely
//! mechanical substitution: `/` in the ToRSh (safetensors) name becomes `.`
//! in the OxiGAF (Candle `VarBuilder`) name, and vice versa. There is no GAF
//! layer whose canonical name differs between the two conventions beyond the
//! separator character.
//!
//! An earlier version of this mapper enumerated every one of the ~2000
//! concrete layer paths up front (one U-Net/VAE/CLIP/Upsampler topology
//! walk building two `HashMap`s), which had three problems:
//!
//! 1. **It duplicated model topology that lives elsewhere.** The U-Net walk
//!    hardcoded `transformer_layers_per_block = [1, 2, 10, 10]`, which does
//!    not match `DiffusionConfig::default()`'s `[1, 1, 1, 1]`
//!    (`oxigaf-diffusion/src/config.rs`) -- the very topology this crate
//!    exists to bridge weights for. Any model built from the default config
//!    produced false negatives (missing transformer blocks) and false
//!    positives (blocks 1..9 that the config doesn't have) when checked
//!    against the table.
//! 2. **It flattened four independent components into one namespace.**
//!    U-Net, VAE, CLIP and Upsampler mappings were inserted into the same
//!    two `HashMap`s, so components that happen to share a prefix (e.g. the
//!    VAE's `encoder/` and the CLIP encoder's `encoder/`) could overwrite
//!    each other's entries.
//! 3. **It was expensive for no benefit.** Every caller
//!    (`torsh_to_oxigaf::convert`, `oxigaf_to_torsh::convert`) already falls
//!    back to the identical mechanical substitution whenever a name misses
//!    the table -- so the ~4000 `String` allocations and two ~2000-entry
//!    `HashMap`s built fresh on every `GafLayerMapper::new()` call were
//!    reproducing, at a cost, exactly the behavior the fallback path
//!    provides for free.
//!
//! This version performs the substitution directly and keeps only a small,
//! explicit override table for names that are genuine exceptions to the
//! rule. There are none today, but [`GafLayerMapper::add_override`] exists
//! so a future GAF component with a real naming exception can be
//! special-cased without reintroducing the enumeration.
//!
//! Note that this mapper only translates *names* -- it does not know
//! whether a given name corresponds to a real tensor in any particular
//! checkpoint. Content validation (missing/extra layers, shape checks,
//! NaN/Inf) is [`crate::validation::validate_converted_checkpoint`]'s job.
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
//! // Validate that names are well-formed (non-empty)
//! let keys = vec!["time_embedding/linear_1/weight".to_string()];
//! mapper.validate_coverage(&keys)?;
//! # Ok::<(), oxigaf_bridge::BridgeError>(())
//! ```

use crate::error::{BridgeError, Result};
use std::collections::HashMap;

/// Layer name mapper for GAF models.
///
/// Converts between ToRSh (safetensors, `/`-separated) and OxiGAF (Candle
/// `VarBuilder`, `.`-separated) naming conventions via direct character
/// substitution, with a small table of explicit overrides for names that
/// are not a mechanical substitution of each other. See the [module-level
/// documentation](self) for why this replaced a fully-enumerated table.
#[derive(Debug, Clone)]
pub struct GafLayerMapper {
    /// Explicit overrides: ToRSh name -> OxiGAF name. Checked before falling
    /// back to the mechanical `/` -> `.` rule.
    torsh_to_oxigaf_overrides: HashMap<String, String>,
    /// Reverse of `torsh_to_oxigaf_overrides`, kept in sync by
    /// [`GafLayerMapper::add_override`].
    oxigaf_to_torsh_overrides: HashMap<String, String>,
}

impl GafLayerMapper {
    /// Create a new GAF layer mapper with no explicit overrides.
    ///
    /// Every ToRSh/OxiGAF name pair is handled by the mechanical `/` <-> `.`
    /// substitution rule unless [`GafLayerMapper::add_override`] registers
    /// an exception for it.
    pub fn new() -> Self {
        Self {
            torsh_to_oxigaf_overrides: HashMap::new(),
            oxigaf_to_torsh_overrides: HashMap::new(),
        }
    }

    /// Register an explicit override for a ToRSh/OxiGAF name pair that is
    /// *not* a plain `/` <-> `.` substitution of each other.
    ///
    /// Overrides are checked before the mechanical rule in both mapping
    /// directions, and count toward [`GafLayerMapper::num_mappings`].
    pub fn add_override(&mut self, torsh_name: &str, oxigaf_name: &str) {
        self.torsh_to_oxigaf_overrides
            .insert(torsh_name.to_string(), oxigaf_name.to_string());
        self.oxigaf_to_torsh_overrides
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
    /// Returns `BridgeError::LayerMapping` if `torsh_name` is empty -- there
    /// is nothing meaningful to map. Every other input is handled by an
    /// override or the mechanical substitution rule and always succeeds.
    pub fn map_torsh_to_oxigaf(&self, torsh_name: &str) -> Result<String> {
        if let Some(mapped) = self.torsh_to_oxigaf_overrides.get(torsh_name) {
            return Ok(mapped.clone());
        }
        if torsh_name.is_empty() {
            return Err(BridgeError::LayerMapping(
                "Cannot map an empty ToRSh layer name".to_string(),
            ));
        }
        Ok(torsh_name.replace('/', "."))
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
    /// Returns `BridgeError::LayerMapping` if `oxigaf_name` is empty -- there
    /// is nothing meaningful to map. Every other input is handled by an
    /// override or the mechanical substitution rule and always succeeds.
    pub fn map_oxigaf_to_torsh(&self, oxigaf_name: &str) -> Result<String> {
        if let Some(mapped) = self.oxigaf_to_torsh_overrides.get(oxigaf_name) {
            return Ok(mapped.clone());
        }
        if oxigaf_name.is_empty() {
            return Err(BridgeError::LayerMapping(
                "Cannot map an empty OxiGAF layer name".to_string(),
            ));
        }
        Ok(oxigaf_name.replace('.', "/"))
    }

    /// Validate that all provided ToRSh keys are well-formed (non-empty, and
    /// so mappable).
    ///
    /// This is a name-syntax check, not a model-topology check: since every
    /// non-empty name maps successfully (see [`GafLayerMapper::map_torsh_to_oxigaf`]),
    /// this cannot detect a name that is merely absent from a particular
    /// checkpoint or model configuration. For that, validate the actual
    /// checkpoint contents with
    /// [`crate::validation::validate_converted_checkpoint`] instead.
    ///
    /// # Errors
    ///
    /// Returns `BridgeError::LayerMapping` with the list of unmappable keys
    /// (currently: empty strings) if any are found.
    pub fn validate_coverage(&self, keys: &[String]) -> Result<()> {
        let unmapped: Vec<&String> = keys
            .iter()
            .filter(|k| self.map_torsh_to_oxigaf(k).is_err())
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

    /// Validate internal consistency of the registered overrides.
    ///
    /// The mechanical substitution rule is bidirectionally consistent by
    /// construction for any name that doesn't itself mix `/` and `.`; this
    /// checks the part that *isn't* provable by construction -- that every
    /// override registered via [`GafLayerMapper::add_override`] has a
    /// matching reverse entry pointing back to it.
    ///
    /// # Errors
    ///
    /// Returns `BridgeError::LayerMapping` if any inconsistency is detected.
    pub fn validate_bidirectional(&self) -> Result<()> {
        for (torsh_name, oxigaf_name) in &self.torsh_to_oxigaf_overrides {
            let reverse = self
                .oxigaf_to_torsh_overrides
                .get(oxigaf_name)
                .ok_or_else(|| {
                    BridgeError::LayerMapping(format!(
                        "Bidirectional inconsistency: ToRSh '{}' overrides to OxiGAF '{}', \
                     but reverse mapping is missing",
                        torsh_name, oxigaf_name
                    ))
                })?;

            if reverse != torsh_name {
                return Err(BridgeError::LayerMapping(format!(
                    "Bidirectional inconsistency: ToRSh '{}' overrides to OxiGAF '{}', \
                     which maps back to ToRSh '{}' (expected '{}')",
                    torsh_name, oxigaf_name, reverse, torsh_name
                )));
            }
        }

        if self.torsh_to_oxigaf_overrides.len() != self.oxigaf_to_torsh_overrides.len() {
            return Err(BridgeError::LayerMapping(format!(
                "Bidirectional count mismatch: {} ToRSh→OxiGAF overrides but {} OxiGAF→ToRSh overrides",
                self.torsh_to_oxigaf_overrides.len(),
                self.oxigaf_to_torsh_overrides.len()
            )));
        }

        Ok(())
    }

    /// Number of explicit name overrides registered via
    /// [`GafLayerMapper::add_override`].
    ///
    /// This does *not* count names handled by the mechanical substitution
    /// rule (i.e. almost all of them) -- there is no enumerable "total
    /// number of GAF layers" for this mapper to report, since it does not
    /// enumerate layers. A freshly-created mapper reports `0`.
    pub fn num_mappings(&self) -> usize {
        self.torsh_to_oxigaf_overrides.len()
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
        assert_eq!(
            mapper.num_mappings(),
            0,
            "a fresh mapper has no explicit overrides"
        );
    }

    #[test]
    fn test_empty_name_is_rejected() {
        let mapper = GafLayerMapper::new();
        assert!(mapper.map_torsh_to_oxigaf("").is_err());
        assert!(mapper.map_oxigaf_to_torsh("").is_err());
    }

    #[test]
    fn test_override_takes_precedence_over_mechanical_rule() {
        let mut mapper = GafLayerMapper::new();
        mapper.add_override("legacy/name", "modern.name");
        assert_eq!(mapper.num_mappings(), 1);

        assert_eq!(
            mapper.map_torsh_to_oxigaf("legacy/name").expect("mapped"),
            "modern.name"
        );
        assert_eq!(
            mapper.map_oxigaf_to_torsh("modern.name").expect("mapped"),
            "legacy/name"
        );

        // Names without an override still go through the mechanical rule.
        assert_eq!(
            mapper.map_torsh_to_oxigaf("other/name").expect("mapped"),
            "other.name"
        );
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
    fn test_bidirectional_consistency() {
        let mapper = GafLayerMapper::new();

        // Validate all registered overrides are bidirectional (there are
        // none by default, so this is trivially satisfied).
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

        // Valid (non-empty) keys
        let valid_keys = vec![
            "time_embedding/linear_1/weight".to_string(),
            "conv_in/weight".to_string(),
        ];
        assert!(mapper.validate_coverage(&valid_keys).is_ok());

        // An empty name is the one input that cannot be mapped.
        let invalid_keys = vec!["time_embedding/linear_1/weight".to_string(), String::new()];
        assert!(mapper.validate_coverage(&invalid_keys).is_err());
    }

    #[test]
    fn test_conv_shortcut_optional_mapping() {
        let mapper = GafLayerMapper::new();

        let oxigaf = mapper
            .map_torsh_to_oxigaf("down_blocks/0/resnets/0/conv_shortcut/weight")
            .expect("Failed to map conv_shortcut");
        assert_eq!(oxigaf, "down_blocks.0.resnets.0.conv_shortcut.weight");
    }

    #[test]
    fn test_handles_arbitrary_topology_without_hardcoded_depths() {
        // Regression test: the previous table-based mapper hardcoded
        // `transformer_layers_per_block = [1, 2, 10, 10]`, which did not
        // match `DiffusionConfig::default()`'s `[1, 1, 1, 1]`
        // (oxigaf-diffusion/src/config.rs), causing false negatives/positives
        // in coverage checking for any model built from the default config.
        // A mechanical mapper has no topology to fall out of sync with: any
        // transformer-block index maps correctly, regardless of how many
        // blocks a given model configuration actually has.
        let mapper = GafLayerMapper::new();
        for depth in [0usize, 1, 2, 3, 9, 10, 41] {
            let torsh = format!(
                "down_blocks/0/attentions/0/transformer_blocks/{}/attn1/to_q/weight",
                depth
            );
            let expected = format!(
                "down_blocks.0.attentions.0.transformer_blocks.{}.attn1.to_q.weight",
                depth
            );
            assert_eq!(
                mapper.map_torsh_to_oxigaf(&torsh).expect("mapped"),
                expected
            );
        }
    }

    #[test]
    fn test_no_cross_component_namespace_collision() {
        // Regression test: the previous table-based mapper flattened U-Net,
        // VAE, CLIP and Upsampler mappings into two shared `HashMap`s keyed
        // only by name, so components that happen to share a prefix (e.g.
        // the VAE encoder's and the CLIP encoder's `encoder/`) could
        // silently overwrite each other's entries. A mechanical mapper has
        // no shared table to collide in -- every name maps independently.
        let mapper = GafLayerMapper::new();
        assert_eq!(
            mapper
                .map_torsh_to_oxigaf("encoder/conv_in/weight")
                .expect("mapped"),
            "encoder.conv_in.weight"
        );
        assert_eq!(
            mapper
                .map_torsh_to_oxigaf("encoder/layers/0/self_attn/q_proj/weight")
                .expect("mapped"),
            "encoder.layers.0.self_attn.q_proj.weight"
        );
    }
}
