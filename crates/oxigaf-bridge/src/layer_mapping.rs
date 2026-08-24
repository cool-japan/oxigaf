//! Layer name mapping between different frameworks
//!
//! This module handles conversion of layer names between PyTorch, OxiGAF, and ToRSh conventions.

use crate::error::Result;
use std::collections::HashMap;

/// Naming convention for different frameworks
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NamingConvention {
    /// PyTorch: `unet.down_blocks.0.resnets.0.conv1.weight`
    PyTorch,
    /// OxiGAF: `down__blocks_0_resnets_0_conv1_weight` (existing underscores
    /// in the source name are doubled before dots are converted to single
    /// underscores, so the two are distinguishable on the way back)
    OxiGAF,
    /// ToRSh: `down_blocks/0/resnets/0/conv1/weight`
    ToRSh,
}

/// Prefixes recognized and stripped by [`LayerMapping::pytorch_to_oxigaf`].
const KNOWN_PYTORCH_PREFIXES: &[&str] = &["unet", "model", "module"];

/// safetensors `__metadata__` key under which `pytorch_to_oxigaf::convert`
/// records, as a JSON object, which recognized PyTorch prefix (if any) was
/// stripped from each converted tensor -- keyed by the resulting OxiGAF
/// name. `oxigaf_to_pytorch::convert` reads this back so it can restore each
/// tensor's exact original prefix instead of assuming a single prefix (e.g.
/// `"unet"`) for the whole checkpoint.
pub(crate) const PREFIX_METADATA_KEY: &str = "oxigaf_bridge:prefixes";

/// Detects a recognized prefix component of `pytorch_name`, returning the
/// prefix and the remainder of the name after the separating dot.
///
/// This performs the same recognition [`LayerMapping::pytorch_to_oxigaf`]
/// uses internally to decide what to strip. Callers that need to know
/// *which* prefix (if any) was stripped -- e.g. to persist it so a later
/// reverse conversion can restore the exact original name -- can call this
/// directly instead of re-deriving it from ad hoc string manipulation.
pub fn detect_prefix(pytorch_name: &str) -> Option<(&'static str, &str)> {
    for &prefix in KNOWN_PYTORCH_PREFIXES {
        if let Some(rest) = pytorch_name.strip_prefix(prefix) {
            if let Some(remainder) = rest.strip_prefix('.') {
                return Some((prefix, remainder));
            }
        }
    }
    None
}

/// Layer name mapping handler
pub struct LayerMapping {
    custom_mappings: HashMap<String, String>,
    /// Reverse of `custom_mappings` (`to` -> `from`), kept in sync by
    /// [`LayerMapping::add_custom_mapping`] so [`LayerMapping::oxigaf_to_pytorch`]
    /// can look up a match in O(1) instead of scanning `custom_mappings`. If
    /// multiple `from` names are registered with the same `to` value, the
    /// most recently added mapping wins the reverse lookup.
    custom_mappings_reverse: HashMap<String, String>,
}

impl LayerMapping {
    /// Create a new layer mapping handler
    pub fn new() -> Self {
        Self {
            custom_mappings: HashMap::new(),
            custom_mappings_reverse: HashMap::new(),
        }
    }

    /// Add a custom mapping from one name to another
    pub fn add_custom_mapping(&mut self, from: String, to: String) {
        self.custom_mappings_reverse
            .insert(to.clone(), from.clone());
        self.custom_mappings.insert(from, to);
    }

    /// Look up a custom mapping by its exact `from` key, if one was
    /// registered via [`LayerMapping::add_custom_mapping`]. Bypasses
    /// prefix-stripping and underscore/dot escaping entirely -- the
    /// registered `to` value is returned verbatim.
    ///
    /// This exists for conversion directions (e.g. ToRSh, whose OxiGAF-side
    /// names use dots as a hierarchical path separator, not the escaped
    /// underscore convention `pytorch_to_oxigaf`/`oxigaf_to_pytorch` use) to
    /// still honor caller-supplied overrides without routing every name
    /// through the PyTorch-specific escaping scheme.
    pub fn lookup_custom(&self, from: &str) -> Option<&str> {
        self.custom_mappings.get(from).map(String::as_str)
    }

    /// Convert PyTorch name to OxiGAF name
    ///
    /// Example: `unet.down_blocks.0.resnets.0.conv1.weight` → `down__blocks_0_resnets_0_conv1_weight`
    ///
    /// Existing underscores in the source name are doubled *before* dots are
    /// replaced with single underscores, so `down_blocks` (one underscore)
    /// and the dot-to-underscore conversion don't collide: only a single
    /// underscore in the OxiGAF name ever originated from a dot.
    pub fn pytorch_to_oxigaf(&self, pytorch_name: &str) -> Result<String> {
        // Check custom mappings first
        if let Some(mapped) = self.custom_mappings.get(pytorch_name) {
            return Ok(mapped.clone());
        }

        // Remove a common prefix like "unet.", "model.", "module.", if present.
        let name = detect_prefix(pytorch_name)
            .map(|(_, remainder)| remainder)
            .unwrap_or(pytorch_name);

        // Escape existing underscores (double them) so we can distinguish them from converted dots
        let escaped = name.replace('_', "__");
        // Replace dots with single underscores
        let oxigaf_name = escaped.replace('.', "_");

        Ok(oxigaf_name)
    }

    /// Convert OxiGAF name to PyTorch name
    ///
    /// Example: `down__blocks_0_resnets_0_conv1_weight` → `unet.down_blocks.0.resnets.0.conv1.weight`
    ///
    /// Note: the underscore-escaping `pytorch_to_oxigaf` applies is not
    /// injective -- `a._b` and `a_.b` both encode to `a___b` -- so a PyTorch
    /// name with a leading-underscore path component (e.g. `torch.compile`'s
    /// `_orig_mod.` wrapper) is not guaranteed to round-trip exactly.
    pub fn oxigaf_to_pytorch(&self, oxigaf_name: &str, prefix: Option<&str>) -> Result<String> {
        // Reverse lookup in custom mappings (O(1) via the maintained reverse map).
        // The prefix is applied here too, same as the non-custom path below, so
        // custom and mechanical mappings produce a consistent output namespace.
        if let Some(pytorch_name) = self.custom_mappings_reverse.get(oxigaf_name) {
            return Ok(match prefix {
                Some(prefix) => format!("{}.{}", prefix, pytorch_name),
                None => pytorch_name.clone(),
            });
        }

        // Replace single underscores with dots (these were from PyTorch dots)
        // Replace double underscores with single underscores (these were original underscores)
        let pytorch_name = oxigaf_name
            .replace("__", "\x00")
            .replace('_', ".")
            .replace('\x00', "_");

        // Add prefix if provided
        if let Some(prefix) = prefix {
            Ok(format!("{}.{}", prefix, pytorch_name))
        } else {
            Ok(pytorch_name)
        }
    }

    /// Convert PyTorch name to ToRSh name
    ///
    /// Example: `unet.down_blocks.0.resnets.0.conv1.weight` → `down_blocks/0/resnets/0/conv1/weight`
    pub fn pytorch_to_torsh(&self, pytorch_name: &str) -> Result<String> {
        // Remove common prefixes
        let name = pytorch_name
            .strip_prefix("unet.")
            .or_else(|| pytorch_name.strip_prefix("model."))
            .or_else(|| pytorch_name.strip_prefix("module."))
            .unwrap_or(pytorch_name);

        // Replace dots with slashes
        let torsh_name = name.replace('.', "/");

        Ok(torsh_name)
    }

    /// Convert ToRSh name to PyTorch name
    ///
    /// Example: `down_blocks/0/resnets/0/conv1/weight` → `unet.down_blocks.0.resnets.0.conv1.weight`
    pub fn torsh_to_pytorch(&self, torsh_name: &str, prefix: Option<&str>) -> Result<String> {
        // Replace slashes with dots
        let mut pytorch_name = torsh_name.replace('/', ".");

        // Add prefix if provided
        if let Some(prefix) = prefix {
            pytorch_name = format!("{}.{}", prefix, pytorch_name);
        }

        Ok(pytorch_name)
    }

    /// Convert OxiGAF name to ToRSh name
    pub fn oxigaf_to_torsh(&self, oxigaf_name: &str) -> Result<String> {
        // First convert to PyTorch, then to ToRSh
        let pytorch_name = self.oxigaf_to_pytorch(oxigaf_name, None)?;
        self.pytorch_to_torsh(&pytorch_name)
    }

    /// Convert ToRSh name to OxiGAF name
    pub fn torsh_to_oxigaf(&self, torsh_name: &str) -> Result<String> {
        // First convert to PyTorch, then to OxiGAF
        let pytorch_name = self.torsh_to_pytorch(torsh_name, None)?;
        self.pytorch_to_oxigaf(&pytorch_name)
    }
}

impl Default for LayerMapping {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pytorch_to_oxigaf() {
        let mapping = LayerMapping::new();

        let test_cases = vec![
            (
                "unet.down_blocks.0.resnets.0.conv1.weight",
                "down__blocks_0_resnets_0_conv1_weight",
            ),
            (
                "model.encoder.layer.2.attention.self.query.bias",
                "encoder_layer_2_attention_self_query_bias",
            ),
            ("simple.layer.weight", "simple_layer_weight"),
        ];

        for (pytorch, expected_oxigaf) in test_cases {
            let result = mapping
                .pytorch_to_oxigaf(pytorch)
                .expect("test: layer mapping should succeed");
            assert_eq!(result, expected_oxigaf, "Failed for: {}", pytorch);
        }
    }

    #[test]
    fn test_oxigaf_to_pytorch() {
        let mapping = LayerMapping::new();

        let test_cases = vec![
            (
                "down__blocks_0_resnets_0_conv1_weight",
                "down_blocks.0.resnets.0.conv1.weight",
            ),
            (
                "encoder_layer_2_attention_self_query_bias",
                "encoder.layer.2.attention.self.query.bias",
            ),
        ];

        for (oxigaf, expected_pytorch) in test_cases {
            let result = mapping
                .oxigaf_to_pytorch(oxigaf, None)
                .expect("test: layer mapping should succeed");
            assert_eq!(result, expected_pytorch, "Failed for: {}", oxigaf);
        }
    }

    #[test]
    fn test_pytorch_to_torsh() {
        let mapping = LayerMapping::new();

        let test_cases = vec![
            (
                "unet.down_blocks.0.resnets.0.conv1.weight",
                "down_blocks/0/resnets/0/conv1/weight",
            ),
            (
                "model.encoder.layer.2.attention.self.query.bias",
                "encoder/layer/2/attention/self/query/bias",
            ),
        ];

        for (pytorch, expected_torsh) in test_cases {
            let result = mapping
                .pytorch_to_torsh(pytorch)
                .expect("test: layer mapping should succeed");
            assert_eq!(result, expected_torsh, "Failed for: {}", pytorch);
        }
    }

    #[test]
    fn test_torsh_to_pytorch() {
        let mapping = LayerMapping::new();

        let test_cases = vec![
            (
                "down_blocks/0/resnets/0/conv1/weight",
                "down_blocks.0.resnets.0.conv1.weight",
            ),
            (
                "encoder/layer/2/attention/self/query/bias",
                "encoder.layer.2.attention.self.query.bias",
            ),
        ];

        for (torsh, expected_pytorch) in test_cases {
            let result = mapping
                .torsh_to_pytorch(torsh, None)
                .expect("test: layer mapping should succeed");
            assert_eq!(result, expected_pytorch, "Failed for: {}", torsh);
        }
    }

    #[test]
    fn test_custom_mapping() {
        let mut mapping = LayerMapping::new();
        mapping.add_custom_mapping(
            "custom.input".to_string(),
            "custom_special_input".to_string(),
        );

        let result = mapping
            .pytorch_to_oxigaf("custom.input")
            .expect("test: layer mapping should succeed");
        assert_eq!(result, "custom_special_input");
    }

    #[test]
    fn test_custom_mapping_reverse_lookup_applies_prefix() {
        // Regression test: the reverse (OxiGAF -> PyTorch) lookup for a
        // custom mapping used to bypass the `prefix` argument entirely,
        // producing an inconsistent output namespace versus every
        // mechanically-mapped name (which does get the prefix). It must now
        // apply the prefix the same way the non-custom path does.
        let mut mapping = LayerMapping::new();
        mapping.add_custom_mapping(
            "custom.input".to_string(),
            "custom_special_input".to_string(),
        );

        let with_prefix = mapping
            .oxigaf_to_pytorch("custom_special_input", Some("unet"))
            .expect("test: layer mapping should succeed");
        assert_eq!(with_prefix, "unet.custom.input");

        let without_prefix = mapping
            .oxigaf_to_pytorch("custom_special_input", None)
            .expect("test: layer mapping should succeed");
        assert_eq!(without_prefix, "custom.input");
    }

    #[test]
    fn test_lookup_custom() {
        let mut mapping = LayerMapping::new();
        assert_eq!(
            mapping.lookup_custom("down_blocks.0.resnets.0.weight"),
            None
        );

        mapping.add_custom_mapping(
            "down_blocks.0.resnets.0.weight".to_string(),
            "special_name".to_string(),
        );
        assert_eq!(
            mapping.lookup_custom("down_blocks.0.resnets.0.weight"),
            Some("special_name")
        );
        // Exact-match only: no prefix stripping or escaping is applied.
        assert_eq!(
            mapping.lookup_custom("unet.down_blocks.0.resnets.0.weight"),
            None
        );
    }

    #[test]
    fn test_detect_prefix() {
        assert_eq!(
            detect_prefix("unet.down_blocks.0.conv.weight"),
            Some(("unet", "down_blocks.0.conv.weight"))
        );
        assert_eq!(
            detect_prefix("model.layer.weight"),
            Some(("model", "layer.weight"))
        );
        assert_eq!(detect_prefix("module.foo"), Some(("module", "foo")));
        assert_eq!(detect_prefix("no_prefix.here"), None);
        // A name that merely starts with the same letters as a known prefix,
        // but without the separating dot, must not match.
        assert_eq!(detect_prefix("unethical.weight"), None);
    }

    #[test]
    fn test_round_trip_pytorch_oxigaf() {
        let mapping = LayerMapping::new();

        let test_cases = vec![
            ("unet.down_blocks.0.conv.weight", "unet"),
            ("model.layer.1.norm.bias", "model"),
        ];

        for (pytorch_name, prefix) in test_cases {
            let oxigaf_name = mapping
                .pytorch_to_oxigaf(pytorch_name)
                .expect("test: layer mapping should succeed");
            let reconstructed = mapping
                .oxigaf_to_pytorch(&oxigaf_name, Some(prefix))
                .expect("test: layer mapping should succeed");

            // The reconstruction should match the original
            assert_eq!(
                reconstructed, pytorch_name,
                "Round-trip failed for {}: got {}",
                pytorch_name, reconstructed
            );
        }
    }
}
