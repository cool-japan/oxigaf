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
    /// OxiGAF: `down_blocks_0_resnets_0_conv1_weight`
    OxiGAF,
    /// ToRSh: `down_blocks/0/resnets/0/conv1/weight`
    ToRSh,
}

/// Layer name mapping handler
pub struct LayerMapping {
    custom_mappings: HashMap<String, String>,
}

impl LayerMapping {
    /// Create a new layer mapping handler
    pub fn new() -> Self {
        Self {
            custom_mappings: HashMap::new(),
        }
    }

    /// Add a custom mapping from one name to another
    pub fn add_custom_mapping(&mut self, from: String, to: String) {
        self.custom_mappings.insert(from, to);
    }

    /// Convert PyTorch name to OxiGAF name
    ///
    /// Example: `unet.down_blocks.0.resnets.0.conv1.weight` → `down_blocks_0_resnets_0_conv1_weight`
    pub fn pytorch_to_oxigaf(&self, pytorch_name: &str) -> Result<String> {
        // Check custom mappings first
        if let Some(mapped) = self.custom_mappings.get(pytorch_name) {
            return Ok(mapped.clone());
        }

        // Remove common prefixes like "unet.", "model.", etc.
        let name = pytorch_name
            .strip_prefix("unet.")
            .or_else(|| pytorch_name.strip_prefix("model."))
            .or_else(|| pytorch_name.strip_prefix("module."))
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
    pub fn oxigaf_to_pytorch(&self, oxigaf_name: &str, prefix: Option<&str>) -> Result<String> {
        // Reverse lookup in custom mappings
        if let Some((pytorch_name, _)) = self
            .custom_mappings
            .iter()
            .find(|(_, v)| v.as_str() == oxigaf_name)
        {
            return Ok(pytorch_name.clone());
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
