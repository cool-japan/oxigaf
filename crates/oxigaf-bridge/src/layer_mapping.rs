//! Layer name mapping between different frameworks
//!
//! This module handles conversion of layer names between PyTorch, OxiGAF, and ToRSh conventions.

use crate::error::{BridgeError, Result};
use std::collections::HashMap;

/// Naming convention for different frameworks
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NamingConvention {
    /// PyTorch: `unet.down_blocks.0.resnets.0.conv1.weight`
    PyTorch,
    /// OxiGAF: `down_blocks.0.resnets.0.conv1.weight` -- the model-rooted,
    /// dot-separated path `candle_nn::VarBuilder::pp` walks. This is the
    /// *same* convention [`crate::gaf_layer_mapper::GafLayerMapper`]
    /// produces from ToRSh names; the two bridges agree.
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

    /// Add a custom mapping from one name to another, replacing any mapping
    /// already registered for `from`.
    ///
    /// Both directions stay consistent: re-registering `from` with a new
    /// `to` also drops the *previous* `to`'s reverse entry, so
    /// [`LayerMapping::oxigaf_to_pytorch`] can never resolve a superseded
    /// target back to `from`. Registering two different `from` names with
    /// the same `to` is last-insert-wins for the reverse direction (the
    /// forward direction keeps both) and logs a warning, because that
    /// collision makes the reverse mapping lossy; use
    /// [`LayerMapping::add_custom_mapping_checked`] to reject it outright
    /// instead.
    pub fn add_custom_mapping(&mut self, from: String, to: String) {
        // Re-registering `from` with a different `to` used to leave the old
        // `to` behind in the reverse map, so a name the forward mapping no
        // longer produces still resolved backwards to `from`.
        if let Some(previous_to) = self.custom_mappings.get(&from) {
            if previous_to != &to {
                let previous_to = previous_to.clone();
                self.custom_mappings_reverse.remove(&previous_to);
            }
        }
        if let Some(existing_from) = self.custom_mappings_reverse.get(&to) {
            if existing_from != &from {
                tracing::warn!(
                    "Custom mapping target '{}' is already registered for '{}'; \
                     '{}' now wins the reverse (OxiGAF -> PyTorch) lookup",
                    to,
                    existing_from,
                    from
                );
            }
        }
        self.custom_mappings_reverse
            .insert(to.clone(), from.clone());
        self.custom_mappings.insert(from, to);
    }

    /// Like [`LayerMapping::add_custom_mapping`], but refuses a `to` value
    /// that is already the target of a *different* `from`, instead of
    /// silently letting the newer mapping win the reverse lookup.
    ///
    /// Re-registering the same `from` (whatever its previous target) is
    /// still allowed and is not a collision -- only two distinct sources
    /// aiming at one target are, since that is what makes the OxiGAF ->
    /// PyTorch direction lossy.
    ///
    /// # Errors
    ///
    /// Returns [`BridgeError::LayerMapping`] if `to` is already registered
    /// as the target of a different `from`. The mapping is left unchanged.
    pub fn add_custom_mapping_checked(&mut self, from: String, to: String) -> Result<()> {
        if let Some(existing_from) = self.custom_mappings_reverse.get(&to) {
            if existing_from != &from {
                return Err(BridgeError::LayerMapping(format!(
                    "Custom mapping collision: '{}' and '{}' both map to '{}'",
                    existing_from, from, to
                )));
            }
        }
        self.add_custom_mapping(from, to);
        Ok(())
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
    /// Example: `unet.down_blocks.0.resnets.0.conv1.weight` →
    /// `down_blocks.0.resnets.0.conv1.weight`
    ///
    /// The only transformation is stripping a recognized top-level prefix
    /// (see [`detect_prefix`]); the dot-separated path itself is preserved
    /// verbatim, because that is exactly the form
    /// `candle_nn::VarBuilder::pp` walks and the form
    /// [`crate::gaf_layer_mapper::GafLayerMapper`] produces from ToRSh
    /// names.
    ///
    /// This *is* injective given the stripped prefix: the transform is the
    /// identity on the remainder, so
    /// `oxigaf_to_pytorch(pytorch_to_oxigaf(n), stripped_prefix) == n` for
    /// every `n`, including names containing underscores, leading-underscore
    /// path components (`_orig_mod.` from `torch.compile`), and consecutive
    /// dots. `pytorch_to_oxigaf::convert` records the stripped prefix per
    /// tensor in the output file's `__metadata__` so the reverse direction
    /// can supply it.
    ///
    /// # Prior format
    ///
    /// Before 0.1.2 this produced a flat, double-underscore-escaped name
    /// (`down__blocks_0_resnets_0_conv1_weight`). That form was not
    /// `VarBuilder`-loadable and its escaping was not injective (`a._b` and
    /// `a_.b` both encoded to `a___b`), so checkpoints written by an older
    /// version of this crate must be re-converted from their PyTorch source.
    pub fn pytorch_to_oxigaf(&self, pytorch_name: &str) -> Result<String> {
        // Check custom mappings first
        if let Some(mapped) = self.custom_mappings.get(pytorch_name) {
            return Ok(mapped.clone());
        }

        // Remove a common prefix like "unet.", "model.", "module.", if present.
        let name = detect_prefix(pytorch_name)
            .map(|(_, remainder)| remainder)
            .unwrap_or(pytorch_name);

        Ok(name.to_string())
    }

    /// Convert OxiGAF name to PyTorch name
    ///
    /// Example: `down_blocks.0.resnets.0.conv1.weight` (with `prefix =
    /// Some("unet")`) → `unet.down_blocks.0.resnets.0.conv1.weight`
    ///
    /// The inverse of [`LayerMapping::pytorch_to_oxigaf`]: the dot-separated
    /// path is preserved verbatim and `prefix`, when supplied, is prepended
    /// with a separating dot. Passing the prefix that
    /// `pytorch_to_oxigaf::convert` recorded for the tensor reproduces the
    /// original PyTorch name exactly.
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

        // Add prefix if provided
        if let Some(prefix) = prefix {
            Ok(format!("{}.{}", prefix, oxigaf_name))
        } else {
            Ok(oxigaf_name.to_string())
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
                "down_blocks.0.resnets.0.conv1.weight",
            ),
            (
                "model.encoder.layer.2.attention.self.query.bias",
                "encoder.layer.2.attention.self.query.bias",
            ),
            ("simple.layer.weight", "simple.layer.weight"),
        ];

        for (pytorch, expected_oxigaf) in test_cases {
            let result = mapping
                .pytorch_to_oxigaf(pytorch)
                .expect("test: layer mapping should succeed");
            assert_eq!(result, expected_oxigaf, "Failed for: {}", pytorch);
        }
    }

    #[test]
    fn test_pytorch_to_oxigaf_produces_varbuilder_loadable_names() {
        // Regression test for the architecture gap this convention change
        // closed: `pytorch_to_oxigaf` used to emit a flat,
        // double-underscore-escaped name (`down__blocks_0_..._weight`) that
        // `candle_nn::VarBuilder::pp` cannot walk, while the ToRSh bridge's
        // `GafLayerMapper` emitted the dot-nested form that it can. The two
        // "OxiGAF" conventions must now agree, so a checkpoint from either
        // path loads in `oxigaf-diffusion`'s `VarBuilder`-based model code.
        let mapping = LayerMapping::new();
        let mapper = crate::GafLayerMapper::new();

        // Same logical layer, reached from each of the two source formats.
        let via_pytorch = mapping
            .pytorch_to_oxigaf("unet.down_blocks.0.resnets.0.conv1.weight")
            .expect("test: layer mapping should succeed");
        let via_torsh = mapper
            .map_torsh_to_oxigaf("down_blocks/0/resnets/0/conv1/weight")
            .expect("test: layer mapping should succeed");

        assert_eq!(via_pytorch, via_torsh);
        assert_eq!(via_pytorch, "down_blocks.0.resnets.0.conv1.weight");
        assert!(!via_pytorch.contains('/'), "VarBuilder paths use dots");
        // `VarBuilder::pp` splits on '.', so every path component the model
        // code walks (`vs.pp("down_blocks").pp("0")...`) must be present.
        let components: Vec<&str> = via_pytorch.split('.').collect();
        assert_eq!(
            components,
            vec!["down_blocks", "0", "resnets", "0", "conv1", "weight"]
        );
    }

    #[test]
    fn test_oxigaf_to_pytorch() {
        let mapping = LayerMapping::new();

        let test_cases = vec![
            (
                "down_blocks.0.resnets.0.conv1.weight",
                "down_blocks.0.resnets.0.conv1.weight",
            ),
            (
                "encoder.layer.2.attention.self.query.bias",
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
    fn test_oxigaf_to_pytorch_restores_prefix() {
        let mapping = LayerMapping::new();
        let result = mapping
            .oxigaf_to_pytorch("down_blocks.0.conv.weight", Some("unet"))
            .expect("test: layer mapping should succeed");
        assert_eq!(result, "unet.down_blocks.0.conv.weight");
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

    #[test]
    fn test_round_trip_survives_leading_underscore_components() {
        // Regression test: the old double-underscore escaping was not
        // injective -- `a._b` and `a_.b` both encoded to `a___b` -- so a
        // `torch.compile`-wrapped checkpoint (`_orig_mod.` components) could
        // not round-trip. The dot-preserving convention is the identity on
        // the un-prefixed remainder, so these must now come back exactly.
        let mapping = LayerMapping::new();

        for name in [
            "a._b",
            "a_.b",
            "_orig_mod.down_blocks.0.conv.weight",
            "down_blocks.0._extra_.weight",
            "trailing_",
            "__dunder__.weight",
        ] {
            let oxigaf = mapping
                .pytorch_to_oxigaf(name)
                .expect("test: layer mapping should succeed");
            let back = mapping
                .oxigaf_to_pytorch(&oxigaf, None)
                .expect("test: layer mapping should succeed");
            assert_eq!(back, name, "round-trip lost information for {}", name);
        }
    }

    #[test]
    fn test_oxigaf_torsh_agrees_with_gaf_layer_mapper() {
        // The two bridges must produce the same OxiGAF names in both
        // directions, or a checkpoint's provenance would decide whether it
        // loads.
        let mapping = LayerMapping::new();
        let mapper = crate::GafLayerMapper::new();

        for torsh in [
            "time_embedding/linear_1/weight",
            "down_blocks/0/resnets/0/norm1/weight",
            "up_blocks/3/attentions/0/transformer_blocks/0/attn1/to_q/weight",
            "conv_in/weight",
        ] {
            let via_layer_mapping = mapping
                .torsh_to_oxigaf(torsh)
                .expect("test: layer mapping should succeed");
            let via_gaf_mapper = mapper
                .map_torsh_to_oxigaf(torsh)
                .expect("test: layer mapping should succeed");
            assert_eq!(via_layer_mapping, via_gaf_mapper, "forward: {}", torsh);

            let back = mapping
                .oxigaf_to_torsh(&via_layer_mapping)
                .expect("test: layer mapping should succeed");
            assert_eq!(back, torsh, "reverse: {}", torsh);
        }
    }

    #[test]
    fn test_add_custom_mapping_drops_superseded_reverse_entry() {
        // Regression test: re-registering a `from` with a new `to` left the
        // *old* `to` behind in the reverse map, so a name the forward
        // mapping no longer produces still resolved backwards to `from` --
        // the two maps disagreed about what the mapping was.
        let mut mapping = LayerMapping::new();
        mapping.add_custom_mapping("a.weight".to_string(), "x".to_string());
        mapping.add_custom_mapping("a.weight".to_string(), "y".to_string());

        assert_eq!(mapping.lookup_custom("a.weight"), Some("y"));
        assert_eq!(
            mapping
                .oxigaf_to_pytorch("y", None)
                .expect("test: layer mapping should succeed"),
            "a.weight"
        );
        // "x" is no longer produced by the forward mapping, so it must fall
        // through to the mechanical path rather than resolving to "a.weight".
        assert_eq!(
            mapping
                .oxigaf_to_pytorch("x", None)
                .expect("test: layer mapping should succeed"),
            "x"
        );
    }

    #[test]
    fn test_add_custom_mapping_checked_rejects_target_collision() {
        let mut mapping = LayerMapping::new();
        mapping
            .add_custom_mapping_checked("a.weight".to_string(), "shared".to_string())
            .expect("test: first registration should succeed");

        // A different source aiming at the same target makes the reverse
        // direction lossy and must be rejected.
        let err = mapping.add_custom_mapping_checked("b.weight".to_string(), "shared".to_string());
        assert!(err.is_err(), "colliding target must be rejected");

        // ... and the rejected call must not have mutated anything.
        assert_eq!(mapping.lookup_custom("b.weight"), None);
        assert_eq!(
            mapping
                .oxigaf_to_pytorch("shared", None)
                .expect("test: layer mapping should succeed"),
            "a.weight"
        );

        // Re-registering the *same* source is not a collision.
        mapping
            .add_custom_mapping_checked("a.weight".to_string(), "shared".to_string())
            .expect("test: idempotent re-registration should succeed");
    }

    proptest::proptest! {
        /// `oxigaf_to_pytorch(pytorch_to_oxigaf(n), stripped_prefix) == n`
        /// for arbitrary dot-separated names, including the underscore and
        /// empty-component shapes the previous escaping mangled.
        #[test]
        fn prop_pytorch_oxigaf_round_trip(
            name in "[a-z_.0-9]{0,40}"
        ) {
            let mapping = LayerMapping::new();
            let prefix = detect_prefix(&name).map(|(prefix, _)| prefix);
            let oxigaf = mapping
                .pytorch_to_oxigaf(&name)
                .expect("test: layer mapping should succeed");
            let back = mapping
                .oxigaf_to_pytorch(&oxigaf, prefix)
                .expect("test: layer mapping should succeed");
            proptest::prop_assert_eq!(back, name);
        }

        /// Names that carry a recognized prefix round-trip through the
        /// prefix-restoring path, which is the one
        /// `pytorch_to_oxigaf::convert` / `oxigaf_to_pytorch::convert` use
        /// via the `__metadata__` prefix map.
        #[test]
        fn prop_prefixed_pytorch_oxigaf_round_trip(
            prefix in proptest::sample::select(KNOWN_PYTORCH_PREFIXES.to_vec()),
            rest in "[a-z_.0-9]{1,40}"
        ) {
            let mapping = LayerMapping::new();
            let name = format!("{}.{}", prefix, rest);
            let oxigaf = mapping
                .pytorch_to_oxigaf(&name)
                .expect("test: layer mapping should succeed");
            proptest::prop_assert_eq!(&oxigaf, &rest);
            let back = mapping
                .oxigaf_to_pytorch(&oxigaf, Some(prefix))
                .expect("test: layer mapping should succeed");
            proptest::prop_assert_eq!(back, name);
        }
    }
}
