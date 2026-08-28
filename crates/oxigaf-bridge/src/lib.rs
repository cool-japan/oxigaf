//! OxiGAF Bridge - Bidirectional Weight Conversion
//!
//! This crate provides bidirectional weight conversion between:
//! - PyTorch safetensors format
//! - OxiGAF native format
//! - ToRSh model format (feature-gated)
//!
//! # Features
//!
//! - Layer name mapping between different frameworks
//! - Precision conversion (FP32, FP16, BF16)
//! - Round-trip conversion with validation
//! - Comprehensive error handling
//!
//! # PyTorch Example
//!
//! ```rust,no_run
//! use oxigaf_bridge::{WeightConverter, Precision};
//! use std::path::Path;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let converter = WeightConverter::new()
//!     .with_precision(Precision::FP32);
//!
//! // PyTorch → OxiGAF
//! converter.pytorch_to_oxigaf(
//!     Path::new("model.safetensors"),
//!     Path::new("model_oxigaf.safetensors")
//! )?;
//!
//! // OxiGAF → PyTorch
//! converter.oxigaf_to_pytorch(
//!     Path::new("model_oxigaf.safetensors"),
//!     Path::new("model_pytorch.safetensors")
//! )?;
//! # Ok(())
//! # }
//! ```
//!
//! # ToRSh Example (requires `torsh` feature)
//!
//! ```rust,no_run
//! # use std::path::Path;
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # #[cfg(feature = "torsh")]
//! # {
//! use oxigaf_bridge::{WeightConverter, Precision};
//!
//! let converter = WeightConverter::new()
//!     .with_precision(Precision::FP16);
//!
//! // ToRSh → OxiGAF
//! converter.torsh_to_oxigaf(
//!     Path::new("gaf_checkpoint.safetensors"),
//!     Path::new("oxigaf/unet.safetensors")
//! )?;
//!
//! // OxiGAF → ToRSh
//! converter.oxigaf_to_torsh(
//!     Path::new("oxigaf/unet.safetensors"),
//!     Path::new("gaf_checkpoint_new.safetensors")
//! )?;
//! # }
//! # Ok(())
//! # }
//! ```
//!
//! # Layer Name Conventions
//!
//! There is exactly **one** OxiGAF name convention, and both bridges emit
//! it: the model-rooted, dot-separated path `candle_nn::VarBuilder::pp`
//! walks.
//!
//! | Convention | Example |
//! |---|---|
//! | PyTorch | `unet.down_blocks.0.resnets.0.conv1.weight` |
//! | OxiGAF | `down_blocks.0.resnets.0.conv1.weight` |
//! | ToRSh | `down_blocks/0/resnets/0/conv1/weight` |
//!
//! - The PyTorch bridge ([`layer_mapping::LayerMapping`]) strips a
//!   recognized top-level prefix (`unet.`, `model.`, `module.`) and
//!   preserves the dot-separated remainder verbatim, recording the stripped
//!   prefix per tensor in the output's `__metadata__` so the reverse
//!   direction restores the exact original name.
//! - The ToRSh bridge ([`gaf_layer_mapper::GafLayerMapper`]) performs the
//!   direct `/` ↔ `.` substitution.
//!
//! A checkpoint produced by `WeightConverter::pytorch_to_oxigaf` is
//! therefore loadable by `oxigaf-diffusion`'s `VarBuilder`-based model code
//! (`vs.pp("down_blocks").pp("0")…`, see `oxigaf-diffusion/src/unet.rs`) on
//! exactly the same footing as one produced by
//! `WeightConverter::torsh_to_oxigaf`.
//!
//! ## Compatibility note (0.1.2)
//!
//! Before 0.1.2 the PyTorch bridge emitted a *different*, flat OxiGAF form
//! (`down__blocks_0_resnets_0_conv1_weight`) that `VarBuilder::pp` could not
//! walk and whose underscore escaping was not injective (`a._b` and `a_.b`
//! both encoded to `a___b`). OxiGAF files written by an older version of
//! this crate must be re-converted from their PyTorch source; the two forms
//! are not interconvertible in general.

// The no-unwrap policy applies to every production path in this crate.
// Test code is exempt: `cfg(test)` is set for the whole crate when
// compiling the test harness, so these only bind in ordinary builds. The
// `panic` lint is gated the same way rather than left unconditional,
// because inline `#[cfg(test)]` modules legitimately `panic!` inside
// `unwrap_or_else` assertion helpers.
#![cfg_attr(not(test), deny(clippy::unwrap_used))]
#![cfg_attr(not(test), deny(clippy::expect_used))]
#![cfg_attr(not(test), warn(clippy::panic))]

pub mod error;
pub mod gaf_layer_mapper;
pub mod layer_mapping;
pub mod oxigaf_to_pytorch;
pub mod pickle;
pub mod precision;
pub mod pytorch_to_oxigaf;

#[cfg(feature = "torsh")]
pub mod oxigaf_to_torsh;
#[cfg(feature = "torsh")]
pub mod torsh_to_oxigaf;
#[cfg(feature = "torsh")]
pub mod validation;

// Re-exports
pub use error::{BridgeError, Result};
pub use gaf_layer_mapper::GafLayerMapper;
pub use layer_mapping::{LayerMapping, NamingConvention};
pub use pickle::{
    convert_flame_model, convert_pytorch_checkpoint, Component, ConversionReport, PickleError,
};
pub use precision::{Precision, PrecisionConfig};

#[cfg(feature = "torsh")]
pub use validation::{validate_converted_checkpoint, ValidationReport};

/// Test-fixture generator, available only with the `test-fixtures` feature.
///
/// This is deliberately **not** part of the crate's stable surface: it
/// exists so this crate's own integration tests (and downstream tests that
/// opt in) can build a synthetic GAF checkpoint without shipping a binary
/// asset. Enabling `test-fixtures` also enables `torsh`.
#[cfg(feature = "test-fixtures")]
pub use validation::create_synthetic_gaf_checkpoint;

use std::path::Path;

/// Main weight converter interface
pub struct WeightConverter {
    layer_mapping: LayerMapping,
    precision_config: PrecisionConfig,
}

impl WeightConverter {
    /// Create a new weight converter with default settings
    pub fn new() -> Self {
        Self {
            layer_mapping: LayerMapping::new(),
            precision_config: PrecisionConfig::default(),
        }
    }

    /// Set the precision for conversion
    pub fn with_precision(mut self, precision: Precision) -> Self {
        self.precision_config.set_default_precision(precision);
        self
    }

    /// Set a custom precision config
    pub fn with_precision_config(mut self, config: PrecisionConfig) -> Self {
        self.precision_config = config;
        self
    }

    /// Set a custom layer mapping
    pub fn with_layer_mapping(mut self, mapping: LayerMapping) -> Self {
        self.layer_mapping = mapping;
        self
    }

    /// Convert PyTorch weights to OxiGAF format
    ///
    /// # Arguments
    ///
    /// * `pytorch_path` - Path to PyTorch safetensors file
    /// * `oxigaf_path` - Output path for OxiGAF format
    ///
    /// # Errors
    ///
    /// Returns error if file I/O fails or conversion fails
    pub fn pytorch_to_oxigaf(&self, pytorch_path: &Path, oxigaf_path: &Path) -> Result<()> {
        pytorch_to_oxigaf::convert(
            pytorch_path,
            oxigaf_path,
            &self.layer_mapping,
            &self.precision_config,
        )
    }

    /// Convert OxiGAF weights to PyTorch format
    ///
    /// # Arguments
    ///
    /// * `oxigaf_path` - Path to OxiGAF safetensors file
    /// * `pytorch_path` - Output path for PyTorch format
    ///
    /// # Errors
    ///
    /// Returns error if file I/O fails or conversion fails
    pub fn oxigaf_to_pytorch(&self, oxigaf_path: &Path, pytorch_path: &Path) -> Result<()> {
        oxigaf_to_pytorch::convert(
            oxigaf_path,
            pytorch_path,
            &self.layer_mapping,
            &self.precision_config,
        )
    }

    /// Convert ToRSh weights to OxiGAF format
    ///
    /// # Arguments
    ///
    /// * `torsh_path` - Path to ToRSh safetensors file
    /// * `oxigaf_path` - Output path for OxiGAF format
    ///
    /// # Errors
    ///
    /// Returns error if conversion fails or file I/O fails
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use oxigaf_bridge::{WeightConverter, Precision};
    /// # use std::path::Path;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # #[cfg(feature = "torsh")]
    /// # {
    /// let converter = WeightConverter::new()
    ///     .with_precision(Precision::FP16);
    ///
    /// converter.torsh_to_oxigaf(
    ///     Path::new("gaf_checkpoint.safetensors"),
    ///     Path::new("oxigaf/unet.safetensors")
    /// )?;
    /// # }
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(feature = "torsh")]
    pub fn torsh_to_oxigaf(&self, torsh_path: &Path, oxigaf_path: &Path) -> Result<()> {
        torsh_to_oxigaf::convert(
            torsh_path,
            oxigaf_path,
            &self.layer_mapping,
            &self.precision_config,
        )
    }

    /// Convert OxiGAF weights to ToRSh format
    ///
    /// # Arguments
    ///
    /// * `oxigaf_path` - Path to OxiGAF safetensors file
    /// * `torsh_path` - Output path for ToRSh format
    ///
    /// # Errors
    ///
    /// Returns error if conversion fails or file I/O fails
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use oxigaf_bridge::{WeightConverter, Precision};
    /// # use std::path::Path;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # #[cfg(feature = "torsh")]
    /// # {
    /// let converter = WeightConverter::new()
    ///     .with_precision(Precision::FP32);
    ///
    /// converter.oxigaf_to_torsh(
    ///     Path::new("oxigaf/unet.safetensors"),
    ///     Path::new("gaf_checkpoint.safetensors")
    /// )?;
    /// # }
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(feature = "torsh")]
    pub fn oxigaf_to_torsh(&self, oxigaf_path: &Path, torsh_path: &Path) -> Result<()> {
        oxigaf_to_torsh::convert(
            oxigaf_path,
            torsh_path,
            &self.layer_mapping,
            &self.precision_config,
        )
    }
}

impl Default for WeightConverter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_converter_creation() {
        let converter = WeightConverter::new();
        assert_eq!(
            converter.precision_config.default_precision(),
            Precision::FP32
        );
    }

    #[test]
    fn test_converter_with_precision() {
        let converter = WeightConverter::new().with_precision(Precision::FP16);
        assert_eq!(
            converter.precision_config.default_precision(),
            Precision::FP16
        );
    }
}
