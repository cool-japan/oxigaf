//! Pure-Rust ingest of Python pickle data: PyTorch `.pt` checkpoints and
//! FLAME `.pkl` head models.
//!
//! # Why this exists
//!
//! Until 0.1.2 the *first mile* of every OxiGAF pipeline ran through Python.
//! `scripts/convert_weights.py` needed PyTorch to turn a `.pt` checkpoint
//! into `.safetensors`; `scripts/convert_flame.py` needed NumPy and SciPy to
//! turn a FLAME `.pkl` into `.npy` files. Both are pickle formats, and
//! reading a pickle was assumed to require a Python interpreter.
//!
//! It does not. A pickle is a bytecode stream for a stack machine, and
//! nothing about *reading* one requires *executing* it. This module
//! implements that reader, and with it the two conversions:
//!
//! | Was | Is now |
//! |---|---|
//! | `python scripts/convert_weights.py model.pt out/` | [`convert_pytorch_checkpoint`] |
//! | `python scripts/convert_flame.py FLAME.pkl out/` | [`convert_flame_model`] |
//!
//! The Python scripts remain in the repository as a reference and an escape
//! hatch for exotic checkpoints, but nothing in the OxiGAF pipeline requires
//! them.
//!
//! # Safety
//!
//! Unpickling arbitrary data is famously dangerous *in Python*, because
//! CPython's unpickler resolves `GLOBAL` opcodes to real callables and
//! `REDUCE` calls them -- which is why `torch.load` grew a `weights_only`
//! flag. This reader never resolves or calls anything: `GLOBAL`, `REDUCE`,
//! `NEWOBJ` and `BUILD` produce inert [`value::Value`] records, and each
//! format interpreter decides which of those *it* recognizes. The classic
//! `os.system` payload decodes to a data structure and does nothing (there
//! is a test asserting exactly that in [`vm`]).
//!
//! # Module layout
//!
//! - [`value`] -- the decoded value tree.
//! - [`vm`] -- the non-executing unpickler (protocols 0-5).
//! - [`torch`] -- PyTorch `.pt` ZIP container + tensor rebuild records.
//! - [`numpy`] -- NumPy arrays, chumpy wrappers, SciPy sparse matrices.
//! - [`flame`] -- the FLAME `.pkl` → `.npy` conversion.
//! - [`error`] -- the error type all of the above share.

pub mod error;
pub mod flame;
pub mod numpy;
pub mod torch;
pub mod value;
pub mod vm;

#[cfg(test)]
mod real_fixtures;
#[cfg(test)]
mod test_support;

pub use error::{PickleError, Result};
pub use flame::{read_flame_model, write_npy_dir, ArrayValues, FlameModelData, NamedArray};
pub use torch::{read_checkpoint, TorchTensor};
pub use value::Value;

use safetensors::{tensor::TensorView, Dtype};
use std::collections::BTreeMap;
use std::path::Path;

/// Which model component a checkpoint tensor belongs to.
///
/// A GAF / ImageDream `.pt` bundles the U-Net, VAE and CLIP encoder in one
/// file under distinguishing prefixes; `oxigaf-diffusion` loads each from
/// its own `.safetensors`, so the conversion has to split them. The prefixes
/// are the same ones `scripts/convert_weights.py` recognized.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Component {
    /// The denoising U-Net.
    Unet,
    /// The variational autoencoder.
    Vae,
    /// The CLIP image encoder.
    Clip,
    /// Everything that matched no known prefix.
    Other,
}

impl Component {
    /// The output file stem, e.g. `unet` for `unet.safetensors`.
    pub fn file_stem(self) -> &'static str {
        match self {
            Self::Unet => "unet",
            Self::Vae => "vae",
            Self::Clip => "clip",
            Self::Other => "other",
        }
    }

    /// Every component, in output order.
    pub fn all() -> [Self; 4] {
        [Self::Unet, Self::Vae, Self::Clip, Self::Other]
    }
}

/// Prefix table: `(prefix, component)`. The prefix is stripped from the
/// tensor name, leaving the model-rooted, dot-separated path
/// `candle_nn::VarBuilder::pp` walks -- the same OxiGAF convention
/// [`crate::layer_mapping::LayerMapping`] produces.
const COMPONENT_PREFIXES: &[(&str, Component)] = &[
    ("model.diffusion_model.", Component::Unet),
    ("unet.", Component::Unet),
    ("first_stage_model.", Component::Vae),
    ("vae.", Component::Vae),
    ("cond_stage_model.", Component::Clip),
    ("clip.", Component::Clip),
];

/// Classifies one tensor name, returning its component and the name with
/// the component prefix removed.
fn classify(name: &str) -> (Component, String) {
    for &(prefix, component) in COMPONENT_PREFIXES {
        if let Some(rest) = name.strip_prefix(prefix) {
            return (component, rest.to_string());
        }
    }
    (Component::Other, name.to_string())
}

/// Report of a completed `.pt` → `.safetensors` conversion.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ConversionReport {
    /// Per-component `(file stem, tensor count, total element count)`, for
    /// components that had at least one tensor.
    pub components: Vec<(&'static str, usize, usize)>,
}

impl ConversionReport {
    /// Total tensors written across every component.
    pub fn total_tensors(&self) -> usize {
        self.components.iter().map(|(_, count, _)| count).sum()
    }
}

/// Converts a PyTorch `.pt` / `.pth` checkpoint into per-component
/// `.safetensors` files in `output_dir`.
///
/// This is the Rust replacement for `scripts/convert_weights.py`: tensors
/// are partitioned into `unet` / `vae` / `clip` / `other` by the same
/// prefixes, their component prefix is stripped, and each non-empty group is
/// written as `<component>.safetensors`. `target_dtype` chooses the stored
/// precision -- pass `None` to keep each tensor's original dtype (the Python
/// script always forced FP16).
///
/// # Errors
///
/// Returns [`crate::BridgeError`] if the checkpoint cannot be read (see
/// [`torch::read_checkpoint`]), if a tensor cannot be re-encoded at
/// `target_dtype`, or if an output file cannot be written.
pub fn convert_pytorch_checkpoint(
    checkpoint: &Path,
    output_dir: &Path,
    target_dtype: Option<crate::Precision>,
) -> crate::Result<ConversionReport> {
    let tensors = torch::read_checkpoint(checkpoint)?;
    tracing::info!(
        "Read {} tensors from {}",
        tensors.len(),
        checkpoint.display()
    );

    let mut grouped: BTreeMap<Component, Vec<(String, TorchTensor)>> = BTreeMap::new();
    for tensor in tensors {
        let (component, name) = classify(&tensor.name);
        grouped.entry(component).or_default().push((name, tensor));
    }

    std::fs::create_dir_all(output_dir)?;
    let mut report = ConversionReport::default();

    for component in Component::all() {
        let Some(members) = grouped.get(&component) else {
            continue;
        };
        if members.is_empty() {
            continue;
        }

        // Re-encode first so the byte buffers outlive the views below.
        let mut encoded: Vec<(String, Vec<u8>, Vec<usize>, Dtype)> =
            Vec::with_capacity(members.len());
        let mut elements = 0usize;
        for (name, tensor) in members {
            let (bytes, dtype) = reencode(tensor, target_dtype)?;
            elements += tensor.shape.iter().product::<usize>();
            encoded.push((name.clone(), bytes, tensor.shape.clone(), dtype));
        }

        let mut views = Vec::with_capacity(encoded.len());
        for (name, bytes, shape, dtype) in &encoded {
            let view = TensorView::new(*dtype, shape.clone(), bytes)
                .map_err(|e| crate::BridgeError::SafeTensors(e.to_string()))?;
            views.push((name.as_str(), view));
        }

        let path = output_dir.join(format!("{}.safetensors", component.file_stem()));
        let serialized = safetensors::tensor::serialize(views, None)
            .map_err(|e| crate::BridgeError::SafeTensors(e.to_string()))?;
        std::fs::write(&path, serialized)?;

        tracing::info!(
            "{}: {} tensors, {} params -> {}",
            component.file_stem(),
            encoded.len(),
            elements,
            path.display()
        );
        report
            .components
            .push((component.file_stem(), encoded.len(), elements));
    }

    if report.components.is_empty() {
        return Err(crate::BridgeError::Conversion(format!(
            "{} yielded no tensors to convert",
            checkpoint.display()
        )));
    }
    Ok(report)
}

/// Re-encodes one tensor at `target_dtype`, or passes it through unchanged.
///
/// A non-float tensor (integer bookkeeping, boolean masks) has no meaningful
/// "precision" and always passes through, matching how
/// [`crate::pytorch_to_oxigaf`] treats one.
fn reencode(
    tensor: &TorchTensor,
    target: Option<crate::Precision>,
) -> crate::Result<(Vec<u8>, Dtype)> {
    use crate::precision::{bytes_to_f32, convert_precision, dtype_of, float_precision_of};

    let (Some(target), Some(source)) = (target, float_precision_of(tensor.dtype)) else {
        return Ok((tensor.data.clone(), tensor.dtype));
    };
    if source == target {
        return Ok((tensor.data.clone(), tensor.dtype));
    }

    let values = bytes_to_f32(&tensor.data, source)?;
    let (bytes, saturated) = convert_precision(&values, target);
    if saturated > 0 {
        tracing::warn!(
            "{} value(s) in '{}' saturated to +/-infinity converting to {}",
            saturated,
            tensor.name,
            target.name()
        );
    }
    Ok((bytes, dtype_of(target)))
}

/// Converts a FLAME `.pkl` model into the directory of `.npy` files
/// `oxigaf_flame::io::load_flame_model` reads.
///
/// This is the Rust replacement for `scripts/convert_flame.py`, including
/// the identity/expression split of `shapedirs` and the densification of the
/// SciPy-sparse `J_regressor` that made `scipy` a dependency of the script.
///
/// # Errors
///
/// Returns [`crate::BridgeError`] if the model cannot be read (see
/// [`flame::read_flame_model`]) or an output file cannot be written.
pub fn convert_flame_model(
    model: &Path,
    output_dir: &Path,
) -> crate::Result<Vec<(String, Vec<usize>)>> {
    let data = flame::read_flame_model(model)?;
    flame::write_npy_dir(&data, output_dir)?;
    Ok(data
        .arrays()
        .iter()
        .map(|array| (array.name.to_string(), array.shape.clone()))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_strips_every_known_prefix() {
        assert_eq!(
            classify("model.diffusion_model.input_blocks.0.weight"),
            (Component::Unet, "input_blocks.0.weight".to_string())
        );
        assert_eq!(
            classify("unet.conv_in.weight"),
            (Component::Unet, "conv_in.weight".to_string())
        );
        assert_eq!(
            classify("first_stage_model.encoder.conv_in.weight"),
            (Component::Vae, "encoder.conv_in.weight".to_string())
        );
        assert_eq!(
            classify("vae.decoder.norm.weight"),
            (Component::Vae, "decoder.norm.weight".to_string())
        );
        assert_eq!(
            classify("cond_stage_model.transformer.weight"),
            (Component::Clip, "transformer.weight".to_string())
        );
        assert_eq!(
            classify("logit_scale"),
            (Component::Other, "logit_scale".to_string())
        );
    }

    #[test]
    fn test_classified_names_are_varbuilder_paths() {
        // The stripped name must be the dot-separated path
        // `candle_nn::VarBuilder::pp` walks -- the same convention
        // `LayerMapping::pytorch_to_oxigaf` produces -- or the converted
        // file would not load in `oxigaf-diffusion`.
        let (_, name) = classify("model.diffusion_model.down_blocks.0.resnets.0.conv1.weight");
        assert_eq!(name, "down_blocks.0.resnets.0.conv1.weight");
        assert!(!name.contains('/'));

        let via_layer_mapping = crate::LayerMapping::new()
            .pytorch_to_oxigaf("unet.down_blocks.0.resnets.0.conv1.weight")
            .expect("test: layer mapping should succeed");
        assert_eq!(name, via_layer_mapping);
    }

    #[test]
    fn test_reencode_passes_non_float_tensors_through() {
        let tensor = TorchTensor {
            name: "position_ids".to_string(),
            dtype: Dtype::I64,
            shape: vec![2],
            data: vec![0u8; 16],
        };
        let (bytes, dtype) =
            reencode(&tensor, Some(crate::Precision::FP16)).expect("test: reencode");
        assert_eq!(dtype, Dtype::I64);
        assert_eq!(bytes, tensor.data);
    }

    #[test]
    fn test_reencode_converts_f32_to_f16() {
        let tensor = TorchTensor {
            name: "w".to_string(),
            dtype: Dtype::F32,
            shape: vec![2],
            data: [1.0f32, 2.0].iter().flat_map(|v| v.to_le_bytes()).collect(),
        };
        let (bytes, dtype) =
            reencode(&tensor, Some(crate::Precision::FP16)).expect("test: reencode");
        assert_eq!(dtype, Dtype::F16);
        assert_eq!(bytes.len(), 4);
    }

    #[test]
    fn test_reencode_without_a_target_keeps_the_source_dtype() {
        let tensor = TorchTensor {
            name: "w".to_string(),
            dtype: Dtype::F32,
            shape: vec![1],
            data: 1.0f32.to_le_bytes().to_vec(),
        };
        let (bytes, dtype) = reencode(&tensor, None).expect("test: reencode");
        assert_eq!(dtype, Dtype::F32);
        assert_eq!(bytes, tensor.data);
    }
}
