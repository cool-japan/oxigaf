# OxiGAF-Bridge

Bidirectional weight conversion between PyTorch, OxiGAF, and ToRSh model formats.

**Status:** Stable — the full public API is implemented and tested (169/169
tests passing, see "Testing" below). If you're upgrading from an earlier
0.1.x release, read the 0.1.2 compatibility note under "OxiGAF" below first:
the OxiGAF-format bytes this crate writes changed at 0.1.2.

## Features

- **PyTorch ↔ OxiGAF**: Convert between PyTorch safetensors and OxiGAF native format
- **ToRSh ↔ OxiGAF** (`torsh` feature): Bidirectional conversion for ML training
- **Pure-Rust `.pt` / `.pkl` Ingest**: Decode a raw PyTorch checkpoint or a
  FLAME head model directly from its pickle stream — no Python, PyTorch,
  NumPy, or SciPy required (see "Pure-Rust `.pt` / `.pkl` Ingest" below)
- **Layer Name Mapping**: Automatic conversion between naming conventions
- **Precision Conversion**: FP32, FP16, BF16, with configurable precision per layer-name pattern
- **Round-Trip Validation**: `precision::validate_conversion` checks a
  converted tensor against a caller-supplied relative-error threshold
- **Checkpoint Validation** (`torsh` feature): `validation::validate_converted_checkpoint`
  structurally checks a converted `.safetensors` file (missing layers,
  malformed names, NaN/Inf, shape mismatches)

## Supported Formats

### PyTorch
- Layer naming: `unet.down_blocks.0.resnets.0.conv1.weight`
- Format: safetensors, or a raw `.pt` / `.pth` checkpoint via the pure-Rust
  pickle ingest — see "Pure-Rust `.pt` / `.pkl` Ingest" below

### OxiGAF
- Layer naming: `down_blocks.0.resnets.0.conv1.weight` — the model-rooted,
  dot-separated path `candle_nn::VarBuilder::pp` walks. There is exactly
  **one** OxiGAF naming convention: the PyTorch bridge and the ToRSh bridge
  both emit it (see "Layer Name Conventions" below).
- Format: safetensors

> **Compatibility note (0.1.2)**: before 0.1.2 the PyTorch bridge emitted a
> *different*, flat OxiGAF form (`down__blocks_0_resnets_0_conv1_weight`)
> that `VarBuilder::pp` could not walk and whose underscore escaping was not
> injective (`a._b` and `a_.b` both encoded to `a___b`). OxiGAF files
> written by an older version of this crate must be **re-converted from
> their PyTorch source** — the two forms are not interconvertible in general.

### ToRSh (feature-gated)
- Layer naming: `down_blocks/0/resnets/0/conv1/weight`
- Format: Native ToRSh model

## API Overview

| Type / function | Location | Feature |
|---|---|---|
| `WeightConverter` — `new`, `with_precision`, `with_precision_config`, `with_layer_mapping`, `pytorch_to_oxigaf`, `oxigaf_to_pytorch` | crate root | always |
| `WeightConverter::torsh_to_oxigaf` / `oxigaf_to_torsh` | crate root | `torsh` |
| `convert_pytorch_checkpoint`, `convert_flame_model`, `Component`, `ConversionReport` | `pickle` (re-exported at crate root) | always |
| `pickle::{read_checkpoint, TorchTensor}` | `pickle::torch` | always |
| `pickle::{read_flame_model, write_npy_dir, FlameModelData, NamedArray, ArrayValues}` | `pickle::flame` | always |
| `pickle::Value`; `pickle::PickleError` (also re-exported at crate root) | `pickle::value`, `pickle::error` | always |
| `LayerMapping`, `NamingConvention` (re-exported); `layer_mapping::detect_prefix` | `layer_mapping` | always |
| `GafLayerMapper` | `gaf_layer_mapper` (re-exported) | always |
| `Precision`, `PrecisionConfig` (re-exported); `precision::{validate_conversion, convert_precision, bytes_to_f32, f32_to_f16_bytes, f16_bytes_to_f32, f32_to_bf16_bytes, bf16_bytes_to_f32, dtype_of, float_precision_of}` | `precision` | always |
| `validation::{validate_converted_checkpoint, ValidationReport}` (re-exported) | `validation` | `torsh` |
| `BridgeError`, `Result` | `error` (re-exported) | always |

`create_synthetic_gaf_checkpoint` also exists, behind a `test-fixtures`
feature, but is intentionally left out of this table: its own doc comment
states it is not part of the crate's stable surface, kept only so this
crate's own tests and examples can build a synthetic checkpoint without
shipping a binary asset.

## Usage

### PyTorch Conversion

```rust
use oxigaf_bridge::{WeightConverter, Precision};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let converter = WeightConverter::new()
        .with_precision(Precision::FP32);

    // PyTorch to OxiGAF
    converter.pytorch_to_oxigaf(
        Path::new("pytorch_model.safetensors"),
        Path::new("oxigaf_model.safetensors")
    )?;

    // OxiGAF to PyTorch
    converter.oxigaf_to_pytorch(
        Path::new("oxigaf_model.safetensors"),
        Path::new("pytorch_model_out.safetensors")
    )?;

    Ok(())
}
```

### Pure-Rust `.pt` / `.pkl` Ingest

Until 0.1.2 the *first mile* of every OxiGAF pipeline ran through Python:
`scripts/convert_weights.py` needed PyTorch to turn a raw `.pt` checkpoint
into `.safetensors`, and `scripts/convert_flame.py` needed NumPy and SciPy to
turn a FLAME `.pkl` into `.npy` files. Both are Python pickle streams, and
reading a pickle does not require executing it: the `pickle` module ships a
non-executing unpickler — `GLOBAL`, `REDUCE`, `NEWOBJ` and `BUILD` opcodes
all produce inert data records instead of resolving or calling anything
(the classic `os.system` payload decodes to a data structure and does
nothing) — and layers two conversions on top of it:

| Was | Is now |
|---|---|
| `python scripts/convert_weights.py model.pt out/` | `convert_pytorch_checkpoint` |
| `python scripts/convert_flame.py FLAME.pkl out/` | `convert_flame_model` |

The Python scripts remain in the repository as a reference and an escape
hatch for exotic checkpoints, but nothing in the OxiGAF pipeline requires
them anymore.

```rust
use oxigaf_bridge::{convert_pytorch_checkpoint, Precision};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Splits the checkpoint's tensors into unet/vae/clip/other by the same
    // prefixes `scripts/convert_weights.py` recognized, and writes each
    // non-empty group as `<component>.safetensors` with dot-separated,
    // VarBuilder-loadable names. `Some(Precision::FP16)` casts every
    // floating-point tensor; pass `None` to keep each tensor's original
    // dtype (the Python script always forced FP16).
    let report = convert_pytorch_checkpoint(
        Path::new("checkpoint.pt"),
        Path::new("weights/"),
        Some(Precision::FP16),
    )?;
    println!("{} tensors written", report.total_tensors());
    Ok(())
}
```

```rust
use oxigaf_bridge::convert_flame_model;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Writes v_template, faces, shapedirs, expressiondirs, posedirs,
    // j_regressor, kintree_table and lbs_weights as .npy — the same
    // identity/expression split of shapedirs and densification of the
    // SciPy-sparse J_regressor that made scipy a dependency of the script.
    let written = convert_flame_model(Path::new("FLAME2023.pkl"), Path::new("flame_model/"))?;
    println!("wrote {} arrays", written.len());
    Ok(())
}
```

Equivalent CLI entry points ship as examples:

```bash
cargo run -p oxigaf-bridge --example convert_pytorch -- \
  --checkpoint checkpoint.pt --output-dir weights/ --precision fp16

cargo run -p oxigaf-bridge --example convert_flame_pkl -- \
  --model FLAME2023.pkl --output-dir flame_model/
```

### ToRSh Integration (Pure Rust)

ToRSh integration enables bidirectional weight conversion for GAF (Generative Avatar Face) models with complete layer mapping support.

#### Installation

Add to `Cargo.toml`:

```toml
[dependencies]
oxigaf-bridge = { version = "0.1.2", features = ["torsh"] }
```

#### Basic ToRSh Conversion

```rust
use oxigaf_bridge::{WeightConverter, Precision};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let converter = WeightConverter::new()
        .with_precision(Precision::FP16);

    // ToRSh → OxiGAF
    converter.torsh_to_oxigaf(
        Path::new("gaf_checkpoint.safetensors"),
        Path::new("oxigaf/unet.safetensors")
    )?;

    // OxiGAF → ToRSh
    converter.oxigaf_to_torsh(
        Path::new("oxigaf/unet.safetensors"),
        Path::new("gaf_checkpoint_new.safetensors")
    )?;

    Ok(())
}
```

#### GAF Model Components

The ToRSh bridge is used with the components that make up a GAF checkpoint:
the Multi-View U-Net, VAE, CLIP image encoder, and latent upsampler.

| Component | Role |
|-----------|------|
| Multi-View U-Net | Denoising backbone: time/camera embeddings, multi-view attention, IP-Adapter |
| VAE | Encoder/decoder, mid-blocks, quantization |
| CLIP Image Encoder | ViT-H/14 transformer stack |
| Latent Upsampler | 32×32 → 64×64 latent upsampling U-Net |

#### Layer Mapping

`GafLayerMapper` maps every ToRSh ⟷ OxiGAF name via a mechanical `/` ↔ `.`
substitution, plus a small override table (empty by default) for names that
are genuine exceptions to that rule — see `GafLayerMapper::add_override`.
This makes the mapping independent of any specific model's topology (U-Net
depth, block counts, ...): it needs no per-component table and cannot drift
out of sync with `oxigaf-diffusion`'s actual config. Coverage is checked by
round-trip and property-based tests in `gaf_layer_mapper.rs` and
`layer_mapping.rs`, not by a fixed layer count —
`GafLayerMapper::num_mappings()` reports the number of *explicit overrides*
registered (`0` on a freshly-created mapper), not a total layer count.

#### Accuracy

`precision::validate_conversion` compares original and converted values
against a caller-supplied *relative* error threshold
(`diff <= max_error * orig.abs().max(1.0)`). This crate's own tests use
`1e-6` for a full-precision (FP32) round trip; FP16/BF16 conversions are
lossy by construction (see "Precision Handling" below), so pick a threshold
that matches what your model actually tolerates.

#### Examples

See the `examples/` directory for complete usage:

- **convert_gaf_checkpoint.rs** - Basic conversion workflow
- **validate_conversion.rs** - Round-trip accuracy validation
- **batch_convert.rs** - Batch processing multiple checkpoints

```bash
# Convert a single checkpoint
cargo run --example convert_gaf_checkpoint --features torsh -- \
  --input gaf_checkpoint.safetensors \
  --output oxigaf/ \
  --precision fp16

# Validate conversion accuracy
cargo run --example validate_conversion --features torsh -- \
  --checkpoint gaf_checkpoint.safetensors

# Batch convert multiple files
cargo run --example batch_convert --features torsh -- \
  --input-dir checkpoints/ \
  --output-dir oxigaf/ \
  --precision fp16
```

### Custom Precision Config

```rust
use oxigaf_bridge::{WeightConverter, PrecisionConfig, Precision};

let mut config = PrecisionConfig::default();
config.set_layer_precision("normalization", Precision::FP32);
config.set_layer_precision("attention", Precision::FP16);

let converter = WeightConverter::new()
    .with_precision_config(config);
```

## COOLJAPAN Policies

- **No Unwrap**: All operations use proper error handling
- **Pure Rust**: 100% Pure Rust implementation
- **Workspace**: Uses workspace dependencies
- **Latest Crates**: safetensors 0.8, half 2.7, oxiarc-archive 0.4.1 (Pure-Rust ZIP reader for `.pt` checkpoints)

## Testing

All 169 tests pass as of 2026-08-28:

```bash
# Run the test suite
cargo test -p oxigaf-bridge

# Via nextest (169 tests)
cargo nextest run -p oxigaf-bridge --all-features
```

This crate's `[dev-dependencies]` include a self-dependency
(`oxigaf-bridge = { path = ".", features = ["test-fixtures"] }`) so its own
integration tests and examples can share fixtures. Because Cargo unifies
feature selection across a package's roles within one build, **every**
`cargo test` / `cargo nextest run` invocation for this crate — even with no
`--features` flag at all — compiles and runs the full `torsh`-gated test
suite; there is no way to exercise only the default-feature surface from
within this crate's own test run. The `torsh` gate is still real for
anyone *depending* on this crate, though: dev-dependencies never propagate
to consumers, so a plain `cargo build` of a crate that depends on
`oxigaf-bridge` without `features = ["torsh"]` does not compile
`oxigaf_to_torsh`, `torsh_to_oxigaf`, or `validation` at all.

## Architecture

### Layer Name Conventions

The bridge converts between three naming conventions. There is exactly
**one** OxiGAF form — both the PyTorch bridge (`LayerMapping`, strips a
recognized top-level prefix: `unet.`, `model.`, or `module.`) and the ToRSh
bridge (`GafLayerMapper`, a direct `/` ↔ `.` substitution) emit it, so a
checkpoint converted through either path loads in `oxigaf-diffusion`'s
`VarBuilder`-based model code on the same footing:

| Component | PyTorch | OxiGAF | ToRSh |
|-----------|---------|--------|-------|
| U-Net time embedding | `unet.time_embedding.linear_1.weight` | `time_embedding.linear_1.weight` | `time_embedding/linear_1/weight` |
| Down block ResNet | `unet.down_blocks.0.resnets.0.norm1.weight` | `down_blocks.0.resnets.0.norm1.weight` | `down_blocks/0/resnets/0/norm1/weight` |
| Self-attention Q | `unet.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight` | `down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight` | `down_blocks/0/attentions/0/transformer_blocks/0/attn1/to_q/weight` |

### Precision Handling

- **FP32**: Full precision, `<1e-6` round-trip error (the threshold this crate's own tests assert)
- **FP16**: Half precision — exactly half the stored bytes of FP32 per tensor; lossy, so validate against a threshold appropriate to your model with `precision::validate_conversion`
- **BF16**: Brain float, better dynamic range than FP16
- **Mixed precision**: Per-layer-pattern precision control via `PrecisionConfig::set_layer_precision` (e.g., FP32 for normalization, FP16 for weights)

## Documentation

For comprehensive documentation see:

- **API docs**: `cargo doc --features torsh --open`
- **Layer mapping reference**: See the `GafLayerMapper` and `LayerMapping` documentation
- **Example timing**: `batch_convert`, `convert_gaf_checkpoint`, and
  `validate_conversion` print wall-clock duration for each run (add
  `--verbose` for debug-level logs); this crate has no calibrated benchmark
  suite (no `benches/` directory)
- **Migration guide**: See the examples for step-by-step conversion workflows

## License

Apache-2.0

## Authors

COOLJAPAN OU (Team Kitasan)

Contributions welcome.
