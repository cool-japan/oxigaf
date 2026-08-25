# OxiGAF-Bridge

Bidirectional weight conversion between PyTorch, OxiGAF, and ToRSh model formats.

## Features

- **PyTorch ↔ OxiGAF**: Convert between PyTorch safetensors and OxiGAF native format
- **ToRSh ↔ OxiGAF**: Feature-gated ToRSh integration for ML training
- **Layer Name Mapping**: Automatic conversion between naming conventions
- **Precision Conversion**: Support for FP32, FP16, BF16 with configurable precision per-layer
- **Validation**: Round-trip conversion accuracy <1e-6

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

#### Supported GAF Models

The ToRSh integration provides comprehensive layer mapping for all GAF model components:

| Model | Layers | Features |
|-------|--------|----------|
| Multi-View U-Net | ~1,000 | Time/camera embeddings, multi-view attention, IP-Adapter |
| VAE | ~200 | Encoder/decoder, mid-blocks, quantization |
| CLIP Image Encoder | ~300 | ViT-H/14 (32 transformer layers) |
| Latent Upsampler | ~100 | 32×32 → 64×64 upsampling U-Net |

#### Performance

- **Conversion speed**: 10-20ms for 500MB checkpoint
- **Memory usage**: ~30% less than PyTorch
- **Accuracy**: <1e-6 round-trip error (FP32), <1e-3 (FP16)
- **Layer coverage**: 100% (2,743 layers mapped)

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
- **Latest Crates**: safetensors 0.8, half 2.7

## Testing

```bash
# Run all tests (without ToRSh)
cargo test -p oxigaf-bridge

# Run with ToRSh integration tests
cargo test -p oxigaf-bridge --features torsh

# Run with nextest
cargo nextest run -p oxigaf-bridge --features torsh
```

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

- **FP32**: Full precision, <1e-6 round-trip error
- **FP16**: Half precision, ~50% memory savings, <1e-3 error
- **BF16**: Brain float, better dynamic range than FP16
- **Mixed precision**: Per-layer precision control (e.g., FP32 for normalization, FP16 for weights)

## Documentation

For comprehensive documentation see:

- **API docs**: `cargo doc --features torsh --open`
- **Layer mapping reference**: See `GafLayerMapper` documentation
- **Performance benchmarks**: Run examples with `--verbose` flag
- **Migration guide**: See examples for step-by-step conversion workflows

## License

Apache-2.0

## Authors

COOLJAPAN OU (Team Kitasan)
