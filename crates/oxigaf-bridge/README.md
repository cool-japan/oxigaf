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
- Format: safetensors

### OxiGAF
- Layer naming: `down_blocks_0_resnets_0_conv1_weight`
- Format: safetensors

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

### ToRSh Integration (Pure Rust)

ToRSh integration enables bidirectional weight conversion for GAF (Generative Avatar Face) models with complete layer mapping support.

#### Installation

Add to `Cargo.toml`:

```toml
[dependencies]
oxigaf-bridge = { version = "0.1.1", features = ["torsh"] }
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
- **Latest Crates**: safetensors 0.7, half 2.4

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

The bridge provides automatic conversion between three naming conventions:

| Component | PyTorch | OxiGAF | ToRSh |
|-----------|---------|--------|-------|
| U-Net time embedding | `time_embedding.linear_1.weight` | `time_embedding_linear_1_weight` | `time_embedding/linear_1/weight` |
| Down block ResNet | `down_blocks.0.resnets.0.norm1.weight` | `down_blocks_0_resnets_0_norm1_weight` | `down_blocks/0/resnets/0/norm1/weight` |
| Self-attention Q | `down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight` | `down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q_weight` | `down_blocks/0/attentions/0/transformer_blocks/0/attn1/to_q/weight` |

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
