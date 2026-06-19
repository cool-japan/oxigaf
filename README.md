# OxiGAF

**Pure Rust Gaussian Avatar Reconstruction** from monocular videos via multi-view diffusion.

Implements the methods from [GAF: Gaussian Avatar Reconstruction from Monocular Videos via Multi-View Diffusion](https://arxiv.org/abs/2412.10209) entirely in the Rust ecosystem.

## What's in v0.1.2

### 512×512 Multi-View Generation (v0.1.0)
- **Latent Upsampler**: 32×32 → 64×64 latent upsampling for 512×512 output resolution
- **IP-Adapter**: Identity-preserving image conditioning for consistent face/object generation
- **Classifier-Free Guidance**: Quality improvement with configurable guidance scale (1.0-20.0)
- **Multi-view UNet**: Cross-view attention for geometric consistency across views
- **Camera Conditioning**: Explicit camera pose embeddings for view-aware generation

### Extended Head Model (v0.1.1)
- **Avatar Rigging & Expressions**: `AvatarRig`, `GazeController`, expression animation with FACS AU coefficients, phoneme-driven animation, emotion recognition
- **Mesh Processing Suite**: Mesh repair, smoothing, Loop/Catmull-Clark subdivision, morphing, geodesic distance, spectral analysis
- **UV & Texture Pipeline**: UV parameterisation, texture baking, face atlas generation, albedo maps, SH lighting
- **Motion & Deformation**: Timeline, warp field, shape retargeting, dynamic landmark tracking

### Comprehensive Rendering Pipeline (v0.1.1)
- **Post-Processing**: SSAO, bloom, depth-of-field, motion blur, HDR tone mapping, film grain, chromatic aberration, TAA
- **Scene Composition**: Render graph, image compositor, silhouette extraction, background synthesis
- **Volumetric Rendering**: Ray types, camera model, volume grid, ray-march result traits
- **Stereo Output**: Side-by-side and top-bottom stereo rendering

### Extended Training & Diffusion (v0.1.1)
- **Sampler Suite**: DDPM, adaptive sampling, consistency model, flow matching, guidance rescaling
- **LoRA & ControlNet Adapters**: Parameter-efficient fine-tuning and conditioning adapters
- **Curriculum Learning**: Progressive training, few-shot adaptation, meta-learning (MAML), continual learning
- **Gradient Tools**: Gradient surgery, OHEM, anomaly detection, activation maps

### Expanded CLI Toolset (v0.1.1)
- **Export Suite**: PLY, glTF, mesh, point cloud, video, and animation sequence export
- **Analysis Tools**: Scene analyser, model inspector, diff tool, model comparison, quality checker
- **Scene Operations**: Scene merging, optimiser, streaming, Gaussian filter and deduplication
- **Visualisation**: Arcball camera controller, LOD generator, camera path editor, live dashboard

### Quality & Performance
- **100% Pure Rust**: Zero C/Fortran dependencies (COOLJAPAN compliant)
- **12537 Tests Passing**: Comprehensive validation across all crates
- **Production Ready**: Zero unwrap(), all files <2000 lines, feature-gated dependencies

## Workspace Structure

| Crate | Type | Description |
|-------|------|-------------|
| `oxigaf-flame` | lib | FLAME parametric head model (LBS, normal maps, safetensors I/O, video sequences) |
| `oxigaf-diffusion` | lib | Multi-view diffusion with IP-Adapter, upsampling, and CFG (candle) |
| `oxigaf-render` | lib | Differentiable 3D Gaussian Splatting rasterizer with CPU reference (wgpu) |
| `oxigaf-trainer` | lib | Optimization pipeline with gradient verification and FLAME binding backward |
| `oxigaf-bridge` | lib | PyTorch ↔ OxiGAF weight conversion and layer mapping utilities |
| `oxigaf` | lib | Meta crate — unified re-export of all sub-crates |
| `oxigaf-cli` | bin | CLI binary (`oxigaf` command) |

## Quick Start

```bash
# Build the workspace
cargo build --workspace

# Run the CLI
cargo run -p oxigaf-cli -- --help

# Run tests
cargo test --workspace
```

## Feature Flags

OxiGAF supports various feature flags for platform-specific optimizations:

### Platform-Independent Features

| Feature | Description |
|---------|-------------|
| `simd` | SIMD optimizations for FLAME model (requires nightly Rust) |
| `parallel` | Parallel processing with rayon |
| `flash_attention` | Memory-efficient attention mechanism |
| `mixed_precision` | FP16/BF16 inference (placeholder) |
| `gpu_debug` | GPU validation layers and debug markers |

### Platform-Specific Features

| Feature | Platforms | Requirements |
|---------|-----------|--------------|
| `cuda` | Linux, Windows | NVIDIA GPU + CUDA Toolkit (nvcc, nvidia-smi) |
| `metal` | macOS | Apple Silicon or Intel Mac with Metal |
| `accelerate` | macOS | Apple Accelerate framework (enabled by default) |

### Building Documentation

**Important:** Do NOT use `--all-features` on macOS as it will attempt to enable CUDA support which requires Linux/Windows.

```bash
# macOS (Metal supported, CUDA not available)
cargo doc --no-deps --features "simd,parallel,flash_attention,mixed_precision,gpu_debug"

# Linux with NVIDIA GPU
cargo doc --no-deps --features "cuda,simd,parallel,flash_attention,mixed_precision,gpu_debug"

# Linux without GPU (CPU only)
cargo doc --no-deps --features "simd,parallel,flash_attention,mixed_precision"

# Enforce warnings as errors (for CI)
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --features "simd,parallel,flash_attention"
```

### Building with Features

```bash
# macOS optimized build
cargo build --release --features "metal,simd,parallel,flash_attention"

# Linux with CUDA
cargo build --release --features "cuda,simd,parallel,flash_attention"

# CPU-only build (all platforms)
cargo build --release --features "simd,parallel,flash_attention"
```

## FLAME Model Setup

OxiGAF supports both legacy NPY and modern Safetensors formats for FLAME models.

### Option 1: Safetensors (Recommended, v0.1.1)

Safetensors format is supported for runtime loading/saving. PyTorch conversion script coming soon.

1. Download the FLAME 2023 model from <https://flame.is.tue.mpg.de/>
2. Convert using PyTorch and save as safetensors (script in development)

### Option 2: NPY (Legacy)

1. Download the FLAME 2023 model from <https://flame.is.tue.mpg.de/>
2. Convert to `.npy` format:
   ```bash
   python scripts/convert_flame.py path/to/FLAME2023.pkl output_dir/
   ```

## Usage Examples

### Multi-View Diffusion Pipeline (v0.1.1)

```rust
use oxigaf_diffusion::{MultiViewDiffusionPipeline, DiffusionConfig};
use candle_core::Device;
use std::path::Path;

// Configure multi-view generation with classifier-free guidance
let config = DiffusionConfig {
    num_views: 4,
    guidance_scale: 7.5,
    num_inference_steps: 50,
    ..Default::default()
};

// Load the complete pipeline
let device = Device::cuda_if_available(0)?;
let pipeline = MultiViewDiffusionPipeline::load(
    config,
    Path::new("weights/"),
    &device,
)?;

// Generate multi-view images with camera conditioning
let output = pipeline.generate(&input_image, &camera_poses)?;
```

### Video Sequence Processing (v0.1.1)

```rust
use oxigaf_flame::{FlameSequence, FlameParams};
use std::path::Path;

// Load video sequence with LRU caching
let mut sequence = FlameSequence::from_json(Path::new("sequence.json"))?;

// Access frames with automatic caching
let frame_42 = sequence.get_frame(42)?;

// Interpolate between frames
let interpolated = sequence.interpolate(42.5)?;
```

### Safetensors I/O (v0.1.1)

```rust
use oxigaf_flame::{load_flame_model_safetensors, save_flame_model_safetensors};
use std::path::Path;

// Load FLAME model from safetensors
let model = load_flame_model_safetensors(Path::new("flame_model.safetensors"))?;

// Save to safetensors (preserves metadata)
save_flame_model_safetensors(&model, Path::new("output.safetensors"))?;
```

### PyTorch Weight Conversion (v0.1.1)

```rust
use oxigaf_bridge::LayerMapping;

// Create layer mapping for weight conversion
let mut mapping = LayerMapping::new();

// Add custom layer name mappings
mapping.add_custom_mapping(
    "pytorch.layer.weight".to_string(),
    "oxigaf_module_weight".to_string(),
);

// Convert PyTorch layer names to OxiGAF format
let oxigaf_name = mapping.pytorch_to_oxigaf("unet.down_blocks.0.conv.weight")?;
// Result: "down_blocks_0_conv_weight"
```

## Documentation

- **[Design Documents](docs/design/)** - Original architecture and design plans with implementation status
- **[Crate TODOs](crates/)** - Current implementation status in `crates/*/TODO.md` files
- **Individual Crate READMEs** - API documentation in `crates/*/README.md`

For new contributors:
1. Start with [docs/design/IMPLEMENTATION_PLAN.md](docs/design/IMPLEMENTATION_PLAN.md) for the big picture
2. Check module-specific plans in [docs/design/](docs/design/)
3. Review current status in the corresponding `crates/*/TODO.md` file

## Sponsorship

OxiGAF is developed and maintained by **COOLJAPAN OU (Team Kitasan)**.

If you find OxiGAF useful, please consider sponsoring the project to support continued development of the Pure Rust ecosystem.

[![Sponsor](https://img.shields.io/badge/Sponsor-%E2%9D%A4-red?logo=github)](https://github.com/sponsors/cool-japan)

**[https://github.com/sponsors/cool-japan](https://github.com/sponsors/cool-japan)**

Your sponsorship helps us:
- Maintain and improve the COOLJAPAN ecosystem
- Keep the entire ecosystem (OxiBLAS, OxiFFT, SciRS2, etc.) 100% Pure Rust
- Provide long-term support and security updates

## License

Apache-2.0
