# oxigaf-flame

FLAME parametric 3D head model implementation in Pure Rust.

## Overview

This crate implements the [FLAME (Faces Learned with an Articulated Model and Expressions)](https://flame.is.tue.mpg.de/) parametric 3D head model in pure Rust, with no dependencies on Python or C/C++ libraries.

FLAME is a statistical 3D head model that represents shape, expression, and pose variations using a Linear Blend Skinning (LBS) framework. It's widely used in computer vision and graphics for facial animation, avatar creation, and 3D reconstruction.

**v0.1.2 — what's included:**
- Linear Blend Skinning (LBS) forward pass with SIMD/parallel acceleration
- CPU software rasterizer for normal map generation
- Mesh surface sampling for Gaussian initialization
- **Safetensors I/O** — load/save FLAME models in `.safetensors` format (`io_safetensors.rs`)
- **FlameSequence** — video frame processing with LRU caching and temporal interpolation (`sequence.rs`)
- 2,582 tests

## Installation

```toml
[dependencies]
oxigaf-flame = "0.1"
```

## Features

| Feature | Description | Performance Gain |
|---------|-------------|------------------|
| `default` | Standard CPU implementation | Baseline |
| `simd` | SIMD-accelerated operations (requires nightly Rust) | 2-4× faster |
| `parallel` | Parallel batch processing with rayon | Near-linear with cores |
| `npz` | NPZ (compressed NumPy archive) support for `FlameSequence` | — |
| `full` | Enable `parallel` + `npz` — both stable-Rust features. `simd` is intentionally excluded; see `full_nightly` | Combined benefits |
| `full_nightly` | Enable `simd` + `parallel` + `npz` (requires nightly Rust) | Maximum |

### Feature Details

- **`simd`**: Enables SIMD acceleration for:
  - Rodrigues rotation computation (axis-angle to rotation matrix)
  - Blend shape evaluation (weighted sum of deformations)
  - Normal map rendering (vectorized pixel operations)
  - Requires nightly Rust with `portable_simd` feature

- **`parallel`**: Enables parallel processing for:
  - `forward_batch_par()` — parallel mesh generation
  - `compute_normals_batch_par()` — parallel normal computation
  - Scales with CPU core count

- **`npz`**: Enables loading/saving `FlameSequence` data (`sequence.rs`) in
  NPZ format, as an alternative to JSON. Without it, NPZ sequence calls
  return an error naming the feature to enable.

### Example Usage

```toml
# Standard CPU implementation
oxigaf-flame = { version = "0.1" }

# With parallel processing
oxigaf-flame = { version = "0.1", features = ["parallel"] }

# All stable-Rust features: parallel + npz (does NOT require nightly)
oxigaf-flame = { version = "0.1", features = ["full"] }

# Maximum performance: simd + parallel + npz (requires nightly Rust)
oxigaf-flame = { version = "0.1", features = ["full_nightly"] }
```

## Usage

### Basic FLAME Forward Pass

```rust
use oxigaf_flame::{FlameModel, FlameParams};

fn main() -> Result<(), oxigaf_flame::FlameError> {
    // Load FLAME model from directory containing .npy files
    let model = FlameModel::load("path/to/flame/model")?;

    // Create neutral parameters (zero shape, expression, pose)
    let params = FlameParams::neutral();

    // Run forward pass to get posed mesh
    let mesh = model.forward(&params);

    println!("Generated mesh with {} vertices", mesh.vertices.len());
    println!("Mesh has {} faces", mesh.faces.len());

    Ok(())
}
```

### Customize Shape and Expression

```rust
use oxigaf_flame::{FlameModel, FlameParams};

fn main() -> Result<(), oxigaf_flame::FlameError> {
    let model = FlameModel::load("path/to/flame/model")?;

    // Build via `FlameParams::builder()` (there is no `FlameParamsBuilder::new`).
    // `shape`/`expression` take a plain `Vec<f32>`, not an `nalgebra` vector
    // type, and jaw rotation is set with `jaw_rotation` (a single swing
    // angle) or `jaw_rotation_full` ([f32; 3]) — there is no `jaw_pose`.
    let params = FlameParams::builder()
        .shape(vec![0.5, -0.3, 0.2, /* ... */])
        .expression(vec![0.8, 0.0, 0.4, /* ... */])
        .jaw_rotation(0.1) // Open jaw slightly
        .build();

    let mesh = model.forward(&params);

    // Access mesh data
    for vertex in mesh.vertices.iter().take(5) {
        println!("Vertex: ({:.3}, {:.3}, {:.3})", vertex[0], vertex[1], vertex[2]);
    }

    Ok(())
}
```

### Safetensors I/O (v0.1.1)

```rust
// Both functions live in the `io_safetensors` submodule — they are not
// re-exported at the crate root.
use oxigaf_flame::io_safetensors::{load_flame_model_safetensors, save_flame_model_safetensors};
use std::path::Path;

fn main() -> Result<(), oxigaf_flame::FlameError> {
    // Load FLAME model from safetensors
    let model = load_flame_model_safetensors(Path::new("flame_model.safetensors"))?;

    // Save to safetensors. The third argument is optional string metadata
    // (`Option<&HashMap<String, String>>`) embedded in the file; `None`
    // skips it.
    save_flame_model_safetensors(&model, Path::new("output.safetensors"), None)?;

    Ok(())
}
```

### Video Sequence Processing (v0.1.1)

```rust
use oxigaf_flame::FlameSequence;
use std::path::Path;

fn main() -> Result<(), oxigaf_flame::FlameError> {
    // Load video sequence with LRU caching
    let mut sequence = FlameSequence::from_json(Path::new("sequence.json"))?;

    println!("Sequence has {} frames", sequence.num_frames());

    // Access frames with automatic caching. `get_frame` returns `&FlameParams`
    // borrowed from `sequence`, so clone it before the next `&mut sequence`
    // call (`interpolate`) — holding the borrow across that call does not
    // borrow-check.
    let frame_42 = sequence.get_frame(42)?.clone();

    // Interpolate between frames
    let interpolated = sequence.interpolate(42.5)?;
    let _ = (frame_42, interpolated);

    Ok(())
}
```

### Render Normal Maps

```rust
use oxigaf_flame::{FlameModel, FlameParams, NormalMapRenderer, Camera};

fn main() -> Result<(), oxigaf_flame::FlameError> {
    let model = FlameModel::load("path/to/flame/model")?;
    let params = FlameParams::neutral();
    let mesh = model.forward(&params);

    // `Camera` models real intrinsics/extrinsics (rotation, translation,
    // focal_x/focal_y, cx/cy, width, height, near, far) rather than a
    // look-at triple; `default_front` builds a ready-made front-facing one
    // sized for the output image (see its doc comment for the derivation
    // of a FLAME-mesh-appropriate rotation/translation).
    let camera = Camera::default_front(512, 512);

    // Render the normal map. `NormalMapRenderer` is a stateless renderer —
    // `render` is an associated function (`Type::render(...)`), not a
    // method on an instance — and returns the `image::RgbImage` directly
    // rather than a `Result`.
    let normal_map = NormalMapRenderer::render(&mesh, &camera);

    // Save to file. `FlameError::Io` wraps a real `std::io::Error`, not a
    // `String` — `std::io::Error::other` adapts `image`'s error into one.
    normal_map.save("normal_map.png").map_err(|e| {
        oxigaf_flame::FlameError::Io(std::io::Error::other(e))
    })?;

    Ok(())
}
```

### Sample Mesh Surface for Gaussian Initialization

This example also uses `rand` directly (for the sampler's RNG argument) —
add it to your own `[dependencies]` to run the snippet as-is.

```rust
use oxigaf_flame::{FlameModel, FlameParams, sample_mesh_surface};
use rand::SeedableRng;

fn main() -> Result<(), oxigaf_flame::FlameError> {
    let model = FlameModel::load("path/to/flame/model")?;
    let params = FlameParams::neutral();
    let mesh = model.forward(&params);

    // Sample 10,000 points uniformly on the mesh surface. `sample_mesh_surface`
    // takes the RNG explicitly (for reproducibility) and returns a `Result`:
    // malformed topology (a `normals` length that doesn't match `vertices`, or
    // an out-of-range face index) is a `FlameError::InvalidParams`. A
    // well-formed but empty input — no faces, `count == 0`, or a fully
    // collapsed zero-area mesh — succeeds with an empty `Vec` instead.
    let num_points = 10000;
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let surface_points = sample_mesh_surface(&mesh, num_points, &mut rng)?;

    println!("Sampled {} points", surface_points.len());

    for point in surface_points.iter().take(5) {
        println!(
            "Position: ({:.3}, {:.3}, {:.3}), Normal: ({:.3}, {:.3}, {:.3})",
            point.position[0], point.position[1], point.position[2],
            point.normal[0], point.normal[1], point.normal[2]
        );
    }

    Ok(())
}
```

### Batch Processing with Parallel Feature

```rust
use oxigaf_flame::{FlameModel, FlameParams};

#[cfg(feature = "parallel")]
fn main() -> Result<(), oxigaf_flame::FlameError> {
    let model = FlameModel::load("path/to/flame/model")?;

    // Create batch of parameters
    let params_batch: Vec<FlameParams> = (0..100)
        .map(|_| FlameParams::neutral())
        .collect();

    // Process batch in parallel (automatically uses rayon). Returns the
    // `Vec<Mesh>` directly, not a `Result`.
    let meshes = model.forward_batch_par(&params_batch);

    println!("Generated {} meshes in parallel", meshes.len());

    Ok(())
}

#[cfg(not(feature = "parallel"))]
fn main() {
    println!("This example requires the 'parallel' feature");
}
```

## FLAME Parameters

The FLAME model is controlled by:

- **Shape parameters (β)**: Control identity-specific features (typically 100-300 coefficients)
  - Examples: face width, nose size, overall head shape

- **Expression parameters (ψ)**: Control facial expressions (typically 50-100 coefficients)
  - Examples: smile, frown, raised eyebrows, mouth open

- **Pose parameters (θ)**: Control joint rotations (5 joints × 3 = 15 values)
  - Root rotation (global head orientation)
  - Neck rotation
  - Jaw rotation (open/close mouth)
  - Left eye rotation
  - Right eye rotation

- **Translation**: Global 3D translation applied after posing

## Coordinate System

FLAME uses a **right-handed coordinate system**:

- **+X**: Right (from the subject's perspective)
- **+Y**: Up
- **+Z**: Forward (out of the face)

Rotations are specified as **axis-angle** vectors and converted to rotation matrices using [Rodrigues' formula](https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula).

## Performance

The LBS forward pass is optimized for real-time performance:

- **~1-2ms** for standard FLAME mesh (5023 vertices) on modern CPUs
- **3-4× faster** with `simd` feature (nightly Rust required)
- **Near-linear scaling** with `parallel` feature on multi-core CPUs

Run benchmarks with:

```bash
cargo bench -p oxigaf-flame
```

## Statistics

- **Tests**: 2,582 `#[test]`-attributed tests in source (`src/` + `tests/`) — the count that actually compiles and runs depends on enabled features and toolchain (e.g. `tests/simd_tests.rs` is `nightly`-gated and contributes 0 on a stable compiler)
- **Benchmark files**: 5 (`flame_bench`, `lbs_forward`, `rodrigues`, `normal_map`, `simd_ops`)
- **Key source files**: `model.rs`, `sequence.rs`, `normal_map.rs`, `io_safetensors.rs`, `io.rs`

## Documentation

- [API Documentation](https://docs.rs/oxigaf-flame)
- [FLAME Paper](https://ps.is.tuebingen.mpg.de/uploads_file/attachment/attachment/400/paper.pdf)
- [FLAME Model](https://flame.is.tue.mpg.de/)
- [Repository](https://github.com/cool-japan/oxigaf)

## License

Licensed under the Apache License, Version 2.0 ([LICENSE](https://github.com/cool-japan/oxigaf/blob/master/LICENSE))
