# oxigaf-flame

FLAME parametric 3D head model implementation in Pure Rust.

**Status: Stable.** The core LBS forward pass, CPU rendering, mesh sampling,
model I/O and the broader avatar-authoring toolkit below (rigging,
expressions, mesh processing, geodesics, gaze control, UV/texture, pose
fitting) are fully implemented and covered by 2,589 passing tests. GPU
rendering, WASM builds, and a couple of narrower items are explicitly not
implemented in this crate — see [TODO.md](TODO.md) for the honest, itemized
breakdown.

## Overview

This crate implements the [FLAME (Faces Learned with an Articulated Model and Expressions)](https://flame.is.tue.mpg.de/) parametric 3D head model in pure Rust, with no dependencies on Python or C/C++ libraries.

FLAME is a statistical 3D head model that represents shape, expression, and pose variations using a Linear Blend Skinning (LBS) framework. It's widely used in computer vision and graphics for facial animation, avatar creation, and 3D reconstruction.

**v0.1.2 — what's included:**
- Linear Blend Skinning (LBS) forward pass with SIMD/parallel acceleration
- CPU software rasterizer for normal map generation
- Mesh surface sampling for Gaussian initialization
- **Safetensors I/O** — load/save FLAME models in `.safetensors` format (`io_safetensors.rs`)
- **FlameSequence** — video frame processing with LRU caching and temporal interpolation (`sequence.rs`)
- A much larger avatar-authoring toolkit beyond the core forward pass —
  rigging, an expression system, mesh processing, UV/texture tools,
  geodesic/spectral geometry, pose fitting, and more — see
  [Module Overview](#module-overview) for the full, real surface
- **New in 0.1.2**: a real heat-method `heat_geodesic`
  (Crane, Weischedel & Wardetzky 2013), replacing what its own 0.1.1 doc
  comment called a simplified approximation — plus `gaze_controller`
  (Listing's-law gaze rotation, I-VT saccade/fixation detection, natural
  blink synthesis, vergence estimation), a comprehensive gaze system that
  has been part of the crate since v0.1.1 and gained one new convenience
  method (`GazeController::synthesize_blinks`) this release
- 2,589 tests passing

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
| `npz` | Load `FlameSequence` data from NPZ (compressed NumPy archive) files | — |
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

- **`npz`**: Enables `FlameSequence::from_npz`, loading sequence data
  (`sequence.rs`) from a NPZ archive as an alternative to JSON. This is a
  reader only — there is no NPZ writer. Without the feature, `from_npz`
  returns an error naming the feature to enable.

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
// Both functions are re-exported at the crate root (`oxigaf_flame::{load_flame_model_safetensors, ...}`);
// this example uses the `io_safetensors` submodule path, which also works.
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

### Gaze Control

```rust
use oxigaf_flame::{GazeController, GazeControllerConfig, GazeDirection, GazeFrame};

fn main() -> Result<(), oxigaf_flame::GazeControllerError> {
    // `GazeControllerConfig::default()` uses 60 fps, a 30 deg/s I-VT
    // saccade threshold, and standard human blink-rate/duration defaults.
    let mut controller = GazeController::new(GazeControllerConfig::default())?;

    // Feed monocular gaze samples (same direction for both eyes) at 60 fps.
    for step in 0..30u64 {
        let azimuth = 0.02 * step as f32; // slow rightward gaze drift
        let gaze = GazeDirection::new(azimuth, 0.0, 0.0);
        let timestamp_ms = step as f64 * (1000.0 / 60.0);
        controller.push_frame(GazeFrame::monocular(step, gaze, timestamp_ms));
    }

    // Classify the history into saccades/fixations (I-VT) and detect blinks.
    controller.update_events();

    println!(
        "{} saccade(s), {} fixation(s)",
        controller.saccades().len(),
        controller.fixations().len()
    );

    Ok(())
}
```

### Geodesic Distance (heat method, new in v0.1.2)

`heat_geodesic` now solves the real heat method of Crane, Weischedel &
Wardetzky (2013) — `(M + t·Lc)u = δ_source`, gradient normalization, then a
Poisson solve — rather than the simplified approximation used before v0.1.2.

```rust
use oxigaf_flame::{GeodesicError, GeodesicMesh, heat_geodesic, heat_time_step};

fn main() -> Result<(), GeodesicError> {
    // A unit square split into two triangles (this is a `GeodesicMesh` —
    // plain vertex/face arrays for geometry processing — not a `Mesh`).
    let mesh = GeodesicMesh::new(
        vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        vec![[0, 1, 2], [0, 2, 3]],
    )?;

    // `heat_time_step` gives the standard `dt` heuristic (squared mean edge
    // length). The CG solver never errors on a slow-converging system — it
    // returns its best iterate if `n_iter` is reached first, so too small a
    // budget degrades accuracy rather than failing; 100 iterations converges
    // this 4-vertex problem in exact arithmetic within a handful of steps.
    let dt = heat_time_step(&mesh);
    let field = heat_geodesic(&mesh, 0, 100, dt)?;

    for (vertex, distance) in field.distances.iter().enumerate() {
        println!("vertex {vertex}: distance {distance:.3} from vertex 0");
    }

    Ok(())
}
```

## Module Overview

Beyond the core LBS forward pass, `oxigaf-flame` includes a broad
avatar-authoring toolkit. Every item below is a real, implemented,
tested module — this table is a map of the crate, not a roadmap.

| Area | Modules | Highlights |
|------|---------|------------|
| Core model & I/O | `model`, `params`, `io`, `io_safetensors`, `sequence`, `conversion` | LBS forward pass, `.npy`/`.safetensors` loading, `FlameSequence` video playback, `.pkl`→`.npy`/`.safetensors` conversion |
| Rendering & sampling | `normal_map`, `sampler`, `visualize`, `gpu_buffers` | CPU normal-map rasterizer, Gaussian-init surface sampling, SVG wireframe preview, padded GPU vertex/normal/face buffers |
| Mesh processing | `mesh`, `mesh_ops`, `mesh_repair`, `mesh_smoothing`, `mesh_subdivision`, `mesh_morphing`, `mesh_analysis`, `multiresolution` | Boolean/clip/split/merge, hole filling, Laplacian/bilateral smoothing, Loop/Catmull-Clark subdivision, morph targets, area/volume/curvature metrics, edge-collapse decimation |
| Geometry & spectral analysis | `geodesic`, `spectral_analysis`, `symmetry`, `statistical_shape_model`, `shape_analysis` | Dijkstra and heat-method (Crane et al.) geodesics, Laplace-Beltrami spectral decomposition, bilateral symmetry detection, PCA shape space, shape-space distance/outlier statistics |
| Rigging & motion | `avatar_rig`, `timeline`, `warp_field`, `retargeting`, `expression_retargeting`, `rigid_alignment`, `canonical` | Skeleton-mesh binding, frame-indexed parameter timelines, per-vertex warp fields, identity-shape retargeting, cross-identity **expression** retargeting (a distinct, linear-mapper module — see below), Procrustes alignment |
| Expression system | `expressions`, `expression_animation`, `expression_clustering`, `expression_transfer`, `facs`, `emotion_recognition`, `phoneme_animation` | Named expression presets + blending (placeholder coefficients — see caveat below), keyframe animation, k-means clustering, cross-identity transfer, FACS Action Units, emotion classification, phoneme-driven lip-sync |
| Landmarks & tracking | `landmarks`, `dynamic_landmarks`, `head_tracker`, `head_geometry`, `vertex_mask`, `visibility_culling` | 68-point static + pose-dependent contour landmarks, head-pose trajectory smoothing, region segmentation, back-face/frustum culling |
| Pose, fitting & gaze | `pose_estimation`, `pose_prior`, `fitting`, `gaze_controller` | Weak-perspective pose solving, a learned pose prior, landmark-to-FLAME fitting, gaze direction/saccade/fixation/blink modeling |
| UV & texture | `uv`, `uv_texture`, `face_atlas`, `texture_baking`, `albedo_map`, `lighting_model` | LSCM UV parameterisation, UV-coordinate texture sampling, UV atlas packing, render-to-texture baking, albedo extraction, Phong/spherical-harmonic lighting |
| Utilities | `param_sampler`, `blend_shape_solver`, `depth_estimation`, `contact_detection`, `face_normalization`, `traits` | Parameter-space sampling, least-squares blend-shape solving, monocular depth estimation, mesh self-intersection detection, face crop/alignment, `oxigaf-diffusion` integration traits |

Two notes on scope, so the table above isn't read as overselling:

- **`expressions::ExpressionLibrary::placeholder_expressions()`** (the
  named presets — "smile", "frown", "surprised", etc.) returns
  hand-authored, illustrative coefficients for exercising the blending/
  animation machinery. They are **not fitted** to any FLAME expression
  basis and will not reproduce the named expression on a real model; for
  real coefficients, fit them with `NamedExpression::fit_to_basis` against
  the model you use. This is why `default_expressions()` — the pre-0.1.2
  name for the same data — is now `#[deprecated]` in favor of the more
  honest `placeholder_expressions()` name.
- **`expression_retargeting`** (`LinearExpressionRetargeter`) and
  **`retargeting`** (`ExpressionRetargeter`) are two distinct modules, not
  a duplicate: `retargeting` transfers FLAME **shape** identity, while
  `expression_retargeting` learns a linear map between two identities'
  **expression** spaces.

Not implemented in this crate: GPU-accelerated rendering/LBS (wgpu compute
shaders — that lives in `oxigaf-render`), WASM builds, and loading FLAME's
official `texture_data_*.npy` texture files directly (the generic
`uv_texture`/`TextureMap` sampling infrastructure exists; the FLAME-specific
loader does not). See [TODO.md](TODO.md) for the complete list.

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
- **2-4× faster** with `simd` feature (requires nightly Rust; not exercised
  by this crate's own tests on a stable toolchain — see `full_nightly`)
- **Near-linear scaling** with `parallel` feature on multi-core CPUs

Run benchmarks with:

```bash
cargo bench -p oxigaf-flame
```

## Statistics

- **Tests**: 2,589 passed, 0 failed via
  `cargo nextest run -p oxigaf-flame --all-features` (2,581 with default
  features — the difference is `parallel`/`npz`-gated tests; `tests/simd_tests.rs`
  needs both the `simd` feature and a `nightly` compiler, so it contributes 0
  on a stable toolchain either way). Doctests run separately and aren't
  included in this count: 39 passed via `cargo test --doc -p oxigaf-flame --all-features`.
- **Benchmark files**: 5 (`flame_bench`, `lbs_forward`, `rodrigues`, `normal_map`, `simd_ops`)
- **Key source files**: `model.rs`, `sequence.rs`, `normal_map.rs`, `io_safetensors.rs`, `io.rs`, `geodesic/`, `gaze_controller/`

## Documentation

- [API Documentation](https://docs.rs/oxigaf-flame)
- [FLAME Paper](https://ps.is.tuebingen.mpg.de/uploads_file/attachment/attachment/400/paper.pdf)
- [FLAME Model](https://flame.is.tue.mpg.de/)
- [Repository](https://github.com/cool-japan/oxigaf)

## License

Licensed under the Apache License, Version 2.0 ([LICENSE](https://github.com/cool-japan/oxigaf/blob/master/LICENSE))
