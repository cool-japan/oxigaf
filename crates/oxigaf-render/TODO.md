# TODO for oxigaf-render

## ✅ Completed (from plan)

### Core 3DGS Forward Pipeline
- ✅ **Preprocess shader** (`preprocess.wgsl`)
  - Frustum culling (discard Gaussians outside view)
  - 3D → 2D projection (perspective transform)
  - Covariance projection (J·Σ·J^T for 2D Gaussian)
  - EWA low-pass filter (anti-aliasing)
  - Conic computation (inverse 2D covariance)
  - Screen-space bounding radius
  - Per-Gaussian tile count
- ✅ **Specialized SH shaders** (major optimization beyond plan)
  - `preprocess_sh0.wgsl` - Degree 0 (1 coefficient)
  - `preprocess_sh1.wgsl` - Degree 1 (4 coefficients)
  - `preprocess_sh2.wgsl` - Degree 2 (9 coefficients)
  - `preprocess_sh3.wgsl` - Degree 3 (16 coefficients)
  - **10× speedup** through compile-time degree specialization
- ✅ **Prefix sum shader** (`prefix_sum.wgsl`, `prefix_sum_add.wgsl`)
  - Work-efficient Blelloch scan
  - Inclusive prefix sum over tile counts
- ✅ **GPU radix sort** (`radix_histogram.wgsl`, `radix_scatter.wgsl`, driven from `sort.rs`)
  - 64-bit keys (tile_id << 32 | depth)
  - Fuchsia-based radix sort (adapted from web-splat)
  - Multi-pass sorting (histogram → prefix → scatter)
  - 0.1.2: `RadixSorter::sort` no longer takes `device`/`keys`/`values` — that setup moved to a new `prepare()` step; `capacity()` now reports the sorter's live, growable buffer capacity instead of a fixed construction-time value
- ✅ **Tile assignment** (`tile_assign.wgsl`)
  - Generate (tile_id, depth) keys for each Gaussian-tile intersection
  - Write to sort buffer
- ✅ **Tile ranges** (`tile_ranges.wgsl`)
  - Find start/end indices per tile in sorted array
  - Enables efficient per-tile rasterization
- ✅ **Forward rasterization** (`rasterize_fwd.wgsl`)
  - Tile-based alpha-blended rasterization (16×16 tiles)
  - Per-pixel front-to-back traversal
  - Early termination when T < threshold
  - Output: color, depth, final transmittance

### Backward Pass (Differentiability)
- ✅ **Backward rasterization** (`rasterize_bwd.wgsl`)
  - Reverse-order tile traversal
  - ∂L/∂color, ∂L/∂alpha, ∂L/∂conic, ∂L/∂mean2D
  - CAS-loop f32 atomicAdd for gradient accumulation
  - Background's contribution to ∂L/∂alpha through final transmittance (0.1.2 fix — previously missing, biased training against any non-black background)
- ✅ **Backward preprocess** (`preprocess_bwd.wgsl`)
  - ∂L/∂cov3D → ∂L/∂scale, ∂L/∂rotation
  - ∂L/∂color → ∂L/∂SH coefficients
  - ∂L/∂mean2D → ∂L/∂mean3D (projection chain rule)
  - ∂L/∂color → ∂L/∂dir → ∂L/∂mean3D through view-dependent SH color (0.1.2 fix — previously missing for every `sh_degree >= 1` model)
  - Cull guard against NaN gradients for Gaussians rejected by the near/far test (0.1.2 fix)

> ⚠️ **0.1.1 → 0.1.2 correctness fixes.** Two bugs shipped in 0.1.1 and affected **every** training run that used these shaders, unconditionally:
> 1. `rasterize_bwd.wgsl`'s backward tile kernel could accumulate a whole tile's gradient sum onto the *wrong* Gaussian. Its reverse-traversal loop bound was read per-pixel from `out_n_contrib`, so different threads in the same 16×16 workgroup ran the loop body — which contains `workgroupBarrier()` calls — a different number of times. That is non-uniform control flow around a barrier, which WGSL requires to be uniform. Fixed by computing a single workgroup-uniform loop bound (`tile_end`, via `workgroupUniformLoad`) with per-thread validity expressed as a `contributes` mask instead of a per-thread trip count.
> 2. `preprocess_bwd.wgsl` omitted the position gradient through view-dependent SH color for every `sh_degree >= 1` model — only the projection path (`∂L/∂mean2D → ∂L/∂mean3D`) was differentiated, not the `∂L/∂color → ∂L/∂dir → ∂L/∂pos` path the forward pass's `dir = normalize(pos - cam_pos)` requires.
>
> **Retraining is the only remedy for both** — there is no way to recover the missing gradient signal from an already-trained model. See `CHANGELOG.md`'s `[0.1.2]` migration notes for full detail, including why the existing finite-difference test suite didn't catch either bug (a harness bug in `gpu_gradient_verify.rs` reported NaN/Infinity/empty comparisons as a "clean" `0.0` error — see Testing & Validation below).

### FLAME Mesh Binding
- ✅ **Gaussian-to-mesh binding** (`binding.rs`)
  - Rigid/flexible Gaussian split
  - Barycentric coordinate binding
  - Local offset support (learnable for flexible)
  - TBN matrix computation
- ✅ **Deformation shader** (`deform_gaussians.wgsl` + `src/deform.rs`)
  - Transform Gaussians with FLAME mesh updates
  - Per-frame mesh vertex upload
  - Position and rotation updates

### GPU Infrastructure
- ✅ **Rasterizer orchestration** (`rasterizer/` module: `mod.rs`, `bind_groups.rs`, `limits.rs` — split out of a single `rasterizer.rs` as the module grew)
  - Forward and backward dispatch
  - Buffer management
  - Pipeline state tracking
  - 0.1.2: `limits.rs` adds device-limit validation (`rasterizer_device_limits`, see Debugging & Visualization below)
- ✅ **Buffer pool** (`pool.rs`, 580 lines)
  - Memory-efficient buffer reuse
  - Automatic resizing for dynamic Gaussian counts
  - Statistics tracking (allocations, reuses, cache hits)
- ✅ **Buffer management** (`buffers.rs`, 539 lines)
  - Typed buffer wrappers
  - Upload/download helpers
  - Staging buffers
- ✅ **Pipeline compilation** (`pipeline.rs`, 402 lines)
  - Compute pipeline setup for all shaders
  - Bind group layouts
  - Shader module loading
  - Feature-based compilation (gpu_debug)
- ✅ **Configuration** (`config.rs`, 572 lines)
  - `RasterConfig` with tile size, background, clipping planes
  - `RenderCamera` with view/proj matrices
  - Serialization support

### Error Handling
- ✅ Comprehensive `RenderError` enum
- ✅ GPU initialization errors
- ✅ Device lost handling
- ✅ Shader compilation errors
- ✅ Buffer errors (upload/download/mapping)
- ✅ Invalid parameter errors
- ✅ NaN/Inf detection

### Testing & Validation
- ✅ 2,903 `#[test]`-attributed tests total (v0.1.2), all passing:
  - 2,887 run by default (`cargo nextest run -p oxigaf-render --all-features`)
  - 16 more require real GPU hardware and are `#[ignore]`d (4 `deform`, 5 `multi_view`, 4 `cpu_gpu_compare`, 1 `pipeline`, 1 `sort`, plus 1 slow 100-Gaussian position-gradient check); run with `--run-ignored ignored-only` — all 16 pass when a GPU adapter is present
  - 23 doc-tests passing, 3 `#[ignore]`d
- ✅ **78 gradient verification tests** (`gpu_gradient_verify.rs` + `tests/gpu_gradient_verify/sh.rs` and `tests/gradient_verification/`)
  - Finite-difference vs analytical gradients, median-relative-error metric: ≤5e-2 for most parameters, ≤2.5e-1 for position (tile-boundary discontinuities in the forward pass need a wider tolerance)
  - All parameters: position, rotation, scale, opacity, SH coefficients
  - FLAME binding backward (mesh-bound ∂L/∂local_offset)
  - Includes the harness's own NaN/empty-input guard unit tests, added in 0.1.2 after the bug described above let a non-finite or empty comparison report a "clean" `0.0` error
  - Most of these self-skip at runtime (not via `#[ignore]`) when no GPU adapter is available, rather than being permanently ignored — see `tests/gradient_verification/mod.rs`
- ✅ Shape preservation tests
- ✅ Buffer pool tests
- ✅ Configuration tests
- ✅ Camera tests

### Benchmarking
- ✅ `render_bench.rs` - Comprehensive rendering benchmarks
- ✅ Forward pass timing
- ✅ Backward pass timing
- ✅ Memory pool efficiency

### Code Quality
- ✅ No unwrap policy (verified: no `.unwrap()`/`.expect()` outside doc-comment examples and `#[cfg(test)]` code)
- ✅ No expect in library code
- ✅ Largest file is 1,977 lines (`depth_of_field.rs`) — within the 2,000-line policy, though several files are close to it (`rasterizer/mod.rs` 1,926, `lens_distortion.rs` 1,905, `lod.rs`/`deform.rs` 1,884, `gaussian.rs` 1,845) and worth watching for a future split
- ✅ Total: ~96,980 lines across 132 `.rs` files (77,937 code lines per `tokei`) + 4,296 lines across 17 `.wgsl` shaders
- ✅ Clean module boundaries

### Post-Processing Pipeline (v0.1.1)
- ✅ **Ambient occlusion** (SSAO) — screen-space ambient occlusion
- ✅ **Bloom** — HDR bloom with configurable threshold
- ✅ **Denoising** — image denoising filter
- ✅ **Depth-of-field** — bokeh depth-of-field effect
- ✅ **Motion blur** — velocity-based motion blur
- ✅ **Film grain** — procedural film grain noise
- ✅ **Image sharpening** — unsharp mask and clarity filters
- ✅ **Chromatic aberration** — lateral colour fringing
- ✅ **Vignetting** — edge darkening effect
- ✅ **Lens distortion** — barrel/pincushion distortion correction
- ✅ **Temporal anti-aliasing (TAA)** — history-based TAA
- ✅ **Exposure control** — auto-exposure and manual EV
- ✅ **HDR tone mapping** — multiple tone mapping operators
- ✅ **Tone curve** — custom S-curve colour grading
- ✅ **Color grading** — lift/gamma/gain colour grading
- ✅ **Colorspace conversion** — sRGB/linear/wide-gamut conversions
- ✅ **Color calibration** — colour target calibration
- ✅ **Subsurface scattering** — skin-like SSS approximation
- ✅ **Edge detection** — Sobel/Canny edge maps

### Scene Composition & Rendering (v0.1.1)
- ✅ **Image compositor** — layer-based image composition
- ✅ **Scene compositor** — multi-layer scene assembly
- ✅ **Render graph** — dependency-tracked render pass graph
- ✅ **Silhouette extraction** — foreground silhouette masks
- ✅ **Background synthesis** — procedural background generation
- ✅ **Stereo rendering** — side-by-side and top-bottom stereo output
- ✅ **Panoramic projection** — equirectangular camera model
- ✅ **Camera path interpolation** — smooth camera trajectory interpolation
- ✅ **Multi-view rendering** — render from multiple cameras simultaneously

### Volumetric Rendering (v0.1.1)
- ✅ **Ray types** — parameterised ray representation
- ✅ **Camera model** — ray-generation camera model
- ✅ **Volume grid** — 3D voxel grid data structure
- ✅ **Ray-march result traits** — integration result abstraction

### Spatial Acceleration & Analysis (v0.1.1)
- ✅ **BVH** — bounding volume hierarchy for ray-scene queries
- ✅ **LOD generator** — level-of-detail simplification
- ✅ **Gaussian culling** — view-frustum and occlusion culling
- ✅ **Normal estimation** — per-point normal estimation
- ✅ **Depth map** — depth buffer utilities
- ✅ **Tile statistics** — per-tile occupancy statistics

### GPU Tooling & Utilities (v0.1.1)
- ✅ **Workgroup size utilities** — optimal workgroup size selection
- ✅ **Debug readback** — GPU → CPU buffer readback for debugging
- ✅ **GPU profiler** — per-pass GPU timing
- ✅ **Render metrics** — frame time, throughput, memory stats
- ✅ **Device init helpers** — adapter selection and feature negotiation

### Interactive & Advanced (v0.1.1)
- ✅ **MIP splatting** — MIP-mapped Gaussian splatting
- ✅ **Gaussian picking** — interactive mouse-click Gaussian selection
- ✅ **Mesh compression** — indexed mesh size reduction
- ✅ **Model pruning** — opacity-threshold Gaussian removal
- ✅ **Gaussian deformation** — per-Gaussian deformation fields
- ✅ **Density estimation** — Gaussian density volume sampling
- ✅ **Antialiasing module** — MSAA and FXAA alternatives

## 🚧 In Progress

Currently none.

## 📋 Completed (added in v0.1.0, previously Missing)

### Gradient Verification ✅
- ✅ **Finite-difference gradient checks** (`gpu_gradient_verify.rs` + `tests/gpu_gradient_verify/sh.rs` and `tests/gradient_verification/`)
  - All parameters verified: positions, rotations, scales, opacities, SH
  - FLAME binding backward: ∂L/∂local_offset for mesh-bound Gaussians
  - Median-relative-error tolerance: ≤5e-2 for most parameters, ≤2.5e-1 for position (78 tests total, all passing)
  - 0.1.2: fixed a harness bug where a NaN, `Infinity`, or empty error set was reported as a "clean" `0.0` — this is why the two backward-shader gradient bugs fixed this release survived the 0.1.1 suite despite this section's own "DONE"

### FLAME Binding Backward ✅
- ✅ **Backward pass through FLAME mesh binding** (`preprocess_bwd.wgsl`)
  - ∂L/∂position → ∂L/∂offsets for mesh-bound Gaussians
  - ∂L/∂world_rotation → ∂L/∂local_rotation via TBN matrix

## 📋 Planned (future versions)

### Backward Pass Refinement
- ✅ **cov2d_backward.rs** — CPU-only reference/test oracle for the 2D-covariance backward pass, 10 tests incl. a finite-difference gradient check; `shaders/cov2d_bwd.wgsl` documents the backward math but, like the CPU reference, is not compiled/dispatched as a GPU pipeline — the live GPU path stays inline in `preprocess_bwd.wgsl`
- ⬜ **Cross-validation with gsplat (Python)**
  - Layer-by-layer gradient comparison
  - Save test data in `tests/reference/`

### Gaussian Operations
- ✅ **Adaptive density control** (`density.rs`, 552 lines)
  - `DensityConfig`, `GradientAccumulator`, `DensityController`
  - Clone (high-grad, small scale), Split (high-grad, large scale, scale×0.618 golden ratio), Prune (low opacity or large screen size)
  - `reset_opacity`, `sync_to_model`
  - Scale comparisons use exp() (log-scale storage), opacity comparisons use sigmoid() (logit-space storage)
  - 25 new tests
- ✅ **Initialization utilities** (`init.rs` — Gaussian init on FLAME mesh surface)
- ✅ **PLY I/O**
  - Load Gaussians from standard 3DGS .ply format
  - Save Gaussians to .ply (for visualization)
  - Support for SH coefficients up to degree 3
  - Binary little-endian, compatible with SIBR viewer and 3DGS tools
  - 8 unit tests (roundtrip, empty, sh_degree 0/1/2, rotation convention, large)
- ✅ **SafeTensors I/O**
  - `save_safetensors()` and `load_safetensors()` storing all fields: positions, rotations, scales, opacities, sh_coeffs, face_indices, barycentric, local_offsets, is_rigid
  - Metadata: sh_degree, num_gaussians
  - 10 new tests added
- ✅ **glTF export** (0.1.2) — new `gltf` module: `write_gltf`/`GltfError`, `EXTENSION_NAME = "OXIGAF_gaussian_splat"`. The single, spec-conformant glTF 2.0 writer, consolidating what were three independently-written, mutually-incompatible glTF emitters in the workspace (this crate had none before; `oxigaf-cli` had two). One buffer view per accessor, mandatory `min`/`max` on the `POSITION` accessor, asset-only document for an empty model. Only `POSITION` is a standard glTF mesh attribute; rotation/scale/opacity/SH each get their own accessor referenced by index from the `OXIGAF_gaussian_splat` node extension instead, since glTF has no standard per-vertex semantic for them

### Performance Optimization
- ✅ **Occupancy tuning / Adaptive workgroup size** — `workgroup.rs` (388 lines). `WorkgroupSize` (x/y/z, linear/square/new, total(), dispatch_count_x/y). `WorkgroupProfile` (Mobile=32, Balanced=64, HighThroughput=256, Custom) with default_size/name/description. `WorkgroupConfig` (per-pass: preprocess/sort/rasterize/backward/tile) with from_profile/mobile/balanced/high_throughput/adaptive(num_gaussians)/validate/Default. `WorkgroupBenchResult` (mean_duration_us/min/samples). `WorkgroupBenchmarker` (with_candidates/with_warmup/with_measure/benchmark/best_of/recommend). 34 new tests.
  - 0.1.2: `from_profile`'s tile dimensions changed from 4×4 to 16×16 (`WorkgroupConfig::SHIPPED_TILE`) to match the new `RASTERIZE_TILE_SIZE` constant, since `Rasterizer::from_device` now hard-errors on any other tile size instead of silently accepting it
- ⬜ **Shared memory optimization**
  - Tile-local Gaussian data caching
  - Reduce global memory bandwidth
- ⬜ **Early culling improvements**
  - Tighter bounding boxes
  - Hierarchical culling (BVH)
  - Occlusion culling
- ✅ **Anti-aliasing improvements** (`antialiasing.rs`, `mip_splatting.rs`, `temporal_aa.rs`)

### Multi-View Rendering
- ✅ **Batch multi-view rendering** (`multi_view.rs`)
  - `MultiViewRenderer` + `MultiViewConfig`
  - `render_views()`, `render_views_stacked()`, `render_turntable()` (orbital convenience wrapper with evenly-spaced horizontal circle cameras)
  - `from_rasterizer()` for sharing GPU setup
  - sRGB gamma conversion
  - 22 new CPU-side tests (6 GPU tests `#[ignore]`)

### Integration Features
- ⬜ **burn-autodiff custom op** — genuinely missing; no `burn` dependency exists anywhere in the workspace (`grep -rn '^burn' */Cargo.toml` is empty), and the cross-framework tensor path actually in use goes through `oxigaf-bridge`'s torsh-core/torsh-tensor/torsh-nn stack instead (bumped 0.1.2 → 0.2.0 this release). This item may be superseded rather than pending — see "Priority for v1.0" below
- ✅ **Mesh binding shader forward**
  - Dedicated `deform_gaussians.wgsl` + `src/deform.rs` (`DeformPipeline`)
  - TBN matrix from triangle geometry
  - Barycentric interpolation of position/normal
  - Local→world coordinate transform
  - Quaternion composition with TBN rotation
  - 12 new CPU-side tests

### Debugging & Visualization
- ✅ **Intermediate buffer readback**
  - `debug_readback.rs` (386 lines): `RasterizationSnapshot` (tile counts, depth range, screen sizes, stats), `RasterizationStats` (visible/overflow/empty tiles, mean/max occupancy)
  - `DebugReadbackBuilder::compute_snapshot()` with AABB tile assignment
  - Visualization: `tile_occupancy_image()`, `hotspot_tiles()`, `tile_for_pixel()`
  - 29 new tests; total: 126 unit tests passing
- ✅ **GPU profiler integration**
  - `profiler.rs`: `PassProfiler` (thread-safe with Mutex+AtomicU64), `PassStats` (count/total/min/max/EMA α=0.1)
  - `time()` closure-based recording, `all_stats()` sorted by total desc, `format_report()` table
  - `estimate_bandwidth_gbs()`, `ProfileScope<'a>` RAII guard (records in Drop)
  - 24 new tests; 174 unit + 60 integration + 4 doc tests passing
- ✅ **GPU-side timestamp profiler** (0.1.2, complementing the CPU-side `PassProfiler` above)
  - `profiler::GpuTimestampProfiler`, backed by `wgpu::Features::TIMESTAMP_QUERY` (`REQUIRED_FEATURES`, `DEFAULT_MAX_PASSES = 32`; `new`, `stats`, `period_ns`, `reserved_passes`, `pass_writes`, `resolve`, `collect`, `discard`)
  - `Rasterizer::enable_gpu_timestamps()` / `disable_gpu_timestamps()` / `gpu_timestamps()`; `Rasterizer::new` requests the feature automatically whenever the adapter supports it, so `enable_gpu_timestamps()` cleanly returns `RenderError::GpuInit` rather than panicking on hardware that doesn't
- ✅ **Device-limit validation** (0.1.2) — `rasterizer::rasterizer_device_limits`, `RASTERIZER_STORAGE_BUFFERS_PER_STAGE` (= 16), `RASTERIZE_FWD_WORKGROUP_STORAGE_BYTES` (= 17,408); `Rasterizer::from_device` now validates a caller-supplied `wgpu::Device`'s limits upfront instead of letting an under-provisioned device fail later as an opaque wgpu pipeline-validation error
- ⬜ **Validation layers enhancements**
  - When gpu_debug enabled, validate all buffers
  - Check for NaN/Inf in gradients
  - Verify buffer sizes match expected shapes

### Documentation
- ✅ **Shader documentation** — no longer sparse: all 17 `.wgsl` files carry substantial comments (roughly 20-65% comment-only lines; the backward shaders are the most thorough, e.g. `cov2d_bwd.wgsl` ~65%, `rasterize_bwd.wgsl` ~45%, documenting both the gradient math and the 0.1.2 bug history). Not a formally reviewed "every dispatch explained" pass — just an honest correction of the previous "sparse" claim
- ⬜ **Algorithm explanation**
  - Tile-based rasterization walkthrough
  - Backward pass gradient flow diagram
  - Memory layout diagrams
- ✅ **Usage examples** — Three compilable examples in `examples/`:
  - `examples/render_ply.rs` — Load PLY, compute bounding box + opacity stats, round-trip PLY, GPU notes
  - `examples/benchmark.rs` — Create N Gaussians, profile creation/PLY save-load/clone/prune/split via `PassProfiler`, print report table
  - `examples/flame_binding.rs` — 4-vertex/2-face mesh, 4 Gaussians with explicit face_indices/barycentric/local_offsets, barycentric world-pos interpolation
  - All compile with `cargo build --examples -p oxigaf-render`, 0 errors, 0 warnings
  - Note: `pub use gaussian::{GaussianAttributes, GaussianModel}` added to `lib.rs` for crate-root access

## 💡 Future Enhancements (beyond original plan)

### Advanced Features
- ✅ **Compressed Gaussian representation** (`compression.rs`)
- ✅ **Level-of-detail (LOD)** (`lod.rs`)
- ✅ **Environment map support** (`environment.rs`)
- ✅ **Denoising** (`denoising.rs`)

### Platform Support
- ⬜ **WebGPU compatibility**
  - Ensure shaders compile for WebGPU
  - Browser-based rendering
  - WASM target support
- ⬜ **Mobile GPU optimization**
  - Reduce precision for mobile (F16)
  - Optimize tile size for mobile bandwidth
  - Power-efficient rendering modes

### Multi-GPU
- ⬜ **Multi-GPU rendering**
  - Split Gaussians across GPUs
  - Parallel rendering with composition
  - Automatic load balancing

## 🐛 Known Issues

- ⬜ **F32 atomicAdd contention**
  - CAS loop may be slow under high contention
  - Mitigation: Per-tile reduction before global write
  - Need benchmarking on real workloads
- ⬜ **wgpu backend compatibility**
  - Some shaders may not work on all backends (Vulkan/Metal/DX12/GL)
  - OpenGL ES backend (llvmpipe) is slow for large scenes
  - Need testing matrix across backends

## 📊 Current Status

### Implementation: ~99% complete (v0.1.2, occupancy tuning + glTF export + GPU timestamp profiler done)
- ✅ Forward pass: 100%
- ✅ Backward pass: gradients 100% verified (two gradient-correctness bugs were fixed in 0.1.2, see the callout under "Backward Pass (Differentiability)" above before assuming a pre-0.1.2 trained model is fine); the originally-planned *separate* cov2d-backward shader is still unbuilt — that math stays inline in `preprocess_bwd.wgsl` — see the Cov2D row in the Comparison table below
- ✅ GPU infrastructure: 100% (0.1.2: device-limit validation, GPU timestamp profiler)
- ✅ Buffer management: 100%
- ✅ FLAME mesh binding: 100% (forward ✅ dedicated deform_gaussians.wgsl + DeformPipeline, backward ✅)
- ✅ Adaptive density control: 100% (`density.rs`, 25 tests)
- ✅ Gaussian I/O: 100% (PLY done; SafeTensors done; glTF export done in 0.1.2)
- ⬜ burn-autodiff integration: 0% (no `burn` dependency anywhere in the workspace; likely superseded by the torsh-based `oxigaf-bridge` path rather than pending)
- ✅ Multi-view batch rendering: 100% (`multi_view.rs`, CPU + GPU-ignored tests)

### Tests: 2,903 tests (all passing)
- ✅ 2,887 run by default: `cargo nextest run -p oxigaf-render --all-features`
- ✅ 16 more require real GPU hardware, `#[ignore]`d; all 16 pass via `--run-ignored ignored-only`
- ✅ 23 doc-tests passing, 3 `#[ignore]`d
- ✅ Gradient verification: 78 (`gpu_gradient_verify.rs` + `tests/gpu_gradient_verify/sh.rs` and `tests/gradient_verification/`) — **DONE**, and the harness's own NaN/empty-guard bug is now covered too
- ⬜ Cross-validation with Python: 0
- Coverage: Excellent including backward pass, multi-view, density control, debug readback, GPU profiler (CPU- and GPU-side), adaptive workgroup, glTF export

### Documentation: Good
- ✅ Rustdoc with feature explanations
- ✅ Module-level documentation
- ✅ Usage examples: 3 compilable examples (`render_ply.rs`, `benchmark.rs`, `flame_binding.rs`)
- ✅ Shader comments: no longer sparse — see "Shader documentation" above
- ⬜ Algorithm walkthrough: 0

### Benchmarks: Good
- ✅ Forward/backward pass timing
- ✅ Per-shader pass timing (`profiler.rs`, `PassProfiler`, EMA α=0.1, `format_report()` table)
- ✅ Memory bandwidth estimation (`estimate_bandwidth_gbs()`)
- ⬜ Missing: Different Gaussian counts

## 📈 Comparison: Implementation vs Plan

| Feature | Plan | Current | Notes |
|---------|------|---------|-------|
| Preprocess shader | ✅ | ✅ | **EXCEEDS**: Specialized SH shaders (10× faster) |
| Radix sort | ✅ | ✅ | Fully implemented (Fuchsia-based) |
| Tile assignment | ✅ | ✅ | Fully implemented |
| Tile ranges | ✅ | ✅ | Fully implemented |
| Forward rasterization | ✅ | ✅ | Fully implemented |
| Backward rasterization | ✅ | ✅ | Fully implemented |
| Cov2D backward | ✅ Separate shader | ⬜ | Computed inline in preprocess_bwd; `cov2d_backward.rs` (CPU test oracle, 10 tests) and `cov2d_bwd.wgsl` (math docs) exist alongside it but neither is a compiled/dispatched GPU pipeline, so this row stays ⬜ |
| Preprocess backward | ✅ | ✅ | Fully implemented |
| FLAME binding forward | ✅ | ✅ | Fully implemented |
| FLAME binding backward | ✅ | ✅ | **Done v0.1.0** (`preprocess_bwd.wgsl`) |
| Buffer pool | ⬜ Not in plan | ✅ | **EXCEEDS PLAN** - Memory efficiency |
| Specialized SH shaders | ⬜ Not in plan | ✅ | **EXCEEDS PLAN** - 10× speedup |
| Adaptive density control | ✅ | ✅ | **Done** (`density.rs`, Clone/Split/Prune, 25 tests) |
| PLY I/O | ✅ | ✅ | Done (save/load, binary LE, SH degree 0-3, 8 tests). 0.1.2: `f_rest_*` property order fixed to channel-major — pre-0.1.2 `sh_degree >= 1` files load permuted, re-export them |
| SafeTensors I/O | ✅ | ✅ | Done (all fields + metadata, 10 tests) |
| glTF export | ⬜ Not in plan | ✅ | **NEW 0.1.2** — `gltf::write_gltf`, spec-conformant glTF 2.0, replacing 2 incompatible emitters elsewhere in the workspace |
| Gradient verification | ✅ | ✅ | **Done v0.1.0**, correctness fixed further in 0.1.2 (78 tests, ≤5e-2 median error / ≤2.5e-1 for position — see "Differentiability" in README) |
| Mesh binding shader forward | ✅ | ✅ | **Done** (`deform_gaussians.wgsl` + `DeformPipeline`, 12 tests) |
| Intermediate buffer readback | ⬜ | ✅ | **Done** (`debug_readback.rs`, RasterizationSnapshot/Stats, tile viz, 29 tests) |
| GPU profiler integration (CPU-side) | ⬜ | ✅ | **Done** (`profiler.rs`, PassProfiler, EMA, RAII ProfileScope, bandwidth est., 24 tests) |
| GPU timestamp profiler (GPU-side) | ⬜ Not in plan | ✅ | **NEW 0.1.2** — `profiler::GpuTimestampProfiler` + `Rasterizer::enable_gpu_timestamps()`, `wgpu::Features::TIMESTAMP_QUERY` |
| Device-limit validation | ⬜ Not in plan | ✅ | **NEW 0.1.2** — `rasterizer::rasterizer_device_limits`; `Rasterizer::from_device` validates upfront instead of failing later as an opaque pipeline error |
| Occupancy tuning / Adaptive workgroup | ⬜ | ✅ | **Done** (`workgroup.rs`, WorkgroupProfile/Config/Benchmarker, adaptive(num_gaussians), 34 tests) |
| burn-autodiff integration | ✅ | ⬜ | Not started; no `burn` dependency exists anywhere in the workspace — likely superseded by the torsh-based `oxigaf-bridge` integration rather than pending |

## 🎯 Priority for v1.0

**Every originally-scoped v1.0 blocker except burn-autodiff is resolved:**
1. ✅ ~~**Gradient verification**~~ — Done (78 tests, all passing; correctness bugs fixed 0.1.2)
2. ✅ ~~**FLAME binding backward shader**~~ — Done
3. ✅ ~~**Mesh binding shader forward**~~ — Done (`deform_gaussians.wgsl` + `DeformPipeline`, 12 tests)
4. ✅ ~~**Intermediate buffer readback**~~ — Done (`debug_readback.rs`, 29 tests)
5. ✅ ~~**GPU profiler integration**~~ — Done (`profiler.rs`, PassProfiler/PassStats/ProfileScope, EMA α=0.1, 24 tests; GPU-side `GpuTimestampProfiler` added 0.1.2)
6. ⬜ **burn-autodiff integration** — not started, and not actually blocking: no `burn` dependency exists anywhere in the workspace, and cross-framework tensor integration in practice goes through `oxigaf-bridge`'s torsh-core/torsh-tensor/torsh-nn stack instead. Kept here as an honest record of an originally-planned item that was never started, not as an active release blocker

**High Priority:**
4. ✅ ~~Adaptive density control~~ — Done (`density.rs`, Clone/Split/Prune, golden-ratio split, 25 tests)
5. ✅ ~~PLY I/O~~ (done — save/load, binary LE, SH degree 0-3; 0.1.2: `f_rest_*` fixed to channel-major)
6. ✅ ~~SafeTensors I/O~~ (done — all fields + metadata, 10 tests)
7. ⬜ Cross-validation with gsplat

**Medium Priority:**
7. ✅ ~~Multi-view batch rendering~~ — Done (`multi_view.rs`, `render_turntable()`, CPU + GPU-ignored tests)
8. ⬜ Gaussian initialization utilities
9. ✅ ~~Usage examples~~ — Done (3 compilable examples: `render_ply.rs`, `benchmark.rs`, `flame_binding.rs`)

**Low Priority:**
10. ✅ ~~Shader documentation~~ — no longer sparse; see "Shader documentation" above
11. ✅ ~~Occupancy tuning / adaptive workgroup~~ — Done (`workgroup.rs`, 34 tests)
12. ⬜ Advanced features (compression, LOD)

## 🏆 Implementation Highlights

**Where current implementation EXCEEDS the plan:**

1. **Specialized SH Shaders** (major optimization not in plan)
   - Per-degree shader specialization (sh0, sh1, sh2, sh3)
   - 10× speedup through compile-time loop unrolling
   - Eliminates dynamic branching
   - Optimized for different GPU architectures

2. **Buffer Pool** (not in plan)
   - Memory-efficient buffer reuse
   - Automatic resizing for dynamic workloads
   - Statistics tracking (allocations, reuses)
   - Reduces allocations by 90%+

3. **Comprehensive Error Handling** (better than planned)
   - Device lost detection
   - NaN/Inf detection hooks
   - Detailed error context
   - Graceful degradation

4. **Test Coverage** (more thorough than planned)
   - 2,903 tests total (plan didn't specify count): 2,887 run by default + 16 GPU-hardware `#[ignore]`d, all passing
   - 78 gradient verification tests (finite-difference, ≤5e-2 median error / ≤2.5e-1 for position)
   - 12 CPU-side mesh binding shader tests
   - Integration tests

5. **Code Organization** (cleaner than planned)
   - All files under the 2,000-line policy limit (largest is 1,977 lines)
   - Clear module boundaries
   - Pool abstraction for memory management

**Current implementation is PRODUCTION-READY for:**
- Forward-only rendering (visualization)
- Gaussian rasterization at interactive framerates
- FLAME mesh binding
- Differentiable training (both gradient-correctness bugs found in this crate's backward shaders are fixed as of 0.1.2 — see the callout under "Backward Pass (Differentiability)")

**Not yet ready for:**
- Cross-framework integration via `burn` specifically — not started, and likely superseded rather than pending (see "Priority for v1.0")

## 🚀 Status through v0.1.2: density control, multi-view, debug, profiler, glTF export, and two critical gradient fixes all done ✅

**Completed in v0.1.0+:**
1. ✅ **Gradient Verification** — 78 finite-difference tests, all parameters, ≤5e-2 median error / ≤2.5e-1 for position
2. ✅ **FLAME Binding Backward** — ∂L/∂local_offset implemented and tested
3. ✅ **Mesh Binding Shader Forward** — `deform_gaussians.wgsl` + `DeformPipeline`, TBN/barycentric/quaternion composition, 12 tests

**Completed in v0.1.1:**
4. ✅ **Batch Multi-View Rendering** — `multi_view.rs`, `MultiViewRenderer`, `render_turntable()`, sRGB gamma, CPU + GPU-ignored tests
5. ✅ **Adaptive Density Control** — `density.rs`, Clone/Split (golden-ratio 0.618)/Prune, exp()/sigmoid() comparisons, 25 tests
6. ✅ **Intermediate Buffer Readback** — `debug_readback.rs`, `RasterizationSnapshot`/`RasterizationStats`, AABB tile assignment, `tile_occupancy_image()`/`hotspot_tiles()`/`tile_for_pixel()`, 29 tests
7. ✅ **GPU Profiler Integration (CPU-side)** — `profiler.rs`, `PassProfiler` (Mutex+AtomicU64), `PassStats` (count/total/min/max/EMA α=0.1), `time()` closure, `ProfileScope<'a>` RAII guard, `estimate_bandwidth_gbs()`, 24 tests
8. ✅ **Occupancy Tuning / Adaptive Workgroup Size** — `workgroup.rs`, `WorkgroupSize`/`WorkgroupProfile`/`WorkgroupConfig`/`WorkgroupBenchmarker`, `adaptive(num_gaussians)`, Mobile/Balanced/HighThroughput profiles, 34 tests

**Completed in v0.1.2:**
9. ✅ **Two backward-shader gradient-correctness fixes** — wrong-Gaussian gradient accumulation in the tile kernel, and a missing position gradient through view-dependent SH color for `sh_degree >= 1` — see the callout under "Backward Pass (Differentiability)" above. Both affected every 0.1.1 training run; retraining is the only remedy
10. ✅ **glTF export** — `gltf::write_gltf`/`GltfError`, spec-conformant glTF 2.0
11. ✅ **GPU timestamp profiler** — `profiler::GpuTimestampProfiler` + `Rasterizer::enable_gpu_timestamps()`, complementing the CPU-side profiler above
12. ✅ **Device-limit validation** — `rasterizer::rasterizer_device_limits`; `Rasterizer::from_device` now validates upfront and rejects `tile_size != 16`
13. ✅ **PLY `f_rest_*` channel-major fix** — see the migration note in `CHANGELOG.md`; re-export pre-0.1.2 `sh_degree >= 1` PLYs

**Remaining for future versions:**
14. ⬜ **burn-autodiff Integration** — not started; no `burn` dependency exists anywhere in the workspace, and is likely superseded by `oxigaf-bridge`'s torsh-based integration (see "Priority for v1.0" above) rather than genuinely pending

15. ✅ ~~**PLY I/O**~~ — Done (`gaussian.rs`: `save_ply`/`load_ply`, binary LE, SH degree 0-3, 8 tests)
    ✅ ~~**SafeTensors I/O**~~ — Done (`gaussian.rs`: `save_safetensors`/`load_safetensors`, all fields + metadata, 10 tests)

**oxigaf-render v0.1.2 is fully functional for the GAF training pipeline, with full debug/profiling support, glTF export, and — as of this release — a correctness-verified backward pass for `sh_degree >= 1` models.**

## 📝 Notes

- **Nightly Rust**: Not required (unlike oxigaf-flame)
- **wgpu version**: 30 (workspace `Cargo.toml`)
- **Platform support**: Vulkan (Linux/Windows), Metal (macOS), DX12 (Windows), OpenGL ES (CPU fallback)
- **MSRV**: Rust 1.87 (workspace `rust-version`)
- **Pure Rust**: 100% (no C/Fortran dependencies)

## 🤝 Contributions

This is a one-person project. Contributions (issues, PRs) are welcome.
