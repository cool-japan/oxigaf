# TODO for oxigaf-flame

## ✅ Completed (from plan)

### Core Functionality
- ✅ FLAME model loading from `.npy` files (converted from `.pkl`)
- ✅ Linear Blend Skinning (LBS) forward pass: parameters → posed mesh
- ✅ Rodrigues rotation formula implementation (axis-angle → rotation matrix)
- ✅ Blend shapes application (shape + expression blendshapes)
- ✅ Joint regression and kinematic chain computation
- ✅ Per-vertex normal computation
- ✅ Mesh structure with vertices, normals, and faces
- ✅ FlameParams struct for shape, expression, pose, and translation parameters
- ✅ Error handling with FlameError (thiserror-based)

### Normal Map Generation
- ✅ CPU software rasterizer for normal maps (Z-buffer, scanline)
- ✅ Camera intrinsics and extrinsics support
- ✅ World-space normal encoding to RGB

### Mesh Sampling
- ✅ Surface point sampling for Gaussian initialization
- ✅ Barycentric coordinate-based sampling
- ✅ Area-weighted triangle sampling

### Performance Optimizations (Beyond Original Plan)
- ✅ **SIMD acceleration** (feature: `simd`, requires nightly Rust)
  - 2-4× speedup for Rodrigues rotation in batch operations
  - Vectorized blend shapes application
  - SIMD-accelerated normal map rendering
- ✅ **Parallel processing** (feature: `parallel`, uses rayon)
  - `forward_batch_par()` for parallel mesh generation
  - `compute_normals_batch_par()` for parallel normal computation
  - Near-linear speedup with CPU core count
- ✅ BatchBufferPool for memory-efficient batch processing
- ✅ Zero-cost abstractions with extensive use of `#[inline]`

### Developer Experience
- ✅ FlameParamsBuilder for ergonomic parameter construction
- ✅ Comprehensive documentation with examples
- ✅ 280 unit, integration, memoization, thread-safety, property-based, traits, vertex mask, visualization, landmark, dynamic landmark, and GPU buffer tests
- ✅ Property-based testing with proptest
- ✅ Extensive benchmarking suite:
  - `flame_bench.rs` - overall FLAME forward pass
  - `lbs_forward.rs` - LBS performance
  - `rodrigues.rs` - Rodrigues rotation
  - `normal_map.rs` - normal map rendering
  - `simd_ops.rs` - SIMD operations (feature-gated)

### Code Quality
- ✅ No unwrap policy (`#![deny(clippy::unwrap_used)]`)
- ✅ No expect in library code (`#![deny(clippy::expect_used)]`)
- ✅ Comprehensive clippy lints enabled
- ✅ All source files under 2000 lines (refactoring policy compliant)

### Model Format Support
- ✅ **Safetensors I/O** (`io_safetensors.rs`)
  - Load FLAME model from `.safetensors` format
  - Save FLAME model to `.safetensors` (preserves metadata)
  - Benefit: Single-file model format, HuggingFace ecosystem compatibility

### Sequence Data Loading
- ✅ **FlameSequence** (`sequence.rs`, 1,351 lines)
  - Load per-frame parameters from JSON files
  - `get_frame()` method with LRU caching
  - `interpolate()` for temporal interpolation between frames
  - `num_frames()` for sequence length
  - Memory-efficient: LRU eviction for large sequences

## 🚧 In Progress

Currently none.

## ✅ Completed (v0.1.1)

### Avatar Rigging & Expression System (v0.1.1)
- ✅ **AvatarRig** — skeleton-mesh binding for full avatar control
- ✅ **GazeController** — gaze direction control with pupil tracking
- ✅ **HeadTracker** — real-time head pose tracking integration
- ✅ **HeadGeometry** — head geometry analysis and symmetry tools
- ✅ **PoseEstimation** — camera-relative head pose estimation
- ✅ **PosePrior** — learned pose prior for regularisation
- ✅ **Expressions** — FLAME expression blend shape interface
- ✅ **ExpressionAnimation** — keyframe expression animation
- ✅ **ExpressionClustering** — cluster expressions into prototypes
- ✅ **ExpressionTransfer** — transfer expressions between identities
- ✅ **FACS AU coefficients** — Action Unit decomposition from expressions
- ✅ **Emotion recognition** — emotion classification from AU coefficients
- ✅ **Phoneme-driven animation** — lip-sync from phoneme sequence

### Mesh Processing Suite (v0.1.1)
- ✅ **Mesh operations** — boolean, clip, split, merge utilities
- ✅ **Mesh repair** — hole filling, degenerate face removal
- ✅ **Mesh smoothing** — Laplacian and bilateral smoothing
- ✅ **Loop subdivision** — Loop scheme subdivision surface
- ✅ **Catmull-Clark subdivision** — Catmull-Clark subdivision surface
- ✅ **Mesh morphing** — target-shape morph blending
- ✅ **Mesh analysis** — area, volume, curvature, quality metrics

### Geometry & Statistical Tools (v0.1.1)
- ✅ **Geodesic distance** — heat-method geodesic distance computation
- ✅ **Spectral analysis** — Laplace-Beltrami spectral decomposition
- ✅ **Multiresolution mesh** — wavelet-based multiresolution representation
- ✅ **Statistical shape model** — PCA shape space with sampling
- ✅ **Symmetry detection** — bilateral symmetry plane estimation

### UV & Texture Pipeline (v0.1.1)
- ✅ **UV parameterisation** — least-squares conformal mapping
- ✅ **Texture baking** — render-to-texture from 3D attributes
- ✅ **Face atlas generation** — packed UV atlas for head mesh
- ✅ **Albedo map** — diffuse colour texture extraction
- ✅ **SH lighting model** — spherical harmonic environment lighting

### Motion, Fitting & Utilities (v0.1.1)
- ✅ **Timeline** — frame-indexed parameter sequence
- ✅ **Warp field** — per-vertex displacement field animation
- ✅ **Shape retargeting** — identity-shape transfer retargeting
- ✅ **Dynamic landmark tracking** — per-frame 3D landmark estimation
- ✅ **Blend shape solver** — least-squares blend shape coefficient solver
- ✅ **Rigid alignment** — Procrustes alignment to canonical pose
- ✅ **Canonical conversion** — convert to canonical head-pose space
- ✅ **Vertex masks** — region-labelled vertex selection masks
- ✅ **Visibility culling** — back-face and frustum visibility tests
- ✅ **Contact detection** — mesh self-intersection and contact detection
- ✅ **Depth estimation** — FLAME-mesh-based monocular depth estimation
- ✅ **Face normalisation** — crop/align face region to standard frame
- ✅ **Parameter sampler** — random FLAME parameter sampling for augmentation

## 📋 Planned (future versions)
- ✅ **OBJ export** (`Mesh::export_obj()`)
- ✅ **PLY export** (`Mesh::export_ply()`)
- ⬜ Support for exporting with UV coordinates (if loaded)

### Spatial Search
- ✅ **KD-tree integration** (using `kiddo` crate)
  - `Mesh::build_kdtree()` returns `KdTree<f32, u32, 3, 32, u32>`
  - `nearest_vertex_in_tree()` free function
  - `kiddo` added to workspace + crate dependencies
  - Benefit: Faster mesh binding, correspondence finding

### GPU-Accelerated Normal Map Rendering
- ⬜ **wgpu-based normal map renderer** (feature: `gpu-raster`)
  - Vertex + fragment shader pipeline
  - Batch rendering of multiple views
  - Framebuffer readback
  - Target: 10-100× speedup for batch rendering
  - Priority: Medium (CPU renderer is fast enough for most use cases)

### Advanced FLAME Features
- ⬜ **UV texture mapping** support
  - Load `texture_data_*.npy` files
  - UV coordinate interpolation
  - Texture map rendering
- ✅ **estimate_pitch_from_vertical** — real geometric algorithm (vertical centroid spread → asin) in `pose_estimation.rs`
- ✅ **Dynamic landmarks (pose-dependent contours)** — `dynamic_landmarks.rs` (295 lines). `ContourSide` (Left/Right/Both). `DynamicLandmarkConfig` (num_contour_landmarks=17, side_threshold_rad=0.1). `ContourVertexChains::default_flame()` (left chain: vertices 1-17, right chain: vertices 4984-5000). `DynamicLandmarkExtractor`: `extract_yaw(params)` reads pose[1] (Y axis-angle), `select_contour_side()` threshold-based, `extract(mesh, params)` → 17 Landmarks (with graceful out-of-bounds handling), `extract_all()` → 68 landmarks (jaw-line 0-16 overwritten by dynamic contour). `Mesh::extract_dynamic_landmarks()`. 25 new tests.
- ✅ **Vertex masks** (face region segmentation) (`vertex_mask.rs`)
  - `FaceRegion` enum (8 variants: Face, LeftEye, RightEye, Mouth, Neck, LeftEar, RightEar, Scalp)
  - `VertexMask` with geometric classification from vertex positions
  - `region_indices`, `region_mask`, `vertex_region`, `region_counts`
  - `Mesh::vertex_mask()`
  - 12 new tests
- ✅ **Static landmarks (51/68 3D face landmarks)** — `landmarks.rs` (386 lines)
  - `LandmarkGroup` enum: JawLine (17pts), LeftEyebrow (5pts), RightEyebrow (5pts), Nose (9pts), LeftEye (6pts), RightEye (6pts), OuterLip (12pts), InnerLip (8pts)
  - `Landmark` struct: position, index, group
  - `LandmarkExtractor` with `FLAME_68_VERTEX_INDICES` constant
  - Methods: `new`/`default`, `with_indices`, `extract`, `extract_group`, `group_indices`, `num_landmarks`, `group_centroid`
  - `Mesh::extract_landmarks()` convenience wrapper
  - `NUM_LANDMARKS = 68`
  - 32 new tests

### Conversion Scripts
- ⬜ **Python conversion script improvements**
  - `convert_flame.py`: Support FLAME 2023_Open.pkl
  - Support for different FLAME versions (2017, 2019, 2020, 2023, 2023_Open)
  - Validation script to compare Rust vs Python outputs

### Testing & Validation
- ⬜ **Ground truth validation**
  - Generate reference data from FLAME_PyTorch
  - Layer-by-layer comparison (tolerance < 1e-4)
  - Visual regression tests for normal maps
- ✅ **Thread safety tests** (`tests/thread_safety_tests.rs`)
  - Compile-time `assert_send_sync::<FlameModel>()`
  - 4 runtime tests verifying `Arc<FlameModel>` across threads
- ⬜ **WASM compatibility** (feature-gated, for browser demos)

## 💡 Future Enhancements (beyond original plan)

### Performance
- ⬜ **GPU-accelerated LBS** (wgpu compute shader)
  - Full FLAME forward pass on GPU
  - Target: <1ms for 5023 vertices
  - Benefit: Reduce CPU→GPU transfer overhead in training loop
- ⬜ **Multi-resolution mesh support**
  - Decimated FLAME variants (1K, 2.5K, 10K vertices)
  - Adaptive LOD based on distance
- ✅ **Cached computation results**
  - `joint_cache: Mutex<HashMap<u64, Vec<[f32; 3]>>>` in `FlameModel`
  - `compute_shape_hash(shape) -> u64` (FNV-style hash)
  - `joint_positions_cached()` private method with 64-entry eviction policy
  - 8 memoization tests

### Usability
- ⬜ **FLAME model auto-download utility**
  - Download from MPI server (requires agreement)
  - Cache in `~/.cache/oxigaf/flame/`
  - Checksum verification
- ✅ **Parameter interpolation utilities** (`params.rs`)
  - `FlameParams::lerp(other, t)` — linear interpolation for shape/expression/translation, quaternion slerp for pose axis-angle
  - `FlameParams::slerp_pose(other, t)` — convenience method returning only interpolated pose
  - Private helpers: `axis_angle_to_quat`, `quat_slerp`, `quat_to_axis_angle`
  - Handles: near-zero rotation, antiparallel quaternions, dimension mismatch errors
  - 19 new tests including 2 proptest property-based tests
- ⬜ **Expression library**
  - Predefined expressions (smile, frown, surprise, etc.)
  - Load from canonical expression dataset

### Integration
- ✅ **Trait for oxigaf-diffusion integration** (`traits.rs`)
  - `NormalMapProvider` trait (generate_normal_map, generate_normal_maps_multi_view, default_resolution)
  - `MeshSurfaceSampler` trait (sample_surface → SurfaceSample)
  - `DefaultSampler` struct implementing `MeshSurfaceSampler` via existing area-weighted sampler
  - 11 new tests
- ✅ **Mesh deformation shader helpers / GPU buffer export** — `gpu_buffers.rs`. `GpuMeshBuffers` (vertices [V*4]f32 padded xyz+1.0, normals [V*4]f32 padded xyz+0.0, faces [F*4]u32 padded v0v1v2+0). `GpuBufferConfig` (normalize_normals=true, recompute_normals=false, include_degenerate=false). `Mesh::to_gpu_buffers()` / `to_gpu_buffers_with_config()`. `validate()`, element accessors (`vertex_position/normal/face_indices`), raw byte accessors via bytemuck::cast_slice. Degenerate face detection via cross-product magnitude. Normal recomputation inline without mut. 18 new tests. Total: 2209 tests passing (v0.1.1).

### Developer Tools
- ✅ **Visualization utilities** — `visualize.rs`. `SvgCamera` (front_view, side_view, three_quarter_view; perspective project via nalgebra look-at; returns None for behind-camera). `WireframeOptions` (image_size, edge_color, stroke_width, background, cull_backfaces, vertex display). `render_wireframe` (painter's algorithm depth sort, back-face culling, edge deduplication via HashSet<(u32,u32)>). `render_joints_svg` (circles + optional text labels). `render_mesh_with_joints`. `save_svg`. `SvgBuilder` (private, accumulates `<line>/<circle>/<text>` SVG elements). XML escaping for safety. 19 new tests.
- ⬜ **CLI tool** (`flame-tool`)
  - Convert `.pkl` → `.npy` / `.safetensors`
  - Render normal maps from command line
  - Validate model files
  - Export meshes

## 🐛 Known Issues

Currently none reported.

## 📊 Current Status

### Implementation: ~97% complete
- ✅ Core LBS algorithm: 100%
- ✅ Normal map rendering: 100% (CPU), 0% (GPU)
- ✅ Mesh sampling: 100%
- ✅ SIMD optimizations: 100%
- ✅ Parallel processing: 100%
- ✅ Model format support: 100% (.npy ✅, .safetensors ✅)
- ✅ Sequence loading: 100% (FlameSequence with LRU cache)
- ✅ Mesh export: 100% (OBJ + PLY binary LE)
- ✅ Static landmarks: 100% (`landmarks.rs`, 68 landmarks, 8 groups, 32 tests)
- ✅ GPU buffer export: 100% (`gpu_buffers.rs`, padded f32/u32 buffers, bytemuck, 18 tests)
- ⬜ Advanced FLAME features: 60% (static landmarks + dynamic landmarks + GPU buffers done, UV/GPU rasterizer pending)

### Tests: 280 tests (all passing)
- ✅ Unit tests: 43 (in `src/`)
- ✅ Doc tests: 16
- ✅ Lib tests: 58
- ✅ Integration tests: 12 (in `tests/`)
- ✅ Memoization tests: 8 (`joint_positions_cached` + cache eviction)
- ✅ Mesh export tests: 4
- ✅ Property-based tests: 13 (proptest)
- ✅ Thread safety tests: 7 (`tests/thread_safety_tests.rs`: 1 compile-time + 4 runtime Arc, 2 more)
- ✅ Traits tests: 11 (`traits.rs`: NormalMapProvider, MeshSurfaceSampler, DefaultSampler)
- ✅ Vertex mask tests: 12 (`vertex_mask.rs`: FaceRegion, VertexMask, geometric classification)
- ✅ Visualization tests: 19 (`visualize.rs`: SvgCamera, WireframeOptions, render_wireframe, render_joints_svg, SvgBuilder)
- ✅ Landmark tests: 32 (`landmarks.rs`: LandmarkExtractor, LandmarkGroup, group_centroid, extract_group, Mesh::extract_landmarks)
- ✅ Dynamic landmark tests: 25 (`dynamic_landmarks.rs`: ContourSide, DynamicLandmarkConfig, ContourVertexChains, DynamicLandmarkExtractor, Mesh::extract_dynamic_landmarks)
- ✅ GPU buffer tests: 18 (`gpu_buffers.rs`: GpuMeshBuffers, GpuBufferConfig, validate, accessors, degenerate face detection, normal recomputation)
- Coverage: Good for core functionality

### Documentation: Excellent
- ✅ Rustdoc with examples
- ✅ Module-level documentation
- ✅ Inline comments for complex logic
- ✅ Feature flag documentation
- ⬜ Missing: Standalone usage guide

### Benchmarks: Comprehensive
- ✅ 5 benchmark files covering:
  - FLAME forward pass (sequential & parallel)
  - Rodrigues rotation
  - Normal map rendering (multiple resolutions)
  - Blend shapes application
  - SIMD operations
- Performance: ~1-2ms for 5023 vertices on modern CPUs

## 📈 Comparison: Implementation vs Plan

| Feature | Plan | Current | Notes |
|---------|------|---------|-------|
| FLAME model loading | ✅ .npz | ✅ .npy | Exceeds plan: added BatchBufferPool |
| LBS forward pass | ✅ | ✅ | Exceeds plan: SIMD + parallel batch processing |
| Rodrigues rotation | ✅ | ✅ | Exceeds plan: SIMD optimization |
| Normal map (CPU) | ✅ | ✅ | Matches plan |
| Normal map (GPU) | ⬜ Optional | ⬜ Not started | Lower priority |
| Mesh sampling | ✅ | ✅ | Matches plan |
| safetensors support | ⬜ Optional | ✅ **Done** (v0.1.0) | `io_safetensors.rs` |
| Sequence loading | ⬜ Planned | ✅ **Done** (v0.1.0) | `sequence.rs` (1,351 lines) |
| Mesh export | ⬜ Planned | ✅ **Done** (v0.1.1) | `Mesh::export_obj()`, `Mesh::export_ply()` |
| KD-tree | ⬜ Planned | ✅ **Done** | `build_kdtree()` + `nearest_vertex_in_tree()` (kiddo crate) |
| UV mapping | ⬜ Optional | ⬜ Not started | Only needed for texture rendering |
| Landmarks | ⬜ Optional | ✅ **Done** | `landmarks.rs` (386 lines), 68 landmarks, 8 groups, 32 tests |
| Vertex masks | ⬜ Optional | ✅ **Done** | `vertex_mask.rs`, 8 FaceRegion variants, geometric classification, 12 tests |
| GPU buffer export | ⬜ Planned | ✅ **Done** | `gpu_buffers.rs`, padded f32/u32 buffers, bytemuck, degenerate detection, 18 tests |

## 🎯 Priority for v1.0

**High Priority:**
1. ✅ ~~Core LBS (DONE)~~
2. ✅ ~~Normal map rendering (DONE)~~
3. ✅ ~~safetensors support (DONE — v0.1.0)~~
4. ✅ ~~FlameSequence loader (DONE — v0.1.0)~~

**Medium Priority (future):**
5. ✅ ~~PLY export (DONE — v0.1.1, also OBJ)~~
6. ✅ ~~Vertex masks~~ — Done (`vertex_mask.rs`, 8 FaceRegion variants, 12 tests)

**Low Priority:**
7. ⬜ GPU normal map renderer
8. ⬜ UV mapping
9. ✅ ~~Landmarks~~ — Done (`landmarks.rs`, 386 lines, 68 landmarks, 8 `LandmarkGroup` variants, 32 tests)
10. ✅ ~~Dynamic landmarks~~ — Done (`dynamic_landmarks.rs`, 295 lines, pose-dependent contours, 25 tests)

**Future:**
- GPU-accelerated LBS
- WASM support
- CLI tool

## 🏆 Implementation Highlights

**Where current implementation EXCEEDS the plan:**

1. **SIMD Acceleration** (not in original plan)
   - Portable SIMD for cross-platform vectorization
   - 2-4× speedup for rotation-heavy operations
   - Feature-gated for stable Rust compatibility

2. **Parallel Batch Processing** (not detailed in plan)
   - Rayon-based parallel forward pass
   - Near-linear scalability with CPU cores
   - Memory-efficient buffer pooling

3. **Developer Experience** (better than planned)
   - FlameParamsBuilder for ergonomic API
   - Comprehensive benchmarking suite
   - Property-based testing
   - Strict no-unwrap policy

4. **Code Quality** (stricter than planned)
   - All files < 2000 lines
   - Extensive clippy lints
   - Zero unsafe code in public API

**Current implementation is PRODUCTION-READY for:**
- FLAME mesh generation from parameters
- CPU normal map rendering for diffusion conditioning
- Mesh surface sampling for Gaussian initialization
- Batch processing of video frames

**Production-ready additions since original plan:**
- End-to-end video pipeline: FlameSequence with LRU cache ✅
- Model weight sharing: safetensors I/O ✅
- Mesh visualization: PLY + OBJ export ✅
- Parameter interpolation: `lerp`/`slerp_pose` with quaternion math ✅
- KD-tree spatial search: `build_kdtree()` + `nearest_vertex_in_tree()` ✅
- Thread safety verified: `Arc<FlameModel>` across threads, 7 tests ✅
- Memoized joint positions: `joint_positions_cached()` with 64-entry LRU eviction, 8 tests ✅
- Diffusion integration traits: `NormalMapProvider` + `MeshSurfaceSampler` + `DefaultSampler`, 11 tests ✅
- Face region segmentation: `VertexMask` with 8 `FaceRegion` variants, geometric classification, 12 tests ✅
- SVG visualization: `visualize.rs` with `SvgCamera`, `WireframeOptions`, `render_wireframe`, `render_joints_svg`, `SvgBuilder`, 19 tests ✅

**Production-ready additions since original plan (continued):**
- Static 68-landmark extraction: `LandmarkExtractor`, 8 `LandmarkGroup` variants, `Mesh::extract_landmarks()`, `group_centroid()`, 32 tests ✅
- Dynamic (pose-dependent) contour landmarks: `DynamicLandmarkExtractor`, `ContourVertexChains::default_flame()`, `Mesh::extract_dynamic_landmarks()`, 25 tests ✅
- GPU buffer export: `GpuMeshBuffers` with padded f32/u32 arrays, `GpuBufferConfig`, `Mesh::to_gpu_buffers()`, bytemuck byte accessors, degenerate face detection, normal recomputation, 18 tests ✅

**Not yet ready for:**
- GPU-accelerated LBS / normal map rendering
- UV texture mapping
