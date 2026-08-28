# TODO for oxigaf-flame

**Current release: v0.1.2.** Provenance tags below such as "(v0.1.1)" mark
when a feature shipped, not the current version — this document tracks what
is done and what remains, not a version-by-version changelog (see
`CHANGELOG.md` at the workspace root for that). Contributions welcome.

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
- ✅ **GazeController** — comprehensive gaze control system (`gaze_controller/`):
  Listing's-law gaze rotation (`gz_listing_rotation`/`gz_listing_axis`),
  I-VT saccade/fixation classification (`gz_detect_saccades`/
  `gz_detect_fixations`), natural blink detection/synthesis
  (`gz_detect_blinks`/`gz_synthesize_blinks`), vergence estimation
  (`gz_vergence_from_iod`/`gz_convergence_angle_deg`), and ring-buffered
  history/statistics via `GazeController`/`GazeStats`. **v0.1.2:** split
  from a single file into a module directory and gained
  `GazeController::synthesize_blinks`, a config-aware convenience wrapper
  around `gz_synthesize_blinks`.
- ✅ **HeadTracker** — real-time head pose tracking integration
- ✅ **HeadGeometry** — head geometry analysis and symmetry tools
- ✅ **PoseEstimation** — camera-relative head pose estimation
- ✅ **PosePrior** — learned pose prior for regularisation
- ✅ **Expressions** — FLAME expression blend shape interface
- ✅ **ExpressionAnimation** — keyframe expression animation
- ✅ **ExpressionClustering** — cluster expressions into prototypes
- ✅ **ExpressionTransfer** — transfer expressions between identities
- ✅ **Expression-space retargeting** — `expression_retargeting.rs`:
  `LinearExpressionRetargeter` learns a linear map between two identities'
  expression spaces, plus trajectory analysis (velocity/acceleration/
  smoothing/resampling) and blending utilities (weighted blend, SLERP).
  Distinct from `ExpressionTransfer` above (direct/scaled/style transfer of
  individual expressions) and from `retargeting.rs`'s `ExpressionRetargeter`
  (shape-*identity* retargeting, not expression-space retargeting) —
  three related but separate modules, not duplicates.
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
- ✅ **Geodesic distance** — Dijkstra and heat-method distance computation
  (`geodesic/`). **v0.1.2:** `heat_geodesic` is now a real implementation of
  the heat method (Crane, Weischedel & Wardetzky 2013) — solving
  `(M + t·Lc)u = δ_source`, normalizing `∇u`, then a Poisson solve, both via
  Jacobi-preconditioned CG — replacing what its own v0.1.1 doc comment
  called "a simplified approximation, not the full heat method." Also new:
  `heat_geodesic_multi` (multi-source), `heat_time_step` (standard `dt`
  heuristic), and `geodesic_center_sampled`/`DEFAULT_CENTER_SAMPLES = 64`.
  **Behavior change:** `geodesic_center` with an empty candidate list now
  farthest-point-samples 64 candidates by default instead of searching every
  vertex exhaustively (which took minutes on a 5023-vertex head); pass
  `(0..mesh.n_vertices()).collect()` for the old exhaustive behavior.
- ✅ **Spectral analysis** — Laplace-Beltrami spectral decomposition
- ✅ **Multiresolution mesh** — Garland-Heckbert Quadric Error Metric (QEM)
  edge-collapse decimation for LOD generation (`multiresolution.rs`) — this
  is decimation-based, not wavelet-based. **v0.1.2:**
  `DecimationConfig::default().target_vertex_count` changed from `0`
  (always rejected by validation — an always-reject footgun) to
  `usize::MAX` (a documented no-op default: decimation only runs while
  `live_vertex_count > target_vertex_count`).
- ✅ **Statistical shape model** — PCA shape space with sampling (`statistical_shape_model.rs`)
- ✅ **Shape-space statistics** — `shape_analysis.rs`: distance metrics,
  descriptive statistics, PCA via power iteration, outlier detection, and
  shape interpolation over the FLAME shape parameter space. Distinct from
  `statistical_shape_model.rs` above (which builds/samples the shape space
  itself) and from `mesh_analysis.rs` (geometric mesh metrics, not
  shape-parameter statistics).
- ✅ **Symmetry detection** — bilateral symmetry plane estimation

### UV & Texture Pipeline (v0.1.1)
- ✅ **UV parameterisation** — least-squares conformal mapping (`uv.rs`:
  `UvAccessor`, `UvMeshExt`, `UvChartInfo`)
- ✅ **UV texture sampling** — `uv_texture.rs`: `TextureMap`,
  `UvTextureSampler` (nearest/bilinear filtering, clamp/repeat/mirror
  wrapping), `TextureMeshExt` for sampling at barycentric surface points
- ✅ **Texture baking** — render-to-texture from 3D attributes
- ✅ **Face atlas generation** — packed UV atlas for head mesh
- ✅ **Albedo map** — diffuse colour texture extraction
- ✅ **SH lighting model** — spherical harmonic environment lighting
  (`lighting_model.rs`). **v0.1.2:** `LightingError` gained
  `InvalidFaceIndex { face, index, vertices }` (a face referencing an
  out-of-range vertex is now reported instead of the caller getting a
  panic or silently-wrong output), and `shade_mesh_directional`/
  `shade_mesh_multi_light` now call the existing light-colour-range check
  and return `Err` for a component outside `[0, 1]` instead of proceeding
  unvalidated.

### Motion, Fitting & Utilities (v0.1.1)
- ✅ **Timeline** — frame-indexed parameter sequence
- ✅ **Warp field** — per-vertex displacement field animation
- ✅ **Shape retargeting** — identity-shape transfer retargeting
- ✅ **Landmark-to-FLAME fitting** — `fitting/`: CPU gradient-descent fitting
  of FLAME parameters (shape/expression/global rotation/translation/jaw) to
  2D landmark observations. `PinholeCamera` for 3D→2D projection, a
  `FlameForward` trait decouples the optimizer from the full model
  (`MockFlameForward` for fast deterministic tests), `fit_landmarks` runs
  steepest descent with a central-difference-estimated gradient (plain
  gradient descent, not Gauss-Newton — no residual Jacobian, normal
  equations, or damping term). **v0.1.2:** added `FlameLandmarkFitter` (a
  landmark-specialised forward pass keeping the finite-difference loop
  proportional to landmark count rather than vertex count); fixed
  `FittingError::CameraProjectionFailed` to be actually reachable
  (previously declared but never constructed — dead code — so a fit where
  zero landmarks project in front of the camera silently proceeded on a
  degenerate projection); added `FittingResult::n_visible_landmarks: usize`.
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
- ✅ **Export with UV coordinates (if loaded)** — `MeshExportConfig::export_uv`
  (default `true`) writes `vt` lines + `f v/vt/vn` faces in OBJ and `s`/`t`
  vertex properties in PLY whenever `uv_coords` matches the vertex count;
  set it `false` to omit UVs even when present.

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
- ✅ **UV coordinate interpolation and texture sampling** (v0.1.1) — see
  "UV & Texture Pipeline (v0.1.1)" above (`uv.rs`, `uv_texture.rs`,
  `texture_baking.rs`, `face_atlas.rs`).
  - ⬜ **Remaining gap:** loading FLAME's own official `texture_data_*.npy`
    files specifically — the infrastructure above samples/bakes textures
    generically but has no loader for that particular FLAME asset format.
- ✅ **`io::REQUIRED_NPY_FILES`** — new in v0.1.2: the single source of truth
  for the 8 `.npy` filenames `load_flame_model` requires. Previously
  `oxigaf`'s `verify_assets` hardcoded its own, disagreeing list that used
  the wrong names for 3 files and omitted `lbs_weights.npy` entirely, so a
  model directory missing the skinning weights was reported as complete;
  it now iterates this constant instead.
- ✅ **estimate_pitch_from_vertical** — reference-based geometric foreshortening
  solve in `pose_estimation.rs` (`PitchReference` supplies the model-space
  upper/lower landmark-group centroids plus a weak-perspective scale; the
  function solves `θ = ±acos(Δy_cam / amplitude) − φ` from the observed
  vertical separation). It returns **both** candidate angles rather than
  picking one: a single foreshortening measurement genuinely cannot
  distinguish them (they collapse to `±θ` when the two groups share a model
  depth). Use `select_pitch_candidate` with a prior (e.g. the previous
  frame's pitch) or `estimate_pose_weak_perspective` on the full landmark
  set to resolve the ambiguity.
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
- ✅ **Multi-resolution mesh decimation** — `multiresolution.rs`'s
  `MeshDecimator`/`DecimationConfig` (QEM edge collapse) produce a decimated
  mesh at any target vertex count on demand, covering the "1K/2.5K/10K
  variants" use case generally rather than as fixed pre-baked assets.
  - ⬜ **Remaining gap:** adaptive LOD *selection* at render time based on
    camera distance — this crate provides the decimator, not a runtime LOD
    switcher.
- ✅ **Cached computation results**
  - `joint_cache: Mutex<HashMap<u64, Vec<[f32; 3]>>>` in `FlameModel`
  - `compute_shape_hash(shape) -> u64` (FNV-style hash)
  - `joint_positions_cached()` private method with 64-entry eviction policy
  - 8 memoization tests

### Usability
- ⬜ **FLAME model auto-download utility**
  - Download from MPI server (requires agreement)
  - Cache in `~/Library/Caches/oxigaf/flame/` (macOS; `~/.cache/oxigaf/flame/` on
    Linux, `%LOCALAPPDATA%\oxigaf\flame\` on Windows — see `oxigaf-cli`'s
    `default_cache_dir()`, which already implements this layout for other
    asset downloads via `dirs::cache_dir()`)
  - Checksum verification
- ✅ **Parameter interpolation utilities** (`params.rs`)
  - `FlameParams::lerp(other, t)` — linear interpolation for shape/expression/translation, quaternion slerp for pose axis-angle
  - `FlameParams::slerp_pose(other, t)` — convenience method returning only interpolated pose
  - Private helpers: `axis_angle_to_quat`, `quat_slerp`, `quat_to_axis_angle`
  - Handles: near-zero rotation, antiparallel quaternions, dimension mismatch errors
  - 19 new tests including 2 proptest property-based tests
- ✅ **Expression library (placeholder presets)** —
  `ExpressionLibrary::placeholder_expressions()` (`expressions.rs`; renamed
  from `default_expressions()` in v0.1.2, which is now `#[deprecated]`)
  provides named presets: smile, grin, frown, surprised, angry, sad,
  disgusted, fearful, winking, open_mouth. **Caveat, by design:** these are
  hand-authored, illustrative coefficients — not fitted to any FLAME
  expression basis — tagged `ExpressionProvenance::Placeholder` so callers
  can detect this at runtime via `has_placeholders()`.
  - ⬜ **Remaining gap:** sourcing real, fitted coefficients from a canonical
    expression dataset. The fitting path already exists
    (`NamedExpression::fit_to_basis` + `ExpressionLibrary::to_json_string`/
    `from_json_file`); only the canonical reference data is missing.

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

### Implementation: ~98% complete
- ✅ Core LBS algorithm: 100%
- ✅ Normal map rendering: 100% (CPU); 0% (GPU — not implemented in this
  crate, see `oxigaf-render` for the GPU rasterizer)
- ✅ Mesh sampling: 100%
- ✅ SIMD optimizations: 100% (functionality complete; only exercised when
  built with `simd` on a nightly compiler — see Statistics below)
- ✅ Parallel processing: 100%
- ✅ Model format support: 100% (.npy, .safetensors always available;
  `FlameSequence` NPZ loading behind the `npz` feature — reader only, no
  writer)
- ✅ Sequence loading: 100% (FlameSequence with LRU cache)
- ✅ Mesh export: 100% (OBJ + PLY binary LE, including UV coordinates when
  present — `MeshExportConfig::export_uv`)
- ✅ Static + dynamic landmarks: 100% (`landmarks.rs`, `dynamic_landmarks.rs`)
- ✅ GPU buffer export: 100% (`gpu_buffers.rs`, padded f32/u32 buffers, bytemuck)
- ✅ Avatar rigging & expression system: 100% (`avatar_rig.rs` plus 8
  expression-related modules — presets are illustrative placeholders, not
  fitted; see the "Expression library" caveat above)
- ✅ Mesh processing suite: 100% (boolean/repair/smoothing/subdivision/
  morphing/analysis, plus QEM decimation)
- ✅ Geometry & statistical tools: 100% (`heat_geodesic` is the real
  Crane et al. heat method as of v0.1.2, not an approximation; spectral
  analysis, symmetry detection, shape-space statistics)
- ✅ UV & texture pipeline: ~95% (parameterisation, sampling, atlas,
  baking, albedo, SH lighting all done; loading FLAME's official
  `texture_data_*.npy` files specifically is not — see above)
- ✅ Pose estimation & landmark fitting: 100%
  (`FittingError::CameraProjectionFailed` now reachable as of v0.1.2)
- ✅ Gaze control: 100% (`gaze_controller/`)
- ⬜ GPU-accelerated LBS: 0% (not implemented in this crate)
- ⬜ WASM compatibility: 0% (not attempted)

### Tests: 2,589 passed, 0 failed

Measured via `cargo nextest run -p oxigaf-flame --all-features --no-fail-fast`
(2,581 with default features — the gap is `parallel`/`npz`-gated tests).
`tests/simd_tests.rs` needs both the `simd` feature and a `nightly` compiler,
so it contributes 0 either way on the stable toolchain this was measured
with. Doctests run separately and aren't included in the 2,589 above: 39
passed via `cargo test --doc -p oxigaf-flame --all-features`.

The category breakdown below is the original v0.1.0 core count (280) and
predates the much larger v0.1.1/v0.1.2 feature set (mesh processing,
geometry/statistical tools, UV/texture pipeline, motion/fitting utilities,
avatar rigging, gaze control — see "Completed (v0.1.1)" and
"Completed (from plan)" above), which accounts for most of the growth to
2,589 and is not broken out per-category here (the "Doc tests: 16" row
below is this same historical v0.1.0 snapshot — current doc tests: 39, above):

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
| UV mapping | ⬜ Optional | ✅ **Done** (v0.1.1) | `uv.rs` (LSCM parameterisation), `uv_texture.rs` (sampling), `texture_baking.rs`, `face_atlas.rs`; loading FLAME's official `texture_data_*.npy` files specifically remains ⬜ |
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
8. ✅ ~~UV mapping~~ — Done (v0.1.1: `uv.rs`, `uv_texture.rs`,
   `texture_baking.rs`, `face_atlas.rs`); loading FLAME's official
   `texture_data_*.npy` files specifically remains a gap
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

**Production-ready additions since original plan (v0.1.2):**
- Real heat-method geodesics: `heat_geodesic` now implements Crane, Weischedel & Wardetzky (2013) instead of the v0.1.1 approximation, plus `heat_geodesic_multi`/`heat_time_step`/`geodesic_center_sampled` ✅
- Landmark-fitting correctness: `FittingError::CameraProjectionFailed` actually reachable, `FittingResult::n_visible_landmarks`, and the new `FlameLandmarkFitter` for landmark-proportional (not vertex-proportional) fitting cost ✅
- Lighting validation: `LightingError::InvalidFaceIndex`, and `shade_mesh_directional`/`shade_mesh_multi_light` now reject out-of-range light colours instead of proceeding unvalidated ✅
- Decimation default fix: `DecimationConfig::default()` no longer always rejects (`target_vertex_count` `0` → `usize::MAX`) ✅
- Honest expression-preset naming: `ExpressionLibrary::placeholder_expressions()` (deprecates `default_expressions()`) ✅
- Asset verification correctness: `io::REQUIRED_NPY_FILES` as the shared source of truth consumed by `oxigaf`'s `verify_assets` (previously checked the wrong filenames and missed `lbs_weights.npy`) ✅

**Not yet ready for:**
- GPU-accelerated LBS / normal map rendering (not implemented in this crate — see `oxigaf-render`)
- Loading FLAME's official `texture_data_*.npy` texture files directly (generic UV/texture sampling infrastructure exists; see "UV & Texture Pipeline" above)
- WASM builds (not attempted)
