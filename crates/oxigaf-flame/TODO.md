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
- ✅ 43 unit and integration tests
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

## 📋 Planned (future versions)
- ⬜ **OBJ export** (`Mesh::export_obj()`)
- ⬜ **PLY export** (`Mesh::export_ply()`)
- ⬜ Support for exporting with UV coordinates (if loaded)

### Spatial Search
- ⬜ **KD-tree integration** (using `kiddo` crate)
  - `Mesh::build_kdtree()` for nearest-vertex queries
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
- ⬜ **Dynamic landmarks** (pose-dependent contours)
  - Load `flame_dynamic_embedding.npy`
  - Landmark position computation
- ⬜ **Vertex masks** (face region segmentation)
  - Load `FLAME_masks.pkl`
  - Semantic region queries (forehead, nose, lips, eyes, etc.)
- ⬜ **Static landmarks** (51 3D face landmarks)
  - Load `flame_static_embedding.pkl`
  - Barycentric coordinate-based landmark extraction

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
- ⬜ **Thread safety tests** (verify `Send + Sync` for `FlameModel`)
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
- ⬜ **Cached computation results**
  - Memoize joint positions for repeated shape parameters
  - Cache blend shape results

### Usability
- ⬜ **FLAME model auto-download utility**
  - Download from MPI server (requires agreement)
  - Cache in `~/.cache/oxigaf/flame/`
  - Checksum verification
- ⬜ **Parameter interpolation utilities**
  - Lerp/slerp between FlameParams
  - Smooth transitions for animation
- ⬜ **Expression library**
  - Predefined expressions (smile, frown, surprise, etc.)
  - Load from canonical expression dataset

### Integration
- ⬜ **Trait for oxigaf-diffusion integration**
  - `NormalMapProvider` trait (planned in design)
  - `MeshSurfaceSampler` trait (planned in design)
- ⬜ **Mesh deformation shader helpers**
  - Export mesh binding data to GPU buffers
  - TBN matrix computation utilities

### Developer Tools
- ⬜ **Visualization utilities**
  - Mesh wireframe rendering
  - Joint skeleton visualization
  - Blend shape visualization (morph targets)
- ⬜ **CLI tool** (`flame-tool`)
  - Convert `.pkl` → `.npy` / `.safetensors`
  - Render normal maps from command line
  - Validate model files
  - Export meshes

## 🐛 Known Issues

Currently none reported.

## 📊 Current Status

### Implementation: ~85% complete
- ✅ Core LBS algorithm: 100%
- ✅ Normal map rendering: 100% (CPU), 0% (GPU)
- ✅ Mesh sampling: 100%
- ✅ SIMD optimizations: 100%
- ✅ Parallel processing: 100%
- ⬜ Model format support: 50% (.npy ✅, .safetensors ❌)
- ⬜ Sequence loading: 0%
- ⬜ Mesh export: 0%
- ⬜ Advanced FLAME features: 0%

### Tests: 124 tests (all passing)
- ✅ Unit tests: ~28 (in `src/`)
- ✅ Integration tests: ~96 (in `tests/`)
- ✅ Property-based tests: Yes (proptest)
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
| Mesh export | ⬜ Planned | ⬜ Not started | Low priority |
| KD-tree | ⬜ Planned | ⬜ Not started | Not critical |
| UV mapping | ⬜ Optional | ⬜ Not started | Only needed for texture rendering |
| Landmarks | ⬜ Optional | ⬜ Not started | Not needed for GAF |
| Vertex masks | ⬜ Optional | ⬜ Not started | May be needed for Gaussian init |

## 🎯 Priority for v1.0

**High Priority:**
1. ✅ ~~Core LBS (DONE)~~
2. ✅ ~~Normal map rendering (DONE)~~
3. ✅ ~~safetensors support (DONE — v0.1.0)~~
4. ✅ ~~FlameSequence loader (DONE — v0.1.0)~~

**Medium Priority (future):**
5. ⬜ PLY export (for visualization)
6. ⬜ Vertex masks (if needed for Gaussian initialization)

**Low Priority:**
7. ⬜ GPU normal map renderer
8. ⬜ UV mapping
9. ⬜ Landmarks

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

**Not yet ready for:**
- End-to-end video pipeline (needs FlameSequence)
- Model weight sharing (needs safetensors)
- Mesh visualization (needs PLY export)
