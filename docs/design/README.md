# OxiGAF Design Documents

This directory contains the original design plans and architecture documents for the OxiGAF project.

> **Note**: These are historical design documents. For current implementation status and TODOs, see the individual `crates/*/TODO.md` files.

## Overview

OxiGAF is a Pure Rust implementation of GAF (Gaussian Avatar Reconstruction from Monocular Videos via Multi-View Diffusion), targeting 100% Pure Rust with zero C/Fortran dependencies (following COOLJAPAN Pure Rust Policy).

## Document Index

### Master Plan
- **[IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md)** - Complete implementation roadmap
  - Status: ~75% complete
  - Major achievements: SIMD (2-4× speedup), Flash Attention (50% memory), Specialized SH shaders (10× speedup)
  - Critical gaps: Diffusion module needs Latent Upsampler, IP-Adapter, and CFG

### Module-Specific Plans

#### oxigaf-flame (85% complete)
- **[oxigaf-flame-plan.md](./oxigaf-flame-plan.md)** - FLAME parametric head model
  - ✅ Core LBS: 100%
  - ✅ Normal map rendering (CPU): 100%
  - ✅ SIMD optimizations: 100% (not in original plan)
  - ✅ Parallel batch processing: 100% (not in original plan)
  - ⬜ Sequence loading: 0%
  - ⬜ safetensors support: 0%

#### oxigaf-diffusion (65% complete)
- **[oxigaf-diffusion-plan.md](./oxigaf-diffusion-plan.md)** - Multi-view diffusion inference
  - ✅ Core U-Net, VAE, CLIP, Scheduler: 100%
  - ✅ Flash Attention: 100% (not in original plan)
  - ⬜ **Latent Upsampler: 0% (CRITICAL)**
  - ⬜ **IP-Adapter: 0% (CRITICAL)**
  - ⬜ **Classifier-Free Guidance: 0% (CRITICAL)**
  - ⬜ Weight conversion script: 0%

#### oxigaf-render (80% complete)
- **[oxigaf-render-plan.md](./oxigaf-render-plan.md)** - 3D Gaussian Splatting rasterizer
  - ✅ Forward pass: 100%
  - ✅ Backward pass: 90%
  - ✅ Specialized SH shaders: 100% (not in original plan)
  - ✅ Buffer pool: 100% (not in original plan)
  - ⬜ **Gradient verification: 0% (CRITICAL)**
  - ⬜ FLAME binding backward shader: Partial

#### oxigaf-trainer (90% complete)
- **[oxigaf-trainer-plan.md](./oxigaf-trainer-plan.md)** - Training pipeline (if exists)
  - ✅ Training loop: 100%
  - ✅ Loss functions (including LPIPS): 100%
  - ✅ TensorBoard integration: 100% (not in original plan)
  - ✅ Adaptive density control: 100%
  - ⬜ End-to-end integration: Blocked by diffusion gaps

#### oxigaf-cli (85% complete)
- **[oxigaf-cli-plan.md](./oxigaf-cli-plan.md)** - Command-line interface (if exists)
  - ✅ Train/render/export/convert commands: 100%
  - ✅ Benchmark/doctor/cache commands: 100% (not in original plan)
  - ✅ HuggingFace Hub integration: 100% (not in original plan)
  - ⬜ Video extraction: 0%
  - ⬜ Real-time preview: 0%

#### oxigaf (meta crate) (100% complete)
- **[oxigaf.md](./oxigaf.md)** - Meta crate architecture
  - ✅ Re-exports: 100%
  - ✅ Unified error handling: 100% (not in original plan)
  - ✅ Comprehensive prelude: 100% (not in original plan)
  - ✅ Feature flag orchestration: 100%

#### Workspace Architecture (90% complete)
- **[oxigaf-workspace-plan.md](./oxigaf-workspace-plan.md)** - Workspace structure (if exists)
  - ✅ Virtual workspace: 100%
  - ✅ Dependency management: 100%
  - ✅ Feature flag design: 100%

## Implementation Status Summary

### Overall Progress: ~75% Complete

#### What's Production-Ready
- ✅ FLAME mesh generation and normal map rendering
- ✅ 3DGS forward pass rendering
- ✅ Core diffusion components (U-Net, VAE, CLIP, Scheduler)
- ✅ Training pipeline (optimizer, losses, metrics, density control)
- ✅ CLI commands (train, render, export, benchmark, doctor)
- ✅ Meta crate with unified error handling

#### Critical Gaps for v1.0 (3-4 weeks)
1. **Diffusion Module** (~2 weeks):
   - Latent Upsampler (512×512 output)
   - IP-Adapter (identity preservation)
   - Classifier-Free Guidance (quality)
   - Weight conversion script

2. **Render Module** (~1 week):
   - Gradient verification (finite-difference tests)
   - FLAME binding backward shader completion

3. **CLI Module** (~1 week):
   - Video frame extraction (ffmpeg-next)
   - Real-time preview window (winit)

### Key Achievements Beyond Plan

1. **Performance Optimizations**:
   - SIMD acceleration in FLAME (2-4× speedup)
   - Flash Attention in diffusion (50% memory reduction)
   - Specialized SH shaders in render (10× speedup)
   - Buffer pool in render (90% allocation reduction)

2. **Developer Experience**:
   - TensorBoard integration (1,181 lines)
   - LPIPS in Pure Rust (689 lines, no Python)
   - Comprehensive CLI (benchmark, doctor, cache commands)
   - HuggingFace Hub integration
   - Unified error handling across all crates

3. **Code Quality**:
   - 573 tests (all passing)
   - No unwrap policy enforced
   - All files under 2000 lines
   - 24,661 lines of Rust code

## How to Use These Documents

### For New Contributors
1. Start with **IMPLEMENTATION_PLAN.md** for the big picture
2. Read the module-specific plan for your area of interest
3. Check the corresponding `crates/*/TODO.md` for current status
4. Look at implementation deviations in the status headers

### For Understanding Architecture
- Design documents explain **why** decisions were made
- TODO files track **what** remains to be done
- Status headers bridge the **gap** between plan and reality

### For Tracking Progress
- Status headers show implementation percentage
- "Key Achievements" list exceeds expectations
- "Not Yet Implemented" shows remaining work
- Check `crates/*/TODO.md` for day-to-day updates

## Design Philosophy

### COOLJAPAN Pure Rust Policy
- No C/Fortran dependencies (default features)
- Replace openblas → oxiblas
- Replace bincode → oxicode
- Replace rustfft → oxifft
- Replace z3 → oxiz
- Replace zip → oxiarc-archive ✅ (completed 2026-02-09)

### Code Quality Standards
- No unwrap policy (use proper error handling)
- No expect in library code
- File size limit: < 2000 lines
- Workspace version management (*.workspace = true)
- Comprehensive testing (573 tests)

### Feature Flag Design
- Default: Pure Rust, CPU-only
- Optional: cuda, metal (GPU backends)
- Optional: simd, parallel (performance)
- Optional: flash_attention (memory efficiency)
- Optional: gpu_debug (debugging)

## Related Documentation

- **Root README.md** - Project overview and quick start
- **CHANGELOG.md** - Version history and release notes
- **Crate READMEs** - Individual `crates/*/README.md` for API docs
- **TODO Files** - Current status in `crates/*/TODO.md`

## Timeline

### Phase 1: Foundation (Weeks 1-4) - ✅ COMPLETE
- FLAME model loading and LBS
- Normal map rendering
- CI pipeline setup

### Phase 2: Renderer (Weeks 5-10) - ✅ COMPLETE
- wgpu setup and forward pass
- Radix sort and tile assignment
- U-Net implementation
- Cross-view attention

### Phase 3: Optimization (Weeks 11-16) - 🚧 80% COMPLETE
- Backward rasterization (done)
- Gradient verification (pending)
- Loss functions (done, with extra LPIPS)
- FLAME mesh binding (partial)

### Phase 4: GAF Integration (Weeks 17-22) - 🚧 75% COMPLETE
- Training loop (done)
- Adaptive density control (done)
- CLI commands (done, with extras)
- Missing: Latent Upsampler, IP-Adapter, CFG

### Estimated Completion: 3-4 weeks from 2026-02-09

## Questions and Feedback

For questions about design decisions or implementation status:
1. Check the design document's status header
2. Review the corresponding TODO.md file
3. Look for "Open Questions" sections in design docs
4. Check the Risk Registry in IMPLEMENTATION_PLAN.md

## License

All design documents are part of the OxiGAF project.
Copyright © COOLJAPAN OU (Team Kitasan)

---

*Last updated: 2026-02-09*
*Documents reorganized: 2026-02-09*
