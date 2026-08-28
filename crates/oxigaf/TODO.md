# TODO for oxigaf (meta crate)

Status: 0.1.2. This file tracks the `oxigaf` meta-crate only (the facade over
`oxigaf-flame` / `oxigaf-diffusion` / `oxigaf-render` / `oxigaf-trainer`) —
see each sub-crate's own `TODO.md` for its scope.

## Completed

### Core responsibilities (docs/design/IMPLEMENTATION_PLAN.md)
- [x] Re-export the unified public API:
  `pub use oxigaf_flame as flame`, `oxigaf_diffusion as diffusion`,
  `oxigaf_render as render`, `oxigaf_trainer as trainer`
- [x] Single entry point for the ecosystem — all four sub-crates reachable
  via `oxigaf::*`, plus a `prelude` module for the common types

### Unified error handling
- [x] `OxigafError` enum wrapping all four sub-crate errors via `#[from]`
- [x] `Result<T>` alias, plus `FlameResult` / `DiffusionResult` /
  `RenderResult` / `CliResult` (all currently identical aliases for
  `Result<T, OxigafError>`, kept distinct for call-site readability)
- [x] `ErrorContext` trait (`with_ctx` / `with_ctx_fn`) wrapping an error with
  a context message via `OxigafError::Context`
- [x] `wgpu::RequestAdapterError` converts into `OxigafError::GpuError`;
  device-creation failures stay on the `OxigafError::Render` route instead
  (see the doc comment on `OxigafError::GpuError`)

### Pipeline module (`src/pipeline.rs`)
- [x] `PipelineBuilder` / `PipelineConfig` — fluent builder with validation
  (`flame_model_path`, `output_dir`, `num_views`, `iterations`)
- [x] `export` and `render_from_file` are now real implementations (0.1.2).
  Through 0.1.1 both were no-op stubs — `export`'s own doc comment called it
  "a thin validation wrapper" that only checked `model_path.exists()`
  (`let _ = (output_path.as_ref(), format); Ok(())`), and `render_from_file`
  did the equivalent. Now:
  - `export` loads a `.ply`/`.safetensors` Gaussian model and writes PLY
    (`GaussianModel::save_ply`), Wavefront OBJ (new point-cloud writer), or
    glTF 2.0 (`oxigaf_render::gltf::write_gltf`)
  - `render_from_file` loads the model, auto-frames a camera from its
    bounding box, rasterizes with `Rasterizer`, and saves the image
- [x] `verify_assets` checks the exact `.npy` file set
  `oxigaf_flame::io::load_flame_model` reads, via the shared
  `oxigaf_flame::io::REQUIRED_NPY_FILES` constant. Previously it checked a
  separately hand-maintained list with wrong names
  (`shape_dirs.npy`/`J_regressor.npy` instead of the loader's
  `shapedirs.npy`/`j_regressor.npy`) and omitted `lbs_weights.npy` entirely,
  so a directory missing the skinning weights was reported as complete.
- [x] `quick_train` — validates a `PipelineConfig` and resolves the output
  directory. Does **not** itself invoke `oxigaf_trainer::Trainer`.
- [x] `check_gpu` — synchronous wgpu adapter enumeration (`Vec<GpuInfo>`)
- [x] `detect_best_backend` — compile-time target-OS backend guess
  (`"Metal"` / `"Vulkan"` / `"Dx12"` / `"Gl"`)
- [x] `export`, `render_from_file`, and `quick_train` each take two
  independent generic path parameters (`P: AsRef<Path>`, `Q: AsRef<Path>`)
  instead of forcing both path arguments to the same concrete type

### Feature flag orchestration
- [x] Pass-through features to sub-crates: `simd`, `parallel`,
  `flash_attention`, `mixed_precision`, `npz`, `gpu_debug`
- [x] Convenience bundles: `full_performance`, `all_features`
- [x] `tests/feature_forwarding_tests.rs` asserts that `parallel`,
  `flash_attention`, `mixed_precision`, `npz`, and the `oxigaf-diffusion`
  half of `gpu_debug` actually reach the sub-crate they name, in both the on
  and off configuration (`simd` and the `oxigaf-render` half of `gpu_debug`
  are unasserted — neither exposes a marker observable without nightly Rust
  or a live GPU adapter)

### Documentation
- [x] Crate-level rustdoc: Quick Start, data-flow diagram, feature-flag
  tables, version-compatibility matrix, GPU requirements, module
  responsibilities, migration-from-Python table
- [x] README.md documents the real `pipeline` module API (`export`,
  `render_from_file`, `quick_train`, `verify_assets`, `check_gpu`), including
  the 0.1.2 stub-to-real change
- [ ] Mirror README's pipeline-API section into lib.rs's crate-level
  rustdoc — its "Key Features" list still names only FLAME / diffusion /
  rasterizer / training pipeline and never mentions the `pipeline` module,
  `export`, or `render_from_file`

### Examples — 7, all compiling against the current API
(`cargo check -p oxigaf --examples --all-features`)
- [x] `basic_flame.rs` — FLAME model loading and normal map rendering
- [x] `gaussian_render.rs` — GPU rasterization
- [x] `training_loop.rs` — full training pipeline
- [x] `diffusion_inference.rs` — multi-view diffusion inference
- [x] `end_to_end_pipeline.rs` — `PipelineBuilder` and the pipeline
  convenience functions, end to end
- [x] `custom_loss.rs` — custom Charbonnier loss alongside the built-in
  trainer losses
- [x] `checkpoint_lifecycle.rs` — checkpoint save/load with state restoration

### Testing — 67 tests, all passing
(`cargo nextest run -p oxigaf --all-features`)
- [x] `src/lib.rs` unit tests (21) — error conversion, prelude re-exports,
  `ErrorContext`, result aliases, the `RequestAdapterError -> GpuError`
  conversion
- [x] `src/pipeline.rs` unit tests (22) — `verify_assets`, `export`
  (PLY/OBJ/glTF, including regression tests for the former stub behaviour),
  `render_from_file` (including a malformed-model regression test), camera
  framing, backend detection, and the mixed-path-type regression test
- [x] `tests/api_tests.rs` (16) — black-box pipeline API tests
- [x] `tests/feature_forwarding_tests.rs` (8) — feature-flag forwarding

### Code quality
- [x] `#![deny(clippy::unwrap_used)]`
- [x] No `todo!()` / `unimplemented!()` remaining in `src/`

## In Progress

None currently.

## Planned (potential enhancements)

### Additional examples
- [ ] `multi_gpu.rs` — multi-GPU training example, once `oxigaf-trainer`
  supports multi-GPU

### Documentation enhancements
- [ ] Architecture / module-dependency diagram
- [ ] Performance-tuning guide (feature-flag combinations, hardware-specific
  recommendations, benchmark results)
- [ ] FAQ / troubleshooting section

### Feature flag improvements
- [ ] Feed `detect_best_backend`'s guess into actual backend selection —
  today it only reports a compile-time guess and nothing reads it back
- [ ] `profile` feature for timing/memory instrumentation

## Future Enhancements

- [ ] Async versions of the pipeline convenience functions (Tokio
  integration) — today `check_gpu` and `render_from_file` block
  synchronously on wgpu's async APIs internally via `pollster::block_on`
- [ ] FFI / Python (PyO3) / WASM bindings — none of the sub-crates expose
  these today
- [ ] REST/gRPC server wrapper for remote inference

## Current Status

- Re-exports, unified error handling, prelude, and feature-flag
  orchestration are complete for the crate's scope as a facade over
  `oxigaf-flame` / `oxigaf-diffusion` / `oxigaf-render` / `oxigaf-trainer`.
- `pipeline` module: `export` and `render_from_file` became real end-to-end
  implementations in 0.1.2 (previously no-op stubs — see Completed above).
  `quick_train` remains config validation only by design; it does not invoke
  `oxigaf_trainer::Trainer` itself.
- 67/67 tests passing (`cargo nextest run -p oxigaf --all-features`), 7/7
  examples compiling against the current API
  (`cargo check -p oxigaf --examples --all-features`), zero
  `todo!()`/`unimplemented!()` in `src/`.
- No known regressions in this crate as of 0.1.2.

## Notes

- MSRV: Rust 1.87 (workspace floor — `usize::is_multiple_of`, stabilized in
  1.87, is used in `oxigaf-bridge` and in `oxigaf-render`'s
  gradient-verification tests; `clippy::incompatible_msrv` flags several
  other 1.87 APIs used across `oxigaf-flame`).
- `oxigaf` itself has no direct C/C++/Fortran dependency; see README.md's
  "What 'pure Rust' means here" for the workspace-wide caveat about the
  offline FLAME/PyTorch conversion scripts.
- One-person project — contributions welcome via issues/PRs.
