# TODO for oxigaf-cli

**Status: v0.1.2** (released 2026-08-28) — 32 subcommands, 3105/3105 tests
passing (`cargo nextest run -p oxigaf-cli --all-features`), zero
`todo!()`/`unimplemented!()` stubs in `src/`. One-person project;
contributions welcome via PR.

## ✅ Completed (from plan)

### Command-Line Interface
- ✅ **train command** (alias: `reconstruct`)
  - Full reconstruction pipeline orchestration
  - Configuration priority: CLI args > env vars > project config > user config > defaults
  - Comprehensive environment variable support (OXIGAF_*)
  - Resume from checkpoint support
  - Early stopping with patience and min_delta
  - Preview image generation
  - Interactive mode with keyboard controls
  - Metrics export (CSV/JSON Lines)
  - TensorBoard integration
  - Training profiles (dev/prod/custom)
- ✅ **render command**
  - Load avatar from multiple formats (.safetensors, .ply, .json)
  - Multiple render modes (frames, turntable, orbit, dolly)
  - Camera trajectory JSON support
  - FLAME animation parameters
  - Quality presets (low/medium/high/ultra)
  - Multiple output formats (PNG, JPEG, EXR)
  - Configurable background color
- ✅ **export command**
  - Multiple export formats (PLY, safetensors, glTF, JSON, point-cloud,
    mesh, and `all` of the first four at once — see *Completed (v0.1.2)*
    below for the `all`/`point-cloud` additions)
  - PLY format variants: ASCII and binary little-endian both work;
    binary-big-endian is accepted by the CLI's parser but rejected by the
    writer at export time (`--ply-format binary-be` errors out — nothing in
    the 3DGS ecosystem emits or reads big-endian PLY)
  - Include training metadata
  - SH degree downsampling
  - Force overwrite option
- ✅ **convert command** (FLAME model conversion)
  - Convert .pkl to .npy format — **historical note:** at 0.1.0 the `.pkl`
    path (`convert_pkl` in `convert.rs`) was structurally unable to succeed
    against real FLAME `.pkl` files (only `.npz` actually worked, despite
    `convert` being listed as shipped). Fixed as of 0.1.2: `convert_pkl` now
    decodes the pickle stream with a pure-Rust virtual machine
    (`convert::pickle::load_arrays`, protocols 0–5, reconstructing
    `numpy.ndarray`/`numpy.dtype`/`chumpy.ch.Ch`/`scipy.sparse` payloads).
  - FLAME 2020/2023 version support
  - Optional UV coordinate inclusion
  - Output verification
  - Force overwrite
- ✅ **Asset management** (setup command)
  - Download model weights
  - Cache directory management — unified as of 0.1.2 via
    `commands::runtime::default_cache_dir()` (`dirs::cache_dir()`:
    `~/Library/Caches/oxigaf` on macOS, `~/.cache/oxigaf` /
    `$XDG_CACHE_HOME/oxigaf` on Linux, `%LOCALAPPDATA%\oxigaf` on Windows;
    override with `OXIGAF_CACHE_DIR`). Previously `setup`/`doctor`/`cache`
    each computed this path a different way, so `setup` and `cache list`
    could disagree about where the cache was. `SetupArgs::cache_dir` is
    `Option<PathBuf>` so it falls through to the shared default.
  - Checksum verification
  - Selective asset download
  - Offline mode
  - **HuggingFace Hub integration** (EXCEEDS PLAN) — as of 0.1.2, `hf-hub`
    (and the C OpenSSL / `ring` it pulled in) is dropped entirely;
    `assets.rs` talks to the HF Hub `resolve` endpoint directly over
    `ureq 3.4` + `rustls 0.23` with the pure-RustCrypto
    `oxitls-rustcrypto-provider`, and `download_file` streams with native,
    byte-accurate progress instead of shelling out to curl/wget.
    - Download from HF repositories
    - Authentication token support
    - Revision/branch/tag support
    - Filename specification

### Enhanced Features (EXCEEDS PLAN)
- ✅ **benchmark command** (not in plan)
  - Component-specific benchmarks (flame, raster, train, export, full)
  - Warmup iterations
  - Configurable iteration counts
  - Multiple output formats (human, JSON, CSV, markdown)
  - Baseline comparison
  - Model size presets (tiny/small/medium/large/xlarge)
- ✅ **doctor command** (not in plan)
  - GPU availability checks
  - FLAME model verification
  - Cache status inspection
  - Version checking
  - Component-specific diagnostics
- ✅ **cache management** (not in plan)
  - List cached assets with details
  - Clean old assets by age
  - Verify cache integrity
  - Print cache directory path
- ✅ **completions command** (not in plan)
  - Shell completion script generation
  - Support for bash, zsh, fish, PowerShell
  - Installation instructions in help text

### Configuration System (EXCEEDS PLAN)
- ✅ **Multi-level configuration** (1049 lines in config.rs)
  - TOML configuration file support
  - Environment variable overrides
  - CLI argument overrides
  - User config (~/.config/oxigaf/config.toml)
  - Project config (./oxigaf.toml)
  - Priority hierarchy: CLI > env > project > user > defaults
- ✅ **Configuration validation**
  - Field-by-field validation
  - Range checking
  - Type validation
  - Descriptive error messages
- ✅ **Comprehensive settings**
  - Training parameters (iterations, image size, views)
  - Optimizer learning rates (position, scale, rotation, opacity, SH)
  - Initialization parameters (SH degree, Gaussian counts)
  - Device configuration (backend, GPU index)
  - Output configuration (checkpoint interval, log interval, export format)

### Logging & Progress (EXCEEDS PLAN)
- ✅ **tracing integration** (264 lines log_rotation.rs)
  - Structured logging with levels (ERROR, WARN, INFO, DEBUG, TRACE)
  - File logging with rotation strategies (never, hourly, daily)
  - Multiple log formats (JSON Lines, pretty, compact)
  - Maximum file retention
  - Automatic old log cleanup
- ✅ **indicatif progress bars** (935 lines progress.rs)
  - Training iteration progress (`TrainingProgress` with multi-bar via Arc<MultiProgress>)
  - `OperationSpinner` (indeterminate spinner), `BatchProgress` (counted bar)
  - `TimingReport` (HashMap<String,Duration> with record/time/total/percentage/format_table)
  - Multi-bar support, ETA estimation, custom styling
  - 22 new tests
- ✅ **Verbosity control** (182 lines verbosity.rs)
  - Multiple verbosity levels (-v, -vv, -vvv)
  - Quiet mode
  - Level-based filtering

### Output & Reporting (EXCEEDS PLAN)
- ✅ **JSON output mode** (248 lines json_output.rs)
  - Machine-readable output for scripting
  - Suppresses all normal output
  - Valid JSON on stdout
  - Progress events
  - Error reporting
- ✅ **Metrics export** (366 lines metrics.rs)
  - CSV format
  - JSON Lines format
  - Per-iteration metrics
  - Training statistics
- ✅ **Training summary** (529 lines summary.rs)
  - Final statistics (loss, PSNR, Gaussian count)
  - Training time
  - Convergence info
  - Best checkpoint info
  - Showcase image paths

### Error Handling (EXCEEDS PLAN)
- ✅ **Comprehensive CliError enum** (243 lines error.rs)
  - Specific error variants for all failure modes
  - User-friendly error messages
  - Actionable suggestions
  - Exit code mapping
  - Configuration errors
  - I/O errors
  - GPU errors
  - Asset download errors
  - Training errors
  - Export errors
- ✅ **Exit codes**
  - EXIT_SUCCESS (0)
  - EXIT_CONFIG_ERROR (2)
  - EXIT_IO_ERROR (3)
  - EXIT_GPU_ERROR (4)
  - EXIT_ASSET_ERROR (5)
  - EXIT_TRAINING_ERROR (6)
  - EXIT_EXPORT_ERROR (7)
  - EXIT_INTERRUPTED (130)

### Pipeline Orchestration
- ✅ **Full reconstruction pipeline** (593 lines pipeline.rs)
  - FLAME model loading
  - Gaussian initialization
  - Trainer creation
  - Training loop with progress
  - Final export
  - Camera trajectory support
  - Result metadata collection

### Interactive Features (EXCEEDS PLAN)
- ✅ **Interactive training mode** (172 lines interactive.rs)
  - Keyboard controls during training
  - Pause/resume
  - Skip iterations
  - Save checkpoint on demand
  - Quit gracefully
  - InteractiveController API

### Utilities
- ✅ **Dry run mode** (213 lines dry_run.rs)
  - Validate inputs without executing
  - Check permissions
  - Verify GPU availability
  - Estimate resource usage
  - Report planned actions
- ✅ **Output formatting** (165 lines output.rs)
  - Colored terminal output
  - Structured formatting
  - Table printing
  - Progress spinners

### Testing
- ✅ **3105 tests total, all passing** (`cargo nextest run -p oxigaf-cli
  --all-features`; measured 2026-08-28) — up from 2375 at v0.1.1, tracking
  the crate's growth to 32 subcommands. Roughly 1900 are inline `#[cfg(test)]`
  unit tests spread across the ~150 files in `src/`; the remainder are
  focused integration-test binaries under `tests/`, including
  `cli_integration.rs` (~90 end-to-end tests), `config_hierarchy_tests`,
  `hf_hub_tests`, `json_output_tests`, `config_tests`, `log_rotation_tests`,
  `metrics_tests`, `interactive_tests`, `cache_tests`, and the `bin/oxigaf`
  binary-level tests (CLI wiring/consistency checks such as
  `cli_definition_is_internally_consistent` and
  `every_subcommand_documents_itself_for_completions`).
- ✅ Integration tests using assert_cmd
- ✅ Predicates for output validation
- ✅ Serial test execution for file I/O

### Code Quality
- ✅ No unwrap policy (`#![cfg_attr(not(test), deny(clippy::unwrap_used))]`
  in `lib.rs`; `#![deny(clippy::unwrap_used)]` in `main.rs`)
- ✅ No expect policy (same pattern with `clippy::expect_used`)
- ✅ All files under the 2000-line workspace policy limit (largest:
  `benchmark_suite.rs`, 1999 lines; several others — `scene_streaming.rs`,
  `camera_path_tool.rs`, `compare.rs`, `animation_export.rs` — sit close
  behind in the 1900s and are refactor-split candidates)
- ✅ Total: 154 files, ~86,600 lines of code per `tokei`  (measured
  2026-08-28; includes inline unit tests, which make up a large share of the
  line count — see Testing above). Grown roughly 13x since the "6,503 lines"
  figure recorded at v0.1.0, tracking the jump from ~12 to 32 subcommands.
- ✅ Clean module boundaries
- ✅ Comprehensive documentation

## ✅ Completed (v0.1.1)

### Export Suite (v0.1.1)
- ✅ **PLY export** — point cloud and Gaussian PLY export
- ✅ **glTF export** — Khronos glTF 2.0 scene export
- ✅ **Mesh export** — OBJ/STL mesh export
- ✅ **Point cloud export** — XYZ/PCD point cloud export
- ✅ **Video export** — frame-sequence to video
- ✅ **Animation export** — keyframe animation sequence JSON

### Analysis & Inspection (v0.1.1)
- ✅ **Scene analyser** — per-object statistics and diagnostics
- ✅ **Model inspector** — weight/parameter inspection tool
- ✅ **Diff tool** — scene or model diff comparison
- ✅ **Quality checker** — resolution, coverage, and artifact detection
- ✅ **Evaluation suite** — PSNR/SSIM/LPIPS batch evaluation

### Scene Operations (v0.1.1)
- ✅ **Scene merging** — merge multiple Gaussian scenes
- ✅ **Scene optimiser** — opacity pruning and densification
- ✅ **Scene streaming** — tile-based progressive scene streaming
- ✅ **Gaussian filter** — attribute-based Gaussian selection filter
- ✅ **Gaussian deduplicator** — spatial deduplication of Gaussians
- ✅ **Gaussian compressor** — k-means based Gaussian compression

### Visualisation & Monitoring (v0.1.1)
- ✅ **Arcball camera** — interactive orbit camera controller
- ✅ **Camera path editor** — spline-based camera path authoring
- ✅ **LOD generator** — level-of-detail preview rendering
- ✅ **Parallel renderer** — multi-view batch renderer
- ✅ **Live dashboard** — real-time training metrics dashboard
- ✅ **Training monitor** — GPU/CPU/memory training resource monitor
- ✅ **Resume analyser** — checkpoint resume state inspector
- ✅ **Parameter sweep** — grid/random hyperparameter sweep runner

### Reporting & Config (v0.1.1)
- ✅ **Experiment report** — auto-generated experiment summary
- ✅ **HTML report** — browser-viewable experiment report
- ✅ **Profiling report** — per-stage timing breakdown
- ✅ **Config presets** — built-in named configuration presets
- ✅ **Workspace manager** — multi-project workspace organisation

### Data & Geometry (v0.1.1)
- ✅ **Format converter** — multi-format scene conversion pipeline
- ✅ **Colour calibration** — colour chart calibration utility
- ✅ **Geometry tools** — mesh boolean and repair CLI commands
- ✅ **Dataset tools** — dataset ingestion and preprocessing pipeline
- ✅ **Point cloud registration** — ICP-based point cloud alignment

## ✅ Completed (v0.1.2)

### Command inventory now complete (32 subcommands, verified against `--help`)
This file previously tracked commands by feature category without keeping
an authoritative top-level list, so a few real, tested top-level commands
were never named as such anywhere in this document. Confirmed present via
`cargo run -p oxigaf-cli --bin oxigaf -- --help`:
- ✅ **`batch`** (`batch_processor.rs`) — run many model conversions as one
  dependency-ordered batch
- ✅ **`perf`** (`memory_estimator.rs` / CPU kernels) — micro-benchmark the
  CPU-side numeric kernels, distinct from the GPU-oriented `benchmark`
  command
- ✅ **`pipeline`** (`stages.rs`, `commands/pipeline_cmd.rs`) — run the
  reconstruction workflow one composable stage at a time
  (`plan`/`track`/`diffuse`/`export`/`status`); the training stage still
  needs `oxigaf train` since it requires a live GPU device/queue
- ✅ **`runs`** — create, list, prune and retire training run workspaces
  (distinct from `workspace`, which browses an existing run's checkpoints)
- ✅ **`training`** — analyse a finished training run (summary, smoothing,
  reports, resume recommendations, timing traces), distinct from the
  `train` command that runs one
- ✅ All 32 top-level commands cross-checked one-to-one against `--help`
  output; see `crates/oxigaf-cli/README.md` for the full list with
  one-line descriptions

### HTTP / cache-directory fixes
(See the `convert`/`setup` bullets above under *Completed (from plan)* for
the `.pkl` decoder and unified cache-directory fixes — both landed in
0.1.2.) Workspace-wide, this release also removed C Oniguruma (via the
`oxicandle-core` fork replacing upstream `candle-core`) and C OpenSSL/`ring`
(via the `oxigaf-cli` HTTP-stack change above) from the default build;
`cargo deny check bans` now passes with both banned outright.

### glTF export — spec conformance fix
- ✅ **`oxigaf export --format gltf`** (`export_gltf.rs`) now writes
  spec-conformant glTF 2.0 via the new, shared
  `oxigaf_render::gltf::write_gltf`. The old writer put all five accessors
  on one buffer view with no `byteStride`, which glTF 2.0 forbids for
  accessors of differing element size; bytes written differ from earlier
  0.1.x output.
- Note (documented scope limit, not a bug): `oxigaf_cli::export::export_gltf`
  — used by `ExportStage` (the `pipeline export` stage and `export --format
  all`'s glTF component) — remains a separate, third, unconsolidated writer
  that emits a self-contained `.glb` with the `OXIGAF_gaussians` extension,
  rather than delegating to the shared writer.
- ✅ New export format value: `--format all` writes PLY, safetensors, glTF
  (as `.glb`, via the separate writer above) and JSON checkpoint
  concurrently into a directory.
- Naming fix for documentation only (no source change): the point-cloud
  export value is `--format point-cloud` (hyphenated); `--format pointcloud`
  is rejected by the parser. Earlier docs (and one in-source doc comment)
  used the unhyphenated spelling.

### API signature/type changes (breaking for direct callers of the library
target; the CLI binary itself is unaffected since these are internal call
sites)
- ✅ `lod_generator::{extract_subset, select_spatial_grid_indices,
  find_optimal_reduction_ratios}` now return `Result<_, LodError>` instead
  of an infallible value.
- ✅ `lod_generator::merge_lod_levels`'s 3rd parameter renamed `weight_a` →
  `weight_b` — it is the weight applied to the *second* level being merged;
  the old name mislabeled its own meaning.
- ✅ `model_inspector::InspectableModel::activated_scale` now returns
  `Result<f32, InspectorError>` and the struct is `#[non_exhaustive]`.
- ✅ `quality_checker::compute_ssim` now returns `Result<Option<f32>, 
  QualityError>`, and `ImageQualityMetrics::ssim` is now `Option<f32>` —
  both previously reported a meaningless SSIM value for degenerate inputs
  instead of "not evaluated".
- ✅ `scene_streaming::ss_chunk_scene` gained an `sh_channels: usize`
  parameter (3 → 4 args) so chunk boundaries respect per-Gaussian SH layout.
- ✅ `cli::RenderArgs::width` / `height` are now `Option<u32>` (previously
  `u32`) — see the Render section of the README for the new default
  behavior (derived from `--quality` instead of hardcoded).

### New types
- ✅ `cloud_registration::RegistrationError::NoCorrespondences` — returned
  by correspondence search instead of silently proceeding with an empty
  correspondence set.
- ✅ `stages::GaussianModel` is now `pub use
  oxigaf::render::gaussian::GaussianModel` (previously a local placeholder
  struct) — pipeline stages operate on the real Gaussian model type.
- ✅ `stages::TrainingSetup` struct, plus `DiffusionStage::with_*` /
  `TrainingStage::with_*` builder methods for constructing pipeline stages
  without a full-struct literal.
- ✅ `quality_checker::QualityThresholds::background_color:
  Option<[u8; 3]>` — lets quality/clipping checks account for a
  non-default (non-black) render background.
- ✅ New `npz` Cargo feature, forwarding to `oxigaf-flame/npz`
  (`FlameSequence::from_npz`).

## 🚧 In Progress

Currently none - implementation is very comprehensive!

## 📋 Planned (from original design)

### Video Input Support (NOT STARTED — corrected 2026-08-28)
> A previous revision of this file marked video frame extraction ✅ for
> v0.1.1. That was inaccurate: no `ffmpeg-next` (or any video-decoding)
> dependency exists anywhere in this workspace's `Cargo.toml` files, and no
> `VideoExtractor`/`VideoConfig` type exists in `src/`. Corrected below.

- ⬜ **Video frame extraction** — NOT implemented. `oxigaf train --input`
  requires a directory of pre-extracted frames (or a single frame image);
  passing a `.mp4`/`.mov` container is rejected with an actionable error
  message pointing at an external tool
  (`ffmpeg -i clip.mp4 frames/%05d.png`). OxiGAF is pure Rust and
  deliberately bundles no video demuxer/decoder (every usable one is a C
  library, which the Pure-Rust policy rules out) — this is a permanent
  design constraint, not a gap expected to close with an `ffmpeg-next`
  integration.
  - ✅ Pre-extracted frame directories work today (this always worked)

### Real-Time Preview (partially real — corrected 2026-08-28)
> A previous revision of this file marked this ✅ for v0.1.1, describing a
> winit-windowed, wgpu-surface live viewport. That implementation does not
> exist. What is real: `oxigaf preview` (`src/commands/preview.rs`,
> `src/preview.rs`, `src/arcball.rs`) — corrected below.

- ✅ **`oxigaf preview` command** — real, tested, but terminal-driven rather
  than a GPU window. OxiGAF ships no windowing backend, so there is no OS
  window to draw into; `preview` instead drives a `PreviewController` +
  `ArcballCamera` from raw-mode keyboard input (via `crossterm`) and
  re-renders through the same CPU software rasterizer `oxigaf render` uses
  (`export::render_point_cloud`), writing the result to
  `<output-dir>/preview.png` after every camera change — a normal image
  viewer left open on that file behaves like a live viewport. The
  screenshot key (`s`) additionally writes numbered
  `preview_000.png`, `preview_001.png`, … that are never overwritten.
  - ✅ Orbit (`w`/`a`/`s`/`d`), dolly (`e`/`f3`), pan (arrows/`f1`/`f2`),
    speed up/slow down (`+`/`-`), reset, quit (`q`/Esc)
  - ✅ Toggle animation / next-frame / previous-frame actions
  - ✅ `--script <file>` mode replays a recorded command list with no
    terminal at all — this is what makes the camera logic unit-testable
    and CI-scriptable
  - ⬜ **NOT implemented**: an actual OS window (`winit`), GPU surface
    rendering, mouse-drag orbit, or scroll-to-zoom — there is no `winit` (or
    any windowing) dependency in `Cargo.toml`
  - ⬜ **NOT implemented**: a `--preview` flag on `render` for
    "real-time parameter tweaking" / "live quality adjustment" — the only
    related flag is `train --no-preview`, which *disables* the turntable
    preview renders `train` writes into `preview/`, and is unrelated to the
    `oxigaf preview` command above

### Additional Commands (nice to have)
- ✅ **config command** (`oxigaf config init/validate/show`; `config-cmd`
  kept as a legacy alias as of 0.1.2)
  - `config init` - Create default config file (uses `ProjectConfig` struct + TOML serde)
  - `config validate` - Validate config file (field-by-field validation)
  - `config show` - Display merged configuration (pretty-printing)
  - 8 new tests
- ✅ **info command** (`oxigaf info <path>`)
  - Handles `.ply`: parse header for property names/count + read positions for bounding box, opacity stats, scale stats
  - Handles `.safetensors`: tensor names/shapes/dtypes + metadata dict
  - Handles `.json`: schema detection
  - 8 new tests
- ✅ **compare command**
  - `compare.rs` (~700 lines): `ModelStats::from_file()` for .ply/.safetensors, `bbox_iou()` 3D IoU, `ComparisonReport::compute()` with weighted similarity (bbox 40%, count 30%, SH 10%, scale 10%, opacity 10%), `format_text()`/`format_json()`
  - 17 new tests
- ✅ **diff command — variable-count Gaussian diff** (`diff_tool.rs`, 2094 lines)
  - `diff_models_variable()` for diffing models with different Gaussian counts
  - Handles unequal sizes via optimal transport / nearest-neighbour matching
  - `DiffConfig`, `DiffReport`, `GaussianDiff`, `DiffStats`
  - Supports .ply and .safetensors input formats

### Enhanced Export Features
- ✅ **Mesh export** (`export_mesh.rs`) — Surface Nets mesh export, `--format mesh`, 5 tests
- ✅ **compute_obb** (`geometry_tools.rs`) — PCA-based OBB via nalgebra SymmetricEigen
- ✅ **get_memory_mb** (`benchmark.rs`) — real `/proc/meminfo` (Linux) + `sysctl` (macOS) impl
- ✅ **EasingType::CubicBezier** (`camera_path_tool.rs`) — real CSS cubic-bezier(0.42,0,0.58,1) via Newton-Raphson
- ✅ **ExportStage::run** (`stages.rs`) — dispatches to real `export_ply`/`export_gltf`/`export_safetensors`
- ✅ **glTF export implementation** (`export_gltf.rs`)
  - Creates `.gltf` + `.bin` file pair
  - Binary buffer layout: positions (N×3 f32), rotations (N×4), scales (N×3), opacities (N×1), SH coefficients (N×C)
  - Custom `OXIGAF_gaussian_splat` extension in JSON
  - 8 new tests
- ✅ **Point cloud export** — `export_pointcloud.rs`. `sh_dc_to_u8()` (0.5 + SH_C0 * dc, SH_C0=0.28209). `PointColorMode` enum (ShDc/White/Opacity/Scale). `gaussian_colors()`. `export_pointcloud()` writes binary little-endian PLY (xyz normals=0 + rgb, 27 bytes/point). `PointCloudStats::compute()` + `format_summary()`. `--format point-cloud` (hyphenated) + `--point-color-mode` flags in CLI. 14 new tests.
- ✅ **Video export** (`video_export.rs`)
- ✅ (v0.1.1) **Advanced mesh export** — Poisson reconstruction, marching cubes, texture baking

### Performance Improvements
- ✅ **Parallel rendering** (`parallel_render.rs`) (~360 lines). `ParallelRenderConfig` (num_threads, chunk_size, output_dir, filename_pattern, width, height). `ParallelRenderer` with rayon thread pool (0=auto uses global pool, N>0 creates dedicated ThreadPool). `execute<F>(tasks, render_fn, progress)` lock-free AtomicUsize success counter. `execute_mock` with deterministic threshold. `turntable_tasks(n, elevation)` evenly-spaced azimuth tasks. `ParallelRenderResult` with format_summary(). `--parallel N` flag added to RenderArgs in cli.rs. rayon added to workspace. 26 new tests.
- ✅ (v0.1.1) **Caching optimizations**
  - LRU cache for loaded models
  - Asset bundle downloads
  - Incremental checkpoint updates

### User Experience
- ✅ **Better progress reporting**
  - `progress.rs` (935 lines): `TrainingProgress` (multi-bar with iteration/loss/timing bars via Arc<MultiProgress>), `OperationSpinner` (indeterminate spinner), `BatchProgress` (counted bar), `TimingReport` (HashMap<String,Duration> with record/time/total/percentage/format_table)
  - `indicatif = "0.17"` added; 22 new tests
- ✅ **Configuration wizard**
  - `config init --interactive` in `config_cmd.rs` (`config-cmd` alias also works)
  - Detects CPU cores via `available_parallelism()`, selects sh_degree and views_per_step based on hardware, generates annotated TOML
  - 4 new tests
- ✅ (v0.1.1) **Example configs**
  - Preset configs for common scenarios
  - Quick start templates
  - Best practices examples

## 💡 Future Enhancements

### Advanced Features
- ⬜ **Distributed training**
  - Multi-GPU support across machines
  - Parameter server
  - Gradient aggregation
- ⬜ **Cloud integration**
  - AWS S3 / GCS storage
  - Cloud GPU training
  - Remote model serving
- ⬜ **REST API server**
  - HTTP endpoint for reconstruction
  - WebSocket progress streaming
  - Cloud deployment ready

### Plugins & Extensions
- ⬜ **Plugin system**
  - Custom loss functions
  - Custom exporters
  - Custom renderers
  - Lua/WASM scripting
- ⬜ **Format converters**
  - NeRF → Gaussian conversion
  - Point cloud → Gaussian
  - Mesh → Gaussian
  - 3D photo → Gaussian

### Developer Tools
- ⬜ **Debug visualization**
  - Gaussian parameter histograms
  - Gradient flow visualization
  - Attention map inspection
  - Loss component breakdown
- ⬜ **Profiling mode**
  - Per-component timing
  - Memory profiling
  - GPU utilization
  - Bottleneck identification

## 📊 Current Status

### Implementation: 32/32 planned-and-tracked commands functional; 2 permanent scope limits
- ✅ CLI commands: all 32 subcommands build, are wired into `--help`/shell
  completions, and are exercised by `bin/oxigaf` consistency tests
  (`every_subcommand_documents_itself_for_completions`,
  `cli_definition_is_internally_consistent`)
- ✅ Configuration system: 100%
- ✅ Logging & progress: 100%
- ✅ Error handling: 100%
- ✅ Asset management: 100% (unified cache directory as of 0.1.2)
- ✅ Export formats: 100% (`ply`/`safetensors`/`gltf`/`json`/`point-cloud`/
  `mesh`/`all`; glTF now spec-conformant as of 0.1.2)
- ✅ Pipeline orchestration: 100%
- ✅ Interactive mode: 100%
- ✅ JSON output: 100%
- ✅ Metrics export: 100%
- ✅ Dry run: 100%
- ✅ Parallel rendering: 100% (`parallel_render.rs`, rayon ThreadPool)
- ✅ Point cloud export: 100% (`export_pointcloud.rs`, binary LE PLY, `PointColorMode`)
- ✅ Terminal-driven live preview: 100% (`oxigaf preview`, see *Real-Time
  Preview* above for exactly what this does and does not cover)
- ⬜ Video **input** extraction from `.mp4`/`.mov`: 0%, and not planned as an
  `ffmpeg-next` integration — a permanent Pure-Rust-policy scope limit, not
  a gap expected to close (pre-extract frames with an external tool instead)
- ⬜ Windowed (`winit`/GPU-surface) real-time preview: 0% — no windowing
  dependency exists; `oxigaf preview`'s terminal-driven design above is the
  supported alternative, not a stepping stone to this

### Tests: 3105 tests, all passing (measured 2026-08-28, `--all-features`)
See *Testing* under *Completed (from plan)* above for the breakdown. Grown
from 2375 at v0.1.1 alongside the jump from ~12 to 32 subcommands.
- ✅ Good coverage for core functionality
- ⬜ Missing: video-input-extraction tests (no such feature exists to test)
- ⬜ Missing: windowed-preview tests (no such feature exists to test)

### Documentation: Excellent
- ✅ Comprehensive rustdoc
- ✅ Help text for all commands
- ✅ Environment variable documentation
- ✅ Configuration examples in help
- ✅ Shell completion installation guides
- ✅ Error suggestions

## 📈 Comparison: Implementation vs Plan

| Feature | Plan | Current | Notes |
|---------|------|---------|-------|
| train/reconstruct command | ✅ | ✅ | Fully implemented with extras |
| render command | ✅ | ✅ | Fully implemented |
| export command | ✅ | ✅ | Fully implemented (glTF done) |
| convert command | ✅ Basic | ✅ | Enhanced with verification |
| Video **input** extraction | ✅ | ⬜ | Not implemented; permanent Pure-Rust-policy scope limit, not feature-gated work in progress |
| Windowed real-time preview | ✅ | ⬜ | Not implemented (no `winit`); terminal-driven `oxigaf preview` (done) is the supported alternative |
| Asset management | ✅ Basic | ✅ | **EXCEEDS** - Full setup/cache system |
| Logging & progress | ✅ | ✅ | **EXCEEDS** - Log rotation, JSON output |
| info command | ⬜ Not started | ✅ | **Done** (.ply/.safetensors/.json, bbox/stats, 8 tests) |
| config command | ⬜ Not started | ✅ | **Done** (init/validate/show + --interactive wizard, 12 tests) |
| benchmark command | ⬜ Not in plan | ✅ | **EXCEEDS PLAN** |
| compare command | ⬜ | ✅ | **Done** (`compare.rs`, ModelStats, bbox_iou, weighted similarity, 17 tests) |
| progress reporting | ⬜ | ✅ | **Done** (`progress.rs` 935 lines, TrainingProgress/OperationSpinner/BatchProgress/TimingReport, 22 tests) |
| Parallel rendering | ⬜ Not in plan | ✅ | **Done** (`parallel_render.rs` ~360 lines, rayon ThreadPool, AtomicUsize counter, 26 tests) |
| Point cloud export | ⬜ Not in plan | ✅ | **Done** (`export_pointcloud.rs`, binary LE PLY, `PointColorMode`, `sh_dc_to_u8`, 14 tests) |
| doctor command | ⬜ Not in plan | ✅ | **EXCEEDS PLAN** |
| completions command | ⬜ Not in plan | ✅ | **EXCEEDS PLAN** |
| Dry run mode | ⬜ Not in plan | ✅ | **EXCEEDS PLAN** |
| JSON output mode | ⬜ Not in plan | ✅ | **EXCEEDS PLAN** |
| Interactive mode | ⬜ Not in plan | ✅ | **EXCEEDS PLAN** |
| HuggingFace Hub | ⬜ Not in plan | ✅ | **EXCEEDS PLAN** |
| Metrics export | ⬜ Not in plan | ✅ | **EXCEEDS PLAN** |
| Environment variables | ⬜ Not in plan | ✅ | **EXCEEDS PLAN** |

## 🎯 Priority for v1.0

**CRITICAL (Blockers for basic usage):**
- Currently none - CLI is functional!

**High Priority (Improve usability):**
1. ⬜ Video frame extraction — not planned as an `ffmpeg-next` integration;
   permanent Pure-Rust-policy scope limit (see *Video Input Support* above)
2. ✅ ~~glTF export implementation~~ — Done (`export_gltf.rs`, `.gltf`+`.bin`
   pair; spec-conformant as of 0.1.2 via `oxigaf_render::gltf::write_gltf`)
3. ⬜ Windowed real-time preview (winit/GPU surface) — not planned; the
   terminal-driven `oxigaf preview` (done, see above) is the supported
   alternative

**Medium Priority:**
4. ✅ ~~config command (init, validate, show)~~ — Done (`oxigaf config`;
   `config-cmd` kept as a legacy alias)
5. ✅ ~~info command (model inspection)~~ — Done
6. ✅ ~~Video export~~ — Done as GIF/frame-sequence/manifest/HTML-viewer
   (`oxigaf video build`/`viewer`); MP4/H.264 is a deliberate permanent
   non-goal (every usable encoder is a C library) rather than a remaining
   gap

**Low Priority:**
7. ✅ ~~compare command~~ — Done (`compare.rs`, ModelStats, bbox_iou, weighted similarity)
8. ✅ ~~Mesh export~~ — Done (`export_mesh.rs`, Surface Nets, `--format mesh`)
9. ✅ ~~Configuration wizard~~ — Done (`config init --interactive`, hardware detection, annotated TOML)
10. ✅ ~~Point cloud export~~ — Done (`export_pointcloud.rs`, `sh_dc_to_u8`, `PointColorMode`, binary LE PLY; `--format point-cloud`, hyphenated)

## 🏆 Implementation Highlights

**Where current implementation EXCEEDS the plan:**

1. **Comprehensive Command Set** (far beyond plan)
   - benchmark command with multiple targets and formats
   - doctor command for system diagnostics
   - cache management subcommands
   - completions for all major shells
   - Enhanced export with multiple formats

2. **Configuration System** (1049 lines - EXCEEDS)
   - Multi-level hierarchy (CLI > env > project > user > defaults)
   - Full environment variable support (OXIGAF_* variables)
   - TOML validation and error reporting
   - Training profile presets

3. **Logging & Output** (EXCEEDS)
   - Log rotation with multiple strategies
   - Multiple log formats (JSON, pretty, compact)
   - JSON output mode for scripting
   - Metrics export (CSV/JSON Lines)
   - Comprehensive progress bars

4. **Error Handling** (EXCEEDS)
   - Specific error variants for all failure modes
   - Actionable suggestions
   - Exit code mapping for automation
   - User-friendly messages

5. **HuggingFace Hub Integration** (not in plan)
   - Download from HF repositories
   - Authentication support
   - Revision/branch/tag support
   - Automatic caching

6. **Interactive Features** (not in plan)
   - Interactive training mode with keyboard controls
   - Dry run validation
   - Pause/resume/skip controls
   - On-demand checkpoint saving

7. **Testing** (exceeds typical CLI coverage)
   - 3105 tests across `tests/` + `src/` unit tests (measured 2026-08-28,
     `--all-features`)
   - Integration tests with assert_cmd
   - Configuration hierarchy tests
   - HuggingFace Hub tests
   - JSON output validation

8. **Code Quality** (exemplary)
   - No unwrap/expect policy enforced
   - All files under the 2000-line workspace policy limit (largest currently
     1999 lines)
   - Comprehensive rustdoc
   - Clean module boundaries

**Current implementation is PRODUCTION-READY for:**
- End-to-end avatar reconstruction (with pre-extracted frames)
- Novel view rendering
- Model export (PLY, safetensors, JSON)
- FLAME model conversion
- Performance benchmarking
- System diagnostics
- Asset management
- Scripting and automation (JSON output)

**Not yet ready for, and not on the roadmap (see corrected sections above):**
- Video **input** from `.mp4`/`.mov` containers — pre-extract frames instead
- Windowed/GPU-surface real-time preview — `oxigaf preview` (terminal-driven,
  done) is the supported alternative, not a stepping stone to this

## 🚀 Post v0.1.0 Next Steps (historical; superseded by v0.1.2 above)

oxigaf-cli v0.1.0 is **functionally complete** for core use cases. Future enhancements:

1. ~~**Video Frame Extraction** (~3-4 days)~~ — **superseded**: not
   implemented, and no longer planned as an `ffmpeg-next` integration; see
   *Video Input Support* above for why this is a permanent scope limit
   rather than deferred work.

2. ~~**Real-Time Preview Window** (~5-7 days)~~ — **superseded**: no
   `winit`/wgpu-surface window was built; see *Real-Time Preview* above for
   what was built instead (`oxigaf preview`, terminal-driven).

3. ✅ ~~**glTF Export**~~ — Done (`export_gltf.rs`, binary buffer layout, `OXIGAF_gaussian_splat` extension); made spec-conformant in 0.1.2.

4. ✅ ~~**config Command**~~ — Done (init/validate/show + --interactive wizard); command renamed `config` in 0.1.2, `config-cmd` kept as an alias.

5. ✅ ~~**info Command**~~ — Done (.ply/.safetensors/.json, bounding box + stats)

6. ✅ ~~**compare Command**~~ — Done (`compare.rs`, ModelStats, bbox_iou 3D IoU, weighted similarity, format_text/format_json)

7. ✅ ~~**Better Progress Reporting**~~ — Done (`progress.rs`, TrainingProgress/OperationSpinner/BatchProgress/TimingReport)

8. ✅ ~~**Parallel Rendering**~~ — Done (`parallel_render.rs`, `ParallelRenderConfig`, `ParallelRenderer` with rayon ThreadPool, lock-free AtomicUsize counter, `turntable_tasks`, `--parallel N` flag)

9. ✅ ~~**Point Cloud Export**~~ — Done (`export_pointcloud.rs`, `sh_dc_to_u8()`, `PointColorMode`, `PointCloudStats`, binary LE PLY, `--format point-cloud` (hyphenated) + `--point-color-mode` flags)

## 📝 Notes

- **ffmpeg-next**: not a dependency of this crate or workspace, and not
  planned — video-container input decoding is a permanent Pure-Rust-policy
  scope limit, not feature-gated work in progress (corrected 2026-08-28;
  a previous revision of this file described it as already feature-gated,
  which was never accurate)
- **winit**: not a dependency of this crate or workspace, and not planned
  for a real-time preview window — see *Real-Time Preview* above for the
  terminal-driven design that shipped instead (corrected 2026-08-28)
- **Current architecture**: Well-designed for extensions
- **Test coverage**: Excellent for a CLI tool
- **Documentation**: Comprehensive help text and rustdoc
- **MSRV**: Rust 1.87 (workspace `rust-version`, per the "Latest Crates" policy)
- **Dependencies**: Well-managed with workspace inheritance

## 🎨 Architecture Quality

The CLI architecture is **exceptionally well-designed**:

1. **Separation of concerns**: Each command has its own module
2. **Testability**: Library functions exposed for integration testing
3. **Error handling**: Comprehensive with actionable messages
4. **Configuration**: Flexible multi-level system
5. **Extensibility**: Easy to add new commands/features
6. **Documentation**: Help text guides users through all features
7. **Performance**: Dry run mode prevents expensive mistakes
8. **Scripting**: JSON output mode for automation
9. **User experience**: Progress bars, interactive mode, helpful errors
10. **Code quality**: No unwrap/expect, clean modules, comprehensive tests

The CLI implementation **sets a high bar** for Rust CLI tools and serves as an excellent reference for CLI design patterns.
