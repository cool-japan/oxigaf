# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.2] - Unreleased

> **Migration notes.** Most of what follows is additive or affects only
> advanced/internal APIs. Two changes are worth reading before you upgrade:
>
> - **PLY files with `sh_degree >= 1` written by `GaussianModel::save_ply`
>   (`oxigaf-render`) before this release load with permuted higher-order SH
>   coefficients.** The `f_rest_*` property order changed to
>   channel-major (matches the reference 3DGS Python convention,
>   `features_rest.transpose(1, 2).flatten()`) — see *Fixed → oxigaf-render*
>   below. Re-export any `.ply` written this way that you care about. (Scoped
>   to this one writer/reader pair — `oxigaf-cli` has its own separate PLY
>   toolchain in `export_ply`/`export.rs`, not covered by this note.)
> - **macOS: the default asset cache directory moved** from `~/.cache/oxigaf`
>   to `~/Library/Caches/oxigaf` (`dirs::cache_dir()`), because `setup`,
>   `doctor`, and `cache` used to each compute it a different way — see
>   *Fixed → oxigaf-cli* below. Move any existing `~/.cache/oxigaf` state, or
>   set `OXIGAF_CACHE_DIR` to keep the old location.

### Changed

- **candle-core / candle-nn** updated 0.10 → 0.11, and switched from
  upstream `huggingface/candle` to the COOLJAPAN fork published as
  `oxicandle-core` / `oxicandle-nn` (dependency *keys* stay `candle-core` /
  `candle-nn`, so no source changes were needed). The fork selects
  `fancy-regex` instead of `onig` in its `tokenizers` dependency.
- **wgpu** updated 29 → 30
- **oxiarc-archive** updated 0.3.3 → 0.4.1
- **torsh-core / torsh-tensor / torsh-nn** (oxigaf-bridge) updated 0.1.2 → 0.2.0
- **pollster** updated 0.4 → 1
- **oxigaf-diffusion**: `flash_attention` is no longer enabled by default.
  Previously `default = ["flash_attention"]` on `oxigaf-diffusion` meant the
  `flash_attention` forwarding feature on `oxigaf` / `oxigaf-cli` could never
  actually be turned off; it is now a real opt-in (`--features
  flash_attention`, or the `full_performance` / `all_features` bundles).

#### oxigaf-diffusion — signature and semantics changes

- `compute_distillation_loss` and `DistillationStep::aggregate_losses`
  (`distillation_loss.rs`) now return `Result` instead of an infallible
  value; `DistillationLossResult` gained a `mode: DistillationMode` field,
  `DistillationConfig` gained `latent_dims: Option<LatentDims>` (required
  when `lpips_proxy_weight > 0`), and `TeacherStudentPair` gained
  `teacher_mid: Option<NoisePrediction>` (set via
  `TeacherStudentPair::with_teacher_mid`) for true two-step teacher
  evaluation.
- `dpm_plus_plus_2m_step` now takes `prev_x0`/`h_prev` instead of a single
  `prev_noise_pred` (6 → 7 arguments) — the DPM-Solver++ 2M multistep
  formula needs the previous denoised sample and step size, not just the
  previous noise prediction.
- `InversionTrajectory::x_0()` / `x_t()` (`inversion.rs`) now return
  `Option<&[f32]>` instead of `&[f32]`, and `reconstruction_error` can
  return `Err` — both were previously infallible against trajectories that
  hadn't reached the requested step.
- `DdimInversionConfig::refinement_threshold` is now compared against the
  RMS (root-mean-square) reconstruction error instead of the raw L2 norm, so
  the same threshold value now means something different (scale-independent
  of the trajectory length) — re-tune any pinned threshold.
- `pad_to_square`, `flip_horizontal`, `flip_vertical` (`image_preprocessing.rs`)
  now return `Result<_, PreprocessError>` instead of an infallible value — an
  empty image or a source buffer that disagrees with its `ImageDims` is
  reported as an error instead of panicking or producing garbage output.
- `FlowStats::divergence_estimate` (`flow_matching.rs`) renamed to
  `mean_squared_norm` — the old name overclaimed what the field measures: it
  is `mean(sum(v_i^2))` over the batch, not an estimate of the velocity
  field's divergence (`div v = sum_i dv_i/dx_i`).
- `EditMask::all_ones` (`image_editing/mod.rs`) takes `(height, width)`, the
  same argument order as `EditMask::new` / `EditMask::from_data`.
- `fm_interpolate` / `fm_target_velocity` (`flow_matching.rs`): the `Linear`
  path now honors `FlowMatchingConfig::sigma_min` (default `0.001`) instead
  of assuming `sigma_min == 0`; at `t = 1` it returns `sigma_min * x_0 +
  x_1` rather than exactly `x_1` whenever `sigma_min > 0`.
- `softmax_over_dim` (`fused_attention.rs`) now returns `Result<(),
  DiffusionError>` instead of an infallible value.
- `sdedit::ImageEditError` removed — `sdedit` now uses the parent
  `image_editing::ImageEditingError` directly instead of a separate enum
  bridged in via `From`. Old variants map onto it as `InvalidStrength` /
  `InvalidParam` → `InvalidConfig`, `InvalidMask` → `InvalidImage`,
  `TimestepOutOfRange` → `NoiseLevelOutOfRange`, and `DimensionMismatch`'s
  `got` field renamed to `actual`.
- `MultiViewDiffusionPipeline::begin_session_from_latents` (`pipeline.rs`)
  now takes a single `SessionRequest<'_>` instead of seven positional
  arguments (four of them `&Tensor`, and therefore trivially transposable at
  a call site) — it bundles `reference_image`, `normal_map_latents`,
  `camera_poses`, `latents`, `seed`, `start_step`, and
  `num_inference_steps`.

#### oxigaf-flame — signature and semantics changes

- `shade_mesh_directional` / `shade_mesh_multi_light` (`lighting_model.rs`)
  now return `Err` when a light colour component is outside `[0, 1]`,
  instead of silently clamping it into range.
- `LightingError` gained an `InvalidFaceIndex` variant; the enum is not
  `#[non_exhaustive]`, so an exhaustive `match` on `LightingError` needs a
  new arm.
- `DecimationConfig::default().target_vertex_count` changed from `0` to
  `usize::MAX` (`multiresolution.rs`). `0` was always rejected by validation
  (`"target_vertex_count must be > 0"`), so the old default was an
  always-reject footgun; `usize::MAX` is a documented no-op default
  (decimation only runs while `live_vertex_count > target_vertex_count`).
- `estimate_pitch_from_vertical` (re-exported at `oxigaf_flame::lib.rs:351`)
  now takes a third `&PitchReference` argument and returns
  `Result<[f32; 2], _>` instead of `Result<f32, _>`. The arity change means
  old call sites fail to compile rather than silently compiling against a
  changed meaning.

#### oxigaf-render — signature and semantics changes

- `RadixSorter::sort` no longer takes `device`/`keys`/`values` — that setup
  moved to a new `prepare()` step (`sort.rs`); `capacity()` now reports the
  sorter's live, growable buffer capacity rather than a fixed construction-
  time value.
- `Rasterizer::from_device` now errors when `tile_size != 16` instead of
  silently accepting it, and `WorkgroupConfig::from_profile`'s tile
  dimensions changed from 4×4 to 16×16 to match `RASTERIZE_TILE_SIZE` (see
  *Added* below).
- `MbStats.mean_samples_used` renamed to `estimated_sample_utilization`
  (motion-blur pipeline) — the old name implied an exact sample count; the
  field is a ratio.
- `hdr_tone_mapping::lottes_approx` renamed to `generalized_reinhard`, next
  to the new, more accurate `hdr_tone_mapping::lottes` (see *Added* below) —
  "approx" was misleading once a non-approximating Lottes implementation
  existed.

#### oxigaf-trainer — signature and semantics changes

- `ViewConsistencyLoss` (`diffusion_target.rs`) gained a `pub enable_warping:
  bool` field, which breaks external `ViewConsistencyLoss { .. }`
  struct-literal construction that doesn't list it. `DiffusionTargetGenerator`
  gained `generate_targets_with_normals`, `view_consistency_loss()`, and
  `target_config()`; the private `cameras_to_tensor` helper it calls now
  takes a `num_views` parameter and tiles cameras cyclically (or truncates)
  to match it, since the pipeline always denoises exactly
  `DiffusionConfig::num_views` latents.
- `detect_convergence_phase` gained a `loss_scale` parameter (4 → 5 args);
  `ConvergenceConfig` gained `oscillation_threshold` and `max_history`
  (`convergence_analysis.rs`).
- `mr_upsample`, `mr_gaussian_blur_3x3`, `mr_sobel_magnitude`
  (`multi_resolution_loss.rs`) and `GaussianOptimizer::step` /
  `step_accumulated` (`optimizer.rs`) now return `Result` instead of an
  infallible value, per the no-`unwrap()` policy.
- `SyncReport` gained `buckets` and `gradient_compression` fields
  (`data_parallel.rs`); `estimated_bandwidth_mb` changed meaning — see
  *Added* below for the new fields' semantics.

#### oxigaf-cli — signature and semantics changes

- `extract_subset`, `select_spatial_grid_indices`, and
  `find_optimal_reduction_ratios` now return `Result` instead of an
  infallible value. `merge_lod_levels`'s 3rd parameter was renamed
  `weight_a` → `weight_b` (the parameter's actual meaning — it is the weight
  applied to the *second* level being merged — was mislabeled, not just
  renamed for style).
- `InspectableModel::activated_scale` now returns `Result` and the type is
  `#[non_exhaustive]`. `compute_ssim` now returns `Result<Option<f32>>` and
  `ImageQualityMetrics::ssim` is now `Option<f32>` (previously both were
  infallible and could silently report a meaningless SSIM for degenerate
  inputs).
- `ss_chunk_scene` gained an `sh_channels` parameter (`scene_streaming.rs`,
  3 → 4 args) — chunk boundaries must respect per-Gaussian SH layout, which
  varies with `sh_degree`.
- `RenderArgs::width` / `height` are now `Option<u32>` instead of `u32` (the
  render command now derives a default from the source model/camera instead
  of hardcoding one).

### Fixed

- **C Oniguruma removed from the default build** — `candle-core` → `tokenizers`
  previously pulled in `onig`/`onig_sys`, compiling the C Oniguruma library
  via `cc`/`pkg-config` on every default build. Fixed by the `oxicandle-core`
  switch above (verified: `Cargo.lock` now has zero `onig`/`onig_sys`
  entries). This was a COOLJAPAN Pure-Rust policy violation.
- **C OpenSSL, and later `ring`, removed from `oxigaf-cli`'s HTTP stack** —
  superseded the entry this replaces (`hf-hub` reconfigured with
  `default-features = false, features = ["ureq"]`, which only removed
  `native-tls`/`openssl-sys`). `hf-hub` has since been dropped entirely: its
  sole consumer, `oxigaf-cli/src/assets.rs`, now talks to the Hugging Face
  Hub `resolve` endpoint directly over `ureq 3.4`
  (`default-features = false`, `rustls-no-provider` +
  `rustls-webpki-roots`) with `rustls 0.23`
  (`default-features = false`: `std`, `tls12`, `logging`) and the process
  `CryptoProvider` installed once from `oxitls-rustcrypto-provider 0.3`
  (COOLJAPAN's pure-RustCrypto fork of `rustls-rustcrypto`,
  RUSTSEC-2026-0104 fixed). This eliminates `ring` (C + hand-written asm) as
  well as `openssl`/`native-tls`, and — as a side effect of no longer
  shelling out to `curl`/`wget` — `download_file` now streams with native,
  byte-accurate progress. Verified: `cargo tree -i ring` and
  `cargo tree -i hf-hub` are both empty; `cargo deny check bans` passes with
  `ring` fully banned (no wrapper exception needed).

### Deprecated

- **`ExpressionLibrary::default_expressions()`** (`oxigaf-flame`) is now
  `#[deprecated]` in favour of `placeholder_expressions()`, which states in
  its name that the returned expressions are placeholders rather than a
  recommended default. A workspace that denies deprecation warnings
  (`-D deprecated`) will fail to build against any future caller that still
  uses the old name.

#### oxigaf-flame — bug fixes

- **`FittingError::CameraProjectionFailed` is now actually reachable** —
  `fit_landmarks` previously declared the variant but never constructed it
  (dead code); it is now returned when not one landmark projects in front of
  the camera, instead of the fit silently proceeding on a degenerate
  projection. `FittingResult` gained a public `n_visible_landmarks: usize`
  field so callers can distinguish "good fit" from "fit against very few
  visible landmarks" without re-deriving the count themselves.

#### oxigaf-render — bug fixes

- **`GaussianModel::save_ply` / `load_ply` (`gaussian.rs`) `f_rest_*`
  property order changed to channel-major** — matches the reference 3D
  Gaussian Splatting Python convention; this crate's own in-memory
  `sh_coeffs` layout is coefficient-major RGB-interleaved, so the two orders
  are now explicitly permuted on write and un-permuted on read instead of
  being written through unchanged. See the migration note at the top of
  this section: files written by this code path before the fix contain
  correctly-valued but permuted higher-order SH coefficients (`sh_degree >=
  1`); re-export them. Covered by a new regression test,
  `test_ply_save_writes_f_rest_channel_major`.

#### oxigaf-trainer — bug fixes

- **`decompose_uncertainty` no longer fabricates an aleatoric/epistemic
  split for a single point sample** (`uncertainty_estimation.rs`) — it
  previously rescaled a single variance estimate into an arbitrary ~0.64 /
  0.36 aleatoric/epistemic split with no statistical basis. It now reports
  `aleatoric == 0.0` and `epistemic == total` for point samples (there is no
  data-noise signal to separate without repeated observations); callers
  that need a real data-noise term must call
  `decompose_uncertainty_with_variances` instead, which takes the repeated
  samples the decomposition actually requires.
- **`CurriculumSchedule::validate` now rejects invalid stage ordering** —
  previously accepted schedules where a stage's `step_start > step_end`, or
  where consecutive stages left an uncovered gap between them; both now
  return a validation error instead of producing a schedule that silently
  skips or inverts a training stage at runtime.

#### oxigaf-cli — bug fixes

- **`setup` / `doctor` / `cache` agree on the asset cache directory** — they
  previously computed it three different ways (`~/.cache/oxigaf`,
  `$HOME/.cache/oxigaf`, `dirs::cache_dir()`), so on macOS `oxigaf setup`
  populated `~/.cache/oxigaf` while `oxigaf cache list` looked in
  `~/Library/Caches/oxigaf` and reported an empty cache. All three now go
  through one `commands::runtime::default_cache_dir()`, which uses
  `dirs::cache_dir()` (macOS: `~/Library/Caches/oxigaf`) unless
  `OXIGAF_CACHE_DIR` is set. `SetupArgs::cache_dir` is now `Option<PathBuf>`
  to let it fall through to the shared default instead of hardcoding
  `~/.cache/oxigaf` as a `clap` default value — see the migration note at
  the top of this section.
- **CLI `convert`'s `.pkl` path now actually works** — see the annotation on
  the `[0.1.0]` `convert` entry below.

### Removed

- **`candle-transformers`** — declared as a workspace dependency of
  `oxigaf-diffusion` but never referenced anywhere in the codebase; removed
  to avoid dragging upstream `candle-core` (and therefore `onig`) back into
  the graph alongside the `oxicandle-core` fork.
- **`.github/workflows/ci.yml.disabled`** — a disabled, non-functional GitHub
  Actions workflow. It also targeted `branches: [main]`, but this
  repository's default branch is `master`, so it would never have run even
  if re-enabled as-is. Project policy permits only `pypi-publish.yml` /
  `npm-publish.yml` under `.github/workflows/`.

### Added

- **deny.toml** — workspace-level `cargo-deny` configuration enforcing the
  COOLJAPAN banned-crate list (compression, BLAS/LAPACK, TLS, tokenizer C
  backends, etc.), with documented `wrappers`/skip exceptions for crates not
  yet migrated off their C dependency.

#### oxigaf-cli

- `RegistrationError::NoCorrespondences` (`cloud_registration/types.rs`) —
  returned by point-cloud correspondence search when not one pair could be
  matched, instead of proceeding with an empty correspondence set.
- `stages::GaussianModel` is now `pub use oxigaf::render::gaussian::
  GaussianModel` instead of a local placeholder struct, so pipeline stages
  operate on the real Gaussian model type. New `TrainingSetup` struct, plus
  `DiffusionStage::with_*` / `TrainingStage::with_*` builder methods for
  constructing pipeline stages without a full-struct literal.
- `QualityThresholds` gained a `background_color` field, letting quality
  checks account for a non-default render background instead of assuming
  black.

#### oxigaf-render

- `GaussianStats` gained `opacity_above_099_count` / `degenerate_scale_count`
  and a new `compute_stats_and_histograms()` entry point that computes both
  in one pass over the model.
- `hdr_tone_mapping::LottesParams` and `hdr_tone_mapping::lottes()` — a real
  (non-approximating) implementation of the Lottes tone-mapping operator,
  alongside the renamed `generalized_reinhard` (see *Changed* above).
- `LensDistortionError::BufferSizeMismatch` — returned instead of an
  out-of-bounds panic/silent truncation when a caller-supplied buffer
  doesn't match the expected pixel count.
- `CullingResult::behind_camera_culled` — a separate counter from the
  existing frustum-cull counts, so callers can distinguish "behind the
  camera" from "outside the side/top/bottom/far planes" when diagnosing an
  unexpectedly empty render.
- `Rasterizer::invalidate_gaussians` / `Rasterizer::workgroups` and a new
  `pub const RASTERIZE_TILE_SIZE: u32 = 16` — the tile size the forward/
  backward shaders are compiled against, now named instead of a bare
  literal repeated at each call site (see the `Rasterizer::from_device` /
  `WorkgroupConfig::from_profile` entries under *Changed* above, which both
  now key off this constant).

#### oxigaf-trainer

- `SyncReport` gained `buckets` and `gradient_compression` fields
  (`data_parallel.rs`), reporting per-bucket gradient-sync detail instead of
  only an aggregate. `estimated_bandwidth_mb` now reflects the
  post-compression transfer size when `gradient_compression` is active,
  rather than always reporting the uncompressed size.

## [0.1.1] - 2026-06-19

### Added

#### oxigaf-flame — Expanded FLAME head model capabilities
- Avatar rigging and pose system: `AvatarRig`, `GazeController`, `HeadTracker`, `HeadGeometry`, `PoseEstimation`, `PosePrior`
- Expression system: `Expressions`, `ExpressionAnimation`, `ExpressionClustering`, `ExpressionTransfer`, FACS AU coefficients, emotion recognition, phoneme-driven animation
- Mesh processing suite: mesh operations, repair, smoothing, subdivision (Loop/Catmull-Clark), morphing, and analysis
- Geometry tools: geodesic distance computation, spectral analysis, multiresolution mesh representation, statistical shape model, symmetry detection
- UV and texture pipeline: UV parameterisation, texture baking, face atlas generation, albedo map, SH lighting model
- Motion and deformation: timeline, warp field, shape retargeting, dynamic landmark tracking
- Model fitting and alignment: blend shape solver, rigid alignment, canonical-space conversion
- Utility: GPU buffer management, vertex masks, visibility culling, contact detection, depth estimation, face normalisation, parameter sampler

#### oxigaf-render — Comprehensive post-processing and rendering pipeline
- Post-processing: ambient occlusion (SSAO), bloom, denoising, depth-of-field, motion blur, film grain, image sharpening, chromatic aberration, vignetting, lens distortion, temporal anti-aliasing, exposure control, HDR tone mapping, tone curve, color grading, colorspace conversion, color calibration
- Volumetric rendering module: ray types, camera model, volume grid, and ray-march result traits
- Scene composition: image compositor, scene compositor, render graph, silhouette extraction, background synthesis
- Stereo rendering: side-by-side / top-bottom stereo output
- Spatial acceleration: BVH, LOD generator, Gaussian culling, normal estimation, depth map
- GPU tooling: workgroup size utilities, debug readback, GPU profiler, render metrics, device init helpers
- Camera: camera path interpolation, panoramic (equirectangular) projection, multi-view rendering
- Additional: MIP splatting, interactive Gaussian picking, mesh compression, model pruning, Gaussian deformation, density estimation, tile statistics, edge detection, subsurface scattering, antialiasing module

#### oxigaf-diffusion — Extended diffusion model features
- Sampler suite: DDPM sampler, adaptive sampling, classifier-free guidance, guidance rescaling, consistency model, flow matching
- Image editing module: SDEdit-style in-context image editing
- Conditioning: identity conditioning, avatar conditioning, classifier guidance, ControlNet adapter, LoRA adapter
- Attention enhancements: attention masking, attention visualisation, fused attention, KV cache
- Latent space: latent blending, latent interpolation, latent space analysis, latent walk, denoising trajectory
- Debugging and evaluation: debug hooks, denoising visualisation, distillation loss, batch generation, CLIP scoring
- Other: cross-frame consistency, dynamic views, image preprocessing, image variations, DDIM inversion

#### oxigaf-trainer — Expanded training infrastructure
- Loss functions: adaptive loss, adaptive loss weighting, contrastive loss, multi-resolution loss, loss reweighting, loss landscape visualisation
- Training regimes: curriculum learning, progressive training, few-shot adaptation, meta-learning (MAML), continual learning
- Gradient tools: gradient accumulation, gradient clipping, gradient flow analysis, gradient surgery
- Online learning: online learning, online hard example mining (OHEM)
- Model analysis: activation maps, anomaly detection, convergence analysis, diagnostics, layer freezing, model pruning
- Optimisation utilities: EMA, spectral normalisation, stochastic weight averaging, mixed precision, custom optimiser, learning-rate scheduler
- Data pipeline: augmentation, data augmentation, camera sampling, synthetic data generation, validation split, noise injection
- Session management: session recorder, profiler integration, training callbacks, checkpoint manager, checkpoint interpolation
- Transfer learning: knowledge distillation, domain adaptation, feature bank, contrastive learning
- Training configuration: `TrainingConfig`, pose conditioning, view scheduler, view importance sampling, view synthesis evaluation
- Other: data parallel training, temperature scaling, uncertainty estimation, hyperparameter search, regularisation

#### oxigaf-cli — Dramatically expanded toolset
- Export: PLY export module, glTF export, mesh export, point cloud export, video export, animation sequence export (JSON)
- Analysis and inspection: scene analyser, model inspector, diff tool, model comparison, quality checker, evaluation suite
- Scene operations: scene merging, scene optimiser, scene streaming, Gaussian filter, Gaussian deduplicator, Gaussian compressor (k-means)
- Training tools: training monitor, resume analyser, parameter sweep, batch processor
- Visualisation: arcball camera controller, preview, LOD generator, parallel renderer, camera path editor, live dashboard
- Reporting: experiment report, HTML report generator, profiling report, telemetry
- Configuration: `config` subcommand, config presets, workspace manager
- Memory and performance: memory estimator, benchmark suite
- Cloud and data: cloud (point cloud) registration, colour calibration, dataset tools, geometry tools, format converter

#### oxigaf — Unified pipeline module
- New `pipeline` module for end-to-end orchestration of training, rendering, and export stages
- New examples: `checkpoint_lifecycle`, `custom_loss`, `end_to_end_pipeline`

### Changed

- **nalgebra** updated 0.34 → 0.35
- **glam** updated 0.32 → 0.33
- **candle-core / candle-nn / candle-transformers** updated 0.9 → 0.10
- **safetensors** updated 0.7 → 0.8
- **wgpu** updated 28 → 29
- **toml** updated 1.0 → 1.1
- **clap_complete** updated 4.5 → 4.6
- **rayon** updated 1.11 → 1.12
- **proptest** updated 1.10 → 1.11
- **oxiarc-archive** updated 0.2.1 → 0.3.3
- **hf-hub** updated 0.4 → 0.5
- **sha2** updated 0.10 → 0.11
- **torsh-core / torsh-tensor / torsh-nn** (oxigaf-bridge) updated 0.1.0 → 0.1.2
- Added `kiddo 5` spatial data structure crate to workspace dependencies

## [0.1.0] - 2026-02-24

### Added

#### Core Libraries

- **oxigaf-flame** — FLAME parametric 3D head model implementation
  - Linear Blend Skinning (LBS) with SIMD optimizations
  - Rodrigues rotation formula for joint transformations
  - Normal map generation from mesh geometry (CPU rasterizer)
  - Blend shapes application for facial expressions
  - Mesh sampling with barycentric coordinates
  - Support for `.npy` FLAME model files
  - **safetensors I/O** — load/save FLAME model in `.safetensors` format
  - **FlameSequence** — video frame processing with LRU caching and interpolation
  - Property-based testing with proptest
  - SIMD feature flag for vectorized operations (2-4× speedup)
  - Parallel feature flag for rayon-based parallelism

- **oxigaf-diffusion** — Multi-view diffusion pipeline
  - Multi-view U-Net architecture with cross-view attention for geometric consistency
  - **Latent Upsampler** — 32×32 → 64×64 latent upsampling for 512×512 output resolution
  - **IP-Adapter** — identity-preserving image conditioning for consistent face generation
  - **Classifier-Free Guidance (CFG)** — quality improvement with configurable guidance scale (1.0–20.0)
  - Camera pose conditioning via explicit camera pose embeddings
  - Flash Attention implementation for memory-efficient attention
  - VAE encoder/decoder for latent space operations
  - DDPM/DDIM noise scheduler support
  - CLIP text encoder integration
  - Mixed precision training support
  - CUDA and Metal GPU backend support

- **oxigaf-render** — 3D Gaussian Splatting rasterizer
  - GPU-accelerated rasterization using wgpu
  - Spherical harmonics (SH) evaluation — specialized shaders for degrees 0–3 (10× speedup)
  - Tile-based radix sort with Fuchsia-based 64-bit key sorting
  - Alpha blending with depth-ordered front-to-back traversal
  - Full backward pass: ∂L/∂color, ∂L/∂alpha, ∂L/∂conic, ∂L/∂mean2D, ∂L/∂SH, ∂L/∂scale, ∂L/∂rotation
  - **FLAME mesh binding** — barycentric coordinate binding with TBN projection
  - **FLAME binding backward pass** — ∂L/∂position → ∂L/∂local_offset for mesh-regularized training
  - **35 gradient verification tests** — numerical vs analytical comparison with <1e-3 relative error
  - Buffer pool for memory-efficient workload management (90%+ allocation reduction)
  - CPU reference rasterizer for gradient validation
  - GPU debug feature flag for validation layers

- **oxigaf-trainer** — Training and optimization pipeline
  - Gaussian model initialization from FLAME mesh surface
  - Adam optimizer with per-parameter group learning rates
  - Comprehensive loss functions:
    - L1 photometric loss
    - MS-SSIM (Multi-Scale SSIM) perceptual loss
    - LPIPS (Learned Perceptual Image Patch Similarity) — Pure Rust VGG network
    - Score Distillation Sampling (SDS) for diffusion guidance
    - Normal consistency loss
    - Opacity and scale regularization
    - Position regularization (binding to FLAME mesh)
  - Adaptive density control (split/prune/clone operations)
  - Checkpoint saving and loading
  - **TensorBoard integration** — scalar, image, histogram, and graph logging
  - Diffusion target generation for pseudo-GT supervision
  - Pipeline orchestration with modular stages and progress tracking

- **oxigaf-bridge** *(new crate)* — PyTorch ↔ OxiGAF weight conversion
  - Bidirectional weight conversion: PyTorch → OxiGAF and OxiGAF → PyTorch
  - Layer name mapping with custom overrides
  - Precision conversion (FP32 ↔ FP16 ↔ BF16)
  - Safetensors-based checkpoint interoperability
  - Validation utilities for conversion correctness
  - CLI examples: `convert_gaf_checkpoint`, `batch_convert`, `validate_conversion`

- **oxigaf-cli** — Command-line interface
  - `convert` — Convert between 3D formats and weight formats. **Historical
    note (added retroactively):** at 0.1.0 the `.pkl` input path
    (`convert_pkl` in `oxigaf-cli/src/convert.rs`) was structurally unable to
    succeed against real FLAME `.pkl` files — only the `.npz` half of this
    command actually worked, despite `convert` being listed as shipped
    above. `convert_pkl` now decodes the pickle stream with a pure-Rust
    virtual machine (`pickle::load_arrays`, understanding protocols 0–5 and
    reconstructing `numpy.ndarray` / `numpy.dtype` / `chumpy.ch.Ch` /
    `scipy.sparse` payloads) instead of the earlier approach; see `[0.1.2]`
    above.
  - `train` — Train Gaussian Avatar models
  - `render` — Render images from trained models
  - `export` — Export to standard formats
  - `benchmark` — Performance benchmarking
  - `doctor` — System diagnostics
  - Configuration hierarchy (file, environment, CLI args)
  - Progress bars and interactive prompts
  - JSON output mode for scripting
  - Log rotation and verbosity control
  - HuggingFace Hub integration
  - Asset caching system with LRU eviction

- **oxigaf** — Unified meta-crate
  - Re-exports all core APIs via `pub use`
  - Comprehensive `prelude` module with 40+ re-exported types
  - Unified `OxigafError` enum wrapping all sub-crate errors
  - Feature flag orchestration (pass-through to all sub-crates)
  - Extensive documentation: Quick Start, Data Flow diagram, Migration guide from Python GAF
  - 4 runnable examples: `basic_flame`, `gaussian_render`, `training_loop`, `diffusion_inference`

#### Development Infrastructure

- Comprehensive test suite (796 tests across all crates, all passing)
- Property-based testing with proptest
- Benchmark suite using criterion
- Examples demonstrating core workflows
- GitHub Actions CI/CD configuration
- Documentation with rustdoc examples

#### Documentation

- Crate-level documentation for all modules
- API documentation with examples
- README.md with installation and usage
- CHANGELOG.md following Keep a Changelog format
- Licensing: Apache-2.0

### Technical Highlights

- **Pure Rust Implementation** — 100% Pure Rust (no C/Fortran dependencies by default)
- **COOLJAPAN Ecosystem** — Uses oxiarc-archive instead of zip; oxiblas instead of openblas; OxiFFT instead of rustfft
- **512×512 Multi-View Generation** — Latent upsampler + IP-Adapter + CFG pipeline fully integrated
- **Verified Gradients** — 35 gradient verification tests; numerical and analytical gradients match (<1e-3 relative error) across all parameters
- **PyTorch Interoperability** — Bidirectional weight conversion via `oxigaf-bridge` crate
- **Performance Optimizations**:
  - SIMD acceleration for FLAME operations (2-4× speedup, feature-gated)
  - Flash Attention for memory-efficient diffusion
  - Specialized SH shaders (10× speedup via compile-time degree specialization)
  - Buffer pool for GPU memory efficiency (90%+ allocation reduction)
  - Parallel batch processing with rayon (near-linear with CPU cores)
  - GPU-accelerated rendering with wgpu 28
- **Code Quality**:
  - Zero `unwrap()` in production code
  - Zero warnings policy (clippy + rustc)
  - Workspace-based dependency management
  - All files under 2000 lines
  - 796 tests — 100% passing

### Statistics

- **Total Lines of Code**: ~38,000 (Rust + WGSL)
- **Crates**: 7 publishable crates
- **Test Coverage**: 796 tests (100% passing)
- **Gradient Verification Tests**: 35 (all passing, <1e-3 relative error)
- **Development Effort**: ~15 months (COCOMO estimate)

### Dependencies

- Linear Algebra: nalgebra 0.34, glam 0.31
- Deep Learning: candle-core 0.9, candle-nn 0.9, candle-transformers 0.9
- GPU Compute: wgpu 28
- Image Processing: image 0.25 (with EXR support)
- Serialization: serde 1, safetensors 0.7
- Error Handling: anyhow 1, thiserror 2
- CLI: clap 4 (with derive, env features)
- Async: tokio 1 (full features)
- Testing: approx 0.5, proptest 1, criterion 0.8

[0.1.1]: https://github.com/cool-japan/oxigaf/releases/tag/v0.1.1
[0.1.0]: https://github.com/cool-japan/oxigaf/releases/tag/v0.1.0
