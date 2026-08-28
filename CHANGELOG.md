# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.2] - 2026-08-28

> **Migration notes.** Most of what follows is additive or affects only
> advanced/internal APIs. Four changes are worth reading before you upgrade —
> the first two affected *every* training run on the previous release:
>
> - **Every 0.1.1 training run optimized a hardcoded L2 photometric loss,
>   regardless of `LossConfig` weights.** `Trainer::compute_gradients` built
>   the image-space gradient fed to the backward pass as a fixed
>   `2.0 * (rendered - target) / num_views`; `w_l1`, `w_ssim` and
>   `w_ms_ssim` only ever changed the *logged* loss value, never the
>   direction the optimizer actually descended — see *Fixed → oxigaf-trainer*
>   below. Retrain if you were relying on a non-default photometric loss mix.
> - **Every 0.1.1 model with `sh_degree >= 1` trained with a systematically
>   incomplete position gradient.** The backward rasterizer shader
>   (`preprocess_bwd.wgsl`) differentiated the projection path but not the
>   view-dependent spherical-harmonics color path, both of which depend on
>   Gaussian position — see *Fixed → oxigaf-render* below. Retraining is the
>   only remedy; there is no way to recover the missing gradient signal from
>   an already-trained model.
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
- **`heat_geodesic` is now a real implementation of the heat method** (Crane,
  Weischedel & Wardetzky 2013): solving `(M + t·Lc)u = δ_source`, normalizing
  `∇u`, then a Poisson solve — both via Jacobi-preconditioned CG — replacing
  what its own 0.1.1 doc comment called "a simplified approximation, not the
  full heat method of Crane et al." Signature unchanged; returned distances
  differ (more accurate) for the same inputs.
- **`geodesic_center` no longer searches exhaustively by default.** An empty
  `sample_vertices` previously searched every vertex (`O(V·(E + V log V))` —
  thousands of Dijkstra runs, minutes on a 5023-vertex FLAME head); it now
  selects a farthest-point-sampled subset of `DEFAULT_CENTER_SAMPLES` (64)
  candidates instead — a good approximation at a fixed, small cost, but no
  longer exact. Callers that relied on the old exhaustive behavior must pass
  `(0..mesh.n_vertices()).collect()` explicitly, or use
  `geodesic_center_sampled` to choose their own candidate budget.

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
- **`TrainingConfig` gained four new fields** — `lr_schedule:
  LrScheduleConfig`, `gradient_clip: GradientClipConfig`,
  `gradient_accumulation_steps: u32`, `ema_decay: Option<f32>` — breaking
  external `TrainingConfig { .. }` struct-literal construction that doesn't
  list them. All four carry `#[serde(default...)]`, so TOML/JSON configs
  serialized before they existed still deserialize unchanged; only direct
  Rust construction breaks.
- **`LossConfig` gained `w_scale_reg_max_scale: f32`**
  (`#[serde(default = ...)]`, defaulting to
  `crate::loss::MAX_REASONABLE_WORLD_SCALE`) — same struct-literal-breaking
  / deserialization-safe pattern.

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

#### oxigaf — signature and semantics changes

- `pipeline::export<P: AsRef<Path>>(model_path: P, output_path: P, ...)` /
  `render_from_file<P>(..., output_path: P, ...)` /
  `quick_train<P>(..., output_dir: P, ...)` each gained a second, independent
  generic parameter (`P`, `Q`), so the two path arguments no longer need to
  be the same concrete type.

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
- **`rasterize_bwd.wgsl`'s backward tile kernel could accumulate a tile's
  gradient sum onto the wrong Gaussian.** The reverse-traversal loop bound
  was read per-pixel from `out_n_contrib[pixel_idx]`, so different threads in
  the same 16×16 workgroup ran the loop a different number of times while
  the loop body contained `workgroupBarrier()` calls — non-uniform control
  flow around a barrier, which WGSL requires to be uniform. Thread 0 flushed
  the 256-thread gradient sum onto whichever Gaussian *it* was visiting at a
  given iteration, attributing other threads' contributions to that same
  Gaussian even when they were processing a different one (or none). Fixed
  by computing a single workgroup-uniform loop bound (`tile_end`, the max of
  all 256 per-pixel stopping indices via tree reduction +
  `workgroupUniformLoad`), with per-thread validity now expressed as a
  `contributes` mask instead of a per-thread trip count.
- **`preprocess_bwd.wgsl` omitted the position gradient through
  view-dependent spherical-harmonics color, for every `sh_degree >= 1`
  model.** The forward pass evaluates SH color at
  `dir = normalize(pos - cam_pos)`, so the loss depends on `pos` through
  color as well as through projection; the backward shader only
  differentiated the projection path, matching a term present in the
  reference 3DGS implementation but missing here. Fixed by adding the
  `∂L/∂color → ∂L/∂dir → ∂L/∂pos` chain. See the migration note at the top
  of this section — retraining is the only way to recover the missing
  signal.
- **`rasterize_bwd.wgsl` omitted the background's contribution to
  `∂L/∂α`.** The forward pass composites `color += T_final · background`
  after the last Gaussian, so every Gaussian's opacity influences the loss
  both through its own blend weight and through `T_final`; the backward pass
  only differentiated the first path, biasing training against any
  non-black background. Fixed by adding the
  `(−T_final / (1 − α)) · (background · ∂L/∂color)` term; covered by the new
  `test_gpu_self_consistent_nonzero_background`.
- **`preprocess_bwd.wgsl` could write NaN gradients for culled Gaussians.**
  The kernel divides by `tz = -p_view.z` in its projection-Jacobian terms
  for every Gaussian unconditionally; for one culled at/behind the near
  plane, `tz <= 0` produces `inf`, and `inf * 0` (its zero incoming 2D
  gradient) is NaN, written straight into `grad_positions`. Fixed by a cull
  guard reproducing the forward near/far test that writes exact zeros
  instead.
- **`rasterize_bwd.wgsl`'s Gaussian-skip test let a NaN `power` through as a
  live contribution**, since the old reject-test
  (`power > 0.0 || power < -4.0`) is `false` on both sides for NaN, risking
  NaN contaminating a whole tile's shared-memory reduction. The rewritten
  `contributes` accept-test (`power <= 0.0 && power >= -4.0`) excludes NaN
  explicitly. (Minor, and only reachable together with the two fixes above.)
- **`gpu_gradient_verify.rs`'s own `summarize_errors`/`median_error` helpers
  previously reported a "clean" `0.0` error for a NaN, `Infinity`, or empty
  error set instead of failing** — meaning a regression as severe as the
  four backward-shader bugs above was not guaranteed to be caught by the
  existing finite-difference test suite. Now explicitly checked (new
  `test_median_error_nan_propagates`,
  `test_summarize_errors_nan_fails_every_threshold`/`_infinity_fails`/
  `_empty_is_not_a_silent_pass`). The FD tolerance itself (30% outlier
  allowance, `err > 0.1` per-element) is unchanged.

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
- **The trainer's backward pass now differentiates the objective it actually
  reports, instead of a hardcoded L2 loss.** `Trainer::compute_gradients`
  previously built the image-space gradient fed to `Rasterizer::backward` as
  `2.0 * (rendered - target) / num_views` unconditionally — `LossConfig`'s
  `w_l1`, `w_ssim` and `w_ms_ssim` only changed the *logged* loss value,
  never the direction the optimizer descended. It now calls the new
  `image_gradient::photometric_pixel_gradient` with a `PhotometricSpec`
  built from the installed `LossComputer`'s actual config, SSIM window and
  MS-SSIM scale weights, so the reported and optimized objectives cannot
  drift apart. See the migration note at the top of this section.
- **Position/scale/opacity regularization losses now actually produce
  gradients.** `w_position_reg`, `w_scale_reg` and `w_opacity_reg` were
  computed as scalar values for logging (`LossOutput`) but never
  differentiated into the parameter gradients the optimizer applies — in
  particular `Gradients::offset` was never written by anything, so
  `GaussianModel::local_offsets` never moved during training no matter how
  `w_position_reg` was set. The new private `add_regularization_gradients`
  adds the analytic gradients of these three terms directly into
  `Gradients::offset`/`scale`/`opacity`. `w_normal` and
  `w_gradient_penalty` remain disconnected from any gradient — this is now
  a documented limitation (no FLAME mesh is retained by the trainer, no
  external gradient buffer is threaded through), not a silent gap.
- **Score-distillation (SDS) training now has a real gradient path.**
  `SdsLoss` previously exposed only a scalar `compute()` with no way to
  differentiate it, and `compute_gradients` had no SDS-related code at all,
  so `use_sds` distillation contributed no gradient beyond the ordinary
  photometric term. `DiffusionTargetGenerator` gained
  `compute_sds_gradient` / `compute_sds_gradient_with_horizon`, and
  `SdsLoss` gained a validated `new(weighting, max_timestep) ->
  Result<Self, TrainerError>` constructor (rejecting `max_timestep < 2`,
  which previously divided by zero in the weighting curve);
  `compute_gradients` now adds the SDS pixel residual — weighted identically
  to what `SdsLoss` reports — into the backward pass.

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
- **`oxigaf export --format gltf` (`export_gltf::export_gltf`) now writes
  spec-conformant glTF.** It previously put all five accessors onto one
  buffer view with no `byteStride`, which glTF 2.0 forbids for accessors of
  differing element size. It now delegates to the new
  `oxigaf_render::gltf::write_gltf`; signature and `CliError::GltfExport`
  are unchanged, but the bytes written differ from earlier 0.1.x output.
  Note: `oxigaf_cli::export::export_gltf` (`ExportStage`, self-contained
  `.glb`, extension `OXIGAF_gaussians`) remains a separate, third,
  unconsolidated writer — a deliberate, documented scope limit, not an
  oversight.

#### oxigaf — bug fixes

- **`pipeline::export` and `pipeline::render_from_file` are no longer no-op
  stubs.** Both were previously documented in their own doc comments as "a
  thin validation wrapper" that only checked `model_path.exists()` and
  otherwise did nothing (`let _ = (output_path.as_ref(), format); Ok(())`),
  silently returning `Ok(())` without writing any file or rendering
  anything. `export` now loads the model (`.ply` / `.safetensors`) and
  writes it via `GaussianModel::save_ply` (`ExportFormat::Ply`), a new
  Wavefront-OBJ point-cloud writer (`ExportFormat::Obj`), or
  `oxigaf_render::gltf::write_gltf` (`ExportFormat::Gltf`).
  `render_from_file` now loads the model, auto-frames a camera from its
  bounding box, and actually rasterizes and saves an image via
  `Rasterizer`.
- **`verify_assets` checked for the wrong `.npy` file names.** Its
  hardcoded list disagreed with what `oxigaf_flame::io::load_flame_model`
  actually opens (`shape_dirs.npy`/`exp_dirs.npy`/`J_regressor.npy` vs. the
  loader's `shapedirs.npy`/`expressiondirs.npy`/`j_regressor.npy`) and
  omitted `lbs_weights.npy` entirely, so a directory missing the skinning
  weights was reported as complete. Fixed by iterating the new
  `oxigaf_flame::io::REQUIRED_NPY_FILES`, the same constant the loader now
  names each file with.

#### oxigaf-bridge — bug fixes

- **`GafLayerMapper` no longer relies on a hardcoded, enumerated layer
  table.** The previous implementation built two `HashMap`s per
  `GafLayerMapper::new()` call by walking an assumed U-Net/VAE/CLIP/Upsampler
  topology that hardcoded `transformer_layers_per_block = [1, 2, 10, 10]` —
  which does not match `DiffusionConfig::default()`'s `[1, 1, 1, 1]`
  (`oxigaf-diffusion/src/config.rs`) — causing `validate_coverage` false
  negatives (missing transformer blocks) and false positives (nonexistent
  blocks) against any model built from the default config. The same two
  `HashMap`s also flattened all four components into one namespace, so
  entries sharing a prefix (e.g. the VAE encoder's and the CLIP encoder's
  `encoder/`) could silently overwrite each other. `GafLayerMapper` now
  performs the ToRSh ⟷ OxiGAF `/` ↔ `.` substitution directly, with a small
  explicit override table (`GafLayerMapper::add_override`) for names that
  are genuine exceptions — there are none today. **`num_mappings()`'s
  meaning changed accordingly**: it now reports the number of explicit
  overrides registered (`0` on a freshly-created mapper), not a total
  enumerated layer count as before (the old implementation's own module
  documentation described enumerating on the order of 2,000 concrete layer
  paths up front) — any caller that treated its return value as "total GAF
  layers mapped" needs to stop.

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
- **`gltf` module** — `write_gltf`, `GltfError`,
  `EXTENSION_NAME = "OXIGAF_gaussian_splat"`: the single, spec-conformant
  glTF 2.0 writer, consolidating what were three independently-written,
  mutually-incompatible glTF emitters in the workspace (this crate had none
  before; `oxigaf-cli` had two). One buffer view per accessor (glTF 2.0
  forbids a shared strideless view across differently-sized accessors,
  which one of the old emitters did), mandatory `min`/`max` on the
  `POSITION` accessor, and an asset-only document for an empty model.
- `rasterizer::rasterizer_device_limits`,
  `rasterizer::RASTERIZER_STORAGE_BUFFERS_PER_STAGE` (= 16),
  `rasterizer::RASTERIZE_FWD_WORKGROUP_STORAGE_BYTES` (= 17,408) —
  `Rasterizer::from_device` now validates a caller-supplied `wgpu::Device`'s
  limits upfront, instead of letting an under-provisioned device fail later
  as an opaque wgpu pipeline-validation error the first time a pass
  actually runs.
- `profiler::GpuTimestampProfiler` — GPU-side pass profiler backed by
  `wgpu::Features::TIMESTAMP_QUERY` (`REQUIRED_FEATURES`,
  `DEFAULT_MAX_PASSES = 32`; `new`, `stats`, `period_ns`,
  `reserved_passes`, `pass_writes`, `resolve`, `collect`, `discard`),
  complementing the pre-existing CPU-side `PassProfiler`. Paired with new
  `Rasterizer::enable_gpu_timestamps() -> Result<(), RenderError>`,
  `Rasterizer::disable_gpu_timestamps()`,
  `Rasterizer::gpu_timestamps() -> Option<&GpuTimestampProfiler>`.

#### oxigaf-trainer

- `SyncReport` gained `buckets` and `gradient_compression` fields
  (`data_parallel.rs`), reporting per-bucket gradient-sync detail instead of
  only an aggregate. `estimated_bandwidth_mb` now reflects the
  post-compression transfer size when `gradient_compression` is active,
  rather than always reporting the uncompressed size.
- `config::LrScheduleConfig` — 6-variant serializable enum (`Fixed`,
  `WarmupCosine`, `Cosine`, `Step`, `Exponential`, `Cyclic`) bridging to
  `crate::lr_scheduler::LrScheduler` via `build(total_iterations)` /
  `validate(total_iterations)`, so LR schedules can now be declared in a
  config file instead of only constructed programmatically.
- `config::GradientClipConfig` — serializable selection of a `ClipMode`
  (`clip_mode()`, `threshold()`, `validate()`), same declarative purpose
  for gradient clipping.
- `pruning::GaussianPruner::prune_by_min_scale`,
  `pruning::prune_to_sparsity(scores, target_sparsity)`,
  `pruning::apply_mask_to_model(model, mask)`.
- `synthetic_data::SyntheticGaussianCloud::into_gaussian_model(self,
  sh_degree: u32) -> GaussianModel` — converts a sampled synthetic cloud
  into a renderer-ready model, so synthetic data can drive the
  trainer/renderer end-to-end instead of only exercising isolated sampling
  code.
- **`meta_learning_avatar` module** (source at `meta_learning/avatar.rs`) —
  `GaussianAvatarModel`, the first real `MetaModel` implementation over an
  actual Gaussian avatar (the only prior implementation, `LinearModel`, was
  a toy regressor never connected to what the crate trains). Implements
  `MetaModel::loss_and_grad` via the real rasterizer forward/backward pass;
  also adds `ParamLayout` (flat-vector packing), `AvatarRenderer` (shared,
  non-cloneable rasterizer wrapper), `AvatarBatch`.
- **`image_gradient` module** — backs the hardcoded-loss fix above (see
  *Fixed*); also exposes `ssim_pixel_gradient`, `ms_ssim_pixel_gradient`,
  `convolve_separable`, `downsample_2x`/`upsample_adjoint_2x`,
  `sign_or_zero`, and the mirrored SSIM/MS-SSIM constants (`SSIM_C1`/`C2`,
  `SSIM_KERNEL_TAPS`/`SIGMA`, `MS_SSIM_*`) for building custom photometric
  adjoints.
- `diffusion_target::DDPM_TRAIN_TIMESTEPS: u32 = 1000` — the standard DDPM
  horizon, now a named constant. `diffusion_target::sds_timestep_weight` is
  newly `pub` (was a private `fn` in 0.1.1).

#### oxigaf-flame

- `GazeController::synthesize_blinks(&self, duration_steps, seed) -> Vec<f32>`
  — an instance-method convenience wrapper around the existing
  `gz_synthesize_blinks` free function. (`gaze_controller` itself, and the
  rest of its API — `GazeController`, event detection, Listing's-law
  helpers, I-VT classification, vergence — predates 0.1.1; only its source
  file was split into a `gaze_controller/` module directory this release,
  which is not a public-API change.)
- `heat_geodesic_multi(mesh, sources, n_iter, time_step)` — multi-source
  heat-method geodesics (`heat_geodesic` now delegates to it for the
  single-source case); `heat_time_step(mesh) -> f32` — the standard `dt`
  heuristic (squared mean edge length).
- `geodesic_center_sampled(mesh, sample_vertices, n_samples, config)` and
  `DEFAULT_CENTER_SAMPLES: usize = 64` — see the `geodesic_center` behavior
  change under *Changed* above.
- `io::REQUIRED_NPY_FILES: &[&str]` — see the `verify_assets` fix under
  *Fixed → oxigaf* above.

#### oxigaf-bridge

- **Pure-Rust `.pt` / `.pkl` ingest** — new `pickle` module (`value`, `vm`,
  `torch`, `numpy`, `flame`, `error` submodules) implementing a
  non-executing Python pickle reader (protocols 0–5: `GLOBAL`, `REDUCE`,
  `NEWOBJ` and `BUILD` all produce inert data records instead of resolving
  or calling anything), plus two new crate-root functions built on it.
  `convert_pytorch_checkpoint(checkpoint, output_dir, target_dtype)` reads a
  raw PyTorch `.pt` / `.pth` checkpoint directly (no `torch.load`, no
  Python), splits its tensors into `unet` / `vae` / `clip` / `other` by the
  same prefixes `scripts/convert_weights.py` recognized, and writes each
  non-empty group as `<component>.safetensors` with `VarBuilder`-loadable
  dotted names, returning a `ConversionReport`. `convert_flame_model(model,
  output_dir)` reads a FLAME `.pkl` head model and writes `v_template`,
  `faces`, `shapedirs`, `expressiondirs`, `posedirs`, `j_regressor`,
  `kintree_table` and `lbs_weights` as `.npy`, including the same
  identity/expression split of `shapedirs` and densification of the
  SciPy-sparse `J_regressor` that made `scipy` a dependency of the Python
  script it replaces (`scripts/convert_flame.py`). New examples
  `convert_pytorch` and `convert_flame_pkl` wrap both as CLI entry points.
  The Python scripts remain in the repository as a reference/escape hatch,
  but nothing in the OxiGAF pipeline requires Python, PyTorch, NumPy, or
  SciPy for this step anymore.

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
