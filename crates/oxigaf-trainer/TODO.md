# TODO for oxigaf-trainer

**Target: 0.1.2 (2026-08-28).** One-person project; contributions welcome.

## ✅ Completed (from plan)

### Core Training Loop
- ✅ Main `Trainer` struct with orchestration (1,921 lines)
- ✅ Iterative denoising distillation loop
- ✅ Gaussian initialization on FLAME mesh (`init.rs`, 533 lines)
- ✅ Per-parameter Adam optimizer with group-wise learning rates (1,402 lines)
- ✅ Learning rate scheduling (exponential decay), plus opt-in multiplier
  schedules wired into `Trainer` as of v0.1.2 (see below)
- ✅ Checkpoint save/load (JSON + flat f32 arrays, 1,028 lines)

### Loss Functions (1,879 lines)
- ✅ Photometric loss (L1 + SSIM + MS-SSIM blend) — as of v0.1.2, the same
  weights that produce the logged loss also produce the backward-pass
  gradient (`image_gradient.rs`, 833 lines); see "Completed (v0.1.2)" below
- ✅ SSIM / MS-SSIM computation (structural similarity)
- ✅ **LPIPS** perceptual loss (805 lines, Pure Rust VGG network)
- ✅ Position regularization (binding to FLAME mesh) — real gradient as of
  v0.1.2
- ✅ Scale regularization (prevent oversized Gaussians) — real gradient as
  of v0.1.2, configurable ceiling (`w_scale_reg_max_scale`)
- ✅ Opacity regularization (sparsity) — real gradient as of v0.1.2
- ✅ Normal consistency loss (logging-only — see "Completed (v0.1.2)" below)

### Adaptive Density Control (638 lines)
- ✅ Clone Gaussians (high gradient, small scale)
- ✅ Split Gaussians (high gradient, large scale)
- ✅ Prune Gaussians (low opacity, large screen size)
- ✅ Gradient accumulation tracking
- ✅ Opacity reset schedule
- ✅ Additional pruning utilities (v0.1.2): `prune_by_min_scale`,
  `prune_to_sparsity`, `apply_mask_to_model` (`pruning.rs`)

### Metrics & Logging
- ✅ PSNR + SSIM tracking (`metrics.rs`, 252 lines)
- ✅ SSIM metric tracking
- ✅ **TensorBoard integration** (1,307 lines - EXCEEDS PLAN!)
  - Scalar logging (loss, metrics)
  - Image logging (renders, pseudo-GT)
  - Histogram logging (parameters, gradients)
  - Graph logging
- ✅ Metric history tracking

### Diffusion Target Generation (1,585 + 431 lines)
- ✅ `diffusion_target.rs` (1,585 lines) + `diffusion_target/tensor_ops.rs` (431 lines)
- ✅ Pseudo-GT generation from current renders
- ✅ Normal map conditioning (`generate_targets_with_normals`, v0.1.2)
- ✅ Camera pose integration
- ✅ Multi-view consistency (`ViewConsistencyLoss`, gained `enable_warping`
  in v0.1.2)
- ✅ Score-distillation (SDS) gradient path (v0.1.2 — see "Completed
  (v0.1.2)" below)

### Testing
- ✅ **3,271 tests** (3,259 passing, 12 ignored GPU/slow) - EXCEEDS PLAN
- ✅ Unit tests across all modules
- ✅ Integration tests
- ✅ Property-based tests (proptest)
- ✅ Video input tests (15 new)

### Code Quality
- ✅ No unwrap policy
- ✅ The 2,000-line file cap violation previously flagged here is resolved:
  `anomaly_detection.rs` (was 2,168 lines) and `domain_adaptation.rs` (was
  2,271 lines) are now module directories (`anomaly_detection/{checks,
  detector,report,stats,types,tests}.rs`,
  `domain_adaptation/{batch,common,config,coral,dann,mmd,self_training,
  stats,tests}.rs`). No file under `src/` exceeds 2,000 lines as of 0.1.2
  (largest: `few_shot_adaptation.rs` at 1,990).
- ✅ Total: ~97,700 lines across 91 source files in `src/` (up from the
  ~92,200 / 70 previously recorded here — grew via the v0.1.1 training
  regime/gradient-tool/optimisation-utility modules and the v0.1.2 items
  below)
- ✅ Comprehensive error handling

## ✅ Completed (v0.1.1)

### Extended Loss Functions (v0.1.1)
- ✅ **Adaptive loss** — task-uncertainty-weighted loss scaling
- ✅ **Contrastive loss** — NT-Xent / InfoNCE contrastive learning
- ✅ **Multi-resolution loss** — image pyramid loss accumulation
- ✅ **Loss reweighting** — dynamic per-sample loss weight adjustment
- ✅ **Loss landscape visualisation** — 2D loss surface visualisation

### Advanced Training Regimes (v0.1.1)
- ✅ **Curriculum learning** — difficulty-ordered training schedule
- ✅ **Progressive training** — resolution or complexity ramp-up
- ✅ **Few-shot adaptation** — K-shot fine-tuning utility
- ✅ **Meta-learning (MAML)** — model-agnostic meta-learning
- ✅ **Continual learning** — EWC-based catastrophic forgetting prevention

### Gradient Tools (v0.1.1)
- ✅ **Gradient accumulation** — micro-batch gradient accumulation
- ✅ **Gradient clipping** — global and per-layer norm clipping
- ✅ **Gradient flow analysis** — per-layer gradient magnitude tracking
- ✅ **Gradient surgery** — conflict-resolving multi-task gradient projection

### Online Learning & Model Analysis (v0.1.1)
- ✅ **Online learning** — streaming mini-batch online update loop
- ✅ **OHEM** — online hard example mining for imbalanced data
- ✅ **Activation maps** — grad-CAM and feature activation visualisation
- ✅ **Anomaly detection** — out-of-distribution sample detection
- ✅ **Convergence analysis** — loss trend and plateau detection
- ✅ **Layer freezing** — selective parameter freeze/unfreeze
- ✅ **Model pruning** — magnitude-based weight pruning

### Optimisation Utilities (v0.1.1)
- ✅ **Spectral normalisation** — Lipschitz-constrained weight normalisation
- ✅ **Stochastic weight averaging (SWA)** — flat-minima weight averaging
- ✅ **Custom optimiser** — pluggable optimiser trait with schedulers
- ✅ **Temperature scaling** — post-hoc calibration for confidence

### Data Pipeline (v0.1.1)
- ✅ **Data augmentation** — geometric and photometric augmentation
- ✅ **Camera sampling** — random camera pose sampling for training
- ✅ **Synthetic data generation** — procedural training data synthesis
- ✅ **Noise injection** — structured noise for robustness training

### Session & Transfer Learning (v0.1.1)
- ✅ **Session recorder** — training session replay and export
- ✅ **Training callbacks** — per-step and per-epoch hooks
- ✅ **Checkpoint interpolation** — weight-space interpolation between checkpoints
- ✅ **Knowledge distillation** — soft-label teacher-student distillation
- ✅ **Domain adaptation** — adversarial domain alignment
- ✅ **Feature bank** — momentum-updated feature memory bank

## ✅ Completed (v0.1.2)

### Training-loop correctness fixes
- ✅ **Photometric loss now actually optimized.** `Trainer::compute_gradients`
  previously built the backward-pass image gradient as a hardcoded
  `2.0 * (rendered - target) / num_views` regardless of `LossConfig` —
  `w_l1`/`w_ssim`/`w_ms_ssim` only ever changed the *logged* loss value,
  never the direction the optimizer descended. Fixed via the new
  `image_gradient::photometric_pixel_gradient` + `PhotometricSpec`, built
  from the installed `LossComputer`'s real config, SSIM window, and
  MS-SSIM scale weights. **This affected every 0.1.1 training run whenever
  `LossConfig` deviated from the implicit hardcoded L2 mix — retrain if you
  relied on non-default weights.**
- ✅ **Position/scale/opacity regularization now produces real gradients.**
  `w_position_reg`/`w_scale_reg`/`w_opacity_reg` were computed as scalars
  for logging only — `Gradients::offset` was never written by anything, so
  `local_offsets` never moved during training regardless of
  `w_position_reg`. Fixed via the new (private) `add_regularization_gradients`.
  **`w_normal` and `w_gradient_penalty` remain logging-only** — documented
  limitation (no FLAME mesh retained by the trainer, no external gradient
  buffer threaded through for either term), not a silent gap.
- ✅ **Score-distillation (SDS) training has a real gradient path.**
  `use_sds` previously contributed no gradient beyond the ordinary
  photometric term. Fixed via
  `DiffusionTargetGenerator::compute_sds_gradient` /
  `compute_sds_gradient_with_horizon`, and a validated
  `SdsLoss::new(weighting, max_timestep) -> Result<Self, TrainerError>`
  (rejects `max_timestep < 2`, which previously divided by zero in the
  weighting curve).
- ✅ `decompose_uncertainty` no longer fabricates an aleatoric/epistemic
  split for a single-point sample — reports `aleatoric == 0.0`,
  `epistemic == total` instead of an arbitrary ~0.64/0.36 split with no
  statistical basis. `decompose_uncertainty_with_variances` is the real
  multi-sample path.
- ✅ `CurriculumSchedule::validate` now rejects invalid stage ordering
  (`step_start > step_end`, or a gap left between consecutive stages)
  instead of producing a schedule that silently skips or inverts a stage
  at runtime.

### Newly wired into `Trainer` (previously standalone-only)
- ✅ **Learning-rate schedules** — `config::LrScheduleConfig` (6 variants:
  `Fixed`, `WarmupCosine`, `Cosine`, `Step`, `Exponential`, `Cyclic`), built
  by `Trainer::new` from `TrainingConfig::lr_schedule` and applied every
  step. Off by default (`Fixed`).
- ✅ **Gradient clipping** — `config::GradientClipConfig` (`Disabled`,
  `GlobalNorm`, `PerGroupNorm`, `Value`, `Adaptive`), wired the same way via
  `TrainingConfig::gradient_clip`. Off by default.
- ✅ **Gradient accumulation** and **EMA shadow weights** — both existed
  only as standalone, self-driven APIs before this release
  (`gradient_accumulation.rs`, `ema.rs`); now built by `Trainer::new` and
  consumed inside `train_step` via
  `TrainingConfig::gradient_accumulation_steps` / `ema_decay`. Both
  off/no-op by default. New `wired_modules_tests.rs` pins the wiring for
  all four of the above.

### New APIs
- ✅ `pruning::GaussianPruner::prune_by_min_scale`, `prune_to_sparsity`,
  `apply_mask_to_model`
- ✅ `synthetic_data::SyntheticGaussianCloud::into_gaussian_model` — bridges
  a sampled synthetic cloud to a renderer-ready `GaussianModel`, so
  synthetic data can drive the trainer/renderer instead of only exercising
  isolated sampling code
- ✅ **`meta_learning_avatar` module** (`GaussianAvatarModel`) — the first
  real `MetaModel` implementation over an actual Gaussian avatar (the only
  prior implementation, `LinearModel`, was a disconnected toy regressor);
  implements `MetaModel::loss_and_grad` via the real rasterizer
  forward/backward pass. Also adds `ParamLayout`, `AvatarRenderer`,
  `AvatarBatch`.
- ✅ **`image_gradient` module** — backs the photometric-loss fix above;
  also exposes `ssim_pixel_gradient`, `ms_ssim_pixel_gradient`,
  `convolve_separable`, `downsample_2x`/`upsample_adjoint_2x`,
  `sign_or_zero` for building custom photometric adjoints.
- ✅ `LossConfig::w_scale_reg_max_scale` — configurable world-space scale
  ceiling for the scale regularizer (previously a hardcoded constant).
- ✅ `SyncReport` gained `buckets` / `gradient_compression` fields
  (per-bucket gradient-sync detail, `data_parallel.rs`);
  `ViewConsistencyLoss::enable_warping`;
  `DiffusionTargetGenerator::generate_targets_with_normals` /
  `view_consistency_loss()` / `target_config()`;
  `diffusion_target::DDPM_TRAIN_TIMESTEPS`; `sds_timestep_weight` now `pub`.
- ✅ `detect_convergence_phase` gained a `loss_scale` parameter (4→5 args);
  `ConvergenceConfig` gained `oscillation_threshold`, `max_history`.

### Code quality
- ✅ `anomaly_detection.rs` / `domain_adaptation.rs` split via splitrs into
  module directories — no `src/` file exceeds 2,000 lines any more (see
  "Code Quality" above).

## 🚧 In Progress

None currently tracked. The training-loop correctness fixes for 0.1.2
(photometric loss, regularization gradients, SDS gradient path — see
"Completed (v0.1.2)" above) are done and covered by tests.

## 📋 Planned (gaps from original design)

### Full Pipeline Integration
- ⬜ **End-to-end training test** with all components — still needs either
  real model weights or an integration test built on the new (v0.1.2)
  `synthetic_data::SyntheticGaussianCloud::into_gaussian_model` bridge,
  which produces a renderer-ready `GaussianModel` from synthetic data but
  is not yet exercised by a `Trainer`-driving integration test
- ✅ **Video input support** (`video.rs`)
  - `VideoConfig` (sequence_dir, max_frames, frame_stride, loop_sequence, shuffle_frames, shuffle_seed)
  - `VideoFrameIterator` (next_frame, next_batch, frame_at, reset, total_frames)
  - `FrameBatch`; uses `FlameSequence` from oxigaf-flame
  - `TrainerError::SequenceError` added
  - 15 new tests
- ✅ **Guidance scale annealing**
  - `guidance_scale_start` (7.5), `guidance_scale_end` (3.0), `guidance_anneal_steps` (10_000) added to `DiffusionTargetConfig`
  - `annealed_guidance_scale(step)` method on `DiffusionTargetGenerator`
  - Linear decay from start to end
  - 7 new tests

### Missing Features from Plan
- ✅ **Camera sampling strategy** (`camera_sampling.rs`)
  - Random, Spiral (Fibonacci), Hemisphere grid, Turntable strategies
  - `CameraSampler::sample(n, rng)` and `sample_iter(iter, total, rng)`
  - `CameraView::view_matrix()` for column-major look-at 4×4 matrix
  - 10 tests (all passing)
- ✅ **Gradient clipping** (`optimizer.rs` global/per-element helpers, plus
  the declarative `config::GradientClipConfig` wired into `Trainer` as of
  v0.1.2 — see "Completed (v0.1.2)" above)
  - `GaussianOptimizer::clip_grad_norm` — global L2 norm clipping, returns pre-clip norm
  - `GaussianOptimizer::clip_grad_value` — per-element value clamping
  - 3 tests (all passing)
- ✅ **EMA (Exponential Moving Average)** of parameters (`ema.rs`; wired
  into `Trainer` via `TrainingConfig::ema_decay` as of v0.1.2)
  - `GaussianEma` shadow copy for positions, rotations, scales, opacities, SH coefficients
  - Bias-corrected effective decay: `min(decay, (1+step)/(10+step))`
  - `update`, `apply_to`, `effective_decay`, `step` API
  - Handles density-control model size changes gracefully
  - 8 tests (all passing)

## 💡 Future Enhancements

- ⬜ **Multi-GPU training** (`data_parallel.rs` provides the sync-report /
  all-reduce primitives standalone; not wired into `Trainer`)
- ✅ **Mixed precision training** (`mixed_precision.rs` — approx_constant fixes applied)
- ✅ **Profiling integration** (`profiler_integration.rs`)
- ✅ **Gradient accumulation** (for larger batch sizes; wired into `Trainer`
  via `TrainingConfig::gradient_accumulation_steps` as of v0.1.2)
  - `accumulate_gradients()` and `step_accumulated()` on `GaussianOptimizer`
  - `accumulate_from()` and `scale()` helpers on `Gradients`
  - Handles density-control model size changes
  - 7 new tests

## 📊 Current Status

### Implementation: ~95% complete
- ✅ Training loop: 100% — as of v0.1.2, differentiates the objective it
  actually reports (photometric + regularization + SDS gradients, not the
  previous hardcoded L2 term)
- ✅ Optimizer: 100% (+ gradient clipping + gradient accumulation + LR
  schedules + EMA, all wired into `Trainer` as of v0.1.2 — previously
  standalone-only)
- ✅ Loss functions: 100% (`w_normal` / `w_gradient_penalty` remain
  logging-only by design; every other term has a real gradient)
- ✅ Metrics: 100%
- ✅ Density control: 100% (+ `GaussianPruner::prune_by_min_scale` /
  `prune_to_sparsity` / `apply_mask_to_model`, v0.1.2)
- ✅ Checkpointing: 100% (save ✅, resume ✅)
- ✅ Camera sampling strategy: 100% (Random, Spiral, Hemisphere, Turntable)
- ✅ EMA parameter tracking: 100% (bias-corrected, density-control aware,
  wired into `Trainer` as of v0.1.2)
- ✅ Meta-learning: `GaussianAvatarModel` (v0.1.2) is the first real
  `MetaModel` implementation over an actual avatar
- ⬜ End-to-end integration: pending (requires real model weights, or an
  integration test built on the new `into_gaussian_model` synthetic-data
  bridge — see "Planned" above)

### Tests: 3,271 tests (3,259 passing, 12 ignored GPU/slow) - **EXCELLENT**

### Documentation: Good

## 🎯 Priority

**High:**
1. ⬜ End-to-end integration test (needs real model weights, or the new
   `synthetic_data::into_gaussian_model` bridge wired into an actual test)
2. ✅ ~~Camera sampling strategy~~ (done)
3. ✅ ~~Gradient clipping~~ (done; wired into `Trainer` as of v0.1.2)
4. ✅ ~~Guidance scale annealing~~ (done)
5. ✅ ~~Gradient accumulation~~ (done; wired into `Trainer` as of v0.1.2)

## 🏆 Implementation Highlights

**EXCEEDS PLAN:**
1. **TensorBoard integration** (1,307 lines) - Not in original plan!
2. **LPIPS in Pure Rust** (805 lines) - No Python dependency
3. **3,271 tests** (3,259 passing, 12 ignored GPU/slow) - Exceptional coverage
4. **Video input support** — `VideoFrameIterator`, `VideoConfig`, `FrameBatch`, `TrainerError::SequenceError`, 15 tests
5. **Comprehensive checkpoint format**
6. **Guidance scale annealing** - `annealed_guidance_scale(step)`, linear decay (7 tests)
7. **Gradient accumulation** - `accumulate_gradients()` / `step_accumulated()`, density-control aware, wired into `Trainer::train_step` as of v0.1.2
8. **Meta-learning avatar model** (v0.1.2) - `GaussianAvatarModel`, the first real `MetaModel` over an actual Gaussian avatar, driving `loss_and_grad` through the real rasterizer forward/backward pass

**v0.1.2 fixed three training-loop correctness bugs**: the photometric loss
optimized a hardcoded L2 term regardless of `LossConfig`; position/scale/
opacity regularization never produced gradients; and score-distillation
(SDS) training had no gradient path at all. All three are fixed and covered
by tests — see "Completed (v0.1.2)" above. `w_normal` and
`w_gradient_penalty` remain logging-only by design (documented, not
silent).
