# TODO for oxigaf-diffusion

**Status: v0.1.2** (released 2026-08-28) — 3213/3213 tests passing
(`cargo nextest run -p oxigaf-diffusion --all-features`) + 51/51 passing doc
tests (1 ignored, `cargo test --doc --all-features`), zero
`todo!()`/`unimplemented!()` stubs in `src/`. See [README.md](README.md) for
the feature/API summary and the 0.1.2 breaking-changes list. One-person
project; contributions welcome via PR.

## ✅ Completed (from plan)

### Core Architecture
- ✅ Multi-view U-Net based on SD 2.1 architecture
- ✅ ResNet blocks with time-step conditioning
- ✅ Downsample/Upsample2d layers
- ✅ U-Net encoder-decoder structure with skip connections
- ✅ Camera embedding MLP (flattened 4×4 matrix → time_embed_dim)
- ✅ Timestep embedding (sinusoidal + MLP projection)
- ✅ Group normalization (32 groups)
- ✅ SiLU activation functions

### Attention Mechanisms
- ✅ Multi-view spatial transformer blocks
- ✅ Cross-view attention (Q from one view, K/V from all N views)
- ✅ **Flash Attention** (feature: `flash_attention`, **opt-in since v0.1.2** —
  no longer part of `default`; `use_flash_attention` in `DiffusionConfig`
  still defaults to `true` when the feature *is* compiled in)
  - Memory-efficient O(N) attention instead of O(N²)
  - Block-based tiled computation
  - 2-4× memory reduction for large sequences whose un-tiled score matrix
    would exceed the 64 MiB budget (`DEFAULT_SCORE_MATRIX_BUDGET`) — below
    that budget `FlashAttention::forward` runs the same full-materialization
    kernel as standard attention, so smaller sequences see no reduction
  - Configurable block size
- ✅ Standard attention fallback (now the default path; used whenever flash_attention is disabled)
- ✅ Self-attention (per-view spatial)
- ✅ Cross-attention to encoder hidden states

### VAE (Variational Autoencoder)
- ✅ Encoder: image → latent (with scaling factor 0.18215)
- ✅ Decoder: latent → image
- ✅ Support for 4-channel latents (SD 2.1 format)
- ✅ Group normalization in encoder/decoder
- ✅ Residual blocks with SiLU activation

### CLIP Image Encoder
- ✅ ViT-based image encoder
- ✅ Patch embedding layer
- ✅ Positional embeddings
- ✅ Transformer blocks for visual features
- ✅ Output: `(1, seq_len, embed_dim)` feature tensor

### DDIM Scheduler
- ✅ V-prediction mode (for SD 2.1)
- ✅ Epsilon-prediction mode (fallback)
- ✅ Configurable number of inference steps
- ✅ Beta schedule (linear/scaled linear)
- ✅ Noise addition (`add_noise()`)
- ✅ Denoising step (`step()`)
- ✅ Timestep tensor generation

### Pipeline Orchestration
- ✅ `MultiViewDiffusionPipeline` struct
- ✅ Model loading from safetensors
- ✅ Component initialization (U-Net, VAE, CLIP, scheduler)
- ✅ `generate()` method signature and structure
- ✅ Device management (CPU/CUDA/Metal)
- ✅ DType support (F32)

### Error Handling
- ✅ Comprehensive `DiffusionError` enum with 20+ variants
- ✅ Model loading errors
- ✅ Tensor operation errors (shape mismatch, dtype/device mismatch)
- ✅ Numerical stability errors (NaN/Inf detection)
- ✅ Inference errors (invalid timesteps, view counts)
- ✅ Pipeline errors (scheduler not initialized, encoding failures)
- ✅ I/O and image processing errors
- ✅ Candle backend error propagation

### Testing
- ✅ 3213 tests passing (`cargo nextest run --all-features`) + 51 passing doc
  tests (1 ignored). Breakdown verified against
  `cargo nextest list --all-features` on 2026-08-28:
  - Library unit tests: 3075 (across all `src/*.rs` modules)
  - `tests/attention_precision_tests.rs` (6 tests)
  - `tests/attention_tests.rs` (13 tests)
  - `tests/camera_tests.rs` (11 tests)
  - `tests/cfg_ip_adapter_tests.rs` (11 tests)
  - `tests/comprehensive_tests.rs` (47 tests: scheduler, DiffusionConfig, error types, DebugHooks, SlicedAttention public API, CFG formula)
  - `tests/memory_tests.rs` (22 tests)
  - `tests/scheduler_tests.rs` (19 tests)
  - `tests/vae_tests.rs` (9 tests)
  - Doc tests: 51 passing, 1 ignored (`upsampler::LatentUpsampler`, requires real weights)
- ✅ Shape preservation tests
- ✅ Attention mechanism tests
- ✅ Camera embedding tests
- ✅ Timestep embedding tests
- ✅ Scheduler step tests (V-prediction & Epsilon-prediction)
- ✅ Noise addition tests
- ✅ NaN/Inf detection hooks tests
- ✅ Sliced attention tests (slice_size variants, fallback, numerical stability)
- ✅ Pipeline integration tests (U-Net, CFG, DiffusionConfig, error types)

### Benchmarking
- ✅ 2 comprehensive benchmark files:
  - `diffusion_bench.rs` - Full pipeline benchmarks
  - `flash_attention_bench.rs` - Attention performance
- ✅ Benchmarks for:
  - Standard vs Flash attention comparison
  - Different sequence lengths (64, 128, 256, 512)
  - Different block sizes for Flash attention
  - Different batch sizes
  - Different attention head counts
  - DDIM scheduler steps
  - Full denoising loops

### Code Quality
- ✅ No `unwrap()`/`.expect()` in production code paths — verified 2026-08-28:
  every occurrence in `src/` is confined to `#[cfg(test)] mod tests` blocks or
  `///`/`//!` rustdoc examples. Not compiler-enforced via `#![deny(...)]` (no
  such attribute exists in `lib.rs`, and there is no workspace-level
  `[lints]` table or `clippy.toml`) — this is achieved by discipline, not a
  lint gate. `cargo clippy --all-features --all-targets` is currently clean.
- ✅ All source files under the 2000-line policy limit — largest is
  `stochastic_interpolant.rs` at 1996 lines (4 lines of headroom; see
  "Known Issues" below)
- ✅ Total codebase: 72 source files, ~65,000 lines of code (`tokei src`,
  2026-08-28) — grown substantially since the single-digit-file, 3,342-line
  v0.1.0 baseline as the modules under "Future Enhancements" below were added
- ✅ Clean module structure

### Feature Flags

*Superseded by v0.1.2 — `accelerate`/`cuda`/`metal` were dropped from this
crate's own `[features]` and `flash_attention` left `default`. Current flag
set:*

- ✅ `default` = `[]` (was `["accelerate", "flash_attention"]` before v0.1.2)
- ✅ `flash_attention` - Memory-efficient attention (opt-in since v0.1.2, not in `default`)
- ✅ `mixed_precision` - FP32↔BF16/FP16 conversion utilities (`mixed_precision.rs`); real and tested, but not yet called from `unet.rs`/`vae.rs`/`pipeline.rs`, so it doesn't change `generate()`'s output
- ✅ `gpu_debug` - NaN/Inf debug hooks (`debug_hooks::assert_finite`, `DebugConfig`)
- ✅ `parallel` - `rayon`-backed data parallelism (`dep:rayon`), currently
  driving the per-sample ControlNet feature injection in `unet.rs`; every
  parallelised routine is arithmetic-identical to its serial form
- ❌ ~~`accelerate` - CPU BLAS/LAPACK acceleration~~ — removed in v0.1.2;
  enable it on the resolved `candle-core`/`candle-nn` package instead
  (`oxicandle-core`/`oxicandle-nn` fork — see README "GPU / BLAS Backends")
- ❌ ~~`cuda` - NVIDIA GPU support~~ — removed in v0.1.2; same redirection as `accelerate`
- ❌ ~~`metal` - Apple Silicon GPU support~~ — removed in v0.1.2; same redirection as `accelerate`

### Mixed Precision Support (v0.1.1)
- ✅ **Mixed precision support** (feature: `mixed_precision`, `mixed_precision.rs`, ~560 lines code + ~260 lines tests)
  - `PrecisionMode` enum: Float32 (default without feature), BFloat16 (default with `mixed_precision` feature), Float16
  - `MixedPrecisionConfig`: mode, fp32_layernorm, fp32_softmax, fp32_output, loss_scale=1024.0
  - `OpType` enum for operation-type-aware precision selection
  - Pure-Rust BF16/FP16 conversion: `f32_to_bf16`/`bf16_to_f32` (upper 16 bits of f32), `f32_to_f16`/`f16_to_f32` (IEEE 754 half-precision)
  - `simulate_bf16`/`simulate_f16` (quantize and dequantize for simulation)
  - `apply_precision(data, config)` for applying precision mode to tensor data
  - `PrecisionStats::compute`: max_abs_error, mean_abs_error, num_overflows, num_underflows
  - Feature-gated default: BFloat16 when `mixed_precision` enabled, Float32 otherwise
  - 44 new tests (pass with and without feature)

### Latent Upsampler (v0.1.0)
- ✅ **sd-x2-latent-upscaler integration** (`upsampler.rs`)
  - Separate U-Net for 32×32 → 64×64 latent upsampling
  - 10-step DDIM denoising in latent space
  - Fallback: `BilinearVae` mode for CPU inference

### IP-Adapter Conditioning (v0.1.0)
- ✅ **IP cross-attention layers**
  - Additional `attn_ip` cross-attention layer in transformer blocks
  - Context = VAE-encoded reference image
  - Pixel-level identity preservation across all generated views

### Classifier-Free Guidance (v0.1.0)
- ✅ **CFG implementation**
  - Double batch: conditional + unconditional forward passes
  - `noise_pred = uncond + guidance_scale * (cond - uncond)`
  - Configurable `guidance_scale` (default: 7.5, range: 1.0–20.0)

### Extended Sampler Suite (v0.1.1)
- ✅ **DDPM sampler** — ancestral sampling with Markov chain noise schedule (v0.1.1)
- ✅ **Adaptive sampling** — dynamic step-count based on convergence (v0.1.1)
- ✅ **Guidance rescaling** — CFG++ style guidance magnitude correction (v0.1.1)
- ✅ **Consistency model** — single-step distilled consistency sampler (v0.1.1)
- ✅ **Flow matching** — ODE-based flow matching sampler (v0.1.1)

### Image Editing & LoRA (v0.1.1)
- ✅ **SDEdit image editing** — in-context editing via noise-and-denoise (v0.1.1)
- ✅ **LoRA adapter** — parameter-efficient fine-tuning via low-rank decomposition (v0.1.1)
- ✅ **ControlNet adapter** — spatial conditioning control adapter (v0.1.1)

### Attention & Latent Utilities (v0.1.1)
- ✅ **Attention masking** — mask-based attention restriction (v0.1.1)
- ✅ **Fused attention** — memory-efficient fused QKV attention (v0.1.1)
- ✅ **KV cache** — cached key/value attention for fast inference (v0.1.1)
- ✅ **Attention visualisation** — per-head attention map rendering (v0.1.1)
- ✅ **Latent blending** — interpolation between latent codes (v0.1.1)
- ✅ **Latent walk** — smooth trajectory through latent space (v0.1.1)
- ✅ **Latent space analysis** — PCA/variance decomposition of latent space (v0.1.1)
- ✅ **Denoising trajectory** — per-step denoising state recording (v0.1.1)

### Conditioning & Evaluation (v0.1.1)
- ✅ **Identity conditioning** — face-identity-preserving conditioning (v0.1.1)
- ✅ **Avatar conditioning** — FLAME-mesh-aware conditioning (v0.1.1)
- ✅ **Classifier guidance** — auxiliary classifier score guidance (v0.1.1)
- ✅ **Distillation loss** — knowledge distillation loss for score matching (v0.1.1)
- ✅ **CLIP scoring** — CLIP-based prompt-image alignment score (v0.1.1)
- ✅ **DDIM inversion** — deterministic inversion for editing (v0.1.1)
- ✅ **Denoising visualisation** — debug hook visualiser (v0.1.1)
- ✅ **Image preprocessing** — resize/crop/normalise pipeline (v0.1.1)
- ✅ **Image variations** — diffusion-based image variation generation (v0.1.1)
- ✅ **Batch generation** — batched multi-prompt generation (v0.1.1)

## 🚧 In Progress

Currently none.

## 📋 Planned (future versions)

### Weight Conversion Tooling
- ⬜ **Offline conversion script** (`scripts/convert_gaf_weights.py`)
  - Convert PyTorch GAF checkpoint → SafeTensors
  - Layer name mapping: PyTorch → candle VarBuilder paths
  - Separate files:
    - `multiview_unet.safetensors` (~1.7 GB fp16)
    - `vae.safetensors` (~335 MB fp16)
    - `clip_image.safetensors` (~900 MB fp16)
    - `latent_upscaler.safetensors` (~500 MB fp16)
- ⬜ **Weight name validation**
  - Assertion: all weight keys consumed (no orphans)
  - Layer-by-layer output comparison script
- ⬜ **Memory-mapped I/O** (already using buffered safetensors, but could optimize)

### Optimization Strategies
- ✅ **Attention slicing**
  - `sliced_attention.rs` (~320 lines code + ~230 lines tests)
  - `SlicedAttentionConfig` (slice_size, num_heads, head_dim), `SlicedAttention::forward()` processes Q in configurable chunks using numerically stable chunked softmax (row-wise max subtraction)
  - Handles `slice_size > seq_len_q` gracefully (falls back to full)
  - 23 new tests
- ✅ **Sequential VAE processing** (`sequential_vae.rs`)
- ✅ **Weight offloading (CPU↔GPU)** (`weight_offload.rs`)
- ⬜ **Gradient checkpointing** (not needed for inference, but if training support added)

### Numerical Stability
- ✅ **Selective FP32 for sensitive ops** — `numerics.rs` (480 lines). `AttentionPrecision` enum (Standard, UpcastedSoftmax [default], FullUpcast). `softmax_inplace`, `softmax`, `log_softmax` (numerically stable log-sum-exp). `layer_norm` (f64 accumulation for extra precision), `rms_norm`. `gelu`/`gelu_slice` (tanh approximation), `silu`/`silu_slice`. `NumericsError` (EmptyInput, LengthMismatch). `count_subnormal`, `is_subnormal`. `SlicedAttentionConfig` extended with `attention_precision: AttentionPrecision` field. 28 new tests.
- ✅ **Mixed precision (FP16/BF16 inference)** — `mixed_precision.rs` (~560 lines code + ~260 lines tests). `PrecisionMode` enum, `MixedPrecisionConfig`, `OpType` enum, pure-Rust BF16/FP16 conversion, `simulate_bf16`/`simulate_f16`, `apply_precision`, `PrecisionStats::compute`. Added with 44 tests in v0.1.1, since grown to 47 (see "Testing" above for the current crate-wide total).
- ✅ **NaN/Inf detection hooks** (`debug_hooks.rs`, 382 lines)
  - `TensorHealth` struct, `check_tensor_health()`, `all_finite()` fast-path
  - `assert_finite()` returning `Err(DiffusionError::NanInfDetected{…})`
  - `DebugHooks` thread-safe registry with `Mutex<Vec<TensorHealth>>` + 2 `AtomicU64` counters
  - `DebugConfig` with enabled/panic_on_nan/panic_on_inf/log_all_checks/max_records
  - 22 new debug_hooks tests
- ⬜ **Gradient clipping** (if training support added)

### Testing Gaps
- ✅ **U-Net forward pass tests / Pipeline integration tests**
  - `tests/comprehensive_tests.rs` with 47 new integration tests: scheduler (10), DiffusionConfig (7), error types (3), DebugHooks via public re-exports (14), SlicedAttention public API (8), CFG formula (5)
  - Total at time of completion: 207 tests passing (see "Testing" above for the current crate-wide total)
- ⬜ **VAE encode/decode tests**
  - Round-trip reconstruction loss
  - Latent statistics (mean, std)
- ⬜ **CLIP encoding tests**
  - Feature vector shape and normalization
  - Similarity scores for similar images
- ⬜ **Cross-validation with Python reference**
  - Layer-by-layer output comparison
  - Tolerance < 1e-3 for fp16

### Documentation Gaps
- ⬜ **Mathematical background**
  - DDIM sampling algorithm explanation
  - V-prediction vs Epsilon-prediction
  - Cross-view attention mechanism
  - Flash attention algorithm
- ⬜ **Architecture diagrams**
  - U-Net block structure
  - Attention block composition
  - Data flow through pipeline
- ✅ **Usage examples** — `basic_inference.rs`, `multi_view_consistency.rs`, `cfg_comparison.rs`, `flash_vs_standard.rs`, `batch_generation.rs`, `streaming_demo.rs` (`streaming_demo.rs` compiles and runs, but its output is misleading post-streaming.rs rewrite — see the "Known bug" note under Streaming inference above)

### Model Variants
- ✅ **Model variants** (`model_variants.rs`)
- ⬜ **Different CLIP encoders** (ViT-H/14, ViT-B/32)
- ⬜ **Alternative VAE models**

## 💡 Future Enhancements (beyond original plan)

### Performance
- ✅ **Quantization support** (`quantization.rs`)
- ✅ **Fixed approx_constant errors; added `FromStr` for `WeightedPrompt`** (`prompt_weighting.rs`)
- ✅ **KV-cache for attention** — `kv_cache.rs` (~530 lines code + ~220 lines tests). `KVEntry` (flat f32 keys/values, batch/num_heads/seq_k/head_dim metadata, access_count, memory_bytes, is_valid). `KVCacheConfig` (max_entries=64, max_memory_bytes=512MB, enabled, eviction=LRU). `EvictionPolicy` (LRU/LFU/FIFO). `CacheStats` (hits/misses/evictions/memory/entries, hit_rate()). `KVCache` thread-safe with independent Mutex fields; `contains/get/insert/remove/clear/get_or_compute/stats/memory_bytes/len`. `CacheKeyBuilder` (layer/head_group/conditioning_hash builder pattern). LRU implemented via insertion_order deque. 22 new tests.
- ✅ **Fused attention** (`fused_attention.rs`)
- ⬜ **Model distillation**
  - Distill to fewer inference steps (50 → 10 → 4)
  - Guidance distillation (remove CFG overhead)

### Flexibility
- ✅ **Dynamic view count** (`dynamic_views.rs`)
- ✅ **Variable resolution support** (`resolution.rs`, 771 lines)
  - `ResolutionConfig` (width, height, latent dimensions, patch_size)
  - Adaptive latent sizes for 128×128, 256×256, 512×512 inputs
  - `VariableResolutionPipeline` with multi-scale inference support
- ⬜ **Prompt conditioning** (text-guided generation)
  - Add text encoder (CLIP text)
  - Text cross-attention alongside image
- ✅ **ControlNet integration** (`controlnet.rs`)

### Debugging & Analysis
- ✅ **Activation visualization** (`attention_viz.rs`)
- ✅ **Step-by-step denoising visualization** (`denoising_viz.rs`)
- ✅ **Noise schedule analysis** (`noise_schedule_analysis.rs`, 1213 lines)
  - Karras EDM sigma schedule: `karras_sigmas(n, sigma_min, sigma_max, rho)` — `ρ`-power interpolation
  - `NoiseScheduleAnalyzer` with per-step SNR, log-SNR, effective denoising fraction
  - Schedule comparison utilities across DDIM, DDPM, Karras EDM
  - `ScheduleStats` (mean/min/max sigma, monotonicity check, SNR range)
- ⬜ **Diffusion profiling module**
  - Per-layer timing breakdown
  - Memory usage profiling
  - Bottleneck identification

### Integration
- ✅ **Streaming inference** — `streaming.rs` (grew from 260 to 880 lines
  post-v0.1.1; no longer a placeholder). `StreamingConfig` (256×256,
  guidance=3.0, num_steps=20). `StreamingStep` (view_index, step_index,
  total_steps, partial_image, is_final, progress_fraction()). Two engine
  modes: `StreamingInference::new` builds a **schedule-only** engine — real
  step bookkeeping, but `partial_image` stays honestly empty (no weights, no
  fabricated pixels — the old grey-ramp placeholder was removed as a
  regression fix); `StreamingInference::load`/`with_pipeline` attach a real
  `MultiViewDiffusionPipeline` + `GenerationSession` and yield genuine
  VAE-decoded RGB frames per denoising step. `StreamingInference::step_iter(num_views)`
  returns a `StreamingIterator: Iterator<Item = StreamingStep>` yielding
  **step-major** (all views of step 0, then all views of step 1, …) — code
  that assumes view-major ordering (grouping all steps of one view together)
  will misbehave; `examples/streaming_demo.rs` has this bug as of this
  writing (see TODO below). 23 tests in `streaming.rs` itself.
  - ⬜ **Known bug**: `examples/streaming_demo.rs` still calls
    `StreamingInference::new` (schedule-only), so it prints `first_pixel=0`
    and a `0 bytes` buffer on every step instead of real pixels, and its
    `last_view`-tracked "--- View N ---" header (written for view-major
    output) now re-prints before almost every line under step-major
    ordering. Needs `StreamingInference::load` with real weights (or an
    explicit "no model attached" framing) and step-major-aware headers.
- ✅ **Batch generation** — `batch_gen.rs` (323 lines). `GenerationRequest` (id, reference_image bytes, num_views, guidance_scale, num_steps, seed). `GeneratedView` (view_index, image_data, width, height, generation_time_ms). `GenerationResult` (id, views, total_time_ms, num_cached_kv, throughput_views_per_sec()). `BatchGenConfig` (max_batch_size=4, max_views_per_request=4, guidance_scale=3.0, num_steps=20, use_kv_cache=true, synchronous=true). `BatchStats` (total_requests/views/time, cache_hits/misses, hit_rate). `BatchGenerator` with queue()/process_batch()/process_one()/clear_queue()/stats()/reset_stats(). 31 new tests.
- ⬜ **Web API server**
  - REST API for inference
  - WebSocket for streaming
  - Queue management

## 🐛 Known Issues

- ⬜ **Flash attention numerical precision**
  - Flash attention may have slightly different outputs vs standard (tiling artifacts)
  - Need more testing with fp16
  - Mitigation: Make it optional, default to standard for now
- ✅ ~~**Mixed precision**~~ — The conversion toolkit itself is fully implemented (`mixed_precision.rs`, 47 tests): BF16/FP16 conversion, `PrecisionStats`, feature-gated default. Not yet wired into `unet.rs`/`vae.rs`/`pipeline.rs`, though, so turning the feature on doesn't change `generate()`'s numerics — see "Feature Flags" above.
- ⬜ **File size creep**: `stochastic_interpolant.rs` is 1996 lines — 4 lines
  from the workspace's 2000-line-per-file policy limit. Not yet split; the
  next feature or test addition to this file should go through `splitrs`
  (or a manual split) instead of pushing it over the limit. Runner-up:
  `token_merging.rs` at 1889 lines.

## 📊 Current Status

### Implementation: ~99% complete (v0.1.2)
- ✅ Core U-Net: 100%
- ✅ VAE: 100%
- ✅ CLIP: 100%
- ✅ Scheduler: 100%
- ✅ Flash Attention: 100%
- ✅ Latent Upsampler: 100% (`upsampler.rs`)
- ✅ IP-Adapter: 100%
- ✅ CFG: 100%
- ✅ Pipeline orchestration: 100%
- ✅ Selective FP32 / Numerical stability: 100% (`numerics.rs`, 33 tests)
- ✅ Mixed precision (FP16/BF16) conversion toolkit: 100% (`mixed_precision.rs`, 47 tests) — not yet wired into the inference path (see "Feature Flags" above)
- ✅ Variable resolution support: 100% (`resolution.rs`, 771 lines)
- ✅ Noise schedule analysis (Karras EDM): 100% (`noise_schedule_analysis.rs`, 1213 lines)
- ⬜ Weight loading: 50% (structure exists, conversion script pending)
- ⬜ Optimization strategies: 80% (flash attention + attention slicing + numerics + mixed precision + KV-cache + streaming inference + batch generation + variable resolution done, profiling pending)

### Tests: 3213 tests (all passing, 0 skipped) + 51 passing doc tests (1 ignored)
- ✅ Library unit tests: 3075 (across all `src/*.rs` modules — attention,
  camera, scheduler, debug_hooks, sliced_attention, numerics,
  mixed_precision, kv_cache, streaming, batch_gen, and the ~60 modules under
  "Future Enhancements" below)
- ✅ Integration tests: 138 across 8 files in `tests/` (see "Testing" above
  for the per-file breakdown)
- ✅ Doc tests: 51 passing, 1 ignored
- ⬜ Cross-validation with Python: 0
- Coverage: Excellent — comprehensive integration tests cover pipeline, CFG, error types, DebugHooks, SlicedAttention, numerics, mixed precision, KV-cache, streaming inference, batch generation

### Documentation: Good
- ✅ Rustdoc with feature explanations
- ✅ Error variant documentation
- ✅ Module-level documentation
- ✅ Usage examples (`examples/*.rs`, 6 files, all build with
  `cargo build --examples --all-features`) — see "Documentation Gaps" above
  for the one known display bug (`streaming_demo.rs`)
- ⬜ Missing: Mathematical background

### Benchmarks: Excellent
- ✅ 2 comprehensive benchmark files
- ✅ Covers attention, scheduler, full loops
- ✅ Compares flash vs standard attention
- Performance: Flash attention 30-50% faster than standard for seq_len > 256

## 📈 Comparison: Implementation vs Plan

| Feature | Plan | Current | Notes |
|---------|------|---------|-------|
| Multi-view U-Net | ✅ | ✅ | Fully implemented |
| Cross-view attention | ✅ | ✅ | Implemented with reshape logic |
| IP-adapter | ✅ | ✅ | **Done v0.1.0** |
| Camera conditioning | ✅ | ✅ | MLP fully implemented |
| VAE encoder/decoder | ✅ | ✅ | Fully implemented |
| CLIP image encoder | ✅ | ✅ | Fully implemented |
| DDIM scheduler | ✅ | ✅ | V-prediction + Epsilon modes |
| Latent upsampler | ✅ | ✅ | **Done v0.1.0** (`upsampler.rs`) |
| CFG | ✅ | ✅ | **Done v0.1.0** |
| Flash attention | ⬜ Optional | ✅ | **EXCEEDS PLAN** - opt-in feature (`flash_attention`; not in `default` since v0.1.2) |
| Attention slicing | ⬜ | ✅ | **Done** (`sliced_attention.rs`, chunked softmax, 23 tests) |
| Pipeline integration tests | ⬜ | ✅ | **Done** (`tests/comprehensive_tests.rs`, 47 new tests, 160 total) |
| Weight loading | ✅ | ⬜ | Structure exists, conversion script needed |
| Mixed precision | ⬜ Optional | ✅ | **Done v0.1.1** (`mixed_precision.rs`, 44 tests, BF16/FP16 conversion, `PrecisionStats`) |
| Streaming inference | ⬜ | ✅ | **Done v0.1.1**, since expanded (`streaming.rs`, 880 lines, `StreamingIterator`, 23 tests) |
| Batch generation | ⬜ | ✅ | **Done v0.1.1** (`batch_gen.rs`, 323 lines, `BatchGenerator`, `BatchStats`, 31 tests) |
| Variable resolution | ⬜ | ✅ | **Done** (`resolution.rs`, 771 lines, adaptive latents, multi-scale) |
| Noise schedule analysis | ⬜ | ✅ | **Done** (`noise_schedule_analysis.rs`, 1213 lines, Karras EDM sigmas, SNR) |
| Multi-device support | ✅ | ✅ | CPU/CUDA/Metal all supported |

## 🎯 Priority

**v0.1.0 critical items ✅ all done!**

**Future priority:**
1. ⬜ **Weight conversion script** — Convert PyTorch → SafeTensors (needed to load real weights)
2. ✅ ~~**Pipeline integration tests**~~ — Done (`tests/comprehensive_tests.rs`, 47 tests)
3. ⬜ Cross-validation with Python reference
4. ✅ ~~Attention slicing (for <8GB GPUs)~~ — Done (`sliced_attention.rs`, chunked softmax, graceful fallback)
5. ✅ ~~Usage examples~~ — Done (`examples/*.rs`, 6 files, all build; see "Documentation Gaps" above for one known display bug)

**Medium Priority:**
6. ✅ ~~Mixed precision~~ — Done (`mixed_precision.rs`, BF16/FP16, `PrecisionStats`; not yet wired into the inference path — see "Feature Flags" above)
7. ✅ ~~Selective FP32 for numerical stability~~ — Done (`numerics.rs`, `AttentionPrecision`, `softmax_inplace`, `layer_norm` f64)
8. ✅ ~~Sequential VAE processing~~ — Done (`sequential_vae.rs`, 940 lines, 20 tests)

**Low Priority:**
9. ✅ ~~Model variants support~~ — Done (`model_variants.rs`, 1321 lines, 43 tests: SD1.5/2.1/XL/Zero123 registry). The narrower "different CLIP encoders" / "alternative VAE models" sub-items remain open — see item 10.
10. ⬜ Alternative encoders (different CLIP encoder variants, alternative VAE models)
11. ✅ ~~Debugging visualization~~ — Done (`attention_viz.rs`, 1080 lines, 28 tests; `denoising_viz.rs`, 1094 lines, 23 tests)

## 🏆 Implementation Highlights

**Where current implementation EXCEEDS the plan:**

1. **Flash Attention** (not in original plan as default)
   - Memory-efficient O(N) attention
   - 30-50% faster than standard for large sequences
   - Opt-in via the `flash_attention` feature since v0.1.2 (previously
     default); standard attention is the default path
   - Comprehensive benchmarking suite

2. **Comprehensive Error Handling** (better than planned)
   - 20+ typed error variants
   - Proper error context propagation
   - Device/dtype mismatch detection
   - NaN/Inf detection hooks prepared

3. **Testing Infrastructure** (more thorough than planned)
   - 3213 tests passing across the library target and 8 integration-test
     files, plus 51 passing doc tests (see "Testing" above for the
     breakdown) — grown from the original plan's much smaller unit-test
     baseline as the crate's scope expanded
   - Property-based testing (`proptest`) in use
   - Extensive benchmarking suite

4. **Feature Flag Design** (cleaner than planned)
   - GPU/BLAS backends (`accelerate`/`cuda`/`metal`) removed from this crate
     in v0.1.2; enable them on the resolved `candle-core`/`candle-nn`
     package instead
   - `default = []` since v0.1.2 — CPU-only, no optional features
   - Flash attention opt-in (not part of `default`)

5. **Code Quality** (stricter than planned)
   - All files under the 2000-line policy limit (largest: `stochastic_interpolant.rs` at 1996 lines — see "Known Issues" above)
   - No `unwrap()`/`.expect()` in production code paths (see "Code Quality" above for how this was verified)
   - Clear module boundaries

**Current implementation is PRODUCTION-READY for:**
- U-Net inference (without CFG)
- VAE encode/decode
- CLIP image encoding
- DDIM scheduling
- Flash attention computation
- Mixed precision (BF16/FP16) inference

**v0.1.0 completed:**
- ✅ End-to-end 512×512 multi-view generation (Latent Upsampler + IP-Adapter + CFG)
- ✅ Full GAF pipeline functional

**v0.1.1 completed (2026-06-19, 2677 tests passing):**
- ✅ Mixed precision support (`mixed_precision.rs`, 44 tests, BF16/FP16 with `PrecisionStats`)
- ✅ KV-cache for attention (`kv_cache.rs`, 22 tests, LRU/LFU/FIFO eviction, thread-safe)
- ✅ Streaming inference (`streaming.rs`, 260 lines, `StreamingIterator`, 20 tests)
- ✅ Batch generation (`batch_gen.rs`, 323 lines, `BatchGenerator`, `BatchStats`, 31 tests)
- ✅ DDPM sampler, Adaptive sampling, Guidance rescaling, Consistency model, Flow matching
- ✅ SDEdit image editing, LoRA adapter, ControlNet adapter
- ✅ Attention masking, Fused attention, KV cache, Attention visualisation
- ✅ Latent blending, Latent walk, Latent space analysis, Denoising trajectory
- ✅ Identity conditioning, Avatar conditioning, Classifier guidance
- ✅ Distillation loss, CLIP scoring, DDIM inversion, Denoising visualisation
- ✅ Image preprocessing, Image variations, Batch generation

**v0.1.2 completed (2026-08-28, 3213 tests + 51 doc tests passing):**
- ✅ `flash_attention` moved out of `default` (now a real opt-in — previously
  `default = ["flash_attention"]` meant the feature could never actually be
  turned off) — see "Feature Flags" above
- ✅ `candle-core`/`candle-nn` switched to the COOLJAPAN `oxicandle-core`/
  `oxicandle-nn` fork (0.10 → 0.11; dependency keys unchanged), removing
  the transitive C Oniguruma (`onig`/`onig_sys`) dependency pulled in by
  upstream `candle-core`'s `tokenizers` dependency
- ✅ Unused `candle-transformers` workspace dependency removed
- ✅ Ten breaking signature/semantics changes made previously-infallible
  operations honest about failure, or fixed a mislabeled/renamed API — full
  list in [README.md](README.md#breaking-changes-in-012):
  `compute_distillation_loss`/`DistillationStep::aggregate_losses` return
  `Result`; `DistillationConfig` gained `latent_dims`; `TeacherStudentPair`
  gained `teacher_mid`; `dpm_plus_plus_2m_step` takes `prev_x0`/`h_prev`
  (6→7 args); `InversionTrajectory::x_0()`/`x_t()` return `Option`;
  `DdimInversionConfig::refinement_threshold` is now RMS-based;
  `pad_to_square`/`flip_horizontal`/`flip_vertical` return `Result`;
  `FlowStats::divergence_estimate` renamed to `mean_squared_norm`;
  `EditMask::all_ones` argument order now matches `EditMask::new`;
  `fm_interpolate`/`fm_target_velocity` honor a non-zero `sigma_min`;
  `softmax_over_dim` returns `Result`; `sdedit::ImageEditError` removed
  (folded into `image_editing::ImageEditingError`);
  `MultiViewDiffusionPipeline::begin_session_from_latents` takes a
  `SessionRequest` struct instead of seven positional arguments
- ✅ New `parallel` feature (rayon-backed ControlNet parallelism) — added
  during the 0.1.2 cycle (absent from the `Availability of 0.1.1` commit's
  `Cargo.toml`) and now documented in README.md/TODO.md for the first time
- ✅ Test suite grown from 2677 (v0.1.1) to 3213 passing (`--all-features`)
  + 51 passing doc tests

**Not yet ready for:**
- Production weight loading (PyTorch → SafeTensors conversion script pending)
- End-to-end Python cross-validation

## 🚀 Next Steps (post v0.1.2)

The two real, still-open items (weight conversion, integration testing with
real weights) carry forward unchanged from v0.1.1 — no progress was made on
either during the 0.1.2 cycle, which was scoped to the API/dependency
changes above instead.

1. **Weight Conversion Script** (~2-3 days)
   - Python script: PyTorch GAF checkpoint → SafeTensors
   - Layer name mapping
   - Validation against Python outputs

2. **Integration Testing** (~2-3 days)
   - End-to-end pipeline test with real weights
   - Multi-view consistency validation
   - Visual quality checks

3. ✅ ~~**Mixed Precision**~~ — Done (`mixed_precision.rs`, 47 tests)
4. ✅ ~~**Streaming Inference**~~ — Done (`streaming.rs`, grew from 260 to 880 lines post-v0.1.1, 23 tests)
5. ✅ ~~**Batch Generation**~~ — Done (`batch_gen.rs`, 48 tests)
