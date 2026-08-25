//! Pure helpers backing [`MultiViewDiffusionPipeline`][super::MultiViewDiffusionPipeline].
//!
//! Split out of `pipeline.rs` (which had grown past the 2000-line limit) so the
//! orchestration type and the free functions it is built from can be read
//! separately. Everything here is deliberately free of pipeline state: seeded
//! noise generation, the CFG combine, VAE chunking, weight-offload arithmetic,
//! KV-cache key derivation, profiling shims and schedule/shape validation.
//! That makes each piece testable without model weights, which is why the
//! regression tests for them live here alongside the code.

use candle_core::{DType, Device, Tensor};

use crate::config::DiffusionConfig;
use crate::profiling::{estimate_unet_memory_bytes, DiffusionProfiler};
use crate::upsampler::UpsamplerMode;
use crate::vae::Vae;
use crate::weight_offload::{ComponentType, OffloadStrategy};
use crate::DiffusionError;

use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// Deterministic noise sampling
// ---------------------------------------------------------------------------

/// Advance a 64-bit xorshift PRNG and return the new state.
///
/// The zero state is a fixed point of xorshift, so it is patched to `1`.
#[inline]
pub(super) fn xorshift64(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    if *state == 0 {
        *state = 1;
    }
    *state
}

/// Uniform `f32` in `[0, 1)` using 53 mantissa bits.
#[inline]
pub(super) fn xorshift_f32(state: &mut u64) -> f32 {
    (xorshift64(state) >> 11) as f32 / (1u64 << 53) as f32
}

/// Box-Muller transform: maps two uniform samples to a pair of standard normals.
#[inline]
pub(super) fn box_muller(u1: f32, u2: f32) -> (f32, f32) {
    let r = (-2.0_f32 * u1.max(1e-10).ln()).sqrt();
    let theta = 2.0 * PI * u2;
    (r * theta.cos(), r * theta.sin())
}

/// Draw `count` standard-normal samples from a stream seeded by `seed`.
///
/// Fully deterministic and device independent: the same `seed` always yields
/// the same values, on CPU as well as on GPU backends (candle's own
/// `Tensor::randn` uses a process-global RNG that cannot be seeded on CPU).
pub(crate) fn seeded_normal_values(count: usize, seed: u64) -> Vec<f32> {
    // xorshift64 requires a non-zero state; 0 is a legitimate user seed.
    let mut state = if seed == 0 {
        0x9E37_79B9_7F4A_7C15
    } else {
        seed
    };
    let mut out = Vec::with_capacity(count);
    while out.len() < count {
        let u1 = xorshift_f32(&mut state);
        let u2 = xorshift_f32(&mut state);
        let (z0, z1) = box_muller(u1, u2);
        out.push(z0);
        // Box-Muller produces samples in pairs; drop the second one when the
        // requested count is odd.
        if out.len() < count {
            out.push(z1);
        }
    }
    out
}

/// Build a `(v, c, h, w)` standard-normal tensor from the seeded stream.
pub(crate) fn seeded_normal_tensor(
    shape: (usize, usize, usize, usize),
    seed: u64,
    device: &Device,
) -> Result<Tensor, DiffusionError> {
    let (v, c, h, w) = shape;
    let values = seeded_normal_values(v * c * h * w, seed);
    Tensor::from_vec(values, shape, device)
        .map_err(|e| DiffusionError::Inference(format!("seeded noise init: {e}")))
}

// ---------------------------------------------------------------------------
// Classifier-free guidance helpers
// ---------------------------------------------------------------------------

/// Whether the unconditional U-Net pass is required for `guidance_scale`.
///
/// At `guidance_scale == 1.0` the CFG formula collapses to
/// `uncond + 1.0 * (cond - uncond) == cond`, so the unconditional pass is pure
/// waste — exactly half of the denoising compute.
#[inline]
pub(super) fn needs_uncond_pass(guidance_scale: f64) -> bool {
    (guidance_scale - 1.0).abs() > f64::EPSILON
}

/// Apply `pred = uncond + guidance_scale * (cond - uncond)`.
pub(super) fn combine_cfg(
    cond: &Tensor,
    uncond: &Tensor,
    guidance_scale: f64,
) -> Result<Tensor, DiffusionError> {
    let diff = (cond - uncond).map_err(|e| DiffusionError::Inference(format!("CFG diff: {e}")))?;
    (uncond + (diff * guidance_scale))
        .map_err(|e| DiffusionError::Inference(format!("CFG combine: {e}")))
}

// ---------------------------------------------------------------------------
// Chunking / offload helpers
// ---------------------------------------------------------------------------

/// Split `num_views` into `(start, len)` ranges of at most `chunk_size` views.
///
/// A `chunk_size` of `0` is treated as `1`.
pub(super) fn chunk_ranges(num_views: usize, chunk_size: usize) -> Vec<(usize, usize)> {
    let chunk = chunk_size.max(1);
    let mut ranges = Vec::with_capacity(num_views.div_ceil(chunk));
    let mut start = 0;
    while start < num_views {
        let len = chunk.min(num_views - start);
        ranges.push((start, len));
        start += len;
    }
    ranges
}

/// Estimated resident weight size of a pipeline-owned component.
///
/// The pipeline holds a single [`Vae`] that owns both halves of the
/// autoencoder, so the decoder entry is charged for encoder + decoder.
pub(super) fn component_size_mb(component: ComponentType) -> f32 {
    match component {
        ComponentType::VaeDecoder | ComponentType::VaeEncoder => {
            ComponentType::VaeDecoder.estimated_size_mb()
                + ComponentType::VaeEncoder.estimated_size_mb()
        }
        other => other.estimated_size_mb(),
    }
}

/// Weight memory (MB) charged for `component` given the configured upsampler.
///
/// Identical to [`component_size_mb`] except that a `BilinearVae` upsampler is
/// pure interpolation and holds no weights at all, so it costs nothing.
pub(super) fn component_weight_mb(
    component: ComponentType,
    upsampler_mode: Option<UpsamplerMode>,
) -> f32 {
    if component == ComponentType::LatentUpsampler
        && upsampler_mode == Some(UpsamplerMode::BilinearVae)
    {
        return 0.0;
    }
    component_size_mb(component)
}

/// Whether `component` should be dropped once its inference phase is over.
pub(super) fn component_should_release(
    strategy: OffloadStrategy,
    component: ComponentType,
) -> bool {
    match strategy {
        // Everything stays resident.
        OffloadStrategy::AllInMemory => false,
        // Nothing stays resident.
        OffloadStrategy::Sequential => true,
        // The U-Net is re-used every denoising step, so it stays cached.
        OffloadStrategy::CacheOne => component != ComponentType::MultiViewUNet,
    }
}

/// Decode `latents` through `vae` in chunks of at most `chunk_size` views.
///
/// Every operation in [`Vae::decode`] is per-sample — convolutions, group
/// normalisation (groups are taken over channels *within* a sample) and the
/// mid-block attention, which reshapes to `(batch, 3, channels, h*w)` and never
/// mixes across the batch dimension. Decoding view-chunks separately and
/// concatenating is therefore numerically identical to one batched decode,
/// while peak activation memory scales with `chunk_size` instead of the total
/// view count.
pub(super) fn decode_chunked(
    vae: &Vae,
    latents: &Tensor,
    chunk_size: usize,
) -> Result<Tensor, DiffusionError> {
    let num_views = latents
        .dim(0)
        .map_err(|e| DiffusionError::Inference(format!("latents dim0: {e}")))?;
    let ranges = chunk_ranges(num_views, chunk_size);
    if ranges.len() <= 1 {
        return vae
            .decode(latents)
            .map_err(|e| DiffusionError::Inference(format!("VAE decode: {e}")));
    }

    let mut decoded_chunks: Vec<Tensor> = Vec::with_capacity(ranges.len());
    for (start, len) in ranges {
        let chunk = latents
            .narrow(0, start, len)
            .map_err(|e| DiffusionError::Inference(format!("latent chunk at {start}: {e}")))?;
        let decoded = vae
            .decode(&chunk)
            .map_err(|e| DiffusionError::Inference(format!("VAE decode chunk at {start}: {e}")))?;
        decoded_chunks.push(decoded);
    }

    Tensor::cat(&decoded_chunks, 0)
        .map_err(|e| DiffusionError::Inference(format!("sequential decode cat: {e}")))
}

/// Encode `images` through `vae` in chunks of at most `chunk_size` views.
///
/// The mirror of [`decode_chunked`]: [`Vae::encode`] is per-sample for exactly
/// the same reasons (convolutions, channel-wise group norm, batch-preserving
/// mid-block attention), so encoding view-chunks separately and concatenating
/// is numerically identical to one batched encode at a fraction of the peak
/// activation memory.
pub(super) fn encode_chunked(
    vae: &Vae,
    images: &Tensor,
    chunk_size: usize,
) -> Result<Tensor, DiffusionError> {
    let num_views = images
        .dim(0)
        .map_err(|e| DiffusionError::Inference(format!("images dim0: {e}")))?;
    let ranges = chunk_ranges(num_views, chunk_size);
    if ranges.len() <= 1 {
        return vae
            .encode(images)
            .map_err(|e| DiffusionError::VaeEncodeFailed(format!("{e}")));
    }

    let mut encoded_chunks: Vec<Tensor> = Vec::with_capacity(ranges.len());
    for (start, len) in ranges {
        let chunk = images
            .narrow(0, start, len)
            .map_err(|e| DiffusionError::Inference(format!("image chunk at {start}: {e}")))?;
        let encoded = vae
            .encode(&chunk)
            .map_err(|e| DiffusionError::VaeEncodeFailed(format!("chunk at {start}: {e}")))?;
        encoded_chunks.push(encoded);
    }

    Tensor::cat(&encoded_chunks, 0)
        .map_err(|e| DiffusionError::Inference(format!("sequential encode cat: {e}")))
}

// ---------------------------------------------------------------------------
// Cross-attention KV cache keys
// ---------------------------------------------------------------------------

/// FNV-1a 64-bit offset basis.
pub(super) const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
/// FNV-1a 64-bit prime.
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// FNV-1a over the raw bytes of `values`.
///
/// Hashing the `f32` *bit patterns* rather than the values keeps the hash a
/// pure function of the tensor contents, `NaN` payloads included.
pub(super) fn fnv1a_f32(values: &[f32], seed: u64) -> u64 {
    let mut hash = seed;
    for value in values {
        for byte in value.to_bits().to_le_bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
    }
    hash
}

/// Identity of one run's IP-Adapter conditioning, for KV-cache keys.
///
/// Mixes the token contents with `num_views` because the cached K/V tensors
/// carry the per-view expanded batch dimension: the same reference image at a
/// different view count produces differently shaped entries.
///
/// # Errors
///
/// `DiffusionError::Inference` when the tokens cannot be read back.
pub(super) fn conditioning_tag(
    ip_tokens: &Tensor,
    num_views: usize,
) -> Result<u64, DiffusionError> {
    let values = ip_tokens
        .flatten_all()
        .and_then(|flat| flat.to_dtype(DType::F32))
        .and_then(|flat| flat.to_vec1::<f32>())
        .map_err(|e| DiffusionError::Inference(format!("IP token hash readback: {e}")))?;
    let seed = fnv1a_f32(&[], FNV_OFFSET_BASIS ^ num_views as u64);
    Ok(fnv1a_f32(&values, seed))
}

// ---------------------------------------------------------------------------
// Profiling helpers
// ---------------------------------------------------------------------------

/// Begin timing `name`, or do nothing when profiling is disabled.
///
/// Free functions rather than methods so the enabled/disabled behaviour is
/// testable without a loaded pipeline.
pub(super) fn profile_start(
    profiler: Option<&mut DiffusionProfiler>,
    name: &str,
    estimated_memory_bytes: usize,
) {
    if let Some(profiler) = profiler {
        profiler.start_with_memory(name.to_string(), estimated_memory_bytes);
    }
}

/// Close the timing opened by [`profile_start`], or do nothing when profiling
/// is disabled.
///
/// A phase that fails part-way simply leaves its entry open; the next
/// successful run of the same phase replaces it.
pub(super) fn profile_stop(profiler: Option<&mut DiffusionProfiler>, name: &str) {
    if let Some(profiler) = profiler {
        let _ = profiler.stop(name);
    }
}

/// Estimated activation memory (bytes) of one U-Net evaluation.
///
/// Feeds [`DiffusionProfiler::start_with_memory`] so a recorded `unet_forward`
/// sample carries both a duration and a memory figure. `spatial` is the latent
/// height of the model input.
pub(super) fn unet_activation_estimate(
    config: &DiffusionConfig,
    num_views: usize,
    spatial: usize,
) -> usize {
    estimate_unet_memory_bytes(
        num_views,
        config.base_channels,
        spatial,
        config.layers_per_block * config.num_stages().max(1),
    )
}

/// Drop the first `start_step` entries of a descending DDIM schedule.
///
/// This is the img2img "strength" control: with `start_step = 0` the run
/// denoises from the noisiest timestep, and each increment starts one step
/// further down the schedule, preserving more of the caller's latents.
///
/// # Errors
///
/// [`DiffusionError::InvalidConfig`] when `start_step` is not strictly below
/// `timesteps.len()` — a session with an empty schedule would report itself
/// finished before applying a single step, silently returning the caller's own
/// (still noisy) latents.
pub(super) fn timesteps_from_start_step(
    timesteps: &[usize],
    start_step: usize,
) -> Result<Vec<usize>, DiffusionError> {
    if start_step >= timesteps.len() {
        return Err(DiffusionError::InvalidConfig(format!(
            "start_step ({start_step}) must be < the schedule length ({})",
            timesteps.len()
        )));
    }
    Ok(timesteps[start_step..].to_vec())
}

/// Validate caller-supplied session latents against the pipeline geometry.
///
/// Pure shape arithmetic, split out of
/// [`MultiViewDiffusionPipeline::begin_session_from_latents`] so it can be
/// exercised without model weights.
///
/// Returns the accepted `(views, channels, h, w)` on success.
///
/// # Errors
///
/// [`DiffusionError::InvalidLatentShape`] when `latents` is not 4-D, carries
/// the wrong view or channel count, or disagrees spatially with
/// `normal_dims` — [`MultiViewDiffusionPipeline::step_session`] concatenates
/// the two along the channel axis, so a spatial mismatch would surface as an
/// opaque tensor error mid-denoise instead.
pub(super) fn validate_session_latents(
    latent_dims: &[usize],
    normal_dims: &[usize],
    num_views: usize,
    latent_channels: usize,
    config_latent_size: usize,
) -> Result<(usize, usize, usize, usize), DiffusionError> {
    let [lat_v, lat_c, lat_h, lat_w] = *latent_dims else {
        return Err(DiffusionError::InvalidLatentShape {
            expected: vec![
                num_views,
                latent_channels,
                config_latent_size,
                config_latent_size,
            ],
            got: latent_dims.to_vec(),
        });
    };
    if lat_v != num_views || lat_c != latent_channels {
        return Err(DiffusionError::InvalidLatentShape {
            expected: vec![num_views, latent_channels, lat_h, lat_w],
            got: vec![lat_v, lat_c, lat_h, lat_w],
        });
    }
    let [nrm_v, nrm_c, nrm_h, nrm_w] = *normal_dims else {
        return Err(DiffusionError::InvalidLatentShape {
            expected: vec![num_views, latent_channels, lat_h, lat_w],
            got: normal_dims.to_vec(),
        });
    };
    if (nrm_h, nrm_w) != (lat_h, lat_w) {
        return Err(DiffusionError::InvalidLatentShape {
            expected: vec![nrm_v, nrm_c, lat_h, lat_w],
            got: vec![nrm_v, nrm_c, nrm_h, nrm_w],
        });
    }
    Ok((lat_v, lat_c, lat_h, lat_w))
}

/// Split a `(V, C, H, W)` batch into `V` tensors of shape `(C, H, W)`.
pub(super) fn split_views(images: &Tensor) -> Result<Vec<Tensor>, DiffusionError> {
    let num_views = images
        .dim(0)
        .map_err(|e| DiffusionError::Inference(format!("images dim0: {e}")))?;
    let mut views = Vec::with_capacity(num_views);
    for i in 0..num_views {
        let img = images
            .narrow(0, i, 1)
            .and_then(|t| t.squeeze(0))
            .map_err(|e| DiffusionError::Inference(format!("split view {i}: {e}")))?;
        views.push(img);
    }
    Ok(views)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
//
// These cover the pure helpers above; every one of them runs without model
// weights, which is exactly why they were split out of `pipeline.rs` along
// with the code they exercise.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduler::{DdimScheduler, PredictionType};
    use candle_nn as nn;

    // -- Seeded noise (regression: `generate` used to ignore its seed) -------

    #[test]
    fn test_seeded_normal_values_is_deterministic() {
        let a = seeded_normal_values(64, 1234);
        let b = seeded_normal_values(64, 1234);
        assert_eq!(a, b, "same seed must produce bit-identical noise");
    }

    #[test]
    fn test_seeded_normal_values_differs_across_seeds() {
        let a = seeded_normal_values(64, 1);
        let b = seeded_normal_values(64, 2);
        assert_ne!(a, b, "different seeds must produce different noise");
    }

    #[test]
    fn test_seeded_normal_values_zero_seed_is_usable() {
        let values = seeded_normal_values(32, 0);
        assert_eq!(values.len(), 32);
        assert!(
            values.iter().any(|v| v.abs() > 1e-6),
            "seed 0 must not collapse the xorshift state to zeros"
        );
        assert!(values.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_seeded_normal_values_odd_count() {
        // Box-Muller yields pairs; an odd request must not over-fill.
        assert_eq!(seeded_normal_values(7, 99).len(), 7);
        assert_eq!(seeded_normal_values(1, 99).len(), 1);
        assert_eq!(seeded_normal_values(0, 99).len(), 0);
    }

    #[test]
    fn test_seeded_normal_values_are_roughly_standard_normal() {
        let values = seeded_normal_values(8192, 7);
        let n = values.len() as f32;
        let mean = values.iter().sum::<f32>() / n;
        let var = values.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
        assert!(mean.abs() < 0.1, "mean {mean} should be near 0");
        assert!((var - 1.0).abs() < 0.15, "variance {var} should be near 1");
    }

    #[test]
    fn test_seeded_normal_tensor_shape_and_determinism() -> Result<(), DiffusionError> {
        let device = Device::Cpu;
        let a = seeded_normal_tensor((2, 4, 8, 8), 42, &device)?;
        let b = seeded_normal_tensor((2, 4, 8, 8), 42, &device)?;
        assert_eq!(a.dims(), &[2, 4, 8, 8]);
        let a_vec = a
            .flatten_all()
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| DiffusionError::Inference(format!("{e}")))?;
        let b_vec = b
            .flatten_all()
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| DiffusionError::Inference(format!("{e}")))?;
        assert_eq!(a_vec, b_vec);
        Ok(())
    }

    // -- CFG ----------------------------------------------------------------

    #[test]
    fn test_needs_uncond_pass() {
        assert!(
            !needs_uncond_pass(1.0),
            "scale 1.0 must skip the uncond pass"
        );
        assert!(needs_uncond_pass(3.0));
        assert!(needs_uncond_pass(7.5));
    }

    #[test]
    fn test_combine_cfg_matches_formula() -> Result<(), DiffusionError> {
        let device = Device::Cpu;
        let cond = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1, 4), &device)
            .map_err(|e| DiffusionError::Inference(format!("{e}")))?;
        let uncond = Tensor::from_vec(vec![0.5f32, 0.5, 0.5, 0.5], (1, 4), &device)
            .map_err(|e| DiffusionError::Inference(format!("{e}")))?;
        let out = combine_cfg(&cond, &uncond, 2.0)?;
        let got = out
            .flatten_all()
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| DiffusionError::Inference(format!("{e}")))?;
        // 0.5 + 2 * (c - 0.5)
        let expected = [1.5f32, 3.5, 5.5, 7.5];
        for (g, e) in got.iter().zip(expected.iter()) {
            assert!((g - e).abs() < 1e-5, "got {g}, expected {e}");
        }
        Ok(())
    }

    // -- VAE chunking --------------------------------------------------------

    #[test]
    fn test_chunk_ranges_exact_split() {
        assert_eq!(chunk_ranges(4, 2), vec![(0, 2), (2, 2)]);
    }

    #[test]
    fn test_chunk_ranges_uneven_tail() {
        assert_eq!(chunk_ranges(5, 2), vec![(0, 2), (2, 2), (4, 1)]);
    }

    #[test]
    fn test_chunk_ranges_single_chunk() {
        assert_eq!(chunk_ranges(4, 8), vec![(0usize, 4usize)]);
        assert!(chunk_ranges(0, 4).is_empty());
    }

    #[test]
    fn test_chunk_ranges_zero_chunk_size_is_one() {
        assert_eq!(chunk_ranges(3, 0), vec![(0, 1), (1, 1), (2, 1)]);
    }

    #[test]
    fn test_chunk_ranges_cover_every_view_once() {
        for chunk in 1..=6 {
            let ranges = chunk_ranges(7, chunk);
            let covered: usize = ranges.iter().map(|(_, len)| len).sum();
            assert_eq!(covered, 7, "chunk {chunk} must cover all views");
            let mut next = 0;
            for (start, len) in ranges {
                assert_eq!(start, next);
                next += len;
            }
        }
    }

    #[test]
    fn test_component_size_charges_both_vae_halves() {
        let expected = ComponentType::VaeEncoder.estimated_size_mb()
            + ComponentType::VaeDecoder.estimated_size_mb();
        assert!((component_size_mb(ComponentType::VaeDecoder) - expected).abs() < 1e-3);
        assert!((component_size_mb(ComponentType::VaeEncoder) - expected).abs() < 1e-3);
        assert!(
            (component_size_mb(ComponentType::MultiViewUNet)
                - ComponentType::MultiViewUNet.estimated_size_mb())
            .abs()
                < 1e-3
        );
    }

    #[test]
    fn test_bilinear_upsampler_costs_no_weight_memory() {
        assert!(
            component_weight_mb(
                ComponentType::LatentUpsampler,
                Some(UpsamplerMode::BilinearVae)
            )
            .abs()
                < 1e-6,
            "BilinearVae holds no weights"
        );
        assert!(
            component_weight_mb(ComponentType::LatentUpsampler, Some(UpsamplerMode::SdX2)) > 0.0
        );
        // Non-upsampler components are unaffected by the mode.
        assert!(
            (component_weight_mb(
                ComponentType::MultiViewUNet,
                Some(UpsamplerMode::BilinearVae)
            ) - component_size_mb(ComponentType::MultiViewUNet))
            .abs()
                < 1e-3
        );
    }

    #[test]
    fn test_all_in_memory_never_releases() {
        for component in ComponentType::all_in_inference_order() {
            assert!(!component_should_release(
                OffloadStrategy::AllInMemory,
                *component
            ));
        }
    }

    #[test]
    fn test_sequential_releases_everything() {
        for component in ComponentType::all_in_inference_order() {
            assert!(component_should_release(
                OffloadStrategy::Sequential,
                *component
            ));
        }
    }

    #[test]
    fn test_cache_one_keeps_the_unet_resident() {
        assert!(!component_should_release(
            OffloadStrategy::CacheOne,
            ComponentType::MultiViewUNet
        ));
        for component in ComponentType::all_in_inference_order()
            .iter()
            .filter(|c| **c != ComponentType::MultiViewUNet)
        {
            assert!(component_should_release(
                OffloadStrategy::CacheOne,
                *component
            ));
        }
    }

    // -- KV-cache conditioning identity --------------------------------------

    fn tokens(values: &[f32], seq: usize, dim: usize) -> Result<Tensor, DiffusionError> {
        Tensor::from_vec(values.to_vec(), (1usize, seq, dim), &Device::Cpu)
            .map_err(|e| DiffusionError::Inference(format!("{e}")))
    }

    #[test]
    fn test_conditioning_tag_is_stable_for_identical_tokens() -> Result<(), DiffusionError> {
        let a = tokens(&[0.5f32, -1.0, 2.0, 3.5], 2, 2)?;
        let b = tokens(&[0.5f32, -1.0, 2.0, 3.5], 2, 2)?;
        assert_eq!(conditioning_tag(&a, 4)?, conditioning_tag(&b, 4)?);
        Ok(())
    }

    #[test]
    fn test_conditioning_tag_changes_with_the_tokens() -> Result<(), DiffusionError> {
        // A different reference image must not be able to hit another image's
        // cached K/V projections.
        let a = tokens(&[0.5f32, -1.0, 2.0, 3.5], 2, 2)?;
        let b = tokens(&[0.5f32, -1.0, 2.0, 3.6], 2, 2)?;
        assert_ne!(conditioning_tag(&a, 4)?, conditioning_tag(&b, 4)?);
        Ok(())
    }

    #[test]
    fn test_conditioning_tag_changes_with_the_view_count() -> Result<(), DiffusionError> {
        // Cached K/V carry the per-view expanded batch dimension, so the same
        // tokens at another view count are a different entry.
        let a = tokens(&[0.5f32, -1.0, 2.0, 3.5], 2, 2)?;
        assert_ne!(conditioning_tag(&a, 2)?, conditioning_tag(&a, 4)?);
        Ok(())
    }

    #[test]
    fn test_fnv1a_distinguishes_signed_zero_and_ordering() {
        // Hashing bit patterns, not values: +0.0 and -0.0 compare equal as
        // floats but are different conditioning.
        assert_ne!(
            fnv1a_f32(&[0.0], FNV_OFFSET_BASIS),
            fnv1a_f32(&[-0.0], FNV_OFFSET_BASIS)
        );
        assert_ne!(
            fnv1a_f32(&[1.0, 2.0], FNV_OFFSET_BASIS),
            fnv1a_f32(&[2.0, 1.0], FNV_OFFSET_BASIS)
        );
        assert_eq!(
            fnv1a_f32(&[1.0, 2.0], FNV_OFFSET_BASIS),
            fnv1a_f32(&[1.0, 2.0], FNV_OFFSET_BASIS)
        );
    }

    // -- Profiling (regression: DiffusionProfiler existed but nothing in the
    //    pipeline ever produced a sample) -----------------------------------

    #[test]
    fn test_profile_helpers_are_a_no_op_when_disabled() {
        // `None` is the disabled state; the helpers must simply return.
        profile_start(None, "clip_encode", 1234);
        profile_stop(None, "clip_encode");
    }

    #[test]
    fn test_profile_helpers_record_a_sample_when_enabled() {
        let mut profiler = DiffusionProfiler::new();
        assert!(profiler.profiles().is_empty());

        profile_start(Some(&mut profiler), "unet_forward", 4096);
        profile_stop(Some(&mut profiler), "unet_forward");

        let samples = profiler.profiles();
        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0].name, "unet_forward");
        assert_eq!(samples[0].estimated_memory_bytes, 4096);
        assert_eq!(profiler.total_estimated_memory_bytes(), 4096);
    }

    #[test]
    fn test_profile_stop_without_start_records_nothing() {
        let mut profiler = DiffusionProfiler::new();
        profile_stop(Some(&mut profiler), "never_started");
        assert!(profiler.profiles().is_empty());
    }

    #[test]
    fn test_unet_activation_estimate_scales_with_views_and_resolution() {
        let config = DiffusionConfig::default();
        let one_view = unet_activation_estimate(&config, 1, 32);
        let four_views = unet_activation_estimate(&config, 4, 32);
        let bigger = unet_activation_estimate(&config, 1, 64);
        assert!(one_view > 0);
        assert!(
            four_views > one_view,
            "four views must estimate more than one: {four_views} vs {one_view}"
        );
        assert!(
            bigger > one_view,
            "a 64×64 latent must estimate more than 32×32: {bigger} vs {one_view}"
        );
    }

    #[test]
    fn test_unet_activation_estimate_survives_a_degenerate_config() {
        // `num_stages()` is 0 for an empty channel_mult; the estimate must not
        // collapse to zero res-block cost (or divide by anything).
        let config = DiffusionConfig {
            channel_mult: Vec::new(),
            ..DiffusionConfig::default()
        };
        assert!(unet_activation_estimate(&config, 1, 32) > 0);
    }

    // -- start_step / img2img strength ---------------------------------------

    #[test]
    fn test_timesteps_from_start_step_zero_keeps_the_whole_schedule() {
        let schedule = [900usize, 800, 700, 600];
        let kept = timesteps_from_start_step(&schedule, 0).expect("start_step 0 must be valid");
        assert_eq!(kept, schedule.to_vec());
    }

    #[test]
    fn test_timesteps_from_start_step_drops_the_noisiest_steps() {
        // Regression: `begin_session_from_latents` used to run the *whole*
        // descending schedule regardless of how lightly the caller had noised
        // its latents, so an SDEdit run at strength 0.4 was denoised from
        // t=999 as if it were pure noise.
        let schedule = [900usize, 800, 700, 600];
        assert_eq!(
            timesteps_from_start_step(&schedule, 2).expect("valid"),
            vec![700, 600]
        );
        assert_eq!(
            timesteps_from_start_step(&schedule, 3).expect("valid"),
            vec![600]
        );
    }

    #[test]
    fn test_timesteps_from_start_step_rejects_an_empty_run() {
        let schedule = [900usize, 800];
        for start in [2usize, 3, 99] {
            let err = timesteps_from_start_step(&schedule, start)
                .expect_err("a start_step past the schedule must be rejected");
            assert!(matches!(err, DiffusionError::InvalidConfig(_)), "{err:?}");
        }
        assert!(timesteps_from_start_step(&[], 0).is_err());
    }

    /// `session_start_timestep` used to answer from a *second*, locally-built
    /// `DdimScheduler` that restated `(1000, VPrediction)` by hand. Verify the
    /// contract it now upholds: the timestep it reports is the one the run's
    /// schedule actually starts at, so an img2img caller noises to a level the
    /// scheduler's own alpha table agrees with.
    #[test]
    fn test_start_timestep_matches_the_schedule_the_run_uses() {
        let mut scheduler = DdimScheduler::new(1000, PredictionType::VPrediction);
        for steps in [10usize, 20, 50] {
            scheduler
                .set_timesteps(steps)
                .expect("set_timesteps failed");
            let schedule = scheduler.timesteps().to_vec();
            for start in [0usize, 1, steps - 1] {
                let kept = timesteps_from_start_step(&schedule, start).expect("valid start_step");
                assert_eq!(
                    kept.first().copied(),
                    Some(schedule[start]),
                    "steps {steps}, start {start}"
                );
            }
        }
    }

    #[test]
    fn test_first_retained_timestep_is_the_one_a_caller_must_noise_to() {
        // The contract documented on `session_start_timestep`: the first
        // timestep the session applies is `timesteps[start_step]`.
        let mut scheduler = DdimScheduler::new(1000, PredictionType::VPrediction);
        scheduler.set_timesteps(20).expect("set_timesteps failed");
        let full = scheduler.timesteps().to_vec();
        for start in [0usize, 1, 7, 19] {
            let kept = timesteps_from_start_step(&full, start).expect("valid start_step");
            assert_eq!(kept.first().copied(), Some(full[start]));
            assert_eq!(kept.len(), full.len() - start);
        }
    }

    #[test]
    fn test_validate_session_latents_accepts_matching_shapes() {
        let accepted = validate_session_latents(&[4, 4, 32, 32], &[4, 4, 32, 32], 4, 4, 32)
            .expect("matching shapes must be accepted");
        assert_eq!(accepted, (4, 4, 32, 32));
    }

    #[test]
    fn test_validate_session_latents_allows_non_config_spatial_size() {
        // The U-Net is fully convolutional, so an img2img caller may run at a
        // resolution other than `config.latent_size` as long as the normal-map
        // latents agree.
        let accepted = validate_session_latents(&[4, 4, 64, 64], &[4, 4, 64, 64], 4, 4, 32)
            .expect("a fully-convolutional run may use another latent size");
        assert_eq!(accepted, (4, 4, 64, 64));
    }

    #[test]
    fn test_validate_session_latents_rejects_wrong_rank() {
        let err = validate_session_latents(&[4, 32, 32], &[4, 4, 32, 32], 4, 4, 32)
            .expect_err("a 3-D latent tensor must be rejected");
        assert!(matches!(err, DiffusionError::InvalidLatentShape { .. }));
    }

    #[test]
    fn test_validate_session_latents_rejects_wrong_view_count() {
        let err = validate_session_latents(&[2, 4, 32, 32], &[2, 4, 32, 32], 4, 4, 32)
            .expect_err("a view-count mismatch must be rejected");
        match err {
            DiffusionError::InvalidLatentShape { expected, got } => {
                assert_eq!(expected, vec![4, 4, 32, 32]);
                assert_eq!(got, vec![2, 4, 32, 32]);
            }
            other => panic!("Expected InvalidLatentShape, got {other:?}"),
        }
    }

    #[test]
    fn test_validate_session_latents_rejects_wrong_channel_count() {
        let err = validate_session_latents(&[4, 8, 32, 32], &[4, 4, 32, 32], 4, 4, 32)
            .expect_err("a channel-count mismatch must be rejected");
        assert!(matches!(err, DiffusionError::InvalidLatentShape { .. }));
    }

    #[test]
    fn test_validate_session_latents_rejects_spatial_disagreement() {
        // step_session cats latents with normal_map_latents on the channel
        // axis, which needs identical spatial dims.
        let err = validate_session_latents(&[4, 4, 32, 32], &[4, 4, 16, 16], 4, 4, 32)
            .expect_err("normal maps at another resolution must be rejected");
        match err {
            DiffusionError::InvalidLatentShape { expected, got } => {
                assert_eq!(expected, vec![4, 4, 32, 32]);
                assert_eq!(got, vec![4, 4, 16, 16]);
            }
            other => panic!("Expected InvalidLatentShape, got {other:?}"),
        }
    }

    /// `encode_chunked` claims to be numerically identical to a batched encode
    /// while bounding peak activation memory; verify that against a real
    /// (randomly initialised) VAE, and check the 8× downsampling shape while a
    /// VAE is built. Both assertions share one `VarMap` because materialising
    /// the encoder's ~34M weights dominates the runtime of this test.
    #[test]
    fn test_encode_chunked_matches_batched_encode() -> Result<(), DiffusionError> {
        let device = Device::Cpu;
        let varmap = nn::VarMap::new();
        let vb = nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let vae = Vae::new(vb, 4, 0.18215)
            .map_err(|e| DiffusionError::Inference(format!("vae build: {e}")))?;

        let images = Tensor::randn(0f32, 1f32, (2usize, 3usize, 8usize, 8usize), &device)
            .map_err(|e| DiffusionError::Inference(format!("randn: {e}")))?;

        let read = |t: &Tensor| -> Result<Vec<f32>, DiffusionError> {
            t.flatten_all()
                .and_then(|f| f.to_vec1::<f32>())
                .map_err(|e| DiffusionError::Inference(format!("readback: {e}")))
        };

        // chunk_size >= num_views takes the single-batch fast path.
        let batched_tensor = encode_chunked(&vae, &images, 2)?;
        // The VAE downsamples 8×: 8×8 pixels → 1×1 latents.
        assert_eq!(batched_tensor.dims(), &[2, 4, 1, 1]);

        let batched = read(&batched_tensor)?;
        let chunked = read(&encode_chunked(&vae, &images, 1)?)?;
        assert_eq!(chunked.len(), batched.len(), "chunked length");
        for (i, (a, b)) in chunked.iter().zip(batched.iter()).enumerate() {
            assert!((a - b).abs() < 1e-5, "element {i}: {a} != {b}");
        }
        Ok(())
    }

    #[test]
    fn test_split_views_shapes() -> Result<(), DiffusionError> {
        let device = Device::Cpu;
        let images = Tensor::zeros((3, 3, 4, 4), DType::F32, &device)
            .map_err(|e| DiffusionError::Inference(format!("{e}")))?;
        let views = split_views(&images)?;
        assert_eq!(views.len(), 3);
        for view in &views {
            assert_eq!(view.dims(), &[3, 4, 4]);
        }
        Ok(())
    }
}
