//! Sequential (chunk-by-chunk) VAE encode/decode for low-memory GPUs.
//!
//! The standard diffusion pipeline passes all N views through the VAE in one
//! batch, which requires O(N · H · W) memory.  On GPUs with less than ~6 GB
//! of VRAM this can OOM.
//!
//! Two API shapes are provided:
//!
//! - [`encode_sequential`] / [`decode_sequential`] — convenience wrappers
//!   that accumulate every view's result into one [`EncodedViews`] /
//!   [`DecodedViews`] before returning. Because they hand back the *entire*
//!   batch at once, their own peak memory is `O(N · H · W)` — the same order
//!   as a single batched call — regardless of `chunk_size`; chunking only
//!   changes how many views are processed per inner pass, not how much is
//!   retained.
//! - [`encode_sequential_streaming`] / [`decode_sequential_streaming`] — the
//!   functions that actually deliver the low-memory guarantee. Each
//!   processed chunk is hand to a caller-supplied callback and then dropped
//!   before the next chunk starts, so *this module's own* working set is
//!   `O(chunk_size · H · W)` regardless of the total view count. [`encode_sequential`]
//!   / [`decode_sequential`] are implemented on top of these by accumulating
//!   every chunk, which is precisely the part that reintroduces `O(N)`
//!   memory — call the streaming variants directly and consume each chunk
//!   immediately (write it out, upload it, etc.) instead of collecting it if
//!   the memory bound needs to hold in your own code too.
//!
//! [`peak_memory_bytes`] reports the bound the *streaming* functions achieve
//! (`chunk_size · element_count · 4` bytes), independent of `num_views` —
//! that independence is the point of streaming. It does not describe
//! [`encode_sequential`] / [`decode_sequential`], whose actual peak scales
//! with `num_views`; see [`batch_memory_bytes`] for that figure.
//!
//! # Simulation note
//!
//! Because OxiGAF ships without pre-trained VAE weights, the encode/decode
//! functions in this module provide a **simulation**:
//!
//! - **Encode**: strided-subsample the spatial dimensions by 8× and apply the
//!   `latent_scale` factor.  Input channels are mapped to latent channels by
//!   cycling modulo the input channel count (3 for RGB).
//! - **Decode**: nearest-neighbour upsample 8× and divide by `latent_scale`.
//!   The output always has 3 channels (RGB) by cycling the latent channels.
//!
//! This lets the full memory-estimation and chunking logic be exercised in
//! tests without real model weights.

use crate::DiffusionError;

// ---------------------------------------------------------------------------
// SequentialVaeConfig
// ---------------------------------------------------------------------------

/// Configuration for sequential (one-chunk-at-a-time) VAE processing.
#[derive(Debug, Clone)]
pub struct SequentialVaeConfig {
    /// Number of views to encode/decode per chunk.
    ///
    /// Default: `1`.
    pub chunk_size: usize,

    /// Number of latent channels produced/consumed by the VAE.
    ///
    /// Default: `4` (matches SD 2.1 VAE).
    pub latent_channels: usize,

    /// Input image height in pixels.
    ///
    /// Default: `256`.
    pub image_height: usize,

    /// Input image width in pixels.
    ///
    /// Default: `256`.
    pub image_width: usize,

    /// Latent scale factor applied to the encoded representation.
    ///
    /// Default: `0.18215` (SD 2.1 VAE scaling factor).
    pub latent_scale: f32,
}

impl Default for SequentialVaeConfig {
    fn default() -> Self {
        Self {
            chunk_size: 1,
            latent_channels: 4,
            image_height: 256,
            image_width: 256,
            latent_scale: 0.18215,
        }
    }
}

impl SequentialVaeConfig {
    /// Create a new configuration with explicit parameters.
    pub fn new(
        chunk_size: usize,
        latent_channels: usize,
        image_height: usize,
        image_width: usize,
        latent_scale: f32,
    ) -> Self {
        Self {
            chunk_size,
            latent_channels,
            image_height,
            image_width,
            latent_scale,
        }
    }

    /// Validate that all parameters are within acceptable ranges.
    ///
    /// # Errors
    ///
    /// Returns [`DiffusionError::InvalidConfig`] when:
    /// - `chunk_size == 0`
    /// - `latent_channels == 0`
    /// - `image_height == 0`
    /// - `image_width == 0`
    /// - `latent_scale <= 0.0`
    pub fn validate(&self) -> Result<(), DiffusionError> {
        if self.chunk_size == 0 {
            return Err(DiffusionError::InvalidConfig(
                "chunk_size must be > 0".into(),
            ));
        }
        if self.latent_channels == 0 {
            return Err(DiffusionError::InvalidConfig(
                "latent_channels must be > 0".into(),
            ));
        }
        if self.image_height == 0 {
            return Err(DiffusionError::InvalidConfig(
                "image_height must be > 0".into(),
            ));
        }
        if self.image_width == 0 {
            return Err(DiffusionError::InvalidConfig(
                "image_width must be > 0".into(),
            ));
        }
        if self.latent_scale <= 0.0 {
            return Err(DiffusionError::InvalidConfig(
                "latent_scale must be > 0.0".into(),
            ));
        }
        Ok(())
    }

    /// Spatial downscale factor applied by the VAE encoder.
    const SPATIAL_FACTOR: usize = 8;

    /// Latent height derived from image height.
    pub fn latent_height(&self) -> usize {
        self.image_height / Self::SPATIAL_FACTOR
    }

    /// Latent width derived from image width.
    pub fn latent_width(&self) -> usize {
        self.image_width / Self::SPATIAL_FACTOR
    }

    /// Number of f32 elements in one latent vector.
    pub fn latent_element_count(&self) -> usize {
        self.latent_channels * self.latent_height() * self.latent_width()
    }

    /// Number of f32 elements in one input image (3-channel RGB).
    pub fn image_element_count(&self) -> usize {
        3 * self.image_height * self.image_width
    }
}

// ---------------------------------------------------------------------------
// EncodedViews
// ---------------------------------------------------------------------------

/// Result of a sequential VAE encoding pass.
#[derive(Debug, Clone)]
pub struct EncodedViews {
    /// One flat `Vec<f32>` per input view.
    ///
    /// Each inner vector has `latent_channels * latent_height * latent_width`
    /// elements, stored in `(C, H, W)` order.
    pub latents: Vec<Vec<f32>>,

    /// Number of views encoded.
    pub num_views: usize,

    /// Latent spatial height (`image_height / 8`).
    pub latent_height: usize,

    /// Latent spatial width (`image_width / 8`).
    pub latent_width: usize,

    /// Number of latent channels.
    pub latent_channels: usize,
}

// ---------------------------------------------------------------------------
// DecodedViews
// ---------------------------------------------------------------------------

/// Result of a sequential VAE decoding pass.
#[derive(Debug, Clone)]
pub struct DecodedViews {
    /// One flat `Vec<f32>` per decoded view.
    ///
    /// Each inner vector has `channels * height * width` elements in `(C, H, W)`
    /// order.
    pub images: Vec<Vec<f32>>,

    /// Number of views decoded.
    pub num_views: usize,

    /// Decoded image height (latent_height × 8).
    pub height: usize,

    /// Decoded image width (latent_width × 8).
    pub width: usize,

    /// Number of image channels (always 3 for RGB).
    pub channels: usize,
}

// ---------------------------------------------------------------------------
// encode_sequential
// ---------------------------------------------------------------------------

/// Encode a slice of RGB images into latent representations one chunk at a
/// time, invoking `on_chunk` with each freshly-computed chunk of latents
/// before moving on to the next one.
///
/// Unlike [`encode_sequential`], this function never retains more than
/// `config.chunk_size` encoded views at once internally — each chunk is
/// handed to `on_chunk` and then dropped. Peak memory inside this function is
/// therefore genuinely `O(chunk_size · H · W)` regardless of the total view
/// count (matching [`peak_memory_bytes`]), as long as `on_chunk` does not
/// itself accumulate every chunk it receives.
///
/// Returns the total number of views encoded. Per-view geometry
/// (`latent_height`/`latent_width`/`latent_channels`) is available from
/// `config` and does not vary across chunks.
///
/// # Errors
///
/// Returns [`DiffusionError::InvalidConfig`] when:
/// - `images` is empty.
/// - Any image has the wrong element count.
/// - Config validation fails.
///
/// Propagates whatever error `on_chunk` returns, short-circuiting any
/// remaining chunks.
pub fn encode_sequential_streaming<F>(
    images: &[Vec<f32>],
    config: &SequentialVaeConfig,
    mut on_chunk: F,
) -> Result<usize, DiffusionError>
where
    F: FnMut(Vec<Vec<f32>>) -> Result<(), DiffusionError>,
{
    config.validate()?;

    if images.is_empty() {
        return Err(DiffusionError::InvalidConfig(
            "images slice must not be empty".into(),
        ));
    }

    let expected_len = config.image_element_count();
    let lh = config.latent_height();
    let lw = config.latent_width();
    let lc = config.latent_channels;
    let ih = config.image_height;
    let iw = config.image_width;
    let scale = config.latent_scale;

    let mut total = 0usize;

    // Process in chunks of `chunk_size`; `chunk_latents` is reallocated
    // fresh for every chunk and moved into `on_chunk`, so at most one
    // chunk's worth of encoded output is ever live inside this function.
    for chunk in images.chunks(config.chunk_size) {
        let mut chunk_latents: Vec<Vec<f32>> = Vec::with_capacity(chunk.len());
        for image in chunk {
            if image.len() != expected_len {
                return Err(DiffusionError::InvalidConfig(format!(
                    "image has {} elements but expected {} (3 × {} × {})",
                    image.len(),
                    expected_len,
                    ih,
                    iw,
                )));
            }

            chunk_latents.push(encode_one_image(image, lc, lh, lw, iw, scale));
        }
        total += chunk_latents.len();
        on_chunk(chunk_latents)?;
    }

    Ok(total)
}

/// Encode a slice of RGB images into latent representations, accumulating
/// every view's result before returning.
///
/// This is a convenience wrapper around [`encode_sequential_streaming`] that
/// collects all chunks into one [`EncodedViews`]. Because it must hold every
/// output simultaneously to return it, **its own peak memory is `O(num_views
/// · H · W)`, not `O(chunk_size · H · W)`** — see the module docs. Call
/// [`encode_sequential_streaming`] directly and consume each chunk
/// immediately instead of collecting it when the low-memory bound needs to
/// hold in the caller's code too.
///
/// # Arguments
///
/// * `images`  – Flat `(C, H, W)` f32 images.  Each inner `Vec` must have
///   length `3 × config.image_height × config.image_width`.
/// * `config`  – Encoding parameters including chunk size and latent scale.
///
/// # Errors
///
/// Returns [`DiffusionError::InvalidConfig`] when:
/// - `images` is empty.
/// - Any image has the wrong element count.
/// - Config validation fails.
pub fn encode_sequential(
    images: &[Vec<f32>],
    config: &SequentialVaeConfig,
) -> Result<EncodedViews, DiffusionError> {
    let lh = config.latent_height();
    let lw = config.latent_width();
    let lc = config.latent_channels;

    let mut latents: Vec<Vec<f32>> = Vec::with_capacity(images.len());
    let num_views = encode_sequential_streaming(images, config, |chunk| {
        latents.extend(chunk);
        Ok(())
    })?;

    Ok(EncodedViews {
        num_views,
        latent_height: lh,
        latent_width: lw,
        latent_channels: lc,
        latents,
    })
}

// ---------------------------------------------------------------------------
// decode_sequential
// ---------------------------------------------------------------------------

/// Decode latent representations back to RGB images one chunk at a time,
/// invoking `on_chunk` with each freshly-decoded chunk of images before
/// moving on to the next one.
///
/// Unlike [`decode_sequential`], this function never retains more than
/// `config.chunk_size` decoded images at once internally, so its own peak
/// memory is genuinely `O(chunk_size · H · W)` — see
/// [`encode_sequential_streaming`] for the same guarantee on the encode
/// side, and the module docs for the full explanation.
///
/// Returns the total number of views decoded.
///
/// # Errors
///
/// Returns [`DiffusionError::InvalidConfig`] when:
/// - Config validation fails.
/// - Any latent has the wrong element count.
///
/// Propagates whatever error `on_chunk` returns, short-circuiting any
/// remaining chunks.
pub fn decode_sequential_streaming<F>(
    encoded: &EncodedViews,
    config: &SequentialVaeConfig,
    mut on_chunk: F,
) -> Result<usize, DiffusionError>
where
    F: FnMut(Vec<Vec<f32>>) -> Result<(), DiffusionError>,
{
    config.validate()?;

    let expected_latent_len = config.latent_element_count();
    let out_h = config.image_height;
    let out_w = config.image_width;
    let out_c: usize = 3; // always RGB
    let lc = config.latent_channels;
    let lh = config.latent_height();
    let lw = config.latent_width();
    let scale = config.latent_scale;

    let mut total = 0usize;

    for chunk in encoded.latents.chunks(config.chunk_size) {
        let mut chunk_images: Vec<Vec<f32>> = Vec::with_capacity(chunk.len());
        for latent in chunk {
            if latent.len() != expected_latent_len {
                return Err(DiffusionError::InvalidConfig(format!(
                    "latent has {} elements but expected {} ({}×{}×{})",
                    latent.len(),
                    expected_latent_len,
                    lc,
                    lh,
                    lw,
                )));
            }

            chunk_images.push(decode_one_latent(
                latent,
                (lc, lh, lw),
                (out_c, out_h, out_w),
                scale,
            ));
        }
        total += chunk_images.len();
        on_chunk(chunk_images)?;
    }

    Ok(total)
}

/// Decode latent representations back to RGB images, accumulating every
/// view's result before returning.
///
/// This is a convenience wrapper around [`decode_sequential_streaming`] that
/// collects all chunks into one [`DecodedViews`]. Because it must hold every
/// output simultaneously to return it, **its own peak memory is `O(num_views
/// · H · W)`, not `O(chunk_size · H · W)`** — see the module docs. Call
/// [`decode_sequential_streaming`] directly and consume each chunk
/// immediately instead of collecting it when the low-memory bound needs to
/// hold in the caller's code too.
///
/// # Errors
///
/// Returns [`DiffusionError::InvalidConfig`] when:
/// - Config validation fails.
/// - Any latent has the wrong element count.
pub fn decode_sequential(
    encoded: &EncodedViews,
    config: &SequentialVaeConfig,
) -> Result<DecodedViews, DiffusionError> {
    let out_h = config.image_height;
    let out_w = config.image_width;
    let out_c: usize = 3;

    let mut images: Vec<Vec<f32>> = Vec::with_capacity(encoded.latents.len());
    let num_views = decode_sequential_streaming(encoded, config, |chunk| {
        images.extend(chunk);
        Ok(())
    })?;

    Ok(DecodedViews {
        num_views,
        height: out_h,
        width: out_w,
        channels: out_c,
        images,
    })
}

// ---------------------------------------------------------------------------
// Memory estimation
// ---------------------------------------------------------------------------

/// Peak memory used (bytes) by the streaming sequential approach for the
/// given configuration.
///
/// This is the bound [`encode_sequential_streaming`] /
/// [`decode_sequential_streaming`] actually achieve: only `chunk_size`
/// images are ever live inside those functions at once, so
/// `memory = chunk_size × image_element_count × 4 bytes/f32` — independent
/// of `num_views` by construction, which is why `_num_views` is unused here
/// (it exists only so this function's signature matches
/// [`batch_memory_bytes`] for [`memory_reduction_ratio`]).
///
/// [`encode_sequential`] / [`decode_sequential`] do *not* achieve this bound
/// — they accumulate every chunk before returning, so their actual peak
/// scales with `num_views` like [`batch_memory_bytes`] rather than with
/// `chunk_size`; see the module docs.
pub fn peak_memory_bytes(config: &SequentialVaeConfig, _num_views: usize) -> usize {
    config.chunk_size * config.image_element_count() * std::mem::size_of::<f32>()
}

/// Memory required (bytes) if all views were processed in one batch.
///
/// Batch memory = num_views × image_element_count × 4 bytes/f32.
pub fn batch_memory_bytes(config: &SequentialVaeConfig, num_views: usize) -> usize {
    num_views * config.image_element_count() * std::mem::size_of::<f32>()
}

/// Ratio of batch memory to sequential peak memory.
///
/// Values greater than 1.0 indicate that sequential processing saves memory.
/// Returns 1.0 when `num_views == chunk_size` (no saving).
pub fn memory_reduction_ratio(config: &SequentialVaeConfig, num_views: usize) -> f32 {
    let batch = batch_memory_bytes(config, num_views) as f32;
    let sequential = peak_memory_bytes(config, num_views) as f32;
    if sequential == 0.0 {
        return 1.0;
    }
    batch / sequential
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Encode a single RGB image into a latent tensor.
///
/// Simulation: strided subsample by 8× in both spatial dimensions, cycle
/// input channels modulo 3 to fill `latent_channels`, then multiply by
/// `latent_scale`.
fn encode_one_image(
    image: &[f32],
    lc: usize,
    lh: usize,
    lw: usize,
    iw: usize,
    scale: f32,
) -> Vec<f32> {
    // Input layout: (3, ih, iw) — flat row-major
    // Output layout: (lc, lh, lw) — flat row-major
    let stride = 8_usize;
    let in_channels = 3_usize;

    let mut latent = Vec::with_capacity(lc * lh * lw);

    for c in 0..lc {
        let src_c = c % in_channels;
        for row in 0..lh {
            for col in 0..lw {
                let src_row = row * stride;
                let src_col = col * stride;
                let src_idx = src_c * (image.len() / in_channels) + src_row * iw + src_col;
                // Guard against out-of-bounds (shouldn't happen with valid config).
                let pixel = if src_idx < image.len() {
                    image[src_idx]
                } else {
                    0.0
                };
                latent.push(pixel * scale);
            }
        }
    }

    latent
}

/// Decode a single latent tensor into an RGB image.
///
/// Simulation: nearest-neighbour upsample by 8×, cycle latent channels to
/// produce exactly 3 output channels, divide by `latent_scale`.
fn decode_one_latent(
    latent: &[f32],
    (lc, lh, lw): (usize, usize, usize),
    (out_c, out_h, out_w): (usize, usize, usize),
    scale: f32,
) -> Vec<f32> {
    // Latent layout: (lc, lh, lw)
    // Output layout: (out_c, out_h, out_w)
    let inv_scale = if scale == 0.0 { 1.0 } else { 1.0 / scale };
    let mut image = Vec::with_capacity(out_c * out_h * out_w);

    for c in 0..out_c {
        let src_c = c % lc;
        for row in 0..out_h {
            for col in 0..out_w {
                let src_row = row / 8;
                let src_col = col / 8;
                // Clamp to latent bounds.
                let src_row = src_row.min(lh.saturating_sub(1));
                let src_col = src_col.min(lw.saturating_sub(1));
                let src_idx = src_c * lh * lw + src_row * lw + src_col;
                let val = if src_idx < latent.len() {
                    latent[src_idx]
                } else {
                    0.0
                };
                image.push(val * inv_scale);
            }
        }
    }

    image
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Build a valid config suitable for small tests (64×64 images → 8×8 latents).
    fn small_config() -> SequentialVaeConfig {
        SequentialVaeConfig::new(1, 4, 64, 64, 0.18215)
    }

    /// Create a synthetic RGB image at the dimensions specified by `config`.
    fn synthetic_image(config: &SequentialVaeConfig) -> Vec<f32> {
        let len = config.image_element_count();
        (0..len).map(|i| (i as f32) / (len as f32)).collect()
    }

    // -----------------------------------------------------------------------
    // test_default_config
    // -----------------------------------------------------------------------

    #[test]
    fn test_default_config() {
        let cfg = SequentialVaeConfig::default();
        assert_eq!(cfg.chunk_size, 1);
        assert_eq!(cfg.latent_channels, 4);
        assert_eq!(cfg.image_height, 256);
        assert_eq!(cfg.image_width, 256);
        assert!((cfg.latent_scale - 0.18215).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // test_config_validation_chunk_size_zero
    // -----------------------------------------------------------------------

    #[test]
    fn test_config_validation_chunk_size_zero() {
        let cfg = SequentialVaeConfig::new(0, 4, 64, 64, 0.18215);
        let result = cfg.validate();
        assert!(result.is_err(), "chunk_size=0 should fail validation");
        let err = result.expect_err("error expected");
        assert!(
            err.to_string().contains("chunk_size"),
            "error message should mention chunk_size, got: {err}"
        );
    }

    // -----------------------------------------------------------------------
    // test_config_validation_negative_scale
    // -----------------------------------------------------------------------

    #[test]
    fn test_config_validation_negative_scale() {
        let cfg = SequentialVaeConfig::new(1, 4, 64, 64, -1.0);
        let result = cfg.validate();
        assert!(
            result.is_err(),
            "negative latent_scale should fail validation"
        );
    }

    // -----------------------------------------------------------------------
    // test_encode_single_view
    // -----------------------------------------------------------------------

    #[test]
    fn test_encode_single_view() {
        let cfg = small_config();
        let image = synthetic_image(&cfg);
        let encoded = encode_sequential(&[image], &cfg).expect("encoding failed");
        assert_eq!(encoded.num_views, 1);
        assert_eq!(encoded.latents.len(), 1);
    }

    // -----------------------------------------------------------------------
    // test_encode_multiple_views
    // -----------------------------------------------------------------------

    #[test]
    fn test_encode_multiple_views() {
        let cfg = small_config();
        let views: Vec<Vec<f32>> = (0..4).map(|_| synthetic_image(&cfg)).collect();
        let encoded = encode_sequential(&views, &cfg).expect("encoding failed");
        assert_eq!(encoded.num_views, 4);
        assert_eq!(encoded.latents.len(), 4);
    }

    // -----------------------------------------------------------------------
    // test_encode_in_chunks
    // -----------------------------------------------------------------------

    #[test]
    fn test_encode_in_chunks() {
        // chunk_size=2, 6 views → 3 chunks
        let cfg = SequentialVaeConfig::new(2, 4, 64, 64, 0.18215);
        let views: Vec<Vec<f32>> = (0..6).map(|_| synthetic_image(&cfg)).collect();
        let encoded = encode_sequential(&views, &cfg).expect("encoding failed");
        assert_eq!(encoded.num_views, 6);
    }

    // -----------------------------------------------------------------------
    // test_decode_restores_shape
    // -----------------------------------------------------------------------

    #[test]
    fn test_decode_restores_shape() {
        let cfg = small_config();
        let image = synthetic_image(&cfg);
        let encoded = encode_sequential(&[image], &cfg).expect("encoding failed");
        let decoded = decode_sequential(&encoded, &cfg).expect("decoding failed");
        assert_eq!(decoded.num_views, 1);
        let expected_len = cfg.image_element_count(); // 3 × 64 × 64
        assert_eq!(decoded.images[0].len(), expected_len);
    }

    // -----------------------------------------------------------------------
    // test_encode_decode_roundtrip_shape
    // -----------------------------------------------------------------------

    #[test]
    fn test_encode_decode_roundtrip_shape() {
        let cfg = small_config();
        let views: Vec<Vec<f32>> = (0..3).map(|_| synthetic_image(&cfg)).collect();
        let encoded = encode_sequential(&views, &cfg).expect("encoding failed");
        let decoded = decode_sequential(&encoded, &cfg).expect("decoding failed");
        assert_eq!(decoded.num_views, 3);
        for img in &decoded.images {
            assert_eq!(img.len(), cfg.image_element_count());
        }
    }

    // -----------------------------------------------------------------------
    // test_encoded_views_metadata
    // -----------------------------------------------------------------------

    #[test]
    fn test_encoded_views_metadata() {
        let cfg = small_config();
        let image = synthetic_image(&cfg);
        let encoded = encode_sequential(&[image], &cfg).expect("encoding failed");
        assert_eq!(encoded.latent_channels, cfg.latent_channels);
        assert_eq!(encoded.latent_height, cfg.latent_height());
        assert_eq!(encoded.latent_width, cfg.latent_width());
        // 64/8 = 8
        assert_eq!(encoded.latent_height, 8);
        assert_eq!(encoded.latent_width, 8);
    }

    // -----------------------------------------------------------------------
    // test_decoded_views_metadata
    // -----------------------------------------------------------------------

    #[test]
    fn test_decoded_views_metadata() {
        let cfg = small_config();
        let image = synthetic_image(&cfg);
        let encoded = encode_sequential(&[image], &cfg).expect("encoding failed");
        let decoded = decode_sequential(&encoded, &cfg).expect("decoding failed");
        assert_eq!(decoded.channels, 3);
        assert_eq!(decoded.height, cfg.image_height);
        assert_eq!(decoded.width, cfg.image_width);
    }

    // -----------------------------------------------------------------------
    // test_peak_memory_sequential_vs_batch
    // -----------------------------------------------------------------------

    #[test]
    fn test_peak_memory_sequential_vs_batch() {
        let cfg = SequentialVaeConfig::new(1, 4, 64, 64, 0.18215);
        let num_views = 8;
        let seq_mem = peak_memory_bytes(&cfg, num_views);
        let batch_mem = batch_memory_bytes(&cfg, num_views);

        // Sequential should use less memory when chunk_size < num_views.
        assert!(
            seq_mem < batch_mem,
            "sequential ({seq_mem} B) should use less than batch ({batch_mem} B)"
        );

        // Sequential peak = chunk_size × image_size × 4
        let expected_seq = cfg.chunk_size * cfg.image_element_count() * 4;
        assert_eq!(seq_mem, expected_seq);
    }

    // -----------------------------------------------------------------------
    // test_memory_reduction_ratio
    // -----------------------------------------------------------------------

    #[test]
    fn test_memory_reduction_ratio() {
        let cfg = SequentialVaeConfig::new(1, 4, 64, 64, 0.18215);
        let ratio = memory_reduction_ratio(&cfg, 8);
        // With chunk_size=1 and 8 views, ratio should be 8.
        assert!(
            (ratio - 8.0).abs() < 1e-4,
            "expected ratio ~8.0, got {ratio}"
        );

        // With chunk_size == num_views, ratio should be 1.
        let cfg2 = SequentialVaeConfig::new(8, 4, 64, 64, 0.18215);
        let ratio2 = memory_reduction_ratio(&cfg2, 8);
        assert!(
            (ratio2 - 1.0).abs() < 1e-4,
            "expected ratio ~1.0, got {ratio2}"
        );
    }

    // -----------------------------------------------------------------------
    // test_empty_views_error
    // -----------------------------------------------------------------------

    #[test]
    fn test_empty_views_error() {
        let cfg = small_config();
        let result = encode_sequential(&[], &cfg);
        assert!(result.is_err(), "empty images slice should return an error");
    }

    // -----------------------------------------------------------------------
    // test_latent_scale_applied
    // -----------------------------------------------------------------------

    #[test]
    fn test_latent_scale_applied() {
        let cfg = SequentialVaeConfig::new(1, 4, 64, 64, 2.0);
        // Image filled with 1.0
        let image = vec![1.0_f32; cfg.image_element_count()];
        let encoded = encode_sequential(&[image], &cfg).expect("encoding failed");
        // After encoding, all latent values should equal 1.0 × 2.0 = 2.0
        for &val in &encoded.latents[0] {
            assert!(
                (val - 2.0).abs() < 1e-5,
                "expected latent value 2.0, got {val}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // test_chunk_size_larger_than_views
    // -----------------------------------------------------------------------

    #[test]
    fn test_chunk_size_larger_than_views() {
        // chunk_size=10 but only 3 views — should work gracefully.
        let cfg = SequentialVaeConfig::new(10, 4, 64, 64, 0.18215);
        let views: Vec<Vec<f32>> = (0..3).map(|_| synthetic_image(&cfg)).collect();
        let encoded = encode_sequential(&views, &cfg).expect("encoding failed");
        assert_eq!(encoded.num_views, 3);
    }

    // -----------------------------------------------------------------------
    // Streaming variants: genuine O(chunk_size) peak memory
    // -----------------------------------------------------------------------
    //
    // Regression coverage for the "cosmetic chunking" bug: `encode_sequential`
    // / `decode_sequential` accumulate every chunk, so chunking them did not
    // change their own peak memory at all — it was semantically identical to
    // a flat iteration. `encode_sequential_streaming` /
    // `decode_sequential_streaming` are the functions that actually bound
    // their own working set to `chunk_size`; the tests below verify that
    // bound is real (never more than `chunk_size` items reach the callback
    // at once) and that the results are numerically identical to the
    // accumulating convenience wrappers.

    #[test]
    fn test_encode_streaming_never_exceeds_chunk_size() {
        let cfg = SequentialVaeConfig::new(2, 4, 64, 64, 0.18215);
        let views: Vec<Vec<f32>> = (0..7).map(|_| synthetic_image(&cfg)).collect();

        let mut max_chunk_len = 0usize;
        let mut total_seen = 0usize;
        let num_views = encode_sequential_streaming(&views, &cfg, |chunk| {
            max_chunk_len = max_chunk_len.max(chunk.len());
            total_seen += chunk.len();
            Ok(())
        })
        .expect("streaming encode failed");

        assert_eq!(num_views, 7);
        assert_eq!(total_seen, 7);
        assert!(
            max_chunk_len <= 2,
            "a chunk exceeded config.chunk_size=2: got {max_chunk_len}"
        );
    }

    #[test]
    fn test_decode_streaming_never_exceeds_chunk_size() {
        let cfg = SequentialVaeConfig::new(3, 4, 64, 64, 0.18215);
        let views: Vec<Vec<f32>> = (0..8).map(|_| synthetic_image(&cfg)).collect();
        let encoded = encode_sequential(&views, &cfg).expect("encode failed");

        let mut max_chunk_len = 0usize;
        let mut total_seen = 0usize;
        let num_views = decode_sequential_streaming(&encoded, &cfg, |chunk| {
            max_chunk_len = max_chunk_len.max(chunk.len());
            total_seen += chunk.len();
            Ok(())
        })
        .expect("streaming decode failed");

        assert_eq!(num_views, 8);
        assert_eq!(total_seen, 8);
        assert!(
            max_chunk_len <= 3,
            "a chunk exceeded config.chunk_size=3: got {max_chunk_len}"
        );
    }

    #[test]
    fn test_encode_streaming_matches_accumulating_wrapper() {
        let cfg = SequentialVaeConfig::new(2, 4, 64, 64, 0.18215);
        let views: Vec<Vec<f32>> = (0..5).map(|_| synthetic_image(&cfg)).collect();

        let accumulated = encode_sequential(&views, &cfg).expect("encode failed");

        let mut streamed: Vec<Vec<f32>> = Vec::new();
        encode_sequential_streaming(&views, &cfg, |chunk| {
            streamed.extend(chunk);
            Ok(())
        })
        .expect("streaming encode failed");

        assert_eq!(accumulated.latents, streamed);
    }

    #[test]
    fn test_encode_streaming_propagates_callback_error() {
        let cfg = SequentialVaeConfig::new(1, 4, 64, 64, 0.18215);
        let views: Vec<Vec<f32>> = (0..3).map(|_| synthetic_image(&cfg)).collect();
        let result = encode_sequential_streaming(&views, &cfg, |_chunk| {
            Err(DiffusionError::InvalidConfig("stop".into()))
        });
        assert!(result.is_err(), "callback error must propagate");
    }

    #[test]
    fn test_encode_streaming_empty_images_error() {
        let cfg = SequentialVaeConfig::new(1, 4, 64, 64, 0.18215);
        let empty: Vec<Vec<f32>> = vec![];
        let result = encode_sequential_streaming(&empty, &cfg, |_| Ok(()));
        assert!(result.is_err(), "empty images slice should return an error");
    }
}
