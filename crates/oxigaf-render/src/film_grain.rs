//! Film grain simulation for rendered 3DGS images.
//!
//! Unlike simple Gaussian noise, film grain:
//! - Varies with local luminance (midtone-peaked parabolic curve)
//! - Has spatial correlation (clumped, not independent pixels), via bilinear
//!   upsampling from a lower-resolution grain map
//! - Supports chroma grain for color separation
//! - Supports temporal sequences with configurable coherence
//!
//! All functions operate on flat `Vec<f32>` / `&[f32]` images (RGB or RGBA,
//! row-major, values in [0, 1]).

use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// PRNG — inline xorshift64 + Box-Muller
// ─────────────────────────────────────────────────────────────────────────────

/// Advance a 64-bit xorshift state and return the new value.
#[inline]
fn xorshift64(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

/// Draw a uniformly distributed f32 in [0, 1) from the xorshift64 state.
#[inline]
fn xorshift_f32(state: &mut u64) -> f32 {
    (xorshift64(state) >> 32) as f32 / u32::MAX as f32
}

/// Box-Muller standard normal sample N(0, 1).
#[inline]
fn sample_normal(state: &mut u64) -> f32 {
    let u1 = xorshift_f32(state).max(1e-10);
    let u2 = xorshift_f32(state);
    (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
}

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors produced by film grain operations.
#[derive(Debug, Error)]
pub enum FilmGrainError {
    /// Configuration parameter is out of valid range.
    #[error("Invalid film grain config: {0}")]
    InvalidConfig(String),

    /// Image buffer length does not match declared dimensions.
    #[error("Invalid image: {0}")]
    InvalidImage(String),

    /// Image has zero pixels.
    #[error("Image is empty (zero width or height)")]
    EmptyImage,
}

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for film grain simulation.
#[derive(Debug, Clone)]
pub struct FilmGrainConfig {
    /// Base grain intensity (sigma of added noise, in [0, 1] scale).
    ///
    /// Default: 0.05.
    pub intensity: f32,

    /// Grain size (spatial correlation) in pixels.
    ///
    /// 1.0 = per-pixel independent grain; larger values produce coarser,
    /// more clumped grain by generating at a lower resolution and bilinearly
    /// upsampling. Must be >= 1.0.
    ///
    /// Default: 1.5.
    pub grain_size: f32,

    /// Luminance-dependent scaling.
    ///
    /// If `true`, grain is strongest in midtones and weaker near pure black or
    /// pure white (parabola `4 * l * (1 - l)`).  If `false`, grain intensity
    /// is constant across all luminance levels.
    ///
    /// Default: true.
    pub luminance_scaling: bool,

    /// Chroma grain fraction in [0, 1].
    ///
    /// `0.0` = only luma (monochromatic) grain; `1.0` = equal independent RGB
    /// grain channels.  Values in between blend luma and per-channel grain.
    ///
    /// Default: 0.3.
    pub chroma_fraction: f32,

    /// Random seed for reproducible grain.
    ///
    /// Default: 42.
    pub seed: u64,

    /// Clip output to [0, 1] after adding grain.
    ///
    /// Default: true.
    pub clip_output: bool,
}

impl Default for FilmGrainConfig {
    fn default() -> Self {
        Self {
            intensity: 0.05,
            grain_size: 1.5,
            luminance_scaling: true,
            chroma_fraction: 0.3,
            seed: 42,
            clip_output: true,
        }
    }
}

impl FilmGrainConfig {
    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// - [`FilmGrainError::InvalidConfig`] if `intensity < 0`, `grain_size < 1.0`,
    ///   or `chroma_fraction` is outside [0, 1].
    pub fn validate(&self) -> Result<(), FilmGrainError> {
        if self.intensity < 0.0 {
            return Err(FilmGrainError::InvalidConfig(format!(
                "intensity must be >= 0, got {}",
                self.intensity
            )));
        }
        if self.grain_size < 1.0 {
            return Err(FilmGrainError::InvalidConfig(format!(
                "grain_size must be >= 1.0, got {}",
                self.grain_size
            )));
        }
        if !(0.0..=1.0).contains(&self.chroma_fraction) {
            return Err(FilmGrainError::InvalidConfig(format!(
                "chroma_fraction must be in [0, 1], got {}",
                self.chroma_fraction
            )));
        }
        Ok(())
    }

    /// Cinematic preset: medium grain with slight chroma.
    pub fn cinematic() -> Self {
        Self {
            intensity: 0.08,
            grain_size: 2.0,
            luminance_scaling: true,
            chroma_fraction: 0.2,
            seed: 42,
            clip_output: true,
        }
    }

    /// Fine grain preset: subtle, nearly invisible grain.
    pub fn fine() -> Self {
        Self {
            intensity: 0.03,
            grain_size: 1.0,
            luminance_scaling: true,
            chroma_fraction: 0.1,
            seed: 42,
            clip_output: true,
        }
    }

    /// Heavy grain preset: strongly visible, coarse grain.
    pub fn heavy() -> Self {
        Self {
            intensity: 0.15,
            grain_size: 3.0,
            luminance_scaling: true,
            chroma_fraction: 0.5,
            seed: 42,
            clip_output: true,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Luminance helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Per-pixel grain scale factor based on luminance.
///
/// Returns a parabola peaking at `l = 0.5`:
/// `grain_scale_fn(l) = 4 * l * (1 - l)`
///
/// This gives 0 at pure black and pure white, and a maximum of 1 at 50 % grey,
/// so midtone regions accumulate the most visible grain — matching the
/// photographic characteristic of traditional film stocks.
#[inline]
pub fn grain_scale_fn(luminance: f32) -> f32 {
    let l = luminance.clamp(0.0, 1.0);
    4.0 * l * (1.0 - l)
}

/// BT.709 luminance from linear-light RGB.
///
/// `Y = 0.2126 * r + 0.7152 * g + 0.0722 * b`
///
/// Named `film_luminance` to avoid collisions with `color_grading::luminance`
/// at the crate root.
#[inline]
pub fn film_luminance(r: f32, g: f32, b: f32) -> f32 {
    0.2126 * r + 0.7152 * g + 0.0722 * b
}

// ─────────────────────────────────────────────────────────────────────────────
// Grain map generation
// ─────────────────────────────────────────────────────────────────────────────

/// Bilinearly upsample a `src_w × src_h` float buffer to `dst_w × dst_h`.
///
/// `src` must have length `src_w * src_h`.
/// Returns a `Vec<f32>` of length `dst_w * dst_h`.
///
/// When `preserve_variance` is `true`, each destination sample is
/// additionally divided by `sqrt(w00² + w10² + w01² + w11²)`, the L2 norm
/// of its four bilinear weights. Bilinear interpolation is a convex
/// combination of the four source samples, so if those samples are
/// independent with unit variance (as [`generate_grain_map`]'s N(0,1)
/// noise is), the interpolated value's variance is `Σwᵢ²` - equal to `1`
/// only exactly on a source grid point, and strictly less than `1`
/// everywhere else - so naive upsampled noise gets systematically quieter
/// the further a destination pixel sits from its nearest source sample
/// (worse as `grain_size` grows and destination pixels spend more of their
/// area away from grid points). Dividing by `sqrt(Σwᵢ²)` restores unit
/// variance at every destination pixel while preserving the intended
/// spatial correlation between neighbouring pixels. This correction would
/// be meaningless for ordinary (non-noise) image data, hence the flag
/// rather than always-on behaviour.
fn bilinear_upsample(
    src: &[f32],
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
    preserve_variance: bool,
) -> Vec<f32> {
    let mut dst = vec![0.0_f32; dst_w * dst_h];

    for dy in 0..dst_h {
        for dx in 0..dst_w {
            // Map destination pixel centre to source coordinates.
            let sx = dx as f32 * (src_w as f32 - 1.0) / (dst_w as f32 - 1.0).max(1.0);
            let sy = dy as f32 * (src_h as f32 - 1.0) / (dst_h as f32 - 1.0).max(1.0);

            let x0 = sx.floor() as usize;
            let y0 = sy.floor() as usize;
            let x1 = (x0 + 1).min(src_w.saturating_sub(1));
            let y1 = (y0 + 1).min(src_h.saturating_sub(1));
            let tx = sx - x0 as f32;
            let ty = sy - y0 as f32;

            let v00 = src[y0 * src_w + x0];
            let v10 = src[y0 * src_w + x1];
            let v01 = src[y1 * src_w + x0];
            let v11 = src[y1 * src_w + x1];

            let w00 = (1.0 - tx) * (1.0 - ty);
            let w10 = tx * (1.0 - ty);
            let w01 = (1.0 - tx) * ty;
            let w11 = tx * ty;

            let mut value = w00 * v00 + w10 * v10 + w01 * v01 + w11 * v11;
            if preserve_variance {
                let weight_sq_sum = w00 * w00 + w10 * w10 + w01 * w01 + w11 * w11;
                if weight_sq_sum > 0.0 {
                    value /= weight_sq_sum.sqrt();
                }
            }
            dst[dy * dst_w + dx] = value;
        }
    }

    dst
}

/// Generate a grayscale grain map for an image.
///
/// Returns a `Vec<f32>` of length `width * height`, containing N(0, 1)
/// samples (the caller scales by `intensity`).
///
/// If `grain_size == 1.0` the map is generated at full resolution (independent
/// per-pixel noise).  For `grain_size > 1.0` the map is generated at a reduced
/// resolution of `⌊width / grain_size⌋ × ⌊height / grain_size⌋` and then
/// bilinearly upsampled, producing spatially correlated (clumped) grain.
pub fn generate_grain_map(width: usize, height: usize, config: &FilmGrainConfig) -> Vec<f32> {
    let grain_w = ((width as f32 / config.grain_size).max(1.0)) as usize;
    let grain_h = ((height as f32 / config.grain_size).max(1.0)) as usize;

    let mut state = config.seed.wrapping_add(0x5851F42D4C957F2D);
    if state == 0 {
        state = 0x6C62272E07BB0142;
    }

    let n_grain = grain_w * grain_h;
    let mut grain_small = Vec::with_capacity(n_grain);
    for _ in 0..n_grain {
        grain_small.push(sample_normal(&mut state));
    }

    if grain_w == width && grain_h == height {
        grain_small
    } else {
        // `preserve_variance = true`: without it, the upsampled map's
        // *actual* standard deviation shrinks well below the intended
        // N(0,1) as `grain_size` grows, so `intensity` would silently stop
        // meaning "sigma of added noise" (see `bilinear_upsample`'s doc).
        bilinear_upsample(&grain_small, grain_w, grain_h, width, height, true)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers for chroma grain maps
// ─────────────────────────────────────────────────────────────────────────────

/// Derive a seed offset for chroma channels so they differ from the luma map
/// and from each other.
#[inline]
fn chroma_seed(base_seed: u64, channel_idx: u64) -> u64 {
    // Use a large prime multiplier (Fibonacci hashing constant) to scatter seeds.
    base_seed
        .wrapping_add(channel_idx.wrapping_mul(0x9E3779B97F4A7C15))
        .wrapping_add(0xD1B54A32D192ED03)
}

/// Derive an independent grain-stream seed from `base_seed` for a numbered
/// stream (`tag`).
///
/// Used to keep the shared ("base") grain map of a sequence and every
/// per-frame grain map on *distinct* streams. A plain `base_seed ^ (tag * K)`
/// derivation collides with `base_seed` itself at `tag == 0`, which made
/// frame 0's supposedly independent grain map bit-identical to the shared
/// base map: the temporal blend `tc*base + (1-tc)*frame` then degenerated to
/// `base / sqrt(tc² + (1-tc)²)`, inflating the measured grain sigma by up to
/// 41% at `tc = 0.5` instead of preserving `intensity`.
///
/// The splitmix64 finaliser below is a bijection on `u64`, and
/// `base_seed + tag * GOLDEN` is injective in `tag`, so distinct tags always
/// yield distinct (and well-scattered) seeds.
#[inline]
fn stream_seed(base_seed: u64, tag: u64) -> u64 {
    let mut z = base_seed.wrapping_add(tag.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Build a fake `FilmGrainConfig` with a different seed (used to generate
/// chroma grain maps with the same spatial scale but independent randomness).
#[inline]
fn config_with_seed(config: &FilmGrainConfig, seed: u64) -> FilmGrainConfig {
    FilmGrainConfig {
        seed,
        ..config.clone()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Core grain application — RGB
// ─────────────────────────────────────────────────────────────────────────────

/// Apply film grain to an RGB image.
///
/// `image` must be a flat row-major RGB buffer: `len == width * height * 3`,
/// with values in [0, 1].
///
/// # Algorithm
///
/// 1. Generate a luma grain map G_luma (N(0,1)) for the full image.
/// 2. If `chroma_fraction > 0`, generate three independent chroma grain maps
///    (one per RGB channel).
/// 3. For each pixel:
///    - Compute BT.709 luminance.
///    - Compute `sigma = intensity` (or `intensity * grain_scale_fn(lum)` when
///      `luminance_scaling` is enabled).
///    - Blend luma and chroma grain according to `chroma_fraction`.
///    - Add grain and optionally clamp to [0, 1].
///
/// # Errors
///
/// - [`FilmGrainError::EmptyImage`] if `width == 0` or `height == 0`.
/// - [`FilmGrainError::InvalidImage`] if `image.len() != width * height * 3`.
pub fn apply_film_grain(
    image: &[f32],
    width: usize,
    height: usize,
    config: &FilmGrainConfig,
) -> Result<Vec<f32>, FilmGrainError> {
    if width == 0 || height == 0 {
        return Err(FilmGrainError::EmptyImage);
    }
    let expected = width * height * 3;
    if image.len() != expected {
        return Err(FilmGrainError::InvalidImage(format!(
            "expected {} bytes for {}×{}×3, got {}",
            expected,
            width,
            height,
            image.len()
        )));
    }

    let n_pixels = width * height;

    // Luma grain map (used for monochromatic component).
    let g_luma = generate_grain_map(width, height, config);

    // Chroma grain maps: one per RGB channel, seeded differently.
    let use_chroma = config.chroma_fraction > 0.0;
    let g_r: Vec<f32>;
    let g_g: Vec<f32>;
    let g_b: Vec<f32>;

    if use_chroma {
        g_r = generate_grain_map(
            width,
            height,
            &config_with_seed(config, chroma_seed(config.seed, 1)),
        );
        g_g = generate_grain_map(
            width,
            height,
            &config_with_seed(config, chroma_seed(config.seed, 2)),
        );
        g_b = generate_grain_map(
            width,
            height,
            &config_with_seed(config, chroma_seed(config.seed, 3)),
        );
    } else {
        g_r = Vec::new();
        g_g = Vec::new();
        g_b = Vec::new();
    }

    // The luma/chroma blend `luma*lf + chroma*cf` combines two *independent*
    // unit-variance sources, so its variance is `lf² + cf²` (<= 1, and < 1
    // whenever both `lf` and `cf` are non-zero) rather than `1` - dividing
    // by `sqrt(lf² + cf²)` restores it, so `intensity` (the target sigma)
    // stays the actual standard deviation of the added noise regardless of
    // `chroma_fraction`. `lf² + cf² >= 0.5` for any `cf` in `[0, 1]`, so
    // this never divides by zero.
    let cf = config.chroma_fraction;
    let lf = 1.0 - cf;
    let chroma_norm = (lf * lf + cf * cf).sqrt();

    let mut out = Vec::with_capacity(expected);

    for pi in 0..n_pixels {
        let base = pi * 3;
        let r = image[base];
        let g = image[base + 1];
        let b = image[base + 2];

        let lum = film_luminance(r, g, b);
        let sigma = if config.luminance_scaling {
            config.intensity * grain_scale_fn(lum)
        } else {
            config.intensity
        };

        let luma_noise = g_luma[pi] * sigma;

        let (r_noise, g_noise, b_noise) = if use_chroma {
            (
                (luma_noise * lf + g_r[pi] * sigma * cf) / chroma_norm,
                (luma_noise * lf + g_g[pi] * sigma * cf) / chroma_norm,
                (luma_noise * lf + g_b[pi] * sigma * cf) / chroma_norm,
            )
        } else {
            (luma_noise, luma_noise, luma_noise)
        };

        let out_r = r + r_noise;
        let out_g = g + g_noise;
        let out_b = b + b_noise;

        if config.clip_output {
            out.push(out_r.clamp(0.0, 1.0));
            out.push(out_g.clamp(0.0, 1.0));
            out.push(out_b.clamp(0.0, 1.0));
        } else {
            out.push(out_r);
            out.push(out_g);
            out.push(out_b);
        }
    }

    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// RGBA variant
// ─────────────────────────────────────────────────────────────────────────────

/// Apply film grain to an RGBA image. The alpha channel is left unchanged.
///
/// `image` must be a flat row-major RGBA buffer: `len == width * height * 4`,
/// with values in [0, 1].
///
/// # Errors
///
/// - [`FilmGrainError::EmptyImage`] if `width == 0` or `height == 0`.
/// - [`FilmGrainError::InvalidImage`] if `image.len() != width * height * 4`.
pub fn apply_film_grain_rgba(
    image: &[f32],
    width: usize,
    height: usize,
    config: &FilmGrainConfig,
) -> Result<Vec<f32>, FilmGrainError> {
    if width == 0 || height == 0 {
        return Err(FilmGrainError::EmptyImage);
    }
    let expected = width * height * 4;
    if image.len() != expected {
        return Err(FilmGrainError::InvalidImage(format!(
            "expected {} bytes for {}×{}×4, got {}",
            expected,
            width,
            height,
            image.len()
        )));
    }

    // Extract RGB planes, apply grain, then re-interleave with alpha.
    let n_pixels = width * height;
    let mut rgb = Vec::with_capacity(n_pixels * 3);
    for pi in 0..n_pixels {
        let base = pi * 4;
        rgb.push(image[base]);
        rgb.push(image[base + 1]);
        rgb.push(image[base + 2]);
    }

    let grained_rgb = apply_film_grain(&rgb, width, height, config)?;

    let mut out = Vec::with_capacity(expected);
    for pi in 0..n_pixels {
        let rb = pi * 3;
        let ab = pi * 4;
        out.push(grained_rgb[rb]);
        out.push(grained_rgb[rb + 1]);
        out.push(grained_rgb[rb + 2]);
        out.push(image[ab + 3]);
    }

    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// Temporal grain (per-frame and sequences)
// ─────────────────────────────────────────────────────────────────────────────

/// Stream tag of the shared ("base") grain map of a sequence.
///
/// Frames use tags `frame_idx + 1`, so no frame can ever land on the base
/// map's stream — see [`stream_seed`].
const BASE_STREAM_TAG: u64 = 0;

/// Stream tag of frame `frame_idx` (see [`BASE_STREAM_TAG`]).
#[inline]
fn frame_stream_tag(frame_idx: usize) -> u64 {
    (frame_idx as u64).wrapping_add(1)
}

/// Apply grain to a specific frame using a frame-unique but reproducible seed.
///
/// The per-frame seed is `stream_seed(config.seed, frame_idx + 1)`, ensuring
/// each frame has independent grain while remaining fully deterministic, and
/// that no frame shares the stream of the shared base map used by
/// [`apply_film_grain_sequence`] (stream `0`).
///
/// # Errors
///
/// Propagates errors from [`apply_film_grain`].
pub fn apply_film_grain_frame(
    image: &[f32],
    width: usize,
    height: usize,
    frame_idx: usize,
    config: &FilmGrainConfig,
) -> Result<Vec<f32>, FilmGrainError> {
    let frame_seed = stream_seed(config.seed, frame_stream_tag(frame_idx));
    let frame_config = config_with_seed(config, frame_seed);
    apply_film_grain(image, width, height, &frame_config)
}

/// Apply temporally coherent film grain to a sequence of RGB images.
///
/// `images` is a slice of flat RGB buffers, each of length `width * height * 3`.
///
/// `temporal_coherence` in [0, 1]:
/// - `1.0` = all frames share the same grain map (perfectly correlated).
/// - `0.0` = each frame has fully independent grain (uncorrelated).
/// - Intermediate values blend a shared base grain with per-frame grain.
///
/// # Errors
///
/// - [`FilmGrainError::InvalidConfig`] if `temporal_coherence` is outside [0, 1].
/// - Propagates errors from [`apply_film_grain`] for each frame.
pub fn apply_film_grain_sequence(
    images: &[Vec<f32>],
    width: usize,
    height: usize,
    config: &FilmGrainConfig,
    temporal_coherence: f32,
) -> Result<Vec<Vec<f32>>, FilmGrainError> {
    if !(0.0..=1.0).contains(&temporal_coherence) {
        return Err(FilmGrainError::InvalidConfig(format!(
            "temporal_coherence must be in [0, 1], got {}",
            temporal_coherence
        )));
    }
    if images.is_empty() {
        return Ok(Vec::new());
    }

    // Shared (base) grain map — same for all frames. It lives on its own
    // stream (`BASE_STREAM_TAG`), disjoint from every per-frame stream, so
    // the temporal blend below always combines two *independent*
    // unit-variance maps and its renormalisation is exact.
    let base_seed = stream_seed(config.seed, BASE_STREAM_TAG);
    let base_cfg = config_with_seed(config, base_seed);
    let base_grain = generate_grain_map(width, height, &base_cfg);

    // Chroma base grain maps.
    let use_chroma = config.chroma_fraction > 0.0;
    let base_gr = if use_chroma {
        generate_grain_map(
            width,
            height,
            &config_with_seed(config, chroma_seed(base_seed, 1)),
        )
    } else {
        Vec::new()
    };
    let base_gg = if use_chroma {
        generate_grain_map(
            width,
            height,
            &config_with_seed(config, chroma_seed(base_seed, 2)),
        )
    } else {
        Vec::new()
    };
    let base_gb = if use_chroma {
        generate_grain_map(
            width,
            height,
            &config_with_seed(config, chroma_seed(base_seed, 3)),
        )
    } else {
        Vec::new()
    };

    images
        .iter()
        .enumerate()
        .map(|(frame_idx, image)| {
            if width == 0 || height == 0 {
                return Err(FilmGrainError::EmptyImage);
            }
            let expected = width * height * 3;
            if image.len() != expected {
                return Err(FilmGrainError::InvalidImage(format!(
                    "frame {}: expected {} bytes for {}×{}×3, got {}",
                    frame_idx,
                    expected,
                    width,
                    height,
                    image.len()
                )));
            }

            let n_pixels = width * height;

            // Per-frame grain map (independent randomness). Frame `i` uses
            // stream `i + 1`, so it can never coincide with the shared base
            // map's stream (`BASE_STREAM_TAG`) — the seed derivation this
            // replaced (`config.seed ^ (frame_idx * FRAME_SEED_MUL)`) gave
            // frame 0 the base map's own seed, i.e. the *same* map, which
            // turned the temporal blend into a scaled copy of the base grain
            // (sigma inflated by `1 / sqrt(tc² + (1-tc)²)`, up to +41%).
            let frame_seed = stream_seed(config.seed, frame_stream_tag(frame_idx));
            let frame_cfg = config_with_seed(config, frame_seed);
            let frame_grain = generate_grain_map(width, height, &frame_cfg);

            let frame_gr = if use_chroma {
                generate_grain_map(
                    width,
                    height,
                    &config_with_seed(&frame_cfg, chroma_seed(frame_seed, 1)),
                )
            } else {
                Vec::new()
            };
            let frame_gg = if use_chroma {
                generate_grain_map(
                    width,
                    height,
                    &config_with_seed(&frame_cfg, chroma_seed(frame_seed, 2)),
                )
            } else {
                Vec::new()
            };
            let frame_gb = if use_chroma {
                generate_grain_map(
                    width,
                    height,
                    &config_with_seed(&frame_cfg, chroma_seed(frame_seed, 3)),
                )
            } else {
                Vec::new()
            };

            let tc = temporal_coherence;
            let fi = 1.0 - tc;
            // Renormalise the base/frame blend the same way as the
            // chroma blend below: `tc*base + fi*frame` combines two
            // independent unit-variance sources, so its variance is
            // `tc² + fi²` (<= 1) rather than `1`. `tc² + fi² >= 0.5` for
            // any `tc` in `[0, 1]`, so this never divides by zero.
            let temporal_norm = (tc * tc + fi * fi).sqrt();
            let cf = config.chroma_fraction;
            let lf = 1.0 - cf;
            let chroma_norm = (lf * lf + cf * cf).sqrt();

            let mut out = Vec::with_capacity(expected);

            for pi in 0..n_pixels {
                let base = pi * 3;
                let r = image[base];
                let g = image[base + 1];
                let b = image[base + 2];

                let lum = film_luminance(r, g, b);
                let sigma = if config.luminance_scaling {
                    config.intensity * grain_scale_fn(lum)
                } else {
                    config.intensity
                };

                // Blend base and frame grain maps, renormalised to
                // preserve unit variance (see `temporal_norm` above).
                let g_luma = (tc * base_grain[pi] + fi * frame_grain[pi]) / temporal_norm;
                let luma_noise = g_luma * sigma;

                let (r_noise, g_noise, b_noise) = if use_chroma {
                    let gr = (tc * base_gr[pi] + fi * frame_gr[pi]) / temporal_norm;
                    let gg = (tc * base_gg[pi] + fi * frame_gg[pi]) / temporal_norm;
                    let gb = (tc * base_gb[pi] + fi * frame_gb[pi]) / temporal_norm;
                    (
                        (luma_noise * lf + gr * sigma * cf) / chroma_norm,
                        (luma_noise * lf + gg * sigma * cf) / chroma_norm,
                        (luma_noise * lf + gb * sigma * cf) / chroma_norm,
                    )
                } else {
                    (luma_noise, luma_noise, luma_noise)
                };

                let out_r = r + r_noise;
                let out_g = g + g_noise;
                let out_b = b + b_noise;

                if config.clip_output {
                    out.push(out_r.clamp(0.0, 1.0));
                    out.push(out_g.clamp(0.0, 1.0));
                    out.push(out_b.clamp(0.0, 1.0));
                } else {
                    out.push(out_r);
                    out.push(out_g);
                    out.push(out_b);
                }
            }

            Ok(out)
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Grain analysis
// ─────────────────────────────────────────────────────────────────────────────

/// Statistics about the grain added to an image.
#[derive(Debug, Clone)]
pub struct GrainStats {
    /// Mean absolute grain value added across all pixels and channels.
    pub mean_grain: f32,
    /// Standard deviation of grain values.
    pub std_grain: f32,
    /// Signal-to-noise ratio: `mean_lum_before / (std_grain + 1e-8)`.
    pub snr: f32,
    /// Mean BT.709 luminance of the original image.
    pub mean_lum_before: f32,
    /// Mean BT.709 luminance of the grained image.
    pub mean_lum_after: f32,
}

/// Compute grain statistics by comparing the original and grained images.
///
/// Both buffers must be flat RGB (`len == width * height * 3`).
///
/// # Errors
///
/// - [`FilmGrainError::EmptyImage`] if `width == 0` or `height == 0`.
/// - [`FilmGrainError::InvalidImage`] if either buffer has the wrong length.
pub fn compute_grain_stats(
    original: &[f32],
    grained: &[f32],
    width: usize,
    height: usize,
) -> Result<GrainStats, FilmGrainError> {
    if width == 0 || height == 0 {
        return Err(FilmGrainError::EmptyImage);
    }
    let expected = width * height * 3;
    if original.len() != expected {
        return Err(FilmGrainError::InvalidImage(format!(
            "original: expected {}, got {}",
            expected,
            original.len()
        )));
    }
    if grained.len() != expected {
        return Err(FilmGrainError::InvalidImage(format!(
            "grained: expected {}, got {}",
            expected,
            grained.len()
        )));
    }

    let n = expected as f32;
    let n_pixels = (width * height) as f32;

    // Compute grain = grained - original per element.
    let mut sum_abs_grain = 0.0_f64;
    let mut sum_grain = 0.0_f64;
    let mut sum_grain_sq = 0.0_f64;
    let mut sum_lum_before = 0.0_f64;
    let mut sum_lum_after = 0.0_f64;

    let n_pixels_usize = width * height;
    for pi in 0..n_pixels_usize {
        let base = pi * 3;
        let gr_elem = [
            grained[base] - original[base],
            grained[base + 1] - original[base + 1],
            grained[base + 2] - original[base + 2],
        ];
        for &g in &gr_elem {
            sum_abs_grain += g.abs() as f64;
            sum_grain += g as f64;
            sum_grain_sq += (g * g) as f64;
        }

        let or_ = original[base];
        let og = original[base + 1];
        let ob = original[base + 2];
        sum_lum_before += film_luminance(or_, og, ob) as f64;

        let gr = grained[base];
        let gg = grained[base + 1];
        let gb = grained[base + 2];
        sum_lum_after += film_luminance(gr, gg, gb) as f64;
    }

    let mean_grain = (sum_abs_grain / n as f64) as f32;
    let mean_g = (sum_grain / n as f64) as f32;
    let variance = ((sum_grain_sq / n as f64) - (mean_g * mean_g) as f64).max(0.0);
    let std_grain = variance.sqrt() as f32;
    let mean_lum_before = (sum_lum_before / n_pixels as f64) as f32;
    let mean_lum_after = (sum_lum_after / n_pixels as f64) as f32;
    let snr = mean_lum_before / (std_grain + 1e-8);

    Ok(GrainStats {
        mean_grain,
        std_grain,
        snr,
        mean_lum_before,
        mean_lum_after,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Tolerance helpers ────────────────────────────────────────────────────

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    // ── 1. grain_scale_fn at 0.0 → 0.0 ─────────────────────────────────────

    #[test]
    fn test_grain_scale_fn_zero() {
        assert!(approx_eq(grain_scale_fn(0.0), 0.0, 1e-6));
    }

    // ── 2. grain_scale_fn at 1.0 → 0.0 ─────────────────────────────────────

    #[test]
    fn test_grain_scale_fn_one() {
        assert!(approx_eq(grain_scale_fn(1.0), 0.0, 1e-6));
    }

    // ── 3. grain_scale_fn at 0.5 → 1.0 ─────────────────────────────────────

    #[test]
    fn test_grain_scale_fn_midpoint() {
        assert!(approx_eq(grain_scale_fn(0.5), 1.0, 1e-6));
    }

    // ── 4. grain_scale_fn clamping ───────────────────────────────────────────

    #[test]
    fn test_grain_scale_fn_clamps_negative() {
        assert!(approx_eq(grain_scale_fn(-1.0), grain_scale_fn(0.0), 1e-6));
    }

    #[test]
    fn test_grain_scale_fn_clamps_above_one() {
        assert!(approx_eq(grain_scale_fn(2.0), grain_scale_fn(1.0), 1e-6));
    }

    // ── 5. film_luminance basic values ──────────────────────────────────────

    #[test]
    fn test_film_luminance_pure_red() {
        // R=1, G=0, B=0 → 0.2126
        assert!(approx_eq(film_luminance(1.0, 0.0, 0.0), 0.2126, 1e-4));
    }

    #[test]
    fn test_film_luminance_pure_green() {
        assert!(approx_eq(film_luminance(0.0, 1.0, 0.0), 0.7152, 1e-4));
    }

    #[test]
    fn test_film_luminance_pure_blue() {
        assert!(approx_eq(film_luminance(0.0, 0.0, 1.0), 0.0722, 1e-4));
    }

    #[test]
    fn test_film_luminance_white() {
        // R=1, G=1, B=1 → 1.0
        assert!(approx_eq(film_luminance(1.0, 1.0, 1.0), 1.0, 1e-4));
    }

    #[test]
    fn test_film_luminance_black() {
        assert!(approx_eq(film_luminance(0.0, 0.0, 0.0), 0.0, 1e-6));
    }

    // ── 6. FilmGrainConfig::validate valid/invalid ───────────────────────────

    #[test]
    fn test_config_validate_default_is_valid() {
        assert!(FilmGrainConfig::default().validate().is_ok());
    }

    #[test]
    fn test_config_validate_negative_intensity() {
        let cfg = FilmGrainConfig {
            intensity: -0.1,
            ..FilmGrainConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_validate_grain_size_below_one() {
        let cfg = FilmGrainConfig {
            grain_size: 0.5,
            ..FilmGrainConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_validate_chroma_fraction_above_one() {
        let cfg = FilmGrainConfig {
            chroma_fraction: 1.5,
            ..FilmGrainConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_validate_chroma_fraction_negative() {
        let cfg = FilmGrainConfig {
            chroma_fraction: -0.1,
            ..FilmGrainConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    // ── 7. FilmGrainConfig::cinematic has intensity > 0 ─────────────────────

    #[test]
    fn test_config_cinematic_has_positive_intensity() {
        assert!(FilmGrainConfig::cinematic().intensity > 0.0);
    }

    #[test]
    fn test_config_cinematic_is_valid() {
        assert!(FilmGrainConfig::cinematic().validate().is_ok());
    }

    #[test]
    fn test_config_fine_is_valid() {
        assert!(FilmGrainConfig::fine().validate().is_ok());
    }

    #[test]
    fn test_config_heavy_is_valid() {
        assert!(FilmGrainConfig::heavy().validate().is_ok());
    }

    // ── 8. generate_grain_map correct size ───────────────────────────────────

    #[test]
    fn test_generate_grain_map_correct_size() {
        let cfg = FilmGrainConfig::default();
        let map = generate_grain_map(16, 12, &cfg);
        assert_eq!(map.len(), 16 * 12);
    }

    #[test]
    fn test_generate_grain_map_coarse_correct_size() {
        let cfg = FilmGrainConfig {
            grain_size: 4.0,
            ..FilmGrainConfig::default()
        };
        let map = generate_grain_map(32, 24, &cfg);
        assert_eq!(map.len(), 32 * 24);
    }

    // ── 9. generate_grain_map different seeds → different results ────────────

    #[test]
    fn test_generate_grain_map_different_seeds() {
        let cfg1 = FilmGrainConfig {
            seed: 1,
            ..FilmGrainConfig::default()
        };
        let cfg2 = FilmGrainConfig {
            seed: 999,
            ..FilmGrainConfig::default()
        };
        let m1 = generate_grain_map(16, 16, &cfg1);
        let m2 = generate_grain_map(16, 16, &cfg2);
        let n_diff = m1.iter().zip(m2.iter()).filter(|(a, b)| a != b).count();
        assert!(n_diff > 0, "Maps with different seeds should differ");
    }

    // ── 10. generate_grain_map mean ≈ 0 ─────────────────────────────────────

    #[test]
    fn test_generate_grain_map_mean_near_zero() {
        let cfg = FilmGrainConfig::default();
        let map = generate_grain_map(64, 64, &cfg);
        let mean: f32 = map.iter().sum::<f32>() / map.len() as f32;
        // Mean of N(0,1) samples should be close to 0 for large sample sizes.
        assert!(mean.abs() < 0.2, "grain map mean too far from 0: {}", mean);
    }

    // ── 11. apply_film_grain empty image → error ─────────────────────────────

    #[test]
    fn test_apply_grain_empty_width() {
        let cfg = FilmGrainConfig::default();
        let result = apply_film_grain(&[], 0, 4, &cfg);
        assert!(matches!(result, Err(FilmGrainError::EmptyImage)));
    }

    #[test]
    fn test_apply_grain_empty_height() {
        let cfg = FilmGrainConfig::default();
        let result = apply_film_grain(&[], 4, 0, &cfg);
        assert!(matches!(result, Err(FilmGrainError::EmptyImage)));
    }

    // ── 12. apply_film_grain wrong length → error ────────────────────────────

    #[test]
    fn test_apply_grain_wrong_length() {
        let cfg = FilmGrainConfig::default();
        let image = vec![0.5_f32; 10]; // wrong length
        let result = apply_film_grain(&image, 4, 4, &cfg);
        assert!(matches!(result, Err(FilmGrainError::InvalidImage(_))));
    }

    // ── 13. apply_film_grain intensity=0 → no change ─────────────────────────

    #[test]
    fn test_apply_grain_zero_intensity_no_change() {
        let cfg = FilmGrainConfig {
            intensity: 0.0,
            ..FilmGrainConfig::default()
        };
        let image: Vec<f32> = (0..4 * 4 * 3).map(|i| i as f32 / 48.0).collect();
        let result = apply_film_grain(&image, 4, 4, &cfg).expect("should succeed");
        for (orig, out) in image.iter().zip(result.iter()) {
            assert!(
                approx_eq(*orig, *out, 1e-6) || out.clamp(0.0, 1.0) == *out,
                "zero intensity should not change values"
            );
        }
    }

    // ── 14. apply_film_grain adds noise (output != input for intensity>0) ────

    #[test]
    fn test_apply_grain_adds_noise() {
        let cfg = FilmGrainConfig {
            intensity: 0.2,
            luminance_scaling: false,
            ..FilmGrainConfig::default()
        };
        let image = vec![0.5_f32; 8 * 8 * 3];
        let result = apply_film_grain(&image, 8, 8, &cfg).expect("should succeed");
        let n_changed = image
            .iter()
            .zip(result.iter())
            .filter(|(a, b)| a != b)
            .count();
        assert!(n_changed > 0, "grain should change at least one pixel");
    }

    // ── 15. apply_film_grain with clip_output=true → all values in [0,1] ────

    #[test]
    fn test_apply_grain_clip_output_in_range() {
        let cfg = FilmGrainConfig {
            intensity: 0.5,
            clip_output: true,
            luminance_scaling: false,
            ..FilmGrainConfig::default()
        };
        // Start near edges to stress clipping.
        let image: Vec<f32> = (0..8 * 8 * 3)
            .map(|i| if i % 2 == 0 { 0.01 } else { 0.99 })
            .collect();
        let result = apply_film_grain(&image, 8, 8, &cfg).expect("should succeed");
        for &v in &result {
            assert!(
                (0.0..=1.0).contains(&v),
                "clipped output out of range: {}",
                v
            );
        }
    }

    // ── 16. apply_film_grain preserves image dimensions ──────────────────────

    #[test]
    fn test_apply_grain_preserves_dimensions() {
        let w = 10;
        let h = 7;
        let cfg = FilmGrainConfig::default();
        let image = vec![0.5_f32; w * h * 3];
        let result = apply_film_grain(&image, w, h, &cfg).expect("should succeed");
        assert_eq!(result.len(), w * h * 3);
    }

    // ── 17. apply_film_grain_rgba alpha channel unchanged ────────────────────

    #[test]
    fn test_apply_grain_rgba_alpha_unchanged() {
        let w = 4;
        let h = 4;
        let cfg = FilmGrainConfig {
            intensity: 0.2,
            ..FilmGrainConfig::default()
        };
        // Build RGBA with known alpha pattern.
        let mut image = vec![0.5_f32; w * h * 4];
        for pi in 0..(w * h) {
            image[pi * 4 + 3] = pi as f32 / (w * h) as f32;
        }
        let result = apply_film_grain_rgba(&image, w, h, &cfg).expect("should succeed");
        for pi in 0..(w * h) {
            let expected_alpha = pi as f32 / (w * h) as f32;
            assert!(
                approx_eq(result[pi * 4 + 3], expected_alpha, 1e-6),
                "alpha changed at pixel {}",
                pi
            );
        }
    }

    // ── 18. apply_film_grain_frame different frames → different grain ────────

    #[test]
    fn test_apply_grain_frame_different_frames() {
        let w = 8;
        let h = 8;
        let cfg = FilmGrainConfig {
            intensity: 0.1,
            luminance_scaling: false,
            chroma_fraction: 0.0,
            ..FilmGrainConfig::default()
        };
        let image = vec![0.5_f32; w * h * 3];
        let f0 = apply_film_grain_frame(&image, w, h, 0, &cfg).expect("frame 0");
        let f1 = apply_film_grain_frame(&image, w, h, 1, &cfg).expect("frame 1");
        let n_diff = f0.iter().zip(f1.iter()).filter(|(a, b)| a != b).count();
        assert!(n_diff > 0, "frames 0 and 1 should have different grain");
    }

    // ── 19. apply_film_grain_sequence correct number of output images ────────

    #[test]
    fn test_apply_grain_sequence_output_count() {
        let w = 4;
        let h = 4;
        let cfg = FilmGrainConfig::default();
        let frames: Vec<Vec<f32>> = (0..5).map(|_| vec![0.5_f32; w * h * 3]).collect();
        let result =
            apply_film_grain_sequence(&frames, w, h, &cfg, 0.5).expect("sequence should succeed");
        assert_eq!(result.len(), 5);
    }

    // ── 20. temporal_coherence=1.0: all frames same grain ────────────────────

    #[test]
    fn test_apply_grain_sequence_full_coherence_same_grain() {
        let w = 8;
        let h = 8;
        let cfg = FilmGrainConfig {
            intensity: 0.1,
            luminance_scaling: false,
            chroma_fraction: 0.0,
            ..FilmGrainConfig::default()
        };
        // All frames have the same input → with tc=1.0, all output frames should be identical.
        let frames: Vec<Vec<f32>> = (0..3).map(|_| vec![0.5_f32; w * h * 3]).collect();
        let result =
            apply_film_grain_sequence(&frames, w, h, &cfg, 1.0).expect("sequence should succeed");
        for i in 1..result.len() {
            let n_diff = result[0]
                .iter()
                .zip(result[i].iter())
                .filter(|(a, b)| a != b)
                .count();
            assert_eq!(
                n_diff, 0,
                "frame {} should be identical to frame 0 at tc=1.0",
                i
            );
        }
    }

    // ── 21. temporal_coherence=0.0: frames have different grain ─────────────

    #[test]
    fn test_apply_grain_sequence_no_coherence_different_grain() {
        let w = 8;
        let h = 8;
        let cfg = FilmGrainConfig {
            intensity: 0.2,
            luminance_scaling: false,
            chroma_fraction: 0.0,
            ..FilmGrainConfig::default()
        };
        let frames: Vec<Vec<f32>> = (0..3).map(|_| vec![0.5_f32; w * h * 3]).collect();
        let result =
            apply_film_grain_sequence(&frames, w, h, &cfg, 0.0).expect("sequence should succeed");
        let n_diff_01 = result[0]
            .iter()
            .zip(result[1].iter())
            .filter(|(a, b)| a != b)
            .count();
        assert!(n_diff_01 > 0, "frames 0 and 1 should differ at tc=0.0");
    }

    // ── 22. compute_grain_stats same image → mean_grain=0 ───────────────────

    #[test]
    fn test_grain_stats_same_image() {
        let w = 8;
        let h = 8;
        let image: Vec<f32> = (0..w * h * 3)
            .map(|i| (i as f32) / (w * h * 3) as f32)
            .collect();
        let stats = compute_grain_stats(&image, &image, w, h).expect("should succeed");
        assert!(approx_eq(stats.mean_grain, 0.0, 1e-6));
        // When grain is zero, SNR should be very large.
        assert!(stats.snr > 1.0);
    }

    // ── 23. compute_grain_stats with added grain has mean_grain > 0 ─────────

    #[test]
    fn test_grain_stats_with_grain() {
        let w = 8;
        let h = 8;
        let cfg = FilmGrainConfig {
            intensity: 0.1,
            luminance_scaling: false,
            clip_output: false,
            chroma_fraction: 0.0,
            ..FilmGrainConfig::default()
        };
        let original = vec![0.5_f32; w * h * 3];
        let grained = apply_film_grain(&original, w, h, &cfg).expect("should succeed");
        let stats = compute_grain_stats(&original, &grained, w, h).expect("should succeed");
        assert!(
            stats.mean_grain > 0.0,
            "grained image should have non-zero mean grain"
        );
    }

    // ── 24. apply_film_grain luminance_scaling=false vs true ────────────────

    #[test]
    fn test_luminance_scaling_affects_output() {
        let w = 8;
        let h = 8;
        let base_cfg = FilmGrainConfig {
            intensity: 0.1,
            chroma_fraction: 0.0,
            clip_output: false,
            ..FilmGrainConfig::default()
        };

        let cfg_scaled = FilmGrainConfig {
            luminance_scaling: true,
            ..base_cfg.clone()
        };
        let cfg_flat = FilmGrainConfig {
            luminance_scaling: false,
            ..base_cfg.clone()
        };

        // Non-uniform luminance image so the scaling curve has a real effect.
        let image: Vec<f32> = (0..w * h * 3)
            .map(|i| (i as f32) / (w * h * 3) as f32)
            .collect();

        let out_scaled = apply_film_grain(&image, w, h, &cfg_scaled).expect("scaled");
        let out_flat = apply_film_grain(&image, w, h, &cfg_flat).expect("flat");

        let n_diff = out_scaled
            .iter()
            .zip(out_flat.iter())
            .filter(|(a, b)| (*a - *b).abs() > 1e-7)
            .count();
        assert!(
            n_diff > 0,
            "luminance_scaling should produce different output"
        );
    }

    // ── 25. generate_grain_map grain_size=1 vs grain_size=3: coarser is smoother

    #[test]
    fn test_grain_size_smoothness() {
        let w = 32;
        let h = 32;
        let cfg_fine = FilmGrainConfig {
            grain_size: 1.0,
            ..FilmGrainConfig::default()
        };
        let cfg_coarse = FilmGrainConfig {
            grain_size: 3.0,
            ..FilmGrainConfig::default()
        };

        let fine_map = generate_grain_map(w, h, &cfg_fine);
        let coarse_map = generate_grain_map(w, h, &cfg_coarse);

        // Measure total variation (sum of |x[i] - x[i-1]|) — coarser grain should be lower.
        let tv_fine: f32 = fine_map.windows(2).map(|p| (p[1] - p[0]).abs()).sum();
        let tv_coarse: f32 = coarse_map.windows(2).map(|p| (p[1] - p[0]).abs()).sum();

        assert!(
            tv_coarse < tv_fine,
            "coarser grain should have lower total variation ({} >= {})",
            tv_coarse,
            tv_fine
        );
    }

    // ── 25b. measured grain std tracks `intensity` (variance-preservation) ──

    /// Sample standard deviation of `grained - original` (population, not
    /// Bessel-corrected - `n` here is large enough that the difference is
    /// negligible).
    fn measured_grain_std(original: &[f32], grained: &[f32]) -> f32 {
        let diffs: Vec<f32> = grained
            .iter()
            .zip(original.iter())
            .map(|(&g, &o)| g - o)
            .collect();
        let n = diffs.len() as f32;
        let mean: f32 = diffs.iter().sum::<f32>() / n;
        let variance: f32 = diffs.iter().map(|&d| (d - mean).powi(2)).sum::<f32>() / n;
        variance.sqrt()
    }

    #[test]
    fn test_grain_std_tracks_intensity_across_grain_size() {
        // Regression test: `intensity` is documented as "sigma of added
        // noise" (`FilmGrainConfig::intensity` doc), but bilinearly
        // upsampling independent N(0,1) source samples attenuates their
        // variance (a weighted average of independent samples has lower
        // variance than any individual sample) - worse as `grain_size`
        // grows, since more destination pixels sit further from an exact
        // source grid point. The measured std of the added noise must
        // track `intensity` regardless of `grain_size`.
        let w = 64;
        let h = 64;
        let intensity = 0.1_f32;

        for &grain_size in &[1.0_f32, 1.5, 2.0, 3.0, 4.0] {
            let cfg = FilmGrainConfig {
                intensity,
                grain_size,
                luminance_scaling: false,
                chroma_fraction: 0.0,
                clip_output: false, // do not distort the measured statistics
                ..FilmGrainConfig::default()
            };
            let original = vec![0.5_f32; w * h * 3];
            let grained = apply_film_grain(&original, w, h, &cfg).expect("grain ok");
            let std = measured_grain_std(&original, &grained);

            assert!(
                (std - intensity).abs() < intensity * 0.15,
                "grain_size={grain_size}: measured std {std} should track intensity \
                 {intensity} (within 15%), ratio = {}",
                std / intensity
            );
        }
    }

    #[test]
    fn test_grain_std_tracks_intensity_across_chroma_fraction() {
        // Regression test: `luma*lf + chroma*cf` combines two independent
        // unit-variance sources, so its variance is `lf² + cf²` (<= 1)
        // rather than `1` unless renormalised - the measured std of the
        // added noise must track `intensity` regardless of
        // `chroma_fraction`.
        let w = 64;
        let h = 64;
        let intensity = 0.1_f32;

        for &chroma_fraction in &[0.0_f32, 0.25, 0.5, 0.75, 1.0] {
            let cfg = FilmGrainConfig {
                intensity,
                grain_size: 1.0, // isolate the chroma-blend effect from upsampling
                luminance_scaling: false,
                chroma_fraction,
                clip_output: false,
                ..FilmGrainConfig::default()
            };
            let original = vec![0.5_f32; w * h * 3];
            let grained = apply_film_grain(&original, w, h, &cfg).expect("grain ok");
            let std = measured_grain_std(&original, &grained);

            assert!(
                (std - intensity).abs() < intensity * 0.15,
                "chroma_fraction={chroma_fraction}: measured std {std} should track \
                 intensity {intensity} (within 15%), ratio = {}",
                std / intensity
            );
        }
    }

    #[test]
    fn test_sequence_base_and_frame_grain_streams_are_independent() {
        // Regression test for the seed collision behind the temporal-blend
        // variance bug: the per-frame seed used to be
        // `config.seed ^ (frame_idx * FRAME_SEED_MUL)`, which for frame 0 is
        // `config.seed` — exactly the seed of the shared base grain map. The
        // two "independent" maps were then bit-identical, so
        // `tc*base + (1-tc)*frame` was just `base` divided by the
        // renormaliser (sigma inflated by up to 41%).
        //
        // With disjoint streams, rendering frame 0 with `tc = 1.0` (base map
        // only) and with `tc = 0.0` (frame map only) must give *different*
        // noise, and the two noise fields must be essentially uncorrelated.
        let w = 32;
        let h = 32;
        let cfg = FilmGrainConfig {
            intensity: 0.1,
            grain_size: 1.0,
            luminance_scaling: false,
            chroma_fraction: 0.0,
            clip_output: false,
            ..FilmGrainConfig::default()
        };
        let frames: Vec<Vec<f32>> = (0..2).map(|_| vec![0.5_f32; w * h * 3]).collect();

        let base_only = apply_film_grain_sequence(&frames, w, h, &cfg, 1.0).expect("tc=1 ok");
        let frame_only = apply_film_grain_sequence(&frames, w, h, &cfg, 0.0).expect("tc=0 ok");

        let noise_base: Vec<f32> = base_only[0]
            .iter()
            .zip(frames[0].iter())
            .map(|(&g, &o)| g - o)
            .collect();
        let noise_frame: Vec<f32> = frame_only[0]
            .iter()
            .zip(frames[0].iter())
            .map(|(&g, &o)| g - o)
            .collect();

        let identical = noise_base
            .iter()
            .zip(noise_frame.iter())
            .all(|(a, b)| (a - b).abs() < 1e-9);
        assert!(
            !identical,
            "frame 0's own grain map must not be the shared base grain map"
        );

        // Pearson correlation of the two noise fields (both zero-mean by
        // construction, but compute it properly anyway).
        let n = noise_base.len() as f32;
        let mean_a = noise_base.iter().sum::<f32>() / n;
        let mean_b = noise_frame.iter().sum::<f32>() / n;
        let mut cov = 0.0_f32;
        let mut var_a = 0.0_f32;
        let mut var_b = 0.0_f32;
        for (&a, &b) in noise_base.iter().zip(noise_frame.iter()) {
            let da = a - mean_a;
            let db = b - mean_b;
            cov += da * db;
            var_a += da * da;
            var_b += db * db;
        }
        let corr = cov / (var_a.sqrt() * var_b.sqrt() + 1e-12);
        assert!(
            corr.abs() < 0.1,
            "base and frame-0 grain must be uncorrelated, got r = {corr}"
        );
    }

    #[test]
    fn test_grain_std_tracks_intensity_across_temporal_coherence() {
        // Regression test: `tc*base + fi*frame` has the same
        // variance-attenuation issue as the chroma blend above, resolved
        // the same way. It also pins the base/frame *stream separation*: if
        // frame 0's map were the base map again (see
        // `test_sequence_base_and_frame_grain_streams_are_independent`), the
        // measured sigma at tc=0.25 would come out ~28% high.
        let w = 64;
        let h = 64;
        let intensity = 0.1_f32;

        for &tc in &[0.0_f32, 0.25, 0.5, 0.75, 1.0] {
            let cfg = FilmGrainConfig {
                intensity,
                grain_size: 1.0,
                luminance_scaling: false,
                chroma_fraction: 0.0,
                clip_output: false,
                ..FilmGrainConfig::default()
            };
            let frames: Vec<Vec<f32>> = (0..2).map(|_| vec![0.5_f32; w * h * 3]).collect();
            let result =
                apply_film_grain_sequence(&frames, w, h, &cfg, tc).expect("sequence grain ok");
            let std = measured_grain_std(&frames[0], &result[0]);

            assert!(
                (std - intensity).abs() < intensity * 0.15,
                "temporal_coherence={tc}: measured std {std} should track intensity \
                 {intensity} (within 15%), ratio = {}",
                std / intensity
            );
        }
    }

    // ── 26. apply_film_grain_rgba wrong length → error ───────────────────────

    #[test]
    fn test_apply_grain_rgba_wrong_length() {
        let cfg = FilmGrainConfig::default();
        let image = vec![0.5_f32; 10]; // wrong length for 4×4×4
        let result = apply_film_grain_rgba(&image, 4, 4, &cfg);
        assert!(matches!(result, Err(FilmGrainError::InvalidImage(_))));
    }

    // ── 27. apply_film_grain_sequence invalid temporal_coherence ────────────

    #[test]
    fn test_apply_grain_sequence_invalid_coherence() {
        let w = 4;
        let h = 4;
        let cfg = FilmGrainConfig::default();
        let frames = vec![vec![0.5_f32; w * h * 3]];
        let result = apply_film_grain_sequence(&frames, w, h, &cfg, 1.5);
        assert!(matches!(result, Err(FilmGrainError::InvalidConfig(_))));
    }

    // ── 28. apply_film_grain_sequence empty input → empty output ────────────

    #[test]
    fn test_apply_grain_sequence_empty_input() {
        let cfg = FilmGrainConfig::default();
        let result =
            apply_film_grain_sequence(&[], 4, 4, &cfg, 0.5).expect("empty sequence should succeed");
        assert!(result.is_empty());
    }
}
