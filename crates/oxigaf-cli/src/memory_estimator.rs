//! Memory estimation for 3D Gaussian Splatting training and inference.
//!
//! This module provides utilities to estimate GPU/CPU memory requirements for
//! different 3DGS configurations before committing to a training run. Estimates
//! cover Gaussian parameter storage, render buffers, training overhead (optimizer
//! states, gradients, EMA), and diffusion model weights.
//!
//! # Example
//! ```rust
//! use oxigaf_cli::memory_estimator::{MemEstimateConfig, estimate_memory};
//!
//! let config = MemEstimateConfig {
//!     n_gaussians: 500_000,
//!     sh_degree: 3,
//!     render_width: 512,
//!     render_height: 512,
//!     ..Default::default()
//! };
//! let estimate = estimate_memory(&config).expect("estimation failed");
//! println!("Recommended VRAM: {:.2} GB", estimate.recommended_vram_gb);
//! ```

use thiserror::Error;

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

/// Errors that can occur during memory estimation.
#[derive(Debug, Error)]
pub enum MemEstimateError {
    /// A configuration parameter has an invalid value.
    #[error("Invalid parameter: {0}")]
    InvalidParam(String),

    /// The number of Gaussians is too large to address.
    #[error("Gaussian count overflow: {0} Gaussians would exceed addressable memory")]
    GaussianCountOverflow(usize),

    /// The render resolution exceeds the maximum supported limit.
    #[error("Resolution too large: {width}×{height} exceeds limits")]
    ResolutionTooLarge { width: usize, height: usize },
}

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// Per-Gaussian memory layout describing how much GPU memory each parameter
/// array occupies.
#[derive(Debug, Clone)]
pub struct GaussianLayout {
    /// Positions array: n × 3 × 4 bytes (f32 xyz).
    pub positions_bytes: usize,
    /// Rotations array: n × 4 × 4 bytes (quaternion f32 wxyz).
    pub rotations_bytes: usize,
    /// Scales array: n × 3 × 4 bytes (log-scale f32 xyz).
    pub scales_bytes: usize,
    /// Opacities array: n × 1 × 4 bytes (pre-sigmoid f32).
    pub opacities_bytes: usize,
    /// Spherical harmonics array: n × sh_coefficients × 3 color channels × 4 bytes (f32).
    pub sh_bytes: usize,
    /// Sum of all parameter arrays.
    pub total_bytes: usize,
    /// Number of Gaussians this layout describes.
    pub n_gaussians: usize,
    /// SH degree used.
    pub sh_degree: u32,
    /// Number of SH coefficients per Gaussian per channel.
    pub sh_coefficients: usize,
}

/// GPU render buffer memory requirements for a given resolution.
#[derive(Debug, Clone)]
pub struct RenderBuffers {
    /// RGBA f32 framebuffer: width × height × 4 channels × 4 bytes.
    pub framebuffer_bytes: usize,
    /// Depth buffer: width × height × 4 bytes (f32).
    pub depth_buffer_bytes: usize,
    /// Tile-based sorting buffer for tile rasterizer (capped at 256 MB).
    pub tile_buffer_bytes: usize,
    /// Projected 2D Gaussian descriptors: n × 32 bytes (8 f32 per Gaussian).
    pub gaussian_2d_bytes: usize,
    /// Sum of all render buffer allocations.
    pub total_bytes: usize,
    /// Render width in pixels.
    pub width: usize,
    /// Render height in pixels.
    pub height: usize,
}

/// Training-specific memory overhead on top of the Gaussian parameters.
#[derive(Debug, Clone)]
pub struct TrainingMemory {
    /// Adam optimizer state (m + v vectors): 2× or 1× parameter bytes.
    pub optimizer_bytes: usize,
    /// Gradient buffer: same size as parameter storage.
    pub gradient_bytes: usize,
    /// Exponential Moving Average copy of parameters (if enabled).
    pub ema_bytes: usize,
    /// Cached renders used during loss computation.
    pub loss_cache_bytes: usize,
    /// Sum of all training overhead allocations.
    pub total_bytes: usize,
}

/// Complete memory estimate for a given training configuration.
#[derive(Debug, Clone)]
pub struct MemoryEstimate {
    /// Gaussian parameter layout.
    pub gaussians: GaussianLayout,
    /// Render buffer requirements.
    pub render_buffers: RenderBuffers,
    /// Training overhead.
    pub training: TrainingMemory,
    /// Diffusion model weights (FP32).
    pub model_weights_bytes: usize,
    /// Total estimated GPU memory consumption in bytes.
    pub total_gpu_bytes: usize,
    /// Total estimated CPU/system memory consumption in bytes.
    pub total_cpu_bytes: usize,
    /// Suggested minimum VRAM in GiB (2^30 bytes; includes 20% overhead
    /// buffer). Uses the same binary-GiB convention as [`mem_format_bytes`],
    /// so it is directly comparable to `total_gpu_bytes` once formatted.
    pub recommended_vram_gb: f32,
    /// Suggested minimum system RAM in GiB (2^30 bytes), same convention as
    /// `recommended_vram_gb`.
    pub recommended_ram_gb: f32,
    /// Whether the total GPU footprint fits within the configured VRAM target.
    pub fits_in_vram: bool,
    /// The VRAM target against which `fits_in_vram` was evaluated.
    pub target_vram_bytes: usize,
}

/// Configuration for a memory estimation run.
#[derive(Debug, Clone)]
pub struct MemEstimateConfig {
    /// Number of 3D Gaussians in the scene.
    pub n_gaussians: usize,
    /// Spherical harmonics degree (0..=3 supported).
    pub sh_degree: u32,
    /// Render viewport width in pixels.
    pub render_width: usize,
    /// Render viewport height in pixels.
    pub render_height: usize,
    /// Whether an EMA copy of parameters is kept during training.
    pub use_ema: bool,
    /// Whether Adam optimizer state is tracked.
    pub use_optimizer: bool,
    /// Number of render views cached simultaneously for loss computation.
    pub n_render_views: usize,
    /// Number of diffusion model parameters (used to estimate model weights).
    pub diffusion_model_params: usize,
    /// Target GPU VRAM budget used to evaluate `fits_in_vram`.
    pub target_vram_bytes: usize,
    /// Use FP16 for optimizer states (halves optimizer memory).
    pub fp16_optimizer: bool,
}

impl Default for MemEstimateConfig {
    fn default() -> Self {
        Self {
            n_gaussians: 100_000_usize,
            sh_degree: 3,
            render_width: 512,
            render_height: 512,
            use_ema: true,
            use_optimizer: true,
            n_render_views: 4,
            diffusion_model_params: 1_000_000_000_usize,
            target_vram_bytes: 8_usize * 1024 * 1024 * 1024,
            fp16_optimizer: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: SH coefficient count
// ---------------------------------------------------------------------------

/// Compute the number of spherical harmonics coefficients for a given degree.
///
/// The formula is `(degree + 1)^2`. For degree 0 this gives 1 (DC only);
/// for degree 3 (the standard 3DGS setting) this gives 16.
///
/// # Examples
/// ```
/// use oxigaf_cli::memory_estimator::sh_coefficients;
/// assert_eq!(sh_coefficients(0), 1);
/// assert_eq!(sh_coefficients(3), 16);
/// ```
pub fn sh_coefficients(degree: u32) -> usize {
    let d = degree as usize + 1;
    d * d
}

// ---------------------------------------------------------------------------
// Maximum supported values
// ---------------------------------------------------------------------------

/// Maximum SH degree accepted by the estimator (degree 3 = 16 coefficients,
/// the standard maximum used in 3DGS literature).
const MAX_SH_DEGREE: u32 = 3;

/// Maximum render dimension in any single axis (16 384 pixels).
const MAX_RENDER_DIM: usize = 16_384;

/// Maximum tile buffer size: 256 MB.
const MAX_TILE_BUFFER_BYTES: usize = 256 * 1024 * 1024;

/// Tile size (in pixels, per axis) used by the tile-based rasterizer.
///
/// Must match `oxigaf_render::config::RasterConfig`'s default `tile_size`
/// of 16 (see `crates/oxigaf-render/src/config.rs`, used by `tiles_x`/
/// `tiles_y`). oxigaf-cli does not depend on oxigaf-render, so the value
/// cannot be imported directly; a previous 64px assumption here understated
/// `tile_buffer_bytes` by up to 16x. Kept as a single named constant so a
/// future correction only needs to happen in one place.
const RASTER_TILE_SIZE_PX: usize = 16;

/// Bytes per gibibyte (2^30), used for VRAM/RAM recommendations so they use
/// the same binary-unit convention as [`mem_format_bytes`] (whose "GB"
/// label is actually GiB-scaled). Kept distinct from a decimal-SI
/// `1_000_000_000.0` divisor, which previously put `recommended_vram_gb`/
/// `recommended_ram_gb` on a different scale than every other size printed
/// by [`format_memory_estimate`].
const BYTES_PER_GIB: f64 = 1024.0 * 1024.0 * 1024.0;

// ---------------------------------------------------------------------------
// estimate_gaussian_layout
// ---------------------------------------------------------------------------

/// Estimate the GPU memory occupied by Gaussian parameters.
///
/// Returns a [`GaussianLayout`] describing each array's size and the total.
/// Returns [`MemEstimateError::InvalidParam`] if `sh_degree > 3`.
/// Returns [`MemEstimateError::GaussianCountOverflow`] if the total would
/// overflow `usize`.
pub fn estimate_gaussian_layout(
    n_gaussians: usize,
    sh_degree: u32,
) -> Result<GaussianLayout, MemEstimateError> {
    if sh_degree > MAX_SH_DEGREE {
        return Err(MemEstimateError::InvalidParam(format!(
            "sh_degree {sh_degree} exceeds maximum supported degree {MAX_SH_DEGREE}"
        )));
    }

    let sh_coeffs = sh_coefficients(sh_degree);

    // Saturating per-Gaussian byte sizes (overflow check via checked_mul)
    let positions_bytes = n_gaussians
        .checked_mul(3)
        .and_then(|x| x.checked_mul(4))
        .ok_or(MemEstimateError::GaussianCountOverflow(n_gaussians))?;

    let rotations_bytes = n_gaussians
        .checked_mul(4)
        .and_then(|x| x.checked_mul(4))
        .ok_or(MemEstimateError::GaussianCountOverflow(n_gaussians))?;

    let scales_bytes = n_gaussians
        .checked_mul(3)
        .and_then(|x| x.checked_mul(4))
        .ok_or(MemEstimateError::GaussianCountOverflow(n_gaussians))?;

    let opacities_bytes = n_gaussians
        .checked_mul(4)
        .ok_or(MemEstimateError::GaussianCountOverflow(n_gaussians))?;

    // A 3DGS Gaussian stores `sh_coeffs` coefficients *per color channel*
    // (RGB), not `sh_coeffs` total -- see oxigaf-render's `gaussian.rs`
    // (`sh_coeffs` layout documented as [N, C] with C = (degree+1)^2 * 3)
    // and `export.rs` (`(sh_degree+1).pow(2) * 3`). Omitting the `* 3` here
    // previously under-counted SH storage by 3x (e.g. 64 B/Gaussian instead
    // of 192 B at the standard degree 3), which cascaded into every
    // estimate derived from `total_bytes` below (training memory,
    // `fits_in_vram`, `recommended_vram_gb`, `max_gaussians_for_vram`).
    let sh_bytes = n_gaussians
        .checked_mul(sh_coeffs)
        .and_then(|x| x.checked_mul(3))
        .and_then(|x| x.checked_mul(4))
        .ok_or(MemEstimateError::GaussianCountOverflow(n_gaussians))?;

    let total_bytes = positions_bytes
        .checked_add(rotations_bytes)
        .and_then(|x| x.checked_add(scales_bytes))
        .and_then(|x| x.checked_add(opacities_bytes))
        .and_then(|x| x.checked_add(sh_bytes))
        .ok_or(MemEstimateError::GaussianCountOverflow(n_gaussians))?;

    Ok(GaussianLayout {
        positions_bytes,
        rotations_bytes,
        scales_bytes,
        opacities_bytes,
        sh_bytes,
        total_bytes,
        n_gaussians,
        sh_degree,
        sh_coefficients: sh_coeffs,
    })
}

// ---------------------------------------------------------------------------
// estimate_render_buffers
// ---------------------------------------------------------------------------

/// Estimate GPU memory for render buffers at the given resolution.
///
/// Returns [`MemEstimateError::ResolutionTooLarge`] when either dimension
/// exceeds `MAX_RENDER_DIM` (16 384).
pub fn estimate_render_buffers(
    width: usize,
    height: usize,
    n_gaussians: usize,
) -> Result<RenderBuffers, MemEstimateError> {
    if width > MAX_RENDER_DIM || height > MAX_RENDER_DIM {
        return Err(MemEstimateError::ResolutionTooLarge { width, height });
    }
    if width == 0 || height == 0 {
        return Err(MemEstimateError::InvalidParam(
            "render dimensions must be non-zero".to_string(),
        ));
    }

    let n_pixels = width
        .checked_mul(height)
        .ok_or(MemEstimateError::ResolutionTooLarge { width, height })?;

    // RGBA f32: 4 channels × 4 bytes
    let framebuffer_bytes = n_pixels
        .checked_mul(4)
        .and_then(|x| x.checked_mul(4))
        .ok_or(MemEstimateError::ResolutionTooLarge { width, height })?;

    // Depth f32: 1 channel × 4 bytes
    let depth_buffer_bytes = n_pixels
        .checked_mul(4)
        .ok_or(MemEstimateError::ResolutionTooLarge { width, height })?;

    // Tile-based sort buffer estimate
    // 16×16 pixel tiles (matches oxigaf-render's default `RasterConfig`);
    // up to min(n_gaussians, 1024) entries per tile × 8 bytes
    let tile_w = width.div_ceil(RASTER_TILE_SIZE_PX);
    let tile_h = height.div_ceil(RASTER_TILE_SIZE_PX);
    let n_tiles = tile_w.saturating_mul(tile_h);
    let per_tile_entries = n_gaussians.min(1024);
    let tile_buffer_bytes = n_tiles
        .saturating_mul(per_tile_entries)
        .saturating_mul(8)
        .min(MAX_TILE_BUFFER_BYTES);

    // Projected 2D Gaussian descriptors: 8 f32 = 32 bytes each
    let gaussian_2d_bytes = n_gaussians.saturating_mul(32);

    let total_bytes = framebuffer_bytes
        .saturating_add(depth_buffer_bytes)
        .saturating_add(tile_buffer_bytes)
        .saturating_add(gaussian_2d_bytes);

    Ok(RenderBuffers {
        framebuffer_bytes,
        depth_buffer_bytes,
        tile_buffer_bytes,
        gaussian_2d_bytes,
        total_bytes,
        width,
        height,
    })
}

// ---------------------------------------------------------------------------
// estimate_training_memory
// ---------------------------------------------------------------------------

/// Estimate additional GPU memory consumed by the training loop.
///
/// The returned [`TrainingMemory`] accounts for gradient buffers, Adam
/// optimizer states (m and v vectors), an optional EMA copy of parameters,
/// and cached render outputs used when computing the loss.
pub fn estimate_training_memory(
    gaussian_layout: &GaussianLayout,
    n_render_views: usize,
    render_width: usize,
    render_height: usize,
    use_ema: bool,
    use_optimizer: bool,
    fp16_optimizer: bool,
) -> TrainingMemory {
    let param_bytes = gaussian_layout.total_bytes;

    // Gradient buffer mirrors the parameter buffer
    let gradient_bytes = param_bytes;

    // Adam: m + v state vectors (same dtype as params unless fp16)
    let optimizer_bytes = if use_optimizer {
        if fp16_optimizer {
            // Half-precision optimizer states: 2 × (param_bytes / 2) = param_bytes
            param_bytes
        } else {
            // Full-precision Adam: m vector + v vector = 2× parameters
            param_bytes.saturating_mul(2)
        }
    } else {
        0
    };

    // EMA is a full-precision copy of all parameters
    let ema_bytes = if use_ema { param_bytes } else { 0 };

    // Cached RGB renders: n_views × width × height × 3 channels × 4 bytes (f32)
    let loss_cache_bytes = n_render_views
        .saturating_mul(render_width)
        .saturating_mul(render_height)
        .saturating_mul(3)
        .saturating_mul(4);

    let total_bytes = gradient_bytes
        .saturating_add(optimizer_bytes)
        .saturating_add(ema_bytes)
        .saturating_add(loss_cache_bytes);

    TrainingMemory {
        optimizer_bytes,
        gradient_bytes,
        ema_bytes,
        loss_cache_bytes,
        total_bytes,
    }
}

// ---------------------------------------------------------------------------
// estimate_model_weights
// ---------------------------------------------------------------------------

/// Estimate diffusion model weight memory given parameter count and dtype size.
///
/// For FP32 use `dtype_bytes = 4`; for FP16/BF16 use `dtype_bytes = 2`.
pub fn estimate_model_weights(n_params: usize, dtype_bytes: usize) -> usize {
    n_params.saturating_mul(dtype_bytes)
}

// ---------------------------------------------------------------------------
// estimate_memory (full)
// ---------------------------------------------------------------------------

/// Produce a complete [`MemoryEstimate`] from a [`MemEstimateConfig`].
///
/// # Errors
/// Propagates errors from [`estimate_gaussian_layout`] and
/// [`estimate_render_buffers`].
pub fn estimate_memory(config: &MemEstimateConfig) -> Result<MemoryEstimate, MemEstimateError> {
    let gaussians = estimate_gaussian_layout(config.n_gaussians, config.sh_degree)?;

    let render_buffers = estimate_render_buffers(
        config.render_width,
        config.render_height,
        config.n_gaussians,
    )?;

    let training = estimate_training_memory(
        &gaussians,
        config.n_render_views,
        config.render_width,
        config.render_height,
        config.use_ema,
        config.use_optimizer,
        config.fp16_optimizer,
    );

    // Diffusion model weights in FP32
    let model_weights_bytes = estimate_model_weights(config.diffusion_model_params, 4);

    // GPU: parameters + render buffers + training overhead + model weights
    let total_gpu_bytes = gaussians
        .total_bytes
        .saturating_add(render_buffers.total_bytes)
        .saturating_add(training.total_bytes)
        .saturating_add(model_weights_bytes);

    // CPU: keep a host copy of Gaussian parameters for density control
    let total_cpu_bytes = gaussians.total_bytes;

    // 20% headroom buffer for VRAM recommendation. Divides by `BYTES_PER_GIB`
    // (2^30), not decimal 1e9, so this is on the same scale as
    // `mem_format_bytes`'s "GB" label used for `total_gpu_bytes` elsewhere
    // in the same report -- these two used to disagree by a factor of
    // ~1.074 (e.g. "Total GPU: 7.45 GB" next to "Recommended VRAM: 9.60 GB"
    // was not actually a clean 1.2x headroom).
    #[allow(clippy::cast_precision_loss)]
    let recommended_vram_gb = (total_gpu_bytes as f64 / BYTES_PER_GIB * 1.2) as f32;

    // 2× the CPU footprint for RAM recommendation (same GiB convention).
    #[allow(clippy::cast_precision_loss)]
    let recommended_ram_gb = (total_cpu_bytes as f64 / BYTES_PER_GIB * 2.0) as f32;

    let fits_in_vram = total_gpu_bytes <= config.target_vram_bytes;

    Ok(MemoryEstimate {
        gaussians,
        render_buffers,
        training,
        model_weights_bytes,
        total_gpu_bytes,
        total_cpu_bytes,
        recommended_vram_gb,
        recommended_ram_gb,
        fits_in_vram,
        target_vram_bytes: config.target_vram_bytes,
    })
}

// ---------------------------------------------------------------------------
// max_gaussians_for_vram
// ---------------------------------------------------------------------------

/// Find the maximum number of Gaussians that fit within the given VRAM budget.
///
/// Uses binary search over `[0, 100_000_000]`. Returns `0` when not even a
/// zero-Gaussian scene fits (e.g. model weights alone exceed VRAM).
///
/// # Errors
/// Returns [`MemEstimateError`] if configuration parameters are invalid.
pub fn max_gaussians_for_vram(
    target_vram_bytes: usize,
    sh_degree: u32,
    render_width: usize,
    render_height: usize,
    use_ema: bool,
    use_optimizer: bool,
    model_weights_bytes: usize,
) -> Result<usize, MemEstimateError> {
    // Validate static parameters early
    if sh_degree > MAX_SH_DEGREE {
        return Err(MemEstimateError::InvalidParam(format!(
            "sh_degree {sh_degree} exceeds maximum {MAX_SH_DEGREE}"
        )));
    }
    if render_width > MAX_RENDER_DIM || render_height > MAX_RENDER_DIM {
        return Err(MemEstimateError::ResolutionTooLarge {
            width: render_width,
            height: render_height,
        });
    }
    if render_width == 0 || render_height == 0 {
        return Err(MemEstimateError::InvalidParam(
            "render dimensions must be non-zero".to_string(),
        ));
    }

    // Compute total GPU bytes for a given Gaussian count, returning `None`
    // on overflow or configuration error.
    let gpu_bytes_for = |n: usize| -> Option<usize> {
        let layout = estimate_gaussian_layout(n, sh_degree).ok()?;
        let render = estimate_render_buffers(render_width, render_height, n).ok()?;
        let training = estimate_training_memory(
            &layout,
            4, // n_render_views (default 4)
            render_width,
            render_height,
            use_ema,
            use_optimizer,
            false, // fp16_optimizer
        );
        Some(
            layout
                .total_bytes
                .saturating_add(render.total_bytes)
                .saturating_add(training.total_bytes)
                .saturating_add(model_weights_bytes),
        )
    };

    // Check if n=0 already exceeds budget
    let base = gpu_bytes_for(0).unwrap_or(usize::MAX);
    if base > target_vram_bytes {
        return Ok(0);
    }

    let mut lo: usize = 0;
    let mut hi: usize = 100_000_000;

    while lo < hi {
        // Avoid mid = (lo + hi) / 2 overflow for large values
        let mid = lo + (hi - lo).div_ceil(2);
        let bytes = gpu_bytes_for(mid).unwrap_or(usize::MAX);
        if bytes <= target_vram_bytes {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }

    Ok(lo)
}

// ---------------------------------------------------------------------------
// memory_breakdown_percent
// ---------------------------------------------------------------------------

/// Memory breakdown as approximate percentages of total GPU usage.
#[derive(Debug, Clone)]
pub struct MemBreakdown {
    /// Percentage used by Gaussian parameter storage.
    pub gaussians_pct: f32,
    /// Percentage used by render buffers.
    pub render_pct: f32,
    /// Percentage used by training overhead.
    pub training_pct: f32,
    /// Percentage used by diffusion model weights.
    pub model_pct: f32,
}

/// Compute percentage breakdown of GPU memory usage across the four major
/// categories in a [`MemoryEstimate`].
///
/// When `total_gpu_bytes` is 0 all percentages are returned as 0.0.
pub fn memory_breakdown_percent(estimate: &MemoryEstimate) -> MemBreakdown {
    let total = estimate.total_gpu_bytes;
    if total == 0 {
        return MemBreakdown {
            gaussians_pct: 0.0,
            render_pct: 0.0,
            training_pct: 0.0,
            model_pct: 0.0,
        };
    }
    #[allow(clippy::cast_precision_loss)]
    let pct = |bytes: usize| -> f32 { bytes as f32 / total as f32 * 100.0 };

    MemBreakdown {
        gaussians_pct: pct(estimate.gaussians.total_bytes),
        render_pct: pct(estimate.render_buffers.total_bytes),
        training_pct: pct(estimate.training.total_bytes),
        model_pct: pct(estimate.model_weights_bytes),
    }
}

// ---------------------------------------------------------------------------
// mem_format_bytes
// ---------------------------------------------------------------------------

/// Format a byte count as a human-readable string.
///
/// - < 1 024 B    → `"N B"`
/// - < 1 MiB      → `"N.NN KB"`
/// - < 1 GiB      → `"N.NN MB"`
/// - >= 1 GiB     → `"N.NN GB"`
///
/// # Examples
/// ```
/// use oxigaf_cli::memory_estimator::mem_format_bytes;
/// assert_eq!(mem_format_bytes(512), "512 B");
/// assert_eq!(mem_format_bytes(1536), "1.50 KB");
/// ```
pub fn mem_format_bytes(bytes: usize) -> String {
    const KB: usize = 1024;
    const MB: usize = 1024 * 1024;
    const GB: usize = 1024 * 1024 * 1024;

    if bytes < KB {
        format!("{bytes} B")
    } else if bytes < MB {
        #[allow(clippy::cast_precision_loss)]
        let val = bytes as f64 / KB as f64;
        format!("{val:.2} KB")
    } else if bytes < GB {
        #[allow(clippy::cast_precision_loss)]
        let val = bytes as f64 / MB as f64;
        format!("{val:.2} MB")
    } else {
        #[allow(clippy::cast_precision_loss)]
        let val = bytes as f64 / GB as f64;
        format!("{val:.2} GB")
    }
}

// ---------------------------------------------------------------------------
// format_memory_estimate
// ---------------------------------------------------------------------------

/// Format a [`MemoryEstimate`] as a multi-line human-readable string.
pub fn format_memory_estimate(estimate: &MemoryEstimate) -> String {
    let breakdown = memory_breakdown_percent(estimate);
    let mut out = String::new();

    out.push_str("=== Memory Estimate ===\n");
    out.push_str(&format!(
        "  Gaussians:       {} ({:.1}%)\n",
        mem_format_bytes(estimate.gaussians.total_bytes),
        breakdown.gaussians_pct
    ));
    out.push_str(&format!(
        "    Positions:     {}\n",
        mem_format_bytes(estimate.gaussians.positions_bytes)
    ));
    out.push_str(&format!(
        "    Rotations:     {}\n",
        mem_format_bytes(estimate.gaussians.rotations_bytes)
    ));
    out.push_str(&format!(
        "    Scales:        {}\n",
        mem_format_bytes(estimate.gaussians.scales_bytes)
    ));
    out.push_str(&format!(
        "    Opacities:     {}\n",
        mem_format_bytes(estimate.gaussians.opacities_bytes)
    ));
    out.push_str(&format!(
        "    SH (deg {}):    {}\n",
        estimate.gaussians.sh_degree,
        mem_format_bytes(estimate.gaussians.sh_bytes)
    ));
    out.push_str(&format!(
        "  Render buffers:  {} ({:.1}%)\n",
        mem_format_bytes(estimate.render_buffers.total_bytes),
        breakdown.render_pct
    ));
    out.push_str(&format!(
        "  Training:        {} ({:.1}%)\n",
        mem_format_bytes(estimate.training.total_bytes),
        breakdown.training_pct
    ));
    out.push_str(&format!(
        "    Gradients:     {}\n",
        mem_format_bytes(estimate.training.gradient_bytes)
    ));
    out.push_str(&format!(
        "    Optimizer:     {}\n",
        mem_format_bytes(estimate.training.optimizer_bytes)
    ));
    out.push_str(&format!(
        "    EMA:           {}\n",
        mem_format_bytes(estimate.training.ema_bytes)
    ));
    out.push_str(&format!(
        "    Loss cache:    {}\n",
        mem_format_bytes(estimate.training.loss_cache_bytes)
    ));
    out.push_str(&format!(
        "  Model weights:   {} ({:.1}%)\n",
        mem_format_bytes(estimate.model_weights_bytes),
        breakdown.model_pct
    ));
    out.push_str(&format!(
        "  Total GPU:       {}\n",
        mem_format_bytes(estimate.total_gpu_bytes)
    ));
    out.push_str(&format!(
        "  Total CPU:       {}\n",
        mem_format_bytes(estimate.total_cpu_bytes)
    ));
    out.push_str(&format!(
        "  Recommended VRAM: {:.2} GB\n",
        estimate.recommended_vram_gb
    ));
    out.push_str(&format!(
        "  Recommended RAM:  {:.2} GB\n",
        estimate.recommended_ram_gb
    ));
    out.push_str(&format!(
        "  Fits in VRAM ({}): {}\n",
        mem_format_bytes(estimate.target_vram_bytes),
        if estimate.fits_in_vram { "YES" } else { "NO" }
    ));

    out
}

// ---------------------------------------------------------------------------
// compare_memory_configs
// ---------------------------------------------------------------------------

/// Delta between two memory configuration estimates.
#[derive(Debug, Clone)]
pub struct MemoryDelta {
    /// Total GPU bytes for configuration A.
    pub a_total_bytes: usize,
    /// Total GPU bytes for configuration B.
    pub b_total_bytes: usize,
    /// Signed difference: B − A (positive means B uses more memory).
    pub delta_bytes: i64,
    /// Percentage change from A to B.  0.0 when A is zero-sized.
    pub delta_pct: f32,
}

/// Compare the GPU memory requirements of two configurations.
///
/// `delta_bytes` and `delta_pct` are both expressed as B − A.
///
/// # Errors
/// Propagates any error from [`estimate_memory`].
pub fn compare_memory_configs(
    a: &MemEstimateConfig,
    b: &MemEstimateConfig,
) -> Result<MemoryDelta, MemEstimateError> {
    let a_est = estimate_memory(a)?;
    let b_est = estimate_memory(b)?;

    let a_total = a_est.total_gpu_bytes;
    let b_total = b_est.total_gpu_bytes;

    let delta_bytes = b_total as i64 - a_total as i64;

    #[allow(clippy::cast_precision_loss)]
    let delta_pct = if a_total == 0 {
        0.0_f32
    } else {
        delta_bytes as f32 / a_total as f32 * 100.0
    };

    Ok(MemoryDelta {
        a_total_bytes: a_total,
        b_total_bytes: b_total,
        delta_bytes,
        delta_pct,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // sh_coefficients
    // ------------------------------------------------------------------

    #[test]
    fn test_sh_coefficients_degree_0() {
        assert_eq!(sh_coefficients(0), 1);
    }

    #[test]
    fn test_sh_coefficients_degree_1() {
        assert_eq!(sh_coefficients(1), 4);
    }

    #[test]
    fn test_sh_coefficients_degree_2() {
        assert_eq!(sh_coefficients(2), 9);
    }

    #[test]
    fn test_sh_coefficients_degree_3() {
        assert_eq!(sh_coefficients(3), 16);
    }

    // ------------------------------------------------------------------
    // estimate_gaussian_layout — valid inputs
    // ------------------------------------------------------------------

    #[test]
    fn test_layout_100k_sh3() {
        let n = 100_000_usize;
        let layout = estimate_gaussian_layout(n, 3).expect("layout failed");
        assert_eq!(layout.n_gaussians, n);
        assert_eq!(layout.sh_degree, 3);
        assert_eq!(layout.sh_coefficients, 16);

        // positions: n × 3 × 4
        assert_eq!(layout.positions_bytes, n * 3 * 4);
        // rotations: n × 4 × 4
        assert_eq!(layout.rotations_bytes, n * 4 * 4);
        // scales: n × 3 × 4
        assert_eq!(layout.scales_bytes, n * 3 * 4);
        // opacities: n × 4
        assert_eq!(layout.opacities_bytes, n * 4);
        // sh: n × 16 coeffs × 3 color channels × 4 bytes
        assert_eq!(layout.sh_bytes, n * 16 * 3 * 4);

        let expected_total = layout.positions_bytes
            + layout.rotations_bytes
            + layout.scales_bytes
            + layout.opacities_bytes
            + layout.sh_bytes;
        assert_eq!(layout.total_bytes, expected_total);
    }

    #[test]
    fn test_layout_zero_gaussians() {
        let layout = estimate_gaussian_layout(0, 3).expect("zero gaussians failed");
        assert_eq!(layout.n_gaussians, 0);
        assert_eq!(layout.positions_bytes, 0);
        assert_eq!(layout.rotations_bytes, 0);
        assert_eq!(layout.scales_bytes, 0);
        assert_eq!(layout.opacities_bytes, 0);
        assert_eq!(layout.sh_bytes, 0);
        assert_eq!(layout.total_bytes, 0);
    }

    #[test]
    fn test_layout_sh_degree_0() {
        let layout = estimate_gaussian_layout(1000, 0).expect("sh0 layout failed");
        assert_eq!(layout.sh_coefficients, 1);
        assert_eq!(layout.sh_bytes, 1000 * 1 * 3 * 4);
    }

    #[test]
    fn test_layout_sh_degree_1() {
        let layout = estimate_gaussian_layout(1000, 1).expect("sh1 layout failed");
        assert_eq!(layout.sh_coefficients, 4);
        assert_eq!(layout.sh_bytes, 1000 * 4 * 3 * 4);
    }

    #[test]
    fn test_layout_sh_degree_2() {
        let layout = estimate_gaussian_layout(1000, 2).expect("sh2 layout failed");
        assert_eq!(layout.sh_coefficients, 9);
        assert_eq!(layout.sh_bytes, 1000 * 9 * 3 * 4);
    }

    #[test]
    fn test_layout_single_gaussian() {
        // Regression: pins the documented 236 B/Gaussian at the standard
        // degree 3 (12 + 16 + 12 + 4 + 192 = 236). `sh_bytes` used to omit
        // the "× 3 color channels" factor (16 coeffs × 4 bytes = 64 instead
        // of 16 × 3 × 4 = 192), under-reporting total_bytes as 108 instead
        // of 236 -- a ~2.2x error that propagated into every derived
        // training/VRAM estimate.
        let layout = estimate_gaussian_layout(1, 3).expect("single gaussian failed");
        assert_eq!(layout.positions_bytes, 12);
        assert_eq!(layout.rotations_bytes, 16);
        assert_eq!(layout.scales_bytes, 12);
        assert_eq!(layout.opacities_bytes, 4);
        assert_eq!(layout.sh_bytes, 16 * 3 * 4); // 16 coeffs × 3 channels × 4 bytes
        assert_eq!(layout.total_bytes, 12 + 16 + 12 + 4 + 192);
        assert_eq!(
            layout.total_bytes, 236,
            "must match the documented 236 B/Gaussian at degree 3"
        );
    }

    #[test]
    fn test_layout_total_is_sum_of_parts() {
        let layout = estimate_gaussian_layout(50_000, 2).expect("layout failed");
        let sum = layout.positions_bytes
            + layout.rotations_bytes
            + layout.scales_bytes
            + layout.opacities_bytes
            + layout.sh_bytes;
        assert_eq!(layout.total_bytes, sum);
    }

    // ------------------------------------------------------------------
    // estimate_gaussian_layout — error cases
    // ------------------------------------------------------------------

    #[test]
    fn test_layout_invalid_sh_degree() {
        let err = estimate_gaussian_layout(100, 4).expect_err("should fail for degree 4");
        assert!(matches!(err, MemEstimateError::InvalidParam(_)));
    }

    #[test]
    fn test_layout_invalid_sh_degree_10() {
        let err = estimate_gaussian_layout(100, 10).expect_err("should fail for degree 10");
        assert!(matches!(err, MemEstimateError::InvalidParam(_)));
    }

    // ------------------------------------------------------------------
    // estimate_render_buffers — valid inputs
    // ------------------------------------------------------------------

    #[test]
    fn test_render_buffers_512x512() {
        let rb = estimate_render_buffers(512, 512, 100_000).expect("render buffers failed");
        assert_eq!(rb.width, 512);
        assert_eq!(rb.height, 512);
        // framebuffer = 512×512×4×4 = 4 194 304
        assert_eq!(rb.framebuffer_bytes, 512 * 512 * 4 * 4);
        // depth = 512×512×4 = 1 048 576
        assert_eq!(rb.depth_buffer_bytes, 512 * 512 * 4);
        assert!(rb.total_bytes > rb.framebuffer_bytes);
    }

    #[test]
    fn test_render_buffers_1x1() {
        let rb = estimate_render_buffers(1, 1, 0).expect("1x1 render buffers failed");
        assert_eq!(rb.framebuffer_bytes, 4 * 4);
        assert_eq!(rb.depth_buffer_bytes, 4);
        assert_eq!(rb.gaussian_2d_bytes, 0);
        assert!(rb.total_bytes > 0);
    }

    #[test]
    fn test_render_buffers_tile_buffer_capped() {
        // Large n_gaussians should cap tile_buffer at 256 MB
        let rb = estimate_render_buffers(512, 512, 100_000_000).expect("large gaussian count");
        assert!(rb.tile_buffer_bytes <= MAX_TILE_BUFFER_BYTES);
    }

    #[test]
    fn test_render_buffers_tile_size_matches_16px_raster_default() {
        // Regression: this used to assume 64×64 pixel tiles, 16x fewer
        // than oxigaf-render's actual default `RasterConfig::tile_size` of
        // 16px, understating `tile_buffer_bytes` by up to 16x. At 512×512
        // with 16px tiles: 32×32 = 1024 tiles.
        let rb = estimate_render_buffers(512, 512, 2000).expect("512x512 buffers failed");
        let expected_tiles = 32 * 32;
        let expected_per_tile_entries = 2000usize.min(1024);
        let expected_tile_bytes =
            (expected_tiles * expected_per_tile_entries * 8).min(MAX_TILE_BUFFER_BYTES);
        assert_eq!(rb.tile_buffer_bytes, expected_tile_bytes);
    }

    #[test]
    fn test_render_buffers_1920x1080() {
        let rb = estimate_render_buffers(1920, 1080, 50_000).expect("1080p buffers failed");
        assert_eq!(rb.framebuffer_bytes, 1920 * 1080 * 16);
        assert_eq!(rb.depth_buffer_bytes, 1920 * 1080 * 4);
    }

    #[test]
    fn test_render_buffers_total_is_sum() {
        let rb = estimate_render_buffers(256, 256, 10_000).expect("render buffers failed");
        let sum = rb.framebuffer_bytes
            + rb.depth_buffer_bytes
            + rb.tile_buffer_bytes
            + rb.gaussian_2d_bytes;
        assert_eq!(rb.total_bytes, sum);
    }

    // ------------------------------------------------------------------
    // estimate_render_buffers — error cases
    // ------------------------------------------------------------------

    #[test]
    fn test_render_buffers_too_large() {
        let err =
            estimate_render_buffers(20_000, 512, 0).expect_err("should fail for oversized width");
        assert!(matches!(err, MemEstimateError::ResolutionTooLarge { .. }));
    }

    #[test]
    fn test_render_buffers_zero_width() {
        let err = estimate_render_buffers(0, 512, 0).expect_err("should fail for zero width");
        assert!(matches!(err, MemEstimateError::InvalidParam(_)));
    }

    #[test]
    fn test_render_buffers_zero_height() {
        let err = estimate_render_buffers(512, 0, 0).expect_err("should fail for zero height");
        assert!(matches!(err, MemEstimateError::InvalidParam(_)));
    }

    // ------------------------------------------------------------------
    // estimate_training_memory
    // ------------------------------------------------------------------

    #[test]
    fn test_training_memory_with_ema_and_optimizer() {
        let layout = estimate_gaussian_layout(100_000, 3).expect("layout failed");
        let training = estimate_training_memory(&layout, 4, 512, 512, true, true, false);

        assert_eq!(training.gradient_bytes, layout.total_bytes);
        // Full-precision Adam: 2× params
        assert_eq!(training.optimizer_bytes, layout.total_bytes * 2);
        // EMA enabled
        assert_eq!(training.ema_bytes, layout.total_bytes);
        // Loss cache: 4 × 512 × 512 × 3 × 4
        assert_eq!(training.loss_cache_bytes, 4 * 512 * 512 * 3 * 4);
    }

    #[test]
    fn test_training_memory_without_ema() {
        let layout = estimate_gaussian_layout(100_000, 3).expect("layout failed");
        let training = estimate_training_memory(&layout, 4, 512, 512, false, true, false);
        assert_eq!(training.ema_bytes, 0);
        assert!(
            training.total_bytes
                < training.gradient_bytes
                    + training.optimizer_bytes
                    + training.loss_cache_bytes
                    + layout.total_bytes
        );
    }

    #[test]
    fn test_training_memory_without_optimizer() {
        let layout = estimate_gaussian_layout(100_000, 3).expect("layout failed");
        let training = estimate_training_memory(&layout, 4, 512, 512, true, false, false);
        assert_eq!(training.optimizer_bytes, 0);
    }

    #[test]
    fn test_training_memory_fp16_optimizer() {
        let layout = estimate_gaussian_layout(100_000, 3).expect("layout failed");
        let fp16_training = estimate_training_memory(&layout, 4, 512, 512, true, true, true);
        let fp32_training = estimate_training_memory(&layout, 4, 512, 512, true, true, false);
        // FP16 optimizer should use less memory than FP32
        assert!(fp16_training.optimizer_bytes < fp32_training.optimizer_bytes);
    }

    #[test]
    fn test_training_memory_no_ema_no_optimizer() {
        let layout = estimate_gaussian_layout(50_000, 2).expect("layout failed");
        let training = estimate_training_memory(&layout, 0, 512, 512, false, false, false);
        assert_eq!(training.ema_bytes, 0);
        assert_eq!(training.optimizer_bytes, 0);
        assert_eq!(training.loss_cache_bytes, 0);
        assert_eq!(training.total_bytes, training.gradient_bytes);
    }

    #[test]
    fn test_training_memory_total_correctness() {
        let layout = estimate_gaussian_layout(10_000, 3).expect("layout failed");
        let training = estimate_training_memory(&layout, 2, 256, 256, true, true, false);
        let expected = training.gradient_bytes
            + training.optimizer_bytes
            + training.ema_bytes
            + training.loss_cache_bytes;
        assert_eq!(training.total_bytes, expected);
    }

    // ------------------------------------------------------------------
    // estimate_model_weights
    // ------------------------------------------------------------------

    #[test]
    fn test_model_weights_1b_fp32() {
        let bytes = estimate_model_weights(1_000_000_000, 4);
        assert_eq!(bytes, 4_000_000_000_usize);
    }

    #[test]
    fn test_model_weights_1b_fp16() {
        let bytes = estimate_model_weights(1_000_000_000, 2);
        assert_eq!(bytes, 2_000_000_000_usize);
    }

    #[test]
    fn test_model_weights_zero_params() {
        assert_eq!(estimate_model_weights(0, 4), 0);
    }

    #[test]
    fn test_model_weights_small() {
        assert_eq!(estimate_model_weights(100, 4), 400);
    }

    // ------------------------------------------------------------------
    // estimate_memory (full)
    // ------------------------------------------------------------------

    #[test]
    fn test_estimate_memory_default_config() {
        let config = MemEstimateConfig::default();
        let est = estimate_memory(&config).expect("default config estimation failed");
        // With 1B param model at FP32 = 4 GB, unlikely to fit in 8 GB total with 100k gaussians
        // Just verify the structure is coherent
        assert!(est.total_gpu_bytes > 0);
        assert!(est.recommended_vram_gb > 0.0);
        assert_eq!(est.target_vram_bytes, config.target_vram_bytes);
        // fits_in_vram must match
        assert_eq!(
            est.fits_in_vram,
            est.total_gpu_bytes <= config.target_vram_bytes
        );
    }

    #[test]
    fn test_estimate_memory_10m_gaussians() {
        let config = MemEstimateConfig {
            n_gaussians: 10_000_000,
            sh_degree: 3,
            target_vram_bytes: 8_usize * 1024 * 1024 * 1024,
            ..Default::default()
        };
        let est = estimate_memory(&config).expect("10M gaussians estimation failed");
        // 10M gaussians at sh3 is enormous — should not fit in 8GB with 1B model
        assert!(!est.fits_in_vram);
    }

    #[test]
    fn test_estimate_memory_small_config_no_model() {
        // Without a diffusion model, a small gaussian count should easily fit in 8GB
        let config = MemEstimateConfig {
            n_gaussians: 10_000,
            sh_degree: 0,
            render_width: 256,
            render_height: 256,
            use_ema: false,
            use_optimizer: false,
            n_render_views: 1,
            diffusion_model_params: 0,
            target_vram_bytes: 8_usize * 1024 * 1024 * 1024,
            fp16_optimizer: false,
        };
        let est = estimate_memory(&config).expect("small config failed");
        assert!(est.fits_in_vram);
    }

    #[test]
    fn test_estimate_memory_total_gpu_components() {
        let config = MemEstimateConfig {
            n_gaussians: 100_000,
            sh_degree: 3,
            diffusion_model_params: 0,
            ..Default::default()
        };
        let est = estimate_memory(&config).expect("estimation failed");
        let expected = est.gaussians.total_bytes
            + est.render_buffers.total_bytes
            + est.training.total_bytes
            + est.model_weights_bytes;
        assert_eq!(est.total_gpu_bytes, expected);
    }

    #[test]
    fn test_estimate_memory_cpu_is_gaussian_params() {
        let config = MemEstimateConfig::default();
        let est = estimate_memory(&config).expect("estimation failed");
        assert_eq!(est.total_cpu_bytes, est.gaussians.total_bytes);
    }

    #[test]
    fn test_estimate_memory_recommended_vram() {
        let config = MemEstimateConfig::default();
        let est = estimate_memory(&config).expect("estimation failed");
        // recommended_vram_gb = total_gpu / GiB (2^30) * 1.2 -- GiB, not the
        // decimal-SI 1e9 this used to divide by, so it is on the same scale
        // as `mem_format_bytes`'s "GB" label used elsewhere in the same
        // report.
        let expected_gib = est.total_gpu_bytes as f64 / BYTES_PER_GIB * 1.2;
        assert!((est.recommended_vram_gb as f64 - expected_gib).abs() < 0.01);

        // Regression guard: must actually differ from the old decimal-GB
        // formula (for any nonzero byte count, 1 GiB > 1 GB so the GiB
        // count is strictly smaller), so a future accidental revert to
        // `/ 1e9` is caught here rather than only in the value above.
        let expected_decimal_gb = est.total_gpu_bytes as f64 / 1e9 * 1.2;
        assert!(est.total_gpu_bytes > 0);
        assert!((est.recommended_vram_gb as f64 - expected_decimal_gb).abs() > 0.01);
    }

    // ------------------------------------------------------------------
    // max_gaussians_for_vram
    // ------------------------------------------------------------------

    #[test]
    fn test_max_gaussians_larger_vram_gives_more_gaussians() {
        let small = max_gaussians_for_vram(
            1024 * 1024 * 1024_usize, // 1 GB
            3,
            512,
            512,
            false,
            false,
            0,
        )
        .expect("1GB search failed");

        let large = max_gaussians_for_vram(
            8_usize * 1024 * 1024 * 1024, // 8 GB
            3,
            512,
            512,
            false,
            false,
            0,
        )
        .expect("8GB search failed");

        assert!(large > small, "8 GB should fit more gaussians than 1 GB");
    }

    #[test]
    fn test_max_gaussians_tiny_vram_returns_zero_or_small() {
        // 1 KB of VRAM with 1B param model should fit nothing
        let result = max_gaussians_for_vram(
            1024,
            3,
            512,
            512,
            false,
            false,
            4_000_000_000, // 4 GB model
        )
        .expect("tiny vram search failed");
        assert_eq!(
            result, 0,
            "should fit 0 gaussians with 1KB VRAM and 4GB model"
        );
    }

    #[test]
    fn test_max_gaussians_no_model_weights() {
        // Without model weights, even small VRAM should fit some gaussians
        let result = max_gaussians_for_vram(
            512 * 1024 * 1024, // 512 MB
            0,                 // sh_degree 0 is cheapest
            256,
            256,
            false,
            false,
            0,
        )
        .expect("512 MB search failed");
        assert!(
            result > 0,
            "should fit some gaussians in 512MB without model"
        );
    }

    #[test]
    fn test_max_gaussians_invalid_sh_degree() {
        let err = max_gaussians_for_vram(8 * 1024 * 1024 * 1024, 5, 512, 512, false, false, 0)
            .expect_err("should fail for sh_degree 5");
        assert!(matches!(err, MemEstimateError::InvalidParam(_)));
    }

    #[test]
    fn test_max_gaussians_zero_vram_with_no_fixed_costs() {
        // With zero model weights and zero VRAM, should return 0
        let result =
            max_gaussians_for_vram(0, 3, 512, 512, false, false, 0).expect("zero vram search");
        assert_eq!(result, 0);
    }

    // ------------------------------------------------------------------
    // memory_breakdown_percent
    // ------------------------------------------------------------------

    #[test]
    fn test_breakdown_percent_sums_to_100() {
        let config = MemEstimateConfig::default();
        let est = estimate_memory(&config).expect("estimation failed");
        let breakdown = memory_breakdown_percent(&est);
        let total = breakdown.gaussians_pct
            + breakdown.render_pct
            + breakdown.training_pct
            + breakdown.model_pct;
        // Allow small floating-point rounding
        assert!(
            (total - 100.0).abs() < 1.0,
            "Breakdown percentages should sum to ~100%, got {total:.2}%"
        );
    }

    #[test]
    fn test_breakdown_percent_zero_total() {
        // Construct a fake zero-total estimate
        let config = MemEstimateConfig {
            n_gaussians: 0,
            sh_degree: 0,
            render_width: 1,
            render_height: 1,
            use_ema: false,
            use_optimizer: false,
            n_render_views: 0,
            diffusion_model_params: 0,
            target_vram_bytes: 8_usize * 1024 * 1024 * 1024,
            fp16_optimizer: false,
        };
        let est = estimate_memory(&config).expect("zero config failed");
        // total_gpu_bytes will be > 0 due to render buffers, but test breakdown logic
        let breakdown = memory_breakdown_percent(&est);
        assert!(breakdown.gaussians_pct >= 0.0);
        assert!(breakdown.render_pct >= 0.0);
        assert!(breakdown.training_pct >= 0.0);
        assert!(breakdown.model_pct >= 0.0);
    }

    #[test]
    fn test_breakdown_model_dominates() {
        // With a huge model and tiny gaussian count, model_pct should be largest
        let config = MemEstimateConfig {
            n_gaussians: 100,
            sh_degree: 0,
            diffusion_model_params: 1_000_000_000, // 1B params = 4 GB
            use_ema: false,
            use_optimizer: false,
            n_render_views: 1,
            render_width: 64,
            render_height: 64,
            ..Default::default()
        };
        let est = estimate_memory(&config).expect("estimation failed");
        let breakdown = memory_breakdown_percent(&est);
        assert!(
            breakdown.model_pct > breakdown.gaussians_pct,
            "model should dominate: model={:.1}% gaussians={:.1}%",
            breakdown.model_pct,
            breakdown.gaussians_pct
        );
    }

    // ------------------------------------------------------------------
    // mem_format_bytes
    // ------------------------------------------------------------------

    #[test]
    fn test_format_bytes_512() {
        assert_eq!(mem_format_bytes(512), "512 B");
    }

    #[test]
    fn test_format_bytes_0() {
        assert_eq!(mem_format_bytes(0), "0 B");
    }

    #[test]
    fn test_format_bytes_1023() {
        assert_eq!(mem_format_bytes(1023), "1023 B");
    }

    #[test]
    fn test_format_bytes_1024() {
        assert_eq!(mem_format_bytes(1024), "1.00 KB");
    }

    #[test]
    fn test_format_bytes_1500() {
        // 1500 / 1024 = 1.4648... → "1.46 KB"
        assert_eq!(mem_format_bytes(1500), "1.46 KB");
    }

    #[test]
    fn test_format_bytes_1536() {
        // 1536 / 1024 = 1.5 → "1.50 KB"
        assert_eq!(mem_format_bytes(1536), "1.50 KB");
    }

    #[test]
    fn test_format_bytes_1_mb() {
        assert_eq!(mem_format_bytes(1024 * 1024), "1.00 MB");
    }

    #[test]
    fn test_format_bytes_1_5_mb() {
        // 1500000 / (1024*1024) ≈ 1.430511... → "1.43 MB"
        assert_eq!(mem_format_bytes(1_500_000), "1.43 MB");
    }

    #[test]
    fn test_format_bytes_1_gb() {
        assert_eq!(mem_format_bytes(1024 * 1024 * 1024), "1.00 GB");
    }

    #[test]
    fn test_format_bytes_2_gb() {
        // 2_000_000_000 / (1024^3) ≈ 1.8626... → "1.86 GB"
        assert_eq!(mem_format_bytes(2_000_000_000), "1.86 GB");
    }

    #[test]
    fn test_format_bytes_8_gb() {
        // 8 * 1024^3
        assert_eq!(mem_format_bytes(8 * 1024 * 1024 * 1024), "8.00 GB");
    }

    // ------------------------------------------------------------------
    // compare_memory_configs
    // ------------------------------------------------------------------

    #[test]
    fn test_compare_identical_configs_zero_delta() {
        let config = MemEstimateConfig::default();
        let delta = compare_memory_configs(&config, &config).expect("compare failed");
        assert_eq!(delta.delta_bytes, 0);
        assert_eq!(delta.a_total_bytes, delta.b_total_bytes);
        assert!((delta.delta_pct).abs() < f32::EPSILON);
    }

    #[test]
    fn test_compare_more_gaussians_positive_delta() {
        let a = MemEstimateConfig {
            n_gaussians: 100_000,
            diffusion_model_params: 0,
            ..Default::default()
        };
        let b = MemEstimateConfig {
            n_gaussians: 500_000,
            diffusion_model_params: 0,
            ..Default::default()
        };
        let delta = compare_memory_configs(&a, &b).expect("compare failed");
        assert!(
            delta.delta_bytes > 0,
            "More gaussians should use more memory"
        );
        assert!(delta.delta_pct > 0.0);
    }

    #[test]
    fn test_compare_fewer_gaussians_negative_delta() {
        let a = MemEstimateConfig {
            n_gaussians: 500_000,
            diffusion_model_params: 0,
            ..Default::default()
        };
        let b = MemEstimateConfig {
            n_gaussians: 100_000,
            diffusion_model_params: 0,
            ..Default::default()
        };
        let delta = compare_memory_configs(&a, &b).expect("compare failed");
        assert!(
            delta.delta_bytes < 0,
            "Fewer gaussians should use less memory"
        );
    }

    #[test]
    fn test_compare_delta_values() {
        let a = MemEstimateConfig {
            n_gaussians: 100_000,
            diffusion_model_params: 0,
            use_ema: false,
            use_optimizer: false,
            ..Default::default()
        };
        let b = MemEstimateConfig {
            n_gaussians: 200_000,
            diffusion_model_params: 0,
            use_ema: false,
            use_optimizer: false,
            ..Default::default()
        };
        let delta = compare_memory_configs(&a, &b).expect("compare failed");
        assert_eq!(
            delta.a_total_bytes + delta.delta_bytes as usize,
            delta.b_total_bytes
        );
    }

    // ------------------------------------------------------------------
    // format_memory_estimate
    // ------------------------------------------------------------------

    #[test]
    fn test_format_memory_estimate_non_empty() {
        let config = MemEstimateConfig::default();
        let est = estimate_memory(&config).expect("estimation failed");
        let s = format_memory_estimate(&est);
        assert!(!s.is_empty());
        assert!(s.contains("Total GPU"));
        assert!(s.contains("Recommended VRAM"));
    }

    #[test]
    fn test_format_memory_estimate_contains_fit_status() {
        let config = MemEstimateConfig {
            n_gaussians: 100,
            sh_degree: 0,
            diffusion_model_params: 0,
            use_ema: false,
            use_optimizer: false,
            render_width: 64,
            render_height: 64,
            n_render_views: 1,
            target_vram_bytes: 8_usize * 1024 * 1024 * 1024,
            fp16_optimizer: false,
        };
        let est = estimate_memory(&config).expect("estimation failed");
        let s = format_memory_estimate(&est);
        assert!(s.contains("YES") || s.contains("NO"));
    }

    #[test]
    fn test_format_memory_estimate_contains_breakdown() {
        let config = MemEstimateConfig::default();
        let est = estimate_memory(&config).expect("estimation failed");
        let s = format_memory_estimate(&est);
        assert!(s.contains("Gaussians"));
        assert!(s.contains("Render"));
        assert!(s.contains("Training"));
        assert!(s.contains("Model"));
    }

    // ------------------------------------------------------------------
    // Additional edge-case tests
    // ------------------------------------------------------------------

    #[test]
    fn test_layout_1m_gaussians_sh3_reasonable_size() {
        let layout = estimate_gaussian_layout(1_000_000, 3).expect("1M layout failed");
        // 1M gaussians at sh3: at minimum positions alone = 12 MB
        // total should be well above 12 MB
        assert!(layout.total_bytes > 12 * 1024 * 1024);
    }

    #[test]
    fn test_estimate_memory_vram_threshold_boundary() {
        // Set target_vram_bytes exactly to total_gpu_bytes
        let config = MemEstimateConfig {
            n_gaussians: 10_000,
            sh_degree: 0,
            diffusion_model_params: 0,
            use_ema: false,
            use_optimizer: false,
            n_render_views: 1,
            render_width: 64,
            render_height: 64,
            target_vram_bytes: usize::MAX, // always fits
            fp16_optimizer: false,
        };
        let est = estimate_memory(&config).expect("boundary test failed");
        assert!(est.fits_in_vram);
    }

    #[test]
    fn test_max_gaussians_ema_and_optimizer_reduce_count() {
        let without =
            max_gaussians_for_vram(2_usize * 1024 * 1024 * 1024, 3, 512, 512, false, false, 0)
                .expect("search failed");

        let with_all =
            max_gaussians_for_vram(2_usize * 1024 * 1024 * 1024, 3, 512, 512, true, true, 0)
                .expect("search failed with ema+opt");

        assert!(
            without >= with_all,
            "EMA+optimizer should reduce max gaussians (or equal): without={without} with_all={with_all}"
        );
    }

    #[test]
    fn test_compare_configs_delta_pct_reasonable() {
        let a = MemEstimateConfig {
            n_gaussians: 100_000,
            diffusion_model_params: 0,
            use_ema: false,
            use_optimizer: false,
            ..Default::default()
        };
        let b = MemEstimateConfig {
            n_gaussians: 200_000,
            diffusion_model_params: 0,
            use_ema: false,
            use_optimizer: false,
            ..Default::default()
        };
        let delta = compare_memory_configs(&a, &b).expect("compare failed");
        // delta_pct should be positive and less than 200%
        assert!(delta.delta_pct > 0.0);
        assert!(delta.delta_pct < 200.0);
    }
}
