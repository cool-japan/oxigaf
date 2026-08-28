//! Rasterizer configuration.

use crate::rasterizer::RASTERIZE_TILE_SIZE;
use crate::RenderError;
use serde::{Deserialize, Serialize};

/// GPU architecture preset for optimal workgroup sizes.
///
/// Different GPU architectures have different optimal workgroup sizes:
/// - NVIDIA: 256 threads (32 threads/warp × 8 warps for high occupancy)
/// - AMD: 64 threads (64 threads/wavefront)
/// - Apple Silicon: 32-64 threads (SIMD width varies)
/// - Intel: 32 threads (8 EU threads × 4 SIMD lanes)
///
/// # What the rasterizer actually honours
///
/// [`Rasterizer::from_device`](crate::Rasterizer::from_device) resolves
/// [`RasterConfig::effective_preprocess_wg_size`] (this preset, or
/// [`RasterConfig::preprocess_workgroup_size`] when set) and compiles it
/// into every 1-D kernel whose `@workgroup_size` is pure dispatch geometry
/// — the `preprocess` variants, `preprocess_bwd`, `tile_assign`,
/// `tile_ranges` and `atomic_to_f32` — by substituting the attribute into
/// the WGSL source before creating the shader module, then derives the
/// dispatch grid from the same number via
/// [`WorkgroupConfig::for_linear_size`](crate::workgroup::WorkgroupConfig::for_linear_size).
///
/// Three kinds of kernel are deliberately **excluded**, because their thread
/// count is baked into their bodies rather than just their attribute:
///
/// * `prefix_sum` / `prefix_sum_add` (a 512-element shared-memory block and
///   literal 256/512 strides) and `radix_histogram` / `radix_scatter`
///   (workgroup-sized shared histograms) — retargeting these silently
///   corrupts their output;
/// * `rasterize_fwd` / `rasterize_bwd` / `cov2d_bwd`, whose
///   `@workgroup_size(16, 16)` *is* the tile size (one thread per tile
///   pixel) and is therefore fixed by `RASTERIZE_TILE_SIZE`, not by
///   [`RasterConfig::rasterize_workgroup_size`].
///
/// [`Self::rasterize_workgroup_size`] and
/// [`Self::prefix_sum_workgroup_size`] consequently remain advisory: they
/// describe what a GPU class would prefer, and
/// [`crate::workgroup::WorkgroupBenchmarker::recommend_on_device`] can rank
/// candidates on real hardware, but the shipped kernels for those stages
/// cannot take the value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum GpuPreset {
    /// Automatically detect based on adapter info.
    #[default]
    Auto,
    /// NVIDIA GPUs (GeForce, Quadro, Tesla).
    /// Optimal: 256 threads for compute, 16×16 for 2D tiles.
    Nvidia,
    /// AMD GPUs (Radeon, Instinct).
    /// Optimal: 64 threads (wavefront size).
    Amd,
    /// Apple Silicon (M1, M2, M3, etc.).
    /// Optimal: 32-64 threads, conservative for power efficiency.
    Apple,
    /// Intel GPUs (Iris, Arc).
    /// Optimal: 32 threads for integrated, 64 for discrete.
    Intel,
    /// Generic/fallback for unknown GPUs.
    /// Uses conservative values that work everywhere.
    Generic,
}

impl GpuPreset {
    /// Detect GPU preset from adapter name.
    ///
    /// This is a heuristic based on common GPU naming patterns.
    #[must_use]
    pub fn from_adapter_name(name: &str) -> Self {
        let name_lower = name.to_lowercase();

        if name_lower.contains("nvidia")
            || name_lower.contains("geforce")
            || name_lower.contains("quadro")
            || name_lower.contains("tesla")
            || name_lower.contains("rtx")
            || name_lower.contains("gtx")
        {
            Self::Nvidia
        } else if name_lower.contains("amd")
            || name_lower.contains("radeon")
            || name_lower.contains("rx ")
            || name_lower.contains("vega")
            || name_lower.contains("navi")
            || name_lower.contains("rdna")
        {
            Self::Amd
        } else if name_lower.contains("apple")
            || name_lower.contains("m1")
            || name_lower.contains("m2")
            || name_lower.contains("m3")
            || name_lower.contains("m4")
        {
            Self::Apple
        } else if name_lower.contains("intel")
            || name_lower.contains("iris")
            || name_lower.contains("arc")
            || name_lower.contains("uhd")
        {
            Self::Intel
        } else {
            Self::Generic
        }
    }

    /// Get recommended preprocess shader workgroup size.
    ///
    /// This is for 1D compute shaders processing Gaussians.
    #[must_use]
    pub const fn preprocess_workgroup_size(&self) -> u32 {
        match self {
            Self::Nvidia | Self::Auto => 256,
            Self::Amd => 64,
            Self::Apple => 64,
            Self::Intel => 32,
            Self::Generic => 64,
        }
    }

    /// Get recommended rasterize shader workgroup size (2D).
    ///
    /// Returns (x, y) dimensions for tile-based rasterization.
    /// Total threads = x × y.
    #[must_use]
    pub const fn rasterize_workgroup_size(&self) -> [u32; 2] {
        match self {
            // 16×16 = 256 threads, matches tile size for NVIDIA
            Self::Nvidia | Self::Auto => [16, 16],
            // 8×8 = 64 threads, matches AMD wavefront
            Self::Amd => [8, 8],
            // 8×8 = 64 threads, good balance for Apple
            Self::Apple => [8, 8],
            // 8×4 = 32 threads for Intel iGPU
            Self::Intel => [8, 4],
            // 8×8 = 64 threads, conservative default
            Self::Generic => [8, 8],
        }
    }

    /// Get recommended sort shader workgroup size.
    #[must_use]
    pub const fn sort_workgroup_size(&self) -> u32 {
        match self {
            Self::Nvidia | Self::Auto => 256,
            Self::Amd => 64,
            Self::Apple => 64,
            Self::Intel => 32,
            Self::Generic => 64,
        }
    }

    /// Get recommended prefix sum workgroup size.
    ///
    /// Larger sizes improve efficiency for prefix sum operations.
    #[must_use]
    pub const fn prefix_sum_workgroup_size(&self) -> u32 {
        match self {
            Self::Nvidia | Self::Auto => 512,
            Self::Amd => 256,
            Self::Apple => 256,
            Self::Intel => 128,
            Self::Generic => 256,
        }
    }
}

/// Configuration for the 3DGS rasterizer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RasterConfig {
    /// Output image width in pixels.
    pub image_width: u32,
    /// Output image height in pixels.
    pub image_height: u32,
    /// Tile size for tiled rasterization (default: 16).
    pub tile_size: u32,
    /// Spherical harmonics degree (0-3).
    pub sh_degree: u32,
    /// Near clipping plane.
    pub near_plane: f32,
    /// Far clipping plane.
    pub far_plane: f32,
    /// Background color RGB.
    pub background: [f32; 3],

    // --- Output options ---
    /// Enable depth buffer output (default: true).
    pub output_depth: bool,
    /// Enable normal buffer output (default: false).
    /// Normals are computed from Gaussian orientations weighted by alpha.
    pub output_normals: bool,

    // --- Performance options ---
    /// GPU architecture preset for workgroup sizes. Supplies the default
    /// 1-D workgroup size the rasterizer compiles into its retargetable
    /// kernels — see [`GpuPreset`]'s doc for exactly which stages honour it.
    pub gpu_preset: GpuPreset,
    /// Override the 1-D (preprocess-class) workgroup size; `None` uses the
    /// preset. Compiled into the retargetable kernels and used for their
    /// dispatch grid — see [`GpuPreset`]'s doc.
    pub preprocess_workgroup_size: Option<u32>,
    /// Preferred rasterize workgroup size (None = use preset).
    /// Format: [width, height].
    ///
    /// Advisory only: the rasterization kernels' `@workgroup_size(16, 16)`
    /// *is* [`crate::rasterizer::RASTERIZE_TILE_SIZE`] (one thread per tile
    /// pixel), so it cannot vary independently of `tile_size`. See
    /// [`GpuPreset`]'s doc.
    pub rasterize_workgroup_size: Option<[u32; 2]>,

    // --- Culling options ---
    /// Transmittance threshold for early termination (default: 1/255).
    /// When transmittance drops below this, stop processing Gaussians.
    pub transmittance_threshold: f32,
    /// Enable hierarchical tile culling (default: true).
    /// Skips tiles with no visible Gaussians.
    ///
    /// Not yet consulted by the rasterizer: no dispatch code reads this
    /// field, so tile culling behaviour does not currently change when it
    /// is toggled. Honouring it needs a coarse pre-pass plus an indirect
    /// dispatch in `rasterize_fwd.wgsl`; until then the per-tile early-out
    /// on an empty `tile_ranges` entry is the only culling in effect, and it
    /// is unconditional.
    pub hierarchical_culling: bool,

    // --- Memory options ---
    /// Maximum GPU memory budget in megabytes (default: 512).
    /// The buffer pool will evict least-recently-used buffers when this limit
    /// is exceeded. Set to 0 to disable pooling.
    pub max_gpu_memory_mb: u32,
    /// Enable buffer pooling for intermediate allocations (default: true).
    /// When enabled, buffers are reused from a pool instead of being
    /// allocated fresh each time.
    pub enable_buffer_pooling: bool,

    // --- SH Optimization options ---
    /// Enable Spherical Harmonics optimization (default: true).
    ///
    /// When enabled, the rasterizer uses several performance optimizations:
    /// - **SH Degree 0 Fast Path**: When `sh_degree=0`, uses simple constant color
    ///   (3 multiplies + 3 adds) instead of full SH computation, skipping
    ///   direction computation entirely.
    /// - **Precomputed SH Basis**: View-independent SH basis functions are
    ///   precomputed and stored in uniform buffers to reduce per-Gaussian
    ///   computation overhead.
    /// - **SIMD-friendly Layout**: SH coefficients are vec4 aligned for
    ///   vectorized GPU operations, improving memory bandwidth utilization.
    /// - **Shader Variants**: Specialized shaders for common `sh_degrees` (0, 1, 2, 3)
    ///   are selected at pipeline creation time, eliminating runtime branching.
    ///
    /// Performance implications:
    /// - SH degree 0: ~10x faster than degree 3 (3 ops vs ~100 ops per Gaussian)
    /// - SIMD optimization: ~20-30% faster for degrees 1-3
    /// - Shader variants: ~5-10% faster by eliminating dynamic branching
    ///
    /// Set to `false` for debugging or when custom SH evaluation is needed.
    pub sh_optimization: bool,

    /// Use specialized shader variants for each SH degree (default: true).
    ///
    /// When enabled, the pipeline selects a specialized preprocess shader
    /// at creation time based on `sh_degree`:
    /// - `sh_degree=0`: Uses optimized DC-only path (no direction computation)
    /// - `sh_degree=1`: Uses optimized linear SH path
    /// - `sh_degree=2`: Uses optimized quadratic SH path
    /// - `sh_degree=3`: Uses full SH evaluation with SIMD optimizations
    ///
    /// This eliminates runtime branching in the shader, improving performance.
    /// Requires recompiling pipelines if `sh_degree` changes at runtime.
    ///
    /// When disabled, uses a single unified shader with runtime branching.
    pub use_sh_variants: bool,
}

impl Default for RasterConfig {
    fn default() -> Self {
        Self {
            image_width: 512,
            image_height: 512,
            tile_size: 16,
            sh_degree: 3,
            near_plane: 0.01,
            far_plane: 100.0,
            background: [0.0, 0.0, 0.0],
            // Output options
            output_depth: true,
            output_normals: false,
            // Performance options
            gpu_preset: GpuPreset::Auto,
            preprocess_workgroup_size: None,
            rasterize_workgroup_size: None,
            // Culling options
            transmittance_threshold: 1.0 / 255.0,
            hierarchical_culling: true,
            // Memory options
            max_gpu_memory_mb: 512,
            enable_buffer_pooling: true,
            // SH optimization options
            sh_optimization: true,
            use_sh_variants: true,
        }
    }
}

impl RasterConfig {
    /// Create a new config with builder pattern.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set image resolution.
    #[must_use]
    pub const fn with_resolution(mut self, width: u32, height: u32) -> Self {
        self.image_width = width;
        self.image_height = height;
        self
    }

    /// Set SH degree (0-3).
    #[must_use]
    pub const fn with_sh_degree(mut self, degree: u32) -> Self {
        self.sh_degree = degree;
        self
    }

    /// Set background color.
    #[must_use]
    pub const fn with_background(mut self, color: [f32; 3]) -> Self {
        self.background = color;
        self
    }

    /// Enable or disable depth output.
    #[must_use]
    pub const fn with_depth_output(mut self, enabled: bool) -> Self {
        self.output_depth = enabled;
        self
    }

    /// Enable or disable normal output.
    #[must_use]
    pub const fn with_normal_output(mut self, enabled: bool) -> Self {
        self.output_normals = enabled;
        self
    }

    /// Set GPU preset for workgroup sizes. Supplies the 1-D workgroup size
    /// the rasterizer compiles into its retargetable kernels — see
    /// [`GpuPreset`]'s doc.
    #[must_use]
    pub const fn with_gpu_preset(mut self, preset: GpuPreset) -> Self {
        self.gpu_preset = preset;
        self
    }

    /// Override the 1-D (preprocess-class) workgroup size.
    ///
    /// Must be a power of two the device supports;
    /// [`Rasterizer::from_device`](crate::Rasterizer::from_device) rejects
    /// anything else rather than failing later in shader compilation.
    #[must_use]
    pub const fn with_preprocess_workgroup_size(mut self, size: u32) -> Self {
        self.preprocess_workgroup_size = Some(size);
        self
    }

    /// Set the preferred rasterize workgroup size. Advisory only — see
    /// [`Self::rasterize_workgroup_size`].
    #[must_use]
    pub const fn with_rasterize_workgroup_size(mut self, size: [u32; 2]) -> Self {
        self.rasterize_workgroup_size = Some(size);
        self
    }

    /// Number of tiles in X direction.
    #[inline]
    #[must_use]
    pub const fn tiles_x(&self) -> u32 {
        self.image_width.div_ceil(self.tile_size)
    }

    /// Number of tiles in Y direction.
    #[inline]
    #[must_use]
    pub const fn tiles_y(&self) -> u32 {
        self.image_height.div_ceil(self.tile_size)
    }

    /// Total number of tiles.
    #[inline]
    #[must_use]
    pub const fn num_tiles(&self) -> u32 {
        self.tiles_x() * self.tiles_y()
    }

    /// Total number of pixels.
    #[inline]
    #[must_use]
    pub const fn num_pixels(&self) -> u32 {
        self.image_width * self.image_height
    }

    /// Number of SH coefficients per Gaussian: `(degree+1)^2 * 3`.
    #[inline]
    #[must_use]
    pub const fn sh_coeffs_per_gaussian(&self) -> u32 {
        let n = (self.sh_degree + 1) * (self.sh_degree + 1);
        n * 3
    }

    /// Get effective preprocess workgroup size.
    ///
    /// Returns the explicit override if set, otherwise the preset's value.
    /// This is the number compiled into the retargetable 1-D kernels and
    /// used for their dispatch grid — see [`GpuPreset`]'s doc.
    #[inline]
    #[must_use]
    pub fn effective_preprocess_wg_size(&self) -> u32 {
        self.preprocess_workgroup_size
            .unwrap_or_else(|| self.gpu_preset.preprocess_workgroup_size())
    }

    /// Get effective rasterize workgroup size.
    ///
    /// Returns the override if set, otherwise the preset's value. Advisory:
    /// the rasterization kernels are locked to the tile size — see
    /// [`Self::rasterize_workgroup_size`].
    #[inline]
    #[must_use]
    pub fn effective_rasterize_wg_size(&self) -> [u32; 2] {
        self.rasterize_workgroup_size
            .unwrap_or_else(|| self.gpu_preset.rasterize_workgroup_size())
    }

    /// Get effective prefix sum workgroup size. Advisory: `prefix_sum.wgsl`
    /// bakes its thread count into shared-memory sizing — see
    /// [`GpuPreset`]'s doc.
    #[inline]
    #[must_use]
    pub fn effective_prefix_sum_wg_size(&self) -> u32 {
        self.gpu_preset.prefix_sum_workgroup_size()
    }

    /// Compute number of workgroups needed for preprocess dispatch, from
    /// [`Self::effective_preprocess_wg_size`] — the same size the rasterizer
    /// compiles into the preprocess kernel.
    #[inline]
    #[must_use]
    pub fn preprocess_workgroups(&self, n_gaussians: u32) -> u32 {
        n_gaussians.div_ceil(self.effective_preprocess_wg_size())
    }

    /// Set maximum GPU memory budget in megabytes.
    ///
    /// The buffer pool will evict least-recently-used buffers when this
    /// limit is exceeded. Set to 0 to disable pooling entirely.
    #[must_use]
    pub const fn with_max_gpu_memory_mb(mut self, memory_mb: u32) -> Self {
        self.max_gpu_memory_mb = memory_mb;
        self
    }

    /// Enable or disable buffer pooling.
    #[must_use]
    pub const fn with_buffer_pooling(mut self, enabled: bool) -> Self {
        self.enable_buffer_pooling = enabled;
        self
    }

    /// Enable or disable SH optimization.
    ///
    /// When enabled, uses optimized SH evaluation including:
    /// - Fast path for degree 0 (DC term only)
    /// - SIMD-friendly vec4 layout
    /// - Precomputed basis functions
    #[must_use]
    pub const fn with_sh_optimization(mut self, enabled: bool) -> Self {
        self.sh_optimization = enabled;
        self
    }

    /// Enable or disable SH shader variants.
    ///
    /// When enabled, uses specialized shaders for each SH degree
    /// to eliminate runtime branching.
    #[must_use]
    pub const fn with_sh_variants(mut self, enabled: bool) -> Self {
        self.use_sh_variants = enabled;
        self
    }

    /// Get output flags as a bitmask for shaders.
    ///
    /// Bit 0: output_depth
    /// Bit 1: output_normals
    #[inline]
    #[must_use]
    pub const fn output_flags(&self) -> u32 {
        let mut flags = 0u32;
        if self.output_depth {
            flags |= 1;
        }
        if self.output_normals {
            flags |= 2;
        }
        flags
    }

    /// Get memory budget in bytes.
    #[inline]
    #[must_use]
    pub const fn memory_budget_bytes(&self) -> u64 {
        (self.max_gpu_memory_mb as u64) * 1024 * 1024
    }

    /// Validate that this configuration's values are internally consistent.
    ///
    /// All fields are public and the type derives `Deserialize`, so a
    /// config loaded from an untrusted file (or built by hand) can carry
    /// values that would otherwise panic deep inside a consumer — e.g.
    /// `tile_size: 0` reaches a `div_ceil` panic in [`Self::tiles_x`] /
    /// [`Self::tiles_y`] / [`Self::num_tiles`] instead of failing here with
    /// a clear message.
    ///
    /// This checks only what is *internally* consistent. The extra
    /// constraints the compiled GPU shaders impose (notably the fixed tile
    /// size) live in [`Self::validate_for_rasterizer`], which
    /// [`Rasterizer::from_device`](crate::Rasterizer::from_device) calls.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::Rasterize`] describing the first invalid
    /// field found:
    /// - `tile_size == 0`.
    /// - `sh_degree > 3` — only degrees 0-3 have a defined SH basis; a
    ///   larger value would silently desync [`Self::sh_coeffs_per_gaussian`]'s
    ///   stride from what shaders and CPU code expect.
    /// - `image_width * image_height` does not fit in `u32` — the width
    ///   and height are multiplied directly in [`Self::num_pixels`].
    /// - `near_plane >= far_plane` — a degenerate or inverted clip range.
    pub fn validate(&self) -> Result<(), RenderError> {
        if self.tile_size == 0 {
            return Err(RenderError::Rasterize(
                "RasterConfig: tile_size must be > 0".to_string(),
            ));
        }
        if self.sh_degree > 3 {
            return Err(RenderError::Rasterize(format!(
                "RasterConfig: sh_degree must be in [0, 3], got {}",
                self.sh_degree
            )));
        }
        if (u64::from(self.image_width)) * (u64::from(self.image_height)) > u64::from(u32::MAX) {
            return Err(RenderError::Rasterize(format!(
                "RasterConfig: image_width * image_height ({} * {}) overflows u32",
                self.image_width, self.image_height
            )));
        }
        // `partial_cmp` rather than `>=`: a NaN plane compares as neither, and
        // must be rejected rather than silently accepted.
        if self.near_plane.partial_cmp(&self.far_plane) != Some(std::cmp::Ordering::Less) {
            return Err(RenderError::Rasterize(format!(
                "RasterConfig: near_plane ({}) must be < far_plane ({})",
                self.near_plane, self.far_plane
            )));
        }
        Ok(())
    }

    /// Validate this configuration against the **compiled GPU rasterizer**.
    ///
    /// Runs every [`Self::validate`] check and additionally rejects a
    /// `tile_size` other than [`RASTERIZE_TILE_SIZE`]: `rasterize_fwd.wgsl`
    /// and `rasterize_bwd.wgsl` declare `@workgroup_size(16, 16)`, which
    /// WGSL fixes at shader-compile time, and one workgroup covers exactly
    /// one tile — so any other tile size desyncs the tile grid from the
    /// workgroup grid and silently renders garbage.
    ///
    /// This is deliberately separate from [`Self::validate`]: a config used
    /// for CPU-side tiling maths (or by a future shader variant) is not
    /// wrong just because it does not match the shipped shaders, but a
    /// config handed to [`Rasterizer`](crate::Rasterizer) is.
    ///
    /// # Errors
    ///
    /// Everything [`Self::validate`] reports, plus [`RenderError::Rasterize`]
    /// when `tile_size != RASTERIZE_TILE_SIZE`.
    pub fn validate_for_rasterizer(&self) -> Result<(), RenderError> {
        self.validate()?;
        if self.tile_size != RASTERIZE_TILE_SIZE {
            return Err(RenderError::Rasterize(format!(
                "RasterConfig::tile_size must be {RASTERIZE_TILE_SIZE}, got {}: \
                 rasterize_fwd.wgsl and rasterize_bwd.wgsl declare \
                 @workgroup_size({RASTERIZE_TILE_SIZE}, {RASTERIZE_TILE_SIZE}) at \
                 shader-compile time, so any other tile size desyncs the tile grid \
                 from the workgroup grid",
                self.tile_size
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_preset_detection() {
        assert_eq!(
            GpuPreset::from_adapter_name("NVIDIA GeForce RTX 4090"),
            GpuPreset::Nvidia
        );
        assert_eq!(
            GpuPreset::from_adapter_name("AMD Radeon RX 7900 XTX"),
            GpuPreset::Amd
        );
        assert_eq!(
            GpuPreset::from_adapter_name("Apple M3 Max"),
            GpuPreset::Apple
        );
        assert_eq!(
            GpuPreset::from_adapter_name("Intel Arc A770"),
            GpuPreset::Intel
        );
        assert_eq!(
            GpuPreset::from_adapter_name("Unknown GPU"),
            GpuPreset::Generic
        );
    }

    #[test]
    fn test_workgroup_sizes() {
        assert_eq!(GpuPreset::Nvidia.preprocess_workgroup_size(), 256);
        assert_eq!(GpuPreset::Amd.preprocess_workgroup_size(), 64);
        assert_eq!(GpuPreset::Apple.preprocess_workgroup_size(), 64);
        assert_eq!(GpuPreset::Intel.preprocess_workgroup_size(), 32);

        assert_eq!(GpuPreset::Nvidia.rasterize_workgroup_size(), [16, 16]);
        assert_eq!(GpuPreset::Amd.rasterize_workgroup_size(), [8, 8]);
    }

    #[test]
    fn test_config_builder() {
        let config = RasterConfig::new()
            .with_resolution(1920, 1080)
            .with_sh_degree(2)
            .with_normal_output(true)
            .with_gpu_preset(GpuPreset::Nvidia);

        assert_eq!(config.image_width, 1920);
        assert_eq!(config.image_height, 1080);
        assert_eq!(config.sh_degree, 2);
        assert!(config.output_normals);
        assert_eq!(config.gpu_preset, GpuPreset::Nvidia);
    }

    #[test]
    fn test_output_flags() {
        let mut config = RasterConfig::default();

        // Default: depth on, normals off
        assert_eq!(config.output_flags(), 0b01);

        config.output_normals = true;
        assert_eq!(config.output_flags(), 0b11);

        config.output_depth = false;
        assert_eq!(config.output_flags(), 0b10);
    }

    #[test]
    fn test_tile_calculations() {
        let config = RasterConfig::new().with_resolution(1920, 1080);
        assert_eq!(config.tiles_x(), 120); // 1920 / 16
        assert_eq!(config.tiles_y(), 68); // 1080 / 16 = 67.5 -> 68
        assert_eq!(config.num_tiles(), 120 * 68);
    }

    #[test]
    fn test_sh_optimization_settings() {
        // Default: both enabled
        let config = RasterConfig::default();
        assert!(config.sh_optimization);
        assert!(config.use_sh_variants);

        // Disable SH optimization
        let config = RasterConfig::new()
            .with_sh_optimization(false)
            .with_sh_variants(false);
        assert!(!config.sh_optimization);
        assert!(!config.use_sh_variants);

        // Enable only sh_optimization, disable variants
        let config = RasterConfig::new()
            .with_sh_optimization(true)
            .with_sh_variants(false);
        assert!(config.sh_optimization);
        assert!(!config.use_sh_variants);
    }

    #[test]
    fn test_sh_coeffs_per_gaussian() {
        // Degree 0: (0+1)^2 * 3 = 3
        let config = RasterConfig::new().with_sh_degree(0);
        assert_eq!(config.sh_coeffs_per_gaussian(), 3);

        // Degree 1: (1+1)^2 * 3 = 12
        let config = RasterConfig::new().with_sh_degree(1);
        assert_eq!(config.sh_coeffs_per_gaussian(), 12);

        // Degree 2: (2+1)^2 * 3 = 27
        let config = RasterConfig::new().with_sh_degree(2);
        assert_eq!(config.sh_coeffs_per_gaussian(), 27);

        // Degree 3: (3+1)^2 * 3 = 48
        let config = RasterConfig::new().with_sh_degree(3);
        assert_eq!(config.sh_coeffs_per_gaussian(), 48);
    }

    #[test]
    fn test_validate_default_is_ok() {
        assert!(RasterConfig::default().validate().is_ok());
    }

    #[test]
    fn test_validate_rejects_zero_tile_size() {
        let config = RasterConfig {
            tile_size: 0,
            ..RasterConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_rejects_sh_degree_above_three() {
        let mut config = RasterConfig {
            sh_degree: 4,
            ..RasterConfig::default()
        };
        assert!(config.validate().is_err());
        // The documented range [0, 3] is inclusive.
        config.sh_degree = 3;
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_validate_rejects_pixel_count_overflow() {
        let config = RasterConfig {
            image_width: u32::MAX,
            image_height: 2,
            ..RasterConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_rejects_inverted_clip_range() {
        let mut config = RasterConfig {
            near_plane: 10.0,
            far_plane: 1.0,
            ..RasterConfig::default()
        };
        assert!(config.validate().is_err());

        config.near_plane = 5.0;
        config.far_plane = 5.0; // equal is also degenerate
        assert!(config.validate().is_err());

        // NaN compares as neither less nor greater and must not slip through.
        config.near_plane = f32::NAN;
        config.far_plane = 100.0;
        assert!(config.validate().is_err());
    }

    /// Regression (F300): a deserialized config with a tile size the shipped
    /// rasterization shaders were not compiled for must be rejected by the
    /// rasterizer-facing validator, not silently render a desynced tile grid.
    #[test]
    fn test_validate_for_rasterizer_rejects_foreign_tile_size() {
        let mut config = RasterConfig::default();
        assert!(config.validate_for_rasterizer().is_ok());

        config.tile_size = 8;
        // Internally consistent...
        assert!(config.validate().is_ok());
        // ...but not something the compiled shaders can rasterize.
        let err = config
            .validate_for_rasterizer()
            .expect_err("a non-16 tile size must be rejected for the GPU rasterizer");
        let msg = err.to_string();
        assert!(msg.contains("tile_size"), "{msg}");
        assert!(msg.contains("16"), "{msg}");
    }

    /// The rasterizer-facing validator must still catch everything the
    /// general one does (it is what `Rasterizer::from_device` calls).
    #[test]
    fn test_validate_for_rasterizer_subsumes_general_validation() {
        let zero_tile = RasterConfig {
            tile_size: 0,
            ..RasterConfig::default()
        };
        assert!(zero_tile.validate_for_rasterizer().is_err());

        let bad_degree = RasterConfig {
            sh_degree: 4,
            ..RasterConfig::default()
        };
        assert!(bad_degree.validate_for_rasterizer().is_err());

        let inverted_clip = RasterConfig {
            near_plane: 10.0,
            far_plane: 1.0,
            ..RasterConfig::default()
        };
        assert!(inverted_clip.validate_for_rasterizer().is_err());
    }

    #[test]
    fn test_validate_rejects_would_have_panicked_tiles_x() {
        // Regression guard: before `validate()` existed, tile_size = 0
        // would panic inside `tiles_x()`'s `div_ceil`. Confirm `validate()`
        // catches it before any such call is made.
        let config = RasterConfig {
            tile_size: 0,
            ..RasterConfig::default()
        };
        assert!(config.validate().is_err());
        // (Not calling `config.tiles_x()` here: that panic is exactly what
        // callers are expected to avoid by checking `validate()` first.)
    }
}
