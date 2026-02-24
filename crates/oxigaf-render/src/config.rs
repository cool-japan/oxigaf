//! Rasterizer configuration.

use serde::{Deserialize, Serialize};

/// GPU architecture preset for optimal workgroup sizes.
///
/// Different GPU architectures have different optimal workgroup sizes:
/// - NVIDIA: 256 threads (32 threads/warp × 8 warps for high occupancy)
/// - AMD: 64 threads (64 threads/wavefront)
/// - Apple Silicon: 32-64 threads (SIMD width varies)
/// - Intel: 32 threads (8 EU threads × 4 SIMD lanes)
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
    /// GPU architecture preset for workgroup sizes.
    pub gpu_preset: GpuPreset,
    /// Override preprocess workgroup size (None = use preset).
    pub preprocess_workgroup_size: Option<u32>,
    /// Override rasterize workgroup size (None = use preset).
    /// Format: [width, height].
    pub rasterize_workgroup_size: Option<[u32; 2]>,

    // --- Culling options ---
    /// Transmittance threshold for early termination (default: 1/255).
    /// When transmittance drops below this, stop processing Gaussians.
    pub transmittance_threshold: f32,
    /// Enable hierarchical tile culling (default: true).
    /// Skips tiles with no visible Gaussians.
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

    /// Set GPU preset for workgroup sizes.
    #[must_use]
    pub const fn with_gpu_preset(mut self, preset: GpuPreset) -> Self {
        self.gpu_preset = preset;
        self
    }

    /// Override preprocess workgroup size.
    #[must_use]
    pub const fn with_preprocess_workgroup_size(mut self, size: u32) -> Self {
        self.preprocess_workgroup_size = Some(size);
        self
    }

    /// Override rasterize workgroup size.
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
    /// Returns custom override if set, otherwise uses preset.
    #[inline]
    #[must_use]
    pub fn effective_preprocess_wg_size(&self) -> u32 {
        self.preprocess_workgroup_size
            .unwrap_or_else(|| self.gpu_preset.preprocess_workgroup_size())
    }

    /// Get effective rasterize workgroup size.
    ///
    /// Returns custom override if set, otherwise uses preset.
    #[inline]
    #[must_use]
    pub fn effective_rasterize_wg_size(&self) -> [u32; 2] {
        self.rasterize_workgroup_size
            .unwrap_or_else(|| self.gpu_preset.rasterize_workgroup_size())
    }

    /// Get effective prefix sum workgroup size.
    #[inline]
    #[must_use]
    pub fn effective_prefix_sum_wg_size(&self) -> u32 {
        self.gpu_preset.prefix_sum_workgroup_size()
    }

    /// Compute number of workgroups needed for preprocess dispatch.
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
}
