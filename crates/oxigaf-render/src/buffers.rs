//! GPU buffer management for Gaussian splatting.
//!
//! # Buffer Layout for Memory Coalescing
//!
//! GPU memory access patterns significantly impact performance. This module uses
//! layouts optimized for coalesced memory access:
//!
//! ## Structure-of-Arrays (SoA) vs Array-of-Structures (AoS)
//!
//! We use a hybrid approach:
//! - **SoA for per-Gaussian attributes**: positions, rotations, scales, opacities
//!   are stored as separate arrays. This allows coalesced access when all threads
//!   in a workgroup read the same attribute type.
//! - **AoS for outputs**: color (RGBA), normals (XYZ) use packed formats for
//!   cache-friendly pixel writes.
//!
//! ## Alignment
//!
//! - `vec4<f32>` (16 bytes): Positions, rotations, scales use vec4 with padding
//!   for 16-byte alignment and optimal memory transactions.
//! - `vec3<f32>` (12 bytes): Colors, normals, cov2d use vec3 which WGSL pads
//!   to 16 bytes internally.
//! - `f32` (4 bytes): Opacities, depths use scalar f32 for minimal memory.
//!
//! ## Cache Line Considerations
//!
//! GPU cache lines are typically 32-128 bytes. Our buffer sizes are chosen to
//! align well with these:
//! - Positions: N × 16 bytes (4 elements/cache line)
//! - Opacities: N × 4 bytes (16 elements/cache line for 64-byte lines)
//! - Colors: N × 12 bytes (padded to 16, so 4 elements/cache line)

use wgpu::util::DeviceExt;

use crate::config::RasterConfig;
use crate::gaussian::GaussianModel;
use crate::RenderError;

/// Number of elements one `prefix_sum` workgroup scans
/// (`shaders/prefix_sum.wgsl`: 256 threads × 2 elements each).
pub const PREFIX_SUM_BLOCK: u32 = 512;

/// Maximum number of Gaussians the hierarchical prefix sum can scan correctly.
///
/// The scan is three levels deep (`PREFIX_SUM_BLOCK³`): elements → level-1 block
/// sums → level-2 block sums, the last of which always fits in a single
/// workgroup. Above this count the level-2 scan would itself span multiple
/// workgroups and its own block totals would be dropped.
pub const MAX_SCAN_GAUSSIANS: u32 = PREFIX_SUM_BLOCK * PREFIX_SUM_BLOCK * PREFIX_SUM_BLOCK;

/// Default assumed number of tiles a single Gaussian overlaps.
///
/// Only a heuristic for the initial sort-buffer allocation: the real count must
/// be verified against [`IntermediateBuffers::check_pair_capacity`] once the
/// prefix sum has run.
pub const DEFAULT_PAIRS_PER_GAUSSIAN: u32 = 4;

/// Floor for the sort-buffer capacity, so tiny scenes still get usable buffers.
const MIN_SORT_PAIRS: u32 = 1024;

/// Dispatch sizes for the hierarchical prefix sum over `num_elements` values.
///
/// Level 0 scans the elements themselves and emits one block sum per workgroup;
/// level 1 scans those block sums and emits one level-2 block sum per
/// workgroup; level 2 scans the level-2 block sums in a single workgroup.
/// Each level's block sums must be scanned **and added back**, otherwise every
/// element past the first block of that level is short by the missing offset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScanPlan {
    /// Number of elements scanned at level 0 (the Gaussian count).
    pub num_elements: u32,
    /// Workgroups dispatched for the level-0 scan (= number of level-1 block sums).
    pub level0_workgroups: u32,
    /// Workgroups dispatched for the level-1 scan (= number of level-2 block sums).
    pub level1_workgroups: u32,
    /// Workgroups dispatched for the level-2 scan (1 for any supported count).
    pub level2_workgroups: u32,
    /// Number of scan levels actually needed: 1, 2 or 3.
    pub levels: u32,
}

impl ScanPlan {
    /// Compute the dispatch plan for scanning `num_elements` values.
    #[must_use]
    pub fn new(num_elements: u32) -> Self {
        let level0_workgroups = num_elements.div_ceil(PREFIX_SUM_BLOCK).max(1);
        let level1_workgroups = level0_workgroups.div_ceil(PREFIX_SUM_BLOCK).max(1);
        let level2_workgroups = level1_workgroups.div_ceil(PREFIX_SUM_BLOCK).max(1);
        let levels = if level0_workgroups <= 1 {
            1
        } else if level1_workgroups <= 1 {
            2
        } else {
            3
        };
        Self {
            num_elements,
            level0_workgroups,
            level1_workgroups,
            level2_workgroups,
            levels,
        }
    }

    /// Whether this plan fits the three-level hierarchy the buffers implement.
    #[inline]
    #[must_use]
    pub fn is_supported(&self) -> bool {
        self.level2_workgroups <= 1
    }
}

/// Uniform data passed to all compute shaders.
///
/// Layout is carefully designed for GPU alignment (std140 rules):
/// - mat4x4: 64 bytes, 16-byte aligned
/// - vec3: 12 bytes + 4 padding = 16 bytes
/// - vec2: 8 bytes
/// - u32/f32: 4 bytes
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Uniforms {
    /// View matrix (4x4, column-major).
    pub view: [f32; 16],
    /// Projection matrix (4x4, column-major).
    pub proj: [f32; 16],
    /// Camera position in world space.
    pub cam_pos: [f32; 3],
    pub _pad0: f32,
    /// Focal length (fx, fy).
    pub focal: [f32; 2],
    /// Image dimensions (width, height).
    pub viewport: [f32; 2],
    /// Tile grid dimensions (tiles_x, tiles_y).
    pub tile_grid: [u32; 2],
    /// Number of Gaussians.
    pub num_gaussians: u32,
    /// SH degree.
    pub sh_degree: u32,
    /// Near plane.
    pub near_plane: f32,
    /// Far plane.
    pub far_plane: f32,
    /// Padding for WGSL vec3 alignment (vec3 has 16-byte alignment in uniform buffers).
    pub _pad_bg: [f32; 2],
    /// Background color.
    pub background: [f32; 3],
    /// Output flags bitmask:
    /// - Bit 0: output_depth
    /// - Bit 1: output_normals
    pub output_flags: u32,
    /// Transmittance threshold for early termination.
    pub transmittance_threshold: f32,
    /// Tile size (typically 16).
    pub tile_size: u32,
    /// Padding to ensure 16-byte alignment.
    pub _pad1: [u32; 2],
}

/// GPU buffers for Gaussian attributes (read in forward, gradients in backward).
///
/// Uses Structure-of-Arrays (SoA) layout for coalesced memory access:
/// - Positions: `[N × vec4]` - xyz + padding
/// - Rotations: `[N × vec4]` - quaternion xyzw
/// - Scales: `[N × vec4]` - xyz (log-scale) + padding
/// - Opacities: `[N × f32]` - sigmoid-inverse opacity
/// - SH coeffs: `[N × sh_count × 3]` - flat f32 array
pub struct GaussianBuffers {
    /// Positions `[N, 3]` as `[f32; 4]` with padding.
    pub positions: wgpu::Buffer,
    /// Rotation quaternions `[N, 4]`.
    pub rotations: wgpu::Buffer,
    /// Log-scales `[N, 3]` as `[f32; 4]` with padding.
    pub scales: wgpu::Buffer,
    /// Sigmoid-inverse opacities `[N]`.
    pub opacities: wgpu::Buffer,
    /// SH coefficients (flat f32 array).
    pub sh_coeffs: wgpu::Buffer,
    /// Number of Gaussians.
    pub count: u32,
}

/// Intermediate buffers used during rasterization.
///
/// These buffers store per-Gaussian computed data that flows between
/// shader stages:
/// - Preprocess -> Tile Assign: means2d, radii, depths, tile_counts
/// - Tile Assign -> Sort: sort_keys, sort_values
/// - Sort -> Tile Ranges -> Rasterize: sorted pairs, tile ranges
pub struct IntermediateBuffers {
    /// 2D covariance (upper triangle) `[N, 3]` as `(a, b, c)` for `[[a,b],[b,c]]`.
    pub cov2d: wgpu::Buffer,
    /// Screen-space means `[N, 2]`.
    pub means2d: wgpu::Buffer,
    /// View-space depths `[N]`.
    pub depths: wgpu::Buffer,
    /// Screen-space radii `[N]` (negative = culled).
    pub radii: wgpu::Buffer,
    /// Per-Gaussian tile overlap count `[N]`.
    pub tile_counts: wgpu::Buffer,
    /// Prefix sum of tile counts `[N]`.
    pub tile_offsets: wgpu::Buffer,
    /// Sort keys `(tile_id << 32 | depth_bits)` - sized to max_pairs.
    pub sort_keys: wgpu::Buffer,
    /// Sort values (Gaussian index) - sized to max_pairs.
    pub sort_values: wgpu::Buffer,
    /// Per-tile start/end ranges `[T, 2]`.
    pub tile_ranges: wgpu::Buffer,
    /// Evaluated RGB color per Gaussian `[N, 3]`.
    pub colors: wgpu::Buffer,
    /// Conic (inverse 2D covariance) `[N, 3]`.
    pub conics: wgpu::Buffer,
    /// Max number of Gaussian-tile pairs allocated.
    ///
    /// Only an estimate (see [`DEFAULT_PAIRS_PER_GAUSSIAN`]); the actual pair
    /// count must be checked with [`IntermediateBuffers::check_pair_capacity`]
    /// once the prefix sum has produced it.
    pub max_pairs: u32,
    /// Level-1 block sums for the hierarchical prefix sum
    /// (`scan_plan.level0_workgroups` elements).
    pub block_sums: wgpu::Buffer,
    /// Scanned level-1 block sums (output of the prefix sum over `block_sums`).
    pub block_sums_scanned: wgpu::Buffer,
    /// Level-2 block sums, emitted while scanning `block_sums`
    /// (`scan_plan.level1_workgroups` elements).
    ///
    /// Required whenever `scan_plan.levels == 3`: without scanning these and
    /// adding them back into `block_sums_scanned`, every Gaussian past the
    /// first `PREFIX_SUM_BLOCK²` receives a wrong tile offset.
    pub block_sums_l2: wgpu::Buffer,
    /// Scanned level-2 block sums.
    pub block_sums_l2_scanned: wgpu::Buffer,
    /// Dispatch plan for the hierarchical prefix sum over the Gaussian count.
    pub scan_plan: ScanPlan,
    /// Per-Gaussian normals (world space) `[N, 3]` - computed from rotation.
    /// This is the principal axis of the Gaussian ellipsoid.
    pub normals: wgpu::Buffer,
}

/// Output buffers from forward rasterization.
///
/// All buffers are `[H × W]` in row-major order (pixel_idx = y * W + x).
pub struct OutputBuffers {
    /// RGBA color image `[H, W, 4]`.
    pub color: wgpu::Buffer,
    /// Depth image `[H, W]` - alpha-weighted depth.
    pub depth: wgpu::Buffer,
    /// Final transmittance per pixel `[H, W]` - retained for backward.
    pub transmittance: wgpu::Buffer,
    /// Per-pixel count of contributing Gaussians (for backward).
    pub n_contrib: wgpu::Buffer,
    /// Normal image `[H, W, 3]` (optional, only if output_normals enabled).
    /// World-space normals weighted by alpha contribution.
    pub normals: Option<wgpu::Buffer>,
}

/// Gradient buffers (same layout as attribute buffers).
pub struct GradientBuffers {
    pub grad_positions: wgpu::Buffer,
    pub grad_rotations: wgpu::Buffer,
    pub grad_scales: wgpu::Buffer,
    pub grad_opacities: wgpu::Buffer,
    pub grad_sh_coeffs: wgpu::Buffer,
    // Intermediate 2D gradients - atomic buffers (written by rasterize_bwd)
    pub grad_means2d_atomic: wgpu::Buffer,
    pub grad_conics_atomic: wgpu::Buffer,
    pub grad_colors_atomic: wgpu::Buffer,
    // Intermediate 2D gradients - regular f32 buffers (read by preprocess_bwd)
    pub grad_means2d: wgpu::Buffer,
    pub grad_conics: wgpu::Buffer,
    pub grad_colors: wgpu::Buffer,
}

/// Uniform buffer on the GPU.
pub struct UniformBuffer {
    pub buffer: wgpu::Buffer,
}

impl UniformBuffer {
    pub fn new(device: &wgpu::Device) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("uniforms"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self { buffer }
    }

    pub fn update(&self, queue: &wgpu::Queue, uniforms: &Uniforms) {
        queue.write_buffer(&self.buffer, 0, bytemuck::bytes_of(uniforms));
    }
}

/// Host-side staging of one model's attribute buffers.
///
/// Shared by [`GaussianBuffers::from_model`] (which uploads it at creation)
/// and [`GaussianBuffers::write_model`] (which rewrites the existing
/// allocations in place), so the two can never disagree about padding or
/// element order.
struct PackedAttributes {
    positions: Vec<[f32; 4]>,
    rotations: Vec<[f32; 4]>,
    scales: Vec<[f32; 4]>,
    opacities: Vec<f32>,
    sh_coeffs: Vec<f32>,
}

impl PackedAttributes {
    fn of(model: &GaussianModel) -> Self {
        let n = model.len();
        Self {
            // Positions/scales are padded to vec4 for 16-byte alignment.
            positions: model
                .gaussians
                .iter()
                .map(|g| [g.position[0], g.position[1], g.position[2], 0.0])
                .collect(),
            rotations: model.gaussians.iter().map(|g| g.rotation).collect(),
            scales: model
                .gaussians
                .iter()
                .map(|g| [g.scale[0], g.scale[1], g.scale[2], 0.0])
                .collect(),
            opacities: model.gaussians.iter().map(|g| g.opacity).collect(),
            // A model without SH data still needs a non-empty, correctly
            // sized buffer: degree-0 black.
            sh_coeffs: if model.sh_coeffs.is_empty() {
                vec![0.0f32; n.max(1) * 3]
            } else {
                model.sh_coeffs.clone()
            },
        }
    }
}

impl GaussianBuffers {
    /// Upload Gaussian model data to the GPU.
    ///
    /// # Buffer Layout
    ///
    /// Positions, rotations, and scales are padded to vec4 for optimal
    /// GPU memory alignment (16 bytes per element).
    pub fn from_model(device: &wgpu::Device, model: &GaussianModel) -> Self {
        let n = model.len();
        let PackedAttributes {
            positions: positions_data,
            rotations: rotations_data,
            scales: scales_data,
            opacities: opacities_data,
            sh_coeffs: sh_data,
        } = PackedAttributes::of(model);

        let positions = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gaussian_positions"),
            contents: bytemuck::cast_slice(&positions_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let rotations = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gaussian_rotations"),
            contents: bytemuck::cast_slice(&rotations_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let scales = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gaussian_scales"),
            contents: bytemuck::cast_slice(&scales_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let opacities = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gaussian_opacities"),
            contents: bytemuck::cast_slice(&opacities_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let sh_coeffs = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gaussian_sh"),
            contents: bytemuck::cast_slice(&sh_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        Self {
            positions,
            rotations,
            scales,
            opacities,
            sh_coeffs,
            count: n as u32,
        }
    }

    /// Rewrite these buffers' *contents* from `model`, reusing the existing
    /// allocations.
    ///
    /// This is the in-place counterpart of [`from_model`](Self::from_model):
    /// a training loop that pushes updated parameters every step would
    /// otherwise destroy and recreate five GPU buffers (plus every buffer and
    /// bind group that depends on them) per step. The packing is shared with
    /// `from_model`, so the layout cannot drift.
    ///
    /// `model` must describe the same allocation the buffers were created for
    /// — same Gaussian count and same SH stride. Anything larger is rejected
    /// rather than silently truncated.
    ///
    /// # Errors
    ///
    /// * [`RenderError::MismatchedBufferSizes`] when `model.len()` differs
    ///   from [`count`](Self::count).
    /// * [`RenderError::BufferOverflow`] when the packed SH coefficients no
    ///   longer fit the allocated SH buffer.
    pub fn write_model(
        &self,
        queue: &wgpu::Queue,
        model: &GaussianModel,
    ) -> Result<(), RenderError> {
        if model.len() != self.count as usize {
            return Err(RenderError::MismatchedBufferSizes {
                expected: self.count as usize,
                actual: model.len(),
            });
        }
        let packed = PackedAttributes::of(model);

        let sh_bytes = std::mem::size_of_val(packed.sh_coeffs.as_slice()) as u64;
        if sh_bytes > self.sh_coeffs.size() {
            return Err(RenderError::BufferOverflow {
                buffer_name: "gaussian_sh".to_string(),
                max_size: self.sh_coeffs.size(),
                requested: sh_bytes,
            });
        }

        if !packed.positions.is_empty() {
            queue.write_buffer(&self.positions, 0, bytemuck::cast_slice(&packed.positions));
            queue.write_buffer(&self.rotations, 0, bytemuck::cast_slice(&packed.rotations));
            queue.write_buffer(&self.scales, 0, bytemuck::cast_slice(&packed.scales));
            queue.write_buffer(&self.opacities, 0, bytemuck::cast_slice(&packed.opacities));
        }
        if !packed.sh_coeffs.is_empty() {
            queue.write_buffer(&self.sh_coeffs, 0, bytemuck::cast_slice(&packed.sh_coeffs));
        }
        Ok(())
    }
}

impl IntermediateBuffers {
    /// Estimate the sort-buffer capacity for `n` Gaussians.
    ///
    /// `pairs_per_gaussian` is a heuristic; the result is clamped to
    /// `n × num_tiles` (a Gaussian cannot overlap more tiles than exist) and
    /// floored at 1024 pairs so degenerate scenes still allocate usable buffers.
    /// All arithmetic is done in `u64` to avoid wrapping.
    #[must_use]
    pub fn estimate_max_pairs(n: u32, pairs_per_gaussian: u32, num_tiles: u32) -> u32 {
        let estimate = u64::from(n).saturating_mul(u64::from(pairs_per_gaussian.max(1)));
        let upper_bound = u64::from(n).saturating_mul(u64::from(num_tiles.max(1)));
        let clamped = estimate.min(upper_bound).min(u64::from(u32::MAX));
        (clamped as u32).max(MIN_SORT_PAIRS)
    }

    /// Check that `n` Gaussians can be scanned by the hierarchical prefix sum.
    ///
    /// # Errors
    ///
    /// [`RenderError::TooManyGaussians`] when `n` exceeds [`MAX_SCAN_GAUSSIANS`].
    pub fn validate_gaussian_count(n: u32) -> Result<(), RenderError> {
        if !ScanPlan::new(n).is_supported() {
            return Err(RenderError::TooManyGaussians {
                count: n,
                max: MAX_SCAN_GAUSSIANS,
            });
        }
        Ok(())
    }

    /// Allocate intermediate buffers for `n` Gaussians, validating the count.
    ///
    /// Prefer this over [`allocate`](Self::allocate): it surfaces an
    /// unsupported Gaussian count as an error instead of silently producing a
    /// corrupt prefix sum.
    ///
    /// # Errors
    ///
    /// [`RenderError::TooManyGaussians`] when `n` exceeds [`MAX_SCAN_GAUSSIANS`].
    pub fn try_allocate(
        device: &wgpu::Device,
        n: u32,
        config: &RasterConfig,
    ) -> Result<Self, RenderError> {
        Self::validate_gaussian_count(n)?;
        Ok(Self::allocate_with_pairs_per_gaussian(
            device,
            n,
            config,
            DEFAULT_PAIRS_PER_GAUSSIAN,
        ))
    }

    /// Allocate intermediate buffers for `n` Gaussians.
    ///
    /// # Memory Allocation Strategy
    ///
    /// - `max_pairs` is estimated as `n × DEFAULT_PAIRS_PER_GAUSSIAN`, clamped
    ///   to `n × num_tiles` and floored at 1024 pairs
    /// - Block-sum buffers cover all three prefix-sum levels
    ///
    /// Callers that can handle failure should use [`try_allocate`](Self::try_allocate):
    /// this entry point can only log when `n` exceeds [`MAX_SCAN_GAUSSIANS`].
    pub fn allocate(device: &wgpu::Device, n: u32, config: &RasterConfig) -> Self {
        if let Err(e) = Self::validate_gaussian_count(n) {
            tracing::error!(
                num_gaussians = n,
                max = MAX_SCAN_GAUSSIANS,
                "{e}; tile offsets will be incorrect — use IntermediateBuffers::try_allocate to reject this up front"
            );
        }
        Self::allocate_with_pairs_per_gaussian(device, n, config, DEFAULT_PAIRS_PER_GAUSSIAN)
    }

    /// Allocate intermediate buffers with an explicit tiles-per-Gaussian estimate.
    ///
    /// Raising `pairs_per_gaussian` costs 12 bytes per pair across `sort_keys`
    /// and `sort_values`, so it is a real memory trade-off rather than free
    /// headroom.
    pub fn allocate_with_pairs_per_gaussian(
        device: &wgpu::Device,
        n: u32,
        config: &RasterConfig,
        pairs_per_gaussian: u32,
    ) -> Self {
        let num_tiles = config.num_tiles();
        let max_pairs = Self::estimate_max_pairs(n, pairs_per_gaussian, num_tiles);
        let scan_plan = ScanPlan::new(n);

        let storage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;

        // Ensure minimum size of 1 element for all buffers to avoid validation errors
        let n_safe = n.max(1) as u64;

        let cov2d = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cov2d"),
            size: n_safe * 4 * 4, // vec3<f32> with vec4 padding (16 bytes)
            usage: storage,
            mapped_at_creation: false,
        });
        let means2d = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("means2d"),
            size: n_safe * 2 * 4, // vec2<f32>
            usage: storage,
            mapped_at_creation: false,
        });
        let depths = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("depths"),
            size: n_safe * 4, // f32
            usage: storage,
            mapped_at_creation: false,
        });
        let radii = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("radii"),
            size: n_safe * 4, // i32
            usage: storage,
            mapped_at_creation: false,
        });
        let tile_counts = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tile_counts"),
            size: n_safe * 4, // u32
            usage: storage,
            mapped_at_creation: false,
        });
        let tile_offsets = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tile_offsets"),
            size: n_safe * 4, // u32
            usage: storage,
            mapped_at_creation: false,
        });
        let sort_keys = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sort_keys"),
            size: (max_pairs as u64) * 8, // vec2<u32> = u64 keys
            usage: storage,
            mapped_at_creation: false,
        });
        let sort_values = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sort_values"),
            size: (max_pairs as u64) * 4, // u32
            usage: storage,
            mapped_at_creation: false,
        });
        let tile_ranges = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tile_ranges"),
            size: (num_tiles.max(1) as u64) * 2 * 4, // vec2<u32>
            usage: storage,
            mapped_at_creation: false,
        });
        let colors = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("colors"),
            size: n_safe * 4 * 4, // vec3<f32> with vec4 padding (16 bytes)
            usage: storage,
            mapped_at_creation: false,
        });
        let conics = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("conics"),
            size: n_safe * 4 * 4, // vec3<f32> with vec4 padding (16 bytes)
            usage: storage,
            mapped_at_creation: false,
        });

        // Block sums for the hierarchical prefix sum.
        // Level 1: one entry per level-0 workgroup.
        // Level 2: one entry per level-1 workgroup (needed above
        // PREFIX_SUM_BLOCK² = 262,144 Gaussians, where the level-1 scan itself
        // spans more than one workgroup).
        let num_wg_l1 = u64::from(scan_plan.level0_workgroups);
        let num_wg_l2 = u64::from(scan_plan.level1_workgroups);
        let block_sums = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("block_sums"),
            size: num_wg_l1 * 4,
            usage: storage,
            mapped_at_creation: false,
        });
        let block_sums_scanned = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("block_sums_scanned"),
            size: num_wg_l1 * 4,
            usage: storage,
            mapped_at_creation: false,
        });
        let block_sums_l2 = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("block_sums_l2"),
            size: num_wg_l2 * 4,
            usage: storage,
            mapped_at_creation: false,
        });
        let block_sums_l2_scanned = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("block_sums_l2_scanned"),
            size: num_wg_l2 * 4,
            usage: storage,
            mapped_at_creation: false,
        });

        // Per-Gaussian normals (computed from rotation in preprocess)
        let normals = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gaussian_normals"),
            size: n_safe * 4 * 4, // vec3<f32> with vec4 padding (16 bytes)
            usage: storage,
            mapped_at_creation: false,
        });

        Self {
            cov2d,
            means2d,
            depths,
            radii,
            tile_counts,
            tile_offsets,
            sort_keys,
            sort_values,
            tile_ranges,
            colors,
            conics,
            max_pairs,
            block_sums,
            block_sums_scanned,
            block_sums_l2,
            block_sums_l2_scanned,
            scan_plan,
            normals,
        }
    }

    /// Check that `count` Gaussian-tile pairs fit in the allocated sort buffers.
    ///
    /// `tile_assign.wgsl` writes at `tile_offsets[i]`, which is derived from the
    /// prefix sum and is not bounded by `max_pairs`, so an overflowing frame
    /// silently loses pairs. Call this with the real total before dispatching.
    ///
    /// # Errors
    ///
    /// [`RenderError::TooManyTilePairs`] when `count` exceeds `max_pairs`.
    pub fn check_pair_capacity(&self, count: u32) -> Result<(), RenderError> {
        if count > self.max_pairs {
            return Err(RenderError::TooManyTilePairs {
                count,
                allocated: self.max_pairs,
            });
        }
        Ok(())
    }

    /// Read a single element of `tile_offsets` back to the CPU.
    ///
    /// This costs a submit and a device wait, so it belongs on a validation path
    /// (e.g. behind `gpu_debug`), not in the steady-state render loop.
    ///
    /// # Errors
    ///
    /// * [`RenderError::BufferOverflow`] when `index` is outside `tile_offsets`.
    /// * [`RenderError::BufferMapFailed`] when the staging buffer cannot be mapped.
    pub fn read_tile_offset(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        index: u32,
    ) -> Result<u32, RenderError> {
        let byte_offset = u64::from(index) * 4;
        if byte_offset + 4 > self.tile_offsets.size() {
            return Err(RenderError::BufferOverflow {
                buffer_name: "tile_offsets".to_string(),
                max_size: self.tile_offsets.size(),
                requested: byte_offset + 4,
            });
        }

        // Staging size honours both the copy and the map alignment requirements.
        let staging_size = wgpu::COPY_BUFFER_ALIGNMENT.max(wgpu::MAP_ALIGNMENT);
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tile_offset_readback"),
            size: staging_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("tile_offset_readback"),
        });
        encoder.copy_buffer_to_buffer(&self.tile_offsets, byte_offset, &staging, 0, 4u64);
        queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).ok();
        });
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        rx.recv()
            .map_err(|e| RenderError::BufferMapFailed {
                buffer_name: "tile_offsets".to_string(),
                error: format!("Channel recv failed: {e}"),
            })?
            .map_err(|e| RenderError::BufferMapFailed {
                buffer_name: "tile_offsets".to_string(),
                error: e.to_string(),
            })?;

        let data = slice
            .get_mapped_range()
            .map_err(|e| RenderError::BufferMapFailed {
                buffer_name: "tile_offsets".to_string(),
                error: format!("Mapped range failed: {e}"),
            })?;
        let value = match data.get(..4) {
            Some(chunk) => {
                let mut raw = [0u8; 4];
                raw.copy_from_slice(chunk);
                u32::from_le_bytes(raw)
            }
            None => 0,
        };
        drop(data);
        staging.unmap();

        Ok(value)
    }

    /// Read the total number of Gaussian-tile pairs produced by the prefix sum.
    ///
    /// Assumes `tile_offsets` holds an **inclusive** scan of `tile_counts`
    /// (as documented in `shaders/prefix_sum.wgsl`), so the total is the last
    /// element.
    ///
    /// # Errors
    ///
    /// Propagates the errors of [`read_tile_offset`](Self::read_tile_offset).
    pub fn read_pair_total(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        n: u32,
    ) -> Result<u32, RenderError> {
        if n == 0 {
            return Ok(0);
        }
        self.read_tile_offset(device, queue, n - 1)
    }

    /// Read the real pair total and verify it against the allocated capacity.
    ///
    /// Returns the total on success.
    ///
    /// # Errors
    ///
    /// [`RenderError::TooManyTilePairs`] when the sort buffers are too small,
    /// plus the readback errors of [`read_tile_offset`](Self::read_tile_offset).
    pub fn verify_pair_capacity(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        n: u32,
    ) -> Result<u32, RenderError> {
        let total = self.read_pair_total(device, queue, n)?;
        self.check_pair_capacity(total)?;
        Ok(total)
    }

    /// Grow the sort buffers so they can hold `required` Gaussian-tile pairs.
    ///
    /// Returns `true` when the buffers were reallocated, in which case the
    /// caller **must** rebuild every bind group referencing
    /// `sort_keys`/`sort_values` and re-record the frame from the `tile_assign`
    /// stage onward; the already-encoded commands still point at the old
    /// buffers.
    ///
    /// This is the recovery half of [`check_pair_capacity`](Self::check_pair_capacity)
    /// and has no in-tree caller yet: `Rasterizer::forward` borrows its
    /// intermediate buffers immutably for the whole frame, so adopting it means
    /// restructuring that borrow. Until then the supported response to an
    /// overflow is the [`RenderError::TooManyTilePairs`] returned by
    /// [`verify_pair_capacity`](Self::verify_pair_capacity).
    pub fn grow_sort_buffers(&mut self, device: &wgpu::Device, required: u32) -> bool {
        if required <= self.max_pairs {
            return false;
        }
        // Grow geometrically to avoid reallocating every frame.
        let new_max = required.max(self.max_pairs.saturating_mul(2));

        let storage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;

        self.sort_keys = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sort_keys"),
            size: u64::from(new_max) * 8, // vec2<u32> = u64 keys
            usage: storage,
            mapped_at_creation: false,
        });
        self.sort_values = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sort_values"),
            size: u64::from(new_max) * 4, // u32
            usage: storage,
            mapped_at_creation: false,
        });
        tracing::debug!(
            old_max_pairs = self.max_pairs,
            new_max_pairs = new_max,
            "Grew sort buffers"
        );
        self.max_pairs = new_max;
        true
    }
}

impl OutputBuffers {
    /// Allocate output framebuffer.
    ///
    /// # Optional Buffers
    ///
    /// - `normals`: Only allocated if `config.output_normals` is true
    pub fn allocate(device: &wgpu::Device, config: &RasterConfig) -> Self {
        let npx = config.num_pixels().max(1) as u64;
        let storage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;

        let color = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output_color"),
            size: npx * 4 * 4, // RGBA f32
            usage: storage,
            mapped_at_creation: false,
        });
        let depth = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output_depth"),
            size: npx * 4, // f32
            usage: storage,
            mapped_at_creation: false,
        });
        let transmittance = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output_transmittance"),
            size: npx * 4, // f32
            usage: storage,
            mapped_at_creation: false,
        });
        let n_contrib = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output_n_contrib"),
            size: npx * 4, // u32
            usage: storage,
            mapped_at_creation: false,
        });

        // Optional normals buffer
        let normals = if config.output_normals {
            Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("output_normals"),
                size: npx * 4 * 4, // vec4<f32> (xyz + padding for alignment)
                usage: storage,
                mapped_at_creation: false,
            }))
        } else {
            None
        };

        Self {
            color,
            depth,
            transmittance,
            n_contrib,
            normals,
        }
    }

    /// Check if normal output is enabled.
    #[inline]
    pub fn has_normals(&self) -> bool {
        self.normals.is_some()
    }
}

impl GradientBuffers {
    /// Allocate gradient buffers for `n` Gaussians.
    pub fn allocate(device: &wgpu::Device, n: u32, sh_coeffs_total: u32) -> Self {
        let storage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;

        let n_safe = n.max(1) as u64;

        let grad_positions = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("grad_positions"),
            size: n_safe * 4 * 4, // [f32; 4] padded
            usage: storage,
            mapped_at_creation: false,
        });
        let grad_rotations = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("grad_rotations"),
            size: n_safe * 4 * 4,
            usage: storage,
            mapped_at_creation: false,
        });
        let grad_scales = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("grad_scales"),
            size: n_safe * 4 * 4,
            usage: storage,
            mapped_at_creation: false,
        });
        let grad_opacities = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("grad_opacities"),
            size: n_safe * 4,
            usage: storage,
            mapped_at_creation: false,
        });
        let grad_sh_coeffs = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("grad_sh_coeffs"),
            size: (sh_coeffs_total.max(1) as u64) * 4,
            usage: storage,
            mapped_at_creation: false,
        });

        // Intermediate 2D gradient buffers - atomic versions (written by rasterize_bwd)
        let grad_means2d_atomic = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("grad_means2d_atomic"),
            size: n_safe * 2 * 4, // [atomic<u32>; 2] per Gaussian
            usage: storage,
            mapped_at_creation: false,
        });
        let grad_conics_atomic = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("grad_conics_atomic"),
            size: n_safe * 3 * 4, // [atomic<u32>; 3] per Gaussian
            usage: storage,
            mapped_at_creation: false,
        });
        let grad_colors_atomic = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("grad_colors_atomic"),
            size: n_safe * 3 * 4, // [atomic<u32>; 3] per Gaussian
            usage: storage,
            mapped_at_creation: false,
        });

        // Intermediate 2D gradient buffers - regular f32 versions (read by preprocess_bwd)
        let grad_means2d = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("grad_means2d"),
            size: n_safe * 2 * 4, // [f32; 2] per Gaussian
            usage: storage,
            mapped_at_creation: false,
        });
        let grad_conics = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("grad_conics"),
            size: n_safe * 3 * 4, // [f32; 3] per Gaussian
            usage: storage,
            mapped_at_creation: false,
        });
        let grad_colors = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("grad_colors"),
            size: n_safe * 3 * 4, // [f32; 3] per Gaussian
            usage: storage,
            mapped_at_creation: false,
        });

        Self {
            grad_positions,
            grad_rotations,
            grad_scales,
            grad_opacities,
            grad_sh_coeffs,
            grad_means2d_atomic,
            grad_conics_atomic,
            grad_colors_atomic,
            grad_means2d,
            grad_conics,
            grad_colors,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Hierarchical prefix-sum plan ---

    #[test]
    fn test_scan_plan_single_level() {
        let plan = ScanPlan::new(500);
        assert_eq!(plan.level0_workgroups, 1);
        assert_eq!(plan.levels, 1);
        assert!(plan.is_supported());
    }

    #[test]
    fn test_scan_plan_two_levels() {
        // 100k Gaussians → 196 level-0 workgroups → a single level-1 workgroup.
        let plan = ScanPlan::new(100_000);
        assert_eq!(plan.level0_workgroups, 196);
        assert_eq!(plan.level1_workgroups, 1);
        assert_eq!(plan.levels, 2);
        assert!(plan.is_supported());
    }

    #[test]
    fn test_scan_plan_two_levels_at_exact_boundary() {
        // 512 × 512 = 262,144 is the largest count a two-level scan covers.
        let plan = ScanPlan::new(PREFIX_SUM_BLOCK * PREFIX_SUM_BLOCK);
        assert_eq!(plan.level0_workgroups, PREFIX_SUM_BLOCK);
        assert_eq!(plan.level1_workgroups, 1);
        assert_eq!(plan.levels, 2);
    }

    /// Regression: above 262,144 Gaussians the level-1 scan spans more than one
    /// workgroup, so its own block totals must be scanned and added back.
    #[test]
    fn test_scan_plan_requires_third_level_above_262144() {
        let plan = ScanPlan::new(PREFIX_SUM_BLOCK * PREFIX_SUM_BLOCK + 1);
        assert_eq!(plan.level0_workgroups, PREFIX_SUM_BLOCK + 1);
        assert_eq!(plan.level1_workgroups, 2);
        assert_eq!(plan.levels, 3);
        assert!(plan.is_supported());
    }

    #[test]
    fn test_scan_plan_three_levels_for_typical_scene() {
        // 1M Gaussians: a realistic 3DGS scene needs the third level.
        let plan = ScanPlan::new(1_000_000);
        assert_eq!(plan.levels, 3);
        assert!(plan.level1_workgroups > 1);
        assert_eq!(plan.level2_workgroups, 1);
        assert!(plan.is_supported());
    }

    #[test]
    fn test_scan_plan_zero_elements() {
        let plan = ScanPlan::new(0);
        assert_eq!(plan.level0_workgroups, 1);
        assert_eq!(plan.levels, 1);
        assert!(plan.is_supported());
    }

    #[test]
    fn test_validate_gaussian_count_accepts_supported_counts() {
        assert!(IntermediateBuffers::validate_gaussian_count(0).is_ok());
        assert!(IntermediateBuffers::validate_gaussian_count(1_000_000).is_ok());
        assert!(IntermediateBuffers::validate_gaussian_count(MAX_SCAN_GAUSSIANS).is_ok());
    }

    #[test]
    fn test_validate_gaussian_count_rejects_beyond_three_levels() {
        let err = IntermediateBuffers::validate_gaussian_count(MAX_SCAN_GAUSSIANS + 1)
            .expect_err("beyond the three-level scan must be rejected");
        assert!(matches!(
            err,
            RenderError::TooManyGaussians {
                max: MAX_SCAN_GAUSSIANS,
                ..
            }
        ));
    }

    // --- Sort-buffer capacity ---

    #[test]
    fn test_estimate_max_pairs_uses_heuristic() {
        // 10,000 Gaussians × 4 tiles, plenty of tiles available.
        assert_eq!(
            IntermediateBuffers::estimate_max_pairs(10_000, DEFAULT_PAIRS_PER_GAUSSIAN, 4096),
            40_000
        );
    }

    #[test]
    fn test_estimate_max_pairs_has_minimum() {
        assert_eq!(
            IntermediateBuffers::estimate_max_pairs(10, DEFAULT_PAIRS_PER_GAUSSIAN, 4096),
            MIN_SORT_PAIRS
        );
    }

    #[test]
    fn test_estimate_max_pairs_clamped_by_tile_count() {
        // With a single tile a Gaussian can produce at most one pair.
        assert_eq!(
            IntermediateBuffers::estimate_max_pairs(10_000, DEFAULT_PAIRS_PER_GAUSSIAN, 1),
            10_000
        );
    }

    /// Regression: the estimate must not wrap u32 for large scenes.
    #[test]
    fn test_estimate_max_pairs_does_not_overflow() {
        let pairs = IntermediateBuffers::estimate_max_pairs(2_000_000_000, 8, 100_000);
        assert_eq!(pairs, u32::MAX);
    }

    // --- Attribute packing (shared by from_model and write_model) ---

    fn packing_model(n: usize) -> GaussianModel {
        use crate::gaussian::GaussianAttributes;
        GaussianModel {
            gaussians: (0..n)
                .map(|i| GaussianAttributes {
                    position: [i as f32, i as f32 + 0.5, i as f32 + 0.25],
                    _pad0: 0.0,
                    rotation: [0.1, 0.2, 0.3, 0.4],
                    scale: [-1.0, -2.0, -3.0],
                    opacity: 0.75,
                })
                .collect(),
            sh_coeffs: vec![1.0; n * 3],
            sh_degree: 0,
            face_indices: vec![0; n],
            barycentric: vec![[1.0, 0.0, 0.0]; n],
            local_offsets: vec![[0.0; 3]; n],
            is_rigid: vec![false; n],
        }
    }

    /// `write_model` reuses this packing, so the vec4 padding and element
    /// order it produces must be exactly what `from_model` uploaded.
    #[test]
    fn test_packed_attributes_pads_to_vec4() {
        let model = packing_model(2);
        let packed = PackedAttributes::of(&model);
        assert_eq!(packed.positions.len(), 2);
        assert_eq!(packed.positions[1], [1.0, 1.5, 1.25, 0.0]);
        assert_eq!(packed.scales[0], [-1.0, -2.0, -3.0, 0.0]);
        assert_eq!(packed.rotations[0], [0.1, 0.2, 0.3, 0.4]);
        assert_eq!(packed.opacities, vec![0.75, 0.75]);
        assert_eq!(packed.sh_coeffs.len(), 6);
    }

    /// A model with no SH data must still pack a correctly sized buffer:
    /// a zero-length storage buffer is a wgpu validation error, and an
    /// in-place rewrite must not shrink what `from_model` allocated.
    #[test]
    fn test_packed_attributes_substitutes_missing_sh() {
        let mut model = packing_model(4);
        model.sh_coeffs.clear();
        let packed = PackedAttributes::of(&model);
        assert_eq!(packed.sh_coeffs.len(), 4 * 3);
        assert!(packed.sh_coeffs.iter().all(|v| *v == 0.0));

        // ...including for an empty model, where the buffer still needs one
        // element of room.
        let empty = packing_model(0);
        assert_eq!(PackedAttributes::of(&empty).sh_coeffs.len(), 3);
    }
}
