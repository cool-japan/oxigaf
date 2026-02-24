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
    pub max_pairs: u32,
    /// Block sums for hierarchical prefix sum (num_workgroups elements).
    pub block_sums: wgpu::Buffer,
    /// Scanned block sums (output of prefix sum on block_sums).
    pub block_sums_scanned: wgpu::Buffer,
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

impl GaussianBuffers {
    /// Upload Gaussian model data to the GPU.
    ///
    /// # Buffer Layout
    ///
    /// Positions, rotations, and scales are padded to vec4 for optimal
    /// GPU memory alignment (16 bytes per element).
    pub fn from_model(device: &wgpu::Device, model: &GaussianModel) -> Self {
        let n = model.len();

        // Positions: extract from GaussianAttributes, pad to [f32; 4]
        let positions_data: Vec<[f32; 4]> = model
            .gaussians
            .iter()
            .map(|g| [g.position[0], g.position[1], g.position[2], 0.0])
            .collect();

        let rotations_data: Vec<[f32; 4]> = model.gaussians.iter().map(|g| g.rotation).collect();

        let scales_data: Vec<[f32; 4]> = model
            .gaussians
            .iter()
            .map(|g| [g.scale[0], g.scale[1], g.scale[2], 0.0])
            .collect();

        let opacities_data: Vec<f32> = model.gaussians.iter().map(|g| g.opacity).collect();

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

        // SH coefficients: ensure minimum size
        let sh_data = if model.sh_coeffs.is_empty() {
            vec![0.0f32; n.max(1) * 3]
        } else {
            model.sh_coeffs.clone()
        };

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
}

impl IntermediateBuffers {
    /// Allocate intermediate buffers for `n` Gaussians.
    ///
    /// # Memory Allocation Strategy
    ///
    /// - `max_pairs` is estimated as `n × avg_tiles_per_gaussian`
    /// - Uses 4 tiles per Gaussian as heuristic (covers most scenes)
    /// - Minimum allocation of 1024 pairs to avoid edge cases
    pub fn allocate(device: &wgpu::Device, n: u32, config: &RasterConfig) -> Self {
        let avg_tiles_per_gaussian = 4u32; // heuristic
        let max_pairs = n.saturating_mul(avg_tiles_per_gaussian).max(1024);
        let num_tiles = config.num_tiles();

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

        // Block sums for hierarchical prefix sum
        let num_wg = n.div_ceil(512).max(1);
        let block_sums = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("block_sums"),
            size: (num_wg as u64) * 4,
            usage: storage,
            mapped_at_creation: false,
        });
        let block_sums_scanned = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("block_sums_scanned"),
            size: (num_wg as u64) * 4,
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
            normals,
        }
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
