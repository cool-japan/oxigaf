//! Main rasterizer: orchestrates the full forward and backward passes.
//!
//! # Per-frame cost
//!
//! Everything that can be allocated once is allocated once. Uniform buffers,
//! placeholder buffers and every bind group are created in
//! [`Rasterizer::from_device`] or [`Rasterizer::upload_gaussians`] (the only
//! point where the GPU buffers they reference can change) and only their
//! *contents* are rewritten per frame with `queue.write_buffer`.
//!
//! The sort buffers are not pre-filled from the host either: the exact number
//! of Gaussian-tile pairs is read back from the prefix sum (four bytes) and
//! used to bound the sort, the tile-range scan and the tile-range dispatch, so
//! no stage ever looks at an entry that `tile_assign` did not write.
//!
//! # Layout
//!
//! * `limits` — the device limits and shader-geometry constants the
//!   pipelines require, plus the checks that reject a device or a config that
//!   cannot host them.
//! * `bind_groups` — the per-upload bind-group tables, which must match
//!   `pipeline.rs`'s layouts entry for entry.
//! * this module — the [`Rasterizer`] itself: buffer residency, the forward
//!   and backward pass recording, and the readback plumbing.

mod bind_groups;
mod limits;

use limits::{check_linear_workgroup_size, check_rasterizer_limits};
pub use limits::{
    rasterizer_device_limits, RASTERIZER_STORAGE_BUFFERS_PER_STAGE,
    RASTERIZE_FWD_WORKGROUP_STORAGE_BYTES, RASTERIZE_TILE_SIZE,
};

use crate::binding::{build_face_frames, FaceFrame, FlameBindingBuffers};
use crate::buffers::{
    GaussianBuffers, GradientBuffers, IntermediateBuffers, OutputBuffers, UniformBuffer, Uniforms,
};
use crate::config::RasterConfig;
use crate::gaussian::GaussianModel;
use crate::pipeline::RasterPipelines;
use crate::pool::{BufferPool, PoolStats, PooledBuffer};
use crate::profiler::GpuTimestampProfiler;
use crate::sort::RadixSorter;
use crate::workgroup::WorkgroupConfig;
use crate::RenderError;

/// Output of a forward rasterization pass.
pub struct RenderOutput {
    /// RGBA color data `[H * W * 4]` as f32.
    pub color_data: Vec<f32>,
    /// Depth data `[H * W]`.
    pub depth_data: Vec<f32>,
    /// Normal data `[H * W * 3]` as `[f32; 3]` (world-space, normalized).
    /// Only present if `config.output_normals` was enabled.
    pub normals: Option<Vec<[f32; 3]>>,
    /// Image width.
    pub width: u32,
    /// Image height.
    pub height: u32,
}

/// Gradients for all Gaussian attributes (result of backward pass).
#[derive(Debug)]
pub struct GaussianGradients {
    pub grad_positions: Vec<[f32; 3]>,
    pub grad_rotations: Vec<[f32; 4]>,
    pub grad_scales: Vec<[f32; 3]>,
    pub grad_opacities: Vec<f32>,
    pub grad_sh_coeffs: Vec<f32>,
    /// `∂L/∂local_offset`, one entry per Gaussian.
    ///
    /// This is `∂L/∂position` projected onto the TBN frame of the FLAME face
    /// each Gaussian is bound to — the exact adjoint of
    /// `binding::apply_binding`'s `position = surface(face, bary) + T·o.x +
    /// B·o.y + N·o.z`. Without it the photometric loss cannot reach the
    /// learnable local offsets at all and that parameter group stays frozen.
    ///
    /// **Empty** unless a binding frame table was installed with
    /// [`Rasterizer::set_binding_mesh`] (or
    /// [`Rasterizer::set_binding_frames`]) before the backward pass; see
    /// those methods for why the rasterizer cannot derive one on its own.
    pub grad_local_offsets: Vec<[f32; 3]>,
}

/// Camera parameters for rendering.
#[derive(Debug, Clone)]
pub struct RenderCamera {
    /// View matrix (world to camera, 4×4 column-major).
    pub view_matrix: [f32; 16],
    /// Projection matrix (4×4 column-major).
    pub proj_matrix: [f32; 16],
    /// Camera position in world space.
    pub position: [f32; 3],
    /// Focal lengths (fx, fy) in pixels.
    pub focal: [f32; 2],
}

/// Cheap identity of the [`GaussianModel`] currently resident on the GPU.
///
/// # Limits
///
/// This identifies the model *allocation*, not its contents: mutating a model
/// in place (the training loop's densification and parameter updates) leaves
/// the identity unchanged, and [`Rasterizer::forward`] will keep rendering the
/// data uploaded earlier. Callers that mutate in place must call
/// [`Rasterizer::upload_gaussians`] (as the trainer does every step) or
/// [`Rasterizer::invalidate_gaussians`].
///
/// What it *does* catch is the case the API previously got silently wrong:
/// calling `forward` with a **different** model than the one that was uploaded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct UploadedModel {
    /// Address of the Gaussian attribute allocation.
    ptr: usize,
    /// Number of Gaussians.
    len: usize,
    /// Number of SH coefficient floats.
    sh_len: usize,
    /// SH degree the buffers were sized for.
    sh_degree: u32,
}

impl UploadedModel {
    fn of(model: &GaussianModel) -> Self {
        Self {
            ptr: model.gaussians.as_ptr() as usize,
            len: model.gaussians.len(),
            sh_len: model.sh_coeffs.len(),
            sh_degree: model.sh_degree,
        }
    }

    /// Whether the resident GPU buffers were uploaded from exactly `model`.
    fn describes(&self, model: &GaussianModel) -> bool {
        self.ptr == model.gaussians.as_ptr() as usize && self.describes_allocation(model)
    }

    /// Whether the resident buffers have the right *shape* for `model`, i.e.
    /// whether its contents can be written into them without reallocating.
    ///
    /// Deliberately ignores the backing pointer: a model that was reallocated
    /// (or a different model of identical shape) still fits the same buffers,
    /// and [`Rasterizer::update_gaussians`] rewrites their contents anyway.
    fn describes_allocation(&self, model: &GaussianModel) -> bool {
        self.len == model.gaussians.len()
            && self.sh_len == model.sh_coeffs.len()
            && self.sh_degree == model.sh_degree
    }
}

/// Persistent 16-byte uniform buffers whose contents are rewritten each frame.
struct FrameParams {
    /// `vec4<u32>` holding the Gaussian count (level-0 scan and its add-back).
    scan_count: wgpu::Buffer,
    /// `vec4<u32>` holding the level-0 workgroup count (level-1 scan and add).
    scan_l1: wgpu::Buffer,
    /// `vec4<u32>` holding the level-1 workgroup count (level-2 scan).
    scan_l2: wgpu::Buffer,
    /// `vec4<u32>` holding `(pair_count, num_tiles)`.
    tile_ranges: wgpu::Buffer,
    /// Element counts for the atomic→f32 conversion passes.
    num_elements_means2d: wgpu::Buffer,
    num_elements_conics: wgpu::Buffer,
    num_elements_colors: wgpu::Buffer,
}

/// Forward-pass bind groups; rebuilt whenever the GPU buffers are reallocated.
struct FrameBindGroups {
    preprocess: wgpu::BindGroup,
    scan_l0: wgpu::BindGroup,
    scan_l1: wgpu::BindGroup,
    scan_l2: wgpu::BindGroup,
    add_l1: wgpu::BindGroup,
    add_l0: wgpu::BindGroup,
    tile_assign: wgpu::BindGroup,
    tile_ranges: wgpu::BindGroup,
    rasterize_fwd: wgpu::BindGroup,
}

/// Backward-pass bind groups; rebuilt alongside [`FrameBindGroups`].
struct BackwardBindGroups {
    rasterize_bwd: wgpu::BindGroup,
    atomic_means2d: wgpu::BindGroup,
    atomic_conics: wgpu::BindGroup,
    atomic_colors: wgpu::BindGroup,
    preprocess_bwd: wgpu::BindGroup,
}

/// A staging buffer, either borrowed from the pool or freshly created.
enum StagingBuffer {
    Pooled(PooledBuffer),
    Direct(wgpu::Buffer),
}

impl std::ops::Deref for StagingBuffer {
    type Target = wgpu::Buffer;

    fn deref(&self) -> &wgpu::Buffer {
        match self {
            Self::Pooled(b) => b,
            Self::Direct(b) => b,
        }
    }
}

/// A readback whose `copy_buffer_to_buffer` has been recorded but not yet
/// submitted, so several can share one submission and one device wait.
struct PendingReadback {
    staging: StagingBuffer,
    count: usize,
}

/// Label carried by every readback staging buffer, in the pool and in the
/// error a refused allocation produces.
const STAGING_READ_LABEL: &str = "staging_read";

/// Refuse a staging allocation the device could not hold.
///
/// [`BufferPool::acquire`](crate::pool::BufferPool::acquire) returns `None`
/// both for a poisoned pool mutex (retryable by allocating directly) and for
/// a request above `max_buffer_size` (not retryable at all: handing that size
/// to `create_buffer` raises a wgpu validation error, which is delivered to
/// the *uncaptured* error handler and aborts the process instead of returning
/// through any `Result`). The direct fallback therefore has to re-apply the
/// limit itself rather than assume the request was merely unpooled.
///
/// Bytes to allocate for a staging buffer that will receive `byte_size`
/// bytes: never zero, and always a multiple of [`wgpu::MAP_ALIGNMENT`] so the
/// buffer can be mapped whatever the float count is.
fn staging_size_for(byte_size: u64) -> u64 {
    byte_size
        .max(wgpu::MAP_ALIGNMENT)
        .next_multiple_of(wgpu::MAP_ALIGNMENT)
}

/// Pure function of its inputs so the policy is unit-testable without a GPU.
fn check_staging_fits(staging_size: u64, max_buffer_size: u64) -> Result<(), RenderError> {
    if staging_size > max_buffer_size {
        return Err(RenderError::BufferOverflow {
            buffer_name: STAGING_READ_LABEL.to_string(),
            max_size: max_buffer_size,
            requested: staging_size,
        });
    }
    Ok(())
}

/// The GPU-accelerated 3DGS rasterizer.
pub struct Rasterizer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: RasterConfig,
    workgroups: WorkgroupConfig,
    pipelines: RasterPipelines,
    sorter: RadixSorter,
    uniform_buf: UniformBuffer,
    gaussian_bufs: Option<GaussianBuffers>,
    intermediate_bufs: Option<IntermediateBuffers>,
    output_bufs: Option<OutputBuffers>,
    gradient_bufs: Option<GradientBuffers>,
    /// Buffer pool for efficient memory reuse.
    buffer_pool: Option<BufferPool>,
    /// Identity of the model currently resident on the GPU.
    uploaded: Option<UploadedModel>,
    /// Persistent per-pass uniform buffers.
    params: FrameParams,
    /// Sink for the block sums of the top scan level (written, never read).
    dummy_block_sums: wgpu::Buffer,
    /// Placeholder bound to `out_normals` when normal output is disabled.
    dummy_out_normals: wgpu::Buffer,
    /// Image-space gradient upload target for `backward()` (sized by config).
    grad_output_buf: wgpu::Buffer,
    frame_bgs: Option<FrameBindGroups>,
    bwd_bgs: Option<BackwardBindGroups>,
    /// FLAME binding frame table (see [`Rasterizer::set_binding_mesh`]).
    ///
    /// Indexed by `GaussianModel::face_indices`, so entry `k` is the frame of
    /// mesh **face** `k` — exactly the frame `binding::apply_binding` applies
    /// the local offset in.
    binding_frames: Option<Vec<FaceFrame>>,
    /// GPU buffers for the FLAME binding backward pass, allocated on demand
    /// and reused while the Gaussian count and frame table length hold.
    flame_bufs: Option<FlameBindingBuffers>,
    /// Optional GPU-timestamp profiler; `None` until
    /// [`Rasterizer::enable_gpu_timestamps`] succeeds.
    gpu_timestamps: Option<GpuTimestampProfiler>,
}

impl Rasterizer {
    /// Create a new rasterizer from existing wgpu device and queue.
    ///
    /// # Errors
    ///
    /// * [`RenderError::Rasterize`] for any configuration
    ///   [`RasterConfig::validate_for_rasterizer`] rejects — a zero
    ///   `tile_size` (which would otherwise panic inside
    ///   [`RasterConfig::tiles_x`]), an `sh_degree` above 3, an overflowing
    ///   pixel count, an inverted clip range, or a `tile_size` other than
    ///   [`RASTERIZE_TILE_SIZE`] (the rasterization shaders hardcode that tile
    ///   size in their `@workgroup_size` attribute).
    /// * [`RenderError::GpuInit`] when `device` was created with limits too
    ///   small for the rasterizer's pipelines — see
    ///   [`rasterizer_device_limits`].
    /// * Whatever [`RasterPipelines::new`] reports for a shader that fails to
    ///   compile or validate.
    pub fn from_device(
        device: wgpu::Device,
        queue: wgpu::Queue,
        config: RasterConfig,
    ) -> Result<Self, RenderError> {
        // Every consistency check up front: a deserialized or hand-built
        // config must fail here with a clear message rather than panic inside
        // `tiles_x()` or silently desync the tile grid from the shader's
        // workgroup grid.
        config.validate_for_rasterizer()?;
        check_rasterizer_limits(&device.limits(), "device")?;

        // The 1-D kernels are compiled with this `@workgroup_size`, so the
        // dispatch grid must be derived from the same number. Validate it
        // before compiling: an oversized or non-power-of-two request should
        // fail with a clear message, not as a shader-compilation error.
        let linear_workgroup_size = config.effective_preprocess_wg_size();
        check_linear_workgroup_size(&device.limits(), linear_workgroup_size)?;
        let workgroups = WorkgroupConfig::for_linear_size(linear_workgroup_size);
        workgroups.validate()?;

        let pipelines = RasterPipelines::new(&device, &config)?;
        debug_assert_eq!(pipelines.linear_workgroup_size, linear_workgroup_size);
        // Start small: `upload_gaussians` grows the sorter to the real pair
        // capacity, so there is no point reserving for a scene we may not get.
        let sorter = RadixSorter::new(&device, 1024)?;
        let uniform_buf = UniformBuffer::new(&device);

        // Scoped so the borrow of `device` taken by the helper ends before
        // `device` is moved into `Self`.
        let params = {
            let uniform_usage = wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST;
            let small_uniform = |label: &'static str| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(label),
                    size: 16, // vec4<u32>
                    usage: uniform_usage,
                    mapped_at_creation: false,
                })
            };
            FrameParams {
                scan_count: small_uniform("prefix_sum_params"),
                scan_l1: small_uniform("prefix_sum_l1_params"),
                scan_l2: small_uniform("prefix_sum_l2_params"),
                tile_ranges: small_uniform("tile_ranges_params"),
                num_elements_means2d: small_uniform("num_elements_means2d"),
                num_elements_conics: small_uniform("num_elements_conics"),
                num_elements_colors: small_uniform("num_elements_colors"),
            }
        };

        let dummy_block_sums = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dummy_block_sums"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let dummy_out_normals = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dummy_out_normals"),
            size: 16, // minimum size
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let grad_output_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("grad_output"),
            size: (u64::from(config.num_pixels().max(1))) * 4 * 4, // RGBA f32
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create buffer pool if enabled
        let buffer_pool = if config.enable_buffer_pooling && config.max_gpu_memory_mb > 0 {
            let budget_bytes = config.memory_budget_bytes();
            tracing::info!(
                budget_mb = config.max_gpu_memory_mb,
                "Buffer pooling enabled"
            );
            Some(BufferPool::new(budget_bytes))
        } else {
            tracing::info!("Buffer pooling disabled");
            None
        };

        Ok(Self {
            device,
            queue,
            config,
            workgroups,
            pipelines,
            sorter,
            uniform_buf,
            gaussian_bufs: None,
            intermediate_bufs: None,
            output_bufs: None,
            gradient_bufs: None,
            buffer_pool,
            uploaded: None,
            params,
            dummy_block_sums,
            dummy_out_normals,
            grad_output_buf,
            frame_bgs: None,
            bwd_bgs: None,
            binding_frames: None,
            flame_bufs: None,
            gpu_timestamps: None,
        })
    }

    /// Create a rasterizer by requesting a GPU device.
    ///
    /// This is async because wgpu device creation is async.
    ///
    /// # GPU Debug Mode
    ///
    /// When the `gpu_debug` feature is enabled, this enables:
    /// - Vulkan validation layers
    /// - Metal API validation
    /// - DirectX debug layer
    /// - Enhanced error messages
    ///
    /// Note: Debug mode adds significant runtime overhead.
    pub async fn new(config: RasterConfig) -> Result<Self, RenderError> {
        #[cfg(feature = "gpu_debug")]
        let instance = {
            tracing::info!("GPU debug mode enabled - validation layers active");
            wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends: wgpu::Backends::all(),
                flags: wgpu::InstanceFlags::VALIDATION | wgpu::InstanceFlags::DEBUG,
                ..wgpu::InstanceDescriptor::new_without_display_handle()
            })
        };

        #[cfg(not(feature = "gpu_debug"))]
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
                apply_limit_buckets: false,
            })
            .await
            .map_err(|e| RenderError::GpuInit(format!("No suitable GPU adapter: {e}")))?;

        tracing::info!(
            adapter = adapter.get_info().name,
            backend = ?adapter.get_info().backend,
            "Selected GPU adapter"
        );

        // The adapter's own ceiling decides whether this GPU can host the
        // rasterizer at all. Checking it here turns "device creation failed"
        // (or, worse, an opaque pipeline-validation error much later) into a
        // message naming the limit that is too small.
        check_rasterizer_limits(&adapter.limits(), &adapter.get_info().name)?;
        let required_limits = rasterizer_device_limits(wgpu::Limits::default());

        // Timestamp queries are optional: ask for them only when the adapter
        // advertises support, so a GPU without them still gets a device.
        let required_features =
            adapter.features() & crate::profiler::GpuTimestampProfiler::REQUIRED_FEATURES;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("oxigaf_render"),
                required_features,
                required_limits,
                memory_hints: wgpu::MemoryHints::Performance,
                experimental_features: wgpu::ExperimentalFeatures::default(),
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| RenderError::GpuInit(e.to_string()))?;

        Self::from_device(device, queue, config)
    }

    /// Upload Gaussian model data to the GPU.
    ///
    /// Reallocates every per-model buffer, re-prepares the radix sorter and
    /// rebuilds all bind groups, so it is also the point that makes an in-place
    /// model edit visible to the GPU.
    pub fn upload_gaussians(&mut self, model: &GaussianModel) {
        let n = model.len() as u32;
        if let Err(e) = IntermediateBuffers::validate_gaussian_count(n) {
            tracing::error!(num_gaussians = n, "{e}");
        }
        // Defence in depth: the per-Gaussian side arrays are only read by the
        // deform / LOD / density paths, so a model whose FLAME arrays are out
        // of step with `gaussians` renders fine here and misbehaves much
        // later. Report it at the point the model enters the GPU.
        if let Err(e) = model.validate() {
            tracing::error!(num_gaussians = n, "upload_gaussians: {e}");
        }

        let gauss = GaussianBuffers::from_model(&self.device, model);
        let inter = IntermediateBuffers::allocate(&self.device, n, &self.config);
        let output = OutputBuffers::allocate(&self.device, &self.config);
        let grads =
            GradientBuffers::allocate(&self.device, n, n * self.config.sh_coeffs_per_gaussian());

        // The sorter's scratch/histogram buffers are sized from the real pair
        // capacity, not a hardcoded constant, and its bind groups are rebound to
        // the freshly allocated sort buffers.
        self.sorter.prepare(
            &self.device,
            &inter.sort_keys,
            &inter.sort_values,
            inter.max_pairs,
        );

        let frame_bgs = self.build_frame_bind_groups(&gauss, &inter, &output);
        let bwd_bgs = self.build_backward_bind_groups(&gauss, &inter, &output, &grads);

        self.gaussian_bufs = Some(gauss);
        self.intermediate_bufs = Some(inter);
        self.output_bufs = Some(output);
        self.gradient_bufs = Some(grads);
        self.frame_bgs = Some(frame_bgs);
        self.bwd_bgs = Some(bwd_bgs);
        self.uploaded = Some(UploadedModel::of(model));

        tracing::debug!(n_gaussians = n, "Uploaded Gaussians to GPU");
    }

    /// Rewrite the resident Gaussian attributes **in place**.
    ///
    /// A training step mutates the model's positions, rotations, scales,
    /// opacities and SH coefficients but almost never its *shape*.
    /// [`upload_gaussians`](Self::upload_gaussians) nonetheless destroys and
    /// recreates every per-model buffer (five attribute buffers, ~17
    /// intermediate buffers, four output buffers, eleven gradient buffers),
    /// re-prepares the radix sorter and rebuilds fourteen bind groups — about
    /// 33 GPU allocations per step, none of which the step needed.
    ///
    /// This entry point writes the five attribute buffers with
    /// `queue.write_buffer` and keeps everything else, falling back to a full
    /// [`upload_gaussians`](Self::upload_gaussians) only when the *allocation*
    /// actually changed: a different Gaussian count, SH coefficient count or
    /// SH degree (all of which resize buffers and invalidate bind groups).
    /// Densification therefore still reallocates, exactly once, on the step
    /// that changes the count.
    ///
    /// Unlike [`forward`](Self::forward)'s automatic re-upload, this always
    /// rewrites the contents, so an in-place model edit is guaranteed visible
    /// to the next render (see `UploadedModel`'s note on why the residency
    /// check alone cannot see such an edit).
    pub fn update_gaussians(&mut self, model: &GaussianModel) {
        let same_allocation = self.uploaded.is_some_and(|u| u.describes_allocation(model))
            && self.gaussian_bufs.is_some();
        if !same_allocation {
            self.upload_gaussians(model);
            return;
        }

        if let Err(e) = model.validate() {
            tracing::error!(num_gaussians = model.len(), "update_gaussians: {e}");
        }

        let write = self
            .gaussian_bufs
            .as_ref()
            .map(|gauss| gauss.write_model(&self.queue, model));
        match write {
            Some(Ok(())) => {
                // Refresh the identity: the allocation is unchanged but the
                // model's backing pointer may differ from the one uploaded.
                self.uploaded = Some(UploadedModel::of(model));
                tracing::trace!(n_gaussians = model.len(), "Updated Gaussians in place");
            }
            Some(Err(e)) => {
                // The size check above should make this unreachable; fall back
                // to a full upload rather than rendering stale data.
                tracing::warn!("update_gaussians fell back to a full upload: {e}");
                self.upload_gaussians(model);
            }
            None => self.upload_gaussians(model),
        }
    }

    /// Forget which model is resident, forcing the next
    /// [`forward`](Self::forward) to re-upload.
    ///
    /// Use this after mutating a [`GaussianModel`] in place: the residency check
    /// identifies the model's allocation, not its contents, so an in-place edit
    /// is invisible to it.
    pub fn invalidate_gaussians(&mut self) {
        self.uploaded = None;
    }

    /// The workgroup geometry every dispatch in this rasterizer is derived from.
    ///
    /// Matches the `@workgroup_size` attributes of the compiled shaders; see
    /// [`WorkgroupConfig::shipped`].
    pub fn workgroups(&self) -> &WorkgroupConfig {
        &self.workgroups
    }

    // --- FLAME binding backward ---

    /// Install the FLAME mesh whose surface the Gaussians are bound to, so
    /// [`backward`](Self::backward) can also produce
    /// [`GaussianGradients::grad_local_offsets`].
    ///
    /// The frame table is [`build_face_frames`], indexed by
    /// `GaussianModel::face_indices` — exactly the frames
    /// `binding::apply_binding` applies each local offset in, which makes the
    /// backward projection the true adjoint of the forward binding rather
    /// than the per-vertex approximation.
    ///
    /// Call this once per pose (the frames move with the mesh). Without it
    /// the rasterizer has no way to know the surface frames: it only ever
    /// sees world-space positions, so `∂L/∂offset` would be identically zero
    /// and `ParameterGroup::Offset` would never train.
    pub fn set_binding_mesh(&mut self, mesh: &oxigaf_flame::Mesh) {
        self.set_binding_frames(build_face_frames(mesh));
    }

    /// Install a FLAME binding frame table directly.
    ///
    /// Entry `k` must be the frame of the surface element that
    /// `GaussianModel::face_indices[i] == k` refers to. Prefer
    /// [`set_binding_mesh`](Self::set_binding_mesh), which builds the table
    /// that matches `binding::apply_binding`.
    pub fn set_binding_frames(&mut self, frames: Vec<FaceFrame>) {
        let changed = self
            .binding_frames
            .as_ref()
            .is_none_or(|old| old.len() != frames.len());
        self.binding_frames = Some(frames);
        if changed {
            // The frame table length is the binding buffers' `num_vertices`.
            self.flame_bufs = None;
        }
    }

    /// Drop the binding frame table; `grad_local_offsets` goes back to empty.
    pub fn clear_binding_frames(&mut self) {
        self.binding_frames = None;
        self.flame_bufs = None;
    }

    /// Whether a binding frame table is installed.
    pub fn has_binding_frames(&self) -> bool {
        self.binding_frames.is_some()
    }

    // --- GPU timestamp profiling ---

    /// Start recording real GPU pass durations with timestamp queries.
    ///
    /// [`PassProfiler`](crate::profiler::PassProfiler) measures how long the
    /// *host* spent recording a dispatch, which says nothing about GPU
    /// execution. Once enabled, every compute pass this rasterizer records
    /// carries `timestamp_writes` and [`gpu_timestamps`](Self::gpu_timestamps)
    /// reports the resolved per-pass durations.
    ///
    /// Opt-in on purpose: timestamp writes are not free, and a device created
    /// without `wgpu::Features::TIMESTAMP_QUERY` cannot serve them at all.
    /// [`Rasterizer::new`] requests the feature whenever the adapter offers
    /// it.
    ///
    /// # Errors
    ///
    /// [`RenderError::GpuInit`] when the device lacks
    /// `wgpu::Features::TIMESTAMP_QUERY`.
    pub fn enable_gpu_timestamps(&mut self) -> Result<(), RenderError> {
        if self.gpu_timestamps.is_some() {
            return Ok(());
        }
        self.gpu_timestamps = Some(GpuTimestampProfiler::new(
            &self.device,
            &self.queue,
            GpuTimestampProfiler::DEFAULT_MAX_PASSES,
        )?);
        Ok(())
    }

    /// Stop recording GPU timestamps and drop the accumulated statistics.
    pub fn disable_gpu_timestamps(&mut self) {
        self.gpu_timestamps = None;
    }

    /// The GPU timestamp profiler, if
    /// [`enable_gpu_timestamps`](Self::enable_gpu_timestamps) succeeded.
    pub fn gpu_timestamps(&self) -> Option<&GpuTimestampProfiler> {
        self.gpu_timestamps.as_ref()
    }

    /// Reserve timestamp slots for a pass about to be recorded.
    fn timestamps_for(&self, pass_name: &str) -> Option<wgpu::ComputePassTimestampWrites<'_>> {
        self.gpu_timestamps
            .as_ref()
            .and_then(|p| p.pass_writes(pass_name))
    }

    /// Resolve this frame's timestamps into the profiler's statistics.
    ///
    /// Must run after the submission carrying the resolve has completed; a
    /// readback failure is logged rather than failing the frame, since the
    /// render itself succeeded.
    fn collect_timestamps(&self) {
        if let Some(profiler) = self.gpu_timestamps.as_ref() {
            if let Err(e) = profiler.collect(&self.device) {
                tracing::warn!("GPU timestamp readback failed: {e}");
            }
        }
    }

    /// Release this frame's timestamp reservations without reading them back,
    /// for a frame abandoned before its submission completes.
    fn discard_timestamps(&self) {
        if let Some(profiler) = self.gpu_timestamps.as_ref() {
            profiler.discard();
        }
    }

    /// Run the forward rasterization pass.
    ///
    /// Re-uploads `model` when it is not the model currently resident on the
    /// GPU (see [`invalidate_gaussians`](Self::invalidate_gaussians) for the
    /// in-place-mutation caveat).
    ///
    /// # Errors
    ///
    /// * [`RenderError::TooManyGaussians`] when the Gaussian count exceeds what
    ///   the hierarchical prefix sum can scan.
    /// * [`RenderError::TooManyTilePairs`] when the Gaussians overlap more tiles
    ///   than the sort buffers were allocated for — previously this silently
    ///   dropped splats from the render.
    /// * [`RenderError::Rasterize`] for readback failures.
    /// * [`RenderError::BufferOverflow`] when a readback's staging buffer
    ///   would exceed the device's `max_buffer_size`.
    pub fn forward(
        &mut self,
        model: &GaussianModel,
        camera: &RenderCamera,
    ) -> Result<RenderOutput, RenderError> {
        // Re-upload whenever this is not the resident model; uploading only on
        // the very first call rendered stale data for every later model.
        if !self.uploaded.is_some_and(|u| u.describes(model)) {
            self.upload_gaussians(model);
        }

        // All buffers are guaranteed to be Some after upload_gaussians()
        let gauss = self.gaussian_bufs.as_ref().ok_or_else(|| {
            RenderError::Rasterize("gaussian_bufs not initialized after upload_gaussians".into())
        })?;
        let inter = self.intermediate_bufs.as_ref().ok_or_else(|| {
            RenderError::Rasterize(
                "intermediate_bufs not initialized after upload_gaussians".into(),
            )
        })?;
        let output = self.output_bufs.as_ref().ok_or_else(|| {
            RenderError::Rasterize("output_bufs not initialized after upload_gaussians".into())
        })?;
        let bgs = self.frame_bgs.as_ref().ok_or_else(|| {
            RenderError::Rasterize(
                "frame bind groups not initialized after upload_gaussians".into(),
            )
        })?;
        let n = gauss.count;

        // The hierarchical scan is three levels deep; above that the tile
        // offsets would simply be wrong, so fail loudly instead.
        IntermediateBuffers::validate_gaussian_count(n)?;

        // Update uniforms
        let uniforms = Uniforms {
            view: camera.view_matrix,
            proj: camera.proj_matrix,
            cam_pos: camera.position,
            _pad0: 0.0,
            focal: camera.focal,
            viewport: [
                self.config.image_width as f32,
                self.config.image_height as f32,
            ],
            tile_grid: [self.config.tiles_x(), self.config.tiles_y()],
            num_gaussians: n,
            sh_degree: self.config.sh_degree,
            near_plane: self.config.near_plane,
            far_plane: self.config.far_plane,
            _pad_bg: [0.0, 0.0],
            background: self.config.background,
            output_flags: self.config.output_flags(),
            transmittance_threshold: self.config.transmittance_threshold,
            tile_size: self.config.tile_size,
            _pad1: [0, 0],
        };
        self.uniform_buf.update(&self.queue, &uniforms);

        let plan = inter.scan_plan;
        self.queue.write_buffer(
            &self.params.scan_count,
            0,
            bytemuck::cast_slice(&[n, 0u32, 0u32, 0u32]),
        );
        self.queue.write_buffer(
            &self.params.scan_l1,
            0,
            bytemuck::cast_slice(&[plan.level0_workgroups, 0u32, 0u32, 0u32]),
        );
        self.queue.write_buffer(
            &self.params.scan_l2,
            0,
            bytemuck::cast_slice(&[plan.level1_workgroups, 0u32, 0u32, 0u32]),
        );

        // ---- Submission 1: preprocess + hierarchical prefix sum ----
        //
        // Split here because the exact Gaussian-tile pair count is the last
        // element of the prefix sum, and everything downstream (sort extent,
        // tile-range scan, buffer-capacity validation) needs it.
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("forward_scan"),
            });

        // ---- Step 1: Preprocess ----
        if n > 0 {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("preprocess"),
                timestamp_writes: self.timestamps_for("preprocess"),
            });
            pass.set_pipeline(&self.pipelines.preprocess);
            pass.set_bind_group(0, &bgs.preprocess, &[]);
            pass.dispatch_workgroups(self.workgroups.preprocess.dispatch_count_x(n), 1, 1);
        }

        // ---- Step 2: Hierarchical prefix sum over tile_counts ----
        //
        // Level 0 scans the counts and emits one block total per workgroup;
        // level 1 scans those totals and emits one level-2 total per workgroup;
        // level 2 scans the level-2 totals in a single workgroup. Each level's
        // offsets must then be added back into the level below it.
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("prefix_sum_l0"),
                timestamp_writes: self.timestamps_for("prefix_sum_l0"),
            });
            pass.set_pipeline(&self.pipelines.prefix_sum);
            pass.set_bind_group(0, &bgs.scan_l0, &[]);
            pass.dispatch_workgroups(plan.level0_workgroups, 1, 1);
        }
        if plan.level0_workgroups > 1 {
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("prefix_sum_l1"),
                    timestamp_writes: self.timestamps_for("prefix_sum_l1"),
                });
                pass.set_pipeline(&self.pipelines.prefix_sum);
                pass.set_bind_group(0, &bgs.scan_l1, &[]);
                pass.dispatch_workgroups(plan.level1_workgroups, 1, 1);
            }
            if plan.level1_workgroups > 1 {
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("prefix_sum_l2"),
                        timestamp_writes: self.timestamps_for("prefix_sum_l2"),
                    });
                    pass.set_pipeline(&self.pipelines.prefix_sum);
                    pass.set_bind_group(0, &bgs.scan_l2, &[]);
                    pass.dispatch_workgroups(plan.level2_workgroups, 1, 1);
                }
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("prefix_sum_add_l1"),
                    timestamp_writes: self.timestamps_for("prefix_sum_add_l1"),
                });
                pass.set_pipeline(&self.pipelines.prefix_sum_add);
                pass.set_bind_group(0, &bgs.add_l1, &[]);
                pass.dispatch_workgroups(plan.level1_workgroups, 1, 1);
            }
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("prefix_sum_add_l0"),
                timestamp_writes: self.timestamps_for("prefix_sum_add_l0"),
            });
            pass.set_pipeline(&self.pipelines.prefix_sum_add);
            pass.set_bind_group(0, &bgs.add_l0, &[]);
            pass.dispatch_workgroups(plan.level0_workgroups, 1, 1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));

        // Exact number of Gaussian-tile pairs (a four-byte readback), validated
        // against the allocated capacity. This replaces the old 0xFF host fill
        // of the whole sort buffer: with the real count driving both the sort
        // and the tile-range scan, no stage ever reads an unwritten entry.
        let actual_pairs = match inter.verify_pair_capacity(&self.device, &self.queue, n) {
            Ok(pairs) => pairs,
            Err(e) => {
                // The scan passes already reserved timestamp slots; release
                // them so the next frame starts from slot zero.
                self.discard_timestamps();
                return Err(e);
            }
        };

        let num_tiles = self.config.num_tiles();
        self.queue.write_buffer(
            &self.params.tile_ranges,
            0,
            bytemuck::cast_slice(&[actual_pairs, num_tiles, 0u32, 0u32]),
        );

        // ---- Submission 2: binning, sort and rasterization ----
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("forward_raster"),
            });

        // tile_ranges must be zero-cleared so tiles with no Gaussians get the
        // empty range (0, 0) rather than last frame's leftovers.
        encoder.clear_buffer(&inter.tile_ranges, 0, None);

        // ---- Step 3: Tile assign ----
        if n > 0 && actual_pairs > 0 {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("tile_assign"),
                timestamp_writes: self.timestamps_for("tile_assign"),
            });
            pass.set_pipeline(&self.pipelines.tile_assign);
            pass.set_bind_group(0, &bgs.tile_assign, &[]);
            pass.dispatch_workgroups(self.workgroups.preprocess.dispatch_count_x(n), 1, 1);
        }

        // ---- Step 4: Radix sort (over the real pair count, not the capacity) ----
        if let Err(e) = self
            .sorter
            .sort(&mut encoder, &self.queue, actual_pairs, num_tiles)
        {
            self.discard_timestamps();
            return Err(e);
        }

        // ---- Step 5: Tile ranges ----
        if actual_pairs > 0 {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("tile_ranges"),
                timestamp_writes: self.timestamps_for("tile_ranges"),
            });
            pass.set_pipeline(&self.pipelines.tile_ranges);
            pass.set_bind_group(0, &bgs.tile_ranges, &[]);
            pass.dispatch_workgroups(self.workgroups.sort.dispatch_count_x(actual_pairs), 1, 1);
        }

        // ---- Step 6: Forward rasterization ----
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rasterize_fwd"),
                timestamp_writes: self.timestamps_for("rasterize_fwd"),
            });
            pass.set_pipeline(&self.pipelines.rasterize_fwd);
            pass.set_bind_group(0, &bgs.rasterize_fwd, &[]);
            pass.dispatch_workgroups(self.config.tiles_x(), self.config.tiles_y(), 1);
        }

        // Record every readback copy into the same encoder, so the frame costs
        // one submission and one device wait instead of three or four.
        let npx = self.config.num_pixels() as usize;
        let want_normals = self.config.output_normals && output.normals.is_some();
        let pending = match self.stage_forward_readbacks(&mut encoder, output, npx, want_normals) {
            Ok(pending) => pending,
            Err(e) => {
                // Same contract as the other post-dispatch bailouts: the
                // passes above already reserved timestamp slots, and this
                // frame is not going to resolve them.
                self.discard_timestamps();
                return Err(e);
            }
        };

        // Resolve the frame's GPU timestamps in the same submission as the
        // last pass, so no extra encoder is needed.
        if let Some(profiler) = self.gpu_timestamps.as_ref() {
            profiler.resolve(&mut encoder);
        }

        self.queue.submit(std::iter::once(encoder.finish()));

        let readback_result = self.map_and_collect(pending);
        // The device has been polled by `map_and_collect` (or the frame
        // failed); either way this frame's reservations must not leak into
        // the next one.
        self.collect_timestamps();
        let mut readbacks = readback_result?.into_iter();
        let color_data = readbacks.next().unwrap_or_default();
        let depth_data = readbacks.next().unwrap_or_default();
        let normals = readbacks.next().map(|raw| {
            raw.chunks_exact(4)
                .map(|c| [c[0], c[1], c[2]])
                .collect::<Vec<[f32; 3]>>()
        });

        Ok(RenderOutput {
            color_data,
            depth_data,
            normals,
            width: self.config.image_width,
            height: self.config.image_height,
        })
    }

    /// Run the backward rasterization pass.
    ///
    /// Given the per-pixel gradient of the loss w.r.t. the rendered image
    /// (`grad_image`, RGBA `[H×W×4]` f32), dispatches the backward compute
    /// shader and reads back per-Gaussian gradients.
    ///
    /// # Errors
    ///
    /// * [`RenderError::Rasterize`] when no model has been uploaded.
    /// * [`RenderError::MismatchedBufferSizes`] when `model` does not match the
    ///   model whose buffers are resident on the GPU, or when `grad_image` is
    ///   larger than the configured image.
    /// * [`RenderError::BufferOverflow`] when a gradient readback's staging
    ///   buffer would exceed the device's `max_buffer_size`.
    pub fn backward(
        &mut self,
        model: &GaussianModel,
        grad_image: &[f32],
    ) -> Result<GaussianGradients, RenderError> {
        let uploaded = self.uploaded.ok_or_else(|| {
            RenderError::Rasterize("backward() called before upload_gaussians()".into())
        })?;
        // The gradient buffers were sized for the resident model. Reading them
        // back with a different model's counts silently truncates (or over-reads)
        // the gradient slices, so reject that outright — before allocating
        // anything sized by `model`.
        if uploaded.len != model.len()
            || uploaded.sh_len != model.sh_coeffs.len()
            || uploaded.sh_degree != model.sh_degree
        {
            return Err(RenderError::MismatchedBufferSizes {
                expected: uploaded.len,
                actual: model.len(),
            });
        }
        // Allocate/refresh the FLAME binding buffers next: this needs `&mut
        // self`, and everything below holds long-lived immutable borrows.
        let binding_active = self.prepare_binding_buffers(model)?;

        let gauss = self.gaussian_bufs.as_ref().ok_or_else(|| {
            RenderError::Rasterize("backward() called before upload_gaussians()".into())
        })?;
        let grads = self
            .gradient_bufs
            .as_ref()
            .ok_or_else(|| RenderError::Rasterize("backward() called before forward()".into()))?;
        let bgs = self
            .bwd_bgs
            .as_ref()
            .ok_or_else(|| RenderError::Rasterize("backward() called before forward()".into()))?;
        let n = gauss.count;

        // Upload the image-space gradient into the persistent buffer instead of
        // allocating a full-image GPU buffer on every training step.
        let expected = self.config.num_pixels() as usize * 4;
        if grad_image.len() > expected {
            return Err(RenderError::MismatchedBufferSizes {
                expected,
                actual: grad_image.len(),
            });
        }
        if grad_image.len() == expected {
            self.queue
                .write_buffer(&self.grad_output_buf, 0, bytemuck::cast_slice(grad_image));
        } else {
            // Short gradient: pad so the tail of a previous step cannot leak in.
            let mut padded = vec![0.0f32; expected];
            padded[..grad_image.len()].copy_from_slice(grad_image);
            self.queue
                .write_buffer(&self.grad_output_buf, 0, bytemuck::cast_slice(&padded));
        }

        // Element counts for the atomic→f32 conversions change only with `n`,
        // but writing 12 bytes is cheaper than tracking that.
        self.queue.write_buffer(
            &self.params.num_elements_means2d,
            0,
            bytemuck::cast_slice(&[n * 2, 0u32, 0u32, 0u32]),
        );
        self.queue.write_buffer(
            &self.params.num_elements_conics,
            0,
            bytemuck::cast_slice(&[n * 3, 0u32, 0u32, 0u32]),
        );
        self.queue.write_buffer(
            &self.params.num_elements_colors,
            0,
            bytemuck::cast_slice(&[n * 3, 0u32, 0u32, 0u32]),
        );

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("backward_pass"),
            });

        // Clear gradient buffers to zero before accumulation.
        encoder.clear_buffer(&grads.grad_positions, 0, None);
        encoder.clear_buffer(&grads.grad_rotations, 0, None);
        encoder.clear_buffer(&grads.grad_scales, 0, None);
        encoder.clear_buffer(&grads.grad_opacities, 0, None);
        encoder.clear_buffer(&grads.grad_sh_coeffs, 0, None);
        encoder.clear_buffer(&grads.grad_means2d_atomic, 0, None);
        encoder.clear_buffer(&grads.grad_conics_atomic, 0, None);
        encoder.clear_buffer(&grads.grad_colors_atomic, 0, None);
        encoder.clear_buffer(&grads.grad_means2d, 0, None);
        encoder.clear_buffer(&grads.grad_conics, 0, None);
        encoder.clear_buffer(&grads.grad_colors, 0, None);

        // --- Rasterize backward ---
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rasterize_bwd"),
                timestamp_writes: self.timestamps_for("rasterize_bwd"),
            });
            pass.set_pipeline(&self.pipelines.rasterize_bwd);
            pass.set_bind_group(0, &bgs.rasterize_bwd, &[]);
            pass.dispatch_workgroups(self.config.tiles_x(), self.config.tiles_y(), 1);
        }

        // --- Atomic to f32 conversion ---
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("atomic_to_f32_means2d"),
                timestamp_writes: self.timestamps_for("atomic_to_f32_means2d"),
            });
            pass.set_pipeline(&self.pipelines.atomic_to_f32);
            pass.set_bind_group(0, &bgs.atomic_means2d, &[]);
            pass.dispatch_workgroups(self.workgroups.sort.dispatch_count_x(n * 2), 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("atomic_to_f32_conics"),
                timestamp_writes: self.timestamps_for("atomic_to_f32_conics"),
            });
            pass.set_pipeline(&self.pipelines.atomic_to_f32);
            pass.set_bind_group(0, &bgs.atomic_conics, &[]);
            pass.dispatch_workgroups(self.workgroups.sort.dispatch_count_x(n * 3), 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("atomic_to_f32_colors"),
                timestamp_writes: self.timestamps_for("atomic_to_f32_colors"),
            });
            pass.set_pipeline(&self.pipelines.atomic_to_f32);
            pass.set_bind_group(0, &bgs.atomic_colors, &[]);
            pass.dispatch_workgroups(self.workgroups.sort.dispatch_count_x(n * 3), 1, 1);
        }

        // --- Preprocess backward ---
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("preprocess_bwd"),
                timestamp_writes: self.timestamps_for("preprocess_bwd"),
            });
            pass.set_pipeline(&self.pipelines.preprocess_bwd);
            pass.set_bind_group(0, &bgs.preprocess_bwd, &[]);
            pass.dispatch_workgroups(self.workgroups.preprocess.dispatch_count_x(n), 1, 1);
        }

        // --- FLAME binding backward ---
        //
        // `preprocess_bwd` has just written ∂L/∂position; projecting it onto
        // each Gaussian's bound face frame yields ∂L/∂local_offset, the only
        // path by which the photometric loss can reach the learnable offsets.
        // The position gradients are copied GPU→GPU inside this same encoder,
        // so no host round trip is involved.
        let flame = if binding_active {
            self.flame_bufs.as_ref()
        } else {
            None
        };
        if let Some(bufs) = flame {
            if let Err(e) = bufs.copy_position_gradients_from(&mut encoder, &grads.grad_positions) {
                self.discard_timestamps();
                return Err(e);
            }
            bufs.record_backward_with_timestamps(
                &mut encoder,
                &self.pipelines.flame_binding_bwd,
                self.timestamps_for("flame_binding_bwd"),
            );
        }

        // Stage every gradient readback into the same submission.
        let sh_total = uploaded.sh_len.max(1);
        let pending = match self.stage_backward_readbacks(&mut encoder, grads, flame, n, sh_total) {
            Ok(pending) => pending,
            Err(e) => {
                self.discard_timestamps();
                return Err(e);
            }
        };

        if let Some(profiler) = self.gpu_timestamps.as_ref() {
            profiler.resolve(&mut encoder);
        }

        self.queue.submit(std::iter::once(encoder.finish()));

        let readback_result = self.map_and_collect(pending);
        self.collect_timestamps();
        let mut readbacks = readback_result?.into_iter();
        let grad_positions = readbacks.next().unwrap_or_default();
        let grad_rotations = readbacks.next().unwrap_or_default();
        let grad_scales = readbacks.next().unwrap_or_default();
        let grad_opacities = readbacks.next().unwrap_or_default();
        let grad_sh_coeffs = readbacks.next().unwrap_or_default();
        let grad_local_offsets_raw = readbacks.next().unwrap_or_default();

        // Validate gradients for NaN/Inf under gpu_debug feature.
        #[cfg(feature = "gpu_debug")]
        {
            use crate::debug_readback::{validate_buffer_count, validate_no_nan_inf};
            validate_no_nan_inf(&grad_positions, "grad_positions")?;
            validate_no_nan_inf(&grad_rotations, "grad_rotations")?;
            validate_no_nan_inf(&grad_scales, "grad_scales")?;
            validate_no_nan_inf(&grad_opacities, "grad_opacities")?;
            validate_no_nan_inf(&grad_sh_coeffs, "grad_sh_coeffs")?;
            validate_buffer_count(grad_opacities.len(), n as usize, "grad_opacities")?;
            validate_buffer_count(grad_sh_coeffs.len(), sh_total, "grad_sh_coeffs")?;
        }

        // Convert from padded [f32; 4] GPU layout to dense [f32; 3] CPU layout
        // for positions and scales.
        let unpad3 = |data: &[f32]| -> Vec<[f32; 3]> {
            data.chunks_exact(4).map(|c| [c[0], c[1], c[2]]).collect()
        };
        let unpad4 = |data: &[f32]| -> Vec<[f32; 4]> {
            data.chunks_exact(4)
                .map(|c| [c[0], c[1], c[2], c[3]])
                .collect()
        };

        Ok(GaussianGradients {
            grad_positions: unpad3(&grad_positions),
            grad_rotations: unpad4(&grad_rotations),
            grad_scales: unpad3(&grad_scales),
            grad_opacities,
            grad_sh_coeffs,
            grad_local_offsets: unpad3(&grad_local_offsets_raw),
        })
    }

    /// Allocate or refresh the FLAME binding backward buffers for `model`.
    ///
    /// Returns `false` (and does nothing) when no frame table is installed or
    /// the model is empty — the backward pass then simply reports no offset
    /// gradients. Buffers are reused across steps and reallocated only when
    /// the Gaussian count or the frame-table length changes.
    ///
    /// # Errors
    ///
    /// * [`RenderError::MismatchedBufferSizes`] when `model.face_indices` does
    ///   not have one entry per Gaussian — without it there is no way to know
    ///   which surface frame each Gaussian's offset lives in.
    /// * [`RenderError::InvalidGaussian`] when a face index falls outside the
    ///   installed frame table (the shader has no bounds check).
    fn prepare_binding_buffers(&mut self, model: &GaussianModel) -> Result<bool, RenderError> {
        let table_len = match self.binding_frames.as_ref() {
            Some(frames) => frames.len(),
            None => return Ok(false),
        };
        let n = model.len();
        if n == 0 || table_len == 0 {
            return Ok(false);
        }
        if model.face_indices.len() != n {
            return Err(RenderError::MismatchedBufferSizes {
                expected: n,
                actual: model.face_indices.len(),
            });
        }

        let needs_alloc = self
            .flame_bufs
            .as_ref()
            .is_none_or(|b| b.num_gaussians() != n || b.num_vertices() != table_len);
        if needs_alloc {
            self.flame_bufs = Some(FlameBindingBuffers::new(
                &self.device,
                &self.pipelines.flame_binding_bwd_bgl,
                n,
                table_len,
            ));
        }

        let bufs = self.flame_bufs.as_ref().ok_or_else(|| {
            RenderError::Rasterize("flame binding buffers missing after allocation".into())
        })?;
        let frames = self.binding_frames.as_ref().ok_or_else(|| {
            RenderError::Rasterize("binding frame table missing after allocation".into())
        })?;
        bufs.update_binding_info(&self.queue, &model.face_indices)?;
        bufs.update_tbn_frames(&self.queue, frames)?;
        Ok(true)
    }

    /// Download rendered image as an `image::RgbaImage`.
    pub fn download_image(&self, output: &RenderOutput) -> image::RgbaImage {
        let w = output.width;
        let h = output.height;
        let buf = color_to_rgba_bytes(&output.color_data, w, h);
        image::RgbaImage::from_raw(w, h, buf).unwrap_or_else(|| {
            tracing::warn!(
                width = w,
                height = h,
                "download_image: could not build an image buffer; returning a blank image"
            );
            image::RgbaImage::new(w, h)
        })
    }

    /// Get a reference to the config.
    pub fn config(&self) -> &RasterConfig {
        &self.config
    }

    /// Get a reference to the buffer pool, if enabled.
    pub fn buffer_pool(&self) -> Option<&BufferPool> {
        self.buffer_pool.as_ref()
    }

    /// Read the rasterizer's own tiling state back and summarise it.
    ///
    /// Copies `tile_ranges`, `depths` and `radii` to the host and builds a
    /// [`RasterizationSnapshot`](crate::debug_readback::RasterizationSnapshot):
    /// per-tile Gaussian counts, load balance, depth range and screen sizes,
    /// as the GPU actually produced them. This is what distinguishes "the
    /// scene is empty" from "the tile binning dropped everything".
    ///
    /// Call it **after a completed [`forward`](Self::forward)**: `tile_ranges`
    /// is zero-cleared at the start of every frame, so an earlier read looks
    /// exactly like an empty scene.
    ///
    /// Costs a submit and a device wait per buffer, which is why it is behind
    /// the `gpu_debug` feature rather than available in the render loop.
    ///
    /// # Errors
    ///
    /// * [`RenderError::Rasterize`] when no model has been uploaded yet.
    /// * Whatever
    ///   [`DebugReadbackBuilder::read_from_gpu`](crate::debug_readback::DebugReadbackBuilder::read_from_gpu)
    ///   reports for a degenerate tile configuration or a failed readback.
    #[cfg(feature = "gpu_debug")]
    pub fn debug_snapshot(
        &self,
    ) -> Result<crate::debug_readback::RasterizationSnapshot, RenderError> {
        let inter = self.intermediate_bufs.as_ref().ok_or_else(|| {
            RenderError::Rasterize("debug_snapshot() called before upload_gaussians()".into())
        })?;
        let n = self.gaussian_bufs.as_ref().map_or(0, |g| g.count);

        crate::debug_readback::DebugReadbackBuilder::new(
            self.config.image_width,
            self.config.image_height,
        )
        .with_tile_size(self.config.tile_size)
        .read_from_gpu(
            &self.device,
            &self.queue,
            &inter.tile_ranges,
            &inter.depths,
            &inter.radii,
            n,
        )
    }

    /// Get current buffer pool statistics.
    ///
    /// Returns `None` if buffer pooling is disabled.
    pub fn pool_stats(&self) -> Option<PoolStats> {
        self.buffer_pool.as_ref().map(|p| p.stats())
    }

    /// Log current memory usage to debug output.
    ///
    /// This includes buffer pool statistics if pooling is enabled.
    pub fn log_memory_usage(&self) {
        if let Some(ref pool) = self.buffer_pool {
            pool.log_usage();
        } else {
            tracing::debug!("Buffer pooling disabled, no pool stats available");
        }
    }

    /// Clear all available buffers from the pool.
    ///
    /// This frees GPU memory held by unused buffers. In-use buffers
    /// will still return to the pool when their references are dropped.
    pub fn clear_buffer_pool(&self) {
        if let Some(ref pool) = self.buffer_pool {
            pool.clear();
        }
    }

    /// Update the buffer pool memory budget.
    ///
    /// # Arguments
    ///
    /// * `max_bytes` - New maximum memory budget in bytes.
    pub fn set_pool_budget(&self, max_bytes: u64) {
        if let Some(ref pool) = self.buffer_pool {
            pool.set_budget(max_bytes);
        }
    }

    /// Record a `count`-float readback of `src` into `encoder`.
    ///
    /// The copy is not submitted here: several readbacks share one submission
    /// and one [`map_and_collect`](Self::map_and_collect) wait.
    ///
    /// # Invariant
    ///
    /// `staging_size` is derived from `src.size()` — a buffer that already
    /// passed device validation, so at most `max_buffer_size` — rounded up by
    /// fewer than [`wgpu::MAP_ALIGNMENT`] bytes. Exceeding the device limit
    /// therefore requires `src.size()` to sit within that alignment of the
    /// limit itself, which no buffer this rasterizer allocates does. The
    /// check is kept regardless: the alternative to returning an error here
    /// is a wgpu validation abort of the whole process (see
    /// [`check_staging_fits`]), which is far too expensive an outcome to
    /// leave resting on an invariant maintained by other code.
    ///
    /// # Errors
    ///
    /// [`RenderError::BufferOverflow`] when the staging buffer would exceed
    /// the device's `max_buffer_size` and so can be served neither from the
    /// pool nor directly.
    fn stage_readback(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        src: &wgpu::Buffer,
        count: usize,
    ) -> Result<PendingReadback, RenderError> {
        // Never copy more than the source holds: an element count that outruns
        // the buffer is a wgpu validation failure, not a truncated read.
        let byte_size = (count as u64).saturating_mul(4).min(src.size());
        let count = (byte_size / 4) as usize;
        // Round up so the staging buffer satisfies the map alignment even for an
        // odd float count.
        let staging_size = staging_size_for(byte_size);
        let usage = wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST;

        let pooled = self
            .buffer_pool
            .as_ref()
            .and_then(|pool| pool.acquire(&self.device, staging_size, usage, STAGING_READ_LABEL));
        let staging = match pooled {
            Some(pooled) => StagingBuffer::Pooled(pooled),
            None => {
                // The pool declined. That is a fallback signal, not a licence
                // to create the same buffer unchecked.
                check_staging_fits(staging_size, self.device.limits().max_buffer_size)?;
                StagingBuffer::Direct(self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(STAGING_READ_LABEL),
                    size: staging_size,
                    usage,
                    mapped_at_creation: false,
                }))
            }
        };

        if byte_size > 0 {
            encoder.copy_buffer_to_buffer(src, 0, &staging, 0, byte_size);
        }

        Ok(PendingReadback { staging, count })
    }

    /// Stage the forward pass's colour, depth and (optional) normal readbacks.
    ///
    /// Grouped so that a refused staging allocation returns before any of them
    /// is submitted, and so the caller has a single error path on which to
    /// release the frame's timestamp reservations.
    ///
    /// # Errors
    ///
    /// Propagates [`stage_readback`](Self::stage_readback)'s
    /// [`RenderError::BufferOverflow`].
    fn stage_forward_readbacks(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        output: &OutputBuffers,
        npx: usize,
        want_normals: bool,
    ) -> Result<Vec<PendingReadback>, RenderError> {
        let mut pending = Vec::with_capacity(3);
        pending.push(self.stage_readback(encoder, &output.color, npx * 4)?);
        pending.push(self.stage_readback(encoder, &output.depth, npx)?);
        if want_normals {
            if let Some(normals_buf) = output.normals.as_ref() {
                pending.push(self.stage_readback(encoder, normals_buf, npx * 4)?);
            }
        }
        Ok(pending)
    }

    /// Stage the backward pass's per-Gaussian gradient readbacks, plus the
    /// FLAME offset gradients when the binding pass ran.
    ///
    /// The order here is the order [`backward`](Self::backward) drains them
    /// in.
    ///
    /// # Errors
    ///
    /// Propagates [`stage_readback`](Self::stage_readback)'s
    /// [`RenderError::BufferOverflow`].
    fn stage_backward_readbacks(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        grads: &GradientBuffers,
        flame: Option<&FlameBindingBuffers>,
        n: u32,
        sh_total: usize,
    ) -> Result<Vec<PendingReadback>, RenderError> {
        let n = n as usize;
        let mut pending = Vec::with_capacity(6);
        pending.push(self.stage_readback(encoder, &grads.grad_positions, n * 4)?);
        pending.push(self.stage_readback(encoder, &grads.grad_rotations, n * 4)?);
        pending.push(self.stage_readback(encoder, &grads.grad_scales, n * 4)?);
        pending.push(self.stage_readback(encoder, &grads.grad_opacities, n)?);
        pending.push(self.stage_readback(encoder, &grads.grad_sh_coeffs, sh_total)?);
        if let Some(bufs) = flame {
            pending.push(self.stage_readback(encoder, &bufs.offset_grads, n * 4)?);
        }
        Ok(pending)
    }

    /// Map every staged readback, wait once, and drain them in order.
    fn map_and_collect(&self, pending: Vec<PendingReadback>) -> Result<Vec<Vec<f32>>, RenderError> {
        let mut receivers = Vec::with_capacity(pending.len());
        for item in &pending {
            let (tx, rx) = std::sync::mpsc::channel();
            item.staging
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |result| {
                    tx.send(result).ok();
                });
            receivers.push(rx);
        }

        // One device wait for the whole frame instead of one per readback.
        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });

        // Every staging buffer is unmapped before any error propagates: a pooled
        // buffer returned to the pool while still mapped makes the *next*
        // `map_async` on it a validation failure, turning one bad readback into
        // a permanently broken pool.
        let mut out = Vec::with_capacity(pending.len());
        let mut first_error: Option<RenderError> = None;
        for (item, rx) in pending.iter().zip(receivers) {
            let mapped = rx
                .recv()
                .map_err(|e| RenderError::Rasterize(format!("Channel recv failed: {e}")))
                .and_then(|r| {
                    r.map_err(|e| RenderError::Rasterize(format!("Buffer map failed: {e}")))
                });
            if let Err(e) = mapped {
                if first_error.is_none() {
                    first_error = Some(e);
                }
                // Not mapped, so it must NOT be unmapped: it returns to the
                // pool in a clean state.
                continue;
            }

            let slice = item.staging.slice(..);
            match slice.get_mapped_range() {
                Ok(data) => {
                    let all_floats: &[f32] = bytemuck::cast_slice(&data);
                    // Only return the requested count, not the entire buffer.
                    let end = item.count.min(all_floats.len());
                    out.push(all_floats[..end].to_vec());
                    drop(data);
                }
                Err(e) => {
                    if first_error.is_none() {
                        first_error =
                            Some(RenderError::Rasterize(format!("Mapped range failed: {e}")));
                    }
                }
            }
            item.staging.unmap();
        }

        match first_error {
            Some(e) => Err(e),
            None => Ok(out),
        }
    }
}
/// Convert linear-f32 RGBA samples to clamped 8-bit RGBA bytes.
///
/// One linear pass instead of `width × height` bounds-checked `put_pixel`
/// calls; a short `color_data` leaves the remaining pixels black rather than
/// panicking.
fn color_to_rgba_bytes(color_data: &[f32], width: u32, height: u32) -> Vec<u8> {
    let needed = (width as usize)
        .saturating_mul(height as usize)
        .saturating_mul(4);
    let mut out: Vec<u8> = Vec::with_capacity(needed);
    out.extend(
        color_data
            .iter()
            .take(needed)
            .map(|v| (v.clamp(0.0, 1.0) * 255.0) as u8),
    );
    out.resize(needed, 0);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffers::{ScanPlan, PREFIX_SUM_BLOCK};
    use crate::gaussian::GaussianAttributes;

    // ---- CPU model of the hierarchical prefix sum ----------------------------
    //
    // `forward()` records the scan as: per-block inclusive scan (prefix_sum.wgsl)
    // at each level, then an add-back of the *exclusive* prefix of the block
    // totals (prefix_sum_add.wgsl). The shaders cannot be unit-tested without a
    // GPU, but their arithmetic contract can: these helpers mirror them exactly,
    // so the off-by-one that made phase 3 add each block's *own* total is pinned
    // by an assertion here.

    const BLOCK: usize = PREFIX_SUM_BLOCK as usize;

    /// One `prefix_sum.wgsl` dispatch: inclusive scan inside each block, plus
    /// the per-block totals it writes to `block_sums`.
    fn scan_blocks(input: &[u32]) -> (Vec<u32>, Vec<u32>) {
        let mut scanned = Vec::with_capacity(input.len());
        let mut totals = Vec::new();
        for chunk in input.chunks(BLOCK) {
            let mut acc = 0u32;
            for &v in chunk {
                acc += v;
                scanned.push(acc);
            }
            totals.push(acc);
        }
        if totals.is_empty() {
            totals.push(0);
        }
        (scanned, totals)
    }

    /// One `prefix_sum_add.wgsl` dispatch: add the EXCLUSIVE prefix of
    /// `block_offsets` (itself an inclusive scan of the block totals) to every
    /// element of the matching block. Block 0 gets nothing.
    fn add_block_offsets(data: &mut [u32], block_offsets: &[u32]) {
        for (b, chunk) in data.chunks_mut(BLOCK).enumerate() {
            let offset = if b == 0 {
                0
            } else {
                block_offsets.get(b - 1).copied().unwrap_or(0)
            };
            for v in chunk.iter_mut() {
                *v += offset;
            }
        }
    }

    /// The full three-level scan `forward()` dispatches.
    fn hierarchical_scan(input: &[u32]) -> Vec<u32> {
        let (mut scanned, l1_totals) = scan_blocks(input);
        if l1_totals.len() > 1 {
            let (mut l1_scanned, l2_totals) = scan_blocks(&l1_totals);
            if l2_totals.len() > 1 {
                let (l2_scanned, _) = scan_blocks(&l2_totals);
                add_block_offsets(&mut l1_scanned, &l2_scanned);
            }
            add_block_offsets(&mut scanned, &l1_scanned);
        }
        scanned
    }

    #[test]
    fn test_hierarchical_scan_matches_plain_inclusive_scan() {
        // 262_145 crosses PREFIX_SUM_BLOCK² and so exercises all three levels.
        for n in [1usize, 511, 512, 513, 600, 2000, 262_145] {
            let input: Vec<u32> = (0..n).map(|i| (i % 7) as u32).collect();
            let mut expected = Vec::with_capacity(n);
            let mut acc = 0u32;
            for &v in &input {
                acc += v;
                expected.push(acc);
            }
            assert_eq!(hierarchical_scan(&input), expected, "n = {n}");
        }
    }

    #[test]
    fn test_block_offset_add_is_exclusive_not_inclusive() {
        // The bug: phase 3 added block_offsets[b] (the INCLUSIVE scan value)
        // instead of block_offsets[b - 1], inflating every block by its own
        // total — block 0 by 512 here, and every later block on top of that.
        let input: Vec<u32> = vec![1; 600]; // two blocks: 512 + 88
        let scanned = hierarchical_scan(&input);
        assert_eq!(scanned[0], 1, "block 0 must not be offset at all");
        assert_eq!(scanned[511], 512);
        assert_eq!(
            scanned[512], 513,
            "block 1 continues from block 0's total, it does not double it"
        );
        assert_eq!(scanned[599], 600);
    }

    #[test]
    fn test_scan_plan_levels_match_the_model() {
        // The dispatch plan the rasterizer follows must agree with the number of
        // levels the model needs.
        assert_eq!(ScanPlan::new(500).levels, 1);
        assert_eq!(ScanPlan::new(600).levels, 2);
        assert_eq!(ScanPlan::new(262_145).levels, 3);
        assert!(ScanPlan::new(262_145).is_supported());
    }

    /// Minimal model for the residency-identity tests (no GPU involved).
    fn make_model(n: usize) -> GaussianModel {
        GaussianModel {
            gaussians: vec![
                GaussianAttributes {
                    position: [0.0, 0.0, 1.0],
                    _pad0: 0.0,
                    rotation: [0.0, 0.0, 0.0, 1.0],
                    scale: [-2.0, -2.0, -2.0],
                    opacity: 0.0,
                };
                n
            ],
            sh_coeffs: vec![0.0; n * 3],
            sh_degree: 0,
            face_indices: vec![0u32; n],
            barycentric: vec![[1.0, 0.0, 0.0]; n],
            local_offsets: vec![[0.0, 0.0, 0.0]; n],
            is_rigid: vec![false; n],
        }
    }

    #[test]
    fn test_color_to_rgba_bytes_matches_per_pixel_conversion() {
        let color_data: Vec<f32> = vec![
            0.0, 0.5, 1.0, 1.0, // pixel 0
            -0.25, 0.25, 2.0, 0.0, // pixel 1 (out-of-range values clamp)
        ];
        let bytes = color_to_rgba_bytes(&color_data, 2, 1);
        assert_eq!(bytes.len(), 2 * 4); // 2 pixels × RGBA
                                        // Same arithmetic the old per-pixel loop used: clamp then scale by 255.
        let expected: Vec<u8> = color_data
            .iter()
            .map(|v| (v.clamp(0.0, 1.0) * 255.0) as u8)
            .collect();
        assert_eq!(bytes, expected);
    }

    #[test]
    fn test_color_to_rgba_bytes_pads_short_input() {
        // A truncated readback must not panic and must produce a full image.
        let bytes = color_to_rgba_bytes(&[1.0, 1.0, 1.0, 1.0], 4, 4);
        assert_eq!(bytes.len(), 4 * 4 * 4);
        assert_eq!(&bytes[..4], &[255, 255, 255, 255]);
        assert!(bytes[4..].iter().all(|&b| b == 0));
    }

    #[test]
    fn test_color_to_rgba_bytes_truncates_long_input() {
        let bytes = color_to_rgba_bytes(&vec![1.0f32; 1024], 2, 2);
        assert_eq!(bytes.len(), 2 * 2 * 4);
    }

    #[test]
    fn test_rasterize_tile_size_matches_config_default() {
        // The rasterization shaders are compiled for this tile size, so the
        // default config must already agree with them.
        assert_eq!(RasterConfig::default().tile_size, RASTERIZE_TILE_SIZE);
    }

    #[test]
    fn test_uploaded_model_identity_distinguishes_models() {
        let a = make_model(4);
        let b = make_model(4);
        let id_a = UploadedModel::of(&a);
        assert_eq!(id_a, UploadedModel::of(&a));
        assert!(id_a.describes(&a));
        // Two distinct allocations must not be mistaken for one another: that
        // is exactly what made `forward()` render a stale scene when called
        // with a second model.
        assert_ne!(id_a, UploadedModel::of(&b));
        assert!(
            !id_a.describes(&b),
            "a second model of the same size must still force a re-upload"
        );
    }

    #[test]
    fn test_uploaded_model_identity_detects_sh_degree_change() {
        let mut model = make_model(4);
        let id = UploadedModel::of(&model);
        // The SH stride the buffers were sized for is part of the identity.
        model.sh_degree = 1;
        assert!(!id.describes(&model));
    }

    #[test]
    fn test_uploaded_model_identity_tracks_gaussian_count() {
        let small = make_model(4);
        let large = make_model(8);
        assert_ne!(UploadedModel::of(&small).len, UploadedModel::of(&large).len);
        assert_eq!(UploadedModel::of(&small).len, 4);
        assert_eq!(UploadedModel::of(&large).sh_len, 8 * 3);
    }

    // ---- In-place update (update_gaussians) ---------------------------------

    /// `update_gaussians` may only skip the reallocation when the resident
    /// buffers still have the right *shape*. The allocation identity must
    /// therefore ignore the backing pointer (a `Vec` that reallocated, or a
    /// second model of identical shape, still fits) while catching every
    /// change that resizes a buffer or invalidates a bind group.
    #[test]
    fn test_allocation_identity_ignores_pointer_but_tracks_shape() {
        let a = make_model(4);
        let b = make_model(4);
        let id = UploadedModel::of(&a);

        // Different allocation, identical shape: contents can be rewritten
        // in place, so no reallocation is needed...
        assert!(id.describes_allocation(&b));
        // ...even though it is emphatically not the same model.
        assert!(!id.describes(&b));

        // Any shape change must force the full upload path.
        let bigger = make_model(8);
        assert!(!id.describes_allocation(&bigger));

        let mut degree_changed = make_model(4);
        degree_changed.sh_degree = 2;
        assert!(!id.describes_allocation(&degree_changed));

        let mut sh_changed = make_model(4);
        sh_changed.sh_coeffs.push(0.0);
        assert!(!id.describes_allocation(&sh_changed));
    }

    // ---- Readback staging size policy --------------------------------------

    #[test]
    fn test_check_staging_fits_refuses_sizes_past_the_device_limit() {
        // Regression: the direct fallback in `stage_readback` created the
        // buffer blind whenever the pool returned `None`. The pool returns
        // `None` for a request above `max_buffer_size` precisely because
        // creating it trips wgpu validation, which aborts the process through
        // the uncaptured-error handler; the fallback would have walked
        // straight into the abort the pool exists to avoid.
        const MAX: u64 = 256 * 1024 * 1024;

        assert!(
            check_staging_fits(MAX, MAX).is_ok(),
            "exactly at the limit is allowed"
        );
        assert!(check_staging_fits(wgpu::MAP_ALIGNMENT, MAX).is_ok());

        match check_staging_fits(MAX + wgpu::MAP_ALIGNMENT, MAX) {
            Err(RenderError::BufferOverflow {
                buffer_name,
                max_size,
                requested,
            }) => {
                assert_eq!(buffer_name, STAGING_READ_LABEL);
                assert_eq!(max_size, MAX);
                assert_eq!(requested, MAX + wgpu::MAP_ALIGNMENT);
            }
            other => panic!("expected BufferOverflow, got {other:?}"),
        }
    }

    #[test]
    fn test_staging_size_only_exceeds_a_misaligned_device_limit() {
        // `staging_size` never comes from user input: it is the source
        // buffer's own size (already validated against the device) rounded up
        // to MAP_ALIGNMENT. With an aligned limit -- what every real adapter
        // reports -- the rounding can never push a legal size past it, which
        // is why the guard above is unreachable in this rasterizer...
        const MAX: u64 = 256 * 1024 * 1024;
        assert!(MAX.is_multiple_of(wgpu::MAP_ALIGNMENT));
        for byte_size in [0, 4, 7, 8, 4095, MAX - 4, MAX] {
            let staging = staging_size_for(byte_size);
            assert!(staging >= byte_size.max(wgpu::MAP_ALIGNMENT));
            assert!(staging.is_multiple_of(wgpu::MAP_ALIGNMENT));
            assert!(
                check_staging_fits(staging, MAX).is_ok(),
                "byte_size {byte_size} rounds to {staging}, still within {MAX}"
            );
        }

        // ...and exactly when the limit is *not* a multiple of the map
        // alignment does the rounding create a size the device cannot hold.
        let misaligned = MAX + 4;
        assert!(!misaligned.is_multiple_of(wgpu::MAP_ALIGNMENT));
        assert!(
            check_staging_fits(staging_size_for(misaligned), misaligned).is_err(),
            "rounding a legal misaligned size up must be refused, not created"
        );
    }
}
