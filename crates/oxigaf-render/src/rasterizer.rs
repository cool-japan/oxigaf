//! Main rasterizer: orchestrates the full forward and backward passes.

use wgpu::util::DeviceExt;

use crate::buffers::{
    GaussianBuffers, GradientBuffers, IntermediateBuffers, OutputBuffers, UniformBuffer, Uniforms,
};
use crate::config::RasterConfig;
use crate::gaussian::GaussianModel;
use crate::pipeline::RasterPipelines;
use crate::pool::{BufferPool, PoolStats};
use crate::sort::RadixSorter;
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

/// The GPU-accelerated 3DGS rasterizer.
pub struct Rasterizer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: RasterConfig,
    pipelines: RasterPipelines,
    sorter: RadixSorter,
    uniform_buf: UniformBuffer,
    gaussian_bufs: Option<GaussianBuffers>,
    intermediate_bufs: Option<IntermediateBuffers>,
    output_bufs: Option<OutputBuffers>,
    gradient_bufs: Option<GradientBuffers>,
    /// Buffer pool for efficient memory reuse.
    buffer_pool: Option<BufferPool>,
}

impl Rasterizer {
    /// Create a new rasterizer from existing wgpu device and queue.
    pub fn from_device(
        device: wgpu::Device,
        queue: wgpu::Queue,
        config: RasterConfig,
    ) -> Result<Self, RenderError> {
        let pipelines = RasterPipelines::new(&device, &config)?;
        let sorter = RadixSorter::new(&device, 1024 * 1024)?; // 1M pairs initial capacity
        let uniform_buf = UniformBuffer::new(&device);

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
            pipelines,
            sorter,
            uniform_buf,
            gaussian_bufs: None,
            intermediate_bufs: None,
            output_bufs: None,
            gradient_bufs: None,
            buffer_pool,
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
            wgpu::Instance::new(&wgpu::InstanceDescriptor {
                backends: wgpu::Backends::all(),
                flags: wgpu::InstanceFlags::VALIDATION | wgpu::InstanceFlags::DEBUG,
                ..Default::default()
            })
        };

        #[cfg(not(feature = "gpu_debug"))]
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .map_err(|e| RenderError::GpuInit(format!("No suitable GPU adapter: {e}")))?;

        tracing::info!(
            adapter = adapter.get_info().name,
            backend = ?adapter.get_info().backend,
            "Selected GPU adapter"
        );

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("oxigaf_render"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits {
                    // Increase storage buffer limit for backward pass (needs 13+ buffers)
                    max_storage_buffers_per_shader_stage: 16,
                    ..wgpu::Limits::default()
                },
                memory_hints: wgpu::MemoryHints::Performance,
                experimental_features: wgpu::ExperimentalFeatures::default(),
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| RenderError::GpuInit(e.to_string()))?;

        Self::from_device(device, queue, config)
    }

    /// Upload Gaussian model data to the GPU.
    pub fn upload_gaussians(&mut self, model: &GaussianModel) {
        let n = model.len() as u32;

        self.gaussian_bufs = Some(GaussianBuffers::from_model(&self.device, model));
        self.intermediate_bufs = Some(IntermediateBuffers::allocate(&self.device, n, &self.config));
        self.output_bufs = Some(OutputBuffers::allocate(&self.device, &self.config));
        self.gradient_bufs = Some(GradientBuffers::allocate(
            &self.device,
            n,
            n * self.config.sh_coeffs_per_gaussian(),
        ));

        tracing::debug!(n_gaussians = n, "Uploaded Gaussians to GPU");
    }

    /// Run the forward rasterization pass.
    pub fn forward(
        &mut self,
        model: &GaussianModel,
        camera: &RenderCamera,
    ) -> Result<RenderOutput, RenderError> {
        // Ensure buffers are uploaded
        if self.gaussian_bufs.is_none() {
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
        let n = gauss.count;

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

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("forward_pass"),
            });

        // ---- Step 1: Preprocess ----
        {
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("preprocess_bg"),
                layout: &self.pipelines.preprocess_bgl,
                entries: &[
                    entry(0, &self.uniform_buf.buffer),
                    entry(1, &gauss.positions),
                    entry(2, &gauss.rotations),
                    entry(3, &gauss.scales),
                    entry(4, &gauss.opacities),
                    entry(5, &inter.means2d),
                    entry(6, &inter.cov2d),
                    entry(7, &inter.conics),
                    entry(8, &inter.depths),
                    entry(9, &inter.radii),
                    entry(10, &inter.tile_counts),
                    entry(11, &gauss.sh_coeffs),
                    entry(12, &inter.colors),
                    entry(13, &inter.normals), // for optional normal output
                ],
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("preprocess"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.preprocess);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(n.div_ceil(256), 1, 1);
        }

        // ---- Step 2: Hierarchical prefix sum ----
        {
            let num_wg = n.div_ceil(512);

            // Phase 1: Local scan + write block totals
            let params = [n, 0u32, 0, 0];
            let params_buf = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("prefix_sum_params"),
                    contents: bytemuck::cast_slice(&params),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("prefix_sum_bg"),
                layout: &self.pipelines.prefix_sum_bgl,
                entries: &[
                    entry(0, &inter.tile_counts),
                    entry(1, &inter.tile_offsets),
                    entry(2, &params_buf),
                    entry(3, &inter.block_sums),
                ],
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("prefix_sum"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.prefix_sum);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(num_wg, 1, 1);
            drop(pass);

            // Phase 2: Scan block sums (if more than one workgroup)
            if num_wg > 1 {
                // For up to 512 workgroups (covering 262K Gaussians), this fits in one dispatch.
                // For more, we'd need recursion, but 262K covers most practical scenes.
                let bs_params = [num_wg, 0u32, 0, 0];
                let bs_params_buf =
                    self.device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("block_sums_params"),
                            contents: bytemuck::cast_slice(&bs_params),
                            usage: wgpu::BufferUsages::UNIFORM,
                        });
                // Need a dummy block_sums for the nested scan (won't be used if num_wg <= 512)
                let dummy_block_sums = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("dummy_block_sums"),
                    size: 16,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let bg2 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("block_sums_scan_bg"),
                    layout: &self.pipelines.prefix_sum_bgl,
                    entries: &[
                        entry(0, &inter.block_sums),
                        entry(1, &inter.block_sums_scanned),
                        entry(2, &bs_params_buf),
                        entry(3, &dummy_block_sums),
                    ],
                });
                let num_wg2 = num_wg.div_ceil(512);
                let mut pass2 = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("block_sums_scan"),
                    timestamp_writes: None,
                });
                pass2.set_pipeline(&self.pipelines.prefix_sum);
                pass2.set_bind_group(0, &bg2, &[]);
                pass2.dispatch_workgroups(num_wg2, 1, 1);
                drop(pass2);

                // Phase 3: Add scanned block offsets back to each element
                let add_params = [n, 0u32, 0, 0];
                let add_params_buf =
                    self.device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("prefix_sum_add_params"),
                            contents: bytemuck::cast_slice(&add_params),
                            usage: wgpu::BufferUsages::UNIFORM,
                        });
                let bg3 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("prefix_sum_add_bg"),
                    layout: &self.pipelines.prefix_sum_add_bgl,
                    entries: &[
                        entry(0, &inter.tile_offsets),
                        entry(1, &inter.block_sums_scanned),
                        entry(2, &add_params_buf),
                    ],
                });
                let mut pass3 = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("prefix_sum_add"),
                    timestamp_writes: None,
                });
                pass3.set_pipeline(&self.pipelines.prefix_sum_add);
                pass3.set_bind_group(0, &bg3, &[]);
                pass3.dispatch_workgroups(num_wg, 1, 1);
            }
        }

        // ---- Step 2b: Clear sort and tile_ranges buffers ----
        // sort_keys and sort_values are allocated for max_pairs entries, but only
        // actual_pairs (determined by the prefix sum) will be written by tile_assign.
        // Uninitialized entries contain garbage. After radix sort, garbage entries
        // with zero keys would sort to the beginning, corrupting tile 0's range.
        //
        // Fix: fill sort_keys with 0xFF bytes so every uninitialized entry has
        // tile_id = 0xFFFFFFFF (greater than any valid num_tiles). The tile_ranges
        // shader's `tile_id < num_tiles` guard will skip them.
        // sort_values are also filled with 0xFF (invalid Gaussian index 0xFFFFFFFF)
        // as a safety measure, though they should never be reached.
        //
        // tile_ranges must be zero-cleared so tiles with no Gaussians get range
        // (0, 0) meaning empty, rather than leftover garbage from a previous frame.
        {
            let sort_keys_byte_size = inter.sort_keys.size() as usize;
            let fill_data_keys = vec![0xFFu8; sort_keys_byte_size];
            self.queue
                .write_buffer(&inter.sort_keys, 0, &fill_data_keys);

            let sort_values_byte_size = inter.sort_values.size() as usize;
            let fill_data_values = vec![0xFFu8; sort_values_byte_size];
            self.queue
                .write_buffer(&inter.sort_values, 0, &fill_data_values);

            encoder.clear_buffer(&inter.tile_ranges, 0, None);
        }

        // ---- Step 3: Tile assign ----
        {
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("tile_assign_bg"),
                layout: &self.pipelines.tile_assign_bgl,
                entries: &[
                    entry(0, &self.uniform_buf.buffer),
                    entry(1, &inter.means2d),
                    entry(2, &inter.depths),
                    entry(3, &inter.radii),
                    entry(4, &inter.tile_offsets),
                    entry(5, &inter.sort_keys),
                    entry(6, &inter.sort_values),
                ],
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("tile_assign"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.tile_assign);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(n.div_ceil(256), 1, 1);
        }

        // ---- Step 4: Radix sort ----
        self.sorter.sort(
            &mut encoder,
            &self.device,
            &inter.sort_keys,
            &inter.sort_values,
            inter.max_pairs,
        );

        // ---- Step 5: Tile ranges ----
        {
            let params = [inter.max_pairs, self.config.num_tiles(), 0u32, 0];
            let params_buf = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("tile_ranges_params"),
                    contents: bytemuck::cast_slice(&params),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("tile_ranges_bg"),
                layout: &self.pipelines.tile_ranges_bgl,
                entries: &[
                    entry(0, &inter.sort_keys),
                    entry(1, &inter.tile_ranges),
                    entry(2, &params_buf),
                ],
            });
            let num_wg = inter.max_pairs.div_ceil(256);
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("tile_ranges"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.tile_ranges);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(num_wg, 1, 1);
        }

        // ---- Step 6: Forward rasterization ----
        {
            // Create dummy normals buffer if normals output is disabled
            // (needed to satisfy bind group layout)
            let dummy_normals_buf;
            let out_normals_buf = if let Some(ref normals_buf) = output.normals {
                normals_buf
            } else {
                dummy_normals_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("dummy_out_normals"),
                    size: 16, // minimum size
                    usage: wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                });
                &dummy_normals_buf
            };

            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("rasterize_fwd_bg"),
                layout: &self.pipelines.rasterize_fwd_bgl,
                entries: &[
                    entry(0, &self.uniform_buf.buffer),
                    entry(1, &inter.means2d),
                    entry(2, &inter.conics),
                    entry(3, &inter.colors),
                    entry(4, &gauss.opacities),
                    entry(5, &inter.depths),
                    entry(6, &inter.tile_ranges),
                    entry(7, &inter.sort_values),
                    entry(8, &output.color),
                    entry(9, &output.depth),
                    entry(10, &output.transmittance),
                    entry(11, &output.n_contrib),
                    entry(12, &inter.normals),
                    entry(13, out_normals_buf),
                ],
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rasterize_fwd"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.rasterize_fwd);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(self.config.tiles_x(), self.config.tiles_y(), 1);
        }

        // Submit
        self.queue.submit(std::iter::once(encoder.finish()));

        // Read back color and depth
        let color_data =
            self.read_buffer_f32(&output.color, self.config.num_pixels() as usize * 4)?;
        let depth_data = self.read_buffer_f32(&output.depth, self.config.num_pixels() as usize)?;

        // Read back normals if enabled
        let normals = if self.config.output_normals {
            if let Some(ref normals_buf) = output.normals {
                let normals_raw =
                    self.read_buffer_f32(normals_buf, self.config.num_pixels() as usize * 4)?;
                // Convert from vec4<f32> GPU layout to [f32; 3] CPU layout
                let normals_vec: Vec<[f32; 3]> =
                    normals_raw.chunks(4).map(|c| [c[0], c[1], c[2]]).collect();
                Some(normals_vec)
            } else {
                None
            }
        } else {
            None
        };

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
    pub fn backward(
        &mut self,
        model: &GaussianModel,
        grad_image: &[f32],
    ) -> Result<GaussianGradients, RenderError> {
        let gauss = self.gaussian_bufs.as_ref().ok_or_else(|| {
            RenderError::Rasterize("backward() called before upload_gaussians()".into())
        })?;
        let inter = self
            .intermediate_bufs
            .as_ref()
            .ok_or_else(|| RenderError::Rasterize("backward() called before forward()".into()))?;
        let output = self
            .output_bufs
            .as_ref()
            .ok_or_else(|| RenderError::Rasterize("backward() called before forward()".into()))?;
        let grads = self
            .gradient_bufs
            .as_ref()
            .ok_or_else(|| RenderError::Rasterize("backward() called before forward()".into()))?;
        let n = gauss.count;

        // Upload the image-space gradient to a temporary buffer.
        let grad_output_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("grad_output"),
                contents: bytemuck::cast_slice(grad_image),
                usage: wgpu::BufferUsages::STORAGE,
            });

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
        // Writes 2D gradients to atomic buffers: grad_colors_atomic, grad_opacities, grad_means2d_atomic, grad_conics_atomic
        {
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("rasterize_bwd_bg"),
                layout: &self.pipelines.rasterize_bwd_bgl,
                entries: &[
                    entry(0, &self.uniform_buf.buffer),
                    entry(1, &inter.means2d),
                    entry(2, &inter.conics),
                    entry(3, &inter.colors),
                    entry(4, &gauss.opacities),
                    entry(5, &output.color),
                    entry(6, &output.transmittance),
                    entry(7, &output.n_contrib),
                    entry(8, &inter.tile_ranges),
                    entry(9, &inter.sort_values),
                    entry(10, &grad_output_buf),
                    entry(11, &grads.grad_colors_atomic),
                    entry(12, &grads.grad_opacities),
                    entry(13, &grads.grad_means2d_atomic),
                    entry(14, &grads.grad_conics_atomic),
                ],
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rasterize_bwd"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.rasterize_bwd);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(self.config.tiles_x(), self.config.tiles_y(), 1);
        }

        // --- Atomic to f32 conversion ---
        // Convert atomic buffers (written by rasterize_bwd) to regular f32 buffers (read by preprocess_bwd)
        // grad_means2d_atomic → grad_means2d
        {
            let num_elements_buf =
                self.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("num_elements_means2d"),
                        contents: bytemuck::cast_slice(&[n * 2]),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("atomic_to_f32_means2d_bg"),
                layout: &self.pipelines.atomic_to_f32_bgl,
                entries: &[
                    entry(0, &num_elements_buf),
                    entry(1, &grads.grad_means2d_atomic),
                    entry(2, &grads.grad_means2d),
                ],
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("atomic_to_f32_means2d"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.atomic_to_f32);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups((n * 2).div_ceil(256), 1, 1);
        }
        // grad_conics_atomic → grad_conics
        {
            let num_elements_buf =
                self.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("num_elements_conics"),
                        contents: bytemuck::cast_slice(&[n * 3]),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("atomic_to_f32_conics_bg"),
                layout: &self.pipelines.atomic_to_f32_bgl,
                entries: &[
                    entry(0, &num_elements_buf),
                    entry(1, &grads.grad_conics_atomic),
                    entry(2, &grads.grad_conics),
                ],
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("atomic_to_f32_conics"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.atomic_to_f32);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups((n * 3).div_ceil(256), 1, 1);
        }
        // grad_colors_atomic → grad_colors
        {
            let num_elements_buf =
                self.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("num_elements_colors"),
                        contents: bytemuck::cast_slice(&[n * 3]),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("atomic_to_f32_colors_bg"),
                layout: &self.pipelines.atomic_to_f32_bgl,
                entries: &[
                    entry(0, &num_elements_buf),
                    entry(1, &grads.grad_colors_atomic),
                    entry(2, &grads.grad_colors),
                ],
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("atomic_to_f32_colors"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.atomic_to_f32);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups((n * 3).div_ceil(256), 1, 1);
        }

        // --- Preprocess backward ---
        // Chains 2D gradients through projection to 3D gradients
        {
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("preprocess_bwd_bg"),
                layout: &self.pipelines.preprocess_bwd_bgl,
                entries: &[
                    entry(0, &self.uniform_buf.buffer),
                    entry(1, &gauss.positions),
                    entry(2, &gauss.rotations),
                    entry(3, &gauss.scales),
                    entry(4, &inter.cov2d),
                    entry(5, &inter.conics),
                    entry(6, &gauss.sh_coeffs),
                    entry(7, &grads.grad_means2d),
                    entry(8, &grads.grad_conics),
                    entry(9, &grads.grad_colors),
                    entry(10, &grads.grad_positions),
                    entry(11, &grads.grad_rotations),
                    entry(12, &grads.grad_scales),
                    entry(13, &grads.grad_sh_coeffs),
                ],
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("preprocess_bwd"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.preprocess_bwd);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(n.div_ceil(256), 1, 1);
        }

        // Submit and wait for GPU completion.
        self.queue.submit(std::iter::once(encoder.finish()));
        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });

        // Read back gradients.
        let grad_positions = self.read_buffer_f32(&grads.grad_positions, n as usize * 4)?;
        let grad_rotations = self.read_buffer_f32(&grads.grad_rotations, n as usize * 4)?;
        let grad_scales = self.read_buffer_f32(&grads.grad_scales, n as usize * 4)?;
        let grad_opacities = self.read_buffer_f32(&grads.grad_opacities, n as usize)?;
        let sh_total = model.sh_coeffs.len();
        let grad_sh_coeffs = self.read_buffer_f32(&grads.grad_sh_coeffs, sh_total.max(1))?;

        // Convert from padded [f32; 4] GPU layout to dense [f32; 3] CPU layout
        // for positions and scales.
        let unpad3 = |data: &[f32]| -> Vec<[f32; 3]> {
            data.chunks(4).map(|c| [c[0], c[1], c[2]]).collect()
        };
        let unpad4 = |data: &[f32]| -> Vec<[f32; 4]> {
            data.chunks(4).map(|c| [c[0], c[1], c[2], c[3]]).collect()
        };

        Ok(GaussianGradients {
            grad_positions: unpad3(&grad_positions),
            grad_rotations: unpad4(&grad_rotations),
            grad_scales: unpad3(&grad_scales),
            grad_opacities,
            grad_sh_coeffs,
        })
    }

    /// Download rendered image as an `image::RgbaImage`.
    pub fn download_image(&self, output: &RenderOutput) -> image::RgbaImage {
        let w = output.width;
        let h = output.height;
        let mut img = image::RgbaImage::new(w, h);

        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) as usize;
                let r = (output.color_data[idx * 4].clamp(0.0, 1.0) * 255.0) as u8;
                let g = (output.color_data[idx * 4 + 1].clamp(0.0, 1.0) * 255.0) as u8;
                let b = (output.color_data[idx * 4 + 2].clamp(0.0, 1.0) * 255.0) as u8;
                let a = (output.color_data[idx * 4 + 3].clamp(0.0, 1.0) * 255.0) as u8;
                img.put_pixel(x, y, image::Rgba([r, g, b, a]));
            }
        }

        img
    }

    /// Get a reference to the config.
    pub fn config(&self) -> &RasterConfig {
        &self.config
    }

    /// Get a reference to the buffer pool, if enabled.
    pub fn buffer_pool(&self) -> Option<&BufferPool> {
        self.buffer_pool.as_ref()
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

    // --- Internal helpers ---

    fn read_buffer_f32(
        &self,
        buffer: &wgpu::Buffer,
        count: usize,
    ) -> Result<Vec<f32>, RenderError> {
        let byte_size = (count * 4) as u64;

        // Use buffer pool if available (saves 18MB per frame)
        let staging_pooled = if let Some(ref pool) = self.buffer_pool {
            pool.acquire(
                &self.device,
                byte_size,
                wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                "staging_read",
            )
        } else {
            None
        };

        let staging_direct;
        let staging: &wgpu::Buffer = if let Some(ref pooled) = staging_pooled {
            pooled
        } else {
            staging_direct = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("staging_read"),
                size: byte_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            &staging_direct
        };

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("readback"),
            });
        encoder.copy_buffer_to_buffer(buffer, 0, staging, 0, byte_size);
        self.queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).ok();
        });
        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        rx.recv()
            .map_err(|e| RenderError::Rasterize(format!("Channel recv failed: {e}")))?
            .map_err(|e| RenderError::Rasterize(format!("Buffer map failed: {e}")))?;

        let data = slice.get_mapped_range();
        let all_floats: &[f32] = bytemuck::cast_slice(&data);
        // Only return the requested count, not the entire buffer
        let floats = all_floats[..count.min(all_floats.len())].to_vec();
        drop(data);
        staging.unmap();

        Ok(floats)
    }
}

fn entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}
