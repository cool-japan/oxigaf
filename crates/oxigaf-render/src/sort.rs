//! GPU radix sort for Gaussian depth-ordering within tiles.
//!
//! Sorts `(key, value)` pairs where `key = (tile_id << 32) | depth_bits`
//! and `value = gaussian_index`. This places Gaussians front-to-back
//! within each tile for correct alpha-blending.
//!
//! Uses a three-pass per-digit approach:
//! 1. **Histogram**: count per-digit occurrences per workgroup
//! 2. **Prefix sum**: inclusive scan of the histogram buffer
//! 3. **Scatter**: place elements at their sorted positions

use crate::RenderError;
use wgpu::util::DeviceExt;

/// GPU radix sort state.
pub struct RadixSorter {
    // Pipelines
    histogram_pipeline: wgpu::ComputePipeline,
    scatter_pipeline: wgpu::ComputePipeline,
    prefix_sum_pipeline: wgpu::ComputePipeline,
    prefix_sum_add_pipeline: wgpu::ComputePipeline,
    // Bind group layouts
    histogram_bgl: wgpu::BindGroupLayout,
    scatter_bgl: wgpu::BindGroupLayout,
    prefix_sum_bgl: wgpu::BindGroupLayout,
    prefix_sum_add_bgl: wgpu::BindGroupLayout,
    // Scratch buffers
    pub scratch_keys: wgpu::Buffer,
    pub scratch_values: wgpu::Buffer,
    /// Histogram buffer: 16 * num_workgroups elements
    pub histogram_buf: wgpu::Buffer,
    /// Prefix-summed histogram
    pub histogram_prefix_buf: wgpu::Buffer,
    /// Block sums for histogram prefix sum
    pub hist_block_sums: wgpu::Buffer,
    /// Scanned block sums
    pub hist_block_sums_scanned: wgpu::Buffer,
    max_elements: u32,
}

/// Number of bits to sort per pass (4-bit radix = 16 buckets).
const RADIX_BITS: u32 = 4;
/// Number of buckets per pass.
const NUM_BUCKETS: u32 = 1 << RADIX_BITS;
/// Workgroup size for sort kernels.
const SORT_WG_SIZE: u32 = 256;
/// Number of passes for 64-bit keys.
const NUM_PASSES: u32 = 64 / RADIX_BITS;

impl RadixSorter {
    /// Create a new radix sorter.
    pub fn new(device: &wgpu::Device, max_elements: u32) -> Result<Self, RenderError> {
        let num_wg = max_elements.div_ceil(SORT_WG_SIZE);
        let histogram_size = (NUM_BUCKETS * num_wg) as u64;

        let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;

        // --- Histogram pipeline ---
        let histogram_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("radix_histogram_bgl"),
            entries: &[
                bgl_entry(0, true),   // keys_in
                bgl_entry(1, false),  // histogram
                uniform_bgl_entry(2), // params
            ],
        });
        let histogram_pipeline = compile_pipeline(
            device,
            "radix_histogram",
            include_str!("../shaders/radix_histogram.wgsl"),
            "radix_histogram",
            &histogram_bgl,
        );

        // --- Scatter pipeline ---
        let scatter_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("radix_scatter_bgl"),
            entries: &[
                bgl_entry(0, true),   // keys_in
                bgl_entry(1, true),   // values_in
                bgl_entry(2, false),  // keys_out
                bgl_entry(3, false),  // values_out
                bgl_entry(4, true),   // histogram (original)
                bgl_entry(5, true),   // histogram_prefix (scanned)
                uniform_bgl_entry(6), // params
            ],
        });
        let scatter_pipeline = compile_pipeline(
            device,
            "radix_scatter",
            include_str!("../shaders/radix_scatter.wgsl"),
            "radix_scatter",
            &scatter_bgl,
        );

        // --- Prefix sum pipeline (recompiled for the sorter) ---
        let prefix_sum_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sort_prefix_sum_bgl"),
            entries: &[
                bgl_entry(0, true),   // input (read-only)
                bgl_entry(1, false),  // output
                uniform_bgl_entry(2), // params
                bgl_entry(3, false),  // block_sums
            ],
        });
        let prefix_sum_pipeline = compile_pipeline(
            device,
            "sort_prefix_sum",
            include_str!("../shaders/prefix_sum.wgsl"),
            "prefix_sum",
            &prefix_sum_bgl,
        );

        // --- Prefix sum add pipeline ---
        let prefix_sum_add_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("sort_prefix_sum_add_bgl"),
                entries: &[
                    bgl_entry(0, false),  // data (in-place)
                    bgl_entry(1, true),   // block_offsets
                    uniform_bgl_entry(2), // params
                ],
            });
        let prefix_sum_add_pipeline = compile_pipeline(
            device,
            "sort_prefix_sum_add",
            include_str!("../shaders/prefix_sum_add.wgsl"),
            "prefix_sum_add",
            &prefix_sum_add_bgl,
        );

        // Allocate buffers
        let scratch_keys = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sort_scratch_keys"),
            size: (max_elements as u64) * 8,
            usage: storage,
            mapped_at_creation: false,
        });
        let scratch_values = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sort_scratch_values"),
            size: (max_elements as u64) * 4,
            usage: storage,
            mapped_at_creation: false,
        });
        let histogram_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sort_histogram"),
            size: histogram_size.max(16) * 4,
            usage: storage,
            mapped_at_creation: false,
        });
        let histogram_prefix_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sort_histogram_prefix"),
            size: histogram_size.max(16) * 4,
            usage: storage,
            mapped_at_creation: false,
        });
        let hist_ps_wg = histogram_size.div_ceil(512).max(1);
        let hist_block_sums = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sort_hist_block_sums"),
            size: hist_ps_wg * 4,
            usage: storage,
            mapped_at_creation: false,
        });
        let hist_block_sums_scanned = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sort_hist_block_sums_scanned"),
            size: hist_ps_wg * 4,
            usage: storage,
            mapped_at_creation: false,
        });

        Ok(Self {
            histogram_pipeline,
            scatter_pipeline,
            prefix_sum_pipeline,
            prefix_sum_add_pipeline,
            histogram_bgl,
            scatter_bgl,
            prefix_sum_bgl,
            prefix_sum_add_bgl,
            scratch_keys,
            scratch_values,
            histogram_buf,
            histogram_prefix_buf,
            hist_block_sums,
            hist_block_sums_scanned,
            max_elements,
        })
    }

    /// Record sort commands into the encoder.
    ///
    /// After execution, the sorted keys/values will be in the *input*
    /// buffers (ping-pong for even number of passes).
    pub fn sort(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
        keys: &wgpu::Buffer,
        values: &wgpu::Buffer,
        count: u32,
    ) {
        if count == 0 {
            return;
        }

        let num_wg = count.div_ceil(SORT_WG_SIZE);
        let histogram_count = NUM_BUCKETS * num_wg;

        for pass in 0..NUM_PASSES {
            let (k_in, k_out, v_in, v_out) = if pass % 2 == 0 {
                (keys, &self.scratch_keys, values, &self.scratch_values)
            } else {
                (&self.scratch_keys, keys, &self.scratch_values, values)
            };

            let pass_params = [count, pass, RADIX_BITS, num_wg];
            let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("sort_params"),
                contents: bytemuck::cast_slice(&pass_params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

            // Clear histogram
            encoder.clear_buffer(&self.histogram_buf, 0, None);

            // --- Pass 1: Histogram ---
            {
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("hist_bg"),
                    layout: &self.histogram_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: k_in.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: self.histogram_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: params_buf.as_entire_binding(),
                        },
                    ],
                });
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("radix_histogram"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.histogram_pipeline);
                cpass.set_bind_group(0, &bg, &[]);
                cpass.dispatch_workgroups(num_wg, 1, 1);
            }

            // --- Pass 2: Prefix sum of histogram ---
            {
                let ps_params = [histogram_count, 0u32, 0, 0];
                let ps_params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("hist_ps_params"),
                    contents: bytemuck::cast_slice(&ps_params),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
                let ps_num_wg = histogram_count.div_ceil(512);
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("hist_ps_bg"),
                    layout: &self.prefix_sum_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.histogram_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: self.histogram_prefix_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: ps_params_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: self.hist_block_sums.as_entire_binding(),
                        },
                    ],
                });
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("hist_prefix_sum"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.prefix_sum_pipeline);
                cpass.set_bind_group(0, &bg, &[]);
                cpass.dispatch_workgroups(ps_num_wg, 1, 1);
                drop(cpass);

                // If histogram needs multi-workgroup scan, add block offsets
                if ps_num_wg > 1 {
                    // Scan block sums
                    let bs_params = [ps_num_wg, 0u32, 0, 0];
                    let bs_params_buf =
                        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("hist_bs_params"),
                            contents: bytemuck::cast_slice(&bs_params),
                            usage: wgpu::BufferUsages::UNIFORM,
                        });
                    let dummy = device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("dummy_bs"),
                        size: 16,
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                    let bg2 = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("hist_bs_bg"),
                        layout: &self.prefix_sum_bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: self.hist_block_sums.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: self.hist_block_sums_scanned.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: bs_params_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: dummy.as_entire_binding(),
                            },
                        ],
                    });
                    let mut cpass2 = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("hist_bs_scan"),
                        timestamp_writes: None,
                    });
                    cpass2.set_pipeline(&self.prefix_sum_pipeline);
                    cpass2.set_bind_group(0, &bg2, &[]);
                    cpass2.dispatch_workgroups(1, 1, 1);
                    drop(cpass2);

                    // Add block offsets back
                    let add_params = [histogram_count, 0u32, 0, 0];
                    let add_params_buf =
                        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("hist_add_params"),
                            contents: bytemuck::cast_slice(&add_params),
                            usage: wgpu::BufferUsages::UNIFORM,
                        });
                    let bg3 = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("hist_add_bg"),
                        layout: &self.prefix_sum_add_bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: self.histogram_prefix_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: self.hist_block_sums_scanned.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: add_params_buf.as_entire_binding(),
                            },
                        ],
                    });
                    let mut cpass3 = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("hist_add"),
                        timestamp_writes: None,
                    });
                    cpass3.set_pipeline(&self.prefix_sum_add_pipeline);
                    cpass3.set_bind_group(0, &bg3, &[]);
                    cpass3.dispatch_workgroups(ps_num_wg, 1, 1);
                }
            }

            // --- Pass 3: Scatter ---
            {
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("scatter_bg"),
                    layout: &self.scatter_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: k_in.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: v_in.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: k_out.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: v_out.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: self.histogram_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: self.histogram_prefix_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 6,
                            resource: params_buf.as_entire_binding(),
                        },
                    ],
                });
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("radix_scatter"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.scatter_pipeline);
                cpass.set_bind_group(0, &bg, &[]);
                cpass.dispatch_workgroups(num_wg, 1, 1);
            }
        }
    }

    /// Maximum elements this sorter was allocated for.
    pub fn capacity(&self) -> u32 {
        self.max_elements
    }
}

fn compile_pipeline(
    device: &wgpu::Device,
    label: &str,
    source: &str,
    entry_point: &str,
    bgl: &wgpu::BindGroupLayout,
) -> wgpu::ComputePipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(&format!("{label}_layout")),
        bind_group_layouts: &[bgl],
        immediate_size: 0,
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(&format!("{label}_pipeline")),
        layout: Some(&layout),
        module: &module,
        entry_point: Some(entry_point),
        compilation_options: Default::default(),
        cache: None,
    })
}

/// Helper to create a storage buffer bind group layout entry.
fn bgl_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// Helper for uniform bind group layout entry.
fn uniform_bgl_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
