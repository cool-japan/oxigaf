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
//!
//! # Capacity
//!
//! The sorter owns scratch and histogram buffers sized for a fixed element
//! count. [`RadixSorter::prepare`] is the single point where the key/value
//! buffers it works on can change: it grows those internal buffers when needed
//! and rebuilds every bind group. [`RadixSorter::sort`] then only *records*
//! commands and refuses a `count` it was not prepared for, instead of letting
//! WGSL bounds-clamping silently produce a mis-sorted array.
//!
//! # Per-frame allocations
//!
//! All uniform buffers and bind groups are created once (in [`RadixSorter::new`]
//! and [`RadixSorter::prepare`]) and only their *contents* are rewritten each
//! frame via `queue.write_buffer`. Nothing is allocated inside the pass loop.

use crate::buffers::PREFIX_SUM_BLOCK;
use crate::RenderError;

/// Number of bits to sort per pass (4-bit radix = 16 buckets).
const RADIX_BITS: u32 = 4;
/// Number of buckets per pass.
const NUM_BUCKETS: u32 = 1 << RADIX_BITS;
/// Workgroup size for sort kernels.
const SORT_WG_SIZE: u32 = 256;
/// Maximum number of passes for 64-bit keys.
const NUM_PASSES: u32 = 64 / RADIX_BITS;

/// Number of 4-bit radix passes actually required for the `(tile_id, depth)`
/// key layout produced by `tile_assign.wgsl`.
///
/// The key is 64 bits wide: the low 32 hold the raw bit pattern of a positive
/// `f32` depth (every bit significant), the high 32 hold `tile_id`, which is
/// bounded by `num_tiles`. At 1920×1080 with 16-pixel tiles there are 8160
/// tiles — 13 bits — so the passes covering bits 45..63 would histogram and
/// scatter a digit that is always zero. Skipping them removes roughly a quarter
/// of the sort work at no cost to the ordering.
///
/// The result is always **even**: [`RadixSorter::sort`] ping-pongs between the
/// caller's buffers and the internal scratch buffers, so an odd pass count
/// would leave the sorted result in the scratch buffers, where no later stage
/// looks for it.
fn radix_pass_count(num_tiles: u32) -> u32 {
    let max_tile_id = num_tiles.saturating_sub(1);
    let tile_bits = u32::BITS - max_tile_id.leading_zeros();
    // Low word (depth) is always fully significant.
    let key_bits = 32 + tile_bits;
    let passes = key_bits.div_ceil(RADIX_BITS).min(NUM_PASSES);
    if passes.is_multiple_of(2) {
        passes
    } else {
        passes + 1
    }
}

/// The scratch/histogram buffers a [`RadixSorter`] needs for a given capacity.
struct SortBuffers {
    scratch_keys: wgpu::Buffer,
    scratch_values: wgpu::Buffer,
    histogram_buf: wgpu::Buffer,
    histogram_prefix_buf: wgpu::Buffer,
    hist_block_sums: wgpu::Buffer,
    hist_block_sums_scanned: wgpu::Buffer,
    hist_block_sums_l2: wgpu::Buffer,
    hist_block_sums_l2_scanned: wgpu::Buffer,
}

/// Bind groups bound to one specific pair of key/value buffers.
///
/// Rebuilt by [`RadixSorter::prepare`] — the only place where the buffers they
/// reference can change — so the per-frame path creates none of them.
struct SortBindGroups {
    /// One per radix pass (the ping-pong direction alternates with the pass).
    histogram: Vec<wgpu::BindGroup>,
    /// One per radix pass.
    scatter: Vec<wgpu::BindGroup>,
    /// Level-0 scan: histogram → histogram_prefix.
    scan_l0: wgpu::BindGroup,
    /// Level-1 scan: hist_block_sums → hist_block_sums_scanned.
    scan_l1: wgpu::BindGroup,
    /// Level-2 scan: hist_block_sums_l2 → hist_block_sums_l2_scanned.
    scan_l2: wgpu::BindGroup,
    /// Add the level-2 offsets into hist_block_sums_scanned.
    add_l1: wgpu::BindGroup,
    /// Add the level-1 offsets into histogram_prefix.
    add_l0: wgpu::BindGroup,
}

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
    /// Level-2 block sums, emitted while scanning `hist_block_sums`.
    ///
    /// Needed whenever the level-1 scan itself spans more than one workgroup;
    /// without scanning these and adding them back, every histogram bucket past
    /// the first `PREFIX_SUM_BLOCK²` entries gets a wrong global offset and the
    /// scatter pass writes elements on top of each other.
    pub hist_block_sums_l2: wgpu::Buffer,
    /// Scanned level-2 block sums.
    pub hist_block_sums_l2_scanned: wgpu::Buffer,
    /// Sink for the block sums the top-level scan emits and nobody reads.
    dummy_block_sums: wgpu::Buffer,
    /// Per-pass `SortParams` uniforms (one per possible pass, reused each frame).
    pass_params: Vec<wgpu::Buffer>,
    /// Scan params holding the histogram element count.
    ps_params_count: wgpu::Buffer,
    /// Scan params holding the level-0 workgroup count.
    ps_params_wg: wgpu::Buffer,
    /// Scan params holding the level-1 workgroup count.
    ps_params_wg2: wgpu::Buffer,
    /// Bind groups for the currently prepared key/value buffers.
    bind_groups: Option<SortBindGroups>,
    max_elements: u32,
}

impl RadixSorter {
    /// Create a new radix sorter with an initial capacity of `max_elements`.
    ///
    /// The capacity grows on demand in [`prepare`](Self::prepare), so a small
    /// initial value is fine.
    ///
    /// # Errors
    ///
    /// Each shader/pipeline is compiled under a `wgpu` validation error
    /// scope: a WGSL parse/type error surfaces as
    /// [`RenderError::ShaderCompilation`], and a pipeline-layout
    /// compatibility error surfaces as [`RenderError::ShaderValidation`],
    /// instead of reaching `wgpu`'s uncaptured-error handler (a panic by
    /// default).
    pub fn new(device: &wgpu::Device, max_elements: u32) -> Result<Self, RenderError> {
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
        )?;

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
        )?;

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
        )?;

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
        )?;

        let max_elements = max_elements.max(1);
        let bufs = allocate_sort_buffers(device, max_elements);

        let dummy_block_sums = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sort_dummy_block_sums"),
            size: (PREFIX_SUM_BLOCK as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let uniform = wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST;
        let pass_params: Vec<wgpu::Buffer> = (0..NUM_PASSES)
            .map(|_| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("sort_pass_params"),
                    size: 16, // vec4<u32>
                    usage: uniform,
                    mapped_at_creation: false,
                })
            })
            .collect();
        let ps_params_count = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sort_ps_params_count"),
            size: 16,
            usage: uniform,
            mapped_at_creation: false,
        });
        let ps_params_wg = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sort_ps_params_wg"),
            size: 16,
            usage: uniform,
            mapped_at_creation: false,
        });
        let ps_params_wg2 = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sort_ps_params_wg2"),
            size: 16,
            usage: uniform,
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
            scratch_keys: bufs.scratch_keys,
            scratch_values: bufs.scratch_values,
            histogram_buf: bufs.histogram_buf,
            histogram_prefix_buf: bufs.histogram_prefix_buf,
            hist_block_sums: bufs.hist_block_sums,
            hist_block_sums_scanned: bufs.hist_block_sums_scanned,
            hist_block_sums_l2: bufs.hist_block_sums_l2,
            hist_block_sums_l2_scanned: bufs.hist_block_sums_l2_scanned,
            dummy_block_sums,
            pass_params,
            ps_params_count,
            ps_params_wg,
            ps_params_wg2,
            bind_groups: None,
            max_elements,
        })
    }

    /// Bind the sorter to a key/value buffer pair with capacity `max_elements`.
    ///
    /// Grows the internal scratch and histogram buffers when `max_elements`
    /// exceeds the current capacity, then rebuilds every bind group. Must be
    /// called whenever `keys`/`values` are (re)allocated — otherwise [`sort`]
    /// records commands against the previous buffers.
    ///
    /// [`sort`]: Self::sort
    pub fn prepare(
        &mut self,
        device: &wgpu::Device,
        keys: &wgpu::Buffer,
        values: &wgpu::Buffer,
        max_elements: u32,
    ) {
        let max_elements = max_elements.max(1);
        if max_elements > self.max_elements {
            let bufs = allocate_sort_buffers(device, max_elements);
            self.scratch_keys = bufs.scratch_keys;
            self.scratch_values = bufs.scratch_values;
            self.histogram_buf = bufs.histogram_buf;
            self.histogram_prefix_buf = bufs.histogram_prefix_buf;
            self.hist_block_sums = bufs.hist_block_sums;
            self.hist_block_sums_scanned = bufs.hist_block_sums_scanned;
            self.hist_block_sums_l2 = bufs.hist_block_sums_l2;
            self.hist_block_sums_l2_scanned = bufs.hist_block_sums_l2_scanned;
            tracing::debug!(
                old_capacity = self.max_elements,
                new_capacity = max_elements,
                "Grew radix sorter scratch buffers"
            );
            self.max_elements = max_elements;
        }

        let bind_groups = self.build_bind_groups(device, keys, values);
        self.bind_groups = Some(bind_groups);
    }

    fn build_bind_groups(
        &self,
        device: &wgpu::Device,
        keys: &wgpu::Buffer,
        values: &wgpu::Buffer,
    ) -> SortBindGroups {
        let mut histogram = Vec::with_capacity(NUM_PASSES as usize);
        let mut scatter = Vec::with_capacity(NUM_PASSES as usize);

        for pass in 0..NUM_PASSES {
            let (k_in, k_out, v_in, v_out) = if pass.is_multiple_of(2) {
                (keys, &self.scratch_keys, values, &self.scratch_values)
            } else {
                (&self.scratch_keys, keys, &self.scratch_values, values)
            };
            let params = &self.pass_params[pass as usize];

            histogram.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("radix_histogram_bg"),
                layout: &self.histogram_bgl,
                entries: &[
                    bg_entry(0, k_in),
                    bg_entry(1, &self.histogram_buf),
                    bg_entry(2, params),
                ],
            }));
            scatter.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("radix_scatter_bg"),
                layout: &self.scatter_bgl,
                entries: &[
                    bg_entry(0, k_in),
                    bg_entry(1, v_in),
                    bg_entry(2, k_out),
                    bg_entry(3, v_out),
                    bg_entry(4, &self.histogram_buf),
                    bg_entry(5, &self.histogram_prefix_buf),
                    bg_entry(6, params),
                ],
            }));
        }

        let scan_l0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hist_scan_l0_bg"),
            layout: &self.prefix_sum_bgl,
            entries: &[
                bg_entry(0, &self.histogram_buf),
                bg_entry(1, &self.histogram_prefix_buf),
                bg_entry(2, &self.ps_params_count),
                bg_entry(3, &self.hist_block_sums),
            ],
        });
        let scan_l1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hist_scan_l1_bg"),
            layout: &self.prefix_sum_bgl,
            entries: &[
                bg_entry(0, &self.hist_block_sums),
                bg_entry(1, &self.hist_block_sums_scanned),
                bg_entry(2, &self.ps_params_wg),
                bg_entry(3, &self.hist_block_sums_l2),
            ],
        });
        let scan_l2 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hist_scan_l2_bg"),
            layout: &self.prefix_sum_bgl,
            entries: &[
                bg_entry(0, &self.hist_block_sums_l2),
                bg_entry(1, &self.hist_block_sums_l2_scanned),
                bg_entry(2, &self.ps_params_wg2),
                bg_entry(3, &self.dummy_block_sums),
            ],
        });
        let add_l1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hist_add_l1_bg"),
            layout: &self.prefix_sum_add_bgl,
            entries: &[
                bg_entry(0, &self.hist_block_sums_scanned),
                bg_entry(1, &self.hist_block_sums_l2_scanned),
                bg_entry(2, &self.ps_params_wg),
            ],
        });
        let add_l0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hist_add_l0_bg"),
            layout: &self.prefix_sum_add_bgl,
            entries: &[
                bg_entry(0, &self.histogram_prefix_buf),
                bg_entry(1, &self.hist_block_sums_scanned),
                bg_entry(2, &self.ps_params_count),
            ],
        });

        SortBindGroups {
            histogram,
            scatter,
            scan_l0,
            scan_l1,
            scan_l2,
            add_l1,
            add_l0,
        }
    }

    /// Record sort commands into the encoder.
    ///
    /// Sorts the first `count` elements of the key/value buffers most recently
    /// passed to [`prepare`](Self::prepare). `num_tiles` bounds the `tile_id`
    /// half of the key and therefore how many radix passes are actually needed:
    /// digits above the highest possible tile bit are always zero and are
    /// skipped, which is roughly a quarter of the work at 1080p.
    ///
    /// After execution the sorted keys/values are back in the *input* buffers:
    /// the pass count is always even, so the ping-pong ends where it started.
    ///
    /// # Errors
    ///
    /// * [`RenderError::TooManyTilePairs`] when `count` exceeds the prepared
    ///   capacity — the internal scratch and histogram buffers would be
    ///   overrun, and WGSL bounds-clamping would turn that into a silently
    ///   mis-sorted array.
    /// * [`RenderError::Rasterize`] when [`prepare`](Self::prepare) has not been
    ///   called, or when the histogram would need more than three scan levels.
    pub fn sort(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        count: u32,
        num_tiles: u32,
    ) -> Result<(), RenderError> {
        if count == 0 {
            return Ok(());
        }
        if count > self.max_elements {
            return Err(RenderError::TooManyTilePairs {
                count,
                allocated: self.max_elements,
            });
        }
        let bgs = self.bind_groups.as_ref().ok_or_else(|| {
            RenderError::Rasterize(
                "RadixSorter::sort called before RadixSorter::prepare".to_string(),
            )
        })?;

        let num_wg = count.div_ceil(SORT_WG_SIZE);
        let histogram_count = NUM_BUCKETS * num_wg;
        let ps_num_wg = histogram_count.div_ceil(PREFIX_SUM_BLOCK);
        let ps_num_wg2 = ps_num_wg.div_ceil(PREFIX_SUM_BLOCK);
        if ps_num_wg2 > PREFIX_SUM_BLOCK {
            return Err(RenderError::Rasterize(format!(
                "radix sort histogram for {count} elements needs more than three scan levels"
            )));
        }
        let passes = radix_pass_count(num_tiles);

        // Per-frame parameter upload. Nothing is allocated here: the buffers
        // were created once and only their contents change.
        for pass in 0..passes {
            queue.write_buffer(
                &self.pass_params[pass as usize],
                0,
                bytemuck::cast_slice(&[count, pass, RADIX_BITS, num_wg]),
            );
        }
        queue.write_buffer(
            &self.ps_params_count,
            0,
            bytemuck::cast_slice(&[histogram_count, 0u32, 0u32, 0u32]),
        );
        queue.write_buffer(
            &self.ps_params_wg,
            0,
            bytemuck::cast_slice(&[ps_num_wg, 0u32, 0u32, 0u32]),
        );
        queue.write_buffer(
            &self.ps_params_wg2,
            0,
            bytemuck::cast_slice(&[ps_num_wg2, 0u32, 0u32, 0u32]),
        );

        for pass in 0..passes {
            let p = pass as usize;

            // Clear only the live prefix of the histogram, not the whole
            // (capacity-sized) buffer.
            encoder.clear_buffer(&self.histogram_buf, 0, Some(u64::from(histogram_count) * 4));

            // --- Pass 1: Histogram ---
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("radix_histogram"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.histogram_pipeline);
                cpass.set_bind_group(0, &bgs.histogram[p], &[]);
                cpass.dispatch_workgroups(num_wg, 1, 1);
            }

            // --- Pass 2: Hierarchical inclusive scan of the histogram ---
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("hist_scan_l0"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.prefix_sum_pipeline);
                cpass.set_bind_group(0, &bgs.scan_l0, &[]);
                cpass.dispatch_workgroups(ps_num_wg, 1, 1);
            }
            if ps_num_wg > 1 {
                {
                    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("hist_scan_l1"),
                        timestamp_writes: None,
                    });
                    cpass.set_pipeline(&self.prefix_sum_pipeline);
                    cpass.set_bind_group(0, &bgs.scan_l1, &[]);
                    cpass.dispatch_workgroups(ps_num_wg2, 1, 1);
                }
                if ps_num_wg2 > 1 {
                    {
                        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("hist_scan_l2"),
                            timestamp_writes: None,
                        });
                        cpass.set_pipeline(&self.prefix_sum_pipeline);
                        cpass.set_bind_group(0, &bgs.scan_l2, &[]);
                        cpass.dispatch_workgroups(1, 1, 1);
                    }
                    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("hist_add_l1"),
                        timestamp_writes: None,
                    });
                    cpass.set_pipeline(&self.prefix_sum_add_pipeline);
                    cpass.set_bind_group(0, &bgs.add_l1, &[]);
                    cpass.dispatch_workgroups(ps_num_wg2, 1, 1);
                }
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("hist_add_l0"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.prefix_sum_add_pipeline);
                cpass.set_bind_group(0, &bgs.add_l0, &[]);
                cpass.dispatch_workgroups(ps_num_wg, 1, 1);
            }

            // --- Pass 3: Scatter ---
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("radix_scatter"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.scatter_pipeline);
                cpass.set_bind_group(0, &bgs.scatter[p], &[]);
                cpass.dispatch_workgroups(num_wg, 1, 1);
            }
        }

        Ok(())
    }

    /// Maximum elements this sorter is currently allocated for.
    pub fn capacity(&self) -> u32 {
        self.max_elements
    }
}

/// Allocate the scratch/histogram buffers for a capacity of `max_elements`.
fn allocate_sort_buffers(device: &wgpu::Device, max_elements: u32) -> SortBuffers {
    let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;

    let elements = max_elements.max(1);
    let num_wg = elements.div_ceil(SORT_WG_SIZE).max(1);
    let histogram_count = (NUM_BUCKETS * num_wg).max(NUM_BUCKETS);
    let ps_wg = histogram_count.div_ceil(PREFIX_SUM_BLOCK).max(1);
    let ps_wg2 = ps_wg.div_ceil(PREFIX_SUM_BLOCK).max(1);

    let scratch_keys = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sort_scratch_keys"),
        size: u64::from(elements) * 8,
        usage: storage,
        mapped_at_creation: false,
    });
    let scratch_values = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sort_scratch_values"),
        size: u64::from(elements) * 4,
        usage: storage,
        mapped_at_creation: false,
    });
    let histogram_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sort_histogram"),
        size: u64::from(histogram_count) * 4,
        usage: storage,
        mapped_at_creation: false,
    });
    let histogram_prefix_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sort_histogram_prefix"),
        size: u64::from(histogram_count) * 4,
        usage: storage,
        mapped_at_creation: false,
    });
    let hist_block_sums = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sort_hist_block_sums"),
        size: u64::from(ps_wg) * 4,
        usage: storage,
        mapped_at_creation: false,
    });
    let hist_block_sums_scanned = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sort_hist_block_sums_scanned"),
        size: u64::from(ps_wg) * 4,
        usage: storage,
        mapped_at_creation: false,
    });
    let hist_block_sums_l2 = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sort_hist_block_sums_l2"),
        size: u64::from(ps_wg2) * 4,
        usage: storage,
        mapped_at_creation: false,
    });
    let hist_block_sums_l2_scanned = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sort_hist_block_sums_l2_scanned"),
        size: u64::from(ps_wg2) * 4,
        usage: storage,
        mapped_at_creation: false,
    });

    SortBuffers {
        scratch_keys,
        scratch_values,
        histogram_buf,
        histogram_prefix_buf,
        hist_block_sums,
        hist_block_sums_scanned,
        hist_block_sums_l2,
        hist_block_sums_l2_scanned,
    }
}

/// Poll a future exactly once using a no-op `Waker`, without pulling in an
/// async executor (`pollster` is a dev-dependency only, unavailable to this
/// production code path).
///
/// `wgpu`'s native (`wgpu_core`) backend resolves `ErrorScopeGuard::pop()`
/// synchronously — the error is already recorded in the device's error sink
/// by the time `create_shader_module` / `create_compute_pipeline` returns,
/// and `pop()` just wraps that already-known value in `std::future::ready`
/// for API parity with the (genuinely async) WebGPU backend — so a single
/// poll is sufficient here. Returns `None` if the future is not yet ready
/// (e.g. under a genuinely async backend), which [`compile_pipeline`] treats
/// the same as "no error captured".
///
/// This mirrors the identical helper in `pipeline.rs`; see the followup note
/// on shader-compile error-scope duplication across those two modules for
/// the plan to consolidate them into one shared helper module.
fn poll_ready_now<F: std::future::Future>(fut: F) -> Option<F::Output> {
    use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

    fn no_op(_: *const ()) {}
    fn clone_raw(_: *const ()) -> RawWaker {
        RawWaker::new(std::ptr::null(), &VTABLE)
    }
    static VTABLE: RawWakerVTable = RawWakerVTable::new(clone_raw, no_op, no_op, no_op);

    // Safety: every VTABLE function is a no-op that never dereferences the
    // data pointer, so a null data pointer is sound here.
    let raw_waker = RawWaker::new(std::ptr::null(), &VTABLE);
    let waker = unsafe { Waker::from_raw(raw_waker) };
    let mut cx = Context::from_waker(&waker);
    let mut boxed = Box::pin(fut);
    match boxed.as_mut().poll(&mut cx) {
        Poll::Ready(v) => Some(v),
        Poll::Pending => None,
    }
}

/// Compile a WGSL shader module and its compute pipeline, capturing any
/// `wgpu` validation error via an error scope instead of letting it reach
/// the uncaptured-error handler (a panic by default).
fn compile_pipeline(
    device: &wgpu::Device,
    label: &str,
    source: &str,
    entry_point: &str,
    bgl: &wgpu::BindGroupLayout,
) -> Result<wgpu::ComputePipeline, RenderError> {
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    if let Some(err) = poll_ready_now(scope.pop()).flatten() {
        return Err(RenderError::ShaderCompilation {
            shader_name: label.to_string(),
            error: err.to_string(),
        });
    }

    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(&format!("{label}_layout")),
        bind_group_layouts: &[Some(bgl)],
        immediate_size: 0,
    });

    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(&format!("{label}_pipeline")),
        layout: Some(&layout),
        module: &module,
        entry_point: Some(entry_point),
        compilation_options: Default::default(),
        cache: None,
    });
    if let Some(err) = poll_ready_now(scope.pop()).flatten() {
        return Err(RenderError::ShaderValidation {
            shader_name: label.to_string(),
            error: err.to_string(),
        });
    }

    Ok(pipeline)
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

/// Helper for a whole-buffer bind group entry.
fn bg_entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pass_count_skips_all_zero_tile_digits() {
        // 1920×1080 with 16px tiles => 120 × 68 = 8160 tiles => 13 tile bits.
        // 32 depth bits + 13 tile bits = 45 => ceil(45 / 4) = 12 passes.
        assert_eq!(radix_pass_count(120 * 68), 12);
        // ...which is strictly fewer than sorting the full 64-bit key.
        assert!(radix_pass_count(120 * 68) < NUM_PASSES);
    }

    #[test]
    fn test_pass_count_is_always_even() {
        // The ping-pong leaves the result in the caller's buffers only when the
        // pass count is even, so every input must produce an even result.
        for num_tiles in [0u32, 1, 2, 3, 17, 255, 256, 4096, 8160, 65_535, u32::MAX] {
            assert_eq!(
                radix_pass_count(num_tiles) % 2,
                0,
                "odd pass count for num_tiles={num_tiles}"
            );
        }
    }

    #[test]
    fn test_pass_count_single_tile_sorts_depth_only() {
        // A single tile needs no tile bits at all: 32 depth bits => 8 passes.
        assert_eq!(radix_pass_count(1), 8);
        assert_eq!(radix_pass_count(0), 8);
    }

    #[test]
    fn test_pass_count_never_exceeds_full_key() {
        assert_eq!(radix_pass_count(u32::MAX), NUM_PASSES);
        for num_tiles in [1u32, 100, 10_000, 1_000_000, u32::MAX] {
            assert!(radix_pass_count(num_tiles) <= NUM_PASSES);
        }
    }

    #[test]
    fn test_pass_count_covers_the_highest_tile_bit() {
        // Every set bit of the largest tile id must be covered by some pass.
        for num_tiles in [2u32, 3, 16, 17, 4096, 8160, 70_000] {
            let passes = radix_pass_count(num_tiles);
            let covered_bits = passes * RADIX_BITS;
            let needed = 32 + (u32::BITS - (num_tiles - 1).leading_zeros());
            assert!(
                covered_bits >= needed,
                "num_tiles={num_tiles}: {covered_bits} covered bits < {needed} needed"
            );
        }
    }

    /// A shader compile/validation failure must surface as a `RenderError`,
    /// not reach `wgpu`'s uncaptured-error handler (a panic by default).
    /// Mirrors the identical regression test in `pipeline.rs`.
    #[test]
    #[ignore = "requires GPU"]
    fn test_compile_pipeline_invalid_wgsl_returns_error_not_panic() {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter =
            match pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
                apply_limit_buckets: false,
            })) {
                Ok(a) => a,
                Err(_) => {
                    eprintln!("No GPU adapter available, skipping GPU test");
                    return;
                }
            };
        let (device, _queue) =
            match pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("sort_test_device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                experimental_features: wgpu::ExperimentalFeatures::default(),
                trace: wgpu::Trace::Off,
            })) {
                Ok(dq) => dq,
                Err(_) => {
                    eprintln!("Failed to create GPU device, skipping GPU test");
                    return;
                }
            };

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("test_bgl"),
            entries: &[uniform_bgl_entry(0)],
        });

        // Deliberately malformed WGSL source.
        let result = compile_pipeline(
            &device,
            "broken_shader",
            "this is not valid wgsl {{{",
            "main",
            &bgl,
        );

        assert!(
            matches!(
                result,
                Err(RenderError::ShaderCompilation { .. })
                    | Err(RenderError::ShaderValidation { .. })
            ),
            "expected a ShaderCompilation or ShaderValidation error, got {result:?}"
        );
    }
}
