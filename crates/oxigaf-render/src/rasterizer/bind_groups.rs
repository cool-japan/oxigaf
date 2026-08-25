//! Bind-group construction for the forward and backward passes.
//!
//! Every bind group the rasterizer uses is built exactly once per model
//! upload — the only point at which the GPU buffers they reference can
//! change — and then reused frame after frame, with only the *contents* of
//! the uniform buffers rewritten. Keeping the two builders here keeps the
//! per-stage binding tables (which must match `pipeline.rs`'s bind group
//! layouts entry for entry) in one place.

use crate::buffers::{GaussianBuffers, GradientBuffers, IntermediateBuffers, OutputBuffers};

use super::{BackwardBindGroups, FrameBindGroups, Rasterizer};

impl Rasterizer {
    pub(super) fn build_frame_bind_groups(
        &self,
        gauss: &GaussianBuffers,
        inter: &IntermediateBuffers,
        output: &OutputBuffers,
    ) -> FrameBindGroups {
        let out_normals_buf = output.normals.as_ref().unwrap_or(&self.dummy_out_normals);

        let preprocess = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
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

        let scan_l0 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("prefix_sum_l0_bg"),
            layout: &self.pipelines.prefix_sum_bgl,
            entries: &[
                entry(0, &inter.tile_counts),
                entry(1, &inter.tile_offsets),
                entry(2, &self.params.scan_count),
                entry(3, &inter.block_sums),
            ],
        });
        let scan_l1 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("prefix_sum_l1_bg"),
            layout: &self.pipelines.prefix_sum_bgl,
            entries: &[
                entry(0, &inter.block_sums),
                entry(1, &inter.block_sums_scanned),
                entry(2, &self.params.scan_l1),
                entry(3, &inter.block_sums_l2),
            ],
        });
        let scan_l2 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("prefix_sum_l2_bg"),
            layout: &self.pipelines.prefix_sum_bgl,
            entries: &[
                entry(0, &inter.block_sums_l2),
                entry(1, &inter.block_sums_l2_scanned),
                entry(2, &self.params.scan_l2),
                entry(3, &self.dummy_block_sums),
            ],
        });
        let add_l1 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("prefix_sum_add_l1_bg"),
            layout: &self.pipelines.prefix_sum_add_bgl,
            entries: &[
                entry(0, &inter.block_sums_scanned),
                entry(1, &inter.block_sums_l2_scanned),
                entry(2, &self.params.scan_l1),
            ],
        });
        let add_l0 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("prefix_sum_add_l0_bg"),
            layout: &self.pipelines.prefix_sum_add_bgl,
            entries: &[
                entry(0, &inter.tile_offsets),
                entry(1, &inter.block_sums_scanned),
                entry(2, &self.params.scan_count),
            ],
        });

        let tile_assign = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
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
        let tile_ranges = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tile_ranges_bg"),
            layout: &self.pipelines.tile_ranges_bgl,
            entries: &[
                entry(0, &inter.sort_keys),
                entry(1, &inter.tile_ranges),
                entry(2, &self.params.tile_ranges),
            ],
        });
        let rasterize_fwd = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
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

        FrameBindGroups {
            preprocess,
            scan_l0,
            scan_l1,
            scan_l2,
            add_l1,
            add_l0,
            tile_assign,
            tile_ranges,
            rasterize_fwd,
        }
    }

    pub(super) fn build_backward_bind_groups(
        &self,
        gauss: &GaussianBuffers,
        inter: &IntermediateBuffers,
        output: &OutputBuffers,
        grads: &GradientBuffers,
    ) -> BackwardBindGroups {
        let rasterize_bwd = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
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
                entry(10, &self.grad_output_buf),
                entry(11, &grads.grad_colors_atomic),
                entry(12, &grads.grad_opacities),
                entry(13, &grads.grad_means2d_atomic),
                entry(14, &grads.grad_conics_atomic),
            ],
        });
        let atomic_means2d = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("atomic_to_f32_means2d_bg"),
            layout: &self.pipelines.atomic_to_f32_bgl,
            entries: &[
                entry(0, &self.params.num_elements_means2d),
                entry(1, &grads.grad_means2d_atomic),
                entry(2, &grads.grad_means2d),
            ],
        });
        let atomic_conics = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("atomic_to_f32_conics_bg"),
            layout: &self.pipelines.atomic_to_f32_bgl,
            entries: &[
                entry(0, &self.params.num_elements_conics),
                entry(1, &grads.grad_conics_atomic),
                entry(2, &grads.grad_conics),
            ],
        });
        let atomic_colors = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("atomic_to_f32_colors_bg"),
            layout: &self.pipelines.atomic_to_f32_bgl,
            entries: &[
                entry(0, &self.params.num_elements_colors),
                entry(1, &grads.grad_colors_atomic),
                entry(2, &grads.grad_colors),
            ],
        });
        let preprocess_bwd = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
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

        BackwardBindGroups {
            rasterize_bwd,
            atomic_means2d,
            atomic_conics,
            atomic_colors,
            preprocess_bwd,
        }
    }
}

fn entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}
