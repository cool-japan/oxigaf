//! GPU compute pipeline setup and shader compilation.
//!
//! This module handles the creation of compute pipelines for 3DGS rasterization.
//! It supports shader variant selection based on SH degree for optimal performance.
//!
//! ## SH Shader Variants
//!
//! When `use_sh_variants` is enabled in the config, specialized shaders are selected
//! based on the configured `sh_degree`:
//! - Degree 0: DC term only, no direction computation (~3 ops)
//! - Degree 1: DC + linear terms (~15 ops)
//! - Degree 2: DC + linear + quadratic terms (~45 ops)
//! - Degree 3: Full SH evaluation (~100 ops)
//!
//! These variants eliminate runtime branching, improving performance by 5-10%.

use crate::config::RasterConfig;
use crate::RenderError;

/// Compiled compute pipelines for the rasterization forward and backward passes.
pub struct RasterPipelines {
    pub preprocess: wgpu::ComputePipeline,
    pub prefix_sum: wgpu::ComputePipeline,
    pub prefix_sum_add: wgpu::ComputePipeline,
    pub tile_assign: wgpu::ComputePipeline,
    pub tile_ranges: wgpu::ComputePipeline,
    pub rasterize_fwd: wgpu::ComputePipeline,
    pub rasterize_bwd: wgpu::ComputePipeline,
    pub atomic_to_f32: wgpu::ComputePipeline,
    pub preprocess_bwd: wgpu::ComputePipeline,
    pub flame_binding_bwd: wgpu::ComputePipeline,

    // Bind group layouts for each stage
    pub preprocess_bgl: wgpu::BindGroupLayout,
    pub prefix_sum_bgl: wgpu::BindGroupLayout,
    pub prefix_sum_add_bgl: wgpu::BindGroupLayout,
    pub tile_assign_bgl: wgpu::BindGroupLayout,
    pub tile_ranges_bgl: wgpu::BindGroupLayout,
    pub rasterize_fwd_bgl: wgpu::BindGroupLayout,
    pub rasterize_bwd_bgl: wgpu::BindGroupLayout,
    pub atomic_to_f32_bgl: wgpu::BindGroupLayout,
    pub preprocess_bwd_bgl: wgpu::BindGroupLayout,
    pub flame_binding_bwd_bgl: wgpu::BindGroupLayout,
}

/// Get the preprocess shader source based on SH degree and optimization settings.
///
/// When `use_sh_variants` is true and `sh_optimization` is enabled, returns
/// a specialized shader variant for the given `sh_degree`. Otherwise, returns
/// the general-purpose shader with runtime branching.
///
/// # Performance
///
/// Using specialized variants eliminates runtime branching overhead:
/// - Degree 0: Skips direction computation entirely
/// - Degree 1-3: Unrolled SH evaluation without conditionals
fn get_preprocess_shader_source(config: &RasterConfig) -> &'static str {
    if config.sh_optimization && config.use_sh_variants {
        match config.sh_degree {
            0 => include_str!("../shaders/preprocess_sh0.wgsl"),
            1 => include_str!("../shaders/preprocess_sh1.wgsl"),
            2 => include_str!("../shaders/preprocess_sh2.wgsl"),
            3 => include_str!("../shaders/preprocess_sh3.wgsl"),
            // For degrees > 3, fall back to general shader
            _ => include_str!("../shaders/preprocess.wgsl"),
        }
    } else {
        // Use general-purpose shader with runtime branching
        include_str!("../shaders/preprocess.wgsl")
    }
}

/// Get a descriptive label for the preprocess shader based on configuration.
fn get_preprocess_shader_label(config: &RasterConfig) -> &'static str {
    if config.sh_optimization && config.use_sh_variants {
        match config.sh_degree {
            0 => "preprocess_sh0",
            1 => "preprocess_sh1",
            2 => "preprocess_sh2",
            3 => "preprocess_sh3",
            _ => "preprocess",
        }
    } else {
        "preprocess"
    }
}

impl RasterPipelines {
    /// Compile all compute pipelines.
    ///
    /// When `config.use_sh_variants` is true, selects specialized preprocess
    /// shaders based on `config.sh_degree` for optimal SH evaluation performance.
    pub fn new(device: &wgpu::Device, config: &RasterConfig) -> Result<Self, RenderError> {
        // --- Preprocess ---
        // Bind group layout includes normals buffer for optional normal output
        let preprocess_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("preprocess_bgl"),
            entries: &[
                uniform_entry(0),     // uniforms
                storage_ro_entry(1),  // positions
                storage_ro_entry(2),  // rotations
                storage_ro_entry(3),  // scales
                storage_ro_entry(4),  // opacities
                storage_rw_entry(5),  // means2d
                storage_rw_entry(6),  // cov2d
                storage_rw_entry(7),  // conics
                storage_rw_entry(8),  // depths
                storage_rw_entry(9),  // radii
                storage_rw_entry(10), // tile_counts
                storage_ro_entry(11), // sh_coeffs
                storage_rw_entry(12), // colors
                storage_rw_entry(13), // normals (for optional normal output)
            ],
        });

        // Select shader variant based on config
        let preprocess_source = get_preprocess_shader_source(config);
        let preprocess_label = get_preprocess_shader_label(config);

        let preprocess = compile_pipeline(
            device,
            preprocess_label,
            preprocess_source,
            "preprocess",
            &preprocess_bgl,
        );

        // --- Prefix sum ---
        let prefix_sum_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("prefix_sum_bgl"),
            entries: &[
                storage_ro_entry(0), // input
                storage_rw_entry(1), // output
                uniform_entry(2),    // params
                storage_rw_entry(3), // block_sums
            ],
        });
        let prefix_sum = compile_pipeline(
            device,
            "prefix_sum",
            include_str!("../shaders/prefix_sum.wgsl"),
            "prefix_sum",
            &prefix_sum_bgl,
        );

        // --- Prefix sum add (propagate block offsets) ---
        let prefix_sum_add_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("prefix_sum_add_bgl"),
                entries: &[
                    storage_rw_entry(0), // data (in-place add)
                    storage_ro_entry(1), // block_offsets
                    uniform_entry(2),    // params
                ],
            });
        let prefix_sum_add = compile_pipeline(
            device,
            "prefix_sum_add",
            include_str!("../shaders/prefix_sum_add.wgsl"),
            "prefix_sum_add",
            &prefix_sum_add_bgl,
        );

        // --- Tile assign ---
        let tile_assign_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tile_assign_bgl"),
            entries: &[
                uniform_entry(0),    // uniforms
                storage_ro_entry(1), // means2d
                storage_ro_entry(2), // depths
                storage_ro_entry(3), // radii
                storage_ro_entry(4), // tile_offsets
                storage_rw_entry(5), // sort_keys
                storage_rw_entry(6), // sort_values
            ],
        });
        let tile_assign = compile_pipeline(
            device,
            "tile_assign",
            include_str!("../shaders/tile_assign.wgsl"),
            "tile_assign",
            &tile_assign_bgl,
        );

        // --- Tile ranges ---
        let tile_ranges_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tile_ranges_bgl"),
            entries: &[
                storage_ro_entry(0), // sort_keys
                storage_rw_entry(1), // tile_ranges
                uniform_entry(2),    // params
            ],
        });
        let tile_ranges = compile_pipeline(
            device,
            "tile_ranges",
            include_str!("../shaders/tile_ranges.wgsl"),
            "tile_ranges_kernel",
            &tile_ranges_bgl,
        );

        // --- Rasterize forward ---
        let rasterize_fwd_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rasterize_fwd_bgl"),
            entries: &[
                uniform_entry(0),     // uniforms
                storage_ro_entry(1),  // means2d
                storage_ro_entry(2),  // conics
                storage_ro_entry(3),  // colors
                storage_ro_entry(4),  // opacities
                storage_ro_entry(5),  // depths
                storage_ro_entry(6),  // tile_ranges
                storage_ro_entry(7),  // sort_values
                storage_rw_entry(8),  // out_color
                storage_rw_entry(9),  // out_depth
                storage_rw_entry(10), // out_transmittance
                storage_rw_entry(11), // out_n_contrib
                storage_ro_entry(12), // normals (per-Gaussian normals from preprocess)
                storage_rw_entry(13), // out_normals (per-pixel normals output)
            ],
        });
        let rasterize_fwd = compile_pipeline(
            device,
            "rasterize_fwd",
            include_str!("../shaders/rasterize_fwd.wgsl"),
            "rasterize_forward",
            &rasterize_fwd_bgl,
        );

        // --- Rasterize backward ---
        let rasterize_bwd_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rasterize_bwd_bgl"),
            entries: &[
                uniform_entry(0),     // uniforms
                storage_ro_entry(1),  // means2d
                storage_ro_entry(2),  // conics
                storage_ro_entry(3),  // colors
                storage_ro_entry(4),  // opacities
                storage_ro_entry(5),  // out_color
                storage_ro_entry(6),  // out_transmittance
                storage_ro_entry(7),  // out_n_contrib
                storage_ro_entry(8),  // tile_ranges
                storage_ro_entry(9),  // sort_values
                storage_ro_entry(10), // grad_output
                storage_rw_entry(11), // grad_colors
                storage_rw_entry(12), // grad_opacities
                storage_rw_entry(13), // grad_means2d
                storage_rw_entry(14), // grad_conics
            ],
        });
        let rasterize_bwd = compile_pipeline(
            device,
            "rasterize_bwd",
            include_str!("../shaders/rasterize_bwd.wgsl"),
            "rasterize_backward",
            &rasterize_bwd_bgl,
        );

        // --- Atomic to f32 conversion ---
        let atomic_to_f32_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("atomic_to_f32_bgl"),
            entries: &[
                uniform_entry(0),    // num_elements
                storage_rw_entry(1), // atomic_buffer (array<atomic<u32>>)
                storage_rw_entry(2), // f32_buffer (array<f32>)
            ],
        });
        let atomic_to_f32 = compile_pipeline(
            device,
            "atomic_to_f32",
            include_str!("../shaders/atomic_to_f32.wgsl"),
            "atomic_to_f32",
            &atomic_to_f32_bgl,
        );

        // --- Preprocess backward ---
        let preprocess_bwd_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("preprocess_bwd_bgl"),
                entries: &[
                    uniform_entry(0),     // uniforms
                    storage_ro_entry(1),  // positions
                    storage_ro_entry(2),  // rotations
                    storage_ro_entry(3),  // scales
                    storage_ro_entry(4),  // cov2d
                    storage_ro_entry(5),  // conics
                    storage_ro_entry(6),  // sh_coeffs
                    storage_ro_entry(7),  // grad_means2d (from rasterize_bwd)
                    storage_ro_entry(8),  // grad_conics (from rasterize_bwd)
                    storage_ro_entry(9),  // grad_colors (from rasterize_bwd)
                    storage_rw_entry(10), // grad_positions
                    storage_rw_entry(11), // grad_rotations
                    storage_rw_entry(12), // grad_scales
                    storage_rw_entry(13), // grad_sh_coeffs
                ],
            });
        let preprocess_bwd = compile_pipeline(
            device,
            "preprocess_bwd",
            include_str!("../shaders/preprocess_bwd.wgsl"),
            "preprocess_backward",
            &preprocess_bwd_bgl,
        );

        // FLAME binding backward pass
        let flame_binding_bwd_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("flame_binding_bwd_bgl"),
                entries: &[
                    uniform_entry(0),    // uniforms (num_gaussians)
                    storage_ro_entry(1), // binding_info (vertex_id per Gaussian)
                    storage_ro_entry(2), // tbn_frames (tangent, bitangent, normal per vertex)
                    storage_ro_entry(3), // position_grads (∂L/∂position per Gaussian)
                    storage_rw_entry(4), // offset_grads (∂L/∂local_offset output)
                ],
            });
        let flame_binding_bwd = compile_pipeline(
            device,
            "flame_binding_bwd",
            include_str!("../shaders/flame_binding_bwd.wgsl"),
            "flame_binding_backward",
            &flame_binding_bwd_bgl,
        );

        Ok(Self {
            preprocess,
            prefix_sum,
            prefix_sum_add,
            tile_assign,
            tile_ranges,
            rasterize_fwd,
            rasterize_bwd,
            atomic_to_f32,
            preprocess_bwd,
            flame_binding_bwd,
            preprocess_bgl,
            prefix_sum_bgl,
            prefix_sum_add_bgl,
            tile_assign_bgl,
            tile_ranges_bgl,
            rasterize_fwd_bgl,
            rasterize_bwd_bgl,
            atomic_to_f32_bgl,
            preprocess_bwd_bgl,
            flame_binding_bwd_bgl,
        })
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
        bind_group_layouts: &[Some(bgl)],
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

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
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

fn storage_ro_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn storage_rw_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_preprocess_shader_label() {
        // With variants enabled
        let mut config = RasterConfig {
            sh_optimization: true,
            use_sh_variants: true,
            ..RasterConfig::default()
        };

        config.sh_degree = 0;
        assert_eq!(get_preprocess_shader_label(&config), "preprocess_sh0");

        config.sh_degree = 1;
        assert_eq!(get_preprocess_shader_label(&config), "preprocess_sh1");

        config.sh_degree = 2;
        assert_eq!(get_preprocess_shader_label(&config), "preprocess_sh2");

        config.sh_degree = 3;
        assert_eq!(get_preprocess_shader_label(&config), "preprocess_sh3");

        // Without variants
        config.use_sh_variants = false;
        config.sh_degree = 0;
        assert_eq!(get_preprocess_shader_label(&config), "preprocess");
    }
}
