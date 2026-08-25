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

use std::borrow::Cow;

use crate::config::RasterConfig;
use crate::RenderError;

/// The `@workgroup_size` attribute every 1-D kernel in `shaders/` ships with.
///
/// [`retarget_linear_workgroup_size`] rewrites exactly this literal, so a
/// shader that spells its attribute differently is left alone rather than
/// silently half-retargeted.
const SHIPPED_LINEAR_ATTRIBUTE: &str = "@workgroup_size(256)";

/// Rewrite a 1-D kernel's `@workgroup_size` attribute to `size`.
///
/// # Which shaders may go through this
///
/// Only kernels whose `@workgroup_size` is *pure dispatch geometry* — they
/// index work with `global_invocation_id` alone and never use
/// `local_invocation_id`, `workgroup_id`, `var<workgroup>` or a hardcoded
/// thread count. In `shaders/` that is the four `preprocess_sh*` variants,
/// `preprocess`, `preprocess_bwd`, `tile_assign`, `tile_ranges` and
/// `atomic_to_f32`.
///
/// It must **not** be applied to:
///
/// * `prefix_sum.wgsl` — `var<workgroup> temp: array<u32, 512>` and a literal
///   256-thread stride in its scan loop;
/// * `prefix_sum_add.wgsl` — `wid.x * 512u` and `lid.x + 256u` in its body,
///   both tied to `prefix_sum`'s 512-element block, which is *not* retargeted;
/// * `radix_histogram.wgsl` / `radix_scatter.wgsl` — workgroup-sized shared
///   histograms (`array<u32, 256>`);
/// * `rasterize_fwd.wgsl` / `rasterize_bwd.wgsl` / `cov2d_bwd.wgsl` — their
///   `@workgroup_size(16, 16)` *is* the tile size, not a tuning knob.
///
/// For each of those, changing the attribute alone would keep the shader
/// compiling and produce wrong results.
///
/// Returns the source unchanged (borrowed) when `size` already matches the
/// shipped attribute, and logs a warning without substituting if the expected
/// attribute is absent.
fn retarget_linear_workgroup_size<'a>(label: &str, source: &'a str, size: u32) -> Cow<'a, str> {
    if size == crate::workgroup::WorkgroupConfig::SHIPPED_LINEAR.x {
        return Cow::Borrowed(source);
    }
    if !source.contains(SHIPPED_LINEAR_ATTRIBUTE) {
        tracing::warn!(
            shader = label,
            expected = SHIPPED_LINEAR_ATTRIBUTE,
            "shader does not declare the expected workgroup attribute; compiling it unchanged"
        );
        return Cow::Borrowed(source);
    }
    Cow::Owned(source.replace(
        SHIPPED_LINEAR_ATTRIBUTE,
        &format!("@workgroup_size({size})"),
    ))
}

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

    /// The `@workgroup_size` the retargetable 1-D kernels were compiled with.
    ///
    /// [`Rasterizer`](crate::Rasterizer) derives their dispatch counts from
    /// this, so the host grid and the compiled attribute cannot drift apart.
    /// It is `config.effective_preprocess_wg_size()` at compile time.
    pub linear_workgroup_size: u32,
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
    ///
    /// # Errors
    ///
    /// Each shader/pipeline is compiled under a `wgpu` validation error
    /// scope: a WGSL parse/type error surfaces as
    /// [`RenderError::ShaderCompilation`], and a pipeline-layout
    /// compatibility error surfaces as [`RenderError::ShaderValidation`],
    /// instead of reaching `wgpu`'s uncaptured-error handler (a panic by
    /// default).
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

        // Workgroup size for every kernel whose @workgroup_size is pure
        // dispatch geometry. `RasterConfig::effective_preprocess_wg_size`
        // resolves the explicit override first and the GPU preset second, so
        // a config that asks for AMD's 64-thread wavefronts really gets them
        // instead of being silently ignored.
        let linear_workgroup_size = config.effective_preprocess_wg_size();
        let retarget = |label: &str, src: &'static str| {
            retarget_linear_workgroup_size(label, src, linear_workgroup_size)
        };

        let preprocess = compile_pipeline(
            device,
            preprocess_label,
            &retarget(preprocess_label, preprocess_source),
            "preprocess",
            &preprocess_bgl,
        )?;

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
        )?;

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
        )?;

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
            &retarget("tile_assign", include_str!("../shaders/tile_assign.wgsl")),
            "tile_assign",
            &tile_assign_bgl,
        )?;

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
            &retarget("tile_ranges", include_str!("../shaders/tile_ranges.wgsl")),
            "tile_ranges_kernel",
            &tile_ranges_bgl,
        )?;

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
        )?;

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
        )?;

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
            &retarget(
                "atomic_to_f32",
                include_str!("../shaders/atomic_to_f32.wgsl"),
            ),
            "atomic_to_f32",
            &atomic_to_f32_bgl,
        )?;

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
            &retarget(
                "preprocess_bwd",
                include_str!("../shaders/preprocess_bwd.wgsl"),
            ),
            "preprocess_backward",
            &preprocess_bwd_bgl,
        )?;

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
        )?;

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
            linear_workgroup_size,
        })
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
/// This mirrors the hand-rolled `noop_waker` `wgpu` itself vendors
/// internally for the same reason (`Waker::noop()` needs Rust 1.85+).
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

    // --- Workgroup retargeting (F307) ---

    /// Every shader the rasterizer retargets must actually carry the literal
    /// attribute the substitution looks for; otherwise the retarget silently
    /// does nothing and the dispatch grid (derived from the *requested* size)
    /// no longer matches the compiled kernel.
    #[test]
    fn test_retargetable_shaders_declare_the_expected_attribute() {
        let retargetable: [(&str, &str); 9] = [
            ("preprocess", include_str!("../shaders/preprocess.wgsl")),
            (
                "preprocess_sh0",
                include_str!("../shaders/preprocess_sh0.wgsl"),
            ),
            (
                "preprocess_sh1",
                include_str!("../shaders/preprocess_sh1.wgsl"),
            ),
            (
                "preprocess_sh2",
                include_str!("../shaders/preprocess_sh2.wgsl"),
            ),
            (
                "preprocess_sh3",
                include_str!("../shaders/preprocess_sh3.wgsl"),
            ),
            (
                "preprocess_bwd",
                include_str!("../shaders/preprocess_bwd.wgsl"),
            ),
            ("tile_assign", include_str!("../shaders/tile_assign.wgsl")),
            ("tile_ranges", include_str!("../shaders/tile_ranges.wgsl")),
            (
                "atomic_to_f32",
                include_str!("../shaders/atomic_to_f32.wgsl"),
            ),
        ];
        for (label, src) in retargetable {
            assert_eq!(
                src.matches(SHIPPED_LINEAR_ATTRIBUTE).count(),
                1,
                "{label} must declare exactly one {SHIPPED_LINEAR_ATTRIBUTE}"
            );
            let out = retarget_linear_workgroup_size(label, src, 64);
            assert!(out.contains("@workgroup_size(64)"), "{label}");
            assert!(!out.contains(SHIPPED_LINEAR_ATTRIBUTE), "{label}");
        }
    }

    /// A retargeted kernel must index work with `global_invocation_id` alone.
    /// The moment one uses `local_invocation_id`, `workgroup_id` or shared
    /// memory, its thread count stops being pure dispatch geometry and
    /// rewriting the attribute corrupts its output instead of retuning it.
    #[test]
    fn test_retargetable_shaders_have_no_workgroup_local_state() {
        let retargetable = [
            include_str!("../shaders/preprocess.wgsl"),
            include_str!("../shaders/preprocess_sh0.wgsl"),
            include_str!("../shaders/preprocess_sh1.wgsl"),
            include_str!("../shaders/preprocess_sh2.wgsl"),
            include_str!("../shaders/preprocess_sh3.wgsl"),
            include_str!("../shaders/preprocess_bwd.wgsl"),
            include_str!("../shaders/tile_assign.wgsl"),
            include_str!("../shaders/tile_ranges.wgsl"),
            include_str!("../shaders/atomic_to_f32.wgsl"),
        ];
        for src in retargetable {
            // Ignore comments: only the code matters here.
            let code: String = src
                .lines()
                .filter(|l| !l.trim_start().starts_with("//"))
                .collect::<Vec<_>>()
                .join("\n");
            for forbidden in [
                "local_invocation_id",
                "workgroup_id",
                "var<workgroup>",
                "workgroupBarrier",
            ] {
                assert!(
                    !code.contains(forbidden),
                    "a retargeted kernel must not use `{forbidden}`"
                );
            }
        }
    }

    /// The scan and sort kernels bake their thread count into shared-memory
    /// sizes and literal strides, so they must stay out of the retarget list.
    /// This pins the reason: they really do contain that state.
    #[test]
    fn test_non_retargetable_shaders_bake_in_their_thread_count() {
        assert!(include_str!("../shaders/prefix_sum.wgsl").contains("var<workgroup>"));
        // prefix_sum_add walks prefix_sum's fixed 512-element blocks by hand.
        let add = include_str!("../shaders/prefix_sum_add.wgsl");
        assert!(
            add.contains("512u"),
            "prefix_sum_add hardcodes the block size"
        );
        assert!(
            add.contains("256u"),
            "prefix_sum_add hardcodes the half-block"
        );
        assert!(include_str!("../shaders/radix_scatter.wgsl").contains("var<workgroup>"));
        assert!(include_str!("../shaders/radix_histogram.wgsl").contains("var<workgroup>"));
    }

    /// The default configuration must produce byte-identical shader sources,
    /// so enabling the retarget path cannot perturb the shipped behaviour.
    #[test]
    fn test_retarget_is_a_no_op_at_the_shipped_size() {
        let src = include_str!("../shaders/tile_assign.wgsl");
        let out = retarget_linear_workgroup_size("tile_assign", src, 256);
        assert!(matches!(out, Cow::Borrowed(_)));
        assert_eq!(out, src);
        // ...and that is what the default config asks for.
        assert_eq!(RasterConfig::default().effective_preprocess_wg_size(), 256);
    }

    /// A source that does not carry the expected attribute is compiled
    /// unchanged rather than half-rewritten.
    #[test]
    fn test_retarget_leaves_unknown_sources_alone() {
        let src = "@compute @workgroup_size(16, 16)\nfn main() {}\n";
        let out = retarget_linear_workgroup_size("tile_kernel", src, 64);
        assert_eq!(out, src);
    }

    /// A shader compile/validation failure must surface as a `RenderError`,
    /// not reach `wgpu`'s uncaptured-error handler (a panic by default).
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
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("pipeline_test_device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                experimental_features: wgpu::ExperimentalFeatures::default(),
                trace: wgpu::Trace::Off,
            }))
            .expect("failed to create device");

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("test_bgl"),
            entries: &[uniform_entry(0)],
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
