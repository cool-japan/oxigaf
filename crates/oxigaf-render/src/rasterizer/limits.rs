//! Device limits and configuration constants the GPU rasterizer requires.
//!
//! Two of `wgpu`'s default limits are simply too small for the shipped
//! shaders — the backward pass binds far more storage buffers than the
//! default eight, and `rasterize_fwd.wgsl`'s shared-memory batch is larger
//! than the default workgroup-storage budget. Both are raised by
//! [`rasterizer_device_limits`], and [`Rasterizer::from_device`] refuses a
//! device that was not built that way rather than letting the shortfall
//! surface as an opaque pipeline-validation failure much later.
//!
//! [`Rasterizer::from_device`]: super::Rasterizer::from_device

use crate::RenderError;

/// Tile size the rasterization shaders are compiled for.
///
/// `rasterize_fwd.wgsl` and `rasterize_bwd.wgsl` declare
/// `@workgroup_size(16, 16)`, which WGSL fixes at shader-compile time, and one
/// workgroup covers exactly one tile. A different
/// [`RasterConfig::tile_size`](crate::config::RasterConfig::tile_size) would
/// therefore desync the tile grid from the workgroup grid, so it is rejected
/// in [`Rasterizer::from_device`](super::Rasterizer::from_device).
pub const RASTERIZE_TILE_SIZE: u32 = 16;

/// Storage buffers a single rasterizer shader stage binds.
///
/// The backward pass (`rasterize_bwd.wgsl`) binds 14 storage buffers plus a
/// uniform in one bind group, against `wgpu::Limits::default()`'s ceiling of
/// eight — a device requested with plain defaults therefore fails on the
/// *first backward pass*, long after the run looks like it started.
pub const RASTERIZER_STORAGE_BUFFERS_PER_STAGE: u32 = 16;

/// Bytes of `var<workgroup>` storage `rasterize_fwd.wgsl` declares.
///
/// The forward tile kernel stages a batch of 256 Gaussians in shared memory
/// (`wg_gidx` u32, `wg_mean` vec2, `wg_conic` vec4, `wg_color` vec4,
/// `wg_opacity` f32, `wg_depth` f32, `wg_normal` vec4), i.e.
/// `256 * (4 + 8 + 16 + 16 + 4 + 4 + 16) = 17_408` bytes.
///
/// That is **above** `wgpu::Limits::default()`'s
/// `max_compute_workgroup_storage_size` of 16_384, so a device created with
/// default limits cannot create the forward pipeline at all.
/// [`rasterizer_device_limits`] raises the limit and
/// [`Rasterizer::from_device`](super::Rasterizer::from_device) rejects a
/// device that was not built that way, instead of letting it fail as an
/// opaque pipeline-validation error.
/// `tests::test_rasterize_fwd_workgroup_storage_matches_shader` recomputes
/// this figure from the shader source, so adding another `var<workgroup>`
/// array without raising the limit fails the test rather than the GPU.
pub const RASTERIZE_FWD_WORKGROUP_STORAGE_BYTES: u32 = 17_408;

/// Device limits the GPU rasterizer needs on top of `base`.
///
/// Raises `max_storage_buffers_per_shader_stage` to
/// [`RASTERIZER_STORAGE_BUFFERS_PER_STAGE`] and
/// `max_compute_workgroup_storage_size` to
/// [`RASTERIZE_FWD_WORKGROUP_STORAGE_BYTES`], leaving every other limit in
/// `base` untouched. Both are raised with `max`, so a `base` that already
/// asks for more (e.g. `adapter.limits()`) is never lowered.
///
/// Pass the result as `DeviceDescriptor::required_limits` when building a
/// device that will be handed to
/// [`Rasterizer::from_device`](super::Rasterizer::from_device).
#[must_use]
pub fn rasterizer_device_limits(base: wgpu::Limits) -> wgpu::Limits {
    wgpu::Limits {
        max_storage_buffers_per_shader_stage: base
            .max_storage_buffers_per_shader_stage
            .max(RASTERIZER_STORAGE_BUFFERS_PER_STAGE),
        max_compute_workgroup_storage_size: base
            .max_compute_workgroup_storage_size
            .max(RASTERIZE_FWD_WORKGROUP_STORAGE_BYTES),
        ..base
    }
}

/// Check that `limits` can host every rasterizer pipeline.
///
/// `what` names the thing being checked ("device" or an adapter name) and is
/// quoted in the error message.
pub(super) fn check_rasterizer_limits(
    limits: &wgpu::Limits,
    what: &str,
) -> Result<(), RenderError> {
    if limits.max_storage_buffers_per_shader_stage < RASTERIZER_STORAGE_BUFFERS_PER_STAGE {
        return Err(RenderError::GpuInit(format!(
            "{what} allows only {} storage buffers per shader stage, but the backward pass \
             binds {RASTERIZER_STORAGE_BUFFERS_PER_STAGE}; build the device with \
             oxigaf_render::rasterizer::rasterizer_device_limits(wgpu::Limits::default())",
            limits.max_storage_buffers_per_shader_stage
        )));
    }
    if limits.max_compute_workgroup_storage_size < RASTERIZE_FWD_WORKGROUP_STORAGE_BYTES {
        return Err(RenderError::GpuInit(format!(
            "{what} allows only {} bytes of compute workgroup storage, but rasterize_fwd.wgsl \
             declares {RASTERIZE_FWD_WORKGROUP_STORAGE_BYTES}; build the device with \
             oxigaf_render::rasterizer::rasterizer_device_limits(wgpu::Limits::default())",
            limits.max_compute_workgroup_storage_size
        )));
    }
    Ok(())
}

/// Check that `size` is a workgroup size the device can actually run.
///
/// [`RasterConfig::effective_preprocess_wg_size`](crate::config::RasterConfig::effective_preprocess_wg_size)
/// is compiled straight into
/// the retargetable 1-D kernels, so a value the device rejects would otherwise
/// surface as an opaque shader-compilation failure.
pub(super) fn check_linear_workgroup_size(
    limits: &wgpu::Limits,
    size: u32,
) -> Result<(), RenderError> {
    let ceiling = limits
        .max_compute_invocations_per_workgroup
        .min(limits.max_compute_workgroup_size_x);
    if size == 0 || size > ceiling {
        return Err(RenderError::Rasterize(format!(
            "RasterConfig's effective preprocess workgroup size {size} is outside the device's \
             supported range 1..={ceiling}"
        )));
    }
    if !size.is_power_of_two() {
        return Err(RenderError::Rasterize(format!(
            "RasterConfig's effective preprocess workgroup size {size} must be a power of two"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RasterConfig;
    use crate::workgroup::WorkgroupConfig;

    // ---- Device limits (F316) ----------------------------------------------

    /// Size and alignment of a WGSL type, in bytes.
    fn wgsl_type_layout(name: &str) -> Option<(u32, u32)> {
        Some(match name {
            "u32" | "i32" | "f32" | "atomic<u32>" | "atomic<i32>" => (4, 4),
            "vec2<u32>" | "vec2<i32>" | "vec2<f32>" => (8, 8),
            // vec3 occupies 12 bytes but aligns to 16.
            "vec3<u32>" | "vec3<i32>" | "vec3<f32>" => (12, 16),
            "vec4<u32>" | "vec4<i32>" | "vec4<f32>" => (16, 16),
            _ => return None,
        })
    }

    /// Total `var<workgroup>` bytes a WGSL source declares, following WGSL's
    /// alignment rules (array stride = element size rounded up to its
    /// alignment; each variable starts at its own alignment).
    fn wgsl_workgroup_storage_bytes(src: &str) -> Option<u32> {
        let mut offset: u32 = 0;
        for line in src.lines() {
            let Some(rest) = line.trim().strip_prefix("var<workgroup>") else {
                continue;
            };
            let (_name, decl) = rest.split_once(':')?;
            let decl = decl.trim().trim_end_matches(';').trim();
            let inner = decl.strip_prefix("array<")?.strip_suffix('>')?;
            let (elem, count) = inner.rsplit_once(',')?;
            let count: u32 = count.trim().trim_end_matches('u').parse().ok()?;
            let (size, align) = wgsl_type_layout(elem.trim())?;
            let stride = size.next_multiple_of(align);
            offset = offset.next_multiple_of(align) + stride * count;
        }
        Some(offset)
    }

    /// Regression (F316): the forward tile kernel's shared memory is bigger
    /// than `wgpu::Limits::default()` allows, so a device built with plain
    /// defaults cannot create the pipeline at all. This pins the figure to
    /// the shader itself — adding another `var<workgroup>` array without
    /// raising the requested limit fails here instead of on the GPU.
    #[test]
    fn test_rasterize_fwd_workgroup_storage_matches_shader() {
        let src = include_str!("../../shaders/rasterize_fwd.wgsl");
        let bytes = wgsl_workgroup_storage_bytes(src)
            .expect("every var<workgroup> in rasterize_fwd.wgsl must be a recognised array type");
        assert_eq!(
            bytes, RASTERIZE_FWD_WORKGROUP_STORAGE_BYTES,
            "rasterize_fwd.wgsl declares {bytes} bytes of workgroup storage but the constant says \
             {RASTERIZE_FWD_WORKGROUP_STORAGE_BYTES}"
        );
        assert!(
            bytes > wgpu::Limits::default().max_compute_workgroup_storage_size,
            "this constant only exists because the shader exceeds the default limit"
        );
    }

    #[test]
    fn test_rasterizer_device_limits_raise_both_ceilings() {
        let limits = rasterizer_device_limits(wgpu::Limits::default());
        assert_eq!(
            limits.max_storage_buffers_per_shader_stage,
            RASTERIZER_STORAGE_BUFFERS_PER_STAGE
        );
        assert_eq!(
            limits.max_compute_workgroup_storage_size,
            RASTERIZE_FWD_WORKGROUP_STORAGE_BYTES
        );
        // Unrelated limits are passed through untouched.
        assert_eq!(
            limits.max_buffer_size,
            wgpu::Limits::default().max_buffer_size
        );
    }

    /// A caller that already asks for more than the rasterizer needs (e.g. by
    /// starting from `adapter.limits()`) must not be silently downgraded.
    #[test]
    fn test_rasterizer_device_limits_never_lower_a_richer_base() {
        let generous = wgpu::Limits {
            max_storage_buffers_per_shader_stage: 64,
            max_compute_workgroup_storage_size: 65_536,
            ..wgpu::Limits::default()
        };
        let limits = rasterizer_device_limits(generous);
        assert_eq!(limits.max_storage_buffers_per_shader_stage, 64);
        assert_eq!(limits.max_compute_workgroup_storage_size, 65_536);
    }

    #[test]
    fn test_check_rasterizer_limits_rejects_defaults_and_accepts_raised() {
        let err = check_rasterizer_limits(&wgpu::Limits::default(), "device")
            .expect_err("default limits cannot host the rasterizer");
        assert!(matches!(err, RenderError::GpuInit(_)));

        check_rasterizer_limits(&rasterizer_device_limits(wgpu::Limits::default()), "device")
            .expect("the limits Rasterizer::new requests must pass its own check");
    }

    /// Regression (F307): the configured 1-D workgroup size is compiled into
    /// the retargetable kernels, so an unusable value must be rejected with a
    /// clear message instead of surfacing as a shader-compilation failure.
    #[test]
    fn test_check_linear_workgroup_size_bounds() {
        let limits = wgpu::Limits::default();
        let ceiling = limits
            .max_compute_invocations_per_workgroup
            .min(limits.max_compute_workgroup_size_x);

        // The shipped size and every preset value must pass.
        for size in [32u32, 64, 128, 256] {
            check_linear_workgroup_size(&limits, size)
                .unwrap_or_else(|e| panic!("{size} threads must be accepted: {e}"));
        }

        assert!(check_linear_workgroup_size(&limits, 0).is_err());
        assert!(check_linear_workgroup_size(&limits, ceiling + 1).is_err());
        let msg = check_linear_workgroup_size(&limits, 48)
            .expect_err("48 is not a power of two")
            .to_string();
        assert!(msg.contains("power of two"), "{msg}");
    }

    /// Every [`GpuPreset`](crate::config::GpuPreset) must produce a workgroup
    /// size the rasterizer can actually compile — otherwise selecting a
    /// preset would make `Rasterizer::from_device` fail.
    #[test]
    fn test_every_gpu_preset_yields_a_usable_workgroup_size() {
        use crate::config::GpuPreset;
        let limits = wgpu::Limits::default();
        for preset in [
            GpuPreset::Auto,
            GpuPreset::Nvidia,
            GpuPreset::Amd,
            GpuPreset::Apple,
            GpuPreset::Intel,
            GpuPreset::Generic,
        ] {
            let config = RasterConfig::default().with_gpu_preset(preset);
            let size = config.effective_preprocess_wg_size();
            check_linear_workgroup_size(&limits, size)
                .unwrap_or_else(|e| panic!("{preset:?} yields an unusable size {size}: {e}"));
            WorkgroupConfig::for_linear_size(size)
                .validate()
                .unwrap_or_else(|e| panic!("{preset:?} yields an invalid config: {e}"));
        }
    }

    /// Each shortfall must be reported on its own terms, so the message names
    /// the limit that is actually too small.
    #[test]
    fn test_check_rasterizer_limits_names_the_failing_limit() {
        let full = rasterizer_device_limits(wgpu::Limits::default());

        let few_buffers = wgpu::Limits {
            max_storage_buffers_per_shader_stage: 8,
            ..full.clone()
        };
        let msg = check_rasterizer_limits(&few_buffers, "device")
            .expect_err("8 storage buffers is not enough")
            .to_string();
        assert!(msg.contains("storage buffers"), "{msg}");

        let small_shared = wgpu::Limits {
            max_compute_workgroup_storage_size: 16_384,
            ..full
        };
        let msg = check_rasterizer_limits(&small_shared, "adapter")
            .expect_err("16 KiB of workgroup storage is not enough")
            .to_string();
        assert!(msg.contains("workgroup storage"), "{msg}");
        assert!(msg.contains("adapter"), "{msg}");
    }
}
