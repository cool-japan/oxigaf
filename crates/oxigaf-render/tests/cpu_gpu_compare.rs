//! CPU vs GPU rendering comparison diagnostic test.
//!
//! This test creates the same scene used by the gradient verification tests,
//! renders it on both the CPU reference rasterizer and the GPU rasterizer,
//! then compares the pixel outputs to identify rendering mismatches.
//!
//! The gradient verification tests fail because numerical gradients (CPU)
//! and analytical gradients (GPU) operate on different loss functions when
//! the two renderers produce different pixel values. This diagnostic helps
//! quantify and pinpoint those differences.

use nalgebra as na;
use oxigaf_render::config::RasterConfig;
use oxigaf_render::gaussian::{GaussianAttributes, GaussianModel};
use oxigaf_render::{CpuCamera, CpuRasterizer, Rasterizer, RenderCamera, RenderError};

// ---------------------------------------------------------------------------
// Scene & camera helpers (mirror gradient_verification/mod.rs)
// ---------------------------------------------------------------------------

/// Create the standard test scene matching gradient_verification::create_test_scene.
fn create_test_scene(num_gaussians: usize, sh_degree: u32, seed: u64) -> GaussianModel {
    let mut gaussians = Vec::new();
    let mut sh_coeffs = Vec::new();
    let sh_coeffs_per_gaussian = ((sh_degree + 1) * (sh_degree + 1) * 3) as usize;

    for i in 0..num_gaussians {
        let offset = (i as f32 + seed as f32 * 0.01) * 0.1;

        let position = [
            (offset * 3.0).sin() * 0.5,
            (offset * 5.0).sin() * 0.5,
            -3.0 - offset,
        ];

        let angle = offset;
        let axis = na::Vector3::new(0.0, 1.0, 0.0);
        let quat = na::UnitQuaternion::from_axis_angle(&na::Unit::new_normalize(axis), angle);
        let rotation = [quat.coords.x, quat.coords.y, quat.coords.z, quat.coords.w];

        let scale = [
            -1.0 + offset * 0.1,
            -1.0 + offset * 0.1,
            -1.0 + offset * 0.1,
        ];

        let opacity = offset * 0.5;

        gaussians.push(GaussianAttributes {
            position,
            _pad0: 0.0,
            rotation,
            scale,
            opacity,
        });

        for j in 0..sh_coeffs_per_gaussian {
            sh_coeffs.push(((i * sh_coeffs_per_gaussian + j) as f32 * 0.01).sin() * 0.5);
        }
    }

    model_from_parts(gaussians, sh_coeffs, sh_degree)
}

/// Assemble a [`GaussianModel`] with its FLAME binding arrays sized to the
/// Gaussian count.
///
/// `GaussianModel`'s invariant is that `face_indices`, `barycentric`,
/// `local_offsets` and `is_rigid` are all parallel to `gaussians` - code that
/// indexes them in lockstep (FLAME deform, density-control clone/split) panics
/// or reads misaligned data otherwise. The hand-written literals this replaces
/// left them empty. The defaults mirror `GaussianModel::load_ply`'s "no
/// binding" case.
fn model_from_parts(
    gaussians: Vec<GaussianAttributes>,
    sh_coeffs: Vec<f32>,
    sh_degree: u32,
) -> GaussianModel {
    let n = gaussians.len();
    let third = 1.0_f32 / 3.0_f32;

    GaussianModel {
        gaussians,
        sh_coeffs,
        sh_degree,
        face_indices: vec![0u32; n],
        barycentric: vec![[third, third, third]; n],
        local_offsets: vec![[0.0, 0.0, 0.0]; n],
        is_rigid: vec![false; n],
    }
}

/// Create a dense cluster of `num_gaussians` Gaussians, all in front of the
/// camera and all overlapping the frame.
///
/// Unlike [`create_test_scene`], which walks Gaussians steadily away from the
/// camera (`z = -3 - i * 0.1`, so the 2000th sits 200 units out and is culled),
/// this keeps every Gaussian inside a compact box, so the whole scene is
/// actually rasterized. Used as the visible payload of
/// `test_gpu_render_is_invariant_across_prefix_sum_blocks`.
fn create_dense_scene(num_gaussians: usize, seed: u64) -> GaussianModel {
    let mut gaussians = Vec::with_capacity(num_gaussians);
    let mut sh_coeffs = Vec::with_capacity(num_gaussians * 3);

    for i in 0..num_gaussians {
        // Deterministic, well-spread pseudo-random coordinates: irrational
        // multipliers keep successive Gaussians from lining up on a lattice.
        let t = (i as f32) + (seed as f32) * 0.001;
        let x = (t * 1.618_034).sin() * 0.9;
        let y = (t * 2.399_963).sin() * 0.9;
        // Depths must be DISTINCT. The sort key is `(tile_id, depth_bits)`,
        // so Gaussians at exactly equal depths are ordered by whatever the
        // radix sort's tie-break happens to be - which depends on the order
        // the tile-assignment pass emitted them in, and therefore on the tile
        // grid. Equal depths would make the rendered image depend on the
        // resolution through blend order alone, destroying the GPU-vs-GPU
        // oracles below. The golden-ratio fractional sequence is
        // equidistributed and repeats no value.
        let z = -3.0 - ((i as f32) * 0.618_034).fract() * 1.5;

        gaussians.push(GaussianAttributes {
            position: [x, y, z],
            _pad0: 0.0,
            rotation: [0.0, 0.0, 0.0, 1.0],
            // exp(-2.3) ~ 0.1 world units: small enough that a few thousand
            // Gaussians stay within the sort-pair budget, large enough to cover
            // several tiles each.
            scale: [-2.3, -2.3, -2.3],
            opacity: -1.0,
        });

        sh_coeffs.push((t * 0.31).sin() * 0.5);
        sh_coeffs.push((t * 0.47).sin() * 0.5);
        sh_coeffs.push((t * 0.53).sin() * 0.5);
    }

    model_from_parts(gaussians, sh_coeffs, 0)
}

/// Cached result of probing for a usable GPU adapter (see [`gpu_available`]).
static GPU_AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

/// Whether a compatible GPU adapter is available in this environment.
///
/// The regression tests below construct a real [`Rasterizer`], which fails with
/// `RenderError::GpuInit`/`AdapterNotFound` on a machine with no compatible
/// adapter (e.g. many headless CI runners). They call this at the top and
/// return early when it is `false` instead of being permanently `#[ignore]`d,
/// so they still run - and gate CI - on any machine that does have a GPU.
fn gpu_available() -> bool {
    *GPU_AVAILABLE.get_or_init(|| {
        match pollster::block_on(Rasterizer::new(RasterConfig::new())) {
            Ok(_) => true,
            Err(err) => {
                eprintln!(
                    "skipping GPU-dependent comparison test: no compatible GPU adapter available ({err})"
                );
                false
            }
        }
    })
}

/// Create the standard test camera matching gradient_verification::create_test_camera.
fn create_test_camera(resolution: (u32, u32)) -> CpuCamera {
    let (width, height) = resolution;

    let view = na::Matrix4::look_at_rh(
        &na::Point3::new(0.0, 0.0, 0.0),
        &na::Point3::new(0.0, 0.0, -1.0),
        &na::Vector3::y(),
    );

    let fov_y = 45.0f32.to_radians();
    let aspect = width as f32 / height as f32;
    let near = 0.1;
    let far = 100.0;
    let proj = na::Matrix4::new_perspective(aspect, fov_y, near, far);

    let focal_y = height as f32 / (2.0 * (fov_y / 2.0).tan());
    let focal_x = focal_y;
    let focal = na::Vector2::new(focal_x, focal_y);

    CpuCamera {
        view,
        proj,
        position: na::Vector3::zeros(),
        focal,
    }
}

/// Convert a CpuCamera to a GPU RenderCamera.
fn cpu_to_render_camera(camera: &CpuCamera) -> Result<RenderCamera, RenderError> {
    let view_matrix: [f32; 16] =
        camera.view.as_slice().try_into().map_err(|_| {
            RenderError::Rasterize("Failed to convert view matrix to [f32; 16]".into())
        })?;
    let proj_matrix: [f32; 16] =
        camera.proj.as_slice().try_into().map_err(|_| {
            RenderError::Rasterize("Failed to convert proj matrix to [f32; 16]".into())
        })?;

    Ok(RenderCamera {
        view_matrix,
        proj_matrix,
        position: [camera.position.x, camera.position.y, camera.position.z],
        focal: [camera.focal.x, camera.focal.y],
    })
}

// ---------------------------------------------------------------------------
// Per-channel statistics
// ---------------------------------------------------------------------------

/// Per-channel pixel difference statistics.
#[derive(Debug, Default)]
struct ChannelStats {
    max_abs_diff: f32,
    sum_abs_diff: f64,
    sum_sq_diff: f64,
    count: usize,
}

impl ChannelStats {
    fn update(&mut self, cpu_val: f32, gpu_val: f32) {
        let diff = (cpu_val - gpu_val).abs();
        if diff > self.max_abs_diff {
            self.max_abs_diff = diff;
        }
        self.sum_abs_diff += diff as f64;
        self.sum_sq_diff += (diff as f64) * (diff as f64);
        self.count += 1;
    }

    fn mean_abs_diff(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        self.sum_abs_diff / self.count as f64
    }

    fn mse(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        self.sum_sq_diff / self.count as f64
    }
}

/// Full comparison report for CPU vs GPU rendering.
struct ComparisonReport {
    total_pixels: usize,
    differing_pixels: usize,
    channel_stats: [ChannelStats; 4], // R, G, B, A
    overall_mse: f64,
    cpu_nonzero_pixels: usize,
    gpu_nonzero_pixels: usize,
}

fn compare_outputs(
    cpu_data: &[f32],
    gpu_data: &[f32],
    width: u32,
    height: u32,
) -> ComparisonReport {
    let total_pixels = (width * height) as usize;
    let mut channel_stats = [
        ChannelStats::default(),
        ChannelStats::default(),
        ChannelStats::default(),
        ChannelStats::default(),
    ];
    let mut differing_pixels = 0usize;
    let mut cpu_nonzero = 0usize;
    let mut gpu_nonzero = 0usize;
    let mut overall_sq_sum = 0.0f64;

    for px in 0..total_pixels {
        let base = px * 4;
        let cpu_r = cpu_data[base];
        let cpu_g = cpu_data[base + 1];
        let cpu_b = cpu_data[base + 2];
        let cpu_a = cpu_data[base + 3];
        let gpu_r = gpu_data[base];
        let gpu_g = gpu_data[base + 1];
        let gpu_b = gpu_data[base + 2];
        let gpu_a = gpu_data[base + 3];

        channel_stats[0].update(cpu_r, gpu_r);
        channel_stats[1].update(cpu_g, gpu_g);
        channel_stats[2].update(cpu_b, gpu_b);
        channel_stats[3].update(cpu_a, gpu_a);

        let any_diff = (cpu_r - gpu_r).abs() > 1e-6
            || (cpu_g - gpu_g).abs() > 1e-6
            || (cpu_b - gpu_b).abs() > 1e-6
            || (cpu_a - gpu_a).abs() > 1e-6;
        if any_diff {
            differing_pixels += 1;
        }

        if cpu_r.abs() > 1e-8 || cpu_g.abs() > 1e-8 || cpu_b.abs() > 1e-8 || cpu_a.abs() > 1e-8 {
            cpu_nonzero += 1;
        }
        if gpu_r.abs() > 1e-8 || gpu_g.abs() > 1e-8 || gpu_b.abs() > 1e-8 || gpu_a.abs() > 1e-8 {
            gpu_nonzero += 1;
        }

        overall_sq_sum += ((cpu_r - gpu_r) as f64).powi(2)
            + ((cpu_g - gpu_g) as f64).powi(2)
            + ((cpu_b - gpu_b) as f64).powi(2)
            + ((cpu_a - gpu_a) as f64).powi(2);
    }

    let overall_mse = overall_sq_sum / (total_pixels as f64 * 4.0);

    ComparisonReport {
        total_pixels,
        differing_pixels,
        channel_stats,
        overall_mse,
        cpu_nonzero_pixels: cpu_nonzero,
        gpu_nonzero_pixels: gpu_nonzero,
    }
}

fn print_report(report: &ComparisonReport, label: &str) {
    let channel_names = ["R", "G", "B", "A"];
    println!();
    println!("=== CPU vs GPU Comparison: {label} ===");
    println!("  Total pixels:        {}", report.total_pixels);
    println!(
        "  Differing pixels:    {} ({:.2}%)",
        report.differing_pixels,
        report.differing_pixels as f64 / report.total_pixels as f64 * 100.0
    );
    println!("  CPU non-zero pixels: {}", report.cpu_nonzero_pixels);
    println!("  GPU non-zero pixels: {}", report.gpu_nonzero_pixels);
    println!("  Overall MSE:         {:.10e}", report.overall_mse);
    println!();
    println!("  Per-channel statistics:");
    for (i, name) in channel_names.iter().enumerate() {
        let s = &report.channel_stats[i];
        println!(
            "    {name}: max_abs_diff={:.8e}, mean_abs_diff={:.8e}, mse={:.8e}",
            s.max_abs_diff,
            s.mean_abs_diff(),
            s.mse()
        );
    }
}

fn print_pixel_samples(cpu_data: &[f32], gpu_data: &[f32], num_samples: usize) {
    println!();
    println!("  First {num_samples} pixels (RGBA):");
    println!("  {:>5}  {:>42}  {:>42}", "idx", "CPU", "GPU");
    for px in 0..num_samples.min(cpu_data.len() / 4) {
        let base = px * 4;
        println!(
            "  {:>5}  [{:>9.6}, {:>9.6}, {:>9.6}, {:>9.6}]  [{:>9.6}, {:>9.6}, {:>9.6}, {:>9.6}]",
            px,
            cpu_data[base],
            cpu_data[base + 1],
            cpu_data[base + 2],
            cpu_data[base + 3],
            gpu_data[base],
            gpu_data[base + 1],
            gpu_data[base + 2],
            gpu_data[base + 3],
        );
    }
}

fn print_first_differing_pixels(
    cpu_data: &[f32],
    gpu_data: &[f32],
    width: u32,
    num_samples: usize,
) {
    let total_pixels = cpu_data.len() / 4;
    let mut printed = 0;
    println!();
    println!("  First {num_samples} differing pixels:");
    println!(
        "  {:>5} ({:>4},{:>4})  {:>42}  {:>42}",
        "idx", "x", "y", "CPU", "GPU"
    );
    for px in 0..total_pixels {
        if printed >= num_samples {
            break;
        }
        let base = px * 4;
        let any_diff = (cpu_data[base] - gpu_data[base]).abs() > 1e-6
            || (cpu_data[base + 1] - gpu_data[base + 1]).abs() > 1e-6
            || (cpu_data[base + 2] - gpu_data[base + 2]).abs() > 1e-6
            || (cpu_data[base + 3] - gpu_data[base + 3]).abs() > 1e-6;
        if any_diff {
            let x = px % width as usize;
            let y = px / width as usize;
            println!(
                "  {:>5} ({:>4},{:>4})  [{:>9.6}, {:>9.6}, {:>9.6}, {:>9.6}]  [{:>9.6}, {:>9.6}, {:>9.6}, {:>9.6}]",
                px, x, y,
                cpu_data[base], cpu_data[base + 1], cpu_data[base + 2], cpu_data[base + 3],
                gpu_data[base], gpu_data[base + 1], gpu_data[base + 2], gpu_data[base + 3],
            );
            printed += 1;
        }
    }
    if printed == 0 {
        println!("  (no differing pixels found)");
    }
}

// ---------------------------------------------------------------------------
// MSE loss comparison
// ---------------------------------------------------------------------------

/// Compute RGB-only MSE loss for a rendering against a black target.
/// This matches the loss used in gradient verification (MseLoss).
fn compute_mse_loss(color_data: &[f32]) -> f64 {
    let num_pixels = color_data.len() / 4;
    let rgb_count = num_pixels * 3;
    let mse: f64 = color_data
        .chunks(4)
        .map(|c| (c[0] as f64).powi(2) + (c[1] as f64).powi(2) + (c[2] as f64).powi(2))
        .sum::<f64>()
        / rgb_count as f64;
    mse
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires GPU hardware"]
fn test_cpu_gpu_compare_default_scene() {
    let resolution = (128, 128);
    let sh_degree = 0u32;
    let num_gaussians = 5;
    let seed = 42u64;

    let model = create_test_scene(num_gaussians, sh_degree, seed);
    let camera = create_test_camera(resolution);

    let config = RasterConfig::new()
        .with_resolution(resolution.0, resolution.1)
        .with_sh_degree(sh_degree);

    // --- CPU rendering ---
    let cpu_rasterizer = CpuRasterizer::new(config.clone());
    let cpu_output = cpu_rasterizer
        .render(&model, &camera)
        .expect("CPU render failed");

    // --- GPU rendering ---
    let render_camera = cpu_to_render_camera(&camera).expect("Camera conversion failed");
    let gpu_output = pollster::block_on(async {
        let mut rasterizer = Rasterizer::new(config.clone())
            .await
            .expect("GPU rasterizer creation failed");
        rasterizer.upload_gaussians(&model);
        rasterizer
            .forward(&model, &render_camera)
            .expect("GPU forward pass failed")
    });

    // --- Compare ---
    let report = compare_outputs(
        &cpu_output.color_data,
        &gpu_output.color_data,
        resolution.0,
        resolution.1,
    );
    print_report(&report, "Default scene (5 Gaussians, 128x128, SH0)");
    print_pixel_samples(&cpu_output.color_data, &gpu_output.color_data, 10);
    print_first_differing_pixels(
        &cpu_output.color_data,
        &gpu_output.color_data,
        resolution.0,
        10,
    );

    // --- Loss comparison ---
    let cpu_loss = compute_mse_loss(&cpu_output.color_data);
    let gpu_loss = compute_mse_loss(&gpu_output.color_data);
    println!();
    println!("  MSE Loss (vs black target):");
    println!("    CPU loss: {cpu_loss:.10e}");
    println!("    GPU loss: {gpu_loss:.10e}");
    println!("    Loss diff (abs): {:.10e}", (cpu_loss - gpu_loss).abs());
    println!(
        "    Loss diff (rel): {:.10e}",
        (cpu_loss - gpu_loss).abs() / (cpu_loss.abs() + 1e-15)
    );

    // Non-fatal: print a warning if there is a mismatch
    if report.overall_mse > 1e-6 {
        println!();
        println!(
            "  WARNING: Significant rendering mismatch detected (MSE = {:.6e})",
            report.overall_mse
        );
        println!("  This will cause gradient verification tests to fail because the");
        println!("  CPU (finite-diff) and GPU (analytical) renderers compute gradients");
        println!("  of DIFFERENT loss surfaces.");
    } else {
        println!();
        println!(
            "  OK: CPU and GPU renderings match within tolerance (MSE = {:.6e})",
            report.overall_mse
        );
    }
}

#[test]
#[ignore = "requires GPU hardware"]
fn test_cpu_gpu_compare_64x64() {
    let resolution = (64, 64);
    let sh_degree = 0u32;
    let num_gaussians = 3;
    let seed = 42u64;

    let model = create_test_scene(num_gaussians, sh_degree, seed);
    let camera = create_test_camera(resolution);

    let config = RasterConfig::new()
        .with_resolution(resolution.0, resolution.1)
        .with_sh_degree(sh_degree);

    let cpu_rasterizer = CpuRasterizer::new(config.clone());
    let cpu_output = cpu_rasterizer
        .render(&model, &camera)
        .expect("CPU render failed");

    let render_camera = cpu_to_render_camera(&camera).expect("Camera conversion failed");
    let gpu_output = pollster::block_on(async {
        let mut rasterizer = Rasterizer::new(config.clone())
            .await
            .expect("GPU rasterizer creation failed");
        rasterizer.upload_gaussians(&model);
        rasterizer
            .forward(&model, &render_camera)
            .expect("GPU forward pass failed")
    });

    let report = compare_outputs(
        &cpu_output.color_data,
        &gpu_output.color_data,
        resolution.0,
        resolution.1,
    );
    print_report(&report, "Small scene (3 Gaussians, 64x64, SH0)");
    print_first_differing_pixels(
        &cpu_output.color_data,
        &gpu_output.color_data,
        resolution.0,
        10,
    );

    let cpu_loss = compute_mse_loss(&cpu_output.color_data);
    let gpu_loss = compute_mse_loss(&gpu_output.color_data);
    println!();
    println!("  MSE Loss (vs black target):");
    println!("    CPU loss: {cpu_loss:.10e}");
    println!("    GPU loss: {gpu_loss:.10e}");
    println!("    Loss diff (abs): {:.10e}", (cpu_loss - gpu_loss).abs());
}

#[test]
#[ignore = "requires GPU hardware"]
fn test_cpu_gpu_compare_single_gaussian() {
    // Simplest possible case: one Gaussian, SH degree 0.
    // If even this disagrees, there is a fundamental projection / blending mismatch.
    let resolution = (64, 64);
    let sh_degree = 0u32;

    let gaussian = GaussianAttributes {
        position: [0.0, 0.0, -3.0],
        _pad0: 0.0,
        rotation: [0.0, 0.0, 0.0, 1.0],
        scale: [-1.0, -1.0, -1.0],
        opacity: 0.0,
    };

    let model = model_from_parts(vec![gaussian], vec![0.5, 0.5, 0.5], sh_degree);

    let camera = create_test_camera(resolution);
    let config = RasterConfig::new()
        .with_resolution(resolution.0, resolution.1)
        .with_sh_degree(sh_degree);

    let cpu_rasterizer = CpuRasterizer::new(config.clone());
    let cpu_output = cpu_rasterizer
        .render(&model, &camera)
        .expect("CPU render failed");

    let render_camera = cpu_to_render_camera(&camera).expect("Camera conversion failed");
    let gpu_output = pollster::block_on(async {
        let mut rasterizer = Rasterizer::new(config.clone())
            .await
            .expect("GPU rasterizer creation failed");
        rasterizer.upload_gaussians(&model);
        rasterizer
            .forward(&model, &render_camera)
            .expect("GPU forward pass failed")
    });

    let report = compare_outputs(
        &cpu_output.color_data,
        &gpu_output.color_data,
        resolution.0,
        resolution.1,
    );
    print_report(
        &report,
        "Single Gaussian (identity rotation, center, 64x64, SH0)",
    );
    print_pixel_samples(&cpu_output.color_data, &gpu_output.color_data, 10);
    print_first_differing_pixels(
        &cpu_output.color_data,
        &gpu_output.color_data,
        resolution.0,
        10,
    );

    // Print center pixel in detail
    let cx = resolution.0 / 2;
    let cy = resolution.1 / 2;
    let center_idx = (cy * resolution.0 + cx) as usize * 4;
    println!();
    println!("  Center pixel ({cx}, {cy}):");
    println!(
        "    CPU: [{:.8}, {:.8}, {:.8}, {:.8}]",
        cpu_output.color_data[center_idx],
        cpu_output.color_data[center_idx + 1],
        cpu_output.color_data[center_idx + 2],
        cpu_output.color_data[center_idx + 3],
    );
    println!(
        "    GPU: [{:.8}, {:.8}, {:.8}, {:.8}]",
        gpu_output.color_data[center_idx],
        gpu_output.color_data[center_idx + 1],
        gpu_output.color_data[center_idx + 2],
        gpu_output.color_data[center_idx + 3],
    );

    let cpu_loss = compute_mse_loss(&cpu_output.color_data);
    let gpu_loss = compute_mse_loss(&gpu_output.color_data);
    println!();
    println!("  MSE Loss (vs black target):");
    println!("    CPU loss: {cpu_loss:.10e}");
    println!("    GPU loss: {gpu_loss:.10e}");
    println!("    Loss diff (abs): {:.10e}", (cpu_loss - gpu_loss).abs());
}

#[test]
#[ignore = "requires GPU hardware"]
fn test_cpu_gpu_compare_intermediate_diagnostics() {
    // Detailed diagnostic: print per-Gaussian projected attributes for a single Gaussian
    // so we can compare projection / SH / blending step by step.
    let resolution = (64, 64);
    let sh_degree = 0u32;

    let model = create_test_scene(1, sh_degree, 42);
    let camera = create_test_camera(resolution);

    let config = RasterConfig::new()
        .with_resolution(resolution.0, resolution.1)
        .with_sh_degree(sh_degree);

    let cpu_rasterizer = CpuRasterizer::new(config.clone());
    let cpu_output = cpu_rasterizer
        .render(&model, &camera)
        .expect("CPU render failed");

    let render_camera = cpu_to_render_camera(&camera).expect("Camera conversion failed");
    let gpu_output = pollster::block_on(async {
        let mut rasterizer = Rasterizer::new(config.clone())
            .await
            .expect("GPU rasterizer creation failed");
        rasterizer.upload_gaussians(&model);
        rasterizer
            .forward(&model, &render_camera)
            .expect("GPU forward pass failed")
    });

    println!();
    println!("=== Intermediate diagnostics (1 Gaussian, 64x64, SH0) ===");

    // Print input Gaussian attributes
    let g = &model.gaussians[0];
    println!("  Input Gaussian:");
    println!(
        "    position:  [{:.6}, {:.6}, {:.6}]",
        g.position[0], g.position[1], g.position[2]
    );
    println!(
        "    rotation:  [{:.6}, {:.6}, {:.6}, {:.6}]",
        g.rotation[0], g.rotation[1], g.rotation[2], g.rotation[3]
    );
    println!(
        "    scale:     [{:.6}, {:.6}, {:.6}]",
        g.scale[0], g.scale[1], g.scale[2]
    );
    println!("    opacity:   {:.6}", g.opacity);
    println!("    sh_coeffs: {:?}", &model.sh_coeffs);

    // Print camera matrices
    println!("  Camera:");
    println!(
        "    view matrix (first 4 vals): [{:.6}, {:.6}, {:.6}, {:.6}]",
        camera.view[(0, 0)],
        camera.view[(0, 1)],
        camera.view[(0, 2)],
        camera.view[(0, 3)]
    );
    println!("    focal: [{:.2}, {:.2}]", camera.focal.x, camera.focal.y);
    println!(
        "    position: [{:.2}, {:.2}, {:.2}]",
        camera.position.x, camera.position.y, camera.position.z
    );

    // Sum of all CPU pixel values (gives a quick overview of brightness)
    let cpu_sum: f64 = cpu_output.color_data.iter().map(|v| *v as f64).sum();
    let gpu_sum: f64 = gpu_output.color_data.iter().map(|v| *v as f64).sum();
    let cpu_rgb_sum: f64 = cpu_output
        .color_data
        .chunks(4)
        .map(|c| c[0] as f64 + c[1] as f64 + c[2] as f64)
        .sum();
    let gpu_rgb_sum: f64 = gpu_output
        .color_data
        .chunks(4)
        .map(|c| c[0] as f64 + c[1] as f64 + c[2] as f64)
        .sum();
    println!();
    println!("  Total pixel value sums:");
    println!("    CPU RGBA sum: {cpu_sum:.6}");
    println!("    GPU RGBA sum: {gpu_sum:.6}");
    println!("    CPU RGB sum:  {cpu_rgb_sum:.6}");
    println!("    GPU RGB sum:  {gpu_rgb_sum:.6}");

    // Histogram of difference magnitudes
    let total_pixels = (resolution.0 * resolution.1) as usize;
    let mut diff_histogram = [0usize; 8]; // bins: 0, <1e-6, <1e-4, <1e-2, <0.1, <0.5, <1.0, >=1.0
    for px in 0..total_pixels {
        let base = px * 4;
        let max_ch_diff = (0..4)
            .map(|c| (cpu_output.color_data[base + c] - gpu_output.color_data[base + c]).abs())
            .fold(0.0f32, f32::max);
        let bin = if max_ch_diff == 0.0 {
            0
        } else if max_ch_diff < 1e-6 {
            1
        } else if max_ch_diff < 1e-4 {
            2
        } else if max_ch_diff < 1e-2 {
            3
        } else if max_ch_diff < 0.1 {
            4
        } else if max_ch_diff < 0.5 {
            5
        } else if max_ch_diff < 1.0 {
            6
        } else {
            7
        };
        diff_histogram[bin] += 1;
    }
    println!();
    println!("  Difference histogram (max per-channel diff per pixel):");
    println!("    exact 0:   {}", diff_histogram[0]);
    println!("    < 1e-6:    {}", diff_histogram[1]);
    println!("    < 1e-4:    {}", diff_histogram[2]);
    println!("    < 1e-2:    {}", diff_histogram[3]);
    println!("    < 0.1:     {}", diff_histogram[4]);
    println!("    < 0.5:     {}", diff_histogram[5]);
    println!("    < 1.0:     {}", diff_histogram[6]);
    println!("    >= 1.0:    {}", diff_histogram[7]);

    let report = compare_outputs(
        &cpu_output.color_data,
        &gpu_output.color_data,
        resolution.0,
        resolution.1,
    );
    print_report(&report, "Intermediate diagnostics");
}

// ---------------------------------------------------------------------------
// GPU forward regression tests (asserting, unlike the diagnostics above)
// ---------------------------------------------------------------------------
//
// These two tests deliberately do NOT use the CPU reference rasterizer as
// their oracle. `CpuRasterizer` and the GPU pipeline disagree by an MSE of
// ~1e-4 on dense, heavily overlapping scenes (see the note in this crate's
// followups) - a pre-existing discrepancy that has nothing to do with the two
// shader fixes being locked in here, and that would drown the effects under
// test. Instead each test compares the GPU against *itself* under a
// transformation that must leave the rendered image unchanged, which makes the
// oracle exact.

/// Render `model` on the GPU only and return the RGBA pixel buffer.
///
/// A discarded warm-up pass runs first. The first forward on a freshly created
/// `Rasterizer` can differ slightly from subsequent ones while the GPU
/// pipeline state settles - `test_gpu_self_consistent_all_params` in
/// `gpu_gradient_verify.rs` documents and works around the same effect - and
/// the comparisons below expect two renders to agree bit-for-bit.
fn render_gpu(model: &GaussianModel, config: &RasterConfig, camera: &CpuCamera) -> Vec<f32> {
    let render_camera = cpu_to_render_camera(camera).expect("Camera conversion failed");
    pollster::block_on(async {
        let mut rasterizer = Rasterizer::new(config.clone())
            .await
            .expect("GPU rasterizer creation failed");
        rasterizer.upload_gaussians(model);
        let _warmup = rasterizer
            .forward(model, &render_camera)
            .expect("GPU warm-up forward pass failed");

        rasterizer.upload_gaussians(model);
        rasterizer
            .forward(model, &render_camera)
            .expect("GPU forward pass failed")
            .color_data
    })
}

/// Mean squared error between two RGBA buffers of the same length.
fn buffer_mse(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len(), "buffers must have the same length");
    assert!(!a.is_empty(), "cannot compare empty buffers");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| ((x - y) as f64).powi(2))
        .sum::<f64>()
        / a.len() as f64
}

/// Count pixels with any non-zero RGB channel.
fn nonzero_pixels(data: &[f32]) -> usize {
    data.chunks_exact(4)
        .filter(|p| p[0].abs() > 1e-8 || p[1].abs() > 1e-8 || p[2].abs() > 1e-8)
        .count()
}

/// Number of coincident copies emitted at each probe position.
///
/// A *cluster* rather than a single Gaussian is essential. The barrier bug
/// this scene guards against corrupts the workgroup's cooperative-load slots
/// owned by threads whose pixel is outside the image, and a thread only owns a
/// slot when `local_idx < batch_size` - i.e. when the tile's Gaussian list is
/// at least as long as that thread's flat index. With one Gaussian per probe
/// the list length is 1, only `local_idx == 0` (an in-bounds pixel) loads
/// anything, and no surviving thread ever reads a slot an out-of-image thread
/// owned: the test would pass with the bug fully reinstated.
///
/// 16 copies put `local_idx 0..15` to work. In the 4-pixel-wide right edge
/// tile of a 100-pixel-wide image, `local_idx = lid.y * 16 + lid.x` means
/// slots 0..3 belong to in-bounds threads and slots 4..15 to threads whose
/// pixel is off the right edge - so three quarters of the batch is loaded by
/// threads the pre-fix shader let leave.
const PROBE_CLUSTER: usize = 16;

/// Build a scene of small Gaussian *clusters* placed at chosen *pixel*
/// positions for `resolution`.
///
/// `create_test_camera` puts the camera at the origin looking down -Z with
/// `focal = height / (2 tan(fov_y / 2))` on both axes, so a Gaussian at world
/// `(x, y, z)` lands at `mean2d = (x, y) * focal / -z + (width, height) / 2`.
/// Inverting that lets a test aim Gaussians straight at the partial edge tiles
/// of an arbitrary resolution.
///
/// The [`PROBE_CLUSTER`] copies at each position are **exactly identical** -
/// same position, rotation, scale, opacity and colour. Alpha compositing of
/// identical layers is permutation-invariant, so the blend order (and hence
/// the radix sort's tie-break, which depends on the tile grid) cannot change
/// the result, and the CPU reference stays an exact oracle. Different probe
/// positions never overlap, so their relative order does not matter either.
fn create_edge_probe_scene(resolution: (u32, u32), pixel_targets: &[(f32, f32)]) -> GaussianModel {
    let (width, height) = (resolution.0 as f32, resolution.1 as f32);
    let fov_y = 45.0f32.to_radians();
    let focal = height / (2.0 * (fov_y / 2.0).tan());
    let depth = 4.0f32;

    // sigma of 1.5 px keeps the 3-sigma footprint near 5 px, far smaller than
    // the spacing between the probe positions.
    let sigma_world = 1.5 * depth / focal;
    let log_scale = sigma_world.ln();

    let count = pixel_targets.len() * PROBE_CLUSTER;
    let mut gaussians = Vec::with_capacity(count);
    let mut sh_coeffs = Vec::with_capacity(count * 3);

    for (i, (px, py)) in pixel_targets.iter().enumerate() {
        let gaussian = GaussianAttributes {
            position: [
                (px - width / 2.0) * depth / focal,
                (py - height / 2.0) * depth / focal,
                // Probe positions get distinct depths; the copies within one
                // probe deliberately share theirs (see the doc comment).
                -depth - i as f32 * 0.01,
            ],
            _pad0: 0.0,
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [log_scale, log_scale, log_scale],
            // sigmoid(-3) ~ 0.047. Deliberately faint: 16 stacked layers then
            // leave transmittance around 0.46 at the core, well above the
            // 1/255 early-out, so every one of the 40 shared-memory slots is
            // actually read. An opaque cluster would terminate after a few
            // layers and never touch the later slots - exactly the ones an
            // out-of-image thread owns.
            opacity: -3.0,
        };

        for _ in 0..PROBE_CLUSTER {
            gaussians.push(gaussian);
            sh_coeffs.extend_from_slice(&[1.0, 0.5, -0.5]);
        }
    }

    model_from_parts(gaussians, sh_coeffs, 0)
}

/// Probe positions for `resolution`: the corner, the right edge, the bottom
/// edge - all inside the partial tiles when the resolution is not a multiple
/// of the 16-pixel tile size - plus two interior controls.
fn edge_probe_targets(resolution: (u32, u32)) -> Vec<(f32, f32)> {
    let (w, h) = (resolution.0 as f32, resolution.1 as f32);
    vec![
        (w - 2.0, h - 2.0),
        (w - 2.0, h / 2.0),
        (w / 2.0, h - 2.0),
        (w / 2.0, h / 2.0),
        (6.0, 6.0),
    ]
}

/// Upper bound on the CPU-vs-GPU MSE for an isolated-Gaussian scene.
///
/// Both rasterizers run the same arithmetic in the same order here, so the
/// only difference is f32 rounding; measured values are around 1e-15. A
/// phantom Gaussian blended out of stale shared memory, or a dropped edge
/// tile, moves whole pixels.
const MAX_SPARSE_CPU_GPU_MSE: f64 = 1e-12;

/// Regression test: partial edge tiles - the tiles at the right and bottom of a
/// resolution that is not a multiple of the 16x16 tile size - must rasterize
/// exactly like the CPU reference.
///
/// `rasterize_fwd.wgsl` used to `return` early from every thread whose pixel
/// lay outside the image. In a partial edge tile that leaves the workgroup's
/// `workgroupBarrier()` calls non-uniform - undefined behaviour - and in
/// practice the shared-memory slots owned by the departed threads keep stale
/// data that the surviving threads then blend as phantom Gaussians. The fix
/// keeps those threads alive through the batch loop and suppresses only their
/// final stores.
///
/// The scene deliberately puts an isolated, opaque Gaussian two pixels inside
/// the bottom-right corner, the right edge and the bottom edge, so at least
/// three partial tiles carry real work; two interior Gaussians act as controls.
/// 100x70 leaves a 4-pixel-wide right edge and a 6-pixel-tall bottom edge;
/// 37x53 is prime on both axes. 96x64 tiles exactly and validates the oracle.
#[test]
fn test_gpu_partial_edge_tiles_match_cpu_reference() {
    if !gpu_available() {
        return;
    }

    for (resolution, partial) in [
        ((96u32, 64u32), false),
        ((100u32, 70u32), true),
        ((37u32, 53u32), true),
    ] {
        assert_eq!(
            !resolution.0.is_multiple_of(16) || !resolution.1.is_multiple_of(16),
            partial,
            "{resolution:?} does not have the expected tiling"
        );

        let model = create_edge_probe_scene(resolution, &edge_probe_targets(resolution));
        let config = RasterConfig::new()
            .with_resolution(resolution.0, resolution.1)
            .with_sh_degree(0);
        let camera = create_test_camera(resolution);

        let cpu = CpuRasterizer::new(config.clone())
            .render(&model, &camera)
            .expect("CPU render failed");
        let gpu = render_gpu(&model, &config, &camera);

        let report = compare_outputs(&cpu.color_data, &gpu, resolution.0, resolution.1);
        print_report(
            &report,
            &format!(
                "Edge probes ({}x{}, partial tiles: {partial})",
                resolution.0, resolution.1
            ),
        );

        // Every probe must actually have rendered something, or a dropped
        // edge tile would look like a pass.
        assert!(
            nonzero_pixels(&cpu.color_data) >= 5,
            "the CPU reference rendered almost nothing at {resolution:?}"
        );
        assert!(
            nonzero_pixels(&gpu) >= 5,
            "the GPU rendered almost nothing at {resolution:?};              an edge tile is being dropped entirely"
        );
        assert_eq!(
            report.differing_pixels, 0,
            "{resolution:?}: {} of {} pixels differ between CPU and GPU -              partial edge tiles are being rasterized incorrectly",
            report.differing_pixels, report.total_pixels
        );
        assert!(
            report.overall_mse < MAX_SPARSE_CPU_GPU_MSE,
            "{resolution:?}: CPU/GPU MSE {:.6e} exceeds {MAX_SPARSE_CPU_GPU_MSE:.0e}",
            report.overall_mse
        );
    }
}

/// Upper bound on the difference between two GPU renders that must be
/// bit-identical: the same visible Gaussians, once alone and once preceded by
/// culled padding. Nothing legitimate differs between them - measured value is
/// exactly 0.0 - so this is pure headroom.
const MAX_PADDING_INVARIANT_MSE: f64 = 1e-12;

/// Regression test: a Gaussian count large enough to need a **multi-block**
/// prefix sum must produce the same image as the same visible Gaussians alone.
///
/// The tile-offset scan runs `shaders/prefix_sum.wgsl` over one element per
/// Gaussian in 512-element blocks, then `shaders/prefix_sum_add.wgsl` folds the
/// scanned block totals back in. That second shader used to add
/// `block_offsets[wid.x]` - the *inclusive* total of the block's own elements -
/// instead of the exclusive prefix `block_offsets[wid.x - 1]`, inflating every
/// block after the first. Each Gaussian's tile-pair write index then landed
/// past its true slot, so `tile_ranges` covered the wrong Gaussians.
///
/// The construction isolates exactly that: the scene is padded at the *front*
/// with Gaussians behind the camera, which `preprocess.wgsl` culls to
/// `tile_counts = 0`. The visible Gaussians therefore sit in block 1 (or 3)
/// with a preceding exclusive prefix of zero, so the correct offsets are
/// identical to the unpadded scene's and the two renders must match exactly.
/// With the bug, block 1 is shifted by its own total and the image changes.
///
/// The bug is invisible below 513 Gaussians - a single block needs no offset
/// pass at all - which is why every other test in this crate missed it.
#[test]
fn test_gpu_render_is_invariant_across_prefix_sum_blocks() {
    if !gpu_available() {
        return;
    }

    // `shaders/prefix_sum.wgsl`: 256 threads x 2 elements each.
    const SCAN_BLOCK: usize = 512;
    const VISIBLE: usize = 48;

    let resolution = (128u32, 128u32);
    let config = RasterConfig::new()
        .with_resolution(resolution.0, resolution.1)
        .with_sh_degree(0);
    let camera = create_test_camera(resolution);

    let visible = create_dense_scene(VISIBLE, 7);
    let reference = render_gpu(&visible, &config, &camera);
    assert!(
        nonzero_pixels(&reference) > 100,
        "the unpadded render is nearly empty; the comparison would be vacuous"
    );

    // 600 -> 2 scan blocks; 2000 -> 4 scan blocks (plus a level-1 scan).
    for total in [600usize, 2000usize] {
        let padding = total - VISIBLE;
        assert!(
            padding >= SCAN_BLOCK,
            "{total} Gaussians leave the visible ones in the first scan block; \
             prefix_sum_add would not be exercised"
        );

        let mut gaussians = Vec::with_capacity(total);
        let mut sh_coeffs = Vec::with_capacity(total * 3);
        for _ in 0..padding {
            // Behind the camera: `preprocess.wgsl` writes radii = -1 and
            // tile_counts = 0, so these contribute nothing but still occupy a
            // scan element each.
            gaussians.push(GaussianAttributes {
                position: [0.0, 0.0, 5.0],
                _pad0: 0.0,
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: [-2.3, -2.3, -2.3],
                opacity: -1.0,
            });
            sh_coeffs.extend_from_slice(&[0.0, 0.0, 0.0]);
        }
        gaussians.extend_from_slice(&visible.gaussians);
        sh_coeffs.extend_from_slice(&visible.sh_coeffs);

        let padded = model_from_parts(gaussians, sh_coeffs, 0);
        assert_eq!(padded.len(), total);

        let image = render_gpu(&padded, &config, &camera);
        let mse = buffer_mse(&image, &reference);
        println!(
            "{} Gaussians ({} scan blocks, {} culled): mse vs unpadded = {:.6e}",
            total,
            total.div_ceil(SCAN_BLOCK),
            padding,
            mse
        );

        assert!(
            nonzero_pixels(&image) > 100,
            "the {total}-Gaussian render is nearly empty; \
             the comparison would be vacuous"
        );
        assert!(
            mse < MAX_PADDING_INVARIANT_MSE,
            "padding to {} Gaussians ({} scan blocks) changed the image \
             (MSE={:.6e}) - the cross-block tile offsets are wrong",
            total,
            total.div_ceil(SCAN_BLOCK),
            mse
        );
    }
}

/// Regression test: the local `create_test_scene` and `create_dense_scene`
/// must keep the FLAME binding arrays parallel to `gaussians`; the
/// hand-written literals they replaced left them empty, breaking the
/// `GaussianModel` invariant that code indexing them in lockstep (FLAME
/// deform, density-control clone/split) relies on.
#[test]
fn test_local_scenes_have_parallel_binding_arrays() {
    for model in [create_test_scene(5, 1, 42), create_dense_scene(9, 3)] {
        let n = model.gaussians.len();
        assert!(n > 0);
        assert_eq!(model.face_indices.len(), n);
        assert_eq!(model.barycentric.len(), n);
        assert_eq!(model.local_offsets.len(), n);
        assert_eq!(model.is_rigid.len(), n);
    }
}

/// The dense scene must put every Gaussian in front of the camera and inside
/// the default clip range, or the tests above would rasterize nothing.
#[test]
fn test_dense_scene_is_in_front_of_the_camera() {
    let model = create_dense_scene(600, 7);

    for (i, gaussian) in model.gaussians.iter().enumerate() {
        let z = gaussian.position[2];
        assert!(
            (-4.5..=-3.0).contains(&z),
            "Gaussian {i} sits at z={z}, outside the intended cluster"
        );
        assert!(
            gaussian.position[0].abs() <= 0.9 && gaussian.position[1].abs() <= 0.9,
            "Gaussian {i} sits outside the intended lateral spread"
        );
    }
}
