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

    GaussianModel {
        gaussians,
        sh_coeffs,
        sh_degree,
        face_indices: vec![],
        barycentric: vec![],
        local_offsets: vec![],
        is_rigid: vec![],
    }
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

    let model = GaussianModel {
        gaussians: vec![gaussian],
        sh_coeffs: vec![0.5, 0.5, 0.5],
        sh_degree,
        face_indices: vec![],
        barycentric: vec![],
        local_offsets: vec![],
        is_rigid: vec![],
    };

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
