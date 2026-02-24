//! Comprehensive benchmarks for oxigaf-trainer.
//!
//! Benchmarks:
//! - L1 loss computation
//! - SSIM loss computation
//! - MS-SSIM loss computation
//! - Adam optimizer step
//! - Gradient operations (clipping, sanitization)
//!
//! Run with: cargo bench -p oxigaf-trainer

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ndarray::Array2;
use std::hint::black_box;

use oxigaf_trainer::loss::{
    clip_gradients_by_norm, clip_gradients_by_value, gaussian_kernel_1d, gradient_penalty,
    gradient_statistics, l1_loss, ms_ssim_loss, sanitize_gradients, ssim_loss, LossComputer,
};
use oxigaf_trainer::LossConfig;

// ---------------------------------------------------------------------------
// L1 Loss Benchmark
// ---------------------------------------------------------------------------

fn bench_l1_loss(c: &mut Criterion) {
    let mut group = c.benchmark_group("l1_loss");

    for &resolution in &[64, 128, 256, 512] {
        let size = resolution * resolution * 3;

        let pred: Vec<f32> = (0..size).map(|i| (i as f32 * 0.001).sin().abs()).collect();
        let target: Vec<f32> = (0..size).map(|i| (i as f32 * 0.002).cos().abs()).collect();

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::new("resolution", resolution),
            &resolution,
            |b, _| b.iter(|| black_box(l1_loss(black_box(&pred), black_box(&target)))),
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// SSIM Loss Benchmark
// ---------------------------------------------------------------------------

fn bench_ssim_loss(c: &mut Criterion) {
    let mut group = c.benchmark_group("ssim_loss");

    let kernel = gaussian_kernel_1d(11, 1.5);

    for &resolution in &[64, 128, 256] {
        let size = resolution * resolution * 3;

        let pred: Vec<f32> = (0..size).map(|i| (i as f32 * 0.001).sin().abs()).collect();
        let target: Vec<f32> = (0..size).map(|i| (i as f32 * 0.002).cos().abs()).collect();

        group.throughput(Throughput::Elements((resolution * resolution) as u64));
        group.bench_with_input(
            BenchmarkId::new("resolution", resolution),
            &resolution,
            |b, _| {
                b.iter(|| {
                    black_box(ssim_loss(
                        black_box(&pred),
                        black_box(&target),
                        resolution,
                        resolution,
                        &kernel,
                    ))
                })
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// MS-SSIM Loss Benchmark
// ---------------------------------------------------------------------------

fn bench_ms_ssim_loss(c: &mut Criterion) {
    let mut group = c.benchmark_group("ms_ssim_loss");

    let weights = LossComputer::DEFAULT_MS_SSIM_WEIGHTS;

    for &resolution in &[64, 128, 256] {
        let size = resolution * resolution * 3;

        let pred: Vec<f32> = (0..size).map(|i| (i as f32 * 0.001).sin().abs()).collect();
        let target: Vec<f32> = (0..size).map(|i| (i as f32 * 0.002).cos().abs()).collect();

        group.throughput(Throughput::Elements((resolution * resolution) as u64));
        group.bench_with_input(
            BenchmarkId::new("resolution", resolution),
            &resolution,
            |b, _| {
                b.iter(|| {
                    black_box(ms_ssim_loss(
                        black_box(&pred),
                        black_box(&target),
                        resolution,
                        resolution,
                        &weights,
                    ))
                })
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Combined Loss Computation Benchmark
// ---------------------------------------------------------------------------

fn bench_combined_loss(c: &mut Criterion) {
    let mut group = c.benchmark_group("combined_loss");

    let config = LossConfig {
        w_l1: 0.8,
        w_ssim: 0.2,
        w_ms_ssim: 0.0,
        w_lpips: 0.0,
        w_position_reg: 0.0001,
        w_scale_reg: 0.0001,
        w_opacity_reg: 0.0001,
        w_normal: 0.0,
        w_gradient_penalty: 0.0,
        gradient_penalty_threshold: 1.0,
    };

    let _loss_computer = LossComputer::new(config);

    for &resolution in &[64, 128, 256] {
        let size = resolution * resolution * 3;

        let pred: Vec<f32> = (0..size).map(|i| (i as f32 * 0.001).sin().abs()).collect();
        let target: Vec<f32> = (0..size).map(|i| (i as f32 * 0.002).cos().abs()).collect();

        // Create a mock GaussianModel-like structure with minimal data
        let rendered = [pred.clone()];
        let targets = [target.clone()];

        group.throughput(Throughput::Elements((resolution * resolution) as u64));
        group.bench_with_input(
            BenchmarkId::new("resolution", resolution),
            &resolution,
            |b, _| {
                b.iter(|| {
                    // Just benchmark L1 + SSIM without model regularization
                    let kernel = gaussian_kernel_1d(11, 1.5);
                    let l1 = l1_loss(&rendered[0], &targets[0]);
                    let ssim =
                        ssim_loss(&rendered[0], &targets[0], resolution, resolution, &kernel);
                    black_box((l1, ssim))
                })
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Gaussian Kernel Generation Benchmark
// ---------------------------------------------------------------------------

fn bench_gaussian_kernel(c: &mut Criterion) {
    let mut group = c.benchmark_group("gaussian_kernel");

    for &size in &[7, 11, 15, 21] {
        group.bench_with_input(BenchmarkId::new("size", size), &size, |b, &s| {
            b.iter(|| black_box(gaussian_kernel_1d(s, 1.5)))
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Separable Convolution Benchmark
// ---------------------------------------------------------------------------

fn bench_separable_convolution(c: &mut Criterion) {
    let mut group = c.benchmark_group("separable_convolution");

    let kernel = gaussian_kernel_1d(11, 1.5);

    for &resolution in &[64, 128, 256, 512] {
        let image = Array2::from_shape_fn((resolution, resolution), |(y, x)| {
            ((y * resolution + x) as f32 * 0.001).sin()
        });

        group.throughput(Throughput::Elements((resolution * resolution) as u64));
        group.bench_with_input(
            BenchmarkId::new("resolution", resolution),
            &resolution,
            |b, _| {
                b.iter(|| {
                    // Inline separable convolution for benchmarking
                    let (h, w) = (resolution, resolution);
                    let k = kernel.len();
                    let half = k / 2;

                    // Horizontal pass
                    let mut temp = Array2::zeros((h, w));
                    for y in 0..h {
                        for x in 0..w {
                            let mut sum = 0.0_f32;
                            #[allow(clippy::needless_range_loop)]
                            for i in 0..k {
                                let ix = (x as isize + i as isize - half as isize)
                                    .max(0)
                                    .min(w as isize - 1)
                                    as usize;
                                sum += image[[y, ix]] * kernel[i];
                            }
                            temp[[y, x]] = sum;
                        }
                    }

                    // Vertical pass
                    let mut out = Array2::zeros((h, w));
                    for y in 0..h {
                        for x in 0..w {
                            let mut sum = 0.0_f32;
                            #[allow(clippy::needless_range_loop)]
                            for i in 0..k {
                                let iy = (y as isize + i as isize - half as isize)
                                    .max(0)
                                    .min(h as isize - 1)
                                    as usize;
                                sum += temp[[iy, x]] * kernel[i];
                            }
                            out[[y, x]] = sum;
                        }
                    }

                    black_box(out)
                })
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Adam Optimizer Step Benchmark
// ---------------------------------------------------------------------------

/// Inline Adam update for benchmarking without needing full GaussianModel.
#[inline]
#[allow(clippy::too_many_arguments)]
fn adam_step_inline(
    params: &mut [f32],
    grads: &[f32],
    m: &mut [f32],
    v: &mut [f32],
    t: u32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
) {
    let bc1 = 1.0 - beta1.powi(t as i32);
    let bc2 = 1.0 - beta2.powi(t as i32);

    for i in 0..params.len() {
        m[i] = beta1 * m[i] + (1.0 - beta1) * grads[i];
        v[i] = beta2 * v[i] + (1.0 - beta2) * grads[i] * grads[i];
        let m_hat = m[i] / bc1;
        let v_hat = v[i] / bc2;
        params[i] -= lr * m_hat / (v_hat.sqrt() + epsilon);
    }
}

fn bench_adam_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("adam_step");

    let lr = 0.001;
    let beta1 = 0.9;
    let beta2 = 0.999;
    let epsilon = 1e-8;

    // Different model sizes (number of Gaussians * parameters per Gaussian)
    for &n_gaussians in &[1000, 10000, 100000] {
        // Each Gaussian has: 3 pos + 4 rot + 3 scale + 1 opacity + 3 offset = 14 params
        // Plus SH coefficients (16 for degree 1)
        let params_per_gaussian = 14 + 16;
        let n_params = n_gaussians * params_per_gaussian;

        let params: Vec<f32> = (0..n_params).map(|i| i as f32 * 0.001).collect();
        let grads: Vec<f32> = (0..n_params).map(|i| (i as f32 * 0.0001).sin()).collect();
        let m = vec![0.0_f32; n_params];
        let v = vec![0.0_f32; n_params];

        group.throughput(Throughput::Elements(n_params as u64));
        group.bench_with_input(
            BenchmarkId::new("gaussians", n_gaussians),
            &n_gaussians,
            |b, _| {
                b.iter(|| {
                    let mut p = params.clone();
                    let mut m_copy = m.clone();
                    let mut v_copy = v.clone();
                    adam_step_inline(
                        &mut p,
                        black_box(&grads),
                        &mut m_copy,
                        &mut v_copy,
                        100, // t
                        lr,
                        beta1,
                        beta2,
                        epsilon,
                    );
                    black_box((p, m_copy, v_copy))
                })
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Gradient Operations Benchmark
// ---------------------------------------------------------------------------

fn bench_gradient_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("gradient_operations");

    for &n in &[10000, 100000, 1000000] {
        let grads: Vec<f32> = (0..n).map(|i| (i as f32 * 0.0001).sin()).collect();

        // Gradient statistics
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("statistics", n), &n, |b, _| {
            b.iter(|| black_box(gradient_statistics(black_box(&grads))))
        });

        // Gradient penalty
        group.bench_with_input(BenchmarkId::new("penalty", n), &n, |b, _| {
            b.iter(|| black_box(gradient_penalty(black_box(&grads), 1.0)))
        });

        // Clip by norm
        group.bench_with_input(BenchmarkId::new("clip_norm", n), &n, |b, _| {
            b.iter(|| {
                let mut g = grads.clone();
                black_box(clip_gradients_by_norm(&mut g, 1.0))
            })
        });

        // Clip by value
        group.bench_with_input(BenchmarkId::new("clip_value", n), &n, |b, _| {
            b.iter(|| {
                let mut g = grads.clone();
                black_box(clip_gradients_by_value(&mut g, 0.5))
            })
        });

        // Sanitize (NaN/Inf removal)
        let mut grads_with_nan = grads.clone();
        for i in (0..n).step_by(1000) {
            grads_with_nan[i] = f32::NAN;
        }
        group.bench_with_input(BenchmarkId::new("sanitize", n), &n, |b, _| {
            b.iter(|| {
                let mut g = grads_with_nan.clone();
                black_box(sanitize_gradients(&mut g))
            })
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Learning Rate Schedule Benchmark
// ---------------------------------------------------------------------------

fn bench_lr_schedule(c: &mut Criterion) {
    let mut group = c.benchmark_group("lr_schedule");

    let lr_start = 1.6e-4_f32;
    let lr_end = 1.6e-6_f32;
    let decay_steps = 30000_u32;

    // Exponential decay
    group.bench_function("exponential_decay", |b| {
        b.iter(|| {
            let iteration = black_box(15000_u32);
            let t = (iteration as f32) / (decay_steps as f32);
            let t = t.min(1.0);
            let log_start = lr_start.ln();
            let log_end = lr_end.ln();
            black_box(((1.0 - t) * log_start + t * log_end).exp())
        })
    });

    // Batch of LR computations (for tracking/logging)
    for &n in &[100, 1000, 10000] {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("batch", n), &n, |b, _| {
            b.iter(|| {
                let lrs: Vec<f32> = (0..n)
                    .map(|i| {
                        let t = (i as f32) / (decay_steps as f32);
                        let t = t.min(1.0);
                        let log_start = lr_start.ln();
                        let log_end = lr_end.ln();
                        ((1.0 - t) * log_start + t * log_end).exp()
                    })
                    .collect();
                black_box(lrs)
            })
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Downsampling Benchmark (for MS-SSIM)
// ---------------------------------------------------------------------------

fn bench_downsample(c: &mut Criterion) {
    let mut group = c.benchmark_group("downsample_2x");

    for &resolution in &[128, 256, 512] {
        let size = resolution * resolution * 3;
        let image: Vec<f32> = (0..size).map(|i| (i as f32 * 0.001).sin()).collect();

        group.throughput(Throughput::Elements((resolution * resolution / 4) as u64));
        group.bench_with_input(
            BenchmarkId::new("resolution", resolution),
            &resolution,
            |b, _| {
                b.iter(|| {
                    let width = resolution;
                    let height = resolution;
                    let new_w = width / 2;
                    let new_h = height / 2;
                    let channels = 3;

                    let mut result = vec![0.0_f32; new_w * new_h * channels];

                    for y in 0..new_h {
                        for x in 0..new_w {
                            for c in 0..channels {
                                let mut sum = 0.0_f32;
                                for dy in 0..2 {
                                    for dx in 0..2 {
                                        let src_y = y * 2 + dy;
                                        let src_x = x * 2 + dx;
                                        let idx = (src_y * width + src_x) * channels + c;
                                        sum += image[idx];
                                    }
                                }
                                let dst_idx = (y * new_w + x) * channels + c;
                                result[dst_idx] = sum / 4.0;
                            }
                        }
                    }

                    black_box(result)
                })
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Position Regularization Benchmark
// ---------------------------------------------------------------------------

fn bench_position_reg(c: &mut Criterion) {
    let mut group = c.benchmark_group("position_regularization");

    for &n_gaussians in &[1000, 10000, 100000] {
        // Local offsets: 3 floats per Gaussian
        let local_offsets: Vec<[f32; 3]> = (0..n_gaussians)
            .map(|i| {
                [
                    (i as f32 * 0.001).sin() * 0.01,
                    (i as f32 * 0.002).cos() * 0.01,
                    (i as f32 * 0.003).sin() * 0.01,
                ]
            })
            .collect();

        group.throughput(Throughput::Elements(n_gaussians as u64));
        group.bench_with_input(
            BenchmarkId::new("gaussians", n_gaussians),
            &n_gaussians,
            |b, _| {
                b.iter(|| {
                    let sum: f32 = local_offsets
                        .iter()
                        .map(|o| o[0] * o[0] + o[1] * o[1] + o[2] * o[2])
                        .sum();
                    black_box(sum / n_gaussians as f32)
                })
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Opacity Regularization (Binary Entropy) Benchmark
// ---------------------------------------------------------------------------

fn bench_opacity_reg(c: &mut Criterion) {
    let mut group = c.benchmark_group("opacity_regularization");

    for &n_gaussians in &[1000, 10000, 100000] {
        // Opacity values (logit space)
        let opacities: Vec<f32> = (0..n_gaussians).map(|i| (i as f32 * 0.01).sin()).collect();

        group.throughput(Throughput::Elements(n_gaussians as u64));
        group.bench_with_input(
            BenchmarkId::new("gaussians", n_gaussians),
            &n_gaussians,
            |b, _| {
                b.iter(|| {
                    let sum: f32 = opacities
                        .iter()
                        .map(|&o| {
                            let s = 1.0 / (1.0 + (-o).exp());
                            let s = s.clamp(1e-6, 1.0 - 1e-6);
                            -(s * s.ln() + (1.0 - s) * (1.0 - s).ln())
                        })
                        .sum();
                    black_box(sum / n_gaussians as f32)
                })
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Criterion Groups
// ---------------------------------------------------------------------------

criterion_group!(
    benches,
    bench_l1_loss,
    bench_ssim_loss,
    bench_ms_ssim_loss,
    bench_combined_loss,
    bench_gaussian_kernel,
    bench_separable_convolution,
    bench_adam_step,
    bench_gradient_operations,
    bench_lr_schedule,
    bench_downsample,
    bench_position_reg,
    bench_opacity_reg,
);

criterion_main!(benches);
