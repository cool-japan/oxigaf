//! Comprehensive benchmarks for oxigaf-diffusion.
//!
//! Benchmarks:
//! - Cross-attention forward pass
//! - DDIM scheduler step
//! - Flash attention comparison
//! - Different input sizes
//!
//! Run with: cargo bench -p oxigaf-diffusion

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;

use candle_core::{DType, Device, Tensor};

use oxigaf_diffusion::{
    flash_attention, flash_attention_with_config, DdimScheduler, FlashAttention,
    FlashAttentionConfig, PredictionType,
};

// ---------------------------------------------------------------------------
// Helper Functions
// ---------------------------------------------------------------------------

/// Create random query, key, value tensors for attention benchmarks.
fn create_attention_tensors(
    batch: usize,
    heads: usize,
    seq_q: usize,
    seq_k: usize,
    dim_head: usize,
    device: &Device,
) -> Result<(Tensor, Tensor, Tensor), candle_core::Error> {
    let q_size = batch * heads * seq_q * dim_head;
    let k_size = batch * heads * seq_k * dim_head;

    let q_data: Vec<f32> = (0..q_size).map(|i| (i as f32 * 0.01).sin()).collect();
    let k_data: Vec<f32> = (0..k_size).map(|i| (i as f32 * 0.02).cos()).collect();
    let v_data: Vec<f32> = (0..k_size).map(|i| (i as f32 * 0.03).sin()).collect();

    let q = Tensor::from_vec(q_data, (batch, heads, seq_q, dim_head), device)?;
    let k = Tensor::from_vec(k_data, (batch, heads, seq_k, dim_head), device)?;
    let v = Tensor::from_vec(v_data, (batch, heads, seq_k, dim_head), device)?;

    Ok((q, k, v))
}

/// Standard attention implementation for comparison.
fn standard_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f64,
) -> Result<Tensor, candle_core::Error> {
    let q = q.to_dtype(DType::F32)?;
    let k = k.to_dtype(DType::F32)?;
    let v = v.to_dtype(DType::F32)?;

    let k_t = k.transpose(candle_core::D::Minus2, candle_core::D::Minus1)?;
    let attn = (q.matmul(&k_t)? * scale)?;
    let attn = candle_nn::ops::softmax_last_dim(&attn)?;
    attn.matmul(&v)
}

// ---------------------------------------------------------------------------
// Attention Forward Benchmark
// ---------------------------------------------------------------------------

fn bench_attention_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("attention_forward");
    let device = Device::Cpu;

    let batch = 1;
    let heads = 8;
    let dim_head = 64;

    // Different sequence lengths
    for &seq_len in &[64, 128, 256, 512] {
        let (q, k, v) =
            match create_attention_tensors(batch, heads, seq_len, seq_len, dim_head, &device) {
                Ok(tensors) => tensors,
                Err(e) => {
                    eprintln!("Failed to create tensors: {e}");
                    continue;
                }
            };
        let scale = 1.0 / (dim_head as f64).sqrt();

        group.throughput(Throughput::Elements((seq_len * seq_len) as u64));
        group.bench_with_input(
            BenchmarkId::new("standard_seq", seq_len),
            &seq_len,
            |b, _| {
                b.iter(|| {
                    black_box(standard_attention(
                        black_box(&q),
                        black_box(&k),
                        black_box(&v),
                        scale,
                    ))
                })
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Flash Attention Benchmark
// ---------------------------------------------------------------------------

fn bench_flash_attention(c: &mut Criterion) {
    let mut group = c.benchmark_group("flash_attention");
    let device = Device::Cpu;

    let batch = 1;
    let heads = 8;
    let dim_head = 64;

    // Different sequence lengths
    for &seq_len in &[64, 128, 256, 512] {
        let (q, k, v) =
            match create_attention_tensors(batch, heads, seq_len, seq_len, dim_head, &device) {
                Ok(tensors) => tensors,
                Err(e) => {
                    eprintln!("Failed to create tensors: {e}");
                    continue;
                }
            };

        group.throughput(Throughput::Elements((seq_len * seq_len) as u64));
        group.bench_with_input(BenchmarkId::new("flash_seq", seq_len), &seq_len, |b, _| {
            b.iter(|| {
                black_box(flash_attention(
                    black_box(&q),
                    black_box(&k),
                    black_box(&v),
                    dim_head,
                ))
            })
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Flash Attention Block Size Comparison
// ---------------------------------------------------------------------------

fn bench_flash_attention_block_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("flash_attention_block_size");
    let device = Device::Cpu;

    let batch = 1;
    let heads = 4;
    let seq_len = 256;
    let dim_head = 64;

    let (q, k, v) =
        match create_attention_tensors(batch, heads, seq_len, seq_len, dim_head, &device) {
            Ok(tensors) => tensors,
            Err(e) => {
                eprintln!("Failed to create tensors: {e}");
                return;
            }
        };

    // Different block sizes
    for &block_size in &[32, 64, 128] {
        let config = FlashAttentionConfig::with_block_size(block_size);

        group.bench_with_input(
            BenchmarkId::new("block_size", block_size),
            &block_size,
            |b, _| {
                b.iter(|| {
                    black_box(flash_attention_with_config(
                        black_box(&q),
                        black_box(&k),
                        black_box(&v),
                        dim_head,
                        config,
                    ))
                })
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Flash vs Standard Comparison
// ---------------------------------------------------------------------------

fn bench_attention_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("attention_comparison");
    let device = Device::Cpu;

    let batch = 1;
    let heads = 8;
    let seq_len = 256;
    let dim_head = 64;

    let (q, k, v) =
        match create_attention_tensors(batch, heads, seq_len, seq_len, dim_head, &device) {
            Ok(tensors) => tensors,
            Err(e) => {
                eprintln!("Failed to create tensors: {e}");
                return;
            }
        };
    let scale = 1.0 / (dim_head as f64).sqrt();

    group.bench_function("standard", |b| {
        b.iter(|| {
            black_box(standard_attention(
                black_box(&q),
                black_box(&k),
                black_box(&v),
                scale,
            ))
        })
    });

    let flash = FlashAttention::with_dim_head(dim_head);
    group.bench_function("flash", |b| {
        b.iter(|| black_box(flash.forward(black_box(&q), black_box(&k), black_box(&v))))
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// DDIM Scheduler Benchmark
// ---------------------------------------------------------------------------

fn bench_scheduler_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("ddim_scheduler");
    let device = Device::Cpu;

    // Standard latent size for SD: 4x64x64
    let batch = 1;
    let channels = 4;
    let height = 64;
    let width = 64;
    let latent_size = batch * channels * height * width;

    // Create scheduler
    let mut scheduler = DdimScheduler::new(1000, PredictionType::VPrediction);
    scheduler.set_timesteps(50);

    let sample_data: Vec<f32> = (0..latent_size).map(|i| (i as f32 * 0.001).sin()).collect();
    let model_output_data: Vec<f32> = (0..latent_size).map(|i| (i as f32 * 0.002).cos()).collect();

    let sample = match Tensor::from_vec(sample_data, (batch, channels, height, width), &device) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Failed to create sample tensor: {e}");
            return;
        }
    };
    let model_output =
        match Tensor::from_vec(model_output_data, (batch, channels, height, width), &device) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("Failed to create model_output tensor: {e}");
                return;
            }
        };

    // Benchmark single step
    group.throughput(Throughput::Elements(latent_size as u64));
    group.bench_function("single_step", |b| {
        let timestep = 500;
        b.iter(|| {
            black_box(scheduler.step(
                black_box(&model_output),
                black_box(timestep),
                black_box(&sample),
            ))
        })
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// DDIM Full Loop Benchmark
// ---------------------------------------------------------------------------

fn bench_scheduler_full_loop(c: &mut Criterion) {
    let mut group = c.benchmark_group("ddim_full_loop");
    let device = Device::Cpu;

    // Smaller latent for full loop benchmark
    let batch = 1;
    let channels = 4;
    let height = 32;
    let width = 32;
    let latent_size = batch * channels * height * width;

    // Different number of steps
    for &num_steps in &[10, 25, 50] {
        let mut scheduler = DdimScheduler::new(1000, PredictionType::VPrediction);
        scheduler.set_timesteps(num_steps);

        let sample_data: Vec<f32> = (0..latent_size).map(|i| (i as f32 * 0.001).sin()).collect();

        let sample = match Tensor::from_vec(sample_data, (batch, channels, height, width), &device)
        {
            Ok(t) => t,
            Err(e) => {
                eprintln!("Failed to create sample tensor: {e}");
                continue;
            }
        };

        group.throughput(Throughput::Elements(num_steps as u64));
        group.bench_with_input(BenchmarkId::new("steps", num_steps), &num_steps, |b, _| {
            b.iter(|| {
                let mut latent = sample.clone();
                for &t in scheduler.timesteps() {
                    // Simulate model output (in real code this would be U-Net)
                    let model_output = &latent * 0.1;
                    match model_output {
                        Ok(mo) => match scheduler.step(&mo, t, &latent) {
                            Ok(new_latent) => latent = new_latent,
                            Err(_) => break,
                        },
                        Err(_) => break,
                    }
                }
                black_box(latent)
            })
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Add Noise Benchmark
// ---------------------------------------------------------------------------

fn bench_add_noise(c: &mut Criterion) {
    let mut group = c.benchmark_group("add_noise");
    let device = Device::Cpu;

    let batch = 1;
    let channels = 4;
    let height = 64;
    let width = 64;
    let latent_size = batch * channels * height * width;

    let scheduler = DdimScheduler::new(1000, PredictionType::VPrediction);

    let original_data: Vec<f32> = (0..latent_size).map(|i| (i as f32 * 0.001).sin()).collect();
    let noise_data: Vec<f32> = (0..latent_size).map(|i| (i as f32 * 0.002).cos()).collect();

    let original = match Tensor::from_vec(original_data, (batch, channels, height, width), &device)
    {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Failed to create original tensor: {e}");
            return;
        }
    };
    let noise = match Tensor::from_vec(noise_data, (batch, channels, height, width), &device) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Failed to create noise tensor: {e}");
            return;
        }
    };

    // Different timesteps
    for &timestep in &[100, 500, 900] {
        group.throughput(Throughput::Elements(latent_size as u64));
        group.bench_with_input(
            BenchmarkId::new("timestep", timestep),
            &timestep,
            |b, &t| {
                b.iter(|| {
                    black_box(scheduler.add_noise(black_box(&original), black_box(&noise), t))
                })
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Multi-head Attention Scaling
// ---------------------------------------------------------------------------

fn bench_attention_heads_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("attention_heads_scaling");
    let device = Device::Cpu;

    let batch = 1;
    let seq_len = 128;
    let dim_head = 64;

    // Different number of heads
    for &heads in &[1, 4, 8, 16] {
        let (q, k, v) =
            match create_attention_tensors(batch, heads, seq_len, seq_len, dim_head, &device) {
                Ok(tensors) => tensors,
                Err(e) => {
                    eprintln!("Failed to create tensors: {e}");
                    continue;
                }
            };

        group.throughput(Throughput::Elements((heads * seq_len * seq_len) as u64));
        group.bench_with_input(BenchmarkId::new("heads", heads), &heads, |b, _| {
            b.iter(|| {
                black_box(flash_attention(
                    black_box(&q),
                    black_box(&k),
                    black_box(&v),
                    dim_head,
                ))
            })
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Criterion Groups
// ---------------------------------------------------------------------------

criterion_group!(
    benches,
    bench_attention_forward,
    bench_flash_attention,
    bench_flash_attention_block_sizes,
    bench_attention_comparison,
    bench_scheduler_step,
    bench_scheduler_full_loop,
    bench_add_noise,
    bench_attention_heads_scaling,
);

criterion_main!(benches);
