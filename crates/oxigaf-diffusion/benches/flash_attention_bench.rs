//! Benchmarks for Flash Attention vs Standard Attention.
//!
//! This benchmark compares the performance of flash attention against standard
//! attention for various sequence lengths and configurations.
//!
//! Run with:
//! ```bash
//! cargo bench -p oxigaf-diffusion --features flash_attention
//! ```

use candle_core::{DType, Device, Result, Tensor, D};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;

#[cfg(feature = "flash_attention")]
use oxigaf_diffusion::{FlashAttention, FlashAttentionConfig};

/// Create test tensors for benchmarking.
fn create_test_tensors(
    batch: usize,
    heads: usize,
    seq_len: usize,
    dim_head: usize,
    device: &Device,
) -> Result<(Tensor, Tensor, Tensor)> {
    // Use deterministic data for reproducibility
    let q_size = batch * heads * seq_len * dim_head;
    let q_data: Vec<f32> = (0..q_size).map(|i| (i as f32 * 0.01).sin()).collect();
    let k_data: Vec<f32> = (0..q_size).map(|i| (i as f32 * 0.02).cos()).collect();
    let v_data: Vec<f32> = (0..q_size).map(|i| (i as f32 * 0.03).sin()).collect();

    let q = Tensor::from_vec(q_data, (batch, heads, seq_len, dim_head), device)?;
    let k = Tensor::from_vec(k_data, (batch, heads, seq_len, dim_head), device)?;
    let v = Tensor::from_vec(v_data, (batch, heads, seq_len, dim_head), device)?;

    Ok((q, k, v))
}

/// Standard attention implementation for comparison.
fn standard_attention(q: &Tensor, k: &Tensor, v: &Tensor, scale: f64) -> Result<Tensor> {
    let q = q.to_dtype(DType::F32)?;
    let k = k.to_dtype(DType::F32)?;
    let v = v.to_dtype(DType::F32)?;

    let k_t = k.transpose(D::Minus2, D::Minus1)?.contiguous()?;
    let attn = (q.matmul(&k_t)? * scale)?;
    let attn = candle_nn::ops::softmax_last_dim(&attn)?;
    attn.matmul(&v)
}

/// Benchmark standard attention for various sequence lengths.
fn bench_standard_attention(c: &mut Criterion) {
    let device = Device::Cpu;
    let batch = 1;
    let heads = 8;
    let dim_head = 64;
    let scale = 1.0 / (dim_head as f64).sqrt();

    let mut group = c.benchmark_group("standard_attention");

    for seq_len in [64, 128, 256, 512] {
        let (q, k, v) = create_test_tensors(batch, heads, seq_len, dim_head, &device)
            .expect("Failed to create test tensors");

        let num_elements = (batch * heads * seq_len * dim_head) as u64;
        group.throughput(Throughput::Elements(num_elements));

        group.bench_with_input(BenchmarkId::new("seq_len", seq_len), &seq_len, |b, _| {
            b.iter(|| {
                standard_attention(black_box(&q), black_box(&k), black_box(&v), scale)
                    .expect("Standard attention failed")
            });
        });
    }

    group.finish();
}

/// Benchmark flash attention for various sequence lengths.
#[cfg(feature = "flash_attention")]
fn bench_flash_attention(c: &mut Criterion) {
    let device = Device::Cpu;
    let batch = 1;
    let heads = 8;
    let dim_head = 64;

    let config = FlashAttentionConfig::with_block_size(64);
    let flash = FlashAttention::new(dim_head, config);

    let mut group = c.benchmark_group("flash_attention");

    for seq_len in [64, 128, 256, 512] {
        let (q, k, v) = create_test_tensors(batch, heads, seq_len, dim_head, &device)
            .expect("Failed to create test tensors");

        let num_elements = (batch * heads * seq_len * dim_head) as u64;
        group.throughput(Throughput::Elements(num_elements));

        group.bench_with_input(BenchmarkId::new("seq_len", seq_len), &seq_len, |b, _| {
            b.iter(|| {
                flash
                    .forward(black_box(&q), black_box(&k), black_box(&v))
                    .expect("Flash attention failed")
            });
        });
    }

    group.finish();
}

/// Benchmark comparison of standard vs flash attention.
#[cfg(feature = "flash_attention")]
fn bench_attention_comparison(c: &mut Criterion) {
    let device = Device::Cpu;
    let batch = 1;
    let heads = 4;
    let dim_head = 64;
    let scale = 1.0 / (dim_head as f64).sqrt();

    let config = FlashAttentionConfig::with_block_size(64);
    let flash = FlashAttention::new(dim_head, config);

    let mut group = c.benchmark_group("attention_comparison");

    for seq_len in [128, 256, 512] {
        let (q, k, v) = create_test_tensors(batch, heads, seq_len, dim_head, &device)
            .expect("Failed to create test tensors");

        group.bench_with_input(BenchmarkId::new("standard", seq_len), &seq_len, |b, _| {
            b.iter(|| {
                standard_attention(black_box(&q), black_box(&k), black_box(&v), scale)
                    .expect("Standard attention failed")
            });
        });

        group.bench_with_input(BenchmarkId::new("flash", seq_len), &seq_len, |b, _| {
            b.iter(|| {
                flash
                    .forward(black_box(&q), black_box(&k), black_box(&v))
                    .expect("Flash attention failed")
            });
        });
    }

    group.finish();
}

/// Benchmark different block sizes for flash attention.
#[cfg(feature = "flash_attention")]
fn bench_flash_block_sizes(c: &mut Criterion) {
    let device = Device::Cpu;
    let batch = 1;
    let heads = 4;
    let dim_head = 64;
    let seq_len = 256;

    let (q, k, v) = create_test_tensors(batch, heads, seq_len, dim_head, &device)
        .expect("Failed to create test tensors");

    let mut group = c.benchmark_group("flash_block_sizes");

    for block_size in [16, 32, 64, 128] {
        let config = FlashAttentionConfig::with_block_size(block_size);
        let flash = FlashAttention::new(dim_head, config);

        group.bench_with_input(
            BenchmarkId::new("block_size", block_size),
            &block_size,
            |b, _| {
                b.iter(|| {
                    flash
                        .forward(black_box(&q), black_box(&k), black_box(&v))
                        .expect("Flash attention failed")
                });
            },
        );
    }

    group.finish();
}

/// Benchmark with different batch sizes.
#[cfg(feature = "flash_attention")]
fn bench_batch_sizes(c: &mut Criterion) {
    let device = Device::Cpu;
    let heads = 4;
    let dim_head = 64;
    let seq_len = 128;

    let config = FlashAttentionConfig::with_block_size(64);
    let flash = FlashAttention::new(dim_head, config);

    let mut group = c.benchmark_group("batch_sizes");

    for batch in [1, 2, 4, 8] {
        let (q, k, v) = create_test_tensors(batch, heads, seq_len, dim_head, &device)
            .expect("Failed to create test tensors");

        let num_elements = (batch * heads * seq_len * dim_head) as u64;
        group.throughput(Throughput::Elements(num_elements));

        group.bench_with_input(BenchmarkId::new("batch", batch), &batch, |b, _| {
            b.iter(|| {
                flash
                    .forward(black_box(&q), black_box(&k), black_box(&v))
                    .expect("Flash attention failed")
            });
        });
    }

    group.finish();
}

// Configure criterion groups based on feature flags
#[cfg(feature = "flash_attention")]
criterion_group!(
    benches,
    bench_standard_attention,
    bench_flash_attention,
    bench_attention_comparison,
    bench_flash_block_sizes,
    bench_batch_sizes,
);

#[cfg(not(feature = "flash_attention"))]
criterion_group!(benches, bench_standard_attention,);

criterion_main!(benches);
