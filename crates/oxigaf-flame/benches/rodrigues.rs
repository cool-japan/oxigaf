//! Benchmarks for Rodrigues rotation formula.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use oxigaf_flame::model::rodrigues;
use std::f32::consts::PI;
use std::hint::black_box;

fn bench_rodrigues_various_angles(c: &mut Criterion) {
    let mut group = c.benchmark_group("rodrigues");

    let test_cases = vec![
        ("zero", (0.0, 0.0, 0.0)),
        ("small", (0.001, 0.002, 0.001)),
        ("medium", (0.5, 0.3, -0.2)),
        ("large", (1.0, 1.5, -0.8)),
        ("180deg_x", (PI, 0.0, 0.0)),
        ("180deg_y", (0.0, PI, 0.0)),
        ("180deg_z", (0.0, 0.0, PI)),
        ("combined", (0.3, -0.5, 0.2)),
    ];

    for (name, (rx, ry, rz)) in test_cases {
        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &(rx, ry, rz),
            |b, &(rx, ry, rz)| {
                b.iter(|| {
                    let mat = rodrigues(black_box(rx), black_box(ry), black_box(rz));
                    black_box(mat);
                });
            },
        );
    }

    group.finish();
}

fn bench_rodrigues_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("rodrigues_batch");

    // Simulate batch processing of rotation vectors
    for batch_size in [1, 5, 10, 100] {
        let rotations: Vec<(f32, f32, f32)> = (0..batch_size)
            .map(|i| {
                let t = i as f32 * 0.1;
                (t.sin(), t.cos(), (t * 2.0).sin())
            })
            .collect();

        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            &batch_size,
            |b, _| {
                b.iter(|| {
                    for &(rx, ry, rz) in &rotations {
                        let mat = rodrigues(black_box(rx), black_box(ry), black_box(rz));
                        black_box(mat);
                    }
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_rodrigues_various_angles,
    bench_rodrigues_batch
);
criterion_main!(benches);
