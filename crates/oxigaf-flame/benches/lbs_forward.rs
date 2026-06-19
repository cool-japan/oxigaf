//! Benchmarks for FLAME LBS forward pass.
//!
//! Run with: `cargo bench -p oxigaf-flame --bench lbs_forward`
//!
//! For SIMD benchmarks: `cargo bench -p oxigaf-flame --bench lbs_forward --features simd`
//!
//! For parallel benchmarks: `cargo bench -p oxigaf-flame --bench lbs_forward --features parallel`

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use oxigaf_flame::{FlameParams, Mesh};
use rayon::prelude::*;
use std::hint::black_box;

/// Create a mock FLAME model for benchmarking.
fn create_mock_flame_model(n_vertices: usize) -> oxigaf_flame::FlameModel {
    use ndarray::{Array2, Array3};

    let n_joints = 5;
    let n_shape = 100;
    let n_expr = 50;

    let v_template = Array2::from_shape_fn((n_vertices, 3), |(i, j)| (i + j) as f32 * 0.001);
    let faces = vec![[0u32, 1, 2]; n_vertices / 3];
    let shapedirs = Array3::from_shape_fn((n_vertices, 3, n_shape), |(i, j, k)| {
        ((i + j + k) as f32 * 0.0001).sin()
    });
    let expressiondirs = Array3::from_shape_fn((n_vertices, 3, n_expr), |(i, j, k)| {
        ((i + j + k) as f32 * 0.0001).cos()
    });
    let posedirs = Array3::from_shape_fn((n_vertices, 3, (n_joints - 1) * 9), |(i, j, k)| {
        ((i + j + k) as f32 * 0.00001).sin()
    });
    let j_regressor = Array2::from_shape_fn((n_joints, n_vertices), |(i, j)| {
        if j % n_joints == i {
            1.0 / (n_vertices / n_joints) as f32
        } else {
            0.0
        }
    });
    let parents = vec![-1i32, 0, 1, 2, 3];
    let lbs_weights = Array2::from_shape_fn((n_vertices, n_joints), |(i, j)| {
        if i % n_joints == j {
            0.8
        } else {
            0.05
        }
    });

    oxigaf_flame::FlameModel::from_arrays(
        v_template,
        faces,
        shapedirs,
        expressiondirs,
        posedirs,
        j_regressor,
        parents,
        lbs_weights,
        n_joints,
    )
}

/// Create a batch of random parameters for benchmarking.
fn create_params_batch(count: usize) -> Vec<FlameParams> {
    (0..count)
        .map(|i| FlameParams {
            shape: vec![(i as f32 * 0.1).sin() * 0.5; 10],
            expression: vec![(i as f32 * 0.2).cos() * 0.5; 10],
            pose: vec![(i as f32 * 0.05).sin() * 0.1; 15],
            translation: [
                (i as f32 * 0.01).sin(),
                (i as f32 * 0.02).cos(),
                (i as f32 * 0.03).sin(),
            ],
        })
        .collect()
}

fn bench_lbs_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("lbs_forward");

    // Test with different model sizes
    for n_verts in [1000, 5023, 10000] {
        let model = create_mock_flame_model(n_verts);
        let params = FlameParams::neutral();

        group.bench_with_input(BenchmarkId::from_parameter(n_verts), &n_verts, |b, _| {
            b.iter(|| {
                let mesh = model.forward(black_box(&params));
                black_box(mesh);
            });
        });
    }

    group.finish();
}

fn bench_lbs_forward_with_params(c: &mut Criterion) {
    let mut group = c.benchmark_group("lbs_forward_with_params");

    let model = create_mock_flame_model(5023); // Standard FLAME size

    // Different parameter configurations
    let neutral = FlameParams::neutral();

    let with_shape = FlameParams {
        shape: vec![0.5, -0.3, 0.2, 0.1, -0.15],
        ..FlameParams::neutral()
    };

    let with_expression = FlameParams {
        expression: vec![0.8, 0.5, -0.4, 0.3],
        ..FlameParams::neutral()
    };

    let with_pose = FlameParams {
        pose: vec![0.1; 15],
        ..FlameParams::neutral()
    };

    group.bench_function("neutral", |b| {
        b.iter(|| {
            let mesh = model.forward(black_box(&neutral));
            black_box(mesh);
        });
    });

    group.bench_function("with_shape", |b| {
        b.iter(|| {
            let mesh = model.forward(black_box(&with_shape));
            black_box(mesh);
        });
    });

    group.bench_function("with_expression", |b| {
        b.iter(|| {
            let mesh = model.forward(black_box(&with_expression));
            black_box(mesh);
        });
    });

    group.bench_function("with_pose", |b| {
        b.iter(|| {
            let mesh = model.forward(black_box(&with_pose));
            black_box(mesh);
        });
    });

    group.finish();
}

fn bench_lbs_forward_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("lbs_forward_batch");

    let model = create_mock_flame_model(5023); // Standard FLAME size

    // Batch sizes to test
    for batch_size in [10, 50, 100] {
        let params_batch = create_params_batch(batch_size);

        group.bench_with_input(
            BenchmarkId::new("sequential", batch_size),
            &params_batch,
            |b, params| {
                b.iter(|| {
                    let meshes = model.forward_batch(black_box(params));
                    black_box(meshes);
                });
            },
        );

        // Parallel benchmark using rayon directly (always available in dev-deps)
        group.bench_with_input(
            BenchmarkId::new("parallel_rayon", batch_size),
            &params_batch,
            |b, params| {
                b.iter(|| {
                    let meshes: Vec<Mesh> = params.par_iter().map(|p| model.forward(p)).collect();
                    black_box(meshes);
                });
            },
        );
    }

    group.finish();
}

#[cfg(all(feature = "simd", nightly))]
fn bench_lbs_forward_simd(c: &mut Criterion) {
    let mut group = c.benchmark_group("lbs_forward_simd");

    let model = create_mock_flame_model(5023); // Standard FLAME size
    let params = FlameParams::neutral();

    group.bench_function("scalar", |b| {
        b.iter(|| {
            let mesh = model.forward(black_box(&params));
            black_box(mesh);
        });
    });

    group.bench_function("simd", |b| {
        b.iter(|| {
            let mesh = model.forward_simd(black_box(&params));
            black_box(mesh);
        });
    });

    group.finish();
}

#[cfg(all(feature = "simd", nightly))]
fn bench_lbs_forward_batch_simd(c: &mut Criterion) {
    let mut group = c.benchmark_group("lbs_forward_batch_simd");

    let model = create_mock_flame_model(5023);

    for batch_size in [10, 50, 100] {
        let params_batch = create_params_batch(batch_size);

        group.bench_with_input(
            BenchmarkId::new("scalar_seq", batch_size),
            &params_batch,
            |b, params| {
                b.iter(|| {
                    let meshes = model.forward_batch(black_box(params));
                    black_box(meshes);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("simd_seq", batch_size),
            &params_batch,
            |b, params| {
                b.iter(|| {
                    let meshes = model.forward_batch_simd(black_box(params));
                    black_box(meshes);
                });
            },
        );

        // SIMD + parallel
        group.bench_with_input(
            BenchmarkId::new("simd_par", batch_size),
            &params_batch,
            |b, params| {
                b.iter(|| {
                    let meshes: Vec<Mesh> =
                        params.par_iter().map(|p| model.forward_simd(p)).collect();
                    black_box(meshes);
                });
            },
        );
    }

    group.finish();
}

fn bench_blend_shapes(c: &mut Criterion) {
    use ndarray::{Array2, Array3};

    let mut group = c.benchmark_group("blend_shapes");

    // Test with different coefficient counts
    for n_coeffs in [10, 50, 100] {
        let n_vertices = 5023;
        let v = Array2::from_shape_fn((n_vertices, 3), |(i, j)| (i + j) as f32 * 0.001);
        let dirs = Array3::from_shape_fn((n_vertices, 3, n_coeffs), |(i, j, k)| {
            (i + j + k) as f32 * 0.0001
        });
        let coeffs: Vec<f32> = (0..n_coeffs).map(|i| (i as f32 * 0.1).sin()).collect();

        group.bench_with_input(
            BenchmarkId::new("scalar", n_coeffs),
            &(&dirs, &coeffs),
            |b, (dirs, coeffs)| {
                b.iter(|| {
                    let mut v_clone = v.clone();
                    // Call through the model's internal function via forward
                    for (i, &coeff) in coeffs.iter().enumerate() {
                        if coeff.abs() > 1e-12 {
                            let dir_slice = dirs.slice(ndarray::s![.., .., i]);
                            v_clone.scaled_add(coeff, &dir_slice);
                        }
                    }
                    black_box(v_clone);
                });
            },
        );

        #[cfg(all(feature = "simd", nightly))]
        {
            group.bench_with_input(
                BenchmarkId::new("simd", n_coeffs),
                &(&dirs, &coeffs),
                |b, (dirs, coeffs)| {
                    b.iter(|| {
                        let mut v_clone = v.clone();
                        oxigaf_flame::simd::apply_blend_shapes_simd(&mut v_clone, dirs, coeffs);
                        black_box(v_clone);
                    });
                },
            );
        }
    }

    group.finish();
}

#[cfg(not(all(feature = "simd", nightly)))]
criterion_group!(
    benches,
    bench_lbs_forward,
    bench_lbs_forward_with_params,
    bench_lbs_forward_batch,
    bench_blend_shapes,
);

#[cfg(all(feature = "simd", nightly))]
criterion_group!(
    benches,
    bench_lbs_forward,
    bench_lbs_forward_with_params,
    bench_lbs_forward_batch,
    bench_blend_shapes,
    bench_lbs_forward_simd,
    bench_lbs_forward_batch_simd,
);

criterion_main!(benches);
