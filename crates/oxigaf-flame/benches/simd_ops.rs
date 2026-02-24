//! Benchmarks for SIMD-accelerated operations.
//!
//! Run with: `cargo +nightly bench -p oxigaf-flame --bench simd_ops --features simd`
//!
//! These benchmarks compare scalar vs SIMD implementations of core operations:
//! - Rodrigues rotation
//! - Matrix-vector multiplication
//! - Blend shape application
//! - Vertex transformation
//!
//! **Note**: Requires nightly Rust. On stable Rust the benchmark is a no-op.

// On stable Rust (or without nightly), provide a dummy main and skip compilation
// of the SIMD-dependent benchmark code entirely.
#[cfg(not(all(feature = "simd", nightly)))]
fn main() {
    eprintln!("SIMD benchmarks require nightly Rust. Skipping.");
}

#[cfg(all(feature = "simd", nightly))]
mod bench_impl {
    use criterion::{criterion_group, BenchmarkId, Criterion, Throughput};
    use nalgebra as na;
    use ndarray::{Array2, Array3};
    use oxigaf_flame::simd::{
        apply_blend_shapes_simd, mat4_mul_simd, mat4_vec4_mul_simd, rodrigues_batch,
        rodrigues_simd, weighted_matrix_sum_simd, VerticesSoA,
    };
    use std::hint::black_box;

    fn bench_rodrigues_simd(c: &mut Criterion) {
        let mut group = c.benchmark_group("rodrigues");

        // Single rotation
        let angles = (0.1, 0.2, 0.3);

        group.bench_function("scalar", |b| {
            b.iter(|| {
                let r = oxigaf_flame::rodrigues(
                    black_box(angles.0),
                    black_box(angles.1),
                    black_box(angles.2),
                );
                black_box(r);
            });
        });

        group.bench_function("simd", |b| {
            b.iter(|| {
                let r = rodrigues_simd(
                    black_box(angles.0),
                    black_box(angles.1),
                    black_box(angles.2),
                );
                black_box(r);
            });
        });

        // Batch rotations
        for count in [5, 10, 50, 100] {
            let rotations: Vec<[f32; 3]> = (0..count)
                .map(|i| {
                    let t = i as f32 * 0.1;
                    [t.sin() * 0.5, t.cos() * 0.5, (t * 0.5).sin() * 0.5]
                })
                .collect();

            group.throughput(Throughput::Elements(count as u64));

            group.bench_with_input(
                BenchmarkId::new("batch_scalar", count),
                &rotations,
                |b, rots| {
                    b.iter(|| {
                        let results: Vec<_> = rots
                            .iter()
                            .map(|&[rx, ry, rz]| oxigaf_flame::rodrigues(rx, ry, rz))
                            .collect();
                        black_box(results);
                    });
                },
            );

            group.bench_with_input(
                BenchmarkId::new("batch_simd", count),
                &rotations,
                |b, rots| {
                    b.iter(|| {
                        let results = rodrigues_batch(black_box(rots));
                        black_box(results);
                    });
                },
            );
        }

        group.finish();
    }

    fn bench_matrix_ops_simd(c: &mut Criterion) {
        let mut group = c.benchmark_group("matrix_ops");

        // Create test matrices
        let a = na::Matrix4::new(
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        );
        let b = na::Matrix4::new(
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
        );
        let v = na::Vector4::new(1.0, 2.0, 3.0, 1.0);

        // Matrix-matrix multiply
        group.bench_function("mat4_mul_scalar", |b_iter| {
            b_iter.iter(|| {
                let result = black_box(&a) * black_box(&b);
                black_box(result);
            });
        });

        group.bench_function("mat4_mul_simd", |b_iter| {
            b_iter.iter(|| {
                let result = mat4_mul_simd(black_box(&a), black_box(&b));
                black_box(result);
            });
        });

        // Matrix-vector multiply
        group.bench_function("mat4_vec4_mul_scalar", |b_iter| {
            b_iter.iter(|| {
                let result = black_box(&a) * black_box(&v);
                black_box(result);
            });
        });

        group.bench_function("mat4_vec4_mul_simd", |b_iter| {
            b_iter.iter(|| {
                let result = mat4_vec4_mul_simd(black_box(&a), black_box(&v));
                black_box(result);
            });
        });

        // Weighted matrix sum (LBS core operation)
        for n_joints in [5, 8, 16] {
            let matrices: Vec<na::Matrix4<f32>> = (0..n_joints)
                .map(|i| {
                    let t = i as f32 * 0.1;
                    na::Matrix4::new(
                        t.cos(),
                        -t.sin(),
                        0.0,
                        t,
                        t.sin(),
                        t.cos(),
                        0.0,
                        t * 2.0,
                        0.0,
                        0.0,
                        1.0,
                        t * 0.5,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                    )
                })
                .collect();

            let weights: Vec<f32> = (0..n_joints).map(|_i| 1.0 / n_joints as f32).collect();

            group.bench_with_input(
                BenchmarkId::new("weighted_sum_scalar", n_joints),
                &(&matrices, &weights),
                |b_iter, (mats, ws)| {
                    b_iter.iter(|| {
                        let mut result = na::Matrix4::<f32>::zeros();
                        for (m, &w) in mats.iter().zip(ws.iter()) {
                            if w.abs() > 1e-12 {
                                result += w * m;
                            }
                        }
                        black_box(result);
                    });
                },
            );

            group.bench_with_input(
                BenchmarkId::new("weighted_sum_simd", n_joints),
                &(&matrices, &weights),
                |b_iter, (mats, ws)| {
                    b_iter.iter(|| {
                        let result = weighted_matrix_sum_simd(black_box(mats), black_box(ws));
                        black_box(result);
                    });
                },
            );
        }

        group.finish();
    }

    fn bench_blend_shapes_simd(c: &mut Criterion) {
        let mut group = c.benchmark_group("blend_shapes_simd");

        for n_vertices in [1000, 5023, 10000] {
            for n_coeffs in [10, 50, 100] {
                group.throughput(Throughput::Elements(n_vertices as u64));

                let v = Array2::from_shape_fn((n_vertices, 3), |(i, j)| (i + j) as f32 * 0.001);
                let dirs = Array3::from_shape_fn((n_vertices, 3, n_coeffs), |(i, j, k)| {
                    ((i + j + k) as f32 * 0.0001).sin()
                });
                let coeffs: Vec<f32> = (0..n_coeffs).map(|i| (i as f32 * 0.1).sin()).collect();

                let label = format!("{}v_{}c", n_vertices, n_coeffs);

                group.bench_with_input(
                    BenchmarkId::new("scalar", &label),
                    &(&dirs, &coeffs),
                    |b_iter, (dirs, coeffs)| {
                        b_iter.iter(|| {
                            let mut v_clone = v.clone();
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

                group.bench_with_input(
                    BenchmarkId::new("simd", &label),
                    &(&dirs, &coeffs),
                    |b_iter, (dirs, coeffs)| {
                        b_iter.iter(|| {
                            let mut v_clone = v.clone();
                            apply_blend_shapes_simd(&mut v_clone, dirs, coeffs);
                            black_box(v_clone);
                        });
                    },
                );
            }
        }

        group.finish();
    }

    fn bench_vertices_soa(c: &mut Criterion) {
        let mut group = c.benchmark_group("vertices_soa");

        for n_vertices in [1000, 5023, 10000] {
            group.throughput(Throughput::Elements(n_vertices as u64));

            let aos: Vec<na::Point3<f32>> = (0..n_vertices)
                .map(|i| na::Point3::new(i as f32 * 0.01, i as f32 * 0.02, i as f32 * 0.03))
                .collect();

            let transform = na::Matrix4::new(
                0.866, -0.5, 0.0, 1.0, 0.5, 0.866, 0.0, 2.0, 0.0, 0.0, 1.0, 3.0, 0.0, 0.0, 0.0, 1.0,
            );

            // AoS transformation (scalar)
            group.bench_with_input(
                BenchmarkId::new("transform_aos", n_vertices),
                &(&aos, &transform),
                |b_iter, (verts, xform)| {
                    b_iter.iter(|| {
                        let result: Vec<na::Point3<f32>> = verts
                            .iter()
                            .map(|v| {
                                let hv = na::Vector4::new(v.x, v.y, v.z, 1.0);
                                let r = *xform * hv;
                                na::Point3::new(r[0], r[1], r[2])
                            })
                            .collect();
                        black_box(result);
                    });
                },
            );

            // SoA transformation (SIMD)
            let soa = VerticesSoA::from_aos(&aos);
            group.bench_with_input(
                BenchmarkId::new("transform_soa_simd", n_vertices),
                &(&soa, &transform),
                |b_iter, (verts, xform)| {
                    b_iter.iter(|| {
                        let mut soa_clone = (*verts).clone();
                        soa_clone.transform_simd(xform);
                        black_box(soa_clone);
                    });
                },
            );

            // Roundtrip conversion overhead
            group.bench_with_input(
                BenchmarkId::new("aos_to_soa_to_aos", n_vertices),
                &aos,
                |b_iter, verts| {
                    b_iter.iter(|| {
                        let soa = VerticesSoA::from_aos(black_box(verts));
                        let back = soa.to_aos();
                        black_box(back);
                    });
                },
            );
        }

        group.finish();
    }

    criterion_group!(
        benches,
        bench_rodrigues_simd,
        bench_matrix_ops_simd,
        bench_blend_shapes_simd,
        bench_vertices_soa,
    );
}

// criterion_group! generates a `pub fn benches(c: &mut Criterion)` inside the module.
// We invoke criterion_main! from the crate root so it generates the real `fn main()`.
#[cfg(all(feature = "simd", nightly))]
criterion::criterion_main!(bench_impl::benches);
