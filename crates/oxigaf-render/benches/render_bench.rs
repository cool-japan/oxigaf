//! Comprehensive benchmarks for oxigaf-render.
//!
//! Benchmarks:
//! - GPU radix sort (CPU-side timing only, setup overhead)
//! - Spherical harmonics evaluation
//! - Buffer pool operations
//! - Matrix operations for preprocessing
//!
//! Note: Full GPU benchmarks require async runtime and device availability.
//! These benchmarks focus on CPU-side operations and setup costs.
//!
//! Run with: cargo bench -p oxigaf-render

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use glam::{Mat4, Quat, Vec3, Vec4};
use std::hint::black_box;

// ---------------------------------------------------------------------------
// Spherical Harmonics Evaluation (CPU Reference)
// ---------------------------------------------------------------------------

/// Evaluate SH degree 0 (constant).
#[inline]
fn sh_eval_0(coeffs: &[f32; 3]) -> Vec3 {
    const C0: f32 = 0.282_094_79;
    Vec3::new(coeffs[0] * C0, coeffs[1] * C0, coeffs[2] * C0)
}

/// Evaluate SH degree 1.
#[inline]
fn sh_eval_1(coeffs: &[[f32; 3]; 4], dir: Vec3) -> Vec3 {
    const C0: f32 = 0.282_094_79;
    const C1: f32 = 0.488_602_51;

    let mut result = Vec3::new(coeffs[0][0] * C0, coeffs[0][1] * C0, coeffs[0][2] * C0);

    // Y_1^(-1) = sqrt(3/(4*pi)) * y
    result.x += C1 * dir.y * coeffs[1][0];
    result.y += C1 * dir.y * coeffs[1][1];
    result.z += C1 * dir.y * coeffs[1][2];

    // Y_1^0 = sqrt(3/(4*pi)) * z
    result.x += C1 * dir.z * coeffs[2][0];
    result.y += C1 * dir.z * coeffs[2][1];
    result.z += C1 * dir.z * coeffs[2][2];

    // Y_1^1 = sqrt(3/(4*pi)) * x
    result.x += C1 * dir.x * coeffs[3][0];
    result.y += C1 * dir.x * coeffs[3][1];
    result.z += C1 * dir.x * coeffs[3][2];

    result
}

/// Evaluate SH degree 2.
#[inline]
#[allow(clippy::many_single_char_names)]
fn sh_eval_2(coeffs: &[[f32; 3]; 9], dir: Vec3) -> Vec3 {
    const C0: f32 = 0.282_094_79;
    const C1: f32 = 0.488_602_51;
    const C2_0: f32 = 1.092_548_5;
    const C2_1: f32 = 0.315_391_57;
    const C2_2: f32 = 0.546_274_22;

    let (x, y, z) = (dir.x, dir.y, dir.z);

    // Degree 0 and 1
    let mut result = Vec3::new(coeffs[0][0] * C0, coeffs[0][1] * C0, coeffs[0][2] * C0);

    result.x += C1 * y * coeffs[1][0] + C1 * z * coeffs[2][0] + C1 * x * coeffs[3][0];
    result.y += C1 * y * coeffs[1][1] + C1 * z * coeffs[2][1] + C1 * x * coeffs[3][1];
    result.z += C1 * y * coeffs[1][2] + C1 * z * coeffs[2][2] + C1 * x * coeffs[3][2];

    // Degree 2
    let xy = x * y;
    let yz = y * z;
    let xz = x * z;
    let x2 = x * x;
    let y2 = y * y;
    let z2 = z * z;

    // Y_2^(-2) = sqrt(15/(4*pi)) * xy
    result.x += C2_0 * xy * coeffs[4][0];
    result.y += C2_0 * xy * coeffs[4][1];
    result.z += C2_0 * xy * coeffs[4][2];

    // Y_2^(-1) = sqrt(15/(4*pi)) * yz
    result.x += C2_0 * yz * coeffs[5][0];
    result.y += C2_0 * yz * coeffs[5][1];
    result.z += C2_0 * yz * coeffs[5][2];

    // Y_2^0 = sqrt(5/(16*pi)) * (3z^2 - 1)
    let t = 3.0 * z2 - 1.0;
    result.x += C2_1 * t * coeffs[6][0];
    result.y += C2_1 * t * coeffs[6][1];
    result.z += C2_1 * t * coeffs[6][2];

    // Y_2^1 = sqrt(15/(4*pi)) * xz
    result.x += C2_0 * xz * coeffs[7][0];
    result.y += C2_0 * xz * coeffs[7][1];
    result.z += C2_0 * xz * coeffs[7][2];

    // Y_2^2 = sqrt(15/(16*pi)) * (x^2 - y^2)
    let t = x2 - y2;
    result.x += C2_2 * t * coeffs[8][0];
    result.y += C2_2 * t * coeffs[8][1];
    result.z += C2_2 * t * coeffs[8][2];

    result
}

fn bench_sh_evaluation(c: &mut Criterion) {
    let mut group = c.benchmark_group("sh_evaluation");

    let dir = Vec3::new(0.577, 0.577, 0.577).normalize();

    // SH degree 0
    let coeffs_0 = [0.5_f32, 0.6, 0.7];
    group.bench_function("degree_0", |b| {
        b.iter(|| black_box(sh_eval_0(black_box(&coeffs_0))))
    });

    // SH degree 1
    let coeffs_1 = [
        [0.5, 0.6, 0.7],
        [0.1, 0.2, 0.3],
        [0.2, 0.3, 0.4],
        [0.3, 0.4, 0.5],
    ];
    group.bench_function("degree_1", |b| {
        b.iter(|| black_box(sh_eval_1(black_box(&coeffs_1), black_box(dir))))
    });

    // SH degree 2
    let coeffs_2 = [
        [0.5, 0.6, 0.7],
        [0.1, 0.2, 0.3],
        [0.2, 0.3, 0.4],
        [0.3, 0.4, 0.5],
        [0.05, 0.06, 0.07],
        [0.06, 0.07, 0.08],
        [0.07, 0.08, 0.09],
        [0.08, 0.09, 0.10],
        [0.09, 0.10, 0.11],
    ];
    group.bench_function("degree_2", |b| {
        b.iter(|| black_box(sh_eval_2(black_box(&coeffs_2), black_box(dir))))
    });

    // Batch evaluation
    for &batch_size in &[1000, 10000, 100000] {
        let directions: Vec<Vec3> = (0..batch_size)
            .map(|i| {
                let theta = (i as f32 / batch_size as f32) * std::f32::consts::PI * 2.0;
                let phi = (i as f32 / batch_size as f32) * std::f32::consts::PI;
                Vec3::new(phi.sin() * theta.cos(), phi.sin() * theta.sin(), phi.cos())
            })
            .collect();

        group.throughput(Throughput::Elements(batch_size as u64));
        group.bench_with_input(
            BenchmarkId::new("batch_degree_2", batch_size),
            &batch_size,
            |b, _| {
                b.iter(|| {
                    let results: Vec<Vec3> = directions
                        .iter()
                        .map(|&d| sh_eval_2(&coeffs_2, d))
                        .collect();
                    black_box(results)
                })
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Covariance Matrix Computation (for 3D Gaussian projection)
// ---------------------------------------------------------------------------

/// Compute 3D covariance matrix from scale and rotation.
#[inline]
fn compute_covariance_3d(scale: Vec3, rotation: Quat) -> Mat4 {
    // S = diag(scale)
    let s = Mat4::from_scale(scale);

    // R = rotation matrix
    let r = Mat4::from_quat(rotation);

    // Covariance = R * S * S^T * R^T = R * S^2 * R^T
    let rs = r * s;
    rs * rs.transpose()
}

/// Project 3D covariance to 2D screen space.
#[inline]
fn project_covariance(cov3d: Mat4, view: Mat4, focal: Vec2, p_view: Vec3) -> [f32; 3] {
    // Jacobian of perspective projection
    let rz = 1.0 / p_view.z;
    let rz2 = rz * rz;

    let j = Mat4::from_cols(
        Vec4::new(focal.x * rz, 0.0, 0.0, 0.0),
        Vec4::new(0.0, focal.y * rz, 0.0, 0.0),
        Vec4::new(
            -focal.x * p_view.x * rz2,
            -focal.y * p_view.y * rz2,
            0.0,
            0.0,
        ),
        Vec4::ZERO,
    );

    // Cov2D = J * W * Cov3D * W^T * J^T
    let w = view;
    let t = j * w * cov3d * w.transpose() * j.transpose();

    // Extract 2x2 upper-left block
    [t.x_axis.x, t.x_axis.y, t.y_axis.y]
}

use glam::Vec2;

fn bench_covariance(c: &mut Criterion) {
    let mut group = c.benchmark_group("covariance");

    let scale = Vec3::new(0.01, 0.02, 0.01);
    let rotation = Quat::from_euler(glam::EulerRot::XYZ, 0.1, 0.2, 0.3);
    let view = Mat4::look_at_rh(Vec3::new(0.0, 0.0, 1.0), Vec3::ZERO, Vec3::Y);
    let focal = Vec2::new(500.0, 500.0);
    let p_view = Vec3::new(0.1, 0.1, 0.5);

    group.bench_function("compute_3d", |b| {
        b.iter(|| black_box(compute_covariance_3d(black_box(scale), black_box(rotation))))
    });

    let cov3d = compute_covariance_3d(scale, rotation);
    group.bench_function("project_2d", |b| {
        b.iter(|| {
            black_box(project_covariance(
                black_box(cov3d),
                black_box(view),
                black_box(focal),
                black_box(p_view),
            ))
        })
    });

    // Batch covariance computation
    for &batch_size in &[1000, 10000, 100000] {
        let scales: Vec<Vec3> = (0..batch_size)
            .map(|i| Vec3::new(0.01 + i as f32 * 0.0001, 0.02, 0.01))
            .collect();
        let rotations: Vec<Quat> = (0..batch_size)
            .map(|i| Quat::from_euler(glam::EulerRot::XYZ, i as f32 * 0.001, 0.2, 0.3))
            .collect();

        group.throughput(Throughput::Elements(batch_size as u64));
        group.bench_with_input(
            BenchmarkId::new("batch_3d", batch_size),
            &batch_size,
            |b, _| {
                b.iter(|| {
                    let results: Vec<Mat4> = scales
                        .iter()
                        .zip(rotations.iter())
                        .map(|(&s, &r)| compute_covariance_3d(s, r))
                        .collect();
                    black_box(results)
                })
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Tile Assignment (CPU reference)
// ---------------------------------------------------------------------------

/// Compute which tiles a 2D Gaussian overlaps.
fn compute_tile_overlap(
    mean_2d: Vec2,
    cov_2d: [f32; 3],
    tile_size: u32,
    grid_width: u32,
    grid_height: u32,
    radius_scale: f32,
) -> Vec<(u32, u32)> {
    // Compute eigenvalues of 2D covariance for bounding radius
    let a = cov_2d[0];
    let b = cov_2d[1];
    let c = cov_2d[2];

    let trace = a + c;
    let det = a * c - b * b;
    let discriminant = (trace * trace / 4.0 - det).max(0.0);
    let sqrt_disc = discriminant.sqrt();

    let lambda1 = trace / 2.0 + sqrt_disc;
    let radius = (lambda1.sqrt() * radius_scale).ceil() as i32;

    let tile_x = (mean_2d.x / tile_size as f32).floor() as i32;
    let tile_y = (mean_2d.y / tile_size as f32).floor() as i32;

    let tiles_radius = (radius / tile_size as i32) + 1;

    let mut tiles = Vec::new();
    for ty in (tile_y - tiles_radius).max(0)..=(tile_y + tiles_radius).min(grid_height as i32 - 1) {
        for tx in
            (tile_x - tiles_radius).max(0)..=(tile_x + tiles_radius).min(grid_width as i32 - 1)
        {
            tiles.push((tx as u32, ty as u32));
        }
    }

    tiles
}

fn bench_tile_assignment(c: &mut Criterion) {
    let mut group = c.benchmark_group("tile_assignment");

    let tile_size = 16;
    let grid_width = 64;
    let grid_height = 64;
    let radius_scale = 3.0;

    // Single Gaussian
    let mean = Vec2::new(512.0, 512.0);
    let cov = [0.001, 0.0, 0.001];

    group.bench_function("single", |b| {
        b.iter(|| {
            black_box(compute_tile_overlap(
                black_box(mean),
                black_box(cov),
                tile_size,
                grid_width,
                grid_height,
                radius_scale,
            ))
        })
    });

    // Batch of Gaussians
    for &batch_size in &[1000, 10000, 100000] {
        let means: Vec<Vec2> = (0..batch_size)
            .map(|i| Vec2::new((i % 1024) as f32, (i / 1024) as f32))
            .collect();
        let covs: Vec<[f32; 3]> = (0..batch_size).map(|_| [0.001, 0.0, 0.001]).collect();

        group.throughput(Throughput::Elements(batch_size as u64));
        group.bench_with_input(
            BenchmarkId::new("batch", batch_size),
            &batch_size,
            |b, _| {
                b.iter(|| {
                    let results: Vec<Vec<(u32, u32)>> = means
                        .iter()
                        .zip(covs.iter())
                        .map(|(&m, &c)| {
                            compute_tile_overlap(
                                m,
                                c,
                                tile_size,
                                grid_width,
                                grid_height,
                                radius_scale,
                            )
                        })
                        .collect();
                    black_box(results)
                })
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Radix Sort Key Generation (CPU)
// ---------------------------------------------------------------------------

/// Generate sort key from tile ID and depth.
#[inline]
fn make_sort_key(tile_id: u32, depth: f32) -> u64 {
    // Encode depth as sortable bits (flip if negative for proper ordering)
    let depth_bits = depth.to_bits();
    let depth_key = if depth >= 0.0 {
        depth_bits ^ 0x8000_0000
    } else {
        !depth_bits
    };

    ((tile_id as u64) << 32) | (depth_key as u64)
}

fn bench_sort_key_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("sort_key_generation");

    // Single key
    group.bench_function("single", |b| {
        b.iter(|| black_box(make_sort_key(black_box(42), black_box(0.5))))
    });

    // Batch
    for &batch_size in &[1000, 10000, 100000, 1000000] {
        let tile_ids: Vec<u32> = (0..batch_size).map(|i| i as u32 % 4096).collect();
        let depths: Vec<f32> = (0..batch_size).map(|i| i as f32 * 0.001).collect();

        group.throughput(Throughput::Elements(batch_size as u64));
        group.bench_with_input(
            BenchmarkId::new("batch", batch_size),
            &batch_size,
            |b, _| {
                b.iter(|| {
                    let keys: Vec<u64> = tile_ids
                        .iter()
                        .zip(depths.iter())
                        .map(|(&t, &d)| make_sort_key(t, d))
                        .collect();
                    black_box(keys)
                })
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// CPU Radix Sort (for comparison)
// ---------------------------------------------------------------------------

fn radix_sort_cpu(keys: &mut [u64], values: &mut [u32]) {
    if keys.is_empty() {
        return;
    }

    let n = keys.len();
    let mut temp_keys = vec![0u64; n];
    let mut temp_values = vec![0u32; n];

    // 16 passes of 4-bit radix sort
    for pass in 0..16 {
        let shift = pass * 4;

        // Count histogram
        let mut counts = [0usize; 16];
        for &key in keys.iter() {
            let digit = ((key >> shift) & 0xF) as usize;
            counts[digit] += 1;
        }

        // Prefix sum
        let mut offsets = [0usize; 16];
        let mut sum = 0;
        for (i, &count) in counts.iter().enumerate() {
            offsets[i] = sum;
            sum += count;
        }

        // Scatter
        for i in 0..n {
            let digit = ((keys[i] >> shift) & 0xF) as usize;
            let pos = offsets[digit];
            offsets[digit] += 1;
            temp_keys[pos] = keys[i];
            temp_values[pos] = values[i];
        }

        // Swap
        keys.copy_from_slice(&temp_keys);
        values.copy_from_slice(&temp_values);
    }
}

fn bench_cpu_radix_sort(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_radix_sort");

    for &n in &[1000, 10000, 100000] {
        let keys: Vec<u64> = (0..n)
            .map(|i| make_sort_key((i as u32) % 4096, (n - i) as f32 * 0.001))
            .collect();
        let values: Vec<u32> = (0..n as u32).collect();

        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("elements", n), &n, |b, _| {
            b.iter(|| {
                let mut k = keys.clone();
                let mut v = values.clone();
                radix_sort_cpu(&mut k, &mut v);
                black_box((k, v))
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
    bench_sh_evaluation,
    bench_covariance,
    bench_tile_assignment,
    bench_sort_key_generation,
    bench_cpu_radix_sort,
);

criterion_main!(benches);
