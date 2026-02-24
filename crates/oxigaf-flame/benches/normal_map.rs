//! Benchmarks for normal map rendering performance.
//!
//! This benchmark suite measures the performance of the optimized normal map
//! renderer with different mesh complexities and image resolutions.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use nalgebra as na;
use oxigaf_flame::{Camera, Mesh, NormalMapRenderer};
use std::hint::black_box;

/// Generate a test mesh with the specified number of vertices.
fn generate_test_mesh(num_vertices: usize) -> Mesh {
    let mut vertices = Vec::with_capacity(num_vertices);
    let mut faces = Vec::new();

    // Generate vertices in a grid pattern
    let grid_size = (num_vertices as f32).sqrt().ceil() as usize;
    for i in 0..grid_size {
        for j in 0..grid_size {
            if vertices.len() >= num_vertices {
                break;
            }
            let x = (i as f32 / grid_size as f32 - 0.5) * 0.2;
            let y = (j as f32 / grid_size as f32 - 0.5) * 0.2;
            let z = 0.0;
            vertices.push(na::Point3::new(x, y, z));
        }
        if vertices.len() >= num_vertices {
            break;
        }
    }

    // Generate triangular faces
    for i in 0..(grid_size - 1) {
        for j in 0..(grid_size - 1) {
            let idx = i * grid_size + j;
            if idx + grid_size + 1 < vertices.len() {
                // First triangle
                faces.push([idx as u32, (idx + 1) as u32, (idx + grid_size) as u32]);
                // Second triangle
                faces.push([
                    (idx + 1) as u32,
                    (idx + grid_size + 1) as u32,
                    (idx + grid_size) as u32,
                ]);
            }
        }
    }

    Mesh::new(vertices, faces)
}

/// Generate a more realistic head-like mesh (sphere approximation).
fn generate_sphere_mesh(subdivisions: usize) -> Mesh {
    let mut vertices = Vec::new();
    let mut faces = Vec::new();

    let radius = 0.1f32;
    let num_lat = subdivisions;
    let num_lon = subdivisions * 2;

    // Generate vertices
    for i in 0..=num_lat {
        let theta = std::f32::consts::PI * i as f32 / num_lat as f32;
        let sin_theta = theta.sin();
        let cos_theta = theta.cos();

        for j in 0..=num_lon {
            let phi = 2.0 * std::f32::consts::PI * j as f32 / num_lon as f32;
            let sin_phi = phi.sin();
            let cos_phi = phi.cos();

            let x = radius * sin_theta * cos_phi;
            let y = radius * sin_theta * sin_phi;
            let z = radius * cos_theta;

            vertices.push(na::Point3::new(x, y, z));
        }
    }

    // Generate faces
    for i in 0..num_lat {
        for j in 0..num_lon {
            let first = i * (num_lon + 1) + j;
            let second = first + num_lon + 1;

            faces.push([first as u32, second as u32, (first + 1) as u32]);
            faces.push([second as u32, (second + 1) as u32, (first + 1) as u32]);
        }
    }

    Mesh::new(vertices, faces)
}

/// Benchmark normal map rendering with different mesh sizes.
fn bench_render_varying_mesh_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("normal_map_render_mesh_size");

    let resolutions = vec![512u32];
    let mesh_sizes = vec![
        ("small", 100),
        ("medium", 1000),
        ("large", 5000),
        ("xlarge", 10000),
    ];

    for (size_name, num_vertices) in mesh_sizes {
        let mesh = generate_test_mesh(num_vertices);
        group.throughput(Throughput::Elements(mesh.num_faces() as u64));

        for resolution in &resolutions {
            let camera = Camera::default_front(*resolution, *resolution);

            group.bench_with_input(
                BenchmarkId::new(size_name, resolution),
                resolution,
                |b, _| {
                    b.iter(|| {
                        let img = NormalMapRenderer::render(black_box(&mesh), black_box(&camera));
                        black_box(img);
                    });
                },
            );
        }
    }

    group.finish();
}

/// Benchmark normal map rendering with different image resolutions.
fn bench_render_varying_resolution(c: &mut Criterion) {
    let mut group = c.benchmark_group("normal_map_render_resolution");

    let mesh = generate_test_mesh(5000); // Standard mesh size
    let resolutions = vec![256u32, 512, 1024, 2048];

    for resolution in resolutions {
        let camera = Camera::default_front(resolution, resolution);
        group.throughput(Throughput::Elements((resolution * resolution) as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(resolution),
            &resolution,
            |b, _| {
                b.iter(|| {
                    let img = NormalMapRenderer::render(black_box(&mesh), black_box(&camera));
                    black_box(img);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark with realistic sphere mesh (head-like).
fn bench_render_sphere_mesh(c: &mut Criterion) {
    let mut group = c.benchmark_group("normal_map_render_sphere");

    let subdivisions = vec![("low", 10), ("medium", 30), ("high", 50), ("very_high", 80)];
    let resolution = 512u32;

    for (quality, subdivs) in subdivisions {
        let mesh = generate_sphere_mesh(subdivs);
        let camera = Camera::default_front(resolution, resolution);

        group.throughput(Throughput::Elements(mesh.num_faces() as u64));
        group.bench_with_input(BenchmarkId::from_parameter(quality), &quality, |b, _| {
            b.iter(|| {
                let img = NormalMapRenderer::render(black_box(&mesh), black_box(&camera));
                black_box(img);
            });
        });
    }

    group.finish();
}

/// Benchmark tile processing performance.
fn bench_tile_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("normal_map_tile_processing");

    // Create a large mesh that covers most of the screen
    let mesh = generate_sphere_mesh(50);
    let resolutions = vec![512u32, 1024];

    for resolution in resolutions {
        let camera = Camera::default_front(resolution, resolution);

        group.bench_with_input(
            BenchmarkId::from_parameter(resolution),
            &resolution,
            |b, _| {
                b.iter(|| {
                    let img = NormalMapRenderer::render(black_box(&mesh), black_box(&camera));
                    black_box(img);
                });
            },
        );
    }

    group.finish();
}

/// Comprehensive rendering benchmark with various scenarios.
fn bench_comprehensive(c: &mut Criterion) {
    let mut group = c.benchmark_group("normal_map_comprehensive");
    group.sample_size(20); // Reduce sample size for longer benchmarks

    // Realistic FLAME-like scenario: 5000 vertices, 512x512 resolution
    let mesh = generate_sphere_mesh(40); // ~6400 faces
    let camera = Camera::default_front(512, 512);

    group.throughput(Throughput::Elements((mesh.num_faces() * 512 * 512) as u64));
    group.bench_function("flame_realistic_512", |b| {
        b.iter(|| {
            let img = NormalMapRenderer::render(black_box(&mesh), black_box(&camera));
            black_box(img);
        });
    });

    // High-resolution rendering
    let camera_hd = Camera::default_front(1024, 1024);
    group.bench_function("flame_realistic_1024", |b| {
        b.iter(|| {
            let img = NormalMapRenderer::render(black_box(&mesh), black_box(&camera_hd));
            black_box(img);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_render_varying_mesh_size,
    bench_render_varying_resolution,
    bench_render_sphere_mesh,
    bench_tile_processing,
    bench_comprehensive,
);
criterion_main!(benches);
