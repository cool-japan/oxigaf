//! Comprehensive benchmarks for oxigaf-flame.
//!
//! Benchmarks:
//! - LBS forward pass (single and batched)
//! - Normal map rendering
//! - Rodrigues rotation
//! - Blend shapes application
//!
//! Run with: cargo bench -p oxigaf-flame

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use nalgebra as na;
use ndarray::{Array2, Array3};
use std::hint::black_box;

use oxigaf_flame::{rodrigues, Camera, FlameParams, Mesh, NormalMapRenderer};

// ---------------------------------------------------------------------------
// Mock Data Generators
// ---------------------------------------------------------------------------

/// Create synthetic FLAME-like model data for benchmarking.
/// Real FLAME has ~5023 vertices, 5 joints, 300 shape params, 100 expr params.
struct MockFlameData {
    v_template: Array2<f32>,
    shapedirs: Array3<f32>,
    expressiondirs: Array3<f32>,
    #[allow(dead_code)]
    posedirs: Array3<f32>,
    j_regressor: Array2<f32>,
    lbs_weights: Array2<f32>,
    faces: Vec<[u32; 3]>,
    num_vertices: usize,
    n_joints: usize,
    n_shape: usize,
    n_expr: usize,
}

impl MockFlameData {
    fn new(num_vertices: usize, n_shape: usize, n_expr: usize) -> Self {
        let n_joints = 5;

        // Create template vertices on a sphere
        let v_template = Array2::from_shape_fn((num_vertices, 3), |(i, j)| {
            let theta = (i as f32 / num_vertices as f32) * std::f32::consts::PI * 2.0;
            let phi = (i as f32 / num_vertices as f32) * std::f32::consts::PI;
            match j {
                0 => phi.sin() * theta.cos() * 0.1, // x
                1 => phi.sin() * theta.sin() * 0.1, // y
                _ => phi.cos() * 0.1,               // z
            }
        });

        // Small blend shape directions
        let shapedirs = Array3::from_shape_fn((num_vertices, 3, n_shape), |(_i, _j, _k)| 0.001);
        let expressiondirs = Array3::from_shape_fn((num_vertices, 3, n_expr), |(_i, _j, _k)| 0.001);
        let posedirs = Array3::from_shape_fn((num_vertices, 3, (n_joints - 1) * 9), |_| 0.0001);

        // Joint regressor (sparse, but we use dense for simplicity)
        let j_regressor = Array2::from_shape_fn((n_joints, num_vertices), |(j, v)| {
            if v < 10 && j == v % n_joints {
                1.0
            } else {
                0.0
            }
        });

        // LBS weights (each vertex influenced by nearby joints)
        let lbs_weights = Array2::from_shape_fn((num_vertices, n_joints), |(v, j)| {
            let dist = ((v % n_joints) as i32 - j as i32).abs() as f32;
            if dist < 2.0 {
                1.0 / (1.0 + dist)
            } else {
                0.0
            }
        });

        // Normalize LBS weights per vertex
        let mut lbs_weights = lbs_weights;
        for i in 0..num_vertices {
            let sum: f32 = (0..n_joints).map(|j| lbs_weights[[i, j]]).sum();
            if sum > 1e-8 {
                for j in 0..n_joints {
                    lbs_weights[[i, j]] /= sum;
                }
            }
        }

        // Generate faces (triangles)
        let num_faces = num_vertices.saturating_sub(2);
        let faces: Vec<[u32; 3]> = (0..num_faces)
            .map(|i| {
                [
                    i as u32,
                    ((i + 1) % num_vertices) as u32,
                    ((i + 2) % num_vertices) as u32,
                ]
            })
            .collect();

        Self {
            v_template,
            shapedirs,
            expressiondirs,
            posedirs,
            j_regressor,
            lbs_weights,
            faces,
            num_vertices,
            n_joints,
            n_shape,
            n_expr,
        }
    }

    /// Simulate a forward pass (blend shapes + LBS).
    fn forward(&self, params: &FlameParams) -> Mesh {
        // 1. Apply shape blend shapes
        let mut v_shaped = self.v_template.clone();
        for (k, &coeff) in params.shape.iter().enumerate().take(self.n_shape) {
            if coeff.abs() > 1e-12 {
                for i in 0..self.num_vertices {
                    for j in 0..3 {
                        v_shaped[[i, j]] += coeff * self.shapedirs[[i, j, k]];
                    }
                }
            }
        }

        // 2. Apply expression blend shapes
        for (k, &coeff) in params.expression.iter().enumerate().take(self.n_expr) {
            if coeff.abs() > 1e-12 {
                for i in 0..self.num_vertices {
                    for j in 0..3 {
                        v_shaped[[i, j]] += coeff * self.expressiondirs[[i, j, k]];
                    }
                }
            }
        }

        // 3. Compute joint positions
        let joints = self.j_regressor.dot(&v_shaped);

        // 4. Compute rotation matrices
        let mut rot_mats = Vec::with_capacity(self.n_joints);
        for jt in 0..self.n_joints {
            let [rx, ry, rz] = params.joint_pose(jt);
            rot_mats.push(rodrigues(rx, ry, rz));
        }

        // 5. Compute skinning transforms
        let mut skinning = vec![na::Matrix4::<f32>::identity(); self.n_joints];
        for jt in 0..self.n_joints {
            let j_pos = na::Vector3::new(joints[[jt, 0]], joints[[jt, 1]], joints[[jt, 2]]);
            let mut local = na::Matrix4::identity();
            for r in 0..3 {
                for c in 0..3 {
                    local[(r, c)] = rot_mats[jt][(r, c)];
                }
            }
            local[(0, 3)] = j_pos.x;
            local[(1, 3)] = j_pos.y;
            local[(2, 3)] = j_pos.z;
            skinning[jt] = local;
        }

        // 6. Apply LBS
        let [tx, ty, tz] = params.translation;
        let mut vertices = Vec::with_capacity(self.num_vertices);
        for i in 0..self.num_vertices {
            let mut t = na::Matrix4::<f32>::zeros();
            for (jt, transform) in skinning.iter().enumerate().take(self.n_joints) {
                let w = self.lbs_weights[[i, jt]];
                if w.abs() > 1e-12 {
                    t += w * transform;
                }
            }
            let v = na::Vector4::new(v_shaped[[i, 0]], v_shaped[[i, 1]], v_shaped[[i, 2]], 1.0);
            let r = t * v;
            vertices.push(na::Point3::new(r[0] + tx, r[1] + ty, r[2] + tz));
        }

        Mesh::new(vertices, self.faces.clone())
    }
}

// ---------------------------------------------------------------------------
// Rodrigues Benchmark
// ---------------------------------------------------------------------------

fn bench_rodrigues(c: &mut Criterion) {
    let mut group = c.benchmark_group("rodrigues");

    // Single rotation
    group.bench_function("single", |b| {
        b.iter(|| black_box(rodrigues(black_box(0.1), black_box(0.2), black_box(0.3))))
    });

    // Batch of rotations (simulating 5 joints)
    group.bench_function("5_joints", |b| {
        let angles = [
            (0.1, 0.0, 0.0),
            (0.0, 0.1, 0.0),
            (0.0, 0.0, 0.1),
            (0.05, 0.05, 0.0),
            (0.0, 0.0, 0.0),
        ];
        b.iter(|| {
            let mut mats = Vec::with_capacity(5);
            for (rx, ry, rz) in angles {
                mats.push(rodrigues(black_box(rx), black_box(ry), black_box(rz)));
            }
            black_box(mats)
        })
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// LBS Forward Pass Benchmark
// ---------------------------------------------------------------------------

fn bench_flame_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("flame_forward_pass");

    // Standard FLAME size (~5023 vertices)
    for &num_verts in &[1000, 5023, 10000] {
        let mock = MockFlameData::new(num_verts, 100, 50);
        let params = FlameParams::neutral();

        group.throughput(Throughput::Elements(num_verts as u64));
        group.bench_with_input(
            BenchmarkId::new("vertices", num_verts),
            &num_verts,
            |b, _| b.iter(|| black_box(mock.forward(black_box(&params)))),
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Batched Forward Pass Benchmark
// ---------------------------------------------------------------------------

fn bench_flame_batched_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("flame_batched_forward");

    let mock = MockFlameData::new(5023, 100, 50);

    // Different batch sizes
    for &batch_size in &[1, 4, 8, 16, 32] {
        let params_batch: Vec<FlameParams> = (0..batch_size)
            .map(|i| {
                let mut p = FlameParams::neutral();
                // Vary some parameters
                if !p.shape.is_empty() {
                    p.shape[0] = i as f32 * 0.1;
                }
                p
            })
            .collect();

        group.throughput(Throughput::Elements(batch_size as u64));
        group.bench_with_input(
            BenchmarkId::new("batch", batch_size),
            &batch_size,
            |b, _| {
                b.iter(|| {
                    let meshes: Vec<Mesh> = params_batch
                        .iter()
                        .map(|p| mock.forward(black_box(p)))
                        .collect();
                    black_box(meshes)
                })
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Normal Map Rendering Benchmark
// ---------------------------------------------------------------------------

fn bench_normal_map(c: &mut Criterion) {
    let mut group = c.benchmark_group("normal_map_rendering");

    // Create a mock mesh
    let num_vertices = 5023;
    let mock = MockFlameData::new(num_vertices, 100, 50);
    let params = FlameParams::neutral();
    let mesh = mock.forward(&params);

    // Different resolutions
    for &resolution in &[256, 512, 1024] {
        let camera = Camera::default_front(resolution, resolution);

        group.throughput(Throughput::Elements((resolution * resolution) as u64));
        group.bench_with_input(
            BenchmarkId::new("resolution", resolution),
            &resolution,
            |b, _| {
                b.iter(|| {
                    black_box(NormalMapRenderer::render(
                        black_box(&mesh),
                        black_box(&camera),
                    ))
                })
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Blend Shapes Benchmark
// ---------------------------------------------------------------------------

fn bench_blend_shapes(c: &mut Criterion) {
    let mut group = c.benchmark_group("blend_shapes");

    let num_vertices = 5023;

    // Different numbers of coefficients
    for &n_coeffs in &[50, 100, 200, 300] {
        let v_template = Array2::from_shape_fn((num_vertices, 3), |(i, _j)| i as f32 * 0.001);
        let shapedirs = Array3::from_shape_fn((num_vertices, 3, n_coeffs), |_| 0.001);
        let coeffs: Vec<f32> = (0..n_coeffs).map(|i| (i as f32 * 0.01).sin()).collect();

        group.throughput(Throughput::Elements((num_vertices * n_coeffs) as u64));
        group.bench_with_input(
            BenchmarkId::new("coefficients", n_coeffs),
            &n_coeffs,
            |b, _| {
                b.iter(|| {
                    let mut v = v_template.clone();
                    for (k, &coeff) in coeffs.iter().enumerate() {
                        if coeff.abs() > 1e-12 {
                            for i in 0..num_vertices {
                                for j in 0..3 {
                                    v[[i, j]] += coeff * shapedirs[[i, j, k]];
                                }
                            }
                        }
                    }
                    black_box(v)
                })
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Compute Normals Benchmark
// ---------------------------------------------------------------------------

fn bench_compute_normals(c: &mut Criterion) {
    let mut group = c.benchmark_group("compute_normals");

    for &num_verts in &[1000_usize, 5023, 10000] {
        // Create vertices on a sphere
        let vertices: Vec<na::Point3<f32>> = (0..num_verts)
            .map(|i| {
                let theta = (i as f32 / num_verts as f32) * std::f32::consts::PI * 2.0;
                let phi = (i as f32 / num_verts as f32) * std::f32::consts::PI;
                na::Point3::new(
                    phi.sin() * theta.cos() * 0.1,
                    phi.sin() * theta.sin() * 0.1,
                    phi.cos() * 0.1,
                )
            })
            .collect();

        let num_faces = num_verts.saturating_sub(2);
        let faces: Vec<[u32; 3]> = (0..num_faces)
            .map(|i| {
                [
                    i as u32,
                    ((i + 1) % num_verts) as u32,
                    ((i + 2) % num_verts) as u32,
                ]
            })
            .collect();

        group.throughput(Throughput::Elements(num_verts as u64));
        group.bench_with_input(
            BenchmarkId::new("vertices", num_verts),
            &num_verts,
            |b, _| {
                b.iter(|| {
                    let mut normals = vec![na::Vector3::<f32>::zeros(); num_verts];
                    oxigaf_flame::compute_normals_into(
                        black_box(&vertices),
                        black_box(&faces),
                        &mut normals,
                    );
                    black_box(normals)
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
    bench_rodrigues,
    bench_flame_forward,
    bench_flame_batched_forward,
    bench_normal_map,
    bench_blend_shapes,
    bench_compute_normals,
);

criterion_main!(benches);
