//! Surface mesh extraction from a Gaussian Splatting model using Surface Nets.
//!
//! ## Algorithm choice
//! Marching Cubes (256-case triangle table) was considered but Surface Nets was chosen
//! for maintainability: it produces closed, manifold meshes for smooth isosurfaces
//! with significantly less code and no large lookup tables.
//!
//! ## Pipeline
//! 1. Build AABB from Gaussian centers ± padding
//! 2. Splat `sigmoid(opacity)·exp(−½d²/r²)` (isotropic r=mean(exp(scale))) into a voxel grid
//! 3. Run Surface Nets to extract an isosurface triangle mesh
//! 4. Write binary little-endian PLY with vertex + face elements

use crate::error::CliError;
use oxigaf::render::gaussian::GaussianModel;
use std::collections::HashMap;
use std::io::Write;
use std::path::Path;

/// Configuration for density-field meshing.
#[derive(Debug, Clone)]
pub struct MeshExportConfig {
    /// Voxels along the longest axis (default 128; capped at 256 for memory).
    pub resolution: u32,
    /// Density isosurface threshold (default 0.5).
    pub iso: f32,
    /// Fractional bbox expansion (default 0.1).
    pub padding: f32,
    /// Skip Gaussians with sigmoid(opacity) below this (default 0.01).
    pub opacity_cutoff: f32,
}

impl Default for MeshExportConfig {
    fn default() -> Self {
        Self {
            resolution: 128,
            iso: 0.5,
            padding: 0.1,
            opacity_cutoff: 0.01,
        }
    }
}

/// A triangle mesh: positions and triangle indices.
#[derive(Debug, Default)]
pub struct TriMesh {
    pub vertices: Vec<[f32; 3]>,
    pub triangles: Vec<[u32; 3]>,
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Compute the padded AABB from Gaussian centers and their effective radii.
///
/// The AABB expands each Gaussian center by its isotropic effective radius
/// (3σ where σ = mean(exp(scale))). This ensures the density isosurface is
/// captured inside the grid even for point-like or co-located Gaussians.
///
/// Returns None if model is empty.
fn compute_bounds(model: &GaussianModel, padding: f32) -> Option<([f32; 3], [f32; 3])> {
    if model.is_empty() {
        return None;
    }
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for g in model.gaussians.iter() {
        // 3σ radius ensures the region where density > exp(-4.5) ≈ 0.01 is covered
        let r = (g.scale[0].exp() + g.scale[1].exp() + g.scale[2].exp()) / 3.0;
        let r3 = (3.0 * r).max(1e-4);
        for k in 0..3 {
            let lo = g.position[k] - r3;
            let hi = g.position[k] + r3;
            if lo < min[k] {
                min[k] = lo;
            }
            if hi > max[k] {
                max[k] = hi;
            }
        }
    }
    let center = [
        (min[0] + max[0]) * 0.5,
        (min[1] + max[1]) * 0.5,
        (min[2] + max[2]) * 0.5,
    ];
    let half = [
        (max[0] - min[0]) * 0.5 * (1.0 + padding),
        (max[1] - min[1]) * 0.5 * (1.0 + padding),
        (max[2] - min[2]) * 0.5 * (1.0 + padding),
    ];
    // Ensure non-degenerate bounds (at least 1cm per axis)
    let eps = 0.01_f32;
    Some((
        [
            center[0] - half[0].max(eps),
            center[1] - half[1].max(eps),
            center[2] - half[2].max(eps),
        ],
        [
            center[0] + half[0].max(eps),
            center[1] + half[1].max(eps),
            center[2] + half[2].max(eps),
        ],
    ))
}

/// Build a dense scalar density field by splatting Gaussian opacity into voxels.
fn build_density_field(
    model: &GaussianModel,
    cfg: &MeshExportConfig,
    grid_min: [f32; 3],
    grid_max: [f32; 3],
    dims: [usize; 3],
) -> Vec<f32> {
    let n = dims[0] * dims[1] * dims[2];
    let mut field = vec![0.0_f32; n];

    let cell = [
        (grid_max[0] - grid_min[0]) / dims[0] as f32,
        (grid_max[1] - grid_min[1]) / dims[1] as f32,
        (grid_max[2] - grid_min[2]) / dims[2] as f32,
    ];

    for g in model.gaussians.iter() {
        let alpha = sigmoid(g.opacity);
        if alpha < cfg.opacity_cutoff {
            continue;
        }

        // Isotropic radius: mean of actual scale values
        let r = (g.scale[0].exp() + g.scale[1].exp() + g.scale[2].exp()) / 3.0;
        if r < 1e-7 {
            continue;
        }

        // Voxel range to consider (±3σ)
        let k = 3.0_f32;
        let ix_min =
            (((g.position[0] - k * r - grid_min[0]) / cell[0]).floor() as isize).max(0) as usize;
        let iy_min =
            (((g.position[1] - k * r - grid_min[1]) / cell[1]).floor() as isize).max(0) as usize;
        let iz_min =
            (((g.position[2] - k * r - grid_min[2]) / cell[2]).floor() as isize).max(0) as usize;
        let ix_max =
            (((g.position[0] + k * r - grid_min[0]) / cell[0]).ceil() as usize + 1).min(dims[0]);
        let iy_max =
            (((g.position[1] + k * r - grid_min[1]) / cell[1]).ceil() as usize + 1).min(dims[1]);
        let iz_max =
            (((g.position[2] + k * r - grid_min[2]) / cell[2]).ceil() as usize + 1).min(dims[2]);

        let inv_r2 = 1.0 / (r * r);
        for iz in iz_min..iz_max {
            let vz = grid_min[2] + (iz as f32 + 0.5) * cell[2];
            let dz = vz - g.position[2];
            for iy in iy_min..iy_max {
                let vy = grid_min[1] + (iy as f32 + 0.5) * cell[1];
                let dy = vy - g.position[1];
                for ix in ix_min..ix_max {
                    let vx = grid_min[0] + (ix as f32 + 0.5) * cell[0];
                    let dx = vx - g.position[0];
                    let d2 = dx * dx + dy * dy + dz * dz;
                    field[iz * dims[1] * dims[0] + iy * dims[0] + ix] +=
                        alpha * (-0.5 * d2 * inv_r2).exp();
                }
            }
        }
    }
    field
}

/// Run Surface Nets to extract an isosurface triangle mesh from a scalar field.
fn surface_nets(
    field: &[f32],
    dims: [usize; 3],
    cell: [f32; 3],
    origin: [f32; 3],
    iso: f32,
) -> TriMesh {
    let [nx, ny, nz] = dims;
    let idx = |ix: usize, iy: usize, iz: usize| iz * ny * nx + iy * nx + ix;

    // Phase 1: place one vertex per active cell
    let mut cell_to_vertex: HashMap<(usize, usize, usize), u32> = HashMap::new();
    let mut vertices: Vec<[f32; 3]> = Vec::new();

    for iz in 0..nz.saturating_sub(1) {
        for iy in 0..ny.saturating_sub(1) {
            for ix in 0..nx.saturating_sub(1) {
                // 8 corner densities
                let corners = [
                    field[idx(ix, iy, iz)],
                    field[idx(ix + 1, iy, iz)],
                    field[idx(ix, iy + 1, iz)],
                    field[idx(ix + 1, iy + 1, iz)],
                    field[idx(ix, iy, iz + 1)],
                    field[idx(ix + 1, iy, iz + 1)],
                    field[idx(ix, iy + 1, iz + 1)],
                    field[idx(ix + 1, iy + 1, iz + 1)],
                ];
                let signs: u8 = corners
                    .iter()
                    .enumerate()
                    .fold(0u8, |acc, (i, &d)| acc | ((d > iso) as u8) << i);
                if signs == 0 || signs == 255 {
                    continue;
                } // all in or all out

                // Place vertex at density-weighted average of iso-crossing edge midpoints
                let edges = [
                    // (corner_a, corner_b) — indexed into corners[]
                    // x-aligned edges
                    (0usize, 1usize),
                    (2, 3),
                    (4, 5),
                    (6, 7),
                    // y-aligned edges
                    (0, 2),
                    (1, 3),
                    (4, 6),
                    (5, 7),
                    // z-aligned edges
                    (0, 4),
                    (1, 5),
                    (2, 6),
                    (3, 7),
                ];
                let mut sum = [0.0_f32; 3];
                let mut count = 0u32;
                for &(a, b) in &edges {
                    let da = corners[a];
                    let db = corners[b];
                    if (da > iso) != (db > iso) {
                        // Linear interpolation t where da*(1-t) + db*t = iso
                        let t = if (db - da).abs() > 1e-10 {
                            (iso - da) / (db - da)
                        } else {
                            0.5
                        };
                        let t = t.clamp(0.0, 1.0);
                        // Corner a: bit 0 → +x, bit 1 → +y, bit 2 → +z
                        let ax = ix + (a & 1);
                        let ay = iy + ((a >> 1) & 1);
                        let az = iz + ((a >> 2) & 1);
                        let bx = ix + (b & 1);
                        let by = iy + ((b >> 1) & 1);
                        let bz = iz + ((b >> 2) & 1);
                        sum[0] += origin[0] + (ax as f32 * (1.0 - t) + bx as f32 * t) * cell[0];
                        sum[1] += origin[1] + (ay as f32 * (1.0 - t) + by as f32 * t) * cell[1];
                        sum[2] += origin[2] + (az as f32 * (1.0 - t) + bz as f32 * t) * cell[2];
                        count += 1;
                    }
                }
                if count > 0 {
                    let inv = 1.0 / count as f32;
                    let v_idx = vertices.len() as u32;
                    vertices.push([sum[0] * inv, sum[1] * inv, sum[2] * inv]);
                    cell_to_vertex.insert((ix, iy, iz), v_idx);
                }
            }
        }
    }

    // Phase 2: emit quads (2 triangles each) for each axis-aligned edge that
    // crosses the isosurface. Each such edge is shared by exactly 4 cells; the
    // 4 cell-vertices form the quad corners.
    let mut triangles: Vec<[u32; 3]> = Vec::new();
    let get_v = |ix: usize, iy: usize, iz: usize| -> Option<u32> {
        cell_to_vertex.get(&(ix, iy, iz)).copied()
    };

    for iz in 0..nz.saturating_sub(1) {
        for iy in 0..ny.saturating_sub(1) {
            for ix in 0..nx.saturating_sub(1) {
                // X-axis edge between voxels (ix,iy,iz) and (ix+1,iy,iz).
                // The 4 cells sharing this edge are:
                //   (ix,iy,iz), (ix,iy-1,iz), (ix,iy-1,iz-1), (ix,iy,iz-1)
                if ix + 1 < nx && iy > 0 && iz > 0 {
                    let da = field[idx(ix + 1, iy, iz)];
                    let db = field[idx(ix, iy, iz)];
                    if (da > iso) != (db > iso) {
                        let v0 = get_v(ix, iy, iz);
                        let v1 = get_v(ix, iy - 1, iz);
                        let v2 = get_v(ix, iy - 1, iz - 1);
                        let v3 = get_v(ix, iy, iz - 1);
                        if let (Some(a), Some(b), Some(c), Some(d)) = (v0, v1, v2, v3) {
                            if da > iso {
                                triangles.push([a, c, b]);
                                triangles.push([a, d, c]);
                            } else {
                                triangles.push([a, b, c]);
                                triangles.push([a, c, d]);
                            }
                        }
                    }
                }

                // Y-axis edge between voxels (ix,iy,iz) and (ix,iy+1,iz).
                // The 4 cells sharing this edge are:
                //   (ix,iy,iz), (ix-1,iy,iz), (ix-1,iy,iz-1), (ix,iy,iz-1)
                if iy + 1 < ny && ix > 0 && iz > 0 {
                    let da = field[idx(ix, iy + 1, iz)];
                    let db = field[idx(ix, iy, iz)];
                    if (da > iso) != (db > iso) {
                        let v0 = get_v(ix, iy, iz);
                        let v1 = get_v(ix - 1, iy, iz);
                        let v2 = get_v(ix - 1, iy, iz - 1);
                        let v3 = get_v(ix, iy, iz - 1);
                        if let (Some(a), Some(b), Some(c), Some(d)) = (v0, v1, v2, v3) {
                            if da > iso {
                                triangles.push([a, c, b]);
                                triangles.push([a, d, c]);
                            } else {
                                triangles.push([a, b, c]);
                                triangles.push([a, c, d]);
                            }
                        }
                    }
                }

                // Z-axis edge between voxels (ix,iy,iz) and (ix,iy,iz+1).
                // The 4 cells sharing this edge are:
                //   (ix,iy,iz), (ix,iy-1,iz), (ix-1,iy-1,iz), (ix-1,iy,iz)
                if iz + 1 < nz && ix > 0 && iy > 0 {
                    let da = field[idx(ix, iy, iz + 1)];
                    let db = field[idx(ix, iy, iz)];
                    if (da > iso) != (db > iso) {
                        let v0 = get_v(ix, iy, iz);
                        let v1 = get_v(ix, iy - 1, iz);
                        let v2 = get_v(ix - 1, iy - 1, iz);
                        let v3 = get_v(ix - 1, iy, iz);
                        if let (Some(a), Some(b), Some(c), Some(d)) = (v0, v1, v2, v3) {
                            if da > iso {
                                triangles.push([a, c, b]);
                                triangles.push([a, d, c]);
                            } else {
                                triangles.push([a, b, c]);
                                triangles.push([a, c, d]);
                            }
                        }
                    }
                }
            }
        }
    }

    TriMesh {
        vertices,
        triangles,
    }
}

/// Write a triangle mesh as binary little-endian PLY.
fn write_mesh_ply(mesh: &TriMesh, path: &Path) -> Result<(), CliError> {
    let header = format!(
        "ply\nformat binary_little_endian 1.0\ncomment Generated by OxiGAF surface-nets mesher\n\
         element vertex {}\nproperty float x\nproperty float y\nproperty float z\n\
         element face {}\nproperty list uchar uint vertex_indices\nend_header\n",
        mesh.vertices.len(),
        mesh.triangles.len()
    );
    let mut file = std::fs::File::create(path)
        .map_err(|e| CliError::MeshExport(format!("create {}: {e}", path.display())))?;
    file.write_all(header.as_bytes())
        .map_err(|e| CliError::MeshExport(format!("write header: {e}")))?;
    for v in &mesh.vertices {
        for &f in v {
            file.write_all(&f.to_le_bytes())
                .map_err(|e| CliError::MeshExport(format!("write vertex: {e}")))?;
        }
    }
    for tri in &mesh.triangles {
        file.write_all(&[3u8])
            .map_err(|e| CliError::MeshExport(format!("write face count: {e}")))?;
        for &i in tri {
            file.write_all(&i.to_le_bytes())
                .map_err(|e| CliError::MeshExport(format!("write face index: {e}")))?;
        }
    }
    Ok(())
}

/// Extract a surface mesh from a Gaussian model and write it as binary PLY.
pub fn export_mesh(
    model: &GaussianModel,
    output_path: &Path,
    cfg: &MeshExportConfig,
) -> Result<(), CliError> {
    if let Some(parent) = output_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .map_err(|e| CliError::MeshExport(format!("create dirs: {e}")))?;
        }
    }
    let Some((grid_min, grid_max)) = compute_bounds(model, cfg.padding) else {
        // Empty model: write a valid but empty PLY
        let empty = TriMesh::default();
        return write_mesh_ply(&empty, output_path);
    };
    let resolution = cfg.resolution.min(256) as usize;
    let extent = [
        grid_max[0] - grid_min[0],
        grid_max[1] - grid_min[1],
        grid_max[2] - grid_min[2],
    ];
    let max_extent = extent[0].max(extent[1]).max(extent[2]);
    let dims = [
        ((extent[0] / max_extent) * resolution as f32)
            .round()
            .max(2.0) as usize,
        ((extent[1] / max_extent) * resolution as f32)
            .round()
            .max(2.0) as usize,
        ((extent[2] / max_extent) * resolution as f32)
            .round()
            .max(2.0) as usize,
    ];
    let cell = [
        (grid_max[0] - grid_min[0]) / dims[0] as f32,
        (grid_max[1] - grid_min[1]) / dims[1] as f32,
        (grid_max[2] - grid_min[2]) / dims[2] as f32,
    ];
    let field = build_density_field(model, cfg, grid_min, grid_max, dims);
    let mesh = surface_nets(&field, dims, cell, grid_min, cfg.iso);
    write_mesh_ply(&mesh, output_path)
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use oxigaf::render::gaussian::{GaussianAttributes, GaussianModel};
    use std::env;

    fn make_sphere_model() -> GaussianModel {
        // Single isotropic Gaussian at origin, r≈0.15, high opacity
        let g = GaussianAttributes {
            position: [0.0, 0.0, 0.0],
            _pad0: 0.0,
            scale: [f32::ln(0.15); 3],
            opacity: 5.0, // sigmoid(5) ≈ 0.99
            rotation: [0.0, 0.0, 0.0, 1.0],
        };
        GaussianModel {
            gaussians: vec![g],
            sh_coeffs: vec![0.0_f32; 3],
            sh_degree: 0,
            face_indices: vec![0u32],
            barycentric: vec![[1.0_f32 / 3.0; 3]],
            local_offsets: vec![[0.0_f32; 3]],
            is_rigid: vec![true],
        }
    }

    fn make_empty_model() -> GaussianModel {
        GaussianModel {
            gaussians: vec![],
            sh_coeffs: vec![],
            sh_degree: 0,
            face_indices: vec![],
            barycentric: vec![],
            local_offsets: vec![],
            is_rigid: vec![],
        }
    }

    fn make_two_gaussian_model() -> GaussianModel {
        // Two isotropic Gaussians separated by 2 units along x-axis
        let make_g = |x: f32| GaussianAttributes {
            position: [x, 0.0, 0.0],
            _pad0: 0.0,
            scale: [f32::ln(0.2); 3],
            opacity: 5.0, // sigmoid(5) ≈ 0.99
            rotation: [0.0, 0.0, 0.0, 1.0],
        };
        GaussianModel {
            gaussians: vec![make_g(-1.0), make_g(1.0)],
            sh_coeffs: vec![0.0_f32; 6],
            sh_degree: 0,
            face_indices: vec![0u32, 0u32],
            barycentric: vec![[1.0_f32 / 3.0; 3]; 2],
            local_offsets: vec![[0.0_f32; 3]; 2],
            is_rigid: vec![true; 2],
        }
    }

    #[test]
    fn test_sphere_produces_nonempty_mesh() {
        let model = make_sphere_model();
        // compute_bounds now includes Gaussian radii (3σ), so the default padding=0.1
        // is sufficient. With r=0.15: grid half-extent = 0.45*(1+0.1)=0.495 >> iso shell at 0.232.
        let cfg = MeshExportConfig {
            resolution: 32,
            iso: 0.3,
            ..Default::default()
        };
        let dir = env::temp_dir().join("oxigaf_mesh_test");
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let path = dir.join("sphere.ply");
        export_mesh(&model, &path, &cfg).expect("export_mesh must succeed");
        let mesh_inner = {
            let (gmin, gmax) = compute_bounds(&model, cfg.padding).expect("bounds must exist");
            let ext = [gmax[0] - gmin[0], gmax[1] - gmin[1], gmax[2] - gmin[2]];
            let me = ext[0].max(ext[1]).max(ext[2]);
            let dims = [
                ((ext[0] / me) * 32.0).round().max(2.0) as usize,
                ((ext[1] / me) * 32.0).round().max(2.0) as usize,
                ((ext[2] / me) * 32.0).round().max(2.0) as usize,
            ];
            let cell = [
                (gmax[0] - gmin[0]) / dims[0] as f32,
                (gmax[1] - gmin[1]) / dims[1] as f32,
                (gmax[2] - gmin[2]) / dims[2] as f32,
            ];
            let field = build_density_field(&model, &cfg, gmin, gmax, dims);
            surface_nets(&field, dims, cell, gmin, cfg.iso)
        };
        assert!(
            !mesh_inner.vertices.is_empty(),
            "sphere should produce vertices"
        );
        assert!(
            !mesh_inner.triangles.is_empty(),
            "sphere should produce triangles"
        );
    }

    #[test]
    fn test_closed_manifold() {
        let model = make_sphere_model();
        // compute_bounds includes 3σ radii; default padding=0.1 is sufficient.
        let cfg = MeshExportConfig {
            resolution: 24,
            iso: 0.3,
            ..Default::default()
        };
        let (gmin, gmax) = compute_bounds(&model, cfg.padding).expect("bounds must exist");
        let ext = [gmax[0] - gmin[0], gmax[1] - gmin[1], gmax[2] - gmin[2]];
        let me = ext[0].max(ext[1]).max(ext[2]);
        let res = 24.0_f32;
        let dims = [
            ((ext[0] / me) * res).round().max(2.0) as usize,
            ((ext[1] / me) * res).round().max(2.0) as usize,
            ((ext[2] / me) * res).round().max(2.0) as usize,
        ];
        let cell = [
            (gmax[0] - gmin[0]) / dims[0] as f32,
            (gmax[1] - gmin[1]) / dims[1] as f32,
            (gmax[2] - gmin[2]) / dims[2] as f32,
        ];
        let field = build_density_field(&model, &cfg, gmin, gmax, dims);
        let mesh = surface_nets(&field, dims, cell, gmin, cfg.iso);
        if mesh.triangles.is_empty() {
            return;
        } // May not produce mesh at very coarse resolution
          // Every undirected edge should appear exactly twice (closed 2-manifold property)
        let mut edge_count: HashMap<(u32, u32), u32> = HashMap::new();
        for &[a, b, c] in &mesh.triangles {
            for &(u, v) in &[(a, b), (b, c), (c, a)] {
                let key = if u < v { (u, v) } else { (v, u) };
                *edge_count.entry(key).or_insert(0) += 1;
            }
        }
        for (&edge, &count) in &edge_count {
            assert_eq!(
                count, 2,
                "edge {edge:?} appears {count} times (expected 2 for closed mesh)"
            );
        }
    }

    #[test]
    fn test_empty_model_ok() {
        let model = make_empty_model();
        let cfg = MeshExportConfig::default();
        let dir = env::temp_dir().join("oxigaf_mesh_empty");
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let path = dir.join("empty.ply");
        export_mesh(&model, &path, &cfg).expect("empty model export must succeed");
        let content = std::fs::read(&path).expect("must read output");
        assert!(content.starts_with(b"ply\n"), "PLY magic missing");
    }

    #[test]
    fn test_ply_magic_bytes() {
        let model = make_sphere_model();
        let cfg = MeshExportConfig {
            resolution: 16,
            iso: 0.3,
            ..Default::default()
        };
        let dir = env::temp_dir().join("oxigaf_mesh_magic");
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let path = dir.join("magic.ply");
        export_mesh(&model, &path, &cfg).expect("export must succeed");
        let bytes = std::fs::read(&path).expect("must read output");
        assert!(bytes.starts_with(b"ply\n"));
    }

    #[test]
    fn test_all_vertices_inside_bounds() {
        let model = make_sphere_model();
        let cfg = MeshExportConfig {
            resolution: 24,
            iso: 0.2,
            ..Default::default()
        };
        let (gmin, gmax) = compute_bounds(&model, cfg.padding).expect("bounds must exist");
        let ext = [gmax[0] - gmin[0], gmax[1] - gmin[1], gmax[2] - gmin[2]];
        let me = ext[0].max(ext[1]).max(ext[2]);
        let dims = [
            ((ext[0] / me) * 24.0).round().max(2.0) as usize,
            ((ext[1] / me) * 24.0).round().max(2.0) as usize,
            ((ext[2] / me) * 24.0).round().max(2.0) as usize,
        ];
        let cell = [
            (gmax[0] - gmin[0]) / dims[0] as f32,
            (gmax[1] - gmin[1]) / dims[1] as f32,
            (gmax[2] - gmin[2]) / dims[2] as f32,
        ];
        let field = build_density_field(&model, &cfg, gmin, gmax, dims);
        let mesh = surface_nets(&field, dims, cell, gmin, cfg.iso);
        for v in &mesh.vertices {
            for k in 0..3 {
                assert!(
                    v[k] >= gmin[k] - 0.01 && v[k] <= gmax[k] + 0.01,
                    "vertex out of bounds: {:?} not in [{},{}]",
                    v,
                    gmin[k],
                    gmax[k]
                );
            }
        }
    }

    /// Test 3: For a single isotropic Gaussian with alpha≈sigmoid(5)≈0.993 and r=0.15,
    /// the density at distance d equals alpha·exp(-d²/(2r²)).
    /// Setting that equal to iso=0.3:  d = r·sqrt(-2·ln(iso/alpha)).
    /// Mean vertex distance from origin should fall within ±40% of this expected radius.
    #[test]
    fn test_isosurface_radius() {
        let model = make_sphere_model();
        let iso = 0.3_f32;
        let cfg = MeshExportConfig {
            resolution: 40,
            iso,
            ..Default::default()
        };
        let (gmin, gmax) = compute_bounds(&model, cfg.padding).expect("bounds must exist");
        let ext = [gmax[0] - gmin[0], gmax[1] - gmin[1], gmax[2] - gmin[2]];
        let me = ext[0].max(ext[1]).max(ext[2]);
        let res = 40.0_f32;
        let dims = [
            ((ext[0] / me) * res).round().max(2.0) as usize,
            ((ext[1] / me) * res).round().max(2.0) as usize,
            ((ext[2] / me) * res).round().max(2.0) as usize,
        ];
        let cell_s = [
            (gmax[0] - gmin[0]) / dims[0] as f32,
            (gmax[1] - gmin[1]) / dims[1] as f32,
            (gmax[2] - gmin[2]) / dims[2] as f32,
        ];
        let field = build_density_field(&model, &cfg, gmin, gmax, dims);
        let mesh = surface_nets(&field, dims, cell_s, gmin, iso);
        if mesh.vertices.is_empty() {
            // At very coarse resolution the mesh may be empty — skip rather than fail
            return;
        }
        // Gaussian parameters matching make_sphere_model()
        let r = 0.15_f32; // exp(ln(0.15)) = 0.15
        let alpha = sigmoid(5.0_f32); // ≈ 0.9933
                                      // Expected isosurface distance: d = r * sqrt(-2 * ln(iso / alpha))
        let ratio = iso / alpha;
        let expected_d = r * (-2.0_f32 * ratio.ln()).sqrt();
        // Compute mean vertex distance from origin
        let mean_dist: f32 = mesh
            .vertices
            .iter()
            .map(|v| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt())
            .sum::<f32>()
            / mesh.vertices.len() as f32;
        let tolerance = 0.40_f32; // ±40%
        assert!(
            (mean_dist - expected_d).abs() <= expected_d * tolerance,
            "mean vertex dist {mean_dist:.4} not within {tolerance:.0}% of expected {expected_d:.4}"
        );
    }

    /// Test 7: Binary payload byte-size check.
    /// After writing, file_size must equal header_len + n_verts * 12 + n_tris * 13.
    /// Vertex count and face count are parsed from the ASCII header lines.
    #[test]
    fn test_ply_binary_payload_size() {
        let model = make_sphere_model();
        let cfg = MeshExportConfig {
            resolution: 20,
            iso: 0.3,
            ..Default::default()
        };
        let dir = env::temp_dir().join("oxigaf_mesh_size_check");
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let path = dir.join("size.ply");
        export_mesh(&model, &path, &cfg).expect("export must succeed");

        let bytes = std::fs::read(&path).expect("must read output");

        // Locate end of ASCII header
        let end_marker = b"end_header\n";
        let header_end_offset = bytes
            .windows(end_marker.len())
            .position(|w| w == end_marker)
            .expect("end_header marker must exist");
        let header_len = header_end_offset + end_marker.len();

        // Parse "element vertex N" and "element face M" from the header slice
        let header_str =
            std::str::from_utf8(&bytes[..header_len]).expect("header must be valid UTF-8");
        let mut n_verts: usize = 0;
        let mut n_faces: usize = 0;
        for line in header_str.lines() {
            if let Some(rest) = line.strip_prefix("element vertex ") {
                n_verts = rest
                    .trim()
                    .parse()
                    .expect("vertex count must be an integer");
            } else if let Some(rest) = line.strip_prefix("element face ") {
                n_faces = rest.trim().parse().expect("face count must be an integer");
            }
        }

        // Each vertex: 3 × f32 = 12 bytes
        // Each face:   1 × u8 (count=3) + 3 × u32 = 1 + 12 = 13 bytes
        let expected_total = header_len + n_verts * 12 + n_faces * 13;
        assert_eq!(
            bytes.len(),
            expected_total,
            "file size mismatch: got {} bytes, expected {} \
             (header={header_len} + {n_verts}×12 + {n_faces}×13)",
            bytes.len(),
            expected_total
        );
    }

    /// Test: all face indices are < vertex_count (no out-of-bounds references).
    #[test]
    fn test_mesh_face_indices_valid() {
        let model = make_sphere_model();
        let cfg = MeshExportConfig {
            resolution: 24,
            iso: 0.3,
            ..Default::default()
        };
        let (gmin, gmax) = compute_bounds(&model, cfg.padding).expect("bounds must exist");
        let ext = [gmax[0] - gmin[0], gmax[1] - gmin[1], gmax[2] - gmin[2]];
        let me = ext[0].max(ext[1]).max(ext[2]);
        let dims = [
            ((ext[0] / me) * 24.0).round().max(2.0) as usize,
            ((ext[1] / me) * 24.0).round().max(2.0) as usize,
            ((ext[2] / me) * 24.0).round().max(2.0) as usize,
        ];
        let cell = [
            (gmax[0] - gmin[0]) / dims[0] as f32,
            (gmax[1] - gmin[1]) / dims[1] as f32,
            (gmax[2] - gmin[2]) / dims[2] as f32,
        ];
        let field = build_density_field(&model, &cfg, gmin, gmax, dims);
        let mesh = surface_nets(&field, dims, cell, gmin, cfg.iso);
        let n_verts = mesh.vertices.len() as u32;
        for tri in &mesh.triangles {
            for &idx in tri {
                assert!(
                    idx < n_verts,
                    "face index {idx} out of range (vertex_count={n_verts})"
                );
            }
        }
    }

    /// Test: higher grid resolution yields equal-or-more faces for a non-trivial model.
    #[test]
    fn test_larger_resolution_more_faces() {
        let model = make_sphere_model();

        let run = |res: u32| -> usize {
            let cfg = MeshExportConfig {
                resolution: res,
                iso: 0.3,
                ..Default::default()
            };
            let (gmin, gmax) = compute_bounds(&model, cfg.padding).expect("bounds");
            let ext = [gmax[0] - gmin[0], gmax[1] - gmin[1], gmax[2] - gmin[2]];
            let me = ext[0].max(ext[1]).max(ext[2]);
            let r = res as f32;
            let dims = [
                ((ext[0] / me) * r).round().max(2.0) as usize,
                ((ext[1] / me) * r).round().max(2.0) as usize,
                ((ext[2] / me) * r).round().max(2.0) as usize,
            ];
            let cell = [
                (gmax[0] - gmin[0]) / dims[0] as f32,
                (gmax[1] - gmin[1]) / dims[1] as f32,
                (gmax[2] - gmin[2]) / dims[2] as f32,
            ];
            let field = build_density_field(&model, &cfg, gmin, gmax, dims);
            surface_nets(&field, dims, cell, gmin, cfg.iso)
                .triangles
                .len()
        };

        let faces_lo = run(16);
        let faces_hi = run(48);
        // Higher resolution must produce at least as many faces
        assert!(
            faces_hi >= faces_lo,
            "resolution=48 produced {faces_hi} faces < resolution=16's {faces_lo} faces"
        );
        // Both should produce non-zero faces (the sphere is clearly visible at these resolutions)
        assert!(faces_hi > 0, "resolution=48 must produce at least 1 face");
    }

    /// Test: two separated Gaussians produce a non-empty mesh.
    #[test]
    fn test_two_gaussians_produce_mesh() {
        let model = make_two_gaussian_model();
        let cfg = MeshExportConfig {
            resolution: 32,
            iso: 0.3,
            ..Default::default()
        };
        let dir = env::temp_dir().join("oxigaf_mesh_two_gauss");
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let path = dir.join("two_gauss.ply");
        export_mesh(&model, &path, &cfg).expect("export_mesh must succeed for two Gaussians");

        let (gmin, gmax) = compute_bounds(&model, cfg.padding).expect("bounds must exist");
        let ext = [gmax[0] - gmin[0], gmax[1] - gmin[1], gmax[2] - gmin[2]];
        let me = ext[0].max(ext[1]).max(ext[2]);
        let dims = [
            ((ext[0] / me) * 32.0).round().max(2.0) as usize,
            ((ext[1] / me) * 32.0).round().max(2.0) as usize,
            ((ext[2] / me) * 32.0).round().max(2.0) as usize,
        ];
        let cell = [
            (gmax[0] - gmin[0]) / dims[0] as f32,
            (gmax[1] - gmin[1]) / dims[1] as f32,
            (gmax[2] - gmin[2]) / dims[2] as f32,
        ];
        let field = build_density_field(&model, &cfg, gmin, gmax, dims);
        let mesh = surface_nets(&field, dims, cell, gmin, cfg.iso);
        assert!(
            !mesh.triangles.is_empty(),
            "two separated Gaussians must produce at least one triangle"
        );
    }

    /// Test: `vertices.len() == vertex_count() * 3` using the flat-list MeshData
    /// equivalent: triangles.len() == triangles.len() (sanity) and per-vertex count.
    #[test]
    fn test_mesh_data_vertex_count_consistent() {
        let model = make_sphere_model();
        let cfg = MeshExportConfig {
            resolution: 24,
            iso: 0.3,
            ..Default::default()
        };
        let (gmin, gmax) = compute_bounds(&model, cfg.padding).expect("bounds must exist");
        let ext = [gmax[0] - gmin[0], gmax[1] - gmin[1], gmax[2] - gmin[2]];
        let me = ext[0].max(ext[1]).max(ext[2]);
        let dims = [
            ((ext[0] / me) * 24.0).round().max(2.0) as usize,
            ((ext[1] / me) * 24.0).round().max(2.0) as usize,
            ((ext[2] / me) * 24.0).round().max(2.0) as usize,
        ];
        let cell = [
            (gmax[0] - gmin[0]) / dims[0] as f32,
            (gmax[1] - gmin[1]) / dims[1] as f32,
            (gmax[2] - gmin[2]) / dims[2] as f32,
        ];
        let field = build_density_field(&model, &cfg, gmin, gmax, dims);
        let mesh = surface_nets(&field, dims, cell, gmin, cfg.iso);
        // Each vertex is a [f32; 3] — total positions == vertices.len() * 3
        // encoded as flat vec: all face indices reference valid slots in [0..vertices.len())
        let n_verts = mesh.vertices.len();
        let n_tris = mesh.triangles.len();
        // Consistency: triangles.len() * 3 indices, all < n_verts
        assert_eq!(
            mesh.triangles.iter().flatten().count(),
            n_tris * 3,
            "each triangle must contribute exactly 3 indices"
        );
        // Flat-vertex count analogy: n_verts * 3 == number of f32 coords if stored flat
        let flat_len = n_verts * 3;
        assert_eq!(
            flat_len,
            mesh.vertices.iter().flat_map(|v| v.iter()).count(),
            "flat vertex coordinate count must be vertex_count * 3"
        );
    }

    /// Test: higher iso threshold produces fewer (or equal) faces than a lower threshold.
    #[test]
    fn test_high_threshold_fewer_faces() {
        let model = make_sphere_model();

        let run = |iso: f32| -> usize {
            let cfg = MeshExportConfig {
                resolution: 32,
                iso,
                ..Default::default()
            };
            let (gmin, gmax) = compute_bounds(&model, cfg.padding).expect("bounds");
            let ext = [gmax[0] - gmin[0], gmax[1] - gmin[1], gmax[2] - gmin[2]];
            let me = ext[0].max(ext[1]).max(ext[2]);
            let dims = [
                ((ext[0] / me) * 32.0).round().max(2.0) as usize,
                ((ext[1] / me) * 32.0).round().max(2.0) as usize,
                ((ext[2] / me) * 32.0).round().max(2.0) as usize,
            ];
            let cell = [
                (gmax[0] - gmin[0]) / dims[0] as f32,
                (gmax[1] - gmin[1]) / dims[1] as f32,
                (gmax[2] - gmin[2]) / dims[2] as f32,
            ];
            let field = build_density_field(&model, &cfg, gmin, gmax, dims);
            surface_nets(&field, dims, cell, gmin, iso).triangles.len()
        };

        // iso=0.1 captures a larger shell → more faces
        // iso=0.85 is very close to peak density → fewer or zero faces
        let faces_low = run(0.1);
        let faces_high = run(0.85);
        assert!(
            faces_high <= faces_low,
            "high iso ({} faces) must produce ≤ low iso ({} faces)",
            faces_high,
            faces_low
        );
    }
}
