//! Example: Demonstrate Gaussian mesh binding.
//!
//! Shows how Gaussians are bound to a FLAME mesh using face indices,
//! barycentric coordinates, and local offsets.
//!
//! Usage: cargo run --example flame_binding

use oxigaf_render::{GaussianAttributes, GaussianModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a simple triangle mesh (like a face surface).
    let vertices: Vec<[f32; 3]> = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [0.5, 0.5, 0.5],
    ];
    let faces: Vec<[u32; 3]> = vec![[0, 1, 2], [0, 1, 3]];
    let normals: Vec<[f32; 3]> = vec![
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ];

    let sh_degree = 0_u32;
    let sh_coeffs_per = ((sh_degree + 1) * (sh_degree + 1) * 3) as usize;

    // Create 4 Gaussians — two per face, specified using barycentric coordinates.
    let face_indices: Vec<u32> = vec![0, 0, 1, 1];
    let barycentric: Vec<[f32; 3]> = vec![
        [0.33, 0.33, 0.34],
        [0.50, 0.25, 0.25],
        [0.33, 0.33, 0.34],
        [0.50, 0.25, 0.25],
    ];
    let local_offsets: Vec<[f32; 3]> = vec![[0.0; 3]; 4];
    let is_rigid: Vec<bool> = vec![false; 4];

    // Compute world-space positions from barycentric interpolation.
    let positions: Vec<[f32; 3]> = face_indices
        .iter()
        .zip(barycentric.iter())
        .map(|(&fi, bary)| {
            let face = faces[fi as usize];
            let v0 = vertices[face[0] as usize];
            let v1 = vertices[face[1] as usize];
            let v2 = vertices[face[2] as usize];
            [
                bary[0] * v0[0] + bary[1] * v1[0] + bary[2] * v2[0],
                bary[0] * v0[1] + bary[1] * v1[1] + bary[2] * v2[1],
                bary[0] * v0[2] + bary[1] * v1[2] + bary[2] * v2[2],
            ]
        })
        .collect();

    let gaussians: Vec<GaussianAttributes> = positions
        .iter()
        .map(|&pos| GaussianAttributes {
            position: pos,
            _pad0: 0.0,
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [-4.0, -4.0, -4.0],
            opacity: 0.8_f32,
        })
        .collect();

    let n = gaussians.len();
    let model = GaussianModel {
        gaussians,
        sh_coeffs: vec![0.5_f32; n * sh_coeffs_per],
        sh_degree,
        face_indices: face_indices.clone(),
        barycentric: barycentric.clone(),
        local_offsets,
        is_rigid,
    };

    // Print binding summary.
    println!("GaussianModel bound to mesh:");
    println!("  {} Gaussians", model.gaussians.len());
    println!("  {} vertices in mesh", vertices.len());
    println!("  {} faces in mesh", faces.len());
    println!("  {} normals in mesh", normals.len());
    println!();

    for (i, g) in model.gaussians.iter().enumerate() {
        let fi = model.face_indices[i];
        let bary = model.barycentric[i];
        let off = model.local_offsets[i];
        let rigid = model.is_rigid[i];
        println!(
            "  Gaussian {:2}: face={}, bary=[{:.2},{:.2},{:.2}], \
             world_pos=[{:.3},{:.3},{:.3}], offset=[{:.2},{:.2},{:.2}], rigid={}",
            i,
            fi,
            bary[0],
            bary[1],
            bary[2],
            g.position[0],
            g.position[1],
            g.position[2],
            off[0],
            off[1],
            off[2],
            rigid
        );
    }

    println!();
    println!("Deform pipeline notes:");
    println!("  - DeformPipeline::deform() applies FLAME mesh deformations on the GPU.");
    println!("  - Each Gaussian's world position is recomputed from the deformed mesh surface.");
    println!("  - Rigid Gaussians move with the face; flexible ones also apply local_offsets.");
    println!("  - Use MultiViewRenderer for rendering the deformed model from multiple cameras.");

    // Round-trip through PLY to verify I/O.
    let tmp = std::env::temp_dir().join("flame_binding_demo.ply");
    model.save_ply(&tmp)?;
    let reloaded = GaussianModel::load_ply(&tmp)?;
    println!();
    println!(
        "PLY round-trip OK: saved {} → reloaded {} Gaussians",
        model.gaussians.len(),
        reloaded.gaussians.len()
    );

    Ok(())
}
