//! 3D Gaussian model data structures and PLY I/O.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::Path;

use bytemuck::{Pod, Zeroable};
use safetensors::tensor::{Dtype, TensorView};
use safetensors::{serialize, SafeTensors};

use crate::RenderError;

/// Attributes of a single 3D Gaussian primitive.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GaussianAttributes {
    /// Position (x, y, z).
    pub position: [f32; 3],
    pub _pad0: f32,
    /// Rotation quaternion (x, y, z, w).
    pub rotation: [f32; 4],
    /// Log-scale (sx, sy, sz) — exponentiated before use.
    pub scale: [f32; 3],
    /// Sigmoid-inverse opacity.
    pub opacity: f32,
}

/// A collection of 3D Gaussians that form an avatar.
#[derive(Debug, Clone)]
pub struct GaussianModel {
    /// Per-Gaussian attributes.
    pub gaussians: Vec<GaussianAttributes>,
    /// Spherical harmonics coefficients per Gaussian `[N, C]`
    /// where C = (sh_degree+1)² × 3.
    pub sh_coeffs: Vec<f32>,
    /// SH degree (0–3).
    pub sh_degree: u32,

    // --- FLAME binding ---
    /// Face index on the FLAME mesh for each Gaussian.
    pub face_indices: Vec<u32>,
    /// Barycentric coordinates on the bound face.
    pub barycentric: Vec<[f32; 3]>,
    /// Learnable local offset from the mesh surface.
    pub local_offsets: Vec<[f32; 3]>,
    /// Whether each Gaussian is rigid (true) or flexible (false).
    pub is_rigid: Vec<bool>,
}

impl GaussianModel {
    /// Number of Gaussians.
    pub fn len(&self) -> usize {
        self.gaussians.len()
    }

    /// Whether the model is empty.
    pub fn is_empty(&self) -> bool {
        self.gaussians.is_empty()
    }

    /// Save this model to a binary little-endian PLY file in the standard 3DGS format.
    ///
    /// The file is compatible with the SIBR viewer and other 3DGS tools.
    /// FLAME binding fields (face_indices, barycentric, local_offsets, is_rigid)
    /// are not written to PLY.
    ///
    /// # Property order (per vertex)
    /// `x y z nx ny nz f_dc_0 f_dc_1 f_dc_2 [f_rest_*] opacity scale_0 scale_1 scale_2 rot_0 rot_1 rot_2 rot_3`
    ///
    /// Where `rot_0..rot_3` = w,x,y,z (PLY convention).
    pub fn save_ply(&self, path: &Path) -> Result<(), RenderError> {
        let n = self.gaussians.len();
        // C = (sh_degree+1)^2 * 3
        let sh_total = ((self.sh_degree + 1) * (self.sh_degree + 1) * 3) as usize;
        // f_rest count: total SH floats minus the 3 DC floats
        let num_rest = sh_total.saturating_sub(3);

        // Validate sh_coeffs length.
        if n > 0 && self.sh_coeffs.len() != n * sh_total {
            return Err(RenderError::PlyIo(format!(
                "sh_coeffs length mismatch: expected {}, got {}",
                n * sh_total,
                self.sh_coeffs.len()
            )));
        }

        let file = std::fs::File::create(path)
            .map_err(|e| RenderError::PlyIo(format!("Cannot create file: {e}")))?;
        let mut w = BufWriter::new(file);

        // --- ASCII header ---
        write_ply_header(&mut w, n, num_rest)?;

        // --- Binary body (little-endian f32 per property) ---
        for i in 0..n {
            let g = &self.gaussians[i];
            let sh_start = i * sh_total;

            // position
            write_f32_le(&mut w, g.position[0])?;
            write_f32_le(&mut w, g.position[1])?;
            write_f32_le(&mut w, g.position[2])?;

            // normals (always 0)
            write_f32_le(&mut w, 0.0_f32)?;
            write_f32_le(&mut w, 0.0_f32)?;
            write_f32_le(&mut w, 0.0_f32)?;

            // f_dc: first 3 SH values (degree-0 term)
            if sh_total >= 3 {
                write_f32_le(&mut w, self.sh_coeffs[sh_start])?;
                write_f32_le(&mut w, self.sh_coeffs[sh_start + 1])?;
                write_f32_le(&mut w, self.sh_coeffs[sh_start + 2])?;
            } else {
                // sh_degree=0 but sh_total<3 shouldn't happen, guard anyway
                write_f32_le(&mut w, 0.0_f32)?;
                write_f32_le(&mut w, 0.0_f32)?;
                write_f32_le(&mut w, 0.0_f32)?;
            }

            // f_rest: remaining SH values
            for k in 0..num_rest {
                write_f32_le(&mut w, self.sh_coeffs[sh_start + 3 + k])?;
            }

            // opacity (logit, raw)
            write_f32_le(&mut w, g.opacity)?;

            // scale (log-scale)
            write_f32_le(&mut w, g.scale[0])?;
            write_f32_le(&mut w, g.scale[1])?;
            write_f32_le(&mut w, g.scale[2])?;

            // rotation: PLY uses w,x,y,z; our struct stores x,y,z,w
            let [x, y, z, w_val] = g.rotation;
            write_f32_le(&mut w, w_val)?;
            write_f32_le(&mut w, x)?;
            write_f32_le(&mut w, y)?;
            write_f32_le(&mut w, z)?;
        }

        w.flush()
            .map_err(|e| RenderError::PlyIo(format!("Flush failed: {e}")))?;

        Ok(())
    }

    /// Load a `GaussianModel` from a binary little-endian PLY file in the standard 3DGS format.
    ///
    /// FLAME binding fields are initialised to defaults:
    /// - `face_indices`: all 0
    /// - `barycentric`: all [1/3, 1/3, 1/3]
    /// - `local_offsets`: all [0, 0, 0]
    /// - `is_rigid`: all false
    pub fn load_ply(path: &Path) -> Result<Self, RenderError> {
        let file = std::fs::File::open(path)
            .map_err(|e| RenderError::PlyIo(format!("Cannot open file: {e}")))?;
        let mut reader = BufReader::new(file);

        // --- Parse ASCII header ---
        let header = parse_ply_header(&mut reader)?;
        let n = header.vertex_count;
        let num_rest = header.num_rest;

        // Derive sh_degree from num_rest.
        // total SH floats = num_rest + 3 (the 3 DC values)
        // total SH floats = (sh_degree+1)^2 * 3
        let sh_total = num_rest + 3;
        let coeffs_per_channel = sh_total / 3;
        if sh_total % 3 != 0 {
            return Err(RenderError::PlyIo(format!(
                "SH coefficient count ({sh_total}) is not divisible by 3"
            )));
        }
        // coeffs_per_channel must be a perfect square (1, 4, 9, 16)
        let sh_degree = perfect_square_root(coeffs_per_channel).ok_or_else(|| {
            RenderError::PlyIo(format!(
                "SH coefficients per channel ({coeffs_per_channel}) is not a perfect square"
            ))
        })?;
        // sh_degree is (sqrt - 1)
        let sh_degree = sh_degree - 1;

        let mut gaussians = Vec::with_capacity(n);
        let mut sh_coeffs = Vec::with_capacity(n * sh_total);

        // --- Binary body ---
        for idx in 0..n {
            // position
            let px = read_f32_le(&mut reader, idx, "x")?;
            let py = read_f32_le(&mut reader, idx, "y")?;
            let pz = read_f32_le(&mut reader, idx, "z")?;

            // normals (read and discard)
            let _nx = read_f32_le(&mut reader, idx, "nx")?;
            let _ny = read_f32_le(&mut reader, idx, "ny")?;
            let _nz = read_f32_le(&mut reader, idx, "nz")?;

            // f_dc
            let dc0 = read_f32_le(&mut reader, idx, "f_dc_0")?;
            let dc1 = read_f32_le(&mut reader, idx, "f_dc_1")?;
            let dc2 = read_f32_le(&mut reader, idx, "f_dc_2")?;
            sh_coeffs.push(dc0);
            sh_coeffs.push(dc1);
            sh_coeffs.push(dc2);

            // f_rest
            for k in 0..num_rest {
                let v = read_f32_le(&mut reader, idx, &format!("f_rest_{k}"))?;
                sh_coeffs.push(v);
            }

            // opacity
            let opacity = read_f32_le(&mut reader, idx, "opacity")?;

            // scale
            let sx = read_f32_le(&mut reader, idx, "scale_0")?;
            let sy = read_f32_le(&mut reader, idx, "scale_1")?;
            let sz = read_f32_le(&mut reader, idx, "scale_2")?;

            // rotation: PLY w,x,y,z → struct x,y,z,w
            let rot_w = read_f32_le(&mut reader, idx, "rot_0")?;
            let rot_x = read_f32_le(&mut reader, idx, "rot_1")?;
            let rot_y = read_f32_le(&mut reader, idx, "rot_2")?;
            let rot_z = read_f32_le(&mut reader, idx, "rot_3")?;

            gaussians.push(GaussianAttributes {
                position: [px, py, pz],
                _pad0: 0.0,
                rotation: [rot_x, rot_y, rot_z, rot_w],
                scale: [sx, sy, sz],
                opacity,
            });
        }

        // Default FLAME binding fields
        let third = 1.0_f32 / 3.0_f32;
        let face_indices = vec![0u32; n];
        let barycentric = vec![[third, third, third]; n];
        let local_offsets = vec![[0.0_f32, 0.0_f32, 0.0_f32]; n];
        let is_rigid = vec![false; n];

        Ok(GaussianModel {
            gaussians,
            sh_coeffs,
            sh_degree,
            face_indices,
            barycentric,
            local_offsets,
            is_rigid,
        })
    }

    /// Save this model to a SafeTensors file.
    ///
    /// All per-Gaussian fields are preserved, including FLAME binding data.
    /// Metadata stores `sh_degree` and `num_gaussians`.
    ///
    /// # Tensor layout
    /// - `"positions"`: `[N, 3]` F32
    /// - `"rotations"`: `[N, 4]` F32 — stored as x,y,z,w (internal order)
    /// - `"scales"`:    `[N, 3]` F32
    /// - `"opacities"`: `[N, 1]` F32
    /// - `"sh_coeffs"`: `[N*C]`  F32 flat (C = (sh_degree+1)² × 3)
    /// - `"face_indices"`: `[N]`  U32
    /// - `"barycentric"`: `[N, 3]` F32
    /// - `"local_offsets"`: `[N, 3]` F32
    /// - `"is_rigid"`: `[N]`  U8 (0 = false, 1 = true)
    pub fn save_safetensors(&self, path: &Path) -> Result<(), RenderError> {
        let n = self.gaussians.len();

        // Flatten per-Gaussian attribute arrays.
        let mut positions = Vec::<f32>::with_capacity(n * 3);
        let mut rotations = Vec::<f32>::with_capacity(n * 4);
        let mut scales = Vec::<f32>::with_capacity(n * 3);
        let mut opacities = Vec::<f32>::with_capacity(n);

        for g in &self.gaussians {
            positions.extend_from_slice(&g.position);
            rotations.extend_from_slice(&g.rotation);
            scales.extend_from_slice(&g.scale);
            opacities.push(g.opacity);
        }

        // Flatten barycentric and local_offsets.
        let mut bary_flat = Vec::<f32>::with_capacity(n * 3);
        for b in &self.barycentric {
            bary_flat.extend_from_slice(b);
        }
        let mut offsets_flat = Vec::<f32>::with_capacity(n * 3);
        for o in &self.local_offsets {
            offsets_flat.extend_from_slice(o);
        }

        // Convert is_rigid to u8.
        let is_rigid_u8: Vec<u8> = self.is_rigid.iter().map(|&r| r as u8).collect();

        // Cast f32 slices to bytes via bytemuck.
        let pos_bytes: &[u8] = bytemuck::cast_slice(&positions);
        let rot_bytes: &[u8] = bytemuck::cast_slice(&rotations);
        let sc_bytes: &[u8] = bytemuck::cast_slice(&scales);
        let op_bytes: &[u8] = bytemuck::cast_slice(&opacities);
        let sh_bytes: &[u8] = bytemuck::cast_slice(&self.sh_coeffs);
        let fi_bytes: &[u8] = bytemuck::cast_slice(&self.face_indices);
        let bary_bytes: &[u8] = bytemuck::cast_slice(&bary_flat);
        let off_bytes: &[u8] = bytemuck::cast_slice(&offsets_flat);

        // Build TensorViews.
        // Empty models get shape [0, k] for matrix tensors and [0] for vectors.
        let positions_view = TensorView::new(Dtype::F32, vec![n, 3], pos_bytes)
            .map_err(|e| RenderError::SafetensorsIo(format!("positions TensorView error: {e}")))?;
        let rotations_view = TensorView::new(Dtype::F32, vec![n, 4], rot_bytes)
            .map_err(|e| RenderError::SafetensorsIo(format!("rotations TensorView error: {e}")))?;
        let scales_view = TensorView::new(Dtype::F32, vec![n, 3], sc_bytes)
            .map_err(|e| RenderError::SafetensorsIo(format!("scales TensorView error: {e}")))?;
        let opacities_view = TensorView::new(Dtype::F32, vec![n, 1], op_bytes)
            .map_err(|e| RenderError::SafetensorsIo(format!("opacities TensorView error: {e}")))?;
        let sh_coeffs_view = TensorView::new(Dtype::F32, vec![self.sh_coeffs.len()], sh_bytes)
            .map_err(|e| RenderError::SafetensorsIo(format!("sh_coeffs TensorView error: {e}")))?;
        let face_indices_view = TensorView::new(Dtype::U32, vec![n], fi_bytes).map_err(|e| {
            RenderError::SafetensorsIo(format!("face_indices TensorView error: {e}"))
        })?;
        let barycentric_view =
            TensorView::new(Dtype::F32, vec![n, 3], bary_bytes).map_err(|e| {
                RenderError::SafetensorsIo(format!("barycentric TensorView error: {e}"))
            })?;
        let local_offsets_view =
            TensorView::new(Dtype::F32, vec![n, 3], off_bytes).map_err(|e| {
                RenderError::SafetensorsIo(format!("local_offsets TensorView error: {e}"))
            })?;
        let is_rigid_view = TensorView::new(Dtype::U8, vec![n], &is_rigid_u8)
            .map_err(|e| RenderError::SafetensorsIo(format!("is_rigid TensorView error: {e}")))?;

        // Build metadata.
        let mut meta: HashMap<String, String> = HashMap::new();
        meta.insert("sh_degree".to_string(), self.sh_degree.to_string());
        meta.insert("num_gaussians".to_string(), n.to_string());

        // Assemble tensor list.
        let tensors: Vec<(&str, TensorView<'_>)> = vec![
            ("positions", positions_view),
            ("rotations", rotations_view),
            ("scales", scales_view),
            ("opacities", opacities_view),
            ("sh_coeffs", sh_coeffs_view),
            ("face_indices", face_indices_view),
            ("barycentric", barycentric_view),
            ("local_offsets", local_offsets_view),
            ("is_rigid", is_rigid_view),
        ];

        let bytes = serialize(tensors, Some(meta))
            .map_err(|e| RenderError::SafetensorsIo(format!("serialize error: {e}")))?;

        std::fs::write(path, &bytes)
            .map_err(|e| RenderError::SafetensorsIo(format!("write error: {e}")))?;

        Ok(())
    }

    /// Load a `GaussianModel` from a SafeTensors file.
    ///
    /// All fields, including FLAME binding data, are restored exactly as saved.
    pub fn load_safetensors(path: &Path) -> Result<Self, RenderError> {
        let bytes = std::fs::read(path)
            .map_err(|e| RenderError::SafetensorsIo(format!("read error: {e}")))?;

        let st = SafeTensors::deserialize(&bytes)
            .map_err(|e| RenderError::SafetensorsIo(format!("deserialize error: {e}")))?;

        // Read metadata header.
        let (_, header_meta) = SafeTensors::read_metadata(&bytes)
            .map_err(|e| RenderError::SafetensorsIo(format!("read_metadata error: {e}")))?;

        let meta_map = header_meta
            .metadata()
            .as_ref()
            .ok_or_else(|| RenderError::SafetensorsIo("missing metadata in file".to_string()))?;

        let sh_degree: u32 = meta_map
            .get("sh_degree")
            .ok_or_else(|| {
                RenderError::SafetensorsIo("metadata missing 'sh_degree' key".to_string())
            })?
            .parse::<u32>()
            .map_err(|e| {
                RenderError::SafetensorsIo(format!("invalid sh_degree in metadata: {e}"))
            })?;

        let n: usize = meta_map
            .get("num_gaussians")
            .ok_or_else(|| {
                RenderError::SafetensorsIo("metadata missing 'num_gaussians' key".to_string())
            })?
            .parse::<usize>()
            .map_err(|e| {
                RenderError::SafetensorsIo(format!("invalid num_gaussians in metadata: {e}"))
            })?;

        // Helper: get tensor bytes and verify expected byte count.
        let get_f32_data = |name: &str, expected_len: usize| -> Result<Vec<f32>, RenderError> {
            let tv = st.tensor(name).map_err(|e| {
                RenderError::SafetensorsIo(format!("tensor '{name}' not found: {e}"))
            })?;
            let data = tv.data();
            if data.len() != expected_len * std::mem::size_of::<f32>() {
                return Err(RenderError::SafetensorsIo(format!(
                    "tensor '{name}': expected {} f32 values ({} bytes), got {} bytes",
                    expected_len,
                    expected_len * std::mem::size_of::<f32>(),
                    data.len()
                )));
            }
            let floats: &[f32] = bytemuck::try_cast_slice::<u8, f32>(data).map_err(|e| {
                RenderError::SafetensorsIo(format!("tensor '{name}' alignment/cast error: {e}"))
            })?;
            Ok(floats.to_vec())
        };

        // positions [N, 3]
        let positions_flat = get_f32_data("positions", n * 3)?;
        // rotations [N, 4]
        let rotations_flat = get_f32_data("rotations", n * 4)?;
        // scales [N, 3]
        let scales_flat = get_f32_data("scales", n * 3)?;
        // opacities [N, 1]
        let opacities_flat = get_f32_data("opacities", n)?;
        // sh_coeffs — length is derived from tensor size, validated vs sh_degree
        let sh_total = ((sh_degree + 1) * (sh_degree + 1) * 3) as usize;
        let sh_coeffs = get_f32_data("sh_coeffs", n * sh_total)?;
        // barycentric [N, 3]
        let bary_flat = get_f32_data("barycentric", n * 3)?;
        // local_offsets [N, 3]
        let offsets_flat = get_f32_data("local_offsets", n * 3)?;

        // face_indices [N] as U32
        let fi_tv = st.tensor("face_indices").map_err(|e| {
            RenderError::SafetensorsIo(format!("tensor 'face_indices' not found: {e}"))
        })?;
        let fi_data = fi_tv.data();
        let expected_fi_bytes = n * std::mem::size_of::<u32>();
        if fi_data.len() != expected_fi_bytes {
            return Err(RenderError::SafetensorsIo(format!(
                "tensor 'face_indices': expected {} bytes, got {}",
                expected_fi_bytes,
                fi_data.len()
            )));
        }
        let face_indices: Vec<u32> = bytemuck::try_cast_slice::<u8, u32>(fi_data)
            .map_err(|e| {
                RenderError::SafetensorsIo(format!(
                    "tensor 'face_indices' alignment/cast error: {e}"
                ))
            })?
            .to_vec();

        // is_rigid [N] as U8
        let ir_tv = st
            .tensor("is_rigid")
            .map_err(|e| RenderError::SafetensorsIo(format!("tensor 'is_rigid' not found: {e}")))?;
        let ir_data = ir_tv.data();
        if ir_data.len() != n {
            return Err(RenderError::SafetensorsIo(format!(
                "tensor 'is_rigid': expected {} bytes, got {}",
                n,
                ir_data.len()
            )));
        }
        let is_rigid: Vec<bool> = ir_data.iter().map(|&b| b != 0).collect();

        // Reconstruct GaussianAttributes.
        let mut gaussians = Vec::with_capacity(n);
        for i in 0..n {
            let position = [
                positions_flat[i * 3],
                positions_flat[i * 3 + 1],
                positions_flat[i * 3 + 2],
            ];
            let rotation = [
                rotations_flat[i * 4],
                rotations_flat[i * 4 + 1],
                rotations_flat[i * 4 + 2],
                rotations_flat[i * 4 + 3],
            ];
            let scale = [
                scales_flat[i * 3],
                scales_flat[i * 3 + 1],
                scales_flat[i * 3 + 2],
            ];
            let opacity = opacities_flat[i];
            gaussians.push(GaussianAttributes {
                position,
                _pad0: 0.0,
                rotation,
                scale,
                opacity,
            });
        }

        // Reconstruct barycentric and local_offsets.
        let mut barycentric = Vec::with_capacity(n);
        for i in 0..n {
            barycentric.push([bary_flat[i * 3], bary_flat[i * 3 + 1], bary_flat[i * 3 + 2]]);
        }
        let mut local_offsets = Vec::with_capacity(n);
        for i in 0..n {
            local_offsets.push([
                offsets_flat[i * 3],
                offsets_flat[i * 3 + 1],
                offsets_flat[i * 3 + 2],
            ]);
        }

        Ok(GaussianModel {
            gaussians,
            sh_coeffs,
            sh_degree,
            face_indices,
            barycentric,
            local_offsets,
            is_rigid,
        })
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Write a single f32 as 4 bytes little-endian.
#[inline]
fn write_f32_le(w: &mut impl Write, v: f32) -> Result<(), RenderError> {
    w.write_all(&v.to_le_bytes())
        .map_err(|e| RenderError::PlyIo(format!("Write error: {e}")))
}

/// Read a single f32 from 4 bytes little-endian.
#[inline]
fn read_f32_le(r: &mut impl Read, idx: usize, prop: &str) -> Result<f32, RenderError> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf).map_err(|e| {
        RenderError::PlyIo(format!(
            "Read error at vertex {idx}, property '{prop}': {e}"
        ))
    })?;
    Ok(f32::from_le_bytes(buf))
}

/// Write the PLY ASCII header to `w`.
fn write_ply_header(w: &mut impl Write, n: usize, num_rest: usize) -> Result<(), RenderError> {
    let mut lines = Vec::new();
    lines.push("ply".to_string());
    lines.push("format binary_little_endian 1.0".to_string());
    lines.push(format!("element vertex {n}"));
    // position
    lines.push("property float x".to_string());
    lines.push("property float y".to_string());
    lines.push("property float z".to_string());
    // normals
    lines.push("property float nx".to_string());
    lines.push("property float ny".to_string());
    lines.push("property float nz".to_string());
    // DC SH
    lines.push("property float f_dc_0".to_string());
    lines.push("property float f_dc_1".to_string());
    lines.push("property float f_dc_2".to_string());
    // rest SH
    for k in 0..num_rest {
        lines.push(format!("property float f_rest_{k}"));
    }
    // opacity
    lines.push("property float opacity".to_string());
    // scale
    lines.push("property float scale_0".to_string());
    lines.push("property float scale_1".to_string());
    lines.push("property float scale_2".to_string());
    // rotation (w, x, y, z)
    lines.push("property float rot_0".to_string());
    lines.push("property float rot_1".to_string());
    lines.push("property float rot_2".to_string());
    lines.push("property float rot_3".to_string());
    lines.push("end_header".to_string());

    for line in &lines {
        w.write_all(line.as_bytes())
            .and_then(|_| w.write_all(b"\n"))
            .map_err(|e| RenderError::PlyIo(format!("Header write error: {e}")))?;
    }
    Ok(())
}

/// Parsed information from a PLY header.
struct PlyHeader {
    vertex_count: usize,
    /// Number of `f_rest_*` properties.
    num_rest: usize,
}

/// Parse a PLY ASCII header from `r`, consuming exactly the header lines.
///
/// Returns counts needed to read the binary body.
fn parse_ply_header(r: &mut impl BufRead) -> Result<PlyHeader, RenderError> {
    let mut line = String::new();

    // First line must be "ply"
    line.clear();
    r.read_line(&mut line)
        .map_err(|e| RenderError::PlyIo(format!("Header read error: {e}")))?;
    if line.trim() != "ply" {
        return Err(RenderError::PlyIo(format!(
            "Not a PLY file (first line: {:?})",
            line.trim()
        )));
    }

    let mut vertex_count: Option<usize> = None;
    let mut num_rest: usize = 0;
    let mut found_binary_le = false;

    loop {
        line.clear();
        let bytes = r
            .read_line(&mut line)
            .map_err(|e| RenderError::PlyIo(format!("Header read error: {e}")))?;
        if bytes == 0 {
            return Err(RenderError::PlyIo(
                "Unexpected EOF in PLY header".to_string(),
            ));
        }
        let trimmed = line.trim();

        if trimmed == "end_header" {
            break;
        } else if trimmed.starts_with("format binary_little_endian") {
            found_binary_le = true;
        } else if trimmed.starts_with("element vertex ") {
            let count_str = trimmed
                .strip_prefix("element vertex ")
                .ok_or_else(|| RenderError::PlyIo("Malformed element vertex line".to_string()))?;
            vertex_count = Some(count_str.trim().parse::<usize>().map_err(|e| {
                RenderError::PlyIo(format!("Invalid vertex count '{count_str}': {e}"))
            })?);
        } else if trimmed.starts_with("property float f_rest_") {
            num_rest += 1;
        }
        // All other property/comment/obj_info lines are silently skipped.
    }

    if !found_binary_le {
        return Err(RenderError::PlyIo(
            "Only binary_little_endian PLY format is supported".to_string(),
        ));
    }

    let vertex_count = vertex_count.ok_or_else(|| {
        RenderError::PlyIo("PLY header missing 'element vertex' line".to_string())
    })?;

    Ok(PlyHeader {
        vertex_count,
        num_rest,
    })
}

/// Return `Some(sqrt)` if `n` is a perfect square (1, 4, 9, 16, …), else `None`.
fn perfect_square_root(n: usize) -> Option<u32> {
    if n == 0 {
        return None;
    }
    let s = (n as f64).sqrt().round() as u32;
    if (s as usize) * (s as usize) == n {
        Some(s)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::FRAC_1_SQRT_2;

    /// Build a small `GaussianModel` for testing.
    fn make_model(n: usize, sh_degree: u32) -> GaussianModel {
        let sh_total = ((sh_degree + 1) * (sh_degree + 1) * 3) as usize;
        let third = 1.0_f32 / 3.0_f32;

        let mut gaussians = Vec::with_capacity(n);
        let mut sh_coeffs = Vec::with_capacity(n * sh_total);

        for i in 0..n {
            let fi = i as f32;
            gaussians.push(GaussianAttributes {
                position: [fi * 0.1, fi * 0.2, fi * 0.3],
                _pad0: 0.0,
                // Store as x,y,z,w
                rotation: [
                    FRAC_1_SQRT_2 * 0.5 * fi.sin(),
                    FRAC_1_SQRT_2 * 0.5 * fi.cos(),
                    0.1 * fi,
                    (1.0_f32
                        - (FRAC_1_SQRT_2 * 0.5 * fi.sin()).powi(2)
                        - (FRAC_1_SQRT_2 * 0.5 * fi.cos()).powi(2)
                        - (0.1 * fi).powi(2))
                    .max(0.0)
                    .sqrt(),
                ],
                scale: [fi * 0.01 - 3.0, fi * 0.02 - 2.0, fi * 0.03 - 1.0],
                opacity: -1.0 + fi * 0.1,
            });
            for k in 0..sh_total {
                sh_coeffs.push((i * sh_total + k) as f32 * 0.001);
            }
        }

        GaussianModel {
            gaussians,
            sh_coeffs,
            sh_degree,
            face_indices: vec![0u32; n],
            barycentric: vec![[third, third, third]; n],
            local_offsets: vec![[0.0, 0.0, 0.0]; n],
            is_rigid: vec![false; n],
        }
    }

    /// Compare two f32 values within tolerance.
    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn test_ply_roundtrip() {
        let original = make_model(8, 3);
        let tmp = std::env::temp_dir().join("test_ply_roundtrip.ply");

        original.save_ply(&tmp).expect("save_ply failed");
        let loaded = GaussianModel::load_ply(&tmp).expect("load_ply failed");

        assert_eq!(loaded.gaussians.len(), original.gaussians.len());
        assert_eq!(loaded.sh_degree, original.sh_degree);
        assert_eq!(loaded.sh_coeffs.len(), original.sh_coeffs.len());

        let tol = 1e-6_f32;
        for (i, (orig, load)) in original
            .gaussians
            .iter()
            .zip(loaded.gaussians.iter())
            .enumerate()
        {
            for c in 0..3 {
                assert!(
                    approx_eq(orig.position[c], load.position[c], tol),
                    "position[{i}][{c}] mismatch: {} vs {}",
                    orig.position[c],
                    load.position[c]
                );
            }
            for c in 0..4 {
                assert!(
                    approx_eq(orig.rotation[c], load.rotation[c], tol),
                    "rotation[{i}][{c}] mismatch: {} vs {}",
                    orig.rotation[c],
                    load.rotation[c]
                );
            }
            for c in 0..3 {
                assert!(
                    approx_eq(orig.scale[c], load.scale[c], tol),
                    "scale[{i}][{c}] mismatch: {} vs {}",
                    orig.scale[c],
                    load.scale[c]
                );
            }
            assert!(
                approx_eq(orig.opacity, load.opacity, tol),
                "opacity[{i}] mismatch: {} vs {}",
                orig.opacity,
                load.opacity
            );
        }

        for (i, (orig, load)) in original
            .sh_coeffs
            .iter()
            .zip(loaded.sh_coeffs.iter())
            .enumerate()
        {
            assert!(
                approx_eq(*orig, *load, tol),
                "sh_coeffs[{i}] mismatch: {} vs {}",
                orig,
                load
            );
        }

        // FLAME binding defaults
        assert!(loaded.face_indices.iter().all(|&v| v == 0));
        assert!(loaded.barycentric.iter().all(|b| {
            let third = 1.0_f32 / 3.0_f32;
            approx_eq(b[0], third, 1e-6)
                && approx_eq(b[1], third, 1e-6)
                && approx_eq(b[2], third, 1e-6)
        }));
        assert!(loaded.local_offsets.iter().all(|o| o == &[0.0, 0.0, 0.0]));
        assert!(loaded.is_rigid.iter().all(|&v| !v));

        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn test_ply_empty() {
        let original = make_model(0, 1);
        let tmp = std::env::temp_dir().join("test_ply_empty.ply");

        original
            .save_ply(&tmp)
            .expect("save_ply failed on empty model");
        let loaded = GaussianModel::load_ply(&tmp).expect("load_ply failed on empty file");

        assert_eq!(loaded.gaussians.len(), 0);
        assert_eq!(loaded.sh_coeffs.len(), 0);

        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn test_ply_sh_degree_0() {
        // sh_degree = 0: 1 coefficient per channel × 3 channels = 3 floats per Gaussian
        let original = make_model(5, 0);
        let tmp = std::env::temp_dir().join("test_ply_sh_degree_0.ply");

        original.save_ply(&tmp).expect("save_ply failed");
        let loaded = GaussianModel::load_ply(&tmp).expect("load_ply failed");

        assert_eq!(loaded.sh_degree, 0u32);
        // With sh_degree=0 there are no f_rest properties → sh_total = 3
        assert_eq!(loaded.sh_coeffs.len(), 5 * 3);

        let tol = 1e-6_f32;
        for (orig, load) in original.sh_coeffs.iter().zip(loaded.sh_coeffs.iter()) {
            assert!(
                approx_eq(*orig, *load, tol),
                "sh coeff mismatch: {orig} vs {load}"
            );
        }

        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn test_ply_sh_degree_1() {
        // sh_degree = 1: 4 coefficients per channel × 3 = 12 floats per Gaussian
        let original = make_model(4, 1);
        let tmp = std::env::temp_dir().join("test_ply_sh_degree_1.ply");

        original.save_ply(&tmp).expect("save_ply failed");
        let loaded = GaussianModel::load_ply(&tmp).expect("load_ply failed");

        assert_eq!(loaded.sh_degree, 1u32);
        assert_eq!(loaded.sh_coeffs.len(), 4 * 12);

        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn test_ply_sh_degree_2() {
        // sh_degree = 2: 9 coefficients per channel × 3 = 27 floats per Gaussian
        let original = make_model(3, 2);
        let tmp = std::env::temp_dir().join("test_ply_sh_degree_2.ply");

        original.save_ply(&tmp).expect("save_ply failed");
        let loaded = GaussianModel::load_ply(&tmp).expect("load_ply failed");

        assert_eq!(loaded.sh_degree, 2u32);
        assert_eq!(loaded.sh_coeffs.len(), 3 * 27);

        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn test_ply_rotation_quaternion_convention() {
        // Verify that the w,x,y,z (PLY) ↔ x,y,z,w (struct) swap is correct.
        let n = 1;
        let sh_degree = 0;
        let sh_total = 3;

        // Make a model with a known rotation
        let rotation_xyzw = [0.1_f32, 0.2_f32, 0.3_f32, 0.9274_f32]; // approx unit
        let gaussians = vec![GaussianAttributes {
            position: [0.0, 0.0, 0.0],
            _pad0: 0.0,
            rotation: rotation_xyzw,
            scale: [0.0, 0.0, 0.0],
            opacity: 0.0,
        }];
        let sh_coeffs = vec![0.0_f32; n * sh_total];
        let third = 1.0_f32 / 3.0_f32;

        let model = GaussianModel {
            gaussians,
            sh_coeffs,
            sh_degree,
            face_indices: vec![0],
            barycentric: vec![[third, third, third]],
            local_offsets: vec![[0.0, 0.0, 0.0]],
            is_rigid: vec![false],
        };

        let tmp = std::env::temp_dir().join("test_ply_rotation_convention.ply");
        model.save_ply(&tmp).expect("save failed");
        let loaded = GaussianModel::load_ply(&tmp).expect("load failed");

        let tol = 1e-5_f32;
        for (c, (&orig, &loaded_val)) in rotation_xyzw
            .iter()
            .zip(loaded.gaussians[0].rotation.iter())
            .enumerate()
        {
            assert!(
                approx_eq(orig, loaded_val, tol),
                "rotation[{c}] mismatch: {} vs {}",
                orig,
                loaded_val
            );
        }

        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn test_ply_large() {
        // Smoke test with a larger model to catch any off-by-one in binary body.
        let original = make_model(100, 3);
        let tmp = std::env::temp_dir().join("test_ply_large.ply");

        original.save_ply(&tmp).expect("save_ply failed");
        let loaded = GaussianModel::load_ply(&tmp).expect("load_ply failed");

        assert_eq!(loaded.gaussians.len(), 100);
        assert_eq!(loaded.sh_degree, 3u32);

        let tol = 1e-6_f32;
        for (orig, load) in original.gaussians.iter().zip(loaded.gaussians.iter()) {
            for c in 0..3 {
                assert!(approx_eq(orig.position[c], load.position[c], tol));
            }
        }

        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn test_perfect_square_root() {
        assert_eq!(perfect_square_root(1), Some(1));
        assert_eq!(perfect_square_root(4), Some(2));
        assert_eq!(perfect_square_root(9), Some(3));
        assert_eq!(perfect_square_root(16), Some(4));
        assert_eq!(perfect_square_root(0), None);
        assert_eq!(perfect_square_root(2), None);
        assert_eq!(perfect_square_root(5), None);
        assert_eq!(perfect_square_root(7), None);
    }

    // -------------------------------------------------------------------------
    // SafeTensors tests
    // -------------------------------------------------------------------------

    /// Build a model with distinct FLAME binding values for roundtrip tests.
    fn make_model_with_flame(n: usize, sh_degree: u32) -> GaussianModel {
        let sh_total = ((sh_degree + 1) * (sh_degree + 1) * 3) as usize;

        let mut gaussians = Vec::with_capacity(n);
        let mut sh_coeffs = Vec::with_capacity(n * sh_total);
        let mut face_indices = Vec::with_capacity(n);
        let mut barycentric = Vec::with_capacity(n);
        let mut local_offsets = Vec::with_capacity(n);
        let mut is_rigid = Vec::with_capacity(n);

        for i in 0..n {
            let fi = i as f32;
            gaussians.push(GaussianAttributes {
                position: [fi * 0.1, fi * 0.2, fi * 0.3],
                _pad0: 0.0,
                rotation: [
                    0.0_f32,
                    0.0_f32,
                    fi.sin() * 0.5,
                    (1.0 - (fi.sin() * 0.5).powi(2)).max(0.0).sqrt(),
                ],
                scale: [fi * 0.01 - 3.0, fi * 0.02 - 2.0, fi * 0.03 - 1.0],
                opacity: -1.0 + fi * 0.1,
            });
            for k in 0..sh_total {
                sh_coeffs.push((i * sh_total + k) as f32 * 0.001);
            }
            face_indices.push(i as u32 * 7 + 3);
            barycentric.push([
                fi * 0.1 + 0.1,
                fi * 0.2 + 0.2,
                1.0 - (fi * 0.1 + 0.1) - (fi * 0.2 + 0.2),
            ]);
            local_offsets.push([fi * 0.01, fi * 0.02, fi * 0.03]);
            is_rigid.push(i % 2 == 0);
        }

        GaussianModel {
            gaussians,
            sh_coeffs,
            sh_degree,
            face_indices,
            barycentric,
            local_offsets,
            is_rigid,
        }
    }

    /// Helper that checks two f32 values are within tolerance.
    fn st_approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn test_safetensors_roundtrip_basic() -> Result<(), RenderError> {
        let original = make_model_with_flame(8, 3);
        let tmp = std::env::temp_dir().join("oxigaf_st_roundtrip_basic.safetensors");

        original.save_safetensors(&tmp)?;
        let loaded = GaussianModel::load_safetensors(&tmp)?;

        let _ = std::fs::remove_file(&tmp);

        assert_eq!(loaded.gaussians.len(), original.gaussians.len());
        assert_eq!(loaded.sh_degree, original.sh_degree);
        assert_eq!(loaded.sh_coeffs.len(), original.sh_coeffs.len());
        assert_eq!(loaded.face_indices.len(), original.face_indices.len());
        assert_eq!(loaded.barycentric.len(), original.barycentric.len());
        assert_eq!(loaded.local_offsets.len(), original.local_offsets.len());
        assert_eq!(loaded.is_rigid.len(), original.is_rigid.len());

        Ok(())
    }

    #[test]
    fn test_safetensors_positions_roundtrip() -> Result<(), RenderError> {
        let original = make_model_with_flame(10, 1);
        let tmp = std::env::temp_dir().join("oxigaf_st_positions.safetensors");

        original.save_safetensors(&tmp)?;
        let loaded = GaussianModel::load_safetensors(&tmp)?;
        let _ = std::fs::remove_file(&tmp);

        let tol = 1e-6_f32;
        for (i, (orig, load)) in original
            .gaussians
            .iter()
            .zip(loaded.gaussians.iter())
            .enumerate()
        {
            for c in 0..3 {
                assert!(
                    st_approx_eq(orig.position[c], load.position[c], tol),
                    "position[{i}][{c}] mismatch: {} vs {}",
                    orig.position[c],
                    load.position[c]
                );
            }
            for c in 0..4 {
                assert!(
                    st_approx_eq(orig.rotation[c], load.rotation[c], tol),
                    "rotation[{i}][{c}] mismatch: {} vs {}",
                    orig.rotation[c],
                    load.rotation[c]
                );
            }
            for c in 0..3 {
                assert!(
                    st_approx_eq(orig.scale[c], load.scale[c], tol),
                    "scale[{i}][{c}] mismatch: {} vs {}",
                    orig.scale[c],
                    load.scale[c]
                );
            }
            assert!(
                st_approx_eq(orig.opacity, load.opacity, tol),
                "opacity[{i}] mismatch: {} vs {}",
                orig.opacity,
                load.opacity
            );
        }

        Ok(())
    }

    #[test]
    fn test_safetensors_sh_coeffs_roundtrip() -> Result<(), RenderError> {
        let original = make_model_with_flame(5, 3);
        let tmp = std::env::temp_dir().join("oxigaf_st_sh_coeffs.safetensors");

        original.save_safetensors(&tmp)?;
        let loaded = GaussianModel::load_safetensors(&tmp)?;
        let _ = std::fs::remove_file(&tmp);

        let tol = 1e-6_f32;
        for (i, (orig, load)) in original
            .sh_coeffs
            .iter()
            .zip(loaded.sh_coeffs.iter())
            .enumerate()
        {
            assert!(
                st_approx_eq(*orig, *load, tol),
                "sh_coeffs[{i}] mismatch: {orig} vs {load}"
            );
        }

        Ok(())
    }

    #[test]
    fn test_safetensors_sh_degree_metadata() -> Result<(), RenderError> {
        for degree in [0, 1, 2, 3] {
            let original = make_model_with_flame(4, degree);
            let tmp =
                std::env::temp_dir().join(format!("oxigaf_st_sh_degree_{degree}.safetensors"));

            original.save_safetensors(&tmp)?;
            let loaded = GaussianModel::load_safetensors(&tmp)?;
            let _ = std::fs::remove_file(&tmp);

            assert_eq!(
                loaded.sh_degree, degree,
                "sh_degree mismatch for degree {degree}"
            );
        }

        Ok(())
    }

    #[test]
    fn test_safetensors_flame_binding_roundtrip() -> Result<(), RenderError> {
        let original = make_model_with_flame(12, 2);
        let tmp = std::env::temp_dir().join("oxigaf_st_flame_binding.safetensors");

        original.save_safetensors(&tmp)?;
        let loaded = GaussianModel::load_safetensors(&tmp)?;
        let _ = std::fs::remove_file(&tmp);

        let tol = 1e-6_f32;
        for (i, (&orig_fi, &load_fi)) in original
            .face_indices
            .iter()
            .zip(loaded.face_indices.iter())
            .enumerate()
        {
            assert_eq!(orig_fi, load_fi, "face_indices[{i}] mismatch");
        }
        for (i, (orig_b, load_b)) in original
            .barycentric
            .iter()
            .zip(loaded.barycentric.iter())
            .enumerate()
        {
            for c in 0..3 {
                assert!(
                    st_approx_eq(orig_b[c], load_b[c], tol),
                    "barycentric[{i}][{c}] mismatch: {} vs {}",
                    orig_b[c],
                    load_b[c]
                );
            }
        }
        for (i, (orig_o, load_o)) in original
            .local_offsets
            .iter()
            .zip(loaded.local_offsets.iter())
            .enumerate()
        {
            for c in 0..3 {
                assert!(
                    st_approx_eq(orig_o[c], load_o[c], tol),
                    "local_offsets[{i}][{c}] mismatch: {} vs {}",
                    orig_o[c],
                    load_o[c]
                );
            }
        }
        for (i, (&orig_r, &load_r)) in original
            .is_rigid
            .iter()
            .zip(loaded.is_rigid.iter())
            .enumerate()
        {
            assert_eq!(orig_r, load_r, "is_rigid[{i}] mismatch");
        }

        Ok(())
    }

    #[test]
    fn test_safetensors_empty_model() -> Result<(), RenderError> {
        let original = make_model_with_flame(0, 1);
        let tmp = std::env::temp_dir().join("oxigaf_st_empty.safetensors");

        original.save_safetensors(&tmp)?;
        let loaded = GaussianModel::load_safetensors(&tmp)?;
        let _ = std::fs::remove_file(&tmp);

        assert_eq!(loaded.gaussians.len(), 0);
        assert_eq!(loaded.sh_coeffs.len(), 0);
        assert_eq!(loaded.face_indices.len(), 0);
        assert_eq!(loaded.barycentric.len(), 0);
        assert_eq!(loaded.local_offsets.len(), 0);
        assert_eq!(loaded.is_rigid.len(), 0);
        assert_eq!(loaded.sh_degree, 1);

        Ok(())
    }

    #[test]
    fn test_safetensors_large_model() -> Result<(), RenderError> {
        let original = make_model_with_flame(200, 3);
        let tmp = std::env::temp_dir().join("oxigaf_st_large.safetensors");

        original.save_safetensors(&tmp)?;
        let loaded = GaussianModel::load_safetensors(&tmp)?;
        let _ = std::fs::remove_file(&tmp);

        assert_eq!(loaded.gaussians.len(), 200);
        assert_eq!(loaded.sh_degree, 3);

        let tol = 1e-6_f32;
        for (orig, load) in original.gaussians.iter().zip(loaded.gaussians.iter()) {
            for c in 0..3 {
                assert!(st_approx_eq(orig.position[c], load.position[c], tol));
            }
        }

        Ok(())
    }

    #[test]
    fn test_safetensors_is_rigid_alternating() -> Result<(), RenderError> {
        let original = make_model_with_flame(16, 0);
        let tmp = std::env::temp_dir().join("oxigaf_st_is_rigid.safetensors");

        original.save_safetensors(&tmp)?;
        let loaded = GaussianModel::load_safetensors(&tmp)?;
        let _ = std::fs::remove_file(&tmp);

        for (i, (&orig_r, &load_r)) in original
            .is_rigid
            .iter()
            .zip(loaded.is_rigid.iter())
            .enumerate()
        {
            assert_eq!(orig_r, load_r, "is_rigid[{i}] mismatch");
            // Verify the alternating pattern set by make_model_with_flame.
            assert_eq!(load_r, i % 2 == 0, "is_rigid[{i}] wrong value");
        }

        Ok(())
    }

    #[test]
    fn test_safetensors_face_indices_values() -> Result<(), RenderError> {
        let original = make_model_with_flame(8, 0);
        let tmp = std::env::temp_dir().join("oxigaf_st_face_indices.safetensors");

        original.save_safetensors(&tmp)?;
        let loaded = GaussianModel::load_safetensors(&tmp)?;
        let _ = std::fs::remove_file(&tmp);

        for (i, (&orig_fi, &load_fi)) in original
            .face_indices
            .iter()
            .zip(loaded.face_indices.iter())
            .enumerate()
        {
            assert_eq!(orig_fi, load_fi, "face_indices[{i}] mismatch");
            // Verify the formula i * 7 + 3 from make_model_with_flame.
            assert_eq!(load_fi, i as u32 * 7 + 3, "face_indices[{i}] wrong value");
        }

        Ok(())
    }
}
