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

    /// Check that every parallel array agrees with `gaussians.len()`.
    ///
    /// `GaussianModel`'s side arrays are independent `Vec`s that only some
    /// consumers read: `DeformPipeline`, `lod.rs` and `density.rs` index
    /// `face_indices` / `barycentric` / `local_offsets` / `is_rigid`, and the
    /// binding and rasterization paths index `sh_coeffs`. A model whose
    /// arrays have drifted out of step therefore renders perfectly and
    /// misbehaves much later, in whichever consumer happens to read the short
    /// array first. This is the cheap up-front check that turns that into one
    /// clear error.
    ///
    /// # What "consistent" means here
    ///
    /// * The four FLAME arrays are each either **empty** or exactly
    ///   `len()` long. Empty is legal and common: PLY and SafeTensors carry no
    ///   binding data, so a model loaded from either has no FLAME arrays at
    ///   all, and `binding::apply_binding` rejects that case on its own terms.
    ///   A *partially* filled array is never legal.
    /// * `sh_coeffs` is either empty (the GPU buffer is then zero-filled at
    ///   degree 0) or exactly `len() * (sh_degree + 1)² * 3` floats.
    /// * `sh_degree` is at most 3 — the largest degree with a defined SH
    ///   basis in the shaders and in [`crate::spherical_harmonics`].
    ///
    /// # Errors
    ///
    /// [`RenderError::MismatchedBufferSizes`] naming the expected and actual
    /// length, or [`RenderError::ValidationError`] for an out-of-range
    /// `sh_degree`.
    pub fn validate(&self) -> Result<(), RenderError> {
        let n = self.gaussians.len();

        if self.sh_degree > 3 {
            return Err(RenderError::ValidationError(format!(
                "GaussianModel: sh_degree must be in [0, 3], got {}",
                self.sh_degree
            )));
        }

        // Each side array is a strict all-or-nothing companion of `gaussians`.
        let side_arrays: [(&str, usize); 4] = [
            ("face_indices", self.face_indices.len()),
            ("barycentric", self.barycentric.len()),
            ("local_offsets", self.local_offsets.len()),
            ("is_rigid", self.is_rigid.len()),
        ];
        for (name, len) in side_arrays {
            if len != 0 && len != n {
                tracing::error!(
                    array = name,
                    expected = n,
                    actual = len,
                    "GaussianModel side array length does not match the Gaussian count"
                );
                return Err(RenderError::MismatchedBufferSizes {
                    expected: n,
                    actual: len,
                });
            }
        }

        let sh_stride = ((self.sh_degree + 1) * (self.sh_degree + 1) * 3) as usize;
        let expected_sh = n.saturating_mul(sh_stride);
        if !self.sh_coeffs.is_empty() && self.sh_coeffs.len() != expected_sh {
            tracing::error!(
                expected = expected_sh,
                actual = self.sh_coeffs.len(),
                sh_degree = self.sh_degree,
                "GaussianModel sh_coeffs length does not match the Gaussian count and degree"
            );
            return Err(RenderError::MismatchedBufferSizes {
                expected: expected_sh,
                actual: self.sh_coeffs.len(),
            });
        }

        Ok(())
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
    /// Where `rot_0..rot_3` = w,x,y,z (PLY convention), and `f_rest_*` is
    /// **channel-major**: all higher-order R coefficients, then all G, then
    /// all B (`features_rest.transpose(1, 2).flatten()` in the reference
    /// 3DGS Python implementation) - this crate's own in-memory
    /// `sh_coeffs` layout is coefficient-major RGB-interleaved (see
    /// [`GaussianModel::sh_coeffs`]), so the two orders are permuted on
    /// write and un-permuted on [`Self::load_ply`].
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

            // f_rest: remaining SH values, permuted from this crate's
            // internal coefficient-major RGB-interleaved layout
            // (`sh_coeffs[coefficient*3 + channel]`) to the PLY format's
            // channel-major layout (all R, then all G, then all B) - see
            // this function's doc comment. `num_rest` is always a multiple
            // of 3 (it is `((sh_degree+1)^2 - 1) * 3`).
            let num_rest_coeffs = num_rest / 3;
            for c in 0..3 {
                for j in 1..=num_rest_coeffs {
                    let internal_idx = sh_start + j * 3 + c;
                    write_f32_le(&mut w, self.sh_coeffs[internal_idx])?;
                }
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
    /// The vertex element's properties are read by name/offset rather than
    /// assumed to be in [`Self::save_ply`]'s exact order, so files written
    /// by other 3DGS tools (which may reorder properties or add/omit
    /// extras) load correctly as long as every property this loader
    /// requires (`x y z f_dc_0..2 [f_rest_*] opacity scale_0..2 rot_0..3`)
    /// is present with a supported scalar type. `f_rest_*` is un-permuted
    /// from the PLY channel-major order back to this crate's internal
    /// coefficient-major order - see [`Self::save_ply`]'s doc.
    ///
    /// FLAME binding fields are initialised to defaults:
    /// - `face_indices`: all 0
    /// - `barycentric`: all [1/3, 1/3, 1/3]
    /// - `local_offsets`: all [0, 0, 0]
    /// - `is_rigid`: all false
    pub fn load_ply(path: &Path) -> Result<Self, RenderError> {
        // Needed up front to sanity-check the header's declared vertex
        // count before pre-allocating anything sized by it (see below).
        let file_len = std::fs::metadata(path)
            .map_err(|e| RenderError::PlyIo(format!("Cannot stat file: {e}")))?
            .len();

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

        // Build the actual per-vertex byte layout from the header's own
        // property list, in file order. Real-world 3DGS exports frequently
        // reorder properties or add/omit extras (confidence, normals,
        // `uchar` colour); reading the body by looking up each required
        // property's offset - rather than assuming this crate's own
        // `write_ply_header` order and a fixed `float`-only property count
        // - means such files are read correctly instead of as silently
        // misaligned garbage.
        let (layout, bytes_per_vertex) = ply_vertex_layout(&header.properties);
        if bytes_per_vertex == 0 {
            return Err(RenderError::PlyIo(
                "PLY vertex element declares no properties".to_string(),
            ));
        }

        // Sanity-check the declared vertex count against the file's actual
        // size before pre-allocating: a tiny malformed/adversarial file
        // declaring e.g. `element vertex 4000000000` would otherwise cause
        // a multi-gigabyte allocation before a single body byte is read.
        let declared_body_bytes = n.checked_mul(bytes_per_vertex).ok_or_else(|| {
            RenderError::PlyIo(format!(
                "declared vertex count {n} overflows when computing required body size"
            ))
        })?;
        if declared_body_bytes as u64 > file_len {
            return Err(RenderError::PlyIo(format!(
                "PLY header declares {n} vertices ({declared_body_bytes} body bytes needed), \
                 but the file is only {file_len} bytes"
            )));
        }
        let sh_coeffs_capacity = n.checked_mul(sh_total).ok_or_else(|| {
            RenderError::PlyIo(format!(
                "sh_coeffs capacity overflow: n={n} * sh_total={sh_total}"
            ))
        })?;

        let mut gaussians = Vec::with_capacity(n);
        let mut sh_coeffs = Vec::with_capacity(sh_coeffs_capacity);
        let num_rest_coeffs = num_rest / 3;
        let mut rest_raw = vec![0.0_f32; num_rest];

        // --- Binary body ---
        let mut record = vec![0u8; bytes_per_vertex];
        for idx in 0..n {
            reader
                .read_exact(&mut record)
                .map_err(|e| RenderError::PlyIo(format!("Read error at vertex {idx}: {e}")))?;

            let px = ply_require_field(&layout, &record, "x")?;
            let py = ply_require_field(&layout, &record, "y")?;
            let pz = ply_require_field(&layout, &record, "z")?;

            // f_dc (coefficient 0, all 3 channels - not permuted).
            sh_coeffs.push(ply_require_field(&layout, &record, "f_dc_0")?);
            sh_coeffs.push(ply_require_field(&layout, &record, "f_dc_1")?);
            sh_coeffs.push(ply_require_field(&layout, &record, "f_dc_2")?);

            // f_rest: the file stores these channel-major (all R, then all
            // G, then all B - see `save_ply`'s doc), but this crate's
            // internal layout is coefficient-major RGB-interleaved
            // (`sh_coeffs[coefficient*3 + channel]`) - collect the raw
            // values in file order, then de-interleave into internal order.
            for (k, slot) in rest_raw.iter_mut().enumerate() {
                let name = format!("f_rest_{k}");
                *slot = ply_require_field(&layout, &record, &name)?;
            }
            for j in 1..=num_rest_coeffs {
                for c in 0..3 {
                    let ply_idx = c * num_rest_coeffs + (j - 1);
                    sh_coeffs.push(rest_raw[ply_idx]);
                }
            }

            let opacity = ply_require_field(&layout, &record, "opacity")?;

            let sx = ply_require_field(&layout, &record, "scale_0")?;
            let sy = ply_require_field(&layout, &record, "scale_1")?;
            let sz = ply_require_field(&layout, &record, "scale_2")?;

            // rotation: PLY w,x,y,z → struct x,y,z,w
            let rot_w = ply_require_field(&layout, &record, "rot_0")?;
            let rot_x = ply_require_field(&layout, &record, "rot_1")?;
            let rot_y = ply_require_field(&layout, &record, "rot_2")?;
            let rot_z = ply_require_field(&layout, &record, "rot_3")?;

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

/// A PLY scalar property type, together with its on-disk byte width.
///
/// Needed to correctly skip properties the loader does not care about
/// (e.g. normals, `uchar` colour) without misaligning the rest of the
/// per-vertex record - see [`ply_vertex_layout`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PlyScalarType {
    Float32,
    Float64,
    Int8,
    UInt8,
    Int16,
    UInt16,
    Int32,
    UInt32,
}

impl PlyScalarType {
    /// Parse a PLY type token (`float`, `float32`, `double`, `uchar`, …).
    fn parse(token: &str) -> Option<Self> {
        match token {
            "float" | "float32" => Some(Self::Float32),
            "double" | "float64" => Some(Self::Float64),
            "char" | "int8" => Some(Self::Int8),
            "uchar" | "uint8" => Some(Self::UInt8),
            "short" | "int16" => Some(Self::Int16),
            "ushort" | "uint16" => Some(Self::UInt16),
            "int" | "int32" => Some(Self::Int32),
            "uint" | "uint32" => Some(Self::UInt32),
            _ => None,
        }
    }

    /// On-disk width in bytes.
    fn byte_size(self) -> usize {
        match self {
            Self::Float32 | Self::Int32 | Self::UInt32 => 4,
            Self::Float64 => 8,
            Self::Int8 | Self::UInt8 => 1,
            Self::Int16 | Self::UInt16 => 2,
        }
    }

    /// Decode a little-endian value of this type from the start of `buf`
    /// (which must be at least `byte_size()` bytes) as an `f32`.
    fn read_le_as_f32(self, buf: &[u8]) -> f32 {
        match self {
            Self::Float32 => f32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]),
            Self::Float64 => f64::from_le_bytes([
                buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7],
            ]) as f32,
            Self::Int8 => buf[0] as i8 as f32,
            Self::UInt8 => buf[0] as f32,
            Self::Int16 => i16::from_le_bytes([buf[0], buf[1]]) as f32,
            Self::UInt16 => u16::from_le_bytes([buf[0], buf[1]]) as f32,
            Self::Int32 => i32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]) as f32,
            Self::UInt32 => u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]) as f32,
        }
    }
}

/// Compute each vertex property's byte offset (and type) within one vertex
/// record, plus the total per-vertex record size, from an ordered property
/// list (in file order, as recorded by [`parse_ply_header`]).
fn ply_vertex_layout(
    properties: &[(String, PlyScalarType)],
) -> (HashMap<&str, (usize, PlyScalarType)>, usize) {
    let mut offsets: HashMap<&str, (usize, PlyScalarType)> =
        HashMap::with_capacity(properties.len());
    let mut offset = 0usize;
    for (name, ty) in properties {
        offsets.insert(name.as_str(), (offset, *ty));
        offset += ty.byte_size();
    }
    (offsets, offset)
}

/// Read one named property's value out of a single vertex record, as an
/// `f32` regardless of its on-disk scalar type.
///
/// # Errors
///
/// Returns [`RenderError::PlyIo`] if `name` is not a property of this PLY
/// file's vertex element.
fn ply_require_field(
    layout: &HashMap<&str, (usize, PlyScalarType)>,
    record: &[u8],
    name: &str,
) -> Result<f32, RenderError> {
    let (offset, ty) = layout.get(name).ok_or_else(|| {
        RenderError::PlyIo(format!(
            "PLY vertex element is missing required property '{name}'"
        ))
    })?;
    let end = offset + ty.byte_size();
    let slice = record.get(*offset..end).ok_or_else(|| {
        RenderError::PlyIo(format!(
            "Internal error: property '{name}' offset {offset}..{end} exceeds record size {}",
            record.len()
        ))
    })?;
    Ok(ty.read_le_as_f32(slice))
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
    /// Every `vertex` element property in file order: `(name, type)`. Used
    /// by [`ply_vertex_layout`] to read the body by property name/offset
    /// instead of assuming a fixed order and property count.
    properties: Vec<(String, PlyScalarType)>,
}

/// Parse a PLY ASCII header from `r`, consuming exactly the header lines.
///
/// Returns the vertex count and full per-vertex property list needed to
/// read the binary body.
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
    let mut properties: Vec<(String, PlyScalarType)> = Vec::new();
    // Only the `vertex` element's properties belong in the per-vertex
    // record layout - a PLY file may declare other elements (e.g. `face`)
    // afterward, whose (possibly list-typed) properties must not be mixed
    // into it.
    let mut in_vertex_element = false;

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
        } else if let Some(rest) = trimmed.strip_prefix("element ") {
            let mut parts = rest.split_whitespace();
            let elem_name = parts.next().unwrap_or("");
            in_vertex_element = elem_name == "vertex";
            if in_vertex_element {
                let count_str = parts.next().ok_or_else(|| {
                    RenderError::PlyIo("Malformed 'element vertex' line".to_string())
                })?;
                vertex_count = Some(count_str.trim().parse::<usize>().map_err(|e| {
                    RenderError::PlyIo(format!("Invalid vertex count '{count_str}': {e}"))
                })?);
            }
        } else if in_vertex_element {
            if let Some(rest) = trimmed.strip_prefix("property ") {
                if rest.trim_start().starts_with("list ") {
                    return Err(RenderError::PlyIo(
                        "List-valued vertex properties are not supported".to_string(),
                    ));
                }
                let mut parts = rest.split_whitespace();
                let type_str = parts.next().ok_or_else(|| {
                    RenderError::PlyIo(format!("Malformed property line: {trimmed:?}"))
                })?;
                let name = parts.next().ok_or_else(|| {
                    RenderError::PlyIo(format!("Malformed property line: {trimmed:?}"))
                })?;
                let ty = PlyScalarType::parse(type_str).ok_or_else(|| {
                    RenderError::PlyIo(format!(
                        "Unsupported PLY property type '{type_str}' for property '{name}'"
                    ))
                })?;
                if name.starts_with("f_rest_") {
                    num_rest += 1;
                }
                properties.push((name.to_string(), ty));
            }
            // Comment / obj_info lines inside the vertex element are
            // silently skipped, as before.
        }
        // Lines outside the `vertex` element (including another element's
        // own property lines) are intentionally ignored.
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
        properties,
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
    fn test_ply_save_writes_f_rest_channel_major() {
        // Regression test for the interleaving bug: `f_rest_*` must be
        // written channel-major (all R higher-order coefficients, then all
        // G, then all B) to match the official 3DGS PLY convention
        // (`features_rest.transpose(1, 2).flatten()`), not interleaved
        // per-coefficient like this crate's internal `sh_coeffs` layout -
        // otherwise any degree>=1 model saved here renders with scrambled
        // view-dependent colour in SIBR/gsplat/supersplat.
        let sh_degree = 1u32; // 4 basis functions: 1 DC + 3 "rest"
        let sh_total = 12usize; // 4 * 3
        let mut sh_coeffs = vec![0.0_f32; sh_total];
        // DC (coefficient 0): arbitrary, unrelated to this check.
        sh_coeffs[0] = 9.0;
        sh_coeffs[1] = 9.1;
        sh_coeffs[2] = 9.2;
        // Coefficient 1 (R=1, G=2, B=3), coefficient-major internal layout:
        sh_coeffs[3] = 1.0;
        sh_coeffs[4] = 2.0;
        sh_coeffs[5] = 3.0;
        // Coefficient 2 (R=4, G=5, B=6):
        sh_coeffs[6] = 4.0;
        sh_coeffs[7] = 5.0;
        sh_coeffs[8] = 6.0;
        // Coefficient 3 (R=7, G=8, B=9):
        sh_coeffs[9] = 7.0;
        sh_coeffs[10] = 8.0;
        sh_coeffs[11] = 9.0;

        let third = 1.0_f32 / 3.0_f32;
        let model = GaussianModel {
            gaussians: vec![GaussianAttributes {
                position: [0.0, 0.0, 0.0],
                _pad0: 0.0,
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: [0.0, 0.0, 0.0],
                opacity: 0.0,
            }],
            sh_coeffs,
            sh_degree,
            face_indices: vec![0],
            barycentric: vec![[third, third, third]],
            local_offsets: vec![[0.0, 0.0, 0.0]],
            is_rigid: vec![false],
        };

        let tmp = std::env::temp_dir().join("test_ply_f_rest_channel_major.ply");
        model.save_ply(&tmp).expect("save_ply failed");

        let raw = std::fs::read(&tmp).expect("read raw ply bytes");
        let needle = b"end_header\n";
        let header_end = raw
            .windows(needle.len())
            .position(|w| w == needle)
            .expect("end_header not found")
            + needle.len();
        let body = &raw[header_end..];

        // x,y,z (12 bytes) + nx,ny,nz (12) + f_dc_0..2 (12) = 36 bytes
        // before f_rest_0.
        let f_rest_start = 36;
        let read_f32 = |offset: usize| -> f32 {
            let b = &body[offset..offset + 4];
            f32::from_le_bytes([b[0], b[1], b[2], b[3]])
        };

        // Channel-major: f_rest_0..2 = R of coefficients 1..3,
        // f_rest_3..5 = G, f_rest_6..8 = B.
        let expected = [1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0];
        for (k, &exp) in expected.iter().enumerate() {
            let got = read_f32(f_rest_start + k * 4);
            assert!(
                (got - exp).abs() < 1e-6,
                "f_rest_{k}: expected {exp}, got {got}"
            );
        }

        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn test_ply_load_tolerates_reordered_and_extra_properties() {
        // Regression test: a real-world PLY may reorder vertex properties,
        // add extras this crate does not need (e.g. a `uchar` confidence
        // value), or omit normals entirely - the loader must read such a
        // file correctly by property name/offset rather than assuming
        // this crate's own `write_ply_header` order and property set.
        let tmp = std::env::temp_dir().join("test_ply_reordered_properties.ply");

        let header = "ply\n\
            format binary_little_endian 1.0\n\
            element vertex 1\n\
            property uchar confidence\n\
            property float opacity\n\
            property float scale_0\n\
            property float scale_1\n\
            property float scale_2\n\
            property float rot_0\n\
            property float rot_1\n\
            property float rot_2\n\
            property float rot_3\n\
            property float x\n\
            property float y\n\
            property float z\n\
            property float f_dc_0\n\
            property float f_dc_1\n\
            property float f_dc_2\n\
            end_header\n";

        let mut body: Vec<u8> = Vec::new();
        body.push(200u8); // confidence (uchar, ignored by this loader)
        body.extend_from_slice(&0.75_f32.to_le_bytes()); // opacity
        body.extend_from_slice(&(-1.0_f32).to_le_bytes()); // scale_0
        body.extend_from_slice(&(-2.0_f32).to_le_bytes()); // scale_1
        body.extend_from_slice(&(-3.0_f32).to_le_bytes()); // scale_2
        body.extend_from_slice(&1.0_f32.to_le_bytes()); // rot_0 (w)
        body.extend_from_slice(&0.0_f32.to_le_bytes()); // rot_1 (x)
        body.extend_from_slice(&0.0_f32.to_le_bytes()); // rot_2 (y)
        body.extend_from_slice(&0.0_f32.to_le_bytes()); // rot_3 (z)
        body.extend_from_slice(&10.0_f32.to_le_bytes()); // x
        body.extend_from_slice(&20.0_f32.to_le_bytes()); // y
        body.extend_from_slice(&30.0_f32.to_le_bytes()); // z
        body.extend_from_slice(&0.5_f32.to_le_bytes()); // f_dc_0
        body.extend_from_slice(&0.6_f32.to_le_bytes()); // f_dc_1
        body.extend_from_slice(&0.7_f32.to_le_bytes()); // f_dc_2

        let mut file_bytes = header.as_bytes().to_vec();
        file_bytes.extend_from_slice(&body);
        std::fs::write(&tmp, &file_bytes).expect("write test ply");

        let loaded = GaussianModel::load_ply(&tmp).expect("load_ply should tolerate reordering");

        assert_eq!(loaded.gaussians.len(), 1);
        let g = loaded.gaussians[0];
        assert_eq!(g.position, [10.0, 20.0, 30.0]);
        assert!((g.opacity - 0.75).abs() < 1e-6);
        assert_eq!(g.scale, [-1.0, -2.0, -3.0]);
        assert_eq!(g.rotation, [0.0, 0.0, 0.0, 1.0]); // struct is x,y,z,w
        assert_eq!(loaded.sh_coeffs, vec![0.5, 0.6, 0.7]);

        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn test_ply_load_rejects_vertex_count_exceeding_file_size() {
        // Regression test: a tiny file declaring an enormous vertex count
        // must be rejected with a clear error instead of attempting a
        // multi-gigabyte pre-allocation before a single body byte is read.
        let tmp = std::env::temp_dir().join("test_ply_huge_vertex_count.ply");
        let header = "ply\n\
            format binary_little_endian 1.0\n\
            element vertex 4000000000\n\
            property float x\n\
            property float y\n\
            property float z\n\
            property float f_dc_0\n\
            property float f_dc_1\n\
            property float f_dc_2\n\
            property float opacity\n\
            property float scale_0\n\
            property float scale_1\n\
            property float scale_2\n\
            property float rot_0\n\
            property float rot_1\n\
            property float rot_2\n\
            property float rot_3\n\
            end_header\n";
        std::fs::write(&tmp, header).expect("write test ply");

        let result = GaussianModel::load_ply(&tmp);
        assert!(
            matches!(result, Err(RenderError::PlyIo(_))),
            "expected a PlyIo error for an over-declared vertex count, got {result:?}"
        );

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

    // --- GaussianModel::validate (F270) ---

    #[test]
    fn test_validate_accepts_a_well_formed_model() {
        for degree in 0..=3 {
            make_model(5, degree)
                .validate()
                .unwrap_or_else(|e| panic!("degree {degree} model must validate: {e}"));
        }
        // An empty model is trivially consistent.
        make_model(0, 0).validate().expect("empty model");
    }

    /// A PLY/SafeTensors model carries no FLAME binding data at all. Empty is
    /// legal; a *partially* filled array is the drift this check exists for.
    #[test]
    fn test_validate_allows_absent_flame_arrays_but_not_short_ones() {
        let mut model = make_model(6, 0);
        model.face_indices.clear();
        model.barycentric.clear();
        model.local_offsets.clear();
        model.is_rigid.clear();
        model.validate().expect("absent binding data is legal");

        for shorten in 0..4usize {
            let mut model = make_model(6, 0);
            match shorten {
                0 => model.face_indices.truncate(3),
                1 => model.barycentric.truncate(3),
                2 => model.local_offsets.truncate(3),
                _ => model.is_rigid.truncate(3),
            }
            let err = model
                .validate()
                .expect_err("a half-filled side array must be rejected");
            assert!(matches!(
                err,
                RenderError::MismatchedBufferSizes {
                    expected: 6,
                    actual: 3
                }
            ));
        }
    }

    /// A side array *longer* than the model is just as wrong as a short one:
    /// the extra entries silently belong to no Gaussian.
    #[test]
    fn test_validate_rejects_overlong_side_array() {
        let mut model = make_model(4, 0);
        model.is_rigid.push(true);
        assert!(matches!(
            model.validate(),
            Err(RenderError::MismatchedBufferSizes {
                expected: 4,
                actual: 5
            })
        ));
    }

    #[test]
    fn test_validate_rejects_sh_coeffs_of_the_wrong_stride() {
        // Degree-2 model (27 floats each) mislabelled as degree 3 (48 each).
        let mut model = make_model(4, 2);
        model.sh_degree = 3;
        let err = model
            .validate()
            .expect_err("an SH stride that does not match the degree must be rejected");
        assert!(matches!(
            err,
            RenderError::MismatchedBufferSizes {
                expected: 192,
                actual: 108
            }
        ));

        // Absent SH data is legal: the GPU buffer is zero-filled at degree 0.
        let mut model = make_model(4, 2);
        model.sh_coeffs.clear();
        model.validate().expect("absent SH data is legal");
    }

    #[test]
    fn test_validate_rejects_sh_degree_above_three() {
        let mut model = make_model(2, 0);
        model.sh_degree = 4;
        assert!(matches!(
            model.validate(),
            Err(RenderError::ValidationError(_))
        ));
    }
}
