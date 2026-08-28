//! Model ⇄ flat-array plumbing shared by the tool command families.
//!
//! Nearly every library tool module takes a scene as flat `Vec<f32>` arrays
//! rather than as a [`GaussianModel`], and each one picks its own conventions
//! for opacity and colour. Rather than let every handler re-derive the
//! conversion (and get it subtly wrong), the decomposition lives here once.
//!
//! # Opacity space — read this before wiring a new module
//!
//! [`GaussianModel`] stores **logit** ("sigmoid-inverse") opacity, and so do
//! most of the flat-array modules. Two of them do not:
//!
//! | Module | Opacity |
//! |--------|---------|
//! | [`crate::export_ply`] | logit ([`FlatScene::opacity_logits`]) |
//! | [`crate::gaussian_compressor`] | logit |
//! | [`crate::gaussian_deduplicator`] | logit |
//! | [`crate::lod_generator`] | logit (it sigmoids internally) |
//! | [`crate::filter_gaussians`](mod@crate::filter_gaussians) | **probability** ([`FlatScene::opacity_probabilities`]) |
//! | [`crate::model_inspector`] | **probability** |
//!
//! Mixing the two compiles cleanly and silently produces wrong thresholds, so
//! the accessor names here are deliberately explicit.
//!
//! # Other conventions
//!
//! * Rotations are `(qx, qy, qz, qw)` — the model's own order, and the order
//!   `geometry_tools`, `export_ply` and the compressor all expect.
//! * Scales are log-space; consumers exponentiate.
//! * `sh_coeffs` is coefficient-major, RGB-interleaved
//!   (`sh_coeffs[i * total + coefficient * 3 + channel]`), so the DC term is
//!   the first three floats of each Gaussian's block and "rest" is the
//!   remainder in unchanged order.

use std::path::Path;

use anyhow::{Context, Result};

use oxigaf::render::gaussian::{GaussianAttributes, GaussianModel};

use crate::filter_gaussians::GaussianData;
use crate::model_inspector::InspectableModel;

/// SH band-0 normalisation constant, matching [`crate::export_pointcloud`].
pub const SH_C0: f32 = 0.282_094_79;

/// Logistic function: logit → probability.
#[must_use]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Inverse logistic function: probability → logit.
///
/// The input is clamped away from 0 and 1 so the result is always finite.
#[must_use]
pub fn logit(p: f32) -> f32 {
    let clamped = p.clamp(1e-7, 1.0 - 1e-7);
    (clamped / (1.0 - clamped)).ln()
}

/// Convert one DC spherical-harmonic coefficient to a linear `[0, 1]` colour.
#[must_use]
pub fn sh_dc_to_rgb(dc: f32) -> f32 {
    (0.5 + SH_C0 * dc).clamp(0.0, 1.0)
}

/// Total SH floats per Gaussian at `degree`: `(degree + 1)² × 3`.
#[must_use]
pub fn sh_total_for_degree(degree: u32) -> usize {
    let bands = (degree as usize).saturating_add(1);
    bands * bands * 3
}

/// SH "rest" floats per Gaussian at `degree` (everything past the DC term).
#[must_use]
pub fn sh_rest_for_degree(degree: u32) -> usize {
    sh_total_for_degree(degree).saturating_sub(3)
}

// ---------------------------------------------------------------------------
// FlatScene
// ---------------------------------------------------------------------------

/// A [`GaussianModel`] decomposed into the flat arrays the tool modules take.
#[derive(Debug, Clone)]
pub struct FlatScene {
    /// Number of Gaussians.
    pub n: usize,
    /// Spherical-harmonics degree of the source model.
    pub sh_degree: u32,
    /// Positions `[N × 3]`.
    pub positions: Vec<f32>,
    /// Rotations `[N × 4]` as `(qx, qy, qz, qw)`.
    pub rotations: Vec<f32>,
    /// Log-space scales `[N × 3]`.
    pub log_scales: Vec<f32>,
    /// Logit-space opacities `[N]`.
    pub opacity_logits: Vec<f32>,
    /// DC SH coefficients `[N × 3]`.
    pub sh_dc: Vec<f32>,
    /// Higher-order SH coefficients `[N × n_rest_per_gaussian]`.
    pub sh_rest: Vec<f32>,
    /// SH rest coefficients per Gaussian (0, 9, 24 or 45).
    pub n_rest_per_gaussian: usize,
}

impl FlatScene {
    /// Decompose a model into flat arrays.
    ///
    /// # Errors
    ///
    /// Returns an error when the model's `sh_coeffs` length disagrees with its
    /// declared `sh_degree`; padding it with zeros would silently fabricate
    /// colour, so a malformed model is rejected instead.
    pub fn from_model(model: &GaussianModel) -> Result<Self> {
        let n = model.gaussians.len();
        let sh_total = sh_total_for_degree(model.sh_degree);
        let expected = n
            .checked_mul(sh_total)
            .ok_or_else(|| anyhow::anyhow!("Model is too large to decompose ({n} Gaussians)"))?;
        if model.sh_coeffs.len() != expected {
            anyhow::bail!(
                "Malformed model: {} SH coefficients for {} Gaussians at degree {} \
                 (expected {expected}).",
                model.sh_coeffs.len(),
                n,
                model.sh_degree,
            );
        }
        let n_rest = sh_total.saturating_sub(3);

        let mut positions = Vec::with_capacity(n * 3);
        let mut rotations = Vec::with_capacity(n * 4);
        let mut log_scales = Vec::with_capacity(n * 3);
        let mut opacity_logits = Vec::with_capacity(n);
        let mut sh_dc = Vec::with_capacity(n * 3);
        let mut sh_rest = Vec::with_capacity(n * n_rest);

        for (i, gaussian) in model.gaussians.iter().enumerate() {
            positions.extend_from_slice(&gaussian.position);
            rotations.extend_from_slice(&gaussian.rotation);
            log_scales.extend_from_slice(&gaussian.scale);
            opacity_logits.push(gaussian.opacity);
            let base = i * sh_total;
            sh_dc.extend_from_slice(&model.sh_coeffs[base..base + 3]);
            sh_rest.extend_from_slice(&model.sh_coeffs[base + 3..base + sh_total]);
        }

        Ok(Self {
            n,
            sh_degree: model.sh_degree,
            positions,
            rotations,
            log_scales,
            opacity_logits,
            sh_dc,
            sh_rest,
            n_rest_per_gaussian: n_rest,
        })
    }

    /// Opacities as probabilities in `[0, 1]` — for `filter_gaussians` and
    /// `model_inspector`.
    #[must_use]
    pub fn opacity_probabilities(&self) -> Vec<f32> {
        self.opacity_logits.iter().copied().map(sigmoid).collect()
    }

    /// Re-interleave `sh_dc` and `sh_rest` back into the model's layout.
    #[must_use]
    pub fn sh_coeffs(&self) -> Vec<f32> {
        let rest = self.n_rest_per_gaussian;
        let mut out = Vec::with_capacity(self.n * (3 + rest));
        for i in 0..self.n {
            out.extend_from_slice(&self.sh_dc[i * 3..i * 3 + 3]);
            if rest > 0 {
                out.extend_from_slice(&self.sh_rest[i * rest..i * rest + rest]);
            }
        }
        out
    }

    /// Linear `[0, 1]` RGB per Gaussian `[N × 3]`, derived from the DC term.
    #[must_use]
    pub fn rgb_colors(&self) -> Vec<f32> {
        self.sh_dc.iter().copied().map(sh_dc_to_rgb).collect()
    }

    /// Build the [`crate::filter_gaussians`](mod@crate::filter_gaussians) view (probability opacity, raw DC
    /// colour).
    #[must_use]
    pub fn gaussian_data(&self) -> Vec<GaussianData> {
        (0..self.n)
            .map(|i| GaussianData {
                position: [
                    self.positions[i * 3],
                    self.positions[i * 3 + 1],
                    self.positions[i * 3 + 2],
                ],
                log_scale: [
                    self.log_scales[i * 3],
                    self.log_scales[i * 3 + 1],
                    self.log_scales[i * 3 + 2],
                ],
                rotation: [
                    self.rotations[i * 4],
                    self.rotations[i * 4 + 1],
                    self.rotations[i * 4 + 2],
                    self.rotations[i * 4 + 3],
                ],
                opacity: sigmoid(self.opacity_logits[i]),
                color: [
                    self.sh_dc[i * 3],
                    self.sh_dc[i * 3 + 1],
                    self.sh_dc[i * 3 + 2],
                ],
            })
            .collect()
    }

    /// Build the [`crate::model_inspector`] view (probability opacity,
    /// activated `[0, 1]` colours).
    ///
    /// # Errors
    ///
    /// Propagates [`crate::model_inspector::InspectorError`] when the arrays
    /// disagree in length, which [`FlatScene::from_model`] already rules out.
    pub fn inspectable(&self) -> Result<InspectableModel> {
        InspectableModel::new(
            self.positions.clone(),
            self.opacity_probabilities(),
            self.log_scales.clone(),
            self.rgb_colors(),
        )
        .map_err(anyhow::Error::from)
    }
}

// ---------------------------------------------------------------------------
// Reassembly
// ---------------------------------------------------------------------------

/// Flat arrays for [`model_from_arrays`].
#[derive(Debug, Clone)]
pub struct SceneArrays {
    /// Positions `[N × 3]`.
    pub positions: Vec<f32>,
    /// Rotations `[N × 4]` as `(qx, qy, qz, qw)`.
    pub rotations: Vec<f32>,
    /// Log-space scales `[N × 3]`.
    pub log_scales: Vec<f32>,
    /// Logit-space opacities `[N]` — these decide `N`.
    pub opacity_logits: Vec<f32>,
    /// SH coefficients `[N × (degree + 1)² × 3]`, coefficient-major.
    pub sh_coeffs: Vec<f32>,
    /// Spherical-harmonics degree.
    pub sh_degree: u32,
}

/// Rebuild a [`GaussianModel`] from flat arrays.
///
/// The FLAME binding fields (`face_indices`, `barycentric`, `local_offsets`,
/// `is_rigid`) are left empty: the modules that produce new arrays (dedup,
/// LOD, compression) reorder or drop Gaussians without reporting an index
/// map, so there is nothing correct to carry over. Use [`subset_model`]
/// instead whenever kept indices *are* available. PLY output carries no FLAME
/// binding either, so nothing is lost on the usual path.
///
/// # Errors
///
/// Returns an error when any array length disagrees with the Gaussian count
/// implied by `opacity_logits`.
pub fn model_from_arrays(arrays: SceneArrays) -> Result<GaussianModel> {
    let SceneArrays {
        positions,
        rotations,
        log_scales,
        opacity_logits,
        sh_coeffs,
        sh_degree,
    } = arrays;

    let n = opacity_logits.len();
    let sh_total = sh_total_for_degree(sh_degree);
    check_len("positions", positions.len(), n * 3)?;
    check_len("rotations", rotations.len(), n * 4)?;
    check_len("scales", log_scales.len(), n * 3)?;
    check_len("sh_coeffs", sh_coeffs.len(), n * sh_total)?;

    let mut gaussians = Vec::with_capacity(n);
    for i in 0..n {
        gaussians.push(GaussianAttributes {
            position: [positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]],
            _pad0: 0.0,
            rotation: [
                rotations[i * 4],
                rotations[i * 4 + 1],
                rotations[i * 4 + 2],
                rotations[i * 4 + 3],
            ],
            scale: [
                log_scales[i * 3],
                log_scales[i * 3 + 1],
                log_scales[i * 3 + 2],
            ],
            opacity: opacity_logits[i],
        });
    }

    Ok(GaussianModel {
        gaussians,
        sh_coeffs,
        sh_degree,
        face_indices: Vec::new(),
        barycentric: Vec::new(),
        local_offsets: Vec::new(),
        is_rigid: Vec::new(),
    })
}

fn check_len(what: &str, got: usize, expected: usize) -> Result<()> {
    if got != expected {
        anyhow::bail!("{what} array has {got} values, expected {expected}");
    }
    Ok(())
}

/// Build a new model from the Gaussians at `keep` (in the given order).
///
/// Unlike [`model_from_arrays`] this carries the FLAME binding fields across,
/// because the index map is known. Out-of-range indices are skipped rather
/// than panicking; auxiliary vectors whose length does not match the model's
/// Gaussian count are dropped whole, since they cannot be indexed safely.
#[must_use]
pub fn subset_model(model: &GaussianModel, keep: &[usize]) -> GaussianModel {
    let n = model.gaussians.len();
    let sh_total = sh_total_for_degree(model.sh_degree);
    let with_faces = model.face_indices.len() == n;
    let with_barycentric = model.barycentric.len() == n;
    let with_offsets = model.local_offsets.len() == n;
    let with_rigid = model.is_rigid.len() == n;

    let mut gaussians = Vec::with_capacity(keep.len());
    let mut sh_coeffs = Vec::with_capacity(keep.len() * sh_total);
    let mut face_indices = Vec::new();
    let mut barycentric = Vec::new();
    let mut local_offsets = Vec::new();
    let mut is_rigid = Vec::new();

    for &index in keep {
        let Some(gaussian) = model.gaussians.get(index) else {
            continue;
        };
        gaussians.push(*gaussian);
        let base = index * sh_total;
        match model.sh_coeffs.get(base..base + sh_total) {
            Some(block) => sh_coeffs.extend_from_slice(block),
            // Unreachable for a well-formed model; keeps the SH stride intact
            // rather than emitting a misaligned buffer.
            None => sh_coeffs.resize(sh_coeffs.len() + sh_total, 0.0),
        }
        if with_faces {
            face_indices.push(model.face_indices[index]);
        }
        if with_barycentric {
            barycentric.push(model.barycentric[index]);
        }
        if with_offsets {
            local_offsets.push(model.local_offsets[index]);
        }
        if with_rigid {
            is_rigid.push(model.is_rigid[index]);
        }
    }

    GaussianModel {
        gaussians,
        sh_coeffs,
        sh_degree: model.sh_degree,
        face_indices,
        barycentric,
        local_offsets,
        is_rigid,
    }
}

// ---------------------------------------------------------------------------
// File I/O
// ---------------------------------------------------------------------------

/// Load a Gaussian model, annotating failures with the path.
///
/// # Errors
///
/// Propagates [`crate::export::load_model`].
pub fn load_scene(path: &Path) -> Result<GaussianModel> {
    crate::export::load_model(path)
        .with_context(|| format!("Failed to load model: {}", path.display()))
}

/// Warn when a scene rebuilt by [`model_from_arrays`] is written somewhere
/// that could otherwise have carried the FLAME binding it had to drop.
///
/// PLY never stores the binding, so only a `.json` checkpoint loses anything
/// a reader would have looked for. Staying silent would let a user
/// unknowingly replace a bound checkpoint with an unbound one.
pub fn warn_if_binding_dropped(source: &GaussianModel, output: &Path) {
    let had_binding = !source.face_indices.is_empty()
        || !source.barycentric.is_empty()
        || !source.local_offsets.is_empty()
        || !source.is_rigid.is_empty();
    if !had_binding {
        return;
    }
    let is_checkpoint = output
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("json"));
    if is_checkpoint {
        tracing::warn!(
            "The source model carried FLAME binding data, but this operation rebuilds the \
             scene from flat arrays with no index map back to the original Gaussians, so {} \
             is written without face indices, barycentric coordinates, local offsets or \
             rigid flags.",
            output.display(),
        );
    }
}

/// Write a Gaussian model, choosing the writer by file extension.
///
/// `.ply` uses the standard 3DGS ASCII writer; `.json` writes a checkpoint.
///
/// # Errors
///
/// Returns an error for any other extension, or when the write fails.
pub fn save_scene(model: &GaussianModel, path: &Path) -> Result<()> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or_default()
        .to_lowercase();
    match ext.as_str() {
        "ply" => crate::export::export_ply(model, path)
            .with_context(|| format!("Failed to write PLY: {}", path.display())),
        "json" => crate::export::export_json_checkpoint(model, path)
            .with_context(|| format!("Failed to write checkpoint: {}", path.display())),
        other => anyhow::bail!(
            "Unsupported output format {other:?} for {}: expected .ply or .json",
            path.display()
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A three-Gaussian degree-1 model with distinguishable values.
    fn sample_model() -> GaussianModel {
        let sh_total = sh_total_for_degree(1);
        let mut sh_coeffs = Vec::with_capacity(3 * sh_total);
        for i in 0..3 {
            for c in 0..sh_total {
                sh_coeffs.push(i as f32 + c as f32 * 0.01);
            }
        }
        GaussianModel {
            gaussians: (0..3)
                .map(|i| GaussianAttributes {
                    position: [i as f32, i as f32 + 0.5, -(i as f32)],
                    _pad0: 0.0,
                    rotation: [0.0, 0.0, 0.0, 1.0],
                    scale: [-2.0, -2.5, -3.0],
                    opacity: i as f32 - 1.0,
                })
                .collect(),
            sh_coeffs,
            sh_degree: 1,
            face_indices: vec![10, 11, 12],
            barycentric: vec![[1.0, 0.0, 0.0]; 3],
            local_offsets: vec![[0.0, 0.0, 0.0]; 3],
            is_rigid: vec![true, false, true],
        }
    }

    #[test]
    fn sigmoid_and_logit_round_trip() {
        for probability in [0.01_f32, 0.25, 0.5, 0.75, 0.99] {
            let back = sigmoid(logit(probability));
            assert!(
                (back - probability).abs() < 1e-5,
                "{probability} round-tripped to {back}"
            );
        }
    }

    #[test]
    fn sh_counts_match_the_3dgs_table() {
        assert_eq!(sh_rest_for_degree(0), 0);
        assert_eq!(sh_rest_for_degree(1), 9);
        assert_eq!(sh_rest_for_degree(2), 24);
        assert_eq!(sh_rest_for_degree(3), 45);
    }

    #[test]
    fn flatten_then_rebuild_is_lossless() {
        let model = sample_model();
        let flat = FlatScene::from_model(&model).expect("decompose");
        assert_eq!(flat.n, 3);
        assert_eq!(flat.n_rest_per_gaussian, 9);
        assert_eq!(flat.sh_dc.len(), 9);
        assert_eq!(flat.sh_rest.len(), 27);

        let rebuilt = model_from_arrays(SceneArrays {
            positions: flat.positions.clone(),
            rotations: flat.rotations.clone(),
            log_scales: flat.log_scales.clone(),
            opacity_logits: flat.opacity_logits.clone(),
            sh_coeffs: flat.sh_coeffs(),
            sh_degree: flat.sh_degree,
        })
        .expect("rebuild");

        assert_eq!(rebuilt.len(), model.len());
        assert_eq!(rebuilt.sh_coeffs, model.sh_coeffs);
        for (a, b) in rebuilt.gaussians.iter().zip(model.gaussians.iter()) {
            assert_eq!(a.position, b.position);
            assert_eq!(a.rotation, b.rotation);
            assert_eq!(a.scale, b.scale);
            assert_eq!(a.opacity, b.opacity);
        }
    }

    #[test]
    fn opacity_accessors_do_not_share_a_space() {
        let model = sample_model();
        let flat = FlatScene::from_model(&model).expect("decompose");
        // Raw logits are what the model stores.
        assert_eq!(flat.opacity_logits, vec![-1.0, 0.0, 1.0]);
        // Probabilities are strictly inside (0, 1) and monotone.
        let probabilities = flat.opacity_probabilities();
        assert!((probabilities[1] - 0.5).abs() < 1e-6);
        assert!(probabilities[0] < probabilities[1] && probabilities[1] < probabilities[2]);
        // The filter view uses the probability space.
        let data = flat.gaussian_data();
        assert!((data[1].opacity - 0.5).abs() < 1e-6);
    }

    #[test]
    fn subset_model_keeps_flame_binding_data() {
        let model = sample_model();
        let subset = subset_model(&model, &[2, 0]);
        assert_eq!(subset.len(), 2);
        assert_eq!(subset.face_indices, vec![12, 10]);
        assert_eq!(subset.is_rigid, vec![true, true]);
        let sh_total = sh_total_for_degree(1);
        assert_eq!(subset.sh_coeffs.len(), 2 * sh_total);
        assert_eq!(subset.sh_coeffs[0], model.sh_coeffs[2 * sh_total]);
    }

    #[test]
    fn subset_model_skips_out_of_range_indices() {
        let model = sample_model();
        let subset = subset_model(&model, &[1, 99]);
        assert_eq!(subset.len(), 1);
        assert_eq!(subset.sh_coeffs.len(), sh_total_for_degree(1));
    }

    #[test]
    fn from_model_rejects_a_truncated_sh_buffer() {
        let mut model = sample_model();
        model.sh_coeffs.pop();
        assert!(FlatScene::from_model(&model).is_err());
    }

    #[test]
    fn save_scene_rejects_an_unknown_extension() {
        let model = sample_model();
        let path = std::env::temp_dir().join("oxigaf_model_io_save.xyz");
        assert!(save_scene(&model, &path).is_err());
        assert!(!path.exists());
    }

    #[test]
    fn save_scene_writes_ply() {
        let model = sample_model();
        let path = std::env::temp_dir().join("oxigaf_model_io_save.ply");
        let _ = std::fs::remove_file(&path);
        save_scene(&model, &path).expect("write ply");
        let text = std::fs::read_to_string(&path).expect("read back");
        assert!(text.starts_with("ply\n"));
        assert!(text.contains("element vertex 3"));
        let _ = std::fs::remove_file(&path);
    }
}
