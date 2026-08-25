//! Checkpoint save / load.
//!
//! Persists the full training state (model parameters + optimiser moments +
//! iteration counter) as a JSON file.  This is intentionally simple — a
//! production implementation would use safetensors or a binary format.

use std::fs;
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};

use oxigaf_render::gaussian::{GaussianAttributes, GaussianModel};

use crate::metrics::{MetricEntry, MetricTracker};
use crate::optimizer::GaussianOptimizer;
use crate::{TrainerError, CHECKPOINT_VERSION};

// ---------------------------------------------------------------------------
// Serialisable types
// ---------------------------------------------------------------------------

/// Top-level checkpoint payload.
#[derive(Debug, Serialize, Deserialize)]
pub struct CheckpointData {
    /// Checkpoint format version for migration compatibility.
    #[serde(default = "default_version")]
    pub version: u32,

    /// Current training iteration.
    pub iteration: u32,

    // --- model ---
    pub positions: Vec<[f32; 3]>,
    pub rotations: Vec<[f32; 4]>,
    pub scales: Vec<[f32; 3]>,
    pub opacities: Vec<f32>,
    pub sh_coeffs: Vec<f32>,
    pub sh_degree: u32,
    pub face_indices: Vec<u32>,
    pub barycentric: Vec<[f32; 3]>,
    pub local_offsets: Vec<[f32; 3]>,
    pub is_rigid: Vec<bool>,

    // --- optimiser ---
    pub optimizer_groups: Vec<GroupCheckpoint>,

    // --- metrics history ---
    /// Metrics history for tracking training progress across resume.
    #[serde(default)]
    pub metrics_history: Vec<MetricEntry>,
}

/// Default version for checkpoints without version field (legacy).
fn default_version() -> u32 {
    1
}

/// Serialisable Adam state for a single parameter group.
#[derive(Debug, Serialize, Deserialize)]
pub struct GroupCheckpoint {
    pub name: String,
    pub m: Vec<f32>,
    pub v: Vec<f32>,
    pub t: u32,
}

/// Scan a slice of fixed-size arrays for NaN/Inf, reporting the outer index
/// (Gaussian index) and component index on failure.
fn check_nan_inf_array<const N: usize>(name: &str, arr: &[[f32; N]]) -> Result<(), TrainerError> {
    for (i, v) in arr.iter().enumerate() {
        for (j, &x) in v.iter().enumerate() {
            if x.is_nan() {
                return Err(TrainerError::NanDetected {
                    parameter: format!("{name}[{i}][{j}]"),
                    index: i,
                });
            }
            if x.is_infinite() {
                return Err(TrainerError::InfDetected {
                    parameter: format!("{name}[{i}][{j}]"),
                    index: i,
                });
            }
        }
    }
    Ok(())
}

/// Scan a flat scalar slice for NaN/Inf, reporting the element index on
/// failure.
fn check_nan_inf_flat(name: &str, arr: &[f32]) -> Result<(), TrainerError> {
    for (i, &x) in arr.iter().enumerate() {
        if x.is_nan() {
            return Err(TrainerError::NanDetected {
                parameter: name.to_string(),
                index: i,
            });
        }
        if x.is_infinite() {
            return Err(TrainerError::InfDetected {
                parameter: name.to_string(),
                index: i,
            });
        }
    }
    Ok(())
}

/// Maximum supported spherical-harmonics degree, matching
/// `InitConfig::validate`'s documented range. Rejecting anything above this
/// before computing `(degree + 1)^2` keeps the arithmetic well inside `u32`
/// (and `usize`) range even for a hostile/corrupted `sh_degree`.
const MAX_SH_DEGREE: u32 = 3;

impl CheckpointData {
    /// Validate checkpoint data for consistency and integrity.
    ///
    /// Checks:
    /// - Version compatibility
    /// - Array length consistency (including per-group optimizer state)
    /// - NaN/Inf detection across every numerical array: positions,
    ///   rotations, scales, opacities, SH coefficients, barycentric
    ///   coordinates, local offsets, and every optimizer group's Adam
    ///   moments (`m`, `v`)
    pub fn validate(&self) -> Result<(), TrainerError> {
        // Check version compatibility
        if self.version > CHECKPOINT_VERSION {
            return Err(TrainerError::CheckpointVersionMismatch {
                found: self.version,
                expected: CHECKPOINT_VERSION,
            });
        }

        let n = self.positions.len();

        // Validate array length consistency
        if self.rotations.len() != n {
            return Err(TrainerError::CheckpointDataMismatch {
                field: "rotations".into(),
                actual: self.rotations.len(),
                expected: n,
            });
        }

        if self.scales.len() != n {
            return Err(TrainerError::CheckpointDataMismatch {
                field: "scales".into(),
                actual: self.scales.len(),
                expected: n,
            });
        }

        if self.opacities.len() != n {
            return Err(TrainerError::CheckpointDataMismatch {
                field: "opacities".into(),
                actual: self.opacities.len(),
                expected: n,
            });
        }

        if self.face_indices.len() != n {
            return Err(TrainerError::CheckpointDataMismatch {
                field: "face_indices".into(),
                actual: self.face_indices.len(),
                expected: n,
            });
        }

        if self.barycentric.len() != n {
            return Err(TrainerError::CheckpointDataMismatch {
                field: "barycentric".into(),
                actual: self.barycentric.len(),
                expected: n,
            });
        }

        if self.local_offsets.len() != n {
            return Err(TrainerError::CheckpointDataMismatch {
                field: "local_offsets".into(),
                actual: self.local_offsets.len(),
                expected: n,
            });
        }

        if self.is_rigid.len() != n {
            return Err(TrainerError::CheckpointDataMismatch {
                field: "is_rigid".into(),
                actual: self.is_rigid.len(),
                expected: n,
            });
        }

        // Validate SH coefficients length. `sh_degree` comes straight from
        // untrusted JSON, so reject implausible degrees *before* computing
        // `(degree + 1)^2 * 3` — the old code performed that arithmetic in
        // u32 first and only widened to usize afterwards, so a corrupt
        // `sh_degree` near u32::MAX (or even a few tens of thousands) would
        // overflow: panic in debug, silent wraparound (and a bypassed or
        // spuriously-failing length check) in release.
        if self.sh_degree > MAX_SH_DEGREE {
            return Err(TrainerError::CheckpointDataMismatch {
                field: "sh_degree".into(),
                actual: self.sh_degree as usize,
                expected: MAX_SH_DEGREE as usize,
            });
        }
        let degree = self.sh_degree as usize;
        let sh_per = (degree + 1) * (degree + 1) * 3;
        let expected_sh = n * sh_per;
        if self.sh_coeffs.len() != expected_sh {
            return Err(TrainerError::CheckpointDataMismatch {
                field: "sh_coeffs".into(),
                actual: self.sh_coeffs.len(),
                expected: expected_sh,
            });
        }

        // NaN/Inf sweep. Previously this only scanned positions, scales,
        // opacities, and SH coefficients despite the rustdoc claiming full
        // coverage of "numerical arrays" — rotations (the most NaN-prone
        // parameter after normalization), barycentric coordinates, local
        // offsets, and every optimizer group's Adam moments were silently
        // skipped, so a checkpoint with e.g. NaN quaternions would pass
        // validation and poison the resumed run on the first optimizer step.
        check_nan_inf_array("positions", &self.positions)?;
        check_nan_inf_array("rotations", &self.rotations)?;
        check_nan_inf_array("scales", &self.scales)?;
        check_nan_inf_array("barycentric", &self.barycentric)?;
        check_nan_inf_array("local_offsets", &self.local_offsets)?;
        check_nan_inf_flat("opacities", &self.opacities)?;
        check_nan_inf_flat("sh_coeffs", &self.sh_coeffs)?;

        // Optimizer group consistency: each group's `m` and `v` must have
        // matching lengths, or a resumed Adam step reads uninitialised /
        // out-of-range moment data. Known group names (mirroring
        // `GaussianOptimizer::new`'s per-Gaussian widths) are additionally
        // checked against the model size `n`; unrecognised names are left
        // to `GaussianOptimizer::restore_states`, which already warns and
        // skips them.
        for group in &self.optimizer_groups {
            if group.m.len() != group.v.len() {
                return Err(TrainerError::CheckpointDataMismatch {
                    field: format!("optimizer_groups[{}].v", group.name),
                    actual: group.v.len(),
                    expected: group.m.len(),
                });
            }
            let expected_group_len = match group.name.as_str() {
                "position" | "scale" | "offset" => Some(n * 3),
                "rotation" => Some(n * 4),
                "opacity" => Some(n),
                "sh" => Some(n * sh_per),
                _ => None,
            };
            if let Some(expected) = expected_group_len {
                if group.m.len() != expected {
                    return Err(TrainerError::CheckpointDataMismatch {
                        field: format!("optimizer_groups[{}].m", group.name),
                        actual: group.m.len(),
                        expected,
                    });
                }
            }
            check_nan_inf_flat(&format!("optimizer_groups[{}].m", group.name), &group.m)?;
            check_nan_inf_flat(&format!("optimizer_groups[{}].v", group.name), &group.v)?;
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Conversion: live state → checkpoint
// ---------------------------------------------------------------------------

/// Snapshot the live model + optimiser into a [`CheckpointData`].
pub fn build_checkpoint(
    model: &GaussianModel,
    optimizer: &GaussianOptimizer,
    iteration: u32,
    metric_tracker: &MetricTracker,
) -> CheckpointData {
    let positions: Vec<[f32; 3]> = model.gaussians.iter().map(|g| g.position).collect();
    let rotations: Vec<[f32; 4]> = model.gaussians.iter().map(|g| g.rotation).collect();
    let scales: Vec<[f32; 3]> = model.gaussians.iter().map(|g| g.scale).collect();
    let opacities: Vec<f32> = model.gaussians.iter().map(|g| g.opacity).collect();

    let optimizer_groups: Vec<GroupCheckpoint> = optimizer
        .checkpoint_states()
        .into_iter()
        .map(|(name, m, v, t)| GroupCheckpoint { name, m, v, t })
        .collect();

    let metrics_history = metric_tracker.checkpoint_state();

    CheckpointData {
        version: CHECKPOINT_VERSION,
        iteration,
        positions,
        rotations,
        scales,
        opacities,
        sh_coeffs: model.sh_coeffs.clone(),
        sh_degree: model.sh_degree,
        face_indices: model.face_indices.clone(),
        barycentric: model.barycentric.clone(),
        local_offsets: model.local_offsets.clone(),
        is_rigid: model.is_rigid.clone(),
        optimizer_groups,
        metrics_history,
    }
}

// ---------------------------------------------------------------------------
// Conversion: checkpoint → live state
// ---------------------------------------------------------------------------

/// Reconstruct a [`GaussianModel`] from a checkpoint.
///
/// `data` is `pub` with all-public fields and derives `Deserialize`, so a
/// caller can build or deserialize one directly (bypassing
/// [`load_checkpoint`], the only path that calls [`CheckpointData::validate`]
/// first) and hand it here with mismatched array lengths. To avoid an
/// index-out-of-bounds panic on that input, the Gaussian count `n` is
/// clamped to the shortest of every per-Gaussian array —
/// `positions`/`rotations`/`scales`/`opacities`/`face_indices`/
/// `barycentric`/`local_offsets`/`is_rigid` — *and* how many whole
/// Gaussians' worth of SH data `sh_coeffs` actually holds at the (also
/// clamped-to-`0..=3`) `sh_degree`. Every one of those arrays (not just the
/// first four, and including `sh_coeffs`) is truncated to match in the
/// returned [`GaussianModel`], so the model's own internal length
/// invariants (`face_indices.len() == gaussians.len()`,
/// `sh_coeffs.len() == gaussians.len() * sh_per`, etc.) hold even when the
/// input was inconsistent — downstream per-Gaussian SH slicing
/// (`sh_coeffs[i * sh_per..]` for `i < n`) can then never run past the end
/// of the array. A mismatch is reported via `tracing::warn!` so it is
/// visible instead of silently losing data. For a checkpoint that has
/// already passed `validate()` (the normal path via `load_checkpoint`) all
/// lengths are already consistent and this is a no-op.
pub fn restore_model(data: &CheckpointData) -> GaussianModel {
    // Clamp sh_degree the same way `CheckpointData::validate` does *first*,
    // since `n` below also depends on how many whole Gaussians' worth of SH
    // coefficients are actually present at that degree.
    let degree = data.sh_degree.min(MAX_SH_DEGREE) as usize;
    let sh_per = (degree + 1) * (degree + 1) * 3; // always >= 3, never 0
    let n_from_sh = data.sh_coeffs.len() / sh_per;

    let n = data
        .positions
        .len()
        .min(data.rotations.len())
        .min(data.scales.len())
        .min(data.opacities.len())
        .min(data.face_indices.len())
        .min(data.barycentric.len())
        .min(data.local_offsets.len())
        .min(data.is_rigid.len())
        .min(n_from_sh);
    // `n * sh_per <= data.sh_coeffs.len()` always holds here (n <= n_from_sh
    // = sh_coeffs.len() / sh_per, integer division), so this slice is
    // exactly `n` whole Gaussians' worth of SH data — never out of bounds,
    // and never leaves a partial trailing Gaussian's coefficients in place.
    let sh_len = n * sh_per;

    if n != data.positions.len()
        || n != data.rotations.len()
        || n != data.scales.len()
        || n != data.opacities.len()
        || n != data.face_indices.len()
        || n != data.barycentric.len()
        || n != data.local_offsets.len()
        || n != data.is_rigid.len()
        || sh_len != data.sh_coeffs.len()
    {
        tracing::warn!(
            "restore_model: mismatched array lengths (positions={}, rotations={}, scales={}, \
             opacities={}, face_indices={}, barycentric={}, local_offsets={}, is_rigid={}, \
             sh_coeffs={} at degree {degree} implying {n_from_sh} whole Gaussians); truncating \
             every per-Gaussian array to {n} (sh_coeffs to {sh_len}) to avoid an out-of-bounds \
             panic or an internally-inconsistent GaussianModel. This checkpoint did not pass \
             CheckpointData::validate() — was it loaded via load_checkpoint()?",
            data.positions.len(),
            data.rotations.len(),
            data.scales.len(),
            data.opacities.len(),
            data.face_indices.len(),
            data.barycentric.len(),
            data.local_offsets.len(),
            data.is_rigid.len(),
            data.sh_coeffs.len(),
        );
    }
    let mut gaussians = Vec::with_capacity(n);
    for i in 0..n {
        gaussians.push(GaussianAttributes {
            position: data.positions[i],
            _pad0: 0.0,
            rotation: data.rotations[i],
            scale: data.scales[i],
            opacity: data.opacities[i],
        });
    }

    GaussianModel {
        gaussians,
        sh_coeffs: data.sh_coeffs[..sh_len].to_vec(),
        sh_degree: degree as u32,
        face_indices: data.face_indices[..n].to_vec(),
        barycentric: data.barycentric[..n].to_vec(),
        local_offsets: data.local_offsets[..n].to_vec(),
        is_rigid: data.is_rigid[..n].to_vec(),
    }
}

/// Restore optimiser Adam states from a checkpoint.
pub fn restore_optimizer(data: &CheckpointData, optimizer: &mut GaussianOptimizer) {
    let states: Vec<(String, Vec<f32>, Vec<f32>, u32)> = data
        .optimizer_groups
        .iter()
        .map(|g| (g.name.clone(), g.m.clone(), g.v.clone(), g.t))
        .collect();
    optimizer.restore_states(&states);
}

/// Restore metrics history from a checkpoint.
pub fn restore_metrics(data: &CheckpointData) -> MetricTracker {
    MetricTracker::from_history(data.metrics_history.clone())
}

// ---------------------------------------------------------------------------
// File I/O
// ---------------------------------------------------------------------------

/// Save a checkpoint to `path` as compact JSON, atomically.
///
/// The payload is streamed directly to a buffered file writer
/// (`serde_json::to_writer`) rather than first being built as one large
/// in-memory `String` — for a multi-hundred-thousand-Gaussian model the old
/// `serde_json::to_string` + `fs::write` approach held a ~100MB+ `String`
/// allocation on top of the source data on every single save.
///
/// The new content is written to a temporary file
/// (`path` with its extension replaced by `.tmp`) and `fsync`'d *before*
/// anything at `path` is touched, then the previous file at `path` (if any)
/// is rotated to `path.with_extension("backup.json")`, and finally the temp
/// file is renamed onto `path` — a rename is atomic within one filesystem.
/// This means an interruption (OOM kill, power loss, full disk) at any
/// point leaves `path` either fully absent/unchanged or fully replaced,
/// never truncated or partially written, and gives
/// [`try_load_checkpoint_with_fallback`] a real, freshly-rotated backup to
/// recover from.
///
/// Note: the JSON is intentionally compact, not pretty-printed — this
/// module's format is already an explicit "simple, human-inspectable"
/// tradeoff (see the module doc); pretty-printing many large float arrays
/// would inflate an already-large payload further for no benefit here.
pub fn save_checkpoint(path: &Path, data: &CheckpointData) -> Result<(), TrainerError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let tmp_path = path.with_extension("tmp");
    {
        let file = fs::File::create(&tmp_path)?;
        let mut writer = BufWriter::new(&file);
        serde_json::to_writer(&mut writer, data)?;
        writer.flush()?;
        file.sync_all()?;
    }

    // Rotate the previous checkpoint to a backup before replacing it, so a
    // fallback is always available even though the rename below is atomic.
    if path.exists() {
        let backup_path = path.with_extension("backup.json");
        fs::rename(path, &backup_path)?;
    }

    // Atomic within a filesystem: `path` either holds the complete old file
    // or the complete new file, never a partial write.
    fs::rename(&tmp_path, path)?;

    tracing::info!("Saved checkpoint to {}", path.display());
    Ok(())
}

/// Load a checkpoint from a JSON file at `path`.
///
/// Streams from a buffered file reader (`serde_json::from_reader`) rather
/// than reading the whole file into a `String` first and parsing that, so
/// the raw text and the parsed structure are never both fully resident in
/// memory at once.
///
/// Validates the checkpoint data after loading. Returns an error if:
/// - The file cannot be read
/// - JSON parsing fails (corrupted checkpoint)
/// - Data validation fails (version mismatch, array length mismatch, NaN/Inf)
pub fn load_checkpoint(path: &Path) -> Result<CheckpointData, TrainerError> {
    let file = fs::File::open(path).map_err(|e| {
        TrainerError::CheckpointCorrupted(format!(
            "Failed to read checkpoint file {}: {}",
            path.display(),
            e
        ))
    })?;
    let reader = BufReader::new(file);

    let data: CheckpointData = serde_json::from_reader(reader).map_err(|e| {
        TrainerError::CheckpointCorrupted(format!(
            "Failed to parse checkpoint JSON {}: {}",
            path.display(),
            e
        ))
    })?;

    // Validate the loaded data
    data.validate()?;

    tracing::info!(
        "Loaded checkpoint from {} (version {}, iteration {}, {} Gaussians)",
        path.display(),
        data.version,
        data.iteration,
        data.positions.len(),
    );

    Ok(data)
}

/// Try to load a checkpoint, falling back to a backup if the primary is corrupted.
///
/// Attempts to load `path` first. If that fails (missing or corrupted),
/// tries `path.with_extension("backup.json")`. [`save_checkpoint`] rotates
/// the previous file to that exact path before every save, so a backup is
/// available as soon as at least two successful saves have occurred.
pub fn try_load_checkpoint_with_fallback(path: &Path) -> Result<CheckpointData, TrainerError> {
    match load_checkpoint(path) {
        Ok(data) => Ok(data),
        Err(e) => {
            tracing::warn!(
                "Primary checkpoint {} is corrupted: {}. Trying backup...",
                path.display(),
                e
            );

            // Try backup
            let backup_path = path.with_extension("backup.json");
            if backup_path.exists() {
                load_checkpoint(&backup_path)
            } else {
                Err(e)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::OptimizerConfig;
    use oxigaf_render::gaussian::GaussianAttributes;

    fn tiny_model() -> GaussianModel {
        GaussianModel {
            gaussians: vec![GaussianAttributes {
                position: [1.0, 2.0, 3.0],
                _pad0: 0.0,
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: [-3.0; 3],
                opacity: -1.0,
            }],
            sh_coeffs: vec![0.5; 3],
            sh_degree: 0,
            face_indices: vec![0],
            barycentric: vec![[1.0, 0.0, 0.0]],
            local_offsets: vec![[0.0; 3]],
            is_rigid: vec![true],
        }
    }

    #[test]
    fn round_trip_checkpoint_data() -> Result<(), Box<dyn std::error::Error>> {
        let model = tiny_model();
        let opt = GaussianOptimizer::new(&OptimizerConfig::default(), &model);
        let mut tracker = MetricTracker::new();
        tracker.record(1, 25.0, 0.8, 0.05);
        tracker.record(2, 26.0, 0.82, 0.04);

        let ckpt = build_checkpoint(&model, &opt, 42, &tracker);

        let json = serde_json::to_string(&ckpt)?;
        let restored: CheckpointData = serde_json::from_str(&json)?;

        assert_eq!(restored.version, CHECKPOINT_VERSION);
        assert_eq!(restored.iteration, 42);
        assert_eq!(restored.positions.len(), 1);
        assert_eq!(restored.positions[0], [1.0, 2.0, 3.0]);
        assert_eq!(restored.metrics_history.len(), 2);

        // Validation should pass for valid checkpoint
        restored.validate()?;

        Ok(())
    }

    #[test]
    fn validate_detects_mismatched_lengths() {
        let mut data = CheckpointData {
            version: CHECKPOINT_VERSION,
            iteration: 0,
            positions: vec![[0.0; 3]; 2],
            rotations: vec![[0.0, 0.0, 0.0, 1.0]; 2],
            scales: vec![[0.0; 3]; 2],
            opacities: vec![0.0; 1], // Wrong length!
            sh_coeffs: vec![0.0; 6], // 2 * 3 for sh_degree 0
            sh_degree: 0,
            face_indices: vec![0; 2],
            barycentric: vec![[1.0, 0.0, 0.0]; 2],
            local_offsets: vec![[0.0; 3]; 2],
            is_rigid: vec![true; 2],
            optimizer_groups: vec![],
            metrics_history: vec![],
        };

        assert!(data.validate().is_err());

        // Fix the length
        data.opacities = vec![0.0; 2];
        assert!(data.validate().is_ok());
    }

    #[test]
    fn validate_detects_nan() {
        let data = CheckpointData {
            version: CHECKPOINT_VERSION,
            iteration: 0,
            positions: vec![[f32::NAN, 0.0, 0.0]],
            rotations: vec![[0.0, 0.0, 0.0, 1.0]],
            scales: vec![[0.0; 3]],
            opacities: vec![0.0],
            sh_coeffs: vec![0.0; 3],
            sh_degree: 0,
            face_indices: vec![0],
            barycentric: vec![[1.0, 0.0, 0.0]],
            local_offsets: vec![[0.0; 3]],
            is_rigid: vec![true],
            optimizer_groups: vec![],
            metrics_history: vec![],
        };

        let result = data.validate();
        assert!(result.is_err());
        assert!(matches!(result, Err(TrainerError::NanDetected { .. })));
    }

    #[test]
    fn checkpoint_file_roundtrip() -> Result<(), TrainerError> {
        let model = tiny_model();
        let opt = GaussianOptimizer::new(&OptimizerConfig::default(), &model);
        let mut tracker = MetricTracker::new();
        tracker.record(50, 28.0, 0.85, 0.03);
        tracker.record(100, 30.0, 0.90, 0.02);

        let ckpt = build_checkpoint(&model, &opt, 100, &tracker);

        // Use temp directory for test file
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_checkpoint_roundtrip.json");

        // Save
        save_checkpoint(&path, &ckpt)?;

        // Load
        let loaded = load_checkpoint(&path)?;

        assert_eq!(loaded.version, CHECKPOINT_VERSION);
        assert_eq!(loaded.iteration, 100);
        assert_eq!(loaded.positions.len(), 1);
        assert_eq!(loaded.metrics_history.len(), 2);
        assert_eq!(loaded.metrics_history[1].iteration, 100);

        // Cleanup
        let _ = std::fs::remove_file(&path);

        Ok(())
    }

    // ── Regression tests ─────────────────────────────────────────────────

    fn valid_data(n: usize) -> CheckpointData {
        CheckpointData {
            version: CHECKPOINT_VERSION,
            iteration: 0,
            positions: vec![[0.0; 3]; n],
            rotations: vec![[0.0, 0.0, 0.0, 1.0]; n],
            scales: vec![[0.0; 3]; n],
            opacities: vec![0.0; n],
            sh_coeffs: vec![0.0; n * 3], // sh_degree 0 -> 3 coeffs / Gaussian
            sh_degree: 0,
            face_indices: vec![0; n],
            barycentric: vec![[1.0, 0.0, 0.0]; n],
            local_offsets: vec![[0.0; 3]; n],
            is_rigid: vec![true; n],
            optimizer_groups: vec![],
            metrics_history: vec![],
        }
    }

    // A corrupt/hostile `sh_degree` near u32::MAX must be rejected cleanly
    // (as a CheckpointDataMismatch) rather than overflowing the
    // `(degree+1)^2 * 3` arithmetic.
    #[test]
    fn validate_rejects_huge_sh_degree_without_overflow() {
        let mut data = valid_data(2);
        data.sh_degree = u32::MAX;
        let result = data.validate(); // must not panic
        assert!(matches!(
            result,
            Err(TrainerError::CheckpointDataMismatch { .. })
        ));
    }

    #[test]
    fn validate_rejects_sh_degree_above_max() {
        let mut data = valid_data(1);
        data.sh_degree = 4; // MAX_SH_DEGREE is 3
        assert!(matches!(
            data.validate(),
            Err(TrainerError::CheckpointDataMismatch { .. })
        ));
    }

    #[test]
    fn validate_accepts_sh_degree_at_max() {
        let mut data = valid_data(1);
        data.sh_degree = 3;
        // n=1, degree=3 -> sh_per = (3+1)*(3+1)*3 = 48, expected_sh = 1*48.
        data.sh_coeffs = vec![0.0; (3 + 1) * (3 + 1) * 3];
        assert!(data.validate().is_ok());
    }

    // NaN in rotations was previously never scanned by validate().
    #[test]
    fn validate_detects_nan_in_rotations() {
        let mut data = valid_data(1);
        data.rotations[0] = [f32::NAN, 0.0, 0.0, 1.0];
        assert!(matches!(
            data.validate(),
            Err(TrainerError::NanDetected { .. })
        ));
    }

    // Inf in barycentric coordinates was previously never scanned.
    #[test]
    fn validate_detects_inf_in_barycentric() {
        let mut data = valid_data(1);
        data.barycentric[0] = [f32::INFINITY, 0.0, 0.0];
        assert!(matches!(
            data.validate(),
            Err(TrainerError::InfDetected { .. })
        ));
    }

    // NaN in local_offsets was previously never scanned.
    #[test]
    fn validate_detects_nan_in_local_offsets() {
        let mut data = valid_data(1);
        data.local_offsets[0] = [0.0, f32::NAN, 0.0];
        assert!(matches!(
            data.validate(),
            Err(TrainerError::NanDetected { .. })
        ));
    }

    // A NaN Adam second moment was previously never scanned — it would pass
    // validate() and poison the resumed run on the first optimizer step.
    #[test]
    fn validate_detects_nan_in_optimizer_group_moments() {
        let mut data = valid_data(1);
        data.optimizer_groups.push(GroupCheckpoint {
            name: "position".into(),
            m: vec![0.0; 3],
            v: vec![f32::NAN, 0.0, 0.0],
            t: 1,
        });
        assert!(matches!(
            data.validate(),
            Err(TrainerError::NanDetected { .. })
        ));
    }

    // An optimizer group whose m/v lengths disagree must be rejected rather
    // than silently accepted (a resumed Adam step would otherwise read
    // uninitialised or out-of-range moment data).
    #[test]
    fn validate_detects_optimizer_group_m_v_length_mismatch() {
        let mut data = valid_data(1);
        data.optimizer_groups.push(GroupCheckpoint {
            name: "position".into(),
            m: vec![0.0; 3],
            v: vec![0.0; 2], // wrong length
            t: 1,
        });
        assert!(matches!(
            data.validate(),
            Err(TrainerError::CheckpointDataMismatch { .. })
        ));
    }

    // A known optimizer group name whose length doesn't match the model
    // size must also be rejected.
    #[test]
    fn validate_detects_optimizer_group_size_mismatch_with_model() {
        let mut data = valid_data(2); // n=2 -> "position" should be len 6
        data.optimizer_groups.push(GroupCheckpoint {
            name: "position".into(),
            m: vec![0.0; 3], // wrong: should be 2*3=6
            v: vec![0.0; 3],
            t: 1,
        });
        assert!(matches!(
            data.validate(),
            Err(TrainerError::CheckpointDataMismatch { .. })
        ));
    }

    #[test]
    fn validate_accepts_correctly_sized_optimizer_groups() {
        let mut data = valid_data(2);
        data.optimizer_groups.push(GroupCheckpoint {
            name: "position".into(),
            m: vec![0.0; 6],
            v: vec![0.0; 6],
            t: 1,
        });
        data.optimizer_groups.push(GroupCheckpoint {
            name: "rotation".into(),
            m: vec![0.0; 8],
            v: vec![0.0; 8],
            t: 1,
        });
        data.optimizer_groups.push(GroupCheckpoint {
            name: "opacity".into(),
            m: vec![0.0; 2],
            v: vec![0.0; 2],
            t: 1,
        });
        assert!(data.validate().is_ok());
    }

    // save_checkpoint must be atomic: a checkpoint saved once, then saved
    // again with different content, must never leave the target path
    // truncated or unparsable, and load_checkpoint must see the final
    // content.
    #[test]
    fn save_checkpoint_is_atomic_and_rotates_backup() -> Result<(), TrainerError> {
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join(format!("test_ckpt_atomic_{}.json", std::process::id()));
        let backup_path = path.with_extension("backup.json");
        let tmp_path = path.with_extension("tmp");
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&backup_path);
        let _ = std::fs::remove_file(&tmp_path);

        let mut first = valid_data(1);
        first.iteration = 1;
        save_checkpoint(&path, &first)?;
        assert!(path.exists());
        assert!(!backup_path.exists(), "no backup before the first save");

        let mut second = valid_data(1);
        second.iteration = 2;
        save_checkpoint(&path, &second)?;
        assert!(
            backup_path.exists(),
            "second save must rotate the first save to a backup"
        );
        assert!(!tmp_path.exists(), "temp file must not linger after rename");

        // Primary now holds the second save's content.
        let loaded = load_checkpoint(&path)?;
        assert_eq!(loaded.iteration, 2);
        // Backup holds the first save's content.
        let loaded_backup = load_checkpoint(&backup_path)?;
        assert_eq!(loaded_backup.iteration, 1);

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&backup_path);
        Ok(())
    }

    // try_load_checkpoint_with_fallback must actually recover from the
    // backup save_checkpoint rotates into place, not just from a file that
    // happens to already exist at the backup path.
    #[test]
    fn try_load_checkpoint_with_fallback_recovers_from_rotated_backup() -> Result<(), TrainerError>
    {
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join(format!("test_ckpt_fallback_{}.json", std::process::id()));
        let backup_path = path.with_extension("backup.json");
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&backup_path);

        let mut first = valid_data(1);
        first.iteration = 10;
        save_checkpoint(&path, &first)?;
        let mut second = valid_data(1);
        second.iteration = 20;
        save_checkpoint(&path, &second)?; // rotates `first` to backup_path

        // Corrupt the primary.
        std::fs::write(&path, "{ not valid json }")?;

        let recovered = try_load_checkpoint_with_fallback(&path)?;
        assert_eq!(
            recovered.iteration, 10,
            "must recover the rotated-backup content, not the corrupted primary"
        );

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&backup_path);
        Ok(())
    }

    // restore_model must not panic on mismatched array lengths (a
    // hand-built or directly-deserialized CheckpointData that bypassed
    // validate()); it should truncate to the shortest length instead.
    #[test]
    fn restore_model_truncates_on_mismatched_lengths_instead_of_panicking() {
        let mut data = valid_data(3);
        data.rotations.truncate(1); // now shorter than positions (3)
        let model = restore_model(&data); // must not panic
        assert_eq!(model.gaussians.len(), 1);
        // Every other per-Gaussian array on the *returned* model must also
        // be truncated to the same length, or the model itself would carry
        // a broken internal invariant (e.g. face_indices.len() != gaussians.len()).
        assert_eq!(model.face_indices.len(), 1);
        assert_eq!(model.barycentric.len(), 1);
        assert_eq!(model.local_offsets.len(), 1);
        assert_eq!(model.is_rigid.len(), 1);
    }

    // A mismatch in a mesh-binding array (not one of the first four) must
    // also be clamped, and must clamp *every* returned array to match —
    // not just gaussians.
    #[test]
    fn restore_model_truncates_on_face_indices_mismatch() {
        let mut data = valid_data(5);
        data.face_indices.truncate(2);
        let model = restore_model(&data); // must not panic
        assert_eq!(model.gaussians.len(), 2);
        assert_eq!(model.face_indices.len(), 2);
        assert_eq!(model.barycentric.len(), 2);
        assert_eq!(model.local_offsets.len(), 2);
        assert_eq!(model.is_rigid.len(), 2);
    }

    #[test]
    fn restore_model_matched_lengths_is_unaffected() {
        let data = valid_data(4);
        let model = restore_model(&data);
        assert_eq!(model.gaussians.len(), 4);
        assert_eq!(model.sh_coeffs.len(), 4 * 3); // sh_degree 0 -> 3 per Gaussian
    }

    // A too-short sh_coeffs array must not cause an out-of-bounds panic
    // (either here, or in downstream code that slices
    // `sh_coeffs[i * sh_per..(i+1) * sh_per]` for every `i < gaussians.len()`).
    // The Gaussian count itself must shrink to whatever sh_coeffs can
    // actually back, and the returned sh_coeffs.len() must be an exact
    // multiple of sh_per.
    #[test]
    fn restore_model_truncates_on_short_sh_coeffs() {
        let mut data = valid_data(5); // sh_degree 0 -> sh_per = 3, needs 15
        data.sh_coeffs.truncate(7); // only 2 whole Gaussians' worth (6) + 1 partial
        let model = restore_model(&data); // must not panic
        assert_eq!(
            model.gaussians.len(),
            2,
            "5th/4th/partial-3rd must be dropped"
        );
        assert_eq!(
            model.sh_coeffs.len(),
            6,
            "must be an exact multiple of sh_per=3"
        );
        assert_eq!(model.face_indices.len(), 2);
        assert_eq!(model.barycentric.len(), 2);
        assert_eq!(model.local_offsets.len(), 2);
        assert_eq!(model.is_rigid.len(), 2);
    }

    // An out-of-range sh_degree must be clamped (matching
    // `CheckpointData::validate`'s MAX_SH_DEGREE), not passed through
    // unchanged into the returned model (which would then disagree with the
    // model's own sh_coeffs layout) or overflow while computing sh_per.
    // After clamping to degree 3, sh_per = 48; providing exactly 48
    // coefficients (sized for degree 3, one Gaussian) must round-trip to
    // exactly 1 Gaussian, not panic and not silently keep the corrupt degree.
    #[test]
    fn restore_model_clamps_out_of_range_sh_degree() {
        let mut data = valid_data(1);
        data.sh_degree = 9000; // corrupt: far above MAX_SH_DEGREE
        data.sh_coeffs = vec![0.0; 48]; // exactly 1 Gaussian's worth at degree 3
        let model = restore_model(&data); // must not overflow or panic
        assert_eq!(model.sh_degree, MAX_SH_DEGREE);
        assert_eq!(model.gaussians.len(), 1);
        assert_eq!(model.sh_coeffs.len(), 48);
    }

    // Same corruption, but with too little sh_coeffs data to back even one
    // Gaussian at the clamped degree: the model must come back empty rather
    // than overflowing or panicking while computing `sh_coeffs.len() / sh_per`.
    #[test]
    fn restore_model_clamps_sh_degree_and_drops_gaussians_with_insufficient_sh_data() {
        let mut data = valid_data(1);
        data.sh_degree = 9000; // corrupt: far above MAX_SH_DEGREE
        data.sh_coeffs = vec![0.0; 3]; // only enough for degree 0, not degree 3
        let model = restore_model(&data); // must not overflow or panic
        assert_eq!(model.sh_degree, MAX_SH_DEGREE);
        assert_eq!(
            model.gaussians.len(),
            0,
            "insufficient sh_coeffs for even one Gaussian at the clamped degree"
        );
        assert_eq!(model.sh_coeffs.len(), 0);
    }
}
