//! Checkpoint save / load.
//!
//! Persists the full training state (model parameters + optimiser moments +
//! iteration counter) as a JSON file.  This is intentionally simple — a
//! production implementation would use safetensors or a binary format.

use std::fs;
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

impl CheckpointData {
    /// Validate checkpoint data for consistency and integrity.
    ///
    /// Checks:
    /// - Version compatibility
    /// - Array length consistency
    /// - NaN/Inf detection in numerical arrays
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

        // Validate SH coefficients length
        let sh_per = ((self.sh_degree + 1) * (self.sh_degree + 1) * 3) as usize;
        let expected_sh = n * sh_per;
        if self.sh_coeffs.len() != expected_sh {
            return Err(TrainerError::CheckpointDataMismatch {
                field: "sh_coeffs".into(),
                actual: self.sh_coeffs.len(),
                expected: expected_sh,
            });
        }

        // Check for NaN/Inf in positions
        for (i, pos) in self.positions.iter().enumerate() {
            for (j, &v) in pos.iter().enumerate() {
                if v.is_nan() {
                    return Err(TrainerError::NanDetected {
                        parameter: format!("positions[{i}][{j}]"),
                        index: i,
                    });
                }
                if v.is_infinite() {
                    return Err(TrainerError::InfDetected {
                        parameter: format!("positions[{i}][{j}]"),
                        index: i,
                    });
                }
            }
        }

        // Check for NaN/Inf in scales
        for (i, scl) in self.scales.iter().enumerate() {
            for (j, &v) in scl.iter().enumerate() {
                if v.is_nan() {
                    return Err(TrainerError::NanDetected {
                        parameter: format!("scales[{i}][{j}]"),
                        index: i,
                    });
                }
                if v.is_infinite() {
                    return Err(TrainerError::InfDetected {
                        parameter: format!("scales[{i}][{j}]"),
                        index: i,
                    });
                }
            }
        }

        // Check for NaN/Inf in opacities
        for (i, &v) in self.opacities.iter().enumerate() {
            if v.is_nan() {
                return Err(TrainerError::NanDetected {
                    parameter: "opacities".into(),
                    index: i,
                });
            }
            if v.is_infinite() {
                return Err(TrainerError::InfDetected {
                    parameter: "opacities".into(),
                    index: i,
                });
            }
        }

        // Check for NaN/Inf in SH coefficients
        for (i, &v) in self.sh_coeffs.iter().enumerate() {
            if v.is_nan() {
                return Err(TrainerError::NanDetected {
                    parameter: "sh_coeffs".into(),
                    index: i,
                });
            }
            if v.is_infinite() {
                return Err(TrainerError::InfDetected {
                    parameter: "sh_coeffs".into(),
                    index: i,
                });
            }
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
pub fn restore_model(data: &CheckpointData) -> GaussianModel {
    let n = data.positions.len();
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
        sh_coeffs: data.sh_coeffs.clone(),
        sh_degree: data.sh_degree,
        face_indices: data.face_indices.clone(),
        barycentric: data.barycentric.clone(),
        local_offsets: data.local_offsets.clone(),
        is_rigid: data.is_rigid.clone(),
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

/// Save a checkpoint to `path` as pretty-printed JSON.
pub fn save_checkpoint(path: &Path, data: &CheckpointData) -> Result<(), TrainerError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string(data)?;
    fs::write(path, json)?;
    tracing::info!("Saved checkpoint to {}", path.display());
    Ok(())
}

/// Load a checkpoint from a JSON file at `path`.
///
/// Validates the checkpoint data after loading. Returns an error if:
/// - The file cannot be read
/// - JSON parsing fails (corrupted checkpoint)
/// - Data validation fails (version mismatch, array length mismatch, NaN/Inf)
pub fn load_checkpoint(path: &Path) -> Result<CheckpointData, TrainerError> {
    let json = fs::read_to_string(path).map_err(|e| {
        TrainerError::CheckpointCorrupted(format!(
            "Failed to read checkpoint file {}: {}",
            path.display(),
            e
        ))
    })?;

    let data: CheckpointData = serde_json::from_str(&json).map_err(|e| {
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
/// Attempts to load `path` first. If that fails due to corruption, tries
/// `path.with_extension("backup.json")`.
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
}
