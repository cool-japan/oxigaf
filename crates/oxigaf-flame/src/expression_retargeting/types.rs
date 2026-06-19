//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::functions::{retar_compute_variance, retar_solve_ridge};

/// Per-dimension variance statistics for a set of expression states.
#[derive(Debug, Clone)]
pub struct ExpressionVarianceStats {
    /// Per-dimension variance of expression coefficients.
    pub per_dim_variance: Vec<f32>,
    /// Sum of all per-dimension variances.
    pub total_variance: f32,
    /// Indices of the most variable dimensions, sorted descending.
    pub top_k_dims: Vec<usize>,
}
/// Errors that can occur during expression retargeting operations.
#[derive(Debug, thiserror::Error)]
pub enum RetargetError {
    /// Input expression dimension does not match what the retargeter expects.
    #[error("dimension mismatch: source has {src_dim} expression dims, got {got}")]
    DimensionMismatch { src_dim: usize, got: usize },
    /// A blend weight outside `[0, 1]` was provided.
    #[error("invalid weight: must be in [0, 1], got {0}")]
    InvalidWeight(f32),
    /// Too few training pairs were supplied.
    #[error("not enough training pairs: need at least {needed}, got {got}")]
    NotEnoughPairs { needed: usize, got: usize },
    /// An empty sequence or state list was provided where non-empty was required.
    #[error("empty sequence")]
    EmptySequence,
    /// A configuration field has an invalid value.
    #[error("invalid config: {0}")]
    InvalidConfig(String),
    /// Matrix factorization encountered a (near-)singular system.
    #[error("singular matrix")]
    Singular,
}
/// Linear expression retargeter.
///
/// Learns an affine mapping `M` from source to target expression space
/// (including optional jaw pose).  Solved via ridge regression:
///
/// `M = argmin ||Source · M − Target||_F² + λ ||M||_F²`
///
/// The mapping matrix `M` is stored row-major as `[target_dim × source_dim]`.
#[derive(Debug, Clone)]
pub struct LinearExpressionRetargeter {
    pub(super) config: RetargetConfig,
    /// Retargeting matrix stored row-major: `[feat_dim × feat_dim]`.
    pub(super) mapping: Vec<f32>,
    /// Feature-space mean of source training data.
    source_mean: Vec<f32>,
    /// Feature-space mean of target training data.
    target_mean: Vec<f32>,
    /// Number of training pairs used to fit the mapper.
    n_training_pairs: usize,
    /// Optional variance statistics of source expression space.
    pub source_variance: Option<ExpressionVarianceStats>,
}
impl LinearExpressionRetargeter {
    /// Effective feature dimension (`expr_dim` or `expr_dim+3` if `include_jaw`).
    pub(super) fn feat_dim(&self) -> usize {
        if self.config.include_jaw {
            self.config.expr_dim + 3
        } else {
            self.config.expr_dim
        }
    }
    /// Build a retargeter using the identity matrix as the mapping.
    ///
    /// Useful as a no-op baseline when no training pairs are available.
    #[must_use]
    pub fn identity(config: RetargetConfig) -> Self {
        let dim = if config.include_jaw {
            config.expr_dim + 3
        } else {
            config.expr_dim
        };
        let mut mapping = vec![0.0_f32; dim * dim];
        for i in 0..dim {
            mapping[i * dim + i] = 1.0_f32;
        }
        Self {
            source_mean: vec![0.0_f32; dim],
            target_mean: vec![0.0_f32; dim],
            n_training_pairs: 0,
            source_variance: None,
            mapping,
            config,
        }
    }
    /// Train the retargeter from source→target expression pairs.
    ///
    /// At least 2 pairs are required.  Fewer pairs return
    /// [`RetargetError::NotEnoughPairs`].
    ///
    /// # Errors
    ///
    /// Returns an error if the operation fails.
    pub fn fit(pairs: &[RetargetPair], config: RetargetConfig) -> Result<Self, RetargetError> {
        if pairs.len() < 2 {
            return Err(RetargetError::NotEnoughPairs {
                needed: 2,
                got: pairs.len(),
            });
        }
        let dim = if config.include_jaw {
            config.expr_dim + 3
        } else {
            config.expr_dim
        };
        let n = pairs.len();
        let mut src_vecs: Vec<Vec<f32>> = Vec::with_capacity(n);
        let mut tgt_vecs: Vec<Vec<f32>> = Vec::with_capacity(n);
        for pair in pairs {
            if pair.source.expr_dim != config.expr_dim {
                return Err(RetargetError::DimensionMismatch {
                    src_dim: config.expr_dim,
                    got: pair.source.expr_dim,
                });
            }
            if pair.target.expr_dim != config.expr_dim {
                return Err(RetargetError::DimensionMismatch {
                    src_dim: config.expr_dim,
                    got: pair.target.expr_dim,
                });
            }
            if config.include_jaw {
                src_vecs.push(pair.source.full_vector());
                tgt_vecs.push(pair.target.full_vector());
            } else {
                src_vecs.push(pair.source.expression_params.clone());
                tgt_vecs.push(pair.target.expression_params.clone());
            }
        }
        let mut source_mean = vec![0.0_f32; dim];
        let mut target_mean = vec![0.0_f32; dim];
        for i in 0..n {
            for d in 0..dim {
                source_mean[d] += src_vecs[i][d];
                target_mean[d] += tgt_vecs[i][d];
            }
        }
        let n_f = n as f32;
        for d in 0..dim {
            source_mean[d] /= n_f;
            target_mean[d] /= n_f;
        }
        let source_variance = if config.scale_by_variance {
            let expr_states: Vec<ExpressionState> =
                pairs.iter().map(|p| p.source.clone()).collect();
            Some(retar_compute_variance(&expr_states)?)
        } else {
            None
        };
        let mut src_centered: Vec<f32> = vec![0.0_f32; n * dim];
        let mut tgt_centered: Vec<f32> = vec![0.0_f32; n * dim];
        for i in 0..n {
            for d in 0..dim {
                src_centered[i * dim + d] = src_vecs[i][d] - source_mean[d];
                tgt_centered[i * dim + d] = tgt_vecs[i][d] - target_mean[d];
            }
        }
        let lambda = config.regularization * n_f;
        let mut mapping = vec![0.0_f32; dim * dim];
        for col in 0..dim {
            let b_col: Vec<f32> = (0..n).map(|i| tgt_centered[i * dim + col]).collect();
            let x = retar_solve_ridge(&src_centered, &b_col, n, dim, lambda)?;
            for row in 0..dim {
                mapping[row * dim + col] = x[row];
            }
        }
        Ok(Self {
            config,
            mapping,
            source_mean,
            target_mean,
            n_training_pairs: n,
            source_variance,
        })
    }
    /// Apply the learned mapping to a source expression state.
    ///
    /// # Errors
    ///
    /// Returns an error if the operation fails.
    pub fn retarget(&self, source: &ExpressionState) -> Result<ExpressionState, RetargetError> {
        if source.expr_dim != self.config.expr_dim {
            return Err(RetargetError::DimensionMismatch {
                src_dim: self.config.expr_dim,
                got: source.expr_dim,
            });
        }
        let dim = self.feat_dim();
        let src_vec = if self.config.include_jaw {
            source.full_vector()
        } else {
            source.expression_params.clone()
        };
        let centered: Vec<f32> = src_vec
            .iter()
            .zip(self.source_mean.iter())
            .map(|(sv, mu)| sv - mu)
            .collect();
        let mut out = vec![0.0_f32; dim];
        for (row, &cen_val) in centered.iter().enumerate() {
            let row_start = row * dim;
            for (col, out_val) in out.iter_mut().enumerate() {
                *out_val += cen_val * self.mapping[row_start + col];
            }
        }
        for (out_val, &tgt_mu) in out.iter_mut().zip(self.target_mean.iter()) {
            *out_val += tgt_mu;
        }
        let expr_dim = self.config.expr_dim;
        let expression_params = out[..expr_dim].to_vec();
        let jaw_pose = if self.config.include_jaw {
            [out[expr_dim], out[expr_dim + 1], out[expr_dim + 2]]
        } else {
            source.jaw_pose
        };
        Ok(ExpressionState {
            expression_params,
            jaw_pose,
            expr_dim,
        })
    }
    /// Retarget a sequence of expression states.
    ///
    /// Returns [`RetargetError::EmptySequence`] if `sequence` is empty.
    ///
    /// # Errors
    ///
    /// Returns an error if the operation fails.
    pub fn retarget_sequence(
        &self,
        sequence: &[ExpressionState],
    ) -> Result<Vec<ExpressionState>, RetargetError> {
        if sequence.is_empty() {
            return Err(RetargetError::EmptySequence);
        }
        sequence.iter().map(|s| self.retarget(s)).collect()
    }
    /// Number of training pairs used to fit this retargeter (0 for identity).
    #[must_use]
    pub fn n_training_pairs(&self) -> usize {
        self.n_training_pairs
    }
    /// Reference to the configuration.
    #[must_use]
    pub fn config(&self) -> &RetargetConfig {
        &self.config
    }
    /// Source feature dimension.
    #[must_use]
    pub fn source_dim(&self) -> usize {
        self.feat_dim()
    }
    /// Target feature dimension.
    #[must_use]
    pub fn target_dim(&self) -> usize {
        self.feat_dim()
    }
}
/// A FLAME expression state: expression coefficients + jaw pose.
///
/// `expression_params` has length `expr_dim` (typically 50 or 100).
/// `jaw_pose` is a 3-element axis-angle rotation of the jaw joint.
#[derive(Debug, Clone)]
pub struct ExpressionState {
    /// FLAME expression coefficients (ψ).
    pub expression_params: Vec<f32>,
    /// Jaw joint axis-angle rotation (3 floats).
    pub jaw_pose: [f32; 3],
    /// Dimensionality of `expression_params`.
    pub expr_dim: usize,
}
impl ExpressionState {
    /// Create a neutral (all-zero) expression state of the given dimensionality.
    #[must_use]
    pub fn neutral(expr_dim: usize) -> Self {
        Self {
            expression_params: vec![0.0_f32; expr_dim],
            jaw_pose: [0.0_f32; 3],
            expr_dim,
        }
    }
    /// Create a state from expression parameters only; jaw pose defaults to zero.
    #[must_use]
    pub fn from_params(params: Vec<f32>) -> Self {
        let expr_dim = params.len();
        Self {
            expression_params: params,
            jaw_pose: [0.0_f32; 3],
            expr_dim,
        }
    }
    /// Create a state from expression parameters and an explicit jaw pose.
    #[must_use]
    pub fn from_params_and_jaw(params: Vec<f32>, jaw: [f32; 3]) -> Self {
        let expr_dim = params.len();
        Self {
            expression_params: params,
            jaw_pose: jaw,
            expr_dim,
        }
    }
    /// Return a new state where all coefficients (expression + jaw) are multiplied by `scale`.
    #[must_use]
    pub fn with_scale(&self, scale: f32) -> Self {
        Self {
            expression_params: self.expression_params.iter().map(|&v| v * scale).collect(),
            jaw_pose: [
                self.jaw_pose[0] * scale,
                self.jaw_pose[1] * scale,
                self.jaw_pose[2] * scale,
            ],
            expr_dim: self.expr_dim,
        }
    }
    /// Linear interpolation between `self` and `other` at parameter `t ∈ [0, 1]`.
    ///
    /// Returns an error if the two states have different `expr_dim`.
    ///
    /// # Errors
    ///
    /// Returns an error if the operation fails.
    pub fn blend(&self, other: &Self, t: f32) -> Result<Self, RetargetError> {
        if self.expr_dim != other.expr_dim {
            return Err(RetargetError::DimensionMismatch {
                src_dim: self.expr_dim,
                got: other.expr_dim,
            });
        }
        let t = t.clamp(0.0_f32, 1.0_f32);
        let expr = self
            .expression_params
            .iter()
            .zip(other.expression_params.iter())
            .map(|(&a, &b)| a + t * (b - a))
            .collect();
        let jaw = [
            self.jaw_pose[0] + t * (other.jaw_pose[0] - self.jaw_pose[0]),
            self.jaw_pose[1] + t * (other.jaw_pose[1] - self.jaw_pose[1]),
            self.jaw_pose[2] + t * (other.jaw_pose[2] - self.jaw_pose[2]),
        ];
        Ok(Self {
            expression_params: expr,
            jaw_pose: jaw,
            expr_dim: self.expr_dim,
        })
    }
    /// Full feature vector: expression params followed by jaw pose.
    pub(super) fn full_vector(&self) -> Vec<f32> {
        let mut v = self.expression_params.clone();
        v.extend_from_slice(&self.jaw_pose);
        v
    }
    /// Total dimension (expr + 3 jaw).
    #[must_use]
    pub fn full_dim(&self) -> usize {
        self.expr_dim + 3
    }
}
/// A source→target expression pair for training the retargeter.
#[derive(Debug, Clone)]
pub struct RetargetPair {
    /// Expression state in the source identity's parameter space.
    pub source: ExpressionState,
    /// Corresponding expression state in the target identity's parameter space.
    pub target: ExpressionState,
}
/// Statistics of a trained retargeter evaluated on a set of pairs.
#[derive(Debug, Clone)]
pub struct RetargetStats {
    /// Mean L2 error across all training pairs.
    pub mean_error: f32,
    /// Maximum L2 error across all training pairs.
    pub max_error: f32,
    /// Per-dimension mean absolute error.
    pub per_dim_error: Vec<f32>,
    /// Frobenius norm of the retargeting matrix.
    pub mapping_frobenius: f32,
}
/// Configuration for [`LinearExpressionRetargeter`].
#[derive(Debug, Clone)]
pub struct RetargetConfig {
    /// Expression parameter dimensionality. Default: `50`.
    pub expr_dim: usize,
    /// L2 (Tikhonov) regularization strength for the ridge-regression solve. Default: `1e-4`.
    pub regularization: f32,
    /// When `true`, expression features are normalized by their per-dimension standard deviation
    /// before learning the mapping. Default: `true`.
    pub scale_by_variance: bool,
    /// When `true`, the jaw pose is included in the joint feature vector. Default: `true`.
    pub include_jaw: bool,
    /// Gaussian smoothing sigma (in frames) used by `retar_smooth_sequence`. Default: `2.0`.
    pub smoothing_sigma: f32,
}
