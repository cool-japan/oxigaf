//! Named facial expression presets and blending utilities for the FLAME model.
//!
//! This module provides a library of named expressions with pre-set parameter
//! values, expression blending, parameter constraints, and extension methods
//! on [`FlameParams`] for convenient expression application.
//!
//! # Where expression coefficients come from
//!
//! FLAME's expression space is a *data-driven* PCA basis: component `k` has no
//! fixed, model-independent meaning, and the same coefficient vector produces
//! different faces under different `expressiondirs` matrices.  There is
//! therefore no such thing as a universal "smile vector".
//!
//! Consequently this module offers three ways to obtain coefficients, in
//! decreasing order of fidelity:
//!
//! 1. [`NamedExpression::fit_to_basis`] — project a reference expression mesh
//!    onto the expression basis of the FLAME model you actually use.  This is
//!    the only way to obtain coefficients that are correct for that model.
//! 2. [`ExpressionLibrary::from_json_file`] — load coefficients that were
//!    fitted earlier (e.g. by step 1) and serialised with
//!    [`ExpressionLibrary::to_json_string`].
//! 3. [`ExpressionLibrary::placeholder_expressions`] — hand-authored,
//!    *illustrative* values used for smoke tests, examples and UI wiring.
//!    They are **not** fitted to any FLAME basis and will not reproduce the
//!    named expression on a real model.
//!
//! # Quick Start
//!
//! ```rust
//! use oxigaf_flame::{
//!     ExpressionLibrary, ExpressionBlend, FlameParams, FlameParamConstraints,
//! };
//!
//! // Load the illustrative placeholder presets (see the caveat above)
//! let lib = ExpressionLibrary::placeholder_expressions();
//!
//! // Apply a single expression to neutral params
//! let params = FlameParams::neutral();
//!
//! // Blend two expressions together
//! let mut blend = ExpressionBlend::new();
//! blend.add_component("smile", 0.6).add_component("surprised", 0.4);
//! let params_vec = blend.evaluate(&lib, 10).expect("blend failed");
//!
//! // Validate and clamp
//! let constraints = FlameParamConstraints::default();
//! let violations = constraints.validate(&FlameParams::neutral());
//! assert!(violations.is_empty());
//! ```

use crate::blend_shape_solver::{fit_expression_coefficients, BlendSolverConfig};
use crate::error::FlameError;
use crate::params::FlameParams;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Constants — number of expression PCA components in the presets
// ---------------------------------------------------------------------------

/// Number of expression parameters used in the placeholder presets.
const PRESET_NUM_PARAMS: usize = 10;

// ---------------------------------------------------------------------------
// Provenance
// ---------------------------------------------------------------------------

/// Where the coefficients of a [`NamedExpression`] came from.
///
/// FLAME's expression basis is data-driven, so coefficients are only
/// meaningful relative to the basis they were fitted against. This enum makes
/// that provenance explicit instead of leaving callers to guess.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExpressionProvenance {
    /// Coefficients were fitted to a concrete FLAME expression basis by
    /// [`NamedExpression::fit_to_basis`] (or an equivalent offline fit) and
    /// reproduce the named expression on *that* model.
    Fitted,
    /// Coefficients were loaded from a user-supplied data file. Fidelity is
    /// the responsibility of whoever produced the file.
    Loaded,
    /// Hand-authored illustrative values. They are **not** fitted to any FLAME
    /// basis and will not reproduce the named expression on a real model — see
    /// [`ExpressionLibrary::placeholder_expressions`].
    Placeholder,
}

impl ExpressionProvenance {
    /// `true` when the coefficients are not tied to any real FLAME basis.
    #[inline]
    #[must_use]
    pub fn is_placeholder(self) -> bool {
        matches!(self, Self::Placeholder)
    }
}

// ---------------------------------------------------------------------------
// NamedExpression
// ---------------------------------------------------------------------------

/// A named facial expression with preset parameter values.
///
/// The `params` vector contains the first N non-zero FLAME expression
/// coefficients. When the actual model uses more components, the remainder
/// are treated as zero. When the model uses fewer, the preset is truncated.
///
/// Coefficients are only meaningful with respect to the expression basis they
/// were fitted against; [`NamedExpression::provenance`] records which basis (if
/// any) that was.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NamedExpression {
    /// Unique identifier for this expression.
    pub name: String,
    /// Human-readable description of the expression.
    pub description: String,
    /// Expression PCA coefficients (first N components; rest implicitly zero).
    pub params: Vec<f32>,
    /// Intensity scaling factor in `[0, 1]`. 1.0 means full expression.
    pub intensity: f32,
    /// Where `params` came from. Defaults to [`ExpressionProvenance::Loaded`]
    /// when absent from a deserialised data file.
    #[serde(default = "default_provenance")]
    pub provenance: ExpressionProvenance,
}

/// Provenance assumed for entries in a data file that do not declare one.
fn default_provenance() -> ExpressionProvenance {
    ExpressionProvenance::Loaded
}

impl NamedExpression {
    /// Create a new named expression from coefficients of unspecified origin.
    ///
    /// The result is tagged [`ExpressionProvenance::Loaded`]; use
    /// [`NamedExpression::with_provenance`] to state the origin explicitly, or
    /// [`NamedExpression::fit_to_basis`] to derive coefficients from a mesh.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        params: Vec<f32>,
        intensity: f32,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            params,
            intensity,
            provenance: ExpressionProvenance::Loaded,
        }
    }

    /// Override the recorded provenance (builder-style).
    #[must_use]
    pub fn with_provenance(mut self, provenance: ExpressionProvenance) -> Self {
        self.provenance = provenance;
        self
    }

    /// Fit expression coefficients by projecting a reference mesh onto a FLAME
    /// expression basis.
    ///
    /// This is the only way to obtain coefficients that actually reproduce the
    /// named expression: the resulting values are specific to
    /// `expression_basis` and must be re-fitted for any other FLAME model.
    ///
    /// - `neutral_verts` — the model's neutral (expression-zero) vertices as a
    ///   flat `[x0, y0, z0, x1, …]` array of length `3 * n_vertices`.
    /// - `target_verts` — the reference expression mesh in the same layout and
    ///   vertex order.
    /// - `expression_basis` — one displacement vector per PCA component, each
    ///   the same length as `neutral_verts` (i.e. `expressiondirs[:, :, k]`
    ///   flattened row-major).
    /// - `n_coeffs` — how many leading components to solve for.
    ///
    /// The returned expression is tagged [`ExpressionProvenance::Fitted`].
    ///
    /// # Errors
    ///
    /// Returns [`FlameError::InvalidParams`] when the inputs are empty or
    /// inconsistent, and [`FlameError::Numerical`] when the underlying solver
    /// fails to converge.
    pub fn fit_to_basis(
        name: impl Into<String>,
        description: impl Into<String>,
        neutral_verts: &[f32],
        target_verts: &[f32],
        expression_basis: Vec<Vec<f32>>,
        n_coeffs: usize,
    ) -> Result<Self, FlameError> {
        if n_coeffs == 0 {
            return Err(FlameError::InvalidParams(
                "n_coeffs must be greater than zero".to_owned(),
            ));
        }
        if expression_basis.is_empty() {
            return Err(FlameError::InvalidParams(
                "expression_basis must contain at least one displacement vector".to_owned(),
            ));
        }
        if neutral_verts.is_empty() || !neutral_verts.len().is_multiple_of(3) {
            return Err(FlameError::InvalidParams(format!(
                "neutral_verts must be a non-empty multiple of 3, got {}",
                neutral_verts.len()
            )));
        }
        if neutral_verts.len() != target_verts.len() {
            return Err(FlameError::InvalidParams(format!(
                "vertex count mismatch: neutral has {} scalars, target has {}",
                neutral_verts.len(),
                target_verts.len()
            )));
        }

        let params = fit_expression_coefficients(
            neutral_verts,
            target_verts,
            expression_basis,
            n_coeffs,
            &BlendSolverConfig::default(),
        )
        .map_err(|e| FlameError::numerical(format!("expression basis fit failed: {e}")))?;

        Ok(Self {
            name: name.into(),
            description: description.into(),
            params,
            intensity: 1.0,
            provenance: ExpressionProvenance::Fitted,
        })
    }
}

// ---------------------------------------------------------------------------
// ExpressionLibrary
// ---------------------------------------------------------------------------

/// A collection of named facial expressions.
///
/// Populate it by fitting against a real FLAME basis
/// ([`NamedExpression::fit_to_basis`]), by loading a data file
/// ([`ExpressionLibrary::from_json_file`]), or — for demos and smoke tests
/// only — with [`ExpressionLibrary::placeholder_expressions`].
pub struct ExpressionLibrary {
    expressions: Vec<NamedExpression>,
}

/// On-disk representation of an [`ExpressionLibrary`].
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ExpressionLibraryJson {
    /// Named expressions in file order.
    expressions: Vec<NamedExpression>,
}

impl ExpressionLibrary {
    /// Create an empty library.
    #[must_use]
    pub fn new() -> Self {
        Self {
            expressions: Vec::new(),
        }
    }

    /// Create a library of hand-authored **placeholder** expressions.
    ///
    /// # These coefficients are illustrative, not fitted
    ///
    /// FLAME's expression space is a data-driven PCA basis whose components
    /// carry no fixed semantics, so no hard-coded coefficient vector can encode
    /// "smile" across models. The values returned here were chosen by hand to
    /// exercise the blending, constraint and animation machinery; applying them
    /// to a real FLAME model produces *some* deformation, but not the named
    /// expression. `"winking"` cannot be expressed at all through this basis —
    /// FLAME drives eyelid closure through the eye-joint pose parameters, not
    /// through expression coefficients.
    ///
    /// Every entry is tagged [`ExpressionProvenance::Placeholder`], so callers
    /// can detect the situation at runtime via
    /// [`ExpressionLibrary::has_placeholders`].
    ///
    /// For coefficients that actually reproduce an expression, fit them against
    /// the model you use with [`NamedExpression::fit_to_basis`] and persist the
    /// result with [`ExpressionLibrary::to_json_string`].
    #[must_use]
    pub fn placeholder_expressions() -> Self {
        let mut lib = Self::new();

        // Helper: build a params Vec of PRESET_NUM_PARAMS elements with
        // specified (index, value) overrides and zeros elsewhere.
        let make_params = |overrides: &[(usize, f32)]| -> Vec<f32> {
            let mut p = vec![0.0_f32; PRESET_NUM_PARAMS];
            for &(idx, val) in overrides {
                if idx < PRESET_NUM_PARAMS {
                    p[idx] = val;
                }
            }
            p
        };

        // Helper: tag every entry as a placeholder.
        let placeholder = |name: &str, description: &str, params: Vec<f32>| {
            NamedExpression::new(name, description, params, 1.0)
                .with_provenance(ExpressionProvenance::Placeholder)
        };

        // "neutral" is exact regardless of basis: zero coefficients always mean
        // the neutral face, so it is the one entry that is genuinely correct.
        lib.add(
            NamedExpression::new(
                "neutral",
                "Resting / neutral face with no deformation",
                vec![0.0_f32; PRESET_NUM_PARAMS],
                1.0,
            )
            .with_provenance(ExpressionProvenance::Fitted),
        );

        lib.add(placeholder(
            "smile",
            "Placeholder: gentle smile with upturned lip corners (not fitted)",
            make_params(&[(0, 0.3), (1, 1.5), (2, -0.3), (3, 0.2)]),
        ));

        lib.add(placeholder(
            "grin",
            "Placeholder: broad grin with wide lip spread (not fitted)",
            make_params(&[(0, 0.5), (1, 2.0), (2, -0.8), (3, 0.4)]),
        ));

        lib.add(placeholder(
            "frown",
            "Placeholder: downturned lip corners indicating displeasure (not fitted)",
            make_params(&[(0, 0.2), (1, -1.2), (2, 0.3), (3, -0.2)]),
        ));

        lib.add(placeholder(
            "surprised",
            "Placeholder: raised brows, wide eyes, slightly open mouth (not fitted)",
            make_params(&[(0, -1.5), (1, 0.5), (2, 1.2), (3, 0.3), (4, 0.8)]),
        ));

        lib.add(placeholder(
            "angry",
            "Placeholder: furrowed brows and tightened mouth (not fitted)",
            make_params(&[(0, 1.0), (1, -0.8), (2, -1.5), (3, 0.5), (5, -0.4)]),
        ));

        lib.add(placeholder(
            "sad",
            "Placeholder: drooping mouth corners and downcast brows (not fitted)",
            make_params(&[(0, 0.4), (1, -0.5), (2, 0.7), (3, -0.8), (4, -0.3)]),
        ));

        lib.add(placeholder(
            "disgusted",
            "Placeholder: raised upper lip and furrowed nose (not fitted)",
            make_params(&[(0, 0.7), (1, -1.0), (2, 1.2), (3, -0.3), (5, 0.6)]),
        ));

        lib.add(placeholder(
            "fearful",
            "Placeholder: wide eyes and tense brow indicating fear (not fitted)",
            make_params(&[(0, -0.8), (1, 0.4), (2, 0.9), (3, 0.5), (4, 0.7)]),
        ));

        lib.add(placeholder(
            "winking",
            "Placeholder: right eye wink — NOT expressible through the FLAME \
             expression basis; drive the eye joints in `pose` instead",
            make_params(&[(0, 0.1), (6, 2.0), (7, -1.5)]),
        ));

        lib.add(placeholder(
            "open_mouth",
            "Placeholder: open jaw / mouth wide open — prefer the jaw pose \
             parameter `pose[6]` for a real jaw opening",
            make_params(&[(0, -1.8), (1, 0.2)]),
        ));

        lib
    }

    /// Deprecated alias for [`ExpressionLibrary::placeholder_expressions`].
    ///
    /// Renamed because the values it returns are illustrative placeholders, not
    /// coefficients fitted to any FLAME expression basis.
    #[must_use]
    #[deprecated(
        since = "0.1.2",
        note = "renamed to `placeholder_expressions`: these coefficients are illustrative, \
                not fitted to any FLAME basis. Use `NamedExpression::fit_to_basis` or \
                `ExpressionLibrary::from_json_file` for real coefficients."
    )]
    pub fn default_expressions() -> Self {
        Self::placeholder_expressions()
    }

    /// Load a library from a JSON file produced by
    /// [`ExpressionLibrary::to_json_string`].
    ///
    /// # Errors
    ///
    /// Returns [`FlameError::IoError`] when the file cannot be read and
    /// [`FlameError::InvalidParams`] when its contents are not a valid library.
    pub fn from_json_file(path: impl AsRef<std::path::Path>) -> Result<Self, FlameError> {
        let path = path.as_ref();
        let json_str = std::fs::read_to_string(path).map_err(|e| FlameError::IoError {
            source: e,
            path: path.to_path_buf(),
        })?;
        Self::from_json_str(&json_str)
    }

    /// Parse a library from a JSON string.
    ///
    /// # Errors
    ///
    /// Returns [`FlameError::InvalidParams`] when the JSON does not describe a
    /// valid library.
    pub fn from_json_str(json_str: &str) -> Result<Self, FlameError> {
        let parsed: ExpressionLibraryJson = serde_json::from_str(json_str).map_err(|e| {
            FlameError::InvalidParams(format!("failed to parse expression library JSON: {e}"))
        })?;
        Ok(Self {
            expressions: parsed.expressions,
        })
    }

    /// Serialise the library to a pretty-printed JSON string.
    ///
    /// # Errors
    ///
    /// Returns [`FlameError::InvalidParams`] if serialisation fails.
    pub fn to_json_string(&self) -> Result<String, FlameError> {
        let doc = ExpressionLibraryJson {
            expressions: self.expressions.clone(),
        };
        serde_json::to_string_pretty(&doc).map_err(|e| {
            FlameError::InvalidParams(format!("failed to serialise expression library: {e}"))
        })
    }

    /// Add an expression to the library.
    pub fn add(&mut self, expr: NamedExpression) {
        self.expressions.push(expr);
    }

    /// Retrieve an expression by name, returning `None` if not found.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&NamedExpression> {
        self.expressions.iter().find(|e| e.name == name)
    }

    /// Return all expression names in insertion order.
    #[must_use]
    pub fn names(&self) -> Vec<&str> {
        self.expressions.iter().map(|e| e.name.as_str()).collect()
    }

    /// Return the number of expressions in the library.
    #[must_use]
    pub fn count(&self) -> usize {
        self.expressions.len()
    }

    /// `true` when any entry carries [`ExpressionProvenance::Placeholder`]
    /// coefficients, i.e. values that are not fitted to a real FLAME basis.
    #[must_use]
    pub fn has_placeholders(&self) -> bool {
        self.expressions
            .iter()
            .any(|e| e.provenance.is_placeholder())
    }

    /// Names of every entry whose coefficients are placeholders.
    #[must_use]
    pub fn placeholder_names(&self) -> Vec<&str> {
        self.expressions
            .iter()
            .filter(|e| e.provenance.is_placeholder())
            .map(|e| e.name.as_str())
            .collect()
    }
}

impl Default for ExpressionLibrary {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ExpressionBlend
// ---------------------------------------------------------------------------

/// A weighted combination of named expressions.
///
/// Weights are arbitrary floats — they are not required to sum to 1.0 unless
/// you call [`ExpressionBlend::normalize`].
///
/// # Example
///
/// ```rust
/// use oxigaf_flame::{ExpressionBlend, ExpressionLibrary};
///
/// let lib = ExpressionLibrary::placeholder_expressions();
/// let params = ExpressionBlend::single("smile", 1.0)
///     .evaluate(&lib, 10)
///     .expect("evaluate failed");
/// ```
#[derive(Debug, Clone, Default)]
pub struct ExpressionBlend {
    /// Ordered list of (`expression_name`, weight) pairs.
    pub components: Vec<(String, f32)>,
}

impl ExpressionBlend {
    /// Create an empty blend.
    #[must_use]
    pub fn new() -> Self {
        Self {
            components: Vec::new(),
        }
    }

    /// Create a blend with a single expression at the given weight.
    #[must_use]
    pub fn single(name: &str, weight: f32) -> Self {
        Self {
            components: vec![(name.to_owned(), weight)],
        }
    }

    /// Add a component to the blend, returning `&mut self` for fluent chaining.
    pub fn add_component(&mut self, name: &str, weight: f32) -> &mut Self {
        self.components.push((name.to_owned(), weight));
        self
    }

    /// Evaluate the blend to produce a parameter vector of length `num_params`.
    ///
    /// For each component `(name, weight)`, the library expression's parameters
    /// are scaled by `weight * expr.intensity` and accumulated element-wise.
    /// Elements beyond the expression's stored length are treated as zero.
    ///
    /// # Errors
    ///
    /// Returns [`FlameError::InvalidParams`] if any expression name is not
    /// found in `library`.
    pub fn evaluate(
        &self,
        library: &ExpressionLibrary,
        num_params: usize,
    ) -> Result<Vec<f32>, FlameError> {
        let mut result = vec![0.0_f32; num_params];

        for (name, weight) in &self.components {
            let expr = library.get(name).ok_or_else(|| {
                FlameError::InvalidParams(format!("expression '{name}' not found in library"))
            })?;

            let scale = weight * expr.intensity;
            let src = &expr.params;
            let copy_len = src.len().min(num_params);
            for i in 0..copy_len {
                result[i] += src[i] * scale;
            }
        }

        Ok(result)
    }

    /// Return the sum of absolute values of all weights.
    #[must_use]
    pub fn total_weight(&self) -> f32 {
        self.components.iter().map(|(_, w)| w.abs()).sum()
    }

    /// Normalize weights so they sum (in absolute value) to 1.0.
    ///
    /// If total weight is zero, this is a no-op to avoid division by zero.
    pub fn normalize(&mut self) {
        let total = self.total_weight();
        if total > f32::EPSILON {
            for (_, w) in &mut self.components {
                *w /= total;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ExpressionExt — trait for FlameParams
// ---------------------------------------------------------------------------

/// Extension methods for applying expressions to [`FlameParams`].
pub trait ExpressionExt {
    /// Replace the expression parameters with a named expression from `library`,
    /// scaled by `intensity`.
    ///
    /// The resulting expression vector is truncated or zero-padded to match the
    /// length of the existing `expression` field.
    ///
    /// # Errors
    ///
    /// Returns [`FlameError::InvalidParams`] if `name` is not found in `library`.
    fn with_expression(
        self,
        name: &str,
        library: &ExpressionLibrary,
        intensity: f32,
    ) -> Result<Self, FlameError>
    where
        Self: Sized;

    /// Linearly interpolate the expression parameters toward `target` by `t`.
    ///
    /// - `t = 0.0` → unchanged (current expression)
    /// - `t = 1.0` → target expression
    ///
    /// If the current expression vector is shorter than the target, it is
    /// extended with zeros before interpolation. If it is longer, the trailing
    /// coefficients beyond the target's length are **preserved unchanged** —
    /// a short preset therefore edits only the leading coefficients it covers
    /// and never silently fades out the rest. Use
    /// [`ExpressionExt::blend_expression_toward_zero`] for the opposite
    /// convention, where the target is treated as zero-padded and the trailing
    /// coefficients decay toward zero.
    #[must_use]
    fn blend_expression(self, target: &NamedExpression, t: f32) -> Self
    where
        Self: Sized;

    /// Linearly interpolate toward `target`, treating the target as
    /// zero-padded to the current expression length.
    ///
    /// Identical to [`ExpressionExt::blend_expression`] for the leading
    /// coefficients the target covers; trailing coefficients beyond the
    /// target's length are scaled by `1 - t` so that `t = 1.0` yields exactly
    /// the target expression (zero-padded).
    #[must_use]
    fn blend_expression_toward_zero(self, target: &NamedExpression, t: f32) -> Self
    where
        Self: Sized;

    /// Clamp every expression parameter to `[-3.0, 3.0]`.
    #[must_use]
    fn clamp_expression_params(self) -> Self
    where
        Self: Sized;
}

impl ExpressionExt for FlameParams {
    fn with_expression(
        mut self,
        name: &str,
        library: &ExpressionLibrary,
        intensity: f32,
    ) -> Result<Self, FlameError> {
        let expr = library.get(name).ok_or_else(|| {
            FlameError::InvalidParams(format!("expression '{name}' not found in library"))
        })?;

        let current_len = self.expression.len();
        let src = &expr.params;
        let effective_len = if current_len == 0 {
            // No existing expression — use the preset's length
            src.len()
        } else {
            current_len
        };

        let mut new_expr = vec![0.0_f32; effective_len];
        let copy_len = src.len().min(effective_len);
        for i in 0..copy_len {
            new_expr[i] = src[i] * intensity;
        }
        self.expression = new_expr;
        Ok(self)
    }

    fn blend_expression(mut self, target: &NamedExpression, t: f32) -> Self {
        let t = t.clamp(0.0, 1.0);
        let current_len = self.expression.len();
        let target_params = &target.params;

        // Determine output length: at least as long as target
        let out_len = current_len.max(target_params.len());
        if out_len == 0 {
            return self;
        }

        let mut new_expr = vec![0.0_f32; out_len];
        for (i, out) in new_expr.iter_mut().enumerate() {
            let cur = if i < current_len {
                self.expression[i]
            } else {
                0.0
            };
            // Indices the target does not cover keep the current value, as
            // documented on the trait: a short preset must not fade out
            // coefficients it says nothing about.
            *out = match target_params.get(i) {
                Some(&tgt) => cur + (tgt - cur) * t,
                None => cur,
            };
        }
        self.expression = new_expr;
        self
    }

    fn blend_expression_toward_zero(mut self, target: &NamedExpression, t: f32) -> Self {
        let t = t.clamp(0.0, 1.0);
        let current_len = self.expression.len();
        let target_params = &target.params;

        let out_len = current_len.max(target_params.len());
        if out_len == 0 {
            return self;
        }

        let mut new_expr = vec![0.0_f32; out_len];
        for (i, out) in new_expr.iter_mut().enumerate() {
            let cur = if i < current_len {
                self.expression[i]
            } else {
                0.0
            };
            // Target is zero-padded: uncovered indices decay toward zero.
            let tgt = target_params.get(i).copied().unwrap_or(0.0);
            *out = cur + (tgt - cur) * t;
        }
        self.expression = new_expr;
        self
    }

    fn clamp_expression_params(mut self) -> Self {
        const EXPR_MIN: f32 = -3.0;
        const EXPR_MAX: f32 = 3.0;
        for v in &mut self.expression {
            *v = v.clamp(EXPR_MIN, EXPR_MAX);
        }
        self
    }
}

// ---------------------------------------------------------------------------
// ConstraintViolation
// ---------------------------------------------------------------------------

/// A single violated constraint on a [`FlameParams`] field.
#[derive(Debug, Clone)]
pub struct ConstraintViolation {
    /// Name of the violated field (e.g., `"shape[2]"`, `"pose[6]"`).
    pub field: String,
    /// Actual value that violated the constraint.
    pub value: f32,
    /// Lower bound of the valid range.
    pub min: f32,
    /// Upper bound of the valid range.
    pub max: f32,
    /// Human-readable explanation.
    pub message: String,
}

impl ConstraintViolation {
    fn new(field: impl Into<String>, value: f32, min: f32, max: f32) -> Self {
        let field = field.into();
        let message = format!("field '{field}' value {value:.4} is outside [{min:.4}, {max:.4}]");
        Self {
            field,
            value,
            min,
            max,
            message,
        }
    }
}

// ---------------------------------------------------------------------------
// FlameParamConstraints
// ---------------------------------------------------------------------------

/// Validity constraints on FLAME parameters.
///
/// Use the associated constructors to get preset constraint sets, or build
/// your own by mutating the public fields.
///
/// # Example
///
/// ```rust
/// use oxigaf_flame::{FlameParamConstraints, FlameParams};
///
/// let params = FlameParams::neutral();
/// let constraints = FlameParamConstraints::strict();
/// let violations = constraints.validate(&params);
/// assert!(violations.is_empty(), "neutral face must satisfy all constraints");
/// ```
#[derive(Debug, Clone)]
pub struct FlameParamConstraints {
    /// Maximum absolute value for shape coefficients. Default: 3.0.
    pub max_shape_abs: f32,
    /// Maximum absolute value for expression coefficients. Default: 3.0.
    pub max_expression_abs: f32,
    /// Maximum jaw opening angle in radians. Default: 0.5 (~28°).
    pub max_jaw_angle_rad: f32,
    /// Minimum jaw angle in radians (slight over-closure). Default: -0.05.
    pub min_jaw_angle_rad: f32,
    /// Maximum absolute value for any component of the global (root) rotation.
    /// Default: 1.57 (90°).
    pub max_global_rotation_abs_rad: f32,
    /// Maximum absolute value for translation components (in metres).
    /// Default: 1.0.
    pub max_translation_abs: f32,
}

impl FlameParamConstraints {
    /// Default constraints — reasonable bounds for typical FLAME usage.
    #[must_use]
    pub fn default_constraints() -> Self {
        Self {
            max_shape_abs: 3.0,
            max_expression_abs: 3.0,
            max_jaw_angle_rad: 0.5,
            min_jaw_angle_rad: -0.05,
            max_global_rotation_abs_rad: 1.57,
            max_translation_abs: 1.0,
        }
    }

    /// Stricter bounds for high-quality, tightly controlled outputs.
    #[must_use]
    pub fn strict() -> Self {
        Self {
            max_shape_abs: 3.0,
            max_expression_abs: 2.0,
            max_jaw_angle_rad: 0.4,
            min_jaw_angle_rad: -0.05,
            max_global_rotation_abs_rad: 1.57,
            max_translation_abs: 1.0,
        }
    }

    /// Permissive bounds for artistic or exaggerated expressions.
    #[must_use]
    pub fn permissive() -> Self {
        Self {
            max_shape_abs: 3.0,
            max_expression_abs: 5.0,
            max_jaw_angle_rad: 0.7,
            min_jaw_angle_rad: -0.05,
            max_global_rotation_abs_rad: 1.57,
            max_translation_abs: 1.0,
        }
    }

    /// Check `params` against all constraints, returning every violation found.
    ///
    /// An empty `Vec` means the parameters are valid under these constraints.
    #[must_use]
    pub fn validate(&self, params: &FlameParams) -> Vec<ConstraintViolation> {
        let mut violations = Vec::new();

        // Shape
        for (i, &s) in params.shape.iter().enumerate() {
            if s.abs() > self.max_shape_abs {
                violations.push(ConstraintViolation::new(
                    format!("shape[{i}]"),
                    s,
                    -self.max_shape_abs,
                    self.max_shape_abs,
                ));
            }
        }

        // Expression
        for (i, &e) in params.expression.iter().enumerate() {
            if e.abs() > self.max_expression_abs {
                violations.push(ConstraintViolation::new(
                    format!("expression[{i}]"),
                    e,
                    -self.max_expression_abs,
                    self.max_expression_abs,
                ));
            }
        }

        // Global (root) rotation — joint 0, pose[0..3]
        for j in 0..3usize {
            if let Some(&v) = params.pose.get(j) {
                if v.abs() > self.max_global_rotation_abs_rad {
                    violations.push(ConstraintViolation::new(
                        format!("pose[{j}] (root_rotation)"),
                        v,
                        -self.max_global_rotation_abs_rad,
                        self.max_global_rotation_abs_rad,
                    ));
                }
            }
        }

        // Jaw — joint 2, pose[6] is the primary opening axis
        if let Some(&jaw) = params.pose.get(6) {
            if jaw > self.max_jaw_angle_rad || jaw < self.min_jaw_angle_rad {
                violations.push(ConstraintViolation::new(
                    "pose[6] (jaw_angle)",
                    jaw,
                    self.min_jaw_angle_rad,
                    self.max_jaw_angle_rad,
                ));
            }
        }

        // Translation
        for (i, &t) in params.translation.iter().enumerate() {
            if t.abs() > self.max_translation_abs {
                violations.push(ConstraintViolation::new(
                    format!("translation[{i}]"),
                    t,
                    -self.max_translation_abs,
                    self.max_translation_abs,
                ));
            }
        }

        violations
    }

    /// Clamp all parameters to valid ranges, returning a new [`FlameParams`].
    #[must_use]
    pub fn clamp(&self, mut params: FlameParams) -> FlameParams {
        // Shape
        for s in &mut params.shape {
            *s = s.clamp(-self.max_shape_abs, self.max_shape_abs);
        }

        // Expression
        for e in &mut params.expression {
            *e = e.clamp(-self.max_expression_abs, self.max_expression_abs);
        }

        // Global rotation (pose[0..3])
        for j in 0..3usize {
            if let Some(v) = params.pose.get_mut(j) {
                *v = v.clamp(
                    -self.max_global_rotation_abs_rad,
                    self.max_global_rotation_abs_rad,
                );
            }
        }

        // Jaw opening (pose[6])
        if let Some(jaw) = params.pose.get_mut(6) {
            *jaw = jaw.clamp(self.min_jaw_angle_rad, self.max_jaw_angle_rad);
        }

        // Translation
        for t in &mut params.translation {
            *t = t.clamp(-self.max_translation_abs, self.max_translation_abs);
        }

        params
    }
}

impl Default for FlameParamConstraints {
    fn default() -> Self {
        Self::default_constraints()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: create FlameParams with specified expression vector
    fn params_with_expr(expr: Vec<f32>) -> FlameParams {
        FlameParams {
            shape: Vec::new(),
            expression: expr,
            pose: vec![0.0; 15],
            translation: [0.0; 3],
        }
    }

    // -----------------------------------------------------------------------
    // ExpressionLibrary tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_library_placeholder_expressions_count() {
        let lib = ExpressionLibrary::placeholder_expressions();
        // Verify all 11 built-in presets are present
        assert_eq!(lib.count(), 11, "expected 11 built-in presets");
    }

    #[test]
    fn test_placeholder_library_is_flagged_as_placeholder() {
        let lib = ExpressionLibrary::placeholder_expressions();
        assert!(
            lib.has_placeholders(),
            "hand-authored presets must report themselves as placeholders"
        );
        let names = lib.placeholder_names();
        // Everything except "neutral" (exact for any basis) is a placeholder.
        assert_eq!(
            names.len(),
            10,
            "expected 10 placeholder entries: {names:?}"
        );
        assert!(
            !names.contains(&"neutral"),
            "zero coefficients are exact for any basis, so neutral is not a placeholder"
        );
        assert!(names.contains(&"winking"));
        let neutral = lib.get("neutral").expect("neutral must exist");
        assert_eq!(neutral.provenance, ExpressionProvenance::Fitted);
        let smile = lib.get("smile").expect("smile must exist");
        assert_eq!(smile.provenance, ExpressionProvenance::Placeholder);
    }

    #[test]
    fn test_fit_to_basis_recovers_known_coefficients() {
        // Two orthogonal displacement directions over 2 vertices (6 scalars).
        let neutral = vec![0.0_f32; 6];
        let basis = vec![
            vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        ];
        // Target = 0.5 * basis[0] + (-0.25) * basis[1]
        let target = vec![0.5, -0.25, 0.0, 0.5, -0.25, 0.0];

        let expr = NamedExpression::fit_to_basis(
            "fitted_smile",
            "fitted against a synthetic basis",
            &neutral,
            &target,
            basis,
            2,
        )
        .expect("fit must succeed");

        assert_eq!(expr.provenance, ExpressionProvenance::Fitted);
        assert_eq!(expr.params.len(), 2);
        assert!(
            (expr.params[0] - 0.5).abs() < 0.05,
            "coefficient 0 should be ~0.5, got {}",
            expr.params[0]
        );
        assert!(
            (expr.params[1] + 0.25).abs() < 0.05,
            "coefficient 1 should be ~-0.25, got {}",
            expr.params[1]
        );
    }

    #[test]
    fn test_fit_to_basis_rejects_mismatched_inputs() {
        let neutral = vec![0.0_f32; 6];
        let target = vec![0.0_f32; 3]; // wrong length
        let basis = vec![vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0]];
        let result = NamedExpression::fit_to_basis("bad", "bad", &neutral, &target, basis, 1);
        assert!(
            matches!(result, Err(FlameError::InvalidParams(_))),
            "mismatched vertex counts must be rejected"
        );
    }

    #[test]
    fn test_library_json_roundtrip() {
        let lib = ExpressionLibrary::placeholder_expressions();
        let json = lib.to_json_string().expect("serialisation must succeed");
        let restored = ExpressionLibrary::from_json_str(&json).expect("parsing must succeed");

        assert_eq!(restored.count(), lib.count());
        let smile = restored.get("smile").expect("smile must survive roundtrip");
        assert_eq!(smile.provenance, ExpressionProvenance::Placeholder);
        assert!((smile.params[1] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_library_from_json_file_roundtrip() {
        let lib = ExpressionLibrary::placeholder_expressions();
        let json = lib.to_json_string().expect("serialisation must succeed");

        let pid = std::process::id();
        let mut path = std::env::temp_dir();
        path.push(format!("oxigaf_expr_lib_roundtrip_{pid}.json"));
        std::fs::write(&path, json).expect("test: writing temp file must succeed");

        let restored = ExpressionLibrary::from_json_file(&path).expect("loading must succeed");
        assert_eq!(restored.count(), lib.count());

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_library_from_json_str_rejects_garbage() {
        let result = ExpressionLibrary::from_json_str("{ not json ]");
        assert!(
            matches!(result, Err(FlameError::InvalidParams(_))),
            "malformed JSON must be rejected"
        );
    }

    #[test]
    fn test_library_get_neutral() {
        let lib = ExpressionLibrary::placeholder_expressions();
        let expr = lib.get("neutral").expect("neutral must exist");
        assert_eq!(expr.name, "neutral");
        assert!(
            expr.params.iter().all(|&v| v == 0.0),
            "neutral params must all be zero"
        );
    }

    #[test]
    fn test_library_get_smile() {
        let lib = ExpressionLibrary::placeholder_expressions();
        let expr = lib.get("smile").expect("smile must exist");
        // params[1] = 1.5 per spec
        assert!(
            (expr.params[1] - 1.5).abs() < f32::EPSILON,
            "smile params[1] must be 1.5, got {}",
            expr.params[1]
        );
    }

    #[test]
    fn test_library_get_nonexistent_returns_none() {
        let lib = ExpressionLibrary::placeholder_expressions();
        assert!(lib.get("does_not_exist").is_none());
    }

    #[test]
    fn test_library_names_contains_all() {
        let lib = ExpressionLibrary::placeholder_expressions();
        let names = lib.names();
        let expected = [
            "neutral",
            "smile",
            "grin",
            "frown",
            "surprised",
            "angry",
            "sad",
            "disgusted",
            "fearful",
            "winking",
            "open_mouth",
        ];
        for &expected_name in &expected {
            assert!(
                names.contains(&expected_name),
                "library must contain '{expected_name}'"
            );
        }
    }

    // -----------------------------------------------------------------------
    // ExpressionBlend tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_expression_blend_single() {
        let blend = ExpressionBlend::single("smile", 1.0);
        assert_eq!(blend.components.len(), 1);
        assert_eq!(blend.components[0].0, "smile");
        assert!((blend.components[0].1 - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_expression_blend_add_component() {
        let mut blend = ExpressionBlend::new();
        blend.add_component("smile", 0.5).add_component("sad", 0.3);
        assert_eq!(blend.components.len(), 2);
        assert_eq!(blend.components[0].0, "smile");
        assert_eq!(blend.components[1].0, "sad");
    }

    #[test]
    fn test_expression_blend_evaluate_neutral_is_zeros() {
        let lib = ExpressionLibrary::placeholder_expressions();
        let result = ExpressionBlend::single("neutral", 1.0)
            .evaluate(&lib, 10)
            .expect("evaluate must succeed");
        assert_eq!(result.len(), 10);
        assert!(
            result.iter().all(|&v| v == 0.0),
            "neutral expression must produce all zeros"
        );
    }

    #[test]
    fn test_expression_blend_evaluate_smile() {
        let lib = ExpressionLibrary::placeholder_expressions();
        let result = ExpressionBlend::single("smile", 1.0)
            .evaluate(&lib, 10)
            .expect("evaluate must succeed");
        // smile params[1] = 1.5, intensity = 1.0, weight = 1.0 → result[1] = 1.5
        assert!(
            (result[1] - 1.5).abs() < 1e-6,
            "result[1] must be 1.5, got {}",
            result[1]
        );
        // smile params[0] = 0.3
        assert!(
            (result[0] - 0.3).abs() < 1e-6,
            "result[0] must be 0.3, got {}",
            result[0]
        );
    }

    #[test]
    fn test_expression_blend_evaluate_missing_name() {
        let lib = ExpressionLibrary::placeholder_expressions();
        let result = ExpressionBlend::single("nonexistent_expression", 1.0).evaluate(&lib, 10);
        assert!(result.is_err(), "missing expression must produce an error");
    }

    #[test]
    fn test_expression_blend_total_weight() {
        let mut blend = ExpressionBlend::new();
        blend
            .add_component("smile", 0.6)
            .add_component("frown", -0.4);
        // sum of |weights| = 0.6 + 0.4 = 1.0
        let total = blend.total_weight();
        assert!(
            (total - 1.0).abs() < 1e-6,
            "total_weight must be 1.0, got {total}"
        );
    }

    #[test]
    fn test_expression_blend_normalize() {
        let mut blend = ExpressionBlend::new();
        blend.add_component("smile", 3.0).add_component("grin", 1.0);
        // total absolute weight = 4.0
        blend.normalize();
        let total = blend.total_weight();
        assert!(
            (total - 1.0).abs() < 1e-6,
            "after normalize, total_weight must be 1.0, got {total}"
        );
        // Individual weights: 3/4 and 1/4
        assert!(
            (blend.components[0].1 - 0.75).abs() < 1e-6,
            "first weight must be 0.75"
        );
        assert!(
            (blend.components[1].1 - 0.25).abs() < 1e-6,
            "second weight must be 0.25"
        );
    }

    // -----------------------------------------------------------------------
    // FlameParams ExpressionExt tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_flame_params_with_expression() {
        use crate::expressions::ExpressionExt;
        let lib = ExpressionLibrary::placeholder_expressions();
        let params = params_with_expr(vec![0.0; 10]);
        let result = params
            .with_expression("smile", &lib, 1.0)
            .expect("with_expression must succeed");
        // params[1] = 1.5
        assert!(
            (result.expression[1] - 1.5).abs() < 1e-6,
            "expression[1] must be 1.5 after applying smile"
        );
    }

    #[test]
    fn test_flame_params_blend_expression() {
        use crate::expressions::ExpressionExt;
        let lib = ExpressionLibrary::placeholder_expressions();
        let smile = lib.get("smile").expect("smile must exist").clone();

        let params = params_with_expr(vec![0.0; 10]);
        // Blend 50% toward smile
        let result = params.blend_expression(&smile, 0.5);
        // At t=0.5, result[1] = 0.0 + (1.5 - 0.0) * 0.5 = 0.75
        assert!(
            (result.expression[1] - 0.75).abs() < 1e-6,
            "expression[1] must be 0.75 at t=0.5, got {}",
            result.expression[1]
        );
    }

    #[test]
    fn test_blend_expression_preserves_trailing_coefficients() {
        use crate::expressions::ExpressionExt;
        // 50-dim current expression, 10-dim target: coefficients 10..50 must be
        // left untouched, exactly as the trait doc promises.
        let current: Vec<f32> = (0..50).map(|i| 0.1 * (i as f32 + 1.0)).collect();
        let params = params_with_expr(current.clone());
        let target = NamedExpression::new("short", "10-dim preset", vec![1.0; 10], 1.0);

        let result = params.blend_expression(&target, 0.5);
        assert_eq!(result.expression.len(), 50);

        // Leading 10: interpolated halfway toward 1.0
        for (i, (&cur, &res)) in current
            .iter()
            .zip(result.expression.iter())
            .enumerate()
            .take(10)
        {
            let expected = cur + (1.0 - cur) * 0.5;
            assert!(
                (res - expected).abs() < 1e-6,
                "expression[{i}] must be {expected}, got {res}"
            );
        }
        // Trailing 40: bit-for-bit unchanged
        for (i, (&cur, &res)) in current
            .iter()
            .zip(result.expression.iter())
            .enumerate()
            .skip(10)
        {
            assert!(
                (res - cur).abs() < f32::EPSILON,
                "expression[{i}] must stay at {cur}, got {res}"
            );
        }
    }

    #[test]
    fn test_blend_expression_toward_zero_fades_trailing_coefficients() {
        use crate::expressions::ExpressionExt;
        let current: Vec<f32> = (0..20).map(|i| 0.1 * (i as f32 + 1.0)).collect();
        let params = params_with_expr(current.clone());
        let target = NamedExpression::new("short", "10-dim preset", vec![1.0; 10], 1.0);

        let result = params.blend_expression_toward_zero(&target, 0.5);
        assert_eq!(result.expression.len(), 20);
        // Trailing coefficients are treated as target = 0 and halve.
        for (i, (&cur, &res)) in current
            .iter()
            .zip(result.expression.iter())
            .enumerate()
            .skip(10)
        {
            let expected = cur * 0.5;
            assert!(
                (res - expected).abs() < 1e-6,
                "expression[{i}] must be {expected}, got {res}"
            );
        }
        // At t = 1.0 the result is exactly the zero-padded target.
        let params2 = params_with_expr(current);
        let full = params2.blend_expression_toward_zero(&target, 1.0);
        for i in 10..20 {
            assert!(full.expression[i].abs() < 1e-6);
        }
    }

    #[test]
    fn test_blend_expression_extends_shorter_current_with_zeros() {
        use crate::expressions::ExpressionExt;
        // Current shorter than target: extended with zeros before interpolation.
        let params = params_with_expr(vec![2.0, 2.0]);
        let target = NamedExpression::new("wide", "4-dim preset", vec![0.0, 0.0, 4.0, 4.0], 1.0);
        let result = params.blend_expression(&target, 0.5);
        assert_eq!(result.expression.len(), 4);
        assert!((result.expression[0] - 1.0).abs() < 1e-6);
        assert!((result.expression[2] - 2.0).abs() < 1e-6);
        assert!((result.expression[3] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_flame_params_clamp_expression_params() {
        use crate::expressions::ExpressionExt;
        let params = params_with_expr(vec![5.0, -4.0, 1.5, -3.0]);
        let result = params.clamp_expression_params();
        assert!(
            (result.expression[0] - 3.0).abs() < f32::EPSILON,
            "5.0 must be clamped to 3.0"
        );
        assert!(
            (result.expression[1] + 3.0).abs() < f32::EPSILON,
            "-4.0 must be clamped to -3.0"
        );
        assert!(
            (result.expression[2] - 1.5).abs() < f32::EPSILON,
            "1.5 must be unchanged"
        );
        assert!(
            (result.expression[3] + 3.0).abs() < f32::EPSILON,
            "-3.0 must remain at -3.0 (boundary)"
        );
    }

    // -----------------------------------------------------------------------
    // FlameParamConstraints tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_constraints_default() {
        let c = FlameParamConstraints::default();
        assert!((c.max_shape_abs - 3.0).abs() < f32::EPSILON);
        assert!((c.max_expression_abs - 3.0).abs() < f32::EPSILON);
        assert!((c.max_jaw_angle_rad - 0.5).abs() < f32::EPSILON);
        assert!((c.min_jaw_angle_rad + 0.05).abs() < f32::EPSILON);
        assert!((c.max_global_rotation_abs_rad - 1.57).abs() < f32::EPSILON);
        assert!((c.max_translation_abs - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_constraints_validate_valid_params() {
        let c = FlameParamConstraints::default();
        let params = FlameParams::neutral();
        let violations = c.validate(&params);
        assert!(
            violations.is_empty(),
            "neutral params must have no violations, got: {:?}",
            violations.iter().map(|v| &v.message).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_constraints_validate_violation_detected() {
        let c = FlameParamConstraints::default();
        // expression[0] = 10.0 is way outside [-3, 3]
        let params = params_with_expr(vec![10.0]);
        let violations = c.validate(&params);
        assert!(
            !violations.is_empty(),
            "out-of-range expression must produce violations"
        );
        assert_eq!(
            violations[0].field, "expression[0]",
            "violation field must identify expression[0]"
        );
        assert!((violations[0].value - 10.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_constraints_clamp() {
        let c = FlameParamConstraints::default();
        let mut params = FlameParams::neutral();
        params.expression = vec![8.0, -7.0, 1.0];
        params.shape = vec![5.0, -5.0];
        params.translation = [2.0, -2.0, 0.5];

        let clamped = c.clamp(params);

        assert!(
            (clamped.expression[0] - 3.0).abs() < f32::EPSILON,
            "expression[0] must clamp to 3.0"
        );
        assert!(
            (clamped.expression[1] + 3.0).abs() < f32::EPSILON,
            "expression[1] must clamp to -3.0"
        );
        assert!(
            (clamped.expression[2] - 1.0).abs() < f32::EPSILON,
            "expression[2] must stay at 1.0"
        );
        assert!(
            (clamped.shape[0] - 3.0).abs() < f32::EPSILON,
            "shape[0] must clamp to 3.0"
        );
        assert!(
            (clamped.shape[1] + 3.0).abs() < f32::EPSILON,
            "shape[1] must clamp to -3.0"
        );
        assert!(
            (clamped.translation[0] - 1.0).abs() < f32::EPSILON,
            "translation[0] must clamp to 1.0"
        );
        assert!(
            (clamped.translation[1] + 1.0).abs() < f32::EPSILON,
            "translation[1] must clamp to -1.0"
        );
        assert!(
            (clamped.translation[2] - 0.5).abs() < f32::EPSILON,
            "translation[2] must stay at 0.5"
        );
    }
}
