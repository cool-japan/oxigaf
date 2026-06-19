//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

/// Errors that can occur during PLY I/O.
#[derive(Debug, thiserror::Error)]
pub enum PlyError {
    /// Underlying I/O failure.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    /// The PLY header is malformed or truncated.
    #[error("invalid PLY header: {0}")]
    InvalidHeader(String),
    /// The format specifier in the header is not supported by this parser.
    #[error("unsupported format: {0}")]
    UnsupportedFormat(String),
    /// A required PLY property was not found.
    #[error("property not found: {0}")]
    PropertyNotFound(String),
    /// A data dimension doesn't match expectations.
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
    /// The scene contains no Gaussians (write would produce an empty file).
    #[error("empty scene: no Gaussians to export")]
    EmptyScene,
    /// A data value is invalid (e.g. NaN, non-standard SH layout).
    #[error("invalid data: {0}")]
    InvalidData(String),
}
/// Scene data returned from `ply_import_scene`.
pub struct PlySceneData {
    /// Flat positions: `[N×3]` as x, y, z.
    pub positions: Vec<f32>,
    /// Flat rotations: `[N×4]` as **qx, qy, qz, qw** (training convention).
    pub rotations: Vec<f32>,
    /// Flat log-space scales: `[N×3]`.
    pub scales: Vec<f32>,
    /// Flat logit opacities: `[N]`.
    pub opacities: Vec<f32>,
    /// Flat SH DC: `[N×3]`.
    pub sh_dc: Vec<f32>,
    /// Flat SH rest: `[N×C]`.
    pub sh_rest: Vec<f32>,
    /// Number of Gaussians.
    pub n_gaussians: usize,
    /// Number of SH rest coefficients per Gaussian.
    pub n_rest_per_gaussian: usize,
}
/// Per-scene summary statistics derived from a list of [`PlyGaussian`]s.
pub struct PlySceneStats {
    /// Number of Gaussians in the scene.
    pub n_gaussians: usize,
    /// SH degree inferred from rest coefficient count.
    pub sh_degree: usize,
    /// Mean opacity after sigmoid, across all Gaussians.
    pub mean_opacity: f32,
    /// Mean world-space scale (averaged over x/y/z and all Gaussians).
    pub mean_scale: f32,
    /// Axis-aligned bounding box minimum (x, y, z).
    pub bbox_min: [f32; 3],
    /// Axis-aligned bounding box maximum (x, y, z).
    pub bbox_max: [f32; 3],
}
/// Statistics reported after writing a PLY file.
pub struct PlyWriteStats {
    /// Number of Gaussians written.
    pub n_gaussians: usize,
    /// Total number of floating-point properties per Gaussian.
    pub n_properties: usize,
    /// File size in bytes after writing.
    pub file_size_bytes: u64,
    /// SH degree inferred from the rest coefficient count.
    pub sh_degree: usize,
    /// PLY format used.
    pub format: PlyFormat,
}
/// Scene data plus export format for `ply_export_scene`.
pub struct PlyExportParams<'a> {
    /// Positions, flat N×3.
    pub positions: &'a [f32],
    /// Rotations (qx, qy, qz, qw), flat N×4.
    pub rotations: &'a [f32],
    /// Log-scales, flat N×3.
    pub scales: &'a [f32],
    /// Logit-space opacities, length N.
    pub opacities: &'a [f32],
    /// DC SH coefficients, flat N×3.
    pub sh_dc: &'a [f32],
    /// Rest SH coefficients, flat N×n_rest_per_gaussian.
    pub sh_rest: &'a [f32],
    /// Number of rest SH coefficients per Gaussian (0, 9, 24, or 45).
    pub n_rest_per_gaussian: usize,
    /// PLY encoding format.
    pub format: PlyFormat,
}
/// Statistics reported after reading a PLY file.
pub struct PlyReadStats {
    /// Number of Gaussians read.
    pub n_gaussians: usize,
    /// Total number of floating-point properties per Gaussian.
    pub n_properties: usize,
    /// SH degree inferred from the rest coefficient count.
    pub sh_degree: usize,
    /// PLY format found in the file.
    pub format: PlyFormat,
}
/// The encoding variant of a PLY file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlyFormat {
    /// Human-readable ASCII.
    Ascii,
    /// Binary, IEEE 754 little-endian f32.
    BinaryLittleEndian,
    /// Binary, IEEE 754 big-endian f32.
    BinaryBigEndian,
}
impl PlyFormat {
    /// Returns the canonical PLY header format string for this variant.
    pub fn as_ply_str(&self) -> &'static str {
        match self {
            Self::Ascii => "ascii",
            Self::BinaryLittleEndian => "binary_little_endian",
            Self::BinaryBigEndian => "binary_big_endian",
        }
    }
}
/// A single 3D Gaussian in PLY-compatible representation.
///
/// This matches the property layout of the original Kerbl et al. 3DGS
/// implementation. Fields are stored in raw (log-space / logit-space) form
/// for lossless serialisation.
#[derive(Debug, Clone)]
pub struct PlyGaussian {
    /// World-space position.
    pub x: f32,
    /// World-space position.
    pub y: f32,
    /// World-space position.
    pub z: f32,
    /// Surface normal X (always 0 in 3DGS, required for PLY compatibility).
    pub nx: f32,
    /// Surface normal Y.
    pub ny: f32,
    /// Surface normal Z.
    pub nz: f32,
    /// SH DC term: [f_dc_0, f_dc_1, f_dc_2] (1 coeff × 3 colour channels).
    pub f_dc: [f32; 3],
    /// SH higher-order rest coefficients.
    /// Length = [`PlyGaussian::n_rest_coeffs`] for the given SH degree.
    pub f_rest: Vec<f32>,
    /// Opacity in logit space.  `real_opacity = sigmoid(opacity)`.
    pub opacity: f32,
    /// Scale in log space: [scale_0, scale_1, scale_2].
    /// `real_scale = exp(scale)`.
    pub scale: [f32; 3],
    /// Rotation quaternion in **wxyz** order: [rot_0=w, rot_1=x, rot_2=y, rot_3=z].
    pub rot: [f32; 4],
}
impl PlyGaussian {
    /// Number of SH rest coefficients for a given SH degree.
    ///
    /// ```text
    /// f(d) = (d+1)^2 × 3 − 3
    /// d=0 →  0, d=1 →  9, d=2 → 24, d=3 → 45
    /// ```
    #[must_use]
    pub fn n_rest_coeffs(sh_degree: usize) -> usize {
        let total = (sh_degree + 1) * (sh_degree + 1) * 3;
        total.saturating_sub(3)
    }
    /// Create a default Gaussian at the origin with identity rotation.
    #[must_use]
    pub fn identity() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            nx: 0.0,
            ny: 0.0,
            nz: 0.0,
            f_dc: [0.0; 3],
            f_rest: Vec::new(),
            opacity: 0.0,
            scale: [0.0; 3],
            rot: [1.0, 0.0, 0.0, 0.0],
        }
    }
    /// Sigmoid of `self.opacity` → actual opacity in [0, 1].
    #[inline]
    #[must_use]
    pub fn real_opacity(&self) -> f32 {
        1.0_f32 / (1.0_f32 + (-self.opacity).exp())
    }
    /// `exp(self.scale)` → actual scale in world space.
    #[inline]
    #[must_use]
    pub fn real_scale(&self) -> [f32; 3] {
        [
            self.scale[0].exp(),
            self.scale[1].exp(),
            self.scale[2].exp(),
        ]
    }
    /// Construct a [`PlyGaussian`] from flat training arrays.
    ///
    /// # Array conventions
    ///
    /// | Field           | Layout                                       |
    /// |-----------------|----------------------------------------------|
    /// | `slices.positions` | `[N×3]` — x, y, z per Gaussian           |
    /// | `slices.rotations` | `[N×4]` — **qx, qy, qz, qw** per Gaussian |
    /// | `slices.scales`    | `[N×3]` — log-space sx, sy, sz            |
    /// | `slices.opacities` | `[N]`   — logit opacity                   |
    /// | `slices.sh_dc`     | `[N×3]` — DC SH coefficients              |
    /// | `slices.sh_rest`   | `[N×C]` — rest SH, C = `n_rest` per Gaussian |
    /// | `idx`              | Gaussian index in `[0, N)`                |
    ///
    /// Note: input rotation order is `(qx, qy, qz, qw)` but `PlyGaussian.rot`
    /// is stored as `(w, x, y, z)`.
    pub fn from_flat(slices: PlyFlatSlices<'_>, idx: usize) -> Result<Self, PlyError> {
        let PlyFlatSlices {
            positions,
            rotations,
            scales,
            opacities,
            sh_dc,
            sh_rest,
            n_rest,
        } = slices;
        let n = opacities.len();
        if positions.len() < (idx + 1) * 3 {
            return Err(PlyError::DimensionMismatch {
                expected: (idx + 1) * 3,
                got: positions.len(),
            });
        }
        if rotations.len() < (idx + 1) * 4 {
            return Err(PlyError::DimensionMismatch {
                expected: (idx + 1) * 4,
                got: rotations.len(),
            });
        }
        if scales.len() < (idx + 1) * 3 {
            return Err(PlyError::DimensionMismatch {
                expected: (idx + 1) * 3,
                got: scales.len(),
            });
        }
        if idx >= n {
            return Err(PlyError::DimensionMismatch {
                expected: idx + 1,
                got: n,
            });
        }
        if sh_dc.len() < (idx + 1) * 3 {
            return Err(PlyError::DimensionMismatch {
                expected: (idx + 1) * 3,
                got: sh_dc.len(),
            });
        }
        if n_rest > 0 && sh_rest.len() < (idx + 1) * n_rest {
            return Err(PlyError::DimensionMismatch {
                expected: (idx + 1) * n_rest,
                got: sh_rest.len(),
            });
        }
        let pi = idx * 3;
        let x = positions[pi];
        let y = positions[pi + 1];
        let z = positions[pi + 2];
        let ri = idx * 4;
        let qx = rotations[ri];
        let qy = rotations[ri + 1];
        let qz = rotations[ri + 2];
        let qw = rotations[ri + 3];
        let rot = [qw, qx, qy, qz];
        let si = idx * 3;
        let scale = [scales[si], scales[si + 1], scales[si + 2]];
        let opacity = opacities[idx];
        let di = idx * 3;
        let f_dc = [sh_dc[di], sh_dc[di + 1], sh_dc[di + 2]];
        let f_rest = if n_rest > 0 {
            let ri_start = idx * n_rest;
            sh_rest[ri_start..ri_start + n_rest].to_vec()
        } else {
            Vec::new()
        };
        Ok(Self {
            x,
            y,
            z,
            nx: 0.0,
            ny: 0.0,
            nz: 0.0,
            f_dc,
            f_rest,
            opacity,
            scale,
            rot,
        })
    }
}
/// Flat training-array slices passed to [`PlyGaussian::from_flat`].
pub struct PlyFlatSlices<'a> {
    /// Positions, flat N×3.
    pub positions: &'a [f32],
    /// Rotations (qx, qy, qz, qw), flat N×4.
    pub rotations: &'a [f32],
    /// Log-scales, flat N×3.
    pub scales: &'a [f32],
    /// Logit-space opacities, length N.
    pub opacities: &'a [f32],
    /// DC SH coefficients, flat N×3.
    pub sh_dc: &'a [f32],
    /// Rest SH coefficients, flat N×n_rest.
    pub sh_rest: &'a [f32],
    /// Number of rest SH coefficients per Gaussian.
    pub n_rest: usize,
}
