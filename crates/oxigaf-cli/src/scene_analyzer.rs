//! Scene analysis utilities for 3D Gaussian Splatting scenes.
//!
//! Computes spatial statistics, opacity distributions, scale profiles,
//! color statistics, and a composite quality score, for automated quality
//! assessment pipelines.
//!
//! Not currently wired into any `oxigaf` subcommand — `oxigaf info`
//! (`crates/oxigaf-cli/src/info.rs`) computes its own statistics by hand
//! rather than calling into this module. Call [`analyze_scene`] directly if
//! you need these statistics from library code.
//!
//! # Example
//! ```rust,no_run
//! use oxigaf_cli::scene_analyzer::{SceneData, analyze_scene};
//!
//! let scene = SceneData {
//!     positions: vec![0.0, 0.0, 0.0],
//!     log_scales: vec![-2.0, -2.0, -2.0],
//!     rotations: vec![0.0, 0.0, 0.0, 1.0],
//!     opacities: vec![0.0],
//!     colors: vec![0.5, 0.5, 0.5],
//! };
//! let report = analyze_scene(&scene).expect("analysis failed");
//! println!("{}", report.format_summary());
//! ```

use thiserror::Error;

// ---------------------------------------------------------------------------
// AnalysisError
// ---------------------------------------------------------------------------

/// Errors that can arise during scene analysis.
#[derive(Debug, Error, PartialEq)]
pub enum AnalysisError {
    /// The scene contains no Gaussians.
    #[error("Scene is empty")]
    EmptyScene,

    /// A field contains invalid data (e.g. NaN values).
    #[error("Invalid data: {0}")]
    InvalidData(String),

    /// A field length is inconsistent with the number of Gaussians.
    #[error("Dimension mismatch for '{field}': expected {expected}, got {got}")]
    DimensionError {
        field: String,
        expected: usize,
        got: usize,
    },
}

// ---------------------------------------------------------------------------
// SceneData
// ---------------------------------------------------------------------------

/// Raw 3DGS scene data for analysis.
#[derive(Debug, Clone)]
pub struct SceneData {
    /// Gaussian positions [x, y, z] interleaved, len = n * 3.
    pub positions: Vec<f32>,
    /// Log-scale [sx, sy, sz] per Gaussian, len = n * 3.
    pub log_scales: Vec<f32>,
    /// Quaternion rotations [qx, qy, qz, qw] per Gaussian, len = n * 4.
    pub rotations: Vec<f32>,
    /// Logit-space opacity per Gaussian, len = n.
    pub opacities: Vec<f32>,
    /// RGB color (or SH DC) per Gaussian [r, g, b], len = n * 3.
    pub colors: Vec<f32>,
}

/// Returns `InvalidData` naming the first non-finite (NaN or Inf) entry in
/// `values`, if any.
fn check_all_finite(field: &str, values: &[f32]) -> Result<(), AnalysisError> {
    if let Some((i, v)) = values.iter().enumerate().find(|(_, v)| !v.is_finite()) {
        return Err(AnalysisError::InvalidData(format!(
            "field '{field}' contains a non-finite value ({v}) at index {i}"
        )));
    }
    Ok(())
}

impl SceneData {
    /// Number of Gaussians in the scene (derived from opacities length).
    #[must_use]
    pub fn num_gaussians(&self) -> usize {
        self.opacities.len()
    }

    /// Validate that all field lengths are consistent with `num_gaussians()`
    /// and that every numeric field is finite.
    ///
    /// Returns `DimensionError` for any field whose length does not match the
    /// expected multiple, or `InvalidData` if any field contains a NaN or
    /// infinite value — such values would otherwise silently propagate into
    /// every downstream statistic (bounding box, centroid, quality score)
    /// and produce a `to_json()` payload with bare `NaN`/`inf` tokens that no
    /// JSON parser accepts.
    pub fn validate(&self) -> Result<(), AnalysisError> {
        let n = self.num_gaussians();

        let checks: &[(&str, usize, usize)] = &[
            ("positions", self.positions.len(), n * 3),
            ("log_scales", self.log_scales.len(), n * 3),
            ("rotations", self.rotations.len(), n * 4),
            ("colors", self.colors.len(), n * 3),
        ];

        for &(field, got, expected) in checks {
            if got != expected {
                return Err(AnalysisError::DimensionError {
                    field: field.to_string(),
                    expected,
                    got,
                });
            }
        }

        check_all_finite("positions", &self.positions)?;
        check_all_finite("log_scales", &self.log_scales)?;
        check_all_finite("rotations", &self.rotations)?;
        check_all_finite("opacities", &self.opacities)?;
        check_all_finite("colors", &self.colors)?;

        Ok(())
    }

    /// Sigmoid activation of logit-space opacity for Gaussian `i`.
    ///
    /// Returns `1 / (1 + exp(-logit))`.
    #[must_use]
    pub fn activated_opacity(&self, i: usize) -> f32 {
        1.0 / (1.0 + (-self.opacities[i]).exp())
    }

    /// Linear-space scale `[exp(sx), exp(sy), exp(sz)]` for Gaussian `i`.
    #[must_use]
    pub fn scale(&self, i: usize) -> [f32; 3] {
        let base = i * 3;
        [
            self.log_scales[base].exp(),
            self.log_scales[base + 1].exp(),
            self.log_scales[base + 2].exp(),
        ]
    }

    /// Maximum scale dimension (max of linear scales) for Gaussian `i`.
    #[must_use]
    pub fn max_scale(&self, i: usize) -> f32 {
        let s = self.scale(i);
        s[0].max(s[1]).max(s[2])
    }

    /// Volume (product of linear scales) for Gaussian `i`.
    #[must_use]
    pub fn volume(&self, i: usize) -> f32 {
        let s = self.scale(i);
        s[0] * s[1] * s[2]
    }
}

// ---------------------------------------------------------------------------
// SpatialStats
// ---------------------------------------------------------------------------

/// Spatial statistics for a 3DGS scene.
#[derive(Debug, Clone)]
pub struct SpatialStats {
    /// Total number of Gaussians.
    pub num_gaussians: usize,
    /// Per-axis minimum of all Gaussian positions.
    pub bounding_box_min: [f32; 3],
    /// Per-axis maximum of all Gaussian positions.
    pub bounding_box_max: [f32; 3],
    /// Mean position (centroid) of all Gaussians.
    pub centroid: [f32; 3],
    /// Euclidean distance between the two bounding-box corners.
    pub scene_diameter: f32,
    /// Mean nearest-neighbour distance estimated from a deterministic sample.
    pub mean_nearest_neighbor_dist: f32,
    /// Rough scene volume computed from the bounding box extents.
    pub volume_estimate: f32,
}

/// Compute spatial statistics for a scene.
///
/// Nearest-neighbour distance is *queried* from a deterministic evenly-spaced
/// sample of up to 100 Gaussians (to keep complexity bounded at O(100n)
/// rather than O(n²)), but each query is matched against the *full* position
/// array — otherwise the result would just be the spacing of the 100-point
/// sample lattice itself, not the scene's actual local density.
pub fn compute_spatial_stats(scene: &SceneData) -> Result<SpatialStats, AnalysisError> {
    let n = scene.num_gaussians();
    if n == 0 {
        return Err(AnalysisError::EmptyScene);
    }

    let mut bb_min = [f32::INFINITY; 3];
    let mut bb_max = [f32::NEG_INFINITY; 3];
    let mut centroid = [0.0_f32; 3];

    for i in 0..n {
        let base = i * 3;
        for axis in 0..3 {
            let v = scene.positions[base + axis];
            if v < bb_min[axis] {
                bb_min[axis] = v;
            }
            if v > bb_max[axis] {
                bb_max[axis] = v;
            }
            centroid[axis] += v;
        }
    }
    let n_f = n as f32;
    centroid[0] /= n_f;
    centroid[1] /= n_f;
    centroid[2] /= n_f;

    let dx = bb_max[0] - bb_min[0];
    let dy = bb_max[1] - bb_min[1];
    let dz = bb_max[2] - bb_min[2];
    let scene_diameter = (dx * dx + dy * dy + dz * dz).sqrt();
    let volume_estimate = dx * dy * dz;

    // Nearest-neighbour: deterministic evenly-spaced sample of <= 100 Gaussians.
    let sample_count = n.min(100);
    let sampled_indices: Vec<usize> = (0..sample_count)
        .map(|k| {
            if sample_count == 1 {
                0
            } else {
                k * (n - 1) / (sample_count - 1)
            }
        })
        .collect();

    let mean_nearest_neighbor_dist = if sample_count == 1 {
        0.0
    } else {
        let mut total_nn_dist = 0.0_f32;
        for &i in &sampled_indices {
            let base_i = i * 3;
            let xi = scene.positions[base_i];
            let yi = scene.positions[base_i + 1];
            let zi = scene.positions[base_i + 2];

            let mut min_sq = f32::INFINITY;
            // Match against every Gaussian in the scene, not just the other
            // sampled points — searching only the sample would report the
            // spacing of the `sample_count`-point lattice itself rather than
            // the scene's true nearest-neighbour distance. `sample_count>1`
            // here implies `n>=2`, so at least one `j != i` is always found.
            for j in 0..n {
                if j == i {
                    continue;
                }
                let base_j = j * 3;
                let dx2 = xi - scene.positions[base_j];
                let dy2 = yi - scene.positions[base_j + 1];
                let dz2 = zi - scene.positions[base_j + 2];
                let sq = dx2 * dx2 + dy2 * dy2 + dz2 * dz2;
                if sq < min_sq {
                    min_sq = sq;
                }
            }
            total_nn_dist += min_sq.sqrt();
        }
        total_nn_dist / sample_count as f32
    };

    Ok(SpatialStats {
        num_gaussians: n,
        bounding_box_min: bb_min,
        bounding_box_max: bb_max,
        centroid,
        scene_diameter,
        mean_nearest_neighbor_dist,
        volume_estimate,
    })
}

// ---------------------------------------------------------------------------
// OpacityStats
// ---------------------------------------------------------------------------

/// Opacity distribution statistics for a 3DGS scene.
#[derive(Debug, Clone)]
pub struct OpacityStats {
    /// Mean activated opacity across all Gaussians.
    pub mean_opacity: f32,
    /// Standard deviation of activated opacities.
    pub std_opacity: f32,
    /// Fraction of Gaussians with activated opacity < 0.1.
    pub fraction_transparent: f32,
    /// Fraction of Gaussians with activated opacity > 0.9.
    pub fraction_opaque: f32,
    /// Fraction of Gaussians with 0.1 ≤ opacity ≤ 0.9.
    pub fraction_midrange: f32,
    /// Median activated opacity.
    pub p50_opacity: f32,
    /// 95th-percentile activated opacity.
    pub p95_opacity: f32,
    /// Histogram counts for opacity in 10 equal-width bins [0, 0.1) … [0.9, 1.0].
    pub histogram: [u32; 10],
}

/// Compute opacity distribution statistics for a scene.
pub fn compute_opacity_stats(scene: &SceneData) -> Result<OpacityStats, AnalysisError> {
    let n = scene.num_gaussians();
    if n == 0 {
        return Err(AnalysisError::EmptyScene);
    }

    let opacities: Vec<f32> = (0..n).map(|i| scene.activated_opacity(i)).collect();

    let mean_opacity = opacities.iter().sum::<f32>() / n as f32;
    let variance = opacities
        .iter()
        .map(|&v| (v - mean_opacity) * (v - mean_opacity))
        .sum::<f32>()
        / n as f32;
    let std_opacity = variance.sqrt();

    let mut n_transparent = 0u32;
    let mut n_opaque = 0u32;
    let mut n_midrange = 0u32;
    let mut histogram = [0u32; 10];

    for &op in &opacities {
        if op < 0.1 {
            n_transparent += 1;
        } else if op > 0.9 {
            n_opaque += 1;
        } else {
            n_midrange += 1;
        }
        let bin = ((op * 10.0).floor() as usize).min(9);
        histogram[bin] += 1;
    }

    let n_f = n as f32;
    let fraction_transparent = n_transparent as f32 / n_f;
    let fraction_opaque = n_opaque as f32 / n_f;
    let fraction_midrange = n_midrange as f32 / n_f;

    let mut sorted = opacities.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p50_opacity = sorted[n / 2];
    let p95_opacity = sorted[((95 * n) / 100).min(n - 1)];

    Ok(OpacityStats {
        mean_opacity,
        std_opacity,
        fraction_transparent,
        fraction_opaque,
        fraction_midrange,
        p50_opacity,
        p95_opacity,
        histogram,
    })
}

// ---------------------------------------------------------------------------
// ScaleStats
// ---------------------------------------------------------------------------

/// Scale distribution statistics for a 3DGS scene.
#[derive(Debug, Clone)]
pub struct ScaleStats {
    /// Mean of the maximum scale dimension across all Gaussians.
    pub mean_max_scale: f32,
    /// Standard deviation of maximum scale.
    pub std_max_scale: f32,
    /// Minimum value of maximum scale.
    pub min_max_scale: f32,
    /// Maximum value of maximum scale.
    pub max_max_scale: f32,
    /// Mean volume (product of three linear scales) across all Gaussians.
    pub mean_volume: f32,
    /// Fraction of Gaussians whose max_scale < 1e-4 (effectively degenerate).
    pub fraction_too_small: f32,
    /// Fraction of Gaussians whose max_scale > 1.0 (very large Gaussians).
    pub fraction_too_large: f32,
    /// Mean anisotropy: mean(max_scale / min_scale) per Gaussian.
    pub mean_anisotropy: f32,
    /// Median max_scale.
    pub p50_scale: f32,
    /// 95th-percentile max_scale.
    pub p95_scale: f32,
}

/// Compute scale distribution statistics for a scene.
pub fn compute_scale_stats(scene: &SceneData) -> Result<ScaleStats, AnalysisError> {
    let n = scene.num_gaussians();
    if n == 0 {
        return Err(AnalysisError::EmptyScene);
    }

    let mut max_scales = Vec::with_capacity(n);
    let mut volumes = Vec::with_capacity(n);
    let mut anisotropies = Vec::with_capacity(n);
    let mut n_too_small = 0u32;
    let mut n_too_large = 0u32;

    for i in 0..n {
        let s = scene.scale(i);
        let max_s = s[0].max(s[1]).max(s[2]);
        let min_s = s[0].min(s[1]).min(s[2]);
        let vol = s[0] * s[1] * s[2];
        let anisotropy = max_s / (min_s + 1e-8);

        max_scales.push(max_s);
        volumes.push(vol);
        anisotropies.push(anisotropy);

        if max_s < 1e-4 {
            n_too_small += 1;
        }
        if max_s > 1.0 {
            n_too_large += 1;
        }
    }

    let n_f = n as f32;
    let mean_max_scale = max_scales.iter().sum::<f32>() / n_f;
    let variance = max_scales
        .iter()
        .map(|&v| (v - mean_max_scale) * (v - mean_max_scale))
        .sum::<f32>()
        / n_f;
    let std_max_scale = variance.sqrt();

    let min_max_scale = max_scales.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_max_scale = max_scales.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mean_volume = volumes.iter().sum::<f32>() / n_f;
    let mean_anisotropy = anisotropies.iter().sum::<f32>() / n_f;
    let fraction_too_small = n_too_small as f32 / n_f;
    let fraction_too_large = n_too_large as f32 / n_f;

    let mut sorted_scales = max_scales.clone();
    sorted_scales.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p50_scale = sorted_scales[n / 2];
    let p95_scale = sorted_scales[((95 * n) / 100).min(n - 1)];

    Ok(ScaleStats {
        mean_max_scale,
        std_max_scale,
        min_max_scale,
        max_max_scale,
        mean_volume,
        fraction_too_small,
        fraction_too_large,
        mean_anisotropy,
        p50_scale,
        p95_scale,
    })
}

// ---------------------------------------------------------------------------
// ColorStats
// ---------------------------------------------------------------------------

/// Color distribution statistics for a 3DGS scene.
#[derive(Debug, Clone)]
pub struct ColorStats {
    /// Mean red channel across all Gaussians (after clamping to [0, 1]).
    pub mean_r: f32,
    /// Mean green channel across all Gaussians.
    pub mean_g: f32,
    /// Mean blue channel across all Gaussians.
    pub mean_b: f32,
    /// Standard deviation of red channel.
    pub std_r: f32,
    /// Standard deviation of green channel.
    pub std_g: f32,
    /// Standard deviation of blue channel.
    pub std_b: f32,
    /// Mean BT.709 luminance: 0.2126·R + 0.7152·G + 0.0722·B.
    pub mean_luminance: f32,
    /// Standard deviation of luminance (used as a colorfulness proxy).
    pub color_diversity: f32,
    /// Fraction of Gaussians with luminance < 0.1.
    pub fraction_dark: f32,
    /// Fraction of Gaussians with luminance > 0.9.
    pub fraction_bright: f32,
}

/// Compute color distribution statistics for a scene.
///
/// Colors are clamped to `[0, 1]` before all statistics are computed.
pub fn compute_color_stats(scene: &SceneData) -> Result<ColorStats, AnalysisError> {
    let n = scene.num_gaussians();
    if n == 0 {
        return Err(AnalysisError::EmptyScene);
    }

    let mut rs = Vec::with_capacity(n);
    let mut gs = Vec::with_capacity(n);
    let mut bs = Vec::with_capacity(n);
    let mut luminances = Vec::with_capacity(n);
    let mut n_dark = 0u32;
    let mut n_bright = 0u32;

    for i in 0..n {
        let base = i * 3;
        let r = scene.colors[base].clamp(0.0, 1.0);
        let g = scene.colors[base + 1].clamp(0.0, 1.0);
        let b = scene.colors[base + 2].clamp(0.0, 1.0);
        let lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;

        rs.push(r);
        gs.push(g);
        bs.push(b);
        luminances.push(lum);

        if lum < 0.1 {
            n_dark += 1;
        }
        if lum > 0.9 {
            n_bright += 1;
        }
    }

    let n_f = n as f32;

    let mean_r = rs.iter().sum::<f32>() / n_f;
    let mean_g = gs.iter().sum::<f32>() / n_f;
    let mean_b = bs.iter().sum::<f32>() / n_f;
    let mean_luminance = luminances.iter().sum::<f32>() / n_f;

    let std_r = (rs.iter().map(|&v| (v - mean_r) * (v - mean_r)).sum::<f32>() / n_f).sqrt();
    let std_g = (gs.iter().map(|&v| (v - mean_g) * (v - mean_g)).sum::<f32>() / n_f).sqrt();
    let std_b = (bs.iter().map(|&v| (v - mean_b) * (v - mean_b)).sum::<f32>() / n_f).sqrt();
    let color_diversity = (luminances
        .iter()
        .map(|&v| (v - mean_luminance) * (v - mean_luminance))
        .sum::<f32>()
        / n_f)
        .sqrt();

    let fraction_dark = n_dark as f32 / n_f;
    let fraction_bright = n_bright as f32 / n_f;

    Ok(ColorStats {
        mean_r,
        mean_g,
        mean_b,
        std_r,
        std_g,
        std_b,
        mean_luminance,
        color_diversity,
        fraction_dark,
        fraction_bright,
    })
}

// ---------------------------------------------------------------------------
// Quality score
// ---------------------------------------------------------------------------

/// Compute a composite quality score in [0, 1] from opacity and scale stats.
///
/// Higher is better. Score components:
/// - Opacity: `1.0 - max(fraction_transparent, fraction_opaque * 0.5)`
/// - Scale:   `1.0 - fraction_too_small - fraction_too_large`
/// - Anisotropy: `1.0 / (1.0 + max(mean_anisotropy - 1.0, 0.0) * 0.1)`
///
/// Final score is the mean of the three components, clamped to [0, 1].
#[must_use]
pub fn compute_quality_score(opacity: &OpacityStats, scale: &ScaleStats) -> f32 {
    let good_opacity = 1.0
        - opacity
            .fraction_transparent
            .max(opacity.fraction_opaque * 0.5);
    let good_scale = (1.0 - scale.fraction_too_small - scale.fraction_too_large).max(0.0);
    let good_anisotropy = 1.0 / (1.0 + (scale.mean_anisotropy - 1.0).max(0.0) * 0.1);

    let raw = (good_opacity + good_scale + good_anisotropy) / 3.0;
    raw.clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------------
// SceneReport
// ---------------------------------------------------------------------------

/// Full analysis report for a 3DGS scene.
#[derive(Debug, Clone)]
pub struct SceneReport {
    /// Spatial statistics.
    pub spatial: SpatialStats,
    /// Opacity distribution.
    pub opacity: OpacityStats,
    /// Scale distribution.
    pub scale: ScaleStats,
    /// Color distribution.
    pub color: ColorStats,
    /// Composite quality score in [0, 1].
    pub quality_score: f32,
}

impl SceneReport {
    /// Short one-line summary suitable for human reading.
    #[must_use]
    pub fn format_summary(&self) -> String {
        format!(
            "Gaussians: {} | BBox: [{:.3},{:.3},{:.3}] .. [{:.3},{:.3},{:.3}] | \
             Mean opacity: {:.3} | Mean scale: {:.5} | Quality: {:.3}",
            self.spatial.num_gaussians,
            self.spatial.bounding_box_min[0],
            self.spatial.bounding_box_min[1],
            self.spatial.bounding_box_min[2],
            self.spatial.bounding_box_max[0],
            self.spatial.bounding_box_max[1],
            self.spatial.bounding_box_max[2],
            self.opacity.mean_opacity,
            self.scale.mean_max_scale,
            self.quality_score,
        )
    }

    /// Detailed multi-line report with all statistics.
    #[must_use]
    pub fn format_detailed(&self) -> String {
        let s = &self.spatial;
        let o = &self.opacity;
        let sc = &self.scale;
        let c = &self.color;

        let mut out = String::new();
        out.push_str("=== Scene Analysis Report ===\n\n");

        out.push_str("--- Spatial ---\n");
        out.push_str(&format!("  Gaussians:         {}\n", s.num_gaussians));
        out.push_str(&format!(
            "  Bounding box min:  [{:.4}, {:.4}, {:.4}]\n",
            s.bounding_box_min[0], s.bounding_box_min[1], s.bounding_box_min[2]
        ));
        out.push_str(&format!(
            "  Bounding box max:  [{:.4}, {:.4}, {:.4}]\n",
            s.bounding_box_max[0], s.bounding_box_max[1], s.bounding_box_max[2]
        ));
        out.push_str(&format!(
            "  Centroid:          [{:.4}, {:.4}, {:.4}]\n",
            s.centroid[0], s.centroid[1], s.centroid[2]
        ));
        out.push_str(&format!("  Scene diameter:    {:.4}\n", s.scene_diameter));
        out.push_str(&format!(
            "  Mean NN dist:      {:.4}\n",
            s.mean_nearest_neighbor_dist
        ));
        out.push_str(&format!("  Volume estimate:   {:.4}\n", s.volume_estimate));

        out.push_str("\n--- Opacity ---\n");
        out.push_str(&format!("  Mean:              {:.4}\n", o.mean_opacity));
        out.push_str(&format!("  Std:               {:.4}\n", o.std_opacity));
        out.push_str(&format!("  p50:               {:.4}\n", o.p50_opacity));
        out.push_str(&format!("  p95:               {:.4}\n", o.p95_opacity));
        out.push_str(&format!(
            "  Fraction transparent (<0.1): {:.4}\n",
            o.fraction_transparent
        ));
        out.push_str(&format!(
            "  Fraction opaque    (>0.9):   {:.4}\n",
            o.fraction_opaque
        ));
        out.push_str(&format!(
            "  Fraction midrange  [0.1,0.9]:{:.4}\n",
            o.fraction_midrange
        ));
        out.push_str(&format!("  Histogram (10 bins): {:?}\n", o.histogram));

        out.push_str("\n--- Scale ---\n");
        out.push_str(&format!("  Mean max scale:    {:.6}\n", sc.mean_max_scale));
        out.push_str(&format!("  Std max scale:     {:.6}\n", sc.std_max_scale));
        out.push_str(&format!("  Min max scale:     {:.6}\n", sc.min_max_scale));
        out.push_str(&format!("  Max max scale:     {:.6}\n", sc.max_max_scale));
        out.push_str(&format!("  Mean volume:       {:.8}\n", sc.mean_volume));
        out.push_str(&format!("  p50 scale:         {:.6}\n", sc.p50_scale));
        out.push_str(&format!("  p95 scale:         {:.6}\n", sc.p95_scale));
        out.push_str(&format!(
            "  Fraction too small (<1e-4): {:.4}\n",
            sc.fraction_too_small
        ));
        out.push_str(&format!(
            "  Fraction too large (>1.0):  {:.4}\n",
            sc.fraction_too_large
        ));
        out.push_str(&format!("  Mean anisotropy:   {:.4}\n", sc.mean_anisotropy));

        out.push_str("\n--- Color ---\n");
        out.push_str(&format!(
            "  Mean RGB:          [{:.4}, {:.4}, {:.4}]\n",
            c.mean_r, c.mean_g, c.mean_b
        ));
        out.push_str(&format!(
            "  Std RGB:           [{:.4}, {:.4}, {:.4}]\n",
            c.std_r, c.std_g, c.std_b
        ));
        out.push_str(&format!("  Mean luminance:    {:.4}\n", c.mean_luminance));
        out.push_str(&format!("  Color diversity:   {:.4}\n", c.color_diversity));
        out.push_str(&format!(
            "  Fraction dark (<0.1):   {:.4}\n",
            c.fraction_dark
        ));
        out.push_str(&format!(
            "  Fraction bright (>0.9): {:.4}\n",
            c.fraction_bright
        ));

        out.push_str(&format!(
            "\n=== Quality Score: {:.4} ===\n",
            self.quality_score
        ));
        out
    }

    /// Hand-rolled JSON representation of the report (no serde dependency).
    ///
    /// Every `f32` field is routed through `json_num`, which maps NaN/Inf
    /// to `null` — `SceneData::validate` already rejects non-finite inputs,
    /// but this keeps `to_json`'s output valid JSON even if some derived
    /// statistic (e.g. a 0/0 ratio) turns non-finite despite finite inputs.
    #[must_use]
    pub fn to_json(&self) -> String {
        let s = &self.spatial;
        let o = &self.opacity;
        let sc = &self.scale;
        let c = &self.color;

        let hist: Vec<String> = o.histogram.iter().map(|v| v.to_string()).collect();
        let hist_json = format!("[{}]", hist.join(", "));

        format!(
            "{{\
             \"num_gaussians\": {num_gaussians}, \
             \"bounding_box_min\": [{bbx0}, {bbx1}, {bbx2}], \
             \"bounding_box_max\": [{bbx3}, {bbx4}, {bbx5}], \
             \"centroid\": [{cx}, {cy}, {cz}], \
             \"scene_diameter\": {diam}, \
             \"mean_nearest_neighbor_dist\": {nn}, \
             \"volume_estimate\": {vol_est}, \
             \"mean_opacity\": {mo}, \
             \"std_opacity\": {so}, \
             \"fraction_transparent\": {ft}, \
             \"fraction_opaque\": {fo}, \
             \"fraction_midrange\": {fm}, \
             \"p50_opacity\": {p50o}, \
             \"p95_opacity\": {p95o}, \
             \"opacity_histogram\": {hist}, \
             \"mean_max_scale\": {mms}, \
             \"std_max_scale\": {sms}, \
             \"min_max_scale\": {min_ms}, \
             \"max_max_scale\": {max_ms}, \
             \"mean_volume\": {mvol}, \
             \"fraction_too_small\": {fts}, \
             \"fraction_too_large\": {ftl}, \
             \"mean_anisotropy\": {ma}, \
             \"p50_scale\": {p50s}, \
             \"p95_scale\": {p95s}, \
             \"mean_r\": {mr}, \
             \"mean_g\": {mg}, \
             \"mean_b\": {mb}, \
             \"std_r\": {sr}, \
             \"std_g\": {sg}, \
             \"std_b\": {sb}, \
             \"mean_luminance\": {ml}, \
             \"color_diversity\": {cd}, \
             \"fraction_dark\": {fd}, \
             \"fraction_bright\": {fb}, \
             \"quality_score\": {qs}\
             }}",
            num_gaussians = s.num_gaussians,
            bbx0 = json_num(s.bounding_box_min[0]),
            bbx1 = json_num(s.bounding_box_min[1]),
            bbx2 = json_num(s.bounding_box_min[2]),
            bbx3 = json_num(s.bounding_box_max[0]),
            bbx4 = json_num(s.bounding_box_max[1]),
            bbx5 = json_num(s.bounding_box_max[2]),
            cx = json_num(s.centroid[0]),
            cy = json_num(s.centroid[1]),
            cz = json_num(s.centroid[2]),
            diam = json_num(s.scene_diameter),
            nn = json_num(s.mean_nearest_neighbor_dist),
            vol_est = json_num(s.volume_estimate),
            mo = json_num(o.mean_opacity),
            so = json_num(o.std_opacity),
            ft = json_num(o.fraction_transparent),
            fo = json_num(o.fraction_opaque),
            fm = json_num(o.fraction_midrange),
            p50o = json_num(o.p50_opacity),
            p95o = json_num(o.p95_opacity),
            hist = hist_json,
            mms = json_num(sc.mean_max_scale),
            sms = json_num(sc.std_max_scale),
            min_ms = json_num(sc.min_max_scale),
            max_ms = json_num(sc.max_max_scale),
            mvol = json_num(sc.mean_volume),
            fts = json_num(sc.fraction_too_small),
            ftl = json_num(sc.fraction_too_large),
            ma = json_num(sc.mean_anisotropy),
            p50s = json_num(sc.p50_scale),
            p95s = json_num(sc.p95_scale),
            mr = json_num(c.mean_r),
            mg = json_num(c.mean_g),
            mb = json_num(c.mean_b),
            sr = json_num(c.std_r),
            sg = json_num(c.std_g),
            sb = json_num(c.std_b),
            ml = json_num(c.mean_luminance),
            cd = json_num(c.color_diversity),
            fd = json_num(c.fraction_dark),
            fb = json_num(c.fraction_bright),
            qs = json_num(self.quality_score),
        )
    }
}

/// Format an `f32` for JSON output. Non-finite values (NaN, +Inf, -Inf) have
/// no JSON token, so they are mapped to `null` rather than emitting the bare
/// `NaN`/`inf` text that `f32`'s `Display` impl would otherwise produce.
fn json_num(v: f32) -> String {
    if v.is_finite() {
        v.to_string()
    } else {
        "null".to_string()
    }
}

// ---------------------------------------------------------------------------
// analyze_scene
// ---------------------------------------------------------------------------

/// Run all sub-analyses on a scene and produce a full `SceneReport`.
pub fn analyze_scene(scene: &SceneData) -> Result<SceneReport, AnalysisError> {
    scene.validate()?;

    let spatial = compute_spatial_stats(scene)?;
    let opacity = compute_opacity_stats(scene)?;
    let scale = compute_scale_stats(scene)?;
    let color = compute_color_stats(scene)?;
    let quality_score = compute_quality_score(&opacity, &scale);

    Ok(SceneReport {
        spatial,
        opacity,
        scale,
        color,
        quality_score,
    })
}

// ---------------------------------------------------------------------------
// SceneComparison
// ---------------------------------------------------------------------------

/// Comparison between two 3DGS scenes (A and B).
#[derive(Debug, Clone)]
pub struct SceneComparison {
    /// Number of Gaussians in scene A.
    pub scene_a_gaussians: usize,
    /// Number of Gaussians in scene B.
    pub scene_b_gaussians: usize,
    /// Size ratio B / A (Gaussian count).
    pub size_ratio: f32,
    /// Difference in mean opacity: B.mean_opacity − A.mean_opacity.
    pub mean_opacity_diff: f32,
    /// Difference in mean max scale: B.mean_max_scale − A.mean_max_scale.
    pub mean_scale_diff: f32,
    /// Difference in quality score: B.quality_score − A.quality_score.
    pub quality_score_diff: f32,
    /// L2 distance between the centroids of A and B.
    pub centroid_distance: f32,
}

/// Compare two scenes by running a full analysis on each.
pub fn compare_scenes(
    scene_a: &SceneData,
    scene_b: &SceneData,
) -> Result<SceneComparison, AnalysisError> {
    let report_a = analyze_scene(scene_a)?;
    let report_b = analyze_scene(scene_b)?;

    let ca = report_a.spatial.centroid;
    let cb = report_b.spatial.centroid;
    let dx = cb[0] - ca[0];
    let dy = cb[1] - ca[1];
    let dz = cb[2] - ca[2];
    let centroid_distance = (dx * dx + dy * dy + dz * dz).sqrt();

    let n_a = report_a.spatial.num_gaussians;
    let n_b = report_b.spatial.num_gaussians;
    let size_ratio = n_b as f32 / n_a as f32;

    Ok(SceneComparison {
        scene_a_gaussians: n_a,
        scene_b_gaussians: n_b,
        size_ratio,
        mean_opacity_diff: report_b.opacity.mean_opacity - report_a.opacity.mean_opacity,
        mean_scale_diff: report_b.scale.mean_max_scale - report_a.scale.mean_max_scale,
        quality_score_diff: report_b.quality_score - report_a.quality_score,
        centroid_distance,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Build a minimal valid SceneData with `n` identical Gaussians.
    fn make_scene(n: usize, log_scale: f32, logit_opacity: f32, color: [f32; 3]) -> SceneData {
        SceneData {
            positions: vec![0.0; n * 3],
            log_scales: vec![log_scale; n * 3],
            rotations: {
                let mut v = vec![0.0f32; n * 4];
                for i in 0..n {
                    v[i * 4 + 3] = 1.0; // qw = 1
                }
                v
            },
            opacities: vec![logit_opacity; n],
            colors: {
                let mut c = Vec::with_capacity(n * 3);
                for _ in 0..n {
                    c.push(color[0]);
                    c.push(color[1]);
                    c.push(color[2]);
                }
                c
            },
        }
    }

    /// Build a SceneData with positions spread along the x-axis.
    fn make_line_scene(n: usize) -> SceneData {
        let mut positions = Vec::with_capacity(n * 3);
        for i in 0..n {
            positions.push(i as f32); // x
            positions.push(0.0); // y
            positions.push(0.0); // z
        }
        SceneData {
            positions,
            log_scales: vec![0.0; n * 3], // exp(0) = 1.0
            rotations: {
                let mut v = vec![0.0f32; n * 4];
                for i in 0..n {
                    v[i * 4 + 3] = 1.0;
                }
                v
            },
            opacities: vec![0.0; n], // sigmoid(0) = 0.5
            colors: vec![0.5; n * 3],
        }
    }

    // -----------------------------------------------------------------------
    // SceneData::validate
    // -----------------------------------------------------------------------

    #[test]
    fn test_validate_correct_data() {
        let scene = make_scene(4, -2.0, 0.0, [0.5, 0.5, 0.5]);
        assert!(scene.validate().is_ok());
    }

    #[test]
    fn test_validate_wrong_positions_length() {
        let mut scene = make_scene(3, -2.0, 0.0, [0.5, 0.5, 0.5]);
        scene.positions.pop(); // now length 8 instead of 9
        let err = scene.validate().unwrap_err();
        assert!(matches!(
            err,
            AnalysisError::DimensionError { ref field, expected: 9, got: 8 }
            if field == "positions"
        ));
    }

    #[test]
    fn test_validate_wrong_rotations_length() {
        let mut scene = make_scene(2, -2.0, 0.0, [0.5, 0.5, 0.5]);
        scene.rotations.push(0.0); // 9 instead of 8
        let err = scene.validate().unwrap_err();
        assert!(matches!(
            err,
            AnalysisError::DimensionError { ref field, expected: 8, got: 9 }
            if field == "rotations"
        ));
    }

    #[test]
    fn test_validate_rejects_nan_position() {
        // Regression test: NaN in any numeric field must be rejected by
        // `validate()` rather than silently propagating into every
        // downstream statistic.
        let mut scene = make_scene(2, -2.0, 0.0, [0.5, 0.5, 0.5]);
        scene.positions[3] = f32::NAN;
        let err = scene.validate().unwrap_err();
        assert!(matches!(err, AnalysisError::InvalidData(ref msg) if msg.contains("positions")));
    }

    #[test]
    fn test_validate_rejects_infinite_opacity() {
        let mut scene = make_scene(2, -2.0, 0.0, [0.5, 0.5, 0.5]);
        scene.opacities[1] = f32::INFINITY;
        let err = scene.validate().unwrap_err();
        assert!(matches!(err, AnalysisError::InvalidData(ref msg) if msg.contains("opacities")));
    }

    #[test]
    fn test_analyze_scene_rejects_nan_scene() {
        let mut scene = make_scene(3, -2.0, 0.0, [0.5, 0.5, 0.5]);
        scene.colors[0] = f32::NAN;
        let err = analyze_scene(&scene).unwrap_err();
        assert!(matches!(err, AnalysisError::InvalidData(_)));
    }

    // -----------------------------------------------------------------------
    // SceneData accessors
    // -----------------------------------------------------------------------

    #[test]
    fn test_activated_opacity_at_zero_is_half() {
        let scene = make_scene(1, 0.0, 0.0, [0.0, 0.0, 0.0]);
        let op = scene.activated_opacity(0);
        assert!(
            (op - 0.5).abs() < 1e-6,
            "sigmoid(0) should be 0.5, got {op}"
        );
    }

    #[test]
    fn test_scale_is_exp_of_log_scale() {
        let log_s = 2.0_f32;
        let scene = make_scene(1, log_s, 0.0, [0.0, 0.0, 0.0]);
        let s = scene.scale(0);
        let expected = log_s.exp();
        assert!((s[0] - expected).abs() < 1e-5);
        assert!((s[1] - expected).abs() < 1e-5);
        assert!((s[2] - expected).abs() < 1e-5);
    }

    #[test]
    fn test_max_scale_picks_largest() {
        let mut scene = SceneData {
            positions: vec![0.0, 0.0, 0.0],
            log_scales: vec![0.0_f32.ln(), 1.0_f32.ln(), 2.0_f32.ln()], // scales: 1, 1, 2
            rotations: vec![0.0, 0.0, 0.0, 1.0],
            opacities: vec![0.0],
            colors: vec![0.5, 0.5, 0.5],
        };
        // Override log_scales manually for a clear anisotropic case.
        scene.log_scales = vec![0.0, 0.0, 2.0_f32.ln()]; // sx=1, sy=1, sz=2
        let ms = scene.max_scale(0);
        assert!((ms - 2.0).abs() < 1e-5, "max_scale should be 2.0, got {ms}");
    }

    #[test]
    fn test_volume_is_product_of_scales() {
        let log_s = 1.0_f32; // exp(1) ≈ 2.718
        let scene = make_scene(1, log_s, 0.0, [0.0, 0.0, 0.0]);
        let expected = log_s.exp().powi(3);
        let v = scene.volume(0);
        assert!(
            (v - expected).abs() < 1e-4,
            "volume mismatch: {v} vs {expected}"
        );
    }

    // -----------------------------------------------------------------------
    // compute_spatial_stats
    // -----------------------------------------------------------------------

    #[test]
    fn test_spatial_stats_empty_is_error() {
        let scene = make_scene(0, 0.0, 0.0, [0.5, 0.5, 0.5]);
        assert_eq!(
            compute_spatial_stats(&scene).unwrap_err(),
            AnalysisError::EmptyScene
        );
    }

    #[test]
    fn test_spatial_stats_single_gaussian() {
        let mut scene = make_scene(1, 0.0, 0.0, [0.5, 0.5, 0.5]);
        scene.positions = vec![1.0, 2.0, 3.0];
        let stats = compute_spatial_stats(&scene).expect("should succeed");
        assert_eq!(stats.num_gaussians, 1);
        assert!((stats.centroid[0] - 1.0).abs() < 1e-6);
        assert!((stats.centroid[1] - 2.0).abs() < 1e-6);
        assert!((stats.centroid[2] - 3.0).abs() < 1e-6);
        assert_eq!(stats.mean_nearest_neighbor_dist, 0.0);
    }

    #[test]
    fn test_spatial_stats_two_gaussians_bounding_box() {
        let scene = SceneData {
            positions: vec![-1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            log_scales: vec![0.0; 6],
            rotations: vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            opacities: vec![0.0, 0.0],
            colors: vec![0.5; 6],
        };
        let stats = compute_spatial_stats(&scene).expect("should succeed");
        assert!((stats.bounding_box_min[0] - (-1.0)).abs() < 1e-6);
        assert!((stats.bounding_box_max[0] - 1.0).abs() < 1e-6);
        assert!((stats.scene_diameter - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_spatial_stats_centroid_is_mean() {
        let scene = make_line_scene(5); // positions x = 0,1,2,3,4
        let stats = compute_spatial_stats(&scene).expect("should succeed");
        assert!(
            (stats.centroid[0] - 2.0).abs() < 1e-5,
            "centroid x should be 2.0"
        );
        assert!(stats.centroid[1].abs() < 1e-6);
        assert!(stats.centroid[2].abs() < 1e-6);
    }

    #[test]
    fn test_spatial_stats_volume_positive() {
        let scene = make_line_scene(3); // x in [0,2], y=z=0 → volume = 2*0*0 = 0 (degenerate)
                                        // Build a 3D spread scene instead.
        let scene3d = SceneData {
            positions: vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0],
            log_scales: vec![0.0; 6],
            rotations: vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            opacities: vec![0.0, 0.0],
            colors: vec![0.5; 6],
        };
        let _ = scene; // suppress unused warning
        let stats = compute_spatial_stats(&scene3d).expect("should succeed");
        assert!(
            stats.volume_estimate > 0.0,
            "volume should be positive for 3D spread"
        );
        assert!((stats.volume_estimate - 6.0).abs() < 1e-5);
    }

    // -----------------------------------------------------------------------
    // compute_opacity_stats
    // -----------------------------------------------------------------------

    #[test]
    fn test_opacity_stats_empty_is_error() {
        let scene = make_scene(0, 0.0, 0.0, [0.5, 0.5, 0.5]);
        assert_eq!(
            compute_opacity_stats(&scene).unwrap_err(),
            AnalysisError::EmptyScene
        );
    }

    #[test]
    fn test_opacity_stats_all_transparent() {
        // sigmoid(-10) ≈ 4.5e-5, well below 0.1
        let scene = make_scene(5, 0.0, -10.0, [0.5, 0.5, 0.5]);
        let stats = compute_opacity_stats(&scene).expect("should succeed");
        assert!((stats.fraction_transparent - 1.0).abs() < 1e-6);
        assert!((stats.fraction_opaque).abs() < 1e-6);
        assert!((stats.fraction_midrange).abs() < 1e-6);
    }

    #[test]
    fn test_opacity_stats_all_opaque() {
        // sigmoid(10) ≈ 0.9999546, above 0.9
        let scene = make_scene(4, 0.0, 10.0, [0.5, 0.5, 0.5]);
        let stats = compute_opacity_stats(&scene).expect("should succeed");
        assert!((stats.fraction_opaque - 1.0).abs() < 1e-6);
        assert!((stats.fraction_transparent).abs() < 1e-6);
    }

    #[test]
    fn test_opacity_histogram_sums_to_n() {
        let scene = make_scene(20, 0.0, 0.0, [0.5, 0.5, 0.5]); // sigmoid(0)=0.5
        let stats = compute_opacity_stats(&scene).expect("should succeed");
        let total: u32 = stats.histogram.iter().sum();
        assert_eq!(total, 20, "histogram should sum to number of Gaussians");
    }

    #[test]
    fn test_opacity_histogram_bin_for_max_opacity() {
        // Ensure opacity = 1.0 (or very close) does not panic due to bin index 10.
        let scene = make_scene(3, 0.0, 100.0, [0.5, 0.5, 0.5]); // sigmoid(100) ≈ 1.0
        let stats = compute_opacity_stats(&scene).expect("should not panic");
        let total: u32 = stats.histogram.iter().sum();
        assert_eq!(total, 3);
    }

    // -----------------------------------------------------------------------
    // compute_scale_stats
    // -----------------------------------------------------------------------

    #[test]
    fn test_scale_stats_empty_is_error() {
        let scene = make_scene(0, 0.0, 0.0, [0.5, 0.5, 0.5]);
        assert_eq!(
            compute_scale_stats(&scene).unwrap_err(),
            AnalysisError::EmptyScene
        );
    }

    #[test]
    fn test_scale_stats_uniform_scales() {
        let log_s = -2.0_f32; // scale ≈ 0.1353
        let scene = make_scene(5, log_s, 0.0, [0.5, 0.5, 0.5]);
        let stats = compute_scale_stats(&scene).expect("should succeed");
        let expected_scale = log_s.exp();
        assert!(
            (stats.mean_max_scale - expected_scale).abs() < 1e-5,
            "mean_max_scale = {}, expected {}",
            stats.mean_max_scale,
            expected_scale
        );
        assert!(stats.std_max_scale.abs() < 1e-6);
        // anisotropy: all dims equal → max/min ≈ 1.0 (with 1e-8 denom adds tiny bias)
        // Allow slight deviation from 1.0 due to floating-point precision in the denom.
        assert!(
            (stats.mean_anisotropy - 1.0).abs() < 1e-3,
            "uniform anisotropy should be ~1.0, got {}",
            stats.mean_anisotropy
        );
    }

    #[test]
    fn test_scale_stats_anisotropy() {
        let scene = SceneData {
            positions: vec![0.0, 0.0, 0.0],
            log_scales: vec![0.0, 0.0, 10.0_f32.ln()], // sx=sy=1, sz=10
            rotations: vec![0.0, 0.0, 0.0, 1.0],
            opacities: vec![0.0],
            colors: vec![0.5, 0.5, 0.5],
        };
        let stats = compute_scale_stats(&scene).expect("should succeed");
        // anisotropy = 10 / (1 + 1e-8) ≈ 10
        assert!(
            (stats.mean_anisotropy - 10.0).abs() < 0.01,
            "anisotropy should be ~10, got {}",
            stats.mean_anisotropy
        );
    }

    #[test]
    fn test_scale_stats_fraction_too_small() {
        // log(1e-5) = -11.51... → exp(-11.51) ≈ 1e-5, well below 1e-4
        let scene = make_scene(3, -11.513, 0.0, [0.5, 0.5, 0.5]);
        let stats = compute_scale_stats(&scene).expect("should succeed");
        assert!(
            (stats.fraction_too_small - 1.0).abs() < 1e-5,
            "all Gaussians should be too small, got {}",
            stats.fraction_too_small
        );
    }

    #[test]
    fn test_scale_stats_fraction_too_large() {
        // log(2) ≈ 0.693, scale = 2 > 1.0
        let scene = make_scene(4, 2.0_f32.ln(), 0.0, [0.5, 0.5, 0.5]);
        let stats = compute_scale_stats(&scene).expect("should succeed");
        assert!(
            (stats.fraction_too_large - 1.0).abs() < 1e-5,
            "all Gaussians should be too large, got {}",
            stats.fraction_too_large
        );
    }

    // -----------------------------------------------------------------------
    // compute_color_stats
    // -----------------------------------------------------------------------

    #[test]
    fn test_color_stats_empty_is_error() {
        let scene = make_scene(0, 0.0, 0.0, [0.5, 0.5, 0.5]);
        assert_eq!(
            compute_color_stats(&scene).unwrap_err(),
            AnalysisError::EmptyScene
        );
    }

    #[test]
    fn test_color_stats_uniform_color() {
        let color = [0.3_f32, 0.6, 0.1];
        let scene = make_scene(10, 0.0, 0.0, color);
        let stats = compute_color_stats(&scene).expect("should succeed");
        assert!((stats.mean_r - 0.3).abs() < 1e-5);
        assert!((stats.mean_g - 0.6).abs() < 1e-5);
        assert!((stats.mean_b - 0.1).abs() < 1e-5);
        // Zero variance for uniform colors
        assert!(stats.std_r.abs() < 1e-6);
        assert!(stats.std_g.abs() < 1e-6);
        assert!(stats.std_b.abs() < 1e-6);
        assert!(stats.color_diversity.abs() < 1e-6);
    }

    #[test]
    fn test_color_stats_mean_luminance_bt709() {
        // Pure red: luminance = 0.2126
        let scene = make_scene(4, 0.0, 0.0, [1.0, 0.0, 0.0]);
        let stats = compute_color_stats(&scene).expect("should succeed");
        let expected_lum = 0.2126_f32;
        assert!(
            (stats.mean_luminance - expected_lum).abs() < 1e-5,
            "BT.709 luminance for pure red should be {}, got {}",
            expected_lum,
            stats.mean_luminance
        );
    }

    #[test]
    fn test_color_stats_clamps_out_of_range() {
        // Colors outside [0,1] should be clamped.
        let mut scene = make_scene(2, 0.0, 0.0, [1.5, -0.5, 2.0]);
        // All channels clamped to [1,0,1] → lum = 0.2126+0.0722 = 0.2848
        let _ = scene.validate(); // validate may fail for this test (colors only checked in context)
                                  // Ensure no panic from clamping.
        scene.positions = vec![0.0; 6];
        scene.log_scales = vec![0.0; 6];
        scene.rotations = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let stats = compute_color_stats(&scene).expect("should succeed with clamping");
        assert!(stats.mean_r <= 1.0 && stats.mean_r >= 0.0);
        assert!(stats.mean_g <= 1.0 && stats.mean_g >= 0.0);
        assert!(stats.mean_b <= 1.0 && stats.mean_b >= 0.0);
    }

    // -----------------------------------------------------------------------
    // compute_quality_score
    // -----------------------------------------------------------------------

    #[test]
    fn test_quality_score_perfect_scene() {
        // Mid-opacity, normal scale, low anisotropy → high score
        let scene = make_scene(10, -2.0, 0.0, [0.5, 0.5, 0.5]);
        let opacity = compute_opacity_stats(&scene).expect("ok");
        let scale = compute_scale_stats(&scene).expect("ok");
        let score = compute_quality_score(&opacity, &scale);
        assert!(
            score > 0.7,
            "perfect scene score should be high, got {score}"
        );
    }

    #[test]
    fn test_quality_score_degenerate_scene() {
        // All transparent AND all too_small → low score
        // logit = -100 → opacity ≈ 0 (all transparent)
        // log_scale = -20 → scale ≈ 2e-9 (all too small)
        let scene = make_scene(10, -20.0, -100.0, [0.5, 0.5, 0.5]);
        let opacity = compute_opacity_stats(&scene).expect("ok");
        let scale = compute_scale_stats(&scene).expect("ok");
        let score = compute_quality_score(&opacity, &scale);
        assert!(
            score < 0.5,
            "degenerate scene score should be low, got {score}"
        );
    }

    // -----------------------------------------------------------------------
    // analyze_scene
    // -----------------------------------------------------------------------

    #[test]
    fn test_analyze_scene_smoke_test() {
        let scene = make_line_scene(10);
        let report = analyze_scene(&scene).expect("analyze_scene should succeed");
        assert_eq!(report.spatial.num_gaussians, 10);
        assert!(report.quality_score >= 0.0 && report.quality_score <= 1.0);
        // Summary and detailed should be non-empty
        assert!(!report.format_summary().is_empty());
        assert!(!report.format_detailed().is_empty());
        // JSON should contain braces
        let json = report.to_json();
        assert!(json.starts_with('{') && json.ends_with('}'));
    }

    // -----------------------------------------------------------------------
    // compare_scenes
    // -----------------------------------------------------------------------

    #[test]
    fn test_compare_scenes_same_scene_ratio_one() {
        let scene = make_line_scene(6);
        let cmp = compare_scenes(&scene, &scene).expect("compare_scenes should succeed");
        assert_eq!(cmp.scene_a_gaussians, 6);
        assert_eq!(cmp.scene_b_gaussians, 6);
        assert!((cmp.size_ratio - 1.0).abs() < 1e-6);
        assert!(cmp.mean_opacity_diff.abs() < 1e-5);
        assert!(cmp.mean_scale_diff.abs() < 1e-5);
        assert!(cmp.quality_score_diff.abs() < 1e-5);
        assert!(cmp.centroid_distance.abs() < 1e-5);
    }

    #[test]
    fn test_compare_scenes_different_sizes() {
        let scene_small = make_scene(4, -2.0, 0.0, [0.5, 0.5, 0.5]);
        let scene_large = make_scene(8, -2.0, 0.0, [0.5, 0.5, 0.5]);
        let cmp =
            compare_scenes(&scene_small, &scene_large).expect("compare_scenes should succeed");
        assert_eq!(cmp.scene_a_gaussians, 4);
        assert_eq!(cmp.scene_b_gaussians, 8);
        assert!(
            (cmp.size_ratio - 2.0).abs() < 1e-5,
            "size_ratio should be 2.0, got {}",
            cmp.size_ratio
        );
    }

    #[test]
    fn test_compare_scenes_different_opacity() {
        // scene_a: low opacity, scene_b: high opacity
        let scene_a = make_scene(4, -2.0, -5.0, [0.5, 0.5, 0.5]); // sigmoid(-5)≈0.0067
        let scene_b = make_scene(4, -2.0, 5.0, [0.5, 0.5, 0.5]); // sigmoid(5) ≈0.9933
        let cmp = compare_scenes(&scene_a, &scene_b).expect("should succeed");
        assert!(
            cmp.mean_opacity_diff > 0.9,
            "b should have higher opacity: diff={}",
            cmp.mean_opacity_diff
        );
    }

    #[test]
    fn test_scene_report_json_valid_format() {
        let scene = make_scene(5, -1.0, 0.0, [0.4, 0.6, 0.2]);
        let report = analyze_scene(&scene).expect("ok");
        let json = report.to_json();
        // Simple structural checks
        assert!(json.contains("\"num_gaussians\""));
        assert!(json.contains("\"quality_score\""));
        assert!(json.contains("\"opacity_histogram\""));
    }

    #[test]
    fn test_scene_report_to_json_maps_non_finite_to_null() {
        // Regression test: even though `SceneData::validate` now rejects
        // non-finite inputs, `to_json` must still degrade gracefully (emit
        // `null`) rather than write a bare `NaN`/`inf` token that breaks
        // every JSON parser, in case some derived statistic (e.g. a 0/0
        // ratio) ever turns non-finite despite finite inputs.
        let scene = make_scene(5, -1.0, 0.0, [0.4, 0.6, 0.2]);
        let mut report = analyze_scene(&scene).expect("ok");
        report.quality_score = f32::NAN;
        report.spatial.scene_diameter = f32::INFINITY;
        report.color.mean_r = f32::NEG_INFINITY;

        let json = report.to_json();

        // Parse rather than substring-scan the raw text. A bare `NaN`/`inf`
        // token is exactly what no JSON parser accepts, so a successful
        // parse *is* the property under test — and a text scan cannot state
        // it: an earlier version of this test asserted the document contains
        // no "nan" anywhere and therefore always failed on the perfectly
        // valid key `mean_lumi(nan)ce`.
        let parsed: serde_json::Value = serde_json::from_str(&json)
            .map_err(|e| format!("{e} — document was: {json}"))
            .expect("to_json must emit parseable JSON");
        let object = parsed
            .as_object()
            .expect("to_json must emit a JSON object at the top level");

        const NON_FINITE: [&str; 3] = ["quality_score", "scene_diameter", "mean_r"];
        for key in NON_FINITE {
            assert_eq!(
                object.get(key),
                Some(&serde_json::Value::Null),
                "non-finite {key} must be emitted as null: {json}"
            );
        }
        // Every other leaf must still be a finite number, so `null` marks
        // genuinely non-finite statistics and nothing else.
        for (key, value) in object {
            let acceptable = match value {
                serde_json::Value::Null => NON_FINITE.contains(&key.as_str()),
                serde_json::Value::Number(n) => n.as_f64().is_some_and(f64::is_finite),
                serde_json::Value::Array(items) => items
                    .iter()
                    .all(|item| item.as_f64().is_some_and(f64::is_finite)),
                _ => false,
            };
            assert!(
                acceptable,
                "every field must be a finite number (or null only where the statistic \
                 really is non-finite); {key} is {value}: {json}"
            );
        }
    }

    #[test]
    fn test_scene_report_format_detailed_contains_sections() {
        let scene = make_scene(5, -1.0, 0.0, [0.4, 0.6, 0.2]);
        let report = analyze_scene(&scene).expect("ok");
        let detailed = report.format_detailed();
        assert!(detailed.contains("Spatial"));
        assert!(detailed.contains("Opacity"));
        assert!(detailed.contains("Scale"));
        assert!(detailed.contains("Color"));
        assert!(detailed.contains("Quality Score"));
    }

    #[test]
    fn test_spatial_stats_nn_distance_is_positive_for_two_gaussians() {
        let scene = SceneData {
            positions: vec![0.0, 0.0, 0.0, 3.0, 4.0, 0.0], // distance = 5.0
            log_scales: vec![0.0; 6],
            rotations: vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            opacities: vec![0.0, 0.0],
            colors: vec![0.5; 6],
        };
        let stats = compute_spatial_stats(&scene).expect("ok");
        assert!(
            (stats.mean_nearest_neighbor_dist - 5.0).abs() < 1e-4,
            "NN distance should be 5.0, got {}",
            stats.mean_nearest_neighbor_dist
        );
    }

    #[test]
    fn test_spatial_stats_nn_distance_searches_full_array_beyond_sample() {
        // Regression test: with more than 100 Gaussians, the nearest-
        // neighbour *query* points are a deterministic 100-point subsample,
        // but each query must be matched against the *entire* scene. On a
        // perfectly evenly-spaced line of 200 points (spacing 1.0), every
        // point's true nearest neighbour is 1.0 away. The old code searched
        // only among the other ~100 sampled (stride-2) points, so it would
        // have reported ~2.0 instead.
        let scene = make_line_scene(200);
        let stats = compute_spatial_stats(&scene).expect("ok");
        assert!(
            (stats.mean_nearest_neighbor_dist - 1.0).abs() < 1e-3,
            "NN distance on an evenly-spaced line should be ~1.0 (the true \
             spacing), not ~2.0 (the sampled-lattice spacing); got {}",
            stats.mean_nearest_neighbor_dist
        );
    }

    #[test]
    fn test_opacity_stats_p50_and_p95_single_gaussian() {
        let scene = make_scene(1, 0.0, 0.0, [0.5, 0.5, 0.5]); // opacity = 0.5
        let stats = compute_opacity_stats(&scene).expect("ok");
        assert!((stats.p50_opacity - 0.5).abs() < 1e-5);
        assert!((stats.p95_opacity - 0.5).abs() < 1e-5);
    }
}
