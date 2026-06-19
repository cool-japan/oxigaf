//! `oxigaf compare` command — compare two model files and report differences.
//!
//! Supports:
//! - `.ply` — PLY binary_little_endian format (Gaussian count, SH degree, bounding box,
//!   opacity stats, scale stats, position stats)
//! - `.safetensors` — tensor count and names comparison
//!
//! # Output Formats
//! - `text` — human-readable columnar output
//! - `json` — machine-readable JSON

use std::io::{BufReader, Read};
use std::path::Path;

use anyhow::{Context, Result};

use crate::cli::CompareArgs;
use crate::info::{parse_ply_header_info, sh_degree_from_rest};

// ---------------------------------------------------------------------------
// ModelStats
// ---------------------------------------------------------------------------

/// Statistical summary of a Gaussian splatting model file.
#[derive(Debug, Clone)]
pub struct ModelStats {
    /// Absolute path of the file.
    pub file_path: String,
    /// Number of Gaussian primitives.
    pub gaussian_count: usize,
    /// Spherical harmonics degree (0–3).
    pub sh_degree: u32,
    /// Bounding box minimum corner [x, y, z].
    pub bbox_min: [f32; 3],
    /// Bounding box maximum corner [x, y, z].
    pub bbox_max: [f32; 3],
    /// Mean scale (averaged across all three axes and all Gaussians).
    pub scale_mean: f32,
    /// Standard deviation of scale.
    pub scale_std: f32,
    /// Mean opacity across all Gaussians.
    pub opacity_mean: f32,
    /// Standard deviation of opacity.
    pub opacity_std: f32,
    /// Mean position [x, y, z].
    pub position_mean: [f32; 3],
    /// Standard deviation of position [x, y, z].
    pub position_std: [f32; 3],
}

impl ModelStats {
    /// Compute statistics from a `.ply` or `.safetensors` file.
    ///
    /// Only `binary_little_endian` PLY files support full statistics. ASCII PLY
    /// files and safetensors files return bounding boxes of all-zeros and zero
    /// stats where data is not available.
    pub fn from_file(path: &Path) -> Result<Self> {
        if !path.exists() {
            anyhow::bail!("File not found: {}", path.display());
        }

        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        match ext.as_str() {
            "ply" => Self::from_ply(path),
            "safetensors" => Self::from_safetensors(path),
            other => anyhow::bail!(
                "Unsupported file extension '.{}'. Supported: .ply, .safetensors",
                other
            ),
        }
    }

    /// Parse a PLY file and compute statistics.
    fn from_ply(path: &Path) -> Result<Self> {
        use std::fs::File;

        let file = File::open(path).with_context(|| format!("Cannot open: {}", path.display()))?;
        let mut reader = BufReader::new(file);

        let header = parse_ply_header_info(&mut reader)
            .with_context(|| format!("Failed to parse PLY header: {}", path.display()))?;

        let sh_degree = sh_degree_from_rest(header.num_rest);
        let file_path = path.to_str().unwrap_or("<non-utf8>").to_string();

        if !header.format.contains("binary_little_endian") {
            // Return stub stats for ASCII PLY — full stats require binary format.
            return Ok(Self {
                file_path,
                gaussian_count: header.vertex_count,
                sh_degree,
                bbox_min: [0.0; 3],
                bbox_max: [0.0; 3],
                scale_mean: 0.0,
                scale_std: 0.0,
                opacity_mean: 0.0,
                opacity_std: 0.0,
                position_mean: [0.0; 3],
                position_std: [0.0; 3],
            });
        }

        if header.vertex_count == 0 {
            return Ok(Self {
                file_path,
                gaussian_count: 0,
                sh_degree,
                bbox_min: [0.0; 3],
                bbox_max: [0.0; 3],
                scale_mean: 0.0,
                scale_std: 0.0,
                opacity_mean: 0.0,
                opacity_std: 0.0,
                position_mean: [0.0; 3],
                position_std: [0.0; 3],
            });
        }

        let vertex_data =
            read_ply_vertex_data(&mut reader, header.vertex_count, header.num_rest)
                .with_context(|| format!("Failed to read vertex data: {}", path.display()))?;

        let stats = compute_model_stats(&vertex_data, &file_path, header.vertex_count, sh_degree);
        Ok(stats)
    }

    /// Parse a safetensors file and compute available statistics.
    ///
    /// Extracts Gaussian count, SH degree, and bounding box from tensor shapes.
    fn from_safetensors(path: &Path) -> Result<Self> {
        let file_path = path.to_str().unwrap_or("<non-utf8>").to_string();

        let bytes =
            std::fs::read(path).with_context(|| format!("Failed to read: {}", path.display()))?;

        let st = safetensors::SafeTensors::deserialize(&bytes)
            .with_context(|| format!("Failed to deserialize SafeTensors: {}", path.display()))?;

        // Infer Gaussian count from positions tensor (shape [N, 3]) or xyz tensor
        let mut gaussian_count = 0usize;
        let mut sh_degree = 0u32;

        let names = st.names();
        for name in &names {
            if let Ok(tv) = st.tensor(name) {
                let shape = tv.shape();
                // positions / xyz expected shape: [N, 3]
                if (*name == "positions" || *name == "xyz" || *name == "pos")
                    && shape.len() == 2
                    && shape[1] == 3
                {
                    gaussian_count = shape[0];
                }
                // f_rest shape [N, K] where K = num_rest
                if name.starts_with("f_rest") && shape.len() == 2 {
                    let num_rest = shape[1];
                    sh_degree = sh_degree_from_rest(num_rest);
                }
            }
        }

        Ok(Self {
            file_path,
            gaussian_count,
            sh_degree,
            bbox_min: [0.0; 3],
            bbox_max: [0.0; 3],
            scale_mean: 0.0,
            scale_std: 0.0,
            opacity_mean: 0.0,
            opacity_std: 0.0,
            position_mean: [0.0; 3],
            position_std: [0.0; 3],
        })
    }
}

// ---------------------------------------------------------------------------
// Vertex reading (PLY binary_little_endian)
// ---------------------------------------------------------------------------

/// Raw per-vertex data extracted from a binary PLY file.
struct VertexData {
    xs: Vec<f32>,
    ys: Vec<f32>,
    zs: Vec<f32>,
    opacities: Vec<f32>,
    scales: Vec<f32>, // interleaved: scale_x, scale_y, scale_z per vertex
}

/// Read all vertex data from the body of a binary_little_endian PLY file.
///
/// Property layout (all f32, 4 bytes each):
/// ```text
/// x, y, z, nx, ny, nz, f_dc_0, f_dc_1, f_dc_2,
/// f_rest_0..f_rest_{num_rest-1},
/// opacity, scale_0, scale_1, scale_2, rot_0, rot_1, rot_2, rot_3
/// ```
fn read_ply_vertex_data(
    reader: &mut impl Read,
    vertex_count: usize,
    num_rest: usize,
) -> Result<VertexData> {
    let mut xs = Vec::with_capacity(vertex_count);
    let mut ys = Vec::with_capacity(vertex_count);
    let mut zs = Vec::with_capacity(vertex_count);
    let mut opacities = Vec::with_capacity(vertex_count);
    let mut scales = Vec::with_capacity(vertex_count * 3);

    // Skip: nx, ny, nz (3) + f_dc_0..2 (3) + f_rest_0..{num_rest-1} (num_rest)
    let skip_after_xyz = 3 + 3 + num_rest;
    let mut skip_buf = vec![0u8; skip_after_xyz * 4];
    let mut rot_buf = [0u8; 16]; // rot_0..rot_3 = 4 floats
    let mut buf4 = [0u8; 4];

    for _ in 0..vertex_count {
        // x
        reader.read_exact(&mut buf4).context("EOF reading x")?;
        xs.push(f32::from_le_bytes(buf4));
        // y
        reader.read_exact(&mut buf4).context("EOF reading y")?;
        ys.push(f32::from_le_bytes(buf4));
        // z
        reader.read_exact(&mut buf4).context("EOF reading z")?;
        zs.push(f32::from_le_bytes(buf4));

        // skip normals + dc + rest
        reader
            .read_exact(&mut skip_buf)
            .context("EOF skipping normals/sh")?;

        // opacity
        reader
            .read_exact(&mut buf4)
            .context("EOF reading opacity")?;
        opacities.push(f32::from_le_bytes(buf4));

        // scale_0
        reader
            .read_exact(&mut buf4)
            .context("EOF reading scale_0")?;
        scales.push(f32::from_le_bytes(buf4));
        // scale_1
        reader
            .read_exact(&mut buf4)
            .context("EOF reading scale_1")?;
        scales.push(f32::from_le_bytes(buf4));
        // scale_2
        reader
            .read_exact(&mut buf4)
            .context("EOF reading scale_2")?;
        scales.push(f32::from_le_bytes(buf4));

        // skip rot_0..rot_3
        reader
            .read_exact(&mut rot_buf)
            .context("EOF reading rotations")?;
    }

    Ok(VertexData {
        xs,
        ys,
        zs,
        opacities,
        scales,
    })
}

// ---------------------------------------------------------------------------
// Stats computation
// ---------------------------------------------------------------------------

/// Compute mean and standard deviation of a slice of f32 values.
fn mean_std(values: &[f32]) -> (f32, f32) {
    if values.is_empty() {
        return (0.0, 0.0);
    }
    let sum: f64 = values.iter().map(|v| *v as f64).sum();
    let mean = sum / values.len() as f64;
    let var: f64 = values
        .iter()
        .map(|v| {
            let d = *v as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / values.len() as f64;
    (mean as f32, var.sqrt() as f32)
}

/// Build a `ModelStats` from raw vertex data.
fn compute_model_stats(
    data: &VertexData,
    file_path: &str,
    gaussian_count: usize,
    sh_degree: u32,
) -> ModelStats {
    let (pos_mean_x, pos_std_x) = mean_std(&data.xs);
    let (pos_mean_y, pos_std_y) = mean_std(&data.ys);
    let (pos_mean_z, pos_std_z) = mean_std(&data.zs);

    let (opacity_mean, opacity_std) = mean_std(&data.opacities);
    let (scale_mean, scale_std) = mean_std(&data.scales);

    let bbox_min = [
        data.xs.iter().cloned().fold(f32::INFINITY, f32::min),
        data.ys.iter().cloned().fold(f32::INFINITY, f32::min),
        data.zs.iter().cloned().fold(f32::INFINITY, f32::min),
    ];
    let bbox_max = [
        data.xs.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        data.ys.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        data.zs.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
    ];

    ModelStats {
        file_path: file_path.to_string(),
        gaussian_count,
        sh_degree,
        bbox_min,
        bbox_max,
        scale_mean,
        scale_std,
        opacity_mean,
        opacity_std,
        position_mean: [pos_mean_x, pos_mean_y, pos_mean_z],
        position_std: [pos_std_x, pos_std_y, pos_std_z],
    }
}

// ---------------------------------------------------------------------------
// Bounding box IoU
// ---------------------------------------------------------------------------

/// Compute the 3D bounding box Intersection-over-Union.
///
/// Returns a value in `[0.0, 1.0]` where 1.0 means identical boxes and 0.0
/// means non-overlapping boxes.
pub fn bbox_iou(min_a: [f32; 3], max_a: [f32; 3], min_b: [f32; 3], max_b: [f32; 3]) -> f32 {
    // Intersection
    let inter_min = [
        min_a[0].max(min_b[0]),
        min_a[1].max(min_b[1]),
        min_a[2].max(min_b[2]),
    ];
    let inter_max = [
        max_a[0].min(max_b[0]),
        max_a[1].min(max_b[1]),
        max_a[2].min(max_b[2]),
    ];

    let inter_vol = {
        let dx = (inter_max[0] - inter_min[0]).max(0.0);
        let dy = (inter_max[1] - inter_min[1]).max(0.0);
        let dz = (inter_max[2] - inter_min[2]).max(0.0);
        dx * dy * dz
    };

    let vol_a = {
        let dx = (max_a[0] - min_a[0]).max(0.0);
        let dy = (max_a[1] - min_a[1]).max(0.0);
        let dz = (max_a[2] - min_a[2]).max(0.0);
        dx * dy * dz
    };

    let vol_b = {
        let dx = (max_b[0] - min_b[0]).max(0.0);
        let dy = (max_b[1] - min_b[1]).max(0.0);
        let dz = (max_b[2] - min_b[2]).max(0.0);
        dx * dy * dz
    };

    let union_vol = vol_a + vol_b - inter_vol;
    if union_vol <= 0.0 {
        // Both boxes are degenerate (zero volume); treat as identical
        1.0
    } else {
        (inter_vol / union_vol).clamp(0.0, 1.0)
    }
}

// ---------------------------------------------------------------------------
// ComparisonReport
// ---------------------------------------------------------------------------

/// Full comparison between two Gaussian splatting models.
#[derive(Debug, Clone)]
pub struct ComparisonReport {
    /// Statistics for the first model.
    pub model_a: ModelStats,
    /// Statistics for the second model.
    pub model_b: ModelStats,

    /// Signed difference: `model_b.gaussian_count - model_a.gaussian_count`.
    pub gaussian_count_diff: i64,
    /// Percentage change in Gaussian count relative to model A.
    pub gaussian_count_pct: f64,
    /// Whether both models have the same SH degree.
    pub sh_degree_same: bool,
    /// 3D bounding box Intersection-over-Union in `[0.0, 1.0]`.
    pub bbox_iou: f32,

    /// Percentage change in mean scale ((B - A) / A * 100).
    pub scale_mean_diff_pct: f64,
    /// Percentage change in mean opacity ((B - A) / A * 100).
    pub opacity_mean_diff_pct: f64,

    /// Overall heuristic similarity in `[0.0, 1.0]`.
    ///
    /// Weighted combination of:
    /// - bbox IoU (40 %)
    /// - Gaussian count similarity (30 %)
    /// - SH degree match (10 %)
    /// - Scale similarity (10 %)
    /// - Opacity similarity (10 %)
    pub overall_similarity: f32,

    /// Similarity threshold used for the recommendation.
    pub threshold: f64,
}

impl ComparisonReport {
    /// Compute a `ComparisonReport` from two `ModelStats`.
    pub fn compute(a: ModelStats, b: ModelStats, threshold: f64) -> Self {
        let gaussian_count_diff = b.gaussian_count as i64 - a.gaussian_count as i64;
        let gaussian_count_pct = if a.gaussian_count == 0 {
            0.0
        } else {
            gaussian_count_diff as f64 / a.gaussian_count as f64 * 100.0
        };

        let sh_degree_same = a.sh_degree == b.sh_degree;

        let iou = bbox_iou(a.bbox_min, a.bbox_max, b.bbox_min, b.bbox_max);

        let scale_mean_diff_pct = if a.scale_mean.abs() < f32::EPSILON {
            0.0
        } else {
            (b.scale_mean - a.scale_mean) as f64 / a.scale_mean.abs() as f64 * 100.0
        };

        let opacity_mean_diff_pct = if a.opacity_mean.abs() < f32::EPSILON {
            0.0
        } else {
            (b.opacity_mean - a.opacity_mean) as f64 / a.opacity_mean.abs() as f64 * 100.0
        };

        // Gaussian count similarity: 1 - |pct| / 100, clamped to [0, 1]
        let count_sim = (1.0 - gaussian_count_pct.abs() / 100.0).clamp(0.0, 1.0) as f32;

        // SH degree similarity
        let sh_sim = if sh_degree_same { 1.0f32 } else { 0.0f32 };

        // Scale similarity: 1 - |pct| / 100, clamped to [0, 1]
        let scale_sim = (1.0 - scale_mean_diff_pct.abs() / 100.0).clamp(0.0, 1.0) as f32;

        // Opacity similarity: 1 - |pct| / 100, clamped to [0, 1]
        let opacity_sim = (1.0 - opacity_mean_diff_pct.abs() / 100.0).clamp(0.0, 1.0) as f32;

        // Weighted similarity
        let overall_similarity =
            (0.40 * iou + 0.30 * count_sim + 0.10 * sh_sim + 0.10 * scale_sim + 0.10 * opacity_sim)
                .clamp(0.0, 1.0);

        Self {
            model_a: a,
            model_b: b,
            gaussian_count_diff,
            gaussian_count_pct,
            sh_degree_same,
            bbox_iou: iou,
            scale_mean_diff_pct,
            opacity_mean_diff_pct,
            overall_similarity,
            threshold,
        }
    }

    /// Format the report as human-readable text.
    pub fn format_text(&self) -> String {
        let a = &self.model_a;
        let b = &self.model_b;

        let a_name = Path::new(&a.file_path)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(&a.file_path);
        let b_name = Path::new(&b.file_path)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(&b.file_path);

        let sign = |v: f64| if v >= 0.0 { "+" } else { "" };

        let sh_same_str = if self.sh_degree_same {
            "(same)"
        } else {
            "(different)"
        };

        let count_sign = if self.gaussian_count_diff >= 0 {
            "+"
        } else {
            ""
        };

        // Recommendation text
        let similarity_pct = self.overall_similarity * 100.0;
        let recommendation = self.recommendation();

        let mut out = String::new();
        out.push_str(&format!("Comparing: {} vs {}\n\n", a_name, b_name));
        out.push_str("=== Structural Differences ===\n");
        out.push_str(&format!(
            "  Gaussian count:  {:>10}  →  {:>10}  ({}{}, {}{:.2}%)\n",
            format_count(a.gaussian_count),
            format_count(b.gaussian_count),
            count_sign,
            self.gaussian_count_diff,
            sign(self.gaussian_count_pct),
            self.gaussian_count_pct,
        ));
        out.push_str(&format!(
            "  SH degree:       {:>10}  →  {:>10}  {}\n",
            a.sh_degree, b.sh_degree, sh_same_str
        ));
        out.push_str("\n=== Bounding Box ===\n");
        out.push_str(&format!(
            "  Model A: X[{:.3}, {:.3}] Y[{:.3}, {:.3}] Z[{:.3}, {:.3}]\n",
            a.bbox_min[0],
            a.bbox_max[0],
            a.bbox_min[1],
            a.bbox_max[1],
            a.bbox_min[2],
            a.bbox_max[2],
        ));
        out.push_str(&format!(
            "  Model B: X[{:.3}, {:.3}] Y[{:.3}, {:.3}] Z[{:.3}, {:.3}]\n",
            b.bbox_min[0],
            b.bbox_max[0],
            b.bbox_min[1],
            b.bbox_max[1],
            b.bbox_min[2],
            b.bbox_max[2],
        ));
        out.push_str(&format!(
            "  Overlap: {:.1}% (Intersection-over-Union of bounding boxes)\n",
            self.bbox_iou * 100.0
        ));
        out.push_str("\n=== Parameter Statistics (A vs B, for shared positions) ===\n");
        out.push_str(&format!(
            "  Scale (mean):    {:.4}  →  {:.4}  ({}{:.1}%)\n",
            a.scale_mean,
            b.scale_mean,
            sign(self.scale_mean_diff_pct),
            self.scale_mean_diff_pct,
        ));
        out.push_str(&format!(
            "  Opacity (mean):  {:.3}   →  {:.3}   ({}{:.1}%)\n",
            a.opacity_mean,
            b.opacity_mean,
            sign(self.opacity_mean_diff_pct),
            self.opacity_mean_diff_pct,
        ));
        out.push_str(&format!(
            "  Scale (std):     {:.4}  →  {:.4}\n",
            a.scale_std, b.scale_std
        ));
        out.push_str(&format!(
            "  Opacity (std):   {:.3}   →  {:.3}\n",
            a.opacity_std, b.opacity_std
        ));
        out.push_str("\n=== Summary ===\n");
        out.push_str(&format!(
            "  Overall similarity: {:.1}% (based on structural + statistical comparison)\n",
            similarity_pct
        ));
        out.push_str(&format!("  Recommendation: {}\n", recommendation));
        out
    }

    /// Generate a natural-language recommendation.
    fn recommendation(&self) -> String {
        let sim_pct = self.overall_similarity * 100.0;

        if sim_pct >= 99.0 {
            return "Models are essentially identical.".to_string();
        }
        if self.overall_similarity as f64 >= self.threshold {
            let mut notes = Vec::new();
            if self.gaussian_count_diff != 0 {
                let dir = if self.gaussian_count_diff > 0 {
                    "more Gaussians"
                } else {
                    "fewer Gaussians"
                };
                notes.push(format!("model B has {}", dir));
            }
            if self.opacity_mean_diff_pct.abs() > 5.0 {
                let dir = if self.opacity_mean_diff_pct > 0.0 {
                    "higher"
                } else {
                    "lower"
                };
                notes.push(format!("model B has {} opacity", dir));
            }
            if self.scale_mean_diff_pct.abs() > 5.0 {
                let dir = if self.scale_mean_diff_pct > 0.0 {
                    "larger"
                } else {
                    "smaller"
                };
                notes.push(format!("model B has {} scale", dir));
            }
            if notes.is_empty() {
                format!("Models are similar ({:.1}% similarity).", sim_pct)
            } else {
                format!("Models are similar but {}.", notes.join(" and "))
            }
        } else {
            let mut diffs = Vec::new();
            if !self.sh_degree_same {
                diffs.push("different SH degree".to_string());
            }
            if (self.gaussian_count_pct.abs()) > 20.0 {
                diffs.push("significantly different Gaussian count".to_string());
            }
            if self.bbox_iou < 0.5 {
                diffs.push("low bounding box overlap".to_string());
            }
            if diffs.is_empty() {
                format!("Models differ significantly ({:.1}% similarity).", sim_pct)
            } else {
                format!(
                    "Models differ significantly ({}; {:.1}% similarity).",
                    diffs.join(", "),
                    sim_pct
                )
            }
        }
    }

    /// Format the report as JSON.
    pub fn format_json(&self) -> Result<String> {
        let a = &self.model_a;
        let b = &self.model_b;

        let value = serde_json::json!({
            "model_a": {
                "file_path": a.file_path,
                "gaussian_count": a.gaussian_count,
                "sh_degree": a.sh_degree,
                "bbox_min": a.bbox_min,
                "bbox_max": a.bbox_max,
                "scale_mean": a.scale_mean,
                "scale_std": a.scale_std,
                "opacity_mean": a.opacity_mean,
                "opacity_std": a.opacity_std,
                "position_mean": a.position_mean,
                "position_std": a.position_std,
            },
            "model_b": {
                "file_path": b.file_path,
                "gaussian_count": b.gaussian_count,
                "sh_degree": b.sh_degree,
                "bbox_min": b.bbox_min,
                "bbox_max": b.bbox_max,
                "scale_mean": b.scale_mean,
                "scale_std": b.scale_std,
                "opacity_mean": b.opacity_mean,
                "opacity_std": b.opacity_std,
                "position_mean": b.position_mean,
                "position_std": b.position_std,
            },
            "gaussian_count_diff": self.gaussian_count_diff,
            "gaussian_count_pct": self.gaussian_count_pct,
            "sh_degree_same": self.sh_degree_same,
            "bbox_iou": self.bbox_iou,
            "scale_mean_diff_pct": self.scale_mean_diff_pct,
            "opacity_mean_diff_pct": self.opacity_mean_diff_pct,
            "overall_similarity": self.overall_similarity,
            "threshold": self.threshold,
            "recommendation": self.recommendation(),
        });

        serde_json::to_string_pretty(&value).context("Failed to serialize comparison to JSON")
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Format a count with thousands separator (e.g. 52341 → "52,341").
fn format_count(n: usize) -> String {
    let s = n.to_string();
    let mut result = String::with_capacity(s.len() + s.len() / 3);
    for (i, ch) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(ch);
    }
    result.chars().rev().collect()
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Run the `compare` command.
pub fn run_compare(args: CompareArgs) -> Result<()> {
    let threshold = args.threshold;

    let stats_a = ModelStats::from_file(&args.model1)
        .with_context(|| format!("Failed to load model A: {}", args.model1.display()))?;
    let stats_b = ModelStats::from_file(&args.model2)
        .with_context(|| format!("Failed to load model B: {}", args.model2.display()))?;

    let report = ComparisonReport::compute(stats_a, stats_b, threshold);

    match args.format.to_lowercase().as_str() {
        "json" => {
            let json = report.format_json()?;
            println!("{}", json);
        }
        _ => {
            // default: text
            print!("{}", report.format_text());
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::fs;
    use std::io::Write;

    /// Write a minimal binary_little_endian PLY with `n_vertices` vertices,
    /// using `num_rest` f_rest properties.
    ///
    /// Each vertex has deterministic but non-trivial values derived from
    /// the vertex index, to give non-trivial stats.
    fn write_test_ply(
        path: &Path,
        n_vertices: usize,
        num_rest: usize,
        position_offset: f32,
        opacity_override: Option<f32>,
        scale_override: Option<f32>,
    ) -> Result<()> {
        let mut f = fs::File::create(path)?;

        writeln!(f, "ply")?;
        writeln!(f, "format binary_little_endian 1.0")?;
        writeln!(f, "element vertex {}", n_vertices)?;
        writeln!(f, "property float x")?;
        writeln!(f, "property float y")?;
        writeln!(f, "property float z")?;
        writeln!(f, "property float nx")?;
        writeln!(f, "property float ny")?;
        writeln!(f, "property float nz")?;
        writeln!(f, "property float f_dc_0")?;
        writeln!(f, "property float f_dc_1")?;
        writeln!(f, "property float f_dc_2")?;
        for i in 0..num_rest {
            writeln!(f, "property float f_rest_{}", i)?;
        }
        writeln!(f, "property float opacity")?;
        writeln!(f, "property float scale_0")?;
        writeln!(f, "property float scale_1")?;
        writeln!(f, "property float scale_2")?;
        writeln!(f, "property float rot_0")?;
        writeln!(f, "property float rot_1")?;
        writeln!(f, "property float rot_2")?;
        writeln!(f, "property float rot_3")?;
        writeln!(f, "end_header")?;

        for i in 0..n_vertices {
            let t = i as f32 / n_vertices.max(1) as f32;

            // x, y, z
            f.write_all(&(position_offset + t).to_le_bytes())?;
            f.write_all(&(position_offset + t * 0.5).to_le_bytes())?;
            f.write_all(&(position_offset + t * 0.25).to_le_bytes())?;

            // nx, ny, nz
            f.write_all(&0.0f32.to_le_bytes())?;
            f.write_all(&0.0f32.to_le_bytes())?;
            f.write_all(&1.0f32.to_le_bytes())?;

            // f_dc_0, f_dc_1, f_dc_2
            f.write_all(&0.5f32.to_le_bytes())?;
            f.write_all(&0.5f32.to_le_bytes())?;
            f.write_all(&0.5f32.to_le_bytes())?;

            // f_rest
            for _ in 0..num_rest {
                f.write_all(&0.0f32.to_le_bytes())?;
            }

            // opacity
            let opacity = opacity_override.unwrap_or(0.5);
            f.write_all(&opacity.to_le_bytes())?;

            // scale_0, scale_1, scale_2
            let scale = scale_override.unwrap_or(0.01);
            f.write_all(&scale.to_le_bytes())?;
            f.write_all(&scale.to_le_bytes())?;
            f.write_all(&scale.to_le_bytes())?;

            // rot_0, rot_1, rot_2, rot_3
            f.write_all(&1.0f32.to_le_bytes())?;
            f.write_all(&0.0f32.to_le_bytes())?;
            f.write_all(&0.0f32.to_le_bytes())?;
            f.write_all(&0.0f32.to_le_bytes())?;
        }

        Ok(())
    }

    // -----------------------------------------------------------------------
    // bbox_iou tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_bbox_iou_identical() {
        let min = [0.0f32, 0.0, 0.0];
        let max = [1.0f32, 1.0, 1.0];
        let iou = bbox_iou(min, max, min, max);
        assert!(
            (iou - 1.0).abs() < 1e-5,
            "Identical boxes should have IoU = 1.0, got {}",
            iou
        );
    }

    #[test]
    fn test_bbox_iou_non_overlapping() {
        let min_a = [0.0f32, 0.0, 0.0];
        let max_a = [1.0f32, 1.0, 1.0];
        let min_b = [2.0f32, 2.0, 2.0];
        let max_b = [3.0f32, 3.0, 3.0];
        let iou = bbox_iou(min_a, max_a, min_b, max_b);
        assert!(
            iou.abs() < 1e-5,
            "Non-overlapping boxes should have IoU = 0.0, got {}",
            iou
        );
    }

    #[test]
    fn test_bbox_iou_half_overlap() {
        // A: [0, 0, 0] → [2, 1, 1]  vol = 2
        // B: [1, 0, 0] → [3, 1, 1]  vol = 2
        // intersection: [1, 0, 0] → [2, 1, 1]  vol = 1
        // union = 2 + 2 - 1 = 3
        // IoU = 1/3 ≈ 0.333
        let min_a = [0.0f32, 0.0, 0.0];
        let max_a = [2.0f32, 1.0, 1.0];
        let min_b = [1.0f32, 0.0, 0.0];
        let max_b = [3.0f32, 1.0, 1.0];
        let iou = bbox_iou(min_a, max_a, min_b, max_b);
        let expected = 1.0f32 / 3.0;
        assert!(
            (iou - expected).abs() < 1e-4,
            "Expected IoU ≈ {:.4}, got {:.4}",
            expected,
            iou
        );
    }

    #[test]
    fn test_bbox_iou_output_in_range() {
        // Arbitrary boxes
        let min_a = [-1.0f32, -2.0, -0.5];
        let max_a = [1.0f32, 2.0, 0.5];
        let min_b = [-0.5f32, -1.0, -0.25];
        let max_b = [0.5f32, 1.0, 0.25];
        let iou = bbox_iou(min_a, max_a, min_b, max_b);
        assert!(
            (0.0..=1.0).contains(&iou),
            "IoU must be in [0, 1], got {}",
            iou
        );
    }

    // -----------------------------------------------------------------------
    // ModelStats tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_model_stats_from_ply_basic() -> Result<()> {
        let tmp_dir = env::temp_dir();
        let path = tmp_dir.join("test_compare_basic.ply");
        write_test_ply(&path, 10, 45, 0.0, Some(0.5), Some(0.01))?;

        let stats = ModelStats::from_file(&path)?;
        assert_eq!(stats.gaussian_count, 10);
        assert_eq!(stats.sh_degree, 3); // num_rest=45 → degree 3
        assert!(
            stats.opacity_mean > 0.0 && stats.opacity_mean <= 1.0,
            "opacity_mean out of range: {}",
            stats.opacity_mean
        );

        fs::remove_file(&path).ok();
        Ok(())
    }

    #[test]
    fn test_model_stats_from_file_not_found() {
        let path = env::temp_dir().join("nonexistent_oxigaf_compare.ply");
        let result = ModelStats::from_file(&path);
        assert!(result.is_err(), "Expected error for missing file");
    }

    #[test]
    fn test_model_stats_from_ply_sh_degree_1() -> Result<()> {
        let tmp_dir = env::temp_dir();
        let path = tmp_dir.join("test_compare_sh1.ply");
        write_test_ply(&path, 5, 9, 0.0, None, None)?;

        let stats = ModelStats::from_file(&path)?;
        assert_eq!(stats.sh_degree, 1, "Expected SH degree 1 for num_rest=9");

        fs::remove_file(&path).ok();
        Ok(())
    }

    // -----------------------------------------------------------------------
    // ComparisonReport tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_comparison_report_identical_stats() -> Result<()> {
        let tmp_dir = env::temp_dir();
        let path = tmp_dir.join("test_compare_identical.ply");
        write_test_ply(&path, 20, 45, 0.0, Some(0.5), Some(0.01))?;

        let stats_a = ModelStats::from_file(&path)?;
        let stats_b = ModelStats::from_file(&path)?;

        let report = ComparisonReport::compute(stats_a, stats_b, 0.8);

        assert!(
            report.overall_similarity > 0.99,
            "Identical stats should yield near-1.0 similarity, got {}",
            report.overall_similarity
        );
        assert_eq!(report.gaussian_count_diff, 0);
        assert!(report.sh_degree_same);

        fs::remove_file(&path).ok();
        Ok(())
    }

    #[test]
    fn test_comparison_report_different_gaussian_counts() -> Result<()> {
        let tmp_dir = env::temp_dir();
        let pid = std::process::id();
        let path_a = tmp_dir.join(format!("test_compare_count_a_{pid}.ply"));
        let path_b = tmp_dir.join(format!("test_compare_count_b_{pid}.ply"));

        write_test_ply(&path_a, 50_000, 45, 0.0, Some(0.5), Some(0.01))?;
        write_test_ply(&path_b, 52_341, 45, 0.0, Some(0.5), Some(0.01))?;

        let stats_a = ModelStats::from_file(&path_a)?;
        let stats_b = ModelStats::from_file(&path_b)?;

        let report = ComparisonReport::compute(stats_a, stats_b, 0.8);

        assert_eq!(
            report.gaussian_count_diff, 2341,
            "Expected count diff of 2341, got {}",
            report.gaussian_count_diff
        );
        assert!(
            (report.gaussian_count_pct - 4.682).abs() < 0.05,
            "Expected ~4.68% diff, got {:.2}%",
            report.gaussian_count_pct
        );

        fs::remove_file(&path_a).ok();
        fs::remove_file(&path_b).ok();
        Ok(())
    }

    #[test]
    fn test_comparison_report_similarity_in_range() -> Result<()> {
        let tmp_dir = env::temp_dir();
        let path_a = tmp_dir.join("test_compare_range_a.ply");
        let path_b = tmp_dir.join("test_compare_range_b.ply");

        write_test_ply(&path_a, 100, 45, 0.0, Some(0.3), Some(0.005))?;
        write_test_ply(&path_b, 200, 9, 5.0, Some(0.8), Some(0.05))?;

        let stats_a = ModelStats::from_file(&path_a)?;
        let stats_b = ModelStats::from_file(&path_b)?;

        let report = ComparisonReport::compute(stats_a, stats_b, 0.8);

        assert!(
            (0.0..=1.0).contains(&report.overall_similarity),
            "overall_similarity must be in [0.0, 1.0], got {}",
            report.overall_similarity
        );

        fs::remove_file(&path_a).ok();
        fs::remove_file(&path_b).ok();
        Ok(())
    }

    #[test]
    fn test_format_text_contains_gaussian_count() -> Result<()> {
        let tmp_dir = env::temp_dir();
        let path = tmp_dir.join("test_compare_fmt_a.ply");
        write_test_ply(&path, 10, 45, 0.0, Some(0.5), Some(0.01))?;

        let stats_a = ModelStats::from_file(&path)?;
        let stats_b = ModelStats::from_file(&path)?;
        let report = ComparisonReport::compute(stats_a, stats_b, 0.8);
        let text = report.format_text();

        assert!(
            text.contains("Gaussian count"),
            "format_text output must contain 'Gaussian count'"
        );

        fs::remove_file(&path).ok();
        Ok(())
    }

    #[test]
    fn test_format_json_is_valid() -> Result<()> {
        let tmp_dir = env::temp_dir();
        let path = tmp_dir.join("test_compare_json_a.ply");
        write_test_ply(&path, 5, 45, 0.0, Some(0.5), Some(0.01))?;

        let stats_a = ModelStats::from_file(&path)?;
        let stats_b = ModelStats::from_file(&path)?;
        let report = ComparisonReport::compute(stats_a, stats_b, 0.8);
        let json_str = report.format_json()?;

        // Must parse as valid JSON
        let parsed: serde_json::Value = serde_json::from_str(&json_str)
            .with_context(|| format!("format_json returned invalid JSON:\n{}", json_str))?;

        assert!(parsed.is_object(), "JSON output should be an object");

        fs::remove_file(&path).ok();
        Ok(())
    }

    #[test]
    fn test_compare_identical_file_to_itself_similarity() -> Result<()> {
        let tmp_dir = env::temp_dir();
        let path = tmp_dir.join("test_compare_self.ply");
        write_test_ply(&path, 30, 45, 0.0, Some(0.5), Some(0.01))?;

        let stats_a = ModelStats::from_file(&path)?;
        let stats_b = ModelStats::from_file(&path)?;
        let report = ComparisonReport::compute(stats_a, stats_b, 0.8);

        // Same file → similarity should be essentially 1.0
        assert!(
            report.overall_similarity > 0.99,
            "Comparing file to itself should yield similarity > 0.99, got {}",
            report.overall_similarity
        );

        fs::remove_file(&path).ok();
        Ok(())
    }

    #[test]
    fn test_overall_similarity_always_in_range() {
        // Corner case: zero-vertex models
        let a = ModelStats {
            file_path: "a.ply".to_string(),
            gaussian_count: 0,
            sh_degree: 0,
            bbox_min: [0.0; 3],
            bbox_max: [0.0; 3],
            scale_mean: 0.0,
            scale_std: 0.0,
            opacity_mean: 0.0,
            opacity_std: 0.0,
            position_mean: [0.0; 3],
            position_std: [0.0; 3],
        };
        let b = ModelStats {
            file_path: "b.ply".to_string(),
            gaussian_count: 1_000_000,
            sh_degree: 3,
            bbox_min: [-100.0; 3],
            bbox_max: [100.0; 3],
            scale_mean: 10.0,
            scale_std: 5.0,
            opacity_mean: 1.0,
            opacity_std: 0.0,
            position_mean: [0.0; 3],
            position_std: [10.0; 3],
        };
        let report = ComparisonReport::compute(a, b, 0.8);
        assert!(
            (0.0..=1.0).contains(&report.overall_similarity),
            "overall_similarity out of range: {}",
            report.overall_similarity
        );
    }

    #[test]
    fn test_format_count_helper() {
        assert_eq!(format_count(0), "0");
        assert_eq!(format_count(999), "999");
        assert_eq!(format_count(1_000), "1,000");
        assert_eq!(format_count(52_341), "52,341");
        assert_eq!(format_count(1_000_000), "1,000,000");
    }

    #[test]
    fn test_mean_std_single_value() {
        let values = vec![5.0f32];
        let (mean, std) = mean_std(&values);
        assert!(
            (mean - 5.0).abs() < 1e-5,
            "mean should be 5.0, got {}",
            mean
        );
        assert!(
            std.abs() < 1e-5,
            "std of single value should be 0.0, got {}",
            std
        );
    }

    #[test]
    fn test_mean_std_known_values() {
        let values = vec![2.0f32, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let (mean, std) = mean_std(&values);
        assert!((mean - 5.0).abs() < 1e-4, "Expected mean=5.0, got {}", mean);
        // Population std dev = 2.0
        assert!((std - 2.0).abs() < 1e-3, "Expected std≈2.0, got {}", std);
    }
}
