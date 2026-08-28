//! `oxigaf compare` command — compare two model files and report differences.
//!
//! Supports:
//! - `.ply` — both `ascii` and `binary_little_endian` PLY (Gaussian count,
//!   SH degree, bounding box, opacity stats, scale stats, position stats)
//! - `.safetensors` — Gaussian count and SH degree always; full
//!   distributional statistics too when `positions`/`scales`/`opacities`
//!   are stored as `F32` (see [`ModelStats::stats_available`])
//!
//! # Output Formats
//! - `text` — human-readable columnar output
//! - `json` — machine-readable JSON

use std::io::{BufRead, BufReader, Read};
use std::path::Path;

use anyhow::{Context, Result};

use crate::cli::CompareArgs;
use crate::info::{parse_ply_header_info, sh_degree_from_rest, MAX_INITIAL_VERTEX_CAPACITY};

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
    /// Whether `bbox_min`/`bbox_max`/`scale_*`/`opacity_*`/`position_*`
    /// above were computed from real per-Gaussian data.
    ///
    /// `false` means those fields are all-zero placeholders — e.g. a
    /// safetensors file whose scale/opacity tensors use an unsupported
    /// dtype — and must be excluded from any similarity computation rather
    /// than compared as though they were genuine zero values (which would
    /// otherwise make two completely different degenerate-stats files
    /// score as a perfect match). `gaussian_count` and `sh_degree` are
    /// always reliable (they come from shape/header metadata alone) and are
    /// unaffected by this flag.
    pub stats_available: bool,
}

impl ModelStats {
    /// Compute statistics from a `.ply` or `.safetensors` file.
    ///
    /// Both `ascii` and `binary_little_endian` PLY files support full
    /// statistics. Safetensors files support full statistics when their
    /// `positions`/`scales`/`opacities` tensors are `F32`; see
    /// [`ModelStats::stats_available`] for what happens otherwise.
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

    /// Parse a PLY file (ascii or binary_little_endian) and compute statistics.
    fn from_ply(path: &Path) -> Result<Self> {
        use std::fs::File;

        let file = File::open(path).with_context(|| format!("Cannot open: {}", path.display()))?;
        let mut reader = BufReader::new(file);

        let header = parse_ply_header_info(&mut reader)
            .with_context(|| format!("Failed to parse PLY header: {}", path.display()))?;

        let sh_degree = sh_degree_from_rest(header.num_rest);
        let file_path = path.to_str().unwrap_or("<non-utf8>").to_string();

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
                stats_available: true,
            });
        }

        let vertex_data = if header.format.contains("binary_little_endian") {
            read_ply_vertex_data_binary(&mut reader, header.vertex_count, &header.properties)
                .with_context(|| format!("Failed to read vertex data: {}", path.display()))?
        } else if header.format.contains("ascii") {
            read_ply_vertex_data_ascii(&mut reader, header.vertex_count, &header.properties)
                .with_context(|| format!("Failed to read vertex data: {}", path.display()))?
        } else {
            anyhow::bail!(
                "Unsupported PLY format '{}' in {}: only 'ascii' and 'binary_little_endian' are supported",
                header.format,
                path.display()
            );
        };

        Ok(compute_model_stats(
            &vertex_data,
            &file_path,
            header.vertex_count,
            sh_degree,
        ))
    }

    /// Parse a safetensors file and compute available statistics.
    ///
    /// Gaussian count and SH degree are read from tensor shapes alone (this
    /// is always reliable, regardless of dtype). Distributional statistics
    /// (bbox, scale, opacity, position) additionally require the
    /// `positions`/`scales`/`opacities` tensors to be present and stored as
    /// `F32` — reinterpreting F16/BF16/integer bytes as f32 would silently
    /// produce garbage, so an unsupported dtype is treated the same as a
    /// missing tensor: `stats_available` is set to `false` rather than
    /// fabricating zeros that would otherwise be compared as genuine data.
    fn from_safetensors(path: &Path) -> Result<Self> {
        let file_path = path.to_str().unwrap_or("<non-utf8>").to_string();

        let bytes =
            std::fs::read(path).with_context(|| format!("Failed to read: {}", path.display()))?;

        let st = safetensors::SafeTensors::deserialize(&bytes)
            .with_context(|| format!("Failed to deserialize SafeTensors: {}", path.display()))?;

        let shape_of = |name: &str| -> Option<Vec<usize>> {
            st.tensor(name).ok().map(|tv| tv.shape().to_vec())
        };
        let f32_values = |name: &str| -> Option<Vec<f32>> {
            let tv = st.tensor(name).ok()?;
            if tv.dtype() != safetensors::tensor::Dtype::F32 {
                return None;
            }
            Some(bytemuck::cast_slice::<u8, f32>(tv.data()).to_vec())
        };

        // Gaussian count from the positions tensor's shape — reliable even
        // when the dtype isn't f32, since we only need N here.
        let mut gaussian_count = 0usize;
        let mut position_tensor_name: Option<&str> = None;
        for name in ["positions", "xyz", "pos"] {
            if let Some(shape) = shape_of(name) {
                if shape.len() == 2 && shape[1] == 3 {
                    gaussian_count = shape[0];
                    position_tensor_name = Some(name);
                    break;
                }
            }
        }

        // SH degree from the sh_coeffs tensor's total element count; shape
        // may be `[N, C]` or a flat `[N*C]` depending on the writer.
        let mut sh_degree = 0u32;
        if let Some(shape) = shape_of("sh_coeffs") {
            let total: usize = shape.iter().product();
            if gaussian_count > 0 && total.is_multiple_of(gaussian_count) {
                let channels = total / gaussian_count;
                sh_degree = sh_degree_from_rest(channels.saturating_sub(3));
            }
        }

        let positions = position_tensor_name.and_then(f32_values);
        let scale_values = f32_values("scales").or_else(|| f32_values("scale"));
        let opacity_values = f32_values("opacities").or_else(|| f32_values("opacity"));

        if let (Some(pos), Some(scale_values), Some(opacity_values)) =
            (positions, scale_values, opacity_values)
        {
            if gaussian_count > 0 && pos.len() == gaussian_count * 3 {
                let xs: Vec<f32> = pos.iter().step_by(3).copied().collect();
                let ys: Vec<f32> = pos.iter().skip(1).step_by(3).copied().collect();
                let zs: Vec<f32> = pos.iter().skip(2).step_by(3).copied().collect();

                let (pos_mean_x, pos_std_x) = mean_std(&xs);
                let (pos_mean_y, pos_std_y) = mean_std(&ys);
                let (pos_mean_z, pos_std_z) = mean_std(&zs);
                let bbox_min = [
                    xs.iter().cloned().fold(f32::INFINITY, f32::min),
                    ys.iter().cloned().fold(f32::INFINITY, f32::min),
                    zs.iter().cloned().fold(f32::INFINITY, f32::min),
                ];
                let bbox_max = [
                    xs.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
                    ys.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
                    zs.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
                ];
                let (opacity_mean, opacity_std) = mean_std(&opacity_values);
                let (scale_mean, scale_std) = mean_std(&scale_values);

                return Ok(Self {
                    file_path,
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
                    stats_available: true,
                });
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
            stats_available: false,
        })
    }
}

// ---------------------------------------------------------------------------
// Vertex reading (PLY: ascii and binary_little_endian)
// ---------------------------------------------------------------------------

/// Raw per-vertex data extracted from a PLY file.
struct VertexData {
    xs: Vec<f32>,
    ys: Vec<f32>,
    zs: Vec<f32>,
    opacities: Vec<f32>,
    scales: Vec<f32>, // interleaved: scale_x, scale_y, scale_z per vertex
}

/// Byte/column indices of the vertex properties this module extracts.
struct VertexFieldIndices {
    x: usize,
    y: usize,
    z: usize,
    opacity: usize,
    scale0: usize,
    scale1: usize,
    scale2: usize,
}

/// Locate the required vertex fields within the header's declared property
/// order (`x`, `y`, `z`, `opacity`, `scale_0..2`), so vertex data can be
/// read correctly regardless of property order, omitted normals, or extra
/// per-vertex properties — rather than assuming a fixed layout the header
/// is never actually checked against.
fn resolve_vertex_field_indices(properties: &[String]) -> Result<VertexFieldIndices> {
    let find = |name: &str| -> Result<usize> {
        properties.iter().position(|p| p == name).ok_or_else(|| {
            anyhow::anyhow!("PLY vertex element is missing required property '{name}'")
        })
    };
    Ok(VertexFieldIndices {
        x: find("x")?,
        y: find("y")?,
        z: find("z")?,
        opacity: find("opacity")?,
        scale0: find("scale_0")?,
        scale1: find("scale_1")?,
        scale2: find("scale_2")?,
    })
}

/// Initial reservation for one per-vertex `Vec`, derived from — but never
/// dictated by — an untrusted PLY header's `element vertex` count.
///
/// Nothing ties that number to the file's actual size, so a malformed or
/// hostile header (`element vertex 999999999999`) used to become five
/// enormous `Vec::with_capacity` requests before a single byte of body was
/// read. Rust aborts the process on allocation failure — it is not a
/// catchable `Result` — so that could crash a command whose whole job is to
/// safely inspect untrusted files, and `vertex_count * 3` for the
/// interleaved scale buffer could additionally overflow `usize` (a panic in
/// debug, a nonsense capacity in release) before any cap applied. Capping
/// first and multiplying afterwards fixes both: a genuinely larger file
/// still reads correctly, since `Vec::push` grows the buffer as needed.
///
/// The cap itself is shared with [`crate::info`], which reserves from the
/// same header field for the same reason.
fn initial_vertex_capacity(vertex_count: usize) -> usize {
    vertex_count.min(MAX_INITIAL_VERTEX_CAPACITY)
}

/// Read all vertex data from the body of a `binary_little_endian` PLY file.
///
/// Every vertex property is assumed to be a 4-byte `float` (the universal
/// convention for 3D Gaussian Splatting PLY exports, including this crate's
/// own writer). `properties` gives the exact per-vertex property order as
/// declared in the header, so this reads exactly as many floats per vertex
/// as the file actually contains.
fn read_ply_vertex_data_binary(
    reader: &mut impl Read,
    vertex_count: usize,
    properties: &[String],
) -> Result<VertexData> {
    let idx = resolve_vertex_field_indices(properties)?;
    let stride = properties.len();

    // Capped, not driven directly by the untrusted header count — see
    // `initial_vertex_capacity`.
    let capacity = initial_vertex_capacity(vertex_count);
    let mut xs = Vec::with_capacity(capacity);
    let mut ys = Vec::with_capacity(capacity);
    let mut zs = Vec::with_capacity(capacity);
    let mut opacities = Vec::with_capacity(capacity);
    let mut scales = Vec::with_capacity(capacity * 3);

    let mut record = vec![0f32; stride];
    let mut buf4 = [0u8; 4];

    for _ in 0..vertex_count {
        for slot in record.iter_mut() {
            reader
                .read_exact(&mut buf4)
                .context("EOF reading PLY vertex record")?;
            *slot = f32::from_le_bytes(buf4);
        }
        xs.push(record[idx.x]);
        ys.push(record[idx.y]);
        zs.push(record[idx.z]);
        opacities.push(record[idx.opacity]);
        scales.push(record[idx.scale0]);
        scales.push(record[idx.scale1]);
        scales.push(record[idx.scale2]);
    }

    Ok(VertexData {
        xs,
        ys,
        zs,
        opacities,
        scales,
    })
}

/// Read all vertex data from the body of an `ascii` PLY file.
///
/// Each vertex is one whitespace-separated line of decimal values, in the
/// same property order as `binary_little_endian`; fields are located by
/// name via [`resolve_vertex_field_indices`] exactly as in the binary path.
fn read_ply_vertex_data_ascii(
    reader: &mut impl BufRead,
    vertex_count: usize,
    properties: &[String],
) -> Result<VertexData> {
    let idx = resolve_vertex_field_indices(properties)?;
    let expected_fields = properties.len();

    // Capped, not driven directly by the untrusted header count — see
    // `initial_vertex_capacity`.
    let capacity = initial_vertex_capacity(vertex_count);
    let mut xs = Vec::with_capacity(capacity);
    let mut ys = Vec::with_capacity(capacity);
    let mut zs = Vec::with_capacity(capacity);
    let mut opacities = Vec::with_capacity(capacity);
    let mut scales = Vec::with_capacity(capacity * 3);

    let mut line = String::new();
    for row in 0..vertex_count {
        line.clear();
        let bytes_read = reader
            .read_line(&mut line)
            .with_context(|| format!("I/O error reading ASCII PLY vertex {row}"))?;
        if bytes_read == 0 {
            anyhow::bail!("Unexpected EOF reading ASCII PLY vertex {row}/{vertex_count}");
        }
        let fields: Vec<&str> = line.split_ascii_whitespace().collect();
        if fields.len() < expected_fields {
            anyhow::bail!(
                "ASCII PLY vertex {row} has {} fields, expected {expected_fields}",
                fields.len()
            );
        }
        let parse_field = |i: usize| -> Result<f32> {
            fields[i]
                .parse::<f32>()
                .with_context(|| format!("Invalid float '{}' in ASCII PLY vertex {row}", fields[i]))
        };
        xs.push(parse_field(idx.x)?);
        ys.push(parse_field(idx.y)?);
        zs.push(parse_field(idx.z)?);
        opacities.push(parse_field(idx.opacity)?);
        scales.push(parse_field(idx.scale0)?);
        scales.push(parse_field(idx.scale1)?);
        scales.push(parse_field(idx.scale2)?);
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
        stats_available: true,
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
        // At least one box is degenerate (zero volume on some axis) — IoU
        // itself is undefined here (0/0), so fall back to a distance-based
        // similarity between the box centers instead of blindly reporting a
        // perfect match. Two coincident degenerate boxes (e.g. two
        // single-point clouds at the same location) still score 1.0; two
        // degenerate boxes far apart (e.g. single points at (0,0,0) and
        // (100,100,100), which previously both scored a perfect 1.0) decay
        // smoothly toward 0 instead.
        let center_a = [
            (min_a[0] + max_a[0]) * 0.5,
            (min_a[1] + max_a[1]) * 0.5,
            (min_a[2] + max_a[2]) * 0.5,
        ];
        let center_b = [
            (min_b[0] + max_b[0]) * 0.5,
            (min_b[1] + max_b[1]) * 0.5,
            (min_b[2] + max_b[2]) * 0.5,
        ];
        let dx = center_a[0] - center_b[0];
        let dy = center_a[1] - center_b[1];
        let dz = center_a[2] - center_b[2];
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();
        (-dist).exp()
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
    /// When [`ComparisonReport::stats_available`] is `true`, a weighted
    /// combination of:
    /// - bbox IoU (40 %)
    /// - Gaussian count similarity (30 %)
    /// - SH degree match (10 %)
    /// - Scale similarity (10 %)
    /// - Opacity similarity (10 %)
    ///
    /// When `false`, only Gaussian count (30 %) and SH degree (10 %) are
    /// available, renormalized to sum to 1.0 — see
    /// [`ComparisonReport::stats_available`].
    pub overall_similarity: f32,

    /// Similarity threshold used for the recommendation.
    pub threshold: f64,

    /// `true` only when both models' distributional statistics (bounding
    /// box, scale, opacity — see [`ModelStats::stats_available`]) were
    /// available. When `false`, `overall_similarity` reflects only
    /// Gaussian count and SH degree, and `bbox_iou` /
    /// `scale_mean_diff_pct` / `opacity_mean_diff_pct` compare all-zero
    /// placeholders for at least one model — they are still computed (so
    /// callers inspecting individual fields see the same shape of report
    /// either way) but are not meaningful and must not be read as real
    /// measurements.
    pub stats_available: bool,
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
        let stats_available = a.stats_available && b.stats_available;

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

        // Weighted similarity. When distributional stats aren't available
        // for at least one model (e.g. an unsupported safetensors dtype),
        // bbox/scale/opacity are all-zero placeholders on both sides —
        // comparing them would score a spurious perfect match on 60% of
        // the weight, so renormalize over just the metadata-only
        // components (count + SH degree) instead.
        let overall_similarity = if stats_available {
            (0.40 * iou + 0.30 * count_sim + 0.10 * sh_sim + 0.10 * scale_sim + 0.10 * opacity_sim)
                .clamp(0.0, 1.0)
        } else {
            ((0.30 * count_sim + 0.10 * sh_sim) / 0.40).clamp(0.0, 1.0)
        };

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
            stats_available,
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
        if !self.stats_available {
            out.push_str(
                "⚠ Partial comparison: bounding box / scale / opacity statistics are \
                 unavailable for one or both models (unsupported format or dtype). \
                 Overall similarity below reflects only Gaussian count and SH degree.\n\n",
            );
        }
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
                "stats_available": a.stats_available,
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
                "stats_available": b.stats_available,
            },
            "gaussian_count_diff": self.gaussian_count_diff,
            "gaussian_count_pct": self.gaussian_count_pct,
            "sh_degree_same": self.sh_degree_same,
            "bbox_iou": self.bbox_iou,
            "scale_mean_diff_pct": self.scale_mean_diff_pct,
            "opacity_mean_diff_pct": self.opacity_mean_diff_pct,
            "overall_similarity": self.overall_similarity,
            "threshold": self.threshold,
            "stats_available": self.stats_available,
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
    if !(0.0..=1.0).contains(&threshold) {
        anyhow::bail!("--threshold must be between 0.0 and 1.0 (inclusive), got {threshold}");
    }

    // `CompareArgs::format` is a bare `String` (not a `ValueEnum`), so an
    // unrecognized value must be rejected here rather than silently falling
    // through to the text-output default — a typo like `--format jsom`
    // should not exit 0 with unparseable-as-JSON output.
    let format = args.format.to_lowercase();

    let stats_a = ModelStats::from_file(&args.model1)
        .with_context(|| format!("Failed to load model A: {}", args.model1.display()))?;
    let stats_b = ModelStats::from_file(&args.model2)
        .with_context(|| format!("Failed to load model B: {}", args.model2.display()))?;

    let report = ComparisonReport::compute(stats_a, stats_b, threshold);

    match format.as_str() {
        "json" => {
            let json = report.format_json()?;
            println!("{}", json);
        }
        "text" => {
            print!("{}", report.format_text());
        }
        other => {
            anyhow::bail!("Unknown --format value '{other}': expected 'text' or 'json'");
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

    /// Write an `ascii`-format PLY with the same property layout and the
    /// same per-vertex values as [`write_test_ply`], so the two can be
    /// compared for equivalent stats.
    fn write_test_ply_ascii(
        path: &Path,
        n_vertices: usize,
        num_rest: usize,
        position_offset: f32,
        opacity_override: Option<f32>,
        scale_override: Option<f32>,
    ) -> Result<()> {
        let mut f = fs::File::create(path)?;

        writeln!(f, "ply")?;
        writeln!(f, "format ascii 1.0")?;
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

        let opacity = opacity_override.unwrap_or(0.5);
        let scale = scale_override.unwrap_or(0.01);
        for i in 0..n_vertices {
            let t = i as f32 / n_vertices.max(1) as f32;
            let mut fields: Vec<f32> = vec![
                position_offset + t,
                position_offset + t * 0.5,
                position_offset + t * 0.25,
                0.0,
                0.0,
                1.0, // normal
                0.5,
                0.5,
                0.5, // f_dc
            ];
            fields.extend(std::iter::repeat_n(0.0f32, num_rest));
            fields.push(opacity);
            fields.extend([scale, scale, scale]);
            fields.extend([1.0f32, 0.0, 0.0, 0.0]); // rotation

            let line = fields
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(" ");
            writeln!(f, "{line}")?;
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

    #[test]
    fn test_bbox_iou_coincident_degenerate_boxes_are_identical() {
        let p = [0.0f32, 0.0, 0.0];
        let iou = bbox_iou(p, p, p, p);
        assert!(
            (iou - 1.0).abs() < 1e-5,
            "two coincident zero-volume boxes should score 1.0, got {}",
            iou
        );
    }

    #[test]
    fn test_bbox_iou_distant_degenerate_boxes_are_not_identical() {
        // Regression test: two single-point clouds at (0,0,0) and
        // (100,100,100) both have zero-volume bounding boxes, and the old
        // "both degenerate => 1.0" rule reported them as a perfect overlap
        // — the dominant (40%) term behind the "essentially identical"
        // false positive for two totally different point clouds.
        let a_pos = [0.0f32, 0.0, 0.0];
        let b_pos = [100.0f32, 100.0, 100.0];
        let iou = bbox_iou(a_pos, a_pos, b_pos, b_pos);
        assert!(
            iou < 0.01,
            "two degenerate boxes 100 units apart must not score as identical, got {}",
            iou
        );
    }

    #[test]
    fn test_bbox_iou_degenerate_output_always_in_range() {
        let a_pos = [1.0f32, -2.0, 3.0];
        let b_pos = [1.5f32, -2.0, 3.0];
        let iou = bbox_iou(a_pos, a_pos, b_pos, b_pos);
        assert!(
            (0.0..=1.0).contains(&iou),
            "degenerate-box IoU must stay in [0, 1], got {}",
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

    #[test]
    fn test_model_stats_from_ascii_ply_has_real_stats() -> Result<()> {
        // Regression test: ASCII PLY used to return `stats_available`-less
        // all-zero "stub stats". It must now report real, non-trivial
        // bounding box / scale / opacity statistics, same as binary.
        let path = env::temp_dir().join("test_compare_ascii_real_stats.ply");
        write_test_ply_ascii(&path, 10, 45, 5.0, Some(0.7), Some(0.02))?;

        let stats = ModelStats::from_file(&path)?;
        assert!(stats.stats_available);
        assert_eq!(stats.gaussian_count, 10);
        assert_eq!(stats.sh_degree, 3);
        assert!(
            (stats.opacity_mean - 0.7).abs() < 1e-4,
            "opacity_mean should reflect real data, got {}",
            stats.opacity_mean
        );
        assert!(
            (stats.scale_mean - 0.02).abs() < 1e-4,
            "scale_mean should reflect real data, got {}",
            stats.scale_mean
        );
        assert!(
            stats.bbox_min != [0.0; 3] || stats.bbox_max != [0.0; 3],
            "bounding box must not be the all-zero placeholder"
        );

        fs::remove_file(&path).ok();
        Ok(())
    }

    #[test]
    fn test_model_stats_ascii_and_binary_ply_agree() -> Result<()> {
        let ascii_path = env::temp_dir().join("test_compare_parity_ascii.ply");
        let binary_path = env::temp_dir().join("test_compare_parity_binary.ply");
        write_test_ply_ascii(&ascii_path, 8, 9, 2.0, Some(0.6), Some(0.03))?;
        write_test_ply(&binary_path, 8, 9, 2.0, Some(0.6), Some(0.03))?;

        let ascii_stats = ModelStats::from_file(&ascii_path)?;
        let binary_stats = ModelStats::from_file(&binary_path)?;

        assert_eq!(ascii_stats.gaussian_count, binary_stats.gaussian_count);
        assert_eq!(ascii_stats.sh_degree, binary_stats.sh_degree);
        assert!((ascii_stats.opacity_mean - binary_stats.opacity_mean).abs() < 1e-4);
        assert!((ascii_stats.scale_mean - binary_stats.scale_mean).abs() < 1e-4);
        for i in 0..3 {
            assert!((ascii_stats.bbox_min[i] - binary_stats.bbox_min[i]).abs() < 1e-4);
            assert!((ascii_stats.bbox_max[i] - binary_stats.bbox_max[i]).abs() < 1e-4);
        }

        fs::remove_file(&ascii_path).ok();
        fs::remove_file(&binary_path).ok();
        Ok(())
    }

    #[test]
    fn test_model_stats_from_ply_missing_required_property_errors() -> Result<()> {
        // A PLY that omits `scale_1` (unusual, but the header property list
        // is what must be trusted, not an assumed fixed layout) must fail
        // with a descriptive error rather than misreading unrelated bytes
        // as scale_1.
        let path = env::temp_dir().join("test_compare_missing_property.ply");
        let mut f = fs::File::create(&path)?;
        writeln!(f, "ply")?;
        writeln!(f, "format ascii 1.0")?;
        writeln!(f, "element vertex 1")?;
        writeln!(f, "property float x")?;
        writeln!(f, "property float y")?;
        writeln!(f, "property float z")?;
        writeln!(f, "property float opacity")?;
        writeln!(f, "property float scale_0")?;
        // scale_1 and scale_2 intentionally omitted
        writeln!(f, "end_header")?;
        writeln!(f, "1.0 2.0 3.0 0.5 0.1")?;
        drop(f);

        let result = ModelStats::from_file(&path);
        assert!(
            result.is_err(),
            "missing required property must be a hard error"
        );

        fs::remove_file(&path).ok();
        Ok(())
    }

    #[test]
    fn test_model_stats_from_safetensors_f32_has_real_stats() -> Result<()> {
        use safetensors::tensor::{Dtype, TensorView};

        let path = env::temp_dir().join("test_compare_safetensors_f32.safetensors");
        let n = 4usize;
        let positions: Vec<f32> = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let scales: Vec<f32> = vec![0.02; n * 3];
        let opacities: Vec<f32> = vec![0.8; n];

        let pos_view = TensorView::new(Dtype::F32, vec![n, 3], bytemuck::cast_slice(&positions))
            .map_err(|e| anyhow::anyhow!("tensor view error: {e}"))?;
        let scl_view = TensorView::new(Dtype::F32, vec![n, 3], bytemuck::cast_slice(&scales))
            .map_err(|e| anyhow::anyhow!("tensor view error: {e}"))?;
        let opa_view = TensorView::new(Dtype::F32, vec![n], bytemuck::cast_slice(&opacities))
            .map_err(|e| anyhow::anyhow!("tensor view error: {e}"))?;
        let tensors: Vec<(&str, TensorView<'_>)> = vec![
            ("positions", pos_view),
            ("scales", scl_view),
            ("opacities", opa_view),
        ];
        let data = safetensors::serialize(tensors, None)
            .map_err(|e| anyhow::anyhow!("serialize error: {e}"))?;
        fs::write(&path, &data)?;

        let stats = ModelStats::from_file(&path)?;
        assert!(stats.stats_available);
        assert_eq!(stats.gaussian_count, 4);
        assert!((stats.opacity_mean - 0.8).abs() < 1e-4);
        assert!((stats.scale_mean - 0.02).abs() < 1e-4);
        assert!(
            stats.bbox_max[0] > stats.bbox_min[0],
            "bounding box should span the real position data"
        );

        fs::remove_file(&path).ok();
        Ok(())
    }

    #[test]
    fn test_model_stats_from_safetensors_unsupported_dtype_marks_unavailable() -> Result<()> {
        use safetensors::tensor::{Dtype, TensorView};

        let path = env::temp_dir().join("test_compare_safetensors_f16.safetensors");
        let n = 4usize;
        let positions: Vec<f32> = vec![0.0; n * 3];
        // 2 bytes/element f16 payload — content is irrelevant, only the
        // dtype tag matters: it must never be reinterpreted as f32.
        let scales_f16_bytes = vec![0u8; n * 3 * 2];
        let opacities: Vec<f32> = vec![0.5; n];

        let pos_view = TensorView::new(Dtype::F32, vec![n, 3], bytemuck::cast_slice(&positions))
            .map_err(|e| anyhow::anyhow!("tensor view error: {e}"))?;
        let scl_view = TensorView::new(Dtype::F16, vec![n, 3], &scales_f16_bytes)
            .map_err(|e| anyhow::anyhow!("tensor view error: {e}"))?;
        let opa_view = TensorView::new(Dtype::F32, vec![n], bytemuck::cast_slice(&opacities))
            .map_err(|e| anyhow::anyhow!("tensor view error: {e}"))?;
        let tensors: Vec<(&str, TensorView<'_>)> = vec![
            ("positions", pos_view),
            ("scales", scl_view),
            ("opacities", opa_view),
        ];
        let data = safetensors::serialize(tensors, None)
            .map_err(|e| anyhow::anyhow!("serialize error: {e}"))?;
        fs::write(&path, &data)?;

        let stats = ModelStats::from_file(&path)?;
        assert!(
            !stats.stats_available,
            "an F16 scales tensor must not be silently reinterpreted as F32"
        );
        // Gaussian count is still reliable — it only needed the shape.
        assert_eq!(stats.gaussian_count, 4);
        assert_eq!(stats.bbox_min, [0.0; 3]);
        assert_eq!(stats.bbox_max, [0.0; 3]);

        fs::remove_file(&path).ok();
        Ok(())
    }

    #[test]
    fn test_compare_two_different_ascii_ply_files_are_not_falsely_identical() -> Result<()> {
        // Direct regression test for the critical finding: two clearly
        // different ASCII PLY models (different positions, opacity, scale
        // — only the Gaussian count matches) must not be reported as
        // "essentially identical". Before the fix, ASCII PLY always
        // produced all-zero stub stats, `bbox_iou` treated two degenerate
        // zero-volume boxes as identical, and the zero-mean guards made
        // scale/opacity diffs also read as identical — overall_similarity
        // was exactly 1.0 for any two same-count ASCII PLY files.
        let path_a = env::temp_dir().join("test_compare_diff_ascii_a.ply");
        let path_b = env::temp_dir().join("test_compare_diff_ascii_b.ply");
        write_test_ply_ascii(&path_a, 20, 9, 0.0, Some(0.9), Some(0.005))?;
        write_test_ply_ascii(&path_b, 20, 9, 50.0, Some(0.1), Some(0.5))?;

        let stats_a = ModelStats::from_file(&path_a)?;
        let stats_b = ModelStats::from_file(&path_b)?;
        assert!(stats_a.stats_available && stats_b.stats_available);

        let report = ComparisonReport::compute(stats_a, stats_b, 0.8);
        assert!(report.stats_available);
        assert!(
            report.overall_similarity < 0.9,
            "clearly different models must not score as essentially identical, got {}",
            report.overall_similarity
        );
        assert!(
            !report.recommendation().contains("essentially identical"),
            "recommendation should not claim identity: {}",
            report.recommendation()
        );

        fs::remove_file(&path_a).ok();
        fs::remove_file(&path_b).ok();
        Ok(())
    }

    #[test]
    fn test_comparison_report_stats_unavailable_renormalizes_weights() {
        // Two models whose distributional stats are unavailable (as
        // `from_safetensors` reports for an unsupported dtype) but with
        // identical Gaussian count and SH degree: overall_similarity must
        // come out to 1.0 via the renormalized (count + SH degree only)
        // formula, not via a spurious bbox/scale/opacity "match" on
        // all-zero placeholders.
        let unavailable = |file_path: &str| ModelStats {
            file_path: file_path.to_string(),
            gaussian_count: 1000,
            sh_degree: 2,
            bbox_min: [0.0; 3],
            bbox_max: [0.0; 3],
            scale_mean: 0.0,
            scale_std: 0.0,
            opacity_mean: 0.0,
            opacity_std: 0.0,
            position_mean: [0.0; 3],
            position_std: [0.0; 3],
            stats_available: false,
        };
        let report = ComparisonReport::compute(unavailable("a"), unavailable("b"), 0.8);
        assert!(!report.stats_available);
        assert!(
            approx_eq_f32(report.overall_similarity, 1.0, 1e-4),
            "renormalized similarity for matching count+SH should be 1.0, got {}",
            report.overall_similarity
        );
        assert!(report.format_text().contains("Partial comparison"));
    }

    fn approx_eq_f32(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol
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

    /// Regression: this used to write `test_compare_range_{a,b}.ply` at
    /// fixed names directly under `env::temp_dir()`, which every concurrent
    /// invocation of this test (a parallel `cargo test`/`nextest` run, or
    /// simply this test racing another one that happens to touch the same
    /// name) shares — one run's `write_test_ply` could truncate a file
    /// another run had just opened for reading. A per-test `tempfile`
    /// directory gives each invocation its own directory, so no name
    /// collides across processes; its `Drop` impl also removes it, which is
    /// why the explicit `fs::remove_file` cleanup this file's other tests
    /// still do is unnecessary here.
    #[test]
    fn test_comparison_report_similarity_in_range() -> Result<()> {
        let tmp_dir = tempfile::tempdir().context("create temp dir")?;
        let path_a = tmp_dir.path().join("test_compare_range_a.ply");
        let path_b = tmp_dir.path().join("test_compare_range_b.ply");

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
            stats_available: true,
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
            stats_available: true,
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

    // -----------------------------------------------------------------------
    // run_compare validation
    // -----------------------------------------------------------------------

    #[test]
    fn test_run_compare_rejects_out_of_range_threshold() {
        // Threshold is validated before any file I/O, so nonexistent paths
        // are fine here — the point is that a bad --threshold value is a
        // real error, not silently clamped or ignored.
        let args = CompareArgs {
            model1: env::temp_dir().join("does_not_exist_a.ply"),
            model2: env::temp_dir().join("does_not_exist_b.ply"),
            format: "text".to_string(),
            threshold: 5.0,
        };
        let result = run_compare(args);
        assert!(result.is_err());
        let msg = format!("{:#}", result.unwrap_err());
        assert!(
            msg.contains("threshold"),
            "error should mention threshold: {msg}"
        );
    }

    #[test]
    fn test_run_compare_rejects_unknown_format() -> Result<()> {
        let path_a = env::temp_dir().join("test_compare_run_format_a.ply");
        let path_b = env::temp_dir().join("test_compare_run_format_b.ply");
        write_test_ply(&path_a, 3, 0, 0.0, None, None)?;
        write_test_ply(&path_b, 3, 0, 0.0, None, None)?;

        let args = CompareArgs {
            model1: path_a.clone(),
            model2: path_b.clone(),
            format: "jsom".to_string(), // typo for "json"
            threshold: 0.8,
        };
        let result = run_compare(args);
        assert!(
            result.is_err(),
            "an unrecognized --format value must be a hard error, not a silent text fallback"
        );

        fs::remove_file(&path_a).ok();
        fs::remove_file(&path_b).ok();
        Ok(())
    }

    #[test]
    fn test_run_compare_accepts_text_and_json() -> Result<()> {
        let path_a = env::temp_dir().join("test_compare_run_valid_a.ply");
        let path_b = env::temp_dir().join("test_compare_run_valid_b.ply");
        write_test_ply(&path_a, 3, 0, 0.0, None, None)?;
        write_test_ply(&path_b, 3, 0, 0.0, None, None)?;

        for fmt in ["text", "TEXT", "json", "JSON"] {
            let args = CompareArgs {
                model1: path_a.clone(),
                model2: path_b.clone(),
                format: fmt.to_string(),
                threshold: 0.8,
            };
            run_compare(args).with_context(|| format!("format '{fmt}' should be accepted"))?;
        }

        fs::remove_file(&path_a).ok();
        fs::remove_file(&path_b).ok();
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Untrusted `element vertex` counts must not drive the reservation.
    // -----------------------------------------------------------------------

    /// The seven properties both PLY readers require, in canonical order.
    fn required_properties() -> Vec<String> {
        ["x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2"]
            .iter()
            .map(|name| (*name).to_string())
            .collect()
    }

    #[test]
    fn initial_vertex_capacity_is_capped_but_exact_below_the_cap() {
        // Ordinary files keep their exact one-shot reservation …
        assert_eq!(initial_vertex_capacity(0), 0);
        assert_eq!(initial_vertex_capacity(52_341), 52_341);
        // … and anything above the cap is clamped, so `capacity * 3` for the
        // interleaved scale buffer cannot overflow `usize` either.
        assert_eq!(
            initial_vertex_capacity(MAX_INITIAL_VERTEX_CAPACITY + 1),
            MAX_INITIAL_VERTEX_CAPACITY
        );
        assert_eq!(
            initial_vertex_capacity(usize::MAX),
            MAX_INITIAL_VERTEX_CAPACITY
        );
        assert!(
            MAX_INITIAL_VERTEX_CAPACITY.checked_mul(3).is_some(),
            "the cap itself must leave room for the *3 scale reservation"
        );
    }

    #[test]
    fn read_ply_vertex_data_binary_huge_untrusted_count_does_not_abort() {
        // Regression: `element vertex 999999999999` used to reach
        // `Vec::with_capacity(vertex_count)` (and `* 3`) directly, asking
        // for terabytes before a single body byte was read. Rust aborts the
        // process on allocation failure rather than returning a catchable
        // `Result`, so a malformed file could crash the command. With the
        // reservation capped this must instead fail fast and cleanly on EOF,
        // since the "body" here is empty.
        let properties = required_properties();
        let result =
            read_ply_vertex_data_binary(&mut std::io::empty(), 999_999_999_999, &properties);
        assert!(result.is_err(), "expected a clean EOF error, not a crash");
    }

    #[test]
    fn read_ply_vertex_data_ascii_huge_untrusted_count_does_not_abort() {
        // Same regression on the ASCII path, which reserved from the very
        // same untrusted header field.
        let properties = required_properties();
        let mut empty_body = BufReader::new(std::io::empty());
        let result = read_ply_vertex_data_ascii(&mut empty_body, 999_999_999_999, &properties);
        let msg = result.err().map(|e| format!("{e:#}")).unwrap_or_default();
        assert!(
            msg.contains("EOF"),
            "expected a clean EOF error, not a crash: {msg}"
        );
    }

    #[test]
    fn read_ply_vertex_data_still_reads_real_files_after_the_cap() -> Result<()> {
        // The cap is an allocation hint only: a real file is still read in
        // full, value for value.
        let path = env::temp_dir().join("test_compare_capped_reservation.ply");
        write_test_ply(&path, 4, 0, 0.0, Some(0.5), Some(0.25))?;
        let stats = ModelStats::from_file(&path)?;
        fs::remove_file(&path).ok();
        assert_eq!(stats.gaussian_count, 4);
        assert!(stats.stats_available);
        assert!(
            (stats.opacity_mean - 0.5).abs() < 1e-4,
            "opacity_mean: {}",
            stats.opacity_mean
        );
        assert!(
            (stats.scale_mean - 0.25).abs() < 1e-4,
            "scale_mean: {}",
            stats.scale_mean
        );
        Ok(())
    }
}
