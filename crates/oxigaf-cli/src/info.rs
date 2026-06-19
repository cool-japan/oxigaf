//! `oxigaf info` command — inspect model files and display their metadata.
//!
//! Supports:
//! - `.ply` — PLY header parse: Gaussian count, SH degree, property list,
//!   file size, bounding box, opacity/scale stats.
//! - `.safetensors` — tensor names, shapes, dtypes, metadata dict, file size.
//! - `.json` — training config or checkpoint metadata, key fields.

use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

use anyhow::{Context, Result};

use crate::cli::InfoArgs;

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Run the `info` command.
pub fn run_info(args: InfoArgs) -> Result<()> {
    let path = &args.path;

    if !path.exists() {
        anyhow::bail!("File not found: {}", path.display());
    }

    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    match ext.as_str() {
        "ply" => inspect_ply(path),
        "safetensors" => inspect_safetensors(path),
        "json" => inspect_json(path),
        other => anyhow::bail!(
            "Unsupported file extension '.{}'. Supported: .ply, .safetensors, .json",
            other
        ),
    }
}

// ---------------------------------------------------------------------------
// PLY inspection
// ---------------------------------------------------------------------------

/// Lightweight parsed PLY header.
pub(crate) struct PlyHeaderInfo {
    pub(crate) vertex_count: usize,
    pub(crate) format: String,
    pub(crate) properties: Vec<String>,
    pub(crate) num_rest: usize,
}

/// Parse only the ASCII header of a PLY file.
pub(crate) fn parse_ply_header_info(reader: &mut impl BufRead) -> Result<PlyHeaderInfo> {
    let mut line = String::new();

    // First line must be "ply"
    line.clear();
    reader
        .read_line(&mut line)
        .context("Failed to read PLY header")?;
    if line.trim() != "ply" {
        anyhow::bail!("Not a PLY file (first line: {:?})", line.trim());
    }

    let mut vertex_count: Option<usize> = None;
    let mut format = String::from("unknown");
    let mut properties: Vec<String> = Vec::new();
    let mut num_rest: usize = 0;
    let mut in_vertex_element = false;

    loop {
        line.clear();
        let bytes = reader.read_line(&mut line).context("Header read error")?;
        if bytes == 0 {
            anyhow::bail!("Unexpected EOF in PLY header");
        }
        let trimmed = line.trim();

        if trimmed == "end_header" {
            break;
        } else if trimmed.starts_with("format ") {
            let parts: Vec<&str> = trimmed.splitn(3, ' ').collect();
            if parts.len() >= 2 {
                format = parts[1..].join(" ");
            }
        } else if trimmed.starts_with("element vertex ") {
            in_vertex_element = true;
            let count_str = trimmed.strip_prefix("element vertex ").unwrap_or("").trim();
            vertex_count = Some(
                count_str
                    .parse::<usize>()
                    .with_context(|| format!("Invalid vertex count '{count_str}'"))?,
            );
        } else if trimmed.starts_with("element ") {
            // Once we encounter another element (not vertex), stop collecting vertex properties.
            in_vertex_element = false;
        } else if trimmed.starts_with("property ") && in_vertex_element {
            let parts: Vec<&str> = trimmed.splitn(3, ' ').collect();
            if parts.len() >= 3 {
                let prop_name = parts[2].to_string();
                if prop_name.starts_with("f_rest_") {
                    num_rest += 1;
                }
                properties.push(prop_name);
            }
        }
    }

    let vertex_count = vertex_count.unwrap_or(0);

    Ok(PlyHeaderInfo {
        vertex_count,
        format,
        properties,
        num_rest,
    })
}

/// Derive the SH degree from the number of f_rest properties.
///
/// SH coefficients layout: (sh_degree+1)^2 * 3 total, where 3 come from f_dc_0..2.
pub(crate) fn sh_degree_from_rest(num_rest: usize) -> u32 {
    // total_sh = num_rest + 3 (3 dc components)
    let total_sh = num_rest + 3;
    // coefficients per channel = total_sh / 3
    if !total_sh.is_multiple_of(3) {
        return 0;
    }
    let coeffs_per_channel = total_sh / 3;
    // find n such that n^2 == coeffs_per_channel
    let sqrt = (coeffs_per_channel as f64).sqrt().round() as u32;
    if (sqrt as usize) * (sqrt as usize) == coeffs_per_channel {
        sqrt.saturating_sub(1)
    } else {
        0
    }
}

/// Read vertex data for a binary_little_endian PLY and compute stats.
///
/// This reads only x, y, z, then fast-skips f_dc (3 floats), f_rest (num_rest floats),
/// then reads opacity, scale_0..2, and skips rot (4 floats) per vertex.
fn read_ply_stats(
    reader: &mut impl Read,
    vertex_count: usize,
    num_rest: usize,
) -> Result<PlyStats> {
    // Property layout (all f32, 4 bytes each):
    // x, y, z, nx, ny, nz, f_dc_0, f_dc_1, f_dc_2,
    // f_rest_0..f_rest_{num_rest-1},
    // opacity, scale_0, scale_1, scale_2, rot_0, rot_1, rot_2, rot_3

    let mut xs: Vec<f32> = Vec::with_capacity(vertex_count);
    let mut ys: Vec<f32> = Vec::with_capacity(vertex_count);
    let mut zs: Vec<f32> = Vec::with_capacity(vertex_count);
    let mut opacities: Vec<f32> = Vec::with_capacity(vertex_count);
    let mut scales_x: Vec<f32> = Vec::with_capacity(vertex_count);
    let mut scales_y: Vec<f32> = Vec::with_capacity(vertex_count);
    let mut scales_z: Vec<f32> = Vec::with_capacity(vertex_count);

    // Bytes to skip between normals-end and opacity:
    // nx, ny, nz (3) + f_dc_0..2 (3) + f_rest (num_rest) = (6 + num_rest) floats
    let skip_after_position = 6 + num_rest;
    // After opacity, we have scale_0, scale_1, scale_2 (3 floats we read), then rot_0..3 (4 floats skip)

    let mut buf4 = [0u8; 4];
    let mut skip_buf: Vec<u8> = vec![0u8; skip_after_position * 4];

    let read_f32 = |reader: &mut dyn Read, buf: &mut [u8; 4]| -> Result<f32> {
        reader
            .read_exact(buf)
            .context("Unexpected EOF in vertex data")?;
        Ok(f32::from_le_bytes(*buf))
    };

    // We need to pass reader as dyn Read for the closure, so use a different approach.
    // Read each vertex field-by-field.
    for _ in 0..vertex_count {
        // x, y, z
        reader.read_exact(&mut buf4).context("EOF reading x")?;
        let x = f32::from_le_bytes(buf4);
        reader.read_exact(&mut buf4).context("EOF reading y")?;
        let y = f32::from_le_bytes(buf4);
        reader.read_exact(&mut buf4).context("EOF reading z")?;
        let z = f32::from_le_bytes(buf4);
        xs.push(x);
        ys.push(y);
        zs.push(z);

        // skip: nx, ny, nz, f_dc_0, f_dc_1, f_dc_2, f_rest_0..{num_rest-1}
        reader
            .read_exact(&mut skip_buf)
            .context("EOF skipping normals/sh")?;

        // opacity
        reader
            .read_exact(&mut buf4)
            .context("EOF reading opacity")?;
        let opacity = f32::from_le_bytes(buf4);
        opacities.push(opacity);

        // scale_0, scale_1, scale_2
        reader
            .read_exact(&mut buf4)
            .context("EOF reading scale_0")?;
        let sx = f32::from_le_bytes(buf4);
        reader
            .read_exact(&mut buf4)
            .context("EOF reading scale_1")?;
        let sy = f32::from_le_bytes(buf4);
        reader
            .read_exact(&mut buf4)
            .context("EOF reading scale_2")?;
        let sz = f32::from_le_bytes(buf4);
        scales_x.push(sx);
        scales_y.push(sy);
        scales_z.push(sz);

        // skip rot_0, rot_1, rot_2, rot_3
        let mut rot_buf = [0u8; 16];
        reader
            .read_exact(&mut rot_buf)
            .context("EOF reading rotations")?;
    }

    // Suppress the unused warning on read_f32 — just call it to satisfy the borrow checker.
    let _ = read_f32;

    let bbox_x = (
        xs.iter().cloned().fold(f32::INFINITY, f32::min),
        xs.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
    );
    let bbox_y = (
        ys.iter().cloned().fold(f32::INFINITY, f32::min),
        ys.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
    );
    let bbox_z = (
        zs.iter().cloned().fold(f32::INFINITY, f32::min),
        zs.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
    );

    let opacity_stats = compute_stats(&opacities);
    let scale_all: Vec<f32> = scales_x
        .iter()
        .chain(scales_y.iter())
        .chain(scales_z.iter())
        .cloned()
        .collect();
    let scale_stats = compute_stats(&scale_all);

    Ok(PlyStats {
        bbox_x,
        bbox_y,
        bbox_z,
        opacity_min: opacity_stats.0,
        opacity_max: opacity_stats.1,
        opacity_mean: opacity_stats.2,
        scale_min: scale_stats.0,
        scale_max: scale_stats.1,
        scale_mean: scale_stats.2,
    })
}

struct PlyStats {
    bbox_x: (f32, f32),
    bbox_y: (f32, f32),
    bbox_z: (f32, f32),
    opacity_min: f32,
    opacity_max: f32,
    opacity_mean: f32,
    scale_min: f32,
    scale_max: f32,
    scale_mean: f32,
}

/// Compute (min, max, mean) for a slice of f32 values.
pub(crate) fn compute_stats(values: &[f32]) -> (f32, f32, f32) {
    if values.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let min = values.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum: f64 = values.iter().map(|v| *v as f64).sum();
    let mean = (sum / values.len() as f64) as f32;
    (min, max, mean)
}

/// Inspect a `.ply` file and print metadata.
fn inspect_ply(path: &Path) -> Result<()> {
    let file_size_bytes = std::fs::metadata(path)
        .with_context(|| format!("Cannot stat: {}", path.display()))?
        .len();
    let file_size_mb = file_size_bytes as f64 / (1024.0 * 1024.0);

    let file = File::open(path).with_context(|| format!("Cannot open: {}", path.display()))?;
    let mut reader = BufReader::new(file);

    let header = parse_ply_header_info(&mut reader)
        .with_context(|| format!("Failed to parse PLY header in: {}", path.display()))?;

    let sh_degree = sh_degree_from_rest(header.num_rest);
    let prop_list = header.properties.join(", ");

    let file_name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("<unknown>");

    println!("File: {}", file_name);
    println!("Type: PLY (3D Gaussian Splat)");
    println!("Gaussians: {:}", header.vertex_count);
    println!("SH Degree: {}", sh_degree);
    println!("Properties: {}", prop_list);
    println!("File size: {:.1} MB", file_size_mb);

    // Only read stats if the format is binary_little_endian (ASCII is not currently supported for stats)
    if header.format.contains("binary_little_endian") && header.vertex_count > 0 {
        match read_ply_stats(&mut reader, header.vertex_count, header.num_rest) {
            Ok(stats) => {
                println!();
                println!("Bounding box:");
                println!("  X: [{:.3}, {:.3}]", stats.bbox_x.0, stats.bbox_x.1);
                println!("  Y: [{:.3}, {:.3}]", stats.bbox_y.0, stats.bbox_y.1);
                println!("  Z: [{:.3}, {:.3}]", stats.bbox_z.0, stats.bbox_z.1);
                println!();
                println!("Opacity stats:");
                println!(
                    "  Min: {:.3}, Max: {:.3}, Mean: {:.3}",
                    stats.opacity_min, stats.opacity_max, stats.opacity_mean
                );
                println!();
                println!("Scale stats:");
                println!(
                    "  Min: {:.4}, Max: {:.3}, Mean: {:.3}",
                    stats.scale_min, stats.scale_max, stats.scale_mean
                );
            }
            Err(e) => {
                println!();
                println!("(Stats unavailable: {})", e);
            }
        }
    } else if header.format.contains("ascii") {
        println!();
        println!("(Stats not available for ASCII PLY format)");
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// SafeTensors inspection
// ---------------------------------------------------------------------------

/// Dtype display string.
fn dtype_str(dtype: safetensors::Dtype) -> &'static str {
    match dtype {
        safetensors::Dtype::BOOL => "bool",
        safetensors::Dtype::U8 => "u8",
        safetensors::Dtype::I8 => "i8",
        safetensors::Dtype::I16 => "i16",
        safetensors::Dtype::U16 => "u16",
        safetensors::Dtype::F16 => "f16",
        safetensors::Dtype::BF16 => "bf16",
        safetensors::Dtype::I32 => "i32",
        safetensors::Dtype::U32 => "u32",
        safetensors::Dtype::F32 => "f32",
        safetensors::Dtype::F64 => "f64",
        safetensors::Dtype::I64 => "i64",
        safetensors::Dtype::U64 => "u64",
        _ => "unknown",
    }
}

/// Inspect a `.safetensors` file and print metadata.
fn inspect_safetensors(path: &Path) -> Result<()> {
    let file_size_bytes = std::fs::metadata(path)
        .with_context(|| format!("Cannot stat: {}", path.display()))?
        .len();
    let file_size_mb = file_size_bytes as f64 / (1024.0 * 1024.0);

    let bytes =
        std::fs::read(path).with_context(|| format!("Failed to read: {}", path.display()))?;

    let st = safetensors::SafeTensors::deserialize(&bytes)
        .with_context(|| format!("Failed to deserialize SafeTensors: {}", path.display()))?;

    let (_header_size, header_meta) = safetensors::SafeTensors::read_metadata(&bytes)
        .with_context(|| format!("Failed to read SafeTensors metadata: {}", path.display()))?;

    let file_name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("<unknown>");

    println!("File: {}", file_name);
    println!("Type: SafeTensors");
    println!("Tensors:");

    // Collect and sort tensor names for deterministic output.
    let mut names: Vec<&str> = st.names();
    names.sort_unstable();

    for name in &names {
        if let Ok(tv) = st.tensor(name) {
            let shape_str = tv
                .shape()
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            println!("  {}: [{}] {}", name, shape_str, dtype_str(tv.dtype()));
        }
    }

    // Metadata dict
    if let Some(meta_map) = header_meta.metadata().as_ref() {
        if !meta_map.is_empty() {
            println!("Metadata:");
            let mut keys: Vec<&String> = meta_map.keys().collect();
            keys.sort_unstable();
            for key in keys {
                if let Some(val) = meta_map.get(key) {
                    println!("  {}: {}", key, val);
                }
            }
        }
    }

    println!("File size: {:.1} MB", file_size_mb);

    Ok(())
}

// ---------------------------------------------------------------------------
// JSON inspection
// ---------------------------------------------------------------------------

/// Inspect a `.json` file and print key fields.
fn inspect_json(path: &Path) -> Result<()> {
    let file_size_bytes = std::fs::metadata(path)
        .with_context(|| format!("Cannot stat: {}", path.display()))?
        .len();
    let file_size_mb = file_size_bytes as f64 / (1024.0 * 1024.0);

    let content = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read: {}", path.display()))?;

    let value: serde_json::Value = serde_json::from_str(&content)
        .with_context(|| format!("Failed to parse JSON: {}", path.display()))?;

    let file_name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("<unknown>");

    println!("File: {}", file_name);
    println!("Type: JSON");
    println!("File size: {:.1} MB", file_size_mb);
    println!();

    // Attempt to detect the JSON schema type.
    if let serde_json::Value::Object(ref map) = value {
        // Determine if this looks like a checkpoint or a config.
        let has_checkpoint_keys = map.contains_key("iteration")
            || map.contains_key("num_gaussians")
            || map.contains_key("final_loss")
            || map.contains_key("model_path");
        let has_config_keys = map.contains_key("training")
            || map.contains_key("optimizer")
            || map.contains_key("device");

        if has_checkpoint_keys {
            println!("Schema: Training Checkpoint");
            println!("Keys:");
            print_json_object_summary(map, 0);
        } else if has_config_keys {
            println!("Schema: Training Config");
            println!("Keys:");
            print_json_object_summary(map, 0);
        } else {
            println!("Schema: Generic JSON Object ({} keys)", map.len());
            println!("Keys:");
            print_json_object_summary(map, 0);
        }
    } else {
        println!("Value type: {}", json_type_name(&value));
        println!("Content: {}", truncate_str(&value.to_string(), 200));
    }

    Ok(())
}

/// Recursively print a JSON object summary (2-level deep).
fn print_json_object_summary(map: &serde_json::Map<String, serde_json::Value>, depth: usize) {
    let indent = "  ".repeat(depth + 1);
    let mut keys: Vec<&String> = map.keys().collect();
    keys.sort_unstable();
    for key in keys {
        if let Some(val) = map.get(key) {
            match val {
                serde_json::Value::Object(sub) if depth == 0 => {
                    println!("{}[{}] ({} keys)", indent, key, sub.len());
                    print_json_object_summary(sub, depth + 1);
                }
                serde_json::Value::Array(arr) => {
                    println!("{}{}: Array({} items)", indent, key, arr.len());
                }
                other => {
                    println!(
                        "{}{}: {}",
                        indent,
                        key,
                        truncate_str(&other.to_string(), 80)
                    );
                }
            }
        }
    }
}

/// Short type name for a JSON value.
fn json_type_name(value: &serde_json::Value) -> &'static str {
    match value {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "bool",
        serde_json::Value::Number(_) => "number",
        serde_json::Value::String(_) => "string",
        serde_json::Value::Array(_) => "array",
        serde_json::Value::Object(_) => "object",
    }
}

/// Truncate a string to at most `max_len` bytes, appending `…` if truncated.
fn truncate_str(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}…", &s[..max_len])
    }
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

    // Helper: create a minimal binary_little_endian PLY with N vertices.
    fn write_minimal_ply(path: &Path, n_vertices: usize, num_rest: usize) -> Result<()> {
        let mut f = fs::File::create(path)?;
        // Header
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

        // Binary body: for each vertex write all floats as LE f32.
        // Layout: x, y, z, nx, ny, nz, f_dc_0, f_dc_1, f_dc_2,
        //         f_rest_0..N, opacity, scale_0..2, rot_0..3
        let floats_per_vertex = 3 + 3 + 3 + num_rest + 1 + 3 + 4;
        for i in 0..n_vertices {
            for j in 0..floats_per_vertex {
                let v: f32 = (i * 100 + j) as f32 * 0.001;
                f.write_all(&v.to_le_bytes())?;
            }
        }
        Ok(())
    }

    #[test]
    fn test_sh_degree_from_rest() {
        // SH degree 0: 1 coeff/channel * 3 = 3 total, num_rest = 0
        assert_eq!(sh_degree_from_rest(0), 0);
        // SH degree 1: 4 coeffs/channel * 3 = 12 total, 3 from dc → num_rest = 9
        assert_eq!(sh_degree_from_rest(9), 1);
        // SH degree 2: 9 coeffs/channel * 3 = 27 total, num_rest = 24
        assert_eq!(sh_degree_from_rest(24), 2);
        // SH degree 3: 16 coeffs/channel * 3 = 48 total, num_rest = 45
        assert_eq!(sh_degree_from_rest(45), 3);
    }

    #[test]
    fn test_compute_stats() {
        let vals = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let (min, max, mean) = compute_stats(&vals);
        assert!((min - 1.0).abs() < 1e-5);
        assert!((max - 5.0).abs() < 1e-5);
        assert!((mean - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_compute_stats_empty() {
        let vals: Vec<f32> = vec![];
        let (min, max, mean) = compute_stats(&vals);
        assert_eq!(min, 0.0);
        assert_eq!(max, 0.0);
        assert_eq!(mean, 0.0);
    }

    #[test]
    fn test_parse_ply_header_info_basic() -> Result<()> {
        let tmp = env::temp_dir().join("test_parse_ply_header.ply");
        write_minimal_ply(&tmp, 5, 9)?; // SH degree 1 → 9 rest coeffs
        let file = File::open(&tmp)?;
        let mut reader = BufReader::new(file);
        let header = parse_ply_header_info(&mut reader)?;
        assert_eq!(header.vertex_count, 5);
        assert_eq!(header.num_rest, 9);
        assert_eq!(sh_degree_from_rest(header.num_rest), 1);
        assert!(header.format.contains("binary_little_endian"));
        // x, y, z, nx, ny, nz, f_dc_0..2, f_rest_0..8, opacity, scale_0..2, rot_0..3
        let expected_prop_count = 3 + 3 + 3 + 9 + 1 + 3 + 4;
        assert_eq!(header.properties.len(), expected_prop_count);
        fs::remove_file(&tmp).ok();
        Ok(())
    }

    #[test]
    fn test_inspect_ply_runs_without_error() -> Result<()> {
        let tmp = env::temp_dir().join("test_inspect_ply.ply");
        write_minimal_ply(&tmp, 3, 45)?; // SH degree 3
        let args = InfoArgs { path: tmp.clone() };
        run_info(args)?;
        fs::remove_file(&tmp).ok();
        Ok(())
    }

    #[test]
    fn test_inspect_json_runs_without_error() -> Result<()> {
        let tmp = env::temp_dir().join("test_inspect.json");
        let content = serde_json::json!({
            "iteration": 5000,
            "num_gaussians": 12345,
            "final_loss": 0.042,
            "model_path": "/tmp/model.safetensors"
        });
        fs::write(&tmp, content.to_string())?;
        let args = InfoArgs { path: tmp.clone() };
        run_info(args)?;
        fs::remove_file(&tmp).ok();
        Ok(())
    }

    #[test]
    fn test_info_unsupported_extension() {
        let tmp = env::temp_dir().join("test_unsupported.bin");
        fs::write(&tmp, b"binary data").ok();
        let args = InfoArgs { path: tmp.clone() };
        let result = run_info(args);
        assert!(result.is_err());
        let err_msg = result.err().map(|e| e.to_string()).unwrap_or_default();
        assert!(err_msg.contains("Unsupported file extension"));
        fs::remove_file(&tmp).ok();
    }

    #[test]
    fn test_info_file_not_found() {
        let path = env::temp_dir().join("does_not_exist_oxigaf_test.ply");
        let args = InfoArgs { path };
        let result = run_info(args);
        assert!(result.is_err());
        let err_msg = result.err().map(|e| e.to_string()).unwrap_or_default();
        assert!(err_msg.contains("not found"));
    }
}
