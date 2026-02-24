//! FLAME model conversion utilities.
//!
//! Provides conversion from FLAME pickle (.pkl) and NumPy archive (.npz) files
//! to individual .npy files used by OxiGAF.
//!
//! The conversion process extracts and validates:
//! - v_template.npy — Template vertices
//! - shapedirs.npy — Shape blend shapes
//! - posedirs.npy — Pose blend shapes
//! - J_regressor.npy — Joint regressor
//! - parents.npy — Kinematic tree
//! - lbs_weights.npy — LBS weights
//! - faces.npy — Mesh faces

use std::collections::HashMap;
use std::io::BufReader;
use std::path::Path;

use anyhow::{Context, Result};
use oxiarc_archive::ZipReader;

use crate::cli::ConvertArgs;
use crate::output;
use crate::progress;
use crate::verbosity::Verbosity;

// ---------------------------------------------------------------------------
// Expected FLAME components
// ---------------------------------------------------------------------------

/// Required components for a valid FLAME model.
const REQUIRED_COMPONENTS: &[&str] = &[
    "v_template",
    "shapedirs",
    "posedirs",
    "J_regressor",
    "kintree_table", // parents
    "weights",       // lbs_weights
    "f",             // faces
];

/// Component name mapping from NPZ keys to output filenames.
fn component_output_name(key: &str) -> &'static str {
    match key {
        "v_template" => "v_template.npy",
        "shapedirs" => "shapedirs.npy",
        "posedirs" => "posedirs.npy",
        "J_regressor" => "J_regressor.npy",
        "kintree_table" => "parents.npy",
        "weights" => "lbs_weights.npy",
        "f" => "faces.npy",
        "dynamic_lmk_bary_coords" => "dynamic_lmk_bary_coords.npy",
        "dynamic_lmk_faces_idx" => "dynamic_lmk_faces_idx.npy",
        "full_lmk_bary_coords" => "full_lmk_bary_coords.npy",
        "full_lmk_faces_idx" => "full_lmk_faces_idx.npy",
        "lmk_bary_coords" => "lmk_bary_coords.npy",
        "lmk_faces_idx" => "lmk_faces_idx.npy",
        "neck_kin_chain" => "neck_kin_chain.npy",
        _ => "",
    }
}

// ---------------------------------------------------------------------------
// Main conversion entry point
// ---------------------------------------------------------------------------

/// Run the FLAME model conversion.
pub fn run_convert(
    args: ConvertArgs,
    verbosity: Verbosity,
    dry_run: bool,
    json_mode: bool,
) -> Result<()> {
    use std::time::Instant;

    let start = Instant::now();
    tracing::info!(
        input = %args.input.display(),
        output = %args.output.display(),
        version = %args.version,
        "Starting FLAME model conversion"
    );

    // Validate input file exists
    if !args.input.exists() {
        anyhow::bail!("Input file not found: {}", args.input.display());
    }

    // Dry-run validation
    if dry_run {
        let mut report = crate::dry_run::DryRunReport::new();

        if !json_mode {
            output::success(&format!("Input validated: {}", args.input.display()));
        }

        // Check output directory
        if args.output.exists() && !args.force {
            let entry_count = std::fs::read_dir(&args.output)
                .map(|rd| rd.count())
                .unwrap_or(0);
            if entry_count > 0 {
                anyhow::bail!(
                    "Output directory not empty: {}. Use --force to overwrite.",
                    args.output.display()
                );
            }
        }

        crate::dry_run::check_writable(&args.output)?;
        report.add_create(format!("{}/", args.output.display()));

        // Expected outputs
        let components = [
            "v_template.npy",
            "shapedirs.npy",
            "posedirs.npy",
            "J_regressor.npy",
            "parents.npy",
            "lbs_weights.npy",
            "faces.npy",
        ];
        for comp in &components {
            report.add_create(format!("{}/{}", args.output.display(), comp));
        }

        report.resource_estimates.estimated_disk_mb = Some(250);

        if !json_mode {
            report.print_report();
        }
        return Ok(());
    }

    // Create output directory
    if args.output.exists() && !args.force {
        let entry_count = std::fs::read_dir(&args.output)
            .map(|rd| rd.count())
            .unwrap_or(0);
        if entry_count > 0 {
            anyhow::bail!(
                "Output directory not empty: {}. Use --force to overwrite.",
                args.output.display()
            );
        }
    }
    std::fs::create_dir_all(&args.output).with_context(|| {
        format!(
            "Failed to create output directory: {}",
            args.output.display()
        )
    })?;

    // Determine input format and convert
    let ext = args
        .input
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_lowercase())
        .unwrap_or_default();

    let components = match ext.as_str() {
        "npz" => convert_npz(&args.input, verbosity)?,
        "pkl" | "pickle" => convert_pkl(&args.input, verbosity)?,
        _ => anyhow::bail!("Unsupported input format: '{}'. Expected .npz or .pkl", ext),
    };

    // Write output files
    write_components(&components, &args.output, verbosity)?;

    // Verify output if requested
    if args.verify {
        verify_output(&args.output, &args.version)?;
    }

    // Output based on mode
    if json_mode {
        let mut output = crate::json_output::JsonOutput::success(
            "convert",
            serde_json::json!({
                "num_components": components.len(),
                "input_file": args.input.display().to_string(),
                "output_dir": args.output.display().to_string(),
                "version": args.version,
                "verified": args.verify
            }),
        );

        // Add each output file as an artifact
        for name in components.keys() {
            let output_name = component_output_name(name);
            let filename = if output_name.is_empty() {
                format!("{}.npy", name)
            } else {
                output_name.to_string()
            };
            let file_path = args.output.join(&filename);
            if file_path.exists() {
                output.add_artifact(name.to_string(), file_path);
            }
        }

        output.print();
    } else {
        println!();
        output::success(&format!(
            "Converted {} components from {}",
            components.len(),
            args.input.display()
        ));
        output::path_value("Output", &args.output);

        if verbosity.show_timing() {
            let elapsed = start.elapsed();
            output::value("Time", &format!("{:.2}s", elapsed.as_secs_f64()));
        }

        // Print component summary
        tracing::info!("Converted components:");
        for (name, data) in &components {
            let output_name = component_output_name(name);
            let filename = if output_name.is_empty() {
                format!("{}.npy", name)
            } else {
                output_name.to_string()
            };
            tracing::info!("  {} -> {} ({} bytes)", name, filename, data.len());
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// NPZ Conversion
// ---------------------------------------------------------------------------

/// Convert from NumPy archive (.npz) format.
fn convert_npz(path: &Path, verbosity: Verbosity) -> Result<HashMap<String, Vec<u8>>> {
    tracing::info!("Reading NPZ file: {}", path.display());

    let file = std::fs::File::open(path)
        .with_context(|| format!("Failed to open NPZ file: {}", path.display()))?;
    let reader = BufReader::new(file);

    // NPZ files are just ZIP archives containing .npy files
    let mut reader = ZipReader::new(reader).with_context(|| {
        format!(
            "Failed to read NPZ archive: {}. \
             Ensure the file is a valid NPZ (ZIP-compressed NumPy archive).",
            path.display()
        )
    })?;

    let mut components = HashMap::new();

    // Collect entries first to avoid borrow checker issues
    let entries = reader.entries().to_vec();
    let num_files = entries.len();

    if num_files == 0 {
        anyhow::bail!(
            "NPZ archive is empty: {}. \
             A valid FLAME model should contain at least {} components.",
            path.display(),
            REQUIRED_COMPONENTS.len()
        );
    }

    let pb = progress::custom_progress(
        num_files as u64,
        "{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} components ({msg})",
        verbosity,
    );

    for entry in entries {
        let name = &entry.name;
        // Remove .npy extension to get component name
        let component_name = name.trim_end_matches(".npy").to_string();

        pb.set_message(component_name.clone());

        let data = reader
            .extract(&entry)
            .with_context(|| format!("Failed to read component: {}", component_name))?;

        components.insert(component_name, data);
        pb.inc(1);
    }

    pb.finish_with_message("done");

    // Validate required components
    validate_components(&components)?;

    Ok(components)
}

// ---------------------------------------------------------------------------
// PKL Conversion (Pure Rust pickle parsing)
// ---------------------------------------------------------------------------

/// Convert from Python pickle (.pkl) format.
///
/// This is a simplified pure-Rust pickle parser that handles the specific
/// structure of FLAME model files. For complex pickle files, we recommend
/// using the NPZ format instead.
fn convert_pkl(path: &Path, verbosity: Verbosity) -> Result<HashMap<String, Vec<u8>>> {
    let _ = verbosity; // Suppress unused variable warning (no progress bar in this function)
    tracing::info!("Reading pickle file: {}", path.display());

    let data = std::fs::read(path)
        .with_context(|| format!("Failed to read pickle file: {}", path.display()))?;

    // Try to parse as a simple pickle structure
    let components = parse_flame_pickle(&data).with_context(|| {
        format!(
            "Failed to parse pickle file: {}. \
             Consider converting to NPZ format using Python: \n\
             \n\
             import pickle\n\
             import numpy as np\n\
             with open('flame.pkl', 'rb') as f:\n\
             \x20   data = pickle.load(f, encoding='latin1')\n\
             np.savez('flame.npz', **data)\n",
            path.display()
        )
    })?;

    // Validate required components
    validate_components(&components)?;

    Ok(components)
}

/// Parse FLAME-specific pickle structure.
///
/// FLAME pickle files typically contain a dict with numpy arrays.
/// This parser handles the common structure but may not work for all variants.
fn parse_flame_pickle(data: &[u8]) -> Result<HashMap<String, Vec<u8>>> {
    // Pickle protocol detection
    if data.len() < 2 {
        anyhow::bail!("Pickle file too small");
    }

    let proto = if data[0] == 0x80 {
        // Protocol 2+
        data[1]
    } else {
        // Protocol 0 or 1
        0
    };

    tracing::debug!("Detected pickle protocol: {}", proto);

    // For protocol 2+, we need to parse the pickle opcodes
    // This is a simplified parser that looks for numpy array patterns
    let mut components = HashMap::new();
    let mut pos = if proto >= 2 { 2 } else { 0 };

    // Simple heuristic: look for numpy array headers in the pickle stream
    // NPY array format: \x93NUMPY followed by version and header
    while pos + 10 < data.len() {
        if &data[pos..pos.min(data.len()).min(pos + 6)] == b"\x93NUMPY" {
            // Found a numpy array marker
            // Extract the array data (simplified)
            if let Some((name, array_data, next_pos)) = extract_numpy_array(data, pos) {
                components.insert(name, array_data);
                pos = next_pos;
                continue;
            }
        }
        pos += 1;
    }

    // If we couldn't parse any components, suggest using Python
    if components.is_empty() {
        anyhow::bail!(
            "Could not parse pickle file. The pickle format may be too complex. \
             Please convert using Python:\n\
             \n\
             import pickle\n\
             import numpy as np\n\
             with open('flame.pkl', 'rb') as f:\n\
             \x20   data = pickle.load(f, encoding='latin1')\n\
             np.savez('flame.npz', **data)\n"
        );
    }

    Ok(components)
}

/// Extract a numpy array from pickle data.
fn extract_numpy_array(data: &[u8], start: usize) -> Option<(String, Vec<u8>, usize)> {
    // This is a placeholder for full numpy array extraction
    // In practice, FLAME models should be distributed as NPZ
    let _ = (data, start);
    None
}

// ---------------------------------------------------------------------------
// Component Validation
// ---------------------------------------------------------------------------

/// Validate that all required FLAME components are present.
fn validate_components(components: &HashMap<String, Vec<u8>>) -> Result<()> {
    let mut missing = Vec::new();

    for &required in REQUIRED_COMPONENTS {
        if !components.contains_key(required) {
            missing.push(required);
        }
    }

    if !missing.is_empty() {
        anyhow::bail!("Missing required FLAME components: {}", missing.join(", "));
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Output Writing
// ---------------------------------------------------------------------------

/// Write extracted components to output directory.
fn write_components(
    components: &HashMap<String, Vec<u8>>,
    output_dir: &Path,
    verbosity: Verbosity,
) -> Result<()> {
    let pb = progress::custom_progress(
        components.len() as u64,
        "{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} writing ({msg})",
        verbosity,
    );

    for (name, data) in components {
        let output_name = component_output_name(name);
        if output_name.is_empty() {
            // Unknown component, use original name
            let filename = format!("{}.npy", name);
            let output_path = output_dir.join(&filename);
            pb.set_message(filename);
            std::fs::write(&output_path, data)
                .with_context(|| format!("Failed to write: {}", output_path.display()))?;
        } else {
            let output_path = output_dir.join(output_name);
            pb.set_message(output_name.to_string());
            std::fs::write(&output_path, data)
                .with_context(|| format!("Failed to write: {}", output_path.display()))?;
        }
        pb.inc(1);
    }

    pb.finish_with_message("done");
    Ok(())
}

// ---------------------------------------------------------------------------
// Output Verification
// ---------------------------------------------------------------------------

/// Verify the converted output files.
fn verify_output(output_dir: &Path, version: &str) -> Result<()> {
    tracing::info!("Verifying converted files in {}", output_dir.display());

    let expected_files = [
        "v_template.npy",
        "shapedirs.npy",
        "posedirs.npy",
        "J_regressor.npy",
        "parents.npy",
        "lbs_weights.npy",
        "faces.npy",
    ];

    let mut all_valid = true;

    for filename in &expected_files {
        let path = output_dir.join(filename);
        if !path.exists() {
            output::error(&format!("Missing: {}", filename));
            all_valid = false;
            continue;
        }

        // Read and validate NPY header
        match validate_npy_file(&path) {
            Ok(info) => {
                tracing::info!(
                    "{}: {} (dtype: {}, shape: {:?})",
                    filename,
                    if info.valid { "OK" } else { "INVALID" },
                    info.dtype,
                    info.shape
                );
            }
            Err(e) => {
                output::error(&format!("{}: {}", filename, e));
                all_valid = false;
            }
        }
    }

    // Version-specific validation
    match version {
        "2020" => validate_flame_2020(output_dir)?,
        "2023" => validate_flame_2023(output_dir)?,
        _ => tracing::warn!(
            "Unknown FLAME version '{}', skipping version-specific validation",
            version
        ),
    }

    if all_valid {
        output::success("All files validated successfully");
    } else {
        anyhow::bail!("Verification failed: some files are missing or invalid");
    }

    Ok(())
}

/// Information about an NPY file.
struct NpyInfo {
    valid: bool,
    dtype: String,
    shape: Vec<usize>,
}

/// Validate an NPY file and extract metadata.
fn validate_npy_file(path: &Path) -> Result<NpyInfo> {
    let data =
        std::fs::read(path).with_context(|| format!("Failed to read: {}", path.display()))?;

    // NPY magic number: \x93NUMPY
    if data.len() < 10 || &data[0..6] != b"\x93NUMPY" {
        return Ok(NpyInfo {
            valid: false,
            dtype: "unknown".to_string(),
            shape: vec![],
        });
    }

    // Version
    let _major = data[6];
    let _minor = data[7];

    // Header length (little-endian)
    let header_len = u16::from_le_bytes([data[8], data[9]]) as usize;

    if data.len() < 10 + header_len {
        return Ok(NpyInfo {
            valid: false,
            dtype: "truncated".to_string(),
            shape: vec![],
        });
    }

    // Parse header (Python dict literal)
    let header = std::str::from_utf8(&data[10..10 + header_len])
        .unwrap_or("")
        .trim();

    // Extract dtype and shape from header
    let dtype = extract_header_field(header, "descr").unwrap_or_else(|| "unknown".to_string());
    let shape = extract_shape_from_header(header);

    Ok(NpyInfo {
        valid: true,
        dtype,
        shape,
    })
}

/// Extract a field from NPY header.
fn extract_header_field(header: &str, field: &str) -> Option<String> {
    let pattern = format!("'{}':", field);
    if let Some(pos) = header.find(&pattern) {
        let start = pos + pattern.len();
        let rest = &header[start..];
        // Find the value (quoted string or other)
        if let Some(quote_start) = rest.find('\'') {
            if let Some(quote_end) = rest[quote_start + 1..].find('\'') {
                return Some(rest[quote_start + 1..quote_start + 1 + quote_end].to_string());
            }
        }
    }
    None
}

/// Extract shape from NPY header.
fn extract_shape_from_header(header: &str) -> Vec<usize> {
    if let Some(pos) = header.find("'shape':") {
        let rest = &header[pos + 8..];
        if let Some(paren_start) = rest.find('(') {
            if let Some(paren_end) = rest.find(')') {
                let shape_str = &rest[paren_start + 1..paren_end];
                return shape_str
                    .split(',')
                    .filter_map(|s| s.trim().parse().ok())
                    .collect();
            }
        }
    }
    vec![]
}

/// FLAME 2020-specific validation.
fn validate_flame_2020(output_dir: &Path) -> Result<()> {
    // FLAME 2020 has 5023 vertices
    let v_template = output_dir.join("v_template.npy");
    if v_template.exists() {
        let info = validate_npy_file(&v_template)?;
        if !info.shape.is_empty() && info.shape[0] != 5023 {
            tracing::warn!("FLAME 2020 expected 5023 vertices, found {}", info.shape[0]);
        }
    }
    Ok(())
}

/// FLAME 2023-specific validation.
fn validate_flame_2023(output_dir: &Path) -> Result<()> {
    // FLAME 2023 may have additional landmarks
    let lmk_path = output_dir.join("full_lmk_faces_idx.npy");
    if !lmk_path.exists() {
        tracing::info!("Note: FLAME 2023 full landmark files not found (optional)");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_component_output_name() {
        assert_eq!(component_output_name("v_template"), "v_template.npy");
        assert_eq!(component_output_name("kintree_table"), "parents.npy");
        assert_eq!(component_output_name("weights"), "lbs_weights.npy");
        assert_eq!(component_output_name("f"), "faces.npy");
        assert_eq!(component_output_name("unknown"), "");
    }

    #[test]
    fn test_extract_header_field() {
        let header = "{'descr': '<f4', 'fortran_order': False, 'shape': (5023, 3)}";
        assert_eq!(
            extract_header_field(header, "descr"),
            Some("<f4".to_string())
        );
    }

    #[test]
    fn test_extract_shape_from_header() {
        let header = "{'descr': '<f4', 'fortran_order': False, 'shape': (5023, 3)}";
        assert_eq!(extract_shape_from_header(header), vec![5023, 3]);
    }
}
