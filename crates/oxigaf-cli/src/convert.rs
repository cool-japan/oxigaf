//! FLAME model conversion utilities.
//!
//! Provides conversion from FLAME pickle (.pkl) and NumPy archive (.npz) files
//! to the individual `.npy` files consumed by `oxigaf_flame::load_flame_model`.
//!
//! The conversion emits exactly the file set that loader expects:
//! - `v_template.npy` — Template vertices `[5023, 3]`, float32
//! - `faces.npy` — Mesh faces `[9976, 3]`, int32
//! - `shapedirs.npy` — Shape blend shapes `[5023, 3, 300]`, float32
//! - `expressiondirs.npy` — Expression blend shapes `[5023, 3, 100]`, float32
//! - `posedirs.npy` — Pose blend shapes `[5023, 3, 36]`, float32
//! - `j_regressor.npy` — Joint regressor `[5, 5023]`, float32
//! - `kintree_table.npy` — Kinematic tree `[2, 5]`, int32
//! - `lbs_weights.npy` — LBS weights `[5023, 5]`, float32
//!
//! FLAME distributions pack the shape and expression bases into a single
//! `shapedirs` array and store `J_regressor` as a SciPy sparse matrix, so the
//! conversion splits the basis, densifies the regressor and casts every array
//! to the dtype the loader requires.

use std::collections::{HashMap, HashSet};
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

/// Required components for a valid FLAME model, keyed by source name.
const REQUIRED_COMPONENTS: &[&str] = &[
    "v_template",
    "shapedirs",
    "posedirs",
    "J_regressor",
    "kintree_table", // parents are derived from row 0 by the loader
    "weights",       // lbs_weights
    "f",             // faces
];

/// Files produced for `oxigaf_flame::load_flame_model`.
const OUTPUT_FILES: &[&str] = &[
    "expressiondirs.npy",
    "faces.npy",
    "j_regressor.npy",
    "kintree_table.npy",
    "lbs_weights.npy",
    "posedirs.npy",
    "shapedirs.npy",
    "v_template.npy",
];

/// Maximum number of identity (shape) basis columns kept in `shapedirs.npy`.
const MAX_SHAPE_COMPONENTS: usize = 300;

/// Component name mapping from source keys to output filenames.
fn component_output_name(key: &str) -> &'static str {
    match key {
        "v_template" => "v_template.npy",
        "shapedirs" => "shapedirs.npy",
        "expressiondirs" => "expressiondirs.npy",
        "posedirs" => "posedirs.npy",
        "J_regressor" => "j_regressor.npy",
        "kintree_table" => "kintree_table.npy",
        "weights" => "lbs_weights.npy",
        "f" => "faces.npy",
        "vt" | "uv" => "uv.npy",
        "ft" | "uv_faces" => "uv_faces.npy",
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

/// The dtype an output file must have for the FLAME loader to read it.
fn required_dtype(filename: &str) -> Option<&'static str> {
    match filename {
        "v_template.npy" | "shapedirs.npy" | "expressiondirs.npy" | "posedirs.npy"
        | "j_regressor.npy" | "lbs_weights.npy" | "uv.npy" => Some("<f4"),
        "faces.npy" | "kintree_table.npy" | "uv_faces.npy" => Some("<i4"),
        _ => None,
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

    // Determine input format up-front so that a dry run cannot promise output
    // for a format the conversion does not support.
    let ext = args
        .input
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_lowercase())
        .unwrap_or_default();
    if !matches!(ext.as_str(), "npz" | "pkl" | "pickle") {
        anyhow::bail!("Unsupported input format: '{}'. Expected .npz or .pkl", ext);
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
        for comp in OUTPUT_FILES {
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

    let components = match ext.as_str() {
        "npz" => convert_npz(&args.input, verbosity)?,
        _ => convert_pkl(&args.input, verbosity)?,
    };

    // Reshape the source components into the loader's file set.
    let files = prepare_outputs(&components, args.include_uv)?;

    // Write output files
    write_components(&files, &args.output, verbosity)?;

    // Verify output if requested
    if args.verify {
        verify_output(&args.output, &args.version)?;
    }

    // Output based on mode
    if json_mode {
        let mut output = crate::json_output::JsonOutput::success(
            "convert",
            serde_json::json!({
                "num_components": files.len(),
                "input_file": args.input.display().to_string(),
                "output_dir": args.output.display().to_string(),
                "version": args.version,
                "verified": args.verify
            }),
        );

        // Add each output file as an artifact
        for (filename, _) in &files {
            let file_path = args.output.join(filename);
            if file_path.exists() {
                let label = filename.trim_end_matches(".npy").to_string();
                output.add_artifact(label, file_path);
            }
        }

        output.print();
    } else {
        println!();
        output::success(&format!(
            "Converted {} components from {}",
            files.len(),
            args.input.display()
        ));
        output::path_value("Output", &args.output);

        if verbosity.show_timing() {
            let elapsed = start.elapsed();
            output::value("Time", &format!("{:.2}s", elapsed.as_secs_f64()));
        }

        // Print component summary
        tracing::info!("Converted components:");
        for (filename, array) in &files {
            tracing::info!(
                "  {} shape {:?} dtype {} ({} bytes)",
                filename,
                array.shape,
                array.descr,
                array.data.len()
            );
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// NPZ Conversion
// ---------------------------------------------------------------------------

/// Convert from NumPy archive (.npz) format.
fn convert_npz(path: &Path, verbosity: Verbosity) -> Result<HashMap<String, npy::NpyArray>> {
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
        let array = npy::parse(&data)
            .with_context(|| format!("Component '{component_name}' is not a valid NPY stream"))?;

        components.insert(component_name, array);
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
/// The pickle stream is decoded by the pure-Rust virtual machine in [`pickle`],
/// which understands protocols 0-5 and reconstructs `numpy.ndarray`,
/// `numpy.dtype`, `chumpy.ch.Ch` and `scipy.sparse` payloads.
fn convert_pkl(path: &Path, verbosity: Verbosity) -> Result<HashMap<String, npy::NpyArray>> {
    tracing::info!("Reading pickle file: {}", path.display());

    let data = std::fs::read(path)
        .with_context(|| format!("Failed to read pickle file: {}", path.display()))?;

    let pb = progress::spinner("Decoding pickle stream", verbosity);
    let parsed = pickle::load_arrays(&data);
    pb.finish_and_clear();

    let components = parsed.with_context(|| {
        format!(
            "Failed to parse pickle file: {}. \
             If the model uses an unsupported Python type, convert it to NPZ first: \n\
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

// ---------------------------------------------------------------------------
// Component Validation
// ---------------------------------------------------------------------------

/// Validate that all required FLAME components are present.
fn validate_components(components: &HashMap<String, npy::NpyArray>) -> Result<()> {
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
// Output Preparation
// ---------------------------------------------------------------------------

/// Register one output file, rejecting duplicates.
fn push_output(
    files: &mut Vec<(String, npy::NpyArray)>,
    emitted: &mut HashSet<String>,
    filename: &str,
    array: npy::NpyArray,
) -> Result<()> {
    if !emitted.insert(filename.to_string()) {
        anyhow::bail!("Duplicate output file '{filename}' produced by the conversion");
    }
    files.push((filename.to_string(), array));
    Ok(())
}

/// Build the loader-compatible output file set from the raw source components.
///
/// This splits the packed FLAME basis into `shapedirs` + `expressiondirs` and
/// casts every loader-visible array to the dtype `load_flame_model` requires.
fn prepare_outputs(
    components: &HashMap<String, npy::NpyArray>,
    include_uv: bool,
) -> Result<Vec<(String, npy::NpyArray)>> {
    let mut files: Vec<(String, npy::NpyArray)> = Vec::new();
    let mut emitted: HashSet<String> = HashSet::new();

    // --- Blend shapes: FLAME packs the identity and expression bases together.
    let shapedirs = components
        .get("shapedirs")
        .context("Missing required FLAME component 'shapedirs'")?;
    if shapedirs.shape.len() != 3 {
        anyhow::bail!(
            "'shapedirs' must be 3-dimensional, found shape {:?}",
            shapedirs.shape
        );
    }
    let total = shapedirs.shape[2];
    let (shape_basis, expr_basis) = match components.get("expressiondirs") {
        Some(expr) => (shapedirs.clone(), expr.clone()),
        None => {
            let n_shape = total.min(MAX_SHAPE_COMPONENTS);
            if n_shape == total {
                output::warning(
                    "Source model has no expression basis; writing an empty expressiondirs.npy",
                );
            }
            (
                npy::slice_last_axis(shapedirs, 0, n_shape)?,
                npy::slice_last_axis(shapedirs, n_shape, total)?,
            )
        }
    };
    push_output(&mut files, &mut emitted, "shapedirs.npy", shape_basis)?;
    push_output(&mut files, &mut emitted, "expressiondirs.npy", expr_basis)?;

    // --- Everything else maps one-to-one.
    for (key, array) in components {
        if key == "shapedirs" || key == "expressiondirs" {
            continue;
        }
        let mapped = component_output_name(key);
        if mapped.is_empty() && !is_simple_name(key) {
            tracing::warn!("Skipping component with an unusable name: {key}");
            continue;
        }
        if matches!(mapped, "uv.npy" | "uv_faces.npy") && !include_uv {
            tracing::debug!("Skipping UV component '{key}' (--include-uv not set)");
            continue;
        }
        let filename = if mapped.is_empty() {
            format!("{key}.npy")
        } else {
            mapped.to_string()
        };
        push_output(&mut files, &mut emitted, &filename, array.clone())?;
    }

    if include_uv && !emitted.contains("uv.npy") {
        output::warning(
            "--include-uv was requested but the source model has no UV coordinates (vt/ft)",
        );
    }

    // --- Cast the loader-visible arrays to the required dtypes.
    for (filename, array) in &mut files {
        if let Some(target) = required_dtype(filename) {
            if array.descr != target {
                let converted = npy::cast(array, target)
                    .with_context(|| format!("Failed to convert {filename} to dtype '{target}'"))?;
                *array = converted;
            }
        }
    }

    // --- Every loader-required file must have been produced.
    let missing: Vec<&str> = OUTPUT_FILES
        .iter()
        .copied()
        .filter(|f| !emitted.contains(*f))
        .collect();
    if !missing.is_empty() {
        anyhow::bail!("Conversion produced no data for: {}", missing.join(", "));
    }

    files.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(files)
}

/// Whether a source key is safe to use verbatim as a filename stem.
fn is_simple_name(key: &str) -> bool {
    !key.is_empty()
        && key != "."
        && key != ".."
        && !key.contains(['/', '\\'])
        && !key.contains('\0')
}

// ---------------------------------------------------------------------------
// Output Writing
// ---------------------------------------------------------------------------

/// Write prepared arrays to the output directory as NPY files.
fn write_components(
    files: &[(String, npy::NpyArray)],
    output_dir: &Path,
    verbosity: Verbosity,
) -> Result<()> {
    let pb = progress::custom_progress(
        files.len() as u64,
        "{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} writing ({msg})",
        verbosity,
    );

    for (filename, array) in files {
        pb.set_message(filename.clone());
        let bytes = npy::serialize(array)
            .with_context(|| format!("Failed to encode NPY stream for {filename}"))?;
        let output_path = output_dir.join(filename);
        std::fs::write(&output_path, &bytes)
            .with_context(|| format!("Failed to write: {}", output_path.display()))?;
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

    let mut all_valid = true;

    for filename in OUTPUT_FILES {
        let path = output_dir.join(filename);
        if !path.exists() {
            output::error(&format!("Missing: {}", filename));
            all_valid = false;
            continue;
        }

        // Read and validate the NPY header
        match validate_npy_file(&path) {
            Ok(info) => {
                if !info.valid {
                    output::error(&format!("{}: INVALID ({})", filename, info.detail));
                    all_valid = false;
                    continue;
                }
                tracing::info!(
                    "{}: OK (dtype: {}, shape: {:?})",
                    filename,
                    info.dtype,
                    info.shape
                );
                if let Some(expected) = required_dtype(filename) {
                    if info.dtype != expected {
                        output::error(&format!(
                            "{}: dtype '{}' but the FLAME loader requires '{}'",
                            filename, info.dtype, expected
                        ));
                        all_valid = false;
                    }
                }
            }
            Err(e) => {
                output::error(&format!("{}: {}", filename, e));
                all_valid = false;
            }
        }
    }

    // Version-specific validation
    let issues = match version {
        "2020" => validate_flame_2020(output_dir)?,
        "2023" => validate_flame_2023(output_dir)?,
        other => {
            tracing::warn!("Unknown FLAME version '{other}', skipping version-specific validation");
            Vec::new()
        }
    };
    for issue in &issues {
        output::error(issue);
    }
    if !issues.is_empty() {
        all_valid = false;
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
    detail: String,
}

/// Validate an NPY file and extract metadata.
fn validate_npy_file(path: &Path) -> Result<NpyInfo> {
    let data =
        std::fs::read(path).with_context(|| format!("Failed to read: {}", path.display()))?;

    let header = match npy::parse_header(&data) {
        Ok(header) => header,
        Err(err) => {
            return Ok(NpyInfo {
                valid: false,
                dtype: "unknown".to_string(),
                shape: Vec::new(),
                detail: format!("{err:#}"),
            });
        }
    };

    let item = match npy::item_size(&header.descr) {
        Ok(item) => item,
        Err(err) => {
            return Ok(NpyInfo {
                valid: false,
                dtype: header.descr.clone(),
                shape: header.shape.clone(),
                detail: format!("{err:#}"),
            });
        }
    };

    let count: usize = header.shape.iter().product();
    let expected = count.saturating_mul(item);
    let available = data.len().saturating_sub(header.data_offset);
    if available < expected {
        return Ok(NpyInfo {
            valid: false,
            dtype: header.descr.clone(),
            shape: header.shape.clone(),
            detail: format!("truncated data: expected {expected} bytes, found {available}"),
        });
    }

    Ok(NpyInfo {
        valid: true,
        dtype: header.descr,
        shape: header.shape,
        detail: String::new(),
    })
}

// ---------------------------------------------------------------------------
// FLAME version validation
// ---------------------------------------------------------------------------

/// Geometry invariants shared by the FLAME 2020 and 2023 releases.
struct FlameSpec {
    /// Number of template vertices.
    n_verts: usize,
    /// Number of skeleton joints.
    n_joints: usize,
    /// Number of pose blend-shape columns.
    n_pose: usize,
    /// Combined identity + expression basis width.
    n_basis: usize,
}

/// Canonical FLAME geometry (5023 vertices, 5 joints, 300 + 100 basis).
const FLAME_SPEC: FlameSpec = FlameSpec {
    n_verts: 5023,
    n_joints: 5,
    n_pose: 36,
    n_basis: 400,
};

/// Read the shape of a converted file, or `None` when it is absent/unreadable.
fn shape_of(output_dir: &Path, name: &str) -> Result<Option<Vec<usize>>> {
    let path = output_dir.join(name);
    if !path.exists() {
        return Ok(None);
    }
    let info = validate_npy_file(&path)?;
    if !info.valid {
        return Ok(None);
    }
    Ok(Some(info.shape))
}

/// Record a shape mismatch, ignoring files that were not produced.
fn expect_shape(issues: &mut Vec<String>, name: &str, want: &[usize], got: Option<Vec<usize>>) {
    if let Some(shape) = got {
        if shape != want {
            issues.push(format!("{name}: expected shape {want:?}, found {shape:?}"));
        }
    }
}

/// Check the array shapes of a converted FLAME directory.
fn validate_flame_shapes(output_dir: &Path, spec: &FlameSpec) -> Result<Vec<String>> {
    let mut issues = Vec::new();
    let n = spec.n_verts;
    let j = spec.n_joints;

    expect_shape(
        &mut issues,
        "v_template.npy",
        &[n, 3],
        shape_of(output_dir, "v_template.npy")?,
    );
    expect_shape(
        &mut issues,
        "posedirs.npy",
        &[n, 3, spec.n_pose],
        shape_of(output_dir, "posedirs.npy")?,
    );
    expect_shape(
        &mut issues,
        "lbs_weights.npy",
        &[n, j],
        shape_of(output_dir, "lbs_weights.npy")?,
    );
    expect_shape(
        &mut issues,
        "kintree_table.npy",
        &[2, j],
        shape_of(output_dir, "kintree_table.npy")?,
    );
    expect_shape(
        &mut issues,
        "j_regressor.npy",
        &[j, n],
        shape_of(output_dir, "j_regressor.npy")?,
    );

    if let Some(shape) = shape_of(output_dir, "faces.npy")? {
        if shape.len() != 2 || shape[1] != 3 {
            issues.push(format!("faces.npy: expected shape [F, 3], found {shape:?}"));
        }
    }

    let shapedirs = shape_of(output_dir, "shapedirs.npy")?;
    let exprdirs = shape_of(output_dir, "expressiondirs.npy")?;
    if let (Some(s), Some(e)) = (&shapedirs, &exprdirs) {
        if s.len() != 3 || s[0] != n || s[1] != 3 {
            issues.push(format!(
                "shapedirs.npy: expected shape [{n}, 3, S], found {s:?}"
            ));
        } else if e.len() != 3 || e[0] != n || e[1] != 3 {
            issues.push(format!(
                "expressiondirs.npy: expected shape [{n}, 3, E], found {e:?}"
            ));
        } else if s[2] + e[2] != spec.n_basis {
            issues.push(format!(
                "shapedirs.npy/expressiondirs.npy: expected {} basis columns in total, found {} + {}",
                spec.n_basis, s[2], e[2]
            ));
        }
    }

    Ok(issues)
}

/// FLAME 2020-specific validation.
fn validate_flame_2020(output_dir: &Path) -> Result<Vec<String>> {
    validate_flame_shapes(output_dir, &FLAME_SPEC)
}

/// FLAME 2023-specific validation.
fn validate_flame_2023(output_dir: &Path) -> Result<Vec<String>> {
    let issues = validate_flame_shapes(output_dir, &FLAME_SPEC)?;
    // FLAME 2023 ships an extended landmark embedding; it is optional for the
    // geometry pipeline, so its absence is reported but not treated as an error.
    if !output_dir.join("full_lmk_faces_idx.npy").exists() {
        tracing::info!("Note: FLAME 2023 full landmark files not found (optional)");
    }
    Ok(issues)
}

// ---------------------------------------------------------------------------
// NPY container
// ---------------------------------------------------------------------------

mod npy;

// ---------------------------------------------------------------------------
// Pure-Rust pickle virtual machine
// ---------------------------------------------------------------------------

mod pickle;

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn scratch_dir(tag: &str) -> std::path::PathBuf {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir =
            std::env::temp_dir().join(format!("oxigaf_convert_{tag}_{}_{n}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::create_dir_all(&dir);
        dir
    }

    fn zeros(descr: &str, shape: &[usize]) -> npy::NpyArray {
        let item = npy::item_size(descr).unwrap_or(4);
        let count: usize = shape.iter().product();
        npy::NpyArray {
            descr: descr.to_string(),
            shape: shape.to_vec(),
            data: vec![0u8; count * item],
        }
    }

    fn f32_array(shape: &[usize], values: &[f32]) -> npy::NpyArray {
        let mut data = Vec::with_capacity(values.len() * 4);
        for v in values {
            data.extend_from_slice(&v.to_le_bytes());
        }
        npy::NpyArray {
            descr: "<f4".to_string(),
            shape: shape.to_vec(),
            data,
        }
    }

    // -----------------------------------------------------------------------
    // Name / dtype mapping
    // -----------------------------------------------------------------------

    #[test]
    fn test_component_output_name_matches_flame_loader() {
        // The FLAME loader (oxigaf-flame/src/io.rs) reads exactly these names.
        assert_eq!(component_output_name("v_template"), "v_template.npy");
        assert_eq!(component_output_name("J_regressor"), "j_regressor.npy");
        assert_eq!(component_output_name("kintree_table"), "kintree_table.npy");
        assert_eq!(component_output_name("weights"), "lbs_weights.npy");
        assert_eq!(component_output_name("f"), "faces.npy");
        assert_eq!(component_output_name("vt"), "uv.npy");
        assert_eq!(component_output_name("ft"), "uv_faces.npy");
        assert_eq!(component_output_name("unknown"), "");
    }

    #[test]
    fn test_required_dtype() {
        assert_eq!(required_dtype("v_template.npy"), Some("<f4"));
        assert_eq!(required_dtype("faces.npy"), Some("<i4"));
        assert_eq!(required_dtype("kintree_table.npy"), Some("<i4"));
        assert_eq!(required_dtype("neck_kin_chain.npy"), None);
    }

    // -----------------------------------------------------------------------
    // NPY container
    // -----------------------------------------------------------------------

    #[test]
    fn test_extract_header_field() {
        let header = "{'descr': '<f4', 'fortran_order': False, 'shape': (5023, 3)}";
        assert_eq!(
            npy::extract_header_field(header, "descr"),
            Some("<f4".to_string())
        );
    }

    #[test]
    fn test_extract_shape_from_header() {
        let header = "{'descr': '<f4', 'fortran_order': False, 'shape': (5023, 3)}";
        assert_eq!(npy::extract_shape_from_header(header), vec![5023, 3]);
    }

    #[test]
    fn test_npy_round_trip() {
        let array = f32_array(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let encoded = match npy::serialize(&array) {
            Ok(bytes) => bytes,
            Err(e) => panic!("serialize failed: {e}"),
        };
        assert_eq!(&encoded[0..6], b"\x93NUMPY");
        assert_eq!(encoded.len() % 64, array.data.len() % 64);
        let decoded = match npy::parse(&encoded) {
            Ok(a) => a,
            Err(e) => panic!("parse failed: {e}"),
        };
        assert_eq!(decoded, array);
    }

    #[test]
    fn test_npy_version2_header_length_is_u32() {
        // Regression: the header length was always read as u16, so NPY 2.0
        // files (4-byte length) were misparsed and silently reported invalid.
        let mut header = "{'descr': '<f4', 'fortran_order': False, 'shape': (2,), }".to_string();
        let unpadded = 12 + header.len() + 1;
        let padding = (64 - (unpadded % 64)) % 64;
        for _ in 0..padding {
            header.push(' ');
        }
        header.push('\n');

        let mut stream = Vec::new();
        stream.extend_from_slice(b"\x93NUMPY");
        stream.push(2);
        stream.push(0);
        stream.extend_from_slice(&(header.len() as u32).to_le_bytes());
        stream.extend_from_slice(header.as_bytes());
        stream.extend_from_slice(&1.0f32.to_le_bytes());
        stream.extend_from_slice(&2.0f32.to_le_bytes());

        let parsed = match npy::parse_header(&stream) {
            Ok(h) => h,
            Err(e) => panic!("v2 header rejected: {e}"),
        };
        assert_eq!(parsed.descr, "<f4");
        assert_eq!(parsed.shape, vec![2]);
        assert_eq!(parsed.data_offset, 12 + header.len());
    }

    #[test]
    fn test_npy_fortran_order_is_normalised() {
        // Column-major [[1,2,3],[4,5,6]] stored as 1,4,2,5,3,6.
        let mut data = Vec::new();
        for v in [1.0f32, 4.0, 2.0, 5.0, 3.0, 6.0] {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let c_order = npy::fortran_to_c(&data, &[2, 3], 4);
        let array = npy::NpyArray {
            descr: "<f4".to_string(),
            shape: vec![2, 3],
            data: c_order,
        };
        let values = match npy::to_f64_vec(&array) {
            Ok(v) => v,
            Err(e) => panic!("to_f64_vec failed: {e}"),
        };
        assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_slice_last_axis() {
        let values: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let array = f32_array(&[2, 3, 4], &values);
        let head = match npy::slice_last_axis(&array, 0, 3) {
            Ok(a) => a,
            Err(e) => panic!("slice failed: {e}"),
        };
        let tail = match npy::slice_last_axis(&array, 3, 4) {
            Ok(a) => a,
            Err(e) => panic!("slice failed: {e}"),
        };
        assert_eq!(head.shape, vec![2, 3, 3]);
        assert_eq!(tail.shape, vec![2, 3, 1]);
        let tail_values = match npy::to_f64_vec(&tail) {
            Ok(v) => v,
            Err(e) => panic!("to_f64_vec failed: {e}"),
        };
        assert_eq!(tail_values, vec![3.0, 7.0, 11.0, 15.0, 19.0, 23.0]);
    }

    #[test]
    fn test_cast_between_dtypes() {
        let mut data = Vec::new();
        for v in [1.5f64, -2.5] {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let wide = npy::NpyArray {
            descr: "<f8".to_string(),
            shape: vec![2],
            data,
        };
        let narrow = match npy::cast(&wide, "<f4") {
            Ok(a) => a,
            Err(e) => panic!("cast failed: {e}"),
        };
        assert_eq!(narrow.descr, "<f4");
        assert_eq!(narrow.data.len(), 8);

        let mut unsigned = Vec::new();
        for v in [7u32, 9] {
            unsigned.extend_from_slice(&v.to_le_bytes());
        }
        let source = npy::NpyArray {
            descr: "<u4".to_string(),
            shape: vec![2],
            data: unsigned,
        };
        let signed = match npy::cast(&source, "<i4") {
            Ok(a) => a,
            Err(e) => panic!("cast failed: {e}"),
        };
        match npy::to_i64_vec(&signed) {
            Ok(v) => assert_eq!(v, vec![7, 9]),
            Err(e) => panic!("to_i64_vec failed: {e}"),
        }
    }

    // -----------------------------------------------------------------------
    // Verification
    // -----------------------------------------------------------------------

    #[test]
    fn test_validate_npy_file_rejects_garbage() {
        let dir = scratch_dir("garbage");
        let path = dir.join("v_template.npy");
        if let Err(e) = std::fs::write(&path, b"definitely not an npy file") {
            panic!("failed to write fixture: {e}");
        }
        match validate_npy_file(&path) {
            Ok(info) => assert!(!info.valid, "garbage must not be reported as valid"),
            Err(e) => panic!("validate_npy_file errored: {e}"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_verify_output_fails_on_invalid_files() {
        // Regression: verify_output used to log "INVALID" without clearing
        // `all_valid`, so a directory of correctly-named garbage passed.
        let dir = scratch_dir("verify");
        for name in OUTPUT_FILES {
            if let Err(e) = std::fs::write(dir.join(name), b"not an npy file at all") {
                panic!("failed to write fixture: {e}");
            }
        }
        assert!(
            verify_output(&dir, "2023").is_err(),
            "verification must fail for structurally invalid NPY files"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_verify_output_detects_missing_files() {
        let dir = scratch_dir("missing");
        assert!(verify_output(&dir, "2020").is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    // -----------------------------------------------------------------------
    // Output preparation
    // -----------------------------------------------------------------------

    #[test]
    fn test_prepare_outputs_splits_basis_and_matches_loader_names() {
        let mut components = HashMap::new();
        components.insert("v_template".to_string(), zeros("<f8", &[2, 3]));
        components.insert("f".to_string(), zeros("<u4", &[1, 3]));
        components.insert("shapedirs".to_string(), zeros("<f8", &[2, 3, 400]));
        components.insert("posedirs".to_string(), zeros("<f8", &[2, 3, 36]));
        components.insert("J_regressor".to_string(), zeros("<f8", &[5, 2]));
        components.insert("kintree_table".to_string(), zeros("<u4", &[2, 5]));
        components.insert("weights".to_string(), zeros("<f8", &[2, 5]));

        let files = match prepare_outputs(&components, false) {
            Ok(f) => f,
            Err(e) => panic!("prepare_outputs failed: {e}"),
        };
        let names: Vec<&str> = files.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, OUTPUT_FILES.to_vec());

        for (name, array) in &files {
            match name.as_str() {
                "shapedirs.npy" => {
                    assert_eq!(array.shape, vec![2, 3, 300]);
                    assert_eq!(array.descr, "<f4");
                }
                "expressiondirs.npy" => {
                    assert_eq!(array.shape, vec![2, 3, 100]);
                    assert_eq!(array.descr, "<f4");
                }
                "faces.npy" | "kintree_table.npy" => assert_eq!(array.descr, "<i4"),
                _ => assert_eq!(array.descr, "<f4"),
            }
            let expected_bytes = array.element_count() * npy::item_size(&array.descr).unwrap_or(0);
            assert_eq!(array.data.len(), expected_bytes);
        }
    }

    #[test]
    fn test_prepare_outputs_honours_include_uv() {
        let mut components = HashMap::new();
        components.insert("v_template".to_string(), zeros("<f4", &[2, 3]));
        components.insert("f".to_string(), zeros("<i4", &[1, 3]));
        components.insert("shapedirs".to_string(), zeros("<f4", &[2, 3, 400]));
        components.insert("posedirs".to_string(), zeros("<f4", &[2, 3, 36]));
        components.insert("J_regressor".to_string(), zeros("<f4", &[5, 2]));
        components.insert("kintree_table".to_string(), zeros("<i4", &[2, 5]));
        components.insert("weights".to_string(), zeros("<f4", &[2, 5]));
        components.insert("vt".to_string(), zeros("<f4", &[4, 2]));
        components.insert("ft".to_string(), zeros("<i4", &[1, 3]));

        let without = match prepare_outputs(&components, false) {
            Ok(f) => f,
            Err(e) => panic!("prepare_outputs failed: {e}"),
        };
        assert!(!without.iter().any(|(n, _)| n == "uv.npy"));

        let with = match prepare_outputs(&components, true) {
            Ok(f) => f,
            Err(e) => panic!("prepare_outputs failed: {e}"),
        };
        assert!(with.iter().any(|(n, _)| n == "uv.npy"));
        assert!(with.iter().any(|(n, _)| n == "uv_faces.npy"));
    }

    // -----------------------------------------------------------------------
    // Pickle machine
    // -----------------------------------------------------------------------

    fn push_global(out: &mut Vec<u8>, module: &str, name: &str) {
        out.push(b'c');
        out.extend_from_slice(module.as_bytes());
        out.push(b'\n');
        out.extend_from_slice(name.as_bytes());
        out.push(b'\n');
    }

    fn push_short_binstring(out: &mut Vec<u8>, payload: &[u8]) {
        out.push(b'U');
        out.push(payload.len() as u8);
        out.extend_from_slice(payload);
    }

    /// Memo slot holding the shared `numpy.dtype('>f4')` object.
    const DTYPE_SLOT: u8 = 0x0b;

    /// Emit `numpy.dtype('>f4')`, memoise it, then `BUILD` its byte order in.
    ///
    /// Real CPython pickles memoise the dtype *before* `BUILD` supplies the
    /// byte order, so a memo that snapshots instead of sharing loses `>`.
    fn push_big_endian_dtype(out: &mut Vec<u8>) {
        push_global(out, "numpy", "dtype");
        push_short_binstring(out, b"f4");
        out.extend_from_slice(b"K\x00K\x01");
        out.push(b'\x87'); // TUPLE3 -> ('f4', 0, 1)
        out.push(b'R'); // REDUCE -> bare dtype
        out.push(b'q'); // BINPUT
        out.push(DTYPE_SLOT);
        out.push(b'('); // MARK
        out.extend_from_slice(b"K\x03");
        push_short_binstring(out, b">");
        out.extend_from_slice(b"NNN");
        out.extend_from_slice(b"J\xff\xff\xff\xff");
        out.extend_from_slice(b"J\xff\xff\xff\xff");
        out.extend_from_slice(b"K\x00");
        out.push(b't'); // TUPLE -> dtype state
        out.push(b'b'); // BUILD dtype
    }

    /// Emit one `numpy.ndarray` value of shape `[rows, cols]`.
    fn push_array(out: &mut Vec<u8>, rows: u8, cols: u8, raw: &[u8], reuse_dtype: bool) {
        // numpy.core.multiarray._reconstruct(numpy.ndarray, (0,), b'b')
        push_global(out, "numpy.core.multiarray", "_reconstruct");
        push_global(out, "numpy", "ndarray");
        out.extend_from_slice(b"K\x00"); // BININT1 0
        out.push(b'\x85'); // TUPLE1 -> (0,)
        push_short_binstring(out, b"b");
        out.push(b'\x87'); // TUPLE3
        out.push(b'R'); // REDUCE -> bare ndarray

        // __setstate__ = (1, (rows, cols), dtype, False, raw)
        out.push(b'('); // MARK
        out.extend_from_slice(b"K\x01");
        out.push(b'K');
        out.push(rows);
        out.push(b'K');
        out.push(cols);
        out.push(b'\x86'); // TUPLE2 -> (rows, cols)
        if reuse_dtype {
            out.push(b'h'); // BINGET — shares the memoised dtype
            out.push(DTYPE_SLOT);
        } else {
            push_big_endian_dtype(out);
        }
        out.push(b'\x89'); // NEWFALSE (fortran_order)
        push_short_binstring(out, raw);
        out.push(b't'); // TUPLE -> array state
        out.push(b'b'); // BUILD array
    }

    /// Big-endian float32 payload, as numpy stores a `>f4` buffer.
    fn be_f32(values: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(values.len() * 4);
        for value in values {
            out.extend_from_slice(&value.to_be_bytes());
        }
        out
    }

    /// Build a protocol-2 pickle of two arrays sharing one memoised dtype.
    fn build_flame_like_pickle(first: &[f32], second: &[f32]) -> Vec<u8> {
        let mut p = Vec::new();
        p.extend_from_slice(b"\x80\x02"); // PROTO 2
        p.push(b'}'); // EMPTY_DICT
        p.extend_from_slice(b"q\x00"); // BINPUT 0

        push_short_binstring(&mut p, b"v_template");
        push_array(&mut p, 2, 3, &be_f32(first), false);
        p.push(b's'); // SETITEM

        push_short_binstring(&mut p, b"weights");
        push_array(&mut p, 2, 3, &be_f32(second), true);
        p.push(b's'); // SETITEM

        p.push(b'.'); // STOP
        p
    }

    #[test]
    fn test_pickle_decodes_numpy_arrays_sharing_a_memoised_dtype() {
        // Regression: extract_numpy_array was a hardcoded `None`, so every
        // `.pkl` conversion failed with "Could not parse pickle file".
        // The second entry additionally pins the memo/BUILD interaction: it
        // reaches its dtype through BINGET, so a snapshotting memo would drop
        // the big-endian byte order and byte-swap every value.
        let first = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let second = [-1.5f32, 0.5, 2.25, 4.0, 8.0, 16.0];
        let stream = build_flame_like_pickle(&first, &second);
        let arrays = match pickle::load_arrays(&stream) {
            Ok(a) => a,
            Err(e) => panic!("pickle decoding failed: {e:#}"),
        };

        for (name, expected) in [
            ("v_template", first.as_slice()),
            ("weights", second.as_slice()),
        ] {
            let array = match arrays.get(name) {
                Some(a) => a,
                None => panic!("{name} missing from decoded pickle"),
            };
            assert_eq!(array.descr, ">f4", "{name} lost its dtype byte order");
            assert_eq!(array.shape, vec![2, 3]);
            let values: Vec<f64> = expected.iter().map(|v| f64::from(*v)).collect();
            match npy::to_f64_vec(array) {
                Ok(v) => assert_eq!(v, values, "{name} decoded to the wrong values"),
                Err(e) => panic!("to_f64_vec failed: {e}"),
            }
        }
    }

    #[test]
    fn test_pickle_rejects_truncated_stream() {
        let stream = build_flame_like_pickle(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1.0; 6]);
        let truncated = &stream[..stream.len() / 2];
        assert!(pickle::load_arrays(truncated).is_err());
    }

    #[test]
    fn test_densify_csc_is_not_transposed() {
        // 2x3 matrix with 4.0 at (0, 2) and 7.0 at (1, 0), stored as CSC.
        // indptr has one entry per column plus one.
        let values = [7.0f64, 4.0];
        let indices = [1i64, 0];
        let indptr = [0i64, 1, 1, 2];
        let dense = match pickle::densify(2, 3, &values, &indices, &indptr, true) {
            Ok(a) => a,
            Err(e) => panic!("densify failed: {e}"),
        };
        assert_eq!(dense.shape, vec![2, 3]);
        match npy::to_f64_vec(&dense) {
            Ok(v) => assert_eq!(v, vec![0.0, 0.0, 4.0, 7.0, 0.0, 0.0]),
            Err(e) => panic!("to_f64_vec failed: {e}"),
        }
    }

    #[test]
    fn test_densify_rejects_out_of_range_indices() {
        let values = [1.0f64];
        let indices = [9i64];
        let indptr = [0i64, 1, 1, 1];
        assert!(pickle::densify(2, 3, &values, &indices, &indptr, true).is_err());
    }
}
