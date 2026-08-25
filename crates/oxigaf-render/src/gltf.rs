//! glTF 2.0 export for 3D Gaussian Splatting models.
//!
//! This module is the **single** glTF writer for the workspace. It replaces two
//! earlier, mutually incompatible emitters that both shipped under the nominal
//! format name `gltf`:
//!
//! * `oxigaf::pipeline::write_gltf` — a `.gltf` + `.bin` pair using the
//!   extension name `OXIGAF_gaussian_splat`,
//! * `oxigaf_cli::export_gltf::export_gltf` — a second `.gltf` + `.bin` pair,
//!   also `OXIGAF_gaussian_splat`, but with all five accessors crammed onto one
//!   buffer view,
//! * `oxigaf_cli::export::export_gltf` — a self-contained GLB using a *third*
//!   extension name, `OXIGAF_gaussians`.
//!
//! Two files that disagree about their own format are worse than either one
//! alone: a consumer written against one output silently mis-reads the other.
//! [`write_gltf`] below is the surviving implementation, hoisted here so both
//! `oxigaf` and `oxigaf-cli` call the same code.
//!
//! # Output
//!
//! Two files are produced from one `path`:
//!
//! * `path` — the JSON glTF 2.0 document,
//! * `path` with its extension replaced by `bin` — the little-endian `f32`
//!   attribute buffer, referenced by file name from the document.
//!
//! The buffer is written first, so a failed write never leaves a JSON document
//! pointing at a file that does not exist.
//!
//! # Binary buffer layout
//!
//! Five tightly-packed, 4-byte-aligned `f32` blocks, in this order:
//!
//! | Block       | Components | Bytes per Gaussian | Cumulative offset |
//! |-------------|------------|--------------------|-------------------|
//! | positions   | N×3        | 12                 | 0                 |
//! | rotations   | N×4        | 16                 | N×12              |
//! | scales      | N×3        | 12                 | N×28              |
//! | opacities   | N×1        | 4                  | N×40              |
//! | sh_coeffs   | N×C        | 4×C                | N×44              |
//!
//! where `C = (sh_degree + 1)² × 3`.
//!
//! # Spec conformance
//!
//! Three properties of this writer are load-bearing, and each one is why the
//! *other* implementation was not the one kept:
//!
//! 1. **One buffer view per accessor.** glTF 2.0 requires `byteStride` on any
//!    buffer view shared by accessors of different element sizes; giving each
//!    accessor its own tightly-packed view sidesteps the requirement entirely.
//!    The CLI writer put all five accessors on a single strideless view, which
//!    the specification forbids.
//! 2. **`min`/`max` on the `POSITION` accessor.** The schema makes these
//!    mandatory for any accessor referenced by a `POSITION` attribute;
//!    conforming loaders use them for frustum culling and auto-framing.
//! 3. **Empty models emit an asset-only document.** `accessor.count`,
//!    `bufferView.byteLength`, `buffer.byteLength`, `nodes` and `scenes` all
//!    carry a minimum of 1 in the schema, so a zero-Gaussian model cannot be
//!    described with zero-valued entries — those are simply omitted, and no
//!    `.bin` sidecar is written.

use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use crate::gaussian::GaussianModel;
use crate::RenderError;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Why a glTF export failed.
///
/// The two variants are deliberately distinct rather than a single message
/// string: a caller mistake (an output path that cannot work) and an
/// environment failure (a full disk, a read-only directory) call for different
/// handling and different exit codes, and the [`std::io::Error`] behind the
/// latter is worth preserving for callers that inspect its kind. `oxigaf`, for
/// instance, maps them onto `InvalidConfig` and `Io` respectively.
#[derive(Debug, thiserror::Error)]
pub enum GltfError {
    /// The output path cannot be used for a glTF document.
    ///
    /// Either it collides with its own `.bin` buffer sidecar (i.e. it already
    /// ends in `.bin`, so writing the buffer would clobber the document), or
    /// no buffer file name can be derived from it.
    #[error("invalid glTF output path: {0}")]
    InvalidOutputPath(String),

    /// A file could not be created, written or flushed.
    #[error("cannot {action} {path}: {source}")]
    Io {
        /// What was being attempted, e.g. `"create glTF document"`.
        action: &'static str,
        /// The file involved.
        path: PathBuf,
        /// The underlying failure.
        #[source]
        source: std::io::Error,
    },
}

impl From<GltfError> for RenderError {
    fn from(err: GltfError) -> Self {
        match err {
            GltfError::InvalidOutputPath(message) => RenderError::ValidationError(message),
            io @ GltfError::Io { .. } => RenderError::ValidationError(io.to_string()),
        }
    }
}

// ---------------------------------------------------------------------------
// glTF constants
// ---------------------------------------------------------------------------

/// glTF `componentType` value for 32-bit floats.
const COMPONENT_TYPE_FLOAT: u32 = 5126;

/// glTF primitive `mode` value for `POINTS`.
const MODE_POINTS: u32 = 0;

/// Name of the custom glTF extension carrying the Gaussian attributes.
///
/// Exposed so consumers (importers, tests, other tools in the workspace) can
/// refer to the one canonical spelling instead of repeating a string literal.
/// The two superseded writers used this name and `OXIGAF_gaussians`
/// respectively; only this one survives.
pub const EXTENSION_NAME: &str = "OXIGAF_gaussian_splat";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Number of spherical-harmonics floats stored per Gaussian: `(degree + 1)² × 3`.
fn sh_coeffs_per_gaussian(sh_degree: u32) -> usize {
    let bands = (sh_degree + 1) as usize;
    bands * bands * 3
}

/// Escape a string so it can be embedded inside a JSON string literal.
fn json_escape(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for ch in input.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

/// Append little-endian `f32` values to a byte buffer.
fn append_f32_le(buf: &mut Vec<u8>, values: &[f32]) {
    for value in values {
        buf.extend_from_slice(&value.to_le_bytes());
    }
}

/// Componentwise axis-aligned bounding box over the finite Gaussian centres.
///
/// Non-finite positions are skipped so a single `NaN` cannot poison the
/// `POSITION` accessor's `min`/`max` (which would make the whole document
/// unreadable to a conforming loader, not merely mis-framed). Returns `None`
/// when the model is empty or every position is non-finite.
fn bounding_box(model: &GaussianModel) -> Option<([f32; 3], [f32; 3])> {
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];

    for gaussian in &model.gaussians {
        let p = gaussian.position;
        if !p[0].is_finite() || !p[1].is_finite() || !p[2].is_finite() {
            continue;
        }
        for k in 0..3 {
            min[k] = min[k].min(p[k]);
            max[k] = max[k].max(p[k]);
        }
    }

    if min.iter().all(|v| v.is_finite()) && max.iter().all(|v| v.is_finite()) {
        Some((min, max))
    } else {
        None
    }
}

/// Derive the `.bin` sidecar path and its file name from the document path.
///
/// # Errors
///
/// Returns [`GltfError::InvalidOutputPath`] when the sidecar path would be the
/// document path itself (i.e. `path` already ends in `.bin`, so the buffer
/// write would clobber the document), or when `path` has no usable file name.
fn bin_sidecar(path: &Path) -> Result<(PathBuf, String), GltfError> {
    let bin_path = path.with_extension("bin");
    if bin_path.as_path() == path {
        return Err(GltfError::InvalidOutputPath(format!(
            "{} collides with its own .bin buffer sidecar; \
             use a different extension such as .gltf",
            path.display()
        )));
    }
    let bin_name = bin_path
        .file_name()
        .and_then(|s| s.to_str())
        .ok_or_else(|| {
            GltfError::InvalidOutputPath(format!(
                "cannot derive a buffer file name from {}",
                path.display()
            ))
        })?
        .to_string();
    Ok((bin_path, bin_name))
}

/// Build a [`GltfError::Io`] that names both the attempted action and the file.
fn io_error(action: &'static str, path: &Path, source: std::io::Error) -> GltfError {
    GltfError::Io {
        action,
        path: path.to_path_buf(),
        source,
    }
}

// ---------------------------------------------------------------------------
// Writer
// ---------------------------------------------------------------------------

/// Write `model` as a glTF 2.0 document plus its binary buffer sidecar.
///
/// See the [module documentation](self) for the file layout, the binary buffer
/// layout, and the specification requirements this writer satisfies.
///
/// Positions are exposed through the standard `POSITION` attribute of a
/// `POINTS` primitive, so ordinary glTF viewers show the model as a point
/// cloud. Attributes with no standard glTF equivalent — rotation, scale,
/// opacity and the SH coefficients — are reachable through the custom
/// [`EXTENSION_NAME`] extension, which records their accessor indices.
///
/// Parent directories are **not** created; callers that accept arbitrary
/// output paths should create them first.
///
/// # Errors
///
/// Returns [`GltfError::InvalidOutputPath`] when `path` collides with its own
/// `.bin` sidecar or yields no usable buffer file name, and [`GltfError::Io`]
/// when either file cannot be created, written or flushed.
pub fn write_gltf(model: &GaussianModel, path: &Path) -> Result<(), GltfError> {
    let count = model.len();
    let sh_degree = model.sh_degree;

    // An empty model has no buffer to point at, and glTF forbids zero-length
    // buffers and buffer views. `scene`, `scenes` and `nodes` are optional but
    // may not be empty arrays (`minItems: 1`), so omit them entirely and emit
    // an asset-only document. No `.bin` sidecar is produced.
    if count == 0 {
        let file =
            std::fs::File::create(path).map_err(|e| io_error("create glTF document", path, e))?;
        let mut w = BufWriter::new(file);
        write_empty_document(&mut w, sh_degree)
            .map_err(|e| io_error("write glTF document", path, e))?;
        return Ok(());
    }

    // Reject before writing anything, so a bad path leaves no partial output.
    let (bin_path, bin_name) = bin_sidecar(path)?;

    let stride = sh_coeffs_per_gaussian(sh_degree);
    let sh_len = count * stride;

    // --- Binary buffer: five tightly-packed f32 blocks, all 4-byte aligned ---
    let mut bin = Vec::<u8>::with_capacity(count * (3 + 4 + 3 + 1 + stride) * 4);

    let positions_offset = bin.len();
    for gaussian in &model.gaussians {
        append_f32_le(&mut bin, &gaussian.position);
    }
    let rotations_offset = bin.len();
    for gaussian in &model.gaussians {
        append_f32_le(&mut bin, &gaussian.rotation);
    }
    let scales_offset = bin.len();
    for gaussian in &model.gaussians {
        append_f32_le(&mut bin, &gaussian.scale);
    }
    let opacities_offset = bin.len();
    for gaussian in &model.gaussians {
        append_f32_le(&mut bin, &[gaussian.opacity]);
    }
    let sh_offset = bin.len();
    let mut sh_coeffs = model.sh_coeffs.clone();
    // Pad or trim so the accessor count always matches the declared stride,
    // even for a model whose `sh_coeffs` disagrees with its `sh_degree`.
    sh_coeffs.resize(sh_len, 0.0);
    append_f32_le(&mut bin, &sh_coeffs);
    let bin_len = bin.len();

    // Buffer first: a failed write must not leave a document pointing at a
    // file that is missing or truncated.
    {
        let file = std::fs::File::create(&bin_path)
            .map_err(|e| io_error("create glTF buffer", &bin_path, e))?;
        let mut w = BufWriter::new(file);
        w.write_all(&bin)
            .and_then(|()| w.flush())
            .map_err(|e| io_error("write glTF buffer", &bin_path, e))?;
    }

    let offsets = BlockOffsets {
        positions: positions_offset,
        rotations: rotations_offset,
        scales: scales_offset,
        opacities: opacities_offset,
        sh: sh_offset,
    };
    let (min, max) = bounding_box(model).unwrap_or(([0.0; 3], [0.0; 3]));

    let file =
        std::fs::File::create(path).map_err(|e| io_error("create glTF document", path, e))?;
    let mut w = BufWriter::new(file);
    write_document(
        &mut w,
        &DocumentParams {
            count,
            sh_degree,
            sh_len,
            offsets,
            bin_name: &bin_name,
            bin_len,
            position_min: min,
            position_max: max,
        },
    )
    .map_err(|e| io_error("write glTF document", path, e))?;

    tracing::info!(
        gaussians = count,
        sh_degree,
        document = %path.display(),
        buffer = %bin_path.display(),
        "wrote glTF 2.0"
    );

    Ok(())
}

/// Byte offsets of the five attribute blocks within the binary buffer.
#[derive(Debug, Clone, Copy)]
struct BlockOffsets {
    /// Offset of the `N×3` position block.
    positions: usize,
    /// Offset of the `N×4` rotation block.
    rotations: usize,
    /// Offset of the `N×3` scale block.
    scales: usize,
    /// Offset of the `N×1` opacity block.
    opacities: usize,
    /// Offset of the `N×C` spherical-harmonics block.
    sh: usize,
}

/// Everything the JSON document needs that is not derivable inside the writer.
#[derive(Debug, Clone, Copy)]
struct DocumentParams<'a> {
    /// Number of Gaussians; always `>= 1` here.
    count: usize,
    /// Spherical-harmonics degree of the model.
    sh_degree: u32,
    /// Total number of SH floats, `count × (sh_degree + 1)² × 3`.
    sh_len: usize,
    /// Byte offsets of the five attribute blocks.
    offsets: BlockOffsets,
    /// File name of the `.bin` sidecar, as referenced by `buffers[0].uri`.
    bin_name: &'a str,
    /// Byte length of the `.bin` sidecar.
    bin_len: usize,
    /// Componentwise minimum of the Gaussian centres.
    position_min: [f32; 3],
    /// Componentwise maximum of the Gaussian centres.
    position_max: [f32; 3],
}

/// Write the opening brace and `asset` block shared by every glTF document.
fn write_header<W: Write>(w: &mut W) -> std::io::Result<()> {
    writeln!(w, "{{")?;
    writeln!(
        w,
        r#"  "asset": {{ "version": "2.0", "generator": "OxiGAF v{}" }},"#,
        json_escape(env!("CARGO_PKG_VERSION"))
    )
}

/// Write the asset-only document a zero-Gaussian model produces.
fn write_empty_document<W: Write>(w: &mut W, sh_degree: u32) -> std::io::Result<()> {
    write_header(w)?;
    writeln!(w, r#"  "extensionsUsed": ["{EXTENSION_NAME}"],"#)?;
    writeln!(
        w,
        r#"  "extensions": {{ "{EXTENSION_NAME}": {{ "gaussianCount": 0, "shDegree": {sh_degree} }} }}"#
    )?;
    writeln!(w, "}}")?;
    w.flush()
}

/// Write the full document for a model with at least one Gaussian.
fn write_document<W: Write>(w: &mut W, params: &DocumentParams<'_>) -> std::io::Result<()> {
    let DocumentParams {
        count,
        sh_degree,
        sh_len,
        offsets,
        bin_name,
        bin_len,
        position_min,
        position_max,
    } = *params;

    write_header(w)?;
    writeln!(w, r#"  "extensionsUsed": ["{EXTENSION_NAME}"],"#)?;
    writeln!(w, r#"  "scene": 0,"#)?;
    writeln!(
        w,
        r#"  "scenes": [{{ "name": "GaussianScene", "nodes": [0] }}],"#
    )?;
    writeln!(w, r#"  "nodes": [{{"#)?;
    writeln!(w, r#"    "name": "GaussianSplat","#)?;
    writeln!(w, r#"    "mesh": 0,"#)?;
    writeln!(w, r#"    "extensions": {{ "{EXTENSION_NAME}": {{"#)?;
    writeln!(w, r#"      "gaussianCount": {count},"#)?;
    writeln!(w, r#"      "shDegree": {sh_degree},"#)?;
    writeln!(w, r#"      "positionsAccessor": 0,"#)?;
    writeln!(w, r#"      "rotationsAccessor": 1,"#)?;
    writeln!(w, r#"      "scalesAccessor": 2,"#)?;
    writeln!(w, r#"      "opacitiesAccessor": 3,"#)?;
    writeln!(w, r#"      "shCoefficientsAccessor": 4"#)?;
    writeln!(w, r#"    }} }}"#)?;
    writeln!(w, r#"  }}],"#)?;
    writeln!(
        w,
        r#"  "meshes": [{{ "name": "GaussianMesh", "primitives": [{{ "attributes": {{ "POSITION": 0 }}, "mode": {MODE_POINTS} }}] }}],"#
    )?;
    writeln!(
        w,
        r#"  "buffers": [{{ "uri": "{}", "byteLength": {bin_len} }}],"#,
        json_escape(bin_name)
    )?;

    // One buffer view per accessor, each tightly packed: no `byteStride` is
    // needed, and none of the views is shared between differently-sized
    // elements (which glTF 2.0 forbids without a stride).
    writeln!(w, r#"  "bufferViews": ["#)?;
    write_buffer_view(w, offsets.positions, count * 3 * 4, true)?;
    write_buffer_view(w, offsets.rotations, count * 4 * 4, true)?;
    write_buffer_view(w, offsets.scales, count * 3 * 4, true)?;
    write_buffer_view(w, offsets.opacities, count * 4, true)?;
    write_buffer_view(w, offsets.sh, sh_len * 4, false)?;
    writeln!(w, r#"  ],"#)?;

    // The POSITION accessor is required by the glTF spec to carry min/max.
    writeln!(w, r#"  "accessors": ["#)?;
    writeln!(
        w,
        r#"    {{ "bufferView": 0, "byteOffset": 0, "componentType": {COMPONENT_TYPE_FLOAT}, "count": {count}, "type": "VEC3", "min": [{min_x}, {min_y}, {min_z}], "max": [{max_x}, {max_y}, {max_z}] }},"#,
        min_x = position_min[0],
        min_y = position_min[1],
        min_z = position_min[2],
        max_x = position_max[0],
        max_y = position_max[1],
        max_z = position_max[2]
    )?;
    write_accessor(w, 1, count, "VEC4", true)?;
    write_accessor(w, 2, count, "VEC3", true)?;
    write_accessor(w, 3, count, "SCALAR", true)?;
    write_accessor(w, 4, sh_len, "SCALAR", false)?;
    writeln!(w, r#"  ],"#)?;

    writeln!(
        w,
        r#"  "extensions": {{ "{EXTENSION_NAME}": {{ "gaussianCount": {count}, "shDegree": {sh_degree} }} }}"#
    )?;
    writeln!(w, "}}")?;
    w.flush()
}

/// Write one glTF `bufferView` entry over the single binary buffer.
///
/// `trailing_comma` must be `false` for the final entry of the array.
fn write_buffer_view<W: Write>(
    w: &mut W,
    byte_offset: usize,
    byte_length: usize,
    trailing_comma: bool,
) -> std::io::Result<()> {
    let comma = if trailing_comma { "," } else { "" };
    writeln!(
        w,
        r#"    {{ "buffer": 0, "byteOffset": {byte_offset}, "byteLength": {byte_length} }}{comma}"#
    )
}

/// Write one glTF `accessor` entry that owns buffer view `buffer_view`.
///
/// `trailing_comma` must be `false` for the final entry of the array.
fn write_accessor<W: Write>(
    w: &mut W,
    buffer_view: usize,
    count: usize,
    kind: &str,
    trailing_comma: bool,
) -> std::io::Result<()> {
    let comma = if trailing_comma { "," } else { "" };
    writeln!(
        w,
        r#"    {{ "bufferView": {buffer_view}, "byteOffset": 0, "componentType": {COMPONENT_TYPE_FLOAT}, "count": {count}, "type": "{kind}" }}{comma}"#
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gaussian::GaussianAttributes;

    /// Create a fresh temporary directory dedicated to one test.
    fn temp_subdir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("oxigaf_render_gltf_{name}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("test: temp dir creation should succeed");
        dir
    }

    /// A deterministic model of `n` Gaussians at `[i, i + 0.1, i + 0.2]`.
    fn sample_model(n: usize, sh_degree: u32) -> GaussianModel {
        let stride = sh_coeffs_per_gaussian(sh_degree);
        let gaussians = (0..n)
            .map(|i| {
                let f = i as f32;
                GaussianAttributes {
                    position: [f, f + 0.1, f + 0.2],
                    _pad0: 0.0,
                    rotation: [0.0, 0.0, 0.0, 1.0],
                    scale: [-2.0, -2.0, -2.0],
                    opacity: 0.5,
                }
            })
            .collect();
        GaussianModel {
            gaussians,
            sh_coeffs: vec![0.25; n * stride],
            sh_degree,
            face_indices: vec![0; n],
            barycentric: vec![[1.0 / 3.0; 3]; n],
            local_offsets: vec![[0.0; 3]; n],
            is_rigid: vec![false; n],
        }
    }

    /// Read a whole file as text, failing the test on error.
    fn read_text(path: &Path) -> String {
        std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("test: could not read {}: {e}", path.display()))
    }

    /// Extract the integer value of the *first* `"key": <int>` occurrence.
    fn first_int_field(text: &str, key: &str) -> Option<i64> {
        let needle = format!("\"{key}\":");
        let start = text.find(&needle)? + needle.len();
        let rest = text[start..].trim_start();
        let end = rest
            .find(|c: char| !c.is_ascii_digit() && c != '-')
            .unwrap_or(rest.len());
        rest[..end].parse().ok()
    }

    // -----------------------------------------------------------------------
    // File pair
    // -----------------------------------------------------------------------

    #[test]
    fn writes_a_document_and_its_binary_buffer() {
        let dir = temp_subdir("pair");
        let dst = dir.join("scene.gltf");

        write_gltf(&sample_model(10, 1), &dst).expect("test: export should succeed");

        assert!(dst.is_file(), "the glTF document must exist");
        assert!(
            dir.join("scene.bin").is_file(),
            "the .bin buffer sidecar must exist"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn binary_buffer_is_n_times_44_plus_the_sh_block() {
        let dir = temp_subdir("bin_size");
        let dst = dir.join("scene.gltf");
        let n = 20usize;
        let sh_degree = 3u32;

        write_gltf(&sample_model(n, sh_degree), &dst).expect("test: export should succeed");

        // positions(12) + rotations(16) + scales(12) + opacities(4) = 44 B
        let expected = n * 44 + n * sh_coeffs_per_gaussian(sh_degree) * 4;
        let actual = std::fs::metadata(dir.join("scene.bin"))
            .expect("test: bin file should exist")
            .len();
        assert_eq!(actual as usize, expected, "bin size must be N*44 + N*C*4");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn document_declares_the_binary_buffers_real_length() {
        let dir = temp_subdir("declared_len");
        let dst = dir.join("scene.gltf");
        write_gltf(&sample_model(7, 2), &dst).expect("test: export should succeed");

        let declared = first_int_field(&read_text(&dst), "byteLength")
            .expect("test: buffers[0].byteLength must be present");
        let on_disk = std::fs::metadata(dir.join("scene.bin"))
            .expect("test: bin file should exist")
            .len();
        assert_eq!(
            declared as u64, on_disk,
            "the declared buffer length must match the file on disk"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    // -----------------------------------------------------------------------
    // Spec conformance
    // -----------------------------------------------------------------------

    #[test]
    fn every_accessor_owns_its_own_buffer_view() {
        // Regression for the superseded CLI writer, which put all five
        // accessors on ONE strideless buffer view — forbidden by glTF 2.0 for
        // accessors of differing element size. Five accessors must map onto
        // five distinct buffer views, in order.
        let dir = temp_subdir("one_view_each");
        let dst = dir.join("scene.gltf");
        write_gltf(&sample_model(5, 1), &dst).expect("test: export should succeed");

        let text = read_text(&dst);
        let views: Vec<&str> = text
            .lines()
            .filter(|line| line.contains(r#""buffer": 0"#))
            .collect();
        assert_eq!(views.len(), 5, "expected five buffer views, got {views:?}");

        let referenced: Vec<i64> = text
            .lines()
            .filter(|line| line.contains(r#""bufferView":"#))
            .filter_map(|line| first_int_field(line, "bufferView"))
            .collect();
        assert_eq!(
            referenced,
            vec![0, 1, 2, 3, 4],
            "each accessor must reference its own buffer view"
        );

        // No stride is needed precisely because the views are not shared.
        assert!(
            !text.contains("byteStride"),
            "tightly-packed per-accessor views need no byteStride"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn position_accessor_carries_min_and_max() {
        let dir = temp_subdir("min_max");
        let dst = dir.join("scene.gltf");
        // sample_model places Gaussian i at [i, i+0.1, i+0.2], so 5 Gaussians
        // span min=[0, 0.1, 0.2] to max=[4, 4.1, 4.2].
        write_gltf(&sample_model(5, 1), &dst).expect("test: export should succeed");

        let text = read_text(&dst);
        assert!(
            text.contains(r#""min": [0, 0.1, 0.2]"#),
            "POSITION accessor must carry the real componentwise min:\n{text}"
        );
        assert!(
            text.contains(r#""max": [4, 4.1, 4.2]"#),
            "POSITION accessor must carry the real componentwise max:\n{text}"
        );
        // The schema requires min/max only for POSITION; the other four
        // accessors must not carry them, so exactly one of each appears.
        assert_eq!(text.matches(r#""min":"#).count(), 1);
        assert_eq!(text.matches(r#""max":"#).count(), 1);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn non_finite_positions_do_not_poison_the_bounding_box() {
        // A NaN centre must be skipped rather than written into min/max: a
        // `NaN` literal is not valid JSON, so it would make the whole document
        // unparseable instead of merely mis-framed.
        let dir = temp_subdir("nan_bbox");
        let dst = dir.join("scene.gltf");
        let mut model = sample_model(3, 0);
        model.gaussians[1].position = [f32::NAN, f32::INFINITY, 0.0];

        write_gltf(&model, &dst).expect("test: export should succeed");

        let text = read_text(&dst);
        assert!(
            !text.contains("NaN") && !text.contains("inf"),
            "non-finite values must never reach the JSON:\n{text}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn empty_model_emits_an_asset_only_document_and_no_buffer() {
        // glTF forbids zero-length buffers/buffer views and empty
        // `nodes`/`scenes` arrays, so a zero-Gaussian model must omit them.
        let dir = temp_subdir("empty");
        let dst = dir.join("empty.gltf");

        write_gltf(&sample_model(0, 1), &dst).expect("test: export should succeed");

        assert!(dst.is_file(), "the glTF document must exist");
        assert!(
            !dir.join("empty.bin").exists(),
            "no zero-length .bin sidecar may be written"
        );

        let text = read_text(&dst);
        for forbidden in [
            "\"nodes\"",
            "\"scenes\"",
            "\"accessors\"",
            "\"bufferViews\"",
            "\"buffers\"",
            "\"meshes\"",
        ] {
            assert!(
                !text.contains(forbidden),
                "an empty model must omit {forbidden} entirely, not emit an empty array:\n{text}"
            );
        }
        assert!(
            text.contains(r#""gaussianCount": 0"#),
            "the extension must still report the count:\n{text}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn refuses_to_clobber_its_own_buffer() {
        let dir = temp_subdir("bin_collision");
        let dst = dir.join("scene.bin");

        let err = write_gltf(&sample_model(4, 0), &dst)
            .expect_err("test: a .bin document path must be rejected");
        assert!(
            matches!(&err, GltfError::InvalidOutputPath(msg) if msg.contains("collides")),
            "got: {err}"
        );
        assert!(
            !dst.exists(),
            "a rejected path must leave no partial output behind"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn io_failures_are_reported_as_io_not_as_a_bad_path() {
        // A caller mistake and a filesystem failure must stay distinguishable:
        // collapsing both into one message string is what would let a caller
        // report "check your output path" for a full disk. Writing into a
        // directory that does not exist is the portable stand-in for the
        // latter.
        //
        // The failure surfaces on the *buffer*, not the document, because the
        // buffer is written first by design — see `write_gltf`. That ordering
        // is the reason a failed export never leaves a document pointing at a
        // missing file, so pinning it here guards it too.
        let dir = temp_subdir("io_failure");
        let missing = dir.join("absent-subdir");
        let dst = missing.join("scene.gltf");

        let err = write_gltf(&sample_model(3, 0), &dst)
            .expect_err("test: writing into a missing directory must fail");
        match &err {
            GltfError::Io { action, path, .. } => {
                assert!(action.contains("glTF"), "unhelpful action: {action}");
                assert_eq!(
                    path,
                    &missing.join("scene.bin"),
                    "the buffer is written first, so it is the file that fails"
                );
            }
            other => panic!("expected an Io error, got: {other}"),
        }
        assert!(
            !dst.exists(),
            "a failed buffer write must leave no document behind"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn json_escape_handles_quotes_backslashes_and_controls() {
        assert_eq!(json_escape("plain"), "plain");
        assert_eq!(json_escape("a\"b"), "a\\\"b");
        assert_eq!(json_escape("a\\b"), "a\\\\b");
        assert_eq!(json_escape("a\nb"), "a\\nb");
        assert_eq!(json_escape("a\rb"), "a\\rb");
        assert_eq!(json_escape("a\tb"), "a\\tb");
        assert_eq!(json_escape("a\u{1}b"), "a\\u0001b");
    }

    #[test]
    fn sh_coeffs_per_gaussian_matches_the_ply_layout() {
        assert_eq!(sh_coeffs_per_gaussian(0), 3);
        assert_eq!(sh_coeffs_per_gaussian(1), 12);
        assert_eq!(sh_coeffs_per_gaussian(2), 27);
        assert_eq!(sh_coeffs_per_gaussian(3), 48);
    }

    #[test]
    fn both_error_kinds_convert_into_render_error() {
        // `oxigaf` and `oxigaf-cli` each map `GltfError` onto their own error
        // type; the blanket conversion into `RenderError` keeps the writer
        // usable from any `RenderError`-returning context too.
        let path_err = GltfError::InvalidOutputPath("bad path".to_string());
        assert!(matches!(
            RenderError::from(path_err),
            RenderError::ValidationError(_)
        ));

        let io_err = GltfError::Io {
            action: "create glTF document",
            path: PathBuf::from("/nonexistent/scene.gltf"),
            source: std::io::Error::other("disk full"),
        };
        let converted = RenderError::from(io_err);
        assert!(
            converted.to_string().contains("disk full"),
            "the underlying cause must survive the conversion: {converted}"
        );
    }

    // -----------------------------------------------------------------------
    // Well-formedness and metadata
    // -----------------------------------------------------------------------

    #[test]
    fn emitted_json_is_balanced() {
        for (count, sh_degree) in [(0usize, 1u32), (1, 0), (12, 3)] {
            let dir = temp_subdir(&format!("balanced_{count}_{sh_degree}"));
            let dst = dir.join("scene.gltf");
            write_gltf(&sample_model(count, sh_degree), &dst).expect("test: export should succeed");

            let text = read_text(&dst);
            assert_eq!(
                text.matches('{').count(),
                text.matches('}').count(),
                "unbalanced braces for count={count}:\n{text}"
            );
            assert_eq!(
                text.matches('[').count(),
                text.matches(']').count(),
                "unbalanced brackets for count={count}:\n{text}"
            );
            assert!(
                !text.contains(",\n}"),
                "a trailing comma before a closing brace is invalid JSON:\n{text}"
            );

            let _ = std::fs::remove_dir_all(&dir);
        }
    }

    #[test]
    fn extension_name_is_the_single_canonical_spelling() {
        // The whole point of this module: one format name, one extension name.
        // The superseded GLB writer used `OXIGAF_gaussians`, which must not
        // appear anywhere in this writer's output.
        let dir = temp_subdir("extension_name");
        let dst = dir.join("scene.gltf");
        write_gltf(&sample_model(3, 0), &dst).expect("test: export should succeed");

        let text = read_text(&dst);
        assert_eq!(EXTENSION_NAME, "OXIGAF_gaussian_splat");
        assert!(
            text.contains(r#""extensionsUsed": ["OXIGAF_gaussian_splat"]"#),
            "the document must declare the canonical extension:\n{text}"
        );
        assert!(
            !text.contains("OXIGAF_gaussians\""),
            "the superseded GLB extension name must not appear:\n{text}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn metadata_reports_the_models_count_and_sh_degree() {
        let dir = temp_subdir("metadata");
        let dst = dir.join("scene.gltf");
        let n = 42usize;
        let sh_degree = 2u32;
        write_gltf(&sample_model(n, sh_degree), &dst).expect("test: export should succeed");

        let text = read_text(&dst);
        // Once on the node extension, once on the document extension.
        assert_eq!(text.matches(&format!(r#""gaussianCount": {n}"#)).count(), 2);
        assert_eq!(
            text.matches(&format!(r#""shDegree": {sh_degree}"#)).count(),
            2
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn sh_coefficients_are_padded_to_the_declared_stride() {
        // A model whose `sh_coeffs` is short for its `sh_degree` must still
        // produce an accessor count that matches the buffer, or the document
        // describes bytes that are not there.
        let dir = temp_subdir("sh_padding");
        let dst = dir.join("scene.gltf");
        let mut model = sample_model(4, 2);
        model.sh_coeffs.truncate(3);

        write_gltf(&model, &dst).expect("test: export should succeed");

        let expected_sh_len = 4 * sh_coeffs_per_gaussian(2);
        let on_disk = std::fs::metadata(dir.join("scene.bin"))
            .expect("test: bin file should exist")
            .len() as usize;
        assert_eq!(
            on_disk,
            4 * 44 + expected_sh_len * 4,
            "the SH block must be padded to the declared stride"
        );
        assert!(
            read_text(&dst).contains(&format!(r#""count": {expected_sh_len}"#)),
            "the SH accessor count must match the padded block"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn block_offsets_follow_the_documented_layout() {
        let dir = temp_subdir("offsets");
        let dst = dir.join("scene.gltf");
        let n = 15usize;
        write_gltf(&sample_model(n, 0), &dst).expect("test: export should succeed");

        let text = read_text(&dst);
        for (label, offset) in [
            ("positions", 0),
            ("rotations", n * 12),
            ("scales", n * 28),
            ("opacities", n * 40),
            ("sh", n * 44),
        ] {
            assert!(
                text.contains(&format!(r#""byteOffset": {offset},"#)),
                "the {label} buffer view must start at byte {offset}:\n{text}"
            );
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn buffer_uri_is_json_escaped() {
        // A file name containing a quote or backslash must not break out of
        // the JSON string literal. Such names are legal on Unix.
        let dir = temp_subdir("escaped_uri");
        let dst = dir.join(r#"od"d.gltf"#);
        write_gltf(&sample_model(2, 0), &dst).expect("test: export should succeed");

        let text = read_text(&dst);
        assert!(
            text.contains(r#""uri": "od\"d.bin""#),
            "the buffer URI must be escaped:\n{text}"
        );
        assert_eq!(
            text.matches('{').count(),
            text.matches('}').count(),
            "escaping must keep the document balanced:\n{text}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}
