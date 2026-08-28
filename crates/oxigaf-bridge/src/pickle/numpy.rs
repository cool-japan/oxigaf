//! Interpretation of pickled NumPy arrays (and the SciPy/chumpy wrappers a
//! FLAME `.pkl` puts around them).
//!
//! A pickled `ndarray` arrives as an inert record of
//! `numpy.core.multiarray._reconstruct(ndarray, (0,), b'b')` with a `BUILD`
//! state of `(version, shape, dtype, is_fortran, data)`. This module reads
//! that state; it never calls anything.

use super::error::{PickleError, Result};
use super::value::Value;

/// The element types this reader materializes.
///
/// Deliberately narrow: a FLAME model's arrays are all float or integer, and
/// silently reinterpreting an unexpected dtype would be worse than a clear
/// error naming it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumpyDtype {
    /// IEEE-754 binary32.
    F32,
    /// IEEE-754 binary64.
    F64,
    /// Signed 32-bit.
    I32,
    /// Signed 64-bit.
    I64,
    /// Unsigned 32-bit.
    U32,
    /// Unsigned 64-bit.
    U64,
    /// Unsigned 8-bit.
    U8,
    /// Signed 8-bit.
    I8,
    /// Boolean, one byte per element.
    Bool,
}

impl NumpyDtype {
    /// Bytes per element.
    pub fn size(self) -> usize {
        match self {
            Self::F64 | Self::I64 | Self::U64 => 8,
            Self::F32 | Self::I32 | Self::U32 => 4,
            Self::U8 | Self::I8 | Self::Bool => 1,
        }
    }

    /// Parses a NumPy dtype *character code* (`'f4'`, `'<i8'`, `'|b1'`, …).
    ///
    /// The optional leading byte-order character is honoured: big-endian
    /// (`>`) data would need byte-swapping this reader does not do, so it is
    /// rejected rather than decoded wrongly. Every FLAME model in
    /// circulation, and every x86/ARM-written pickle, is little-endian.
    fn parse(code: &str) -> Result<(Self, bool)> {
        let (order, rest) = match code.as_bytes().first() {
            Some(b'<') | Some(b'|') | Some(b'=') => (b'<', &code[1..]),
            Some(b'>') => (b'>', &code[1..]),
            _ => (b'<', code),
        };
        let dtype = match rest {
            "f4" => Self::F32,
            "f8" => Self::F64,
            "i4" => Self::I32,
            "i8" => Self::I64,
            "u4" => Self::U32,
            "u8" => Self::U64,
            "u1" => Self::U8,
            "i1" => Self::I8,
            "b1" => Self::Bool,
            other => {
                return Err(PickleError::UnsupportedDtype(format!(
                    "numpy dtype '{other}'"
                )))
            }
        };
        // Multi-byte big-endian data is the only case needing a swap; a
        // one-byte dtype is order-independent.
        let big_endian = order == b'>' && dtype.size() > 1;
        Ok((dtype, big_endian))
    }
}

/// A decoded NumPy array.
#[derive(Debug, Clone, PartialEq)]
pub struct NumpyArray {
    /// Dimensions, outermost first.
    pub shape: Vec<usize>,
    /// Element type.
    pub dtype: NumpyDtype,
    /// Little-endian element bytes in row-major (C) order.
    pub data: Vec<u8>,
}

impl NumpyArray {
    /// Total number of elements.
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    /// Whether the array has no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The elements as `f32`, converting from whatever the source dtype was.
    ///
    /// # Errors
    ///
    /// Returns an error if the byte length does not match the declared shape
    /// and dtype.
    pub fn to_f32(&self) -> Result<Vec<f32>> {
        self.decode(|dtype, chunk| match dtype {
            NumpyDtype::F32 => f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]),
            NumpyDtype::F64 => f64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ]) as f32,
            NumpyDtype::I32 => i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as f32,
            NumpyDtype::I64 => i64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ]) as f32,
            NumpyDtype::U32 => u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as f32,
            NumpyDtype::U64 => u64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ]) as f32,
            NumpyDtype::U8 | NumpyDtype::Bool => f32::from(chunk[0]),
            NumpyDtype::I8 => f32::from(chunk[0] as i8),
        })
    }

    /// The elements as `i32`, converting from whatever the source dtype was.
    ///
    /// # Errors
    ///
    /// As [`NumpyArray::to_f32`].
    pub fn to_i32(&self) -> Result<Vec<i32>> {
        self.decode(|dtype, chunk| match dtype {
            NumpyDtype::F32 => f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as i32,
            NumpyDtype::F64 => f64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ]) as i32,
            NumpyDtype::I32 => i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]),
            NumpyDtype::I64 => i64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ]) as i32,
            NumpyDtype::U32 => u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as i32,
            NumpyDtype::U64 => u64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ]) as i32,
            NumpyDtype::U8 | NumpyDtype::Bool => i32::from(chunk[0]),
            NumpyDtype::I8 => i32::from(chunk[0] as i8),
        })
    }

    fn decode<T>(&self, convert: impl Fn(NumpyDtype, &[u8]) -> T) -> Result<Vec<T>> {
        let size = self.dtype.size();
        let expected = self
            .len()
            .checked_mul(size)
            .ok_or_else(|| PickleError::Structure("array size overflows".to_string()))?;
        if self.data.len() != expected {
            return Err(PickleError::Structure(format!(
                "array data is {} bytes but shape {:?} of {:?} needs {}",
                self.data.len(),
                self.shape,
                self.dtype,
                expected
            )));
        }
        Ok(self
            .data
            .chunks_exact(size)
            .map(|chunk| convert(self.dtype, chunk))
            .collect())
    }
}

/// Extracts an array's raw element bytes, seeing through the
/// `_codecs.encode(text, 'latin1')` indirection Python 3 uses to pickle a
/// `bytes` object under protocols 0-2.
///
/// Under those protocols there is no `BINBYTES` opcode, so `bytes` is
/// written as a `str` plus a recorded call to `_codecs.encode(..., 'latin1')`
/// that would decode it back. Latin-1 maps each of the 256 byte values to
/// the code point of the same number, so undoing it is exactly
/// "take each character's code point as one byte" -- and it must be undone,
/// or a NumPy 2-written FLAME model (which is every one produced today)
/// looks like it has no array data at all.
///
/// Returns `Cow` so the overwhelmingly common `BINBYTES` path stays a
/// borrow rather than copying multi-megabyte tensors.
fn raw_bytes(value: &Value) -> Option<std::borrow::Cow<'_, [u8]>> {
    if let Some(bytes) = value.as_bytes() {
        return Some(std::borrow::Cow::Borrowed(bytes));
    }

    let (module, name) = value.class_path()?;
    if module != "_codecs" || name != "encode" {
        return None;
    }
    let args = value.ctor_args()?;
    // The codec argument, when present, must be latin1: any other codec
    // would need a real decoder, and guessing would corrupt the data.
    if let Some(codec) = args.get(1).and_then(Value::as_text) {
        if !matches!(codec.as_str(), "latin1" | "latin-1" | "iso-8859-1") {
            return None;
        }
    }
    match args.first()? {
        Value::Str(text) => Some(std::borrow::Cow::Owned(
            text.chars().map(|c| c as u32 as u8).collect(),
        )),
        Value::Bytes(bytes) => Some(std::borrow::Cow::Borrowed(bytes)),
        _ => None,
    }
}

/// Whether `module` is NumPy's multiarray module under either of its names.
///
/// NumPy 2.0 renamed the private `numpy.core` package to `numpy._core`, so a
/// pickle written by NumPy >= 2 says `numpy._core.multiarray._reconstruct`
/// where an older one says `numpy.core.multiarray._reconstruct`. Both name
/// the same constructor, and a FLAME `.pkl` may have been written by either,
/// so both must be recognized -- matching only one would make the reader
/// silently fail on half the files in circulation.
fn is_numpy_multiarray(module: &str) -> bool {
    matches!(
        module,
        "numpy.core.multiarray" | "numpy._core.multiarray" | "numpy.core._multiarray_umath"
    )
}

/// Extracts a NumPy array from a decoded pickle value, seeing through the
/// wrappers a FLAME `.pkl` uses.
///
/// Handles, in order:
/// 1. a plain `numpy.ndarray` (`_reconstruct` + `BUILD` state),
/// 2. a `numpy` scalar (`numpy.core.multiarray.scalar`),
/// 3. a `chumpy` `Ch` object, whose value lives in its `x` attribute -- the
///    FLAME models are authored in chumpy, so their arrays arrive wrapped,
/// 4. anything with an `.x` or `.r` attribute holding one of the above,
///    which covers the remaining chumpy node types.
///
/// # Errors
///
/// Returns an error if the value is not (and does not wrap) an array this
/// reader understands, naming what was found instead.
pub fn as_array(value: &Value) -> Result<NumpyArray> {
    // Depth is bounded: each step strips one wrapper, and the wrappers are
    // never mutually recursive in practice, but bound it explicitly so a
    // crafted self-referential-looking graph cannot spin.
    let mut current = value;
    for _ in 0..8 {
        if let Some((module, name)) = current.class_path() {
            if is_numpy_multiarray(module) || module == "numpy" {
                match name {
                    "_reconstruct" | "ndarray" => return from_reconstruct(current),
                    "scalar" => return from_scalar(current),
                    _ => {}
                }
            }
            if is_numpy_numeric(module) && name == "_frombuffer" {
                return from_frombuffer(current);
            }
        }
        // chumpy and friends: unwrap the attribute holding the real array.
        let next = current
            .get("x")
            .or_else(|| current.get("r"))
            .or_else(|| current.get("_data"));
        match next {
            Some(inner) => current = inner,
            None => break,
        }
    }
    Err(PickleError::Structure(format!(
        "expected a numpy array, found {value}"
    )))
}

/// Decodes `numpy.core.multiarray._reconstruct` + its `BUILD` state
/// `(version, shape, dtype, is_fortran, data)`.
fn from_reconstruct(value: &Value) -> Result<NumpyArray> {
    let state = value
        .state()
        .ok_or_else(|| PickleError::Structure("ndarray has no __setstate__ payload".to_string()))?;
    let parts = state.as_seq().ok_or_else(|| {
        PickleError::Structure(format!(
            "ndarray state is {}, not a tuple",
            state.type_name()
        ))
    })?;
    if parts.len() < 5 {
        return Err(PickleError::Structure(format!(
            "ndarray state has {} entries, expected 5",
            parts.len()
        )));
    }

    let shape: Vec<usize> = parts[1]
        .as_seq()
        .ok_or_else(|| PickleError::Structure("ndarray state has no shape tuple".to_string()))?
        .iter()
        .map(Value::as_usize)
        .collect::<Option<_>>()
        .ok_or_else(|| PickleError::Structure("ndarray shape has a negative axis".to_string()))?;

    let (dtype, big_endian) = NumpyDtype::parse(&dtype_code(&parts[2])?)?;
    if big_endian {
        return Err(PickleError::UnsupportedDtype(
            "big-endian numpy array (byte-swapped data is not supported)".to_string(),
        ));
    }

    let fortran = parts[3].as_i64().unwrap_or(0) != 0;
    let raw = raw_bytes(&parts[4]).ok_or_else(|| {
        PickleError::Structure(format!(
            "ndarray data is {}, not raw bytes (object-dtype arrays are not supported)",
            parts[4].type_name()
        ))
    })?;
    let raw = raw.as_ref();

    let count: usize = shape.iter().product();
    let expected = count
        .checked_mul(dtype.size())
        .ok_or_else(|| PickleError::Structure("array size overflows".to_string()))?;
    if raw.len() != expected {
        return Err(PickleError::Structure(format!(
            "ndarray data is {} bytes but shape {:?} of {:?} needs {}",
            raw.len(),
            shape,
            dtype,
            expected
        )));
    }

    // Fortran (column-major) order must be transposed into row-major, or
    // every consumer would silently read the array transposed.
    let data = if fortran && shape.len() > 1 {
        fortran_to_c(raw, &shape, dtype.size())
    } else {
        raw.to_vec()
    };

    Ok(NumpyArray { shape, dtype, data })
}

/// Whether `module` is NumPy's `numeric` module under either of its names.
fn is_numpy_numeric(module: &str) -> bool {
    matches!(module, "numpy.core.numeric" | "numpy._core.numeric")
}

/// Decodes `numpy._core.numeric._frombuffer(buffer, dtype, shape, order)`.
///
/// This is the form NumPy emits under pickle protocol 5 (and protocols 3-4),
/// where a `BYTEARRAY8`/`BINBYTES` opcode can carry the raw buffer directly
/// and no `_reconstruct` + `BUILD` dance is needed. A protocol-2 file uses
/// [`from_reconstruct`] instead; a FLAME `.pkl` may be either, so both paths
/// must exist.
fn from_frombuffer(value: &Value) -> Result<NumpyArray> {
    let args = value
        .ctor_args()
        .ok_or_else(|| PickleError::Structure("_frombuffer has no argument tuple".to_string()))?;
    if args.len() < 3 {
        return Err(PickleError::Structure(format!(
            "_frombuffer has {} arguments, expected at least 3",
            args.len()
        )));
    }

    let raw = raw_bytes(&args[0]).ok_or_else(|| {
        PickleError::Structure(format!(
            "_frombuffer's buffer is {}, not raw bytes",
            args[0].type_name()
        ))
    })?;
    let (dtype, big_endian) = NumpyDtype::parse(&dtype_code(&args[1])?)?;
    if big_endian {
        return Err(PickleError::UnsupportedDtype(
            "big-endian numpy array (byte-swapped data is not supported)".to_string(),
        ));
    }

    let shape: Vec<usize> = args[2]
        .as_seq()
        .ok_or_else(|| PickleError::Structure("_frombuffer has no shape tuple".to_string()))?
        .iter()
        .map(Value::as_usize)
        .collect::<Option<_>>()
        .ok_or_else(|| {
            PickleError::Structure("_frombuffer shape has a negative axis".to_string())
        })?;

    let count: usize = shape.iter().product();
    let expected = count
        .checked_mul(dtype.size())
        .ok_or_else(|| PickleError::Structure("array size overflows".to_string()))?;
    if raw.len() != expected {
        return Err(PickleError::Structure(format!(
            "_frombuffer buffer is {} bytes but shape {:?} of {:?} needs {}",
            raw.len(),
            shape,
            dtype,
            expected
        )));
    }

    // The fourth argument is the memory order; 'F' means column-major and
    // must be transposed into row-major exactly as in `from_reconstruct`.
    let fortran = args
        .get(3)
        .and_then(Value::as_text)
        .is_some_and(|order| order.eq_ignore_ascii_case("f"));
    let data = if fortran && shape.len() > 1 {
        fortran_to_c(&raw, &shape, dtype.size())
    } else {
        raw.into_owned()
    };

    Ok(NumpyArray { shape, dtype, data })
}

/// Decodes `numpy.core.multiarray.scalar(dtype, bytes)` into a
/// zero-dimensional array.
fn from_scalar(value: &Value) -> Result<NumpyArray> {
    let args = value
        .ctor_args()
        .ok_or_else(|| PickleError::Structure("numpy scalar has no arguments".to_string()))?;
    let (dtype, big_endian) =
        NumpyDtype::parse(&dtype_code(args.first().ok_or_else(|| {
            PickleError::Structure("numpy scalar has no dtype".to_string())
        })?)?)?;
    if big_endian {
        return Err(PickleError::UnsupportedDtype(
            "big-endian numpy scalar".to_string(),
        ));
    }
    let data = args
        .get(1)
        .and_then(raw_bytes)
        .ok_or_else(|| PickleError::Structure("numpy scalar has no data".to_string()))?
        .into_owned();
    Ok(NumpyArray {
        shape: Vec::new(),
        dtype,
        data,
    })
}

/// Extracts the character code from a pickled `numpy.dtype`, which arrives
/// as `numpy.dtype('f4', False, True)` with a `BUILD` state whose third
/// entry is the byte order.
fn dtype_code(value: &Value) -> Result<String> {
    // A bare string dtype (some picklers emit one directly).
    if let Some(code) = value.as_text() {
        return Ok(code);
    }

    let args = value
        .ctor_args()
        .ok_or_else(|| PickleError::Structure(format!("expected a numpy dtype, found {value}")))?;
    let base = args
        .first()
        .and_then(Value::as_text)
        .ok_or_else(|| PickleError::Structure("numpy dtype has no type code".to_string()))?;

    // The BUILD state is (version, byteorder, subdescr, names, fields,
    // itemsize, alignment, flags); its byte-order character overrides the
    // constructor's, so honour it rather than reading big-endian data as
    // little-endian.
    if let Some(order) = value
        .state()
        .and_then(Value::as_seq)
        .and_then(|parts| parts.get(1))
        .and_then(Value::as_text)
    {
        if matches!(order.as_str(), "<" | ">" | "|" | "=") {
            return Ok(format!("{order}{base}"));
        }
    }
    Ok(base)
}

/// Re-orders column-major bytes into row-major.
fn fortran_to_c(raw: &[u8], shape: &[usize], element_size: usize) -> Vec<u8> {
    let count: usize = shape.iter().product();
    let mut out = Vec::with_capacity(raw.len());

    // Fortran strides: first axis fastest.
    let mut f_stride = vec![1usize; shape.len()];
    for axis in 1..shape.len() {
        f_stride[axis] = f_stride[axis - 1] * shape[axis - 1];
    }

    let mut index = vec![0usize; shape.len()];
    for _ in 0..count {
        let source: usize = index
            .iter()
            .zip(f_stride.iter())
            .map(|(&i, &s)| i * s)
            .sum();
        let start = source * element_size;
        // `count` and `raw.len()` were reconciled by the caller, so this
        // slice is always in bounds; `get` keeps it panic-free regardless.
        if let Some(chunk) = raw.get(start..start + element_size) {
            out.extend_from_slice(chunk);
        }
        for axis in (0..shape.len()).rev() {
            index[axis] += 1;
            if index[axis] < shape[axis] {
                break;
            }
            index[axis] = 0;
        }
    }
    out
}

/// Materializes a SciPy sparse matrix (`csc_matrix` / `csr_matrix` /
/// `coo_matrix`) as a dense row-major `f32` array.
///
/// FLAME's `J_regressor` is stored this way, which is why
/// `scripts/convert_flame.py` needed `scipy` at all.
///
/// # Errors
///
/// Returns an error if the value is not a sparse matrix this reader
/// understands, or if its index arrays are inconsistent with its declared
/// shape.
pub fn as_dense_from_sparse(value: &Value) -> Result<NumpyArray> {
    let Some((module, class)) = value.class_path() else {
        return Err(PickleError::Structure(format!(
            "expected a scipy sparse matrix, found {value}"
        )));
    };
    if !module.starts_with("scipy.sparse") {
        return Err(PickleError::Structure(format!(
            "expected a scipy sparse matrix, found {module}.{class}"
        )));
    }

    let shape: Vec<usize> = value
        .get("_shape")
        .or_else(|| value.get("shape"))
        .and_then(Value::as_seq)
        .ok_or_else(|| PickleError::Structure(format!("{class} has no _shape")))?
        .iter()
        .map(Value::as_usize)
        .collect::<Option<_>>()
        .ok_or_else(|| PickleError::Structure(format!("{class} has a negative axis")))?;
    let [rows, cols] = shape[..] else {
        return Err(PickleError::Structure(format!(
            "{class} has {} dimensions, expected 2",
            shape.len()
        )));
    };

    let values = value
        .get("data")
        .ok_or_else(|| PickleError::Structure(format!("{class} has no data array")))
        .and_then(as_array)?
        .to_f32()?;

    let mut dense = vec![
        0f32;
        rows.checked_mul(cols).ok_or_else(|| {
            PickleError::Structure(format!("{class} shape {rows}x{cols} overflows"))
        })?
    ];

    let set = |dense: &mut Vec<f32>, row: usize, col: usize, v: f32| -> Result<()> {
        if row >= rows || col >= cols {
            return Err(PickleError::Structure(format!(
                "{class} index ({row}, {col}) is outside its declared {rows}x{cols} shape"
            )));
        }
        dense[row * cols + col] = v;
        Ok(())
    };

    match class {
        "coo_matrix" | "coo_array" => {
            let row_idx = as_array(
                value
                    .get("row")
                    .ok_or_else(|| PickleError::Structure("coo_matrix has no row".to_string()))?,
            )?
            .to_i32()?;
            let col_idx = as_array(
                value
                    .get("col")
                    .ok_or_else(|| PickleError::Structure("coo_matrix has no col".to_string()))?,
            )?
            .to_i32()?;
            for ((&r, &c), &v) in row_idx.iter().zip(col_idx.iter()).zip(values.iter()) {
                set(&mut dense, r.max(0) as usize, c.max(0) as usize, v)?;
            }
        }
        "csr_matrix" | "csr_array" | "csc_matrix" | "csc_array" => {
            let indices = as_array(
                value
                    .get("indices")
                    .ok_or_else(|| PickleError::Structure(format!("{class} has no indices")))?,
            )?
            .to_i32()?;
            let indptr = as_array(
                value
                    .get("indptr")
                    .ok_or_else(|| PickleError::Structure(format!("{class} has no indptr")))?,
            )?
            .to_i32()?;

            // CSR indexes rows by indptr; CSC indexes columns. Everything
            // else about the traversal is identical.
            let by_row = class.starts_with("csr");
            let major_count = if by_row { rows } else { cols };
            if indptr.len() != major_count + 1 {
                return Err(PickleError::Structure(format!(
                    "{class} indptr has {} entries, expected {}",
                    indptr.len(),
                    major_count + 1
                )));
            }
            for major in 0..major_count {
                let start = indptr[major].max(0) as usize;
                let end = indptr[major + 1].max(0) as usize;
                if end > indices.len() || end > values.len() || start > end {
                    return Err(PickleError::Structure(format!(
                        "{class} indptr range {start}..{end} is outside its data arrays"
                    )));
                }
                for k in start..end {
                    let minor = indices[k].max(0) as usize;
                    let (row, col) = if by_row {
                        (major, minor)
                    } else {
                        (minor, major)
                    };
                    set(&mut dense, row, col, values[k])?;
                }
            }
        }
        other => {
            return Err(PickleError::Structure(format!(
                "unsupported scipy sparse format '{other}'; \
                 csr, csc and coo are supported"
            )))
        }
    }

    Ok(NumpyArray {
        shape: vec![rows, cols],
        dtype: NumpyDtype::F32,
        data: dense.iter().flat_map(|v| v.to_le_bytes()).collect(),
    })
}

/// Extracts an array, accepting either a dense NumPy array or a SciPy
/// sparse matrix.
///
/// # Errors
///
/// Returns an error if the value is neither.
pub fn as_array_or_sparse(value: &Value) -> Result<NumpyArray> {
    match as_array(value) {
        Ok(array) => Ok(array),
        Err(dense_error) => as_dense_from_sparse(value).map_err(|_| dense_error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pickle::test_support::{pickle, PickleBuilder};
    use crate::pickle::vm;

    /// Emits the opcodes for a pickled `numpy.ndarray`, exactly as CPython's
    /// `numpy` does: `_reconstruct(ndarray, (0,), b'b')` followed by a
    /// `BUILD` of `(1, shape, dtype, fortran, raw_bytes)`.
    fn push_ndarray(
        p: &mut PickleBuilder,
        shape: &[usize],
        dtype_code: &str,
        fortran: bool,
        raw: &[u8],
    ) {
        p.global("numpy.core.multiarray", "_reconstruct");
        p.mark();
        p.global("numpy", "ndarray");
        p.int_tuple(&[0]);
        p.py2_str(b"b");
        p.tuple();
        p.reduce();

        // BUILD state tuple
        p.mark();
        p.int(1);
        p.int_tuple(shape);
        push_dtype(p, dtype_code);
        p.bool(fortran);
        p.py2_str(raw);
        p.tuple();
        p.build_state();
    }

    /// `numpy.dtype('f4', False, True)` + `BUILD (3, '<', None, None, None, -1, -1, 0)`.
    fn push_dtype(p: &mut PickleBuilder, code: &str) {
        p.global("numpy", "dtype");
        p.mark();
        p.unicode(code);
        p.bool(false);
        p.bool(true);
        p.tuple();
        p.reduce();
        p.mark();
        p.int(3);
        p.unicode("<");
        p.none();
        p.none();
        p.none();
        p.int(0);
        p.int(0);
        p.int(0);
        p.tuple();
        p.build_state();
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    #[test]
    fn test_reads_a_c_order_f32_array() {
        let bytes = pickle(|p| {
            push_ndarray(
                p,
                &[2, 3],
                "f4",
                false,
                &f32_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            );
        });
        let value = vm::load(&bytes).expect("test: unpickle should succeed");
        let array = as_array(&value).expect("test: should decode as an ndarray");
        assert_eq!(array.shape, vec![2, 3]);
        assert_eq!(array.dtype, NumpyDtype::F32);
        assert_eq!(
            array.to_f32().expect("test: decode"),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
    }

    #[test]
    fn test_fortran_order_is_transposed_into_c_order() {
        // Regression guard: reading Fortran-ordered bytes verbatim would
        // silently transpose the array. Column-major [1,4,2,5,3,6] is the
        // same 2x3 matrix as row-major [1,2,3,4,5,6].
        let bytes = pickle(|p| {
            push_ndarray(
                p,
                &[2, 3],
                "f4",
                true,
                &f32_bytes(&[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]),
            );
        });
        let value = vm::load(&bytes).expect("test: unpickle should succeed");
        let array = as_array(&value).expect("test: should decode as an ndarray");
        assert_eq!(
            array.to_f32().expect("test: decode"),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
    }

    #[test]
    fn test_int_and_double_dtypes_convert() {
        let bytes = pickle(|p| {
            push_ndarray(
                p,
                &[3],
                "i4",
                false,
                &[1i32, -2, 3]
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            );
        });
        let value = vm::load(&bytes).expect("test: unpickle should succeed");
        let array = as_array(&value).expect("test: should decode");
        assert_eq!(array.to_i32().expect("test: decode"), vec![1, -2, 3]);
        assert_eq!(array.to_f32().expect("test: decode"), vec![1.0, -2.0, 3.0]);

        let bytes = pickle(|p| {
            push_ndarray(
                p,
                &[2],
                "f8",
                false,
                &[0.5f64, 1.5]
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            );
        });
        let value = vm::load(&bytes).expect("test: unpickle should succeed");
        let array = as_array(&value).expect("test: should decode");
        assert_eq!(array.to_f32().expect("test: decode"), vec![0.5, 1.5]);
    }

    #[test]
    fn test_big_endian_is_rejected_rather_than_misread() {
        // Reading '>f4' data as little-endian would produce plausible-looking
        // garbage, which is far worse than a clear error.
        let mut array = NumpyDtype::parse(">f4").expect("test: parse should succeed");
        assert!(array.1, "big-endian flag should be set");
        array = NumpyDtype::parse("<f4").expect("test: parse should succeed");
        assert!(!array.1);
        // A one-byte dtype has no byte order to get wrong.
        assert!(!NumpyDtype::parse(">u1").expect("test: parse").1);
    }

    #[test]
    fn test_unsupported_dtype_names_itself() {
        let err = NumpyDtype::parse("c16").expect_err("complex must be rejected");
        assert!(err.to_string().contains("c16"), "got: {err}");
    }

    #[test]
    fn test_length_mismatch_is_reported_not_panicked_on() {
        // The shape says six elements; the payload holds two. A slice-based
        // decoder would panic on exactly this kind of corrupt input.
        let bytes = pickle(|p| {
            push_ndarray(p, &[2, 3], "f4", false, &f32_bytes(&[1.0, 2.0]));
        });
        let value = vm::load(&bytes).expect("test: unpickle should succeed");
        let err = as_array(&value).expect_err("length mismatch must error");
        assert!(matches!(err, PickleError::Structure(_)), "got: {err}");
    }

    #[test]
    fn test_unwraps_a_chumpy_array() {
        // FLAME models are authored in chumpy, so their arrays arrive as
        // `Ch` objects whose `x` attribute holds the real ndarray.
        let bytes = pickle(|p| {
            p.global("chumpy.ch", "Ch");
            p.mark();
            p.tuple();
            p.reduce();
            p.empty_dict();
            p.unicode("x");
            push_ndarray(p, &[2], "f4", false, &f32_bytes(&[7.0, 8.0]));
            p.setitem();
            p.build_state();
        });
        let value = vm::load(&bytes).expect("test: unpickle should succeed");
        let array = as_array(&value).expect("test: chumpy wrapper should be unwrapped");
        assert_eq!(array.to_f32().expect("test: decode"), vec![7.0, 8.0]);
    }

    #[test]
    fn test_csc_sparse_matrix_densifies() {
        // The 3x2 matrix [[0, 5], [1, 0], [0, 0]] in CSC form:
        //   data = [1, 5], indices = [1, 0], indptr = [0, 1, 2]
        let bytes = pickle(|p| {
            p.global("scipy.sparse.csc", "csc_matrix");
            p.mark();
            p.tuple();
            p.reduce();
            p.empty_dict();

            p.unicode("_shape");
            p.int_tuple(&[3, 2]);
            p.setitem();

            p.unicode("data");
            push_ndarray(p, &[2], "f4", false, &f32_bytes(&[1.0, 5.0]));
            p.setitem();

            p.unicode("indices");
            push_ndarray(
                p,
                &[2],
                "i4",
                false,
                &[1i32, 0]
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            );
            p.setitem();

            p.unicode("indptr");
            push_ndarray(
                p,
                &[3],
                "i4",
                false,
                &[0i32, 1, 2]
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            );
            p.setitem();

            p.build_state();
        });
        let value = vm::load(&bytes).expect("test: unpickle should succeed");
        let dense = as_dense_from_sparse(&value).expect("test: should densify");
        assert_eq!(dense.shape, vec![3, 2]);
        assert_eq!(
            dense.to_f32().expect("test: decode"),
            vec![0.0, 5.0, 1.0, 0.0, 0.0, 0.0]
        );
    }

    #[test]
    fn test_sparse_index_outside_shape_is_rejected() {
        // A corrupt `indices` entry pointing past the declared shape must
        // error rather than write out of bounds (or silently wrap).
        let bytes = pickle(|p| {
            p.global("scipy.sparse.csc", "csc_matrix");
            p.mark();
            p.tuple();
            p.reduce();
            p.empty_dict();
            p.unicode("_shape");
            p.int_tuple(&[2, 1]);
            p.setitem();
            p.unicode("data");
            push_ndarray(p, &[1], "f4", false, &f32_bytes(&[1.0]));
            p.setitem();
            p.unicode("indices");
            push_ndarray(p, &[1], "i4", false, &99i32.to_le_bytes());
            p.setitem();
            p.unicode("indptr");
            push_ndarray(
                p,
                &[2],
                "i4",
                false,
                &[0i32, 1]
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            );
            p.setitem();
            p.build_state();
        });
        let value = vm::load(&bytes).expect("test: unpickle should succeed");
        let err = as_dense_from_sparse(&value).expect_err("out-of-range index must error");
        assert!(matches!(err, PickleError::Structure(_)), "got: {err}");
    }

    #[test]
    fn test_reads_the_numpy_2_module_path() {
        // Regression test, found against real NumPy 2.4 output: NumPy 2.0
        // renamed `numpy.core` to `numpy._core`, so a model pickled today
        // says `numpy._core.multiarray._reconstruct`. Matching only the old
        // name made every modern FLAME `.pkl` fail with "expected a numpy
        // array".
        let bytes = pickle(|p| {
            p.global("numpy._core.multiarray", "_reconstruct");
            p.mark();
            p.global("numpy", "ndarray");
            p.int_tuple(&[0]);
            p.py2_str(b"b");
            p.tuple();
            p.reduce();
            p.mark();
            p.int(1);
            p.int_tuple(&[2]);
            push_dtype(p, "f4");
            p.bool(false);
            p.py2_str(&f32_bytes(&[1.0, 2.0]));
            p.tuple();
            p.build_state();
        });
        let value = vm::load(&bytes).expect("test: unpickle should succeed");
        let array = as_array(&value).expect("test: numpy._core path should be recognized");
        assert_eq!(array.to_f32().expect("test: decode"), vec![1.0, 2.0]);
    }

    #[test]
    fn test_unwraps_codecs_latin1_encoded_array_data() {
        // Regression test, found against real NumPy output: under pickle
        // protocols 0-2 there is no `bytes` opcode, so Python 3 writes an
        // array's raw data as `_codecs.encode(<str>, 'latin1')`. Without
        // undoing that, every protocol-2 array looks like it has no data.
        let raw = f32_bytes(&[1.0, 2.0]);
        let latin1: String = raw.iter().map(|&b| b as char).collect();

        let bytes = pickle(|p| {
            p.global("numpy._core.multiarray", "_reconstruct");
            p.mark();
            p.global("numpy", "ndarray");
            p.int_tuple(&[0]);
            p.py2_str(b"b");
            p.tuple();
            p.reduce();
            p.mark();
            p.int(1);
            p.int_tuple(&[2]);
            push_dtype(p, "f4");
            p.bool(false);
            // _codecs.encode(latin1_text, 'latin1')
            p.global("_codecs", "encode");
            p.mark();
            p.unicode(&latin1);
            p.unicode("latin1");
            p.tuple();
            p.reduce();
            p.tuple();
            p.build_state();
        });
        let value = vm::load(&bytes).expect("test: unpickle should succeed");
        let array = as_array(&value).expect("test: latin1-encoded data should be decoded");
        assert_eq!(array.to_f32().expect("test: decode"), vec![1.0, 2.0]);
    }

    #[test]
    fn test_rejects_a_codec_other_than_latin1() {
        // Only latin1 is a byte-for-byte identity; guessing at another
        // codec would silently corrupt the array.
        let encoded = Value::Object {
            class: Box::new(Value::Global {
                module: "_codecs".into(),
                name: "encode".into(),
            }),
            args: Box::new(Value::Tuple(vec![
                Value::Str("abc".into()),
                Value::Str("utf-8".into()),
            ])),
            state: None,
            list_items: Vec::new(),
            dict_items: Vec::new(),
        };
        assert!(raw_bytes(&encoded).is_none());
    }

    #[test]
    fn test_reads_protocol_5_frombuffer_arrays() {
        // Regression test, found against real NumPy protocol-5 output:
        // protocols 3+ pickle an array as
        // `numpy._core.numeric._frombuffer(buf, dtype, shape, order)`
        // rather than `_reconstruct` + BUILD, so a protocol-5 FLAME model
        // needs this second path.
        let bytes = pickle(|p| {
            p.global("numpy._core.numeric", "_frombuffer");
            p.mark();
            p.py2_str(&f32_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
            push_dtype(p, "f4");
            p.int_tuple(&[2, 3]);
            p.unicode("C");
            p.tuple();
            p.reduce();
        });
        let value = vm::load(&bytes).expect("test: unpickle should succeed");
        let array = as_array(&value).expect("test: _frombuffer should be recognized");
        assert_eq!(array.shape, vec![2, 3]);
        assert_eq!(
            array.to_f32().expect("test: decode"),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
    }

    #[test]
    fn test_frombuffer_honors_fortran_order() {
        let bytes = pickle(|p| {
            p.global("numpy._core.numeric", "_frombuffer");
            p.mark();
            p.py2_str(&f32_bytes(&[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]));
            push_dtype(p, "f4");
            p.int_tuple(&[2, 3]);
            p.unicode("F");
            p.tuple();
            p.reduce();
        });
        let value = vm::load(&bytes).expect("test: unpickle should succeed");
        let array = as_array(&value).expect("test: _frombuffer should be recognized");
        assert_eq!(
            array.to_f32().expect("test: decode"),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
    }

    #[test]
    fn test_frombuffer_length_mismatch_is_reported() {
        let bytes = pickle(|p| {
            p.global("numpy._core.numeric", "_frombuffer");
            p.mark();
            p.py2_str(&f32_bytes(&[1.0]));
            push_dtype(p, "f4");
            p.int_tuple(&[2, 3]);
            p.unicode("C");
            p.tuple();
            p.reduce();
        });
        let value = vm::load(&bytes).expect("test: unpickle should succeed");
        let err = as_array(&value).expect_err("length mismatch must error");
        assert!(matches!(err, PickleError::Structure(_)), "got: {err}");
    }

    #[test]
    fn test_as_array_or_sparse_accepts_both() {
        let dense_bytes = pickle(|p| {
            push_ndarray(p, &[2], "f4", false, &f32_bytes(&[1.0, 2.0]));
        });
        let dense = vm::load(&dense_bytes).expect("test: unpickle should succeed");
        assert!(as_array_or_sparse(&dense).is_ok());

        let err = as_array_or_sparse(&Value::Int(3)).expect_err("an int is neither");
        assert!(err.to_string().contains("numpy array"), "got: {err}");
    }
}
