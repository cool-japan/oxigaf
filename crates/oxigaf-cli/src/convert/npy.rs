//! Minimal, dependency-free NPY reader/writer used by the FLAME conversion.
//!
//! Arrays are always normalised to C (row-major) order on read and written
//! back as NPY format 1.0 streams.

use anyhow::{bail, Context, Result};

/// NPY magic prefix.
pub const MAGIC: &[u8; 6] = b"\x93NUMPY";

/// A decoded N-dimensional array with C-order (row-major) element data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NpyArray {
    /// NumPy dtype descriptor, e.g. `"<f4"`.
    pub descr: String,
    /// Array shape.
    pub shape: Vec<usize>,
    /// Raw element bytes in C order.
    pub data: Vec<u8>,
}

impl NpyArray {
    /// Number of elements described by the shape.
    pub fn element_count(&self) -> usize {
        self.shape.iter().product()
    }
}

/// Parsed NPY header metadata.
pub struct Header {
    /// NumPy dtype descriptor.
    pub descr: String,
    /// Whether elements are stored in column-major order.
    pub fortran_order: bool,
    /// Array shape.
    pub shape: Vec<usize>,
    /// Offset of the first data byte.
    pub data_offset: usize,
}

/// Numeric value decoded from a single dtype element.
#[derive(Clone, Copy)]
enum Scalar {
    Float(f64),
    Signed(i64),
    Unsigned(u64),
}

impl Scalar {
    fn as_f64(self) -> f64 {
        match self {
            Scalar::Float(v) => v,
            Scalar::Signed(v) => v as f64,
            Scalar::Unsigned(v) => v as f64,
        }
    }

    fn as_i64(self) -> i64 {
        match self {
            Scalar::Float(v) => v as i64,
            Scalar::Signed(v) => v,
            Scalar::Unsigned(v) => v as i64,
        }
    }
}

/// Byte-order character of a dtype descriptor.
fn byte_order(descr: &str) -> char {
    // Not a plain `.next().unwrap_or('<')`: an unprefixed descriptor (e.g.
    // "f4") must also default to '<', not return its own first character.
    descr
        .chars()
        .next()
        .filter(|c| matches!(c, '<' | '>' | '|' | '='))
        .unwrap_or('<')
}

/// Kind character of a dtype descriptor (`f`, `i`, `u`, `b`, ...).
fn kind(descr: &str) -> char {
    let mut chars = descr.chars();
    match chars.next() {
        Some('<' | '>' | '|' | '=') => chars.next().unwrap_or('?'),
        Some(c) => c,
        None => '?',
    }
}

/// Byte width of a dtype descriptor such as `"<f4"`.
pub fn item_size(descr: &str) -> Result<usize> {
    let body = descr.trim_start_matches(['<', '>', '|', '=']);
    let mut chars = body.chars();
    let kind = chars
        .next()
        .with_context(|| format!("dtype descriptor '{descr}' has no type code"))?;
    let digits: String = chars.collect();
    if digits.is_empty() {
        bail!("dtype '{descr}' has no element size (object dtypes are unsupported)");
    }
    let size: usize = digits
        .parse()
        .with_context(|| format!("unsupported dtype descriptor '{descr}'"))?;
    if size == 0 {
        bail!("dtype '{descr}' has a zero element size");
    }
    if !matches!(kind, 'f' | 'i' | 'u' | 'b' | 'S' | 'V' | 'c') {
        bail!("unsupported dtype kind '{kind}' in '{descr}'");
    }
    Ok(size)
}

/// Extract a quoted field from an NPY header dict literal.
pub fn extract_header_field(header: &str, field: &str) -> Option<String> {
    let pattern = format!("'{}':", field);
    let pos = header.find(&pattern)?;
    let rest = &header[pos + pattern.len()..];
    let quote_start = rest.find('\'')?;
    let quote_end = rest[quote_start + 1..].find('\'')?;
    Some(rest[quote_start + 1..quote_start + 1 + quote_end].to_string())
}

/// Extract the shape tuple from an NPY header dict literal.
pub fn extract_shape_from_header(header: &str) -> Vec<usize> {
    let Some(pos) = header.find("'shape':") else {
        return Vec::new();
    };
    let rest = &header[pos + 8..];
    let (Some(open), Some(close)) = (rest.find('('), rest.find(')')) else {
        return Vec::new();
    };
    if close <= open {
        return Vec::new();
    }
    rest[open + 1..close]
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect()
}

/// Parse the header of an NPY byte stream.
///
/// Supports format version 1.0 (2-byte header length) as well as 2.0 and
/// 3.0, which numpy emits automatically once the header exceeds 64 KiB.
pub fn parse_header(data: &[u8]) -> Result<Header> {
    if data.len() < 10 || &data[0..6] != MAGIC {
        bail!("not an NPY stream (missing \\x93NUMPY magic)");
    }
    let major = data[6];
    let (header_len, header_start) = match major {
        1 => (usize::from(u16::from_le_bytes([data[8], data[9]])), 10usize),
        2 | 3 => {
            if data.len() < 12 {
                bail!("truncated NPY header (version {major}.x needs a 4-byte length)");
            }
            let len = u32::from_le_bytes([data[8], data[9], data[10], data[11]]);
            (
                usize::try_from(len).context("NPY header length does not fit in memory")?,
                12usize,
            )
        }
        other => bail!("unsupported NPY format version {other}.x"),
    };
    let header_end = header_start
        .checked_add(header_len)
        .context("NPY header length overflows")?;
    if data.len() < header_end {
        bail!(
            "truncated NPY header (needs {header_end} bytes, file has {})",
            data.len()
        );
    }
    let header = std::str::from_utf8(&data[header_start..header_end])
        .context("NPY header is not valid UTF-8")?;
    let descr = extract_header_field(header, "descr").context("NPY header has no 'descr' field")?;
    let fortran_order =
        header.contains("'fortran_order': True") || header.contains("'fortran_order':True");
    let shape = extract_shape_from_header(header);
    Ok(Header {
        descr,
        fortran_order,
        shape,
        data_offset: header_end,
    })
}

/// Parse a full NPY byte stream, normalising the data to C order.
pub fn parse(data: &[u8]) -> Result<NpyArray> {
    let header = parse_header(data)?;
    let item = item_size(&header.descr)?;
    let count: usize = header.shape.iter().product();
    let expected = count
        .checked_mul(item)
        .context("NPY array size overflows")?;
    let body = data
        .get(header.data_offset..)
        .context("NPY stream has no data section")?;
    if body.len() < expected {
        bail!(
            "truncated NPY data: expected {expected} bytes for shape {:?}, found {}",
            header.shape,
            body.len()
        );
    }
    let raw = body[..expected].to_vec();
    let data = if header.fortran_order {
        fortran_to_c(&raw, &header.shape, item)
    } else {
        raw
    };
    Ok(NpyArray {
        descr: header.descr,
        shape: header.shape,
        data,
    })
}

/// Reorder Fortran-ordered element bytes into C order.
///
/// Arrays that are already unambiguous (fewer than two axes, empty, or
/// backed by too little data) are returned unchanged.
pub fn fortran_to_c(raw: &[u8], shape: &[usize], item: usize) -> Vec<u8> {
    let count: usize = shape.iter().product();
    let Some(total) = count.checked_mul(item) else {
        return raw.to_vec();
    };
    if shape.len() < 2 || count == 0 || raw.len() < total {
        return raw.to_vec();
    }
    let mut out = vec![0u8; total];
    let mut strides = vec![1usize; shape.len()];
    for axis in 1..shape.len() {
        strides[axis] = strides[axis - 1] * shape[axis - 1];
    }
    let mut index = vec![0usize; shape.len()];
    for c_pos in 0..count {
        let src: usize = index.iter().zip(&strides).map(|(i, s)| i * s).sum();
        out[c_pos * item..(c_pos + 1) * item].copy_from_slice(&raw[src * item..(src + 1) * item]);
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

/// Serialise an array into an NPY 1.0 byte stream.
pub fn serialize(array: &NpyArray) -> Result<Vec<u8>> {
    let item = item_size(&array.descr)?;
    let expected = array
        .element_count()
        .checked_mul(item)
        .context("NPY array size overflows")?;
    if array.data.len() != expected {
        bail!(
            "array data length {} does not match shape {:?} and dtype '{}'",
            array.data.len(),
            array.shape,
            array.descr
        );
    }
    let shape_repr = match array.shape.len() {
        0 => "()".to_string(),
        1 => format!("({},)", array.shape[0]),
        _ => {
            let dims: Vec<String> = array.shape.iter().map(|d| d.to_string()).collect();
            format!("({})", dims.join(", "))
        }
    };
    let mut header = format!(
        "{{'descr': '{}', 'fortran_order': False, 'shape': {}, }}",
        array.descr, shape_repr
    );
    // numpy pads the header so that the data section starts 64-byte aligned.
    let unpadded = 10 + header.len() + 1;
    let padding = (64 - (unpadded % 64)) % 64;
    for _ in 0..padding {
        header.push(' ');
    }
    header.push('\n');
    let header_len =
        u16::try_from(header.len()).context("NPY header is too large for format version 1.0")?;
    let mut out = Vec::with_capacity(10 + header.len() + array.data.len());
    out.extend_from_slice(MAGIC);
    out.push(1);
    out.push(0);
    out.extend_from_slice(&header_len.to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(&array.data);
    Ok(out)
}

/// Decode a single element.
fn read_scalar(bytes: &[u8], kind: char, size: usize, little: bool) -> Result<Scalar> {
    if bytes.len() < size {
        bail!("dtype element is truncated");
    }
    let mut bits: u64 = 0;
    if little {
        for i in (0..size).rev() {
            bits = (bits << 8) | u64::from(bytes[i]);
        }
    } else {
        for &b in bytes.iter().take(size) {
            bits = (bits << 8) | u64::from(b);
        }
    }
    match kind {
        'f' => match size {
            4 => Ok(Scalar::Float(f64::from(f32::from_bits(bits as u32)))),
            8 => Ok(Scalar::Float(f64::from_bits(bits))),
            other => bail!("unsupported float width {other}"),
        },
        'i' => {
            let shift = 64 - size * 8;
            if shift == 0 {
                Ok(Scalar::Signed(bits as i64))
            } else {
                Ok(Scalar::Signed(((bits << shift) as i64) >> shift))
            }
        }
        'u' | 'b' => Ok(Scalar::Unsigned(bits)),
        other => bail!("unsupported dtype kind '{other}'"),
    }
}

/// Encode a single element in little-endian order.
fn write_scalar(out: &mut Vec<u8>, value: Scalar, kind: char, size: usize) -> Result<()> {
    let bits: u64 = match kind {
        'f' => match size {
            4 => u64::from((value.as_f64() as f32).to_bits()),
            8 => value.as_f64().to_bits(),
            other => bail!("unsupported float width {other}"),
        },
        'i' | 'u' | 'b' => value.as_i64() as u64,
        other => bail!("unsupported dtype kind '{other}'"),
    };
    for i in 0..size {
        out.push(((bits >> (8 * i)) & 0xff) as u8);
    }
    Ok(())
}

/// Convert an array to another numeric dtype (little-endian output).
pub fn cast(array: &NpyArray, target: &str) -> Result<NpyArray> {
    if array.descr == target {
        return Ok(array.clone());
    }
    let src_item = item_size(&array.descr)?;
    let dst_item = item_size(target)?;
    let src_kind = kind(&array.descr);
    let dst_kind = kind(target);
    let src_little = byte_order(&array.descr) != '>';
    let count = array.element_count();
    let needed = count
        .checked_mul(src_item)
        .context("array size overflows")?;
    if array.data.len() < needed {
        bail!(
            "array data is {} bytes, expected {needed} for shape {:?}",
            array.data.len(),
            array.shape
        );
    }
    let mut data = Vec::with_capacity(count * dst_item);
    for i in 0..count {
        let chunk = &array.data[i * src_item..(i + 1) * src_item];
        let value = read_scalar(chunk, src_kind, src_item, src_little)?;
        write_scalar(&mut data, value, dst_kind, dst_item)?;
    }
    Ok(NpyArray {
        descr: target.to_string(),
        shape: array.shape.clone(),
        data,
    })
}

/// Decode every element as `f64`.
pub fn to_f64_vec(array: &NpyArray) -> Result<Vec<f64>> {
    let wide = cast(array, "<f8")?;
    let mut out = Vec::with_capacity(wide.element_count());
    for chunk in wide.data.chunks_exact(8) {
        let mut buf = [0u8; 8];
        buf.copy_from_slice(chunk);
        out.push(f64::from_le_bytes(buf));
    }
    Ok(out)
}

/// Decode every element as `i64`.
pub fn to_i64_vec(array: &NpyArray) -> Result<Vec<i64>> {
    let wide = cast(array, "<i8")?;
    let mut out = Vec::with_capacity(wide.element_count());
    for chunk in wide.data.chunks_exact(8) {
        let mut buf = [0u8; 8];
        buf.copy_from_slice(chunk);
        out.push(i64::from_le_bytes(buf));
    }
    Ok(out)
}

/// Build a float64 array from values given in C order.
pub fn from_f64(shape: Vec<usize>, values: &[f64]) -> Result<NpyArray> {
    let count: usize = shape.iter().product();
    if values.len() != count {
        bail!(
            "expected {count} values for shape {shape:?}, found {}",
            values.len()
        );
    }
    let mut data = Vec::with_capacity(count * 8);
    for value in values {
        data.extend_from_slice(&value.to_le_bytes());
    }
    Ok(NpyArray {
        descr: "<f8".to_string(),
        shape,
        data,
    })
}

/// Take `array[..., start..end]` along the last axis.
pub fn slice_last_axis(array: &NpyArray, start: usize, end: usize) -> Result<NpyArray> {
    let last = *array
        .shape
        .last()
        .context("cannot slice a zero-dimensional array")?;
    if start > end || end > last {
        bail!("slice {start}..{end} is out of bounds for a last axis of length {last}");
    }
    let item = item_size(&array.descr)?;
    let outer: usize = array.shape[..array.shape.len() - 1].iter().product();
    let needed = outer
        .checked_mul(last)
        .and_then(|n| n.checked_mul(item))
        .context("array size overflows")?;
    if array.data.len() < needed {
        bail!(
            "array data is {} bytes, expected {needed} for shape {:?}",
            array.data.len(),
            array.shape
        );
    }
    let width = end - start;
    let mut data = Vec::with_capacity(outer * width * item);
    for row in 0..outer {
        let base = (row * last + start) * item;
        data.extend_from_slice(&array.data[base..base + width * item]);
    }
    let mut shape = array.shape.clone();
    if let Some(dim) = shape.last_mut() {
        *dim = width;
    }
    Ok(NpyArray {
        descr: array.descr.clone(),
        shape,
        data,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression guard for the `manual_unwrap_or` clippy fix on
    /// [`byte_order`]: `descr.chars().next().unwrap_or('<')` is *not*
    /// equivalent here, because it would return the descriptor's own first
    /// character (e.g. `'f'` for `"f4"`) instead of defaulting to `'<'` for
    /// an unprefixed descriptor.
    #[test]
    fn byte_order_defaults_unprefixed_descriptors_to_little_endian() {
        assert_eq!(byte_order("f4"), '<');
        assert_eq!(byte_order("i8"), '<');
        assert_eq!(byte_order(""), '<');
    }

    #[test]
    fn byte_order_recognises_every_prefix_character() {
        assert_eq!(byte_order("<f4"), '<');
        assert_eq!(byte_order(">f4"), '>');
        assert_eq!(byte_order("|u1"), '|');
        assert_eq!(byte_order("=f8"), '=');
    }
}
