//! Pure-Rust ingest of PyTorch `.pt` / `.pth` checkpoints.
//!
//! A modern `torch.save` file is a ZIP archive holding one pickle
//! (`<name>/data.pkl`) that describes the object graph, plus one member per
//! tensor storage (`<name>/data/<key>`) holding raw little-endian element
//! bytes. The pickle refers to those storages by *persistent id* rather than
//! inlining them, which is why the unpickler in [`super::vm`] delegates
//! persistent-id resolution to a container-aware resolver -- implemented
//! here against [`oxiarc_archive`]'s ZIP reader.
//!
//! Nothing is executed. `torch._utils._rebuild_tensor_v2(...)` and friends
//! arrive as inert [`Value::Object`] records, and this module *interprets*
//! the ones it recognizes, rejecting the rest with a specific error rather
//! than guessing.

use super::error::{PickleError, Result};
use super::value::Value;
use super::vm::{self, PersistentIdResolver};
use safetensors::Dtype;
use std::collections::HashMap;
use std::io::{Cursor, Read, Seek};
use std::path::Path;

/// Maximum number of elements a single tensor may declare, as a guard
/// against a crafted header asking for a petabyte allocation. 2^34 elements
/// is 64 GiB even at one byte each -- far beyond any real checkpoint tensor,
/// and far below what would exhaust address space during validation.
const MAX_TENSOR_ELEMENTS: usize = 1 << 34;

/// One tensor recovered from a checkpoint, with its data already
/// materialized in row-major (C-contiguous) order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TorchTensor {
    /// Dotted state-dict key, e.g. `unet.conv_in.weight`.
    pub name: String,
    /// Element type.
    pub dtype: Dtype,
    /// Dimensions, outermost first.
    pub shape: Vec<usize>,
    /// Little-endian element bytes, row-major.
    pub data: Vec<u8>,
}

/// Reads every tensor from a PyTorch `.pt` / `.pth` checkpoint.
///
/// The top-level object may be the state dict itself, or a wrapper dict with
/// a `state_dict` key (the Lightning / ImageDream convention); both are
/// accepted. Nested dicts are flattened with `.`-joined keys, matching how
/// PyTorch itself names parameters.
///
/// # Errors
///
/// Returns [`PickleError`] if the file is not a readable checkpoint
/// archive, if its pickle is malformed, if it uses a tensor rebuild function
/// or dtype this reader does not implement, or if a referenced storage is
/// missing or too small.
pub fn read_checkpoint(path: &Path) -> Result<Vec<TorchTensor>> {
    let file = std::fs::File::open(path)?;
    read_checkpoint_from(file)
}

/// Reads a checkpoint from any seekable reader (a file, or an in-memory
/// buffer in tests).
///
/// # Errors
///
/// As [`read_checkpoint`].
pub fn read_checkpoint_from<R: Read + Seek>(mut reader: R) -> Result<Vec<TorchTensor>> {
    reject_legacy_format(&mut reader)?;
    let mut archive = ZipCheckpoint::open(reader)?;
    let pickle = archive.read_pickle()?;

    let root = {
        let mut resolver = StorageResolver {
            archive: &mut archive,
        };
        vm::load_with(&pickle, &mut resolver)?
    };

    let state = unwrap_state_dict(&root)?;
    let mut tensors = Vec::new();
    flatten(state, "", &mut archive, &mut tensors)?;
    if tensors.is_empty() {
        return Err(PickleError::Structure(
            "checkpoint contains no tensors".to_string(),
        ));
    }
    Ok(tensors)
}

/// Rejects a non-ZIP input up front with an actionable message.
///
/// `torch.save` produced a bare pickle (not a ZIP) before PyTorch 1.6, and
/// such a file is still common in the wild. Detecting it here -- by its
/// missing ZIP local-file-header magic -- lets the error say *what* the file
/// is and how to fix it, instead of the ZIP reader's generic "no data.pkl
/// member", which reads like a corrupt archive rather than a format
/// mismatch.
fn reject_legacy_format<R: Read + Seek>(reader: &mut R) -> Result<()> {
    let start = reader.stream_position()?;
    let mut magic = [0u8; 4];
    let read = read_up_to(reader, &mut magic)?;
    reader.seek(std::io::SeekFrom::Start(start))?;

    // "PK\x03\x04" (local file header) or "PK\x05\x06" (empty archive).
    if read == 4 && magic[0] == b'P' && magic[1] == b'K' {
        return Ok(());
    }
    Err(PickleError::Archive(format!(
        "input is not a ZIP archive (magic {magic:02x?}); legacy pre-1.6 \
         `torch.save` files are a bare pickle rather than an archive and are \
         not supported -- re-save with a modern PyTorch \
         (`torch.save(sd, path)` on 1.6+), or export to .safetensors"
    )))
}

/// Fills `buf` as far as the reader allows, returning how many bytes were
/// read. A short read is not an error here: a file too small to hold a ZIP
/// header is simply not a ZIP.
fn read_up_to<R: Read>(reader: &mut R, buf: &mut [u8]) -> Result<usize> {
    let mut filled = 0;
    while filled < buf.len() {
        match reader.read(&mut buf[filled..])? {
            0 => break,
            n => filled += n,
        }
    }
    Ok(filled)
}

/// Follows the common `{"state_dict": {...}}` wrapper, and rejects a
/// top-level object that is not a mapping at all.
fn unwrap_state_dict(root: &Value) -> Result<&Value> {
    if let Some(inner) = root.get("state_dict") {
        if inner.as_mapping().is_some() {
            return Ok(inner);
        }
    }
    if root.as_mapping().is_some() {
        return Ok(root);
    }
    Err(PickleError::Structure(format!(
        "expected the checkpoint's top-level object to be a state dict, found {}",
        root
    )))
}

/// Walks a (possibly nested) state dict, materializing every tensor it
/// finds under its `.`-joined key.
fn flatten<R: Read + Seek>(
    node: &Value,
    prefix: &str,
    archive: &mut ZipCheckpoint<R>,
    out: &mut Vec<TorchTensor>,
) -> Result<()> {
    let Some(entries) = node.as_mapping() else {
        return Ok(());
    };
    for (key, value) in entries {
        let Some(key) = key.as_text() else {
            continue;
        };
        let name = if prefix.is_empty() {
            key
        } else {
            format!("{prefix}.{key}")
        };

        if let Some(tensor) = try_rebuild_tensor(&name, value, archive)? {
            out.push(tensor);
        } else if value.as_mapping().is_some() {
            flatten(value, &name, archive, out)?;
        }
        // Anything else (ints, strings, optimizer bookkeeping) is not a
        // tensor and is skipped: a checkpoint routinely carries an `epoch`
        // or `global_step` alongside its weights, and refusing the whole
        // file over one would be useless.
    }
    Ok(())
}

/// Recognizes a tensor-rebuild record and materializes it, or returns
/// `Ok(None)` if `value` is not a tensor at all.
///
/// # Errors
///
/// Returns an error when `value` *is* a `torch._utils` rebuild call but of a
/// kind this reader does not implement -- silently skipping such a tensor
/// would produce a checkpoint that is quietly missing weights.
fn try_rebuild_tensor<R: Read + Seek>(
    name: &str,
    value: &Value,
    archive: &mut ZipCheckpoint<R>,
) -> Result<Option<TorchTensor>> {
    let Some((module, function)) = value.class_path() else {
        return Ok(None);
    };
    if module != "torch._utils" {
        return Ok(None);
    }

    match function {
        // _rebuild_tensor_v2(storage, storage_offset, size, stride,
        //                    requires_grad, backward_hooks[, metadata])
        // _rebuild_tensor(storage, storage_offset, size, stride)
        "_rebuild_tensor" | "_rebuild_tensor_v2" | "_rebuild_tensor_v3" => {
            let args = value.ctor_args().ok_or_else(|| {
                PickleError::Structure(format!("{name}: {function} has no argument tuple"))
            })?;
            rebuild_tensor(name, args, archive).map(Some)
        }
        other => Err(PickleError::Structure(format!(
            "{name}: unsupported tensor rebuild function torch._utils.{other}; \
             this reader implements _rebuild_tensor, _rebuild_tensor_v2 and \
             _rebuild_tensor_v3 (sparse and quantized tensors are not supported)"
        ))),
    }
}

fn rebuild_tensor<R: Read + Seek>(
    name: &str,
    args: &[Value],
    archive: &mut ZipCheckpoint<R>,
) -> Result<TorchTensor> {
    let structure = |what: &str| PickleError::Structure(format!("{name}: {what}"));

    let storage = args
        .first()
        .ok_or_else(|| structure("rebuild call is missing its storage argument"))?;
    let StorageRef { key, dtype } = storage_ref(storage)
        .ok_or_else(|| structure("first rebuild argument is not a tensor storage"))?;

    let offset = args
        .get(1)
        .and_then(Value::as_usize)
        .ok_or_else(|| structure("rebuild call has a missing or negative storage offset"))?;
    let shape = int_seq(args.get(2))
        .ok_or_else(|| structure("rebuild call has a missing or invalid size tuple"))?;
    let stride = int_seq(args.get(3))
        .ok_or_else(|| structure("rebuild call has a missing or invalid stride tuple"))?;

    if stride.len() != shape.len() {
        return Err(structure(&format!(
            "stride has {} entries but size has {}",
            stride.len(),
            shape.len()
        )));
    }

    let element_count = shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .filter(|&n| n <= MAX_TENSOR_ELEMENTS)
        .ok_or_else(|| structure("size tuple describes an implausibly large tensor"))?;

    let element_size = dtype_size(dtype);
    let raw = archive.storage_bytes(&key)?;

    let data = gather_strided(
        &raw,
        offset,
        &shape,
        &stride,
        element_size,
        element_count,
        &key,
    )?;

    Ok(TorchTensor {
        name: name.to_string(),
        dtype,
        shape,
        data,
    })
}

/// Copies a tensor's elements out of its backing storage in row-major
/// order, honouring an arbitrary stride.
///
/// The fast path is a plain slice copy for the overwhelmingly common
/// C-contiguous case. The general path exists because a checkpoint *can*
/// legitimately hold a non-contiguous view -- a transposed weight, or a
/// tensor sharing storage with another -- and safetensors has no notion of
/// stride, so the data must be materialized rather than the tensor rejected.
fn gather_strided(
    raw: &[u8],
    offset: usize,
    shape: &[usize],
    stride: &[usize],
    element_size: usize,
    element_count: usize,
    key: &str,
) -> Result<Vec<u8>> {
    let too_small = |needed: usize| PickleError::Storage {
        key: key.to_string(),
        problem: format!(
            "holds {} bytes but the tensor needs {} (offset {} elements, {} bytes each)",
            raw.len(),
            needed,
            offset,
            element_size
        ),
    };

    if element_count == 0 {
        return Ok(Vec::new());
    }

    if is_contiguous(shape, stride) {
        let start = offset
            .checked_mul(element_size)
            .ok_or_else(|| too_small(usize::MAX))?;
        let len = element_count
            .checked_mul(element_size)
            .ok_or_else(|| too_small(usize::MAX))?;
        let end = start
            .checked_add(len)
            .ok_or_else(|| too_small(usize::MAX))?;
        return raw
            .get(start..end)
            .map(<[u8]>::to_vec)
            .ok_or_else(|| too_small(end));
    }

    // Bound the allocation by what the storage could possibly supply before
    // reserving it. A crafted header can declare a huge shape over a tiny
    // storage; the per-element bounds check below would catch it, but only
    // after `with_capacity` had already tried to reserve the full amount.
    let needed = element_count
        .checked_mul(element_size)
        .ok_or_else(|| too_small(usize::MAX))?;
    if needed > raw.len() {
        return Err(too_small(needed));
    }

    let mut out = Vec::with_capacity(needed);
    let mut index = vec![0usize; shape.len()];
    for _ in 0..element_count {
        let mut element = offset;
        for (axis, &i) in index.iter().enumerate() {
            element = element
                .checked_add(
                    i.checked_mul(stride[axis])
                        .ok_or_else(|| too_small(usize::MAX))?,
                )
                .ok_or_else(|| too_small(usize::MAX))?;
        }
        let start = element
            .checked_mul(element_size)
            .ok_or_else(|| too_small(usize::MAX))?;
        let end = start
            .checked_add(element_size)
            .ok_or_else(|| too_small(usize::MAX))?;
        out.extend_from_slice(raw.get(start..end).ok_or_else(|| too_small(end))?);

        // Odometer increment, last axis fastest -- row-major output order.
        for axis in (0..shape.len()).rev() {
            index[axis] += 1;
            if index[axis] < shape[axis] {
                break;
            }
            index[axis] = 0;
        }
    }
    Ok(out)
}

/// Whether `stride` is the C-contiguous stride for `shape`.
///
/// Axes of length 1 are ignored: PyTorch leaves their stride arbitrary
/// (a `[1, N]` tensor may report stride `[N, 1]` or `[1, 1]`), and either
/// describes the same, contiguous, memory.
fn is_contiguous(shape: &[usize], stride: &[usize]) -> bool {
    let mut expected = 1usize;
    for axis in (0..shape.len()).rev() {
        if shape[axis] != 1 && stride[axis] != expected {
            return false;
        }
        expected = match expected.checked_mul(shape[axis]) {
            Some(next) => next,
            None => return false,
        };
    }
    true
}

/// A resolved reference to one storage member of the archive.
struct StorageRef {
    key: String,
    dtype: Dtype,
}

/// Extracts the storage key and dtype from a resolved persistent id.
fn storage_ref(value: &Value) -> Option<StorageRef> {
    let Value::PersistentId(id) = value else {
        return None;
    };
    let parts = id.as_seq()?;
    // ('storage', <torch.XStorage>, key, location, numel)
    if parts.first()?.as_text().as_deref() != Some("storage") {
        return None;
    }
    let (_, storage_type) = parts.get(1)?.class_path()?;
    let key = parts.get(2)?.as_text()?;
    Some(StorageRef {
        key,
        dtype: storage_dtype(storage_type)?,
    })
}

/// Maps a `torch.<X>Storage` class name to the safetensors dtype holding the
/// same bytes.
fn storage_dtype(storage_type: &str) -> Option<Dtype> {
    Some(match storage_type {
        "FloatStorage" => Dtype::F32,
        "DoubleStorage" => Dtype::F64,
        "HalfStorage" => Dtype::F16,
        "BFloat16Storage" => Dtype::BF16,
        "LongStorage" => Dtype::I64,
        "IntStorage" => Dtype::I32,
        "ShortStorage" => Dtype::I16,
        "CharStorage" => Dtype::I8,
        "ByteStorage" => Dtype::U8,
        "BoolStorage" => Dtype::BOOL,
        _ => return None,
    })
}

/// Byte width of one element.
fn dtype_size(dtype: Dtype) -> usize {
    match dtype {
        Dtype::F64 | Dtype::I64 | Dtype::U64 => 8,
        Dtype::F32 | Dtype::I32 | Dtype::U32 => 4,
        Dtype::F16 | Dtype::BF16 | Dtype::I16 | Dtype::U16 => 2,
        _ => 1,
    }
}

/// Reads a tuple/list of non-negative integers.
fn int_seq(value: Option<&Value>) -> Option<Vec<usize>> {
    value?.as_seq()?.iter().map(Value::as_usize).collect()
}

// ---------------------------------------------------------------------------
// ZIP container
// ---------------------------------------------------------------------------

/// The ZIP container a `torch.save` file is, indexed by suffix so the
/// archive's arbitrary top-level directory name does not matter.
struct ZipCheckpoint<R: Read + Seek> {
    reader: oxiarc_archive::zip::ZipReader<R>,
    /// Storage key (`"0"`, `"1"`, …) -> index into the archive's entry list.
    storages: HashMap<String, usize>,
    /// Index of the `data.pkl` member.
    pickle_index: Option<usize>,
    /// Storages already extracted, so a checkpoint whose tensors share a
    /// storage (views) decompresses it once.
    cache: HashMap<String, Vec<u8>>,
}

impl<R: Read + Seek> ZipCheckpoint<R> {
    fn open(reader: R) -> Result<Self> {
        let reader = oxiarc_archive::zip::ZipReader::new(reader)
            .map_err(|e| PickleError::Archive(format!("not a readable checkpoint archive: {e}")))?;

        let mut storages = HashMap::new();
        let mut pickle_index = None;
        for (index, entry) in reader.entries().iter().enumerate() {
            let name = entry.name.replace('\\', "/");
            if name.ends_with("data.pkl") {
                pickle_index = Some(index);
            } else if let Some((_, key)) = name.rsplit_once("/data/") {
                if !key.is_empty() && !key.contains('/') {
                    storages.insert(key.to_string(), index);
                }
            }
        }

        Ok(Self {
            reader,
            storages,
            pickle_index,
            cache: HashMap::new(),
        })
    }

    fn read_pickle(&mut self) -> Result<Vec<u8>> {
        let index = self.pickle_index.ok_or_else(|| {
            PickleError::Archive(
                "archive has no data.pkl member; it is not a PyTorch checkpoint".to_string(),
            )
        })?;
        self.extract(index)
    }

    fn storage_bytes(&mut self, key: &str) -> Result<Vec<u8>> {
        if let Some(cached) = self.cache.get(key) {
            return Ok(cached.clone());
        }
        let index = *self.storages.get(key).ok_or_else(|| PickleError::Storage {
            key: key.to_string(),
            problem: "is referenced by a tensor but absent from the archive".to_string(),
        })?;
        let bytes = self.extract(index)?;
        self.cache.insert(key.to_string(), bytes.clone());
        Ok(bytes)
    }

    fn extract(&mut self, index: usize) -> Result<Vec<u8>> {
        let entry = self
            .reader
            .entries()
            .get(index)
            .ok_or_else(|| PickleError::Archive("archive entry disappeared".to_string()))?
            .clone();
        self.reader
            .extract(&entry)
            .map_err(|e| PickleError::Archive(format!("failed to extract '{}': {e}", entry.name)))
    }
}

/// Resolves `('storage', …)` persistent ids by recording the reference; the
/// bytes are pulled later, once the tensor's shape and offset are known, so
/// a storage shared by several views is not copied per reference.
struct StorageResolver<'a, R: Read + Seek> {
    archive: &'a mut ZipCheckpoint<R>,
}

impl<R: Read + Seek> PersistentIdResolver for StorageResolver<'_, R> {
    fn resolve(&mut self, id: &Value) -> Result<Value> {
        let recorded = Value::PersistentId(Box::new(id.clone()));
        // Validate eagerly so a checkpoint referencing a missing storage
        // fails with the storage's name rather than much later, or not at
        // all if the tensor is skipped.
        if let Some(StorageRef { key, .. }) = storage_ref(&recorded) {
            if !self.archive.storages.contains_key(&key) {
                return Err(PickleError::Storage {
                    key,
                    problem: "is referenced by the pickle but absent from the archive".to_string(),
                });
            }
        }
        Ok(recorded)
    }
}

/// Reads a checkpoint from an in-memory buffer.
///
/// # Errors
///
/// As [`read_checkpoint`].
pub fn read_checkpoint_bytes(bytes: &[u8]) -> Result<Vec<TorchTensor>> {
    read_checkpoint_from(Cursor::new(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pickle::test_support::pickle;

    /// One tensor's rebuild description.
    struct TensorSpec {
        name: String,
        storage_type: &'static str,
        storage_key: String,
        shape: Vec<usize>,
        stride: Vec<usize>,
        offset: usize,
    }

    /// Builds a minimal but *real* `.pt` archive: a ZIP holding a
    /// `data.pkl` describing one `_rebuild_tensor_v2` call per tensor, plus
    /// one `data/<key>` member per storage. Hand-building it (rather than
    /// shipping a binary fixture) keeps the test hermetic and documents the
    /// container layout the reader parses.
    struct CheckpointBuilder {
        entries: Vec<(String, Vec<u8>)>,
        tensors: Vec<TensorSpec>,
    }

    impl CheckpointBuilder {
        fn new() -> Self {
            Self {
                entries: Vec::new(),
                tensors: Vec::new(),
            }
        }

        fn storage(mut self, key: &str, bytes: Vec<u8>) -> Self {
            self.entries.push((format!("archive/data/{key}"), bytes));
            self
        }

        fn tensor(
            mut self,
            name: &str,
            storage_type: &'static str,
            key: &str,
            shape: Vec<usize>,
            stride: Vec<usize>,
            offset: usize,
        ) -> Self {
            self.tensors.push(TensorSpec {
                name: name.to_string(),
                storage_type,
                storage_key: key.to_string(),
                shape,
                stride,
                offset,
            });
            self
        }

        fn build(self) -> Vec<u8> {
            let tensors = self.tensors;
            let pickle_bytes = pickle(|p| {
                p.empty_dict();
                for spec in &tensors {
                    p.unicode(&spec.name);
                    p.global("torch._utils", "_rebuild_tensor_v2");
                    // MARK, then the six rebuild arguments, then TUPLE.
                    p.mark();
                    // ('storage', torch.XStorage, key, 'cpu', numel)
                    p.mark();
                    p.unicode("storage");
                    p.global("torch", spec.storage_type);
                    p.unicode(&spec.storage_key);
                    p.unicode("cpu");
                    p.int(0);
                    p.tuple();
                    p.binpersid();
                    p.int(spec.offset as i64);
                    p.int_tuple(&spec.shape);
                    p.int_tuple(&spec.stride);
                    p.bool(false);
                    p.empty_dict();
                    p.tuple();
                    p.reduce();
                    p.setitem();
                }
            });

            let mut entries = self.entries;
            entries.push(("archive/data.pkl".to_string(), pickle_bytes));
            entries.push(("archive/version".to_string(), b"3\n".to_vec()));
            zip_of(entries)
        }
    }

    fn zip_of(entries: Vec<(String, Vec<u8>)>) -> Vec<u8> {
        let mut buffer = Vec::new();
        {
            let mut writer = oxiarc_archive::zip::ZipWriter::new(Cursor::new(&mut buffer));
            for (name, data) in &entries {
                writer
                    .add_file_stored(name, data)
                    .expect("test: zip write should succeed");
            }
            writer.finish().expect("test: zip finish should succeed");
        }
        buffer
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    #[test]
    fn test_reads_a_contiguous_f32_tensor() {
        let archive = CheckpointBuilder::new()
            .storage("0", f32_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
            .tensor(
                "conv.weight",
                "FloatStorage",
                "0",
                vec![2, 3],
                vec![3, 1],
                0,
            )
            .build();

        let tensors = read_checkpoint_bytes(&archive).expect("test: checkpoint should read");
        assert_eq!(tensors.len(), 1);
        assert_eq!(tensors[0].name, "conv.weight");
        assert_eq!(tensors[0].dtype, Dtype::F32);
        assert_eq!(tensors[0].shape, vec![2, 3]);
        assert_eq!(tensors[0].data, f32_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
    }

    #[test]
    fn test_honors_storage_offset_and_shared_storage() {
        // Two tensors viewing disjoint windows of one storage -- exactly
        // what `torch.save` writes when parameters were sliced from a
        // single buffer.
        let archive = CheckpointBuilder::new()
            .storage("0", f32_bytes(&[1.0, 2.0, 3.0, 4.0]))
            .tensor("a", "FloatStorage", "0", vec![2], vec![1], 0)
            .tensor("b", "FloatStorage", "0", vec![2], vec![1], 2)
            .build();

        let tensors = read_checkpoint_bytes(&archive).expect("test: checkpoint should read");
        let a = tensors
            .iter()
            .find(|t| t.name == "a")
            .expect("test: tensor 'a'");
        let b = tensors
            .iter()
            .find(|t| t.name == "b")
            .expect("test: tensor 'b'");
        assert_eq!(a.data, f32_bytes(&[1.0, 2.0]));
        assert_eq!(b.data, f32_bytes(&[3.0, 4.0]));
    }

    #[test]
    fn test_materializes_a_non_contiguous_transposed_view() {
        // A [2,3] row-major buffer viewed as its [3,2] transpose has stride
        // [1,3]. safetensors has no stride, so the bytes must be gathered
        // into row-major order rather than copied verbatim (which would
        // silently transpose every weight in the file).
        let archive = CheckpointBuilder::new()
            .storage("0", f32_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
            .tensor("t", "FloatStorage", "0", vec![3, 2], vec![1, 3], 0)
            .build();

        let tensors = read_checkpoint_bytes(&archive).expect("test: checkpoint should read");
        assert_eq!(tensors[0].shape, vec![3, 2]);
        assert_eq!(tensors[0].data, f32_bytes(&[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]));
    }

    #[test]
    fn test_dtypes_map_to_safetensors() {
        let archive = CheckpointBuilder::new()
            .storage("0", vec![0x00, 0x3c]) // f16 1.0
            .storage("1", 1i64.to_le_bytes().to_vec())
            .storage("2", vec![1u8])
            .tensor("h", "HalfStorage", "0", vec![1], vec![1], 0)
            .tensor("l", "LongStorage", "1", vec![1], vec![1], 0)
            .tensor("m", "BoolStorage", "2", vec![1], vec![1], 0)
            .build();

        let tensors = read_checkpoint_bytes(&archive).expect("test: checkpoint should read");
        let dtype_of = |name: &str| {
            tensors
                .iter()
                .find(|t| t.name == name)
                .map(|t| t.dtype)
                .expect("test: tensor present")
        };
        assert_eq!(dtype_of("h"), Dtype::F16);
        assert_eq!(dtype_of("l"), Dtype::I64);
        assert_eq!(dtype_of("m"), Dtype::BOOL);
    }

    #[test]
    fn test_truncated_storage_is_reported_not_panicked_on() {
        // Regression guard: the shape says four elements, the storage holds
        // two. Slicing without a bounds check would panic on input this
        // reader is specifically meant to survive.
        let archive = CheckpointBuilder::new()
            .storage("0", f32_bytes(&[1.0, 2.0]))
            .tensor("w", "FloatStorage", "0", vec![4], vec![1], 0)
            .build();

        let err = read_checkpoint_bytes(&archive).expect_err("truncated storage must error");
        assert!(
            matches!(err, PickleError::Storage { .. }),
            "expected a storage error, got {err}"
        );
    }

    #[test]
    fn test_oversized_strided_tensor_is_rejected() {
        // A crafted header declaring an enormous non-contiguous shape over a
        // tiny storage must be rejected, and rejected *before*
        // `gather_strided` reserves the full declared size (here 4 GiB) on
        // the say-so of the untrusted header alone. The size is reconciled
        // against the actual storage first.
        //
        // What this test pins is the rejection and its message. The
        // reservation itself is not directly observable -- a `with_capacity`
        // that is never written to costs only virtual address space on the
        // platforms tested here -- so the guard is defence in depth for
        // memory-constrained and 32-bit targets, where the reservation does
        // fail. Do not read a pass here as proof the reservation is absent;
        // read it as proof the malformed input is refused with a useful
        // error rather than accepted or panicked on.
        //
        // Stride [1, N] rather than the contiguous [N, 1] forces the
        // gathering path where the reservation lives.
        const HUGE: usize = 1 << 15; // 32768^2 elements * 4 bytes = 4 GiB
        let archive = CheckpointBuilder::new()
            .storage("0", f32_bytes(&[1.0, 2.0]))
            .tensor(
                "huge",
                "FloatStorage",
                "0",
                vec![HUGE, HUGE],
                vec![1, HUGE],
                0,
            )
            .build();

        let err = read_checkpoint_bytes(&archive).expect_err("oversized tensor must error");
        assert!(
            matches!(err, PickleError::Storage { .. }),
            "expected a storage error, got {err}"
        );
        // The message must name the real storage size, so the operator can
        // see the header is lying rather than that their machine is small.
        assert!(
            err.to_string().contains("8 bytes"),
            "error should report the actual storage size, got: {err}"
        );
    }

    #[test]
    fn test_element_count_limit_rejects_an_absurd_shape() {
        // Above MAX_TENSOR_ELEMENTS the shape is refused outright, before
        // any per-element reasoning at all.
        let archive = CheckpointBuilder::new()
            .storage("0", f32_bytes(&[1.0]))
            .tensor(
                "absurd",
                "FloatStorage",
                "0",
                vec![MAX_TENSOR_ELEMENTS, 2],
                vec![2, 1],
                0,
            )
            .build();

        let err = read_checkpoint_bytes(&archive).expect_err("absurd shape must error");
        assert!(err.to_string().contains("implausibly large"), "got: {err}");
    }

    #[test]
    fn test_missing_storage_is_reported() {
        let archive = CheckpointBuilder::new()
            .tensor("w", "FloatStorage", "7", vec![1], vec![1], 0)
            .build();
        let err = read_checkpoint_bytes(&archive).expect_err("missing storage must error");
        assert!(matches!(err, PickleError::Storage { .. }));
    }

    #[test]
    fn test_non_zip_input_gets_an_actionable_error() {
        let err = read_checkpoint_bytes(b"\x80\x02}q\x00.")
            .expect_err("a bare pickle is not a .pt archive");
        let message = err.to_string();
        assert!(
            message.contains("legacy"),
            "the error should explain the legacy format, got: {message}"
        );
    }

    #[test]
    fn test_is_contiguous_ignores_unit_axes() {
        assert!(is_contiguous(&[2, 3], &[3, 1]));
        assert!(is_contiguous(&[1, 4], &[4, 1]));
        // PyTorch may report an arbitrary stride for a length-1 axis.
        assert!(is_contiguous(&[1, 4], &[1, 1]));
        assert!(!is_contiguous(&[3, 2], &[1, 3]));
    }
}
