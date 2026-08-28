//! Video sequences of FLAME parameters
//!
//! This module provides efficient handling of FLAME parameter sequences for video processing,
//! with support for lazy loading, caching, and interpolation.

use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

use lru::LruCache;
use serde::{Deserialize, Serialize};

use crate::error::FlameError;
use crate::params::FlameParams;

/// Default LRU cache size (number of frames to keep in memory)
/// Increased from 64 to 256 for better video playback performance
const DEFAULT_CACHE_SIZE: NonZeroUsize = match NonZeroUsize::new(256) {
    Some(n) => n,
    None => unreachable!(),
};

/// A sequence of FLAME parameters for video processing
///
/// `FlameSequence` provides efficient access to FLAME parameters across video frames
/// with lazy loading and LRU caching to minimize memory usage.
///
/// # Example
///
/// ```rust,no_run
/// use oxigaf_flame::sequence::FlameSequence;
/// use std::path::Path;
///
/// // Load from JSON file
/// let mut sequence = FlameSequence::from_json(Path::new("params.json"))?;
/// println!("Loaded {} frames at {} fps", sequence.num_frames(), sequence.fps().unwrap_or(30.0));
///
/// // Access frames
/// let frame_0 = sequence.get_frame(0)?;
/// let frame_10 = sequence.get_frame(10)?;
///
/// // Interpolate between frames
/// let interpolated = sequence.interpolate(5.5)?;
/// # Ok::<(), oxigaf_flame::FlameError>(())
/// ```
pub struct FlameSequence {
    /// Source of the sequence data
    source: SequenceSource,
    /// LRU cache for loaded frames
    cache: LruCache<usize, FlameParams>,
    /// Total number of frames
    num_frames: usize,
    /// Frames per second (optional)
    fps: Option<f32>,
}

/// Source of FLAME parameter sequence data
enum SequenceSource {
    /// All frames loaded in memory
    Memory(Vec<FlameParams>),
    /// Parsed JSON sequence held in memory; each frame is validated and
    /// converted to `FlameParams` lazily, on access.
    ///
    /// The source file is read and parsed exactly once (in `from_json`);
    /// unlike a naive "lazy" design that re-reads and re-parses the whole
    /// file on every single frame access, only the (cheap) `FrameJson` ->
    /// `FlameParams` conversion is deferred to access time.
    JsonFile {
        frames: Vec<FrameJson>,
        metadata: SequenceMetadata,
    },
    /// Decompressed NPZ arrays held in memory; each frame's `FlameParams`
    /// is constructed lazily, on access. As with `JsonFile`, the NPZ file
    /// itself is opened and decompressed exactly once (in `from_npz`).
    ///
    /// Only [`FlameSequence::from_npz`] constructs this, and that
    /// constructor exists only under the `npz` feature — without it the
    /// other `from_npz` is a stub that returns an error before any source
    /// is built. The variant is therefore gated on the same feature rather
    /// than carried (and suppressed with `#[allow(dead_code)]`) in default
    /// builds, where it is genuinely unreachable and would otherwise force
    /// every `SequenceSource` match to handle a state that cannot occur.
    #[cfg(feature = "npz")]
    NpzFile {
        shape: ndarray::Array2<f32>,
        expression: ndarray::Array2<f32>,
        pose: ndarray::Array2<f32>,
        translation: Box<Option<ndarray::Array2<f32>>>,
        metadata: SequenceMetadata,
    },
    /// Load from directory of per-frame files
    Directory {
        path: PathBuf,
        file_pattern: String,
        metadata: SequenceMetadata,
    },
}

/// Metadata about a sequence
#[derive(Debug, Clone)]
struct SequenceMetadata {
    num_frames: usize,
    fps: Option<f32>,
    n_shape: usize,
    n_expression: usize,
    n_pose: usize,
}

/// JSON format for sequence data
#[derive(Debug, Serialize, Deserialize)]
struct SequenceJson {
    fps: Option<f32>,
    frames: Vec<FrameJson>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FrameJson {
    shape: Vec<f32>,
    expression: Vec<f32>,
    pose: Vec<f32>,
    #[serde(default)]
    translation: Option<[f32; 3]>,
}

impl FlameSequence {
    /// Create a sequence from a vector of FLAME parameters
    ///
    /// # Arguments
    ///
    /// * `frames` - Vector of FLAME parameters for each frame
    /// * `fps` - Optional frames per second
    #[must_use]
    pub fn from_memory(frames: Vec<FlameParams>, fps: Option<f32>) -> Self {
        let num_frames = frames.len();
        Self {
            source: SequenceSource::Memory(frames),
            cache: LruCache::new(DEFAULT_CACHE_SIZE),
            num_frames,
            fps,
        }
    }

    /// Load a sequence from a JSON file
    ///
    /// # JSON Format
    ///
    /// ```json
    /// {
    ///   "fps": 30.0,
    ///   "frames": [
    ///     {
    ///       "shape": [0.1, -0.2, ...],
    ///       "expr": [0.0, 0.3, ...],
    ///       "pose": [0.0, 0.0, 0.0, ...],
    ///       "translation": [0.0, 0.0, 0.0]
    ///     },
    ///     ...
    ///   ]
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns error if file cannot be read or JSON is invalid
    pub fn from_json(path: &Path) -> Result<Self, FlameError> {
        tracing::info!("Loading FLAME sequence from JSON: {}", path.display());

        let json_str = std::fs::read_to_string(path).map_err(|e| FlameError::IoError {
            source: e,
            path: path.to_path_buf(),
        })?;

        let sequence_data: SequenceJson = serde_json::from_str(&json_str).map_err(|e| {
            FlameError::InvalidParams(format!("Failed to parse JSON sequence: {e}"))
        })?;

        if sequence_data.frames.is_empty() {
            return Err(FlameError::InvalidParams(
                "Sequence contains no frames".to_string(),
            ));
        }

        // Determine parameter dimensions from first frame
        let first_frame = &sequence_data.frames[0];
        let metadata = SequenceMetadata {
            num_frames: sequence_data.frames.len(),
            fps: sequence_data.fps,
            n_shape: first_frame.shape.len(),
            n_expression: first_frame.expression.len(),
            n_pose: first_frame.pose.len(),
        };

        // For small sequences, load all into memory
        // For large sequences (>1000 frames), use lazy loading
        if sequence_data.frames.len() <= 1000 {
            let frames: Result<Vec<FlameParams>, FlameError> = sequence_data
                .frames
                .into_iter()
                .map(|f| frame_json_to_params(f, &metadata))
                .collect();

            Ok(Self {
                source: SequenceSource::Memory(frames?),
                cache: LruCache::new(DEFAULT_CACHE_SIZE),
                num_frames: metadata.num_frames,
                fps: metadata.fps,
            })
        } else {
            // Lazy loading for large sequences: the parsed frames stay
            // resident (they were already fully parsed above, into
            // `sequence_data.frames`), so per-frame access below never
            // re-reads or re-parses the source file.
            Ok(Self {
                source: SequenceSource::JsonFile {
                    frames: sequence_data.frames,
                    metadata: metadata.clone(),
                },
                cache: LruCache::new(DEFAULT_CACHE_SIZE),
                num_frames: metadata.num_frames,
                fps: metadata.fps,
            })
        }
    }

    /// Load a sequence from an NPZ file
    ///
    /// # NPZ Format
    ///
    /// Expected arrays:
    /// - `shape`: [`num_frames`, `n_shape_params`]
    /// - `expression` or `expr`: [`num_frames`, `n_expression_params`]
    /// - `pose`: [`num_frames`, `n_pose_params`]
    /// - `translation` (optional): [`num_frames`, 3]
    /// - `fps`: scalar (optional)
    ///
    /// # Errors
    ///
    /// Returns error if file cannot be read or arrays are invalid
    ///
    /// # Feature Flag
    ///
    /// Requires the "npz" feature to be enabled.
    #[cfg(feature = "npz")]
    pub fn from_npz(path: &Path) -> Result<Self, FlameError> {
        tracing::info!("Loading FLAME sequence from NPZ: {}", path.display());

        let arrays = NpzArrays::read(path)?;
        let metadata = arrays.validated_metadata()?;
        let num_frames = metadata.num_frames;
        let fps = metadata.fps;

        // For small sequences, load all into memory
        // For large sequences (>1000 frames), use lazy loading
        if num_frames <= 1000 {
            Ok(Self {
                source: SequenceSource::Memory(arrays.into_frames()),
                cache: LruCache::new(DEFAULT_CACHE_SIZE),
                num_frames,
                fps,
            })
        } else {
            // Lazy loading for large sequences: the decompressed arrays
            // stay resident (this is the same data an eager load would
            // hold), so per-frame access below never re-opens or
            // re-decompresses the NPZ file.
            let NpzArrays {
                shape,
                expression,
                pose,
                translation,
                fps: _,
            } = arrays;
            Ok(Self {
                source: SequenceSource::NpzFile {
                    shape,
                    expression,
                    pose,
                    translation: Box::new(translation),
                    metadata,
                },
                cache: LruCache::new(DEFAULT_CACHE_SIZE),
                num_frames,
                fps,
            })
        }
    }

    /// Load a sequence from an NPZ file (feature not enabled)
    ///
    /// # Errors
    ///
    /// Returns error indicating that the "npz" feature is not enabled.
    #[cfg(not(feature = "npz"))]
    pub fn from_npz(_path: &Path) -> Result<Self, FlameError> {
        Err(FlameError::InvalidParams(
            "NPZ support not enabled. Enable the 'npz' feature flag.".to_string(),
        ))
    }

    /// Load a sequence from a directory of per-frame files
    ///
    /// # Arguments
    ///
    /// * `dir` - Directory containing frame files
    /// * `pattern` - File pattern (e.g., "frame_{:04}.json")
    /// * `num_frames` - Total number of frames
    /// * `fps` - Optional frames per second
    ///
    /// # Errors
    ///
    /// Returns error if directory is invalid
    pub fn from_directory(
        dir: &Path,
        pattern: &str,
        num_frames: usize,
        fps: Option<f32>,
    ) -> Result<Self, FlameError> {
        if !dir.is_dir() {
            return Err(FlameError::ModelDir(format!(
                "Not a directory: {}",
                dir.display()
            )));
        }

        // Load first frame to determine per-frame dimensions (bootstrap --
        // no `SequenceMetadata` exists yet to validate against).
        let first_frame_path = dir.join(pattern.replace("{}", "0").replace("{:04}", "0000"));
        let first_frame = parse_frame_file(&first_frame_path)?;

        let metadata = SequenceMetadata {
            num_frames,
            fps,
            n_shape: first_frame.shape.len(),
            n_expression: first_frame.expression.len(),
            n_pose: first_frame.pose.len(),
        };

        Ok(Self {
            source: SequenceSource::Directory {
                path: dir.to_path_buf(),
                file_pattern: pattern.to_string(),
                metadata,
            },
            cache: LruCache::new(DEFAULT_CACHE_SIZE),
            num_frames,
            fps,
        })
    }

    /// Set the cache size (number of frames to keep in memory)
    ///
    /// # Errors
    ///
    /// Returns error if size is zero
    pub fn set_cache_size(&mut self, size: usize) -> Result<(), FlameError> {
        let non_zero_size = NonZeroUsize::new(size)
            .ok_or_else(|| FlameError::InvalidParams("Cache size must be non-zero".to_string()))?;
        self.cache.resize(non_zero_size);
        Ok(())
    }

    /// Builder pattern: set the cache size and return self
    ///
    /// # Arguments
    ///
    /// * `size` - Number of frames to keep in cache (must be non-zero)
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use oxigaf_flame::sequence::FlameSequence;
    /// use std::path::Path;
    ///
    /// let mut seq = FlameSequence::from_json(Path::new("params.json"))?
    ///     .with_cache_size(512);
    /// # Ok::<(), oxigaf_flame::FlameError>(())
    /// ```
    #[must_use]
    pub fn with_cache_size(mut self, size: usize) -> Self {
        if let Ok(non_zero_size) = NonZeroUsize::new(size)
            .ok_or_else(|| FlameError::InvalidParams("Cache size must be non-zero".to_string()))
        {
            self.cache.resize(non_zero_size);
        }
        self
    }

    /// Prefetch a range of frames into the cache
    ///
    /// This is useful for sequential access patterns, such as video playback.
    /// Frames are loaded synchronously in order.
    ///
    /// # Arguments
    ///
    /// * `range` - Range of frame indices to prefetch
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use oxigaf_flame::sequence::FlameSequence;
    /// use std::path::Path;
    ///
    /// let mut seq = FlameSequence::from_json(Path::new("params.json"))?;
    /// // Prefetch frames 0-99
    /// seq.prefetch(0..100)?;
    /// # Ok::<(), oxigaf_flame::FlameError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns error if any frame in the range fails to load
    pub fn prefetch(&mut self, range: std::ops::Range<usize>) -> Result<(), FlameError> {
        for idx in range {
            if idx < self.num_frames() {
                // Load frame into cache
                self.get_frame(idx)?;
            }
        }
        Ok(())
    }

    /// Prefetch the next N frames for sequential access
    ///
    /// This is optimized for sequential playback patterns.
    ///
    /// # Arguments
    ///
    /// * `current_frame` - Current frame index
    /// * `count` - Number of frames ahead to prefetch
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use oxigaf_flame::sequence::FlameSequence;
    /// use std::path::Path;
    ///
    /// let mut seq = FlameSequence::from_json(Path::new("params.json"))?;
    /// // When at frame 10, prefetch frames 10-29
    /// seq.prefetch_ahead(10, 20)?;
    /// # Ok::<(), oxigaf_flame::FlameError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns error if any frame fails to load
    pub fn prefetch_ahead(&mut self, current_frame: usize, count: usize) -> Result<(), FlameError> {
        let end = (current_frame + count).min(self.num_frames());
        self.prefetch(current_frame..end)
    }

    /// Prefetch a range of frames in parallel using rayon
    ///
    /// This is significantly faster than sequential prefetching for large ranges
    /// when loading from disk. Requires the "parallel" feature.
    ///
    /// # Arguments
    ///
    /// * `range` - Range of frame indices to prefetch
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use oxigaf_flame::sequence::FlameSequence;
    /// use std::path::Path;
    ///
    /// let mut seq = FlameSequence::from_json(Path::new("params.json"))?;
    /// // Prefetch frames 0-99 in parallel
    /// seq.prefetch_parallel(0..100)?;
    /// # Ok::<(), oxigaf_flame::FlameError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns error if any frame in the range fails to load
    #[cfg(feature = "parallel")]
    pub fn prefetch_parallel(&mut self, range: std::ops::Range<usize>) -> Result<(), FlameError> {
        use rayon::prelude::*;

        // Collect indices that need loading (not already in cache)
        let indices_to_load: Vec<usize> = range
            .filter(|&idx| idx < self.num_frames() && self.cache.peek(&idx).is_none())
            .collect();

        // Load frames in parallel
        let frames: Result<Vec<(usize, FlameParams)>, FlameError> = indices_to_load
            .par_iter()
            .map(|&idx| {
                let params = match &self.source {
                    SequenceSource::Memory(frames) => frames[idx].clone(),
                    SequenceSource::JsonFile { frames, metadata } => {
                        frame_from_json_frames(frames, idx, metadata)?
                    }
                    #[cfg(feature = "npz")]
                    SequenceSource::NpzFile {
                        shape,
                        expression,
                        pose,
                        translation,
                        metadata,
                    } => lookup_frame_from_npz(
                        shape,
                        expression,
                        pose,
                        translation.as_ref().as_ref(),
                        idx,
                        metadata,
                    )?,
                    SequenceSource::Directory {
                        path,
                        file_pattern,
                        metadata,
                    } => {
                        let frame_path = format_frame_path(path, file_pattern, idx);
                        load_frame_from_file(&frame_path, metadata)?
                    }
                };
                Ok((idx, params))
            })
            .collect();

        // Insert loaded frames into cache
        for (idx, params) in frames? {
            self.cache.put(idx, params);
        }

        Ok(())
    }

    /// Get the total number of frames in the sequence
    #[must_use]
    pub fn num_frames(&self) -> usize {
        self.num_frames
    }

    /// Get the frames per second (if available)
    #[must_use]
    pub fn fps(&self) -> Option<f32> {
        self.fps
    }

    /// Get FLAME parameters for a specific frame
    ///
    /// # Arguments
    ///
    /// * `frame_idx` - Frame index (0-based)
    ///
    /// # Errors
    ///
    /// Returns error if frame index is out of bounds or loading fails
    pub fn get_frame(&mut self, frame_idx: usize) -> Result<&FlameParams, FlameError> {
        if frame_idx >= self.num_frames {
            return Err(FlameError::index_out_of_bounds(
                "FlameSequence::get_frame",
                frame_idx,
                self.num_frames,
            ));
        }

        // Check cache first
        if self.cache.peek(&frame_idx).is_some() {
            // Use get to update LRU order and return reference
            return self
                .cache
                .get(&frame_idx)
                .ok_or_else(|| FlameError::InvalidParams("Frame vanished from cache".to_string()))
                .map(Ok)?;
        }

        // Load from source
        let params = match &self.source {
            SequenceSource::Memory(frames) => frames[frame_idx].clone(),
            SequenceSource::JsonFile { frames, metadata } => {
                frame_from_json_frames(frames, frame_idx, metadata)?
            }
            #[cfg(feature = "npz")]
            SequenceSource::NpzFile {
                shape,
                expression,
                pose,
                translation,
                metadata,
            } => lookup_frame_from_npz(
                shape,
                expression,
                pose,
                translation.as_ref().as_ref(),
                frame_idx,
                metadata,
            )?,
            SequenceSource::Directory {
                path,
                file_pattern,
                metadata,
            } => {
                let frame_path = format_frame_path(path, file_pattern, frame_idx);
                load_frame_from_file(&frame_path, metadata)?
            }
        };

        // Insert into cache and return reference
        self.cache.put(frame_idx, params);
        self.cache
            .get(&frame_idx)
            .ok_or_else(|| FlameError::InvalidParams("Failed to cache frame".to_string()))
    }

    /// Interpolate between frames using linear interpolation
    ///
    /// # Arguments
    ///
    /// * `frame_f` - Fractional frame index (e.g., 5.5 for halfway between frames 5 and 6)
    ///
    /// # Errors
    ///
    /// Returns error if frame index is out of bounds or loading fails
    pub fn interpolate(&mut self, frame_f: f32) -> Result<FlameParams, FlameError> {
        if frame_f < 0.0 || frame_f >= self.num_frames as f32 {
            return Err(FlameError::InvalidParams(format!(
                "Frame index {} out of bounds [0, {})",
                frame_f, self.num_frames
            )));
        }

        let frame_0 = frame_f.floor() as usize;
        let frame_1 = (frame_0 + 1).min(self.num_frames - 1);
        let t = frame_f - frame_0 as f32;

        if t < 1e-6 {
            // No interpolation needed
            return Ok(self.get_frame(frame_0)?.clone());
        }

        let params_0 = self.get_frame(frame_0)?.clone();
        let params_1 = self.get_frame(frame_1)?.clone();

        // Delegate to `FlameParams::lerp`: shape/expression/translation are
        // interpolated linearly and pose via quaternion slerp (axis-angle ->
        // quaternion -> slerp -> axis-angle), which keeps rotational paths
        // on the rotation manifold -- a naive component-wise lerp of
        // axis-angle vectors does not (e.g. it can silently cancel a
        // near-pi rotation to zero rotation at t=0.5). This also validates
        // that `params_0` and `params_1` have matching shape/expression/
        // pose lengths instead of silently truncating to the shorter one.
        params_0.lerp(&params_1, t)
    }

    /// Get an iterator over all frames (with caching)
    ///
    /// Note: This loads frames on-demand and caches them
    pub fn iter(&mut self) -> SequenceIterator<'_> {
        SequenceIterator {
            sequence: self,
            current: 0,
        }
    }
}

/// Iterator over FLAME sequence frames
pub struct SequenceIterator<'a> {
    sequence: &'a mut FlameSequence,
    current: usize,
}

impl Iterator for SequenceIterator<'_> {
    type Item = Result<FlameParams, FlameError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.sequence.num_frames() {
            return None;
        }

        let result = self.sequence.get_frame(self.current).cloned();
        self.current += 1;
        Some(result)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.sequence.num_frames() - self.current;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for SequenceIterator<'_> {}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Convert `FrameJson` to `FlameParams`
fn frame_json_to_params(
    frame: FrameJson,
    metadata: &SequenceMetadata,
) -> Result<FlameParams, FlameError> {
    if frame.shape.len() != metadata.n_shape {
        return Err(FlameError::InvalidParams(format!(
            "Shape parameter count mismatch: expected {}, got {}",
            metadata.n_shape,
            frame.shape.len()
        )));
    }
    if frame.expression.len() != metadata.n_expression {
        return Err(FlameError::InvalidParams(format!(
            "Expression parameter count mismatch: expected {}, got {}",
            metadata.n_expression,
            frame.expression.len()
        )));
    }
    if frame.pose.len() != metadata.n_pose {
        return Err(FlameError::InvalidParams(format!(
            "Pose parameter count mismatch: expected {}, got {}",
            metadata.n_pose,
            frame.pose.len()
        )));
    }

    Ok(FlameParams {
        shape: frame.shape,
        expression: frame.expression,
        pose: frame.pose,
        translation: frame.translation.unwrap_or([0.0, 0.0, 0.0]),
    })
}

/// Look up and validate a single frame from an already-parsed, in-memory
/// JSON frame list (see [`SequenceSource::JsonFile`]).
///
/// Unlike the old per-frame loader this performs no file I/O and no JSON
/// parsing -- both happen exactly once, in `from_json`.
fn frame_from_json_frames(
    frames: &[FrameJson],
    frame_idx: usize,
    metadata: &SequenceMetadata,
) -> Result<FlameParams, FlameError> {
    let frame = frames.get(frame_idx).ok_or_else(|| {
        FlameError::index_out_of_bounds("FlameSequence::get_frame", frame_idx, frames.len())
    })?;
    frame_json_to_params(frame.clone(), metadata)
}

/// The raw arrays of an NPZ sequence file, read exactly once.
///
/// [`FlameSequence::from_npz`] used to do all of reading, validating and
/// converting inline, which is what earned it a `#[allow(clippy::too_many_lines)]`.
/// The three phases are separated here instead: [`NpzArrays::read`] does the
/// I/O and decompression, [`NpzArrays::validated_metadata`] checks the array
/// geometry agrees frame-for-frame, and [`NpzArrays::into_frames`] materialises
/// the eager (small-sequence) representation. The lazy representation keeps
/// these same arrays alive inside [`SequenceSource::NpzFile`].
#[cfg(feature = "npz")]
struct NpzArrays {
    shape: ndarray::Array2<f32>,
    expression: ndarray::Array2<f32>,
    pose: ndarray::Array2<f32>,
    translation: Option<ndarray::Array2<f32>>,
    fps: Option<f32>,
}

#[cfg(feature = "npz")]
impl NpzArrays {
    /// Open `path` and decompress every array the sequence format defines.
    ///
    /// This is the only place the file is touched: both the eager and the
    /// lazy sequence representations are built from the result.
    fn read(path: &Path) -> Result<Self, FlameError> {
        use ndarray_npy::NpzReader;
        use std::fs::File;

        let file = File::open(path).map_err(|e| FlameError::IoError {
            source: e,
            path: path.to_path_buf(),
        })?;

        let mut npz = NpzReader::new(file)
            .map_err(|e| FlameError::InvalidParams(format!("Failed to open NPZ file: {e}")))?;

        // Load shape array
        let shape: ndarray::Array2<f32> =
            npz.by_name("shape").map_err(|e| FlameError::NpzLoad {
                name: "shape".to_string(),
                source: e,
            })?;

        // Load expression array (try both "expression" and "expr" keys)
        let expression: ndarray::Array2<f32> = npz
            .by_name("expression")
            .or_else(|_| npz.by_name("expr"))
            .map_err(|e| FlameError::NpzLoad {
                name: "expression/expr".to_string(),
                source: e,
            })?;

        // Load pose array
        let pose: ndarray::Array2<f32> = npz.by_name("pose").map_err(|e| FlameError::NpzLoad {
            name: "pose".to_string(),
            source: e,
        })?;

        // Optional translation array
        let translation: Option<ndarray::Array2<f32>> = npz.by_name("translation").ok();

        // Optional scalar `fps` metadata. Numpy scalars round-trip as
        // either a 1-element 1-D array or a genuine 0-D array depending on
        // how they were saved, so both forms are tried.
        let fps: Option<f32> = {
            let one_d: Result<ndarray::Array1<f32>, _> = npz.by_name("fps");
            if let Ok(arr) = one_d {
                arr.first().copied()
            } else {
                let zero_d: Result<ndarray::Array0<f32>, _> = npz.by_name("fps");
                zero_d.ok().map(ndarray::Array0::into_scalar)
            }
        };

        Ok(Self {
            shape,
            expression,
            pose,
            translation,
            fps,
        })
    }

    /// Check that every array describes the same number of frames, then
    /// derive the sequence metadata from their column counts.
    ///
    /// `shape` defines the frame count; `expression` and `pose` must match it
    /// row-for-row, and `translation` (when present) must additionally be
    /// exactly three columns wide.
    fn validated_metadata(&self) -> Result<SequenceMetadata, FlameError> {
        let num_frames = self.shape.nrows();
        if self.expression.nrows() != num_frames {
            return Err(FlameError::ShapeMismatch {
                name: "expression".to_string(),
                expected: format!("{num_frames} frames"),
                got: format!("{} frames", self.expression.nrows()),
            });
        }
        if self.pose.nrows() != num_frames {
            return Err(FlameError::ShapeMismatch {
                name: "pose".to_string(),
                expected: format!("{num_frames} frames"),
                got: format!("{} frames", self.pose.nrows()),
            });
        }
        if let Some(ref trans) = self.translation {
            if trans.nrows() != num_frames {
                return Err(FlameError::ShapeMismatch {
                    name: "translation".to_string(),
                    expected: format!("{num_frames} frames"),
                    got: format!("{} frames", trans.nrows()),
                });
            }
            if trans.ncols() != 3 {
                return Err(FlameError::ShapeMismatch {
                    name: "translation".to_string(),
                    expected: "3 columns".to_string(),
                    got: format!("{} columns", trans.ncols()),
                });
            }
        }

        Ok(SequenceMetadata {
            num_frames,
            fps: self.fps,
            n_shape: self.shape.ncols(),
            n_expression: self.expression.ncols(),
            n_pose: self.pose.ncols(),
        })
    }

    /// Materialise every frame eagerly, for sequences small enough that
    /// holding the whole `Vec<FlameParams>` is cheaper than keeping the
    /// arrays and rebuilding a frame per access.
    ///
    /// Only call after [`Self::validated_metadata`] has succeeded: the row
    /// indexing here relies on the frame counts already agreeing.
    fn into_frames(self) -> Vec<FlameParams> {
        let num_frames = self.shape.nrows();
        let mut frames = Vec::with_capacity(num_frames);
        for i in 0..num_frames {
            let translation = self.translation.as_ref().map_or([0.0, 0.0, 0.0], |trans| {
                [trans[[i, 0]], trans[[i, 1]], trans[[i, 2]]]
            });
            frames.push(FlameParams {
                shape: self.shape.row(i).to_vec(),
                expression: self.expression.row(i).to_vec(),
                pose: self.pose.row(i).to_vec(),
                translation,
            });
        }
        frames
    }
}

/// Look up and validate a single frame from already-decompressed NPZ
/// arrays (see [`SequenceSource::NpzFile`]).
///
/// Unlike the old per-frame loader this performs no file I/O and no NPZ
/// decompression -- both happen exactly once, in `from_npz`.
#[cfg(feature = "npz")]
fn lookup_frame_from_npz(
    shape_arr: &ndarray::Array2<f32>,
    expression_arr: &ndarray::Array2<f32>,
    pose_arr: &ndarray::Array2<f32>,
    translation_arr: Option<&ndarray::Array2<f32>>,
    frame_idx: usize,
    metadata: &SequenceMetadata,
) -> Result<FlameParams, FlameError> {
    // Validate frame index
    if frame_idx >= shape_arr.nrows() {
        return Err(FlameError::index_out_of_bounds(
            "FlameSequence::get_frame",
            frame_idx,
            shape_arr.nrows(),
        ));
    }

    // Extract frame data
    let shape = shape_arr.row(frame_idx).to_vec();
    let expression = expression_arr.row(frame_idx).to_vec();
    let pose = pose_arr.row(frame_idx).to_vec();
    let translation = if let Some(trans) = translation_arr {
        [
            trans[[frame_idx, 0]],
            trans[[frame_idx, 1]],
            trans[[frame_idx, 2]],
        ]
    } else {
        [0.0, 0.0, 0.0]
    };

    // Validate dimensions
    if shape.len() != metadata.n_shape {
        return Err(FlameError::InvalidParams(format!(
            "Shape parameter count mismatch: expected {}, got {}",
            metadata.n_shape,
            shape.len()
        )));
    }
    if expression.len() != metadata.n_expression {
        return Err(FlameError::InvalidParams(format!(
            "Expression parameter count mismatch: expected {}, got {}",
            metadata.n_expression,
            expression.len()
        )));
    }
    if pose.len() != metadata.n_pose {
        return Err(FlameError::InvalidParams(format!(
            "Pose parameter count mismatch: expected {}, got {}",
            metadata.n_pose,
            pose.len()
        )));
    }

    Ok(FlameParams {
        shape,
        expression,
        pose,
        translation,
    })
}

/// Parse a single per-frame JSON file into a `FrameJson`, without dimension
/// validation.
///
/// Used to bootstrap [`SequenceMetadata`] from the first frame in
/// [`FlameSequence::from_directory`], before a `SequenceMetadata` exists to
/// validate against. Per-frame access afterward goes through
/// [`load_frame_from_file`], which validates dimensions.
fn parse_frame_file(path: &Path) -> Result<FrameJson, FlameError> {
    let json_str = std::fs::read_to_string(path).map_err(|e| FlameError::IoError {
        source: e,
        path: path.to_path_buf(),
    })?;

    serde_json::from_str(&json_str)
        .map_err(|e| FlameError::InvalidParams(format!("Failed to parse frame JSON: {e}")))
}

/// Load and validate a single frame from an individual per-frame file,
/// checking its dimensions against the sequence's [`SequenceMetadata`] --
/// the same validation the JSON and NPZ sequence formats already apply.
/// Without this, a directory of per-frame files with inconsistent
/// coefficient counts would load without error and surface later as
/// silently wrong-length parameter vectors.
fn load_frame_from_file(
    path: &Path,
    metadata: &SequenceMetadata,
) -> Result<FlameParams, FlameError> {
    let frame = parse_frame_file(path)?;
    frame_json_to_params(frame, metadata)
}

/// Format frame path from pattern
fn format_frame_path(dir: &Path, pattern: &str, frame_idx: usize) -> PathBuf {
    let filename = if pattern.contains("{:04}") {
        pattern.replace("{:04}", &format!("{frame_idx:04}"))
    } else if pattern.contains("{}") {
        pattern.replace("{}", &frame_idx.to_string())
    } else {
        pattern.to_string()
    };
    dir.join(filename)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn create_test_params(idx: usize) -> FlameParams {
        FlameParams {
            shape: vec![idx as f32; 10],
            expression: vec![idx as f32 * 2.0; 5],
            pose: vec![idx as f32 * 3.0; 6],
            translation: [idx as f32, 0.0, 0.0],
        }
    }

    #[test]
    fn test_from_memory() {
        let frames = vec![create_test_params(0), create_test_params(1)];
        let mut seq = FlameSequence::from_memory(frames, Some(30.0));

        assert_eq!(seq.num_frames(), 2);
        assert_eq!(seq.fps(), Some(30.0));

        let frame_0 = seq.get_frame(0).expect("test: frame should be available");
        assert!((frame_0.shape[0] - 0.0).abs() < 1e-5);

        let frame_1 = seq.get_frame(1).expect("test: frame should be available");
        assert!((frame_1.shape[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_interpolation() {
        let frames = vec![create_test_params(0), create_test_params(10)];
        let mut seq = FlameSequence::from_memory(frames, Some(30.0));

        // Interpolate at t=0.5 (midpoint)
        let interp = seq
            .interpolate(0.5)
            .expect("test: interpolation should succeed");
        assert!((interp.shape[0] - 5.0).abs() < 1e-5);
        assert!((interp.expression[0] - 10.0).abs() < 1e-5);

        // Interpolate at t=0.0 (should match frame 0)
        let interp = seq
            .interpolate(0.0)
            .expect("test: interpolation should succeed");
        assert!((interp.shape[0] - 0.0).abs() < 1e-5);

        // Interpolate at t=1.0 (should match frame 1)
        let interp = seq
            .interpolate(1.0)
            .expect("test: interpolation should succeed");
        assert!((interp.shape[0] - 10.0).abs() < 1e-5);
    }

    #[test]
    fn test_json_roundtrip() {
        let temp_dir = TempDir::new().expect("test: temp dir creation should succeed");
        let json_path = temp_dir.path().join("sequence.json");

        // Create test sequence
        let sequence_json = SequenceJson {
            fps: Some(30.0),
            frames: vec![
                FrameJson {
                    shape: vec![0.0; 10],
                    expression: vec![0.0; 5],
                    pose: vec![0.0; 6],
                    translation: Some([0.0, 0.0, 0.0]),
                },
                FrameJson {
                    shape: vec![1.0; 10],
                    expression: vec![2.0; 5],
                    pose: vec![3.0; 6],
                    translation: Some([1.0, 0.0, 0.0]),
                },
            ],
        };

        let json_str = serde_json::to_string_pretty(&sequence_json)
            .expect("test: JSON serialization should succeed");
        fs::write(&json_path, json_str).expect("test: file operation should succeed");

        // Load and verify
        let mut seq =
            FlameSequence::from_json(&json_path).expect("test: sequence loading should succeed");
        assert_eq!(seq.num_frames(), 2);
        assert_eq!(seq.fps(), Some(30.0));

        let frame_0 = seq.get_frame(0).expect("test: frame should be available");
        assert!((frame_0.shape[0] - 0.0).abs() < 1e-5);

        let frame_1 = seq.get_frame(1).expect("test: frame should be available");
        assert!((frame_1.shape[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_cache() {
        let frames = vec![
            create_test_params(0),
            create_test_params(1),
            create_test_params(2),
        ];
        let mut seq = FlameSequence::from_memory(frames, None);
        seq.set_cache_size(2)
            .expect("test: cache size setting should succeed");

        // Access frames
        let _f0 = seq.get_frame(0).expect("test: frame should be available");
        let _f1 = seq.get_frame(1).expect("test: frame should be available");
        let _f2 = seq.get_frame(2).expect("test: frame should be available");

        // Frame 0 should be evicted from cache (LRU)
        // But should still be accessible from memory source
        let f0_again = seq.get_frame(0).expect("test: frame should be available");
        assert!((f0_again.shape[0] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_iterator() {
        let frames = vec![create_test_params(0), create_test_params(1)];
        let mut seq = FlameSequence::from_memory(frames, None);

        let collected: Vec<_> = seq
            .iter()
            .map(|r| r.expect("test: frame should be available"))
            .collect();
        assert_eq!(collected.len(), 2);
        assert!((collected[0].shape[0] - 0.0).abs() < 1e-5);
        assert!((collected[1].shape[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_out_of_bounds() {
        let frames = vec![create_test_params(0)];
        let mut seq = FlameSequence::from_memory(frames, None);

        assert!(seq.get_frame(1).is_err());
        assert!(seq.interpolate(-0.5).is_err());
        assert!(seq.interpolate(2.0).is_err());
    }

    #[test]
    fn test_with_cache_size() {
        let frames = vec![
            create_test_params(0),
            create_test_params(1),
            create_test_params(2),
        ];
        let mut seq = FlameSequence::from_memory(frames, None).with_cache_size(512);

        // Cache should be able to hold all 3 frames
        let _f0 = seq.get_frame(0).expect("test: frame should be available");
        let _f1 = seq.get_frame(1).expect("test: frame should be available");
        let _f2 = seq.get_frame(2).expect("test: frame should be available");

        // All should still be in cache
        let f0_again = seq.get_frame(0).expect("test: frame should be available");
        assert!((f0_again.shape[0] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_prefetch() {
        let frames: Vec<_> = (0..10).map(create_test_params).collect();
        let mut seq = FlameSequence::from_memory(frames, None);

        // Prefetch frames 0-4
        seq.prefetch(0..5).expect("test: prefetch should succeed");

        // All prefetched frames should be accessible
        for i in 0..5 {
            let frame = seq.get_frame(i).expect("test: frame should be available");
            assert!((frame.shape[0] - i as f32).abs() < 1e-5);
        }
    }

    #[test]
    fn test_prefetch_ahead() {
        let frames: Vec<_> = (0..20).map(create_test_params).collect();
        let mut seq = FlameSequence::from_memory(frames, None);

        // Prefetch 10 frames ahead from frame 5
        seq.prefetch_ahead(5, 10)
            .expect("test: prefetch should succeed");

        // Frames 5-14 should be accessible
        for i in 5..15 {
            let frame = seq.get_frame(i).expect("test: frame should be available");
            assert!((frame.shape[0] - i as f32).abs() < 1e-5);
        }
    }

    #[test]
    fn test_prefetch_ahead_near_end() {
        let frames: Vec<_> = (0..10).map(create_test_params).collect();
        let mut seq = FlameSequence::from_memory(frames, None);

        // Prefetch beyond sequence end (should stop at last frame)
        seq.prefetch_ahead(8, 10)
            .expect("test: prefetch should succeed");

        // Should be able to access frames up to the end
        let frame = seq.get_frame(9).expect("test: frame should be available");
        assert!((frame.shape[0] - 9.0).abs() < 1e-5);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_prefetch_parallel() {
        let frames: Vec<_> = (0..50).map(create_test_params).collect();
        let mut seq = FlameSequence::from_memory(frames, None);

        // Prefetch frames 0-29 in parallel
        seq.prefetch_parallel(0..30)
            .expect("test: prefetch should succeed");

        // All prefetched frames should be accessible
        for i in 0..30 {
            let frame = seq.get_frame(i).expect("test: frame should be available");
            assert!((frame.shape[0] - i as f32).abs() < 1e-5);
        }
    }

    #[test]
    fn bench_sequential_access() {
        // Create a larger sequence for more realistic benchmarking
        let frames: Vec<_> = (0..1000).map(create_test_params).collect();
        let mut seq = FlameSequence::from_memory(frames, Some(30.0));

        let start = std::time::Instant::now();
        for i in 0..1000 {
            let _ = seq
                .get_frame(i % seq.num_frames())
                .expect("test: frame should be available");
        }
        let elapsed = start.elapsed();

        println!("1000 sequential accesses: {elapsed:?}");
        println!("Average per frame: {:?}", elapsed / 1000);

        // With caching, this should be very fast (< 1ms total)
        assert!(elapsed.as_millis() < 100);
    }

    #[test]
    fn bench_prefetch_performance() {
        let frames: Vec<_> = (0..500).map(create_test_params).collect();
        let mut seq = FlameSequence::from_memory(frames, None);

        // Benchmark prefetching
        let start = std::time::Instant::now();
        seq.prefetch(0..100).expect("test: prefetch should succeed");
        let prefetch_time = start.elapsed();

        println!("Prefetch 100 frames: {prefetch_time:?}");

        // Benchmark sequential access after prefetching
        let start = std::time::Instant::now();
        for i in 0..100 {
            let _ = seq.get_frame(i).expect("test: frame should be available");
        }
        let access_time = start.elapsed();

        println!("Access 100 cached frames: {access_time:?}");
        println!(
            "Speedup: {:.2}x",
            prefetch_time.as_nanos() as f64 / access_time.as_nanos() as f64
        );

        // Cached access should be much faster
        assert!(access_time < prefetch_time);
    }

    #[cfg(feature = "npz")]
    #[test]
    fn test_npz_loading() {
        use ndarray::Array2;
        use ndarray_npy::NpzWriter;
        use std::fs::File;

        let temp_dir = TempDir::new().expect("test: temp dir creation should succeed");
        let npz_path = temp_dir.path().join("test_sequence.npz");

        // Create test NPZ file
        let num_frames = 10;
        let n_shape = 5;
        let n_expr = 3;
        let n_pose = 6;

        // Use explicit f32 arrays
        let shape_data: Array2<f32> =
            Array2::from_shape_fn((num_frames, n_shape), |(i, j)| (i * n_shape + j) as f32);
        let expr_data: Array2<f32> =
            Array2::from_shape_fn((num_frames, n_expr), |(i, j)| (i * n_expr + j) as f32 * 2.0);
        let pose_data: Array2<f32> =
            Array2::from_shape_fn((num_frames, n_pose), |(i, j)| (i * n_pose + j) as f32 * 3.0);
        let translation_data: Array2<f32> =
            Array2::from_shape_fn((num_frames, 3), |(i, j)| i as f32 + j as f32 * 0.1);

        let file = File::create(&npz_path).expect("test: file creation should succeed");
        let mut npz = NpzWriter::new(file);
        npz.add_array("shape", &shape_data)
            .expect("test: array write should succeed");
        npz.add_array("expression", &expr_data)
            .expect("test: array write should succeed");
        npz.add_array("pose", &pose_data)
            .expect("test: array write should succeed");
        npz.add_array("translation", &translation_data)
            .expect("test: array write should succeed");
        npz.finish().expect("test: npz write should succeed");

        // Load sequence
        let mut seq = FlameSequence::from_npz(&npz_path).expect("test: npz load should succeed");

        // Verify metadata
        assert_eq!(seq.num_frames(), num_frames);

        // Verify frame data
        let frame_0 = seq.get_frame(0).expect("test: frame should be available");
        assert_eq!(frame_0.shape.len(), n_shape);
        assert_eq!(frame_0.expression.len(), n_expr);
        assert_eq!(frame_0.pose.len(), n_pose);
        assert!((frame_0.shape[0] - 0.0).abs() < 1e-5);
        assert!((frame_0.translation[0] - 0.0).abs() < 1e-5);

        let frame_5 = seq.get_frame(5).expect("test: frame should be available");
        assert!((frame_5.shape[0] - 25.0).abs() < 1e-5); // 5 * 5 + 0
                                                         // Expected: (5 * 3 + 1) * 2 = 16 * 2 = 32, not 22
                                                         // Correct calculation: row 5, col 1 = (5, 1) -> 5*3+1=16, *2=32
        let expected_expr_1 = ((5 * n_expr + 1) as f32) * 2.0;
        assert!((frame_5.expression[1] - expected_expr_1).abs() < 1e-5);
        assert!((frame_5.translation[1] - 5.1).abs() < 1e-5); // 5 + 1 * 0.1
    }

    #[cfg(feature = "npz")]
    #[test]
    fn test_npz_with_expr_key() {
        use ndarray::Array2;
        use ndarray_npy::NpzWriter;
        use std::fs::File;

        let temp_dir = TempDir::new().expect("test: temp dir creation should succeed");
        let npz_path = temp_dir.path().join("test_sequence_expr.npz");

        // Create NPZ with "expr" instead of "expression"
        let num_frames = 5;
        let shape_data: Array2<f32> =
            Array2::from_shape_fn((num_frames, 10), |(i, j)| (i + j) as f32);
        let expr_data: Array2<f32> =
            Array2::from_shape_fn((num_frames, 5), |(i, j)| (i * 10 + j) as f32);
        let pose_data: Array2<f32> = Array2::from_shape_fn((num_frames, 6), |(_, _)| 0.0);

        let file = File::create(&npz_path).expect("test: file creation should succeed");
        let mut npz = NpzWriter::new(file);
        npz.add_array("shape", &shape_data)
            .expect("test: array write should succeed");
        npz.add_array("expr", &expr_data)
            .expect("test: array write should succeed"); // Use "expr" key
        npz.add_array("pose", &pose_data)
            .expect("test: array write should succeed");
        npz.finish().expect("test: npz write should succeed");

        // Should load successfully with "expr" key
        let mut seq = FlameSequence::from_npz(&npz_path).expect("test: npz load should succeed");
        assert_eq!(seq.num_frames(), num_frames);

        let frame = seq.get_frame(0).expect("test: frame should be available");
        assert_eq!(frame.expression.len(), 5);
    }

    #[cfg(feature = "npz")]
    #[test]
    fn test_npz_lazy_loading() {
        use ndarray::Array2;
        use ndarray_npy::NpzWriter;
        use std::fs::File;

        let temp_dir = TempDir::new().expect("test: temp dir creation should succeed");
        let npz_path = temp_dir.path().join("test_large_sequence.npz");

        // Create large NPZ file (>1000 frames for lazy loading)
        let num_frames = 1500;
        let shape_data: Array2<f32> =
            Array2::from_shape_fn((num_frames, 10), |(i, j)| (i + j) as f32);
        let expr_data: Array2<f32> = Array2::from_shape_fn((num_frames, 5), |(i, _)| i as f32);
        let pose_data: Array2<f32> = Array2::from_shape_fn((num_frames, 6), |(_, _)| 0.0);

        let file = File::create(&npz_path).expect("test: file creation should succeed");
        let mut npz = NpzWriter::new(file);
        npz.add_array("shape", &shape_data)
            .expect("test: array write should succeed");
        npz.add_array("expression", &expr_data)
            .expect("test: array write should succeed");
        npz.add_array("pose", &pose_data)
            .expect("test: array write should succeed");
        npz.finish().expect("test: npz write should succeed");

        // Load with lazy loading
        let mut seq = FlameSequence::from_npz(&npz_path).expect("test: npz load should succeed");
        assert_eq!(seq.num_frames(), num_frames);

        // Access frames (should load on demand)
        let frame_0 = seq.get_frame(0).expect("test: frame should be available");
        assert!((frame_0.expression[0] - 0.0).abs() < 1e-5);

        let frame_500 = seq.get_frame(500).expect("test: frame should be available");
        assert!((frame_500.expression[0] - 500.0).abs() < 1e-5);
    }

    /// Regression: `from_npz` was one 143-line function carrying
    /// `#[allow(clippy::too_many_lines)]`; its read / validate / convert
    /// phases are now `NpzArrays::read`, `validated_metadata` and
    /// `into_frames`. The geometry checks in the middle phase had no direct
    /// coverage, so a mis-split could have dropped them silently.
    #[cfg(feature = "npz")]
    #[test]
    fn test_npz_rejects_mismatched_array_geometry() {
        use ndarray::Array2;
        use ndarray_npy::NpzWriter;
        use std::fs::File;

        fn write_npz(
            path: &std::path::Path,
            expr_rows: usize,
            translation: Option<Array2<f32>>,
        ) -> Result<(), FlameError> {
            let shape_data: Array2<f32> = Array2::zeros((4, 10));
            let expr_data: Array2<f32> = Array2::zeros((expr_rows, 5));
            let pose_data: Array2<f32> = Array2::zeros((4, 6));

            let file = File::create(path).expect("test: file creation should succeed");
            let mut npz = NpzWriter::new(file);
            npz.add_array("shape", &shape_data)
                .expect("test: array write should succeed");
            npz.add_array("expression", &expr_data)
                .expect("test: array write should succeed");
            npz.add_array("pose", &pose_data)
                .expect("test: array write should succeed");
            if let Some(trans) = translation {
                npz.add_array("translation", &trans)
                    .expect("test: array write should succeed");
            }
            npz.finish().expect("test: npz write should succeed");
            FlameSequence::from_npz(path).map(|_| ())
        }

        let temp_dir = TempDir::new().expect("test: temp dir creation should succeed");

        // expression has fewer frames than shape.
        let err = write_npz(&temp_dir.path().join("rows.npz"), 3, None)
            .expect_err("test: mismatched frame counts must be rejected");
        assert!(
            matches!(&err, FlameError::ShapeMismatch { name, .. } if name == "expression"),
            "expected a ShapeMismatch on `expression`, got {err:?}"
        );

        // translation is present but is not 3 columns wide.
        let err = write_npz(
            &temp_dir.path().join("cols.npz"),
            4,
            Some(Array2::zeros((4, 2))),
        )
        .expect_err("test: a non-3-column translation must be rejected");
        assert!(
            matches!(&err, FlameError::ShapeMismatch { name, .. } if name == "translation"),
            "expected a ShapeMismatch on `translation`, got {err:?}"
        );

        // The same arrays with consistent geometry load fine.
        write_npz(
            &temp_dir.path().join("ok.npz"),
            4,
            Some(Array2::zeros((4, 3))),
        )
        .expect("test: consistent geometry should load");
    }

    /// Regression: `SequenceSource::NpzFile` is now `#[cfg(feature = "npz")]`
    /// rather than always-present-and-`#[allow(dead_code)]`. That variant is
    /// only reachable above the 1000-frame lazy-loading threshold, so this
    /// pins that the eager path (below the threshold) and the lazy path
    /// (above it) still return the same parameters for the same data.
    #[cfg(feature = "npz")]
    #[test]
    fn test_npz_eager_and_lazy_paths_agree() {
        use ndarray::Array2;
        use ndarray_npy::NpzWriter;
        use std::fs::File;

        fn write_and_load(path: &std::path::Path, num_frames: usize) -> FlameSequence {
            let shape_data: Array2<f32> =
                Array2::from_shape_fn((num_frames, 4), |(i, j)| (i * 4 + j) as f32);
            let expr_data: Array2<f32> =
                Array2::from_shape_fn((num_frames, 3), |(i, j)| (i + j) as f32 * 0.5);
            let pose_data: Array2<f32> = Array2::from_shape_fn((num_frames, 6), |(i, _)| i as f32);
            let trans_data: Array2<f32> =
                Array2::from_shape_fn((num_frames, 3), |(i, j)| (i as f32) - (j as f32));

            let file = File::create(path).expect("test: file creation should succeed");
            let mut npz = NpzWriter::new(file);
            for (name, arr) in [
                ("shape", &shape_data),
                ("expression", &expr_data),
                ("pose", &pose_data),
                ("translation", &trans_data),
            ] {
                npz.add_array(name, arr)
                    .expect("test: array write should succeed");
            }
            npz.finish().expect("test: npz write should succeed");
            FlameSequence::from_npz(path).expect("test: npz load should succeed")
        }

        let temp_dir = TempDir::new().expect("test: temp dir creation should succeed");
        // 1000 frames or fewer => eager `Memory`; more => lazy `NpzFile`.
        let mut eager = write_and_load(&temp_dir.path().join("eager.npz"), 1000);
        let mut lazy = write_and_load(&temp_dir.path().join("lazy.npz"), 1001);

        for idx in [0usize, 1, 499, 999] {
            let a = eager
                .get_frame(idx)
                .expect("test: eager frame should be available")
                .clone();
            let b = lazy
                .get_frame(idx)
                .expect("test: lazy frame should be available");
            assert_eq!(a.shape, b.shape, "frame {idx}: shape differs");
            assert_eq!(
                a.expression, b.expression,
                "frame {idx}: expression differs"
            );
            assert_eq!(a.pose, b.pose, "frame {idx}: pose differs");
            assert_eq!(
                a.translation, b.translation,
                "frame {idx}: translation differs"
            );
        }
    }

    #[cfg(not(feature = "npz"))]
    #[test]
    fn test_npz_feature_disabled() {
        use std::path::Path;

        let result = FlameSequence::from_npz(Path::new("dummy.npz"));
        assert!(result.is_err());

        if let Err(FlameError::InvalidParams(msg)) = result {
            assert!(msg.contains("not enabled"));
        } else {
            panic!("Expected InvalidParams error");
        }
    }

    #[test]
    fn test_lazy_json_no_reread_after_load() {
        // Regression test for lazy JSON loading re-reading and re-parsing
        // the whole file on every single frame access. If that regresses,
        // this test fails because the source file is deleted right after
        // loading, before any frame is accessed.
        let temp_dir = TempDir::new().expect("test: temp dir creation should succeed");
        let json_path = temp_dir.path().join("large_sequence.json");

        let frames: Vec<FrameJson> = (0..1200)
            .map(|i| FrameJson {
                shape: vec![i as f32; 10],
                expression: vec![i as f32 * 2.0; 5],
                pose: vec![i as f32 * 3.0; 6],
                translation: Some([i as f32, 0.0, 0.0]),
            })
            .collect();
        let sequence_json = SequenceJson {
            fps: Some(30.0),
            frames,
        };
        let json_str =
            serde_json::to_string(&sequence_json).expect("test: JSON serialization should succeed");
        fs::write(&json_path, json_str).expect("test: file operation should succeed");

        let mut seq =
            FlameSequence::from_json(&json_path).expect("test: sequence loading should succeed");
        assert_eq!(
            seq.num_frames(),
            1200,
            "should pick lazy mode (>1000 frames)"
        );

        // Remove the source file: lazy mode must have parsed it exactly
        // once, during `from_json` above, and must not touch it again.
        fs::remove_file(&json_path).expect("test: file removal should succeed");

        seq.prefetch(0..1200)
            .expect("test: prefetch should succeed after source file removal");
        let frame_999 = seq
            .get_frame(999)
            .expect("test: frame access should succeed after source file removal");
        assert!((frame_999.shape[0] - 999.0).abs() < 1e-5);
    }

    #[test]
    fn test_interpolate_pose_slerp_near_pi_wrap() {
        // Two frames whose single-joint pose sits on opposite sides of the
        // +/-pi branch cut for axis-angle rotation about Z. A naive
        // component-wise lerp of (0,0,3.10) and (0,0,-3.10) at t=0.5 gives
        // (0,0,0) -- i.e. NO rotation -- which is wrong: the two rotations
        // are actually close together (~5 degrees apart) on the rotation
        // manifold, on the far side from identity, so the correctly
        // interpolated rotation angle is close to pi, not 0.
        let frame_0 = FlameParams {
            shape: vec![],
            expression: vec![],
            pose: vec![0.0, 0.0, 3.10],
            translation: [0.0, 0.0, 0.0],
        };
        let frame_1 = FlameParams {
            shape: vec![],
            expression: vec![],
            pose: vec![0.0, 0.0, -3.10],
            translation: [0.0, 0.0, 0.0],
        };
        let mut seq = FlameSequence::from_memory(vec![frame_0, frame_1], None);

        let interp = seq
            .interpolate(0.5)
            .expect("test: interpolation should succeed");
        let angle = (interp.pose[0] * interp.pose[0]
            + interp.pose[1] * interp.pose[1]
            + interp.pose[2] * interp.pose[2])
            .sqrt();
        assert!(
            angle > 3.0,
            "expected interpolated rotation angle near pi, got {angle} (pose={:?})",
            interp.pose
        );
    }

    #[test]
    fn test_interpolate_mismatched_lengths_errors() {
        // `from_memory` performs no cross-frame dimension validation, so
        // two frames with different shape-vector lengths can end up
        // adjacent in a sequence. `interpolate` must reject this rather
        // than silently truncating to the shorter vector (the old
        // `lerp_vec`-based implementation did exactly that, via `zip`).
        let frame_0 = create_test_params(0); // shape len 10
        let mut frame_1 = create_test_params(1);
        frame_1.shape = vec![1.0; 5]; // mismatched length
        let mut seq = FlameSequence::from_memory(vec![frame_0, frame_1], None);

        let result = seq.interpolate(0.5);
        assert!(
            result.is_err(),
            "expected an error for mismatched shape lengths, got {result:?}"
        );
    }

    #[test]
    fn test_directory_rejects_mismatched_frame_dimensions() {
        let temp_dir = TempDir::new().expect("test: temp dir creation should succeed");

        // Frame 0 establishes the sequence's dimensions (shape len 4).
        let frame0 = FrameJson {
            shape: vec![0.0; 4],
            expression: vec![0.0; 2],
            pose: vec![0.0; 3],
            translation: Some([0.0, 0.0, 0.0]),
        };
        let json0 = serde_json::to_string(&frame0).expect("test: serialize");
        fs::write(temp_dir.path().join("frame_0000.json"), json0)
            .expect("test: file operation should succeed");

        // Frame 1 has a DIFFERENT shape length -- this must be rejected
        // rather than silently accepted. The `Directory` source previously
        // performed no dimension validation at all, unlike the JSON and
        // NPZ sources.
        let frame1 = FrameJson {
            shape: vec![0.0; 7],
            expression: vec![0.0; 2],
            pose: vec![0.0; 3],
            translation: Some([0.0, 0.0, 0.0]),
        };
        let json1 = serde_json::to_string(&frame1).expect("test: serialize");
        fs::write(temp_dir.path().join("frame_0001.json"), json1)
            .expect("test: file operation should succeed");

        let mut seq = FlameSequence::from_directory(temp_dir.path(), "frame_{:04}.json", 2, None)
            .expect("test: from_directory should succeed (frame 0 sets the dimensions)");

        // Frame 0 matches the established dimensions.
        let f0 = seq.get_frame(0).expect("test: frame 0 should load");
        assert_eq!(f0.shape.len(), 4);

        // Frame 1's mismatched shape length must be rejected.
        let result = seq.get_frame(1);
        assert!(
            result.is_err(),
            "expected an error for mismatched directory frame dimensions"
        );
    }

    #[cfg(feature = "npz")]
    #[test]
    fn test_npz_fps_roundtrip() {
        use ndarray::{Array1, Array2};
        use ndarray_npy::NpzWriter;
        use std::fs::File;

        let temp_dir = TempDir::new().expect("test: temp dir creation should succeed");
        let npz_path = temp_dir.path().join("test_sequence_fps.npz");

        let num_frames = 4;
        let shape_data: Array2<f32> =
            Array2::from_shape_fn((num_frames, 3), |(i, j)| (i + j) as f32);
        let expr_data: Array2<f32> = Array2::from_shape_fn((num_frames, 2), |(_, _)| 0.0);
        let pose_data: Array2<f32> = Array2::from_shape_fn((num_frames, 3), |(_, _)| 0.0);
        let fps_data: Array1<f32> = Array1::from_vec(vec![24.0]);

        let file = File::create(&npz_path).expect("test: file creation should succeed");
        let mut npz = NpzWriter::new(file);
        npz.add_array("shape", &shape_data)
            .expect("test: array write should succeed");
        npz.add_array("expression", &expr_data)
            .expect("test: array write should succeed");
        npz.add_array("pose", &pose_data)
            .expect("test: array write should succeed");
        npz.add_array("fps", &fps_data)
            .expect("test: array write should succeed");
        npz.finish().expect("test: npz write should succeed");

        let seq = FlameSequence::from_npz(&npz_path).expect("test: npz load should succeed");
        assert_eq!(
            seq.fps(),
            Some(24.0),
            "fps array in the NPZ file should be read, not defaulted to None"
        );
    }
}
