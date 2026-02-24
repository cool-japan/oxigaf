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
    /// Load from JSON file on demand
    JsonFile {
        path: PathBuf,
        metadata: SequenceMetadata,
    },
    /// Load from NPZ file on demand
    #[allow(dead_code)]
    NpzFile {
        path: PathBuf,
        metadata: SequenceMetadata,
    },
    /// Load from directory of per-frame files
    Directory {
        path: PathBuf,
        file_pattern: String,
        #[allow(dead_code)]
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
            // Lazy loading for large sequences
            Ok(Self {
                source: SequenceSource::JsonFile {
                    path: path.to_path_buf(),
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
    #[allow(clippy::too_many_lines)]
    pub fn from_npz(path: &Path) -> Result<Self, FlameError> {
        use ndarray_npy::NpzReader;
        use std::fs::File;

        tracing::info!("Loading FLAME sequence from NPZ: {}", path.display());

        let file = File::open(path).map_err(|e| FlameError::IoError {
            source: e,
            path: path.to_path_buf(),
        })?;

        let mut npz = NpzReader::new(file)
            .map_err(|e| FlameError::InvalidParams(format!("Failed to open NPZ file: {e}")))?;

        // Load shape array
        let shape_arr: ndarray::Array2<f32> =
            npz.by_name("shape").map_err(|e| FlameError::NpzLoad {
                name: "shape".to_string(),
                source: e,
            })?;

        // Load expression array (try both "expression" and "expr" keys)
        let expression_arr: ndarray::Array2<f32> = npz
            .by_name("expression")
            .or_else(|_| npz.by_name("expr"))
            .map_err(|e| FlameError::NpzLoad {
                name: "expression/expr".to_string(),
                source: e,
            })?;

        // Load pose array
        let pose_arr: ndarray::Array2<f32> =
            npz.by_name("pose").map_err(|e| FlameError::NpzLoad {
                name: "pose".to_string(),
                source: e,
            })?;

        // Optional translation array
        let translation_arr: Option<ndarray::Array2<f32>> = npz.by_name("translation").ok();

        // Validate shapes
        let num_frames = shape_arr.nrows();
        if expression_arr.nrows() != num_frames {
            return Err(FlameError::ShapeMismatch {
                name: "expression".to_string(),
                expected: format!("{num_frames} frames"),
                got: format!("{} frames", expression_arr.nrows()),
            });
        }
        if pose_arr.nrows() != num_frames {
            return Err(FlameError::ShapeMismatch {
                name: "pose".to_string(),
                expected: format!("{num_frames} frames"),
                got: format!("{} frames", pose_arr.nrows()),
            });
        }
        if let Some(ref trans) = translation_arr {
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

        let metadata = SequenceMetadata {
            num_frames,
            fps: None, // NPZ doesn't typically store FPS metadata
            n_shape: shape_arr.ncols(),
            n_expression: expression_arr.ncols(),
            n_pose: pose_arr.ncols(),
        };

        // For small sequences, load all into memory
        // For large sequences (>1000 frames), use lazy loading
        if num_frames <= 1000 {
            let mut frames = Vec::with_capacity(num_frames);
            for i in 0..num_frames {
                let shape = shape_arr.row(i).to_vec();
                let expression = expression_arr.row(i).to_vec();
                let pose = pose_arr.row(i).to_vec();
                let translation = if let Some(ref trans) = translation_arr {
                    [trans[[i, 0]], trans[[i, 1]], trans[[i, 2]]]
                } else {
                    [0.0, 0.0, 0.0]
                };

                frames.push(FlameParams {
                    shape,
                    expression,
                    pose,
                    translation,
                });
            }

            Ok(Self {
                source: SequenceSource::Memory(frames),
                cache: LruCache::new(DEFAULT_CACHE_SIZE),
                num_frames,
                fps: None,
            })
        } else {
            // Lazy loading for large sequences
            Ok(Self {
                source: SequenceSource::NpzFile {
                    path: path.to_path_buf(),
                    metadata: metadata.clone(),
                },
                cache: LruCache::new(DEFAULT_CACHE_SIZE),
                num_frames,
                fps: None,
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

        // Load first frame to get metadata
        let first_frame_path = dir.join(pattern.replace("{}", "0").replace("{:04}", "0000"));
        let first_params = load_frame_from_file(&first_frame_path)?;

        let metadata = SequenceMetadata {
            num_frames,
            fps,
            n_shape: first_params.shape.len(),
            n_expression: first_params.expression.len(),
            n_pose: first_params.pose.len(),
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
                    SequenceSource::JsonFile { path, metadata } => {
                        load_frame_from_json(path, idx, metadata)?
                    }
                    SequenceSource::NpzFile { path, metadata } => {
                        load_frame_from_npz(path, idx, metadata)?
                    }
                    SequenceSource::Directory {
                        path,
                        file_pattern,
                        metadata: _,
                    } => {
                        let frame_path = format_frame_path(path, file_pattern, idx);
                        load_frame_from_file(&frame_path)?
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
            SequenceSource::JsonFile { path, metadata } => {
                load_frame_from_json(path, frame_idx, metadata)?
            }
            SequenceSource::NpzFile { path, metadata } => {
                load_frame_from_npz(path, frame_idx, metadata)?
            }
            SequenceSource::Directory {
                path,
                file_pattern,
                metadata: _,
            } => {
                let frame_path = format_frame_path(path, file_pattern, frame_idx);
                load_frame_from_file(&frame_path)?
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

        // Linear interpolation: params = (1-t) * params_0 + t * params_1
        Ok(FlameParams {
            shape: lerp_vec(&params_0.shape, &params_1.shape, t),
            expression: lerp_vec(&params_0.expression, &params_1.expression, t),
            pose: lerp_vec(&params_0.pose, &params_1.pose, t),
            translation: [
                (1.0 - t) * params_0.translation[0] + t * params_1.translation[0],
                (1.0 - t) * params_0.translation[1] + t * params_1.translation[1],
                (1.0 - t) * params_0.translation[2] + t * params_1.translation[2],
            ],
        })
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

/// Load a single frame from JSON file
fn load_frame_from_json(
    path: &Path,
    frame_idx: usize,
    metadata: &SequenceMetadata,
) -> Result<FlameParams, FlameError> {
    let json_str = std::fs::read_to_string(path).map_err(|e| FlameError::IoError {
        source: e,
        path: path.to_path_buf(),
    })?;

    let sequence_data: SequenceJson = serde_json::from_str(&json_str)
        .map_err(|e| FlameError::InvalidParams(format!("Failed to parse JSON sequence: {e}")))?;

    if frame_idx >= sequence_data.frames.len() {
        return Err(FlameError::index_out_of_bounds(
            "load_frame_from_json",
            frame_idx,
            sequence_data.frames.len(),
        ));
    }

    frame_json_to_params(sequence_data.frames[frame_idx].clone(), metadata)
}

/// Load a single frame from NPZ file
#[cfg(feature = "npz")]
fn load_frame_from_npz(
    path: &Path,
    frame_idx: usize,
    metadata: &SequenceMetadata,
) -> Result<FlameParams, FlameError> {
    use ndarray_npy::NpzReader;
    use std::fs::File;

    let file = File::open(path).map_err(|e| FlameError::IoError {
        source: e,
        path: path.to_path_buf(),
    })?;

    let mut npz = NpzReader::new(file)
        .map_err(|e| FlameError::InvalidParams(format!("Failed to open NPZ file: {e}")))?;

    // Load arrays
    let shape_arr: ndarray::Array2<f32> =
        npz.by_name("shape").map_err(|e| FlameError::NpzLoad {
            name: "shape".to_string(),
            source: e,
        })?;

    let expression_arr: ndarray::Array2<f32> = npz
        .by_name("expression")
        .or_else(|_| npz.by_name("expr"))
        .map_err(|e| FlameError::NpzLoad {
            name: "expression/expr".to_string(),
            source: e,
        })?;

    let pose_arr: ndarray::Array2<f32> = npz.by_name("pose").map_err(|e| FlameError::NpzLoad {
        name: "pose".to_string(),
        source: e,
    })?;

    let translation_arr: Option<ndarray::Array2<f32>> = npz.by_name("translation").ok();

    // Validate frame index
    if frame_idx >= shape_arr.nrows() {
        return Err(FlameError::index_out_of_bounds(
            "load_frame_from_npz",
            frame_idx,
            shape_arr.nrows(),
        ));
    }

    // Extract frame data
    let shape = shape_arr.row(frame_idx).to_vec();
    let expression = expression_arr.row(frame_idx).to_vec();
    let pose = pose_arr.row(frame_idx).to_vec();
    let translation = if let Some(ref trans) = translation_arr {
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

/// Load a single frame from NPZ file (feature not enabled)
#[cfg(not(feature = "npz"))]
fn load_frame_from_npz(
    _path: &Path,
    _frame_idx: usize,
    _metadata: &SequenceMetadata,
) -> Result<FlameParams, FlameError> {
    Err(FlameError::InvalidParams(
        "NPZ support not enabled. Enable the 'npz' feature flag.".to_string(),
    ))
}

/// Load a single frame from individual file
fn load_frame_from_file(path: &Path) -> Result<FlameParams, FlameError> {
    let json_str = std::fs::read_to_string(path).map_err(|e| FlameError::IoError {
        source: e,
        path: path.to_path_buf(),
    })?;

    let frame: FrameJson = serde_json::from_str(&json_str)
        .map_err(|e| FlameError::InvalidParams(format!("Failed to parse frame JSON: {e}")))?;

    Ok(FlameParams {
        shape: frame.shape,
        expression: frame.expression,
        pose: frame.pose,
        translation: frame.translation.unwrap_or([0.0, 0.0, 0.0]),
    })
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

/// Linear interpolation between two vectors
fn lerp_vec(a: &[f32], b: &[f32], t: f32) -> Vec<f32> {
    a.iter()
        .zip(b.iter())
        .map(|(&a_i, &b_i)| (1.0 - t) * a_i + t * b_i)
        .collect()
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
}
