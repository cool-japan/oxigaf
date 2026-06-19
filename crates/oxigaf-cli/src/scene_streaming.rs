//! Streaming scene management and progressive loading for large 3DGS scenes.
//!
//! This module provides view-frustum culling, spatial chunking, LOD selection,
//! LRU caching, and priority-based progressive loading for 3D Gaussian Splatting
//! scenes that exceed available GPU memory.
//!
//! # Example
//! ```rust
//! use oxigaf_cli::scene_streaming::{
//!     StreamingConfig, StreamingScene, ViewFrustum,
//!     ss_compute_stats, ss_format_stats,
//! };
//!
//! let n = 64usize;
//! let positions: Vec<f32> = (0..n * 3).map(|i| (i as f32) * 0.1 - 3.2).collect();
//! let config = StreamingConfig {
//!     memory_budget_bytes: 256 * 1024 * 1024,
//!     chunk_divisions: [4, 4, 4],
//!     lod_distances: [5.0, 20.0, 50.0],
//!     max_chunks_per_frame: 4,
//!     preload_radius: 10.0,
//! };
//! let scene = StreamingScene::new(config, &positions, n, 9)
//!     .expect("scene init failed");
//! let frustum = ViewFrustum::default_front([0.0, 0.0, 0.0]);
//! let stats = ss_compute_stats(&scene, &frustum);
//! println!("{}", ss_format_stats(&stats));
//! ```

use thiserror::Error;

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Xorshift64 PRNG (no rand crate). Satisfies the COOLJAPAN no-rand policy.
#[cfg(test)]
fn xorshift64(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    if *state == 0 {
        *state = 1;
    }
    *state
}

/// Normalize a 3-vector; returns the input unchanged if near-zero length.
#[inline]
fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len < 1e-9 {
        return v;
    }
    [v[0] / len, v[1] / len, v[2] / len]
}

/// Cross product of two 3-vectors.
#[inline]
fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Dot product of two 3-vectors.
#[inline]
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Subtract two 3-vectors.
#[inline]
fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Length of a 3-vector.
#[inline]
fn len3(v: [f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

// ---------------------------------------------------------------------------
// StreamingError
// ---------------------------------------------------------------------------

/// Errors from the streaming scene management pipeline.
#[derive(Debug, Error)]
pub enum StreamingError {
    /// Chunk size (or chunk division) is invalid (must be > 0).
    #[error("Invalid chunk size: {size} (must be > 0)")]
    InvalidChunkSize { size: usize },

    /// Memory budget was exceeded.
    #[error("Memory budget exceeded: {used} > {budget} bytes")]
    MemoryBudgetExceeded { used: usize, budget: usize },

    /// A chunk with the given ID could not be found.
    #[error("Chunk {id} not found")]
    ChunkNotFound { id: u64 },

    /// The scene has no Gaussians.
    #[error("Scene is empty")]
    EmptyScene,

    /// The frustum near/far range is degenerate.
    #[error("Invalid frustum: near={near} >= far={far}")]
    InvalidFrustum { near: f32, far: f32 },

    /// Positions array length does not match n * 3.
    #[error("Array length mismatch: positions={pos}, expected n*3={expected}")]
    LengthMismatch { pos: usize, expected: usize },
}

// ---------------------------------------------------------------------------
// ViewFrustum
// ---------------------------------------------------------------------------

/// A perspective view frustum defined by eye position, orientation, and FOV.
#[derive(Debug, Clone)]
pub struct ViewFrustum {
    /// Eye / camera position in world space.
    pub eye: [f32; 3],
    /// Unit forward vector (camera looks in this direction).
    pub forward: [f32; 3],
    /// Unit right vector (orthogonal to forward and up).
    pub right: [f32; 3],
    /// Unit up vector.
    pub up: [f32; 3],
    /// Horizontal FOV in radians.
    pub fov_x_rad: f32,
    /// Vertical FOV in radians.
    pub fov_y_rad: f32,
    /// Near clipping plane distance (> 0).
    pub near: f32,
    /// Far clipping plane distance (> near).
    pub far: f32,
}

impl ViewFrustum {
    /// Construct a ViewFrustum, deriving the right vector automatically.
    ///
    /// `forward` and `up` do not have to be unit length — they will be
    /// normalized internally. `near` must be strictly less than `far`.
    pub fn new(
        eye: [f32; 3],
        forward: [f32; 3],
        up: [f32; 3],
        fov_x: f32,
        fov_y: f32,
        near: f32,
        far: f32,
    ) -> Result<Self, StreamingError> {
        if near >= far {
            return Err(StreamingError::InvalidFrustum { near, far });
        }
        let fwd = normalize3(forward);
        let right = normalize3(cross3(fwd, normalize3(up)));
        let up_ortho = normalize3(cross3(right, fwd));
        Ok(Self {
            eye,
            forward: fwd,
            right,
            up: up_ortho,
            fov_x_rad: fov_x,
            fov_y_rad: fov_y,
            near,
            far,
        })
    }

    /// Convenience constructor: eye looks down −Z with 90° FOV (square aspect).
    pub fn default_front(eye: [f32; 3]) -> Self {
        Self {
            eye,
            forward: [0.0, 0.0, -1.0],
            right: [1.0, 0.0, 0.0],
            up: [0.0, 1.0, 0.0],
            fov_x_rad: std::f32::consts::FRAC_PI_2,
            fov_y_rad: std::f32::consts::FRAC_PI_2,
            near: 0.1,
            far: 1000.0,
        }
    }

    /// Distance from the eye to a world-space point (Euclidean).
    #[inline]
    pub fn distance(&self, point: [f32; 3]) -> f32 {
        len3(sub3(point, self.eye))
    }

    /// Fast cone pre-culling test: is `point` within the half-angle fov_x/2
    /// cone centred on the forward direction?
    pub fn point_in_cone(&self, point: [f32; 3]) -> bool {
        let to_point = sub3(point, self.eye);
        let dist = len3(to_point);
        if dist < 1e-9 {
            return true; // at eye
        }
        let dir = [to_point[0] / dist, to_point[1] / dist, to_point[2] / dist];
        let cos_angle = dot3(dir, self.forward);
        let half_angle = self.fov_x_rad * 0.5;
        cos_angle >= half_angle.cos()
    }
}

// ---------------------------------------------------------------------------
// Frustum culling functions
// ---------------------------------------------------------------------------

/// Per-point frustum visibility mask (true = visible).
///
/// A point is visible when:
/// 1. Its depth along the forward axis is in \[near, far\].
/// 2. Its horizontal angle from forward is within fov_x/2.
/// 3. Its vertical angle from forward is within fov_y/2.
///
/// `positions` is a flat `n×3` array (stride 3). `n` is the number of points.
pub fn ss_frustum_cull_points(frustum: &ViewFrustum, positions: &[f32], n: usize) -> Vec<bool> {
    let half_x = frustum.fov_x_rad * 0.5;
    let half_y = frustum.fov_y_rad * 0.5;
    let tan_hx = half_x.tan();
    let tan_hy = half_y.tan();

    (0..n)
        .map(|i| {
            let base = i * 3;
            if base + 2 >= positions.len() {
                return false;
            }
            let p = [positions[base], positions[base + 1], positions[base + 2]];
            let rel = sub3(p, frustum.eye);

            // Depth along forward axis
            let depth = dot3(rel, frustum.forward);
            if depth < frustum.near || depth > frustum.far {
                return false;
            }

            // Lateral extents at that depth
            let lateral_x = dot3(rel, frustum.right);
            let lateral_y = dot3(rel, frustum.up);

            // Plane-based test: |lateral| / depth ≤ tan(half_fov)
            let abs_x = lateral_x.abs();
            let abs_y = lateral_y.abs();
            abs_x <= depth * tan_hx && abs_y <= depth * tan_hy
        })
        .collect()
}

/// Per-sphere frustum visibility mask.
///
/// A sphere (centre at `positions[i*3..i*3+3]`, radius `radii[i]`) is visible
/// when the sphere overlaps the frustum volume. Uses a conservative depth +
/// cone test with radius expansion.
pub fn ss_frustum_cull_spheres(
    frustum: &ViewFrustum,
    positions: &[f32],
    radii: &[f32],
    n: usize,
) -> Vec<bool> {
    let half_x = frustum.fov_x_rad * 0.5;
    let half_y = frustum.fov_y_rad * 0.5;
    let tan_hx = half_x.tan();
    let tan_hy = half_y.tan();

    (0..n)
        .map(|i| {
            let base = i * 3;
            if base + 2 >= positions.len() || i >= radii.len() {
                return false;
            }
            let p = [positions[base], positions[base + 1], positions[base + 2]];
            let r = radii[i];
            let rel = sub3(p, frustum.eye);

            // Depth along forward axis (expanded by radius)
            let depth = dot3(rel, frustum.forward);
            if depth + r < frustum.near || depth - r > frustum.far {
                return false;
            }

            // Use the clamped depth for plane tests (avoid division by zero)
            let test_depth = depth.max(frustum.near);

            // Lateral extents: expand frustum plane margins by radius
            let lateral_x = dot3(rel, frustum.right).abs();
            let lateral_y = dot3(rel, frustum.up).abs();
            lateral_x - r <= test_depth * tan_hx && lateral_y - r <= test_depth * tan_hy
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Scene bounds
// ---------------------------------------------------------------------------

/// Compute the axis-aligned bounding box of all Gaussian positions.
///
/// Returns `([min_x, min_y, min_z], [max_x, max_y, max_z])`.
/// For `n == 0`, returns `([0,0,0], [0,0,0])`.
pub fn ss_compute_scene_bounds(positions: &[f32], n: usize) -> ([f32; 3], [f32; 3]) {
    if n == 0 {
        return ([0.0; 3], [0.0; 3]);
    }
    let mut min = [f32::MAX; 3];
    let mut max = [f32::MIN; 3];
    for i in 0..n {
        let base = i * 3;
        if base + 2 < positions.len() {
            for d in 0..3 {
                let v = positions[base + d];
                if v < min[d] {
                    min[d] = v;
                }
                if v > max[d] {
                    max[d] = v;
                }
            }
        }
    }
    (min, max)
}

// ---------------------------------------------------------------------------
// Spatial chunking
// ---------------------------------------------------------------------------

/// Priority level for loading a streaming chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LoadPriority {
    Low = 0,
    Medium = 1,
    High = 2,
    Critical = 3,
}

/// A spatially bounded chunk of Gaussians used for progressive loading.
#[derive(Debug, Clone)]
pub struct StreamingChunk {
    /// Unique chunk identifier (Morton-code-like).
    pub id: u64,
    /// World-space minimum corner of this chunk's AABB.
    pub bounds_min: [f32; 3],
    /// World-space maximum corner of this chunk's AABB.
    pub bounds_max: [f32; 3],
    /// Global indices of the Gaussians that reside in this chunk.
    pub gaussian_indices: Vec<usize>,
    /// Number of Gaussians (mirrors `gaussian_indices.len()`).
    pub n_gaussians: usize,
    /// Current load priority for this chunk.
    pub priority: LoadPriority,
    /// LOD level: 0 = full, 1 = halved, 2 = quartered.
    pub lod_level: u8,
    /// Whether this chunk has been loaded into GPU memory.
    pub loaded: bool,
    /// Estimated memory usage in bytes for this chunk.
    pub memory_bytes: usize,
}

/// Compute the unique chunk ID from grid indices.
///
/// Uses a Morton-code-like encoding:
/// `id = (iz << 42) | (iy << 21) | ix`
#[inline]
pub fn ss_chunk_id(ix: u32, iy: u32, iz: u32, _nx: u32, _ny: u32) -> u64 {
    ((iz as u64) << 42) | ((iy as u64) << 21) | (ix as u64)
}

/// Cell index `(ix, iy, iz)` within the streaming grid.
pub struct ChunkCellIndex {
    /// Index along the X axis.
    pub ix: u32,
    /// Index along the Y axis.
    pub iy: u32,
    /// Index along the Z axis.
    pub iz: u32,
}

/// Grid dimensions `(nx, ny, nz)` for [`ss_chunk_bounds`].
pub struct ChunkGridDims {
    /// Number of divisions along X.
    pub nx: u32,
    /// Number of divisions along Y.
    pub ny: u32,
    /// Number of divisions along Z.
    pub nz: u32,
}

/// Compute the AABB of a single grid cell within a parent AABB.
pub fn ss_chunk_bounds(
    bounds_min: [f32; 3],
    bounds_max: [f32; 3],
    cell: ChunkCellIndex,
    grid: ChunkGridDims,
) -> ([f32; 3], [f32; 3]) {
    let ChunkCellIndex { ix, iy, iz } = cell;
    let ChunkGridDims { nx, ny, nz } = grid;
    let step = [
        (bounds_max[0] - bounds_min[0]) / nx as f32,
        (bounds_max[1] - bounds_min[1]) / ny as f32,
        (bounds_max[2] - bounds_min[2]) / nz as f32,
    ];
    let cell_min = [
        bounds_min[0] + ix as f32 * step[0],
        bounds_min[1] + iy as f32 * step[1],
        bounds_min[2] + iz as f32 * step[2],
    ];
    let cell_max = [
        cell_min[0] + step[0],
        cell_min[1] + step[1],
        cell_min[2] + step[2],
    ];
    (cell_min, cell_max)
}

/// Estimate memory usage for a chunk (positions + rotations + scales +
/// opacities + SH coefficients per Gaussian, all f32).
fn estimate_chunk_bytes(n_gaussians: usize, sh_channels: usize) -> usize {
    // per Gaussian: 3 (pos) + 4 (rot) + 3 (scale) + 1 (opacity) + sh_channels
    let floats_per_gaussian = 3 + 4 + 3 + 1 + sh_channels;
    n_gaussians * floats_per_gaussian * std::mem::size_of::<f32>()
}

/// Divide the scene into a regular grid of chunks.
///
/// Each Gaussian is assigned to exactly one cell based on its position.
/// `chunk_divisions` must have all components > 0 and `n` must be > 0.
pub fn ss_chunk_scene(
    positions: &[f32],
    n: usize,
    chunk_divisions: [u32; 3],
) -> Result<Vec<StreamingChunk>, StreamingError> {
    if n == 0 {
        return Err(StreamingError::EmptyScene);
    }
    if positions.len() < n * 3 {
        return Err(StreamingError::LengthMismatch {
            pos: positions.len(),
            expected: n * 3,
        });
    }
    for &div in &chunk_divisions {
        if div == 0 {
            return Err(StreamingError::InvalidChunkSize { size: div as usize });
        }
    }

    let [nx, ny, nz] = chunk_divisions;
    let (bmin, bmax) = ss_compute_scene_bounds(positions, n);

    // Guard against degenerate (flat) scenes by adding a tiny epsilon
    let extent = [
        (bmax[0] - bmin[0]).max(1e-6),
        (bmax[1] - bmin[1]).max(1e-6),
        (bmax[2] - bmin[2]).max(1e-6),
    ];

    // Pre-build all chunks
    let total_cells = (nx * ny * nz) as usize;
    let mut gaussian_lists: Vec<Vec<usize>> = vec![Vec::new(); total_cells];

    for i in 0..n {
        let base = i * 3;
        let p = [positions[base], positions[base + 1], positions[base + 2]];

        // Determine grid cell for each dimension; clamp to [0, n-1]
        let gx = (((p[0] - bmin[0]) / extent[0]) * nx as f32)
            .floor()
            .clamp(0.0, (nx - 1) as f32) as u32;
        let gy = (((p[1] - bmin[1]) / extent[1]) * ny as f32)
            .floor()
            .clamp(0.0, (ny - 1) as f32) as u32;
        let gz = (((p[2] - bmin[2]) / extent[2]) * nz as f32)
            .floor()
            .clamp(0.0, (nz - 1) as f32) as u32;

        let flat_idx = (gz * ny * nx + gy * nx + gx) as usize;
        gaussian_lists[flat_idx].push(i);
    }

    let mut chunks = Vec::with_capacity(total_cells);
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let flat = (iz * ny * nx + iy * nx + ix) as usize;
                let indices = std::mem::take(&mut gaussian_lists[flat]);
                let n_g = indices.len();
                let (cmin, cmax) = ss_chunk_bounds(
                    bmin,
                    bmax,
                    ChunkCellIndex { ix, iy, iz },
                    ChunkGridDims { nx, ny, nz },
                );
                let id = ss_chunk_id(ix, iy, iz, nx, ny);
                let mem = estimate_chunk_bytes(n_g, 9); // default sh_channels=9
                chunks.push(StreamingChunk {
                    id,
                    bounds_min: cmin,
                    bounds_max: cmax,
                    gaussian_indices: indices,
                    n_gaussians: n_g,
                    priority: LoadPriority::Low,
                    lod_level: 0,
                    loaded: false,
                    memory_bytes: mem,
                });
            }
        }
    }
    Ok(chunks)
}

// ---------------------------------------------------------------------------
// Priority computation
// ---------------------------------------------------------------------------

/// Minimum distance from a point to an AABB (0 if inside).
pub fn ss_distance_to_aabb(point: [f32; 3], bounds_min: [f32; 3], bounds_max: [f32; 3]) -> f32 {
    let mut sq_dist = 0.0f32;
    for d in 0..3 {
        let v = point[d];
        if v < bounds_min[d] {
            let diff = bounds_min[d] - v;
            sq_dist += diff * diff;
        } else if v > bounds_max[d] {
            let diff = v - bounds_max[d];
            sq_dist += diff * diff;
        }
    }
    sq_dist.sqrt()
}

/// Compute the AABB centroid.
#[inline]
fn aabb_centroid(bmin: [f32; 3], bmax: [f32; 3]) -> [f32; 3] {
    [
        (bmin[0] + bmax[0]) * 0.5,
        (bmin[1] + bmax[1]) * 0.5,
        (bmin[2] + bmax[2]) * 0.5,
    ]
}

/// Assign a load priority to a chunk relative to the view frustum.
///
/// - Distance < near * 2   → Critical
/// - Chunk centroid in frustum cone AND mid-range → High
/// - Chunk centroid in frustum cone              → Medium
/// - Otherwise                                  → Low
pub fn ss_compute_priority(chunk: &StreamingChunk, frustum: &ViewFrustum) -> LoadPriority {
    let centroid = aabb_centroid(chunk.bounds_min, chunk.bounds_max);
    let dist = ss_distance_to_aabb(frustum.eye, chunk.bounds_min, chunk.bounds_max);

    if dist < frustum.near * 2.0 {
        return LoadPriority::Critical;
    }

    let in_cone = frustum.point_in_cone(centroid);
    let range = frustum.far - frustum.near;
    let mid_far = frustum.near + range * 0.67;

    if in_cone && dist < mid_far {
        LoadPriority::High
    } else if in_cone {
        LoadPriority::Medium
    } else {
        LoadPriority::Low
    }
}

/// Sort chunks by priority descending, then by distance to eye ascending.
pub fn ss_sort_chunks_by_priority(chunks: &mut [StreamingChunk], frustum: &ViewFrustum) {
    // Recompute priorities before sorting
    for chunk in chunks.iter_mut() {
        chunk.priority = ss_compute_priority(chunk, frustum);
    }
    chunks.sort_by(|a, b| {
        // Higher priority first
        let pa = a.priority as i32;
        let pb = b.priority as i32;
        if pa != pb {
            return pb.cmp(&pa);
        }
        // Closer first among same priority
        let da = ss_distance_to_aabb(frustum.eye, a.bounds_min, a.bounds_max);
        let db = ss_distance_to_aabb(frustum.eye, b.bounds_min, b.bounds_max);
        da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
    });
}

// ---------------------------------------------------------------------------
// LOD selection
// ---------------------------------------------------------------------------

/// Select the LOD level for a Gaussian cluster at a given distance.
///
/// Returns 0 (full), 1 (halved), or 2 (quartered) based on `lod_distances`.
pub fn ss_select_lod(
    distance: f32,
    _max_gaussians: usize,
    _n_gaussians: usize,
    lod_distances: &[f32; 3],
) -> u8 {
    if distance <= lod_distances[0] {
        0
    } else if distance <= lod_distances[2] {
        // Between threshold 0 and threshold 2
        if distance <= lod_distances[1] {
            1
        } else {
            2
        }
    } else {
        2
    }
}

/// Deterministically subsample Gaussian indices according to LOD level.
///
/// - LOD 0: returns all indices unchanged.
/// - LOD 1: every other index (even indices: 0, 2, 4, …).
/// - LOD 2: every fourth index (indices 0, 4, 8, …).
///
/// The `seed` parameter is reserved for future stochastic LOD selection.
pub fn ss_lod_subsample_indices(indices: &[usize], lod_level: u8, _seed: u64) -> Vec<usize> {
    match lod_level {
        0 => indices.to_vec(),
        1 => indices.iter().step_by(2).copied().collect(),
        _ => indices.iter().step_by(4).copied().collect(),
    }
}

// ---------------------------------------------------------------------------
// Chunk cache (LRU eviction)
// ---------------------------------------------------------------------------

/// A simple LRU cache for streaming chunks.
///
/// Tracks which chunks are resident in GPU memory and evicts the least recently
/// used chunk when the memory budget is exceeded.
#[derive(Debug, Clone)]
pub struct ChunkCache {
    /// Total memory capacity in bytes.
    pub capacity_bytes: usize,
    /// Currently used memory in bytes.
    pub used_bytes: usize,
    /// Entries: `(chunk_id, last_access_step, bytes)`.
    pub chunks: Vec<(u64, usize)>,
    /// Monotonically increasing logical clock.
    pub access_step: usize,
    // Private field storing the byte counts per cached chunk
    entry_bytes: Vec<usize>,
}

impl ChunkCache {
    /// Create an empty cache with the given memory capacity.
    pub fn new(capacity_bytes: usize) -> Self {
        Self {
            capacity_bytes,
            used_bytes: 0,
            chunks: Vec::new(),
            access_step: 0,
            entry_bytes: Vec::new(),
        }
    }

    /// Returns true if `bytes` can fit in the remaining free space.
    #[inline]
    pub fn can_fit(&self, bytes: usize) -> bool {
        self.used_bytes + bytes <= self.capacity_bytes
    }

    /// Update the access step for `chunk_id`; returns true if found.
    pub fn touch(&mut self, chunk_id: u64) -> bool {
        self.access_step += 1;
        let step = self.access_step;
        if let Some(pos) = self.chunks.iter().position(|(id, _)| *id == chunk_id) {
            self.chunks[pos].1 = step;
            true
        } else {
            false
        }
    }

    /// Remove the least recently used chunk; returns its ID (if any).
    pub fn evict_lru(&mut self) -> Option<u64> {
        if self.chunks.is_empty() {
            return None;
        }
        // Find index with lowest last_access_step
        let lru_pos = self
            .chunks
            .iter()
            .enumerate()
            .min_by_key(|(_, (_, step))| *step)
            .map(|(i, _)| i)?;

        let (evicted_id, _) = self.chunks.remove(lru_pos);
        let bytes = self.entry_bytes.remove(lru_pos);
        self.used_bytes = self.used_bytes.saturating_sub(bytes);
        Some(evicted_id)
    }

    /// Insert a chunk into the cache, evicting LRU entries until it fits.
    pub fn insert(&mut self, chunk_id: u64, bytes: usize) -> Result<(), StreamingError> {
        if bytes > self.capacity_bytes {
            return Err(StreamingError::MemoryBudgetExceeded {
                used: bytes,
                budget: self.capacity_bytes,
            });
        }
        // Evict until space is available
        while !self.can_fit(bytes) {
            if self.evict_lru().is_none() {
                break;
            }
        }
        self.access_step += 1;
        self.chunks.push((chunk_id, self.access_step));
        self.entry_bytes.push(bytes);
        self.used_bytes += bytes;
        Ok(())
    }

    /// Remove all cached entries.
    pub fn evict_all(&mut self) {
        self.chunks.clear();
        self.entry_bytes.clear();
        self.used_bytes = 0;
    }

    /// Fraction of capacity currently in use (`used_bytes / capacity_bytes`).
    pub fn utilization(&self) -> f32 {
        if self.capacity_bytes == 0 {
            return 0.0;
        }
        self.used_bytes as f32 / self.capacity_bytes as f32
    }
}

// ---------------------------------------------------------------------------
// StreamingConfig and StreamingScene
// ---------------------------------------------------------------------------

/// Configuration parameters for the streaming scene manager.
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Maximum GPU/RAM memory that may be used for resident chunks.
    pub memory_budget_bytes: usize,
    /// How many grid cells to create along each axis.
    pub chunk_divisions: [u32; 3],
    /// Distance thresholds for LOD 0 / 1 / 2 transitions.
    pub lod_distances: [f32; 3],
    /// Maximum number of chunk load requests issued per `update_view` call.
    pub max_chunks_per_frame: usize,
    /// Chunks within this world-space distance of the eye are pre-loaded.
    pub preload_radius: f32,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            memory_budget_bytes: 512 * 1024 * 1024,
            chunk_divisions: [4, 4, 4],
            lod_distances: [10.0, 30.0, 80.0],
            max_chunks_per_frame: 4,
            preload_radius: 15.0,
        }
    }
}

/// Scene manager that owns the spatial chunk decomposition and the LRU cache.
pub struct StreamingScene {
    /// All spatial chunks (one per grid cell, even if empty).
    pub chunks: Vec<StreamingChunk>,
    /// LRU memory cache for loaded chunks.
    pub cache: ChunkCache,
    /// Active configuration.
    pub config: StreamingConfig,
    /// Total number of Gaussians in the original scene.
    pub total_gaussians: usize,
    /// Number of spherical-harmonic channels per Gaussian.
    pub sh_channels: usize,
    /// Frame counter (incremented each `update_view`).
    frame_counter: u64,
}

impl StreamingScene {
    /// Build a new `StreamingScene` from flat position data.
    ///
    /// Chunks the scene according to `config.chunk_divisions` and initialises
    /// the LRU cache with `config.memory_budget_bytes`.
    pub fn new(
        config: StreamingConfig,
        positions: &[f32],
        n: usize,
        sh_channels: usize,
    ) -> Result<Self, StreamingError> {
        if n == 0 {
            return Err(StreamingError::EmptyScene);
        }
        if positions.len() < n * 3 {
            return Err(StreamingError::LengthMismatch {
                pos: positions.len(),
                expected: n * 3,
            });
        }
        let mut chunks = ss_chunk_scene(positions, n, config.chunk_divisions)?;
        // Correct memory estimates using actual sh_channels
        for chunk in &mut chunks {
            chunk.memory_bytes = estimate_chunk_bytes(chunk.n_gaussians, sh_channels);
        }
        let cache = ChunkCache::new(config.memory_budget_bytes);
        Ok(Self {
            chunks,
            cache,
            config,
            total_gaussians: n,
            sh_channels,
            frame_counter: 0,
        })
    }

    /// Recompute priorities, sort chunks, and return up to
    /// `config.max_chunks_per_frame` chunk IDs that should be loaded next.
    pub fn update_view(&mut self, frustum: &ViewFrustum) -> Vec<u64> {
        self.frame_counter += 1;

        // Recompute LOD levels and priorities for every chunk
        for chunk in &mut self.chunks {
            let dist = ss_distance_to_aabb(frustum.eye, chunk.bounds_min, chunk.bounds_max);
            chunk.lod_level = ss_select_lod(
                dist,
                self.total_gaussians,
                chunk.n_gaussians,
                &self.config.lod_distances,
            );
            chunk.priority = ss_compute_priority(chunk, frustum);
        }

        // Sort (clones priority internally, then sorts)
        ss_sort_chunks_by_priority(&mut self.chunks, frustum);

        // Collect chunks to load: not yet loaded, high enough priority, within limits
        let mut to_load = Vec::new();
        for chunk in &self.chunks {
            if to_load.len() >= self.config.max_chunks_per_frame {
                break;
            }
            if chunk.loaded {
                continue;
            }
            if chunk.priority == LoadPriority::Low && chunk.n_gaussians == 0 {
                continue;
            }
            // Check if within preload radius
            let dist = ss_distance_to_aabb(frustum.eye, chunk.bounds_min, chunk.bounds_max);
            if chunk.priority >= LoadPriority::Medium || dist <= self.config.preload_radius {
                to_load.push(chunk.id);
            }
        }
        to_load
    }

    /// Mark a chunk as loaded, inserting it into the LRU cache.
    pub fn mark_loaded(&mut self, chunk_id: u64) -> Result<(), StreamingError> {
        let chunk_pos = self
            .chunks
            .iter()
            .position(|c| c.id == chunk_id)
            .ok_or(StreamingError::ChunkNotFound { id: chunk_id })?;

        let bytes = self.chunks[chunk_pos].memory_bytes;
        self.cache.insert(chunk_id, bytes)?;
        self.chunks[chunk_pos].loaded = true;
        Ok(())
    }

    /// Return references to all chunks whose AABB overlaps the frustum cone.
    pub fn get_visible_chunks(&self, frustum: &ViewFrustum) -> Vec<&StreamingChunk> {
        self.chunks
            .iter()
            .filter(|chunk| {
                // Use centroid-based cone test (conservative but fast)
                let centroid = aabb_centroid(chunk.bounds_min, chunk.bounds_max);
                let dist = ss_distance_to_aabb(frustum.eye, chunk.bounds_min, chunk.bounds_max);
                // Visible if centroid is in cone OR the eye is inside the chunk
                frustum.point_in_cone(centroid) || dist < 1e-6
            })
            .collect()
    }

    /// Collect every Gaussian index from all currently loaded chunks.
    pub fn get_loaded_gaussian_indices(&self) -> Vec<usize> {
        let mut out = Vec::new();
        for chunk in &self.chunks {
            if chunk.loaded {
                out.extend_from_slice(&chunk.gaussian_indices);
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Streaming statistics
// ---------------------------------------------------------------------------

/// A snapshot of streaming performance and memory state.
#[derive(Debug, Clone)]
pub struct StreamingStats {
    /// Total number of chunks in the scene.
    pub total_chunks: usize,
    /// Number of chunks currently marked as loaded.
    pub loaded_chunks: usize,
    /// `loaded_chunks / total_chunks` (0.0 if no chunks).
    pub loading_ratio: f32,
    /// Total Gaussians in the scene.
    pub total_gaussians: usize,
    /// Gaussians whose chunk centroid is in the view frustum.
    pub visible_gaussians: usize,
    /// Cache utilisation in \[0, 1\].
    pub cache_utilization: f32,
    /// Bytes currently used by the cache.
    pub memory_used_bytes: usize,
    /// Total memory budget in bytes.
    pub memory_budget_bytes: usize,
}

/// Compute streaming statistics for the current frame.
pub fn ss_compute_stats(scene: &StreamingScene, frustum: &ViewFrustum) -> StreamingStats {
    let total_chunks = scene.chunks.len();
    let loaded_chunks = scene.chunks.iter().filter(|c| c.loaded).count();
    let loading_ratio = if total_chunks == 0 {
        0.0
    } else {
        loaded_chunks as f32 / total_chunks as f32
    };

    let visible_gaussians: usize = scene
        .get_visible_chunks(frustum)
        .iter()
        .map(|c| c.n_gaussians)
        .sum();

    StreamingStats {
        total_chunks,
        loaded_chunks,
        loading_ratio,
        total_gaussians: scene.total_gaussians,
        visible_gaussians,
        cache_utilization: scene.cache.utilization(),
        memory_used_bytes: scene.cache.used_bytes,
        memory_budget_bytes: scene.config.memory_budget_bytes,
    }
}

/// Format streaming statistics as a human-readable string.
pub fn ss_format_stats(stats: &StreamingStats) -> String {
    format!(
        "StreamingStats {{ chunks: {}/{} ({:.1}%), gaussians: {} total / {} visible, \
         cache: {:.1}%, mem: {}MB / {}MB }}",
        stats.loaded_chunks,
        stats.total_chunks,
        stats.loading_ratio * 100.0,
        stats.total_gaussians,
        stats.visible_gaussians,
        stats.cache_utilization * 100.0,
        stats.memory_used_bytes / (1024 * 1024),
        stats.memory_budget_bytes / (1024 * 1024),
    )
}

/// Format streaming configuration as a human-readable string.
pub fn ss_format_config(config: &StreamingConfig) -> String {
    format!(
        "StreamingConfig {{ budget: {}MB, divisions: {:?}, lod_dist: {:?}, \
         max_per_frame: {}, preload_radius: {} }}",
        config.memory_budget_bytes / (1024 * 1024),
        config.chunk_divisions,
        config.lod_distances,
        config.max_chunks_per_frame,
        config.preload_radius,
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- helpers ----

    fn make_frustum() -> ViewFrustum {
        ViewFrustum::default_front([0.0, 0.0, 0.0])
    }

    #[allow(dead_code)]
    fn make_frustum_at(eye: [f32; 3]) -> ViewFrustum {
        ViewFrustum::default_front(eye)
    }

    /// 8 Gaussians arranged at ±0.5 on each axis.
    fn make_positions_8() -> Vec<f32> {
        vec![
            -0.5, -0.5, -0.5, // 0
            0.5, -0.5, -0.5, // 1
            -0.5, 0.5, -0.5, // 2
            0.5, 0.5, -0.5, // 3
            -0.5, -0.5, 0.5, // 4
            0.5, -0.5, 0.5, // 5
            -0.5, 0.5, 0.5, // 6
            0.5, 0.5, 0.5, // 7
        ]
    }

    fn make_scene() -> StreamingScene {
        let positions = make_positions_8();
        let config = StreamingConfig {
            memory_budget_bytes: 16 * 1024 * 1024,
            chunk_divisions: [2, 2, 2],
            lod_distances: [5.0, 20.0, 50.0],
            max_chunks_per_frame: 4,
            preload_radius: 10.0,
        };
        StreamingScene::new(config, &positions, 8, 9).expect("scene creation failed")
    }

    // ---- ViewFrustum tests ----

    #[test]
    fn test_frustum_new_invalid_near_far() {
        let result = ViewFrustum::new(
            [0.0; 3],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
            std::f32::consts::FRAC_PI_2,
            std::f32::consts::FRAC_PI_2,
            10.0,
            5.0, // near > far → error
        );
        assert!(result.is_err());
        match result.unwrap_err() {
            StreamingError::InvalidFrustum { near, far } => {
                assert_eq!(near, 10.0);
                assert_eq!(far, 5.0);
            }
            e => panic!("wrong error: {e}"),
        }
    }

    #[test]
    fn test_frustum_new_near_equal_far() {
        let result = ViewFrustum::new(
            [0.0; 3],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
            1.0,
            1.0,
            5.0,
            5.0, // equal → error
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_frustum_default_front_fov() {
        let f = ViewFrustum::default_front([1.0, 2.0, 3.0]);
        let expected_fov = std::f32::consts::FRAC_PI_2;
        assert!((f.fov_x_rad - expected_fov).abs() < 1e-6);
        assert!((f.fov_y_rad - expected_fov).abs() < 1e-6);
        assert_eq!(f.eye, [1.0, 2.0, 3.0]);
        // forward should be (0,0,-1)
        assert!((f.forward[2] + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_frustum_new_valid() {
        let f = ViewFrustum::new(
            [0.0; 3],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
            std::f32::consts::FRAC_PI_4,
            std::f32::consts::FRAC_PI_4,
            0.1,
            100.0,
        );
        assert!(f.is_ok());
    }

    #[test]
    fn test_frustum_distance() {
        let f = make_frustum();
        let dist = f.distance([3.0, 4.0, 0.0]);
        assert!((dist - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_frustum_point_in_cone_ahead() {
        let f = make_frustum();
        // Point directly ahead (along -Z) → in cone
        assert!(f.point_in_cone([0.0, 0.0, -10.0]));
    }

    #[test]
    fn test_frustum_point_in_cone_behind() {
        let f = make_frustum();
        // Point behind (+Z) → not in cone
        assert!(!f.point_in_cone([0.0, 0.0, 10.0]));
    }

    // ---- ss_frustum_cull_points ----

    #[test]
    fn test_cull_points_visible_ahead() {
        let f = make_frustum();
        // Point straight ahead at depth 5, within near=0.1 far=1000
        let positions = vec![0.0f32, 0.0, -5.0];
        let mask = ss_frustum_cull_points(&f, &positions, 1);
        assert_eq!(mask.len(), 1);
        assert!(mask[0], "point directly ahead should be visible");
    }

    #[test]
    fn test_cull_points_behind_eye() {
        let f = make_frustum();
        let positions = vec![0.0f32, 0.0, 5.0]; // behind (+Z)
        let mask = ss_frustum_cull_points(&f, &positions, 1);
        assert!(!mask[0], "point behind eye should not be visible");
    }

    #[test]
    fn test_cull_points_too_far() {
        let f = make_frustum();
        let positions = vec![0.0f32, 0.0, -1001.0]; // beyond far=1000
        let mask = ss_frustum_cull_points(&f, &positions, 1);
        assert!(!mask[0], "point beyond far plane should not be visible");
    }

    #[test]
    fn test_cull_points_beyond_horizontal_fov() {
        let f = make_frustum();
        // At depth 10, horizontal extent 100 (well beyond 45° half-FOV)
        let positions = vec![100.0f32, 0.0, -10.0];
        let mask = ss_frustum_cull_points(&f, &positions, 1);
        assert!(
            !mask[0],
            "point to the side beyond FOV should not be visible"
        );
    }

    #[test]
    fn test_cull_points_within_fov() {
        let f = make_frustum();
        // At depth 10, horizontal extent 5 (well within 45° half-FOV)
        let positions = vec![5.0f32, 0.0, -10.0];
        let mask = ss_frustum_cull_points(&f, &positions, 1);
        assert!(mask[0], "point within FOV should be visible");
    }

    #[test]
    fn test_cull_points_multiple() {
        let f = make_frustum();
        let positions = vec![
            0.0, 0.0, -5.0, // visible
            0.0, 0.0, 5.0, // behind
        ];
        let mask = ss_frustum_cull_points(&f, &positions, 2);
        assert!(mask[0]);
        assert!(!mask[1]);
    }

    // ---- ss_frustum_cull_spheres ----

    #[test]
    fn test_cull_spheres_overlap_boundary() {
        let f = make_frustum();
        // Sphere centred just outside the far plane but overlapping it
        let positions = vec![0.0f32, 0.0, -1001.0];
        let radii = vec![5.0f32]; // radius 5 reaches the far plane at 1000
        let mask = ss_frustum_cull_spheres(&f, &positions, &radii, 1);
        assert!(mask[0], "sphere overlapping far boundary should be visible");
    }

    #[test]
    fn test_cull_spheres_fully_outside() {
        let f = make_frustum();
        let positions = vec![0.0f32, 0.0, -2000.0];
        let radii = vec![1.0f32];
        let mask = ss_frustum_cull_spheres(&f, &positions, &radii, 1);
        assert!(!mask[0]);
    }

    #[test]
    fn test_cull_spheres_fully_inside() {
        let f = make_frustum();
        let positions = vec![0.0f32, 0.0, -10.0];
        let radii = vec![0.5f32];
        let mask = ss_frustum_cull_spheres(&f, &positions, &radii, 1);
        assert!(mask[0]);
    }

    // ---- ss_compute_scene_bounds ----

    #[test]
    fn test_bounds_single_point() {
        let positions = vec![3.0f32, -1.0, 5.0];
        let (mn, mx) = ss_compute_scene_bounds(&positions, 1);
        assert_eq!(mn, [3.0, -1.0, 5.0]);
        assert_eq!(mx, [3.0, -1.0, 5.0]);
    }

    #[test]
    fn test_bounds_symmetric() {
        let positions = make_positions_8();
        let (mn, mx) = ss_compute_scene_bounds(&positions, 8);
        assert!((mn[0] + 0.5).abs() < 1e-6);
        assert!((mx[0] - 0.5).abs() < 1e-6);
        assert!((mn[1] + 0.5).abs() < 1e-6);
        assert!((mx[1] - 0.5).abs() < 1e-6);
        assert!((mn[2] + 0.5).abs() < 1e-6);
        assert!((mx[2] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_bounds_empty() {
        let (mn, mx) = ss_compute_scene_bounds(&[], 0);
        assert_eq!(mn, [0.0; 3]);
        assert_eq!(mx, [0.0; 3]);
    }

    // ---- ss_chunk_scene ----

    #[test]
    fn test_chunk_scene_2x2x2() {
        let positions = make_positions_8();
        let chunks = ss_chunk_scene(&positions, 8, [2, 2, 2]).expect("chunk failed");
        // Should produce 8 chunks (2×2×2), each with 1 Gaussian
        assert_eq!(chunks.len(), 8);
        let total: usize = chunks.iter().map(|c| c.n_gaussians).sum();
        assert_eq!(total, 8);
    }

    #[test]
    fn test_chunk_scene_all_assigned() {
        let positions = make_positions_8();
        let chunks = ss_chunk_scene(&positions, 8, [2, 2, 2]).expect("chunk failed");
        let mut all_indices: Vec<usize> = chunks
            .iter()
            .flat_map(|c| c.gaussian_indices.iter().copied())
            .collect();
        all_indices.sort_unstable();
        assert_eq!(all_indices, (0..8).collect::<Vec<_>>());
    }

    #[test]
    fn test_chunk_scene_empty_error() {
        let err = ss_chunk_scene(&[], 0, [2, 2, 2]);
        assert!(matches!(err, Err(StreamingError::EmptyScene)));
    }

    #[test]
    fn test_chunk_scene_zero_division_error() {
        let positions = make_positions_8();
        let err = ss_chunk_scene(&positions, 8, [0, 2, 2]);
        assert!(matches!(err, Err(StreamingError::InvalidChunkSize { .. })));
    }

    // ---- ss_chunk_id ----

    #[test]
    fn test_chunk_id_origin() {
        assert_eq!(ss_chunk_id(0, 0, 0, 4, 4), 0);
    }

    #[test]
    fn test_chunk_id_different_cells() {
        let id_000 = ss_chunk_id(0, 0, 0, 4, 4);
        let id_100 = ss_chunk_id(1, 0, 0, 4, 4);
        let id_010 = ss_chunk_id(0, 1, 0, 4, 4);
        let id_001 = ss_chunk_id(0, 0, 1, 4, 4);
        assert_ne!(id_000, id_100);
        assert_ne!(id_000, id_010);
        assert_ne!(id_000, id_001);
        assert_ne!(id_100, id_010);
        assert_ne!(id_100, id_001);
        assert_ne!(id_010, id_001);
    }

    // ---- ss_chunk_bounds ----

    #[test]
    fn test_chunk_bounds_nested() {
        let bmin = [0.0f32; 3];
        let bmax = [4.0f32; 3];
        let (cmin, cmax) = ss_chunk_bounds(
            bmin,
            bmax,
            ChunkCellIndex {
                ix: 1,
                iy: 1,
                iz: 1,
            },
            ChunkGridDims {
                nx: 4,
                ny: 4,
                nz: 4,
            },
        );
        // Each cell is [1,2] in each dimension
        for d in 0..3 {
            assert!((cmin[d] - 1.0).abs() < 1e-5, "cmin[{d}]={}", cmin[d]);
            assert!((cmax[d] - 2.0).abs() < 1e-5, "cmax[{d}]={}", cmax[d]);
            assert!(cmin[d] >= bmin[d]);
            assert!(cmax[d] <= bmax[d]);
        }
    }

    #[test]
    fn test_chunk_bounds_first_cell() {
        let bmin = [0.0f32; 3];
        let bmax = [2.0f32; 3];
        let (cmin, cmax) = ss_chunk_bounds(
            bmin,
            bmax,
            ChunkCellIndex {
                ix: 0,
                iy: 0,
                iz: 0,
            },
            ChunkGridDims {
                nx: 2,
                ny: 2,
                nz: 2,
            },
        );
        for d in 0..3 {
            assert!((cmin[d] - 0.0).abs() < 1e-5);
            assert!((cmax[d] - 1.0).abs() < 1e-5);
        }
    }

    // ---- ss_distance_to_aabb ----

    #[test]
    fn test_distance_aabb_inside() {
        let dist = ss_distance_to_aabb([0.5, 0.5, 0.5], [0.0; 3], [1.0; 3]);
        assert!((dist - 0.0).abs() < 1e-6, "point inside → dist 0");
    }

    #[test]
    fn test_distance_aabb_outside() {
        let dist = ss_distance_to_aabb([2.0, 0.5, 0.5], [0.0; 3], [1.0; 3]);
        assert!(dist > 0.0);
        assert!((dist - 1.0).abs() < 1e-6, "one unit outside");
    }

    #[test]
    fn test_distance_aabb_at_corner() {
        let dist = ss_distance_to_aabb([2.0, 2.0, 2.0], [0.0; 3], [1.0; 3]);
        let expected = (3.0f32).sqrt(); // sqrt(1²+1²+1²)
        assert!((dist - expected).abs() < 1e-5);
    }

    // ---- ss_compute_priority ----

    #[test]
    fn test_priority_close_chunk() {
        let frustum = make_frustum();
        let chunk = StreamingChunk {
            id: 0,
            bounds_min: [-0.1; 3],
            bounds_max: [0.1; 3],
            gaussian_indices: vec![],
            n_gaussians: 0,
            priority: LoadPriority::Low,
            lod_level: 0,
            loaded: false,
            memory_bytes: 0,
        };
        let p = ss_compute_priority(&chunk, &frustum);
        // Distance is basically 0, near=0.1, so dist < near*2=0.2 → Critical
        assert!(
            p >= LoadPriority::High,
            "close chunk should be High or Critical, got {p:?}"
        );
    }

    #[test]
    fn test_priority_far_chunk() {
        let frustum = make_frustum();
        let chunk = StreamingChunk {
            id: 1,
            bounds_min: [500.0, 500.0, 500.0],
            bounds_max: [600.0, 600.0, 600.0],
            gaussian_indices: vec![],
            n_gaussians: 0,
            priority: LoadPriority::Low,
            lod_level: 0,
            loaded: false,
            memory_bytes: 0,
        };
        let p = ss_compute_priority(&chunk, &frustum);
        assert_eq!(p, LoadPriority::Low, "very far chunk should be Low");
    }

    // ---- ss_sort_chunks_by_priority ----

    #[test]
    fn test_sort_chunks_priority_descending() {
        let frustum = make_frustum();
        let mut scene = make_scene();
        ss_sort_chunks_by_priority(&mut scene.chunks, &frustum);
        if scene.chunks.len() >= 2 {
            let first = scene.chunks[0].priority;
            let last = scene.chunks[scene.chunks.len() - 1].priority;
            assert!(
                first >= last,
                "first chunk priority={first:?} should be ≥ last={last:?}"
            );
        }
    }

    // ---- ss_select_lod ----

    #[test]
    fn test_select_lod_level0() {
        let thresholds = [5.0f32, 20.0, 50.0];
        assert_eq!(ss_select_lod(1.0, 1000, 100, &thresholds), 0);
        assert_eq!(ss_select_lod(5.0, 1000, 100, &thresholds), 0);
    }

    #[test]
    fn test_select_lod_level1() {
        let thresholds = [5.0f32, 20.0, 50.0];
        assert_eq!(ss_select_lod(10.0, 1000, 100, &thresholds), 1);
    }

    #[test]
    fn test_select_lod_level2() {
        let thresholds = [5.0f32, 20.0, 50.0];
        assert_eq!(ss_select_lod(60.0, 1000, 100, &thresholds), 2);
        assert_eq!(ss_select_lod(100.0, 1000, 100, &thresholds), 2);
    }

    // ---- ss_lod_subsample_indices ----

    #[test]
    fn test_lod_subsample_lod0_returns_all() {
        let indices: Vec<usize> = (0..20).collect();
        let result = ss_lod_subsample_indices(&indices, 0, 42);
        assert_eq!(result, indices);
    }

    #[test]
    fn test_lod_subsample_lod1_returns_half() {
        let indices: Vec<usize> = (0..20).collect();
        let result = ss_lod_subsample_indices(&indices, 1, 42);
        assert_eq!(result.len(), 10);
        // Should be even indices
        for (i, &idx) in result.iter().enumerate() {
            assert_eq!(idx, i * 2);
        }
    }

    #[test]
    fn test_lod_subsample_lod2_returns_quarter() {
        let indices: Vec<usize> = (0..20).collect();
        let result = ss_lod_subsample_indices(&indices, 2, 42);
        assert_eq!(result.len(), 5);
    }

    #[test]
    fn test_lod_subsample_empty() {
        let result = ss_lod_subsample_indices(&[], 1, 0);
        assert!(result.is_empty());
    }

    // ---- ChunkCache ----

    #[test]
    fn test_cache_new_starts_empty() {
        let cache = ChunkCache::new(1024);
        assert_eq!(cache.used_bytes, 0);
        assert!(cache.chunks.is_empty());
    }

    #[test]
    fn test_cache_can_fit() {
        let cache = ChunkCache::new(1024);
        assert!(cache.can_fit(512));
        assert!(cache.can_fit(1024));
        assert!(!cache.can_fit(1025));
    }

    #[test]
    fn test_cache_insert_basic() {
        let mut cache = ChunkCache::new(1024);
        cache.insert(1, 256).expect("insert failed");
        assert_eq!(cache.used_bytes, 256);
        assert_eq!(cache.chunks.len(), 1);
    }

    #[test]
    fn test_cache_insert_evicts_lru() {
        let mut cache = ChunkCache::new(512);
        cache.insert(1, 300).expect("insert 1");
        cache.insert(2, 300).expect("insert 2 should evict 1");
        // After eviction of chunk 1, only chunk 2 should remain
        assert!(
            cache.chunks.iter().all(|(id, _)| *id != 1),
            "chunk 1 should be evicted"
        );
        assert!(cache.chunks.iter().any(|(id, _)| *id == 2));
    }

    #[test]
    fn test_cache_touch_updates_access() {
        let mut cache = ChunkCache::new(1024);
        cache.insert(1, 100).expect("insert");
        let old_step = cache.chunks[0].1;
        let found = cache.touch(1);
        assert!(found);
        assert!(cache.chunks[0].1 > old_step, "access step should increase");
    }

    #[test]
    fn test_cache_touch_not_found() {
        let mut cache = ChunkCache::new(1024);
        let found = cache.touch(999);
        assert!(!found);
    }

    #[test]
    fn test_cache_evict_lru_removes_oldest() {
        let mut cache = ChunkCache::new(1024);
        cache.insert(10, 100).expect("insert 10");
        cache.insert(20, 100).expect("insert 20");
        cache.touch(20); // touch 20, making 10 the LRU
        let evicted = cache.evict_lru();
        assert_eq!(evicted, Some(10), "oldest accessed should be evicted first");
    }

    #[test]
    fn test_cache_evict_all() {
        let mut cache = ChunkCache::new(1024);
        cache.insert(1, 200).expect("insert 1");
        cache.insert(2, 200).expect("insert 2");
        cache.evict_all();
        assert_eq!(cache.used_bytes, 0);
        assert!(cache.chunks.is_empty());
    }

    #[test]
    fn test_cache_utilization_empty() {
        let cache = ChunkCache::new(1024);
        assert!((cache.utilization() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_cache_utilization_full() {
        let mut cache = ChunkCache::new(1024);
        cache.insert(1, 1024).expect("insert");
        assert!((cache.utilization() - 1.0).abs() < 1e-6);
    }

    // ---- StreamingScene ----

    #[test]
    fn test_streaming_scene_new_chunk_count() {
        let scene = make_scene();
        assert_eq!(scene.chunks.len(), 8); // 2×2×2
    }

    #[test]
    fn test_streaming_scene_new_empty_error() {
        let config = StreamingConfig::default();
        let err = StreamingScene::new(config, &[], 0, 9);
        assert!(matches!(err, Err(StreamingError::EmptyScene)));
    }

    #[test]
    fn test_streaming_scene_update_view_returns_ids() {
        let mut scene = make_scene();
        let frustum = make_frustum();
        let to_load = scene.update_view(&frustum);
        // Should return at most max_chunks_per_frame
        assert!(to_load.len() <= scene.config.max_chunks_per_frame);
    }

    #[test]
    fn test_streaming_scene_mark_loaded() {
        let mut scene = make_scene();
        let frustum = make_frustum();
        let to_load = scene.update_view(&frustum);
        if !to_load.is_empty() {
            let id = to_load[0];
            scene.mark_loaded(id).expect("mark_loaded failed");
            let chunk = scene
                .chunks
                .iter()
                .find(|c| c.id == id)
                .expect("chunk not found");
            assert!(chunk.loaded);
        }
    }

    #[test]
    fn test_streaming_scene_mark_loaded_unknown_id() {
        let mut scene = make_scene();
        let err = scene.mark_loaded(u64::MAX);
        assert!(matches!(err, Err(StreamingError::ChunkNotFound { .. })));
    }

    #[test]
    fn test_streaming_scene_get_visible_chunks() {
        let scene = make_scene();
        let frustum = make_frustum();
        let visible = scene.get_visible_chunks(&frustum);
        // At least the chunks whose centroids are ahead of the eye should appear
        assert!(!visible.is_empty() || scene.chunks.iter().all(|c| c.n_gaussians == 0));
    }

    #[test]
    fn test_streaming_scene_get_loaded_empty() {
        let scene = make_scene();
        let indices = scene.get_loaded_gaussian_indices();
        assert!(indices.is_empty(), "nothing loaded yet");
    }

    // ---- ss_compute_stats ----

    #[test]
    fn test_compute_stats_loading_ratio_in_range() {
        let scene = make_scene();
        let frustum = make_frustum();
        let stats = ss_compute_stats(&scene, &frustum);
        assert!(stats.loading_ratio >= 0.0 && stats.loading_ratio <= 1.0);
    }

    #[test]
    fn test_compute_stats_total_chunks() {
        let scene = make_scene();
        let frustum = make_frustum();
        let stats = ss_compute_stats(&scene, &frustum);
        assert_eq!(stats.total_chunks, scene.chunks.len());
    }

    #[test]
    fn test_compute_stats_total_gaussians() {
        let scene = make_scene();
        let frustum = make_frustum();
        let stats = ss_compute_stats(&scene, &frustum);
        assert_eq!(stats.total_gaussians, 8);
    }

    #[test]
    fn test_compute_stats_after_load() {
        let mut scene = make_scene();
        let frustum = make_frustum();
        let to_load = scene.update_view(&frustum);
        for id in &to_load {
            let _ = scene.mark_loaded(*id);
        }
        let stats = ss_compute_stats(&scene, &frustum);
        assert_eq!(stats.loaded_chunks, to_load.len());
        assert!(stats.loading_ratio > 0.0);
    }

    // ---- ss_format_stats / ss_format_config ----

    #[test]
    fn test_format_stats_non_empty() {
        let scene = make_scene();
        let frustum = make_frustum();
        let stats = ss_compute_stats(&scene, &frustum);
        let s = ss_format_stats(&stats);
        assert!(!s.is_empty());
        assert!(s.contains("StreamingStats"));
    }

    #[test]
    fn test_format_config_non_empty() {
        let config = StreamingConfig::default();
        let s = ss_format_config(&config);
        assert!(!s.is_empty());
        assert!(s.contains("StreamingConfig"));
    }

    // ---- Error variants ----

    #[test]
    fn test_error_empty_scene() {
        let err = StreamingError::EmptyScene;
        let msg = err.to_string();
        assert!(msg.contains("empty") || msg.contains("Scene"));
    }

    #[test]
    fn test_error_invalid_chunk_size() {
        let err = StreamingError::InvalidChunkSize { size: 0 };
        let msg = err.to_string();
        assert!(msg.contains("0"));
    }

    #[test]
    fn test_error_memory_budget_exceeded() {
        let err = StreamingError::MemoryBudgetExceeded {
            used: 1024,
            budget: 512,
        };
        let msg = err.to_string();
        assert!(msg.contains("1024") && msg.contains("512"));
    }

    #[test]
    fn test_error_chunk_not_found() {
        let err = StreamingError::ChunkNotFound { id: 42 };
        let msg = err.to_string();
        assert!(msg.contains("42"));
    }

    #[test]
    fn test_error_invalid_frustum() {
        let err = StreamingError::InvalidFrustum {
            near: 10.0,
            far: 5.0,
        };
        let msg = err.to_string();
        assert!(msg.contains("10") && msg.contains("5"));
    }

    #[test]
    fn test_error_length_mismatch() {
        let err = StreamingError::LengthMismatch {
            pos: 6,
            expected: 9,
        };
        let msg = err.to_string();
        assert!(msg.contains("6") && msg.contains("9"));
    }

    // ---- Memory budget: cache can't exceed budget ----

    #[test]
    fn test_cache_cannot_exceed_budget() {
        let mut cache = ChunkCache::new(1000);
        // Insert a chunk larger than the budget → error
        let result = cache.insert(1, 2000);
        assert!(matches!(
            result,
            Err(StreamingError::MemoryBudgetExceeded { .. })
        ));
    }

    // ---- Round-trip: chunk + get_loaded gives all indices ----

    #[test]
    fn test_round_trip_chunk_and_load() {
        let positions = make_positions_8();
        let config = StreamingConfig {
            memory_budget_bytes: 256 * 1024 * 1024,
            chunk_divisions: [2, 2, 2],
            lod_distances: [5.0, 20.0, 50.0],
            max_chunks_per_frame: 32, // allow loading all chunks
            preload_radius: 1000.0,   // large preload radius
        };
        let mut scene = StreamingScene::new(config, &positions, 8, 9).expect("scene creation");

        // Load all chunks
        let chunk_ids: Vec<u64> = scene.chunks.iter().map(|c| c.id).collect();
        for id in chunk_ids {
            let _ = scene.mark_loaded(id);
        }

        let loaded = scene.get_loaded_gaussian_indices();
        let mut sorted = loaded.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(
            sorted,
            (0..8usize).collect::<Vec<_>>(),
            "all 8 Gaussians should be reachable"
        );
    }

    // ---- xorshift64 internal test ----

    #[test]
    fn test_xorshift64_nonzero() {
        let mut state = 1u64;
        for _ in 0..1000 {
            let v = xorshift64(&mut state);
            assert_ne!(v, 0);
        }
    }

    // ---- Additional coverage ----

    #[test]
    fn test_update_view_increments_frame_counter() {
        let mut scene = make_scene();
        let frustum = make_frustum();
        assert_eq!(scene.frame_counter, 0);
        scene.update_view(&frustum);
        assert_eq!(scene.frame_counter, 1);
        scene.update_view(&frustum);
        assert_eq!(scene.frame_counter, 2);
    }

    #[test]
    fn test_chunk_scene_larger_grid() {
        let mut positions = Vec::new();
        for i in 0..64 {
            positions.push((i % 8) as f32);
            positions.push(((i / 8) % 8) as f32);
            positions.push((i / 64) as f32);
        }
        let chunks = ss_chunk_scene(&positions, 64, [4, 4, 4]).expect("chunk failed");
        let total: usize = chunks.iter().map(|c| c.n_gaussians).sum();
        assert_eq!(total, 64);
    }

    #[test]
    fn test_priority_ordering() {
        assert!(LoadPriority::Critical > LoadPriority::High);
        assert!(LoadPriority::High > LoadPriority::Medium);
        assert!(LoadPriority::Medium > LoadPriority::Low);
    }

    #[test]
    fn test_select_lod_exact_threshold() {
        // Exactly at threshold 0 → LOD 0
        let thresholds = [10.0f32, 30.0, 60.0];
        assert_eq!(ss_select_lod(10.0, 100, 50, &thresholds), 0);
        // Exactly at threshold 2 → LOD 2
        assert_eq!(ss_select_lod(60.0, 100, 50, &thresholds), 2);
    }
}
