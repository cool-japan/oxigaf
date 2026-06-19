//! Dense per-vertex 3D displacement (warp) fields for non-rigid mesh deformation.
//!
//! A warp field stores a 3D displacement vector for every vertex of a mesh.
//! It augments the linear blend skinning output of FLAME with fine-grained
//! detail such as wrinkles, identity asymmetry, or residual deformations from
//! a neural network.
//!
//! ## Core types
//!
//! | Type | Purpose |
//! |------|---------|
//! | [`WarpField`] | Dense per-vertex displacement field |
//! | [`WarpMask`] | Per-vertex weight mask for selective warping |
//! | [`WarpFieldStats`] | Statistical summary of a field |
//! | [`WarpFieldSequence`] | Temporal sequence of fields for animation |
//!
//! ## Free functions
//!
//! - [`linear_combination`] — weighted sum of basis fields
//! - [`build_vertex_adjacency`] — adjacency list from face indices
//! - [`laplacian_smooth_warp_field`] — Laplacian smoothing for displacement fields
//! - [`compute_warp_stats`] — statistical analysis
//! - [`per_vertex_magnitude`] — per-vertex displacement magnitudes
//! - [`find_large_displacements`] — threshold-based outlier detection

use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors produced by warp-field operations.
#[derive(Debug, thiserror::Error)]
pub enum WarpFieldError {
    /// Buffer length did not match the expected size.
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    /// A configuration parameter was invalid.
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),

    /// Attempted an operation on an empty or uninitialised field.
    #[error("warp field is empty")]
    EmptyField,

    /// Weight vector contains invalid values (NaN, negative, etc.).
    #[error("invalid weights: {0}")]
    InvalidWeights(String),
}

// ---------------------------------------------------------------------------
// WarpField
// ---------------------------------------------------------------------------

/// A dense per-vertex 3D displacement field.
///
/// Displacements are stored as interleaved `[dx, dy, dz]` triples:
/// `displacements[3*i..3*i+3]` is the displacement for vertex `i`.
#[derive(Debug, Clone)]
pub struct WarpField {
    /// Interleaved `[dx, dy, dz]` displacements.  Length = `num_vertices * 3`.
    pub displacements: Vec<f32>,
    /// Number of vertices described by this field.
    pub num_vertices: usize,
}

impl WarpField {
    /// Create a zero-displacement field for `num_vertices` vertices.
    #[must_use]
    pub fn new(num_vertices: usize) -> Self {
        Self {
            displacements: vec![0.0_f32; num_vertices * 3],
            num_vertices,
        }
    }

    /// Build a [`WarpField`] from an existing displacement buffer.
    ///
    /// # Errors
    /// Returns [`WarpFieldError::DimensionMismatch`] if
    /// `displacements.len()` is not divisible by 3.
    pub fn from_displacements(displacements: Vec<f32>) -> Result<Self, WarpFieldError> {
        let len = displacements.len();
        if !len.is_multiple_of(3) {
            return Err(WarpFieldError::DimensionMismatch {
                expected: len - (len % 3),
                got: len,
            });
        }
        let num_vertices = len / 3;
        Ok(Self {
            displacements,
            num_vertices,
        })
    }

    /// Return the `[dx, dy, dz]` displacement for vertex `i`.
    ///
    /// Returns `[0.0, 0.0, 0.0]` for out-of-range indices.
    #[must_use]
    pub fn get(&self, i: usize) -> [f32; 3] {
        if i >= self.num_vertices {
            return [0.0, 0.0, 0.0];
        }
        let base = i * 3;
        [
            self.displacements[base],
            self.displacements[base + 1],
            self.displacements[base + 2],
        ]
    }

    /// Overwrite the displacement for vertex `i`.
    ///
    /// # Errors
    /// Returns [`WarpFieldError::DimensionMismatch`] when `i >= num_vertices`.
    pub fn set(&mut self, i: usize, d: [f32; 3]) -> Result<(), WarpFieldError> {
        if i >= self.num_vertices {
            return Err(WarpFieldError::DimensionMismatch {
                expected: self.num_vertices,
                got: i + 1,
            });
        }
        let base = i * 3;
        self.displacements[base] = d[0];
        self.displacements[base + 1] = d[1];
        self.displacements[base + 2] = d[2];
        Ok(())
    }

    /// Maximum displacement magnitude across all vertices.
    #[must_use]
    pub fn max_magnitude(&self) -> f32 {
        per_vertex_magnitude(self)
            .into_iter()
            .fold(0.0_f32, f32::max)
    }

    /// Mean displacement magnitude across all vertices.
    ///
    /// Returns `0.0` for an empty field.
    #[must_use]
    pub fn mean_magnitude(&self) -> f32 {
        if self.num_vertices == 0 {
            return 0.0;
        }
        let magnitudes = per_vertex_magnitude(self);
        let sum: f32 = magnitudes.iter().sum();
        sum / self.num_vertices as f32
    }

    /// Root-mean-square displacement magnitude.
    ///
    /// Defined as `sqrt( mean( dx^2 + dy^2 + dz^2 ) )`.
    /// Returns `0.0` for an empty field.
    #[must_use]
    pub fn rms_magnitude(&self) -> f32 {
        if self.num_vertices == 0 {
            return 0.0;
        }
        let sum_sq: f32 = self
            .displacements
            .chunks_exact(3)
            .map(|c| c[0] * c[0] + c[1] * c[1] + c[2] * c[2])
            .sum();
        (sum_sq / self.num_vertices as f32).sqrt()
    }

    /// Return a new field with all displacements multiplied by `factor`.
    #[must_use]
    pub fn scale(&self, factor: f32) -> Self {
        let displacements = self.displacements.iter().map(|v| v * factor).collect();
        Self {
            displacements,
            num_vertices: self.num_vertices,
        }
    }

    /// Element-wise add another field's displacements to this one.
    ///
    /// # Errors
    /// Returns [`WarpFieldError::DimensionMismatch`] when vertex counts differ.
    pub fn add(&self, other: &WarpField) -> Result<Self, WarpFieldError> {
        if self.num_vertices != other.num_vertices {
            return Err(WarpFieldError::DimensionMismatch {
                expected: self.num_vertices,
                got: other.num_vertices,
            });
        }
        let displacements = self
            .displacements
            .iter()
            .zip(other.displacements.iter())
            .map(|(a, b)| a + b)
            .collect();
        Ok(Self {
            displacements,
            num_vertices: self.num_vertices,
        })
    }

    /// Linearly interpolate between `self` (`t = 0`) and `other` (`t = 1`).
    ///
    /// `result = (1 - t) * self + t * other`
    ///
    /// # Errors
    /// Returns [`WarpFieldError::DimensionMismatch`] when vertex counts differ.
    pub fn lerp(&self, other: &WarpField, t: f32) -> Result<Self, WarpFieldError> {
        if self.num_vertices != other.num_vertices {
            return Err(WarpFieldError::DimensionMismatch {
                expected: self.num_vertices,
                got: other.num_vertices,
            });
        }
        let one_minus_t = 1.0 - t;
        let displacements = self
            .displacements
            .iter()
            .zip(other.displacements.iter())
            .map(|(a, b)| one_minus_t * a + t * b)
            .collect();
        Ok(Self {
            displacements,
            num_vertices: self.num_vertices,
        })
    }

    /// Add this displacement field to a vertex buffer and return the result.
    ///
    /// `vertices` must be an interleaved `[x, y, z]` buffer with exactly
    /// `num_vertices * 3` elements.
    ///
    /// # Errors
    /// Returns [`WarpFieldError::DimensionMismatch`] when sizes differ.
    pub fn apply_to_vertices(&self, vertices: &[f32]) -> Result<Vec<f32>, WarpFieldError> {
        let expected = self.num_vertices * 3;
        if vertices.len() != expected {
            return Err(WarpFieldError::DimensionMismatch {
                expected,
                got: vertices.len(),
            });
        }
        let result = vertices
            .iter()
            .zip(self.displacements.iter())
            .map(|(v, d)| v + d)
            .collect();
        Ok(result)
    }

    /// Set all displacements to zero in-place.
    pub fn zero_out(&mut self) {
        for v in &mut self.displacements {
            *v = 0.0;
        }
    }

    /// Clamp per-vertex displacement magnitudes to `max_displacement` in-place.
    ///
    /// Vertices whose magnitude exceeds the threshold are scaled down so that
    /// the direction is preserved but the magnitude equals `max_displacement`.
    pub fn clamp_magnitudes(&mut self, max_displacement: f32) {
        for chunk in self.displacements.chunks_exact_mut(3) {
            let mag = (chunk[0] * chunk[0] + chunk[1] * chunk[1] + chunk[2] * chunk[2]).sqrt();
            if mag > max_displacement && mag > 0.0 {
                let scale = max_displacement / mag;
                chunk[0] *= scale;
                chunk[1] *= scale;
                chunk[2] *= scale;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Warp field construction functions
// ---------------------------------------------------------------------------

/// Compute a weighted linear combination of basis warp fields.
///
/// `result = Σ weight_i * field_i`
///
/// # Errors
/// - [`WarpFieldError::EmptyField`] if `basis` is empty.
/// - [`WarpFieldError::DimensionMismatch`] if any field has a different
///   `num_vertices` than the first field.
pub fn linear_combination(basis: &[(WarpField, f32)]) -> Result<WarpField, WarpFieldError> {
    if basis.is_empty() {
        return Err(WarpFieldError::EmptyField);
    }
    let num_vertices = basis[0].0.num_vertices;
    let mut result = WarpField::new(num_vertices);

    for (field, weight) in basis {
        if field.num_vertices != num_vertices {
            return Err(WarpFieldError::DimensionMismatch {
                expected: num_vertices,
                got: field.num_vertices,
            });
        }
        for (r, d) in result
            .displacements
            .iter_mut()
            .zip(field.displacements.iter())
        {
            *r += weight * d;
        }
    }
    Ok(result)
}

/// Build a per-vertex adjacency list from a flat face index buffer.
///
/// `faces` must be a flat `[i0, i1, i2, ...]` buffer where every triplet
/// describes a triangle.  Edges are treated as undirected; each neighbor list
/// is sorted and deduplicated.
///
/// # Errors
/// Returns [`WarpFieldError::DimensionMismatch`] if `faces.len()` is not
/// divisible by 3.
pub fn build_vertex_adjacency(
    num_vertices: usize,
    faces: &[u32],
) -> Result<Vec<Vec<usize>>, WarpFieldError> {
    if !faces.len().is_multiple_of(3) {
        return Err(WarpFieldError::DimensionMismatch {
            expected: faces.len() - (faces.len() % 3),
            got: faces.len(),
        });
    }

    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); num_vertices];

    for tri in faces.chunks_exact(3) {
        let [a, b, c] = [tri[0] as usize, tri[1] as usize, tri[2] as usize];
        // Undirected: add each vertex to the others' neighbor lists.
        for (src, dst) in [(a, b), (a, c), (b, a), (b, c), (c, a), (c, b)] {
            if src < num_vertices && dst < num_vertices {
                adj[src].push(dst);
            }
        }
    }

    for list in &mut adj {
        list.sort_unstable();
        list.dedup();
    }

    Ok(adj)
}

/// Apply one or more iterations of Laplacian smoothing to a warp field.
///
/// For each iteration the update rule is:
///
/// ```text
/// new_d[v] = d[v] + λ * (mean(neighbors' d) - d[v])
/// ```
///
/// `lambda = 0` leaves the field unchanged; `lambda = 1` replaces each
/// vertex with the mean of its neighbours (full Laplacian step).
///
/// Zero-iteration calls return a clone of the input unchanged.
///
/// # Errors
/// Returns [`WarpFieldError::DimensionMismatch`] if
/// `adjacency.len() != field.num_vertices`.
pub fn laplacian_smooth_warp_field(
    field: &WarpField,
    adjacency: &[Vec<usize>],
    iterations: usize,
    lambda: f32,
) -> Result<WarpField, WarpFieldError> {
    if adjacency.len() != field.num_vertices {
        return Err(WarpFieldError::DimensionMismatch {
            expected: field.num_vertices,
            got: adjacency.len(),
        });
    }

    if iterations == 0 {
        return Ok(field.clone());
    }

    let mut current = field.displacements.clone();
    let n = field.num_vertices;

    for _ in 0..iterations {
        let mut next = current.clone();
        for (vert_idx, neighbors) in adjacency.iter().enumerate().take(n) {
            if neighbors.is_empty() {
                continue;
            }
            let v = vert_idx;
            // Compute mean of neighbors.
            let mut mean = [0.0_f32; 3];
            for &nb in neighbors {
                let base = nb * 3;
                mean[0] += current[base];
                mean[1] += current[base + 1];
                mean[2] += current[base + 2];
            }
            let count = neighbors.len() as f32;
            mean[0] /= count;
            mean[1] /= count;
            mean[2] /= count;

            // Blended update: d[v] + lambda * (mean - d[v])
            let base = v * 3;
            next[base] = current[base] + lambda * (mean[0] - current[base]);
            next[base + 1] = current[base + 1] + lambda * (mean[1] - current[base + 1]);
            next[base + 2] = current[base + 2] + lambda * (mean[2] - current[base + 2]);
        }
        current = next;
    }

    WarpField::from_displacements(current)
}

// ---------------------------------------------------------------------------
// WarpMask
// ---------------------------------------------------------------------------

/// Per-vertex weight mask for selective warp-field application.
///
/// Each weight lives in `[0, 1]`.  A weight of `0` means no displacement is
/// applied to that vertex; `1` means full displacement.
#[derive(Debug, Clone)]
pub struct WarpMask {
    /// Per-vertex weights, all in `[0, 1]`.
    pub weights: Vec<f32>,
}

impl WarpMask {
    /// Create a mask with all weights set to `1.0` (full warp everywhere).
    #[must_use]
    pub fn full(num_vertices: usize) -> Self {
        Self {
            weights: vec![1.0_f32; num_vertices],
        }
    }

    /// Create a mask with all weights set to `0.0` (no warp).
    #[must_use]
    pub fn zero(num_vertices: usize) -> Self {
        Self {
            weights: vec![0.0_f32; num_vertices],
        }
    }

    /// Create a mask from a boolean slice: `true → 1.0`, `false → 0.0`.
    #[must_use]
    pub fn from_bool(mask: &[bool]) -> Self {
        let weights = mask
            .iter()
            .map(|&b| if b { 1.0_f32 } else { 0.0_f32 })
            .collect();
        Self { weights }
    }

    /// Build a soft mask using BFS hop distance from seed vertices.
    ///
    /// The weight for vertex `v` is:
    ///
    /// ```text
    /// weight[v] = exp( -hop_distance[v]^2 / (2 * sigma_hops^2) )
    /// ```
    ///
    /// Seed vertices have hop distance `0` and therefore weight `1.0`.
    /// Vertices not reachable from any seed receive weight `0.0`.
    #[must_use]
    pub fn from_geodesic_distance(
        adjacency: &[Vec<usize>],
        center_vertices: &[usize],
        sigma_hops: f32,
    ) -> Self {
        let n = adjacency.len();
        let two_sigma_sq = 2.0 * sigma_hops * sigma_hops;
        let mut hop_dist = vec![u32::MAX; n];
        let mut queue: VecDeque<usize> = VecDeque::new();

        for &seed in center_vertices {
            if seed < n && hop_dist[seed] == u32::MAX {
                hop_dist[seed] = 0;
                queue.push_back(seed);
            }
        }

        // BFS
        while let Some(v) = queue.pop_front() {
            let d = hop_dist[v];
            for &nb in &adjacency[v] {
                if hop_dist[nb] == u32::MAX {
                    hop_dist[nb] = d + 1;
                    queue.push_back(nb);
                }
            }
        }

        let weights = hop_dist
            .iter()
            .map(|&d| {
                if d == u32::MAX {
                    0.0_f32
                } else {
                    let d_f = d as f32;
                    let w = (-d_f * d_f / two_sigma_sq).exp();
                    w.clamp(0.0, 1.0)
                }
            })
            .collect();

        Self { weights }
    }

    /// Return the number of vertices described by this mask.
    #[must_use]
    pub fn num_vertices(&self) -> usize {
        self.weights.len()
    }

    /// Apply this mask to a warp field.
    ///
    /// Each vertex displacement is scaled by its mask weight:
    /// `new_disp[v] = field.disp[v] * weights[v]`.
    ///
    /// # Errors
    /// Returns [`WarpFieldError::DimensionMismatch`] when sizes differ.
    pub fn apply_to_field(&self, field: &WarpField) -> Result<WarpField, WarpFieldError> {
        if self.weights.len() != field.num_vertices {
            return Err(WarpFieldError::DimensionMismatch {
                expected: field.num_vertices,
                got: self.weights.len(),
            });
        }
        let displacements = field
            .displacements
            .chunks_exact(3)
            .zip(self.weights.iter())
            .flat_map(|(chunk, &w)| [chunk[0] * w, chunk[1] * w, chunk[2] * w])
            .collect();
        Ok(WarpField {
            displacements,
            num_vertices: field.num_vertices,
        })
    }

    /// Return a new mask with each weight replaced by `1.0 - weight`.
    #[must_use]
    pub fn invert(&self) -> Self {
        let weights = self.weights.iter().map(|&w| 1.0 - w).collect();
        Self { weights }
    }

    /// Element-wise multiply this mask with `other`.
    ///
    /// # Errors
    /// Returns [`WarpFieldError::DimensionMismatch`] when sizes differ.
    pub fn intersect(&self, other: &WarpMask) -> Result<WarpMask, WarpFieldError> {
        if self.weights.len() != other.weights.len() {
            return Err(WarpFieldError::DimensionMismatch {
                expected: self.weights.len(),
                got: other.weights.len(),
            });
        }
        let weights = self
            .weights
            .iter()
            .zip(other.weights.iter())
            .map(|(a, b)| a * b)
            .collect();
        Ok(WarpMask { weights })
    }
}

// ---------------------------------------------------------------------------
// Warp field analysis
// ---------------------------------------------------------------------------

/// Statistical summary of a [`WarpField`].
#[derive(Debug, Clone)]
pub struct WarpFieldStats {
    /// Number of vertices in the field.
    pub num_vertices: usize,
    /// Maximum per-vertex displacement magnitude.
    pub max_magnitude: f32,
    /// Mean per-vertex displacement magnitude.
    pub mean_magnitude: f32,
    /// Root-mean-square displacement magnitude.
    pub rms_magnitude: f32,
    /// Fraction of vertices with magnitude > `1e-6`.
    pub fraction_nonzero: f32,
    /// 50th-percentile displacement magnitude.
    pub p50_magnitude: f32,
    /// 95th-percentile displacement magnitude.
    pub p95_magnitude: f32,
}

/// Compute summary statistics for a [`WarpField`].
#[must_use]
pub fn compute_warp_stats(field: &WarpField) -> WarpFieldStats {
    let mut magnitudes = per_vertex_magnitude(field);
    let n = magnitudes.len();

    if n == 0 {
        return WarpFieldStats {
            num_vertices: 0,
            max_magnitude: 0.0,
            mean_magnitude: 0.0,
            rms_magnitude: 0.0,
            fraction_nonzero: 0.0,
            p50_magnitude: 0.0,
            p95_magnitude: 0.0,
        };
    }

    let max_magnitude = magnitudes.iter().copied().fold(0.0_f32, f32::max);
    let sum: f32 = magnitudes.iter().sum();
    let mean_magnitude = sum / n as f32;
    let rms_magnitude = field.rms_magnitude();

    let nonzero_count = magnitudes.iter().filter(|&&m| m > 1e-6).count();
    let fraction_nonzero = nonzero_count as f32 / n as f32;

    // Sort for percentiles.
    magnitudes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let p50_idx = ((n as f32 * 0.50) as usize).min(n - 1);
    let p95_idx = ((n as f32 * 0.95) as usize).min(n - 1);
    let p50_magnitude = magnitudes[p50_idx];
    let p95_magnitude = magnitudes[p95_idx];

    WarpFieldStats {
        num_vertices: n,
        max_magnitude,
        mean_magnitude,
        rms_magnitude,
        fraction_nonzero,
        p50_magnitude,
        p95_magnitude,
    }
}

/// Compute the per-vertex displacement magnitude vector.
///
/// Returns a `Vec<f32>` of length `num_vertices` where element `i` is
/// `sqrt(dx_i^2 + dy_i^2 + dz_i^2)`.
#[must_use]
pub fn per_vertex_magnitude(field: &WarpField) -> Vec<f32> {
    field
        .displacements
        .chunks_exact(3)
        .map(|c| (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt())
        .collect()
}

/// Find vertex indices whose displacement magnitude exceeds `threshold`.
///
/// Returns the indices in ascending order.
#[must_use]
pub fn find_large_displacements(field: &WarpField, threshold: f32) -> Vec<usize> {
    per_vertex_magnitude(field)
        .into_iter()
        .enumerate()
        .filter(|(_, mag)| *mag > threshold)
        .map(|(i, _)| i)
        .collect()
}

// ---------------------------------------------------------------------------
// WarpFieldSequence
// ---------------------------------------------------------------------------

/// A temporal sequence of warp fields enabling animated mesh deformation.
///
/// All fields in the sequence must share the same `num_vertices`.  The
/// sequence supports temporal sampling with continuous interpolation and
/// frame-to-frame velocity queries.
#[derive(Debug, Clone)]
pub struct WarpFieldSequence {
    /// Ordered frames.
    pub fields: Vec<WarpField>,
    /// Vertex count shared by every frame.
    pub num_vertices: usize,
}

impl WarpFieldSequence {
    /// Create an empty sequence for meshes with `num_vertices` vertices.
    #[must_use]
    pub fn new(num_vertices: usize) -> Self {
        Self {
            fields: Vec::new(),
            num_vertices,
        }
    }

    /// Append a frame to the sequence.
    ///
    /// # Errors
    /// Returns [`WarpFieldError::DimensionMismatch`] when
    /// `field.num_vertices != self.num_vertices`.
    pub fn push(&mut self, field: WarpField) -> Result<(), WarpFieldError> {
        if field.num_vertices != self.num_vertices {
            return Err(WarpFieldError::DimensionMismatch {
                expected: self.num_vertices,
                got: field.num_vertices,
            });
        }
        self.fields.push(field);
        Ok(())
    }

    /// Return the number of frames in the sequence.
    #[must_use]
    pub fn len(&self) -> usize {
        self.fields.len()
    }

    /// Return `true` if the sequence contains no frames.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }

    /// Sample the sequence at continuous time `t`.
    ///
    /// `t` is clamped to `[0, len - 1]`.  The sample is a linear
    /// interpolation between the floor and ceil frame indices using the
    /// fractional part of `t`.
    ///
    /// # Errors
    /// Returns [`WarpFieldError::EmptyField`] if the sequence has no frames.
    pub fn sample_at(&self, t: f32) -> Result<WarpField, WarpFieldError> {
        if self.fields.is_empty() {
            return Err(WarpFieldError::EmptyField);
        }
        let max_t = (self.fields.len() - 1) as f32;
        let t_clamped = t.clamp(0.0, max_t);

        let lo = t_clamped.floor() as usize;
        let hi = (lo + 1).min(self.fields.len() - 1);
        let frac = t_clamped - lo as f32;

        self.fields[lo].lerp(&self.fields[hi], frac)
    }

    /// Compute the frame-to-frame velocity field at frame `i`.
    ///
    /// Velocity is defined as `fields[i+1].displacements - fields[i].displacements`.
    ///
    /// # Errors
    /// - [`WarpFieldError::EmptyField`] if the sequence has fewer than 2 frames.
    /// - [`WarpFieldError::DimensionMismatch`] if `i + 1 >= len`.
    pub fn frame_velocity(&self, i: usize) -> Result<WarpField, WarpFieldError> {
        if self.fields.len() < 2 {
            return Err(WarpFieldError::EmptyField);
        }
        if i + 1 >= self.fields.len() {
            return Err(WarpFieldError::DimensionMismatch {
                expected: self.fields.len() - 1,
                got: i + 1,
            });
        }
        let a = &self.fields[i];
        let b = &self.fields[i + 1];
        let displacements = b
            .displacements
            .iter()
            .zip(a.displacements.iter())
            .map(|(bi, ai)| bi - ai)
            .collect();
        Ok(WarpField {
            displacements,
            num_vertices: self.num_vertices,
        })
    }

    /// Maximum displacement magnitude across all frames and all vertices.
    ///
    /// Returns `0.0` for empty sequences.
    #[must_use]
    pub fn global_max_magnitude(&self) -> f32 {
        self.fields
            .iter()
            .map(WarpField::max_magnitude)
            .fold(0.0_f32, f32::max)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // WarpField::new
    // ------------------------------------------------------------------

    #[test]
    fn test_warp_field_new_zeros() {
        let wf = WarpField::new(5);
        assert_eq!(wf.num_vertices, 5);
        assert_eq!(wf.displacements.len(), 15);
        assert!(wf.displacements.iter().all(|&v| v == 0.0));
    }

    // ------------------------------------------------------------------
    // WarpField::from_displacements
    // ------------------------------------------------------------------

    #[test]
    fn test_from_displacements_valid() {
        let data = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let wf = WarpField::from_displacements(data).expect("should succeed");
        assert_eq!(wf.num_vertices, 2);
        assert_eq!(wf.displacements[3], 4.0);
    }

    #[test]
    fn test_from_displacements_invalid() {
        let data = vec![1.0_f32, 2.0]; // length 2, not divisible by 3
        let result = WarpField::from_displacements(data);
        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(WarpFieldError::DimensionMismatch { .. })
        ));
    }

    // ------------------------------------------------------------------
    // WarpField::get / set
    // ------------------------------------------------------------------

    #[test]
    fn test_get_in_range() {
        let mut wf = WarpField::new(3);
        wf.set(1, [1.0, 2.0, 3.0]).expect("in range");
        assert_eq!(wf.get(1), [1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_get_out_of_range() {
        let wf = WarpField::new(3);
        assert_eq!(wf.get(10), [0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_set_out_of_range_error() {
        let mut wf = WarpField::new(3);
        let result = wf.set(5, [1.0, 2.0, 3.0]);
        assert!(result.is_err());
    }

    // ------------------------------------------------------------------
    // WarpField magnitudes
    // ------------------------------------------------------------------

    #[test]
    fn test_max_magnitude() {
        let data = vec![3.0_f32, 4.0, 0.0, 1.0, 0.0, 0.0]; // magnitudes: 5.0, 1.0
        let wf = WarpField::from_displacements(data).expect("ok");
        let max = wf.max_magnitude();
        assert!((max - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_mean_magnitude() {
        let data = vec![3.0_f32, 4.0, 0.0, 0.0, 0.0, 0.0]; // magnitudes: 5.0, 0.0
        let wf = WarpField::from_displacements(data).expect("ok");
        let mean = wf.mean_magnitude();
        assert!((mean - 2.5).abs() < 1e-5); // (5.0 + 0.0) / 2
    }

    #[test]
    fn test_rms_magnitude() {
        // 1 vertex with displacement [1, 0, 0]: RMS = sqrt(1/1) = 1
        let data = vec![1.0_f32, 0.0, 0.0];
        let wf = WarpField::from_displacements(data).expect("ok");
        assert!((wf.rms_magnitude() - 1.0).abs() < 1e-5);
    }

    // ------------------------------------------------------------------
    // WarpField::scale
    // ------------------------------------------------------------------

    #[test]
    fn test_scale() {
        let data = vec![1.0_f32, 2.0, 3.0];
        let wf = WarpField::from_displacements(data).expect("ok");
        let scaled = wf.scale(2.0);
        assert!((scaled.displacements[0] - 2.0).abs() < 1e-5);
        assert!((scaled.displacements[1] - 4.0).abs() < 1e-5);
        assert!((scaled.displacements[2] - 6.0).abs() < 1e-5);
    }

    // ------------------------------------------------------------------
    // WarpField::add
    // ------------------------------------------------------------------

    #[test]
    fn test_add_dimension_mismatch() {
        let a = WarpField::new(3);
        let b = WarpField::new(4);
        assert!(matches!(
            a.add(&b),
            Err(WarpFieldError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_add_correct() {
        let a = WarpField::from_displacements(vec![1.0_f32, 0.0, 0.0]).expect("ok");
        let b = WarpField::from_displacements(vec![0.0_f32, 1.0, 0.0]).expect("ok");
        let c = a.add(&b).expect("add ok");
        assert_eq!(c.get(0), [1.0, 1.0, 0.0]);
    }

    // ------------------------------------------------------------------
    // WarpField::lerp
    // ------------------------------------------------------------------

    #[test]
    fn test_lerp_t0_gives_self() {
        let a = WarpField::from_displacements(vec![1.0_f32, 2.0, 3.0]).expect("ok");
        let b = WarpField::from_displacements(vec![7.0_f32, 8.0, 9.0]).expect("ok");
        let result = a.lerp(&b, 0.0).expect("lerp ok");
        assert!((result.get(0)[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_lerp_t1_gives_other() {
        let a = WarpField::from_displacements(vec![1.0_f32, 2.0, 3.0]).expect("ok");
        let b = WarpField::from_displacements(vec![7.0_f32, 8.0, 9.0]).expect("ok");
        let result = a.lerp(&b, 1.0).expect("lerp ok");
        assert!((result.get(0)[0] - 7.0).abs() < 1e-5);
    }

    // ------------------------------------------------------------------
    // WarpField::apply_to_vertices
    // ------------------------------------------------------------------

    #[test]
    fn test_apply_to_vertices() {
        let vertices = vec![0.0_f32, 0.0, 0.0, 1.0, 1.0, 1.0];
        let wf = WarpField::from_displacements(vec![1.0_f32, 0.0, 0.0, 0.0, 0.0, 1.0]).expect("ok");
        let result = wf.apply_to_vertices(&vertices).expect("apply ok");
        assert!((result[0] - 1.0).abs() < 1e-5); // x of vertex 0: 0+1=1
        assert!((result[5] - 2.0).abs() < 1e-5); // z of vertex 1: 1+1=2
    }

    // ------------------------------------------------------------------
    // WarpField::clamp_magnitudes
    // ------------------------------------------------------------------

    #[test]
    fn test_clamp_magnitudes() {
        // Vertex 0: [3, 4, 0] → magnitude 5.  Clamp to 2.5 → scale by 0.5.
        let mut wf = WarpField::from_displacements(vec![3.0_f32, 4.0, 0.0]).expect("ok");
        wf.clamp_magnitudes(2.5);
        let d = wf.get(0);
        let mag = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        assert!((mag - 2.5).abs() < 1e-4);
    }

    // ------------------------------------------------------------------
    // linear_combination
    // ------------------------------------------------------------------

    #[test]
    fn test_linear_combination_empty() {
        let result = linear_combination(&[]);
        assert!(matches!(result, Err(WarpFieldError::EmptyField)));
    }

    #[test]
    fn test_linear_combination_single_field() {
        let field = WarpField::from_displacements(vec![1.0_f32, 0.0, 0.0]).expect("ok");
        let result = linear_combination(&[(field, 3.0)]).expect("ok");
        assert!((result.get(0)[0] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_linear_combination_two_fields() {
        let a = WarpField::from_displacements(vec![1.0_f32, 0.0, 0.0]).expect("ok");
        let b = WarpField::from_displacements(vec![0.0_f32, 1.0, 0.0]).expect("ok");
        let result = linear_combination(&[(a, 2.0), (b, 3.0)]).expect("ok");
        let d = result.get(0);
        assert!((d[0] - 2.0).abs() < 1e-5);
        assert!((d[1] - 3.0).abs() < 1e-5);
    }

    // ------------------------------------------------------------------
    // build_vertex_adjacency
    // ------------------------------------------------------------------

    #[test]
    fn test_build_vertex_adjacency_triangle() {
        // Single triangle: vertices 0,1,2 — each has exactly 2 neighbors.
        let faces = vec![0_u32, 1, 2];
        let adj = build_vertex_adjacency(3, &faces).expect("ok");
        assert_eq!(adj[0].len(), 2);
        assert_eq!(adj[1].len(), 2);
        assert_eq!(adj[2].len(), 2);
        assert!(adj[0].contains(&1));
        assert!(adj[0].contains(&2));
    }

    #[test]
    fn test_build_vertex_adjacency_invalid_faces() {
        let faces = vec![0_u32, 1]; // not divisible by 3
        let result = build_vertex_adjacency(3, &faces);
        assert!(result.is_err());
    }

    // ------------------------------------------------------------------
    // laplacian_smooth_warp_field
    // ------------------------------------------------------------------

    #[test]
    fn test_laplacian_smooth_zero_iterations_unchanged() {
        let mut wf = WarpField::new(3);
        wf.set(0, [1.0, 0.0, 0.0]).expect("ok");
        let faces = vec![0_u32, 1, 2];
        let adj = build_vertex_adjacency(3, &faces).expect("ok");
        let smoothed = laplacian_smooth_warp_field(&wf, &adj, 0, 1.0).expect("ok");
        assert!((smoothed.get(0)[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_laplacian_smooth_reduces_variance() {
        // Vertex 0 has large displacement; neighbours have zero.
        // After smoothing, variance should decrease.
        let mut wf = WarpField::new(3);
        wf.set(0, [6.0, 0.0, 0.0]).expect("ok");
        let faces = vec![0_u32, 1, 2];
        let adj = build_vertex_adjacency(3, &faces).expect("ok");

        let smoothed = laplacian_smooth_warp_field(&wf, &adj, 1, 1.0).expect("ok");
        // After full Laplacian step, vertex 0 displacement should be reduced.
        let d0_before = wf.get(0)[0];
        let d0_after = smoothed.get(0)[0];
        assert!(
            d0_after < d0_before,
            "variance should decrease: before={d0_before}, after={d0_after}"
        );
    }

    // ------------------------------------------------------------------
    // WarpMask
    // ------------------------------------------------------------------

    #[test]
    fn test_warp_mask_full_all_ones() {
        let mask = WarpMask::full(4);
        assert!(mask.weights.iter().all(|&w| (w - 1.0).abs() < 1e-5));
    }

    #[test]
    fn test_warp_mask_from_bool() {
        let bools = [true, false, true];
        let mask = WarpMask::from_bool(&bools);
        assert!((mask.weights[0] - 1.0).abs() < 1e-5);
        assert!((mask.weights[1]).abs() < 1e-5);
        assert!((mask.weights[2] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_warp_mask_geodesic_center_weight_one() {
        // Build a 3-vertex triangle adjacency.
        let faces = vec![0_u32, 1, 2];
        let adj = build_vertex_adjacency(3, &faces).expect("ok");
        let mask = WarpMask::from_geodesic_distance(&adj, &[0], 1.0);
        // Center vertex should have weight 1.0.
        assert!((mask.weights[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_warp_mask_apply_to_field() {
        let wf = WarpField::from_displacements(vec![2.0_f32, 0.0, 0.0, 4.0, 0.0, 0.0]).expect("ok");
        let mask = WarpMask::from_bool(&[true, false]);
        let result = mask.apply_to_field(&wf).expect("ok");
        // vertex 0 fully warped, vertex 1 zeroed out.
        assert!((result.get(0)[0] - 2.0).abs() < 1e-5);
        assert!((result.get(1)[0]).abs() < 1e-5);
    }

    #[test]
    fn test_warp_mask_invert() {
        let mask = WarpMask::from_bool(&[true, false]);
        let inv = mask.invert();
        assert!((inv.weights[0]).abs() < 1e-5);
        assert!((inv.weights[1] - 1.0).abs() < 1e-5);
    }

    // ------------------------------------------------------------------
    // per_vertex_magnitude
    // ------------------------------------------------------------------

    #[test]
    fn test_per_vertex_magnitude_correct() {
        // Vertex 0: [3,4,0] → 5.0; vertex 1: [0,0,0] → 0.0
        let data = vec![3.0_f32, 4.0, 0.0, 0.0, 0.0, 0.0];
        let wf = WarpField::from_displacements(data).expect("ok");
        let mags = per_vertex_magnitude(&wf);
        assert!((mags[0] - 5.0).abs() < 1e-5);
        assert!(mags[1].abs() < 1e-5);
    }

    // ------------------------------------------------------------------
    // find_large_displacements
    // ------------------------------------------------------------------

    #[test]
    fn test_find_large_displacements() {
        // v0 mag=5, v1 mag=1, v2 mag=0  → only v0 exceeds threshold 2.0
        let data = vec![3.0_f32, 4.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let wf = WarpField::from_displacements(data).expect("ok");
        let large = find_large_displacements(&wf, 2.0);
        assert_eq!(large, vec![0]);
    }

    // ------------------------------------------------------------------
    // WarpFieldSequence
    // ------------------------------------------------------------------

    #[test]
    fn test_sequence_push_len() {
        let mut seq = WarpFieldSequence::new(2);
        assert!(seq.is_empty());
        seq.push(WarpField::new(2)).expect("ok");
        seq.push(WarpField::new(2)).expect("ok");
        assert_eq!(seq.len(), 2);
    }

    #[test]
    fn test_sequence_sample_at() {
        let mut seq = WarpFieldSequence::new(1);
        let mut f0 = WarpField::new(1);
        f0.set(0, [0.0, 0.0, 0.0]).expect("ok");
        let mut f1 = WarpField::new(1);
        f1.set(0, [2.0, 0.0, 0.0]).expect("ok");
        seq.push(f0).expect("ok");
        seq.push(f1).expect("ok");

        // t=0.5 → midpoint between frame 0 and frame 1
        let sample = seq.sample_at(0.5).expect("ok");
        assert!((sample.get(0)[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_sequence_sample_empty() {
        let seq = WarpFieldSequence::new(2);
        assert!(matches!(
            seq.sample_at(0.0),
            Err(WarpFieldError::EmptyField)
        ));
    }

    #[test]
    fn test_sequence_frame_velocity() {
        let mut seq = WarpFieldSequence::new(1);
        let mut f0 = WarpField::new(1);
        f0.set(0, [1.0, 0.0, 0.0]).expect("ok");
        let mut f1 = WarpField::new(1);
        f1.set(0, [4.0, 0.0, 0.0]).expect("ok");
        seq.push(f0).expect("ok");
        seq.push(f1).expect("ok");

        let vel = seq.frame_velocity(0).expect("ok");
        // velocity = f1 - f0 = [3, 0, 0]
        assert!((vel.get(0)[0] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_sequence_frame_velocity_empty() {
        let seq = WarpFieldSequence::new(2);
        assert!(matches!(
            seq.frame_velocity(0),
            Err(WarpFieldError::EmptyField)
        ));
    }

    // ------------------------------------------------------------------
    // WarpFieldStats via compute_warp_stats
    // ------------------------------------------------------------------

    #[test]
    fn test_compute_warp_stats_correctness() {
        // Two vertices: magnitudes 3.0, 5.0
        let data = vec![3.0_f32, 0.0, 0.0, 3.0, 4.0, 0.0];
        let wf = WarpField::from_displacements(data).expect("ok");
        let stats = compute_warp_stats(&wf);
        assert_eq!(stats.num_vertices, 2);
        assert!((stats.max_magnitude - 5.0).abs() < 1e-4);
        assert!((stats.mean_magnitude - 4.0).abs() < 1e-4);
        assert_eq!(stats.fraction_nonzero, 1.0);
    }

    // ------------------------------------------------------------------
    // Additional coverage
    // ------------------------------------------------------------------

    #[test]
    fn test_warp_field_zero_out() {
        let mut wf = WarpField::from_displacements(vec![1.0_f32, 2.0, 3.0]).expect("ok");
        wf.zero_out();
        assert!(wf.displacements.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_warp_mask_intersect() {
        let a = WarpMask {
            weights: vec![1.0, 0.5],
        };
        let b = WarpMask {
            weights: vec![0.5, 1.0],
        };
        let c = a.intersect(&b).expect("ok");
        assert!((c.weights[0] - 0.5).abs() < 1e-5);
        assert!((c.weights[1] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_warp_field_sequence_wrong_vertex_count() {
        let mut seq = WarpFieldSequence::new(3);
        let bad = WarpField::new(5);
        assert!(matches!(
            seq.push(bad),
            Err(WarpFieldError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_laplacian_smooth_dimension_mismatch() {
        let wf = WarpField::new(3);
        let bad_adj: Vec<Vec<usize>> = vec![vec![], vec![]]; // only 2 entries
        let result = laplacian_smooth_warp_field(&wf, &bad_adj, 1, 0.5);
        assert!(matches!(
            result,
            Err(WarpFieldError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_global_max_magnitude() {
        let mut seq = WarpFieldSequence::new(1);
        let mut f0 = WarpField::new(1);
        f0.set(0, [3.0, 4.0, 0.0]).expect("ok");
        let f1 = WarpField::new(1);
        seq.push(f0).expect("ok");
        seq.push(f1).expect("ok");
        assert!((seq.global_max_magnitude() - 5.0).abs() < 1e-4);
    }

    #[test]
    fn test_linear_combination_dimension_mismatch() {
        let a = WarpField::new(3);
        let b = WarpField::new(5);
        let result = linear_combination(&[(a, 1.0), (b, 1.0)]);
        assert!(matches!(
            result,
            Err(WarpFieldError::DimensionMismatch { .. })
        ));
    }
}
