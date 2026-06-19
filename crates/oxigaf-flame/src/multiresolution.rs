//! Multi-resolution mesh (Level-of-Detail) support via Quadric Error Metric decimation.
//!
//! This module implements the Garland-Heckbert Quadric Error Metric (QEM) algorithm
//! for producing multiple LoD levels from a base mesh. Each level is produced by
//! collapsing the lowest-cost edge repeatedly until the target vertex count is reached.
//!
//! The FLAME head model has 5023 vertices. For `O(N²)` QEM this is very fast.

#![allow(clippy::doc_markdown)]

use std::collections::{HashMap, HashSet};

use crate::error::FlameError;

// ---------------------------------------------------------------------------
// Quadric — 4×4 symmetric error matrix stored as upper triangle (10 floats)
// ---------------------------------------------------------------------------
// Ordering: [a00, a01, a02, a03, a11, a12, a13, a22, a23, a33]
// Indices:    [0]   [1]   [2]   [3]   [4]   [5]   [6]   [7]   [8]   [9]
//
// Matrix layout (row, col → array index):
//  (0,0)→0  (0,1)→1  (0,2)→2  (0,3)→3
//           (1,1)→4  (1,2)→5  (1,3)→6
//                    (2,2)→7  (2,3)→8
//                             (3,3)→9
// ---------------------------------------------------------------------------
#[derive(Debug, Clone, Copy)]
struct Quadric([f64; 10]);

impl Quadric {
    /// All-zeros quadric.
    fn zero() -> Self {
        Self([0.0; 10])
    }

    /// Construct a quadric from a plane equation `a*x + b*y + c*z + d = 0`.
    ///
    /// The plane vector is `p = [a, b, c, d]^T`.  The fundamental error quadric
    /// is `Q = p * p^T` (outer product), giving a 4×4 symmetric matrix.
    #[allow(clippy::many_single_char_names)]
    fn from_plane(a: f64, b: f64, c: f64, d: f64) -> Self {
        // Q[i][j] = p[i] * p[j] — store upper triangle only
        let p = [a, b, c, d];
        let mut q = [0.0f64; 10];
        // Row 0: (0,0) (0,1) (0,2) (0,3)
        q[0] = p[0] * p[0];
        q[1] = p[0] * p[1];
        q[2] = p[0] * p[2];
        q[3] = p[0] * p[3];
        // Row 1: (1,1) (1,2) (1,3)
        q[4] = p[1] * p[1];
        q[5] = p[1] * p[2];
        q[6] = p[1] * p[3];
        // Row 2: (2,2) (2,3)
        q[7] = p[2] * p[2];
        q[8] = p[2] * p[3];
        // Row 3: (3,3)
        q[9] = p[3] * p[3];
        Self(q)
    }

    /// Element-wise sum of two quadrics.
    fn add(&self, other: &Quadric) -> Self {
        let mut r = [0.0f64; 10];
        for (r_elem, (s_elem, o_elem)) in r.iter_mut().zip(self.0.iter().zip(other.0.iter())) {
            *r_elem = s_elem + o_elem;
        }
        Self(r)
    }

    /// Evaluate `v_hom^T Q v_hom` where `v_hom = [x, y, z, 1]`.
    ///
    /// This gives the squared distance from the optimal position for vertex `v`
    /// given this quadric's accumulated plane constraints.
    #[allow(clippy::many_single_char_names)]
    fn error(&self, v: [f64; 3]) -> f64 {
        let [x, y, z] = v;
        let q = &self.0;
        // Expand [x,y,z,1]^T * Q * [x,y,z,1]
        // using symmetric storage (off-diagonal terms appear twice)
        x * x * q[0]
            + 2.0 * x * y * q[1]
            + 2.0 * x * z * q[2]
            + 2.0 * x * q[3]
            + y * y * q[4]
            + 2.0 * y * z * q[5]
            + 2.0 * y * q[6]
            + z * z * q[7]
            + 2.0 * z * q[8]
            + q[9]
    }

    /// Solve the 3×3 linear system (top-left sub-matrix of Q) for the optimal
    /// vertex position that minimises this quadric's error.
    ///
    /// The system is `A * v = b` where:
    /// - `A` = upper-left 3×3 block: [[q00, q01, q02], [q01, q11, q12], [q02, q12, q22]]
    /// - `b` = negated last column of the top 3 rows: [-q03, -q13, -q23]
    ///
    /// Returns `None` if the determinant is below the singularity threshold.
    fn minimize_position(&self) -> Option<[f64; 3]> {
        let q = &self.0;
        // 3×3 matrix rows: [a00, a01, a02 | a10, a11, a12 | a20, a21, a22]
        // stored symmetrically: a01=a10, a02=a20, a12=a21
        let a00 = q[0];
        let a01 = q[1];
        let a02 = q[2];
        let a11 = q[4];
        let a12 = q[5];
        let a22 = q[7];

        // Compute 3×3 determinant via cofactor expansion along row 0
        let det = a00 * (a11 * a22 - a12 * a12) - a01 * (a01 * a22 - a12 * a02)
            + a02 * (a01 * a12 - a11 * a02);

        if det.abs() < 1e-10 {
            return None;
        }

        // RHS: b = [-q03, -q13, -q23]
        let b0 = -q[3];
        let b1 = -q[6];
        let b2 = -q[8];

        // Solve using Cramer's rule
        let inv_det = 1.0 / det;

        // x = det(A with col 0 replaced by b) / det(A)
        let x = inv_det
            * (b0 * (a11 * a22 - a12 * a12) - a01 * (b1 * a22 - a12 * b2)
                + a02 * (b1 * a12 - a11 * b2));

        let y = inv_det
            * (a00 * (b1 * a22 - a12 * b2) - b0 * (a01 * a22 - a12 * a02)
                + a02 * (a01 * b2 - b1 * a02));

        let z = inv_det
            * (a00 * (a11 * b2 - b1 * a12) - a01 * (a01 * b2 - b1 * a02)
                + b0 * (a01 * a12 - a11 * a02));

        // Guard against NaN/Inf (can occur with near-degenerate quadrics)
        if x.is_finite() && y.is_finite() && z.is_finite() {
            Some([x, y, z])
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Public data types
// ---------------------------------------------------------------------------

/// A single LoD mesh level produced by decimation.
#[derive(Debug, Clone)]
pub struct MeshLevel {
    /// Vertex positions at this level.
    pub vertices: Vec<[f32; 3]>,
    /// Triangle faces (indices into `vertices`).
    pub faces: Vec<[u32; 3]>,
    /// Per-vertex normals (area-weighted and normalised).
    pub normals: Vec<[f32; 3]>,
    /// Maps each original (level-0) vertex index to this level's vertex index.
    ///
    /// Length equals the number of vertices in the *original* full-resolution mesh
    /// (level 0).  Vertices that were collapsed into another vertex point to the
    /// surviving vertex's new compact index.
    pub vertex_map: Vec<u32>,
}

/// A multi-resolution mesh containing several LoD levels.
///
/// `levels[0]` is the full-resolution mesh (identical to input).
/// Each subsequent level has fewer vertices.
#[derive(Debug, Clone)]
pub struct MultiResMesh {
    /// LoD levels, from finest (index 0) to coarsest (last index).
    pub levels: Vec<MeshLevel>,
    /// Target vertex count used when generating each level.
    pub target_counts: Vec<usize>,
}

impl MultiResMesh {
    /// Number of LoD levels available.
    #[inline]
    #[must_use]
    pub fn level_count(&self) -> usize {
        self.levels.len()
    }

    /// Number of vertices at the given level index.
    ///
    /// Returns 0 if `level` is out of range.
    #[inline]
    #[must_use]
    pub fn vertex_count_at(&self, level: usize) -> usize {
        self.levels.get(level).map_or(0, |l| l.vertices.len())
    }

    /// Ratio of the coarsest level's vertex count to the finest level.
    ///
    /// Returns `1.0` if there is only one level.
    #[must_use]
    pub fn compression_ratio(&self) -> f32 {
        if self.levels.len() <= 1 {
            return 1.0;
        }
        let finest = self.levels[0].vertices.len();
        let coarsest = self.levels.last().map_or(0, |l| l.vertices.len());
        if finest == 0 {
            return 1.0;
        }
        finest as f32 / coarsest.max(1) as f32
    }
}

/// Configuration controlling the decimation process.
#[derive(Debug, Clone)]
pub struct DecimationConfig {
    /// Stop once the mesh reaches at most this many vertices.
    pub target_vertex_count: usize,
    /// Stop early if the minimum edge error exceeds this value.
    ///
    /// Set to `f64::MAX` (the default) to disable the threshold.
    pub max_error_threshold: f64,
    /// If `true`, do not collapse edges that lie on a mesh boundary.
    pub preserve_boundary: bool,
    /// Minimum sin of the smallest angle in a triangle after collapse.
    ///
    /// Set to `0.0` (the default) to allow any triangle quality.
    pub min_triangle_quality: f32,
}

impl Default for DecimationConfig {
    fn default() -> Self {
        Self {
            target_vertex_count: 0,
            max_error_threshold: f64::MAX,
            preserve_boundary: true,
            min_triangle_quality: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Normal computation utility
// ---------------------------------------------------------------------------

/// Compute area-weighted per-vertex normals.
///
/// Each face contributes its (unnormalised) cross-product normal to all three
/// of its vertices.  Zero-area faces are silently skipped.  The final per-vertex
/// normals are normalised to unit length (zero-sum vertices produce `[0,0,1]`
/// as a fallback).
///
/// # Arguments
///
/// * `vertices` — vertex positions as `[x, y, z]`.
/// * `faces` — triangle indices; each face is `[i0, i1, i2]`.
///
/// # Returns
///
/// A `Vec<[f32; 3]>` with one unit normal per vertex.
#[must_use]
pub fn compute_vertex_normals(vertices: &[[f32; 3]], faces: &[[u32; 3]]) -> Vec<[f32; 3]> {
    let n_verts = vertices.len();
    let mut accum = vec![[0.0f32; 3]; n_verts];

    for face in faces {
        let i0 = face[0] as usize;
        let i1 = face[1] as usize;
        let i2 = face[2] as usize;

        // Guard against out-of-bounds (shouldn't happen with valid meshes)
        if i0 >= n_verts || i1 >= n_verts || i2 >= n_verts {
            continue;
        }

        let v0 = vertices[i0];
        let v1 = vertices[i1];
        let v2 = vertices[i2];

        // edge vectors
        let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
        let e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];

        // cross product (area-proportional normal)
        let nx = e1[1] * e2[2] - e1[2] * e2[1];
        let ny = e1[2] * e2[0] - e1[0] * e2[2];
        let nz = e1[0] * e2[1] - e1[1] * e2[0];

        // Skip zero-area faces
        let len_sq = nx * nx + ny * ny + nz * nz;
        if len_sq < f32::EPSILON {
            continue;
        }

        accum[i0][0] += nx;
        accum[i0][1] += ny;
        accum[i0][2] += nz;
        accum[i1][0] += nx;
        accum[i1][1] += ny;
        accum[i1][2] += nz;
        accum[i2][0] += nx;
        accum[i2][1] += ny;
        accum[i2][2] += nz;
    }

    accum
        .into_iter()
        .map(|n| {
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            if len > 1e-10 {
                [n[0] / len, n[1] / len, n[2] / len]
            } else {
                [0.0, 0.0, 1.0]
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// MeshDecimator
// ---------------------------------------------------------------------------

/// Internal mesh decimation engine using Quadric Error Metrics.
///
/// Implements the Garland-Heckbert edge-collapse strategy: at each step the
/// edge with the smallest QEM cost is collapsed.
pub struct MeshDecimator {
    vertices: Vec<[f64; 3]>,
    faces: Vec<[u32; 3]>,
    quadrics: Vec<Quadric>,
    valid_vertices: Vec<bool>,
    valid_faces: Vec<bool>,
    /// Collapse chain: `vertex_map[v]` gives the index that vertex `v` was
    /// merged into (initially the identity `v → v`).
    vertex_map: Vec<u32>,
    /// Number of currently valid vertices.
    live_vertex_count: usize,
    /// Original vertex count (for vertex_map output length).
    original_vertex_count: usize,
}

impl MeshDecimator {
    /// Create a new decimator from a set of vertices and faces.
    ///
    /// Initialises per-vertex quadrics from the incident face planes.
    ///
    /// # Errors
    ///
    /// Returns [`FlameError::InvalidParams`] if `vertices` or `faces` is empty.
    pub fn new(vertices: &[[f32; 3]], faces: &[[u32; 3]]) -> Result<Self, FlameError> {
        if vertices.is_empty() {
            return Err(FlameError::InvalidParams(
                "MeshDecimator: vertices slice is empty".into(),
            ));
        }
        if faces.is_empty() {
            return Err(FlameError::InvalidParams(
                "MeshDecimator: faces slice is empty".into(),
            ));
        }

        let n_verts = vertices.len();
        let original_vertex_count = n_verts;

        // Convert to f64 for numerical stability
        let verts_f64: Vec<[f64; 3]> = vertices
            .iter()
            .map(|v| [f64::from(v[0]), f64::from(v[1]), f64::from(v[2])])
            .collect();

        // Build per-vertex quadrics by accumulating face plane quadrics
        let mut quadrics = vec![Quadric::zero(); n_verts];

        for face in faces {
            let i0 = face[0] as usize;
            let i1 = face[1] as usize;
            let i2 = face[2] as usize;

            if i0 >= n_verts || i1 >= n_verts || i2 >= n_verts {
                continue; // skip degenerate indices
            }

            let v0 = verts_f64[i0];
            let v1 = verts_f64[i1];
            let v2 = verts_f64[i2];

            // Compute face plane normal (not normalised — Garland & Heckbert
            // use the normalised plane equation for correctness)
            let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
            let e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];

            let nx = e1[1] * e2[2] - e1[2] * e2[1];
            let ny = e1[2] * e2[0] - e1[0] * e2[2];
            let nz = e1[0] * e2[1] - e1[1] * e2[0];

            let len = (nx * nx + ny * ny + nz * nz).sqrt();
            if len < 1e-12 {
                continue; // degenerate face
            }

            // Normalise the plane equation: plane_a*x + plane_b*y + plane_c*z + plane_d = 0
            let plane_a = nx / len;
            let plane_b = ny / len;
            let plane_c = nz / len;
            let plane_d = -(plane_a * v0[0] + plane_b * v0[1] + plane_c * v0[2]);

            let fq = Quadric::from_plane(plane_a, plane_b, plane_c, plane_d);
            quadrics[i0] = quadrics[i0].add(&fq);
            quadrics[i1] = quadrics[i1].add(&fq);
            quadrics[i2] = quadrics[i2].add(&fq);
        }

        let valid_faces: Vec<bool> = vec![true; faces.len()];
        let valid_vertices: Vec<bool> = vec![true; n_verts];
        let vertex_map: Vec<u32> = (0..n_verts as u32).collect();

        Ok(Self {
            vertices: verts_f64,
            faces: faces.to_vec(),
            quadrics,
            valid_vertices,
            valid_faces,
            vertex_map,
            live_vertex_count: n_verts,
            original_vertex_count,
        })
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Follow the collapse chain to find the current canonical vertex for `v`.
    fn canonical(&self, mut v: u32) -> u32 {
        while self.vertex_map[v as usize] != v {
            v = self.vertex_map[v as usize];
        }
        v
    }

    /// Collect all unique valid edges from valid faces.
    ///
    /// An edge `(a, b)` is stored as `(min, max)` so each edge appears once.
    fn collect_edges(&self) -> Vec<(u32, u32)> {
        let mut edge_set: HashSet<(u32, u32)> = HashSet::with_capacity(self.faces.len() * 3);

        for (fi, face) in self.faces.iter().enumerate() {
            if !self.valid_faces[fi] {
                continue;
            }
            let a = self.canonical(face[0]);
            let b = self.canonical(face[1]);
            let c = self.canonical(face[2]);

            // Skip degenerate faces that collapsed to the same vertex
            if a == b || b == c || a == c {
                continue;
            }

            for (u, v) in [(a, b), (b, c), (a, c)] {
                let key = if u < v { (u, v) } else { (v, u) };
                edge_set.insert(key);
            }
        }

        edge_set.into_iter().collect()
    }

    /// Build a set of boundary edges (edges belonging to exactly one face).
    ///
    /// Returns a `HashSet`-equivalent via a `HashMap<edge, count>`.
    fn boundary_edges(&self) -> HashMap<(u32, u32), usize> {
        let mut counts: HashMap<(u32, u32), usize> = HashMap::new();

        for (fi, face) in self.faces.iter().enumerate() {
            if !self.valid_faces[fi] {
                continue;
            }
            let a = self.canonical(face[0]);
            let b = self.canonical(face[1]);
            let c = self.canonical(face[2]);

            if a == b || b == c || a == c {
                continue;
            }

            for (u, v) in [(a, b), (b, c), (a, c)] {
                let key = if u < v { (u, v) } else { (v, u) };
                *counts.entry(key).or_insert(0) += 1;
            }
        }

        counts
    }

    /// Check whether collapsing edge `(v1, v2)` to position `pos` would produce
    /// a degenerate triangle (sin of smallest angle < `min_quality`).
    ///
    /// Returns `true` if the quality check passes (collapse is acceptable).
    fn check_triangle_quality(&self, v1: u32, v2: u32, pos: [f64; 3], min_quality: f32) -> bool {
        if min_quality <= 0.0 {
            return true; // check disabled
        }

        // For each face sharing v1 or v2, simulate what it would look like
        // after collapse and check quality
        let min_q = f64::from(min_quality);

        for (fi, face) in self.faces.iter().enumerate() {
            if !self.valid_faces[fi] {
                continue;
            }
            let a = self.canonical(face[0]);
            let b = self.canonical(face[1]);
            let c = self.canonical(face[2]);

            let contains_v1 = a == v1 || b == v1 || c == v1;
            let contains_v2 = a == v2 || b == v2 || c == v2;

            if !contains_v1 && !contains_v2 {
                continue;
            }

            // Would this face become degenerate after collapse?
            if contains_v1 && contains_v2 {
                continue; // This face gets removed
            }

            // Map v2 → v1, use new pos for v1
            let vp = |vi: u32| -> [f64; 3] {
                if vi == v1 || vi == v2 {
                    pos
                } else {
                    self.vertices[vi as usize]
                }
            };

            let pa = vp(a);
            let pb = vp(b);
            let pc = vp(c);

            // Cross product magnitude = 2 * area
            let e1 = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
            let e2 = [pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]];
            let cross_mag = {
                let cx = e1[1] * e2[2] - e1[2] * e2[1];
                let cy = e1[2] * e2[0] - e1[0] * e2[2];
                let cz = e1[0] * e2[1] - e1[1] * e2[0];
                (cx * cx + cy * cy + cz * cz).sqrt()
            };

            // Lengths of edges
            let len_e1 = (e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2]).sqrt();
            let len_e2 = (e2[0] * e2[0] + e2[1] * e2[1] + e2[2] * e2[2]).sqrt();

            let denom = len_e1 * len_e2;
            if denom < 1e-12 {
                return false; // degenerate
            }

            // sin(angle at a) = |e1 × e2| / (|e1| * |e2|)
            let sin_a = cross_mag / denom;
            if sin_a < min_q {
                return false;
            }
        }

        true
    }

    // -----------------------------------------------------------------------
    // Main decimation loop
    // -----------------------------------------------------------------------

    /// Decimate the mesh according to `config` and return the resulting [`MeshLevel`].
    ///
    /// The decimator's internal state is mutated in place; call again on a fresh
    /// instance to decimate to a different target.
    ///
    /// # Errors
    ///
    /// Returns [`FlameError::InvalidParams`] if the config target is 0.
    pub fn decimate(&mut self, config: &DecimationConfig) -> Result<MeshLevel, FlameError> {
        if config.target_vertex_count == 0 {
            return Err(FlameError::InvalidParams(
                "DecimationConfig: target_vertex_count must be > 0".into(),
            ));
        }

        // Iteratively collapse lowest-cost edges until target is reached
        loop {
            if self.live_vertex_count <= config.target_vertex_count {
                break;
            }

            let edges = self.collect_edges();
            if edges.is_empty() {
                break;
            }

            // Build boundary edge map if we need to preserve boundaries
            let boundary = if config.preserve_boundary {
                self.boundary_edges()
            } else {
                HashMap::new()
            };

            // Find the best (lowest-cost) valid edge to collapse
            let mut best_cost = f64::MAX;
            let mut best_edge: Option<(u32, u32)> = None;
            let mut best_pos = [0.0f64; 3];

            for (v1, v2) in &edges {
                let v1 = *v1;
                let v2 = *v2;

                // Skip boundary edges when preserve_boundary is set
                if config.preserve_boundary {
                    let key = if v1 < v2 { (v1, v2) } else { (v2, v1) };
                    if boundary.get(&key).copied().unwrap_or(0) == 1 {
                        continue;
                    }
                }

                let q_combined = self.quadrics[v1 as usize].add(&self.quadrics[v2 as usize]);

                // Try to find optimal collapse position; fall back to midpoint
                let candidate_pos = q_combined.minimize_position().unwrap_or_else(|| {
                    let p1 = self.vertices[v1 as usize];
                    let p2 = self.vertices[v2 as usize];
                    [
                        (p1[0] + p2[0]) * 0.5,
                        (p1[1] + p2[1]) * 0.5,
                        (p1[2] + p2[2]) * 0.5,
                    ]
                });

                let cost = q_combined.error(candidate_pos);

                if cost < best_cost {
                    // Quality gate
                    if config.min_triangle_quality > 0.0
                        && !self.check_triangle_quality(
                            v1,
                            v2,
                            candidate_pos,
                            config.min_triangle_quality,
                        )
                    {
                        continue;
                    }
                    best_cost = cost;
                    best_edge = Some((v1, v2));
                    best_pos = candidate_pos;
                }
            }

            // Check error threshold
            if best_cost > config.max_error_threshold {
                break;
            }

            let Some((v1, v2)) = best_edge else {
                break; // no collapsible edge found (all blocked by boundary / quality)
            };

            // Perform the collapse: merge v2 into v1
            self.collapse_edge(v1, v2, best_pos);
        }

        Ok(self.build_mesh_level())
    }

    /// Collapse edge `(v1, v2)`: move v1 to `new_pos`, redirect all references
    /// to v2 to v1, and remove now-degenerate faces.
    fn collapse_edge(&mut self, v1: u32, v2: u32, new_pos: [f64; 3]) {
        // Merge quadrics
        let q_new = self.quadrics[v1 as usize].add(&self.quadrics[v2 as usize]);
        self.quadrics[v1 as usize] = q_new;

        // Move v1 to the optimal position
        self.vertices[v1 as usize] = new_pos;

        // Mark v2 as invalid and point it to v1
        self.valid_vertices[v2 as usize] = false;
        self.vertex_map[v2 as usize] = v1;
        self.live_vertex_count -= 1;

        // Update faces: replace v2 with v1, invalidate degenerate faces
        for fi in 0..self.faces.len() {
            if !self.valid_faces[fi] {
                continue;
            }

            let face = &mut self.faces[fi];

            // Remap each vertex through the canonical chain
            for slot in face.iter_mut() {
                if *slot == v2 {
                    *slot = v1;
                }
            }

            // After remapping, check for degenerate face (two identical vertices)
            let (a, b, c) = (face[0], face[1], face[2]);
            if a == b || b == c || a == c {
                self.valid_faces[fi] = false;
            }
        }
    }

    /// Extract the current mesh state into a [`MeshLevel`].
    ///
    /// Valid vertices are compacted to a new consecutive index range.
    /// The `vertex_map` in the returned level maps each *original* vertex
    /// (index in the level-0 mesh) to the appropriate compact index.
    fn build_mesh_level(&self) -> MeshLevel {
        let n_orig = self.vertices.len();

        // Map: original index → compact index for surviving vertices
        let mut old_to_new: Vec<Option<u32>> = vec![None; n_orig];
        let mut new_vertices: Vec<[f32; 3]> = Vec::new();

        for (i, &valid) in self.valid_vertices.iter().enumerate() {
            if valid {
                let new_idx = new_vertices.len() as u32;
                old_to_new[i] = Some(new_idx);
                let v = self.vertices[i];
                new_vertices.push([v[0] as f32, v[1] as f32, v[2] as f32]);
            }
        }

        // Collect valid faces with remapped indices
        let mut new_faces: Vec<[u32; 3]> = Vec::new();
        for (fi, face) in self.faces.iter().enumerate() {
            if !self.valid_faces[fi] {
                continue;
            }

            // Resolve through collapse chain
            let a = self.resolve_to_new(face[0], &old_to_new);
            let b = self.resolve_to_new(face[1], &old_to_new);
            let c = self.resolve_to_new(face[2], &old_to_new);

            match (a, b, c) {
                (Some(na), Some(nb), Some(nc)) if na != nb && nb != nc && na != nc => {
                    new_faces.push([na, nb, nc]);
                }
                _ => {} // skip degenerate or unmapped faces
            }
        }

        // Build vertex_map for the original level-0 vertex count
        let vertex_map: Vec<u32> = (0..self.original_vertex_count as u32)
            .map(|orig_idx| {
                // Follow the collapse chain until we reach a valid surviving vertex
                let canonical = self.canonical(orig_idx);
                // Map the canonical vertex to its compact index
                old_to_new[canonical as usize].unwrap_or(0)
            })
            .collect();

        let normals = compute_vertex_normals(&new_vertices, &new_faces);

        MeshLevel {
            vertices: new_vertices,
            faces: new_faces,
            normals,
            vertex_map,
        }
    }

    /// Resolve vertex `v` through the collapse chain and return its compact index.
    fn resolve_to_new(&self, v: u32, old_to_new: &[Option<u32>]) -> Option<u32> {
        let canon = self.canonical(v);
        old_to_new.get(canon as usize).copied().flatten()
    }
}

// ---------------------------------------------------------------------------
// MultiResMeshBuilder
// ---------------------------------------------------------------------------

/// Builder for creating a [`MultiResMesh`] with multiple LoD levels.
///
/// Each level is decimated from the *original* full-resolution mesh
/// (not chained), which avoids error accumulation between levels and keeps
/// vertex-map composition trivial.
#[derive(Debug, Clone)]
pub struct MultiResMeshBuilder {
    /// Target vertex counts, one per level (including the full-resolution level).
    ///
    /// The first element should equal (or exceed) the input vertex count for a
    /// no-op full-resolution level.  Subsequent elements must be strictly
    /// decreasing.
    pub target_counts: Vec<usize>,
}

impl MultiResMeshBuilder {
    /// Create a builder with explicit target vertex counts.
    #[must_use]
    pub fn new(targets: Vec<usize>) -> Self {
        Self {
            target_counts: targets,
        }
    }

    /// Default targets for the FLAME head model (5023 vertices).
    ///
    /// Produces four levels: full (5023), 2500, 1000, and 500.
    #[must_use]
    pub fn default_flame() -> Self {
        Self {
            target_counts: vec![5023, 2500, 1000, 500],
        }
    }

    /// Build the multi-resolution mesh from `vertices` and `faces`.
    ///
    /// # Algorithm
    ///
    /// 1. Level 0 is the input mesh (no decimation).
    /// 2. For each subsequent target, a fresh `MeshDecimator` is created from
    ///    the original input and run until the target vertex count is reached.
    ///    This ensures each level is independent and vertex-map composition is
    ///    straightforward.
    ///
    /// # Errors
    ///
    /// Returns [`FlameError::InvalidParams`] if `vertices` or `faces` is empty,
    /// or if any `target_count` is 0.
    pub fn build(
        &self,
        vertices: &[[f32; 3]],
        faces: &[[u32; 3]],
    ) -> Result<MultiResMesh, FlameError> {
        if vertices.is_empty() {
            return Err(FlameError::InvalidParams(
                "MultiResMeshBuilder: vertices is empty".into(),
            ));
        }
        if faces.is_empty() {
            return Err(FlameError::InvalidParams(
                "MultiResMeshBuilder: faces is empty".into(),
            ));
        }

        let n_orig = vertices.len();
        let mut levels: Vec<MeshLevel> = Vec::with_capacity(self.target_counts.len());

        // Level 0: full resolution, identity vertex_map
        let normals_l0 = compute_vertex_normals(vertices, faces);
        let vertex_map_l0: Vec<u32> = (0..n_orig as u32).collect();
        levels.push(MeshLevel {
            vertices: vertices.to_vec(),
            faces: faces.to_vec(),
            normals: normals_l0,
            vertex_map: vertex_map_l0,
        });

        // Build each subsequent level from the original mesh
        for &target in self.target_counts.iter().skip(1) {
            if target == 0 {
                return Err(FlameError::InvalidParams(
                    "MultiResMeshBuilder: target_count of 0 is not allowed".into(),
                ));
            }

            // If target >= current vertex count, clone the previous level
            if target >= n_orig {
                let prev = levels.last().map_or_else(
                    || MeshLevel {
                        vertices: vertices.to_vec(),
                        faces: faces.to_vec(),
                        normals: compute_vertex_normals(vertices, faces),
                        vertex_map: (0..n_orig as u32).collect(),
                    },
                    MeshLevel::clone,
                );
                levels.push(prev);
                continue;
            }

            let config = DecimationConfig {
                target_vertex_count: target,
                max_error_threshold: f64::MAX,
                preserve_boundary: true,
                min_triangle_quality: 0.0,
            };

            let mut decimator = MeshDecimator::new(vertices, faces)?;
            let level = decimator.decimate(&config)?;
            levels.push(level);
        }

        Ok(MultiResMesh {
            levels,
            target_counts: self.target_counts.clone(),
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Helper: a simple tetrahedron mesh
    // -----------------------------------------------------------------------

    fn tetrahedron() -> (Vec<[f32; 3]>, Vec<[u32; 3]>) {
        let verts = vec![
            [0.0f32, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [0.5, 0.5, 1.0],
        ];
        let faces = vec![[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]];
        (verts, faces)
    }

    // -----------------------------------------------------------------------
    // Quadric tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_quadric_from_plane() {
        // Plane z = 0: a=0, b=0, c=1, d=0
        let q = Quadric::from_plane(0.0, 0.0, 1.0, 0.0);
        // Q = [a,b,c,d]^T [a,b,c,d] = outer product
        // (0,0)→0, (2,2)→1 (c²), rest→0
        assert!((q.0[0]).abs() < 1e-12); // a² = 0
        assert!((q.0[7] - 1.0).abs() < 1e-12); // c² = 1
        assert!((q.0[9]).abs() < 1e-12); // d² = 0
    }

    #[test]
    fn test_quadric_error_on_plane() {
        // Plane z = 0: a=0, b=0, c=1, d=0
        let q = Quadric::from_plane(0.0, 0.0, 1.0, 0.0);

        // Any point on z=0 should have error ≈ 0
        let err_on_plane = q.error([1.0, 2.0, 0.0]);
        assert!(err_on_plane.abs() < 1e-12, "error on plane: {err_on_plane}");

        // A point at z=1 should have error = 1² = 1
        let err_above = q.error([1.0, 2.0, 1.0]);
        assert!(
            (err_above - 1.0).abs() < 1e-12,
            "error above plane: {err_above}"
        );

        // A point at z=3 should have error = 3² = 9
        let err_far = q.error([0.0, 0.0, 3.0]);
        assert!(
            (err_far - 9.0).abs() < 1e-12,
            "error far above plane: {err_far}"
        );
    }

    #[test]
    fn test_quadric_add() {
        // Two identical plane quadrics; sum should be 2× each element
        let q = Quadric::from_plane(1.0, 0.0, 0.0, 0.0);
        let q2 = q.add(&q);
        for i in 0..10 {
            assert!(
                (q2.0[i] - 2.0 * q.0[i]).abs() < 1e-12,
                "index {i}: expected {}, got {}",
                2.0 * q.0[i],
                q2.0[i]
            );
        }
    }

    #[test]
    fn test_quadric_minimize_position_xy_plane() {
        // For the plane z=0, the quadric is singular (can't fix x,y uniquely)
        // so minimize_position should return None.
        let q = Quadric::from_plane(0.0, 0.0, 1.0, 0.0);
        assert!(
            q.minimize_position().is_none(),
            "singular quadric should return None"
        );
    }

    #[test]
    fn test_quadric_minimize_position_three_planes() {
        // Three mutually orthogonal planes through a known point (1,2,3):
        //   x = 1  → (1, 0, 0, -1)
        //   y = 2  → (0, 1, 0, -2)
        //   z = 3  → (0, 0, 1, -3)
        let q1 = Quadric::from_plane(1.0, 0.0, 0.0, -1.0);
        let q2 = Quadric::from_plane(0.0, 1.0, 0.0, -2.0);
        let q3 = Quadric::from_plane(0.0, 0.0, 1.0, -3.0);
        let q = q1.add(&q2).add(&q3);

        let pos = q.minimize_position().expect("should have unique minimum");
        assert!((pos[0] - 1.0).abs() < 1e-9, "x: {}", pos[0]);
        assert!((pos[1] - 2.0).abs() < 1e-9, "y: {}", pos[1]);
        assert!((pos[2] - 3.0).abs() < 1e-9, "z: {}", pos[2]);
    }

    // -----------------------------------------------------------------------
    // Normal computation tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_compute_vertex_normals_single_triangle() {
        // A triangle in the XY plane: normal should point in +Z
        let verts = vec![[0.0f32, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let faces = vec![[0u32, 1, 2]];
        let normals = compute_vertex_normals(&verts, &faces);

        assert_eq!(normals.len(), 3);
        for n in &normals {
            assert!(n[2] > 0.99, "normal should point in +Z, got {n:?}");
        }
    }

    #[test]
    fn test_compute_vertex_normals() {
        // Two triangles forming a square in XY plane; all normals should be +Z
        let verts = vec![
            [0.0f32, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let faces = vec![[0u32, 1, 2], [0, 2, 3]];
        let normals = compute_vertex_normals(&verts, &faces);

        assert_eq!(normals.len(), 4);
        for (i, n) in normals.iter().enumerate() {
            assert!(n[2] > 0.99, "vertex {i}: normal should be ~+Z, got {n:?}");
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            assert!((len - 1.0).abs() < 1e-6, "normal should be unit length");
        }
    }

    // -----------------------------------------------------------------------
    // MeshDecimator tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_decimator_new_empty_error() {
        let result = MeshDecimator::new(&[], &[[0u32, 1, 2]]);
        assert!(result.is_err(), "empty vertices should return error");

        let verts = vec![[0.0f32, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]];
        let result2 = MeshDecimator::new(&verts, &[]);
        assert!(result2.is_err(), "empty faces should return error");
    }

    #[test]
    fn test_decimator_decimate_small_mesh() {
        // A tetrahedron (4 verts) → target 3 vertices
        let (verts, faces) = tetrahedron();
        let mut dec = MeshDecimator::new(&verts, &faces).expect("valid mesh");
        let config = DecimationConfig {
            target_vertex_count: 3,
            ..DecimationConfig::default()
        };
        let level = dec.decimate(&config).expect("decimation should succeed");

        assert!(
            level.vertices.len() <= 3,
            "expected ≤3 vertices, got {}",
            level.vertices.len()
        );
        assert!(!level.vertices.is_empty(), "must have at least 1 vertex");
    }

    #[test]
    fn test_decimator_target_vertex_count_respected() {
        // Create a slightly larger mesh (two tetrahedra sharing a face)
        let verts = vec![
            [0.0f32, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [0.5, 0.5, 1.0],
            [0.5, 0.5, -1.0],
        ];
        let faces = vec![
            [0u32, 1, 2],
            [0, 1, 3],
            [0, 2, 3],
            [1, 2, 3],
            [0, 1, 4],
            [0, 2, 4],
            [1, 2, 4],
        ];

        let config = DecimationConfig {
            target_vertex_count: 3,
            preserve_boundary: false, // small mesh, allow all collapses
            ..DecimationConfig::default()
        };

        let mut dec = MeshDecimator::new(&verts, &faces).expect("valid mesh");
        let level = dec.decimate(&config).expect("decimation ok");
        assert!(
            level.vertices.len() <= 3,
            "expected ≤3 vertices, got {}",
            level.vertices.len()
        );
    }

    #[test]
    fn test_decimator_no_degenerate_faces_after_collapse() {
        let (verts, faces) = tetrahedron();
        let mut dec = MeshDecimator::new(&verts, &faces).expect("valid mesh");
        let config = DecimationConfig {
            target_vertex_count: 3,
            preserve_boundary: false,
            ..DecimationConfig::default()
        };
        let level = dec.decimate(&config).expect("decimation ok");

        // All faces must reference valid indices with no duplicated vertices
        for face in &level.faces {
            assert!(
                (face[0] as usize) < level.vertices.len(),
                "face index {} out of bounds ({})",
                face[0],
                level.vertices.len()
            );
            assert!(
                (face[1] as usize) < level.vertices.len(),
                "face index {} out of bounds",
                face[1]
            );
            assert!(
                (face[2] as usize) < level.vertices.len(),
                "face index {} out of bounds",
                face[2]
            );
            assert_ne!(face[0], face[1], "degenerate face: equal vertex indices");
            assert_ne!(face[1], face[2], "degenerate face: equal vertex indices");
            assert_ne!(face[0], face[2], "degenerate face: equal vertex indices");
        }
    }

    // -----------------------------------------------------------------------
    // MultiResMesh tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_multi_res_builder_default_flame_targets() {
        let builder = MultiResMeshBuilder::default_flame();
        assert_eq!(builder.target_counts, vec![5023, 2500, 1000, 500]);
    }

    #[test]
    fn test_multi_res_build_all_levels() {
        // Build a tiny mesh with enough vertices to decimate meaningfully
        // Use a grid of triangles (6 vertices, 4 faces)
        let verts = vec![
            [0.0f32, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 1.0, 0.0],
        ];
        let faces = vec![[0u32, 1, 3], [1, 4, 3], [1, 2, 4], [2, 5, 4]];

        let builder = MultiResMeshBuilder::new(vec![6, 4, 3]);
        let multi = builder.build(&verts, &faces).expect("build ok");

        assert_eq!(multi.levels.len(), 3);
        // Level 0 is full resolution
        assert_eq!(multi.levels[0].vertices.len(), 6);
    }

    #[test]
    fn test_multi_res_level_count() {
        let builder = MultiResMeshBuilder::default_flame();
        // We can test level_count without actually running decimation
        let (verts, faces) = tetrahedron();
        let builder_small = MultiResMeshBuilder::new(vec![4, 3]);
        let multi = builder_small.build(&verts, &faces).expect("build ok");
        assert_eq!(multi.level_count(), 2);

        // The default FLAME builder has 4 targets
        assert_eq!(builder.target_counts.len(), 4);
    }

    #[test]
    fn test_multi_res_vertex_counts_decreasing() {
        let verts = vec![
            [0.0f32, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 1.0, 0.0],
            [1.5, 0.5, 1.0],
        ];
        let faces = vec![
            [0u32, 1, 3],
            [1, 4, 3],
            [1, 2, 4],
            [2, 5, 4],
            [0, 1, 6],
            [1, 2, 6],
            [2, 5, 6],
            [0, 3, 6],
        ];

        let builder = MultiResMeshBuilder::new(vec![7, 5, 3]);
        let multi = builder.build(&verts, &faces).expect("build ok");

        // Vertex counts must be non-increasing across levels
        let counts: Vec<usize> = multi.levels.iter().map(|l| l.vertices.len()).collect();
        for i in 1..counts.len() {
            assert!(
                counts[i] <= counts[i - 1],
                "vertex counts not decreasing: level {i} has {} > level {} has {}",
                counts[i],
                i - 1,
                counts[i - 1]
            );
        }
    }

    #[test]
    fn test_mesh_level_faces_valid_indices() {
        let (verts, faces) = tetrahedron();
        let builder = MultiResMeshBuilder::new(vec![4, 3]);
        let multi = builder.build(&verts, &faces).expect("build ok");

        for (li, level) in multi.levels.iter().enumerate() {
            let n_verts = level.vertices.len();
            for face in &level.faces {
                assert!(
                    (face[0] as usize) < n_verts,
                    "level {li}: face[0]={} >= n_verts={n_verts}",
                    face[0]
                );
                assert!(
                    (face[1] as usize) < n_verts,
                    "level {li}: face[1]={} >= n_verts={n_verts}",
                    face[1]
                );
                assert!(
                    (face[2] as usize) < n_verts,
                    "level {li}: face[2]={} >= n_verts={n_verts}",
                    face[2]
                );
            }
        }
    }

    #[test]
    fn test_multi_res_compression_ratio() {
        let (verts, faces) = tetrahedron();
        // Single level → ratio = 1.0
        let builder_one = MultiResMeshBuilder::new(vec![4]);
        let multi_one = builder_one.build(&verts, &faces).expect("build ok");
        assert!((multi_one.compression_ratio() - 1.0).abs() < 1e-6);

        // Two levels: 4 → some smaller count
        let builder_two = MultiResMeshBuilder::new(vec![4, 3]);
        let multi_two = builder_two.build(&verts, &faces).expect("build ok");
        let ratio = multi_two.compression_ratio();
        assert!(
            ratio >= 1.0,
            "compression ratio must be >= 1.0, got {ratio}"
        );
    }

    #[test]
    fn test_vertex_map_length_equals_original() {
        let (verts, faces) = tetrahedron();
        let builder = MultiResMeshBuilder::new(vec![4, 3]);
        let multi = builder.build(&verts, &faces).expect("build ok");

        let orig_count = verts.len();
        for (li, level) in multi.levels.iter().enumerate() {
            assert_eq!(
                level.vertex_map.len(),
                orig_count,
                "level {li}: vertex_map length should equal original vertex count"
            );
        }
    }

    #[test]
    fn test_vertex_map_indices_in_range() {
        let verts = vec![
            [0.0f32, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 1.0, 0.0],
        ];
        let faces = vec![[0u32, 1, 3], [1, 4, 3], [1, 2, 4], [2, 5, 4]];

        let builder = MultiResMeshBuilder::new(vec![6, 4, 3]);
        let multi = builder.build(&verts, &faces).expect("build ok");

        for (li, level) in multi.levels.iter().enumerate() {
            let n_verts = level.vertices.len() as u32;
            for (orig_idx, &mapped) in level.vertex_map.iter().enumerate() {
                assert!(
                    mapped < n_verts,
                    "level {li}: vertex_map[{orig_idx}]={mapped} >= n_verts={n_verts}"
                );
            }
        }
    }
}
