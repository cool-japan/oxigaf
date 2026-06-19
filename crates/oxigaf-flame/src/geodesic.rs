//! Geodesic distance computation on triangle mesh surfaces.
//!
//! Provides Dijkstra-based and heat-method-based geodesic distance computation
//! for spatially-aware deformation, region growing, and proximity-based
//! influence fields on FLAME head model meshes.
//!
//! # Quick Start
//!
//! ```rust
//! use oxigaf_flame::geodesic::{GeodesicMesh, GeodesicConfig, dijkstra};
//!
//! let vertices = vec![
//!     [0.0_f32, 0.0, 0.0],
//!     [1.0, 0.0, 0.0],
//!     [0.5, 1.0, 0.0],
//! ];
//! let faces = vec![[0usize, 1, 2]];
//! let mesh = GeodesicMesh::new(vertices, faces).expect("valid mesh");
//! let config = GeodesicConfig::default();
//! let field = dijkstra(&mesh, &[0], &config).expect("dijkstra");
//! println!("Distance 0→1: {:.4}", field.distance_to(1).unwrap_or(f32::INFINITY));
//! ```

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::collections::HashSet;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during geodesic distance computation.
#[derive(Debug, thiserror::Error)]
pub enum GeodesicError {
    /// Mesh has no vertices.
    #[error("Empty mesh: no vertices")]
    EmptyMesh,

    /// Mesh has no faces.
    #[error("Empty mesh: no faces")]
    EmptyFaces,

    /// A vertex index exceeds the number of vertices in the mesh.
    #[error("Vertex index out of bounds: {idx}, mesh has {n} vertices")]
    VertexOutOfBounds { idx: usize, n: usize },

    /// A vertex is not reachable from any source vertex.
    #[error("Disconnected mesh: vertex {idx} is unreachable from source")]
    Unreachable { idx: usize },

    /// Configuration parameter is invalid.
    #[error("Invalid config: {0}")]
    InvalidConfig(String),

    /// A numerical error occurred during computation.
    #[error("Numerical error: {0}")]
    NumericalError(String),
}

// ---------------------------------------------------------------------------
// GeodesicMesh
// ---------------------------------------------------------------------------

/// Compact triangle mesh representation for geodesic computations.
///
/// Vertices are 3D positions stored as `[f32; 3]`, faces are triangles
/// stored as `[usize; 3]` vertex indices.
#[derive(Debug, Clone)]
pub struct GeodesicMesh {
    /// 3D vertex positions.
    pub vertices: Vec<[f32; 3]>,
    /// Triangle faces as vertex index triples.
    pub faces: Vec<[usize; 3]>,
}

impl GeodesicMesh {
    /// Create a new mesh, validating that face indices are in bounds.
    ///
    /// # Errors
    /// - [`GeodesicError::EmptyMesh`] if vertices is empty.
    /// - [`GeodesicError::EmptyFaces`] if faces is empty.
    /// - [`GeodesicError::VertexOutOfBounds`] if any face index ≥ vertex count.
    pub fn new(vertices: Vec<[f32; 3]>, faces: Vec<[usize; 3]>) -> Result<Self, GeodesicError> {
        if vertices.is_empty() {
            return Err(GeodesicError::EmptyMesh);
        }
        if faces.is_empty() {
            return Err(GeodesicError::EmptyFaces);
        }
        let n = vertices.len();
        for face in &faces {
            for &idx in face {
                if idx >= n {
                    return Err(GeodesicError::VertexOutOfBounds { idx, n });
                }
            }
        }
        Ok(Self { vertices, faces })
    }

    /// Number of vertices in the mesh.
    #[inline]
    #[must_use]
    pub fn n_vertices(&self) -> usize {
        self.vertices.len()
    }

    /// Number of triangular faces in the mesh.
    #[inline]
    #[must_use]
    pub fn n_faces(&self) -> usize {
        self.faces.len()
    }

    /// Build vertex adjacency list: for each vertex, the list of neighboring vertices.
    ///
    /// Edges are undirected and deduplicated.
    #[must_use]
    pub fn build_adjacency(&self) -> Vec<Vec<usize>> {
        let n = self.n_vertices();
        let mut adj: Vec<HashSet<usize>> = vec![HashSet::new(); n];
        for face in &self.faces {
            let [a, b, c] = *face;
            adj[a].insert(b);
            adj[a].insert(c);
            adj[b].insert(a);
            adj[b].insert(c);
            adj[c].insert(a);
            adj[c].insert(b);
        }
        adj.into_iter().map(|s| s.into_iter().collect()).collect()
    }

    /// Build adjacency list and edge lengths in parallel arrays.
    ///
    /// Returns `(adjacency, lengths)` where `lengths[v][i]` is the Euclidean
    /// length of the edge from vertex `v` to its `i`-th neighbor.
    #[must_use]
    pub fn build_edge_lengths(&self) -> (Vec<Vec<usize>>, Vec<Vec<f32>>) {
        let adj = self.build_adjacency();
        let lengths: Vec<Vec<f32>> = adj
            .iter()
            .enumerate()
            .map(|(v, neighbors)| neighbors.iter().map(|&u| self.edge_length(v, u)).collect())
            .collect();
        (adj, lengths)
    }

    /// Euclidean distance between two vertices.
    #[inline]
    #[must_use]
    pub fn edge_length(&self, a: usize, b: usize) -> f32 {
        let pa = self.vertices[a];
        let pb = self.vertices[b];
        let dx = pb[0] - pa[0];
        let dy = pb[1] - pa[1];
        let dz = pb[2] - pa[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Compute the centroid of a face.
    ///
    /// # Errors
    /// Returns [`GeodesicError::VertexOutOfBounds`] if `face_idx` is out of range.
    pub fn face_center(&self, face_idx: usize) -> Result<[f32; 3], GeodesicError> {
        if face_idx >= self.faces.len() {
            return Err(GeodesicError::VertexOutOfBounds {
                idx: face_idx,
                n: self.faces.len(),
            });
        }
        let [a, b, c] = self.faces[face_idx];
        let pa = self.vertices[a];
        let pb = self.vertices[b];
        let pc = self.vertices[c];
        Ok([
            (pa[0] + pb[0] + pc[0]) / 3.0,
            (pa[1] + pb[1] + pc[1]) / 3.0,
            (pa[2] + pb[2] + pc[2]) / 3.0,
        ])
    }
}

// ---------------------------------------------------------------------------
// GeodesicField
// ---------------------------------------------------------------------------

/// Geodesic distance field from one or more source vertices.
///
/// Contains per-vertex distances and predecessor information for path
/// reconstruction.
#[derive(Debug, Clone)]
pub struct GeodesicField {
    /// Geodesic distance from source(s) to each vertex. `f32::INFINITY` = unreachable.
    pub distances: Vec<f32>,
    /// For path reconstruction: predecessor of each vertex on shortest path.
    pub predecessors: Vec<Option<usize>>,
    /// Number of vertices in the mesh this field was computed for.
    pub n_vertices: usize,
}

impl GeodesicField {
    /// Geodesic distance to the given vertex.
    ///
    /// # Errors
    /// - [`GeodesicError::VertexOutOfBounds`] if `vertex >= n_vertices`.
    pub fn distance_to(&self, vertex: usize) -> Result<f32, GeodesicError> {
        if vertex >= self.n_vertices {
            return Err(GeodesicError::VertexOutOfBounds {
                idx: vertex,
                n: self.n_vertices,
            });
        }
        Ok(self.distances[vertex])
    }

    /// Return true if the vertex is reachable (distance < `f32::INFINITY`).
    #[inline]
    #[must_use]
    pub fn is_reachable(&self, vertex: usize) -> bool {
        vertex < self.n_vertices && self.distances[vertex].is_finite()
    }

    /// Reconstruct the shortest path from any source to `target`.
    ///
    /// Returns a vector of vertex indices starting at a source and ending at
    /// `target`.
    ///
    /// # Errors
    /// - [`GeodesicError::VertexOutOfBounds`] if `target >= n_vertices`.
    /// - [`GeodesicError::Unreachable`] if no path exists.
    pub fn shortest_path(&self, target: usize) -> Result<Vec<usize>, GeodesicError> {
        if target >= self.n_vertices {
            return Err(GeodesicError::VertexOutOfBounds {
                idx: target,
                n: self.n_vertices,
            });
        }
        if !self.is_reachable(target) {
            return Err(GeodesicError::Unreachable { idx: target });
        }
        let mut path = Vec::new();
        let mut current = target;
        // Guard against cycles (shouldn't happen in a correct Dijkstra but be safe)
        let limit = self.n_vertices + 1;
        let mut steps = 0;
        loop {
            path.push(current);
            match self.predecessors[current] {
                None => break,
                Some(prev) => {
                    current = prev;
                    steps += 1;
                    if steps > limit {
                        return Err(GeodesicError::NumericalError(
                            "cycle detected in predecessor array".into(),
                        ));
                    }
                }
            }
        }
        path.reverse();
        Ok(path)
    }

    /// Return the vertex with the smallest distance (the source, or index 0 on empty).
    #[must_use]
    pub fn nearest_source(&self) -> usize {
        self.distances
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(i, _)| i)
    }

    /// Maximum finite distance in the field (0.0 if no reachable vertices).
    pub fn max_distance(&self) -> f32 {
        self.distances
            .iter()
            .copied()
            .filter(|d| d.is_finite())
            .fold(0.0_f32, f32::max)
    }

    /// Mean geodesic distance over reachable vertices (0.0 if none reachable).
    #[must_use]
    pub fn mean_distance(&self) -> f32 {
        let reachable: Vec<f32> = self
            .distances
            .iter()
            .copied()
            .filter(|d| d.is_finite())
            .collect();
        if reachable.is_empty() {
            return 0.0;
        }
        reachable.iter().sum::<f32>() / reachable.len() as f32
    }

    /// All vertex indices with geodesic distance ≤ `radius`.
    #[must_use]
    pub fn within_radius(&self, radius: f32) -> Vec<usize> {
        self.distances
            .iter()
            .enumerate()
            .filter_map(|(i, &d)| if d <= radius { Some(i) } else { None })
            .collect()
    }

    /// Distances normalized to [0, 1] by dividing by the maximum finite distance.
    ///
    /// Unreachable vertices become 1.0. Returns all-zeros if max is 0.
    #[must_use]
    pub fn normalize(&self) -> Vec<f32> {
        let max = self.max_distance();
        if max == 0.0 {
            return vec![0.0; self.n_vertices];
        }
        self.distances
            .iter()
            .map(|&d| if d.is_finite() { d / max } else { 1.0 })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// GeodesicConfig
// ---------------------------------------------------------------------------

/// Configuration for geodesic distance computation.
#[derive(Debug, Clone)]
pub struct GeodesicConfig {
    /// If true, use face-based graph (more accurate). If false, use edge-graph (faster).
    pub use_face_graph: bool,
    /// Stop propagation once all settled distances exceed this threshold.
    /// Use `f32::INFINITY` for no limit.
    pub max_distance: f32,
}

impl Default for GeodesicConfig {
    fn default() -> Self {
        Self {
            use_face_graph: false,
            max_distance: f32::INFINITY,
        }
    }
}

// ---------------------------------------------------------------------------
// Internal: shared Dijkstra implementation
// ---------------------------------------------------------------------------

/// Internal result of a Dijkstra pass, augmented with source assignment.
struct DijkstraResult {
    distances: Vec<f32>,
    predecessors: Vec<Option<usize>>,
    /// For each vertex, which source index (into the sources slice) settled it.
    source_of: Vec<usize>,
}

/// Core multi-source Dijkstra algorithm.
///
/// Uses a min-heap via `BinaryHeap<Reverse<(u32, usize)>>` where the u32 stores
/// `f32::to_bits()`. Positive IEEE 754 floats sort correctly by bit pattern,
/// so this gives correct ordering without any external crate.
fn dijkstra_impl(
    n: usize,
    adj: &[Vec<usize>],
    lengths: &[Vec<f32>],
    sources: &[usize],
    max_distance: f32,
) -> DijkstraResult {
    let mut dist = vec![f32::INFINITY; n];
    let mut pred: Vec<Option<usize>> = vec![None; n];
    let mut source_of = vec![0usize; n];

    // BinaryHeap is a max-heap; Reverse turns it into a min-heap.
    // Key: (distance_bits, vertex_index)
    let mut heap: BinaryHeap<Reverse<(u32, usize)>> = BinaryHeap::new();

    for (si, &src) in sources.iter().enumerate() {
        dist[src] = 0.0;
        source_of[src] = si;
        heap.push(Reverse((0_u32, src)));
    }

    while let Some(Reverse((d_bits, u))) = heap.pop() {
        let d = f32::from_bits(d_bits);
        // Skip stale entries
        if d > dist[u] {
            continue;
        }
        // Early termination if we are past the max distance limit
        if d > max_distance {
            break;
        }
        let neighbors = &adj[u];
        let edge_lens = &lengths[u];
        for (i, &v) in neighbors.iter().enumerate() {
            let new_dist = dist[u] + edge_lens[i];
            if new_dist < dist[v] {
                dist[v] = new_dist;
                pred[v] = Some(u);
                source_of[v] = source_of[u];
                heap.push(Reverse((new_dist.to_bits(), v)));
            }
        }
    }

    DijkstraResult {
        distances: dist,
        predecessors: pred,
        source_of,
    }
}

// ---------------------------------------------------------------------------
// Public API: dijkstra / multi_source_dijkstra
// ---------------------------------------------------------------------------

/// Dijkstra's algorithm on the edge graph.
///
/// Computes shortest-path geodesic distances from one or more source vertices
/// to all reachable vertices in the mesh.
///
/// # Errors
/// - [`GeodesicError::EmptyMesh`] if the mesh has no vertices.
/// - [`GeodesicError::EmptyFaces`] if the mesh has no faces.
/// - [`GeodesicError::VertexOutOfBounds`] if any source index is out of range.
/// - [`GeodesicError::InvalidConfig`] if sources slice is empty.
pub fn dijkstra(
    mesh: &GeodesicMesh,
    sources: &[usize],
    config: &GeodesicConfig,
) -> Result<GeodesicField, GeodesicError> {
    if mesh.n_vertices() == 0 {
        return Err(GeodesicError::EmptyMesh);
    }
    if mesh.n_faces() == 0 {
        return Err(GeodesicError::EmptyFaces);
    }
    if sources.is_empty() {
        return Err(GeodesicError::InvalidConfig(
            "sources must be non-empty".into(),
        ));
    }
    let n = mesh.n_vertices();
    for &s in sources {
        if s >= n {
            return Err(GeodesicError::VertexOutOfBounds { idx: s, n });
        }
    }

    let (adj, lengths) = mesh.build_edge_lengths();
    let result = dijkstra_impl(n, &adj, &lengths, sources, config.max_distance);

    Ok(GeodesicField {
        distances: result.distances,
        predecessors: result.predecessors,
        n_vertices: n,
    })
}

/// Multi-source Dijkstra: each source starts at distance 0.
///
/// Semantically identical to [`dijkstra`] with multiple sources — included
/// as a named alias for clarity at call sites.
///
/// # Errors
/// Same as [`dijkstra`].
pub fn multi_source_dijkstra(
    mesh: &GeodesicMesh,
    sources: &[usize],
    config: &GeodesicConfig,
) -> Result<GeodesicField, GeodesicError> {
    dijkstra(mesh, sources, config)
}

// ---------------------------------------------------------------------------
// heat_geodesic
// ---------------------------------------------------------------------------

/// Approximate geodesic distances using a simplified heat-diffusion method.
///
/// Initializes heat at `source`, diffuses it for `n_iter` iterations with
/// time step `dt`, then converts the heat field to distances via
/// `distance[v] = -ln(h[v])` (shifted so source = 0).
///
/// This is a simplified approximation, not the full heat method of
/// Crane et al. For high accuracy, prefer [`dijkstra`].
///
/// # Errors
/// - [`GeodesicError::EmptyMesh`] / [`GeodesicError::EmptyFaces`] for empty inputs.
/// - [`GeodesicError::VertexOutOfBounds`] if `source` is out of range.
/// - [`GeodesicError::InvalidConfig`] if `dt` ≤ 0 or `n_iter` is 0.
pub fn heat_geodesic(
    mesh: &GeodesicMesh,
    source: usize,
    n_iter: usize,
    dt: f32,
) -> Result<GeodesicField, GeodesicError> {
    const EPS: f32 = 1e-30;

    if mesh.n_vertices() == 0 {
        return Err(GeodesicError::EmptyMesh);
    }
    if mesh.n_faces() == 0 {
        return Err(GeodesicError::EmptyFaces);
    }
    let n = mesh.n_vertices();
    if source >= n {
        return Err(GeodesicError::VertexOutOfBounds { idx: source, n });
    }
    if dt <= 0.0 {
        return Err(GeodesicError::InvalidConfig(format!(
            "dt must be positive, got {dt}"
        )));
    }
    if n_iter == 0 {
        return Err(GeodesicError::InvalidConfig("n_iter must be > 0".into()));
    }

    let adj = mesh.build_adjacency();

    // Initialise heat: 1 at source, 0 elsewhere.
    let mut heat = vec![0.0_f32; n];
    heat[source] = 1.0;

    // Iterative diffusion: h_new[v] = h[v] + dt * mean(h[neighbour] - h[v])
    let mut heat_new = vec![0.0_f32; n];
    for _ in 0..n_iter {
        for v in 0..n {
            let neighbors = &adj[v];
            let deg = neighbors.len();
            if deg == 0 {
                heat_new[v] = heat[v];
                continue;
            }
            let laplacian: f32 =
                neighbors.iter().map(|&u| heat[u] - heat[v]).sum::<f32>() / deg as f32;
            heat_new[v] = (heat[v] + dt * laplacian).clamp(0.0, 1.0);
        }
        heat.copy_from_slice(&heat_new);
    }

    // Convert heat to distance: d[v] = -ln(max(h[v], eps))
    let raw_distances: Vec<f32> = heat.iter().map(|&h| -(h.max(EPS).ln())).collect();

    // Shift so the source vertex has distance 0.
    let source_dist = raw_distances[source];
    let distances: Vec<f32> = raw_distances
        .iter()
        .map(|&d| (d - source_dist).max(0.0))
        .collect();

    // Build predecessor array (not available from heat method — set to None everywhere).
    let predecessors = vec![None; n];

    Ok(GeodesicField {
        distances,
        predecessors,
        n_vertices: n,
    })
}

// ---------------------------------------------------------------------------
// pairwise_geodesic
// ---------------------------------------------------------------------------

/// Compute pairwise geodesic distances between a set of landmark vertices.
///
/// Returns a flat, row-major `[n * n]` symmetric distance matrix.
///
/// # Errors
/// - [`GeodesicError::EmptyMesh`] / [`GeodesicError::EmptyFaces`] for empty inputs.
/// - [`GeodesicError::VertexOutOfBounds`] if any landmark index is out of range.
/// - [`GeodesicError::InvalidConfig`] if `landmarks` is empty.
pub fn pairwise_geodesic(
    mesh: &GeodesicMesh,
    landmarks: &[usize],
    config: &GeodesicConfig,
) -> Result<Vec<f32>, GeodesicError> {
    if mesh.n_vertices() == 0 {
        return Err(GeodesicError::EmptyMesh);
    }
    if mesh.n_faces() == 0 {
        return Err(GeodesicError::EmptyFaces);
    }
    if landmarks.is_empty() {
        return Err(GeodesicError::InvalidConfig(
            "landmarks must be non-empty".into(),
        ));
    }
    let n = mesh.n_vertices();
    for &l in landmarks {
        if l >= n {
            return Err(GeodesicError::VertexOutOfBounds { idx: l, n });
        }
    }

    let k = landmarks.len();
    let (adj, lengths) = mesh.build_edge_lengths();
    let mut matrix = vec![0.0_f32; k * k];

    for (i, &src) in landmarks.iter().enumerate() {
        let result = dijkstra_impl(n, &adj, &lengths, &[src], config.max_distance);
        for (j, &dst) in landmarks.iter().enumerate() {
            matrix[i * k + j] = result.distances[dst];
        }
    }

    Ok(matrix)
}

// ---------------------------------------------------------------------------
// geodesic_voronoi
// ---------------------------------------------------------------------------

/// Voronoi segmentation: assign each vertex to its nearest source.
///
/// Returns a `Vec<usize>` of length `n_vertices` where the value at each
/// position is the index (into `sources`) of the nearest source vertex.
///
/// # Errors
/// - [`GeodesicError::EmptyMesh`] / [`GeodesicError::EmptyFaces`] for empty inputs.
/// - [`GeodesicError::VertexOutOfBounds`] if any source index is out of range.
/// - [`GeodesicError::InvalidConfig`] if `sources` is empty.
pub fn geodesic_voronoi(
    mesh: &GeodesicMesh,
    sources: &[usize],
    config: &GeodesicConfig,
) -> Result<Vec<usize>, GeodesicError> {
    if mesh.n_vertices() == 0 {
        return Err(GeodesicError::EmptyMesh);
    }
    if mesh.n_faces() == 0 {
        return Err(GeodesicError::EmptyFaces);
    }
    if sources.is_empty() {
        return Err(GeodesicError::InvalidConfig(
            "sources must be non-empty".into(),
        ));
    }
    let n = mesh.n_vertices();
    for &s in sources {
        if s >= n {
            return Err(GeodesicError::VertexOutOfBounds { idx: s, n });
        }
    }

    let (adj, lengths) = mesh.build_edge_lengths();
    let result = dijkstra_impl(n, &adj, &lengths, sources, config.max_distance);
    Ok(result.source_of)
}

// ---------------------------------------------------------------------------
// geodesic_weights
// ---------------------------------------------------------------------------

/// Compute Gaussian geodesic smoothing weights.
///
/// For each vertex `v`:
/// ```text
/// w[v] = exp(-dist(v)^2 / (2 * sigma^2))
/// ```
///
/// Returns a `Vec<f32>` of length `n_vertices`. Unreachable vertices (infinite
/// distance) receive weight 0.
#[must_use]
pub fn geodesic_weights(field: &GeodesicField, sigma: f32) -> Vec<f32> {
    let two_sigma_sq = 2.0 * sigma * sigma;
    field
        .distances
        .iter()
        .map(|&d| {
            if d.is_finite() {
                (-(d * d) / two_sigma_sq).exp()
            } else {
                0.0
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// geodesic_ball
// ---------------------------------------------------------------------------

/// Find all vertices within geodesic distance `radius` from `source`.
///
/// The source vertex itself is always included (distance = 0).
///
/// # Errors
/// - [`GeodesicError::EmptyMesh`] / [`GeodesicError::EmptyFaces`] for empty inputs.
/// - [`GeodesicError::VertexOutOfBounds`] if `source >= n_vertices`.
pub fn geodesic_ball(
    mesh: &GeodesicMesh,
    source: usize,
    radius: f32,
    config: &GeodesicConfig,
) -> Result<Vec<usize>, GeodesicError> {
    let mut cfg = config.clone();
    // Use the smaller of the two limits so Dijkstra can terminate early.
    cfg.max_distance = cfg.max_distance.min(radius);
    let field = dijkstra(mesh, &[source], &cfg)?;
    Ok(field.within_radius(radius))
}

// ---------------------------------------------------------------------------
// geodesic_diameter
// ---------------------------------------------------------------------------

/// Approximate geodesic diameter: the maximum geodesic distance between any
/// two vertices.
///
/// Uses a two-pass heuristic: run Dijkstra from vertex 0, take the farthest
/// vertex, then run Dijkstra again from that vertex. The maximum of the second
/// pass approximates the diameter.
///
/// # Errors
/// - [`GeodesicError::EmptyMesh`] / [`GeodesicError::EmptyFaces`] for empty inputs.
pub fn geodesic_diameter(
    mesh: &GeodesicMesh,
    config: &GeodesicConfig,
) -> Result<f32, GeodesicError> {
    if mesh.n_vertices() == 0 {
        return Err(GeodesicError::EmptyMesh);
    }
    if mesh.n_faces() == 0 {
        return Err(GeodesicError::EmptyFaces);
    }
    // Single vertex mesh → diameter = 0
    if mesh.n_vertices() == 1 {
        return Ok(0.0);
    }

    // First pass from vertex 0
    let field1 = dijkstra(mesh, &[0], config)?;
    let farthest = field1
        .distances
        .iter()
        .enumerate()
        .filter(|(_, d)| d.is_finite())
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map_or(0, |(i, _)| i);

    // Second pass from the farthest vertex
    let field2 = dijkstra(mesh, &[farthest], config)?;
    Ok(field2.max_distance())
}

// ---------------------------------------------------------------------------
// geodesic_center
// ---------------------------------------------------------------------------

/// Find the vertex that minimizes the total geodesic distance to all others.
///
/// Searches over `sample_vertices`; if empty, searches all vertices.
///
/// # Errors
/// - [`GeodesicError::EmptyMesh`] / [`GeodesicError::EmptyFaces`] for empty inputs.
pub fn geodesic_center(
    mesh: &GeodesicMesh,
    sample_vertices: &[usize],
    config: &GeodesicConfig,
) -> Result<usize, GeodesicError> {
    if mesh.n_vertices() == 0 {
        return Err(GeodesicError::EmptyMesh);
    }
    if mesh.n_faces() == 0 {
        return Err(GeodesicError::EmptyFaces);
    }
    let candidates: Vec<usize> = if sample_vertices.is_empty() {
        (0..mesh.n_vertices()).collect()
    } else {
        sample_vertices.to_vec()
    };

    // Validate candidates
    let n = mesh.n_vertices();
    for &v in &candidates {
        if v >= n {
            return Err(GeodesicError::VertexOutOfBounds { idx: v, n });
        }
    }

    let (adj, lengths) = mesh.build_edge_lengths();

    let mut best_vertex = candidates[0];
    let mut best_sum = f32::INFINITY;

    for &v in &candidates {
        let result = dijkstra_impl(n, &adj, &lengths, &[v], config.max_distance);
        let sum: f32 = result.distances.iter().filter(|d| d.is_finite()).sum();
        if sum < best_sum {
            best_sum = sum;
            best_vertex = v;
        }
    }

    Ok(best_vertex)
}

// ---------------------------------------------------------------------------
// smooth_geodesic_path
// ---------------------------------------------------------------------------

/// Smooth a discrete vertex path using neighborhood averaging.
///
/// For each vertex on the path, replaces its position with the average of
/// itself and its two path neighbors. Runs `smoothing_iters` passes.
///
/// Returns smoothed 3D positions (not vertex indices).
///
/// # Errors
/// - [`GeodesicError::VertexOutOfBounds`] if any path vertex is out of range.
pub fn smooth_geodesic_path(
    mesh: &GeodesicMesh,
    path: &[usize],
    smoothing_iters: usize,
) -> Result<Vec<[f32; 3]>, GeodesicError> {
    if path.is_empty() {
        return Ok(Vec::new());
    }
    let num_verts = mesh.n_vertices();
    for &v in path {
        if v >= num_verts {
            return Err(GeodesicError::VertexOutOfBounds {
                idx: v,
                n: num_verts,
            });
        }
    }

    // Lift path to 3D positions
    let mut positions: Vec<[f32; 3]> = path.iter().map(|&v| mesh.vertices[v]).collect();
    let path_len = positions.len();

    for _ in 0..smoothing_iters {
        let prev = positions.clone();
        for i in 0..path_len {
            if i == 0 || i == path_len - 1 {
                // Endpoints are pinned
                continue;
            }
            let prev_pos = prev[i - 1];
            let curr_pos = prev[i];
            let next_pos = prev[i + 1];
            positions[i] = [
                (prev_pos[0] + curr_pos[0] + next_pos[0]) / 3.0,
                (prev_pos[1] + curr_pos[1] + next_pos[1]) / 3.0,
                (prev_pos[2] + curr_pos[2] + next_pos[2]) / 3.0,
            ];
        }
    }

    Ok(positions)
}

// ---------------------------------------------------------------------------
// GeodesicStats / compute_geodesic_stats
// ---------------------------------------------------------------------------

/// Descriptive statistics about a geodesic distance field.
#[derive(Debug, Clone)]
pub struct GeodesicStats {
    /// Number of vertices reachable from the source(s).
    pub n_reachable: usize,
    /// Minimum geodesic distance among reachable vertices (0 for source vertex).
    pub min_distance: f32,
    /// Maximum geodesic distance among reachable vertices.
    pub max_distance: f32,
    /// Mean geodesic distance over reachable vertices.
    pub mean_distance: f32,
    /// Population standard deviation of geodesic distances (reachable only).
    pub std_distance: f32,
    /// Geodesic diameter estimate (same as max in single-source case).
    pub diameter: f32,
}

/// Compute descriptive statistics about a geodesic distance field.
pub fn compute_geodesic_stats(field: &GeodesicField) -> GeodesicStats {
    let reachable: Vec<f32> = field
        .distances
        .iter()
        .copied()
        .filter(|d| d.is_finite())
        .collect();

    let n_reachable = reachable.len();
    if n_reachable == 0 {
        return GeodesicStats {
            n_reachable: 0,
            min_distance: 0.0,
            max_distance: 0.0,
            mean_distance: 0.0,
            std_distance: 0.0,
            diameter: 0.0,
        };
    }

    let min_distance = reachable.iter().copied().fold(f32::INFINITY, f32::min);
    let max_distance = reachable.iter().copied().fold(0.0_f32, f32::max);
    let mean_distance = reachable.iter().sum::<f32>() / n_reachable as f32;
    let variance = reachable
        .iter()
        .map(|&d| {
            let diff = d - mean_distance;
            diff * diff
        })
        .sum::<f32>()
        / n_reachable as f32;
    let std_distance = variance.sqrt();

    GeodesicStats {
        n_reachable,
        min_distance,
        max_distance,
        mean_distance,
        std_distance,
        diameter: max_distance,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Shared test meshes
    // -----------------------------------------------------------------------

    /// Single triangle mesh: vertices at (0,0,0), (1,0,0), (0,1,0).
    fn triangle_mesh() -> GeodesicMesh {
        GeodesicMesh::new(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            vec![[0, 1, 2]],
        )
        .expect("valid triangle mesh")
    }

    /// Linear chain mesh: 4 vertices connected as 0-1-2-3 via triangles
    /// (using degenerate-ish strips). We create two triangles: [0,1,2] and [1,2,3].
    fn chain_mesh() -> GeodesicMesh {
        // 0--1--2--3 as a strip of two triangles
        GeodesicMesh::new(
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ],
            vec![[0, 1, 2], [1, 2, 3]],
        )
        .expect("valid chain mesh")
    }

    /// Two-triangle mesh with 4 vertices forming a square split diagonally.
    fn square_mesh() -> GeodesicMesh {
        GeodesicMesh::new(
            vec![
                [0.0, 0.0, 0.0], // 0
                [1.0, 0.0, 0.0], // 1
                [1.0, 1.0, 0.0], // 2
                [0.0, 1.0, 0.0], // 3
            ],
            vec![[0, 1, 2], [0, 2, 3]],
        )
        .expect("valid square mesh")
    }

    // -----------------------------------------------------------------------
    // GeodesicMesh::new
    // -----------------------------------------------------------------------

    #[test]
    fn test_mesh_new_valid() {
        let mesh = triangle_mesh();
        assert_eq!(mesh.n_vertices(), 3);
        assert_eq!(mesh.n_faces(), 1);
    }

    #[test]
    fn test_mesh_new_empty_vertices() {
        let result = GeodesicMesh::new(vec![], vec![[0, 1, 2]]);
        assert!(matches!(result, Err(GeodesicError::EmptyMesh)));
    }

    #[test]
    fn test_mesh_new_empty_faces() {
        let result = GeodesicMesh::new(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            vec![],
        );
        assert!(matches!(result, Err(GeodesicError::EmptyFaces)));
    }

    #[test]
    fn test_mesh_new_out_of_bounds_face() {
        let result = GeodesicMesh::new(vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], vec![[0, 1, 99]]);
        assert!(matches!(
            result,
            Err(GeodesicError::VertexOutOfBounds { idx: 99, n: 2 })
        ));
    }

    // -----------------------------------------------------------------------
    // GeodesicMesh::build_adjacency
    // -----------------------------------------------------------------------

    #[test]
    fn test_build_adjacency_triangle() {
        let mesh = triangle_mesh();
        let adj = mesh.build_adjacency();
        assert_eq!(adj.len(), 3);
        // Each vertex should have exactly 2 neighbors in a single triangle
        for (i, neighbors) in adj.iter().enumerate() {
            assert_eq!(neighbors.len(), 2, "vertex {i} should have 2 neighbors");
        }
        // Vertex 0 should be adjacent to 1 and 2
        let adj0: HashSet<usize> = adj[0].iter().copied().collect();
        assert!(adj0.contains(&1));
        assert!(adj0.contains(&2));
    }

    #[test]
    fn test_build_adjacency_chain() {
        let mesh = chain_mesh();
        let adj = mesh.build_adjacency();
        // Vertex 0: connected to 1, 2 (from face [0,1,2])
        let adj0: HashSet<usize> = adj[0].iter().copied().collect();
        assert!(adj0.contains(&1));
        assert!(adj0.contains(&2));
        // Vertex 3: connected to 1, 2 (from face [1,2,3])
        let adj3: HashSet<usize> = adj[3].iter().copied().collect();
        assert!(adj3.contains(&1));
        assert!(adj3.contains(&2));
        // Vertex 1 and 2 are interior; connected to each other and to both 0 and 3
        let adj1: HashSet<usize> = adj[1].iter().copied().collect();
        assert!(adj1.contains(&0) && adj1.contains(&2) && adj1.contains(&3));
    }

    // -----------------------------------------------------------------------
    // GeodesicMesh::edge_length
    // -----------------------------------------------------------------------

    #[test]
    fn test_edge_length_unit() {
        let mesh = triangle_mesh();
        let len = mesh.edge_length(0, 1);
        assert!(
            (len - 1.0).abs() < 1e-6,
            "distance 0→1 should be 1.0, got {len}"
        );
    }

    #[test]
    fn test_edge_length_diagonal() {
        let mesh = square_mesh();
        // Diagonal from (0,0,0) to (1,1,0)
        let len = mesh.edge_length(0, 2);
        assert!((len - std::f32::consts::SQRT_2).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // GeodesicMesh::face_center
    // -----------------------------------------------------------------------

    #[test]
    fn test_face_center_triangle() {
        let mesh = triangle_mesh();
        let center = mesh.face_center(0).expect("valid face");
        // Centroid of (0,0,0), (1,0,0), (0,1,0) = (1/3, 1/3, 0)
        assert!((center[0] - 1.0 / 3.0).abs() < 1e-6);
        assert!((center[1] - 1.0 / 3.0).abs() < 1e-6);
        assert!(center[2].abs() < 1e-6);
    }

    #[test]
    fn test_face_center_out_of_bounds() {
        let mesh = triangle_mesh();
        assert!(matches!(
            mesh.face_center(99),
            Err(GeodesicError::VertexOutOfBounds { .. })
        ));
    }

    // -----------------------------------------------------------------------
    // dijkstra: basic correctness
    // -----------------------------------------------------------------------

    #[test]
    fn test_dijkstra_triangle_distances() {
        let mesh = triangle_mesh();
        let config = GeodesicConfig::default();
        let field = dijkstra(&mesh, &[0], &config).expect("dijkstra ok");
        assert_eq!(field.n_vertices, 3);
        // Source vertex has distance 0
        assert_eq!(field.distance_to(0).expect("ok"), 0.0);
        // Distance 0→1 = 1.0 (along x-axis)
        let d01 = field.distance_to(1).expect("ok");
        assert!((d01 - 1.0).abs() < 1e-6, "d01={d01}");
        // Distance 0→2 = 1.0 (along y-axis)
        let d02 = field.distance_to(2).expect("ok");
        assert!((d02 - 1.0).abs() < 1e-6, "d02={d02}");
    }

    #[test]
    fn test_dijkstra_invalid_source() {
        let mesh = triangle_mesh();
        let config = GeodesicConfig::default();
        let result = dijkstra(&mesh, &[99], &config);
        assert!(matches!(
            result,
            Err(GeodesicError::VertexOutOfBounds { idx: 99, n: 3 })
        ));
    }

    #[test]
    fn test_dijkstra_empty_sources() {
        let mesh = triangle_mesh();
        let config = GeodesicConfig::default();
        let result = dijkstra(&mesh, &[], &config);
        assert!(matches!(result, Err(GeodesicError::InvalidConfig(_))));
    }

    #[test]
    fn test_dijkstra_chain_distances() {
        let mesh = chain_mesh();
        let config = GeodesicConfig::default();
        let field = dijkstra(&mesh, &[0], &config).expect("ok");
        // Source at 0
        assert_eq!(field.distance_to(0).expect("ok"), 0.0);
        // Distance 0→1 = 1.0
        let d01 = field.distance_to(1).expect("ok");
        assert!((d01 - 1.0).abs() < 1e-5, "d01={d01}");
        // Distance 0→3 ≥ 2.0 (shortest path through intermediate vertices)
        let d03 = field.distance_to(3).expect("ok");
        assert!(d03 >= 2.0 - 1e-5, "d03={d03} should be >= 2.0");
    }

    // -----------------------------------------------------------------------
    // GeodesicField methods
    // -----------------------------------------------------------------------

    #[test]
    fn test_field_distance_to_out_of_bounds() {
        let mesh = triangle_mesh();
        let config = GeodesicConfig::default();
        let field = dijkstra(&mesh, &[0], &config).expect("ok");
        assert!(matches!(
            field.distance_to(99),
            Err(GeodesicError::VertexOutOfBounds { idx: 99, n: 3 })
        ));
    }

    #[test]
    fn test_field_is_reachable() {
        let mesh = triangle_mesh();
        let config = GeodesicConfig::default();
        let field = dijkstra(&mesh, &[0], &config).expect("ok");
        assert!(field.is_reachable(0));
        assert!(field.is_reachable(1));
        assert!(field.is_reachable(2));
        assert!(!field.is_reachable(99)); // out-of-bounds treated as unreachable
    }

    #[test]
    fn test_field_shortest_path_one_hop() {
        let mesh = triangle_mesh();
        let config = GeodesicConfig::default();
        let field = dijkstra(&mesh, &[0], &config).expect("ok");
        let path = field.shortest_path(1).expect("ok");
        assert_eq!(path.first(), Some(&0));
        assert_eq!(path.last(), Some(&1));
    }

    #[test]
    fn test_field_shortest_path_source_itself() {
        let mesh = triangle_mesh();
        let config = GeodesicConfig::default();
        let field = dijkstra(&mesh, &[0], &config).expect("ok");
        let path = field.shortest_path(0).expect("ok");
        assert_eq!(path, vec![0]);
    }

    #[test]
    fn test_field_shortest_path_unreachable() {
        // Manually craft a field with infinite distance
        let field = GeodesicField {
            distances: vec![0.0, f32::INFINITY],
            predecessors: vec![None, None],
            n_vertices: 2,
        };
        assert!(matches!(
            field.shortest_path(1),
            Err(GeodesicError::Unreachable { idx: 1 })
        ));
    }

    #[test]
    fn test_field_within_radius() {
        let mesh = triangle_mesh();
        let config = GeodesicConfig::default();
        let field = dijkstra(&mesh, &[0], &config).expect("ok");
        // Radius 0.0: only source
        let r0 = field.within_radius(0.0);
        assert_eq!(r0, vec![0]);
        // Radius 1.5: all vertices (distances are 0, 1, 1)
        let r15 = field.within_radius(1.5);
        assert_eq!(r15.len(), 3);
    }

    #[test]
    fn test_field_normalize() {
        let mesh = triangle_mesh();
        let config = GeodesicConfig::default();
        let field = dijkstra(&mesh, &[0], &config).expect("ok");
        let normalized = field.normalize();
        assert_eq!(normalized.len(), 3);
        // Source is 0.0, max becomes 1.0
        assert!((normalized[0] - 0.0).abs() < 1e-6);
        // Both vertex 1 and 2 have distance 1.0 = max, so normalized = 1.0
        assert!((normalized[1] - 1.0).abs() < 1e-6);
        assert!((normalized[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_field_max_mean_distance() {
        let mesh = chain_mesh();
        let config = GeodesicConfig::default();
        let field = dijkstra(&mesh, &[0], &config).expect("ok");
        let max = field.max_distance();
        assert!(max > 0.0);
        let mean = field.mean_distance();
        assert!(mean > 0.0 && mean <= max);
    }

    // -----------------------------------------------------------------------
    // multi_source_dijkstra
    // -----------------------------------------------------------------------

    #[test]
    fn test_multi_source_dijkstra_two_sources() {
        let mesh = chain_mesh();
        let config = GeodesicConfig::default();
        // Sources at both ends: 0 and 3
        let field = multi_source_dijkstra(&mesh, &[0, 3], &config).expect("ok");
        // Both sources should have distance 0
        assert_eq!(field.distance_to(0).expect("ok"), 0.0);
        assert_eq!(field.distance_to(3).expect("ok"), 0.0);
        // Interior vertices should be closer than if only one source
        let d1 = field.distance_to(1).expect("ok");
        let d2 = field.distance_to(2).expect("ok");
        assert!(d1 < 2.0);
        assert!(d2 < 2.0);
    }

    #[test]
    fn test_multi_source_dijkstra_nearest_source() {
        let mesh = chain_mesh();
        let config = GeodesicConfig::default();
        let field = multi_source_dijkstra(&mesh, &[0, 3], &config).expect("ok");
        // nearest_source should be a source vertex (distance 0)
        let nearest = field.nearest_source();
        assert!(nearest == 0 || nearest == 3);
    }

    // -----------------------------------------------------------------------
    // pairwise_geodesic
    // -----------------------------------------------------------------------

    #[test]
    fn test_pairwise_geodesic_single_landmark() {
        let mesh = triangle_mesh();
        let config = GeodesicConfig::default();
        let matrix = pairwise_geodesic(&mesh, &[0], &config).expect("ok");
        assert_eq!(matrix.len(), 1);
        assert_eq!(matrix[0], 0.0);
    }

    #[test]
    fn test_pairwise_geodesic_two_landmarks() {
        let mesh = triangle_mesh();
        let config = GeodesicConfig::default();
        let matrix = pairwise_geodesic(&mesh, &[0, 1], &config).expect("ok");
        // 2x2 matrix, row-major
        assert_eq!(matrix.len(), 4);
        // Diagonal is 0
        assert_eq!(matrix[0], 0.0); // [0,0]
        assert_eq!(matrix[3], 0.0); // [1,1]
                                    // Off-diagonal should be symmetric
        assert!(
            (matrix[1] - matrix[2]).abs() < 1e-5,
            "symmetric: {}",
            (matrix[1] - matrix[2]).abs()
        );
    }

    #[test]
    fn test_pairwise_geodesic_symmetric() {
        let mesh = square_mesh();
        let config = GeodesicConfig::default();
        let landmarks = vec![0, 1, 2, 3];
        let matrix = pairwise_geodesic(&mesh, &landmarks, &config).expect("ok");
        let k = 4;
        for i in 0..k {
            for j in 0..k {
                let d_ij = matrix[i * k + j];
                let d_ji = matrix[j * k + i];
                assert!(
                    (d_ij - d_ji).abs() < 1e-5,
                    "asymmetric: d[{i},{j}]={d_ij} vs d[{j},{i}]={d_ji}"
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // geodesic_voronoi
    // -----------------------------------------------------------------------

    #[test]
    fn test_geodesic_voronoi_single_source() {
        let mesh = triangle_mesh();
        let config = GeodesicConfig::default();
        let labels = geodesic_voronoi(&mesh, &[0], &config).expect("ok");
        assert_eq!(labels.len(), 3);
        assert!(labels.iter().all(|&l| l == 0));
    }

    #[test]
    fn test_geodesic_voronoi_two_sources() {
        let mesh = chain_mesh();
        let config = GeodesicConfig::default();
        let labels = geodesic_voronoi(&mesh, &[0, 3], &config).expect("ok");
        assert_eq!(labels.len(), 4);
        // Each vertex gets one of two labels (0 or 1)
        assert!(labels.iter().all(|&l| l == 0 || l == 1));
    }

    // -----------------------------------------------------------------------
    // geodesic_weights
    // -----------------------------------------------------------------------

    #[test]
    fn test_geodesic_weights_large_sigma() {
        let mesh = triangle_mesh();
        let config = GeodesicConfig::default();
        let field = dijkstra(&mesh, &[0], &config).expect("ok");
        // Very large sigma → weights approach 1.0 uniformly
        let weights = geodesic_weights(&field, 1000.0);
        for w in &weights {
            assert!((w - 1.0).abs() < 0.01, "w={w}");
        }
    }

    #[test]
    fn test_geodesic_weights_source_highest() {
        let mesh = triangle_mesh();
        let config = GeodesicConfig::default();
        let field = dijkstra(&mesh, &[0], &config).expect("ok");
        let weights = geodesic_weights(&field, 0.5);
        // Source (dist=0) → weight=1.0, others lower
        assert!(
            (weights[0] - 1.0).abs() < 1e-6,
            "source weight={}",
            weights[0]
        );
        assert!(weights[0] >= weights[1]);
        assert!(weights[0] >= weights[2]);
    }

    // -----------------------------------------------------------------------
    // geodesic_ball
    // -----------------------------------------------------------------------

    #[test]
    fn test_geodesic_ball_source_included() {
        let mesh = triangle_mesh();
        let config = GeodesicConfig::default();
        let ball = geodesic_ball(&mesh, 0, 0.5, &config).expect("ok");
        assert!(ball.contains(&0), "source must be in ball");
    }

    #[test]
    fn test_geodesic_ball_radius_zero() {
        let mesh = triangle_mesh();
        let config = GeodesicConfig::default();
        let ball = geodesic_ball(&mesh, 0, 0.0, &config).expect("ok");
        assert_eq!(ball, vec![0]);
    }

    #[test]
    fn test_geodesic_ball_all_vertices() {
        let mesh = triangle_mesh();
        let config = GeodesicConfig::default();
        // Radius larger than diameter → all 3 vertices
        let ball = geodesic_ball(&mesh, 0, 10.0, &config).expect("ok");
        assert_eq!(ball.len(), 3);
    }

    // -----------------------------------------------------------------------
    // geodesic_diameter
    // -----------------------------------------------------------------------

    #[test]
    fn test_geodesic_diameter_triangle() {
        let mesh = triangle_mesh();
        let config = GeodesicConfig::default();
        let diam = geodesic_diameter(&mesh, &config).expect("ok");
        // Max distance in equilateral-ish triangle is 1.0
        assert!(diam > 0.0);
        assert!(diam <= 2.0);
    }

    #[test]
    fn test_geodesic_diameter_chain() {
        let mesh = chain_mesh();
        let config = GeodesicConfig::default();
        let diam = geodesic_diameter(&mesh, &config).expect("ok");
        // Chain 0-1-2-3, min path from 0 to 3 is at least 2.0
        assert!(diam >= 2.0 - 1e-5, "diameter={diam}");
    }

    // -----------------------------------------------------------------------
    // geodesic_center
    // -----------------------------------------------------------------------

    #[test]
    fn test_geodesic_center_triangle() {
        let mesh = triangle_mesh();
        let config = GeodesicConfig::default();
        let center = geodesic_center(&mesh, &[], &config).expect("ok");
        // All vertices are symmetric in the triangle — any is valid
        assert!(center < 3);
    }

    #[test]
    fn test_geodesic_center_with_sample() {
        let mesh = square_mesh();
        let config = GeodesicConfig::default();
        // Restrict to vertices 0 and 2
        let center = geodesic_center(&mesh, &[0, 2], &config).expect("ok");
        assert!(center == 0 || center == 2);
    }

    // -----------------------------------------------------------------------
    // smooth_geodesic_path
    // -----------------------------------------------------------------------

    #[test]
    fn test_smooth_geodesic_path_empty() {
        let mesh = triangle_mesh();
        let result = smooth_geodesic_path(&mesh, &[], 5).expect("ok");
        assert!(result.is_empty());
    }

    #[test]
    fn test_smooth_geodesic_path_single_vertex() {
        let mesh = triangle_mesh();
        let result = smooth_geodesic_path(&mesh, &[1], 3).expect("ok");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], mesh.vertices[1]);
    }

    #[test]
    fn test_smooth_geodesic_path_endpoints_pinned() {
        let mesh = chain_mesh();
        let path = vec![0, 1, 2, 3];
        let result = smooth_geodesic_path(&mesh, &path, 5).expect("ok");
        assert_eq!(result.len(), 4);
        // Endpoints should remain unchanged
        assert_eq!(result[0], mesh.vertices[0]);
        assert_eq!(result[3], mesh.vertices[3]);
    }

    #[test]
    fn test_smooth_geodesic_path_out_of_bounds() {
        let mesh = triangle_mesh();
        let result = smooth_geodesic_path(&mesh, &[0, 99], 1);
        assert!(matches!(
            result,
            Err(GeodesicError::VertexOutOfBounds { idx: 99, n: 3 })
        ));
    }

    // -----------------------------------------------------------------------
    // compute_geodesic_stats
    // -----------------------------------------------------------------------

    #[test]
    fn test_compute_geodesic_stats_known_distances() {
        let field = GeodesicField {
            distances: vec![0.0, 1.0, 2.0],
            predecessors: vec![None, Some(0), Some(1)],
            n_vertices: 3,
        };
        let stats = compute_geodesic_stats(&field);
        assert_eq!(stats.n_reachable, 3);
        assert!((stats.min_distance - 0.0).abs() < 1e-6);
        assert!((stats.max_distance - 2.0).abs() < 1e-6);
        assert!((stats.mean_distance - 1.0).abs() < 1e-6);
        // std of [0, 1, 2] = sqrt(2/3) ≈ 0.8165
        assert!((stats.std_distance - (2.0_f32 / 3.0).sqrt()).abs() < 1e-5);
    }

    #[test]
    fn test_compute_geodesic_stats_with_inf() {
        let field = GeodesicField {
            distances: vec![0.0, 1.0, f32::INFINITY],
            predecessors: vec![None, Some(0), None],
            n_vertices: 3,
        };
        let stats = compute_geodesic_stats(&field);
        assert_eq!(stats.n_reachable, 2);
        assert!((stats.max_distance - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_geodesic_stats_empty_field() {
        let field = GeodesicField {
            distances: vec![f32::INFINITY],
            predecessors: vec![None],
            n_vertices: 1,
        };
        let stats = compute_geodesic_stats(&field);
        assert_eq!(stats.n_reachable, 0);
        assert_eq!(stats.mean_distance, 0.0);
    }

    // -----------------------------------------------------------------------
    // heat_geodesic
    // -----------------------------------------------------------------------

    #[test]
    fn test_heat_geodesic_output_size() {
        let mesh = triangle_mesh();
        let field = heat_geodesic(&mesh, 0, 100, 0.1).expect("ok");
        assert_eq!(field.n_vertices, 3);
        assert_eq!(field.distances.len(), 3);
    }

    #[test]
    fn test_heat_geodesic_source_min_distance() {
        let mesh = triangle_mesh();
        let field = heat_geodesic(&mesh, 0, 200, 0.05).expect("ok");
        // Source should have the minimum distance (≈ 0)
        let d0 = field.distance_to(0).expect("ok");
        assert!(d0 < 1e-4, "source distance should be ~0, got {d0}");
        // Other vertices should have non-negative distances
        for i in 0..3 {
            let d = field.distance_to(i).expect("ok");
            assert!(d >= 0.0, "distance to {i} should be >= 0, got {d}");
        }
    }

    #[test]
    fn test_heat_geodesic_invalid_dt() {
        let mesh = triangle_mesh();
        let result = heat_geodesic(&mesh, 0, 10, -0.1);
        assert!(matches!(result, Err(GeodesicError::InvalidConfig(_))));
    }

    #[test]
    fn test_heat_geodesic_zero_iters() {
        let mesh = triangle_mesh();
        let result = heat_geodesic(&mesh, 0, 0, 0.1);
        assert!(matches!(result, Err(GeodesicError::InvalidConfig(_))));
    }
}
