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

mod heat_method;

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::collections::HashMap;
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
    /// Which graph the Dijkstra-based functions propagate over.
    ///
    /// - `false` (default) — the **edge graph**: vertices connected by mesh
    ///   edges, weighted by edge length. Fastest, but paths are constrained to
    ///   mesh edges, which systematically over-estimates true geodesic
    ///   distance (up to roughly 40% on an irregular triangulation).
    /// - `true` — the **face-augmented graph**: the edge graph plus one
    ///   through-face shortcut per interior edge, obtained by unfolding the two
    ///   triangles sharing that edge into a common plane and connecting their
    ///   apexes by the straight line across it. Paths may then cut across a
    ///   triangle pair instead of detouring around their shared edge, which
    ///   removes most of the edge graph's bias — on a regular grid it cuts the
    ///   mean error from roughly 4% to under 1% — for one extra edge per
    ///   interior edge.
    ///
    /// Both modes search over the mesh's own vertices only, so distances and
    /// reconstructed paths are directly comparable between them.
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

/// The weighted graph the Dijkstra passes run over.
///
/// Nodes are always exactly the mesh's own vertices. What
/// [`GeodesicConfig::use_face_graph`] changes is which *edges* exist: the edge
/// graph has only mesh edges, while the face-augmented graph adds one
/// through-face shortcut per interior edge (see [`SearchGraph::build`]).
struct SearchGraph {
    /// Neighbour list per vertex.
    adjacency: Vec<Vec<usize>>,
    /// Edge weight per neighbour, parallel to `adjacency`.
    lengths: Vec<Vec<f32>>,
    /// Number of mesh vertices (== number of nodes).
    n_vertices: usize,
}

impl SearchGraph {
    /// Build the graph `config` selects.
    ///
    /// The face-augmented variant uses **edge unfolding**: for each interior
    /// edge `(b, c)` shared by triangles `(b, c, a)` and `(b, c, d)`, the two
    /// triangles are flattened into a common plane about their shared edge and
    /// a shortcut `a — d` is added whose weight is the straight-line distance
    /// *in that unfolded plane*. That straight line is the true geodesic across
    /// the pair whenever it actually crosses the shared edge, which is the
    /// condition checked below; when it does not, no shortcut is added and the
    /// path correctly stays on mesh edges.
    ///
    /// A shortcut is never longer than the two-edge detour it replaces (that is
    /// the triangle inequality in the unfolded plane), so face-graph distances
    /// are always ≤ edge-graph distances and are typically several times more
    /// accurate. Because the shortcut connects two existing vertices, no
    /// auxiliary nodes are introduced and reconstructed paths remain sequences
    /// of mesh vertices.
    fn build(mesh: &GeodesicMesh, config: &GeodesicConfig) -> Self {
        let n_vertices = mesh.n_vertices();
        let (mut adjacency, mut lengths) = mesh.build_edge_lengths();

        if !config.use_face_graph {
            return Self {
                adjacency,
                lengths,
                n_vertices,
            };
        }

        // Map each undirected edge to the vertices opposite it, one per
        // incident triangle. Exactly two means an interior (manifold) edge.
        let mut opposite: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
        for face in &mesh.faces {
            let f = *face;
            for k in 0..3 {
                let b = f[k];
                let c = f[(k + 1) % 3];
                let a = f[(k + 2) % 3];
                let key = (b.min(c), b.max(c));
                opposite.entry(key).or_default().push(a);
            }
        }

        for (&(b, c), opps) in &opposite {
            // Only manifold interior edges unfold; boundary (1) and non-manifold
            // (3+) edges are left to the plain edge graph.
            if opps.len() != 2 {
                continue;
            }
            let (a, d) = (opps[0], opps[1]);
            if a == d {
                continue;
            }

            let pb = mesh.vertices[b];
            let pc = mesh.vertices[c];
            let axis = [pc[0] - pb[0], pc[1] - pb[1], pc[2] - pb[2]];
            let edge_len = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
            if edge_len < 1e-12 {
                continue;
            }
            let u = [axis[0] / edge_len, axis[1] / edge_len, axis[2] / edge_len];

            // Place both apexes in 2D: x along the shared edge, y the distance
            // from it. Unfolding lays them on opposite sides of the edge, which
            // is why the shortcut length below adds the two y values.
            let planar = |v: usize| -> (f32, f32) {
                let p = mesh.vertices[v];
                let w = [p[0] - pb[0], p[1] - pb[1], p[2] - pb[2]];
                let x = w[0] * u[0] + w[1] * u[1] + w[2] * u[2];
                // Clamp at zero: for a near-collinear triangle this difference
                // of similar magnitudes can go slightly negative through
                // rounding alone, and a degenerate apex simply sits on the edge.
                let y_sq = ((w[0] * w[0] + w[1] * w[1] + w[2] * w[2]) - x * x).max(0.0);
                (x, y_sq.sqrt())
            };
            let (xa, ya) = planar(a);
            let (xd, yd) = planar(d);
            if ya + yd < 1e-12 {
                // Both apexes lie on the shared edge: the "triangles" are
                // degenerate slivers with no interior to cut across.
                continue;
            }

            // Where the unfolded straight line a—d crosses the shared edge.
            // Outside [0, edge_len] the true geodesic leaves through a
            // different edge, so this shortcut would be an over-shortcut.
            let s = xa + (xd - xa) * ya / (ya + yd);
            if s < -1e-6 || s > edge_len + 1e-6 {
                continue;
            }

            let dx = xd - xa;
            let dy = yd + ya;
            let shortcut = (dx * dx + dy * dy).sqrt();
            if !shortcut.is_finite() {
                continue;
            }

            // If `a` and `d` also happen to share a mesh edge (a tetrahedral
            // fan, say), this adds a parallel entry rather than replacing one.
            // That is harmless: `dijkstra_impl` relaxes every entry and keeps
            // whichever is shorter.
            adjacency[a].push(d);
            lengths[a].push(shortcut);
            adjacency[d].push(a);
            lengths[d].push(shortcut);
        }

        Self {
            adjacency,
            lengths,
            n_vertices,
        }
    }

    /// Run Dijkstra over this graph.
    fn run(&self, sources: &[usize], max_distance: f32) -> DijkstraResult {
        dijkstra_impl(
            self.n_vertices,
            &self.adjacency,
            &self.lengths,
            sources,
            max_distance,
        )
    }
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

/// Dijkstra's algorithm over the graph selected by
/// [`GeodesicConfig::use_face_graph`].
///
/// Computes shortest-path geodesic distances from one or more source vertices
/// to all reachable vertices in the mesh. With `use_face_graph = false` paths
/// follow mesh edges; with `use_face_graph = true` they may also cut across a
/// pair of triangles through their shared edge, which removes most of the edge
/// graph's over-estimation.
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
    validate_sources(mesh, sources)?;
    let n = mesh.n_vertices();

    let graph = SearchGraph::build(mesh, config);
    let result = graph.run(sources, config.max_distance);

    Ok(GeodesicField {
        distances: result.distances,
        predecessors: result.predecessors,
        n_vertices: n,
    })
}

/// Shared validation for the public entry points: non-empty mesh, non-empty
/// source list, and every source index in range.
fn validate_sources(mesh: &GeodesicMesh, sources: &[usize]) -> Result<(), GeodesicError> {
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
    Ok(())
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

/// Geodesic distances by the heat method of Crane, Weischedel & Wardetzky.
///
/// Unlike [`dijkstra`], which is restricted to paths along mesh edges and
/// therefore over-estimates true geodesic distance, the heat method solves for
/// distance over the smooth surface and returns values in mesh units that
/// depend on the actual vertex positions.
///
/// # Algorithm
///
/// 1. Diffuse heat from the source for a short time `t` by solving the
///    backward-Euler system `(M + t·Lc) u = δ_source`, with `M` the lumped mass
///    matrix and `Lc` the cotangent Laplacian.
/// 2. Form the unit vector field `X = −∇u/‖∇u‖`, which points along geodesics
///    back toward the source. Normalising discards the magnitude of `∇u`,
///    which is exactly where the diffusion approximation is unreliable.
/// 3. Recover the distance by solving the Poisson equation `Lc φ = ∇·X`, then
///    shift so the source sits at 0.
///
/// Both linear systems are solved with a Jacobi-preconditioned conjugate
/// gradient iteration.
///
/// # Arguments
///
/// - `source` — the vertex heat is released from.
/// - `n_iter` — iteration budget for each conjugate-gradient solve. A few
///   hundred is ample for FLAME-scale meshes; the solver stops early once the
///   residual is small.
/// - `dt` — the diffusion time `t`. The standard choice is the squared mean
///   edge length, available as [`heat_time_step`]; larger values smooth the
///   result, smaller values sharpen it at the cost of conditioning.
///
/// # Errors
/// - [`GeodesicError::EmptyMesh`] / [`GeodesicError::EmptyFaces`] for empty inputs.
/// - [`GeodesicError::VertexOutOfBounds`] if `source` is out of range.
/// - [`GeodesicError::InvalidConfig`] if `dt` ≤ 0 or `n_iter` is 0.
/// - [`GeodesicError::NumericalError`] if the mesh is fully degenerate (zero
///   total area), leaving no metric in which to measure distance.
pub fn heat_geodesic(
    mesh: &GeodesicMesh,
    source: usize,
    n_iter: usize,
    dt: f32,
) -> Result<GeodesicField, GeodesicError> {
    heat_geodesic_multi(mesh, &[source], n_iter, Some(dt))
}

/// The standard heat-method time step for `mesh`: the squared mean edge length.
///
/// Returns 0.0 for a mesh with no faces or fully degenerate edges.
#[must_use]
pub fn heat_time_step(mesh: &GeodesicMesh) -> f32 {
    let mut total = 0.0f32;
    let mut count = 0usize;
    for face in &mesh.faces {
        let [a, b, c] = *face;
        total += mesh.edge_length(a, b) + mesh.edge_length(b, c) + mesh.edge_length(c, a);
        count += 3;
    }
    if count == 0 {
        return 0.0;
    }
    let h = total / count as f32;
    h * h
}

/// Multi-source heat-method geodesic distances.
///
/// Every vertex in `sources` starts at distance 0 and the field measures the
/// distance to the nearest of them. Passing `time_step = None` selects the
/// standard [`heat_time_step`] heuristic.
///
/// See [`heat_geodesic`] for the algorithm.
///
/// # Errors
/// Same as [`heat_geodesic`], plus [`GeodesicError::InvalidConfig`] when
/// `sources` is empty.
pub fn heat_geodesic_multi(
    mesh: &GeodesicMesh,
    sources: &[usize],
    n_iter: usize,
    time_step: Option<f32>,
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
    if let Some(dt) = time_step {
        if dt <= 0.0 {
            return Err(GeodesicError::InvalidConfig(format!(
                "dt must be positive, got {dt}"
            )));
        }
    }
    if n_iter == 0 {
        return Err(GeodesicError::InvalidConfig("n_iter must be > 0".into()));
    }

    let distances = heat_method::heat_distances(mesh, sources, time_step, n_iter)?;

    // The heat method solves a global system rather than propagating along
    // edges, so no predecessor chain exists; reconstruct paths with `dijkstra`
    // when one is needed.
    Ok(GeodesicField {
        distances,
        predecessors: vec![None; n],
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
    // Built once and reused for every source — this is what makes k Dijkstra
    // runs cheaper than k independent `dijkstra` calls.
    let graph = SearchGraph::build(mesh, config);
    let mut matrix = vec![0.0_f32; k * k];

    for (i, &src) in landmarks.iter().enumerate() {
        let result = graph.run(&[src], config.max_distance);
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
    validate_sources(mesh, sources)?;

    let graph = SearchGraph::build(mesh, config);
    let result = graph.run(sources, config.max_distance);
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

/// Default number of farthest-point-sampled candidates [`geodesic_center`]
/// considers when the caller supplies no explicit sample set.
///
/// Chosen so the search stays a fixed multiple of a single Dijkstra run
/// regardless of mesh size; raise it via [`geodesic_center_sampled`] when a
/// more exhaustive search is wanted.
pub const DEFAULT_CENTER_SAMPLES: usize = 64;

/// Find the vertex that minimizes the total geodesic distance to all others.
///
/// # Cost
///
/// Each candidate costs one full single-source Dijkstra, i.e.
/// `O(E + V log V)`. Passing every vertex as a candidate is therefore
/// `O(V·(E + V log V))` — on a 5023-vertex FLAME head that is thousands of
/// Dijkstra runs and takes minutes.
///
/// To keep the default cheap, an **empty `sample_vertices` selects a
/// farthest-point-sampled subset of [`DEFAULT_CENTER_SAMPLES`] candidates**
/// rather than every vertex. Farthest-point sampling spreads the candidates
/// evenly over the surface, so the returned vertex is a good approximation of
/// the true centre at a fixed, small cost. Pass an explicit candidate list —
/// `(0..mesh.n_vertices()).collect()` for the exhaustive search — to control
/// this yourself, or use [`geodesic_center_sampled`] to choose the sample
/// count.
///
/// # Errors
/// - [`GeodesicError::EmptyMesh`] / [`GeodesicError::EmptyFaces`] for empty inputs.
/// - [`GeodesicError::VertexOutOfBounds`] if a candidate index is out of range.
pub fn geodesic_center(
    mesh: &GeodesicMesh,
    sample_vertices: &[usize],
    config: &GeodesicConfig,
) -> Result<usize, GeodesicError> {
    geodesic_center_sampled(mesh, sample_vertices, DEFAULT_CENTER_SAMPLES, config)
}

/// [`geodesic_center`] with an explicit candidate budget.
///
/// When `sample_vertices` is empty, `n_samples` candidates are chosen by
/// farthest-point sampling (clamped to the vertex count). When it is
/// non-empty, `n_samples` is ignored and exactly those candidates are searched.
///
/// # Errors
/// Same as [`geodesic_center`], plus [`GeodesicError::InvalidConfig`] when
/// `sample_vertices` is empty and `n_samples` is 0.
pub fn geodesic_center_sampled(
    mesh: &GeodesicMesh,
    sample_vertices: &[usize],
    n_samples: usize,
    config: &GeodesicConfig,
) -> Result<usize, GeodesicError> {
    if mesh.n_vertices() == 0 {
        return Err(GeodesicError::EmptyMesh);
    }
    if mesh.n_faces() == 0 {
        return Err(GeodesicError::EmptyFaces);
    }
    let n = mesh.n_vertices();
    let graph = SearchGraph::build(mesh, config);

    let candidates: Vec<usize> = if sample_vertices.is_empty() {
        if n_samples == 0 {
            return Err(GeodesicError::InvalidConfig(
                "n_samples must be > 0 when sample_vertices is empty".into(),
            ));
        }
        farthest_point_samples(&graph, n, n_samples.min(n), config.max_distance)
    } else {
        // Validate caller-supplied candidates.
        for &v in sample_vertices {
            if v >= n {
                return Err(GeodesicError::VertexOutOfBounds { idx: v, n });
            }
        }
        sample_vertices.to_vec()
    };

    // `farthest_point_samples` always returns at least vertex 0 for a non-empty
    // mesh, and the caller-supplied branch is non-empty by construction.
    let mut best_vertex = candidates.first().copied().unwrap_or(0);
    let mut best_sum = f32::INFINITY;

    for &v in &candidates {
        let result = graph.run(&[v], config.max_distance);
        let sum: f32 = result.distances.iter().filter(|d| d.is_finite()).sum();
        if sum < best_sum {
            best_sum = sum;
            best_vertex = v;
        }
    }

    Ok(best_vertex)
}

/// Pick `k` well-spread vertices by farthest-point sampling.
///
/// Starts at vertex 0 and repeatedly adds the vertex farthest (geodesically)
/// from everything chosen so far, maintaining the distance-to-set field
/// incrementally so the whole routine costs `k` Dijkstra runs rather than `V`.
fn farthest_point_samples(
    graph: &SearchGraph,
    n_vertices: usize,
    k: usize,
    max_distance: f32,
) -> Vec<usize> {
    if n_vertices == 0 || k == 0 {
        return Vec::new();
    }

    let mut chosen = Vec::with_capacity(k);
    let mut is_chosen = vec![false; n_vertices];
    chosen.push(0usize);
    is_chosen[0] = true;

    // Distance from each vertex to the nearest already-chosen sample.
    let mut nearest = graph.run(&[0], max_distance).distances;

    while chosen.len() < k {
        // Farthest reachable vertex from the current sample set.
        let next = nearest
            .iter()
            .enumerate()
            .filter(|&(v, d)| d.is_finite() && !is_chosen[v])
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(v, _)| v);

        let Some(next) = next else {
            // Every reachable vertex is already a sample (or the mesh is
            // disconnected and the rest is unreachable): stop early.
            break;
        };

        chosen.push(next);
        is_chosen[next] = true;
        let field = graph.run(&[next], max_distance);
        for (slot, &d) in nearest.iter_mut().zip(field.distances.iter()) {
            if d < *slot {
                *slot = d;
            }
        }
    }

    chosen
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
mod tests;
