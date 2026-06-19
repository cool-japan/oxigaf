//! Mesh repair and cleanup utilities.
//!
//! Provides algorithms to fix common mesh defects: duplicate vertices, inconsistent
//! winding order, and topological holes. Operations can be applied individually or
//! through the unified [`repair_mesh`] pipeline.
//!
//! # Example
//!
//! ```rust,no_run
//! use oxigaf_flame::mesh_repair::{repair_mesh, MeshRepairConfig};
//!
//! let vertices: Vec<[f32; 3]> = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
//! let faces: Vec<[u32; 3]> = vec![[0, 1, 2]];
//!
//! let config = MeshRepairConfig::default();
//! let result = repair_mesh(&vertices, &faces, &config).expect("repair failed");
//! println!("{}", result.stats.format_summary());
//! ```

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during mesh repair operations.
#[derive(Debug)]
pub enum MeshRepairError {
    /// Input mesh has no vertices or no faces when they are required.
    EmptyMesh,
    /// A face references a vertex index that does not exist in the vertex array.
    InvalidFaceIndex {
        /// Face index (0-based) that contains the invalid reference.
        face: usize,
        /// The out-of-range vertex index found in the face.
        index: u32,
        /// The total number of vertices (valid indices are `0..vertex_count`).
        vertex_count: usize,
    },
    /// General repair failure with a descriptive message.
    RepairFailed(String),
}

impl fmt::Display for MeshRepairError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyMesh => write!(f, "mesh repair error: input mesh is empty"),
            Self::InvalidFaceIndex {
                face,
                index,
                vertex_count,
            } => write!(
                f,
                "mesh repair error: face {face} references vertex index {index}, \
                 but mesh only has {vertex_count} vertices"
            ),
            Self::RepairFailed(msg) => write!(f, "mesh repair error: {msg}"),
        }
    }
}

impl std::error::Error for MeshRepairError {}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration controlling which repair operations are applied.
#[derive(Debug, Clone)]
pub struct MeshRepairConfig {
    /// Unify vertices whose positions are within `merge_threshold` of each other.
    pub remove_duplicate_vertices: bool,
    /// Spatial distance threshold for duplicate vertex merging.
    ///
    /// Two vertices are considered duplicates if each coordinate rounds to the
    /// same grid cell of size `merge_threshold`.
    pub merge_threshold: f32,
    /// Flood-fill consistent winding order across all faces.
    pub fix_winding_order: bool,
    /// Detect boundary loops and triangulate holes that are small enough.
    pub fill_holes: bool,
    /// Maximum number of boundary edges allowed in a hole for auto-fill.
    ///
    /// Holes with more edges than this limit are left unfilled.
    pub max_hole_size: usize,
}

impl Default for MeshRepairConfig {
    fn default() -> Self {
        Self {
            remove_duplicate_vertices: true,
            merge_threshold: 1e-5,
            fix_winding_order: true,
            fill_holes: true,
            max_hole_size: 10,
        }
    }
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Summary of changes made during a mesh repair operation.
#[derive(Debug, Clone, Default)]
pub struct MeshRepairStats {
    /// Number of vertices removed by the deduplication pass.
    pub duplicates_removed: usize,
    /// Number of faces whose winding was flipped for consistency.
    pub faces_flipped: usize,
    /// Number of holes that were triangulated and filled.
    pub holes_filled: usize,
    /// Vertex count before any repair operations.
    pub original_vertex_count: usize,
    /// Face count before any repair operations.
    pub original_face_count: usize,
    /// Vertex count after all repair operations.
    pub final_vertex_count: usize,
    /// Face count after all repair operations (includes added hole-fill faces).
    pub final_face_count: usize,
}

impl MeshRepairStats {
    /// Format statistics as a human-readable multi-line summary.
    #[must_use]
    pub fn format_summary(&self) -> String {
        format!(
            "Mesh Repair Summary\n\
             --------------------------------------------------\n\
             Vertices: {} → {} (removed {})\n\
             Faces:    {} → {} (flipped {}, holes filled {})\n\
             Holes filled: {}\n",
            self.original_vertex_count,
            self.final_vertex_count,
            self.duplicates_removed,
            self.original_face_count,
            self.final_face_count,
            self.faces_flipped,
            self.holes_filled,
            self.holes_filled,
        )
    }
}

// ---------------------------------------------------------------------------
// Result
// ---------------------------------------------------------------------------

/// Output of a mesh repair pipeline run.
#[derive(Debug, Clone)]
pub struct MeshRepairResult {
    /// Repaired vertex positions.
    pub vertices: Vec<[f32; 3]>,
    /// Repaired face index triples.
    pub faces: Vec<[u32; 3]>,
    /// Statistics about the changes made.
    pub stats: MeshRepairStats,
}

// ---------------------------------------------------------------------------
// Extension trait
// ---------------------------------------------------------------------------

/// Adds a [`repair`](MeshRepairExt::repair) convenience method to mesh types.
pub trait MeshRepairExt {
    /// Apply mesh repair operations according to `config`.
    ///
    /// # Errors
    ///
    /// Returns [`MeshRepairError`] if the mesh is invalid or a repair step fails.
    fn repair(&self, config: &MeshRepairConfig) -> Result<MeshRepairResult, MeshRepairError>;
}

impl MeshRepairExt for crate::mesh::Mesh {
    fn repair(&self, config: &MeshRepairConfig) -> Result<MeshRepairResult, MeshRepairError> {
        // Convert nalgebra vertices to plain arrays
        let raw_vertices: Vec<[f32; 3]> = self.vertices.iter().map(|v| [v.x, v.y, v.z]).collect();
        repair_mesh(&raw_vertices, &self.faces, config)
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Quantize a float coordinate to a grid of size `threshold`.
///
/// Returns the grid cell index as an `i64` so that floats on opposite sides of
/// zero map to different cells.
#[inline]
fn quantize(val: f32, threshold: f32) -> i64 {
    (val / threshold).floor() as i64
}

/// Directed-edge key: `(from_vertex, to_vertex)`.
type DirectedEdge = (u32, u32);

// ---------------------------------------------------------------------------
// Duplicate vertex removal
// ---------------------------------------------------------------------------

/// Remove duplicate vertices by snapping positions to a uniform grid.
///
/// # Returns
///
/// `(new_vertices, new_faces, duplicates_removed)`.
fn remove_duplicates(
    vertices: &[[f32; 3]],
    faces: &[[u32; 3]],
    threshold: f32,
) -> (Vec<[f32; 3]>, Vec<[u32; 3]>, usize) {
    // Map quantized position → canonical vertex index in the new list.
    let mut grid: HashMap<(i64, i64, i64), u32> = HashMap::new();
    // old index → new (canonical) index
    let mut remap: Vec<u32> = Vec::with_capacity(vertices.len());
    let mut new_verts: Vec<[f32; 3]> = Vec::with_capacity(vertices.len());

    for v in vertices {
        let key = (
            quantize(v[0], threshold),
            quantize(v[1], threshold),
            quantize(v[2], threshold),
        );
        let new_idx = if let Some(&existing) = grid.get(&key) {
            existing
        } else {
            let idx = new_verts.len() as u32;
            grid.insert(key, idx);
            new_verts.push(*v);
            idx
        };
        remap.push(new_idx);
    }

    let duplicates_removed = vertices.len().saturating_sub(new_verts.len());

    // Rebuild faces, dropping degenerate ones (two or more equal indices after remap).
    let mut new_faces: Vec<[u32; 3]> = Vec::with_capacity(faces.len());
    for face in faces {
        let i0 = remap[face[0] as usize];
        let i1 = remap[face[1] as usize];
        let i2 = remap[face[2] as usize];
        // Skip degenerate face
        if i0 == i1 || i1 == i2 || i0 == i2 {
            continue;
        }
        new_faces.push([i0, i1, i2]);
    }

    (new_verts, new_faces, duplicates_removed)
}

// ---------------------------------------------------------------------------
// Winding order fix
// ---------------------------------------------------------------------------

/// Make the winding order consistent across all faces using BFS flood-fill.
///
/// Starting from face 0, propagates a consistent orientation to all reachable
/// faces through shared edges. Faces in separate connected components may end
/// up with either orientation (whichever matches the seed for that component).
///
/// # Returns
///
/// `(new_faces, faces_flipped)`.
fn fix_winding_order(vertices_count: usize, faces: &[[u32; 3]]) -> (Vec<[u32; 3]>, usize) {
    let _ = vertices_count; // not needed, but kept for symmetry with callers
    if faces.is_empty() {
        return (Vec::new(), 0);
    }

    let mut out_faces: Vec<[u32; 3]> = faces.to_vec();
    let face_count = out_faces.len();

    // Build directed-edge → face-indices map from current face list.
    // We rebuild this lazily per BFS iteration after potential flips.
    // For efficiency, build once and update on flip.
    let build_edge_map = |fcs: &[[u32; 3]]| -> HashMap<DirectedEdge, Vec<usize>> {
        let mut map: HashMap<DirectedEdge, Vec<usize>> = HashMap::new();
        for (fi, face) in fcs.iter().enumerate() {
            let edges = [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])];
            for e in edges {
                map.entry(e).or_default().push(fi);
            }
        }
        map
    };

    let mut edge_map = build_edge_map(&out_faces);
    let mut visited: Vec<bool> = vec![false; face_count];
    let mut faces_flipped = 0usize;

    for start in 0..face_count {
        if visited[start] {
            continue;
        }
        // BFS queue contains face indices.
        let mut queue: std::collections::VecDeque<usize> = std::collections::VecDeque::new();
        queue.push_back(start);
        visited[start] = true;

        while let Some(fi) = queue.pop_front() {
            let face = out_faces[fi];
            let directed_edges = [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])];

            for (a, b) in directed_edges {
                // A consistently wound neighbor would have edge (b, a) — reversed.
                let reverse_edge: DirectedEdge = (b, a);
                // An inconsistently wound neighbor would have edge (a, b) — same direction.
                let same_edge: DirectedEdge = (a, b);

                // Check reverse-edge neighbors (already consistent — mark visited).
                if let Some(neighbors) = edge_map.get(&reverse_edge) {
                    for &nfi in neighbors {
                        if nfi != fi && !visited[nfi] {
                            visited[nfi] = true;
                            queue.push_back(nfi);
                        }
                    }
                }

                // Check same-direction neighbors (inconsistent — need flip).
                let same_neighbors: Vec<usize> =
                    edge_map.get(&same_edge).cloned().unwrap_or_default();
                for nfi in same_neighbors {
                    if nfi != fi && !visited[nfi] {
                        // Flip this face by swapping indices 1 and 2.
                        let f = &mut out_faces[nfi];
                        f.swap(1, 2);
                        faces_flipped += 1;
                        visited[nfi] = true;
                        // Rebuild edge map to reflect the flip.
                        edge_map = build_edge_map(&out_faces);
                        queue.push_back(nfi);
                    }
                }
            }
        }
    }

    (out_faces, faces_flipped)
}

// ---------------------------------------------------------------------------
// Hole filling
// ---------------------------------------------------------------------------

/// Find boundary directed edges: edges that appear only once (no reverse edge).
fn find_boundary_directed_edges(faces: &[[u32; 3]]) -> HashMap<u32, u32> {
    // Count how many times each directed edge appears.
    let mut directed: HashMap<DirectedEdge, u32> = HashMap::new();
    for face in faces {
        for (a, b) in [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])] {
            *directed.entry((a, b)).or_insert(0) += 1;
        }
    }

    // A boundary directed edge (a→b) exists when (a,b) is present but (b,a) is not.
    let mut boundary_next: HashMap<u32, u32> = HashMap::new();
    for &(a, b) in directed.keys() {
        if !directed.contains_key(&(b, a)) {
            boundary_next.insert(a, b);
        }
    }
    boundary_next
}

/// Trace closed boundary loops from the adjacency map `start → next`.
///
/// Returns a list of loops, each expressed as an ordered list of vertex indices.
fn trace_boundary_loops(boundary_next: &HashMap<u32, u32>) -> Vec<Vec<u32>> {
    let mut visited: HashMap<u32, bool> = HashMap::new();
    let mut loops: Vec<Vec<u32>> = Vec::new();

    for &start in boundary_next.keys() {
        if visited.get(&start).copied().unwrap_or(false) {
            continue;
        }

        let mut loop_verts: Vec<u32> = Vec::new();
        let mut current = start;
        let mut ok = true;

        loop {
            if visited.get(&current).copied().unwrap_or(false) {
                // If we've come back to the start, the loop is complete.
                if current == start {
                    break;
                }
                // Otherwise this boundary is non-simple; skip it.
                ok = false;
                break;
            }
            visited.insert(current, true);
            loop_verts.push(current);

            if let Some(&next) = boundary_next.get(&current) {
                current = next;
            } else {
                ok = false;
                break;
            }
        }

        if ok && loop_verts.len() >= 3 {
            loops.push(loop_verts);
        }
    }

    loops
}

/// Fill holes by fan-triangulating boundary loops that are ≤ `max_hole_size` edges.
///
/// # Returns
///
/// `(new_faces_to_add, holes_filled)`.
fn fill_holes(faces: &[[u32; 3]], max_hole_size: usize) -> (Vec<[u32; 3]>, usize) {
    let boundary_next = find_boundary_directed_edges(faces);
    if boundary_next.is_empty() {
        return (Vec::new(), 0);
    }

    let loops = trace_boundary_loops(&boundary_next);
    let mut new_faces: Vec<[u32; 3]> = Vec::new();
    let mut holes_filled = 0usize;

    for loop_verts in &loops {
        let n = loop_verts.len();
        // Only fill holes up to max_hole_size boundary edges.
        if n > max_hole_size {
            continue;
        }
        // Fan-triangulate from loop_verts[0].
        let pivot = loop_verts[0];
        for i in 1..(n - 1) {
            new_faces.push([pivot, loop_verts[i], loop_verts[i + 1]]);
        }
        holes_filled += 1;
    }

    (new_faces, holes_filled)
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

/// Validate that all face indices are within bounds.
fn validate_faces(vertices: &[[f32; 3]], faces: &[[u32; 3]]) -> Result<(), MeshRepairError> {
    let vertex_count = vertices.len();
    for (fi, face) in faces.iter().enumerate() {
        for &idx in face {
            if idx as usize >= vertex_count {
                return Err(MeshRepairError::InvalidFaceIndex {
                    face: fi,
                    index: idx,
                    vertex_count,
                });
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Apply mesh repair operations in order: dedup → winding → hole-fill.
///
/// # Errors
///
/// - [`MeshRepairError::EmptyMesh`] if the vertex array is empty.
/// - [`MeshRepairError::InvalidFaceIndex`] if a face references a nonexistent vertex.
/// - [`MeshRepairError::RepairFailed`] if an internal step fails unexpectedly.
pub fn repair_mesh(
    vertices: &[[f32; 3]],
    faces: &[[u32; 3]],
    config: &MeshRepairConfig,
) -> Result<MeshRepairResult, MeshRepairError> {
    if vertices.is_empty() {
        return Err(MeshRepairError::EmptyMesh);
    }

    // Validate all face indices before any operation.
    validate_faces(vertices, faces)?;

    let original_vertex_count = vertices.len();
    let original_face_count = faces.len();

    let mut stats = MeshRepairStats {
        original_vertex_count,
        original_face_count,
        ..MeshRepairStats::default()
    };

    // Step 1: Remove duplicate vertices.
    let (work_verts, work_faces, dups_removed) = if config.remove_duplicate_vertices {
        let (v, f, d) = remove_duplicates(vertices, faces, config.merge_threshold);
        (v, f, d)
    } else {
        (vertices.to_vec(), faces.to_vec(), 0)
    };
    stats.duplicates_removed = dups_removed;

    // Step 2: Fix winding order.
    let (work_faces, flipped) = if config.fix_winding_order {
        fix_winding_order(work_verts.len(), &work_faces)
    } else {
        (work_faces, 0)
    };
    stats.faces_flipped = flipped;

    // Step 3: Fill holes.
    let (extra_faces, holes) = if config.fill_holes {
        fill_holes(&work_faces, config.max_hole_size)
    } else {
        (Vec::new(), 0)
    };
    stats.holes_filled = holes;

    // Merge extra faces from hole filling.
    let final_faces: Vec<[u32; 3]> = work_faces.into_iter().chain(extra_faces).collect();

    stats.final_vertex_count = work_verts.len();
    stats.final_face_count = final_faces.len();

    Ok(MeshRepairResult {
        vertices: work_verts,
        faces: final_faces,
        stats,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Helper: create a simple tetrahedron-like open mesh with a triangle hole
    // -----------------------------------------------------------------------

    /// Three vertices of a single triangle (boundary edges on all three sides).
    fn single_triangle() -> (Vec<[f32; 3]>, Vec<[u32; 3]>) {
        let verts = vec![[0.0f32, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let faces = vec![[0u32, 1, 2]];
        (verts, faces)
    }

    /// A closed strip of two triangles sharing an edge (a quad made of two tris).
    /// Faces: [0,1,2] and [1,3,2] — consistently wound (CCW from above).
    fn two_tri_strip() -> (Vec<[f32; 3]>, Vec<[u32; 3]>) {
        let verts = vec![
            [0.0f32, 0.0, 0.0], // 0
            [1.0, 0.0, 0.0],    // 1
            [0.0, 1.0, 0.0],    // 2
            [1.0, 1.0, 0.0],    // 3
        ];
        let faces = vec![[0u32, 1, 2], [1, 3, 2]];
        (verts, faces)
    }

    // -----------------------------------------------------------------------
    // test_default_config
    // -----------------------------------------------------------------------

    #[test]
    fn test_default_config() {
        let cfg = MeshRepairConfig::default();
        assert!(cfg.remove_duplicate_vertices);
        assert!(cfg.fix_winding_order);
        assert!(cfg.fill_holes);
        assert!(cfg.merge_threshold > 0.0);
        assert!(cfg.max_hole_size > 0);
    }

    // -----------------------------------------------------------------------
    // test_repair_empty_mesh_error
    // -----------------------------------------------------------------------

    #[test]
    fn test_repair_empty_mesh_error() {
        let cfg = MeshRepairConfig::default();
        let result = repair_mesh(&[], &[], &cfg);
        assert!(matches!(result, Err(MeshRepairError::EmptyMesh)));
    }

    // -----------------------------------------------------------------------
    // test_remove_duplicates_simple
    // -----------------------------------------------------------------------

    #[test]
    fn test_remove_duplicates_simple() {
        // Two identical vertices → should collapse to one.
        let verts: Vec<[f32; 3]> = vec![
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0], // exact duplicate of [0]
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        // Valid face using all four original indices.
        let faces: Vec<[u32; 3]> = vec![[0, 2, 3], [1, 3, 2]];

        let (new_verts, _new_faces, removed) = remove_duplicates(&verts, &faces, 1e-5);
        assert_eq!(removed, 1, "one duplicate should have been removed");
        assert_eq!(new_verts.len(), 3);
    }

    // -----------------------------------------------------------------------
    // test_remove_duplicates_threshold
    // -----------------------------------------------------------------------

    #[test]
    fn test_remove_duplicates_threshold() {
        // Two vertices very close together — within threshold → merged.
        let threshold = 0.01_f32;
        let verts: Vec<[f32; 3]> = vec![
            [0.0, 0.0, 0.0],
            [0.0, 0.005, 0.0], // 0.005 < threshold → same grid cell
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let faces: Vec<[u32; 3]> = vec![[0, 2, 3]];
        let (new_verts, _, removed) = remove_duplicates(&verts, &faces, threshold);
        assert_eq!(removed, 1);
        assert_eq!(new_verts.len(), 3);
    }

    // -----------------------------------------------------------------------
    // test_remove_duplicates_degenerate_faces_removed
    // -----------------------------------------------------------------------

    #[test]
    fn test_remove_duplicates_degenerate_faces_removed() {
        // A face whose vertices all collapse to the same canonical vertex.
        let verts: Vec<[f32; 3]> = vec![
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0], // duplicate of 0
            [0.0, 0.0, 0.0], // duplicate of 0
        ];
        let faces: Vec<[u32; 3]> = vec![[0, 1, 2]];
        let (_new_verts, new_faces, _) = remove_duplicates(&verts, &faces, 1e-5);
        assert!(
            new_faces.is_empty(),
            "degenerate face should have been dropped"
        );
    }

    // -----------------------------------------------------------------------
    // test_winding_order_simple_two_faces
    // -----------------------------------------------------------------------

    #[test]
    fn test_winding_order_simple_two_faces() {
        // Two faces sharing edge (1,2). Face 0 has winding [0,1,2] (directed edge 1→2).
        // Face 1 is [0,2,1] which also has edge 2→1 from vertex 0 side and
        // same-direction edge relative to face 0. It should be flipped.
        let verts = [
            [0.0f32, 0.0, 0.0], // 0
            [1.0, 0.0, 0.0],    // 1
            [0.0, 1.0, 0.0],    // 2
            [1.0, 1.0, 0.0],    // 3
        ];
        let faces: Vec<[u32; 3]> = vec![
            [0, 1, 2], // CCW face
            [3, 1, 2], // same edge direction (1→2) as face 0 → inconsistent → should flip
        ];
        let (fixed_faces, flipped) = fix_winding_order(verts.len(), &faces);
        assert_eq!(flipped, 1, "one face should have been flipped");
        // After flip, the second face should no longer share edge (1,2) in same direction.
        let f1 = fixed_faces[1];
        let has_same_edge =
            (f1[0], f1[1]) == (1, 2) || (f1[1], f1[2]) == (1, 2) || (f1[2], f1[0]) == (1, 2);
        assert!(
            !has_same_edge,
            "after fix, face 1 should not have directed edge (1,2)"
        );
    }

    // -----------------------------------------------------------------------
    // test_winding_order_already_consistent
    // -----------------------------------------------------------------------

    #[test]
    fn test_winding_order_already_consistent() {
        // The two-tri strip is already consistently wound.
        let (verts, faces) = two_tri_strip();
        let (fixed_faces, flipped) = fix_winding_order(verts.len(), &faces);
        assert_eq!(flipped, 0, "no flips should be needed");
        assert_eq!(fixed_faces, faces);
    }

    // -----------------------------------------------------------------------
    // test_fill_holes_triangle_hole
    // -----------------------------------------------------------------------

    #[test]
    fn test_fill_holes_triangle_hole() {
        // A single triangle has 3 boundary edges forming one triangular hole.
        let (_, faces) = single_triangle();
        let (new_faces, filled) = fill_holes(&faces, 10);
        assert_eq!(filled, 1, "should fill the one triangular hole");
        // The hole-filling fan for a 3-vertex loop adds exactly 1 face.
        assert_eq!(new_faces.len(), 1);
    }

    // -----------------------------------------------------------------------
    // test_fill_holes_quad_hole
    // -----------------------------------------------------------------------

    #[test]
    fn test_fill_holes_quad_hole() {
        // Four vertices arranged in a square. Two triangles cover the interior,
        // but we only add one triangle so three boundary edges remain forming a tri-hole.
        // Actually: build a 4-vert quad with only one triangle → 3-edge hole.
        // For a genuine quad hole we need 4 boundary edges.
        // Build: vertices 0,1,2,3 in a cycle with no faces (all boundary).
        // Add a face [0,1,2] → boundary edges: 2→0, 1→3 is separate.
        // Instead: a cycle of 4 edges as boundary → 2 fill triangles.
        //
        // Create a mesh that has a 4-edge boundary loop by taking two triangles
        // that share one edge and leave a quad boundary.
        // Arrangement:
        //   0---1
        //   |\ |
        //   | \|
        //   3---2
        // Add faces [0,1,3] and [1,2,3] — that closes the quad.
        // Remove face [1,2,3] → boundary edges 1→2, 2→3, 3→1 (triangle hole)
        // That's only 3 edges. For a 4-edge hole, start fresh:
        //   Vertices 0,1,2,3,4 arranged so there is a quad hole 1-2-3-4.
        // Faces: only [0,1,4] — leaves edges 1→2, 2→3, 3→4, 4→1 as boundary? No.
        //
        // Simplest: just build faces that leave a known 4-edge boundary loop.
        // Use vertices 0..4, add face [0,1,2] and [0,2,3]. Boundary is 3→0 and 0→1→2→3 cycle.
        // Actually boundary: 0→1, 1→2, 2→3, 3→0 → 4-edge loop.
        let verts: Vec<[f32; 3]> = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        // No faces at all → all edges are missing, so no boundary edges
        // to detect (boundary detection needs at least some faces).
        // Instead use a half-filled quad: add diagonals [0,1,2] only.
        // Boundary: 2→0, 0→... wait that's only 3 edges.
        // Correct approach: faces [0,1,2] has boundary 2→0, 1→2 wait...
        // Let me reason carefully.
        // face [0,1,2] has directed edges: 0→1, 1→2, 2→0.
        // boundary edges: those without reverse in any face.
        // With only face [0,1,2] all three are boundary (no other faces).
        // Add face [2,0,3] which has: 2→0, 0→3, 3→2.
        //   But (2→0) is already in face0 and face1 with (0→2)... wait.
        //   face0: 0→1, 1→2, 2→0
        //   face1 [2,0,3]: 2→0, 0→3, 3→2
        //   Edge 0→2 does not exist anywhere, but 2→0 appears twice (face0 and face1)!
        //   That makes 2→0 non-boundary but still has no (0→2) reverse in any face.
        //   So 2→0 appears twice: that's non-manifold. Let's avoid that.
        //
        // Correct for a 4-vert cycle with one face missing:
        // Add face [0,1,3] → directed: 0→1, 1→3, 3→0.
        // Boundary: 0→1 has reverse 1→0? No → boundary.
        //           1→3 has reverse 3→1? No → boundary.
        //           3→0 has reverse 0→3? No → boundary.
        // So still only a 3-edge hole. Need 2 faces sharing an interior edge.
        // Add [0,1,2] and [0,2,3]:
        //   face0: 0→1, 1→2, 2→0
        //   face1: 0→2, 2→3, 3→0
        //   2→0 from face0 and 0→2 from face1 cancel each other (manifold).
        //   Boundary: 0→1 (no 1→0), 1→2 (no 2→1), 2→3 (no 3→2), 3→0 (no 0→3).
        //   That's a 4-edge boundary loop! ✓
        let faces: Vec<[u32; 3]> = vec![[0, 1, 2], [0, 2, 3]];
        let _ = verts; // not needed for hole fill; faces only uses indices
        let (new_faces, filled) = fill_holes(&faces, 10);
        assert_eq!(filled, 1, "should fill the one quad hole");
        // Fan triangulation of a 4-vertex loop: 2 triangles.
        assert_eq!(new_faces.len(), 2);
    }

    // -----------------------------------------------------------------------
    // test_fill_holes_max_size_limit
    // -----------------------------------------------------------------------

    #[test]
    fn test_fill_holes_max_size_limit() {
        // A triangle hole (3 edges) with max_hole_size = 2 → should NOT be filled.
        let (_, faces) = single_triangle();
        let (new_faces, filled) = fill_holes(&faces, 2);
        assert_eq!(filled, 0);
        assert!(new_faces.is_empty());
    }

    // -----------------------------------------------------------------------
    // test_repair_stats_format_summary
    // -----------------------------------------------------------------------

    #[test]
    fn test_repair_stats_format_summary() {
        let stats = MeshRepairStats {
            duplicates_removed: 5,
            faces_flipped: 3,
            holes_filled: 2,
            original_vertex_count: 100,
            original_face_count: 200,
            final_vertex_count: 95,
            final_face_count: 204,
        };
        let summary = stats.format_summary();
        assert!(summary.contains("95"));
        assert!(summary.contains("100"));
        assert!(summary.contains('5'));
        assert!(summary.contains('3'));
        assert!(summary.contains('2'));
        assert!(summary.contains("204"));
    }

    // -----------------------------------------------------------------------
    // test_repair_all_disabled
    // -----------------------------------------------------------------------

    #[test]
    fn test_repair_all_disabled() {
        let (verts, faces) = two_tri_strip();
        let cfg = MeshRepairConfig {
            remove_duplicate_vertices: false,
            fix_winding_order: false,
            fill_holes: false,
            merge_threshold: 1e-5,
            max_hole_size: 10,
        };
        let result = repair_mesh(&verts, &faces, &cfg).expect("repair should succeed");
        // Nothing changed.
        assert_eq!(result.vertices.len(), verts.len());
        assert_eq!(result.faces.len(), faces.len());
        assert_eq!(result.stats.duplicates_removed, 0);
        assert_eq!(result.stats.faces_flipped, 0);
        assert_eq!(result.stats.holes_filled, 0);
    }

    // -----------------------------------------------------------------------
    // test_repair_pipeline_order
    // -----------------------------------------------------------------------

    #[test]
    fn test_repair_pipeline_order() {
        // Build a mesh with duplicates AND a hole, verify both are handled.
        let verts: Vec<[f32; 3]> = vec![
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0], // duplicate of 0
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        // Only one valid face after dedup (no hole in this simple case).
        let faces: Vec<[u32; 3]> = vec![[0, 2, 3]];
        let cfg = MeshRepairConfig {
            remove_duplicate_vertices: true,
            fix_winding_order: true,
            fill_holes: false, // disable hole fill to test dedup + winding
            merge_threshold: 1e-5,
            max_hole_size: 10,
        };
        let result = repair_mesh(&verts, &faces, &cfg).expect("repair should succeed");
        // Dedup should have removed vertex index 1.
        assert!(result.stats.duplicates_removed >= 1);
        assert_eq!(result.stats.original_vertex_count, 4);
    }

    // -----------------------------------------------------------------------
    // test_mesh_repair_ext_trait
    // -----------------------------------------------------------------------

    #[test]
    fn test_mesh_repair_ext_trait() {
        use crate::mesh::Mesh;
        use crate::mesh_repair::MeshRepairExt;

        let vertices = vec![
            nalgebra::Point3::new(0.0f32, 0.0, 0.0),
            nalgebra::Point3::new(1.0, 0.0, 0.0),
            nalgebra::Point3::new(0.0, 1.0, 0.0),
        ];
        let faces = vec![[0u32, 1, 2]];
        let mesh = Mesh::new(vertices, faces);

        let cfg = MeshRepairConfig::default();
        let result = mesh.repair(&cfg).expect("repair via trait should succeed");
        // At least one vertex remains.
        assert!(!result.vertices.is_empty());
    }

    // -----------------------------------------------------------------------
    // test_invalid_face_index_error
    // -----------------------------------------------------------------------

    #[test]
    fn test_invalid_face_index_error() {
        let verts: Vec<[f32; 3]> = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let faces: Vec<[u32; 3]> = vec![[0, 1, 99]]; // index 99 is out of range
        let cfg = MeshRepairConfig::default();
        let result = repair_mesh(&verts, &faces, &cfg);
        assert!(
            matches!(
                result,
                Err(MeshRepairError::InvalidFaceIndex { index: 99, .. })
            ),
            "expected InvalidFaceIndex error"
        );
    }

    // -----------------------------------------------------------------------
    // test_repair_result_fields
    // -----------------------------------------------------------------------

    #[test]
    fn test_repair_result_fields() {
        let (verts, faces) = two_tri_strip();
        let cfg = MeshRepairConfig::default();
        let result = repair_mesh(&verts, &faces, &cfg).expect("repair should succeed");
        assert_eq!(
            result.vertices.len(),
            result.stats.final_vertex_count,
            "final_vertex_count must match actual vertex count"
        );
        assert_eq!(
            result.faces.len(),
            result.stats.final_face_count,
            "final_face_count must match actual face count"
        );
    }

    // -----------------------------------------------------------------------
    // Additional edge case tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_single_face_repair() {
        let (verts, faces) = single_triangle();
        let cfg = MeshRepairConfig::default();
        let result = repair_mesh(&verts, &faces, &cfg).expect("repair should succeed");
        // Hole fill on a single triangle should add one fill face.
        assert!(result.stats.holes_filled >= 1);
    }

    #[test]
    fn test_repair_preserves_valid_mesh() {
        // A fully closed mesh (tetrahedron) has no holes, no duplicates needed.
        // Tetrahedron: 4 vertices, 4 triangles, all consistently wound outward.
        let verts: Vec<[f32; 3]> = vec![
            [1.0, 1.0, 1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [1.0, -1.0, -1.0],
        ];
        let faces: Vec<[u32; 3]> = vec![[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]];
        let cfg = MeshRepairConfig {
            remove_duplicate_vertices: true,
            fix_winding_order: true,
            fill_holes: true,
            merge_threshold: 1e-5,
            max_hole_size: 10,
        };
        let result = repair_mesh(&verts, &faces, &cfg).expect("repair should succeed");
        assert_eq!(result.stats.duplicates_removed, 0);
        assert_eq!(result.stats.holes_filled, 0);
        // Vertex count should be unchanged.
        assert_eq!(result.vertices.len(), 4);
    }

    #[test]
    fn test_error_display_empty_mesh() {
        let err = MeshRepairError::EmptyMesh;
        let msg = err.to_string();
        assert!(msg.contains("empty"));
    }

    #[test]
    fn test_error_display_invalid_face_index() {
        let err = MeshRepairError::InvalidFaceIndex {
            face: 5,
            index: 99,
            vertex_count: 10,
        };
        let msg = err.to_string();
        assert!(msg.contains("99"));
        assert!(msg.contains("10"));
    }

    #[test]
    fn test_error_display_repair_failed() {
        let err = MeshRepairError::RepairFailed("test failure".to_string());
        let msg = err.to_string();
        assert!(msg.contains("test failure"));
    }
}
