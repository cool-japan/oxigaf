//! Bilateral symmetry analysis and enforcement for FLAME meshes.
//!
//! The FLAME mesh has 5023 vertices with approximate bilateral symmetry along
//! the YZ plane (X = 0). This module provides:
//!
//! - **Symmetry maps**: lookup tables pairing left/right vertices
//! - **Symmetry analysis**: quantify asymmetry in a reconstructed mesh
//! - **Symmetrization**: snap a mesh toward bilateral symmetry
//! - **Shape parameter symmetry**: enforce symmetric PCA components
//! - **Asymmetry detection**: find and rank the most asymmetric regions

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during symmetry operations.
#[derive(Debug, thiserror::Error)]
pub enum SymmetryError {
    /// Vertex count does not match the symmetry map length.
    #[error("Vertex count {got} does not match expected {expected}")]
    VertexCountMismatch { got: usize, expected: usize },

    /// A symmetry map entry references an out-of-bounds vertex index.
    #[error("Invalid symmetry map: vertex {vertex} mapped to {mapped} which is out of bounds")]
    InvalidSymmetryMap { vertex: usize, mapped: usize },

    /// Shape parameter vector is longer than the maximum supported length.
    #[error("Shape parameter length {0} exceeds max supported")]
    ShapeParamTooLong(usize),

    /// A vertex index is out of bounds.
    #[error("Vertex index {0} out of bounds")]
    VertexOutOfBounds(usize),
}

// ---------------------------------------------------------------------------
// Symmetry Map
// ---------------------------------------------------------------------------

/// Mapping from each vertex to its symmetric counterpart.
///
/// Midline vertices map to themselves.  Length equals the number of vertices.
pub type SymmetryMap = Vec<usize>;

/// Generate a synthetic FLAME symmetry map for `num_vertices` vertices.
///
/// The pairing strategy approximates the real FLAME bilateral symmetry:
///
/// - Vertices `0..num_vertices/4` are paired with `num_vertices-1-i` (left ↔ right).
/// - Vertices `num_vertices/4..num_vertices-num_vertices/4` map to themselves (midline).
/// - Vertices `num_vertices-num_vertices/4..num_vertices` are paired with `num_vertices-1-i`.
///
/// `q3` is derived as `num_vertices - q1` (rather than `3*num_vertices/4`
/// independently) so the two mirrored ranges `[0,q1)` and `[q3,num_vertices)`
/// always have equal length `q1`, making `i -> num_vertices-1-i` an exact
/// involution for every `num_vertices` -- not only multiples of 4. (With
/// `q3 = 3*num_vertices/4` computed independently, the two ranges can have
/// different lengths whenever `num_vertices % 4 != 0`, breaking the
/// mapping: e.g. for FLAME's real n=5023, `map[map[3767]] != 3767`.)
///
/// When `num_vertices < 4` every vertex maps to itself.
#[must_use]
pub fn generate_synthetic_symmetry_map(num_vertices: usize) -> SymmetryMap {
    if num_vertices < 4 {
        return (0..num_vertices).collect();
    }

    let q1 = num_vertices / 4;
    let q3 = num_vertices - q1;

    let map: Vec<usize> = (0..num_vertices)
        .map(|vertex_idx| {
            if vertex_idx < q1 || vertex_idx >= q3 {
                // Paired with mirror vertex
                num_vertices - 1 - vertex_idx
            } else {
                // Midline: maps to itself
                vertex_idx
            }
        })
        .collect();
    map
}

/// Validate a symmetry map.
///
/// Checks that:
/// 1. No entry is out of bounds.
/// 2. The mapping is an involution: `map[map[i]] == i` for all `i`.
///
/// # Errors
///
/// Returns an error if the operation fails.
pub fn validate_symmetry_map(map: &SymmetryMap) -> Result<(), SymmetryError> {
    let n = map.len();
    for (vertex, &mapped) in map.iter().enumerate() {
        if mapped >= n {
            return Err(SymmetryError::InvalidSymmetryMap { vertex, mapped });
        }
        // Involution check: map[map[vertex]] must equal vertex
        let back = map[mapped];
        if back != vertex {
            // map[mapped] might itself be out of bounds — bounds already checked above for
            // `mapped`, but `map[mapped]` is also within 0..n by the previous iteration or
            // will be caught when we reach that index.  Report as an invalid entry.
            return Err(SymmetryError::InvalidSymmetryMap {
                vertex,
                mapped: back,
            });
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Vertex reflection
// ---------------------------------------------------------------------------

/// Reflect a vertex position across the YZ plane by negating its X component.
#[must_use]
#[inline]
pub fn reflect_vertex(v: [f32; 3]) -> [f32; 3] {
    [-v[0], v[1], v[2]]
}

// ---------------------------------------------------------------------------
// Symmetry Analysis
// ---------------------------------------------------------------------------

/// Result of bilateral symmetry analysis on a mesh.
#[derive(Debug, Clone)]
pub struct SymmetryReport {
    /// Mean L2 distance between a vertex and its reflected symmetric counterpart.
    pub mean_asymmetry: f32,
    /// Maximum L2 distance (most asymmetric vertex pair).
    pub max_asymmetry: f32,
    /// Population standard deviation of per-pair asymmetry distances.
    pub std_asymmetry: f32,
    /// Bilateral symmetry score in \[0, 1\]: 1.0 = perfectly symmetric.
    ///
    /// `score = exp(-mean_asymmetry / 0.01)` where 0.01 world units ≈ 1 cm.
    pub symmetry_score: f32,
    /// Index of the most asymmetric vertex and its symmetric counterpart.
    pub most_asymmetric_pair: (usize, usize),
    /// Number of non-midline vertex pairs analysed.
    pub num_pairs: usize,
}

/// Analyze bilateral symmetry of a mesh.
///
/// For each pair `(i, j = map[i])` where `i < j` (to avoid double-counting),
/// the per-pair asymmetry distance is:
///
/// ```text
/// reflected_j = [-vertices[j][0], vertices[j][1], vertices[j][2]]
/// distance    = |vertices[i] - reflected_j|  (Euclidean)
/// ```
///
/// Statistics (mean, max, std, score) are computed over all such pairs.
///
/// # Errors
///
/// Returns an error if the operation fails.
pub fn analyze_symmetry(
    vertices: &[[f32; 3]],
    map: &SymmetryMap,
) -> Result<SymmetryReport, SymmetryError> {
    let n = vertices.len();
    if map.len() != n {
        return Err(SymmetryError::VertexCountMismatch {
            got: n,
            expected: map.len(),
        });
    }

    let mut distances: Vec<f32> = Vec::new();
    let mut max_dist = 0.0_f32;
    let mut max_pair = (0usize, 0usize);

    for i in 0..n {
        let j = map[i];
        if j <= i {
            // Skip midline (j==i) and already-processed mirror (j<i)
            continue;
        }
        let vi = vertices[i];
        let reflected_j = reflect_vertex(vertices[j]);
        let dx = vi[0] - reflected_j[0];
        let dy = vi[1] - reflected_j[1];
        let dz = vi[2] - reflected_j[2];
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();
        if dist > max_dist {
            max_dist = dist;
            max_pair = (i, j);
        }
        distances.push(dist);
    }

    let num_pairs = distances.len();

    // Handle the edge case of no pairs (all vertices are midline)
    if num_pairs == 0 {
        return Ok(SymmetryReport {
            mean_asymmetry: 0.0,
            max_asymmetry: 0.0,
            std_asymmetry: 0.0,
            symmetry_score: 1.0,
            most_asymmetric_pair: (0, 0),
            num_pairs: 0,
        });
    }

    let sum: f32 = distances.iter().sum();
    let mean = sum / num_pairs as f32;

    let variance: f32 = distances
        .iter()
        .map(|&d| (d - mean) * (d - mean))
        .sum::<f32>()
        / num_pairs as f32;
    let std = variance.sqrt();

    let symmetry_score = (-mean / 0.01_f32).exp();

    Ok(SymmetryReport {
        mean_asymmetry: mean,
        max_asymmetry: max_dist,
        std_asymmetry: std,
        symmetry_score,
        most_asymmetric_pair: max_pair,
        num_pairs,
    })
}

// ---------------------------------------------------------------------------
// Symmetrization
// ---------------------------------------------------------------------------

/// Create a symmetrized version of the mesh.
///
/// For each non-midline pair `(i, j = map[i])` where `i < j`:
/// ```text
/// new_i = average(vertices[i], reflect(vertices[j]))
/// new_j = reflect(new_i)
/// ```
///
/// Midline vertices (`map[i] == i`) have their X component zeroed (snapped to
/// the YZ plane).
///
/// # Errors
///
/// Returns an error if the operation fails.
pub fn symmetrize_mesh(
    vertices: &[[f32; 3]],
    map: &SymmetryMap,
) -> Result<Vec<[f32; 3]>, SymmetryError> {
    let n = vertices.len();
    if map.len() != n {
        return Err(SymmetryError::VertexCountMismatch {
            got: n,
            expected: map.len(),
        });
    }

    let mut out = vertices.to_vec();

    // Snap midline vertices to the YZ plane first
    for i in 0..n {
        if map[i] == i {
            out[i][0] = 0.0;
        }
    }

    // Symmetrize paired vertices (process each pair once, i < j)
    for i in 0..n {
        let j = map[i];
        if j <= i {
            continue;
        }
        let vi = vertices[i];
        let vj = vertices[j];
        let reflected_j = reflect_vertex(vj);

        // Average of vertex i and reflected j
        let new_i = [
            (vi[0] + reflected_j[0]) * 0.5,
            (vi[1] + reflected_j[1]) * 0.5,
            (vi[2] + reflected_j[2]) * 0.5,
        ];
        let new_j = reflect_vertex(new_i);

        out[i] = new_i;
        out[j] = new_j;
    }

    Ok(out)
}

/// Blend between the original and the symmetrized mesh.
///
/// - `alpha = 0.0` → identical to the original mesh
/// - `alpha = 1.0` → identical to the output of [`symmetrize_mesh`]
///
/// # Errors
///
/// Returns an error if the operation fails.
pub fn blend_with_symmetric(
    vertices: &[[f32; 3]],
    map: &SymmetryMap,
    alpha: f32,
) -> Result<Vec<[f32; 3]>, SymmetryError> {
    let symmetrized = symmetrize_mesh(vertices, map)?;
    let one_minus = 1.0 - alpha;
    let blended = vertices
        .iter()
        .zip(symmetrized.iter())
        .map(|(orig, sym)| {
            [
                one_minus * orig[0] + alpha * sym[0],
                one_minus * orig[1] + alpha * sym[1],
                one_minus * orig[2] + alpha * sym[2],
            ]
        })
        .collect();
    Ok(blended)
}

// ---------------------------------------------------------------------------
// Shape Parameter Symmetry
// ---------------------------------------------------------------------------

/// Return a parameter vector with the asymmetric components zeroed out.
///
/// The first `symmetric_dims` entries are considered to encode symmetric
/// features and are left unchanged.  The remaining entries (asymmetric
/// components) are set to zero.  `symmetric_dims` is clamped to `params.len()`.
#[must_use]
pub fn symmetrize_shape_params(params: &[f32], symmetric_dims: usize) -> Vec<f32> {
    let keep = symmetric_dims.min(params.len());
    let mut out = params.to_vec();
    for v in out.iter_mut().skip(keep) {
        *v = 0.0;
    }
    out
}

/// Compute the L2 norm of the asymmetric (zeroed-out) shape components.
///
/// This quantifies how much the parameter vector deviates from its symmetric
/// subspace.
#[must_use]
pub fn asymmetry_contribution(params: &[f32], symmetric_dims: usize) -> f32 {
    let keep = symmetric_dims.min(params.len());
    params[keep..].iter().map(|&v| v * v).sum::<f32>().sqrt()
}

/// Interpolate shape parameters toward their symmetric version.
///
/// - `alpha = 0.0` → original params unchanged
/// - `alpha = 1.0` → equivalent to [`symmetrize_shape_params`]
#[must_use]
pub fn blend_to_symmetric_params(params: &[f32], symmetric_dims: usize, alpha: f32) -> Vec<f32> {
    let symmetric = symmetrize_shape_params(params, symmetric_dims);
    let one_minus = 1.0 - alpha;
    params
        .iter()
        .zip(symmetric.iter())
        .map(|(&orig, &sym)| one_minus * orig + alpha * sym)
        .collect()
}

// ---------------------------------------------------------------------------
// Asymmetry Detection
// ---------------------------------------------------------------------------

/// Compute per-vertex asymmetry distances.
///
/// Each vertex `i` receives the L2 distance between itself and the reflection
/// of its symmetric counterpart.  Midline vertices (where `map[i] == i`)
/// receive distance `0.0` regardless of their actual X coordinate.
///
/// # Errors
///
/// Returns an error if the operation fails.
pub fn per_vertex_asymmetry(
    vertices: &[[f32; 3]],
    map: &SymmetryMap,
) -> Result<Vec<f32>, SymmetryError> {
    let n = vertices.len();
    if map.len() != n {
        return Err(SymmetryError::VertexCountMismatch {
            got: n,
            expected: map.len(),
        });
    }

    let mut result = vec![0.0_f32; n];

    for i in 0..n {
        let j = map[i];
        if j == i {
            // Midline: always 0
            continue;
        }
        // Compute once per pair, then assign to both ends
        if j > i {
            let vi = vertices[i];
            let reflected_j = reflect_vertex(vertices[j]);
            let dx = vi[0] - reflected_j[0];
            let dy = vi[1] - reflected_j[1];
            let dz = vi[2] - reflected_j[2];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            result[i] = dist;
            result[j] = dist;
        }
    }

    Ok(result)
}

/// Find the `n` most asymmetric vertex pairs, sorted in descending order.
///
/// Returns a `Vec` of `(vertex_i, vertex_j, asymmetry_distance)` tuples.
/// If `n` exceeds the number of non-midline pairs, all pairs are returned.
///
/// # Errors
///
/// Returns an error if the operation fails.
pub fn top_asymmetric_pairs(
    vertices: &[[f32; 3]],
    map: &SymmetryMap,
    n: usize,
) -> Result<Vec<(usize, usize, f32)>, SymmetryError> {
    let num_verts = vertices.len();
    if map.len() != num_verts {
        return Err(SymmetryError::VertexCountMismatch {
            got: num_verts,
            expected: map.len(),
        });
    }

    let mut pairs: Vec<(usize, usize, f32)> = Vec::new();

    for i in 0..num_verts {
        let j = map[i];
        if j <= i {
            continue;
        }
        let vi = vertices[i];
        let reflected_j = reflect_vertex(vertices[j]);
        let dx = vi[0] - reflected_j[0];
        let dy = vi[1] - reflected_j[1];
        let dz = vi[2] - reflected_j[2];
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();
        pairs.push((i, j, dist));
    }

    // Sort descending by asymmetry distance
    pairs.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    pairs.truncate(n);
    Ok(pairs)
}

/// Build a per-vertex asymmetry heatmap with values normalised to \[0, 1\].
///
/// Values are the per-vertex asymmetry distances divided by the maximum
/// asymmetry distance.  Midline vertices always receive `0.0`.  When the
/// maximum asymmetry is zero (perfectly symmetric mesh), every vertex gets
/// `0.0`.
///
/// # Errors
///
/// Returns an error if the operation fails.
pub fn asymmetry_heatmap(
    vertices: &[[f32; 3]],
    map: &SymmetryMap,
) -> Result<Vec<f32>, SymmetryError> {
    let raw = per_vertex_asymmetry(vertices, map)?;
    let max = raw.iter().copied().fold(0.0_f32, f32::max);
    if max == 0.0 {
        return Ok(raw); // all zeros already
    }
    Ok(raw.iter().map(|&v| v / max).collect())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- helpers -----------------------------------------------------------

    fn make_vertices(n: usize) -> Vec<[f32; 3]> {
        (0..n).map(|i| [i as f32 * 0.01, 0.0, 0.0]).collect()
    }

    // ---- generate_synthetic_symmetry_map -----------------------------------

    #[test]
    fn test_map_length() {
        let n = 5023;
        let map = generate_synthetic_symmetry_map(n);
        assert_eq!(map.len(), n);
    }

    #[test]
    fn test_map_midline_self() {
        let n = 100;
        let map = generate_synthetic_symmetry_map(n);
        let q1 = n / 4; // 25
        let q3 = 3 * n / 4; // 75
        for (i, &mapped) in map[q1..q3].iter().enumerate() {
            let global_i = q1 + i;
            assert_eq!(mapped, global_i, "vertex {global_i} should be midline");
        }
    }

    #[test]
    fn test_map_small_n() {
        // n < 4: all vertices map to themselves
        for n in 0..4 {
            let map = generate_synthetic_symmetry_map(n);
            for (i, &mapped) in map.iter().enumerate().take(n) {
                assert_eq!(mapped, i);
            }
        }
    }

    // Regression test for the non-involution bug: FLAME's real mesh has
    // 5023 vertices, and 5023 % 4 == 3 -- exactly the non-multiple-of-4
    // case that broke the old `q1 = n/4, q3 = 3*n/4` independent
    // computation (map[map[3767]] != 3767 under the old formula).
    #[test]
    fn test_map_flame_vertex_count_is_involution() {
        let map = generate_synthetic_symmetry_map(5023);
        assert!(validate_symmetry_map(&map).is_ok());
    }

    proptest::proptest! {
        #[test]
        fn prop_synthetic_symmetry_map_is_involution(num_vertices in 0usize..2000) {
            let map = generate_synthetic_symmetry_map(num_vertices);
            proptest::prop_assert!(validate_symmetry_map(&map).is_ok());
        }
    }

    // ---- validate_symmetry_map ---------------------------------------------

    #[test]
    fn test_validate_valid_map() {
        let map = generate_synthetic_symmetry_map(100);
        assert!(validate_symmetry_map(&map).is_ok());
    }

    #[test]
    fn test_validate_out_of_bounds() {
        let mut map = generate_synthetic_symmetry_map(10);
        map[2] = 999; // out of bounds
        assert!(matches!(
            validate_symmetry_map(&map),
            Err(SymmetryError::InvalidSymmetryMap { .. })
        ));
    }

    // ---- reflect_vertex ----------------------------------------------------

    #[test]
    fn test_reflect_negates_x() {
        let v = [1.0, 2.0, 3.0];
        let r = reflect_vertex(v);
        assert!((r[0] - (-1.0)).abs() < 1e-7);
        assert!((r[1] - 2.0).abs() < 1e-7);
        assert!((r[2] - 3.0).abs() < 1e-7);
    }

    #[test]
    fn test_reflect_double_is_identity() {
        let v = [1.5, -2.3, 0.7];
        let r = reflect_vertex(reflect_vertex(v));
        assert!((r[0] - v[0]).abs() < 1e-7);
        assert!((r[1] - v[1]).abs() < 1e-7);
        assert!((r[2] - v[2]).abs() < 1e-7);
    }

    // ---- analyze_symmetry --------------------------------------------------

    /// Build a mesh that is exactly bilateral-symmetric and verify score ≈ 1.
    #[test]
    fn test_analyze_perfectly_symmetric() {
        // Use a mesh where every pair is already symmetric (distance ≈ 0).
        let n = 100;
        let map = generate_synthetic_symmetry_map(n);
        // Construct vertices so that for every paired (i,j):
        // vertices[i] = [x, y, z]  and  vertices[j] = [-x, y, z]
        let mut verts = vec![[0.0f32; 3]; n];
        let q1 = n / 4;
        let q3 = 3 * n / 4;
        for i in 0..n {
            if i < q1 || i >= q3 {
                let j = n - 1 - i;
                let x = (i as f32 + 1.0) * 0.005;
                verts[i] = [x, 0.1, 0.2];
                verts[j] = [-x, 0.1, 0.2];
            } else {
                verts[i] = [0.0, i as f32 * 0.001, 0.0];
            }
        }
        let report = analyze_symmetry(&verts, &map).expect("analyze_symmetry failed");
        assert!(
            report.symmetry_score > 0.99,
            "expected score ≈ 1, got {}",
            report.symmetry_score
        );
    }

    #[test]
    fn test_analyze_asymmetric_mesh() {
        let n = 100;
        let map = generate_synthetic_symmetry_map(n);
        let verts = make_vertices(n); // asymmetric
        let report = analyze_symmetry(&verts, &map).expect("analyze_symmetry failed");
        assert!(
            report.symmetry_score < 1.0,
            "asymmetric mesh should score < 1, got {}",
            report.symmetry_score
        );
    }

    #[test]
    fn test_analyze_vertex_count_mismatch() {
        let map = generate_synthetic_symmetry_map(10);
        let verts = make_vertices(5); // wrong length
        assert!(matches!(
            analyze_symmetry(&verts, &map),
            Err(SymmetryError::VertexCountMismatch { .. })
        ));
    }

    #[test]
    fn test_analyze_mean_nonnegative() {
        let n = 100;
        let map = generate_synthetic_symmetry_map(n);
        let verts = make_vertices(n);
        let report = analyze_symmetry(&verts, &map).expect("analyze_symmetry failed");
        assert!(report.mean_asymmetry >= 0.0);
    }

    // ---- symmetrize_mesh ---------------------------------------------------

    #[test]
    fn test_symmetrize_length() {
        let n = 100;
        let map = generate_synthetic_symmetry_map(n);
        let verts = make_vertices(n);
        let sym = symmetrize_mesh(&verts, &map).expect("symmetrize failed");
        assert_eq!(sym.len(), n);
    }

    #[test]
    fn test_symmetrize_more_symmetric() {
        let n = 100;
        let map = generate_synthetic_symmetry_map(n);
        let verts = make_vertices(n);

        let before = analyze_symmetry(&verts, &map)
            .expect("analyze before failed")
            .mean_asymmetry;
        let sym = symmetrize_mesh(&verts, &map).expect("symmetrize failed");
        let after = analyze_symmetry(&sym, &map)
            .expect("analyze after failed")
            .mean_asymmetry;

        assert!(
            after <= before + 1e-5,
            "symmetrized mesh should be no more asymmetric than original: before={before}, after={after}"
        );
    }

    #[test]
    fn test_symmetrize_idempotent() {
        let n = 100;
        let map = generate_synthetic_symmetry_map(n);
        let verts = make_vertices(n);
        let sym1 = symmetrize_mesh(&verts, &map).expect("symmetrize first failed");
        let sym2 = symmetrize_mesh(&sym1, &map).expect("symmetrize second failed");

        for (a, b) in sym1.iter().zip(sym2.iter()) {
            for k in 0..3 {
                assert!(
                    (a[k] - b[k]).abs() < 1e-5,
                    "idempotency failed at component {k}: {a:?} vs {b:?}"
                );
            }
        }
    }

    // ---- blend_with_symmetric ----------------------------------------------

    #[test]
    fn test_blend_alpha0_is_original() {
        let n = 100;
        let map = generate_synthetic_symmetry_map(n);
        let verts = make_vertices(n);
        let blended = blend_with_symmetric(&verts, &map, 0.0).expect("blend failed");
        for (orig, bl) in verts.iter().zip(blended.iter()) {
            for k in 0..3 {
                assert!(
                    (orig[k] - bl[k]).abs() < 1e-6,
                    "alpha=0 should match original"
                );
            }
        }
    }

    #[test]
    fn test_blend_alpha1_is_symmetric() {
        let n = 100;
        let map = generate_synthetic_symmetry_map(n);
        let verts = make_vertices(n);
        let blended = blend_with_symmetric(&verts, &map, 1.0).expect("blend alpha=1 failed");
        let sym = symmetrize_mesh(&verts, &map).expect("symmetrize failed");
        for (bl, s) in blended.iter().zip(sym.iter()) {
            for k in 0..3 {
                assert!(
                    (bl[k] - s[k]).abs() < 1e-5,
                    "alpha=1 should match symmetrize_mesh"
                );
            }
        }
    }

    // ---- symmetrize_shape_params -------------------------------------------

    #[test]
    fn test_symmetrize_shape_params_zeros_asym() {
        let params = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let out = symmetrize_shape_params(&params, 3);
        assert_eq!(out.len(), 5);
        assert!((out[3] - 0.0).abs() < 1e-7);
        assert!((out[4] - 0.0).abs() < 1e-7);
    }

    #[test]
    fn test_symmetrize_shape_params_keeps_symmetric() {
        let params = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let out = symmetrize_shape_params(&params, 3);
        assert!((out[0] - 1.0).abs() < 1e-7);
        assert!((out[1] - 2.0).abs() < 1e-7);
        assert!((out[2] - 3.0).abs() < 1e-7);
    }

    // ---- asymmetry_contribution --------------------------------------------

    #[test]
    fn test_asymmetry_contribution_all_zero() {
        let params = vec![0.0f32; 10];
        assert!((asymmetry_contribution(&params, 5) - 0.0).abs() < 1e-7);
    }

    #[test]
    fn test_asymmetry_contribution_nonzero() {
        let params = vec![1.0, 0.0, 0.0, 3.0, 4.0]; // symmetric_dims=3 → keep [1,0,0], asym=[3,4]
        let contrib = asymmetry_contribution(&params, 3);
        // sqrt(9+16) = 5
        assert!((contrib - 5.0).abs() < 1e-5, "expected 5, got {contrib}");
    }

    // ---- blend_to_symmetric_params -----------------------------------------

    #[test]
    fn test_blend_params_alpha0() {
        let params = vec![1.0, 2.0, 3.0, 4.0];
        let out = blend_to_symmetric_params(&params, 2, 0.0);
        for (o, &p) in out.iter().zip(params.iter()) {
            assert!((o - p).abs() < 1e-7, "alpha=0 should preserve all params");
        }
    }

    // ---- per_vertex_asymmetry ----------------------------------------------

    #[test]
    fn test_per_vertex_asymmetry_length() {
        let n = 100;
        let map = generate_synthetic_symmetry_map(n);
        let verts = make_vertices(n);
        let asym = per_vertex_asymmetry(&verts, &map).expect("per_vertex_asymmetry failed");
        assert_eq!(asym.len(), n);
    }

    #[test]
    fn test_per_vertex_asymmetry_symmetric_mesh() {
        let n = 100;
        let map = generate_synthetic_symmetry_map(n);
        // Build a mesh symmetric under `map`
        let mut verts = vec![[0.0f32; 3]; n];
        let q1 = n / 4;
        let q3 = 3 * n / 4;
        for i in 0..n {
            if i < q1 || i >= q3 {
                let j = n - 1 - i;
                let x = (i as f32 + 1.0) * 0.003;
                verts[i] = [x, 0.5, 0.1];
                verts[j] = [-x, 0.5, 0.1];
            } else {
                verts[i] = [0.0, 0.5, 0.1];
            }
        }
        let asym = per_vertex_asymmetry(&verts, &map).expect("per_vertex_asymmetry failed");
        for (idx, &v) in asym.iter().enumerate() {
            assert!(
                v.abs() < 1e-5,
                "symmetric mesh: expected 0 at vertex {idx}, got {v}"
            );
        }
    }

    // ---- top_asymmetric_pairs ----------------------------------------------

    #[test]
    fn test_top_asymmetric_pairs_sorted() {
        let n = 100;
        let map = generate_synthetic_symmetry_map(n);
        let verts = make_vertices(n);
        let pairs = top_asymmetric_pairs(&verts, &map, 5).expect("top_asymmetric_pairs failed");
        assert!(pairs.len() <= 5);
        // Verify descending order
        for w in pairs.windows(2) {
            assert!(
                w[0].2 >= w[1].2,
                "pairs not sorted descending: {} < {}",
                w[0].2,
                w[1].2
            );
        }
    }

    // ---- asymmetry_heatmap -------------------------------------------------

    #[test]
    fn test_heatmap_range() {
        let n = 100;
        let map = generate_synthetic_symmetry_map(n);
        let verts = make_vertices(n);
        let heat = asymmetry_heatmap(&verts, &map).expect("heatmap failed");
        for &v in &heat {
            assert!((0.0..=1.0).contains(&v), "heatmap out of [0,1]: {v}");
        }
    }

    #[test]
    fn test_heatmap_symmetric_mesh_all_zeros() {
        let n = 100;
        let map = generate_synthetic_symmetry_map(n);
        let mut verts = vec![[0.0f32; 3]; n];
        let q1 = n / 4;
        let q3 = 3 * n / 4;
        for i in 0..n {
            if i < q1 || i >= q3 {
                let j = n - 1 - i;
                let x = (i as f32 + 1.0) * 0.003;
                verts[i] = [x, 0.2, 0.3];
                verts[j] = [-x, 0.2, 0.3];
            } else {
                verts[i] = [0.0, 0.2, 0.3];
            }
        }
        let heat = asymmetry_heatmap(&verts, &map).expect("heatmap failed");
        for &v in &heat {
            assert!(
                v.abs() < 1e-5,
                "symmetric mesh heatmap should be 0, got {v}"
            );
        }
    }
}
