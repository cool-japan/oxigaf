//! Correspondence search: centroids, nearest-neighbour matching, outlier
//! rejection, and the pre-registration RMSE that reports on the raw pair.

use rayon::prelude::*;

use super::kdtree::{KdTree, PAR_QUERY_THRESHOLD};
use super::math::{vec3_dot, vec3_sub};
use super::types::{Correspondence, RegistrationError};

/// Compute the mean position (centroid) of a flat position array.
///
/// Returns `[mean_x, mean_y, mean_z]`.
pub fn compute_centroid_3d(positions: &[f32]) -> Result<[f32; 3], RegistrationError> {
    if positions.is_empty() {
        return Err(RegistrationError::EmptyCloud);
    }
    if !positions.len().is_multiple_of(3) {
        return Err(RegistrationError::InvalidPositionLength {
            len: positions.len(),
        });
    }
    let n = (positions.len() / 3) as f32;
    let mut sum = [0.0f32; 3];
    for chunk in positions.chunks_exact(3) {
        sum[0] += chunk[0];
        sum[1] += chunk[1];
        sum[2] += chunk[2];
    }
    Ok([sum[0] / n, sum[1] / n, sum[2] / n])
}

/// Reject a source/target pair that is not two well-formed flat xyz arrays.
///
/// The source is checked before the target so that every entry point reports
/// the same error for the same malformed input.
pub(super) fn validate_cloud_pair(source: &[f32], target: &[f32]) -> Result<(), RegistrationError> {
    if source.is_empty() {
        return Err(RegistrationError::EmptyCloud);
    }
    if !source.len().is_multiple_of(3) {
        return Err(RegistrationError::InvalidPositionLength { len: source.len() });
    }
    if target.is_empty() {
        return Err(RegistrationError::EmptyCloud);
    }
    if !target.len().is_multiple_of(3) {
        return Err(RegistrationError::InvalidPositionLength { len: target.len() });
    }
    Ok(())
}

/// For each source point, find the nearest target point.
///
/// A balanced k-d tree is built over the target cloud once per call, so the
/// search costs `O(n_tgt log n_tgt + n_src log n_tgt)` instead of the
/// `O(n_src n_tgt)` of a brute-force scan, and returns the identical matches.
/// Only correspondences with distance < `max_dist` are included.
///
/// Callers that query the *same* target repeatedly hoist the tree out of their
/// loop instead — that is what [`register_point_clouds`] does, which is why it
/// should be preferred over hand-rolling an ICP loop around
/// [`icp_step`].
///
/// [`register_point_clouds`]: super::register_point_clouds
/// [`icp_step`]: super::icp_step
pub fn find_correspondences(
    source: &[f32],
    target: &[f32],
    max_dist: f32,
) -> Result<Vec<Correspondence>, RegistrationError> {
    validate_cloud_pair(source, target)?;
    let tree = KdTree::build_all(target);
    Ok(query_nearest(source, target, &tree, max_dist))
}

/// [`find_correspondences`] against a k-d tree that already exists.
///
/// # Invariant
///
/// `tree` must have been built over exactly the `target` slice passed here
/// (`KdTree::build_all(target)`); the tree stores indices into that slice and
/// nothing in the type system ties the two together. Pairing a tree with a
/// different — or mutated — target yields matches for the cloud the tree was
/// built from, not for `target`.
pub(super) fn find_correspondences_with_tree(
    source: &[f32],
    target: &[f32],
    tree: &KdTree,
    max_dist: f32,
) -> Result<Vec<Correspondence>, RegistrationError> {
    validate_cloud_pair(source, target)?;
    Ok(query_nearest(source, target, tree, max_dist))
}

/// Shared query loop behind both correspondence entry points.
///
/// Queries are independent, so they parallelise once the cloud is large enough
/// to outweigh the thread-pool hand-off. `collect` keeps source order on both
/// paths, so the two produce byte-identical output.
fn query_nearest(
    source: &[f32],
    target: &[f32],
    tree: &KdTree,
    max_dist: f32,
) -> Vec<Correspondence> {
    let n_src = source.len() / 3;

    let match_one = |source_idx: usize| -> Option<Correspondence> {
        let sp = [
            source[source_idx * 3],
            source[source_idx * 3 + 1],
            source[source_idx * 3 + 2],
        ];
        let (target_idx, sq) = tree.nearest(target, sp)?;
        let distance = sq.sqrt();
        (distance < max_dist).then_some(Correspondence {
            source_idx,
            target_idx,
            distance,
        })
    };

    if n_src >= PAR_QUERY_THRESHOLD {
        (0..n_src).into_par_iter().filter_map(match_one).collect()
    } else {
        (0..n_src).filter_map(match_one).collect()
    }
}

/// Remove the worst `outlier_fraction` of correspondences (by distance).
///
/// If `outlier_fraction` is 0.0, returns all correspondences unchanged.
pub fn filter_correspondences(
    correspondences: Vec<Correspondence>,
    outlier_fraction: f32,
) -> Vec<Correspondence> {
    if outlier_fraction <= 0.0 || correspondences.is_empty() {
        return correspondences;
    }
    let mut sorted = correspondences;
    sorted.sort_by(|a, b| {
        a.distance
            .partial_cmp(&b.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let keep = ((sorted.len() as f32) * (1.0 - outlier_fraction.min(1.0))).ceil() as usize;
    let keep = keep.max(1).min(sorted.len());
    sorted.truncate(keep);
    sorted
}

/// Compute the initial RMSE between source and target using the identity transform.
///
/// Finds correspondences without applying any transform, then computes RMSE.
///
/// # Errors
///
/// Fails on empty or mis-sized clouds, and with
/// [`RegistrationError::NoCorrespondences`] when not a single pair could be
/// matched — reporting `0.0` there would read as a perfect alignment.
pub fn compute_initial_rmse(source: &[f32], target: &[f32]) -> Result<f32, RegistrationError> {
    validate_cloud_pair(source, target)?;

    let corr = find_correspondences(source, target, f32::MAX)?;
    if corr.is_empty() {
        return Err(RegistrationError::NoCorrespondences);
    }

    let mut sq_sum = 0.0f32;
    for c in &corr {
        let sp = [
            source[c.source_idx * 3],
            source[c.source_idx * 3 + 1],
            source[c.source_idx * 3 + 2],
        ];
        let tp = [
            target[c.target_idx * 3],
            target[c.target_idx * 3 + 1],
            target[c.target_idx * 3 + 2],
        ];
        let diff = vec3_sub(sp, tp);
        sq_sum += vec3_dot(diff, diff);
    }
    Ok((sq_sum / corr.len() as f32).sqrt())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cloud_registration::test_support::{approx_eq, pseudo_cloud};

    // -----------------------------------------------------------------------
    // compute_centroid_3d
    // -----------------------------------------------------------------------

    #[test]
    fn test_centroid_single_point() {
        let pts = vec![3.0f32, -1.0, 2.0];
        let c = compute_centroid_3d(&pts).unwrap();
        assert!(approx_eq(c[0], 3.0, 1e-6));
        assert!(approx_eq(c[1], -1.0, 1e-6));
        assert!(approx_eq(c[2], 2.0, 1e-6));
    }

    #[test]
    fn test_centroid_two_points() {
        let pts = vec![0.0f32, 0.0, 0.0, 2.0, 2.0, 2.0];
        let c = compute_centroid_3d(&pts).unwrap();
        assert!(approx_eq(c[0], 1.0, 1e-6));
        assert!(approx_eq(c[1], 1.0, 1e-6));
        assert!(approx_eq(c[2], 1.0, 1e-6));
    }

    #[test]
    fn test_centroid_empty_error() {
        let result = compute_centroid_3d(&[]);
        assert!(matches!(result, Err(RegistrationError::EmptyCloud)));
    }

    #[test]
    fn test_centroid_invalid_length_error() {
        let result = compute_centroid_3d(&[1.0, 2.0]);
        assert!(matches!(
            result,
            Err(RegistrationError::InvalidPositionLength { len: 2 })
        ));
    }

    // -----------------------------------------------------------------------
    // find_correspondences
    // -----------------------------------------------------------------------

    #[test]
    fn test_find_correspondences_basic() {
        let source = vec![0.0f32, 0.0, 0.0, 10.0, 0.0, 0.0];
        let target = vec![0.1f32, 0.0, 0.0, 10.1, 0.0, 0.0, 100.0, 0.0, 0.0];
        let corr = find_correspondences(&source, &target, f32::MAX).unwrap();
        assert_eq!(corr.len(), 2);
        assert_eq!(corr[0].source_idx, 0);
        assert_eq!(corr[0].target_idx, 0);
        assert_eq!(corr[1].source_idx, 1);
        assert_eq!(corr[1].target_idx, 1);
    }

    #[test]
    fn test_find_correspondences_max_dist_filter() {
        let source = vec![0.0f32, 0.0, 0.0, 10.0, 0.0, 0.0];
        let target = vec![0.1f32, 0.0, 0.0, 100.0, 0.0, 0.0];
        // Second source point is 90 units from its nearest target
        let corr = find_correspondences(&source, &target, 1.0).unwrap();
        assert_eq!(corr.len(), 1);
        assert_eq!(corr[0].source_idx, 0);
    }

    #[test]
    fn test_find_correspondences_empty_errors() {
        let no_source = find_correspondences(&[], &[0.0, 0.0, 0.0], f32::MAX);
        assert!(matches!(no_source, Err(RegistrationError::EmptyCloud)));
        let no_target = find_correspondences(&[0.0, 0.0, 0.0], &[], f32::MAX);
        assert!(matches!(no_target, Err(RegistrationError::EmptyCloud)));
    }

    #[test]
    fn test_find_correspondences_invalid_length_errors() {
        // The source is reported before the target, whichever is malformed.
        assert!(matches!(
            find_correspondences(&[1.0, 2.0], &[0.0, 0.0, 0.0], f32::MAX),
            Err(RegistrationError::InvalidPositionLength { len: 2 })
        ));
        assert!(matches!(
            find_correspondences(&[0.0, 0.0, 0.0], &[1.0, 2.0, 3.0, 4.0], f32::MAX),
            Err(RegistrationError::InvalidPositionLength { len: 4 })
        ));
    }

    #[test]
    fn test_find_correspondences_matches_brute_force() {
        // The k-d tree must return exactly what an exhaustive scan would.
        let src = pseudo_cloud(60, 12_345, 10.0);
        let tgt = pseudo_cloud(80, 987, 10.0);
        let corr = find_correspondences(&src, &tgt, f32::MAX).unwrap();
        assert_eq!(corr.len(), 60);
        for c in &corr {
            let sp = [
                src[c.source_idx * 3],
                src[c.source_idx * 3 + 1],
                src[c.source_idx * 3 + 2],
            ];
            // Same rule as the tree: smallest squared distance, lowest index.
            let mut best = (usize::MAX, f32::MAX);
            for ti in 0..80 {
                let tp = [tgt[ti * 3], tgt[ti * 3 + 1], tgt[ti * 3 + 2]];
                let diff = vec3_sub(sp, tp);
                let sq = vec3_dot(diff, diff);
                if sq < best.1 {
                    best = (ti, sq);
                }
            }
            assert_eq!(c.target_idx, best.0, "source {}", c.source_idx);
            assert!(approx_eq(c.distance, best.1.sqrt(), 1e-5));
        }
    }

    #[test]
    fn test_find_correspondences_parallel_path_keeps_order() {
        // Above the threshold the queries run on the rayon pool; the output
        // must still be one correspondence per source point, in source order.
        let n = PAR_QUERY_THRESHOLD + 16;
        let src = pseudo_cloud(n, 7, 25.0);
        let tgt = pseudo_cloud(n, 99, 25.0);
        let corr = find_correspondences(&src, &tgt, f32::MAX).unwrap();
        assert_eq!(corr.len(), n);
        for (i, c) in corr.iter().enumerate() {
            assert_eq!(c.source_idx, i);
        }
    }

    #[test]
    fn test_find_correspondences_self_nearest() {
        // Each point is its own nearest neighbour
        let pts = vec![0.0f32, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 100.0, 0.0];
        let corr = find_correspondences(&pts, &pts, f32::MAX).unwrap();
        for c in &corr {
            assert_eq!(c.source_idx, c.target_idx);
            assert!(approx_eq(c.distance, 0.0, 1e-6));
        }
    }

    // -----------------------------------------------------------------------
    // find_correspondences_with_tree — must equal the tree-building path
    // -----------------------------------------------------------------------

    #[test]
    fn test_with_tree_matches_fresh_tree() {
        // Reusing a hoisted tree may not change a single match, on either the
        // serial or the rayon query path.
        for n in [64usize, PAR_QUERY_THRESHOLD + 8] {
            let src = pseudo_cloud(n, 31, 9.0);
            let tgt = pseudo_cloud(n + 5, 1_009, 9.0);
            let tree = KdTree::build_all(&tgt);
            let fresh = find_correspondences(&src, &tgt, f32::MAX).unwrap();
            let reused = find_correspondences_with_tree(&src, &tgt, &tree, f32::MAX).unwrap();
            assert_eq!(fresh.len(), reused.len(), "n = {}", n);
            for (a, b) in fresh.iter().zip(reused.iter()) {
                assert_eq!(a.source_idx, b.source_idx);
                assert_eq!(a.target_idx, b.target_idx);
                assert_eq!(a.distance.to_bits(), b.distance.to_bits());
            }
        }
    }

    #[test]
    fn test_with_tree_applies_max_dist_and_validation() {
        let src = vec![0.0f32, 0.0, 0.0, 10.0, 0.0, 0.0];
        let tgt = vec![0.1f32, 0.0, 0.0, 100.0, 0.0, 0.0];
        let tree = KdTree::build_all(&tgt);
        let corr = find_correspondences_with_tree(&src, &tgt, &tree, 1.0).unwrap();
        assert_eq!(corr.len(), 1);
        assert_eq!(corr[0].source_idx, 0);
        // Malformed input is rejected here exactly as in the public entry point.
        assert!(matches!(
            find_correspondences_with_tree(&[], &tgt, &tree, 1.0),
            Err(RegistrationError::EmptyCloud)
        ));
    }

    // -----------------------------------------------------------------------
    // filter_correspondences
    // -----------------------------------------------------------------------

    #[test]
    fn test_filter_correspondences_edge_cases() {
        let corr: Vec<Correspondence> = [5.0f32, 1.0, 9.0]
            .iter()
            .enumerate()
            .map(|(i, &distance)| Correspondence {
                source_idx: i,
                target_idx: i,
                distance,
            })
            .collect();
        // A zero fraction keeps everything.
        assert_eq!(filter_correspondences(corr.clone(), 0.0).len(), 3);
        // Discarding everything still keeps one correspondence.
        assert!(!filter_correspondences(corr, 1.0).is_empty());
        assert!(filter_correspondences(vec![], 0.5).is_empty());
    }

    #[test]
    fn test_filter_correspondences_half() {
        let corr: Vec<Correspondence> = (0..10)
            .map(|i| Correspondence {
                source_idx: i,
                target_idx: i,
                distance: i as f32,
            })
            .collect();
        let filtered = filter_correspondences(corr, 0.5);
        // Keep the 5 closest (distance 0..4) – ceil(10 * 0.5) = 5
        assert!(filtered.len() <= 6);
        // All remaining should have distance <= 5.0
        for c in &filtered {
            assert!(
                c.distance <= 5.0,
                "expected distance <= 5.0, got {}",
                c.distance
            );
        }
    }

    // -----------------------------------------------------------------------
    // compute_initial_rmse
    // -----------------------------------------------------------------------

    #[test]
    fn test_initial_rmse_same_cloud() {
        let pts = vec![0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0];
        let rmse = compute_initial_rmse(&pts, &pts).unwrap();
        assert!(approx_eq(rmse, 0.0, 1e-6));
    }

    #[test]
    fn test_initial_rmse_offset() {
        let src = vec![0.0f32, 0.0, 0.0];
        let tgt = vec![1.0f32, 0.0, 0.0];
        let rmse = compute_initial_rmse(&src, &tgt).unwrap();
        assert!(approx_eq(rmse, 1.0, 1e-5));
    }

    #[test]
    fn test_initial_rmse_empty_error() {
        let result = compute_initial_rmse(&[], &[1.0, 2.0, 3.0]);
        assert!(matches!(result, Err(RegistrationError::EmptyCloud)));
    }

    #[test]
    fn test_initial_rmse_without_correspondences_errors() {
        // A non-finite target yields no valid pair; 0.0 would read as a
        // perfect alignment, so an error must come back instead.
        let src = vec![0.0f32, 0.0, 0.0];
        let tgt = vec![f32::NAN, 0.0, 0.0];
        let result = compute_initial_rmse(&src, &tgt);
        assert!(matches!(result, Err(RegistrationError::NoCorrespondences)));
    }
}
