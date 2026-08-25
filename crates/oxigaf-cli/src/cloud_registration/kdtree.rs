//! Nearest-neighbour acceleration structure over a target cloud.
//!
//! A [`KdTree`] is a pure function of the `(target, n)` pair it was built from
//! and is immutable afterwards, which is what lets the ICP loop build one tree
//! and reuse it across every iteration.

use super::math::{vec3_dot, vec3_sub};

/// Source-cloud size from which nearest-neighbour queries are run on the rayon
/// pool. Below it the pool hand-off costs more than the queries themselves.
pub(super) const PAR_QUERY_THRESHOLD: usize = 1024;

/// Balanced k-d tree over a target cloud, stored as a permuted index array.
///
/// Every subtree occupies one contiguous slice laid out as
/// `[left | median | right]` and is split on axis `depth % 3`, so the whole tree
/// is a single `Vec<usize>` with no per-node allocation. Queries prune with the
/// squared distance to the splitting plane, which turns the per-point cost from
/// the `O(n_tgt)` of a brute-force scan into `O(log n_tgt)` on average while
/// returning exactly the same match (ties resolved towards the lower index).
pub(super) struct KdTree {
    /// Target point indices, permuted into k-d tree order.
    order: Vec<usize>,
}

impl KdTree {
    /// Build the tree over the first `n` points of `target` (flat xyz).
    pub(super) fn build(target: &[f32], n: usize) -> Self {
        let mut order: Vec<usize> = (0..n).collect();
        Self::split(&mut order, target, 0);
        Self { order }
    }

    /// Build the tree over every point of `target`, which must already be a
    /// valid flat xyz array (length a multiple of 3).
    pub(super) fn build_all(target: &[f32]) -> Self {
        Self::build(target, target.len() / 3)
    }

    /// Recursively partition `idx` around its median along axis `depth % 3`.
    fn split(idx: &mut [usize], target: &[f32], depth: usize) {
        if idx.len() <= 1 {
            return;
        }
        let axis = depth % 3;
        let mid = idx.len() / 2;
        let (left, _median, right) = idx.select_nth_unstable_by(mid, |&a, &b| {
            target[a * 3 + axis]
                .partial_cmp(&target[b * 3 + axis])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Self::split(left, target, depth + 1);
        Self::split(right, target, depth + 1);
    }

    /// Nearest target point to `p`, as `(index, squared distance)`.
    ///
    /// Returns `None` when the cloud is empty or no finite distance exists
    /// (non-finite coordinates compare as "not better" and are skipped).
    pub(super) fn nearest(&self, target: &[f32], p: [f32; 3]) -> Option<(usize, f32)> {
        let mut best = (usize::MAX, f32::MAX);
        Self::search(&self.order, target, p, 0, &mut best);
        (best.0 != usize::MAX).then_some(best)
    }

    /// Depth-first search with splitting-plane pruning.
    ///
    /// The far subtree is only skipped when the splitting plane is strictly
    /// farther than the incumbent — and never on a non-finite comparison — so
    /// every point at exactly the minimum distance is still visited and the
    /// lower-index tie-break matches a brute-force scan.
    fn search(idx: &[usize], target: &[f32], p: [f32; 3], depth: usize, best: &mut (usize, f32)) {
        if idx.is_empty() {
            return;
        }
        let axis = depth % 3;
        let mid = idx.len() / 2;
        let ti = idx[mid];
        let tp = [target[ti * 3], target[ti * 3 + 1], target[ti * 3 + 2]];
        let diff = vec3_sub(p, tp);
        let sq = vec3_dot(diff, diff);
        if sq < best.1 || (sq == best.1 && ti < best.0) {
            *best = (ti, sq);
        }
        let delta = p[axis] - tp[axis];
        let (near, far) = if delta < 0.0 {
            (&idx[..mid], &idx[mid + 1..])
        } else {
            (&idx[mid + 1..], &idx[..mid])
        };
        Self::search(near, target, p, depth + 1, best);
        let plane_sq = delta * delta;
        if plane_sq.is_nan() || plane_sq <= best.1 {
            Self::search(far, target, p, depth + 1, best);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cloud_registration::test_support::{approx_eq, pseudo_cloud};

    /// Brute-force nearest neighbour with the same tie-break as the tree.
    fn brute_nearest(target: &[f32], p: [f32; 3]) -> Option<(usize, f32)> {
        let mut best = (usize::MAX, f32::MAX);
        for ti in 0..target.len() / 3 {
            let tp = [target[ti * 3], target[ti * 3 + 1], target[ti * 3 + 2]];
            let diff = vec3_sub(p, tp);
            let sq = vec3_dot(diff, diff);
            if sq < best.1 {
                best = (ti, sq);
            }
        }
        (best.0 != usize::MAX).then_some(best)
    }

    #[test]
    fn test_kdtree_matches_brute_force() {
        let tgt = pseudo_cloud(200, 4_242, 12.0);
        let probes = pseudo_cloud(50, 77, 14.0);
        let tree = KdTree::build_all(&tgt);
        for p in probes.chunks_exact(3) {
            let q = [p[0], p[1], p[2]];
            let (got_idx, got_sq) = tree
                .nearest(&tgt, q)
                .expect("a finite target cloud always yields a match");
            let (want_idx, want_sq) =
                brute_nearest(&tgt, q).expect("the brute-force scan finds the same match");
            assert_eq!(got_idx, want_idx, "probe {:?}", q);
            assert!(approx_eq(got_sq, want_sq, 1e-4));
        }
    }

    #[test]
    fn test_kdtree_empty_has_no_nearest() {
        let tree = KdTree::build_all(&[]);
        assert!(tree.nearest(&[], [0.0, 0.0, 0.0]).is_none());
    }

    #[test]
    fn test_kdtree_skips_non_finite_targets() {
        // A NaN coordinate never compares as "better", so no match is reported.
        let tgt = vec![f32::NAN, 0.0, 0.0];
        let tree = KdTree::build_all(&tgt);
        assert!(tree.nearest(&tgt, [0.0, 0.0, 0.0]).is_none());
    }

    #[test]
    fn test_kdtree_ties_resolve_to_lowest_index() {
        // Three coincident targets: the lowest index must win.
        let tgt = vec![1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let tree = KdTree::build_all(&tgt);
        let (idx, sq) = tree
            .nearest(&tgt, [1.0, 1.0, 1.0])
            .expect("three finite targets must yield a match");
        assert_eq!(idx, 0);
        assert!(approx_eq(sq, 0.0, 1e-7));
    }

    #[test]
    fn test_kdtree_build_all_matches_explicit_count() {
        let tgt = pseudo_cloud(33, 5, 3.0);
        let a = KdTree::build_all(&tgt);
        let b = KdTree::build(&tgt, tgt.len() / 3);
        assert_eq!(a.order, b.order);
    }
}
