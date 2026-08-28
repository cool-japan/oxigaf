//! Nearest-neighbour search (brute-force and KD-tree accelerated) and
//! Iterative Closest Point alignment.

use kiddo::{KdTree, SquaredEuclidean};

use super::procrustes::{align_procrustes, align_procrustes_rigid, parse_points};
use super::types::{AlignmentError, IcpConfig, IcpResult, SimilarityTransform};

// ─── Nearest-neighbour search ────────────────────────────────────────────────

/// Find nearest point in `target` for each point in `source`.
///
/// Returns indices: `result[i]` = index in target nearest to `source[i]`.
///
/// # Errors
/// Returns [`AlignmentError`] for empty or malformed inputs.
pub fn align_nearest_neighbors(
    source: &[f32],
    target: &[f32],
) -> Result<Vec<usize>, AlignmentError> {
    let n = parse_points(source, 1, "source")?;
    let m = parse_points(target, 1, "target")?;

    let mut indices = Vec::with_capacity(n);
    for i in 0..n {
        let sx = source[i * 3];
        let sy = source[i * 3 + 1];
        let sz = source[i * 3 + 2];
        let mut best_idx = 0usize;
        let mut best_d2 = f32::INFINITY;
        for j in 0..m {
            let dx = sx - target[j * 3];
            let dy = sy - target[j * 3 + 1];
            let dz = sz - target[j * 3 + 2];
            let d2 = dx * dx + dy * dy + dz * dz;
            if d2 < best_d2 {
                best_d2 = d2;
                best_idx = j;
            }
        }
        indices.push(best_idx);
    }
    Ok(indices)
}

/// Find nearest points with distance filtering.
///
/// Returns `(indices, distances)`. Source points with nearest-target distance
/// exceeding `max_dist` get index `usize::MAX`.
///
/// # Errors
/// Returns [`AlignmentError`] for empty or malformed inputs.
pub fn align_nearest_neighbors_filtered(
    source: &[f32],
    target: &[f32],
    max_dist: f32,
) -> Result<(Vec<usize>, Vec<f32>), AlignmentError> {
    let n = parse_points(source, 1, "source")?;
    let m = parse_points(target, 1, "target")?;
    let max_d2 = max_dist * max_dist;

    let mut indices = Vec::with_capacity(n);
    let mut distances = Vec::with_capacity(n);
    for i in 0..n {
        let sx = source[i * 3];
        let sy = source[i * 3 + 1];
        let sz = source[i * 3 + 2];
        let mut best_idx = 0usize;
        let mut best_d2 = f32::INFINITY;
        for j in 0..m {
            let dx = sx - target[j * 3];
            let dy = sy - target[j * 3 + 1];
            let dz = sz - target[j * 3 + 2];
            let d2 = dx * dx + dy * dy + dz * dz;
            if d2 < best_d2 {
                best_d2 = d2;
                best_idx = j;
            }
        }
        if best_d2 <= max_d2 {
            indices.push(best_idx);
            distances.push(best_d2.sqrt());
        } else {
            indices.push(usize::MAX);
            distances.push(best_d2.sqrt());
        }
    }
    Ok((indices, distances))
}

/// Like [`align_nearest_neighbors_filtered`] (same `usize::MAX` rejection
/// convention) but querying a pre-built KD-tree: `O(N log M)` instead of
/// `O(N*M)`, for callers (namely [`align_icp`]) that query the same
/// `target` repeatedly.
pub(super) fn nearest_neighbors_kdtree(
    target_tree: &KdTree<f32, 3>,
    source: &[f32],
    max_dist: f32,
) -> Vec<usize> {
    let max_d2 = max_dist * max_dist;
    let n = source.len() / 3;
    (0..n)
        .map(|i| {
            let p = [source[i * 3], source[i * 3 + 1], source[i * 3 + 2]];
            let nearest = target_tree.nearest_one::<SquaredEuclidean>(&p);
            if nearest.distance <= max_d2 {
                nearest.item as usize
            } else {
                usize::MAX
            }
        })
        .collect()
}

/// Iterative Closest Point alignment.
///
/// Aligns `source` (N×3 flat) to `target` (M×3 flat) by alternating between
/// nearest-neighbour correspondence and Procrustes fitting.
///
/// # Errors
/// Returns [`AlignmentError`] for invalid inputs or if `max_iterations == 0`.
pub fn align_icp(
    source: &[f32],
    target: &[f32],
    config: &IcpConfig,
) -> Result<IcpResult, AlignmentError> {
    let n = parse_points(source, 3, "source")?;
    let m = parse_points(target, 3, "target")?;

    if config.max_iterations == 0 {
        return Err(AlignmentError::InvalidConfig(
            "max_iterations must be > 0".to_string(),
        ));
    }

    // Build the target KD-tree ONCE (target is fixed across iterations).
    let mut target_tree: KdTree<f32, 3> = KdTree::with_capacity(m);
    for i in 0..m {
        target_tree.add(
            &[target[i * 3], target[i * 3 + 1], target[i * 3 + 2]],
            i as u64,
        );
    }

    // Work on a mutable copy of source
    let mut current: Vec<f32> = source.to_vec();
    let mut accumulated = SimilarityTransform::identity();
    let mut rmse_history = Vec::with_capacity(config.max_iterations);
    let mut prev_rmse = f32::INFINITY;
    let mut converged = false;

    for _iter in 0..config.max_iterations {
        // 1. Find correspondences with distance filter (KD-tree accelerated)
        let nn_idx =
            nearest_neighbors_kdtree(&target_tree, &current, config.max_correspondence_dist);

        // 2. Build filtered correspondence point sets
        let mut src_corr: Vec<f32> = Vec::new();
        let mut tgt_corr: Vec<f32> = Vec::new();
        for i in 0..n {
            if nn_idx[i] != usize::MAX {
                let j = nn_idx[i];
                src_corr.extend_from_slice(&current[i * 3..i * 3 + 3]);
                tgt_corr.extend_from_slice(&target[j * 3..j * 3 + 3]);
            }
        }

        if src_corr.len() < 9 {
            // fewer than 3 correspondences — cannot fit
            return Err(AlignmentError::NotEnoughPoints {
                needed: 3,
                got: src_corr.len() / 3,
            });
        }

        // 3. Compute RMSE over valid correspondences
        let n_corr = src_corr.len() / 3;
        let mut mse = 0.0f32;
        for i in 0..n_corr {
            let dx = src_corr[i * 3] - tgt_corr[i * 3];
            let dy = src_corr[i * 3 + 1] - tgt_corr[i * 3 + 1];
            let dz = src_corr[i * 3 + 2] - tgt_corr[i * 3 + 2];
            mse += dx * dx + dy * dy + dz * dz;
        }
        let rmse = (mse / n_corr as f32).sqrt();
        rmse_history.push(rmse);

        // 4. Check convergence
        if (prev_rmse - rmse).abs() < config.convergence_threshold {
            converged = true;
            prev_rmse = rmse;
            // still run the fit for the final transform, then break
            let step = if config.use_scale {
                align_procrustes(&src_corr, &tgt_corr)?
            } else {
                align_procrustes_rigid(&src_corr, &tgt_corr)?
            };
            accumulated = accumulated.compose(&step);
            break;
        }
        prev_rmse = rmse;

        // 5. Fit Procrustes transform on correspondences
        let step = if config.use_scale {
            align_procrustes(&src_corr, &tgt_corr)?
        } else {
            align_procrustes_rigid(&src_corr, &tgt_corr)?
        };

        // 6. Apply step transform to current source
        let next: Vec<f32> = (0..n)
            .flat_map(|i| {
                let p = [current[i * 3], current[i * 3 + 1], current[i * 3 + 2]];
                let q = step.apply(p);
                [q[0], q[1], q[2]]
            })
            .collect();
        current = next;

        // 7. Accumulate: accumulated = accumulated ∘ step
        accumulated = accumulated.compose(&step);
    }

    Ok(IcpResult {
        transform: accumulated,
        final_rmse: prev_rmse,
        n_iterations: rmse_history.len(),
        converged,
        rmse_history,
    })
}
