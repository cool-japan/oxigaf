//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use rayon::prelude::*;

use super::types::{
    CompressedScene, CompressionConfig, CompressionStats, CompressorError, DecompressedScene,
    GcSceneSlices, KMeansConfig, PositionClustering, QuantizedAttribute, ScenePruningConfig,
};

/// Fixed PRNG seed used by [`gc_compress`] for k-means position clustering.
///
/// Compression must be reproducible: the same scene and the same
/// [`CompressionConfig`] have to produce byte-identical output, so the
/// k-means++ seeding inside `gc_compress` cannot draw entropy from the
/// environment. Callers that want a different clustering can drive
/// [`gc_kmeans_positions`] directly with their own `rng_state`.
const GC_CLUSTERING_SEED: u64 = 0x5DEE_CE66_D3D8_1F1D;

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
/// Xorshift64 PRNG (no rand crate).
fn xorshift64(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    if *state == 0 {
        *state = 1;
    }
    *state
}
fn xorshift_f32(state: &mut u64) -> f32 {
    xorshift64(state) as f32 / u64::MAX as f32
}
/// Compute scale and offset for min-max quantization.
/// `scale` maps integer range [-range/2, range/2] back to [min, max].
/// When min == max (constant input), scale is 0 and offset is the constant.
pub(super) fn compute_scale_offset(values: &[f32], range: f32) -> (f32, f32) {
    if values.is_empty() {
        return (0.0, 0.0);
    }
    let mut min_v = values[0];
    let mut max_v = values[0];
    for &v in values.iter().skip(1) {
        if v < min_v {
            min_v = v;
        }
        if v > max_v {
            max_v = v;
        }
    }
    if (max_v - min_v).abs() < f32::EPSILON {
        (0.0, min_v)
    } else {
        let scale = (max_v - min_v) / range;
        let offset = (max_v + min_v) / 2.0;
        (scale, offset)
    }
}
pub(super) fn quantize_to_i16(
    values: &[f32],
    scale: f32,
    offset: f32,
) -> Result<Vec<i16>, CompressorError> {
    if scale == 0.0 {
        return Ok(vec![0i16; values.len()]);
    }
    let mut out = Vec::with_capacity(values.len());
    for &v in values {
        let q = ((v - offset) / scale).round();
        let clamped = q.clamp(-32767.0, 32767.0) as i16;
        out.push(clamped);
    }
    Ok(out)
}
pub(super) fn quantize_to_i8(
    values: &[f32],
    scale: f32,
    offset: f32,
) -> Result<Vec<i8>, CompressorError> {
    if scale == 0.0 {
        return Ok(vec![0i8; values.len()]);
    }
    let mut out = Vec::with_capacity(values.len());
    for &v in values {
        let q = ((v - offset) / scale).round();
        let clamped = q.clamp(-127.0, 127.0) as i8;
        out.push(clamped);
    }
    Ok(out)
}
/// Compute a boolean keep-mask for each Gaussian.
///
/// `true` = keep; `false` = prune.
/// `opacities` are raw logit values; `log_scales` is flat N×3.
pub fn gc_compute_prune_mask(
    opacities: &[f32],
    log_scales: &[f32],
    config: &ScenePruningConfig,
) -> Result<Vec<bool>, CompressorError> {
    let n = opacities.len();
    if n == 0 {
        return Err(CompressorError::EmptyScene);
    }
    if log_scales.len() != n * 3 {
        return Err(CompressorError::DimensionMismatch {
            expected: n * 3,
            got: log_scales.len(),
        });
    }
    if !(0.0..=1.0).contains(&config.opacity_threshold) {
        return Err(CompressorError::InvalidConfig(format!(
            "opacity_threshold must be in [0, 1], got {}",
            config.opacity_threshold
        )));
    }
    if config.preserve_top_fraction < 0.0 || config.preserve_top_fraction > 1.0 {
        return Err(CompressorError::InvalidConfig(format!(
            "preserve_top_fraction must be in [0, 1], got {}",
            config.preserve_top_fraction
        )));
    }
    let mut mask = vec![true; n];
    for i in 0..n {
        let real_opacity = sigmoid(opacities[i]);
        if real_opacity < config.opacity_threshold {
            mask[i] = false;
        }
    }
    for i in 0..n {
        let ls = &log_scales[i * 3..i * 3 + 3];
        if ls[0] > config.max_log_scale
            || ls[1] > config.max_log_scale
            || ls[2] > config.max_log_scale
        {
            mask[i] = false;
        }
        if ls[0] < config.min_log_scale
            && ls[1] < config.min_log_scale
            && ls[2] < config.min_log_scale
        {
            mask[i] = false;
        }
    }
    if config.preserve_top_fraction < 1.0 {
        let kept_indices: Vec<usize> = (0..n).filter(|&i| mask[i]).collect();
        let keep_count = (kept_indices.len() as f32 * config.preserve_top_fraction).ceil() as usize;
        if keep_count < kept_indices.len() {
            let mut sorted = kept_indices.clone();
            // `sigmoid` is strictly monotonic, so comparing raw opacity
            // logits descending yields the identical order as comparing
            // activated (sigmoid'd) opacities, without an `exp()` call per
            // comparison.
            sorted.sort_by(|&a, &b| {
                opacities[b]
                    .partial_cmp(&opacities[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            for &idx in &sorted[keep_count..] {
                mask[idx] = false;
            }
        }
    }
    if let Some(target) = config.target_n_gaussians {
        let kept: Vec<usize> = (0..n).filter(|&i| mask[i]).collect();
        if kept.len() > target {
            let mut sorted = kept.clone();
            sorted.sort_by(|&a, &b| {
                opacities[b]
                    .partial_cmp(&opacities[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            for &idx in &sorted[target..] {
                mask[idx] = false;
            }
        }
    }
    Ok(mask)
}
/// Apply a boolean mask to select rows from a flat N×K array.
///
/// Returns a new flat array containing only the rows where `mask[i]` is true.
pub fn gc_apply_mask_flat(
    data: &[f32],
    mask: &[bool],
    k: usize,
) -> Result<Vec<f32>, CompressorError> {
    let n = mask.len();
    if k == 0 {
        return Ok(Vec::new());
    }
    if data.len() != n * k {
        return Err(CompressorError::DimensionMismatch {
            expected: n * k,
            got: data.len(),
        });
    }
    let mut out = Vec::new();
    for i in 0..n {
        if mask[i] {
            out.extend_from_slice(&data[i * k..(i + 1) * k]);
        }
    }
    Ok(out)
}
/// Return indices (sorted by sigmoid-opacity descending) of top-N Gaussians.
///
/// If `n >= opacities.len()`, all indices are returned.
pub fn gc_prune_to_topn(opacities: &[f32], n: usize) -> Result<Vec<usize>, CompressorError> {
    let total = opacities.len();
    if total == 0 {
        return Err(CompressorError::EmptyScene);
    }
    if n == 0 {
        return Ok(Vec::new());
    }
    let keep = n.min(total);
    let mut indices: Vec<usize> = (0..total).collect();
    // See `gc_compute_prune_mask`: compare raw logits, not `sigmoid`-activated
    // values — the ordering is identical since `sigmoid` is monotonic, but
    // this way sorting costs zero `exp()` calls instead of O(n log n) of them.
    indices.sort_by(|&a, &b| {
        opacities[b]
            .partial_cmp(&opacities[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    indices.truncate(keep);
    Ok(indices)
}
fn sq_dist3(a: &[f32], a_off: usize, b: &[f32], b_off: usize) -> f32 {
    let dx = a[a_off] - b[b_off];
    let dy = a[a_off + 1] - b[b_off + 1];
    let dz = a[a_off + 2] - b[b_off + 2];
    dx * dx + dy * dy + dz * dz
}
/// K-means++ initialization: returns K seed center indices.
pub fn gc_kmeans_plus_plus_init(
    positions: &[f32],
    k: usize,
    rng_state: &mut u64,
) -> Result<Vec<usize>, CompressorError> {
    if positions.is_empty() {
        return Err(CompressorError::EmptyScene);
    }
    let n = positions.len() / 3;
    if n == 0 {
        return Err(CompressorError::EmptyScene);
    }
    if k > n {
        return Err(CompressorError::InvalidConfig(format!(
            "k ({}) > number of points ({})",
            k, n
        )));
    }
    if k == 0 {
        return Ok(Vec::new());
    }
    let mut centers: Vec<usize> = Vec::with_capacity(k);
    let first = (xorshift_f32(rng_state) * n as f32) as usize;
    let first = first.min(n - 1);
    centers.push(first);
    let mut distances = vec![f32::MAX; n];
    for _ in 1..k {
        let last = *centers.last().ok_or(CompressorError::EmptyScene)?;
        for (i, dist_val) in distances.iter_mut().enumerate().take(n) {
            let d = sq_dist3(positions, i * 3, positions, last * 3);
            if d < *dist_val {
                *dist_val = d;
            }
        }
        let total: f32 = distances.iter().sum();
        if total <= 0.0 {
            for i in 0..n {
                if !centers.contains(&i) {
                    centers.push(i);
                    break;
                }
            }
        } else {
            let threshold = xorshift_f32(rng_state) * total;
            let mut cumsum = 0.0f32;
            let mut chosen = n - 1;
            for (i, &dist_val) in distances.iter().enumerate().take(n) {
                cumsum += dist_val;
                if cumsum >= threshold {
                    chosen = i;
                    break;
                }
            }
            centers.push(chosen);
        }
    }
    Ok(centers)
}
/// Run k-means clustering on 3D positions.
///
/// Returns `(cluster_centers: Vec<f32> [K×3], assignments: Vec<usize> [N])`.
///
/// # Convergence
/// Lloyd's algorithm runs for at most `config.n_iterations` iterations,
/// stopping early once the largest per-centre shift drops below
/// `config.tolerance`. Two distinct outcomes are *not* treated the same:
/// - `config.n_iterations == 0` (a degenerate configuration — the loop body
///   never runs, so convergence is impossible by construction) is a hard
///   error: [`CompressorError::KMeansNoConvergence`].
/// - Exhausting a *positive* iteration budget without the shift dropping
///   below `tolerance` is not an error — real k-means runs are not
///   guaranteed to converge within any fixed budget, and the centers/
///   assignments found so far are still a valid (if not fully settled)
///   clustering. This case logs a `tracing::warn!` and returns `Ok` with
///   the best result found, rather than promoting ordinary non-convergence
///   to a hard failure.
pub fn gc_kmeans_positions(
    positions: &[f32],
    config: &KMeansConfig,
    rng_state: &mut u64,
) -> Result<(Vec<f32>, Vec<usize>), CompressorError> {
    if positions.is_empty() {
        return Err(CompressorError::EmptyScene);
    }
    let n = positions.len() / 3;
    if n == 0 {
        return Err(CompressorError::EmptyScene);
    }
    if !positions.len().is_multiple_of(3) {
        return Err(CompressorError::DimensionMismatch {
            expected: (positions.len() / 3) * 3,
            got: positions.len(),
        });
    }
    let k = config.n_clusters;
    if k == 0 {
        return Err(CompressorError::InvalidConfig(
            "n_clusters must be > 0".to_string(),
        ));
    }
    if k > n {
        return Err(CompressorError::InvalidConfig(format!(
            "n_clusters ({}) > n_points ({})",
            k, n
        )));
    }
    let seed_indices = gc_kmeans_plus_plus_init(positions, k, rng_state)?;
    let mut centers: Vec<f32> = Vec::with_capacity(k * 3);
    for &idx in &seed_indices {
        centers.extend_from_slice(&positions[idx * 3..idx * 3 + 3]);
    }
    let mut assignments = vec![0usize; n];
    let mut converged = false;
    for _iter in 0..config.n_iterations {
        // Assignment step: each point's nearest centre is independent of
        // every other point's, so this O(n*k) scan (the dominant cost per
        // iteration) is parallelised across points. The centroid-update step
        // below stays serial: it accumulates into shared per-cluster sums in
        // a fixed order, which keeps floating-point results deterministic
        // (a parallel reduction would sum in a different, run-dependent
        // order) and its own cost is only O(n), not the bottleneck.
        assignments
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, assignment)| {
                let mut best_dist = f32::MAX;
                let mut best_k = 0usize;
                for ci in 0..k {
                    let d = sq_dist3(positions, i * 3, &centers, ci * 3);
                    if d < best_dist {
                        best_dist = d;
                        best_k = ci;
                    }
                }
                *assignment = best_k;
            });
        let mut new_centers = vec![0.0f32; k * 3];
        let mut counts = vec![0usize; k];
        for i in 0..n {
            let ci = assignments[i];
            new_centers[ci * 3] += positions[i * 3];
            new_centers[ci * 3 + 1] += positions[i * 3 + 1];
            new_centers[ci * 3 + 2] += positions[i * 3 + 2];
            counts[ci] += 1;
        }
        for ci in 0..k {
            if counts[ci] > 0 {
                let c = counts[ci] as f32;
                new_centers[ci * 3] /= c;
                new_centers[ci * 3 + 1] /= c;
                new_centers[ci * 3 + 2] /= c;
            } else {
                let ri = (xorshift_f32(rng_state) * n as f32) as usize;
                let ri = ri.min(n - 1);
                new_centers[ci * 3] = positions[ri * 3];
                new_centers[ci * 3 + 1] = positions[ri * 3 + 1];
                new_centers[ci * 3 + 2] = positions[ri * 3 + 2];
            }
        }
        let mut max_shift: f32 = 0.0;
        for ci in 0..k {
            let shift = sq_dist3(&new_centers, ci * 3, &centers, ci * 3).sqrt();
            if shift > max_shift {
                max_shift = shift;
            }
        }
        centers = new_centers;
        if max_shift < config.tolerance {
            converged = true;
            break;
        }
    }
    if !converged {
        if config.n_iterations == 0 {
            return Err(CompressorError::KMeansNoConvergence);
        }
        // Ran every requested iteration without the max centre shift dropping
        // below `tolerance`. This is not necessarily a problem (k-means is
        // best-effort and this is still the best clustering found), so it is
        // not promoted to a hard error here — but the caller should know,
        // since previously this state was indistinguishable from a clean
        // convergence.
        tracing::warn!(
            "k-means position clustering did not converge within {} iteration(s) (tolerance \
             {}); returning the best centers/assignments found so far.",
            config.n_iterations,
            config.tolerance,
        );
    }
    Ok((centers, assignments))
}
/// Compute per-point residuals between positions and their assigned cluster centers.
///
/// Returns residuals flat N×3.
pub fn gc_cluster_residuals(
    positions: &[f32],
    assignments: &[usize],
    centers: &[f32],
) -> Result<Vec<f32>, CompressorError> {
    if positions.is_empty() {
        return Err(CompressorError::EmptyScene);
    }
    let n = positions.len() / 3;
    if !positions.len().is_multiple_of(3) {
        return Err(CompressorError::DimensionMismatch {
            expected: (n) * 3,
            got: positions.len(),
        });
    }
    if assignments.len() != n {
        return Err(CompressorError::DimensionMismatch {
            expected: n,
            got: assignments.len(),
        });
    }
    if centers.is_empty() || !centers.len().is_multiple_of(3) {
        return Err(CompressorError::InvalidConfig(format!(
            "centers must be a non-empty flat array of 3D points (length a multiple of 3), got \
             length {}",
            centers.len()
        )));
    }
    let k = centers.len() / 3;
    let mut residuals = vec![0.0f32; n * 3];
    for i in 0..n {
        let ci = assignments[i];
        if ci >= k {
            // `k` is >= 1 here (guarded above), so this never subtracts.
            return Err(CompressorError::DimensionMismatch {
                expected: k,
                got: ci,
            });
        }
        residuals[i * 3] = positions[i * 3] - centers[ci * 3];
        residuals[i * 3 + 1] = positions[i * 3 + 1] - centers[ci * 3 + 1];
        residuals[i * 3 + 2] = positions[i * 3 + 2] - centers[ci * 3 + 2];
    }
    Ok(residuals)
}
/// Compress an entire Gaussian scene.
///
/// Performs pruning, optional position clustering, and per-attribute
/// quantization.
///
/// # Position clustering
/// With `config.use_position_clustering`, positions are run through k-means
/// (see [`gc_kmeans_positions`]) and what gets quantized is each Gaussian's
/// **residual** from its assigned centre. The codebook and the per-Gaussian
/// assignment are persisted on [`CompressedScene::position_clustering`], so
/// [`gc_decompress`] can rebuild absolute positions exactly. Residuals span
/// a fraction of the scene's extent, so a fixed bit width resolves them far
/// more finely than absolute coordinates — the accuracy win that pays for
/// the codebook plus one index per Gaussian.
///
/// Clustering is deterministic: it seeds k-means from a fixed internal
/// constant, never from the environment, so the same scene and config always
/// compress to the same bytes. `kmeans.n_clusters` is clamped down to the number
/// of Gaussians that survived pruning (k-means needs at least one point per
/// centre); a zero `n_clusters` or `n_iterations` is rejected as
/// [`CompressorError::InvalidConfig`], since neither can produce a
/// clustering.
///
/// # Pruning provenance
/// Pruning happens *before* quantization, so compressed row `j` is the j-th
/// survivor, not original row `j`. The mapping is recorded on
/// [`CompressedScene::kept_indices`] (ascending original indices) so callers
/// — [`gc_compute_stats`] in particular — can realign survivors with their
/// originals.
pub fn gc_compress(
    slices: GcSceneSlices<'_>,
    config: &CompressionConfig,
) -> Result<CompressedScene, CompressorError> {
    let GcSceneSlices {
        positions,
        rotations,
        scales,
        opacities,
        sh_dc,
        sh_rest,
        n_rest_per_gaussian,
    } = slices;
    let n_pos = positions.len();
    if n_pos == 0 || opacities.is_empty() {
        return Err(CompressorError::EmptyScene);
    }
    if !n_pos.is_multiple_of(3) {
        return Err(CompressorError::DimensionMismatch {
            expected: (n_pos / 3) * 3,
            got: n_pos,
        });
    }
    let n = n_pos / 3;
    if rotations.len() != n * 4 {
        return Err(CompressorError::DimensionMismatch {
            expected: n * 4,
            got: rotations.len(),
        });
    }
    if scales.len() != n * 3 {
        return Err(CompressorError::DimensionMismatch {
            expected: n * 3,
            got: scales.len(),
        });
    }
    if opacities.len() != n {
        return Err(CompressorError::DimensionMismatch {
            expected: n,
            got: opacities.len(),
        });
    }
    if sh_dc.len() != n * 3 {
        return Err(CompressorError::DimensionMismatch {
            expected: n * 3,
            got: sh_dc.len(),
        });
    }
    if sh_rest.len() != n * n_rest_per_gaussian {
        return Err(CompressorError::DimensionMismatch {
            expected: n * n_rest_per_gaussian,
            got: sh_rest.len(),
        });
    }
    let prune_mask = gc_compute_prune_mask(opacities, scales, &config.pruning)?;
    let kept_positions = gc_apply_mask_flat(positions, &prune_mask, 3)?;
    let kept_rotations = gc_apply_mask_flat(rotations, &prune_mask, 4)?;
    let kept_scales = gc_apply_mask_flat(scales, &prune_mask, 3)?;
    let kept_opacities: Vec<f32> = opacities
        .iter()
        .zip(prune_mask.iter())
        .filter_map(|(&v, &keep)| if keep { Some(v) } else { None })
        .collect();
    let kept_sh_dc = gc_apply_mask_flat(sh_dc, &prune_mask, 3)?;
    let kept_sh_rest = if n_rest_per_gaussian > 0 {
        gc_apply_mask_flat(sh_rest, &prune_mask, n_rest_per_gaussian)?
    } else {
        Vec::new()
    };
    let n_kept = kept_opacities.len();
    if n_kept == 0 {
        return Err(CompressorError::EmptyScene);
    }
    // Record which ORIGINAL Gaussian each survivor came from. Both
    // `gc_apply_mask_flat` and the `kept_opacities` filter above walk the
    // mask in index order, so survivor `j` is original `kept_indices[j]` and
    // the list is ascending. `gc_compute_stats` needs exactly this to
    // compare a survivor's dequantized value against the right original.
    let mut kept_indices: Vec<u32> = Vec::with_capacity(n_kept);
    for (i, &keep) in prune_mask.iter().enumerate() {
        if keep {
            let idx = u32::try_from(i).map_err(|_| {
                CompressorError::InvalidConfig(format!(
                    "Gaussian index {i} exceeds the u32 index space used by \
                     CompressedScene::kept_indices"
                ))
            })?;
            kept_indices.push(idx);
        }
    }
    // Position clustering: quantize residuals from a k-means codebook rather
    // than absolute coordinates, and persist the codebook + assignments so
    // decompression can undo it. (An earlier version quantized the residuals
    // but had nowhere to store the codebook, so `gc_decompress` handed the
    // residuals back as world positions and the scene silently collapsed
    // toward the origin; a later version disabled the feature outright with
    // a warning. Both are gone: `CompressedScene` now carries the codebook.)
    let (final_positions, position_clustering) = if config.use_position_clustering {
        let requested_k = config.kmeans.n_clusters;
        if requested_k == 0 {
            return Err(CompressorError::InvalidConfig(
                "use_position_clustering is enabled but kmeans.n_clusters is 0: a codebook needs \
                 at least one centre"
                    .to_string(),
            ));
        }
        if config.kmeans.n_iterations == 0 {
            return Err(CompressorError::InvalidConfig(
                "use_position_clustering is enabled but kmeans.n_iterations is 0: Lloyd's \
                 algorithm would never run, so no clustering could be produced"
                    .to_string(),
            ));
        }
        // k-means needs at least one point per centre. A scene can easily
        // have fewer survivors than the (deliberately generous) default
        // codebook size, so clamp instead of failing.
        let k = requested_k.min(n_kept);
        if k < requested_k {
            tracing::warn!(
                "position clustering: reducing k from {} to {}, the number of Gaussians \
                 surviving pruning",
                requested_k,
                k,
            );
        }
        let kmeans_config = KMeansConfig {
            n_clusters: k,
            ..config.kmeans.clone()
        };
        let mut rng_state = GC_CLUSTERING_SEED;
        let (centers, assignments) =
            gc_kmeans_positions(&kept_positions, &kmeans_config, &mut rng_state)?;
        let residuals = gc_cluster_residuals(&kept_positions, &assignments, &centers)?;
        let mut assignments_u32: Vec<u32> = Vec::with_capacity(assignments.len());
        for &assignment in &assignments {
            let idx = u32::try_from(assignment).map_err(|_| {
                CompressorError::InvalidConfig(format!(
                    "cluster index {assignment} exceeds the u32 index space used by \
                     PositionClustering::assignments"
                ))
            })?;
            assignments_u32.push(idx);
        }
        (
            residuals,
            Some(PositionClustering {
                centers,
                assignments: assignments_u32,
            }),
        )
    } else {
        (kept_positions, None)
    };
    let q_positions = QuantizedAttribute::quantize(&final_positions, config.position_precision)?;
    let q_rotations = QuantizedAttribute::quantize(&kept_rotations, config.rotation_precision)?;
    let q_scales = QuantizedAttribute::quantize(&kept_scales, config.scale_precision)?;
    let q_opacities = QuantizedAttribute::quantize(&kept_opacities, config.opacity_precision)?;
    let q_sh_dc = QuantizedAttribute::quantize(&kept_sh_dc, config.sh_dc_precision)?;
    let q_sh_rest = if n_rest_per_gaussian > 0 {
        QuantizedAttribute::quantize(&kept_sh_rest, config.sh_rest_precision)?
    } else {
        QuantizedAttribute::quantize(&[], config.sh_rest_precision)?
    };
    Ok(CompressedScene {
        positions: q_positions,
        rotations: q_rotations,
        scales: q_scales,
        opacities: q_opacities,
        sh_dc: q_sh_dc,
        sh_rest: q_sh_rest,
        n_gaussians: n_kept,
        n_sh_rest: n_rest_per_gaussian,
        position_clustering,
        kept_indices,
        compression_config: config.clone(),
    })
}
/// Decompress a compressed scene back to f32 arrays.
///
/// Positions go through [`CompressedScene::reconstruct_positions`], which
/// adds the k-means centre back onto each residual when the scene was
/// compressed with position clustering.
pub fn gc_decompress(scene: &CompressedScene) -> Result<DecompressedScene, CompressorError> {
    let positions = scene.reconstruct_positions()?;
    let rotations = scene.rotations.dequantize();
    let scales = scene.scales.dequantize();
    let opacities = scene.opacities.dequantize();
    let sh_dc = scene.sh_dc.dequantize();
    let sh_rest = scene.sh_rest.dequantize();
    let n = scene.n_gaussians;
    Ok(DecompressedScene {
        positions,
        rotations,
        scales,
        opacities,
        sh_dc,
        sh_rest,
        n_gaussians: n,
    })
}
/// Root-mean-square error between two f32 slices, truncated to the shorter length.
fn rmse(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    if len == 0 {
        return 0.0;
    }
    let mse: f32 = a[..len]
        .iter()
        .zip(b[..len].iter())
        .map(|(&x, &y)| (x - y) * (x - y))
        .sum::<f32>()
        / len as f32;
    mse.sqrt()
}
/// Gather the rows named by `kept_indices` out of a flat N×`k` array.
///
/// Returns `None` when `data` is too short to hold every requested row —
/// the caller must then fall back rather than compare against a truncated,
/// silently misaligned reference.
fn gather_rows(data: &[f32], kept_indices: &[u32], k: usize) -> Option<Vec<f32>> {
    let mut out = Vec::with_capacity(kept_indices.len().saturating_mul(k));
    for &idx in kept_indices {
        let start = (idx as usize).checked_mul(k)?;
        let end = start.checked_add(k)?;
        if end > data.len() {
            return None;
        }
        out.extend_from_slice(&data[start..end]);
    }
    Some(out)
}
/// Compute compression statistics comparing original arrays to a compressed scene.
///
/// `position_quantization_rmse`/`opacity_quantization_rmse` are only
/// meaningful when each compressed value is compared against the *same*
/// Gaussian it came from. `gc_compress` prunes Gaussians (by opacity
/// threshold, scale bounds, and/or top-N/fraction truncation) *before*
/// quantizing, so compressed row `j` is the j-th SURVIVOR, not original
/// index `j` — and with `preserve_top_fraction`/`target_n_gaussians` the
/// survivors are not even a prefix of the originals.
///
/// [`CompressedScene::kept_indices`] records the survivor→original mapping,
/// so both RMSEs are computed exactly: each survivor is compared against
/// `original[kept_indices[j]]`. Positions go through
/// [`CompressedScene::reconstruct_positions`], so a clustered scene is
/// measured on rebuilt world positions rather than on raw residuals.
///
/// The `NaN` fallback survives for scenes not produced by `gc_compress`: if
/// `kept_indices` is missing/malformed or the supplied originals are too
/// short to contain every kept index, and the counts show pruning happened,
/// the RMSEs are `NaN` with a `tracing::warn!` rather than a misleading,
/// index-misaligned number. When nothing was pruned, survivor `j` provably
/// *is* original `j`, so the direct comparison is used.
pub fn gc_compute_stats(
    original_positions: &[f32],
    original_opacities: &[f32],
    scene: &CompressedScene,
) -> Result<CompressionStats, CompressorError> {
    let n_before = if original_positions.len() >= 3 {
        original_positions.len() / 3
    } else {
        original_opacities.len()
    };
    let n_after = scene.n_gaussians;
    let pruned_fraction = if n_before == 0 {
        0.0
    } else {
        let pruned = n_before.saturating_sub(n_after);
        pruned as f32 / n_before as f32
    };
    let compressed_mb = scene.compressed_bytes() as f32 / (1024.0 * 1024.0);
    let uncompressed_mb = scene.uncompressed_bytes() as f32 / (1024.0 * 1024.0);
    let compression_ratio = scene.compression_ratio();

    let deq_positions = scene.reconstruct_positions()?;
    let deq_opacities = scene.opacities.dequantize();

    // Exact path: realign every survivor with the original it came from.
    let realigned = if scene.kept_indices.len() == n_after {
        match (
            gather_rows(original_positions, &scene.kept_indices, 3),
            gather_rows(original_opacities, &scene.kept_indices, 1),
        ) {
            (Some(ref_positions), Some(ref_opacities)) => Some((ref_positions, ref_opacities)),
            _ => None,
        }
    } else {
        None
    };

    let (pos_rmse, op_rmse) = match realigned {
        Some((ref_positions, ref_opacities)) => (
            rmse(&deq_positions, &ref_positions),
            rmse(&deq_opacities, &ref_opacities),
        ),
        // Nothing was pruned, so survivor j is original j and the direct
        // comparison is already aligned.
        None if n_after == n_before => (
            rmse(&deq_positions, original_positions),
            rmse(&deq_opacities, original_opacities),
        ),
        None => {
            // Deliberately reports the two counts rather than a subtraction:
            // this branch is also reached when the supplied originals are
            // SHORTER than the compressed scene, where a "pruned" count
            // would render as a nonsensical zero.
            tracing::warn!(
                "gc_compute_stats: the compressed scene holds {} Gaussian(s) but {} original(s) \
                 were supplied, and the survivor→original mapping \
                 (CompressedScene::kept_indices) is missing, the wrong length, or names indices \
                 past those originals — so survivors cannot be realigned. \
                 position_quantization_rmse/opacity_quantization_rmse are reported as NaN rather \
                 than a misleading, index-misaligned comparison.",
                n_after,
                n_before,
            );
            (f32::NAN, f32::NAN)
        }
    };

    Ok(CompressionStats {
        n_gaussians_before: n_before,
        n_gaussians_after: n_after,
        pruned_fraction,
        compressed_mb,
        uncompressed_mb,
        compression_ratio,
        position_quantization_rmse: pos_rmse,
        opacity_quantization_rmse: op_rmse,
    })
}
/// Format compression statistics as a human-readable string.
pub fn gc_format_stats(stats: &CompressionStats) -> String {
    format!(
        "Compression Stats:\n\
         Gaussians: {} → {} (pruned {:.1}%)\n\
         Size: {:.3} MB → {:.3} MB (ratio {:.2}x)\n\
         Position RMSE: {:.6}\n\
         Opacity RMSE:  {:.6}",
        stats.n_gaussians_before,
        stats.n_gaussians_after,
        stats.pruned_fraction * 100.0,
        stats.uncompressed_mb,
        stats.compressed_mb,
        stats.compression_ratio,
        stats.position_quantization_rmse,
        stats.opacity_quantization_rmse,
    )
}
/// Format compression configuration as a human-readable string.
pub fn gc_format_config(config: &CompressionConfig) -> String {
    format!(
        "CompressionConfig:\n\
         position={}  rotation={}  scale={}\n\
         opacity={}   sh_dc={}     sh_rest={}\n\
         position_clustering={}  kmeans_k={}\n\
         pruning: opacity_thresh={:.4}  max_log_scale={:.1}  min_log_scale={:.1}",
        config.position_precision.as_str(),
        config.rotation_precision.as_str(),
        config.scale_precision.as_str(),
        config.opacity_precision.as_str(),
        config.sh_dc_precision.as_str(),
        config.sh_rest_precision.as_str(),
        config.use_position_clustering,
        config.kmeans.n_clusters,
        config.pruning.opacity_threshold,
        config.pruning.max_log_scale,
        config.pruning.min_log_scale,
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
//
// `gaussian_compressor::tests` (declared in mod.rs, sibling to this module)
// covers the pipeline end-to-end; these are focused regression tests for the
// specific bugs fixed in this file, living here since this module has no
// sibling test file of its own before this change.

#[cfg(test)]
mod tests {
    use super::super::types::QuantizationPrecision;
    use super::*;

    /// `(positions, rotations, scales, opacities, sh_dc, sh_rest)`.
    type SceneTuple = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);

    fn make_scene(n: usize) -> SceneTuple {
        let positions: Vec<f32> = (0..n)
            .flat_map(|i| {
                let f = i as f32;
                [f * 0.37 - 5.0, f * 0.19 + 2.0, f * 0.53 - 1.0]
            })
            .collect();
        let rotations: Vec<f32> = (0..n).flat_map(|_| [0.0f32, 0.0, 0.0, 1.0]).collect();
        let scales: Vec<f32> = (0..n).flat_map(|_| [-1.0f32, -1.0, -1.0]).collect();
        let opacities: Vec<f32> = (0..n).map(|i| 2.0 - (i as f32) * 0.001).collect();
        let sh_dc: Vec<f32> = (0..n).flat_map(|_| [0.1f32, 0.2, 0.3]).collect();
        let sh_rest: Vec<f32> = Vec::new();
        (positions, rotations, scales, opacities, sh_dc, sh_rest)
    }

    fn full_precision_config() -> CompressionConfig {
        CompressionConfig {
            position_precision: QuantizationPrecision::Full,
            rotation_precision: QuantizationPrecision::Full,
            scale_precision: QuantizationPrecision::Full,
            opacity_precision: QuantizationPrecision::Full,
            sh_dc_precision: QuantizationPrecision::Full,
            sh_rest_precision: QuantizationPrecision::Full,
            pruning: ScenePruningConfig {
                opacity_threshold: 0.0,
                max_log_scale: 100.0,
                min_log_scale: -100.0,
                target_n_gaussians: None,
                preserve_top_fraction: 1.0,
            },
            use_position_clustering: false,
            kmeans: KMeansConfig {
                n_clusters: 4,
                n_iterations: 20,
                tolerance: 1e-4,
            },
        }
    }

    // -----------------------------------------------------------------------
    // gc_cluster_residuals: empty/malformed centers must error, not panic
    // -----------------------------------------------------------------------

    #[test]
    fn cluster_residuals_empty_centers_is_error_not_panic() {
        let positions = vec![1.0f32, 2.0, 3.0];
        let assignments = vec![0usize];
        let centers: Vec<f32> = vec![];
        let result = gc_cluster_residuals(&positions, &assignments, &centers);
        assert!(
            matches!(result, Err(CompressorError::InvalidConfig(_))),
            "expected InvalidConfig, got {result:?}"
        );
    }

    #[test]
    fn cluster_residuals_centers_not_multiple_of_3_is_error() {
        let positions = vec![1.0f32, 2.0, 3.0];
        let assignments = vec![0usize];
        let centers: Vec<f32> = vec![0.0, 0.0]; // length 2, not a multiple of 3
        let result = gc_cluster_residuals(&positions, &assignments, &centers);
        assert!(matches!(result, Err(CompressorError::InvalidConfig(_))));
    }

    #[test]
    fn cluster_residuals_out_of_range_assignment_is_error_not_underflow_panic() {
        let positions = vec![1.0f32, 2.0, 3.0];
        let assignments = vec![5usize]; // only 1 center (k=1) exists
        let centers = vec![0.0f32, 0.0, 0.0];
        let result = gc_cluster_residuals(&positions, &assignments, &centers);
        assert!(matches!(
            result,
            Err(CompressorError::DimensionMismatch {
                expected: 1,
                got: 5
            })
        ));
    }

    // -----------------------------------------------------------------------
    // gc_compress: use_position_clustering must never corrupt positions
    // -----------------------------------------------------------------------

    fn compress_slices(
        scene: &SceneTuple,
        config: &CompressionConfig,
    ) -> Result<CompressedScene, CompressorError> {
        let (pos, rot, scl, op, shd, shr) = scene;
        gc_compress(
            GcSceneSlices {
                positions: pos,
                rotations: rot,
                scales: scl,
                opacities: op,
                sh_dc: shd,
                sh_rest: shr,
                n_rest_per_gaussian: 0,
            },
            config,
        )
    }

    #[test]
    fn compress_with_position_clustering_does_not_corrupt_positions() {
        // Regression for the critical bug: positions used to be replaced by
        // near-zero k-means residuals with nowhere to store the centers, so
        // decompression silently returned a scene collapsed toward the
        // origin. The codebook is now persisted, so with Full (lossless)
        // precision decompressed positions must match the originals.
        let n = 40usize;
        let scene_data = make_scene(n);
        let mut config = full_precision_config();
        config.use_position_clustering = true;
        config.kmeans.n_clusters = 4;

        let scene = compress_slices(&scene_data, &config).expect("compress with clustering");
        let decomp = gc_decompress(&scene).expect("decompress");
        assert_eq!(decomp.n_gaussians, n);
        assert_eq!(decomp.positions.len(), scene_data.0.len());
        for (a, b) in decomp.positions.iter().zip(scene_data.0.iter()) {
            assert!(
                (a - b).abs() < 1e-3,
                "position corrupted by clustering: {a} vs {b}"
            );
        }
    }

    #[test]
    fn compress_with_position_clustering_persists_a_real_codebook() {
        // The feature must actually do something: a codebook of the requested
        // size, one assignment per Gaussian, and stored positions that are
        // residuals (small) rather than absolute coordinates (large).
        let n = 40usize;
        let scene_data = make_scene(n);
        let mut config = full_precision_config();
        config.use_position_clustering = true;
        config.kmeans.n_clusters = 4;

        let scene = compress_slices(&scene_data, &config).expect("compress with clustering");
        let clustering = scene
            .position_clustering
            .as_ref()
            .expect("clustering must be persisted, not discarded");
        assert_eq!(clustering.n_clusters(), 4);
        assert_eq!(clustering.centers.len(), 4 * 3);
        assert_eq!(clustering.assignments.len(), n);
        assert!(clustering.assignments.iter().all(|&a| (a as usize) < 4));

        let residuals = scene.positions.dequantize();
        let max_abs_position = scene_data.0.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let max_abs_residual = residuals.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(
            max_abs_residual < max_abs_position * 0.5,
            "stored values should be residuals (max |r| = {max_abs_residual}) not absolute \
             positions (max |p| = {max_abs_position})"
        );
    }

    #[test]
    fn position_clustering_sharpens_byte_precision_positions() {
        // The point of residual coding: at a fixed bit width, residuals from
        // a codebook resolve far more finely than absolute coordinates.
        let n = 64usize;
        let positions: Vec<f32> = (0..n)
            .flat_map(|i| {
                let (base, j) = if i < n / 2 {
                    (-100.0f32, i as f32)
                } else {
                    (100.0f32, (i - n / 2) as f32)
                };
                [base + j * 0.01, base + j * 0.02, base + j * 0.03]
            })
            .collect();
        let rotations: Vec<f32> = (0..n).flat_map(|_| [0.0f32, 0.0, 0.0, 1.0]).collect();
        let scales = vec![-1.0f32; n * 3];
        let opacities = vec![3.0f32; n];
        let sh_dc = vec![0.1f32; n * 3];
        let scene_data = (
            positions.clone(),
            rotations,
            scales,
            opacities.clone(),
            sh_dc,
            Vec::new(),
        );

        let mut plain = full_precision_config();
        plain.position_precision = QuantizationPrecision::Byte;
        let plain_scene = compress_slices(&scene_data, &plain).expect("compress unclustered");
        let plain_stats = gc_compute_stats(&positions, &opacities, &plain_scene).expect("stats");

        let mut clustered = plain.clone();
        clustered.use_position_clustering = true;
        clustered.kmeans.n_clusters = 2;
        let clustered_scene = compress_slices(&scene_data, &clustered).expect("compress clustered");
        let clustered_stats =
            gc_compute_stats(&positions, &opacities, &clustered_scene).expect("stats");

        assert!(
            clustered_stats.position_quantization_rmse
                < plain_stats.position_quantization_rmse * 0.25,
            "clustering should sharply reduce position error: clustered {} vs plain {}",
            clustered_stats.position_quantization_rmse,
            plain_stats.position_quantization_rmse
        );
    }

    #[test]
    fn clustering_payload_counts_toward_compressed_bytes() {
        let n = 40usize;
        let scene_data = make_scene(n);
        let plain = full_precision_config();
        let plain_scene = compress_slices(&scene_data, &plain).expect("compress");

        let mut clustered = plain.clone();
        clustered.use_position_clustering = true;
        clustered.kmeans.n_clusters = 4;
        let clustered_scene = compress_slices(&scene_data, &clustered).expect("compress clustered");
        let clustering = clustered_scene
            .position_clustering
            .as_ref()
            .expect("clustering");
        // 4 centres → a one-byte codebook index per Gaussian.
        assert_eq!(clustering.index_byte_width(), 1);
        assert_eq!(clustering.byte_size(), 4 * 3 * 4 + n);
        assert_eq!(
            clustered_scene.compressed_bytes(),
            plain_scene.compressed_bytes() + clustering.byte_size(),
            "the codebook and indices are payload and must be accounted for"
        );
    }

    #[test]
    fn clustering_is_deterministic_across_runs() {
        // gc_compress seeds k-means from a fixed constant, never from the
        // environment: identical input plus identical config must produce an
        // identical codebook, or compression stops being reproducible.
        let scene_data = make_scene(50);
        let mut config = full_precision_config();
        config.use_position_clustering = true;
        config.kmeans.n_clusters = 5;
        let first = compress_slices(&scene_data, &config).expect("compress");
        let second = compress_slices(&scene_data, &config).expect("compress again");
        let (a, b) = (
            first.position_clustering.as_ref().expect("clustering"),
            second.position_clustering.as_ref().expect("clustering"),
        );
        assert_eq!(a.centers, b.centers);
        assert_eq!(a.assignments, b.assignments);
        assert_eq!(first.positions.dequantize(), second.positions.dequantize());
    }

    #[test]
    fn clustering_clamps_k_to_the_surviving_gaussian_count() {
        // The default codebook (256) is far larger than this scene; k-means
        // needs one point per centre, so k must clamp rather than error.
        let n = 5usize;
        let scene_data = make_scene(n);
        let mut config = full_precision_config();
        config.use_position_clustering = true;
        config.kmeans.n_clusters = 256;
        let scene = compress_slices(&scene_data, &config).expect("compress with clamped k");
        let clustering = scene.position_clustering.as_ref().expect("clustering");
        assert_eq!(clustering.n_clusters(), n);
    }

    #[test]
    fn clustering_with_zero_clusters_is_invalid_config() {
        let scene_data = make_scene(10);
        let mut config = full_precision_config();
        config.use_position_clustering = true;
        config.kmeans.n_clusters = 0;
        let result = compress_slices(&scene_data, &config);
        assert!(
            matches!(result, Err(CompressorError::InvalidConfig(_))),
            "expected InvalidConfig naming the empty codebook"
        );
    }

    #[test]
    fn clustering_with_zero_iterations_is_invalid_config() {
        // Must surface as a *config* error, not as a bare KMeansNoConvergence
        // that reads like an algorithm failure.
        let scene_data = make_scene(10);
        let mut config = full_precision_config();
        config.use_position_clustering = true;
        config.kmeans.n_clusters = 2;
        config.kmeans.n_iterations = 0;
        let result = compress_slices(&scene_data, &config);
        assert!(
            matches!(result, Err(CompressorError::InvalidConfig(_))),
            "expected InvalidConfig naming n_iterations, not KMeansNoConvergence"
        );
    }

    #[test]
    fn decompress_rejects_out_of_range_cluster_assignment() {
        let n = 20usize;
        let scene_data = make_scene(n);
        let mut config = full_precision_config();
        config.use_position_clustering = true;
        config.kmeans.n_clusters = 2;
        let mut scene = compress_slices(&scene_data, &config).expect("compress");
        if let Some(clustering) = scene.position_clustering.as_mut() {
            clustering.assignments[0] = 999;
        }
        let result = gc_decompress(&scene);
        assert!(
            matches!(
                result,
                Err(CompressorError::DimensionMismatch {
                    expected: 2,
                    got: 999
                })
            ),
            "a corrupt codebook index must be an error, not an out-of-bounds panic"
        );
    }

    // -----------------------------------------------------------------------
    // gc_kmeans_positions: correctness after parallelising the assignment step
    // -----------------------------------------------------------------------

    #[test]
    fn kmeans_positions_parallel_assignment_matches_expected_shape() {
        let n = 60usize;
        // Two well-separated clusters (all points exactly coincide within
        // each cluster), so k-means++ init is *guaranteed* to seed one
        // centre in each cluster regardless of RNG state: any point sharing
        // a location with an already-chosen centre has distance exactly 0,
        // so it carries zero weight in the roulette-wheel seed selection.
        let positions: Vec<f32> = (0..n)
            .flat_map(|i| {
                if i % 2 == 0 {
                    [0.0f32, 0.0, 0.0]
                } else {
                    [100.0f32, 100.0, 100.0]
                }
            })
            .collect();
        let config = KMeansConfig {
            n_clusters: 2,
            n_iterations: 20,
            tolerance: 1e-4,
        };
        let mut rng = 42u64;
        let (centers, assignments) =
            gc_kmeans_positions(&positions, &config, &mut rng).expect("kmeans");
        assert_eq!(centers.len(), 2 * 3);
        assert_eq!(assignments.len(), n);
        let cluster_of_0 = assignments[0];
        let cluster_of_1 = assignments[1];
        assert_ne!(
            cluster_of_0, cluster_of_1,
            "well-separated clusters must not merge"
        );
        for (i, &a) in assignments.iter().enumerate() {
            let expected = if i % 2 == 0 {
                cluster_of_0
            } else {
                cluster_of_1
            };
            assert_eq!(a, expected, "point {i} assigned to the wrong cluster");
        }
    }

    // -----------------------------------------------------------------------
    // gc_compute_stats: honest realignment behaviour
    // -----------------------------------------------------------------------

    #[test]
    fn compute_stats_exact_rmse_when_nothing_pruned() {
        let n = 30usize;
        let (pos, rot, scl, op, shd, shr) = make_scene(n);
        let config = full_precision_config(); // opacity_threshold 0.0: everything kept
        let scene = gc_compress(
            GcSceneSlices {
                positions: &pos,
                rotations: &rot,
                scales: &scl,
                opacities: &op,
                sh_dc: &shd,
                sh_rest: &shr,
                n_rest_per_gaussian: 0,
            },
            &config,
        )
        .expect("compress");
        assert_eq!(scene.n_gaussians, n, "nothing should be pruned here");
        let stats = gc_compute_stats(&pos, &op, &scene).expect("stats");
        assert!(
            stats.position_quantization_rmse < 1e-4,
            "Full precision + no pruning should round-trip near-exactly, got {}",
            stats.position_quantization_rmse
        );
        assert!(!stats.position_quantization_rmse.is_nan());
        assert!(!stats.opacity_quantization_rmse.is_nan());
    }

    /// A scene whose survivors are a *non-contiguous* subset: opacity is high
    /// on odd indices, low on even ones, and each Gaussian sits at a distinct
    /// position. `preserve_top_fraction = 0.5` therefore keeps exactly the
    /// odd indices — the case where index-order emission and opacity-order
    /// selection disagree, so any realignment mistake shows up as a huge RMSE
    /// instead of passing by luck.
    fn interleaved_prune_scene(n: usize) -> (Vec<f32>, Vec<f32>, CompressionConfig) {
        let positions: Vec<f32> = (0..n)
            .flat_map(|i| {
                let f = i as f32;
                [f, f * 2.0, f * 3.0]
            })
            .collect();
        let opacities: Vec<f32> = (0..n)
            .map(|i| if i.is_multiple_of(2) { 1.0f32 } else { 5.0 })
            .collect();
        let mut config = full_precision_config();
        config.pruning.preserve_top_fraction = 0.5;
        (positions, opacities, config)
    }

    #[test]
    fn compute_stats_exact_rmse_under_pruning_realigns_survivors() {
        // Regression: this used to zip dequantized survivor j against
        // original index j regardless of which Gaussians pruning kept, then
        // (after that was spotted) gave up and reported NaN. Neither is
        // needed: CompressedScene::kept_indices records the mapping, so the
        // RMSE is computed exactly.
        let n = 40usize;
        let (positions, opacities, config) = interleaved_prune_scene(n);
        let rotations: Vec<f32> = (0..n).flat_map(|_| [0.0f32, 0.0, 0.0, 1.0]).collect();
        let scales = vec![-1.0f32; n * 3];
        let sh_dc = vec![0.0f32; n * 3];
        let scene = gc_compress(
            GcSceneSlices {
                positions: &positions,
                rotations: &rotations,
                scales: &scales,
                opacities: &opacities,
                sh_dc: &sh_dc,
                sh_rest: &[],
                n_rest_per_gaussian: 0,
            },
            &config,
        )
        .expect("compress");
        assert_eq!(scene.n_gaussians, n / 2, "top half by opacity is kept");
        let expected_kept: Vec<u32> = (0..n as u32).filter(|i| !i.is_multiple_of(2)).collect();
        assert_eq!(
            scene.kept_indices, expected_kept,
            "kept_indices must name the surviving ORIGINAL indices, ascending"
        );

        let stats = gc_compute_stats(&positions, &opacities, &scene).expect("stats");
        assert!(
            !stats.position_quantization_rmse.is_nan(),
            "pruning no longer defeats RMSE computation"
        );
        assert!(
            stats.position_quantization_rmse < 1e-4,
            "Full precision round-trips exactly once survivors are realigned, got {}",
            stats.position_quantization_rmse
        );
        assert!(stats.opacity_quantization_rmse < 1e-4);

        // Sanity check that the assertion above is load-bearing: the naive
        // survivor-j-vs-original-j comparison it replaces is wildly wrong.
        let naive = rmse(
            &scene.reconstruct_positions().expect("positions"),
            &positions,
        );
        assert!(
            naive > 1.0,
            "test scene must make misalignment visible, naive RMSE was {naive}"
        );
    }

    #[test]
    fn compute_stats_rmse_is_nan_when_the_survivor_mapping_is_unusable() {
        // Defensive path for a CompressedScene not built by gc_compress (its
        // fields are public): with pruning evident from the counts but no
        // usable kept_indices, report NaN rather than a misleading,
        // index-misaligned number.
        let n = 40usize;
        let (positions, opacities, config) = interleaved_prune_scene(n);
        let rotations: Vec<f32> = (0..n).flat_map(|_| [0.0f32, 0.0, 0.0, 1.0]).collect();
        let scales = vec![-1.0f32; n * 3];
        let sh_dc = vec![0.0f32; n * 3];
        let mut scene = gc_compress(
            GcSceneSlices {
                positions: &positions,
                rotations: &rotations,
                scales: &scales,
                opacities: &opacities,
                sh_dc: &sh_dc,
                sh_rest: &[],
                n_rest_per_gaussian: 0,
            },
            &config,
        )
        .expect("compress");
        assert!(
            scene.n_gaussians < n,
            "pruning should have removed something"
        );
        scene.kept_indices.clear();
        let stats = gc_compute_stats(&positions, &opacities, &scene).expect("stats");
        assert!(
            stats.position_quantization_rmse.is_nan(),
            "RMSE must be NaN (honest 'cannot realign') rather than a misleading number"
        );
        assert!(stats.opacity_quantization_rmse.is_nan());
    }

    #[test]
    fn compute_stats_rmse_is_nan_when_kept_indices_exceed_the_originals() {
        // Same guard, different failure: the mapping is present and the right
        // length, but the caller passed originals too short to contain the
        // named indices, so gathering would silently truncate.
        let n = 40usize;
        let (positions, opacities, config) = interleaved_prune_scene(n);
        let rotations: Vec<f32> = (0..n).flat_map(|_| [0.0f32, 0.0, 0.0, 1.0]).collect();
        let scales = vec![-1.0f32; n * 3];
        let sh_dc = vec![0.0f32; n * 3];
        let scene = gc_compress(
            GcSceneSlices {
                positions: &positions,
                rotations: &rotations,
                scales: &scales,
                opacities: &opacities,
                sh_dc: &sh_dc,
                sh_rest: &[],
                n_rest_per_gaussian: 0,
            },
            &config,
        )
        .expect("compress");
        let truncated_positions = &positions[..(n / 4) * 3];
        let truncated_opacities = &opacities[..n / 4];
        let stats =
            gc_compute_stats(truncated_positions, truncated_opacities, &scene).expect("stats");
        assert!(stats.position_quantization_rmse.is_nan());
        assert!(stats.opacity_quantization_rmse.is_nan());
    }

    #[test]
    fn compute_stats_realigns_clustered_positions_not_residuals() {
        // With clustering the stored positions are residuals; the RMSE must
        // be measured on reconstructed world positions.
        let n = 40usize;
        let (positions, opacities, mut config) = interleaved_prune_scene(n);
        config.use_position_clustering = true;
        config.kmeans.n_clusters = 4;
        let rotations: Vec<f32> = (0..n).flat_map(|_| [0.0f32, 0.0, 0.0, 1.0]).collect();
        let scales = vec![-1.0f32; n * 3];
        let sh_dc = vec![0.0f32; n * 3];
        let scene = gc_compress(
            GcSceneSlices {
                positions: &positions,
                rotations: &rotations,
                scales: &scales,
                opacities: &opacities,
                sh_dc: &sh_dc,
                sh_rest: &[],
                n_rest_per_gaussian: 0,
            },
            &config,
        )
        .expect("compress");
        assert!(scene.position_clustering.is_some());
        let stats = gc_compute_stats(&positions, &opacities, &scene).expect("stats");
        assert!(
            stats.position_quantization_rmse < 1e-3,
            "clustered + Full precision must still round-trip, got {}",
            stats.position_quantization_rmse
        );
    }

    // -----------------------------------------------------------------------
    // Sigmoid-free sort comparators still select the correct top-opacity set
    // -----------------------------------------------------------------------

    #[test]
    fn prune_to_topn_selects_highest_opacity_indices() {
        let opacities = vec![0.1f32, 5.0, -3.0, 2.0, 0.0];
        let top2 = gc_prune_to_topn(&opacities, 2).expect("topn");
        assert_eq!(
            top2,
            vec![1, 3],
            "expected indices 1 (5.0) and 3 (2.0) in that order"
        );
    }

    #[test]
    fn compute_prune_mask_preserve_top_fraction_keeps_highest_opacity() {
        let opacities = vec![5.0f32, 1.0, 4.0, 0.5];
        let log_scales = vec![-1.0f32; 4 * 3];
        let config = ScenePruningConfig {
            opacity_threshold: 0.0,
            max_log_scale: 100.0,
            min_log_scale: -100.0,
            target_n_gaussians: None,
            preserve_top_fraction: 0.5, // keep top 2 of 4
        };
        let mask = gc_compute_prune_mask(&opacities, &log_scales, &config).expect("mask");
        // Highest two opacities are index 0 (5.0) and index 2 (4.0).
        assert!(mask[0] && mask[2]);
        assert!(!mask[1] && !mask[3]);
    }
}
