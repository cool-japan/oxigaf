//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use rayon::prelude::*;

use super::types::{
    CompressedScene, CompressionConfig, CompressionStats, CompressorError, DecompressedScene,
    GcSceneSlices, KMeansConfig, QuantizedAttribute, ScenePruningConfig,
};

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
/// Performs pruning, optional position clustering, and per-attribute quantization.
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
    let kept_sh_rest_k = if n_rest_per_gaussian == 0 {
        1
    } else {
        n_rest_per_gaussian
    };
    let kept_sh_rest = if n_rest_per_gaussian > 0 {
        gc_apply_mask_flat(sh_rest, &prune_mask, n_rest_per_gaussian)?
    } else {
        Vec::new()
    };
    let n_kept = kept_opacities.len();
    if n_kept == 0 {
        return Err(CompressorError::EmptyScene);
    }
    // `use_position_clustering` is meant to k-means-cluster positions and
    // store per-point *residuals* (offsets from the assigned cluster centre)
    // instead of absolute positions, which quantize tighter. Doing that for
    // real requires persisting the cluster centers and per-point assignments
    // somewhere a decompressor can read them back — but `CompressedScene`
    // (gaussian_compressor::types) has no field for either, so a previous
    // version of this function ran the clustering, quantized the *residuals*
    // (values near zero) as if they were `positions`, and `gc_decompress`
    // returned those residuals straight back as world-space positions: the
    // entire scene silently collapsed toward the origin, with no error and
    // no warning. Until `CompressedScene` gains storage for centers/
    // assignments, fall back to quantizing absolute positions directly (the
    // same, correct path taken when clustering is off) and say so loudly
    // rather than corrupt the scene.
    if config.use_position_clustering && n_kept >= config.kmeans.n_clusters {
        tracing::warn!(
            "use_position_clustering=true has no effect yet: CompressedScene cannot persist \
             k-means cluster centers/assignments, so residual-encoded positions would not be \
             reconstructible on decompression. Falling back to direct position quantization \
             instead of corrupting the scene."
        );
    }
    let final_positions = kept_positions;
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
    let _ = kept_sh_rest_k;
    Ok(CompressedScene {
        positions: q_positions,
        rotations: q_rotations,
        scales: q_scales,
        opacities: q_opacities,
        sh_dc: q_sh_dc,
        sh_rest: q_sh_rest,
        n_gaussians: n_kept,
        n_sh_rest: n_rest_per_gaussian,
        compression_config: config.clone(),
    })
}
/// Decompress a compressed scene back to f32 arrays.
pub fn gc_decompress(scene: &CompressedScene) -> Result<DecompressedScene, CompressorError> {
    let positions = scene.positions.dequantize();
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
/// Compute compression statistics comparing original arrays to a compressed scene.
///
/// `position_quantization_rmse`/`opacity_quantization_rmse` are only
/// meaningful when compared index-for-index against the *same* Gaussian.
/// `gc_compress` prunes Gaussians (by opacity threshold, scale bounds, and/or
/// top-N/fraction truncation) *before* quantizing, so `scene`'s dequantized
/// attribute `i` is the i-th SURVIVOR of that pruning, not original index
/// `i`. This function is handed only the original (unpruned) arrays and the
/// compressed scene — not the prune mask `gc_compress` used — so when
/// anything was pruned it has no way to know *which* indices survived.
/// Rather than silently compare mismatched indices (which used to report a
/// misleading, sometimes orders-of-magnitude-too-large "RMSE" that actually
/// just measured index misalignment), the two RMSE fields are `NaN` whenever
/// `n_gaussians_after != n_gaussians_before`, with a `tracing::warn!`
/// explaining why. When nothing was pruned, survivor `i` provably *is*
/// original index `i`, so the direct comparison is exact.
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

    let (pos_rmse, op_rmse) = if n_after == n_before {
        let deq_positions = scene.positions.dequantize();
        let deq_opacities = scene.opacities.dequantize();
        (
            rmse(&deq_positions, original_positions),
            rmse(&deq_opacities, original_opacities),
        )
    } else {
        tracing::warn!(
            "gc_compute_stats: {} of {} Gaussian(s) were pruned before quantization, and the \
             prune mask gc_compress used is not available here to realign survivors with their \
             original indices. position_quantization_rmse/opacity_quantization_rmse cannot be \
             computed correctly and are reported as NaN rather than a misleading, index-\
             misaligned comparison.",
            n_before.saturating_sub(n_after),
            n_before,
        );
        (f32::NAN, f32::NAN)
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

    fn make_scene(n: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
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

    #[test]
    fn compress_with_position_clustering_does_not_corrupt_positions() {
        // Regression for the critical bug: positions used to be replaced by
        // near-zero k-means residuals with nowhere to store the centers, so
        // decompression silently returned a scene collapsed toward the
        // origin. With Full (lossless) precision, decompressed positions
        // must now match the originals exactly.
        let n = 40usize;
        let (pos, rot, scl, op, shd, shr) = make_scene(n);
        let mut config = full_precision_config();
        config.use_position_clustering = true;
        config.kmeans.n_clusters = 4;

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
        .expect("compress with clustering should still succeed");
        let decomp = gc_decompress(&scene).expect("decompress");
        assert_eq!(decomp.n_gaussians, n);
        assert_eq!(decomp.positions.len(), pos.len());
        for (a, b) in decomp.positions.iter().zip(pos.iter()) {
            assert!(
                (a - b).abs() < 1e-3,
                "position corrupted by clustering fallback: {a} vs {b}"
            );
        }
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

    #[test]
    fn compute_stats_rmse_is_nan_when_pruning_makes_realignment_impossible() {
        // Regression: this used to zip dequantized survivor i against
        // original index i regardless of whether pruning reordered anything,
        // reporting a misleading "RMSE" that actually measured index
        // misalignment. Now it must honestly report NaN instead.
        let n = 50usize;
        let mut opacities: Vec<f32> = vec![5.0f32; n];
        for op in opacities.iter_mut().take(n / 2) {
            *op = -10.0; // sigmoid(-10) << default opacity_threshold (0.01)
        }
        let positions: Vec<f32> = (0..n * 3).map(|i| i as f32 * 0.01).collect();
        let rotations: Vec<f32> = (0..n * 4)
            .map(|i| if i % 4 == 3 { 1.0f32 } else { 0.0 })
            .collect();
        let scales = vec![-1.0f32; n * 3];
        let sh_dc = vec![0.0f32; n * 3];
        let config = CompressionConfig::default();
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
        assert!(
            scene.n_gaussians < n,
            "pruning should have removed something"
        );
        let stats = gc_compute_stats(&positions, &opacities, &scene).expect("stats");
        assert!(
            stats.position_quantization_rmse.is_nan(),
            "RMSE must be NaN (honest 'cannot realign') rather than a misleading number"
        );
        assert!(stats.opacity_quantization_rmse.is_nan());
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
