//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

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
            sorted.sort_by(|&a, &b| {
                let oa = sigmoid(opacities[a]);
                let ob = sigmoid(opacities[b]);
                ob.partial_cmp(&oa).unwrap_or(std::cmp::Ordering::Equal)
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
                let oa = sigmoid(opacities[a]);
                let ob = sigmoid(opacities[b]);
                ob.partial_cmp(&oa).unwrap_or(std::cmp::Ordering::Equal)
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
    indices.sort_by(|&a, &b| {
        let oa = sigmoid(opacities[a]);
        let ob = sigmoid(opacities[b]);
        ob.partial_cmp(&oa).unwrap_or(std::cmp::Ordering::Equal)
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
        for (i, assignment) in assignments.iter_mut().enumerate().take(n) {
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
        }
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
    if !converged && config.n_iterations == 0 {
        return Err(CompressorError::KMeansNoConvergence);
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
    let k = centers.len() / 3;
    let mut residuals = vec![0.0f32; n * 3];
    for i in 0..n {
        let ci = assignments[i];
        if ci >= k {
            return Err(CompressorError::DimensionMismatch {
                expected: k - 1,
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
    let final_positions = if config.use_position_clustering && n_kept >= config.kmeans.n_clusters {
        let mut rng = 0xDEAD_BEEF_u64;
        let (centers, assignments) =
            gc_kmeans_positions(&kept_positions, &config.kmeans, &mut rng)?;
        gc_cluster_residuals(&kept_positions, &assignments, &centers)?
    } else {
        kept_positions.clone()
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
/// Compute compression statistics comparing original arrays to a compressed scene.
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
    let deq_positions = scene.positions.dequantize();
    let pos_len = deq_positions.len().min(original_positions.len());
    let pos_rmse = if pos_len > 0 {
        let mse: f32 = deq_positions[..pos_len]
            .iter()
            .zip(original_positions[..pos_len].iter())
            .map(|(&a, &b)| (a - b) * (a - b))
            .sum::<f32>()
            / pos_len as f32;
        mse.sqrt()
    } else {
        0.0
    };
    let deq_opacities = scene.opacities.dequantize();
    let op_len = deq_opacities.len().min(original_opacities.len());
    let op_rmse = if op_len > 0 {
        let mse: f32 = deq_opacities[..op_len]
            .iter()
            .zip(original_opacities[..op_len].iter())
            .map(|(&a, &b)| (a - b) * (a - b))
            .sum::<f32>()
            / op_len as f32;
        mse.sqrt()
    } else {
        0.0
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
