//! Attention map visualization for multi-view diffusion models.
//!
//! Attention maps show which spatial regions each query position attends to,
//! useful for debugging and understanding the multi-view diffusion model's
//! cross-view attention patterns.

use crate::DiffusionError;

// ---------------------------------------------------------------------------
// Viridis colour table (same 5-point table as in denoising_viz.rs)
// ---------------------------------------------------------------------------

/// Viridis 5-point colour table (value → [R, G, B]).
const VIRIDIS_TABLE: [(f32, [u8; 3]); 5] = [
    (0.00, [68, 1, 84]),
    (0.25, [59, 82, 139]),
    (0.50, [33, 145, 140]),
    (0.75, [94, 201, 98]),
    (1.00, [253, 231, 37]),
];

// ---------------------------------------------------------------------------
// AttentionColormap
// ---------------------------------------------------------------------------

/// Colormap variants for rendering attention weight heatmaps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionColormap {
    /// Jet: blue→cyan→green→yellow→red (6-stop piecewise linear).
    Jet,
    /// Viridis: dark purple→blue→teal→green→yellow (5-stop table).
    Viridis,
    /// Grayscale: black→white.
    Grayscale,
    /// Hot: black→red→yellow→white (3-segment).
    Hot,
}

// ---------------------------------------------------------------------------
// AttentionMap
// ---------------------------------------------------------------------------

/// A 2-D attention map from one query position to all key positions.
///
/// `weights` is a flat row-major array of shape
/// `[query_height * query_width, key_height * key_width]`,
/// i.e. `weights[qy * query_width + qx][ky * key_width + kx]`.
#[derive(Debug, Clone)]
pub struct AttentionMap {
    /// Raw attention weights (post-softmax).
    pub weights: Vec<f32>,
    pub query_height: usize,
    pub query_width: usize,
    pub key_height: usize,
    pub key_width: usize,
    pub num_heads: usize,
    /// Which head this map is from (`None` = averaged across heads).
    pub head_index: Option<usize>,
}

impl AttentionMap {
    /// Construct a validated `AttentionMap`.
    ///
    /// # Errors
    /// Returns [`DiffusionError::InvalidConfig`] when
    /// `weights.len() != query_height * query_width * key_height * key_width`.
    pub fn new(
        weights: Vec<f32>,
        query_height: usize,
        query_width: usize,
        key_height: usize,
        key_width: usize,
        num_heads: usize,
        head_index: Option<usize>,
    ) -> Result<Self, DiffusionError> {
        let expected = query_height * query_width * key_height * key_width;
        if weights.len() != expected {
            return Err(DiffusionError::InvalidConfig(format!(
                "AttentionMap weight length mismatch: expected {} ({}×{}×{}×{}), got {}",
                expected,
                query_height,
                query_width,
                key_height,
                key_width,
                weights.len()
            )));
        }
        Ok(Self {
            weights,
            query_height,
            query_width,
            key_height,
            key_width,
            num_heads,
            head_index,
        })
    }

    /// Total number of query positions (`query_height * query_width`).
    #[inline]
    pub fn query_len(&self) -> usize {
        self.query_height * self.query_width
    }

    /// Total number of key positions (`key_height * key_width`).
    #[inline]
    pub fn key_len(&self) -> usize {
        self.key_height * self.key_width
    }

    /// Return the `key_len` weights for query pixel `(qy, qx)`.
    ///
    /// Returns an empty slice when coordinates are out of bounds or the
    /// internal buffer is too short.
    pub fn for_query_pixel(&self, qy: usize, qx: usize) -> &[f32] {
        if qy >= self.query_height || qx >= self.query_width {
            return &[];
        }
        let row = qy * self.query_width + qx;
        let key_len = self.key_len();
        let start = row * key_len;
        let end = start + key_len;
        if end > self.weights.len() {
            return &[];
        }
        &self.weights[start..end]
    }

    /// Return a `Vec` of `query_len` weights that attend to key pixel `(ky, kx)`.
    ///
    /// Returns an empty `Vec` when coordinates are out of bounds.
    pub fn for_key_pixel(&self, ky: usize, kx: usize) -> Vec<f32> {
        if ky >= self.key_height || kx >= self.key_width {
            return Vec::new();
        }
        let key_idx = ky * self.key_width + kx;
        let key_len = self.key_len();
        let query_len = self.query_len();
        let mut result = Vec::with_capacity(query_len);
        for q in 0..query_len {
            let idx = q * key_len + key_idx;
            if idx < self.weights.len() {
                result.push(self.weights[idx]);
            } else {
                result.push(0.0);
            }
        }
        result
    }

    /// Return the top-`k` `(key_y, key_x, weight)` tuples for query pixel `(qy, qx)`.
    ///
    /// Results are sorted descending by weight.  If `k` exceeds the number of
    /// key positions, all positions are returned.
    pub fn top_k_keys(&self, qy: usize, qx: usize, k: usize) -> Vec<(usize, usize, f32)> {
        let row = self.for_query_pixel(qy, qx);
        if row.is_empty() {
            return Vec::new();
        }

        let key_width = self.key_width;
        let mut indexed: Vec<(usize, usize, f32)> = row
            .iter()
            .enumerate()
            .map(|(idx, &w)| {
                let ky = idx / key_width;
                let kx = idx % key_width;
                (ky, kx, w)
            })
            .collect();

        // Sort descending by weight; treat NaN as equal to avoid panicking.
        indexed.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        indexed.truncate(k);
        indexed
    }

    /// Mean attention weight for each key position, averaged over all queries.
    ///
    /// Returns a `Vec` of length `key_h * key_w`.
    pub fn aggregate_over_queries(&self) -> Vec<f32> {
        let key_len = self.key_len();
        let query_len = self.query_len();
        if key_len == 0 || query_len == 0 {
            return vec![0.0; key_len];
        }

        let mut sums = vec![0.0_f32; key_len];
        for q in 0..query_len {
            let start = q * key_len;
            let end = (start + key_len).min(self.weights.len());
            for (k, &w) in self.weights[start..end].iter().enumerate() {
                sums[k] += w;
            }
        }
        let n = query_len as f32;
        sums.iter_mut().for_each(|s| *s /= n);
        sums
    }
}

// ---------------------------------------------------------------------------
// CrossViewAttentionMap
// ---------------------------------------------------------------------------

/// Cross-view attention map: query from `source_view`, keys from `target_view`.
#[derive(Debug, Clone)]
pub struct CrossViewAttentionMap {
    pub source_view: usize,
    pub target_view: usize,
    /// One map per head.
    pub maps_per_head: Vec<AttentionMap>,
    /// Element-wise average across all heads.
    pub averaged_map: AttentionMap,
}

impl CrossViewAttentionMap {
    /// Build a `CrossViewAttentionMap` from per-head raw weight vectors.
    ///
    /// Each entry of `weights_per_head` must have length
    /// `q_h * q_w * k_h * k_w`.
    ///
    /// # Errors
    /// Propagates [`DiffusionError::InvalidConfig`] from [`AttentionMap::new`]
    /// when any head's weight vector has the wrong length.
    pub fn from_raw(
        source_view: usize,
        target_view: usize,
        weights_per_head: Vec<Vec<f32>>,
        q_h: usize,
        q_w: usize,
        k_h: usize,
        k_w: usize,
    ) -> Result<Self, DiffusionError> {
        if weights_per_head.is_empty() {
            return Err(DiffusionError::InvalidConfig(
                "weights_per_head must not be empty".to_string(),
            ));
        }

        let num_heads = weights_per_head.len();
        let map_len = q_h * q_w * k_h * k_w;

        let mut maps_per_head = Vec::with_capacity(num_heads);
        for (i, head_weights) in weights_per_head.iter().enumerate() {
            let map =
                AttentionMap::new(head_weights.clone(), q_h, q_w, k_h, k_w, num_heads, Some(i))?;
            maps_per_head.push(map);
        }

        // Element-wise average across heads.
        let mut avg_weights = vec![0.0_f32; map_len];
        for head_weights in &weights_per_head {
            for (a, &w) in avg_weights.iter_mut().zip(head_weights.iter()) {
                *a += w;
            }
        }
        let n = num_heads as f32;
        avg_weights.iter_mut().for_each(|v| *v /= n);

        let averaged_map = AttentionMap::new(avg_weights, q_h, q_w, k_h, k_w, num_heads, None)?;

        Ok(Self {
            source_view,
            target_view,
            maps_per_head,
            averaged_map,
        })
    }

    /// Return the head-averaged attention map.
    pub fn head_averaged_weights(&self) -> &AttentionMap {
        &self.averaged_map
    }
}

// ---------------------------------------------------------------------------
// Colormap functions
// ---------------------------------------------------------------------------

/// Map a scalar in `[0, 1]` to an RGB triple using the given colormap.
pub fn apply_colormap(value: f32, colormap: AttentionColormap) -> [u8; 3] {
    let v = value.clamp(0.0, 1.0);
    match colormap {
        AttentionColormap::Jet => jet_lookup(v),
        AttentionColormap::Viridis => viridis_lookup(v),
        AttentionColormap::Grayscale => {
            let byte = (v * 255.0).round().clamp(0.0, 255.0) as u8;
            [byte, byte, byte]
        }
        AttentionColormap::Hot => hot_lookup(v),
    }
}

/// Jet colormap: 6-stop piecewise linear.
///
/// Stops: 0.0→[0,0,128], 0.125→[0,0,255], 0.375→[0,255,255],
///        0.625→[255,255,0], 0.875→[255,0,0], 1.0→[128,0,0]
fn jet_lookup(t: f32) -> [u8; 3] {
    const JET_TABLE: [(f32, [u8; 3]); 6] = [
        (0.000, [0, 0, 128]),
        (0.125, [0, 0, 255]),
        (0.375, [0, 255, 255]),
        (0.625, [255, 255, 0]),
        (0.875, [255, 0, 0]),
        (1.000, [128, 0, 0]),
    ];

    piecewise_linear_lookup(t, &JET_TABLE)
}

/// Hot colormap: black→red→yellow→white (3-segment).
///
/// Stops: 0→[0,0,0], 1/3→[255,0,0], 2/3→[255,255,0], 1→[255,255,255]
fn hot_lookup(t: f32) -> [u8; 3] {
    const HOT_TABLE: [(f32, [u8; 3]); 4] = [
        (0.000_000, [0, 0, 0]),
        (0.333_333, [255, 0, 0]),
        (0.666_667, [255, 255, 0]),
        (1.000_000, [255, 255, 255]),
    ];

    piecewise_linear_lookup(t, &HOT_TABLE)
}

/// Viridis colormap from the 5-point constant table.
fn viridis_lookup(t: f32) -> [u8; 3] {
    piecewise_linear_lookup(t, &VIRIDIS_TABLE)
}

/// Generic piecewise linear colour lookup.
fn piecewise_linear_lookup(t: f32, table: &[(f32, [u8; 3])]) -> [u8; 3] {
    let t = t.clamp(0.0, 1.0);
    if table.is_empty() {
        return [0, 0, 0];
    }
    if table.len() == 1 {
        return table[0].1;
    }

    let mut lo_idx = 0usize;
    for (i, &(pos, _)) in table.iter().enumerate() {
        if pos <= t {
            lo_idx = i;
        }
    }

    let hi_idx = (lo_idx + 1).min(table.len() - 1);

    let (lo_t, lo_col) = table[lo_idx];
    let (hi_t, hi_col) = table[hi_idx];

    let span = hi_t - lo_t;
    let frac = if span == 0.0 { 0.0 } else { (t - lo_t) / span };

    let lerp = |a: u8, b: u8, f: f32| -> u8 {
        let av = a as f32;
        let bv = b as f32;
        (av + (bv - av) * f).round().clamp(0.0, 255.0) as u8
    };

    [
        lerp(lo_col[0], hi_col[0], frac),
        lerp(lo_col[1], hi_col[1], frac),
        lerp(lo_col[2], hi_col[2], frac),
    ]
}

// ---------------------------------------------------------------------------
// Normalization and heatmap helpers
// ---------------------------------------------------------------------------

/// Scale weights to `[0, 1]` by dividing by the maximum value.
///
/// When the max is zero (or the slice is empty) all values become zero.
pub fn normalize_attention(weights: &[f32]) -> Vec<f32> {
    if weights.is_empty() {
        return Vec::new();
    }
    let max = weights.iter().copied().fold(0.0_f32, f32::max);
    if max == 0.0 {
        return vec![0.0; weights.len()];
    }
    weights.iter().map(|&w| w / max).collect()
}

/// Convert a flat weight array to an RGBA byte image using `colormap`.
///
/// Weights are normalized to `[0, 1]` before colour-mapping.  Alpha is always
/// 255.  When `weights.len() != height * width` a zero-filled (black) buffer
/// is returned.
///
/// This always normalizes; see [`heatmap_to_rgba_normalized`] to make that
/// optional (e.g. to honour [`AttentionVizConfig::normalize_per_map`]).
pub fn heatmap_to_rgba(
    weights: &[f32],
    height: usize,
    width: usize,
    colormap: AttentionColormap,
) -> Vec<u8> {
    heatmap_to_rgba_normalized(weights, height, width, colormap, true)
}

/// Convert a flat weight array to an RGBA byte image using `colormap`.
///
/// Identical to [`heatmap_to_rgba`], except normalization is optional: when
/// `normalize` is `false`, raw weights are passed straight to
/// [`apply_colormap`], which clamps to `[0, 1]` internally. Alpha is always
/// 255.  When `weights.len() != height * width` a zero-filled (black) buffer
/// is returned.
pub fn heatmap_to_rgba_normalized(
    weights: &[f32],
    height: usize,
    width: usize,
    colormap: AttentionColormap,
    normalize: bool,
) -> Vec<u8> {
    let num_pixels = height * width;
    let mut output = vec![0u8; num_pixels * 4];

    if weights.len() != num_pixels {
        // Fill alpha channel; RGB stays black.
        for p in 0..num_pixels {
            output[p * 4 + 3] = 255;
        }
        return output;
    }

    let normalized;
    let source: &[f32] = if normalize {
        normalized = normalize_attention(weights);
        &normalized
    } else {
        weights
    };
    for (p, &v) in source.iter().enumerate() {
        let [r, g, b] = apply_colormap(v, colormap);
        output[p * 4] = r;
        output[p * 4 + 1] = g;
        output[p * 4 + 2] = b;
        output[p * 4 + 3] = 255;
    }
    output
}

// ---------------------------------------------------------------------------
// AttentionVizConfig
// ---------------------------------------------------------------------------

/// Configuration for attention map visualization.
#[derive(Debug, Clone)]
pub struct AttentionVizConfig {
    /// Colormap for the heatmap.
    pub colormap: AttentionColormap,
    /// Alpha blend weight when overlaying attention on a base image (default: 0.7).
    pub overlay_alpha: f32,
    /// Optional target size `(width, height)` to upsample the heatmap to.
    pub upsample_to: Option<(usize, usize)>,
    /// Normalize each map independently (default: `true`).
    pub normalize_per_map: bool,
}

impl Default for AttentionVizConfig {
    fn default() -> Self {
        Self {
            colormap: AttentionColormap::Viridis,
            overlay_alpha: 0.7,
            upsample_to: None,
            normalize_per_map: true,
        }
    }
}

impl AttentionVizConfig {
    /// Create a config with the given colormap and all other fields at their defaults.
    pub fn new(colormap: AttentionColormap) -> Self {
        Self {
            colormap,
            ..Self::default()
        }
    }
}

// ---------------------------------------------------------------------------
// Full visualization pipeline
// ---------------------------------------------------------------------------

/// Convert an `AttentionMap` to an RGBA image using the configured pipeline.
///
/// 1. Aggregate weights over all query positions (mean per key).
/// 2. Reshape to `key_h × key_w` and call [`heatmap_to_rgba`].
/// 3. If `config.upsample_to` is set, nearest-neighbor upsample.
pub fn attention_map_to_image(map: &AttentionMap, config: &AttentionVizConfig) -> Vec<u8> {
    let agg = map.aggregate_over_queries();
    let src_h = map.key_height;
    let src_w = map.key_width;

    let heatmap = heatmap_to_rgba_normalized(
        &agg,
        src_h,
        src_w,
        config.colormap,
        config.normalize_per_map,
    );

    match config.upsample_to {
        None => heatmap,
        Some((dst_w, dst_h)) => {
            nearest_neighbor_upsample_rgba(&heatmap, src_w, src_h, dst_w, dst_h)
        }
    }
}

/// Blend an attention RGBA heatmap over a base RGBA image.
///
/// `output = base * (1 - alpha) + attention * alpha`
///
/// Both slices must have the same length (`h * w * 4`).  If lengths differ,
/// the base image is returned unchanged.
pub fn overlay_attention_on_image(base_image: &[u8], attention_rgba: &[u8], alpha: f32) -> Vec<u8> {
    if base_image.len() != attention_rgba.len() {
        return base_image.to_vec();
    }
    let alpha = alpha.clamp(0.0, 1.0);
    let inv_alpha = 1.0 - alpha;
    base_image
        .iter()
        .zip(attention_rgba.iter())
        .map(|(&b, &a)| {
            (b as f32 * inv_alpha + a as f32 * alpha)
                .round()
                .clamp(0.0, 255.0) as u8
        })
        .collect()
}

/// Blend an attention RGBA heatmap over a base RGBA image using
/// `config.overlay_alpha` for the blend weight.
///
/// Identical to [`overlay_attention_on_image`], except the alpha comes from
/// `config` instead of being passed explicitly.
pub fn overlay_attention_on_image_with_config(
    base_image: &[u8],
    attention_rgba: &[u8],
    config: &AttentionVizConfig,
) -> Vec<u8> {
    overlay_attention_on_image(base_image, attention_rgba, config.overlay_alpha)
}

/// Format a human-readable statistics string for an `AttentionMap`.
///
/// Format: `"Attention map [QH×QW → KH×KW]: min=X.XXX max=X.XXX mean=X.XXX mean_entropy=X.XXX"`
///
/// `mean_entropy` is the per-query Shannon entropy (in bits) averaged over all
/// `query_len()` queries — *not* the raw sum over the whole flat `weights`
/// buffer, which would scale with the number of queries and not be comparable
/// between maps of different sizes.
pub fn format_attention_stats(map: &AttentionMap) -> String {
    let w = &map.weights;
    if w.is_empty() {
        return format!(
            "Attention map [{}×{} → {}×{}]: min=0.000 max=0.000 mean=0.000 mean_entropy=0.000",
            map.query_height, map.query_width, map.key_height, map.key_width
        );
    }

    let mut min_val = f32::INFINITY;
    let mut max_val = f32::NEG_INFINITY;
    let mut sum = 0.0_f32;
    for &v in w {
        if v < min_val {
            min_val = v;
        }
        if v > max_val {
            max_val = v;
        }
        sum += v;
    }
    let mean = sum / w.len() as f32;

    // Shannon entropy summed over every (query, key) entry: since each of the
    // query_len() rows is a post-softmax distribution summing to 1, this sum
    // is query_len() independent per-row entropies added together. Divide by
    // query_len() to report the mean per-query entropy instead.
    const EPS: f32 = 1e-10;
    let entropy_sum = w
        .iter()
        .map(|&v| {
            let v_eps = v + EPS;
            -v * v_eps.log2()
        })
        .sum::<f32>();
    let mean_entropy = entropy_sum / map.query_len().max(1) as f32;

    format!(
        "Attention map [{}×{} → {}×{}]: min={:.3} max={:.3} mean={:.3} mean_entropy={:.3}",
        map.query_height,
        map.query_width,
        map.key_height,
        map.key_width,
        min_val,
        max_val,
        mean,
        mean_entropy
    )
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Nearest-neighbor upsample of an RGBA byte buffer.
fn nearest_neighbor_upsample_rgba(
    src: &[u8],
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
) -> Vec<u8> {
    let mut dst = vec![0u8; dst_w * dst_h * 4];
    for dy in 0..dst_h {
        for dx in 0..dst_w {
            // Map destination pixel back to nearest source pixel.
            let sx = (dx * src_w)
                .checked_div(dst_w)
                .unwrap_or(0)
                .min(src_w.saturating_sub(1));
            let sy = (dy * src_h)
                .checked_div(dst_h)
                .unwrap_or(0)
                .min(src_h.saturating_sub(1));
            let src_idx = (sy * src_w + sx) * 4;
            let dst_idx = (dy * dst_w + dx) * 4;
            if src_idx + 3 < src.len() && dst_idx + 3 < dst.len() {
                dst[dst_idx] = src[src_idx];
                dst[dst_idx + 1] = src[src_idx + 1];
                dst[dst_idx + 2] = src[src_idx + 2];
                dst[dst_idx + 3] = src[src_idx + 3];
            }
        }
    }
    dst
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // AttentionMap construction
    // -----------------------------------------------------------------------

    #[test]
    fn test_attention_map_new_valid() {
        // 2×2 query, 3×3 key → 4 * 9 = 36 weights
        let weights = vec![0.1f32; 36];
        let map = AttentionMap::new(weights, 2, 2, 3, 3, 1, Some(0))
            .expect("valid map should be created");
        assert_eq!(map.query_len(), 4);
        assert_eq!(map.key_len(), 9);
        assert_eq!(map.num_heads, 1);
        assert_eq!(map.head_index, Some(0));
    }

    #[test]
    fn test_attention_map_invalid_length() {
        // Provide wrong number of weights.
        let weights = vec![0.1f32; 10]; // should be 4*9=36
        let result = AttentionMap::new(weights, 2, 2, 3, 3, 1, None);
        assert!(result.is_err(), "expected error for wrong length");
    }

    // -----------------------------------------------------------------------
    // for_query_pixel
    // -----------------------------------------------------------------------

    #[test]
    fn test_attention_map_for_query_pixel() {
        // 2×2 query, 2×2 key → 16 weights
        // Row layout: [q=0 → [k0,k1,k2,k3], q=1 → [...], ...]
        let weights: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let map = AttentionMap::new(weights, 2, 2, 2, 2, 1, None).expect("valid map");
        // Query (0,0) → row 0 → weights[0..4]
        let row = map.for_query_pixel(0, 0);
        assert_eq!(row, &[0.0, 1.0, 2.0, 3.0]);
        // Query (1,1) → row 3 → weights[12..16]
        let row2 = map.for_query_pixel(1, 1);
        assert_eq!(row2, &[12.0, 13.0, 14.0, 15.0]);
    }

    #[test]
    fn test_attention_map_for_query_pixel_oob() {
        let map = AttentionMap::new(vec![0.5; 4], 1, 1, 2, 2, 1, None).expect("valid map");
        // Out-of-bounds should return empty slice.
        assert_eq!(map.for_query_pixel(5, 5), &[] as &[f32]);
    }

    // -----------------------------------------------------------------------
    // for_key_pixel
    // -----------------------------------------------------------------------

    #[test]
    fn test_attention_map_for_key_pixel() {
        // 2×1 query (2 rows, 1 col), 1×2 key (1 row, 2 cols) → 4 weights
        // weights = [q0_k0, q0_k1, q1_k0, q1_k1]
        let weights = vec![0.1_f32, 0.2, 0.3, 0.4];
        let map = AttentionMap::new(weights, 2, 1, 1, 2, 1, None).expect("valid map");
        // Key (0,0): column 0 → [weights[0], weights[2]] = [0.1, 0.3]
        let col = map.for_key_pixel(0, 0);
        assert_eq!(col.len(), 2);
        assert!((col[0] - 0.1).abs() < 1e-6);
        assert!((col[1] - 0.3).abs() < 1e-6);
        // Key (0,1): column 1 → [weights[1], weights[3]] = [0.2, 0.4]
        let col2 = map.for_key_pixel(0, 1);
        assert!((col2[0] - 0.2).abs() < 1e-6);
        assert!((col2[1] - 0.4).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // top_k_keys
    // -----------------------------------------------------------------------

    #[test]
    fn test_attention_map_top_k_keys() {
        // 1×1 query, 2×2 key → 4 weights
        let weights = vec![0.1_f32, 0.4, 0.2, 0.3];
        let map = AttentionMap::new(weights, 1, 1, 2, 2, 1, None).expect("valid map");
        let top2 = map.top_k_keys(0, 0, 2);
        assert_eq!(top2.len(), 2);
        // Best weight is 0.4 at key index 1 → (0, 1)
        assert_eq!(top2[0].0, 0);
        assert_eq!(top2[0].1, 1);
        assert!((top2[0].2 - 0.4).abs() < 1e-6);
        // Second best is 0.3 at key index 3 → (1, 1)
        assert_eq!(top2[1].0, 1);
        assert_eq!(top2[1].1, 1);
        assert!((top2[1].2 - 0.3).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // aggregate_over_queries
    // -----------------------------------------------------------------------

    #[test]
    fn test_attention_map_aggregate_over_queries() {
        // 2×1 query, 1×2 key → 4 weights
        // q0 → [0.2, 0.8], q1 → [0.4, 0.6]
        let weights = vec![0.2_f32, 0.8, 0.4, 0.6];
        let map = AttentionMap::new(weights, 2, 1, 1, 2, 1, None).expect("valid map");
        let agg = map.aggregate_over_queries();
        assert_eq!(agg.len(), 2);
        assert!((agg[0] - 0.3).abs() < 1e-6, "expected 0.3, got {}", agg[0]);
        assert!((agg[1] - 0.7).abs() < 1e-6, "expected 0.7, got {}", agg[1]);
    }

    // -----------------------------------------------------------------------
    // CrossViewAttentionMap
    // -----------------------------------------------------------------------

    #[test]
    fn test_cross_view_from_raw() {
        // 2 heads, 1×1 query, 1×2 key → 2 weights per head
        let h0 = vec![0.3_f32, 0.7];
        let h1 = vec![0.5_f32, 0.5];
        let cv = CrossViewAttentionMap::from_raw(0, 1, vec![h0, h1], 1, 1, 1, 2)
            .expect("valid cross-view map");
        assert_eq!(cv.source_view, 0);
        assert_eq!(cv.target_view, 1);
        assert_eq!(cv.maps_per_head.len(), 2);
        assert_eq!(cv.maps_per_head[0].head_index, Some(0));
        assert_eq!(cv.maps_per_head[1].head_index, Some(1));
    }

    #[test]
    fn test_cross_view_head_averaged() {
        // 2 heads, weights [0.0, 1.0] and [1.0, 0.0] → average [0.5, 0.5]
        let h0 = vec![0.0_f32, 1.0];
        let h1 = vec![1.0_f32, 0.0];
        let cv = CrossViewAttentionMap::from_raw(0, 1, vec![h0, h1], 1, 1, 1, 2)
            .expect("valid cross-view map");
        let avg = cv.head_averaged_weights();
        assert!(avg.head_index.is_none());
        assert!((avg.weights[0] - 0.5).abs() < 1e-6);
        assert!((avg.weights[1] - 0.5).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // Colormaps
    // -----------------------------------------------------------------------

    #[test]
    fn test_colormap_jet_endpoints() {
        // t=0.0 → [0,0,128], t=1.0 → [128,0,0]
        let lo = apply_colormap(0.0, AttentionColormap::Jet);
        assert_eq!(lo, [0, 0, 128]);
        let hi = apply_colormap(1.0, AttentionColormap::Jet);
        assert_eq!(hi, [128, 0, 0]);
        // t=0.5 should be between green and yellow, specifically [0,255,255]→...
        // Just verify it's deterministic by calling twice.
        let mid1 = apply_colormap(0.5, AttentionColormap::Jet);
        let mid2 = apply_colormap(0.5, AttentionColormap::Jet);
        assert_eq!(mid1, mid2);
    }

    #[test]
    fn test_colormap_viridis_endpoints() {
        // t=0.0 → [68,1,84], t=1.0 → [253,231,37]
        let lo = apply_colormap(0.0, AttentionColormap::Viridis);
        assert_eq!(lo, [68, 1, 84]);
        let hi = apply_colormap(1.0, AttentionColormap::Viridis);
        assert_eq!(hi, [253, 231, 37]);
    }

    #[test]
    fn test_colormap_grayscale() {
        let black = apply_colormap(0.0, AttentionColormap::Grayscale);
        assert_eq!(black, [0, 0, 0]);
        let white = apply_colormap(1.0, AttentionColormap::Grayscale);
        assert_eq!(white, [255, 255, 255]);
        let mid = apply_colormap(0.5, AttentionColormap::Grayscale);
        assert_eq!(mid[0], mid[1]);
        assert_eq!(mid[1], mid[2]);
    }

    #[test]
    fn test_colormap_hot() {
        // t=0 → [0,0,0]
        let lo = apply_colormap(0.0, AttentionColormap::Hot);
        assert_eq!(lo, [0, 0, 0]);
        // t=1 → [255,255,255]
        let hi = apply_colormap(1.0, AttentionColormap::Hot);
        assert_eq!(hi, [255, 255, 255]);
        // t≈1/3 → [255,0,0]
        let red = apply_colormap(1.0 / 3.0, AttentionColormap::Hot);
        assert_eq!(red, [255, 0, 0]);
        // t≈2/3 → [255,255,0]
        let yellow = apply_colormap(2.0 / 3.0, AttentionColormap::Hot);
        assert_eq!(yellow, [255, 255, 0]);
    }

    // -----------------------------------------------------------------------
    // normalize_attention
    // -----------------------------------------------------------------------

    #[test]
    fn test_normalize_attention() {
        let weights = vec![0.0_f32, 0.5, 1.0, 2.0];
        let norm = normalize_attention(&weights);
        assert_eq!(norm.len(), 4);
        assert!((norm[0] - 0.0).abs() < 1e-6);
        assert!((norm[1] - 0.25).abs() < 1e-6);
        assert!((norm[2] - 0.5).abs() < 1e-6);
        assert!((norm[3] - 1.0).abs() < 1e-6);

        // All zeros → all zeros.
        let zeros = normalize_attention(&[0.0, 0.0, 0.0]);
        assert!(zeros.iter().all(|&v| v == 0.0));

        // Empty → empty.
        let empty: Vec<f32> = normalize_attention(&[]);
        assert!(empty.is_empty());
    }

    // -----------------------------------------------------------------------
    // heatmap_to_rgba
    // -----------------------------------------------------------------------

    #[test]
    fn test_heatmap_to_rgba_shape() {
        let weights = vec![0.0_f32, 0.5, 0.5, 1.0];
        let rgba = heatmap_to_rgba(&weights, 2, 2, AttentionColormap::Grayscale);
        assert_eq!(rgba.len(), 2 * 2 * 4);
        // All alpha bytes should be 255.
        for p in 0..4 {
            assert_eq!(rgba[p * 4 + 3], 255);
        }
    }

    #[test]
    fn test_heatmap_to_rgba_length_mismatch() {
        // weights.len() ≠ height * width → zero-filled black RGBA
        let weights = vec![1.0_f32; 5]; // wrong length for 2×2
        let rgba = heatmap_to_rgba(&weights, 2, 2, AttentionColormap::Grayscale);
        assert_eq!(rgba.len(), 2 * 2 * 4);
        // RGB should be 0, alpha should be 255.
        for p in 0..4 {
            assert_eq!(rgba[p * 4], 0);
            assert_eq!(rgba[p * 4 + 1], 0);
            assert_eq!(rgba[p * 4 + 2], 0);
            assert_eq!(rgba[p * 4 + 3], 255);
        }
    }

    // -----------------------------------------------------------------------
    // attention_map_to_image
    // -----------------------------------------------------------------------

    #[test]
    fn test_attention_map_to_image() {
        // 2×2 query, 2×2 key, uniform weights.
        let weights = vec![0.25_f32; 16];
        let map = AttentionMap::new(weights, 2, 2, 2, 2, 1, None).expect("valid map");
        let config = AttentionVizConfig::default();
        let img = attention_map_to_image(&map, &config);
        // Without upsample: key_h * key_w * 4 = 16 bytes.
        assert_eq!(img.len(), 2 * 2 * 4);
        // All alpha bytes = 255.
        for p in 0..4 {
            assert_eq!(img[p * 4 + 3], 255);
        }
    }

    #[test]
    fn test_attention_map_to_image_with_upsample() {
        let weights = vec![0.25_f32; 4]; // 1×1 query, 2×2 key (but wait: 1*1*2*2=4)
        let map = AttentionMap::new(weights, 1, 1, 2, 2, 1, None).expect("valid map");
        let config = AttentionVizConfig {
            upsample_to: Some((8, 8)),
            ..Default::default()
        };
        let img = attention_map_to_image(&map, &config);
        assert_eq!(img.len(), 8 * 8 * 4);
    }

    // -----------------------------------------------------------------------
    // overlay_attention_on_image
    // -----------------------------------------------------------------------

    #[test]
    fn test_overlay_attention_on_image_alpha_0() {
        // alpha=0 → output equals base_image
        let base = vec![100u8, 150, 200, 255, 50, 60, 70, 255];
        let attention = vec![0u8, 0, 0, 255, 255, 255, 255, 255];
        let out = overlay_attention_on_image(&base, &attention, 0.0);
        assert_eq!(out, base);
    }

    #[test]
    fn test_overlay_attention_on_image_alpha_1() {
        // alpha=1 → output equals attention
        let base = vec![100u8, 150, 200, 255, 50, 60, 70, 255];
        let attention = vec![10u8, 20, 30, 255, 40, 50, 60, 255];
        let out = overlay_attention_on_image(&base, &attention, 1.0);
        assert_eq!(out, attention);
    }

    #[test]
    fn test_overlay_attention_on_image_blend() {
        // alpha=0.5 → arithmetic mean of base and attention
        let base = vec![100u8, 200, 0, 255];
        let attention = vec![0u8, 0, 200, 255];
        let out = overlay_attention_on_image(&base, &attention, 0.5);
        // (100*0.5 + 0*0.5) = 50, (200*0.5 + 0*0.5) = 100, (0*0.5 + 200*0.5) = 100
        assert_eq!(out[0], 50);
        assert_eq!(out[1], 100);
        assert_eq!(out[2], 100);
        assert_eq!(out[3], 255);
    }

    // -----------------------------------------------------------------------
    // format_attention_stats
    // -----------------------------------------------------------------------

    #[test]
    fn test_format_attention_stats() {
        let weights = vec![0.25_f32; 16]; // 2×2 query, 2×2 key
        let map = AttentionMap::new(weights, 2, 2, 2, 2, 1, None).expect("valid map");
        let stats = format_attention_stats(&map);
        assert!(
            stats.contains("Attention map [2×2 → 2×2]"),
            "got: {}",
            stats
        );
        assert!(stats.contains("min="), "got: {}", stats);
        assert!(stats.contains("max="), "got: {}", stats);
        assert!(stats.contains("mean="), "got: {}", stats);
        assert!(stats.contains("mean_entropy="), "got: {}", stats);
    }

    #[test]
    fn test_format_attention_stats_entropy_is_per_query_mean() {
        // 4 queries (2x2), each a uniform distribution over 4 keys (2x2):
        // per-query entropy = log2(4) = 2 bits. The raw (unfixed) sum over
        // all 16 entries would instead report 4 * 2.0 = 8.0.
        let weights = vec![0.25_f32; 16];
        let map = AttentionMap::new(weights, 2, 2, 2, 2, 1, None).expect("valid map");
        let stats = format_attention_stats(&map);
        assert!(
            stats.contains("mean_entropy=2.000"),
            "expected mean_entropy=2.000 (per-query, not summed over queries), got: {}",
            stats
        );
    }

    #[test]
    fn test_format_attention_stats_empty() {
        // Zero-weight map should not panic.
        let map = AttentionMap::new(vec![0.0_f32; 4], 2, 1, 2, 1, 1, None).expect("valid map");
        let stats = format_attention_stats(&map);
        assert!(stats.contains("min=0.000"), "got: {}", stats);
        assert!(stats.contains("mean_entropy=0.000"), "got: {}", stats);
    }

    // -----------------------------------------------------------------------
    // heatmap_to_rgba_normalized / normalize_per_map
    // -----------------------------------------------------------------------

    #[test]
    fn test_heatmap_to_rgba_normalized_false_skips_normalization() {
        // Raw weights already small (e.g. post-softmax means); with
        // normalize=false they should map through apply_colormap unscaled,
        // differing from the normalize=true result (which rescales to [0,1]).
        let weights = vec![0.1_f32, 0.2, 0.05, 0.4];
        let normalized_on =
            heatmap_to_rgba_normalized(&weights, 2, 2, AttentionColormap::Grayscale, true);
        let normalized_off =
            heatmap_to_rgba_normalized(&weights, 2, 2, AttentionColormap::Grayscale, false);
        assert_ne!(
            normalized_on, normalized_off,
            "normalize_per_map=false should produce a different (unnormalized) heatmap"
        );
    }

    #[test]
    fn test_heatmap_to_rgba_matches_normalized_true() {
        let weights = vec![0.1_f32, 0.2, 0.05, 0.4];
        let via_default = heatmap_to_rgba(&weights, 2, 2, AttentionColormap::Viridis);
        let via_explicit =
            heatmap_to_rgba_normalized(&weights, 2, 2, AttentionColormap::Viridis, true);
        assert_eq!(via_default, via_explicit);
    }

    #[test]
    fn test_attention_map_to_image_respects_normalize_per_map_false() {
        let weights = vec![0.1_f32, 0.2, 0.05, 0.4, 0.1, 0.2, 0.05, 0.4];
        let map = AttentionMap::new(weights, 1, 2, 2, 2, 1, None).expect("valid map");
        let mut config = AttentionVizConfig::default();
        config.normalize_per_map = false;
        let img_off = attention_map_to_image(&map, &config);
        config.normalize_per_map = true;
        let img_on = attention_map_to_image(&map, &config);
        assert_ne!(
            img_off, img_on,
            "normalize_per_map should change the rendered heatmap"
        );
    }

    // -----------------------------------------------------------------------
    // overlay_attention_on_image_with_config / overlay_alpha
    // -----------------------------------------------------------------------

    #[test]
    fn test_overlay_attention_on_image_with_config_uses_configured_alpha() {
        let base = vec![0u8, 0, 0, 255];
        let attention = vec![255u8, 255, 255, 255];
        let mut config = AttentionVizConfig::default();
        config.overlay_alpha = 0.25;
        let via_config = overlay_attention_on_image_with_config(&base, &attention, &config);
        let via_explicit = overlay_attention_on_image(&base, &attention, 0.25);
        assert_eq!(via_config, via_explicit);
        // And it must actually differ from the *default* overlay_alpha (0.7).
        let via_default_alpha = overlay_attention_on_image(&base, &attention, 0.7);
        assert_ne!(via_config, via_default_alpha);
    }
}
