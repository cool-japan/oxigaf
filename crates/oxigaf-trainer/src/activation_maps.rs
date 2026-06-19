//! Activation map generation and analysis.
//!
//! Provides heat-map utilities (CAM, saliency, Sobel) for visualising which
//! spatial regions of an input image are most important for a model's output.
//! All implementations are gradient-free and run on plain `Vec<f32>` tensors —
//! no autograd engine is required.
//!
//! # Conventions
//! - Images and feature maps are stored **row-major** (`y` is the slow axis).
//! - Feature maps are laid out as `[channel, height, width]`, i.e. element
//!   `(c, y, x)` lives at index `c * height * width + y * width + x`.
//! - All normalisation produces values in `[0, 1]`.

use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors that can be returned by activation-map operations.
#[derive(Debug, Error)]
pub enum ActivationMapError {
    /// A tensor or buffer had an unexpected number of elements.
    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionError { expected: usize, got: usize },

    /// A configuration value was out of range or inconsistent.
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// An activation buffer was completely empty (zero elements).
    #[error("Empty activations")]
    EmptyActivations,

    /// A numerical issue (overflow, underflow, NaN) was detected.
    #[error("Numerical error: {0}")]
    NumericalError(String),
}

// ─────────────────────────────────────────────────────────────────────────────
// ActivationMap
// ─────────────────────────────────────────────────────────────────────────────

/// A spatial activation map (heat map) with values normalised to `[0, 1]`.
///
/// Values are stored **row-major**: element `(x, y)` lives at index
/// `y * width + x`.
#[derive(Debug, Clone)]
pub struct ActivationMap {
    /// Number of columns.
    pub width: usize,
    /// Number of rows.
    pub height: usize,
    /// Heat-map values in `[0, 1]`, row-major (length `width * height`).
    pub values: Vec<f32>,
}

impl ActivationMap {
    /// Create a zero-filled activation map of the given dimensions.
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            values: vec![0.0_f32; width * height],
        }
    }

    /// Create an activation map from an existing flat buffer.
    ///
    /// Returns [`ActivationMapError::DimensionError`] if
    /// `values.len() != width * height`.
    pub fn from_values(
        width: usize,
        height: usize,
        values: Vec<f32>,
    ) -> Result<Self, ActivationMapError> {
        let expected = width * height;
        if values.len() != expected {
            return Err(ActivationMapError::DimensionError {
                expected,
                got: values.len(),
            });
        }
        Ok(Self {
            width,
            height,
            values,
        })
    }

    /// Get the value at `(x, y)`.  Returns `0.0` when the coordinates are
    /// out of bounds rather than panicking.
    pub fn get(&self, x: usize, y: usize) -> f32 {
        if x >= self.width || y >= self.height {
            return 0.0;
        }
        self.values[y * self.width + x]
    }

    /// Set the value at `(x, y)`.
    ///
    /// Returns [`ActivationMapError::DimensionError`] when out of bounds.
    pub fn set(&mut self, x: usize, y: usize, v: f32) -> Result<(), ActivationMapError> {
        if x >= self.width || y >= self.height {
            return Err(ActivationMapError::DimensionError {
                expected: self.width * self.height,
                got: y.saturating_mul(self.width).saturating_add(x) + 1,
            });
        }
        self.values[y * self.width + x] = v;
        Ok(())
    }

    /// Normalise values to `[0, 1]` using min–max scaling.
    ///
    /// `v' = (v - min) / (max - min + 1e-8)`.  No-op when all values are
    /// identical (the map remains unchanged to avoid introducing numerical
    /// artefacts).
    pub fn normalize(&mut self) {
        let min = self.values.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = self
            .values
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let range = max - min;
        if range < 1e-8 {
            // All values are effectively equal — leave them as-is.
            return;
        }
        let denom = range + 1e-8;
        for v in &mut self.values {
            *v = (*v - min) / denom;
        }
    }

    /// Zero-out every value that is strictly below `threshold`.
    pub fn threshold(&mut self, threshold: f32) {
        for v in &mut self.values {
            if *v < threshold {
                *v = 0.0;
            }
        }
    }

    /// Resize to `(new_width, new_height)` using bilinear interpolation.
    pub fn resize(&self, new_width: usize, new_height: usize) -> Self {
        let mut result = ActivationMap::new(new_width, new_height);
        if self.width == 0 || self.height == 0 || new_width == 0 || new_height == 0 {
            return result;
        }

        let x_scale = self.width as f32 / new_width as f32;
        let y_scale = self.height as f32 / new_height as f32;

        for ny in 0..new_height {
            for nx in 0..new_width {
                // Map destination pixel centre back to source space.
                let src_x = (nx as f32 + 0.5) * x_scale - 0.5;
                let src_y = (ny as f32 + 0.5) * y_scale - 0.5;

                let x0 = src_x.floor() as isize;
                let y0 = src_y.floor() as isize;
                let x1 = x0 + 1;
                let y1 = y0 + 1;

                let fx = src_x - x0 as f32;
                let fy = src_y - y0 as f32;

                let s = |xi: isize, yi: isize| -> f32 {
                    let xi = xi.clamp(0, self.width as isize - 1) as usize;
                    let yi = yi.clamp(0, self.height as isize - 1) as usize;
                    self.values[yi * self.width + xi]
                };

                let top = s(x0, y0) * (1.0 - fx) + s(x1, y0) * fx;
                let bot = s(x0, y1) * (1.0 - fx) + s(x1, y1) * fx;
                result.values[ny * new_width + nx] = top * (1.0 - fy) + bot * fy;
            }
        }
        result
    }

    /// Mean activation value.  Returns `0.0` for an empty map.
    pub fn mean(&self) -> f32 {
        if self.values.is_empty() {
            return 0.0;
        }
        self.values.iter().sum::<f32>() / self.values.len() as f32
    }

    /// Location of the maximum value `(x, y)`.  Returns `(0, 0)` for an
    /// empty map.
    pub fn argmax(&self) -> (usize, usize) {
        if self.values.is_empty() || self.width == 0 {
            return (0, 0);
        }
        let (idx, _) = self.values.iter().enumerate().fold(
            (0usize, f32::NEG_INFINITY),
            |(best_i, best_v), (i, &v)| {
                if v > best_v {
                    (i, v)
                } else {
                    (best_i, best_v)
                }
            },
        );
        let x = idx % self.width;
        let y = idx / self.width;
        (x, y)
    }

    /// Top-`k` activation locations sorted by value descending.
    ///
    /// Each entry is `(x, y, value)`.  If `k` exceeds the number of pixels,
    /// all pixels are returned.
    pub fn top_k(&self, k: usize) -> Vec<(usize, usize, f32)> {
        let mut indexed: Vec<(usize, f32)> = self
            .values
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();

        // Partial-sort: stable descending order.
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.truncate(k);

        indexed
            .into_iter()
            .map(|(idx, v)| {
                let x = idx % self.width.max(1);
                let y = idx / self.width.max(1);
                (x, y, v)
            })
            .collect()
    }

    /// Fraction of pixels where both `self > 0.5` and `other > 0.5`.
    ///
    /// Returns [`ActivationMapError::DimensionError`] when the maps have
    /// different sizes.
    pub fn overlap_fraction(&self, other: &ActivationMap) -> Result<f32, ActivationMapError> {
        let n = self.width * self.height;
        let n_other = other.width * other.height;
        if n != n_other || self.width != other.width {
            return Err(ActivationMapError::DimensionError {
                expected: n,
                got: n_other,
            });
        }
        if n == 0 {
            return Ok(0.0);
        }
        let both_active = self
            .values
            .iter()
            .zip(other.values.iter())
            .filter(|(&a, &b)| a > 0.5 && b > 0.5)
            .count();
        Ok(both_active as f32 / n as f32)
    }

    /// Convert to an RGB heat-map using a jet-like colourmap.
    ///
    /// The returned buffer has length `width * height * 3` (RGB, row-major).
    /// Channel order: R, G, B interleaved.
    pub fn to_rgb_heatmap(&self) -> Vec<f32> {
        let n = self.width * self.height;
        let mut out = vec![0.0_f32; n * 3];

        for (i, &v) in self.values.iter().enumerate() {
            let v = v.clamp(0.0, 1.0);
            let (r, g, b) = jet_color(v);
            out[i * 3] = r;
            out[i * 3 + 1] = g;
            out[i * 3 + 2] = b;
        }
        out
    }
}

/// Jet colourmap approximation.
///
/// ```text
/// v in [0.00, 0.25]: b = lerp(0.5, 1,  v/0.25),          g = 0, r = 0
/// v in [0.25, 0.50]: b = 1,              g = lerp(0, 1, (v-0.25)/0.25), r = 0
/// v in [0.50, 0.75]: b = lerp(1, 0, (v-0.5)/0.25),  g = 1, r = lerp(0, 1, (v-0.5)/0.25)
/// v in [0.75, 1.00]: b = 0,              g = lerp(1, 0, (v-0.75)/0.25), r = 1
/// ```
fn jet_color(v: f32) -> (f32, f32, f32) {
    if v < 0.25 {
        let t = v / 0.25;
        let b = 0.5 + 0.5 * t; // lerp(0.5, 1, t)
        (0.0, 0.0, b)
    } else if v < 0.5 {
        let t = (v - 0.25) / 0.25;
        let g = t; // lerp(0, 1, t)
        (0.0, g, 1.0)
    } else if v < 0.75 {
        let t = (v - 0.5) / 0.25;
        let b = 1.0 - t; // lerp(1, 0, t)
        let r = t; // lerp(0, 1, t)
        (r, 1.0, b)
    } else {
        let t = (v - 0.75) / 0.25;
        let g = 1.0 - t; // lerp(1, 0, t)
        (1.0, g, 0.0)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Class Activation Mapping (gradient-free CAM)
// ─────────────────────────────────────────────────────────────────────────────

/// Compute a gradient-free Class Activation Map (CAM).
///
/// Each spatial position's score is the dot product of the per-channel
/// `class_weights` with the feature values at that position.  The result is
/// ReLU-clamped and then normalised to `[0, 1]`.
///
/// # Parameters
/// * `feature_maps` — flat `[num_channels, height, width]` buffer.
/// * `class_weights` — per-channel weights (e.g. from the final linear layer).
/// * `num_channels`, `height`, `width` — shape of the feature tensor.
///
/// # Errors
/// [`ActivationMapError::DimensionError`] if the buffer sizes are inconsistent.
pub fn compute_cam(
    feature_maps: &[f32],
    class_weights: &[f32],
    num_channels: usize,
    height: usize,
    width: usize,
) -> Result<ActivationMap, ActivationMapError> {
    let expected_features = num_channels * height * width;
    if feature_maps.len() != expected_features {
        return Err(ActivationMapError::DimensionError {
            expected: expected_features,
            got: feature_maps.len(),
        });
    }
    if class_weights.len() != num_channels {
        return Err(ActivationMapError::DimensionError {
            expected: num_channels,
            got: class_weights.len(),
        });
    }

    let spatial = height * width;
    let mut values = vec![0.0_f32; spatial];

    for (c, &w) in class_weights.iter().enumerate().take(num_channels) {
        let channel_offset = c * spatial;
        for s in 0..spatial {
            values[s] += w * feature_maps[channel_offset + s];
        }
    }

    // ReLU: clamp to [0, ∞)
    for v in &mut values {
        if *v < 0.0 {
            *v = 0.0;
        }
    }

    let mut map = ActivationMap {
        width,
        height,
        values,
    };
    map.normalize();
    Ok(map)
}

// ─────────────────────────────────────────────────────────────────────────────
// Gradient-based attention (simplified finite-difference saliency)
// ─────────────────────────────────────────────────────────────────────────────

/// Compute a saliency map via a Sobel-based approximation of the input gradient.
///
/// The image is converted to luminance and then passed through a Sobel edge
/// detector.  The `eps` parameter is accepted for API compatibility with
/// gradient-based implementations but is not used in this Sobel path.
///
/// # Parameters
/// * `image` — RGB f32 buffer, length `width * height * 3`.
/// * `width`, `height` — image dimensions.
/// * `eps` — perturbation size (reserved; not used in the Sobel path).
///
/// # Errors
/// [`ActivationMapError::DimensionError`] if `image.len() != width * height * 3`.
pub fn finite_difference_saliency(
    image: &[f32],
    width: usize,
    height: usize,
    eps: f32,
) -> Result<ActivationMap, ActivationMapError> {
    // eps is accepted for API compatibility; suppress unused-variable lint.
    let _ = eps;

    let expected = width * height * 3;
    if image.len() != expected {
        return Err(ActivationMapError::DimensionError {
            expected,
            got: image.len(),
        });
    }

    // Convert to luminance.
    let luma: Vec<f32> = (0..width * height)
        .map(|i| {
            let r = image[i * 3];
            let g = image[i * 3 + 1];
            let b = image[i * 3 + 2];
            0.2126 * r + 0.7152 * g + 0.0722 * b
        })
        .collect();

    Ok(sobel_saliency(&luma, width, height))
}

/// Apply a Sobel edge detector to a single-channel image and return the
/// normalised edge-magnitude map.
///
/// Sobel kernels:
/// ```text
/// Gx = [-1, 0, 1; -2, 0, 2; -1, 0, 1]
/// Gy = [ 1, 2, 1;  0, 0, 0; -1,-2,-1]
/// ```
///
/// Pixels at the border have their out-of-bounds neighbours clamped to the
/// nearest valid pixel (border-replicate padding).
///
/// # Parameters
/// * `image_luma` — single-channel f32 image, length `width * height`.
/// * `width`, `height` — image dimensions.
pub fn sobel_saliency(image_luma: &[f32], width: usize, height: usize) -> ActivationMap {
    let n = width * height;
    let mut edge_mag = vec![0.0_f32; n];

    // Safe pixel sampler with border-replicate padding.
    let px = |xi: isize, yi: isize| -> f32 {
        let xi = xi.clamp(0, width as isize - 1) as usize;
        let yi = yi.clamp(0, height as isize - 1) as usize;
        if yi * width + xi < image_luma.len() {
            image_luma[yi * width + xi]
        } else {
            0.0
        }
    };

    for y in 0..height {
        for x in 0..width {
            let xi = x as isize;
            let yi = y as isize;

            // Gx convolution
            let gx = -px(xi - 1, yi - 1)
                + px(xi + 1, yi - 1)
                + -2.0 * px(xi - 1, yi)
                + 2.0 * px(xi + 1, yi)
                + -px(xi - 1, yi + 1)
                + px(xi + 1, yi + 1);

            // Gy convolution
            let gy = px(xi - 1, yi - 1)
                + 2.0 * px(xi, yi - 1)
                + px(xi + 1, yi - 1)
                + -px(xi - 1, yi + 1)
                + -2.0 * px(xi, yi + 1)
                + -px(xi + 1, yi + 1);

            edge_mag[y * width + x] = (gx * gx + gy * gy).sqrt();
        }
    }

    let mut map = ActivationMap {
        width,
        height,
        values: edge_mag,
    };
    map.normalize();
    map
}

// ─────────────────────────────────────────────────────────────────────────────
// Feature attribution
// ─────────────────────────────────────────────────────────────────────────────

/// How to normalise channel scores before computing score-weighted attribution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttributionNorm {
    /// Use raw scores with no normalisation.
    None,
    /// Apply a softmax across all channels (scores sum to 1.0).
    Softmax,
    /// L1-normalise (divide by the sum of absolute values).
    L1,
}

/// Configuration for [`score_weighted_attribution`].
#[derive(Debug, Clone)]
pub struct AttributionConfig {
    /// How to normalise channel scores.
    pub normalization: AttributionNorm,
    /// Softmax temperature (only used when `normalization == Softmax`).
    pub temperature: f32,
}

impl Default for AttributionConfig {
    fn default() -> Self {
        Self {
            normalization: AttributionNorm::None,
            temperature: 1.0,
        }
    }
}

/// Score-weighted feature attribution.
///
/// For each spatial position `(x, y)`:
///
/// ```text
/// attribution[y*W + x] = Σ_c  score[c] · relu(feature[c*H*W + y*W + x])
/// ```
///
/// The result is normalised to `[0, 1]`.
///
/// # Errors
/// [`ActivationMapError::DimensionError`] if sizes are inconsistent.
/// [`ActivationMapError::InvalidConfig`] if the temperature is non-positive.
pub fn score_weighted_attribution(
    feature_maps: &[f32],
    channel_scores: &[f32],
    num_channels: usize,
    height: usize,
    width: usize,
    config: &AttributionConfig,
) -> Result<ActivationMap, ActivationMapError> {
    let expected_features = num_channels * height * width;
    if feature_maps.len() != expected_features {
        return Err(ActivationMapError::DimensionError {
            expected: expected_features,
            got: feature_maps.len(),
        });
    }
    if channel_scores.len() != num_channels {
        return Err(ActivationMapError::DimensionError {
            expected: num_channels,
            got: channel_scores.len(),
        });
    }
    if config.temperature <= 0.0 {
        return Err(ActivationMapError::InvalidConfig(format!(
            "temperature must be positive, got {}",
            config.temperature
        )));
    }

    // Normalise scores according to config.
    let scores: Vec<f32> = match config.normalization {
        AttributionNorm::None => channel_scores.to_vec(),
        AttributionNorm::Softmax => softmax_vec(channel_scores, config.temperature),
        AttributionNorm::L1 => {
            let l1_sum: f32 = channel_scores.iter().map(|s| s.abs()).sum();
            if l1_sum < 1e-12 {
                channel_scores.to_vec()
            } else {
                channel_scores.iter().map(|s| s / l1_sum).collect()
            }
        }
    };

    let spatial = height * width;
    let mut values = vec![0.0_f32; spatial];

    for (c, &s) in scores.iter().enumerate().take(num_channels) {
        let offset = c * spatial;
        for i in 0..spatial {
            let feat = feature_maps[offset + i].max(0.0); // ReLU
            values[i] += s * feat;
        }
    }

    let mut map = ActivationMap {
        width,
        height,
        values,
    };
    map.normalize();
    Ok(map)
}

/// Numerically stable softmax with temperature.
fn softmax_vec(scores: &[f32], temperature: f32) -> Vec<f32> {
    if scores.is_empty() {
        return Vec::new();
    }
    let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let scaled: Vec<f32> = scores
        .iter()
        .map(|s| ((s - max_s) / temperature).exp())
        .collect();
    let sum: f32 = scaled.iter().sum();
    if sum < 1e-30 {
        // Degenerate: return uniform distribution.
        let n = scores.len();
        return vec![1.0 / n as f32; n];
    }
    scaled.iter().map(|v| v / sum).collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Activation map operations
// ─────────────────────────────────────────────────────────────────────────────

/// Linearly blend two activation maps.
///
/// `output = weight_a * a + (1 - weight_a) * b`, then normalise to `[0, 1]`.
///
/// # Errors
/// [`ActivationMapError::DimensionError`] if the maps have different sizes.
pub fn blend_maps(
    a: &ActivationMap,
    b: &ActivationMap,
    weight_a: f32,
) -> Result<ActivationMap, ActivationMapError> {
    let na = a.width * a.height;
    let nb = b.width * b.height;
    if na != nb || a.width != b.width {
        return Err(ActivationMapError::DimensionError {
            expected: na,
            got: nb,
        });
    }
    let weight_b = 1.0 - weight_a;
    let values: Vec<f32> = a
        .values
        .iter()
        .zip(b.values.iter())
        .map(|(&va, &vb)| weight_a * va + weight_b * vb)
        .collect();
    let mut map = ActivationMap {
        width: a.width,
        height: a.height,
        values,
    };
    map.normalize();
    Ok(map)
}

/// Compute the pixel-wise union of two binary-thresholded activation maps.
///
/// `output[i] = max(a[i] > threshold ? 1.0 : 0.0, b[i] > threshold ? 1.0 : 0.0)`
///
/// # Errors
/// [`ActivationMapError::DimensionError`] if the maps have different sizes.
pub fn union_maps(
    a: &ActivationMap,
    b: &ActivationMap,
    threshold: f32,
) -> Result<ActivationMap, ActivationMapError> {
    let na = a.width * a.height;
    let nb = b.width * b.height;
    if na != nb || a.width != b.width {
        return Err(ActivationMapError::DimensionError {
            expected: na,
            got: nb,
        });
    }
    let values: Vec<f32> = a
        .values
        .iter()
        .zip(b.values.iter())
        .map(|(&va, &vb)| {
            let ba = if va > threshold { 1.0_f32 } else { 0.0_f32 };
            let bb = if vb > threshold { 1.0_f32 } else { 0.0_f32 };
            ba.max(bb)
        })
        .collect();
    Ok(ActivationMap {
        width: a.width,
        height: a.height,
        values,
    })
}

/// Compute the pixel-wise intersection of two binary-thresholded activation maps.
///
/// `output[i] = (a[i] > threshold AND b[i] > threshold) ? 1.0 : 0.0`
///
/// # Errors
/// [`ActivationMapError::DimensionError`] if the maps have different sizes.
pub fn intersect_maps(
    a: &ActivationMap,
    b: &ActivationMap,
    threshold: f32,
) -> Result<ActivationMap, ActivationMapError> {
    let na = a.width * a.height;
    let nb = b.width * b.height;
    if na != nb || a.width != b.width {
        return Err(ActivationMapError::DimensionError {
            expected: na,
            got: nb,
        });
    }
    let values: Vec<f32> = a
        .values
        .iter()
        .zip(b.values.iter())
        .map(|(&va, &vb)| {
            if va > threshold && vb > threshold {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    Ok(ActivationMap {
        width: a.width,
        height: a.height,
        values,
    })
}

/// Smooth an activation map with a separable Gaussian blur.
///
/// The kernel radius is `ceil(3 * sigma)`.  When `sigma <= 0` the input is
/// returned unchanged (clone).  The result is normalised after blurring.
pub fn smooth_map(map: &ActivationMap, sigma: f32) -> ActivationMap {
    if sigma <= 0.0 || map.width == 0 || map.height == 0 {
        return map.clone();
    }

    let radius = (3.0 * sigma).ceil() as usize;
    let kernel = gaussian_kernel_1d(sigma, radius);

    // Horizontal pass: blur along x.
    let h_blurred = convolve_horizontal(&map.values, map.width, map.height, &kernel, radius);
    // Vertical pass: blur along y.
    let v_blurred = convolve_vertical(&h_blurred, map.width, map.height, &kernel, radius);

    let mut result = ActivationMap {
        width: map.width,
        height: map.height,
        values: v_blurred,
    };
    result.normalize();
    result
}

/// Build a normalised 1-D Gaussian kernel of half-width `radius`.
fn gaussian_kernel_1d(sigma: f32, radius: usize) -> Vec<f32> {
    let size = 2 * radius + 1;
    let mut k = Vec::with_capacity(size);
    let two_s2 = 2.0 * sigma * sigma;
    for i in 0..size {
        let d = i as f32 - radius as f32;
        k.push((-d * d / two_s2).exp());
    }
    let sum: f32 = k.iter().sum();
    if sum > 1e-30 {
        for v in &mut k {
            *v /= sum;
        }
    }
    k
}

/// Convolve a flat row-major image along the x-axis with a 1-D kernel.
fn convolve_horizontal(
    src: &[f32],
    width: usize,
    height: usize,
    kernel: &[f32],
    radius: usize,
) -> Vec<f32> {
    let mut dst = vec![0.0_f32; width * height];
    for y in 0..height {
        for x in 0..width {
            let mut acc = 0.0_f32;
            for (ki, &kv) in kernel.iter().enumerate() {
                let sx = x as isize + ki as isize - radius as isize;
                let sx = sx.clamp(0, width as isize - 1) as usize;
                acc += src[y * width + sx] * kv;
            }
            dst[y * width + x] = acc;
        }
    }
    dst
}

/// Convolve a flat row-major image along the y-axis with a 1-D kernel.
fn convolve_vertical(
    src: &[f32],
    width: usize,
    height: usize,
    kernel: &[f32],
    radius: usize,
) -> Vec<f32> {
    let mut dst = vec![0.0_f32; width * height];
    for y in 0..height {
        for x in 0..width {
            let mut acc = 0.0_f32;
            for (ki, &kv) in kernel.iter().enumerate() {
                let sy = y as isize + ki as isize - radius as isize;
                let sy = sy.clamp(0, height as isize - 1) as usize;
                acc += src[sy * width + x] * kv;
            }
            dst[y * width + x] = acc;
        }
    }
    dst
}

/// Overlay an activation heat map on an RGB image.
///
/// `output[i] = (1 - alpha) * image[i] + alpha * heatmap_rgb[i]`
///
/// # Parameters
/// * `image` — RGB f32 buffer, length `width * height * 3`.
/// * `map` — activation map of the same spatial dimensions.
/// * `alpha` — blending weight in `[0, 1]` (0 = original, 1 = heat map).
///
/// # Errors
/// [`ActivationMapError::DimensionError`] if `image.len() != width * height * 3`.
pub fn overlay_on_image(
    image: &[f32],
    map: &ActivationMap,
    alpha: f32,
) -> Result<Vec<f32>, ActivationMapError> {
    let expected = map.width * map.height * 3;
    if image.len() != expected {
        return Err(ActivationMapError::DimensionError {
            expected,
            got: image.len(),
        });
    }
    let heatmap = map.to_rgb_heatmap();
    let inv_alpha = 1.0 - alpha;
    let output: Vec<f32> = image
        .iter()
        .zip(heatmap.iter())
        .map(|(&img_v, &heat_v)| inv_alpha * img_v + alpha * heat_v)
        .collect();
    Ok(output)
}

// ─────────────────────────────────────────────────────────────────────────────
// ActivationStats
// ─────────────────────────────────────────────────────────────────────────────

/// Summary statistics for an activation map.
#[derive(Debug, Clone)]
pub struct ActivationStats {
    /// Mean of all activation values.
    pub mean_activation: f32,
    /// Maximum activation value.
    pub max_activation: f32,
    /// Fraction of pixels with value > 0.5.
    pub fraction_active: f32,
    /// Shannon entropy (bits) of the normalised activation distribution.
    ///
    /// Computed as `-Σ p · log₂(p + 1e-8)` where `p` is each value divided
    /// by the sum of all values.
    pub spatial_entropy: f32,
    /// X coordinate of the peak activation.
    pub peak_x: usize,
    /// Y coordinate of the peak activation.
    pub peak_y: usize,
}

/// Compute summary statistics for an activation map.
pub fn compute_activation_stats(map: &ActivationMap) -> ActivationStats {
    let n = map.values.len();

    if n == 0 {
        return ActivationStats {
            mean_activation: 0.0,
            max_activation: 0.0,
            fraction_active: 0.0,
            spatial_entropy: 0.0,
            peak_x: 0,
            peak_y: 0,
        };
    }

    let sum: f32 = map.values.iter().sum();
    let mean_activation = sum / n as f32;
    let max_activation = map.values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let active_count = map.values.iter().filter(|&&v| v > 0.5).count();
    let fraction_active = active_count as f32 / n as f32;

    // Spatial entropy: normalise to form a probability distribution.
    let norm_sum = sum.max(1e-30);
    let spatial_entropy: f32 = map
        .values
        .iter()
        .map(|&v| {
            let p = v / norm_sum;
            -p * (p + 1e-8_f32).log2()
        })
        .sum();

    let (peak_x, peak_y) = map.argmax();

    ActivationStats {
        mean_activation,
        max_activation,
        fraction_active,
        spatial_entropy,
        peak_x,
        peak_y,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: roughly-equal for f32 comparisons.
    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    // ── ActivationMap::new ────────────────────────────────────────────────────

    #[test]
    fn test_new_zeros() {
        let m = ActivationMap::new(4, 3);
        assert_eq!(m.width, 4);
        assert_eq!(m.height, 3);
        assert_eq!(m.values.len(), 12);
        assert!(m.values.iter().all(|&v| v == 0.0));
    }

    // ── ActivationMap::from_values ────────────────────────────────────────────

    #[test]
    fn test_from_values_correct() {
        let v = vec![1.0, 2.0, 3.0, 4.0];
        let m = ActivationMap::from_values(2, 2, v.clone()).expect("should succeed");
        assert_eq!(m.values, v);
    }

    #[test]
    fn test_from_values_dimension_error() {
        let v = vec![1.0, 2.0, 3.0]; // 3 values for 2×2 → error
        let err = ActivationMap::from_values(2, 2, v).expect_err("expected error");
        matches!(
            err,
            ActivationMapError::DimensionError {
                expected: 4,
                got: 3
            }
        );
    }

    // ── ActivationMap::get / set ──────────────────────────────────────────────

    #[test]
    fn test_get_set_roundtrip() {
        let mut m = ActivationMap::new(3, 3);
        m.set(1, 2, 0.75).expect("set should succeed");
        assert!(approx_eq(m.get(1, 2), 0.75, 1e-7));
    }

    #[test]
    fn test_get_out_of_bounds() {
        let m = ActivationMap::new(3, 3);
        assert_eq!(m.get(10, 10), 0.0);
    }

    #[test]
    fn test_set_out_of_bounds_error() {
        let mut m = ActivationMap::new(3, 3);
        assert!(m.set(5, 5, 1.0).is_err());
    }

    // ── ActivationMap::normalize ──────────────────────────────────────────────

    #[test]
    fn test_normalize_uniform_unchanged() {
        let v = vec![0.5, 0.5, 0.5, 0.5];
        let mut m = ActivationMap::from_values(2, 2, v).expect("ok");
        m.normalize();
        // When all values are equal the map is left unchanged.
        assert!(m.values.iter().all(|&v| approx_eq(v, 0.5, 1e-6)));
    }

    #[test]
    fn test_normalize_range() {
        let v = vec![0.0, 2.0, 4.0, 8.0];
        let mut m = ActivationMap::from_values(2, 2, v).expect("ok");
        m.normalize();
        let min = m.values.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = m.values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(approx_eq(min, 0.0, 1e-5));
        assert!(approx_eq(max, 1.0, 1e-5));
    }

    // ── ActivationMap::threshold ──────────────────────────────────────────────

    #[test]
    fn test_threshold_zeros_low_values() {
        let v = vec![0.1, 0.3, 0.6, 0.9];
        let mut m = ActivationMap::from_values(2, 2, v).expect("ok");
        m.threshold(0.5);
        assert_eq!(m.values[0], 0.0);
        assert_eq!(m.values[1], 0.0);
        assert!(m.values[2] > 0.0);
        assert!(m.values[3] > 0.0);
    }

    // ── ActivationMap::resize ─────────────────────────────────────────────────

    #[test]
    fn test_resize_dimensions() {
        let m = ActivationMap::new(2, 2);
        let r = m.resize(4, 4);
        assert_eq!(r.width, 4);
        assert_eq!(r.height, 4);
        assert_eq!(r.values.len(), 16);
    }

    #[test]
    fn test_resize_uniform_stays_uniform() {
        let v = vec![0.5; 4];
        let m = ActivationMap::from_values(2, 2, v).expect("ok");
        let r = m.resize(4, 4);
        for &v in &r.values {
            assert!(approx_eq(v, 0.5, 1e-5));
        }
    }

    // ── ActivationMap::mean ───────────────────────────────────────────────────

    #[test]
    fn test_mean_basic() {
        let v = vec![0.0, 0.5, 0.5, 1.0];
        let m = ActivationMap::from_values(2, 2, v).expect("ok");
        assert!(approx_eq(m.mean(), 0.5, 1e-6));
    }

    // ── ActivationMap::argmax ─────────────────────────────────────────────────

    #[test]
    fn test_argmax_single_peak() {
        // 3×3 map with peak at (2, 1) → index 5
        let mut v = vec![0.0; 9];
        v[3 + 2] = 1.0; // y=1, x=2
        let m = ActivationMap::from_values(3, 3, v).expect("ok");
        assert_eq!(m.argmax(), (2, 1));
    }

    // ── ActivationMap::top_k ──────────────────────────────────────────────────

    #[test]
    fn test_top_k_sorted_descending() {
        let v = vec![0.1, 0.9, 0.4, 0.7];
        let m = ActivationMap::from_values(2, 2, v).expect("ok");
        let top = m.top_k(3);
        assert_eq!(top.len(), 3);
        // Values must be in descending order.
        for i in 0..top.len() - 1 {
            assert!(top[i].2 >= top[i + 1].2);
        }
        // Highest value is 0.9.
        assert!(approx_eq(top[0].2, 0.9, 1e-6));
    }

    // ── ActivationMap::overlap_fraction ──────────────────────────────────────

    #[test]
    fn test_overlap_fraction_identical_above_half() {
        let v = vec![0.8, 0.8, 0.8, 0.8];
        let m = ActivationMap::from_values(2, 2, v).expect("ok");
        let frac = m.overlap_fraction(&m).expect("same size");
        assert!(approx_eq(frac, 1.0, 1e-6));
    }

    #[test]
    fn test_overlap_fraction_no_overlap() {
        let a = ActivationMap::from_values(2, 2, vec![0.8, 0.8, 0.2, 0.2]).expect("ok");
        let b = ActivationMap::from_values(2, 2, vec![0.2, 0.2, 0.8, 0.8]).expect("ok");
        let frac = a.overlap_fraction(&b).expect("same size");
        assert!(approx_eq(frac, 0.0, 1e-6));
    }

    #[test]
    fn test_overlap_fraction_size_mismatch() {
        let a = ActivationMap::new(2, 2);
        let b = ActivationMap::new(3, 3);
        assert!(a.overlap_fraction(&b).is_err());
    }

    // ── ActivationMap::to_rgb_heatmap ─────────────────────────────────────────

    #[test]
    fn test_to_rgb_heatmap_correct_size() {
        let m = ActivationMap::new(4, 3);
        let rgb = m.to_rgb_heatmap();
        assert_eq!(rgb.len(), 4 * 3 * 3);
    }

    #[test]
    fn test_to_rgb_heatmap_values_in_range() {
        let v: Vec<f32> = (0..16).map(|i| i as f32 / 15.0).collect();
        let m = ActivationMap::from_values(4, 4, v).expect("ok");
        let rgb = m.to_rgb_heatmap();
        for &c in &rgb {
            assert!((0.0..=1.0).contains(&c), "channel value {} out of [0,1]", c);
        }
    }

    // ── compute_cam ───────────────────────────────────────────────────────────

    #[test]
    fn test_compute_cam_uniform_weights() {
        // Uniform feature maps with uniform weights → uniform CAM (all equal).
        let nc = 4;
        let h = 3;
        let w = 3;
        let features = vec![0.5_f32; nc * h * w];
        let weights = vec![1.0_f32; nc];
        let cam = compute_cam(&features, &weights, nc, h, w).expect("ok");
        let mean = cam.mean();
        for &v in &cam.values {
            assert!(approx_eq(v, mean, 1e-5));
        }
    }

    #[test]
    fn test_compute_cam_wrong_dimensions() {
        let nc = 3;
        let h = 4;
        let w = 4;
        let features = vec![0.0_f32; nc * h * w + 1]; // one extra → error
        let weights = vec![1.0_f32; nc];
        assert!(compute_cam(&features, &weights, nc, h, w).is_err());
    }

    // ── finite_difference_saliency ────────────────────────────────────────────

    #[test]
    fn test_finite_difference_saliency_dimensions() {
        let w = 8;
        let h = 6;
        let image = vec![0.5_f32; w * h * 3];
        let map = finite_difference_saliency(&image, w, h, 0.01).expect("ok");
        assert_eq!(map.width, w);
        assert_eq!(map.height, h);
        assert_eq!(map.values.len(), w * h);
    }

    #[test]
    fn test_finite_difference_saliency_uniform_image_low_saliency() {
        let w = 8;
        let h = 8;
        let image = vec![0.5_f32; w * h * 3];
        let map = finite_difference_saliency(&image, w, h, 0.01).expect("ok");
        // A uniform image has no edges → all saliency values should be 0.
        assert!(map.values.iter().all(|&v| approx_eq(v, 0.0, 1e-5)));
    }

    // ── sobel_saliency ────────────────────────────────────────────────────────

    #[test]
    fn test_sobel_uniform_image_near_zero() {
        let w = 6;
        let h = 6;
        let luma = vec![0.5_f32; w * h];
        let map = sobel_saliency(&luma, w, h);
        assert!(map.values.iter().all(|&v| approx_eq(v, 0.0, 1e-5)));
    }

    #[test]
    fn test_sobel_step_function_has_high_edge() {
        // Left half = 0, right half = 1 → strong vertical edge in the middle.
        let w = 8;
        let h = 8;
        let luma: Vec<f32> = (0..w * h)
            .map(|i| if (i % w) < w / 2 { 0.0 } else { 1.0 })
            .collect();
        let map = sobel_saliency(&luma, w, h);
        // Verify the edge column (x = w/2 - 1 or w/2) has the maximum value.
        let (peak_x, _) = map.argmax();
        // Peak should be at the step boundary.
        assert!(peak_x == w / 2 - 1 || peak_x == w / 2);
        assert!(
            map.get(peak_x, h / 2) > 0.5,
            "expected high saliency at edge"
        );
    }

    // ── score_weighted_attribution ────────────────────────────────────────────

    #[test]
    fn test_score_weighted_attribution_basic() {
        let nc = 2;
        let h = 2;
        let w = 2;
        // Channel 0 all 1s, channel 1 all 0s.
        let features = vec![1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        let scores = vec![1.0_f32, 0.0_f32];
        let config = AttributionConfig::default();
        let map = score_weighted_attribution(&features, &scores, nc, h, w, &config).expect("ok");
        // All pixels get score 1 from channel 0.
        let mean = map.mean();
        for &v in &map.values {
            assert!(approx_eq(v, mean, 1e-5));
        }
    }

    // ── blend_maps ────────────────────────────────────────────────────────────

    #[test]
    fn test_blend_maps_weight_a_one() {
        let a = ActivationMap::from_values(2, 2, vec![1.0, 1.0, 1.0, 1.0]).expect("ok");
        let b = ActivationMap::from_values(2, 2, vec![0.0, 0.0, 0.0, 0.0]).expect("ok");
        let result = blend_maps(&a, &b, 1.0).expect("ok");
        // weight_a = 1 → same as a (all same value after normalize).
        let mean = result.mean();
        for &v in &result.values {
            assert!(approx_eq(v, mean, 1e-5));
        }
    }

    #[test]
    fn test_blend_maps_weight_a_zero() {
        let a = ActivationMap::from_values(2, 2, vec![0.0; 4]).expect("ok");
        let b = ActivationMap::from_values(2, 2, vec![1.0, 0.5, 0.5, 0.0]).expect("ok");
        // weight_a = 0 → blend is just b, then normalised
        let result = blend_maps(&a, &b, 0.0).expect("ok");
        // The max of b (=1.0) should map to 1.0, min (=0.0) to 0.0 after normalize.
        let max_v = result
            .values
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        assert!(approx_eq(max_v, 1.0, 1e-5));
    }

    // ── union_maps / intersect_maps ───────────────────────────────────────────

    #[test]
    fn test_union_maps_no_overlap() {
        let a = ActivationMap::from_values(2, 2, vec![0.8, 0.8, 0.2, 0.2]).expect("ok");
        let b = ActivationMap::from_values(2, 2, vec![0.2, 0.2, 0.8, 0.8]).expect("ok");
        let u = union_maps(&a, &b, 0.5).expect("ok");
        // All pixels are above threshold in at least one map.
        assert!(u.values.iter().all(|&v| approx_eq(v, 1.0, 1e-6)));
    }

    #[test]
    fn test_intersect_maps_full_overlap() {
        let a = ActivationMap::from_values(2, 2, vec![0.8; 4]).expect("ok");
        let b = ActivationMap::from_values(2, 2, vec![0.9; 4]).expect("ok");
        let inter = intersect_maps(&a, &b, 0.5).expect("ok");
        assert!(inter.values.iter().all(|&v| approx_eq(v, 1.0, 1e-6)));
    }

    // ── smooth_map ────────────────────────────────────────────────────────────

    #[test]
    fn test_smooth_map_sigma_zero_unchanged() {
        let v = vec![0.1, 0.9, 0.4, 0.7];
        let m = ActivationMap::from_values(2, 2, v.clone()).expect("ok");
        let s = smooth_map(&m, 0.0);
        // Should be a clone of m (values unchanged).
        for (a, b) in m.values.iter().zip(s.values.iter()) {
            assert!(approx_eq(*a, *b, 1e-6));
        }
    }

    #[test]
    fn test_smooth_map_blurs_peak() {
        // 5×5 map with a single bright pixel in the centre.
        let mut v = vec![0.0_f32; 25];
        v[2 * 5 + 2] = 1.0;
        let m = ActivationMap::from_values(5, 5, v).expect("ok");
        let s = smooth_map(&m, 1.0);
        // After blurring the peak value should be less than 1.0.
        let max_v = s.values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(max_v < 1.0 + 1e-5); // normalise brings max back to 1 …
                                     // … but the surrounding pixels should also be non-zero.
        assert!(s.get(2, 1) > 0.0 || s.get(1, 2) > 0.0);
    }

    // ── overlay_on_image ──────────────────────────────────────────────────────

    #[test]
    fn test_overlay_alpha_zero_same_as_image() {
        let w = 4;
        let h = 4;
        let image: Vec<f32> = (0..w * h * 3)
            .map(|i| i as f32 / (w * h * 3) as f32)
            .collect();
        let map = ActivationMap::new(w, h);
        let out = overlay_on_image(&image, &map, 0.0).expect("ok");
        for (a, b) in image.iter().zip(out.iter()) {
            assert!(approx_eq(*a, *b, 1e-6));
        }
    }

    // ── compute_activation_stats ──────────────────────────────────────────────

    #[test]
    fn test_compute_activation_stats_mean() {
        let v = vec![0.0, 0.5, 0.5, 1.0];
        let m = ActivationMap::from_values(2, 2, v).expect("ok");
        let stats = compute_activation_stats(&m);
        assert!(approx_eq(stats.mean_activation, 0.5, 1e-6));
    }

    #[test]
    fn test_compute_activation_stats_fraction_active() {
        // Two pixels above 0.5, two below.
        let v = vec![0.2, 0.2, 0.8, 0.9];
        let m = ActivationMap::from_values(2, 2, v).expect("ok");
        let stats = compute_activation_stats(&m);
        assert!(approx_eq(stats.fraction_active, 0.5, 1e-6));
    }

    #[test]
    fn test_compute_activation_stats_peak_location() {
        let mut v = vec![0.0_f32; 9];
        v[3 + 2] = 1.0; // peak at (x=2, y=1)
        let m = ActivationMap::from_values(3, 3, v).expect("ok");
        let stats = compute_activation_stats(&m);
        assert_eq!(stats.peak_x, 2);
        assert_eq!(stats.peak_y, 1);
    }

    #[test]
    fn test_compute_activation_stats_entropy_nonneg() {
        let v: Vec<f32> = (0..16).map(|i| i as f32 / 15.0).collect();
        let m = ActivationMap::from_values(4, 4, v).expect("ok");
        let stats = compute_activation_stats(&m);
        assert!(stats.spatial_entropy >= 0.0);
    }
}
