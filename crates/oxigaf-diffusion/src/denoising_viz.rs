//! Denoising step visualization for DDIM inference.
//!
//! During DDIM inference, capturing each denoising step's output allows users
//! to animate the denoising process. This module handles that visualization
//! without needing real weights.

use crate::DiffusionError;

// ---------------------------------------------------------------------------
// DenoisingStep
// ---------------------------------------------------------------------------

/// A snapshot of one DDIM denoising step.
#[derive(Debug, Clone)]
pub struct DenoisingStep {
    pub step_index: usize,
    pub total_steps: usize,
    pub timestep: u32,
    /// Latent representation at this step: `[channels * height * width]` f32.
    pub latent: Vec<f32>,
    pub latent_channels: usize,
    pub latent_height: usize,
    pub latent_width: usize,
    /// Optional decoded image (may be None if we skipped decoding this step).
    /// RGBA bytes, `image_width * image_height * 4`.
    pub decoded_image: Option<Vec<u8>>,
    pub image_width: usize,
    pub image_height: usize,
}

impl DenoisingStep {
    /// Create a new denoising step snapshot.
    pub fn new(
        step_index: usize,
        total_steps: usize,
        timestep: u32,
        latent: Vec<f32>,
        latent_channels: usize,
        latent_height: usize,
        latent_width: usize,
    ) -> Self {
        Self {
            step_index,
            total_steps,
            timestep,
            latent,
            latent_channels,
            latent_height,
            latent_width,
            decoded_image: None,
            image_width: 0,
            image_height: 0,
        }
    }

    /// Returns `step_index as f32 / max(total_steps, 1)`.
    pub fn progress_fraction(&self) -> f32 {
        self.step_index as f32 / self.total_steps.max(1) as f32
    }

    /// Returns `(min, max, mean)` of latent values.
    ///
    /// Returns `(0.0, 0.0, 0.0)` for an empty latent.
    pub fn latent_stats(&self) -> (f32, f32, f32) {
        if self.latent.is_empty() {
            return (0.0, 0.0, 0.0);
        }

        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;
        let mut sum = 0.0_f32;

        for &v in &self.latent {
            if v < min_val {
                min_val = v;
            }
            if v > max_val {
                max_val = v;
            }
            sum += v;
        }

        let mean = sum / self.latent.len() as f32;
        (min_val, max_val, mean)
    }

    /// Builder: attach a decoded RGBA image to this step.
    pub fn with_decoded(mut self, image: Vec<u8>, width: usize, height: usize) -> Self {
        self.decoded_image = Some(image);
        self.image_width = width;
        self.image_height = height;
        self
    }
}

// ---------------------------------------------------------------------------
// DenoisingTimeline
// ---------------------------------------------------------------------------

/// A complete timeline of denoising steps for a single view.
#[derive(Debug, Clone)]
pub struct DenoisingTimeline {
    pub steps: Vec<DenoisingStep>,
    pub view_index: usize,
    pub total_views: usize,
}

impl DenoisingTimeline {
    /// Create a new empty timeline for a given view.
    pub fn new(view_index: usize, total_views: usize) -> Self {
        Self {
            steps: Vec::new(),
            view_index,
            total_views,
        }
    }

    /// Append a step to this timeline.
    pub fn push_step(&mut self, step: DenoisingStep) {
        self.steps.push(step);
    }

    /// Number of captured steps.
    pub fn num_steps(&self) -> usize {
        self.steps.len()
    }

    /// First step captured, if any.
    pub fn first_step(&self) -> Option<&DenoisingStep> {
        self.steps.first()
    }

    /// Last step captured, if any.
    pub fn last_step(&self) -> Option<&DenoisingStep> {
        self.steps.last()
    }

    /// Human-readable summary of this timeline.
    ///
    /// Format: `"View 0/4 | Steps: 20 | Latent: 4x32x32"`
    pub fn format_summary(&self) -> String {
        let (channels, height, width) = if let Some(step) = self.steps.first() {
            (step.latent_channels, step.latent_height, step.latent_width)
        } else {
            (0, 0, 0)
        };
        format!(
            "View {}/{} | Steps: {} | Latent: {}x{}x{}",
            self.view_index,
            self.total_views,
            self.steps.len(),
            channels,
            height,
            width
        )
    }
}

// ---------------------------------------------------------------------------
// LatentColormap
// ---------------------------------------------------------------------------

/// Colormap for converting latent tensors to RGBA images.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LatentColormap {
    /// Normalize to \[0,1\], use first 3 channels as RGB.
    RgbNormalized,
    /// Map to grayscale using channel 0 only.
    GrayscaleChannel0,
    /// Viridis-like colormap (5-point interpolation table).
    Viridis,
    /// Red-blue diverging (negative=blue, zero=white, positive=red).
    Diverging,
}

// ---------------------------------------------------------------------------
// DenoisingVizConfig
// ---------------------------------------------------------------------------

/// Configuration for denoising visualization.
#[derive(Debug, Clone)]
pub struct DenoisingVizConfig {
    /// Capture every N steps (default: 1 — capture all).
    pub capture_every_n_steps: usize,
    /// Decode every N steps (default: 5 — only decode every 5 steps).
    pub decode_every_n_steps: usize,
    /// Output image width (default: 256).
    pub image_width: usize,
    /// Output image height (default: 256).
    pub image_height: usize,
    /// Colormap used to visualize latents.
    pub colormap: LatentColormap,
    /// Total number of views being generated (default: 1). Used to
    /// populate [`DenoisingTimeline::total_views`] for every timeline
    /// [`DenoisingVisualizer::capture_step`] creates.
    pub total_views: usize,
}

impl Default for DenoisingVizConfig {
    fn default() -> Self {
        Self {
            capture_every_n_steps: 1,
            decode_every_n_steps: 5,
            image_width: 256,
            image_height: 256,
            colormap: LatentColormap::RgbNormalized,
            total_views: 1,
        }
    }
}

impl DenoisingVizConfig {
    /// Validate configuration, returning an error if any field is invalid.
    pub fn validate(&self) -> Result<(), DiffusionError> {
        if self.capture_every_n_steps == 0 {
            return Err(DiffusionError::InvalidConfig(
                "capture_every_n_steps must be >= 1".to_string(),
            ));
        }
        if self.decode_every_n_steps == 0 {
            return Err(DiffusionError::InvalidConfig(
                "decode_every_n_steps must be >= 1".to_string(),
            ));
        }
        if self.image_width == 0 {
            return Err(DiffusionError::InvalidConfig(
                "image_width must be >= 1".to_string(),
            ));
        }
        if self.image_height == 0 {
            return Err(DiffusionError::InvalidConfig(
                "image_height must be >= 1".to_string(),
            ));
        }
        if self.total_views == 0 {
            return Err(DiffusionError::InvalidConfig(
                "total_views must be >= 1".to_string(),
            ));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// DenoisingVisualizer
// ---------------------------------------------------------------------------

/// Converts latent tensors to viewable images and accumulates timelines.
pub struct DenoisingVisualizer {
    pub config: DenoisingVizConfig,
    timelines: Vec<DenoisingTimeline>,
}

/// Viridis 5-point colour table (value → [R, G, B]).
const VIRIDIS_TABLE: [(f32, [u8; 3]); 5] = [
    (0.00, [68, 1, 84]),
    (0.25, [59, 82, 139]),
    (0.50, [33, 145, 140]),
    (0.75, [94, 201, 98]),
    (1.00, [253, 231, 37]),
];

impl DenoisingVisualizer {
    /// Create a new visualizer with the given configuration.
    ///
    /// # Errors
    /// Propagates [`DenoisingVizConfig::validate`]'s error when `config` is
    /// invalid (e.g. `capture_every_n_steps == 0`, which would otherwise
    /// silently record only step 0 of every view).
    pub fn new(config: DenoisingVizConfig) -> Result<Self, DiffusionError> {
        config.validate()?;
        Ok(Self {
            config,
            timelines: Vec::new(),
        })
    }

    /// Update the configured `total_views` and back-fill it onto every
    /// timeline captured so far (as well as any created afterward).
    pub fn set_total_views(&mut self, total_views: usize) {
        self.config.total_views = total_views;
        for timeline in &mut self.timelines {
            timeline.total_views = total_views;
        }
    }

    /// Capture a step for the given view.
    ///
    /// Only records the step when `step_index % capture_every_n_steps == 0`.
    /// When the step doesn't already carry a decoded image (i.e. the caller
    /// did not attach one via [`DenoisingStep::with_decoded`]) and
    /// `step_index % decode_every_n_steps == 0`, converts the latent to an
    /// RGBA preview via [`Self::latent_to_rgba`] and rescales it to the
    /// configured `image_width`/`image_height` before storing it.
    pub fn capture_step(&mut self, view_index: usize, mut step: DenoisingStep) {
        let capture_every = self.config.capture_every_n_steps.max(1);
        if !step.step_index.is_multiple_of(capture_every) {
            return;
        }

        let decode_every = self.config.decode_every_n_steps.max(1);
        if step.decoded_image.is_none() && step.step_index.is_multiple_of(decode_every) {
            let raw = self.latent_to_rgba(
                &step.latent,
                step.latent_channels,
                step.latent_height,
                step.latent_width,
            );
            let resized = resize_rgba_nearest(
                &raw,
                step.latent_width,
                step.latent_height,
                self.config.image_width,
                self.config.image_height,
            );
            step.decoded_image = Some(resized);
            step.image_width = self.config.image_width;
            step.image_height = self.config.image_height;
        }

        // Find existing timeline for this view or create one.
        let timeline_pos = self
            .timelines
            .iter()
            .position(|t| t.view_index == view_index);
        let pos = match timeline_pos {
            Some(p) => p,
            None => {
                let new_timeline = DenoisingTimeline::new(view_index, self.config.total_views);
                self.timelines.push(new_timeline);
                self.timelines.len() - 1
            }
        };

        self.timelines[pos].push_step(step);
    }

    /// Convert a latent tensor slice to RGBA bytes using the configured colormap.
    ///
    /// The output length is `height * width * 4` (RGBA).
    pub fn latent_to_rgba(
        &self,
        latent: &[f32],
        channels: usize,
        height: usize,
        width: usize,
    ) -> Vec<u8> {
        let num_pixels = height * width;
        let mut output = vec![0u8; num_pixels * 4];

        match self.config.colormap {
            LatentColormap::RgbNormalized => {
                self.apply_rgb_normalized(latent, channels, height, width, &mut output);
            }
            LatentColormap::GrayscaleChannel0 => {
                self.apply_grayscale_ch0(latent, channels, height, width, &mut output);
            }
            LatentColormap::Viridis => {
                self.apply_viridis(latent, channels, height, width, &mut output);
            }
            LatentColormap::Diverging => {
                self.apply_diverging(latent, channels, height, width, &mut output);
            }
        }

        output
    }

    /// Get the timeline for a specific view index.
    pub fn view_timeline(&self, view_index: usize) -> Option<&DenoisingTimeline> {
        self.timelines.iter().find(|t| t.view_index == view_index)
    }

    /// Get all timelines.
    pub fn all_timelines(&self) -> &[DenoisingTimeline] {
        &self.timelines
    }

    /// Format a report of all captured timelines.
    pub fn format_report(&self) -> String {
        if self.timelines.is_empty() {
            return "No timelines captured.".to_string();
        }

        let mut lines = Vec::new();
        lines.push(format!(
            "Denoising Visualizer Report ({} views)",
            self.timelines.len()
        ));
        lines.push(format!(
            "  Config: capture_every={}, decode_every={}, {}x{}, colormap={:?}",
            self.config.capture_every_n_steps,
            self.config.decode_every_n_steps,
            self.config.image_width,
            self.config.image_height,
            self.config.colormap
        ));
        for timeline in &self.timelines {
            lines.push(format!("  {}", timeline.format_summary()));
        }

        lines.join("\n")
    }

    // ------------------------------------------------------------------
    // Private colormap helpers
    // ------------------------------------------------------------------

    fn apply_rgb_normalized(
        &self,
        latent: &[f32],
        channels: usize,
        height: usize,
        width: usize,
        output: &mut [u8],
    ) {
        let num_pixels = height * width;
        // Normalize each channel independently.
        let channel_data: Vec<Vec<f32>> = (0..channels.min(3))
            .map(|c| {
                (0..num_pixels)
                    .map(|p| {
                        let idx = c * num_pixels + p;
                        if idx < latent.len() {
                            latent[idx]
                        } else {
                            0.0
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        // Compute per-channel min/max.
        let channel_ranges: Vec<(f32, f32)> = channel_data
            .iter()
            .map(|ch| {
                let (mn, mx) = ch
                    .iter()
                    .fold((f32::INFINITY, f32::NEG_INFINITY), |(mn, mx), &v| {
                        (mn.min(v), mx.max(v))
                    });
                (mn, mx)
            })
            .collect();

        for p in 0..num_pixels {
            let r = if channels >= 1 {
                normalize_value(channel_data[0][p], channel_ranges[0].0, channel_ranges[0].1)
            } else {
                0.0
            };
            let g = if channels >= 2 {
                normalize_value(channel_data[1][p], channel_ranges[1].0, channel_ranges[1].1)
            } else {
                0.0
            };
            let b = if channels >= 3 {
                normalize_value(channel_data[2][p], channel_ranges[2].0, channel_ranges[2].1)
            } else {
                0.0
            };

            output[p * 4] = (r * 255.0) as u8;
            output[p * 4 + 1] = (g * 255.0) as u8;
            output[p * 4 + 2] = (b * 255.0) as u8;
            output[p * 4 + 3] = 255;
        }
    }

    fn apply_grayscale_ch0(
        &self,
        latent: &[f32],
        channels: usize,
        height: usize,
        width: usize,
        output: &mut [u8],
    ) {
        let num_pixels = height * width;
        if channels == 0 || latent.is_empty() {
            // Fill with black + full alpha.
            for p in 0..num_pixels {
                output[p * 4 + 3] = 255;
            }
            return;
        }

        let ch0: Vec<f32> = (0..num_pixels)
            .map(|p| if p < latent.len() { latent[p] } else { 0.0 })
            .collect();

        let (mn, mx) = ch0
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(mn, mx), &v| {
                (mn.min(v), mx.max(v))
            });

        for p in 0..num_pixels {
            let normalized = normalize_value(ch0[p], mn, mx);
            let byte = (normalized * 255.0) as u8;
            output[p * 4] = byte;
            output[p * 4 + 1] = byte;
            output[p * 4 + 2] = byte;
            output[p * 4 + 3] = 255;
        }
    }

    fn apply_viridis(
        &self,
        latent: &[f32],
        channels: usize,
        height: usize,
        width: usize,
        output: &mut [u8],
    ) {
        let num_pixels = height * width;
        if channels == 0 || latent.is_empty() {
            for p in 0..num_pixels {
                output[p * 4 + 3] = 255;
            }
            return;
        }

        let ch0: Vec<f32> = (0..num_pixels)
            .map(|p| if p < latent.len() { latent[p] } else { 0.0 })
            .collect();

        let (mn, mx) = ch0
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(mn, mx), &v| {
                (mn.min(v), mx.max(v))
            });

        for p in 0..num_pixels {
            let t = normalize_value(ch0[p], mn, mx);
            let [r, g, b] = viridis_lookup(t);
            output[p * 4] = r;
            output[p * 4 + 1] = g;
            output[p * 4 + 2] = b;
            output[p * 4 + 3] = 255;
        }
    }

    fn apply_diverging(
        &self,
        latent: &[f32],
        channels: usize,
        height: usize,
        width: usize,
        output: &mut [u8],
    ) {
        let num_pixels = height * width;
        if channels == 0 || latent.is_empty() {
            for p in 0..num_pixels {
                // White
                output[p * 4] = 255;
                output[p * 4 + 1] = 255;
                output[p * 4 + 2] = 255;
                output[p * 4 + 3] = 255;
            }
            return;
        }

        let ch0: Vec<f32> = (0..num_pixels)
            .map(|p| if p < latent.len() { latent[p] } else { 0.0 })
            .collect();

        let abs_max = ch0.iter().fold(0.0_f32, |acc, &v| acc.max(v.abs()));

        for p in 0..num_pixels {
            let normalized = if abs_max == 0.0 {
                0.0
            } else {
                ch0[p] / abs_max // [-1, 1]
            };

            let (r, g, b) = if normalized < 0.0 {
                // Negative: blue blend from white (0) to blue (255).
                let t = (-normalized).clamp(0.0, 1.0);
                let white_part = ((1.0 - t) * 255.0) as u8;
                (white_part, white_part, 255u8)
            } else {
                // Positive: red blend from white (0) to red (255).
                let t = normalized.clamp(0.0, 1.0);
                let white_part = ((1.0 - t) * 255.0) as u8;
                (255u8, white_part, white_part)
            };

            output[p * 4] = r;
            output[p * 4 + 1] = g;
            output[p * 4 + 2] = b;
            output[p * 4 + 3] = 255;
        }
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Nearest-neighbour resize of an RGBA buffer from `(src_w, src_h)` to
/// `(dst_w, dst_h)`. Returns an all-zero buffer of the destination size if
/// either the source or destination has a zero dimension.
fn resize_rgba_nearest(
    src: &[u8],
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
) -> Vec<u8> {
    let mut out = vec![0u8; dst_w * dst_h * 4];
    if src_w == 0 || src_h == 0 || dst_w == 0 || dst_h == 0 {
        return out;
    }
    for dy in 0..dst_h {
        let sy = (dy * src_h / dst_h).min(src_h - 1);
        for dx in 0..dst_w {
            let sx = (dx * src_w / dst_w).min(src_w - 1);
            let src_idx = (sy * src_w + sx) * 4;
            let dst_idx = (dy * dst_w + dx) * 4;
            out[dst_idx..dst_idx + 4].copy_from_slice(&src[src_idx..src_idx + 4]);
        }
    }
    out
}

/// Normalize a value to [0, 1].  When min == max returns 0.5.
fn normalize_value(v: f32, min_val: f32, max_val: f32) -> f32 {
    let range = max_val - min_val;
    if range == 0.0 {
        return 0.5;
    }
    ((v - min_val) / range).clamp(0.0, 1.0)
}

/// Look up a colour from the 5-point viridis table via linear interpolation.
fn viridis_lookup(t: f32) -> [u8; 3] {
    let t = t.clamp(0.0, 1.0);

    // Find surrounding control points.
    let mut lo_idx = 0usize;
    for (i, &(pos, _)) in VIRIDIS_TABLE.iter().enumerate() {
        if pos <= t {
            lo_idx = i;
        }
    }

    let hi_idx = (lo_idx + 1).min(VIRIDIS_TABLE.len() - 1);

    let (lo_t, lo_col) = VIRIDIS_TABLE[lo_idx];
    let (hi_t, hi_col) = VIRIDIS_TABLE[hi_idx];

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
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- DenoisingStep -------------------------------------------------------

    #[test]
    fn test_denoising_step_new() {
        let latent = vec![0.1, 0.2, 0.3, 0.4];
        let step = DenoisingStep::new(3, 20, 500, latent.clone(), 1, 2, 2);
        assert_eq!(step.step_index, 3);
        assert_eq!(step.total_steps, 20);
        assert_eq!(step.timestep, 500);
        assert_eq!(step.latent, latent);
        assert_eq!(step.latent_channels, 1);
        assert_eq!(step.latent_height, 2);
        assert_eq!(step.latent_width, 2);
        assert!(step.decoded_image.is_none());
        assert_eq!(step.image_width, 0);
        assert_eq!(step.image_height, 0);
    }

    #[test]
    fn test_denoising_step_progress_fraction() {
        let step = DenoisingStep::new(10, 20, 0, vec![], 1, 1, 1);
        let frac = step.progress_fraction();
        assert!((frac - 0.5).abs() < 1e-6, "expected 0.5, got {}", frac);

        let step_zero = DenoisingStep::new(0, 0, 0, vec![], 1, 1, 1);
        assert_eq!(step_zero.progress_fraction(), 0.0);

        let step_start = DenoisingStep::new(0, 20, 0, vec![], 1, 1, 1);
        assert_eq!(step_start.progress_fraction(), 0.0);

        let step_end = DenoisingStep::new(20, 20, 0, vec![], 1, 1, 1);
        assert!((step_end.progress_fraction() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_denoising_step_latent_stats() {
        let latent = vec![1.0, 2.0, 3.0, 4.0];
        let step = DenoisingStep::new(0, 1, 0, latent, 1, 2, 2);
        let (mn, mx, mean) = step.latent_stats();
        assert!((mn - 1.0).abs() < 1e-6);
        assert!((mx - 4.0).abs() < 1e-6);
        assert!((mean - 2.5).abs() < 1e-6);
    }

    #[test]
    fn test_denoising_step_latent_stats_empty() {
        let step = DenoisingStep::new(0, 1, 0, vec![], 1, 0, 0);
        let (mn, mx, mean) = step.latent_stats();
        assert_eq!(mn, 0.0);
        assert_eq!(mx, 0.0);
        assert_eq!(mean, 0.0);
    }

    #[test]
    fn test_denoising_step_with_decoded() {
        let step = DenoisingStep::new(0, 1, 0, vec![0.5], 1, 1, 1);
        let img = vec![255u8; 256 * 256 * 4];
        let step = step.with_decoded(img.clone(), 256, 256);
        assert_eq!(step.image_width, 256);
        assert_eq!(step.image_height, 256);
        assert!(step.decoded_image.is_some());
        assert_eq!(
            step.decoded_image.as_ref().map(|v| v.len()),
            Some(img.len())
        );
    }

    // -- DenoisingTimeline ---------------------------------------------------

    #[test]
    fn test_timeline_push_step() {
        let mut timeline = DenoisingTimeline::new(0, 4);
        assert_eq!(timeline.num_steps(), 0);

        let step = DenoisingStep::new(0, 20, 1000, vec![0.0; 4], 1, 2, 2);
        timeline.push_step(step);
        assert_eq!(timeline.num_steps(), 1);

        let step2 = DenoisingStep::new(1, 20, 900, vec![0.1; 4], 1, 2, 2);
        timeline.push_step(step2);
        assert_eq!(timeline.num_steps(), 2);
    }

    #[test]
    fn test_timeline_format_summary() {
        let mut timeline = DenoisingTimeline::new(0, 4);
        let step = DenoisingStep::new(0, 20, 1000, vec![0.0; 4 * 32 * 32], 4, 32, 32);
        timeline.push_step(step);
        let summary = timeline.format_summary();
        assert!(summary.contains("View 0/4"), "got: {}", summary);
        assert!(summary.contains("Steps: 1"), "got: {}", summary);
        assert!(summary.contains("4x32x32"), "got: {}", summary);
    }

    #[test]
    fn test_timeline_first_last_step() {
        let mut timeline = DenoisingTimeline::new(0, 4);
        assert!(timeline.first_step().is_none());
        assert!(timeline.last_step().is_none());

        timeline.push_step(DenoisingStep::new(0, 5, 1000, vec![], 1, 1, 1));
        timeline.push_step(DenoisingStep::new(4, 5, 200, vec![], 1, 1, 1));

        assert_eq!(timeline.first_step().map(|s| s.step_index), Some(0));
        assert_eq!(timeline.last_step().map(|s| s.step_index), Some(4));
    }

    // -- DenoisingVizConfig --------------------------------------------------

    #[test]
    fn test_viz_config_default() {
        let cfg = DenoisingVizConfig::default();
        assert_eq!(cfg.capture_every_n_steps, 1);
        assert_eq!(cfg.decode_every_n_steps, 5);
        assert_eq!(cfg.image_width, 256);
        assert_eq!(cfg.image_height, 256);
        assert_eq!(cfg.colormap, LatentColormap::RgbNormalized);
    }

    #[test]
    fn test_viz_config_validate() {
        let cfg = DenoisingVizConfig::default();
        assert!(cfg.validate().is_ok());

        let bad = DenoisingVizConfig {
            capture_every_n_steps: 0,
            ..Default::default()
        };
        assert!(bad.validate().is_err());

        let bad2 = DenoisingVizConfig {
            decode_every_n_steps: 0,
            ..Default::default()
        };
        assert!(bad2.validate().is_err());

        let bad3 = DenoisingVizConfig {
            image_width: 0,
            ..Default::default()
        };
        assert!(bad3.validate().is_err());

        let bad4 = DenoisingVizConfig {
            image_height: 0,
            ..Default::default()
        };
        assert!(bad4.validate().is_err());
    }

    // -- Colormap conversions ------------------------------------------------

    fn make_visualizer(colormap: LatentColormap) -> DenoisingVisualizer {
        let cfg = DenoisingVizConfig {
            colormap,
            ..Default::default()
        };
        DenoisingVisualizer::new(cfg).expect("valid config in test helper")
    }

    #[test]
    fn test_latent_to_rgba_rgb_normalized() {
        let viz = make_visualizer(LatentColormap::RgbNormalized);
        // 3 channels, 1x1 pixel — values [0.0, 0.5, 1.0]
        let latent = vec![0.0f32, 0.5, 1.0];
        let rgba = viz.latent_to_rgba(&latent, 3, 1, 1);
        assert_eq!(rgba.len(), 4);
        // With min==max per channel the value is 0.5 → ~127 or if range >0:
        // ch0: only val 0.0, range=0 → normalized to 0.5 → 127
        // ch1: only val 0.5, range=0 → normalized to 0.5 → 127
        // ch2: only val 1.0, range=0 → normalized to 0.5 → 127
        assert_eq!(rgba[3], 255); // alpha always 255
    }

    #[test]
    fn test_latent_to_rgba_rgb_normalized_multi_pixel() {
        let viz = make_visualizer(LatentColormap::RgbNormalized);
        // 1 channel, 1x2 pixels — values [0.0, 1.0] normalized to [0, 255]
        let latent = vec![0.0f32, 1.0];
        let rgba = viz.latent_to_rgba(&latent, 1, 1, 2);
        assert_eq!(rgba.len(), 8);
        assert_eq!(rgba[0], 0); // pixel 0 R = 0
        assert_eq!(rgba[4], 255); // pixel 1 R = 255
        assert_eq!(rgba[3], 255); // pixel 0 alpha
        assert_eq!(rgba[7], 255); // pixel 1 alpha
    }

    #[test]
    fn test_latent_to_rgba_grayscale() {
        let viz = make_visualizer(LatentColormap::GrayscaleChannel0);
        let latent = vec![0.0f32, 1.0]; // 1 channel, 1x2
        let rgba = viz.latent_to_rgba(&latent, 1, 1, 2);
        assert_eq!(rgba.len(), 8);
        // pixel 0 → black
        assert_eq!(rgba[0], 0);
        assert_eq!(rgba[1], 0);
        assert_eq!(rgba[2], 0);
        assert_eq!(rgba[3], 255);
        // pixel 1 → white
        assert_eq!(rgba[4], 255);
        assert_eq!(rgba[5], 255);
        assert_eq!(rgba[6], 255);
        assert_eq!(rgba[7], 255);
    }

    #[test]
    fn test_latent_to_rgba_viridis() {
        let viz = make_visualizer(LatentColormap::Viridis);
        // 2 pixels: min and max
        let latent = vec![0.0f32, 1.0];
        let rgba = viz.latent_to_rgba(&latent, 1, 1, 2);
        assert_eq!(rgba.len(), 8);
        // pixel 0 → viridis at t=0.0 → [68, 1, 84]
        assert_eq!(rgba[0], 68);
        assert_eq!(rgba[1], 1);
        assert_eq!(rgba[2], 84);
        assert_eq!(rgba[3], 255);
        // pixel 1 → viridis at t=1.0 → [253, 231, 37]
        assert_eq!(rgba[4], 253);
        assert_eq!(rgba[5], 231);
        assert_eq!(rgba[6], 37);
        assert_eq!(rgba[7], 255);
    }

    #[test]
    fn test_latent_to_rgba_diverging() {
        let viz = make_visualizer(LatentColormap::Diverging);
        // 3 pixels: negative, zero, positive
        let latent = vec![-1.0f32, 0.0, 1.0];
        let rgba = viz.latent_to_rgba(&latent, 1, 1, 3);
        assert_eq!(rgba.len(), 12);
        // pixel 0 (most negative) → blue: R=white_part, G=white_part, B=255
        // t = 1.0, white_part = (1-1)*255 = 0 → (0, 0, 255)
        assert_eq!(rgba[0], 0);
        assert_eq!(rgba[1], 0);
        assert_eq!(rgba[2], 255);
        assert_eq!(rgba[3], 255);
        // pixel 1 (zero) → white: (255, 255, 255)
        assert_eq!(rgba[4], 255);
        assert_eq!(rgba[5], 255);
        assert_eq!(rgba[6], 255);
        assert_eq!(rgba[7], 255);
        // pixel 2 (most positive) → red: (255, 0, 0)
        assert_eq!(rgba[8], 255);
        assert_eq!(rgba[9], 0);
        assert_eq!(rgba[10], 0);
        assert_eq!(rgba[11], 255);
    }

    // -- capture_step filtering ----------------------------------------------

    #[test]
    fn test_capture_step_filtering() {
        let cfg = DenoisingVizConfig {
            capture_every_n_steps: 3,
            ..Default::default()
        };
        let mut viz = DenoisingVisualizer::new(cfg).expect("valid config");

        // Steps 0, 1, 2, 3, 4, 5 — only 0 and 3 are multiples of 3.
        for i in 0..6usize {
            let step = DenoisingStep::new(i, 6, 0, vec![], 1, 1, 1);
            viz.capture_step(0, step);
        }

        let timeline = viz.view_timeline(0);
        assert!(timeline.is_some());
        let timeline = timeline.expect("timeline should exist");
        assert_eq!(timeline.num_steps(), 2);
        assert_eq!(timeline.steps[0].step_index, 0);
        assert_eq!(timeline.steps[1].step_index, 3);
    }

    #[test]
    fn test_visualizer_new_rejects_invalid_config() {
        let cfg = DenoisingVizConfig {
            capture_every_n_steps: 0,
            ..Default::default()
        };
        assert!(DenoisingVisualizer::new(cfg).is_err());
    }

    // -- view_timeline -------------------------------------------------------

    #[test]
    fn test_visualizer_view_timeline() {
        let mut viz =
            DenoisingVisualizer::new(DenoisingVizConfig::default()).expect("valid config");

        viz.capture_step(0, DenoisingStep::new(0, 5, 0, vec![], 1, 1, 1));
        viz.capture_step(2, DenoisingStep::new(0, 5, 0, vec![], 1, 1, 1));

        assert!(viz.view_timeline(0).is_some());
        assert!(viz.view_timeline(1).is_none());
        assert!(viz.view_timeline(2).is_some());

        assert_eq!(viz.all_timelines().len(), 2);
    }

    // -- format_report -------------------------------------------------------

    #[test]
    fn test_format_report() {
        let mut viz =
            DenoisingVisualizer::new(DenoisingVizConfig::default()).expect("valid config");

        // Empty visualizer.
        let report = viz.format_report();
        assert!(report.contains("No timelines"), "got: {}", report);

        // With one timeline.
        viz.capture_step(0, DenoisingStep::new(0, 10, 1000, vec![0.0; 16], 4, 2, 2));
        let report = viz.format_report();
        assert!(
            report.contains("Denoising Visualizer Report"),
            "got: {}",
            report
        );
        assert!(report.contains("View 0"), "got: {}", report);
    }

    // -- total_views wiring ---------------------------------------------------

    #[test]
    fn test_capture_step_uses_configured_total_views() {
        let cfg = DenoisingVizConfig {
            total_views: 4,
            ..Default::default()
        };
        let mut viz = DenoisingVisualizer::new(cfg).expect("valid config");

        // Capture views out of order and starting from a nonzero index —
        // the old code fabricated total_views = view_index + 1 per
        // timeline, giving inconsistent values across views.
        viz.capture_step(2, DenoisingStep::new(0, 5, 0, vec![], 1, 1, 1));
        viz.capture_step(0, DenoisingStep::new(0, 5, 0, vec![], 1, 1, 1));

        let t0 = viz.view_timeline(0).expect("view 0 timeline");
        let t2 = viz.view_timeline(2).expect("view 2 timeline");
        assert_eq!(t0.total_views, 4);
        assert_eq!(t2.total_views, 4);
    }

    #[test]
    fn test_set_total_views_backfills_existing_timelines() {
        let mut viz =
            DenoisingVisualizer::new(DenoisingVizConfig::default()).expect("valid config");
        viz.capture_step(0, DenoisingStep::new(0, 5, 0, vec![], 1, 1, 1));
        assert_eq!(viz.view_timeline(0).unwrap().total_views, 1);

        viz.set_total_views(6);
        assert_eq!(viz.view_timeline(0).unwrap().total_views, 6);

        // New timelines created afterward should also pick up the new value.
        viz.capture_step(1, DenoisingStep::new(0, 5, 0, vec![], 1, 1, 1));
        assert_eq!(viz.view_timeline(1).unwrap().total_views, 6);
    }

    // -- auto-decode / resize wiring -------------------------------------------

    #[test]
    fn test_capture_step_auto_decodes_at_configured_interval() {
        let cfg = DenoisingVizConfig {
            capture_every_n_steps: 1,
            decode_every_n_steps: 2,
            image_width: 8,
            image_height: 8,
            ..Default::default()
        };
        let mut viz = DenoisingVisualizer::new(cfg).expect("valid config");

        for i in 0..4usize {
            let latent = vec![0.25_f32; 3 * 2 * 2]; // 3 channels, 2x2
            let step = DenoisingStep::new(i, 4, 0, latent, 3, 2, 2);
            viz.capture_step(0, step);
        }

        let timeline = viz.view_timeline(0).expect("timeline");
        assert_eq!(timeline.steps.len(), 4);
        for step in &timeline.steps {
            if step.step_index.is_multiple_of(2) {
                let img = step
                    .decoded_image
                    .as_ref()
                    .expect("should have been auto-decoded");
                assert_eq!(img.len(), 8 * 8 * 4);
                assert_eq!(step.image_width, 8);
                assert_eq!(step.image_height, 8);
            } else {
                assert!(
                    step.decoded_image.is_none(),
                    "step {} should not be decoded (not a multiple of decode_every_n_steps)",
                    step.step_index
                );
            }
        }
    }

    #[test]
    fn test_capture_step_does_not_overwrite_existing_decoded_image() {
        let cfg = DenoisingVizConfig {
            decode_every_n_steps: 1,
            ..Default::default()
        };
        let mut viz = DenoisingVisualizer::new(cfg).expect("valid config");
        let custom_img = vec![42u8; 4 * 4 * 4];
        let step = DenoisingStep::new(0, 1, 0, vec![0.1; 4], 1, 2, 2).with_decoded(
            custom_img.clone(),
            4,
            4,
        );
        viz.capture_step(0, step);

        let timeline = viz.view_timeline(0).expect("timeline");
        assert_eq!(timeline.steps[0].decoded_image, Some(custom_img));
        assert_eq!(timeline.steps[0].image_width, 4);
        assert_eq!(timeline.steps[0].image_height, 4);
    }
}
