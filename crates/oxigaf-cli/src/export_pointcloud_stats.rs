//! Summary statistics for point cloud exports.
//!
//! [`PointCloudStats`] is computed from a [`GaussianModel`] after export and
//! provides per-export metadata useful in integration tests and tooling.
//!
//! This module is only included in the library compilation unit (`lib.rs`),
//! not in the binary's module tree, so the dead-code lint does not fire when
//! the type is only used from test code.

use oxigaf::render::gaussian::GaussianModel;

use crate::cli::PointColorMode;

// ---------------------------------------------------------------------------
// Sigmoid helper (duplicated from export_pointcloud to avoid cross-module dep)
// ---------------------------------------------------------------------------

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0_f32 / (1.0_f32 + (-x).exp())
}

// ---------------------------------------------------------------------------
// PointCloudStats
// ---------------------------------------------------------------------------

/// Summary statistics about a point cloud export.
#[derive(Debug, Clone)]
pub struct PointCloudStats {
    /// Total number of points exported.
    pub num_points: usize,
    /// Axis-aligned bounding box minimum corner.
    pub bbox_min: [f32; 3],
    /// Axis-aligned bounding box maximum corner.
    pub bbox_max: [f32; 3],
    /// Mean sigmoid(opacity) across all Gaussians.
    pub mean_opacity: f32,
    /// Color mode used when exporting.
    pub color_mode: PointColorMode,
}

impl PointCloudStats {
    /// Compute statistics from a [`GaussianModel`].
    #[must_use]
    pub fn compute(model: &GaussianModel, color_mode: PointColorMode) -> Self {
        let n = model.len();
        if n == 0 {
            return Self {
                num_points: 0,
                bbox_min: [0.0; 3],
                bbox_max: [0.0; 3],
                mean_opacity: 0.0,
                color_mode,
            };
        }

        let mut bbox_min = [f32::INFINITY; 3];
        let mut bbox_max = [f32::NEG_INFINITY; 3];
        let mut opacity_sum = 0.0_f32;

        for g in &model.gaussians {
            for axis in 0..3 {
                if g.position[axis] < bbox_min[axis] {
                    bbox_min[axis] = g.position[axis];
                }
                if g.position[axis] > bbox_max[axis] {
                    bbox_max[axis] = g.position[axis];
                }
            }
            opacity_sum += sigmoid(g.opacity);
        }

        let mean_opacity = opacity_sum / n as f32;

        Self {
            num_points: n,
            bbox_min,
            bbox_max,
            mean_opacity,
            color_mode,
        }
    }

    /// Format a human-readable summary string.
    #[must_use]
    pub fn format_summary(&self) -> String {
        let mode_name = match self.color_mode {
            PointColorMode::ShDc => "SH DC",
            PointColorMode::White => "White",
            PointColorMode::Opacity => "Opacity",
            PointColorMode::Scale => "Scale",
        };
        format!(
            "Point cloud: {} points | Color: {} | AABB [{:.3},{:.3},{:.3}] \
             to [{:.3},{:.3},{:.3}] | Mean opacity: {:.4}",
            self.num_points,
            mode_name,
            self.bbox_min[0],
            self.bbox_min[1],
            self.bbox_min[2],
            self.bbox_max[0],
            self.bbox_max[1],
            self.bbox_max[2],
            self.mean_opacity,
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use oxigaf::render::gaussian::{GaussianAttributes, GaussianModel};

    fn make_model(n: usize, sh_degree: u32) -> GaussianModel {
        let sh_channels = ((sh_degree + 1).pow(2) * 3) as usize;
        let gaussians: Vec<GaussianAttributes> = (0..n)
            .map(|i| {
                let f = i as f32 * 0.1;
                GaussianAttributes {
                    position: [f, f + 0.1, f + 0.2],
                    _pad0: 0.0,
                    rotation: [0.0, 0.0, 0.0, 1.0],
                    scale: [0.01_f32.ln(), 0.01_f32.ln(), 0.01_f32.ln()],
                    opacity: 0.0,
                }
            })
            .collect();
        let sh_coeffs = vec![0.0_f32; n * sh_channels];
        GaussianModel {
            gaussians,
            sh_coeffs,
            sh_degree,
            face_indices: vec![0u32; n],
            barycentric: vec![[1.0_f32 / 3.0; 3]; n],
            local_offsets: vec![[0.0_f32; 3]; n],
            is_rigid: vec![true; n],
        }
    }

    #[test]
    fn test_point_cloud_stats_num_points() {
        let model = make_model(42, 1);
        let stats = PointCloudStats::compute(&model, PointColorMode::ShDc);
        assert_eq!(
            stats.num_points, 42,
            "stats.num_points must match model size"
        );
    }

    #[test]
    fn test_point_cloud_stats_format_summary_non_empty() {
        let model = make_model(10, 0);
        let stats = PointCloudStats::compute(&model, PointColorMode::White);
        let summary = stats.format_summary();
        assert!(
            !summary.is_empty(),
            "format_summary must return a non-empty string"
        );
    }

    #[test]
    fn test_point_cloud_stats_empty_model() {
        let model = make_model(0, 1);
        let stats = PointCloudStats::compute(&model, PointColorMode::ShDc);
        assert_eq!(stats.num_points, 0);
        assert_eq!(stats.mean_opacity, 0.0);
    }

    #[test]
    fn test_point_cloud_stats_mean_opacity_sigmoid_zero() {
        // opacity=0.0 → sigmoid(0)=0.5 → mean_opacity should be 0.5
        let model = make_model(4, 1);
        let stats = PointCloudStats::compute(&model, PointColorMode::Opacity);
        assert!(
            (stats.mean_opacity - 0.5_f32).abs() < 1e-5,
            "mean_opacity should be 0.5 for zero-opacity gaussians, got {}",
            stats.mean_opacity
        );
    }
}
