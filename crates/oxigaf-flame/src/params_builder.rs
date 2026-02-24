//! Builder pattern for `FlameParams`.

use crate::params::FlameParams;

/// Builder for constructing `FlameParams` with a fluent API.
///
/// # Example
///
/// ```rust
/// use oxigaf_flame::FlameParams;
///
/// let params = FlameParams::builder()
///     .shape(vec![0.5, -0.3, 0.2, 0.1])
///     .expression(vec![0.8, 0.5, -0.4])
///     .root_rotation([0.1, 0.0, 0.0])
///     .jaw_rotation(0.15)
///     .translation([0.0, 0.05, 0.0])
///     .build();
/// ```
#[derive(Debug, Clone, Default)]
pub struct FlameParamsBuilder {
    shape: Vec<f32>,
    expression: Vec<f32>,
    pose: Vec<f32>,
    translation: [f32; 3],
}

impl FlameParamsBuilder {
    /// Set shape parameters (identity coefficients).
    ///
    /// Typically 100-300 coefficients, each in range [-3.0, 3.0].
    #[must_use]
    pub fn shape(mut self, shape: Vec<f32>) -> Self {
        self.shape = shape;
        self
    }

    /// Set expression parameters.
    ///
    /// Typically 50-100 coefficients, each in range [-2.0, 2.0].
    #[must_use]
    pub fn expression(mut self, expression: Vec<f32>) -> Self {
        self.expression = expression;
        self
    }

    /// Set full pose parameters (15 values for 5 joints).
    #[must_use]
    pub fn pose(mut self, pose: Vec<f32>) -> Self {
        self.pose = pose;
        self
    }

    /// Set root (global head) rotation as axis-angle.
    ///
    /// # Arguments
    /// * `rotation` - [rx, ry, rz] axis-angle vector in radians
    #[must_use]
    pub fn root_rotation(mut self, rotation: [f32; 3]) -> Self {
        self.ensure_pose_size();
        self.pose[0..3].copy_from_slice(&rotation);
        self
    }

    /// Set neck rotation as axis-angle.
    #[must_use]
    pub fn neck_rotation(mut self, rotation: [f32; 3]) -> Self {
        self.ensure_pose_size();
        self.pose[3..6].copy_from_slice(&rotation);
        self
    }

    /// Set jaw rotation (single value for jaw opening).
    ///
    /// Typically in range [-0.5, 0.2] radians.
    /// Positive values open the jaw.
    #[must_use]
    pub fn jaw_rotation(mut self, angle: f32) -> Self {
        self.ensure_pose_size();
        self.pose[6] = angle;
        // Y and Z components typically remain zero for jaw
        self
    }

    /// Set jaw rotation as full axis-angle.
    #[must_use]
    pub fn jaw_rotation_full(mut self, rotation: [f32; 3]) -> Self {
        self.ensure_pose_size();
        self.pose[6..9].copy_from_slice(&rotation);
        self
    }

    /// Set left eye rotation as axis-angle.
    #[must_use]
    pub fn left_eye_rotation(mut self, rotation: [f32; 3]) -> Self {
        self.ensure_pose_size();
        self.pose[9..12].copy_from_slice(&rotation);
        self
    }

    /// Set right eye rotation as axis-angle.
    #[must_use]
    pub fn right_eye_rotation(mut self, rotation: [f32; 3]) -> Self {
        self.ensure_pose_size();
        self.pose[12..15].copy_from_slice(&rotation);
        self
    }

    /// Set global translation.
    ///
    /// In meters, typically in range [-1.0, 1.0] per axis.
    #[must_use]
    pub fn translation(mut self, translation: [f32; 3]) -> Self {
        self.translation = translation;
        self
    }

    /// Build the final `FlameParams`.
    #[must_use]
    pub fn build(self) -> FlameParams {
        let pose = if self.pose.is_empty() {
            vec![0.0; FlameParams::NUM_JOINTS * 3]
        } else {
            self.pose
        };

        FlameParams {
            shape: self.shape,
            expression: self.expression,
            pose,
            translation: self.translation,
        }
    }

    /// Ensure pose vector has correct size.
    fn ensure_pose_size(&mut self) {
        if self.pose.len() < FlameParams::NUM_JOINTS * 3 {
            self.pose.resize(FlameParams::NUM_JOINTS * 3, 0.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_builder_empty() {
        let params = FlameParams::builder().build();
        assert_eq!(params.shape.len(), 0);
        assert_eq!(params.expression.len(), 0);
        assert_eq!(params.pose.len(), 15);
        assert_relative_eq!(params.translation[0], 0.0);
        assert_relative_eq!(params.translation[1], 0.0);
        assert_relative_eq!(params.translation[2], 0.0);
    }

    #[test]
    fn test_builder_with_shape() {
        let params = FlameParams::builder().shape(vec![0.5, -0.3]).build();

        assert_eq!(params.shape, vec![0.5, -0.3]);
    }

    #[test]
    fn test_builder_jaw_rotation() {
        let params = FlameParams::builder().jaw_rotation(0.15).build();

        assert!((params.pose[6] - 0.15).abs() < 1e-6);
        assert!((params.pose[7]).abs() < 1e-6);
        assert!((params.pose[8]).abs() < 1e-6);
    }

    #[test]
    fn test_builder_full() {
        let params = FlameParams::builder()
            .shape(vec![0.1, 0.2])
            .expression(vec![0.5])
            .root_rotation([0.1, 0.0, 0.0])
            .jaw_rotation(0.2)
            .translation([0.0, 0.1, 0.0])
            .build();

        assert_eq!(params.shape, vec![0.1, 0.2]);
        assert_eq!(params.expression, vec![0.5]);
        assert!((params.pose[0] - 0.1).abs() < 1e-6);
        assert!((params.pose[6] - 0.2).abs() < 1e-6);
        assert_relative_eq!(params.translation[0], 0.0);
        assert_relative_eq!(params.translation[1], 0.1);
        assert_relative_eq!(params.translation[2], 0.0);
    }

    #[test]
    fn test_validate_success() {
        let params = FlameParams::builder()
            .shape(vec![1.0, -1.5, 0.5])
            .expression(vec![0.8, -0.5])
            .jaw_rotation(0.15)
            .build();

        assert!(params.validate());
    }

    #[test]
    fn test_validate_shape_out_of_range() {
        let params = FlameParams::builder()
            .shape(vec![5.0]) // Out of range
            .build();

        assert!(!params.validate());
    }
}
