//! FLAME model parameters for a single frame / pose.
//!
//! ## Parameter Ranges
//!
//! FLAME parameters typically lie in specific ranges for realistic results:
//!
//! - **Shape**: Each coefficient typically in range `[-3.0, 3.0]` (3 standard deviations)
//! - **Expression**: Each coefficient typically in range `[-2.0, 2.0]`
//! - **Pose** (axis-angle): Each component typically in range `[-pi, pi]` radians
//!   - Root rotation: Full 3D rotation of the head
//!   - Neck: Limited range for natural neck movement
//!   - Jaw: Typically `[-0.5, 0.2]` radians (opening range)
//!   - Eyes: Typically `[-0.3, 0.3]` radians per axis
//! - **Translation**: In meters, typically `[-1.0, 1.0]` for each axis
//!
//! ## Example
//!
//! ```rust
//! use oxigaf_flame::FlameParams;
//!
//! // Neutral face (zero deformation)
//! let neutral = FlameParams::neutral();
//!
//! // Smiling face with slight head tilt
//! let smiling = FlameParams {
//!     shape: vec![0.0; 10],  // Use neutral identity shape
//!     expression: vec![0.5, 0.3, -0.2],  // Smile expression
//!     pose: vec![0.1, 0.0, 0.0,  // Slight head tilt
//!                0.0, 0.0, 0.0,  // No neck rotation
//!                0.1, 0.0, 0.0,  // Slight jaw opening
//!                0.0, 0.0, 0.0,  // Neutral left eye
//!                0.0, 0.0, 0.0], // Neutral right eye
//!     translation: [0.0, 0.0, 0.0],
//! };
//! ```

use crate::params_builder::FlameParamsBuilder;
use serde::{Deserialize, Serialize};

/// FLAME model parameters for a single frame.
///
/// All vectors can be shorter than the maximum; missing trailing coefficients
/// are treated as zero during the forward pass.
///
/// See module-level documentation for parameter ranges and examples.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlameParams {
    /// Shape (identity) blend-shape coefficients. Up to 300, typically 100.
    pub shape: Vec<f32>,

    /// Expression blend-shape coefficients. Up to 100, typically 50.
    pub expression: Vec<f32>,

    /// Joint pose as axis-angle vectors, concatenated.
    /// 5 joints x 3 = 15 values.
    /// Order: `[root(3), neck(3), jaw(3), left_eye(3), right_eye(3)]`
    pub pose: Vec<f32>,

    /// Global translation applied after posing.
    pub translation: [f32; 3],
}

impl FlameParams {
    /// Number of FLAME joints.
    pub const NUM_JOINTS: usize = 5;

    /// Create a neutral (zero) parameter set.
    #[must_use]
    pub fn neutral() -> Self {
        Self {
            shape: Vec::new(),
            expression: Vec::new(),
            pose: vec![0.0; Self::NUM_JOINTS * 3],
            translation: [0.0; 3],
        }
    }

    /// Start building `FlameParams` with a builder pattern.
    ///
    /// # Example
    ///
    /// ```rust
    /// use oxigaf_flame::FlameParams;
    ///
    /// let params = FlameParams::builder()
    ///     .shape(vec![0.5, -0.3, 0.2])
    ///     .expression(vec![0.8, 0.5])
    ///     .jaw_rotation(0.1)
    ///     .translation([0.0, 0.1, 0.0])
    ///     .build();
    /// ```
    #[must_use]
    pub fn builder() -> FlameParamsBuilder {
        FlameParamsBuilder::default()
    }

    /// Return the axis-angle triple for joint `j` (0-indexed), or zeros if
    /// the pose vector is too short.
    #[must_use]
    pub fn joint_pose(&self, j: usize) -> [f32; 3] {
        let off = j * 3;
        if off + 2 < self.pose.len() {
            [self.pose[off], self.pose[off + 1], self.pose[off + 2]]
        } else {
            [0.0; 3]
        }
    }

    /// Validate that parameters are within typical ranges.
    ///
    /// Returns `true` if all parameters are within reasonable bounds:
    /// - Shape: [-3.0, 3.0]
    /// - Expression: [-2.0, 2.0]
    /// - Pose: [-pi, pi]
    /// - Translation: [-1.0, 1.0]
    #[must_use]
    pub fn validate(&self) -> bool {
        use std::f32::consts::PI;

        // Check shape coefficients
        if self.shape.iter().any(|&s| s.abs() > 3.0) {
            return false;
        }

        // Check expression coefficients
        if self.expression.iter().any(|&e| e.abs() > 2.0) {
            return false;
        }

        // Check pose angles
        if self.pose.iter().any(|&p| p.abs() > PI) {
            return false;
        }

        // Check translation
        if self.translation.iter().any(|&t| t.abs() > 1.0) {
            return false;
        }

        true
    }
}

impl Default for FlameParams {
    fn default() -> Self {
        Self::neutral()
    }
}
