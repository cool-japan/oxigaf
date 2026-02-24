//! Camera-pose conditioning MLP.
//!
//! Embeds the flattened 4×3 extrinsics matrix (12 floats) into the same
//! dimension as the U-Net time embedding so it can be added to the timestep
//! conditioning signal.

use candle_core::{Result, Tensor};
use candle_nn as nn;
use candle_nn::Module;

/// MLP that lifts a flat camera-pose vector to the time-embedding dimension.
#[derive(Debug)]
pub struct CameraEmbedding {
    linear1: nn::Linear,
    linear2: nn::Linear,
}

impl CameraEmbedding {
    /// Create a new camera embedding MLP.
    ///
    /// - `pose_dim`: input dimension (typically 12 for a flattened 4×3 matrix).
    /// - `embed_dim`: output dimension (should match the time-embedding dim).
    pub fn new(vs: nn::VarBuilder, pose_dim: usize, embed_dim: usize) -> Result<Self> {
        let linear1 = nn::linear(pose_dim, embed_dim, vs.pp("linear1"))?;
        let linear2 = nn::linear(embed_dim, embed_dim, vs.pp("linear2"))?;
        Ok(Self { linear1, linear2 })
    }
}

impl Module for CameraEmbedding {
    fn forward(&self, pose: &Tensor) -> Result<Tensor> {
        let h = self.linear1.forward(pose)?.silu()?;
        self.linear2.forward(&h)
    }
}

/// Build sinusoidal timestep embeddings (same as Stable Diffusion).
///
/// - `timesteps`: 1-D tensor of shape `(B,)` containing integer timesteps.
/// - `dim`: embedding dimension (must be even).
pub fn timestep_embedding(timesteps: &Tensor, dim: usize) -> Result<Tensor> {
    let half = dim / 2;
    let device = timesteps.device();
    let dtype = candle_core::DType::F32;

    // freq = exp(-ln(10000) * i / half)  for i in 0..half
    let exponent = (Tensor::arange(0u32, half as u32, device)?.to_dtype(dtype)?
        * (-f64::ln(10_000.0) / half as f64))?;
    let freqs = exponent.exp()?;

    // timesteps might be float already; ensure f32
    let ts = timesteps.to_dtype(dtype)?.unsqueeze(1)?; // (B, 1)
    let args = ts.broadcast_mul(&freqs.unsqueeze(0)?)?; // (B, half)

    let cos = args.cos()?;
    let sin = args.sin()?;
    Tensor::cat(&[cos, sin], 1) // (B, dim)
}

/// Timestep-embedding MLP (projects sinusoidal embeddings to a wider space).
#[derive(Debug)]
pub struct TimestepEmbedding {
    linear1: nn::Linear,
    linear2: nn::Linear,
}

impl TimestepEmbedding {
    pub fn new(vs: nn::VarBuilder, in_dim: usize, out_dim: usize) -> Result<Self> {
        let linear1 = nn::linear(in_dim, out_dim, vs.pp("linear_1"))?;
        let linear2 = nn::linear(out_dim, out_dim, vs.pp("linear_2"))?;
        Ok(Self { linear1, linear2 })
    }
}

impl Module for TimestepEmbedding {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let h = self.linear1.forward(xs)?.silu()?;
        self.linear2.forward(&h)
    }
}
