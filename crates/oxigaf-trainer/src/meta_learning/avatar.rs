//! A [`MetaModel`] over a real Gaussian avatar.
//!
//! [`crate::meta_learning`] ships MAML-style inner/outer loops that work for any
//! [`MetaModel`], but the only implementation was [`LinearModel`] — a toy
//! regressor.  Nothing connected them to the thing this crate actually trains.
//!
//! [`GaussianAvatarModel`](crate::meta_learning_avatar::GaussianAvatarModel)
//! closes that gap: it presents an
//! [`oxigaf_render::gaussian::GaussianModel`] as the flat `&[f32]` parameter
//! vector the trait wants, and implements
//! [`loss_and_grad`](MetaModel::loss_and_grad) by running the **real**
//! rasterizer forward/backward pass and flattening
//! [`crate::optimizer::Gradients`] back into that same layout.  Both meta-loops
//! ([`crate::meta_learning::inner_loop_adapt`],
//! [`crate::meta_learning::run_meta_training`]) accept it unchanged.
//!
//! ## Layout
//!
//! The flat vector concatenates the six parameter groups in
//! [`crate::optimizer::ParameterGroup`] order, so it lines up element-for-element
//! with [`crate::optimizer::Gradients::to_group_vecs`]:
//!
//! ```text
//! [ position 3N | rotation 4N | scale 3N | opacity N | sh C | offset 3N ]
//! ```
//!
//! [`ParamLayout`](crate::meta_learning_avatar::ParamLayout) is the single
//! description of that packing; both
//! [`GaussianAvatarModel::flatten_model`](crate::meta_learning_avatar::GaussianAvatarModel::flatten_model)
//! and
//! [`GaussianAvatarModel::flatten_gradients`](crate::meta_learning_avatar::GaussianAvatarModel::flatten_gradients)
//! go through it, so parameters and
//! their gradients can never end up in different orders.
//!
//! ## GPU requirement
//!
//! [`loss_and_grad`](MetaModel::loss_and_grad) rasterizes, so it needs a
//! `wgpu` device.  Construction therefore takes an
//! [`AvatarRenderer`](crate::meta_learning_avatar::AvatarRenderer), and the
//! renderer is shared (`Rc<RefCell<_>>`) across the copies the inner loop makes
//! via [`with_params`](MetaModel::with_params) — the trait hands out owned
//! models, and a rasterizer is neither cloneable nor cheap.
//!
//! The pure half — packing, unpacking and gradient flattening — needs no device
//! and is tested directly.

use std::cell::RefCell;
use std::rc::Rc;

use oxigaf_flame::Camera;
use oxigaf_render::gaussian::GaussianModel;
use oxigaf_render::{RasterConfig, Rasterizer};

use crate::config::LossConfig;
use crate::image_gradient::{photometric_pixel_gradient, PhotometricSpec};
use crate::loss::LossComputer;
use crate::meta_learning::{MetaLearningError, MetaModel};
use crate::optimizer::Gradients;
use crate::trainer::camera_to_render_camera;

// ---------------------------------------------------------------------------
// ParamLayout
// ---------------------------------------------------------------------------

/// Offsets and lengths of the six parameter groups inside the flat vector.
///
/// Derived once from a model's shape; every pack/unpack goes through it so a
/// parameter and its gradient cannot disagree about where a group lives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParamLayout {
    /// Number of Gaussians.
    pub num_gaussians: usize,
    /// SH coefficients **per Gaussian** (`(degree + 1)² × 3`).
    pub sh_channels: usize,
}

impl ParamLayout {
    /// Derive the layout of `model`.
    pub fn of(model: &GaussianModel) -> Self {
        let num_gaussians = model.len();
        // An empty model has no per-Gaussian SH width to derive.
        let sh_channels = model
            .sh_coeffs
            .len()
            .checked_div(num_gaussians)
            .unwrap_or(0);
        Self {
            num_gaussians,
            sh_channels,
        }
    }

    /// Total number of scalars in the flat vector.
    pub fn total(&self) -> usize {
        let n = self.num_gaussians;
        n * 3 + n * 4 + n * 3 + n + n * self.sh_channels + n * 3
    }

    /// `[start, end)` of each group, in [`crate::optimizer::ParameterGroup`]
    /// order: position, rotation, scale, opacity, SH, offset.
    pub fn spans(&self) -> [(usize, usize); 6] {
        let n = self.num_gaussians;
        let mut cursor = 0;
        let mut span = |len: usize| {
            let start = cursor;
            cursor += len;
            (start, cursor)
        };
        [
            span(n * 3),
            span(n * 4),
            span(n * 3),
            span(n),
            span(n * self.sh_channels),
            span(n * 3),
        ]
    }
}

// ---------------------------------------------------------------------------
// AvatarBatch
// ---------------------------------------------------------------------------

/// One support or query set: observed views paired with the cameras that saw
/// them.
///
/// `targets[i]` is a flat HWC RGB buffer of `width · height · 3` values in
/// `[0, 1]` seen from `cameras[i]`.  Surplus entries on either side are ignored
/// so a caller cannot index out of bounds by mispairing them.
#[derive(Debug, Clone, Default)]
pub struct AvatarBatch {
    /// Ground-truth views, one flat HWC RGB buffer each.
    pub targets: Vec<Vec<f32>>,
    /// Camera for each view, paired by index.
    pub cameras: Vec<Camera>,
}

impl AvatarBatch {
    /// Build a batch from paired views and cameras.
    pub fn new(targets: Vec<Vec<f32>>, cameras: Vec<Camera>) -> Self {
        Self { targets, cameras }
    }

    /// Number of usable (target, camera) pairs.
    pub fn len(&self) -> usize {
        self.targets.len().min(self.cameras.len())
    }

    /// Whether there is nothing to evaluate.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// ---------------------------------------------------------------------------
// AvatarRenderer
// ---------------------------------------------------------------------------

/// The GPU side of [`GaussianAvatarModel`]: a rasterizer plus the objective it
/// differentiates.
///
/// Kept separate from the model so the flat-vector plumbing stays device-free
/// and testable, and so the (expensive, non-cloneable) rasterizer can be shared
/// across the model copies the meta-learning inner loop produces.
pub struct AvatarRenderer {
    rasterizer: Rasterizer,
    raster_config: RasterConfig,
    loss_computer: LossComputer,
}

impl AvatarRenderer {
    /// Wrap an existing rasterizer and the objective to optimise.
    ///
    /// `raster_config` must be the one `rasterizer` was built with: it supplies
    /// the image dimensions every target buffer is interpreted at.
    pub fn new(
        rasterizer: Rasterizer,
        raster_config: RasterConfig,
        loss_config: LossConfig,
    ) -> Self {
        Self {
            rasterizer,
            raster_config,
            loss_computer: LossComputer::new(loss_config),
        }
    }

    /// Number of scalars in one view's flat HWC RGB buffer.
    pub fn view_len(&self) -> usize {
        (self.raster_config.image_width as usize) * (self.raster_config.image_height as usize) * 3
    }

    /// Share this renderer with the model copies the inner loop creates.
    pub fn shared(self) -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(self))
    }
}

// ---------------------------------------------------------------------------
// GaussianAvatarModel
// ---------------------------------------------------------------------------

/// A Gaussian avatar presented as a flat-parameter [`MetaModel`].
pub struct GaussianAvatarModel {
    /// Flat parameter vector — the storage [`MetaModel::params`] hands out.
    params: Vec<f32>,
    /// Topology and bindings the flat vector is re-inflated into: face indices,
    /// barycentrics, rigidity flags and `sh_degree` are *not* learnable, so they
    /// live here rather than in the parameter vector.
    template: GaussianModel,
    /// Packing description of `params`.
    layout: ParamLayout,
    /// Shared GPU renderer, or `None` for a device-free model.
    renderer: Option<Rc<RefCell<AvatarRenderer>>>,
}

impl GaussianAvatarModel {
    /// Wrap `model`, rendering through `renderer`.
    pub fn new(model: GaussianModel, renderer: Rc<RefCell<AvatarRenderer>>) -> Self {
        let layout = ParamLayout::of(&model);
        let params = Self::flatten_model(&model, &layout);
        Self {
            params,
            template: model,
            layout,
            renderer: Some(renderer),
        }
    }

    /// Wrap `model` **without** a renderer.
    ///
    /// Packing, unpacking and gradient flattening all work; only
    /// [`loss_and_grad`](MetaModel::loss_and_grad) requires a device, and it
    /// reports [`MetaLearningError::GradientError`] naming the missing renderer
    /// rather than fabricating a loss.
    pub fn without_renderer(model: GaussianModel) -> Self {
        let layout = ParamLayout::of(&model);
        let params = Self::flatten_model(&model, &layout);
        Self {
            params,
            template: model,
            layout,
            renderer: None,
        }
    }

    /// The packing description of this model's flat parameter vector.
    pub fn layout(&self) -> ParamLayout {
        self.layout
    }

    /// Rebuild a [`GaussianModel`] from the current flat parameters.
    pub fn to_gaussian_model(&self) -> GaussianModel {
        let mut model = self.template.clone();
        Self::unflatten_into(&self.params, &self.layout, &mut model);
        model
    }

    /// Pack a model's learnable parameters into a flat vector.
    ///
    /// Group order matches [`crate::optimizer::Gradients::to_group_vecs`], which
    /// is what lets [`flatten_gradients`](Self::flatten_gradients) line up.
    pub fn flatten_model(model: &GaussianModel, layout: &ParamLayout) -> Vec<f32> {
        let mut flat = Vec::with_capacity(layout.total());
        for g in model.gaussians.iter() {
            flat.extend_from_slice(&g.position);
        }
        for g in model.gaussians.iter() {
            flat.extend_from_slice(&g.rotation);
        }
        for g in model.gaussians.iter() {
            flat.extend_from_slice(&g.scale);
        }
        for g in model.gaussians.iter() {
            flat.push(g.opacity);
        }
        flat.extend_from_slice(&model.sh_coeffs);
        for offset in model.local_offsets.iter() {
            flat.extend_from_slice(offset);
        }
        flat
    }

    /// Write a flat vector back into `model`'s learnable parameters.
    ///
    /// A short `flat` leaves the remaining parameters untouched instead of
    /// panicking: every read is bounds-checked and skipped when absent.
    pub fn unflatten_into(flat: &[f32], layout: &ParamLayout, model: &mut GaussianModel) {
        let spans = layout.spans();
        let read = |start: usize, len: usize| -> Option<&[f32]> { flat.get(start..start + len) };

        if let Some(src) = read(spans[0].0, layout.num_gaussians * 3) {
            for (g, chunk) in model.gaussians.iter_mut().zip(src.chunks_exact(3)) {
                g.position.copy_from_slice(chunk);
            }
        }
        if let Some(src) = read(spans[1].0, layout.num_gaussians * 4) {
            for (g, chunk) in model.gaussians.iter_mut().zip(src.chunks_exact(4)) {
                g.rotation.copy_from_slice(chunk);
            }
        }
        if let Some(src) = read(spans[2].0, layout.num_gaussians * 3) {
            for (g, chunk) in model.gaussians.iter_mut().zip(src.chunks_exact(3)) {
                g.scale.copy_from_slice(chunk);
            }
        }
        if let Some(src) = read(spans[3].0, layout.num_gaussians) {
            for (g, &value) in model.gaussians.iter_mut().zip(src.iter()) {
                g.opacity = value;
            }
        }
        if let Some(src) = read(spans[4].0, layout.num_gaussians * layout.sh_channels) {
            let len = model.sh_coeffs.len().min(src.len());
            model.sh_coeffs[..len].copy_from_slice(&src[..len]);
        }
        if let Some(src) = read(spans[5].0, layout.num_gaussians * 3) {
            for (offset, chunk) in model.local_offsets.iter_mut().zip(src.chunks_exact(3)) {
                offset.copy_from_slice(chunk);
            }
        }
    }

    /// Flatten [`Gradients`] into the same layout as the parameter vector.
    ///
    /// This is the piece that makes the meta-loops usable at all: they add
    /// `params − lr · grad` element-wise, so the two vectors must agree on
    /// element order exactly.
    pub fn flatten_gradients(gradients: &Gradients, layout: &ParamLayout) -> Vec<f32> {
        let mut flat = Vec::with_capacity(layout.total());
        flat.extend_from_slice(&gradients.position);
        flat.extend_from_slice(&gradients.rotation);
        flat.extend_from_slice(&gradients.scale);
        flat.extend_from_slice(&gradients.opacity);
        flat.extend_from_slice(&gradients.sh);
        flat.extend_from_slice(&gradients.offset);
        flat.resize(layout.total(), 0.0);
        flat
    }
}

impl MetaModel for GaussianAvatarModel {
    type Batch = AvatarBatch;

    fn params(&self) -> &[f32] {
        &self.params
    }

    fn params_mut(&mut self) -> &mut [f32] {
        &mut self.params
    }

    fn with_params(&self, params: Vec<f32>) -> Result<Self, MetaLearningError> {
        if params.len() != self.params.len() {
            return Err(MetaLearningError::DimensionMismatch {
                expected: self.params.len(),
                actual: params.len(),
            });
        }
        Ok(Self {
            params,
            template: self.template.clone(),
            layout: self.layout,
            renderer: self.renderer.clone(),
        })
    }

    /// Rasterize every view in `batch`, score it against the observed target,
    /// and return the loss together with the flat parameter gradient.
    ///
    /// The gradient is the analytic derivative of the configured photometric
    /// objective (see [`crate::image_gradient`]) pushed through
    /// `Rasterizer::backward`, averaged over views exactly as
    /// [`crate::loss::LossComputer::compute`] averages the loss.
    ///
    /// # Errors
    ///
    /// * [`MetaLearningError::GradientError`] — no renderer was supplied
    ///   ([`GaussianAvatarModel::without_renderer`]), the renderer is already
    ///   borrowed, or the rasterizer failed.  A render failure must not be
    ///   replaced by a fabricated image: the caller would step on it.
    /// * [`MetaLearningError::EmptyTaskBatch`] — the batch pairs no views.
    /// * [`MetaLearningError::DimensionMismatch`] — a target does not hold
    ///   `width · height · 3` values.
    fn loss_and_grad(&self, batch: &Self::Batch) -> Result<(f32, Vec<f32>), MetaLearningError> {
        let renderer = self.renderer.as_ref().ok_or_else(|| {
            MetaLearningError::GradientError(
                "GaussianAvatarModel has no AvatarRenderer: loss_and_grad needs a wgpu device \
                 (build it with GaussianAvatarModel::new)"
                    .into(),
            )
        })?;
        let num_views = batch.len();
        if num_views == 0 {
            return Err(MetaLearningError::EmptyTaskBatch);
        }

        let mut renderer = renderer.try_borrow_mut().map_err(|_| {
            MetaLearningError::GradientError(
                "AvatarRenderer is already mutably borrowed — loss_and_grad cannot be \
                 re-entered on the same renderer"
                    .into(),
            )
        })?;

        let expected_len = renderer.view_len();
        for (idx, target) in batch.targets.iter().take(num_views).enumerate() {
            if target.len() != expected_len {
                tracing::error!(
                    view = idx,
                    expected = expected_len,
                    actual = target.len(),
                    "AvatarBatch target does not match the rasterizer resolution"
                );
                return Err(MetaLearningError::DimensionMismatch {
                    expected: expected_len,
                    actual: target.len(),
                });
            }
        }

        let model = self.to_gaussian_model();
        let width = renderer.raster_config.image_width;
        let height = renderer.raster_config.image_height;
        let w = width as usize;
        let h = height as usize;
        let npx = w * h;

        renderer.rasterizer.upload_gaussians(&model);

        // Render every view first: the loss is a mean over views, so the
        // image-space gradient needs the same `1/V` normalisation the reported
        // loss uses, and that is only known once all views exist.
        let mut rendered = Vec::with_capacity(num_views);
        for (idx, camera) in batch.cameras.iter().take(num_views).enumerate() {
            let render_camera = camera_to_render_camera(camera, width, height);
            let output = renderer
                .rasterizer
                .forward(&model, &render_camera)
                .map_err(|e| {
                    MetaLearningError::GradientError(format!("forward pass on view {idx}: {e}"))
                })?;
            let mut rgb = vec![0.0_f32; npx * 3];
            for (dst, src) in rgb
                .chunks_exact_mut(3)
                .zip(output.color_data.chunks_exact(4))
            {
                dst.copy_from_slice(&src[..3]);
            }
            rendered.push(rgb);
        }

        let targets: Vec<Vec<f32>> = batch.targets.iter().take(num_views).cloned().collect();
        let loss_output = renderer
            .loss_computer
            .compute(&rendered, &targets, w, h, &model, None);

        let loss_config = renderer.loss_computer.config().clone();
        let ssim_kernel = renderer.loss_computer.ssim_kernel().to_vec();
        let ms_ssim_weights = *renderer.loss_computer.ms_ssim_weights();
        let spec = PhotometricSpec {
            config: &loss_config,
            ssim_kernel: &ssim_kernel,
            ms_ssim_weights: &ms_ssim_weights,
            num_views,
        };

        let mut accumulated = Gradients::zeros(self.layout.num_gaussians, self.layout.sh_channels);
        for (idx, (render, target)) in rendered.iter().zip(targets.iter()).enumerate() {
            let grad_rgb = photometric_pixel_gradient(&spec, render, target, w, h);
            let mut grad_image = vec![0.0_f32; npx * 4];
            for (dst, src) in grad_image.chunks_exact_mut(4).zip(grad_rgb.chunks_exact(3)) {
                dst[..3].copy_from_slice(src);
            }
            let gpu = renderer
                .rasterizer
                .backward(&model, &grad_image)
                .map_err(|e| {
                    MetaLearningError::GradientError(format!("backward pass on view {idx}: {e}"))
                })?;
            add_flat(&mut accumulated.position, &gpu.grad_positions);
            add_flat(&mut accumulated.rotation, &gpu.grad_rotations);
            add_flat(&mut accumulated.scale, &gpu.grad_scales);
            add_slice(&mut accumulated.opacity, &gpu.grad_opacities);
            add_slice(&mut accumulated.sh, &gpu.grad_sh_coeffs);
        }

        let flat = Self::flatten_gradients(&accumulated, &self.layout);
        Ok((loss_output.total, flat))
    }
}

/// Accumulate a slice of fixed-size per-Gaussian tuples into a flat buffer.
fn add_flat<const N: usize>(dst: &mut [f32], src: &[[f32; N]]) {
    for (chunk, values) in dst.chunks_exact_mut(N).zip(src.iter()) {
        for (d, v) in chunk.iter_mut().zip(values.iter()) {
            *d += v;
        }
    }
}

/// Accumulate a flat slice into a flat buffer (`dst[i] += src[i]`).
fn add_slice(dst: &mut [f32], src: &[f32]) {
    for (d, &s) in dst.iter_mut().zip(src.iter()) {
        *d += s;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use oxigaf_render::gaussian::GaussianAttributes;

    fn sample_model(n: usize) -> GaussianModel {
        let gaussians = (0..n)
            .map(|i| {
                let f = i as f32;
                GaussianAttributes {
                    position: [f, f + 0.1, f + 0.2],
                    _pad0: 0.0,
                    rotation: [0.0, 0.1 * f, 0.0, 1.0],
                    scale: [-1.0 - f, -2.0 - f, -3.0 - f],
                    opacity: 0.5 + f,
                }
            })
            .collect();
        GaussianModel {
            gaussians,
            sh_coeffs: (0..n * 3).map(|i| i as f32 * 0.01).collect(),
            sh_degree: 0,
            face_indices: vec![0; n],
            barycentric: vec![[1.0, 0.0, 0.0]; n],
            local_offsets: (0..n).map(|i| [i as f32, -1.0, 2.0]).collect(),
            is_rigid: vec![false; n],
        }
    }

    #[test]
    fn layout_covers_every_learnable_parameter_exactly_once() {
        let model = sample_model(4);
        let layout = ParamLayout::of(&model);
        assert_eq!(layout.num_gaussians, 4);
        assert_eq!(layout.sh_channels, 3);
        // 3 + 4 + 3 + 1 + 3 (sh) + 3 = 17 scalars per Gaussian.
        assert_eq!(layout.total(), 4 * 17);

        let spans = layout.spans();
        assert_eq!(spans[0].0, 0);
        assert_eq!(spans[5].1, layout.total());
        for window in spans.windows(2) {
            assert_eq!(window[0].1, window[1].0, "spans must tile with no gap");
        }
    }

    #[test]
    fn flatten_and_unflatten_round_trip() {
        let model = sample_model(3);
        let avatar = GaussianAvatarModel::without_renderer(model.clone());
        assert_eq!(avatar.params().len(), avatar.layout().total());

        let restored = avatar.to_gaussian_model();
        for (a, b) in model.gaussians.iter().zip(restored.gaussians.iter()) {
            assert_eq!(a.position, b.position);
            assert_eq!(a.rotation, b.rotation);
            assert_eq!(a.scale, b.scale);
            assert_eq!(a.opacity, b.opacity);
        }
        assert_eq!(model.sh_coeffs, restored.sh_coeffs);
        assert_eq!(model.local_offsets, restored.local_offsets);
        // Non-learnable structure survives untouched.
        assert_eq!(model.face_indices, restored.face_indices);
        assert_eq!(model.sh_degree, restored.sh_degree);
    }

    #[test]
    fn with_params_replaces_the_vector_and_checks_its_length() {
        let avatar = GaussianAvatarModel::without_renderer(sample_model(2));
        let mut updated: Vec<f32> = avatar.params().to_vec();
        updated[0] = 99.0;
        let replaced = avatar
            .with_params(updated)
            .expect("same length is accepted");
        assert_eq!(replaced.params()[0], 99.0);
        assert_eq!(replaced.to_gaussian_model().gaussians[0].position[0], 99.0);
        // The original is untouched: the trait hands out owned models.
        assert_eq!(avatar.params()[0], 0.0);

        assert!(avatar.with_params(vec![0.0; 3]).is_err());
    }

    #[test]
    fn gradients_flatten_into_the_same_layout_as_the_parameters() {
        // Regression: the whole point of this adapter is that the meta-loops
        // compute `params − lr · grad` element-wise, so a gradient laid out in
        // a different order would corrupt every parameter group.
        let model = sample_model(2);
        let layout = ParamLayout::of(&model);
        let mut gradients = Gradients::zeros(layout.num_gaussians, layout.sh_channels);
        // Mark one element in each group with a distinguishable value.
        gradients.position[0] = 1.0;
        gradients.rotation[0] = 2.0;
        gradients.scale[0] = 3.0;
        gradients.opacity[0] = 4.0;
        gradients.sh[0] = 5.0;
        gradients.offset[0] = 6.0;

        let flat = GaussianAvatarModel::flatten_gradients(&gradients, &layout);
        assert_eq!(flat.len(), layout.total());
        let spans = layout.spans();
        for (group, (start, _)) in [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0].iter().zip(spans.iter()) {
            assert_eq!(flat[*start], *group, "group starting at {start} misplaced");
        }
    }

    #[test]
    fn loss_and_grad_reports_the_missing_renderer_instead_of_faking_a_loss() {
        let avatar = GaussianAvatarModel::without_renderer(sample_model(2));
        let batch = AvatarBatch::new(vec![vec![0.0; 12]], vec![Camera::default_front(2, 2)]);
        let err = avatar
            .loss_and_grad(&batch)
            .expect_err("no renderer must be an error");
        assert!(
            format!("{err}").contains("wgpu device"),
            "the error should name what is missing: {err}"
        );
    }

    #[test]
    fn empty_batches_are_rejected() {
        let avatar = GaussianAvatarModel::without_renderer(sample_model(1));
        assert!(AvatarBatch::default().is_empty());
        // The empty-batch check runs before the renderer check only when a
        // renderer exists; without one the missing-device error is returned
        // first, which is still an error rather than a fabricated zero loss.
        assert!(avatar.loss_and_grad(&AvatarBatch::default()).is_err());

        // Mispaired targets/cameras count as the shorter side.
        let batch = AvatarBatch::new(vec![vec![0.0; 3]; 5], vec![Camera::default_front(1, 1); 2]);
        assert_eq!(batch.len(), 2);
    }
}
