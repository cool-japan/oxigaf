//! Screen-space Level-of-Detail (LoD) filtering for 3D Gaussian Splatting.
//!
//! This module provides tools to compute per-Gaussian screen-space projected sizes
//! and use them to fade or cull Gaussians that are too small to contribute
//! meaningfully to the rendered image.
//!
//! ## Overview
//!
//! The pipeline:
//! 1. [`LodCamera`] computes per-Gaussian depths and screen-space radii.
//! 2. [`opacity_scale_from_radius`] maps a screen radius to an opacity multiplier,
//!    linearly ramping from 0 (sub-pixel) to 1 (full-size).
//! 3. [`LodFilter`] drives the full process — compute LoD info, optionally apply it
//!    by adjusting stored opacity values and removing culled Gaussians.
//!
//! ## Opacity storage
//!
//! Gaussian opacity is stored as a **logit** (`opacity_logit = ln(p / (1-p))`).
//! When an opacity scale factor `s ∈ [0, 1]` is applied the stored value becomes
//! `logit(sigmoid(opacity_logit) * s)`.  The helpers `sigmoid` and `logit`
//! inside this module implement these conversions without any `unwrap`.

use crate::{GaussianAttributes, GaussianModel};

// ─────────────────────────────────────────────────────────────────────────────
// Private math helpers (mirrors density.rs private helpers)
// ─────────────────────────────────────────────────────────────────────────────

/// Numerically stable sigmoid: `1 / (1 + exp(-x))`.
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0_f32 / (1.0_f32 + (-x).exp())
}

/// Logit (inverse sigmoid): `ln(p / (1 - p))`, clamped to avoid ±∞.
#[inline]
fn logit(p: f32) -> f32 {
    let p = p.clamp(1e-7_f32, 1.0_f32 - 1e-7_f32);
    (p / (1.0_f32 - p)).ln()
}

// ─────────────────────────────────────────────────────────────────────────────
// LodCamera
// ─────────────────────────────────────────────────────────────────────────────

/// Camera parameters used for LoD size computation.
///
/// The camera looks along `view_dir` (a unit vector).  Depth is measured as the
/// signed projection of the vector from the camera to a point onto `view_dir`.
#[derive(Debug, Clone)]
pub struct LodCamera {
    /// Camera position in world space.
    pub position: [f32; 3],
    /// View direction (should be a unit vector).
    pub view_dir: [f32; 3],
    /// Vertical field-of-view in radians.
    pub fov_y: f32,
    /// Image height in pixels.
    pub image_height: u32,
    /// Image width in pixels.
    pub image_width: u32,
    /// Near clip plane distance (world units).
    pub near: f32,
    /// Far clip plane distance (world units, 0 = no far clip).
    pub far: f32,
}

impl LodCamera {
    /// Construct a camera with default near (0.01) and far (100.0) clip planes.
    pub fn new(
        position: [f32; 3],
        view_dir: [f32; 3],
        fov_y: f32,
        width: u32,
        height: u32,
    ) -> Self {
        Self {
            position,
            view_dir,
            fov_y,
            image_width: width,
            image_height: height,
            near: 0.01,
            far: 100.0,
        }
    }

    /// Pixels per world-unit at a given depth from the camera.
    ///
    /// Returns 0.0 for non-positive depth (behind or at the camera).
    pub fn pixels_per_unit(&self, depth: f32) -> f32 {
        if depth <= 0.0 {
            return 0.0;
        }
        let half_height = self.image_height as f32 / 2.0;
        let half_tan = (self.fov_y / 2.0).tan();
        if half_tan <= 0.0 {
            return 0.0;
        }
        half_height / (depth * half_tan)
    }

    /// Pixels per world-unit at a given depth, computed from the
    /// *horizontal* field of view derived from `fov_y` and the image
    /// aspect ratio (`image_width / image_height`).
    ///
    /// For a standard square-pixel perspective camera this equals
    /// [`LodCamera::pixels_per_unit`] exactly (the two can diverge only if
    /// `image_width`/`image_height` describe a non-square-pixel aspect on
    /// purpose). Returns `0.0` for non-positive depth or a degenerate
    /// (zero-height) image.
    pub fn pixels_per_unit_horizontal(&self, depth: f32) -> f32 {
        if depth <= 0.0 || self.image_height == 0 {
            return 0.0;
        }
        let half_tan_y = (self.fov_y / 2.0).tan();
        if half_tan_y <= 0.0 {
            return 0.0;
        }
        let aspect = self.image_width as f32 / self.image_height as f32;
        let half_tan_x = half_tan_y * aspect;
        if half_tan_x <= 0.0 {
            return 0.0;
        }
        let half_width = self.image_width as f32 / 2.0;
        half_width / (depth * half_tan_x)
    }

    /// Signed depth of a world-space point (projection onto `view_dir`).
    ///
    /// Positive = in front of the camera.
    pub fn depth_of(&self, point: [f32; 3]) -> f32 {
        let dx = point[0] - self.position[0];
        let dy = point[1] - self.position[1];
        let dz = point[2] - self.position[2];
        dx * self.view_dir[0] + dy * self.view_dir[1] + dz * self.view_dir[2]
    }

    /// Approximate screen-space radius of a Gaussian in pixels.
    ///
    /// Uses `max(exp(scale[i]))` as the world-space radius of the Gaussian
    /// ellipsoid and projects it using the perspective scale at the Gaussian's
    /// depth.
    ///
    /// Returns 0.0 for Gaussians at or behind the near clip plane.
    pub fn screen_radius(&self, gaussian: &GaussianAttributes) -> f32 {
        let depth = self.depth_of(gaussian.position);
        if depth <= self.near {
            return 0.0;
        }
        let world_radius = gaussian
            .scale
            .iter()
            .copied()
            .map(f32::exp)
            .fold(f32::NEG_INFINITY, f32::max);
        world_radius * self.pixels_per_unit(depth)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// LodConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for LoD-based filtering and opacity ramping.
#[derive(Debug, Clone)]
pub struct LodConfig {
    /// Gaussians smaller than this many pixels are faded out.
    ///
    /// Default: `1.0` (fade sub-pixel Gaussians).
    pub min_pixel_radius: f32,

    /// Gaussians with screen radius ≥ this value get full (unmodified) opacity.
    ///
    /// Between `min_pixel_radius * cull_threshold_fraction` and
    /// `full_opacity_radius` the opacity is linearly ramped.
    ///
    /// Default: `2.0`.
    pub full_opacity_radius: f32,

    /// Fraction of `min_pixel_radius` below which a Gaussian is fully culled
    /// (opacity set to 0 and removed from the model on [`LodFilter::apply`]).
    ///
    /// Default: `0.1` (cull Gaussians with radius < 0.1 px).
    pub cull_threshold_fraction: f32,

    /// Maximum depth before culling (world units).
    ///
    /// `0.0` disables depth-based culling.
    pub max_depth: f32,

    /// Caller-facing hint for which entry point to prefer: when `true`,
    /// callers should use [`LodFilter::apply_mut`] (filters `model` in
    /// place); when `false`, [`LodFilter::apply`] (returns a new model,
    /// leaving the input untouched). Both methods work regardless of this
    /// flag's value — it does not change `LodFilter`'s own behaviour.
    pub modify_in_place: bool,
}

impl Default for LodConfig {
    fn default() -> Self {
        Self {
            min_pixel_radius: 1.0,
            full_opacity_radius: 2.0,
            cull_threshold_fraction: 0.1,
            max_depth: 0.0,
            modify_in_place: false,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GaussianLodInfo
// ─────────────────────────────────────────────────────────────────────────────

/// Per-Gaussian LoD computation result.
#[derive(Debug, Clone)]
pub struct GaussianLodInfo {
    /// Screen-space radius in pixels.
    pub screen_radius_px: f32,
    /// Depth from the camera (projection onto view direction).
    pub depth: f32,
    /// Opacity multiplier in `[0, 1]` applied to the Gaussian's stored opacity.
    pub opacity_scale: f32,
    /// Whether this Gaussian should be removed from the scene entirely.
    pub is_culled: bool,
}

// ─────────────────────────────────────────────────────────────────────────────
// opacity_scale_from_radius (free function)
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the opacity multiplier for a Gaussian with the given screen radius.
///
/// Returns `(scale, is_culled)`:
/// - `radius < cull_radius` → `(0.0, true)` — Gaussian should be removed.
/// - `radius >= full_opacity_radius` → `(1.0, false)` — full opacity.
/// - Otherwise → linear ramp in `(0.0, 1.0)`, `is_culled = false`.
///
/// The cull threshold is `config.min_pixel_radius * config.cull_threshold_fraction`.
pub fn opacity_scale_from_radius(radius: f32, config: &LodConfig) -> (f32, bool) {
    let cull_radius = config.min_pixel_radius * config.cull_threshold_fraction;
    if radius < cull_radius {
        return (0.0, true);
    }
    if radius >= config.full_opacity_radius {
        return (1.0, false);
    }
    let range = config.full_opacity_radius - cull_radius;
    if range <= 0.0 {
        // Degenerate config: full_opacity_radius ≤ cull_radius.
        return (1.0, false);
    }
    let t = (radius - cull_radius) / range;
    (t.clamp(0.0, 1.0), false)
}

// ─────────────────────────────────────────────────────────────────────────────
// LodFilter
// ─────────────────────────────────────────────────────────────────────────────

/// Computes and applies screen-space Level-of-Detail filtering to a
/// [`GaussianModel`].
pub struct LodFilter {
    /// LoD configuration.
    pub config: LodConfig,
}

impl LodFilter {
    /// Create a new `LodFilter` with the given configuration.
    pub fn new(config: LodConfig) -> Self {
        Self { config }
    }

    /// Compute LoD info for every Gaussian in `model`.
    ///
    /// The returned `Vec` has the same length as `model.gaussians`.
    pub fn compute(&self, model: &GaussianModel, camera: &LodCamera) -> Vec<GaussianLodInfo> {
        model
            .gaussians
            .iter()
            .map(|g| {
                let depth = camera.depth_of(g.position);
                let screen_radius_px = camera.screen_radius(g);

                // Depth-based culling.
                let depth_culled = self.config.max_depth > 0.0 && depth > self.config.max_depth;

                let (opacity_scale, radius_culled) =
                    opacity_scale_from_radius(screen_radius_px, &self.config);

                let is_culled = depth_culled || radius_culled;
                let opacity_scale = if depth_culled { 0.0 } else { opacity_scale };

                GaussianLodInfo {
                    screen_radius_px,
                    depth,
                    opacity_scale,
                    is_culled,
                }
            })
            .collect()
    }

    /// Apply LoD filtering: returns a new [`GaussianModel`] with:
    ///
    /// - Culled Gaussians **removed**.
    /// - Remaining Gaussians' opacity adjusted in logit space by the
    ///   `opacity_scale` factor.
    ///
    /// All parallel data vectors (`sh_coeffs`, `face_indices`, `barycentric`,
    /// `local_offsets`, `is_rigid`) are sliced/filtered consistently.
    pub fn apply(&self, model: &GaussianModel, camera: &LodCamera) -> GaussianModel {
        let lod_info = self.compute(model, camera);

        // Number of SH coefficients per Gaussian.
        let sh_c = ((model.sh_degree + 1) * (model.sh_degree + 1) * 3) as usize;

        // Capacity hints.
        let kept_count = lod_info.iter().filter(|info| !info.is_culled).count();
        let mut gaussians = Vec::with_capacity(kept_count);
        let mut sh_coeffs = Vec::with_capacity(kept_count * sh_c);
        let mut face_indices = Vec::with_capacity(kept_count);
        let mut barycentric = Vec::with_capacity(kept_count);
        let mut local_offsets = Vec::with_capacity(kept_count);
        let mut is_rigid = Vec::with_capacity(kept_count);

        for (i, info) in lod_info.iter().enumerate() {
            if info.is_culled {
                continue;
            }

            // Adjust opacity in logit space.
            let mut g = model.gaussians[i];
            if info.opacity_scale < 1.0 {
                let p = sigmoid(g.opacity);
                let new_p = (p * info.opacity_scale).clamp(1e-7_f32, 1.0_f32 - 1e-7_f32);
                g.opacity = logit(new_p);
            }
            gaussians.push(g);

            // SH coefficients slice for Gaussian i.
            let sh_start = i * sh_c;
            let sh_end = sh_start + sh_c;
            if sh_end <= model.sh_coeffs.len() {
                sh_coeffs.extend_from_slice(&model.sh_coeffs[sh_start..sh_end]);
            } else {
                // Pad with zeros if sh_coeffs is shorter (e.g. empty model).
                sh_coeffs.extend(std::iter::repeat_n(0.0_f32, sh_c));
            }

            // FLAME binding (same-length vecs; guard for missing data).
            face_indices.push(model.face_indices.get(i).copied().unwrap_or(0));
            barycentric.push(model.barycentric.get(i).copied().unwrap_or([
                1.0 / 3.0,
                1.0 / 3.0,
                1.0 / 3.0,
            ]));
            local_offsets.push(model.local_offsets.get(i).copied().unwrap_or([0.0; 3]));
            is_rigid.push(model.is_rigid.get(i).copied().unwrap_or(false));
        }

        GaussianModel {
            gaussians,
            sh_coeffs,
            sh_degree: model.sh_degree,
            face_indices,
            barycentric,
            local_offsets,
            is_rigid,
        }
    }

    /// In-place counterpart to [`LodFilter::apply`].
    ///
    /// Filters `model` according to this filter's configuration and
    /// overwrites `*model` with the result, avoiding a separate owned
    /// return value at the call site. This is the entry point
    /// [`LodConfig::modify_in_place`] refers to.
    ///
    /// Returns the number of Gaussians removed.
    pub fn apply_mut(&self, model: &mut GaussianModel, camera: &LodCamera) -> usize {
        let filtered = self.apply(model, camera);
        let culled = model
            .gaussians
            .len()
            .saturating_sub(filtered.gaussians.len());
        *model = filtered;
        culled
    }

    /// Count the number of Gaussians that would be culled by this filter.
    pub fn count_culled(&self, model: &GaussianModel, camera: &LodCamera) -> usize {
        self.compute(model, camera)
            .iter()
            .filter(|info| info.is_culled)
            .count()
    }

    /// Estimate the memory (bytes) freed by removing culled Gaussians.
    ///
    /// Accounts for:
    /// - `GaussianAttributes` struct (48 bytes each: position+`_pad0` 16B,
    ///   rotation 16B, scale 12B, opacity 4B)
    /// - SH coefficients: `(sh_degree+1)^2 * 3 * 4` bytes per Gaussian
    /// - `face_indices`: 4 bytes
    /// - `barycentric`: 12 bytes (3 × f32)
    /// - `local_offsets`: 12 bytes (3 × f32)
    /// - `is_rigid`: 1 byte
    pub fn memory_saved_bytes(&self, model: &GaussianModel, camera: &LodCamera) -> usize {
        let n_culled = self.count_culled(model, camera);
        let sh_c = ((model.sh_degree + 1) * (model.sh_degree + 1) * 3) as usize;
        let bytes_per_gaussian = std::mem::size_of::<GaussianAttributes>()
            + sh_c * std::mem::size_of::<f32>()
            + std::mem::size_of::<u32>()  // face_index
            + 3 * std::mem::size_of::<f32>()  // barycentric
            + 3 * std::mem::size_of::<f32>()  // local_offset
            + std::mem::size_of::<bool>(); // is_rigid
        n_culled * bytes_per_gaussian
    }

    /// Generate a human-readable statistics string for the LoD result.
    ///
    /// Includes total count, culled count, kept count, and the fraction culled.
    pub fn stats_string(&self, model: &GaussianModel, camera: &LodCamera) -> String {
        let lod_info = self.compute(model, camera);
        let total = lod_info.len();
        let culled = lod_info.iter().filter(|i| i.is_culled).count();
        let kept = total - culled;
        let pct = if total > 0 {
            culled as f64 / total as f64 * 100.0
        } else {
            0.0
        };
        let mem_bytes = self.memory_saved_bytes(model, camera);

        format!(
            "LodFilter stats: total={total}, kept={kept}, culled={culled} ({pct:.1}%), \
             memory_saved={mem_bytes}B, \
             config=(min_px={:.2}, full_px={:.2}, cull_frac={:.3}, max_depth={:.1})",
            self.config.min_pixel_radius,
            self.config.full_opacity_radius,
            self.config.cull_threshold_fraction,
            self.config.max_depth,
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Adaptive LOD management: LodLevel, AdaptiveLodConfig, LodSelector,
//                          LodTransition, LodStats, LodManager, LodError
// ─────────────────────────────────────────────────────────────────────────────

// ── LodError ─────────────────────────────────────────────────────────────────

/// Errors produced by the adaptive LOD management system.
#[derive(Debug)]
pub enum LodError {
    /// The levels list is empty.
    EmptyLevels,
    /// Levels are not sorted by `max_distance` in ascending order.
    LevelsNotSorted {
        /// Index of the out-of-order level.
        idx: usize,
        /// Distance of the preceding level.
        dist_a: f32,
        /// Distance of the level at `idx`.
        dist_b: f32,
    },
    /// Other configuration error.
    InvalidConfig(String),
}

impl std::fmt::Display for LodError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyLevels => write!(f, "LOD levels list must not be empty"),
            Self::LevelsNotSorted {
                idx,
                dist_a,
                dist_b,
            } => write!(
                f,
                "LOD levels must be sorted by max_distance; level[{idx}] has \
                 dist={dist_b} which is ≤ preceding dist={dist_a}"
            ),
            Self::InvalidConfig(msg) => write!(f, "Invalid LOD config: {msg}"),
        }
    }
}

impl std::error::Error for LodError {}

// ── LodLevel ──────────────────────────────────────────────────────────────────

/// One level in the adaptive LOD hierarchy.
///
/// Levels are sorted by `max_distance` ascending — level 0 is the highest
/// detail, used closest to the camera.
#[derive(Debug, Clone)]
pub struct LodLevel {
    /// Identifier for this LOD level (0 = highest detail).
    pub level_index: usize,
    /// Target number of Gaussians rendered at this LOD.
    pub target_gaussian_count: usize,
    /// Maximum camera distance at which this LOD is used.
    /// Use `f32::INFINITY` for the lowest (coarsest) LOD.
    pub max_distance: f32,
    /// Pixel coverage threshold: if the projected Gaussian diameter is below
    /// this threshold the system switches to the next (coarser) LOD.
    pub min_pixel_coverage: f32,
    /// Quality factor in `[0.0, 1.0]` for frame-budget decisions
    /// (1.0 = full quality, 0.5 = half quality).
    pub quality_factor: f32,
}

impl LodLevel {
    /// Return `true` when `distance` falls within this level's operating range.
    #[inline]
    pub fn is_applicable_at_distance(&self, distance: f32) -> bool {
        distance <= self.max_distance
    }
}

// ── AdaptiveLodConfig ─────────────────────────────────────────────────────────

/// Configuration for the adaptive LOD management system.
///
/// Note: this is distinct from [`LodConfig`], which controls per-Gaussian
/// screen-space opacity ramping.  `AdaptiveLodConfig` controls which
/// *population* of Gaussians is rendered based on camera distance.
#[derive(Debug, Clone)]
pub struct AdaptiveLodConfig {
    /// Levels sorted by `max_distance` ascending (highest detail first).
    pub levels: Vec<LodLevel>,
    /// Hysteresis factor to prevent LOD flickering at transition boundaries.
    ///
    /// `0.0` = no hysteresis; `0.1` = 10% band around each threshold.
    pub hysteresis: f32,
    /// If `true`, blend between LOD levels during transitions.
    pub enable_blending: bool,
    /// Transition extent in world distance units used when `enable_blending`
    /// is `true`.
    pub blend_distance: f32,
}

impl AdaptiveLodConfig {
    /// Validate the configuration.
    ///
    /// Returns `Err` if:
    /// - `levels` is empty, or
    /// - `levels` are not sorted by `max_distance` in strictly ascending order.
    pub fn validate(&self) -> Result<(), LodError> {
        if self.levels.is_empty() {
            return Err(LodError::EmptyLevels);
        }
        for i in 1..self.levels.len() {
            let prev = self.levels[i - 1].max_distance;
            let curr = self.levels[i].max_distance;
            if curr <= prev {
                return Err(LodError::LevelsNotSorted {
                    idx: i,
                    dist_a: prev,
                    dist_b: curr,
                });
            }
        }
        // `LodManager::update` and `LodStats::record_frame` index by array
        // position, not by `level_index` — a mismatch would make transitions
        // fire spuriously and misattribute per-level frame/histogram stats.
        for (i, level) in self.levels.iter().enumerate() {
            if level.level_index != i {
                return Err(LodError::InvalidConfig(format!(
                    "levels[{i}].level_index = {} does not match its array position {i}",
                    level.level_index
                )));
            }
        }
        Ok(())
    }

    /// Build a default 4-level LOD configuration suitable for a scene with
    /// `total_gaussians` Gaussians.
    ///
    /// | Level | Fraction | Max distance |
    /// |-------|----------|--------------|
    /// | 0     | 100 %    | 2.0          |
    /// | 1     | 50 %     | 5.0          |
    /// | 2     | 25 %     | 15.0         |
    /// | 3     | 10 %     | ∞            |
    pub fn default_for_count(total_gaussians: usize) -> Self {
        Self {
            levels: vec![
                LodLevel {
                    level_index: 0,
                    target_gaussian_count: total_gaussians,
                    max_distance: 2.0,
                    min_pixel_coverage: 2.0,
                    quality_factor: 1.0,
                },
                LodLevel {
                    level_index: 1,
                    target_gaussian_count: total_gaussians / 2,
                    max_distance: 5.0,
                    min_pixel_coverage: 1.0,
                    quality_factor: 0.75,
                },
                LodLevel {
                    level_index: 2,
                    target_gaussian_count: total_gaussians / 4,
                    max_distance: 15.0,
                    min_pixel_coverage: 0.5,
                    quality_factor: 0.5,
                },
                LodLevel {
                    level_index: 3,
                    target_gaussian_count: total_gaussians / 10,
                    max_distance: f32::INFINITY,
                    min_pixel_coverage: 0.25,
                    quality_factor: 0.25,
                },
            ],
            hysteresis: 0.05,
            enable_blending: false,
            blend_distance: 0.5,
        }
    }
}

impl Default for AdaptiveLodConfig {
    fn default() -> Self {
        Self::default_for_count(100_000)
    }
}

// ── LodSelector ───────────────────────────────────────────────────────────────

/// Stateful LOD level selector with optional hysteresis.
///
/// Maintains the currently active LOD level index and applies a hysteresis
/// band around each transition threshold to avoid rapid level toggling when
/// the camera hovers near a boundary.
pub struct LodSelector {
    config: AdaptiveLodConfig,
    current_level_idx: usize,
}

impl LodSelector {
    /// Create a new `LodSelector` from the given config.
    ///
    /// Returns `Err` if `config.validate()` fails.
    pub fn new(config: AdaptiveLodConfig) -> Result<Self, LodError> {
        config.validate()?;
        Ok(Self {
            config,
            current_level_idx: 0,
        })
    }

    /// Return a reference to the underlying configuration.
    pub fn config(&self) -> &AdaptiveLodConfig {
        &self.config
    }

    /// Return the currently active LOD level.
    pub fn current_level(&self) -> &LodLevel {
        // Safety: validate() guarantees levels is non-empty and current_level_idx
        // is always clamped to a valid index.
        &self.config.levels[self.current_level_idx]
    }

    /// Select the LOD level appropriate for `distance` with hysteresis.
    ///
    /// Hysteresis rules (h = `config.hysteresis`):
    /// - **Upgrade** (move to a finer level): only if
    ///   `distance < max_dist * (1.0 - h)`.
    /// - **Downgrade** (move to a coarser level): only if
    ///   `distance > max_dist * (1.0 + h)`.
    ///
    /// Returns a reference to the selected [`LodLevel`].
    pub fn select_by_distance(&mut self, distance: f32) -> &LodLevel {
        let h = self.config.hysteresis;
        let n = self.config.levels.len();

        // Try to upgrade (move towards finer = lower index).
        while self.current_level_idx > 0 {
            let candidate_idx = self.current_level_idx - 1;
            let threshold = self.config.levels[candidate_idx].max_distance;
            if distance < threshold * (1.0 - h) {
                self.current_level_idx = candidate_idx;
            } else {
                break;
            }
        }

        // Try to downgrade (move towards coarser = higher index).
        while self.current_level_idx < n - 1 {
            let threshold = self.config.levels[self.current_level_idx].max_distance;
            if distance > threshold * (1.0 + h) {
                self.current_level_idx += 1;
            } else {
                break;
            }
        }

        &self.config.levels[self.current_level_idx]
    }

    /// Select the LOD level based on the projected pixel coverage of a
    /// representative Gaussian.
    ///
    /// ```text
    /// pixel_coverage = focal_length * gaussian_avg_scale / distance
    /// ```
    ///
    /// Selects the finest (lowest index) level whose `min_pixel_coverage`
    /// is ≤ the computed coverage.  Falls back to the coarsest level if
    /// no level matches.
    ///
    /// Hysteresis is not applied to this selection path — the result is
    /// written into `current_level_idx` directly.
    pub fn select_by_pixel_coverage(
        &mut self,
        focal_length: f32,
        avg_scale: f32,
        distance: f32,
    ) -> &LodLevel {
        let coverage = if distance > 0.0 {
            focal_length * avg_scale / distance
        } else {
            f32::INFINITY
        };

        // Find the finest level whose min_pixel_coverage <= coverage.
        let mut chosen = self.config.levels.len() - 1;
        for (i, lvl) in self.config.levels.iter().enumerate() {
            if lvl.min_pixel_coverage <= coverage {
                chosen = i;
                break;
            }
        }

        self.current_level_idx = chosen;
        &self.config.levels[self.current_level_idx]
    }

    /// Reset the selector to the finest LOD level, clearing hysteresis state.
    ///
    /// Call this on a scene cut or after a camera teleport.
    pub fn reset(&mut self) {
        self.current_level_idx = 0;
    }

    /// Stateless distance-based level selection (ignores hysteresis).
    ///
    /// Returns the index of the finest level whose `max_distance >= distance`.
    /// If no level covers `distance`, returns the last (coarsest) level index.
    pub fn level_at_distance(config: &AdaptiveLodConfig, distance: f32) -> usize {
        for (i, lvl) in config.levels.iter().enumerate() {
            if distance <= lvl.max_distance {
                return i;
            }
        }
        config.levels.len() - 1
    }
}

// ── LodTransition ─────────────────────────────────────────────────────────────

/// Describes a blend transition between two LOD levels.
///
/// When `enable_blending` is active in [`AdaptiveLodConfig`], the renderer
/// can use `alpha` to interpolate Gaussian counts across a level boundary,
/// producing a smooth visual transition rather than a hard cut.
#[derive(Debug, Clone)]
pub struct LodTransition {
    /// Index of the level the transition starts from.
    pub from_level: usize,
    /// Index of the level the transition targets, or `None` if no transition is
    /// active.
    pub to_level: Option<usize>,
    /// Blend alpha in `[0.0, 1.0]`: `0.0` = fully `from_level`,
    /// `1.0` = fully `to_level`.
    pub alpha: f32,
    /// Whether a transition is currently in progress.
    pub is_blending: bool,
}

impl LodTransition {
    /// Create a non-blending transition anchored at `level`.
    pub fn none(level: usize) -> Self {
        Self {
            from_level: level,
            to_level: None,
            alpha: 0.0,
            is_blending: false,
        }
    }

    /// Advance the blend by `delta_distance / blend_distance`.
    ///
    /// Returns `true` when the transition completes (alpha reaches 1.0),
    /// at which point `is_blending` is set to `false` and `from_level` is
    /// updated to `to_level`.
    pub fn advance(&mut self, delta_distance: f32, blend_distance: f32) -> bool {
        if !self.is_blending {
            return false;
        }
        if blend_distance <= 0.0 {
            // Instant transition.
            if let Some(to) = self.to_level {
                self.from_level = to;
            }
            self.to_level = None;
            self.alpha = 0.0;
            self.is_blending = false;
            return true;
        }
        self.alpha = (self.alpha + delta_distance / blend_distance).clamp(0.0, 1.0);
        if self.alpha >= 1.0 {
            if let Some(to) = self.to_level {
                self.from_level = to;
            }
            self.to_level = None;
            self.alpha = 0.0;
            self.is_blending = false;
            true
        } else {
            false
        }
    }

    /// Interpolate between `from_count` and `to_count` using `alpha`.
    ///
    /// When not blending, returns `from_count`.
    pub fn blended_count(&self, from_count: usize, to_count: usize) -> usize {
        let a = self.alpha.clamp(0.0, 1.0);
        (from_count as f32 * (1.0 - a) + to_count as f32 * a) as usize
    }
}

// ── LodStats ──────────────────────────────────────────────────────────────────

/// Accumulated statistics for the adaptive LOD manager.
#[derive(Debug, Clone, Default)]
pub struct LodStats {
    /// Number of frames rendered at each LOD level.
    pub frames_at_level: Vec<u64>,
    /// Total frames recorded.
    pub total_frames: u64,
    /// Total LOD-level transitions that occurred.
    pub transitions: u64,
    /// Gaussian count during the most recently recorded frame.
    pub current_gaussian_count: usize,
}

impl LodStats {
    /// Record one rendered frame at `level_idx` with `gaussian_count` Gaussians.
    pub fn record_frame(&mut self, level_idx: usize, gaussian_count: usize) {
        // Extend the histogram if needed.
        if self.frames_at_level.len() <= level_idx {
            self.frames_at_level.resize(level_idx + 1, 0);
        }
        self.frames_at_level[level_idx] += 1;
        self.total_frames += 1;
        self.current_gaussian_count = gaussian_count;
    }

    /// Record one LOD transition event.
    pub fn record_transition(&mut self) {
        self.transitions += 1;
    }

    /// Return the fraction of total frames that were rendered at `level_idx`.
    ///
    /// Returns `0.0` if no frames have been recorded or `level_idx` is out of
    /// range.
    pub fn level_usage_fraction(&self, level_idx: usize) -> f32 {
        if self.total_frames == 0 {
            return 0.0;
        }
        let count = self.frames_at_level.get(level_idx).copied().unwrap_or(0);
        count as f32 / self.total_frames as f32
    }

    /// Format a human-readable summary of the statistics.
    pub fn format_summary(&self) -> String {
        let mut s = format!(
            "LodStats: total_frames={}, transitions={}",
            self.total_frames, self.transitions
        );
        for (i, &n) in self.frames_at_level.iter().enumerate() {
            let frac = self.level_usage_fraction(i) * 100.0;
            s.push_str(&format!(", level[{i}]={n} ({frac:.1}%)"));
        }
        s
    }
}

// ── LodManager ────────────────────────────────────────────────────────────────

/// Scale a level's `target_gaussian_count` by its `quality_factor`
/// (clamped to `[0.0, 1.0]`), rounding to the nearest Gaussian count.
#[inline]
fn quality_scaled_count(level: &LodLevel) -> usize {
    let scale = level.quality_factor.clamp(0.0, 1.0);
    ((level.target_gaussian_count as f32) * scale).round() as usize
}

/// Top-level adaptive LOD manager combining selection, transitions, and stats.
pub struct LodManager {
    selector: LodSelector,
    transition: LodTransition,
    stats: LodStats,
}

impl LodManager {
    /// Create a new `LodManager` from the given config.
    ///
    /// Returns `Err` if `config.validate()` fails.
    pub fn new(config: AdaptiveLodConfig) -> Result<Self, LodError> {
        let selector = LodSelector::new(config)?;
        let transition = LodTransition::none(0);
        Ok(Self {
            selector,
            transition,
            stats: LodStats::default(),
        })
    }

    /// Update LOD state for the current `camera_distance`.
    ///
    /// Internally selects the appropriate level, records stats, and returns
    /// the target Gaussian count for this frame.
    pub fn update(&mut self, camera_distance: f32) -> usize {
        let prev_idx = self.selector.current_level_idx;
        let level = self.selector.select_by_distance(camera_distance);
        let count = level.target_gaussian_count;
        // `level`'s borrow of `self.selector` ends after the read above, so
        // this is a fresh read of the array index `select_by_distance` just
        // wrote — NOT `level.level_index` (a user-supplied field that need
        // not match the array position; see `AdaptiveLodConfig::validate`).
        let new_idx = self.selector.current_level_idx;

        if new_idx != prev_idx {
            self.stats.record_transition();
            if self.selector.config().enable_blending {
                self.transition = LodTransition {
                    from_level: prev_idx,
                    to_level: Some(new_idx),
                    alpha: 0.0,
                    is_blending: true,
                };
            } else {
                self.transition = LodTransition::none(new_idx);
            }
        }

        self.stats.record_frame(new_idx, count);
        count
    }

    /// Return the Gaussian count for the current level, clamped to
    /// `total_gaussians` and scaled by [`LodLevel::quality_factor`] (a
    /// frame-budget throttle independent of the level's base target count).
    pub fn current_gaussian_count(&self, total_gaussians: usize) -> usize {
        let level = self.selector.current_level();
        quality_scaled_count(level).min(total_gaussians)
    }

    /// Return the (possibly blended) Gaussian count for the current frame,
    /// clamped to `total_gaussians`.
    ///
    /// While a blend transition is in progress (see [`LodManager::advance_transition`])
    /// this interpolates between the `from_level` and `to_level` quality-scaled
    /// target counts using [`LodTransition::blended_count`]; otherwise it
    /// behaves like [`LodManager::current_gaussian_count`].
    pub fn blended_gaussian_count(&self, total_gaussians: usize) -> usize {
        let count = if self.transition.is_blending {
            let levels = &self.selector.config().levels;
            let from_count = levels
                .get(self.transition.from_level)
                .map(quality_scaled_count)
                .unwrap_or(0);
            let to_count = self
                .transition
                .to_level
                .and_then(|idx| levels.get(idx))
                .map(quality_scaled_count)
                .unwrap_or(from_count);
            self.transition.blended_count(from_count, to_count)
        } else {
            quality_scaled_count(self.selector.current_level())
        };
        count.min(total_gaussians)
    }

    /// Advance any in-progress blend transition by `delta_distance` (world
    /// units the camera has travelled since the last call), driven by
    /// [`AdaptiveLodConfig::blend_distance`].
    ///
    /// Returns `true` when the transition completes during this call.
    /// No-op (returns `false`) when no transition is in progress — call
    /// this once per frame after [`LodManager::update`] to make
    /// [`LodManager::is_transitioning`] and [`LodManager::blended_gaussian_count`]
    /// actually progress instead of latching at `alpha = 0.0` forever.
    pub fn advance_transition(&mut self, delta_distance: f32) -> bool {
        let blend_distance = self.selector.config().blend_distance;
        self.transition.advance(delta_distance, blend_distance)
    }

    /// Return the currently active LOD level.
    pub fn current_level(&self) -> &LodLevel {
        self.selector.current_level()
    }

    /// Return `true` when a blend transition is in progress.
    pub fn is_transitioning(&self) -> bool {
        self.transition.is_blending
    }

    /// Return a reference to the accumulated statistics.
    pub fn stats(&self) -> &LodStats {
        &self.stats
    }

    /// Reset statistics counters to zero.
    pub fn reset_stats(&mut self) {
        self.stats = LodStats::default();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GaussianAttributes;

    // ── helpers ──────────────────────────────────────────────────────────────

    fn make_camera() -> LodCamera {
        LodCamera::new(
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],             // looking along +Z
            std::f32::consts::FRAC_PI_2, // 90° fov_y
            800,
            600,
        )
    }

    /// Build a GaussianAttributes with all-zero fields except position and scale.
    fn make_gaussian(
        position: [f32; 3],
        log_scale: [f32; 3],
        opacity_logit: f32,
    ) -> GaussianAttributes {
        GaussianAttributes {
            position,
            _pad0: 0.0,
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: log_scale,
            opacity: opacity_logit,
        }
    }

    fn make_model_with_gaussians(gaussians: Vec<GaussianAttributes>) -> GaussianModel {
        let n = gaussians.len();
        GaussianModel {
            sh_coeffs: vec![0.0; n * 3], // sh_degree=0 → (0+1)^2 * 3 = 3
            sh_degree: 0,
            face_indices: vec![0u32; n],
            barycentric: vec![[1.0 / 3.0; 3]; n],
            local_offsets: vec![[0.0; 3]; n],
            is_rigid: vec![false; n],
            gaussians,
        }
    }

    fn empty_model() -> GaussianModel {
        make_model_with_gaussians(vec![])
    }

    // ── LodCamera ────────────────────────────────────────────────────────────

    #[test]
    fn test_pixels_per_unit_positive_depth() {
        let cam = make_camera();
        let ppu = cam.pixels_per_unit(1.0);
        assert!(
            ppu > 0.0,
            "pixels_per_unit at depth=1 should be positive, got {ppu}"
        );
    }

    #[test]
    fn test_pixels_per_unit_zero_depth() {
        let cam = make_camera();
        assert_eq!(cam.pixels_per_unit(0.0), 0.0);
    }

    #[test]
    fn test_pixels_per_unit_negative_depth() {
        let cam = make_camera();
        assert_eq!(cam.pixels_per_unit(-1.0), 0.0);
    }

    #[test]
    fn test_depth_of_camera_position() {
        let cam = make_camera();
        // Point at the camera itself — depth should be 0.
        let d = cam.depth_of(cam.position);
        assert!(
            d.abs() < 1e-6,
            "depth at camera position should be 0, got {d}"
        );
    }

    #[test]
    fn test_depth_of_point_in_front() {
        let cam = make_camera(); // looking along +Z
        let d = cam.depth_of([0.0, 0.0, 5.0]);
        assert!(
            d > 0.0,
            "point in front should have positive depth, got {d}"
        );
        assert!((d - 5.0).abs() < 1e-5, "expected depth ~5.0, got {d}");
    }

    #[test]
    fn test_depth_of_point_behind() {
        let cam = make_camera(); // looking along +Z
        let d = cam.depth_of([0.0, 0.0, -3.0]);
        assert!(
            d < 0.0,
            "point behind camera should have negative depth, got {d}"
        );
    }

    #[test]
    fn test_screen_radius_large_nearby_gaussian() {
        let cam = make_camera();
        // Large scale (exp(2.0) ≈ 7.4), close to camera at depth 1.
        let g = make_gaussian([0.0, 0.0, 1.0], [2.0, 2.0, 2.0], 0.0);
        let r = cam.screen_radius(&g);
        assert!(
            r > 10.0,
            "large nearby Gaussian should have large screen radius, got {r}"
        );
    }

    #[test]
    fn test_screen_radius_tiny_distant_gaussian() {
        let cam = make_camera();
        // Tiny scale (exp(-5.0) ≈ 0.007), far away at depth 50.
        let g = make_gaussian([0.0, 0.0, 50.0], [-5.0, -5.0, -5.0], 0.0);
        let r = cam.screen_radius(&g);
        assert!(
            r < 0.1,
            "tiny distant Gaussian should have small screen radius, got {r}"
        );
    }

    #[test]
    fn test_screen_radius_behind_near_plane() {
        let cam = make_camera(); // near = 0.01
        let g = make_gaussian([0.0, 0.0, -1.0], [0.0, 0.0, 0.0], 0.0);
        let r = cam.screen_radius(&g);
        assert_eq!(r, 0.0, "Gaussian behind camera should have radius=0");
    }

    // ── LodConfig ────────────────────────────────────────────────────────────

    #[test]
    fn test_lod_config_defaults() {
        let cfg = LodConfig::default();
        assert!((cfg.min_pixel_radius - 1.0).abs() < 1e-6);
        assert!((cfg.full_opacity_radius - 2.0).abs() < 1e-6);
        assert!((cfg.cull_threshold_fraction - 0.1).abs() < 1e-6);
        assert!((cfg.max_depth - 0.0).abs() < 1e-6);
        assert!(!cfg.modify_in_place);
    }

    // ── opacity_scale_from_radius ─────────────────────────────────────────────

    #[test]
    fn test_opacity_scale_zero_radius_culled() {
        let cfg = LodConfig::default();
        let (scale, culled) = opacity_scale_from_radius(0.0, &cfg);
        assert!(culled, "radius=0 should be culled");
        assert_eq!(scale, 0.0);
    }

    #[test]
    fn test_opacity_scale_full_at_full_opacity_radius() {
        let cfg = LodConfig::default(); // full_opacity_radius = 2.0
        let (scale, culled) = opacity_scale_from_radius(2.0, &cfg);
        assert!(!culled);
        assert!(
            (scale - 1.0).abs() < 1e-6,
            "scale should be 1.0 at full_opacity_radius, got {scale}"
        );
    }

    #[test]
    fn test_opacity_scale_full_above_full_opacity_radius() {
        let cfg = LodConfig::default();
        let (scale, culled) = opacity_scale_from_radius(10.0, &cfg);
        assert!(!culled);
        assert!((scale - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_opacity_scale_midpoint_linear_ramp() {
        let cfg = LodConfig::default();
        // cull_radius = 1.0 * 0.1 = 0.1
        // full_opacity_radius = 2.0
        // midpoint radius = (0.1 + 2.0) / 2 = 1.05
        // t = (1.05 - 0.1) / (2.0 - 0.1) = 0.95 / 1.9 = 0.5
        let mid_radius =
            (cfg.min_pixel_radius * cfg.cull_threshold_fraction + cfg.full_opacity_radius) / 2.0;
        let (scale, culled) = opacity_scale_from_radius(mid_radius, &cfg);
        assert!(!culled, "midpoint should not be culled");
        assert!(
            (scale - 0.5).abs() < 1e-5,
            "midpoint should give scale≈0.5, got {scale}"
        );
    }

    #[test]
    fn test_opacity_scale_just_below_cull_threshold_is_culled() {
        let cfg = LodConfig::default();
        let cull_radius = cfg.min_pixel_radius * cfg.cull_threshold_fraction; // 0.1
        let (_, culled) = opacity_scale_from_radius(cull_radius - 0.001, &cfg);
        assert!(culled, "just below cull threshold should be culled");
    }

    // ── LodFilter::compute ────────────────────────────────────────────────────

    #[test]
    fn test_compute_returns_same_length_as_model() {
        let cam = make_camera();
        let filter = LodFilter::new(LodConfig::default());
        let gaussians: Vec<GaussianAttributes> = (0..20)
            .map(|i| make_gaussian([0.0, 0.0, (i + 1) as f32], [0.0; 3], 0.0))
            .collect();
        let model = make_model_with_gaussians(gaussians);
        let info = filter.compute(&model, &cam);
        assert_eq!(info.len(), model.len());
    }

    #[test]
    fn test_count_culled_empty_model() {
        let cam = make_camera();
        let filter = LodFilter::new(LodConfig::default());
        let model = empty_model();
        assert_eq!(filter.count_culled(&model, &cam), 0);
    }

    // ── LodFilter::apply ──────────────────────────────────────────────────────

    #[test]
    fn test_apply_removes_culled_gaussians() {
        let cam = make_camera();
        // One tiny distant Gaussian (will be culled) + one large nearby (will be kept).
        let tiny = make_gaussian([0.0, 0.0, 100.0], [-10.0, -10.0, -10.0], 0.0);
        let big = make_gaussian([0.0, 0.0, 1.0], [3.0, 3.0, 3.0], 0.0);
        let model = make_model_with_gaussians(vec![tiny, big]);
        let filter = LodFilter::new(LodConfig::default());

        let filtered = filter.apply(&model, &cam);
        let orig_culled = filter.count_culled(&model, &cam);
        assert!(orig_culled > 0, "at least one Gaussian should be culled");
        assert_eq!(filtered.len(), model.len() - orig_culled);
    }

    #[test]
    fn test_apply_preserves_non_culled_gaussians_exactly() {
        let cam = make_camera();
        // Very large Gaussian at moderate depth — guaranteed full opacity.
        let g = make_gaussian([0.0, 0.0, 2.0], [5.0, 5.0, 5.0], 0.0);
        let model = make_model_with_gaussians(vec![g]);
        let filter = LodFilter::new(LodConfig::default());

        let info = filter.compute(&model, &cam);
        // Verify our assumption: full opacity, not culled.
        assert!(!info[0].is_culled);
        assert!((info[0].opacity_scale - 1.0).abs() < 1e-6);

        let filtered = filter.apply(&model, &cam);
        assert_eq!(filtered.len(), 1);
        // Position should be identical.
        assert_eq!(filtered.gaussians[0].position, g.position);
        // Opacity should be unchanged (scale = 1.0).
        assert!((filtered.gaussians[0].opacity - g.opacity).abs() < 1e-5);
    }

    #[test]
    fn test_apply_adjusts_opacity_of_faded_gaussians() {
        let cam = make_camera();
        // Gaussian with screen radius in the fade zone.
        // With fov_y=PI/2, image_height=600, depth=5:
        //   pixels_per_unit = 300 / (5 * tan(PI/4)) = 300 / 5 = 60
        //   world_radius = exp(log_scale) = e^(-3) ≈ 0.0498
        //   screen_radius ≈ 60 * 0.0498 ≈ 2.99 px
        // That's just above full_opacity_radius=2.0 so let's pick something smaller.
        // Depth=10: ppu = 300 / 10 = 30. exp(-4) ≈ 0.0183. r ≈ 0.55 px — in the ramp zone.
        let g = make_gaussian([0.0, 0.0, 10.0], [-4.0, -4.0, -4.0], 0.0);
        let original_opacity_logit = g.opacity; // 0.0 → sigmoid(0) = 0.5
        let model = make_model_with_gaussians(vec![g]);
        let filter = LodFilter::new(LodConfig::default());

        let info = filter.compute(&model, &cam);
        assert!(!info[0].is_culled, "Gaussian should not be culled");
        assert!(
            info[0].opacity_scale < 1.0,
            "opacity_scale should be < 1.0 in fade zone, got {}",
            info[0].opacity_scale
        );

        let filtered = filter.apply(&model, &cam);
        assert_eq!(filtered.len(), 1);

        // After applying scale < 1.0, the stored opacity should decrease.
        let new_opacity_logit = filtered.gaussians[0].opacity;
        assert!(
            new_opacity_logit < original_opacity_logit,
            "opacity logit should decrease after fading; original={original_opacity_logit}, new={new_opacity_logit}"
        );
    }

    // ── LodFilter::memory_saved_bytes ─────────────────────────────────────────

    #[test]
    fn test_memory_saved_empty_model() {
        let cam = make_camera();
        let filter = LodFilter::new(LodConfig::default());
        let model = empty_model();
        assert_eq!(filter.memory_saved_bytes(&model, &cam), 0);
    }

    #[test]
    fn test_memory_saved_proportional_to_culled_count() {
        let cam = make_camera();
        // Two tiny Gaussians that will be culled.
        let t1 = make_gaussian([0.0, 0.0, 90.0], [-10.0; 3], 0.0);
        let t2 = make_gaussian([0.0, 0.0, 95.0], [-10.0; 3], 0.0);
        let model = make_model_with_gaussians(vec![t1, t2]);
        let filter = LodFilter::new(LodConfig::default());
        let saved = filter.memory_saved_bytes(&model, &cam);
        // Should be > 0 since both will be culled.
        let culled = filter.count_culled(&model, &cam);
        assert!(culled > 0, "at least one should be culled");
        assert!(
            saved > 0,
            "memory saved should be > 0 when some Gaussians are culled"
        );
    }

    // ── LodFilter::stats_string ───────────────────────────────────────────────

    #[test]
    fn test_stats_string_non_empty() {
        let cam = make_camera();
        let filter = LodFilter::new(LodConfig::default());
        let g = make_gaussian([0.0, 0.0, 5.0], [0.0; 3], 0.0);
        let model = make_model_with_gaussians(vec![g]);
        let s = filter.stats_string(&model, &cam);
        assert!(!s.is_empty(), "stats_string should be non-empty");
        assert!(s.contains("total="), "stats_string should contain 'total='");
    }

    // ── Large model: culled + kept = total ────────────────────────────────────

    #[test]
    fn test_large_model_culled_plus_kept_equals_total() {
        let cam = make_camera();
        let filter = LodFilter::new(LodConfig::default());

        // Mix of tiny distant and large nearby Gaussians.
        let mut gaussians = Vec::new();
        for i in 0..50 {
            let depth = (i + 1) as f32 * 2.0;
            let log_scale = if i % 3 == 0 { -8.0 } else { 1.0 };
            gaussians.push(make_gaussian([0.0, 0.0, depth], [log_scale; 3], 0.0));
        }

        let n = gaussians.len();
        let model = make_model_with_gaussians(gaussians);
        let lod_info = filter.compute(&model, &cam);

        let culled: usize = lod_info.iter().filter(|i| i.is_culled).count();
        let kept: usize = lod_info.iter().filter(|i| !i.is_culled).count();

        assert_eq!(culled + kept, n, "culled + kept must equal total Gaussians");
        assert_eq!(filter.count_culled(&model, &cam), culled);

        let filtered = filter.apply(&model, &cam);
        assert_eq!(filtered.len(), kept);
        // Verify SH and FLAME vecs are consistent.
        assert_eq!(filtered.sh_coeffs.len(), kept * 3); // sh_degree=0 → 3 per gaussian
        assert_eq!(filtered.face_indices.len(), kept);
        assert_eq!(filtered.barycentric.len(), kept);
        assert_eq!(filtered.local_offsets.len(), kept);
        assert_eq!(filtered.is_rigid.len(), kept);
    }

    // ── Adaptive LOD management tests ─────────────────────────────────────────

    fn make_adaptive_config() -> AdaptiveLodConfig {
        AdaptiveLodConfig {
            levels: vec![
                LodLevel {
                    level_index: 0,
                    target_gaussian_count: 10_000,
                    max_distance: 2.0,
                    min_pixel_coverage: 2.0,
                    quality_factor: 1.0,
                },
                LodLevel {
                    level_index: 1,
                    target_gaussian_count: 5_000,
                    max_distance: 5.0,
                    min_pixel_coverage: 1.0,
                    quality_factor: 0.75,
                },
                LodLevel {
                    level_index: 2,
                    target_gaussian_count: 2_500,
                    max_distance: 15.0,
                    min_pixel_coverage: 0.5,
                    quality_factor: 0.5,
                },
                LodLevel {
                    level_index: 3,
                    target_gaussian_count: 1_000,
                    max_distance: f32::INFINITY,
                    min_pixel_coverage: 0.25,
                    quality_factor: 0.25,
                },
            ],
            hysteresis: 0.1,
            enable_blending: false,
            blend_distance: 0.5,
        }
    }

    #[test]
    fn test_lod_level_applicable_at_distance() {
        let lvl = LodLevel {
            level_index: 0,
            target_gaussian_count: 1000,
            max_distance: 5.0,
            min_pixel_coverage: 1.0,
            quality_factor: 1.0,
        };
        assert!(lvl.is_applicable_at_distance(0.0));
        assert!(lvl.is_applicable_at_distance(5.0));
        assert!(!lvl.is_applicable_at_distance(5.001));
        assert!(!lvl.is_applicable_at_distance(100.0));
    }

    #[test]
    fn test_lod_config_validate_empty_error() {
        let cfg = AdaptiveLodConfig {
            levels: vec![],
            hysteresis: 0.0,
            enable_blending: false,
            blend_distance: 1.0,
        };
        assert!(
            matches!(cfg.validate(), Err(LodError::EmptyLevels)),
            "empty levels should produce LodError::EmptyLevels"
        );
    }

    #[test]
    fn test_lod_config_validate_unsorted_error() {
        let cfg = AdaptiveLodConfig {
            levels: vec![
                LodLevel {
                    level_index: 0,
                    target_gaussian_count: 10000,
                    max_distance: 5.0,
                    min_pixel_coverage: 1.0,
                    quality_factor: 1.0,
                },
                LodLevel {
                    level_index: 1,
                    target_gaussian_count: 5000,
                    max_distance: 3.0, // ← out of order
                    min_pixel_coverage: 0.5,
                    quality_factor: 0.5,
                },
            ],
            hysteresis: 0.0,
            enable_blending: false,
            blend_distance: 0.5,
        };
        assert!(
            matches!(
                cfg.validate(),
                Err(LodError::LevelsNotSorted { idx: 1, .. })
            ),
            "out-of-order levels should produce LodError::LevelsNotSorted"
        );
    }

    #[test]
    fn test_lod_config_validate_ok() {
        let cfg = make_adaptive_config();
        assert!(
            cfg.validate().is_ok(),
            "valid config should pass validate()"
        );
    }

    #[test]
    fn test_lod_config_default_for_count() {
        let total = 80_000usize;
        let cfg = AdaptiveLodConfig::default_for_count(total);
        assert_eq!(cfg.levels.len(), 4);
        assert_eq!(cfg.levels[0].target_gaussian_count, total);
        assert_eq!(cfg.levels[1].target_gaussian_count, total / 2);
        assert_eq!(cfg.levels[2].target_gaussian_count, total / 4);
        assert_eq!(cfg.levels[3].target_gaussian_count, total / 10);
        assert!((cfg.levels[0].max_distance - 2.0).abs() < 1e-5);
        assert!((cfg.levels[1].max_distance - 5.0).abs() < 1e-5);
        assert!((cfg.levels[2].max_distance - 15.0).abs() < 1e-5);
        assert!(cfg.levels[3].max_distance.is_infinite());
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_lod_selector_select_by_distance_basic() {
        let cfg = make_adaptive_config();
        let mut sel = LodSelector::new(cfg).expect("valid config");
        // Distance well within level 0 range.
        let lvl = sel.select_by_distance(1.0);
        assert_eq!(lvl.level_index, 0);
        // Distance beyond level 0 but within level 1 (no hysteresis band yet).
        // Since hysteresis=0.1, to downgrade from 0 we need dist > 2.0*(1+0.1)=2.2.
        let lvl = sel.select_by_distance(3.0);
        assert_eq!(lvl.level_index, 1);
    }

    #[test]
    fn test_lod_selector_hysteresis_upgrade() {
        let cfg = make_adaptive_config(); // hysteresis = 0.1
        let mut sel = LodSelector::new(cfg).expect("valid config");

        // Move to level 1 by going far.
        sel.select_by_distance(10.0);
        assert_eq!(sel.current_level_idx, 2); // should be at level 2 (dist=10 > 5*(1.1)=5.5)

        // Now move back; to upgrade from level 2 to level 1 we need dist < 5.0*(1-0.1)=4.5.
        let lvl = sel.select_by_distance(4.0);
        assert_eq!(lvl.level_index, 1, "should upgrade to level 1 at dist=4.0");
    }

    #[test]
    fn test_lod_selector_hysteresis_downgrade() {
        let cfg = make_adaptive_config(); // hysteresis = 0.1
        let mut sel = LodSelector::new(cfg).expect("valid config");

        // Start at level 0 (close).
        sel.select_by_distance(1.0);
        assert_eq!(sel.current_level_idx, 0);

        // Distance = 2.05: within the hysteresis band (2.0 to 2.0*1.1=2.2) — should stay at 0.
        let lvl = sel.select_by_distance(2.05);
        assert_eq!(
            lvl.level_index, 0,
            "inside hysteresis band should NOT downgrade"
        );

        // Distance = 2.25: beyond the band — should downgrade to level 1.
        let lvl = sel.select_by_distance(2.25);
        assert_eq!(
            lvl.level_index, 1,
            "beyond hysteresis band should downgrade"
        );
    }

    #[test]
    fn test_lod_selector_reset() {
        let cfg = make_adaptive_config();
        let mut sel = LodSelector::new(cfg).expect("valid config");

        // Move to a coarser level.
        sel.select_by_distance(100.0);
        assert!(sel.current_level_idx > 0, "should be at a coarser level");

        sel.reset();
        assert_eq!(sel.current_level_idx, 0, "reset should return to level 0");
        assert_eq!(sel.current_level().level_index, 0);
    }

    #[test]
    fn test_lod_selector_stateless_level_at_distance() {
        let cfg = make_adaptive_config();
        assert_eq!(LodSelector::level_at_distance(&cfg, 1.0), 0);
        assert_eq!(LodSelector::level_at_distance(&cfg, 2.0), 0);
        assert_eq!(LodSelector::level_at_distance(&cfg, 3.0), 1);
        assert_eq!(LodSelector::level_at_distance(&cfg, 5.0), 1);
        assert_eq!(LodSelector::level_at_distance(&cfg, 10.0), 2);
        assert_eq!(LodSelector::level_at_distance(&cfg, 1000.0), 3);
    }

    #[test]
    fn test_lod_transition_none() {
        let t = LodTransition::none(2);
        assert_eq!(t.from_level, 2);
        assert!(t.to_level.is_none());
        assert!((t.alpha - 0.0).abs() < 1e-6);
        assert!(!t.is_blending);
    }

    #[test]
    fn test_lod_transition_advance_completes() {
        let mut t = LodTransition {
            from_level: 0,
            to_level: Some(1),
            alpha: 0.0,
            is_blending: true,
        };
        // blend_distance = 1.0; advance by 0.6 → alpha=0.6, not done.
        let done = t.advance(0.6, 1.0);
        assert!(!done, "transition should not be complete after 60%");
        assert!((t.alpha - 0.6).abs() < 1e-5);

        // Advance by another 0.5 → alpha ≥ 1.0 → complete.
        let done = t.advance(0.5, 1.0);
        assert!(done, "transition should complete after ≥ 100%");
        assert!(!t.is_blending);
        assert_eq!(
            t.from_level, 1,
            "from_level should update to to_level on completion"
        );
        assert!(t.to_level.is_none());
    }

    #[test]
    fn test_lod_transition_blended_count() {
        let t_none = LodTransition::none(0);
        // alpha=0 → fully from_count.
        assert_eq!(t_none.blended_count(10_000, 5_000), 10_000);

        let t_half = LodTransition {
            from_level: 0,
            to_level: Some(1),
            alpha: 0.5,
            is_blending: true,
        };
        let count = t_half.blended_count(10_000, 5_000);
        // (10000 * 0.5 + 5000 * 0.5) = 7500
        assert_eq!(count, 7_500);

        let t_full = LodTransition {
            from_level: 0,
            to_level: Some(1),
            alpha: 1.0,
            is_blending: true,
        };
        assert_eq!(t_full.blended_count(10_000, 5_000), 5_000);
    }

    #[test]
    fn test_lod_stats_record_and_usage_fraction() {
        let mut stats = LodStats::default();
        assert_eq!(stats.total_frames, 0);
        assert!((stats.level_usage_fraction(0) - 0.0).abs() < 1e-6);

        stats.record_frame(0, 10_000);
        stats.record_frame(0, 10_000);
        stats.record_frame(1, 5_000);

        assert_eq!(stats.total_frames, 3);
        assert_eq!(stats.current_gaussian_count, 5_000);

        let frac0 = stats.level_usage_fraction(0);
        let frac1 = stats.level_usage_fraction(1);
        assert!(
            (frac0 - 2.0 / 3.0).abs() < 1e-5,
            "level 0 should be 2/3, got {frac0}"
        );
        assert!(
            (frac1 - 1.0 / 3.0).abs() < 1e-5,
            "level 1 should be 1/3, got {frac1}"
        );

        stats.record_transition();
        assert_eq!(stats.transitions, 1);

        let summary = stats.format_summary();
        assert!(summary.contains("total_frames=3"), "summary: {summary}");
        assert!(summary.contains("transitions=1"), "summary: {summary}");
    }

    #[test]
    fn test_lod_manager_update() {
        let cfg = make_adaptive_config();
        let mut mgr = LodManager::new(cfg).expect("valid config");

        // First update at close distance → level 0.
        let count = mgr.update(1.0);
        assert_eq!(count, 10_000);
        assert_eq!(mgr.current_level().level_index, 0);
        assert_eq!(mgr.stats().total_frames, 1);

        // Update at far distance → level 3.
        let count = mgr.update(100.0);
        assert_eq!(count, 1_000);
        assert_eq!(mgr.current_level().level_index, 3);
        assert_eq!(mgr.stats().transitions, 1, "one transition recorded");
        assert_eq!(mgr.stats().total_frames, 2);
    }

    #[test]
    fn test_lod_manager_current_gaussian_count() {
        let cfg = make_adaptive_config();
        let mut mgr = LodManager::new(cfg).expect("valid config");
        mgr.update(1.0); // level 0 → target = 10_000

        // total_gaussians > target → clamped to target.
        assert_eq!(mgr.current_gaussian_count(50_000), 10_000);
        // total_gaussians < target → clamped to total.
        assert_eq!(mgr.current_gaussian_count(8_000), 8_000);
    }

    // ── AdaptiveLodConfig::validate: level_index must match array position ──

    #[test]
    fn test_lod_config_validate_rejects_mismatched_level_index() {
        let cfg = AdaptiveLodConfig {
            levels: vec![
                LodLevel {
                    level_index: 0,
                    target_gaussian_count: 1000,
                    max_distance: 5.0,
                    min_pixel_coverage: 1.0,
                    quality_factor: 1.0,
                },
                LodLevel {
                    level_index: 5, // ← does not match array position 1
                    target_gaussian_count: 500,
                    max_distance: 10.0,
                    min_pixel_coverage: 0.5,
                    quality_factor: 0.5,
                },
            ],
            hysteresis: 0.0,
            enable_blending: false,
            blend_distance: 0.5,
        };
        assert!(
            matches!(cfg.validate(), Err(LodError::InvalidConfig(_))),
            "mismatched level_index should be rejected by validate()"
        );
        // LodSelector::new / LodManager::new both call validate() and must
        // propagate the same rejection.
        assert!(LodSelector::new(cfg.clone()).is_err());
        assert!(LodManager::new(cfg).is_err());
    }

    // ── LodManager::advance_transition / blended_gaussian_count ─────────────

    #[test]
    fn test_lod_manager_advance_transition_completes_and_clears_flag() {
        let mut cfg = make_adaptive_config();
        cfg.enable_blending = true;
        cfg.blend_distance = 1.0;
        let mut mgr = LodManager::new(cfg).expect("valid config");

        mgr.update(1.0); // level 0, no transition yet (first frame)
        assert!(!mgr.is_transitioning());

        mgr.update(3.0); // crosses into level 1 → starts a blend
        assert!(mgr.is_transitioning(), "level change should start a blend");

        // Advance by less than blend_distance: still blending.
        let done = mgr.advance_transition(0.5);
        assert!(!done);
        assert!(mgr.is_transitioning());

        // Advance past blend_distance: transition completes.
        let done = mgr.advance_transition(0.6);
        assert!(done);
        assert!(
            !mgr.is_transitioning(),
            "transition should clear once alpha reaches 1.0"
        );
    }

    #[test]
    fn test_lod_manager_advance_transition_noop_without_blending_enabled() {
        let cfg = make_adaptive_config(); // enable_blending = false
        let mut mgr = LodManager::new(cfg).expect("valid config");
        mgr.update(1.0);
        mgr.update(10.0); // level change, but blending disabled
        assert!(!mgr.is_transitioning());
        assert!(!mgr.advance_transition(100.0), "no transition to advance");
    }

    #[test]
    fn test_lod_manager_blended_gaussian_count_interpolates() {
        let mut cfg = make_adaptive_config();
        cfg.enable_blending = true;
        cfg.blend_distance = 2.0;
        let mut mgr = LodManager::new(cfg).expect("valid config");

        mgr.update(1.0); // level 0, target=10_000, quality_factor=1.0
        mgr.update(3.0); // → level 1, target=5_000, quality_factor=0.75; starts blend at alpha=0

        // Immediately after the transition starts, alpha=0 → fully from_count.
        let from_count = (10_000.0_f32 * 1.0).round() as usize;
        assert_eq!(mgr.blended_gaussian_count(1_000_000), from_count);

        // Halfway through the blend distance.
        mgr.advance_transition(1.0);
        let to_count = (5_000.0_f32 * 0.75).round() as usize;
        let expected_half = ((from_count as f32) * 0.5 + (to_count as f32) * 0.5) as usize;
        assert_eq!(mgr.blended_gaussian_count(1_000_000), expected_half);
    }

    // ── LodFilter::apply_mut ─────────────────────────────────────────────────

    #[test]
    fn test_lod_filter_apply_mut_matches_apply() {
        let cam = make_camera();
        let config = LodConfig {
            min_pixel_radius: 1.0,
            full_opacity_radius: 5.0,
            cull_threshold_fraction: 0.5,
            max_depth: 0.0,
            modify_in_place: true,
        };
        let filter = LodFilter::new(config);
        let gaussians = vec![
            make_gaussian([0.0, 0.0, 2.0], [0.0, 0.0, 0.0], 2.0),
            make_gaussian([0.0, 0.0, 50.0], [-5.0, -5.0, -5.0], 2.0),
        ];
        let model_a = make_model_with_gaussians(gaussians.clone());
        let mut model_b = make_model_with_gaussians(gaussians);

        let expected = filter.apply(&model_a, &cam);
        let culled = filter.apply_mut(&mut model_b, &cam);

        assert_eq!(model_b.gaussians.len(), expected.gaussians.len());
        assert_eq!(culled, model_a.gaussians.len() - expected.gaussians.len());
        for (a, b) in expected.gaussians.iter().zip(model_b.gaussians.iter()) {
            assert_eq!(a.position, b.position);
            assert!((a.opacity - b.opacity).abs() < 1e-6);
        }
    }

    // ── LodCamera::pixels_per_unit_horizontal ────────────────────────────────

    #[test]
    fn test_pixels_per_unit_horizontal_matches_vertical_for_square_pixels() {
        let cam = make_camera(); // 800x600, matches its own aspect ratio
        let depth = 4.0_f32;
        let vert = cam.pixels_per_unit(depth);
        let horiz = cam.pixels_per_unit_horizontal(depth);
        assert!(
            (vert - horiz).abs() < 1e-3,
            "expected pixels_per_unit_horizontal ({horiz}) to match pixels_per_unit ({vert})"
        );
    }

    #[test]
    fn test_pixels_per_unit_horizontal_zero_depth() {
        let cam = make_camera();
        assert_eq!(cam.pixels_per_unit_horizontal(0.0), 0.0);
    }

    #[test]
    fn test_pixels_per_unit_horizontal_zero_height() {
        let cam = LodCamera::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 1.0, 100, 0);
        assert_eq!(cam.pixels_per_unit_horizontal(1.0), 0.0);
    }
}
