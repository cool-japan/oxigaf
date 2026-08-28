//! # oxigaf-flame
//!
//! FLAME parametric head model implementation in Pure Rust.
//!
//! This crate implements the [FLAME (Faces Learned with an Articulated Model and Expressions)](https://flame.is.tue.mpg.de/)
//! parametric 3D head model in pure Rust, with no dependencies on Python or C/C++ libraries.
//!
//! ## Features
//!
//! - **FLAME model loading** from `.npy` files (converted from the original `.pkl`)
//! - **Linear Blend Skinning (LBS)** forward pass: parameters → posed mesh
//! - **CPU normal map rendering** for diffusion model conditioning
//! - **Mesh surface point sampling** for Gaussian initialization
//! - **Zero-cost abstractions** with extensive use of `#[inline]` for performance
//!
//! ### Cargo Features
//!
//! This crate supports the following feature flags:
//!
//! - **`simd`** (optional, requires nightly Rust with `portable_simd`):
//!   Enables SIMD-accelerated vector operations for:
//!   - Normal map rendering (3-4× faster)
//!   - Rodrigues rotation computation
//!   - Blend shape evaluation
//!
//! - **`parallel`** (optional):
//!   Enables parallel batch processing with `rayon`:
//!   - `forward_batch_par()` - parallel mesh generation
//!   - `compute_normals_batch_par()` - parallel normal computation
//!   - Near-linear speedup with CPU core count
//!
//! - **`full`** (convenience): Enables `parallel` and `npz` (both stable-Rust
//!   features). `simd` is intentionally excluded, since it requires nightly
//!   Rust -- see `full_nightly` below.
//!
//! - **`full_nightly`**: Enables `simd`, `parallel`, and `npz`. Requires
//!   nightly Rust.
//!
//! Example usage:
//! ```toml
//! # In Cargo.toml
//! oxigaf-flame = { version = "0.1", features = ["parallel"] }
//! ```
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use oxigaf_flame::{FlameModel, FlameParams};
//!
//! // Load FLAME model from directory containing .npy files
//! let model = FlameModel::load("path/to/flame/model")?;
//!
//! // Create neutral parameters (zero shape, expression, pose)
//! let params = FlameParams::neutral();
//!
//! // Run forward pass to get posed mesh
//! let mesh = model.forward(&params);
//!
//! println!("Generated mesh with {} vertices", mesh.vertices.len());
//! # Ok::<(), oxigaf_flame::FlameError>(())
//! ```
//!
//! ## FLAME Parameters
//!
//! The FLAME model is controlled by several parameter types:
//!
//! - **Shape parameters** (β): Control identity-specific features (typically 100-300 coefficients)
//! - **Expression parameters** (ψ): Control facial expressions (typically 50-100 coefficients)
//! - **Pose parameters** (θ): Control joint rotations (5 joints × 3 = 15 values)
//!   - Root rotation (global head orientation)
//!   - Neck rotation
//!   - Jaw rotation
//!   - Left eye rotation
//!   - Right eye rotation
//! - **Translation**: Global 3D translation applied after posing
//!
//! ## Coordinate System
//!
//! FLAME uses a **right-handed coordinate system**:
//! - +X: Left (from the subject's perspective; the viewer's right)
//! - +Y: Up
//! - +Z: Forward (out of the face, toward the viewer)
//!
//! This matches [`vertex_mask::FaceRegion`] and `head_geometry`'s region
//! classification, both of which place the subject's left eye/cheek/ear at
//! positive X.
//!
//! Rotations are specified as **axis-angle** vectors and converted to rotation
//! matrices using [Rodrigues' formula](https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula).
//!
//! ## Performance
//!
//! The LBS forward pass is optimized for real-time performance:
//! - ~1-2ms for standard FLAME mesh (5023 vertices) on modern CPUs
//! - Critical path functions are marked with `#[inline]`
//! - Uses `ndarray` for efficient array operations (pure Rust, no BLAS
//!   backend; see the COOLJAPAN Pure Rust policy)
//!
//! Run benchmarks with: `cargo bench -p oxigaf-flame`
//!
//! ## References
//!
//! - [FLAME Paper](https://ps.is.tuebingen.mpg.de/uploads_file/attachment/attachment/400/paper.pdf)
//! - [FLAME Model](https://flame.is.tue.mpg.de/)

// Allow unwrap in test code but deny in library code
#![cfg_attr(not(test), deny(clippy::unwrap_used))]
// Allow expect in test code but deny in library code
#![cfg_attr(not(test), deny(clippy::expect_used))]
// Additional quality lints
#![warn(clippy::all, clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::similar_names)]
#![allow(clippy::float_cmp)]
#![allow(clippy::approx_constant)]
// Test code builds configs via Default then overrides a single field to test
// boundary conditions; struct-update syntax would obscure the intent.
#![allow(clippy::field_reassign_with_default)]
// `items_after_statements` fires on nested helper `fn` definitions placed
// inside test functions after setup code.  Relocating them to the top of
// the function would separate them from the test logic they support.
#![allow(clippy::items_after_statements)]
// Nightly feature for portable SIMD (requires both 'simd' feature AND nightly Rust)
#![cfg_attr(all(feature = "simd", nightly), feature(portable_simd))]

pub mod albedo_map;
pub mod avatar_rig;
pub mod blend_shape_solver;
pub mod canonical;
pub mod contact_detection;
pub mod conversion;
pub mod depth_estimation;
pub mod dynamic_landmarks;
pub mod emotion_recognition;
pub mod error;
pub mod expression_animation;
pub mod expression_clustering;
pub mod expression_retargeting;
pub mod expression_transfer;
pub mod expressions;
pub mod face_atlas;
pub mod face_normalization;
pub mod facs;
pub mod fitting;
pub mod geodesic;
pub mod gpu_buffers;
pub mod head_geometry;
pub mod head_tracker;
pub mod io;
pub mod io_safetensors;
pub mod landmarks;
pub mod lighting_model;
pub mod mesh;
pub mod mesh_analysis;
pub mod mesh_morphing;
pub mod mesh_ops;
pub mod mesh_repair;
pub mod mesh_smoothing;
pub mod mesh_subdivision;
pub mod model;
pub mod multiresolution;
pub mod normal_map;
pub mod param_sampler;
pub mod params;
mod params_builder;
pub mod phoneme_animation;
pub mod pose_estimation;
pub mod pose_prior;
pub mod retargeting;
pub mod rigid_alignment;
pub mod sampler;
pub mod sequence;
pub mod shape_analysis;
pub mod statistical_shape_model;
pub mod symmetry;
pub mod texture_baking;
pub mod timeline;
pub mod traits;
pub mod uv;
pub mod uv_texture;
pub mod vertex_mask;
pub mod visibility_culling;
pub mod visualize;
pub mod warp_field;

// SIMD module (requires nightly + simd feature)
#[cfg(all(feature = "simd", nightly))]
pub mod simd;

pub use albedo_map::{
    albedo_brightness, albedo_to_vertex_colors, apply_ambient_occlusion, bake_vertex_albedo,
    blend_albedos, checker_texture, compute_albedo_stats, normalize_albedo,
    per_vertex_albedo_from_texture, sh_to_rgb, AlbedoColor, AlbedoConfig, AlbedoMapError,
    AlbedoStats, AlbedoTexture,
};
pub use avatar_rig::{
    apply_rig_to_params, compute_rig_stats, generate_talking_animation, interpolate_rigs,
    standard_face_rig, AvatarRig, BlendTarget, FlameRigParams, RigAnimation, RigControl, RigError,
    RigKeyframe, RigStats,
};
pub use blend_shape_solver::{
    apply_blend_displacements, blend_solver_residual_curve, compute_residual, compute_solver_stats,
    compute_vertex_errors, fit_expression_coefficients, format_solver_result, gradient_wrt_weights,
    nonneg_solve_blend_shapes, project_weights, solve_blend_shapes, BlendBasis, BlendSolverConfig,
    BlendSolverError, BlendSolverResult, SolverStats, WeightConstraints,
};
pub use canonical::{
    compute_canonical_transform, compute_face_scale_for_image, estimate_head_orientation,
    face_bbox_2d, orthographic_project, CanonicalError, CanonicalFace, FaceBoundingBox,
    FaceKeypoints, HeadOrientation,
};
pub use contact_detection::{
    analyze_contact, check_eye_contact, check_mouth_contact, classify_eye_state,
    classify_jaw_state, detect_mouth_transitions, detect_self_contact, find_contact_pairs,
    mean_distance_between_regions, mean_position, min_distance_between_regions,
    smooth_mouth_openings, ClippingStatus, ContactConfig, ContactError, ContactFlags, ContactPair,
    ContactReport, EyeContactState, EyeState, FlameContactRegions, JawState, MouthContactState,
};
pub use conversion::{convert_npy_to_safetensors, convert_npy_to_safetensors_with_metadata};
pub use depth_estimation::{
    blend_depth_maps, compute_depth_stats, depth_discontinuity_map, depth_to_point_cloud,
    front_depth_camera, project_point, render_conditioning_depth_maps, render_depth_map,
    side_depth_camera, three_quarter_depth_camera, DepthConfig, DepthError, DepthMap, DepthStats,
};
pub use dynamic_landmarks::{
    ContourSide, ContourVertexChains, DynamicLandmarkConfig, DynamicLandmarkExtractor,
};
pub use emotion_recognition::{
    blend_emotion_params, compute_arousal_valence, compute_emotion_scores,
    compute_emotion_transition_rate, compute_expression_intensity, dominant_in_window,
    emotion_trajectory, format_emotion_result, recognize_emotion, smooth_emotion_trajectory,
    ArousalValence, BasicEmotion, EmotionConfig, EmotionError, EmotionResult, EmotionScore,
    EmotionTrajectory,
};
pub use error::FlameError;
pub use expression_animation::{
    AnimationClip, AnimationError, AnimationPlayer, EasingFunction, ExpressionKeyframe,
    ExpressionTimeline, LoopMode,
};
pub use expression_clustering::{
    assign_to_cluster, cluster_path, cluster_prototypes, compute_cluster_stats,
    davies_bouldin_index, describe_clusters, elbow_analysis, expression_cosine_similarity,
    expression_distance, kmeans, kmeans_iteration, kmeans_plus_plus_init,
    pairwise_distance_matrix_expr, ClusterStats, ClusteringResult, ExpressionCluster,
    ExpressionClusterError, ExpressionDataset, KMeansConfig,
};
pub use expression_retargeting::{
    retar_blend_states, retar_build_expression_mirror_matrix, retar_compute_stats,
    retar_compute_variance, retar_expression_acceleration, retar_expression_similarity,
    retar_expression_velocity, retar_find_neutral_frame, retar_format_config, retar_format_stats,
    retar_mirror_expression, retar_resample_sequence, retar_slerp_states, retar_smooth_sequence,
    retar_standardize, retar_unstandardize, ExpressionState, ExpressionVarianceStats,
    LinearExpressionRetargeter, RetargetConfig, RetargetError, RetargetPair, RetargetStats,
};
pub use expression_transfer::{
    blend_transferred, classify_intensity, direct_transfer, expression_intensity,
    expression_similarity, find_nearest, scaled_transfer, style_transfer, ExpressionPcaBasis,
    ExpressionSpace, ExpressionTransferError,
};
pub use expressions::{
    ConstraintViolation, ExpressionBlend, ExpressionExt, ExpressionLibrary, ExpressionProvenance,
    FlameParamConstraints, NamedExpression,
};
pub use face_atlas::{
    blit_into_atlas, compute_atlas_stats, create_flame_face_atlas, extract_from_atlas,
    next_power_of_two, pack_regions, rasterize_atlas_layout, AtlasConfig, AtlasRect, AtlasRegion,
    AtlasStats, FaceAtlas, FaceAtlasError,
};
pub use face_normalization::{
    align_eye_line, align_frontal, apply_norm_transform, axis_angle_rotation, face_diagonal,
    format_norm_result, inter_pupil_distance, normalize_mesh, pca_axes, rotation_align,
    vertex_centroid, AlignMode, CenterMode, FaceNormError, NormConfig, NormResult, NormTransform,
};
pub use facs::{ActionUnit, AuMapping, FacsIntensity, FacsLibrary, FacsPresets, FacsToFlame};
pub use fitting::{
    fit_landmarks, FittingConfig, FittingError, FittingParams, FittingResult, FlameForward,
    FlameLandmarkFitter, LandmarkObservation, MockFlameForward, PinholeCamera,
};
pub use geodesic::{
    compute_geodesic_stats, dijkstra, geodesic_ball, geodesic_center, geodesic_center_sampled,
    geodesic_diameter, geodesic_voronoi, geodesic_weights, heat_geodesic, heat_geodesic_multi,
    heat_time_step, multi_source_dijkstra, pairwise_geodesic, smooth_geodesic_path, GeodesicConfig,
    GeodesicError, GeodesicField, GeodesicMesh, GeodesicStats, DEFAULT_CENTER_SAMPLES,
};
pub use gpu_buffers::{GpuBufferConfig, GpuMeshBuffers};
pub use head_geometry::{
    classify_vertex_region, compute_head_geometry_stats, convex_hull_2d, find_nearest_vertex,
    format_head_geometry_stats, format_head_measurements, frontal_face_area, frontal_silhouette,
    head_bounding_box, head_centroid, head_dist3, head_profile_xz, head_surface_area, head_volume,
    height_histogram, label_vertices_by_region, max_pairwise_distance, measure_head,
    principal_axis, project_to_plane, region_centroid, vertex_asymmetry_scores, vertices_in_region,
    HeadGeometryError, HeadGeometryStats, HeadMeasurements, HeadRegion,
};
pub use head_tracker::{
    compute_trajectory_stats, detect_pose_jumps, ema_smooth_trajectory, head_coverage_score,
    interpolate_missing_frames, one_euro_filter_sequence, resample_trajectory, rotation_velocity,
    segment_by_motion, slice_trajectory, sma_smooth_trajectory, HeadTrackFrame, HeadTrackPose,
    HeadTracker, HeadTrackerConfig, HeadTrackerError, HeadTrajectory, TrackingFilter,
    TrajectoryStats,
};
// `io::load_flame_model` is the file-loading primitive behind
// `FlameModel::load` (see `model.rs`); it is re-exported here too, purely
// for consistency with every other module in this crate root (all of which
// are flattened even where a nicer wrapper exists, e.g. `rodrigues` next to
// `FlameModel::forward`). `io_safetensors`'s pair has no inherent-method
// equivalent on `FlameModel` at all -- this re-export is their only
// crate-root entry point, not just a stylistic convenience.
pub use io::load_flame_model;
pub use io_safetensors::{load_flame_model_safetensors, save_flame_model_safetensors};
pub use landmarks::{
    BarycentricLandmark, Landmark, LandmarkExtractor, LandmarkGroup, NUM_LANDMARKS,
};
pub use lighting_model::{
    apply_rim_lighting, approximate_ambient_occlusion, lambertian_diffuse, phong_vertex,
    phong_vertex_point, reinhard_tone_map, shade_mesh_directional, shade_mesh_multi_light,
    shade_mesh_sh_lighting, studio_lighting, AmbientLight, DirectionalLight, LightingError,
    LightingResult, PhongMaterial, PointLight,
};
pub use mesh::{Mesh, MeshExportConfig};
pub use mesh_analysis::{
    compute_face_areas, compute_face_aspect_ratios, compute_mesh_quality, find_boundary_edges,
    is_manifold_mesh, FaceQualityStats, MeshQualityReport, VertexQualityStats,
};
pub use mesh_morphing::{
    apply_morph_targets, compute_delta_sequence, compute_morph_delta, delta_magnitudes,
    format_morph_clip, max_delta_magnitude, mean_delta_magnitude, morph_blend_n, morph_cosine,
    morph_cubic_hermite, morph_interpolate, morph_lerp, resample_morph_sequence,
    smooth_morph_sequence, MorphClip, MorphError, MorphInterpolation, MorphKeyframe, MorphLoopMode,
    MorphTarget, MorphTargetSet,
};
// Explicit re-export list (not `pub use mesh_ops::*`): `build_adjacency`,
// `laplacian_smooth`, and `taubin_smooth` also exist in `mesh_smoothing`
// with different signatures. A glob import here would be silently shadowed
// by the explicit `mesh_smoothing` re-exports below wherever names collide,
// making `mesh_ops`'s own versions of those three unreachable from the
// crate root without any compiler diagnostic. Naming every export
// explicitly (and omitting the three collisions) keeps that shadowing
// impossible to reintroduce by accident.
pub use mesh_ops::{
    compute_laplacian_cotangent, compute_laplacian_uniform, cotangent_smooth,
    find_boundary_vertices, laplacian_smooth_step, loop_subdivide, midpoint_subdivide,
    MeshOpsError, MeshSmoothingConfig, WeightMode,
};
pub use mesh_repair::{
    repair_mesh, MeshRepairConfig, MeshRepairError, MeshRepairExt, MeshRepairResult,
    MeshRepairStats,
};
pub use mesh_smoothing::{
    build_adjacency, build_cotan_adjacency, compute_mesh_volume, compute_smoothing_stats,
    cotan_laplacian_smooth, hc_laplacian_smooth, laplacian_smooth, restore_volume, taubin_smooth,
    HcLaplacianConfig, LaplacianConfig, SmoothingError, SmoothingStats, TaubinConfig,
};
pub use mesh_subdivision::{
    compute_mean_edge_length, compute_subdivision_stats, estimate_subdivided_vertex_count,
    format_subdivision_result, recompute_mesh_normals, subdivide_mesh, subdivide_once,
    validate_mesh_for_subdivision, SubdivisionConfig, SubdivisionError, SubdivisionResult,
    SubdivisionStats,
};
pub use model::{
    compute_normals_batch, compute_normals_into, recompute_batch_normals, rodrigues,
    BatchBufferPool, BatchedFlameOutput, FlameModel,
};
#[cfg(feature = "parallel")]
pub use model::{compute_normals_batch_par, recompute_batch_normals_par};
pub use multiresolution::{
    compute_vertex_normals, DecimationConfig, MeshDecimator, MeshLevel, MultiResMesh,
    MultiResMeshBuilder,
};
pub use normal_map::{Camera, NormalMapRenderer};
pub use param_sampler::{
    compute_sample_stats, flatten_params, van_der_corput, FlameParamsSampler, FlameParamsSpace,
    ParameterDimension, ParameterRange, SampleSetStats, SamplingStrategy,
};
pub use params::FlameParams;
pub use params_builder::FlameParamsBuilder;
pub use phoneme_animation::{
    apply_coarticulation, blend_phoneme_with_base, extract_expression_sequence,
    extract_jaw_sequence, format_phoneme_clip, format_phoneme_stats, generate_breath_animation,
    parse_phoneme_string, phoneme_clip_stats, smooth_phoneme_keyframes,
    synthesize_phoneme_animation, PhonemeClip, PhonemeError, PhonemeEvent, PhonemeKeyframe,
    PhonemeLibrary, PhonemeParams, PhonemeStats, Viseme, VisemeExpressionTargets,
};
pub use pose_estimation::{
    count_inliers, estimate_pitch_from_vertical, estimate_pose_weak_perspective,
    estimate_yaw_from_symmetry, reprojection_error, select_pitch_candidate, HeadPose, Landmark2D,
    Landmark3D, PitchReference, PointCorrespondence, PoseConfig, PoseEstimationError,
    PosePinholeCamera, PoseTracker,
};
pub use pose_prior::{
    aa_magnitude, aa_to_euler_approx, default_joint_limits, get_joint, set_joint,
    GaussianPosePrior, JointLimits, PosePriorError, PoseScorer, PoseValidityReport,
};
pub use retargeting::{
    apply_neutral_correction, compute_expr_basis_scales, compute_expression_scale_factors,
    compute_neutral_correction, compute_retargeting_stats, decompose_expression,
    recompose_expression, retarget_expression, retarget_sequence, smooth_expression_sequence,
    ExpressionRetargeter, RetargetingConfig, RetargetingError, RetargetingStats,
};
pub use rigid_alignment::{
    align_by_landmarks, align_by_weighted_landmarks, align_compute_stats, align_format_icp_result,
    align_format_stats, align_icp, align_nearest_neighbors, align_nearest_neighbors_filtered,
    align_procrustes, align_procrustes_rigid, align_rmse, AlignmentError, AlignmentStats,
    IcpConfig, IcpResult, SimilarityTransform,
};
pub use sampler::{sample_mesh_surface, SurfacePoint};
pub use sequence::{FlameSequence, SequenceIterator};
pub use shape_analysis::{
    compute_shape_distance, path_arc_length, shape_interpolation_path, ShapeAnalysisError,
    ShapeDistanceMetric, ShapeOutlierDetector, ShapeSpacePca, ShapeStatistics,
};
pub use statistical_shape_model::{
    ssm_build, ssm_component_sweep, ssm_components_for_variance, ssm_compute_stats, ssm_fit,
    ssm_format_model, ssm_format_stats, ssm_interpolate, ssm_most_variable_vertices, ssm_project,
    ssm_random_shape, ssm_reconstruct, ssm_reconstruction_error, ssm_vertex_std, ShapeParameters,
    SsmError, SsmStats, StatisticalShapeModel,
};
pub use symmetry::{
    analyze_symmetry, asymmetry_contribution, asymmetry_heatmap, blend_to_symmetric_params,
    blend_with_symmetric, generate_synthetic_symmetry_map, per_vertex_asymmetry, reflect_vertex,
    symmetrize_mesh, symmetrize_shape_params, top_asymmetric_pairs, validate_symmetry_map,
    SymmetryError, SymmetryMap, SymmetryReport,
};
pub use texture_baking::{
    apply_uv_padding, bake, bake_attribute, bake_normals, bake_positions, bake_vertex_colors,
    baked_texture_to_rgb, compute_face_uv_areas, compute_uv_mask, format_bake_stats, BakeAttribute,
    BakeConfig, BakeError, BakedTexture, TriangleUv, UvCoord as BakeUvCoord,
};
pub use timeline::{
    timeline_from_clip, AnimationTimeline, BlendMode, TimelineError, TimelineLayer, TimelineMarker,
};
pub use traits::{
    DefaultSampler, FlameNormalMapProvider, MeshSurfaceSampler, NormalMapProvider, SurfaceSample,
};
pub use uv::{UvAccessor, UvChartInfo, UvMeshExt};
pub use uv_texture::{FilterMode, TextureMap, TextureMeshExt, UvTextureSampler, WrapMode};
pub use vertex_mask::{FaceRegion, VertexMask, VertexMaskError};
pub use visibility_culling::{
    compute_face_visibility, compute_greedy_view_selection, compute_multi_view_visibility,
    compute_optimal_view_coverage, compute_per_view_coverage, compute_per_view_visibility,
    compute_vertex_visibility, compute_visibility_stats, find_view_dependent_vertices,
    format_multi_view_stats, format_visibility_stats, select_greedy_covering_views,
    select_maximally_covering_views, select_top_coverage_views, FaceVisibility,
    MultiViewVisibility, VertexVisibility, VisibilityCullerConfig, VisibilityError,
    VisibilityStats,
};
pub use visualize::{
    render_joints_svg, render_mesh_with_joints, render_wireframe, save_svg, SvgCamera,
    WireframeOptions,
};
pub use warp_field::{
    build_vertex_adjacency, compute_warp_stats, find_large_displacements,
    laplacian_smooth_warp_field, linear_combination, per_vertex_magnitude, WarpField,
    WarpFieldError, WarpFieldSequence, WarpFieldStats, WarpMask,
};
pub mod gaze_controller;
pub use gaze_controller::{
    gz_angular_velocity, gz_blink_waveform, gz_compute_stats, gz_convergence_angle_deg,
    gz_detect_blinks, gz_detect_fixations, gz_detect_saccades, gz_dispersion, gz_format_stats,
    gz_listing_axis, gz_listing_rotation, gz_synthesize_blinks, gz_vergence_from_iod, BlinkEvent,
    BlinkPhase, FixationEvent, GazeController, GazeControllerConfig, GazeControllerError,
    GazeDirection, GazeEventKind, GazeFrame, GazeStats, SaccadeEvent,
};
pub mod spectral_analysis;
pub use spectral_analysis::{
    spec_build_combinatorial_laplacian, spec_build_cotangent_laplacian, spec_cluster_vertices,
    spec_compute_stats, spec_format_config, spec_format_stats, spec_gram_schmidt,
    spec_high_pass_filter, spec_laplacian_matvec, spec_laplacian_smooth, spec_low_pass_filter,
    spec_normalize_laplacian, spec_power_iteration, spec_project, spec_rayleigh_quotient,
    spec_reconstruct, spec_smoothness, spec_taubin_smooth, LaplacianKind, MeshLaplacian,
    SpectralBasis, SpectralConfig, SpectralError, SpectralSignal, SpectralStats,
};

// ---------------------------------------------------------------------------
// Crate-root re-export wiring
// ---------------------------------------------------------------------------
//
// The `pub use` blocks above are this crate's public surface: every other
// module gets one, flattening its types and functions to the crate root.
// A name missing from one of them still compiles (the item stays reachable
// via its module path) and so is never caught by `cargo check` -- it only
// shows up as an inconsistency on the next audit. Pin every re-export added
// here so a future edit that silently drops one (or changes its signature)
// fails this file to compile instead.
#[cfg(test)]
mod crate_root_reexport_tests {
    use std::collections::HashMap;
    use std::path::Path;

    #[test]
    fn retar_build_expression_mirror_matrix_is_at_crate_root() {
        type RetarBuildMirrorFn =
            fn(&[f32], &[usize], usize) -> Result<Vec<f32>, crate::RetargetError>;
        let _: RetarBuildMirrorFn = crate::retar_build_expression_mirror_matrix;
    }

    #[test]
    fn geodesic_additions_are_at_crate_root() {
        type HeatGeodesicMultiFn = fn(
            &crate::GeodesicMesh,
            &[usize],
            usize,
            Option<f32>,
        )
            -> Result<crate::GeodesicField, crate::GeodesicError>;
        type GeodesicCenterSampledFn = fn(
            &crate::GeodesicMesh,
            &[usize],
            usize,
            &crate::GeodesicConfig,
        ) -> Result<usize, crate::GeodesicError>;

        let _: HeatGeodesicMultiFn = crate::heat_geodesic_multi;
        let _: fn(&crate::GeodesicMesh) -> f32 = crate::heat_time_step;
        let _: GeodesicCenterSampledFn = crate::geodesic_center_sampled;
        let _: usize = crate::DEFAULT_CENTER_SAMPLES;
    }

    #[test]
    fn pose_estimation_additions_are_at_crate_root() {
        let _: fn([f32; 2], f32) -> f32 = crate::select_pitch_candidate;
        let _: Option<crate::PitchReference> = None;
    }

    #[test]
    fn visibility_culling_additions_are_at_crate_root() {
        type PerViewCoverageFn = fn(
            &crate::Mesh,
            &[crate::Camera],
            &crate::VisibilityCullerConfig,
        ) -> Result<Vec<f32>, crate::VisibilityError>;
        type PerViewVisibilityFn =
            fn(
                &crate::Mesh,
                &[crate::Camera],
                &crate::VisibilityCullerConfig,
            ) -> Result<Vec<crate::VertexVisibility>, crate::VisibilityError>;
        type GreedyViewSelectionFn = fn(
            &crate::Mesh,
            &[crate::Camera],
            &crate::VisibilityCullerConfig,
            usize,
        ) -> Result<Vec<usize>, crate::VisibilityError>;

        let _: PerViewCoverageFn = crate::compute_per_view_coverage;
        let _: PerViewVisibilityFn = crate::compute_per_view_visibility;
        let _: GreedyViewSelectionFn = crate::compute_greedy_view_selection;
        let _: fn(&[f32], usize) -> Vec<usize> = crate::select_top_coverage_views;
        let _: fn(&[crate::VertexVisibility], usize) -> Vec<usize> =
            crate::select_greedy_covering_views;
    }

    #[test]
    fn io_additions_are_at_crate_root() {
        type SaveFlameModelSafetensorsFn = fn(
            &crate::FlameModel,
            &Path,
            Option<&HashMap<String, String>>,
        ) -> Result<(), crate::FlameError>;

        let _: fn(&Path) -> Result<crate::FlameModel, crate::FlameError> = crate::load_flame_model;
        let _: fn(&Path) -> Result<crate::FlameModel, crate::FlameError> =
            crate::load_flame_model_safetensors;
        let _: SaveFlameModelSafetensorsFn = crate::save_flame_model_safetensors;
    }

    #[test]
    fn newly_wired_types_are_at_crate_root() {
        let _: Option<crate::ExpressionProvenance> = None;
        let _: Option<crate::VisemeExpressionTargets> = None;
        let _: Option<crate::VertexMaskError> = None;
        let _: Option<crate::HeadTrackPose> = None;
        let _: Option<crate::MeshDecimator> = None;
        let _: Option<crate::FlameNormalMapProvider> = None;
        let _: Option<crate::FlameLandmarkFitter<'static>> = None;
    }
}
