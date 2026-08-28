//! OxiGAF CLI library interface.
//!
//! This is the crate's **single module root**. `src/main.rs` is a thin shim
//! that `use`s this library; it declares no modules of its own. Anything
//! reachable from the `oxigaf` binary — and anything an integration test
//! under `tests/` can drive — is declared exactly once, here.
//!
//! # Adding a module
//!
//! 1. `pub mod <name>;` in the list below.
//! 2. If it should be a subcommand, add a handler under
//!    [`commands`] and one variant to [`cli::Command`].
//!
//! Do **not** add a `mod` declaration to `main.rs`: a module declared in
//! both roots is compiled twice into two unrelated types, which is how the
//! forty-plus "library-only, unreachable from the binary" modules came
//! about in the first place.
//!
//! # Subcommand coverage
//!
//! Every tool module below is reachable from the shipped `oxigaf` binary
//! through one of the twenty subcommand families, grouped by what a caller
//! is trying to do rather than one flag per module:
//!
//! | Family | Modules it exposes |
//! |--------|--------------------|
//! | `anim` | [`animation_export`] |
//! | `analyze` | [`color_calibration`], [`diff_tool`], [`evaluation_suite`] |
//! | `batch` | [`batch_processor`] |
//! | `camera` | [`camera_path_tool`], [`arcball`] |
//! | `dataset` | [`dataset_tools`] |
//! | `inspect` | [`model_inspector`], [`memory_estimator`], [`export_ply`], [`export_pointcloud_stats`] |
//! | `monitor` | [`dashboard`] |
//! | `perf` | [`benchmark_suite`] |
//! | `pipeline` | [`stages`] |
//! | `preset` | [`config_presets`] |
//! | `preview` | [`preview`] |
//! | `profile` | [`profiling_report`] |
//! | `quality` | [`quality_checker`] |
//! | `report` | [`experiment_report`] |
//! | `runs` | [`workspace_manager`] |
//! | `scene` | [`scene_analyzer`], [`scene_merge`], [`scene_optimizer`], [`scene_streaming`], [`cloud_registration`], [`geometry_tools`], [`mod@filter_gaussians`], [`gaussian_deduplicator`], [`gaussian_compressor`], [`lod_generator`], [`format_converter`] |
//! | `sweep` | [`parameter_sweep`] |
//! | `training` | [`training_monitor`], [`report_generator`], [`resume_analyzer`], [`telemetry`] |
//! | `video` | [`video_export`] |
//! | `workspace` | [`checkpoint_browser`] |
//!
//! [`progress_types`] and [`parallel_render`] are consumed by the handlers
//! themselves rather than exposed as subcommands: they are the progress
//! bars, timing tables and frame scheduler those commands run on.

// Test code builds configs via Default then overrides individual fields to
// exercise boundary conditions with intentionally invalid values.  Using the
// struct-update syntax would obscure which single field is under test.
#![allow(clippy::field_reassign_with_default)]
// The no-unwrap policy applies to every production path in this crate, not
// just the binary target (`main.rs` carries the same three lints). Test code
// is exempt — `cfg(test)` is set for the whole crate when compiling the test
// harness, so the deny only binds in ordinary builds.
#![cfg_attr(not(test), deny(clippy::unwrap_used))]
#![cfg_attr(not(test), deny(clippy::expect_used))]
#![warn(clippy::panic)]

pub mod animation_export;
pub mod arcball;
pub mod assets;
pub mod batch_processor;
pub mod benchmark;
pub mod benchmark_suite;
pub mod cache;
pub mod camera_path_tool;
pub mod checkpoint_browser;
pub mod cli;
pub mod cloud_registration;
pub mod color_calibration;
pub mod commands;
pub mod compare;
pub mod config;
pub mod config_cmd;
pub mod config_presets;
pub mod convert;
pub mod dashboard;
pub mod dataset_tools;
pub mod diff_tool;
pub mod dry_run;
pub mod error;
pub mod evaluation_suite;
pub mod experiment_report;
pub mod export;
pub mod export_gltf;
pub mod export_mesh;
pub mod export_ply;
pub mod export_pointcloud;
pub mod export_pointcloud_stats;
pub mod filter_gaussians;
pub mod format_converter;
pub mod gaussian_compressor;
pub mod gaussian_deduplicator;
pub mod geometry_tools;
pub mod info;
pub mod interactive;
pub mod json_output;
pub mod lod_generator;
pub mod log_rotation;
pub mod memory_estimator;
pub mod metrics;
pub mod model_inspector;
pub mod output;
pub mod parallel_render;
pub mod parameter_sweep;
pub mod pipeline;
pub mod preview;
pub mod profiling_report;
pub mod progress;
pub mod progress_types;
pub mod quality_checker;
pub mod report_generator;
pub mod resume_analyzer;
pub mod scene_analyzer;
pub mod scene_merge;
pub mod scene_optimizer;
pub mod scene_streaming;
pub mod stages;
pub mod summary;
pub mod telemetry;
pub mod training_monitor;
pub mod verbosity;
pub mod video_export;
pub mod workspace_manager;

// Re-export for tests
pub use animation_export::{
    build_animation_meta, compute_animation_stats, compute_frame_stats, concatenate_animations,
    export_animation_json, format_animation_summary, interpolate_frames, load_animation_json,
    loop_animation, resample_animation, reverse_animation, subsample_animation, trim_animation,
    AnimExportConfig, AnimationError, AnimationFrame, AnimationMeta, AnimationSequence,
    AnimationStats, FrameStats,
};
pub use arcball::ArcballCamera;
pub use batch_processor::{
    compute_execution_waves, jobs_from_directory, jobs_from_file_list, merge_batches, BatchConfig,
    BatchError, BatchJob, BatchProcessor, BatchStats, JobExecutor, JobResult,
};
pub use benchmark_suite::{
    bench_gaussian_bbox, bench_gaussian_centroid, bench_sort_f32, bench_vec_dot, bench_vec_sum,
    build_benchmark_result, build_suite_report, compare_benchmarks, create_default_suite,
    filter_outliers, format_benchmark_result, format_duration_ns, format_suite_report,
    run_benchmark, run_suite_entry, time_fn, BenchmarkComparison, BenchmarkConfig, BenchmarkEntry,
    BenchmarkError, BenchmarkResult, BenchmarkSuite, SuiteReport,
};
pub use camera_path_tool::{
    blend_paths, catmull_rom, compute_path_stats, figure_eight_path, keyframe_path, orbit_path,
    path_from_json, path_to_json, path_velocities, smooth_path, spiral_orbit_path,
    turntable_preset, zoom_in_path, CameraPath, CameraPathError, CameraPose, EasingType,
    PathConfig, PathKeyframe, PathStats,
};
pub use checkpoint_browser::{
    checkpoint_spacing_stats, compare_checkpoints, describe_checkpoint, estimate_steps_to_psnr,
    extract_tags_from_path, find_psnr_elbow, format_checkpoint_diff, format_checkpoint_table,
    format_spacing_stats, parse_psnr_from_path, parse_step_from_path, psnr_trend,
    BrowserCheckpoint, BrowserConfig, BrowserError, BrowserFilter, BrowserSort, CheckpointBrowser,
    CheckpointDiff, SpacingStats,
};
pub use cloud_registration::{
    apply_registration_transform, compute_centroid_3d, compute_initial_rmse,
    compute_registration_stats, estimate_transform_umeyama_approx, filter_correspondences,
    find_correspondences, format_registration_result, icp_step, register_point_clouds,
    subsample_positions, Correspondence, RegistrationConfig, RegistrationError, RegistrationResult,
    RegistrationStats, RegistrationTransform,
};
pub use color_calibration::{
    cal_apply_ccm, cal_apply_gamma, cal_apply_gamma_channel, cal_apply_gamma_inv,
    cal_apply_gamma_inv_channel, cal_apply_white_balance, cal_delta_e_2000, cal_delta_e_76,
    cal_evaluate, cal_format_stats, cal_lab_to_xyz, cal_macbeth_patches, cal_solve_ccm,
    cal_srgb_to_lab, cal_srgb_to_xyz, cal_xyz_to_lab, cal_xyz_to_srgb, CalibrationError,
    CalibrationMatrix, CalibrationStats, ColorPatch, GammaProfile, WhiteBalance,
};
pub use config_presets::{
    apply_override, apply_overrides, PresetDiff, PresetError, TrainingPreset, TrainingPresetName,
};
pub use dashboard::{
    format_spark_line, DashboardConfig, DashboardPanel, DashboardRenderer, DashboardState,
    MetricBar,
};
pub use dataset_tools::{
    apply_split, compute_dataset_stats, compute_split_stats, filter_by_size, find_size_duplicates,
    format_stats_table, load_split, save_split, shuffle_indices, split_dataset, validate_dataset,
    validate_split, DatasetError, DatasetScanner, DatasetSplit, DatasetSplitStrategy, DatasetStats,
    FileEntry, FileType, SplitConfig, SplitStats,
};
pub use diff_tool::{
    change_score_histogram, compute_field_diff, detect_regression, diff_models,
    diff_models_variable, diff_sequence, format_field_diff, format_field_diff_header,
    format_model_diff, largest_position_changes, opacity_changes, per_gaussian_change_score,
    snapshots_approximately_equal, summarize_progress, DiffConfig, DiffError, FieldDiff, ModelDiff,
    ModelSnapshot, ProgressSummary, RegressionReport,
};
pub use error::CliError;
pub use evaluation_suite::{
    eval_compare, eval_convolve, eval_downsample_2x, eval_format_comparison,
    eval_format_suite_result, eval_format_view_result, eval_gaussian_kernel_11, eval_lpips_approx,
    eval_mae, eval_psnr, eval_psnr_histogram, eval_psnr_percentiles, eval_rmse, eval_single_view,
    eval_sobel, eval_ssim, eval_ssim_ms, eval_suite, EvalComparison, EvalConfig, EvalError,
    EvalMetricKind, EvalSuiteResult, EvalTestItem, ViewEvalResult,
};
pub use experiment_report::{
    generate_svg_line_chart, ExperimentComparison, ExperimentMetrics, HtmlReportConfig,
    HtmlReportGenerator, ReportError,
};
pub use export_ply::{
    ply_build_header, ply_compute_scene_stats, ply_export_scene, ply_format_stats,
    ply_format_write_stats, ply_import_scene, ply_parse_element_count, ply_parse_format,
    ply_parse_properties, ply_read, ply_write, ply_write_ascii, ply_write_binary, PlyError,
    PlyExportParams, PlyFlatSlices, PlyFormat, PlyGaussian, PlyReadStats, PlySceneData,
    PlySceneStats, PlyWriteStats,
};
pub use export_pointcloud_stats::PointCloudStats;
pub use filter_gaussians::{
    compute_scene_stats, filter_anisotropic, filter_dominant, filter_gaussians,
    filter_gaussians_multi, filter_gaussians_pipeline, filter_spatial_outliers, filter_transparent,
    prune_gaussians, FilterCriterion, FilterError, FilterResult, GaussianData, PruningConfig,
    SceneStats,
};
pub use format_converter::{
    compute_conversion_stats, convert, filter_valid, from_binary, from_csv, from_json, to_binary,
    to_csv, to_json, validate_record, BinaryHeader, ConversionStats, ConvertError, FileFormat,
    GaussianRecord, BINARY_MAGIC, BINARY_VERSION,
};
pub use gaussian_compressor::{
    gc_apply_mask_flat, gc_cluster_residuals, gc_compress, gc_compute_prune_mask, gc_compute_stats,
    gc_decompress, gc_format_config, gc_format_stats, gc_kmeans_plus_plus_init,
    gc_kmeans_positions, gc_prune_to_topn, CompressedScene, CompressionConfig, CompressionStats,
    CompressorError, DecompressedScene, GcSceneSlices, KMeansConfig, QuantizationPrecision,
    QuantizedAttribute, ScenePruningConfig,
};
pub use gaussian_deduplicator::{
    gd_analyze_duplicates, gd_apply_mask, gd_apply_scalar_mask, gd_are_duplicates,
    gd_build_remove_mask, gd_build_report, gd_compute_stats, gd_deduplicate,
    gd_find_duplicates_brute, gd_find_duplicates_spatial, gd_format_config, gd_format_report,
    gd_format_stats, gd_hash_cell, gd_pick_representative, gd_world_to_cell, DedupConfig,
    DedupKeepPolicy, DedupReport, DedupResult, DedupStats, DeduplicatorError, DuplicateGroup,
    GdDeduplicateInput, GdSceneSlices, SpatialHashMap,
};
pub use geometry_tools::{
    center_at_origin, cloud_distance, compute_bounding_sphere, compute_centroid,
    compute_gaussian_bbox, compute_geometry_stats, compute_obb, filter_by_bbox, filter_by_sphere,
    mean_gaussian_scale, nearest_neighbor_distances, normalize_to_unit_cube, rescale_gaussians,
    spatial_coverage, transform_positions, transform_rotations, transform_scales, BoundingSphere,
    GaussianBBox, GeometryError, GeometryStats, ObbResult, RigidTransform,
};
pub use interactive::InteractiveController;
pub use lod_generator::{
    compute_lod_stats, compute_opacity_values, estimate_lod_memory, extract_subset,
    find_optimal_reduction_ratios, format_lod_stats, generate_lod_chain, generate_lod_level,
    merge_lod_levels, select_random_indices, select_spatial_grid_indices,
    select_top_opacity_indices, select_uniform_indices, LodChain, LodConfig, LodError,
    LodInputSlices, LodLevel, LodSelector, LodStats, LodStrategy,
};
pub use memory_estimator::{
    compare_memory_configs, estimate_gaussian_layout, estimate_memory, estimate_model_weights,
    estimate_render_buffers, estimate_training_memory, format_memory_estimate,
    max_gaussians_for_vram, mem_format_bytes, memory_breakdown_percent, sh_coefficients,
    GaussianLayout, MemBreakdown, MemEstimateConfig, MemEstimateError, MemoryDelta, MemoryEstimate,
    RenderBuffers, TrainingMemory,
};
pub use model_inspector::{
    analyze_color_distribution, density_voxel_grid, dump_gaussians_csv, find_high_anisotropy,
    find_large_gaussians, find_low_opacity, find_spatial_outliers, format_inspection_report,
    histogram, inspect_model, percentile, query_aabb, query_knn, query_sphere, BoundingBox3d,
    ColorDistribution, GaussianProperties, InspectableModel, InspectionReport, InspectorError,
    QueryResult,
};
pub use parallel_render::{
    FrameTask, ParallelRenderConfig, ParallelRenderResult, ParallelRenderer,
};
pub use parameter_sweep::{
    format_sweep_summary, format_sweep_trial, hyperband_bracket, sweep_grid_indices,
    sweep_param_importance, sweep_sample_continuous, sweep_sample_discrete,
    sweep_surrogate_predict, ParamSpec, ParamValue, ParameterSweep, SweepConfig, SweepError,
    SweepStrategy, SweepSummary, SweepTrial,
};
pub use preview::{CameraAction, KeyBindings, KeyCode, PreviewConfig, PreviewController};
pub use profiling_report::{
    format_bytes, format_duration_ms, format_throughput, PhaseRecord, PhaseStats,
    ProfilingCollector, ProfilingConfig, ProfilingError, ProfilingReport,
};
pub use progress_types::{BatchProgress, OperationSpinner, TimingReport, TrainingProgress};
pub use quality_checker::{
    check_quality, check_quality_batch, compute_histogram, compute_mae, compute_mse, compute_psnr,
    compute_quality_metrics, compute_ssim, detect_artifacts, error_map, error_map_to_heatmap,
    histogram_distance, is_blank_image, psnr_from_mse, rgba_u8_to_rgb_f32, ArtifactReport,
    BatchQualityReport, ImageQualityMetrics, QualityError, QualityReport, QualityThresholds,
};
pub use report_generator::{
    compute_trend, downsample_series, format_metric_table, generate_training_report,
    render_html_report, render_markdown_report, render_text_report, series_stats, svg_line_chart,
    write_report, ChartData, ChartSeries, GeneratorError, MetricSummary, MetricTrend,
    ReportBuilder, ReportFormat, ReportGeneratorConfig, ReportPage, ReportSection, SectionContent,
};
pub use resume_analyzer::{
    analyze_checkpoints, analyze_checkpoints_default, CheckpointMetadata, CheckpointScanner,
    CheckpointScorer, ResumeError, ResumeRecommendation, ScoringWeights,
};
pub use scene_analyzer::{
    analyze_scene, compare_scenes, compute_color_stats, compute_opacity_stats,
    compute_quality_score, compute_scale_stats, compute_spatial_stats, AnalysisError, ColorStats,
    OpacityStats, ScaleStats, SceneComparison, SceneData, SceneReport, SpatialStats,
};
pub use scene_merge::{
    apply_transform, apply_transform_rotation, concatenate_scenes, find_duplicates,
    merge_at_boundary, merge_scenes, merge_scenes_with_stats, GaussianEntry, MergeError,
    MergeStats, SceneGaussians, SceneMergeConfig,
};
// `scene_optimizer` was the one scene module with no re-export block, so
// `OpacitySpace` — the type that decides whether an opacity array is read as
// logits or as already-activated probabilities, and therefore whether
// `--prune-opacity 0.5` keeps half a scene or all of it — was reachable only
// through the module path while its siblings' types were not. Every public
// item of the module is surfaced here, `OpacitySpace` included.
pub use scene_optimizer::{
    so_apply_keep_mask_1d, so_apply_keep_mask_3d, so_apply_keep_mask_4d, so_apply_keep_mask_nd,
    so_clamp_scales, so_clip_to_aabb, so_clip_to_sphere, so_compute_snapshot, so_deduplicate_near,
    so_format_config, so_format_report, so_format_snapshot, so_format_step_result, so_morton_code,
    so_morton_interleave, so_normalize_opacity, so_profile_config, so_prune_by_opacity,
    so_quantize_position, so_quick_optimize, so_reorder_by_indices, so_sort_morton,
    so_top_n_by_opacity, GaussianArrays, OpacitySpace, OptimizationPipeline, OptimizationProfile,
    OptimizationReport, OptimizationStep, OptimizationStepResult, OptimizedScene, OptimizerError,
    SceneOptimizerConfig, SceneSnapshot,
};
pub use scene_streaming::{
    ss_chunk_bounds, ss_chunk_id, ss_chunk_scene, ss_compute_priority, ss_compute_scene_bounds,
    ss_compute_stats, ss_distance_to_aabb, ss_format_config, ss_format_stats,
    ss_frustum_cull_points, ss_frustum_cull_spheres, ss_lod_subsample_indices, ss_select_lod,
    ss_sort_chunks_by_priority, ChunkCache, ChunkCellIndex, ChunkGridDims, LoadPriority,
    StreamingChunk, StreamingConfig, StreamingError, StreamingScene, StreamingStats, ViewFrustum,
};
pub use telemetry::{
    tel_compute_latency_stats, tel_detect_regression, tel_detect_spikes, tel_format_event,
    tel_format_latency_stats, tel_format_report, tel_generate_report, tel_stats_by_category,
    tel_stats_by_label, LatencyStats, RollingWindow, TelemetryCategory, TelemetryCollector,
    TelemetryConfig, TelemetryError, TelemetryEvent, TelemetryReport, ThroughputTracker,
};
pub use training_monitor::{
    compute_throughput, detect_divergence, ema_smooth, format_elapsed, format_eta,
    format_status_line, format_training_summary, linear_regression, loss_percentile,
    robust_smooth_loss, sma_smooth, summarize_training, ImprovementTracker, MonitorConfig,
    MonitorError, MonitorSnapshot, ThroughputStats, TrainingEvent, TrainingMonitor,
    TrainingSummary,
};
pub use video_export::{
    generate_html_viewer, FrameCollector, FrameMetadata, HtmlViewerConfig, VideoExportConfig,
    VideoExportResult, VideoFormat, VideoManifest,
};
pub use workspace_manager::{
    ws_checkpoint_size, ws_compute_stats, ws_current_timestamp, ws_format_stats,
    ws_format_status_counts, ws_format_summary, ws_format_table, ws_list_checkpoints,
    ws_prune_checkpoints, ws_timestamped_name, ws_validate_name, Workspace, WorkspaceConfig,
    WorkspaceError, WorkspaceManager, WorkspaceStats, WorkspaceStatus, WorkspaceStatusCounts,
};

#[cfg(test)]
mod tests {
    // A glob import of the crate root: this is what makes the regression
    // test below meaningful. `GaussianArrays` lives in `scene_optimizer`, so
    // `super::scene_optimizer::GaussianArrays` would resolve whether or not
    // it is re-exported here — only the *bare* name reaching this glob
    // proves the `pub use scene_optimizer::{ .. }` list above actually
    // carries it.
    use super::*;

    /// Regression: `scene_optimizer` was the one scene module whose
    /// re-export block omitted a type its own public function needs —
    /// `so_quick_optimize` takes `GaussianArrays<'_>`, but `GaussianArrays`
    /// itself was not in the `pub use` list, so a caller reaching
    /// `so_quick_optimize` through the crate root had no crate-root path to
    /// its own argument type. This would fail to compile if the re-export
    /// regressed, since `GaussianArrays` here is the bare name brought in by
    /// `use super::*` above, not a `scene_optimizer::` qualified path.
    #[test]
    fn gaussian_arrays_is_reexported_alongside_so_quick_optimize() {
        let n = 2;
        let sh_channels = 3;
        let positions: Vec<f32> = (0..n * 3).map(|i| i as f32 * 0.01).collect();
        let rotations: Vec<f32> = (0..n).flat_map(|_| [0.0f32, 0.0, 0.0, 1.0]).collect();
        let scales: Vec<f32> = vec![0.05f32; n * 3];
        let opacities: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5 - 1.0).collect();
        let sh_coefficients: Vec<f32> = vec![0.0f32; n * sh_channels];

        let arrays = GaussianArrays {
            positions: &positions,
            rotations: &rotations,
            scales: &scales,
            opacities: &opacities,
            sh_coefficients: &sh_coefficients,
        };

        let result = so_quick_optimize(arrays, n, sh_channels, OptimizationProfile::Quality);
        assert!(
            result.is_ok(),
            "so_quick_optimize with a valid GaussianArrays must succeed: {:?}",
            result.err()
        );
    }
}
