//! # oxigaf-trainer
//!
//! GAF optimization pipeline — iterative denoising distillation.
//!
//! Provides:
//! - Gaussian initialization on FLAME mesh surfaces
//! - Per-parameter Adam optimizer with group-wise learning rates
//! - Photometric + structural loss computation (L1, SSIM)
//! - Adaptive density control (split / clone / prune)
//! - Checkpoint save / load (JSON + flat f32 arrays)
//! - Metric tracking (PSNR, SSIM history)
//! - Main training loop orchestrating render ↔ diffusion distillation

#![cfg_attr(not(test), deny(clippy::unwrap_used))]
#![cfg_attr(not(test), deny(clippy::expect_used))]
// Test code builds configs via Default then overrides fields to test boundary
// conditions with invalid values.  Using struct-update syntax would obscure
// which single field is being exercised in each test case.
#![allow(clippy::field_reassign_with_default)]
// Complex return type in internal pruning helper (5-tuple of Vec<f32>).
// A named struct would require a public type for a purely internal function.
#![allow(clippy::type_complexity)]

pub mod activation_maps;
pub mod adaptive_loss;
pub mod adaptive_loss_weighting;
pub mod anomaly_detection;
pub mod augmentation;
pub mod callback;
pub mod camera_sampling;
pub mod checkpoint;
pub mod checkpoint_interpolation;
pub mod checkpoint_manager;
pub mod config;
pub mod continual_learning;
pub mod contrastive_learning;
pub mod contrastive_loss;
pub mod convergence_analysis;
pub mod curriculum;
pub mod curriculum_learning;
pub mod data_augmentation;
pub mod data_parallel;
pub mod density;
pub mod diagnostics;
pub mod diffusion_target;
pub mod domain_adaptation;
pub mod ema;
pub mod feature_bank;
pub mod few_shot_adaptation;
pub mod gradient_accumulation;
pub mod gradient_clipping;
pub mod gradient_flow;
pub mod gradient_surgery;
pub mod hparam_search;
pub mod init;
pub mod knowledge_distillation;
pub mod layer_freezing;
pub mod loss;
pub mod loss_landscape;
pub mod loss_reweighting;
pub mod lpips;
pub mod lr_scheduler;
pub mod meta_learning;
pub mod metrics;
pub mod mixed_precision;
pub mod multi_resolution_loss;
pub mod noise_injection;
pub mod ohem;
pub mod online_hard_example_mining;
pub mod online_learning;
pub mod optimizer;
pub mod pose_conditioning;
pub mod profiler_integration;
pub mod progressive_training;
pub mod pruning;
pub mod regularization;
pub mod session_recorder;
pub mod spectral_norm;
pub mod synthetic_data;
pub mod tensorboard;
pub mod trainer;
pub mod training_config;
pub mod uncertainty_estimation;
pub mod validation_split;
pub mod video;
pub mod view_importance;
pub mod view_scheduler;
pub mod view_synthesis_eval;
pub mod weight_averaging;

use thiserror::Error;

/// Current checkpoint format version.
pub const CHECKPOINT_VERSION: u32 = 1;

/// Errors produced by the trainer subsystem.
#[derive(Debug, Error)]
pub enum TrainerError {
    // ---- Initialization ----
    #[error("Initialization error: {0}")]
    Init(String),

    #[error("Empty model: no Gaussians to optimize")]
    EmptyModel,

    #[error("Mesh has no faces for Gaussian initialization")]
    EmptyMesh,

    // ---- Training ----
    #[error("Training error: {0}")]
    Training(String),

    // ---- Numerical Issues ----
    #[error("NaN detected in {parameter}: index {index}")]
    NanDetected { parameter: String, index: usize },

    #[error("Infinity detected in {parameter}: index {index}")]
    InfDetected { parameter: String, index: usize },

    #[error("Gradient explosion: norm {norm:.2e} exceeds threshold {threshold:.2e}")]
    GradientExplosion { norm: f32, threshold: f32 },

    // ---- Configuration ----
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Parameter out of range: {param} = {value}, expected {expected}")]
    ParameterOutOfRange {
        param: String,
        value: String,
        expected: String,
    },

    // ---- Checkpoint ----
    #[error("Checkpoint error: {0}")]
    Checkpoint(String),

    #[error("Checkpoint corrupted: {0}")]
    CheckpointCorrupted(String),

    #[error("Checkpoint version mismatch: found {found}, expected {expected}")]
    CheckpointVersionMismatch { found: u32, expected: u32 },

    #[error("Checkpoint data mismatch: {field} has length {actual}, expected {expected}")]
    CheckpointDataMismatch {
        field: String,
        actual: usize,
        expected: usize,
    },

    // ---- Optimizer ----
    #[error("Optimizer error: {0}")]
    Optimizer(String),

    #[error("Gradient buffer size mismatch: expected {expected}, got {actual}")]
    GradientSizeMismatch { expected: usize, actual: usize },

    // ---- Loss ----
    #[error("Loss computation error: {0}")]
    Loss(String),

    #[error("Image dimension mismatch: expected {expected}, got {actual}")]
    ImageDimensionMismatch { expected: usize, actual: usize },

    // ---- Density Control ----
    #[error("Density control error: {0}")]
    DensityControl(String),

    #[error("Model size mismatch: expected {expected}, got {actual}")]
    ModelSizeMismatch { expected: usize, actual: usize },

    // ---- GPU/Memory ----
    #[error("GPU out of memory: requested {requested} bytes, available {available}")]
    GpuOom { requested: usize, available: usize },

    #[error("GPU buffer overflow: {0}")]
    GpuBufferOverflow(String),

    // ---- Diffusion ----
    #[error("Diffusion pipeline not loaded")]
    DiffusionNotLoaded,

    #[error("View generation failed for camera {camera_idx}: {reason}")]
    ViewGenerationFailed { camera_idx: usize, reason: String },

    // ---- External Errors ----
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON serialization error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Render error: {0}")]
    Render(#[from] oxigaf_render::RenderError),

    #[error("Diffusion error: {0}")]
    Diffusion(#[from] oxigaf_diffusion::DiffusionError),

    #[error("FLAME error: {0}")]
    Flame(#[from] oxigaf_flame::FlameError),

    // ---- Video / Sequence ----
    #[error("Sequence error: {0}")]
    SequenceError(String),
}

// ---- Re-exports ----

pub use multi_resolution_loss::{
    format_mr_result, format_mr_stats, mr_compute_loss, mr_compute_stats, mr_downsample,
    mr_gaussian_blur_3x3, mr_gradient_l1_loss, mr_l1_loss, mr_l2_loss, mr_laplacian_loss,
    mr_laplacian_pyramid, mr_level_loss, mr_sobel_magnitude, mr_ssim_loss, mr_upsample,
    ImagePyramid, MultiResLossConfig, MultiResLossError, MultiResLossResult, MultiResLossType,
    MultiResStats, PyramidLevel,
};

pub use activation_maps::{
    blend_maps, compute_activation_stats, compute_cam, finite_difference_saliency, intersect_maps,
    overlay_on_image, score_weighted_attribution, smooth_map, sobel_saliency, union_maps,
    ActivationMap, ActivationMapError, ActivationStats, AttributionConfig, AttributionNorm,
};
pub use adaptive_loss::{
    AdaptiveLossController, AdaptiveLossWeights, GradNormEntry, GradNormTracker, LossComponent,
    LossHistory, WeightingStrategy,
};
pub use anomaly_detection::{
    anom_check_convergence, anom_check_gradient_norm, anom_check_gradient_numerical,
    anom_check_loss_divergence, anom_check_loss_spike, anom_check_mode_collapse,
    anom_check_numerical, anom_check_opacity_collapse, anom_check_position_drift,
    anom_check_scale_explosion, anom_count_nonfinite, anom_format_event, anom_format_report,
    anom_generate_report, anom_is_monotone_increasing, anom_l2_norm, anom_max_abs,
    anom_max_pairwise_dist, anom_mean_std, AnomalyDetectionError, AnomalyDetector,
    AnomalyDetectorConfig, AnomalyEvent, AnomalyKind, AnomalyReport, AnomalySeverity,
    AnomalyThresholds,
};
pub use augmentation::{
    hsv_to_rgb, rgb_to_hsv, AugImage, AugStep, AugmentationError, AugmentationPipeline,
    ColorJitter, GammaDistortion, GaussianNoise, RandomCropResize, RandomFlip,
};
pub use callback::{
    default_callbacks, production_callbacks, Callback, CallbackChain, CallbackError, CallbackStats,
    CheckpointCallback, EarlyStoppingCallback, LossLoggerCallback, LrSchedulerCallback,
    MetricsHistoryCallback, TrainingContext,
};
pub use camera_sampling::{CameraSampler, CameraView, SamplingStrategy};
pub use checkpoint_interpolation::{
    build_checkpoint_path, compute_interpolation_stats, dequantize_error, find_optimal_blend,
    format_checkpoint_path, interpolate_along_path, interpolation_sequence, linear_interpolate,
    linear_mode_connectivity, model_soup, params_cosine_similarity, params_l2_distance,
    params_l2_norm, params_mean, params_std, quantize_params, uniform_average_params,
    weighted_average_params, CheckpointPath, InterpolationConfig, InterpolationError,
    InterpolationStats, ParamSnapshot,
};
pub use checkpoint_manager::{
    CheckpointIndex, CheckpointManager, CheckpointPolicy, CheckpointRecord,
};
pub use config::{DensityConfig, InitConfig, LossConfig, OptimizerConfig, TrainingConfig};
pub use continual_learning::{
    average_task_accuracy, backward_transfer, compute_cl_stats, compute_parameter_importance,
    create_task_mask, estimate_fisher_diagonal, ewc_gradient_single, ewc_penalty_single,
    forgetting_measure, replay_loss, simulate_task_gradients, update_online_fisher,
    ContinualLearningError, ContinualLearningStats, EwcConfig, EwcRegularizer, FisherInformation,
    ReplayBuffer, TaskMask,
};
pub use contrastive_loss::{
    contrastive_loss_batch,
    contrastive_loss_pair,
    cosine_similarity,
    infonce_loss,
    // l2_norm is intentionally omitted: conflicts with gradient_clipping::l2_norm.
    // Access it as contrastive_loss::l2_norm instead.
    l2_normalize,
    mine_triplets,
    mined_triplet_loss,
    pairwise_distance_matrix,
    squared_l2_distance,
    triplet_loss,
    triplet_loss_batch,
    triplet_violation_rate,
    ContrastiveConfig,
    ContrastiveLossError,
    InfoNceConfig,
    MiningStrategy,
    TripletConfig,
    TripletIndex,
};
pub use convergence_analysis::{
    compute_loss_percentile, compute_loss_slope, compute_oscillation_score,
    compute_relative_improvement, detect_convergence_phase, detect_phase_transitions,
    ema_smooth_losses, estimate_steps_to_convergence, format_convergence_report,
    generate_convergence_report, loss_improvement_rate, ConvergenceAnalyzer, ConvergenceConfig,
    ConvergenceError, ConvergencePhase, ConvergenceReport, ConvergenceStats,
};
pub use curriculum::{
    CurriculumConfig, CurriculumController, CurriculumError, CurriculumSchedule, CurriculumStage,
    CurriculumState, DifficultyDimension, ProgressTracker,
};
pub use curriculum_learning::{
    curr_compute_pacing, curr_compute_stats, curr_format_config, curr_format_stats, curr_normalize,
    curr_percentile_idx, curr_select_indices, curr_shuffle, CurrLearningConfig, CurrLearningError,
    CurriculumSampler, CurriculumStats, CurriculumStrategy, DifficultyEstimator, PacingFunction,
    SampleDifficulty,
};
pub use data_augmentation::{
    aug_add_gaussian_noise, aug_box_muller, aug_color_jitter, aug_denormalize, aug_gaussian_blur,
    aug_gaussian_kernel, aug_hsv_to_rgb, aug_image_stats, aug_normalize, aug_random_crop,
    aug_random_erasing, aug_rgb_to_hsv, aug_rotate_90, aug_separable_convolve, horizontal_flip,
    vertical_flip, xorshift64, xorshift_f32, AugImageStats, AugmentConfig, AugmentError, AugmentOp,
    ImageAugmenter,
};
pub use data_parallel::{DataParallelConfig, GradientAggregator, SyncMode, SyncReport};
pub use diagnostics::{
    DensityStats, EmaTracker, GradientNormTracker, LossTracker, TrainingDiagnostics,
};
pub use diffusion_target::{
    DiffusionTargetConfig, DiffusionTargetGenerator, SdsLoss, SdsWeighting, TemporalConsistency,
    ViewConsistencyLoss,
};
pub use ema::GaussianEma;
pub use feature_bank::{
    compute_bank_stats, sample_negatives, sample_positive, BankConfig, BankStatistics, FeatureBank,
    FeatureBankError, FeatureEntry, MomentumEncoder,
};
pub use gradient_accumulation::{
    gradients_have_inf, gradients_have_nan, scale_loss, unscale_gradients, AccumulationConfig,
    AccumulationError, AccumulationMonitor, AccumulationScheduler, AccumulationStats,
    GradNormalization, GradientAccumulator,
};
pub use gradient_clipping::{
    check_gradient_health, clip_by_global_norm, clip_by_per_group_norm, clip_by_value,
    global_gradient_norm, l2_norm, AdaptiveNormHistory, ClipError, ClipMode, ClipStats,
    GradientClipper, GradientHealth,
};
pub use gradient_flow::{
    classify_flow_health, compare_group_signals, compute_grad_trend, compute_group_snapshot,
    flow_l2_norm, flow_l_inf_norm, flow_linear_regression, flow_mean_abs, flow_std,
    format_flow_report, worst_health, FlowHealth, FlowSnapshot, GradTrend, GradientFlowConfig,
    GradientFlowError, GradientFlowReport, GradientFlowTracker, GroupFlowReport, GroupGradSnapshot,
};
pub use gradient_surgery::{
    aggregate_gradients, analyze_conflicts, gradients_conflict, pcgrad, project_gradient,
    AggregationStrategy, ConflictReport, ConflictTracker, GradientSurgeryError, TaskGradient,
};
pub use hparam_search::{
    AcquisitionFunction, HparamDef, HparamRange, HparamSearcher, SearchError, SearchSpace,
    SearchStrategy, Trial, TrialHistory,
};
pub use knowledge_distillation::{
    kd_attention_map, kd_attention_transfer_loss, kd_combined_loss, kd_cosine_similarity,
    kd_feature_loss, kd_format_loss, kd_format_stats, kd_hard_loss, kd_kl_divergence,
    kd_pairwise_distances, kd_relational_loss, kd_soft_loss, kd_softmax_with_temperature,
    kd_total_loss, DistillationConfig, DistillationError, DistillationHistory, DistillationLoss,
    DistillationStats, StudentResponse, TeacherResponse,
};
pub use layer_freezing::{
    apply_frozen_mask, back_frozen_mask, check_frozen_gradients_zeroed, front_frozen_mask,
    frozen_compute_savings, progressive_mask, split_into_groups, unfreeze_fraction, FreezeConfig,
    FreezingError, FrozenMask, GaussianParamGroup, ParameterFreezer, ProgressiveUnfreezeSchedule,
};
pub use loss::LpipsLossComputer;
pub use loss_landscape::{
    compute_landscape_stats, compute_sharpness, interpolate_params, scan_1d, InterpolationResult,
    LandscapeError, LandscapeScan1D, LandscapeStats, LossEvaluator, QuadraticLoss, SharpnessConfig,
    SharpnessMetrics,
};
pub use loss_reweighting::{
    apply_sample_weights, build_sample_weights, compute_exponential_weights, compute_focal_weights,
    compute_hardness_weights, compute_inverse_hardness_weights, compute_rank_weights,
    compute_sample_reweights, compute_uniform_weights, compute_weight_entropy, curriculum_weights,
    detect_weight_collapse, format_weight_summary, interpolate_strategies, weighted_mean_loss,
    CurriculumWeightConfig, FocalLossConfig, HardnessConfig, ReweightingError,
    SampleWeightingStrategy, SampleWeights, WeightTracker,
};
pub use lpips::{lpips_loss, LpipsDistance, LpipsWeights, VggFeatureExtractor};
pub use lr_scheduler::{
    cosine_decay_factor, schedule_summary, warmup_factor, ConstantSchedule,
    CosineAnnealingSchedule, CyclicSchedule, ExponentialDecaySchedule, LrSchedule, LrScheduler,
    LrSchedulerError, PolynomialDecaySchedule, SchedulerKind, StepDecaySchedule,
    WarmupCosineSchedule, WarmupLinearSchedule,
};
pub use meta_learning::{
    aggregate_meta_stats, apply_gradient_update, clip_gradient, compute_meta_gradient,
    evaluate_query_loss, grad_norm, inner_loop_adapt, meta_update_step, mse_loss_and_grad,
    run_meta_training, FewShotTask, LinearModel, MamlConfig, MetaLearningError, MetaTrainingStats,
    TaskSampler,
};
pub use mixed_precision::{LossScaler, LossScalerStats, MixedPrecisionTrainer, TrainingPrecision};
pub use noise_injection::{
    analyze_noise, inject_additive_noise, inject_multiplicative_noise, perturb_rotations,
    sample_gaussian_noise, NoiseAnalysis, NoiseConfig, NoiseInjectionError, NoiseInjector,
    NoiseSchedule, NoiseTarget,
};
pub use ohem::{
    compute_priority_weights, ExampleRecord, OhemConfig, OhemError, OhemStats, OhemTracker,
};
pub use online_hard_example_mining::{
    ohem_focal_weights,
    // Formatting
    ohem_format_result,
    ohem_format_stats,
    ohem_hard_loss,
    ohem_mine,
    ohem_pixel_l1,
    // Pixel-level mining
    ohem_pixel_mining,
    ohem_pixel_mse,
    ohem_select_by_threshold,
    // Core mining functions (ohem_ prefix, no conflicts with existing exports)
    ohem_select_top_k,
    ohem_soft_weights,
    ohem_weighted_loss,
    OhemResult,
    OnlineMiningConfig,
    // Error and config (names differ from ohem.rs to avoid conflicts)
    OnlineMiningError,
    OnlineMiningStats,
    OnlineMiningStrategy,
    OnlineMiningTracker,
    // Core data structures
    SampleLossHistory,
};
pub use online_learning::{
    compute_importance_weights, detect_stall, ema_update, linear_regression, reservoir_sample,
    weighted_sample_indices, welford_update, AdaptiveLrState, OnlineGradientStats,
    OnlineLearningError, OnlineLossHistory, StreamingBuffer,
};
pub use optimizer::{GaussianOptimizer, Gradients, ParameterGroup};
pub use pose_conditioning::{
    format_coverage_stats, grid_idx_to_pose, interpolate_poses, pairwise_angular_distances,
    pose_centroid, pose_diversity_score, pose_to_grid_idx, select_diverse_poses,
    spherical_gaussian_coverage, CoverageStats, PoseCondConfig, PoseCondError, PoseConditioner,
    RegisteredPose, SphericalPose,
};
pub use profiler_integration::{PhaseGuard, PhaseStats, TrainingPhase, TrainingProfiler};
pub use progressive_training::{
    collect_progressive_stats, densification_enabled_at_step, format_prog_stage,
    format_progressive_stats, interpolate_loss_weights, loss_weights_at_step,
    max_gaussians_at_step, opacity_reset_enabled_at_step, progressive_resolution,
    resolution_at_step, scale_resolution, sh_degree_at_step, should_increase_sh_degree,
    ProgressiveConfig, ProgressiveError, ProgressiveStats, ProgressiveTrainer, StageLossWeights,
    StageTransition, TrainingStage,
};
pub use pruning::{
    apply_mask,
    count_prunable_by_opacity,
    cubic_sparsity,
    // sigmoid intentionally omitted: already re-exported from regularization
    prune_by_opacity,
    prune_large_gaussians,
    prune_small_gaussians,
    random_prune,
    GaussianPruner,
    PruningConfig,
    PruningError,
    PruningMask,
    PruningSchedule,
    PruningStats,
};
pub use regularization::{
    rms, sigmoid, soft_l1, soft_l1_grad, CompositeRegularizer, L1Regularization, L2Regularization,
    OpacityRegMode, OpacityRegularization, PosRegMode, PositionalRegularization, RegBreakdown,
    RegularizationConfig, RegularizationError, ScaleRegMode, ScaleRegularization,
};
pub use session_recorder::{
    generate_session_id, HardwareInfo, MetricsSnapshot, SessionConfigSnapshot, SessionError,
    SessionRecord, SessionRecorder,
};
pub use spectral_norm::{
    batch_spectral_norm, estimate_condition_number, frobenius_norm, is_well_conditioned,
    mat_transpose_vec_mul, mat_vec_mul, normalize_by_spectral_norm, power_iteration,
    power_iteration_step, sequence_lipschitz_bound, spectral_l2_norm,
    spectral_norm as compute_spectral_norm, spectral_normalize, stable_rank, PowerIterationConfig,
    SingularValueEstimate, SpectralNormError, SpectralNormTracker,
};
pub use synthetic_data::{
    generate_synthetic_batch, sample_at_difficulty, sample_gaussian_cloud, sample_unit_quaternion,
    DifficultyLevel, FlameParamSampler, FlameParamSamplerConfig, GaussianCloudConfig,
    SyntheticBatch, SyntheticDataError, SyntheticFlameParams, SyntheticGaussianCloud,
};
pub use tensorboard::{LearningRates, TensorBoardConfig, TensorBoardWriter, TrainingMetricsLogger};
pub use trainer::{StepOutput, Trainer};
pub use training_config::{TrainingProfile, TrainingProfileConfig};
pub use uncertainty_estimation::{
    aggregate_region_uncertainty, apply_dropout_mask, bald_score, compute_calibration,
    decompose_uncertainty, ensemble_variance, high_uncertainty_indices, mc_dropout_stats,
    per_gaussian_position_uncertainty, prediction_entropy, reliability_diagram_points,
    stable_softmax, temperature_scale, uncertainty_weighted_loss, variance_to_confidence,
    CalibrationResult, ConfidenceMap, PredictionUncertainty, RegionUncertainty, UncertaintyConfig,
    UncertaintyDecomposition, UncertaintyError,
};
pub use validation_split::{
    DataSplit, EarlyStopper, EarlyStopperConfig, MonitorMetric, SplitError, SplitStrategy,
    ValidationStep, ValidationTracker,
};
pub use video::{FrameBatch, VideoConfig, VideoFrameIterator};
pub use view_importance::{
    combine_importance, compute_loss_importance, compute_loss_variance, compute_mean_loss,
    compute_recency_importance, compute_variance_importance, format_importance_summary,
    importance_entropy, importance_softmax, normalize_importance, sample_from_weights,
    select_views_by_importance, view_importance_summary_stats, ImportanceStrategy,
    ViewImportanceConfig, ViewImportanceError, ViewImportanceSampler, ViewImportanceScore,
};
pub use view_scheduler::{
    analyze_view_coverage, angular_distance, geometric_importance, uniform_view_angles,
    CoverageReport, ViewInfo, ViewScheduler, ViewSchedulerConfig, ViewSchedulerError, ViewStats,
};
pub use view_synthesis_eval::{
    eval_aggregate_metrics, eval_error_map, eval_lpips_approx, eval_mae, eval_mse, eval_psnr,
    eval_ssim, eval_view_metrics, find_extreme_views, format_eval_metrics, format_view_metrics,
    EvalConfig, EvalError, EvalMetrics, EvalRecord, ViewMetrics, ViewSynthesisEvaluator,
};
pub use weight_averaging::{
    compute_weight_stats, count_params, weights_cosine_similarity, weights_l2_distance, ModelSoup,
    ModelWeights, PolyakAverager, StochasticWeightAverager, SwaConfig, WeightAveragingError,
    WeightStats,
};
pub mod temperature_scaling;
pub use adaptive_loss_weighting::{
    alw_clip_weights, alw_format_history_summary, alw_format_weights, alw_imbalance_ratio,
    alw_normalize_weights, alw_relative_training_rate, alw_weighted_sum, GradNormWeighter,
    HomoscedasticWeighter, LossStatTracker, LossTask, LossWeightError, ScheduledWeighter,
    TaskWeightSchedule, WeightHistory, WeightScheduleKind,
};
pub use contrastive_learning::{
    cl_alignment,
    cl_cosine_sim,
    cl_dot,
    cl_format_config,
    cl_format_stats,
    // InfoNCE loss
    cl_info_nce_loss,
    cl_l2_distance,
    // Hard negative mining
    cl_mine_hard_negatives,
    cl_mine_semi_hard_negatives,
    // Primitive utilities
    cl_normalize,
    // NT-Xent (SimCLR) loss
    cl_nt_xent_loss,
    cl_similarity_matrix,
    // Supervised contrastive loss
    cl_supcon_loss,
    // Triplet loss
    cl_triplet_loss,
    cl_uniformity,
    cl_update_stats,
    // Error type
    ContrastiveError,
    // Config (renamed to avoid conflict with contrastive_loss::ContrastiveConfig)
    ContrastiveLearningConfig,
    // Statistics and tracking
    ContrastiveStats,
    // Memory queue
    EmbeddingQueue,
};
pub use domain_adaptation::{
    da_center_features, da_combined_loss, da_compute_stats, da_confidence_threshold_mask,
    da_coral_loss, da_covariance, da_dann_loss, da_domain_accuracy, da_entropy, da_entropy_loss,
    da_feature_mean, da_format_config, da_format_stats, da_frobenius_sq, da_gaussian_kernel,
    da_median_bandwidth, da_mmd_biased, da_mmd_multiscale, da_mmd_unbiased, da_pseudo_label_loss,
    da_reversal_loss_scale, AdaptationStats, DannConfig, DomainAdaptConfig, DomainAdaptMethod,
    DomainAdaptationError, DomainBatch, DomainDiscriminator, MmdConfig,
};
pub use few_shot_adaptation::{
    fsa_class_indices, fsa_compute_stats, fsa_episode_accuracy, fsa_format_config,
    fsa_format_result, fsa_format_stats, fsa_inner_gradient, fsa_inner_loss, fsa_lora_apply,
    fsa_maml_adapt, fsa_maml_query_loss, fsa_proto_accuracy, fsa_proto_loss, fsa_sample_episode,
    AdaptationResult, Episode, FewShotConfig, FewShotError, FewShotStats, LoraAdapter, MamlState,
    PrototypicalNet, QuerySet, SupportSet,
};
pub use temperature_scaling::{
    ts_binary_nll,
    ts_brier_score,
    ts_compute_stats,
    ts_ece,
    ts_format_reliability_diagram,
    ts_format_result,
    ts_format_stats,
    ts_golden_section_search,
    ts_log_loss,
    ts_mce,
    ts_overconfidence_error,
    ts_pav_isotonic,
    ts_reliability_diagram,
    CalibrationConfig,
    CalibrationError,
    // CalibrationResult aliased to avoid collision with uncertainty_estimation::CalibrationResult
    CalibrationResult as TsCalibrationResult,
    CalibrationStats,
    IsotonicCalibrator,
    PlattScaler,
    ReliabilityDiagram,
    TemperatureScaler,
};
