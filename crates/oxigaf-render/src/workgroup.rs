//! Workgroup size configuration and benchmark-based auto-selection.
//!
//! This module provides a `WorkgroupConfig` system for choosing GPU dispatch
//! sizes across different GPU architectures, along with a benchmarker that can
//! either time a caller-supplied closure or really dispatch a compute kernel at
//! each candidate workgroup size.
//!
//! # What is and is not wired to the GPU
//!
//! [`WorkgroupConfig::shipped`] is the geometry the compiled shaders actually
//! declare, and [`Rasterizer`](crate::Rasterizer) derives every
//! `dispatch_workgroups` count from it — the host grid and the WGSL
//! `@workgroup_size` attributes therefore have one source of truth.
//!
//! The *preset* profiles cannot be handed to the rasterizer yet: WGSL fixes
//! `@workgroup_size` at shader-compile time, so honouring a runtime choice
//! means emitting one shader variant per size.
//! [`WorkgroupBenchmarker::benchmark_dispatch`] shows the technique (it
//! substitutes the size into its measurement kernel's source and compiles one
//! pipeline per candidate) and gives a real, GPU-timed ranking; applying that
//! ranking to the rasterization pipeline is not implemented.
//!
//! # Example
//!
//! ```rust
//! use oxigaf_render::workgroup::{WorkgroupConfig, WorkgroupProfile, WorkgroupBenchmarker};
//!
//! // Use a preset profile
//! let config = WorkgroupConfig::balanced();
//! assert_eq!(config.profile, WorkgroupProfile::Balanced);
//!
//! // Adapt automatically to the Gaussian count
//! let config = WorkgroupConfig::adaptive(50_000);
//!
//! // Closure-driven recommendation. The caller owns the measurement: the
//! // ranking is only as meaningful as the work the closure performs. For a
//! // GPU-timed ranking use `WorkgroupBenchmarker::recommend_on_device`.
//! let benchmarker = WorkgroupBenchmarker::new();
//! let config = benchmarker.recommend(50_000, |ws| {
//!     let start = std::time::Instant::now();
//!     let _ = ws.total(); // stand-in for the caller's real work
//!     start.elapsed()
//! });
//! ```

use crate::RenderError;

// ---------------------------------------------------------------------------
// WorkgroupSize
// ---------------------------------------------------------------------------

/// Three-dimensional workgroup size for compute shaders.
///
/// For 1-D compute shaders (most Gaussian processing passes) use
/// [`WorkgroupSize::linear`]. For 2-D tile operations use
/// [`WorkgroupSize::square`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkgroupSize {
    /// Workgroup threads in the X dimension.
    pub x: u32,
    /// Workgroup threads in the Y dimension.
    pub y: u32,
    /// Workgroup threads in the Z dimension.
    pub z: u32,
}

impl WorkgroupSize {
    /// Create a workgroup size with explicit x/y/z dimensions.
    #[inline]
    pub const fn new(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }

    /// Create a 1-D workgroup of `n` threads (y = z = 1).
    #[inline]
    pub const fn linear(n: u32) -> Self {
        Self::new(n, 1, 1)
    }

    /// Create a 2-D square workgroup of `n × n` threads (z = 1).
    #[inline]
    pub const fn square(n: u32) -> Self {
        Self::new(n, n, 1)
    }

    /// Total number of threads in this workgroup.
    ///
    /// Saturating: [`WorkgroupConfig::validate`] calls this *before* it can
    /// reject an oversized configuration, so a hand-built or deserialized
    /// `WorkgroupSize` with huge dimensions must clamp here rather than
    /// overflow (which panics in debug builds and wraps — possibly back under
    /// the 1024 limit — in release).
    #[inline]
    pub fn total(&self) -> u32 {
        self.x.saturating_mul(self.y).saturating_mul(self.z)
    }

    /// Number of workgroups needed to cover `n` threads in the X dimension
    /// (rounds up to avoid leaving elements unprocessed).
    #[inline]
    pub fn dispatch_count_x(&self, n: u32) -> u32 {
        n.div_ceil(self.x)
    }

    /// Number of workgroups needed to cover `n` threads in the Y dimension
    /// (rounds up).
    #[inline]
    pub fn dispatch_count_y(&self, n: u32) -> u32 {
        n.div_ceil(self.y)
    }
}

// ---------------------------------------------------------------------------
// WorkgroupProfile
// ---------------------------------------------------------------------------

/// Preset workgroup size profiles targeting different GPU classes.
///
/// `WorkgroupProfile` is *workload-oriented* (small vs. large thread counts)
/// while [`crate::config::GpuPreset`] is *vendor-oriented*.  The two can be
/// used together or independently.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkgroupProfile {
    /// Mobile / integrated GPUs: 32 threads per workgroup.
    Mobile,
    /// Balanced default: 64 threads per workgroup (good for most GPUs).
    Balanced,
    /// High-throughput desktop GPUs: 256 threads per workgroup.
    HighThroughput,
    /// User-defined configuration; the profile field acts as a tag only.
    Custom,
}

impl WorkgroupProfile {
    /// Return the default 1-D workgroup size for this profile.
    #[must_use]
    pub fn default_size(&self) -> WorkgroupSize {
        match self {
            Self::Mobile => WorkgroupSize::linear(32),
            Self::Balanced => WorkgroupSize::linear(64),
            Self::HighThroughput => WorkgroupSize::linear(256),
            Self::Custom => WorkgroupSize::linear(64), // sensible fallback
        }
    }

    /// Short machine-readable name for this profile.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Mobile => "mobile",
            Self::Balanced => "balanced",
            Self::HighThroughput => "high_throughput",
            Self::Custom => "custom",
        }
    }

    /// Human-readable description of this profile.
    #[must_use]
    pub const fn description(&self) -> &'static str {
        match self {
            Self::Mobile => "Small workgroups (32 threads) for mobile/integrated GPUs",
            Self::Balanced => "Balanced workgroups (64 threads) suitable for most GPUs",
            Self::HighThroughput => {
                "Large workgroups (256 threads) for powerful desktop/server GPUs"
            }
            Self::Custom => "User-defined workgroup configuration",
        }
    }
}

// ---------------------------------------------------------------------------
// WorkgroupConfig
// ---------------------------------------------------------------------------

/// Complete workgroup configuration covering all rasterization passes.
///
/// Each pass can be tuned independently; use [`WorkgroupConfig::from_profile`]
/// to start from a preset and then override individual fields as needed.
#[derive(Debug, Clone)]
pub struct WorkgroupConfig {
    /// The profile that was used to create this configuration.
    pub profile: WorkgroupProfile,

    /// Preprocess pass (Gaussian projection, covariance computation).
    pub preprocess: WorkgroupSize,

    /// Sorting pass (radix sort over depth-keyed Gaussians).
    pub sort: WorkgroupSize,

    /// Forward rasterization pass (per-tile alpha-blending).
    pub rasterize: WorkgroupSize,

    /// Backward pass (gradient computation through rasterizer).
    pub backward: WorkgroupSize,

    /// Tile-based operations (2-D, typically square).
    pub tile: WorkgroupSize,
}

impl WorkgroupConfig {
    /// Build a `WorkgroupConfig` from a [`WorkgroupProfile`].
    ///
    /// All 1-D passes use the profile's default linear size; the tile pass
    /// always uses a 16 × 16 square (256 threads), matching the tile size the
    /// rasterization shaders declare — one thread per pixel of a 16 × 16 tile.
    #[must_use]
    pub fn from_profile(profile: WorkgroupProfile) -> Self {
        let size = profile.default_size();
        Self {
            profile,
            preprocess: size,
            sort: size,
            rasterize: size,
            backward: size,
            tile: Self::SHIPPED_TILE, // 16×16 = 256 threads, one per tile pixel
        }
    }

    /// The 2-D tile workgroup the shipped rasterization shaders declare.
    ///
    /// `rasterize_fwd.wgsl` / `rasterize_bwd.wgsl` use `@workgroup_size(16, 16)`
    /// so that one workgroup covers exactly one 16-pixel tile.
    pub const SHIPPED_TILE: WorkgroupSize = WorkgroupSize::square(16);

    /// The 1-D workgroup the shipped compute kernels declare.
    ///
    /// Every 1-D kernel in `shaders/` uses `@workgroup_size(256)`.
    pub const SHIPPED_LINEAR: WorkgroupSize = WorkgroupSize::linear(256);

    /// The workgroup geometry the **compiled** shaders actually use.
    ///
    /// [`Rasterizer`](crate::Rasterizer) derives every `dispatch_workgroups`
    /// count from this configuration rather than from open-coded literals, so
    /// the host-side dispatch grid and the WGSL `@workgroup_size` attributes
    /// have a single source of truth.
    ///
    /// The other presets ([`mobile`](Self::mobile), [`balanced`](Self::balanced),
    /// [`high_throughput`](Self::high_throughput),
    /// [`adaptive`](Self::adaptive)) describe what a given GPU class *would*
    /// prefer. They cannot be fed to the rasterizer today because
    /// `@workgroup_size` is fixed at shader-compile time; selecting them
    /// requires emitting per-size shader variants (see
    /// [`WorkgroupBenchmarker::benchmark_dispatch`], which does exactly that for
    /// its measurement kernel).
    #[must_use]
    pub fn shipped() -> Self {
        Self::for_linear_size(Self::SHIPPED_LINEAR.x)
    }

    /// The geometry of a rasterizer whose **1-D** kernels were compiled with
    /// `@workgroup_size(linear)`.
    ///
    /// [`RasterPipelines::new`](crate::pipeline::RasterPipelines::new)
    /// substitutes the configured size into the sources of the kernels whose
    /// `@workgroup_size` is pure dispatch geometry (the preprocess variants,
    /// `preprocess_bwd`, `tile_assign`, `tile_ranges` and `atomic_to_f32`), so
    /// the host grid must be derived from the same number — that is what this
    /// constructor is for.
    ///
    /// The 2-D passes keep [`SHIPPED_TILE`](Self::SHIPPED_TILE): the
    /// rasterization kernels' `@workgroup_size(16, 16)` is not an independent
    /// tuning knob but the tile size itself, one thread per tile pixel (see
    /// `RASTERIZE_TILE_SIZE`). The `prefix_sum`, `prefix_sum_add`,
    /// `radix_histogram` and `radix_scatter` kernels are likewise excluded —
    /// they bake their thread count into shared-memory sizes and loop bounds,
    /// so substituting a different attribute would silently corrupt their
    /// output rather than retune it.
    ///
    /// [`shipped`](Self::shipped) is this with the default 256.
    #[must_use]
    pub fn for_linear_size(linear: u32) -> Self {
        let linear = WorkgroupSize::linear(linear);
        Self {
            profile: WorkgroupProfile::HighThroughput,
            preprocess: linear,
            sort: linear,
            rasterize: Self::SHIPPED_TILE,
            backward: Self::SHIPPED_TILE,
            tile: Self::SHIPPED_TILE,
        }
    }

    /// Preset configuration suitable for mobile / integrated GPUs.
    #[must_use]
    pub fn mobile() -> Self {
        Self::from_profile(WorkgroupProfile::Mobile)
    }

    /// Preset configuration suitable for most GPUs (default).
    #[must_use]
    pub fn balanced() -> Self {
        Self::from_profile(WorkgroupProfile::Balanced)
    }

    /// Preset configuration for high-throughput desktop/server GPUs.
    #[must_use]
    pub fn high_throughput() -> Self {
        Self::from_profile(WorkgroupProfile::HighThroughput)
    }

    /// Choose a profile adaptively based on the number of Gaussians.
    ///
    /// | Gaussian count   | Profile       |
    /// |-----------------|---------------|
    /// | < 10 000        | Mobile        |
    /// | 10 000 – 99 999 | Balanced      |
    /// | ≥ 100 000       | HighThroughput|
    #[must_use]
    pub fn adaptive(num_gaussians: usize) -> Self {
        if num_gaussians < 10_000 {
            Self::mobile()
        } else if num_gaussians < 100_000 {
            Self::balanced()
        } else {
            Self::high_throughput()
        }
    }

    /// Validate that all workgroup sizes are well-formed.
    ///
    /// Checks:
    /// - All dimensions are non-zero.
    /// - Total threads per workgroup do not exceed 1024 (safe WebGPU / Vulkan limit).
    /// - Each dimension is a power of two (required by most GPU architectures for
    ///   efficient scheduling).
    pub fn validate(&self) -> Result<(), RenderError> {
        let passes = [
            ("preprocess", self.preprocess),
            ("sort", self.sort),
            ("rasterize", self.rasterize),
            ("backward", self.backward),
            ("tile", self.tile),
        ];

        for (name, ws) in passes {
            if ws.x == 0 || ws.y == 0 || ws.z == 0 {
                return Err(RenderError::Rasterize(format!(
                    "workgroup '{}' has a zero dimension: {:?}",
                    name, ws
                )));
            }

            let total = ws.total();
            if total > 1024 {
                return Err(RenderError::Rasterize(format!(
                    "workgroup '{}' total threads {} exceeds maximum 1024",
                    name, total
                )));
            }

            if !is_power_of_two(ws.x) || !is_power_of_two(ws.y) || !is_power_of_two(ws.z) {
                return Err(RenderError::Rasterize(format!(
                    "workgroup '{}' dimensions must be powers of two, got {:?}",
                    name, ws
                )));
            }
        }

        Ok(())
    }
}

impl Default for WorkgroupConfig {
    fn default() -> Self {
        Self::balanced()
    }
}

/// Returns `true` if `n` is a power of two (and non-zero).
#[inline]
fn is_power_of_two(n: u32) -> bool {
    n != 0 && n.is_power_of_two()
}

// ---------------------------------------------------------------------------
// WorkgroupBenchResult
// ---------------------------------------------------------------------------

/// Result of benchmarking a single workgroup size.
#[derive(Debug, Clone)]
pub struct WorkgroupBenchResult {
    /// The workgroup size that was benchmarked.
    pub size: WorkgroupSize,
    /// Arithmetic mean of measured durations in microseconds.
    pub mean_duration_us: f64,
    /// Minimum measured duration across all samples, in microseconds.
    pub min_duration_us: f64,
    /// Number of measurement samples taken.
    pub samples: usize,
}

// ---------------------------------------------------------------------------
// WorkgroupBenchmarker
// ---------------------------------------------------------------------------

/// Benchmarker for selecting workgroup sizes.
///
/// Two measurement modes:
///
/// * [`benchmark`](Self::benchmark) / [`recommend`](Self::recommend) — the
///   *caller* supplies a closure and therefore owns the measurement. Useful
///   when the work to time lives outside this crate; meaningless if the closure
///   does not actually do the dispatch.
/// * [`benchmark_dispatch`](Self::benchmark_dispatch) /
///   [`recommend_on_device`](Self::recommend_on_device) — this module compiles
///   one compute pipeline per candidate size and times real GPU submissions.
///
/// # Example
///
/// ```rust
/// use std::time::Instant;
/// use oxigaf_render::workgroup::{WorkgroupBenchmarker, WorkgroupSize};
///
/// let benchmarker = WorkgroupBenchmarker::new()
///     .with_warmup(2)
///     .with_measure(5);
///
/// let results = benchmarker.benchmark(|ws| {
///     let start = Instant::now();
///     let _n = ws.total(); // stand-in for the caller's real work
///     start.elapsed()
/// });
///
/// let best = benchmarker.best_of(&results);
/// ```
pub struct WorkgroupBenchmarker {
    /// Candidate linear workgroup sizes to benchmark.
    candidates: Vec<u32>,
    /// Number of warm-up invocations (results discarded).
    warmup_rounds: usize,
    /// Number of measurement invocations per candidate.
    measure_rounds: usize,
}

impl WorkgroupBenchmarker {
    /// Create a benchmarker with sensible defaults:
    /// - Candidates: 32, 64, 128, 256
    /// - Warm-up rounds: 3
    /// - Measurement rounds: 10
    #[must_use]
    pub fn new() -> Self {
        Self {
            candidates: vec![32, 64, 128, 256],
            warmup_rounds: 3,
            measure_rounds: 10,
        }
    }

    /// Override the candidate workgroup sizes (linear, 1-D).
    ///
    /// Each value becomes a `WorkgroupSize::linear(n)` internally.
    #[must_use]
    pub fn with_candidates(mut self, sizes: Vec<u32>) -> Self {
        self.candidates = sizes;
        self
    }

    /// Set the number of warm-up rounds (results discarded).
    #[must_use]
    pub fn with_warmup(mut self, rounds: usize) -> Self {
        self.warmup_rounds = rounds;
        self
    }

    /// Set the number of measurement rounds per candidate.
    #[must_use]
    pub fn with_measure(mut self, rounds: usize) -> Self {
        self.measure_rounds = rounds;
        self
    }

    /// Benchmark `f` for every candidate workgroup size.
    ///
    /// For each candidate the closure is called `warmup_rounds` times
    /// (results discarded) and then `measure_rounds` times for the actual
    /// measurement.  Each [`WorkgroupBenchResult`] contains the mean and
    /// minimum measured durations in microseconds.
    ///
    /// The closure is the entire measurement: if it does not perform work whose
    /// cost depends on the workgroup size, the ranking is noise. Use
    /// [`benchmark_dispatch`](Self::benchmark_dispatch) for a GPU-timed
    /// alternative that needs no closure.
    pub fn benchmark<F>(&self, f: F) -> Vec<WorkgroupBenchResult>
    where
        F: Fn(WorkgroupSize) -> std::time::Duration,
    {
        let mut results = Vec::with_capacity(self.candidates.len());

        for &size_n in &self.candidates {
            let ws = WorkgroupSize::linear(size_n);

            // Warm-up: prime caches, JIT, branch predictors, etc.
            for _ in 0..self.warmup_rounds {
                let _ = f(ws);
            }

            // Measurement
            let mut durations_us: Vec<f64> = Vec::with_capacity(self.measure_rounds);
            for _ in 0..self.measure_rounds {
                let d = f(ws);
                durations_us.push(duration_to_us(d));
            }

            let n = durations_us.len();
            let mean = if n == 0 {
                0.0
            } else {
                durations_us.iter().copied().sum::<f64>() / n as f64
            };

            let min = durations_us.iter().copied().fold(f64::INFINITY, f64::min);

            results.push(WorkgroupBenchResult {
                size: ws,
                mean_duration_us: mean,
                min_duration_us: if min.is_infinite() { 0.0 } else { min },
                samples: n,
            });
        }

        results
    }

    /// Return the workgroup size with the lowest mean duration, or `None` if
    /// `results` is empty.
    #[must_use]
    pub fn best_of(&self, results: &[WorkgroupBenchResult]) -> Option<WorkgroupSize> {
        results
            .iter()
            .min_by(|a, b| {
                a.mean_duration_us
                    .partial_cmp(&b.mean_duration_us)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|r| r.size)
    }

    /// Run the closure benchmark and return a recommended [`WorkgroupConfig`].
    ///
    /// The best workgroup size found is applied to all 1-D passes (preprocess,
    /// sort, rasterize, backward) while the tile pass keeps
    /// [`WorkgroupConfig::SHIPPED_TILE`], the 16 × 16 geometry the tile shaders
    /// declare.  If the benchmark yields no results (e.g., no candidates),
    /// falls back to [`WorkgroupConfig::adaptive`].
    ///
    /// The quality of the recommendation is entirely the quality of `f` — see
    /// [`benchmark`](Self::benchmark).
    #[must_use]
    pub fn recommend<F>(&self, num_gaussians: usize, f: F) -> WorkgroupConfig
    where
        F: Fn(WorkgroupSize) -> std::time::Duration,
    {
        let results = self.benchmark(f);
        self.config_from_results(&results, num_gaussians)
    }

    /// Benchmark every candidate size with a **real GPU dispatch**.
    ///
    /// For each candidate `s` a small element-wise compute kernel is compiled
    /// with `@workgroup_size(s)` substituted into its source (WGSL fixes the
    /// workgroup size at compile time, so one pipeline per candidate is the
    /// only way to vary it), then dispatched over `num_elements` elements.
    /// Each measurement submits the command buffer and blocks until the device
    /// reports it complete, so the timing covers actual GPU execution rather
    /// than command recording.
    ///
    /// Returns one [`WorkgroupBenchResult`] per candidate, in candidate order.
    pub fn benchmark_dispatch(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        num_elements: u32,
    ) -> Vec<WorkgroupBenchResult> {
        let elements = num_elements.max(1);

        // Ask the device, not a hardcoded constant: `benchmark_dispatch` takes
        // an arbitrary device, which may have been created with raised (or
        // lowered) limits. Compiling `@workgroup_size(n)` above the ceiling is a
        // shader-creation failure, so those candidates are skipped rather than
        // measured.
        let limits = device.limits();
        let max_workgroup_size = limits
            .max_compute_invocations_per_workgroup
            .min(limits.max_compute_workgroup_size_x)
            .max(1);

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("workgroup_bench_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let data = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("workgroup_bench_data"),
            size: u64::from(elements) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("workgroup_bench_params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(
            &params,
            0,
            bytemuck::cast_slice(&[elements, 0u32, 0u32, 0u32]),
        );

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("workgroup_bench_bg"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: data.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params.as_entire_binding(),
                },
            ],
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("workgroup_bench_layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

        let mut results = Vec::with_capacity(self.candidates.len());
        for &size_n in &self.candidates {
            let requested = size_n.max(1);
            if requested > max_workgroup_size {
                tracing::warn!(
                    candidate = requested,
                    max = max_workgroup_size,
                    "skipping workgroup candidate above the device's per-workgroup invocation limit"
                );
                continue;
            }
            let ws = WorkgroupSize::linear(requested);
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("workgroup_bench_kernel"),
                source: wgpu::ShaderSource::Wgsl(bench_kernel_source(ws.x).into()),
            });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("workgroup_bench_pipeline"),
                layout: Some(&layout),
                module: &module,
                entry_point: Some("bench_kernel"),
                compilation_options: Default::default(),
                cache: None,
            });
            let dispatch = ws.dispatch_count_x(elements).max(1);

            let run_once = || {
                let start = std::time::Instant::now();
                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("workgroup_bench"),
                });
                {
                    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("workgroup_bench"),
                        timestamp_writes: None,
                    });
                    cpass.set_pipeline(&pipeline);
                    cpass.set_bind_group(0, &bind_group, &[]);
                    cpass.dispatch_workgroups(dispatch, 1, 1);
                }
                queue.submit(std::iter::once(encoder.finish()));
                let _ = device.poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: None,
                });
                start.elapsed()
            };

            for _ in 0..self.warmup_rounds {
                let _ = run_once();
            }
            let mut durations_us: Vec<f64> = Vec::with_capacity(self.measure_rounds);
            for _ in 0..self.measure_rounds {
                durations_us.push(duration_to_us(run_once()));
            }

            let n = durations_us.len();
            let mean = if n == 0 {
                0.0
            } else {
                durations_us.iter().copied().sum::<f64>() / n as f64
            };
            let min = durations_us.iter().copied().fold(f64::INFINITY, f64::min);
            results.push(WorkgroupBenchResult {
                size: ws,
                mean_duration_us: mean,
                min_duration_us: if min.is_infinite() { 0.0 } else { min },
                samples: n,
            });
        }

        results
    }

    /// Run [`benchmark_dispatch`](Self::benchmark_dispatch) and turn the
    /// measurements into a [`WorkgroupConfig`].
    ///
    /// Falls back to [`WorkgroupConfig::adaptive`] when there are no candidates.
    #[must_use]
    pub fn recommend_on_device(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        num_elements: u32,
    ) -> WorkgroupConfig {
        let results = self.benchmark_dispatch(device, queue, num_elements);
        self.config_from_results(&results, num_elements as usize)
    }

    /// Build a config from ranked results, falling back to `adaptive`.
    fn config_from_results(
        &self,
        results: &[WorkgroupBenchResult],
        num_gaussians: usize,
    ) -> WorkgroupConfig {
        match self.best_of(results) {
            Some(best_size) => WorkgroupConfig {
                profile: WorkgroupProfile::Custom,
                preprocess: best_size,
                sort: best_size,
                rasterize: best_size,
                backward: best_size,
                tile: WorkgroupConfig::SHIPPED_TILE,
            },
            None => WorkgroupConfig::adaptive(num_gaussians),
        }
    }
}

/// WGSL source for the dispatch benchmark kernel, with `workgroup_size`
/// substituted in.
///
/// The body is deliberately memory-bound and data-dependent (a linear
/// congruential step written back in place) so the compiler cannot eliminate it
/// and the measurement reflects occupancy rather than instruction count.
fn bench_kernel_source(workgroup_size: u32) -> String {
    format!(
        r#"@group(0) @binding(0) var<storage, read_write> data: array<u32>;
@group(0) @binding(1) var<uniform> params: vec4<u32>;

@compute @workgroup_size({workgroup_size})
fn bench_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if i >= params.x {{
        return;
    }}
    var v = data[i];
    v = v * 1664525u + 1013904223u;
    data[i] = v;
}}
"#
    )
}

impl Default for WorkgroupBenchmarker {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert a [`std::time::Duration`] to microseconds as `f64`.
#[inline]
fn duration_to_us(d: std::time::Duration) -> f64 {
    d.as_secs_f64() * 1_000_000.0
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    // --- WorkgroupSize basic ---

    #[test]
    fn test_linear_total() {
        assert_eq!(WorkgroupSize::linear(64).total(), 64);
    }

    #[test]
    fn test_square_total() {
        assert_eq!(WorkgroupSize::square(8).total(), 64);
    }

    #[test]
    fn test_dispatch_count_x_rounds_up() {
        let ws = WorkgroupSize::linear(64);
        // 65 threads → needs 2 workgroups (128 threads dispatched)
        assert_eq!(ws.dispatch_count_x(65), 2);
        // 1 thread → needs 1 workgroup
        assert_eq!(ws.dispatch_count_x(1), 1);
    }

    #[test]
    fn test_dispatch_count_x_exact_multiple() {
        let ws = WorkgroupSize::linear(64);
        // 128 is exactly 2 × 64, should not round up
        assert_eq!(ws.dispatch_count_x(128), 2);
    }

    #[test]
    fn test_dispatch_count_y_rounds_up() {
        let ws = WorkgroupSize::square(8);
        assert_eq!(ws.dispatch_count_y(9), 2);
        assert_eq!(ws.dispatch_count_y(8), 1);
    }

    // --- WorkgroupProfile ---

    #[test]
    fn test_profile_mobile_total() {
        assert_eq!(WorkgroupProfile::Mobile.default_size().total(), 32);
    }

    #[test]
    fn test_profile_balanced_total() {
        assert_eq!(WorkgroupProfile::Balanced.default_size().total(), 64);
    }

    #[test]
    fn test_profile_high_throughput_total() {
        assert_eq!(WorkgroupProfile::HighThroughput.default_size().total(), 256);
    }

    #[test]
    fn test_profile_name() {
        assert_eq!(WorkgroupProfile::Mobile.name(), "mobile");
        assert_eq!(WorkgroupProfile::Balanced.name(), "balanced");
        assert_eq!(WorkgroupProfile::HighThroughput.name(), "high_throughput");
        assert_eq!(WorkgroupProfile::Custom.name(), "custom");
    }

    #[test]
    fn test_profile_description_non_empty() {
        for profile in [
            WorkgroupProfile::Mobile,
            WorkgroupProfile::Balanced,
            WorkgroupProfile::HighThroughput,
            WorkgroupProfile::Custom,
        ] {
            assert!(!profile.description().is_empty());
        }
    }

    // --- WorkgroupConfig construction ---

    #[test]
    fn test_mobile_preprocess_total() {
        assert_eq!(WorkgroupConfig::mobile().preprocess.total(), 32);
    }

    #[test]
    fn test_balanced_profile_tag() {
        assert_eq!(
            WorkgroupConfig::balanced().profile,
            WorkgroupProfile::Balanced
        );
    }

    #[test]
    fn test_adaptive_small() {
        let cfg = WorkgroupConfig::adaptive(5_000);
        assert_eq!(cfg.profile, WorkgroupProfile::Mobile);
    }

    #[test]
    fn test_adaptive_medium() {
        let cfg = WorkgroupConfig::adaptive(50_000);
        assert_eq!(cfg.profile, WorkgroupProfile::Balanced);
    }

    #[test]
    fn test_adaptive_large() {
        let cfg = WorkgroupConfig::adaptive(500_000);
        assert_eq!(cfg.profile, WorkgroupProfile::HighThroughput);
    }

    #[test]
    fn test_default_is_balanced() {
        let cfg = WorkgroupConfig::default();
        assert_eq!(cfg.profile, WorkgroupProfile::Balanced);
    }

    // --- Validation ---

    #[test]
    fn test_validate_balanced_ok() {
        assert!(WorkgroupConfig::balanced().validate().is_ok());
    }

    #[test]
    fn test_validate_mobile_ok() {
        assert!(WorkgroupConfig::mobile().validate().is_ok());
    }

    #[test]
    fn test_validate_high_throughput_ok() {
        assert!(WorkgroupConfig::high_throughput().validate().is_ok());
    }

    #[test]
    fn test_validate_zero_dimension_fails() {
        let mut cfg = WorkgroupConfig::balanced();
        cfg.preprocess = WorkgroupSize::new(0, 1, 1);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_exceeds_max_threads_fails() {
        let mut cfg = WorkgroupConfig::balanced();
        // 32 × 32 × 2 = 2048 > 1024
        cfg.sort = WorkgroupSize::new(32, 32, 2);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_non_power_of_two_fails() {
        let mut cfg = WorkgroupConfig::balanced();
        cfg.backward = WorkgroupSize::new(48, 1, 1); // 48 is not a power of 2
        assert!(cfg.validate().is_err());
    }

    // --- WorkgroupBenchmarker ---

    #[test]
    fn test_benchmarker_default_candidates() {
        let b = WorkgroupBenchmarker::new();
        assert!(!b.candidates.is_empty());
        assert!(b.candidates.contains(&32));
        assert!(b.candidates.contains(&64));
        assert!(b.candidates.contains(&128));
        assert!(b.candidates.contains(&256));
    }

    #[test]
    fn test_benchmark_result_count() {
        let b = WorkgroupBenchmarker::new().with_warmup(1).with_measure(2);
        let results = b.benchmark(|_ws| Duration::from_nanos(100));
        // Should produce one result per default candidate (4)
        assert_eq!(results.len(), b.candidates.len());
    }

    #[test]
    fn test_best_of_returns_lowest_mean() {
        let b = WorkgroupBenchmarker::new();

        let results = vec![
            WorkgroupBenchResult {
                size: WorkgroupSize::linear(32),
                mean_duration_us: 200.0,
                min_duration_us: 180.0,
                samples: 5,
            },
            WorkgroupBenchResult {
                size: WorkgroupSize::linear(64),
                mean_duration_us: 50.0,
                min_duration_us: 45.0,
                samples: 5,
            },
            WorkgroupBenchResult {
                size: WorkgroupSize::linear(128),
                mean_duration_us: 120.0,
                min_duration_us: 110.0,
                samples: 5,
            },
        ];

        let best = b.best_of(&results);
        assert_eq!(best, Some(WorkgroupSize::linear(64)));
    }

    #[test]
    fn test_best_of_empty_returns_none() {
        let b = WorkgroupBenchmarker::new();
        assert_eq!(b.best_of(&[]), None);
    }

    #[test]
    fn test_recommend_returns_config() {
        let b = WorkgroupBenchmarker::new().with_warmup(1).with_measure(2);

        let config = b.recommend(50_000, |_ws| Duration::from_nanos(1));
        // Should return a valid config
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_recommend_trivial_closure_works() {
        let b = WorkgroupBenchmarker::new().with_warmup(1).with_measure(3);

        // Trivial closure: time an instant capture
        let config = b.recommend(10_000, |ws| {
            let start = Instant::now();
            let _total = ws.total();
            start.elapsed()
        });

        // Result must be a well-formed config
        assert!(config.validate().is_ok());
        // Profile is Custom (benchmarker chose a winner) or Mobile (fallback)
        // Either way the config must have non-zero dimensions
        assert!(config.preprocess.total() > 0);
    }

    #[test]
    fn test_recommend_empty_candidates_fallback() {
        let b = WorkgroupBenchmarker::new()
            .with_candidates(vec![])
            .with_warmup(0)
            .with_measure(1);

        // No candidates → best_of returns None → falls back to adaptive
        let config = b.recommend(5_000, |_ws| Duration::from_nanos(1));
        assert_eq!(config.profile, WorkgroupProfile::Mobile);
    }

    #[test]
    fn test_custom_candidates_respected() {
        let b = WorkgroupBenchmarker::new()
            .with_candidates(vec![128])
            .with_warmup(0)
            .with_measure(2);

        let results = b.benchmark(|_ws| Duration::from_nanos(50));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].size, WorkgroupSize::linear(128));
    }

    #[test]
    fn test_duration_to_us_conversion() {
        let d = std::time::Duration::from_micros(42);
        assert!((duration_to_us(d) - 42.0).abs() < 0.01);
    }

    #[test]
    fn test_is_power_of_two() {
        assert!(is_power_of_two(1));
        assert!(is_power_of_two(2));
        assert!(is_power_of_two(64));
        assert!(is_power_of_two(256));
        assert!(!is_power_of_two(0));
        assert!(!is_power_of_two(48));
        assert!(!is_power_of_two(100));
    }

    #[test]
    fn test_tile_workgroup_size() {
        let cfg = WorkgroupConfig::balanced();
        // Tile is always 16×16 regardless of the main profile: that is what
        // rasterize_fwd.wgsl / rasterize_bwd.wgsl declare, one thread per pixel
        // of a 16-pixel tile.
        assert_eq!(cfg.tile, WorkgroupSize::square(16));
        assert_eq!(cfg.tile.total(), 256);
    }

    // --- Shipped geometry / overflow regressions ---

    #[test]
    fn test_total_saturates_instead_of_overflowing() {
        // validate() calls total() *before* it can reject the size, so an
        // absurd configuration must clamp rather than overflow (which would
        // panic in debug and could wrap back under the 1024 limit in release).
        let ws = WorkgroupSize::new(u32::MAX, u32::MAX, u32::MAX);
        assert_eq!(ws.total(), u32::MAX);

        let mut cfg = WorkgroupConfig::balanced();
        cfg.preprocess = ws;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_shipped_matches_compiled_shader_geometry() {
        let cfg = WorkgroupConfig::shipped();
        // Every 1-D kernel in shaders/ declares @workgroup_size(256)...
        assert_eq!(cfg.preprocess, WorkgroupSize::linear(256));
        assert_eq!(cfg.sort, WorkgroupSize::linear(256));
        // ...and both rasterization kernels declare @workgroup_size(16, 16).
        assert_eq!(cfg.rasterize, WorkgroupSize::square(16));
        assert_eq!(cfg.backward, WorkgroupSize::square(16));
        assert_eq!(cfg.tile, WorkgroupSize::square(16));
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_shipped_dispatch_counts_cover_every_element() {
        let cfg = WorkgroupConfig::shipped();
        // 1-D: 1000 elements over 256-thread workgroups.
        assert_eq!(cfg.preprocess.dispatch_count_x(1000), 4);
        assert_eq!(cfg.preprocess.dispatch_count_x(1024), 4);
        assert_eq!(cfg.preprocess.dispatch_count_x(1025), 5);
        // 2-D: a 100×70 image needs 7×5 tiles of 16×16 pixels.
        assert_eq!(cfg.rasterize.dispatch_count_x(100), 7);
        assert_eq!(cfg.rasterize.dispatch_count_y(70), 5);
    }

    /// Regression (F307): a config that asks for a different 1-D workgroup
    /// size must move the *dispatch grid* with it, or the host would launch
    /// the shipped 256-thread grid over kernels compiled for 64 threads and
    /// process only a quarter of the Gaussians.
    #[test]
    fn test_for_linear_size_retargets_only_the_one_dimensional_passes() {
        let cfg = WorkgroupConfig::for_linear_size(64);
        assert_eq!(cfg.preprocess, WorkgroupSize::linear(64));
        assert_eq!(cfg.sort, WorkgroupSize::linear(64));
        // The 2-D passes are the tile size itself and never move.
        assert_eq!(cfg.rasterize, WorkgroupConfig::SHIPPED_TILE);
        assert_eq!(cfg.backward, WorkgroupConfig::SHIPPED_TILE);
        assert_eq!(cfg.tile, WorkgroupConfig::SHIPPED_TILE);
        assert!(cfg.validate().is_ok());

        // 1000 elements over 64-thread workgroups, not 256-thread ones.
        assert_eq!(cfg.preprocess.dispatch_count_x(1000), 16);
    }

    /// `shipped()` must stay exactly `for_linear_size(256)`, so the default
    /// path is unchanged by the retargeting machinery.
    #[test]
    fn test_shipped_is_for_linear_size_of_the_shipped_attribute() {
        let shipped = WorkgroupConfig::shipped();
        let explicit = WorkgroupConfig::for_linear_size(WorkgroupConfig::SHIPPED_LINEAR.x);
        assert_eq!(shipped.preprocess, explicit.preprocess);
        assert_eq!(shipped.sort, explicit.sort);
        assert_eq!(shipped.rasterize, explicit.rasterize);
        assert_eq!(shipped.backward, explicit.backward);
        assert_eq!(shipped.tile, explicit.tile);
        assert_eq!(shipped.profile, explicit.profile);
    }

    #[test]
    fn test_bench_kernel_source_substitutes_workgroup_size() {
        let src = bench_kernel_source(128);
        assert!(src.contains("@workgroup_size(128)"), "{src}");
        assert!(src.contains("fn bench_kernel("), "{src}");
        // Braces must survive format!'s escaping.
        assert!(src.contains("if i >= params.x {"), "{src}");
        assert!(!src.contains("{{"), "{src}");
    }
}
