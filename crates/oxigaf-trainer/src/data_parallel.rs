//! Data parallel training infrastructure.
//!
//! Multi-worker training requires gradient synchronization across the workers,
//! and this module implements that synchronization for real:
//!
//! * [`run_parallel_step`] drives `num_workers` worker closures **concurrently**
//!   — one OS thread each via [`std::thread::scope`] — and all-reduces the
//!   gradients they produce.
//! * [`GradientAggregator`] performs a *bucketed* all-reduce.
//!   [`DataParallelConfig::bucket_size_mb`] decides how many whole parameters
//!   share a bucket ([`DataParallelConfig::bucket_plan`]), and buckets are
//!   reduced on parallel threads. Buckets never split a parameter, so the
//!   per-element accumulation order is exactly the serial order and the
//!   parallel result is **bit-identical** to the serial one.
//! * [`DataParallelConfig::gradient_compression`] enables symmetric `int8`
//!   quantization of every worker's contribution *before* it is summed
//!   ([`compress_gradients`]), which is the arithmetic a compressed all-reduce
//!   actually performs on the values that reach the reduction.
//!
//! # Scope of "parallel" here
//!
//! Workers are OS threads inside one process that share host memory. Each
//! worker closure owns its own device context (a wgpu queue plus rasterizer, a
//! CPU shard, ...) supplied by the caller, so several GPUs *are* driven
//! concurrently when the caller hands out one device per worker; enumerating
//! and creating those devices is the caller's job, not this module's.
//!
//! What this module does **not** implement is a cross-process / cross-node
//! transport: there is no NCCL- or MPI-style wire protocol. Consequently
//! [`SyncReport::estimated_bandwidth_mb`] is an explicit cost *model* — the
//! standard ring all-reduce volume `2·(n−1)/n · payload`, see
//! [`ring_all_reduce_bytes`] — and not a measurement.

use crate::TrainerError;
use std::future::Future;
use std::task::{Context, Poll, Waker};
use std::thread;

/// Size of one gradient element in bytes.
const BYTES_PER_GRAD: usize = std::mem::size_of::<f32>();

/// Default all-reduce bucket size in megabytes.
pub const DEFAULT_BUCKET_SIZE_MB: f32 = 25.0;

/// Largest magnitude representable by the symmetric `int8` gradient codec.
const INT8_MAX: f32 = 127.0;

/// Minimum number of gradient elements before [`GradientAggregator::aggregate`]
/// spawns threads; below this the spawn cost dominates the reduction.
const PARALLEL_REDUCE_MIN_ELEMENTS: usize = 1 << 16;

// ---------------------------------------------------------------------------
// Sync mode
// ---------------------------------------------------------------------------

/// Gradient synchronization mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncMode {
    /// Average gradients across all workers.
    AllReduce,
    /// Sum gradients (caller divides manually).
    AllReduceSum,
    /// No synchronization (single device).
    NoSync,
}

// ---------------------------------------------------------------------------
// DataParallelConfig
// ---------------------------------------------------------------------------

/// Configuration for data parallel training.
#[derive(Debug, Clone)]
pub struct DataParallelConfig {
    /// Number of workers, one OS thread and one device context each (default: 1).
    pub num_workers: usize,
    pub sync_mode: SyncMode,
    /// Quantize each worker's gradients to symmetric `int8` before they are
    /// summed (see [`compress_gradients`]).
    pub gradient_compression: bool,
    /// All-reduce bucket size in megabytes (default: [`DEFAULT_BUCKET_SIZE_MB`]).
    ///
    /// Parameters are grouped into buckets of at most this many bytes and each
    /// bucket is reduced as a unit; see [`DataParallelConfig::bucket_plan`].
    pub bucket_size_mb: f32,
}

impl DataParallelConfig {
    /// Single-device configuration — no gradient sync.
    pub fn single_device() -> Self {
        Self {
            num_workers: 1,
            sync_mode: SyncMode::NoSync,
            gradient_compression: false,
            bucket_size_mb: DEFAULT_BUCKET_SIZE_MB,
        }
    }

    /// Multi-worker configuration with AllReduce (average) sync.
    ///
    /// Each of the `n` workers runs on its own OS thread and drives its own
    /// caller-supplied device context; see [`run_parallel_step`].
    pub fn multi_gpu(n: usize) -> Self {
        Self {
            num_workers: n,
            sync_mode: SyncMode::AllReduce,
            gradient_compression: false,
            bucket_size_mb: DEFAULT_BUCKET_SIZE_MB,
        }
    }

    /// Configuration sized to the host's available parallelism.
    ///
    /// This counts **CPU** parallelism, not devices; see
    /// [`Self::from_gpu_devices`] to size the worker count from the GPUs that
    /// actually exist.
    ///
    /// Falls back to a single device when the platform cannot report a count.
    pub fn auto() -> Self {
        let workers = thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        if workers <= 1 {
            Self::single_device()
        } else {
            Self::multi_gpu(workers)
        }
    }

    /// Configuration sized to the GPU adapters this machine really exposes.
    ///
    /// [`Self::multi_gpu`] takes the worker count on trust; this asks the
    /// graphics stack instead, so a run configured this way cannot hand out
    /// more worker slots than there are devices to put them on. One adapter
    /// yields [`Self::single_device`] (no sync work for a lone GPU), several
    /// yield [`Self::multi_gpu`] with one worker per adapter.
    ///
    /// `backends` selects which graphics APIs to look at;
    /// [`gpu_backends_default`] is the usual choice.
    ///
    /// # Errors
    ///
    /// [`TrainerError::Training`] when no adapter matches `backends` — a
    /// headless CI box, a container without a GPU, or a `backends` mask no
    /// build feature supports. Callers that want a CPU-only fallback should
    /// handle that explicitly rather than have a "GPU" config silently mean
    /// "one thread".
    pub fn from_gpu_devices(backends: wgpu::Backends) -> Result<Self, TrainerError> {
        let devices = enumerate_gpu_devices(backends)?;
        match devices.len() {
            0 => Err(TrainerError::Training(format!(
                "no GPU adapter found for backends {backends:?}; cannot size a \
                 data-parallel configuration from real devices"
            ))),
            1 => Ok(Self::single_device()),
            n => Ok(Self::multi_gpu(n)),
        }
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), TrainerError> {
        if self.num_workers == 0 {
            return Err(TrainerError::InvalidConfig(
                "num_workers must be > 0".to_string(),
            ));
        }
        if !self.bucket_size_mb.is_finite() || self.bucket_size_mb <= 0.0 {
            return Err(TrainerError::InvalidConfig(format!(
                "bucket_size_mb must be a positive finite value, got {}",
                self.bucket_size_mb
            )));
        }
        Ok(())
    }

    /// Returns `true` if this is a multi-worker (distributed) configuration.
    pub fn is_distributed(&self) -> bool {
        self.num_workers > 1
    }

    /// Effective batch scale factor (linear batch scaling law).
    pub fn effective_batch_scale(&self) -> f32 {
        self.num_workers as f32
    }

    /// [`Self::bucket_size_mb`] expressed in bytes.
    ///
    /// An invalid (non-finite or non-positive) setting falls back to
    /// [`DEFAULT_BUCKET_SIZE_MB`] rather than producing a degenerate plan;
    /// [`Self::validate`] rejects such values up front.
    pub fn bucket_capacity_bytes(&self) -> usize {
        const MB: f64 = 1024.0 * 1024.0;
        let mb = if self.bucket_size_mb.is_finite() && self.bucket_size_mb > 0.0 {
            f64::from(self.bucket_size_mb)
        } else {
            f64::from(DEFAULT_BUCKET_SIZE_MB)
        };
        let bytes = mb * MB;
        if bytes >= usize::MAX as f64 {
            usize::MAX
        } else {
            (bytes as usize).max(BYTES_PER_GRAD)
        }
    }

    /// Group parameters into all-reduce buckets of at most
    /// [`Self::bucket_size_mb`].
    ///
    /// `param_lens[i]` is the element count of parameter `i`. Buckets hold
    /// **whole parameters only** and cover the parameters in order, so bucket
    /// boundaries never change the arithmetic of a reduction. A single
    /// parameter larger than the bucket capacity gets a bucket of its own that
    /// exceeds the capacity.
    pub fn bucket_plan(&self, param_lens: &[usize]) -> Vec<GradientBucket> {
        let cap_bytes = self.bucket_capacity_bytes();
        let mut plan: Vec<GradientBucket> = Vec::new();
        let mut start = 0usize;
        let mut len = 0usize;
        let mut bytes = 0usize;

        for (idx, &elements) in param_lens.iter().enumerate() {
            let param_bytes = elements.saturating_mul(BYTES_PER_GRAD);
            if len > 0 && bytes.saturating_add(param_bytes) > cap_bytes {
                plan.push(GradientBucket { start, len, bytes });
                start = idx;
                len = 0;
                bytes = 0;
            }
            len += 1;
            bytes = bytes.saturating_add(param_bytes);
        }

        if len > 0 {
            plan.push(GradientBucket { start, len, bytes });
        }
        plan
    }
}

// ---------------------------------------------------------------------------
// GPU device enumeration
// ---------------------------------------------------------------------------

/// What one physical adapter reports about itself.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuDeviceInfo {
    /// Index of this adapter in the enumeration order.
    pub index: usize,
    /// Driver-reported adapter name, e.g. `"Apple M2 Max"`.
    pub name: String,
    /// Graphics API this adapter was found through.
    pub backend: wgpu::Backend,
    /// Discrete / integrated / virtual / CPU classification.
    pub device_type: wgpu::DeviceType,
    /// PCI (or platform equivalent) vendor id.
    pub vendor: u32,
    /// PCI (or platform equivalent) device id.
    pub device: u32,
}

impl GpuDeviceInfo {
    /// Whether this adapter is a discrete GPU, i.e. one worth a worker of its
    /// own rather than a software or integrated fallback.
    pub fn is_discrete(&self) -> bool {
        self.device_type == wgpu::DeviceType::DiscreteGpu
    }
}

/// Backends worth enumerating for compute work.
///
/// Every real API, minus [`wgpu::Backends::NOOP`] — the no-op backend reports
/// an adapter that executes nothing, which must never be counted as a training
/// device.
pub fn gpu_backends_default() -> wgpu::Backends {
    wgpu::Backends::all().difference(wgpu::Backends::NOOP)
}

/// Enumerate the GPU adapters this machine exposes for `backends`.
///
/// This is the real device list from the graphics stack, not a guess: it is
/// what makes [`DataParallelConfig::from_gpu_devices`] able to size a run to
/// the hardware. Only adapters are enumerated — no logical device is
/// requested, so this stays cheap and does not fail on a machine whose GPU is
/// busy.
///
/// Returns an empty vector on a machine with no matching adapter; that is not
/// an error here (see [`DataParallelConfig::from_gpu_devices`], which does
/// treat it as one).
///
/// # Errors
///
/// [`TrainerError::Training`] when the enumeration future does not resolve
/// immediately. On every native backend it does — enumeration is a synchronous
/// driver query behind an `async` signature — so this signals a target (such
/// as WebGPU in a browser) whose adapter list only arrives via an async
/// executor, which this synchronous API cannot drive.
pub fn enumerate_gpu_devices(backends: wgpu::Backends) -> Result<Vec<GpuDeviceInfo>, TrainerError> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapters = poll_once(instance.enumerate_adapters(backends)).ok_or_else(|| {
        TrainerError::Training(
            "GPU adapter enumeration did not resolve synchronously; this target requires an \
             async executor to list adapters"
                .to_string(),
        )
    })?;

    Ok(adapters
        .iter()
        .enumerate()
        .map(|(index, adapter)| {
            let info = adapter.get_info();
            GpuDeviceInfo {
                index,
                name: info.name,
                backend: info.backend,
                device_type: info.device_type,
                vendor: info.vendor,
                device: info.device,
            }
        })
        .collect())
}

/// Poll `future` exactly once and return its output if it is already ready.
///
/// Deliberately not a `block_on`: this crate has no async runtime, and the one
/// future it drives (adapter enumeration) completes on its first poll on every
/// native backend. Anything that actually suspends is reported to the caller
/// as an error instead of being parked on, so a caller can never deadlock a
/// training thread here.
fn poll_once<F: Future>(future: F) -> Option<F::Output> {
    let mut future = std::pin::pin!(future);
    let waker = Waker::noop();
    let mut cx = Context::from_waker(waker);
    match future.as_mut().poll(&mut cx) {
        Poll::Ready(value) => Some(value),
        Poll::Pending => None,
    }
}

// ---------------------------------------------------------------------------
// GradientBucket
// ---------------------------------------------------------------------------

/// A contiguous run of whole parameters reduced as one all-reduce unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GradientBucket {
    /// Index of the first parameter in the bucket.
    pub start: usize,
    /// Number of parameters in the bucket (always ≥ 1).
    pub len: usize,
    /// Total gradient bytes held by the bucket for one worker.
    pub bytes: usize,
}

impl GradientBucket {
    /// Parameter index range covered by this bucket.
    pub fn range(&self) -> std::ops::Range<usize> {
        self.start..self.start.saturating_add(self.len)
    }

    /// Number of gradient elements in the bucket for one worker.
    pub fn elements(&self) -> usize {
        self.bytes / BYTES_PER_GRAD
    }
}

// ---------------------------------------------------------------------------
// Gradient compression
// ---------------------------------------------------------------------------

/// Round-trip a gradient buffer through symmetric `int8` compression.
///
/// This is the arithmetic a compressed all-reduce performs: the buffer is
/// quantized with a single per-buffer scale `max_abs / 127`, "transmitted" as
/// `int8` codes, and dequantized before it is summed. The returned buffer
/// therefore holds the values the reduction actually sees.
///
/// * An all-zero buffer (or one with no finite values) is returned unchanged —
///   there is no scale to divide by.
/// * Non-finite elements pass through untouched so that downstream NaN/Inf
///   detection still fires instead of seeing a saturated `0`.
pub fn compress_gradients(grads: &[f32]) -> Vec<f32> {
    let mut out = Vec::new();
    compress_into(grads, &mut out);
    out
}

/// [`compress_gradients`] into a caller-owned buffer, avoiding a per-call
/// allocation in the reduction loop.
fn compress_into(src: &[f32], dst: &mut Vec<f32>) {
    dst.clear();
    dst.reserve(src.len());

    let max_abs = src.iter().fold(
        0.0f32,
        |acc, &v| {
            if v.is_finite() {
                acc.max(v.abs())
            } else {
                acc
            }
        },
    );

    // `max_abs` is finite by construction — non-finite inputs are skipped — so
    // a zero maximum is the only degenerate case, and it has no scale.
    if max_abs <= 0.0 {
        dst.extend_from_slice(src);
        return;
    }

    let scale = max_abs / INT8_MAX;
    let inv_scale = INT8_MAX / max_abs;
    for &v in src {
        if !v.is_finite() {
            dst.push(v);
            continue;
        }
        // Clamp before the cast: `f32 as i8` saturates silently, which would
        // turn an out-of-range value into quiet garbage.
        let code = (v * inv_scale).round().clamp(-INT8_MAX, INT8_MAX) as i8;
        dst.push(f32::from(code) * scale);
    }
}

/// Bytes moved **per worker** by a ring all-reduce of `payload_bytes`.
///
/// A ring all-reduce is a reduce-scatter followed by an all-gather, each
/// `n − 1` steps of one `payload / n` chunk, hence `2·(n−1)/n · payload`.
/// A single worker transfers nothing.
pub fn ring_all_reduce_bytes(payload_bytes: usize, num_workers: usize) -> usize {
    if num_workers <= 1 {
        return 0;
    }
    let chunk = payload_bytes / num_workers;
    chunk.saturating_mul(2).saturating_mul(num_workers - 1)
}

// ---------------------------------------------------------------------------
// GradientAggregator
// ---------------------------------------------------------------------------

/// Tracks gradient aggregation across workers for one optimizer step.
pub struct GradientAggregator {
    /// Configuration this aggregator reduces with.
    ///
    /// `num_workers` is fixed at construction time: the buffers are sized once
    /// in [`GradientAggregator::new`]. Changing it afterwards makes every
    /// method return an error instead of indexing past the buffers — build a
    /// new aggregator to change the worker count.
    pub config: DataParallelConfig,
    /// Accumulated gradients from each worker: `[worker_idx][param_idx]`.
    worker_grads: Vec<Vec<Vec<f32>>>,
    /// Per-worker completion flags.
    worker_done: Vec<bool>,
    completed_workers: usize,
    generation: u64,
}

impl GradientAggregator {
    /// Create a new aggregator.
    ///
    /// Initialises `worker_grads` as `num_workers × num_params` with empty
    /// inner `Vec<f32>` (gradients are supplied via `submit_gradients`).
    pub fn new(config: DataParallelConfig, num_params: usize) -> Self {
        let num_workers = config.num_workers;
        let worker_grads: Vec<Vec<Vec<f32>>> = (0..num_workers)
            .map(|_| (0..num_params).map(|_| Vec::new()).collect())
            .collect();
        let worker_done = vec![false; num_workers];
        Self {
            config,
            worker_grads,
            worker_done,
            completed_workers: 0,
            generation: 0,
        }
    }

    /// Number of workers this aggregator was built for.
    pub fn num_workers(&self) -> usize {
        self.worker_done.len()
    }

    /// Number of parameters this aggregator was built for.
    pub fn num_params(&self) -> usize {
        self.worker_grads.first().map(|w| w.len()).unwrap_or(0)
    }

    /// Reject a `config.num_workers` that no longer matches the buffers.
    ///
    /// `config` is public, so it can be mutated after construction; without
    /// this check the reduction would index past `worker_grads` and panic.
    fn ensure_consistent(&self) -> Result<(), TrainerError> {
        let registered = self.worker_done.len();
        if self.config.num_workers != registered || self.worker_grads.len() != registered {
            return Err(TrainerError::InvalidConfig(format!(
                "GradientAggregator holds buffers for {registered} worker(s) but \
                 config.num_workers is {}; rebuild it with GradientAggregator::new \
                 after changing the worker count",
                self.config.num_workers
            )));
        }
        Ok(())
    }

    /// Length of parameter `p`, taken from the first worker that submitted a
    /// non-empty buffer for it (`0` when nobody did).
    fn param_len(&self, p: usize) -> usize {
        self.worker_grads
            .iter()
            .filter_map(|wg| wg.get(p))
            .map(|grads| grads.len())
            .find(|&l| l > 0)
            .unwrap_or(0)
    }

    /// Per-parameter element counts, as used for bucketing.
    fn param_lens(&self) -> Vec<usize> {
        (0..self.num_params()).map(|p| self.param_len(p)).collect()
    }

    /// Submit gradient buffer for a parameter from a specific worker.
    ///
    /// Errors if `worker_idx >= num_workers` or `param_idx >= num_params`.
    pub fn submit_gradients(
        &mut self,
        worker_idx: usize,
        param_idx: usize,
        grads: Vec<f32>,
    ) -> Result<(), TrainerError> {
        self.ensure_consistent()?;
        let num_workers = self.num_workers();
        let num_params = self.num_params();

        if worker_idx >= num_workers {
            return Err(TrainerError::Training(format!(
                "worker_idx {worker_idx} out of range (num_workers={num_workers})"
            )));
        }
        if param_idx >= num_params {
            return Err(TrainerError::Training(format!(
                "param_idx {param_idx} out of range (num_params={num_params})"
            )));
        }

        let slot = self
            .worker_grads
            .get_mut(worker_idx)
            .and_then(|wg| wg.get_mut(param_idx))
            .ok_or_else(|| {
                TrainerError::Training(format!(
                    "gradient slot [worker {worker_idx}][param {param_idx}] is missing"
                ))
            })?;
        *slot = grads;
        Ok(())
    }

    /// Submit every parameter of one worker and mark that worker done.
    ///
    /// `grads` is indexed `[param_idx]` and must hold exactly `num_params`
    /// buffers, in the parameter order shared by all workers.
    pub fn submit_worker(
        &mut self,
        worker_idx: usize,
        grads: Vec<Vec<f32>>,
    ) -> Result<(), TrainerError> {
        self.ensure_consistent()?;
        let num_params = self.num_params();
        if grads.len() != num_params {
            return Err(TrainerError::Training(format!(
                "worker {worker_idx} submitted {} parameter buffer(s), expected {num_params}",
                grads.len()
            )));
        }
        // Check everything that can fail *before* the first write, so a
        // rejected submission never leaves half of a worker's gradients behind.
        let num_workers = self.num_workers();
        let generation = self.generation;
        let already_done = *self.worker_done.get(worker_idx).ok_or_else(|| {
            TrainerError::Training(format!(
                "worker_idx {worker_idx} out of range (num_workers={num_workers})"
            ))
        })?;
        if already_done {
            return Err(TrainerError::Training(format!(
                "worker {worker_idx} already marked done for generation {generation}"
            )));
        }
        for (param_idx, buffer) in grads.into_iter().enumerate() {
            self.submit_gradients(worker_idx, param_idx, buffer)?;
        }
        self.mark_worker_done(worker_idx)
    }

    /// Mark a worker as done for this iteration.
    ///
    /// Errors if the worker was already marked done.
    pub fn mark_worker_done(&mut self, worker_idx: usize) -> Result<(), TrainerError> {
        self.ensure_consistent()?;
        let num_workers = self.num_workers();
        if worker_idx >= num_workers {
            return Err(TrainerError::Training(format!(
                "worker_idx {worker_idx} out of range (num_workers={num_workers})"
            )));
        }
        let generation = self.generation;
        let flag = self.worker_done.get_mut(worker_idx).ok_or_else(|| {
            TrainerError::Training(format!("worker slot {worker_idx} is missing"))
        })?;
        if *flag {
            return Err(TrainerError::Training(format!(
                "worker {worker_idx} already marked done for generation {generation}"
            )));
        }
        *flag = true;
        self.completed_workers += 1;
        Ok(())
    }

    /// Returns `true` when all registered workers have called
    /// [`Self::mark_worker_done`].
    pub fn is_ready(&self) -> bool {
        self.completed_workers == self.num_workers()
    }

    /// Aggregate gradients across workers according to [`SyncMode`].
    ///
    /// Returns `Vec<Vec<f32>>` indexed `[param_idx][grad_element]`.
    ///
    /// * **AllReduce**: element-wise average.
    /// * **AllReduceSum**: element-wise sum.
    /// * **NoSync**: returns worker 0's gradients unchanged.
    ///
    /// The reduction runs bucket-by-bucket (see
    /// [`DataParallelConfig::bucket_plan`]) and spreads the buckets over
    /// worker threads once there is enough work; because a bucket owns whole
    /// parameters the result is bit-identical to a serial reduction.
    ///
    /// Missing / empty worker grad buffers are treated as zeros. A *non-empty*
    /// buffer whose length disagrees with the other workers is an error
    /// ([`TrainerError::GradientSizeMismatch`]) rather than a silently
    /// truncated sum.
    pub fn aggregate(&self) -> Result<Vec<Vec<f32>>, TrainerError> {
        self.ensure_consistent()?;

        let num_workers = self.num_workers();
        if num_workers == 0 {
            return Err(TrainerError::InvalidConfig(
                "no workers registered".to_string(),
            ));
        }
        if !self.is_ready() {
            return Err(TrainerError::Training(format!(
                "Cannot aggregate: only {}/{} workers done in generation {}",
                self.completed_workers, num_workers, self.generation
            )));
        }

        match self.config.sync_mode {
            SyncMode::NoSync => {
                // Return worker 0's gradients
                let worker0 = self
                    .worker_grads
                    .first()
                    .ok_or_else(|| TrainerError::Training("no workers registered".to_string()))?;
                Ok(worker0.to_vec())
            }

            SyncMode::AllReduce | SyncMode::AllReduceSum => {
                let plan = self.config.bucket_plan(&self.param_lens());
                self.reduce_buckets(&plan)
            }
        }
    }

    /// Reduce every bucket of `plan`, in parallel when it pays off.
    fn reduce_buckets(&self, plan: &[GradientBucket]) -> Result<Vec<Vec<f32>>, TrainerError> {
        if plan.is_empty() {
            return Ok(Vec::new());
        }

        let num_params = self.num_params();
        let total_elements: usize = plan.iter().map(|bucket| bucket.elements()).sum();
        let parallel = plan.len() > 1 && total_elements >= PARALLEL_REDUCE_MIN_ELEMENTS;

        let mut result: Vec<Vec<f32>> = Vec::with_capacity(num_params);

        if !parallel {
            for bucket in plan {
                result.append(&mut self.reduce_bucket(bucket)?);
            }
            return Ok(result);
        }

        // One lane per available core; each lane reduces a contiguous run of
        // buckets, so lane boundaries — like bucket boundaries — fall between
        // whole parameters and never reorder an accumulation.
        let lanes = thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .clamp(1, plan.len());
        let lane_size = plan.len().div_ceil(lanes).max(1);

        let lane_results = thread::scope(|scope| {
            let handles: Vec<_> = plan
                .chunks(lane_size)
                .map(|lane| {
                    scope.spawn(move || -> Result<Vec<Vec<f32>>, TrainerError> {
                        let mut reduced: Vec<Vec<f32>> = Vec::new();
                        for bucket in lane {
                            reduced.append(&mut self.reduce_bucket(bucket)?);
                        }
                        Ok(reduced)
                    })
                })
                .collect();

            handles
                .into_iter()
                .enumerate()
                .map(|(lane_idx, handle)| {
                    handle.join().unwrap_or_else(|_| {
                        Err(TrainerError::Training(format!(
                            "gradient reduction lane {lane_idx} panicked"
                        )))
                    })
                })
                .collect::<Vec<_>>()
        });

        for lane_result in lane_results {
            result.append(&mut lane_result?);
        }
        Ok(result)
    }

    /// Reduce the parameters of a single bucket across all workers.
    fn reduce_bucket(&self, bucket: &GradientBucket) -> Result<Vec<Vec<f32>>, TrainerError> {
        let num_workers = self.num_workers();
        let mut reduced: Vec<Vec<f32>> = Vec::with_capacity(bucket.len);
        // Reused across parameters so compression allocates once per bucket.
        let mut staging: Vec<f32> = Vec::new();

        for p in bucket.range() {
            let param_len = self.param_len(p);
            if param_len == 0 {
                reduced.push(Vec::new());
                continue;
            }

            let mut acc = vec![0.0f32; param_len];
            for (w, worker) in self.worker_grads.iter().enumerate() {
                let worker_param = worker.get(p).ok_or_else(|| {
                    TrainerError::Training(format!(
                        "gradient slot [worker {w}][param {p}] is missing"
                    ))
                })?;
                if worker_param.is_empty() {
                    // Treat missing grads as zeros — nothing to add.
                    continue;
                }
                if worker_param.len() != param_len {
                    return Err(TrainerError::GradientSizeMismatch {
                        expected: param_len,
                        actual: worker_param.len(),
                    });
                }

                let contribution: &[f32] = if self.config.gradient_compression {
                    compress_into(worker_param, &mut staging);
                    staging.as_slice()
                } else {
                    worker_param.as_slice()
                };
                for (a, &g) in acc.iter_mut().zip(contribution.iter()) {
                    *a += g;
                }
            }

            if self.config.sync_mode == SyncMode::AllReduce {
                let inv = 1.0 / num_workers as f32;
                for a in acc.iter_mut() {
                    *a *= inv;
                }
            }

            reduced.push(acc);
        }

        Ok(reduced)
    }

    /// Reset state for the next iteration and increment the generation counter.
    pub fn reset(&mut self) {
        let num_params = self.num_params();
        let num_workers = self.num_workers();

        // Clear all gradient buffers
        for wg in self.worker_grads.iter_mut() {
            for pg in wg.iter_mut() {
                pg.clear();
            }
        }
        // Reset per-worker done flags
        for flag in self.worker_done.iter_mut() {
            *flag = false;
        }
        self.completed_workers = 0;
        self.generation += 1;

        // Ensure structure is correct (safety re-init if needed)
        debug_assert_eq!(self.worker_grads.len(), num_workers);
        debug_assert!(self.worker_grads.iter().all(|wg| wg.len() == num_params));
    }

    /// Current generation (number of completed aggregation cycles).
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Single-worker gradient payload in bytes, i.e. what one worker would put
    /// on the wire — `int8` codes plus one `f32` scale per parameter when
    /// compression is enabled.
    fn payload_bytes(&self, param_lens: &[usize]) -> usize {
        param_lens
            .iter()
            .map(|&len| {
                if len == 0 {
                    0
                } else if self.config.gradient_compression {
                    len.saturating_add(BYTES_PER_GRAD)
                } else {
                    len.saturating_mul(BYTES_PER_GRAD)
                }
            })
            .sum()
    }

    /// Build a [`SyncReport`] for observability.
    pub fn sync_report(&self) -> SyncReport {
        let param_lens = self.param_lens();
        let num_workers = self.num_workers();

        let total_gradient_floats: usize = self
            .worker_grads
            .iter()
            .flat_map(|wg| wg.iter())
            .map(|grads| grads.len())
            .sum();

        let buckets = self.config.bucket_plan(&param_lens).len();
        let transferred_bytes = match self.config.sync_mode {
            SyncMode::NoSync => 0,
            SyncMode::AllReduce | SyncMode::AllReduceSum => {
                ring_all_reduce_bytes(self.payload_bytes(&param_lens), num_workers)
            }
        };
        let estimated_bandwidth_mb = transferred_bytes as f32 / (1024.0 * 1024.0);

        SyncReport {
            num_workers,
            sync_mode: self.config.sync_mode,
            total_params: param_lens.len(),
            total_gradient_floats,
            buckets,
            gradient_compression: self.config.gradient_compression,
            estimated_bandwidth_mb,
            generation: self.generation,
        }
    }
}

// ---------------------------------------------------------------------------
// SyncReport
// ---------------------------------------------------------------------------

/// Observability report for one data-parallel synchronization.
#[derive(Debug, Clone)]
pub struct SyncReport {
    pub num_workers: usize,
    pub sync_mode: SyncMode,
    pub total_params: usize,
    pub total_gradient_floats: usize,
    /// Number of all-reduce buckets the parameters were grouped into.
    pub buckets: usize,
    /// Whether contributions are `int8`-compressed before the reduction.
    pub gradient_compression: bool,
    /// Modelled MB moved **per worker** by a ring all-reduce of the
    /// single-worker payload — `2·(n−1)/n · payload`, see
    /// [`ring_all_reduce_bytes`]. Zero for [`SyncMode::NoSync`].
    pub estimated_bandwidth_mb: f32,
    pub generation: u64,
}

impl SyncReport {
    /// Human-readable summary string.
    pub fn format(&self) -> String {
        format!(
            "SyncReport {{ generation={}, workers={}, sync_mode={:?}, \
             params={}, grad_floats={}, buckets={}, compression={}, bandwidth_mb={:.3} }}",
            self.generation,
            self.num_workers,
            self.sync_mode,
            self.total_params,
            self.total_gradient_floats,
            self.buckets,
            if self.gradient_compression {
                "int8"
            } else {
                "off"
            },
            self.estimated_bandwidth_mb,
        )
    }
}

// ---------------------------------------------------------------------------
// Parallel step driver
// ---------------------------------------------------------------------------

/// Result of one data-parallel step.
#[derive(Debug, Clone)]
pub struct StepOutcome {
    /// All-reduced gradients, indexed `[param_idx][grad_element]`.
    pub gradients: Vec<Vec<f32>>,
    /// Observability report for the synchronization that produced them.
    pub report: SyncReport,
}

/// Run one data-parallel step: every worker computes its gradients on its own
/// OS thread, then the results are all-reduced according to `config`.
///
/// `worker_states` supplies one mutable state per worker — typically that
/// worker's device context (wgpu queue, rasterizer, ...) plus its shard of the
/// batch. `compute_grads` is invoked as `compute_grads(worker_idx, &mut state)`
/// and must return one buffer per parameter in the parameter order shared by
/// all workers.
///
/// Gradients are collected and submitted in worker order, so the reduction is
/// deterministic regardless of the order in which the threads finish.
///
/// # Errors
///
/// * [`TrainerError::InvalidConfig`] when `config` is invalid or
///   `worker_states.len() != config.num_workers`.
/// * The worker's own error when `compute_grads` fails, or
///   [`TrainerError::Training`] when a worker thread panics.
/// * [`TrainerError::GradientSizeMismatch`] when the workers disagree on a
///   parameter's length.
///
/// # Example
///
/// ```
/// use oxigaf_trainer::data_parallel::{run_parallel_step, DataParallelConfig};
///
/// let config = DataParallelConfig::multi_gpu(2);
/// // One state per worker; in a real trainer this holds a device context.
/// let mut shards: Vec<f32> = vec![1.0, 3.0];
/// let outcome = run_parallel_step(&config, shards.as_mut_slice(), |_idx, shard| {
///     Ok(vec![vec![*shard; 4]])
/// })?;
/// assert_eq!(outcome.gradients[0], vec![2.0f32; 4]);
/// # Ok::<(), oxigaf_trainer::TrainerError>(())
/// ```
pub fn run_parallel_step<W, F>(
    config: &DataParallelConfig,
    worker_states: &mut [W],
    compute_grads: F,
) -> Result<StepOutcome, TrainerError>
where
    W: Send,
    F: Fn(usize, &mut W) -> Result<Vec<Vec<f32>>, TrainerError> + Sync,
{
    config.validate()?;
    if worker_states.len() != config.num_workers {
        return Err(TrainerError::InvalidConfig(format!(
            "run_parallel_step: {} worker state(s) supplied but config.num_workers is {}",
            worker_states.len(),
            config.num_workers
        )));
    }

    let compute = &compute_grads;
    let per_worker = thread::scope(|scope| {
        let handles: Vec<_> = worker_states
            .iter_mut()
            .enumerate()
            .map(|(idx, state)| scope.spawn(move || compute(idx, state)))
            .collect();

        handles
            .into_iter()
            .enumerate()
            .map(|(idx, handle)| {
                handle.join().unwrap_or_else(|_| {
                    Err(TrainerError::Training(format!(
                        "data-parallel worker {idx} panicked"
                    )))
                })
            })
            .collect::<Vec<_>>()
    });

    let mut worker_grads: Vec<Vec<Vec<f32>>> = Vec::with_capacity(per_worker.len());
    for result in per_worker {
        worker_grads.push(result?);
    }

    let num_params = worker_grads.first().map(|g| g.len()).unwrap_or(0);
    let mut aggregator = GradientAggregator::new(config.clone(), num_params);
    for (idx, grads) in worker_grads.into_iter().enumerate() {
        aggregator.submit_worker(idx, grads)?;
    }

    let gradients = aggregator.aggregate()?;
    let report = aggregator.sync_report();
    Ok(StepOutcome { gradients, report })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use std::collections::HashSet;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    /// Deterministic pseudo-random values for reduction tests.
    fn lcg_values(seed: u64, count: usize) -> Vec<f32> {
        let mut state = seed | 1;
        (0..count)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                let bits = (state >> 33) as u32;
                (bits as f32 / u32::MAX as f32) * 2.0 - 1.0
            })
            .collect()
    }

    // --- Config tests -------------------------------------------------------

    #[test]
    fn test_config_single_device() {
        let cfg = DataParallelConfig::single_device();
        assert_eq!(cfg.num_workers, 1);
        assert_eq!(cfg.sync_mode, SyncMode::NoSync);
        assert!(!cfg.gradient_compression);
        assert_abs_diff_eq!(cfg.bucket_size_mb, 25.0, epsilon = 1e-6);
    }

    #[test]
    fn test_config_multi_gpu() {
        let cfg = DataParallelConfig::multi_gpu(4);
        assert_eq!(cfg.num_workers, 4);
        assert_eq!(cfg.sync_mode, SyncMode::AllReduce);
    }

    #[test]
    fn test_config_auto_is_valid() {
        let cfg = DataParallelConfig::auto();
        assert!(cfg.validate().is_ok());
        assert!(cfg.num_workers >= 1);
    }

    // --- GPU device enumeration (F122) --------------------------------------
    // `multi_gpu(n)` takes the worker count on trust; these cover the path
    // that sizes it from adapters that actually exist. They must pass on a
    // headless box too, so nothing here asserts a device count.

    #[test]
    fn test_gpu_backends_default_excludes_noop() {
        let backends = gpu_backends_default();
        assert!(
            !backends.contains(wgpu::Backends::NOOP),
            "the no-op backend executes nothing and must never be counted as a device"
        );
        assert!(backends.contains(wgpu::Backends::PRIMARY));
    }

    #[test]
    fn test_enumerate_gpu_devices_reports_consistent_info() {
        let devices = enumerate_gpu_devices(gpu_backends_default())
            .expect("adapter enumeration must resolve synchronously on native targets");
        for (i, dev) in devices.iter().enumerate() {
            assert_eq!(dev.index, i, "indices must match enumeration order");
            assert_ne!(
                dev.backend,
                wgpu::Backend::Noop,
                "the no-op backend was excluded from the mask"
            );
            assert_eq!(
                dev.is_discrete(),
                dev.device_type == wgpu::DeviceType::DiscreteGpu
            );
        }
        // Enumerating an empty backend mask is well defined and finds nothing.
        let none = enumerate_gpu_devices(wgpu::Backends::empty()).expect("empty mask enumerates");
        assert!(none.is_empty());
    }

    #[test]
    fn test_from_gpu_devices_matches_the_real_device_count() {
        let devices = enumerate_gpu_devices(gpu_backends_default()).expect("enumerate");
        match DataParallelConfig::from_gpu_devices(gpu_backends_default()) {
            Ok(cfg) => {
                assert!(!devices.is_empty(), "a config implies at least one adapter");
                assert_eq!(
                    cfg.num_workers,
                    devices.len(),
                    "worker count must equal the real adapter count"
                );
                assert!(cfg.validate().is_ok());
                if devices.len() == 1 {
                    assert_eq!(cfg.sync_mode, SyncMode::NoSync, "a lone GPU needs no sync");
                } else {
                    assert_eq!(cfg.sync_mode, SyncMode::AllReduce);
                }
            }
            Err(err) => {
                // Only legitimate when the machine really has no adapter.
                assert!(
                    devices.is_empty(),
                    "errored despite {} adapters",
                    devices.len()
                );
                assert!(matches!(err, TrainerError::Training(_)), "{err:?}");
            }
        }
    }

    #[test]
    fn test_from_gpu_devices_errors_when_no_backend_can_match() {
        // An empty backend mask can never yield an adapter, so this must be a
        // reported error rather than a silent one-worker "GPU" config.
        let err = DataParallelConfig::from_gpu_devices(wgpu::Backends::empty())
            .expect_err("an empty backend mask has no devices");
        match err {
            TrainerError::Training(msg) => assert!(msg.contains("no GPU adapter"), "{msg}"),
            other => panic!("expected Training error, got {other:?}"),
        }
    }

    #[test]
    fn test_config_validate() {
        assert!(DataParallelConfig::single_device().validate().is_ok());
        assert!(DataParallelConfig::multi_gpu(8).validate().is_ok());

        let mut bad = DataParallelConfig::single_device();
        bad.num_workers = 0;
        assert!(bad.validate().is_err());

        let mut bad2 = DataParallelConfig::single_device();
        bad2.bucket_size_mb = -1.0;
        assert!(bad2.validate().is_err());
    }

    #[test]
    fn test_config_is_distributed() {
        assert!(!DataParallelConfig::single_device().is_distributed());
        assert!(DataParallelConfig::multi_gpu(2).is_distributed());
        assert!(DataParallelConfig::multi_gpu(8).is_distributed());
    }

    #[test]
    fn test_config_effective_batch_scale() {
        let cfg = DataParallelConfig::multi_gpu(4);
        assert_abs_diff_eq!(cfg.effective_batch_scale(), 4.0, epsilon = 1e-6);
    }

    // --- Bucketing ----------------------------------------------------------

    #[test]
    fn test_bucket_plan_groups_whole_params_under_cap() {
        let mut cfg = DataParallelConfig::multi_gpu(2);
        // 1 KiB cap: exactly two 128-element (512 byte) parameters per bucket.
        cfg.bucket_size_mb = 1024.0 / (1024.0 * 1024.0);
        assert_eq!(cfg.bucket_capacity_bytes(), 1024);

        let plan = cfg.bucket_plan(&[128, 128, 128, 128, 128]);
        assert_eq!(plan.len(), 3);
        assert_eq!(
            plan[0],
            GradientBucket {
                start: 0,
                len: 2,
                bytes: 1024
            }
        );
        assert_eq!(
            plan[1],
            GradientBucket {
                start: 2,
                len: 2,
                bytes: 1024
            }
        );
        assert_eq!(
            plan[2],
            GradientBucket {
                start: 4,
                len: 1,
                bytes: 512
            }
        );

        // Buckets cover every parameter exactly once, in order.
        let covered: Vec<usize> = plan.iter().flat_map(|b| b.range()).collect();
        assert_eq!(covered, vec![0, 1, 2, 3, 4]);
        assert_eq!(plan[0].elements(), 256);
    }

    #[test]
    fn test_bucket_plan_oversized_param_gets_own_bucket() {
        let mut cfg = DataParallelConfig::multi_gpu(2);
        cfg.bucket_size_mb = 1024.0 / (1024.0 * 1024.0);

        let plan = cfg.bucket_plan(&[64, 4096, 64]);
        assert_eq!(plan.len(), 3);
        assert_eq!(plan[1].start, 1);
        assert_eq!(plan[1].len, 1);
        assert!(plan[1].bytes > cfg.bucket_capacity_bytes());
    }

    #[test]
    fn test_bucket_plan_default_keeps_small_model_in_one_bucket() {
        let cfg = DataParallelConfig::multi_gpu(4);
        let plan = cfg.bucket_plan(&[1024; 16]);
        assert_eq!(plan.len(), 1);
        assert_eq!(plan[0].len, 16);

        assert!(cfg.bucket_plan(&[]).is_empty());
    }

    #[test]
    fn test_bucket_capacity_falls_back_on_invalid_size() {
        let mut cfg = DataParallelConfig::single_device();
        cfg.bucket_size_mb = f32::NAN;
        assert_eq!(cfg.bucket_capacity_bytes(), 25 * 1024 * 1024);
        cfg.bucket_size_mb = 0.0;
        assert_eq!(cfg.bucket_capacity_bytes(), 25 * 1024 * 1024);
    }

    // --- Compression --------------------------------------------------------

    #[test]
    fn test_compress_gradients_bounded_error() {
        let values = lcg_values(7, 512);
        let compressed = compress_gradients(&values);
        assert_eq!(compressed.len(), values.len());

        let max_abs = values.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let step = max_abs / INT8_MAX;
        for (original, quantized) in values.iter().zip(compressed.iter()) {
            assert!(
                (original - quantized).abs() <= step * 0.5 + 1e-6,
                "quantization error too large: {original} vs {quantized}"
            );
            if *original != 0.0 {
                assert!(quantized.is_finite());
            }
        }
    }

    #[test]
    fn test_compress_gradients_zero_and_nonfinite_are_safe() {
        let zeros = compress_gradients(&[0.0f32; 8]);
        assert_eq!(zeros, vec![0.0f32; 8]);

        let mixed = compress_gradients(&[f32::NAN, 1.0, -1.0, f32::INFINITY]);
        assert!(mixed[0].is_nan());
        assert_abs_diff_eq!(mixed[1], 1.0, epsilon = 1e-2);
        assert_abs_diff_eq!(mixed[2], -1.0, epsilon = 1e-2);
        assert!(mixed[3].is_infinite());

        // A buffer with no finite values has no scale and passes through.
        let all_nan = compress_gradients(&[f32::NAN, f32::NAN]);
        assert!(all_nan.iter().all(|v| v.is_nan()));
    }

    #[test]
    fn test_aggregate_with_compression_matches_within_quantization_error() {
        let a = lcg_values(11, 256);
        let b = lcg_values(29, 256);

        let mut plain = GradientAggregator::new(DataParallelConfig::multi_gpu(2), 1);
        plain.submit_worker(0, vec![a.clone()]).expect("submit");
        plain.submit_worker(1, vec![b.clone()]).expect("submit");
        let exact = plain.aggregate().expect("aggregate");

        let mut cfg = DataParallelConfig::multi_gpu(2);
        cfg.gradient_compression = true;
        let mut compressed = GradientAggregator::new(cfg, 1);
        compressed.submit_worker(0, vec![a]).expect("submit");
        compressed.submit_worker(1, vec![b]).expect("submit");
        let approx_result = compressed.aggregate().expect("aggregate");

        assert_eq!(approx_result[0].len(), exact[0].len());
        for (lhs, rhs) in exact[0].iter().zip(approx_result[0].iter()) {
            assert!(
                (lhs - rhs).abs() < 0.02,
                "compressed all-reduce drifted: {lhs} vs {rhs}"
            );
        }
        // Compression must actually change the values, not silently no-op.
        assert!(exact[0] != approx_result[0]);
        assert!(compressed.sync_report().gradient_compression);
    }

    // --- Aggregator tests ---------------------------------------------------

    #[test]
    fn test_aggregator_new() {
        let cfg = DataParallelConfig::multi_gpu(3);
        let agg = GradientAggregator::new(cfg, 10);
        assert_eq!(agg.worker_grads.len(), 3);
        for wg in &agg.worker_grads {
            assert_eq!(wg.len(), 10);
            for pg in wg {
                assert!(pg.is_empty());
            }
        }
        assert_eq!(agg.generation(), 0);
        assert_eq!(agg.num_workers(), 3);
        assert_eq!(agg.num_params(), 10);
    }

    #[test]
    fn test_submit_gradients() {
        let cfg = DataParallelConfig::multi_gpu(2);
        let mut agg = GradientAggregator::new(cfg, 3);

        let grads = vec![1.0f32, 2.0f32, 3.0f32];
        assert!(agg.submit_gradients(0, 1, grads.clone()).is_ok());
        assert_eq!(agg.worker_grads[0][1], grads);

        // Out-of-range worker
        assert!(agg.submit_gradients(5, 0, vec![]).is_err());
        // Out-of-range param
        assert!(agg.submit_gradients(0, 10, vec![]).is_err());
    }

    #[test]
    fn test_submit_worker_checks_param_count() {
        let cfg = DataParallelConfig::multi_gpu(2);
        let mut agg = GradientAggregator::new(cfg, 2);

        assert!(agg
            .submit_worker(0, vec![vec![1.0f32], vec![2.0f32]])
            .is_ok());
        assert!(agg.worker_done[0]);
        // Wrong number of parameter buffers.
        assert!(agg.submit_worker(1, vec![vec![1.0f32]]).is_err());
    }

    #[test]
    fn test_mark_worker_done() {
        let cfg = DataParallelConfig::multi_gpu(2);
        let mut agg = GradientAggregator::new(cfg, 2);

        assert!(agg.mark_worker_done(0).is_ok());
        assert_eq!(agg.completed_workers, 1);

        // Double-mark should error
        assert!(agg.mark_worker_done(0).is_err());

        // Out-of-range
        assert!(agg.mark_worker_done(99).is_err());
    }

    /// Regression: `config` is public, so `num_workers` can be raised after the
    /// buffers were sized. Every entry point must report that instead of
    /// indexing past `worker_grads` / `worker_done` and panicking.
    #[test]
    fn test_mutated_worker_count_errors_instead_of_panicking() {
        let cfg = DataParallelConfig::multi_gpu(2);
        let mut agg = GradientAggregator::new(cfg, 1);
        agg.submit_gradients(0, 0, vec![1.0f32]).expect("submit");
        agg.submit_gradients(1, 0, vec![3.0f32]).expect("submit");
        agg.mark_worker_done(0).expect("done");
        agg.mark_worker_done(1).expect("done");

        agg.config.num_workers = 8;

        assert!(agg.submit_gradients(4, 0, vec![1.0f32]).is_err());
        assert!(agg.mark_worker_done(4).is_err());
        assert!(agg.aggregate().is_err());
        // Restoring the count restores normal operation.
        agg.config.num_workers = 2;
        assert!(agg.aggregate().is_ok());
    }

    #[test]
    fn test_is_ready_false_before_all_done() {
        let cfg = DataParallelConfig::multi_gpu(3);
        let mut agg = GradientAggregator::new(cfg, 1);

        assert!(!agg.is_ready());
        agg.mark_worker_done(0).expect("mark done");
        assert!(!agg.is_ready());
        agg.mark_worker_done(1).expect("mark done");
        assert!(!agg.is_ready());
    }

    #[test]
    fn test_is_ready_true_when_all_done() {
        let cfg = DataParallelConfig::multi_gpu(2);
        let mut agg = GradientAggregator::new(cfg, 1);

        agg.mark_worker_done(0).expect("done");
        agg.mark_worker_done(1).expect("done");
        assert!(agg.is_ready());
    }

    #[test]
    fn test_aggregate_all_reduce_average() {
        let cfg = DataParallelConfig::multi_gpu(2);
        let mut agg = GradientAggregator::new(cfg, 1);

        agg.submit_gradients(0, 0, vec![1.0f32, 3.0f32])
            .expect("submit");
        agg.submit_gradients(1, 0, vec![3.0f32, 5.0f32])
            .expect("submit");
        agg.mark_worker_done(0).expect("done");
        agg.mark_worker_done(1).expect("done");

        let result = agg.aggregate().expect("aggregate");
        assert_eq!(result.len(), 1);
        assert_abs_diff_eq!(result[0][0], 2.0, epsilon = 1e-6); // (1+3)/2
        assert_abs_diff_eq!(result[0][1], 4.0, epsilon = 1e-6); // (3+5)/2
    }

    #[test]
    fn test_aggregate_sum() {
        let mut cfg = DataParallelConfig::multi_gpu(2);
        cfg.sync_mode = SyncMode::AllReduceSum;
        let mut agg = GradientAggregator::new(cfg, 1);

        agg.submit_gradients(0, 0, vec![1.0f32, 2.0f32])
            .expect("submit");
        agg.submit_gradients(1, 0, vec![3.0f32, 4.0f32])
            .expect("submit");
        agg.mark_worker_done(0).expect("done");
        agg.mark_worker_done(1).expect("done");

        let result = agg.aggregate().expect("aggregate");
        assert_abs_diff_eq!(result[0][0], 4.0, epsilon = 1e-6); // 1+3
        assert_abs_diff_eq!(result[0][1], 6.0, epsilon = 1e-6); // 2+4
    }

    #[test]
    fn test_aggregate_no_sync_single_worker() {
        let cfg = DataParallelConfig::single_device();
        let mut agg = GradientAggregator::new(cfg, 2);

        agg.submit_gradients(0, 0, vec![7.0f32, 8.0f32])
            .expect("submit");
        agg.submit_gradients(0, 1, vec![9.0f32]).expect("submit");
        agg.mark_worker_done(0).expect("done");

        let result = agg.aggregate().expect("aggregate");
        assert_abs_diff_eq!(result[0][0], 7.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result[0][1], 8.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result[1][0], 9.0, epsilon = 1e-6);
    }

    #[test]
    fn test_aggregate_missing_worker_grads_count_as_zero() {
        let cfg = DataParallelConfig::multi_gpu(2);
        let mut agg = GradientAggregator::new(cfg, 1);

        agg.submit_gradients(0, 0, vec![2.0f32, 4.0f32])
            .expect("submit");
        agg.mark_worker_done(0).expect("done");
        agg.mark_worker_done(1).expect("done");

        let result = agg.aggregate().expect("aggregate");
        assert_abs_diff_eq!(result[0][0], 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result[0][1], 2.0, epsilon = 1e-6);
    }

    /// Regression: a shorter non-empty buffer used to be zipped against the
    /// accumulator, silently dropping the tail of the reduction.
    #[test]
    fn test_aggregate_ragged_lengths_error() {
        let cfg = DataParallelConfig::multi_gpu(2);
        let mut agg = GradientAggregator::new(cfg, 1);

        agg.submit_gradients(0, 0, vec![1.0f32, 2.0f32, 3.0f32])
            .expect("submit");
        agg.submit_gradients(1, 0, vec![1.0f32, 2.0f32])
            .expect("submit");
        agg.mark_worker_done(0).expect("done");
        agg.mark_worker_done(1).expect("done");

        match agg.aggregate() {
            Err(TrainerError::GradientSizeMismatch { expected, actual }) => {
                assert_eq!(expected, 3);
                assert_eq!(actual, 2);
            }
            other => panic!("expected GradientSizeMismatch, got {other:?}"),
        }
    }

    #[test]
    fn test_aggregate_not_ready_error() {
        let cfg = DataParallelConfig::multi_gpu(2);
        let agg = GradientAggregator::new(cfg, 1);
        assert!(agg.aggregate().is_err());
    }

    /// The bucketed parallel reduction must be bit-identical to the serial one:
    /// buckets hold whole parameters, so no accumulation is ever reordered.
    #[test]
    fn test_parallel_reduction_is_bit_exact_with_serial() {
        const WORKERS: usize = 4;
        const PARAMS: usize = 40;
        const ELEMENTS: usize = 2048; // 81_920 elements total → parallel path

        let build = |bucket_size_mb: f32| {
            let mut cfg = DataParallelConfig::multi_gpu(WORKERS);
            cfg.bucket_size_mb = bucket_size_mb;
            let mut agg = GradientAggregator::new(cfg, PARAMS);
            for w in 0..WORKERS {
                let grads: Vec<Vec<f32>> = (0..PARAMS)
                    .map(|p| lcg_values((w * PARAMS + p) as u64 + 1, ELEMENTS))
                    .collect();
                agg.submit_worker(w, grads).expect("submit");
            }
            agg
        };

        // ~8 KiB buckets → one parameter per bucket → threaded reduction.
        let parallel_agg = build(8192.0 / (1024.0 * 1024.0));
        let parallel_plan = parallel_agg.config.bucket_plan(&parallel_agg.param_lens());
        assert!(parallel_plan.len() > 1);
        let parallel = parallel_agg.aggregate().expect("aggregate");

        // 64 MiB buckets → a single bucket → serial reduction.
        let serial_agg = build(64.0);
        let serial_plan = serial_agg.config.bucket_plan(&serial_agg.param_lens());
        assert_eq!(serial_plan.len(), 1);
        let serial = serial_agg.aggregate().expect("aggregate");

        assert_eq!(parallel.len(), PARAMS);
        assert!(parallel == serial, "bucketing changed the reduction result");
    }

    #[test]
    fn test_reset_clears_state() {
        let cfg = DataParallelConfig::multi_gpu(2);
        let mut agg = GradientAggregator::new(cfg, 1);

        agg.submit_gradients(0, 0, vec![1.0f32]).expect("submit");
        agg.submit_gradients(1, 0, vec![2.0f32]).expect("submit");
        agg.mark_worker_done(0).expect("done");
        agg.mark_worker_done(1).expect("done");
        assert!(agg.is_ready());
        assert_eq!(agg.generation(), 0);

        agg.reset();
        assert!(!agg.is_ready());
        assert_eq!(agg.generation(), 1);
        // Grads should be cleared
        for wg in &agg.worker_grads {
            for pg in wg {
                assert!(pg.is_empty());
            }
        }
    }

    #[test]
    fn test_sync_report() {
        let cfg = DataParallelConfig::multi_gpu(2);
        let mut agg = GradientAggregator::new(cfg, 2);

        agg.submit_gradients(0, 0, vec![1.0f32; 100])
            .expect("submit");
        agg.submit_gradients(1, 0, vec![2.0f32; 100])
            .expect("submit");

        let report = agg.sync_report();
        assert_eq!(report.num_workers, 2);
        assert_eq!(report.sync_mode, SyncMode::AllReduce);
        assert_eq!(report.total_params, 2);
        assert_eq!(report.total_gradient_floats, 200); // 100 + 100
        assert_eq!(report.buckets, 1);
        assert!(!report.gradient_compression);

        // Ring all-reduce of a 400-byte payload over 2 workers: 2·(1/2)·400.
        let expected_mb = 400.0 / (1024.0 * 1024.0);
        assert_abs_diff_eq!(report.estimated_bandwidth_mb, expected_mb, epsilon = 1e-9);
        assert_eq!(report.generation, 0);

        let formatted = report.format();
        assert!(formatted.contains("workers=2"));
        assert!(formatted.contains("generation=0"));
        assert!(formatted.contains("buckets=1"));
    }

    #[test]
    fn test_sync_report_no_sync_transfers_nothing() {
        let cfg = DataParallelConfig::single_device();
        let mut agg = GradientAggregator::new(cfg, 1);
        agg.submit_gradients(0, 0, vec![1.0f32; 64])
            .expect("submit");

        let report = agg.sync_report();
        assert_abs_diff_eq!(report.estimated_bandwidth_mb, 0.0, epsilon = 1e-12);
        assert_eq!(ring_all_reduce_bytes(4096, 1), 0);
        assert_eq!(ring_all_reduce_bytes(4096, 4), 2 * 3 * 1024);
    }

    // --- Parallel step driver ----------------------------------------------

    #[test]
    fn test_run_parallel_step_runs_workers_concurrently() {
        let config = DataParallelConfig::multi_gpu(4);
        let mut states: Vec<Option<thread::ThreadId>> = vec![None; config.num_workers];

        let live = AtomicUsize::new(0);
        let peak = AtomicUsize::new(0);

        let outcome = run_parallel_step(&config, states.as_mut_slice(), |idx, slot| {
            *slot = Some(thread::current().id());
            let now = live.fetch_add(1, Ordering::SeqCst) + 1;
            peak.fetch_max(now, Ordering::SeqCst);
            // Long enough that overlap is certain even on a loaded machine,
            // short enough to stay negligible in the suite's runtime.
            thread::sleep(Duration::from_millis(50));
            live.fetch_sub(1, Ordering::SeqCst);
            Ok(vec![vec![idx as f32; 4], vec![1.0f32; 2]])
        })
        .expect("parallel step");

        // Workers really overlapped in time, and each ran on its own thread.
        assert!(
            peak.load(Ordering::SeqCst) >= 2,
            "workers did not run concurrently"
        );
        // `ThreadId`s may be reused once a thread has exited, so only assert
        // that the work left the calling thread and spread over several.
        let ids: HashSet<thread::ThreadId> = states.iter().filter_map(|s| *s).collect();
        assert!(ids.len() >= 2, "workers shared a single thread");
        assert!(!ids.contains(&thread::current().id()));

        // (0 + 1 + 2 + 3) / 4 = 1.5
        assert_eq!(outcome.gradients.len(), 2);
        for value in &outcome.gradients[0] {
            assert_abs_diff_eq!(*value, 1.5, epsilon = 1e-6);
        }
        for value in &outcome.gradients[1] {
            assert_abs_diff_eq!(*value, 1.0, epsilon = 1e-6);
        }
        assert_eq!(outcome.report.num_workers, 4);
        assert_eq!(outcome.report.total_params, 2);
    }

    #[test]
    fn test_run_parallel_step_propagates_worker_error() {
        let config = DataParallelConfig::multi_gpu(3);
        let mut states = vec![0u8; 3];

        let result = run_parallel_step(&config, states.as_mut_slice(), |idx, _state| {
            if idx == 2 {
                Err(TrainerError::Training("worker 2 failed".to_string()))
            } else {
                Ok(vec![vec![1.0f32; 2]])
            }
        });

        match result {
            Err(TrainerError::Training(msg)) => assert!(msg.contains("worker 2 failed")),
            other => panic!("expected worker error, got {other:?}"),
        }
    }

    #[test]
    fn test_run_parallel_step_rejects_state_count_mismatch() {
        let config = DataParallelConfig::multi_gpu(4);
        let mut states = vec![0u8; 2];
        let result = run_parallel_step(&config, states.as_mut_slice(), |_idx, _state| {
            Ok(vec![vec![0.0f32]])
        });
        assert!(matches!(result, Err(TrainerError::InvalidConfig(_))));
    }

    #[test]
    fn test_run_parallel_step_rejects_disagreeing_shapes() {
        let config = DataParallelConfig::multi_gpu(2);
        let mut states = vec![0u8; 2];
        let result = run_parallel_step(&config, states.as_mut_slice(), |idx, _state| {
            Ok(vec![vec![1.0f32; if idx == 0 { 4 } else { 3 }]])
        });
        assert!(matches!(
            result,
            Err(TrainerError::GradientSizeMismatch { .. })
        ));
    }

    #[test]
    fn test_run_parallel_step_single_device_returns_worker_zero() {
        let config = DataParallelConfig::single_device();
        let mut states = vec![5.0f32];
        let outcome = run_parallel_step(&config, states.as_mut_slice(), |_idx, state| {
            Ok(vec![vec![*state; 3]])
        })
        .expect("step");
        assert_eq!(outcome.gradients[0], vec![5.0f32; 3]);
        assert_abs_diff_eq!(outcome.report.estimated_bandwidth_mb, 0.0, epsilon = 1e-12);
    }
}
