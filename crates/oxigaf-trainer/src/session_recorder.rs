//! Training session recorder for reproducibility and experiment tracking.
//!
//! Records hyperparameters, hardware information, per-step metrics snapshots,
//! and produces JSON output for offline analysis and experiment comparison.
//!
//! # Key Types
//! - [`HardwareInfo`]           — CPU/OS/Rust environment fingerprint.
//! - [`SessionConfigSnapshot`]  — Immutable copy of training hyperparameters.
//! - [`MetricsSnapshot`]        — Per-step loss, PSNR, SSIM, Gaussian count.
//! - [`SessionRecord`]          — Complete session record (config + hardware + snapshots).
//! - [`SessionRecorder`]        — Stateful recorder with rolling-window support.
//!
//! # JSON serialisation
//! All serialisation is manual (no serde dependency in this module).

use std::fmt;
use std::fmt::Write as FmtWrite;
use std::path::Path;

// ---------------------------------------------------------------------------
// SessionError
// ---------------------------------------------------------------------------

/// Errors produced by [`SessionRecorder`] and related types.
#[derive(Debug)]
pub enum SessionError {
    /// Wraps an underlying I/O failure.
    IoError(std::io::Error),
    /// Indicates a problem parsing an existing JSON file.
    ParseError(String),
}

impl fmt::Display for SessionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SessionError::IoError(e) => write!(f, "I/O error: {}", e),
            SessionError::ParseError(s) => write!(f, "Parse error: {}", s),
        }
    }
}

impl std::error::Error for SessionError {}

impl From<std::io::Error> for SessionError {
    fn from(e: std::io::Error) -> Self {
        SessionError::IoError(e)
    }
}

// ---------------------------------------------------------------------------
// xorshift64 PRNG utilities (for session ID generation)
// ---------------------------------------------------------------------------

/// xorshift64 pseudo-random number generator step.
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// Generate a 16-character hex session ID from the given seed.
///
/// Uses two xorshift64 steps, taking the low 32 bits of each.
/// Same seed always produces the same ID (deterministic).
pub fn generate_session_id(seed: u64) -> String {
    let mut state = if seed == 0 { 0xdeadbeef_cafebabe } else { seed };
    let a = xorshift64(&mut state);
    let b = xorshift64(&mut state);
    format!("{:08x}{:08x}", a as u32, b as u32)
}

// ---------------------------------------------------------------------------
// JSON helpers (manual, no serde)
// ---------------------------------------------------------------------------

/// Escape a string for inclusion as a JSON string value.
///
/// Handles the characters required by the JSON spec:
/// `"` → `\"`, `\` → `\\`, newline → `\n`, carriage-return → `\r`, tab → `\t`.
fn escape_json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 4);
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out
}

/// Format a JSON string field: `"key": "escaped_value"`.
fn json_str_field(key: &str, value: &str) -> String {
    format!("\"{}\":\"{}\"", key, escape_json_string(value))
}

/// Format a JSON u64 field: `"key": 12345`.
fn json_u64_field(key: &str, value: u64) -> String {
    format!("\"{}\":{}", key, value)
}

/// Format a JSON usize field: `"key": 42`.
fn json_usize_field(key: &str, value: usize) -> String {
    format!("\"{}\":{}", key, value)
}

/// Format a JSON u32 field: `"key": 3`.
fn json_u32_field(key: &str, value: u32) -> String {
    format!("\"{}\":{}", key, value)
}

/// Format a JSON f64 field: `"key":value`.
///
/// Non-finite values (`NaN`/`±Inf` — which routinely occur in a diverging
/// training run, exactly the run whose record is most worth inspecting)
/// serialize to `null`, since Rust's `NaN`/`inf` literals are not valid
/// JSON tokens and would otherwise produce a file no conforming JSON
/// reader can parse. Finite values use `{:?}` (Debug) rather than a fixed
/// 6-decimal format so small magnitudes (e.g. a learning rate of `1e-9`)
/// survive instead of rounding to `0.000000`.
fn json_f64_field(key: &str, value: f64) -> String {
    if value.is_finite() {
        format!("\"{key}\":{value:?}")
    } else {
        format!("\"{key}\":null")
    }
}

/// Format a JSON f32 field: `"key":value`. See [`json_f64_field`].
fn json_f32_field(key: &str, value: f32) -> String {
    if value.is_finite() {
        format!("\"{key}\":{value:?}")
    } else {
        format!("\"{key}\":null")
    }
}

// ---------------------------------------------------------------------------
// HardwareInfo
// ---------------------------------------------------------------------------

/// Hardware and runtime environment fingerprint for a training session.
#[derive(Debug, Clone)]
pub struct HardwareInfo {
    /// Number of logical CPU cores available to the process.
    pub cpu_cores: usize,
    /// Hostname of the machine, sourced from `$HOSTNAME` or `$COMPUTERNAME`.
    pub hostname: String,
    /// Rust compiler version from `CARGO_PKG_RUST_VERSION`, or `"unknown"`.
    pub rust_version: String,
    /// Operating system name (`"macos"`, `"linux"`, `"windows"`, …).
    pub os_name: String,
    /// CPU architecture (`"x86_64"`, `"aarch64"`, …).
    pub arch: String,
}

impl HardwareInfo {
    /// Detect hardware/runtime information from the current environment.
    pub fn detect() -> Self {
        let cpu_cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);

        let hostname = std::env::var("HOSTNAME")
            .or_else(|_| std::env::var("COMPUTERNAME"))
            .unwrap_or_else(|_| "unknown".to_string());

        let rust_version = option_env!("CARGO_PKG_RUST_VERSION")
            .unwrap_or("unknown")
            .to_string();

        let os_name = std::env::consts::OS.to_string();
        let arch = std::env::consts::ARCH.to_string();

        HardwareInfo {
            cpu_cores,
            hostname,
            rust_version,
            os_name,
            arch,
        }
    }

    /// Serialize to a JSON object string (no surrounding braces from caller needed).
    fn to_json_object(&self) -> String {
        let mut out = String::from("{");
        out.push_str(&json_usize_field("cpu_cores", self.cpu_cores));
        out.push(',');
        out.push_str(&json_str_field("hostname", &self.hostname));
        out.push(',');
        out.push_str(&json_str_field("rust_version", &self.rust_version));
        out.push(',');
        out.push_str(&json_str_field("os_name", &self.os_name));
        out.push(',');
        out.push_str(&json_str_field("arch", &self.arch));
        out.push('}');
        out
    }
}

impl Default for HardwareInfo {
    fn default() -> Self {
        HardwareInfo {
            cpu_cores: 1,
            hostname: "unknown".to_string(),
            rust_version: "unknown".to_string(),
            os_name: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// SessionConfigSnapshot
// ---------------------------------------------------------------------------

/// Immutable snapshot of training hyperparameters captured at session start.
///
/// All numeric fields use the same units and semantics as the live training
/// configuration.  The `extra` field carries arbitrary key-value metadata.
#[derive(Debug, Clone)]
pub struct SessionConfigSnapshot {
    /// Number of Gaussians in the initial point cloud.
    pub num_gaussians_init: usize,
    /// Spherical harmonics degree (0–3).
    pub sh_degree: u32,
    /// Adam learning rate for Gaussian positions.
    pub learning_rate_position: f64,
    /// Adam learning rate for opacity parameters.
    pub learning_rate_opacity: f64,
    /// Adam learning rate for scale parameters.
    pub learning_rate_scale: f64,
    /// Adam learning rate for rotation (quaternion) parameters.
    pub learning_rate_rotation: f64,
    /// Adam learning rate for spherical-harmonics coefficients.
    pub learning_rate_sh: f64,
    /// Total number of optimisation iterations.
    pub num_iterations: usize,
    /// Number of images processed per optimisation step.
    pub batch_size: usize,
    /// Rendered image width in pixels.
    pub image_width: usize,
    /// Rendered image height in pixels.
    pub image_height: usize,
    /// Number of camera views sampled per step.
    pub num_views_per_step: usize,
    /// First iteration at which adaptive density control is applied.
    pub density_start_step: usize,
    /// Number of steps between consecutive density-control operations.
    pub density_interval: usize,
    /// Number of steps between opacity resets.
    pub opacity_reset_interval: usize,
    /// Number of steps between checkpoint saves.
    pub checkpoint_interval: usize,
    /// Arbitrary extra metadata as key-value string pairs.
    pub extra: Vec<(String, String)>,
}

impl Default for SessionConfigSnapshot {
    fn default() -> Self {
        SessionConfigSnapshot {
            num_gaussians_init: 100_000,
            sh_degree: 3,
            learning_rate_position: 1.6e-4,
            learning_rate_opacity: 5e-2,
            learning_rate_scale: 5e-3,
            learning_rate_rotation: 1e-3,
            learning_rate_sh: 1e-3,
            num_iterations: 30_000,
            batch_size: 1,
            image_width: 800,
            image_height: 800,
            num_views_per_step: 1,
            density_start_step: 500,
            density_interval: 100,
            opacity_reset_interval: 3_000,
            checkpoint_interval: 1_000,
            extra: Vec::new(),
        }
    }
}

impl SessionConfigSnapshot {
    /// Attach an arbitrary key-value pair to the config snapshot.
    ///
    /// This is the builder-style API for adding metadata without subclassing.
    pub fn with_extra(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.extra.push((key.into(), value.into()));
        self
    }

    /// Serialize to a JSON object string.
    fn to_json_object(&self) -> String {
        let mut out = String::from("{");
        out.push_str(&json_usize_field(
            "num_gaussians_init",
            self.num_gaussians_init,
        ));
        out.push(',');
        out.push_str(&json_u32_field("sh_degree", self.sh_degree));
        out.push(',');
        out.push_str(&json_f64_field(
            "learning_rate_position",
            self.learning_rate_position,
        ));
        out.push(',');
        out.push_str(&json_f64_field(
            "learning_rate_opacity",
            self.learning_rate_opacity,
        ));
        out.push(',');
        out.push_str(&json_f64_field(
            "learning_rate_scale",
            self.learning_rate_scale,
        ));
        out.push(',');
        out.push_str(&json_f64_field(
            "learning_rate_rotation",
            self.learning_rate_rotation,
        ));
        out.push(',');
        out.push_str(&json_f64_field("learning_rate_sh", self.learning_rate_sh));
        out.push(',');
        out.push_str(&json_usize_field("num_iterations", self.num_iterations));
        out.push(',');
        out.push_str(&json_usize_field("batch_size", self.batch_size));
        out.push(',');
        out.push_str(&json_usize_field("image_width", self.image_width));
        out.push(',');
        out.push_str(&json_usize_field("image_height", self.image_height));
        out.push(',');
        out.push_str(&json_usize_field(
            "num_views_per_step",
            self.num_views_per_step,
        ));
        out.push(',');
        out.push_str(&json_usize_field(
            "density_start_step",
            self.density_start_step,
        ));
        out.push(',');
        out.push_str(&json_usize_field("density_interval", self.density_interval));
        out.push(',');
        out.push_str(&json_usize_field(
            "opacity_reset_interval",
            self.opacity_reset_interval,
        ));
        out.push(',');
        out.push_str(&json_usize_field(
            "checkpoint_interval",
            self.checkpoint_interval,
        ));

        // extra key-value pairs as a nested object
        out.push_str(",\"extra\":{");
        let mut first = true;
        for (k, v) in &self.extra {
            if !first {
                out.push(',');
            }
            out.push_str(&json_str_field(k, v));
            first = false;
        }
        out.push('}');
        out.push('}');
        out
    }
}

// ---------------------------------------------------------------------------
// MetricsSnapshot
// ---------------------------------------------------------------------------

/// Per-step metric snapshot recorded during training.
#[derive(Debug, Clone)]
pub struct MetricsSnapshot {
    /// Optimisation step index.
    pub step: usize,
    /// Weighted sum of all loss components.
    pub total_loss: f32,
    /// L1 / L2 photometric loss component.
    pub photometric_loss: f32,
    /// Perceptual (LPIPS) loss component.
    pub perceptual_loss: f32,
    /// Regularisation loss component.
    pub regularization_loss: f32,
    /// PSNR in dB.
    pub psnr: f32,
    /// SSIM in [0, 1].
    pub ssim: f32,
    /// Current number of Gaussians in the scene.
    pub num_gaussians: usize,
    /// Wall-clock time since UNIX epoch in milliseconds.
    pub timestamp_ms: u64,
    /// Current effective learning rate for the position parameter group.
    pub learning_rate: f64,
    /// L2 gradient norm across all parameters at this step.
    pub grad_norm: f32,
}

impl MetricsSnapshot {
    /// Return the number of milliseconds since the UNIX epoch.
    ///
    /// Returns `0` if the system clock is unavailable (unlikely in practice).
    pub fn now_ms() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0)
    }

    /// Serialize to a compact JSON object string.
    fn to_json_object(&self) -> String {
        let mut out = String::from("{");
        out.push_str(&json_usize_field("step", self.step));
        out.push(',');
        out.push_str(&json_f32_field("total_loss", self.total_loss));
        out.push(',');
        out.push_str(&json_f32_field("photometric_loss", self.photometric_loss));
        out.push(',');
        out.push_str(&json_f32_field("perceptual_loss", self.perceptual_loss));
        out.push(',');
        out.push_str(&json_f32_field(
            "regularization_loss",
            self.regularization_loss,
        ));
        out.push(',');
        out.push_str(&json_f32_field("psnr", self.psnr));
        out.push(',');
        out.push_str(&json_f32_field("ssim", self.ssim));
        out.push(',');
        out.push_str(&json_usize_field("num_gaussians", self.num_gaussians));
        out.push(',');
        out.push_str(&json_u64_field("timestamp_ms", self.timestamp_ms));
        out.push(',');
        out.push_str(&json_f64_field("learning_rate", self.learning_rate));
        out.push(',');
        out.push_str(&json_f32_field("grad_norm", self.grad_norm));
        out.push('}');
        out
    }
}

// ---------------------------------------------------------------------------
// SessionRecord
// ---------------------------------------------------------------------------

/// Complete record of a training session.
///
/// Created via [`SessionRecorder::new`] and populated incrementally by
/// [`SessionRecorder::record_step`].  Call [`SessionRecord::finish`] (or
/// [`SessionRecorder::finish`]) to stamp the end timestamp.
#[derive(Debug, Clone)]
pub struct SessionRecord {
    /// Unique 16-character hex session identifier.
    pub session_id: String,
    /// Wall-clock time when the session started (ms since UNIX epoch).
    pub start_timestamp_ms: u64,
    /// Wall-clock time when the session ended (ms since UNIX epoch); `0` until [`finish`](Self::finish) is called.
    pub end_timestamp_ms: u64,
    /// Hyperparameter snapshot captured at session start.
    pub config: SessionConfigSnapshot,
    /// Hardware / runtime environment snapshot.
    pub hardware: HardwareInfo,
    /// Ordered sequence of per-step metric snapshots.
    pub snapshots: Vec<MetricsSnapshot>,
    /// Freeform notes attached to this session.
    pub notes: String,
    /// Git commit hash at the time of training (from `$GIT_COMMIT`).
    pub git_hash: String,
}

impl SessionRecord {
    /// Create a new session record, seeding the session ID from the current time.
    pub fn new(config: SessionConfigSnapshot) -> Self {
        let start_timestamp_ms = MetricsSnapshot::now_ms();
        let session_id = generate_session_id(start_timestamp_ms);
        let git_hash = std::env::var("GIT_COMMIT").unwrap_or_default();

        SessionRecord {
            session_id,
            start_timestamp_ms,
            end_timestamp_ms: 0,
            config,
            hardware: HardwareInfo::detect(),
            snapshots: Vec::new(),
            notes: String::new(),
            git_hash,
        }
    }

    /// Stamp the end timestamp.
    ///
    /// Idempotent: calling `finish` more than once overwrites the previous
    /// end timestamp with the current wall-clock time.
    pub fn finish(&mut self) {
        self.end_timestamp_ms = MetricsSnapshot::now_ms();
    }

    /// Session duration in seconds.
    ///
    /// Returns `0.0` if [`finish`](Self::finish) has not been called yet.
    /// Uses a saturating subtraction: both timestamps come from
    /// `SystemTime::now()`, which is not monotonic — an NTP step or manual
    /// clock adjustment during a long training run could otherwise make
    /// `end_timestamp_ms < start_timestamp_ms`, underflowing the `u64`
    /// subtraction (a panic in debug builds, a ~584-million-year wraparound
    /// in release).
    pub fn duration_secs(&self) -> f64 {
        if self.end_timestamp_ms == 0 {
            return 0.0;
        }
        self.end_timestamp_ms
            .saturating_sub(self.start_timestamp_ms) as f64
            / 1000.0
    }
}

// ---------------------------------------------------------------------------
// SessionRecorder
// ---------------------------------------------------------------------------

/// Stateful recorder that builds a [`SessionRecord`] incrementally.
///
/// # Rolling snapshot window
/// Set `max_snapshots` to a positive value to keep only the most recent N
/// snapshots, dropping the oldest when the limit is reached.  `0` means
/// unlimited (keep all snapshots).
///
/// # Snapshot interval
/// Set `snapshot_interval` to N to record only every Nth step (steps where
/// `step % N != 0` are silently skipped).  `1` records every step.
pub struct SessionRecorder {
    record: SessionRecord,
    /// Maximum number of snapshots retained.  `0` = unlimited.
    max_snapshots: usize,
    /// Only record when `step % snapshot_interval == 0`.
    snapshot_interval: usize,
}

impl SessionRecorder {
    /// Create a new recorder with default settings (unlimited snapshots, every step).
    pub fn new(config: SessionConfigSnapshot) -> Self {
        SessionRecorder {
            record: SessionRecord::new(config),
            max_snapshots: 0,
            snapshot_interval: 1,
        }
    }

    /// Set the maximum number of snapshots to retain (rolling window).
    ///
    /// `0` means unlimited.  Existing snapshots exceeding the new limit are
    /// **not** retroactively pruned; the limit takes effect on the next
    /// [`record_step`](Self::record_step) call.
    pub fn with_max_snapshots(mut self, n: usize) -> Self {
        self.max_snapshots = n;
        self
    }

    /// Set the step interval at which snapshots are recorded.
    ///
    /// Steps where `step % interval != 0` are silently skipped.
    /// `1` records every step (the default).  `0` is treated as `1`.
    pub fn with_snapshot_interval(mut self, interval: usize) -> Self {
        self.snapshot_interval = if interval == 0 { 1 } else { interval };
        self
    }

    /// Record a metrics snapshot for the given step.
    ///
    /// The snapshot is silently dropped when `snapshot.step % snapshot_interval != 0`.
    /// When `max_snapshots > 0` and the buffer is full, the oldest snapshot is removed
    /// before appending the new one.
    pub fn record_step(&mut self, snapshot: MetricsSnapshot) {
        let interval = if self.snapshot_interval == 0 {
            1
        } else {
            self.snapshot_interval
        };

        if !snapshot.step.is_multiple_of(interval) {
            return;
        }

        // Enforce rolling window: drop the oldest entry when at capacity.
        if self.max_snapshots > 0 && self.record.snapshots.len() >= self.max_snapshots {
            self.record.snapshots.remove(0);
        }

        self.record.snapshots.push(snapshot);
    }

    /// Stamp the end timestamp on the underlying record.
    pub fn finish(&mut self) {
        self.record.finish();
    }

    /// Borrow the underlying session record.
    pub fn record(&self) -> &SessionRecord {
        &self.record
    }

    /// Return the snapshot with the highest PSNR, or `None` if no snapshots exist.
    pub fn best_snapshot(&self) -> Option<&MetricsSnapshot> {
        self.record.snapshots.iter().max_by(|a, b| {
            a.psnr
                .partial_cmp(&b.psnr)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Return the snapshot with the lowest total loss, or `None` if no snapshots exist.
    pub fn best_loss_snapshot(&self) -> Option<&MetricsSnapshot> {
        self.record.snapshots.iter().min_by(|a, b| {
            a.total_loss
                .partial_cmp(&b.total_loss)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Format a human-readable summary table of the session.
    pub fn format_summary(&self) -> String {
        let r = &self.record;
        let mut out = String::new();

        let _ = writeln!(out, "╔═══════════════════════════════════════════════════╗");
        let _ = writeln!(
            out,
            "║          OxiGAF Training Session Summary           ║"
        );
        let _ = writeln!(out, "╠═══════════════════════════════════════════════════╣");
        let _ = writeln!(out, "║  Session ID  : {:<34}║", r.session_id);
        let _ = writeln!(
            out,
            "║  Git Hash    : {:<34}║",
            if r.git_hash.is_empty() {
                "(none)".to_string()
            } else {
                r.git_hash.clone()
            }
        );
        let _ = writeln!(out, "║  Start (ms)  : {:<34}║", r.start_timestamp_ms);
        let _ = writeln!(
            out,
            "║  Duration    : {:<34}║",
            format!("{:.3}s", r.duration_secs())
        );
        let _ = writeln!(out, "╠═══════════════════════════════════════════════════╣");
        let _ = writeln!(
            out,
            "║  Hardware                                          ║"
        );
        let _ = writeln!(out, "║    OS        : {:<34}║", r.hardware.os_name);
        let _ = writeln!(out, "║    Arch      : {:<34}║", r.hardware.arch);
        let _ = writeln!(out, "║    CPU cores : {:<34}║", r.hardware.cpu_cores);
        let _ = writeln!(
            out,
            "║    Host      : {:<34}║",
            // Truncate by character count, not byte index: `hostname`
            // comes from the `$HOSTNAME`/`$COMPUTERNAME` environment
            // variable and can contain multi-byte UTF-8, and slicing by
            // byte index panics if the cut point isn't a char boundary.
            // Character-based truncation also fixes the column alignment,
            // since `{:<34}` pads by character count while `.len()`
            // measures bytes.
            if r.hardware.hostname.chars().count() > 34 {
                r.hardware.hostname.chars().take(34).collect::<String>()
            } else {
                r.hardware.hostname.clone()
            }
        );
        let _ = writeln!(out, "╠═══════════════════════════════════════════════════╣");
        let _ = writeln!(
            out,
            "║  Config                                            ║"
        );
        let _ = writeln!(out, "║    Gaussians : {:<34}║", r.config.num_gaussians_init);
        let _ = writeln!(out, "║    SH degree : {:<34}║", r.config.sh_degree);
        let _ = writeln!(out, "║    Iters     : {:<34}║", r.config.num_iterations);
        let _ = writeln!(
            out,
            "║    LR pos    : {:<34}║",
            format!("{:.2e}", r.config.learning_rate_position)
        );
        let _ = writeln!(out, "╠═══════════════════════════════════════════════════╣");
        let _ = writeln!(
            out,
            "║  Metrics                                           ║"
        );
        let _ = writeln!(out, "║    Snapshots : {:<34}║", r.snapshots.len());

        if let Some(best) = self.best_snapshot() {
            let _ = writeln!(
                out,
                "║    Best PSNR : {:<34}║",
                format!("{:.2} dB (step {})", best.psnr, best.step)
            );
        } else {
            let _ = writeln!(out, "║    Best PSNR : {:<34}║", "(none)");
        }

        if let Some(best) = self.best_loss_snapshot() {
            let _ = writeln!(
                out,
                "║    Best loss : {:<34}║",
                format!("{:.5} (step {})", best.total_loss, best.step)
            );
        } else {
            let _ = writeln!(out, "║    Best loss : {:<34}║", "(none)");
        }

        if !r.notes.is_empty() {
            let _ = writeln!(out, "╠═══════════════════════════════════════════════════╣");
            let _ = writeln!(
                out,
                "║  Notes       : {:<34}║",
                // Character-based truncation (see the hostname comment
                // above) — `r.notes` is a public field, so it may contain
                // multi-byte UTF-8 (e.g. a 12-character Japanese note is
                // 36 bytes, which would panic slicing at byte 33).
                if r.notes.chars().count() > 34 {
                    format!("{}…", r.notes.chars().take(33).collect::<String>())
                } else {
                    r.notes.clone()
                }
            );
        }

        let _ = writeln!(out, "╚═══════════════════════════════════════════════════╝");
        out
    }

    /// Serialize the complete session to a JSON string.
    ///
    /// The format follows the schema described in the module documentation.
    /// All field values are manually escaped — no serde dependency is required.
    pub fn to_json(&self) -> String {
        let r = &self.record;
        let mut out = String::from("{\n");

        // Top-level scalar fields
        out.push_str(&format!(
            "  {},\n",
            json_str_field("session_id", &r.session_id)
        ));
        out.push_str(&format!(
            "  {},\n",
            json_u64_field("start_timestamp_ms", r.start_timestamp_ms)
        ));
        out.push_str(&format!(
            "  {},\n",
            json_u64_field("end_timestamp_ms", r.end_timestamp_ms)
        ));
        out.push_str(&format!(
            "  {},\n",
            json_f64_field("duration_secs", r.duration_secs())
        ));

        // Hardware object
        out.push_str(&format!(
            "  \"hardware\":{},\n",
            r.hardware.to_json_object()
        ));

        // Config object
        out.push_str(&format!("  \"config\":{},\n", r.config.to_json_object()));

        // Aggregate metrics (computed from snapshots)
        let snapshots_count = r.snapshots.len();
        out.push_str(&format!(
            "  {},\n",
            json_usize_field("snapshots_count", snapshots_count)
        ));

        let best_psnr = self.best_snapshot().map(|s| s.psnr).unwrap_or(0.0);
        out.push_str(&format!("  {},\n", json_f32_field("best_psnr", best_psnr)));

        let best_loss = self
            .best_loss_snapshot()
            .map(|s| s.total_loss)
            .unwrap_or(0.0);
        out.push_str(&format!("  {},\n", json_f32_field("best_loss", best_loss)));

        out.push_str(&format!("  {},\n", json_str_field("notes", &r.notes)));
        out.push_str(&format!("  {},\n", json_str_field("git_hash", &r.git_hash)));

        // Snapshots array
        out.push_str("  \"snapshots\":[\n");
        for (i, snap) in r.snapshots.iter().enumerate() {
            if i > 0 {
                out.push_str(",\n");
            }
            out.push_str("    ");
            out.push_str(&snap.to_json_object());
        }
        if !r.snapshots.is_empty() {
            out.push('\n');
        }
        out.push_str("  ]\n");
        out.push('}');
        out
    }

    /// Write the JSON representation to the file at `path`.
    ///
    /// Creates or overwrites the file.  Parent directories must exist.
    pub fn save_json(&self, path: &Path) -> Result<(), SessionError> {
        let json = self.to_json();
        std::fs::write(path, json.as_bytes())?;
        Ok(())
    }

    /// Load a session record header from a JSON file.
    ///
    /// This is a "header reader" that extracts:
    /// - `session_id`
    /// - `start_timestamp_ms`
    /// - `end_timestamp_ms`
    /// - `snapshots_count` (used to set `snapshots` to an empty vec; the full
    ///   array is not parsed, keeping this operation lightweight).
    ///
    /// The returned [`SessionRecord`] has `config` and `hardware` set to their
    /// `Default` values; `snapshots` is always empty.
    pub fn load_json(path: &Path) -> Result<SessionRecord, SessionError> {
        let raw = std::fs::read_to_string(path)?;

        let session_id = extract_json_str(&raw, "session_id")
            .ok_or_else(|| SessionError::ParseError("missing session_id".to_string()))?;

        let start_timestamp_ms = extract_json_u64(&raw, "start_timestamp_ms")
            .ok_or_else(|| SessionError::ParseError("missing start_timestamp_ms".to_string()))?;

        let end_timestamp_ms = extract_json_u64(&raw, "end_timestamp_ms").unwrap_or(0);

        let _snapshots_count = extract_json_usize(&raw, "snapshots_count").unwrap_or(0);

        let git_hash = extract_json_str(&raw, "git_hash").unwrap_or_default();
        let notes = extract_json_str(&raw, "notes").unwrap_or_default();

        Ok(SessionRecord {
            session_id,
            start_timestamp_ms,
            end_timestamp_ms,
            config: SessionConfigSnapshot::default(),
            hardware: HardwareInfo::default(),
            snapshots: Vec::new(),
            notes,
            git_hash,
        })
    }
}

// ---------------------------------------------------------------------------
// Minimal JSON value extractors (no serde)
// ---------------------------------------------------------------------------

/// Extract a string value for `"key":"value"` from raw JSON text.
///
/// Handles simple (non-nested) string fields.  Returns `None` if the key is
/// not found or the value cannot be parsed as a JSON string.
fn extract_json_str(json: &str, key: &str) -> Option<String> {
    let needle = format!("\"{}\":", key);
    let start = json.find(&needle)?;
    let after_colon = start + needle.len();

    // Skip whitespace
    let rest = json[after_colon..].trim_start();

    if !rest.starts_with('"') {
        return None;
    }

    let inside = &rest[1..];
    let mut result = String::new();
    let mut chars = inside.chars();
    loop {
        match chars.next()? {
            '"' => break,
            '\\' => match chars.next()? {
                '"' => result.push('"'),
                '\\' => result.push('\\'),
                'n' => result.push('\n'),
                'r' => result.push('\r'),
                't' => result.push('\t'),
                c => result.push(c),
            },
            c => result.push(c),
        }
    }
    Some(result)
}

/// Extract a `u64` value for `"key":12345` from raw JSON text.
fn extract_json_u64(json: &str, key: &str) -> Option<u64> {
    let needle = format!("\"{}\":", key);
    let start = json.find(&needle)?;
    let after_colon = start + needle.len();

    let rest = json[after_colon..].trim_start();
    let end = rest
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(rest.len());
    let digits = &rest[..end];
    digits.parse::<u64>().ok()
}

/// Extract a `usize` value for `"key":42` from raw JSON text.
fn extract_json_usize(json: &str, key: &str) -> Option<usize> {
    extract_json_u64(json, key).map(|v| v as usize)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_snapshot(step: usize, psnr: f32, total_loss: f32) -> MetricsSnapshot {
        MetricsSnapshot {
            step,
            total_loss,
            photometric_loss: total_loss * 0.8,
            perceptual_loss: total_loss * 0.1,
            regularization_loss: total_loss * 0.1,
            psnr,
            ssim: 0.9,
            num_gaussians: 100_000,
            timestamp_ms: MetricsSnapshot::now_ms(),
            learning_rate: 1.6e-4,
            grad_norm: 0.01,
        }
    }

    // ------------------------------------------------------------------
    // json_f64_field / json_f32_field
    // ------------------------------------------------------------------

    #[test]
    fn test_json_f32_field_nan_serializes_to_null_not_nan_literal() {
        // Regression: `format!("{:.6}", f32::NAN)` produces `NaN`, which is
        // not a valid JSON token and would make the output unparseable by
        // any conforming reader — exactly the diverging-run record you
        // most want to be able to load.
        let out = json_f32_field("grad_norm", f32::NAN);
        assert_eq!(out, "\"grad_norm\":null");
    }

    #[test]
    fn test_json_f32_field_infinity_serializes_to_null_not_inf_literal() {
        let out = json_f32_field("grad_norm", f32::INFINITY);
        assert_eq!(out, "\"grad_norm\":null");
        let out_neg = json_f32_field("grad_norm", f32::NEG_INFINITY);
        assert_eq!(out_neg, "\"grad_norm\":null");
    }

    #[test]
    fn test_json_f64_field_small_magnitude_survives_round_trip() {
        // Regression: the old `{:.6}` format rendered a learning rate of
        // 1e-9 as "0.000000", destroying the value. The serialized number
        // must now parse back to (approximately) the original value
        // instead of rounding to zero.
        let out = json_f64_field("learning_rate_position", 1e-9);
        let value_str = out
            .strip_prefix("\"learning_rate_position\":")
            .expect("expected key prefix");
        let parsed: f64 = value_str.parse().expect("must be a valid JSON number");
        assert!(
            (parsed - 1e-9).abs() < 1e-15,
            "expected ~1e-9, got {parsed} (raw: {value_str})"
        );
        assert_ne!(parsed, 0.0, "value must not have rounded to zero");
    }

    #[test]
    fn test_json_f64_field_finite_value_is_valid_json_number() {
        let out = json_f64_field("duration_secs", 12.5);
        let value_str = out.split(':').nth(1).expect("has a value part");
        assert!(
            value_str.parse::<f64>().is_ok(),
            "value {value_str} must parse as a JSON number"
        );
    }

    #[test]
    fn test_hardware_info_detect() {
        let hw = HardwareInfo::detect();
        assert!(hw.cpu_cores >= 1, "cpu_cores must be at least 1");
        assert!(!hw.os_name.is_empty(), "os_name should not be empty");
        assert!(!hw.arch.is_empty(), "arch should not be empty");
    }

    #[test]
    fn test_session_config_default() {
        let cfg = SessionConfigSnapshot::default();
        assert_eq!(cfg.sh_degree, 3);
        assert_eq!(cfg.num_gaussians_init, 100_000);
        assert!(cfg.learning_rate_position > 0.0);
        assert!(cfg.extra.is_empty());
    }

    #[test]
    fn test_session_config_with_extra() {
        let cfg = SessionConfigSnapshot::default()
            .with_extra("experiment", "baseline")
            .with_extra("user", "tester");
        assert_eq!(cfg.extra.len(), 2);
        assert_eq!(cfg.extra[0].0, "experiment");
        assert_eq!(cfg.extra[0].1, "baseline");
        assert_eq!(cfg.extra[1].0, "user");
        assert_eq!(cfg.extra[1].1, "tester");
    }

    #[test]
    fn test_metrics_snapshot_now_ms_nonzero() {
        let ms = MetricsSnapshot::now_ms();
        // Should be a large number (ms since epoch 1970) — definitely > 0
        assert!(
            ms > 1_000_000_000_000,
            "timestamp should be > Jan 2001 epoch ms"
        );
    }

    #[test]
    fn test_session_record_new() {
        let record = SessionRecord::new(SessionConfigSnapshot::default());
        assert_eq!(
            record.session_id.len(),
            16,
            "session_id must be 16 hex chars"
        );
        assert!(record.start_timestamp_ms > 0);
        assert_eq!(
            record.end_timestamp_ms, 0,
            "end_timestamp should be 0 before finish"
        );
        assert!(record.snapshots.is_empty());
    }

    #[test]
    fn test_session_record_finish_sets_end_timestamp() {
        let mut record = SessionRecord::new(SessionConfigSnapshot::default());
        assert_eq!(record.end_timestamp_ms, 0);
        record.finish();
        assert!(
            record.end_timestamp_ms > 0,
            "end_timestamp should be set after finish"
        );
        assert!(record.end_timestamp_ms >= record.start_timestamp_ms);
    }

    #[test]
    fn test_session_record_duration_secs() {
        let mut record = SessionRecord::new(SessionConfigSnapshot::default());
        // Before finish: duration should be 0
        assert!((record.duration_secs() - 0.0).abs() < 1e-9);
        record.finish();
        // After finish: duration should be non-negative
        assert!(record.duration_secs() >= 0.0);
    }

    #[test]
    fn test_session_record_duration_secs_clock_moved_backwards_no_panic() {
        // Regression: `duration_secs` used to subtract two plain `u64`
        // timestamps with no ordering guarantee. Both come from
        // `SystemTime::now()`, which is not monotonic, so an NTP step
        // during a long run could make `end < start` — this used to panic
        // in debug builds ("attempt to subtract with overflow") and wrap
        // to ~584 million years in release.
        let mut record = SessionRecord::new(SessionConfigSnapshot::default());
        record.start_timestamp_ms = 10_000;
        record.end_timestamp_ms = 5_000; // clock moved backwards
        let duration = record.duration_secs();
        assert_eq!(duration, 0.0, "saturating_sub should floor at 0");
    }

    #[test]
    fn test_recorder_record_step_basic() {
        let mut recorder = SessionRecorder::new(SessionConfigSnapshot::default());
        recorder.record_step(make_snapshot(0, 20.0, 0.5));
        recorder.record_step(make_snapshot(1, 21.0, 0.4));
        assert_eq!(recorder.record().snapshots.len(), 2);
        assert_eq!(recorder.record().snapshots[0].step, 0);
        assert_eq!(recorder.record().snapshots[1].step, 1);
    }

    #[test]
    fn test_recorder_snapshot_interval() {
        // interval = 5 → only record steps 0, 5, 10, 15, …
        let mut recorder =
            SessionRecorder::new(SessionConfigSnapshot::default()).with_snapshot_interval(5);

        for step in 0..20 {
            recorder.record_step(make_snapshot(step, 20.0, 0.5));
        }

        // Steps 0, 5, 10, 15 → 4 snapshots
        assert_eq!(recorder.record().snapshots.len(), 4);
        let steps: Vec<usize> = recorder.record().snapshots.iter().map(|s| s.step).collect();
        assert_eq!(steps, vec![0, 5, 10, 15]);
    }

    #[test]
    fn test_recorder_max_snapshots_rolling() {
        let mut recorder =
            SessionRecorder::new(SessionConfigSnapshot::default()).with_max_snapshots(3);

        for step in 0..10 {
            recorder.record_step(make_snapshot(step, 20.0 + step as f32, 0.5));
        }

        let snaps = &recorder.record().snapshots;
        assert_eq!(snaps.len(), 3, "should only retain 3 snapshots");
        // The 3 most recent steps should be 7, 8, 9
        assert_eq!(snaps[0].step, 7);
        assert_eq!(snaps[1].step, 8);
        assert_eq!(snaps[2].step, 9);
    }

    #[test]
    fn test_recorder_best_snapshot_by_psnr() {
        let mut recorder = SessionRecorder::new(SessionConfigSnapshot::default());
        recorder.record_step(make_snapshot(0, 20.0, 0.5));
        recorder.record_step(make_snapshot(1, 35.0, 0.3));
        recorder.record_step(make_snapshot(2, 28.0, 0.2));

        let best = recorder.best_snapshot();
        assert!(best.is_some());
        let fallback_psnr = make_snapshot(0, 0.0, 0.0);
        let best = best.unwrap_or(&fallback_psnr);
        assert_eq!(best.step, 1, "step 1 has the highest PSNR (35.0)");
        assert!((best.psnr - 35.0).abs() < 1e-5);
    }

    #[test]
    fn test_recorder_best_loss_snapshot() {
        let mut recorder = SessionRecorder::new(SessionConfigSnapshot::default());
        recorder.record_step(make_snapshot(0, 20.0, 0.8));
        recorder.record_step(make_snapshot(1, 25.0, 0.1));
        recorder.record_step(make_snapshot(2, 30.0, 0.4));

        let best = recorder.best_loss_snapshot();
        assert!(best.is_some());
        let fallback_loss = make_snapshot(0, 0.0, f32::MAX);
        let best = best.unwrap_or(&fallback_loss);
        assert_eq!(best.step, 1, "step 1 has the lowest loss (0.1)");
        assert!((best.total_loss - 0.1).abs() < 1e-5);
    }

    #[test]
    fn test_recorder_format_summary() {
        let mut recorder = SessionRecorder::new(SessionConfigSnapshot::default());
        recorder.record_step(make_snapshot(0, 22.0, 0.5));
        recorder.record_step(make_snapshot(1, 28.0, 0.3));
        recorder.finish();

        let summary = recorder.format_summary();
        assert!(!summary.is_empty(), "summary should not be empty");
        assert!(
            summary.contains("Session ID"),
            "summary should contain Session ID header"
        );
        assert!(summary.contains("PSNR"), "summary should report best PSNR");
    }

    #[test]
    fn test_recorder_format_summary_multibyte_notes_and_hostname_no_panic() {
        // Regression: `format_summary` used to truncate `notes`/`hostname`
        // by BYTE index (`&s[..33]` / `&s[..34]`), which panics with "byte
        // index N is not a char boundary" whenever the cut point lands
        // inside a multi-byte UTF-8 character. `notes` is a public field
        // and `hostname` comes from an environment variable, so either can
        // contain non-ASCII. A 2-ASCII-character prefix followed by
        // 3-byte-per-character CJK text is constructed so neither the
        // notes cut point (byte 33) nor the hostname cut point (byte 34)
        // lands on a character boundary (boundaries fall at `2 + 3k`,
        // which is congruent to 2 mod 3, while 33 mod 3 = 0 and 34 mod 3 = 1).
        let mut recorder = SessionRecorder::new(SessionConfigSnapshot::default());
        recorder.record_step(make_snapshot(0, 22.0, 0.5));
        recorder.finish();
        let long_multibyte = "AB\u{65e5}\u{672c}\u{8a9e}\u{306e}\u{30c6}\u{30b9}\u{30c8}\u{30ce}\u{30fc}\u{30c8}\u{3067}\u{3059}";
        assert!(
            long_multibyte.len() > 34,
            "fixture must exceed the truncation threshold"
        );
        recorder.record.notes = long_multibyte.to_string();
        recorder.record.hardware.hostname = long_multibyte.to_string();

        // Must not panic.
        let summary = recorder.format_summary();
        assert!(!summary.is_empty());
        assert!(summary.contains("Notes"));
        assert!(summary.contains("Host"));
    }

    #[test]
    fn test_recorder_to_json_valid() {
        let mut recorder =
            SessionRecorder::new(SessionConfigSnapshot::default().with_extra("run", "test"));
        recorder.record_step(make_snapshot(0, 22.0, 0.5));
        recorder.record_step(make_snapshot(1, 28.0, 0.3));
        recorder.finish();

        let json = recorder.to_json();

        // Must be non-empty
        assert!(!json.is_empty());
        // Must start/end with braces
        assert!(json.trim().starts_with('{'));
        assert!(json.trim().ends_with('}'));
        // Key fields must be present
        assert!(
            json.contains("\"session_id\""),
            "JSON must contain session_id"
        );
        assert!(
            json.contains("\"start_timestamp_ms\""),
            "JSON must contain start_timestamp_ms"
        );
        assert!(
            json.contains("\"snapshots\""),
            "JSON must contain snapshots"
        );
        assert!(json.contains("\"hardware\""), "JSON must contain hardware");
        assert!(json.contains("\"config\""), "JSON must contain config");
        assert!(
            json.contains("\"snapshots_count\""),
            "JSON must contain snapshots_count"
        );
        assert!(
            json.contains("\"best_psnr\""),
            "JSON must contain best_psnr"
        );
        assert!(
            json.contains("\"best_loss\""),
            "JSON must contain best_loss"
        );
        assert!(
            json.contains("\"psnr\""),
            "JSON snapshots must include psnr field"
        );
    }

    #[test]
    fn test_recorder_save_and_load_json() {
        let mut recorder = SessionRecorder::new(SessionConfigSnapshot::default());
        recorder.record_step(make_snapshot(0, 25.0, 0.6));
        recorder.record_step(make_snapshot(1, 30.0, 0.4));
        recorder.finish();

        let original_id = recorder.record().session_id.clone();
        let original_start = recorder.record().start_timestamp_ms;

        let mut path = std::env::temp_dir();
        path.push("oxigaf_test_session_recorder.json");

        // Save
        recorder
            .save_json(&path)
            .unwrap_or_else(|e| panic!("save_json failed: {}", e));

        // Load header
        let loaded =
            SessionRecorder::load_json(&path).unwrap_or_else(|e| panic!("load_json failed: {}", e));

        assert_eq!(loaded.session_id, original_id, "session_id must round-trip");
        assert_eq!(
            loaded.start_timestamp_ms, original_start,
            "start_timestamp_ms must round-trip"
        );
        assert!(
            loaded.end_timestamp_ms > 0,
            "end_timestamp_ms should be loaded"
        );

        // Cleanup (best-effort)
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_generate_session_id_length() {
        let id = generate_session_id(12345);
        assert_eq!(
            id.len(),
            16,
            "session ID must be exactly 16 hex characters, got: {}",
            id
        );
    }

    #[test]
    fn test_generate_session_id_deterministic() {
        let id1 = generate_session_id(99999);
        let id2 = generate_session_id(99999);
        assert_eq!(id1, id2, "same seed must produce identical session IDs");
    }

    #[test]
    fn test_generate_session_id_different_seeds_differ() {
        let id1 = generate_session_id(1);
        let id2 = generate_session_id(2);
        assert_ne!(
            id1, id2,
            "different seeds should (almost always) produce different IDs"
        );
    }

    #[test]
    fn test_json_string_escaping() {
        let s = "hello \"world\"\nnew line\ttab";
        let escaped = escape_json_string(s);
        assert!(escaped.contains("\\\""));
        assert!(escaped.contains("\\n"));
        assert!(escaped.contains("\\t"));
        assert!(!escaped.contains('\n'), "raw newlines must be escaped");
    }

    #[test]
    fn test_hardware_info_default() {
        let hw = HardwareInfo::default();
        assert_eq!(hw.cpu_cores, 1);
        assert_eq!(hw.hostname, "unknown");
        assert!(!hw.os_name.is_empty());
    }

    #[test]
    fn test_recorder_no_snapshots_returns_none_for_best() {
        let recorder = SessionRecorder::new(SessionConfigSnapshot::default());
        assert!(recorder.best_snapshot().is_none());
        assert!(recorder.best_loss_snapshot().is_none());
    }

    #[test]
    fn test_session_error_display() {
        let err = SessionError::ParseError("test error".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("test error"));

        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let err2 = SessionError::from(io_err);
        let msg2 = format!("{}", err2);
        assert!(msg2.contains("I/O error"));
    }
}
