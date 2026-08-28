//! Index-addressable GPU adapter probing for `oxigaf doctor`.
//!
//! [`crate::commands::runtime::check_gpu`] asks wgpu for *an* adapter with
//! `PowerPreference::HighPerformance` and reports whatever it gets. On a
//! multi-GPU machine that is the wrong question: `oxigaf train --device 1`
//! runs on adapter 1, so a diagnostic that only ever inspects the
//! high-performance default can report a healthy GPU while the one the run
//! will actually use is missing, is a software fallback, or has too little
//! memory. `doctor` could not be used to check the device a job would run
//! on, which is most of what a pre-flight diagnostic is for.
//!
//! [`probe_adapter`] takes the same `--device <index>` the training pipeline
//! resolves (see `pipeline::resolve_device_index`) and inspects *that*
//! adapter, enumerating the available ones so a bad index is answered with
//! the list rather than a bare failure.

use anyhow::{Context, Result};

/// What a probe found out about one adapter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdapterReport {
    /// The index that was probed.
    pub index: usize,
    /// Adapter name as reported by the driver.
    pub name: String,
    /// Graphics backend (Metal, Vulkan, DX12, GL).
    pub backend: String,
    /// Adapter class (discrete, integrated, virtual, CPU, other).
    pub device_type: String,
    /// Driver name and version, when the backend reports them.
    pub driver: String,
}

impl AdapterReport {
    /// One-line human rendering, matching what `doctor` used to print for
    /// the default adapter plus the index it belongs to.
    #[must_use]
    pub fn summary(&self) -> String {
        let driver = if self.driver.trim().is_empty() {
            String::new()
        } else {
            format!(", driver {}", self.driver)
        };
        format!(
            "[{}] {} ({}, {}{})",
            self.index, self.name, self.backend, self.device_type, driver
        )
    }

    /// Whether this adapter is a software rasteriser rather than real hardware.
    ///
    /// wgpu will happily hand back a CPU adapter, on which a 3DGS training
    /// run is orders of magnitude too slow to be useful. `doctor` reporting
    /// that as "GPU adapter found" is the kind of true-but-useless answer a
    /// diagnostic exists to avoid.
    #[must_use]
    pub fn is_software_fallback(&self) -> bool {
        self.device_type == "Cpu"
    }
}

/// Choose the adapter at `device_index` out of `available`, or explain why
/// that is not possible.
///
/// Split out from the wgpu call so the index arithmetic — the part with an
/// off-by-one to get wrong — is testable without a GPU.
///
/// # Errors
///
/// Returns an error when no adapter is available at all, or when
/// `device_index` is past the end of the list.
pub fn select_index(available: usize, device_index: usize) -> Result<usize> {
    if available == 0 {
        anyhow::bail!(
            "No GPU adapters are available. Install or update your graphics drivers; \
             on a headless machine, check that a software or virtual adapter is present."
        );
    }
    if device_index >= available {
        anyhow::bail!(
            "--device {device_index} is out of range: {available} adapter(s) are available \
             (valid indices are 0..={})",
            available - 1
        );
    }
    Ok(device_index)
}

/// Whether `device_index` selects wgpu's preferred adapter rather than an
/// enumerated one.
///
/// This mirrors `pipeline::request_gpu_device`, which special-cases index 0:
/// it asks for `PowerPreference::HighPerformance` instead of taking
/// `enumerate_adapters()[0]`. On a machine where the two disagree — a laptop
/// whose integrated GPU enumerates first but whose discrete GPU is preferred
/// is the common case — a probe that always enumerated would describe a
/// *different* adapter from the one the default `train` run uses, which is
/// the same class of mismatch this module exists to remove, only with the
/// sign flipped. The rule is shared so `doctor --device N` and `train
/// --device N` always mean the same adapter.
#[must_use]
pub fn uses_power_preference(device_index: usize) -> bool {
    device_index == 0
}

/// Describe one adapter, tagged with the index that selected it.
fn report_for(adapter: &wgpu::Adapter, index: usize) -> AdapterReport {
    let info = adapter.get_info();
    AdapterReport {
        index,
        name: info.name,
        backend: format!("{:?}", info.backend),
        device_type: format!("{:?}", info.device_type),
        driver: if info.driver_info.is_empty() {
            info.driver
        } else {
            format!("{} {}", info.driver, info.driver_info)
        },
    }
}

/// Inspect the adapter `oxigaf --device <index>` would run on.
///
/// Selection follows `pipeline::request_gpu_device` exactly — see
/// [`uses_power_preference`] for why index 0 is not simply
/// `enumerate_adapters()[0]`.
///
/// # Errors
///
/// Returns an error when no adapter exists, or when `device_index` names one
/// that does not — with the enumerated adapters attached as context so the
/// caller can see what the valid indices are.
pub fn probe_adapter(device_index: usize) -> Result<AdapterReport> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });

    if uses_power_preference(device_index) {
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }))
        .map_err(|e| {
            anyhow::anyhow!(
                "No GPU adapter found: {e}. Install or update your graphics drivers; \
                 on a headless machine, check that a software or virtual adapter is present."
            )
        })?;
        return Ok(report_for(&adapter, device_index));
    }

    let adapters = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()));
    let listing: Vec<String> = adapters
        .iter()
        .enumerate()
        .map(|(i, adapter)| {
            let info = adapter.get_info();
            format!("[{i}] {} ({:?})", info.name, info.backend)
        })
        .collect();

    let index = select_index(adapters.len(), device_index)
        .with_context(|| format!("Enumerated GPU adapters: {}", listing.join(", ")))?;

    let adapter = adapters
        .get(index)
        .ok_or_else(|| anyhow::anyhow!("GPU adapter {index} vanished during selection"))?;

    Ok(report_for(adapter, index))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A machine with no adapter at all gets a message about drivers, not an
    /// out-of-range complaint about index 0.
    #[test]
    fn select_index_reports_an_empty_adapter_list_distinctly() {
        let err = select_index(0, 0).expect_err("no adapters must fail");
        let message = format!("{err}");
        assert!(
            message.contains("No GPU adapters"),
            "message was: {message}"
        );
        assert!(
            !message.contains("out of range"),
            "an empty list must not be reported as a bad index: {message}"
        );
    }

    /// The last valid index is `available - 1`; the boundary is where an
    /// off-by-one would hide.
    #[test]
    fn select_index_accepts_every_valid_index_and_rejects_the_next() {
        for available in 1..=4usize {
            for index in 0..available {
                assert_eq!(
                    select_index(available, index).ok(),
                    Some(index),
                    "--device {index} of {available} should be valid"
                );
            }
            let err =
                select_index(available, available).expect_err("one past the end must be rejected");
            let message = format!("{err}");
            assert!(
                message.contains(&format!("0..={}", available - 1)),
                "the message must name the valid range, was: {message}"
            );
        }
    }

    /// `doctor --device N` and `train --device N` must name the same
    /// adapter. `pipeline::request_gpu_device` special-cases index 0 to
    /// `PowerPreference::HighPerformance` rather than
    /// `enumerate_adapters()[0]`, and on a laptop whose integrated GPU
    /// enumerates first those are different adapters — so a probe that
    /// always enumerated would describe the wrong one for the default run.
    #[test]
    fn index_zero_follows_the_pipeline_power_preference_rule() {
        assert!(
            uses_power_preference(0),
            "index 0 must use the same power-preference request the pipeline does"
        );
        for index in 1..=4usize {
            assert!(
                !uses_power_preference(index),
                "--device {index} must be resolved by enumeration, as the pipeline does"
            );
        }
    }

    /// A CPU adapter is real to wgpu and useless to a 3DGS trainer, so the
    /// diagnostic has to distinguish it.
    #[test]
    fn software_fallback_is_recognised() {
        let cpu = AdapterReport {
            index: 0,
            name: "llvmpipe".to_string(),
            backend: "Vulkan".to_string(),
            device_type: "Cpu".to_string(),
            driver: String::new(),
        };
        assert!(cpu.is_software_fallback());

        let gpu = AdapterReport {
            device_type: "DiscreteGpu".to_string(),
            ..cpu.clone()
        };
        assert!(!gpu.is_software_fallback());
    }

    /// The summary names the index, because "which GPU is this?" is the
    /// question `--device` exists to answer.
    #[test]
    fn summary_names_the_index_and_omits_an_empty_driver() {
        let report = AdapterReport {
            index: 1,
            name: "Apple M3".to_string(),
            backend: "Metal".to_string(),
            device_type: "IntegratedGpu".to_string(),
            driver: String::new(),
        };
        let text = report.summary();
        assert!(text.starts_with("[1] Apple M3"), "summary was: {text}");
        assert!(!text.contains("driver"), "summary was: {text}");

        let with_driver = AdapterReport {
            driver: "Metal 3.2".to_string(),
            ..report
        };
        assert!(
            with_driver.summary().contains("driver Metal 3.2"),
            "summary was: {}",
            with_driver.summary()
        );
    }
}
