//! Weight offloading infrastructure for low-VRAM inference.
//!
//! For GPUs with less than ~6 GB VRAM, model components should be loaded
//! one at a time. This module provides the scheduling logic and memory
//! tracking to orchestrate that process.

use crate::DiffusionError;
use std::fmt::Write as FmtWrite;

// ---------------------------------------------------------------------------
// ComponentType
// ---------------------------------------------------------------------------

/// A model component that can be offloaded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ComponentType {
    ClipImageEncoder,
    VaeEncoder,
    MultiViewUNet,
    LatentUpsampler,
    VaeDecoder,
}

impl ComponentType {
    /// All components in the order they are used during inference.
    pub fn all_in_inference_order() -> &'static [ComponentType] {
        &[
            ComponentType::ClipImageEncoder,
            ComponentType::VaeEncoder,
            ComponentType::MultiViewUNet,
            ComponentType::LatentUpsampler,
            ComponentType::VaeDecoder,
        ]
    }

    /// Estimated FP16 weight size in megabytes.
    pub fn estimated_size_mb(&self) -> f32 {
        match self {
            ComponentType::ClipImageEncoder => 900.0,
            ComponentType::VaeEncoder => 167.0,
            ComponentType::MultiViewUNet => 1700.0,
            ComponentType::LatentUpsampler => 500.0,
            ComponentType::VaeDecoder => 167.0,
        }
    }

    /// Human-readable component name.
    pub fn display_name(&self) -> &'static str {
        match self {
            ComponentType::ClipImageEncoder => "CLIP Image Encoder",
            ComponentType::VaeEncoder => "VAE Encoder",
            ComponentType::MultiViewUNet => "Multi-View U-Net",
            ComponentType::LatentUpsampler => "Latent Upsampler",
            ComponentType::VaeDecoder => "VAE Decoder",
        }
    }
}

// ---------------------------------------------------------------------------
// OffloadStrategy
// ---------------------------------------------------------------------------

/// How to sequence component loading.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OffloadStrategy {
    /// Keep all components in VRAM (sufficient VRAM).
    AllInMemory,
    /// Load each component just before use, offload when done.
    Sequential,
    /// Keep most-used component (UNet) cached, offload others.
    CacheOne,
}

// ---------------------------------------------------------------------------
// InferencePhase
// ---------------------------------------------------------------------------

/// One phase of inference requiring specific components.
#[derive(Debug, Clone)]
pub struct InferencePhase {
    pub name: String,
    pub required_components: Vec<ComponentType>,
    pub estimated_duration_ms: f32,
    pub peak_activation_mb: f32,
}

impl InferencePhase {
    /// Total weight memory required by this phase (sum of component sizes).
    pub fn weight_memory_mb(&self) -> f32 {
        self.required_components
            .iter()
            .map(|c| c.estimated_size_mb())
            .sum()
    }

    /// Peak memory including activations.
    pub fn peak_memory_mb(&self) -> f32 {
        self.weight_memory_mb() + self.peak_activation_mb
    }
}

// ---------------------------------------------------------------------------
// OffloadSchedule
// ---------------------------------------------------------------------------

/// Defines which components to load for each inference phase.
pub struct OffloadSchedule {
    pub strategy: OffloadStrategy,
    pub phases: Vec<InferencePhase>,
}

/// Typical activation overhead estimate (MB) per phase.
const ACTIVATION_OVERHEAD_MB: f32 = 256.0;

impl OffloadSchedule {
    /// Build a schedule for the given strategy.
    pub fn for_strategy(strategy: OffloadStrategy) -> Self {
        let phases = match strategy {
            OffloadStrategy::AllInMemory => {
                vec![InferencePhase {
                    name: "full_inference".to_string(),
                    required_components: ComponentType::all_in_inference_order().to_vec(),
                    estimated_duration_ms: 5000.0,
                    peak_activation_mb: ACTIVATION_OVERHEAD_MB,
                }]
            }

            OffloadStrategy::Sequential => ComponentType::all_in_inference_order()
                .iter()
                .map(|&component| InferencePhase {
                    name: format!(
                        "phase_{}",
                        component.display_name().to_lowercase().replace(' ', "_")
                    ),
                    required_components: vec![component],
                    estimated_duration_ms: 1000.0,
                    peak_activation_mb: ACTIVATION_OVERHEAD_MB,
                })
                .collect(),

            OffloadStrategy::CacheOne => {
                // UNet always resident; one phase per non-UNet component.
                ComponentType::all_in_inference_order()
                    .iter()
                    .filter(|&&c| c != ComponentType::MultiViewUNet)
                    .map(|&component| InferencePhase {
                        name: format!(
                            "cached_unet_with_{}",
                            component.display_name().to_lowercase().replace(' ', "_")
                        ),
                        required_components: vec![ComponentType::MultiViewUNet, component],
                        estimated_duration_ms: 1200.0,
                        peak_activation_mb: ACTIVATION_OVERHEAD_MB,
                    })
                    .collect()
            }
        };

        Self { strategy, phases }
    }

    /// Number of phases in this schedule.
    pub fn total_phases(&self) -> usize {
        self.phases.len()
    }

    /// Maximum peak memory across all phases (weights + activations).
    pub fn peak_memory_mb(&self) -> f32 {
        self.phases
            .iter()
            .map(|p| p.peak_memory_mb())
            .fold(0.0_f32, f32::max)
    }

    /// Format a human-readable schedule table.
    pub fn format_schedule(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(
            out,
            "Offload Schedule ({:?}) — {} phases, peak {:.0} MB",
            self.strategy,
            self.phases.len(),
            self.peak_memory_mb()
        );
        let _ = writeln!(
            out,
            "{:<40} {:>12} {:>14} {:>12}",
            "Phase", "Weight MB", "Activation MB", "Peak MB"
        );
        let _ = writeln!(out, "{}", "-".repeat(80));

        for phase in &self.phases {
            let _ = writeln!(
                out,
                "{:<40} {:>12.0} {:>14.0} {:>12.0}",
                phase.name,
                phase.weight_memory_mb(),
                phase.peak_activation_mb,
                phase.peak_memory_mb()
            );
        }

        out
    }
}

// ---------------------------------------------------------------------------
// MemoryBudget
// ---------------------------------------------------------------------------

/// Tracks available and used GPU memory.
#[derive(Debug, Clone)]
pub struct MemoryBudget {
    pub total_vram_mb: f32,
    pub reserved_mb: f32,
    pub available_mb: f32,
    pub currently_loaded_mb: f32,
}

impl MemoryBudget {
    /// Create a budget given total VRAM and the fraction to reserve for the OS/driver.
    pub fn new(total_vram_mb: f32, reserved_fraction: f32) -> Self {
        let reserved_mb = total_vram_mb * reserved_fraction;
        let available_mb = (total_vram_mb - reserved_mb).max(0.0);
        Self {
            total_vram_mb,
            reserved_mb,
            available_mb,
            currently_loaded_mb: 0.0,
        }
    }

    /// Returns `true` if `component_mb` more memory can be loaded.
    pub fn can_load(&self, component_mb: f32) -> bool {
        self.currently_loaded_mb + component_mb <= self.available_mb
    }

    /// Load `component_mb` into VRAM, returning an error if there is not enough space.
    pub fn load(&mut self, component_mb: f32) -> Result<(), DiffusionError> {
        if !self.can_load(component_mb) {
            return Err(DiffusionError::InvalidConfig(format!(
                "Not enough VRAM: need {:.1} MB, have {:.1} MB free (available={:.1} MB, loaded={:.1} MB)",
                component_mb,
                self.free_mb(),
                self.available_mb,
                self.currently_loaded_mb
            )));
        }
        self.currently_loaded_mb += component_mb;
        Ok(())
    }

    /// Unload `component_mb` from VRAM (clamps to zero).
    pub fn unload(&mut self, component_mb: f32) {
        self.currently_loaded_mb = (self.currently_loaded_mb - component_mb).max(0.0);
    }

    /// Free memory remaining (available − currently_loaded).
    pub fn free_mb(&self) -> f32 {
        (self.available_mb - self.currently_loaded_mb).max(0.0)
    }

    /// Fraction of available memory currently in use.
    pub fn utilization(&self) -> f32 {
        if self.available_mb == 0.0 {
            return 1.0;
        }
        (self.currently_loaded_mb / self.available_mb).clamp(0.0, 1.0)
    }
}

// ---------------------------------------------------------------------------
// recommend_strategy
// ---------------------------------------------------------------------------

/// Recommend an offload strategy given the available memory budget.
///
/// - AllInMemory if the sum of all components fits.
/// - CacheOne if UNet + the largest non-UNet component fits.
/// - Sequential otherwise.
pub fn recommend_strategy(budget: &MemoryBudget) -> OffloadStrategy {
    let all_components = ComponentType::all_in_inference_order();
    let total_weight_mb: f32 = all_components.iter().map(|c| c.estimated_size_mb()).sum();

    if total_weight_mb <= budget.available_mb {
        return OffloadStrategy::AllInMemory;
    }

    let unet_mb = ComponentType::MultiViewUNet.estimated_size_mb();
    let max_other_mb = all_components
        .iter()
        .filter(|&&c| c != ComponentType::MultiViewUNet)
        .map(|c| c.estimated_size_mb())
        .fold(0.0_f32, f32::max);

    if unet_mb + max_other_mb <= budget.available_mb {
        return OffloadStrategy::CacheOne;
    }

    OffloadStrategy::Sequential
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- ComponentType -------------------------------------------------------

    #[test]
    fn test_component_inference_order() {
        let order = ComponentType::all_in_inference_order();
        assert_eq!(order.len(), 5);
        assert_eq!(order[0], ComponentType::ClipImageEncoder);
        assert_eq!(order[1], ComponentType::VaeEncoder);
        assert_eq!(order[2], ComponentType::MultiViewUNet);
        assert_eq!(order[3], ComponentType::LatentUpsampler);
        assert_eq!(order[4], ComponentType::VaeDecoder);
    }

    #[test]
    fn test_component_estimated_sizes() {
        assert!((ComponentType::ClipImageEncoder.estimated_size_mb() - 900.0).abs() < 1.0);
        assert!((ComponentType::VaeEncoder.estimated_size_mb() - 167.0).abs() < 1.0);
        assert!((ComponentType::MultiViewUNet.estimated_size_mb() - 1700.0).abs() < 1.0);
        assert!((ComponentType::LatentUpsampler.estimated_size_mb() - 500.0).abs() < 1.0);
        assert!((ComponentType::VaeDecoder.estimated_size_mb() - 167.0).abs() < 1.0);
    }

    // -- OffloadStrategy / OffloadSchedule -----------------------------------

    #[test]
    fn test_offload_strategy_all_in_memory() {
        let schedule = OffloadSchedule::for_strategy(OffloadStrategy::AllInMemory);
        assert_eq!(schedule.total_phases(), 1);
        assert_eq!(schedule.phases[0].required_components.len(), 5);
        assert!(schedule.peak_memory_mb() > 0.0);
    }

    #[test]
    fn test_offload_strategy_sequential_phase_count() {
        let schedule = OffloadSchedule::for_strategy(OffloadStrategy::Sequential);
        // One phase per component.
        assert_eq!(schedule.total_phases(), 5);
        // Each phase has exactly one component.
        for phase in &schedule.phases {
            assert_eq!(phase.required_components.len(), 1);
        }
    }

    #[test]
    fn test_offload_strategy_cache_one_phase_count() {
        let schedule = OffloadSchedule::for_strategy(OffloadStrategy::CacheOne);
        // One phase per non-UNet component (4 phases).
        assert_eq!(schedule.total_phases(), 4);
        for phase in &schedule.phases {
            // Each phase must include the UNet.
            assert!(phase
                .required_components
                .contains(&ComponentType::MultiViewUNet));
            // And exactly one additional component.
            assert_eq!(phase.required_components.len(), 2);
        }
    }

    #[test]
    fn test_offload_schedule_peak_memory() {
        let sequential = OffloadSchedule::for_strategy(OffloadStrategy::Sequential);
        // The heaviest single component is UNet at 1700 MB + activation overhead.
        let expected_peak = 1700.0 + ACTIVATION_OVERHEAD_MB;
        assert!(
            (sequential.peak_memory_mb() - expected_peak).abs() < 1.0,
            "expected {}, got {}",
            expected_peak,
            sequential.peak_memory_mb()
        );
    }

    // -- MemoryBudget --------------------------------------------------------

    #[test]
    fn test_memory_budget_new() {
        let budget = MemoryBudget::new(8192.0, 0.1);
        assert!((budget.total_vram_mb - 8192.0).abs() < 0.01);
        assert!((budget.reserved_mb - 819.2).abs() < 0.01);
        assert!((budget.available_mb - 7372.8).abs() < 0.1);
        assert_eq!(budget.currently_loaded_mb, 0.0);
    }

    #[test]
    fn test_memory_budget_can_load() {
        let budget = MemoryBudget::new(4096.0, 0.1);
        // available ≈ 3686.4 MB
        assert!(budget.can_load(1000.0));
        assert!(!budget.can_load(4000.0));
    }

    #[test]
    fn test_memory_budget_load_unload() {
        let mut budget = MemoryBudget::new(8192.0, 0.1);
        assert!(budget.load(1000.0).is_ok());
        assert!((budget.currently_loaded_mb - 1000.0).abs() < 0.01);

        budget.unload(500.0);
        assert!((budget.currently_loaded_mb - 500.0).abs() < 0.01);

        budget.unload(1000.0); // clamp to 0
        assert_eq!(budget.currently_loaded_mb, 0.0);
    }

    #[test]
    fn test_memory_budget_load_exceeds_capacity() {
        let mut budget = MemoryBudget::new(2048.0, 0.1);
        // available ≈ 1843.2 MB
        let result = budget.load(2000.0);
        assert!(result.is_err(), "should fail when capacity exceeded");
    }

    #[test]
    fn test_memory_budget_utilization() {
        let mut budget = MemoryBudget::new(1000.0, 0.0);
        assert_eq!(budget.utilization(), 0.0);

        budget.load(500.0).expect("load should succeed");
        assert!((budget.utilization() - 0.5).abs() < 1e-6);

        budget.load(500.0).expect("load should succeed");
        assert!((budget.utilization() - 1.0).abs() < 1e-6);
    }

    // -- recommend_strategy --------------------------------------------------

    #[test]
    fn test_recommend_strategy_all_in_memory() {
        // Total weight: 900 + 167 + 1700 + 500 + 167 = 3434 MB
        // Give it 4000 MB available.
        let budget = MemoryBudget::new(4000.0, 0.0);
        assert_eq!(recommend_strategy(&budget), OffloadStrategy::AllInMemory);
    }

    #[test]
    fn test_recommend_strategy_cache_one() {
        // UNet (1700) + largest other (CLIP 900) = 2600 MB.
        // Give budget of exactly 2600 MB available.
        let budget = MemoryBudget::new(2600.0, 0.0);
        assert_eq!(recommend_strategy(&budget), OffloadStrategy::CacheOne);
    }

    #[test]
    fn test_recommend_strategy_sequential() {
        // Only 1000 MB available — can't hold even UNet (1700 MB) + anything.
        let budget = MemoryBudget::new(1000.0, 0.0);
        assert_eq!(recommend_strategy(&budget), OffloadStrategy::Sequential);
    }

    #[test]
    fn test_format_schedule_contains_phases() {
        let schedule = OffloadSchedule::for_strategy(OffloadStrategy::Sequential);
        let formatted = schedule.format_schedule();
        assert!(formatted.contains("Sequential"), "got: {}", formatted);
        assert!(formatted.contains("5 phases"), "got: {}", formatted);
    }
}
